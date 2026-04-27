"""
DeepData trajectory bridge for W-SHARD episodes.

Bridges wshard files (on disk, authoritative store) with DeepData (vector search).
DeepData indexes metadata + embeddings only; wshard files remain the source of truth.

Architecture:
    Disk (wshard files)              DeepData
    ────────────────────             ──────────────────────────────
    episode_abc.wshard               traj_episodes collection:
      signal/obs [T,D]                 vector: pooled obs embedding
      action/ctrl [T,A]                meta: {episode_id, env_id,
      reward [T]                              length_T, total_reward,
      meta/*                                  file_path, chunk_index}
"""

import json
import os
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np

from .wshard import load_wshard
from .types import Episode

# Collection names
COLLECTION_EPISODES = "traj_episodes"
COLLECTION_TIMESTEPS = "traj_timesteps"


@dataclass
class EpisodeRef:
    """Reference to a retrieved episode with similarity score."""

    episode_id: str
    file_path: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


def _http_json(url: str, data: Optional[dict] = None, method: str = "GET") -> dict:
    """
    Send an HTTP request and parse the JSON response.

    Args:
        url: Full URL to request.
        data: JSON body (triggers POST unless method is overridden).
        method: HTTP method.

    Returns:
        Parsed JSON response as a dict.

    Raises:
        urllib.error.HTTPError: On non-2xx responses.
        json.JSONDecodeError: On malformed response body.
    """
    headers = {"Content-Type": "application/json"}
    body = None
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        if method == "GET":
            method = "POST"
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(req) as resp:
        resp_body = resp.read()
        if not resp_body:
            return {}
        return json.loads(resp_body)


class TrajectoryIngestor:
    """
    Ingests wshard episode files into DeepData for vector similarity search.

    Wshard files stay on disk as the authoritative store.  DeepData only
    holds pooled observation embeddings and lightweight metadata so that
    similar episodes (or sub-trajectories) can be found quickly.
    """

    def __init__(
        self,
        deepdata_url: str,
        embedder: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        """
        Args:
            deepdata_url: Base URL of the DeepData HTTP API (e.g. "http://localhost:8080").
            embedder: Function that maps an observation array [T, D] to a
                      single embedding vector [E].  Typically mean-pools across
                      time, then runs through a learned encoder.
        """
        self._url = deepdata_url.rstrip("/")
        self._embedder = embedder
        self._collections_ensured: set = set()

    def ensure_collection(self, dim: int, collection: str = COLLECTION_EPISODES) -> None:
        """
        Create a DeepData collection with HNSW index if it doesn't exist.

        Args:
            dim: Embedding dimensionality.
            collection: Collection name to create.
        """
        if collection in self._collections_ensured:
            return
        url = f"{self._url}/v2/collections"
        payload = {
            "name": collection,
            "dimension": dim,
            "index_type": "hnsw",
        }
        try:
            _http_json(url, data=payload)
        except urllib.error.HTTPError as e:
            # 409 Conflict means collection already exists -- that's fine.
            if e.code == 409:
                pass
            else:
                raise
        self._collections_ensured.add(collection)

    def ingest_episode(self, path: str) -> str:
        """
        Load a wshard episode, compute a pooled embedding, and insert into DeepData.

        The embedding is produced by mean-pooling the first observation channel
        across time, then passing the result through the configured embedder.

        Args:
            path: Filesystem path to a .wshard file.

        Returns:
            The episode_id that was inserted.

        Raises:
            FileNotFoundError: If the wshard file does not exist.
            ValueError: If the episode has no observation channels.
        """
        ep = load_wshard(path)
        obs = self._get_primary_obs(ep)
        if obs is None:
            raise ValueError(f"Episode {ep.id} has no observation channels")

        # Mean-pool observations across time → [D]
        pooled = obs.mean(axis=0).astype(np.float32)
        embedding = self._embedder(pooled)
        dim = int(embedding.shape[0])
        self.ensure_collection(dim, COLLECTION_EPISODES)

        total_reward = 0.0
        if ep.rewards is not None and ep.rewards.data is not None:
            total_reward = float(np.sum(ep.rewards.data))

        metadata = {
            "episode_id": ep.id,
            "env_id": ep.env_id,
            "length_T": ep.length,
            "total_reward": total_reward,
            "file_path": os.path.abspath(path),
        }
        if ep.chunk_index is not None:
            metadata["chunk_index"] = ep.chunk_index

        url = f"{self._url}/v2/collections/{COLLECTION_EPISODES}/insert"
        payload = {
            "vectors": [embedding.tolist()],
            "ids": [ep.id],
            "metadata": [metadata],
        }
        _http_json(url, data=payload)
        return ep.id

    def ingest_timesteps(
        self,
        path: str,
        window_size: int = 32,
        stride: int = 16,
    ) -> List[str]:
        """
        Insert per-window embeddings for behavioral cloning retrieval.

        Slides a window over the observation time axis, embeds each window
        independently, and batch-inserts into the ``traj_timesteps`` collection.

        Args:
            path: Filesystem path to a .wshard file.
            window_size: Number of timesteps per window.
            stride: Step size between consecutive windows.

        Returns:
            List of inserted vector IDs (one per window).

        Raises:
            FileNotFoundError: If the wshard file does not exist.
            ValueError: If the episode has no observation channels.
        """
        ep = load_wshard(path)
        obs = self._get_primary_obs(ep)
        if obs is None:
            raise ValueError(f"Episode {ep.id} has no observation channels")

        T = obs.shape[0]
        vectors = []
        ids = []
        metas = []

        abs_path = os.path.abspath(path)

        for start in range(0, max(T - window_size + 1, 1), stride):
            end = min(start + window_size, T)
            window = obs[start:end]
            pooled = window.mean(axis=0).astype(np.float32)
            embedding = self._embedder(pooled)
            vec_id = f"{ep.id}_w{start}"
            vectors.append(embedding.tolist())
            ids.append(vec_id)
            metas.append({
                "episode_id": ep.id,
                "env_id": ep.env_id,
                "file_path": abs_path,
                "window_start": start,
                "window_end": end,
                "length_T": ep.length,
            })

        if not vectors:
            return []

        dim = len(vectors[0])
        self.ensure_collection(dim, COLLECTION_TIMESTEPS)

        url = f"{self._url}/v2/collections/{COLLECTION_TIMESTEPS}/insert"
        payload = {
            "vectors": vectors,
            "ids": ids,
            "metadata": metas,
        }
        _http_json(url, data=payload)
        return ids

    @staticmethod
    def _get_primary_obs(ep: Episode) -> Optional[np.ndarray]:
        """Return the data array of the first observation channel, or None."""
        if not ep.observations:
            return None
        first_key = next(iter(ep.observations))
        ch = ep.observations[first_key]
        if ch.data is None or ch.data.size == 0:
            return None
        # Flatten per-timestep dims so embedder gets [T, D]
        data = ch.data
        if data.ndim > 2:
            data = data.reshape(data.shape[0], -1)
        return data


class TrajectoryRetriever:
    """
    Queries DeepData to find similar episodes or sub-trajectories.

    Search results carry enough metadata to locate the original wshard file
    on disk so the caller can load the full episode data.
    """

    def __init__(
        self,
        deepdata_url: str,
        embedder: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        """
        Args:
            deepdata_url: Base URL of the DeepData HTTP API.
            embedder: Same embedder used at ingest time so that query
                      observations land in the same vector space.
        """
        self._url = deepdata_url.rstrip("/")
        self._embedder = embedder

    def search_similar_episodes(
        self,
        query_obs: np.ndarray,
        top_k: int = 10,
        env_id: Optional[str] = None,
        min_length: Optional[int] = None,
        reward_range: Optional[Tuple[float, float]] = None,
    ) -> List[EpisodeRef]:
        """
        Find episodes whose pooled observation embedding is nearest to a query.

        Args:
            query_obs: Observation array to embed and search with.
                       If 2-D [T, D], mean-pooled across time first.
                       If 1-D [D], used directly.
            top_k: Maximum number of results.
            env_id: If set, restrict results to this environment.
            min_length: If set, exclude episodes shorter than this.
            reward_range: If set, (min_reward, max_reward) inclusive filter.

        Returns:
            List of EpisodeRef sorted by descending similarity.
        """
        # Prepare embedding
        obs = np.asarray(query_obs, dtype=np.float32)
        if obs.ndim > 1:
            obs = obs.mean(axis=0)
        embedding = self._embedder(obs)

        # Build metadata filter
        filters = self._build_filters(env_id, min_length, reward_range)

        url = f"{self._url}/v2/collections/{COLLECTION_EPISODES}/search"
        payload: Dict[str, Any] = {
            "vector": embedding.tolist(),
            "top_k": top_k,
        }
        if filters:
            payload["filter"] = filters

        resp = _http_json(url, data=payload)

        results = []
        for hit in resp.get("results", []):
            meta = hit.get("metadata", {})
            results.append(EpisodeRef(
                episode_id=meta.get("episode_id", hit.get("id", "")),
                file_path=meta.get("file_path", ""),
                score=float(hit.get("score", 0.0)),
                metadata=meta,
            ))
        return results

    def search_similar_timesteps(
        self,
        query_obs: np.ndarray,
        top_k: int = 10,
        env_id: Optional[str] = None,
    ) -> List[EpisodeRef]:
        """
        Find sub-trajectory windows similar to a query observation.

        Args:
            query_obs: Observation array (1-D or 2-D, mean-pooled if needed).
            top_k: Maximum number of results.
            env_id: If set, restrict results to this environment.

        Returns:
            List of EpisodeRef with window metadata.
        """
        obs = np.asarray(query_obs, dtype=np.float32)
        if obs.ndim > 1:
            obs = obs.mean(axis=0)
        embedding = self._embedder(obs)

        filters = self._build_filters(env_id, None, None)

        url = f"{self._url}/v2/collections/{COLLECTION_TIMESTEPS}/search"
        payload: Dict[str, Any] = {
            "vector": embedding.tolist(),
            "top_k": top_k,
        }
        if filters:
            payload["filter"] = filters

        resp = _http_json(url, data=payload)

        results = []
        for hit in resp.get("results", []):
            meta = hit.get("metadata", {})
            results.append(EpisodeRef(
                episode_id=meta.get("episode_id", hit.get("id", "")),
                file_path=meta.get("file_path", ""),
                score=float(hit.get("score", 0.0)),
                metadata=meta,
            ))
        return results

    @staticmethod
    def _build_filters(
        env_id: Optional[str],
        min_length: Optional[int],
        reward_range: Optional[Tuple[float, float]],
    ) -> Optional[dict]:
        """Build a DeepData metadata filter using $and/$eq/$gte/$lte operators."""
        conditions = []
        if env_id is not None:
            conditions.append({"field": "env_id", "$eq": env_id})
        if min_length is not None:
            conditions.append({"field": "length_T", "$gte": min_length})
        if reward_range is not None:
            lo, hi = reward_range
            conditions.append({"field": "total_reward", "$gte": lo})
            conditions.append({"field": "total_reward", "$lte": hi})
        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}
