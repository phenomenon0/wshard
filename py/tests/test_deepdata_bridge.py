"""
Tests for DeepData trajectory bridge.

Uses mock embedder (random vectors) and mocked HTTP responses so that
tests run without a live DeepData instance.
"""

import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from wshard import Episode, Channel, DType, save_wshard
from wshard.deepdata_bridge import (
    TrajectoryIngestor,
    TrajectoryRetriever,
    EpisodeRef,
    COLLECTION_EPISODES,
    COLLECTION_TIMESTEPS,
    _http_json,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EMBED_DIM = 16


def mock_embedder(obs: np.ndarray) -> np.ndarray:
    """Deterministic mock embedder: hashes input to a fixed-dim vector."""
    rng = np.random.RandomState(int(np.abs(obs).sum() * 1000) % (2**31))
    return rng.randn(EMBED_DIM).astype(np.float32)


def _make_episode(
    episode_id: str = "ep_001",
    env_id: str = "CartPole-v1",
    length: int = 50,
    obs_dim: int = 4,
    action_dim: int = 2,
) -> Episode:
    """Create a minimal episode for testing."""
    ep = Episode(id=episode_id, env_id=env_id, length=length)
    ep.observations["obs"] = Channel(
        name="obs",
        dtype=DType.FLOAT32,
        shape=[obs_dim],
        data=np.random.randn(length, obs_dim).astype(np.float32),
    )
    ep.actions["ctrl"] = Channel(
        name="ctrl",
        dtype=DType.FLOAT32,
        shape=[action_dim],
        data=np.random.randn(length, action_dim).astype(np.float32),
    )
    ep.rewards = Channel(
        name="reward",
        dtype=DType.FLOAT32,
        shape=[],
        data=np.random.randn(length).astype(np.float32),
    )
    return ep


def _save_tmp_episode(ep: Episode, tmp_dir: str) -> str:
    """Save episode to a temp wshard file, return path."""
    path = str(Path(tmp_dir) / f"{ep.id}.wshard")
    save_wshard(ep, path)
    return path


def _mock_urlopen_factory(response_body: dict, status: int = 200):
    """Create a mock urlopen context manager returning the given JSON body."""
    def _mock_urlopen(req):
        resp = MagicMock()
        resp.read.return_value = json.dumps(response_body).encode("utf-8")
        resp.status = status
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        return resp
    return _mock_urlopen


# ---------------------------------------------------------------------------
# Tests: EpisodeRef
# ---------------------------------------------------------------------------

class TestEpisodeRef:
    def test_dataclass_fields(self):
        ref = EpisodeRef(
            episode_id="ep1",
            file_path="/tmp/ep1.wshard",
            score=0.95,
            metadata={"env_id": "HalfCheetah-v4"},
        )
        assert ref.episode_id == "ep1"
        assert ref.score == 0.95
        assert ref.metadata["env_id"] == "HalfCheetah-v4"

    def test_default_metadata(self):
        ref = EpisodeRef(episode_id="x", file_path="/x", score=0.0)
        assert ref.metadata == {}


# ---------------------------------------------------------------------------
# Tests: TrajectoryIngestor
# ---------------------------------------------------------------------------

class TestTrajectoryIngestor:

    def test_ensure_collection_posts_create(self):
        """ensure_collection sends a POST to /v2/collections."""
        with patch("wshard.deepdata_bridge.urllib.request.urlopen",
                    _mock_urlopen_factory({})):
            ingestor = TrajectoryIngestor("http://localhost:8080", mock_embedder)
            ingestor.ensure_collection(EMBED_DIM)

        assert COLLECTION_EPISODES in ingestor._collections_ensured

    def test_ensure_collection_idempotent(self):
        """Second call skips the HTTP request."""
        call_count = 0
        original_factory = _mock_urlopen_factory({})

        def counting_urlopen(req):
            nonlocal call_count
            call_count += 1
            return original_factory(req)

        with patch("wshard.deepdata_bridge.urllib.request.urlopen", counting_urlopen):
            ingestor = TrajectoryIngestor("http://localhost:8080", mock_embedder)
            ingestor.ensure_collection(EMBED_DIM)
            ingestor.ensure_collection(EMBED_DIM)

        assert call_count == 1

    def test_ensure_collection_handles_409(self):
        """409 Conflict (already exists) is silently ignored."""
        import urllib.error

        def raise_409(req):
            raise urllib.error.HTTPError(
                req.full_url, 409, "Conflict", {}, io.BytesIO(b"{}"),
            )

        with patch("wshard.deepdata_bridge.urllib.request.urlopen", raise_409):
            ingestor = TrajectoryIngestor("http://localhost:8080", mock_embedder)
            # Should not raise
            ingestor.ensure_collection(EMBED_DIM)

    def test_ingest_episode(self):
        """ingest_episode loads wshard, embeds, and POSTs to DeepData."""
        ep = _make_episode()
        captured_payloads = []

        def capturing_urlopen(req):
            body = req.data
            if body:
                captured_payloads.append(json.loads(body))
            resp = MagicMock()
            resp.read.return_value = b"{}"
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        with tempfile.TemporaryDirectory() as tmp:
            path = _save_tmp_episode(ep, tmp)
            with patch("wshard.deepdata_bridge.urllib.request.urlopen",
                        capturing_urlopen):
                ingestor = TrajectoryIngestor("http://localhost:8080", mock_embedder)
                result = ingestor.ingest_episode(path)

        assert result == "ep_001"
        # Two calls: ensure_collection + insert
        assert len(captured_payloads) == 2

        insert_payload = captured_payloads[1]
        assert insert_payload["ids"] == ["ep_001"]
        assert len(insert_payload["vectors"]) == 1
        assert len(insert_payload["vectors"][0]) == EMBED_DIM

        meta = insert_payload["metadata"][0]
        assert meta["episode_id"] == "ep_001"
        assert meta["env_id"] == "CartPole-v1"
        assert meta["length_T"] == 50
        assert "total_reward" in meta
        assert "file_path" in meta

    def test_ingest_episode_no_obs_raises(self):
        """Episode with no observations raises ValueError."""
        ep = Episode(id="empty", length=10)
        # Manually create a wshard with no obs — just use an episode with
        # an action channel so it passes validate()
        ep.actions["ctrl"] = Channel(
            name="ctrl", dtype=DType.FLOAT32, shape=[2],
            data=np.zeros((10, 2), dtype=np.float32),
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = _save_tmp_episode(ep, tmp)
            with patch("wshard.deepdata_bridge.urllib.request.urlopen",
                        _mock_urlopen_factory({})):
                ingestor = TrajectoryIngestor("http://localhost:8080", mock_embedder)
                with pytest.raises(ValueError, match="no observation"):
                    ingestor.ingest_episode(path)

    def test_ingest_timesteps(self):
        """ingest_timesteps creates windowed embeddings."""
        ep = _make_episode(length=64)
        captured_payloads = []

        def capturing_urlopen(req):
            body = req.data
            if body:
                captured_payloads.append(json.loads(body))
            resp = MagicMock()
            resp.read.return_value = b"{}"
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        with tempfile.TemporaryDirectory() as tmp:
            path = _save_tmp_episode(ep, tmp)
            with patch("wshard.deepdata_bridge.urllib.request.urlopen",
                        capturing_urlopen):
                ingestor = TrajectoryIngestor("http://localhost:8080", mock_embedder)
                ids = ingestor.ingest_timesteps(path, window_size=32, stride=16)

        # T=64, window=32, stride=16 → windows at [0, 16, 32]
        assert len(ids) == 3
        assert ids[0] == "ep_001_w0"
        assert ids[1] == "ep_001_w16"
        assert ids[2] == "ep_001_w32"

        # Last payload is the batch insert
        insert_payload = captured_payloads[-1]
        assert len(insert_payload["vectors"]) == 3
        assert insert_payload["metadata"][0]["window_start"] == 0
        assert insert_payload["metadata"][2]["window_start"] == 32

    def test_ingest_timesteps_short_episode(self):
        """Episode shorter than window_size still produces one window."""
        ep = _make_episode(length=10)

        def noop_urlopen(req):
            resp = MagicMock()
            resp.read.return_value = b"{}"
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        with tempfile.TemporaryDirectory() as tmp:
            path = _save_tmp_episode(ep, tmp)
            with patch("wshard.deepdata_bridge.urllib.request.urlopen",
                        noop_urlopen):
                ingestor = TrajectoryIngestor("http://localhost:8080", mock_embedder)
                ids = ingestor.ingest_timesteps(path, window_size=32, stride=16)

        assert len(ids) == 1
        assert ids[0] == "ep_001_w0"

    def test_chunk_index_in_metadata(self):
        """chunk_index is included in metadata when present on the episode."""
        ep = _make_episode()
        ep.chunk_index = 2
        ep.total_chunks = 5

        captured_payloads = []

        def capturing_urlopen(req):
            body = req.data
            if body:
                captured_payloads.append(json.loads(body))
            resp = MagicMock()
            resp.read.return_value = b"{}"
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        with tempfile.TemporaryDirectory() as tmp:
            path = _save_tmp_episode(ep, tmp)
            with patch("wshard.deepdata_bridge.urllib.request.urlopen",
                        capturing_urlopen):
                ingestor = TrajectoryIngestor("http://localhost:8080", mock_embedder)
                ingestor.ingest_episode(path)

        insert_payload = captured_payloads[-1]
        assert insert_payload["metadata"][0]["chunk_index"] == 2


# ---------------------------------------------------------------------------
# Tests: TrajectoryRetriever
# ---------------------------------------------------------------------------

class TestTrajectoryRetriever:

    def _search_response(self, n: int = 3) -> dict:
        """Build a mock DeepData search response."""
        results = []
        for i in range(n):
            results.append({
                "id": f"ep_{i:03d}",
                "score": 1.0 - i * 0.1,
                "metadata": {
                    "episode_id": f"ep_{i:03d}",
                    "env_id": "CartPole-v1",
                    "file_path": f"/data/ep_{i:03d}.wshard",
                    "length_T": 50 + i * 10,
                    "total_reward": 100.0 - i * 5,
                },
            })
        return {"results": results}

    def test_search_similar_episodes(self):
        """Basic search returns EpisodeRef objects."""
        mock_resp = self._search_response(3)

        with patch("wshard.deepdata_bridge.urllib.request.urlopen",
                    _mock_urlopen_factory(mock_resp)):
            retriever = TrajectoryRetriever("http://localhost:8080", mock_embedder)
            query = np.random.randn(4).astype(np.float32)
            results = retriever.search_similar_episodes(query, top_k=3)

        assert len(results) == 3
        assert isinstance(results[0], EpisodeRef)
        assert results[0].episode_id == "ep_000"
        assert results[0].score == 1.0
        assert results[0].file_path == "/data/ep_000.wshard"

    def test_search_with_2d_query(self):
        """2-D query is mean-pooled before embedding."""
        mock_resp = self._search_response(1)

        with patch("wshard.deepdata_bridge.urllib.request.urlopen",
                    _mock_urlopen_factory(mock_resp)):
            retriever = TrajectoryRetriever("http://localhost:8080", mock_embedder)
            query = np.random.randn(20, 4).astype(np.float32)
            results = retriever.search_similar_episodes(query, top_k=1)

        assert len(results) == 1

    def test_search_with_env_filter(self):
        """env_id filter is sent in the request body."""
        captured = []

        def capturing_urlopen(req):
            if req.data:
                captured.append(json.loads(req.data))
            resp = MagicMock()
            resp.read.return_value = json.dumps({"results": []}).encode()
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        with patch("wshard.deepdata_bridge.urllib.request.urlopen",
                    capturing_urlopen):
            retriever = TrajectoryRetriever("http://localhost:8080", mock_embedder)
            retriever.search_similar_episodes(
                np.zeros(4, dtype=np.float32),
                env_id="Humanoid-v4",
            )

        payload = captured[0]
        assert payload["filter"] == {"field": "env_id", "$eq": "Humanoid-v4"}

    def test_search_with_combined_filters(self):
        """Multiple filters are combined with $and."""
        captured = []

        def capturing_urlopen(req):
            if req.data:
                captured.append(json.loads(req.data))
            resp = MagicMock()
            resp.read.return_value = json.dumps({"results": []}).encode()
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        with patch("wshard.deepdata_bridge.urllib.request.urlopen",
                    capturing_urlopen):
            retriever = TrajectoryRetriever("http://localhost:8080", mock_embedder)
            retriever.search_similar_episodes(
                np.zeros(4, dtype=np.float32),
                env_id="Walker-v4",
                min_length=100,
                reward_range=(50.0, 200.0),
            )

        filt = captured[0]["filter"]
        assert "$and" in filt
        conditions = filt["$and"]
        assert len(conditions) == 4  # env_id + min_length + reward lo + reward hi
        fields = [c["field"] for c in conditions]
        assert "env_id" in fields
        assert "length_T" in fields
        assert fields.count("total_reward") == 2

    def test_search_no_filters(self):
        """No filters means no 'filter' key in the payload."""
        captured = []

        def capturing_urlopen(req):
            if req.data:
                captured.append(json.loads(req.data))
            resp = MagicMock()
            resp.read.return_value = json.dumps({"results": []}).encode()
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        with patch("wshard.deepdata_bridge.urllib.request.urlopen",
                    capturing_urlopen):
            retriever = TrajectoryRetriever("http://localhost:8080", mock_embedder)
            retriever.search_similar_episodes(
                np.zeros(4, dtype=np.float32),
            )

        payload = captured[0]
        assert "filter" not in payload

    def test_search_similar_timesteps(self):
        """Timestep search hits the correct collection endpoint."""
        captured_urls = []

        def capturing_urlopen(req):
            captured_urls.append(req.full_url)
            resp = MagicMock()
            resp.read.return_value = json.dumps({
                "results": [{
                    "id": "ep_001_w0",
                    "score": 0.99,
                    "metadata": {
                        "episode_id": "ep_001",
                        "file_path": "/data/ep_001.wshard",
                        "window_start": 0,
                        "window_end": 32,
                    },
                }]
            }).encode()
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        with patch("wshard.deepdata_bridge.urllib.request.urlopen",
                    capturing_urlopen):
            retriever = TrajectoryRetriever("http://localhost:8080", mock_embedder)
            results = retriever.search_similar_timesteps(
                np.zeros(4, dtype=np.float32), top_k=5,
            )

        assert len(results) == 1
        assert results[0].metadata["window_start"] == 0
        assert COLLECTION_TIMESTEPS in captured_urls[0]

    def test_empty_results(self):
        """Empty result set returns empty list, not error."""
        with patch("wshard.deepdata_bridge.urllib.request.urlopen",
                    _mock_urlopen_factory({"results": []})):
            retriever = TrajectoryRetriever("http://localhost:8080", mock_embedder)
            results = retriever.search_similar_episodes(
                np.zeros(4, dtype=np.float32),
            )

        assert results == []


# ---------------------------------------------------------------------------
# Tests: _http_json helper
# ---------------------------------------------------------------------------

class TestHttpJson:

    def test_get_request(self):
        with patch("wshard.deepdata_bridge.urllib.request.urlopen",
                    _mock_urlopen_factory({"ok": True})):
            result = _http_json("http://localhost:8080/health")
        assert result == {"ok": True}

    def test_post_with_data(self):
        captured_methods = []

        def capturing_urlopen(req):
            captured_methods.append(req.get_method())
            resp = MagicMock()
            resp.read.return_value = b'{"inserted": 1}'
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        with patch("wshard.deepdata_bridge.urllib.request.urlopen",
                    capturing_urlopen):
            result = _http_json("http://localhost:8080/insert", data={"x": 1})

        assert captured_methods[0] == "POST"
        assert result == {"inserted": 1}

    def test_empty_response_body(self):
        def empty_urlopen(req):
            resp = MagicMock()
            resp.read.return_value = b""
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        with patch("wshard.deepdata_bridge.urllib.request.urlopen",
                    empty_urlopen):
            result = _http_json("http://localhost:8080/empty")

        assert result == {}
