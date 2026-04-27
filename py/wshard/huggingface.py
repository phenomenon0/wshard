"""
HuggingFace Hub adapter for W-SHARD.

Provides functionality to:
- Upload W-SHARD datasets to HuggingFace Hub
- Download W-SHARD datasets from HuggingFace Hub
- Convert between HuggingFace datasets and W-SHARD episodes

Requires: pip install huggingface_hub datasets
"""

import io
import os
import json
import tempfile
from pathlib import Path
from typing import (
    Union,
    Optional,
    List,
    Dict,
    Any,
    Iterator,
    Callable,
    Tuple,
)
import numpy as np

from .types import Episode, Channel, DType, Format, TimebaseSpec, TimebaseType
from .wshard import load_wshard, save_wshard
from .compress import CompressionType

# Optional imports - gracefully handle missing dependencies
try:
    from huggingface_hub import (
        HfApi,
        hf_hub_download,
        snapshot_download,
        upload_file,
        upload_folder,
        create_repo,
        CommitOperationAdd,
        CommitOperationDelete,
    )

    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

try:
    import datasets
    from datasets import (
        Dataset,
        DatasetDict,
        Features,
        Sequence,
        Value,
        Array2D,
        Array3D,
    )

    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False


class HuggingFaceAdapter:
    """
    Adapter for HuggingFace Hub integration.

    Provides methods to upload/download W-SHARD datasets and convert
    between HuggingFace datasets format and W-SHARD episodes.

    Example usage:

        # Upload episodes to Hub
        adapter = HuggingFaceAdapter(token="hf_xxx")
        adapter.upload_episodes(
            episodes=my_episodes,
            repo_id="username/my-rl-dataset",
            env_id="CartPole-v1",
        )

        # Download episodes from Hub
        episodes = adapter.download_episodes(
            repo_id="username/my-rl-dataset",
            split="train",
        )

        # Convert HuggingFace dataset to episodes
        episodes = adapter.from_hf_dataset(hf_dataset)
    """

    def __init__(
        self,
        token: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the HuggingFace adapter.

        Args:
            token: HuggingFace API token. If None, uses HF_TOKEN env var or cached token.
            cache_dir: Directory for caching downloaded files.
        """
        if not HF_HUB_AVAILABLE:
            raise ImportError(
                "huggingface_hub is required. Install with: pip install huggingface_hub"
            )

        self.token = token or os.environ.get("HF_TOKEN")
        self.cache_dir = cache_dir
        self.api = HfApi(token=self.token)

    # =========================================================================
    # Upload Methods
    # =========================================================================

    def upload_episodes(
        self,
        episodes: List[Episode],
        repo_id: str,
        env_id: Optional[str] = None,
        split: str = "train",
        compression: CompressionType = CompressionType.ZSTD,
        private: bool = False,
        commit_message: Optional[str] = None,
        batch_size: int = 100,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> str:
        """
        Upload a list of episodes to HuggingFace Hub as W-SHARD files.

        Episodes are stored as individual .wshard files in a structured directory:
            {split}/episodes/{episode_id}.wshard

        A metadata JSON file is also created with dataset info.

        Args:
            episodes: List of Episode objects to upload
            repo_id: HuggingFace repo ID (e.g., "username/dataset-name")
            env_id: Environment ID (e.g., "CartPole-v1"). If None, uses first episode's env_id.
            split: Dataset split name (e.g., "train", "test", "validation")
            compression: Compression mode ("none", "zstd", "lz4")
            private: Whether to create a private repository
            commit_message: Custom commit message
            batch_size: Number of episodes per commit batch
            progress_callback: Optional callback(current, total) for progress reporting

        Returns:
            Repository URL
        """
        if not episodes:
            raise ValueError("No episodes provided")

        # Create repository if it doesn't exist
        try:
            self.api.create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=private,
                exist_ok=True,
            )
        except Exception as e:
            # Repo might already exist
            pass

        # Determine env_id
        if env_id is None:
            env_id = episodes[0].env_id or "unknown"

        # Create metadata
        metadata = {
            "format": "wshard",
            "format_version": "0.1",
            "env_id": env_id,
            "split": split,
            "num_episodes": len(episodes),
            "compression": compression.value,
            "total_timesteps": sum(ep.length for ep in episodes),
            "observation_keys": list(episodes[0].observations.keys())
            if episodes
            else [],
            "action_keys": list(episodes[0].actions.keys()) if episodes else [],
            "has_rewards": episodes[0].rewards is not None if episodes else False,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create directory structure
            episodes_dir = tmpdir / split / "episodes"
            episodes_dir.mkdir(parents=True, exist_ok=True)

            # Write episodes
            for i, ep in enumerate(episodes):
                ep_path = episodes_dir / f"{ep.id}.wshard"
                save_wshard(ep, ep_path, compression=compression)

                if progress_callback:
                    progress_callback(i + 1, len(episodes))

            # Write metadata
            metadata_path = tmpdir / split / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            # Write README
            readme_path = tmpdir / "README.md"
            readme_content = self._generate_readme(metadata, episodes)
            with open(readme_path, "w") as f:
                f.write(readme_content)

            # Upload folder
            self.api.upload_folder(
                folder_path=str(tmpdir),
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=commit_message
                or f"Upload {len(episodes)} episodes ({split})",
            )

        return f"https://huggingface.co/datasets/{repo_id}"

    def upload_episode(
        self,
        episode: Episode,
        repo_id: str,
        split: str = "train",
        compression: CompressionType = CompressionType.ZSTD,
        commit_message: Optional[str] = None,
    ) -> str:
        """
        Upload a single episode to HuggingFace Hub.

        Args:
            episode: Episode to upload
            repo_id: HuggingFace repo ID
            split: Dataset split name
            compression: Compression mode
            commit_message: Custom commit message

        Returns:
            URL of the uploaded file
        """
        with tempfile.NamedTemporaryFile(suffix=".wshard", delete=False) as f:
            save_wshard(episode, f.name, compression=compression)

            path_in_repo = f"{split}/episodes/{episode.id}.wshard"

            self.api.upload_file(
                path_or_fileobj=f.name,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=commit_message or f"Upload episode {episode.id}",
            )

            os.unlink(f.name)

        return f"https://huggingface.co/datasets/{repo_id}/blob/main/{path_in_repo}"

    # =========================================================================
    # Download Methods
    # =========================================================================

    def download_episodes(
        self,
        repo_id: str,
        split: str = "train",
        episode_ids: Optional[List[str]] = None,
        max_episodes: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Episode]:
        """
        Download episodes from HuggingFace Hub.

        Args:
            repo_id: HuggingFace repo ID
            split: Dataset split name
            episode_ids: Optional list of specific episode IDs to download
            max_episodes: Maximum number of episodes to download
            progress_callback: Optional callback(current, total) for progress reporting

        Returns:
            List of Episode objects
        """
        # Download the dataset snapshot
        local_dir = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            cache_dir=self.cache_dir,
            token=self.token,
        )

        episodes_dir = Path(local_dir) / split / "episodes"

        if not episodes_dir.exists():
            raise ValueError(f"No episodes found for split '{split}' in {repo_id}")

        # Find episode files
        episode_files = sorted(episodes_dir.glob("*.wshard"))

        # Filter by episode IDs if specified
        if episode_ids:
            episode_files = [f for f in episode_files if f.stem in episode_ids]

        # Limit number of episodes
        if max_episodes:
            episode_files = episode_files[:max_episodes]

        # Load episodes
        episodes = []
        for i, ep_file in enumerate(episode_files):
            ep = load_wshard(ep_file)
            episodes.append(ep)

            if progress_callback:
                progress_callback(i + 1, len(episode_files))

        return episodes

    def download_episode(
        self,
        repo_id: str,
        episode_id: str,
        split: str = "train",
    ) -> Episode:
        """
        Download a single episode from HuggingFace Hub.

        Args:
            repo_id: HuggingFace repo ID
            episode_id: Episode ID
            split: Dataset split name

        Returns:
            Episode object
        """
        path_in_repo = f"{split}/episodes/{episode_id}.wshard"

        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=path_in_repo,
            repo_type="dataset",
            cache_dir=self.cache_dir,
            token=self.token,
        )

        return load_wshard(local_path)

    def list_episodes(
        self,
        repo_id: str,
        split: str = "train",
    ) -> List[str]:
        """
        List available episode IDs in a repository.

        Args:
            repo_id: HuggingFace repo ID
            split: Dataset split name

        Returns:
            List of episode IDs
        """
        files = self.api.list_repo_files(
            repo_id=repo_id,
            repo_type="dataset",
        )

        prefix = f"{split}/episodes/"
        suffix = ".wshard"

        episode_ids = [
            f[len(prefix) : -len(suffix)]
            for f in files
            if f.startswith(prefix) and f.endswith(suffix)
        ]

        return sorted(episode_ids)

    def get_metadata(self, repo_id: str, split: str = "train") -> Dict[str, Any]:
        """
        Get dataset metadata from HuggingFace Hub.

        Args:
            repo_id: HuggingFace repo ID
            split: Dataset split name

        Returns:
            Metadata dictionary
        """
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{split}/metadata.json",
            repo_type="dataset",
            cache_dir=self.cache_dir,
            token=self.token,
        )

        with open(local_path) as f:
            return json.load(f)

    # =========================================================================
    # HuggingFace Datasets Conversion
    # =========================================================================

    def to_hf_dataset(
        self,
        episodes: List[Episode],
        flatten: bool = True,
    ) -> "Dataset":
        """
        Convert episodes to a HuggingFace Dataset.

        Args:
            episodes: List of Episode objects
            flatten: If True, flatten timesteps into rows. If False, each episode is a row.

        Returns:
            HuggingFace Dataset
        """
        if not DATASETS_AVAILABLE:
            raise ImportError(
                "datasets is required. Install with: pip install datasets"
            )

        if flatten:
            return self._to_hf_dataset_flat(episodes)
        else:
            return self._to_hf_dataset_episodic(episodes)

    def from_hf_dataset(
        self,
        dataset: "Dataset",
        episode_column: str = "episode_id",
        flatten: bool = True,
    ) -> List[Episode]:
        """
        Convert a HuggingFace Dataset to episodes.

        Args:
            dataset: HuggingFace Dataset
            episode_column: Column name containing episode IDs
            flatten: If True, dataset has flattened timesteps. If False, each row is an episode.

        Returns:
            List of Episode objects
        """
        if not DATASETS_AVAILABLE:
            raise ImportError(
                "datasets is required. Install with: pip install datasets"
            )

        if flatten:
            return self._from_hf_dataset_flat(dataset, episode_column)
        else:
            return self._from_hf_dataset_episodic(dataset)

    def _to_hf_dataset_flat(self, episodes: List[Episode]) -> "Dataset":
        """Convert episodes to flat HuggingFace Dataset (one row per timestep)."""
        rows = []

        for ep in episodes:
            for t in range(ep.length):
                row = {
                    "episode_id": ep.id,
                    "timestep": t,
                }

                # Add observations
                for name, ch in ep.observations.items():
                    key = f"obs_{name.replace('/', '_')}"
                    row[key] = (
                        ch.data[t].tolist()
                        if ch.data[t].ndim > 0
                        else ch.data[t].item()
                    )

                # Add actions
                for name, ch in ep.actions.items():
                    key = f"action_{name.replace('/', '_')}"
                    row[key] = (
                        ch.data[t].tolist()
                        if ch.data[t].ndim > 0
                        else ch.data[t].item()
                    )

                # Add reward
                if ep.rewards is not None:
                    row["reward"] = float(ep.rewards.data[t])

                # Add termination
                if ep.terminations is not None:
                    row["terminated"] = bool(ep.terminations.data[t])

                # Add truncation
                if ep.truncations is not None:
                    row["truncated"] = bool(ep.truncations.data[t])

                rows.append(row)

        return Dataset.from_list(rows)

    def _to_hf_dataset_episodic(self, episodes: List[Episode]) -> "Dataset":
        """Convert episodes to episodic HuggingFace Dataset (one row per episode)."""
        rows = []

        for ep in episodes:
            row = {
                "episode_id": ep.id,
                "env_id": ep.env_id,
                "length": ep.length,
            }

            # Add observations (as nested arrays)
            for name, ch in ep.observations.items():
                key = f"obs_{name.replace('/', '_')}"
                row[key] = ch.data.tolist()

            # Add actions
            for name, ch in ep.actions.items():
                key = f"action_{name.replace('/', '_')}"
                row[key] = ch.data.tolist()

            # Add reward
            if ep.rewards is not None:
                row["rewards"] = ep.rewards.data.tolist()

            # Add terminations
            if ep.terminations is not None:
                row["terminations"] = ep.terminations.data.tolist()

            # Add truncations
            if ep.truncations is not None:
                row["truncations"] = ep.truncations.data.tolist()

            rows.append(row)

        return Dataset.from_list(rows)

    def _from_hf_dataset_flat(
        self, dataset: "Dataset", episode_column: str
    ) -> List[Episode]:
        """Convert flat HuggingFace Dataset to episodes."""
        # Group rows by episode ID
        episode_rows: Dict[str, List[Dict]] = {}

        for row in dataset:
            ep_id = row[episode_column]
            if ep_id not in episode_rows:
                episode_rows[ep_id] = []
            episode_rows[ep_id].append(row)

        # Sort rows by timestep within each episode
        for ep_id in episode_rows:
            episode_rows[ep_id].sort(key=lambda r: r.get("timestep", 0))

        # Convert to episodes
        episodes = []

        for ep_id, rows in episode_rows.items():
            length = len(rows)
            ep = Episode(id=ep_id, length=length)

            # Identify columns
            obs_cols = [c for c in rows[0].keys() if c.startswith("obs_")]
            action_cols = [c for c in rows[0].keys() if c.startswith("action_")]

            # Build observations
            for col in obs_cols:
                name = col[4:].replace("_", "/")  # Remove "obs_" prefix
                data = np.array([r[col] for r in rows])
                shape = list(data.shape[1:]) if data.ndim > 1 else []
                ep.observations[name] = Channel(
                    name=name,
                    dtype=DType.from_numpy(data.dtype),
                    shape=shape,
                    data=data,
                )

            # Build actions
            for col in action_cols:
                name = col[7:].replace("_", "/")  # Remove "action_" prefix
                data = np.array([r[col] for r in rows])
                shape = list(data.shape[1:]) if data.ndim > 1 else []
                ep.actions[name] = Channel(
                    name=name,
                    dtype=DType.from_numpy(data.dtype),
                    shape=shape,
                    data=data,
                )

            # Build rewards
            if "reward" in rows[0]:
                data = np.array([r["reward"] for r in rows], dtype=np.float32)
                ep.rewards = Channel(
                    name="reward", dtype=DType.FLOAT32, shape=[], data=data
                )

            # Build terminations
            if "terminated" in rows[0]:
                data = np.array([r["terminated"] for r in rows], dtype=np.bool_)
                ep.terminations = Channel(
                    name="terminated", dtype=DType.BOOL, shape=[], data=data
                )

            # Build truncations
            if "truncated" in rows[0]:
                data = np.array([r["truncated"] for r in rows], dtype=np.bool_)
                ep.truncations = Channel(
                    name="truncated", dtype=DType.BOOL, shape=[], data=data
                )

            episodes.append(ep)

        return episodes

    def _from_hf_dataset_episodic(self, dataset: "Dataset") -> List[Episode]:
        """Convert episodic HuggingFace Dataset to episodes."""
        episodes = []

        for row in dataset:
            ep_id = row.get("episode_id", f"episode_{len(episodes)}")
            env_id = row.get("env_id", "")
            length = row.get("length", 0)

            # Determine length from data if not provided
            if length == 0:
                for key in row:
                    if key.startswith("obs_") or key.startswith("action_"):
                        val = row[key]
                        if isinstance(val, (list, np.ndarray)):
                            length = len(val)
                            break

            ep = Episode(id=ep_id, env_id=env_id, length=length)

            # Build observations
            for key in row:
                if key.startswith("obs_"):
                    name = key[4:].replace("_", "/")
                    data = np.array(row[key])
                    shape = list(data.shape[1:]) if data.ndim > 1 else []
                    ep.observations[name] = Channel(
                        name=name,
                        dtype=DType.from_numpy(data.dtype),
                        shape=shape,
                        data=data,
                    )

            # Build actions
            for key in row:
                if key.startswith("action_"):
                    name = key[7:].replace("_", "/")
                    data = np.array(row[key])
                    shape = list(data.shape[1:]) if data.ndim > 1 else []
                    ep.actions[name] = Channel(
                        name=name,
                        dtype=DType.from_numpy(data.dtype),
                        shape=shape,
                        data=data,
                    )

            # Build rewards
            if "rewards" in row:
                data = np.array(row["rewards"], dtype=np.float32)
                ep.rewards = Channel(
                    name="reward", dtype=DType.FLOAT32, shape=[], data=data
                )

            # Build terminations
            if "terminations" in row:
                data = np.array(row["terminations"], dtype=np.bool_)
                ep.terminations = Channel(
                    name="terminated", dtype=DType.BOOL, shape=[], data=data
                )

            # Build truncations
            if "truncations" in row:
                data = np.array(row["truncations"], dtype=np.bool_)
                ep.truncations = Channel(
                    name="truncated", dtype=DType.BOOL, shape=[], data=data
                )

            episodes.append(ep)

        return episodes

    # =========================================================================
    # Utilities
    # =========================================================================

    def _generate_readme(
        self, metadata: Dict[str, Any], episodes: List[Episode]
    ) -> str:
        """Generate a README.md for the dataset."""
        obs_shapes = {}
        action_shapes = {}

        if episodes:
            for name, ch in episodes[0].observations.items():
                obs_shapes[name] = ch.shape
            for name, ch in episodes[0].actions.items():
                action_shapes[name] = ch.shape

        readme = f"""---
license: mit
task_categories:
  - reinforcement-learning
tags:
  - wshard
  - world-model
  - rl-episodes
---

# {metadata.get("env_id", "RL Dataset")}

This dataset contains RL episodes in W-SHARD format.

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Environment | `{metadata.get("env_id", "unknown")}` |
| Number of Episodes | {metadata.get("num_episodes", 0)} |
| Total Timesteps | {metadata.get("total_timesteps", 0)} |
| Format | W-SHARD v0.1 |
| Compression | {metadata.get("compression", "zstd")} |

## Observation Space

| Channel | Shape |
|---------|-------|
"""
        for name, shape in obs_shapes.items():
            readme += f"| `{name}` | {shape} |\n"

        readme += """
## Action Space

| Channel | Shape |
|---------|-------|
"""
        for name, shape in action_shapes.items():
            readme += f"| `{name}` | {shape} |\n"

        readme += f"""
## Usage

### With W-SHARD Python

```python
from wshard import HuggingFaceAdapter

adapter = HuggingFaceAdapter()
episodes = adapter.download_episodes("{metadata.get("repo_id", "username/dataset")}")

for ep in episodes:
    print(f"Episode {{ep.id}}: {{ep.length}} timesteps")
```

### Convert to NumPy

```python
from wshard import load_wshard

ep = load_wshard("path/to/episode.wshard")
observations = ep.observations["state"].data  # NumPy array
actions = ep.actions["action"].data
rewards = ep.rewards.data
```

## Format Details

W-SHARD (World-Model Episode Shard) is a compact, versioned RL episode format optimized for:
- Fast random access (50x faster than NPZ)
- Cross-framework compatibility (DreamerV3, TD-MPC2, Minari)
- Signal/Omen/Residual lanes for world model training

Learn more: [W-SHARD Specification](https://github.com/Neumenon/shard/blob/main/wshard/docs/DEEP_DIVE.md)
"""

        return readme


# Convenience functions for direct use without instantiating adapter


def upload_to_hub(
    episodes: List[Episode],
    repo_id: str,
    token: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Upload episodes to HuggingFace Hub.

    Convenience function that creates an adapter and uploads episodes.

    Args:
        episodes: List of Episode objects
        repo_id: HuggingFace repo ID
        token: HuggingFace API token
        **kwargs: Additional arguments passed to upload_episodes()

    Returns:
        Repository URL
    """
    adapter = HuggingFaceAdapter(token=token)
    return adapter.upload_episodes(episodes, repo_id, **kwargs)


def download_from_hub(
    repo_id: str,
    token: Optional[str] = None,
    **kwargs,
) -> List[Episode]:
    """
    Download episodes from HuggingFace Hub.

    Convenience function that creates an adapter and downloads episodes.

    Args:
        repo_id: HuggingFace repo ID
        token: HuggingFace API token
        **kwargs: Additional arguments passed to download_episodes()

    Returns:
        List of Episode objects
    """
    adapter = HuggingFaceAdapter(token=token)
    return adapter.download_episodes(repo_id, **kwargs)
