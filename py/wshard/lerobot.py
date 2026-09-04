"""LeRobot <-> W-SHARD: one sealed ``.wshard`` file per episode.

Two halves that share one naming rule:

* :func:`convert_dataset` writes a ``LeRobotDataset`` out as one file per
  episode (roadmap item 6, the import direction).
* :class:`WShardDatasetReader` reads those files back *through LeRobot itself*.
  ``lerobot.datasets.storage`` exposes a backend registry, so this plugs in
  behind the stock ``lerobot-train`` rather than beside it::

      from wshard.lerobot import register
      register()
      # then any lerobot entry point, unchanged, against a root whose
      # meta/info.json carries "storage_format": "wshard".

What is deliberately *not* stored: ``episode_index``, ``frame_index``,
``index``, ``timestamp``. Every one is derivable from ``meta/`` plus the frame's
position, and LeRobot hands the reader its ``LeRobotDatasetMetadata``. Storing
them would be writing down what the index already knows.

Frames are stored as raw ``uint8`` HWC tensors, not video. LeRobot's decoder
returns ``frames.type(torch.float32) / 255`` (``video_utils.py:227``), so the
uint8 round-trip through this reader is exact rather than approximate.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .compress import CompressionType
from .types import Channel, DType, Episode, Provenance
from .wshard import load_wshard, save_wshard

# ponytail: whole-episode LRU. Fine while an episode is a few MB; switch to
# per-frame block reads if episodes grow past what N copies of RAM allows.
_EPISODE_CACHE_SIZE = 256

_NP_TO_DTYPE = {
    np.dtype("float32"): DType.FLOAT32,
    np.dtype("float64"): DType.FLOAT64,
    np.dtype("int64"): DType.INT64,
    np.dtype("int32"): DType.INT32,
    np.dtype("uint8"): DType.UINT8,
    np.dtype("bool"): DType.BOOL,
}


# --------------------------------------------------------------------------
# The naming rule. Both directions import it from here, so they cannot drift.
# --------------------------------------------------------------------------

def block_name(lerobot_key: str) -> str:
    """``observation.image`` -> ``signal/image``; ``action`` -> ``action/ctrl``.

    ``reward`` and ``done`` are bare block names in W-SHARD, not namespaced.
    """
    if lerobot_key == "action":
        return "action/ctrl"
    if lerobot_key in ("next.reward", "reward"):
        return "reward"
    if lerobot_key in ("next.done", "done"):
        return "done"
    if lerobot_key.startswith("observation."):
        return "signal/" + lerobot_key[len("observation."):]
    # next.success and anything else the dataset carries: keep it, under signal/.
    return "signal/" + lerobot_key.replace("next.", "")


def channel_id(lerobot_key: str) -> str:
    """The bare id inside the lane -- what ``load_wshard(channels=[...])`` filters on."""
    name = block_name(lerobot_key)
    for lane in ("signal/", "action/"):
        if name.startswith(lane):
            return name[len(lane):]
    return name


# --------------------------------------------------------------------------
# LeRobot -> W-SHARD
# --------------------------------------------------------------------------

def _data_keys(meta) -> List[str]:
    """Feature keys worth storing: everything except what meta/ already knows."""
    derivable = {"episode_index", "frame_index", "index", "timestamp"}
    return [k for k in meta.features if k not in derivable]


def convert_dataset(
    repo_id: str,
    out_root: str | Path,
    episodes: Optional[List[int]] = None,
    compression: CompressionType = CompressionType.ZSTD,
    root: Optional[str | Path] = None,
) -> List[Path]:
    """Write ``repo_id`` as a ``.wshard`` dataset root, one file per episode.

    Produces ``out_root/meta/`` (LeRobot's own, copied verbatim except for
    ``storage_format``) and ``out_root/wshard/episode_%06d.wshard``. Copying
    ``meta/`` rather than regenerating it is the point: a training run against
    this root differs from one against the source *only* in where frame bytes
    come from.

    Returns the paths written, in episode order. Frames are pulled with
    ``return_uint8=True``, so no float conversion happens on either side.

    A partial conversion (``episodes=[...]``) leaves ``meta/`` describing the
    whole dataset, so the resulting root is only valid when read back with the
    same filter.
    """
    import json as _json  # noqa: PLC0415
    import shutil  # noqa: PLC0415

    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

    out_root = Path(out_root)
    out_dir = out_root / "wshard"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = LeRobotDataset(repo_id, root=root, episodes=episodes, return_uint8=True)
    meta = ds.meta
    keys = _data_keys(meta)
    ep_indices = episodes if episodes is not None else list(range(meta.total_episodes))
    # ds is indexed *relatively* into the (possibly filtered) view, never by the
    # absolute frame index. The two coincide only when nothing is filtered out,
    # which is why this is computed rather than special-cased.
    offsets, _acc = {}, 0
    for e in ep_indices:
        offsets[e] = _acc
        r = meta.episodes[e]
        _acc += r["dataset_to_index"] - r["dataset_from_index"]

    if (out_root / "meta").exists():
        shutil.rmtree(out_root / "meta")
    shutil.copytree(Path(ds.root) / "meta", out_root / "meta")
    info_path = out_root / "meta" / "info.json"
    info = _json.loads(info_path.read_text())
    info["storage_format"] = "wshard"
    info_path.write_text(_json.dumps(info, indent=4))

    written: List[Path] = []
    prev_identity = ""
    for ep_idx in ep_indices:
        row = meta.episodes[ep_idx]
        start, stop = row["dataset_from_index"], row["dataset_to_index"]
        length = stop - start

        frames = [ds[offsets[ep_idx] + i] for i in range(length)]

        ep = Episode(id=f"{repo_id.replace('/', '_')}_ep{ep_idx:06d}", length=length)
        ep.env_id = repo_id
        for key in keys:
            stacked = np.stack([_as_numpy(f[key], key, meta) for f in frames])
            name = block_name(key)
            if name == "done":
                # the reader always parses the bare ``done`` block as uint8
                stacked = stacked.astype(np.uint8)
            ch = Channel(
                name=name.rsplit("/", 1)[-1],
                dtype=_NP_TO_DTYPE[stacked.dtype],
                shape=list(stacked.shape[1:]),
                data=stacked,
            )
            if name == "reward":
                ep.rewards = ch
            elif name == "done":
                ep.terminations = ch
            elif name.startswith("action/"):
                ep.actions[ch.name] = ch
            else:
                ep.observations[ch.name] = ch

        ep.provenance = Provenance(
            run_id=f"lerobot_to_wshard:{repo_id}",
            prev_identity=prev_identity,
            source={
                "repo_id": repo_id,
                "episode_index": str(ep_idx),
                "fps": str(meta.fps),
                "dataset_from_index": str(start),
                "keys": json.dumps(keys, sort_keys=True),
            },
        )

        path = out_dir / f"episode_{ep_idx:06d}.wshard"
        save_wshard(ep, path, compression=compression)
        from .wshard import episode_identity  # noqa: PLC0415
        prev_identity = episode_identity(path)
        written.append(path)

    return written


def _as_numpy(value, key: str, meta) -> np.ndarray:
    """One frame's value as the numpy array W-SHARD stores.

    Camera frames arrive CHW from the decoder; store HWC, which is the layout
    the feature shape in ``meta`` declares.
    """
    arr = value.numpy() if hasattr(value, "numpy") else np.asarray(value)
    if key in meta.camera_keys and arr.ndim == 3:
        arr = np.transpose(arr, (1, 2, 0))
    return arr


# --------------------------------------------------------------------------
# W-SHARD -> LeRobot (the storage backend)
# --------------------------------------------------------------------------

_TORCH_DTYPE: Dict[str, str] = {
    "float32": "float32", "float64": "float64", "int64": "int64",
    "int32": "int32", "bool": "bool", "uint8": "uint8",
}


def _channel_for(ep: Episode, lerobot_key: str):
    """The channel in ``ep`` holding ``lerobot_key``, or None if absent."""
    name = block_name(lerobot_key)
    if name == "reward":
        return ep.rewards
    if name == "done":
        return ep.terminations
    cid = channel_id(lerobot_key)
    return ep.actions.get(cid) if name.startswith("action/") else ep.observations.get(cid)


@lru_cache(maxsize=_EPISODE_CACHE_SIZE)
def _load_episode_arrays(path: str, keys: tuple) -> Dict[str, np.ndarray]:
    """Decode one ``.wshard`` into ``{lerobot_key: writable array [T, ...]}``.

    Writable because the decoder hands back ``np.frombuffer`` views, and
    ``torch.from_numpy`` on a read-only array warns on every single frame.
    One copy per episode beats one warning per sample.
    """
    ep = load_wshard(path)
    out = {}
    for key in keys:
        ch = _channel_for(ep, key)
        if ch is not None:
            out[key] = np.array(ch.data)
    return out


class WShardDatasetReader:
    """Serve a LeRobot dataset out of one sealed ``.wshard`` file per episode.

    Layout under ``root``: LeRobot's own ``meta/`` (with ``storage_format``
    set to ``"wshard"`` in ``info.json``) beside ``wshard/episode_%06d.wshard``.
    Keeping ``meta/`` verbatim is deliberate -- it means a run against this
    backend differs from a run against the stock one *only* in where the frame
    bytes come from.
    """

    def __init__(
        self,
        meta,
        root,
        episodes: Optional[List[int]],
        tolerance_s: float,
        delta_timestamps: Optional[Dict[str, List[float]]],
        image_transforms: Optional[Callable],
        return_uint8: bool = False,
        depth_output_unit: str = "mm",
        revision: Optional[str] = None,
        token: Any = None,
        video_backend: Optional[str] = None,
    ):
        from lerobot.datasets.feature_utils import (  # noqa: PLC0415
            check_delta_timestamps,
            get_delta_indices,
        )
        from lerobot.datasets.utils import resolve_episode_indices  # noqa: PLC0415

        self._meta = meta
        self.root = Path(root)
        self.episodes = resolve_episode_indices(episodes, meta.total_episodes)
        self._image_transforms = image_transforms
        self._return_uint8 = return_uint8
        self._depth_output_unit = depth_output_unit
        self._keys = tuple(_data_keys(meta))
        self._camera_keys = set(meta.camera_keys)
        self._dtypes = {k: _TORCH_DTYPE[meta.features[k]["dtype"]]
                        for k in self._keys if k not in self._camera_keys}

        self.delta_indices = None
        if delta_timestamps is not None:
            check_delta_timestamps(delta_timestamps, meta.fps, tolerance_s)
            self.delta_indices = get_delta_indices(delta_timestamps, meta.fps)

        selected = self.episodes if self.episodes is not None else range(meta.total_episodes)
        rows = [meta.episodes[e] for e in selected]
        self._ep_ids = list(selected)
        self._ep_start = np.array([r["dataset_from_index"] for r in rows], dtype=np.int64)
        lengths = np.array([r["dataset_to_index"] - r["dataset_from_index"] for r in rows],
                           dtype=np.int64)
        self._lengths = lengths
        self._cum = np.concatenate([[0], np.cumsum(lengths)])

    # -- BaseDatasetReader surface -----------------------------------------

    @property
    def num_frames(self) -> int:
        return int(self._cum[-1])

    @property
    def num_episodes(self) -> int:
        return len(self._ep_ids)

    @property
    def absolute_to_relative_idx(self) -> Optional[Dict[int, int]]:
        if self.episodes is None:
            return None
        return {int(self._ep_start[s]) + off: int(self._cum[s]) + off
                for s in range(len(self._ep_ids)) for off in range(int(self._lengths[s]))}

    def get_items(self, indices: List[int]) -> List[dict]:
        return [self.get_item(i) for i in indices]

    def __len__(self) -> int:
        return self.num_frames

    def set_image_transforms(self, image_transforms: Optional[Callable]) -> None:
        if image_transforms is not None and not callable(image_transforms):
            raise TypeError("image_transforms must be callable or None.")
        self._image_transforms = image_transforms

    def clear_image_transforms(self) -> None:
        self._image_transforms = None

    # -- the actual read ---------------------------------------------------

    def episode_path(self, ep_idx: int) -> Path:
        return self.root / "wshard" / f"episode_{ep_idx:06d}.wshard"

    def get_item(self, idx: int) -> dict:
        import torch  # noqa: PLC0415

        slot = int(np.searchsorted(self._cum, idx, side="right") - 1)
        off = idx - int(self._cum[slot])
        ep_idx = self._ep_ids[slot]
        length = int(self._lengths[slot])
        arrays = _load_episode_arrays(str(self.episode_path(ep_idx)), self._keys)

        item: Dict[str, Any] = {}
        for key in self._keys:
            deltas = None if self.delta_indices is None else self.delta_indices.get(key)
            if deltas is None:
                sel = off
            else:
                sel = [min(max(off + d, 0), length - 1) for d in deltas]
                item[f"{key}_is_pad"] = torch.BoolTensor(
                    [(off + d < 0) or (off + d >= length) for d in deltas]
                )
            raw = arrays[key][sel]
            if key in self._camera_keys:
                # stored HWC uint8; LeRobot hands policies CHW, float32 in [0,1]
                t = torch.from_numpy(raw)
                t = t.permute(2, 0, 1) if t.ndim == 3 else t.permute(0, 3, 1, 2)
                item[key] = t if self._return_uint8 else t.type(torch.float32) / 255
            else:
                item[key] = torch.as_tensor(raw).to(getattr(torch, self._dtypes[key]))

        abs_idx = int(self._ep_start[slot]) + off
        task_idx = int(arrays["task_index"][off])
        item["episode_index"] = torch.tensor(ep_idx, dtype=torch.int64)
        item["frame_index"] = torch.tensor(off, dtype=torch.int64)
        item["index"] = torch.tensor(abs_idx, dtype=torch.int64)
        item["timestamp"] = torch.tensor(off / self._meta.fps, dtype=torch.float32)
        item["task"] = self._meta.tasks.iloc[task_idx].name

        if self._image_transforms is not None:
            for cam in self._camera_keys:
                if cam in item and cam not in self._meta.depth_keys:
                    item[cam] = self._image_transforms(item[cam])
        return item


DATASET_READER = WShardDatasetReader


def localize_root(repo_id, root, revision=None, token=None, force_cache_sync=False) -> Path:
    """Object-store roots are not supported; ``.wshard`` datasets are read locally."""
    raise ValueError("wshard datasets are read from a local root, not an object store")


def register() -> None:
    """Make ``storage_format: "wshard"`` resolvable. Idempotent."""
    from lerobot.datasets.storage import register_dataset_reader  # noqa: PLC0415

    register_dataset_reader("wshard", "wshard.lerobot")
