"""
DreamerV3 NPZ format adapter.

DreamerV3 stores episodes as NumPy NPZ (zip of .npy files).
File naming convention: {timestamp}-{uuid}-{succ}-{length}.npz

Common keys:
  - is_first, is_last, is_terminal: episode boundaries
  - image: RGB observations [T, H, W, C]
  - reward: scalar rewards [T]
  - action, act_disc, act_cont: actions
  - stepid: step indices
"""

import io
from pathlib import Path
from typing import Union, Optional, BinaryIO
import numpy as np

from .types import Episode, Channel, DType, Format


# Known non-observation keys in DreamerV3
NON_OBS_KEYS = {
    "action",
    "act",
    "act_disc",
    "act_cont",
    "reward",
    "is_first",
    "is_last",
    "is_terminal",
    "stepid",
    "discount",
    "done",
}


def numpy_dtype_to_dtype(np_dtype: np.dtype) -> DType:
    """Convert NumPy dtype to DType."""
    mapping = {
        np.float64: DType.FLOAT64,
        np.float32: DType.FLOAT32,
        np.float16: DType.FLOAT16,
        np.int64: DType.INT64,
        np.int32: DType.INT32,
        np.int16: DType.INT16,
        np.int8: DType.INT8,
        np.uint64: DType.UINT64,
        np.uint32: DType.UINT32,
        np.uint16: DType.UINT16,
        np.uint8: DType.UINT8,
        np.bool_: DType.BOOL,
    }
    return mapping.get(np_dtype.type, DType.FLOAT32)


def load_dreamer(
    path_or_file: Union[str, Path, BinaryIO],
    uuid: Optional[str] = None,
) -> Episode:
    """
    Load an episode from DreamerV3 NPZ format.

    Args:
        path_or_file: File path or file-like object
        uuid: Optional episode UUID (extracted from filename if not provided)

    Returns:
        Episode with loaded data
    """
    # Load NPZ
    if isinstance(path_or_file, (str, Path)):
        path = Path(path_or_file)
        npz = np.load(path, allow_pickle=True)
        if uuid is None:
            # Try to extract from filename: {timestamp}-{uuid}-{succ}-{length}.npz
            parts = path.stem.split("-")
            if len(parts) >= 4:
                uuid = parts[1]
            else:
                uuid = path.stem
    else:
        npz = np.load(path_or_file, allow_pickle=True)
        uuid = uuid or "unknown"

    # Determine episode length
    length = 0
    for key in npz.files:
        arr = npz[key]
        if len(arr.shape) > 0 and arr.shape[0] > length:
            length = arr.shape[0]

    if length == 0:
        raise ValueError("No valid arrays found in NPZ")

    # Create episode
    ep = Episode(id=uuid, length=length)
    ep.source_format = Format.DREAMER_V3

    # Known observation keys to check first
    obs_keys = ["image", "vector", "state", "proprio", "obs"]

    # Load known observations
    for key in obs_keys:
        if key in npz.files:
            arr = npz[key]
            ch = _array_to_channel(key, arr)
            if ch:
                ep.observations[key] = ch

    # Load remaining observation-like keys
    for key in npz.files:
        if key in NON_OBS_KEYS:
            continue
        if key in ep.observations:
            continue
        arr = npz[key]
        if len(arr.shape) > 1:  # Multi-dimensional = likely observation
            ch = _array_to_channel(key, arr)
            if ch:
                ep.observations[key] = ch

    # Load actions
    for key in ["action", "act", "act_disc", "act_cont"]:
        if key in npz.files:
            arr = npz[key]
            ch = _array_to_channel(key, arr)
            if ch:
                ep.actions[key] = ch

    # Load reward
    if "reward" in npz.files:
        arr = npz["reward"]
        ep.rewards = _array_to_channel("reward", arr)

    # Load terminations
    if "is_terminal" in npz.files:
        arr = npz["is_terminal"]
        ep.terminations = _array_to_channel("is_terminal", arr)
    elif "is_last" in npz.files:
        arr = npz["is_last"]
        ep.terminations = _array_to_channel("is_last", arr)

    # Load truncations (is_last when not terminal)
    if "is_last" in npz.files and "is_terminal" in npz.files:
        arr = npz["is_last"]
        ep.truncations = _array_to_channel("is_last", arr)

    # Store metadata
    if "is_first" in npz.files:
        ep.metadata["is_first"] = npz["is_first"]
    if "stepid" in npz.files:
        ep.metadata["stepid"] = npz["stepid"]

    return ep


def save_dreamer(
    ep: Episode,
    path_or_file: Union[str, Path, BinaryIO],
) -> None:
    """
    Save an episode to DreamerV3 NPZ format.

    Args:
        ep: Episode to save
        path_or_file: Output file path or file-like object
    """
    # Validate episode
    ep.validate()

    # Build arrays dict
    arrays = {}

    # Write observations
    for name, ch in ep.observations.items():
        arrays[name] = ch.data

    # Write actions
    for name, ch in ep.actions.items():
        arrays[name] = ch.data

    # Write reward
    if ep.rewards is not None:
        arrays["reward"] = ep.rewards.data

    # Write is_terminal
    if ep.terminations is not None:
        arrays["is_terminal"] = ep.terminations.data

    # Compute is_last (terminations OR truncations)
    is_last = np.zeros(ep.length, dtype=np.bool_)
    if ep.terminations is not None:
        is_last |= ep.terminations.data.astype(np.bool_)
    if ep.truncations is not None:
        is_last |= ep.truncations.data.astype(np.bool_)
    is_last[-1] = True  # Last step is always last
    arrays["is_last"] = is_last

    # Compute is_first
    is_first = np.zeros(ep.length, dtype=np.bool_)
    is_first[0] = True
    if ep.terminations is not None:
        for i in range(ep.length - 1):
            if ep.terminations.data[i]:
                is_first[i + 1] = True
    if ep.truncations is not None:
        for i in range(ep.length - 1):
            if ep.truncations.data[i]:
                is_first[i + 1] = True
    arrays["is_first"] = is_first

    # Save
    if isinstance(path_or_file, (str, Path)):
        np.savez(path_or_file, **arrays)
    else:
        np.savez(path_or_file, **arrays)


def _array_to_channel(name: str, arr: np.ndarray) -> Optional[Channel]:
    """Convert a NumPy array to a Channel."""
    if arr is None or arr.size == 0:
        return None

    if len(arr.shape) == 0:
        return None

    # Shape is [T, ...] - first dim is timesteps
    per_step_shape = list(arr.shape[1:]) if len(arr.shape) > 1 else []

    return Channel(
        name=name,
        dtype=numpy_dtype_to_dtype(arr.dtype),
        shape=per_step_shape,
        data=arr,
    )
