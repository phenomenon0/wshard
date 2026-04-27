"""Convert a .wshard episode to LeRobot-style Parquet (experimental).

EXPERIMENTAL — mirrors LeRobot's flat row-per-timestep Parquet schema as of
lerobot v2.x (data_format.md). Real LeRobot datasets also include an info.json
and MP4 video files; this script produces only the Parquet shard.

LeRobot Parquet schema used here:
  observation.state.{0..6}   f32   flattened state vector (7 dims)
  action.{0..5}              f32   flattened action vector (6 dims)
  reward                     f64   scalar reward
  next.done                  bool  episode termination
  episode_index              i64   always 0 (single episode)
  frame_index                i64   timestep within episode
  timestamp                  f64   frame_index / fps

Output: /tmp/lerobot_demo/data/chunk-000/episode_000000.parquet

Requires: pyarrow  (pip install pyarrow)

Run:
    python examples/wshard_to_lerobot.py
    python examples/wshard_to_lerobot.py /tmp/my_output_dir
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Optional dependency guard — skip cleanly if pyarrow not installed
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    print("pyarrow not installed — skipping wshard_to_lerobot example.")
    print("Install with: pip install pyarrow")
    sys.exit(0)

from wshard import load_wshard, save_wshard
from wshard.types import Channel, DType, Episode


FPS = 30
T = 200
STATE_DIM = 7
ACTION_DIM = 6


# ---------------------------------------------------------------------------
# Episode -> LeRobot Parquet converter (inline; not part of wshard library yet)
# ---------------------------------------------------------------------------

def _episode_to_lerobot_table(ep: Episode, fps: float = FPS) -> pa.Table:
    """Convert a wshard Episode to a PyArrow Table matching LeRobot schema.

    The state channel is expected at ``ep.observations["signal"]`` with shape
    [T, STATE_DIM], and the action channel at ``ep.actions["ctrl"]`` with shape
    [T, ACTION_DIM]. Adjust key names and dims for other environments.

    NOTE: save_wshard does not preserve multi-dim action channel shape in the
    shard metadata, so actions reload as flat [T*ACTION_DIM]. This function
    reshapes them using the channel's declared ``shape`` list when available,
    or falls back to the caller-supplied ``ACTION_DIM`` constant. This is a
    known library limitation; see ``examples/dreamer_roundtrip.py`` for details.

    Args:
        ep: Source episode.
        fps: Frames-per-second used to compute the ``timestamp`` column.

    Returns:
        PyArrow Table with one row per timestep.
    """
    state = ep.observations["signal"].data   # [T, STATE_DIM]
    action_raw = ep.actions["ctrl"].data     # [T, ACTION_DIM] or [T*ACTION_DIM] (flat reload)

    # Reshape action if it came back flat (known save_wshard limitation)
    if action_raw.ndim == 1:
        action = action_raw.reshape(ep.length, ACTION_DIM)
    else:
        action = action_raw

    reward = ep.rewards.data                 # [T]
    done = ep.terminations.data              # [T]
    frame_index = np.arange(ep.length, dtype=np.int64)
    timestamp = frame_index.astype(np.float64) / fps
    episode_index = np.zeros(ep.length, dtype=np.int64)

    arrays: dict[str, pa.Array] = {}

    # Flatten state: observation.state.0 .. observation.state.{STATE_DIM-1}
    for i in range(state.shape[1]):
        arrays[f"observation.state.{i}"] = pa.array(state[:, i].astype(np.float32))

    # Flatten action: action.0 .. action.{ACTION_DIM-1}
    for i in range(action.shape[1]):
        arrays[f"action.{i}"] = pa.array(action[:, i].astype(np.float32))

    arrays["reward"]        = pa.array(reward.astype(np.float64))
    arrays["next.done"]     = pa.array(done.astype(bool))
    arrays["episode_index"] = pa.array(episode_index)
    arrays["frame_index"]   = pa.array(frame_index)
    arrays["timestamp"]     = pa.array(timestamp)

    return pa.table(arrays)


# ---------------------------------------------------------------------------
# Fixture synthesis
# ---------------------------------------------------------------------------

def _synth_episode() -> Episode:
    """Build a small synthetic wshard Episode."""
    rng = np.random.default_rng(42)

    state = rng.standard_normal((T, STATE_DIM)).astype(np.float32)
    action = rng.standard_normal((T, ACTION_DIM)).astype(np.float32)
    reward = rng.standard_normal(T).astype(np.float32)
    done = np.zeros(T, dtype=np.bool_)
    done[-1] = True

    ep = Episode(id="lerobot_demo_000", length=T)
    ep.env_id = "gym_pusht/PushT-v0"
    ep.observations["signal"] = Channel(
        name="signal", dtype=DType.FLOAT32, shape=[STATE_DIM], data=state
    )
    ep.actions["ctrl"] = Channel(
        name="ctrl", dtype=DType.FLOAT32, shape=[ACTION_DIM], data=action
    )
    ep.rewards = Channel(name="reward", dtype=DType.FLOAT32, shape=[], data=reward)
    ep.terminations = Channel(name="done", dtype=DType.BOOL, shape=[], data=done)
    return ep


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(tmp_dir: str = "/tmp") -> None:
    base = Path(tmp_dir)
    wshard_path = base / "lerobot_source.wshard"
    parquet_dir = base / "lerobot_demo" / "data" / "chunk-000"
    parquet_path = parquet_dir / "episode_000000.parquet"

    # 1. Build and save synthetic episode as .wshard
    ep_orig = _synth_episode()
    save_wshard(ep_orig, wshard_path)
    wshard_size = wshard_path.stat().st_size
    print(f"input  .wshard     : {wshard_path}  ({wshard_size:,} bytes)")

    # 2. Reload episode from .wshard
    ep = load_wshard(wshard_path)

    # 3. Convert to LeRobot Parquet
    parquet_dir.mkdir(parents=True, exist_ok=True)
    table = _episode_to_lerobot_table(ep, fps=FPS)
    pq.write_table(table, parquet_path, compression="zstd")
    parquet_size = parquet_path.stat().st_size
    print(f"output .parquet    : {parquet_path}  ({parquet_size:,} bytes)")

    # 4. Round-trip verify: read Parquet back and compare
    table2 = pq.read_table(parquet_path)
    state_orig = ep.observations["signal"].data
    state_back = np.stack(
        [table2[f"observation.state.{i}"].to_pylist() for i in range(STATE_DIM)],
        axis=1,
    ).astype(np.float32)
    state_ok = np.array_equal(state_orig, state_back)

    reward_orig = ep.rewards.data
    reward_back = np.array(table2["reward"].to_pylist(), dtype=np.float32)
    reward_ok = np.allclose(reward_orig, reward_back, atol=1e-6)

    n_cols = len(table.schema)
    print(f"parquet columns    : {n_cols}")
    print(f"episode length     : {ep.length}  rows")
    print(f"state round-trip   : {'OK' if state_ok else 'FAIL'}")
    print(f"reward round-trip  : {'OK' if reward_ok else 'FAIL'}")
    overall = "OK" if (state_ok and reward_ok) else "FAIL"
    print(f"conversion         : {overall}")

    # 5. Print info.json-style stub (stdout only; no file written)
    info = {
        "fps": FPS,
        "episode_count": 1,
        "total_frames": ep.length,
        "action_space": {"shape": [ACTION_DIM], "dtype": "float32"},
        "observation_space": {"state": {"shape": [STATE_DIM], "dtype": "float32"}},
        "parquet_schema": [f.name for f in table.schema],
        "note": "MP4 video tracks and info.json not written — add via lerobot CLI for full compat",
    }
    import json
    print("\n--- info.json stub (not written to disk) ---")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    tmp = sys.argv[1] if len(sys.argv) > 1 else "/tmp"
    main(tmp)
