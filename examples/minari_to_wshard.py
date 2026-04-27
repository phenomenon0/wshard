"""Convert a Minari HDF5 dataset episode to a .wshard file (experimental).

EXPERIMENTAL — schema-specific to common Minari layout (v0.5+). Non-default
datasets with nested observation dicts or image observations will need the
``_minari_episode_to_wshard`` function to be adjusted accordingly.

The Minari HDF5 layout synthesized here mirrors the standard Minari schema:

  <root>/
    episode_0/
      observations    f32 [T, obs_dim]
      actions         f32 [T, act_dim]
      rewards         f32 [T]
      terminations    bool [T]
      truncations     bool [T]

This script implements the conversion inline via ``_minari_episode_to_wshard``
rather than patching ``wshard.convert`` — the library converter can be wired up
later by following this function's pattern.

Requires: h5py  (pip install h5py)

Run:
    python examples/minari_to_wshard.py
    python examples/minari_to_wshard.py /tmp/my_output_dir
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Optional dependency guard — skip cleanly if h5py not installed
try:
    import h5py
except ImportError:
    print("h5py not installed — skipping minari_to_wshard example.")
    print("Install with: pip install h5py")
    sys.exit(0)

from wshard import load_wshard, save_wshard
from wshard.types import Channel, DType, Episode, Format


# ---------------------------------------------------------------------------
# Minari HDF5 -> Episode converter (inline; not part of wshard library yet)
# ---------------------------------------------------------------------------

def _minari_episode_to_wshard(h5_path: Path, episode_key: str = "episode_0") -> Episode:
    """Load one Minari HDF5 episode and return a wshard Episode.

    Mapping:
      observations  -> ep.observations["state"]   shape=[obs_dim]
      actions       -> ep.actions["main"]          shape=[act_dim]
      rewards       -> ep.rewards                  shape=[]
      terminations  -> ep.terminations             shape=[]
      truncations   -> ep.truncations              shape=[]

    Args:
        h5_path: Path to the .h5 file.
        episode_key: HDF5 group key for the episode (e.g. "episode_0").

    Returns:
        Populated Episode.
    """
    with h5py.File(h5_path, "r") as f:
        grp = f[episode_key]

        obs = grp["observations"][:]          # [T, obs_dim]
        acts = grp["actions"][:]              # [T, act_dim]
        rewards = grp["rewards"][:]           # [T]
        terminations = grp["terminations"][:] # [T]
        truncations = grp["truncations"][:]   # [T]

    T = obs.shape[0]
    obs_dim = obs.shape[1] if obs.ndim > 1 else 1
    act_dim = acts.shape[1] if acts.ndim > 1 else 1

    ep = Episode(id=episode_key, length=T)
    ep.source_format = Format.MINARI

    ep.observations["state"] = Channel(
        name="state",
        dtype=DType.FLOAT32,
        shape=[obs_dim],
        data=obs.astype(np.float32),
    )
    ep.actions["main"] = Channel(
        name="main",
        dtype=DType.FLOAT32,
        shape=[act_dim],
        data=acts.astype(np.float32),
    )
    ep.rewards = Channel(
        name="reward",
        dtype=DType.FLOAT32,
        shape=[],
        data=rewards.astype(np.float32),
    )
    ep.terminations = Channel(
        name="terminations",
        dtype=DType.BOOL,
        shape=[],
        data=terminations.astype(np.bool_),
    )
    ep.truncations = Channel(
        name="truncations",
        dtype=DType.BOOL,
        shape=[],
        data=truncations.astype(np.bool_),
    )

    return ep


# ---------------------------------------------------------------------------
# Fixture synthesis
# ---------------------------------------------------------------------------

def _synth_minari_h5(path: Path, T: int = 100) -> None:
    """Write a minimal Minari-layout HDF5 fixture."""
    rng = np.random.default_rng(42)

    obs_dim = 17   # Hopper-v2 observation space
    act_dim = 6    # Hopper-v2 action space (3 joints, but use 6 for realism)

    terminations = np.zeros(T, dtype=np.bool_)
    terminations[-1] = True
    truncations = np.zeros(T, dtype=np.bool_)

    with h5py.File(path, "w") as f:
        grp = f.create_group("episode_0")
        grp.create_dataset("observations",  data=rng.standard_normal((T, obs_dim)).astype(np.float32))
        grp.create_dataset("actions",       data=rng.standard_normal((T, act_dim)).astype(np.float32))
        grp.create_dataset("rewards",       data=rng.standard_normal(T).astype(np.float32))
        grp.create_dataset("terminations",  data=terminations)
        grp.create_dataset("truncations",   data=truncations)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(tmp_dir: str = "/tmp") -> None:
    base = Path(tmp_dir)
    h5_path = base / "minari_input.h5"
    wshard_path = base / "minari.wshard"

    # 1. Synthesize Minari HDF5 fixture
    _synth_minari_h5(h5_path)
    h5_size = h5_path.stat().st_size
    print(f"input  HDF5        : {h5_path}  ({h5_size:,} bytes)")

    # 2. Convert: Minari HDF5 -> Episode
    ep = _minari_episode_to_wshard(h5_path, episode_key="episode_0")
    assert ep.length == 100, f"unexpected episode length {ep.length}"

    # 3. Save as .wshard
    save_wshard(ep, wshard_path)
    wshard_size = wshard_path.stat().st_size
    print(f"output .wshard     : {wshard_path}  ({wshard_size:,} bytes)")

    # 4. Reload and verify observation channel
    ep2 = load_wshard(wshard_path)

    obs_orig = ep.observations["state"].data
    obs_back = ep2.observations["state"].data
    obs_ok = np.array_equal(obs_orig, obs_back)

    reward_ok = np.array_equal(ep.rewards.data, ep2.rewards.data)

    n_channels = (
        len(ep.observations)
        + len(ep.actions)
        + (1 if ep.rewards is not None else 0)
        + (1 if ep.terminations is not None else 0)
        + (1 if ep.truncations is not None else 0)
    )
    print(f"channels written   : {n_channels}")
    print(f"episode length     : {ep.length}")
    print(f"obs shape          : {obs_orig.shape}  (Hopper-v2 dims)")
    print(f"obs round-trip     : {'OK' if obs_ok else 'FAIL'}")
    print(f"reward round-trip  : {'OK' if reward_ok else 'FAIL'}")
    overall = "OK" if (obs_ok and reward_ok) else "FAIL"
    print(f"conversion         : {overall}")


if __name__ == "__main__":
    tmp = sys.argv[1] if len(sys.argv) > 1 else "/tmp"
    main(tmp)
