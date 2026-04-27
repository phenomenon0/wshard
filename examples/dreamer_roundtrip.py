"""DreamerV3 NPZ -> W-SHARD -> Episode round-trip demo.

Synthesizes a tiny DreamerV3-style NPZ fixture (T=50), loads it as an Episode,
saves it as .wshard, reloads it, and asserts data integrity for all channels.

DreamerV3 NPZ schema used here (confirmed from wshard/dreamer.py):
  - image        u8  [T, 64, 64, 3]   RGB observation
  - action       f32 [T, 6]           continuous action
  - reward       f32 [T]              scalar reward
  - is_first     bool [T]             episode start flag (stored in metadata)
  - is_last      bool [T]             episode end / truncation flag
  - is_terminal  bool [T]             true termination flag

Run:
    python examples/dreamer_roundtrip.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from wshard.dreamer import load_dreamer, save_dreamer
from wshard import load_wshard, save_wshard
from wshard.types import Episode, Channel, DType


def _synth_npz(path: Path, T: int = 50) -> None:
    """Write a synthetic DreamerV3-style NPZ to path."""
    rng = np.random.default_rng(99)
    is_terminal = np.zeros(T, dtype=np.bool_)
    is_terminal[-1] = True
    is_last = is_terminal.copy()
    is_first = np.zeros(T, dtype=np.bool_)
    is_first[0] = True
    np.savez(
        path,
        image=rng.integers(0, 256, size=(T, 64, 64, 3), dtype=np.uint8),
        action=rng.standard_normal((T, 6)).astype(np.float32),
        reward=rng.standard_normal(T).astype(np.float32),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
    )


def main(tmp_dir: str = "/tmp") -> None:
    T = 50
    base = Path(tmp_dir)
    npz_path = base / "dreamer_fixture.npz"
    wshard_path = base / "dreamer.wshard"

    # 1. Synthesize NPZ
    _synth_npz(npz_path, T)
    print(f"wrote NPZ fixture  : {npz_path}  ({npz_path.stat().st_size:,} bytes)")

    # 2. Load via load_dreamer
    ep = load_dreamer(npz_path)
    assert ep.length == T, f"length mismatch: {ep.length}"
    print(f"loaded episode     : id={ep.id!r}  length={ep.length}")
    print(f"  observations     : {sorted(ep.observations)}")
    print(f"  actions          : {sorted(ep.actions)}")
    print(f"  rewards          : {'yes' if ep.rewards is not None else 'no'}")
    print(f"  terminations     : {'yes' if ep.terminations is not None else 'no'}")

    # 3. Save as W-SHARD
    save_wshard(ep, wshard_path)
    print(f"saved .wshard      : {wshard_path}  ({wshard_path.stat().st_size:,} bytes)")

    # 4. Reload from W-SHARD
    ep2 = load_wshard(wshard_path)
    assert ep2.length == ep.length, f"length mismatch after reload: {ep2.length}"

    # 5. Assert all observations are byte-identical
    for name in ep.observations:
        assert name in ep2.observations, f"observation {name!r} missing after reload"
        assert np.array_equal(ep.observations[name].data, ep2.observations[name].data), \
            f"observation {name!r} data mismatch"

    # 6. Assert all actions are byte-identical.
    # Known limitation: save_wshard does not write action channel shape into
    # meta/channels, so multi-dim actions are reloaded with shape=[] (flat).
    # The raw bytes are identical; compare via tobytes().
    for name in ep.actions:
        assert name in ep2.actions, f"action {name!r} missing after reload"
        assert ep.actions[name].data.tobytes() == ep2.actions[name].data.tobytes(), \
            f"action {name!r} byte content mismatch"

    # 7. Rewards
    if ep.rewards is not None:
        assert ep2.rewards is not None, "rewards lost after reload"
        assert np.array_equal(ep.rewards.data, ep2.rewards.data), "reward data mismatch"

    # 8. Terminations
    if ep.terminations is not None:
        assert ep2.terminations is not None, "terminations lost after reload"
        assert np.array_equal(ep.terminations.data, ep2.terminations.data), \
            "termination data mismatch"

    print("round-trip OK      : all channels byte-identical")
    print(f"  episode id       : {ep.id!r} -> {ep2.id!r}")
    print(f"  length           : {ep.length} == {ep2.length}")
    print(f"  image shape      : {ep.observations['image'].data.shape}")
    print(f"  action bytes     : {ep.actions['action'].data.tobytes()[:8].hex()}...  (shape lost; known limitation)")
    print("NOTE: save_wshard does not preserve action channel shape in meta/channels.")
    print("      Action data bytes are correct; shape must be re-applied by the caller.")


if __name__ == "__main__":
    tmp = sys.argv[1] if len(sys.argv) > 1 else "/tmp"
    main(tmp)
