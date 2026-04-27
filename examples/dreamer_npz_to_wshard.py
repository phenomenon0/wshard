"""Convert a DreamerV3 NPZ episode to a .wshard file (one-way conversion demo).

Synthesizes a tiny DreamerV3-style NPZ fixture (T=100), loads it via
``wshard.dreamer.load_dreamer``, saves it as a .wshard file, and verifies the
image channel is byte-for-byte identical after reload.

This is the **one-way conversion** companion to ``examples/dreamer_roundtrip.py``
(which is the full round-trip / byte-level verification demo). The key difference:
this script shows the import path for production pipelines — take a DreamerV3 NPZ
from your replay buffer and emit a .wshard file for downstream training.

DreamerV3 NPZ schema used (confirmed from wshard/dreamer.py):
  - image        u8  [T, 64, 64, 3]   RGB observation
  - action       f32 [T, 4]           continuous action
  - reward       f32 [T]              scalar reward
  - is_terminal  bool [T]             true termination flag
  - is_last      bool [T]             episode end / truncation flag
  - is_first     bool [T]             episode start flag (stored in metadata)

Run:
    python examples/dreamer_npz_to_wshard.py
    python examples/dreamer_npz_to_wshard.py /tmp/my_output_dir
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from wshard.dreamer import load_dreamer
from wshard import load_wshard, save_wshard


T = 100


def synth_dreamer_npz(path: Path) -> None:
    """Write a synthetic DreamerV3-style NPZ fixture."""
    rng = np.random.default_rng(42)
    is_terminal = np.zeros(T, dtype=np.bool_)
    is_terminal[-1] = True
    is_last = is_terminal.copy()
    is_first = np.zeros(T, dtype=np.bool_)
    is_first[0] = True
    np.savez(
        path,
        image=rng.integers(0, 256, size=(T, 64, 64, 3), dtype=np.uint8),
        action=rng.standard_normal((T, 4)).astype(np.float32),
        reward=rng.standard_normal(T).astype(np.float32),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
    )


def main(tmp_dir: str = "/tmp") -> None:
    base = Path(tmp_dir)
    npz_path = base / "dreamer_input.npz"
    wshard_path = base / "dreamer.wshard"

    # 1. Synthesize NPZ fixture
    synth_dreamer_npz(npz_path)
    npz_size = npz_path.stat().st_size
    print(f"input  NPZ         : {npz_path}  ({npz_size:,} bytes)")

    # 2. Load via load_dreamer
    ep = load_dreamer(npz_path)
    assert ep.length == T, f"unexpected episode length {ep.length}"

    # 3. Save as .wshard
    save_wshard(ep, wshard_path)
    wshard_size = wshard_path.stat().st_size
    print(f"output .wshard     : {wshard_path}  ({wshard_size:,} bytes)")

    # 4. Reload and verify
    ep2 = load_wshard(wshard_path)

    # Image comparison: use tobytes() because shape metadata may be lost on
    # reload for multi-dim channels (known limitation; see dreamer_roundtrip.py).
    img_orig = ep.observations["image"].data
    img_back = ep2.observations["image"].data
    ok = img_orig.tobytes() == img_back.tobytes()

    reward_ok = np.array_equal(ep.rewards.data, ep2.rewards.data)

    n_channels = (
        len(ep.observations)
        + len(ep.actions)
        + (1 if ep.rewards is not None else 0)
        + (1 if ep.terminations is not None else 0)
    )
    print(f"channels written   : {n_channels}")
    print(f"episode length     : {ep.length}")
    print(f"image shape        : {img_orig.shape}")
    print(f"image round-trip   : {'OK' if ok else 'FAIL'}")
    print(f"reward round-trip  : {'OK' if reward_ok else 'FAIL'}")
    overall = "OK" if (ok and reward_ok) else "FAIL"
    print(f"conversion         : {overall}")


if __name__ == "__main__":
    tmp = sys.argv[1] if len(sys.argv) > 1 else "/tmp"
    main(tmp)
