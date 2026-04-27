"""Live-record a 500-step synthetic episode using WShardStreamWriter.

Shows the .partial -> final rename pattern for crash-safe online recording.

Run:
    python examples/streaming_demo.py [output_path]

Channels:
  - signal/joint_pos   [500, 7] float32   synthetic joint positions
  - action/ctrl        [500, 7] float32   synthetic control outputs
  - reward             [500]    float32
  - done               [500]    uint8      terminal flag
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from wshard.streaming import WShardStreamWriter, ChannelDef
from wshard.types import DType
from wshard import load_wshard


def main(out_path: str = "examples/streaming_demo.wshard") -> None:
    rng = np.random.default_rng(42)
    T = 500
    D = 7

    # Pre-generate deterministic data
    joint_pos = rng.standard_normal((T, D)).astype(np.float32)
    ctrl = (rng.standard_normal((T, D)) * 0.1).astype(np.float32)
    rewards = rng.uniform(0.0, 1.0, T).astype(np.float32)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(out) + ".partial")

    channel_defs = [
        ChannelDef("joint_pos", DType.FLOAT32, [D]),
        ChannelDef("ctrl", DType.FLOAT32, [D]),
    ]

    writer = WShardStreamWriter(out, "streaming_demo_000", channel_defs)
    writer.begin_episode(env_id="SyntheticArm-v0")

    # .partial exists from step 0 — the final path is not written until finalize
    assert partial.exists(), ".partial file must exist after begin_episode()"

    for t in range(T):
        done = t == T - 1
        writer.write_timestep(
            t=t,
            observations={"joint_pos": joint_pos[t]},
            actions={"ctrl": ctrl[t]},
            reward=float(rewards[t]),
            done=done,
        )
        # Crash-safe: at every step the .partial file is on disk
        assert partial.exists(), f"crash safety violated at step {t}"

    total_bytes = writer.end_episode()

    # After finalization: final exists, .partial gone
    assert out.exists(), "final file must exist after end_episode()"
    assert not partial.exists(), ".partial must be renamed away"

    # Verify the file loads correctly
    ep = load_wshard(out)
    assert ep.length == T, f"length mismatch: {ep.length} != {T}"

    print(f"wrote {out_path}")
    print(f"  steps          : {T}")
    print(f"  channels       : joint_pos [7], ctrl [7], reward, done")
    print(f"  file size      : {total_bytes:,} bytes  ({total_bytes / 1024:.1f} KiB)")
    print(f"  crash-safe     : yes — .partial present at every step, renamed on finalize")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "examples/streaming_demo.wshard"
    main(out)
