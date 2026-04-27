"""Write a synthetic CartPole-style episode as a .wshard file.

Run:
    python examples/write_cartpole.py [output_path]

Produces an episode of length 200 with:
  - signal/state  [200, 4] float32   pole angle, angular velocity, etc.
  - action/ctrl   [200, 1] int32     discrete left/right action
  - reward        [200]    float32
  - done          [200]    bool      single terminal at the end
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from wshard import save_wshard
from wshard.types import Channel, DType, Episode


def main(out_path: str = "examples/cartpole.wshard") -> None:
    rng = np.random.default_rng(0)
    T = 200

    state = rng.standard_normal((T, 4)).astype(np.float32)
    ctrl = rng.integers(0, 2, size=(T, 1), dtype=np.int32)
    reward = np.ones(T, dtype=np.float32)
    done = np.zeros(T, dtype=bool)
    done[-1] = True

    ep = Episode(id="cartpole_demo_000", length=T)
    ep.env_id = "CartPole-v1"
    ep.observations["state"] = Channel(name="state", dtype=DType.FLOAT32, shape=[4], data=state)
    ep.actions["ctrl"] = Channel(name="ctrl", dtype=DType.INT32, shape=[1], data=ctrl)
    ep.rewards = Channel(name="reward", dtype=DType.FLOAT32, shape=[], data=reward)
    ep.terminations = Channel(name="done", dtype=DType.BOOL, shape=[], data=done)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    save_wshard(ep, out_path)
    print(f"wrote {out_path}  ({T} steps)")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "examples/cartpole.wshard"
    main(out)
