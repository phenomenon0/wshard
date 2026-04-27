"""Read a .wshard episode and print its summary.

Run:
    python examples/read_cartpole.py [path]

Defaults to examples/cartpole.wshard (produced by write_cartpole.py).
"""

from __future__ import annotations

import sys

from wshard import load_wshard


def main(path: str = "examples/cartpole.wshard") -> None:
    ep = load_wshard(path)
    print(f"episode_id : {ep.id}")
    print(f"env_id     : {ep.env_id}")
    print(f"length     : {ep.length}")
    print(f"observations: {sorted(ep.observations.keys())}")
    print(f"actions    : {sorted(ep.actions.keys())}")
    if ep.rewards is not None:
        print(f"reward     : shape={ep.rewards.data.shape} mean={ep.rewards.data.mean():.4f}")
    if ep.terminations is not None:
        print(f"done       : terminal_steps={int(ep.terminations.data.sum())}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "examples/cartpole.wshard")
