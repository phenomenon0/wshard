"""Print the block layout of a .wshard file without decoding tensor data.

Run:
    python examples/inspect_wshard.py path/to/episode.wshard

Lists every block with its dtype, shape, compression flag, and on-disk size.
Useful for sanity-checking files written by other implementations.
"""

from __future__ import annotations

import sys

from wshard import load_wshard


def main(path: str) -> None:
    ep = load_wshard(path)
    print(f"=== {path} ===")
    print(f"id={ep.id!r}  env_id={ep.env_id!r}  length={ep.length}")
    print()
    print(f"{'block':40s}  {'dtype':6s}  {'shape':20s}")
    print("-" * 70)

    def row(prefix: str, name: str, ch) -> None:
        full = f"{prefix}{name}"
        dtype = ch.dtype.name.lower() if hasattr(ch.dtype, "name") else str(ch.dtype)
        shape = "x".join(str(s) for s in ch.data.shape) if ch.data is not None else "?"
        print(f"{full:40s}  {dtype:6s}  {shape:20s}")

    for name, ch in sorted(ep.observations.items()):
        row("signal/", name, ch)
    for name, ch in sorted(ep.actions.items()):
        row("action/", name, ch)
    if ep.rewards is not None:
        row("", "reward", ep.rewards)
    if ep.terminations is not None:
        row("", "done", ep.terminations)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python examples/inspect_wshard.py <path.wshard>", file=sys.stderr)
        raise SystemExit(2)
    main(sys.argv[1])
