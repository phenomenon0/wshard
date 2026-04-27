"""bench_python.py — WShard vs NumPy NPZ write/read benchmark.

Workload: synthetic T=1000 step episode
  signal/joint_pos  [1000, 7]       float32   ~28 KB
  signal/rgb        [1000, 84, 84, 3] uint8   ~21 MB
  action/ctrl       [1000, 7]       float32   ~28 KB
  reward            [1000]          float32   ~4 KB
  done              [1000]          bool      ~1 KB
  Total raw payload: ~21 MB

Configurations:
  wshard-none   — WShard, CompressionType.NONE
  wshard-zstd   — WShard, CompressionType.ZSTD
  wshard-lz4    — WShard, CompressionType.LZ4
  npz-deflate   — numpy savez_compressed (deflate/zlib)

Metrics:
  write_median_s, write_min_s  (seconds, 5 runs)
  read_median_s,  read_min_s
  file_bytes, raw_bytes, ratio
"""

from __future__ import annotations

import argparse
import io
import os
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from wshard import load_wshard, save_wshard
from wshard.compress import CompressionType
from wshard.types import Channel, DType, Episode

# ── workload parameters ────────────────────────────────────────────────────────

SEED = 42
T = 1000

JOINT_SHAPE = (T, 7)         # float32
RGB_SHAPE   = (T, 84, 84, 3) # uint8
CTRL_SHAPE  = (T, 7)         # float32
REWARD_SHAPE = (T,)           # float32
DONE_SHAPE   = (T,)           # bool

RUNS = 5


def make_data() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED)
    return {
        "joint_pos": rng.standard_normal(JOINT_SHAPE).astype(np.float32),
        "rgb":       rng.integers(0, 256, RGB_SHAPE, dtype=np.uint8),
        "ctrl":      rng.standard_normal(CTRL_SHAPE).astype(np.float32),
        "reward":    rng.standard_normal(REWARD_SHAPE).astype(np.float32),
        "done":      (rng.integers(0, 100, DONE_SHAPE) == 0),
    }


def raw_bytes(data: dict[str, np.ndarray]) -> int:
    return sum(a.nbytes for a in data.values())


def make_episode(data: dict[str, np.ndarray]) -> Episode:
    ep = Episode(id="bench_ep", length=T)
    ep.env_id = "BenchEnv-v0"
    ep.observations["joint_pos"] = Channel(
        name="joint_pos", dtype=DType.FLOAT32, shape=[7],
        data=data["joint_pos"],
    )
    ep.observations["rgb"] = Channel(
        name="rgb", dtype=DType.UINT8, shape=[84, 84, 3],
        data=data["rgb"],
    )
    ep.actions["ctrl"] = Channel(
        name="ctrl", dtype=DType.FLOAT32, shape=[7],
        data=data["ctrl"],
    )
    ep.rewards = Channel(
        name="reward", dtype=DType.FLOAT32, shape=[],
        data=data["reward"],
    )
    ep.terminations = Channel(
        name="done", dtype=DType.BOOL, shape=[],
        data=data["done"],
    )
    return ep


# ── timing helpers ─────────────────────────────────────────────────────────────

def _median_min(times: list[float]) -> tuple[float, float]:
    return statistics.median(times), min(times)


# ── wshard benchmarks ──────────────────────────────────────────────────────────

def bench_wshard_write(ep: Episode, comp: CompressionType, path: str) -> tuple[float, float]:
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        save_wshard(ep, path, compression=comp)
        times.append(time.perf_counter() - t0)
    return _median_min(times)


def bench_wshard_read(path: str) -> tuple[float, float]:
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        _ = load_wshard(path)
        times.append(time.perf_counter() - t0)
    return _median_min(times)


# ── npz benchmarks ─────────────────────────────────────────────────────────────

def bench_npz_write(data: dict[str, np.ndarray], path: str) -> tuple[float, float]:
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        np.savez_compressed(path, **data)
        times.append(time.perf_counter() - t0)
    # savez_compressed appends .npz if not present; normalise
    if not path.endswith(".npz") and not os.path.exists(path):
        path = path + ".npz"
    return _median_min(times)


def bench_npz_read(path: str) -> tuple[float, float]:
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        with np.load(path) as f:
            _ = {k: f[k] for k in f.files}
        times.append(time.perf_counter() - t0)
    return _median_min(times)


# ── table formatting ───────────────────────────────────────────────────────────

def fmt_s(v: float) -> str:
    """Format seconds with adaptive precision."""
    if v < 0.001:
        return f"{v*1000:.2f} ms"
    if v < 1.0:
        return f"{v*1000:.1f} ms"
    return f"{v:.3f} s"


def fmt_mb(b: int) -> str:
    return f"{b/1_048_576:.2f} MB"


def fmt_ratio(raw: int, disk: int) -> str:
    if disk == 0:
        return "—"
    return f"{raw/disk:.2f}×"


def build_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "Config",
        "Write (median)",
        "Write (min)",
        "Read (median)",
        "Read (min)",
        "File size",
        "Ratio",
    ]
    sep = "|" + "|".join("-" * (len(h) + 2) for h in headers) + "|"
    header_row = "| " + " | ".join(headers) + " |"
    lines = [header_row, sep]
    for r in rows:
        cells = [
            r["config"],
            fmt_s(r["write_med"]),
            fmt_s(r["write_min"]),
            fmt_s(r["read_med"]),
            fmt_s(r["read_min"]),
            fmt_mb(r["file_bytes"]),
            fmt_ratio(r["raw_bytes"], r["file_bytes"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ── main ───────────────────────────────────────────────────────────────────────

def run_benchmarks() -> tuple[list[dict[str, Any]], int]:
    data = make_data()
    uncompressed = raw_bytes(data)
    ep = make_episode(data)

    print(f"Workload: T={T}, raw payload = {fmt_mb(uncompressed)}")
    print(f"Runs per measurement: {RUNS}\n")

    results: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory() as tmpdir:

        # ── WShard none ────────────────────────────────────────────────────────
        print("  [1/4] WShard-none ...")
        p = os.path.join(tmpdir, "ep_none.wshard")
        wm, wmin = bench_wshard_write(ep, CompressionType.NONE, p)
        file_sz = os.path.getsize(p)
        rm, rmin = bench_wshard_read(p)
        results.append(dict(config="wshard-none",  write_med=wm, write_min=wmin,
                            read_med=rm, read_min=rmin,
                            file_bytes=file_sz, raw_bytes=uncompressed))

        # ── WShard zstd ────────────────────────────────────────────────────────
        print("  [2/4] WShard-zstd ...")
        p = os.path.join(tmpdir, "ep_zstd.wshard")
        wm, wmin = bench_wshard_write(ep, CompressionType.ZSTD, p)
        file_sz = os.path.getsize(p)
        rm, rmin = bench_wshard_read(p)
        results.append(dict(config="wshard-zstd", write_med=wm, write_min=wmin,
                            read_med=rm, read_min=rmin,
                            file_bytes=file_sz, raw_bytes=uncompressed))

        # ── WShard lz4 ─────────────────────────────────────────────────────────
        print("  [3/4] WShard-lz4 ...")
        p = os.path.join(tmpdir, "ep_lz4.wshard")
        wm, wmin = bench_wshard_write(ep, CompressionType.LZ4, p)
        file_sz = os.path.getsize(p)
        rm, rmin = bench_wshard_read(p)
        results.append(dict(config="wshard-lz4",  write_med=wm, write_min=wmin,
                            read_med=rm, read_min=rmin,
                            file_bytes=file_sz, raw_bytes=uncompressed))

        # ── NumPy NPZ deflate ──────────────────────────────────────────────────
        print("  [4/4] NumPy NPZ-deflate ...")
        p = os.path.join(tmpdir, "ep.npz")
        wm, wmin = bench_npz_write(data, p)
        # numpy appends .npz; ensure correct path
        if not os.path.exists(p):
            p = p  # savez_compressed doesn't re-append when extension is already .npz
        file_sz = os.path.getsize(p)
        rm, rmin = bench_npz_read(p)
        results.append(dict(config="npz-deflate", write_med=wm, write_min=wmin,
                            read_med=rm, read_min=rmin,
                            file_bytes=file_sz, raw_bytes=uncompressed))

    return results, uncompressed


def main() -> None:
    parser = argparse.ArgumentParser(description="WShard vs NumPy NPZ benchmark")
    parser.add_argument("--output", metavar="FILE",
                        help="Write markdown table to FILE in addition to stdout")
    args = parser.parse_args()

    results, uncompressed = run_benchmarks()

    print()
    table = build_table(results)
    machine_note = (
        f"\n> Raw payload: {fmt_mb(uncompressed)} "
        f"({T} steps · joint_pos f32[7] + rgb u8[84,84,3] + ctrl f32[7] + reward f32 + done bool)"
    )
    output = table + machine_note

    print(output)

    if args.output:
        Path(args.output).write_text(output + "\n")
        print(f"\nTable written to {args.output}")


if __name__ == "__main__":
    main()
