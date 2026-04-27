"""bench_python.py — WShard vs NumPy NPZ write/read benchmark.

Workload: synthetic T=1000 step episode — structured-but-realistic
  signal/joint_pos  [1000, 7]         float32  ~28 KB
  signal/rgb        [1000, 84, 84, 3] uint8    ~21 MB  (gradient + low noise)
  action/ctrl       [1000, 7]         float32  ~28 KB
  reward            [1000]            float32  ~4 KB
  done              [1000]            bool     ~1 KB
  Total raw payload: ~21 MB

Configurations (full-episode write/read):
  wshard-none   — WShard, CompressionType.NONE
  wshard-zstd   — WShard, CompressionType.ZSTD
  wshard-lz4    — WShard, CompressionType.LZ4
  npz-deflate   — numpy savez_compressed (deflate/zlib)
  hdf5-deflate  — h5py gzip level 4 (optional — install h5py to enable)

Partial-block read section:
  Write ONE large wshard (~50 MB: rgb + 4 depth fillers) then time fetching
  only the 28 KB action/ctrl block, vs HDF5 and NPZ.

Metrics:
  write_median_s, write_min_s  (seconds, 5 runs)
  read_median_s,  read_min_s
  file_bytes, raw_bytes, ratio
"""

from __future__ import annotations

import argparse
import os
import statistics
import struct
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

JOINT_SHAPE  = (T, 7)          # float32
RGB_SHAPE    = (T, 84, 84, 3)  # uint8
CTRL_SHAPE   = (T, 7)          # float32
REWARD_SHAPE = (T,)             # float32
DONE_SHAPE   = (T,)             # bool

RUNS = 5


def make_rgb(T: int = 1000, H: int = 84, W: int = 84) -> np.ndarray:
    """Structured-but-realistic RGB: vertical gradient + low-amplitude noise.

    A smooth gradient (0-255, top-to-bottom) with ±2 per-pixel noise mimics
    what a static camera in a uniform-light scene looks like.  This achieves
    ~2× zstd compression, which is a conservative lower bound for real RGB.
    Random bytes (the previous workload) are incompressible and give 1.00×.

    Noise amplitude: ±2 LSB  (rng.integers(0, 4) − 2)
    Compresses ~2× with zstd, ~1.1× with lz4.
    """
    rng = np.random.default_rng(SEED + 1)  # distinct sub-seed for RGB
    base = np.linspace(0, 255, H, dtype=np.float32)[:, None, None]
    base = np.broadcast_to(base, (H, W, 3)).copy()  # vertical gradient
    noise = (rng.integers(0, 4, (T, H, W, 3)) - 2).astype(np.int16)
    frames = (base[None] + noise).clip(0, 255).astype(np.uint8)
    return frames


def make_data() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED)
    return {
        "joint_pos": rng.standard_normal(JOINT_SHAPE).astype(np.float32),
        "rgb":       make_rgb(T),
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


# ── hdf5 benchmarks (optional) ─────────────────────────────────────────────────

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("  [info] h5py not installed — skipping hdf5-deflate rows.")
    print("         Install with: pip install h5py")


def bench_hdf5_write(data: dict[str, np.ndarray], path: str) -> tuple[float, float]:
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        with h5py.File(path, "w") as f:
            for k, v in data.items():
                f.create_dataset(k, data=v, compression="gzip", compression_opts=4)
        times.append(time.perf_counter() - t0)
    return _median_min(times)


def bench_hdf5_read(path: str) -> tuple[float, float]:
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        with h5py.File(path, "r") as f:
            _ = {k: f[k][:] for k in f.keys()}
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


def fmt_ms(v: float) -> str:
    return f"{v*1000:.2f} ms"


def fmt_us(v: float) -> str:
    return f"{v*1e6:.0f} µs"


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


# ── partial-block read helpers ─────────────────────────────────────────────────

def _wshard_read_block_partial(path: str, block_name: str) -> bytes:
    """Read a single named block from a wshard file without decoding others.

    Uses the low-level index scan directly on the raw file so we touch only
    the header + index + the one requested data block on disk.
    """
    from wshard.wshard import (
        HEADER_SIZE, INDEX_ENTRY_SIZE,
        _parse_index_entry, compute_crc32,
        BLOCK_FLAG_COMPRESSED, BLOCK_FLAG_ZSTD, BLOCK_FLAG_LZ4,
    )
    from wshard.compress import Compressor, compression_from_byte

    with open(path, "rb") as fh:
        # Read header
        header = fh.read(HEADER_SIZE)
        compression_default = compression_from_byte(header[9])
        index_entry_size = struct.unpack("<H", header[10:12])[0]
        entry_count = struct.unpack("<I", header[12:16])[0]
        string_table_offset = struct.unpack("<Q", header[16:24])[0]
        data_section_offset = struct.unpack("<Q", header[24:32])[0]

        # Read full index
        index_data = fh.read(entry_count * index_entry_size)

        # Read string table
        string_table_size = data_section_offset - string_table_offset
        fh.seek(string_table_offset)
        string_table = fh.read(string_table_size)

        # Find the entry for block_name
        for i in range(entry_count):
            raw = index_data[i * index_entry_size:(i + 1) * index_entry_size]
            entry = _parse_index_entry(raw)
            name_offset = entry["name_offset"]
            name_len = entry["name_len"]
            if name_offset + name_len > len(string_table):
                continue
            name = string_table[name_offset:name_offset + name_len].decode("utf-8")
            if name != block_name:
                continue

            # Seek directly to the block's data
            fh.seek(entry["data_offset"])
            block_data = fh.read(entry["disk_size"])

            # Decompress if needed
            entry_flags = entry["flags"]
            orig_size = entry["orig_size"]
            disk_size = entry["disk_size"]
            if (entry_flags & BLOCK_FLAG_COMPRESSED) and disk_size != orig_size:
                if entry_flags & BLOCK_FLAG_LZ4:
                    comp_type = CompressionType.LZ4
                elif entry_flags & BLOCK_FLAG_ZSTD:
                    comp_type = CompressionType.ZSTD
                else:
                    comp_type = compression_default
                decompressor = Compressor(comp_type)
                block_data = decompressor.decompress(block_data, orig_size)

            return block_data

    raise KeyError(f"Block '{block_name}' not found in {path}")


def make_large_episode(rng: np.random.Generator) -> Episode:
    """Build a ~50 MB episode: signal/rgb (21 MB) + 4 depth fillers (7 MB each).

    The target partial-read block is action/ctrl (~28 KB).
    """
    ep = Episode(id="bench_partial_ep", length=T)
    ep.env_id = "BenchEnv-v0"

    rgb = make_rgb(T)
    ep.observations["rgb"] = Channel(
        name="rgb", dtype=DType.UINT8, shape=[84, 84, 3], data=rgb,
    )

    # 4 depth-like filler channels, each [T, 84, 84, 1] uint8
    for i, tag in enumerate(["a", "b", "c", "d"]):
        depth_rng = np.random.default_rng(SEED + 10 + i)
        # Smooth depth-like data: gradient + low noise (±2 LSB), compressible
        base_d = np.linspace(0, 200, 84, dtype=np.float32)[:, None, None]
        base_d = np.broadcast_to(base_d, (84, 84, 1)).copy()
        noise_d = (depth_rng.integers(0, 4, (T, 84, 84, 1)) - 2).astype(np.int16)
        depth = (base_d[None] + noise_d).clip(0, 255).astype(np.uint8)
        ep.observations[f"depth_{tag}"] = Channel(
            name=f"depth_{tag}", dtype=DType.UINT8, shape=[84, 84, 1], data=depth,
        )

    # joint_pos
    ep.observations["joint_pos"] = Channel(
        name="joint_pos", dtype=DType.FLOAT32, shape=[7],
        data=rng.standard_normal((T, 7)).astype(np.float32),
    )

    # action/ctrl — the small block we want to fetch selectively
    ep.actions["ctrl"] = Channel(
        name="ctrl", dtype=DType.FLOAT32, shape=[7],
        data=rng.standard_normal((T, 7)).astype(np.float32),
    )

    ep.rewards = Channel(
        name="reward", dtype=DType.FLOAT32, shape=[],
        data=rng.standard_normal((T,)).astype(np.float32),
    )
    ep.terminations = Channel(
        name="done", dtype=DType.BOOL, shape=[],
        data=(rng.integers(0, 100, (T,)) == 0),
    )
    return ep


def bench_partial_reads(tmpdir: str) -> str:
    """Write a ~50 MB file, then time fetching only action/ctrl (28 KB).

    Returns a markdown table string.
    """
    rng = np.random.default_rng(SEED)
    ep = make_large_episode(rng)

    # Build flat dict for npz and hdf5
    flat: dict[str, np.ndarray] = {}
    flat["joint_pos"] = ep.observations["joint_pos"].data
    flat["rgb"]       = ep.observations["rgb"].data
    for tag in ["a", "b", "c", "d"]:
        flat[f"depth_{tag}"] = ep.observations[f"depth_{tag}"].data
    flat["ctrl"]      = ep.actions["ctrl"].data
    flat["reward"]    = ep.rewards.data
    flat["done"]      = ep.terminations.data.astype(np.uint8)

    wshard_path = os.path.join(tmpdir, "large.wshard")
    npz_path    = os.path.join(tmpdir, "large.npz")

    print("    Writing large wshard (~50 MB)...")
    save_wshard(ep, wshard_path, compression=CompressionType.ZSTD)
    wshard_sz = os.path.getsize(wshard_path)

    print("    Writing large npz (~50 MB)...")
    np.savez_compressed(npz_path, **flat)
    npz_sz = os.path.getsize(npz_path)

    results = []

    # ── WShard partial (index + single block seek) ─────────────────────────────
    times_wshard = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        _ = _wshard_read_block_partial(wshard_path, "action/ctrl")
        times_wshard.append(time.perf_counter() - t0)
    ws_med, ws_min = _median_min(times_wshard)
    results.append(("WShard (zstd)", f"{wshard_sz/1e6:.1f} MB", ws_med, ws_min))

    # ── NPZ partial (np.load lazy — only inflates the requested array) ─────────
    times_npz = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        with np.load(npz_path) as f:
            _ = f["ctrl"]  # lazy: deflates just this member
        times_npz.append(time.perf_counter() - t0)
    npz_med, npz_min = _median_min(times_npz)
    results.append(("NPZ (deflate)", f"{npz_sz/1e6:.1f} MB", npz_med, npz_min))

    # ── HDF5 partial ───────────────────────────────────────────────────────────
    if HAS_H5PY:
        hdf5_path = os.path.join(tmpdir, "large.h5")
        print("    Writing large hdf5 (~50 MB)...")
        with h5py.File(hdf5_path, "w") as f:
            for k, v in flat.items():
                f.create_dataset(k, data=v, compression="gzip", compression_opts=4)
        hdf5_sz = os.path.getsize(hdf5_path)

        times_hdf5 = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            with h5py.File(hdf5_path, "r") as f:
                _ = f["ctrl"][:]
            times_hdf5.append(time.perf_counter() - t0)
        hdf5_med, hdf5_min = _median_min(times_hdf5)
        results.append(("HDF5 (gzip-4)", f"{hdf5_sz/1e6:.1f} MB", hdf5_med, hdf5_min))

    # Format table
    headers = ["Format", "File size", "Median time", "Min time"]
    sep = "|" + "|".join("-" * (len(h) + 2) for h in headers) + "|"
    header_row = "| " + " | ".join(headers) + " |"
    lines = [header_row, sep]
    for (fmt, fsz, med, mn) in results:
        lines.append(f"| {fmt} | {fsz} | {fmt_us(med)} | {fmt_us(mn)} |")
    return "\n".join(lines)


# ── main ───────────────────────────────────────────────────────────────────────

def run_benchmarks() -> tuple[list[dict[str, Any]], int, str]:
    data = make_data()
    uncompressed = raw_bytes(data)
    ep = make_episode(data)

    print(f"Workload: T={T}, raw payload = {fmt_mb(uncompressed)}")
    print(f"Runs per measurement: {RUNS}\n")

    results: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory() as tmpdir:

        # ── WShard none ────────────────────────────────────────────────────────
        print("  [1/5] WShard-none ...")
        p = os.path.join(tmpdir, "ep_none.wshard")
        wm, wmin = bench_wshard_write(ep, CompressionType.NONE, p)
        file_sz = os.path.getsize(p)
        rm, rmin = bench_wshard_read(p)
        results.append(dict(config="wshard-none",  write_med=wm, write_min=wmin,
                            read_med=rm, read_min=rmin,
                            file_bytes=file_sz, raw_bytes=uncompressed))

        # ── WShard zstd ────────────────────────────────────────────────────────
        print("  [2/5] WShard-zstd ...")
        p = os.path.join(tmpdir, "ep_zstd.wshard")
        wm, wmin = bench_wshard_write(ep, CompressionType.ZSTD, p)
        file_sz = os.path.getsize(p)
        rm, rmin = bench_wshard_read(p)
        results.append(dict(config="wshard-zstd", write_med=wm, write_min=wmin,
                            read_med=rm, read_min=rmin,
                            file_bytes=file_sz, raw_bytes=uncompressed))

        # ── WShard lz4 ─────────────────────────────────────────────────────────
        print("  [3/5] WShard-lz4 ...")
        p = os.path.join(tmpdir, "ep_lz4.wshard")
        wm, wmin = bench_wshard_write(ep, CompressionType.LZ4, p)
        file_sz = os.path.getsize(p)
        rm, rmin = bench_wshard_read(p)
        results.append(dict(config="wshard-lz4",  write_med=wm, write_min=wmin,
                            read_med=rm, read_min=rmin,
                            file_bytes=file_sz, raw_bytes=uncompressed))

        # ── NumPy NPZ deflate ──────────────────────────────────────────────────
        print("  [4/5] NumPy NPZ-deflate ...")
        p = os.path.join(tmpdir, "ep.npz")
        wm, wmin = bench_npz_write(data, p)
        if not os.path.exists(p):
            p = p  # savez_compressed doesn't re-append when extension is already .npz
        file_sz = os.path.getsize(p)
        rm, rmin = bench_npz_read(p)
        results.append(dict(config="npz-deflate", write_med=wm, write_min=wmin,
                            read_med=rm, read_min=rmin,
                            file_bytes=file_sz, raw_bytes=uncompressed))

        # ── HDF5 deflate (optional) ────────────────────────────────────────────
        if HAS_H5PY:
            print("  [5/5] HDF5-deflate (gzip-4) ...")
            p = os.path.join(tmpdir, "ep.h5")
            wm, wmin = bench_hdf5_write(data, p)
            file_sz = os.path.getsize(p)
            rm, rmin = bench_hdf5_read(p)
            results.append(dict(config="hdf5-deflate", write_med=wm, write_min=wmin,
                                read_med=rm, read_min=rmin,
                                file_bytes=file_sz, raw_bytes=uncompressed))
        else:
            print("  [5/5] HDF5-deflate — SKIPPED (h5py not installed)")

        # ── Partial-block reads ────────────────────────────────────────────────
        print("\n  [partial] Writing large episode (~50 MB) and benchmarking partial reads...")
        partial_table = bench_partial_reads(tmpdir)

    return results, uncompressed, partial_table


def main() -> None:
    parser = argparse.ArgumentParser(description="WShard vs NumPy NPZ benchmark")
    parser.add_argument("--output", metavar="FILE",
                        help="Write markdown table to FILE in addition to stdout")
    args = parser.parse_args()

    results, uncompressed, partial_table = run_benchmarks()

    print()
    table = build_table(results)
    machine_note = (
        f"\n> Raw payload: {fmt_mb(uncompressed)} "
        f"({T} steps · joint_pos f32[7] + rgb u8[84,84,3] + ctrl f32[7] + reward f32 + done bool)"
    )
    output = table + machine_note

    print(output)
    print()
    print("### Partial-block reads — action/ctrl (28 KB) from ~50 MB file")
    print()
    print(partial_table)

    if args.output:
        full = output + "\n\n### Partial-block reads\n\n" + partial_table + "\n"
        Path(args.output).write_text(full)
        print(f"\nTable written to {args.output}")


if __name__ == "__main__":
    main()
