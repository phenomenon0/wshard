"""bench_parquet.py — WShard vs Parquet (LeRobot-style flat encoding).

Workload: same footprint as bench_python.py (T=1000, ~21 MB raw).
  signal/joint_pos   f32[1000, 7]
  signal/rgb         u8[1000, 84, 84, 3]  — vertical gradient + low-amp noise
  action/ctrl        f32[1000, 7]
  reward             f32[1000]
  done               bool[1000]

Parquet encoding: flat row-per-timestep table (LeRobot's actual schema shape).
  Columns: joint_pos_0..6, ctrl_0..6, reward, done, rgb_bytes (binary blob per row).
  Compression: zstd per column (parquet default).

WShard encoding: wshard-none, wshard-zstd (for direct comparison).

Soft dep: pyarrow. Script prints a skip message and exits gracefully if absent.

Seed: 42. No pip installs.
"""

from __future__ import annotations

import os
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "py"))

SEED = 42
T = 1000
RUNS = 5


# ── pyarrow soft-import ────────────────────────────────────────────────────────

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False


# ── wshard imports ─────────────────────────────────────────────────────────────

from wshard import load_wshard, save_wshard
from wshard.compress import CompressionType
from wshard.types import Channel, DType, Episode


# ── data synthesis ─────────────────────────────────────────────────────────────

def make_data() -> dict[str, np.ndarray]:
    """Structured RGB (vertical gradient + noise) matching bench_python.py workload."""
    rng = np.random.default_rng(SEED)
    H, W, C = 84, 84, 3

    # Smooth vertical gradient 0-255, replicated across T
    gradient = np.linspace(0, 255, H, dtype=np.float32)   # [H]
    rgb_base = np.broadcast_to(gradient[:, None, None], (H, W, C))   # [H,W,C]
    rgb_base = np.broadcast_to(rgb_base[None], (T, H, W, C)).copy()  # [T,H,W,C]
    noise = rng.integers(-10, 11, (T, H, W, C), dtype=np.int16).astype(np.float32)
    rgb = np.clip(rgb_base + noise, 0, 255).astype(np.uint8)

    return {
        "joint_pos": rng.standard_normal((T, 7)).astype(np.float32),
        "rgb":       rgb,
        "ctrl":      rng.standard_normal((T, 7)).astype(np.float32),
        "reward":    rng.standard_normal(T).astype(np.float32),
        "done":      (rng.integers(0, 100, T) == 0),
    }


def raw_bytes(data: dict[str, np.ndarray]) -> int:
    return sum(a.nbytes for a in data.values())


def make_episode(data: dict[str, np.ndarray]) -> Episode:
    ep = Episode(id="bench_parquet_ep", length=T)
    ep.env_id = "BenchEnv-v0"
    ep.observations["joint_pos"] = Channel(
        name="joint_pos", dtype=DType.FLOAT32, shape=[7], data=data["joint_pos"]
    )
    ep.observations["rgb"] = Channel(
        name="rgb", dtype=DType.UINT8, shape=[84, 84, 3], data=data["rgb"]
    )
    ep.actions["ctrl"] = Channel(
        name="ctrl", dtype=DType.FLOAT32, shape=[7], data=data["ctrl"]
    )
    ep.rewards = Channel(name="reward", dtype=DType.FLOAT32, shape=[], data=data["reward"])
    ep.terminations = Channel(name="done", dtype=DType.BOOL, shape=[], data=data["done"])
    return ep


# ── parquet encoding/decoding ─────────────────────────────────────────────────

def data_to_parquet_table(data: dict[str, np.ndarray]) -> pa.Table:
    """Convert episode data to a flat Parquet table (LeRobot-style).
    Each timestep is a row.  Multi-dim channels are column-expanded or binary.
    """
    columns = {}

    # Expand joint_pos into 7 scalar columns
    for i in range(7):
        columns[f"joint_pos_{i}"] = data["joint_pos"][:, i]

    # Expand ctrl into 7 scalar columns
    for i in range(7):
        columns[f"ctrl_{i}"] = data["ctrl"][:, i]

    columns["reward"] = data["reward"]
    columns["done"] = data["done"]

    # RGB: serialize each frame as a binary blob (most natural for Parquet)
    rgb_flat = data["rgb"].reshape(T, -1)  # [T, 84*84*3]
    columns["rgb_bytes"] = pa.array(
        [row.tobytes() for row in rgb_flat],
        type=pa.large_binary(),
    )

    return pa.table(columns)


def parquet_table_to_arrays(table: pa.Table) -> dict[str, np.ndarray]:
    """Round-trip: read flat Parquet table back to numpy arrays."""
    joint_pos = np.stack(
        [table.column(f"joint_pos_{i}").to_pylist() for i in range(7)], axis=1
    ).astype(np.float32)
    ctrl = np.stack(
        [table.column(f"ctrl_{i}").to_pylist() for i in range(7)], axis=1
    ).astype(np.float32)
    reward = np.array(table.column("reward").to_pylist(), dtype=np.float32)
    done = np.array(table.column("done").to_pylist(), dtype=np.bool_)
    rgb_list = table.column("rgb_bytes").to_pylist()
    rgb = np.frombuffer(b"".join(rgb_list), dtype=np.uint8).reshape(T, 84, 84, 3)
    return dict(joint_pos=joint_pos, ctrl=ctrl, reward=reward, done=done, rgb=rgb)


# ── timing helpers ─────────────────────────────────────────────────────────────

def median_min(fn, runs: int = RUNS) -> tuple[float, float]:
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), min(times)


def fmt_s(v: float) -> str:
    if v < 0.001:
        return f"{v*1e3:.2f} ms"
    if v < 1.0:
        return f"{v*1e3:.1f} ms"
    return f"{v:.3f} s"


def fmt_mb(b: int) -> str:
    return f"{b / 1_048_576:.2f} MB"


def fmt_ratio(raw: int, disk: int) -> str:
    if disk == 0:
        return "n/a"
    return f"{raw / disk:.2f}x"


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("bench_parquet.py — WShard vs Parquet (LeRobot-style flat)")
    print(f"Seed: {SEED}, T={T}, runs={RUNS}")
    print("=" * 60)

    if not HAS_PARQUET:
        print()
        print("SKIP: pyarrow not installed. Install with: pip install pyarrow")
        print("      Parquet comparison requires pyarrow >= 10.0.")
        return

    data = make_data()
    total_raw = raw_bytes(data)
    ep = make_episode(data)

    print(f"\nWorkload: T={T}, raw payload = {fmt_mb(total_raw)}")

    results = []

    with tempfile.TemporaryDirectory() as tmpdir:

        # ── WShard-none ────────────────────────────────────────────────────────
        print("  [1/4] WShard-none ...")
        p = os.path.join(tmpdir, "ep_none.wshard")
        wm, wmin = median_min(lambda: save_wshard(ep, p, compression=CompressionType.NONE))
        wsz = os.path.getsize(p)
        rm, rmin = median_min(lambda: load_wshard(p))
        results.append(dict(
            config="wshard-none", write_med=wm, write_min=wmin,
            read_med=rm, read_min=rmin, file_bytes=wsz, raw_bytes=total_raw,
        ))

        # ── WShard-zstd ────────────────────────────────────────────────────────
        print("  [2/4] WShard-zstd ...")
        p = os.path.join(tmpdir, "ep_zstd.wshard")
        wm, wmin = median_min(lambda: save_wshard(ep, p, compression=CompressionType.ZSTD))
        wsz = os.path.getsize(p)
        rm, rmin = median_min(lambda: load_wshard(p))
        results.append(dict(
            config="wshard-zstd", write_med=wm, write_min=wmin,
            read_med=rm, read_min=rmin, file_bytes=wsz, raw_bytes=total_raw,
        ))

        # ── Parquet-zstd (flat) ────────────────────────────────────────────────
        print("  [3/4] Parquet-zstd (flat, LeRobot-style) ...")
        pq_path = os.path.join(tmpdir, "ep.parquet")
        table = data_to_parquet_table(data)

        def parquet_write():
            pq.write_table(
                table,
                pq_path,
                compression="zstd",
                use_dictionary=False,
            )

        def parquet_read():
            t2 = pq.read_table(pq_path)
            # Force materialisation of all columns
            return t2.to_pydict()

        wm_pq, wmin_pq = median_min(parquet_write)
        pq_size = os.path.getsize(pq_path)
        rm_pq, rmin_pq = median_min(parquet_read)
        results.append(dict(
            config="parquet-zstd", write_med=wm_pq, write_min=wmin_pq,
            read_med=rm_pq, read_min=rmin_pq, file_bytes=pq_size, raw_bytes=total_raw,
        ))

        # ── Parquet-none (flat, uncompressed) ─────────────────────────────────
        print("  [4/4] Parquet-none (flat, uncompressed) ...")
        pq_nocomp_path = os.path.join(tmpdir, "ep_nocomp.parquet")

        def parquet_write_none():
            pq.write_table(
                table,
                pq_nocomp_path,
                compression="none",
                use_dictionary=False,
            )

        def parquet_read_none():
            return pq.read_table(pq_nocomp_path).to_pydict()

        wm_pqn, wmin_pqn = median_min(parquet_write_none)
        pq_none_size = os.path.getsize(pq_nocomp_path)
        rm_pqn, rmin_pqn = median_min(parquet_read_none)
        results.append(dict(
            config="parquet-none", write_med=wm_pqn, write_min=wmin_pqn,
            read_med=rm_pqn, read_min=rmin_pqn, file_bytes=pq_none_size, raw_bytes=total_raw,
        ))

        # ── Parquet round-trip verification ───────────────────────────────────
        pq.write_table(table, pq_path, compression="zstd", use_dictionary=False)
        recovered = parquet_table_to_arrays(pq.read_table(pq_path))
        assert np.array_equal(data["rgb"], recovered["rgb"]), "Parquet: rgb round-trip failed"
        assert np.allclose(data["reward"], recovered["reward"]), "Parquet: reward mismatch"
        print("  [Parquet] round-trip OK")

    # ── table output ──────────────────────────────────────────────────────────
    print()
    headers = ["Config", "Write (med)", "Write (min)", "Read (med)", "Read (min)",
               "File size", "Ratio"]
    sep = "|" + "|".join("-" * (len(h) + 2) for h in headers) + "|"
    print("| " + " | ".join(headers) + " |")
    print(sep)
    for r in results:
        cells = [
            r["config"],
            fmt_s(r["write_med"]),
            fmt_s(r["write_min"]),
            fmt_s(r["read_med"]),
            fmt_s(r["read_min"]),
            fmt_mb(r["file_bytes"]),
            fmt_ratio(r["raw_bytes"], r["file_bytes"]),
        ]
        print("| " + " | ".join(cells) + " |")

    print()
    print(f"> Raw payload: {fmt_mb(total_raw)}")
    print(f"> T={T}, joint_pos f32[7] + rgb u8[84,84,3] (gradient+noise) + ctrl f32[7] + reward f32 + done bool")
    print("> Parquet encoding: flat row-per-timestep; rgb stored as binary blob per row.")
    print("> pyarrow", pa.__version__)

    print()
    print("Notes:")
    print("  - Parquet is columnar/row-oriented, not natively tensor-shaped.")
    print("  - RGB serialised as bytes blob — Parquet cannot natively store nd-arrays.")
    print("  - wshard stores raw tensor bytes with optional per-block compression.")
    print("  - Read time includes full column materialisation for Parquet.")


if __name__ == "__main__":
    main()
