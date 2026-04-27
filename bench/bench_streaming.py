"""bench_streaming.py — WShardStreamWriter vs batch save_wshard.

Workload: T=10000 step recording.
Channels:
  signal/joint_pos   f32[7]
  action/ctrl        f32[7]
  reward             f32 scalar
  done               bool scalar

Two paths:
  Streaming — WShardStreamWriter, write_timestep() × 10000, end_episode()
  Batch     — build numpy arrays in memory, single save_wshard() call

Metrics:
  Total wall-time (s), per-step overhead (µs), peak memory (MiB via tracemalloc).

Target: per-step streaming overhead < 100 µs for robot control loops.

Seed: 42. No external dependencies beyond wshard stdlib.
"""

from __future__ import annotations

import os
import statistics
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "py"))

from wshard import load_wshard, save_wshard
from wshard.compress import CompressionType
from wshard.streaming import WShardStreamWriter, ChannelDef
from wshard.types import Channel, DType, Episode

SEED = 42
T = 10_000
RUNS = 3  # repeat whole path to get stable median


# ── data synthesis ─────────────────────────────────────────────────────────────

def make_data():
    rng = np.random.default_rng(SEED)
    joint_pos = rng.standard_normal((T, 7)).astype(np.float32)
    ctrl = rng.standard_normal((T, 7)).astype(np.float32)
    reward = rng.uniform(0.0, 1.0, T).astype(np.float32)
    done = np.zeros(T, dtype=np.bool_)
    done[-1] = True
    return joint_pos, ctrl, reward, done


def make_batch_episode(joint_pos, ctrl, reward, done) -> Episode:
    ep = Episode(id="bench_stream_ep", length=T)
    ep.env_id = "BenchBot-v0"
    ep.observations["joint_pos"] = Channel(
        name="joint_pos", dtype=DType.FLOAT32, shape=[7], data=joint_pos
    )
    ep.actions["ctrl"] = Channel(
        name="ctrl", dtype=DType.FLOAT32, shape=[7], data=ctrl
    )
    ep.rewards = Channel(name="reward", dtype=DType.FLOAT32, shape=[], data=reward)
    ep.terminations = Channel(name="done", dtype=DType.BOOL, shape=[], data=done)
    return ep


# ── streaming bench ────────────────────────────────────────────────────────────

def run_streaming(
    joint_pos: np.ndarray,
    ctrl: np.ndarray,
    reward: np.ndarray,
    done: np.ndarray,
    path: str,
) -> float:
    """Run streaming path, return wall-time in seconds."""
    channel_defs = [
        ChannelDef("joint_pos", DType.FLOAT32, [7]),
        ChannelDef("ctrl", DType.FLOAT32, [7]),
    ]

    t0 = time.perf_counter()
    writer = WShardStreamWriter(path, "bench_stream_ep", channel_defs)
    writer.begin_episode(env_id="BenchBot-v0")

    for t in range(T):
        writer.write_timestep(
            t=t,
            observations={"joint_pos": joint_pos[t]},
            actions={"ctrl": ctrl[t]},
            reward=float(reward[t]),
            done=bool(done[t]),
        )

    writer.end_episode()
    return time.perf_counter() - t0


# ── batch bench ────────────────────────────────────────────────────────────────

def run_batch(ep: Episode, path: str) -> float:
    """Run batch path, return wall-time in seconds."""
    t0 = time.perf_counter()
    save_wshard(ep, path, compression=CompressionType.NONE)
    return time.perf_counter() - t0


# ── memory helpers ─────────────────────────────────────────────────────────────

def measure_peak_memory(fn) -> tuple[float, int]:
    """Run fn() under tracemalloc, return (elapsed_s, peak_bytes)."""
    tracemalloc.start()
    t0 = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed, peak


# ── formatting ─────────────────────────────────────────────────────────────────

def fmt_s(v: float) -> str:
    if v < 1.0:
        return f"{v*1e3:.1f} ms"
    return f"{v:.3f} s"


def fmt_us(v: float) -> str:
    return f"{v*1e6:.2f} µs"


def fmt_mb(b: int) -> str:
    return f"{b / 1_048_576:.2f} MiB"


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("bench_streaming.py — Streaming vs Batch write (T=10000)")
    print(f"Seed: {SEED}, T: {T:,}, runs: {RUNS}")
    print("=" * 60)

    joint_pos, ctrl, reward, done = make_data()
    ep = make_batch_episode(joint_pos, ctrl, reward, done)

    streaming_times = []
    streaming_peak = None
    batch_times = []
    batch_peak = None

    with tempfile.TemporaryDirectory() as tmpdir:
        stream_path = os.path.join(tmpdir, "stream.wshard")
        batch_path = os.path.join(tmpdir, "batch.wshard")

        # Streaming runs
        print(f"\n  [Streaming] {RUNS} runs ...")
        for i in range(RUNS):
            elapsed, peak = measure_peak_memory(
                lambda p=stream_path: run_streaming(joint_pos, ctrl, reward, done, p)
            )
            streaming_times.append(elapsed)
            if i == 0:
                streaming_peak = peak
            print(f"    run {i+1}: {fmt_s(elapsed)} ({fmt_mb(peak)} peak)")

        stream_size = os.path.getsize(stream_path)

        # Verify streaming output is readable
        ep_stream = load_wshard(stream_path)
        assert ep_stream.length == T, f"Streaming: length mismatch {ep_stream.length} != {T}"
        print(f"  [Streaming] round-trip OK: T={ep_stream.length}, file={stream_size:,} bytes")

        # Batch runs
        print(f"\n  [Batch] {RUNS} runs ...")
        for i in range(RUNS):
            elapsed, peak = measure_peak_memory(lambda p=batch_path: run_batch(ep, p))
            batch_times.append(elapsed)
            if i == 0:
                batch_peak = peak
            print(f"    run {i+1}: {fmt_s(elapsed)} ({fmt_mb(peak)} peak)")

        batch_size = os.path.getsize(batch_path)

        # Verify batch output
        ep_batch = load_wshard(batch_path)
        assert ep_batch.length == T, f"Batch: length mismatch"
        print(f"  [Batch] round-trip OK: T={ep_batch.length}, file={batch_size:,} bytes")

    stream_med = statistics.median(streaming_times)
    stream_min = min(streaming_times)
    batch_med = statistics.median(batch_times)
    batch_min = min(batch_times)
    overhead_per_step_med = stream_med / T
    overhead_per_step_min = stream_min / T

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    headers = ["Path", "Total (med)", "Total (min)", "Per-step (med)", "Peak mem", "File size"]
    sep = "|" + "|".join("-" * (len(h) + 2) for h in headers) + "|"
    print("| " + " | ".join(headers) + " |")
    print(sep)
    print(
        "| Streaming | "
        f"{fmt_s(stream_med)} | "
        f"{fmt_s(stream_min)} | "
        f"{fmt_us(overhead_per_step_med)} | "
        f"{fmt_mb(streaming_peak)} | "
        f"{stream_size / 1_048_576:.2f} MiB |"
    )
    print(
        "| Batch     | "
        f"{fmt_s(batch_med)} | "
        f"{fmt_s(batch_min)} | "
        f"{fmt_us(batch_med / T)} | "
        f"{fmt_mb(batch_peak)} | "
        f"{batch_size / 1_048_576:.2f} MiB |"
    )

    target_us = 100.0
    status = "PASS" if overhead_per_step_med * 1e6 < target_us else "FAIL"
    print()
    print(f"Per-step overhead target: < {target_us} µs  →  {status}")
    print(f"  Streaming median per-step: {overhead_per_step_med*1e6:.2f} µs")

    print()
    print("Notes:")
    print(f"  - T={T:,} steps, channels: joint_pos f32[7], ctrl f32[7], reward f32, done bool")
    print("  - Streaming uses CompressionType.NONE (default for real-time path).")
    print("  - Batch uses CompressionType.NONE for a fair size/timing comparison.")
    print("  - Peak memory measured via tracemalloc (stdlib) on first run only.")
    print("  - Per-step = total_wall_time / T (amortised, not per-call instrumented).")


if __name__ == "__main__":
    main()
