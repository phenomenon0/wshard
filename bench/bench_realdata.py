"""bench_realdata.py — WShard on realistic RL schemas.

Schemas (synthesized, deterministic seed=42 — NO external downloads):

DreamerV3-style episode (T=200):
  image       u8[200, 64, 64, 3]  — gradient + noise, smooth-scene proxy (~2.46 MB raw)
  action      f32[200, 6]
  reward      f32[200]
  is_first    bool[200]
  is_last     bool[200]
  is_terminal bool[200]

D4RL Hopper-v2-style episode (T=1000):
  observations  f32[1000, 17]   Hopper-v2 proprioceptive observation
  actions       f32[1000, 6]
  rewards       f32[1000]
  terminals     bool[1000]

Methodology:
  - DreamerV3: write synthetic NPZ first, convert via load_dreamer, save_wshard(zstd).
  - D4RL:      build Episode directly (no D4RL converter), save_wshard(zstd).
  - 3 write/read repetitions for timing stability.
  - Verify no data loss on round-trip (numpy array_equal / tobytes match).
  - Print markdown summary table.
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

from wshard import load_wshard, save_wshard, load_dreamer
from wshard.compress import CompressionType
from wshard.types import Channel, DType, Episode

SEED = 42
RUNS = 3


# ── data synthesis ─────────────────────────────────────────────────────────────

def make_dreamer_npz(path: str) -> dict:
    """Synthesize a DreamerV3-style NPZ and write it to path.
    Image is a smooth vertical gradient + low-amplitude noise (better gzip proxy).
    Returns the dict of raw arrays for later comparison."""
    rng = np.random.default_rng(SEED)
    T = 200
    H, W, C = 64, 64, 3

    # Smooth gradient: row index ramps 0-255 across height, replicated across time
    gradient = np.linspace(0, 255, H, dtype=np.float32)  # [H]
    image_base = np.broadcast_to(gradient[:, None, None], (H, W, C))  # [H,W,C]
    image_base = np.broadcast_to(image_base[None], (T, H, W, C)).copy()  # [T,H,W,C]
    noise = rng.integers(-15, 16, (T, H, W, C), dtype=np.int16).astype(np.float32)
    image = np.clip(image_base + noise, 0, 255).astype(np.uint8)

    action = rng.standard_normal((T, 6)).astype(np.float32)
    reward = rng.standard_normal(T).astype(np.float32)
    is_terminal = np.zeros(T, dtype=np.bool_)
    is_terminal[-1] = True
    is_last = is_terminal.copy()
    is_first = np.zeros(T, dtype=np.bool_)
    is_first[0] = True

    arrays = dict(
        image=image,
        action=action,
        reward=reward,
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
    )
    np.savez(path, **arrays)
    return arrays


def make_d4rl_episode() -> tuple[Episode, dict]:
    """Build a D4RL Hopper-v2-style Episode directly (T=1000, obs_dim=17).
    Returns (episode, raw_arrays_dict)."""
    rng = np.random.default_rng(SEED)
    T = 1000
    obs_dim = 17
    act_dim = 6

    obs = rng.standard_normal((T, obs_dim)).astype(np.float32)
    act = rng.standard_normal((T, act_dim)).astype(np.float32)
    rew = rng.standard_normal(T).astype(np.float32)
    terminals = np.zeros(T, dtype=np.bool_)
    terminals[-1] = True

    ep = Episode(id="d4rl_hopper_bench", length=T)
    ep.env_id = "Hopper-v2"
    ep.observations["observations"] = Channel(
        name="observations", dtype=DType.FLOAT32, shape=[obs_dim], data=obs
    )
    ep.actions["actions"] = Channel(
        name="actions", dtype=DType.FLOAT32, shape=[act_dim], data=act
    )
    ep.rewards = Channel(name="reward", dtype=DType.FLOAT32, shape=[], data=rew)
    ep.terminations = Channel(name="terminals", dtype=DType.BOOL, shape=[], data=terminals)

    raw = dict(observations=obs, actions=act, rewards=rew, terminals=terminals)
    return ep, raw


# ── timing helpers ─────────────────────────────────────────────────────────────

def timed(fn, runs: int = RUNS) -> tuple[float, float]:
    """Return (median_s, min_s) over `runs` calls."""
    times = [_run(fn) for _ in range(runs)]
    return statistics.median(times), min(times)


def _run(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


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


# ── DreamerV3 benchmark ────────────────────────────────────────────────────────

def bench_dreamer(tmpdir: str) -> dict:
    npz_path = os.path.join(tmpdir, "dreamer.npz")
    wshard_path = os.path.join(tmpdir, "dreamer.wshard")

    print("  [DreamerV3] synthesising NPZ...")
    raw_arrays = make_dreamer_npz(npz_path)
    npz_size = os.path.getsize(npz_path)

    raw_bytes = sum(a.nbytes for a in raw_arrays.values())

    # --- write: load_dreamer + save_wshard(zstd) ---
    def write_fn():
        ep = load_dreamer(npz_path)
        save_wshard(ep, wshard_path, compression=CompressionType.ZSTD)

    print("  [DreamerV3] timing wshard-zstd write ...")
    w_med, w_min = timed(write_fn)
    wshard_size = os.path.getsize(wshard_path)

    # --- read ---
    def read_fn():
        return load_wshard(wshard_path)

    print("  [DreamerV3] timing wshard-zstd read ...")
    r_med, r_min = timed(read_fn)

    # --- verification ---
    ep_orig = load_dreamer(npz_path)
    ep_reload = load_wshard(wshard_path)
    assert ep_reload.length == ep_orig.length, "DreamerV3: length mismatch"
    assert np.array_equal(
        ep_orig.observations["image"].data,
        ep_reload.observations["image"].data,
    ), "DreamerV3: image data mismatch"
    assert np.array_equal(
        ep_orig.rewards.data,
        ep_reload.rewards.data,
    ), "DreamerV3: reward data mismatch"
    print("  [DreamerV3] round-trip OK")

    return dict(
        schema="DreamerV3 (T=200)",
        raw_bytes=raw_bytes,
        npz_size=npz_size,
        wshard_size=wshard_size,
        write_med=w_med,
        write_min=w_min,
        read_med=r_med,
        read_min=r_min,
    )


# ── D4RL benchmark ─────────────────────────────────────────────────────────────

def bench_d4rl(tmpdir: str) -> dict:
    wshard_path = os.path.join(tmpdir, "d4rl_hopper.wshard")

    print("  [D4RL] building episode in memory...")
    ep, raw_arrays = make_d4rl_episode()
    raw_bytes = sum(a.nbytes for a in raw_arrays.values())

    # --- write ---
    def write_fn():
        save_wshard(ep, wshard_path, compression=CompressionType.ZSTD)

    print("  [D4RL] timing wshard-zstd write ...")
    w_med, w_min = timed(write_fn)
    wshard_size = os.path.getsize(wshard_path)

    # --- read ---
    def read_fn():
        return load_wshard(wshard_path)

    print("  [D4RL] timing wshard-zstd read ...")
    r_med, r_min = timed(read_fn)

    # --- verification ---
    ep_reload = load_wshard(wshard_path)
    assert ep_reload.length == ep.length, "D4RL: length mismatch"
    assert np.array_equal(
        ep.observations["observations"].data,
        ep_reload.observations["observations"].data,
    ), "D4RL: observation data mismatch"
    assert np.allclose(ep.rewards.data, ep_reload.rewards.data), "D4RL: reward data mismatch"
    print("  [D4RL] round-trip OK")

    return dict(
        schema="D4RL Hopper-v2 (T=1000)",
        raw_bytes=raw_bytes,
        npz_size=None,  # no D4RL native baseline
        wshard_size=wshard_size,
        write_med=w_med,
        write_min=w_min,
        read_med=r_med,
        read_min=r_min,
    )


# ── table formatting ───────────────────────────────────────────────────────────

def build_table(rows: list[dict]) -> str:
    headers = [
        "Schema",
        "Raw size",
        "NPZ size",
        "WShard-zstd",
        "Ratio vs raw",
        "Write (med)",
        "Read (med)",
    ]
    sep = "|" + "|".join("-" * (len(h) + 2) for h in headers) + "|"
    header_row = "| " + " | ".join(headers) + " |"
    lines = [header_row, sep]
    for r in rows:
        npz_cell = fmt_mb(r["npz_size"]) if r["npz_size"] is not None else "n/a"
        cells = [
            r["schema"],
            fmt_mb(r["raw_bytes"]),
            npz_cell,
            fmt_mb(r["wshard_size"]),
            fmt_ratio(r["raw_bytes"], r["wshard_size"]),
            fmt_s(r["write_med"]),
            fmt_s(r["read_med"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("bench_realdata.py — Realistic RL schema benchmarks")
    print(f"Seed: {SEED}, runs per timing: {RUNS}")
    print("=" * 60)

    results = []
    with tempfile.TemporaryDirectory() as tmpdir:
        results.append(bench_dreamer(tmpdir))
        results.append(bench_d4rl(tmpdir))

    print()
    table = build_table(results)
    print(table)

    print()
    print("Notes:")
    print("  - DreamerV3 image: vertical gradient [0-255] + noise in [-15,+15] per channel")
    print("  - NPZ size uses np.savez (uncompressed zip); wshard uses zstd.")
    print("  - D4RL has no native baseline (npz=n/a); wshard-zstd size reported only.")
    print("  - Write time includes load_dreamer conversion for DreamerV3.")
    print("  - Verification: array_equal on image/obs; allclose on rewards.")

    return results


if __name__ == "__main__":
    main()
