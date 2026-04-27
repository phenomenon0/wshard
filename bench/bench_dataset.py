"""bench_dataset.py — Many-files scaling: 1000 small episodes.

Validates:
  (a) Total disk usage for 1000 episodes.
  (b) Sequential full-read: load_wshard for all 1000.
  (c) Header+index-only pass: open each file, parse header+index, close.
  (d) NPZ comparison: same payload as .npz per episode.

Episode schema (T=100, ~2 KB raw payload):
  observations  f32[100, 4]   — CartPole-like (pos, vel, angle, ang_vel)
  actions       f32[100, 1]   — force scalar
  reward        f32[100]
  done          bool[100]

Seed: 42 for determinism. No external dependencies.
"""

from __future__ import annotations

import io
import os
import statistics
import struct
import tempfile
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "py"))

from wshard import load_wshard, save_wshard
from wshard.compress import CompressionType
from wshard.types import Channel, DType, Episode
from wshard.wshard import MAGIC, VERSION, ROLE_WSHARD, HEADER_SIZE, INDEX_ENTRY_SIZE

SEED = 42
N_EPISODES = 1000
T = 100
OBS_DIM = 4
ACT_DIM = 1


# ── episode generation ─────────────────────────────────────────────────────────

def make_episode(ep_idx: int, rng: np.random.Generator) -> tuple[Episode, dict]:
    """Make one small CartPole-style episode. Returns (Episode, raw_arrays_dict)."""
    obs = rng.standard_normal((T, OBS_DIM)).astype(np.float32)
    act = rng.standard_normal((T, ACT_DIM)).astype(np.float32)
    rew = rng.uniform(0.0, 1.0, T).astype(np.float32)
    done = np.zeros(T, dtype=np.bool_)
    done[-1] = True

    ep = Episode(id=f"ep_{ep_idx:04d}", length=T)
    ep.env_id = "CartPole-v1"
    ep.observations["obs"] = Channel(
        name="obs", dtype=DType.FLOAT32, shape=[OBS_DIM], data=obs
    )
    ep.actions["action"] = Channel(
        name="action", dtype=DType.FLOAT32, shape=[ACT_DIM], data=act
    )
    ep.rewards = Channel(name="reward", dtype=DType.FLOAT32, shape=[], data=rew)
    ep.terminations = Channel(name="done", dtype=DType.BOOL, shape=[], data=done)

    raw = dict(obs=obs, action=act, reward=rew, done=done)
    return ep, raw


# ── header+index-only parser ───────────────────────────────────────────────────

def parse_header_and_index(path: str) -> dict:
    """Open wshard file, parse header and index without reading data blocks.
    Returns dict with entry_count and block name list."""
    with open(path, "rb") as f:
        header = f.read(HEADER_SIZE)
        if len(header) < HEADER_SIZE:
            raise ValueError("Header too short")
        if header[:4] != MAGIC:
            raise ValueError(f"Bad magic: {header[:4]!r}")

        entry_count = struct.unpack("<I", header[12:16])[0]
        string_table_offset = struct.unpack("<Q", header[16:24])[0]
        data_section_offset = struct.unpack("<Q", header[24:32])[0]

        # Read index
        index_data = f.read(entry_count * INDEX_ENTRY_SIZE)
        entries = []
        for i in range(entry_count):
            off = i * INDEX_ENTRY_SIZE
            entry = {
                "name_offset": struct.unpack("<I", index_data[off + 8:off + 12])[0],
                "name_len":    struct.unpack("<H", index_data[off + 12:off + 14])[0],
            }
            entries.append(entry)

        # Read string table
        string_table_size = data_section_offset - string_table_offset
        f.seek(string_table_offset)
        string_table = f.read(string_table_size)

        names = []
        for entry in entries:
            no = entry["name_offset"]
            nl = entry["name_len"]
            names.append(string_table[no:no + nl].decode("utf-8"))

    return {"entry_count": entry_count, "block_names": names}


# ── formatting ─────────────────────────────────────────────────────────────────

def fmt_s(v: float) -> str:
    if v < 0.001:
        return f"{v*1e6:.1f} µs"
    if v < 1.0:
        return f"{v*1e3:.2f} ms"
    return f"{v:.3f} s"


def fmt_us(v: float) -> str:
    return f"{v*1e6:.1f} µs"


def fmt_mb(b: int) -> str:
    return f"{b / 1_048_576:.3f} MB"


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print(f"bench_dataset.py — Many-files scaling ({N_EPISODES} episodes, T={T})")
    print(f"Schema: obs f32[{OBS_DIM}], action f32[{ACT_DIM}], reward f32, done bool")
    print(f"Seed: {SEED}")
    print("=" * 60)

    rng = np.random.default_rng(SEED)

    with tempfile.TemporaryDirectory() as tmpdir:
        wshard_dir = os.path.join(tmpdir, "wshard")
        npz_dir = os.path.join(tmpdir, "npz")
        os.makedirs(wshard_dir)
        os.makedirs(npz_dir)

        # ── write all episodes ─────────────────────────────────────────────────
        print(f"\nWriting {N_EPISODES} wshard + npz episodes ...")
        wshard_paths = []
        npz_paths = []
        t_write_start = time.perf_counter()
        for i in range(N_EPISODES):
            ep, raw = make_episode(i, rng)
            wp = os.path.join(wshard_dir, f"ep_{i:04d}.wshard")
            np_path = os.path.join(npz_dir, f"ep_{i:04d}.npz")
            save_wshard(ep, wp, compression=CompressionType.ZSTD)
            np.savez_compressed(np_path, **raw)
            wshard_paths.append(wp)
            npz_paths.append(np_path)
        t_write_total = time.perf_counter() - t_write_start
        print(f"  done in {t_write_total:.2f}s ({t_write_total/N_EPISODES*1e3:.2f} ms/ep)")

        # ── disk usage ────────────────────────────────────────────────────────
        wshard_total_bytes = sum(os.path.getsize(p) for p in wshard_paths)
        npz_total_bytes = sum(os.path.getsize(p) for p in npz_paths)
        raw_bytes_per_ep = (
            T * OBS_DIM * 4   # obs f32
            + T * ACT_DIM * 4  # action f32
            + T * 4            # reward f32
            + T * 1            # done bool
        )
        raw_total_bytes = raw_bytes_per_ep * N_EPISODES
        wshard_per_ep = wshard_total_bytes / N_EPISODES
        npz_per_ep = npz_total_bytes / N_EPISODES
        print(f"\nDisk usage ({N_EPISODES} episodes):")
        print(f"  Raw total         : {fmt_mb(raw_total_bytes)}")
        print(f"  WShard-zstd total : {fmt_mb(wshard_total_bytes)}  ({wshard_per_ep:.0f} bytes/ep)")
        print(f"  NPZ-deflate total : {fmt_mb(npz_total_bytes)}  ({npz_per_ep:.0f} bytes/ep)")

        # ── sequential full-read ──────────────────────────────────────────────
        print(f"\nSequential full-read ({N_EPISODES} load_wshard calls) ...")
        t0 = time.perf_counter()
        for p in wshard_paths:
            _ = load_wshard(p)
        t_full_read = time.perf_counter() - t0
        per_ep_full = t_full_read / N_EPISODES
        print(f"  Total: {t_full_read:.3f}s  Per-episode: {fmt_us(per_ep_full)}")

        # ── header+index-only pass ────────────────────────────────────────────
        print(f"\nHeader+index-only pass ({N_EPISODES} files) ...")
        t0 = time.perf_counter()
        for p in wshard_paths:
            _ = parse_header_and_index(p)
        t_index_pass = time.perf_counter() - t0
        per_ep_index = t_index_pass / N_EPISODES
        print(f"  Total: {t_index_pass:.3f}s  Per-episode: {fmt_us(per_ep_index)}")

        # ── NPZ sequential full-read ──────────────────────────────────────────
        print(f"\nNPZ sequential full-read ({N_EPISODES} np.load calls) ...")
        t0 = time.perf_counter()
        for p in npz_paths:
            with np.load(p) as f:
                _ = {k: f[k] for k in f.files}
        t_npz_read = time.perf_counter() - t0
        per_ep_npz = t_npz_read / N_EPISODES
        print(f"  Total: {t_npz_read:.3f}s  Per-episode: {fmt_us(per_ep_npz)}")

        # ── verification (spot-check first and last) ──────────────────────────
        rng2 = np.random.default_rng(SEED)
        for check_idx in [0, N_EPISODES - 1]:
            ep_check, raw_check = make_episode(check_idx, rng2 if check_idx == 0 else rng2)
            ep_loaded = load_wshard(wshard_paths[check_idx])
            assert ep_loaded.length == T, f"Length mismatch for ep {check_idx}"
        print("\n  Spot-check: episodes 0 and 999 load OK")

    # ── summary table ─────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    headers = ["Metric", "WShard-zstd", "NPZ-deflate"]
    sep = "|" + "|".join("-" * (len(h) + 2) for h in headers) + "|"
    print("| " + " | ".join(headers) + " |")
    print(sep)
    rows = [
        ("Total disk (1000 eps)", fmt_mb(wshard_total_bytes), fmt_mb(npz_total_bytes)),
        ("Bytes/episode",
         f"{wshard_per_ep:.0f} B",
         f"{npz_per_ep:.0f} B"),
        ("Full-read per-episode", fmt_us(per_ep_full), fmt_us(per_ep_npz)),
        ("Index-only per-episode", fmt_us(per_ep_index), "n/a"),
        ("Full-read total (1000)", f"{t_full_read:.3f} s", f"{t_npz_read:.3f} s"),
    ]
    for cells in rows:
        print("| " + " | ".join(str(c) for c in cells) + " |")

    print()
    print("Notes:")
    print(f"  - {N_EPISODES} episodes × T={T} steps × obs f32[{OBS_DIM}] + action f32[{ACT_DIM}]"
          f" + reward f32 + done bool")
    print(f"  - Raw payload per episode: {raw_bytes_per_ep:,} bytes ({raw_bytes_per_ep/1024:.1f} KB)")
    print("  - Header+index-only: reads HEADER_SIZE + entry_count*INDEX_ENTRY_SIZE + string table.")
    print("  - Useful for dataset enumeration / channel listing without decoding tensors.")
    print("  - WShard-zstd compression is less effective at this tiny payload size.")
    print(f"  - Compression ratio wshard: {raw_total_bytes/wshard_total_bytes:.2f}x  "
          f"npz: {raw_total_bytes/npz_total_bytes:.2f}x")


if __name__ == "__main__":
    main()
