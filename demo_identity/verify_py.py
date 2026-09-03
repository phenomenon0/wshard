#!/usr/bin/env python3
"""verify_py.py — companion verifier for the W-SHARD meta/identity dogfood demo.

Mirrors the synthetic fabric episode built in main.go, then proves, from the
Python side:

  A1 encoding-agnosticism: one logical episode saved as none / zstd / lz4 has
     byte-identical episode_identity, and each file passes verify_identity();
  A2 tamper: one flipped byte inside signal/state with its CRC32C patched —
     the container loads fine (CRC cannot tell), verify_identity() raises and
     names signal/state;
  A3 cross-language: Python reads the Go-written file and derives the same
     64-hex identity that Go computed.

Writes machine-readable results to --out (JSON) for main.go to assert on.
Runs against the dirty working tree of ../py (sys.path) — no install needed.
"""

import argparse
import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "py"))

import numpy as np  # noqa: E402
import crc32c  # noqa: E402

from wshard import (  # noqa: E402
    Channel,
    CompressionType,
    DType,
    Episode,
    TimebaseSpec,
    TimebaseType,
    episode_identity,
    save_wshard,
    verify_identity,
)

# Mirrors the constants in main.go — change both or the demo lies.
EP_ID = "demo-0001"
ENV_ID = "fabric/sandbox"
T = 128
STATE_DIM = 8
NOTES_LIT = "ep=demo-0001 worker=fabric step=retrieve tool=web_search ok q=wshard-identity"

HEADER_SIZE = 64
INDEX_ENTRY_SIZE = 48


def notes_text() -> bytes:
    s = NOTES_LIT.ljust(T)  # exact T bytes, one per timestep
    assert len(s) == T, len(s)
    return s.encode("utf-8")


def state_array() -> np.ndarray:
    """T×STATE_DIM float32, bit-identical to main.go (exact m/8 values)."""
    state = np.empty((T, STATE_DIM), dtype="<f4")
    for t in range(T):
        for k in range(STATE_DIM):
            state[t, k] = np.float32(((t + k * 13) % 64) * 0.125)
    return state


def build_episode() -> Episode:
    done = np.zeros(T, dtype=np.uint8)
    done[-1] = 1
    return Episode(
        id=EP_ID,
        env_id=ENV_ID,
        length=T,
        timebase=TimebaseSpec(type=TimebaseType.TICKS, tick_hz=30.0),
        observations={
            "state": Channel(
                name="state", dtype=DType.FLOAT32, shape=[STATE_DIM], data=state_array()
            ),
            "notes": Channel(
                name="notes",
                dtype=DType.UINT8,
                shape=[],
                data=np.frombuffer(notes_text(), dtype=np.uint8),
            ),
        },
        rewards=Channel(
            name="reward",
            dtype=DType.FLOAT32,
            shape=[],
            data=(np.arange(T, dtype=np.float32) * 0.25),
        ),
        terminations=Channel(name="done", dtype=DType.UINT8, shape=[], data=done),
        metadata={"kind": "agent-memory", "worker": "fabric/7"},
    )


def _index_entries(data: bytes):
    """Yield (i, name, entry) parsed from the index — minimal struct layout."""
    entry_count = struct.unpack("<I", data[12:16])[0]
    st_off = struct.unpack("<Q", data[16:24])[0]
    entries = []
    for i in range(entry_count):
        off = HEADER_SIZE + i * INDEX_ENTRY_SIZE
        e = data[off : off + INDEX_ENTRY_SIZE]
        name_off, name_len = struct.unpack("<IH", e[8:14])
        name = data[st_off + name_off : st_off + name_off + name_len].decode("utf-8")
        d_off, d_size, o_size = struct.unpack("<QQQ", e[16:40])
        entries.append((i, name, off, d_off, d_size, o_size))
    return entries


def flip_signal_state(path: str) -> None:
    """Flip one byte of signal/state and patch its CRC32C so the container
    cannot tell. The identity is the only thing left that can."""
    data = bytearray(open(path, "rb").read())
    for i, name, off, d_off, d_size, _ in _index_entries(bytes(data)):
        if name == "signal/state":
            data[d_off] ^= 0xFF
            crc = crc32c.crc32c(bytes(data[d_off : d_off + d_size]))
            data[off + 40 : off + 44] = struct.pack("<I", crc)
            open(path, "wb").write(bytes(data))
            return
    raise RuntimeError("signal/state entry not found")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()

    results: dict = {}
    ep = build_episode()
    ep.validate()

    # A3: derive identity of the Go-written file from Python.
    results["go_file_identity"] = episode_identity(args.go_file)
    results["go_file_verify"] = verify_identity(args.go_file)

    # A1: one logical episode, three encodings, one identity.
    codec_ids = {}
    zstd_path = None
    for comp in (CompressionType.NONE, CompressionType.ZSTD, CompressionType.LZ4):
        p = os.path.join(args.dir, f"episode_py_{comp.value}.wshard")
        save_wshard(ep, p, compression=comp)
        ident = episode_identity(p)
        assert verify_identity(p) == ident, f"verify != identity for {comp.value}"
        codec_ids[comp.name] = ident
        if comp == CompressionType.ZSTD:
            zstd_path = p
        print(
            f"[py] {comp.value:4s} file={os.path.basename(p):28s} "
            f"bytes={os.path.getsize(p):6d} identity={ident}"
        )
    assert len(set(codec_ids.values())) == 1, f"codec identities diverged: {codec_ids}"
    results["codec_ids"] = codec_ids
    results["zstd_path"] = zstd_path

    # A2: tamper a Python copy too — verify must raise, trust must not move.
    tampered = os.path.join(args.dir, "episode_py_tampered.wshard")
    none_path = os.path.join(args.dir, "episode_py_none.wshard")
    with open(none_path, "rb") as f:
        open(tampered, "wb").write(f.read())
    flip_signal_state(tampered)

    # trusted lookup still returns the committed identity
    assert episode_identity(tampered) == codec_ids["NONE"], "trusted id moved!"
    try:
        verify_identity(tampered)
    except ValueError as exc:  # expected
        results["py_tamper_error"] = str(exc)
    else:
        raise AssertionError("verify_identity accepted the tampered file")

    print(f"[py] tamper: episode_identity unchanged (trusts), verify_identity -> "
          f"{results['py_tamper_error']}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"[py] results written to {args.out}")


if __name__ == "__main__":
    main()
