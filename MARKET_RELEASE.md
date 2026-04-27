# WShard 0.1 — One file. One episode. Three languages.

*Show HN release announcement — April 2026*

> A small, neutral binary file format for trajectory data — robots, RL,
> world-model lanes — with verified parity across Python, TypeScript, and Go.
> MIT-licensed. Beta. We are looking for sharp edges.

---

## The 30-second pitch

If you train robots, run RL agents, or build world models, you produce the
same shape of data: time-indexed observations, actions, rewards, terminations,
sometimes camera frames, sometimes proprioception, sometimes model
predictions. Today that data lives in HDF5, NPZ, RLDS, LeRobot's Parquet+MP4,
MCAP, Zarr, or a lab-specific custom format. Each is great at what it was
built for. None was built for **self-contained tensor episodes plus
world-model prediction lanes**, written and read identically from Python,
TypeScript, and Go.

That's the gap WShard fills. Not "yet another format." A **small, focused**
format for one specific shape of problem.

---

## What's in 0.1

| Capability | Status |
|---|---|
| Python reader / writer | beta, 139 tests passing |
| TypeScript reader / writer | beta, 15 tests passing |
| Go reader / writer | beta, full test coverage |
| Cross-language byte-parity | verified by golden-file CI |
| Per-block compression (none / zstd / lz4) | yes |
| CRC32C integrity per block | yes |
| Streaming append (`.partial` → atomic rename) | yes |
| Chunked episodes with continuity validation | yes |
| Single-channel partial reads | yes — `load_wshard(path, channels=[…])` |
| Multi-modal observation lanes (RGB / depth / proprio / language) | yes |
| Residual encoding (sign-of-2nd-difference, 2 bits/element) | experimental |
| CLI (`wshard inspect / verify / convert / export / doctor`) | yes |
| Browser-only viewer (drag a file, see the block layout) | yes |
| Converters: DreamerV3 NPZ ⇄ WShard | round-trip tested |
| Converters: Minari, D4RL, LeRobot bridges | exist, **not** integration-tested yet |
| H.264 / H.265 / AV1 video block type | **future work** |
| Arrow / Polars / DuckDB integration | **future work** |
| Formal schema registry | **future work** — block-name conventions are advisory today |
| PyPI / npm publishes | **deferred** until format and API stabilize |

---

## Numbers we're willing to defend

These come from `bench/` in the repo. All single-machine, x86_64, NVMe,
warm cache where noted.

**Full-episode read** (T=1000, ~21 MB raw, 84×84×3 RGB + joint pos + ctrl + reward + done):

| Format          | Read time |
|-----------------|-----------|
| WShard (no compression) | ~6 ms (Go) / ~4 ms (Python) |
| WShard (zstd)           | ~28 ms (Python) |
| HDF5 (deflate)          | ~100 ms (Python) |
| NumPy NPZ (deflate)     | ~94 ms (Python) |
| Parquet (LeRobot-style) | ~67 ms (Python) |

**Single-channel partial fetch** (one 28 KB action block from a 50 MB file,
public API only, no internal helpers):

| Format            | Median time |
|-------------------|-------------|
| **WShard (zstd)** | **101 µs** |
| NPZ (deflate)     | 194 µs |
| HDF5 (gzip-4)     | 255 µs |

**Streaming append**: per-step write cost is ~25 µs. Crash-safe by virtue of
the `.partial` file pattern — episodes either finalize via atomic rename or
remain visibly unfinished.

We are not claiming WShard is fastest at everything. We are claiming it is
fast at the things you do most when iterating on RL or world-model training:
seek to one block, read a whole episode end-to-end, append timesteps live.

---

## What WShard is not

- Not a database.
- Not a training framework.
- Not a replacement for HDF5, LeRobot, MCAP, Zarr, or Parquet.
- Not a publishing format for HuggingFace Hub (use LeRobot for that).
- Not a pub/sub message bus (use MCAP for that).

WShard is a **layer**: the binary file your training loop loads, your
recorder writes, and your dashboard parses. Bridges to the layers above and
below already exist or are in progress.

| When you want…                                | Reach for… |
|-----------------------------------------------|------------|
| HuggingFace-Hub-hosted robotics datasets      | LeRobot |
| Timestamped pub/sub robot logs                | MCAP |
| Chunked N-dim arrays in cloud object storage  | Zarr |
| Mature scientific array storage               | HDF5 |
| Self-contained episode files **with** world-model lanes | WShard |

---

## Format at a glance

```
[Header 64 B] [Index N×48 B] [String Table] [Padding] [Data Blocks…]
```

- **Header**: magic (`SHRD`), version, role (`0x05` = WShard), alignment,
  default compression, entry count, offsets.
- **Index entry (48 B)**: name offset/length, xxHash64 of name, data offset,
  on-disk size, original size, dtype, shape rank, flags, CRC32C.
- **Block names** are hierarchical paths:
  - `meta/wshard`, `meta/episode`, `meta/channels` — JSON metadata
  - `signal/<id>` — observation tensors
  - `action/<id>` — action tensors
  - `time/ticks`, `time/timestamps_ns` — temporal axis
  - `omen/<id>/<model>` — model predictions
  - `uncert/<id>/<kind>` — uncertainty estimates
  - `residual/<id>/<kind>` — packed residual encodings
  - `reward`, `done` — scalar tracks

Per-block compression is decoded from the index entry, not the header — so a
file can mix raw and compressed blocks freely. Data blocks are 32-byte
aligned: read with `mmap()` + cast for low-copy access.

The full byte-level spec is in [`docs/DEEP_DIVE.md`](docs/DEEP_DIVE.md).

---

## Honest about edges

This is a beta launch. We expect to find bugs. The biggest unknowns:

1. **Object stores.** S3/GCS/Azure haven't been stress-tested. WShard is a
   single file with a header at offset 0 — should be fine for HTTP range
   requests, but we haven't measured.
2. **Schema drift.** Block names are conventions, not contracts. If two labs
   call their joint angles `joint_pos` and `q`, today nothing flags that.
3. **Video.** RGB is currently raw uint8. Adding native H.264/H.265/AV1
   block types is the most-requested item we know about. It will land in
   0.2 if the launch produces clear demand.
4. **Cross-language quirks.** xxHash64 in WASM (TypeScript) is a recent fix.
   We caught it because golden-file CI compares hashes byte-for-byte. We
   don't believe other parity gaps remain — but this is exactly the kind of
   issue an external user reproduces in five minutes.

---

## The ask

If you build, train on, or ship trajectory data:

1. **Try WShard against your own data.**
   `pip install "git+https://github.com/phenomenon0/wshard.git#subdirectory=py"`
2. **Tell us where it breaks.** Schema, throughput, cloud, converters,
   API ergonomics — all in scope. [Open an issue.](https://github.com/phenomenon0/wshard/issues)
3. **Tell us which converter should exist next.** LeRobot? MCAP? RLDS?
   Minari? D4RL? Zarr? We will build the most-asked-for one first.

We're not trying to win. We're trying to find out whether a small, neutral
episode format is something anyone other than us actually needs.

---

## Links

- **Code**: <https://github.com/phenomenon0/wshard>
- **Spec**: [`docs/DEEP_DIVE.md`](docs/DEEP_DIVE.md)
- **Format crash course**: [`docs/FORMAT.md`](docs/FORMAT.md)
- **FAQ**: [`docs/FAQ.md`](docs/FAQ.md)
- **Why not HDF5?**: [`docs/WHY_NOT_HDF5.md`](docs/WHY_NOT_HDF5.md)
- **Why not LeRobot?**: [`docs/WHY_NOT_LEROBOT.md`](docs/WHY_NOT_LEROBOT.md)
- **Benchmarks**: [`bench/README.md`](bench/README.md)
- **Security**: [`SECURITY.md`](SECURITY.md)
- **Browser viewer**: [`viewer/`](viewer/)

MIT-licensed. No telemetry. No phone-home. No registration.
