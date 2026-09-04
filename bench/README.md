# WShard Benchmarks

Write/read performance across compression types, compared against NumPy NPZ, HDF5, and
a TypeScript runtime.

## Methodology

**Workload:** one synthetic episode, fixed seed (42), single file, no I/O batching.

| Channel | Shape | DType | Raw size |
|---------|-------|-------|----------|
| `signal/joint_pos` | [1000, 7] | float32 | 28 KB |
| `signal/rgb` | [1000, 84, 84, 3] | uint8 | 21 MB |
| `action/ctrl` | [1000, 7] | float32 | 28 KB |
| `reward` | [1000] | float32 | 4 KB |
| `done` | [1000] | bool | 1 KB |
| **Total raw** | | | **~20.25 MB** |

**RGB workload — structured-but-realistic (smooth-scene proxy):**
The synthetic RGB is a vertical gradient (0–255, top-to-bottom) plus ±2 LSB
per-pixel noise: `noise = rng.integers(0, 4) − 2`. This is the same recipe in
Python, Go, and TypeScript so the three tables compare apples-to-apples.
This pattern represents a static camera under constant illumination — the most
compressible end of the real-world RGB spectrum. A textured outdoor scene with
motion will compress 3–5× less. The previous random-byte RGB gave 1.00×
compression regardless of codec; with this workload zstd achieves ~2× on
the RGB block and about 2× overall (the float32 channels are harder to compress).

**Measured:**
- **Write:** time to encode and write the full episode to a temp file. This
  includes sealing: the writer sha256s every block's uncompressed bytes to build
  `meta/identity`, so an uncompressed write pays one extra pass over the payload.
- **Read:** time to decode the full episode back into Python/Go/TS objects. Reads
  verify CRC32C; they do not re-derive the identity (`verify_identity` /
  `VerifyIdentity` do, and cost a second pass).
- **File size on disk** and compression ratio (raw bytes ÷ on-disk bytes)

**Not measured:** mmap reads, cold-cache reads, OS page-cache effects (runs
after a warm-up write). For mmap usage, expect the uncompressed read path to be
memory-bandwidth limited.

**Runs:** 5 iterations per measurement; median and minimum reported.

For Go, `-benchtime=5x` is right for the whole-episode benchmarks (each does tens
of MB of work) and useless for the two microsecond-scale ones — at five
iterations `BenchmarkOpenAndIndex` is mostly cold-start noise, which is where the
"~15 µs" in earlier revisions of this file came from. Those two are run at
`-benchtime=5000x -count=3` instead, noted in the table.

**Machine:** AMD Ryzen 7 7700X (16 logical cores), Fedora 43, Go 1.23, Python 3.14, Node 22.

## How to run

```bash
# Python (run from py/ so wshard package is on sys.path)
cd py && python ../bench/bench_python.py

# HDF5 baseline (optional, soft dep)
pip install h5py
cd py && python ../bench/bench_python.py   # hdf5-deflate row appears automatically

# Go
cd bench && go test -bench=. -benchmem -benchtime=5x
# the two microsecond-scale ones need real iteration counts:
cd bench && go test -bench='OpenAndIndex|PartialReadCtrl' -benchmem -benchtime=5000x -count=3

# TypeScript
cd js && npm run bench:node
# or: cd js && npx tsx bench/bench_node.ts
```

## Results

> Numbers below are from one machine, one run. Reproduce on your hardware before trusting.

### Python

| Config | Write (median) | Write (min) | Read (median) | Read (min) | File size | Ratio |
|--------|----------------|-------------|---------------|------------|-----------|-------|
| wshard-none | 43.1 ms | 42.3 ms | 5.6 ms | 5.5 ms | 20.25 MB | 1.00× |
| wshard-zstd | 165.4 ms | 158.1 ms | 28.8 ms | 28.5 ms | 10.29 MB | 1.97× |
| wshard-lz4 | 396.9 ms | 387.2 ms | 29.7 ms | 29.6 ms | 11.93 MB | 1.70× |
| npz-deflate | 228.3 ms | 221.9 ms | 38.7 ms | 38.5 ms | 9.37 MB | 2.16× |
| hdf5-deflate | 253.0 ms | 252.1 ms | 97.6 ms | 96.7 ms | 9.71 MB | 2.08× |

> Raw payload: 20.25 MB (1000 steps · joint_pos f32[7] + rgb u8[84,84,3] + ctrl f32[7] + reward f32 + done bool)

**Note on lz4 write speed:** The Python lz4 path defaults to `high_compression` mode (level 9 equivalent), which is slower than lz4's standard fast mode. The Go lz4 path uses the fast compressor. See Caveats.

### Go

```
goos: linux / goarch: amd64 / cpu: AMD Ryzen 7 7700X 8-Core Processor
```

| Benchmark | ns/op | MB/s | B/op | allocs/op |
|-----------|-------|------|------|-----------|
| BenchmarkWriteNone | 32,248,829 | 658 | 21,247,161 | 101 |
| BenchmarkWriteZstd | 160,907,283 | 132 | 79,582,492 | 150 |
| BenchmarkWriteLz4 | 99,724,901 | 213 | 39,212,556 | 126 |
| BenchmarkReadNone | 5,190,951 | 4,090 | 21,243,320 | 151 |
| BenchmarkReadZstd | 33,135,416 | 641 | 33,431,484 | 146 |
| BenchmarkReadLz4 | 12,977,979 | 1,636 | 38,946,460 | 137 |
| BenchmarkOpenAndIndex † | 8,171 | — | 2,456 | 32 |
| BenchmarkPartialReadCtrl † | 11,605 | — | 31,488 | 39 |

† at `-benchtime=5000x -count=3` (median of three); the rest at `-benchtime=5x`.

`BenchmarkWriteNone` costs ~10 ms more than it did before the writer started sealing files.
That is the sha256 pass over the 20 MB payload that builds `meta/identity` — one extra read of
every block. The compressed paths absorb it inside the codec's own cost.

`BenchmarkOpenAndIndex` reads only the header + index (~1 KB), not the payload, so MB/s is omitted.
At ~8 µs per call, a process can enumerate 120,000+ episode indices per second from local disk.
The Python equivalent measured over 1000 real files is 9.8 µs (see many-files section), which is
the more honest number to plan against — it includes opening a distinct file each time.

`BenchmarkPartialReadCtrl` fetches only `action/ctrl` (28 KB) from a ~50 MB file (rgb + 4 depth
fillers). It reads header + index + the single requested block — ~12 µs total. MB/s is omitted
because 28 KB / 50 MB is the point: the file size is irrelevant to how long selective fetch takes.

The compressed paths (zstd/lz4) are slower on write because the structured RGB compresses
meaningfully (~2×) — the compressor does real work, unlike with random bytes.

### TypeScript (Node 22)

| Config | Write (median) | Write (min) | Read (median) | Read (min) | File size | Ratio |
|--------|----------------|-------------|---------------|------------|-----------|-------|
| wshard-none | 142.6 ms | 140.5 ms | 142.4 ms | 140.7 ms | 20.25 MB | 1.00× |
| wshard-zstd | 346.7 ms | 330.9 ms | 130.9 ms | 128.1 ms | 10.30 MB | 1.97× |
| wshard-lz4 | 663.5 ms | 640.8 ms | 131.3 ms | 126.6 ms | 16.66 MB | 1.22× |

> Run with: `cd js && npm run bench:node`

The TS implementation uses WASM codecs (@bokuweb/zstd-wasm, fflate lz4) so write throughput is
lower than Go native. Read throughput is comparable because decompression of the same output is
the bottleneck for all runtimes. The wshard-none ratio of 1.00× is expected — no compression.

Note: zstd-wasm must be initialized before first use (`initZstd()`); the bench handles this
automatically. The one-time init cost (~30 ms) is excluded from the measured runs.

### Partial-block reads — `action/ctrl` (28 KB) from a ~50 MB file

Setup: one file containing `signal/rgb` [1000,84,84,3] (21 MB) + four depth-filler blocks
[1000,84,84,1] (~7 MB each) + `action/ctrl` [1000,7] (28 KB). Total on-disk size varies by codec.
Time measured: wall time to open the file and fetch only the `action/ctrl` block.

| Format | File size | Median time | Min time |
|--------|-----------|-------------|----------|
| WShard (zstd) | 25.8 MB | 111 µs | 100 µs |
| NPZ (deflate) | 23.8 MB | 175 µs | 170 µs |
| HDF5 (gzip-4) | 22.7 MB | 273 µs | 248 µs |

WShard reads only the requested channel ~1.6× faster than NPZ and ~2.5× faster than HDF5 because:
- The index at the front of the file lets the reader seek directly to `action/ctrl` without
  reading or decompressing any other block.
- NPZ is a ZIP file: `np.load` is lazy but must decompress only the matching member, which
  still requires scanning the ZIP central directory (at the end of the file) and seeking back.
- HDF5 also has random-access through its B-tree index, but the index I/O and metadata
  overhead are higher than WShard's flat sequential index.

For Go, `BenchmarkPartialReadCtrl` shows ~12 µs for the same operation (WShard, zstd).
The Python figure includes Episode/Channel object construction; the underlying file IO
itself is in the same single-digit-µs range.

**API:** Python now exposes `load_wshard(path, channels=[...])` — when both a path and a
channel allow-list are given, the loader streams only the requested blocks from disk.
In Go, `ReadEntryByName("action/ctrl")` on an open `ShardReader` is the equivalent.

## Caveats

1. **Structured but synthetic — not all real workloads.** The RGB data uses a vertical gradient
   with ±2 LSB noise: the smoothest plausible scene. Real outdoor or robot-manipulation RGB
   can be less compressible (2–4×) or, with complex textures and motion, closer to incompressible.
   The float32 channels (joint_pos, ctrl, reward) are Gaussian random and compress modestly.

2. **Compression ratio dominated by RGB.** The 21 MB RGB block is 99%+ of the payload. The
   ~2× ratio reflects almost entirely how well the codec handles the gradient+noise pattern.

3. **Python lz4 uses high-compression mode.** The Python wshard codec defaults lz4 to
   `high_compression` (level 9), which is much slower on write than lz4's standard fast mode.
   This is a tuning choice, not a format limitation. Expect lz4-write to be comparable to
   none-write once that default is adjusted.

4. **TypeScript WASM codecs.** The zstd and lz4 paths in JS use WASM implementations
   (@bokuweb/zstd-wasm, fflate). Performance is lower than native Go; this is expected for
   browser-compatible WASM. lz4 via fflate achieves 1.22× compression (lower than the Go
   path's ~1.7×) because fflate uses a block-level lz4 implementation with different defaults.

5. **OS page cache effects.** Read benchmarks after a prior write may be partially cache-warm.
   Results may differ on a cold system or for files larger than RAM.

6. **Read benchmarks above are single-episode.** Live recording, real-data schemas,
   Parquet comparison, and many-files dataset scaling are covered in the additional
   benchmarks below.

---

# Additional benchmarks

These extend the headline single-file table with four common real-world scenarios:
real-data schemas, streaming/append, a Parquet baseline (LeRobot's storage format),
and dataset-scale many-files reads.

## Real-Data Schemas

**Script:** `bench/bench_realdata.py` &nbsp;·&nbsp; runs per timing: 3.

Two realistic RL schemas synthesised from scratch (no external dataset downloads):

- **DreamerV3-style (T=200):** `image` u8[200,64,64,3] (gradient + ±15 noise),
  `action` f32[200,6], `reward`, `is_first`/`is_last`/`is_terminal` bool[200]. Loaded via
  `load_dreamer` (NPZ → Episode), then `save_wshard(zstd)`.
- **D4RL Hopper-v2-style (T=1000):** `observations` f32[1000,17], `actions` f32[1000,6],
  `rewards`, `terminals`. Built as `Episode` directly; `save_wshard(zstd)`.

Round-trip verified with `np.array_equal` and `np.allclose`.

| Schema | Raw size | NPZ size | WShard-zstd | Ratio | Write (med) | Read (med) |
|--------|----------|----------|-------------|-------|-------------|------------|
| DreamerV3 (T=200) | 2.35 MB | 2.35 MB | 2.35 MB | 1.00× | 7.3 ms | 0.30 ms |
| D4RL Hopper-v2 (T=1000) | 0.09 MB | n/a | 0.09 MB | 1.05× | 0.30 ms | 0.14 ms |

The DreamerV3 image at ±15 noise sits near the uint8 entropy ceiling; real correlated
camera scenes compress 1.5–3×. NPZ here uses `np.savez` (uncompressed); the compressed-NPZ
comparison is in the headline Python table above.

## Streaming writer vs batch write

**Script:** `bench/bench_streaming.py` &nbsp;·&nbsp; T=10,000 &nbsp;·&nbsp; runs: 3.

Identical data via two write paths:

- **Streaming:** `WShardStreamWriter.begin_episode()` → `write_timestep()` × 10000 →
  `end_episode()`. Buffers in memory, then writes, seals, `fsync`s, and renames
  `.partial` → `episode.wshard` in one step.
- **Batch:** preallocate numpy arrays, single `save_wshard()` call.

| Path | Total (med) | Per-step (med) | Peak mem | File size |
|------|-------------|----------------|----------|-----------|
| Streaming | 239.9 ms | **24.0 µs** | 1.08 MiB | 0.62 MiB |
| Batch | 1.1 ms | 0.11 µs | 1.98 MiB | 0.62 MiB |

24 µs per step is safe for 1 kHz robot control loops (40× headroom under the 1000 µs
budget). Batch is ~220× faster wall-clock for offline construction. File sizes match
within ~1 KiB (header reservation rounding).

**"Streaming" here means the call shape, not durability.** The streaming writer holds
the whole episode in memory until `end_episode()` — the 1.08 MiB peak *is* the episode
— because the index describes each block as one extent, so a block can be written only
once. Nothing is on disk before the final call, and a crash before it loses the episode
(cleanly: no partial file is left readable). Batch's higher peak is the preallocated
numpy arrays plus the encode buffer. For crash-survivable long runs, write chunk files
instead; each one is a complete sealed episode.

## Parquet baseline (LeRobot's choice)

**Script:** `bench/bench_parquet.py` &nbsp;·&nbsp; same workload as headline table &nbsp;·&nbsp; pyarrow 22.0.0.

Parquet is encoded flat row-per-timestep (closest to LeRobot's actual schema). Multi-dim
channels are expanded into scalar columns (`joint_pos_0..6`); RGB is serialised as a
`large_binary` blob per row. Two configs: `compression="zstd"` and `none`.

| Config | Write (med) | Read (med) | File size |
|--------|-------------|------------|-----------|
| wshard-none | 43.1 ms | **6.0 ms** | 20.25 MB |
| wshard-zstd | 75.4 ms | **5.7 ms** | 20.24 MB |
| parquet-zstd | 27.8 ms | 13.8 ms | 20.25 MB |
| parquet-none | 11.8 ms | 12.1 ms | 20.25 MB |

WShard reads ~2.3× faster than Parquet on the same payload. Parquet's column materialisation
requires Python list construction per column; WShard uses `np.frombuffer` directly into the
mmap'd block. Parquet has no native nd-array type, so each RGB frame must be serialised to
bytes or exploded into flat columns; WShard stores the tensor block verbatim.

Parquet write is faster than wshard-none (11.8 ms vs 43 ms) — pyarrow's column-write path
is highly optimised C++, while WShard's Python writer is pure struct-packing plus one sha256
pass over the payload to seal the file.

## Many-files dataset scaling

**Script:** `bench/bench_dataset.py` &nbsp;·&nbsp; 1000 episodes × T=100, ~2.4 KB raw/episode.

| Metric | WShard-zstd | NPZ-deflate |
|--------|-------------|-------------|
| Total disk (1000 eps) | 3.854 MB | 3.003 MB |
| Bytes/episode | 4041 B | 3149 B |
| Full-read per-episode | **46.1 µs** | 179.9 µs |
| Index-only per-episode | **9.8 µs** | n/a |
| Full-read total (1000) | 0.046 s | 0.180 s |

WShard full-read is 3.9× faster than NPZ per episode (avoids `zipfile` overhead). The
header+index-only pass at 9.8 µs/episode lets a process enumerate channel names, dtypes,
and block offsets for a 1000-episode dataset in about 10 ms total — without decoding any
tensors. NPZ has no equivalent: parsing the central directory still requires a full file
read for each member listing.

WShard's ~28% disk overhead at T=100 is the fixed 64-byte header + 48-byte index entries +
string table, plus `meta/identity` — which is JSON naming every block and its sha256, so
it costs a few hundred bytes regardless of how small the episode is (~540 B of the 4041 B
here). On a 2.5 KB payload that is most of the overhead. It is flat, so it disappears into
the noise by T ≈ 1000; at T ≥ 500 the ratio inverts and zstd gains beat the fixed cost.
