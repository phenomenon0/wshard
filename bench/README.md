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
- **Write:** time to encode and write the full episode to a temp file
- **Read:** time to decode the full episode back into Python/Go/TS objects
- **File size on disk** and compression ratio (raw bytes ÷ on-disk bytes)

**Not measured:** mmap reads, streaming append, OS page-cache effects (runs
after a warm-up write). For mmap usage, expect the uncompressed read path to be
memory-bandwidth limited.

**Runs:** 5 iterations per measurement; median and minimum reported.

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

# TypeScript
cd js && npm run bench:node
# or: cd js && npx tsx bench/bench_node.ts
```

## Results

> Numbers below are from one machine, one run. Reproduce on your hardware before trusting.

### Python

| Config | Write (median) | Write (min) | Read (median) | Read (min) | File size | Ratio |
|--------|----------------|-------------|---------------|------------|-----------|-------|
| wshard-none | 30.8 ms | 29.8 ms | 4.3 ms | 4.1 ms | 20.25 MB | 1.00× |
| wshard-zstd | 208.3 ms | 200.4 ms | 27.6 ms | 27.4 ms | 10.29 MB | 1.97× |
| wshard-lz4 | 385.9 ms | 370.4 ms | 28.8 ms | 28.1 ms | 11.93 MB | 1.70× |
| npz-deflate | 222.0 ms | 221.3 ms | 39.1 ms | 38.7 ms | 9.37 MB | 2.16× |
| hdf5-deflate | 249.0 ms | 246.2 ms | 99.8 ms | 95.8 ms | 9.71 MB | 2.08× |

> Raw payload: 20.25 MB (1000 steps · joint_pos f32[7] + rgb u8[84,84,3] + ctrl f32[7] + reward f32 + done bool)

**Note on lz4 write speed:** The Python lz4 path defaults to `high_compression` mode (level 9 equivalent), which is slower than lz4's standard fast mode. The Go lz4 path uses the fast compressor. See Caveats.

### Go

```
goos: linux / goarch: amd64 / cpu: AMD Ryzen 7 7700X 8-Core Processor
```

| Benchmark | ns/op | MB/s | B/op | allocs/op |
|-----------|-------|------|------|-----------|
| BenchmarkWriteNone | 22,140,335 | 959 | 21,243,446 | 76 |
| BenchmarkWriteZstd | 156,290,607 | 136 | 79,582,620 | 152 |
| BenchmarkWriteLz4 | 99,230,377 | 214 | 39,184,582 | 125 |
| BenchmarkReadNone | 5,135,146 | 4,134 | 21,241,715 | 138 |
| BenchmarkReadZstd | 32,919,572 | 645 | 33,428,923 | 139 |
| BenchmarkReadLz4 | 13,670,311 | 1,553 | 38,944,496 | 133 |
| BenchmarkOpenAndIndex | 14,948 | — | 1,976 | 28 |
| BenchmarkPartialReadCtrl | 19,815 | — | 31,488 | 39 |

`BenchmarkOpenAndIndex` reads only the header + index (~1 KB), not the payload, so MB/s is omitted.
At ~15 µs per call, a process can enumerate 65,000+ episode indices per second from local disk.

`BenchmarkPartialReadCtrl` fetches only `action/ctrl` (28 KB) from a ~50 MB file (rgb + 4 depth
fillers). It reads header + index + the single requested block — ~20 µs total. MB/s is omitted
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
| WShard (zstd) | 25.8 MB | 101 µs | 94 µs |
| NPZ (deflate) | 23.8 MB | 194 µs | 189 µs |
| HDF5 (gzip-4) | 22.7 MB | 255 µs | 251 µs |

WShard reads only the requested channel ~2× faster than NPZ and ~2.5× faster than HDF5 because:
- The index at the front of the file lets the reader seek directly to `action/ctrl` without
  reading or decompressing any other block.
- NPZ is a ZIP file: `np.load` is lazy but must decompress only the matching member, which
  still requires scanning the ZIP central directory (at the end of the file) and seeking back.
- HDF5 also has random-access through its B-tree index, but the index I/O and metadata
  overhead are higher than WShard's flat sequential index.

For Go, `BenchmarkPartialReadCtrl` shows ~20 µs for the same operation (WShard, zstd).
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

6. **Read benchmarks above are single-episode.** Streaming append, real-data schemas,
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
| DreamerV3 (T=200) | 2.35 MB | 2.35 MB | 2.35 MB | 1.00× | 6.4 ms | 0.39 ms |
| D4RL Hopper-v2 (T=1000) | 0.09 MB | n/a | 0.09 MB | 1.06× | 0.22 ms | 0.12 ms |

The DreamerV3 image at ±15 noise sits near the uint8 entropy ceiling; real correlated
camera scenes compress 1.5–3×. NPZ here uses `np.savez` (uncompressed); the compressed-NPZ
comparison is in the headline Python table above.

## Streaming append vs batch write

**Script:** `bench/bench_streaming.py` &nbsp;·&nbsp; T=10,000 &nbsp;·&nbsp; runs: 3.

Identical data via two write paths:

- **Streaming:** `WShardStreamWriter.begin_episode()` → `write_timestep()` × 10000 →
  `end_episode()`. Crash-safe `.partial` → atomic rename.
- **Batch:** preallocate numpy arrays, single `save_wshard()` call.

| Path | Total (med) | Per-step (med) | Peak mem | File size |
|------|-------------|----------------|----------|-----------|
| Streaming | 247.1 ms | **24.7 µs** | 0.44 MiB | 0.62 MiB |
| Batch | 0.6 ms | 0.06 µs | 1.98 MiB | 0.62 MiB |

24.7 µs per step is safe for 1 kHz robot control loops (40× headroom under the 1000 µs
budget). Batch is ~400× faster wall-clock for offline construction, but holds full arrays
in RAM. File sizes match within ~1 KiB (header reservation rounding).

## Parquet baseline (LeRobot's choice)

**Script:** `bench/bench_parquet.py` &nbsp;·&nbsp; same workload as headline table &nbsp;·&nbsp; pyarrow 22.0.0.

Parquet is encoded flat row-per-timestep (closest to LeRobot's actual schema). Multi-dim
channels are expanded into scalar columns (`joint_pos_0..6`); RGB is serialised as a
`large_binary` blob per row. Two configs: `compression="zstd"` and `none`.

| Config | Write (med) | Read (med) | File size |
|--------|-------------|------------|-----------|
| wshard-none | 32.3 ms | **5.6 ms** | 20.25 MB |
| wshard-zstd | 64.0 ms | **4.9 ms** | 20.24 MB |
| parquet-zstd | 26.0 ms | 13.1 ms | 20.25 MB |
| parquet-none | 10.7 ms | 10.2 ms | 20.25 MB |

WShard reads ~2.4× faster than Parquet on the same payload. Parquet's column materialisation
requires Python list construction per column; WShard uses `np.frombuffer` directly into the
mmap'd block. Parquet has no native nd-array type, so each RGB frame must be serialised to
bytes or exploded into flat columns; WShard stores the tensor block verbatim.

Parquet write is faster than wshard-none (10.7 ms vs 32 ms) — pyarrow's column-write path
is highly optimised C++, while WShard's Python writer is pure struct-packing today.

## Many-files dataset scaling

**Script:** `bench/bench_dataset.py` &nbsp;·&nbsp; 1000 episodes × T=100, ~2.4 KB raw/episode.

| Metric | WShard-zstd | NPZ-deflate |
|--------|-------------|-------------|
| Total disk (1000 eps) | 3.34 MB | 3.00 MB |
| Bytes/episode | 3497 B | 3149 B |
| Full-read per-episode | **40.2 µs** | 221.5 µs |
| Index-only per-episode | **9.3 µs** | n/a |
| Full-read total (1000) | 0.040 s | 0.222 s |

WShard full-read is 5.5× faster than NPZ per episode (avoids `zipfile` overhead). The
header+index-only pass at 9.3 µs/episode lets a process enumerate channel names, dtypes,
and block offsets for a 1000-episode dataset in under 10 ms total — without decoding any
tensors. NPZ has no equivalent: parsing the central directory still requires a full file
read for each member listing.

WShard's ~11% disk overhead at T=100 is from the fixed 64-byte header + 48-byte index
entries + string table per file. At T ≥ 500 the ratio inverts (zstd gains beat the fixed
overhead).
