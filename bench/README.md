# WShard Benchmarks

Write/read performance across compression types, compared against NumPy NPZ.

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

The synthetic RGB data is generated with `rng.integers(0, 256, ...)` (uniform random bytes). Random data is incompressible, so compression ratios here reflect worst-case file size overhead — not typical real-sensor data. Real RGB frames with spatial structure compress 2–4× with zstd.

**Measured:**
- **Write:** time to encode and write the full episode to a temp file
- **Read:** time to decode the full episode back into Python/Go objects
- **File size on disk** and compression ratio (raw bytes ÷ on-disk bytes)

**Not measured:** mmap reads, streaming append, partial reads (single block), OS page-cache effects (runs after a warm-up write). For mmap usage, expect the uncompressed read path to be memory-bandwidth limited.

**Runs:** 5 iterations per measurement; median and minimum reported.

**Machine:** AMD Ryzen 7 7700X (16 logical cores), Fedora 43, Go 1.23, Python 3.14.

## How to run

```bash
# Python (run from py/ so wshard package is on sys.path)
cd py && python ../bench/bench_python.py

# Go
cd bench && go test -bench=. -benchmem -benchtime=5x
```

## Results

> Numbers below are from one machine, one run. Reproduce on your hardware before trusting.

### Python

| Config | Write (median) | Write (min) | Read (median) | Read (min) | File size | Ratio |
|--------|----------------|-------------|---------------|------------|-----------|-------|
| wshard-none | 31.1 ms | 30.0 ms | 4.3 ms | 4.0 ms | 20.25 MB | 1.00× |
| wshard-zstd | 36.0 ms | 27.1 ms | 4.2 ms | 4.0 ms | 20.24 MB | 1.00× |
| wshard-lz4 | 334.6 ms | 313.0 ms | 5.6 ms | 5.3 ms | 20.25 MB | 1.00× |
| npz-deflate | 243.0 ms | 237.5 ms | 6.2 ms | 6.0 ms | 20.25 MB | 1.00× |

> Raw payload: 20.25 MB (1000 steps · joint_pos f32[7] + rgb u8[84,84,3] + ctrl f32[7] + reward f32 + done bool)

**Note on lz4 write speed:** The Python lz4 path currently defaults to `high_compression` mode (equivalent to lz4hc level 9), which is slower than lz4's standard fast mode. The Go lz4 path uses the fast compressor and is 10× faster on the write path. See Caveats below.

### Go

```
goos: linux / goarch: amd64 / cpu: AMD Ryzen 7 7700X 8-Core Processor
```

| Benchmark | ns/op | MB/s | B/op | allocs/op |
|-----------|-------|------|------|-----------|
| BenchmarkWriteNone | 22,086,626 | 961 | 21,242,284 | 74 |
| BenchmarkWriteZstd | 39,198,845 | 542 | 148,733,137 | 154 |
| BenchmarkWriteLz4 | 27,924,272 | 760 | 42,622,297 | 124 |
| BenchmarkReadNone | 5,702,534 | 3,723 | 21,241,715 | 138 |
| BenchmarkReadZstd | 7,232,841 | 2,935 | 21,244,420 | 135 |
| BenchmarkReadLz4 | 6,164,857 | 3,444 | 21,241,584 | 132 |
| BenchmarkOpenAndIndex | 11,756 | — | 1,976 | 28 |

`BenchmarkOpenAndIndex` reads only the header + index (~1 KB), not the payload, so MB/s is omitted. At ~12 µs per call, a process can enumerate 80,000+ episode indices per second from local disk.

The MB/s figures are computed against the raw uncompressed payload (~21 MB). For random-byte RGB, zstd write throughput drops to 542 MB/s (vs 961 MB/s uncompressed) because the compressor tries and fails to compress each block. With real structured RGB, compressed write is typically faster overall because fewer bytes hit disk I/O.

## Caveats

1. **Synthetic workload, not representative of all sensor mixes.** Real robotics episodes often have structured joint trajectories that compress 3–10× and RGB images that compress 2–4×. Compression ratio of 1.00× here is a consequence of random-byte inputs, not a property of WShard.

2. **Compression ratio dominated by RGB.** The 21 MB RGB block is 99%+ of the payload. Even a 2× compression ratio on RGB would halve the file size and the read time for the compressed paths.

3. **Python lz4 uses high-compression mode.** The Python wshard codec defaults lz4 to `high_compression` (level 9), which is 10× slower on write than lz4's standard fast mode. This is a tuning choice, not a format limitation. Expect lz4-write to be comparable to none-write once that default is adjusted.

4. **We do not benchmark HDF5.** The comparison-by-installation friction (libhdf5, h5py, versioning) is not worth it for v0.1. HDF5 benchmarks are welcome in issues.

5. **OS page cache effects.** Read benchmarks after a prior write may be partially cache-warm. Results may differ on a cold system or for files larger than RAM.

6. **Single-episode benchmark.** We do not measure batch loads, parallel reads, or streaming append. Those paths exist but are not benchmarked here yet.
