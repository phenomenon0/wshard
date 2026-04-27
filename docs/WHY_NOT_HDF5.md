# Why Not HDF5?

HDF5 is the most cited self-contained array format in scientific computing, and it is the right tool for a large class of problems. This document explains where HDF5 excels, where the trade-offs land for WShard's specific use case, and when you should pick HDF5 instead.

---

## HDF5 strengths

HDF5 has 25+ years of production use. Its strengths are real:

- **Broad ecosystem.** h5py, MATLAB, IDL, R, Julia, Fortran — HDF5 readers exist across almost every scientific computing stack.
- **Rich data model.** Compound types (structs), virtual datasets (cross-file views), soft and hard links, hierarchical groups with arbitrary nesting, chunked datasets with fill values, dimension scales.
- **Advanced chunking.** HDF5 lets you define chunk shapes per dataset and choose per-chunk filters (gzip, szip, lz4, zstd via plugins). For workloads that need slice-access along arbitrary axes, HDF5's chunked layout is well-engineered.
- **Parallel I/O.** HDF5 with MPI (HDF5-mpio, h5py with `driver="mpio"`) supports concurrent reads and coordinated parallel writes across compute nodes. This is a genuine capability WShard does not have.
- **Single-writer/many-reader (SWMR).** HDF5's SWMR mode allows one writer and multiple readers simultaneously — useful for live monitoring of ongoing writes.

---

## HDF5 trade-offs for streaming episode files

These are not bugs in HDF5; they are trade-offs that land poorly for our specific use case.

**Single-writer concurrency.** HDF5's default locking model means one process owns the file for writing. SWMR mitigates this for reads but does not support multiple writers. For streaming robot recording where the writer crashes and must be recovered, the file is left in an undefined state. HDF5 4.x has improved crash recovery, but it requires explicit transaction journaling that most users do not enable.

**Minimum overhead per file.** An empty HDF5 file is roughly 99 KB on disk (superblock, root group B-tree, symbol table, free-space manager). For workloads with many small episodes (T=100, ~2 KB of data), the fixed overhead dominates. WShard's fixed overhead is 64 bytes (header) + 48 bytes per block (index entry) + string table — for a 5-block episode it is under 400 bytes.

**Complex spec, one reference implementation.** The HDF5 specification is over 1,000 pages. In practice, "HDF5 support" means "depends on libhdf5," the C library from The HDF Group. Go and Rust HDF5 readers are wrappers around libhdf5 via CGo or bindgen. A pure-Go or pure-Rust HDF5 implementation from spec would be a multi-year project. WShard's full spec fits in roughly 200 lines; Go, Python, and TypeScript implementations exist without any shared C library.

**No semantic lanes for prediction data.** HDF5 has groups (directories) and datasets (arrays), but nothing in the format model distinguishes ground truth from model predictions. You impose that convention yourself. WShard's `omen/`, `uncert/`, and `residual/` prefixes are a thin convention layer on top of the same flat-binary concept — but they are built into the reader routing.

---

## Performance comparison (measured)

All numbers from `bench/bench_python.py` and the partial-read benchmark in `bench/README.md`. Same workload: T=1000 episode, joint_pos + rgb + ctrl + reward + done, ~20 MB raw.

### Full episode read (Python)

| Config | Read median | Read min | File size | Ratio |
|--------|-------------|----------|-----------|-------|
| wshard-none | 4.3 ms | 4.1 ms | 20.25 MB | 1.00× |
| wshard-zstd | 27.6 ms | 27.4 ms | 10.29 MB | 1.97× |
| hdf5-deflate | 99.8 ms | 95.8 ms | 9.71 MB | 2.08× |

WShard (uncompressed) reads **~23× faster** than HDF5-deflate on this workload. Even WShard-zstd, which achieves similar compression to HDF5-deflate, reads **3.6× faster** (27.6 ms vs 99.8 ms). The difference is mainly that WShard's index is at the front of the file (sequential read), each block decompresses independently, and there is no HDF5 B-tree traversal or metadata decode overhead.

### Partial-block reads — fetching one channel from a large file

| Format | File size | Median time | Min time |
|--------|-----------|-------------|----------|
| WShard (zstd) | 25.8 MB | 39 µs | 34 µs |
| HDF5 (gzip-4) | 22.7 MB | 255 µs | 251 µs |

WShard is **~6.5× faster** for partial-block reads. This is the workload most training loops actually run: fetch `action/ctrl` (28 KB) from an episode that also contains a 21 MB RGB block. WShard seeks directly to `action/ctrl` using the front-of-file index. HDF5 uses a B-tree index, but the metadata I/O and object-header overhead add up to ~6× more latency on this machine.

Both benchmarks: AMD Ryzen 7 7700X, Fedora 43, Python 3.14, h5py 3.x, pyarrow 22.0.0. Reproduce on your hardware before making decisions.

---

## Why WShard is different (not necessarily better)

**Tiny spec.** The complete binary format is documented in a few hundred lines. Anyone can implement a reader in any language without a shared C library. TypeScript runs in the browser, no native dependencies.

**Streaming append with crash safety.** WShard's `.partial` file pattern means a robot process that crashes mid-episode leaves the previous file intact. HDF5 in streaming mode is risky without explicit journaling.

**Per-block compression, not per-dataset.** You can zstd one block and leave another uncompressed within the same file, controlled per block at write time.

**No minimum file overhead.** The 64-byte header scales to tiny files (T=10, sub-kilobyte episodes) without the ~99 KB HDF5 superblock cost.

**64-byte header fits in one disk sector.** The header is always parseable with a single aligned read. For latency-sensitive index enumeration (65,000+ episodes/second from local disk), this matters.

---

## When to pick HDF5 instead

1. **You need parallel MPI writes.** HDF5-mpio supports coordinated parallel writes from many compute nodes. WShard does not; it is single-writer by design.
2. **Your data uses compound types or virtual datasets.** HDF5's structured dtypes (C-style structs per row), dimension scales, and virtual dataset views have no equivalent in WShard.
3. **Your team or collaborators are already on the HDF5 ecosystem.** MATLAB, IDL, and Fortran users read HDF5 natively. WShard has no readers in those languages.
4. **You need long-term archival with broad tool support.** HDF5 files from 2002 are still readable today. WShard is new; long-term tool coverage is unproven.
