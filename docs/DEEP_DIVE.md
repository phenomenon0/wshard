# WShard Deep Dive

## Table of Contents

1. [What WShard Is](#1-what-wshard-is)
2. [The Binary Format — Byte by Byte](#2-the-binary-format--byte-by-byte)
3. [The Problem It Solves — And Why Nobody Else Has](#3-the-problem-it-solves--and-why-nobody-else-has)
4. [Cross-Language Interop — One Format, Three Runtimes](#4-cross-language-interop--one-format-three-runtimes)
5. [DeepData Bridge — Similarity Search Over Episodes](#5-deepdata-bridge--similarity-search-over-episodes)

For internal positioning notes see [`../notes/MARKETING_BRIEF.md`](../notes/MARKETING_BRIEF.md). They are not part of the published spec — they are kept for context, not as claims.

---

## 1. What WShard Is

WShard (World-Model Episode Shard) is a binary file format and cross-language library for storing, reading, and managing trajectory data from robotics, reinforcement learning, and world model training pipelines.

A single `.wshard` file is a self-contained episode: a time-indexed bundle of observations, actions, rewards, and termination signals recorded from an agent interacting with an environment. The file is a flat binary with fast name-hash lookup, per-block compression, and data alignment for low-copy reads.

### Not a database. Not a framework. A format.

WShard does not run a server. It does not require GPUs. It does not impose a training loop. It is a file format — like Parquet is to tabular data or EXR is to HDR images — purpose-built for one domain: sequential decision-making episodes.

### The Episode as First-Class Citizen

```
┌─────────────────────────────────────────────────┐
│  episode_abc.wshard                             │
│                                                 │
│  meta/wshard      → {"version": 1, "timebase":…}│
│  meta/episode     → {"episode_id": "abc", …}   │
│  meta/channels    → {"channels": [{…}, {…}]}   │
│                                                 │
│  signal/rgb_cam   → [T, 84, 84, 3] uint8       │
│  signal/joint_pos → [T, 7] float32             │
│  signal/force     → [T, 6] float32             │
│  action/ctrl      → [T, 7] float32             │
│  reward           → [T] float32                │
│  done             → [T] bool                   │
│                                                 │
│  omen/joint_pos/dreamer → [T, 7] float32       │ ← model predictions
│  residual/joint_pos/sign2nddiff → packed bits  │ ← compressed deltas
│  uncert/joint_pos/dreamer/std → [T, 7] float32 │ ← uncertainty
└─────────────────────────────────────────────────┘
```

Each named block is independently addressable. A training loop that only needs `signal/joint_pos` and `action/ctrl` reads exactly those two blocks — no deserialization of video frames or metadata.

### What Ships Today

| Component | Language | Lines | Status |
|-----------|----------|-------|--------|
| `wshard` Python package | Python | ~4,000 | Beta |
| `@wshard/core` npm package | TypeScript | ~3,500 | Beta |
| `shard` Go package | Go | ~2,000 | Beta |
| Golden test fixtures | Go generator | 553 | Verified |
| DeepData trajectory bridge | Python | ~500 | Experimental |

Python: 103 tests, TypeScript: 15 tests, all passing locally. The cross-language conformance matrix runs in CI. Beta means the API and on-disk format are stable enough to try; we are looking for external users to surface real-world breakage before declaring 1.0.

---

## 2. The Binary Format — Byte by Byte

WShard rides on **Shard**, a general-purpose binary container format (like a simplified ZIP with aligned data sections). WShard is Shard with `role = 0x05`.

### File Layout

```
Offset    Size    Content
────────────────────────────────────────────────
0x00      4       Magic: "SHRD" (0x53 0x48 0x52 0x44)
0x04      1       Version: 0x02
0x05      1       Role: 0x05 (WShard)
0x06      2       Flags (LE uint16)
0x08      1       Alignment (0, 16, 32, 64 bytes)
0x09      1       Compression default (0=none, 1=zstd, 2=lz4)
0x0A      2       Index entry size: 48 (LE uint16)
0x0C      4       Entry count (LE uint32)
0x10      8       String table offset (LE uint64)
0x18      8       Data section offset (LE uint64)
0x20      8       Schema offset (LE uint64, 0 if absent)
0x28      8       Total file size (LE uint64)
0x30      16      Reserved (zeroed)
────────────────────────────────────────────────
0x40      N×48    Index entries
          var     String table
          pad     Alignment padding (0x00)
          var     Data blocks (each aligned)
```

**Total header: 64 bytes.** Fixed. Parseable with a single `read(64)` call.

### Index Entry (48 bytes)

Each block in the file has one index entry:

```
Offset    Size    Field
────────────────────────────────────
0x00      8       Name hash (xxHash64 of UTF-8 name)
0x08      4       Name offset into string table (LE)
0x0C      2       Name length (LE)
0x0E      2       Flags (LE) — bit 0: compressed,
                               bit 1: zstd, bit 2: lz4
0x10      8       Data offset in file (absolute, LE)
0x18      8       Disk size (compressed, LE)
0x20      8       Original size (uncompressed, LE)
0x28      4       CRC32C checksum (of uncompressed data)
0x2C      2       Content type (0=raw, 2=JSON)
0x2E      2       Reserved
```

### Why This Matters

**Fast name lookup.** Each index entry carries an 8-byte xxHash64 of the block name. Finding `signal/joint_pos` requires reading the 64-byte header plus a scan over the index hashes — for a 50-block file, that is 64 + 50×8 = 464 bytes, with no string-table parse and no full deserialization. Readers may build an in-memory hash map at open time for O(1) repeat lookups, or scan in place for one-shot reads.

**Per-block compression.** Each block carries its own compression flag. Video blocks can use zstd at high ratios. Small scalar blocks (reward, done) stay uncompressed. The reader detects compression from the entry flags — no header-level assumption needed.

**Alignment.** Data blocks start at 32-byte or 64-byte boundaries. This means `mmap()` + pointer cast gives you an AVX-aligned float32 array with zero copies. The Go reader uses this for memory-mapped reads; Python can use it with `np.frombuffer()` on mmap'd files.

**Checksums.** CRC32C (Castagnoli polynomial `0x82F63B78`) on uncompressed data. This is the hardware-accelerated CRC on x86 (`_mm_crc32_*`) and ARM (`__crc32c*`). Go's `crc32.Castagnoli`, Python's `crc32c` package, and the TypeScript implementation all use this polynomial.

### Block Naming Convention

Names are hierarchical paths separated by `/`:

| Prefix | Purpose | Examples |
|--------|---------|----------|
| `meta/` | JSON metadata | `meta/wshard`, `meta/episode`, `meta/channels` |
| `signal/` | Ground truth observations | `signal/rgb`, `signal/joint_pos` |
| `action/` | Agent actions | `action/ctrl`, `action/gripper` |
| `omen/` | Model predictions | `omen/joint_pos/dreamer` |
| `uncert/` | Uncertainty estimates | `uncert/joint_pos/dreamer/std` |
| `residual/` | Compressed residuals | `residual/joint_pos/sign2nddiff` |
| `time/` | Timestamps | `time/ticks`, `time/timestamps_ns` |
| `reward` | Reward signal (no prefix) | `reward` |
| `done` | Termination flags | `done` |

This naming convention is semantic, not syntactic. The reader uses it to route blocks to the correct Episode fields. Action blocks go to `ep.actions`, signal blocks to `ep.observations`, omen blocks to `ep.omens`.

### Timebase

The `meta/wshard` block contains a `timebase` object describing the episode's time axis:

```json
{
  "timebase": {
    "type": "ticks",
    "tick_hz": 30.0
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | `string` | `"ticks"` (fixed-rate) or `"timestamps_ns"` (variable-rate wall clock) |
| `tick_hz` | `float64` | Ticks per second. Only meaningful when `type == "ticks"`. |

When `type == "ticks"`, each timestep `t` corresponds to real time `t / tick_hz` seconds. When `type == "timestamps_ns"`, the `time/timestamps_ns` block contains per-timestep nanosecond timestamps (int64 LE).

### Multi-Modal Signal Naming

Multi-modal observations use a two-level signal path: `signal/{group}/{modality}`:

| Modality | Constant | Example block name |
|----------|----------|--------------------|
| RGB camera | `rgb` | `signal/cam0/rgb` |
| Depth sensor | `depth` | `signal/cam0/depth` |
| Language | `language` | `signal/cmd/language` |
| Proprioception | `proprioception` | `signal/arm/proprioception` |
| Audio | `audio` | `signal/mic/audio` |
| Video | `video` | `signal/cam0/video` |
| Point cloud | `pointcloud` | `signal/lidar/pointcloud` |

### Latent Action Naming

Latent action embeddings and codebook indices use the `omen/` namespace:

| Block name pattern | Description |
|--------------------|-------------|
| `omen/latent_action/{model}` | Latent action embeddings from model |
| `omen/latent_action_codebook/{model}` | Discrete codebook indices for latent actions |

### Supported Data Types

13 types, matching the union of numpy, PyTorch, and Go primitive types:

| WShard | Size | numpy | Notes |
|--------|------|-------|-------|
| `f32` | 4 | float32 | Default for most signals |
| `f64` | 8 | float64 | High-precision physics |
| `f16` | 2 | float16 | Inference outputs |
| `bf16` | 2 | bfloat16* | Training-native (requires ml_dtypes) |
| `i64` | 8 | int64 | Timestamps, large indices |
| `i32` | 4 | int32 | Action indices, labels |
| `i16` | 2 | int16 | Quantized signals |
| `i8` | 1 | int8 | Quantized weights |
| `u64` | 8 | uint64 | Hashes, addresses |
| `u32` | 4 | uint32 | Indices, masks |
| `u16` | 2 | uint16 | Depth images |
| `u8` | 1 | uint8 | RGB pixels, raw bytes |
| `bool` | 1 | bool_ | Done/termination flags |

*bf16 uses `ml_dtypes.bfloat16` when available, falls back to `uint16` to preserve byte layout.

---

## 3. The Problem It Solves — And Why Nobody Else Has

### The Trajectory Data Problem

Every world model, every RL agent, every robot learning system consumes the same thing: episodes. Sequences of (observation, action, reward, done) tuples collected from environments. The datasets are large (DROID: 76K demonstrations, 2TB+), multi-modal (cameras + joint states + force/torque + audio), and heterogeneous (different robots, different environments, different sampling rates).

There is no small neutral format focused specifically on **one-file tensor episodes plus world-model prediction lanes**. Several mature ecosystems exist for adjacent problems — each is the right tool for a different shape of data:

| When you reach for… | What it is great at | Where WShard fits differently |
|---|---|---|
| **HDF5** | Mature scientific array storage; SWMR for one-writer/many-reader | Heavier than we wanted for self-contained per-episode files; no semantic lanes for predictions/uncertainty |
| **RLDS / TFDS** | Standardized RL datasets in the TensorFlow ecosystem | We want a small file format, no TF dependency, with omen/uncert/residual lanes built in |
| **LeRobot (Parquet + MP4)** | Hugging Face Hub-hosted robotics datasets, great for publishing | We target local training-time and runtime episode files with named tensor blocks |
| **NPZ (DreamerV3)** | Quick numpy dumps | We add metadata, per-block compression, cross-language readers, and integrity checks |
| **MCAP (Foxglove)** | Timestamped pub/sub robot logs | We are tensor/episode-first rather than message-log-first |
| **Zarr** | Chunked N-dimensional arrays on cloud / object storage | We are single-episode-file oriented; a Zarr export bridge is on the roadmap |
| **Custom binary** | Whatever a lab needs locally | We try to keep the spec small enough to be the boring default |

The bet is that episode-shaped data — observations, actions, rewards, plus model predictions — deserves a small specialized container. If the user feedback says "bridges to existing formats are more valuable than the format itself," we will follow that.

### What's Actually Different About WShard

**1. Per-block compression in a flat binary.**

An episode with 5 camera streams and 20 scalar channels shouldn't compress everything the same way. WShard lets you zstd the video at level 19 and leave the 40-byte reward vector uncompressed. Each block carries its own compression flag — the reader auto-detects from the index entry bits.

```python
# Write with per-block compression control
writer = WShardStreamWriter(path, "ep_001", channels=[
    {"id": "rgb", "dtype": "u8", "shape": [84, 84, 3], "compression": "zstd"},
    {"id": "joint", "dtype": "f32", "shape": [7]},  # no compression needed
])
```

**2. Streaming append with crash safety.**

Robot data collection runs for hours. If the process crashes at minute 47, you lose everything with HDF5 (corrupted file). WShard's streaming writer uses a reserve-write-finalize pattern:

1. Write to `episode.wshard.partial`
2. Reserve space for the header (rewritten at finalization)
3. Append timesteps incrementally
4. On success: atomic `rename()` to `episode.wshard`
5. On crash: `.partial` file is deleted or identifiable as incomplete

```python
with WShardStreamWriter(path, "ep_001", channels) as w:
    w.begin_episode()
    for t in range(T):
        w.write_timestep({"state": obs, "ctrl": act}, reward=r, done=d)
    w.end_episode()  # atomic finalize
```

**3. Cross-language without serialization overhead.**

The same binary file is readable from Go, Python, and TypeScript. No protobuf compilation step. No schema registry. No code generation.

- Go robots write `.wshard` files during data collection
- Python training scripts read them directly with `load_wshard()`
- TypeScript dashboards visualize episodes in the browser

All three implementations agree on CRC32C checksums (`0x9a71bb4c` for "hello"), xxHash64 name hashes, dtype sizes, and block layout. This is verified by golden file tests: Go generates `.wshard` files, Python and TypeScript read them and assert byte-level correctness.

**4. Chunked episodes for distributed training.**

A 10-minute manipulation episode at 30Hz with 3 cameras is ~2GB. You don't want that as a single file on a networked filesystem. WShard splits episodes into chunks with a manifest shard that tracks continuity:

```python
writer = ChunkedEpisodeWriter("data/ep_001", "ep_001", chunk_size_t=1000)
for chunk in chunks:
    writer.write_chunk(chunk)
manifest = writer.finalize_manifest()

# Validation catches gaps, duplicates, and discontinuities
validate_chunk_continuity(manifest)
```

Each chunk is a standalone `.wshard` file with `chunk_index`, `total_chunks`, and `timestep_range` in its metadata. The manifest shard (role=0x04) ties them together.

**5. Semantic lanes for model training.**

WShard doesn't just store raw data. It has dedicated namespaces for the artifacts of model training:

- **omen/** — Model predictions stored alongside ground truth for comparison
- **uncert/** — Uncertainty estimates (ensemble variance, dropout entropy)
- **residual/** — Sign2ndDiff compressed deltas between prediction and truth

This means a single `.wshard` file can contain the ground truth episode AND the model's predictions about that episode, enabling offline evaluation without joins across separate files.

### What WShard Deliberately Does Not Do

- **No server.** It's a file format. Put the files on S3, NFS, local SSD — doesn't matter.
- **No training loop.** Use PyTorch DataLoader, JAX data pipeline, whatever you want.
- **No schema enforcement.** Blocks are named byte arrays. The convention layer is advisory.
- **No video codec.** Camera data is stored as raw tensors. If you want H.264, encode before writing.
- **No query language.** For search-by-similarity, use the DeepData bridge (see below).

---

## 4. Cross-Language Interop — One Format, Three Runtimes

### The Parity Problem

Cross-language file formats sound simple. They are not. The wshard codebase had three critical interop bugs that would have caused silent data corruption in production:

| Bug | What Happened | Impact |
|-----|--------------|--------|
| CRC32 IEEE vs Castagnoli | Python/TS used polynomial `0xEDB88320`. Go used `0x82F63B78`. | Every checksum validation fails silently or rejects valid files |
| FNV-1a vs xxHash64 | Python/TS used inline FNV-1a hash. Go used `xxhash.Sum64String()`. | O(1) name lookup returns wrong block or misses entirely |
| bf16 → float32 reinterpret | Python mapped bf16 (2 bytes) to numpy float32 (4 bytes). | Every bf16 tensor has wrong shape and corrupted values |

These are exactly the bugs that unit tests don't catch. Each implementation passes its own tests. The bug only appears when Go writes a file and Python reads it. This is Class 6 (Cross-Language Parity Drift) from the CoGS testing philosophy.

### How We Fixed It

**Golden file testing.** A standalone Go program (`golden/generate.go`) writes three `.wshard` files using the authoritative Go shard implementation:

| File | Purpose |
|------|---------|
| `simple_episode.wshard` | Basic episode: obs [T,4], actions [T,2], reward, done |
| `dtype_zoo.wshard` | All 13 dtypes exercised |
| `per_block_compressed.wshard` | Zstd-compressed blocks with 100 timesteps |
| `omen_uncert.wshard` | Omen predictions, uncertainty estimates, sign2nddiff residuals |
| `multimodal.wshard` | Multi-modal observations (RGB + proprioception groups) |
| `latent_action.wshard` | Latent action embeddings and codebook indices |

Python and TypeScript tests read these files and assert:
- CRC32C checksums match `golden_hashes.json`
- xxHash64 of known strings match
- Dtype sizes match
- Episode metadata (id, env_id, length) parses correctly
- Tensor shapes and values are correct
- Compressed blocks decompress correctly

```python
# From test_interop.py — golden file parity test
def test_golden_simple_episode_loads():
    ep = load_wshard(GOLDEN_DIR / "simple_episode.wshard")
    assert ep.id == "golden_simple"
    assert ep.env_id == "TestEnv-v1"
    assert ep.length == 10
    assert ep.observations["state"].data.shape == (10, 4)
    assert ep.actions["ctrl"].data.shape == (10, 2)
    np.testing.assert_allclose(ep.observations["state"].data[0], [0, 1, 2, 3])
```

### Hash and Checksum Alignment

All three implementations now produce identical outputs for reference inputs:

```
CRC32C("hello")       = 0x9a71bb4c  (Go, Python, TypeScript)
xxHash64("signal/obs") = 0x86f8c8413116a0ae
xxHash64("meta/manifest") = 0x9a191dcd325813d3
```

These values are committed in `golden/golden_hashes.json` and verified by CI.

### Implementation Details by Language

**Python** uses `crc32c` (C extension, hardware-accelerated on x86/ARM) and `xxhash` (C extension wrapping xxHash):

```python
import crc32c
import xxhash

def compute_crc32(data: bytes) -> int:
    return crc32c.crc32c(data)

def name_hash(name: str) -> int:
    return xxhash.xxh64(name.encode("utf-8")).intdigest()
```

**TypeScript** uses a pure-JS CRC32C table and `xxhash-wasm` (WebAssembly):

```typescript
// CRC32C with Castagnoli polynomial
const CRC32C_TABLE = makeCrc32Table(0x82f63b78);

export function crc32C(data: Uint8Array): number { ... }

// xxHash64 via WASM (async init)
import xxhashWasm from 'xxhash-wasm';
let _xxh64: ((s: string) => bigint) | null = null;

export async function initXxHash(): Promise<void> {
    const hasher = await xxhashWasm();
    _xxh64 = (s) => hasher.h64(s);
}
```

**Go** uses the standard library and `cespare/xxhash/v2`:

```go
var crc32cTable = crc32.MakeTable(crc32.Castagnoli)

func computeChecksum(data []byte) uint32 {
    return crc32.Checksum(data, crc32cTable)
}

func nameHash(name string) uint64 {
    return xxhash.Sum64String(name)
}
```

---

## 5. DeepData Bridge — Similarity Search Over Episodes

WShard files stay on disk as the authoritative store. The `deepdata_bridge` module indexes episode metadata and observation embeddings into DeepData (a vector database) so callers can retrieve episodes by behavioural similarity:

```python
from wshard.deepdata_bridge import TrajectoryIngestor, TrajectoryRetriever

# Index episodes
ingestor = TrajectoryIngestor("http://deepdata:8080", embedder=my_embedder)
ingestor.ingest_episode("episodes/ep_001.wshard")

# Search by behavioral similarity
retriever = TrajectoryRetriever("http://deepdata:8080", embedder=my_embedder)
results = retriever.search_similar_episodes(
    query_obs=current_observation,
    top_k=10,
    env_id="ManipulationEnv-v2",
    min_length=100,
    reward_range=(0.8, 1.0),
)
# Returns EpisodeRef(episode_id, file_path, score)
# Caller loads wshard file directly for data access
```

Hits return episode references; the caller reads the `.wshard` file directly for bulk data.

---

## Appendix: File Locations

| Item | Path |
|------|------|
| Python package | `py/wshard/` |
| TypeScript package | `js/src/` |
| Go shard package | `go/shard/` |
| Golden fixtures | `golden/` |
| Python tests | `py/tests/` |
| TypeScript tests | `js/tests/` |
| Go tests | `go/shard/*_test.go` |
| DeepData bridge | `py/wshard/deepdata_bridge.py` |

## Appendix: Dependency Map

**Python** (`pyproject.toml`):
- `numpy>=1.20` — Array operations
- `crc32c>=2.3` — Hardware-accelerated CRC32C
- `xxhash>=3.0` — xxHash64 name hashing
- `zstandard>=0.21.0` — Zstd compression
- `lz4>=4.0.0` — LZ4 compression
- Optional: `ml-dtypes>=0.3` (bf16), `h5py>=3.0` (HDF5 import), `torch>=2.0` (PyTorch tensors)

**TypeScript** (`package.json`):
- `@bokuweb/zstd-wasm` — Zstd compression via WebAssembly
- `fflate` — Deflate/LZ4 compression
- `xxhash-wasm` — xxHash64 via WebAssembly

**Go** (`go.mod`):
- `github.com/cespare/xxhash/v2` — xxHash64
- `github.com/klauspost/compress` — Zstd and LZ4
- Standard library `hash/crc32` — CRC32C (Castagnoli)
