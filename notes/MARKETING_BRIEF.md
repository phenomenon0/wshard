# WShard — Marketing Brief

## One-Liner

WShard is the binary file format for trajectory data — episodes from robots, simulators, and RL environments stored as flat files with per-block compression, cross-language readers, and zero-copy aligned data.

## Positioning Statement

For robotics engineers, RL researchers, and world model teams who collect, store, and train on sequential decision-making data, WShard is a binary episode format that replaces the patchwork of HDF5, NPZ, RLDS, and custom formats with a single cross-language file that handles multi-modal observations, per-block compression, streaming recording, and chunked episodes in one container.

Unlike HDF5 (single-writer, no streaming, Python-centric), RLDS (TensorFlow lock-in, rigid schema), LeRobot Parquet+MP4 (publishing format, not training-time), or lab-specific custom formats (no interop), WShard is a flat binary with O(1) block lookup, crash-safe streaming append, and verified cross-language parity between Go, Python, and TypeScript.

---

## The Problem

### Every lab builds their own trajectory format. Every format has the same gaps.

Teams collecting robot demonstrations, running RL experiments, or generating synthetic data from simulators all produce the same thing: time-indexed sequences of observations, actions, rewards, and termination signals. These datasets are:

- **Large** — DROID: 76K episodes, 2TB+. NVIDIA DreamGen: 100K+ synthetic trajectories. Individual episodes range 1MB to 400MB depending on cameras and resolution.
- **Multi-modal** — Joint positions (float32), RGB images (uint8), depth maps (uint16), force/torque (float32), audio (int16), language instructions (variable-length) — all at different sampling rates.
- **Cross-platform** — Robots run Go or C++. Training runs Python. Dashboards run JavaScript. The data format must work across all three without code generation or schema compilation.

There is no standard format. The current landscape:

| Format | What It Gets Right | Where It Breaks |
|--------|--------------------|-----------------|
| HDF5 | Proven, handles large arrays | Single-writer lock. No streaming append. Corrupt on crash. Python-centric in practice. |
| NPZ (DreamerV3) | Simple, numpy-native | No metadata. No compression choice. No cross-language. No streaming. |
| RLDS/TFDS | Google-backed, typed schema | TensorFlow dependency. Rigid schema. No streaming. |
| Parquet + MP4 (LeRobot v3) | Good for publishing to HuggingFace Hub | Two files per episode. Poor training-time random access. No per-block compression. |
| MCAP (Foxglove) | Multi-channel binary, ROS-native | Message-oriented, not tensor-oriented. No episode semantics. |
| Custom | Fits the team's exact needs | Breaks when the team changes. Zero interop. |

### The result

57% of organizations report their data isn't AI-ready. Trajectory data is worse. Teams spend engineering cycles on data pipeline plumbing instead of model architecture.

---

## The Solution

### WShard: one file, one episode, three languages.

A `.wshard` file is a self-contained episode with named, independently-addressable data blocks:

```
episode_42.wshard (1.8 MB)
├── meta/episode      8 KB   JSON metadata (id, env, length, timebase)
├── signal/rgb      1.2 MB   [500, 84, 84, 3] uint8   ← zstd compressed
├── signal/joints    14 KB   [500, 7] float32          ← uncompressed, aligned
├── action/ctrl      14 KB   [500, 7] float32
├── reward            2 KB   [500] float32
└── done              500 B  [500] bool
```

### Key technical differentiators

**1. Per-block compression.** Each block carries its own compression flag. Zstd the 1.2MB RGB block. Leave the 2KB reward vector uncompressed. The reader auto-detects from the 48-byte index entry — no header-level assumption.

**2. Zero-copy aligned reads.** Data blocks start at 32-byte boundaries. `mmap()` the file, cast the pointer, and you have an AVX-aligned numpy array. No deserialization. No copy.

**3. Crash-safe streaming.** The streaming writer uses a `.partial` file pattern: write incrementally, atomic `rename()` on success. If the robot process crashes at minute 47, the main file is untouched.

**4. Cross-language by construction.** The same binary file is read by Go, Python, and TypeScript. Not through protobuf stubs or generated code — through native implementations that share the same 64-byte header format, CRC32C polynomial, and xxHash64 name hashing. Verified by golden file tests where Go writes and Python/TypeScript read.

**5. Chunked episodes.** A 2GB manipulation episode splits into 1000-timestep chunks with a manifest shard tracking continuity. Each chunk is a standalone `.wshard` file. Continuity validation catches gaps, duplicates, and timestep discontinuities.

**6. Semantic lanes.** Named block prefixes (`signal/`, `action/`, `omen/`, `uncert/`, `residual/`) give meaning to raw data. A single file can hold ground truth observations alongside model predictions and uncertainty estimates — no separate prediction files, no join logic.

---

## Target Users

### Primary: Robotics data engineers

**Profile:** Building data pipelines for robot learning. Managing 10K-1M+ episodes. Dealing with multiple sensor modalities, variable episode lengths, and distributed training infrastructure.

**Pain:** HDF5 corruption on crash. Custom formats that break when the robot or env changes. No way to search episodes by behavior.

**WShard value:** Crash-safe streaming writes, per-block compression for mixed modalities, chunked episodes for distributed training, DeepData bridge for similarity search.

### Secondary: RL researchers

**Profile:** Running experiments with DreamerV3, TD-MPC2, or custom architectures. Outgrowing NPZ files. Need to compare predictions against ground truth.

**Pain:** NPZ has no metadata. No compression. No cross-language support. Predictions stored in separate files.

**WShard value:** Format conversion from NPZ/D4RL/Minari. Omen/uncertainty lanes for model comparison. Standard metadata schema.

### Tertiary: World model teams

**Profile:** Training video-prediction models. Generating synthetic trajectory data. Managing many episodes.

**Pain:** No standard format for synthetic trajectories. HDF5 doesn't scale. Custom formats prevent collaboration.

**WShard value:** Flat binary with O(1) lookup scales to any number of channels. Per-block compression handles heterogeneous data. Cross-language access for heterogeneous training stacks.

---

## Competitive Landscape

```
                    Tensor-native
                         ↑
                         │
              WShard ●   │
                         │
    Cross-     ──────────┼────────── Single-
    language             │          language
                         │
                ● MCAP   │  ● HDF5
                         │  ● NPZ
                         │  ● RLDS
                    Message-based
```

| | WShard | HDF5 | LeRobot v3 | RLDS | MCAP |
|---|---|---|---|---|---|
| Episode-native semantics | Yes | No | Partial | Yes | No |
| Per-block compression | Yes | Dataset-level | Format-level | No | Per-channel |
| Streaming append | Yes (.partial) | No | No | No | Yes |
| Cross-language | Go, Python, TS | Python (practical) | Python | Python (TF) | Many |
| Zero-copy reads | Yes (mmap, aligned) | Partial | No | No | Yes |
| Chunked episodes | Yes (manifest) | Manual | No | No | No |
| Model prediction lanes | Yes (omen/) | No | No | No | No |
| Similarity search | Yes (DeepData) | No | HF Hub search | No | No |
| Crash safety | .partial pattern | Corrupt on crash | N/A | N/A | Yes |

---

## Messaging Framework

### Headline

**The file format for physical AI data.**

### Sub-headlines (by audience)

- **Robotics:** Record robot episodes to crash-safe binary files. Read them from Python, Go, or TypeScript.
- **RL Research:** Replace NPZ with a format that has metadata, compression, and cross-language support.
- **World Models:** Store millions of multi-modal trajectory episodes with per-block compression and O(1) lookup.

### Key messages

1. **One file = one episode.** Not a database. Not a framework. A file format that works on local SSD, NFS, or S3.

2. **Three languages, one binary.** Go writes it during data collection. Python reads it during training. TypeScript renders it in the browser. All three implementations verified against golden reference files.

3. **Per-block compression.** Zstd the video. LZ4 the point clouds. Leave the scalars raw. Each block in the file carries its own compression flag.

4. **Crash-safe recording.** The streaming writer never corrupts your existing data. Atomic rename on success. Clean `.partial` deletion on failure.

5. **Search by behavior.** The DeepData bridge indexes episode embeddings for similarity search. "Find me the 100 episodes most similar to this failed grasp" is one API call.

### What we don't say

- No performance benchmarks against HDF5/Parquet (haven't run them formally yet)
- No "faster than X" claims without data
- No "enterprise-ready" or "production-grade" — let the test suite and format spec speak
- No "AI-native" — vague and overused

---

## Technical Proof Points

| Claim | Evidence |
|-------|---------|
| Cross-language parity | 164 tests across Go/Python/TS. Golden files written by Go, read by Python and TS. CRC32C and xxHash64 values match to the bit. |
| 13 data types including bf16 | `types.py:22-109`, `types.ts:156-188`, `dtype.go:3-17`. bf16 uses ml_dtypes when available, uint16 fallback. |
| Per-block compression | `compress.py` BLOCK_FLAG_ZSTD/LZ4 constants. `golden/per_block_compressed.wshard` test fixture. |
| Crash-safe streaming | `streaming.py` .partial pattern with atomic `os.replace()`. Test: `test_streaming_partial_deleted_on_error`. |
| Chunked episode continuity | `chunked.py:validate_chunk_continuity()` checks gaps, duplicates, timestep range, length sum. 6 test cases. |
| Format conversion | `convert.py:detect_format()` auto-detects DreamerV3, Minari, D4RL, WShard. |

---

## Open Questions

1. **Naming:** "WShard" works technically but isn't immediately descriptive. "WorldShard" is clearer. "Episode" is too generic. Current name is fine for the technical audience.

2. **Video codec support:** Currently stores camera data as raw tensors. For large-scale video datasets, native H.264/H.265 codec blocks would reduce storage significantly. Not yet implemented — but the per-block architecture supports it (just add a new content type).

3. **Arrow integration:** An Arrow reader that exposes wshard channels as Arrow arrays would enable zero-copy interop with Polars, DuckDB, and Spark. Not implemented, but architecturally straightforward.

4. **Schema registry:** Currently schema-free (blocks are named byte arrays). A formal schema definition (like Protobuf for episodes) would enable validation and code generation. Trade-off: flexibility vs safety.

---

## Assets

| Document | Location | Purpose |
|----------|----------|---------|
| Deep Dive (5 topics) | `docs/DEEP_DIVE.md` | Technical deep-dive for engineers evaluating the format |
| README | `README.md` | Quick start, API examples, format overview |
| Golden test fixtures | `golden/` | Cross-language verification files |
| Format spec | Inline in `DEEP_DIVE.md` Section 2 | Byte-level binary format documentation |
| Python API reference | `wshard/py/wshard/*.py` docstrings | Function-level documentation |
| TypeScript API reference | `wshard/js/src/*.ts` JSDoc | Function-level documentation |
