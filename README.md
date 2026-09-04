# WShard

One-file episodes for robots, RL, and world models.

A `.wshard` file stores one episode: observations, actions, rewards, done
flags, metadata, and optional model prediction / uncertainty / residual
lanes. Each block is independently addressable and independently
compressed.

```
episode.wshard
├── meta/episode      → {"episode_id": "ep_001", "env_id": "Manip-v2", "length_T": 500}
├── signal/rgb        → [500, 84, 84, 3] uint8     (zstd compressed)
├── signal/joint_pos  → [500, 7] float32           (uncompressed, 32-byte aligned)
├── action/ctrl       → [500, 7] float32
├── reward            → [500] float32
├── done              → [500] bool
├── omen/joint_pos/dreamer    → [500, 7] float32   (model predictions)
├── uncert/joint_pos/std      → [500, 7] float32   (uncertainty)
└── residual/joint_pos/sign2nddiff → packed bits   (residual encoding)
```

## Try it in 60 seconds

```bash
git clone https://github.com/phenomenon0/wshard
cd wshard
pip install -e "./py[dev]"

python examples/write_cartpole.py
python examples/read_cartpole.py
wshard inspect examples/cartpole.wshard      # CLI: also `wshard verify`, `convert`, `export`, `doctor`
```

## Why WShard?

Robotics and RL trajectory data has several good ecosystems already, but no
small neutral file format focused specifically on **self-contained tensor
episodes plus world-model prediction lanes**. WShard tries to fill that
narrow gap:

- **One file = one episode.** Move it, archive it, mmap it, ship it across a
  network — the file is the unit of work.
- **Per-block compression.** Zstd the video, leave the reward vector raw.
- **Fast name lookup.** Index entries carry an xxHash64 of the block name, so
  finding `signal/joint_pos` is a quick scan over fixed-size index slots,
  no string-table parse, no full deserialization.
- **32-byte aligned data.** Read with `mmap()` + pointer cast for low-copy
  reads.
- **CRC32C checksums.** Hardware-accelerated integrity per block.
- **Streaming writer.** Record a live loop timestep by timestep and publish
  once: `end_episode()` writes the episode, seals it, and renames `.partial`
  into place. Nothing partial is ever visible — and nothing is durable before
  that call. For incremental durability, write chunk files (see below).
- **Cross-language.** Python, TypeScript, and Go readers/writers from one
  spec.

## What WShard is not

WShard is not a database. It is not a training framework. It is not a
replacement for [LeRobot][lerobot], [MCAP][mcap], [Zarr][zarr], [HDF5][hdf5],
or Parquet.

| When you want… | Reach for… |
|---|---|
| Hugging Face Hub-hosted robotics datasets | [LeRobot][lerobot] (Parquet + MP4) |
| Timestamped pub/sub robot logs | [MCAP][mcap] |
| Chunked N-dimensional arrays on cloud storage | [Zarr][zarr] |
| Mature scientific array storage | [HDF5][hdf5] |
| Self-contained episode files with named tensor blocks **and** world-model lanes | WShard |

WShard complements these formats. We provide bridges where it makes sense.

[lerobot]: https://github.com/huggingface/lerobot
[mcap]: https://mcap.dev/
[zarr]: https://zarr.dev/
[hdf5]: https://www.hdfgroup.org/solutions/hdf5/

## Install

These are git-install commands. Registry packages on PyPI / npm will follow
once the format and API stabilize.

```bash
# Python
pip install "git+https://github.com/phenomenon0/wshard.git#subdirectory=py"

# TypeScript (clone + build is the most reliable path today)
git clone https://github.com/phenomenon0/wshard
cd wshard/js && npm ci && npm run build

# Go
go get github.com/phenomenon0/wshard/go@latest
```

## Quick Start

### Python

```python
from wshard import load_wshard, save_wshard
from wshard.types import Episode, Channel, DType
import numpy as np

# Read
ep = load_wshard("episode.wshard")
print(ep.id, ep.env_id, ep.length)
obs = ep.observations["joint_pos"].data   # np.ndarray [T, 7]
act = ep.actions["ctrl"].data             # np.ndarray [T, 7]

# Write
ep = Episode(id="ep_001", length=100)
ep.env_id = "ManipulationEnv-v2"
ep.observations["joint_pos"] = Channel(
    name="joint_pos", dtype=DType.FLOAT32,
    shape=[7], data=np.random.randn(100, 7).astype(np.float32),
)
ep.actions["ctrl"] = Channel(
    name="ctrl", dtype=DType.FLOAT32,
    shape=[7], data=np.random.randn(100, 7).astype(np.float32),
)
ep.rewards = Channel(
    name="reward", dtype=DType.FLOAT32,
    shape=[], data=np.zeros(100, dtype=np.float32),
)
save_wshard(ep, "episode.wshard", compression="zstd")
```

### Streaming (live recording)

```python
from wshard.streaming import ChannelDef, WShardStreamWriter
from wshard.types import DType

channels = [
    ChannelDef("joint_pos", DType.FLOAT32, [7]),
    ChannelDef("ctrl", DType.FLOAT32, [7]),
]

with WShardStreamWriter("recording.wshard", "ep_001", channels) as w:
    w.begin_episode(env_id="ManipulationEnv-v2")
    for t in range(1000):
        obs, reward, done = env.step(action)
        w.write_timestep(t, {"joint_pos": obs}, {"ctrl": action}, reward, done)
    w.end_episode()  # write, seal with meta/identity, rename off .partial
```

`write_timestep` buffers; the file gets its bytes at `end_episode()`. The index
describes each block as one extent, so a block can only be written once — which
is why the writer keeps the episode in RAM (bounded by `max_timesteps`) and
passing `flush_interval` is an error rather than a no-op. Episodes too long for
memory want chunk files.

### TypeScript

```typescript
import { WShardWriter, WShardReader } from '@wshard/core';

// Write
const writer = new WShardWriter('episode.wshard', { compression: 'lz4' });
writer
  .setEpisode({ episode_id: 'ep_001', length_t: 100, env_id: 'MyEnv-v0' })
  .addChannel({ id: 'state', dtype: 'f32', shape: [4] })
  .setSignalFloat32_2D('state', observations)
  .setActionFloat32_2D('ctrl', actions)
  .setReward(rewards)
  .setDone(doneFlags);
await writer.write();

// Read
const reader = new WShardReader('episode.wshard');
await reader.open();
const obs = await reader.getSignalFloat32_2D('state', 4);
const act = await reader.getActionFloat32_2D('ctrl', 2);
await reader.close();
```

### Go

```go
import "github.com/phenomenon0/wshard/go/shard"

// Write
ep := &shard.WShardEpisode{
    ID:      "ep_001",
    EnvID:   "ManipulationEnv-v2",
    LengthT: 100,
    Observations: map[string]*shard.WShardChannel{
        "joint_pos": {Name: "joint_pos", DType: "f32", Shape: []int{7}, Data: jointBytes},
    },
    Actions: map[string]*shard.WShardChannel{
        "ctrl": {Name: "ctrl", DType: "f32", Shape: []int{7}, Data: ctrlBytes},
    },
    Rewards: rewards,
}
shard.CreateWShard("episode.wshard", ep)

// Read
ep, err := shard.OpenWShard("episode.wshard")
```

## Performance

Go reads a T=1000 episode (~21 MB raw) in ~5 ms (4.1 GB/s, memory-bandwidth limited).
Python reads in ~5.6 ms and writes in ~43 ms uncompressed; zstd costs ~165 ms to
write and ~29 ms to read, for a 1.97× smaller file.
Header + index lookup alone takes ~8 µs in Go, and 9.8 µs per file in Python
measured across 1000 real episode files — 100,000+ index scans per second.

Writes include sealing: the writer sha256s every block to build `meta/identity`,
which is one extra pass over the payload (visible mostly on the uncompressed path).

See [`bench/`](bench/) for write/read benchmarks across compression types vs NumPy NPZ.

## Format

64-byte header, 48-byte index entries, aligned data blocks. Built on the
Shard container (role `0x05`).

```
[Header 64B] [Index N×48B] [String Table] [Padding] [Data Blocks...]
```

Block names are hierarchical paths:

| Prefix | Content |
|--------|---------|
| `meta/` | JSON metadata (`meta/wshard`, `meta/episode`, `meta/channels`, `meta/identity`, `meta/provenance`) |
| `signal/` | Observation tensors |
| `action/` | Action tensors |
| `time/` | Timestamps (`time/ticks`, `time/timestamps_ns`) |
| `omen/` | Model prediction tensors |
| `uncert/` | Uncertainty estimates |
| `residual/` | Compressed residual encodings |
| `reward` | Reward signal (float32) |
| `done` | Termination flags (bool) |

13 data types: `f32`, `f64`, `f16`, `bf16`, `i64`, `i32`, `i16`, `i8`, `u64`, `u32`, `u16`, `u8`, `bool`.

Compression: none, zstd, or lz4 — per block. Checksums: CRC32C (Castagnoli).

Identity: `meta/identity` is written last and holds
`{"entries":{"<block>":"<sha256 of the block's uncompressed bytes>",…},"leaf":"sha256","v":1}`
as canonical JSON (glyph SPEC-CANON §4). The file's
identity is `sha256` of that block — `episode_identity(path)` in Python,
`EpisodeIdentity(path)` in Go: identical across none/zstd/lz4 layouts, different
for any content change. `verify_identity` / `VerifyIdentity` re-hash every block
against it; CRC32C only proves a block matches its own index slot, which whoever
edits the file can rewrite. The streaming and chunked writers seal what they
write the same way: `end_episode(provenance=…)` in Python, `SetProvenance`
before `EndEpisode()` in Go.

Provenance: `meta/provenance` is optional and says where an episode came from —

```json
{"v":1,"run_id":"…","epoch":7,"first_seq":10001,"last_seq":20000,
 "start_state":"<64hex>","end_state":"<64hex>","prev_identity":"<64hex>",
 "source":{"git_commit":"…","schema_fp":"…"}}
```

also canonical JSON, and written immediately *before* `meta/identity` so the
identity commits to it: rewriting which run produced an episode breaks
`verify_identity`, not just a checksum anyone could recompute. `prev_identity`
is the `episode_identity` of the preceding episode of the same run, which makes
a set of chunks a chain rather than a set — chunk N names the exact bytes of
chunk N-1, so a missing or swapped chunk is detectable without trusting the
manifest that lists them. Read it without decoding tensors via
`episode_provenance(path)` in Python, `EpisodeProvenance(path)` in Go. Every
field is always written, even when empty: an absent key and a key holding `""`
are different canonical JSON, so omitting one would make two equal provenances
hash differently across producers.

See [docs/DEEP_DIVE.md](docs/DEEP_DIVE.md) for the byte-level spec.

## Cross-language parity

All three implementations produce identical binary output for the same
input. Verified by golden-file tests:

- Go generates reference `.wshard` files (`golden/generate.go`)
- Python and TypeScript read them and assert byte-level correctness
- CRC32C, xxHash64, dtype sizes, and block layout are checked against
  committed reference values (`golden/golden_hashes.json`)
- `meta/identity` in every golden file is re-derived by Python `verify_identity`
  and Go `VerifyIdentity` and matched to `identity_<file>` in the same JSON

```
CRC32C("hello")          = 0x9a71bb4c
xxHash64("signal/obs")   = 0x86f8c8413116a0ae
```

## Testing

```bash
# Python
cd py && pytest tests/ -v

# TypeScript
cd js && npm test

# Go
cd go && go test ./shard/...

# Regenerate golden fixtures (rare; CI checks for drift)
cd golden && go run generate.go
```

## Status

**Beta.** The format is stable enough to try, but we are still looking for
feedback on schemas, video blocks, and converters. The Show HN launch is
intentional — the goal is to find sharp edges before declaring 1.0.

## Known limitations

- **No native video block type.** Camera data is stored as raw tensors. H.264
  / H.265 / AV1 block types are future work.
- **No Arrow / Polars / DuckDB integration** yet.
- **No formal schema registry** — block-name conventions are advisory.
- **Cloud / object-store behavior** has not been stress-tested.
- **Few external users** so far. The launch is the bug-finding pass.

## Features

### Chunked Episodes

Split long episodes across multiple files:

```python
from wshard.chunked import (
    ChunkedEpisodeWriter, validate_chunk_chain, validate_chunk_continuity,
)

writer = ChunkedEpisodeWriter("data/ep_001", "ep_001", chunk_size_t=1000)
writer.write_episode_chunked(long_episode)   # slices and writes every chunk
manifest_path = writer.finalize_manifest()   # returns the path it wrote

validate_chunk_continuity(writer.manifest)            # gaps, duplicates, ranges
validate_chunk_chain(writer.manifest, "data/ep_001")  # re-reads and re-links chunks
```

`validate_chunk_continuity` reads only the manifest. `validate_chunk_chain` opens
each chunk and follows its `prev_identity` back, so a swapped, edited, or missing
chunk is caught even if the manifest was rewritten to match.

### Multi-modal observations

```python
from wshard import add_multimodal_observation, get_multimodal_observations
from wshard.types import Modality

add_multimodal_observation(ep, "camera_0", Modality.RGB, rgb_channel)
add_multimodal_observation(ep, "camera_0", Modality.DEPTH, depth_channel)
add_multimodal_observation(ep, "wrist", Modality.PROPRIOCEPTION, joint_channel)

rgb_channels = get_multimodal_observations(ep, modality=Modality.RGB)
```

### Residual compression

```python
from wshard.residual import compute_sign2nd_diff, pack_residual_bits, unpack_residual_bits

residual = compute_sign2nd_diff(signal)   # 1-D signal -> {-1, 0, +1} array
packed = pack_residual_bits(residual)     # 1 bit per element
```

`compute_sign2nd_diff` takes a 1-D signal; use `compute_sign2nd_diff_multidim`
for `[T, D]`. The bit packing is lossy on purpose: it keeps one bit per element,
so `0` and `-1` both unpack as `-1`. Use it as a coarse side-channel, not as a
reconstruction of the signal.

### Format conversion (experimental)

Converters for DreamerV3 NPZ, Minari, and D4RL exist in `wshard.convert`.
The DreamerV3 NPZ path (`load_dreamer` / `save_dreamer`) is now covered by
integration tests (`py/tests/test_dreamer_roundtrip.py`) and verified
byte-identical on synthetic fixtures. Minari and D4RL converters are **not
yet covered by integration tests against real fixtures from those frameworks**;
treat those paths as experimental and report breakage:

```python
from wshard import load, save

ep = load("dreamer_episode.npz")    # auto-detect: DreamerV3
save(ep, "episode.wshard")          # convert to WShard
```

## Dependencies

| Language | Core Dependencies |
|----------|------------------|
| Python | numpy, crc32c, xxhash, zstandard, lz4 |
| TypeScript | @bokuweb/zstd-wasm, fflate, xxhash-wasm (optional) |
| Go | cespare/xxhash/v2, klauspost/compress, stdlib crc32 |

Optional Python: `ml-dtypes` (bf16), `h5py` (HDF5 import), `torch` (PyTorch tensors).

## Contributing

Open issues for broken readers, bad benchmarks, malformed files, or
converter requests. The most useful feedback right now:

1. What format do you use today?
2. Where would WShard break in your pipeline?
3. What benchmark would make this credible?
4. Which converter should exist first: LeRobot, MCAP, RLDS, Minari, D4RL, or Zarr?

## CLI

After `pip install`, the `wshard` script is on PATH:

```bash
wshard inspect episode.wshard          # block list + dtype + shape + size
wshard verify episode.wshard           # CRC32C check every block (not meta/identity —
                                       # use verify_identity / VerifyIdentity for that)
wshard convert input.npz output.wshard # auto-detect (DreamerV3 NPZ today)
wshard export episode.wshard --format dreamer
wshard doctor                          # version + dependency check
```

`python -m wshard ...` also works.

## Web viewer

A small browser-only viewer for `.wshard` files lives in [`viewer/`](viewer/).
Drag a file in to see episode metadata + the block list (no decoding, no upload —
parsing happens client-side). Build: `cd viewer && npm install && npm run build`.

## Docs

- [Format at a glance](docs/FORMAT.md) — header, index, namespaces in 2 minutes.
- [Deep Dive](docs/DEEP_DIVE.md) — full byte-level spec.
- [FAQ](docs/FAQ.md) — common questions.
- [Why not HDF5?](docs/WHY_NOT_HDF5.md) — honest comparison.
- [Why not LeRobot?](docs/WHY_NOT_LEROBOT.md) — different layer, different problem.
- [Security](SECURITY.md) — threat model and reader hardening.
- [Benchmarks](bench/README.md) — write/read across formats and compressions.

## License

MIT — see [LICENSE](LICENSE).
