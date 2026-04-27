# WShard

Binary file format for world model episode data. One file per episode. Cross-language. Zero-copy reads.

```
episode.wshard
├── meta/episode     → {"episode_id": "ep_001", "env_id": "Manip-v2", "length_T": 500}
├── signal/rgb       → [500, 84, 84, 3] uint8     (zstd compressed)
├── signal/joint_pos → [500, 7] float32            (uncompressed, 32-byte aligned)
├── action/ctrl      → [500, 7] float32
├── reward           → [500] float32
└── done             → [500] bool
```

## Why

Trajectory data from robots, simulators, and RL environments has no standard format. Teams use HDF5 (single-writer, no streaming), NPZ (no metadata, no cross-language), RLDS (TensorFlow lock-in), or custom formats that break on the next project.

WShard is a flat binary with:

- **Per-block compression** — zstd the video, leave the reward vector raw
- **O(1) block lookup** — xxHash64 index, no deserialization to find one channel
- **32-byte aligned data** — `mmap()` + pointer cast for zero-copy reads
- **CRC32C checksums** — hardware-accelerated integrity verification
- **Streaming append** — crash-safe `.partial` file pattern for live recording
- **Chunked episodes** — split long episodes across files with continuity validation

## Install

```bash
# Python
pip install wshard

# TypeScript
npm install @wshard/core

# Go
import "github.com/Neumenon/shard/go/shard"
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
ep.rewards = Channel(name="reward", dtype=DType.FLOAT32, data=np.zeros(100, dtype=np.float32))
save_wshard(ep, "episode.wshard", compression="zstd")
```

### Streaming (live recording)

```python
from wshard.streaming import WShardStreamWriter

channels = [
    {"id": "joint_pos", "dtype": "f32", "shape": [7]},
    {"id": "ctrl", "dtype": "f32", "shape": [7]},
]

with WShardStreamWriter("recording.wshard", "ep_001", channels) as w:
    w.begin_episode()
    for t in range(1000):
        obs, reward, done = env.step(action)
        w.write_timestep({"joint_pos": obs, "ctrl": action}, reward=reward, done=done)
    w.end_episode()  # atomic rename from .partial
```

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
import "github.com/Neumenon/shard/go/shard"

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

## Format

Shard container (role `0x05`). 64-byte header, 48-byte index entries, aligned data blocks.

```
[Header 64B] [Index N×48B] [String Table] [Padding] [Data Blocks...]
```

Block names are hierarchical paths:

| Prefix | Content |
|--------|---------|
| `meta/` | JSON metadata (`meta/wshard`, `meta/episode`, `meta/channels`) |
| `signal/` | Observation tensors |
| `action/` | Action tensors |
| `omen/` | Model prediction tensors |
| `uncert/` | Uncertainty estimates |
| `residual/` | Compressed residual encodings |
| `reward` | Reward signal (float32) |
| `done` | Termination flags (bool) |

13 data types: `f32`, `f64`, `f16`, `bf16`, `i64`, `i32`, `i16`, `i8`, `u64`, `u32`, `u16`, `u8`, `bool`.

Compression: none, zstd, or lz4 — per block. Checksums: CRC32C (Castagnoli).

## Features

### Chunked Episodes

Split long episodes across multiple files:

```python
from wshard.chunked import ChunkedEpisodeWriter, validate_chunk_continuity

writer = ChunkedEpisodeWriter("data/ep_001", "ep_001", chunk_size_t=1000)
for chunk in episode_chunks:
    writer.write_chunk(chunk)
manifest = writer.finalize_manifest()
validate_chunk_continuity(manifest)  # catches gaps, duplicates, discontinuities
```

### Multi-Modal Observations (VLA)

```python
from wshard import add_multimodal_observation, get_multimodal_observations
from wshard.types import Modality

add_multimodal_observation(ep, "camera_0", Modality.RGB, rgb_channel)
add_multimodal_observation(ep, "camera_0", Modality.DEPTH, depth_channel)
add_multimodal_observation(ep, "wrist", Modality.PROPRIOCEPTION, joint_channel)

rgb_channels = get_multimodal_observations(ep, modality=Modality.RGB)
```

### Residual Compression

```python
from wshard.residual import compute_sign2nd_diff, pack_residual_bits, unpack_residual_bits

residual = compute_sign2nd_diff(signal)  # {-1, 0, +1} array
packed = pack_residual_bits(residual)     # 2 bits per element
```

### DeepData Bridge (Trajectory Search)

```python
from wshard.deepdata_bridge import TrajectoryIngestor, TrajectoryRetriever

ingestor = TrajectoryIngestor("http://deepdata:8080", embedder=encoder)
ingestor.ingest_episode("episodes/ep_001.wshard")

retriever = TrajectoryRetriever("http://deepdata:8080", embedder=encoder)
similar = retriever.search_similar_episodes(
    query_obs=observation, top_k=10, env_id="Manip-v2"
)
```

### Format Conversion

```python
from wshard import load, save

ep = load("dreamer_episode.npz")     # auto-detect: DreamerV3
save(ep, "episode.wshard")           # convert to WShard

ep = load("episode.wshard")
save(ep, "episode.npz")             # convert back
```

## Cross-Language Parity

All three implementations produce identical binary output for the same input. Verified by golden file tests:

- Go generates reference `.wshard` files (`golden/generate.go`)
- Python and TypeScript read them and assert byte-level correctness
- CRC32C, xxHash64, dtype sizes, and block layout are tested against committed reference values (`golden/golden_hashes.json`)

```
CRC32C("hello")          = 0x9a71bb4c
xxHash64("signal/obs")   = 0x86f8c8413116a0ae
```

## Testing

```bash
# Python (103 tests)
cd wshard/py && python -m pytest tests/ -v

# TypeScript (15 tests)
cd wshard/js && npm test

# Go — wshard lives inside the full shard package
cd go && go test ./shard/ -v
```

## Dependencies

| Language | Core Dependencies |
|----------|------------------|
| Python | numpy, crc32c, xxhash, zstandard, lz4 |
| TypeScript | @bokuweb/zstd-wasm, fflate, xxhash-wasm |
| Go | cespare/xxhash/v2, klauspost/compress, stdlib crc32 |

Optional Python: `ml-dtypes` (bf16), `h5py` (HDF5 import), `torch` (PyTorch tensors)

## Docs

- [Deep Dive](docs/DEEP_DIVE.md) — Format specification, market analysis, architecture decisions
- [Marketing Brief](docs/MARKETING_BRIEF.md) — Positioning, messaging, competitive landscape

## License

MIT
