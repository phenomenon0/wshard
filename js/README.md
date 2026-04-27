# W-SHARD TypeScript Implementation

TypeScript implementation of the W-SHARD (World-Model Episode Shard) format for action-conditioned world modeling.

## Features

- Full Shard format support (role=0x05)
- LZ4 block compression (zstd via deflate fallback)
- Sign2ndDiff residual encoding
- Cross-language compatible with Go and Python implementations
- TypeScript-first with full type definitions

## Installation

```bash
npm install @wshard/core
```

## Usage

### Writing W-SHARD files

```typescript
import { WShardWriter } from '@wshard/core';

const writer = new WShardWriter('episode.wshard', { compression: 'lz4' });

writer
  .setEpisode({
    episode_id: 'ep_001',
    length_t: 100,
    source: 'simulation',
    env_id: 'MyEnv-v0',
    timebase: 'ticks',
    action_space: 'continuous',
  })
  .addChannel({
    id: 'state/pos',
    dtype: 'f32',
    shape: [3],
    unit: 'meters',
  })
  .setTimeTicks(ticks)
  .setActionFloat32_2D('main', actions)
  .setSignalFloat32_2D('state/pos', positions)
  .setReward(rewards)
  .setDone(doneFlags);

await writer.write();
```

### Reading W-SHARD files

```typescript
import { WShardReader } from '@wshard/core';

const reader = new WShardReader('episode.wshard');
await reader.open();

console.log('Episode ID:', reader.episodeMeta?.episode_id);
console.log('Timesteps:', reader.lengthT);
console.log('Channels:', reader.channelIDs());

const actions = await reader.getActionFloat32_2D('main', 4);
const signals = await reader.getSignalFloat32_2D('state/pos', 3);
const rewards = await reader.getReward();

await reader.close();
```

### Using compression

```typescript
import { Compressor } from '@wshard/core';

const compressor = new Compressor('lz4', 'default');
const compressed = compressor.compress(data);
const decompressed = compressor.decompress(compressed, originalSize);
compressor.close();
```

## Format

W-SHARD uses the Shard container format with role=0x05. Layout:

```
Header (64 bytes)
Index entries (entry_count * 48 bytes)
String table (variable)
[Padding to alignment]
Data section (aligned entries)
```

### Block naming conventions

| Namespace | Purpose | Example |
|-----------|---------|---------|
| `meta/` | Metadata blocks | `meta/wshard`, `meta/episode` |
| `time/` | Time representation | `time/ticks`, `time/timestamps_ns` |
| `action/` | Action streams | `action/main` |
| `signal/` | Ground truth observations | `signal/state/pos` |
| `omen/` | Model predictions | `omen/state/pos/dreamer` |
| `uncert/` | Uncertainty estimates | `uncert/state/pos/dreamer/std` |
| `residual/` | Residual encodings | `residual/state/pos/sign2nddiff` |

### Special blocks

- `reward` - Reward signal (float32)
- `done` - Episode termination flags (bool/uint8)

## Cross-language compatibility

W-SHARD files are binary-compatible across:
- Go: `go/shard/`
- Python: `wshard/py/wshard/`
- TypeScript: `wshard/js/src/` (this package)

## Testing

```bash
npm test
```

## License

MIT
