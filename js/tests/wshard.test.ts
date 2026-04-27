/**
 * W-SHARD tests.
 *
 * Tests for the TypeScript W-SHARD implementation.
 */

import { describe, it, expect, afterAll } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import {
  WShardWriter,
  WShardReader,
  SHARD_MAGIC,
  WSHARD_ROLE,
  SHARD_HEADER_SIZE,
  SHARD_INDEX_ENTRY_SIZE,
  crc32C,
  simpleHash64,
  encodeFloat32,
  encodeFloat32_2D,
  decodeFloat32,
  encodeInt32,
  decodeInt32,
  encodeBool,
  decodeBool,
  Compressor,
  parseIndexEntry,
} from '../src/index.js';

function readIndexEntryByName(filePath: string, targetName: string) {
  const file = fs.readFileSync(filePath);
  const entryCount = file.readUInt32LE(12);
  const stringTableOffset = Number(file.readBigUInt64LE(16));
  const dataSectionOffset = Number(file.readBigUInt64LE(24));
  const stringTable = file.subarray(stringTableOffset, dataSectionOffset);

  for (let i = 0; i < entryCount; i++) {
    const offset = SHARD_HEADER_SIZE + i * SHARD_INDEX_ENTRY_SIZE;
    const entry = parseIndexEntry(file.subarray(offset, offset + SHARD_INDEX_ENTRY_SIZE));
    entry.name = stringTable
      .subarray(entry.nameOffset, entry.nameOffset + entry.nameLen)
      .toString('utf-8');
    if (entry.name === targetName) {
      return entry;
    }
  }

  throw new Error(`Index entry not found: ${targetName}`);
}

describe('W-SHARD types', () => {
  it('should compute CRC32C (Castagnoli) correctly', () => {
    const data = Buffer.from('hello', 'utf-8');
    const crc = crc32C(data);
    // Expected CRC32C (Castagnoli) for "hello" — matches Go's crc32.Castagnoli
    expect(crc).toBe(0x9a71bb4c);
  });

  it('should compute simple hash', () => {
    const hash = simpleHash64('test');
    expect(hash).toBeTypeOf('bigint');
    expect(hash).toBeGreaterThan(0n);
  });

  it('should encode/decode float32', () => {
    const original = [1.5, 2.5, 3.5, -4.5];
    const encoded = encodeFloat32(original);
    expect(encoded.length).toBe(16); // 4 floats * 4 bytes
    const decoded = decodeFloat32(encoded);
    expect(decoded).toEqual(original);
  });

  it('should encode/decode int32', () => {
    const original = [1, -2, 3, -4, 1000000];
    const encoded = encodeInt32(original);
    expect(encoded.length).toBe(20); // 5 ints * 4 bytes
    const decoded = decodeInt32(encoded);
    expect(decoded).toEqual(original);
  });

  it('should encode/decode bool', () => {
    const original = [true, false, true, true, false];
    const encoded = encodeBool(original);
    expect(encoded.length).toBe(5);
    const decoded = decodeBool(encoded);
    expect(decoded).toEqual(original);
  });
});

describe('W-SHARD compression', () => {
  it('should compress with none (passthrough)', () => {
    const c = new Compressor('none', 'default');
    const data = Buffer.from('hello world', 'utf-8');
    const compressed = c.compress(data);
    expect(compressed).toEqual(data);
    c.close();
  });

  it('should compress/decompress with lz4', () => {
    const c = new Compressor('lz4', 'default');
    // Create compressible data
    const data = Buffer.alloc(1000);
    for (let i = 0; i < 1000; i++) {
      data[i] = i % 256;
    }
    const compressed = c.compress(data);
    expect(compressed.length).toBeLessThan(data.length);
    const decompressed = c.decompress(compressed, data.length);
    expect(decompressed).toEqual(data);
    c.close();
  });

  it('should handle incompressible data gracefully', () => {
    const c = new Compressor('lz4', 'default');
    // Random-ish data that's hard to compress
    const data = Buffer.alloc(100);
    for (let i = 0; i < 100; i++) {
      data[i] = (i * 17 + 23) % 256;
    }
    const compressed = c.compress(data);
    // Should either compress or return original
    expect(compressed.length).toBeLessThanOrEqual(data.length * 2);
    c.close();
  });
});

describe('W-SHARD roundtrip', () => {
  const tmpDir = os.tmpdir();
  const testFile = path.join(tmpDir, `wshard_test_${Date.now()}.wshard`);

  afterAll(() => {
    try {
      fs.unlinkSync(testFile);
    } catch {
      // Ignore
    }
  });

  it('should write and read basic W-SHARD file', async () => {
    const T = 100; // timesteps
    const D = 4; // action dimension

    // Create test data
    const ticks = Array.from({ length: T }, (_, i) => i);
    const actions: number[][] = Array.from({ length: T }, (_, t) =>
      Array.from({ length: D }, (_, d) => t * 0.1 + d * 0.01)
    );
    const signals: number[][] = Array.from({ length: T }, (_, t) =>
      Array.from({ length: 2 }, (_, d) => Math.sin(t * 0.1) + d)
    );
    const rewards = Array.from({ length: T }, (_, t) => t * 0.01);
    const done = Array.from({ length: T }, (_, t) => t === T - 1);

    // Write
    const writer = new WShardWriter(testFile);
    writer
      .setEpisode({
        episode_id: 'test_001',
        length_T: T,
        source: 'test',
        env_id: 'TestEnv-v0',
        timebase: { type: 'ticks', dt_ns: 33333333 },
        action_space: 'continuous',
      })
      .addChannel({
        id: 'state/pos',
        dtype: 'f32',
        shape: [2],
        unit: 'meters',
        description: 'Position',
      })
      .setTimeTicks(ticks)
      .setActionFloat32_2D('main', actions)
      .setSignalFloat32_2D('state/pos', signals)
      .setReward(rewards)
      .setDone(done);

    const bytesWritten = await writer.write();
    expect(bytesWritten).toBeGreaterThan(0);
    expect(fs.existsSync(testFile)).toBe(true);

    // Read
    const reader = new WShardReader(testFile);
    await reader.open();

    // Check metadata
    expect(reader.episodeMeta?.episode_id).toBe('test_001');
    expect(reader.episodeMeta?.length_T).toBe(T);
    expect(reader.lengthT).toBe(T);

    // Check channels
    expect(reader.channelIDs()).toContain('state/pos');
    const chDef = reader.getChannelDef('state/pos');
    expect(chDef?.dtype).toBe('f32');
    expect(chDef?.shape).toEqual([2]);

    // Check blocks
    expect(reader.blockNames()).toContain('time/ticks');
    expect(reader.blockNames()).toContain('action/main');
    expect(reader.blockNames()).toContain('signal/state/pos');
    expect(reader.blockNames()).toContain('reward');
    expect(reader.blockNames()).toContain('done');

    // Read time
    const readTicks = await reader.getTimeTicks();
    expect(readTicks).toEqual(ticks);

    // Read actions
    const readActions = await reader.getActionFloat32_2D('main', D);
    expect(readActions.length).toBe(T);
    expect(readActions[0].length).toBe(D);
    for (let t = 0; t < T; t++) {
      for (let d = 0; d < D; d++) {
        expect(readActions[t][d]).toBeCloseTo(actions[t][d], 5);
      }
    }

    // Read signals
    const readSignals = await reader.getSignalFloat32_2D('state/pos', 2);
    expect(readSignals.length).toBe(T);
    for (let t = 0; t < T; t++) {
      for (let d = 0; d < 2; d++) {
        expect(readSignals[t][d]).toBeCloseTo(signals[t][d], 5);
      }
    }

    // Read reward
    const readRewards = await reader.getReward();
    expect(readRewards.length).toBe(T);
    for (let t = 0; t < T; t++) {
      expect(readRewards[t]).toBeCloseTo(rewards[t], 5);
    }

    // Read done
    const readDone = await reader.getDone();
    expect(readDone).toEqual(done);

    await reader.close();
  });

  it('should write and read with compression', async () => {
    const compressedFile = path.join(tmpDir, `wshard_compressed_${Date.now()}.wshard`);
    const T = 200;

    try {
      // Create test data with repeated values so compression definitely kicks in.
      const ticks = Array.from({ length: T }, (_, i) => i);
      const actions = Array.from({ length: T }, () => [0, 0]);
      const signals = Array.from({ length: T }, () => [0]);
      const rewards = Array.from({ length: T }, () => 1.0);
      const done = Array.from({ length: T }, (_, t) => t === T - 1);

      // Write with LZ4 compression
      const writer = new WShardWriter(compressedFile, { compression: 'lz4' });
      writer
        .setEpisode({
          episode_id: 'compressed_001',
          length_T: T,
          timebase: { type: 'ticks' },
        })
        .addChannel({ id: 'obs', dtype: 'f32', shape: [1] })
        .setTimeTicks(ticks)
        .setActionFloat32_2D('main', actions)
        .setSignalFloat32_2D('obs', signals)
        .setReward(rewards)
        .setDone(done);

      await writer.write();

      const obsEntry = readIndexEntryByName(compressedFile, 'signal/obs');
      const obsBytes = encodeFloat32_2D(signals);
      expect(Number(obsEntry.diskSize)).toBeLessThan(Number(obsEntry.origSize));
      expect(obsEntry.checksum).toBe(crc32C(obsBytes));

      // Read back
      const reader = new WShardReader(compressedFile);
      await reader.open();

      expect(reader.episodeMeta?.episode_id).toBe('compressed_001');
      expect(reader.lengthT).toBe(T);

      const readSignals = await reader.getSignalFloat32_2D('obs', 1);
      expect(readSignals.length).toBe(T);
      for (let t = 0; t < T; t++) {
        expect(readSignals[t][0]).toBeCloseTo(signals[t][0], 5);
      }

      await reader.close();
    } finally {
      try {
        fs.unlinkSync(compressedFile);
      } catch {
        // Ignore
      }
    }
  });
});

describe('W-SHARD multi-modal + latent actions (Gap 2, 5)', () => {
  it('should write and read multi-modal signals', async () => {
    const testFile = path.join(os.tmpdir(), `wshard_mm_${Date.now()}.wshard`);
    const T = 20;

    try {
      const ticks = Array.from({ length: T }, (_, i) => i);
      const rgbData = Array.from({ length: T }, (_, t) => [t * 0.1, t * 0.2, t * 0.3]);
      const propData = Array.from({ length: T }, (_, t) => [t * 0.01, t * 0.02]);
      const actions = Array.from({ length: T }, (_, t) => [t * 0.1]);
      const done = Array.from({ length: T }, (_, t) => t === T - 1);

      const writer = new WShardWriter(testFile);
      writer
        .setEpisode({ episode_id: 'mm_001', length_T: T, timebase: { type: 'ticks' } })
        .addChannel({ id: 'obs/rgb', dtype: 'f32', shape: [3], modality: 'rgb' })
        .addChannel({ id: 'obs/proprioception', dtype: 'f32', shape: [2], modality: 'proprioception' })
        .setTimeTicks(ticks)
        .setMultiModalSignalFloat32_2D('obs', 'rgb', rgbData)
        .setMultiModalSignalFloat32_2D('obs', 'proprioception', propData)
        .setActionFloat32_2D('main', actions)
        .setReward(Array.from({ length: T }, () => 0))
        .setDone(done);

      await writer.write();

      const reader = new WShardReader(testFile);
      await reader.open();

      expect(reader.hasBlock('signal/obs/rgb')).toBe(true);
      expect(reader.hasBlock('signal/obs/proprioception')).toBe(true);

      const readRgb = await reader.getMultiModalSignalFloat32_2D('obs', 'rgb', 3);
      expect(readRgb.length).toBe(T);
      for (let t = 0; t < T; t++) {
        expect(readRgb[t][0]).toBeCloseTo(rgbData[t][0], 5);
        expect(readRgb[t][1]).toBeCloseTo(rgbData[t][1], 5);
      }

      await reader.close();
    } finally {
      try { fs.unlinkSync(testFile); } catch { /* ignore */ }
    }
  });

  it('should write and read latent actions', async () => {
    const testFile = path.join(os.tmpdir(), `wshard_latent_${Date.now()}.wshard`);
    const T = 10;
    const latentDim = 8;

    try {
      const ticks = Array.from({ length: T }, (_, i) => i);
      const signals = Array.from({ length: T }, (_, t) => [t * 0.1]);
      const actions = Array.from({ length: T }, (_, t) => [t * 0.1]);
      const done = Array.from({ length: T }, (_, t) => t === T - 1);

      const latentData = Array.from({ length: T }, (_, t) =>
        Array.from({ length: latentDim }, (_, d) => t * 0.01 + d * 0.1)
      );
      const codebookData = Array.from({ length: T }, (_, t) => t % 256);

      const writer = new WShardWriter(testFile);
      writer
        .setEpisode({ episode_id: 'latent_001', length_T: T, timebase: { type: 'ticks' } })
        .addChannel({ id: 'obs', dtype: 'f32', shape: [1] })
        .setTimeTicks(ticks)
        .setSignalFloat32_2D('obs', signals)
        .setActionFloat32_2D('main', actions)
        .setLatentActionFloat32_2D('genie3', latentData)
        .setLatentActionCodebookInt32('genie3', codebookData)
        .setReward(Array.from({ length: T }, () => 0))
        .setDone(done);

      await writer.write();

      const reader = new WShardReader(testFile);
      await reader.open();

      expect(reader.hasBlock('omen/latent_action/genie3')).toBe(true);
      expect(reader.hasBlock('omen/latent_action_codebook/genie3')).toBe(true);

      const readLatent = await reader.getLatentActionFloat32_2D('genie3', latentDim);
      expect(readLatent.length).toBe(T);
      expect(readLatent[0].length).toBe(latentDim);
      for (let d = 0; d < latentDim; d++) {
        expect(readLatent[0][d]).toBeCloseTo(latentData[0][d], 5);
      }

      const readCb = await reader.getLatentActionCodebookInt32('genie3');
      expect(readCb).toEqual(codebookData);

      await reader.close();
    } finally {
      try { fs.unlinkSync(testFile); } catch { /* ignore */ }
    }
  });
});

describe('W-SHARD chunked episode metadata (Gap 1)', () => {
  it('should preserve chunk fields through write/read', async () => {
    const testFile = path.join(os.tmpdir(), `wshard_chunk_${Date.now()}.wshard`);
    const T = 20;

    try {
      const ticks = Array.from({ length: T }, (_, i) => i);
      const signals = Array.from({ length: T }, (_, t) => [t * 0.1]);
      const actions = Array.from({ length: T }, (_, t) => [t * 0.1]);
      const done = Array.from({ length: T }, (_, t) => t === T - 1);

      const writer = new WShardWriter(testFile);
      writer
        .setEpisode({
          episode_id: 'chunked_001',
          length_T: T,
          timebase: { type: 'ticks' },
          chunk_index: 2,
          total_chunks: 10,
          timestep_range: [40, 59],
        })
        .addChannel({ id: 'obs', dtype: 'f32', shape: [1] })
        .setTimeTicks(ticks)
        .setSignalFloat32_2D('obs', signals)
        .setActionFloat32_2D('main', actions)
        .setReward(Array.from({ length: T }, () => 0))
        .setDone(done);

      await writer.write();

      const reader = new WShardReader(testFile);
      await reader.open();

      expect(reader.episodeMeta?.chunk_index).toBe(2);
      expect(reader.episodeMeta?.total_chunks).toBe(10);
      expect(reader.episodeMeta?.timestep_range).toEqual([40, 59]);

      await reader.close();
    } finally {
      try { fs.unlinkSync(testFile); } catch { /* ignore */ }
    }
  });
});

describe('W-SHARD stream writer (Gap 4)', () => {
  it('should stream-write and read with standard reader', async () => {
    const { WShardStreamWriter } = await import('../src/stream-writer.js');
    const testFile = path.join(os.tmpdir(), `wshard_stream_${Date.now()}.wshard`);
    const T = 50;

    try {
      const channelDefs = [{ name: 'obs', dtype: 'f32' as const, shape: [2] }];
      const writer = new WShardStreamWriter(testFile, 'stream_001', channelDefs);
      writer.beginEpisode('TestEnv');

      for (let t = 0; t < T; t++) {
        const obs = encodeFloat32([t * 0.1, t * 0.2]);
        const act = encodeFloat32([0.0, 0.0]);
        writer.writeTimestep(t, { obs }, { obs: act }, t * 0.01, t === T - 1);
      }

      const totalBytes = writer.endEpisode();
      expect(totalBytes).toBeGreaterThan(0);
      expect(writer.isFinalized).toBe(true);

      // Read back with standard reader
      const reader = new WShardReader(testFile);
      await reader.open();

      expect(reader.episodeMeta?.episode_id).toBe('stream_001');
      expect(reader.episodeMeta?.length_T).toBe(T);
      expect(reader.hasBlock('signal/obs')).toBe(true);
      expect(reader.hasBlock('reward')).toBe(true);

      const readSignals = await reader.getSignalFloat32_2D('obs', 2);
      expect(readSignals.length).toBe(T);
      expect(readSignals[0][0]).toBeCloseTo(0.0, 5);
      expect(readSignals[1][0]).toBeCloseTo(0.1, 5);

      await reader.close();
    } finally {
      try { fs.unlinkSync(testFile); } catch { /* ignore */ }
    }
  });
});

describe('W-SHARD residual encoding', () => {
  it('should compute and pack sign2nddiff residuals', async () => {
    const testFile = path.join(os.tmpdir(), `wshard_residual_${Date.now()}.wshard`);
    const T = 50;

    try {
      // Create signal with curvature changes
      const signal = Array.from({ length: T }, (_, t) => Math.sin(t * 0.2));
      const actions = Array.from({ length: T }, (_, t) => [t * 0.1]);
      const ticks = Array.from({ length: T }, (_, i) => i);
      const done = Array.from({ length: T }, (_, t) => t === T - 1);

      const writer = new WShardWriter(testFile);
      writer
        .setEpisode({ episode_id: 'residual_001', length_T: T, timebase: { type: 'ticks' } })
        .addChannel({ id: 'x', dtype: 'f32', shape: [] })
        .setTimeTicks(ticks)
        .setActionFloat32_2D('main', actions)
        .setSignalFloat32('x', signal)
        .computeAndSetResidual('x', signal)
        .setReward(Array.from({ length: T }, () => 0))
        .setDone(done);

      await writer.write();

      // Read and verify
      const reader = new WShardReader(testFile);
      await reader.open();

      const packed = await reader.getResidualSign2ndDiff('x');
      expect(packed.length).toBeGreaterThan(0);

      const unpacked = await reader.getResidualSign2ndDiffUnpacked('x');
      expect(unpacked.length).toBe(T);
      // Edge samples should be 0
      expect(unpacked[0]).toBe(0);
      expect(unpacked[T - 1]).toBe(0);

      await reader.close();
    } finally {
      try {
        fs.unlinkSync(testFile);
      } catch {
        // Ignore
      }
    }
  });
});
