import { describe, it, expect } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { WShardStreamWriter } from '../src/stream-writer.js';
import { WShardReader } from '../src/reader.js';
import { encodeFloat32 } from '../src/types.js';

/**
 * A block's index entry is a single (offset, size) extent, so a block must be
 * written exactly once. When the writer flushed periodically, every flush
 * appended every non-empty block, so a block's bytes stopped being contiguous
 * while its extent grew over its neighbours'. The byte count still came out
 * right -- T inferred correctly, the reshape succeeded, CRC passed -- and the
 * values were a neighbouring channel's. Shape and length assertions cannot
 * catch that; only comparing values can. The old T=50 streaming test sat just
 * under the 64-timestep default interval and never flushed mid-episode.
 */
describe('W-SHARD stream writer: single-extent invariant', () => {
  it('keeps every channel\'s values across an episode longer than the old flush interval', async () => {
    const testFile = path.join(os.tmpdir(), `ws_flush_${Date.now()}.wshard`);
    const T = 135; // > 2x the 64-timestep interval that used to trigger a flush
    const defs = [
      { name: 'a', dtype: 'f32' as const, shape: [1] },
      { name: 'b', dtype: 'f32' as const, shape: [1] },
    ];
    try {
      const w = new WShardStreamWriter(testFile, 'multiflush', defs);
      w.beginEpisode('Test');
      for (let t = 0; t < T; t++) {
        w.writeTimestep(t,
          { a: encodeFloat32([t]), b: encodeFloat32([1000 + t]) },
          { a: encodeFloat32([-t]), b: encodeFloat32([0]) },
          t * 0.5, t === T - 1);
      }
      w.endEpisode();

      const r = new WShardReader(testFile);
      await r.open();
      expect((await r.getSignalFloat32_2D('a', 1)).map(v => v[0]))
        .toEqual(Array.from({ length: T }, (_, t) => t));
      expect((await r.getSignalFloat32_2D('b', 1)).map(v => v[0]))
        .toEqual(Array.from({ length: T }, (_, t) => 1000 + t));
      await r.close();
    } finally {
      try { fs.unlinkSync(testFile); } catch { /* ignore */ }
    }
  });

  it('rejects flushInterval rather than ignoring it', () => {
    const testFile = path.join(os.tmpdir(), `ws_reject_${Date.now()}.wshard`);
    expect(() => new WShardStreamWriter(
      testFile, 'ep', [{ name: 'a', dtype: 'f32' as const, shape: [1] }],
      { flushInterval: 4 },
    )).toThrow(/flushInterval is not supported/);
  });
});
