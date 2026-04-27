/**
 * Streaming append-only episode writer for W-SHARD (Gap 4).
 *
 * Enables incremental episode building for online learning.
 * Uses a reserve-write-finalize pattern.
 */

import * as fs from 'fs';
import {
  SHARD_MAGIC,
  SHARD_VERSION_2,
  WSHARD_ROLE,
  SHARD_HEADER_SIZE,
  SHARD_INDEX_ENTRY_SIZE,
  DEFAULT_ALIGNMENT,
  BLOCK_FLAG_COMPRESSED,
  compressionByte,
  simpleHash64,
  crc32C,
  encodeFloat32,
  encodeInt32,
  type CompressionType,
  type CompressionLevel,
  type DType,
  type Modality,
} from './types.js';
import { Compressor, shouldCompress } from './compress.js';

// Header flag indicating streaming file
const FLAG_STREAMING = 0x0040;

// Default buffer flush interval (timesteps)
const DEFAULT_FLUSH_INTERVAL = 64;

/**
 * Channel definition for shape validation during streaming.
 */
export interface StreamChannelDef {
  name: string;
  dtype: DType;
  shape: number[];
  modality?: Modality;
}

interface BlockPosition {
  startOffset: number;
  totalWritten: number;
  origSize?: number;
  flags?: number;
  checksum?: number;
}

/**
 * Append-only streaming episode writer.
 */
export class WShardStreamWriter {
  private path: string;
  private episodeId: string;
  private channelDefs: Map<string, StreamChannelDef>;
  private maxTimesteps: number;
  private compression: CompressionType;
  private compressionLevel: CompressionLevel;
  private flushInterval: number;

  // State
  private fd: number | null = null;
  private started = false;
  private finalized = false;
  private timestepCount = 0;
  private envId = '';
  private reservedSize = 0;
  private _writeOffset = 0;
  private partialPath: string;

  // Buffered data per block
  private buffers: Map<string, Buffer[]> = new Map();
  private bufferedCount = 0;

  // Track written block positions
  private blockPositions: Map<string, BlockPosition> = new Map();

  constructor(
    path: string,
    episodeId: string,
    channelDefs: StreamChannelDef[],
    options: {
      maxTimesteps?: number;
      compression?: CompressionType;
      compressionLevel?: CompressionLevel;
      flushInterval?: number;
    } = {},
  ) {
    this.path = path;
    this.partialPath = path + '.partial';
    this.episodeId = episodeId;
    this.channelDefs = new Map(channelDefs.map(cd => [cd.name, cd]));
    this.maxTimesteps = options.maxTimesteps ?? 100000;
    this.compression = options.compression ?? 'none';
    this.compressionLevel = options.compressionLevel ?? 'default';
    this.flushInterval = options.flushInterval ?? DEFAULT_FLUSH_INTERVAL;
  }

  /**
   * Begin a new streaming episode.
   */
  beginEpisode(envId = ''): void {
    if (this.started) throw new Error('Episode already started');
    this.envId = envId;
    this.started = true;

    // Estimate max blocks
    const maxBlocks = 4 + 1 + 1 + 1 + this.channelDefs.size * 2;
    const reservedIndexSize = maxBlocks * SHARD_INDEX_ENTRY_SIZE;
    let stringEstimate = 200;
    for (const name of this.channelDefs.keys()) {
      stringEstimate += `signal/${name}`.length + `action/${name}`.length;
    }

    this.reservedSize = align(
      SHARD_HEADER_SIZE + reservedIndexSize + stringEstimate,
      DEFAULT_ALIGNMENT,
    );

    // Validate reserved space can fit expected index
    const minBlocks = 4 + 1 + 1 + 1 + this.channelDefs.size * 2;
    const minHeaderSpace = SHARD_HEADER_SIZE + minBlocks * SHARD_INDEX_ENTRY_SIZE;
    if (this.reservedSize < minHeaderSpace) {
      throw new Error(
        `Reserved space ${this.reservedSize} too small for ${minBlocks} blocks (need ${minHeaderSpace})`
      );
    }

    // Write to .partial file, atomic rename on finalize
    this.fd = fs.openSync(this.partialPath, 'w+');
    const reserved = Buffer.alloc(this.reservedSize);
    fs.writeSync(this.fd, reserved);
    this._writeOffset = this.reservedSize;

    // Initialize buffers
    for (const name of this.channelDefs.keys()) {
      this.buffers.set(`signal/${name}`, []);
      this.buffers.set(`action/${name}`, []);
    }
    this.buffers.set('reward', []);
    this.buffers.set('done', []);
    this.buffers.set('time/ticks', []);
  }

  /**
   * Write a single timestep.
   */
  writeTimestep(
    t: number,
    observations: Record<string, Buffer>,
    actions: Record<string, Buffer>,
    reward: number,
    done: boolean,
  ): void {
    if (!this.started) throw new Error('Call beginEpisode() first');
    if (this.finalized) throw new Error('Episode already finalized');
    if (this.timestepCount >= this.maxTimesteps) {
      throw new Error(`Max timesteps (${this.maxTimesteps}) exceeded`);
    }

    // Buffer observations
    for (const [name, data] of Object.entries(observations)) {
      const buf = this.buffers.get(`signal/${name}`);
      if (!buf) throw new Error(`Unknown channel: ${name}`);
      buf.push(data);
    }

    // Buffer actions
    for (const [name, data] of Object.entries(actions)) {
      const buf = this.buffers.get(`action/${name}`);
      if (!buf) throw new Error(`Unknown action channel: ${name}`);
      buf.push(data);
    }

    // Buffer reward
    const rewardBuf = Buffer.alloc(4);
    rewardBuf.writeFloatLE(reward, 0);
    this.buffers.get('reward')!.push(rewardBuf);

    // Buffer done
    const doneBuf = Buffer.alloc(1);
    doneBuf[0] = done ? 1 : 0;
    this.buffers.get('done')!.push(doneBuf);

    // Buffer time tick
    const tickBuf = Buffer.alloc(4);
    tickBuf.writeInt32LE(t, 0);
    this.buffers.get('time/ticks')!.push(tickBuf);

    this.timestepCount++;
    this.bufferedCount++;

    if (this.bufferedCount >= this.flushInterval) {
      this.flushBuffers();
    }
  }

  /**
   * Write a timestep with typed arrays (convenience).
   */
  writeTimestepFloat32(
    t: number,
    observations: Record<string, number[]>,
    actions: Record<string, number[]>,
    reward: number,
    done: boolean,
  ): void {
    const obsBufs: Record<string, Buffer> = {};
    for (const [name, data] of Object.entries(observations)) {
      obsBufs[name] = encodeFloat32(data);
    }
    const actBufs: Record<string, Buffer> = {};
    for (const [name, data] of Object.entries(actions)) {
      actBufs[name] = encodeFloat32(data);
    }
    this.writeTimestep(t, obsBufs, actBufs, reward, done);
  }

  private flushBuffers(): void {
    if (!this.fd || this.bufferedCount === 0) return;

    for (const [blockName, bufs] of this.buffers.entries()) {
      if (bufs.length === 0) continue;

      const data = Buffer.concat(bufs);
      if (data.length === 0) continue;

      if (!this.blockPositions.has(blockName)) {
        this.blockPositions.set(blockName, {
          startOffset: this._writeOffset,
          totalWritten: 0,
        });
      }

      // Append to file
      fs.writeSync(this.fd, data);
      this._writeOffset += data.length;
      const bp = this.blockPositions.get(blockName)!;
      bp.totalWritten += data.length;

      bufs.length = 0; // Clear buffer
    }

    fs.fsyncSync(this.fd);
    this.bufferedCount = 0;
  }

  /**
   * Finalize the episode: flush, write metadata, seek-back to write header.
   */
  endEpisode(): number {
    if (!this.started) throw new Error('Call beginEpisode() first');
    if (this.finalized) throw new Error('Already finalized');

    this.flushBuffers();

    // Build and write metadata blocks
    const metaBlocks = this.buildMetadata();
    let compressor: Compressor | null = null;
    if (this.compression !== 'none') {
      compressor = new Compressor(this.compression, this.compressionLevel);
    }

    for (const [blockName, blockData] of Object.entries(metaBlocks)) {
      let pos = this._writeOffset;

      let diskData = blockData;
      let flags = 0;

      if (compressor && shouldCompress(blockName, blockData)) {
        const compressed = compressor.compress(blockData);
        if (compressed.length < blockData.length) {
          diskData = compressed;
          flags = BLOCK_FLAG_COMPRESSED;
        }
      }

      // Align
      const padding = (DEFAULT_ALIGNMENT - (pos % DEFAULT_ALIGNMENT)) % DEFAULT_ALIGNMENT;
      if (padding > 0) {
        fs.writeSync(this.fd!, Buffer.alloc(padding));
        pos += padding;
        this._writeOffset += padding;
      }

      fs.writeSync(this.fd!, diskData);
      this._writeOffset += diskData.length;
      this.blockPositions.set(blockName, {
        startOffset: pos,
        totalWritten: diskData.length,
        origSize: blockData.length,
        flags,
        checksum: crc32C(blockData),
      });
    }

    compressor?.close();

    const totalSize = this._writeOffset;

    // Build header + index + string table
    const allBlocks = [...this.blockPositions.keys()].sort();

    // String table
    const stringParts: Buffer[] = [];
    const stringOffsets: Map<string, number> = new Map();
    let strOff = 0;
    for (const name of allBlocks) {
      stringOffsets.set(name, strOff);
      const nameBytes = Buffer.from(name, 'utf-8');
      stringParts.push(nameBytes);
      strOff += nameBytes.length;
    }
    const stringTable = Buffer.concat(stringParts);

    const entryCount = allBlocks.length;
    const indexSize = entryCount * SHARD_INDEX_ENTRY_SIZE;
    const stringTableOffset = SHARD_HEADER_SIZE + indexSize;
    const dataSectionOffset = align(stringTableOffset + stringTable.length, DEFAULT_ALIGNMENT);

    if (dataSectionOffset > this.reservedSize) {
      throw new Error(`Reserved space too small: need ${dataSectionOffset}, have ${this.reservedSize}`);
    }

    // Build header
    const header = Buffer.alloc(SHARD_HEADER_SIZE);
    SHARD_MAGIC.copy(header, 0);
    header.writeUInt8(SHARD_VERSION_2, 4);
    header.writeUInt8(WSHARD_ROLE, 5);
    header.writeUInt16LE(0, 6); // clear streaming flag
    header.writeUInt8(DEFAULT_ALIGNMENT, 8);
    header.writeUInt8(compressionByte(this.compression), 9);
    header.writeUInt16LE(SHARD_INDEX_ENTRY_SIZE, 10);
    header.writeUInt32LE(entryCount, 12);
    header.writeBigUInt64LE(BigInt(stringTableOffset), 16);
    header.writeBigUInt64LE(BigInt(dataSectionOffset), 24);
    header.writeBigUInt64LE(0n, 32);
    header.writeBigUInt64LE(BigInt(totalSize), 40);

    // Build index entries
    const indexBuf = Buffer.alloc(indexSize);
    for (let i = 0; i < allBlocks.length; i++) {
      const name = allBlocks[i];
      const bp = this.blockPositions.get(name)!;
      const offset = i * SHARD_INDEX_ENTRY_SIZE;

      let checksum = bp.checksum;
      if (checksum === undefined) {
        // Streaming data blocks are written uncompressed, so disk bytes equal logical bytes.
        const diskData = Buffer.alloc(bp.totalWritten);
        fs.readSync(this.fd!, diskData, 0, bp.totalWritten, bp.startOffset);
        checksum = crc32C(diskData);
      }

      const nameHash = simpleHash64(name);
      const nameBytes = Buffer.from(name, 'utf-8');

      indexBuf.writeBigUInt64LE(nameHash, offset);
      indexBuf.writeUInt32LE(stringOffsets.get(name)!, offset + 8);
      indexBuf.writeUInt16LE(nameBytes.length, offset + 12);
      indexBuf.writeUInt16LE(bp.flags ?? 0, offset + 14);
      indexBuf.writeBigUInt64LE(BigInt(bp.startOffset), offset + 16);
      indexBuf.writeBigUInt64LE(BigInt(bp.totalWritten), offset + 24);
      indexBuf.writeBigUInt64LE(BigInt(bp.origSize ?? bp.totalWritten), offset + 32);
      indexBuf.writeUInt32LE(checksum, offset + 40);
    }

    // Seek to start and write
    fs.writeSync(this.fd!, header, 0, header.length, 0);
    fs.writeSync(this.fd!, indexBuf, 0, indexBuf.length, SHARD_HEADER_SIZE);
    fs.writeSync(this.fd!, stringTable, 0, stringTable.length, stringTableOffset);

    // Pad between string table and data section
    const remainingPad = dataSectionOffset - (stringTableOffset + stringTable.length);
    if (remainingPad > 0) {
      fs.writeSync(this.fd!, Buffer.alloc(remainingPad), 0, remainingPad, stringTableOffset + stringTable.length);
    }

    fs.ftruncateSync(this.fd!, totalSize);
    fs.closeSync(this.fd!);
    this.fd = null;

    // Atomic rename from .partial to final path
    fs.renameSync(this.partialPath, this.path);
    this.finalized = true;

    return totalSize;
  }

  private buildMetadata(): Record<string, Buffer> {
    const blocks: Record<string, Buffer> = {};

    // meta/wshard
    blocks['meta/wshard'] = Buffer.from(JSON.stringify({
      format: 'W-SHARD',
      version: '0.1',
      residual_edges: 'pad',
      streaming: true,
    }), 'utf-8');

    // meta/episode
    blocks['meta/episode'] = Buffer.from(JSON.stringify({
      episode_id: this.episodeId,
      env_id: this.envId,
      length_T: this.timestepCount,
      timebase: { type: 'ticks' },
    }), 'utf-8');

    // meta/channels
    const channels: any[] = [];
    for (const [name, cd] of this.channelDefs) {
      const ch: any = {
        id: name,
        dtype: cd.dtype,
        shape: cd.shape,
        signal_block: `signal/${name}`,
      };
      if (cd.modality) ch.modality = cd.modality;
      channels.push(ch);
    }
    blocks['meta/channels'] = Buffer.from(JSON.stringify({ channels }), 'utf-8');

    return blocks;
  }

  /** Number of timesteps written so far. */
  get count(): number {
    return this.timestepCount;
  }

  /** Whether the episode has been finalized. */
  get isFinalized(): boolean {
    return this.finalized;
  }
}

function align(n: number, alignment: number): number {
  if (alignment === 0) return n;
  return Math.ceil(n / alignment) * alignment;
}
