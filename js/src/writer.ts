/**
 * W-SHARD writer implementation.
 *
 * Writes W-SHARD files with the Shard v2 format (role=0x05).
 *
 * Shard v2 layout:
 *     Header (64 bytes)
 *     Index entries (entry_count * 48 bytes)
 *     String table (variable)
 *     [Padding to alignment]
 *     Data section (aligned entries)
 */

import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import {
  ShardHeader,
  IndexEntry,
  WShardMeta,
  EpisodeMeta,
  ChannelsMeta,
  ChannelDef,
  ModelsMeta,
  ModelDef,
  CompressionType,
  CompressionLevel,
  SHARD_MAGIC,
  SHARD_VERSION_2,
  WSHARD_ROLE,
  SHARD_HEADER_SIZE,
  SHARD_INDEX_ENTRY_SIZE,
  DEFAULT_ALIGNMENT,
  BLOCK_FLAG_COMPRESSED,
  BLOCK_META_WSHARD,
  BLOCK_META_EPISODE,
  BLOCK_META_CHANNELS,
  BLOCK_META_MODELS,
  BLOCK_TIME_TICKS,
  BLOCK_TIME_TIMESTAMPS_NS,
  BLOCK_REWARD,
  BLOCK_DONE,
  compressionByte,
  simpleHash64,
  signalBlock,
  actionBlock,
  omenBlock,
  uncertBlock,
  residualSign2ndDiffBlock,
  latentActionBlock,
  latentCodebookBlock,
  multiModalSignalBlock,
  crc32C,
  encodeFloat32,
  encodeFloat32_2D,
  encodeInt32,
  encodeInt64,
  encodeBool,
} from './types.js';
import type { Modality } from './types.js';
import { Compressor, shouldCompress } from './compress.js';

interface PendingBlock {
  name: string;
  data: Buffer;
  compressed: Buffer;
  flags: number;
  checksum: number;
}

/**
 * Writer for W-SHARD files.
 */
export class WShardWriter {
  private filePath: string;
  private alignment: number;
  private compression: CompressionType;
  private compressionLevel: CompressionLevel;

  // Metadata
  private _wshardMeta: WShardMeta;
  private _episodeMeta: EpisodeMeta | null = null;
  private _channelsMeta: ChannelsMeta;
  private _modelsMeta: ModelsMeta | null = null;

  // Block data
  private blocks: Map<string, Buffer> = new Map();

  constructor(
    filePath: string,
    options: {
      alignment?: number;
      compression?: CompressionType;
      compressionLevel?: CompressionLevel;
    } = {}
  ) {
    this.filePath = filePath;
    this.alignment = options.alignment ?? DEFAULT_ALIGNMENT;
    this.compression = options.compression ?? 'none';
    this.compressionLevel = options.compressionLevel ?? 'default';

    // Initialize default metadata
    this._wshardMeta = {
      format: 'W-SHARD',
      version: '0.1',
      residual_edges: 'pad',
    };
    this._channelsMeta = { channels: [] };
  }

  // ============================================================
  // Configuration
  // ============================================================

  /**
   * Set compression type.
   */
  withCompression(ct: CompressionType): this {
    this.compression = ct;
    return this;
  }

  /**
   * Set compression level.
   */
  withCompressionLevel(level: CompressionLevel): this {
    this.compressionLevel = level;
    return this;
  }

  /**
   * Set residual edge rule.
   */
  setResidualEdgeRule(rule: 'pad' | 'overlap'): this {
    this._wshardMeta.residual_edges = rule;
    return this;
  }

  // ============================================================
  // Metadata setters
  // ============================================================

  /**
   * Set episode metadata.
   */
  setEpisode(meta: EpisodeMeta): this {
    this._episodeMeta = meta;
    return this;
  }

  /**
   * Set models metadata.
   */
  setModels(meta: ModelsMeta): this {
    this._modelsMeta = meta;
    return this;
  }

  /**
   * Add a channel definition.
   */
  addChannel(ch: ChannelDef): this {
    this._channelsMeta.channels.push(ch);
    return this;
  }

  // ============================================================
  // Time blocks
  // ============================================================

  /**
   * Set time/ticks block.
   */
  setTimeTicks(ticks: number[]): this {
    this.blocks.set(BLOCK_TIME_TICKS, encodeInt32(ticks));
    return this;
  }

  /**
   * Set time/timestamps_ns block.
   */
  setTimeTimestampsNs(timestamps: bigint[]): this {
    this.blocks.set(BLOCK_TIME_TIMESTAMPS_NS, encodeInt64(timestamps));
    return this;
  }

  // ============================================================
  // Action blocks
  // ============================================================

  /**
   * Set action block with raw bytes.
   */
  setAction(name: string, data: Buffer): this {
    this.blocks.set(actionBlock(name), data);
    return this;
  }

  /**
   * Set action block with float32 array.
   */
  setActionFloat32(name: string, data: number[]): this {
    this.blocks.set(actionBlock(name), encodeFloat32(data));
    return this;
  }

  /**
   * Set action block with 2D float32 array [T][D].
   */
  setActionFloat32_2D(name: string, data: number[][]): this {
    this.blocks.set(actionBlock(name), encodeFloat32_2D(data));
    return this;
  }

  // ============================================================
  // Control blocks
  // ============================================================

  /**
   * Set reward block.
   */
  setReward(data: number[]): this {
    this.blocks.set(BLOCK_REWARD, encodeFloat32(data));
    return this;
  }

  /**
   * Set done block.
   */
  setDone(data: boolean[]): this {
    this.blocks.set(BLOCK_DONE, encodeBool(data));
    return this;
  }

  /**
   * Set done block from uint8 array.
   */
  setDoneUint8(data: Uint8Array): this {
    this.blocks.set(BLOCK_DONE, Buffer.from(data));
    return this;
  }

  // ============================================================
  // Signal blocks
  // ============================================================

  /**
   * Set signal block with raw bytes.
   */
  setSignal(channelID: string, data: Buffer): this {
    this.blocks.set(signalBlock(channelID), data);
    return this;
  }

  /**
   * Set signal block with float32 array.
   */
  setSignalFloat32(channelID: string, data: number[]): this {
    this.blocks.set(signalBlock(channelID), encodeFloat32(data));
    return this;
  }

  /**
   * Set signal block with 2D float32 array [T][D].
   */
  setSignalFloat32_2D(channelID: string, data: number[][]): this {
    this.blocks.set(signalBlock(channelID), encodeFloat32_2D(data));
    return this;
  }

  // ============================================================
  // Omen blocks
  // ============================================================

  /**
   * Set omen block with raw bytes.
   */
  setOmen(channelID: string, modelID: string, data: Buffer): this {
    this.blocks.set(omenBlock(channelID, modelID), data);
    return this;
  }

  /**
   * Set omen block with float32 array.
   */
  setOmenFloat32(channelID: string, modelID: string, data: number[]): this {
    this.blocks.set(omenBlock(channelID, modelID), encodeFloat32(data));
    return this;
  }

  /**
   * Set omen block with 2D float32 array [T][D].
   */
  setOmenFloat32_2D(channelID: string, modelID: string, data: number[][]): this {
    this.blocks.set(omenBlock(channelID, modelID), encodeFloat32_2D(data));
    return this;
  }

  // ============================================================
  // Uncertainty blocks
  // ============================================================

  /**
   * Set uncertainty block with raw bytes.
   */
  setUncert(channelID: string, modelID: string, uncertType: string, data: Buffer): this {
    this.blocks.set(uncertBlock(channelID, modelID, uncertType), data);
    return this;
  }

  /**
   * Set uncertainty block with float32 array.
   */
  setUncertFloat32(
    channelID: string,
    modelID: string,
    uncertType: string,
    data: number[]
  ): this {
    this.blocks.set(uncertBlock(channelID, modelID, uncertType), encodeFloat32(data));
    return this;
  }

  // ============================================================
  // Gap 5: Multi-modal signal blocks
  // ============================================================

  /**
   * Set a multi-modal signal block using hierarchical naming.
   * Convention: signal/{group}/{modality}
   */
  setMultiModalSignal(group: string, modality: Modality, data: Buffer): this {
    this.blocks.set(multiModalSignalBlock(group, modality), data);
    return this;
  }

  /**
   * Set a multi-modal signal block with float32 data.
   */
  setMultiModalSignalFloat32(group: string, modality: Modality, data: number[]): this {
    this.blocks.set(multiModalSignalBlock(group, modality), encodeFloat32(data));
    return this;
  }

  /**
   * Set a multi-modal signal block with 2D float32 data.
   */
  setMultiModalSignalFloat32_2D(group: string, modality: Modality, data: number[][]): this {
    this.blocks.set(multiModalSignalBlock(group, modality), encodeFloat32_2D(data));
    return this;
  }

  // ============================================================
  // Gap 2: Latent action blocks
  // ============================================================

  /**
   * Set latent action embeddings for a model.
   */
  setLatentAction(modelID: string, data: Buffer): this {
    this.blocks.set(latentActionBlock(modelID), data);
    return this;
  }

  /**
   * Set latent action embeddings with float32 data.
   */
  setLatentActionFloat32(modelID: string, data: number[]): this {
    this.blocks.set(latentActionBlock(modelID), encodeFloat32(data));
    return this;
  }

  /**
   * Set latent action embeddings with 2D float32 data [T, latent_dim].
   */
  setLatentActionFloat32_2D(modelID: string, data: number[][]): this {
    this.blocks.set(latentActionBlock(modelID), encodeFloat32_2D(data));
    return this;
  }

  /**
   * Set latent action codebook indices for a model.
   */
  setLatentActionCodebook(modelID: string, data: Buffer): this {
    this.blocks.set(latentCodebookBlock(modelID), data);
    return this;
  }

  /**
   * Set latent action codebook indices with int32 data.
   */
  setLatentActionCodebookInt32(modelID: string, data: number[]): this {
    this.blocks.set(latentCodebookBlock(modelID), encodeInt32(data));
    return this;
  }

  // ============================================================
  // Residual blocks
  // ============================================================

  /**
   * Set sign2nddiff residual block (already packed).
   */
  setResidualSign2ndDiff(channelID: string, packed: Buffer): this {
    this.blocks.set(residualSign2ndDiffBlock(channelID), packed);
    return this;
  }

  /**
   * Compute and set sign2nddiff residual from signal.
   */
  computeAndSetResidual(channelID: string, signal: number[]): this {
    const residuals = computeSign2ndDiff(signal, this._wshardMeta.residual_edges);
    const packed = packResidualBits(residuals);
    this.blocks.set(residualSign2ndDiffBlock(channelID), packed);
    return this;
  }

  // ============================================================
  // Write to file
  // ============================================================

  /**
   * Write the W-SHARD file.
   */
  async write(): Promise<number> {
    // Validate
    this.validate();

    // Serialize metadata
    this.serializeMetadata();

    // Create compressor if needed
    let compressor: Compressor | null = null;
    if (this.compression !== 'none') {
      compressor = new Compressor(this.compression, this.compressionLevel);
    }

    try {
      // Compress blocks and build index
      const pendingBlocks = this.buildPendingBlocks(compressor);

      // Sort blocks by name for deterministic output
      pendingBlocks.sort((a, b) => a.name.localeCompare(b.name));

      // Calculate offsets
      const headerSize = SHARD_HEADER_SIZE;
      const indexSize = pendingBlocks.length * SHARD_INDEX_ENTRY_SIZE;
      const stringTableOffset = headerSize + indexSize;
      const stringTable = this.buildStringTable(pendingBlocks);
      const dataSectionOffset = align(stringTableOffset + stringTable.length, this.alignment);

      // Assign data offsets
      let currentOffset = dataSectionOffset;
      const dataOffsets: number[] = [];
      for (const block of pendingBlocks) {
        currentOffset = align(currentOffset, this.alignment);
        dataOffsets.push(currentOffset);
        currentOffset += block.compressed.length;
      }
      const totalSize = currentOffset;

      // Open file
      const fd = fs.openSync(this.filePath, 'w');
      let written = 0;

      try {
        // Write header
        const header = this.buildHeader(
          pendingBlocks.length,
          stringTableOffset,
          dataSectionOffset,
          totalSize
        );
        fs.writeSync(fd, header);
        written += header.length;

        // Write index entries
        let nameOffset = 0;
        for (let i = 0; i < pendingBlocks.length; i++) {
          const block = pendingBlocks[i];
          const entry = this.buildIndexEntry(
            block,
            nameOffset,
            dataOffsets[i]
          );
          fs.writeSync(fd, entry);
          written += entry.length;
          nameOffset += Buffer.from(block.name, 'utf-8').length;
        }

        // Write string table
        fs.writeSync(fd, stringTable);
        written += stringTable.length;

        // Pad to data section
        const padding = dataSectionOffset - (headerSize + indexSize + stringTable.length);
        if (padding > 0) {
          fs.writeSync(fd, Buffer.alloc(padding));
          written += padding;
        }

        // Write data blocks
        let writePos = dataSectionOffset;
        for (let i = 0; i < pendingBlocks.length; i++) {
          const block = pendingBlocks[i];
          const expectedOffset = dataOffsets[i];

          // Alignment padding
          const alignPad = expectedOffset - writePos;
          if (alignPad > 0) {
            fs.writeSync(fd, Buffer.alloc(alignPad));
            written += alignPad;
            writePos += alignPad;
          }

          fs.writeSync(fd, block.compressed);
          written += block.compressed.length;
          writePos += block.compressed.length;
        }

        return written;
      } finally {
        fs.closeSync(fd);
      }
    } finally {
      compressor?.close();
    }
  }

  // ============================================================
  // Internal helpers
  // ============================================================

  /**
   * Get the number of timesteps from episode metadata.
   */
  get lengthT(): number {
    return this._episodeMeta?.length_T ?? 0;
  }

  private validate(): void {
    if (!this._episodeMeta) {
      throw new Error('Episode metadata required');
    }
    if (this._channelsMeta.channels.length === 0) {
      throw new Error('At least one channel required');
    }

    // Check time block
    const hasTicks = this.blocks.has(BLOCK_TIME_TICKS);
    const hasTimestamps = this.blocks.has(BLOCK_TIME_TIMESTAMPS_NS);
    if (!hasTicks && !hasTimestamps) {
      throw new Error('time/ticks or time/timestamps_ns required');
    }

    // Check done block
    if (!this.blocks.has(BLOCK_DONE)) {
      throw new Error('done block required');
    }

    // Check at least one action
    let hasAction = false;
    for (const name of this.blocks.keys()) {
      if (name.startsWith('action/')) {
        hasAction = true;
        break;
      }
    }
    if (!hasAction) {
      throw new Error('At least one action block required');
    }

    // Check at least one signal
    let hasSignal = false;
    for (const name of this.blocks.keys()) {
      if (name.startsWith('signal/')) {
        hasSignal = true;
        break;
      }
    }
    if (!hasSignal) {
      throw new Error('At least one signal block required');
    }
  }

  private serializeMetadata(): void {
    // meta/wshard
    this.blocks.set(BLOCK_META_WSHARD, Buffer.from(JSON.stringify(this._wshardMeta), 'utf-8'));

    // meta/episode
    this.blocks.set(BLOCK_META_EPISODE, Buffer.from(JSON.stringify(this._episodeMeta), 'utf-8'));

    // meta/channels
    this.blocks.set(
      BLOCK_META_CHANNELS,
      Buffer.from(JSON.stringify(this._channelsMeta), 'utf-8')
    );

    // meta/models (optional)
    if (this._modelsMeta && this._modelsMeta.models.length > 0) {
      this.blocks.set(BLOCK_META_MODELS, Buffer.from(JSON.stringify(this._modelsMeta), 'utf-8'));
    }
  }

  private buildPendingBlocks(compressor: Compressor | null): PendingBlock[] {
    const pending: PendingBlock[] = [];

    for (const [name, data] of this.blocks.entries()) {
      let compressed: Buffer;
      let flags = 0;

      if (compressor && shouldCompress(name, data)) {
        const result = compressor.compress(data);
        if (result.length < data.length) {
          compressed = result;
          flags = BLOCK_FLAG_COMPRESSED;
        } else {
          compressed = data;
        }
      } else {
        compressed = data;
      }

      pending.push({
        name,
        data,
        compressed,
        flags,
        checksum: crc32C(data),
      });
    }

    return pending;
  }

  private buildStringTable(blocks: PendingBlock[]): Buffer {
    const parts: Buffer[] = [];
    for (const block of blocks) {
      parts.push(Buffer.from(block.name, 'utf-8'));
    }
    return Buffer.concat(parts);
  }

  private buildHeader(
    entryCount: number,
    stringTableOffset: number,
    dataSectionOffset: number,
    totalSize: number
  ): Buffer {
    const header = Buffer.alloc(SHARD_HEADER_SIZE);

    // Magic
    SHARD_MAGIC.copy(header, 0);

    // Version
    header.writeUInt8(SHARD_VERSION_2, 4);

    // Role (W-SHARD = 5)
    header.writeUInt8(WSHARD_ROLE, 5);

    // Flags
    header.writeUInt16LE(0, 6);

    // Alignment
    header.writeUInt8(this.alignment, 8);

    // Compression default
    header.writeUInt8(compressionByte(this.compression), 9);

    // Index entry size
    header.writeUInt16LE(SHARD_INDEX_ENTRY_SIZE, 10);

    // Entry count
    header.writeUInt32LE(entryCount, 12);

    // String table offset
    header.writeBigUInt64LE(BigInt(stringTableOffset), 16);

    // Data section offset
    header.writeBigUInt64LE(BigInt(dataSectionOffset), 24);

    // Schema offset (unused)
    header.writeBigUInt64LE(0n, 32);

    // Total file size
    header.writeBigUInt64LE(BigInt(totalSize), 40);

    // Reserved (48-63) - already zeroed

    return header;
  }

  private buildIndexEntry(block: PendingBlock, nameOffset: number, dataOffset: number): Buffer {
    const entry = Buffer.alloc(SHARD_INDEX_ENTRY_SIZE);
    const nameBytes = Buffer.from(block.name, 'utf-8');

    // Name hash
    entry.writeBigUInt64LE(simpleHash64(block.name), 0);

    // Name offset
    entry.writeUInt32LE(nameOffset, 8);

    // Name length
    entry.writeUInt16LE(nameBytes.length, 12);

    // Flags
    entry.writeUInt16LE(block.flags, 14);

    // Data offset
    entry.writeBigUInt64LE(BigInt(dataOffset), 16);

    // Disk size
    entry.writeBigUInt64LE(BigInt(block.compressed.length), 24);

    // Orig size
    entry.writeBigUInt64LE(BigInt(block.data.length), 32);

    // Checksum
    entry.writeUInt32LE(block.checksum, 40);

    // Reserved (44-47) - already zeroed

    return entry;
  }
}

// ============================================================
// Utility functions
// ============================================================

function align(n: number, alignment: number): number {
  if (alignment === 0) return n;
  return Math.ceil(n / alignment) * alignment;
}

/**
 * Compute sign of second difference for float32 signal.
 */
function computeSign2ndDiff(signal: number[], edgeRule: 'pad' | 'overlap'): Int8Array {
  const T = signal.length;
  const result = new Int8Array(T);

  for (let t = 1; t < T - 1; t++) {
    const secondDiff = signal[t + 1] - 2 * signal[t] + signal[t - 1];
    if (secondDiff > 0) {
      result[t] = 1;
    } else if (secondDiff < 0) {
      result[t] = -1;
    } else {
      result[t] = 0;
    }
  }

  // Edge handling (pad rule: set to 0)
  result[0] = 0;
  if (T > 1) result[T - 1] = 0;

  return result;
}

/**
 * Pack residual signs into bits.
 * 1 -> 1 bit, -1 or 0 -> 0 bit
 */
function packResidualBits(residuals: Int8Array): Buffer {
  const T = residuals.length;
  const byteCount = Math.ceil(T / 8);
  const packed = Buffer.alloc(byteCount);

  for (let t = 0; t < T; t++) {
    if (residuals[t] === 1) {
      const byteIdx = Math.floor(t / 8);
      const bitIdx = t % 8;
      packed[byteIdx] |= 1 << bitIdx;
    }
  }

  return packed;
}
