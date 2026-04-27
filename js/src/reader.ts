/**
 * W-SHARD reader implementation.
 *
 * Reads W-SHARD files with the Shard v2 format (role=0x05).
 *
 * Shard v2 layout:
 *     Header (64 bytes)
 *     Index entries (entry_count * 48 bytes)
 *     String table (variable)
 *     [Padding to alignment]
 *     Data section (aligned entries)
 */

import * as fs from 'fs';
import {
  ShardHeader,
  IndexEntry,
  WShardMeta,
  EpisodeMeta,
  ChannelsMeta,
  ModelsMeta,
  ChannelDef,
  SHARD_HEADER_SIZE,
  SHARD_INDEX_ENTRY_SIZE,
  BLOCK_FLAG_COMPRESSED,
  BLOCK_META_WSHARD,
  BLOCK_META_EPISODE,
  BLOCK_META_CHANNELS,
  BLOCK_META_MODELS,
  BLOCK_TIME_TICKS,
  BLOCK_TIME_TIMESTAMPS_NS,
  BLOCK_REWARD,
  BLOCK_DONE,
  parseHeader,
  parseIndexEntry,
  isCompressed,
  compressionFromByte,
  signalBlock,
  actionBlock,
  omenBlock,
  uncertBlock,
  residualSign2ndDiffBlock,
  latentActionBlock,
  latentCodebookBlock,
  multiModalSignalBlock,
  crc32C,
  decodeFloat32,
  decodeFloat32_2D,
  decodeInt32,
  decodeInt64,
  decodeBool,
} from './types.js';
import type { Modality } from './types.js';
import { Compressor, decompressBlock } from './compress.js';
import type { CompressionType } from './types.js';

/**
 * Reader for W-SHARD files.
 */
export class WShardReader {
  private path: string;
  private fd: number | null = null;
  private header: ShardHeader | null = null;
  private entries: IndexEntry[] = [];
  private entryByName: Map<string, IndexEntry> = new Map();
  private stringTable: Buffer | null = null;
  private compressionDefault: CompressionType = 'none';

  // Cached metadata
  private _wshardMeta: WShardMeta | null = null;
  private _episodeMeta: EpisodeMeta | null = null;
  private _channelsMeta: ChannelsMeta | null = null;
  private _modelsMeta: ModelsMeta | null = null;

  // Lazy decompressor
  private decompressor: Compressor | null = null;

  constructor(path: string) {
    this.path = path;
  }

  /**
   * Open the file for reading.
   */
  async open(): Promise<void> {
    if (this.fd !== null) {
      return;
    }

    this.fd = fs.openSync(this.path, 'r');

    // Read header
    const headerBuf = Buffer.alloc(SHARD_HEADER_SIZE);
    fs.readSync(this.fd, headerBuf, 0, SHARD_HEADER_SIZE, 0);
    this.header = parseHeader(headerBuf);
    this.compressionDefault = compressionFromByte(this.header.compressionDefault);

    // Read index entries
    const indexSize = this.header.entryCount * SHARD_INDEX_ENTRY_SIZE;
    const indexData = Buffer.alloc(indexSize);
    fs.readSync(this.fd, indexData, 0, indexSize, SHARD_HEADER_SIZE);

    for (let i = 0; i < this.header.entryCount; i++) {
      const offset = i * SHARD_INDEX_ENTRY_SIZE;
      const entryBuf = indexData.subarray(offset, offset + SHARD_INDEX_ENTRY_SIZE);
      const entry = parseIndexEntry(entryBuf);
      this.entries.push(entry);
    }

    // Read string table
    const stringTableOffset = Number(this.header.stringTableOffset);
    const stringTableSize = Number(this.header.dataSectionOffset) - stringTableOffset;
    this.stringTable = Buffer.alloc(stringTableSize);
    fs.readSync(this.fd, this.stringTable, 0, stringTableSize, stringTableOffset);

    // Extract names from string table
    for (const entry of this.entries) {
      if (entry.nameOffset + entry.nameLen <= this.stringTable.length) {
        entry.name = this.stringTable
          .subarray(entry.nameOffset, entry.nameOffset + entry.nameLen)
          .toString('utf-8');
      }
      this.entryByName.set(entry.name, entry);
    }

    // Load metadata
    await this.loadMetadata();
  }

  private async loadMetadata(): Promise<void> {
    // Load meta/wshard
    const wshardData = await this.readBlock(BLOCK_META_WSHARD);
    this._wshardMeta = JSON.parse(wshardData.toString('utf-8'));

    // Load meta/episode
    const episodeData = await this.readBlock(BLOCK_META_EPISODE);
    this._episodeMeta = JSON.parse(episodeData.toString('utf-8'));

    // Load meta/channels
    const channelsData = await this.readBlock(BLOCK_META_CHANNELS);
    this._channelsMeta = JSON.parse(channelsData.toString('utf-8'));

    // Load meta/models (optional)
    if (this.hasBlock(BLOCK_META_MODELS)) {
      const modelsData = await this.readBlock(BLOCK_META_MODELS);
      this._modelsMeta = JSON.parse(modelsData.toString('utf-8'));
    }
  }

  // ============================================================
  // Metadata accessors
  // ============================================================

  get wshardMeta(): WShardMeta | null {
    return this._wshardMeta;
  }

  get episodeMeta(): EpisodeMeta | null {
    return this._episodeMeta;
  }

  get channelsMeta(): ChannelsMeta | null {
    return this._channelsMeta;
  }

  get modelsMeta(): ModelsMeta | null {
    return this._modelsMeta;
  }

  /**
   * Get the number of timesteps.
   */
  get lengthT(): number {
    return this._episodeMeta?.length_T ?? 0;
  }

  // ============================================================
  // Block listing
  // ============================================================

  /**
   * Get all block names.
   */
  blockNames(): string[] {
    return this.entries.map((e) => e.name);
  }

  /**
   * Check if a block exists.
   */
  hasBlock(name: string): boolean {
    return this.entryByName.has(name);
  }

  /**
   * Get block size.
   */
  blockSize(name: string): number {
    const entry = this.entryByName.get(name);
    if (!entry) {
      throw new Error(`Block not found: ${name}`);
    }
    return Number(entry.origSize);
  }

  // ============================================================
  // Raw block reading
  // ============================================================

  /**
   * Read a block's raw bytes, decompressing if needed.
   */
  async readBlock(name: string): Promise<Buffer> {
    const entry = this.entryByName.get(name);
    if (!entry) {
      throw new Error(`Block not found: ${name}`);
    }

    const fd = this.fd!;
    const data = Buffer.alloc(Number(entry.diskSize));
    fs.readSync(fd, data, 0, Number(entry.diskSize), Number(entry.dataOffset));

    // Decompress if needed — detect compression type per-block from flags
    if (isCompressed(entry) && entry.diskSize !== entry.origSize) {
      let blockCompType: CompressionType = this.compressionDefault;
      // Per-block compression type flags (matching Go's EntryFlag bits)
      if (entry.flags & 0x0004) {  // BLOCK_FLAG_LZ4
        blockCompType = 'lz4';
      } else if (entry.flags & 0x0002) {  // BLOCK_FLAG_ZSTD
        blockCompType = 'zstd';
      }
      const decompressor = new Compressor(blockCompType, 'default');
      const uncompressed = decompressor.decompress(data, Number(entry.origSize));
      if (entry.checksum !== 0) {
        const actual = crc32C(uncompressed);
        if (actual !== entry.checksum) {
          throw new Error(
            `Checksum mismatch for ${name}: expected ${entry.checksum.toString(16)}, got ${actual.toString(16)}`
          );
        }
      }
      return uncompressed;
    }

    if (entry.checksum !== 0) {
      const actual = crc32C(data);
      if (actual !== entry.checksum) {
        throw new Error(
          `Checksum mismatch for ${name}: expected ${entry.checksum.toString(16)}, got ${actual.toString(16)}`
        );
      }
    }

    return data;
  }

  // ============================================================
  // Time accessors
  // ============================================================

  /**
   * Get time/ticks block as int32 array.
   */
  async getTimeTicks(): Promise<number[]> {
    const data = await this.readBlock(BLOCK_TIME_TICKS);
    return decodeInt32(data);
  }

  /**
   * Get time/timestamps_ns block as int64 array.
   */
  async getTimeTimestampsNs(): Promise<bigint[]> {
    const data = await this.readBlock(BLOCK_TIME_TIMESTAMPS_NS);
    return decodeInt64(data);
  }

  // ============================================================
  // Action accessors
  // ============================================================

  /**
   * Get action block as raw bytes.
   */
  async getAction(name: string): Promise<Buffer> {
    return this.readBlock(actionBlock(name));
  }

  /**
   * Get action block as 1D float32 array.
   */
  async getActionFloat32(name: string): Promise<number[]> {
    const data = await this.readBlock(actionBlock(name));
    return decodeFloat32(data);
  }

  /**
   * Get action block as 2D float32 array [T][D].
   */
  async getActionFloat32_2D(name: string, D: number): Promise<number[][]> {
    const data = await this.readBlock(actionBlock(name));
    return decodeFloat32_2D(data, D);
  }

  // ============================================================
  // Control accessors
  // ============================================================

  /**
   * Get reward block as float32 array.
   */
  async getReward(): Promise<number[]> {
    const data = await this.readBlock(BLOCK_REWARD);
    return decodeFloat32(data);
  }

  /**
   * Get done block as bool array.
   */
  async getDone(): Promise<boolean[]> {
    const data = await this.readBlock(BLOCK_DONE);
    return decodeBool(data);
  }

  // ============================================================
  // Signal accessors
  // ============================================================

  /**
   * Get signal block as raw bytes.
   */
  async getSignal(channelID: string): Promise<Buffer> {
    return this.readBlock(signalBlock(channelID));
  }

  /**
   * Get signal block as 1D float32 array.
   */
  async getSignalFloat32(channelID: string): Promise<number[]> {
    const data = await this.readBlock(signalBlock(channelID));
    return decodeFloat32(data);
  }

  /**
   * Get signal block as 2D float32 array [T][D].
   */
  async getSignalFloat32_2D(channelID: string, D: number): Promise<number[][]> {
    const data = await this.readBlock(signalBlock(channelID));
    return decodeFloat32_2D(data, D);
  }

  // ============================================================
  // Omen accessors
  // ============================================================

  /**
   * Get omen block as raw bytes.
   */
  async getOmen(channelID: string, modelID: string): Promise<Buffer> {
    return this.readBlock(omenBlock(channelID, modelID));
  }

  /**
   * Get omen block as 1D float32 array.
   */
  async getOmenFloat32(channelID: string, modelID: string): Promise<number[]> {
    const data = await this.readBlock(omenBlock(channelID, modelID));
    return decodeFloat32(data);
  }

  /**
   * Get omen block as 2D float32 array [T][D].
   */
  async getOmenFloat32_2D(channelID: string, modelID: string, D: number): Promise<number[][]> {
    const data = await this.readBlock(omenBlock(channelID, modelID));
    return decodeFloat32_2D(data, D);
  }

  // ============================================================
  // Uncertainty accessors
  // ============================================================

  /**
   * Get uncertainty block as raw bytes.
   */
  async getUncert(channelID: string, modelID: string, uncertType: string): Promise<Buffer> {
    return this.readBlock(uncertBlock(channelID, modelID, uncertType));
  }

  /**
   * Get uncertainty block as 1D float32 array.
   */
  async getUncertFloat32(
    channelID: string,
    modelID: string,
    uncertType: string
  ): Promise<number[]> {
    const data = await this.readBlock(uncertBlock(channelID, modelID, uncertType));
    return decodeFloat32(data);
  }

  // ============================================================
  // Gap 5: Multi-modal signal accessors
  // ============================================================

  /**
   * Get multi-modal signal block as raw bytes.
   */
  async getMultiModalSignal(group: string, modality: Modality): Promise<Buffer> {
    return this.readBlock(multiModalSignalBlock(group, modality));
  }

  /**
   * Get multi-modal signal as 1D float32 array.
   */
  async getMultiModalSignalFloat32(group: string, modality: Modality): Promise<number[]> {
    const data = await this.readBlock(multiModalSignalBlock(group, modality));
    return decodeFloat32(data);
  }

  /**
   * Get multi-modal signal as 2D float32 array [T][D].
   */
  async getMultiModalSignalFloat32_2D(group: string, modality: Modality, D: number): Promise<number[][]> {
    const data = await this.readBlock(multiModalSignalBlock(group, modality));
    return decodeFloat32_2D(data, D);
  }

  // ============================================================
  // Gap 2: Latent action accessors
  // ============================================================

  /**
   * Get latent action embeddings as raw bytes.
   */
  async getLatentAction(modelID: string): Promise<Buffer> {
    return this.readBlock(latentActionBlock(modelID));
  }

  /**
   * Get latent action embeddings as 1D float32 array.
   */
  async getLatentActionFloat32(modelID: string): Promise<number[]> {
    const data = await this.readBlock(latentActionBlock(modelID));
    return decodeFloat32(data);
  }

  /**
   * Get latent action embeddings as 2D float32 array [T][latent_dim].
   */
  async getLatentActionFloat32_2D(modelID: string, latentDim: number): Promise<number[][]> {
    const data = await this.readBlock(latentActionBlock(modelID));
    return decodeFloat32_2D(data, latentDim);
  }

  /**
   * Get latent action codebook indices as raw bytes.
   */
  async getLatentActionCodebook(modelID: string): Promise<Buffer> {
    return this.readBlock(latentCodebookBlock(modelID));
  }

  /**
   * Get latent action codebook indices as int32 array.
   */
  async getLatentActionCodebookInt32(modelID: string): Promise<number[]> {
    const data = await this.readBlock(latentCodebookBlock(modelID));
    return decodeInt32(data);
  }

  // ============================================================
  // Residual accessors
  // ============================================================

  /**
   * Get sign2nddiff residual as packed bytes.
   */
  async getResidualSign2ndDiff(channelID: string): Promise<Buffer> {
    return this.readBlock(residualSign2ndDiffBlock(channelID));
  }

  /**
   * Get sign2nddiff residual unpacked to int8 array.
   */
  async getResidualSign2ndDiffUnpacked(channelID: string): Promise<Int8Array> {
    const packed = await this.getResidualSign2ndDiff(channelID);
    const edgeRule = this._wshardMeta?.residual_edges ?? 'pad';
    return unpackResidualBits(packed, this.lengthT, edgeRule);
  }

  // ============================================================
  // Channel helpers
  // ============================================================

  /**
   * Get channel definition by ID.
   */
  getChannelDef(channelID: string): ChannelDef | null {
    if (!this._channelsMeta) {
      return null;
    }
    return this._channelsMeta.channels.find((ch) => ch.id === channelID) ?? null;
  }

  /**
   * Get all channel IDs.
   */
  channelIDs(): string[] {
    if (!this._channelsMeta) {
      return [];
    }
    return this._channelsMeta.channels.map((ch) => ch.id);
  }

  // ============================================================
  // Cleanup
  // ============================================================

  /**
   * Close the reader.
   */
  async close(): Promise<void> {
    if (this.fd !== null) {
      fs.closeSync(this.fd);
      this.fd = null;
    }
    if (this.decompressor !== null) {
      this.decompressor.close();
      this.decompressor = null;
    }
  }
}

// ============================================================
// Residual unpacking
// ============================================================

/**
 * Unpack residual bits to int8 array.
 * Values are -1, 0, or 1.
 */
function unpackResidualBits(packed: Buffer, T: number, edgeRule: 'pad' | 'overlap' = 'pad'): Int8Array {
  const result = new Int8Array(T);
  for (let i = 0; i < T; i++) {
    const byteIdx = Math.floor(i / 8);
    const bitIdx = i % 8;
    if (byteIdx < packed.length) {
      const bit = (packed[byteIdx] >> bitIdx) & 1;
      result[i] = bit === 1 ? 1 : -1;
    }
  }
  // Only zero-pad edges when edgeRule is 'pad'
  if (edgeRule === 'pad') {
    if (T >= 1) result[0] = 0;
    if (T >= 2) result[T - 1] = 0;
  }
  return result;
}
