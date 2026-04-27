/**
 * W-SHARD - World-Model Episode Shard format
 *
 * TypeScript implementation for action-conditioned world modeling.
 * Compatible with Go and Python implementations.
 */

// Re-export types
export {
  // Constants
  SHARD_MAGIC,
  SHARD_VERSION_2,
  WSHARD_ROLE,
  SHARD_HEADER_SIZE,
  SHARD_INDEX_ENTRY_SIZE,
  DEFAULT_ALIGNMENT,
  COMPRESS_NONE,
  COMPRESS_ZSTD,
  COMPRESS_LZ4,
  BLOCK_FLAG_COMPRESSED,

  // Namespace prefixes
  NS_META,
  NS_TIME,
  NS_ACTION,
  NS_SIGNAL,
  NS_OMEN,
  NS_UNCERT,
  NS_RESIDUAL,
  NS_AUX,
  NS_DEBUG,

  // Block names
  BLOCK_REWARD,
  BLOCK_DONE,
  BLOCK_META_WSHARD,
  BLOCK_META_EPISODE,
  BLOCK_META_CHANNELS,
  BLOCK_META_MODELS,
  BLOCK_META_MANIFEST,
  BLOCK_TIME_TICKS,
  BLOCK_TIME_TIMESTAMPS_NS,

  // Residual encoding (Gap 3)
  RESIDUAL_ENCODING_RAW,
  RESIDUAL_ENCODING_COWRIE_BITMASK,

  // VLA Multi-Modal (Gap 5)
  MODALITY_RGB,
  MODALITY_DEPTH,
  MODALITY_LANGUAGE,
  MODALITY_PROPRIOCEPTION,
  MODALITY_AUDIO,
  MODALITY_VIDEO,
  MODALITY_POINTCLOUD,
  type Modality,
  modalityContentType,
  multiModalSignalBlock,

  // Latent Action (Gap 2)
  LATENT_ACTION_CHANNEL,
  LATENT_CODEBOOK_CHANNEL,
  latentActionBlock,
  latentCodebookBlock,

  // Types
  type CompressionType,
  type CompressionLevel,
  type DType,
  type ShardHeader,
  type IndexEntry,
  type WShardMeta,
  type EpisodeMeta,
  type ChannelDef,
  type ChannelsMeta,
  type ModelDef,
  type ModelsMeta,

  // Functions
  dtypeSize,
  createShardHeader,
  serializeHeader,
  parseHeader,
  serializeIndexEntry,
  parseIndexEntry,
  isCompressed,
  compressionByte,
  compressionFromByte,
  blockPath,
  normalizePath,
  signalBlock,
  actionBlock,
  omenBlock,
  uncertBlock,
  residualSign2ndDiffBlock,
  crc32C,
  crc32IEEE,
  simpleHash64,
  encodeFloat32,
  encodeFloat32_2D,
  encodeInt32,
  encodeInt64,
  encodeBool,
  decodeFloat32,
  decodeFloat32_2D,
  decodeInt32,
  decodeInt64,
  decodeBool,
} from './types.js';

// Re-export compression
export {
  Compressor,
  compressBlock,
  compressBlockAsync,
  decompressBlock,
  decompressBlockAsync,
  shouldCompress,
  initZstd,
  hasZstdSupport,
} from './compress.js';

// Re-export reader
export { WShardReader } from './reader.js';

// Re-export writer
export { WShardWriter } from './writer.js';

// Re-export stream writer (Gap 4)
export { WShardStreamWriter, type StreamChannelDef } from './stream-writer.js';
