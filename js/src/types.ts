/**
 * W-SHARD type definitions.
 *
 * W-SHARD v0.1: World-Model Episode Shard format for action-conditioned world modeling.
 * Built on Shard v2 container format with role=0x05.
 *
 * Core data structures:
 * - ShardHeader: 64-byte v2 header
 * - IndexEntry: 48-byte index entry
 * - WShardMeta: W-SHARD specific metadata
 */

// Magic bytes
export const SHARD_MAGIC = Buffer.from('SHRD', 'ascii');

// Version
export const SHARD_VERSION_2 = 0x02;

// W-SHARD role in Shard v2
export const WSHARD_ROLE = 0x05;

// Header and index sizes
export const SHARD_HEADER_SIZE = 64;
export const SHARD_INDEX_ENTRY_SIZE = 48;

// Default alignment (AVX-friendly)
export const DEFAULT_ALIGNMENT = 32;

// Compression types
export const COMPRESS_NONE = 0x00;
export const COMPRESS_ZSTD = 0x01;
export const COMPRESS_LZ4 = 0x02;

// Content types (entry.reserved[0:2]) - Shard v2.1
export const CONTENT_TYPE_UNKNOWN = 0x0000;
export const CONTENT_TYPE_TENSOR = 0x0001;  // TensorV1 encoded tensor
export const CONTENT_TYPE_JSON = 0x0002;    // Standard JSON
export const CONTENT_TYPE_COWRIE = 0x0003;  // Cowrie binary format
export const CONTENT_TYPE_GLYPH = 0x0004;   // GLYPH text format
export const CONTENT_TYPE_TEXT = 0x0005;    // Plain text (UTF-8)
export const CONTENT_TYPE_IMAGE = 0x0006;   // Image (PNG, JPEG, etc.)
export const CONTENT_TYPE_AUDIO = 0x0007;   // Audio (WAV, MP3, etc.)
export const CONTENT_TYPE_VIDEO = 0x0008;   // Video (MP4, WebM, etc.)
export const CONTENT_TYPE_PROTO = 0x0009;   // Protocol Buffers
export const CONTENT_TYPE_BLOB = 0x000A;    // Opaque binary blob
export const CONTENT_TYPE_USER_BASE = 0x8000;

// Header flag for content types
export const SHARD_FLAG_HAS_CONTENT_TYPES = 0x0080;

// Block flag for compression
export const BLOCK_FLAG_COMPRESSED = 0x0001;

// Namespace prefixes for block names
export const NS_META = 'meta';
export const NS_TIME = 'time';
export const NS_ACTION = 'action';
export const NS_SIGNAL = 'signal';
export const NS_OMEN = 'omen';
export const NS_UNCERT = 'uncert';
export const NS_RESIDUAL = 'residual';
export const NS_AUX = 'aux';
export const NS_DEBUG = 'debug';

// Special block names (no namespace)
export const BLOCK_REWARD = 'reward';
export const BLOCK_DONE = 'done';

// Required metadata block names
export const BLOCK_META_WSHARD = 'meta/wshard';
export const BLOCK_META_EPISODE = 'meta/episode';
export const BLOCK_META_CHANNELS = 'meta/channels';
export const BLOCK_META_MODELS = 'meta/models';
export const BLOCK_META_MANIFEST = 'meta/manifest';

// Time block names
export const BLOCK_TIME_TICKS = 'time/ticks';
export const BLOCK_TIME_TIMESTAMPS_NS = 'time/timestamps_ns';

// Residual encoding identifiers (Gap 3: BITMASK integration)
export const RESIDUAL_ENCODING_RAW = 'raw';
export const RESIDUAL_ENCODING_COWRIE_BITMASK = 'cowrie_bitmask';

// Gap 5: VLA Multi-Modal modality constants
export const MODALITY_RGB = 'rgb';
export const MODALITY_DEPTH = 'depth';
export const MODALITY_LANGUAGE = 'language';
export const MODALITY_PROPRIOCEPTION = 'proprioception';
export const MODALITY_AUDIO = 'audio';
export const MODALITY_VIDEO = 'video';
export const MODALITY_POINTCLOUD = 'pointcloud';

export type Modality =
  | typeof MODALITY_RGB
  | typeof MODALITY_DEPTH
  | typeof MODALITY_LANGUAGE
  | typeof MODALITY_PROPRIOCEPTION
  | typeof MODALITY_AUDIO
  | typeof MODALITY_VIDEO
  | typeof MODALITY_POINTCLOUD;

// Gap 2: Latent Action Storage constants
export const LATENT_ACTION_CHANNEL = 'latent_action';
export const LATENT_CODEBOOK_CHANNEL = 'latent_action_codebook';

/**
 * Get the shard v2 content type code for a modality.
 */
export function modalityContentType(m: Modality): number {
  switch (m) {
    case MODALITY_RGB: return CONTENT_TYPE_IMAGE;
    case MODALITY_DEPTH: return CONTENT_TYPE_TENSOR;
    case MODALITY_LANGUAGE: return CONTENT_TYPE_TEXT;
    case MODALITY_PROPRIOCEPTION: return CONTENT_TYPE_TENSOR;
    case MODALITY_AUDIO: return CONTENT_TYPE_AUDIO;
    case MODALITY_VIDEO: return CONTENT_TYPE_VIDEO;
    case MODALITY_POINTCLOUD: return CONTENT_TYPE_TENSOR;
    default: return CONTENT_TYPE_UNKNOWN;
  }
}

/**
 * Construct a latent action omen block path.
 */
export function latentActionBlock(modelID: string): string {
  return blockPath(NS_OMEN, LATENT_ACTION_CHANNEL, modelID);
}

/**
 * Construct a latent action codebook omen block path.
 */
export function latentCodebookBlock(modelID: string): string {
  return blockPath(NS_OMEN, LATENT_CODEBOOK_CHANNEL, modelID);
}

/**
 * Construct a multi-modal signal block path.
 */
export function multiModalSignalBlock(group: string, modality: Modality): string {
  return blockPath(NS_SIGNAL, group, modality);
}

/**
 * Compression type enumeration.
 */
export type CompressionType = 'none' | 'zstd' | 'lz4';

/**
 * Compression level enumeration.
 */
export type CompressionLevel = 'fastest' | 'default' | 'best';

/**
 * Data types for tensors (13 total, matching Python DType enum).
 */
export type DType =
  | 'f32' | 'f64' | 'f16' | 'bf16'
  | 'i32' | 'i64' | 'i16' | 'i8'
  | 'u8' | 'u16' | 'u32' | 'u64'
  | 'bool';

/**
 * Get byte size for a dtype.
 * Throws on unknown dtype to prevent silent corruption.
 */
export function dtypeSize(dt: DType): number {
  switch (dt) {
    case 'f64':
    case 'i64':
    case 'u64':
      return 8;
    case 'f32':
    case 'i32':
    case 'u32':
      return 4;
    case 'f16':
    case 'bf16':
    case 'i16':
    case 'u16':
      return 2;
    case 'i8':
    case 'u8':
    case 'bool':
      return 1;
    default:
      throw new Error(`Unknown dtype: '${dt}'`);
  }
}

/**
 * Shard v2 header (64 bytes).
 */
export interface ShardHeader {
  magic: Buffer;
  version: number;
  role: number;
  flags: number;
  alignment: number;
  compressionDefault: number;
  entryCount: number;
  stringTableOffset: bigint;
  dataSectionOffset: bigint;
  schemaOffset: bigint;
  totalFileSize: bigint;
}

/**
 * Shard v2 index entry (48 bytes).
 */
export interface IndexEntry {
  nameHash: bigint;
  nameOffset: number;
  nameLen: number;
  flags: number;
  dataOffset: bigint;
  diskSize: bigint;
  origSize: bigint;
  checksum: number;
  contentType: number; // Parsed from reserved field (v2.1)
  name: string; // Populated after reading string table
}

/**
 * W-SHARD metadata (meta/wshard block).
 */
export interface WShardMeta {
  format: string;
  version: string;
  residual_edges: 'pad' | 'overlap';
  residual_encoding?: string;
  schema_hash?: string;
  schema_hash_kind?: string;
}

/**
 * Timebase metadata.
 */
export interface TimebaseMeta {
  type: 'ticks' | 'timestamps_ns';
  dt_ns?: number;
}

/**
 * Episode metadata (meta/episode block).
 */
export interface EpisodeMeta {
  episode_id: string;
  length_T: number;
  source?: string;
  env_id?: string;
  timebase: TimebaseMeta;
  action_space?: 'discrete' | 'continuous';
  // Gap 1: Chunked episode fields
  chunk_index?: number;
  total_chunks?: number;
  timestep_range?: [number, number];
}

/**
 * Channel definition.
 */
export interface ChannelDef {
  id: string;
  dtype: DType;
  shape: number[];
  unit?: string;
  description?: string;
  // Gap 5: VLA Multi-Modal
  modality?: Modality;
  sampling_rate_hz?: number;
  content_type?: string;
}

/**
 * Channels metadata (meta/channels block).
 */
export interface ChannelsMeta {
  channels: ChannelDef[];
}

/**
 * Model definition.
 */
export interface ModelDef {
  id: string;
  name: string;
  type?: string;
  version?: string;
  // Gap 2: Latent Action Storage
  latent_dim?: number;
  codebook_size?: number;
  action_type?: 'continuous' | 'discrete' | 'vq';
}

/**
 * Models metadata (meta/models block).
 */
export interface ModelsMeta {
  models: ModelDef[];
}

/**
 * Create a new ShardHeader with W-SHARD defaults.
 */
export function createShardHeader(): ShardHeader {
  return {
    magic: SHARD_MAGIC,
    version: SHARD_VERSION_2,
    role: WSHARD_ROLE,
    flags: 0,
    alignment: DEFAULT_ALIGNMENT,
    compressionDefault: COMPRESS_NONE,
    entryCount: 0,
    stringTableOffset: 0n,
    dataSectionOffset: BigInt(SHARD_HEADER_SIZE),
    schemaOffset: 0n,
    totalFileSize: 0n,
  };
}

/**
 * Serialize header to 64 bytes.
 */
export function serializeHeader(header: ShardHeader): Buffer {
  const buf = Buffer.alloc(SHARD_HEADER_SIZE);

  // Magic (4 bytes)
  header.magic.copy(buf, 0);

  // Version (1 byte)
  buf.writeUInt8(header.version, 4);

  // Role (1 byte)
  buf.writeUInt8(header.role, 5);

  // Flags (2 bytes, little-endian)
  buf.writeUInt16LE(header.flags, 6);

  // Alignment (1 byte)
  buf.writeUInt8(header.alignment, 8);

  // Compression default (1 byte)
  buf.writeUInt8(header.compressionDefault, 9);

  // Index entry size (2 bytes, little-endian) - MUST be 48
  buf.writeUInt16LE(SHARD_INDEX_ENTRY_SIZE, 10);

  // Entry count (4 bytes, little-endian)
  buf.writeUInt32LE(header.entryCount, 12);

  // String table offset (8 bytes, little-endian)
  buf.writeBigUInt64LE(header.stringTableOffset, 16);

  // Data section offset (8 bytes, little-endian)
  buf.writeBigUInt64LE(header.dataSectionOffset, 24);

  // Schema offset (8 bytes, little-endian)
  buf.writeBigUInt64LE(header.schemaOffset, 32);

  // Total file size (8 bytes, little-endian)
  buf.writeBigUInt64LE(header.totalFileSize, 40);

  // Reserved (16 bytes) - already zeroed

  return buf;
}

/**
 * Parse header from 64 bytes.
 */
export function parseHeader(data: Buffer): ShardHeader {
  if (data.length < SHARD_HEADER_SIZE) {
    throw new Error(`Header too short: ${data.length} < ${SHARD_HEADER_SIZE}`);
  }

  const magic = data.subarray(0, 4);
  if (!magic.equals(SHARD_MAGIC)) {
    throw new Error(`Invalid magic: ${magic.toString('hex')}`);
  }

  const version = data.readUInt8(4);
  if (version !== SHARD_VERSION_2) {
    throw new Error(`Unsupported version: ${version}`);
  }

  const role = data.readUInt8(5);
  if (role !== WSHARD_ROLE) {
    throw new Error(`Not a W-SHARD file: role=${role}, expected=${WSHARD_ROLE}`);
  }

  return {
    magic,
    version,
    role,
    flags: data.readUInt16LE(6),
    alignment: data.readUInt8(8),
    compressionDefault: data.readUInt8(9),
    entryCount: data.readUInt32LE(12),
    stringTableOffset: data.readBigUInt64LE(16),
    dataSectionOffset: data.readBigUInt64LE(24),
    schemaOffset: data.readBigUInt64LE(32),
    totalFileSize: data.readBigUInt64LE(40),
  };
}

/**
 * Serialize index entry to 48 bytes.
 */
export function serializeIndexEntry(entry: IndexEntry): Buffer {
  const buf = Buffer.alloc(SHARD_INDEX_ENTRY_SIZE);

  buf.writeBigUInt64LE(entry.nameHash, 0);
  buf.writeUInt32LE(entry.nameOffset, 8);
  buf.writeUInt16LE(entry.nameLen, 12);
  buf.writeUInt16LE(entry.flags, 14);
  buf.writeBigUInt64LE(entry.dataOffset, 16);
  buf.writeBigUInt64LE(entry.diskSize, 24);
  buf.writeBigUInt64LE(entry.origSize, 32);
  buf.writeUInt32LE(entry.checksum, 40);
  // Reserved (4 bytes) - already zeroed

  return buf;
}

/**
 * Parse index entry from 48 bytes.
 */
export function parseIndexEntry(data: Buffer): IndexEntry {
  if (data.length < SHARD_INDEX_ENTRY_SIZE) {
    throw new Error(`Entry too short: ${data.length} < ${SHARD_INDEX_ENTRY_SIZE}`);
  }

  const reserved = data.readUInt32LE(44);
  return {
    nameHash: data.readBigUInt64LE(0),
    nameOffset: data.readUInt32LE(8),
    nameLen: data.readUInt16LE(12),
    flags: data.readUInt16LE(14),
    dataOffset: data.readBigUInt64LE(16),
    diskSize: data.readBigUInt64LE(24),
    origSize: data.readBigUInt64LE(32),
    checksum: data.readUInt32LE(40),
    contentType: reserved & 0xFFFF,
    name: '', // Populated after reading string table
  };
}

/**
 * Get human-readable content type name.
 */
export function contentTypeName(ct: number): string {
  switch (ct) {
    case CONTENT_TYPE_TENSOR: return 'tensor';
    case CONTENT_TYPE_JSON: return 'json';
    case CONTENT_TYPE_COWRIE: return 'cowrie';
    case CONTENT_TYPE_GLYPH: return 'glyph';
    case CONTENT_TYPE_TEXT: return 'text';
    case CONTENT_TYPE_IMAGE: return 'image';
    case CONTENT_TYPE_AUDIO: return 'audio';
    case CONTENT_TYPE_VIDEO: return 'video';
    case CONTENT_TYPE_PROTO: return 'proto';
    case CONTENT_TYPE_BLOB: return 'blob';
    default:
      if (ct >= CONTENT_TYPE_USER_BASE) {
        return `user:${ct - CONTENT_TYPE_USER_BASE}`;
      }
      return 'unknown';
  }
}

/**
 * Check if entry is compressed.
 */
export function isCompressed(entry: IndexEntry): boolean {
  return (entry.flags & BLOCK_FLAG_COMPRESSED) !== 0;
}

/**
 * Get compression byte from type.
 */
export function compressionByte(ct: CompressionType): number {
  switch (ct) {
    case 'zstd':
      return COMPRESS_ZSTD;
    case 'lz4':
      return COMPRESS_LZ4;
    default:
      return COMPRESS_NONE;
  }
}

/**
 * Get compression type from byte.
 */
export function compressionFromByte(b: number): CompressionType {
  switch (b) {
    case COMPRESS_ZSTD:
      return 'zstd';
    case COMPRESS_LZ4:
      return 'lz4';
    default:
      return 'none';
  }
}

// ============================================================
// Block path helpers
// ============================================================

/**
 * Construct a block path from components.
 */
export function blockPath(...parts: string[]): string {
  return parts.join('/');
}

/**
 * Normalize a path to use the canonical prefix.
 */
export function normalizePath(namespace: string, name: string): string {
  const prefix = namespace + '/';
  if (name.startsWith(prefix)) {
    return name;
  }
  return prefix + name;
}

/**
 * Get signal block name.
 */
export function signalBlock(channelID: string): string {
  return normalizePath(NS_SIGNAL, channelID);
}

/**
 * Get action block name.
 */
export function actionBlock(name: string): string {
  return normalizePath(NS_ACTION, name);
}

/**
 * Get omen block name.
 */
export function omenBlock(channelID: string, modelID: string): string {
  return blockPath(NS_OMEN, channelID, modelID);
}

/**
 * Get uncertainty block name.
 */
export function uncertBlock(channelID: string, modelID: string, uncertType: string): string {
  return blockPath(NS_UNCERT, channelID, modelID, uncertType);
}

/**
 * Get sign2nddiff residual block name.
 */
export function residualSign2ndDiffBlock(channelID: string): string {
  return blockPath(NS_RESIDUAL, channelID, 'sign2nddiff');
}

// ============================================================
// Hashing and checksum
// ============================================================

// CRC32C (Castagnoli) table — matching Go's crc32.MakeTable(crc32.Castagnoli)
let crc32CTable: number[] | null = null;

function getCrc32CTable(): number[] {
  if (crc32CTable !== null) {
    return crc32CTable;
  }

  // Castagnoli polynomial: 0x82F63B78 (reversed form of 0x1EDC6F41)
  const polynomial = 0x82f63b78;
  crc32CTable = new Array(256);

  for (let i = 0; i < 256; i++) {
    let crc = i;
    for (let j = 0; j < 8; j++) {
      if (crc & 1) {
        crc = (crc >>> 1) ^ polynomial;
      } else {
        crc >>>= 1;
      }
    }
    crc32CTable[i] = crc >>> 0;
  }

  return crc32CTable;
}

/**
 * CRC32C (Castagnoli) checksum — matching Go's crc32.Castagnoli.
 *
 * @deprecated Use crc32C() instead. This alias exists for migration.
 */
export function crc32IEEE(data: Buffer): number {
  return crc32C(data);
}

/**
 * CRC32C (Castagnoli) checksum — matching Go's crc32.Castagnoli.
 */
export function crc32C(data: Buffer): number {
  const table = getCrc32CTable();
  let crc = 0xffffffff;

  for (const byte of data) {
    crc = (crc >>> 8) ^ table[(crc ^ byte) & 0xff];
  }

  return (crc ^ 0xffffffff) >>> 0;
}

// xxHash64 state — initialized asynchronously from xxhash-wasm
let xxhash64Fn: ((input: string) => bigint) | null = null;
let xxhashInitPromise: Promise<void> | null = null;

/**
 * Initialize xxHash64. Call once at startup before using nameHash64().
 * Falls back to FNV-1a if xxhash-wasm is unavailable.
 */
export async function initXxHash(): Promise<void> {
  if (xxhash64Fn) return;
  if (xxhashInitPromise) return xxhashInitPromise;

  xxhashInitPromise = (async () => {
    try {
      const mod = await import('xxhash-wasm');
      const hasher = await mod.default();
      xxhash64Fn = (s: string) => hasher.h64(s);
    } catch {
      // xxhash-wasm not available — use built-in fallback
      xxhash64Fn = fnv1aHash64;
    }
  })();

  return xxhashInitPromise;
}

/**
 * FNV-1a 64-bit hash — fallback when xxhash-wasm is unavailable.
 */
function fnv1aHash64(s: string): bigint {
  let h = 14695981039346656037n;
  for (const c of s) {
    h ^= BigInt(c.charCodeAt(0));
    h = (h * 1099511628211n) & 0xffffffffffffffffn;
  }
  return h;
}

/**
 * Compute xxHash64 name hash, matching Go's xxhash.Sum64String().
 * Falls back to FNV-1a if initXxHash() has not been called.
 *
 * @deprecated Alias for nameHash64(). Use nameHash64() in new code.
 */
export function simpleHash64(s: string): bigint {
  return nameHash64(s);
}

/**
 * Compute name hash for index entries.
 * Uses xxHash64 (matching Go) if initialized, otherwise FNV-1a fallback.
 */
export function nameHash64(s: string): bigint {
  if (xxhash64Fn) {
    return xxhash64Fn(s);
  }
  return fnv1aHash64(s);
}

// ============================================================
// Encoding helpers
// ============================================================

/**
 * Encode float32 array to bytes (little-endian).
 */
export function encodeFloat32(data: number[]): Buffer {
  const buf = Buffer.alloc(data.length * 4);
  for (let i = 0; i < data.length; i++) {
    buf.writeFloatLE(data[i], i * 4);
  }
  return buf;
}

/**
 * Encode 2D float32 array to bytes [T][D] -> flat.
 */
export function encodeFloat32_2D(data: number[][]): Buffer {
  if (data.length === 0) return Buffer.alloc(0);
  const T = data.length;
  const D = data[0].length;
  const buf = Buffer.alloc(T * D * 4);
  for (let t = 0; t < T; t++) {
    for (let d = 0; d < D; d++) {
      buf.writeFloatLE(data[t][d], (t * D + d) * 4);
    }
  }
  return buf;
}

/**
 * Encode int32 array to bytes.
 */
export function encodeInt32(data: number[]): Buffer {
  const buf = Buffer.alloc(data.length * 4);
  for (let i = 0; i < data.length; i++) {
    buf.writeInt32LE(data[i], i * 4);
  }
  return buf;
}

/**
 * Encode int64 array to bytes.
 */
export function encodeInt64(data: bigint[]): Buffer {
  const buf = Buffer.alloc(data.length * 8);
  for (let i = 0; i < data.length; i++) {
    buf.writeBigInt64LE(data[i], i * 8);
  }
  return buf;
}

/**
 * Encode bool array to bytes.
 */
export function encodeBool(data: boolean[]): Buffer {
  const buf = Buffer.alloc(data.length);
  for (let i = 0; i < data.length; i++) {
    buf[i] = data[i] ? 1 : 0;
  }
  return buf;
}

// ============================================================
// Decoding helpers
// ============================================================

/**
 * Decode float32 array from bytes.
 */
export function decodeFloat32(data: Buffer): number[] {
  const n = data.length / 4;
  const result = new Array(n);
  for (let i = 0; i < n; i++) {
    result[i] = data.readFloatLE(i * 4);
  }
  return result;
}

/**
 * Decode 2D float32 array from bytes.
 */
export function decodeFloat32_2D(data: Buffer, D: number): number[][] {
  const total = data.length / 4;
  const T = total / D;
  const result = new Array(T);
  for (let t = 0; t < T; t++) {
    result[t] = new Array(D);
    for (let d = 0; d < D; d++) {
      result[t][d] = data.readFloatLE((t * D + d) * 4);
    }
  }
  return result;
}

/**
 * Decode int32 array from bytes.
 */
export function decodeInt32(data: Buffer): number[] {
  const n = data.length / 4;
  const result = new Array(n);
  for (let i = 0; i < n; i++) {
    result[i] = data.readInt32LE(i * 4);
  }
  return result;
}

/**
 * Decode int64 array from bytes.
 */
export function decodeInt64(data: Buffer): bigint[] {
  const n = data.length / 8;
  const result = new Array(n);
  for (let i = 0; i < n; i++) {
    result[i] = data.readBigInt64LE(i * 8);
  }
  return result;
}

/**
 * Decode bool array from bytes.
 */
export function decodeBool(data: Buffer): boolean[] {
  const result = new Array(data.length);
  for (let i = 0; i < data.length; i++) {
    result[i] = data[i] !== 0;
  }
  return result;
}

// ============================================================
// Path helpers (Shard v2.1)
// ============================================================

/** Canonical path separator for hierarchical names. */
export const PATH_SEPARATOR = '/';

/**
 * Split a hierarchical name into path components.
 */
export function splitPath(name: string): string[] {
  return name.split(PATH_SEPARATOR).filter(p => p !== '');
}

/**
 * Join path components into a hierarchical name.
 */
export function joinPath(...parts: string[]): string {
  return parts.join(PATH_SEPARATOR);
}

/**
 * Check if name starts with the given prefix.
 */
export function pathPrefix(name: string, prefix: string): boolean {
  if (prefix === '') return true;
  if (!prefix.endsWith(PATH_SEPARATOR)) prefix += PATH_SEPARATOR;
  return name.startsWith(prefix) || name === prefix.slice(0, -1);
}

/**
 * Get parent path (everything before last "/").
 */
export function pathParent(name: string): string {
  const idx = name.lastIndexOf(PATH_SEPARATOR);
  if (idx < 0) return '';
  return name.substring(0, idx);
}

/**
 * Get base name (everything after last "/").
 */
export function pathBase(name: string): string {
  const idx = name.lastIndexOf(PATH_SEPARATOR);
  if (idx < 0) return name;
  return name.substring(idx + 1);
}
