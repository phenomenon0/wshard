/**
 * W-SHARD compression support.
 *
 * Supports zstd and LZ4 compression matching Go/Python implementations.
 * - Zstd: Uses @bokuweb/zstd-wasm for true zstd compression/decompression
 * - LZ4: Native JavaScript implementation matching Go's block format
 * - Deflate fallback: Uses fflate when zstd is unavailable
 */

import * as fflate from 'fflate';
import type { CompressionType, CompressionLevel } from './types.js';

// Zstd WASM module state
// Using @bokuweb/zstd-wasm which provides full compress + decompress support
let zstdModule: {
  compress: (data: Uint8Array, level?: number) => Uint8Array;
  decompress: (data: Uint8Array) => Uint8Array;
} | null = null;
let zstdInitialized = false;
let zstdInitPromise: Promise<void> | null = null;

/**
 * Initialize the Zstd WASM module.
 * Call this before using Zstd compression for best performance.
 * If not called, initialization happens automatically on first use.
 */
export async function initZstd(): Promise<void> {
  if (zstdInitialized) return;
  
  if (zstdInitPromise) {
    return zstdInitPromise;
  }
  
  zstdInitPromise = (async () => {
    try {
      const zstd = await import('@bokuweb/zstd-wasm');
      await zstd.init();
      zstdModule = {
        compress: zstd.compress,
        decompress: zstd.decompress,
      };
      zstdInitialized = true;
    } catch (e) {
      // zstd-wasm not available, will fall back to deflate
      console.warn('@bokuweb/zstd-wasm not available, falling back to deflate for zstd mode:', e);
      zstdInitialized = true;
    }
  })();
  
  return zstdInitPromise;
}

/**
 * Check if true Zstd support is available.
 */
export function hasZstdSupport(): boolean {
  return zstdModule !== null;
}

/**
 * Compressor handles block compression.
 */
export class Compressor {
  readonly type: CompressionType;
  readonly level: CompressionLevel;

  constructor(type: CompressionType, level: CompressionLevel = 'default') {
    this.type = type;
    this.level = level;
  }

  /**
   * Compress data synchronously.
   * Note: For zstd, call initZstd() first for best performance.
   */
  compress(data: Buffer): Buffer {
    if (this.type === 'none' || data.length === 0) {
      return data;
    }

    switch (this.type) {
      case 'zstd':
        return this.compressZstd(data);
      case 'lz4':
        return this.compressLz4(data);
      default:
        return data;
    }
  }

  /**
   * Compress data asynchronously (recommended for zstd).
   */
  async compressAsync(data: Buffer): Promise<Buffer> {
    if (this.type === 'none' || data.length === 0) {
      return data;
    }

    if (this.type === 'zstd' && !zstdInitialized) {
      await initZstd();
    }

    return this.compress(data);
  }

  /**
   * Decompress data synchronously.
   */
  decompress(data: Buffer, origSize: number): Buffer {
    if (this.type === 'none' || data.length === 0) {
      return data;
    }

    switch (this.type) {
      case 'zstd':
        return this.decompressZstd(data, origSize);
      case 'lz4':
        return this.decompressLz4(data, origSize);
      default:
        return data;
    }
  }

  /**
   * Decompress data asynchronously (recommended for zstd).
   */
  async decompressAsync(data: Buffer, origSize: number): Promise<Buffer> {
    if (this.type === 'none' || data.length === 0) {
      return data;
    }

    if (this.type === 'zstd' && !zstdInitialized) {
      await initZstd();
    }

    return this.decompress(data, origSize);
  }

  private compressZstd(data: Buffer): Buffer {
    // Require true zstd — no silent deflate fallback on write path
    if (zstdModule) {
      const level = this.getZstdLevel();
      const compressed = zstdModule.compress(new Uint8Array(data), level);
      return Buffer.from(compressed);
    }

    throw new Error(
      'Zstd compression requested but @bokuweb/zstd-wasm not initialized. ' +
      'Call initZstd() before writing zstd-compressed data.'
    );
  }

  private decompressZstd(data: Buffer, origSize: number): Buffer {
    // Try true zstd first via @bokuweb/zstd-wasm
    if (zstdModule) {
      try {
        const decompressed = zstdModule.decompress(new Uint8Array(data));
        return Buffer.from(decompressed);
      } catch (e) {
        // Data might be deflate-compressed (fallback mode), try that
      }
    }

    // Fallback to deflate decompression
    try {
      const decompressed = fflate.inflateSync(new Uint8Array(data));
      return Buffer.from(decompressed);
    } catch (e) {
      throw new Error(`Failed to decompress zstd data: ${e}`);
    }
  }

  private compressLz4(data: Buffer): Buffer {
    // LZ4 block compression
    // Use simple implementation matching Go's block format
    return lz4CompressBlock(data);
  }

  private decompressLz4(data: Buffer, origSize: number): Buffer {
    // LZ4 block decompression
    return lz4DecompressBlock(data, origSize);
  }

  private getZstdLevel(): number {
    switch (this.level) {
      case 'fastest':
        return 1;
      case 'best':
        return 19; // Zstd max level
      default:
        return 3; // Default zstd level
    }
  }

  private getDeflateLevel(): 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 {
    switch (this.level) {
      case 'fastest':
        return 1;
      case 'best':
        return 9;
      default:
        return 6;
    }
  }

  /**
   * Close and release resources.
   */
  close(): void {
    // No resources to release for sync compression
  }
}

/**
 * Standalone block compression.
 */
export function compressBlock(
  data: Buffer,
  type: CompressionType,
  level: CompressionLevel = 'default'
): Buffer {
  const c = new Compressor(type, level);
  try {
    return c.compress(data);
  } finally {
    c.close();
  }
}

/**
 * Standalone block compression (async, recommended for zstd).
 */
export async function compressBlockAsync(
  data: Buffer,
  type: CompressionType,
  level: CompressionLevel = 'default'
): Promise<Buffer> {
  const c = new Compressor(type, level);
  try {
    return await c.compressAsync(data);
  } finally {
    c.close();
  }
}

/**
 * Standalone block decompression.
 */
export function decompressBlock(
  data: Buffer,
  origSize: number,
  type: CompressionType
): Buffer {
  const c = new Compressor(type, 'default');
  try {
    return c.decompress(data, origSize);
  } finally {
    c.close();
  }
}

/**
 * Standalone block decompression (async, recommended for zstd).
 */
export async function decompressBlockAsync(
  data: Buffer,
  origSize: number,
  type: CompressionType
): Promise<Buffer> {
  const c = new Compressor(type, 'default');
  try {
    return await c.decompressAsync(data, origSize);
  } finally {
    c.close();
  }
}

/**
 * Check if block should be compressed.
 */
export function shouldCompress(blockName: string, data: Buffer): boolean {
  // Skip very small blocks (overhead not worth it)
  if (data.length < 64) {
    return false;
  }
  return true;
}

// ============================================================
// LZ4 Block Format Implementation
// ============================================================

// LZ4 constants
const LZ4_MIN_MATCH = 4;
const LZ4_MAX_MATCH_DISTANCE = 65535;
const LZ4_HASH_LOG = 12;
const LZ4_HASH_SIZE = 1 << LZ4_HASH_LOG;

/**
 * LZ4 block compression.
 * Matches Go's pierrec/lz4/v4 block format.
 */
function lz4CompressBlock(src: Buffer): Buffer {
  if (src.length === 0) {
    return Buffer.alloc(0);
  }

  // Quick incompressibility check for very small inputs
  if (src.length < LZ4_MIN_MATCH) {
    return lz4StoreLiterals(src);
  }

  const dst: number[] = [];
  const hashTable = new Int32Array(LZ4_HASH_SIZE).fill(-1);

  let srcPos = 0;
  let anchor = 0;

  // Main compression loop
  while (srcPos < src.length - LZ4_MIN_MATCH) {
    // Hash current position
    const h = lz4Hash(src, srcPos);
    const ref = hashTable[h];
    hashTable[h] = srcPos;

    // Check for match
    if (
      ref >= 0 &&
      srcPos - ref <= LZ4_MAX_MATCH_DISTANCE &&
      src.readUInt32LE(ref) === src.readUInt32LE(srcPos)
    ) {
      // Found a match - emit literals first
      const literalLen = srcPos - anchor;

      // Find match length
      let matchLen = LZ4_MIN_MATCH;
      while (
        srcPos + matchLen < src.length &&
        src[ref + matchLen] === src[srcPos + matchLen]
      ) {
        matchLen++;
      }

      // Emit token
      const token =
        (Math.min(literalLen, 15) << 4) | Math.min(matchLen - LZ4_MIN_MATCH, 15);
      dst.push(token);

      // Emit literal length extension
      if (literalLen >= 15) {
        let remaining = literalLen - 15;
        while (remaining >= 255) {
          dst.push(255);
          remaining -= 255;
        }
        dst.push(remaining);
      }

      // Emit literals
      for (let i = anchor; i < srcPos; i++) {
        dst.push(src[i]);
      }

      // Emit offset (little-endian)
      const offset = srcPos - ref;
      dst.push(offset & 0xff);
      dst.push((offset >> 8) & 0xff);

      // Emit match length extension
      if (matchLen - LZ4_MIN_MATCH >= 15) {
        let remaining = matchLen - LZ4_MIN_MATCH - 15;
        while (remaining >= 255) {
          dst.push(255);
          remaining -= 255;
        }
        dst.push(remaining);
      }

      // Advance
      srcPos += matchLen;
      anchor = srcPos;
    } else {
      srcPos++;
    }
  }

  // Emit remaining literals
  const literalLen = src.length - anchor;
  if (literalLen > 0) {
    const token = Math.min(literalLen, 15) << 4;
    dst.push(token);

    if (literalLen >= 15) {
      let remaining = literalLen - 15;
      while (remaining >= 255) {
        dst.push(255);
        remaining -= 255;
      }
      dst.push(remaining);
    }

    for (let i = anchor; i < src.length; i++) {
      dst.push(src[i]);
    }
  }

  // Check if compression helped
  if (dst.length >= src.length) {
    return src; // Return original if no savings
  }

  return Buffer.from(dst);
}

/**
 * Store literals only (no compression).
 */
function lz4StoreLiterals(src: Buffer): Buffer {
  const dst: number[] = [];
  const len = src.length;

  // Token with literal count
  const token = Math.min(len, 15) << 4;
  dst.push(token);

  // Length extension
  if (len >= 15) {
    let remaining = len - 15;
    while (remaining >= 255) {
      dst.push(255);
      remaining -= 255;
    }
    dst.push(remaining);
  }

  // Copy literals
  for (let i = 0; i < len; i++) {
    dst.push(src[i]);
  }

  return Buffer.from(dst);
}

/**
 * LZ4 hash function.
 */
function lz4Hash(data: Buffer, pos: number): number {
  const v = data.readUInt32LE(pos);
  return ((v * 2654435761) >>> (32 - LZ4_HASH_LOG)) & (LZ4_HASH_SIZE - 1);
}

/**
 * LZ4 block decompression.
 */
const LZ4_MAX_DECOMPRESS_SIZE = 256 * 1024 * 1024; // 256 MB

function lz4DecompressBlock(src: Buffer, dstSize: number): Buffer {
  if (src.length === 0) {
    return Buffer.alloc(0);
  }

  if (dstSize <= 0 || dstSize > LZ4_MAX_DECOMPRESS_SIZE) {
    throw new Error(
      `LZ4 decompress: dstSize ${dstSize} out of bounds (0, ${LZ4_MAX_DECOMPRESS_SIZE}]`
    );
  }

  const dst = Buffer.alloc(dstSize);
  let srcPos = 0;
  let dstPos = 0;

  while (srcPos < src.length) {
    // Read token
    const token = src[srcPos++];
    let literalLen = token >> 4;
    const matchLen = token & 0x0f;

    // Read literal length extension
    if (literalLen === 15) {
      let s: number;
      do {
        s = src[srcPos++];
        literalLen += s;
      } while (s === 255);
    }

    // Copy literals
    if (literalLen > 0) {
      src.copy(dst, dstPos, srcPos, srcPos + literalLen);
      srcPos += literalLen;
      dstPos += literalLen;
    }

    // Check if we're done (last sequence has no match)
    if (srcPos >= src.length) {
      break;
    }

    // Read offset
    const offset = src[srcPos] | (src[srcPos + 1] << 8);
    srcPos += 2;

    if (offset === 0) {
      throw new Error('LZ4: invalid offset 0');
    }

    // Read match length extension
    let actualMatchLen = matchLen + LZ4_MIN_MATCH;
    if (matchLen === 15) {
      let s: number;
      do {
        s = src[srcPos++];
        actualMatchLen += s;
      } while (s === 255);
    }

    // Copy match
    const matchPos = dstPos - offset;
    if (matchPos < 0) {
      throw new Error('LZ4: invalid match offset');
    }

    // Handle overlapping copies
    for (let i = 0; i < actualMatchLen; i++) {
      dst[dstPos++] = dst[matchPos + i];
    }
  }

  return dst.subarray(0, dstPos);
}
