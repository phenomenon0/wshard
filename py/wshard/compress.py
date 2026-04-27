"""
Compression support for W-SHARD blocks.

Supports:
- None: No compression
- Zstd: Zstandard compression (default)
- LZ4: LZ4 compression (fast)
"""

from enum import Enum
from typing import Optional, Tuple


class CompressionType(str, Enum):
    """Compression algorithm types."""

    NONE = "none"
    ZSTD = "zstd"
    LZ4 = "lz4"


class CompressionLevel(int, Enum):
    """Compression quality levels."""

    FASTEST = 1
    DEFAULT = 3
    BEST = 9


# Lazy imports for compression libraries
_zstd = None
_lz4 = None


def _get_zstd():
    """Lazy load zstandard library."""
    global _zstd
    if _zstd is None:
        try:
            import zstandard as zstd

            _zstd = zstd
        except ImportError:
            raise ImportError(
                "zstandard library required for zstd compression. "
                "Install with: pip install zstandard"
            )
    return _zstd


def _get_lz4():
    """Lazy load lz4 library."""
    global _lz4
    if _lz4 is None:
        try:
            import lz4.block

            _lz4 = lz4.block
        except ImportError:
            raise ImportError(
                "lz4 library required for lz4 compression. "
                "Install with: pip install lz4"
            )
    return _lz4


def compression_byte(ct: CompressionType) -> int:
    """Get the byte value for header compression field."""
    if ct == CompressionType.ZSTD:
        return 1
    elif ct == CompressionType.LZ4:
        return 2
    return 0


def compression_from_byte(b: int) -> CompressionType:
    """Get compression type from header byte."""
    if b == 1:
        return CompressionType.ZSTD
    elif b == 2:
        return CompressionType.LZ4
    return CompressionType.NONE


# Block flag bits (matching Go's EntryFlag* constants in shard_format.go)
BLOCK_FLAG_COMPRESSED = 0x0001
BLOCK_FLAG_ZSTD = 0x0002
BLOCK_FLAG_LZ4 = 0x0004

# Safety limit for LZ4 decompression output
LZ4_MAX_DECOMPRESS_SIZE = 256 * 1024 * 1024  # 256 MB


def compress(
    data: bytes,
    compression_type: CompressionType = CompressionType.NONE,
    level: CompressionLevel = CompressionLevel.DEFAULT,
) -> bytes:
    """
    Compress data with the specified algorithm.

    Args:
        data: Raw bytes to compress
        compression_type: Algorithm to use
        level: Compression level

    Returns:
        Compressed bytes (or original if compression not helpful)
    """
    if compression_type == CompressionType.NONE or len(data) == 0:
        return data

    if compression_type == CompressionType.ZSTD:
        zstd = _get_zstd()
        # Map level to zstd level (1-22)
        zstd_level = {
            CompressionLevel.FASTEST: 1,
            CompressionLevel.DEFAULT: 3,
            CompressionLevel.BEST: 19,
        }.get(level, 3)

        cctx = zstd.ZstdCompressor(level=zstd_level)
        return cctx.compress(data)

    elif compression_type == CompressionType.LZ4:
        lz4 = _get_lz4()
        # Map level to lz4 mode
        mode = {
            CompressionLevel.FASTEST: "default",
            CompressionLevel.DEFAULT: "high_compression",
            CompressionLevel.BEST: "high_compression",
        }.get(level, "high_compression")

        # Use store_size=False to match Go's raw block format
        if mode == "high_compression":
            return lz4.compress(data, mode=mode, compression=9, store_size=False)
        return lz4.compress(data, store_size=False)

    return data


def decompress(data: bytes, orig_size: int, compression_type: CompressionType) -> bytes:
    """
    Decompress data with the specified algorithm.

    Args:
        data: Compressed bytes
        orig_size: Original uncompressed size
        compression_type: Algorithm used

    Returns:
        Decompressed bytes
    """
    if compression_type == CompressionType.NONE:
        return data

    if compression_type == CompressionType.ZSTD:
        zstd = _get_zstd()
        dctx = zstd.ZstdDecompressor()
        return dctx.decompress(data, max_output_size=orig_size)

    elif compression_type == CompressionType.LZ4:
        if orig_size <= 0 or orig_size > LZ4_MAX_DECOMPRESS_SIZE:
            raise ValueError(
                f"LZ4 decompress: orig_size {orig_size} out of bounds "
                f"(0, {LZ4_MAX_DECOMPRESS_SIZE}]"
            )
        lz4 = _get_lz4()
        # Use uncompressed_size to match Go's raw block format (no size header)
        return lz4.decompress(data, uncompressed_size=orig_size)

    return data


def should_compress(block_name: str, data: bytes) -> bool:
    """
    Determine if a block should be compressed.

    Small blocks or already-compressed data may not benefit.

    Args:
        block_name: Name of the block
        data: Block data

    Returns:
        True if compression is recommended
    """
    # Skip very small blocks (overhead not worth it)
    if len(data) < 64:
        return False

    # Metadata blocks are already well-structured
    # but still benefit from compression

    return True


class Compressor:
    """
    Reusable compressor/decompressor.

    Maintains state for efficient repeated compression/decompression.
    """

    def __init__(
        self,
        compression_type: CompressionType = CompressionType.NONE,
        level: CompressionLevel = CompressionLevel.DEFAULT,
    ):
        self.type = compression_type
        self.level = level
        self._cctx = None
        self._dctx = None

    def compress(self, data: bytes) -> bytes:
        """Compress data."""
        if self.type == CompressionType.NONE or len(data) == 0:
            return data

        if self.type == CompressionType.ZSTD:
            zstd = _get_zstd()
            if self._cctx is None:
                zstd_level = {
                    CompressionLevel.FASTEST: 1,
                    CompressionLevel.DEFAULT: 3,
                    CompressionLevel.BEST: 19,
                }.get(self.level, 3)
                self._cctx = zstd.ZstdCompressor(level=zstd_level)
            return self._cctx.compress(data)

        elif self.type == CompressionType.LZ4:
            lz4 = _get_lz4()
            mode = (
                "high_compression"
                if self.level >= CompressionLevel.DEFAULT
                else "default"
            )
            # Use store_size=False to match Go's raw block format
            if mode == "high_compression":
                return lz4.compress(data, mode=mode, compression=9, store_size=False)
            return lz4.compress(data, store_size=False)

        return data

    def decompress(self, data: bytes, orig_size: int) -> bytes:
        """Decompress data."""
        if self.type == CompressionType.NONE:
            return data

        if self.type == CompressionType.ZSTD:
            zstd = _get_zstd()
            if self._dctx is None:
                self._dctx = zstd.ZstdDecompressor()
            return self._dctx.decompress(data, max_output_size=orig_size)

        elif self.type == CompressionType.LZ4:
            if orig_size <= 0 or orig_size > LZ4_MAX_DECOMPRESS_SIZE:
                raise ValueError(
                    f"LZ4 decompress: orig_size {orig_size} out of bounds "
                    f"(0, {LZ4_MAX_DECOMPRESS_SIZE}]"
                )
            lz4 = _get_lz4()
            return lz4.decompress(data, uncompressed_size=orig_size)

        return data
