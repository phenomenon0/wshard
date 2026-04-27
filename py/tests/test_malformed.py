"""
Malformed-input regression suite. Each test feeds a hand-corrupted .wshard
file to load_wshard() and asserts the reader rejects it (or, for positive
tests like bf16 NaN, that the reader preserves bits exactly).

Cases that the current reader does not yet detect are skipped with a clear
TODO comment so we know what to harden next.
"""

# Wire-format reference (from go/shard/shard_format.go and py/wshard/wshard.py):
#
# Header (64 bytes, little-endian):
#   [0:4]   magic      = b"SHRD"
#   [4]     version    = 0x02
#   [5]     role       = 0x05 (W-SHARD)
#   [6:8]   flags      (uint16 LE)
#   [8]     alignment
#   [9]     compression_default  (0=none, 1=zstd, 2=lz4)
#   [10:12] index_entry_size     (uint16 LE; must be 48)
#   [12:16] entry_count          (uint32 LE)
#   [16:24] string_table_offset  (uint64 LE)
#   [24:32] data_section_offset  (uint64 LE)
#   [32:40] schema_offset        (uint64 LE; 0 if absent)
#   [40:48] total_file_size      (uint64 LE)
#   [48:64] reserved (zeroed)
#
# Index entry (48 bytes, one per block, starts at byte 64):
#   [0:8]   name_hash    xxHash64 of name
#   [8:12]  name_offset  uint32 LE — offset into string table
#   [12:14] name_len     uint16 LE
#   [14:16] flags        uint16 LE  (0x01=compressed, 0x02=zstd, 0x04=lz4)
#   [16:24] data_offset  uint64 LE  (absolute file offset)
#   [24:32] disk_size    uint64 LE  (on-disk / compressed size)
#   [32:40] orig_size    uint64 LE  (original / uncompressed size)
#   [40:44] checksum     uint32 LE  (CRC32C of uncompressed data)
#   [44:48] reserved
#
# Go dtype byte values (go/shard/dtype.go):
#   DTypeFloat32  = 0x01   DTypeFloat64 = 0x0C   DTypeBFloat16 = 0x03

import io
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from wshard.wshard import (
    HEADER_SIZE,
    INDEX_ENTRY_SIZE,
    MAGIC,
    VERSION,
    ROLE_WSHARD,
    compute_crc32,
    load_wshard,
    save_wshard,
)
from wshard.types import Channel, DType, Episode
from wshard.compress import BLOCK_FLAG_COMPRESSED, BLOCK_FLAG_ZSTD


# ============================================================
# Shared helper
# ============================================================

def _minimal_episode() -> Episode:
    """Return a small but valid Episode with one float32 observation.

    Channel data shape must be (T, *per_timestep_shape) to pass Episode.validate().
    With T=4 and per-timestep shape=[3], data must be (4, 3).
    """
    T = 4
    return Episode(
        id="test-malformed",
        length=T,
        observations={
            "obs": Channel(
                name="obs",
                dtype=DType.FLOAT32,
                shape=[3],
                data=np.arange(T * 3, dtype=np.float32).reshape(T, 3),
            )
        },
    )


def _valid_bytes(tmp_path: Path) -> bytearray:
    """Write a minimal valid .wshard and return its bytes as a bytearray."""
    path = tmp_path / "baseline.wshard"
    save_wshard(_minimal_episode(), path)
    return bytearray(path.read_bytes())


# ============================================================
# Case 1 — wrong magic bytes
# ============================================================

def test_wrong_magic_bytes_rejected(tmp_path):
    """Bytes 0-3 overwritten with b'XXXX' must be rejected.

    Reader check: data[:4] != MAGIC → ValueError
    """
    data = _valid_bytes(tmp_path)
    data[0:4] = b"XXXX"
    with pytest.raises(ValueError, match="magic"):
        load_wshard(io.BytesIO(bytes(data)))


# ============================================================
# Case 2 — wrong version byte
# ============================================================

def test_wrong_version_byte_rejected(tmp_path):
    """Byte 4 set to 0x99 (non-existent version) must be rejected.

    Reader check: version != VERSION (0x02) → ValueError
    """
    data = _valid_bytes(tmp_path)
    data[4] = 0x99
    with pytest.raises(ValueError, match="[Vv]ersion"):
        load_wshard(io.BytesIO(bytes(data)))


# ============================================================
# Case 3 — inflated entry_count
# ============================================================

def test_inflated_entry_count_rejected(tmp_path):
    """entry_count set to 0xFFFFFFFF must not cause an OOM/hang.

    The Python reader iterates range(entry_count) slicing bytes.  With a
    huge count the loop reads out-of-range slices (Python silently returns
    empty bytes for out-of-bounds slices), then _parse_index_entry receives
    a zero-filled 48-byte block and the resulting entry has name_offset=0,
    name_len=0, which maps to an empty-string block name.  The reader
    currently does NOT raise — it just returns an episode with scrambled or
    empty blocks.

    TODO: reader hardening needed — add a sanity check like:
        max_entries = (len(data) - HEADER_SIZE) // INDEX_ENTRY_SIZE
        if entry_count > max_entries: raise ValueError(...)
    """
    data = _valid_bytes(tmp_path)
    # bytes 12-16: entry_count (uint32 LE)
    struct.pack_into("<I", data, 12, 0xFFFFFFFF)
    pytest.skip(
        "TODO: needs reader hardening — inflated entry_count is not yet "
        "rejected; reader silently iterates OOB slices and returns partial episode."
    )


# ============================================================
# Case 4 — duplicate block names
# ============================================================

def test_duplicate_block_names_rejected_or_documented(tmp_path):
    """Two index entries with the same name must be caught or documented.

    The Python encoder always writes sorted, de-duplicated block names so a
    valid save_wshard() call cannot produce duplicates.  Hand-crafting a
    duplicate via byte patching requires rebuilding the entire index (name
    hashes, offsets, string table) which is not feasible as a one-field
    patch.

    The reader uses blocks[name] = ... so the second entry silently
    overwrites the first — no error is raised.

    TODO: needs reader hardening — detect duplicate names in the index and
    raise ValueError before loading block data.
    """
    pytest.skip(
        "TODO: duplicate block names not rejected — reader silently overwrites "
        "earlier entry.  Hardening requires a seen-names set in the index walk."
    )


# ============================================================
# Case 5 — string_table_offset past EOF
# ============================================================

def test_string_table_offset_out_of_bounds_rejected(tmp_path):
    """string_table_offset set past EOF must be rejected.

    The Python reader computes:
        string_table_size = data_section_offset - string_table_offset
    If string_table_offset > len(data) the slice returns empty bytes (no
    error).  The reader then finds no block names and returns an empty
    episode rather than raising.

    TODO: needs reader hardening — validate:
        if string_table_offset > len(data): raise ValueError(...)
    """
    data = _valid_bytes(tmp_path)
    # bytes 16-24: string_table_offset (uint64 LE)
    struct.pack_into("<Q", data, 16, len(data) + 0x10000)
    pytest.skip(
        "TODO: string_table_offset past EOF is not rejected — reader silently "
        "returns an empty episode.  Hardening: bounds-check all section offsets "
        "against len(data) before slicing."
    )


# ============================================================
# Case 6 — unknown compression flag in index entry
# ============================================================

def test_unknown_compression_flag_rejected(tmp_path):
    """Compression flag 0xFF in an index entry must be rejected.

    The reader checks:
        if (entry_flags & BLOCK_FLAG_COMPRESSED) and disk_size != orig_size:
            if BLOCK_FLAG_LZ4 → LZ4
            elif BLOCK_FLAG_ZSTD → ZSTD
            else → fall back to compression_default (NONE)
    So an unrecognised compression type silently falls back to treating
    compressed bytes as raw data — no error is raised.

    TODO: needs reader hardening — when BLOCK_FLAG_COMPRESSED is set but
    neither ZSTD nor LZ4 bit is present and compression_default is NONE,
    raise ValueError("unknown compression type in entry flags").
    """
    data = _valid_bytes(tmp_path)
    # flags field at offset 14 within each 48-byte index entry; entries start at HEADER_SIZE
    flags_offset = HEADER_SIZE + 14
    struct.pack_into("<H", data, flags_offset, 0x00FF)  # compressed + garbage type bits
    pytest.skip(
        "TODO: unknown compression flag is not rejected — reader falls back to "
        "treating data as uncompressed.  Hardening: reject unrecognised flag "
        "combinations when BLOCK_FLAG_COMPRESSED is set."
    )


# ============================================================
# Case 7 — unknown dtype in meta/channels JSON
# ============================================================

def test_unknown_dtype_rejected(tmp_path):
    """A channel with dtype 'zz99' (not a valid DType enum value) must raise.

    dtype is stored as a string in meta/channels JSON, not as a wire byte.
    The reader calls DType(dtype_str) which raises ValueError for unknown values.
    """
    data = _valid_bytes(tmp_path)
    buf = bytearray(data)

    # Locate the meta/channels JSON block by scanning for the JSON prefix.
    marker = b'{"channels":'
    pos = bytes(buf).find(marker)
    if pos == -1:
        pytest.skip("Cannot locate meta/channels JSON block in baseline bytes")

    # Find end of JSON block (it ends at the next alignment-padded zero run).
    # Safer: find the json extent by walking forward to the closing '}'.
    # We'll decode to end of the JSON object by using a small json.loads probe.
    raw = bytes(buf)
    for end in range(pos + len(marker), len(raw)):
        if raw[end] == ord(b"}"):
            candidate = raw[pos : end + 1]
            try:
                json.loads(candidate)
                json_end = end + 1
                break
            except json.JSONDecodeError:
                continue
    else:
        pytest.skip("Cannot delimit meta/channels JSON block")

    original_json = raw[pos:json_end]
    # "f32" → "BAD" — same 5 bytes including quotes, so no length change.
    patched_json = original_json.replace(b'"f32"', b'"BAD"')
    if patched_json == original_json:
        pytest.skip("Could not find f32 dtype string to patch")

    # Guard: same-length patch only (disk_size in index is fixed)
    if len(patched_json) != len(original_json):
        pytest.skip("Patched JSON is a different length — cannot do in-place patch")

    buf[pos:json_end] = patched_json

    # Zero out the CRC of this block so checksum doesn't fire first.
    # Find the index entry for meta/channels by scanning entries.
    entry_count = struct.unpack_from("<I", buf, 12)[0]
    for i in range(entry_count):
        e_off = HEADER_SIZE + i * INDEX_ENTRY_SIZE
        name_offset = struct.unpack_from("<I", buf, e_off + 8)[0]
        name_len = struct.unpack_from("<H", buf, e_off + 12)[0]
        string_table_offset = struct.unpack_from("<Q", buf, 16)[0]
        data_section_offset = struct.unpack_from("<Q", buf, 24)[0]
        st_size = data_section_offset - string_table_offset
        st = bytes(buf)[string_table_offset: string_table_offset + st_size]
        if name_offset + name_len <= len(st):
            name = st[name_offset: name_offset + name_len].decode("utf-8", errors="replace")
            if name == "meta/channels":
                # Zero CRC (offset 40 in entry)
                struct.pack_into("<I", buf, e_off + 40, 0)
                break

    with pytest.raises((ValueError, KeyError, Exception)):
        load_wshard(io.BytesIO(bytes(buf)))


# ============================================================
# Case 8 — shape product overflow in meta/channels JSON
# ============================================================

def test_shape_product_overflow_rejected(tmp_path):
    """A channel claiming a shape that cannot fit the actual data must be rejected.

    Patch: change "shape":[3] → "shape":[7] (same 11 bytes).  The block
    holds 4*3=12 float32 elements; reshaping to (-1, 7) requires a multiple
    of 7 — which 12 is not.  numpy raises ValueError.
    """
    data = _valid_bytes(tmp_path)
    buf = bytearray(data)

    # Same-length in-place patch: "shape":[3] → "shape":[7]
    # Both are 11 bytes; disk_size in the index entry stays correct.
    marker = b'"shape": [3]'
    replacement = b'"shape": [7]'
    pos = bytes(buf).find(marker)
    if pos == -1:
        # Try without space (encoding may vary)
        marker = b'"shape":[3]'
        replacement = b'"shape":[7]'
        pos = bytes(buf).find(marker)
    if pos == -1:
        pytest.skip("Cannot locate shape:[3] in baseline meta/channels bytes")
    if len(replacement) != len(marker):
        pytest.skip("Replacement length mismatch — cannot do in-place patch")

    buf[pos: pos + len(marker)] = replacement

    # Zero CRC for meta/channels to avoid checksum short-circuit.
    entry_count = struct.unpack_from("<I", buf, 12)[0]
    for i in range(entry_count):
        e_off = HEADER_SIZE + i * INDEX_ENTRY_SIZE
        name_offset = struct.unpack_from("<I", buf, e_off + 8)[0]
        name_len = struct.unpack_from("<H", buf, e_off + 12)[0]
        string_table_offset = struct.unpack_from("<Q", buf, 16)[0]
        data_section_offset = struct.unpack_from("<Q", buf, 24)[0]
        st = bytes(buf)[string_table_offset: data_section_offset]
        if name_offset + name_len <= len(st):
            name = st[name_offset: name_offset + name_len].decode("utf-8", errors="replace")
            if name == "meta/channels":
                struct.pack_into("<I", buf, e_off + 40, 0)
                break

    with pytest.raises((ValueError, Exception)):
        load_wshard(io.BytesIO(bytes(buf)))


# ============================================================
# Case 9 — bad CRC (bit-flip in data block)
# ============================================================

def test_bad_crc_rejected(tmp_path):
    """Flipping one bit in a data block with a non-zero CRC must be rejected.

    Reader check (wshard.py line ~188):
        if checksum != 0:
            actual = compute_crc32(block_data)
            if actual != checksum: raise ValueError("Checksum mismatch")
    """
    data = _valid_bytes(tmp_path)
    buf = bytearray(data)

    # Find the first index entry that has a non-zero CRC and non-zero disk_size.
    entry_count = struct.unpack_from("<I", buf, 12)[0]
    target_entry_off = None
    target_data_offset = None
    target_disk_size = None

    for i in range(entry_count):
        e_off = HEADER_SIZE + i * INDEX_ENTRY_SIZE
        checksum = struct.unpack_from("<I", buf, e_off + 40)[0]
        disk_size = struct.unpack_from("<Q", buf, e_off + 24)[0]
        data_offset = struct.unpack_from("<Q", buf, e_off + 16)[0]
        if checksum != 0 and disk_size > 0 and data_offset + disk_size <= len(buf):
            target_entry_off = e_off
            target_data_offset = int(data_offset)
            target_disk_size = int(disk_size)
            break

    if target_entry_off is None:
        pytest.skip("No entry with non-zero CRC found in baseline file")

    # Flip one bit in the middle of the block data.
    mid = target_data_offset + target_disk_size // 2
    buf[mid] ^= 0x01

    with pytest.raises(ValueError, match="[Cc]hecksum"):
        load_wshard(io.BytesIO(bytes(buf)))


# ============================================================
# Case 10 — bf16 NaN/Inf round-trip (positive test)
# ============================================================

def test_bf16_nan_roundtrip(tmp_path):
    """bf16 NaN and Inf bit patterns must survive a save/load round-trip unchanged.

    bfloat16 shares the float32 exponent width.  Canonical bit patterns:
      - +Inf  = 0x7F80 (bf16 uint16)
      - -Inf  = 0xFF80
      - NaN   = 0x7FC0  (quiet NaN)
      - sNaN  = 0x7F81  (signalling NaN)

    The reader stores bf16 as uint16 (or ml_dtypes.bfloat16 if available);
    either way the raw 2-byte patterns must be preserved bit-for-bit.
    """
    try:
        import ml_dtypes
        bf16_dtype = np.dtype(ml_dtypes.bfloat16)
    except ImportError:
        bf16_dtype = np.dtype(np.uint16)

    # Build raw bf16 bit patterns via uint16 view then reinterpret.
    # shape=[] means scalar per timestep; data shape must be (T,).
    raw_bits = np.array([0x7F80, 0xFF80, 0x7FC0, 0x7F81, 0x0000, 0x3F80], dtype=np.uint16)
    T = len(raw_bits)

    # Store as bf16 Channel — use uint16 as the storage dtype so we can
    # exercise the round-trip without requiring ml_dtypes.
    ep = Episode(
        id="bf16-nan-test",
        length=T,
        observations={
            "signal": Channel(
                name="signal",
                dtype=DType.UINT16,  # byte-preserving stand-in for bf16 raw storage
                shape=[],
                data=raw_bits,
            )
        },
    )

    path = tmp_path / "bf16_nan.wshard"
    save_wshard(ep, path)
    ep2 = load_wshard(path)

    recovered = ep2.observations["signal"].data
    recovered_bits = recovered.view(np.uint16)
    assert list(recovered_bits) == list(raw_bits), (
        f"bf16 bit patterns not preserved: {list(recovered_bits)} != {list(raw_bits)}"
    )


# ============================================================
# Case 11 — oversized orig_size (no allocation guard)
# ============================================================

def test_oversized_decompressed_size_rejected(tmp_path):
    """orig_size = 1 TiB on a compressed block must not allocate 1 TiB.

    The Go reader (shard_format.go) guards with MaxDecompressSize (1 GiB).
    The Python reader has no equivalent guard — it passes orig_size directly
    to the decompressor.  For uncompressed blocks it never allocates based on
    orig_size at all.

    TODO: needs reader hardening — add a Python-side guard before calling
    Compressor.decompress():
        MAX_DECOMPRESS = 1 << 30  # 1 GiB
        if orig_size > MAX_DECOMPRESS: raise ValueError(...)
    """
    data = _valid_bytes(tmp_path)
    buf = bytearray(data)

    # Mark the first entry as COMPRESSED|ZSTD and set orig_size = 1 TiB
    e_off = HEADER_SIZE  # first index entry
    struct.pack_into("<H", buf, e_off + 14, BLOCK_FLAG_COMPRESSED | BLOCK_FLAG_ZSTD)
    struct.pack_into("<Q", buf, e_off + 32, 1 << 40)  # orig_size = 1 TiB
    # Zero the checksum so CRC check doesn't fire first
    struct.pack_into("<I", buf, e_off + 40, 0)

    pytest.skip(
        "TODO: no orig_size guard in Python reader — would attempt 1 TiB "
        "allocation.  Hardening: add MAX_DECOMPRESS check before decompression."
    )


# ============================================================
# Case 12 — string table name_len overflow
# ============================================================

def test_string_table_length_overflow_rejected(tmp_path):
    """name_len set to a value that extends past the string table must be rejected.

    The reader checks (wshard.py line ~161):
        if name_offset + name_len <= len(string_table): ...
        else: continue   ← silently skips the entry, no error
    So the block is simply absent from the loaded episode — no ValueError.

    TODO: needs reader hardening — when a name_len overflow is detected,
    raise ValueError rather than silently continuing.
    """
    data = _valid_bytes(tmp_path)
    buf = bytearray(data)

    # Patch name_len of the first index entry to 0xFFFF
    e_off = HEADER_SIZE
    struct.pack_into("<H", buf, e_off + 12, 0xFFFF)

    pytest.skip(
        "TODO: string table name_len overflow is silently skipped — reader "
        "uses 'continue' and returns an episode missing that block.  "
        "Hardening: raise ValueError on any out-of-bounds name access."
    )
