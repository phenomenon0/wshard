# WShard Format — At a Glance

A quick reference for readers who need the layout without reading every byte of [DEEP_DIVE.md](DEEP_DIVE.md). All field offsets are confirmed against `go/shard/shard_format.go` and `py/wshard/wshard.py`.

---

## 1. File layout

```
┌─────────────────────────────────────────────┐
│  Header (64 bytes, fixed)                   │
├─────────────────────────────────────────────┤
│  Index  (N × 48 bytes)                      │
│  one entry per named block                  │
├─────────────────────────────────────────────┤
│  String table  (variable)                   │
│  UTF-8 block names, concatenated            │
├─────────────────────────────────────────────┤
│  Alignment padding (0x00 bytes)             │
├─────────────────────────────────────────────┤
│  Data blocks  (each 32-byte aligned)        │
│  independently addressable, independently   │
│  compressed                                 │
└─────────────────────────────────────────────┘
```

The header is always parseable with a single `read(64)` call. The index immediately follows at byte 64.

---

## 2. Header (64 bytes)

All multi-byte integers are little-endian. Source: `WriteShardHeader` / `ReadShardHeader` in `go/shard/shard_format.go`.

| Offset | Size | Field | Value / Notes |
|--------|------|-------|---------------|
| 0x00 | 4 | magic | `SHRD` (0x53 0x48 0x52 0x44) |
| 0x04 | 1 | version | `0x02` |
| 0x05 | 1 | role | `0x05` for WShard |
| 0x06 | 2 | flags | LE uint16; feature flag bits (see below) |
| 0x08 | 1 | alignment | 0, 16, 32, or 64 bytes |
| 0x09 | 1 | compression\_default | 0=none, 1=zstd, 2=lz4 |
| 0x0A | 2 | index\_entry\_size | LE uint16; must be 48 |
| 0x0C | 4 | entry\_count | LE uint32; number of blocks |
| 0x10 | 8 | string\_table\_offset | LE uint64; absolute byte offset |
| 0x18 | 8 | data\_section\_offset | LE uint64; absolute byte offset |
| 0x20 | 8 | schema\_offset | LE uint64; 0 if no schema |
| 0x28 | 8 | total\_file\_size | LE uint64; for validation |
| 0x30 | 16 | reserved | zeroed |

**Total: 64 bytes.** The header fits in one 512-byte disk sector with room to spare.

**Flag bits (offset 0x06):**

| Bit | Constant | Meaning |
|-----|----------|---------|
| 0x0010 | `ShardFlagHasSchema` | Schema section present |
| 0x0020 | `ShardFlagHasChecksums` | Per-entry checksums enabled |
| 0x0040 | `ShardFlagStreaming` | Written by streaming writer |
| 0x0080 | `ShardFlagHasContentTypes` | Content type hints present |

---

## 3. Index entries (48 bytes each)

The index starts at byte 64 and contains `entry_count` entries of exactly 48 bytes each. Confirmed against `WriteIndexEntry` / `ReadIndexEntry` in `go/shard/shard_format.go`.

| Offset | Size | Field | Notes |
|--------|------|-------|-------|
| 0x00 | 8 | name\_hash | xxHash64 of UTF-8 name (LE uint64) |
| 0x08 | 4 | name\_offset | Offset into string table (LE uint32) |
| 0x0C | 2 | name\_len | Length of name in bytes (LE uint16) |
| 0x0E | 2 | flags | Compression flags (LE uint16; see below) |
| 0x10 | 8 | data\_offset | Absolute byte offset in file (LE uint64) |
| 0x18 | 8 | disk\_size | Size on disk, compressed (LE uint64) |
| 0x20 | 8 | orig\_size | Uncompressed size (LE uint64) |
| 0x28 | 4 | checksum | CRC32C of uncompressed data (LE uint32) |
| 0x2C | 4 | reserved | Zeroed (includes content-type hint) |

**Total: 48 bytes per entry.**

**Entry flag bits (offset 0x0E):**

| Bit | Constant | Meaning |
|-----|----------|---------|
| 0x0001 | `EntryFlagCompressed` | Block is compressed |
| 0x0002 | `EntryFlagZstd` | Compression type: zstd |
| 0x0004 | `EntryFlagLZ4` | Compression type: lz4 |
| 0x0008 | `EntryFlagChunked` | Block is chunked |

When `EntryFlagCompressed` is set, check `EntryFlagZstd` and `EntryFlagLZ4` for the codec. If neither is set, fall back to `compression_default` from the header.

**Fast lookup:** a reader scanning for a named block hashes the target name with xxHash64 and compares against `name_hash` values — 8 bytes per entry, no string-table parse needed. For a 50-block file: 64 + 50×8 = 464 bytes read total.

---

## 4. String table

The string table occupies the bytes from `string_table_offset` to `data_section_offset`. It contains the full UTF-8 names of all blocks, concatenated without separators or length prefixes. Each index entry points into it via `name_offset` (from the start of the string table) and `name_len`.

No null terminators. No padding between names. The string table is only used when you need to resolve a full name; the index hash scan does not require it.

---

## 5. Block namespaces

These are advisory naming conventions, not format-level constraints. The Python reader uses them to route blocks into the correct `Episode` fields.

| Prefix | Role | Examples |
|--------|------|---------|
| `meta/` | JSON metadata blobs | `meta/wshard`, `meta/episode`, `meta/channels` |
| `signal/` | Ground truth observations | `signal/rgb`, `signal/joint_pos` |
| `action/` | Agent actions | `action/ctrl`, `action/gripper` |
| `time/` | Timestamps | `time/ticks`, `time/timestamps_ns` |
| `omen/` | Model predictions | `omen/joint_pos/dreamer` |
| `uncert/` | Uncertainty estimates | `uncert/joint_pos/dreamer/std` |
| `residual/` | Compressed residual encodings | `residual/joint_pos/sign2nddiff` |
| `reward` | Reward signal (no prefix) | `reward` |
| `done` | Termination flags | `done` |

You can use custom namespaces freely (`my_team/sensor_x`). Unrecognized prefixes are accessible via the low-level index API as raw byte blocks.

---

## 6. Data types

13 types, matching the union of numpy, PyTorch, and Go primitive types.

| WShard | Bytes | numpy equiv | Notes |
|--------|-------|-------------|-------|
| `f32` | 4 | float32 | Default for most signals |
| `f64` | 8 | float64 | High-precision physics |
| `f16` | 2 | float16 | Inference outputs |
| `bf16` | 2 | bfloat16\* | Requires ml\_dtypes; stored as 2-byte raw |
| `i64` | 8 | int64 | Timestamps, large indices |
| `i32` | 4 | int32 | Action indices, labels |
| `i16` | 2 | int16 | Quantized signals |
| `i8` | 1 | int8 | Quantized weights |
| `u64` | 8 | uint64 | Hashes, addresses |
| `u32` | 4 | uint32 | Indices, masks |
| `u16` | 2 | uint16 | Depth images |
| `u8` | 1 | uint8 | RGB pixels, raw bytes |
| `bool` | 1 | bool\_ | Done/termination flags |

\*bf16 uses `ml_dtypes.bfloat16` when available; falls back to uint16 to preserve byte layout.

---

## 7. Compression

Compression is per-block. Each index entry carries its own compression flags independently of the header `compression_default`.

| Codec | Flag bits | Notes |
|-------|-----------|-------|
| none | 0x0000 | Raw bytes; 32-byte-aligned for zero-copy mmap |
| zstd | 0x0003 | `EntryFlagCompressed` + `EntryFlagZstd` |
| lz4 | 0x0005 | `EntryFlagCompressed` + `EntryFlagLZ4` |

CRC32C is always computed over the **uncompressed** (original) bytes and stored in the index entry. The reader decompresses the block, then verifies the checksum.

---

## 8. Alignment

Data blocks start at boundaries determined by the `alignment` field in the header (0, 16, 32, or 64 bytes). The default is 32 bytes, which satisfies AVX2 alignment requirements. This means `mmap()` + pointer cast gives you a properly aligned float32 or uint8 array with zero copies on Linux and macOS. Padding between blocks (and between the string table and first block) is filled with `0x00` bytes.

---

## 9. Cross-language byte parity

Go is the reference implementation. Python and TypeScript produce byte-identical output for the same input. Verified by golden-file tests in `golden/`:

- `golden/generate.go` writes reference `.wshard` files
- Python and TypeScript read them and assert byte-level correctness
- Reference values are committed in `golden/golden_hashes.json`

```
CRC32C("hello")          = 0x9a71bb4c
xxHash64("signal/obs")   = 0x86f8c8413116a0ae
```

---

## 10. For more detail

This document covers the layout well enough to write a reader. For the full byte-level spec — including the `meta/wshard` timebase object, multi-modal signal naming, chunked episode manifest format, the DeepData bridge, and cross-language interop bugs that were found and fixed — read [DEEP_DIVE.md](DEEP_DIVE.md).
