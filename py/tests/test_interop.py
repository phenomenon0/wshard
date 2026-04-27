"""
Interop and edge-case tests for W-SHARD Python package.

Covers: CRC32C, xxHash64, bf16, per-block compression, reshape errors,
LZ4 bounds, streaming .partial files, and chunk continuity validation.
"""

import io
import json
import os
import shutil
import struct
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pytest
import xxhash

from wshard.wshard import (
    compute_crc32,
    _parse_tensor_block,
    load_wshard,
    save_wshard,
    decode_residuals,
    MAGIC,
    VERSION,
    ROLE_WSHARD,
    HEADER_SIZE,
    INDEX_ENTRY_SIZE,
    DEFAULT_ALIGNMENT,
)
from wshard.types import DType, Episode, Channel, Modality
from wshard.compress import (
    CompressionType,
    decompress,
    BLOCK_FLAG_COMPRESSED,
    BLOCK_FLAG_ZSTD,
    BLOCK_FLAG_LZ4,
    LZ4_MAX_DECOMPRESS_SIZE,
)
from wshard.streaming import WShardStreamWriter, ChannelDef
from wshard.chunked import ChunkManifest, validate_chunk_continuity


# ============================================================
# 1. CRC32C verification
# ============================================================


def test_crc32c_hello():
    """CRC32C('hello') must equal the well-known constant 0x9a71bb4c."""
    assert compute_crc32(b"hello") == 0x9A71BB4C


def test_crc32c_empty():
    """CRC32C of empty bytes is 0."""
    assert compute_crc32(b"") == 0


def test_crc32c_deterministic():
    """Same input always gives the same checksum."""
    data = b"deterministic test payload"
    assert compute_crc32(data) == compute_crc32(data)


# ============================================================
# 2. xxHash64 verification
# ============================================================


def test_xxhash64_signal_obs():
    """xxHash64 of 'signal/obs' must match Go's xxhash.Sum64String output.

    Go reference (cespare/xxhash/v2):
        xxhash.Sum64String("signal/obs") == 0x86f8c8413116a0ae
    Python xxhash uses the same algorithm (XXH64, seed=0) so values must match.
    """
    h = xxhash.xxh64(b"signal/obs").intdigest()
    # Verify it is a stable 64-bit value and matches the Go constant.
    assert h == 0x86F8C8413116A0AE


def test_xxhash64_meta_manifest():
    """Verify another known key used in the codebase."""
    h = xxhash.xxh64(b"meta/manifest").intdigest()
    # Must be non-zero and deterministic.
    assert h != 0
    assert h == xxhash.xxh64(b"meta/manifest").intdigest()


# ============================================================
# 3. bf16 handling
# ============================================================


def test_bf16_numpy_dtype_is_2_bytes():
    """DType.BFLOAT16.numpy_dtype must be a 2-byte dtype regardless of
    whether ml_dtypes is installed."""
    dt = DType.BFLOAT16.numpy_dtype
    assert dt.itemsize == 2


def test_bf16_numpy_dtype_is_ml_dtypes_or_uint16():
    """bf16 dtype is either ml_dtypes.bfloat16 or numpy.uint16."""
    dt = DType.BFLOAT16.numpy_dtype
    try:
        import ml_dtypes
        assert dt == np.dtype(ml_dtypes.bfloat16)
    except ImportError:
        assert dt == np.dtype(np.uint16)


def test_bf16_byte_roundtrip():
    """Create bf16 bytes, wrap in ndarray, and verify byte equality."""
    # Two bf16 values: 1.0 (0x3F80) and -2.0 (0xC000)
    raw = b"\x80\x3f\x00\xc0"
    dt = DType.BFLOAT16.numpy_dtype
    arr = np.frombuffer(raw, dtype=dt)
    assert arr.nbytes == 4
    assert arr.tobytes() == raw


def test_bf16_size_property():
    """DType.BFLOAT16.size must report 2."""
    assert DType.BFLOAT16.size == 2


# ============================================================
# 4. Per-block compression
# ============================================================


def _build_wshard_with_per_block_zstd(signal_data: bytes) -> bytes:
    """Build a minimal W-SHARD file where compression_default=NONE but one
    signal block is compressed with ZSTD and flagged per-block."""
    import zstandard as zstd

    compressed = zstd.ZstdCompressor().compress(signal_data)
    orig_size = len(signal_data)
    disk_size = len(compressed)

    # We need blocks: meta/channels, meta/episode, meta/wshard, signal/x
    blocks_meta = {
        "meta/channels": json.dumps({
            "channels": [{"id": "x", "dtype": "f32", "shape": [], "signal_block": "signal/x"}]
        }).encode(),
        "meta/episode": json.dumps({
            "episode_id": "perblock", "length_T": orig_size // 4,
            "timebase": {"type": "ticks", "dt_ns": 33333333},
        }).encode(),
        "meta/wshard": json.dumps({
            "format": "W-SHARD", "wshard_version": "0.1",
        }).encode(),
    }

    sorted_names = sorted(list(blocks_meta.keys()) + ["signal/x"])

    # String table
    string_table = bytearray()
    string_offsets = {}
    for name in sorted_names:
        string_offsets[name] = len(string_table)
        string_table.extend(name.encode("utf-8"))

    entry_count = len(sorted_names)
    index_size = entry_count * INDEX_ENTRY_SIZE
    st_offset = HEADER_SIZE + index_size
    ds_offset = st_offset + len(string_table)
    pad = (DEFAULT_ALIGNMENT - (ds_offset % DEFAULT_ALIGNMENT)) % DEFAULT_ALIGNMENT
    ds_offset += pad

    # Calculate data offsets
    current = ds_offset
    block_offsets = {}
    block_disk_data = {}
    block_orig_sizes = {}
    block_flags = {}

    for name in sorted_names:
        block_offsets[name] = current
        if name == "signal/x":
            block_disk_data[name] = compressed
            block_orig_sizes[name] = orig_size
            block_flags[name] = BLOCK_FLAG_COMPRESSED | BLOCK_FLAG_ZSTD
        else:
            raw = blocks_meta[name]
            block_disk_data[name] = raw
            block_orig_sizes[name] = len(raw)
            block_flags[name] = 0
        blen = len(block_disk_data[name])
        padded = blen + (DEFAULT_ALIGNMENT - (blen % DEFAULT_ALIGNMENT)) % DEFAULT_ALIGNMENT
        current += padded

    total_size = current

    # Header — compression_default byte = 0 (NONE)
    header = bytearray(HEADER_SIZE)
    header[0:4] = MAGIC
    header[4] = VERSION
    header[5] = ROLE_WSHARD
    header[8] = DEFAULT_ALIGNMENT
    header[9] = 0  # compression_default = NONE
    struct.pack_into("<H", header, 10, INDEX_ENTRY_SIZE)
    struct.pack_into("<I", header, 12, entry_count)
    struct.pack_into("<Q", header, 16, st_offset)
    struct.pack_into("<Q", header, 24, ds_offset)
    struct.pack_into("<Q", header, 32, 0)
    struct.pack_into("<Q", header, 40, total_size)

    # Index
    index = bytearray()
    for name in sorted_names:
        dd = block_disk_data[name]
        h = xxhash.xxh64(name.encode("utf-8")).intdigest()
        checksum = compute_crc32(signal_data if name == "signal/x" else blocks_meta[name])
        entry = bytearray(INDEX_ENTRY_SIZE)
        struct.pack_into("<Q", entry, 0, h)
        struct.pack_into("<I", entry, 8, string_offsets[name])
        struct.pack_into("<H", entry, 12, len(name.encode("utf-8")))
        struct.pack_into("<H", entry, 14, block_flags[name])
        struct.pack_into("<Q", entry, 16, block_offsets[name])
        struct.pack_into("<Q", entry, 24, len(dd))
        struct.pack_into("<Q", entry, 32, block_orig_sizes[name])
        struct.pack_into("<I", entry, 40, checksum)
        index.extend(entry)

    result = bytearray()
    result.extend(header)
    result.extend(index)
    result.extend(string_table)
    result.extend(b"\x00" * pad)
    for name in sorted_names:
        dd = block_disk_data[name]
        result.extend(dd)
        p = (DEFAULT_ALIGNMENT - (len(dd) % DEFAULT_ALIGNMENT)) % DEFAULT_ALIGNMENT
        result.extend(b"\x00" * p)

    return bytes(result)


def test_per_block_zstd_decompression(tmp_path):
    """Reader must decompress a ZSTD-flagged block even when
    compression_default is NONE."""
    T = 20
    signal = np.arange(T, dtype=np.float32)
    raw_bytes = signal.tobytes()

    shard_bytes = _build_wshard_with_per_block_zstd(raw_bytes)
    path = tmp_path / "perblock.wshard"
    path.write_bytes(shard_bytes)

    ep = load_wshard(path)
    assert ep.length == T
    assert "x" in ep.observations
    np.testing.assert_array_almost_equal(ep.observations["x"].data, signal)


# ============================================================
# 5. Reshape error propagation
# ============================================================


def test_parse_tensor_block_raises_on_incompatible_shape():
    """_parse_tensor_block must raise ValueError when shape is incompatible
    with data size — it must NOT silently flatten."""
    # 10 float32 elements = 40 bytes. Shape [3] requires a multiple of 3.
    data = np.arange(10, dtype=np.float32).tobytes()
    ch_def = {"id": "bad", "dtype": "f32", "shape": [3]}

    with pytest.raises(ValueError, match="Cannot reshape"):
        _parse_tensor_block(data, ch_def)


def test_parse_tensor_block_succeeds_on_compatible_shape():
    """_parse_tensor_block must succeed when shape divides data evenly."""
    data = np.arange(12, dtype=np.float32).tobytes()
    ch_def = {"id": "ok", "dtype": "f32", "shape": [3]}

    ch = _parse_tensor_block(data, ch_def)
    assert ch.data.shape == (4, 3)


# ============================================================
# 6. LZ4 bounds guard
# ============================================================


def test_lz4_decompress_orig_size_zero():
    """decompress must raise ValueError when orig_size is 0."""
    with pytest.raises(ValueError, match="orig_size.*out of bounds"):
        decompress(b"\x00" * 10, 0, CompressionType.LZ4)


def test_lz4_decompress_orig_size_too_large():
    """decompress must raise ValueError when orig_size > 256 MB."""
    with pytest.raises(ValueError, match="orig_size.*out of bounds"):
        decompress(b"\x00" * 10, LZ4_MAX_DECOMPRESS_SIZE + 1, CompressionType.LZ4)


def test_lz4_decompress_negative_orig_size():
    """decompress must raise ValueError when orig_size is negative."""
    with pytest.raises(ValueError, match="orig_size.*out of bounds"):
        decompress(b"\x00" * 10, -1, CompressionType.LZ4)


# ============================================================
# 7. Streaming .partial file
# ============================================================


def test_streaming_writes_partial_during_recording(tmp_path):
    """During recording, the writer must write to a .partial file.
    On finalize, the .partial is renamed to the final path."""
    final_path = tmp_path / "episode.wshard"
    partial_path = Path(str(final_path) + ".partial")

    defs = [ChannelDef("s", DType.FLOAT32, [2])]
    writer = WShardStreamWriter(final_path, "ep1", defs)
    writer.begin_episode(env_id="TestEnv")

    # .partial must exist, final must not
    assert partial_path.exists()
    assert not final_path.exists()

    for t in range(5):
        writer.write_timestep(
            t=t,
            observations={"s": np.array([float(t), float(t)], dtype=np.float32)},
            actions={"s": np.array([0.0, 0.0], dtype=np.float32)},
            reward=0.0,
            done=(t == 4),
        )

    writer.end_episode()

    # After finalize: final exists, .partial gone
    assert final_path.exists()
    assert not partial_path.exists()

    # Verify it is a valid wshard
    ep = load_wshard(final_path)
    assert ep.length == 5


def test_streaming_partial_deleted_on_error(tmp_path):
    """On error during streaming (context manager), the .partial file
    must be cleaned up."""
    final_path = tmp_path / "episode_err.wshard"
    partial_path = Path(str(final_path) + ".partial")

    defs = [ChannelDef("s", DType.FLOAT32, [])]

    class FakeError(Exception):
        pass

    with pytest.raises(FakeError):
        with WShardStreamWriter(final_path, "ep_err", defs) as writer:
            writer.begin_episode()
            writer.write_timestep(
                t=0,
                observations={"s": np.float32(1.0)},
                actions={"s": np.float32(0.0)},
                reward=0.0,
                done=False,
            )
            raise FakeError("simulated failure")

    # .partial must be cleaned up, final must not exist
    assert not partial_path.exists()
    assert not final_path.exists()


# ============================================================
# 8. Chunk continuity validation
# ============================================================


def test_validate_chunk_continuity_valid():
    """A properly ordered manifest passes validation."""
    m = ChunkManifest("ep1")
    m.add_chunk(0, "c0.wshard", "aaa", [0, 99], 100)
    m.add_chunk(1, "c1.wshard", "bbb", [100, 199], 100)
    m.add_chunk(2, "c2.wshard", "ccc", [200, 299], 100)
    # Should not raise
    validate_chunk_continuity(m)


def test_validate_chunk_continuity_gap():
    """Gaps in chunk_index sequence must be detected."""
    m = ChunkManifest("ep1")
    m.add_chunk(0, "c0.wshard", "aaa", [0, 99], 100)
    m.add_chunk(2, "c2.wshard", "ccc", [200, 299], 100)  # missing index 1

    with pytest.raises(ValueError, match="gaps or duplicates"):
        validate_chunk_continuity(m)


def test_validate_chunk_continuity_duplicate():
    """Duplicate chunk_index must be detected."""
    m = ChunkManifest("ep1")
    m.add_chunk(0, "c0.wshard", "aaa", [0, 99], 100)
    m.add_chunk(0, "c0b.wshard", "bbb", [0, 99], 100)  # duplicate

    with pytest.raises(ValueError, match="gaps or duplicates"):
        validate_chunk_continuity(m)


def test_validate_chunk_continuity_timestep_discontinuity():
    """Timestep ranges that do not connect must be detected."""
    m = ChunkManifest("ep1")
    m.add_chunk(0, "c0.wshard", "aaa", [0, 99], 100)
    m.add_chunk(1, "c1.wshard", "bbb", [150, 249], 100)  # gap at 100-149

    with pytest.raises(ValueError, match="Timestep gap"):
        validate_chunk_continuity(m)


def test_validate_chunk_continuity_length_mismatch():
    """Sum of chunk lengths must match total_timesteps."""
    m = ChunkManifest("ep1")
    m.add_chunk(0, "c0.wshard", "aaa", [0, 99], 100)
    m.add_chunk(1, "c1.wshard", "bbb", [100, 199], 100)
    # Artificially break the total
    m.total_timesteps = 999

    with pytest.raises(ValueError, match="total_timesteps"):
        validate_chunk_continuity(m)


def test_validate_chunk_continuity_empty():
    """Empty manifest passes validation (no chunks is valid)."""
    m = ChunkManifest("empty")
    validate_chunk_continuity(m)  # should not raise


# ============================================================
# 9. Codec Invariant 1 — decode(encode(x)) == x
# ============================================================


def _make_test_episode() -> Episode:
    """Build a small Episode with known data for round-trip tests."""
    T = 8
    obs_data = np.arange(T * 3, dtype=np.float32).reshape(T, 3)
    act_data = np.zeros(T, dtype=np.float32)
    rew_data = np.linspace(0.0, 1.0, T, dtype=np.float32)
    term_data = np.zeros(T, dtype=np.bool_)
    term_data[-1] = True

    return Episode(
        id="roundtrip-ep",
        env_id="TestEnv-v0",
        length=T,
        observations={
            "state": Channel(name="state", dtype=DType.FLOAT32, shape=[3], data=obs_data),
        },
        actions={
            "ctrl": Channel(name="ctrl", dtype=DType.FLOAT32, shape=[], data=act_data),
        },
        rewards=Channel(name="reward", dtype=DType.FLOAT32, shape=[], data=rew_data),
        terminations=Channel(name="term", dtype=DType.BOOL, shape=[], data=term_data),
    )


def test_roundtrip_decode_encode(tmp_path):
    """Codec Invariant 1: decode(encode(x)) == x.

    All fields must survive a save→load round-trip exactly, not just
    'loads without error'.
    """
    ep = _make_test_episode()
    path = tmp_path / "roundtrip.wshard"

    save_wshard(ep, path)
    loaded = load_wshard(path)

    # Identity
    assert loaded.id == ep.id, f"episode_id mismatch: {loaded.id!r} != {ep.id!r}"
    assert loaded.env_id == ep.env_id, f"env_id mismatch: {loaded.env_id!r} != {ep.env_id!r}"
    assert loaded.length == ep.length, f"length mismatch: {loaded.length} != {ep.length}"

    # Observations
    assert set(loaded.observations.keys()) == set(ep.observations.keys())
    for name in ep.observations:
        orig = ep.observations[name]
        got = loaded.observations[name]
        assert got.shape == orig.shape, f"obs {name} shape mismatch"
        assert got.dtype == orig.dtype, f"obs {name} dtype mismatch"
        np.testing.assert_array_equal(got.data, orig.data, err_msg=f"obs {name} data mismatch")

    # Actions
    assert set(loaded.actions.keys()) == set(ep.actions.keys())
    for name in ep.actions:
        orig = ep.actions[name]
        got = loaded.actions[name]
        assert got.shape == orig.shape, f"act {name} shape mismatch"
        np.testing.assert_array_equal(got.data, orig.data, err_msg=f"act {name} data mismatch")

    # Rewards
    assert loaded.rewards is not None, "rewards lost in round-trip"
    np.testing.assert_array_equal(loaded.rewards.data, ep.rewards.data, err_msg="reward data mismatch")

    # Terminations
    assert loaded.terminations is not None, "terminations lost in round-trip"
    np.testing.assert_array_equal(
        loaded.terminations.data, ep.terminations.data, err_msg="termination data mismatch"
    )


# ============================================================
# 10. Codec Invariant 5 — Deterministic output
# ============================================================


def test_deterministic_encode(tmp_path):
    """Codec Invariant 5: encoding the same Episode twice must produce
    byte-identical output."""
    ep = _make_test_episode()

    buf1 = io.BytesIO()
    save_wshard(ep, buf1)
    bytes1 = buf1.getvalue()

    buf2 = io.BytesIO()
    save_wshard(ep, buf2)
    bytes2 = buf2.getvalue()

    assert bytes1 == bytes2, (
        f"Non-deterministic encode: {len(bytes1)} bytes vs {len(bytes2)} bytes, "
        f"first diff at byte {next(i for i, (a, b) in enumerate(zip(bytes1, bytes2)) if a != b) if bytes1 != bytes2 else -1}"
    )


# ============================================================
# 11. Codec Invariant 3 — Error on truncated input
# ============================================================


def test_truncated_input_raises(tmp_path):
    """Codec Invariant 3: progressively truncated wshard bytes must raise
    an error, not silently return garbage."""
    ep = _make_test_episode()
    path = tmp_path / "trunc_source.wshard"
    save_wshard(ep, path)
    full_bytes = path.read_bytes()

    for cut in [1, 10, 32, 64]:
        truncated = full_bytes[:-cut]
        with pytest.raises((ValueError, struct.error, Exception)):
            load_wshard(io.BytesIO(truncated))


# ============================================================
# 12. Overflow Gauntlet (Class 4)
# ============================================================


def test_index_entry_disk_size_exceeds_file(tmp_path):
    """Overflow Gauntlet: an index entry whose disk_size exceeds the actual
    file length must raise, not read out-of-bounds memory."""
    ep = _make_test_episode()
    path = tmp_path / "overflow.wshard"
    save_wshard(ep, path)
    data = bytearray(path.read_bytes())

    # Patch the first index entry's disk_size (offset 24 in each 48-byte entry,
    # entries start at HEADER_SIZE).
    entry_offset = HEADER_SIZE + 24  # disk_size field of first index entry
    # Set disk_size to something absurdly large (but not max uint64)
    struct.pack_into("<Q", data, entry_offset, len(data) * 100)

    with pytest.raises(Exception):
        load_wshard(io.BytesIO(bytes(data)))


def test_index_entry_orig_size_max_uint64(tmp_path):
    """Overflow Gauntlet: orig_size = 0xFFFFFFFFFFFFFFFF on a compressed block
    must not cause an OOM allocation — should raise before allocating."""
    ep = _make_test_episode()
    path = tmp_path / "origmax.wshard"
    save_wshard(ep, path)
    data = bytearray(path.read_bytes())

    # Patch first index entry: set COMPRESSED|ZSTD flags and orig_size = max uint64.
    # flags field is at offset 14 (2 bytes) within each 48-byte entry.
    flags_offset = HEADER_SIZE + 14
    struct.pack_into("<H", data, flags_offset, BLOCK_FLAG_COMPRESSED | BLOCK_FLAG_ZSTD)
    # orig_size field is at offset 32 within each 48-byte entry.
    orig_offset = HEADER_SIZE + 32
    struct.pack_into("<Q", data, orig_offset, 0xFFFFFFFFFFFFFFFF)
    # Zero the checksum so CRC check doesn't fire first.
    crc_offset = HEADER_SIZE + 40
    struct.pack_into("<I", data, crc_offset, 0)

    # Must raise (decompression error or OOM guard), not try to allocate 16 EB
    with pytest.raises(Exception):
        load_wshard(io.BytesIO(bytes(data)))


# ============================================================
# 13. Cross-Language Parity (golden files)
# ============================================================


_GOLDEN_DIR = Path(__file__).resolve().parent.parent.parent / "golden"
_REPO_ROOT = Path(__file__).resolve().parents[4]


def _go_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("GOCACHE", "/tmp/go-build-cache")
    return env


def _load_golden_hashes():
    """Load golden_hashes.json if it exists."""
    hashes_file = _GOLDEN_DIR / "golden_hashes.json"
    if not _GOLDEN_DIR.exists() or not hashes_file.exists():
        return None
    import json as _json
    return _json.loads(hashes_file.read_text())


def test_golden_hash_parity():
    """Cross-language parity: CRC32C and xxHash64 match Go-computed golden values."""
    hashes = _load_golden_hashes()
    if hashes is None:
        pytest.skip("golden directory or golden_hashes.json not found")

    # CRC32C of "hello" must match Go
    expected_crc = int(hashes["crc32c_hello"], 16)
    assert compute_crc32(b"hello") == expected_crc, (
        f"CRC32C('hello') = 0x{compute_crc32(b'hello'):08x}, "
        f"Go says {hashes['crc32c_hello']}"
    )

    # xxHash64 of "signal/obs" must match Go
    expected_xx = int(hashes["xxhash64_signal_obs"], 16)
    got_xx = xxhash.xxh64(b"signal/obs").intdigest()
    assert got_xx == expected_xx, (
        f"xxHash64('signal/obs') = 0x{got_xx:016x}, "
        f"Go says {hashes['xxhash64_signal_obs']}"
    )

    # xxHash64 of "meta/manifest" must match Go
    expected_mm = int(hashes["xxhash64_meta_manifest"], 16)
    got_mm = xxhash.xxh64(b"meta/manifest").intdigest()
    assert got_mm == expected_mm, (
        f"xxHash64('meta/manifest') = 0x{got_mm:016x}, "
        f"Go says {hashes['xxhash64_meta_manifest']}"
    )


def test_golden_dtype_sizes_parity():
    """Cross-language parity: dtype sizes match Go reference."""
    hashes = _load_golden_hashes()
    if hashes is None:
        pytest.skip("golden directory or golden_hashes.json not found")

    from wshard.types import DType
    for dtype_str, expected_size in hashes["dtype_sizes"].items():
        dt = DType(dtype_str)
        assert dt.size == expected_size, (
            f"DType('{dtype_str}').size = {dt.size}, Go says {expected_size}"
        )


def test_golden_simple_episode_loads():
    """Cross-language parity: simple_episode.wshard written by Go loads correctly in Python."""
    fpath = _GOLDEN_DIR / "simple_episode.wshard"
    if not fpath.exists():
        pytest.skip("golden/simple_episode.wshard not found")

    ep = load_wshard(fpath)
    assert ep.id == "golden_simple"
    assert ep.env_id == "TestEnv-v1"
    assert ep.length == 10
    assert "state" in ep.observations
    assert ep.observations["state"].data.shape == (10, 4)
    assert ep.observations["state"].dtype == DType.FLOAT32
    # First timestep obs should be [0.0, 1.0, 2.0, 3.0]
    np.testing.assert_allclose(ep.observations["state"].data[0], [0.0, 1.0, 2.0, 3.0])
    assert "ctrl" in ep.actions
    assert ep.actions["ctrl"].data.shape == (10, 2)
    assert ep.rewards is not None
    assert ep.rewards.length == 10


def test_golden_dtype_zoo_loads():
    """Cross-language parity: dtype_zoo.wshard with multiple dtypes loads correctly."""
    fpath = _GOLDEN_DIR / "dtype_zoo.wshard"
    if not fpath.exists():
        pytest.skip("golden/dtype_zoo.wshard not found")

    ep = load_wshard(fpath)
    assert ep.id == "golden_dtypes"
    assert ep.length == 4
    assert "f32_ch" in ep.observations
    assert "f64_ch" in ep.observations
    assert "i32_ch" in ep.observations
    assert "u8_ch" in ep.observations


def test_golden_compressed_loads():
    """Cross-language parity: per_block_compressed.wshard with zstd loads correctly."""
    fpath = _GOLDEN_DIR / "per_block_compressed.wshard"
    if not fpath.exists():
        pytest.skip("golden/per_block_compressed.wshard not found")

    ep = load_wshard(fpath)
    assert ep.id == "golden_compressed"
    assert ep.length == 100
    assert "obs" in ep.observations
    assert ep.observations["obs"].data.shape == (100, 8)


# ============================================================
# 14. Cross-language package interop (Go package <-> Python package)
# ============================================================


def test_go_package_interop_roundtrip(tmp_path):
    """Cross-language parity: the real Go package must read/write the same
    canonical W-SHARD metadata shape as the Python package."""
    if shutil.which("go") is None:
        pytest.skip("go toolchain not available")

    go_module_dir = _REPO_ROOT / "shard" / "go"
    go_out_path = tmp_path / "go_package_writer.wshard"
    go_writer = tmp_path / "go_writer_probe.go"
    go_out_literal = json.dumps(go_out_path.as_posix())
    go_writer.write_text(
        textwrap.dedent(
            f"""
            package main

            import (
                "log"
                shard "github.com/Neumenon/shard/go/shard"
            )

            func main() {{
                ep := &shard.WShardEpisode{{
                    ID:      "go-probe",
                    EnvID:   "GoEnv-v0",
                    LengthT: 3,
                    Timebase: shard.WShardTimebase{{Type: "ticks", TickHz: 25}},
                    Observations: map[string]*shard.WShardChannel{{
                        "state/pos": {{
                            Name:  "state/pos",
                            DType: "float32",
                            Shape: []int{{2}},
                            Data:  make([]byte, 3*2*4),
                        }},
                    }},
                    Actions: map[string]*shard.WShardChannel{{
                        "ctrl": {{
                            Name:  "ctrl",
                            DType: "int32",
                            Shape: []int{{1}},
                            Data:  make([]byte, 3*4),
                        }},
                    }},
                    Rewards:      []float32{{0, 1, 2}},
                    Terminations: []bool{{false, false, true}},
                }}
                if err := shard.CreateWShard({go_out_literal}, ep); err != nil {{
                    log.Fatal(err)
                }}
            }}
            """
        ),
        encoding="utf-8",
    )
    subprocess.run(
        ["go", "run", str(go_writer)],
        cwd=go_module_dir,
        check=True,
        env=_go_env(),
    )

    loaded = load_wshard(go_out_path)
    assert loaded.id == "go-probe"
    assert loaded.env_id == "GoEnv-v0"
    assert loaded.length == 3
    assert "state/pos" in loaded.observations
    assert loaded.observations["state/pos"].dtype == DType.FLOAT32
    assert loaded.observations["state/pos"].data.shape == (3, 2)
    assert "ctrl" in loaded.actions
    assert loaded.actions["ctrl"].dtype == DType.INT32
    assert loaded.actions["ctrl"].data.shape == (3, 1)

    py_path = tmp_path / "python_package_writer.wshard"
    py_path_literal = json.dumps(py_path.as_posix())
    py_ep = Episode(id="py-probe", length=4)
    py_ep.env_id = "PyEnv-v0"
    py_ep.observations["state/pos"] = Channel(
        name="state/pos",
        dtype=DType.FLOAT32,
        shape=[2],
        data=np.zeros((4, 2), dtype=np.float32),
    )
    py_ep.actions["ctrl"] = Channel(
        name="ctrl",
        dtype=DType.FLOAT32,
        shape=[1],
        data=np.zeros((4, 1), dtype=np.float32),
    )
    py_ep.rewards = Channel(
        name="reward",
        dtype=DType.FLOAT32,
        shape=[],
        data=np.arange(4, dtype=np.float32),
    )
    py_ep.terminations = Channel(
        name="done",
        dtype=DType.BOOL,
        shape=[],
        data=np.array([False, False, False, True], dtype=np.bool_),
    )
    save_wshard(py_ep, py_path)

    go_reader = tmp_path / "go_reader_probe.go"
    go_reader.write_text(
        textwrap.dedent(
            f"""
            package main

            import (
                "encoding/json"
                "log"
                "os"
                shard "github.com/Neumenon/shard/go/shard"
            )

            func main() {{
                ep, err := shard.OpenWShard({py_path_literal})
                if err != nil {{
                    log.Fatal(err)
                }}
                summary := map[string]any{{
                    "id": ep.ID,
                    "env_id": ep.EnvID,
                    "length_t": ep.LengthT,
                    "obs_count": len(ep.Observations),
                    "action_count": len(ep.Actions),
                    "has_state_pos": ep.Observations["state/pos"] != nil,
                    "has_ctrl": ep.Actions["ctrl"] != nil,
                }}
                if ch := ep.Observations["state/pos"]; ch != nil {{
                    summary["state_pos_dtype"] = ch.DType
                }}
                if err := json.NewEncoder(os.Stdout).Encode(summary); err != nil {{
                    log.Fatal(err)
                }}
            }}
            """
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        ["go", "run", str(go_reader)],
        cwd=go_module_dir,
        check=True,
        capture_output=True,
        env=_go_env(),
        text=True,
    )
    summary = json.loads(proc.stdout)
    assert summary["id"] == "py-probe"
    assert summary["env_id"] == "PyEnv-v0"
    assert summary["length_t"] == 4
    assert summary["obs_count"] == 1
    assert summary["action_count"] == 1
    assert summary["has_state_pos"] is True
    assert summary["has_ctrl"] is True
    assert summary["state_pos_dtype"] == "f32"
