"""
W-SHARD native format adapter.

W-SHARD is a single-file episode container for action-conditioned world modeling.
It standardizes three lanes per channel:
  - Signal: Ground truth observations
  - Omen: Cached model predictions
  - Residual: Compact innovation representation

Container format: Shard v2 with role=0x05
"""

import io
import json
import struct
from pathlib import Path
from typing import Union, BinaryIO, Dict, Any, Optional
import numpy as np

import crc32c
import xxhash

from .types import Episode, Channel, DType, Format, TimebaseSpec, TimebaseType, Residual, Modality
from .compress import (
    CompressionType,
    CompressionLevel,
    Compressor,
    compression_byte,
    compression_from_byte,
    BLOCK_FLAG_COMPRESSED,
    BLOCK_FLAG_ZSTD,
    BLOCK_FLAG_LZ4,
    should_compress,
)
from .residual import (
    pack_residual_bitmask,
    unpack_residual_bitmask,
    HAS_COWRIE,
)

# Residual encoding identifiers
RESIDUAL_ENCODING_RAW = "raw"
RESIDUAL_ENCODING_COWRIE_BITMASK = "cowrie_bitmask"


# W-SHARD magic bytes
MAGIC = b"SHRD"
VERSION = 0x02
ROLE_WSHARD = 0x05

# Header size
HEADER_SIZE = 64
INDEX_ENTRY_SIZE = 48

# Default alignment
DEFAULT_ALIGNMENT = 32


def load_wshard(path_or_file: Union[str, Path, BinaryIO]) -> Episode:
    """
    Load an episode from W-SHARD format.

    Args:
        path_or_file: File path or file-like object

    Returns:
        Episode with loaded data
    """
    if isinstance(path_or_file, (str, Path)):
        with open(path_or_file, "rb") as f:
            data = f.read()
    else:
        data = path_or_file.read()

    return _decode_wshard(data)


def compute_crc32(data: bytes) -> int:
    """
    Compute CRC32C (Castagnoli) checksum, matching Go's crc32.Castagnoli.
    """
    return crc32c.crc32c(data)


def save_wshard(
    ep: Episode,
    path_or_file: Union[str, Path, BinaryIO],
    compression: CompressionType = CompressionType.NONE,
    compression_level: CompressionLevel = CompressionLevel.DEFAULT,
) -> None:
    """
    Save an episode to W-SHARD format.

    Args:
        ep: Episode to save
        path_or_file: Output file path or file-like object
        compression: Compression algorithm to use
        compression_level: Compression level
    """
    ep.validate()
    data = _encode_wshard(ep, compression, compression_level)

    if isinstance(path_or_file, (str, Path)):
        with open(path_or_file, "wb") as f:
            f.write(data)
    else:
        path_or_file.write(data)


def _decode_wshard(data: bytes) -> Episode:
    """Decode W-SHARD binary data to Episode."""
    if len(data) < HEADER_SIZE:
        raise ValueError("Data too short for W-SHARD header")

    # Check magic
    if data[:4] != MAGIC:
        raise ValueError(f"Invalid W-SHARD magic: {data[:4]!r}")

    # Check version and role
    version = data[4]
    role = data[5]
    if version != VERSION:
        raise ValueError(f"Unsupported W-SHARD version: {version}")
    if role != ROLE_WSHARD:
        raise ValueError(f"Invalid W-SHARD role: {role}")

    # Parse header
    flags = struct.unpack("<H", data[6:8])[0]
    alignment = data[8]
    compression_default = compression_from_byte(data[9])
    index_entry_size = struct.unpack("<H", data[10:12])[0]
    entry_count = struct.unpack("<I", data[12:16])[0]
    string_table_offset = struct.unpack("<Q", data[16:24])[0]
    data_section_offset = struct.unpack("<Q", data[24:32])[0]
    schema_offset = struct.unpack("<Q", data[32:40])[0]
    total_file_size = struct.unpack("<Q", data[40:48])[0]

    # Parse index entries
    index_start = HEADER_SIZE
    entries = []
    for i in range(entry_count):
        offset = index_start + i * index_entry_size
        entry = _parse_index_entry(data[offset : offset + index_entry_size])
        entries.append(entry)

    # Build string table (Go format: concatenated strings, indexed by offset)
    # The string table contains names concatenated without length prefixes
    # Each entry's name_offset and name_len point into this table
    string_table_size = int(data_section_offset) - int(string_table_offset)
    string_table = data[
        int(string_table_offset) : int(string_table_offset) + string_table_size
    ]

    # Load blocks — per-block compression detection from entry flags
    blocks = {}
    for entry in entries:
        name_offset = entry["name_offset"]
        name_len = entry["name_len"]

        # Extract name from string table
        if name_offset + name_len <= len(string_table):
            name = string_table[name_offset : name_offset + name_len].decode("utf-8")
        else:
            continue

        # Read block data (data_offset is absolute in Go format)
        block_offset = int(entry["data_offset"])
        disk_size = int(entry["disk_size"])
        orig_size = int(entry["orig_size"])
        entry_flags = entry["flags"]

        block_data = data[block_offset : block_offset + disk_size]

        # Decompress if needed — detect compression type per-block from flags
        if (entry_flags & BLOCK_FLAG_COMPRESSED) and disk_size != orig_size:
            if entry_flags & BLOCK_FLAG_LZ4:
                block_comp_type = CompressionType.LZ4
            elif entry_flags & BLOCK_FLAG_ZSTD:
                block_comp_type = CompressionType.ZSTD
            else:
                # Fall back to header-level default
                block_comp_type = compression_default
            decompressor = Compressor(block_comp_type)
            block_data = decompressor.decompress(block_data, orig_size)

        # Verify checksum on the logical block payload (uncompressed bytes)
        checksum = entry["checksum"]
        if checksum != 0:
            actual = compute_crc32(block_data)
            if actual != checksum:
                raise ValueError(
                    f"Checksum mismatch for {name}: "
                    f"expected {checksum:08x}, got {actual:08x}"
                )

        blocks[name] = block_data

    # Parse metadata
    meta_wshard = {}
    meta_episode = {}
    meta_channels = {}

    if "meta/wshard" in blocks:
        meta_wshard = json.loads(blocks["meta/wshard"].decode("utf-8"))
    if "meta/episode" in blocks:
        meta_episode = json.loads(blocks["meta/episode"].decode("utf-8"))
    if "meta/channels" in blocks:
        meta_channels = json.loads(blocks["meta/channels"].decode("utf-8"))

    # Create episode
    ep_id = meta_episode.get("episode_id", "unknown")
    length_t = meta_episode.get("length_T", 0)

    ep = Episode(id=ep_id, length=length_t)
    ep.source_format = Format.WSHARD
    ep.env_id = meta_episode.get("env_id", "")

    # Gap 1: Parse chunk fields
    if "chunk_index" in meta_episode:
        ep.chunk_index = meta_episode["chunk_index"]
        ep.total_chunks = meta_episode.get("total_chunks")
    if "timestep_range" in meta_episode:
        ep.timestep_range = meta_episode["timestep_range"]

    # Parse timebase
    timebase_meta = meta_episode.get("timebase", {})
    tb_type = timebase_meta.get("type", "ticks")
    if tb_type == "ticks":
        ep.timebase.type = TimebaseType.TICKS
        dt_ns = timebase_meta.get("dt_ns", 33333333)
        ep.timebase.tick_hz = 1e9 / dt_ns if dt_ns > 0 else 30.0
    elif tb_type == "timestamps_ns":
        ep.timebase.type = TimebaseType.TIMESTAMPS_NS

    # Parse channels from meta — route to observations or actions based on signal_block
    channels_list = meta_channels.get("channels", [])
    meta_parsed_actions = set()
    for ch_def in channels_list:
        ch_id = ch_def.get("id", "")
        signal_block = ch_def.get("signal_block", f"signal/{ch_id}")

        if signal_block in blocks:
            ch = _parse_tensor_block(blocks[signal_block], ch_def)
            if ch:
                if signal_block.startswith("action/"):
                    ep.actions[ch_id] = ch
                    meta_parsed_actions.add(signal_block)
                else:
                    # Gap 5: Restore modality fields
                    modality_str = ch_def.get("modality")
                    if modality_str:
                        try:
                            ch.modality = Modality(modality_str)
                        except ValueError:
                            pass
                    if "sampling_rate_hz" in ch_def:
                        ch.sampling_rate_hz = ch_def["sampling_rate_hz"]
                    if "content_type" in ch_def:
                        ch.content_type = ch_def["content_type"]
                    ep.observations[ch_id] = ch

    # Parse actions not already handled by meta/channels
    for name, block_data in blocks.items():
        if name.startswith("action/") and name not in meta_parsed_actions:
            action_name = name[7:]  # Strip 'action/'
            # Try to find action shape in episode metadata
            action_spaces = meta_episode.get("action_spaces", [])
            action_def = next((a for a in action_spaces if a.get("name") == name), None)
            if action_def:
                ch = _parse_tensor_block(
                    block_data,
                    {
                        "id": action_name,
                        "dtype": action_def.get("dtype", "f32"),
                        "shape": action_def.get("shape", []),
                    },
                )
            else:
                ch = _parse_simple_tensor(block_data, action_name)
            if ch:
                ep.actions[action_name] = ch

    # Parse reward
    if "reward" in blocks:
        ep.rewards = _parse_simple_tensor(blocks["reward"], "reward")

    # Parse done/terminations (bool type - stored as uint8)
    if "done" in blocks:
        ep.terminations = _parse_simple_tensor(blocks["done"], "done", DType.UINT8)
        # Convert to bool
        ep.terminations.data = ep.terminations.data.astype(np.bool_)
        ep.terminations.dtype = DType.BOOL

    # Detect residual encoding from metadata
    residual_encoding = meta_wshard.get("residual_encoding", RESIDUAL_ENCODING_RAW)

    # Parse residual blocks
    for name, block_data in blocks.items():
        if name.startswith("residual/") and name.endswith("/sign2nddiff"):
            channel_id = name[9:-12]  # Strip 'residual/' and '/sign2nddiff'
            ep.residuals[channel_id] = Residual(
                channel_id=channel_id,
                type="sign2nddiff",
                data=bytes(block_data),
            )
            # Store encoding hint in episode metadata for downstream consumers
            ep.metadata.setdefault("_residual_encoding", residual_encoding)

    return ep


def _encode_wshard(
    ep: Episode,
    compression: CompressionType = CompressionType.NONE,
    compression_level: CompressionLevel = CompressionLevel.DEFAULT,
) -> bytes:
    """Encode Episode to W-SHARD binary format."""
    # Build blocks
    blocks = {}

    # Meta/wshard
    meta_wshard = {
        "format": "W-SHARD",
        "wshard_version": "0.1",
        "endianness": "little",
        "alignment": 32,
        "residual_encoding": RESIDUAL_ENCODING_COWRIE_BITMASK if HAS_COWRIE else RESIDUAL_ENCODING_RAW,
    }
    blocks["meta/wshard"] = json.dumps(meta_wshard).encode("utf-8")

    # Meta/episode
    dt_ns = int(1e9 / ep.timebase.tick_hz) if ep.timebase.tick_hz > 0 else 33333333
    meta_episode: Dict[str, Any] = {
        "episode_id": ep.id,
        "env_id": ep.env_id,
        "length_T": ep.length,
        "timebase": {
            "type": ep.timebase.type.value,
            "dt_ns": dt_ns,
        },
    }
    # Gap 1: Chunked episode fields (optional, backward compat)
    if ep.chunk_index is not None:
        meta_episode["chunk_index"] = ep.chunk_index
        meta_episode["total_chunks"] = ep.total_chunks
    if ep.timestep_range is not None:
        meta_episode["timestep_range"] = ep.timestep_range
    blocks["meta/episode"] = json.dumps(meta_episode).encode("utf-8")

    # Meta/channels — include modality info (Gap 5) if present
    channels_list = []
    modality_groups: Dict[str, list] = {}
    for name, ch in ep.observations.items():
        ch_def: Dict[str, Any] = {
            "id": name,
            "dtype": ch.dtype.value,
            "shape": ch.shape,
            "signal_block": f"signal/{name}",
        }
        if ch.modality is not None:
            ch_def["modality"] = ch.modality.value
            ch_def["content_type_code"] = ch.modality.content_type
        if ch.sampling_rate_hz is not None:
            ch_def["sampling_rate_hz"] = ch.sampling_rate_hz
        if ch.content_type is not None:
            ch_def["content_type"] = ch.content_type
        channels_list.append(ch_def)
        # Build modality_groups index
        if ch.modality is not None:
            parts = name.split("/")
            group = parts[0] if len(parts) > 1 else "default"
            modality_groups.setdefault(group, []).append(
                {"channel_id": name, "modality": ch.modality.value}
            )
    meta_channels: Dict[str, Any] = {"channels": channels_list}
    if modality_groups:
        meta_channels["modality_groups"] = modality_groups
    blocks["meta/channels"] = json.dumps(meta_channels).encode("utf-8")

    # Signal blocks (observations)
    for name, ch in ep.observations.items():
        blocks[f"signal/{name}"] = _encode_tensor(ch)

    # Action blocks
    for name, ch in ep.actions.items():
        blocks[f"action/{name}"] = _encode_tensor(ch)

    # Reward
    if ep.rewards:
        blocks["reward"] = _encode_tensor(ep.rewards)

    # Done
    if ep.terminations:
        blocks["done"] = _encode_tensor(ep.terminations)

    # Time ticks
    ticks = np.arange(ep.length, dtype=np.int32)
    blocks["time/ticks"] = ticks.tobytes()

    # Sort block names for consistent ordering
    sorted_names = sorted(blocks.keys())

    # Build string table (Go format: concatenated strings without length prefix)
    string_table = bytearray()
    string_offsets = {}

    for name in sorted_names:
        string_offsets[name] = len(string_table)
        name_bytes = name.encode("utf-8")
        string_table.extend(name_bytes)

    # Create compressor if needed
    compressor = None
    if compression != CompressionType.NONE:
        compressor = Compressor(compression, compression_level)

    # Compress blocks and track sizes
    compressed_blocks = {}
    block_flags = {}
    orig_sizes = {}

    # Per-block compression type flags (matching Go's EntryFlag bits)
    comp_type_flag = 0
    if compression == CompressionType.ZSTD:
        comp_type_flag = BLOCK_FLAG_ZSTD
    elif compression == CompressionType.LZ4:
        comp_type_flag = BLOCK_FLAG_LZ4

    for name in sorted_names:
        orig_data = blocks[name]
        orig_sizes[name] = len(orig_data)

        if compressor and should_compress(name, orig_data):
            compressed = compressor.compress(orig_data)
            if len(compressed) < len(orig_data):
                compressed_blocks[name] = compressed
                block_flags[name] = BLOCK_FLAG_COMPRESSED | comp_type_flag
            else:
                compressed_blocks[name] = orig_data
                block_flags[name] = 0
        else:
            compressed_blocks[name] = orig_data
            block_flags[name] = 0

    # Calculate offsets
    entry_count = len(sorted_names)
    index_size = entry_count * INDEX_ENTRY_SIZE
    string_table_offset = HEADER_SIZE + index_size

    # Align data section to 32 bytes
    data_section_offset = string_table_offset + len(string_table)
    padding = (
        DEFAULT_ALIGNMENT - (data_section_offset % DEFAULT_ALIGNMENT)
    ) % DEFAULT_ALIGNMENT
    data_section_offset += padding

    # Calculate data offsets (absolute positions)
    current_offset = data_section_offset
    block_offsets = {}

    for name in sorted_names:
        block_offsets[name] = current_offset
        disk_size = len(compressed_blocks[name])
        padded_size = (
            disk_size
            + (DEFAULT_ALIGNMENT - (disk_size % DEFAULT_ALIGNMENT)) % DEFAULT_ALIGNMENT
        )
        current_offset += padded_size

    total_size = current_offset

    # Build header
    header = bytearray(HEADER_SIZE)
    header[0:4] = MAGIC
    header[4] = VERSION
    header[5] = ROLE_WSHARD
    struct.pack_into("<H", header, 6, 0)  # flags
    header[8] = DEFAULT_ALIGNMENT  # alignment
    header[9] = compression_byte(compression)  # compression
    struct.pack_into("<H", header, 10, INDEX_ENTRY_SIZE)
    struct.pack_into("<I", header, 12, entry_count)
    struct.pack_into("<Q", header, 16, string_table_offset)
    struct.pack_into("<Q", header, 24, data_section_offset)
    struct.pack_into("<Q", header, 32, 0)  # schema_offset
    struct.pack_into("<Q", header, 40, total_size)

    # Build index
    index = bytearray()
    for name in sorted_names:
        disk_data = compressed_blocks[name]
        disk_size = len(disk_data)
        orig_size = orig_sizes[name]
        flags = block_flags[name]
        checksum = compute_crc32(blocks[name])

        entry = bytearray(INDEX_ENTRY_SIZE)
        # xxHash64 name hash, matching Go's xxhash.Sum64String()
        h = xxhash.xxh64(name.encode("utf-8")).intdigest()
        struct.pack_into("<Q", entry, 0, h)
        struct.pack_into("<I", entry, 8, string_offsets[name])
        struct.pack_into("<H", entry, 12, len(name.encode("utf-8")))
        struct.pack_into("<H", entry, 14, flags)
        struct.pack_into("<Q", entry, 16, block_offsets[name])  # absolute offset
        struct.pack_into("<Q", entry, 24, disk_size)
        struct.pack_into("<Q", entry, 32, orig_size)
        struct.pack_into("<I", entry, 40, checksum)
        index.extend(entry)

    # Assemble file
    result = bytearray()
    result.extend(header)
    result.extend(index)
    result.extend(string_table)
    result.extend(b"\x00" * padding)

    # Write data blocks with alignment padding
    for name in sorted_names:
        disk_data = compressed_blocks[name]
        result.extend(disk_data)
        # Pad to alignment
        pad_size = (
            DEFAULT_ALIGNMENT - (len(disk_data) % DEFAULT_ALIGNMENT)
        ) % DEFAULT_ALIGNMENT
        result.extend(b"\x00" * pad_size)

    return bytes(result)


def _parse_index_entry(data: bytes) -> Dict[str, Any]:
    """Parse a single index entry."""
    return {
        "name_hash": struct.unpack("<Q", data[0:8])[0],
        "name_offset": struct.unpack("<I", data[8:12])[0],
        "name_len": struct.unpack("<H", data[12:14])[0],
        "flags": struct.unpack("<H", data[14:16])[0],
        "data_offset": struct.unpack("<Q", data[16:24])[0],
        "disk_size": struct.unpack("<Q", data[24:32])[0],
        "orig_size": struct.unpack("<Q", data[32:40])[0],
        "checksum": struct.unpack("<I", data[40:44])[0],
    }


def _parse_tensor_block(data: bytes, ch_def: Dict) -> Channel:
    """Parse a tensor block using channel definition."""
    dtype_str = ch_def.get("dtype", "f32")
    shape = ch_def.get("shape", [])
    name = ch_def.get("id", "unknown")

    dtype = DType(dtype_str)
    np_dtype = dtype.numpy_dtype

    arr = np.frombuffer(data, dtype=np_dtype)

    # Reshape if shape is defined
    if shape:
        full_shape = [-1] + shape
        try:
            arr = arr.reshape(full_shape)
        except ValueError as e:
            raise ValueError(
                f"Cannot reshape block '{name}': data has {arr.size} elements, "
                f"shape {full_shape} requires a multiple of {np.prod(shape)}"
            ) from e

    return Channel(
        name=name,
        dtype=dtype,
        shape=shape,
        data=arr,
    )


# ============================================================
# Gap 5: VLA Multi-Modal helpers
# ============================================================

def add_multimodal_observation(
    ep: Episode,
    group: str,
    modality: Modality,
    channel: Channel,
) -> None:
    """
    Add a multi-modal observation to an episode using hierarchical naming.

    Convention: signal/{group}/{modality} e.g. signal/obs/rgb, signal/obs/language

    Args:
        ep: Episode to add observation to
        group: Observation group (e.g. "obs", "goal")
        modality: Sensor modality type
        channel: Channel data to add
    """
    key = f"{group}/{modality.value}"
    channel.modality = modality
    ep.observations[key] = channel


def get_multimodal_observations(
    ep: Episode,
    group: Optional[str] = None,
    modality: Optional[Modality] = None,
) -> Dict[str, Channel]:
    """
    Get multi-modal observations, optionally filtered by group and/or modality.

    Args:
        ep: Episode to query
        group: Filter by group (e.g. "obs"), or None for all
        modality: Filter by modality, or None for all

    Returns:
        Dict of matching observation channels
    """
    result = {}
    for key, ch in ep.observations.items():
        parts = key.split("/")
        if group is not None and (len(parts) < 1 or parts[0] != group):
            continue
        if modality is not None and ch.modality != modality:
            continue
        result[key] = ch
    return result


# ============================================================
# Gap 2: Latent Action Storage helpers
# ============================================================

# Channel name conventions for latent actions
LATENT_ACTION_CHANNEL = "latent_action"
LATENT_CODEBOOK_CHANNEL = "latent_action_codebook"


def set_latent_actions(
    ep: Episode,
    model_id: str,
    embeddings: Channel,
    codebook_indices: Optional[Channel] = None,
) -> None:
    """
    Store latent action embeddings (and optional codebook indices) in the omen lane.

    Convention:
      - omen/latent_action/{model_id} for continuous embeddings
      - omen/latent_action_codebook/{model_id} for VQ-VAE indices

    Args:
        ep: Episode to add latent actions to
        model_id: Model identifier (e.g. "genie3_v1")
        embeddings: Continuous latent action embeddings [T, latent_dim]
        codebook_indices: Optional VQ-VAE codebook indices [T]
    """
    if LATENT_ACTION_CHANNEL not in ep.omens:
        ep.omens[LATENT_ACTION_CHANNEL] = {}
    ep.omens[LATENT_ACTION_CHANNEL][model_id] = embeddings

    if codebook_indices is not None:
        if LATENT_CODEBOOK_CHANNEL not in ep.omens:
            ep.omens[LATENT_CODEBOOK_CHANNEL] = {}
        ep.omens[LATENT_CODEBOOK_CHANNEL][model_id] = codebook_indices


def get_latent_actions(
    ep: Episode,
    model_id: str,
) -> Optional[Channel]:
    """Get latent action embeddings for a model."""
    return ep.omens.get(LATENT_ACTION_CHANNEL, {}).get(model_id)


def get_latent_codebook(
    ep: Episode,
    model_id: str,
) -> Optional[Channel]:
    """Get latent action codebook indices for a model."""
    return ep.omens.get(LATENT_CODEBOOK_CHANNEL, {}).get(model_id)


def _parse_simple_tensor(
    data: bytes, name: str, dtype: DType = DType.FLOAT32
) -> Channel:
    """Parse a simple tensor block."""
    np_dtype = dtype.numpy_dtype
    arr = np.frombuffer(data, dtype=np_dtype)
    return Channel(
        name=name,
        dtype=dtype,
        shape=[],
        data=arr,
    )


def _encode_tensor(ch: Channel) -> bytes:
    """Encode a channel to raw bytes."""
    return ch.data.tobytes()


def decode_residuals(ep: Episode) -> None:
    """
    Auto-decode residual blocks based on residual_encoding metadata.

    Dispatches to the appropriate unpacking function based on the
    encoding type stored in the episode metadata.
    """
    encoding = ep.metadata.get("_residual_encoding", RESIDUAL_ENCODING_RAW)
    for ch_id, residual in ep.residuals.items():
        if residual.type == "sign2nddiff":
            if encoding == RESIDUAL_ENCODING_COWRIE_BITMASK and HAS_COWRIE:
                residual.mask = bytes(unpack_residual_bitmask(
                    residual.data, ep.length
                ))
            else:
                residual.mask = bytes(unpack_residual_bitmask(
                    residual.data, ep.length
                ))
