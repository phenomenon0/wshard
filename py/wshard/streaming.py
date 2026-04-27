"""
Streaming append-only episode writer for W-SHARD (Gap 4).

Enables incremental episode building for online learning. Episodes
build incrementally as the agent acts. Uses a reserve-write-finalize
pattern: header space is reserved upfront, data is written forward,
and the header is finalized with a seek-back at the end.
"""

import io
import json
import os
import struct
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import numpy as np
import xxhash

from .types import Episode, Channel, DType, TimebaseSpec, TimebaseType, Modality
from .wshard import (
    MAGIC, VERSION, ROLE_WSHARD,
    HEADER_SIZE, INDEX_ENTRY_SIZE, DEFAULT_ALIGNMENT,
    compute_crc32,
)
from .compress import (
    CompressionType,
    CompressionLevel,
    Compressor,
    compression_byte,
    should_compress,
    BLOCK_FLAG_COMPRESSED,
)

# Header flag indicating streaming/incomplete file
FLAG_STREAMING = 0x0040

# Default buffer size before flush
DEFAULT_FLUSH_INTERVAL = 64


class ChannelDef:
    """Channel definition for validation during streaming."""

    def __init__(self, name: str, dtype: DType, shape: List[int],
                 modality: Optional[Modality] = None):
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.modality = modality

    @property
    def element_size(self) -> int:
        """Bytes per timestep for this channel."""
        dim = 1
        for s in self.shape:
            dim *= s
        return dim * self.dtype.size if dim > 0 else self.dtype.size


class WShardStreamWriter:
    """
    Append-only episode writer for online learning.

    Usage:
        writer = WShardStreamWriter("episode.wshard", "ep_001", channel_defs)
        writer.begin_episode(env_id="CartPole-v1")

        for t, obs in enumerate(agent_loop):
            writer.write_timestep(
                t=t,
                observations={"state": obs},
                actions={"ctrl": action},
                reward=reward,
                done=(t == last_t),
            )

        writer.end_episode()
    """

    def __init__(
        self,
        path: Union[str, Path],
        episode_id: str,
        channel_defs: List[ChannelDef],
        max_timesteps: int = 100000,
        compression: CompressionType = CompressionType.NONE,
        compression_level: CompressionLevel = CompressionLevel.DEFAULT,
        flush_interval: int = DEFAULT_FLUSH_INTERVAL,
    ):
        self.path = Path(path)
        self.episode_id = episode_id
        self.channel_defs = {cd.name: cd for cd in channel_defs}
        self.max_timesteps = max_timesteps
        self.compression = compression
        self.compression_level = compression_level
        self.flush_interval = flush_interval

        # State
        self._file = None
        self._started = False
        self._finalized = False
        self._timestep_count = 0
        self._env_id = ""
        self._timebase = TimebaseSpec()

        # Buffered data per block
        self._buffers: Dict[str, bytearray] = {}
        self._buffered_count = 0

        # Track written block positions for finalization
        self._block_positions: Dict[str, Dict[str, Any]] = {}

        # Compressor
        self._compressor = None
        if compression != CompressionType.NONE:
            self._compressor = Compressor(compression, compression_level)

    def begin_episode(
        self,
        env_id: str = "",
        timebase: Optional[TimebaseSpec] = None,
    ) -> None:
        """
        Begin a new streaming episode.

        Reserves space for header and index, then starts writing data.
        """
        if self._started:
            raise RuntimeError("Episode already started")

        self._env_id = env_id
        if timebase:
            self._timebase = timebase
        self._started = True

        # Estimate max blocks: meta blocks + time + reward + done + channels + actions
        max_blocks = 4 + 1 + 1 + 1 + len(self.channel_defs) + len(self.channel_defs)
        reserved_index_size = max_blocks * INDEX_ENTRY_SIZE

        # String table estimate
        string_table_estimate = sum(
            len(f"signal/{name}".encode("utf-8")) + len(f"action/{name}".encode("utf-8"))
            for name in self.channel_defs
        ) + 200  # meta blocks + reward + done + time

        # Calculate reserved header space
        self._reserved_size = HEADER_SIZE + reserved_index_size + string_table_estimate
        self._reserved_size = _align(self._reserved_size, DEFAULT_ALIGNMENT)

        # Validate reserved space can fit expected index
        # Each channel produces signal + action blocks, plus meta/reward/done/time
        min_blocks = 4 + 1 + 1 + 1 + len(self.channel_defs) * 2
        min_header_space = HEADER_SIZE + min_blocks * INDEX_ENTRY_SIZE
        if self._reserved_size < min_header_space:
            raise RuntimeError(
                f"Reserved space {self._reserved_size} too small for {min_blocks} blocks "
                f"(need at least {min_header_space})"
            )

        # Write to .partial file, atomic rename on finalize
        self._partial_path = Path(str(self.path) + ".partial")
        self._file = open(self._partial_path, "w+b")
        self._file.write(b"\x00" * self._reserved_size)
        self._data_start = self._reserved_size

        # Initialize buffers
        for name in self.channel_defs:
            self._buffers[f"signal/{name}"] = bytearray()
            self._buffers[f"action/{name}"] = bytearray()
        self._buffers["reward"] = bytearray()
        self._buffers["done"] = bytearray()
        self._buffers["time/ticks"] = bytearray()

    def write_timestep(
        self,
        t: int,
        observations: Dict[str, np.ndarray],
        actions: Dict[str, np.ndarray],
        reward: float,
        done: bool,
    ) -> None:
        """
        Write a single timestep.

        Args:
            t: Timestep index
            observations: Dict of channel_name -> observation array
            actions: Dict of channel_name -> action array
            reward: Scalar reward
            done: Episode termination flag

        Raises:
            ValueError: If shapes don't match channel_defs
        """
        if not self._started:
            raise RuntimeError("Call begin_episode() first")
        if self._finalized:
            raise RuntimeError("Episode already finalized")
        if self._timestep_count >= self.max_timesteps:
            raise RuntimeError(f"Max timesteps ({self.max_timesteps}) exceeded")

        # Validate and buffer observations
        for name, data in observations.items():
            if name not in self.channel_defs:
                raise ValueError(f"Unknown channel: {name}")
            ch_def = self.channel_defs[name]
            expected_shape = ch_def.shape
            if expected_shape and list(data.shape) != expected_shape:
                raise ValueError(
                    f"Shape mismatch for {name}: expected {expected_shape}, got {list(data.shape)}"
                )
            self._buffers[f"signal/{name}"].extend(
                data.astype(ch_def.dtype.numpy_dtype).tobytes()
            )

        # Validate and buffer actions
        for name, data in actions.items():
            if name not in self.channel_defs:
                raise ValueError(f"Unknown action channel: {name}")
            ch_def = self.channel_defs[name]
            self._buffers[f"action/{name}"].extend(
                data.astype(ch_def.dtype.numpy_dtype).tobytes()
            )

        # Buffer reward and done
        self._buffers["reward"].extend(
            np.float32(reward).tobytes()
        )
        self._buffers["done"].extend(
            np.uint8(1 if done else 0).tobytes()
        )

        # Buffer time tick
        self._buffers["time/ticks"].extend(
            np.int32(t).tobytes()
        )

        self._timestep_count += 1
        self._buffered_count += 1

        # Flush buffer if interval reached
        if self._buffered_count >= self.flush_interval:
            self._flush_buffers()

    def _flush_buffers(self) -> None:
        """Flush buffered data to disk."""
        if not self._file or self._buffered_count == 0:
            return

        for block_name, buf in self._buffers.items():
            if len(buf) == 0:
                continue

            # Write to file at current position
            if block_name not in self._block_positions:
                self._block_positions[block_name] = {
                    "start_offset": self._file.tell(),
                    "total_written": 0,
                }

            self._file.write(bytes(buf))
            self._block_positions[block_name]["total_written"] += len(buf)
            buf.clear()

        self._file.flush()
        self._buffered_count = 0

    def end_episode(self) -> int:
        """
        Finalize the episode.

        Flushes remaining data, writes metadata blocks, then seeks
        back to write the real header and index.

        Returns:
            Total bytes written
        """
        if not self._started:
            raise RuntimeError("Call begin_episode() first")
        if self._finalized:
            raise RuntimeError("Already finalized")

        # Flush remaining buffer
        self._flush_buffers()

        # Build metadata blocks and write them at current position
        meta_blocks = self._build_metadata()
        for block_name, block_data in meta_blocks.items():
            pos = self._file.tell()

            # Compress if beneficial
            disk_data = block_data
            flags = 0
            if self._compressor and should_compress(block_name, block_data):
                compressed = self._compressor.compress(block_data)
                if len(compressed) < len(block_data):
                    disk_data = compressed
                    flags = BLOCK_FLAG_COMPRESSED

            # Align
            padding = (DEFAULT_ALIGNMENT - (pos % DEFAULT_ALIGNMENT)) % DEFAULT_ALIGNMENT
            if padding > 0:
                self._file.write(b"\x00" * padding)
                pos += padding

            self._file.write(disk_data)
            self._block_positions[block_name] = {
                "start_offset": pos,
                "total_written": len(disk_data),
                "orig_size": len(block_data),
                "flags": flags,
            }

        total_size = self._file.tell()

        # Now build the real header + index + string table
        all_blocks = sorted(self._block_positions.keys())

        # String table
        string_table = bytearray()
        string_offsets = {}
        for name in all_blocks:
            string_offsets[name] = len(string_table)
            string_table.extend(name.encode("utf-8"))

        entry_count = len(all_blocks)
        index_size = entry_count * INDEX_ENTRY_SIZE
        string_table_offset = HEADER_SIZE + index_size
        data_section_offset = string_table_offset + len(string_table)
        padding = (DEFAULT_ALIGNMENT - (data_section_offset % DEFAULT_ALIGNMENT)) % DEFAULT_ALIGNMENT
        data_section_offset += padding

        # Check that our header + index + string table fits in reserved space
        if data_section_offset > self._reserved_size:
            raise RuntimeError(
                f"Reserved space too small: need {data_section_offset}, have {self._reserved_size}"
            )

        # Build header
        header = bytearray(HEADER_SIZE)
        header[0:4] = MAGIC
        header[4] = VERSION
        header[5] = ROLE_WSHARD
        struct.pack_into("<H", header, 6, 0)  # clear streaming flag
        header[8] = DEFAULT_ALIGNMENT
        header[9] = compression_byte(self.compression)
        struct.pack_into("<H", header, 10, INDEX_ENTRY_SIZE)
        struct.pack_into("<I", header, 12, entry_count)
        struct.pack_into("<Q", header, 16, string_table_offset)
        struct.pack_into("<Q", header, 24, data_section_offset)
        struct.pack_into("<Q", header, 32, 0)  # schema offset
        struct.pack_into("<Q", header, 40, total_size)

        # Build index entries
        index = bytearray()
        for name in all_blocks:
            bp = self._block_positions[name]
            disk_size = bp["total_written"]
            orig_size = bp.get("orig_size", disk_size)
            flags = bp.get("flags", 0)

            # Read the disk data back for checksum
            self._file.seek(bp["start_offset"])
            disk_data = self._file.read(disk_size)
            checksum = compute_crc32(disk_data)

            # xxHash64, matching Go's xxhash.Sum64String()
            h = xxhash.xxh64(name.encode("utf-8")).intdigest()

            entry = bytearray(INDEX_ENTRY_SIZE)
            struct.pack_into("<Q", entry, 0, h)
            struct.pack_into("<I", entry, 8, string_offsets[name])
            struct.pack_into("<H", entry, 12, len(name.encode("utf-8")))
            struct.pack_into("<H", entry, 14, flags)
            struct.pack_into("<Q", entry, 16, bp["start_offset"])
            struct.pack_into("<Q", entry, 24, disk_size)
            struct.pack_into("<Q", entry, 32, orig_size)
            struct.pack_into("<I", entry, 40, checksum)
            index.extend(entry)

        # Seek to start and write the real header + index + string table
        self._file.seek(0)
        self._file.write(bytes(header))
        self._file.write(bytes(index))
        self._file.write(bytes(string_table))

        # Pad between string table and data section
        written_so_far = HEADER_SIZE + index_size + len(string_table)
        remaining_padding = data_section_offset - written_so_far
        if remaining_padding > 0:
            self._file.write(b"\x00" * remaining_padding)

        # Truncate file to total size
        self._file.seek(total_size)
        self._file.truncate()

        self._file.close()
        self._file = None

        # Atomic rename from .partial to final path
        os.replace(str(self._partial_path), str(self.path))
        self._finalized = True

        return total_size

    def _build_metadata(self) -> Dict[str, bytes]:
        """Build metadata blocks for finalization."""
        blocks = {}

        # meta/wshard
        meta_wshard = {
            "format": "W-SHARD",
            "wshard_version": "0.1",
            "endianness": "little",
            "alignment": 32,
            "streaming": True,
        }
        blocks["meta/wshard"] = json.dumps(meta_wshard).encode("utf-8")

        # meta/episode
        dt_ns = int(1e9 / self._timebase.tick_hz) if self._timebase.tick_hz > 0 else 33333333
        meta_episode = {
            "episode_id": self.episode_id,
            "env_id": self._env_id,
            "length_T": self._timestep_count,
            "timebase": {
                "type": self._timebase.type.value,
                "dt_ns": dt_ns,
            },
        }
        blocks["meta/episode"] = json.dumps(meta_episode).encode("utf-8")

        # meta/channels
        channels_list = []
        for name, ch_def in self.channel_defs.items():
            ch_entry: Dict[str, Any] = {
                "id": name,
                "dtype": ch_def.dtype.value,
                "shape": ch_def.shape,
                "signal_block": f"signal/{name}",
            }
            if ch_def.modality is not None:
                ch_entry["modality"] = ch_def.modality.value
            channels_list.append(ch_entry)
        blocks["meta/channels"] = json.dumps({"channels": channels_list}).encode("utf-8")

        return blocks

    @property
    def timestep_count(self) -> int:
        """Number of timesteps written so far."""
        return self._timestep_count

    @property
    def is_finalized(self) -> bool:
        """Whether the episode has been finalized."""
        return self._finalized

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._started and not self._finalized:
            if exc_type is None:
                self.end_episode()
            else:
                # On error, close file and delete .partial
                if self._file:
                    self._file.close()
                    self._file = None
                if hasattr(self, "_partial_path") and self._partial_path.exists():
                    self._partial_path.unlink()
        return False


def _align(n: int, alignment: int) -> int:
    if alignment == 0:
        return n
    return ((n + alignment - 1) // alignment) * alignment
