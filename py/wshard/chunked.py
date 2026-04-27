"""
Chunked episode support for W-SHARD (Gap 1).

Enables episodes spanning multiple shard files for Cosmos-scale datasets.
Each chunk is a normal wshard file. A manifest shard (role=0x04) indexes
all chunks with URIs, SHA-256 hashes, and timestep ranges.
"""

import hashlib
import json
import struct
from pathlib import Path
from typing import Iterator, List, Optional, Dict, Any

import numpy as np

from .types import Episode, Channel, DType
from .wshard import save_wshard, load_wshard, MAGIC, VERSION, HEADER_SIZE, INDEX_ENTRY_SIZE, DEFAULT_ALIGNMENT
from .compress import CompressionType, CompressionLevel

# Manifest shard role
ROLE_MANIFEST = 0x04


class ChunkManifest:
    """Manifest describing all chunks of a chunked episode."""

    def __init__(self, episode_id: str, env_id: str = ""):
        self.episode_id = episode_id
        self.env_id = env_id
        self.chunks: List[Dict[str, Any]] = []
        self.total_timesteps: int = 0

    def add_chunk(
        self,
        chunk_index: int,
        uri: str,
        sha256: str,
        timestep_range: List[int],
        length_t: int,
    ) -> None:
        """Add a chunk entry to the manifest."""
        self.chunks.append({
            "chunk_index": chunk_index,
            "uri": uri,
            "sha256": sha256,
            "timestep_range": timestep_range,
            "length_T": length_t,
        })
        self.total_timesteps += length_t

    def to_json(self) -> bytes:
        """Serialize manifest to JSON bytes."""
        manifest = {
            "episode_id": self.episode_id,
            "env_id": self.env_id,
            "total_chunks": len(self.chunks),
            "total_timesteps": self.total_timesteps,
            "chunks": self.chunks,
        }
        return json.dumps(manifest, indent=2).encode("utf-8")

    @classmethod
    def from_json(cls, data: bytes) -> "ChunkManifest":
        """Deserialize manifest from JSON bytes."""
        obj = json.loads(data.decode("utf-8"))
        manifest = cls(
            episode_id=obj["episode_id"],
            env_id=obj.get("env_id", ""),
        )
        manifest.total_timesteps = obj.get("total_timesteps", 0)
        manifest.chunks = obj.get("chunks", [])
        return manifest


class ChunkedEpisodeWriter:
    """
    Writes a long episode as multiple chunk files plus a manifest.

    Usage:
        writer = ChunkedEpisodeWriter("output/", "ep_001", chunk_size_t=10000)
        for chunk_ep in chunk_episodes:
            writer.write_chunk(chunk_ep)
        writer.finalize_manifest()
    """

    def __init__(
        self,
        base_path: str,
        episode_id: str,
        chunk_size_t: int = 10000,
        env_id: str = "",
        compression: CompressionType = CompressionType.NONE,
        compression_level: CompressionLevel = CompressionLevel.DEFAULT,
    ):
        self.base_path = Path(base_path)
        self.episode_id = episode_id
        self.chunk_size_t = chunk_size_t
        self.compression = compression
        self.compression_level = compression_level
        self.manifest = ChunkManifest(episode_id, env_id)
        self._chunk_count = 0

        self.base_path.mkdir(parents=True, exist_ok=True)

    def _chunk_filename(self, chunk_index: int) -> str:
        return f"{self.episode_id}_chunk_{chunk_index:04d}.wshard"

    def write_chunk(self, episode_slice: Episode, chunk_index: Optional[int] = None) -> Path:
        """
        Write one chunk file.

        Args:
            episode_slice: Episode containing this chunk's data
            chunk_index: Explicit index, or auto-incremented

        Returns:
            Path to the written chunk file
        """
        if chunk_index is None:
            chunk_index = self._chunk_count

        # Set chunk metadata on the episode
        episode_slice.chunk_index = chunk_index

        filename = self._chunk_filename(chunk_index)
        chunk_path = self.base_path / filename

        save_wshard(episode_slice, chunk_path, self.compression, self.compression_level)

        # Compute SHA-256
        with open(chunk_path, "rb") as f:
            sha256 = hashlib.sha256(f.read()).hexdigest()

        # Compute timestep range
        global_start = chunk_index * self.chunk_size_t
        timestep_range = [global_start, global_start + episode_slice.length - 1]
        if episode_slice.timestep_range is not None:
            timestep_range = episode_slice.timestep_range

        self.manifest.add_chunk(
            chunk_index=chunk_index,
            uri=filename,
            sha256=sha256,
            timestep_range=timestep_range,
            length_t=episode_slice.length,
        )

        self._chunk_count = max(self._chunk_count, chunk_index + 1)
        return chunk_path

    def write_episode_chunked(self, episode: Episode) -> List[Path]:
        """
        Split a full episode into chunks and write all of them.

        Args:
            episode: Full episode to split

        Returns:
            List of chunk file paths
        """
        T = episode.length
        total_chunks = (T + self.chunk_size_t - 1) // self.chunk_size_t
        paths = []

        for ci in range(total_chunks):
            start_t = ci * self.chunk_size_t
            end_t = min(start_t + self.chunk_size_t, T)
            chunk_len = end_t - start_t

            chunk_ep = Episode(
                id=episode.id,
                env_id=episode.env_id,
                length=chunk_len,
                timebase=episode.timebase,
                chunk_index=ci,
                total_chunks=total_chunks,
                timestep_range=[start_t, end_t - 1],
            )

            # Slice observations
            for name, ch in episode.observations.items():
                chunk_ep.observations[name] = Channel(
                    name=ch.name,
                    dtype=ch.dtype,
                    shape=ch.shape,
                    data=ch.data[start_t:end_t].copy(),
                    semantics=ch.semantics,
                    modality=ch.modality,
                    sampling_rate_hz=ch.sampling_rate_hz,
                    content_type=ch.content_type,
                )

            # Slice actions
            for name, ch in episode.actions.items():
                chunk_ep.actions[name] = Channel(
                    name=ch.name,
                    dtype=ch.dtype,
                    shape=ch.shape,
                    data=ch.data[start_t:end_t].copy(),
                )

            # Slice rewards
            if episode.rewards is not None:
                chunk_ep.rewards = Channel(
                    name=episode.rewards.name,
                    dtype=episode.rewards.dtype,
                    shape=episode.rewards.shape,
                    data=episode.rewards.data[start_t:end_t].copy(),
                )

            # Slice terminations
            if episode.terminations is not None:
                chunk_ep.terminations = Channel(
                    name=episode.terminations.name,
                    dtype=episode.terminations.dtype,
                    shape=episode.terminations.shape,
                    data=episode.terminations.data[start_t:end_t].copy(),
                )

            paths.append(self.write_chunk(chunk_ep, ci))

        return paths

    def finalize_manifest(self) -> Path:
        """
        Write the manifest shard file.

        Returns:
            Path to the manifest file
        """
        # Update total_chunks on all chunks
        total = len(self.manifest.chunks)
        for chunk in self.manifest.chunks:
            chunk["total_chunks"] = total

        manifest_path = self.base_path / f"{self.episode_id}_manifest.wshard"
        manifest_data = self.manifest.to_json()

        # Write a minimal shard v2 file with role=0x04 containing the manifest JSON
        _write_manifest_shard(manifest_path, manifest_data)

        return manifest_path


class ChunkedEpisodeReader:
    """
    Reads a chunked episode from a manifest shard.

    Usage:
        reader = ChunkedEpisodeReader("output/ep_001_manifest.wshard")
        for chunk_ep in reader.iter_chunks():
            process(chunk_ep)
    """

    def __init__(self, manifest_path: str):
        self.manifest_path = Path(manifest_path)
        self.base_dir = self.manifest_path.parent
        self.manifest: Optional[ChunkManifest] = None

    def load_manifest(self) -> ChunkManifest:
        """Load and parse the manifest shard."""
        data = self.manifest_path.read_bytes()

        # Parse minimal shard v2 header to find manifest JSON block
        if len(data) < HEADER_SIZE:
            raise ValueError("Manifest file too short")
        if data[:4] != MAGIC:
            raise ValueError("Invalid manifest magic")
        role = data[5]
        if role != ROLE_MANIFEST:
            raise ValueError(f"Not a manifest shard: role={role}")

        entry_count = struct.unpack("<I", data[12:16])[0]
        string_table_offset = struct.unpack("<Q", data[16:24])[0]
        data_section_offset = struct.unpack("<Q", data[24:32])[0]

        # Read the first (and only) data block
        if entry_count < 1:
            raise ValueError("Manifest has no entries")

        entry_offset = HEADER_SIZE
        entry_data = data[entry_offset:entry_offset + INDEX_ENTRY_SIZE]
        block_offset = struct.unpack("<Q", entry_data[16:24])[0]
        block_size = struct.unpack("<Q", entry_data[24:32])[0]

        manifest_json = data[block_offset:block_offset + block_size]
        self.manifest = ChunkManifest.from_json(manifest_json)
        return self.manifest

    def iter_chunks(self) -> Iterator[Episode]:
        """Iterate over chunks, loading each lazily."""
        if self.manifest is None:
            self.load_manifest()

        for chunk_info in sorted(self.manifest.chunks, key=lambda c: c["chunk_index"]):
            uri = chunk_info["uri"]
            chunk_path = self.base_dir / uri
            ep = load_wshard(chunk_path)

            # Verify SHA-256 if present
            expected_sha = chunk_info.get("sha256")
            if expected_sha:
                actual_sha = hashlib.sha256(chunk_path.read_bytes()).hexdigest()
                if actual_sha != expected_sha:
                    raise ValueError(
                        f"SHA-256 mismatch for chunk {chunk_info['chunk_index']}: "
                        f"expected {expected_sha}, got {actual_sha}"
                    )

            yield ep

    def load_chunk(self, chunk_index: int) -> Episode:
        """Load a specific chunk by index."""
        if self.manifest is None:
            self.load_manifest()

        for chunk_info in self.manifest.chunks:
            if chunk_info["chunk_index"] == chunk_index:
                uri = chunk_info["uri"]
                return load_wshard(self.base_dir / uri)

        raise ValueError(f"Chunk {chunk_index} not found in manifest")

    @property
    def total_timesteps(self) -> int:
        if self.manifest is None:
            self.load_manifest()
        return self.manifest.total_timesteps

    @property
    def num_chunks(self) -> int:
        if self.manifest is None:
            self.load_manifest()
        return len(self.manifest.chunks)


def validate_chunk_continuity(manifest: ChunkManifest) -> None:
    """
    Validate continuity of chunks in a manifest.

    Checks:
    - No gaps or duplicates in chunk_index sequence
    - Contiguous timestep_range between adjacent chunks
    - Sum of length_T matches total_timesteps

    Raises:
        ValueError: If any continuity check fails
    """
    if not manifest.chunks:
        return

    sorted_chunks = sorted(manifest.chunks, key=lambda c: c["chunk_index"])

    # Check for gaps/duplicates in chunk_index
    indices = [c["chunk_index"] for c in sorted_chunks]
    expected = list(range(len(sorted_chunks)))
    if indices != expected:
        raise ValueError(
            f"Chunk index sequence has gaps or duplicates: {indices}, expected {expected}"
        )

    # Check contiguous timestep_range
    for i in range(1, len(sorted_chunks)):
        prev = sorted_chunks[i - 1]
        curr = sorted_chunks[i]
        prev_range = prev.get("timestep_range")
        curr_range = curr.get("timestep_range")
        if prev_range and curr_range:
            if curr_range[0] != prev_range[1] + 1:
                raise ValueError(
                    f"Timestep gap between chunk {prev['chunk_index']} "
                    f"(ends {prev_range[1]}) and chunk {curr['chunk_index']} "
                    f"(starts {curr_range[0]})"
                )

    # Check total length
    total_len = sum(c.get("length_T", 0) for c in sorted_chunks)
    if total_len != manifest.total_timesteps:
        raise ValueError(
            f"Sum of chunk lengths ({total_len}) != total_timesteps ({manifest.total_timesteps})"
        )


def _write_manifest_shard(path: Path, manifest_json: bytes) -> None:
    """Write a minimal shard v2 file with role=0x04 containing manifest JSON."""
    import crc32c
    import xxhash

    block_name = b"meta/manifest"
    name_len = len(block_name)

    # Layout: header + 1 index entry + string table + padding + data
    index_size = INDEX_ENTRY_SIZE
    string_table_offset = HEADER_SIZE + index_size
    data_section_offset = string_table_offset + name_len
    # Align data section
    padding = (DEFAULT_ALIGNMENT - (data_section_offset % DEFAULT_ALIGNMENT)) % DEFAULT_ALIGNMENT
    data_section_offset += padding
    total_size = data_section_offset + len(manifest_json)

    # Header
    header = bytearray(HEADER_SIZE)
    header[0:4] = MAGIC
    header[4] = VERSION
    header[5] = ROLE_MANIFEST
    struct.pack_into("<H", header, 6, 0)  # flags
    header[8] = DEFAULT_ALIGNMENT
    header[9] = 0  # no compression
    struct.pack_into("<H", header, 10, INDEX_ENTRY_SIZE)
    struct.pack_into("<I", header, 12, 1)  # entry count
    struct.pack_into("<Q", header, 16, string_table_offset)
    struct.pack_into("<Q", header, 24, data_section_offset)
    struct.pack_into("<Q", header, 32, 0)  # schema offset
    struct.pack_into("<Q", header, 40, total_size)

    # Index entry — xxHash64 + CRC32C (matching Go)
    h = xxhash.xxh64(b"meta/manifest").intdigest()
    checksum = crc32c.crc32c(manifest_json)

    entry = bytearray(INDEX_ENTRY_SIZE)
    struct.pack_into("<Q", entry, 0, h)
    struct.pack_into("<I", entry, 8, 0)  # name offset
    struct.pack_into("<H", entry, 12, name_len)
    struct.pack_into("<H", entry, 14, 0)  # flags
    struct.pack_into("<Q", entry, 16, data_section_offset)
    struct.pack_into("<Q", entry, 24, len(manifest_json))
    struct.pack_into("<Q", entry, 32, len(manifest_json))
    struct.pack_into("<I", entry, 40, checksum)

    # Assemble
    result = bytearray()
    result.extend(header)
    result.extend(entry)
    result.extend(block_name)
    result.extend(b"\x00" * padding)
    result.extend(manifest_json)

    path.write_bytes(bytes(result))
