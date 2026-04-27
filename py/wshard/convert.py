"""
W-SHARD conversion utilities.

High-level API for format detection and conversion.
"""

from pathlib import Path
from typing import Union, BinaryIO, Optional

from .types import Episode, Format
from .dreamer import load_dreamer, save_dreamer
from .wshard import load_wshard, save_wshard


def detect_format(path: Union[str, Path]) -> Format:
    """
    Detect episode format from file path or content.

    Args:
        path: File path

    Returns:
        Detected format
    """
    path = Path(path)
    suffix = path.suffix.lower()
    name = path.name.lower()

    # Check by extension
    if suffix == ".wshard" or suffix == ".shard":
        return Format.WSHARD

    if suffix == ".npz":
        # Could be DreamerV3
        # Check filename pattern: {timestamp}-{uuid}-{succ}-{length}.npz
        parts = path.stem.split("-")
        if len(parts) >= 4:
            return Format.DREAMER_V3
        return Format.DREAMER_V3  # Default NPZ to DreamerV3

    if suffix in (".pt", ".pth"):
        return Format.TDMPC2

    if suffix in (".hdf5", ".h5"):
        return Format.MINARI

    # Try content-based detection
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
        return detect_format_bytes(magic)
    except OSError:
        pass

    return Format.UNKNOWN


def detect_format_bytes(data: bytes) -> Format:
    """
    Detect format from magic bytes.

    Args:
        data: First 4+ bytes of file

    Returns:
        Detected format
    """
    if len(data) < 4:
        return Format.UNKNOWN

    # W-SHARD: 'SHRD'
    if data[:4] == b"SHRD":
        return Format.WSHARD

    # NPZ/ZIP: 'PK\x03\x04'
    if data[:4] == b"PK\x03\x04":
        return Format.DREAMER_V3

    # HDF5: '\x89HDF'
    if data[:4] == b"\x89HDF":
        return Format.MINARI

    return Format.UNKNOWN


def load(
    path_or_file: Union[str, Path, BinaryIO],
    format: Optional[Format] = None,
) -> Episode:
    """
    Load an episode from any supported format.

    Args:
        path_or_file: File path or file-like object
        format: Optional format override (auto-detected if not provided)

    Returns:
        Loaded episode
    """
    if format is None:
        if isinstance(path_or_file, (str, Path)):
            format = detect_format(path_or_file)
        else:
            # Try to read magic bytes
            pos = path_or_file.tell()
            magic = path_or_file.read(4)
            path_or_file.seek(pos)
            format = detect_format_bytes(magic)

    if format == Format.WSHARD:
        return load_wshard(path_or_file)
    elif format == Format.DREAMER_V3:
        return load_dreamer(path_or_file)
    elif format == Format.TDMPC2:
        raise NotImplementedError("TD-MPC2 format not yet implemented")
    elif format == Format.MINARI or format == Format.D4RL:
        raise NotImplementedError("Minari/D4RL format not yet implemented")
    else:
        raise ValueError(f"Unknown format: {format}")


def save(
    ep: Episode,
    path_or_file: Union[str, Path, BinaryIO],
    format: Optional[Format] = None,
) -> None:
    """
    Save an episode to any supported format.

    Args:
        ep: Episode to save
        path_or_file: Output file path or file-like object
        format: Optional format override (inferred from extension if not provided)
    """
    if format is None:
        if isinstance(path_or_file, (str, Path)):
            format = detect_format(path_or_file)
        else:
            format = Format.WSHARD  # Default to W-SHARD

    if format == Format.WSHARD:
        save_wshard(ep, path_or_file)
    elif format == Format.DREAMER_V3:
        save_dreamer(ep, path_or_file)
    elif format == Format.TDMPC2:
        raise NotImplementedError("TD-MPC2 format not yet implemented")
    elif format == Format.MINARI or format == Format.D4RL:
        raise NotImplementedError("Minari/D4RL format not yet implemented")
    else:
        raise ValueError(f"Unknown format: {format}")


def convert(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    input_format: Optional[Format] = None,
    output_format: Optional[Format] = None,
) -> Episode:
    """
    Convert an episode between formats.

    Args:
        input_path: Input file path
        output_path: Output file path
        input_format: Optional input format override
        output_format: Optional output format override

    Returns:
        The converted episode
    """
    # Load
    ep = load(input_path, format=input_format)

    # Save
    save(ep, output_path, format=output_format)

    return ep
