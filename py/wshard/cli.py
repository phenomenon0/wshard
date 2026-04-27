"""
wshard CLI — five subcommands for the W-SHARD episode format.

Usage:
    wshard inspect <path.wshard>
    wshard verify <path.wshard> [--strict]
    wshard convert <input> <output> [--input-format FMT] [--output-format FMT]
    wshard export <path.wshard> --format FMT [-o OUTPUT]
    wshard doctor
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# ANSI helpers (no external deps)
# ---------------------------------------------------------------------------
_RED = "\033[31m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RESET = "\033[0m"

# Disable colour when stdout is not a tty (e.g. piped into a file).
def _coloured(text: str, code: str) -> str:
    if sys.stdout.isatty():
        return f"{code}{text}{_RESET}"
    return text

def red(t: str) -> str:    return _coloured(t, _RED)
def green(t: str) -> str:  return _coloured(t, _GREEN)
def yellow(t: str) -> str: return _coloured(t, _YELLOW)


# ---------------------------------------------------------------------------
# Low-level W-SHARD header / index parser (no decompression)
# Used by `verify` to walk blocks without needing to import wshard internals.
# ---------------------------------------------------------------------------
_MAGIC = b"SHRD"
_HEADER_SIZE = 64
_INDEX_ENTRY_SIZE = 48

# EntryFlag bits (must match wshard.compress)
_BLOCK_FLAG_COMPRESSED = 0x0001
_BLOCK_FLAG_ZSTD       = 0x0002
_BLOCK_FLAG_LZ4        = 0x0004


def _read_header(data: bytes) -> dict:
    """Parse the 64-byte W-SHARD file header."""
    if len(data) < _HEADER_SIZE:
        raise ValueError(f"File too short for W-SHARD header ({len(data)} bytes)")
    if data[:4] != _MAGIC:
        raise ValueError(f"Not a W-SHARD file (magic={data[:4]!r})")
    version = data[4]
    role    = data[5]
    flags              = struct.unpack_from("<H", data, 6)[0]
    alignment          = data[8]
    compression_default = data[9]
    index_entry_size   = struct.unpack_from("<H", data, 10)[0]
    entry_count        = struct.unpack_from("<I", data, 12)[0]
    string_table_offset = struct.unpack_from("<Q", data, 16)[0]
    data_section_offset = struct.unpack_from("<Q", data, 24)[0]
    schema_offset       = struct.unpack_from("<Q", data, 32)[0]
    total_file_size     = struct.unpack_from("<Q", data, 40)[0]
    return dict(
        version=version,
        role=role,
        flags=flags,
        alignment=alignment,
        compression_default=compression_default,
        index_entry_size=index_entry_size,
        entry_count=entry_count,
        string_table_offset=string_table_offset,
        data_section_offset=data_section_offset,
        schema_offset=schema_offset,
        total_file_size=total_file_size,
    )


def _read_index_entries(data: bytes, hdr: dict) -> list[dict]:
    """Return a list of parsed index entry dicts."""
    es = hdr["index_entry_size"]
    n  = hdr["entry_count"]
    entries = []
    for i in range(n):
        off = _HEADER_SIZE + i * es
        e = data[off : off + es]
        if len(e) < 44:
            break
        entries.append(dict(
            name_hash  = struct.unpack_from("<Q", e, 0)[0],
            name_offset= struct.unpack_from("<I", e, 8)[0],
            name_len   = struct.unpack_from("<H", e, 12)[0],
            flags      = struct.unpack_from("<H", e, 14)[0],
            data_offset= struct.unpack_from("<Q", e, 16)[0],
            disk_size  = struct.unpack_from("<Q", e, 24)[0],
            orig_size  = struct.unpack_from("<Q", e, 32)[0],
            checksum   = struct.unpack_from("<I", e, 40)[0],
        ))
    return entries


def _resolve_names(data: bytes, hdr: dict, entries: list[dict]) -> list[tuple[str, dict]]:
    """Attach names (from string table) to each entry."""
    st_off = int(hdr["string_table_offset"])
    ds_off = int(hdr["data_section_offset"])
    st_size = ds_off - st_off
    string_table = data[st_off : st_off + st_size]
    named = []
    for e in entries:
        no = e["name_offset"]
        nl = e["name_len"]
        name = string_table[no : no + nl].decode("utf-8", errors="replace")
        named.append((name, e))
    return named


# ---------------------------------------------------------------------------
# `wshard inspect`
# ---------------------------------------------------------------------------

def cmd_inspect(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if not path.exists():
        print(f"error: file not found: {path}", file=sys.stderr)
        return 1

    file_size = path.stat().st_size

    # Load via public API — gives us Channel objects with dtype/shape/data.
    from wshard import load_wshard
    try:
        ep = load_wshard(path)
    except Exception as exc:
        print(f"error: failed to load {path}: {exc}", file=sys.stderr)
        return 1

    # Re-read raw data for on-disk block sizes.
    with open(path, "rb") as fh:
        raw = fh.read()

    try:
        hdr = _read_header(raw)
        entries = _read_index_entries(raw, hdr)
        named_entries = _resolve_names(raw, hdr, entries)
        # Build lookup: name -> entry
        entry_by_name = {n: e for n, e in named_entries}
    except Exception:
        entry_by_name = {}

    total_blocks = len(entry_by_name)

    # Header
    print(f"path       : {path}")
    print(f"file size  : {file_size:,} bytes")
    print(f"blocks     : {total_blocks}")
    print(f"episode id : {ep.id!r}")
    print(f"env id     : {ep.env_id!r}")
    print(f"length     : {ep.length} timesteps")
    print()

    col_w = (44, 8, 24, 10, 12)
    header_row = (
        f"{'block':{col_w[0]}s}"
        f"  {'dtype':{col_w[1]}s}"
        f"  {'shape':{col_w[2]}s}"
        f"  {'compressed':{col_w[3]}s}"
        f"  {'disk bytes':>{col_w[4]}s}"
    )
    print(header_row)
    print("-" * (sum(col_w) + 4 * 2))

    def _entry_info(block_name: str):
        e = entry_by_name.get(block_name)
        if e is None:
            return "-", "-"
        is_comp = bool(e["flags"] & _BLOCK_FLAG_COMPRESSED)
        return ("yes" if is_comp else "no"), f"{e['disk_size']:,}"

    def _fmt_shape(ch) -> str:
        if ch.data is None:
            return "?"
        return "x".join(str(s) for s in ch.data.shape)

    def _fmt_dtype(ch) -> str:
        if hasattr(ch.dtype, "name"):
            return ch.dtype.name.lower()
        return str(ch.dtype)

    def _print_channel(block_prefix: str, name: str, ch) -> None:
        full_block = f"{block_prefix}{name}"
        dtype  = _fmt_dtype(ch)
        shape  = _fmt_shape(ch)
        comp, disk = _entry_info(full_block)
        print(
            f"{full_block:{col_w[0]}s}"
            f"  {dtype:{col_w[1]}s}"
            f"  {shape:{col_w[2]}s}"
            f"  {comp:{col_w[3]}s}"
            f"  {disk:>{col_w[4]}s}"
        )

    for name, ch in sorted(ep.observations.items()):
        _print_channel("signal/", name, ch)
    for name, ch in sorted(ep.actions.items()):
        _print_channel("action/", name, ch)
    if ep.rewards is not None:
        _print_channel("", "reward", ep.rewards)
    if ep.terminations is not None:
        _print_channel("", "done", ep.terminations)
    if ep.truncations is not None:
        _print_channel("", "truncations", ep.truncations)

    # Print meta blocks that won't show up through channels
    meta_blocks = sorted(n for n in entry_by_name if n.startswith("meta/") or n.startswith("time/"))
    for block_name in meta_blocks:
        e = entry_by_name[block_name]
        is_comp = bool(e["flags"] & _BLOCK_FLAG_COMPRESSED)
        comp = "yes" if is_comp else "no"
        disk = f"{e['disk_size']:,}"
        print(
            f"{block_name:{col_w[0]}s}"
            f"  {'(meta)':{col_w[1]}s}"
            f"  {'':{col_w[2]}s}"
            f"  {comp:{col_w[3]}s}"
            f"  {disk:>{col_w[4]}s}"
        )

    return 0


# ---------------------------------------------------------------------------
# `wshard verify`
# ---------------------------------------------------------------------------

def cmd_verify(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if not path.exists():
        print(f"error: file not found: {path}", file=sys.stderr)
        return 1

    with open(path, "rb") as fh:
        raw = fh.read()

    # Parse header
    try:
        hdr = _read_header(raw)
    except ValueError as exc:
        print(f"FAIL  header: {exc}", file=sys.stderr)
        return 1

    print(f"W-SHARD version={hdr['version']} role=0x{hdr['role']:02x}  "
          f"entries={hdr['entry_count']}  file={len(raw):,} bytes")

    try:
        entries = _read_index_entries(raw, hdr)
        named   = _resolve_names(raw, hdr, entries)
    except Exception as exc:
        print(f"FAIL  index: {exc}", file=sys.stderr)
        return 1

    # CRC32C is on the *uncompressed* (logical) block bytes.
    # We must decompress to validate — or skip blocks with checksum==0.
    # The existing reader (load_wshard) validates CRC during decompression,
    # so delegate to it for the integrity path.  That means we need to
    # decompress each block here too.
    import crc32c as _crc32c

    # Lazy decompressors
    def _decompress(block_data: bytes, e: dict) -> bytes:
        flags    = e["flags"]
        disk_sz  = e["disk_size"]
        orig_sz  = int(e["orig_size"])

        if not (flags & _BLOCK_FLAG_COMPRESSED) or disk_sz == orig_sz:
            return block_data  # Uncompressed

        if flags & _BLOCK_FLAG_LZ4:
            import lz4.frame as _lz4f
            return _lz4f.decompress(block_data)
        else:
            # ZSTD (default compressed path)
            import zstandard as _zstd
            ctx = _zstd.ZstdDecompressor()
            return ctx.decompress(block_data, max_length=orig_sz)

    failures = 0
    checked  = 0

    for name, e in named:
        off      = int(e["data_offset"])
        disk_sz  = int(e["disk_size"])
        checksum = e["checksum"]

        block_disk = raw[off : off + disk_sz]

        # Decompress so we can hash the logical payload
        try:
            logical = _decompress(block_disk, e)
        except Exception as exc:
            print(f"FAIL  {name!r}: decompress error: {exc}")
            failures += 1
            checked  += 1
            continue

        if checksum == 0:
            # No checksum stored — skip CRC check but count it
            checked += 1
            continue

        actual = _crc32c.crc32c(logical)
        if actual != checksum:
            print(f"FAIL  {name!r}: CRC mismatch expected=0x{checksum:08x} got=0x{actual:08x}")
            failures += 1
        checked += 1

    if failures == 0:
        print(f"OK    {checked} block(s) verified — no errors")
    else:
        print(f"FAIL  {failures}/{checked} block(s) failed CRC check")

    # --strict: verify xxHash64 name hashes against index entries
    if args.strict:
        import xxhash
        hash_failures = 0
        for name, e in named:
            expected_hash = e["name_hash"]
            actual_hash   = xxhash.xxh64(name.encode("utf-8")).intdigest()
            if actual_hash != expected_hash:
                print(f"STRICT FAIL  {name!r}: name hash mismatch "
                      f"expected=0x{expected_hash:016x} got=0x{actual_hash:016x}")
                hash_failures += 1
        if hash_failures == 0:
            print(f"STRICT OK  all {len(named)} name hashes verified")
        else:
            failures += hash_failures
            print(f"STRICT FAIL  {hash_failures} name hash(es) invalid")

    return 0 if failures == 0 else 1


# ---------------------------------------------------------------------------
# `wshard convert`
# ---------------------------------------------------------------------------

_FORMAT_NAMES = {
    "wshard": "WSHARD",
    "shard":  "WSHARD",
    "dreamer": "DREAMER_V3",
    "dreamerv3": "DREAMER_V3",
    "npz": "DREAMER_V3",
    "tdmpc2": "TDMPC2",
    "tdmpc": "TDMPC2",
    "minari": "MINARI",
    "d4rl": "D4RL",
}


def _parse_format(name: str):
    """Parse a user-supplied format name into a wshard.Format member."""
    from wshard.types import Format
    key = name.lower().strip()
    canonical = _FORMAT_NAMES.get(key, key.upper())
    try:
        return Format(canonical.lower())
    except ValueError:
        pass
    # Try by enum name
    for member in Format:
        if member.name.upper() == canonical:
            return member
    raise ValueError(f"Unknown format {name!r}. "
                     f"Choices: wshard, dreamerv3/npz, tdmpc2, minari, d4rl")


_NOT_IMPLEMENTED_FORMATS = {"TDMPC2", "MINARI", "D4RL"}


def cmd_convert(args: argparse.Namespace) -> int:
    from wshard.convert import detect_format, load, save
    from wshard.types import Format

    in_path  = Path(args.input)
    out_path = Path(args.output)

    # Resolve formats
    try:
        in_fmt = _parse_format(args.input_format) if args.input_format else detect_format(in_path)
        out_fmt = _parse_format(args.output_format) if args.output_format else detect_format(out_path)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Check for unimplemented save targets
    if out_fmt.name in _NOT_IMPLEMENTED_FORMATS:
        print(
            f"not implemented: saving to {out_fmt.name} format is not yet supported.\n"
            f"Currently working save paths: wshard (.wshard), dreamerv3 (.npz).",
            file=sys.stderr,
        )
        return 2

    if in_fmt.name in _NOT_IMPLEMENTED_FORMATS:
        print(
            f"not implemented: loading from {in_fmt.name} format is not yet supported.\n"
            f"Currently working load paths: wshard (.wshard), dreamerv3 (.npz).",
            file=sys.stderr,
        )
        return 2

    if not in_path.exists():
        print(f"error: input file not found: {in_path}", file=sys.stderr)
        return 1

    try:
        ep = load(in_path, format=in_fmt)
    except NotImplementedError as exc:
        print(f"not implemented: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error loading {in_path}: {exc}", file=sys.stderr)
        return 1

    try:
        save(ep, out_path, format=out_fmt)
    except NotImplementedError as exc:
        print(f"not implemented: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error saving {out_path}: {exc}", file=sys.stderr)
        return 1

    print(f"converted  {in_path}  ->  {out_path}")
    print(f"           {in_fmt.name}  ->  {out_fmt.name}  ({ep.length} timesteps)")
    return 0


# ---------------------------------------------------------------------------
# `wshard export`
# ---------------------------------------------------------------------------

_FORMAT_EXT = {
    "WSHARD":    ".wshard",
    "DREAMER_V3": ".npz",
    "TDMPC2":    ".pt",
    "MINARI":    ".hdf5",
    "D4RL":      ".hdf5",
}


def cmd_export(args: argparse.Namespace) -> int:
    from wshard.convert import detect_format, load, save
    from wshard.types import Format

    in_path = Path(args.path)
    if not in_path.exists():
        print(f"error: file not found: {in_path}", file=sys.stderr)
        return 1

    try:
        out_fmt = _parse_format(args.format)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if out_fmt.name in _NOT_IMPLEMENTED_FORMATS:
        print(
            f"not implemented: exporting to {out_fmt.name} is not yet supported.\n"
            f"Currently working export formats: wshard, dreamerv3/npz.",
            file=sys.stderr,
        )
        return 2

    if args.output:
        out_path = Path(args.output)
    else:
        ext = _FORMAT_EXT.get(out_fmt.name, f".{out_fmt.value}")
        out_path = in_path.with_suffix(ext)

    try:
        ep = load(in_path, format=Format.WSHARD)
    except Exception as exc:
        print(f"error loading {in_path}: {exc}", file=sys.stderr)
        return 1

    try:
        save(ep, out_path, format=out_fmt)
    except NotImplementedError as exc:
        print(f"not implemented: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error saving {out_path}: {exc}", file=sys.stderr)
        return 1

    print(f"exported  {in_path}  ->  {out_path}  ({out_fmt.name})")
    return 0


# ---------------------------------------------------------------------------
# `wshard doctor`
# ---------------------------------------------------------------------------

_REQUIRED_DEPS = ["numpy", "crc32c", "xxhash", "zstandard", "lz4"]
_OPTIONAL_DEPS = ["ml_dtypes", "h5py", "torch", "pyarrow", "huggingface_hub"]

# Mapping from install name -> importable name (where they differ)
_IMPORT_NAME = {
    "zstandard": "zstandard",
    "lz4":       "lz4",
    "xxhash":    "xxhash",
    "crc32c":    "crc32c",
    "ml_dtypes": "ml_dtypes",
    "huggingface_hub": "huggingface_hub",
}


def _get_version(pkg: str) -> str | None:
    """Return installed version string, or None if not importable."""
    try:
        from importlib.metadata import version
        return version(pkg)
    except Exception:
        pass
    # Try import-based version attr
    import_name = _IMPORT_NAME.get(pkg, pkg)
    try:
        mod = __import__(import_name)
        return getattr(mod, "__version__", "installed (version unknown)")
    except ImportError:
        return None


def cmd_doctor(args: argparse.Namespace) -> int:
    import platform

    # wshard version + location
    try:
        import wshard as _wshard
        ws_version = _wshard.__version__
        ws_file    = getattr(_wshard, "__file__", "<unknown>")
    except ImportError:
        ws_version = "<import error>"
        ws_file    = "<unknown>"

    print(f"wshard {ws_version}")
    print(f"  loaded from : {ws_file}")
    print(f"  python      : {sys.version}")
    print(f"  platform    : {platform.platform()}")
    print()

    any_missing = False

    print("Required dependencies:")
    for pkg in _REQUIRED_DEPS:
        ver = _get_version(pkg)
        if ver is None:
            print(f"  {pkg:<22s}  {red('MISSING')}")
            any_missing = True
        else:
            print(f"  {pkg:<22s}  {green(ver)}")

    print()
    print("Optional dependencies:")
    for pkg in _OPTIONAL_DEPS:
        ver = _get_version(pkg)
        if ver is None:
            print(f"  {pkg:<22s}  not installed")
        else:
            print(f"  {pkg:<22s}  {ver}")

    if any_missing:
        print()
        print(red("Some required dependencies are missing. Install them with:"))
        missing = [p for p in _REQUIRED_DEPS if _get_version(p) is None]
        print(f"  pip install {' '.join(missing)}")
        return 1

    return 0


# ---------------------------------------------------------------------------
# Argument parser construction
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wshard",
        description="W-SHARD episode format toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Subcommands:\n"
            "  inspect   Print episode metadata and block layout\n"
            "  verify    Validate CRC32C integrity of every block\n"
            "  convert   Convert between episode formats\n"
            "  export    Export a .wshard to another format (sugar over convert)\n"
            "  doctor    Show package version and dependency status\n"
        ),
    )
    parser.add_argument(
        "--version", action="version",
        version=_get_wshard_version(),
    )

    sub = parser.add_subparsers(dest="subcommand", metavar="<subcommand>")

    # ---- inspect ----
    p_inspect = sub.add_parser(
        "inspect",
        help="Print episode metadata and block layout",
        description=(
            "Load a .wshard file and print episode metadata (id, env_id, length) "
            "followed by a table of every block with dtype, shape, compression flag, "
            "and on-disk byte size."
        ),
    )
    p_inspect.add_argument("path", metavar="<path.wshard>", help="W-SHARD file to inspect")

    # ---- verify ----
    p_verify = sub.add_parser(
        "verify",
        help="Validate file integrity (CRC32C of every block)",
        description=(
            "Parse the W-SHARD header and index, then verify the CRC32C checksum of "
            "every block. CRC is computed on the uncompressed (logical) payload bytes, "
            "so each block must be decompressed to validate. Exits 0 on success, 1 on "
            "any failure.\n\n"
            "Note: blocks with a stored checksum of 0x00000000 are skipped (no checksum "
            "was recorded for that block at write time)."
        ),
    )
    p_verify.add_argument("path", metavar="<path.wshard>", help="W-SHARD file to verify")
    p_verify.add_argument(
        "--strict", action="store_true",
        help="Also re-hash all xxHash64 block-name entries against the stored index values",
    )

    # ---- convert ----
    p_convert = sub.add_parser(
        "convert",
        help="Convert between episode formats",
        description=(
            "Convert an episode file from one format to another. Format is auto-detected "
            "from the file extension unless overridden.\n\n"
            "Fully supported round-trips: wshard <-> dreamerv3 (NPZ).\n"
            "Unimplemented save targets (tdmpc2, minari, d4rl): print 'not implemented' "
            "and exit with code 2 instead of crashing."
        ),
    )
    p_convert.add_argument("input",  metavar="<input>",  help="Input file path")
    p_convert.add_argument("output", metavar="<output>", help="Output file path")
    p_convert.add_argument(
        "--input-format", metavar="FMT", default=None,
        dest="input_format",
        help="Override input format (wshard, dreamerv3, npz, tdmpc2, minari, d4rl)",
    )
    p_convert.add_argument(
        "--output-format", metavar="FMT", default=None,
        dest="output_format",
        help="Override output format (wshard, dreamerv3, npz, tdmpc2, minari, d4rl)",
    )

    # ---- export ----
    p_export = sub.add_parser(
        "export",
        help="Export a .wshard file to another format",
        description=(
            "Sugar over 'wshard convert' with the input fixed to W-SHARD format. "
            "Default output path is <input-stem>.<ext-for-format>.\n\n"
            "Currently working export formats: wshard, dreamerv3 / npz.\n"
            "Others (tdmpc2, minari, d4rl) print 'not implemented' and exit 2."
        ),
    )
    p_export.add_argument("path", metavar="<path.wshard>", help="Source W-SHARD file")
    p_export.add_argument(
        "--format", "-f", metavar="FMT", required=True,
        help="Target format: wshard, dreamerv3/npz, tdmpc2, minari, d4rl",
    )
    p_export.add_argument(
        "-o", "--output", metavar="OUTPUT", default=None,
        help="Output file path (default: <stem>.<ext-for-format>)",
    )

    # ---- doctor ----
    p_doctor = sub.add_parser(
        "doctor",
        help="Show package version and dependency status",
        description=(
            "Print wshard package version, Python version, and the install status of "
            "all required and optional dependencies. Exits 0 if all required deps are "
            "present, 1 if any are missing."
        ),
    )

    return parser


def _get_wshard_version() -> str:
    try:
        from importlib.metadata import version
        v = version("wshard")
        return f"wshard {v}"
    except Exception:
        try:
            import wshard as _w
            return f"wshard {_w.__version__}"
        except Exception:
            return "wshard (unknown version)"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.subcommand is None:
        parser.print_help()
        sys.exit(2)

    dispatch = {
        "inspect": cmd_inspect,
        "verify":  cmd_verify,
        "convert": cmd_convert,
        "export":  cmd_export,
        "doctor":  cmd_doctor,
    }

    handler = dispatch.get(args.subcommand)
    if handler is None:
        print(f"error: unknown subcommand {args.subcommand!r}", file=sys.stderr)
        parser.print_help(sys.stderr)
        sys.exit(2)

    sys.exit(handler(args))


if __name__ == "__main__":
    main()
