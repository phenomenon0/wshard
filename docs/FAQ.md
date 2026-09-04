# WShard — Frequently Asked Questions

---

## Why another file format?

There are good formats for adjacent problems — HDF5 for scientific arrays, Parquet for columnar tables, MCAP for timestamped robot logs, LeRobot for hub-distributed datasets. None of them are focused specifically on **self-contained tensor episodes plus named prediction lanes**. WShard is a narrow answer to a narrow question: one file, one episode, independently-addressable named blocks, per-block compression. If your problem fits an existing format better, use that format.

See the "What WShard is not" table in [README.md](../README.md) for a quick decision guide.

---

## Should I migrate from HDF5 / NPZ / Parquet?

Probably not, unless you specifically need named tensor blocks per episode file, an episode identity that survives recompression, or cross-language access (Go + Python + TypeScript) with no code-generation step. If HDF5 is working for you today, the migration cost isn't worth it. If NPZ is fine, keep using it.

Where WShard tends to help: collecting data live from robots (streaming writer), running training pipelines across multiple languages, or storing model predictions alongside ground truth in the same file.

See [WHY_NOT_HDF5.md](WHY_NOT_HDF5.md) for the detailed comparison with HDF5.

---

## Is this production-ready?

No. WShard is **beta**. The on-disk format and core API are stable enough to use, but we are still looking for external users to surface real-world breakage before declaring 1.0. The Show HN launch is intentional — it is a bug-finding pass.

Known limitations, verbatim from the README:

- **No native video block type.** Camera data is stored as raw tensors. H.264 / H.265 / AV1 block types are future work.
- **No Arrow / Polars / DuckDB integration** yet.
- **No formal schema registry** — block-name conventions are advisory.
- **Cloud / object-store behavior** has not been stress-tested.
- **Few external users** so far. The launch is the bug-finding pass.

---

## How big can a .wshard file get?

The format imposes the following hard limits, taken directly from `go/shard/shard_format.go`:

| Limit | Value | Field |
|-------|-------|-------|
| Max entry count | 10,000,000 blocks | `MaxEntryCount` |
| Max index section | 1 GB | `MaxIndexSize` |
| Max string table | 100 MB | `MaxStringTableSize` |

In practice you will never reach those limits in a single episode. A file is a single seek-addressable region, so random reads are fast but the whole file must live on one storage unit. For episodes larger than about 10 GB, use the chunked-episode API (`ChunkedEpisodeWriter`) to split across multiple files with a manifest shard tracking continuity. Cloud and object-store behavior at large scale has not been stress-tested.

---

## Does WShard support video natively?

No. Camera data is stored as raw tensor blocks (`signal/rgb`, shape `[T, H, W, 3]`). H.264, H.265, and AV1 block types are listed as future work; the per-block architecture supports adding them (as new content type values), but no codec is implemented today.

For now, the recommended pattern for camera-heavy pipelines is an external MP4
sidecar: write the video separately and store only the decoded frames (or a
downsampled version) inside the `.wshard` file if you need in-file access.

**Keep the sidecar under the seal.** A path alone breaks the format's main
claim — the file is no longer self-contained and its identity says nothing about
the video. Put the video's own sha256 in `provenance.source`, which is a
free-form string map written to `meta/provenance`, which `meta/identity` commits
to. The seal then reaches the video transitively: file identity → `meta/provenance`
→ video sha256 → video bytes. Change one frame and the episode identity moves.

```python
import hashlib, pathlib
from wshard.types import Provenance

video = pathlib.Path("cam.mp4")
video.write_bytes(encoded_h264)   # whatever your encoder produced
ep.provenance = Provenance(
    run_id="run-42",
    source={"video_path": video.name,
            "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest()},
)
save_wshard(ep, "episode.wshard")
```

Two files still have to travel together, which is a real cost against the
one-file pitch — but "self-contained" as a *property* survives even when
self-contained as a *file layout* does not. Note that `Episode.metadata` is
**not** the place for this: it is not written to the file and does not affect
the identity.

---

## Can I stream / append to a file mid-episode?

You can feed a file timestep by timestep. You cannot read back a partially
written one. `WShardStreamWriter` is a live-recording writer, not an append log:

1. `begin_episode()` opens `episode.wshard.partial` and reserves header + index space
2. `write_timestep()` validates the step and buffers it in memory
3. `end_episode()` writes every buffer, seals the file with `meta/identity`, writes
   the real header and index, `fsync`s, and atomically renames to `episode.wshard`
4. If the process crashes first, no `episode.wshard` exists and the orphan
   `.partial` is by name not a readable episode

So a crash loses the whole episode, cleanly, rather than leaving a corrupt file.
Nothing is durable before `end_episode()`. The constraint is the index: each
block is one `(offset, size)` extent, so a block can be written exactly once —
flushing twice would interleave the channels on disk while each extent grew over
its neighbours' bytes, and the result would pass CRC32C while holding another
channel's values. All three writers flush only at the end, and all three reject
a flush-interval argument rather than accepting one that does nothing.

For collection runs that must survive a crash, write chunk files: each chunk is a
complete sealed episode, and `prev_identity` links chunk N to the exact bytes of
chunk N-1 (`ChunkedEpisodeWriter`, `validate_chunk_chain`).

From the streaming benchmark (`bench/bench_streaming.py`, T=10,000 episode):
**24.0 µs per step** median, 1.08 MiB peak RSS. That is safe for 1 kHz robot
control loops (40× headroom); the memory is the whole episode, bounded by
`max_timesteps`.

---

## What about parallel reads?

Each block is independently addressable by file offset, so **multi-threaded reads of different blocks are safe** — threads can each seek and read different blocks simultaneously on a memory-mapped file. The Go reader supports this today; Python can use `np.frombuffer` on an mmap'd file.

**Multi-process append is not supported.** WShard uses a single-writer model. If you need concurrent writes, write separate `.wshard` files per process and merge them later.

---

## What if a file is corrupted?

Every block has a CRC32C (Castagnoli) checksum stored in its index entry, computed over the uncompressed data. The reader verifies it on every block read and raises an error on mismatch.

To validate a whole file without reading all block data into memory:

```bash
wshard verify path/to/episode.wshard
```

`wshard verify` checks CRC32C only. CRC32C detects accidental corruption (bit
rot, truncation, partial writes) and nothing else: whoever rewrites a block can
recompute its checksum in the same pass, so it does not detect editing.

For that, use the identity. `meta/identity` holds a sha256 of every other block's
uncompressed bytes, and `verify_identity(path)` (Python) / `VerifyIdentity(path)`
(Go) re-hash the file against it and name the first block that differs. The
identity itself is `sha256` of that block — 64 hex characters, unchanged by
recompression, different for any content change.

Which question you asked depends on whether you pass the expected value:

```python
verify_identity(path)                          # internally consistent? (bit rot, truncation)
verify_identity(path, expected=known_64_hex)   # and is it the file that identity names?
```
```go
VerifyIdentity(path)                       // same two calls in Go
VerifyIdentityAgainst(path, known64Hex)
```

A recipient who knows the expected identity out of band detects an edited file;
a recipient who does not still only knows the file is self-consistent. It is a
fingerprint, not a signature — for files from untrusted sources, sign the
identity.

---

## Cross-language byte-identical?

Yes, verified. Go is the reference implementation. Python and TypeScript read the same binary files and assert byte-level correctness via golden-file tests:

- Reference files are in `golden/` (`simple_episode.wshard`, `dtype_zoo.wshard`, `per_block_compressed.wshard`, `omen_uncert.wshard`, `multimodal.wshard`, `latent_action.wshard`)
- Reference hash values are committed in `golden/golden_hashes.json`
- CI verifies these on every push

Identity is Go + Python. Every golden file's `meta/identity` is re-derived by
both and matched against `identity_<file>` in the same JSON. The TypeScript
reader parses these files but does not compute identity yet.

Reference values:
```
CRC32C("hello")          = 0x9a71bb4c
xxHash64("signal/obs")   = 0x86f8c8413116a0ae
```

---

## Why xxHash64 for block names?

Speed. The index at the front of the file stores an 8-byte xxHash64 hash per block. Finding a named block requires reading the 64-byte header and scanning hash values in the index — for a 50-block file, that is roughly 64 + 50×8 = 464 bytes, with no string-table parse. xxHash64 is fast to compute on all three platforms and wide enough (64 bits) that collisions at realistic block counts (tens to hundreds per episode) are not a practical concern. It is not a cryptographic hash.

---

## Can I add my own block types / namespaces?

Yes. Block names are arbitrary UTF-8 strings with `/` as a hierarchy separator. The existing namespaces (`meta/`, `signal/`, `action/`, `omen/`, `uncert/`, `residual/`, `time/`) are conventions that the Python reader uses to route blocks to the correct `Episode` fields. They are advisory, not enforced by the format.

You can use `my_team/sensor_x` or `experiment/ablation_1/ctrl` freely. Reader routing will treat unrecognized prefixes as raw blocks, accessible via the lower-level index API.

---

## What's the relationship to DeepData / Mosh / ColumnShard?

They all use the same underlying **Shard** binary container (magic `SHRD`, v2 header). The `role` byte in the header distinguishes them. WShard is role `0x05`. Other roles (Mosh, ColumnShard, etc.) are separate formats for separate purposes, not published in this repository. WShard is the only format released here.

---

## Why not just Parquet?

Parquet has no native multi-dimensional array type. Tensor data must be either serialized to bytes (opaque blob per row) or exploded into flat scalar columns — both are awkward for training loops that want `np.frombuffer` directly into an aligned array.

From the Parquet benchmark (`bench/bench_parquet.py`, same 20 MB workload, pyarrow 22.0.0): **WShard reads 2.3× faster** than Parquet on the same payload (wshard-none: 6.0 ms, parquet-zstd: 13.8 ms). Parquet write is faster (11.8 ms vs 43 ms for wshard-none) because pyarrow's column-write path is optimized C++, while WShard's Python writer is pure struct-packing plus one sha256 pass to seal the file.

Also, Parquet is row-group-oriented. For accessing one block out of a large file, WShard's flat index wins: `BenchmarkPartialReadCtrl` reads a 28 KB block from a 50 MB file in **~12 µs** (Go).

---

## Why not just HDF5?

See [WHY_NOT_HDF5.md](WHY_NOT_HDF5.md) for the full comparison.

Short version: HDF5 is a better choice for scientific data with compound types, virtual datasets, or parallel MPI writes. WShard is a simpler answer for episode files where you need a tiny self-contained spec, a content identity that survives recompression, and cross-language readers without a libhdf5 dependency.

---

## Where do I report bugs?

Open a GitHub issue. For security issues, follow [SECURITY.md](../SECURITY.md). The most useful reports right now are real-world breakage — malformed files, wrong values on roundtrip, format conversion failures, or benchmarks that don't match what you see on your hardware.

---

## Things we're still figuring out

These are open questions, not planned features:

1. **Video block type.** Native H.264/H.265/AV1 blocks would significantly reduce storage for camera-heavy episodes. The architecture supports it (new content type constant, codec implementation), but it is not prioritized until there is concrete user demand.
2. **Per-channel compression.** The format stores a compression flag per block, but the shipped writers take one codec for the whole file and decide per block only whether to apply it (skipped under 64 bytes, or when the compressed form is not smaller). Choosing zstd for `signal/rgb` and none for `signal/depth` in one file is possible in the container and has no API in front of it. Level is `CompressionLevel.FASTEST` / `DEFAULT` / `BEST`, per file, not per channel.
3. **Batch-shard format.** A multi-episode container (many `.wshard` episodes in one file) would help with dataset distribution. The ColumnShard role exists in the Shard family but is not part of this repo.
4. **Arrow / Polars / DuckDB bridge.** An Arrow reader that exposes wshard channels as Arrow arrays would enable zero-copy interop with the columnar analytics ecosystem. Architecturally straightforward, not yet implemented.
5. **Schema registry.** Block-name conventions are advisory. A formal schema definition (think: episode "type" declarations, required fields, expected dtypes) would enable validation and code generation, at the cost of flexibility.
