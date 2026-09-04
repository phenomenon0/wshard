# Roadmap — considered, not committed

What came out of an adversarial review of the format on 2026-09-03: attack it,
then defend it, then keep only what survived both passes. Everything below
survived. Nothing here is scheduled; this is the list to argue with, and each
item says what it closes and roughly what it costs.

Four items from that review are already done and are not repeated here: the
flush-interval argument now fails instead of being ignored (it was silently
corrupting TypeScript files), `verify_identity` takes the expected 64-hex,
`docs/FAQ.md` documents bringing a sidecar under the seal, and every ```python
block in the live docs is executed by `py/tests/test_docs.py`.

---

## The core held

Worth stating, because it is what the rest of this list is relative to. Two
adversarial passes did not move the flat index, the per-block addressing, or the
recompression-invariant identity. What is exposed is packaging: documentation,
missing evidence, and gaps between what the container supports and what the
writers expose.

---

## 1. Per-block compression API

**Size: 1–2 days** (Python + Go).

The container stores a compression flag per block. The shipped writers take one
codec for the whole file and decide per block only whether to apply it. So the
structural property the format is built on — blocks are independent — is only
half-exercised. Choosing zstd for `signal/rgb` and none for `signal/depth` in
one file is expressible on disk and has no API in front of it.

Cheap and sharp to test: identity is invariant across codec choice, so a mixed-codec
file and a single-codec file of the same episode must have the same identity.

## 2. TypeScript identity

**Size: 1 day** if glyph's JS canon implementation is reusable, **2 days**
writing RFC 8785 escaping from scratch.

`meta/identity` is the headline feature and it exists in two of three languages.
TypeScript parses sealed files but computes nothing. Needs canonical JSON,
sha256 leaves over uncompressed block bytes, `episodeIdentity` / `verifyIdentity`,
and golden-file tests against the committed `identity_<file>` values.

Until this lands, "three implementations" is true of the format and not of the
feature the format leads with.

## 3. Object-store benchmark

**Size: 2–3 days** including the harness.

This is the only survivor where the *claim* is unevidenced rather than merely
incomplete. Every number in the repo is single-machine local disk. Real
datasets live in object stores, and that is exactly where a 64-byte header plus
a flat index should beat a format that has to page through metadata — one range
GET to find a block, one to read it, versus downloading the file.

Needs MinIO locally for iteration plus at least one real S3 run for credibility,
and a comparison against HDF5 and Parquet on the same store. Until it exists,
the partial-read advantage is a claim about laptops.

## 4. Arrow bridge (Python, read-only first)

**Size: 3–5 days.**

The cheapest lever on ecosystem isolation. Blocks are already contiguous,
aligned, typed and known-shape, which is an Arrow `FixedSizeList` buffer — the
copy path is nearly free. Landing it gives polars and duckdb for nothing.

Fiddly parts are bf16 and mapping shape to nesting, not the zero-copy itself.

## 5. H.264 bitstream block type

**Size: 2–3 days.**

Not a decoder — a content-type constant plus store/retrieve of an encoded
bitstream as opaque bytes. That puts camera data inside the file and inside the
seal without pulling a codec into the format library.

The workaround documented in the FAQ (sidecar plus its sha256 in
`provenance.source`) keeps the seal intact today, but two files still have to
travel together, which is a real cost against the one-file pitch.

**Explicitly not doing:** in-library H.264 *decode* across Go, Python and
TypeScript. Weeks of work, drags ffmpeg/libav into all three, for something the
caller's pipeline already has.

## 6. One-way importer (LeRobot or HDF5 → wshard)

**Size: 2–3 days.**

The only lever on switching cost that is engineering rather than marketing.

## 7. `Episode.metadata` is silently dropped

**Size: unscoped — decide the semantics first.**

Found while documenting the sidecar pattern. `Episode.metadata` is a public
`Dict[str, Any]`, documented in the class docstring as "format-specific
auxiliary data", and it is never written: it does not round-trip and does not
affect the identity. Callers putting anything there are losing it silently.

Three options, and the choice is a design decision, not a fix:

1. Write it into `meta/episode`, so it rides under the seal. Changes the
   identity of any file that sets it, and needs the canonical-JSON treatment
   because arbitrary values have to hash the same way in both languages.
2. Raise on a non-empty `metadata` on save, the way `flush_interval` now does.
3. Delete the field.

Option 1 is the useful one and the most work; option 2 is honest and cheap.
Not silently dropping user data is the only part that is not optional.

---

## Not a ticket

**Network effects.** Zero external users is the largest real risk to the format
and no amount of engineering closes it directly. Items 4 and 6 are the only two
here that touch it at all.

**Bus factor.** Raised and then withdrawn: the mitigation is the spec itself.
Two pages, a 64-byte header, three independent implementations, golden files. If
the maintainer disappears, someone writes a reader in an afternoon. HDF5 cannot
say that, and it is a reason to keep the format small rather than a problem to
solve.
