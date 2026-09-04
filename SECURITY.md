# Security

WShard is a data format, not an execution format. Readers must not execute
code from `.wshard` files.

## Current protections

- **CRC32C block integrity checks** on every data block (Castagnoli polynomial).
- **Bounded entry count** in the header — readers reject files with implausible
  index sizes before allocating buffers.
- **Bounded string-table size** — name strings are validated against the file's
  stated string-table extent.
- **Bounded decompressed block size** — readers reject blocks whose declared
  uncompressed size would exceed reasonable memory limits.
- **Dtype and shape validation** — only the 13 documented dtypes are accepted;
  shape products are checked against block size to prevent overflow.
- **Content identity** — `meta/identity` holds a sha256 of every other block's
  uncompressed bytes, and `verify_identity(path)` (Python) / `VerifyIdentity(path)`
  (Go) re-hash the file against it. Pass the expected 64-hex to check origin too:
  `verify_identity(path, expected=...)` / `VerifyIdentityAgainst(path, expected)`.
  Not implemented in the TypeScript reader.

## Threat model

CRC32C is **not cryptographic authentication**, and neither is the identity
block. What each one actually buys is different, and the difference matters:

| | Detects bit rot / truncation | Detects deliberate editing |
|---|---|---|
| CRC32C per block | yes | no — the editor recomputes it in the same pass |
| `meta/identity`, recipient knows the expected 64-hex | yes | yes |
| `meta/identity`, recipient does not | yes | no — the editor reseals the file |

The identity is a fingerprint, not a signature. `verify_identity` proves a file
is internally consistent with what its own identity block claims; it cannot
prove *who* sealed it. An attacker who can rewrite the file can rewrite
`meta/identity` and `meta/provenance` with it. The identity becomes an
authenticity check only once the expected value reaches the recipient over a
channel the attacker does not control — a signature, a manifest, a git commit, a
`prev_identity` in the next chunk of the same run.

Treat WShard files from untrusted sources as untrusted bytes:

- Compare against an expected value you obtained out of band, or verify a
  signature over it. Reading and re-verifying a file's own identity block proves
  nothing about its origin. The API takes the expected value directly, and that
  is the call to reach for:

  ```python
  verify_identity(path, expected=known_64_hex)   # raises if the file is not that one
  ```
  ```go
  id, err := VerifyIdentityAgainst(path, known64Hex)
  ```

  The no-argument forms (`verify_identity(path)`, `VerifyIdentity(path)`) are
  the bit-rot check: they prove the file is internally consistent across
  recompression, which CRC32C cannot do, and nothing more.
- Use the malformed-file test suite (`py/tests/test_malformed.py`) as a
  reference for the kinds of inputs the readers are expected to reject.
- Run readers in a sandbox if processing files from a public pipeline.
- Note that `wshard verify` checks CRC32C only. It does not check
  `meta/identity`; use `verify_identity` / `VerifyIdentity` for that.

## Reporting issues

Please report security issues by opening a GitHub issue at
https://github.com/phenomenon0/wshard/issues. For sensitive disclosures,
contact the repository owner directly via the email on their GitHub profile.

We do not currently operate a security bug bounty program.
