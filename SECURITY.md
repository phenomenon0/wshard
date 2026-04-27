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

## Threat model

CRC32C is **not cryptographic authentication**. A motivated attacker can
forge a file whose CRC matches the corruption they introduce. Treat WShard
files from untrusted sources as untrusted bytes:

- Verify integrity with a signature, hash, or trusted transport before reading.
- Use the malformed-file test suite (`py/tests/test_malformed.py`) as a
  reference for the kinds of inputs the readers are expected to reject.
- Run readers in a sandbox if processing files from a public pipeline.

## Reporting issues

Please report security issues by opening a GitHub issue at
https://github.com/phenomenon0/wshard/issues. For sensitive disclosures,
contact the repository owner directly via the email on their GitHub profile.

We do not currently operate a security bug bounty program.
