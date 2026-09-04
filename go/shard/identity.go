package shard

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
)

// IdentityBlock commits to every other entry's uncompressed bytes (glyph
// SPEC-CANON.md §4): {"entries":{name:sha256hex,…},"leaf":"sha256","v":1} as
// canonical JSON, so sha256(block) is the glyph fingerprint of that value and
// survives recompression or re-indexing. CreateWShard writes it last.
// ponytail: flat leaf map; switch to an RFC 6962 tree when a dataset needs
// O(log n) proofs. WShardStreamWriter gets one too: EndEpisode writes through
// CreateWShard.
const IdentityBlock = "meta/identity"

func leafHex(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

// identityBlockBytes renders the leaf map as canonical JSON. Written out by
// hand rather than through encoding/json, which escapes more than the canon
// does -- see canonjson.go. The three top-level keys are already in UTF-8 byte
// order (entries < leaf < v).
func identityBlockBytes(leaves map[string]string) ([]byte, error) {
	var b bytes.Buffer
	b.WriteString(`{"entries":`)
	writeCanonStringMap(&b, leaves)
	b.WriteString(`,"leaf":"sha256","v":1}`)
	return b.Bytes(), nil
}

// EpisodeIdentity returns the 64-hex identity of a W-SHARD file: sha256 of
// its meta/identity entry. Trusts the entry as written; VerifyIdentity
// re-hashes every entry.
func EpisodeIdentity(path string) (string, error) {
	r, err := OpenShard(path)
	if err != nil {
		return "", err
	}
	defer r.Close()
	ident, err := r.ReadEntryByName(IdentityBlock)
	if err != nil {
		return "", fmt.Errorf("wshard: no %s (written before identity existed?): %w", IdentityBlock, err)
	}
	return leafHex(ident), nil
}

// VerifyIdentityAgainst is VerifyIdentity plus the check that makes it mean
// something against an editor: expected is the 64-hex identity obtained out of
// band -- a signature, a manifest, a git commit, the prev_identity of the next
// chunk. VerifyIdentity alone proves a file is internally consistent, which an
// attacker who rewrote a block and resealed meta/identity also satisfies.
func VerifyIdentityAgainst(path, expected string) (string, error) {
	if len(expected) != 64 {
		return "", fmt.Errorf("wshard: expected must be 64 hex characters, got %d", len(expected))
	}
	if _, err := hex.DecodeString(expected); err != nil {
		return "", fmt.Errorf("wshard: expected is not hex: %w", err)
	}
	got, err := VerifyIdentity(path)
	if err != nil {
		return "", err
	}
	if got != expected {
		return "", fmt.Errorf(
			"wshard: identity is %s, expected %s: the file is internally consistent "+
				"but is not the one that identity names", got, expected)
	}
	return got, nil
}

// VerifyIdentity re-hashes every entry, checks the result against
// meta/identity and returns the identity. CRC32C only proves an entry matches
// its own index slot, which whoever edits the file can rewrite; this proves
// every entry matches what the identity committed to. The error names the
// first entry that differs.
//
// This proves the file is internally consistent -- not bit-rotted, truncated or
// partially written, across recompression, which CRC32C cannot do. It proves
// nothing about origin. For that use VerifyIdentityAgainst with a 64-hex you
// obtained out of band.
func VerifyIdentity(path string) (string, error) {
	r, err := OpenShard(path)
	if err != nil {
		return "", err
	}
	defer r.Close()
	leaves := map[string]string{}
	var ident []byte
	for i := 0; i < r.EntryCount(); i++ {
		data, err := r.ReadEntry(i)
		if err != nil {
			return "", err
		}
		if name := r.EntryName(i); name == IdentityBlock {
			ident = data
		} else {
			leaves[name] = leafHex(data)
		}
	}
	if ident == nil {
		return "", fmt.Errorf("wshard: no %s (written before identity existed?)", IdentityBlock)
	}
	want, err := identityBlockBytes(leaves)
	if err != nil {
		return "", err
	}
	if bytes.Equal(ident, want) {
		return leafHex(ident), nil
	}
	var doc struct {
		Entries map[string]string `json:"entries"`
	}
	_ = json.Unmarshal(ident, &doc)
	names := make([]string, 0, len(leaves)+len(doc.Entries))
	for n := range leaves {
		names = append(names, n)
	}
	for n := range doc.Entries {
		if _, ok := leaves[n]; !ok {
			names = append(names, n)
		}
	}
	sort.Strings(names)
	for _, n := range names {
		if doc.Entries[n] != leaves[n] {
			return "", fmt.Errorf("wshard: identity mismatch at %s: committed %q, file has %q", n, doc.Entries[n], leaves[n])
		}
	}
	return "", fmt.Errorf("wshard: %s is not canonical", IdentityBlock)
}
