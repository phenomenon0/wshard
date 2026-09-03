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
// O(log n) proofs. ShardStreamWriter writes no identity yet.
const IdentityBlock = "meta/identity"

func leafHex(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

// identityBlockBytes renders the leaf map as canonical JSON. encoding/json
// with sorted map keys and HTML escaping off is byte-identical to glyph
// CanonJSON for this shape (ASCII names, hex strings, one small int).
// ponytail: import glyph CanonJSON if entry names ever carry U+2028/9 or C0
// controls.
func identityBlockBytes(leaves map[string]string) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	err := enc.Encode(map[string]any{"v": 1, "leaf": "sha256", "entries": leaves})
	return bytes.TrimSuffix(buf.Bytes(), []byte("\n")), err
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

// VerifyIdentity re-hashes every entry, checks the result against
// meta/identity and returns the identity. CRC32C only proves an entry matches
// its own index slot, which whoever edits the file can rewrite; this proves
// every entry matches what the identity committed to. The error names the
// first entry that differs.
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
