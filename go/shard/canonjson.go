// canonjson.go — the sliver of glyph canonical JSON that meta/identity and
// meta/provenance need.
//
// Both blocks are hashed, so their bytes have to be the canonical JSON of
// their value (glyph SPEC-CANON.md §1-§3) and not merely valid JSON. Go's
// encoding/json is close but not equal, and every difference is a fingerprint
// that disagrees with Python's for the same episode:
//
//	U+2028 / U+2029  encoding/json escapes them; the canon writes them raw
//	backspace, formfeed  encoding/json writes  / , canon \b / \f
//	invalid UTF-8    encoding/json substitutes U+FFFD, canon passes bytes through
//
// Block names come from channel ids and the provenance source map is
// free-form, so all three are reachable from ordinary input.
//
// ponytail: strings, string maps and small ints -- the only shapes these two
// blocks hold. Depend on glyph/go's CanonJSON if a third block ever needs
// floats, nesting or lists.
package shard

import (
	"bytes"
	"fmt"
	"sort"
)

// writeCanonString escapes per RFC 8785 §3.2.2.2, matching glyph's
// writeJSONString byte for byte: short forms for " \ \b \f \n \r \t, \u00xx
// for the other C0 controls, every other byte raw.
func writeCanonString(b *bytes.Buffer, s string) {
	b.WriteByte('"')
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch c {
		case '"':
			b.WriteString(`\"`)
		case '\\':
			b.WriteString(`\\`)
		case '\b':
			b.WriteString(`\b`)
		case '\f':
			b.WriteString(`\f`)
		case '\n':
			b.WriteString(`\n`)
		case '\r':
			b.WriteString(`\r`)
		case '\t':
			b.WriteString(`\t`)
		default:
			if c < 0x20 {
				fmt.Fprintf(b, `\u%04x`, c)
			} else {
				b.WriteByte(c)
			}
		}
	}
	b.WriteByte('"')
}

// writeCanonStringMap writes m as a JSON object with keys sorted by UTF-8
// bytes, which for Go strings is plain < . No duplicate check: a Go map cannot
// hold one.
func writeCanonStringMap(b *bytes.Buffer, m map[string]string) {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	b.WriteByte('{')
	for i, k := range keys {
		if i > 0 {
			b.WriteByte(',')
		}
		writeCanonString(b, k)
		b.WriteByte(':')
		writeCanonString(b, m[k])
	}
	b.WriteByte('}')
}
