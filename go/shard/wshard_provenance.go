package shard

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
)

// ProvenanceBlock records where an episode came from: which run and epoch
// produced it, which sequence range it covers, the state fingerprints it
// starts and ends on, and the identity of the episode before it. Canonical
// JSON like IdentityBlock, so sha256 of the entry is the glyph fingerprint of
// the value. CreateWShard writes it immediately before IdentityBlock, so the
// identity commits to it -- editing provenance after the fact breaks
// VerifyIdentity.
const ProvenanceBlock = "meta/provenance"

// WShardProvenance is the decoded ProvenanceBlock. PrevIdentity is the
// EpisodeIdentity of the preceding episode of the same run, which is what
// turns a set of chunks into a chain; empty means first, or unchained.
//
// Source carries free-form producer facts (git_commit, schema_fp, whatever a
// later producer needs). A string map rather than a struct so Go and Python
// sort its keys the same way without anyone maintaining a field order.
type WShardProvenance struct {
	RunID        string
	Epoch        int64
	FirstSeq     int64
	LastSeq      int64
	StartState   string // 64-hex glyph fingerprint of state before FirstSeq
	EndState     string // 64-hex glyph fingerprint of state after LastSeq
	PrevIdentity string // 64-hex EpisodeIdentity of the preceding episode
	Source       map[string]string
}

// provenanceBlockBytes renders provenance as canonical JSON, via a map so
// encoding/json sorts the keys rather than emitting them in struct order.
// Every field is written even when empty: an absent key and a key holding ""
// are different canonical JSON, so omitting would make two equal provenances
// hash differently across producers.
// maxSafeInt is the largest integer canonical JSON renders exactly. glyph
// SPEC-CANON.md §4 makes anything past it an error rather than a silent float64
// collapse, but Python reaches canon_json through from_json_loose, which
// collapses first: 2^53 and 2^53+1 both render 9.007199254740992e+15 and so
// share a fingerprint. Go would emit the exact int64 and diverge. Both sides
// check instead.
const maxSafeInt = int64(1)<<53 - 1

func checkSafeInt(field string, v int64) error {
	if v > maxSafeInt || v < -maxSafeInt {
		return fmt.Errorf(
			"wshard: provenance %s=%d exceeds +/-2^53; canonical JSON would "+
				"render it as a float and lose the exact value", field, v)
	}
	return nil
}

func provenanceBlockBytes(p *WShardProvenance) ([]byte, error) {
	for _, f := range []struct {
		name string
		v    int64
	}{{"epoch", p.Epoch}, {"first_seq", p.FirstSeq}, {"last_seq", p.LastSeq}} {
		if err := checkSafeInt(f.name, f.v); err != nil {
			return nil, err
		}
	}
	src := p.Source
	if src == nil {
		src = map[string]string{}
	}

	// Written out by hand rather than through encoding/json, which escapes more
	// than the canon does -- see canonjson.go. RunID and the Source map are
	// free-form producer strings, so that difference is reachable here. Fields
	// are in UTF-8 byte order, which is the order the canon sorts them into.
	var b bytes.Buffer
	b.WriteString(`{"end_state":`)
	writeCanonString(&b, p.EndState)
	fmt.Fprintf(&b, `,"epoch":%d,"first_seq":%d,"last_seq":%d,"prev_identity":`,
		p.Epoch, p.FirstSeq, p.LastSeq)
	writeCanonString(&b, p.PrevIdentity)
	b.WriteString(`,"run_id":`)
	writeCanonString(&b, p.RunID)
	b.WriteString(`,"source":`)
	writeCanonStringMap(&b, src)
	b.WriteString(`,"start_state":`)
	writeCanonString(&b, p.StartState)
	b.WriteString(`,"v":1}`)
	return b.Bytes(), nil
}

type provenanceJ struct {
	V            int               `json:"v"`
	RunID        string            `json:"run_id"`
	Epoch        int64             `json:"epoch"`
	FirstSeq     int64             `json:"first_seq"`
	LastSeq      int64             `json:"last_seq"`
	StartState   string            `json:"start_state"`
	EndState     string            `json:"end_state"`
	PrevIdentity string            `json:"prev_identity"`
	Source       map[string]string `json:"source"`
}

func parseProvenance(data []byte) (*WShardProvenance, error) {
	var j provenanceJ
	if err := json.Unmarshal(data, &j); err != nil {
		return nil, fmt.Errorf("wshard: parse %s: %w", ProvenanceBlock, err)
	}
	if j.V != 1 {
		return nil, fmt.Errorf("wshard: unsupported %s version: %d", ProvenanceBlock, j.V)
	}
	src := j.Source
	if src == nil {
		src = map[string]string{}
	}
	return &WShardProvenance{
		RunID:        j.RunID,
		Epoch:        j.Epoch,
		FirstSeq:     j.FirstSeq,
		LastSeq:      j.LastSeq,
		StartState:   j.StartState,
		EndState:     j.EndState,
		PrevIdentity: j.PrevIdentity,
		Source:       src,
	}, nil
}

// EpisodeProvenance reads just the provenance entry, or nil if the file has
// none. Decodes no tensors, so walking a PrevIdentity chain backwards costs
// one small read per episode instead of a full OpenWShard.
func EpisodeProvenance(path string) (*WShardProvenance, error) {
	r, err := OpenShard(path)
	if err != nil {
		return nil, err
	}
	defer r.Close()
	data, err := r.ReadEntryByName(ProvenanceBlock)
	if errors.Is(err, ErrEntryNotFound) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return parseProvenance(data)
}
