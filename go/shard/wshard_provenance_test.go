package shard

import (
	"path/filepath"
	"reflect"
	"testing"
)

func probeProvenance() *WShardProvenance {
	return &WShardProvenance{
		RunID:        "run-7f3a",
		Epoch:        7,
		FirstSeq:     10001,
		LastSeq:      20000,
		StartState:   "aa",
		EndState:     "bb",
		PrevIdentity: "cc",
		Source:       map[string]string{"schema_fp": "dd", "git_commit": "deadbeef"},
	}
}

func TestProvenanceBlockIsCanonicalJSON(t *testing.T) {
	// Pinned bytes: what Python's glyph.canon_json emits for the same value.
	// Keys ascend by UTF-8 byte, so "v" lands last and "source" nests sorted.
	got, err := provenanceBlockBytes(probeProvenance())
	if err != nil {
		t.Fatal(err)
	}
	want := `{"end_state":"bb","epoch":7,"first_seq":10001,"last_seq":20000,` +
		`"prev_identity":"cc","run_id":"run-7f3a",` +
		`"source":{"git_commit":"deadbeef","schema_fp":"dd"},` +
		`"start_state":"aa","v":1}`
	if string(got) != want {
		t.Fatalf("provenance block\n got %s\nwant %s", got, want)
	}
}

func TestProvenanceRejectsIntsPast2To53(t *testing.T) {
	// Python reaches canon_json through from_json_loose, which collapses an
	// integer past 2^53 to float64 before canon_json can object: 2^53 and
	// 2^53+1 both render 9.007199254740992e+15. Go would emit the exact int64
	// and diverge, so it refuses the same values Python refuses.
	if _, err := provenanceBlockBytes(&WShardProvenance{LastSeq: maxSafeInt}); err != nil {
		t.Fatalf("boundary value rejected: %v", err)
	}
	for _, tc := range []struct {
		name string
		p    *WShardProvenance
	}{
		{"epoch", &WShardProvenance{Epoch: maxSafeInt + 1}},
		{"first_seq", &WShardProvenance{FirstSeq: maxSafeInt + 1}},
		{"last_seq", &WShardProvenance{LastSeq: maxSafeInt + 1}},
		{"negative", &WShardProvenance{LastSeq: -maxSafeInt - 1}},
	} {
		if _, err := provenanceBlockBytes(tc.p); err == nil {
			t.Fatalf("%s: expected refusal past 2^53, got none", tc.name)
		}
	}
}

func TestProvenanceRoundTrip(t *testing.T) {
	p := filepath.Join(t.TempDir(), "ep.wshard")
	ep := identityEpisode()
	ep.Provenance = probeProvenance()
	if err := CreateWShard(p, ep); err != nil {
		t.Fatal(err)
	}

	got, err := OpenWShard(p)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.Provenance, ep.Provenance) {
		t.Fatalf("provenance\n got %+v\nwant %+v", got.Provenance, ep.Provenance)
	}

	// Cheap read path agrees with the full open.
	only, err := EpisodeProvenance(p)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(only, ep.Provenance) {
		t.Fatalf("EpisodeProvenance\n got %+v\nwant %+v", only, ep.Provenance)
	}
}

func TestProvenanceAbsentWhenNil(t *testing.T) {
	p := filepath.Join(t.TempDir(), "ep.wshard")
	if err := CreateWShard(p, identityEpisode()); err != nil {
		t.Fatal(err)
	}
	got, err := EpisodeProvenance(p)
	if err != nil {
		t.Fatal(err)
	}
	if got != nil {
		t.Fatalf("expected no provenance, got %+v", got)
	}
	ep, err := OpenWShard(p)
	if err != nil {
		t.Fatal(err)
	}
	if ep.Provenance != nil {
		t.Fatalf("expected nil provenance on episode, got %+v", ep.Provenance)
	}
}

func TestIdentityCommitsToProvenance(t *testing.T) {
	// Provenance is written before meta/identity precisely so the identity
	// covers it. Without that, anyone could rewrite which run produced an
	// episode and every checksum in the file would still verify.
	dir := t.TempDir()
	a := filepath.Join(dir, "a.wshard")
	b := filepath.Join(dir, "b.wshard")

	ep := identityEpisode()
	ep.Provenance = probeProvenance()
	if err := CreateWShard(a, ep); err != nil {
		t.Fatal(err)
	}

	ep.Provenance.RunID = "run-somebody-else"
	if err := CreateWShard(b, ep); err != nil {
		t.Fatal(err)
	}

	idA, err := EpisodeIdentity(a)
	if err != nil {
		t.Fatal(err)
	}
	idB, err := EpisodeIdentity(b)
	if err != nil {
		t.Fatal(err)
	}
	if idA == idB {
		t.Fatal("provenance changed, identity did not")
	}
	if _, err := VerifyIdentity(a); err != nil {
		t.Fatalf("verify: %v", err)
	}
}
