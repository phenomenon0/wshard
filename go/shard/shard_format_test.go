package shard

import (
	"bytes"
	"encoding/binary"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"testing"
)

// ============================================================
// Header Tests
// ============================================================

func TestShardHeaderRoundtrip(t *testing.T) {
	h := NewShardHeader(ShardRoleMoSH)
	h.EntryCount = 42
	h.Alignment = Align64

	var buf bytes.Buffer
	if err := WriteShardHeader(&buf, h); err != nil {
		t.Fatalf("write header: %v", err)
	}
	if buf.Len() != 64 {
		t.Fatalf("expected 64 bytes, got %d", buf.Len())
	}

	h2, err := ReadShardHeader(&buf)
	if err != nil {
		t.Fatalf("read header: %v", err)
	}

	if h2.Version != 0x02 {
		t.Errorf("version: %d", h2.Version)
	}
	if h2.Role != ShardRoleMoSH {
		t.Errorf("role: %d", h2.Role)
	}
	if h2.EntryCount != 42 {
		t.Errorf("entry count: %d", h2.EntryCount)
	}
	if h2.Alignment != Align64 {
		t.Errorf("alignment: %d", h2.Alignment)
	}
}

func TestShardHeaderAllRoles(t *testing.T) {
	roles := []ShardRole{
		ShardRoleUnknown, ShardRoleMoSH, ShardRoleSample,
		ShardRoleGemmPanel, ShardRoleManifest, ShardRoleWShard, ShardRoleUMSH,
	}
	for _, role := range roles {
		h := NewShardHeader(role)
		var buf bytes.Buffer
		if err := WriteShardHeader(&buf, h); err != nil {
			t.Fatalf("write role %d: %v", role, err)
		}
		h2, err := ReadShardHeader(&buf)
		if err != nil {
			t.Fatalf("read role %d: %v", role, err)
		}
		if h2.Role != role {
			t.Errorf("role mismatch: got %d, want %d", h2.Role, role)
		}
	}
}

func TestShardHeaderAllFlags(t *testing.T) {
	h := NewShardHeader(ShardRoleMoSH)
	h.Flags = ShardFlagLittleEndian | ShardFlagHasSchema | ShardFlagHasChecksums | ShardFlagStreaming | ShardFlagHasContentTypes

	var buf bytes.Buffer
	WriteShardHeader(&buf, h)
	h2, err := ReadShardHeader(&buf)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if h2.Flags != h.Flags {
		t.Errorf("flags: got 0x%04x, want 0x%04x", h2.Flags, h.Flags)
	}
}

func TestShardHeaderBadMagic(t *testing.T) {
	buf := make([]byte, 64)
	copy(buf[0:4], []byte("NOPE"))
	buf[4] = ShardVersion2
	binary.LittleEndian.PutUint16(buf[10:12], ShardIndexEntrySize)
	_, err := ReadShardHeader(bytes.NewReader(buf))
	if err == nil {
		t.Fatal("expected error for bad magic")
	}
}

func TestShardHeaderBadVersion(t *testing.T) {
	buf := make([]byte, 64)
	copy(buf[0:4], ShardMagic[:])
	buf[4] = 0xFF // bad version
	binary.LittleEndian.PutUint16(buf[10:12], ShardIndexEntrySize)
	_, err := ReadShardHeader(bytes.NewReader(buf))
	if err == nil {
		t.Fatal("expected error for bad version")
	}
}

func TestShardHeaderBadIndexEntrySize(t *testing.T) {
	buf := make([]byte, 64)
	copy(buf[0:4], ShardMagic[:])
	buf[4] = ShardVersion2
	binary.LittleEndian.PutUint16(buf[10:12], 99) // wrong size
	_, err := ReadShardHeader(bytes.NewReader(buf))
	if err == nil {
		t.Fatal("expected error for bad index entry size")
	}
}

func TestShardHeaderTruncated(t *testing.T) {
	_, err := ReadShardHeader(bytes.NewReader(make([]byte, 10)))
	if err == nil {
		t.Fatal("expected error for truncated header")
	}
}

func TestShardHeaderAllAlignments(t *testing.T) {
	for _, align := range []uint8{AlignNone, Align16, Align32, Align64} {
		h := NewShardHeader(ShardRoleMoSH)
		h.Alignment = align
		var buf bytes.Buffer
		WriteShardHeader(&buf, h)
		h2, _ := ReadShardHeader(&buf)
		if h2.Alignment != align {
			t.Errorf("alignment %d roundtrip: got %d", align, h2.Alignment)
		}
	}
}

// ============================================================
// Index Entry Tests
// ============================================================

func TestIndexEntryRoundtrip(t *testing.T) {
	e := &IndexEntry{
		NameHash:   0xDEADBEEF12345678,
		NameOffset: 42,
		NameLen:    10,
		Flags:      EntryFlagCompressed | EntryFlagZstd,
		DataOffset: 1024,
		DiskSize:   500,
		OrigSize:   1000,
		Checksum:   0xAABBCCDD,
		Reserved:   uint32(ContentTypeJSON),
	}

	var buf bytes.Buffer
	if err := WriteIndexEntry(&buf, e); err != nil {
		t.Fatalf("write: %v", err)
	}
	if buf.Len() != 48 {
		t.Fatalf("expected 48 bytes, got %d", buf.Len())
	}

	e2, err := ReadIndexEntry(&buf)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if e2.NameHash != e.NameHash {
		t.Errorf("NameHash: %x", e2.NameHash)
	}
	if e2.Flags != e.Flags {
		t.Errorf("Flags: %x", e2.Flags)
	}
	if e2.DiskSize != e.DiskSize || e2.OrigSize != e.OrigSize {
		t.Errorf("sizes: disk=%d orig=%d", e2.DiskSize, e2.OrigSize)
	}
	if e2.Checksum != e.Checksum {
		t.Errorf("checksum: %x", e2.Checksum)
	}
}

func TestIndexEntryFlags(t *testing.T) {
	e := &IndexEntry{}

	// Not compressed initially
	if e.IsCompressed() {
		t.Error("should not be compressed")
	}
	if e.CompressionType() != CompressNone {
		t.Error("should be CompressNone")
	}

	// Set zstd
	e.Flags = EntryFlagCompressed | EntryFlagZstd
	if !e.IsCompressed() {
		t.Error("should be compressed")
	}
	if e.CompressionType() != CompressZstd {
		t.Errorf("expected zstd, got %d", e.CompressionType())
	}

	// Set lz4
	e.Flags = EntryFlagCompressed | EntryFlagLZ4
	if e.CompressionType() != CompressLZ4 {
		t.Errorf("expected lz4, got %d", e.CompressionType())
	}

	// Chunked
	if e.IsChunked() {
		t.Error("should not be chunked")
	}
	e.Flags |= EntryFlagChunked
	if !e.IsChunked() {
		t.Error("should be chunked")
	}
}

func TestIndexEntryContentType(t *testing.T) {
	e := &IndexEntry{}

	if e.ContentType() != ContentTypeUnknown {
		t.Errorf("expected unknown, got %d", e.ContentType())
	}

	e.SetContentType(ContentTypeTensor)
	if e.ContentType() != ContentTypeTensor {
		t.Errorf("expected tensor, got %d", e.ContentType())
	}

	// Verify upper bits are not affected
	e.SetTagBits(0xFFFF)
	if e.ContentType() != ContentTypeTensor {
		t.Errorf("content type corrupted by tag bits: got %d", e.ContentType())
	}
}

func TestIndexEntryTagBits(t *testing.T) {
	e := &IndexEntry{}

	if e.TagBits() != 0 {
		t.Errorf("expected 0 tag bits, got %d", e.TagBits())
	}

	// Set individual bits
	for bit := 0; bit < 16; bit++ {
		e.SetTagBit(bit)
		if !e.HasTagBit(bit) {
			t.Errorf("bit %d not set", bit)
		}
	}
	if e.TagBits() != 0xFFFF {
		t.Errorf("expected 0xFFFF, got 0x%04x", e.TagBits())
	}

	// Content type should be preserved
	e.SetContentType(ContentTypeJSON)
	if e.ContentType() != ContentTypeJSON {
		t.Errorf("content type lost: got %d", e.ContentType())
	}
	if e.TagBits() != 0xFFFF {
		t.Errorf("tag bits lost: got 0x%04x", e.TagBits())
	}

	// Out of range bits are no-ops
	e2 := &IndexEntry{}
	e2.SetTagBit(-1)
	e2.SetTagBit(16)
	e2.SetTagBit(100)
	if e2.TagBits() != 0 {
		t.Errorf("out-of-range bits should be ignored: got 0x%04x", e2.TagBits())
	}
	if e2.HasTagBit(-1) || e2.HasTagBit(16) {
		t.Error("out-of-range HasTagBit should return false")
	}
}

func TestIndexEntryTagBitsPreserveContentType(t *testing.T) {
	e := &IndexEntry{}
	e.SetContentType(ContentTypeImage)
	e.SetTagBits(0xABCD)

	if e.ContentType() != ContentTypeImage {
		t.Errorf("content type: got %d, want %d", e.ContentType(), ContentTypeImage)
	}
	if e.TagBits() != 0xABCD {
		t.Errorf("tag bits: got 0x%04x, want 0xABCD", e.TagBits())
	}

	// Verify raw reserved field
	expected := (uint32(0xABCD) << 16) | uint32(ContentTypeImage)
	if e.Reserved != expected {
		t.Errorf("reserved: got 0x%08x, want 0x%08x", e.Reserved, expected)
	}
}

func TestContentTypeName(t *testing.T) {
	tests := map[uint16]string{
		ContentTypeUnknown: "unknown",
		ContentTypeTensor:  "tensor",
		ContentTypeJSON:    "json",
		ContentTypeCowrie:  "cowrie",
		ContentTypeGLYPH:   "glyph",
		ContentTypeText:    "text",
		ContentTypeImage:   "image",
		ContentTypeAudio:   "audio",
		ContentTypeVideo:   "video",
		ContentTypeProto:   "proto",
		ContentTypeBlob:    "blob",
	}
	for ct, name := range tests {
		if got := ContentTypeName(ct); got != name {
			t.Errorf("ContentTypeName(%d) = %q, want %q", ct, got, name)
		}
	}
	// User-defined
	got := ContentTypeName(ContentTypeUserBase + 5)
	if got != "user:5" {
		t.Errorf("user content type: %q", got)
	}
}

// ============================================================
// Path Helper Tests
// ============================================================

func TestSplitPath(t *testing.T) {
	tests := []struct {
		input string
		want  []string
	}{
		{"", nil},
		{"a", []string{"a"}},
		{"a/b/c", []string{"a", "b", "c"}},
		{"layer0/weight", []string{"layer0", "weight"}},
	}
	for _, tt := range tests {
		got := SplitPath(tt.input)
		if len(got) != len(tt.want) {
			t.Errorf("SplitPath(%q) = %v, want %v", tt.input, got, tt.want)
			continue
		}
		for i := range got {
			if got[i] != tt.want[i] {
				t.Errorf("SplitPath(%q)[%d] = %q, want %q", tt.input, i, got[i], tt.want[i])
			}
		}
	}
}

func TestJoinPath(t *testing.T) {
	if got := JoinPath("a", "b", "c"); got != "a/b/c" {
		t.Errorf("JoinPath: %q", got)
	}
}

func TestPathPrefix(t *testing.T) {
	tests := []struct {
		name, prefix string
		want         bool
	}{
		{"layer0/weight", "layer0", true},
		{"layer0/weight", "layer0/", true},
		{"layer0/weight", "layer1", false},
		{"config", "", true},
		{"layer0", "layer0", true},
	}
	for _, tt := range tests {
		if got := PathPrefix(tt.name, tt.prefix); got != tt.want {
			t.Errorf("PathPrefix(%q, %q) = %v, want %v", tt.name, tt.prefix, got, tt.want)
		}
	}
}

func TestPathParent(t *testing.T) {
	if got := PathParent("a/b/c"); got != "a/b" {
		t.Errorf("got %q", got)
	}
	if got := PathParent("root"); got != "" {
		t.Errorf("got %q", got)
	}
}

func TestPathBase(t *testing.T) {
	if got := PathBase("a/b/c"); got != "c" {
		t.Errorf("got %q", got)
	}
	if got := PathBase("root"); got != "root" {
		t.Errorf("got %q", got)
	}
}

// ============================================================
// Writer/Reader Integration Tests
// ============================================================

func TestShardWriteRead(t *testing.T) {
	path := filepath.Join(t.TempDir(), "test.shard")

	w, err := NewShardWriter(path, ShardRoleMoSH)
	if err != nil {
		t.Fatalf("create writer: %v", err)
	}

	w.SetAlignment(Align64)
	w.SetCompression(CompressZstd)

	if err := w.WriteEntry("config", []byte(`{"model":"test"}`)); err != nil {
		t.Fatalf("write config: %v", err)
	}

	bigData := make([]byte, 4096)
	for i := range bigData {
		bigData[i] = byte(i % 256)
	}
	if err := w.WriteEntryCompressed("weights", bigData); err != nil {
		t.Fatalf("write weights: %v", err)
	}

	if err := w.WriteEntryTyped("metadata", []byte(`{"version":1}`), ContentTypeJSON); err != nil {
		t.Fatalf("write metadata: %v", err)
	}

	if err := w.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	// Read back
	r, err := OpenShard(path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer r.Close()

	if r.EntryCount() != 3 {
		t.Fatalf("expected 3 entries, got %d", r.EntryCount())
	}

	// O(1) lookup
	idx := r.Lookup("config")
	if idx < 0 {
		t.Fatal("config not found")
	}
	data, err := r.ReadEntry(idx)
	if err != nil {
		t.Fatalf("read config: %v", err)
	}
	if string(data) != `{"model":"test"}` {
		t.Errorf("config data: %q", data)
	}

	// Compressed entry
	idx = r.Lookup("weights")
	if idx < 0 {
		t.Fatal("weights not found")
	}
	info := r.GetEntryInfo(idx)
	if !info.IsCompressed() {
		t.Error("expected weights to be compressed")
	}
	data, err = r.ReadEntry(idx)
	if err != nil {
		t.Fatalf("read weights: %v", err)
	}
	if len(data) != 4096 {
		t.Errorf("expected 4096 bytes, got %d", len(data))
	}
	for i, b := range data {
		if b != byte(i%256) {
			t.Fatalf("data mismatch at %d: got %d, want %d", i, b, byte(i%256))
		}
	}

	// Content type
	idx = r.Lookup("metadata")
	if idx < 0 {
		t.Fatal("metadata not found")
	}
	info = r.GetEntryInfo(idx)
	if info.ContentType() != ContentTypeJSON {
		t.Errorf("expected JSON content type, got %d", info.ContentType())
	}
}

func TestShardEmptyShard(t *testing.T) {
	path := filepath.Join(t.TempDir(), "empty.shard")

	w, _ := NewShardWriter(path, ShardRoleMoSH)
	if err := w.Close(); err != nil {
		t.Fatalf("close empty: %v", err)
	}

	r, err := OpenShard(path)
	if err != nil {
		t.Fatalf("open empty: %v", err)
	}
	defer r.Close()

	if r.EntryCount() != 0 {
		t.Errorf("expected 0 entries, got %d", r.EntryCount())
	}
	if r.Lookup("anything") >= 0 {
		t.Error("lookup in empty shard should return -1")
	}
	if names := r.EntryNames(); len(names) != 0 {
		t.Errorf("expected empty names, got %v", names)
	}
}

func TestShardSingleByteEntry(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tiny.shard")

	w, _ := NewShardWriter(path, ShardRoleMoSH)
	w.WriteEntry("x", []byte{0x42})
	w.Close()

	r, _ := OpenShard(path)
	defer r.Close()

	data, err := r.ReadEntry(0)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if len(data) != 1 || data[0] != 0x42 {
		t.Errorf("data: %v", data)
	}
}

func TestShardEmptyData(t *testing.T) {
	path := filepath.Join(t.TempDir(), "empty-data.shard")

	w, _ := NewShardWriter(path, ShardRoleMoSH)
	w.WriteEntry("empty", []byte{})
	w.Close()

	r, _ := OpenShard(path)
	defer r.Close()

	data, err := r.ReadEntry(0)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if len(data) != 0 {
		t.Errorf("expected empty, got %d bytes", len(data))
	}
}

func TestShardLargeEntry(t *testing.T) {
	path := filepath.Join(t.TempDir(), "large.shard")

	data := make([]byte, 1<<20) // 1MB
	for i := range data {
		data[i] = byte(i % 251) // prime modulus for variety
	}

	w, _ := NewShardWriter(path, ShardRoleMoSH)
	w.WriteEntry("big", data)
	w.Close()

	r, _ := OpenShard(path)
	defer r.Close()

	got, err := r.ReadEntry(0)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if !bytes.Equal(got, data) {
		t.Error("large data mismatch")
	}
}

func TestShardManyEntries(t *testing.T) {
	path := filepath.Join(t.TempDir(), "many.shard")
	count := 500

	w, _ := NewShardWriter(path, ShardRoleMoSH)
	for i := 0; i < count; i++ {
		name := filepath.Join("entries", string(rune('A'+i%26)), string(rune('0'+i%10)))
		w.WriteEntry(name, []byte{byte(i)})
	}
	w.Close()

	r, _ := OpenShard(path)
	defer r.Close()

	if r.EntryCount() != count {
		t.Fatalf("expected %d entries, got %d", count, r.EntryCount())
	}

	// Verify all entries are readable
	for i := 0; i < count; i++ {
		data, err := r.ReadEntry(i)
		if err != nil {
			t.Fatalf("read entry %d: %v", i, err)
		}
		if len(data) != 1 || data[0] != byte(i) {
			t.Fatalf("entry %d: got %v", i, data)
		}
	}
}

func TestShardBinaryData(t *testing.T) {
	path := filepath.Join(t.TempDir(), "binary.shard")

	// All 256 byte values
	data := make([]byte, 256)
	for i := range data {
		data[i] = byte(i)
	}

	w, _ := NewShardWriter(path, ShardRoleMoSH)
	w.WriteEntry("allbytes", data)
	w.Close()

	r, _ := OpenShard(path)
	defer r.Close()

	got, _ := r.ReadEntry(0)
	if !bytes.Equal(got, data) {
		t.Error("binary data mismatch")
	}
}

func TestShardUnicodeNames(t *testing.T) {
	path := filepath.Join(t.TempDir(), "unicode.shard")

	names := []string{"日本語", "Ω/α/β", "emoji🎉", "layer.0/wëights"}

	w, _ := NewShardWriter(path, ShardRoleMoSH)
	for i, name := range names {
		w.WriteEntry(name, []byte{byte(i)})
	}
	w.Close()

	r, _ := OpenShard(path)
	defer r.Close()

	for i, name := range names {
		idx := r.Lookup(name)
		if idx < 0 {
			t.Errorf("unicode name %q not found", name)
			continue
		}
		data, _ := r.ReadEntry(idx)
		if len(data) != 1 || data[0] != byte(i) {
			t.Errorf("data for %q: %v", name, data)
		}
	}
}

// ============================================================
// Compression Tests
// ============================================================

func TestShardZstdRoundtrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "zstd.shard")

	data := bytes.Repeat([]byte("compressible data pattern "), 200) // ~5KB, highly compressible

	w, _ := NewShardWriter(path, ShardRoleMoSH)
	w.SetCompression(CompressZstd)
	w.WriteEntryCompressed("data", data)
	w.Close()

	r, _ := OpenShard(path)
	defer r.Close()

	info := r.GetEntryInfo(0)
	if !info.IsCompressed() {
		t.Fatal("expected compressed")
	}
	if info.CompressionType() != CompressZstd {
		t.Errorf("expected zstd, got %d", info.CompressionType())
	}
	if info.DiskSize >= info.OrigSize {
		t.Errorf("compression didn't shrink data: disk=%d orig=%d", info.DiskSize, info.OrigSize)
	}

	got, err := r.ReadEntry(0)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if !bytes.Equal(got, data) {
		t.Error("decompressed data mismatch")
	}
}

func TestShardLZ4Roundtrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "lz4.shard")

	data := bytes.Repeat([]byte("lz4 compressible pattern!! "), 200)

	w, _ := NewShardWriter(path, ShardRoleMoSH)
	w.SetCompression(CompressLZ4)
	w.WriteEntryCompressed("data", data)
	w.Close()

	r, _ := OpenShard(path)
	defer r.Close()

	info := r.GetEntryInfo(0)
	if !info.IsCompressed() {
		t.Fatal("expected compressed")
	}
	if info.CompressionType() != CompressLZ4 {
		t.Errorf("expected lz4, got %d", info.CompressionType())
	}

	got, err := r.ReadEntry(0)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if !bytes.Equal(got, data) {
		t.Error("decompressed data mismatch")
	}
}

func TestShardSmallDataNotCompressed(t *testing.T) {
	path := filepath.Join(t.TempDir(), "small.shard")

	data := []byte("tiny") // < minCompressSize (256)

	w, _ := NewShardWriter(path, ShardRoleMoSH)
	w.SetCompression(CompressZstd)
	w.WriteEntryCompressed("data", data)
	w.Close()

	r, _ := OpenShard(path)
	defer r.Close()

	info := r.GetEntryInfo(0)
	if info.IsCompressed() {
		t.Error("small data should not be compressed")
	}
}

func TestShardMixedCompression(t *testing.T) {
	path := filepath.Join(t.TempDir(), "mixed.shard")

	compressible := bytes.Repeat([]byte("AAAA"), 500)
	plain := []byte("uncompressed")

	w, _ := NewShardWriter(path, ShardRoleMoSH)
	w.SetCompression(CompressZstd)
	w.WriteEntryCompressed("compressed", compressible)
	w.WriteEntry("plain", plain)
	w.Close()

	r, _ := OpenShard(path)
	defer r.Close()

	info0 := r.GetEntryInfo(0)
	info1 := r.GetEntryInfo(1)

	if !info0.IsCompressed() {
		t.Error("first entry should be compressed")
	}
	if info1.IsCompressed() {
		t.Error("second entry should not be compressed")
	}

	d0, _ := r.ReadEntry(0)
	d1, _ := r.ReadEntry(1)
	if !bytes.Equal(d0, compressible) {
		t.Error("compressed data mismatch")
	}
	if !bytes.Equal(d1, plain) {
		t.Error("plain data mismatch")
	}
}

// ============================================================
// Checksum / Corruption Tests
// ============================================================

func TestShardChecksumMismatch(t *testing.T) {
	path := filepath.Join(t.TempDir(), "corrupt.shard")

	w, _ := NewShardWriter(path, ShardRoleMoSH)
	w.WriteEntry("data", []byte("hello world"))
	w.Close()

	// Corrupt the data section
	raw, _ := os.ReadFile(path)
	if len(raw) > 100 {
		raw[len(raw)-5] ^= 0xFF
		os.WriteFile(path, raw, 0o644)
	}

	r, err := OpenShard(path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer r.Close()

	_, err = r.ReadEntry(0)
	if err == nil {
		t.Error("expected checksum error")
	}
}

func TestShardChecksumVerifyFalse(t *testing.T) {
	path := filepath.Join(t.TempDir(), "corrupt-noverify.shard")

	w, _ := NewShardWriter(path, ShardRoleMoSH)
	w.WriteEntry("data", []byte("hello world"))
	w.Close()

	raw, _ := os.ReadFile(path)
	if len(raw) > 100 {
		raw[len(raw)-5] ^= 0xFF
		os.WriteFile(path, raw, 0o644)
	}

	r, _ := OpenShard(path)
	defer r.Close()

	// Should succeed with verify=false
	_, err := r.ReadEntryWithVerify(0, false)
	if err != nil {
		t.Errorf("expected no error with verify=false: %v", err)
	}
}

func TestShardTruncatedFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "truncated.shard")

	w, _ := NewShardWriter(path, ShardRoleMoSH)
	w.WriteEntry("data", bytes.Repeat([]byte("A"), 1000))
	w.Close()

	raw, _ := os.ReadFile(path)
	// Truncate to half
	os.WriteFile(path, raw[:len(raw)/2], 0o644)

	_, err := OpenShard(path)
	if err == nil {
		t.Error("expected error opening truncated file")
	}
}

func TestShardEmptyFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "empty.shard")
	os.WriteFile(path, []byte{}, 0o644)

	_, err := OpenShard(path)
	if err == nil {
		t.Error("expected error opening empty file")
	}
}

func TestShardWrongMagicFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "badmagic.shard")
	os.WriteFile(path, bytes.Repeat([]byte("X"), 200), 0o644)

	_, err := OpenShard(path)
	if err == nil {
		t.Error("expected error for bad magic")
	}
}

func TestShardCRCOnDecompressedData(t *testing.T) {
	// Verify CRC is computed on decompressed data, not compressed
	path := filepath.Join(t.TempDir(), "crc-decompressed.shard")

	data := bytes.Repeat([]byte("verify CRC is on decompressed "), 200)
	expectedCRC := ComputeChecksum(data)

	w, _ := NewShardWriter(path, ShardRoleMoSH)
	w.SetCompression(CompressZstd)
	w.WriteEntryCompressed("data", data)
	w.Close()

	r, _ := OpenShard(path)
	defer r.Close()

	info := r.GetEntryInfo(0)
	if !info.IsCompressed() {
		t.Fatal("expected compressed")
	}
	if info.Checksum != expectedCRC {
		t.Errorf("CRC should be on decompressed data: stored=0x%08x expected=0x%08x", info.Checksum, expectedCRC)
	}
}

// ============================================================
// Alignment Tests
// ============================================================

func TestShardAlignmentValues(t *testing.T) {
	for _, align := range []uint8{AlignNone, Align16, Align32, Align64} {
		path := filepath.Join(t.TempDir(), "align.shard")

		w, _ := NewShardWriter(path, ShardRoleMoSH)
		w.SetAlignment(align)
		w.WriteEntry("a", []byte("hello"))
		w.WriteEntry("b", []byte("world"))
		w.Close()

		r, _ := OpenShard(path)

		if align > 0 {
			for i := 0; i < r.EntryCount(); i++ {
				info := r.GetEntryInfo(i)
				if info.DataOffset%uint64(align) != 0 {
					t.Errorf("align=%d: entry %d offset %d not aligned", align, i, info.DataOffset)
				}
			}
		}

		// Verify data integrity regardless of alignment
		d0, _ := r.ReadEntry(0)
		d1, _ := r.ReadEntry(1)
		if string(d0) != "hello" || string(d1) != "world" {
			t.Errorf("align=%d: data mismatch", align)
		}
		r.Close()
	}
}

func TestShardInvalidAlignment(t *testing.T) {
	path := filepath.Join(t.TempDir(), "badalign.shard")
	w, _ := NewShardWriter(path, ShardRoleMoSH)
	err := w.SetAlignment(7)
	if err == nil {
		t.Error("expected error for invalid alignment 7")
	}
	w.Close()
}

// ============================================================
// Lookup Tests
// ============================================================

func TestShardLookupMissing(t *testing.T) {
	path := filepath.Join(t.TempDir(), "lookup.shard")

	w, _ := NewShardWriter(path, ShardRoleMoSH)
	w.WriteEntry("exists", []byte("yes"))
	w.Close()

	r, _ := OpenShard(path)
	defer r.Close()

	if r.Lookup("exists") < 0 {
		t.Error("expected to find 'exists'")
	}
	if r.Lookup("missing") >= 0 {
		t.Error("expected -1 for missing entry")
	}
}

func TestShardReadEntryByName(t *testing.T) {
	path := filepath.Join(t.TempDir(), "byname.shard")

	w, _ := NewShardWriter(path, ShardRoleMoSH)
	w.WriteEntry("alpha", []byte("A"))
	w.WriteEntry("beta", []byte("B"))
	w.Close()

	r, _ := OpenShard(path)
	defer r.Close()

	d, err := r.ReadEntryByName("beta")
	if err != nil {
		t.Fatalf("read by name: %v", err)
	}
	if string(d) != "B" {
		t.Errorf("got %q", d)
	}

	_, err = r.ReadEntryByName("missing")
	if err == nil {
		t.Error("expected error for missing entry")
	}
}

func TestShardReadEntryOutOfBounds(t *testing.T) {
	path := filepath.Join(t.TempDir(), "oob.shard")

	w, _ := NewShardWriter(path, ShardRoleMoSH)
	w.WriteEntry("x", []byte("data"))
	w.Close()

	r, _ := OpenShard(path)
	defer r.Close()

	_, err := r.ReadEntry(-1)
	if err == nil {
		t.Error("expected error for index -1")
	}
	_, err = r.ReadEntry(999)
	if err == nil {
		t.Error("expected error for index 999")
	}
}

func TestShardGetEntryInfoOutOfBounds(t *testing.T) {
	path := filepath.Join(t.TempDir(), "info-oob.shard")
	w, _ := NewShardWriter(path, ShardRoleMoSH)
	w.WriteEntry("x", []byte("y"))
	w.Close()

	r, _ := OpenShard(path)
	defer r.Close()

	if r.GetEntryInfo(-1) != nil {
		t.Error("expected nil for -1")
	}
	if r.GetEntryInfo(100) != nil {
		t.Error("expected nil for 100")
	}
}

func TestShardEntryNameOutOfBounds(t *testing.T) {
	path := filepath.Join(t.TempDir(), "name-oob.shard")
	w, _ := NewShardWriter(path, ShardRoleMoSH)
	w.WriteEntry("x", []byte("y"))
	w.Close()

	r, _ := OpenShard(path)
	defer r.Close()

	if r.EntryName(-1) != "" {
		t.Error("expected empty for -1")
	}
	if r.EntryName(100) != "" {
		t.Error("expected empty for 100")
	}
}

// ============================================================
// ReadEntryPrefix Tests
// ============================================================

func TestShardReadEntryPrefix(t *testing.T) {
	path := filepath.Join(t.TempDir(), "prefix.shard")

	data := bytes.Repeat([]byte("X"), 1000)

	w, _ := NewShardWriter(path, ShardRoleMoSH)
	w.WriteEntry("data", data)
	w.Close()

	r, _ := OpenShard(path)
	defer r.Close()

	// Read first 10 bytes
	prefix, err := r.ReadEntryPrefix(0, 10)
	if err != nil {
		t.Fatalf("read prefix: %v", err)
	}
	if len(prefix) != 10 {
		t.Errorf("expected 10 bytes, got %d", len(prefix))
	}

	// Read more than available
	full, err := r.ReadEntryPrefix(0, 5000)
	if err != nil {
		t.Fatalf("read full: %v", err)
	}
	if len(full) != 1000 {
		t.Errorf("expected 1000 bytes, got %d", len(full))
	}
}

func TestShardReadEntryPrefixCompressed(t *testing.T) {
	path := filepath.Join(t.TempDir(), "prefix-compressed.shard")

	data := bytes.Repeat([]byte("compressible "), 200)

	w, _ := NewShardWriter(path, ShardRoleMoSH)
	w.SetCompression(CompressZstd)
	w.WriteEntryCompressed("data", data)
	w.Close()

	r, _ := OpenShard(path)
	defer r.Close()

	// Even for compressed entries, prefix works (decompresses full, then truncates)
	prefix, err := r.ReadEntryPrefix(0, 20)
	if err != nil {
		t.Fatalf("read prefix: %v", err)
	}
	if len(prefix) != 20 {
		t.Errorf("expected 20 bytes, got %d", len(prefix))
	}
	if !bytes.Equal(prefix, data[:20]) {
		t.Error("prefix data mismatch")
	}
}

// ============================================================
// ListPrefix / ListChildren Tests
// ============================================================

func TestShardListPrefix(t *testing.T) {
	path := filepath.Join(t.TempDir(), "prefix.shard")

	w, _ := NewShardWriter(path, ShardRoleMoSH)
	w.WriteEntry("layer0/weight", []byte("w0"))
	w.WriteEntry("layer0/bias", []byte("b0"))
	w.WriteEntry("layer1/weight", []byte("w1"))
	w.WriteEntry("config", []byte("cfg"))
	w.Close()

	r, _ := OpenShard(path)
	defer r.Close()

	matches := r.ListPrefix("layer0/")
	if len(matches) != 2 {
		t.Errorf("expected 2 matches for layer0/, got %d: %v", len(matches), matches)
	}

	all := r.EntryNames()
	if len(all) != 4 {
		t.Errorf("expected 4 names, got %d", len(all))
	}
}

func TestShardListPrefixEmpty(t *testing.T) {
	path := filepath.Join(t.TempDir(), "prefix-empty.shard")

	w, _ := NewShardWriter(path, ShardRoleMoSH)
	w.WriteEntry("a", []byte("1"))
	w.Close()

	r, _ := OpenShard(path)
	defer r.Close()

	// Empty prefix matches all
	all := r.ListPrefix("")
	if len(all) != 1 {
		t.Errorf("expected 1, got %d", len(all))
	}

	// No match
	none := r.ListPrefix("nonexistent/")
	if len(none) != 0 {
		t.Errorf("expected 0, got %d", len(none))
	}
}

func TestShardListChildren(t *testing.T) {
	path := filepath.Join(t.TempDir(), "children.shard")

	w, _ := NewShardWriter(path, ShardRoleMoSH)
	w.WriteEntry("model/encoder/layer0/weight", []byte("w"))
	w.WriteEntry("model/encoder/layer0/bias", []byte("b"))
	w.WriteEntry("model/encoder/layer1/weight", []byte("w"))
	w.WriteEntry("model/decoder/weight", []byte("w"))
	w.WriteEntry("config", []byte("c"))
	w.Close()

	r, _ := OpenShard(path)
	defer r.Close()

	// Root children
	root := r.ListChildren("")
	sort.Strings(root)
	if len(root) != 2 {
		t.Fatalf("root children: %v", root)
	}
	if root[0] != "config" || root[1] != "model" {
		t.Errorf("root children: %v", root)
	}

	// model/ children
	model := r.ListChildren("model/")
	sort.Strings(model)
	if len(model) != 2 {
		t.Fatalf("model children: %v", model)
	}
	if model[0] != "decoder" || model[1] != "encoder" {
		t.Errorf("model children: %v", model)
	}

	// model/encoder/ children
	encoder := r.ListChildren("model/encoder/")
	sort.Strings(encoder)
	if len(encoder) != 2 {
		t.Fatalf("encoder children: %v", encoder)
	}
	if encoder[0] != "layer0" || encoder[1] != "layer1" {
		t.Errorf("encoder children: %v", encoder)
	}

	// Leaf level
	layer0 := r.ListChildren("model/encoder/layer0/")
	sort.Strings(layer0)
	if len(layer0) != 2 {
		t.Fatalf("layer0 children: %v", layer0)
	}
	if layer0[0] != "bias" || layer0[1] != "weight" {
		t.Errorf("layer0 children: %v", layer0)
	}

	// No children
	none := r.ListChildren("nonexistent/")
	if len(none) != 0 {
		t.Errorf("expected 0, got %d", len(none))
	}
}

func TestShardListChildrenBareComponents(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bare.shard")

	w, _ := NewShardWriter(path, ShardRoleMoSH)
	w.WriteEntry("a/b/c", []byte("1"))
	w.WriteEntry("a/b/d", []byte("2"))
	w.WriteEntry("a/x", []byte("3"))
	w.Close()

	r, _ := OpenShard(path)
	defer r.Close()

	children := r.ListChildren("a/")
	sort.Strings(children)
	// Should be bare components: "b" and "x", NOT "a/b" or "a/x"
	if len(children) != 2 {
		t.Fatalf("children: %v", children)
	}
	if children[0] != "b" || children[1] != "x" {
		t.Errorf("expected bare components [b, x], got %v", children)
	}
}

// ============================================================
// Mmap Tests
// ============================================================

func TestShardMmap(t *testing.T) {
	path := filepath.Join(t.TempDir(), "mmap.shard")

	w, _ := NewShardWriter(path, ShardRoleMoSH)
	w.WriteEntry("data", []byte("mmap test data"))
	w.Close()

	r, _ := OpenShard(path)
	defer r.Close()

	if err := r.EnableMmap(); err != nil {
		t.Fatalf("enable mmap: %v", err)
	}

	// Second call should be no-op
	if err := r.EnableMmap(); err != nil {
		t.Fatalf("second enable mmap: %v", err)
	}

	data, err := r.ReadEntry(0)
	if err != nil {
		t.Fatalf("read with mmap: %v", err)
	}
	if string(data) != "mmap test data" {
		t.Errorf("mmap data: %q", data)
	}
}

func TestShardMmapCompressed(t *testing.T) {
	path := filepath.Join(t.TempDir(), "mmap-compressed.shard")

	data := bytes.Repeat([]byte("mmap compressible "), 200)

	w, _ := NewShardWriter(path, ShardRoleMoSH)
	w.SetCompression(CompressZstd)
	w.WriteEntryCompressed("data", data)
	w.Close()

	r, _ := OpenShard(path)
	defer r.Close()
	r.EnableMmap()

	got, err := r.ReadEntry(0)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if !bytes.Equal(got, data) {
		t.Error("mmap compressed data mismatch")
	}
}

// ============================================================
// Concurrent Read Tests
// ============================================================

func TestShardConcurrentReads(t *testing.T) {
	path := filepath.Join(t.TempDir(), "concurrent.shard")

	entries := 20
	w, _ := NewShardWriter(path, ShardRoleMoSH)
	for i := 0; i < entries; i++ {
		data := bytes.Repeat([]byte{byte(i)}, 100)
		w.WriteEntry(strings.Repeat(string(rune('A'+i)), 5), data)
	}
	w.Close()

	r, _ := OpenShard(path)
	defer r.Close()

	var wg sync.WaitGroup
	errs := make(chan error, entries*4)

	for goroutine := 0; goroutine < 4; goroutine++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < entries; i++ {
				data, err := r.ReadEntry(i)
				if err != nil {
					errs <- err
					return
				}
				if len(data) != 100 {
					errs <- err
					return
				}
				for _, b := range data {
					if b != byte(i) {
						errs <- err
						return
					}
				}
			}
		}()
	}

	wg.Wait()
	close(errs)
	for err := range errs {
		t.Errorf("concurrent read error: %v", err)
	}
}

func TestShardConcurrentMmapReads(t *testing.T) {
	path := filepath.Join(t.TempDir(), "concurrent-mmap.shard")

	entries := 20
	w, _ := NewShardWriter(path, ShardRoleMoSH)
	for i := 0; i < entries; i++ {
		data := bytes.Repeat([]byte{byte(i)}, 100)
		w.WriteEntry(strings.Repeat(string(rune('A'+i)), 5), data)
	}
	w.Close()

	r, _ := OpenShard(path)
	defer r.Close()
	r.EnableMmap()

	var wg sync.WaitGroup
	errs := make(chan error, entries*4)

	for goroutine := 0; goroutine < 4; goroutine++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < entries; i++ {
				data, err := r.ReadEntry(i)
				if err != nil {
					errs <- err
					return
				}
				if len(data) != 100 || data[0] != byte(i) {
					errs <- err
					return
				}
			}
		}()
	}

	wg.Wait()
	close(errs)
	for err := range errs {
		t.Errorf("concurrent mmap error: %v", err)
	}
}

// ============================================================
// Stream Writer Tests
// ============================================================

func TestShardStreamWriter(t *testing.T) {
	path := filepath.Join(t.TempDir(), "stream.shard")

	sw, err := NewShardStreamWriter(path, ShardRoleMoSH, 100)
	if err != nil {
		t.Fatalf("create stream writer: %v", err)
	}

	sw.SetAlignment(Align64)
	if err := sw.BeginData(); err != nil {
		t.Fatalf("begin data: %v", err)
	}

	for i := 0; i < 50; i++ {
		data := make([]byte, 128)
		for j := range data {
			data[j] = byte(i)
		}
		if err := sw.WriteEntry("entry_"+string(rune('A'+i%26))+"_"+string(rune('0'+i/26)), data); err != nil {
			t.Fatalf("write entry %d: %v", i, err)
		}
	}

	if err := sw.Finalize(); err != nil {
		t.Fatalf("finalize: %v", err)
	}

	// Read back
	r, err := OpenShard(path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer r.Close()

	if r.EntryCount() != 50 {
		t.Errorf("expected 50 entries, got %d", r.EntryCount())
	}

	// Verify streaming flag
	if r.Header().Flags&ShardFlagStreaming == 0 {
		t.Error("expected streaming flag set")
	}
}

func TestShardStreamWriterCompressed(t *testing.T) {
	path := filepath.Join(t.TempDir(), "stream-compressed.shard")

	sw, _ := NewShardStreamWriter(path, ShardRoleMoSH, 10)
	sw.SetCompression(CompressZstd)
	sw.BeginData()

	data := bytes.Repeat([]byte("stream compress test "), 100)
	sw.WriteEntryCompressed("data", data)
	sw.Finalize()

	r, _ := OpenShard(path)
	defer r.Close()

	got, err := r.ReadEntry(0)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if !bytes.Equal(got, data) {
		t.Error("stream compressed data mismatch")
	}
}

func TestShardStreamWriterContentType(t *testing.T) {
	path := filepath.Join(t.TempDir(), "stream-typed.shard")

	sw, _ := NewShardStreamWriter(path, ShardRoleMoSH, 10)
	sw.BeginData()
	sw.WriteEntryTyped("data", []byte("hello"), ContentTypeText)
	sw.Finalize()

	r, _ := OpenShard(path)
	defer r.Close()

	info := r.GetEntryInfo(0)
	if info.ContentType() != ContentTypeText {
		t.Errorf("content type: got %d, want %d", info.ContentType(), ContentTypeText)
	}
}

// ============================================================
// Metadata Tests
// ============================================================

func TestShardMetadataRoundtrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "meta.shard")

	w, _ := NewShardWriter(path, ShardRoleMoSH)
	meta := &ShardMetadata{
		SchemaVersion: "shard-v2.1",
		Producer:      "test",
		Description:   "unit test shard",
		Tags:          []string{"test", "v2"},
	}
	w.SetMetadata(meta)
	w.WriteEntry("data", []byte("test"))
	w.Close()

	r, _ := OpenShard(path)
	defer r.Close()

	restored, err := r.ReadMetadata()
	if err != nil {
		t.Fatalf("read metadata: %v", err)
	}
	if restored == nil {
		t.Fatal("metadata is nil")
	}
	if restored.Producer != "test" {
		t.Errorf("producer: %q", restored.Producer)
	}
	if restored.Description != "unit test shard" {
		t.Errorf("description: %q", restored.Description)
	}
	if len(restored.Tags) != 2 {
		t.Errorf("tags: %v", restored.Tags)
	}
}

func TestShardMetadataNoMetadata(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nometa.shard")

	w, _ := NewShardWriter(path, ShardRoleMoSH)
	w.WriteEntry("data", []byte("no metadata"))
	w.Close()

	r, _ := OpenShard(path)
	defer r.Close()

	meta, err := r.ReadMetadata()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if meta != nil {
		t.Error("expected nil metadata")
	}
}

func TestShardMetadataProfilesRoundtrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "profiles.shard")

	w, _ := NewShardWriter(path, ShardRoleSample)
	meta := NewShardMetadata()
	meta.SetEntryMeta("samples/0", &EntryMeta{
		ContentType:       "application/cowrie",
		Codec:             "cowrie-gen2",
		CodecVersion:      "2",
		SchemaFingerprint: "sha256:weights",
		SemanticType:      "sample",
		CanonicalHash:     "sha256:canonical",
		BaseHash:          "sha256:base",
		RowCount:          1,
		Shape:             []int64{1, 28, 28},
		Stats:             map[string]any{"mean": 0.1307, "std": 0.3081},
	})
	meta.SetSampleProfile(&SampleProfile{
		DatasetName:   "mnist-train",
		SampleIDType:  "uint64",
		KeyEncoding:   "decimal-string",
		SampleCount:   60000,
		DatasetSchema: map[string]any{"input": "tensor[u8,28,28]", "target": "uint8"},
		Splits:        map[string]any{"train": map[string]any{"start": 0, "end": 59999}},
		LabelMap:      map[string]any{"0": "zero", "1": "one"},
		FeatureStats:  map[string]any{"input": map[string]any{"mean": 0.1307}},
	})
	meta.Manifest = &ManifestMeta{
		Files: []*ManifestFileRef{
			{
				URI:        "s3://bucket/train-000.smpl",
				SHA256:     "abc123",
				Role:       "sample",
				Profile:    "sampleshard.v1",
				StartKey:   "0",
				EndKey:     "59999",
				EntryCount: 60000,
			},
		},
		Partitions: map[string]any{"train": []string{"train-000.smpl"}},
	}
	w.SetMetadata(meta)
	w.WriteEntry("samples/0", []byte("test"))
	w.Close()

	r, _ := OpenShard(path)
	defer r.Close()

	restored, err := r.ReadMetadata()
	if err != nil {
		t.Fatalf("read metadata: %v", err)
	}
	if restored == nil {
		t.Fatal("metadata is nil")
	}
	entryMeta := restored.GetEntryMeta("samples/0")
	if entryMeta == nil {
		t.Fatal("entry metadata missing")
	}
	if entryMeta.Codec != "cowrie-gen2" || entryMeta.SchemaFingerprint != "sha256:weights" {
		t.Fatalf("entry descriptor lost: %+v", entryMeta)
	}
	if entryMeta.RowCount != 1 {
		t.Fatalf("row_count = %d", entryMeta.RowCount)
	}
	if len(entryMeta.Shape) != 3 || entryMeta.Shape[0] != 1 || entryMeta.Shape[1] != 28 {
		t.Fatalf("shape = %v", entryMeta.Shape)
	}
	if restored.Profile != "sampleshard.v1" {
		t.Fatalf("profile = %q", restored.Profile)
	}
	if restored.SampleShard == nil || restored.SampleShard.DatasetName != "mnist-train" {
		t.Fatalf("sample profile lost: %+v", restored.SampleShard)
	}
	if restored.SampleShard.KeyEncoding != "decimal-string" {
		t.Fatalf("key encoding = %q", restored.SampleShard.KeyEncoding)
	}
	if restored.Manifest == nil || len(restored.Manifest.Files) != 1 {
		t.Fatalf("manifest lost: %+v", restored.Manifest)
	}
	if restored.Manifest.Files[0].URI != "s3://bucket/train-000.smpl" {
		t.Fatalf("manifest uri = %q", restored.Manifest.Files[0].URI)
	}
}

func TestShardWriterBackfillsSampleProfileDefaults(t *testing.T) {
	path := filepath.Join(t.TempDir(), "sample-profile-defaults.shard")

	w, _ := NewShardWriter(path, ShardRoleSample)
	meta := NewShardMetadata()
	meta.SetSampleProfile(&SampleProfile{
		DatasetName: "mnist-train",
	})
	w.SetMetadata(meta)
	if err := w.WriteEntry("samples/0", []byte("test")); err != nil {
		t.Fatalf("write entry: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("close writer: %v", err)
	}

	r, err := OpenShard(path)
	if err != nil {
		t.Fatalf("open shard: %v", err)
	}
	defer r.Close()

	restored, err := r.ReadMetadata()
	if err != nil {
		t.Fatalf("read metadata: %v", err)
	}
	if restored == nil || restored.SampleShard == nil {
		t.Fatalf("sample profile missing: %+v", restored)
	}
	if restored.SampleShard.SampleIDType != "uint64" {
		t.Fatalf("sample id type = %q", restored.SampleShard.SampleIDType)
	}
	if restored.SampleShard.KeyEncoding != "decimal-string" {
		t.Fatalf("key encoding = %q", restored.SampleShard.KeyEncoding)
	}
	if restored.SampleShard.SampleCount != 1 {
		t.Fatalf("sample count = %d", restored.SampleShard.SampleCount)
	}
}

func TestShardMetadataEntryMetaOperations(t *testing.T) {
	m := NewShardMetadata()

	// GetEntryMeta on empty
	if m.GetEntryMeta("nonexistent") != nil {
		t.Error("expected nil for nonexistent")
	}

	m.SetEntryMeta("a", &EntryMeta{Codec: "cowrie"})
	if em := m.GetEntryMeta("a"); em == nil || em.Codec != "cowrie" {
		t.Errorf("entry meta: %+v", em)
	}

	// AddTag dedup
	m.AddTag("train")
	m.AddTag("train")
	m.AddTag("v2")
	if len(m.Tags) != 2 {
		t.Errorf("expected 2 tags, got %d", len(m.Tags))
	}
}

func TestShardMetadataMarshalUnmarshal(t *testing.T) {
	m := NewShardMetadata()
	m.Producer = "test"
	m.Tags = []string{"a", "b"}
	m.SetEntryMeta("x", &EntryMeta{Codec: "cowrie", Shape: []int64{3, 4}})

	data, err := m.Marshal()
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	m2, err := ParseShardMetadata(data)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if m2.Producer != "test" {
		t.Errorf("producer: %q", m2.Producer)
	}
	if len(m2.Tags) != 2 {
		t.Errorf("tags: %v", m2.Tags)
	}
	em := m2.GetEntryMeta("x")
	if em == nil || em.Codec != "cowrie" || len(em.Shape) != 2 {
		t.Errorf("entry meta: %+v", em)
	}
}

func TestShardMetadataGetEntryMetaNilMap(t *testing.T) {
	m := &ShardMetadata{} // no EntryMetadata map
	if m.GetEntryMeta("anything") != nil {
		t.Error("expected nil from nil map")
	}
}

// ============================================================
// Tag Dictionary & Schema ID Tests (Tier 1)
// ============================================================

func TestShardTagDictionaryRoundtrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tagdict.shard")

	w, _ := NewShardWriter(path, ShardRoleMoSH)
	meta := NewShardMetadata()
	meta.TagDictionary = []string{"train", "eval", "frozen", "quantized"}
	w.SetMetadata(meta)
	w.WriteEntry("data", []byte("x"))
	w.Close()

	r, _ := OpenShard(path)
	defer r.Close()

	restored, _ := r.ReadMetadata()
	if restored == nil {
		t.Fatal("metadata is nil")
	}
	if len(restored.TagDictionary) != 4 {
		t.Fatalf("tag dictionary: %v", restored.TagDictionary)
	}
	if restored.TagDictionary[0] != "train" || restored.TagDictionary[3] != "quantized" {
		t.Errorf("tag dictionary: %v", restored.TagDictionary)
	}
}

func TestShardSchemaIDRoundtrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "schemaid.shard")

	w, _ := NewShardWriter(path, ShardRoleMoSH)
	meta := NewShardMetadata()
	meta.TagDictionary = []string{"train", "eval"}
	meta.SetEntryMeta("weights", &EntryMeta{ContentType: "tensor", SemanticType: "embedding"})
	meta.SchemaID = meta.ComputeSchemaID()
	w.SetMetadata(meta)
	w.WriteEntry("weights", []byte("w"))
	w.Close()

	r, _ := OpenShard(path)
	defer r.Close()

	restored, _ := r.ReadMetadata()
	if restored.SchemaID == "" {
		t.Fatal("schema ID missing")
	}
	if len(restored.SchemaID) != 16 {
		t.Errorf("schema ID length: %d", len(restored.SchemaID))
	}
	if restored.SchemaID != meta.SchemaID {
		t.Errorf("schema ID mismatch: got %q, want %q", restored.SchemaID, meta.SchemaID)
	}
}

func TestShardComputeSchemaIDDeterministic(t *testing.T) {
	m1 := NewShardMetadata()
	m1.TagDictionary = []string{"a", "b"}
	m1.SetEntryMeta("x", &EntryMeta{ContentType: "json", SemanticType: "config"})
	m1.SetEntryMeta("y", &EntryMeta{ContentType: "tensor", SemanticType: "weight"})

	m2 := NewShardMetadata()
	m2.TagDictionary = []string{"a", "b"}
	m2.SetEntryMeta("y", &EntryMeta{ContentType: "tensor", SemanticType: "weight"})
	m2.SetEntryMeta("x", &EntryMeta{ContentType: "json", SemanticType: "config"})

	if m1.ComputeSchemaID() != m2.ComputeSchemaID() {
		t.Error("same schema shape should produce same ID regardless of insertion order")
	}
}

func TestShardComputeSchemaIDDiffers(t *testing.T) {
	m1 := NewShardMetadata()
	m1.TagDictionary = []string{"a"}
	m1.SetEntryMeta("x", &EntryMeta{ContentType: "json"})

	m2 := NewShardMetadata()
	m2.TagDictionary = []string{"b"}
	m2.SetEntryMeta("x", &EntryMeta{ContentType: "json"})

	if m1.ComputeSchemaID() == m2.ComputeSchemaID() {
		t.Error("different tag dictionaries should produce different IDs")
	}
}

// ============================================================
// ListWithTag / ListWithTagBit / ListWithTagFast Tests
// ============================================================

func TestShardListWithTag(t *testing.T) {
	path := filepath.Join(t.TempDir(), "listtag.shard")

	w, _ := NewShardWriter(path, ShardRoleMoSH)
	meta := NewShardMetadata()
	meta.SetEntryMeta("layer0/weight", &EntryMeta{Tags: []string{"trainable", "float32"}})
	meta.SetEntryMeta("layer0/bias", &EntryMeta{Tags: []string{"trainable"}})
	meta.SetEntryMeta("config", &EntryMeta{Tags: []string{"readonly"}})
	w.SetMetadata(meta)
	w.WriteEntry("layer0/weight", []byte("w"))
	w.WriteEntry("layer0/bias", []byte("b"))
	w.WriteEntry("config", []byte("c"))
	w.Close()

	r, _ := OpenShard(path)
	defer r.Close()

	trainable, err := r.ListWithTag("trainable")
	if err != nil {
		t.Fatalf("ListWithTag: %v", err)
	}
	sort.Strings(trainable)
	if len(trainable) != 2 {
		t.Fatalf("expected 2 trainable, got %d: %v", len(trainable), trainable)
	}
	if trainable[0] != "layer0/bias" || trainable[1] != "layer0/weight" {
		t.Errorf("trainable: %v", trainable)
	}

	readonly, _ := r.ListWithTag("readonly")
	if len(readonly) != 1 || readonly[0] != "config" {
		t.Errorf("readonly: %v", readonly)
	}

	missing, _ := r.ListWithTag("nonexistent")
	if len(missing) != 0 {
		t.Errorf("expected 0 for nonexistent tag, got %d", len(missing))
	}
}

func TestShardListWithTagNoMetadata(t *testing.T) {
	path := filepath.Join(t.TempDir(), "notag.shard")

	w, _ := NewShardWriter(path, ShardRoleMoSH)
	w.WriteEntry("x", []byte("y"))
	w.Close()

	r, _ := OpenShard(path)
	defer r.Close()

	result, err := r.ListWithTag("anything")
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if result != nil {
		t.Errorf("expected nil, got %v", result)
	}
}

func TestShardListWithTagBit(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tagbit.shard")

	w, _ := NewShardWriter(path, ShardRoleMoSH)
	// Write entries, then we'll manipulate tag bits via the reader
	w.WriteEntry("a", []byte("1"))
	w.WriteEntry("b", []byte("2"))
	w.WriteEntry("c", []byte("3"))
	w.Close()

	// Directly set tag bits in the file by rewriting
	// Instead, test via the writer approach — write with content type that includes tag bits
	// Actually, let's test via the reader's index manipulation:
	r, _ := OpenShard(path)
	defer r.Close()

	// Manually set tag bits on entries for testing
	r.index[0].SetTagBit(0) // "a" has bit 0
	r.index[0].SetTagBit(2) // "a" has bit 2
	r.index[1].SetTagBit(0) // "b" has bit 0
	r.index[2].SetTagBit(1) // "c" has bit 1

	bit0 := r.ListWithTagBit(0)
	sort.Strings(bit0)
	if len(bit0) != 2 {
		t.Fatalf("bit0: %v", bit0)
	}
	if bit0[0] != "a" || bit0[1] != "b" {
		t.Errorf("bit0: %v", bit0)
	}

	bit1 := r.ListWithTagBit(1)
	if len(bit1) != 1 || bit1[0] != "c" {
		t.Errorf("bit1: %v", bit1)
	}

	bit2 := r.ListWithTagBit(2)
	if len(bit2) != 1 || bit2[0] != "a" {
		t.Errorf("bit2: %v", bit2)
	}

	// Out of range
	if r.ListWithTagBit(-1) != nil {
		t.Error("expected nil for -1")
	}
	if r.ListWithTagBit(16) != nil {
		t.Error("expected nil for 16")
	}
}

func TestShardListWithTagFast(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tagfast.shard")

	w, _ := NewShardWriter(path, ShardRoleMoSH)
	meta := NewShardMetadata()
	meta.TagDictionary = []string{"train", "eval", "frozen"}
	w.SetMetadata(meta)
	w.WriteEntry("a", []byte("1"))
	w.WriteEntry("b", []byte("2"))
	w.WriteEntry("c", []byte("3"))
	w.Close()

	r, _ := OpenShard(path)
	defer r.Close()

	// Set tag bits matching dictionary
	r.index[0].SetTagBit(0) // "a" = train
	r.index[1].SetTagBit(0) // "b" = train
	r.index[1].SetTagBit(1) // "b" = eval
	r.index[2].SetTagBit(2) // "c" = frozen

	train, err := r.ListWithTagFast("train")
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	sort.Strings(train)
	if len(train) != 2 || train[0] != "a" || train[1] != "b" {
		t.Errorf("train: %v", train)
	}

	frozen, _ := r.ListWithTagFast("frozen")
	if len(frozen) != 1 || frozen[0] != "c" {
		t.Errorf("frozen: %v", frozen)
	}

	// Tag not in dictionary
	unknown, _ := r.ListWithTagFast("nonexistent")
	if unknown != nil {
		t.Errorf("expected nil for unknown tag, got %v", unknown)
	}
}

func TestShardListWithTagFastNoMetadata(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tagfast-nometa.shard")

	w, _ := NewShardWriter(path, ShardRoleMoSH)
	w.WriteEntry("x", []byte("y"))
	w.Close()

	r, _ := OpenShard(path)
	defer r.Close()

	result, err := r.ListWithTagFast("anything")
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if result != nil {
		t.Errorf("expected nil, got %v", result)
	}
}

// ============================================================
// Writer State Tests
// ============================================================

func TestShardWriterDoubleClose(t *testing.T) {
	path := filepath.Join(t.TempDir(), "dclose.shard")

	w, _ := NewShardWriter(path, ShardRoleMoSH)
	w.WriteEntry("x", []byte("y"))
	w.Close()

	err := w.Close()
	if err == nil {
		t.Error("expected error on double close")
	}
}

func TestShardWriterWriteAfterClose(t *testing.T) {
	path := filepath.Join(t.TempDir(), "wac.shard")

	w, _ := NewShardWriter(path, ShardRoleMoSH)
	w.Close()

	err := w.WriteEntry("x", []byte("y"))
	if err == nil {
		t.Error("expected error writing after close")
	}
}

func TestShardReaderReadAfterClose(t *testing.T) {
	path := filepath.Join(t.TempDir(), "rac.shard")

	w, _ := NewShardWriter(path, ShardRoleMoSH)
	w.WriteEntry("x", []byte("y"))
	w.Close()

	r, _ := OpenShard(path)
	r.Close()

	_, err := r.ReadEntry(0)
	if err == nil {
		t.Error("expected error reading after close")
	}
}

func TestShardReaderDoubleClose(t *testing.T) {
	path := filepath.Join(t.TempDir(), "rdc.shard")

	w, _ := NewShardWriter(path, ShardRoleMoSH)
	w.WriteEntry("x", []byte("y"))
	w.Close()

	r, _ := OpenShard(path)
	r.Close()
	// Second close should be no-op
	err := r.Close()
	if err != nil {
		t.Errorf("double close: %v", err)
	}
}

// ============================================================
// Header Field Roundtrip Tests
// ============================================================

func TestShardHeaderFieldsRoundtrip(t *testing.T) {
	h := &ShardHeader{
		Magic:              ShardMagic,
		Version:            ShardVersion2,
		Role:               ShardRoleSample,
		Flags:              0x00F3,
		Alignment:          Align32,
		CompressionDefault: CompressLZ4,
		IndexEntrySize:     ShardIndexEntrySize,
		EntryCount:         12345,
		StringTableOffset:  0x100,
		DataSectionOffset:  0x200,
		SchemaOffset:       0x300,
		TotalFileSize:      0x400,
	}

	var buf bytes.Buffer
	WriteShardHeader(&buf, h)
	h2, err := ReadShardHeader(&buf)
	if err != nil {
		t.Fatalf("read: %v", err)
	}

	if h2.Role != ShardRoleSample {
		t.Errorf("role: %d", h2.Role)
	}
	if h2.Flags != 0x00F3 {
		t.Errorf("flags: 0x%04x", h2.Flags)
	}
	if h2.CompressionDefault != CompressLZ4 {
		t.Errorf("compression: %d", h2.CompressionDefault)
	}
	if h2.EntryCount != 12345 {
		t.Errorf("entry count: %d", h2.EntryCount)
	}
	if h2.StringTableOffset != 0x100 {
		t.Errorf("string table offset: %d", h2.StringTableOffset)
	}
	if h2.SchemaOffset != 0x300 {
		t.Errorf("schema offset: %d", h2.SchemaOffset)
	}
	if h2.TotalFileSize != 0x400 {
		t.Errorf("total file size: %d", h2.TotalFileSize)
	}
}

// ============================================================
// Checksum Computation Tests
// ============================================================

func TestComputeChecksum(t *testing.T) {
	// CRC32C of empty data
	crc := ComputeChecksum([]byte{})
	if crc != 0 {
		t.Errorf("CRC of empty: 0x%08x (expected 0)", crc)
	}

	// CRC32C of known data should be deterministic
	data := []byte("hello")
	crc1 := ComputeChecksum(data)
	crc2 := ComputeChecksum(data)
	if crc1 != crc2 {
		t.Error("CRC not deterministic")
	}

	// Different data should produce different CRC
	crc3 := ComputeChecksum([]byte("world"))
	if crc1 == crc3 {
		t.Error("different data produced same CRC")
	}
}

// ============================================================
// xxHash Tests
// ============================================================

func TestXxHash64Deterministic(t *testing.T) {
	h1 := xxHash64String("test")
	h2 := xxHash64String("test")
	if h1 != h2 {
		t.Error("xxHash not deterministic")
	}

	h3 := xxHash64String("other")
	if h1 == h3 {
		t.Error("collision on simple strings")
	}
}

// ============================================================
// Fuzz-like Random Data Tests
// ============================================================

func TestShardRandomData(t *testing.T) {
	path := filepath.Join(t.TempDir(), "random.shard")
	rng := rand.New(rand.NewSource(42))

	w, _ := NewShardWriter(path, ShardRoleMoSH)
	w.SetCompression(CompressZstd)

	entries := make(map[string][]byte)
	for i := 0; i < 50; i++ {
		size := rng.Intn(10000)
		data := make([]byte, size)
		rng.Read(data)
		name := strings.Repeat(string(rune('a'+i%26)), rng.Intn(10)+1)

		// Ensure unique names
		for _, ok := entries[name]; ok; _, ok = entries[name] {
			name += "x"
		}

		entries[name] = data
		if rng.Float32() < 0.5 {
			w.WriteEntryCompressed(name, data)
		} else {
			w.WriteEntry(name, data)
		}
	}
	w.Close()

	r, _ := OpenShard(path)
	defer r.Close()

	if r.EntryCount() != 50 {
		t.Fatalf("expected 50 entries, got %d", r.EntryCount())
	}

	for i := 0; i < r.EntryCount(); i++ {
		name := r.EntryName(i)
		expected, ok := entries[name]
		if !ok {
			t.Fatalf("unexpected entry name: %q", name)
		}
		data, err := r.ReadEntry(i)
		if err != nil {
			t.Fatalf("read %q: %v", name, err)
		}
		if !bytes.Equal(data, expected) {
			t.Fatalf("data mismatch for %q: got %d bytes, want %d", name, len(data), len(expected))
		}
	}
}

// ============================================================
// Content Type on Writer Tests
// ============================================================

func TestShardAllContentTypes(t *testing.T) {
	path := filepath.Join(t.TempDir(), "alltypes.shard")

	types := []uint16{
		ContentTypeTensor, ContentTypeJSON, ContentTypeCowrie,
		ContentTypeGLYPH, ContentTypeText, ContentTypeImage,
		ContentTypeAudio, ContentTypeVideo, ContentTypeProto, ContentTypeBlob,
	}

	w, _ := NewShardWriter(path, ShardRoleMoSH)
	for i, ct := range types {
		name := ContentTypeName(ct)
		w.WriteEntryTyped(name, []byte{byte(i)}, ct)
	}
	w.Close()

	r, _ := OpenShard(path)
	defer r.Close()

	for i, ct := range types {
		info := r.GetEntryInfo(i)
		if info.ContentType() != ct {
			t.Errorf("entry %d: expected %s, got %s", i, ContentTypeName(ct), ContentTypeName(info.ContentType()))
		}
	}

	// Verify content types flag is set
	if r.Header().Flags&ShardFlagHasContentTypes == 0 {
		t.Error("expected ShardFlagHasContentTypes flag")
	}
}

func TestShardUserContentType(t *testing.T) {
	path := filepath.Join(t.TempDir(), "usertype.shard")

	ct := ContentTypeUserBase + 42

	w, _ := NewShardWriter(path, ShardRoleMoSH)
	w.WriteEntryTyped("custom", []byte("data"), ct)
	w.Close()

	r, _ := OpenShard(path)
	defer r.Close()

	info := r.GetEntryInfo(0)
	if info.ContentType() != ct {
		t.Errorf("user content type: got %d, want %d", info.ContentType(), ct)
	}
	if ContentTypeName(ct) != "user:42" {
		t.Errorf("user content type name: %q", ContentTypeName(ct))
	}
}

// ============================================================
// Role String Test
// ============================================================

func TestShardRoleString(t *testing.T) {
	tests := map[ShardRole]string{
		ShardRoleMoSH:      "MoSH",
		ShardRoleSample:    "Sample",
		ShardRoleGemmPanel: "GemmPanel",
		ShardRoleManifest:  "Manifest",
		ShardRoleWShard:    "WShard",
	}
	for role, want := range tests {
		if got := role.String(); got != want {
			t.Errorf("ShardRole(%d).String() = %q, want %q", role, got, want)
		}
	}
}

// ============================================================
// File Size Validation Tests
// ============================================================

func TestShardTotalFileSizeValidated(t *testing.T) {
	path := filepath.Join(t.TempDir(), "filesize.shard")

	w, _ := NewShardWriter(path, ShardRoleMoSH)
	w.WriteEntry("data", []byte("hello"))
	w.Close()

	// Verify TotalFileSize matches
	r, _ := OpenShard(path)
	info, _ := r.file.Stat()
	if r.Header().TotalFileSize != uint64(info.Size()) {
		t.Errorf("TotalFileSize %d != actual %d", r.Header().TotalFileSize, info.Size())
	}
	r.Close()

	// Append extra bytes to corrupt file size check
	raw, _ := os.ReadFile(path)
	raw = append(raw, 0, 0, 0, 0)
	os.WriteFile(path, raw, 0o644)

	_, err := OpenShard(path)
	if err == nil {
		t.Error("expected error for mismatched file size")
	}
}
