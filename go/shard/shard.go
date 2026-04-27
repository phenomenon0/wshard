// Package shard provides the unified Shard container format.
//
// # What is a Shard?
//
// A Shard is a binary container format for storing named binary entries with
// fast random access. Think of it as a "ZIP file optimized for tensors" - but
// with mmap support, 64-byte alignment for SIMD, and streaming writes.
//
// Key properties:
//   - Named entries: Each entry has a string name (e.g., "layer.0.weight")
//   - Binary blobs: Entries are opaque byte arrays (typically TensorV1-encoded)
//   - Fast lookup: O(1) hash-based lookup by name (xxHash64)
//   - Streaming: Index at start enables single-pass writes (no seek-back)
//   - Aligned: 64-byte alignment for AVX-512/SIMD efficiency
//   - Compressed: Optional per-entry zstd/lz4 compression
//   - Checksummed: CRC32C for data integrity
//
// # Shard vs Cowrie vs GLYPH
//
//	Cowrie  = Value codec (like JSON, but binary with tensors)
//	GLYPH  = Value codec (like JSON, but text optimized for LLMs)
//	Shard  = Container format (stores multiple named blobs)
//
// Analogy:
//   - Cowrie/GLYPH are like individual "files"
//   - Shard is like a "directory" or "archive" containing many files
//
// A Shard typically contains Cowrie-encoded or TensorV1-encoded entries,
// but the format is agnostic - entries can be any binary data.
//
// # Roles (Profiles)
//
// Shards have a "role" byte that hints at the intended use case:
//
//	MoSH (role=1)        - Model weights keyed by layer name
//	SampleShard (role=2) - Training samples keyed by sample ID
//	GemmPanel (role=3)   - BLIS-style packed matrix panels for distributed GEMM
//
// The role is metadata only - it doesn't change the container format.
//
// # File Layout (v2)
//
//	┌─────────────────────────────────────┐
//	│ Header (64 bytes)                   │
//	│   magic: "SHRD"                     │
//	│   version: 0x02                     │
//	│   role: 1=MoSH, 2=Sample, etc.      │
//	│   alignment: 64                     │
//	│   entry_count, offsets, etc.        │
//	├─────────────────────────────────────┤
//	│ Index Section (48 bytes × N)        │
//	│   per entry: name_hash, offsets,    │
//	│              sizes, checksum        │
//	├─────────────────────────────────────┤
//	│ String Table (variable)             │
//	│   null-terminated entry names       │
//	├─────────────────────────────────────┤
//	│ [Schema Section - optional]         │
//	├─────────────────────────────────────┤
//	│ Data Section (aligned)              │
//	│   entry 0 data (64-byte aligned)    │
//	│   entry 1 data (64-byte aligned)    │
//	│   ...                               │
//	└─────────────────────────────────────┘
//
// # Usage Examples
//
// Writing a shard:
//
//	w, _ := NewShardWriter("model.shard", ShardRoleMoSH)
//	w.SetAlignment(64)
//	w.SetCompression(CompressZstd)
//	w.WriteEntry("layer.0.weight", tensorBytes)
//	w.WriteEntryCompressed("layer.1.weight", tensorBytes)
//	w.Close()
//
// Reading a shard:
//
//	r, _ := OpenShard("model.shard")
//	data, _ := r.ReadEntryByName("layer.0.weight")
//	r.Close()
//
// Streaming write (for very large models):
//
//	sw, _ := NewShardStreamWriter("model.shard", ShardRoleMoSH, 1000)
//	sw.BeginData()
//	sw.WriteEntry("layer.0.weight", tensorBytes)
//	sw.Finalize()
//
// # See Also
//
//   - shard_format.go: v2 implementation
//   - mosh.go: MoSH-specific wrapper for model weights
//   - tensor_v1.go: TensorV1 encoding for entry data
//   - schema.go: Optional schema validation
package shard

import (
	"errors"
	"fmt"
)

// Shard magic bytes
var ShardMagic = [4]byte{'S', 'H', 'R', 'D'}

// ShardRole identifies the profile/type of shard.
type ShardRole uint8

const (
	ShardRoleUnknown   ShardRole = 0x00
	ShardRoleMoSH      ShardRole = 0x01 // MoSH: Model Shard - model weights (keyed by layer name)
	ShardRoleSample    ShardRole = 0x02 // Training samples (keyed by sample ID)
	ShardRoleGemmPanel ShardRole = 0x03 // GEMM panels (packed A/B tiles for BLIS)
	ShardRoleManifest  ShardRole = 0x04 // Multi-file manifest (references other shards)
	ShardRoleWShard    ShardRole = 0x05 // W-SHARD: World-model episode data
	ShardRoleUMSH      ShardRole = 0x06 // UMSH: Universal Model Shard - variant of MoSH
	ShardRoleColumn    ShardRole = 0x08 // ColumnShard: columnar tabular data
)

// String returns a human-readable role name.
func (r ShardRole) String() string {
	switch r {
	case ShardRoleMoSH:
		return "MoSH"
	case ShardRoleSample:
		return "Sample"
	case ShardRoleGemmPanel:
		return "GemmPanel"
	case ShardRoleManifest:
		return "Manifest"
	case ShardRoleWShard:
		return "WShard"
	case ShardRoleUMSH:
		return "UMSH"
	case ShardRoleColumn:
		return "ColumnShard"
	default:
		return fmt.Sprintf("Unknown(0x%02x)", uint8(r))
	}
}

// ShardFlags contains optional flags for the shard.
type ShardFlags uint16

const (
	ShardFlagCompressed   ShardFlags = 0x0001 // Data is compressed
	ShardFlagLittleEndian ShardFlags = 0x0002 // Explicit little-endian (default)
	ShardFlagBigEndian    ShardFlags = 0x0004 // Big-endian data
	// Reserved: 0x0008 - 0x8000
)

// Errors
var (
	ErrInvalidShardMagic  = errors.New("shard: invalid magic bytes")
	ErrUnsupportedVersion = errors.New("shard: unsupported version")
	ErrInvalidShardHeader = errors.New("shard: invalid header")
	ErrShardIndexCorrupt  = errors.New("shard: index is corrupt")
	ErrEntryNotFound      = errors.New("shard: entry not found")
	ErrShardClosed        = errors.New("shard: closed")
)

// ShardIndex is the interface for profile-specific indexes.
type ShardIndex interface {
	// Len returns the number of entries in the index.
	Len() int

	// EntryOffset returns the byte offset of entry i in the data section.
	EntryOffset(i int) int64

	// EntrySize returns the byte size of entry i.
	EntrySize(i int) int64
}
