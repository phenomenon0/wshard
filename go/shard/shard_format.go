// Shard implementation.
//
// See shard.go for full documentation on what a Shard is.
//
// # Header Format (64 bytes)
//
//	Bytes 0-3:   magic = 'S','H','R','D'
//	Byte 4:      version (0x02)
//	Byte 5:      role (1=MoSH, 2=PTShard, 3=GemmPanel)
//	Bytes 6-7:   flags (uint16 LE)
//	Byte 8:      alignment (0=none, 16, 32, 64)
//	Byte 9:      compression_default (0=none, 1=zstd, 2=lz4)
//	Bytes 10-11: index_entry_size (uint16 LE, must be 48)
//	Bytes 12-15: entry_count (uint32 LE)
//	Bytes 16-23: string_table_offset (uint64 LE)
//	Bytes 24-31: data_section_offset (uint64 LE)
//	Bytes 32-39: schema_offset (uint64 LE, 0 if no schema)
//	Bytes 40-47: total_file_size (uint64 LE)
//	Bytes 48-63: reserved (zeroed)
//
// # Index Entry (48 bytes)
//
//	Bytes 0-7:   name_hash (xxHash64 of name)
//	Bytes 8-11:  name_offset (into string table)
//	Bytes 12-13: name_len
//	Bytes 14-15: flags (compressed, compression type)
//	Bytes 16-23: data_offset
//	Bytes 24-31: disk_size (compressed size)
//	Bytes 32-39: orig_size (uncompressed size)
//	Bytes 40-43: checksum (CRC32C of uncompressed data)
//	Bytes 44-47: reserved
package shard

import (
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"os"
	"strings"
	"sync"

	"github.com/cespare/xxhash/v2"
	"github.com/klauspost/compress/zstd"
	"github.com/pierrec/lz4/v4"
)

// ============================================================
// Hierarchical Name Path Helpers
// ============================================================

// PathSeparator is the canonical separator for hierarchical names.
const PathSeparator = "/"

// SplitPath splits a hierarchical name into components.
func SplitPath(name string) []string {
	if name == "" {
		return nil
	}
	return strings.Split(name, PathSeparator)
}

// JoinPath joins path components into a hierarchical name.
func JoinPath(parts ...string) string {
	return strings.Join(parts, PathSeparator)
}

// PathPrefix returns true if name starts with the given prefix.
func PathPrefix(name, prefix string) bool {
	if prefix == "" {
		return true
	}
	if !strings.HasSuffix(prefix, PathSeparator) {
		prefix += PathSeparator
	}
	return strings.HasPrefix(name, prefix) || name == strings.TrimSuffix(prefix, PathSeparator)
}

// PathParent returns the parent path (everything before last "/").
func PathParent(name string) string {
	idx := strings.LastIndex(name, PathSeparator)
	if idx < 0 {
		return ""
	}
	return name[:idx]
}

// PathBase returns the base name (everything after last "/").
func PathBase(name string) string {
	idx := strings.LastIndex(name, PathSeparator)
	if idx < 0 {
		return name
	}
	return name[idx+1:]
}

// Version constants
const (
	ShardVersion2 = 0x02
)

// Header sizes
const (
	ShardHeaderSize     = 64
	ShardIndexEntrySize = 48 // Fixed size per index entry
)

// Alignment values
const (
	AlignNone uint8 = 0
	Align16   uint8 = 16
	Align32   uint8 = 32
	Align64   uint8 = 64
)

// Compression types
const (
	CompressNone uint8 = 0
	CompressZstd uint8 = 1
	CompressLZ4  uint8 = 2
)

// Content types for entry.Reserved[0:2] (lower 16 bits)
const (
	ContentTypeUnknown uint16 = 0
	ContentTypeTensor  uint16 = 1  // TensorV1 encoded tensor
	ContentTypeJSON    uint16 = 2  // Standard JSON
	ContentTypeCowrie  uint16 = 3  // Cowrie binary format
	ContentTypeGLYPH   uint16 = 4  // GLYPH text format
	ContentTypeText    uint16 = 5  // Plain text (UTF-8)
	ContentTypeImage   uint16 = 6  // Image (PNG, JPEG, etc.)
	ContentTypeAudio   uint16 = 7  // Audio (WAV, MP3, etc.)
	ContentTypeVideo   uint16 = 8  // Video (MP4, WebM, etc.)
	ContentTypeProto   uint16 = 9  // Protocol Buffers
	ContentTypeBlob           uint16 = 10 // Opaque binary blob
	ContentTypeQMLN           uint16 = 11 // QMLN encoded
	ContentTypeTensorV3       uint16 = 12 // TensorV3 encoded tensor
	ContentTypeAnchorShared   uint16 = 13 // Shared anchor data
	ContentTypeDeltaExpert    uint16 = 14 // Delta expert data
	ContentTypeCodebookShared uint16 = 15 // Shared codebook
	ContentTypeExpertIndices  uint16 = 16 // Expert indices
	ContentTypeColumn        uint16 = 17 // Column chunk data
	// User-defined: >= 0x8000
	ContentTypeUserBase uint16 = 0x8000
)

// V2 flag bits
const (
	ShardFlagHasSchema       ShardFlags = 0x0010 // Schema section present
	ShardFlagHasChecksums    ShardFlags = 0x0020 // Per-entry checksums
	ShardFlagStreaming       ShardFlags = 0x0040 // Written in streaming mode
	ShardFlagHasContentTypes ShardFlags = 0x0080 // Content type hints present
)

// Security limits for shard reader
const (
	// MaxIndexSize is the maximum index section size (1GB)
	MaxIndexSize = 1 << 30
	// MaxStringTableSize is the maximum string table size (100MB)
	MaxStringTableSize = 100 * 1024 * 1024
	// MaxEntryCount is the maximum number of entries
	MaxEntryCount = 10_000_000
)

// ShardHeader represents the 64-byte v2 header.
type ShardHeader struct {
	Magic              [4]byte    // 'S','H','R','D'
	Version            uint8      // 0x02
	Role               ShardRole  // Profile selector
	Flags              ShardFlags // Feature flags
	Alignment          uint8      // Data alignment (0, 16, 32, 64)
	CompressionDefault uint8      // Default compression for new entries
	IndexEntrySize     uint16     // Size of each index entry
	EntryCount         uint32     // Number of entries
	StringTableOffset  uint64     // Offset to string table
	DataSectionOffset  uint64     // Offset to data section
	SchemaOffset       uint64     // Offset to schema (0 if none)
	TotalFileSize      uint64     // Total file size for validation
	Reserved           [16]byte   // Future use
}

// IndexEntry represents a single entry in the v2 index (48 bytes).
type IndexEntry struct {
	NameHash   uint64 // xxHash64 of name (for fast lookup)
	NameOffset uint32 // Offset into string table
	NameLen    uint16 // Length of name
	Flags      uint16 // bit0=compressed, bit1-2=compType, bit3=chunked
	DataOffset uint64 // Byte offset to entry data
	DiskSize   uint64 // Size on disk (compressed)
	OrigSize   uint64 // Original size (uncompressed)
	Checksum   uint32 // CRC32C of uncompressed data
	Reserved   uint32 // Future use
}

// Entry flag bits
const (
	EntryFlagCompressed uint16 = 0x0001
	EntryFlagZstd       uint16 = 0x0002 // Compression type bits
	EntryFlagLZ4        uint16 = 0x0004
	EntryFlagChunked    uint16 = 0x0008
)

// CRC32C table (Castagnoli polynomial, hardware accelerated)
var crc32cTable = crc32.MakeTable(crc32.Castagnoli)

// ComputeChecksum computes CRC32C checksum for data.
func ComputeChecksum(data []byte) uint32 {
	return crc32.Checksum(data, crc32cTable)
}

// Errors
var (
	ErrHeaderTooShort    = errors.New("shard: header too short")
	ErrIndexCorrupt      = errors.New("shard: index corrupt")
	ErrChecksumMismatch  = errors.New("shard: checksum mismatch")
	ErrCompressionFailed = errors.New("shard: compression failed")
	ErrInvalidAlignment  = errors.New("shard: invalid alignment (must be 0, 16, 32, or 64)")
)

// WriteShardHeader writes a v2 header to the given writer.
func WriteShardHeader(w io.Writer, h *ShardHeader) error {
	buf := make([]byte, ShardHeaderSize)

	copy(buf[0:4], h.Magic[:])
	buf[4] = h.Version
	buf[5] = uint8(h.Role)
	binary.LittleEndian.PutUint16(buf[6:8], uint16(h.Flags))
	buf[8] = h.Alignment
	buf[9] = h.CompressionDefault
	binary.LittleEndian.PutUint16(buf[10:12], h.IndexEntrySize)
	binary.LittleEndian.PutUint32(buf[12:16], h.EntryCount)
	binary.LittleEndian.PutUint64(buf[16:24], h.StringTableOffset)
	binary.LittleEndian.PutUint64(buf[24:32], h.DataSectionOffset)
	binary.LittleEndian.PutUint64(buf[32:40], h.SchemaOffset)
	binary.LittleEndian.PutUint64(buf[40:48], h.TotalFileSize)
	// bytes 48-63 are reserved (already zeroed)

	_, err := w.Write(buf)
	return err
}

// ReadShardHeader reads a v2 header from the given reader.
func ReadShardHeader(r io.Reader) (*ShardHeader, error) {
	buf := make([]byte, ShardHeaderSize)
	if _, err := io.ReadFull(r, buf); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrHeaderTooShort, err)
	}

	h := &ShardHeader{}
	copy(h.Magic[:], buf[0:4])

	if h.Magic != ShardMagic {
		return nil, fmt.Errorf("%w: got %q", ErrInvalidShardMagic, string(h.Magic[:]))
	}

	h.Version = buf[4]
	if h.Version != ShardVersion2 {
		return nil, fmt.Errorf("%w: expected v2, got v%d", ErrUnsupportedVersion, h.Version)
	}

	h.Role = ShardRole(buf[5])
	h.Flags = ShardFlags(binary.LittleEndian.Uint16(buf[6:8]))
	h.Alignment = buf[8]
	h.CompressionDefault = buf[9]
	h.IndexEntrySize = binary.LittleEndian.Uint16(buf[10:12])

	// Validate index entry size matches expected v2 size
	if h.IndexEntrySize != ShardIndexEntrySize {
		return nil, fmt.Errorf("%w: index entry size %d != expected %d",
			ErrIndexCorrupt, h.IndexEntrySize, ShardIndexEntrySize)
	}

	h.EntryCount = binary.LittleEndian.Uint32(buf[12:16])
	h.StringTableOffset = binary.LittleEndian.Uint64(buf[16:24])
	h.DataSectionOffset = binary.LittleEndian.Uint64(buf[24:32])
	h.SchemaOffset = binary.LittleEndian.Uint64(buf[32:40])
	h.TotalFileSize = binary.LittleEndian.Uint64(buf[40:48])
	copy(h.Reserved[:], buf[48:64])

	return h, nil
}

// NewShardHeader creates a new v2 header with default values.
func NewShardHeader(role ShardRole) *ShardHeader {
	return &ShardHeader{
		Magic:              ShardMagic,
		Version:            ShardVersion2,
		Role:               role,
		Flags:              ShardFlagLittleEndian | ShardFlagHasChecksums,
		Alignment:          Align64,
		CompressionDefault: CompressNone,
		IndexEntrySize:     ShardIndexEntrySize,
		EntryCount:         0,
		StringTableOffset:  0,
		DataSectionOffset:  0,
		SchemaOffset:       0,
		TotalFileSize:      0,
	}
}

// WriteIndexEntry writes a single v2 index entry.
func WriteIndexEntry(w io.Writer, e *IndexEntry) error {
	buf := make([]byte, ShardIndexEntrySize)

	binary.LittleEndian.PutUint64(buf[0:8], e.NameHash)
	binary.LittleEndian.PutUint32(buf[8:12], e.NameOffset)
	binary.LittleEndian.PutUint16(buf[12:14], e.NameLen)
	binary.LittleEndian.PutUint16(buf[14:16], e.Flags)
	binary.LittleEndian.PutUint64(buf[16:24], e.DataOffset)
	binary.LittleEndian.PutUint64(buf[24:32], e.DiskSize)
	binary.LittleEndian.PutUint64(buf[32:40], e.OrigSize)
	binary.LittleEndian.PutUint32(buf[40:44], e.Checksum)
	binary.LittleEndian.PutUint32(buf[44:48], e.Reserved)

	_, err := w.Write(buf)
	return err
}

// ReadIndexEntry reads a single v2 index entry.
func ReadIndexEntry(r io.Reader) (*IndexEntry, error) {
	buf := make([]byte, ShardIndexEntrySize)
	if _, err := io.ReadFull(r, buf); err != nil {
		return nil, err
	}

	return ParseIndexEntry(buf), nil
}

// ParseIndexEntry parses a v2 index entry from bytes.
func ParseIndexEntry(buf []byte) *IndexEntry {
	return &IndexEntry{
		NameHash:   binary.LittleEndian.Uint64(buf[0:8]),
		NameOffset: binary.LittleEndian.Uint32(buf[8:12]),
		NameLen:    binary.LittleEndian.Uint16(buf[12:14]),
		Flags:      binary.LittleEndian.Uint16(buf[14:16]),
		DataOffset: binary.LittleEndian.Uint64(buf[16:24]),
		DiskSize:   binary.LittleEndian.Uint64(buf[24:32]),
		OrigSize:   binary.LittleEndian.Uint64(buf[32:40]),
		Checksum:   binary.LittleEndian.Uint32(buf[40:44]),
		Reserved:   binary.LittleEndian.Uint32(buf[44:48]),
	}
}

// IsCompressed returns true if the entry is compressed.
func (e *IndexEntry) IsCompressed() bool {
	return e.Flags&EntryFlagCompressed != 0
}

// CompressionType returns the compression type for the entry.
func (e *IndexEntry) CompressionType() uint8 {
	if e.Flags&EntryFlagLZ4 != 0 {
		return CompressLZ4
	}
	if e.Flags&EntryFlagZstd != 0 {
		return CompressZstd
	}
	return CompressNone
}

// IsChunked returns true if the entry is chunked.
func (e *IndexEntry) IsChunked() bool {
	return e.Flags&EntryFlagChunked != 0
}

// ContentType returns the content type stored in the lower 16 bits of Reserved.
func (e *IndexEntry) ContentType() uint16 {
	return uint16(e.Reserved & 0xFFFF)
}

// SetContentType sets the content type in the lower 16 bits of Reserved.
func (e *IndexEntry) SetContentType(ct uint16) {
	e.Reserved = (e.Reserved & 0xFFFF0000) | uint32(ct)
}

// TagBits returns the 16-bit tag bitmask stored in the upper 16 bits of Reserved.
func (e *IndexEntry) TagBits() uint16 {
	return uint16(e.Reserved >> 16)
}

// SetTagBits sets the 16-bit tag bitmask in the upper 16 bits of Reserved.
func (e *IndexEntry) SetTagBits(bits uint16) {
	e.Reserved = (e.Reserved & 0x0000FFFF) | (uint32(bits) << 16)
}

// HasTagBit checks if a specific tag bit (0-15) is set.
func (e *IndexEntry) HasTagBit(bit int) bool {
	if bit < 0 || bit > 15 {
		return false
	}
	return (e.TagBits() & (1 << uint(bit))) != 0
}

// SetTagBit sets a specific tag bit (0-15).
func (e *IndexEntry) SetTagBit(bit int) {
	if bit < 0 || bit > 15 {
		return
	}
	e.SetTagBits(e.TagBits() | (1 << uint(bit)))
}

// ContentTypeName returns the human-readable name for a content type.
func ContentTypeName(ct uint16) string {
	switch ct {
	case ContentTypeTensor:
		return "tensor"
	case ContentTypeJSON:
		return "json"
	case ContentTypeCowrie:
		return "cowrie"
	case ContentTypeGLYPH:
		return "glyph"
	case ContentTypeText:
		return "text"
	case ContentTypeImage:
		return "image"
	case ContentTypeAudio:
		return "audio"
	case ContentTypeVideo:
		return "video"
	case ContentTypeProto:
		return "proto"
	case ContentTypeBlob:
		return "blob"
	case ContentTypeQMLN:
		return "qmln"
	case ContentTypeTensorV3:
		return "tensor_v3"
	case ContentTypeAnchorShared:
		return "anchor_shared"
	case ContentTypeDeltaExpert:
		return "delta_expert"
	case ContentTypeCodebookShared:
		return "codebook_shared"
	case ContentTypeExpertIndices:
		return "expert_indices"
	case ContentTypeColumn:
		return "column"
	default:
		if ct >= ContentTypeUserBase {
			return fmt.Sprintf("user:%d", ct-ContentTypeUserBase)
		}
		return "unknown"
	}
}

// xxHash64String computes xxHash64 of a string for fast index lookup.
func xxHash64String(s string) uint64 {
	return xxhash.Sum64String(s)
}

// ============================================================
// ShardWriter - Streaming writer with index at start
// ============================================================

// pendingEntry holds info about an entry being written.
// Data is stored in temp file, not RAM, to support large tensors.
type pendingEntry struct {
	name        string
	tempOffset  int64  // Offset in temp file where data starts
	diskSize    int64  // Size on disk (compressed size)
	origSize    int64  // Original uncompressed size
	checksum    uint32 // CRC32C of uncompressed data
	flags       uint16 // Entry flags (compression, etc.)
	contentType uint16 // Content type hint
}

// ShardWriter provides streaming writing of v2 shard files.
type ShardWriter struct {
	file           *os.File
	tempFile       *os.File // Temp file for data buffering
	tempFileOffset int64    // Current write position in temp file
	header         *ShardHeader
	entries        []pendingEntry
	metadata       *ShardMetadata // Optional schema metadata
	closed         bool
}

// SetMetadata sets the shard metadata to be written to the schema section.
func (w *ShardWriter) SetMetadata(meta *ShardMetadata) {
	w.metadata = meta
}

// NewShardWriter creates a new v2 shard writer.
func NewShardWriter(path string, role ShardRole) (*ShardWriter, error) {
	f, err := os.Create(path)
	if err != nil {
		return nil, err
	}

	// Create temp file for buffering data
	tempFile, err := os.CreateTemp("", "shard_data_*")
	if err != nil {
		f.Close()
		return nil, err
	}

	header := NewShardHeader(role)

	return &ShardWriter{
		file:           f,
		tempFile:       tempFile,
		tempFileOffset: 0,
		header:         header,
		entries:        make([]pendingEntry, 0),
	}, nil
}

// SetAlignment sets the data alignment (0, 16, 32, or 64 bytes).
// Returns ErrInvalidAlignment if the value is not one of the valid options.
func (w *ShardWriter) SetAlignment(align uint8) error {
	switch align {
	case AlignNone, Align16, Align32, Align64:
		w.header.Alignment = align
		return nil
	default:
		return fmt.Errorf("%w: got %d", ErrInvalidAlignment, align)
	}
}

// SetCompression sets the default compression type.
func (w *ShardWriter) SetCompression(comp uint8) {
	w.header.CompressionDefault = comp
}

// WriteEntry writes an entry to the shard.
func (w *ShardWriter) WriteEntry(name string, data []byte) error {
	return w.WriteEntryWithOptions(name, data, false, CompressNone)
}

// WriteEntryCompressed writes a compressed entry.
func (w *ShardWriter) WriteEntryCompressed(name string, data []byte) error {
	return w.WriteEntryWithOptions(name, data, true, w.header.CompressionDefault)
}

// WriteEntryTyped writes an entry with a content type hint.
func (w *ShardWriter) WriteEntryTyped(name string, data []byte, contentType uint16) error {
	return w.writeEntryFull(name, data, false, CompressNone, contentType)
}

// WriteEntryWithOptions writes an entry with explicit options.
func (w *ShardWriter) WriteEntryWithOptions(name string, data []byte, compress bool, compType uint8) error {
	return w.writeEntryFull(name, data, compress, compType, ContentTypeUnknown)
}

// writeEntryFull writes an entry with all options including content type.
func (w *ShardWriter) writeEntryFull(name string, data []byte, compress bool, compType uint8, contentType uint16) error {
	if w.closed {
		return ErrShardClosed
	}

	origSize := int64(len(data))
	checksum := ComputeChecksum(data)

	var flags uint16
	writeData := data

	// Compression (if requested and beneficial)
	if compress && compType != CompressNone && len(data) > minCompressSize {
		compressed, err := compressData(data, compType)
		if err == nil && len(compressed) < len(data)*compressionSavingsNumerator/compressionSavingsDenominator {
			writeData = compressed
			flags |= EntryFlagCompressed
			if compType == CompressZstd {
				flags |= EntryFlagZstd
			} else if compType == CompressLZ4 {
				flags |= EntryFlagLZ4
			}
		}
	}

	// Record temp file offset before writing
	tempOffset := w.tempFileOffset

	// Write to temp file
	n, err := w.tempFile.Write(writeData)
	if err != nil {
		return err
	}
	w.tempFileOffset += int64(n)

	// Store only metadata, not data - data stays in temp file
	w.entries = append(w.entries, pendingEntry{
		name:        name,
		tempOffset:  tempOffset,
		diskSize:    int64(len(writeData)),
		origSize:    origSize,
		checksum:    checksum,
		flags:       flags,
		contentType: contentType,
	})

	// Set content types flag if any non-zero content type is used
	if contentType != ContentTypeUnknown {
		w.header.Flags |= ShardFlagHasContentTypes
	}

	return nil
}

// Close finalizes the shard file.
func (w *ShardWriter) Close() error {
	if w.closed {
		return ErrShardClosed
	}
	w.closed = true

	defer func() {
		w.tempFile.Close()
		os.Remove(w.tempFile.Name())
	}()

	// Calculate layout
	entryCount := uint32(len(w.entries))
	indexSize := int64(entryCount) * int64(ShardIndexEntrySize)

	// Build string table
	stringTable := make([]byte, 0)
	nameOffsets := make([]uint32, len(w.entries))
	for i, e := range w.entries {
		nameOffsets[i] = uint32(len(stringTable))
		stringTable = append(stringTable, []byte(e.name)...)
		stringTable = append(stringTable, 0) // null terminator
	}

	// Calculate offsets
	stringTableOffset := int64(ShardHeaderSize) + indexSize
	dataSectionOffset := stringTableOffset + int64(len(stringTable))

	// Align data section
	if w.header.Alignment > 0 {
		align := int64(w.header.Alignment)
		dataSectionOffset = (dataSectionOffset + align - 1) & ^(align - 1)
	}

	// Calculate entry data offsets in final file
	currentDataOffset := dataSectionOffset
	dataOffsets := make([]uint64, len(w.entries))
	for i, e := range w.entries {
		// Align each entry
		if w.header.Alignment > 0 {
			align := int64(w.header.Alignment)
			currentDataOffset = (currentDataOffset + align - 1) & ^(align - 1)
		}
		dataOffsets[i] = uint64(currentDataOffset)
		currentDataOffset += e.diskSize
	}

	totalSize := currentDataOffset

	// Prepare metadata section if present
	var metadataBytes []byte
	schemaOffset := uint64(0)
	if w.metadata != nil {
		metadata := *w.metadata
		if metadata.SchemaVersion == "" {
			metadata.SchemaVersion = "shard-v2.1"
		}
		if metadata.Profile == "sampleshard.v1" {
			profile := &SampleProfile{}
			if metadata.SampleShard != nil {
				copyProfile := *metadata.SampleShard
				profile = &copyProfile
			}
			if profile.SampleIDType == "" {
				profile.SampleIDType = "uint64"
			}
			if profile.KeyEncoding == "" {
				profile.KeyEncoding = "decimal-string"
			}
			profile.SampleCount = uint64(entryCount)
			metadata.SampleShard = profile
		}
		var err error
		metadataBytes, err = metadata.Marshal()
		if err != nil {
			return fmt.Errorf("failed to marshal metadata: %w", err)
		}
		schemaOffset = uint64(totalSize)
		totalSize += int64(len(metadataBytes))
	}

	// Update header
	w.header.EntryCount = entryCount
	w.header.StringTableOffset = uint64(stringTableOffset)
	w.header.DataSectionOffset = uint64(dataSectionOffset)
	w.header.SchemaOffset = schemaOffset
	if w.metadata != nil {
		w.header.Flags |= ShardFlagHasSchema
	}
	w.header.TotalFileSize = uint64(totalSize)

	// Write header
	if err := WriteShardHeader(w.file, w.header); err != nil {
		return err
	}

	// Write index entries
	for i, e := range w.entries {
		entry := &IndexEntry{
			NameHash:   xxHash64String(e.name),
			NameOffset: nameOffsets[i],
			NameLen:    uint16(len(e.name)),
			Flags:      e.flags,
			DataOffset: dataOffsets[i],
			DiskSize:   uint64(e.diskSize),
			OrigSize:   uint64(e.origSize),
			Checksum:   e.checksum,
			Reserved:   uint32(e.contentType),
		}
		if err := WriteIndexEntry(w.file, entry); err != nil {
			return err
		}
	}

	// Write string table
	if _, err := w.file.Write(stringTable); err != nil {
		return err
	}

	// Write padding to data section
	paddingNeeded := dataSectionOffset - stringTableOffset - int64(len(stringTable))
	if paddingNeeded > 0 {
		padding := make([]byte, paddingNeeded)
		if _, err := w.file.Write(padding); err != nil {
			return err
		}
	}

	// Write data entries from temp file with alignment padding
	currentPos := dataSectionOffset
	for i, e := range w.entries {
		// Alignment padding
		expectedOffset, ok := uint64ToInt64(dataOffsets[i])
		if !ok {
			return fmt.Errorf("shard: data offset %d exceeds int64 range", dataOffsets[i])
		}
		if currentPos < expectedOffset {
			padding := make([]byte, expectedOffset-currentPos)
			if _, err := w.file.Write(padding); err != nil {
				return err
			}
			currentPos = expectedOffset
		}

		// VALIDATION: Assert currentPos matches expected offset
		if currentPos != expectedOffset {
			return fmt.Errorf("shard: offset invariant violated at entry %d: expected %d, got %d",
				i, expectedOffset, currentPos)
		}

		// Seek to entry's position in temp file
		if _, err := w.tempFile.Seek(e.tempOffset, io.SeekStart); err != nil {
			return err
		}

		// Read from temp and write to final file
		entryData := make([]byte, e.diskSize)
		if _, err := io.ReadFull(w.tempFile, entryData); err != nil {
			return err
		}
		if _, err := w.file.Write(entryData); err != nil {
			return err
		}
		currentPos += e.diskSize
	}

	// Write metadata section if present
	if len(metadataBytes) > 0 {
		if _, err := w.file.Write(metadataBytes); err != nil {
			return err
		}
		currentPos += int64(len(metadataBytes))
	}

	// VALIDATION: Assert final file size matches header
	totalFileSize, ok := uint64ToInt64(w.header.TotalFileSize)
	if !ok {
		return fmt.Errorf("shard: total file size %d exceeds int64 range", w.header.TotalFileSize)
	}
	if currentPos != totalFileSize {
		return fmt.Errorf("shard: file size mismatch: header says %d, actual %d",
			w.header.TotalFileSize, currentPos)
	}

	return w.file.Close()
}

// ============================================================
// ShardStreamWriter - Zero-copy streaming writer
// ============================================================
//
// StreamWriter writes directly to the output file without buffering
// entry data in memory or temp files. This is ideal for very large
// models where even temp file I/O is undesirable.
//
// Trade-off: Requires knowing entry count upfront OR accepting that
// the index is written at the end (requiring a seek-back to finalize).
//
// Usage:
//
//	sw, _ := NewShardStreamWriter(path, role)
//	sw.SetAlignment(64)
//	sw.BeginData() // Reserves space for header + max index entries
//	sw.WriteEntry("layer.0.weight", tensorData)
//	sw.WriteEntry("layer.1.weight", tensorData)
//	sw.Finalize() // Seeks back and writes header + index

// ShardStreamWriter provides zero-copy streaming writes.
type ShardStreamWriter struct {
	file            *os.File
	header          *ShardHeader
	entries         []streamEntryV2
	dataStartOffset int64 // Where data section begins
	currentOffset   int64 // Current write position
	maxEntries      int   // Reserved index slots
	begun           bool  // BeginData called
	closed          bool
}

// streamEntryV2 holds metadata for a streamed entry.
type streamEntryV2 struct {
	name        string
	dataOffset  int64
	diskSize    int64
	origSize    int64
	checksum    uint32
	flags       uint16
	contentType uint16
}

// NewShardStreamWriter creates a new streaming writer.
// maxEntries is the maximum number of entries (determines reserved index space).
// If maxEntries is 0, defaults to 10000.
func NewShardStreamWriter(path string, role ShardRole, maxEntries int) (*ShardStreamWriter, error) {
	if maxEntries <= 0 {
		maxEntries = 10000
	}

	f, err := os.Create(path)
	if err != nil {
		return nil, err
	}

	return &ShardStreamWriter{
		file:       f,
		header:     NewShardHeader(role),
		entries:    make([]streamEntryV2, 0, maxEntries),
		maxEntries: maxEntries,
	}, nil
}

// SetAlignment sets the data alignment (0, 16, 32, or 64 bytes).
// Returns ErrInvalidAlignment if the value is not valid, or an error if BeginData was already called.
func (sw *ShardStreamWriter) SetAlignment(align uint8) error {
	if sw.begun {
		return fmt.Errorf("shard: cannot set alignment after BeginData")
	}
	switch align {
	case AlignNone, Align16, Align32, Align64:
		sw.header.Alignment = align
		return nil
	default:
		return fmt.Errorf("%w: got %d", ErrInvalidAlignment, align)
	}
}

// SetCompression sets the default compression type.
func (sw *ShardStreamWriter) SetCompression(comp uint8) {
	sw.header.CompressionDefault = comp
}

// BeginData reserves space for header and index, then positions for data writes.
func (sw *ShardStreamWriter) BeginData() error {
	if sw.begun {
		return fmt.Errorf("shard stream: BeginData already called")
	}
	sw.begun = true

	// Calculate reserved space
	// Header: 64 bytes
	// Index: maxEntries * 48 bytes
	// String table: estimate avg name length of 64 bytes
	reservedIndexSize := int64(sw.maxEntries) * int64(ShardIndexEntrySize)
	reservedStringTableSize := int64(sw.maxEntries) * 64

	sw.dataStartOffset = int64(ShardHeaderSize) + reservedIndexSize + reservedStringTableSize

	// Align data start
	if sw.header.Alignment > 0 {
		align := int64(sw.header.Alignment)
		sw.dataStartOffset = (sw.dataStartOffset + align - 1) & ^(align - 1)
	}

	// Seek to data start position
	if _, err := sw.file.Seek(sw.dataStartOffset, io.SeekStart); err != nil {
		return err
	}
	sw.currentOffset = sw.dataStartOffset

	return nil
}

// WriteEntry writes an entry directly to the data section.
func (sw *ShardStreamWriter) WriteEntry(name string, data []byte) error {
	return sw.WriteEntryWithOptions(name, data, false, CompressNone)
}

// WriteEntryCompressed writes a compressed entry directly.
func (sw *ShardStreamWriter) WriteEntryCompressed(name string, data []byte) error {
	return sw.WriteEntryWithOptions(name, data, true, sw.header.CompressionDefault)
}

// WriteEntryTyped writes an entry with a content type hint.
func (sw *ShardStreamWriter) WriteEntryTyped(name string, data []byte, contentType uint16) error {
	return sw.writeEntryFull(name, data, false, CompressNone, contentType)
}

// WriteEntryWithOptions writes an entry with explicit options.
func (sw *ShardStreamWriter) WriteEntryWithOptions(name string, data []byte, compress bool, compType uint8) error {
	return sw.writeEntryFull(name, data, compress, compType, ContentTypeUnknown)
}

// writeEntryFull writes an entry with all options including content type.
func (sw *ShardStreamWriter) writeEntryFull(name string, data []byte, compress bool, compType uint8, contentType uint16) error {
	if sw.closed {
		return ErrShardClosed
	}
	if !sw.begun {
		return fmt.Errorf("shard stream: must call BeginData before writing entries")
	}
	if len(sw.entries) >= sw.maxEntries {
		return fmt.Errorf("shard stream: exceeded max entries %d", sw.maxEntries)
	}

	origSize := int64(len(data))
	checksum := ComputeChecksum(data)

	var flags uint16
	writeData := data

	// Compression (if requested and beneficial)
	if compress && compType != CompressNone && len(data) > minCompressSize {
		compressed, err := compressData(data, compType)
		if err == nil && len(compressed) < len(data)*compressionSavingsNumerator/compressionSavingsDenominator {
			writeData = compressed
			flags |= EntryFlagCompressed
			if compType == CompressZstd {
				flags |= EntryFlagZstd
			} else if compType == CompressLZ4 {
				flags |= EntryFlagLZ4
			}
		}
	}

	// Align current position
	if sw.header.Alignment > 0 {
		align := int64(sw.header.Alignment)
		alignedPos := (sw.currentOffset + align - 1) & ^(align - 1)
		if alignedPos > sw.currentOffset {
			padding := make([]byte, alignedPos-sw.currentOffset)
			if _, err := sw.file.Write(padding); err != nil {
				return err
			}
			sw.currentOffset = alignedPos
		}
	}

	// Record entry metadata
	entryOffset := sw.currentOffset

	// Write directly to file
	n, err := sw.file.Write(writeData)
	if err != nil {
		return err
	}
	sw.currentOffset += int64(n)

	sw.entries = append(sw.entries, streamEntryV2{
		name:        name,
		dataOffset:  entryOffset,
		diskSize:    int64(len(writeData)),
		origSize:    origSize,
		checksum:    checksum,
		flags:       flags,
		contentType: contentType,
	})

	// Set content types flag if any non-zero content type is used
	if contentType != ContentTypeUnknown {
		sw.header.Flags |= ShardFlagHasContentTypes
	}

	return nil
}

// Finalize writes the header, index, and string table, then closes the file.
func (sw *ShardStreamWriter) Finalize() error {
	if sw.closed {
		return ErrShardClosed
	}
	sw.closed = true

	if !sw.begun {
		return fmt.Errorf("shard stream: must call BeginData before Finalize")
	}

	entryCount := uint32(len(sw.entries))
	indexSize := int64(entryCount) * int64(ShardIndexEntrySize)

	// Build string table
	stringTable := make([]byte, 0)
	nameOffsets := make([]uint32, len(sw.entries))
	for i, e := range sw.entries {
		nameOffsets[i] = uint32(len(stringTable))
		stringTable = append(stringTable, []byte(e.name)...)
		stringTable = append(stringTable, 0)
	}

	// Calculate actual offsets
	stringTableOffset := int64(ShardHeaderSize) + indexSize
	actualDataOffset := stringTableOffset + int64(len(stringTable))
	if sw.header.Alignment > 0 {
		align := int64(sw.header.Alignment)
		actualDataOffset = (actualDataOffset + align - 1) & ^(align - 1)
	}

	// Verify we didn't overflow reserved space
	if actualDataOffset > sw.dataStartOffset {
		return fmt.Errorf("shard stream: index+strings overflow reserved space: need %d, reserved %d",
			actualDataOffset, sw.dataStartOffset)
	}

	totalSize := sw.currentOffset

	// Update header
	sw.header.Flags |= ShardFlagStreaming
	sw.header.EntryCount = entryCount
	sw.header.StringTableOffset = uint64(stringTableOffset)
	sw.header.DataSectionOffset = uint64(sw.dataStartOffset)
	sw.header.TotalFileSize = uint64(totalSize)

	// Seek to start and write header
	if _, err := sw.file.Seek(0, io.SeekStart); err != nil {
		return err
	}
	if err := WriteShardHeader(sw.file, sw.header); err != nil {
		return err
	}

	// Write index entries
	for i, e := range sw.entries {
		entry := &IndexEntry{
			NameHash:   xxHash64String(e.name),
			NameOffset: nameOffsets[i],
			NameLen:    uint16(len(e.name)),
			Flags:      e.flags,
			DataOffset: uint64(e.dataOffset),
			DiskSize:   uint64(e.diskSize),
			OrigSize:   uint64(e.origSize),
			Checksum:   e.checksum,
			Reserved:   uint32(e.contentType),
		}
		if err := WriteIndexEntry(sw.file, entry); err != nil {
			return err
		}
	}

	// Write string table
	if _, err := sw.file.Write(stringTable); err != nil {
		return err
	}

	// Write padding from string table to data section
	paddingNeeded := sw.dataStartOffset - stringTableOffset - int64(len(stringTable))
	if paddingNeeded > 0 {
		padding := make([]byte, paddingNeeded)
		if _, err := sw.file.Write(padding); err != nil {
			return err
		}
	}

	return sw.file.Close()
}

// Close aborts the write if not finalized.
func (sw *ShardStreamWriter) Close() error {
	if sw.closed {
		return nil
	}
	sw.closed = true
	return sw.file.Close()
}

// ============================================================
// ShardReader - Reader with fast index lookup
// ============================================================

// ShardReader provides reading of v2 shard files.
type ShardReader struct {
	file        *os.File
	data        MMap
	header      *ShardHeader
	index       []*IndexEntry
	stringTable []byte
	nameMap     map[uint64][]int // hash -> candidate indices (collision-safe)
	closed      bool
}

// OpenShard opens a v2 shard file for reading.
func OpenShard(path string) (*ShardReader, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}

	header, err := ReadShardHeader(f)
	if err != nil {
		f.Close()
		return nil, err
	}

	r := &ShardReader{
		file:   f,
		header: header,
	}

	// Validate IndexEntrySize is exactly 48 bytes for v2
	if header.IndexEntrySize != ShardIndexEntrySize {
		f.Close()
		return nil, fmt.Errorf("%w: IndexEntrySize %d != expected %d", ErrIndexCorrupt, header.IndexEntrySize, ShardIndexEntrySize)
	}

	// Security: validate entry count
	if header.EntryCount > MaxEntryCount {
		f.Close()
		return nil, fmt.Errorf("%w: entry count %d exceeds limit %d", ErrIndexCorrupt, header.EntryCount, MaxEntryCount)
	}

	// Security: calculate index size with overflow protection
	indexSize := int64(header.EntryCount) * int64(header.IndexEntrySize)
	if indexSize > MaxIndexSize {
		f.Close()
		return nil, fmt.Errorf("%w: index size %d exceeds limit %d", ErrIndexCorrupt, indexSize, MaxIndexSize)
	}

	// Read index
	indexData := make([]byte, indexSize)
	if _, err := io.ReadFull(f, indexData); err != nil {
		f.Close()
		return nil, fmt.Errorf("%w: %v", ErrIndexCorrupt, err)
	}

	r.index = make([]*IndexEntry, header.EntryCount)
	for i := uint32(0); i < header.EntryCount; i++ {
		offset := int64(i) * int64(header.IndexEntrySize)
		r.index[i] = ParseIndexEntry(indexData[offset:])
	}

	// Security: validate string table size
	if header.DataSectionOffset < header.StringTableOffset {
		f.Close()
		return nil, fmt.Errorf("%w: invalid section offsets", ErrIndexCorrupt)
	}
	stringTableSize := header.DataSectionOffset - header.StringTableOffset
	if stringTableSize > uint64(MaxStringTableSize) {
		f.Close()
		return nil, fmt.Errorf("%w: string table size %d invalid or exceeds limit %d", ErrIndexCorrupt, stringTableSize, MaxStringTableSize)
	}
	stringTableSizeInt, ok := uint64ToInt(stringTableSize)
	if !ok {
		f.Close()
		return nil, fmt.Errorf("%w: string table size %d exceeds int range", ErrIndexCorrupt, stringTableSize)
	}
	stringTableOffset, ok := uint64ToInt64(header.StringTableOffset)
	if !ok {
		f.Close()
		return nil, fmt.Errorf("%w: string table offset %d exceeds int64 range", ErrIndexCorrupt, header.StringTableOffset)
	}

	// Read string table
	r.stringTable = make([]byte, stringTableSizeInt)
	if _, err := f.Seek(stringTableOffset, io.SeekStart); err != nil {
		f.Close()
		return nil, err
	}
	if _, err := io.ReadFull(f, r.stringTable); err != nil {
		f.Close()
		return nil, err
	}

	// Validate TotalFileSize matches actual file size
	info, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}
	totalFileSize, ok := uint64ToInt64(header.TotalFileSize)
	if !ok {
		f.Close()
		return nil, fmt.Errorf("%w: total file size %d exceeds int64 range", ErrIndexCorrupt, header.TotalFileSize)
	}
	if totalFileSize != info.Size() {
		f.Close()
		return nil, fmt.Errorf("%w: TotalFileSize %d != actual file size %d", ErrIndexCorrupt, header.TotalFileSize, info.Size())
	}

	// Validate entry offsets are within file bounds and monotonically increasing
	var prevOffset uint64
	for i, e := range r.index {
		// Check offset is within data section
		if e.DataOffset < header.DataSectionOffset {
			f.Close()
			return nil, fmt.Errorf("%w: entry %d offset %d < data section offset %d", ErrIndexCorrupt, i, e.DataOffset, header.DataSectionOffset)
		}
		// Check offset + size doesn't exceed file
		if e.DataOffset+e.DiskSize > header.TotalFileSize {
			f.Close()
			return nil, fmt.Errorf("%w: entry %d extends past file end", ErrIndexCorrupt, i)
		}
		// Check monotonically increasing (entries are ordered by offset)
		if i > 0 && e.DataOffset < prevOffset {
			f.Close()
			return nil, fmt.Errorf("%w: entry %d offset %d < previous offset %d (not monotonic)", ErrIndexCorrupt, i, e.DataOffset, prevOffset)
		}
		prevOffset = e.DataOffset + e.DiskSize
	}

	// Build name lookup map (collision-safe: append indices per hash)
	r.nameMap = make(map[uint64][]int, len(r.index))
	for i, e := range r.index {
		r.nameMap[e.NameHash] = append(r.nameMap[e.NameHash], i)
	}

	return r, nil
}

// Header returns the v2 header.
func (r *ShardReader) Header() *ShardHeader {
	return r.header
}

// EntryCount returns the number of entries.
func (r *ShardReader) EntryCount() int {
	return len(r.index)
}

// EntryName returns the name of entry i.
func (r *ShardReader) EntryName(i int) string {
	if i < 0 || i >= len(r.index) {
		return ""
	}
	e := r.index[i]
	start := e.NameOffset
	// Bounds check start offset
	if int(start) >= len(r.stringTable) {
		return ""
	}
	// Check for overflow in end calculation
	if uint32(e.NameLen) > uint32(len(r.stringTable))-start {
		return ""
	}
	end := start + uint32(e.NameLen)
	return string(r.stringTable[start:end])
}

// EntryNames returns all entry names.
func (r *ShardReader) EntryNames() []string {
	names := make([]string, len(r.index))
	for i := range r.index {
		names[i] = r.EntryName(i)
	}
	return names
}

// Lookup returns the index of an entry by name, or -1 if not found.
// This is collision-safe: if multiple entries share the same hash,
// all candidates are checked until a name match is found.
func (r *ShardReader) Lookup(name string) int {
	hash := xxHash64String(name)
	candidates, ok := r.nameMap[hash]
	if !ok {
		return -1
	}
	for _, idx := range candidates {
		if r.EntryName(idx) == name {
			return idx
		}
	}
	return -1
}

// ReadEntry reads entry data by index with checksum verification enabled.
// Returns ErrChecksumMismatch if the checksum doesn't match.
func (r *ShardReader) ReadEntry(i int) ([]byte, error) {
	return r.ReadEntryWithVerify(i, true)
}

// ReadEntryWithVerify reads entry data with optional checksum verification.
//
// Checksum policy:
//   - verify=true: If ShardFlagHasChecksums is set and the computed checksum
//     doesn't match the stored checksum, returns ErrChecksumMismatch.
//     This is a hard error; there is no "warn and continue" mode.
//   - verify=false: No checksum validation is performed. Use this for
//     performance-critical paths where data integrity is verified elsewhere.
//
// The checksum is computed on the uncompressed data using CRC32C (Castagnoli).
func (r *ShardReader) ReadEntryWithVerify(i int, verify bool) ([]byte, error) {
	if r.closed {
		return nil, ErrShardClosed
	}
	if i < 0 || i >= len(r.index) {
		return nil, ErrEntryNotFound
	}

	e := r.index[i]

	// Read raw data
	var rawData []byte
	if r.data != nil {
		// mmap path
		dataLen := uint64(len(r.data))
		if e.DataOffset > dataLen || e.DiskSize > dataLen-e.DataOffset {
			return nil, io.ErrUnexpectedEOF
		}
		start, ok := uint64ToInt(e.DataOffset)
		if !ok {
			return nil, io.ErrUnexpectedEOF
		}
		endOffset := e.DataOffset + e.DiskSize
		end, ok := uint64ToInt(endOffset)
		if !ok {
			return nil, io.ErrUnexpectedEOF
		}
		rawData = r.data[start:end]
	} else {
		// file read path — use ReadAt (pread) for concurrent safety.
		// Unlike Seek+Read, ReadAt does not mutate the file offset,
		// so multiple goroutines can call it simultaneously.
		size, ok := uint64ToInt(e.DiskSize)
		if !ok {
			return nil, io.ErrUnexpectedEOF
		}
		offset, ok := uint64ToInt64(e.DataOffset)
		if !ok {
			return nil, io.ErrUnexpectedEOF
		}
		rawData = make([]byte, size)
		if _, err := r.file.ReadAt(rawData, offset); err != nil {
			return nil, err
		}
	}

	// Decompress if needed — decompressData allocates per-call buffers,
	// and zstd.Decoder.DecodeAll / lz4.UncompressBlock are both safe
	// for concurrent use, so no additional synchronization is needed.
	var data []byte
	if e.IsCompressed() {
		var err error
		data, err = decompressData(rawData, e.CompressionType(), e.OrigSize)
		if err != nil {
			return nil, err
		}
	} else {
		data = rawData
	}

	// Verify checksum — also verify when entry has a non-zero checksum,
	// even if the header flag was stripped (CRC flag bypass defense).
	if verify && (r.header.Flags&ShardFlagHasChecksums != 0 || e.Checksum != 0) {
		checksum := ComputeChecksum(data)
		if checksum != e.Checksum {
			return nil, fmt.Errorf("%w: entry %d expected 0x%08x, got 0x%08x",
				ErrChecksumMismatch, i, e.Checksum, checksum)
		}
	}

	return data, nil
}

// ReadEntryByName reads entry data by name.
func (r *ShardReader) ReadEntryByName(name string) ([]byte, error) {
	i := r.Lookup(name)
	if i < 0 {
		return nil, fmt.Errorf("%w: %q", ErrEntryNotFound, name)
	}
	return r.ReadEntry(i)
}

// ReadEntryPrefix reads only the first maxBytes of entry i.
// This enables header-only scanning without loading full tensor data.
// If the entry is smaller than maxBytes, returns the full entry.
// If the entry is compressed, decompresses and returns up to maxBytes of the result.
func (r *ShardReader) ReadEntryPrefix(i int, maxBytes int64) ([]byte, error) {
	if r.closed {
		return nil, ErrShardClosed
	}
	if i < 0 || i >= len(r.index) {
		return nil, ErrEntryNotFound
	}
	if maxBytes < 0 {
		return nil, fmt.Errorf("%w: negative prefix size %d", ErrIndexCorrupt, maxBytes)
	}

	e := r.index[i]

	// For compressed entries, we must decompress the full entry first
	// (compression doesn't allow partial decompression)
	if e.IsCompressed() {
		data, err := r.ReadEntry(i)
		if err != nil {
			return nil, err
		}
		if int64(len(data)) <= maxBytes {
			return data, nil
		}
		return data[:maxBytes], nil
	}

	// For uncompressed entries, read only what we need
	readSize := e.DiskSize
	if uint64(maxBytes) < readSize {
		readSize = uint64(maxBytes)
	}

	var rawData []byte
	if r.data != nil {
		// mmap path - just slice
		dataLen := uint64(len(r.data))
		if e.DataOffset > dataLen || readSize > dataLen-e.DataOffset {
			return nil, io.ErrUnexpectedEOF
		}
		start, ok := uint64ToInt(e.DataOffset)
		if !ok {
			return nil, io.ErrUnexpectedEOF
		}
		endOffset := e.DataOffset + readSize
		end, ok := uint64ToInt(endOffset)
		if !ok {
			return nil, io.ErrUnexpectedEOF
		}
		rawData = r.data[start:end]
	} else {
		// file read path — use ReadAt (pread) for concurrent safety
		size, ok := uint64ToInt(readSize)
		if !ok {
			return nil, io.ErrUnexpectedEOF
		}
		offset, ok := uint64ToInt64(e.DataOffset)
		if !ok {
			return nil, io.ErrUnexpectedEOF
		}
		rawData = make([]byte, size)
		if _, err := r.file.ReadAt(rawData, offset); err != nil {
			return nil, err
		}
	}

	return rawData, nil
}

// EnableMmap enables memory-mapped access.
func (r *ShardReader) EnableMmap() error {
	if r.data != nil {
		return nil
	}

	info, err := r.file.Stat()
	if err != nil {
		return err
	}

	data, err := mmapFile(r.file, info.Size())
	if err != nil {
		return err
	}

	r.data = data
	return nil
}

// Close closes the reader.
func (r *ShardReader) Close() error {
	if r.closed {
		return nil
	}
	r.closed = true

	if r.data != nil {
		if err := r.data.Unmap(); err != nil {
			r.file.Close()
			return err
		}
		r.data = nil
	}

	return r.file.Close()
}

// GetEntryInfo returns the index entry for entry i.
func (r *ShardReader) GetEntryInfo(i int) *IndexEntry {
	if i < 0 || i >= len(r.index) {
		return nil
	}
	return r.index[i]
}

// ReadMetadata reads and parses the schema metadata from the shard.
// Returns nil, nil if no metadata is present.
func (r *ShardReader) ReadMetadata() (*ShardMetadata, error) {
	if r.closed {
		return nil, ErrShardClosed
	}
	if r.header.SchemaOffset == 0 {
		return nil, nil // No metadata
	}
	if r.header.SchemaOffset > r.header.TotalFileSize {
		return nil, fmt.Errorf("%w: schema offset %d exceeds total file size %d", ErrIndexCorrupt, r.header.SchemaOffset, r.header.TotalFileSize)
	}

	// Calculate metadata size: from SchemaOffset to TotalFileSize
	metadataSize := r.header.TotalFileSize - r.header.SchemaOffset
	if metadataSize == 0 {
		return nil, nil
	}

	// Read metadata bytes
	var metadataBytes []byte
	if r.data != nil {
		// mmap path
		dataLen := uint64(len(r.data))
		if r.header.SchemaOffset > dataLen || metadataSize > dataLen-r.header.SchemaOffset {
			return nil, io.ErrUnexpectedEOF
		}
		start, ok := uint64ToInt(r.header.SchemaOffset)
		if !ok {
			return nil, io.ErrUnexpectedEOF
		}
		endOffset := r.header.SchemaOffset + metadataSize
		end, ok := uint64ToInt(endOffset)
		if !ok {
			return nil, io.ErrUnexpectedEOF
		}
		metadataBytes = r.data[start:end]
	} else {
		// file read path — use ReadAt (pread) for concurrent safety
		size, ok := uint64ToInt(metadataSize)
		if !ok {
			return nil, io.ErrUnexpectedEOF
		}
		offset, ok := uint64ToInt64(r.header.SchemaOffset)
		if !ok {
			return nil, io.ErrUnexpectedEOF
		}
		metadataBytes = make([]byte, size)
		if _, err := r.file.ReadAt(metadataBytes, offset); err != nil {
			return nil, err
		}
	}

	return ParseShardMetadata(metadataBytes)
}

// ListPrefix returns all entry names that start with the given prefix.
func (r *ShardReader) ListPrefix(prefix string) []string {
	var matches []string
	for i := 0; i < len(r.index); i++ {
		name := r.EntryName(i)
		if PathPrefix(name, prefix) {
			matches = append(matches, name)
		}
	}
	return matches
}

// ListChildren returns immediate children under the given prefix.
func (r *ShardReader) ListChildren(prefix string) []string {
	seen := make(map[string]bool)
	var children []string

	prefixLen := len(prefix)
	if prefix != "" && !strings.HasSuffix(prefix, PathSeparator) {
		prefixLen++
	}

	for i := 0; i < len(r.index); i++ {
		name := r.EntryName(i)
		if !PathPrefix(name, prefix) {
			continue
		}
		rest := name[prefixLen:]
		if idx := strings.Index(rest, PathSeparator); idx >= 0 {
			child := rest[:idx]
			if !seen[child] {
				seen[child] = true
				children = append(children, child)
			}
		} else if rest != "" {
			if !seen[rest] {
				seen[rest] = true
				children = append(children, rest)
			}
		}
	}
	return children
}

// ListWithTag returns entry names that have the given tag in their per-entry metadata.
// Requires metadata to be present in the schema section.
func (r *ShardReader) ListWithTag(tag string) ([]string, error) {
	meta, err := r.ReadMetadata()
	if err != nil {
		return nil, err
	}
	if meta == nil {
		return nil, nil
	}
	var matches []string
	for name, em := range meta.EntryMetadata {
		if em == nil {
			continue
		}
		for _, t := range em.Tags {
			if t == tag {
				matches = append(matches, name)
				break
			}
		}
	}
	return matches, nil
}

// ListWithTagBit returns entry names where the given tag bit (0-15) is set in the index.
// This is O(n) over the index but does not require loading metadata — pure index scan.
func (r *ShardReader) ListWithTagBit(bit int) []string {
	if bit < 0 || bit > 15 {
		return nil
	}
	var matches []string
	for i := 0; i < len(r.index); i++ {
		if r.index[i].HasTagBit(bit) {
			matches = append(matches, r.EntryName(i))
		}
	}
	return matches
}

// ListWithTagFast combines tag_dictionary lookup with tag bitmask for fast O(n) filtering.
// Returns nil, nil if no tag_dictionary is present or tag not found in dictionary.
func (r *ShardReader) ListWithTagFast(tag string) ([]string, error) {
	meta, err := r.ReadMetadata()
	if err != nil {
		return nil, err
	}
	if meta == nil || len(meta.TagDictionary) == 0 {
		return nil, nil
	}
	for i, t := range meta.TagDictionary {
		if t == tag {
			return r.ListWithTagBit(i), nil
		}
	}
	return nil, nil // Tag not in dictionary
}

// ============================================================
// Compression helpers
// ============================================================

// Compression thresholds
const (
	// minCompressSize is the minimum data size to attempt compression
	minCompressSize = 256
	// compressionSavingsThreshold is the minimum savings ratio to use compression (9/10 = 90%)
	compressionSavingsNumerator   = 9
	compressionSavingsDenominator = 10
)

// zstdEncoder is a reusable zstd encoder (created lazily, thread-safe)
var (
	zstdEncoder     *zstd.Encoder
	zstdEncoderOnce sync.Once
	zstdEncoderErr  error
)

// zstdDecoder is a reusable zstd decoder (created lazily, thread-safe)
var (
	zstdDecoder     *zstd.Decoder
	zstdDecoderOnce sync.Once
	zstdDecoderErr  error
)

// getZstdEncoder returns a reusable zstd encoder (thread-safe initialization)
func getZstdEncoder() (*zstd.Encoder, error) {
	zstdEncoderOnce.Do(func() {
		zstdEncoder, zstdEncoderErr = zstd.NewWriter(nil,
			zstd.WithEncoderLevel(zstd.SpeedDefault),
			zstd.WithEncoderConcurrency(1),
		)
		if zstdEncoderErr != nil {
			zstdEncoderErr = fmt.Errorf("failed to create zstd encoder: %w", zstdEncoderErr)
		}
	})
	return zstdEncoder, zstdEncoderErr
}

// getZstdDecoder returns a reusable zstd decoder (thread-safe initialization)
func getZstdDecoder() (*zstd.Decoder, error) {
	zstdDecoderOnce.Do(func() {
		zstdDecoder, zstdDecoderErr = zstd.NewReader(nil,
			zstd.WithDecoderConcurrency(1),
		)
		if zstdDecoderErr != nil {
			zstdDecoderErr = fmt.Errorf("failed to create zstd decoder: %w", zstdDecoderErr)
		}
	})
	return zstdDecoder, zstdDecoderErr
}

// compressData compresses data using the specified algorithm.
func compressData(data []byte, compType uint8) ([]byte, error) {
	switch compType {
	case CompressZstd:
		enc, err := getZstdEncoder()
		if err != nil {
			return nil, err
		}
		return enc.EncodeAll(data, nil), nil
	case CompressLZ4:
		// Pre-allocate buffer based on worst-case compression size
		maxSize := lz4.CompressBlockBound(len(data))
		compressed := make([]byte, maxSize)
		n, err := lz4.CompressBlock(data, compressed, nil)
		if err != nil {
			return nil, fmt.Errorf("lz4 compression failed: %w", err)
		}
		if n == 0 {
			// Data is incompressible, return original
			return data, nil
		}
		return compressed[:n], nil
	default:
		return data, nil
	}
}

// MaxDecompressSize is the maximum allowed decompressed size (1GB) to prevent DoS.
const MaxDecompressSize = 1 << 30

// decompressData decompresses data using the specified algorithm.
// origSize is the expected decompressed size (from IndexEntry.OrigSize).
func decompressData(data []byte, compType uint8, origSize uint64) ([]byte, error) {
	// Security: cap decompressed size to prevent memory exhaustion
	if origSize > MaxDecompressSize {
		return nil, fmt.Errorf("%w: origSize %d exceeds limit %d", ErrCompressionFailed, origSize, MaxDecompressSize)
	}

	switch compType {
	case CompressZstd:
		dec, err := getZstdDecoder()
		if err != nil {
			return nil, err
		}
		// Pre-allocate based on origSize for efficiency
		dst := make([]byte, 0, origSize)
		result, err := dec.DecodeAll(data, dst)
		if err != nil {
			return nil, fmt.Errorf("zstd decompression failed: %w", err)
		}
		if uint64(len(result)) != origSize {
			return nil, fmt.Errorf("zstd size mismatch: got %d, expected %d", len(result), origSize)
		}
		return result, nil
	case CompressLZ4:
		// LZ4 block decompression requires exact destination size
		decompressed := make([]byte, origSize)
		n, err := lz4.UncompressBlock(data, decompressed)
		if err != nil {
			return nil, fmt.Errorf("lz4 decompression failed: %w", err)
		}
		if uint64(n) != origSize {
			return nil, fmt.Errorf("lz4 size mismatch: got %d, expected %d", n, origSize)
		}
		return decompressed[:n], nil
	default:
		return data, nil
	}
}
