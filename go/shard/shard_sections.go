// Package ucodec provides optional section support for Shard+.
//
// This file adds forward-compatible sections for:
//   - 64KB page alignment (align_pow2)
//   - Section Directory (SDIR) for extensibility
//   - String Table (STRT) for name storage
//   - Name Map (NMAP) for source/canon name mapping
//   - Region Table (RGON) for prefetch hints
//   - Quant Meta (QMET) for quantization metadata linking
//
// All sections are optional and readers skip unknown section types.
package shard

import (
	"encoding/binary"
	"fmt"
	"io"
	"sort"
)

// ============================================================
// Section Type Constants (FourCC as uint32 LE)
// ============================================================

const (
	// Core sections
	SecTypeINDX uint32 = 0x58444E49 // 'INDX' - Tensor index
	SecTypeDATA uint32 = 0x41544144 // 'DATA' - Tensor data blob
	SecTypeSTRT uint32 = 0x54525453 // 'STRT' - String table
	SecTypeMETA uint32 = 0x4154454D // 'META' - Model metadata (JSON)

	// Optional sections (Level 1)
	SecTypeNMAP uint32 = 0x50414D4E // 'NMAP' - Name map (canon -> source)
	SecTypeRGON uint32 = 0x4E4F4752 // 'RGON' - Region table (prefetch hints)
	SecTypeQMET uint32 = 0x54454D51 // 'QMET' - Quant metadata linking

	// Section directory
	SecTypeSDIR uint32 = 0x52494453 // 'SDIR' - Section directory
)

// Alignment power-of-2 values
const (
	AlignPow2_64B  uint8 = 6  // 1<<6 = 64 bytes (SIMD)
	AlignPow2_4KB  uint8 = 12 // 1<<12 = 4096 bytes (page)
	AlignPow2_64KB uint8 = 16 // 1<<16 = 65536 bytes (large page)
)

// Section flags
const (
	SecFlagCompressedZstd uint32 = 0x0001
	SecFlagCompressedLZ4  uint32 = 0x0002
	SecFlagHasCRC32C      uint32 = 0x0004
	SecFlagImmutable      uint32 = 0x0008
	SecFlagPageAligned    uint32 = 0x0010
)

// Header flags for extended features
const (
	ShardFlagHasSectionDir ShardFlags = 0x0100 // Section directory present
	ShardFlagHasNameMap    ShardFlags = 0x0200 // Name map section present
	ShardFlagHasRegionTbl  ShardFlags = 0x0400 // Region table present
	ShardFlagHasQuantMeta  ShardFlags = 0x0800 // Quant meta section present
	ShardFlagAlignPow2     ShardFlags = 0x1000 // Uses align_pow2 semantics
)

// ============================================================
// Section Directory
// ============================================================

// SectionDirHeader is the header for the section directory.
type SectionDirHeader struct {
	Magic     uint32 // 'SDIR' (SecTypeSDIR)
	Version   uint16 // 1
	EntrySize uint16 // sizeof(SectionDirEntry) = 40
	Count     uint32 // Number of entries
	Reserved  uint32
}

// SectionDirEntry describes a single section in the file.
type SectionDirEntry struct {
	Type     uint32 // FourCC section type
	Flags    uint32 // Per-section flags
	Offset   uint64 // File offset
	Bytes    uint64 // Section size on disk
	UBytes   uint64 // Uncompressed size (0 if not compressed)
	CRC32C   uint32 // CRC over section payload (0 if unused)
	Reserved uint32
}

const (
	SectionDirHeaderSize = 16
	SectionDirEntrySize  = 40
)

// WriteSectionDirHeader writes the section directory header.
func WriteSectionDirHeader(w io.Writer, h *SectionDirHeader) error {
	buf := make([]byte, SectionDirHeaderSize)
	binary.LittleEndian.PutUint32(buf[0:4], h.Magic)
	binary.LittleEndian.PutUint16(buf[4:6], h.Version)
	binary.LittleEndian.PutUint16(buf[6:8], h.EntrySize)
	binary.LittleEndian.PutUint32(buf[8:12], h.Count)
	binary.LittleEndian.PutUint32(buf[12:16], h.Reserved)
	_, err := w.Write(buf)
	return err
}

// ReadSectionDirHeader reads the section directory header.
func ReadSectionDirHeader(r io.Reader) (*SectionDirHeader, error) {
	buf := make([]byte, SectionDirHeaderSize)
	if _, err := io.ReadFull(r, buf); err != nil {
		return nil, err
	}
	return &SectionDirHeader{
		Magic:     binary.LittleEndian.Uint32(buf[0:4]),
		Version:   binary.LittleEndian.Uint16(buf[4:6]),
		EntrySize: binary.LittleEndian.Uint16(buf[6:8]),
		Count:     binary.LittleEndian.Uint32(buf[8:12]),
		Reserved:  binary.LittleEndian.Uint32(buf[12:16]),
	}, nil
}

// WriteSectionDirEntry writes a single section directory entry.
func WriteSectionDirEntry(w io.Writer, e *SectionDirEntry) error {
	buf := make([]byte, SectionDirEntrySize)
	binary.LittleEndian.PutUint32(buf[0:4], e.Type)
	binary.LittleEndian.PutUint32(buf[4:8], e.Flags)
	binary.LittleEndian.PutUint64(buf[8:16], e.Offset)
	binary.LittleEndian.PutUint64(buf[16:24], e.Bytes)
	binary.LittleEndian.PutUint64(buf[24:32], e.UBytes)
	binary.LittleEndian.PutUint32(buf[32:36], e.CRC32C)
	binary.LittleEndian.PutUint32(buf[36:40], e.Reserved)
	_, err := w.Write(buf)
	return err
}

// ReadSectionDirEntry reads a single section directory entry.
func ReadSectionDirEntry(r io.Reader) (*SectionDirEntry, error) {
	buf := make([]byte, SectionDirEntrySize)
	if _, err := io.ReadFull(r, buf); err != nil {
		return nil, err
	}
	return &SectionDirEntry{
		Type:     binary.LittleEndian.Uint32(buf[0:4]),
		Flags:    binary.LittleEndian.Uint32(buf[4:8]),
		Offset:   binary.LittleEndian.Uint64(buf[8:16]),
		Bytes:    binary.LittleEndian.Uint64(buf[16:24]),
		UBytes:   binary.LittleEndian.Uint64(buf[24:32]),
		CRC32C:   binary.LittleEndian.Uint32(buf[32:36]),
		Reserved: binary.LittleEndian.Uint32(buf[36:40]),
	}, nil
}

// SectionTypeName returns a human-readable name for a section type.
func SectionTypeName(t uint32) string {
	switch t {
	case SecTypeINDX:
		return "INDX"
	case SecTypeDATA:
		return "DATA"
	case SecTypeSTRT:
		return "STRT"
	case SecTypeMETA:
		return "META"
	case SecTypeNMAP:
		return "NMAP"
	case SecTypeRGON:
		return "RGON"
	case SecTypeQMET:
		return "QMET"
	case SecTypeSDIR:
		return "SDIR"
	default:
		// Return as FourCC string
		b := make([]byte, 4)
		binary.LittleEndian.PutUint32(b, t)
		return string(b)
	}
}

// ============================================================
// Name Map Section (NMAP)
// ============================================================

// NameMapHeader is the header for the name map section.
type NameMapHeader struct {
	Magic    uint32 // 'NMAP' (SecTypeNMAP)
	Version  uint16 // 1
	Flags    uint16 // Reserved
	Count    uint32 // Number of mappings
	Reserved uint32
}

// NameMapEntry maps canonical name to source name.
type NameMapEntry struct {
	CanonOff  uint32 // Offset into string table for canonical name
	SourceOff uint32 // Offset into string table for source name
	CanonHash uint32 // xxHash32 of canonical name (for fast lookup)
	Reserved  uint32
}

const (
	NameMapHeaderSize = 16
	NameMapEntrySize  = 16
)

// WriteNameMapHeader writes the name map header.
func WriteNameMapHeader(w io.Writer, h *NameMapHeader) error {
	buf := make([]byte, NameMapHeaderSize)
	binary.LittleEndian.PutUint32(buf[0:4], h.Magic)
	binary.LittleEndian.PutUint16(buf[4:6], h.Version)
	binary.LittleEndian.PutUint16(buf[6:8], h.Flags)
	binary.LittleEndian.PutUint32(buf[8:12], h.Count)
	binary.LittleEndian.PutUint32(buf[12:16], h.Reserved)
	_, err := w.Write(buf)
	return err
}

// WriteNameMapEntry writes a single name map entry.
func WriteNameMapEntry(w io.Writer, e *NameMapEntry) error {
	buf := make([]byte, NameMapEntrySize)
	binary.LittleEndian.PutUint32(buf[0:4], e.CanonOff)
	binary.LittleEndian.PutUint32(buf[4:8], e.SourceOff)
	binary.LittleEndian.PutUint32(buf[8:12], e.CanonHash)
	binary.LittleEndian.PutUint32(buf[12:16], e.Reserved)
	_, err := w.Write(buf)
	return err
}

// NameMap holds the in-memory representation of canon -> source mappings.
type NameMap struct {
	CanonToSource map[string]string // canon_name -> source_name
	SourceToCanon map[string]string // source_name -> canon_name
}

// NewNameMap creates an empty name map.
func NewNameMap() *NameMap {
	return &NameMap{
		CanonToSource: make(map[string]string),
		SourceToCanon: make(map[string]string),
	}
}

// Add adds a mapping from canonical name to source name.
func (nm *NameMap) Add(canon, source string) {
	nm.CanonToSource[canon] = source
	nm.SourceToCanon[source] = canon
}

// GetSource returns the source name for a canonical name.
func (nm *NameMap) GetSource(canon string) (string, bool) {
	s, ok := nm.CanonToSource[canon]
	return s, ok
}

// GetCanon returns the canonical name for a source name.
func (nm *NameMap) GetCanon(source string) (string, bool) {
	c, ok := nm.SourceToCanon[source]
	return c, ok
}

// ============================================================
// Region Table Section (RGON)
// ============================================================

// RegionTableHeader is the header for the region table.
type RegionTableHeader struct {
	Magic       uint32 // 'RGON' (SecTypeRGON)
	Version     uint16 // 1
	Flags       uint16 // Reserved
	RegionCount uint32 // Number of regions
	EntryCount  uint32 // Total tensor refs across all regions
	Reserved    uint32
}

// RegionDesc describes a single region (e.g., a layer).
type RegionDesc struct {
	RegionID  uint16 // Region identifier (e.g., layer number)
	Reserved0 uint16
	First     uint32 // First index into tensor_refs array
	Count     uint32 // Number of tensor refs in this region
	Reserved1 uint32
}

const (
	RegionTableHeaderSize = 20
	RegionDescSize        = 16
)

// WriteRegionTableHeader writes the region table header.
func WriteRegionTableHeader(w io.Writer, h *RegionTableHeader) error {
	buf := make([]byte, RegionTableHeaderSize)
	binary.LittleEndian.PutUint32(buf[0:4], h.Magic)
	binary.LittleEndian.PutUint16(buf[4:6], h.Version)
	binary.LittleEndian.PutUint16(buf[6:8], h.Flags)
	binary.LittleEndian.PutUint32(buf[8:12], h.RegionCount)
	binary.LittleEndian.PutUint32(buf[12:16], h.EntryCount)
	binary.LittleEndian.PutUint32(buf[16:20], h.Reserved)
	_, err := w.Write(buf)
	return err
}

// WriteRegionDesc writes a single region descriptor.
func WriteRegionDesc(w io.Writer, r *RegionDesc) error {
	buf := make([]byte, RegionDescSize)
	binary.LittleEndian.PutUint16(buf[0:2], r.RegionID)
	binary.LittleEndian.PutUint16(buf[2:4], r.Reserved0)
	binary.LittleEndian.PutUint32(buf[4:8], r.First)
	binary.LittleEndian.PutUint32(buf[8:12], r.Count)
	binary.LittleEndian.PutUint32(buf[12:16], r.Reserved1)
	_, err := w.Write(buf)
	return err
}

// RegionTable holds the in-memory region table.
type RegionTable struct {
	Regions    []RegionInfo      // Regions in order
	TensorRefs []uint32          // Tensor indices referenced by regions
	NameToRgn  map[string]uint16 // tensor name -> region ID
}

// RegionInfo describes a region with its tensors.
type RegionInfo struct {
	ID      uint16   // Region ID
	Name    string   // Human-readable name (e.g., "layer_00", "embeddings")
	Tensors []string // Tensor names in this region
}

// NewRegionTable creates an empty region table.
func NewRegionTable() *RegionTable {
	return &RegionTable{
		Regions:    make([]RegionInfo, 0),
		TensorRefs: make([]uint32, 0),
		NameToRgn:  make(map[string]uint16),
	}
}

// AddRegion adds a new region with the given tensors.
func (rt *RegionTable) AddRegion(id uint16, name string, tensors []string) {
	rt.Regions = append(rt.Regions, RegionInfo{
		ID:      id,
		Name:    name,
		Tensors: tensors,
	})
	for _, t := range tensors {
		rt.NameToRgn[t] = id
	}
}

// GetRegion returns the region ID for a tensor name.
func (rt *RegionTable) GetRegion(tensorName string) (uint16, bool) {
	id, ok := rt.NameToRgn[tensorName]
	return id, ok
}

// InferRegionsFromNames auto-infers regions from tensor names.
// Uses naming conventions:
//   - embed/tok_embed -> region 0
//   - layer_N.* or layers.N.* -> region 1+N
//   - norm/lm_head -> last region
func InferRegionsFromNames(names []string) *RegionTable {
	rt := NewRegionTable()

	// Group tensors by inferred region
	groups := make(map[uint16][]string)
	maxRegion := uint16(0)

	for _, name := range names {
		region := inferRegionFromName(name)
		groups[region] = append(groups[region], name)
		if region > maxRegion && region < 0xFFFF {
			maxRegion = region
		}
	}

	// Create regions in order
	regionIDs := make([]uint16, 0, len(groups))
	for id := range groups {
		regionIDs = append(regionIDs, id)
	}
	sort.Slice(regionIDs, func(i, j int) bool {
		return regionIDs[i] < regionIDs[j]
	})

	for _, id := range regionIDs {
		tensors := groups[id]
		name := fmt.Sprintf("region_%d", id)
		if id == 0 {
			name = "embeddings"
		} else if id == 0xFFFF {
			name = "output"
		} else {
			name = fmt.Sprintf("layer_%d", id-1)
		}
		rt.AddRegion(id, name, tensors)
	}

	return rt
}

// inferRegionFromName returns a region ID based on tensor naming conventions.
func inferRegionFromName(name string) uint16 {
	// Embeddings
	if containsAny(name, []string{"embed", "tok_embed", "wte", "wpe"}) {
		return 0
	}

	// Layer patterns: layer_N, layers.N, h.N
	layerNum := extractLayerNumber(name)
	if layerNum >= 0 {
		return uint16(1 + layerNum) // Region 1 = layer 0, etc.
	}

	// Output: norm, lm_head, output
	if containsAny(name, []string{"norm_final", "ln_f", "lm_head", "output"}) {
		return 0xFFFF // Last region
	}

	// Default to region 0
	return 0
}

// extractLayerNumber extracts layer number from tensor name.
// Returns -1 if no layer number found.
func extractLayerNumber(name string) int {
	// Patterns: layer_00, layers.0, h.0, transformer.h.0
	patterns := []string{
		"layer_", "layers.", "h.", "block.", "blocks.",
	}

	for _, p := range patterns {
		idx := 0
		for {
			pos := indexAt(name, p, idx)
			if pos < 0 {
				break
			}
			// Extract number after pattern
			numStart := pos + len(p)
			numEnd := numStart
			for numEnd < len(name) && name[numEnd] >= '0' && name[numEnd] <= '9' {
				numEnd++
			}
			if numEnd > numStart {
				num := 0
				for i := numStart; i < numEnd; i++ {
					num = num*10 + int(name[i]-'0')
				}
				return num
			}
			idx = pos + 1
		}
	}
	return -1
}

// indexAt finds substring starting from index.
func indexAt(s, substr string, start int) int {
	if start >= len(s) {
		return -1
	}
	for i := start; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return i
		}
	}
	return -1
}

// containsAny returns true if s contains any of the substrings.
func containsAny(s string, substrs []string) bool {
	for _, sub := range substrs {
		for i := 0; i <= len(s)-len(sub); i++ {
			if s[i:i+len(sub)] == sub {
				return true
			}
		}
	}
	return false
}

// ============================================================
// Quant Meta Section (QMET)
// ============================================================

// QuantMetaHeader is the header for the quant metadata section.
type QuantMetaHeader struct {
	Magic    uint32 // 'QMET' (SecTypeQMET)
	Version  uint16 // 1
	Flags    uint16 // Reserved
	Count    uint32 // Number of quant records
	Reserved uint32
}

// QuantRecord links a tensor to its quantization metadata.
// Uses existing QuantScheme* constants from tensor_v3.go (uint8).
type QuantRecord struct {
	TensorIndex uint32 // Which base tensor this applies to
	Scheme      uint8  // Quantization scheme (use QuantScheme* constants from tensor_v3.go)
	Reserved0   uint8  // Padding
	GroupSize   uint16 // Group size (32/64/128)
	OrigDType   uint16 // Original dtype enum
	AuxCount    uint16 // Number of aux refs following
	// Followed by AuxCount AuxRef entries
}

// AuxRole defines the role of an auxiliary tensor.
type AuxRole uint16

const (
	AuxRoleScales   AuxRole = 1
	AuxRoleZeros    AuxRole = 2
	AuxRoleMins     AuxRole = 3
	AuxRoleCodebook AuxRole = 4
	AuxRoleMeta     AuxRole = 5
)

// AuxRef references an auxiliary tensor for quantization.
type AuxRef struct {
	Role           AuxRole // Role of this aux tensor
	Reserved0      uint16
	AuxTensorIndex uint32 // Tensor index of aux buffer
}

const (
	QuantMetaHeaderSize = 16
	QuantRecordBaseSize = 12 // Without aux refs
	AuxRefSize          = 8
)

// WriteQuantMetaHeader writes the quant meta header.
func WriteQuantMetaHeader(w io.Writer, h *QuantMetaHeader) error {
	buf := make([]byte, QuantMetaHeaderSize)
	binary.LittleEndian.PutUint32(buf[0:4], h.Magic)
	binary.LittleEndian.PutUint16(buf[4:6], h.Version)
	binary.LittleEndian.PutUint16(buf[6:8], h.Flags)
	binary.LittleEndian.PutUint32(buf[8:12], h.Count)
	binary.LittleEndian.PutUint32(buf[12:16], h.Reserved)
	_, err := w.Write(buf)
	return err
}

// WriteQuantRecord writes a quant record with its aux refs.
func WriteQuantRecord(w io.Writer, r *QuantRecord, auxRefs []AuxRef) error {
	buf := make([]byte, QuantRecordBaseSize)
	binary.LittleEndian.PutUint32(buf[0:4], r.TensorIndex)
	binary.LittleEndian.PutUint16(buf[4:6], uint16(r.Scheme))
	binary.LittleEndian.PutUint16(buf[6:8], r.GroupSize)
	binary.LittleEndian.PutUint16(buf[8:10], r.OrigDType)
	binary.LittleEndian.PutUint16(buf[10:12], r.AuxCount)
	if _, err := w.Write(buf); err != nil {
		return err
	}

	// Write aux refs
	for _, aux := range auxRefs {
		auxBuf := make([]byte, AuxRefSize)
		binary.LittleEndian.PutUint16(auxBuf[0:2], uint16(aux.Role))
		binary.LittleEndian.PutUint16(auxBuf[2:4], aux.Reserved0)
		binary.LittleEndian.PutUint32(auxBuf[4:8], aux.AuxTensorIndex)
		if _, err := w.Write(auxBuf); err != nil {
			return err
		}
	}

	return nil
}

// QuantMeta holds in-memory quant metadata.
type QuantMeta struct {
	Records []QuantMetaRecord
}

// QuantMetaRecord is an in-memory quant record.
type QuantMetaRecord struct {
	TensorName string
	Scheme     uint8 // Use QuantScheme* constants from tensor_v3.go
	GroupSize  uint16
	OrigDType  uint16
	AuxTensors map[AuxRole]string // role -> aux tensor name
}

// NewQuantMeta creates empty quant metadata.
func NewQuantMeta() *QuantMeta {
	return &QuantMeta{
		Records: make([]QuantMetaRecord, 0),
	}
}

// AddRecord adds a quantization record.
func (qm *QuantMeta) AddRecord(tensorName string, scheme uint8, groupSize, origDType uint16, auxTensors map[AuxRole]string) {
	qm.Records = append(qm.Records, QuantMetaRecord{
		TensorName: tensorName,
		Scheme:     scheme,
		GroupSize:  groupSize,
		OrigDType:  origDType,
		AuxTensors: auxTensors,
	})
}

// ============================================================
// Alignment Helpers
// ============================================================

// AlignUp aligns x up to the next multiple of align.
func AlignUp(x, align uint64) uint64 {
	if align == 0 {
		return x
	}
	return (x + align - 1) &^ (align - 1)
}

// AlignPow2ToBytes converts align_pow2 to byte alignment.
func AlignPow2ToBytes(pow2 uint8) uint64 {
	if pow2 > 63 {
		return 0
	}
	return uint64(1) << pow2
}

// BytesToAlignPow2 converts byte alignment to align_pow2.
// Returns the smallest power of 2 >= align.
func BytesToAlignPow2(align uint64) uint8 {
	if align <= 1 {
		return 0
	}
	pow2 := uint8(0)
	for (uint64(1) << pow2) < align {
		pow2++
	}
	return pow2
}

// ============================================================
// String Table Helpers
// ============================================================

// StringTableBuilder builds a string table with deduplication.
type StringTableBuilder struct {
	strings  []string          // Strings in order
	offsets  map[string]uint32 // String -> offset
	totalLen uint32            // Current total length
}

// NewStringTableBuilder creates a new string table builder.
func NewStringTableBuilder() *StringTableBuilder {
	return &StringTableBuilder{
		strings: make([]string, 0),
		offsets: make(map[string]uint32),
	}
}

// Add adds a string and returns its offset.
// Returns existing offset if string already in table.
func (stb *StringTableBuilder) Add(s string) uint32 {
	if off, ok := stb.offsets[s]; ok {
		return off
	}
	off := stb.totalLen
	stb.strings = append(stb.strings, s)
	stb.offsets[s] = off
	// Length-prefixed: 4 bytes len + string bytes
	stb.totalLen += 4 + uint32(len(s))
	return off
}

// Build returns the serialized string table bytes.
func (stb *StringTableBuilder) Build() []byte {
	data := make([]byte, stb.totalLen)
	pos := 0
	for _, s := range stb.strings {
		binary.LittleEndian.PutUint32(data[pos:], uint32(len(s)))
		pos += 4
		copy(data[pos:], s)
		pos += len(s)
	}
	return data
}

// GetOffset returns the offset for a string, or -1 if not found.
func (stb *StringTableBuilder) GetOffset(s string) (uint32, bool) {
	off, ok := stb.offsets[s]
	return off, ok
}

// Len returns the total byte length of the string table.
func (stb *StringTableBuilder) Len() uint32 {
	return stb.totalLen
}

// ReadStringAt reads a length-prefixed string from data at offset.
func ReadStringAt(data []byte, offset uint32) (string, error) {
	if int(offset)+4 > len(data) {
		return "", fmt.Errorf("string offset %d out of bounds", offset)
	}
	strLen := binary.LittleEndian.Uint32(data[offset:])
	if int(offset)+4+int(strLen) > len(data) {
		return "", fmt.Errorf("string length %d exceeds data bounds", strLen)
	}
	return string(data[offset+4 : offset+4+strLen]), nil
}
