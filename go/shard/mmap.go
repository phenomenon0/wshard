// Package shard provides cross-platform mmap utilities using github.com/edsrzf/mmap-go.
package shard

import (
	"os"

	mmap "github.com/edsrzf/mmap-go"
)

// MMap is a type alias for mmap.MMap for use in other files.
type MMap = mmap.MMap

// mmapFile wraps the mmap-go package for internal use.
func mmapFile(f *os.File, size int64) (mmap.MMap, error) {
	if size == 0 {
		return nil, nil
	}
	return mmap.Map(f, mmap.RDONLY, 0)
}

// MappedFile represents a memory-mapped file for zero-copy access.
type MappedFile struct {
	data mmap.MMap
	file *os.File
}

// MmapFile opens and memory-maps a file for reading.
func MmapFile(path string) (*MappedFile, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}

	info, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}

	data, err := mmapFile(f, info.Size())
	if err != nil {
		f.Close()
		return nil, err
	}

	return &MappedFile{
		data: data,
		file: f,
	}, nil
}

// Data returns the raw mapped bytes.
func (m *MappedFile) Data() []byte {
	return m.data
}

// Size returns the size of the mapped region.
func (m *MappedFile) Size() int64 {
	return int64(len(m.data))
}

// Slice returns a slice of the mapped region at the given offset and size.
// This is a zero-copy operation - the returned slice points into the mmap region.
func (m *MappedFile) Slice(offset, size int64) []byte {
	if offset+size > int64(len(m.data)) {
		return nil
	}
	return m.data[offset : offset+size]
}

// Close unmaps and closes the file.
func (m *MappedFile) Close() error {
	if m.data != nil {
		if err := m.data.Unmap(); err != nil {
			m.file.Close()
			return err
		}
		m.data = nil
	}
	return m.file.Close()
}

// MappedTensor represents a tensor view into a memory-mapped region.
// This provides zero-copy access to tensor data stored in a shard file.
type MappedTensor struct {
	DType DType    // Data type
	Dims  []uint64 // Shape dimensions
	Data  []byte   // Points into mmap region (zero-copy)
}

