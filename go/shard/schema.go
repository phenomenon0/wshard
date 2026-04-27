// Package shard provides schema validation for Shard containers.
//
// # Schema Registry
//
// The schema section allows optional validation of tensor shapes and types.
// This enables:
//   - Early detection of shape mismatches before inference
//   - Documentation of expected tensor layouts
//   - Version compatibility checking
//
// # Wire Format
//
//	Magic:      'S','C','H','M' (4 bytes)
//	Version:    uint16 LE
//	NameLen:    uint16 LE
//	Name:       UTF-8 bytes (e.g., "llama-7b-v1")
//	SpecCount:  uint32 LE
//	Specs:      SpecCount × TensorSpec
//
// # TensorSpec Format
//
//	PatternLen: uint16 LE
//	Pattern:    UTF-8 bytes (glob pattern)
//	DType:      uint8
//	Rank:       uint8
//	Dims:       Rank × int64 LE (-1 = any)
//	Flags:      uint8 (bit 0 = optional)
package shard

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"path/filepath"
	"strings"
)

// Schema magic bytes
var SchemaMagic = [4]byte{'S', 'C', 'H', 'M'}

// Schema version
const SchemaVersion = 1

// TensorSpec describes expected properties for matching tensors.
type TensorSpec struct {
	Pattern  string  // Glob pattern (e.g., "layers.*.attention.*.weight")
	DType    DType   // Expected dtype (0 = any)
	Shape    []int64 // Expected shape (-1 = any dimension)
	Optional bool    // Whether tensor is required
}

// ShardSchema defines expected tensors for a shard.
type ShardSchema struct {
	Name    string       // Schema name (e.g., "llama-7b-v1")
	Version uint16       // Schema version
	Specs   []TensorSpec // Tensor specifications
}

// Errors
var (
	ErrInvalidSchemaMagic = errors.New("shard: invalid schema magic")
	ErrSchemaValidation   = errors.New("shard: schema validation failed")
)

// ValidationError describes a schema validation failure.
type ValidationError struct {
	TensorName string
	Message    string
}

func (e ValidationError) Error() string {
	return fmt.Sprintf("%s: %s", e.TensorName, e.Message)
}

// EncodeSchema encodes a schema to bytes.
func EncodeSchema(s *ShardSchema) ([]byte, error) {
	// Calculate size
	size := 4 + 2 + 2 + len(s.Name) + 4 // magic + version + nameLen + name + specCount
	for _, spec := range s.Specs {
		size += 2 + len(spec.Pattern) + 1 + 1 + len(spec.Shape)*8 + 1
	}

	buf := make([]byte, size)
	offset := 0

	// Magic
	copy(buf[offset:], SchemaMagic[:])
	offset += 4

	// Version
	binary.LittleEndian.PutUint16(buf[offset:], s.Version)
	offset += 2

	// Name
	binary.LittleEndian.PutUint16(buf[offset:], uint16(len(s.Name)))
	offset += 2
	copy(buf[offset:], s.Name)
	offset += len(s.Name)

	// Spec count
	binary.LittleEndian.PutUint32(buf[offset:], uint32(len(s.Specs)))
	offset += 4

	// Specs
	for _, spec := range s.Specs {
		// Pattern
		binary.LittleEndian.PutUint16(buf[offset:], uint16(len(spec.Pattern)))
		offset += 2
		copy(buf[offset:], spec.Pattern)
		offset += len(spec.Pattern)

		// DType
		buf[offset] = uint8(spec.DType)
		offset++

		// Rank
		buf[offset] = uint8(len(spec.Shape))
		offset++

		// Dims
		for _, d := range spec.Shape {
			binary.LittleEndian.PutUint64(buf[offset:], uint64(d))
			offset += 8
		}

		// Flags
		var flags uint8
		if spec.Optional {
			flags |= 0x01
		}
		buf[offset] = flags
		offset++
	}

	return buf[:offset], nil
}

// DecodeSchema decodes a schema from bytes.
func DecodeSchema(data []byte) (*ShardSchema, error) {
	if len(data) < 12 {
		return nil, ErrInvalidSchemaMagic
	}

	offset := 0

	// Magic
	var magic [4]byte
	copy(magic[:], data[offset:offset+4])
	if magic != SchemaMagic {
		return nil, ErrInvalidSchemaMagic
	}
	offset += 4

	s := &ShardSchema{}

	// Version
	s.Version = binary.LittleEndian.Uint16(data[offset:])
	offset += 2

	// Name
	nameLen := int(binary.LittleEndian.Uint16(data[offset:]))
	offset += 2
	if offset+nameLen > len(data) {
		return nil, ErrInvalidSchemaMagic
	}
	s.Name = string(data[offset : offset+nameLen])
	offset += nameLen

	// Spec count
	if offset+4 > len(data) {
		return nil, ErrInvalidSchemaMagic
	}
	specCount := int(binary.LittleEndian.Uint32(data[offset:]))
	offset += 4

	s.Specs = make([]TensorSpec, 0, specCount)

	// Specs
	for i := 0; i < specCount; i++ {
		spec := TensorSpec{}

		// Pattern
		if offset+2 > len(data) {
			return nil, ErrInvalidSchemaMagic
		}
		patternLen := int(binary.LittleEndian.Uint16(data[offset:]))
		offset += 2
		if offset+patternLen > len(data) {
			return nil, ErrInvalidSchemaMagic
		}
		spec.Pattern = string(data[offset : offset+patternLen])
		offset += patternLen

		// DType
		if offset+2 > len(data) {
			return nil, ErrInvalidSchemaMagic
		}
		spec.DType = DType(data[offset])
		offset++

		// Rank
		rank := int(data[offset])
		offset++

		// Dims
		if offset+rank*8 > len(data) {
			return nil, ErrInvalidSchemaMagic
		}
		spec.Shape = make([]int64, rank)
		for j := 0; j < rank; j++ {
			rawDim := binary.LittleEndian.Uint64(data[offset:])
			dim, ok := uint64ToInt64(rawDim)
			if !ok {
				return nil, fmt.Errorf("%w: shape dimension %d exceeds int64 range", ErrInvalidSchemaMagic, rawDim)
			}
			spec.Shape[j] = dim
			offset += 8
		}

		// Flags
		if offset >= len(data) {
			return nil, ErrInvalidSchemaMagic
		}
		flags := data[offset]
		offset++
		spec.Optional = flags&0x01 != 0

		s.Specs = append(s.Specs, spec)
	}

	return s, nil
}

// WriteSchema writes a schema to a writer.
func WriteSchema(w io.Writer, s *ShardSchema) error {
	data, err := EncodeSchema(s)
	if err != nil {
		return err
	}
	_, err = w.Write(data)
	return err
}

// ReadSchema reads a schema from a reader.
func ReadSchema(r io.Reader, size int) (*ShardSchema, error) {
	data := make([]byte, size)
	if _, err := io.ReadFull(r, data); err != nil {
		return nil, err
	}
	return DecodeSchema(data)
}

// FindSpec finds the first spec matching the given tensor name.
func (s *ShardSchema) FindSpec(name string) *TensorSpec {
	for i := range s.Specs {
		if matchPattern(s.Specs[i].Pattern, name) {
			return &s.Specs[i]
		}
	}
	return nil
}

// matchPattern matches a glob-style pattern against a name.
// Supports * for single component and ** for multiple components.
func matchPattern(pattern, name string) bool {
	// Handle ** patterns by converting to regex-like matching
	if strings.Contains(pattern, "**") {
		// Split by **
		parts := strings.Split(pattern, "**")
		if len(parts) == 2 {
			prefix := parts[0]
			suffix := parts[1]
			if !strings.HasPrefix(name, strings.TrimSuffix(prefix, ".")) {
				return false
			}
			if suffix != "" && !strings.HasSuffix(name, strings.TrimPrefix(suffix, ".")) {
				return false
			}
			return true
		}
	}

	// Use filepath.Match for single * patterns
	matched, _ := filepath.Match(pattern, name)
	if matched {
		return true
	}

	// Handle dot-separated components with * wildcard
	patternParts := strings.Split(pattern, ".")
	nameParts := strings.Split(name, ".")

	if len(patternParts) != len(nameParts) {
		return false
	}

	for i, p := range patternParts {
		if p == "*" {
			continue
		}
		if p != nameParts[i] {
			return false
		}
	}

	return true
}

// Validate validates tensor metadata against the schema.
func (s *ShardSchema) Validate(tensors map[string]TensorInfo) []ValidationError {
	var errors []ValidationError

	// Check each tensor against matching spec
	for name, info := range tensors {
		spec := s.FindSpec(name)
		if spec == nil {
			// Unknown tensor - not an error unless strict mode
			continue
		}

		// Check dtype
		if spec.DType != 0 && spec.DType != info.DType {
			errors = append(errors, ValidationError{
				TensorName: name,
				Message:    fmt.Sprintf("dtype mismatch: expected %s, got %s", DTypeName(spec.DType), DTypeName(info.DType)),
			})
		}

		// Check shape
		if len(spec.Shape) > 0 {
			if len(spec.Shape) != len(info.Shape) {
				errors = append(errors, ValidationError{
					TensorName: name,
					Message:    fmt.Sprintf("rank mismatch: expected %d, got %d", len(spec.Shape), len(info.Shape)),
				})
			} else {
				for i, expected := range spec.Shape {
					actualDim, ok := uint64ToInt64(info.Shape[i])
					if !ok {
						errors = append(errors, ValidationError{
							TensorName: name,
							Message:    fmt.Sprintf("dim[%d] exceeds int64 range: %d", i, info.Shape[i]),
						})
						break
					}
					if expected >= 0 && expected != actualDim {
						errors = append(errors, ValidationError{
							TensorName: name,
							Message:    fmt.Sprintf("dim[%d] mismatch: expected %d, got %d", i, expected, info.Shape[i]),
						})
						break
					}
				}
			}
		}
	}

	// Check for missing required tensors
	for _, spec := range s.Specs {
		if spec.Optional {
			continue
		}
		found := false
		for name := range tensors {
			if matchPattern(spec.Pattern, name) {
				found = true
				break
			}
		}
		if !found {
			errors = append(errors, ValidationError{
				TensorName: spec.Pattern,
				Message:    "required tensor not found",
			})
		}
	}

	return errors
}

// TensorInfo holds tensor metadata for validation.
type TensorInfo struct {
	DType DType
	Shape []uint64
}

// NewSchemaBuilder creates a builder for constructing schemas.
func NewSchemaBuilder(name string) *SchemaBuilder {
	return &SchemaBuilder{
		schema: &ShardSchema{
			Name:    name,
			Version: SchemaVersion,
			Specs:   make([]TensorSpec, 0),
		},
	}
}

// SchemaBuilder helps construct schemas fluently.
type SchemaBuilder struct {
	schema *ShardSchema
}

// AddSpec adds a tensor specification.
func (b *SchemaBuilder) AddSpec(pattern string, dtype DType, shape ...int64) *SchemaBuilder {
	b.schema.Specs = append(b.schema.Specs, TensorSpec{
		Pattern:  pattern,
		DType:    dtype,
		Shape:    shape,
		Optional: false,
	})
	return b
}

// AddOptionalSpec adds an optional tensor specification.
func (b *SchemaBuilder) AddOptionalSpec(pattern string, dtype DType, shape ...int64) *SchemaBuilder {
	b.schema.Specs = append(b.schema.Specs, TensorSpec{
		Pattern:  pattern,
		DType:    dtype,
		Shape:    shape,
		Optional: true,
	})
	return b
}

// Build returns the constructed schema.
func (b *SchemaBuilder) Build() *ShardSchema {
	return b.schema
}

// ============================================================
// Predefined schemas for common model architectures
// ============================================================

// LlamaSchema returns a schema for LLaMA-style models.
func LlamaSchema(name string, hiddenSize, numLayers int) *ShardSchema {
	b := NewSchemaBuilder(name)

	// Embeddings
	b.AddSpec("model.embed_tokens.weight", DTypeFloat32, -1, int64(hiddenSize))

	// Layers
	for i := 0; i < numLayers; i++ {
		prefix := fmt.Sprintf("model.layers.%d", i)

		// Attention
		b.AddSpec(prefix+".self_attn.q_proj.weight", DTypeFloat32, int64(hiddenSize), int64(hiddenSize))
		b.AddSpec(prefix+".self_attn.k_proj.weight", DTypeFloat32, -1, int64(hiddenSize))
		b.AddSpec(prefix+".self_attn.v_proj.weight", DTypeFloat32, -1, int64(hiddenSize))
		b.AddSpec(prefix+".self_attn.o_proj.weight", DTypeFloat32, int64(hiddenSize), int64(hiddenSize))

		// MLP
		b.AddSpec(prefix+".mlp.gate_proj.weight", DTypeFloat32, -1, int64(hiddenSize))
		b.AddSpec(prefix+".mlp.up_proj.weight", DTypeFloat32, -1, int64(hiddenSize))
		b.AddSpec(prefix+".mlp.down_proj.weight", DTypeFloat32, int64(hiddenSize), -1)

		// Norms
		b.AddSpec(prefix+".input_layernorm.weight", DTypeFloat32, int64(hiddenSize))
		b.AddSpec(prefix+".post_attention_layernorm.weight", DTypeFloat32, int64(hiddenSize))
	}

	// Final norm and head
	b.AddSpec("model.norm.weight", DTypeFloat32, int64(hiddenSize))
	b.AddSpec("lm_head.weight", DTypeFloat32, -1, int64(hiddenSize))

	return b.Build()
}
