package shard

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
)

// TensorV1 represents a tensor in the Shard v1 blob format.
// This is the canonical storage format for tensors in MoSH profiles.
//
// Wire format:
//
//	Byte 0:     dtype (uint8)
//	Byte 1:     ndim (uint8, 0-255)
//	Bytes 2-3:  flags (uint16, reserved)
//	Bytes 4+:   dims[ndim] as uint64 LE (8 bytes each)
//	Bytes N+:   raw data (row-major, little-endian)
type TensorV1 struct {
	DType DType    // Data type (float32, float16, int8, etc.)
	Dims  []uint64 // Shape dimensions (row-major)
	Data  []byte   // Raw tensor bytes
}

// TensorV1 header constants
const (
	tensorV1HeaderSize = 4 // dtype(1) + ndim(1) + flags(2)
	tensorV1DimSize    = 8 // uint64 little-endian
)

// Errors
var (
	ErrInvalidTensorHeader = errors.New("shard: invalid tensor v1 header")
	ErrTensorDataTooShort  = errors.New("shard: tensor data shorter than declared size")
	ErrTooManyDimensions   = errors.New("shard: tensor has too many dimensions (max 255)")
	ErrInvalidDType        = errors.New("shard: invalid tensor dtype")
	ErrDimensionOverflow   = errors.New("shard: tensor dimension product overflows int64")
)

// isValidDType returns true if dt is a known dtype.
func isValidDType(dt DType) bool {
	return DTypeSize(dt) > 0
}

// numElementsChecked calculates the total number of elements from dimensions
// with overflow detection.
func numElementsChecked(dims []uint64) (int64, error) {
	n := int64(1)
	for _, d := range dims {
		if d > math.MaxInt64 {
			return 0, ErrDimensionOverflow
		}
		di := int64(d)
		if di > 0 && n > math.MaxInt64/di {
			return 0, ErrDimensionOverflow
		}
		n *= di
	}
	return n, nil
}

// TensorV1Size calculates the total byte size of an encoded TensorV1 blob.
// Returns -1 if dtype is invalid or dimensions overflow.
func TensorV1Size(dtype DType, dims []uint64) int64 {
	elementSize := DTypeSize(dtype)
	if elementSize == 0 {
		return -1
	}
	headerSize := int64(tensorV1HeaderSize)
	dimsSize := int64(len(dims)) * tensorV1DimSize
	numElements, err := numElementsChecked(dims)
	if err != nil {
		return -1
	}
	es := int64(elementSize)
	if numElements > math.MaxInt64/es {
		return -1 // Overflow: numElements * elementSize
	}
	dataSize := numElements * es
	return headerSize + dimsSize + dataSize
}

// TensorV1DataSize calculates just the raw data size for a tensor.
// Returns -1 if dtype is invalid or dimensions overflow.
func TensorV1DataSize(dtype DType, dims []uint64) int64 {
	elementSize := DTypeSize(dtype)
	if elementSize == 0 {
		return -1
	}
	numElements, err := numElementsChecked(dims)
	if err != nil {
		return -1
	}
	es := int64(elementSize)
	if numElements > math.MaxInt64/es {
		return -1 // Overflow: numElements * elementSize
	}
	return numElements * es
}

// TensorV1HeaderSize returns the header size for a tensor with ndim dimensions.
func TensorV1HeaderSize(ndim int) int {
	return tensorV1HeaderSize + ndim*tensorV1DimSize
}

// MaxTensorV1HeaderSize is the maximum possible header size (255 dimensions).
const MaxTensorV1HeaderSize = tensorV1HeaderSize + 255*tensorV1DimSize // 2044 bytes

// EncodeTensorV1 encodes a TensorV1 into its binary blob format.
func EncodeTensorV1(t *TensorV1) ([]byte, error) {
	if err := t.Validate(); err != nil {
		return nil, err
	}
	totalSize := tensorV1HeaderSize + len(t.Dims)*tensorV1DimSize + len(t.Data)
	buf := make([]byte, totalSize)

	buf[0] = uint8(t.DType)
	buf[1] = uint8(len(t.Dims))
	buf[2] = 0 // flags low
	buf[3] = 0 // flags high

	offset := tensorV1HeaderSize
	for _, dim := range t.Dims {
		binary.LittleEndian.PutUint64(buf[offset:], dim)
		offset += tensorV1DimSize
	}
	copy(buf[offset:], t.Data)
	return buf, nil
}

// DecodeTensorV1 decodes a TensorV1 from its binary blob format.
func DecodeTensorV1(data []byte) (*TensorV1, error) {
	if len(data) < tensorV1HeaderSize {
		return nil, ErrInvalidTensorHeader
	}

	dtype := DType(data[0])
	ndim := int(data[1])

	if !isValidDType(dtype) {
		return nil, fmt.Errorf("%w: 0x%02x", ErrInvalidDType, uint8(dtype))
	}

	dimsSize := ndim * tensorV1DimSize
	headerAndDimsSize := tensorV1HeaderSize + dimsSize
	if len(data) < headerAndDimsSize {
		return nil, ErrInvalidTensorHeader
	}

	dims := make([]uint64, ndim)
	offset := tensorV1HeaderSize
	for i := 0; i < ndim; i++ {
		dims[i] = binary.LittleEndian.Uint64(data[offset:])
		offset += tensorV1DimSize
	}

	expectedDataSize := TensorV1DataSize(dtype, dims)
	if expectedDataSize < 0 {
		return nil, ErrDimensionOverflow
	}
	actualDataSize := int64(len(data) - headerAndDimsSize)
	if actualDataSize < expectedDataSize {
		return nil, fmt.Errorf("%w: expected %d bytes, got %d",
			ErrTensorDataTooShort, expectedDataSize, actualDataSize)
	}

	tensorData := data[headerAndDimsSize : headerAndDimsSize+int(expectedDataSize)]
	return &TensorV1{
		DType: dtype,
		Dims:  dims,
		Data:  tensorData,
	}, nil
}

// DecodeTensorV1Header decodes only the header portion of a TensorV1 blob.
// Returns dtype, dimensions, and the number of bytes consumed.
func DecodeTensorV1Header(data []byte) (dtype DType, dims []uint64, bytesRead int, err error) {
	if len(data) < tensorV1HeaderSize {
		return 0, nil, 0, ErrInvalidTensorHeader
	}

	dtype = DType(data[0])
	ndim := int(data[1])

	if !isValidDType(dtype) {
		return 0, nil, 0, fmt.Errorf("%w: 0x%02x", ErrInvalidDType, uint8(dtype))
	}

	dimsSize := ndim * tensorV1DimSize
	headerAndDimsSize := tensorV1HeaderSize + dimsSize
	if len(data) < headerAndDimsSize {
		return 0, nil, 0, ErrInvalidTensorHeader
	}

	dims = make([]uint64, ndim)
	offset := tensorV1HeaderSize
	for i := 0; i < ndim; i++ {
		dims[i] = binary.LittleEndian.Uint64(data[offset:])
		offset += tensorV1DimSize
	}

	return dtype, dims, headerAndDimsSize, nil
}

// NumElements returns the total number of elements in the tensor.
// Returns -1 if the dimensions would overflow.
func (t *TensorV1) NumElements() int64 {
	n, err := numElementsChecked(t.Dims)
	if err != nil {
		return -1
	}
	return n
}

// NumElementsChecked returns the total number of elements with overflow checking.
func (t *TensorV1) NumElementsChecked() (int64, error) {
	return numElementsChecked(t.Dims)
}

// ByteSize returns the expected data size in bytes.
// Returns -1 if dimensions overflow or dtype is invalid.
func (t *TensorV1) ByteSize() int64 {
	n := t.NumElements()
	if n < 0 {
		return -1
	}
	elementSize := DTypeSize(t.DType)
	if elementSize == 0 {
		return -1
	}
	if n > math.MaxInt64/int64(elementSize) {
		return -1
	}
	return n * int64(elementSize)
}

// Shape returns a copy of the tensor dimensions.
func (t *TensorV1) Shape() []uint64 {
	shape := make([]uint64, len(t.Dims))
	copy(shape, t.Dims)
	return shape
}

// Validate checks that the tensor data is consistent.
func (t *TensorV1) Validate() error {
	if !isValidDType(t.DType) {
		return ErrInvalidDType
	}
	if len(t.Dims) > 255 {
		return ErrTooManyDimensions
	}
	numElements, err := numElementsChecked(t.Dims)
	if err != nil {
		return err
	}
	expected := numElements * int64(DTypeSize(t.DType))
	actual := int64(len(t.Data))
	if actual < expected {
		return fmt.Errorf("%w: expected %d bytes, got %d",
			ErrTensorDataTooShort, expected, actual)
	}
	return nil
}

// ============================================================
// Constructors
// ============================================================

// NewTensorV1Float32 creates a TensorV1 from float32 data.
// If dims are not provided, creates a 1D tensor.
func NewTensorV1Float32(data []float32, dims ...uint64) *TensorV1 {
	b := make([]byte, len(data)*4)
	for i, v := range data {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(v))
	}
	if len(dims) == 0 {
		dims = []uint64{uint64(len(data))}
	}
	return &TensorV1{DType: DTypeFloat32, Dims: dims, Data: b}
}

// NewTensorV1Float64 creates a TensorV1 from float64 data.
func NewTensorV1Float64(data []float64, dims ...uint64) *TensorV1 {
	b := make([]byte, len(data)*8)
	for i, v := range data {
		binary.LittleEndian.PutUint64(b[i*8:], math.Float64bits(v))
	}
	if len(dims) == 0 {
		dims = []uint64{uint64(len(data))}
	}
	return &TensorV1{DType: DTypeFloat64, Dims: dims, Data: b}
}

// NewTensorV1Int32 creates a TensorV1 from int32 data.
func NewTensorV1Int32(data []int32, dims ...uint64) *TensorV1 {
	b := make([]byte, len(data)*4)
	for i, v := range data {
		binary.LittleEndian.PutUint32(b[i*4:], uint32(v))
	}
	if len(dims) == 0 {
		dims = []uint64{uint64(len(data))}
	}
	return &TensorV1{DType: DTypeInt32, Dims: dims, Data: b}
}

// NewTensorV1Int64 creates a TensorV1 from int64 data.
func NewTensorV1Int64(data []int64, dims ...uint64) *TensorV1 {
	b := make([]byte, len(data)*8)
	for i, v := range data {
		binary.LittleEndian.PutUint64(b[i*8:], uint64(v))
	}
	if len(dims) == 0 {
		dims = []uint64{uint64(len(data))}
	}
	return &TensorV1{DType: DTypeInt64, Dims: dims, Data: b}
}

// NewTensorV1Int8 creates a TensorV1 from int8 data.
func NewTensorV1Int8(data []int8, dims ...uint64) *TensorV1 {
	b := make([]byte, len(data))
	for i, v := range data {
		b[i] = byte(v)
	}
	if len(dims) == 0 {
		dims = []uint64{uint64(len(data))}
	}
	return &TensorV1{DType: DTypeInt8, Dims: dims, Data: b}
}

// ============================================================
// Extractors
// ============================================================

// AsFloat32 extracts tensor data as float32 slice.
// Returns nil if dtype is not float32.
func (t *TensorV1) AsFloat32() []float32 {
	if t.DType != DTypeFloat32 {
		return nil
	}
	return extractFloat32(t.Data)
}

// AsFloat64 extracts tensor data as float64 slice.
func (t *TensorV1) AsFloat64() []float64 {
	if t.DType != DTypeFloat64 {
		return nil
	}
	count := len(t.Data) / 8
	result := make([]float64, count)
	for i := 0; i < count; i++ {
		result[i] = math.Float64frombits(binary.LittleEndian.Uint64(t.Data[i*8:]))
	}
	return result
}

// AsInt32 extracts tensor data as int32 slice.
func (t *TensorV1) AsInt32() []int32 {
	if t.DType != DTypeInt32 {
		return nil
	}
	count := len(t.Data) / 4
	result := make([]int32, count)
	for i := 0; i < count; i++ {
		result[i] = int32(binary.LittleEndian.Uint32(t.Data[i*4:]))
	}
	return result
}

// AsInt64 extracts tensor data as int64 slice.
func (t *TensorV1) AsInt64() []int64 {
	if t.DType != DTypeInt64 {
		return nil
	}
	count := len(t.Data) / 8
	result := make([]int64, count)
	for i := 0; i < count; i++ {
		result[i] = int64(binary.LittleEndian.Uint64(t.Data[i*8:]))
	}
	return result
}

// AsInt8 extracts tensor data as int8 slice.
func (t *TensorV1) AsInt8() []int8 {
	if t.DType != DTypeInt8 {
		return nil
	}
	result := make([]int8, len(t.Data))
	for i, b := range t.Data {
		result[i] = int8(b)
	}
	return result
}

// ============================================================
// Conversion
// ============================================================

// ToFloat32 converts tensor data to float32, handling F16/BF16 conversion.
// Only F32/F16/BF16 dtypes are supported; returns nil for other types.
func (t *TensorV1) ToFloat32() []float32 {
	switch t.DType {
	case DTypeFloat32:
		return extractFloat32(t.Data)
	case DTypeFloat16:
		count := len(t.Data) / 2
		result := make([]float32, count)
		for i := 0; i < count; i++ {
			bits := binary.LittleEndian.Uint16(t.Data[i*2:])
			result[i] = fp16ToFp32(bits)
		}
		return result
	case DTypeBFloat16:
		count := len(t.Data) / 2
		result := make([]float32, count)
		for i := 0; i < count; i++ {
			bits := binary.LittleEndian.Uint16(t.Data[i*2:])
			f32bits := uint32(bits) << 16
			result[i] = math.Float32frombits(f32bits)
		}
		return result
	default:
		return nil
	}
}

// ToFloat32Quantized dequantizes INT8 tensor data using per-channel scales.
// For a [M, N] weight matrix, scales should be [M] float32 values.
func (t *TensorV1) ToFloat32Quantized(scales []float32) []float32 {
	if t.DType != DTypeInt8 {
		return nil
	}
	count := len(t.Data)
	result := make([]float32, count)
	if len(t.Dims) == 0 || len(scales) == 0 {
		return nil
	}
	if len(t.Dims) == 2 {
		rows := int(t.Dims[0])
		cols := int(t.Dims[1])
		if len(scales) != rows {
			scale := scales[0]
			for i := 0; i < count; i++ {
				result[i] = float32(int8(t.Data[i])) * scale
			}
			return result
		}
		for row := 0; row < rows; row++ {
			scale := scales[row]
			rowStart := row * cols
			for col := 0; col < cols; col++ {
				idx := rowStart + col
				result[idx] = float32(int8(t.Data[idx])) * scale
			}
		}
		return result
	}
	scale := scales[0]
	for i := 0; i < count; i++ {
		result[i] = float32(int8(t.Data[i])) * scale
	}
	return result
}

// ToFloat32QuantizedPerRow dequantizes INT8 tensor data using per-row scales.
// Requires a 2D tensor [rows, cols] with exactly len(scales) == rows.
func (t *TensorV1) ToFloat32QuantizedPerRow(scales []float32) ([]float32, error) {
	if t.DType != DTypeInt8 {
		return nil, fmt.Errorf("ToFloat32QuantizedPerRow: expected int8 dtype, got %s", DTypeName(t.DType))
	}
	if len(t.Dims) != 2 {
		return nil, fmt.Errorf("ToFloat32QuantizedPerRow: expected 2D tensor, got %d dimensions", len(t.Dims))
	}
	rows := int(t.Dims[0])
	cols := int(t.Dims[1])
	if len(scales) != rows {
		return nil, fmt.Errorf("ToFloat32QuantizedPerRow: expected %d scales, got %d", rows, len(scales))
	}
	result := make([]float32, rows*cols)
	for row := 0; row < rows; row++ {
		scale := scales[row]
		rowStart := row * cols
		for col := 0; col < cols; col++ {
			idx := rowStart + col
			result[idx] = float32(int8(t.Data[idx])) * scale
		}
	}
	return result, nil
}

// ToFloat32QuantizedPerTensor dequantizes INT8 tensor data using a single scale.
func (t *TensorV1) ToFloat32QuantizedPerTensor(scale float32) []float32 {
	if t.DType != DTypeInt8 {
		return nil
	}
	count := len(t.Data)
	result := make([]float32, count)
	for i := 0; i < count; i++ {
		result[i] = float32(int8(t.Data[i])) * scale
	}
	return result
}

// ============================================================
// Helpers
// ============================================================

func extractFloat32(data []byte) []float32 {
	count := len(data) / 4
	result := make([]float32, count)
	for i := 0; i < count; i++ {
		result[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return result
}

// fp16ToFp32 converts a single FP16 value to FP32.
func fp16ToFp32(h uint16) float32 {
	sign := uint32(h>>15) & 1
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h) & 0x3ff

	var f32bits uint32
	if exp == 0 {
		if mant == 0 {
			f32bits = sign << 31
		} else {
			exp = 1
			for (mant & 0x400) == 0 {
				mant <<= 1
				exp--
			}
			mant &= 0x3ff
			f32bits = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
		}
	} else if exp == 0x1f {
		f32bits = (sign << 31) | (0xff << 23) | (mant << 13)
	} else {
		f32bits = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
	}
	return math.Float32frombits(f32bits)
}
