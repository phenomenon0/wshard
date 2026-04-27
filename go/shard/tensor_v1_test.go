package shard

import (
	"encoding/binary"
	"math"
	"strings"
	"testing"
)

// ============================================================
// Invariant 1: decode(encode(x)) == x — all supported dtypes
// ============================================================

func TestTensorV1RoundtripFloat32(t *testing.T) {
	data := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	orig := NewTensorV1Float32(data, 2, 3)

	blob, err := EncodeTensorV1(orig)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}

	dec, err := DecodeTensorV1(blob)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}

	if dec.DType != DTypeFloat32 {
		t.Errorf("dtype = 0x%02x, want 0x%02x (float32)", uint8(dec.DType), uint8(DTypeFloat32))
	}
	if len(dec.Dims) != 2 || dec.Dims[0] != 2 || dec.Dims[1] != 3 {
		t.Errorf("dims = %v, want [2, 3]", dec.Dims)
	}
	got := dec.AsFloat32()
	if len(got) != len(data) {
		t.Fatalf("len(data) = %d, want %d", len(got), len(data))
	}
	for i, v := range data {
		if got[i] != v {
			t.Errorf("data[%d] = %v, want %v", i, got[i], v)
		}
	}
}

func TestTensorV1RoundtripFloat64(t *testing.T) {
	data := []float64{1.1, 2.2, 3.3}
	orig := NewTensorV1Float64(data)
	blob, err := EncodeTensorV1(orig)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	dec, err := DecodeTensorV1(blob)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if dec.DType != DTypeFloat64 {
		t.Errorf("dtype = 0x%02x, want 0x%02x (float64)", uint8(dec.DType), uint8(DTypeFloat64))
	}
	got := dec.AsFloat64()
	if len(got) != len(data) {
		t.Fatalf("len(data) = %d, want %d", len(got), len(data))
	}
	for i, v := range data {
		if got[i] != v {
			t.Errorf("data[%d] = %v, want %v", i, got[i], v)
		}
	}
}

func TestTensorV1RoundtripInt32(t *testing.T) {
	data := []int32{-1, 0, 1, 42, math.MinInt32, math.MaxInt32}
	orig := NewTensorV1Int32(data, 6)
	blob, err := EncodeTensorV1(orig)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	dec, err := DecodeTensorV1(blob)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	got := dec.AsInt32()
	if len(got) != len(data) {
		t.Fatalf("len(data) = %d, want %d", len(got), len(data))
	}
	for i, v := range data {
		if got[i] != v {
			t.Errorf("data[%d] = %v, want %v", i, got[i], v)
		}
	}
}

func TestTensorV1RoundtripInt64(t *testing.T) {
	data := []int64{math.MinInt64, -1, 0, 1, math.MaxInt64}
	orig := NewTensorV1Int64(data)
	blob, err := EncodeTensorV1(orig)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	dec, err := DecodeTensorV1(blob)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	got := dec.AsInt64()
	if len(got) != len(data) {
		t.Fatalf("len(data) = %d, want %d", len(got), len(data))
	}
	for i, v := range data {
		if got[i] != v {
			t.Errorf("data[%d] = %v, want %v", i, got[i], v)
		}
	}
}

func TestTensorV1RoundtripInt8(t *testing.T) {
	data := []int8{-128, -1, 0, 1, 127}
	orig := NewTensorV1Int8(data)
	blob, err := EncodeTensorV1(orig)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	dec, err := DecodeTensorV1(blob)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	got := dec.AsInt8()
	if len(got) != len(data) {
		t.Fatalf("len(data) = %d, want %d", len(got), len(data))
	}
	for i, v := range data {
		if got[i] != v {
			t.Errorf("data[%d] = %v, want %v", i, got[i], v)
		}
	}
}

func TestTensorV1RoundtripScalar(t *testing.T) {
	// 0-dim scalar tensor
	orig := &TensorV1{
		DType: DTypeFloat32,
		Dims:  []uint64{},
		Data:  make([]byte, 4),
	}
	binary.LittleEndian.PutUint32(orig.Data, math.Float32bits(3.14))

	blob, err := EncodeTensorV1(orig)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	dec, err := DecodeTensorV1(blob)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if dec.NumElements() != 1 {
		t.Errorf("NumElements = %d, want 1", dec.NumElements())
	}
	if len(dec.Dims) != 0 {
		t.Errorf("dims = %v, want []", dec.Dims)
	}
	val := math.Float32frombits(binary.LittleEndian.Uint32(dec.Data))
	if val != 3.14 {
		t.Errorf("scalar value = %v, want 3.14", val)
	}
}

func TestTensorV1RoundtripHighRankTensor(t *testing.T) {
	// 4D tensor [2,3,4,5] = 120 float32 elements
	data := make([]float32, 120)
	for i := range data {
		data[i] = float32(i) * 0.1
	}
	orig := NewTensorV1Float32(data, 2, 3, 4, 5)
	blob, err := EncodeTensorV1(orig)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	dec, err := DecodeTensorV1(blob)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(dec.Dims) != 4 {
		t.Fatalf("ndim = %d, want 4", len(dec.Dims))
	}
	if dec.Dims[0] != 2 || dec.Dims[1] != 3 || dec.Dims[2] != 4 || dec.Dims[3] != 5 {
		t.Errorf("dims = %v, want [2,3,4,5]", dec.Dims)
	}
	got := dec.AsFloat32()
	for i := range data {
		if got[i] != data[i] {
			t.Errorf("data[%d] = %v, want %v", i, got[i], data[i])
			break
		}
	}
}

// ============================================================
// Invariant 2: encode(decode(bytes)) == bytes (canonical)
// ============================================================

func TestTensorV1CanonicalRoundtrip(t *testing.T) {
	// Encode a tensor, decode it, re-encode — bytes must be identical.
	orig := NewTensorV1Float32([]float32{1.0, 2.0, -3.0, 0.0}, 2, 2)
	blob1, err := EncodeTensorV1(orig)
	if err != nil {
		t.Fatalf("encode1: %v", err)
	}

	dec, err := DecodeTensorV1(blob1)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}

	blob2, err := EncodeTensorV1(dec)
	if err != nil {
		t.Fatalf("encode2: %v", err)
	}

	if len(blob1) != len(blob2) {
		t.Fatalf("blob lengths differ: %d vs %d", len(blob1), len(blob2))
	}
	for i := range blob1 {
		if blob1[i] != blob2[i] {
			t.Fatalf("blobs differ at byte %d: 0x%02x vs 0x%02x", i, blob1[i], blob2[i])
		}
	}
}

// ============================================================
// Invariant 3: Error on truncated input (progressive)
// ============================================================

func TestTensorV1RejectsTruncatedInput(t *testing.T) {
	orig := NewTensorV1Float32([]float32{1.0, 2.0, 3.0}, 3)
	blob, err := EncodeTensorV1(orig)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}

	// Progressive truncation: cut at every byte position
	for cutAt := 0; cutAt < len(blob)-1; cutAt++ {
		truncated := blob[:cutAt]
		_, err := DecodeTensorV1(truncated)
		if err == nil {
			t.Errorf("truncated at byte %d/%d: expected error, got nil", cutAt, len(blob))
		}
	}

	// Also test header-only decode with truncation
	for cutAt := 0; cutAt < tensorV1HeaderSize+tensorV1DimSize; cutAt++ {
		truncated := blob[:cutAt]
		_, _, _, err := DecodeTensorV1Header(truncated)
		if err == nil {
			t.Errorf("header truncated at byte %d: expected error, got nil", cutAt)
		}
	}
}

// ============================================================
// Invariant 4: Error on trailing garbage
// ============================================================

func TestTensorV1IgnoresTrailingPadding(t *testing.T) {
	// TensorV1 format allows trailing bytes (padding for alignment).
	// Decode should succeed but only consume expected bytes.
	orig := NewTensorV1Float32([]float32{1.0, 2.0}, 2)
	blob, err := EncodeTensorV1(orig)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}

	// Append garbage — decode should succeed (shard alignment may add padding)
	padded := make([]byte, len(blob)+64)
	copy(padded, blob)
	for i := len(blob); i < len(padded); i++ {
		padded[i] = 0xDE // garbage byte
	}

	dec, err := DecodeTensorV1(padded)
	if err != nil {
		t.Fatalf("decode with padding: %v", err)
	}

	// Verify the data is correct (not contaminated by padding)
	got := dec.AsFloat32()
	if len(got) != 2 || got[0] != 1.0 || got[1] != 2.0 {
		t.Errorf("data = %v, want [1.0, 2.0]", got)
	}

	// Data slice must NOT include the padding bytes
	if int64(len(dec.Data)) != TensorV1DataSize(DTypeFloat32, []uint64{2}) {
		t.Errorf("data length = %d, want %d", len(dec.Data), TensorV1DataSize(DTypeFloat32, []uint64{2}))
	}
}

// ============================================================
// Invariant 5: Deterministic output
// ============================================================

func TestTensorV1DeterministicOutput(t *testing.T) {
	// Same input encoded twice must produce identical bytes.
	for i := 0; i < 10; i++ {
		data := make([]float32, 50)
		for j := range data {
			data[j] = float32(j) * 0.3
		}
		tensor := NewTensorV1Float32(data, 5, 10)

		blob1, err := EncodeTensorV1(tensor)
		if err != nil {
			t.Fatalf("encode1 iter %d: %v", i, err)
		}
		blob2, err := EncodeTensorV1(tensor)
		if err != nil {
			t.Fatalf("encode2 iter %d: %v", i, err)
		}

		if len(blob1) != len(blob2) {
			t.Fatalf("iter %d: lengths differ %d vs %d", i, len(blob1), len(blob2))
		}
		for j := range blob1 {
			if blob1[j] != blob2[j] {
				t.Fatalf("iter %d: differ at byte %d: 0x%02x vs 0x%02x", i, j, blob1[j], blob2[j])
			}
		}
	}
}

// ============================================================
// Invariant 6: NaN/Inf handling
// ============================================================

func TestTensorV1NaNInfRoundtrip(t *testing.T) {
	// TensorV1 is a raw binary format — NaN/Inf must survive roundtrip.
	nan := float32(math.NaN())
	pinf := float32(math.Inf(1))
	ninf := float32(math.Inf(-1))
	nzero := float32(math.Float32frombits(0x80000000)) // -0.0

	orig := NewTensorV1Float32([]float32{nan, pinf, ninf, nzero, 1.0}, 5)
	blob, err := EncodeTensorV1(orig)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}

	dec, err := DecodeTensorV1(blob)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}

	got := dec.AsFloat32()
	if len(got) != 5 {
		t.Fatalf("len = %d, want 5", len(got))
	}

	// NaN: must be NaN (bit pattern preserved)
	if !math.IsNaN(float64(got[0])) {
		t.Errorf("got[0] = %v, want NaN", got[0])
	}
	// +Inf
	if !math.IsInf(float64(got[1]), 1) {
		t.Errorf("got[1] = %v, want +Inf", got[1])
	}
	// -Inf
	if !math.IsInf(float64(got[2]), -1) {
		t.Errorf("got[2] = %v, want -Inf", got[2])
	}
	// -0.0: verify sign bit
	if got[3] != 0 || math.Float32bits(got[3]) != 0x80000000 {
		t.Errorf("got[3] bits = 0x%08x, want 0x80000000 (-0.0)", math.Float32bits(got[3]))
	}
	// Normal value
	if got[4] != 1.0 {
		t.Errorf("got[4] = %v, want 1.0", got[4])
	}
}

func TestTensorV1NaNInfFloat64Roundtrip(t *testing.T) {
	data := []float64{math.NaN(), math.Inf(1), math.Inf(-1)}
	orig := NewTensorV1Float64(data)
	blob, err := EncodeTensorV1(orig)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	dec, err := DecodeTensorV1(blob)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	got := dec.AsFloat64()
	if !math.IsNaN(got[0]) {
		t.Errorf("got[0] = %v, want NaN", got[0])
	}
	if !math.IsInf(got[1], 1) {
		t.Errorf("got[1] = %v, want +Inf", got[1])
	}
	if !math.IsInf(got[2], -1) {
		t.Errorf("got[2] = %v, want -Inf", got[2])
	}
}

// ============================================================
// Invariant 7: Size/depth limits enforced
// ============================================================

func TestTensorV1RejectsTooManyDimensions(t *testing.T) {
	tensor := &TensorV1{DType: DTypeFloat32, Dims: make([]uint64, 256), Data: []byte{}}
	err := tensor.Validate()
	if err != ErrTooManyDimensions {
		t.Errorf("256 dims: got %v, want ErrTooManyDimensions", err)
	}

	// 255 dims is the maximum — should be valid (with matching data)
	dims255 := make([]uint64, 255)
	for i := range dims255 {
		dims255[i] = 1 // each dim is 1, so total elements = 1
	}
	tensor255 := &TensorV1{DType: DTypeFloat32, Dims: dims255, Data: make([]byte, 4)}
	if err := tensor255.Validate(); err != nil {
		t.Errorf("255 dims (all size 1): unexpected error %v", err)
	}
}

// ============================================================
// Class 4: Integer overflow gauntlet
// ============================================================

func TestTensorV1OverflowGauntlet(t *testing.T) {
	// Cases where the dim product itself overflows int64
	t.Run("dim product overflow", func(t *testing.T) {
		dimOverflows := []struct {
			name string
			dims []uint64
		}{
			{"MaxUint64 single dim", []uint64{math.MaxUint64}},
			{"MaxUint64 x 2", []uint64{math.MaxUint64, 2}},
			{"1<<32 x 1<<32", []uint64{1 << 32, 1 << 32}},
			{"three large dims", []uint64{1 << 22, 1 << 22, 1 << 22}}, // 2^66 elements
		}
		for _, tc := range dimOverflows {
			t.Run(tc.name, func(t *testing.T) {
				_, err := numElementsChecked(tc.dims)
				if err == nil {
					t.Error("numElementsChecked should return error")
				}
				if TensorV1Size(DTypeFloat32, tc.dims) != -1 {
					t.Error("TensorV1Size should return -1")
				}
				if TensorV1DataSize(DTypeFloat32, tc.dims) != -1 {
					t.Error("TensorV1DataSize should return -1")
				}
			})
		}
	})

	// Cases where dim product fits int64, but product * elementSize overflows
	t.Run("data size overflow", func(t *testing.T) {
		// MaxInt64/8 + 1 elements * 8 bytes/elem overflows int64
		dims := []uint64{uint64(math.MaxInt64/8 + 1)}
		n, err := numElementsChecked(dims)
		if err != nil {
			t.Fatalf("numElementsChecked should succeed: %v (n=%d)", err, n)
		}
		if TensorV1DataSize(DTypeFloat64, dims) != -1 {
			t.Error("TensorV1DataSize should return -1 for count*elemSize overflow")
		}
		if TensorV1Size(DTypeFloat64, dims) != -1 {
			t.Error("TensorV1Size should return -1 for count*elemSize overflow")
		}
	})
}

func TestTensorV1DecodeOverflowDims(t *testing.T) {
	// Craft a blob that claims huge dimensions but has tiny data.
	// This must error, not allocate terabytes.
	buf := make([]byte, tensorV1HeaderSize+2*tensorV1DimSize)
	buf[0] = uint8(DTypeFloat32)
	buf[1] = 2 // 2 dims
	binary.LittleEndian.PutUint64(buf[4:], math.MaxUint64)
	binary.LittleEndian.PutUint64(buf[12:], 2)

	_, err := DecodeTensorV1(buf)
	if err == nil {
		t.Fatal("expected error for overflowing dims, got nil")
	}
	// Must be specifically an overflow error, not a generic one
	if !strings.Contains(err.Error(), "overflow") {
		t.Errorf("error = %q, want overflow-related error", err.Error())
	}
}

func TestTensorV1DecodeHugeSingleDim(t *testing.T) {
	// Dim claims 1 billion float32 elements (4GB data) but blob is tiny.
	buf := make([]byte, tensorV1HeaderSize+tensorV1DimSize+16)
	buf[0] = uint8(DTypeFloat32)
	buf[1] = 1
	binary.LittleEndian.PutUint64(buf[4:], 1_000_000_000)

	_, err := DecodeTensorV1(buf)
	if err == nil {
		t.Fatal("expected error for data too short, got nil")
	}
	if !strings.Contains(err.Error(), "shorter") {
		t.Errorf("error = %q, want data-too-short error", err.Error())
	}
}

// ============================================================
// Class 5: Resource exhaustion — invalid dtype
// ============================================================

func TestTensorV1RejectsInvalidDType(t *testing.T) {
	invalids := []DType{0x00, 0x0D, 0x0E, 0x0F, 0x15, 0x20, 0xFF}
	for _, dt := range invalids {
		buf := make([]byte, tensorV1HeaderSize)
		buf[0] = uint8(dt)
		buf[1] = 0 // 0 dims

		_, err := DecodeTensorV1(buf)
		if err == nil {
			t.Errorf("dtype 0x%02x: expected error, got nil", uint8(dt))
			continue
		}
		if !strings.Contains(err.Error(), "invalid") {
			t.Errorf("dtype 0x%02x: error = %q, want 'invalid' error", uint8(dt), err.Error())
		}
	}
}

func TestTensorV1ValidateRejectsInvalidDType(t *testing.T) {
	tensor := &TensorV1{DType: DType(0xFF), Dims: []uint64{1}, Data: []byte{0}}
	err := tensor.Validate()
	if err != ErrInvalidDType {
		t.Errorf("got %v, want ErrInvalidDType", err)
	}
}

func TestTensorV1ValidateRejectsDataMismatch(t *testing.T) {
	// Claims [4] float32 (16 bytes) but only has 8 bytes
	tensor := &TensorV1{DType: DTypeFloat32, Dims: []uint64{4}, Data: make([]byte, 8)}
	err := tensor.Validate()
	if err == nil {
		t.Fatal("expected error for data too short")
	}
	if !strings.Contains(err.Error(), "shorter") {
		t.Errorf("error = %q, want data-too-short error", err.Error())
	}
}

// ============================================================
// Header-only decode
// ============================================================

func TestTensorV1HeaderOnlyDecode(t *testing.T) {
	orig := NewTensorV1Float32([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	blob, err := EncodeTensorV1(orig)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}

	dtype, dims, bytesRead, err := DecodeTensorV1Header(blob)
	if err != nil {
		t.Fatalf("header decode: %v", err)
	}
	if dtype != DTypeFloat32 {
		t.Errorf("dtype = 0x%02x, want 0x%02x", uint8(dtype), uint8(DTypeFloat32))
	}
	if len(dims) != 2 || dims[0] != 2 || dims[1] != 3 {
		t.Errorf("dims = %v, want [2, 3]", dims)
	}
	expectedHeader := TensorV1HeaderSize(2)
	if bytesRead != expectedHeader {
		t.Errorf("bytesRead = %d, want %d", bytesRead, expectedHeader)
	}
}

func TestTensorV1HeaderOnlyDecodeInvalidDType(t *testing.T) {
	buf := make([]byte, tensorV1HeaderSize)
	buf[0] = 0xFF // invalid
	buf[1] = 0
	_, _, _, err := DecodeTensorV1Header(buf)
	if err == nil {
		t.Fatal("expected error for invalid dtype in header decode")
	}
	if !strings.Contains(err.Error(), "invalid") {
		t.Errorf("error = %q, want 'invalid' error", err.Error())
	}
}

// ============================================================
// Conversion: F16/BF16 → F32
// ============================================================

func TestTensorV1F16ToF32KnownPatterns(t *testing.T) {
	cases := []struct {
		name string
		bits uint16
		want float32
	}{
		{"1.0", 0x3C00, 1.0},
		{"0.5", 0x3800, 0.5},
		{"-1.0", 0xBC00, -1.0},
		{"+0", 0x0000, 0.0},
		{"-0", 0x8000, float32(math.Copysign(0, -1))},
		{"65504 (max)", 0x7BFF, 65504.0}, // largest representable f16
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			raw := make([]byte, 2)
			binary.LittleEndian.PutUint16(raw, tc.bits)
			tensor := &TensorV1{DType: DTypeFloat16, Dims: []uint64{1}, Data: raw}
			result := tensor.ToFloat32()
			if len(result) != 1 {
				t.Fatalf("len = %d, want 1", len(result))
			}
			if result[0] != tc.want {
				t.Errorf("fp16(0x%04x) → %v, want %v", tc.bits, result[0], tc.want)
			}
		})
	}
}

func TestTensorV1F16NaNInf(t *testing.T) {
	raw := make([]byte, 6) // 3 f16 values
	binary.LittleEndian.PutUint16(raw[0:], 0x7C00)  // +Inf
	binary.LittleEndian.PutUint16(raw[2:], 0xFC00)  // -Inf
	binary.LittleEndian.PutUint16(raw[4:], 0x7E00)  // NaN

	tensor := &TensorV1{DType: DTypeFloat16, Dims: []uint64{3}, Data: raw}
	result := tensor.ToFloat32()
	if !math.IsInf(float64(result[0]), 1) {
		t.Errorf("f16 +Inf → %v", result[0])
	}
	if !math.IsInf(float64(result[1]), -1) {
		t.Errorf("f16 -Inf → %v", result[1])
	}
	if !math.IsNaN(float64(result[2])) {
		t.Errorf("f16 NaN → %v", result[2])
	}
}

func TestTensorV1BF16ToF32(t *testing.T) {
	cases := []struct {
		name string
		bits uint16
		want float32
	}{
		{"1.0", 0x3F80, 1.0},
		{"-1.0", 0xBF80, -1.0},
		{"0.0", 0x0000, 0.0},
		{"2.0", 0x4000, 2.0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			raw := make([]byte, 2)
			binary.LittleEndian.PutUint16(raw, tc.bits)
			tensor := &TensorV1{DType: DTypeBFloat16, Dims: []uint64{1}, Data: raw}
			result := tensor.ToFloat32()
			if len(result) != 1 || result[0] != tc.want {
				t.Errorf("bf16(0x%04x) → %v, want %v", tc.bits, result[0], tc.want)
			}
		})
	}
}

func TestTensorV1ToFloat32RejectsInvalidDType(t *testing.T) {
	tensor := NewTensorV1Int32([]int32{1, 2, 3})
	if tensor.ToFloat32() != nil {
		t.Error("ToFloat32 on int32 tensor should return nil")
	}
}

// ============================================================
// Dequantization
// ============================================================

func TestTensorV1DequantPerTensor(t *testing.T) {
	data := []int8{10, -20, 30, -40}
	tensor := NewTensorV1Int8(data)
	result := tensor.ToFloat32QuantizedPerTensor(0.5)
	expected := []float32{5.0, -10.0, 15.0, -20.0}
	if len(result) != len(expected) {
		t.Fatalf("len = %d, want %d", len(result), len(expected))
	}
	for i, v := range expected {
		if result[i] != v {
			t.Errorf("deq[%d] = %v, want %v", i, result[i], v)
		}
	}
}

func TestTensorV1DequantPerRow(t *testing.T) {
	data := []int8{10, 20, 30, 40}
	tensor := NewTensorV1Int8(data, 2, 2)
	scales := []float32{0.1, 0.5}
	result, err := tensor.ToFloat32QuantizedPerRow(scales)
	if err != nil {
		t.Fatalf("deq: %v", err)
	}
	expected := []float32{1.0, 2.0, 15.0, 20.0}
	if len(result) != len(expected) {
		t.Fatalf("len = %d, want %d", len(result), len(expected))
	}
	for i, v := range expected {
		if math.Abs(float64(result[i]-v)) > 1e-6 {
			t.Errorf("deq[%d] = %v, want %v", i, result[i], v)
		}
	}
}

func TestTensorV1DequantPerRowRejectsWrongDType(t *testing.T) {
	tensor := NewTensorV1Float32([]float32{1, 2, 3, 4}, 2, 2)
	_, err := tensor.ToFloat32QuantizedPerRow([]float32{1, 1})
	if err == nil {
		t.Fatal("expected error for non-int8 dtype")
	}
	if !strings.Contains(err.Error(), "int8") {
		t.Errorf("error = %q, want mention of int8", err.Error())
	}
}

func TestTensorV1DequantPerRowRejectsWrongDims(t *testing.T) {
	tensor := NewTensorV1Int8([]int8{1, 2, 3})
	_, err := tensor.ToFloat32QuantizedPerRow([]float32{1})
	if err == nil {
		t.Fatal("expected error for non-2D tensor")
	}
	if !strings.Contains(err.Error(), "2D") {
		t.Errorf("error = %q, want mention of 2D", err.Error())
	}
}

func TestTensorV1DequantPerRowRejectsScaleCountMismatch(t *testing.T) {
	tensor := NewTensorV1Int8([]int8{1, 2, 3, 4}, 2, 2)
	_, err := tensor.ToFloat32QuantizedPerRow([]float32{1}) // 1 scale for 2 rows
	if err == nil {
		t.Fatal("expected error for scale count mismatch")
	}
	if !strings.Contains(err.Error(), "scales") {
		t.Errorf("error = %q, want mention of scales", err.Error())
	}
}

func TestTensorV1DequantPerTensorRejectsWrongDType(t *testing.T) {
	tensor := NewTensorV1Float32([]float32{1})
	result := tensor.ToFloat32QuantizedPerTensor(1.0)
	if result != nil {
		t.Error("dequant on float32 tensor should return nil")
	}
}

// ============================================================
// Extractor wrong-dtype rejection
// ============================================================

func TestTensorV1ExtractorWrongDType(t *testing.T) {
	tensor := NewTensorV1Float32([]float32{1, 2, 3})
	if tensor.AsFloat64() != nil {
		t.Error("AsFloat64 on float32 should return nil")
	}
	if tensor.AsInt32() != nil {
		t.Error("AsInt32 on float32 should return nil")
	}
	if tensor.AsInt64() != nil {
		t.Error("AsInt64 on float32 should return nil")
	}
	if tensor.AsInt8() != nil {
		t.Error("AsInt8 on float32 should return nil")
	}

	int8tensor := NewTensorV1Int8([]int8{1})
	if int8tensor.AsFloat32() != nil {
		t.Error("AsFloat32 on int8 should return nil")
	}
}

// ============================================================
// Size helpers
// ============================================================

func TestTensorV1SizeHelpers(t *testing.T) {
	// [2,3] float32: header(4) + 2 dims(16) + 6*4 data(24) = 44
	size := TensorV1Size(DTypeFloat32, []uint64{2, 3})
	if size != 44 {
		t.Errorf("TensorV1Size([2,3] f32) = %d, want 44", size)
	}

	dataSize := TensorV1DataSize(DTypeFloat32, []uint64{2, 3})
	if dataSize != 24 {
		t.Errorf("TensorV1DataSize([2,3] f32) = %d, want 24", dataSize)
	}

	headerSize := TensorV1HeaderSize(2)
	if headerSize != 20 {
		t.Errorf("TensorV1HeaderSize(2) = %d, want 20", headerSize)
	}

	// 0 dims (scalar): header(4) + 0 dims + 4 data = 8
	scalarSize := TensorV1Size(DTypeFloat32, []uint64{})
	if scalarSize != 8 {
		t.Errorf("TensorV1Size(scalar f32) = %d, want 8", scalarSize)
	}

	// Invalid dtype returns -1
	if TensorV1Size(DType(0xFF), []uint64{1}) != -1 {
		t.Error("TensorV1Size(invalid) should return -1")
	}
	if TensorV1DataSize(DType(0xFF), []uint64{1}) != -1 {
		t.Error("TensorV1DataSize(invalid) should return -1")
	}
}

func TestTensorV1ByteSize(t *testing.T) {
	tensor := NewTensorV1Float32([]float32{1, 2, 3}, 3)
	if tensor.ByteSize() != 12 {
		t.Errorf("ByteSize = %d, want 12", tensor.ByteSize())
	}

	// Invalid dtype
	bad := &TensorV1{DType: DType(0xFF), Dims: []uint64{1}, Data: []byte{0}}
	if bad.ByteSize() != -1 {
		t.Error("ByteSize for invalid dtype should be -1")
	}
}

func TestTensorV1ShapeReturnsDefensiveCopy(t *testing.T) {
	tensor := NewTensorV1Float32([]float32{1, 2, 3, 4}, 2, 2)
	shape := tensor.Shape()
	shape[0] = 999 // mutate the copy
	if tensor.Dims[0] != 2 {
		t.Error("Shape() should return a defensive copy; original was mutated")
	}
}

// ============================================================
// DType helpers
// ============================================================

func TestDTypeEnumValues(t *testing.T) {
	// Verify canonical values match Agent-GO
	checks := []struct {
		dt   DType
		val  uint8
		name string
	}{
		{DTypeFloat32, 0x01, "Float32"},
		{DTypeFloat16, 0x02, "Float16"},
		{DTypeBFloat16, 0x03, "BFloat16"},
		{DTypeInt8, 0x04, "Int8"},
		{DTypeInt16, 0x05, "Int16"},
		{DTypeInt32, 0x06, "Int32"},
		{DTypeInt64, 0x07, "Int64"},
		{DTypeUint8, 0x08, "Uint8"},
		{DTypeUint16, 0x09, "Uint16"},
		{DTypeUint32, 0x0A, "Uint32"},
		{DTypeUint64, 0x0B, "Uint64"},
		{DTypeFloat64, 0x0C, "Float64"},
		{DTypeQINT4, 0x10, "QINT4"},
		{DTypeQINT2, 0x11, "QINT2"},
		{DTypeQINT3, 0x12, "QINT3"},
		{DTypeTernary, 0x13, "Ternary"},
		{DTypeBinary, 0x14, "Binary"},
	}
	for _, c := range checks {
		if uint8(c.dt) != c.val {
			t.Errorf("DType%s = 0x%02x, want 0x%02x", c.name, uint8(c.dt), c.val)
		}
	}
}

func TestDTypeName(t *testing.T) {
	cases := map[DType]string{
		DTypeFloat32:  "float32",
		DTypeFloat16:  "float16",
		DTypeBFloat16: "bfloat16",
		DTypeFloat64:  "float64",
		DTypeInt8:     "int8",
		DTypeInt16:    "int16",
		DTypeInt32:    "int32",
		DTypeInt64:    "int64",
		DTypeUint8:    "uint8",
		DTypeUint16:   "uint16",
		DTypeUint32:   "uint32",
		DTypeUint64:   "uint64",
		DTypeQINT4:    "qint4",
		DTypeQINT2:    "qint2",
		DTypeQINT3:    "qint3",
		DTypeTernary:  "ternary",
		DTypeBinary:   "binary",
	}
	for dt, want := range cases {
		got := DTypeName(dt)
		if got != want {
			t.Errorf("DTypeName(0x%02x) = %q, want %q", uint8(dt), got, want)
		}
	}
	// Unknown
	unknown := DTypeName(DType(0xFF))
	if !strings.Contains(unknown, "unknown") {
		t.Errorf("DTypeName(0xFF) = %q, want 'unknown' prefix", unknown)
	}
}

func TestParseDTypeName(t *testing.T) {
	// Canonical names + aliases
	cases := []struct {
		name string
		dt   DType
	}{
		{"float32", DTypeFloat32}, {"f32", DTypeFloat32},
		{"float64", DTypeFloat64}, {"f64", DTypeFloat64},
		{"float16", DTypeFloat16}, {"f16", DTypeFloat16},
		{"bfloat16", DTypeBFloat16}, {"bf16", DTypeBFloat16},
		{"int8", DTypeInt8}, {"i8", DTypeInt8},
		{"int16", DTypeInt16}, {"i16", DTypeInt16},
		{"int32", DTypeInt32}, {"i32", DTypeInt32},
		{"int64", DTypeInt64}, {"i64", DTypeInt64},
		{"uint8", DTypeUint8}, {"u8", DTypeUint8},
		{"uint16", DTypeUint16}, {"u16", DTypeUint16},
		{"uint32", DTypeUint32}, {"u32", DTypeUint32},
		{"uint64", DTypeUint64}, {"u64", DTypeUint64},
		{"qint4", DTypeQINT4}, {"q4", DTypeQINT4}, {"int4", DTypeQINT4},
		{"qint2", DTypeQINT2}, {"q2", DTypeQINT2},
		{"qint3", DTypeQINT3}, {"q3", DTypeQINT3},
		{"ternary", DTypeTernary}, {"trit", DTypeTernary},
		{"binary", DTypeBinary}, {"bit", DTypeBinary}, {"b1", DTypeBinary},
	}
	for _, tc := range cases {
		dt, ok := ParseDTypeName(tc.name)
		if !ok {
			t.Errorf("ParseDTypeName(%q) returned false", tc.name)
			continue
		}
		if dt != tc.dt {
			t.Errorf("ParseDTypeName(%q) = 0x%02x, want 0x%02x", tc.name, uint8(dt), uint8(tc.dt))
		}
	}

	// Unknown names must return false
	unknowns := []string{"", "float128", "complex64", "void", "FLOAT32"}
	for _, name := range unknowns {
		_, ok := ParseDTypeName(name)
		if ok {
			t.Errorf("ParseDTypeName(%q) should return false", name)
		}
	}
}

func TestDTypeSize(t *testing.T) {
	cases := map[DType]int{
		DTypeFloat32: 4, DTypeFloat64: 8,
		DTypeFloat16: 2, DTypeBFloat16: 2,
		DTypeInt8: 1, DTypeInt16: 2, DTypeInt32: 4, DTypeInt64: 8,
		DTypeUint8: 1, DTypeUint16: 2, DTypeUint32: 4, DTypeUint64: 8,
		DTypeQINT4: 1, DTypeQINT2: 1, DTypeQINT3: 1,
		DTypeTernary: 1, DTypeBinary: 1,
	}
	for dt, want := range cases {
		if got := DTypeSize(dt); got != want {
			t.Errorf("DTypeSize(%s) = %d, want %d", DTypeName(dt), got, want)
		}
	}
	if DTypeSize(DType(0xFF)) != 0 {
		t.Error("DTypeSize(unknown) should return 0")
	}
}

func TestDTypeBitsPerElement(t *testing.T) {
	cases := map[DType]int{
		DTypeFloat32: 32, DTypeFloat64: 64,
		DTypeFloat16: 16, DTypeBFloat16: 16,
		DTypeInt8: 8, DTypeUint8: 8,
		DTypeQINT4: 4, DTypeQINT3: 3, DTypeQINT2: 2,
		DTypeTernary: 2, DTypeBinary: 1,
	}
	for dt, want := range cases {
		if got := DTypeBitsPerElement(dt); got != want {
			t.Errorf("DTypeBitsPerElement(%s) = %d, want %d", DTypeName(dt), got, want)
		}
	}
	if DTypeBitsPerElement(DType(0xFF)) != 0 {
		t.Error("DTypeBitsPerElement(unknown) should return 0")
	}
}

func TestDTypeIsQuantized(t *testing.T) {
	quantized := []DType{DTypeQINT4, DTypeQINT2, DTypeQINT3, DTypeTernary, DTypeBinary}
	for _, dt := range quantized {
		if !DTypeIsQuantized(dt) {
			t.Errorf("%s should be quantized", DTypeName(dt))
		}
	}
	notQuantized := []DType{DTypeFloat32, DTypeFloat16, DTypeBFloat16, DTypeFloat64,
		DTypeInt8, DTypeInt16, DTypeInt32, DTypeInt64,
		DTypeUint8, DTypeUint16, DTypeUint32, DTypeUint64}
	for _, dt := range notQuantized {
		if DTypeIsQuantized(dt) {
			t.Errorf("%s should not be quantized", DTypeName(dt))
		}
	}
}

// ============================================================
// Encode validation
// ============================================================

func TestTensorV1EncodeRejectsInvalid(t *testing.T) {
	// Invalid dtype
	_, err := EncodeTensorV1(&TensorV1{DType: DType(0xFF), Dims: []uint64{1}, Data: []byte{0}})
	if err == nil {
		t.Error("encode with invalid dtype should fail")
	}

	// Data size mismatch
	_, err = EncodeTensorV1(&TensorV1{DType: DTypeFloat32, Dims: []uint64{4}, Data: make([]byte, 8)})
	if err == nil {
		t.Error("encode with data too short should fail")
	}

	// Too many dims
	_, err = EncodeTensorV1(&TensorV1{DType: DTypeFloat32, Dims: make([]uint64, 256)})
	if err == nil {
		t.Error("encode with 256 dims should fail")
	}
}
