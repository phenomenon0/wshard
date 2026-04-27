package shard

import "fmt"

// DType represents tensor data types. Mirrors cowrie.DType for standalone use.
type DType uint8

const (
	DTypeUnknown  DType = 0x00
	DTypeFloat32  DType = 0x01
	DTypeFloat16  DType = 0x02
	DTypeBFloat16 DType = 0x03
	DTypeInt8     DType = 0x04
	DTypeInt16    DType = 0x05
	DTypeInt32    DType = 0x06
	DTypeInt64    DType = 0x07
	DTypeUint8    DType = 0x08
	DTypeUint16   DType = 0x09
	DTypeUint32   DType = 0x0A
	DTypeUint64   DType = 0x0B
	DTypeFloat64  DType = 0x0C
	// Quantized types (0x10-0x1F)
	DTypeQINT4   DType = 0x10 // 4-bit quantized integer
	DTypeQINT2   DType = 0x11 // 2-bit quantized integer
	DTypeQINT3   DType = 0x12 // 3-bit quantized integer
	DTypeTernary DType = 0x13 // Ternary (-1, 0, 1)
	DTypeBinary  DType = 0x14 // Binary (0, 1)
)

// DTypeName returns a human-readable name for a DType.
func DTypeName(dt DType) string {
	switch dt {
	case DTypeFloat32:
		return "float32"
	case DTypeFloat16:
		return "float16"
	case DTypeBFloat16:
		return "bfloat16"
	case DTypeFloat64:
		return "float64"
	case DTypeInt8:
		return "int8"
	case DTypeInt16:
		return "int16"
	case DTypeInt32:
		return "int32"
	case DTypeInt64:
		return "int64"
	case DTypeUint8:
		return "uint8"
	case DTypeUint16:
		return "uint16"
	case DTypeUint32:
		return "uint32"
	case DTypeUint64:
		return "uint64"
	case DTypeQINT4:
		return "qint4"
	case DTypeQINT2:
		return "qint2"
	case DTypeQINT3:
		return "qint3"
	case DTypeTernary:
		return "ternary"
	case DTypeBinary:
		return "binary"
	default:
		return fmt.Sprintf("unknown(0x%02x)", uint8(dt))
	}
}

// DTypeSize returns the byte size of a single element for the given dtype.
// Returns 0 for unknown dtypes.
// For packed quantized types (QINT4, QINT2, etc.), returns 1 as a placeholder.
// Use DTypeBitsPerElement for accurate sub-byte sizes.
func DTypeSize(dt DType) int {
	switch dt {
	case DTypeFloat32, DTypeInt32, DTypeUint32:
		return 4
	case DTypeFloat64, DTypeInt64, DTypeUint64:
		return 8
	case DTypeFloat16, DTypeBFloat16, DTypeInt16, DTypeUint16:
		return 2
	case DTypeInt8, DTypeUint8:
		return 1
	case DTypeQINT4, DTypeQINT2, DTypeQINT3, DTypeTernary, DTypeBinary:
		return 1
	default:
		return 0
	}
}

// DTypeBitsPerElement returns the bits per logical element for the dtype.
// For packed types, this is the sub-byte bit count.
// For standard types, returns bytes * 8.
func DTypeBitsPerElement(dt DType) int {
	switch dt {
	case DTypeFloat32, DTypeInt32, DTypeUint32:
		return 32
	case DTypeFloat64, DTypeInt64, DTypeUint64:
		return 64
	case DTypeFloat16, DTypeBFloat16, DTypeInt16, DTypeUint16:
		return 16
	case DTypeInt8, DTypeUint8:
		return 8
	case DTypeQINT4:
		return 4
	case DTypeQINT3:
		return 3
	case DTypeQINT2:
		return 2
	case DTypeTernary:
		return 2
	case DTypeBinary:
		return 1
	default:
		return 0
	}
}

// DTypeIsQuantized returns true if the dtype is a packed quantized type.
func DTypeIsQuantized(dt DType) bool {
	switch dt {
	case DTypeQINT4, DTypeQINT2, DTypeQINT3, DTypeTernary, DTypeBinary:
		return true
	default:
		return false
	}
}

// ParseDTypeName parses a dtype name string and returns the corresponding DType.
// Returns the dtype and true on success, or 0 and false for unknown names.
// Accepts canonical names (float32, int8, etc.) and common aliases (f32, i8, etc.).
func ParseDTypeName(name string) (DType, bool) {
	switch name {
	case "float32", "f32":
		return DTypeFloat32, true
	case "float64", "f64":
		return DTypeFloat64, true
	case "float16", "f16":
		return DTypeFloat16, true
	case "bfloat16", "bf16":
		return DTypeBFloat16, true
	case "int8", "i8":
		return DTypeInt8, true
	case "int16", "i16":
		return DTypeInt16, true
	case "int32", "i32":
		return DTypeInt32, true
	case "int64", "i64":
		return DTypeInt64, true
	case "uint8", "u8":
		return DTypeUint8, true
	case "uint16", "u16":
		return DTypeUint16, true
	case "uint32", "u32":
		return DTypeUint32, true
	case "uint64", "u64":
		return DTypeUint64, true
	case "qint4", "q4", "int4":
		return DTypeQINT4, true
	case "qint2", "q2", "int2":
		return DTypeQINT2, true
	case "qint3", "q3", "int3":
		return DTypeQINT3, true
	case "ternary", "trit":
		return DTypeTernary, true
	case "binary", "bit", "b1":
		return DTypeBinary, true
	default:
		return 0, false
	}
}
