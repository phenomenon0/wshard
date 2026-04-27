// wshard_residual.go — Sign2ndDiff residual encoding for W-SHARD.
//
// The 1-bit curvature-sign residual captures whether each point is:
//   - +1: Above the line between neighbors (convex/peak)
//   - -1: Below the line between neighbors (concave/dip)
//   - 0:  On the line (linear)
//
// This matches the Python wshard.residual module bit-for-bit.
package shard

import "math"

// ComputeSign2ndDiff computes the sign of the second difference for a 1D signal.
// Returns int8 values in {-1, 0, +1} with edges padded to 0.
func ComputeSign2ndDiff(signal []float32) []int8 {
	T := len(signal)
	if T < 3 {
		return make([]int8, T)
	}

	residuals := make([]int8, T)
	// edges stay 0 (pad rule)
	for i := 1; i < T-1; i++ {
		diff := 2.0*float64(signal[i]) - float64(signal[i-1]) - float64(signal[i+1])
		if diff > 0 {
			residuals[i] = 1
		} else if diff < 0 {
			residuals[i] = -1
		}
	}
	return residuals
}

// ComputeSign2ndDiffMultiDim computes sign2nddiff residuals for each column of a [T][D] signal.
func ComputeSign2ndDiffMultiDim(signal [][]float32) [][]int8 {
	T := len(signal)
	if T == 0 {
		return nil
	}
	D := len(signal[0])
	result := make([][]int8, T)
	for i := range result {
		result[i] = make([]int8, D)
	}
	if T < 3 {
		return result
	}
	for d := 0; d < D; d++ {
		for i := 1; i < T-1; i++ {
			diff := 2.0*float64(signal[i][d]) - float64(signal[i-1][d]) - float64(signal[i+1][d])
			if diff > 0 {
				result[i][d] = 1
			} else if diff < 0 {
				result[i][d] = -1
			}
		}
	}
	return result
}

// PackResidualBits packs residuals into bits (LSB-first).
// +1 maps to bit 1, all else maps to bit 0.
// Returns ceil(T/8) bytes.
func PackResidualBits(residuals []int8) []byte {
	T := len(residuals)
	numBytes := (T + 7) / 8
	packed := make([]byte, numBytes)
	for i := 0; i < T; i++ {
		if residuals[i] > 0 {
			packed[i/8] |= 1 << uint(i%8)
		}
	}
	return packed
}

// UnpackResidualBits unpacks residual bits to int8.
// Bit 1 → +1, Bit 0 → -1.
func UnpackResidualBits(packed []byte, T int) []int8 {
	residuals := make([]int8, T)
	for i := 0; i < T; i++ {
		if packed[i/8]&(1<<uint(i%8)) != 0 {
			residuals[i] = 1
		} else {
			residuals[i] = -1
		}
	}
	return residuals
}

// PackMultiDimResidualBits packs [T][D] residuals. Layout: D * ceil(T/8) bytes,
// dimensions concatenated.
func PackMultiDimResidualBits(residuals [][]int8) []byte {
	T := len(residuals)
	if T == 0 {
		return nil
	}
	D := len(residuals[0])
	bytesPerDim := (T + 7) / 8
	packed := make([]byte, D*bytesPerDim)

	for d := 0; d < D; d++ {
		offset := d * bytesPerDim
		for i := 0; i < T; i++ {
			if residuals[i][d] > 0 {
				packed[offset+i/8] |= 1 << uint(i%8)
			}
		}
	}
	return packed
}

// UnpackMultiDimResidualBits unpacks D * ceil(T/8) bytes to [T][D] residuals.
func UnpackMultiDimResidualBits(packed []byte, T, D int) [][]int8 {
	bytesPerDim := (T + 7) / 8
	result := make([][]int8, T)
	for i := range result {
		result[i] = make([]int8, D)
	}
	for d := 0; d < D; d++ {
		offset := d * bytesPerDim
		for i := 0; i < T; i++ {
			if packed[offset+i/8]&(1<<uint(i%8)) != 0 {
				result[i][d] = 1
			} else {
				result[i][d] = -1
			}
		}
	}
	return result
}

// QuantizeDelta quantizes the signal-omen delta to int8 with per-window scales.
// Returns (quantized int8 deltas, float32 scales per window).
func QuantizeDelta(signal, omen []float32, windowSize int) ([]int8, []float32) {
	T := len(signal)
	numWindows := (T + windowSize - 1) / windowSize
	quantized := make([]int8, T)
	scales := make([]float32, numWindows)

	for w := 0; w < numWindows; w++ {
		start := w * windowSize
		end := start + windowSize
		if end > T {
			end = T
		}

		// Compute max absolute delta in window
		var maxAbs float64
		for i := start; i < end; i++ {
			d := math.Abs(float64(signal[i]) - float64(omen[i]))
			if d > maxAbs {
				maxAbs = d
			}
		}

		// Compute scale
		scale := float32(1.0)
		if maxAbs > 0 {
			scale = float32(maxAbs / 127.0)
		}
		scales[w] = scale

		// Quantize
		for i := start; i < end; i++ {
			delta := float64(signal[i]) - float64(omen[i])
			q := math.Round(delta / float64(scale))
			if q > 127 {
				q = 127
			} else if q < -127 {
				q = -127
			}
			quantized[i] = int8(q)
		}
	}

	return quantized, scales
}

// DequantizeDelta dequantizes int8 deltas using per-window scales.
func DequantizeDelta(quantized []int8, scales []float32, windowSize int) []float32 {
	T := len(quantized)
	delta := make([]float32, T)

	for w := 0; w < len(scales); w++ {
		start := w * windowSize
		end := start + windowSize
		if end > T {
			end = T
		}
		for i := start; i < end; i++ {
			delta[i] = float32(quantized[i]) * scales[w]
		}
	}
	return delta
}

// ReconstructFromOmenAndDelta reconstructs signal from omen + quantized delta.
func ReconstructFromOmenAndDelta(omen []float32, quantized []int8, scales []float32, windowSize int) []float32 {
	delta := DequantizeDelta(quantized, scales, windowSize)
	result := make([]float32, len(omen))
	for i := range result {
		result[i] = omen[i] + delta[i]
	}
	return result
}
