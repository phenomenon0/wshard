package shard

import (
	"math"
	"testing"
)

func TestComputeSign2ndDiff(t *testing.T) {
	// Linear signal: 1, 2, 3, 4, 5 → all interior points are 0
	linear := []float32{1, 2, 3, 4, 5}
	got := ComputeSign2ndDiff(linear)
	for i, v := range got {
		if v != 0 {
			t.Errorf("linear signal[%d] = %d, want 0", i, v)
		}
	}

	// Convex peak: 0, 0, 10, 0, 0 → middle is +1
	peak := []float32{0, 0, 10, 0, 0}
	got = ComputeSign2ndDiff(peak)
	if got[2] != 1 {
		t.Errorf("peak[2] = %d, want +1", got[2])
	}

	// Concave dip: 10, 10, 0, 10, 10 → middle is -1
	dip := []float32{10, 10, 0, 10, 10}
	got = ComputeSign2ndDiff(dip)
	if got[2] != -1 {
		t.Errorf("dip[2] = %d, want -1", got[2])
	}

	// Edges are always 0
	if got[0] != 0 || got[4] != 0 {
		t.Errorf("edges should be 0, got [0]=%d, [4]=%d", got[0], got[4])
	}

	// Short signals
	if len(ComputeSign2ndDiff([]float32{1, 2})) != 2 {
		t.Error("T<3 should return zeros")
	}
	if len(ComputeSign2ndDiff(nil)) != 0 {
		t.Error("nil should return empty")
	}
}

func TestComputeSign2ndDiffMultiDim(t *testing.T) {
	signal := [][]float32{
		{0, 10},
		{0, 10},
		{10, 0},
		{0, 10},
		{0, 10},
	}
	got := ComputeSign2ndDiffMultiDim(signal)
	// dim 0: 0,0,10,0,0 → [2] = +1
	if got[2][0] != 1 {
		t.Errorf("dim0[2] = %d, want +1", got[2][0])
	}
	// dim 1: 10,10,0,10,10 → [2] = -1
	if got[2][1] != -1 {
		t.Errorf("dim1[2] = %d, want -1", got[2][1])
	}
}

func TestPackUnpackResidualBits(t *testing.T) {
	residuals := []int8{1, -1, 0, 1, -1, 1, -1, -1, 1, 0}
	packed := PackResidualBits(residuals)
	expectedBytes := (len(residuals) + 7) / 8
	if len(packed) != expectedBytes {
		t.Fatalf("packed length = %d, want %d", len(packed), expectedBytes)
	}

	unpacked := UnpackResidualBits(packed, len(residuals))
	for i, r := range residuals {
		expected := int8(-1)
		if r > 0 {
			expected = 1
		}
		if unpacked[i] != expected {
			t.Errorf("unpacked[%d] = %d, want %d", i, unpacked[i], expected)
		}
	}
}

func TestPackUnpackMultiDimResidualBits(t *testing.T) {
	// 5 timesteps, 3 dims
	residuals := [][]int8{
		{1, -1, 1},
		{-1, 1, -1},
		{1, 1, -1},
		{-1, -1, 1},
		{1, -1, -1},
	}
	T, D := 5, 3
	packed := PackMultiDimResidualBits(residuals)
	expectedLen := D * ((T + 7) / 8)
	if len(packed) != expectedLen {
		t.Fatalf("packed length = %d, want %d", len(packed), expectedLen)
	}

	unpacked := UnpackMultiDimResidualBits(packed, T, D)
	for i := 0; i < T; i++ {
		for d := 0; d < D; d++ {
			expected := int8(-1)
			if residuals[i][d] > 0 {
				expected = 1
			}
			if unpacked[i][d] != expected {
				t.Errorf("unpacked[%d][%d] = %d, want %d", i, d, unpacked[i][d], expected)
			}
		}
	}
}

func TestQuantizeDequantizeDelta(t *testing.T) {
	signal := make([]float32, 100)
	omen := make([]float32, 100)
	for i := range signal {
		signal[i] = float32(i) * 0.1
		omen[i] = float32(i)*0.1 + 0.01 // small offset
	}

	quantized, scales := QuantizeDelta(signal, omen, 32)
	if len(quantized) != 100 {
		t.Fatalf("quantized length = %d, want 100", len(quantized))
	}
	if len(scales) != 4 { // ceil(100/32)
		t.Fatalf("scales length = %d, want 4", len(scales))
	}

	delta := DequantizeDelta(quantized, scales, 32)
	if len(delta) != 100 {
		t.Fatalf("delta length = %d, want 100", len(delta))
	}

	reconstructed := ReconstructFromOmenAndDelta(omen, quantized, scales, 32)
	for i := range reconstructed {
		err := math.Abs(float64(reconstructed[i] - signal[i]))
		if err > 0.01 { // quantization error should be small for this test
			t.Errorf("reconstructed[%d] = %f, signal = %f, error = %f",
				i, reconstructed[i], signal[i], err)
		}
	}
}
