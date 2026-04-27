package shard

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestStreamingWriterBasic(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "stream_test.wshard")

	defs := []*WShardChannelDef{
		{Name: "state", DType: "f32", Shape: []int{2}},
	}

	w, err := NewWShardStreamWriter(path, "stream-001", defs,
		WithFlushInterval(4),
		WithMaxTimesteps(100),
	)
	if err != nil {
		t.Fatalf("NewWShardStreamWriter: %v", err)
	}

	if err := w.BeginEpisode("CartPole-v1", WShardTimebase{Type: "ticks", TickHz: 30}); err != nil {
		t.Fatalf("BeginEpisode: %v", err)
	}

	T := 10
	for i := 0; i < T; i++ {
		obsBuf := make([]byte, 2*4) // 2 floats
		binary.LittleEndian.PutUint32(obsBuf[0:4], math.Float32bits(float32(i)))
		binary.LittleEndian.PutUint32(obsBuf[4:8], math.Float32bits(float32(i)*0.5))

		actBuf := make([]byte, 2*4)
		binary.LittleEndian.PutUint32(actBuf[0:4], math.Float32bits(0.1))
		binary.LittleEndian.PutUint32(actBuf[4:8], math.Float32bits(0.2))

		err := w.WriteTimestep(i,
			map[string][]byte{"state": obsBuf},
			map[string][]byte{"state": actBuf},
			float32(i)*0.1,
			i == T-1,
		)
		if err != nil {
			t.Fatalf("WriteTimestep(%d): %v", i, err)
		}
	}

	totalSize, err := w.EndEpisode()
	if err != nil {
		t.Fatalf("EndEpisode: %v", err)
	}
	if totalSize <= 0 {
		t.Fatalf("totalSize = %d, want > 0", totalSize)
	}

	// Verify the .partial file is gone
	if _, err := os.Stat(path + ".partial"); !os.IsNotExist(err) {
		t.Error(".partial file still exists after EndEpisode")
	}

	// Verify the final file exists
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("final file not found: %v", err)
	}
	if info.Size() != totalSize {
		t.Errorf("file size = %d, want %d", info.Size(), totalSize)
	}

	// Read it back with OpenWShard
	ep, err := OpenWShard(path)
	if err != nil {
		t.Fatalf("OpenWShard: %v", err)
	}
	if ep.ID != "stream-001" {
		t.Errorf("ID = %q, want %q", ep.ID, "stream-001")
	}
	if ep.LengthT != T {
		t.Errorf("LengthT = %d, want %d", ep.LengthT, T)
	}
	if len(ep.Rewards) != T {
		t.Errorf("Rewards length = %d, want %d", len(ep.Rewards), T)
	}
	if len(ep.Terminations) != T {
		t.Errorf("Terminations length = %d, want %d", len(ep.Terminations), T)
	}
	if !ep.Terminations[T-1] {
		t.Error("last termination should be true")
	}
}

func TestStreamingWriterCloseWithoutFinalize(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "abandon.wshard")

	defs := []*WShardChannelDef{
		{Name: "x", DType: "f32", Shape: []int{1}},
	}
	w, err := NewWShardStreamWriter(path, "abandon-001", defs)
	if err != nil {
		t.Fatal(err)
	}
	if err := w.BeginEpisode("test", WShardTimebase{Type: "ticks", TickHz: 10}); err != nil {
		t.Fatal(err)
	}

	// Write a few timesteps then abandon
	for i := 0; i < 3; i++ {
		buf := make([]byte, 4)
		binary.LittleEndian.PutUint32(buf, math.Float32bits(1.0))
		w.WriteTimestep(i, map[string][]byte{"x": buf}, nil, 0, false)
	}

	// Close without EndEpisode should clean up .partial
	w.Close()

	if _, err := os.Stat(path + ".partial"); !os.IsNotExist(err) {
		t.Error(".partial file should be removed on Close without finalize")
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Error("final file should not exist when not finalized")
	}
}
