package shard

import (
	"os"
	"path/filepath"
	"testing"
)

func TestChunkedEpisodeRoundtrip(t *testing.T) {
	dir := t.TempDir()

	// Create a full episode with 20 timesteps
	T := 20
	ep := &WShardEpisode{
		ID:      "chunked-001",
		EnvID:   "test-env",
		LengthT: T,
		Timebase: WShardTimebase{
			Type:   "ticks",
			TickHz: 30,
		},
		Observations: map[string]*WShardChannel{
			"state": {
				Name:  "state",
				DType: "f32",
				Shape: []int{2},
				Data:  make([]byte, T*2*4),
			},
		},
		Actions: map[string]*WShardChannel{
			"ctrl": {
				Name:  "ctrl",
				DType: "f32",
				Shape: []int{1},
				Data:  make([]byte, T*1*4),
			},
		},
		Rewards:      make([]float32, T),
		Terminations: make([]bool, T),
	}
	// Fill with pattern
	for i := range ep.Observations["state"].Data {
		ep.Observations["state"].Data[i] = byte(i % 256)
	}
	ep.Terminations[T-1] = true

	// Write chunked (5 timesteps per chunk → 4 chunks)
	writer := NewChunkedEpisodeWriter(dir, "chunked-001", 5)
	paths, err := writer.WriteEpisodeChunked(ep)
	if err != nil {
		t.Fatalf("WriteEpisodeChunked: %v", err)
	}
	if len(paths) != 4 {
		t.Fatalf("chunk count = %d, want 4", len(paths))
	}

	// Finalize manifest
	manifestPath, err := writer.FinalizeManifest()
	if err != nil {
		t.Fatalf("FinalizeManifest: %v", err)
	}
	if _, err := os.Stat(manifestPath); err != nil {
		t.Fatalf("manifest file not found: %v", err)
	}

	// Read back
	reader := NewChunkedEpisodeReader(manifestPath)
	manifest, err := reader.LoadManifest()
	if err != nil {
		t.Fatalf("LoadManifest: %v", err)
	}
	if manifest.EpisodeID != "chunked-001" {
		t.Errorf("EpisodeID = %q, want %q", manifest.EpisodeID, "chunked-001")
	}
	if len(manifest.Chunks) != 4 {
		t.Fatalf("manifest chunks = %d, want 4", len(manifest.Chunks))
	}
	if manifest.TotalTimesteps != T {
		t.Errorf("TotalTimesteps = %d, want %d", manifest.TotalTimesteps, T)
	}

	// Validate continuity
	if err := ValidateChunkContinuity(manifest); err != nil {
		t.Fatalf("ValidateChunkContinuity: %v", err)
	}

	// Load specific chunk
	chunk1, err := reader.LoadChunk(1)
	if err != nil {
		t.Fatalf("LoadChunk(1): %v", err)
	}
	if chunk1.LengthT != 5 {
		t.Errorf("chunk1 LengthT = %d, want 5", chunk1.LengthT)
	}

	// Iterate all chunks
	var totalT int
	err = reader.IterChunks(func(ep *WShardEpisode) error {
		totalT += ep.LengthT
		return nil
	})
	if err != nil {
		t.Fatalf("IterChunks: %v", err)
	}
	if totalT != T {
		t.Errorf("total timesteps from iteration = %d, want %d", totalT, T)
	}
}

func TestChunkManifestJSON(t *testing.T) {
	m := &ChunkManifest{
		EpisodeID: "test-ep",
		EnvID:     "test-env",
	}
	m.AddChunk(ChunkEntry{
		ChunkIndex:    0,
		URI:           "chunk_0000.wshard",
		SHA256:        "abc123",
		TimestepRange: [2]int{0, 99},
		LengthT:       100,
	})
	m.AddChunk(ChunkEntry{
		ChunkIndex:    1,
		URI:           "chunk_0001.wshard",
		SHA256:        "def456",
		TimestepRange: [2]int{100, 199},
		LengthT:       100,
	})

	data, err := m.ToJSON()
	if err != nil {
		t.Fatalf("ToJSON: %v", err)
	}

	got, err := ChunkManifestFromJSON(data)
	if err != nil {
		t.Fatalf("FromJSON: %v", err)
	}
	if got.EpisodeID != "test-ep" {
		t.Errorf("EpisodeID = %q, want %q", got.EpisodeID, "test-ep")
	}
	if len(got.Chunks) != 2 {
		t.Fatalf("chunks = %d, want 2", len(got.Chunks))
	}
	if got.TotalTimesteps != 200 {
		t.Errorf("TotalTimesteps = %d, want 200", got.TotalTimesteps)
	}
}

func TestValidateChunkContinuityErrors(t *testing.T) {
	// Gap in indices
	m := &ChunkManifest{
		EpisodeID:      "test",
		TotalTimesteps: 200,
		Chunks: []ChunkEntry{
			{ChunkIndex: 0, TimestepRange: [2]int{0, 99}, LengthT: 100},
			{ChunkIndex: 2, TimestepRange: [2]int{100, 199}, LengthT: 100},
		},
	}
	if err := ValidateChunkContinuity(m); err == nil {
		t.Error("expected error for gap in indices")
	}

	// Timestep gap
	m2 := &ChunkManifest{
		EpisodeID:      "test",
		TotalTimesteps: 200,
		Chunks: []ChunkEntry{
			{ChunkIndex: 0, TimestepRange: [2]int{0, 99}, LengthT: 100},
			{ChunkIndex: 1, TimestepRange: [2]int{105, 199}, LengthT: 100},
		},
	}
	if err := ValidateChunkContinuity(m2); err == nil {
		t.Error("expected error for timestep gap")
	}

	// Total length mismatch
	m3 := &ChunkManifest{
		EpisodeID:      "test",
		TotalTimesteps: 999,
		Chunks: []ChunkEntry{
			{ChunkIndex: 0, TimestepRange: [2]int{0, 99}, LengthT: 100},
			{ChunkIndex: 1, TimestepRange: [2]int{100, 199}, LengthT: 100},
		},
	}
	if err := ValidateChunkContinuity(m3); err == nil {
		t.Error("expected error for total length mismatch")
	}
}

func TestBlockNaming(t *testing.T) {
	cases := []struct {
		fn   func() string
		want string
	}{
		{func() string { return OmenBlockName("joint_pos", "dreamer") }, "omen/joint_pos/dreamer"},
		{func() string { return UncertBlockName("joint_pos", "dreamer", "variance") }, "uncert/joint_pos/dreamer/variance"},
		{func() string { return ResidualSign2ndDiffBlockName("joint_pos") }, "residual/joint_pos/sign2nddiff"},
		{func() string { return LatentActionBlockName("genie3") }, "omen/latent_action/genie3"},
		{func() string { return LatentCodebookBlockName("genie3") }, "omen/latent_action_codebook/genie3"},
		{func() string { return MultiModalSignalBlockName("obs", ModalityRGB) }, "signal/obs/rgb"},
	}

	for _, tc := range cases {
		got := tc.fn()
		if got != tc.want {
			t.Errorf("got %q, want %q", got, tc.want)
		}
	}
}

func TestOmenUncertResidualRoundtrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "omen.wshard")

	T := 10
	ep := &WShardEpisode{
		ID:      "omen-ep",
		EnvID:   "test-env",
		LengthT: T,
		Timebase: WShardTimebase{
			Type:   "ticks",
			TickHz: 30,
		},
		Observations: map[string]*WShardChannel{
			"joint_pos": {Name: "joint_pos", DType: "f32", Shape: []int{3}, Data: make([]byte, T*3*4)},
		},
		Actions: map[string]*WShardChannel{
			"torque": {Name: "torque", DType: "f32", Shape: []int{3}, Data: make([]byte, T*3*4)},
		},
		Rewards:      make([]float32, T),
		Terminations: make([]bool, T),
		Omens: map[string]map[string]*WShardChannel{
			"joint_pos": {
				"dreamer": {Name: "omen_jp_dreamer", DType: "f32", Shape: []int{3}, Data: make([]byte, T*3*4)},
			},
		},
		Uncerts: map[string]*WShardChannel{
			"uncert/joint_pos/dreamer/variance": {
				Name: "uncert_jp_dreamer_var", DType: "f32", Shape: []int{3}, Data: make([]byte, T*3*4),
			},
		},
		Residuals: map[string]*WShardResidual{
			"joint_pos": {
				ChannelID: "joint_pos",
				Type:      "sign2nddiff",
				Data:      PackResidualBits(ComputeSign2ndDiff(make([]float32, T))),
			},
		},
	}
	ep.Terminations[T-1] = true

	// Fill omen data with pattern
	for i := range ep.Omens["joint_pos"]["dreamer"].Data {
		ep.Omens["joint_pos"]["dreamer"].Data[i] = byte((i * 3) % 256)
	}
	for i := range ep.Uncerts["uncert/joint_pos/dreamer/variance"].Data {
		ep.Uncerts["uncert/joint_pos/dreamer/variance"].Data[i] = byte((i * 5) % 256)
	}

	if err := CreateWShard(path, ep); err != nil {
		t.Fatalf("CreateWShard: %v", err)
	}

	got, err := OpenWShard(path)
	if err != nil {
		t.Fatalf("OpenWShard: %v", err)
	}

	// Verify omens
	if len(got.Omens) == 0 {
		t.Fatal("Omens is empty")
	}
	if got.Omens["joint_pos"] == nil {
		t.Fatal("Omens[joint_pos] is nil")
	}
	omenCh := got.Omens["joint_pos"]["dreamer"]
	if omenCh == nil {
		t.Fatal("Omens[joint_pos][dreamer] is nil")
	}
	if len(omenCh.Data) != T*3*4 {
		t.Errorf("omen data length = %d, want %d", len(omenCh.Data), T*3*4)
	}
	for i := range omenCh.Data {
		if omenCh.Data[i] != byte((i*3)%256) {
			t.Errorf("omen data[%d] = %d, want %d", i, omenCh.Data[i], byte((i*3)%256))
			break
		}
	}

	// Verify uncerts
	if len(got.Uncerts) == 0 {
		t.Fatal("Uncerts is empty")
	}
	uncertCh := got.Uncerts["uncert/joint_pos/dreamer/variance"]
	if uncertCh == nil {
		t.Fatal("Uncerts[uncert/joint_pos/dreamer/variance] is nil")
	}
	if len(uncertCh.Data) != T*3*4 {
		t.Errorf("uncert data length = %d, want %d", len(uncertCh.Data), T*3*4)
	}

	// Verify residuals
	if len(got.Residuals) == 0 {
		t.Fatal("Residuals is empty")
	}
	res := got.Residuals["joint_pos"]
	if res == nil {
		t.Fatal("Residuals[joint_pos] is nil")
	}
	if res.Type != "sign2nddiff" {
		t.Errorf("residual type = %q, want %q", res.Type, "sign2nddiff")
	}
}
