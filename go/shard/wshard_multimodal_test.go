package shard

import (
	"os"
	"path/filepath"
	"testing"
)

func TestMultiModalObservation(t *testing.T) {
	ep := &WShardEpisode{
		ID:      "mm-001",
		LengthT: 4,
	}

	rgbCh := &WShardChannel{
		Name:  "rgb",
		DType: "u8",
		Shape: []int{84, 84, 3},
		Data:  make([]byte, 4*84*84*3),
	}
	ep.AddMultiModalObservation("obs", ModalityRGB, rgbCh)

	depthCh := &WShardChannel{
		Name:  "depth",
		DType: "f32",
		Shape: []int{84, 84},
		Data:  make([]byte, 4*84*84*4),
	}
	ep.AddMultiModalObservation("obs", ModalityDepth, depthCh)

	// Verify storage
	if len(ep.Observations) != 2 {
		t.Fatalf("Observations count = %d, want 2", len(ep.Observations))
	}

	got := ep.GetMultiModalObservations("obs", ModalityRGB)
	if len(got) != 1 {
		t.Fatalf("GetMultiModalObservations(obs, rgb) = %d, want 1", len(got))
	}

	got = ep.GetMultiModalObservations("obs", ModalityDepth)
	if len(got) != 1 {
		t.Fatalf("GetMultiModalObservations(obs, depth) = %d, want 1", len(got))
	}
}

func TestLatentActions(t *testing.T) {
	ep := &WShardEpisode{
		ID:      "latent-001",
		LengthT: 10,
	}

	embeddings := &WShardChannel{
		Name:  "latent_embed",
		DType: "f32",
		Shape: []int{32},
		Data:  make([]byte, 10*32*4),
	}
	ep.SetLatentActions("genie3_v1", embeddings)

	codebook := &WShardChannel{
		Name:  "codebook_idx",
		DType: "i32",
		Shape: []int{1},
		Data:  make([]byte, 10*4),
	}
	ep.SetLatentActionCodebook("genie3_v1", codebook)

	got := ep.GetLatentActions("genie3_v1")
	if got == nil {
		t.Fatal("GetLatentActions returned nil")
	}
	if got.Name != "latent_embed" {
		t.Errorf("got name %q, want %q", got.Name, "latent_embed")
	}

	gotCB := ep.GetLatentActionCodebook("genie3_v1")
	if gotCB == nil {
		t.Fatal("GetLatentActionCodebook returned nil")
	}

	// Non-existent model
	if ep.GetLatentActions("nonexistent") != nil {
		t.Error("expected nil for nonexistent model")
	}
}

func TestLatentActionsRoundtrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "latent.wshard")

	ep := &WShardEpisode{
		ID:      "latent-rt",
		EnvID:   "test-env",
		LengthT: 4,
		Timebase: WShardTimebase{
			Type:   "ticks",
			TickHz: 30.0,
		},
		Observations: map[string]*WShardChannel{
			"state": {Name: "state", DType: "f32", Shape: []int{2}, Data: make([]byte, 4*2*4)},
		},
		Actions: map[string]*WShardChannel{
			"ctrl": {Name: "ctrl", DType: "f32", Shape: []int{1}, Data: make([]byte, 4*1*4)},
		},
		Rewards:      []float32{0, 0, 0, 1},
		Terminations: []bool{false, false, false, true},
	}

	latentData := make([]byte, 4*8*4) // 4 timesteps * 8 dims * 4 bytes
	for i := range latentData {
		latentData[i] = byte(i % 256)
	}
	ep.SetLatentActions("dreamer", &WShardChannel{
		Name: "dreamer_latent", DType: "f32", Shape: []int{8}, Data: latentData,
	})

	if err := CreateWShard(path, ep); err != nil {
		t.Fatalf("CreateWShard: %v", err)
	}

	got, err := OpenWShard(path)
	if err != nil {
		t.Fatalf("OpenWShard: %v", err)
	}

	// Latent actions are stored under omen/latent_action/dreamer
	latent := got.GetLatentActions("dreamer")
	if latent == nil {
		// Check omens directly
		if got.Omens == nil {
			t.Fatal("Omens is nil")
		}
		t.Fatalf("GetLatentActions(dreamer) returned nil, omens=%v", got.Omens)
	}
	if len(latent.Data) != len(latentData) {
		t.Errorf("latent data length = %d, want %d", len(latent.Data), len(latentData))
	}

	// Verify file cleanup
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("file not found: %v", err)
	}
}
