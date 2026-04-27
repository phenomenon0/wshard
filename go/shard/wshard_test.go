package shard

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestWShardRoundtrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.wshard")

	chunkIdx := 0
	totalChunks := 3

	original := &WShardEpisode{
		ID:      "ep-001",
		EnvID:   "cartpole-v1",
		LengthT: 4,
		Timebase: WShardTimebase{
			Type:   "ticks",
			TickHz: 60.0,
		},
		Observations: map[string]*WShardChannel{
			"rgb": {
				Name:     "rgb",
				DType:    "uint8",
				Shape:    []int{84, 84, 3},
				Data:     make([]byte, 4*84*84*3), // 4 timesteps
				Modality: "visual",
			},
			"velocity": {
				Name:  "velocity",
				DType: "float32",
				Shape: []int{3},
				Data:  make([]byte, 4*3*4), // 4 timesteps * 3 floats * 4 bytes
			},
		},
		Actions: map[string]*WShardChannel{
			"discrete": {
				Name:  "discrete",
				DType: "int32",
				Shape: []int{1},
				Data:  make([]byte, 4*1*4), // 4 timesteps * 1 int * 4 bytes
			},
		},
		Rewards:       []float32{1.0, 0.5, -0.1, 2.0},
		Terminations:  []bool{false, false, false, true},
		ChunkIndex:    &chunkIdx,
		TotalChunks:   &totalChunks,
		TimestepRange: [2]int{0, 3},
		Metadata: map[string]any{
			"seed": float64(42), // JSON numbers decode as float64
		},
	}

	// Fill observation data with non-zero pattern so we can verify
	for i := range original.Observations["rgb"].Data {
		original.Observations["rgb"].Data[i] = byte(i % 256)
	}
	for i := range original.Observations["velocity"].Data {
		original.Observations["velocity"].Data[i] = byte((i * 7) % 256)
	}
	for i := range original.Actions["discrete"].Data {
		original.Actions["discrete"].Data[i] = byte((i * 13) % 256)
	}

	// Write
	if err := CreateWShard(path, original); err != nil {
		t.Fatalf("CreateWShard: %v", err)
	}

	// Verify file exists
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat: %v", err)
	}
	if info.Size() == 0 {
		t.Fatal("written file is empty")
	}

	// Read back
	got, err := OpenWShard(path)
	if err != nil {
		t.Fatalf("OpenWShard: %v", err)
	}

	// Verify scalar fields
	if got.ID != original.ID {
		t.Errorf("ID: got %q, want %q", got.ID, original.ID)
	}
	if got.EnvID != original.EnvID {
		t.Errorf("EnvID: got %q, want %q", got.EnvID, original.EnvID)
	}
	if got.LengthT != original.LengthT {
		t.Errorf("LengthT: got %d, want %d", got.LengthT, original.LengthT)
	}
	if got.Timebase.Type != original.Timebase.Type {
		t.Errorf("Timebase.Type: got %q, want %q", got.Timebase.Type, original.Timebase.Type)
	}
	if got.Timebase.TickHz != original.Timebase.TickHz {
		t.Errorf("Timebase.TickHz: got %f, want %f", got.Timebase.TickHz, original.Timebase.TickHz)
	}
	if got.TimestepRange != original.TimestepRange {
		t.Errorf("TimestepRange: got %v, want %v", got.TimestepRange, original.TimestepRange)
	}

	// Chunk fields
	if got.ChunkIndex == nil {
		t.Fatal("ChunkIndex is nil, want non-nil")
	}
	if *got.ChunkIndex != *original.ChunkIndex {
		t.Errorf("ChunkIndex: got %d, want %d", *got.ChunkIndex, *original.ChunkIndex)
	}
	if got.TotalChunks == nil {
		t.Fatal("TotalChunks is nil, want non-nil")
	}
	if *got.TotalChunks != *original.TotalChunks {
		t.Errorf("TotalChunks: got %d, want %d", *got.TotalChunks, *original.TotalChunks)
	}

	// Metadata
	if !reflect.DeepEqual(got.Metadata, original.Metadata) {
		t.Errorf("Metadata: got %v, want %v", got.Metadata, original.Metadata)
	}

	// Rewards
	if !reflect.DeepEqual(got.Rewards, original.Rewards) {
		t.Errorf("Rewards: got %v, want %v", got.Rewards, original.Rewards)
	}

	// Terminations
	if !reflect.DeepEqual(got.Terminations, original.Terminations) {
		t.Errorf("Terminations: got %v, want %v", got.Terminations, original.Terminations)
	}

	// Observations
	if len(got.Observations) != len(original.Observations) {
		t.Fatalf("Observations count: got %d, want %d", len(got.Observations), len(original.Observations))
	}
	for name, origCh := range original.Observations {
		gotCh, ok := got.Observations[name]
		if !ok {
			t.Fatalf("missing observation %q", name)
		}
		if gotCh.Name != origCh.Name {
			t.Errorf("obs %s Name: got %q, want %q", name, gotCh.Name, origCh.Name)
		}
		if gotCh.DType != canonicalDType(origCh.DType) {
			t.Errorf("obs %s DType: got %q, want %q", name, gotCh.DType, canonicalDType(origCh.DType))
		}
		if !reflect.DeepEqual(gotCh.Shape, origCh.Shape) {
			t.Errorf("obs %s Shape: got %v, want %v", name, gotCh.Shape, origCh.Shape)
		}
		if gotCh.Modality != origCh.Modality {
			t.Errorf("obs %s Modality: got %q, want %q", name, gotCh.Modality, origCh.Modality)
		}
		if len(gotCh.Data) != len(origCh.Data) {
			t.Errorf("obs %s Data length: got %d, want %d", name, len(gotCh.Data), len(origCh.Data))
		} else {
			for i := range origCh.Data {
				if gotCh.Data[i] != origCh.Data[i] {
					t.Errorf("obs %s Data[%d]: got %d, want %d", name, i, gotCh.Data[i], origCh.Data[i])
					break
				}
			}
		}
	}

	// Actions
	if len(got.Actions) != len(original.Actions) {
		t.Fatalf("Actions count: got %d, want %d", len(got.Actions), len(original.Actions))
	}
	for name, origCh := range original.Actions {
		gotCh, ok := got.Actions[name]
		if !ok {
			t.Fatalf("missing action %q", name)
		}
		if gotCh.Name != origCh.Name {
			t.Errorf("act %s Name: got %q, want %q", name, gotCh.Name, origCh.Name)
		}
		if gotCh.DType != canonicalDType(origCh.DType) {
			t.Errorf("act %s DType: got %q, want %q", name, gotCh.DType, canonicalDType(origCh.DType))
		}
		if !reflect.DeepEqual(gotCh.Shape, origCh.Shape) {
			t.Errorf("act %s Shape: got %v, want %v", name, gotCh.Shape, origCh.Shape)
		}
		if len(gotCh.Data) != len(origCh.Data) {
			t.Errorf("act %s Data length: got %d, want %d", name, len(gotCh.Data), len(origCh.Data))
		} else {
			for i := range origCh.Data {
				if gotCh.Data[i] != origCh.Data[i] {
					t.Errorf("act %s Data[%d]: got %d, want %d", name, i, gotCh.Data[i], origCh.Data[i])
					break
				}
			}
		}
	}
}

func TestDTypeSizeBytes(t *testing.T) {
	cases := []struct {
		dtype string
		want  int
	}{
		{"bool", 1},
		{"uint8", 1},
		{"int8", 1},
		{"uint16", 2},
		{"int16", 2},
		{"float16", 2},
		{"bfloat16", 2},
		{"uint32", 4},
		{"int32", 4},
		{"float32", 4},
		{"uint64", 8},
		{"int64", 8},
		{"float64", 8},
		{"u8", 1},
		{"i8", 1},
		{"u16", 2},
		{"i16", 2},
		{"f16", 2},
		{"bf16", 2},
		{"u32", 4},
		{"i32", 4},
		{"f32", 4},
		{"u64", 8},
		{"i64", 8},
		{"f64", 8},
	}

	for _, tc := range cases {
		got := dtypeSizeBytes(tc.dtype)
		if got != tc.want {
			t.Errorf("dtypeSizeBytes(%q) = %d, want %d", tc.dtype, got, tc.want)
		}
	}

	// Unknown dtype should return 0
	if got := dtypeSizeBytes("complex128"); got != 0 {
		t.Errorf("dtypeSizeBytes(\"complex128\") = %d, want 0", got)
	}
}

func TestCreateWShardWritesCanonicalMetadata(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "canonical.wshard")

	ep := &WShardEpisode{
		ID:      "canonical-ep",
		EnvID:   "CanonicalEnv-v0",
		LengthT: 2,
		Timebase: WShardTimebase{
			Type:   "ticks",
			TickHz: 50,
		},
		Observations: map[string]*WShardChannel{
			"state/pos": {
				Name:  "state/pos",
				DType: "float32",
				Shape: []int{2},
				Data:  make([]byte, 2*2*4),
			},
		},
		Actions: map[string]*WShardChannel{
			"ctrl": {
				Name:  "ctrl",
				DType: "int32",
				Shape: []int{1},
				Data:  make([]byte, 2*4),
			},
		},
		Rewards:      []float32{0, 1},
		Terminations: []bool{false, true},
	}

	if err := CreateWShard(path, ep); err != nil {
		t.Fatalf("CreateWShard: %v", err)
	}

	r, err := OpenShard(path)
	if err != nil {
		t.Fatalf("OpenShard: %v", err)
	}
	defer r.Close()

	metaEpisodeRaw, err := r.ReadEntryByName("meta/episode")
	if err != nil {
		t.Fatalf("ReadEntryByName(meta/episode): %v", err)
	}
	var metaEpisode map[string]any
	if err := json.Unmarshal(metaEpisodeRaw, &metaEpisode); err != nil {
		t.Fatalf("json.Unmarshal(meta/episode): %v", err)
	}
	if _, ok := metaEpisode["episode_id"]; !ok {
		t.Fatal("meta/episode missing canonical episode_id")
	}
	if _, ok := metaEpisode["length_T"]; !ok {
		t.Fatal("meta/episode missing canonical length_T")
	}
	if _, ok := metaEpisode["id"]; ok {
		t.Fatal("meta/episode unexpectedly contains legacy id key")
	}
	if _, ok := metaEpisode["length_t"]; ok {
		t.Fatal("meta/episode unexpectedly contains legacy length_t key")
	}

	metaChannelsRaw, err := r.ReadEntryByName("meta/channels")
	if err != nil {
		t.Fatalf("ReadEntryByName(meta/channels): %v", err)
	}
	var metaChannels wshardChannelsMeta
	if err := json.Unmarshal(metaChannelsRaw, &metaChannels); err != nil {
		t.Fatalf("json.Unmarshal(meta/channels): %v", err)
	}
	if len(metaChannels.Channels) != 2 {
		t.Fatalf("meta/channels entries = %d, want 2", len(metaChannels.Channels))
	}

	gotBlocks := map[string]string{}
	gotDTypes := map[string]string{}
	for _, ch := range metaChannels.Channels {
		gotBlocks[ch.ID] = ch.SignalBlock
		gotDTypes[ch.ID] = ch.DType
	}
	if gotBlocks["state/pos"] != "signal/state/pos" {
		t.Fatalf("state/pos signal_block = %q, want %q", gotBlocks["state/pos"], "signal/state/pos")
	}
	if gotBlocks["ctrl"] != "action/ctrl" {
		t.Fatalf("ctrl signal_block = %q, want %q", gotBlocks["ctrl"], "action/ctrl")
	}
	if gotDTypes["state/pos"] != "f32" {
		t.Fatalf("state/pos dtype = %q, want %q", gotDTypes["state/pos"], "f32")
	}
	if gotDTypes["ctrl"] != "i32" {
		t.Fatalf("ctrl dtype = %q, want %q", gotDTypes["ctrl"], "i32")
	}
}

func TestOpenWShardReadsCanonicalGoldenFixtures(t *testing.T) {
	cases := []struct {
		name         string
		path         string
		wantID       string
		wantLength   int
		wantObsCount int
		wantAction   bool
	}{
		{
			name:         "simple",
			path:         filepath.Join("..", "..", "wshard", "golden", "simple_episode.wshard"),
			wantID:       "golden_simple",
			wantLength:   10,
			wantObsCount: 1,
			wantAction:   true,
		},
		{
			name:         "compressed",
			path:         filepath.Join("..", "..", "wshard", "golden", "per_block_compressed.wshard"),
			wantID:       "golden_compressed",
			wantLength:   100,
			wantObsCount: 1,
			wantAction:   false,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := os.Stat(tc.path); err != nil {
				t.Skipf("fixture missing: %v", err)
			}

			ep, err := OpenWShard(tc.path)
			if err != nil {
				t.Fatalf("OpenWShard(%s): %v", tc.path, err)
			}
			if ep.ID != tc.wantID {
				t.Fatalf("ID = %q, want %q", ep.ID, tc.wantID)
			}
			if ep.LengthT != tc.wantLength {
				t.Fatalf("LengthT = %d, want %d", ep.LengthT, tc.wantLength)
			}
			if len(ep.Observations) != tc.wantObsCount {
				t.Fatalf("Observations = %d, want %d", len(ep.Observations), tc.wantObsCount)
			}
			if tc.wantAction && len(ep.Actions) == 0 {
				t.Fatal("expected at least one action channel from canonical fixture")
			}
		})
	}
}

func TestOpenWShardGoldenOmenUncert(t *testing.T) {
	path := filepath.Join("..", "..", "wshard", "golden", "omen_uncert.wshard")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("fixture missing: %v", err)
	}

	ep, err := OpenWShard(path)
	if err != nil {
		t.Fatalf("OpenWShard: %v", err)
	}
	if ep.ID != "golden_omen_uncert" {
		t.Fatalf("ID = %q, want %q", ep.ID, "golden_omen_uncert")
	}
	if ep.LengthT != 10 {
		t.Fatalf("LengthT = %d, want 10", ep.LengthT)
	}

	// Verify omen block
	if ep.Omens["joint_pos"] == nil {
		t.Fatal("Omens[joint_pos] is nil")
	}
	dreamerOmen := ep.Omens["joint_pos"]["dreamer"]
	if dreamerOmen == nil {
		t.Fatal("Omens[joint_pos][dreamer] is nil")
	}
	if len(dreamerOmen.Data) != 10*3*4 { // 10 timesteps * 3 dims * 4 bytes
		t.Errorf("omen data length = %d, want %d", len(dreamerOmen.Data), 10*3*4)
	}

	// Verify uncert block
	uncertKey := "uncert/joint_pos/dreamer/variance"
	if ep.Uncerts[uncertKey] == nil {
		t.Fatalf("Uncerts[%s] is nil", uncertKey)
	}
	if len(ep.Uncerts[uncertKey].Data) != 10*3*4 {
		t.Errorf("uncert data length = %d, want %d", len(ep.Uncerts[uncertKey].Data), 10*3*4)
	}

	// Verify residual block
	if ep.Residuals["joint_pos"] == nil {
		t.Fatal("Residuals[joint_pos] is nil")
	}
	if ep.Residuals["joint_pos"].Type != "sign2nddiff" {
		t.Errorf("residual type = %q, want %q", ep.Residuals["joint_pos"].Type, "sign2nddiff")
	}
}

func TestOpenWShardGoldenMultiModal(t *testing.T) {
	path := filepath.Join("..", "..", "wshard", "golden", "multimodal.wshard")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("fixture missing: %v", err)
	}

	ep, err := OpenWShard(path)
	if err != nil {
		t.Fatalf("OpenWShard: %v", err)
	}
	if ep.ID != "golden_multimodal" {
		t.Fatalf("ID = %q, want %q", ep.ID, "golden_multimodal")
	}
	if ep.LengthT != 5 {
		t.Fatalf("LengthT = %d, want 5", ep.LengthT)
	}
	// Should have 3 observation channels: obs/rgb, obs/depth, obs/proprioception
	if len(ep.Observations) != 3 {
		t.Fatalf("Observations = %d, want 3", len(ep.Observations))
	}
	// Verify modalities are preserved
	rgbCh := ep.Observations["obs/rgb"]
	if rgbCh == nil {
		t.Fatal("obs/rgb channel missing")
	}
	if rgbCh.Modality != "rgb" {
		t.Errorf("obs/rgb modality = %q, want %q", rgbCh.Modality, "rgb")
	}
}

func TestOpenWShardGoldenLatentAction(t *testing.T) {
	path := filepath.Join("..", "..", "wshard", "golden", "latent_action.wshard")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("fixture missing: %v", err)
	}

	ep, err := OpenWShard(path)
	if err != nil {
		t.Fatalf("OpenWShard: %v", err)
	}
	if ep.ID != "golden_latent_action" {
		t.Fatalf("ID = %q, want %q", ep.ID, "golden_latent_action")
	}
	if ep.LengthT != 8 {
		t.Fatalf("LengthT = %d, want 8", ep.LengthT)
	}

	// Verify latent action via GetLatentActions
	latent := ep.GetLatentActions("genie3")
	if latent == nil {
		t.Fatal("GetLatentActions(genie3) returned nil")
	}
	if len(latent.Data) != 8*16*4 { // 8 timesteps * 16 dims * 4 bytes
		t.Errorf("latent data length = %d, want %d", len(latent.Data), 8*16*4)
	}

	// Verify codebook
	codebook := ep.GetLatentActionCodebook("genie3")
	if codebook == nil {
		t.Fatal("GetLatentActionCodebook(genie3) returned nil")
	}
	if len(codebook.Data) != 8*4 { // 8 timesteps * 4 bytes (i32)
		t.Errorf("codebook data length = %d, want %d", len(codebook.Data), 8*4)
	}
}
