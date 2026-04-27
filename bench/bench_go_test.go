// bench_go_test.go — Go benchmarks for WShard write/read across compression types.
//
// Workload: T=1000 step synthetic episode
//   signal/joint_pos  [1000, 7]         float32  ~28 KB
//   signal/rgb        [1000, 84, 84, 3] uint8    ~21 MB
//   action/ctrl       [1000, 7]         float32  ~28 KB
//   reward            [1000]            float32  ~4 KB
//   done              [1000]            bool     ~1 KB
//   Total raw payload: ~21 MB
//
// Run:
//   cd bench && go test -bench=. -benchmem
package bench

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"github.com/phenomenon0/wshard/go/shard"
)

// ── workload constants ─────────────────────────────────────────────────────────

const (
	T       = 1000
	jointDim = 7
	rgbH    = 84
	rgbW    = 84
	rgbC    = 3
	ctrlDim = 7
)

// rawBytes is the uncompressed payload size, passed to b.SetBytes so Go's
// testing framework can compute MB/s throughput automatically.
var rawBytes = int64(
	T*jointDim*4 + // joint_pos f32[7]
		T*rgbH*rgbW*rgbC + // rgb u8[84,84,3]
		T*ctrlDim*4 + // ctrl f32[7]
		T*4 + // reward f32
		T*1, // done bool
)

// ── synthetic data builders ────────────────────────────────────────────────────

func makeJointPos() []byte {
	rng := rand.New(rand.NewSource(42))
	buf := make([]byte, T*jointDim*4)
	for i := 0; i < T*jointDim; i++ {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(float32(rng.NormFloat64())))
	}
	return buf
}

// makeRGB returns a structured-but-realistic RGB workload: vertical gradient
// (0–255 top-to-bottom) plus ±2 low-amplitude noise per pixel.
// This mirrors the Python bench (same gradient recipe, same noise range) so
// Python and Go numbers compare apples-to-apples.
// Noise amplitude: ±2 LSB — expected zstd ratio ~2×.
func makeRGB() []byte {
	rng := rand.New(rand.NewSource(43))
	buf := make([]byte, T*rgbH*rgbW*rgbC)
	for t := 0; t < T; t++ {
		for row := 0; row < rgbH; row++ {
			// Vertical gradient: 0 at top, 255 at bottom
			base := float64(row) / float64(rgbH-1) * 255.0
			for col := 0; col < rgbW; col++ {
				for ch := 0; ch < rgbC; ch++ {
					noise := float64(rng.Intn(4)) - 2.0 // [-2, +1]
					v := base + noise
					if v < 0 {
						v = 0
					} else if v > 255 {
						v = 255
					}
					idx := t*rgbH*rgbW*rgbC + row*rgbW*rgbC + col*rgbC + ch
					buf[idx] = byte(v)
				}
			}
		}
	}
	return buf
}

func makeCtrl() []byte {
	rng := rand.New(rand.NewSource(44))
	buf := make([]byte, T*ctrlDim*4)
	for i := 0; i < T*ctrlDim; i++ {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(float32(rng.NormFloat64())))
	}
	return buf
}

func makeRewards() []float32 {
	rng := rand.New(rand.NewSource(45))
	r := make([]float32, T)
	for i := range r {
		r[i] = float32(rng.NormFloat64())
	}
	return r
}

func makeTerminations() []bool {
	rng := rand.New(rand.NewSource(46))
	t := make([]bool, T)
	for i := range t {
		t[i] = rng.Intn(100) == 0
	}
	return t
}

// ── episode builder ────────────────────────────────────────────────────────────

func makeEpisode() *shard.WShardEpisode {
	return &shard.WShardEpisode{
		ID:      "bench_ep",
		EnvID:   "BenchEnv-v0",
		LengthT: T,
		Observations: map[string]*shard.WShardChannel{
			"joint_pos": {
				Name:  "joint_pos",
				DType: "f32",
				Shape: []int{jointDim},
				Data:  makeJointPos(),
			},
			"rgb": {
				Name:  "rgb",
				DType: "u8",
				Shape: []int{rgbH, rgbW, rgbC},
				Data:  makeRGB(),
			},
		},
		Actions: map[string]*shard.WShardChannel{
			"ctrl": {
				Name:  "ctrl",
				DType: "f32",
				Shape: []int{ctrlDim},
				Data:  makeCtrl(),
			},
		},
		Rewards:      makeRewards(),
		Terminations: makeTerminations(),
	}
}

// ── local write helper with configurable compression ─────────────────────────
//
// shard.CreateWShard always writes with no compression. To benchmark the
// compressed paths, we replicate the write sequence using the public
// ShardWriter API: NewShardWriter + SetCompression + WriteEntryCompressed.
//
// JSON meta blocks are always written uncompressed (tiny, < minCompressSize);
// tensor data blocks are written via WriteEntryCompressed so the compression
// default is used.

func writeEpisodeComp(path string, ep *shard.WShardEpisode, comp uint8) error {
	w, err := shard.NewShardWriter(path, shard.ShardRoleWShard)
	if err != nil {
		return err
	}
	w.SetCompression(comp)

	cleanup := func() {
		_ = w.Close()
		_ = os.Remove(path)
	}

	// meta/wshard (JSON, always uncompressed — tiny)
	metaJSON, _ := json.Marshal(map[string]any{
		"format":  "W-SHARD",
		"version": "0.1",
	})
	if err := w.WriteEntryTyped("meta/wshard", metaJSON, shard.ContentTypeJSON); err != nil {
		cleanup()
		return err
	}

	// meta/episode
	lengthT := ep.LengthT
	epMeta := map[string]any{
		"episode_id": ep.ID,
		"env_id":     ep.EnvID,
		"length_T":   lengthT,
	}
	epJSON, _ := json.Marshal(epMeta)
	if err := w.WriteEntryTyped("meta/episode", epJSON, shard.ContentTypeJSON); err != nil {
		cleanup()
		return err
	}

	// meta/channels (JSON)
	chanMeta := map[string]any{
		"channels": []map[string]any{
			{"id": "joint_pos", "dtype": "f32", "shape": []int{jointDim}, "signal_block": "signal/joint_pos"},
			{"id": "rgb", "dtype": "u8", "shape": []int{rgbH, rgbW, rgbC}, "signal_block": "signal/rgb"},
			{"id": "ctrl", "dtype": "f32", "shape": []int{ctrlDim}, "signal_block": "action/ctrl"},
		},
	}
	chanJSON, _ := json.Marshal(chanMeta)
	if err := w.WriteEntryTyped("meta/channels", chanJSON, shard.ContentTypeJSON); err != nil {
		cleanup()
		return err
	}

	// signal/joint_pos
	if err := w.WriteEntryCompressed("signal/joint_pos", ep.Observations["joint_pos"].Data); err != nil {
		cleanup()
		return err
	}

	// signal/rgb
	if err := w.WriteEntryCompressed("signal/rgb", ep.Observations["rgb"].Data); err != nil {
		cleanup()
		return err
	}

	// action/ctrl
	if err := w.WriteEntryCompressed("action/ctrl", ep.Actions["ctrl"].Data); err != nil {
		cleanup()
		return err
	}

	// reward ([]float32 → []byte LE)
	rewardBuf := make([]byte, len(ep.Rewards)*4)
	for i, r := range ep.Rewards {
		binary.LittleEndian.PutUint32(rewardBuf[i*4:], math.Float32bits(r))
	}
	if err := w.WriteEntryCompressed("reward", rewardBuf); err != nil {
		cleanup()
		return err
	}

	// done ([]bool → []uint8)
	doneBuf := make([]byte, len(ep.Terminations))
	for i, t := range ep.Terminations {
		if t {
			doneBuf[i] = 1
		}
	}
	if err := w.WriteEntryCompressed("done", doneBuf); err != nil {
		cleanup()
		return err
	}

	return w.Close()
}

// ── benchmarks: write ─────────────────────────────────────────────────────────

func BenchmarkWriteNone(b *testing.B) {
	ep := makeEpisode()
	b.SetBytes(rawBytes)
	b.ResetTimer()
	dir := b.TempDir()
	for i := 0; i < b.N; i++ {
		p := filepath.Join(dir, "ep_none.wshard")
		if err := shard.CreateWShard(p, ep); err != nil {
			b.Fatal(err)
		}
		os.Remove(p)
	}
}

func BenchmarkWriteZstd(b *testing.B) {
	ep := makeEpisode()
	b.SetBytes(rawBytes)
	b.ResetTimer()
	dir := b.TempDir()
	for i := 0; i < b.N; i++ {
		p := filepath.Join(dir, "ep_zstd.wshard")
		if err := writeEpisodeComp(p, ep, shard.CompressZstd); err != nil {
			b.Fatal(err)
		}
		os.Remove(p)
	}
}

func BenchmarkWriteLz4(b *testing.B) {
	ep := makeEpisode()
	b.SetBytes(rawBytes)
	b.ResetTimer()
	dir := b.TempDir()
	for i := 0; i < b.N; i++ {
		p := filepath.Join(dir, "ep_lz4.wshard")
		if err := writeEpisodeComp(p, ep, shard.CompressLZ4); err != nil {
			b.Fatal(err)
		}
		os.Remove(p)
	}
}

// ── benchmarks: read ──────────────────────────────────────────────────────────

func writeTempEpisode(tb testing.TB, comp uint8) string {
	tb.Helper()
	ep := makeEpisode()
	dir := tb.TempDir()
	p := filepath.Join(dir, "ep.wshard")
	var err error
	if comp == shard.CompressNone {
		err = shard.CreateWShard(p, ep)
	} else {
		err = writeEpisodeComp(p, ep, comp)
	}
	if err != nil {
		tb.Fatalf("setup write (comp=%d): %v", comp, err)
	}
	return p
}

func BenchmarkReadNone(b *testing.B) {
	p := writeTempEpisode(b, shard.CompressNone)
	b.SetBytes(rawBytes)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := shard.OpenWShard(p); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkReadZstd(b *testing.B) {
	p := writeTempEpisode(b, shard.CompressZstd)
	b.SetBytes(rawBytes)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := shard.OpenWShard(p); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkReadLz4(b *testing.B) {
	p := writeTempEpisode(b, shard.CompressLZ4)
	b.SetBytes(rawBytes)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := shard.OpenWShard(p); err != nil {
			b.Fatal(err)
		}
	}
}

// ── benchmark: open + scan index only (no payload decode) ────────────────────
//
// This benchmark measures only the cost of reading the shard header and index.
// b.SetBytes is intentionally omitted: reporting MB/s against the full
// payload size would be misleading since the index is ~1 KB, not ~21 MB.

func BenchmarkOpenAndIndex(b *testing.B) {
	p := writeTempEpisode(b, shard.CompressNone)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r, err := shard.OpenShard(p)
		if err != nil {
			b.Fatal(err)
		}
		_ = r.Header()
		r.Close()
	}
}

// ── benchmark: partial-block read ─────────────────────────────────────────────
//
// BenchmarkPartialReadCtrl writes ONE large wshard (~50 MB: rgb 21 MB + 4 depth
// fillers ~7 MB each) and measures the cost of fetching ONLY action/ctrl (28 KB).
//
// This is WShard's core differentiator: index-guided random access reads only
// the one requested block from disk, regardless of total file size.

func makeDepthFiller(seed int64) []byte {
	rng := rand.New(rand.NewSource(seed))
	buf := make([]byte, T*rgbH*rgbW*1)
	// Structured like depth: gradient + low noise (±2 LSB)
	for t := 0; t < T; t++ {
		for row := 0; row < rgbH; row++ {
			base := float64(row) / float64(rgbH-1) * 200.0
			for col := 0; col < rgbW; col++ {
				noise := float64(rng.Intn(4)) - 2.0
				v := base + noise
				if v < 0 {
					v = 0
				} else if v > 255 {
					v = 255
				}
				idx := t*rgbH*rgbW + row*rgbW + col
				buf[idx] = byte(v)
			}
		}
	}
	return buf
}

func writeLargeEpisode(tb testing.TB, comp uint8) string {
	tb.Helper()
	dir := tb.TempDir()
	p := filepath.Join(dir, "large.wshard")

	ep := makeEpisode()
	// Add 4 depth filler channels to get ~50 MB total
	// We inject them by building a custom episode directly via the writer API.
	w, err := shard.NewShardWriter(p, shard.ShardRoleWShard)
	if err != nil {
		tb.Fatalf("open writer: %v", err)
	}
	w.SetCompression(comp)

	cleanup := func() {
		_ = w.Close()
		_ = os.Remove(p)
	}

	metaJSON, _ := json.Marshal(map[string]any{"format": "W-SHARD", "version": "0.1"})
	if err := w.WriteEntryTyped("meta/wshard", metaJSON, shard.ContentTypeJSON); err != nil {
		cleanup()
		tb.Fatal(err)
	}
	epMeta, _ := json.Marshal(map[string]any{"episode_id": "bench_partial", "env_id": "BenchEnv-v0", "length_T": T})
	if err := w.WriteEntryTyped("meta/episode", epMeta, shard.ContentTypeJSON); err != nil {
		cleanup()
		tb.Fatal(err)
	}
	chanMeta, _ := json.Marshal(map[string]any{"channels": []map[string]any{
		{"id": "rgb", "dtype": "u8", "shape": []int{rgbH, rgbW, rgbC}, "signal_block": "signal/rgb"},
		{"id": "ctrl", "dtype": "f32", "shape": []int{ctrlDim}, "signal_block": "action/ctrl"},
	}})
	if err := w.WriteEntryTyped("meta/channels", chanMeta, shard.ContentTypeJSON); err != nil {
		cleanup()
		tb.Fatal(err)
	}

	if err := w.WriteEntryCompressed("signal/rgb", ep.Observations["rgb"].Data); err != nil {
		cleanup()
		tb.Fatal(err)
	}

	// 4 depth filler blocks
	for i, tag := range []string{"a", "b", "c", "d"} {
		depth := makeDepthFiller(int64(100 + i))
		if err := w.WriteEntryCompressed("signal/depth_"+tag, depth); err != nil {
			cleanup()
			tb.Fatal(err)
		}
	}

	if err := w.WriteEntryCompressed("action/ctrl", ep.Actions["ctrl"].Data); err != nil {
		cleanup()
		tb.Fatal(err)
	}
	if err := w.WriteEntryCompressed("signal/joint_pos", ep.Observations["joint_pos"].Data); err != nil {
		cleanup()
		tb.Fatal(err)
	}

	rewardBuf := make([]byte, T*4)
	for i, r := range ep.Rewards {
		binary.LittleEndian.PutUint32(rewardBuf[i*4:], math.Float32bits(r))
	}
	if err := w.WriteEntryCompressed("reward", rewardBuf); err != nil {
		cleanup()
		tb.Fatal(err)
	}

	doneBuf := make([]byte, T)
	for i, t := range ep.Terminations {
		if t {
			doneBuf[i] = 1
		}
	}
	if err := w.WriteEntryCompressed("done", doneBuf); err != nil {
		cleanup()
		tb.Fatal(err)
	}

	if err := w.Close(); err != nil {
		tb.Fatalf("close writer: %v", err)
	}
	return p
}

// BenchmarkPartialReadCtrl measures fetching only action/ctrl (28 KB) from a
// ~50 MB wshard file. b.SetBytes is intentionally omitted because we are
// measuring the cost of fetching 28 KB, not reading all 50 MB.
func BenchmarkPartialReadCtrl(b *testing.B) {
	p := writeLargeEpisode(b, shard.CompressZstd)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r, err := shard.OpenShard(p)
		if err != nil {
			b.Fatal(err)
		}
		_, err = r.ReadEntryByName("action/ctrl")
		if err != nil {
			b.Fatal(err)
		}
		r.Close()
	}
}
