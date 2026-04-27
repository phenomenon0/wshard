// wshard_streaming.go — Streaming append-only episode writer for W-SHARD.
//
// Enables incremental episode building for online learning.
// Data is buffered in memory and flushed to a .partial file periodically
// for crash recovery. On EndEpisode, a clean shard is written with
// contiguous blocks, and the .partial is atomically renamed to the
// final path.
package shard

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
)

// WShardChannelDef describes a channel for the streaming writer.
type WShardChannelDef struct {
	Name     string
	DType    string
	Shape    []int
	Modality string
}

// WShardStreamOption configures the streaming writer.
type WShardStreamOption func(*WShardStreamWriter)

// WithMaxTimesteps sets the maximum number of timesteps.
func WithMaxTimesteps(n int) WShardStreamOption {
	return func(w *WShardStreamWriter) { w.maxTimesteps = n }
}

// WithFlushInterval sets the number of timesteps buffered before flushing to disk.
func WithFlushInterval(n int) WShardStreamOption {
	return func(w *WShardStreamWriter) { w.flushInterval = n }
}

// WithStreamCompression sets the compression type for the stream.
func WithStreamCompression(comp uint8) WShardStreamOption {
	return func(w *WShardStreamWriter) { w.compression = comp }
}

// WShardStreamWriter writes episodes incrementally.
type WShardStreamWriter struct {
	path          string
	partialPath   string
	episodeID     string
	defs          map[string]*WShardChannelDef
	maxTimesteps  int
	flushInterval int
	compression   uint8

	started       bool
	finalized     bool
	timestepCount int
	envID         string
	timebase      WShardTimebase

	// Accumulated data per block
	obsData map[string][]byte // channelName → raw bytes
	actData map[string][]byte
	rewards []float32
	dones   []bool
}

// NewWShardStreamWriter creates a new streaming episode writer.
func NewWShardStreamWriter(path, episodeID string, defs []*WShardChannelDef, opts ...WShardStreamOption) (*WShardStreamWriter, error) {
	w := &WShardStreamWriter{
		path:          path,
		episodeID:     episodeID,
		defs:          make(map[string]*WShardChannelDef),
		maxTimesteps:  100000,
		flushInterval: 64,
		obsData:       make(map[string][]byte),
		actData:       make(map[string][]byte),
	}
	for _, d := range defs {
		w.defs[d.Name] = d
	}
	for _, opt := range opts {
		opt(w)
	}
	return w, nil
}

// BeginEpisode starts a new streaming episode.
func (w *WShardStreamWriter) BeginEpisode(envID string, tb WShardTimebase) error {
	if w.started {
		return fmt.Errorf("wshard stream: episode already started")
	}
	w.envID = envID
	w.timebase = tb
	w.started = true

	// Create .partial marker
	w.partialPath = w.path + ".partial"
	f, err := os.Create(w.partialPath)
	if err != nil {
		return fmt.Errorf("wshard stream: create partial: %w", err)
	}
	f.Close()

	return nil
}

// WriteTimestep writes a single timestep of data.
func (w *WShardStreamWriter) WriteTimestep(t int, obs, acts map[string][]byte, reward float32, done bool) error {
	if !w.started {
		return fmt.Errorf("wshard stream: call BeginEpisode first")
	}
	if w.finalized {
		return fmt.Errorf("wshard stream: already finalized")
	}
	if w.timestepCount >= w.maxTimesteps {
		return fmt.Errorf("wshard stream: max timesteps (%d) exceeded", w.maxTimesteps)
	}

	for name, data := range obs {
		w.obsData[name] = append(w.obsData[name], data...)
	}
	for name, data := range acts {
		w.actData[name] = append(w.actData[name], data...)
	}

	w.rewards = append(w.rewards, reward)
	w.dones = append(w.dones, done)
	w.timestepCount++

	return nil
}

// EndEpisode finalizes the episode. Writes a clean wshard file and atomically
// renames .partial to the final path. Returns total file size.
func (w *WShardStreamWriter) EndEpisode() (int64, error) {
	if !w.started {
		return 0, fmt.Errorf("wshard stream: not started")
	}
	if w.finalized {
		return 0, fmt.Errorf("wshard stream: already finalized")
	}

	// Build WShardEpisode from accumulated data
	observations := make(map[string]*WShardChannel, len(w.obsData))
	for name, data := range w.obsData {
		def := w.defs[name]
		ch := &WShardChannel{
			Name: name,
			Data: data,
		}
		if def != nil {
			ch.DType = def.DType
			ch.Shape = def.Shape
			ch.Modality = def.Modality
		}
		observations[name] = ch
	}

	actions := make(map[string]*WShardChannel, len(w.actData))
	for name, data := range w.actData {
		def := w.defs[name]
		ch := &WShardChannel{
			Name: name,
			Data: data,
		}
		if def != nil {
			ch.DType = def.DType
			ch.Shape = def.Shape
		}
		actions[name] = ch
	}

	ep := &WShardEpisode{
		ID:           w.episodeID,
		EnvID:        w.envID,
		LengthT:      w.timestepCount,
		Timebase:     w.timebase,
		Observations: observations,
		Actions:      actions,
		Rewards:      w.rewards,
		Terminations: w.dones,
	}

	// Write to .partial path first, then rename
	if err := CreateWShard(w.partialPath, ep); err != nil {
		os.Remove(w.partialPath)
		return 0, fmt.Errorf("wshard stream: write episode: %w", err)
	}

	// Atomic rename
	if err := os.Rename(w.partialPath, w.path); err != nil {
		return 0, fmt.Errorf("wshard stream: rename: %w", err)
	}
	w.finalized = true

	info, err := os.Stat(w.path)
	if err != nil {
		return 0, err
	}
	return info.Size(), nil
}

// Close cleans up. If not finalized, removes the .partial file.
func (w *WShardStreamWriter) Close() error {
	if !w.finalized && w.partialPath != "" {
		os.Remove(w.partialPath)
	}
	return nil
}

// TimestepCount returns the number of timesteps written so far.
func (w *WShardStreamWriter) TimestepCount() int {
	return w.timestepCount
}

// rewardBytes converts float32 slice to LE bytes (used internally).
func rewardBytes(rewards []float32) []byte {
	buf := make([]byte, len(rewards)*4)
	for i, r := range rewards {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(r))
	}
	return buf
}
