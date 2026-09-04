// wshard_streaming.go — Streaming append-only episode writer for W-SHARD.
//
// Enables incremental episode building for online learning.
// Data is buffered in memory for the whole episode. On EndEpisode a clean
// shard is written to a .partial file with contiguous blocks, which is then
// atomically renamed to the final path. There is no periodic flush: a block's
// index entry is a single (offset, size) extent, so writing a block more than
// once would interleave it with its neighbours on disk while its extent grew
// over their bytes -- the byte count still comes out right, so T infers
// correctly, the reshape succeeds and CRC passes while the values are a
// neighbouring channel's. WithFlushInterval is therefore rejected by
// NewWShardStreamWriter rather than accepted-and-ignored; Python and
// TypeScript match.
//
// EndEpisode writes through CreateWShard, so a streamed file is sealed with
// meta/identity -- and meta/provenance when SetProvenance was called -- exactly
// as a batch-written one is.
package shard

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
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

// WithFlushInterval is not supported: each block is written exactly once, at
// EndEpisode. Passing it makes NewWShardStreamWriter fail rather than silently
// doing nothing. For crash-durable collection write chunk files instead.
//
// Deprecated: always an error. See the package comment for why.
func WithFlushInterval(n int) WShardStreamOption {
	return func(w *WShardStreamWriter) { w.flushIntervalSet = true }
}

// WithStreamCompression sets the compression type for the stream.
func WithStreamCompression(comp uint8) WShardStreamOption {
	return func(w *WShardStreamWriter) { w.compression = comp }
}

// WShardStreamWriter writes episodes incrementally.
type WShardStreamWriter struct {
	path         string
	partialPath  string
	episodeID    string
	defs         map[string]*WShardChannelDef
	maxTimesteps int
	compression  uint8

	// flushIntervalSet records a WithFlushInterval call so the constructor can
	// reject it; there is no interval to store because there is no flush.
	flushIntervalSet bool

	started       bool
	finalized     bool
	timestepCount int
	envID         string
	timebase      WShardTimebase
	provenance    *WShardProvenance

	// Accumulated data per block
	obsData map[string][]byte // channelName → raw bytes
	actData map[string][]byte
	rewards []float32
	dones   []bool
}

// NewWShardStreamWriter creates a new streaming episode writer.
func NewWShardStreamWriter(path, episodeID string, defs []*WShardChannelDef, opts ...WShardStreamOption) (*WShardStreamWriter, error) {
	w := &WShardStreamWriter{
		path:         path,
		episodeID:    episodeID,
		defs:         make(map[string]*WShardChannelDef),
		maxTimesteps: 100000,
		obsData:      make(map[string][]byte),
		actData:      make(map[string][]byte),
	}
	for _, d := range defs {
		w.defs[d.Name] = d
	}
	for _, opt := range opts {
		opt(w)
	}
	if w.flushIntervalSet {
		return nil, fmt.Errorf(
			"shard: WithFlushInterval is not supported: a block index entry is a " +
				"single (offset, size) extent, so each block is written exactly once, " +
				"at EndEpisode. For crash-durable collection write chunk files instead")
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

// SetProvenance attaches run provenance to the episode. Not a constructor
// option: EndState and LastSeq are only knowable once the episode has ended,
// so this is callable any time before EndEpisode. Python's writer takes the
// same value as an end_episode argument.
func (w *WShardStreamWriter) SetProvenance(p *WShardProvenance) {
	w.provenance = p
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
		Provenance:   w.provenance,
	}

	// Write to .partial path first, then rename
	if err := CreateWShard(w.partialPath, ep); err != nil {
		os.Remove(w.partialPath)
		return 0, fmt.Errorf("wshard stream: write episode: %w", err)
	}

	// Atomic rename. CreateWShard has already synced the .partial's contents,
	// so the rename only publishes bytes the disk has actually taken.
	if err := os.Rename(w.partialPath, w.path); err != nil {
		return 0, fmt.Errorf("wshard stream: rename: %w", err)
	}
	// Sync the directory so the rename itself survives a crash. Best effort:
	// losing it only leaves the .partial behind, which is the safe outcome,
	// and not every platform lets you sync a directory handle.
	if d, err := os.Open(filepath.Dir(w.path)); err == nil {
		_ = d.Sync()
		_ = d.Close()
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
