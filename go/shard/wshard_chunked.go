// wshard_chunked.go — Chunked episode support for W-SHARD.
//
// Enables episodes spanning multiple shard files. Each chunk is a normal
// wshard file. A manifest shard (role=0x04) indexes all chunks with URIs,
// SHA-256 hashes, and timestep ranges.
package shard

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
)

// ChunkEntry describes one chunk in a manifest.
type ChunkEntry struct {
	ChunkIndex    int    `json:"chunk_index"`
	URI           string `json:"uri"`
	SHA256        string `json:"sha256"`
	TimestepRange [2]int `json:"timestep_range"`
	LengthT       int    `json:"length_T"`
	TotalChunks   int    `json:"total_chunks,omitempty"`
}

// ChunkManifest describes all chunks of a chunked episode.
type ChunkManifest struct {
	EpisodeID      string       `json:"episode_id"`
	EnvID          string       `json:"env_id,omitempty"`
	Chunks         []ChunkEntry `json:"chunks"`
	TotalTimesteps int          `json:"total_timesteps"`
}

// AddChunk appends a chunk entry and updates total timesteps.
func (m *ChunkManifest) AddChunk(e ChunkEntry) {
	m.Chunks = append(m.Chunks, e)
	m.TotalTimesteps += e.LengthT
}

// ToJSON serializes the manifest to JSON bytes.
func (m *ChunkManifest) ToJSON() ([]byte, error) {
	return json.MarshalIndent(m, "", "  ")
}

// ChunkManifestFromJSON deserializes a manifest from JSON bytes.
func ChunkManifestFromJSON(data []byte) (*ChunkManifest, error) {
	var m ChunkManifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("wshard chunked: parse manifest: %w", err)
	}
	return &m, nil
}

// ChunkedEpisodeWriter splits long episodes into multiple chunk files.
type ChunkedEpisodeWriter struct {
	basePath   string
	episodeID  string
	chunkSizeT int
	manifest   *ChunkManifest
	chunkCount int
}

// NewChunkedEpisodeWriter creates a new chunked episode writer.
func NewChunkedEpisodeWriter(basePath, episodeID string, chunkSizeT int) *ChunkedEpisodeWriter {
	return &ChunkedEpisodeWriter{
		basePath:   basePath,
		episodeID:  episodeID,
		chunkSizeT: chunkSizeT,
		manifest:   &ChunkManifest{EpisodeID: episodeID},
	}
}

func (w *ChunkedEpisodeWriter) chunkFilename(index int) string {
	return fmt.Sprintf("%s_chunk_%04d.wshard", w.episodeID, index)
}

// WriteChunk writes a single chunk episode to disk.
// Returns the path to the written chunk file.
func (w *ChunkedEpisodeWriter) WriteChunk(ep *WShardEpisode) (string, error) {
	chunkIndex := w.chunkCount
	filename := w.chunkFilename(chunkIndex)
	chunkPath := filepath.Join(w.basePath, filename)

	// Set chunk metadata
	ep.ChunkIndex = &chunkIndex

	if err := os.MkdirAll(w.basePath, 0755); err != nil {
		return "", fmt.Errorf("wshard chunked: mkdir: %w", err)
	}

	if err := CreateWShard(chunkPath, ep); err != nil {
		return "", fmt.Errorf("wshard chunked: write chunk %d: %w", chunkIndex, err)
	}

	// Compute SHA-256
	data, err := os.ReadFile(chunkPath)
	if err != nil {
		return "", fmt.Errorf("wshard chunked: read chunk for hash: %w", err)
	}
	hash := sha256.Sum256(data)
	hashStr := fmt.Sprintf("%x", hash)

	// Timestep range
	tsRange := ep.TimestepRange
	if tsRange == [2]int{0, 0} {
		globalStart := chunkIndex * w.chunkSizeT
		tsRange = [2]int{globalStart, globalStart + ep.LengthT - 1}
	}

	w.manifest.AddChunk(ChunkEntry{
		ChunkIndex:    chunkIndex,
		URI:           filename,
		SHA256:        hashStr,
		TimestepRange: tsRange,
		LengthT:       ep.LengthT,
	})
	w.chunkCount++

	return chunkPath, nil
}

// WriteEpisodeChunked splits a full episode into chunks and writes all of them.
func (w *ChunkedEpisodeWriter) WriteEpisodeChunked(ep *WShardEpisode) ([]string, error) {
	T := ep.LengthT
	totalChunks := (T + w.chunkSizeT - 1) / w.chunkSizeT
	var paths []string

	for ci := 0; ci < totalChunks; ci++ {
		startT := ci * w.chunkSizeT
		endT := startT + w.chunkSizeT
		if endT > T {
			endT = T
		}
		chunkLen := endT - startT

		chunk := &WShardEpisode{
			ID:            ep.ID,
			EnvID:         ep.EnvID,
			LengthT:       chunkLen,
			Timebase:      ep.Timebase,
			Observations:  sliceChannels(ep.Observations, startT, endT),
			Actions:       sliceChannels(ep.Actions, startT, endT),
			Rewards:       ep.Rewards[startT:endT],
			Terminations:  ep.Terminations[startT:endT],
			TimestepRange: [2]int{startT, endT - 1},
		}

		idx := ci
		total := totalChunks
		chunk.ChunkIndex = &idx
		chunk.TotalChunks = &total

		path, err := w.WriteChunk(chunk)
		if err != nil {
			return paths, err
		}
		paths = append(paths, path)
	}
	return paths, nil
}

// FinalizeManifest writes the manifest shard file (role=0x04).
// Returns the path to the manifest file.
func (w *ChunkedEpisodeWriter) FinalizeManifest() (string, error) {
	// Update total_chunks on all entries
	total := len(w.manifest.Chunks)
	for i := range w.manifest.Chunks {
		w.manifest.Chunks[i].TotalChunks = total
	}

	manifestPath := filepath.Join(w.basePath, w.episodeID+"_manifest.wshard")
	manifestJSON, err := w.manifest.ToJSON()
	if err != nil {
		return "", fmt.Errorf("wshard chunked: marshal manifest: %w", err)
	}

	mw, err := NewShardWriter(manifestPath, ShardRoleManifest)
	if err != nil {
		return "", fmt.Errorf("wshard chunked: create manifest writer: %w", err)
	}
	if err := mw.WriteEntryTyped("meta/manifest", manifestJSON, ContentTypeJSON); err != nil {
		mw.Close()
		os.Remove(manifestPath)
		return "", fmt.Errorf("wshard chunked: write manifest entry: %w", err)
	}
	if err := mw.Close(); err != nil {
		return "", fmt.Errorf("wshard chunked: close manifest: %w", err)
	}

	return manifestPath, nil
}

// ChunkedEpisodeReader reads a chunked episode from a manifest shard.
type ChunkedEpisodeReader struct {
	manifestPath string
	baseDir      string
	manifest     *ChunkManifest
}

// NewChunkedEpisodeReader creates a new chunked episode reader.
func NewChunkedEpisodeReader(manifestPath string) *ChunkedEpisodeReader {
	return &ChunkedEpisodeReader{
		manifestPath: manifestPath,
		baseDir:      filepath.Dir(manifestPath),
	}
}

// LoadManifest reads and parses the manifest shard.
func (r *ChunkedEpisodeReader) LoadManifest() (*ChunkManifest, error) {
	reader, err := OpenShard(r.manifestPath)
	if err != nil {
		return nil, fmt.Errorf("wshard chunked: open manifest: %w", err)
	}
	defer reader.Close()

	if reader.Header().Role != ShardRoleManifest {
		return nil, fmt.Errorf("wshard chunked: expected role Manifest (0x04), got 0x%02x",
			uint8(reader.Header().Role))
	}

	data, err := reader.ReadEntryByName("meta/manifest")
	if err != nil {
		return nil, fmt.Errorf("wshard chunked: read meta/manifest: %w", err)
	}

	r.manifest, err = ChunkManifestFromJSON(data)
	if err != nil {
		return nil, err
	}
	return r.manifest, nil
}

// LoadChunk loads a specific chunk by index.
func (r *ChunkedEpisodeReader) LoadChunk(index int) (*WShardEpisode, error) {
	if r.manifest == nil {
		if _, err := r.LoadManifest(); err != nil {
			return nil, err
		}
	}

	for _, chunk := range r.manifest.Chunks {
		if chunk.ChunkIndex == index {
			chunkPath := filepath.Join(r.baseDir, chunk.URI)
			return OpenWShard(chunkPath)
		}
	}
	return nil, fmt.Errorf("wshard chunked: chunk %d not found in manifest", index)
}

// IterChunks calls fn for each chunk in order.
func (r *ChunkedEpisodeReader) IterChunks(fn func(*WShardEpisode) error) error {
	if r.manifest == nil {
		if _, err := r.LoadManifest(); err != nil {
			return err
		}
	}

	sorted := make([]ChunkEntry, len(r.manifest.Chunks))
	copy(sorted, r.manifest.Chunks)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].ChunkIndex < sorted[j].ChunkIndex
	})

	for _, chunk := range sorted {
		chunkPath := filepath.Join(r.baseDir, chunk.URI)
		ep, err := OpenWShard(chunkPath)
		if err != nil {
			return fmt.Errorf("wshard chunked: load chunk %d: %w", chunk.ChunkIndex, err)
		}
		if err := fn(ep); err != nil {
			return err
		}
	}
	return nil
}

// ValidateChunkContinuity checks for gaps, duplicates, and range consistency.
func ValidateChunkContinuity(m *ChunkManifest) error {
	if len(m.Chunks) == 0 {
		return nil
	}

	sorted := make([]ChunkEntry, len(m.Chunks))
	copy(sorted, m.Chunks)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].ChunkIndex < sorted[j].ChunkIndex
	})

	// Check for gaps/duplicates in chunk_index
	for i, chunk := range sorted {
		if chunk.ChunkIndex != i {
			return fmt.Errorf("wshard chunked: chunk index sequence has gaps or duplicates at position %d (got %d)", i, chunk.ChunkIndex)
		}
	}

	// Check contiguous timestep_range
	for i := 1; i < len(sorted); i++ {
		prev := sorted[i-1]
		curr := sorted[i]
		if curr.TimestepRange[0] != prev.TimestepRange[1]+1 {
			return fmt.Errorf("wshard chunked: timestep gap between chunk %d (ends %d) and chunk %d (starts %d)",
				prev.ChunkIndex, prev.TimestepRange[1], curr.ChunkIndex, curr.TimestepRange[0])
		}
	}

	// Check total length
	var totalLen int
	for _, chunk := range sorted {
		totalLen += chunk.LengthT
	}
	if totalLen != m.TotalTimesteps {
		return fmt.Errorf("wshard chunked: sum of chunk lengths (%d) != total_timesteps (%d)",
			totalLen, m.TotalTimesteps)
	}

	return nil
}

// sliceChannels creates a shallow slice of channel data for chunking.
func sliceChannels(channels map[string]*WShardChannel, startT, endT int) map[string]*WShardChannel {
	result := make(map[string]*WShardChannel, len(channels))
	for name, ch := range channels {
		elemSize := dtypeSizeBytes(ch.DType)
		if elemSize == 0 {
			elemSize = 1
		}
		dimProduct := 1
		for _, s := range ch.Shape {
			dimProduct *= s
		}
		bytesPerT := elemSize * dimProduct
		startByte := startT * bytesPerT
		endByte := endT * bytesPerT
		if endByte > len(ch.Data) {
			endByte = len(ch.Data)
		}
		if startByte > len(ch.Data) {
			startByte = len(ch.Data)
		}

		sliced := make([]byte, endByte-startByte)
		copy(sliced, ch.Data[startByte:endByte])

		result[name] = &WShardChannel{
			Name:     ch.Name,
			DType:    ch.DType,
			Shape:    ch.Shape,
			Data:     sliced,
			Modality: ch.Modality,
		}
	}
	return result
}
