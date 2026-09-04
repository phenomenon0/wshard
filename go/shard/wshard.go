// wshard.go - Typed API for WShard (world-model episode) shard files.
//
// A WShard stores a single RL episode: observations, actions, rewards,
// terminations, plus metadata (env, timebase, channel definitions).
//
// Layout inside the shard (all entries are path-separated):
//
//	meta/wshard       - JSON: format/version metadata
//	meta/episode      - JSON: canonical episode metadata (episode_id, length_T, ...)
//	meta/channels     - JSON: {"channels":[...]} canonical channel definitions
//	signal/<name>     - raw bytes for observation channel <name>
//	action/<name>     - raw bytes for action channel <name>
//	reward            - []float32 LE
//	done              - []uint8 (0/1 booleans)
package shard

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
)

// ============================================================
// Public types
// ============================================================

// WShardEpisode is the top-level structure for a single episode.
type WShardEpisode struct {
	ID            string
	EnvID         string
	LengthT       int
	Timebase      WShardTimebase
	Observations  map[string]*WShardChannel
	Actions       map[string]*WShardChannel
	Rewards       []float32
	Terminations  []bool
	Omens         map[string]map[string]*WShardChannel // [channelID][modelID] → data
	Uncerts       map[string]*WShardChannel            // full block name → data
	Residuals     map[string]*WShardResidual           // channelID → packed bits
	ChunkIndex    *int                                 // nil if not chunked
	TotalChunks   *int
	TimestepRange [2]int
	Metadata      map[string]any
	Provenance    *WShardProvenance // nil means the block is not written
}

// WShardResidual holds a compact residual encoding for a channel.
type WShardResidual struct {
	ChannelID string
	Type      string // "sign2nddiff"
	Data      []byte
}

// WShardChannel holds a single named signal channel.
type WShardChannel struct {
	Name     string
	DType    string
	Shape    []int
	Data     []byte
	Modality string // empty if N/A
}

// WShardTimebase describes the episode's time axis.
type WShardTimebase struct {
	Type   string  // "ticks" or "timestamps_ns"
	TickHz float64 // meaningful only when Type == "ticks"
}

// ============================================================
// Block naming helpers
// ============================================================

// OmenBlockName returns the block name for an omen channel: "omen/{channelID}/{modelID}".
func OmenBlockName(channelID, modelID string) string {
	return JoinPath("omen", JoinPath(channelID, modelID))
}

// UncertBlockName returns the block name for an uncertainty channel:
// "uncert/{channelID}/{modelID}/{uncertType}".
func UncertBlockName(channelID, modelID, uncertType string) string {
	return JoinPath("uncert", JoinPath(channelID, JoinPath(modelID, uncertType)))
}

// ResidualSign2ndDiffBlockName returns the block name for sign2nddiff residuals:
// "residual/{channelID}/sign2nddiff".
func ResidualSign2ndDiffBlockName(channelID string) string {
	return JoinPath("residual", JoinPath(channelID, "sign2nddiff"))
}

// LatentActionBlockName returns the block name for latent actions:
// "omen/latent_action/{modelID}".
func LatentActionBlockName(modelID string) string {
	return JoinPath("omen", JoinPath("latent_action", modelID))
}

// LatentCodebookBlockName returns the block name for latent action codebooks:
// "omen/latent_action_codebook/{modelID}".
func LatentCodebookBlockName(modelID string) string {
	return JoinPath("omen", JoinPath("latent_action_codebook", modelID))
}

// MultiModalSignalBlockName returns the block name for a multi-modal signal:
// "signal/{group}/{modality}".
func MultiModalSignalBlockName(group string, mod Modality) string {
	return JoinPath("signal", JoinPath(group, string(mod)))
}

// canonicalJSON marshals a metadata block the way glyph's canon_json does:
// keys sorted, no spaces. Every block meta/identity commits to has to be
// byte-identical across writers and languages, or two implementations compute
// two identities for one episode.
//
// json.Marshal already does this for a map, but emits a struct's fields in
// declaration order, so the structs below go out through a map. The round trip
// also renders integers exactly, up to the 2^53 the format caps them at anyway
// (see FORMAT.md on meta/provenance).
//
// ponytail: Go and glyph diverge on extreme floats (1e20, 1e-7) and no
// metadata field carries one. TestIdentityParity catches it if that changes.
func canonicalJSON(v any) ([]byte, error) {
	raw, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	var doc map[string]any
	if err := json.Unmarshal(raw, &doc); err != nil {
		return nil, err
	}
	return json.Marshal(doc)
}

// WShardMetaVersion is meta/wshard's "version". It names the metadata schema,
// not the binary container revision in the header, and must match the Python
// writer's WSHARD_VERSION: the block is inside the seal.
const WShardMetaVersion = "0.1"

// ============================================================
// Internal JSON structs for meta serialization
// ============================================================

type wshardMetaBlock struct {
	Format        string           `json:"format,omitempty"`
	ResidualEdges string           `json:"residual_edges,omitempty"`
	Timebase      *wshardTimebaseJ `json:"timebase,omitempty"`
}

type wshardTimebaseJ struct {
	Type   string  `json:"type"`
	TickHz float64 `json:"tick_hz,omitempty"`
	DtNS   int64   `json:"dt_ns,omitempty"`
}

type wshardEpisodeMeta struct {
	EpisodeID     string           `json:"episode_id,omitempty"`
	LegacyID      string           `json:"id,omitempty"`
	EnvID         string           `json:"env_id,omitempty"`
	LengthT       *int             `json:"length_T,omitempty"`
	LegacyLengthT *int             `json:"length_t,omitempty"`
	Timebase      *wshardTimebaseJ `json:"timebase,omitempty"`
	ChunkIndex    *int             `json:"chunk_index,omitempty"`
	TotalChunks   *int             `json:"total_chunks,omitempty"`
	// Pointer, not [2]int: `omitempty` is a no-op on a Go array, so a value type
	// would always emit `"timestep_range":[0,0]` and diverge from writers that omit
	// the key. [0,0] means "unset" here, as it already does in wshard_chunked.go.
	TimestepRange *[2]int        `json:"timestep_range,omitempty"`
	Metadata      map[string]any `json:"metadata,omitempty"`
}

type wshardChannelDef struct {
	ID          string `json:"id,omitempty"`
	DType       string `json:"dtype"`
	Shape       []int  `json:"shape"`
	Modality    string `json:"modality,omitempty"`
	SignalBlock string `json:"signal_block,omitempty"`
}

type wshardChannelsMeta struct {
	Channels []wshardChannelDef `json:"channels"`
}

func (m *wshardEpisodeMeta) resolvedID() string {
	if m.EpisodeID != "" {
		return m.EpisodeID
	}
	return m.LegacyID
}

func (m *wshardEpisodeMeta) resolvedLengthT() int {
	if m.LengthT != nil {
		return *m.LengthT
	}
	if m.LegacyLengthT != nil {
		return *m.LegacyLengthT
	}
	return 0
}

func (tb *wshardTimebaseJ) toPublic() WShardTimebase {
	if tb == nil {
		return WShardTimebase{}
	}
	out := WShardTimebase{Type: tb.Type}
	if tb.Type == "ticks" {
		switch {
		case tb.TickHz > 0:
			out.TickHz = tb.TickHz
		case tb.DtNS > 0:
			out.TickHz = 1e9 / float64(tb.DtNS)
		}
	}
	return out
}

// ============================================================
// Writer
// ============================================================

// CreateWShard writes a WShardEpisode to a new shard file at path.
func CreateWShard(path string, ep *WShardEpisode) error {
	w, err := NewShardWriter(path, ShardRoleWShard)
	if err != nil {
		return fmt.Errorf("wshard: create writer: %w", err)
	}

	cleanup := func() {
		_ = w.Close()
		_ = os.Remove(path)
	}

	// Every entry's uncompressed sha256, committed by meta/identity (written last).
	leaves := map[string]string{}
	put := func(name string, data []byte, contentType uint16) error {
		leaves[name] = leafHex(data)
		return w.WriteEntryTyped(name, data, contentType)
	}

	// meta/wshard. Keys agreed with the Python writer: this block is in the
	// leaf map, so any key one writer emits and the other does not gives two
	// identities for one episode. "timebase" is gone from here because
	// meta/episode always carries it and both writers read it from there;
	// resolvedEpisodeTimebase keeps the fallback for files written before this.
	//
	metaJSON, err := canonicalJSON(map[string]any{
		"endianness":     "little",
		"format":         "W-SHARD",
		"residual_edges": "pad",
		"version":        WShardMetaVersion,
	})
	if err != nil {
		cleanup()
		return fmt.Errorf("wshard: marshal meta/wshard: %w", err)
	}
	if err := put("meta/wshard", metaJSON, ContentTypeJSON); err != nil {
		cleanup()
		return fmt.Errorf("wshard: write meta/wshard: %w", err)
	}

	// meta/episode
	lengthT := ep.LengthT
	var tsRange *[2]int
	if ep.TimestepRange != [2]int{0, 0} {
		tsRange = &ep.TimestepRange
	}
	epJSON, err := canonicalJSON(wshardEpisodeMeta{
		EpisodeID:     ep.ID,
		EnvID:         ep.EnvID,
		LengthT:       &lengthT,
		Timebase:      episodeTimebaseMeta(ep.Timebase),
		ChunkIndex:    ep.ChunkIndex,
		TotalChunks:   ep.TotalChunks,
		TimestepRange: tsRange,
		Metadata:      ep.Metadata,
	})
	if err != nil {
		cleanup()
		return fmt.Errorf("wshard: marshal meta/episode: %w", err)
	}
	if err := put("meta/episode", epJSON, ContentTypeJSON); err != nil {
		cleanup()
		return fmt.Errorf("wshard: write meta/episode: %w", err)
	}

	// meta/channels
	channelsMeta := wshardChannelsMeta{
		Channels: make([]wshardChannelDef, 0, len(ep.Observations)+len(ep.Actions)),
	}
	for _, name := range sortedChannelNames(ep.Observations) {
		ch := ep.Observations[name]
		channelsMeta.Channels = append(channelsMeta.Channels, wshardChannelDef{
			ID:          name,
			DType:       canonicalDType(ch.DType),
			Shape:       ch.Shape,
			Modality:    ch.Modality,
			SignalBlock: JoinPath("signal", name),
		})
	}
	for _, name := range sortedChannelNames(ep.Actions) {
		ch := ep.Actions[name]
		channelsMeta.Channels = append(channelsMeta.Channels, wshardChannelDef{
			ID:          name,
			DType:       canonicalDType(ch.DType),
			Shape:       ch.Shape,
			Modality:    ch.Modality,
			SignalBlock: JoinPath("action", name),
		})
	}
	// omen/ and uncert/ blocks are bare tensor bytes; meta/channels is where
	// their dtype and shape are declared, same as every other channel. Sorted by
	// block name so a Go map's iteration order never reaches the seal.
	omenBlocks := map[string]*WShardChannel{}
	for chID, models := range ep.Omens {
		for modelID, ch := range models {
			omenBlocks[OmenBlockName(chID, modelID)] = ch
		}
	}
	for _, block := range sortedChannelNames(omenBlocks) {
		ch := omenBlocks[block]
		channelsMeta.Channels = append(channelsMeta.Channels, wshardChannelDef{
			ID:          block,
			DType:       canonicalDType(ch.DType),
			Shape:       ch.Shape,
			SignalBlock: block,
		})
	}
	for _, block := range sortedChannelNames(ep.Uncerts) {
		ch := ep.Uncerts[block]
		channelsMeta.Channels = append(channelsMeta.Channels, wshardChannelDef{
			ID:          block,
			DType:       canonicalDType(ch.DType),
			Shape:       ch.Shape,
			SignalBlock: block,
		})
	}

	chanJSON, err := canonicalJSON(channelsMeta)
	if err != nil {
		cleanup()
		return fmt.Errorf("wshard: marshal meta/channels: %w", err)
	}
	if err := put("meta/channels", chanJSON, ContentTypeJSON); err != nil {
		cleanup()
		return fmt.Errorf("wshard: write meta/channels: %w", err)
	}

	// signal/*
	for _, name := range sortedChannelNames(ep.Observations) {
		ch := ep.Observations[name]
		if err := put(JoinPath("signal", name), ch.Data, ContentTypeUnknown); err != nil {
			cleanup()
			return fmt.Errorf("wshard: write signal/%s: %w", name, err)
		}
	}

	// action/*
	for _, name := range sortedChannelNames(ep.Actions) {
		ch := ep.Actions[name]
		if err := put(JoinPath("action", name), ch.Data, ContentTypeUnknown); err != nil {
			cleanup()
			return fmt.Errorf("wshard: write action/%s: %w", name, err)
		}
	}

	// reward
	rewardBuf := make([]byte, len(ep.Rewards)*4)
	for i, r := range ep.Rewards {
		binary.LittleEndian.PutUint32(rewardBuf[i*4:], math.Float32bits(r))
	}
	if err := put("reward", rewardBuf, ContentTypeUnknown); err != nil {
		cleanup()
		return fmt.Errorf("wshard: write reward: %w", err)
	}

	// done
	doneBuf := make([]byte, len(ep.Terminations))
	for i, t := range ep.Terminations {
		if t {
			doneBuf[i] = 1
		}
	}
	if err := put("done", doneBuf, ContentTypeUnknown); err != nil {
		cleanup()
		return fmt.Errorf("wshard: write done: %w", err)
	}

	// time/ticks. FORMAT.md lists it and the Python writer emits it; a block
	// present in one writer's output and absent from the other's is a
	// difference in the leaf map, so the two never agree on an identity.
	ticksBuf := make([]byte, ep.LengthT*4)
	for i := 0; i < ep.LengthT; i++ {
		binary.LittleEndian.PutUint32(ticksBuf[i*4:], uint32(int32(i)))
	}
	if err := put("time/ticks", ticksBuf, ContentTypeUnknown); err != nil {
		cleanup()
		return fmt.Errorf("wshard: write time/ticks: %w", err)
	}

	// omen blocks
	for chID, models := range ep.Omens {
		for modelID, ch := range models {
			blockName := OmenBlockName(chID, modelID)
			if err := put(blockName, ch.Data, ContentTypeUnknown); err != nil {
				cleanup()
				return fmt.Errorf("wshard: write %s: %w", blockName, err)
			}
		}
	}

	// uncert blocks
	for blockName, ch := range ep.Uncerts {
		if err := put(blockName, ch.Data, ContentTypeUnknown); err != nil {
			cleanup()
			return fmt.Errorf("wshard: write %s: %w", blockName, err)
		}
	}

	// residual blocks
	for chID, res := range ep.Residuals {
		blockName := ResidualSign2ndDiffBlockName(chID)
		if err := put(blockName, res.Data, ContentTypeUnknown); err != nil {
			cleanup()
			return fmt.Errorf("wshard: write %s: %w", blockName, err)
		}
	}

	if ep.Provenance != nil {
		provJSON, err := provenanceBlockBytes(ep.Provenance)
		if err != nil {
			cleanup()
			return fmt.Errorf("wshard: marshal %s: %w", ProvenanceBlock, err)
		}
		if err := put(ProvenanceBlock, provJSON, ContentTypeJSON); err != nil {
			cleanup()
			return fmt.Errorf("wshard: write %s: %w", ProvenanceBlock, err)
		}
	}

	identity, err := identityBlockBytes(leaves)
	if err != nil {
		cleanup()
		return fmt.Errorf("wshard: marshal %s: %w", IdentityBlock, err)
	}
	if err := w.WriteEntryTyped(IdentityBlock, identity, ContentTypeJSON); err != nil {
		cleanup()
		return fmt.Errorf("wshard: write %s: %w", IdentityBlock, err)
	}

	return w.Close()
}

// ============================================================
// Reader
// ============================================================

// OpenWShard reads a WShard file and returns the decoded episode.
func OpenWShard(path string) (*WShardEpisode, error) {
	r, err := OpenShard(path)
	if err != nil {
		return nil, fmt.Errorf("wshard: open: %w", err)
	}
	defer r.Close()

	if r.Header().Role != ShardRoleWShard {
		return nil, fmt.Errorf("wshard: expected role WShard (0x%02x), got %s (0x%02x)",
			uint8(ShardRoleWShard), r.Header().Role, uint8(r.Header().Role))
	}

	metaRaw, err := r.ReadEntryByName("meta/wshard")
	if err != nil {
		return nil, fmt.Errorf("wshard: read meta/wshard: %w", err)
	}
	var metaBlock wshardMetaBlock
	if err := json.Unmarshal(metaRaw, &metaBlock); err != nil {
		return nil, fmt.Errorf("wshard: parse meta/wshard: %w", err)
	}

	epRaw, err := r.ReadEntryByName("meta/episode")
	if err != nil {
		return nil, fmt.Errorf("wshard: read meta/episode: %w", err)
	}
	var epMeta wshardEpisodeMeta
	if err := json.Unmarshal(epRaw, &epMeta); err != nil {
		return nil, fmt.Errorf("wshard: parse meta/episode: %w", err)
	}

	chanRaw, err := r.ReadEntryByName("meta/channels")
	if err != nil {
		return nil, fmt.Errorf("wshard: read meta/channels: %w", err)
	}

	var provenance *WShardProvenance
	if provRaw, err := r.ReadEntryByName(ProvenanceBlock); err == nil {
		provenance, err = parseProvenance(provRaw)
		if err != nil {
			return nil, err
		}
	} else if !errors.Is(err, ErrEntryNotFound) {
		return nil, fmt.Errorf("wshard: read %s: %w", ProvenanceBlock, err)
	}

	observations := make(map[string]*WShardChannel)
	actions := make(map[string]*WShardChannel)
	parsedEntries := make(map[string]struct{})

	// meta/channels also declares dtype/shape for omen/ and uncert/ blocks, whose
	// bytes carry neither. Those rows are held back from readWShardChannel --
	// they are not observations -- and applied when the blocks are read below.
	chanDefs := make(map[string]wshardChannelDef)

	var channelsMeta wshardChannelsMeta
	if err := json.Unmarshal(chanRaw, &channelsMeta); err == nil && len(channelsMeta.Channels) > 0 {
		for _, def := range channelsMeta.Channels {
			fullName := def.SignalBlock
			if fullName == "" {
				fullName = JoinPath("signal", def.ID)
			}
			chanDefs[fullName] = def
			if strings.HasPrefix(fullName, "omen/") || strings.HasPrefix(fullName, "uncert/") {
				continue
			}
			if err := readWShardChannel(r, fullName, def, observations, actions); err != nil {
				return nil, err
			}
			parsedEntries[fullName] = struct{}{}
		}
	} else {
		var legacy map[string]wshardChannelDef
		if err := json.Unmarshal(chanRaw, &legacy); err != nil {
			return nil, fmt.Errorf("wshard: parse meta/channels: %w", err)
		}
		for fullName, def := range legacy {
			if def.ID == "" {
				def.ID = pathRemainder(fullName)
			}
			if err := readWShardChannel(r, fullName, def, observations, actions); err != nil {
				return nil, err
			}
			parsedEntries[fullName] = struct{}{}
		}
	}

	// Canonical Python/TS files may only describe observation channels in metadata.
	for _, fullName := range r.ListPrefix("signal/") {
		if _, ok := parsedEntries[fullName]; ok {
			continue
		}
		if err := readWShardChannel(r, fullName, wshardChannelDef{
			ID:    pathRemainder(fullName),
			DType: "",
		}, observations, actions); err != nil {
			return nil, err
		}
	}
	for _, fullName := range r.ListPrefix("action/") {
		if _, ok := parsedEntries[fullName]; ok {
			continue
		}
		if err := readWShardChannel(r, fullName, wshardChannelDef{
			ID:    pathRemainder(fullName),
			DType: "",
		}, observations, actions); err != nil {
			return nil, err
		}
	}

	// Parse omen blocks: "omen/{channelID}/{modelID}"
	omens := make(map[string]map[string]*WShardChannel)
	for _, fullName := range r.ListPrefix("omen/") {
		data, err := r.ReadEntryByName(fullName)
		if err != nil {
			return nil, fmt.Errorf("wshard: read %s: %w", fullName, err)
		}
		// Split: "omen" / channelID / modelID
		rest := strings.TrimPrefix(fullName, "omen/")
		parts := strings.SplitN(rest, PathSeparator, 2)
		if len(parts) != 2 {
			continue // skip malformed
		}
		chID, modelID := parts[0], parts[1]
		if omens[chID] == nil {
			omens[chID] = make(map[string]*WShardChannel)
		}
		omens[chID][modelID] = &WShardChannel{
			Name:  fullName,
			DType: declaredDType(chanDefs, fullName),
			Shape: chanDefs[fullName].Shape,
			Data:  data,
		}
	}

	// Parse uncert blocks: "uncert/{channelID}/{modelID}/{type}"
	uncerts := make(map[string]*WShardChannel)
	for _, fullName := range r.ListPrefix("uncert/") {
		data, err := r.ReadEntryByName(fullName)
		if err != nil {
			return nil, fmt.Errorf("wshard: read %s: %w", fullName, err)
		}
		uncerts[fullName] = &WShardChannel{
			Name:  fullName,
			DType: declaredDType(chanDefs, fullName),
			Shape: chanDefs[fullName].Shape,
			Data:  data,
		}
	}

	// Parse residual blocks: "residual/{channelID}/sign2nddiff"
	residuals := make(map[string]*WShardResidual)
	for _, fullName := range r.ListPrefix("residual/") {
		if !strings.HasSuffix(fullName, "/sign2nddiff") {
			continue
		}
		data, err := r.ReadEntryByName(fullName)
		if err != nil {
			return nil, fmt.Errorf("wshard: read %s: %w", fullName, err)
		}
		// Extract channelID: strip "residual/" prefix and "/sign2nddiff" suffix
		chID := fullName[len("residual/") : len(fullName)-len("/sign2nddiff")]
		residuals[chID] = &WShardResidual{
			ChannelID: chID,
			Type:      "sign2nddiff",
			Data:      data,
		}
	}

	rewardRaw, err := r.ReadEntryByName("reward")
	if err != nil {
		return nil, fmt.Errorf("wshard: read reward: %w", err)
	}
	if len(rewardRaw)%4 != 0 {
		return nil, fmt.Errorf("wshard: reward data length %d not multiple of 4", len(rewardRaw))
	}
	rewards := make([]float32, len(rewardRaw)/4)
	for i := range rewards {
		rewards[i] = math.Float32frombits(binary.LittleEndian.Uint32(rewardRaw[i*4:]))
	}

	doneRaw, err := r.ReadEntryByName("done")
	if err != nil {
		return nil, fmt.Errorf("wshard: read done: %w", err)
	}
	terminations := make([]bool, len(doneRaw))
	for i, b := range doneRaw {
		terminations[i] = b != 0
	}

	var readTsRange [2]int
	if epMeta.TimestepRange != nil {
		readTsRange = *epMeta.TimestepRange
	}

	return &WShardEpisode{
		ID:            epMeta.resolvedID(),
		EnvID:         epMeta.EnvID,
		LengthT:       epMeta.resolvedLengthT(),
		Timebase:      resolvedEpisodeTimebase(&metaBlock, &epMeta),
		Observations:  observations,
		Actions:       actions,
		Rewards:       rewards,
		Terminations:  terminations,
		Omens:         omens,
		Uncerts:       uncerts,
		Residuals:     residuals,
		ChunkIndex:    epMeta.ChunkIndex,
		TotalChunks:   epMeta.TotalChunks,
		TimestepRange: readTsRange,
		Metadata:      epMeta.Metadata,
		Provenance:    provenance,
	}, nil
}

// ============================================================
// Helpers
// ============================================================

func episodeTimebaseMeta(tb WShardTimebase) *wshardTimebaseJ {
	meta := &wshardTimebaseJ{Type: tb.Type}
	if tb.Type == "ticks" && tb.TickHz > 0 {
		meta.TickHz = tb.TickHz
		meta.DtNS = int64(math.Round(1e9 / tb.TickHz))
	}
	return meta
}

func resolvedEpisodeTimebase(meta *wshardMetaBlock, episode *wshardEpisodeMeta) WShardTimebase {
	if episode != nil && episode.Timebase != nil {
		return episode.Timebase.toPublic()
	}
	if meta != nil && meta.Timebase != nil {
		return meta.Timebase.toPublic()
	}
	return WShardTimebase{}
}

func readWShardChannel(
	r *ShardReader,
	fullName string,
	def wshardChannelDef,
	observations map[string]*WShardChannel,
	actions map[string]*WShardChannel,
) error {
	data, err := r.ReadEntryByName(fullName)
	if err != nil {
		return fmt.Errorf("wshard: read %s: %w", fullName, err)
	}
	prefix, name, ok := splitNamespace(fullName)
	if !ok {
		return fmt.Errorf("wshard: unexpected channel path %q", fullName)
	}
	if def.ID != "" {
		name = def.ID
	}

	ch := &WShardChannel{
		Name:     name,
		DType:    canonicalDType(def.DType),
		Shape:    def.Shape,
		Data:     data,
		Modality: def.Modality,
	}

	switch prefix {
	case "signal":
		observations[name] = ch
	case "action":
		actions[name] = ch
	default:
		return fmt.Errorf("wshard: unknown channel prefix %q", prefix)
	}
	return nil
}

func splitNamespace(name string) (prefix string, remainder string, ok bool) {
	prefix, remainder, ok = strings.Cut(name, PathSeparator)
	if !ok || prefix == "" || remainder == "" {
		return "", "", false
	}
	return prefix, remainder, true
}

func pathRemainder(name string) string {
	_, remainder, ok := splitNamespace(name)
	if !ok {
		return name
	}
	return remainder
}

// declaredDType is the dtype meta/channels gives a block, defaulting to u8 --
// raw bytes, which is what a block with no declaration is. The Python reader
// falls back the same way, and the fallback is written back into the seal.
func declaredDType(defs map[string]wshardChannelDef, block string) string {
	if dt := defs[block].DType; dt != "" {
		return dt
	}
	return "u8"
}

func sortedChannelNames(channels map[string]*WShardChannel) []string {
	names := make([]string, 0, len(channels))
	for name := range channels {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func canonicalDType(dtype string) string {
	switch strings.ToLower(dtype) {
	case "bool":
		return "bool"
	case "u8", "uint8":
		return "u8"
	case "i8", "int8":
		return "i8"
	case "u16", "uint16":
		return "u16"
	case "i16", "int16":
		return "i16"
	case "f16", "float16":
		return "f16"
	case "bf16", "bfloat16":
		return "bf16"
	case "u32", "uint32":
		return "u32"
	case "i32", "int32":
		return "i32"
	case "f32", "float32":
		return "f32"
	case "u64", "uint64":
		return "u64"
	case "i64", "int64":
		return "i64"
	case "f64", "float64":
		return "f64"
	default:
		return dtype
	}
}

// dtypeSizeBytes returns the byte size for a dtype string.
// Returns 0 for unknown dtypes.
func dtypeSizeBytes(dtype string) int {
	switch canonicalDType(dtype) {
	case "bool":
		return 1
	case "u8", "i8":
		return 1
	case "u16", "i16", "f16", "bf16":
		return 2
	case "u32", "i32", "f32":
		return 4
	case "u64", "i64", "f64":
		return 8
	default:
		return 0
	}
}
