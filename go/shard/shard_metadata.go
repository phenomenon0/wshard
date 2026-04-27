package shard

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"sort"
	"time"
)

// ShardMetadata represents the standardized schema JSON structure.
type ShardMetadata struct {
	// Schema identification
	SchemaVersion string `json:"schema_version,omitempty"` // e.g., "shard-v2.1"
	SchemaURI     string `json:"schema_uri,omitempty"`     // URL to schema definition
	SchemaID      string `json:"schema_id,omitempty"`      // e.g., SHA256 hash prefix of schema JSON

	// Provenance
	CreatedAt time.Time `json:"created_at,omitempty"`
	SourceURI string    `json:"source_uri,omitempty"`
	Producer  string    `json:"producer,omitempty"`

	// Shard-level metadata
	Description string         `json:"description,omitempty"`
	Tags        []string       `json:"tags,omitempty"`
	// Tag dictionary: maps bit position (0-15) to tag name for O(1) tag filtering
	TagDictionary []string `json:"tag_dictionary,omitempty"`
	Extra       map[string]any `json:"extra,omitempty"`
	Profile     string         `json:"profile,omitempty"`
	SampleShard *SampleProfile `json:"sample_shard,omitempty"`
	Manifest    *ManifestMeta  `json:"manifest,omitempty"`

	// Per-entry metadata (keyed by entry name)
	EntryMetadata map[string]*EntryMeta `json:"entry_metadata,omitempty"`
}

// EntryMeta holds per-entry metadata.
type EntryMeta struct {
	ContentType       string         `json:"content_type,omitempty"` // MIME-like type
	Tags              []string       `json:"tags,omitempty"`
	Description       string         `json:"description,omitempty"`
	Extra             map[string]any `json:"extra,omitempty"`
	Codec             string         `json:"codec,omitempty"`
	CodecVersion      string         `json:"codec_version,omitempty"`
	SchemaFingerprint string         `json:"schema_fingerprint,omitempty"`
	SemanticType      string         `json:"semantic_type,omitempty"`
	CanonicalHash     string         `json:"canonical_hash,omitempty"`
	BaseHash          string         `json:"base_hash,omitempty"`
	RowCount          uint64         `json:"row_count,omitempty"`
	Shape             []int64        `json:"shape,omitempty"`
	Stats             map[string]any `json:"stats,omitempty"`
}

// SampleProfile describes shard-level dataset metadata for SampleShard files.
type SampleProfile struct {
	DatasetName   string         `json:"dataset_name,omitempty"`
	SampleIDType  string         `json:"sample_id_type,omitempty"`
	KeyEncoding   string         `json:"key_encoding,omitempty"`
	SampleCount   uint64         `json:"sample_count,omitempty"`
	DatasetSchema map[string]any `json:"dataset_schema,omitempty"`
	Splits        map[string]any `json:"splits,omitempty"`
	LabelMap      map[string]any `json:"label_map,omitempty"`
	FeatureStats  map[string]any `json:"feature_stats,omitempty"`
}

// ManifestMeta describes shard-level metadata for manifest shards.
type ManifestMeta struct {
	Files      []*ManifestFileRef `json:"files,omitempty"`
	Partitions map[string]any     `json:"partitions,omitempty"`
}

// ManifestFileRef describes a single file referenced by a manifest shard.
type ManifestFileRef struct {
	URI        string `json:"uri,omitempty"`
	SHA256     string `json:"sha256,omitempty"`
	Role       string `json:"role,omitempty"`
	Profile    string `json:"profile,omitempty"`
	StartKey   string `json:"start_key,omitempty"`
	EndKey     string `json:"end_key,omitempty"`
	EntryCount uint64 `json:"entry_count,omitempty"`
}

// NewShardMetadata creates a new metadata instance with defaults.
func NewShardMetadata() *ShardMetadata {
	return &ShardMetadata{
		SchemaVersion: "shard-v2.1",
		CreatedAt:     time.Now().UTC(),
		EntryMetadata: make(map[string]*EntryMeta),
	}
}

// SetSampleProfile configures the shard as a SampleShard-style dataset container.
func (m *ShardMetadata) SetSampleProfile(profile *SampleProfile) {
	m.Profile = "sampleshard.v1"
	m.SampleShard = profile
}

// SetManifestProfile configures the shard as a manifest container.
func (m *ShardMetadata) SetManifestProfile(profile *ManifestMeta) {
	m.Profile = "manifest.v1"
	m.Manifest = profile
}

// SetEntryMeta sets metadata for an entry.
func (m *ShardMetadata) SetEntryMeta(name string, meta *EntryMeta) {
	if m.EntryMetadata == nil {
		m.EntryMetadata = make(map[string]*EntryMeta)
	}
	m.EntryMetadata[name] = meta
}

// GetEntryMeta gets metadata for an entry.
func (m *ShardMetadata) GetEntryMeta(name string) *EntryMeta {
	if m.EntryMetadata == nil {
		return nil
	}
	return m.EntryMetadata[name]
}

// AddTag adds a shard-level tag.
func (m *ShardMetadata) AddTag(tag string) {
	for _, t := range m.Tags {
		if t == tag {
			return
		}
	}
	m.Tags = append(m.Tags, tag)
}

// Marshal serializes to JSON.
func (m *ShardMetadata) Marshal() ([]byte, error) {
	return json.MarshalIndent(m, "", "  ")
}

// Unmarshal deserializes from JSON.
func (m *ShardMetadata) Unmarshal(data []byte) error {
	return json.Unmarshal(data, m)
}

// ComputeSchemaID computes a schema identifier from the metadata by hashing the
// tag_dictionary and entry_metadata keys+content_types (the "shape" of the schema).
func (m *ShardMetadata) ComputeSchemaID() string {
	h := sha256.New()
	// Include tag dictionary
	for _, t := range m.TagDictionary {
		h.Write([]byte(t))
		h.Write([]byte{0})
	}
	// Include entry names and content types (sorted for determinism)
	names := make([]string, 0, len(m.EntryMetadata))
	for k := range m.EntryMetadata {
		names = append(names, k)
	}
	sort.Strings(names)
	for _, name := range names {
		em := m.EntryMetadata[name]
		h.Write([]byte(name))
		h.Write([]byte{0})
		if em != nil {
			h.Write([]byte(em.ContentType))
			h.Write([]byte{0})
			h.Write([]byte(em.SemanticType))
			h.Write([]byte{0})
		}
	}
	return hex.EncodeToString(h.Sum(nil))[:16] // 16-char prefix
}

// ParseShardMetadata parses metadata from JSON bytes.
func ParseShardMetadata(data []byte) (*ShardMetadata, error) {
	m := &ShardMetadata{}
	if err := m.Unmarshal(data); err != nil {
		return nil, err
	}
	return m, nil
}
