// wshard_multimodal.go — Multi-modal VLA and latent action helpers for W-SHARD.
package shard

// Modality represents a sensor modality type for VLA multi-modal observations.
type Modality string

const (
	ModalityRGB            Modality = "rgb"
	ModalityDepth          Modality = "depth"
	ModalityLanguage       Modality = "language"
	ModalityProprioception Modality = "proprioception"
	ModalityAudio          Modality = "audio"
	ModalityVideo          Modality = "video"
	ModalityPointcloud     Modality = "pointcloud"
)

// AddMultiModalObservation adds a multi-modal observation channel to the episode.
// Convention: stored as ep.Observations["{group}/{modality}"] with ch.Modality set.
func (ep *WShardEpisode) AddMultiModalObservation(group string, mod Modality, ch *WShardChannel) {
	if ep.Observations == nil {
		ep.Observations = make(map[string]*WShardChannel)
	}
	key := group + "/" + string(mod)
	ch.Modality = string(mod)
	ep.Observations[key] = ch
}

// GetMultiModalObservations returns observation channels matching the given group and modality.
func (ep *WShardEpisode) GetMultiModalObservations(group string, mod Modality) map[string]*WShardChannel {
	result := make(map[string]*WShardChannel)
	prefix := group + "/" + string(mod)
	for key, ch := range ep.Observations {
		if key == prefix {
			result[key] = ch
		}
	}
	return result
}

// SetLatentActions stores latent action embeddings in the omen lane.
// Convention: omen/latent_action/{modelID}
func (ep *WShardEpisode) SetLatentActions(modelID string, embeddings *WShardChannel) {
	if ep.Omens == nil {
		ep.Omens = make(map[string]map[string]*WShardChannel)
	}
	if ep.Omens["latent_action"] == nil {
		ep.Omens["latent_action"] = make(map[string]*WShardChannel)
	}
	ep.Omens["latent_action"][modelID] = embeddings
}

// SetLatentActionCodebook stores VQ-VAE codebook indices in the omen lane.
// Convention: omen/latent_action_codebook/{modelID}
func (ep *WShardEpisode) SetLatentActionCodebook(modelID string, indices *WShardChannel) {
	if ep.Omens == nil {
		ep.Omens = make(map[string]map[string]*WShardChannel)
	}
	if ep.Omens["latent_action_codebook"] == nil {
		ep.Omens["latent_action_codebook"] = make(map[string]*WShardChannel)
	}
	ep.Omens["latent_action_codebook"][modelID] = indices
}

// GetLatentActions retrieves latent action embeddings for a model.
func (ep *WShardEpisode) GetLatentActions(modelID string) *WShardChannel {
	if ep.Omens == nil {
		return nil
	}
	models := ep.Omens["latent_action"]
	if models == nil {
		return nil
	}
	return models[modelID]
}

// GetLatentActionCodebook retrieves latent action codebook indices for a model.
func (ep *WShardEpisode) GetLatentActionCodebook(modelID string) *WShardChannel {
	if ep.Omens == nil {
		return nil
	}
	models := ep.Omens["latent_action_codebook"]
	if models == nil {
		return nil
	}
	return models[modelID]
}
