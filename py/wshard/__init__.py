"""
W-SHARD Python - World Episode Adapter for Python.

Supports bidirectional conversion between:
- W-SHARD: Native format with Signal/Omen/Residual lanes
- DreamerV3: NPZ files with episode data
- TD-MPC2: PyTorch TensorDict files (optional)
- Minari/D4RL: HDF5 datasets (optional)

Example:
    from wshard import Episode, load, save, convert

    # Load a DreamerV3 episode
    ep = load("episode.npz")

    # Save as W-SHARD
    save(ep, "episode.wshard")

    # Or convert directly
    convert("input.npz", "output.wshard")
"""

from .types import (
    Format,
    DType,
    Channel,
    Episode,
    Residual,
    TimebaseSpec,
    TimebaseType,
    Modality,
)
from .compress import (
    CompressionType,
    CompressionLevel,
    Compressor,
)
from .residual import (
    compute_sign2nd_diff,
    compute_sign2nd_diff_multidim,
    compute_error_residual,
    pack_residual_bits,
    unpack_residual_bits,
    pack_multidim_residual_bits,
    unpack_multidim_residual_bits,
    quantize_delta,
    dequantize_delta,
    reconstruct_from_omen_and_delta,
    # Gap 3: Cowrie BITMASK integration
    pack_residual_bitmask,
    unpack_residual_bitmask,
    pack_multidim_residual_bitmask,
    unpack_multidim_residual_bitmask,
    HAS_COWRIE,
)
from .streaming import WShardStreamWriter, ChannelDef as StreamChannelDef
from .chunked import (
    ChunkedEpisodeWriter,
    ChunkedEpisodeReader,
    ChunkManifest,
)
from .dreamer import load_dreamer, save_dreamer
from .wshard import (
    load_wshard,
    save_wshard,
    # Gap 5: VLA Multi-Modal
    add_multimodal_observation,
    get_multimodal_observations,
    # Gap 2: Latent Action Storage
    LATENT_ACTION_CHANNEL,
    LATENT_CODEBOOK_CHANNEL,
    set_latent_actions,
    get_latent_actions,
    get_latent_codebook,
)
from .convert import load, save, convert, detect_format

# Optional HuggingFace Hub integration
try:
    from .huggingface import (
        HuggingFaceAdapter,
        upload_to_hub,
        download_from_hub,
    )

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    HuggingFaceAdapter = None  # type: ignore
    upload_to_hub = None  # type: ignore
    download_from_hub = None  # type: ignore

__version__ = "0.1.0"
__all__ = [
    # Types
    "Format",
    "DType",
    "Channel",
    "Episode",
    "Residual",
    "TimebaseSpec",
    "TimebaseType",
    "Modality",
    # Compression
    "CompressionType",
    "CompressionLevel",
    "Compressor",
    # Residuals
    "compute_sign2nd_diff",
    "compute_sign2nd_diff_multidim",
    "compute_error_residual",
    "pack_residual_bits",
    "unpack_residual_bits",
    "pack_multidim_residual_bits",
    "unpack_multidim_residual_bits",
    "quantize_delta",
    "dequantize_delta",
    "reconstruct_from_omen_and_delta",
    # Cowrie BITMASK residuals
    "pack_residual_bitmask",
    "unpack_residual_bitmask",
    "pack_multidim_residual_bitmask",
    "unpack_multidim_residual_bitmask",
    "HAS_COWRIE",
    # VLA Multi-Modal
    "add_multimodal_observation",
    "get_multimodal_observations",
    # Latent Action Storage
    "LATENT_ACTION_CHANNEL",
    "LATENT_CODEBOOK_CHANNEL",
    "set_latent_actions",
    "get_latent_actions",
    "get_latent_codebook",
    # Chunked Episodes
    "ChunkedEpisodeWriter",
    "ChunkedEpisodeReader",
    "ChunkManifest",
    # Streaming Append
    "WShardStreamWriter",
    "StreamChannelDef",
    # Functions
    "load",
    "save",
    "convert",
    "detect_format",
    "load_dreamer",
    "save_dreamer",
    "load_wshard",
    "save_wshard",
    # HuggingFace Hub (optional)
    "HuggingFaceAdapter",
    "upload_to_hub",
    "download_from_hub",
    "HF_AVAILABLE",
]
