"""
W-SHARD Types - Core data structures for world model episodes.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np


class Format(str, Enum):
    """Episode format identifiers."""

    UNKNOWN = ""
    WSHARD = "wshard"
    DREAMER_V3 = "dreamerv3"
    TDMPC2 = "tdmpc2"
    MINARI = "minari"
    D4RL = "d4rl"  # Legacy, maps to Minari


class DType(str, Enum):
    """Tensor data types."""

    FLOAT64 = "f64"
    FLOAT32 = "f32"
    FLOAT16 = "f16"
    BFLOAT16 = "bf16"
    INT64 = "i64"
    INT32 = "i32"
    INT16 = "i16"
    INT8 = "i8"
    UINT64 = "u64"
    UINT32 = "u32"
    UINT16 = "u16"
    UINT8 = "u8"
    BOOL = "bool"

    @property
    def numpy_dtype(self) -> np.dtype:
        """Convert to NumPy dtype."""
        mapping = {
            DType.FLOAT64: np.float64,
            DType.FLOAT32: np.float32,
            DType.FLOAT16: np.float16,
            DType.INT64: np.int64,
            DType.INT32: np.int32,
            DType.INT16: np.int16,
            DType.INT8: np.int8,
            DType.UINT64: np.uint64,
            DType.UINT32: np.uint32,
            DType.UINT16: np.uint16,
            DType.UINT8: np.uint8,
            DType.BOOL: np.bool_,
        }
        # Handle bfloat16: prefer ml_dtypes if available, else uint16 for byte-preserving storage
        if self == DType.BFLOAT16:
            try:
                import ml_dtypes
                return np.dtype(ml_dtypes.bfloat16)
            except ImportError:
                return np.dtype(np.uint16)  # Preserves 2-byte layout
        return np.dtype(mapping.get(self, np.float32))

    @property
    def size(self) -> int:
        """Bytes per element."""
        sizes = {
            DType.FLOAT64: 8,
            DType.INT64: 8,
            DType.UINT64: 8,
            DType.FLOAT32: 4,
            DType.INT32: 4,
            DType.UINT32: 4,
            DType.FLOAT16: 2,
            DType.BFLOAT16: 2,
            DType.INT16: 2,
            DType.UINT16: 2,
            DType.INT8: 1,
            DType.UINT8: 1,
            DType.BOOL: 1,
        }
        return sizes.get(self, 4)

    @classmethod
    def from_numpy(cls, dtype: np.dtype) -> "DType":
        """Convert NumPy dtype to DType."""
        # Check for ml_dtypes bfloat16 first
        try:
            import ml_dtypes
            if dtype == np.dtype(ml_dtypes.bfloat16):
                return cls.BFLOAT16
        except ImportError:
            pass
        mapping = {
            np.float64: cls.FLOAT64,
            np.float32: cls.FLOAT32,
            np.float16: cls.FLOAT16,
            np.int64: cls.INT64,
            np.int32: cls.INT32,
            np.int16: cls.INT16,
            np.int8: cls.INT8,
            np.uint64: cls.UINT64,
            np.uint32: cls.UINT32,
            np.uint16: cls.UINT16,
            np.uint8: cls.UINT8,
            np.bool_: cls.BOOL,
        }
        return mapping.get(dtype.type, cls.FLOAT32)


class Modality(str, Enum):
    """Sensor modality types for VLA multi-modal observations."""

    RGB = "rgb"
    DEPTH = "depth"
    LANGUAGE = "language"
    PROPRIOCEPTION = "proprioception"
    AUDIO = "audio"
    VIDEO = "video"
    POINTCLOUD = "pointcloud"

    @property
    def content_type(self) -> int:
        """Map modality to shard v2 content type code."""
        mapping = {
            Modality.RGB: 0x0006,           # IMAGE
            Modality.DEPTH: 0x0001,         # TENSOR
            Modality.LANGUAGE: 0x0005,      # TEXT
            Modality.PROPRIOCEPTION: 0x0001, # TENSOR
            Modality.AUDIO: 0x0007,         # AUDIO
            Modality.VIDEO: 0x0008,         # VIDEO
            Modality.POINTCLOUD: 0x0001,    # TENSOR
        }
        return mapping.get(self, 0x0000)


class TimebaseType(str, Enum):
    """Time representation types."""

    TICKS = "ticks"
    TIMESTAMPS_NS = "timestamps_ns"
    TIMESTAMPS_US = "timestamps_us"


@dataclass
class TimebaseSpec:
    """Timebase specification for an episode."""

    type: TimebaseType = TimebaseType.TICKS
    tick_hz: float = 30.0
    epoch_ns: int = 0


@dataclass
class Channel:
    """A single named data channel (observation, action, etc)."""

    name: str
    dtype: DType
    shape: List[int]  # Per-timestep shape (excluding T)
    data: np.ndarray  # Raw data, shape [T, ...shape]
    semantics: str = ""
    # VLA multi-modal fields (Gap 5)
    modality: Optional["Modality"] = None
    sampling_rate_hz: Optional[float] = None
    content_type: Optional[str] = None

    def __post_init__(self):
        """Validate and convert data if needed."""
        if not isinstance(self.data, np.ndarray):
            self.data = np.array(self.data)

    @property
    def length(self) -> int:
        """Number of timesteps in the channel."""
        if self.data is None or self.data.size == 0:
            return 0
        return self.data.shape[0]

    def clone(self) -> "Channel":
        """Create a deep copy."""
        return Channel(
            name=self.name,
            dtype=self.dtype,
            shape=self.shape.copy(),
            data=self.data.copy(),
            semantics=self.semantics,
            modality=self.modality,
            sampling_rate_hz=self.sampling_rate_hz,
            content_type=self.content_type,
        )


@dataclass
class Residual:
    """Compact residual encoding for a channel."""

    channel_id: str
    type: str  # "sign2nddiff" or "delta_q"
    data: bytes
    scales: Optional[List[float]] = None
    mask: Optional[bytes] = None


@dataclass
class Episode:
    """
    Universal episode interchange format.

    Captures all common episode data across formats:
    - Observations: named channels (e.g., "image", "state/pos")
    - Actions: discrete and/or continuous action channels
    - Rewards: scalar reward signal
    - Terminations/Truncations: episode boundary flags
    - Omens: cached model predictions per channel per model
    - Residuals: compact innovation representations
    - Metadata: format-specific auxiliary data
    """

    # Identity
    id: str
    env_id: str = ""

    # Temporal info
    length: int = 0
    timebase: TimebaseSpec = field(default_factory=TimebaseSpec)

    # Core data streams
    observations: Dict[str, Channel] = field(default_factory=dict)
    actions: Dict[str, Channel] = field(default_factory=dict)
    rewards: Optional[Channel] = None
    terminations: Optional[Channel] = None
    truncations: Optional[Channel] = None

    # W-SHARD specific: cached predictions
    # Key structure: channel_id -> model_id -> Channel
    omens: Dict[str, Dict[str, Channel]] = field(default_factory=dict)

    # W-SHARD specific: residual encodings
    residuals: Dict[str, Residual] = field(default_factory=dict)

    # Format-specific metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Source format (set on load, informational)
    source_format: Format = Format.UNKNOWN

    # Chunked episode fields (Gap 1)
    chunk_index: Optional[int] = None
    total_chunks: Optional[int] = None
    timestep_range: Optional[List[int]] = None  # [start_t, end_t]

    @property
    def is_chunked(self) -> bool:
        """Whether this episode is a chunk of a larger episode."""
        return self.chunk_index is not None

    def validate(self) -> None:
        """
        Validate episode for internal consistency.

        Raises:
            ValueError: If validation fails
        """
        if not self.id:
            raise ValueError("Episode ID is required")
        if self.length <= 0:
            raise ValueError(f"Invalid episode length: {self.length}")

        # Validate chunk fields if present
        if self.chunk_index is not None:
            if self.total_chunks is None or self.total_chunks <= 0:
                raise ValueError("total_chunks required when chunk_index is set")
            if self.chunk_index < 0 or self.chunk_index >= self.total_chunks:
                raise ValueError(
                    f"chunk_index {self.chunk_index} out of range [0, {self.total_chunks})"
                )
            if self.timestep_range is not None:
                if len(self.timestep_range) != 2:
                    raise ValueError("timestep_range must be [start_t, end_t]")
                if self.timestep_range[0] > self.timestep_range[1]:
                    raise ValueError("timestep_range start must be <= end")

        # Validate observations
        for name, ch in self.observations.items():
            if ch is None:
                raise ValueError(f"Nil observation channel: {name}")
            if ch.length != self.length:
                raise ValueError(
                    f"Observation '{name}' length mismatch: "
                    f"got {ch.length}, want {self.length}"
                )

        # Validate actions
        for name, ch in self.actions.items():
            if ch is None:
                raise ValueError(f"Nil action channel: {name}")
            if ch.length != self.length:
                raise ValueError(
                    f"Action '{name}' length mismatch: "
                    f"got {ch.length}, want {self.length}"
                )

        # Validate rewards
        if self.rewards is not None and self.rewards.length != self.length:
            raise ValueError(
                f"Rewards length mismatch: "
                f"got {self.rewards.length}, want {self.length}"
            )

        # Validate terminations
        if self.terminations is not None and self.terminations.length != self.length:
            raise ValueError(
                f"Terminations length mismatch: "
                f"got {self.terminations.length}, want {self.length}"
            )

        # Validate truncations
        if self.truncations is not None and self.truncations.length != self.length:
            raise ValueError(
                f"Truncations length mismatch: "
                f"got {self.truncations.length}, want {self.length}"
            )

    def clone(self) -> "Episode":
        """Create a deep copy."""
        ep = Episode(
            id=self.id,
            env_id=self.env_id,
            length=self.length,
            timebase=TimebaseSpec(
                type=self.timebase.type,
                tick_hz=self.timebase.tick_hz,
                epoch_ns=self.timebase.epoch_ns,
            ),
            source_format=self.source_format,
            chunk_index=self.chunk_index,
            total_chunks=self.total_chunks,
            timestep_range=self.timestep_range.copy() if self.timestep_range else None,
        )

        # Clone observations
        for k, v in self.observations.items():
            ep.observations[k] = v.clone()

        # Clone actions
        for k, v in self.actions.items():
            ep.actions[k] = v.clone()

        # Clone rewards, terminations, truncations
        if self.rewards:
            ep.rewards = self.rewards.clone()
        if self.terminations:
            ep.terminations = self.terminations.clone()
        if self.truncations:
            ep.truncations = self.truncations.clone()

        # Clone omens
        for ch_id, models in self.omens.items():
            ep.omens[ch_id] = {}
            for model_id, ch in models.items():
                ep.omens[ch_id][model_id] = ch.clone()

        # Clone residuals (shallow - data is bytes)
        for k, v in self.residuals.items():
            ep.residuals[k] = Residual(
                channel_id=v.channel_id,
                type=v.type,
                data=v.data,
                scales=v.scales.copy() if v.scales else None,
                mask=v.mask,
            )

        # Clone metadata (shallow)
        ep.metadata = self.metadata.copy()

        return ep
