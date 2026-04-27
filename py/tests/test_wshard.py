"""
Tests for W-SHARD Python package.
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from wshard import (
    Episode,
    Channel,
    DType,
    Format,
    CompressionType,
    CompressionLevel,
    load,
    save,
    convert,
    detect_format,
    load_dreamer,
    save_dreamer,
    load_wshard,
    save_wshard,
    compute_sign2nd_diff,
    compute_sign2nd_diff_multidim,
    pack_residual_bits,
    unpack_residual_bits,
    pack_multidim_residual_bits,
    unpack_multidim_residual_bits,
)


class TestTypes:
    """Test type definitions."""

    def test_dtype_sizes(self):
        """Test dtype byte sizes."""
        assert DType.FLOAT64.size == 8
        assert DType.FLOAT32.size == 4
        assert DType.FLOAT16.size == 2
        assert DType.INT64.size == 8
        assert DType.INT32.size == 4
        assert DType.INT8.size == 1
        assert DType.BOOL.size == 1

    def test_dtype_numpy(self):
        """Test dtype to numpy conversion."""
        assert DType.FLOAT32.numpy_dtype == np.float32
        assert DType.INT64.numpy_dtype == np.int64
        assert DType.BOOL.numpy_dtype == np.bool_

    def test_dtype_from_numpy(self):
        """Test numpy to dtype conversion."""
        assert DType.from_numpy(np.dtype(np.float32)) == DType.FLOAT32
        assert DType.from_numpy(np.dtype(np.int64)) == DType.INT64


class TestChannel:
    """Test Channel class."""

    def test_channel_length(self):
        """Test channel length property."""
        data = np.random.randn(100, 16).astype(np.float32)
        ch = Channel(name="test", dtype=DType.FLOAT32, shape=[16], data=data)
        assert ch.length == 100

    def test_channel_clone(self):
        """Test channel cloning."""
        data = np.random.randn(50, 8).astype(np.float32)
        ch = Channel(name="test", dtype=DType.FLOAT32, shape=[8], data=data)
        clone = ch.clone()

        assert clone.name == ch.name
        assert clone.dtype == ch.dtype
        assert clone.shape == ch.shape
        assert np.array_equal(clone.data, ch.data)
        assert clone.data is not ch.data  # Deep copy


class TestEpisode:
    """Test Episode class."""

    def test_episode_validate(self):
        """Test episode validation."""
        ep = Episode(id="test", length=100)
        ep.observations["state"] = Channel(
            name="state",
            dtype=DType.FLOAT32,
            shape=[8],
            data=np.random.randn(100, 8).astype(np.float32),
        )
        ep.actions["action"] = Channel(
            name="action",
            dtype=DType.FLOAT32,
            shape=[4],
            data=np.random.randn(100, 4).astype(np.float32),
        )

        # Should not raise
        ep.validate()

    def test_episode_validate_length_mismatch(self):
        """Test validation catches length mismatches."""
        ep = Episode(id="test", length=100)
        ep.observations["state"] = Channel(
            name="state",
            dtype=DType.FLOAT32,
            shape=[8],
            data=np.random.randn(50, 8).astype(np.float32),  # Wrong length
        )

        with pytest.raises(ValueError, match="length mismatch"):
            ep.validate()

    def test_episode_clone(self):
        """Test episode cloning."""
        ep = Episode(id="test", length=50)
        ep.observations["state"] = Channel(
            name="state",
            dtype=DType.FLOAT32,
            shape=[8],
            data=np.random.randn(50, 8).astype(np.float32),
        )

        clone = ep.clone()
        assert clone.id == ep.id
        assert clone.length == ep.length
        assert "state" in clone.observations
        assert clone.observations["state"].data is not ep.observations["state"].data


class TestDreamerAdapter:
    """Test DreamerV3 adapter."""

    def test_dreamer_roundtrip(self):
        """Test DreamerV3 save/load roundtrip."""
        # Create episode
        ep = Episode(id="test-ep", length=100)
        ep.observations["vector"] = Channel(
            name="vector",
            dtype=DType.FLOAT32,
            shape=[16],
            data=np.random.randn(100, 16).astype(np.float32),
        )
        ep.actions["action"] = Channel(
            name="action",
            dtype=DType.FLOAT32,
            shape=[4],
            data=np.random.randn(100, 4).astype(np.float32),
        )
        ep.rewards = Channel(
            name="reward",
            dtype=DType.FLOAT32,
            shape=[],
            data=np.random.randn(100).astype(np.float32),
        )
        term_data = np.zeros(100, dtype=np.bool_)
        term_data[-1] = True
        ep.terminations = Channel(
            name="is_terminal",
            dtype=DType.BOOL,
            shape=[],
            data=term_data,
        )

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = Path(f.name)

        try:
            # Save
            save_dreamer(ep, path)

            # Load
            loaded = load_dreamer(path)

            # Verify
            assert loaded.length == ep.length
            assert "vector" in loaded.observations
            assert np.allclose(
                loaded.observations["vector"].data, ep.observations["vector"].data
            )
            assert "action" in loaded.actions
            assert loaded.rewards is not None
        finally:
            path.unlink()

    def test_load_existing_fixture(self):
        """Test loading existing DreamerV3 fixture."""
        fixture_path = (
            Path(__file__).parent.parent.parent
            / "wshard"
            / "wea"
            / "testdata"
            / "dreamer_basic.npz"
        )

        if not fixture_path.exists():
            pytest.skip("Fixture not found")

        ep = load_dreamer(fixture_path)

        assert ep.length == 100
        assert ep.source_format == Format.DREAMER_V3
        assert "vector" in ep.observations
        assert "action" in ep.actions


class TestWShardAdapter:
    """Test W-SHARD adapter."""

    def test_wshard_roundtrip(self):
        """Test W-SHARD save/load roundtrip."""
        # Create episode
        ep = Episode(id="test-wshard", length=50)
        ep.env_id = "test-env"
        ep.observations["state"] = Channel(
            name="state",
            dtype=DType.FLOAT32,
            shape=[8],
            data=np.random.randn(50, 8).astype(np.float32),
        )
        ep.actions["ctrl"] = Channel(
            name="ctrl",
            dtype=DType.FLOAT32,
            shape=[2],
            data=np.random.randn(50, 2).astype(np.float32),
        )
        ep.rewards = Channel(
            name="reward",
            dtype=DType.FLOAT32,
            shape=[],
            data=np.random.randn(50).astype(np.float32),
        )
        term_data = np.zeros(50, dtype=np.bool_)
        term_data[-1] = True
        ep.terminations = Channel(
            name="done",
            dtype=DType.BOOL,
            shape=[],
            data=term_data,
        )

        with tempfile.NamedTemporaryFile(suffix=".wshard", delete=False) as f:
            path = Path(f.name)

        try:
            # Save
            save_wshard(ep, path)

            # Verify magic bytes
            with open(path, "rb") as f:
                magic = f.read(4)
            assert magic == b"SHRD"

            # Load
            loaded = load_wshard(path)

            # Verify
            assert loaded.id == ep.id
            assert loaded.length == ep.length
            assert loaded.source_format == Format.WSHARD
        finally:
            path.unlink()


class TestResidualEncoding:
    """Test Sign2ndDiff residual encoding."""

    def test_sign2nd_diff_peak(self):
        """Test peak detection (convex)."""
        signal = np.array([0.0, 2.0, 0.0], dtype=np.float32)
        residuals = compute_sign2nd_diff(signal)
        assert residuals[0] == 0  # Edge (padded)
        assert residuals[1] == 1  # Peak: 2*2 - 0 - 0 = 4 > 0
        assert residuals[2] == 0  # Edge (padded)

    def test_sign2nd_diff_dip(self):
        """Test dip detection (concave)."""
        signal = np.array([2.0, 0.0, 2.0], dtype=np.float32)
        residuals = compute_sign2nd_diff(signal)
        assert residuals[0] == 0  # Edge (padded)
        assert residuals[1] == -1  # Dip: 2*0 - 2 - 2 = -4 < 0
        assert residuals[2] == 0  # Edge (padded)

    def test_sign2nd_diff_linear(self):
        """Test linear detection."""
        signal = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        residuals = compute_sign2nd_diff(signal)
        assert residuals[1] == 0  # Linear: 2*1 - 0 - 2 = 0
        assert residuals[2] == 0  # Linear: 2*2 - 1 - 3 = 0

    def test_pack_unpack_roundtrip(self):
        """Test packing and unpacking residuals."""
        residuals = np.array([1, -1, 1, 1, -1, -1, 1, -1, 1, -1], dtype=np.int8)
        packed = pack_residual_bits(residuals)

        # 10 bits = 2 bytes
        assert len(packed) == 2

        # Unpack
        unpacked = unpack_residual_bits(packed, len(residuals))

        # On unpack: bit 1 -> +1, bit 0 -> -1
        for i in range(len(residuals)):
            expected = 1 if residuals[i] > 0 else -1
            assert unpacked[i] == expected

    def test_multidim_residuals(self):
        """Test multi-dimensional residual encoding."""
        T, D = 100, 4
        signal = np.random.randn(T, D).astype(np.float32)

        residuals = compute_sign2nd_diff_multidim(signal)
        assert residuals.shape == (T, D)

        packed = pack_multidim_residual_bits(residuals)
        bytes_per_dim = (T + 7) // 8
        assert len(packed) == D * bytes_per_dim

        unpacked = unpack_multidim_residual_bits(packed, T, D)
        assert unpacked.shape == (T, D)


class TestCompression:
    """Test W-SHARD compression."""

    def create_test_episode(self, T=100):
        """Create a test episode."""
        ep = Episode(id="compression-test", length=T)
        ep.env_id = "test_env"

        signals = np.array([[i * 0.5, i * 0.25] for i in range(T)], dtype=np.float32)
        ep.observations["state/pos"] = Channel(
            name="state/pos",
            dtype=DType.FLOAT32,
            shape=[2],
            data=signals,
        )

        actions = np.array([[i * 0.1, i * 0.2] for i in range(T)], dtype=np.float32)
        ep.actions["main"] = Channel(
            name="main",
            dtype=DType.FLOAT32,
            shape=[2],
            data=actions,
        )

        dones = np.zeros(T, dtype=np.uint8)
        dones[-1] = 1
        ep.terminations = Channel(
            name="done",
            dtype=DType.UINT8,
            shape=[],
            data=dones,
        )

        return ep

    @pytest.mark.parametrize(
        "compression",
        [
            CompressionType.NONE,
            CompressionType.ZSTD,
            CompressionType.LZ4,
        ],
    )
    def test_compression_roundtrip(self, compression):
        """Test write/read roundtrip with each compression type."""
        ep = self.create_test_episode()

        with tempfile.NamedTemporaryFile(suffix=".wshard", delete=False) as f:
            path = Path(f.name)

        try:
            save_wshard(ep, path, compression=compression)
            assert path.exists()

            loaded = load_wshard(path)

            assert loaded.id == ep.id
            assert loaded.length == ep.length
            assert "state/pos" in loaded.observations

            np.testing.assert_array_almost_equal(
                loaded.observations["state/pos"].data,
                ep.observations["state/pos"].data,
            )
        finally:
            path.unlink()

    def test_zstd_saves_space(self):
        """Test that zstd compression reduces file size."""
        T = 1000
        ep = Episode(id="compressible", length=T)
        ep.env_id = "test"

        # Highly compressible data
        signals = np.tile(np.array([[1.0, 2.0]], dtype=np.float32), (T, 1))
        ep.observations["state"] = Channel(
            name="state",
            dtype=DType.FLOAT32,
            shape=[2],
            data=signals,
        )
        ep.actions["main"] = Channel(
            name="main",
            dtype=DType.FLOAT32,
            shape=[2],
            data=signals.copy(),
        )
        dones = np.zeros(T, dtype=np.uint8)
        dones[-1] = 1
        ep.terminations = Channel(name="done", dtype=DType.UINT8, shape=[], data=dones)

        with tempfile.TemporaryDirectory() as tmpdir:
            path_none = Path(tmpdir) / "test_none.wshard"
            path_zstd = Path(tmpdir) / "test_zstd.wshard"

            save_wshard(ep, path_none, compression=CompressionType.NONE)
            save_wshard(ep, path_zstd, compression=CompressionType.ZSTD)

            size_none = path_none.stat().st_size
            size_zstd = path_zstd.stat().st_size

            # Zstd should be smaller for repetitive data
            assert size_zstd < size_none


class TestConvert:
    """Test format conversion."""

    def test_detect_format_by_extension(self):
        """Test format detection by extension."""
        assert detect_format("episode.wshard") == Format.WSHARD
        assert detect_format("episode.npz") == Format.DREAMER_V3
        assert detect_format("buffer.pt") == Format.TDMPC2
        assert detect_format("data.hdf5") == Format.MINARI

    def test_convert_dreamer_to_wshard(self):
        """Test converting DreamerV3 to W-SHARD."""
        # Create episode
        ep = Episode(id="convert-test", length=30)
        ep.observations["obs"] = Channel(
            name="obs",
            dtype=DType.FLOAT32,
            shape=[4],
            data=np.random.randn(30, 4).astype(np.float32),
        )
        ep.actions["action"] = Channel(
            name="action",
            dtype=DType.FLOAT32,
            shape=[2],
            data=np.random.randn(30, 2).astype(np.float32),
        )
        ep.rewards = Channel(
            name="reward",
            dtype=DType.FLOAT32,
            shape=[],
            data=np.random.randn(30).astype(np.float32),
        )
        term_data = np.zeros(30, dtype=np.bool_)
        term_data[-1] = True
        ep.terminations = Channel(
            name="is_terminal",
            dtype=DType.BOOL,
            shape=[],
            data=term_data,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = Path(tmpdir) / "episode.npz"
            wshard_path = Path(tmpdir) / "episode.wshard"

            # Save as DreamerV3
            save(ep, npz_path)

            # Convert to W-SHARD
            convert(npz_path, wshard_path)

            # Load converted
            loaded = load(wshard_path)

            assert loaded.source_format == Format.WSHARD
            assert loaded.length == ep.length


# ============================================================
# Gap 3: BITMASK Integration tests
# ============================================================


class TestBitmaskIntegration:
    """Test cowrie BITMASK integration for residual encoding."""

    def test_bitmask_pack_unpack_roundtrip(self):
        """Pack with bitmask, unpack with bitmask."""
        from wshard import pack_residual_bitmask, unpack_residual_bitmask

        residuals = np.array([1, -1, 1, 1, -1, -1, 1, -1, 1, -1], dtype=np.int8)
        packed = pack_residual_bitmask(residuals)
        unpacked = unpack_residual_bitmask(packed, len(residuals))

        for i in range(len(residuals)):
            expected = 1 if residuals[i] > 0 else -1
            assert unpacked[i] == expected

    def test_cross_compat_old_pack_new_unpack(self):
        """Pack with old function, unpack with new bitmask function (raw fallback)."""
        from wshard import (
            pack_residual_bits,
            unpack_residual_bitmask,
        )

        residuals = np.array([1, -1, 1, 1, -1, 1, -1, 1], dtype=np.int8)
        packed = pack_residual_bits(residuals)
        unpacked = unpack_residual_bitmask(packed, len(residuals))

        for i in range(len(residuals)):
            expected = 1 if residuals[i] > 0 else -1
            assert unpacked[i] == expected

    def test_cross_compat_new_pack_old_unpack(self):
        """Pack with bitmask, unpack with old function (if cowrie not installed, same bytes)."""
        from wshard import (
            pack_residual_bitmask,
            unpack_residual_bits,
            HAS_COWRIE,
        )

        residuals = np.array([1, -1, 1, 1, -1, -1, 1, -1], dtype=np.int8)
        packed = pack_residual_bitmask(residuals)

        if not HAS_COWRIE:
            # Without cowrie, bitmask falls back to raw packing
            unpacked = unpack_residual_bits(packed, len(residuals))
            for i in range(len(residuals)):
                expected = 1 if residuals[i] > 0 else -1
                assert unpacked[i] == expected

    @pytest.mark.parametrize("T", [0, 1, 7, 8, 9, 100])
    def test_bitmask_edge_cases(self, T):
        """Test bitmask at various sizes including byte boundaries."""
        from wshard import pack_residual_bitmask, unpack_residual_bitmask

        if T == 0:
            residuals = np.array([], dtype=np.int8)
        else:
            residuals = np.random.choice([-1, 1], size=T).astype(np.int8)

        packed = pack_residual_bitmask(residuals)
        unpacked = unpack_residual_bitmask(packed, T)
        assert len(unpacked) == T

        for i in range(T):
            expected = 1 if residuals[i] > 0 else -1
            assert unpacked[i] == expected

    def test_multidim_bitmask_roundtrip(self):
        """Test multi-dimensional bitmask packing."""
        from wshard import (
            pack_multidim_residual_bitmask,
            unpack_multidim_residual_bitmask,
        )

        T, D = 50, 4
        residuals = np.random.choice([-1, 1], size=(T, D)).astype(np.int8)
        packed = pack_multidim_residual_bitmask(residuals)
        unpacked = unpack_multidim_residual_bitmask(packed, T, D)

        np.testing.assert_array_equal(unpacked, residuals)

    def test_wshard_roundtrip_with_residual_encoding_field(self):
        """Verify residual_encoding field in meta/wshard on roundtrip."""
        ep = Episode(id="bitmask-test", length=20)
        ep.observations["x"] = Channel(
            name="x", dtype=DType.FLOAT32, shape=[],
            data=np.sin(np.arange(20, dtype=np.float32) * 0.3),
        )
        ep.actions["a"] = Channel(
            name="a", dtype=DType.FLOAT32, shape=[],
            data=np.zeros(20, dtype=np.float32),
        )
        ep.terminations = Channel(
            name="done", dtype=DType.BOOL, shape=[],
            data=np.array([False] * 19 + [True]),
        )

        with tempfile.NamedTemporaryFile(suffix=".wshard", delete=False) as f:
            path = Path(f.name)

        try:
            save_wshard(ep, path)
            loaded = load_wshard(path)
            assert loaded.length == 20

            # Read raw meta/wshard to verify encoding field
            import json
            with open(path, "rb") as f:
                data = f.read()
            # Find meta/wshard block
            assert b'"residual_encoding"' in data
        finally:
            path.unlink()


# ============================================================
# Gap 5: VLA Multi-Modal tests
# ============================================================


class TestMultiModal:
    """Test VLA multi-modal observation support."""

    def test_modality_enum(self):
        """Test Modality enum values."""
        from wshard import Modality

        assert Modality.RGB.value == "rgb"
        assert Modality.LANGUAGE.value == "language"
        assert Modality.RGB.content_type == 0x0006  # IMAGE
        assert Modality.LANGUAGE.content_type == 0x0005  # TEXT

    def test_add_multimodal_observation(self):
        """Test adding multi-modal observations."""
        from wshard import Modality, add_multimodal_observation

        ep = Episode(id="mm-test", length=10)

        rgb = Channel(name="rgb", dtype=DType.FLOAT32, shape=[3, 64, 64],
                      data=np.random.randn(10, 3, 64, 64).astype(np.float32))
        lang = Channel(name="language", dtype=DType.FLOAT32, shape=[768],
                       data=np.random.randn(10, 768).astype(np.float32))
        proprio = Channel(name="proprio", dtype=DType.FLOAT32, shape=[7],
                          data=np.random.randn(10, 7).astype(np.float32))

        add_multimodal_observation(ep, "obs", Modality.RGB, rgb)
        add_multimodal_observation(ep, "obs", Modality.LANGUAGE, lang)
        add_multimodal_observation(ep, "obs", Modality.PROPRIOCEPTION, proprio)

        assert "obs/rgb" in ep.observations
        assert "obs/language" in ep.observations
        assert "obs/proprioception" in ep.observations
        assert ep.observations["obs/rgb"].modality == Modality.RGB

    def test_get_multimodal_observations(self):
        """Test filtering multi-modal observations."""
        from wshard import Modality, add_multimodal_observation, get_multimodal_observations

        ep = Episode(id="mm-test", length=5)

        rgb = Channel(name="rgb", dtype=DType.FLOAT32, shape=[3],
                      data=np.random.randn(5, 3).astype(np.float32))
        lang = Channel(name="lang", dtype=DType.FLOAT32, shape=[10],
                       data=np.random.randn(5, 10).astype(np.float32))

        add_multimodal_observation(ep, "obs", Modality.RGB, rgb)
        add_multimodal_observation(ep, "goal", Modality.LANGUAGE, lang)

        # Filter by group
        obs_chs = get_multimodal_observations(ep, group="obs")
        assert len(obs_chs) == 1
        assert "obs/rgb" in obs_chs

        # Filter by modality
        lang_chs = get_multimodal_observations(ep, modality=Modality.LANGUAGE)
        assert len(lang_chs) == 1

    def test_multimodal_wshard_roundtrip(self):
        """Test writing and reading multi-modal episode."""
        from wshard import Modality, add_multimodal_observation

        ep = Episode(id="mm-roundtrip", length=5)
        ep.env_id = "VLAEnv-v0"

        rgb = Channel(name="rgb", dtype=DType.FLOAT32, shape=[3],
                      data=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9],
                                     [10, 11, 12], [13, 14, 15]], dtype=np.float32))
        add_multimodal_observation(ep, "obs", Modality.RGB, rgb)

        ep.actions["ctrl"] = Channel(
            name="ctrl", dtype=DType.FLOAT32, shape=[2],
            data=np.random.randn(5, 2).astype(np.float32),
        )
        ep.terminations = Channel(
            name="done", dtype=DType.BOOL, shape=[],
            data=np.array([False, False, False, False, True]),
        )

        with tempfile.NamedTemporaryFile(suffix=".wshard", delete=False) as f:
            path = Path(f.name)
        try:
            save_wshard(ep, path)
            loaded = load_wshard(path)
            assert "obs/rgb" in loaded.observations
            assert loaded.observations["obs/rgb"].modality == Modality.RGB
            np.testing.assert_array_almost_equal(
                loaded.observations["obs/rgb"].data, rgb.data,
            )
        finally:
            path.unlink()


# ============================================================
# Gap 2: Latent Action Storage tests
# ============================================================


class TestLatentActions:
    """Test Genie 3-style latent action storage."""

    def test_set_get_latent_actions(self):
        """Test setting and getting latent action embeddings."""
        from wshard import set_latent_actions, get_latent_actions, get_latent_codebook

        ep = Episode(id="latent-test", length=10)

        embeddings = Channel(
            name="embeddings", dtype=DType.FLOAT32, shape=[16],
            data=np.random.randn(10, 16).astype(np.float32),
        )
        codebook = Channel(
            name="codebook", dtype=DType.INT32, shape=[],
            data=np.random.randint(0, 256, size=10, dtype=np.int32),
        )

        set_latent_actions(ep, "genie3_v1", embeddings, codebook)

        # Retrieve
        got_embed = get_latent_actions(ep, "genie3_v1")
        assert got_embed is not None
        np.testing.assert_array_equal(got_embed.data, embeddings.data)

        got_cb = get_latent_codebook(ep, "genie3_v1")
        assert got_cb is not None
        np.testing.assert_array_equal(got_cb.data, codebook.data)

    def test_latent_actions_without_codebook(self):
        """Test latent actions without VQ-VAE codebook."""
        from wshard import set_latent_actions, get_latent_actions, get_latent_codebook

        ep = Episode(id="latent-no-cb", length=5)

        embeddings = Channel(
            name="embeddings", dtype=DType.FLOAT32, shape=[8],
            data=np.random.randn(5, 8).astype(np.float32),
        )

        set_latent_actions(ep, "model_v2", embeddings)

        assert get_latent_actions(ep, "model_v2") is not None
        assert get_latent_codebook(ep, "model_v2") is None

    def test_latent_actions_missing_model(self):
        """Test querying non-existent model."""
        from wshard import get_latent_actions

        ep = Episode(id="empty", length=5)
        assert get_latent_actions(ep, "nonexistent") is None


# ============================================================
# Gap 1: Chunked Episodes tests
# ============================================================


class TestChunkedEpisodes:
    """Test chunked episode support."""

    def test_chunk_fields_on_episode(self):
        """Test chunk fields on Episode dataclass."""
        ep = Episode(id="chunk-test", length=100,
                     chunk_index=0, total_chunks=5,
                     timestep_range=[0, 99])
        assert ep.is_chunked
        ep.validate()

    def test_unchunked_episode(self):
        """Test that unchunked episodes work normally."""
        ep = Episode(id="normal", length=50)
        assert not ep.is_chunked
        ep.observations["s"] = Channel(
            name="s", dtype=DType.FLOAT32, shape=[],
            data=np.zeros(50, dtype=np.float32),
        )
        ep.validate()

    def test_chunk_validation_errors(self):
        """Test chunk field validation."""
        # chunk_index without total_chunks
        ep = Episode(id="bad", length=10, chunk_index=0)
        with pytest.raises(ValueError, match="total_chunks required"):
            ep.validate()

        # chunk_index out of range
        ep = Episode(id="bad", length=10, chunk_index=5, total_chunks=3)
        with pytest.raises(ValueError, match="out of range"):
            ep.validate()

    def test_chunked_wshard_roundtrip(self):
        """Test chunk fields survive save/load."""
        ep = Episode(id="chunk-rt", length=20,
                     chunk_index=2, total_chunks=10,
                     timestep_range=[40, 59])
        ep.observations["s"] = Channel(
            name="s", dtype=DType.FLOAT32, shape=[],
            data=np.arange(20, dtype=np.float32),
        )
        ep.actions["a"] = Channel(
            name="a", dtype=DType.FLOAT32, shape=[],
            data=np.zeros(20, dtype=np.float32),
        )
        ep.terminations = Channel(
            name="done", dtype=DType.BOOL, shape=[],
            data=np.array([False] * 19 + [True]),
        )

        with tempfile.NamedTemporaryFile(suffix=".wshard", delete=False) as f:
            path = Path(f.name)
        try:
            save_wshard(ep, path)
            loaded = load_wshard(path)
            assert loaded.chunk_index == 2
            assert loaded.total_chunks == 10
            assert loaded.timestep_range == [40, 59]
        finally:
            path.unlink()

    def test_chunked_episode_writer(self):
        """Test ChunkedEpisodeWriter splits and writes correctly."""
        from wshard import ChunkedEpisodeWriter, ChunkedEpisodeReader

        T = 100
        ep = Episode(id="big-ep", length=T, env_id="Test-v0")
        ep.observations["s"] = Channel(
            name="s", dtype=DType.FLOAT32, shape=[],
            data=np.arange(T, dtype=np.float32),
        )
        ep.actions["a"] = Channel(
            name="a", dtype=DType.FLOAT32, shape=[],
            data=np.zeros(T, dtype=np.float32),
        )
        ep.rewards = Channel(
            name="reward", dtype=DType.FLOAT32, shape=[],
            data=np.ones(T, dtype=np.float32),
        )
        ep.terminations = Channel(
            name="done", dtype=DType.BOOL, shape=[],
            data=np.array([False] * (T - 1) + [True]),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ChunkedEpisodeWriter(tmpdir, "big-ep", chunk_size_t=30)
            paths = writer.write_episode_chunked(ep)
            manifest_path = writer.finalize_manifest()

            assert len(paths) == 4  # 100 / 30 = 3.33 -> 4 chunks
            assert manifest_path.exists()

            # Read back with ChunkedEpisodeReader
            reader = ChunkedEpisodeReader(str(manifest_path))
            assert reader.num_chunks == 4

            all_obs = []
            for chunk_ep in reader.iter_chunks():
                all_obs.extend(chunk_ep.observations["s"].data.tolist())

            np.testing.assert_array_almost_equal(
                all_obs, np.arange(T, dtype=np.float32),
            )

    def test_single_chunk_backward_compat(self):
        """Test single-chunk episode is equivalent to unchunked."""
        ep = Episode(id="single", length=10, env_id="Env-v0")
        ep.observations["s"] = Channel(
            name="s", dtype=DType.FLOAT32, shape=[],
            data=np.arange(10, dtype=np.float32),
        )
        ep.actions["a"] = Channel(
            name="a", dtype=DType.FLOAT32, shape=[],
            data=np.zeros(10, dtype=np.float32),
        )
        ep.terminations = Channel(
            name="done", dtype=DType.BOOL, shape=[],
            data=np.array([False] * 9 + [True]),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save unchunked
            path = Path(tmpdir) / "single.wshard"
            save_wshard(ep, path)

            loaded = load_wshard(path)
            assert loaded.chunk_index is None
            assert not loaded.is_chunked


# ============================================================
# Gap 4: Streaming Append tests
# ============================================================


class TestStreamingAppend:
    """Test streaming append-only episode writer."""

    def test_basic_streaming(self):
        """Test basic streaming write and standard read."""
        from wshard import WShardStreamWriter, StreamChannelDef

        with tempfile.NamedTemporaryFile(suffix=".wshard", delete=False) as f:
            path = Path(f.name)

        try:
            channel_defs = [
                StreamChannelDef("state", DType.FLOAT32, [4]),
            ]

            writer = WShardStreamWriter(path, "stream-test", channel_defs)
            writer.begin_episode(env_id="StreamEnv-v0")

            T = 100
            for t in range(T):
                obs = np.random.randn(4).astype(np.float32)
                act = np.random.randn(4).astype(np.float32)
                writer.write_timestep(
                    t=t,
                    observations={"state": obs},
                    actions={"state": act},
                    reward=float(t) * 0.1,
                    done=(t == T - 1),
                )

            total = writer.end_episode()
            assert total > 0
            assert writer.is_finalized

            # Read back with standard reader
            loaded = load_wshard(path)
            assert loaded.length == T
            assert loaded.id == "stream-test"
            assert "state" in loaded.observations
            assert loaded.observations["state"].data.shape[0] == T
        finally:
            path.unlink()

    def test_streaming_context_manager(self):
        """Test streaming with context manager."""
        from wshard import WShardStreamWriter, StreamChannelDef

        with tempfile.NamedTemporaryFile(suffix=".wshard", delete=False) as f:
            path = Path(f.name)

        try:
            channel_defs = [StreamChannelDef("x", DType.FLOAT32, [])]

            with WShardStreamWriter(path, "ctx-test", channel_defs) as writer:
                writer.begin_episode()
                for t in range(10):
                    writer.write_timestep(
                        t=t,
                        observations={"x": np.float32(t * 0.5)},
                        actions={"x": np.float32(0.0)},
                        reward=0.0,
                        done=(t == 9),
                    )

            loaded = load_wshard(path)
            assert loaded.length == 10
        finally:
            path.unlink()

    def test_streaming_shape_validation(self):
        """Test that shape mismatches are caught during streaming."""
        from wshard import WShardStreamWriter, StreamChannelDef

        with tempfile.NamedTemporaryFile(suffix=".wshard", delete=False) as f:
            path = Path(f.name)

        try:
            channel_defs = [StreamChannelDef("state", DType.FLOAT32, [3])]
            writer = WShardStreamWriter(path, "shape-test", channel_defs)
            writer.begin_episode()

            with pytest.raises(ValueError, match="Shape mismatch"):
                writer.write_timestep(
                    t=0,
                    observations={"state": np.array([1.0, 2.0], dtype=np.float32)},
                    actions={"state": np.array([0.0, 0.0, 0.0], dtype=np.float32)},
                    reward=0.0,
                    done=False,
                )
        finally:
            if writer._file:
                writer._file.close()
            path.unlink(missing_ok=True)

    def test_streaming_without_begin(self):
        """Test error when writing without begin_episode."""
        from wshard import WShardStreamWriter, StreamChannelDef

        with tempfile.NamedTemporaryFile(suffix=".wshard", delete=False) as f:
            path = Path(f.name)

        try:
            channel_defs = [StreamChannelDef("x", DType.FLOAT32, [])]
            writer = WShardStreamWriter(path, "no-begin", channel_defs)

            with pytest.raises(RuntimeError, match="begin_episode"):
                writer.write_timestep(
                    t=0,
                    observations={"x": np.float32(1.0)},
                    actions={"x": np.float32(0.0)},
                    reward=0.0,
                    done=False,
                )
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
