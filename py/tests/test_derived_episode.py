"""Predictions cannot be appended to a sealed episode -- they fork a new one.

A block's index entry is a single ``(offset, size)`` extent and ``meta/identity``
commits to every block, so writing an ``omen/`` into an existing file means
rewriting the file under a different identity. The supported shape is a *derived*
episode: a new sealed file whose ``provenance.prev_identity`` names the exact
source bytes it was computed from.

This is the pattern Phase 3 used to cache a world model's predictions over a
206-episode corpus. It is exercised here because two bugs made it silently
lossy before it worked: ``Episode`` had no ``uncerts`` field at all (Go had
``Uncerts`` since the format shipped), so an uncert block round-tripped to
nothing while still passing ``verify_identity``; and the ``channels=`` filter
compared the whole post-lane remainder against the allowed set, so a filtered
read returned ``signal`` and ``residual`` but silently dropped ``omen`` and
``uncert``. Both are invisible unless a test reads all four lanes back through
a filter.
"""
import numpy as np

from wshard import (
    Channel,
    DType,
    Episode,
    Provenance,
    Residual,
    compute_error_residual,
    episode_identity,
    episode_provenance,
    load_wshard,
    pack_multidim_residual_bitmask,
    save_wshard,
    unpack_multidim_residual_bitmask,
)

T, D = 40, 8
MODEL = "dyn_v1"


def _source(path):
    ep = Episode(id="src", length=T)
    ep.observations["state"] = Channel(
        name="state", dtype=DType.FLOAT32, shape=[D],
        data=np.linspace(0, 1, T * D, dtype=np.float32).reshape(T, D),
    )
    save_wshard(ep, path)
    return ep


def test_derived_episode_carries_all_four_lanes(tmp_path):
    src_path = tmp_path / "src.wshard"
    src = _source(src_path)
    actual = src.observations["state"].data
    pred = actual + 0.01 * np.sin(np.arange(T * D, dtype=np.float32)).reshape(T, D)
    var = np.abs(pred) + 1e-3

    d = Episode(id=f"src:{MODEL}", length=T)
    d.observations["latent"] = Channel(name="latent", dtype=DType.FLOAT32,
                                       shape=[D], data=actual)
    d.omens["latent"] = {MODEL: Channel(name="omen", dtype=DType.FLOAT32,
                                        shape=[D], data=pred)}
    d.uncerts["latent"] = {MODEL: {"variance": Channel(
        name="var", dtype=DType.FLOAT32, shape=[D], data=var)}}
    sgn = compute_error_residual(actual, pred)
    d.residuals["latent"] = Residual(channel_id="latent", type="sign2nddiff",
                                     data=pack_multidim_residual_bitmask(sgn))
    d.provenance = Provenance(run_id="t", prev_identity=episode_identity(str(src_path)),
                              source={"model": MODEL})
    out = tmp_path / "derived.wshard"
    save_wshard(d, out)

    # Read back through a channel filter -- the path that dropped omen/uncert.
    r = load_wshard(str(out), channels=["latent"])
    assert np.array_equal(r.observations["latent"].data, actual)
    assert np.array_equal(r.omens["latent"][MODEL].data, pred)
    assert np.array_equal(r.uncerts["latent"][MODEL]["variance"].data, var)

    # The 1-bit lane stores 2 states, so the 0s that compute_error_residual
    # emits at the padded edges (t=0, t=T-1, which have no neighbour on one
    # side) come back as -1. Everything interior must be exact.
    back = unpack_multidim_residual_bitmask(r.residuals["latent"].data, T, D)
    assert np.array_equal(back[1:-1], sgn[1:-1])
    assert set(np.unique(back)) <= {-1, 1}


def test_derived_names_the_source_bytes_and_leaves_them_alone(tmp_path):
    """prev_identity must match the source, and deriving must not touch it."""
    src_path = tmp_path / "src.wshard"
    _source(src_path)
    before = episode_identity(str(src_path))

    d = Episode(id="src:m", length=T)
    d.observations["latent"] = Channel(name="latent", dtype=DType.FLOAT32, shape=[1],
                                       data=np.zeros((T, 1), dtype=np.float32))
    d.provenance = Provenance(run_id="t", prev_identity=before)
    out = tmp_path / "derived.wshard"
    save_wshard(d, out)

    assert episode_provenance(str(out)).prev_identity == before
    assert episode_identity(str(src_path)) == before, "deriving mutated the source"
    # A derived episode is its own artifact, not a copy of the source.
    assert episode_identity(str(out)) != before
