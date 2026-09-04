"""The LeRobot bridge is only correct if it is *invisible*.

A converted dataset read back through ``LeRobotDataset`` must hand a policy
byte-identical items to the stock parquet/mp4 path -- same keys, dtypes,
shapes, values and padding masks. Anything less and a training result on
``.wshard`` cannot be compared against the same run on the native format,
because a difference in eval score would have two possible causes.

The conversion is deliberately run **unfiltered**. The first version of the
converter indexed the source dataset with an offset of 0 for every episode
when no ``episodes=`` filter was given, so all 206 files held episode 0's
frames. Every one of them still passed ``verify_identity`` -- identity is a
fingerprint of the bytes written, not a claim that they are the right bytes.
Only comparing against the source caught it.

Needs ``lerobot`` and a locally cached ``lerobot/pusht``; skipped otherwise.
"""
import pytest

pytest.importorskip("lerobot")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from wshard.lerobot import block_name, channel_id, convert_dataset, register  # noqa: E402

REPO_ID = "lerobot/pusht"
# Diffusion Policy's real windowing: the case that exercises boundary clamping
# and the *_is_pad masks, which a single-frame comparison cannot reach.
DELTA_TS = {
    "observation.image": [-0.1, 0.0],
    "observation.state": [-0.1, 0.0],
    "action": [i / 10 for i in range(-1, 15)],
}


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    try:
        LeRobotDatasetMetadata(REPO_ID)
    except Exception as exc:  # no network, no cache -- not a bridge failure
        pytest.skip(f"{REPO_ID} unavailable: {exc}")
    root = tmp_path_factory.mktemp("pusht_wshard")
    convert_dataset(REPO_ID, root)
    register()
    return root


def test_block_naming_round_trips():
    for key, name, cid in [
        ("observation.image", "signal/image", "image"),
        ("observation.state", "signal/state", "state"),
        ("action", "action/ctrl", "ctrl"),
        ("next.reward", "reward", "reward"),
        ("next.done", "done", "done"),
        ("next.success", "signal/success", "success"),
    ]:
        assert block_name(key) == name
        assert channel_id(key) == cid


def test_identity_holds_on_every_converted_episode(converted):
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from wshard import verify_identity

    meta = LeRobotDatasetMetadata(REPO_ID)
    files = sorted((converted / "wshard").glob("*.wshard"))
    assert len(files) == meta.total_episodes
    for f in files:
        assert len(verify_identity(str(f))) == 64


def test_identity_survives_recompression(converted):
    """The claim that makes a converted corpus citable: the identity names the
    *content*, so re-encoding an episode to save space does not rename it.

    Checked on a real converted episode rather than a synthetic one, because the
    converter is what decides which bytes are hashed. NONE vs ZSTD is a ~22x
    size change here; if identity moved with the codec, every downstream
    reference to an episode would break the moment it was recompressed.
    """
    from wshard import load_wshard, save_wshard, verify_identity
    from wshard.compress import CompressionType

    src = sorted((converted / "wshard").glob("*.wshard"))[42]
    base = verify_identity(str(src))
    ep = load_wshard(str(src))
    for codec in (CompressionType.NONE, CompressionType.LZ4, CompressionType.ZSTD):
        out = src.parent.parent / f"recompressed_{codec.name}.wshard"
        save_wshard(ep, out, compression=codec)
        assert verify_identity(str(out)) == base, f"identity moved under {codec.name}"
        out.unlink()


def test_each_file_holds_its_own_episode(converted):
    """Regression: identity cannot tell you the file holds the *right* frames."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from wshard import load_wshard

    ds = LeRobotDataset(REPO_ID, return_uint8=True)
    for ep_idx in (0, 1, 2, ds.meta.total_episodes // 2, ds.meta.total_episodes - 1):
        row = ds.meta.episodes[ep_idx]
        ep = load_wshard(str(converted / "wshard" / f"episode_{ep_idx:06d}.wshard"))
        assert ep.length == row["dataset_to_index"] - row["dataset_from_index"]
        src = ds[row["dataset_from_index"]]
        assert np.array_equal(ep.observations["state"].data[0], src["observation.state"].numpy())
        assert np.array_equal(
            np.asarray(ep.observations["image"].data)[0],
            src["observation.image"].permute(1, 2, 0).numpy(),
        )


def test_items_match_the_stock_reader(converted):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ref = LeRobotDataset(REPO_ID, delta_timestamps=DELTA_TS)
    got = LeRobotDataset(REPO_ID, root=converted, delta_timestamps=DELTA_TS)
    assert len(ref) == len(got)

    boundary = ref.meta.episodes[0]["dataset_to_index"]
    rng = np.random.default_rng(0)
    probes = {0, 1, boundary - 1, boundary, boundary + 1, len(ref) - 1}
    probes |= set(rng.integers(0, len(ref), 40).tolist())

    for i in sorted(probes):
        a, b = ref[i], got[i]
        assert a.keys() == b.keys(), i
        for key in a:
            if key == "task":
                assert a[key] == b[key], (i, key)
                continue
            assert a[key].dtype == b[key].dtype, (i, key)
            assert a[key].shape == b[key].shape, (i, key)
            assert torch.equal(a[key], b[key]), (i, key)
