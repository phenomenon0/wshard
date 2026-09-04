"""Every ```python block in the live docs is executed.

Documentation drifts silently: a renamed argument, a changed return type or an
API that was never real all read fine. The sweep that produced this file found
four such bugs by running the snippets -- a missing ``shape=[]``, a per-block
compression API that did not exist, a wrong streaming signature, and a
``finalize_manifest()`` return type documented as ``str`` when it is a ``Path``.
Reading cannot catch those; only running can.

Each snippet is its own test so a failure names the document and the block.
Snippets that reference something outside this repo (h5py, gym, LeRobot) or are
deliberately partial are skipped by SKIP below.
"""
import pathlib
import re
import subprocess
import sys
import textwrap

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]

DOCS = [
    "README.md", "py/README.md", "docs/FORMAT.md", "docs/DEEP_DIVE.md",
    "docs/FAQ.md", "docs/WHY_NOT_HDF5.md", "docs/WHY_NOT_LEROBOT.md",
    "SECURITY.md", "demo_identity/README.md",
]

# Snippets that cannot run here: they need a package this repo does not depend
# on, or they are an illustrative fragment rather than a script.
SKIP = re.compile(
    r"h5py\.|import gym|lerobot|LeRobotDataset|minari|d4rl|my_embedder"
    r"|dreamer_episode\.npz|your_env|# pseudo|\.\.\."
)

# Bindings a snippet may assume from its surrounding prose. Kept deliberately
# close to what the docs describe rather than minimal, so a snippet fails when
# the API changed and not when this prelude is thin.
PRELUDE = '''
import numpy as np
from wshard.types import Episode, Channel, DType
from wshard import save_wshard, verify_identity

T = 100

def _ep(id="ep_001", n=T):
    e = Episode(id=id, length=n)
    e.env_id = "ManipulationEnv-v2"
    e.observations["joint_pos"] = Channel(name="joint_pos", dtype=DType.FLOAT32,
        shape=[7], data=np.random.randn(n, 7).astype(np.float32))
    e.observations["state"] = Channel(name="state", dtype=DType.FLOAT32,
        shape=[4], data=np.random.randn(n, 4).astype(np.float32))
    e.actions["ctrl"] = Channel(name="ctrl", dtype=DType.FLOAT32,
        shape=[7], data=np.random.randn(n, 7).astype(np.float32))
    e.rewards = Channel(name="reward", dtype=DType.FLOAT32, shape=[],
        data=np.ones(n, dtype=np.float32))
    return e

ep = _ep()
long_episode = _ep("long_ep", 3000)

# Snippets that read before they write are two-part illustrations, not scripts.
save_wshard(_ep(), "episode.wshard")
path = "episode.wshard"
known_64_hex = verify_identity(path)   # stands in for a value obtained out of band

observations = np.random.randn(T, 4).astype(np.float32)
actions = np.zeros((T, 1), dtype=np.int32)
obs = np.random.randn(4).astype(np.float32)   # one timestep
act = np.zeros(7, dtype=np.float32)
reward = 1.0
done = False
encoded_h264 = bytes([0, 0, 1, 9])   # stands in for a real encode
signal = np.random.randn(T).astype(np.float32)
rgb_channel = Channel(name="rgb", dtype=DType.UINT8, shape=[8, 8, 3],
    data=np.zeros((T, 8, 8, 3), dtype=np.uint8))
depth_channel = Channel(name="depth", dtype=DType.UINT16, shape=[8, 8],
    data=np.zeros((T, 8, 8), dtype=np.uint16))
joint_channel = Channel(name="joint", dtype=DType.FLOAT32, shape=[7],
    data=np.zeros((T, 7), dtype=np.float32))

class _Env:
    def step(self, a):
        return np.random.randn(7).astype(np.float32), 0.0, False

env = _Env()
action = np.zeros(7, dtype=np.float32)
'''


def _snippets():
    for doc in DOCS:
        p = ROOT / doc
        if not p.exists():
            continue
        # Fences inside a list item are indented; dedent so they compile.
        blocks = [textwrap.dedent(b) for b in
                  re.findall(r"```python\n(.*?)^[ \t]*```", p.read_text(), re.S | re.M)]
        for i, code in enumerate(blocks):
            yield pytest.param(doc, code, id=f"{doc}#{i}")


@pytest.mark.parametrize("doc,code", list(_snippets()))
def test_doc_snippet_runs(doc, code, tmp_path):
    if SKIP.search(code):
        pytest.skip("references something outside this repo, or is a fragment")
    f = tmp_path / "snippet.py"
    f.write_text(PRELUDE + "\n" + code)
    r = subprocess.run([sys.executable, str(f)], cwd=tmp_path,
                       capture_output=True, text=True)
    assert r.returncode == 0, f"{doc} python block failed:\n{r.stderr[-2000:]}"
