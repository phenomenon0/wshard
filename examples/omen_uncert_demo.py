"""First Python end-to-end demo of all four lanes: signal, omen, uncert, residual.

`examples/residual_demo.py` covers residuals alone, and the only end-to-end
omen+uncert writer was on the Go side (`golden/generate.go`). This mirrors that
structure in Python.

The shape it demonstrates matters more than the toy model. A block index entry
is one `(offset, size)` extent and `meta/identity` commits to every block, so a
prediction cannot be appended to a sealed episode. It forks a *derived* episode
whose `provenance.prev_identity` names the exact source bytes -- which means a
model's cached output is a first-class artifact you can hand to someone else and
they can prove what it was computed from.

Run:  python examples/omen_uncert_demo.py
"""
import tempfile
from pathlib import Path

import numpy as np

from wshard import (
    Channel, DType, Episode, Provenance, Residual,
    compute_error_residual, episode_identity, episode_provenance,
    load_wshard, pack_multidim_residual_bitmask, save_wshard,
    unpack_multidim_residual_bitmask,
)

T, D, MODEL = 64, 4, "toy_v1"
tmp = Path(tempfile.mkdtemp())

# --- the source episode: a sealed recording -------------------------------
t = np.linspace(0, 4 * np.pi, T, dtype=np.float32)
truth = np.stack([np.sin(t + i) for i in range(D)], axis=1)
src = Episode(id="wave", length=T)
src.observations["joint_pos"] = Channel(name="joint_pos", dtype=DType.FLOAT32,
                                        shape=[D], data=truth)
src_path = tmp / "source.wshard"
save_wshard(src, src_path)
src_id = episode_identity(str(src_path))

# --- a model predicts it, badly, and says how unsure it is ----------------
rng = np.random.default_rng(0)
error = 0.05 * rng.standard_normal((T, D)).astype(np.float32)
pred = truth + error
# a real model emits this; here it is just the error scale it "expects"
var = np.full((T, D), 0.05 ** 2, dtype=np.float32)

# --- the derived episode: predictions chained to the source bytes ---------
d = Episode(id=f"wave:{MODEL}", length=T)
d.observations["joint_pos"] = Channel(name="joint_pos", dtype=DType.FLOAT32,
                                      shape=[D], data=truth)
d.omens["joint_pos"] = {MODEL: Channel(name="omen", dtype=DType.FLOAT32,
                                       shape=[D], data=pred)}
d.uncerts["joint_pos"] = {MODEL: {"variance": Channel(
    name="variance", dtype=DType.FLOAT32, shape=[D], data=var)}}
sgn = compute_error_residual(truth, pred)
d.residuals["joint_pos"] = Residual(channel_id="joint_pos", type="sign2nddiff",
                                    data=pack_multidim_residual_bitmask(sgn))
d.provenance = Provenance(run_id="demo", prev_identity=src_id,
                          source={"model": MODEL})
out = tmp / "derived.wshard"
save_wshard(d, out)

# --- read it back, selecting one channel ----------------------------------
r = load_wshard(str(out), channels=["joint_pos"])
back = unpack_multidim_residual_bitmask(r.residuals["joint_pos"].data, T, D)

print(f"source identity   : {src_id[:16]}...")
print(f"derived identity  : {episode_identity(str(out))[:16]}...  (a different artifact)")
print(f"prev_identity     : {episode_provenance(str(out)).prev_identity[:16]}...")
print(f"link valid        : {episode_provenance(str(out)).prev_identity == src_id}")
print(f"source untouched  : {episode_identity(str(src_path)) == src_id}")
print()
print(f"omen   : {r.omens['joint_pos'][MODEL].data.shape} f32")
print(f"uncert : {list(r.uncerts['joint_pos'][MODEL])} -> "
      f"{r.uncerts['joint_pos'][MODEL]['variance'].data.shape} f32")
# sign2nddiff is 1 bit, so the 0s at the padded edges (no neighbour on one side)
# decode as -1. Interior signs are exact.
print(f"residual: {len(r.residuals['joint_pos'].data)} bytes for {T*D} elements "
      f"({8*len(r.residuals['joint_pos'].data)/(T*D):.1f} bit/elem), "
      f"interior exact: {np.array_equal(back[1:-1], sgn[1:-1])}")
print()
print(f"the residual ranks error *roughness*, not magnitude: flip rate "
      f"{(back[1:] != back[:-1]).mean():.3f}")
