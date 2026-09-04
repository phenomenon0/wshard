"""The property W-SHARD's case rests on: one episode, one identity.

A sealed ``meta/identity`` is only worth something if two parties who never
spoke -- a Go writer and a Python reader, a batch job and a streaming job, an
implementation written next year from FORMAT.md alone -- compute the same 64
hex from the same episode. If it instead fingerprints one writer's key order or
one interpreter's ``json.dumps``, the file records who wrote it, not what is in
it, and every claim built on it is a claim about a single implementation.

These tests exist because that property was false and nothing here noticed:
the suite verified that Go and Python *hash a given file* the same way, never
that they *write a given episode* the same way. Everything below writes one
logical episode two ways and compares.
"""

import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pytest

from wshard import (
    Channel,
    DType,
    Episode,
    StreamChannelDef,
    WShardStreamWriter,
    episode_identity,
    load_wshard,
    save_wshard,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GOLDEN_DIR = _REPO_ROOT / "golden"
_GO_MODULE_DIR = _REPO_ROOT / "go"

T = 8


def _fixture_episode() -> Episode:
    """The one logical episode every writer in this file is handed.

    Two observation channels, inserted in the order the sort does *not* produce,
    and an action whose name sorts between them: that pins both that
    meta/channels is sorted and that it is sorted within groups (signal, then
    action) rather than globally.
    """
    ep = Episode(id="parity-ep", length=T, env_id="ParityEnv-v0")
    ep.observations["zulu"] = Channel(
        name="zulu",
        dtype=DType.FLOAT32,
        shape=[2],
        data=np.repeat(np.arange(T, dtype=np.float32), 2).reshape(T, 2),
    )
    ep.observations["alpha"] = Channel(
        name="alpha",
        dtype=DType.FLOAT32,
        shape=[2],
        # +100, not -arange: at t=0 that would be -0.0, whose bytes differ from
        # the 0.0 a writer handed a plain int would produce. Real difference,
        # wrong thing for this test to be about.
        data=np.repeat(np.arange(T, dtype=np.float32) + 100.0, 2).reshape(T, 2),
    )
    ep.actions["ctrl"] = Channel(
        name="ctrl",
        dtype=DType.FLOAT32,
        shape=[2],
        data=np.repeat(np.arange(T, dtype=np.float32) + 200.0, 2).reshape(T, 2),
    )
    ep.rewards = Channel(
        name="reward", dtype=DType.FLOAT32, shape=[], data=np.arange(T, dtype=np.float32)
    )
    ep.terminations = Channel(
        name="done",
        dtype=DType.BOOL,
        shape=[],
        data=np.array([False] * (T - 1) + [True]),
    )
    return ep


def _require_go():
    if shutil.which("go") is None:
        pytest.skip("go toolchain not available")
    if not _GO_MODULE_DIR.exists():
        pytest.skip(f"go module not present at {_GO_MODULE_DIR}")


def _run_go(probe: Path, body: str) -> str:
    probe.write_text(textwrap.dedent(body), encoding="utf-8")
    out = subprocess.run(
        ["go", "run", str(probe)],
        cwd=_GO_MODULE_DIR,
        check=True,
        capture_output=True,
        text=True,
        env={"GOCACHE": "/tmp/go-build-cache", "HOME": str(Path.home()), "PATH": __import__("os").environ["PATH"]},
    )
    return out.stdout.strip()


def test_batch_and_streaming_writers_agree(tmp_path):
    """Two Python writers, one episode, one identity.

    The streaming writer builds the container itself instead of going through
    save_wshard. A file written a timestep at a time and the same file written
    all at once are the same episode; if the two disagree, the identity is
    recording which code path ran.
    """
    batch = tmp_path / "batch.wshard"
    save_wshard(_fixture_episode(), batch)

    stream = tmp_path / "stream.wshard"
    w = WShardStreamWriter(
        stream,
        "parity-ep",
        [
            StreamChannelDef("zulu", DType.FLOAT32, [2]),
            StreamChannelDef("alpha", DType.FLOAT32, [2]),
            StreamChannelDef("ctrl", DType.FLOAT32, [2]),
        ],
    )
    w.begin_episode(env_id="ParityEnv-v0")
    for t in range(T):
        w.write_timestep(
            t=t,
            observations={
                "zulu": np.array([t, t], dtype=np.float32),
                "alpha": np.array([t + 100, t + 100], dtype=np.float32),
            },
            actions={"ctrl": np.array([t + 200, t + 200], dtype=np.float32)},
            reward=float(t),
            done=(t == T - 1),
        )
    w.end_episode()

    assert episode_identity(stream) == episode_identity(batch)


def test_channel_insertion_order_does_not_change_the_identity(tmp_path):
    """The same channels added in a different order are the same episode.

    meta/channels used to go out in dict insertion order, which records how the
    caller happened to build the episode -- so the same recording assembled by
    two callers sealed two identities. Sorted order is the only writer-
    independent choice, and it is what the Go writer already used.
    """
    a = Episode(id="order", length=T, env_id="OrderEnv-v0")
    b = Episode(id="order", length=T, env_id="OrderEnv-v0")
    chans = {
        "zulu": np.ones((T, 2), dtype=np.float32),
        "alpha": np.zeros((T, 3), dtype=np.float32),
    }
    for name in ("zulu", "alpha"):
        a.observations[name] = Channel(name, DType.FLOAT32, list(chans[name].shape[1:]), chans[name])
    for name in ("alpha", "zulu"):
        b.observations[name] = Channel(name, DType.FLOAT32, list(chans[name].shape[1:]), chans[name])

    pa, pb = tmp_path / "a.wshard", tmp_path / "b.wshard"
    save_wshard(a, pa)
    save_wshard(b, pb)
    assert episode_identity(pa) == episode_identity(pb)


@pytest.mark.parametrize("name", sorted(p.stem for p in _GOLDEN_DIR.glob("*.wshard")))
def test_golden_survives_a_python_resave(name, tmp_path):
    """A file written by the golden generator keeps its identity through Python.

    The generator is a third writer, hand-rolled from the spec with no
    dependency on go/shard -- which is the point: it is the only in-repo
    evidence that the format is implementable from FORMAT.md rather than by
    copying an implementation. It writes at align64 and the Python writer at
    align32, so this also pins the documented invariant that container
    alignment does not reach the identity.
    """
    src = _GOLDEN_DIR / f"{name}.wshard"
    sealed = episode_identity(src)

    hashes = json.loads((_GOLDEN_DIR / "golden_hashes.json").read_text())
    assert sealed == hashes[f"identity_{name}"], "golden_hashes.json is stale"

    out = tmp_path / f"{name}-resaved.wshard"
    save_wshard(load_wshard(src), out)
    assert episode_identity(out) == sealed


def test_go_and_python_writers_agree(tmp_path):
    """Two languages, one episode, one identity.

    This is the whole claim in one assertion. It failed before canonical JSON
    reached meta/wshard, meta/episode and meta/channels: every tensor block was
    already byte-identical across the two, and the identity diverged purely on
    metadata key order, a Go array that could not encode "absent", and a
    time/ticks block one writer emitted and the other did not.
    """
    _require_go()

    py_path = tmp_path / "py.wshard"
    save_wshard(_fixture_episode(), py_path)

    go_path = tmp_path / "go.wshard"
    got = _run_go(
        tmp_path / "write_probe.go",
        f"""
        package main

        import (
            "encoding/binary"
            "fmt"
            "log"
            "math"

            "github.com/phenomenon0/wshard/go/shard"
        )

        func f32Bytes(vals []float32) []byte {{
            buf := make([]byte, len(vals)*4)
            for i, v := range vals {{
                binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
            }}
            return buf
        }}

        func main() {{
            const T = {T}
            zulu := make([]float32, T*2)
            alpha := make([]float32, T*2)
            ctrl := make([]float32, T*2)
            rewards := make([]float32, T)
            done := make([]bool, T)
            for t := 0; t < T; t++ {{
                zulu[t*2], zulu[t*2+1] = float32(t), float32(t)
                alpha[t*2], alpha[t*2+1] = float32(t)+100, float32(t)+100
                ctrl[t*2], ctrl[t*2+1] = float32(t)+200, float32(t)+200
                rewards[t] = float32(t)
            }}
            done[T-1] = true

            ep := &shard.WShardEpisode{{
                ID:      "parity-ep",
                EnvID:   "ParityEnv-v0",
                LengthT: T,
                Timebase: shard.WShardTimebase{{Type: "ticks", TickHz: 30}},
                Observations: map[string]*shard.WShardChannel{{
                    "zulu":  {{Name: "zulu", DType: "f32", Shape: []int{{2}}, Data: f32Bytes(zulu)}},
                    "alpha": {{Name: "alpha", DType: "f32", Shape: []int{{2}}, Data: f32Bytes(alpha)}},
                }},
                Actions: map[string]*shard.WShardChannel{{
                    "ctrl": {{Name: "ctrl", DType: "f32", Shape: []int{{2}}, Data: f32Bytes(ctrl)}},
                }},
                Rewards:      rewards,
                Terminations: done,
            }}
            if err := shard.CreateWShard({json.dumps(go_path.as_posix())}, ep); err != nil {{
                log.Fatal(err)
            }}
            id, err := shard.EpisodeIdentity({json.dumps(go_path.as_posix())})
            if err != nil {{
                log.Fatal(err)
            }}
            fmt.Print(id)
        }}
        """,
    )
    assert got == episode_identity(py_path)
    assert got == episode_identity(go_path)


def test_python_written_episode_survives_a_go_resave(tmp_path):
    """The mirror: Go reads a Python file, writes it back, identity unchanged.

    A reader that silently drops or invents a field is invisible until someone
    reseals -- Go's meta/episode used to gain "timestep_range":[0,0] on the way
    through, because omitempty is a no-op on a Go array.
    """
    _require_go()

    py_path = tmp_path / "py.wshard"
    save_wshard(_fixture_episode(), py_path)
    rewritten = tmp_path / "go-rewrite.wshard"

    got = _run_go(
        tmp_path / "mirror_probe.go",
        f"""
        package main

        import (
            "fmt"
            "log"

            "github.com/phenomenon0/wshard/go/shard"
        )

        func main() {{
            ep, err := shard.OpenWShard({json.dumps(py_path.as_posix())})
            if err != nil {{
                log.Fatal(err)
            }}
            if err := shard.CreateWShard({json.dumps(rewritten.as_posix())}, ep); err != nil {{
                log.Fatal(err)
            }}
            id, err := shard.EpisodeIdentity({json.dumps(rewritten.as_posix())})
            if err != nil {{
                log.Fatal(err)
            }}
            fmt.Print(id)
        }}
        """,
    )
    assert got == episode_identity(py_path)
