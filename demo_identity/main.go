// Command demo_identity dogfoods the W-SHARD meta/identity feature the way a
// fabric-style agent-memory worker would:
//
//  1. a worker "write path" builds a synthetic episode payload (real float32
//     tensor bytes + state/metadata JSON + text notes) and persists it;
//  2. the file's episode_identity is shown to be encoding-agnostic: the same
//     logical episode written uncompressed vs zstd vs lz4 (Python writer) has
//     byte-identical identity, and Go's VerifyIdentity re-derives the same
//     value from each encoding, including a Python-compressed file;
//  3. a single flipped byte inside one entry (with its CRC32C patched so the
//     container itself cannot tell) makes verify_identity() fail while the
//     trusted episode_identity() lookup does not;
//  4. cross-language: Go and Python compute the same identity for the same
//     file.
//
// Files it produces live in a fresh temp dir each run; nothing in the repo is
// modified except this demo directory itself.
package main

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"hash/crc32"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/phenomenon0/wshard/go/shard"
)

// ---- shared episode spec (mirrored exactly by verify_py.py) ----------------

const (
	epID     = "demo-0001"
	envID    = "fabric/sandbox"
	T        = 128
	stateDim = 8
	notesLit = "ep=demo-0001 worker=fabric step=retrieve tool=web_search ok q=wshard-identity"
)

// notesText returns a UTF-8 note log of exactly T bytes (one byte per
// timestep, so it round-trips as a scalar-per-step u8 observation channel).
func notesText() string {
	s := notesLit
	if len(s) > T {
		panic(fmt.Sprintf("notes literal longer than T=%d", T))
	}
	return s + strings.Repeat(" ", T-len(s))
}

// stateBytes returns T*stateDim float32 LE bytes, bit-identical to what
// verify_py.py computes: every value is m/8 with integer m, exactly
// representable in float32, so Go and Python agree without any float library.
func stateBytes() []byte {
	out := make([]byte, T*stateDim*4)
	for t := 0; t < T; t++ {
		for k := 0; k < stateDim; k++ {
			m := (t + k*13) % 64
			v := float32(m) * 0.125 // exact: m/8
			binary.LittleEndian.PutUint32(out[(t*stateDim+k)*4:], math.Float32bits(v))
		}
	}
	return out
}

// buildEpisode mirrors the synthetic fabric episode in verify_py.py.
func buildEpisode() *shard.WShardEpisode {
	rewards := make([]float32, T)
	dones := make([]bool, T)
	for i := range rewards {
		rewards[i] = float32(i) * 0.25
	}
	dones[T-1] = true
	return &shard.WShardEpisode{
		ID:       epID,
		EnvID:    envID,
		LengthT:  T,
		Timebase: shard.WShardTimebase{Type: "ticks", TickHz: 30},
		Observations: map[string]*shard.WShardChannel{
			"state": {Name: "state", DType: "float32", Shape: []int{stateDim}, Data: stateBytes()},
			"notes": {Name: "notes", DType: "u8", Shape: []int{}, Data: []byte(notesText())},
		},
		Rewards:       rewards,
		Terminations:  dones,
		TimestepRange: [2]int{0, T - 1},
		// Canonical-JSON-friendly ASCII state/metadata JSON.
		Metadata: map[string]any{
			"kind":         "agent-memory",
			"worker":       "fabric/7",
			"session":      "s-2024-001",
			"episode_type": "episodic_memory",
			"tools":        []string{"web_search", "shell", "file_edit"},
			"steps":        12,
			"tags":         []string{"identity", "dogfood"},
		},
	}
}

// ---- tamper helper: flip one byte of signal/state and patch its CRC32C -----

func flipSignalStateByte(path string) error {
	r, err := shard.OpenShard(path)
	if err != nil {
		return err
	}
	i := r.Lookup("signal/state")
	if i < 0 {
		r.Close()
		return fmt.Errorf("signal/state not found")
	}
	e := *r.GetEntryInfo(i)
	r.Close()

	buf, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	buf[e.DataOffset] ^= 0xFF
	// Whoever edits the file can recompute the CRC; that is exactly the hole
	// meta/identity exists to close.
	crc := crc32.Checksum(buf[e.DataOffset:e.DataOffset+e.DiskSize], crc32.MakeTable(crc32.Castagnoli))
	binary.LittleEndian.PutUint32(buf[shard.ShardHeaderSize+i*shard.ShardIndexEntrySize+40:], crc)
	return os.WriteFile(path, buf, 0o644)
}

// ---- tiny helpers ----------------------------------------------------------

func leafHex(b []byte) string {
	s := sha256.Sum256(b)
	return hex.EncodeToString(s[:])
}

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "FATAL:", err)
		os.Exit(1)
	}
}

func check(name string, ok bool, extra ...string) {
	status := "PASS"
	if !ok {
		status = "FAIL"
	}
	fmt.Printf("[%s] %s", status, name)
	for _, e := range extra {
		fmt.Printf("  (%s)", e)
	}
	fmt.Println()
	if !ok {
		os.Exit(1)
	}
}

func copyFile(src, dst string) error {
	b, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	return os.WriteFile(dst, b, 0o644)
}

func main() {
	dir, err := os.MkdirTemp("", "wshard-demo-identity")
	must(err)
	fmt.Println("demo dir:", dir)

	// ---- Go worker write path ------------------------------------------
	goFile := filepath.Join(dir, "episode_go_none.wshard")
	must(shard.CreateWShard(goFile, buildEpisode()))
	fmt.Printf("Go wrote %s (%d bytes)\n", filepath.Base(goFile), fileSize(goFile))

	verified, err := shard.VerifyIdentity(goFile)
	must(err)
	trusted, err := shard.EpisodeIdentity(goFile)
	must(err)
	check("Go VerifyIdentity == EpisodeIdentity, 64 hex", verified == trusted && len(verified) == 64, verified)
	idGo := verified

	// Identity leaf map sanity: print the committed block's contents.
	r, err := shard.OpenShard(goFile)
	must(err)
	ident, err := r.ReadEntryByName(shard.IdentityBlock)
	must(err)
	var doc struct {
		V       int               `json:"v"`
		Leaf    string            `json:"leaf"`
		Entries map[string]string `json:"entries"`
	}
	must(json.Unmarshal(ident, &doc))
	fmt.Printf("meta/identity block: v=%d leaf=%s entries=%d  block sha256=%s\n",
		doc.V, doc.Leaf, len(doc.Entries), idGo)
	fmt.Printf("  sha256(block bytes) == sha256(canonical JSON of leaf map): %v\n",
		leafHex(ident) == idGo && doc.V == 1 && doc.Leaf == "sha256")
	r.Close()

	// ---- tamper: one flipped byte, CRC patched -------------------------
	tampered := filepath.Join(dir, "episode_tampered.wshard")
	must(copyFile(goFile, tampered))
	must(flipSignalStateByte(tampered))
	if _, err := shard.OpenWShard(tampered); err != nil {
		must(fmt.Errorf("container should not notice the forged CRC, got: %v", err))
	}
	check("OpenWShard accepts tampered file (CRC matches, container cannot tell)", true)

	idTampered, err := shard.EpisodeIdentity(tampered)
	must(err)
	check("trusted EpisodeIdentity(tampered) unchanged (it trusts meta/identity)",
		idTampered == idGo, idTampered)

	_, err = shard.VerifyIdentity(tampered)
	check("Go VerifyIdentity(tampered) fails naming signal/state",
		err != nil && strings.Contains(err.Error(), "signal/state"), errStr(err))

	// ---- companion Python verifier ------------------------------------
	pyOut := filepath.Join(dir, "python_results.json")
	py := filepath.Join("verify_py.py")
	cmd := exec.Command("python3", py, "--go-file", goFile, "--out", pyOut, "--dir", dir)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	must(cmd.Run())

	raw, err := os.ReadFile(pyOut)
	must(err)
	var res struct {
		GoFileIdentity string            `json:"go_file_identity"`
		GoFileVerify   string            `json:"go_file_verify"`
		CodecIDs       map[string]string `json:"codec_ids"`
		ZstdPath       string            `json:"zstd_path"`
		PyTamper       string            `json:"py_tamper_error"`
	}
	must(json.Unmarshal(raw, &res))

	// Assertion 3a: same file, same identity in both languages.
	check("cross-language: Python episode_identity(Go file) == Go identity",
		res.GoFileIdentity == idGo && res.GoFileVerify == idGo, res.GoFileIdentity)

	// Assertion 1: encoding-agnostic identity (none == zstd == lz4).
	all := map[string]bool{}
	for _, id := range res.CodecIDs {
		all[id] = true
	}
	check("encoding-agnostic: none == zstd == lz4 identity (Python writer)",
		len(all) == 1 && len(res.CodecIDs) == 3, first(res.CodecIDs))

	// Assertion 3b: Go verifies the Python-compressed file and agrees on the id.
	idPyZstd, err := shard.VerifyIdentity(res.ZstdPath)
	must(err)
	idPyZstdTrust, err := shard.EpisodeIdentity(res.ZstdPath)
	must(err)
	check("cross-language: Go VerifyIdentity(Python zstd file) == Python identity",
		idPyZstd == res.CodecIDs["ZSTD"] && idPyZstdTrust == res.CodecIDs["ZSTD"], idPyZstd)

	// Assertion 2 (Python half): flipped byte fails Python verify too.
	check("Python verify_identity(tampered copy) also fails naming signal/state",
		strings.Contains(res.PyTamper, "signal/state"), res.PyTamper)

	fmt.Println("\nALL DEMO ASSERTIONS PASS")
	fmt.Println("go file:      ", goFile)
	fmt.Println("python results:", pyOut)
}

func fileSize(p string) int64 {
	fi, err := os.Stat(p)
	must(err)
	return fi.Size()
}

func errStr(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

func first(m map[string]string) string {
	for _, v := range m {
		return v
	}
	return ""
}
