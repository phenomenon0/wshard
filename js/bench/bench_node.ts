/**
 * bench_node.ts — WShard write/read benchmark for Node.js / TypeScript.
 *
 * Workload: T=1000 step synthetic episode — same structured-but-realistic
 *   workload as bench_python.py for apples-to-apples comparison.
 *
 *   signal/joint_pos  [1000, 7]         float32  ~28 KB
 *   signal/rgb        [1000, 84, 84, 3] uint8    ~21 MB  (gradient + ±2 noise)
 *   action/ctrl       [1000, 7]         float32  ~28 KB
 *   reward            [1000]            float32  ~4 KB
 *   done              [1000]            bool     ~1 KB
 *   Total raw payload: ~21 MB
 *
 * Configurations:
 *   wshard-none   — WShard, no compression
 *   wshard-zstd   — WShard, zstd
 *   wshard-lz4    — WShard, lz4 (via fflate lz4-block)
 *
 * Run:
 *   cd js && npx tsx bench/bench_node.ts
 */

import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import {
  WShardWriter,
  WShardReader,
  initZstd,
  type CompressionType,
} from '../src/index.js';

// ── workload parameters ────────────────────────────────────────────────────────

const SEED = 42;
const T = 1000;
const H = 84;
const W = 84;
const C = 3;
const JOINT_DIM = 7;
const CTRL_DIM = 7;
const RUNS = 5;

// ── simple deterministic PRNG (LCG, seed-based) ───────────────────────────────
// We need a seeded RNG to mirror the Python/Go fixed-seed approach.
// Using a simple 64-bit LCG adapted for JS (BigInt for 64-bit state).

class SeededRNG {
  private state: bigint;

  constructor(seed: number) {
    this.state = BigInt(seed >>> 0);
  }

  // Returns an integer in [0, n)
  nextInt(n: number): number {
    // LCG: multiplier and increment from Knuth
    this.state = (this.state * 6364136223846793005n + 1442695040888963407n) & 0xFFFFFFFFFFFFFFFFn;
    return Number((this.state >> 33n) % BigInt(n));
  }

  // Returns a float in [0, 1)
  nextFloat(): number {
    return this.nextInt(0x7FFFFFFF) / 0x7FFFFFFF;
  }

  // Returns a standard-normal float (Box-Muller)
  nextNormal(): number {
    const u1 = this.nextFloat() + 1e-10;
    const u2 = this.nextFloat();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }
}

// ── structured RGB workload ────────────────────────────────────────────────────
// Vertical gradient (0..255 top-to-bottom) + ±2 noise per pixel.
// Mirrors the Python bench exactly. Compresses ~2× with zstd.

function makeRGB(): Uint8Array {
  const rng = new SeededRNG(SEED + 1);
  const buf = new Uint8Array(T * H * W * C);
  let idx = 0;
  for (let t = 0; t < T; t++) {
    for (let row = 0; row < H; row++) {
      const base = (row / (H - 1)) * 255;
      for (let col = 0; col < W; col++) {
        for (let ch = 0; ch < C; ch++) {
          const noise = rng.nextInt(4) - 2; // [-2, +1]
          const v = Math.max(0, Math.min(255, base + noise));
          buf[idx++] = v;
        }
      }
    }
  }
  return buf;
}

function makeFloat32(length: number, seed: number): Float32Array {
  const rng = new SeededRNG(seed);
  const arr = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    arr[i] = rng.nextNormal();
  }
  return arr;
}

function makeBool(length: number, seed: number): boolean[] {
  const rng = new SeededRNG(seed);
  const arr: boolean[] = [];
  for (let i = 0; i < length; i++) {
    arr.push(rng.nextInt(100) === 0);
  }
  return arr;
}

// ── workload data ──────────────────────────────────────────────────────────────

interface WorkloadData {
  jointPos: Float32Array;
  rgb: Uint8Array;
  ctrl: Float32Array;
  reward: Float32Array;
  done: boolean[];
}

function makeData(): WorkloadData {
  return {
    jointPos: makeFloat32(T * JOINT_DIM, SEED),
    rgb:      makeRGB(),
    ctrl:     makeFloat32(T * CTRL_DIM, SEED + 2),
    reward:   makeFloat32(T, SEED + 3),
    done:     makeBool(T, SEED + 4),
  };
}

function rawBytes(data: WorkloadData): number {
  return (
    data.jointPos.byteLength +
    data.rgb.byteLength +
    data.ctrl.byteLength +
    data.reward.byteLength +
    data.done.length
  );
}

// ── write helper ───────────────────────────────────────────────────────────────

async function writeEpisode(
  filePath: string,
  data: WorkloadData,
  compression: CompressionType
): Promise<number> {
  const writer = new WShardWriter(filePath, { compression });

  writer.setEpisode({
    episode_id: 'bench_ep',
    env_id: 'BenchEnv-v0',
    length_T: T,
  });

  writer.addChannel({
    id: 'joint_pos',
    dtype: 'f32',
    shape: [JOINT_DIM],
    signal_block: 'signal/joint_pos',
  });
  writer.addChannel({
    id: 'rgb',
    dtype: 'u8',
    shape: [H, W, C],
    signal_block: 'signal/rgb',
  });
  writer.addChannel({
    id: 'ctrl',
    dtype: 'f32',
    shape: [CTRL_DIM],
    signal_block: 'action/ctrl',
  });

  writer.setTimeTicks(Array.from({ length: T }, (_, i) => i));
  writer.setSignal('joint_pos', Buffer.from(data.jointPos.buffer));
  writer.setSignal('rgb', Buffer.from(data.rgb.buffer));
  writer.setAction('ctrl', Buffer.from(data.ctrl.buffer));
  writer.setReward(Array.from(data.reward));
  writer.setDone(data.done);

  return writer.write();
}

// ── read helper ────────────────────────────────────────────────────────────────

async function readEpisode(filePath: string): Promise<void> {
  const reader = new WShardReader(filePath);
  await reader.open();
  // Read all data blocks (mirrors full-episode read in Python bench)
  await reader.getSignal('joint_pos');
  await reader.getSignal('rgb');
  await reader.getAction('ctrl');
  await reader.getReward();
  await reader.getDone();
  await reader.close();
}

// ── timing helpers ─────────────────────────────────────────────────────────────

async function timeMs(fn: () => Promise<void>): Promise<number> {
  const t0 = performance.now();
  await fn();
  return performance.now() - t0;
}

function medianMin(times: number[]): [number, number] {
  const sorted = [...times].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  const median = sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid];
  return [median, sorted[0]];
}

// ── table formatting ───────────────────────────────────────────────────────────

function fmtMs(ms: number): string {
  if (ms < 1) return `${(ms * 1000).toFixed(0)} µs`;
  return `${ms.toFixed(1)} ms`;
}

function fmtMB(bytes: number): string {
  return `${(bytes / 1_048_576).toFixed(2)} MB`;
}

function fmtRatio(raw: number, disk: number): string {
  if (disk === 0) return '—';
  return `${(raw / disk).toFixed(2)}×`;
}

interface BenchRow {
  config: string;
  writeMed: number;
  writeMin: number;
  readMed: number;
  readMin: number;
  fileBytes: number;
  rawBytes: number;
}

function buildTable(rows: BenchRow[]): string {
  const headers = ['Config', 'Write (median)', 'Write (min)', 'Read (median)', 'Read (min)', 'File size', 'Ratio'];
  const sep = '|' + headers.map(h => '-'.repeat(h.length + 2)).join('|') + '|';
  const headerRow = '| ' + headers.join(' | ') + ' |';
  const lines = [headerRow, sep];
  for (const r of rows) {
    const cells = [
      r.config,
      fmtMs(r.writeMed),
      fmtMs(r.writeMin),
      fmtMs(r.readMed),
      fmtMs(r.readMin),
      fmtMB(r.fileBytes),
      fmtRatio(r.rawBytes, r.fileBytes),
    ];
    lines.push('| ' + cells.join(' | ') + ' |');
  }
  return lines.join('\n');
}

// ── main ───────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  // Initialize zstd WASM before benchmarking
  await initZstd();

  const data = makeData();
  const raw = rawBytes(data);

  console.log(`Workload: T=${T}, raw payload = ${fmtMB(raw)}`);
  console.log(`Runs per measurement: ${RUNS}\n`);

  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'wshard-bench-'));
  const rows: BenchRow[] = [];

  try {
    const configs: Array<{ name: string; comp: CompressionType }> = [
      { name: 'wshard-none', comp: 'none' },
      { name: 'wshard-zstd', comp: 'zstd' },
      { name: 'wshard-lz4',  comp: 'lz4'  },
    ];

    for (const { name, comp } of configs) {
      console.log(`  [bench] ${name} ...`);
      const filePath = path.join(tmpDir, `ep_${name}.wshard`);

      // Write runs
      const writeTimes: number[] = [];
      for (let i = 0; i < RUNS; i++) {
        const ms = await timeMs(() => writeEpisode(filePath, data, comp));
        writeTimes.push(ms);
      }
      const [writeMed, writeMin] = medianMin(writeTimes);
      const fileBytes = fs.statSync(filePath).size;

      // Read runs
      const readTimes: number[] = [];
      for (let i = 0; i < RUNS; i++) {
        const ms = await timeMs(() => readEpisode(filePath));
        readTimes.push(ms);
      }
      const [readMed, readMin] = medianMin(readTimes);

      rows.push({ config: name, writeMed, writeMin, readMed, readMin, fileBytes, rawBytes: raw });
    }

    console.log();
    const table = buildTable(rows);
    console.log(table);
    console.log();
    console.log(
      `> Raw payload: ${fmtMB(raw)} (${T} steps · joint_pos f32[${JOINT_DIM}] + rgb u8[${H},${W},${C}] + ctrl f32[${CTRL_DIM}] + reward f32 + done bool)`
    );
  } finally {
    // Cleanup temp files
    try {
      fs.readdirSync(tmpDir).forEach(f => fs.unlinkSync(path.join(tmpDir, f)));
      fs.rmdirSync(tmpDir);
    } catch (_) {
      // best-effort cleanup
    }
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
