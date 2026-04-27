/**
 * WShard Viewer — minimal structural browser parser.
 *
 * Does NOT use @wshard/core (Node-only fs dependency).
 * Implements just enough of the Shard v2 binary format to display
 * episode metadata and a block inventory without decoding tensor data.
 *
 * Format reference: docs/DEEP_DIVE.md · js/src/types.ts
 */

// ── Constants ────────────────────────────────────────────────────────────────
const MAGIC = [0x53, 0x48, 0x52, 0x44]; // "SHRD"
const HDR   = 64;  // header size
const ESIZ  = 48;  // index entry size
const COMP  = 0x0001; // BLOCK_FLAG_COMPRESSED

// ── Binary parser ────────────────────────────────────────────────────────────
interface Entry { name:string; flags:number; dataOffset:bigint; diskSize:bigint; origSize:bigint; compressed:boolean }
interface Header { entryCount:number; strOff:number; datOff:number; comprDefault:number }
interface EpMeta { episode_id?:string; env_id?:string; length_T?:number; action_space?:string }
interface Chan   { id:string; dtype:string; shape:number[] }

function parseHeader(v: DataView): Header {
  for (let i = 0; i < 4; i++)
    if (v.getUint8(i) !== MAGIC[i]) throw new Error('Invalid magic — not a .wshard file');
  if (v.getUint8(4) !== 0x02) throw new Error(`Unsupported Shard version 0x${v.getUint8(4).toString(16)}`);
  if (v.getUint8(5) !== 0x05) throw new Error(`Not a W-SHARD file: role=0x${v.getUint8(5).toString(16)}`);
  return {
    entryCount:   v.getUint32(12, true),
    strOff:       Number(v.getBigUint64(16, true)),
    datOff:       Number(v.getBigUint64(24, true)),
    comprDefault: v.getUint8(9),
  };
}

function parseEntries(v: DataView, h: Header, ab: ArrayBuffer): Entry[] {
  const dec = new TextDecoder();
  const str = new Uint8Array(ab, h.strOff, h.datOff - h.strOff);
  return Array.from({ length: h.entryCount }, (_, i) => {
    const b = HDR + i * ESIZ;
    const nOff = v.getUint32(b + 8, true), nLen = v.getUint16(b + 12, true);
    const flags = v.getUint16(b + 14, true);
    return {
      name:       dec.decode(str.subarray(nOff, nOff + nLen)),
      flags,
      dataOffset: v.getBigUint64(b + 16, true),
      diskSize:   v.getBigUint64(b + 24, true),
      origSize:   v.getBigUint64(b + 32, true),
      compressed: (flags & COMP) !== 0,
    };
  });
}

function lz4Decompress(src: Uint8Array, dstSize: number): Uint8Array {
  const dst = new Uint8Array(dstSize);
  let sp = 0, dp = 0;
  while (sp < src.length) {
    const tok = src[sp++];
    let lit = tok >> 4, ml = tok & 0xf;
    if (lit === 15) { let s: number; do { s = src[sp++]; lit += s; } while (s === 255); }
    dst.set(src.subarray(sp, sp + lit), dp); sp += lit; dp += lit;
    if (sp >= src.length) break;
    const off = src[sp] | (src[sp + 1] << 8); sp += 2;
    if (!off) throw new Error('LZ4: zero offset');
    let mlen = ml + 4;
    if (ml === 15) { let s: number; do { s = src[sp++]; mlen += s; } while (s === 255); }
    const ms = dp - off;
    for (let k = 0; k < mlen; k++) dst[dp++] = dst[ms + k];
  }
  return dst.subarray(0, dp);
}

function readJson(ab: ArrayBuffer, e: Entry, comprDefault: number): unknown {
  let bytes = new Uint8Array(ab, Number(e.dataOffset), Number(e.diskSize));
  if (e.compressed && e.diskSize !== e.origSize) {
    const isLz4 = (e.flags & 0x0004) !== 0 || comprDefault === 0x02;
    if (!isLz4) throw new Error(`Block "${e.name}" is zstd-compressed (metadata blocks rarely are — unusual file).`);
    bytes = lz4Decompress(bytes, Number(e.origSize));
  }
  return JSON.parse(new TextDecoder().decode(bytes));
}

async function parseFile(ab: ArrayBuffer) {
  const v = new DataView(ab);
  const h = parseHeader(v);
  const entries = parseEntries(v, h, ab);
  const byName = new Map(entries.map(e => [e.name, e]));
  const get = (k: string, label: string) => {
    const e = byName.get(k); if (!e) throw new Error(`Missing required block: ${label}`);
    return readJson(ab, e, h.comprDefault);
  };
  return {
    header: h, entries, fileSize: ab.byteLength,
    episode:  get('meta/episode',  'meta/episode')  as EpMeta,
    channels: (get('meta/channels', 'meta/channels') as { channels: Chan[] }).channels,
  };
}

// ── Helpers ──────────────────────────────────────────────────────────────────
const esc = (s: unknown) => String(s ?? '—').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
const fmt = (n: number|bigint) => { const v = Number(n);
  if (v < 1024) return `${v} B`; if (v < 1<<20) return `${(v/1024).toFixed(1)} KB`;
  if (v < 1<<30) return `${(v/(1<<20)).toFixed(2)} MB`; return `${(v/(1<<30)).toFixed(2)} GB`; };
const ns = (name: string) => { const i = name.indexOf('/'); return i < 0 ? '—' : name.slice(0, i); };
const comprName = (c: number) => c === 1 ? 'zstd' : c === 2 ? 'lz4' : 'none';

// ── State ────────────────────────────────────────────────────────────────────
type SK = 'name'|'ns'|'dtype'|'shape'|'diskSize'|'origSize'|'compressed';
let _entries: Entry[] = [], _chans: Chan[] = [], _sk: SK = 'name', _asc = true;

// ── Render ───────────────────────────────────────────────────────────────────
function el(id: string) { return document.getElementById(id)!; }

function renderMeta(ep: EpMeta, entries: Entry[], fileSize: number, comprDefault: number) {
  const compressed = entries.filter(e => e.compressed).length;
  el('meta-section').innerHTML = `<h2>Episode Metadata</h2><table class="meta-table"><tbody>
    <tr><th>episode_id</th><td class="mono">${esc(ep.episode_id)}</td></tr>
    <tr><th>env_id</th><td class="mono">${esc(ep.env_id)}</td></tr>
    <tr><th>length_T</th><td class="mono">${esc(ep.length_T)} timesteps</td></tr>
    <tr><th>action_space</th><td class="mono">${esc(ep.action_space)}</td></tr>
    <tr><th>file size</th><td class="mono">${fmt(fileSize)}</td></tr>
    <tr><th>total blocks</th><td class="mono">${entries.length} (${compressed} compressed)</td></tr>
    <tr><th>default compression</th><td class="mono">${comprName(comprDefault)}</td></tr>
  </tbody></table>`;
  el('meta-section').classList.remove('hidden');
}

function renderBlocks() {
  const chanByShortId = new Map(_chans.map(c => [c.id, c]));
  interface Row { name:string; ns:string; dtype:string; shape:string; diskSize:bigint; origSize:bigint; compressed:boolean }
  const rows: Row[] = _entries.map(e => {
    const short = e.name.includes('/') ? e.name.slice(e.name.lastIndexOf('/') + 1) : e.name;
    const ch = chanByShortId.get(short) ?? chanByShortId.get(e.name);
    return { name: e.name, ns: ns(e.name), dtype: ch?.dtype ?? '—',
      shape: ch?.shape ? `[${ch.shape.join(', ')}]` : '—',
      diskSize: e.diskSize, origSize: e.origSize, compressed: e.compressed };
  });

  rows.sort((a, b) => {
    const k = _sk;
    let cmp = k === 'diskSize' ? (a.diskSize < b.diskSize ? -1 : a.diskSize > b.diskSize ? 1 : 0)
            : k === 'origSize' ? (a.origSize < b.origSize ? -1 : a.origSize > b.origSize ? 1 : 0)
            : k === 'compressed' ? Number(a.compressed) - Number(b.compressed)
            : String((a as Record<string,unknown>)[k]) < String((b as Record<string,unknown>)[k]) ? -1 : 1;
    return _asc ? cmp : -cmp;
  });

  const cols: {key: SK; label: string}[] = [
    {key:'name',label:'Block Name'},{key:'ns',label:'Namespace'},
    {key:'dtype',label:'DType'},{key:'shape',label:'Shape'},
    {key:'diskSize',label:'On-disk'},{key:'origSize',label:'Orig size'},{key:'compressed',label:'Compressed'},
  ];
  const ind = (k: SK) => k === _sk ? (_asc ? ' ▲' : ' ▼') : '';
  const thead = cols.map(c => `<th data-key="${c.key}" class="sortable">${esc(c.label)}${ind(c.key)}</th>`).join('');
  const tbody = rows.map((r, i) =>
    `<tr${i%2?' class="alt"':''}>
      <td class="mono name-cell">${esc(r.name)}</td>
      <td class="mono ns-cell">${esc(r.ns)}</td>
      <td class="mono">${esc(r.dtype)}</td>
      <td class="mono">${esc(r.shape)}</td>
      <td class="mono">${fmt(r.diskSize)}</td>
      <td class="mono">${fmt(r.origSize)}</td>
      <td>${r.compressed?'<span class="badge badge-yes">yes</span>':'<span class="badge">no</span>'}</td>
    </tr>`).join('');

  el('block-section').innerHTML =
    `<h2>Block Inventory <span class="count">(${rows.length})</span></h2>
     <div class="table-wrap"><table class="block-table">
       <thead><tr>${thead}</tr></thead><tbody>${tbody}</tbody>
     </table></div>`;

  el('block-section').querySelectorAll<HTMLElement>('th[data-key]').forEach(th =>
    th.addEventListener('click', () => {
      const k = th.dataset.key as SK;
      _sk === k ? (_asc = !_asc) : (_sk = k, _asc = true);
      renderBlocks();
    }));

  el('block-section').classList.remove('hidden');
}

// ── File handling ─────────────────────────────────────────────────────────────
function showError(msg: string) { const e = el('error-box'); e.textContent = msg; e.classList.remove('hidden'); }
function clearAll() { ['error-box','meta-section','block-section'].forEach(id => el(id).classList.add('hidden')); }

async function handleFile(file: File) {
  clearAll();
  el('drop-zone').classList.remove('drag-over');
  el('drop-label').textContent = `Parsing ${file.name}…`;
  try {
    const parsed = await parseFile(await file.arrayBuffer());
    _entries = parsed.entries; _chans = parsed.channels; _sk = 'name'; _asc = true;
    el('drop-label').textContent = `${file.name} · ${fmt(file.size)}`;
    renderMeta(parsed.episode, parsed.entries, parsed.fileSize, parsed.header.comprDefault);
    renderBlocks();
  } catch (err) {
    el('drop-label').textContent = 'Drop a .wshard file here — or click to browse';
    showError(err instanceof Error ? err.message : String(err));
  }
}

// ── Init ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  const zone  = el('drop-zone');
  const input = el('file-input') as HTMLInputElement;
  zone.addEventListener('dragover',  e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => { e.preventDefault(); const f = e.dataTransfer?.files[0]; if (f) handleFile(f); });
  zone.addEventListener('click', () => input.click());
  input.addEventListener('change', () => { const f = input.files?.[0]; if (f) handleFile(f); });
});
