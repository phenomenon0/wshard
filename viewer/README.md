# WShard Viewer

A lightweight browser-based structural viewer for `.wshard` episode files. Drop a file in to see its episode metadata and a full block inventory — no server required.

## How to build

```bash
cd viewer
npm install
npm run build
```

This emits `dist/viewer.js` (single bundled ES module, ~30 KB).

## How to run

```bash
npm run serve
# then open http://localhost:5173
```

Or with Python:

```bash
python -m http.server 5173
```

## What you see

- **Episode metadata**: `episode_id`, `env_id`, `length_T`, `action_space`, file size, block count, compression.
- **Block inventory**: every block in the file with its namespace prefix, dtype and shape (from `meta/channels`), on-disk size, original size, and whether it is compressed. Click any column header to sort.

## Deploying

The viewer is 100% client-side. Copy `index.html` and `dist/viewer.js` to any static host (GitHub Pages, Netlify, Cloudflare Pages, a plain nginx `root`). No build-time configuration needed.

## Limits

- Maximum file size is limited by browser memory; recommend < 500 MB in practice.
- Tensor data is never decoded — this is intentional. Only structural metadata is read.
- zstd-compressed metadata blocks are not supported (rare in practice; metadata blocks are almost never compressed).
- LZ4-compressed metadata blocks are supported.
