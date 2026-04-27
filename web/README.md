# WShard webpage

Single-file static site for WShard. Two pages:

- `index.html` — landing page, renders the project README inline.
- `MARKET_RELEASE.html` — Show HN launch announcement.

Both are self-contained: the markdown source is embedded as a `<script
type="text/markdown">` block and rendered client-side with `marked.js` from
jsDelivr. Open them directly in a browser:

```bash
xdg-open web/index.html
# or:
python -m http.server -d web 8080  # http://localhost:8080
```

Or deploy as static files (GitHub Pages, Netlify, Cloudflare Pages — the
whole `web/` directory is a publishable artifact).
