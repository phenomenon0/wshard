# WShard webpage

Single-file static site for WShard. Two pages:

- `index.html` — landing page, renders the project README inline.
- `MARKET_RELEASE.html` — Show HN launch announcement.

Both are self-contained: the markdown source is embedded as a `<script
type="text/markdown">` block and rendered client-side with `marked.js` from
jsDelivr. The embedded copies are hand-trimmed, not generated — `index.html`
carries a shortened README and `MARKET_RELEASE.html` swaps the title block for
HTML — so editing `README.md` or `MARKET_RELEASE.md` does not update the site.
Re-check the embeds by hand when a claim or a benchmark number changes.

Open them directly in a browser:

```bash
xdg-open web/index.html
# or:
python -m http.server -d web 8080  # http://localhost:8080
```

Or deploy as static files (GitHub Pages, Netlify, Cloudflare Pages — the
whole `web/` directory is a publishable artifact).
