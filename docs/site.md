# The landing page (`site/`)

A single static page for sharing the project: the interactive toy, one figure per experiment, the honest bottom line and
links to the evidence. It needs no server and no build tools beyond Python, and **no login is needed to build it**.

| step | command |
|---|---|
| regenerate the toy sweep behind the demo (CPU, about 30 s; uses the repository's own toy code) | `make site-data` |
| build `site/` from `results/` and the figures | `make site` |
| fail if `site/` is out of date with `results/` (CI runs this) | `make site-check` |
| preview | `python -m http.server -d site 8000`, then open <http://localhost:8000> (or open `site/index.html` directly) |

Every number on the page is read from `results/**/*.json`, the same files as the README tables, so the page cannot drift
from the results. The demo's data (`site/data/toy_mix.json`, 117 KB) is embedded in `index.html`; the page loads no external
script, font or image. `site/explorer.html` is a copy of `docs/explorer.html`.

## Deploying (you do this; it needs your own account)

- **Vercel CLI:** `cd site && npx vercel` (it opens your browser to log in; `npx vercel --prod` for production). No settings are
  needed: `site/vercel.json` turns on clean URLs and caching.
- **Vercel dashboard:** import the GitHub repository and set *Root Directory* to `site`, *Framework Preset* to "Other", and leave
  the build command empty.
- **GitHub Pages** (once the repository is public): serve the `site/` folder, for example with a workflow that uploads it as the
  Pages artifact.

Before sharing the link, make the repository public (the page links to it), or change `REPO` in `scripts/build_site.py` and
rebuild.
