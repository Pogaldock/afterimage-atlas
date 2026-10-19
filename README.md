# Afterimage Atlas

A visual atlas of subculture aesthetics (free party, techno, acid, aliens, psychedelia, black metal, Y2K, vaporwave, witch house and more), built from real archive photos, flyers and record art, with an element library cut out of them. By SYMBIOZE (@fpv.symbioze). Live at atlas.symbiozefpv.com.

- `src/` — a small Cloudflare Worker: the atlas API (keep/kill flags, cut-out library, curator endpoints) and the admin login. Its own R2 bucket, its own secrets; it shares nothing with any other site.
- `public/atlas/` — the page. Thumbnails, manifest and the element library are unpacked at deploy from a nightly bundle (`scripts/atlas-assets.mjs`), so viewing costs no Worker requests.
- `curator/curate.py` — nightly job: sources reference images per style (image search, Cover Art Archive, Wikimedia Commons), scores them with CLIP against each style and against the images kept on the site, removes near-duplicates and AI-generated renders, builds the bundle.
- `curator/extract.py` — nightly job: cuts elements out of the images (FastSAM + matting) and learns from which cut-outs get kept.
- `curator/styles.json` — per-style prompts, searches and gates. Edit and push.

Secrets live in the Cloudflare dashboard (Worker: `ATLAS_TOKEN`, `ATLAS_ADMIN_KEY`; build variable `ATLAS_TOKEN`) and in this repository's Actions secrets (`ATLAS_TOKEN`, `DEPLOY_HOOK`). None are in the code.
