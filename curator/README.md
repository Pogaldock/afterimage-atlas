# Atlas curator

Nightly GitHub Action (free minutes) that does the heavy work the Worker can't: CLIP scoring, near-duplicate removal,
thumbnails, and image search. Cloudflare only stores and serves.

- `styles.json` — prompts (what a good reference looks like) and search queries per style. Edit and push; the job reruns.
- `curate.py` — the pipeline. Tunables at the top (`DUPE`, `KILL_BELOW`, `UPLOAD_MIN`, `APPLY_LOW_KILLS`…).
- `last-run.md` — committed by the job after each run: counts, score deciles per style, errors.
- Auth: `ATLAS_TOKEN` repository secret (atlas-only scope).

Run manually: Actions → "atlas curator" → Run workflow (optionally skip steps).
