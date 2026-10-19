#!/usr/bin/env python3
"""One-off: copy the atlas from the main site's bucket (old API) into this Worker's own bucket (new API).
Run once from Actions → "atlas migrate". Needs OLD_ATLAS_TOKEN (the main site's atlas token) and ATLAS_TOKEN (this Worker's).
Idempotent: re-running skips nothing harmful (objects are overwritten with identical bytes)."""
import os, sys, json, time
from concurrent.futures import ThreadPoolExecutor
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
OLD, NEW = os.environ.get("OLD_SITE", "https://symbiozefpv.com"), os.environ.get("SITE", "https://atlas.symbiozefpv.com")
OA = {"authorization": "Bearer " + os.environ["OLD_ATLAS_TOKEN"]}; NA = {"authorization": "Bearer " + os.environ["ATLAS_TOKEN"]}
S = requests.Session(); S.mount("https://", HTTPAdapter(pool_maxsize=16, max_retries=Retry(total=6, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=frozenset(["GET", "PUT"]), raise_on_status=False)))
stats = {"objects": 0, "images": 0, "elements": 0, "cands": 0, "mont": 0, "failed": [], "bytes": {}}
def oget(key):
    r = S.get(OLD + "/api/atlas/cur/obj", params={"o": key}, headers=OA, timeout=300); return r.content if r.status_code == 200 else None
def nput(key, data, ct):
    r = S.put(NEW + "/api/atlas/cur/obj", params={"o": key}, data=data, headers={**NA, "content-type": ct}, timeout=600); r.raise_for_status()
def raw(key, data, ct):
    r = S.put(NEW + "/api/atlas/cur/raw", params={"o": key}, data=data, headers={**NA, "content-type": ct}, timeout=300); r.raise_for_status()
def copy_public(path, key, kind):
    try:
        g = S.get(OLD + path, timeout=120)
        if g.status_code != 200: stats["failed"].append(f"{kind} {key} {g.status_code}"); return
        ct = g.headers.get("content-type", "image/jpeg").split(";")[0]
        raw(key, g.content, ct); stats[kind] += 1; stats["bytes"][kind] = stats["bytes"].get(kind, 0) + len(g.content)
    except Exception as e: stats["failed"].append(f"{kind} {key} {type(e).__name__}")
import traceback
items = []
try:
    # 1. state + bundle objects
    keys = ["manifest.json", "flags.json", "elements.json", "cands.json", "clip.npz", "elclip.npz", "killed.json", "qstats.json", "qtallied.json", "seen.json",
            "extracted.json", "elhist.json", "curator.json", "extractor.json", "mont.json", "static.json"]
    static_idx = json.loads(oget("atlas/static.json") or b'{"parts":[]}')
    keys += [p["key"][len("atlas/"):] for p in static_idx.get("parts", [])]
    ONLY = set(filter(None, os.environ.get("ONLY", "").split(",")))   # re-run for just these styles' images
    for k in ([] if ONLY else keys):
        data = oget("atlas/" + k)
        if data is None: print("missing", k); continue
        ct = "application/json" if k.endswith(".json") else ("application/x-tar" if k.endswith(".tar") else "application/octet-stream")
        nput("atlas/" + k, data, ct); stats["objects"] += 1; stats["bytes"]["objects"] = stats["bytes"].get("objects", 0) + len(data); print("obj", k, len(data))
    # 2. originals, elements, candidates, edits
    man = json.loads(oget("atlas/manifest.json") or b'{"styles":{}}')
    items = [it for st_, arr in man.get("styles", {}).items() for it in arr if not ONLY or st_ in ONLY]
    els = [] if ONLY else json.loads(oget("atlas/elements.json") or b'{"items":[]}').get("items", [])
    cands = [] if ONLY else json.loads(oget("atlas/cands.json") or b'{"items":[]}').get("items", [])
    mont = [] if ONLY else json.loads(oget("atlas/mont.json") or b'{"items":[]}').get("items", [])
    print(f"{len(items)} images, {len(els)} elements, {len(cands)} candidates, {len(mont)} edits")
    with ThreadPoolExecutor(12) as ex:
        for it in items:
            key = it.get("key") or ("atlas/" + it["url"].split("/api/atlas/f/")[1]); ex.submit(copy_public, "/api/atlas/f/" + key[len("atlas/"):], key, "images")
        for e in els: ex.submit(copy_public, "/api/atlas/el/" + e["id"], "atlas/el/" + e["id"] + ".png", "elements")
        for c in cands: ex.submit(copy_public, "/api/atlas/cand/" + c["id"], "atlas/cand/" + c["id"] + ".png", "cands")
    for m in mont:
        for ext, ct in (("mp4", "video/mp4"), ("jpg", "image/jpeg")):
            data = oget(f"atlas/mont/{m['id']}.{ext}")
            if data: nput(f"atlas/mont/{m['id']}.{ext}", data, ct); stats["mont"] += 1; stats["bytes"]["mont"] = stats["bytes"].get("mont", 0) + len(data)
except Exception:
    stats["crash"] = traceback.format_exc()[-1500:]; print(stats["crash"])
print(json.dumps({k: (v if k != "failed" else len(v)) for k, v in stats.items()}))
for f in stats["failed"][:40]: print("FAILED", f)
open("curator/last-migrate.md", "w").write("# migrate\n\n" + json.dumps({k: (v if k != "failed" else len(v)) for k, v in stats.items() if k != "crash"}, indent=1) + "\n\n```\n" + stats.get("crash", "") + "\n```" + "\n\n" + "\n".join("- " + f for f in stats["failed"][:200]) + "\n")
sys.exit(1 if stats.get("crash") or len(stats["failed"]) > max(20, len(items) // 50) else 0)
