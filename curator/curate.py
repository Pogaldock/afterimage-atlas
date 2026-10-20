#!/usr/bin/env python3
"""Afterimage atlas curator — nightly on GitHub Actions (free). Cloudflare only stores originals and serves static files.

Pipeline (steps can be skipped with --skip name):
  sync     master manifest + flags (your keeps/kills/moves) + embedding cache + previous static bundle
  embed    CLIP-embed images that have no embedding yet (ViT-B-32, CPU)
  score    per style: similarity to text prompts and to the images you kept → score 0..1 + best style
  dedupe   near-identical images → kill the weaker copy (reliable, always applied)
  apply    kills = your kills + dupes (+ low scores if APPLY_LOW_KILLS); moves = yours (+ auto if APPLY_MOVES) → Worker
  source   image search (ddgs) + Cover Art Archive + Wikimedia Commons → download → size/dupe/score gate → upload
  bundle   thumbnails (480px webp) + public manifest.json + elements → one tar in R2; the site unpacks it at deploy
  report   atlas/curator.json (in /api/atlas/stats) + curator/last-run.md (committed)

Config: curator/styles.json. Auth: ATLAS_TOKEN env. Site: SITE env.
"""
import io, json, os, re, sys, time, tarfile, argparse, traceback
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import requests
from PIL import Image, ImageOps

SITE = os.environ.get("SITE", "https://atlas.symbiozefpv.com").rstrip("/")
TOKEN = os.environ.get("ATLAS_TOKEN", "")
HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("ATLAS_WORK", os.path.join(HERE, "_work")); os.makedirs(WORK, exist_ok=True)
STYLES = json.load(open(os.path.join(HERE, "styles.json"))); [STYLES.pop(k) for k in list(STYLES) if k.startswith("_")]
# AI-image guard: hosts that serve generated images are never fetched; on no_ai styles CLIP pushes out images that read as renders
AI_HOSTS = re.compile(r"(midjourney|openart\.ai|lexica\.art|nightcafe|playgroundai|playground\.com|civitai|leonardo\.ai|krea\.ai|starryai|craiyon|dreamstudio|stablediffusion|tensor\.art|seaart|pixai|getimg\.ai|aiart|ai-art|ai_art|ai-generated|aigenerated|generative-ai|/ai-image|imagine\.art|deepai|neural\.love|artbreeder|dall-?e|pngtree.*ai|freepik\.com/(premium-)?ai)", re.I)
AI_TEXT = ["an AI generated image, smooth glossy digital painting, midjourney style", "a 3D render or CGI concept art, artstation", "hyperrealistic AI art with plastic skin and perfect lighting", "a fantasy digital illustration with glowing effects"]
REAL_TEXT = ["a grainy film photograph", "a scanned photocopied paper flyer", "an old snapshot photo with flash", "a printed poster or record sleeve scan", "a documentary photo taken with a cheap camera"]

# ---- tunables ----
MODEL = ("ViT-B-32", "laion2b_s34b_b79k")
DUPE = 0.965
KILL_BELOW = 0.22; APPLY_LOW_KILLS = False; MAX_KILL_FRACTION = 0.15
MOVE_MARGIN = 0.15; APPLY_MOVES = False
UPLOAD_MIN = 0.42
MIN_SIDE, MAX_ASPECT = 360, 2.8
MIN_BYTES, MAX_BYTES = 15_000, 8_000_000
QUERIES_PER_STYLE, RESULTS_PER_QUERY, MAX_UPLOADS = 4, 40, 300
SAME = 0.90                 # "more of the same": a new image this close to an existing one in its style is skipped
PER_QUERY_CAP = 12          # uploads per query per run
RETIRE_AFTER, RETIRE_KILL_RATE = 8, 0.7   # a query with ≥8 judged images and ≥70% kills is retired
CAA_PER_STYLE, COMMONS_PER_STYLE = 2, 1     # extra source queries per style per run (rotating)
THUMB_W = 480

S = requests.Session()
# Be polite to the site: back off on 429/5xx (honours Retry-After), few parallel connections, small pauses.
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
S.mount("https://", HTTPAdapter(max_retries=Retry(total=7, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=frozenset(["GET", "PUT", "POST"]), respect_retry_after_header=True, raise_on_status=False), pool_maxsize=6))
S.headers["user-agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
AUTH = {"authorization": "Bearer " + TOKEN}
MB_UA = {"user-agent": "SymbiozeAtlas/1.0 (https://atlas.symbiozefpv.com; atlas reference curator)"}
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
report = {"ran": int(time.time()), "errors": []}
def web(u):  # only plain web links ever reach the page
    u = str(u or ""); return u if re.match(r"^https?://", u, re.I) else ""

def err(where, e):
    msg = f"{where}: {type(e).__name__}: {str(e)[:200]}"; log("ERR", msg); report["errors"].append(msg)

# ---- worker api ----
def obj_get(key):
    r = S.get(SITE + "/api/atlas/cur/obj", params={"o": key}, headers=AUTH, timeout=300); return r.content if r.status_code == 200 else None
def obj_put(key, data, ctype="application/octet-stream"):
    r = S.put(SITE + "/api/atlas/cur/obj", params={"o": key}, data=data, headers={**AUTH, "content-type": ctype}, timeout=600); r.raise_for_status(); return r.json()
def post_json(path, body):
    r = S.post(SITE + path, json=body, headers=AUTH, timeout=300); r.raise_for_status(); return r.json()
def post_files(path, fields, files):
    r = S.post(SITE + path, files=[(n, (None, v)) for n, v in fields] + files, headers=AUTH, timeout=600); r.raise_for_status(); return r.json()

# ---- clip ----
_clip = None
def clip():
    global _clip
    if _clip is None:
        import torch, open_clip
        torch.set_num_threads(os.cpu_count() or 2)
        model, _, pre = open_clip.create_model_and_transforms(MODEL[0], pretrained=MODEL[1]); model.eval()
        _clip = (model, pre, open_clip.get_tokenizer(MODEL[0]), torch)
    return _clip
def embed_images(pils):
    model, pre, _, torch = clip(); out = []
    with torch.no_grad():
        for i in range(0, len(pils), 32):
            x = torch.stack([pre(p) for p in pils[i:i + 32]]); f = model.encode_image(x); f = f / f.norm(dim=-1, keepdim=True); out.append(f.cpu().numpy().astype(np.float32))
    return np.concatenate(out) if out else np.zeros((0, 512), np.float32)
def embed_texts(texts):
    model, _, tok, torch = clip()
    with torch.no_grad(): f = model.encode_text(tok(texts)); f = f / f.norm(dim=-1, keepdim=True)
    return f.cpu().numpy().astype(np.float32)

# ---- images ----
def load_image(b): im = Image.open(io.BytesIO(b)); im = ImageOps.exif_transpose(im); return im.convert("RGB")
def size_ok(im): w, h = im.size; s, l = min(w, h), max(w, h); return s >= MIN_SIDE and l / s <= MAX_ASPECT
def thumb_bytes(im): t = im.copy(); t.thumbnail((THUMB_W, THUMB_W * 3)); b = io.BytesIO(); t.save(b, "WEBP", quality=78, method=4); return b.getvalue()
def fetch(url, referer=None, timeout=25, headers=None):
    h = {"accept": "image/avif,image/webp,image/*,*/*;q=0.8", **(headers or {})}
    if referer: h["referer"] = referer
    r = S.get(url, headers=h, timeout=timeout, stream=True, allow_redirects=True); r.raise_for_status()
    ct = r.headers.get("content-type", "").split(";")[0].strip().lower()
    if not ct.startswith("image/"): raise ValueError("not an image: " + ct)
    b = r.raw.read(MAX_BYTES + 1, decode_content=True)
    if len(b) > MAX_BYTES: raise ValueError("too big")
    if len(b) < MIN_BYTES: raise ValueError("too small")
    return b

# ---- state ----
class State:
    def __init__(self):
        self.man = None; self.items = []; self.flags = {"ok": [], "dead": [], "moves": {}, "els": []}
        self.ids = []; self.E = np.zeros((0, 512), np.float32); self.text = {}
        self.killed = {}   # id -> style, every kill ever applied (embeddings stay cached → negative anchors)
        self.bundle = os.path.join(WORK, "bundle"); os.makedirs(os.path.join(self.bundle, "t"), exist_ok=True); os.makedirs(os.path.join(self.bundle, "el"), exist_ok=True)
    def index(self): return {i: k for k, i in enumerate(self.ids)}
    def ok(self): return set(self.flags.get("ok", []))

def sync(st):
    st.man = json.loads(obj_get("atlas/manifest.json") or '{"styles":{}}')
    for style, arr in st.man.get("styles", {}).items():
        if style == "inbox": continue
        for it in arr: st.items.append({"id": it["id"], "style": style, "key": it.get("key") or ("atlas/" + it["url"].split("/api/atlas/f/")[1]), "url": it["url"], "page": it.get("page") or "", "src": it.get("src") or "", "q": it.get("q") or ""})
    QSTATS.update(json.loads(obj_get("atlas/qstats.json") or "{}"))
    st.flags = S.get(SITE + "/api/atlas/flags", timeout=60).json()
    st.killed = json.loads(obj_get("atlas/killed.json") or "{}")
    raw = obj_get("atlas/clip.npz")
    if raw: z = np.load(io.BytesIO(raw), allow_pickle=False); st.ids = list(z["ids"]); st.E = z["E"].astype(np.float32)
    idx = obj_get("atlas/static.json")
    tars = [obj_get(p["key"]) for p in json.loads(idx)["parts"]] if idx else [obj_get("atlas/static.tar")]
    for tar in tars:
        if tar:
            with tarfile.open(fileobj=io.BytesIO(tar)) as tf: tf.extractall(st.bundle, filter="data")
    # provenance salvaged from the retired D1 table: fill in missing source links (sent with the next verdicts)
    try:
        prov = json.load(open(os.path.join(HERE, "provenance.json"))); st.prov = {}
        for it in st.items:
            if not it.get("page") and it["id"] in prov: it["page"] = prov[it["id"]]["page"]; it["src"] = prov[it["id"]]["src"]; st.prov[it["id"]] = {"page": it["page"], "src": it["src"]}
    except Exception: st.prov = {}
    report.update(items=len(st.items), cached=len(st.ids), kept=len(st.flags.get("ok", [])), user_kills=len(st.flags.get("dead", [])), user_moves=len(st.flags.get("moves", {})))
    log(f"sync: {len(st.items)} items, {len(st.ids)} embeddings, {len(os.listdir(os.path.join(st.bundle,'t')))} thumbs in bundle, flags ok={report['kept']} dead={report['user_kills']}")

def save_cache(st):
    b = io.BytesIO(); np.savez_compressed(b, ids=np.array(st.ids), E=st.E.astype(np.float16)); obj_put("atlas/clip.npz", b.getvalue())

def get_original(it):
    return load_image(S.get(SITE + it["url"], timeout=90).content)

def embed_missing(st):
    have = set(st.ids); todo = [it for it in st.items if it["id"] not in have]; log(f"embed: {len(todo)} new")
    def get(it):
        try: return it, get_original(it)
        except Exception as e: err("embed fetch " + it["id"], e); return it, None
    n = 0
    for i in range(0, len(todo), 64):
        with ThreadPoolExecutor(4) as ex: got = [g for g in ex.map(get, todo[i:i + 64]) if g[1] is not None]
        if not got: continue
        for it, im in got:   # thumbnails for free while the originals are in hand
            p = os.path.join(st.bundle, "t", it["id"] + ".webp")
            if not os.path.exists(p): open(p, "wb").write(thumb_bytes(im))
        E = embed_images([im for _, im in got]); st.ids += [it["id"] for it, _ in got]; st.E = np.concatenate([st.E, E]); n += len(got)
    if n: save_cache(st)
    report["embedded"] = n

def ai_score(st, E):
    if "_ai" not in st.text: st.text["_ai"] = embed_texts(AI_TEXT); st.text["_real"] = embed_texts(REAL_TEXT)
    return (E @ st.text["_ai"].T).max(axis=1) - (E @ st.text["_real"].T).max(axis=1)     # > 0 = reads as generated
def style_scores(st, E):
    idx = st.index(); ok = st.ok(); out = {}; ai = None
    for style, cfg in STYLES.items():
        if style not in st.text: st.text[style] = embed_texts(cfg["prompts"])
        ts = np.clip(((E @ st.text[style].T).max(axis=1) - 0.17) / 0.16, 0, 1)
        anchors = [idx[it["id"]] for it in st.items if it["style"] == style and it["id"] in ok and it["id"] in idx]
        sc = 0.5 * ts + 0.5 * np.clip(((E @ st.E[anchors].T).max(axis=1) - 0.55) / 0.35, 0, 1) if anchors else ts
        negs = [idx[i] for i, sty in st.killed.items() if sty == style and i in idx]
        if negs:   # looks like something you killed in this style → pushed down, hard when it is a near-copy
            neg = np.clip(((E @ st.E[negs].T).max(axis=1) - 0.6) / 0.3, 0, 1)
            sc = np.clip(sc - 0.7 * neg, 0, 1)
        if cfg.get("no_ai"):
            if ai is None: ai = ai_score(st, E)
            sc = np.clip(sc - np.clip((ai - 0.0) / 0.04, 0, 1) * 0.8, 0, 1)
        out[style] = sc
    return out

def score_archive(st):
    idx = st.index(); rows = [it for it in st.items if it["id"] in idx]
    if not rows: return {}, [], {}
    E = st.E[[idx[it["id"]] for it in rows]]; sc = style_scores(st, E); names = list(sc); M = np.stack([sc[n] for n in names], axis=1)
    ok = st.ok(); meta, low, move = {}, [], {}
    for r, it in enumerate(rows):
        own = float(sc[it["style"]][r]) if it["style"] in sc else 0.0
        b = int(M[r].argmax()); best, bs = names[b], float(M[r, b])
        meta[it["id"]] = {"cs": round(own, 3), "cb": best if (best != it["style"] and bs > own + MOVE_MARGIN) else None}
        if it["id"] in ok: continue
        if own < KILL_BELOW and bs < KILL_BELOW + 0.1: low.append(it["id"])
        elif best != it["style"] and bs > own + MOVE_MARGIN and bs >= UPLOAD_MIN: move[it["id"]] = best
    report["dist"] = {n: [round(float(x), 2) for x in np.percentile(sc[n][[k for k, it in enumerate(rows) if it["style"] == n]], [10, 25, 50, 75, 90])] for n in names if any(it["style"] == n for it in rows)}
    report.update(scored=len(rows), low_candidates=len(low), move_candidates=len(move))
    kill = []
    if APPLY_LOW_KILLS:
        per = {}; lowset = set(low)
        for it in rows:
            if it["id"] in lowset: per.setdefault(it["style"], []).append(it["id"])
        for sname, ids in per.items():
            total = sum(1 for it in rows if it["style"] == sname)
            if len(ids) > MAX_KILL_FRACTION * total: report.setdefault("guard", []).append(f"{sname}: {len(ids)}/{total} low — skipped")
            else: kill += ids
    if not APPLY_MOVES: move = {}
    log(f"score: {len(rows)} scored; low {len(low)} (applied {len(kill)}), moves {report['move_candidates']} (applied {len(move)})")
    return meta, kill, move

def dedupe(st, exclude):
    idx = st.index(); ok = st.ok(); rows = [it for it in st.items if it["id"] in idx and it["id"] not in exclude]
    if not rows: return []
    E = st.E[[idx[it["id"]] for it in rows]]; kill = set()
    for i in range(len(rows)):
        if rows[i]["id"] in kill: continue
        for j in np.where((E[i + 1:] @ E[i]) > DUPE)[0]:
            a, b = rows[i], rows[int(j) + i + 1]
            kill.add(a["id"] if (b["id"] in ok and a["id"] not in ok) else b["id"])
    report["killed_dupes"] = len(kill); log(f"dedupe: {len(kill)} near-duplicates")
    return list(kill)

def apply(st, meta, kill, move):
    dead = set(kill) | set(st.flags.get("dead", []))
    moves = dict(move); moves.update(st.flags.get("moves", {}))
    for id_, pv in getattr(st, "prov", {}).items(): meta[id_] = {**meta.get(id_, {}), **pv}
    r = post_json("/api/atlas/cur/verdicts", {"kill": sorted(dead), "move": moves, "meta": meta, "clearFlags": True})
    log("apply:", r); report["applied"] = r
    sty = {it["id"]: it["style"] for it in st.items}
    qof = {it["id"]: it.get("q") for it in st.items}
    tallied = set(json.loads(obj_get("atlas/qtallied.json") or "[]"))
    for id_ in list(dead) + list(st.ok()):
        q = qof.get(id_)
        if not q or id_ in tallied: continue
        QSTATS.setdefault(q, {"kept": 0, "killed": 0})["killed" if id_ in dead else "kept"] += 1; tallied.add(id_)
    obj_put("atlas/qstats.json", json.dumps(QSTATS), "application/json"); obj_put("atlas/qtallied.json", json.dumps(sorted(tallied)[-50000:]), "application/json")
    report["queries_retired"] = [q for q, v in QSTATS.items() if v["kept"] + v["killed"] >= RETIRE_AFTER and v["killed"] / (v["kept"] + v["killed"]) >= RETIRE_KILL_RATE]
    for id_ in dead: st.killed[id_] = sty.get(id_, st.killed.get(id_, "?"))
    obj_put("atlas/killed.json", json.dumps(st.killed), "application/json")
    report["negatives"] = len(st.killed)
    for id_ in dead:
        p = os.path.join(st.bundle, "t", id_ + ".webp")
        if os.path.exists(p): os.remove(p)
    st.items = [it for it in st.items if it["id"] not in dead]
    for it in st.items:
        if it["id"] in moves: it["style"] = moves[it["id"]]
        v = meta.get(it["id"])
        if v: it["cs"] = v["cs"]; it["cb"] = v["cb"]
    return dead

# ---- sourcing ----
QSTATS = {}   # query -> {"kept": n, "killed": n}; filled in sync(), updated in apply()
SOURCE_ONLY = set()   # --only free → image search runs for these styles only
def pick_queries(qs, n, run_no, prefix=""):
    """Rotate through the list, skipping retired queries; productive ones get a second slot."""
    live = []
    for q in qs:
        st_ = QSTATS.get(prefix + q, {}); k, d = st_.get("kept", 0), st_.get("killed", 0)
        if k + d >= RETIRE_AFTER and d / (k + d) >= RETIRE_KILL_RATE: continue
        live.append(q)
        if k + d >= 4 and k / (k + d) >= 0.6: live.append(q)
    if not live: return []
    return list(dict.fromkeys(live[(run_no * n + i) % len(live)] for i in range(min(n, len(live)))))
def src_search(style, cfg, run_no):
    try: from ddgs import DDGS
    except Exception as e: err("ddgs import", e); return [], {}
    out, status = [], {}
    for q in pick_queries(cfg.get("queries", []), int(cfg.get("qpr", QUERIES_PER_STYLE)), run_no):
        try:
            res = DDGS().images(q, max_results=RESULTS_PER_QUERY, safesearch="off") or []
            status[q] = len(res)
            for r in res:
                u = r.get("image") or ""
                try: w = int(r.get("width") or 0)
                except (TypeError, ValueError): w = 0
                if AI_HOSTS.search(u) or AI_HOSTS.search(r.get("url") or "") or AI_HOSTS.search(r.get("title") or ""): report["ai_blocked"] = report.get("ai_blocked", 0) + 1; continue
                if u.startswith("http") and not (w and w < MIN_SIDE): out.append((u, r.get("url") or "", {}, q))
            time.sleep(2.5)
        except Exception as e: status[q] = "fail: " + str(e)[:80]; time.sleep(6)
    return out, status
def src_caa(style, cfg, run_no):
    out, status = [], {}
    for q in pick_queries(cfg.get("caa", []), CAA_PER_STYLE, run_no, "caa: "):
        try:
            r = S.get("https://musicbrainz.org/ws/2/release-group/", params={"query": q, "fmt": "json", "limit": 60}, headers=MB_UA, timeout=60); r.raise_for_status()
            n = 0
            for g in r.json().get("release-groups", []):
                if g.get("id"): out.append(("https://coverartarchive.org/release-group/" + g["id"] + "/front-500", "https://musicbrainz.org/release-group/" + g["id"], {}, "caa: " + q)); n += 1
            status["caa: " + q] = n; time.sleep(1.2)
        except Exception as e: status["caa: " + q] = "fail: " + str(e)[:80]
    return out, status
def src_commons(style, cfg, run_no):
    out, status = [], {}
    for q in pick_queries(cfg.get("commons", []), int(cfg.get("cpr", COMMONS_PER_STYLE)), run_no, "commons: "):
        try:
            r = S.get("https://commons.wikimedia.org/w/api.php", params={"action": "query", "generator": "search", "gsrsearch": q, "gsrnamespace": 6, "gsrlimit": 40, "prop": "imageinfo", "iiprop": "url|size|mime", "iiurlwidth": 1400, "format": "json"}, headers=MB_UA, timeout=60); r.raise_for_status()
            n = 0
            for p in (r.json().get("query", {}).get("pages", {}) or {}).values():
                ii = (p.get("imageinfo") or [{}])[0]
                if re.match(r"image/(jpeg|png|webp)$", ii.get("mime", "")) and (ii.get("width") or 0) >= MIN_SIDE: out.append((ii.get("thumburl") or ii.get("url"), ii.get("descriptionurl", ""), {"headers": MB_UA}, "commons: " + q)); n += 1
            status["commons: " + q] = n
        except Exception as e: status["commons: " + q] = "fail: " + str(e)[:80]
    return out, status

def source(st, run_no):
    seen = set(json.loads(obj_get("atlas/seen.json") or "[]")); status = {}; found = downloaded = kept = 0; uploads = []
    for style, cfg in STYLES.items():
        if SOURCE_ONLY and style not in SOURCE_ONLY: continue
        cands = []
        for fn in (src_search, src_caa, src_commons):
            c, s_ = fn(style, cfg, run_no); status.update(s_); cands += [x for x in c if x[0] not in seen]
        for c in cands: seen.add(c[0])
        found += len(cands)
        def dl(c):
            try:
                b = fetch(c[0], referer=c[1], headers=c[2].get("headers")); im = load_image(b)
                return (c, b, im) if size_ok(im) else None
            except Exception: return None
        with ThreadPoolExecutor(16) as ex: got = [g for g in ex.map(dl, cands) if g]   # many slow hosts: wait on 16 at once
        downloaded += len(got)
        if not got: continue
        E = embed_images([g[2] for g in got]); sc = style_scores(st, E)[style]; n0 = len(uploads)
        idx = st.index(); mine = [idx[it["id"]] for it in st.items if it["style"] == style and it["id"] in idx]
        Es = st.E[mine] if mine else None; perq = {}
        order = sorted(range(len(got)), key=lambda k: -float(sc[k]))          # best first, so the cap keeps the strongest
        for k in order:
            c, b, im = got[k]
            if sc[k] < float(cfg.get("upload_min", UPLOAD_MIN)): continue
            if len(st.E) and float((st.E @ E[k]).max()) > DUPE: continue              # already have this picture
            same = float(cfg.get("same", SAME))
            if Es is not None and float((Es @ E[k]).max()) > same: continue           # more of the same for this style
            if any(float(u[4] @ E[k]) > same for u in uploads if u[0] == style): continue
            q = c[3] if len(c) > 3 else ""
            if perq.get(q, 0) >= int(cfg.get("per_query", PER_QUERY_CAP)): continue
            perq[q] = perq.get(q, 0) + 1
            uploads.append((style, b, c[0], c[1], E[k], im, q))
            if len(uploads) >= MAX_UPLOADS: break
        log(f"source {style}: {len(cands)} found, {len(got)} ok, {len(uploads)-n0} pass")
        if len(uploads) >= MAX_UPLOADS: break
    for i in range(0, len(uploads), 25):
        batch = uploads[i:i + 25]
        try:
            res = post_files("/api/atlas/cur/put", [("style", u[0]) for u in batch] + [("src", u[2]) for u in batch] + [("page", web(u[3])) for u in batch] + [("q", u[6]) for u in batch],
                             [("file", ("img", u[1], Image.open(io.BytesIO(u[1])).get_format_mimetype() or "image/jpeg")) for u in batch])
            for o in res.get("out", []):
                if not o.get("key"): continue
                k = o["i"]; u = batch[k]
                open(os.path.join(st.bundle, "t", o["id"] + ".webp"), "wb").write(thumb_bytes(u[5]))
                st.ids.append(o["id"]); st.E = np.concatenate([st.E, u[4][None]]); kept += 1
                st.items.append({"id": o["id"], "style": u[0], "key": o["key"], "url": "/api/atlas/f/" + o["key"][6:], "page": u[3], "src": u[2], "q": u[6]})
        except Exception as e: err("upload batch", e)
    if kept: save_cache(st)
    obj_put("atlas/seen.json", json.dumps(sorted(seen)[-30000:]), "application/json")
    report["source"] = {"queries": status, "found": found, "downloaded": downloaded, "uploaded": kept}

# ---- element candidates: your review decisions → library ----
def promote(st):
    pro, dis = st.flags.get("elok", []), st.flags.get("eldead", [])
    if not pro and not dis: return
    r = post_json("/api/atlas/cur/cand-apply", {"promote": pro, "discard": dis, "clearFlags": True})
    log("candidates:", r); report["candidates"] = r

# ---- map + palettes (feed the page's Map view and the swatches) ----
def project(st):
    """2-D layout of every image from its CLIP embedding (UMAP; PCA fallback) → map.json in the bundle."""
    idx = st.index(); rows = [it for it in st.items if it["id"] in idx]
    if len(rows) < 20: return
    X = st.E[[idx[it["id"]] for it in rows]]
    try:
        import umap, warnings; warnings.filterwarnings("ignore")
        Y = umap.UMAP(n_components=2, n_neighbors=25, min_dist=0.08, metric="cosine", random_state=7).fit_transform(X)
    except Exception as e:
        err("umap (pca fallback)", e)
        Xc = X - X.mean(0); U, S_, Vt = np.linalg.svd(Xc, full_matrices=False); Y = Xc @ Vt[:2].T
    Y = (Y - Y.min(0)) / (Y.max(0) - Y.min(0) + 1e-9)
    ok = st.ok()
    pts = [[it["id"], it["style"], round(float(Y[k, 0]), 4), round(float(Y[k, 1]), 4), 1 if it["id"] in ok else 0] for k, it in enumerate(rows)]
    json.dump({"v": 1, "built": int(time.time()), "n": len(pts), "pts": pts}, open(os.path.join(st.bundle, "map.json"), "w"), separators=(",", ":"))
    report["map"] = len(pts)

def palettes(st):
    """Dominant colours per style from the images you kept (fallback: the whole style), via k-means on thumbnails."""
    ok = st.ok(); out = {}
    for style in STYLES:
        its = [it for it in st.items if it["style"] == style and it["id"] in ok] or [it for it in st.items if it["style"] == style][:40]
        px = []
        for it in its[:60]:
            p = os.path.join(st.bundle, "t", it["id"] + ".webp")
            if not os.path.exists(p): continue
            try: im = Image.open(p).convert("RGB"); im.thumbnail((48, 48)); px.append(np.asarray(im, dtype=np.float32).reshape(-1, 3))
            except Exception: pass
        if not px: continue
        A = np.concatenate(px); rng = np.random.default_rng(1); C = A[rng.choice(len(A), 6, replace=False)]
        for _ in range(12):
            d = ((A[:, None, :] - C[None, :, :]) ** 2).sum(-1); lab = d.argmin(1)
            for j in range(6):
                if (lab == j).any(): C[j] = A[lab == j].mean(0)
        counts = np.bincount(lab, minlength=6); order = np.argsort(-counts)
        out[style] = ["#%02x%02x%02x" % tuple(int(v) for v in C[j]) for j in order]
    json.dump(out, open(os.path.join(st.bundle, "palette.json"), "w"))
    report["palettes"] = len(out)

# ---- static bundle ----
def bundle(st):
    ok = st.ok(); styles = {}
    for it in st.items:
        th = os.path.join(st.bundle, "t", it["id"] + ".webp")
        if not os.path.exists(th):
            try: open(th, "wb").write(thumb_bytes(get_original(it)))
            except Exception as e: err("thumb " + it["id"], e); continue
        styles.setdefault(it["style"], []).append({"id": it["id"], "th": "/atlas/t/" + it["id"] + ".webp", "url": it["url"], "page": web(it.get("page", "")), "ok": 1 if it["id"] in ok else 0, "cs": it.get("cs"), "cb": it.get("cb")})
    for arr in styles.values(): arr.sort(key=lambda x: (-(x["ok"]), -(x["cs"] if x["cs"] is not None else 0.5)))
    json.dump({"v": 3, "built": int(time.time()), "styles": styles}, open(os.path.join(st.bundle, "manifest.json"), "w"))
    # elements: the grid gets a small webp thumbnail (static); the full transparent PNG stays in R2, fetched on open/download
    els = json.loads(obj_get("atlas/elements.json") or '{"items":[]}').get("items", [])
    for e in els:
        p = os.path.join(st.bundle, "el", e["id"] + ".webp")
        if not os.path.exists(p):
            try:
                im = Image.open(io.BytesIO(S.get(SITE + "/api/atlas/el/" + e["id"], timeout=60).content)).convert("RGBA"); im.thumbnail((400, 400))
                im.save(p, "WEBP", quality=80, method=4)
            except Exception as ex: err("element " + e["id"], ex)
        e["th"] = "/atlas/el/" + e["id"] + ".webp"; e["url"] = "/api/atlas/el/" + e["id"]
    json.dump({"items": els}, open(os.path.join(st.bundle, "elements.json"), "w"))
    live_el = {e["id"] for e in els}
    for f in os.listdir(os.path.join(st.bundle, "el")):
        if (f.endswith(".webp") and f[:-5] not in live_el) or f.endswith(".png"): os.remove(os.path.join(st.bundle, "el", f))
    # montages (nightly edits from montage.py): mp4 + poster into the bundle, killed ones dropped from the site
    try:
        mont = json.loads(obj_get("atlas/mont.json") or '{"items":[]}').get("items", []); mdead = set(st.flags.get("mdead", [])); mok = set(st.flags.get("mok", [])); mdl = set(st.flags.get("mdl", [])); mseen = st.flags.get("mseen") or {}
        md = os.path.join(st.bundle, "mont"); os.makedirs(md, exist_ok=True); shown = []
        for m in mont:
            if m["id"] in mdead: continue
            if m["id"] not in mok and m["id"] not in mdl and time.time() - mseen.get(m["id"], 1e12) > 48 * 3600: continue   # watched, left: gone
            for ext, key in (("mp4", f"atlas/mont/{m['id']}.mp4"), ("jpg", f"atlas/mont/{m['id']}.jpg")):
                fp = os.path.join(md, f"{m['id']}.{ext}")
                if not os.path.exists(fp):
                    data = obj_get(key)
                    if data: open(fp, "wb").write(data)
            if os.path.exists(os.path.join(md, m["id"] + ".mp4")): m["ok"] = 1 if m["id"] in mok else 0; shown.append(m)
        keep_files = {f"{m['id']}.{e}" for m in shown for e in ("mp4", "jpg")}
        for f in os.listdir(md):
            if f not in keep_files: os.remove(os.path.join(md, f))
        json.dump({"items": shown}, open(os.path.join(st.bundle, "mont.json"), "w")); report["montages"] = len(shown)
    except Exception as e: err("montages", e)
    live = {it["id"] for it in st.items}
    for f in os.listdir(os.path.join(st.bundle, "t")):
        if f.endswith(".webp") and f[:-5] not in live: os.remove(os.path.join(st.bundle, "t", f))
    # Pack in parts of ≤ PART_MB (the Worker accepts ≤ 100 MB per request); atlas/static.json lists them.
    PART_MB = 45
    try: project(st)
    except Exception as e: err("map", e)
    try: palettes(st)
    except Exception as e: err("palettes", e)
    files = [("manifest.json", os.path.join(st.bundle, "manifest.json")), ("elements.json", os.path.join(st.bundle, "elements.json"))]
    for extra in ["map.json", "palette.json", "mont.json"]:
        if os.path.exists(os.path.join(st.bundle, extra)): files.append((extra, os.path.join(st.bundle, extra)))
    for sub in ["t", "el", "mont"]:
        if os.path.isdir(os.path.join(st.bundle, sub)):
            for f in sorted(os.listdir(os.path.join(st.bundle, sub))): files.append((sub + "/" + f, os.path.join(st.bundle, sub, f)))
    parts, buf, tf, size = [], None, None, 0
    def close_part():
        nonlocal buf, tf, size
        if tf is None: return
        tf.close(); data = buf.getvalue(); name = f"atlas/static-{len(parts)+1}.tar"; obj_put(name, data, "application/x-tar"); parts.append({"key": name, "bytes": len(data)}); buf = tf = None; size = 0
    for arc, path in files:
        n = os.path.getsize(path)
        if tf is not None and size + n > PART_MB * 1e6: close_part()
        if tf is None: buf = io.BytesIO(); tf = tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT)
        tf.add(path, arcname=arc); size += n + 512
    close_part()
    obj_put("atlas/static.json", json.dumps({"v": 1, "built": int(time.time()), "parts": parts}), "application/json")   # montage parts (static-9NN) are superseded: mont/ is in these
    report["bundle"] = {"images": sum(len(a) for a in styles.values()), "thumbs": len(os.listdir(os.path.join(st.bundle, "t"))), "elements": len(els), "parts": len(parts), "mb": round(sum(p["bytes"] for p in parts) / 1e6, 1)}
    log("bundle:", report["bundle"])

LIB_MAX = int(os.environ.get("LIB_MAX", "9000"))   # images kept on the site: stays well under the free plan's 20,000 static files and 10 GB
def budget(st, meta):
    """before searching, make room: drop the lowest-scoring images you haven't kept, never below 60 per style"""
    room = LIB_MAX - MAX_UPLOADS; ok = set(st.flags.get("ok", [])); dead = set(st.flags.get("dead", []))
    live = [it for it in st.items if it["id"] not in dead]
    if len(live) <= room: report["budget"] = f"{len(live)} images, limit {LIB_MAX}"; return []
    per = {}
    for it in live: per.setdefault(it["style"], []).append(it)
    cands = []
    for sty, arr in per.items():
        arr = sorted([it for it in arr if it["id"] not in ok], key=lambda it: (meta.get(it["id"], {}).get("cs") or it.get("cs") or 0))
        cands += arr[:max(0, len(per[sty]) - 60)]
    cands.sort(key=lambda it: (meta.get(it["id"], {}).get("cs") or it.get("cs") or 0)); drop = [it["id"] for it in cands[:len(live) - room]]
    report["budget"] = f"{len(live)} images, limit {LIB_MAX}: dropped {len(drop)} lowest-scoring unkept"; return drop

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--skip", action="append", default=[]); ap.add_argument("--run", type=int, default=int(time.time() // 86400)); ap.add_argument("--only", default=""); ap.add_argument("--qpr", type=int, default=0); a = ap.parse_args()
    if a.only:                                   # catch-up run: source only these styles (everything else still scored/bundled)
        global SOURCE_ONLY; SOURCE_ONLY = set(a.only.split(","))
    if a.qpr:
        for k in SOURCE_ONLY or STYLES: STYLES[k]["qpr"] = a.qpr
    if not TOKEN: sys.exit("ATLAS_TOKEN missing")
    st = State(); t0 = time.time()
    try:
        sync(st)
        if "embed" not in a.skip: embed_missing(st)
        meta, kill, move = ({}, [], {})
        if "score" not in a.skip: meta, kill, move = score_archive(st)
        dupes = [] if "dedupe" in a.skip else dedupe(st, set(kill) | set(st.flags.get("dead", [])))
        apply(st, meta, kill + dupes + budget(st, meta), move)
        if "source" not in a.skip: source(st, a.run)
        promote(st)
        if "bundle" not in a.skip: bundle(st)
    except Exception as e:
        err("run", e); traceback.print_exc()
    report["seconds"] = int(time.time() - t0)
    try: obj_put("atlas/curator.json", json.dumps(report), "application/json")
    except Exception as e: err("report upload", e)
    md = ["# Curator run " + time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), ""]
    for k, v in report.items():
        if k == "source": md.append(f"- source: found {v['found']}, downloaded {v['downloaded']}, uploaded {v['uploaded']}"); md += [f"  - `{q}` → {n}" for q, n in v["queries"].items()]
        elif k == "errors": md.append(f"- errors: {len(v)}"); md += [f"  - {e}" for e in v[:25]]
        else: md.append(f"- {k}: {v}")
    open(os.path.join(HERE, "last-run.md"), "w").write("\n".join(md) + "\n")
    log("done", json.dumps({k: v for k, v in report.items() if k not in ("source", "dist")}))

if __name__ == "__main__": main()
