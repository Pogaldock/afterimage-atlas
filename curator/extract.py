#!/usr/bin/env python3
"""Auto cut-outs — finds the elements inside atlas images and lifts them out as transparent PNGs.

Per image (not yet processed, newest first, up to --cap):
  1. FastSAM "everything" segmentation → candidate regions (area/border/overlap filters)
  2. each region → crop + anti-aliased alpha from the mask; the one big main subject also gets matting-grade
     edges from ISNet (rembg) when it is large enough to be worth the CPU
  3. photocopy/stencil-type images (low saturation, bimodal luminance) also get an "ink" cut-out of the whole image
  4. CLIP names each cut-out (logo / lettering / person / face / object / symbol / illustration) and throws away
     textures, backgrounds and meaningless fragments; near-duplicate cut-outs are dropped
  5. survivors are uploaded as CANDIDATES (atlas/cand/…) — reviewed in the Cuts page; keeps join the library

State in R2: extracted.json (image id → n cut-outs), elclip.npz (cut-out embeddings), cands.json (Worker-side).
"""
import io, json, os, sys, time, argparse, hashlib, traceback
import numpy as np
import requests
from PIL import Image, ImageOps, ImageFilter

SITE = os.environ.get("SITE", "https://atlas.symbiozefpv.com").rstrip("/")
TOKEN = os.environ.get("ATLAS_TOKEN", "")
HERE = os.path.dirname(os.path.abspath(__file__))
S = requests.Session()
# Be polite to the site: back off on 429/5xx (honours Retry-After), few parallel connections, small pauses.
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
S.mount("https://", HTTPAdapter(max_retries=Retry(total=7, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=frozenset(["GET", "PUT", "POST"]), respect_retry_after_header=True, raise_on_status=False), pool_maxsize=6)); AUTH = {"authorization": "Bearer " + TOKEN}
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
report = {"ran": int(time.time()), "errors": []}
def err(where, e): msg = f"{where}: {type(e).__name__}: {str(e)[:160]}"; log("ERR", msg); report["errors"].append(msg)

# ---- tunables ----
SAM_MODEL, SAM_SIZE = "FastSAM-x.pt", 1024
AREA_MIN, AREA_MAX = 0.012, 0.62        # fraction of the image a region may cover
MIN_SIDE_PX = 110                       # of the region's bounding box
MAX_PER_IMAGE = 3
NEAR = 0.86                             # two cut-outs from one image this similar → keep only the better one
NEED_HISTORY = 40                       # once you have judged this many, your taste replaces the generic filter
REFINE_FROM = 0.10                      # regions above this area fraction get ISNet matting edges
DUPE = 0.95                             # cosine between cut-outs
MAX_SIDE = 1600
KINDS = {"a logo or emblem": "logo", "lettering, a word or typography": "lettering", "a person or figure": "person", "a face": "face",
         "a loudspeaker, speaker stack or sound system": "speaker", "a vinyl record, turntable or DJ decks": "vinyl", "a van, truck, bus or vehicle": "vehicle",
         "an object": "object", "a symbol or sigil": "symbol", "a drawing or illustration": "illustration",
         "a texture, background or pattern": None, "a blank, blurry or meaningless fragment": None}

def obj_get(key):
    r = S.get(SITE + "/api/atlas/cur/obj", params={"o": key}, headers=AUTH, timeout=300); return r.content if r.status_code == 200 else None
def obj_put(key, data, ctype="application/octet-stream"):
    r = S.put(SITE + "/api/atlas/cur/obj", params={"o": key}, data=data, headers={**AUTH, "content-type": ctype}, timeout=600); r.raise_for_status(); return r.json()

# ---- models (lazy) ----
_sam = _sess = _clip = None
def sam():
    global _sam
    if _sam is None:
        from ultralytics import FastSAM; _sam = FastSAM(SAM_MODEL)
    return _sam
def isnet():
    global _sess
    if _sess is None:
        from rembg import new_session; _sess = new_session("isnet-general-use")
    return _sess
def clip():
    global _clip
    if _clip is None:
        import torch, open_clip
        torch.set_num_threads(os.cpu_count() or 2)
        m, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k"); m.eval()
        tok = open_clip.get_tokenizer("ViT-B-32")
        with torch.no_grad(): T = m.encode_text(tok(list(KINDS))); T = T / T.norm(dim=-1, keepdim=True)
        _clip = (m, pre, torch, T.cpu().numpy().astype(np.float32))
    return _clip
def embed(pils):
    m, pre, torch, _ = clip()
    with torch.no_grad():
        x = torch.stack([pre(p) for p in pils]); f = m.encode_image(x); f = f / f.norm(dim=-1, keepdim=True)
    return f.cpu().numpy().astype(np.float32)

# ---- image helpers ----
def load_image(b): im = Image.open(io.BytesIO(b)); im = ImageOps.exif_transpose(im); return im.convert("RGB")
def on_grey(rgba):
    """composite over mid grey for CLIP (so alpha edges do not read as a white or black shape)"""
    bg = Image.new("RGB", rgba.size, (128, 128, 128)); bg.paste(rgba, mask=rgba.split()[3]); return bg
def trim(rgba, pad=2):
    a = np.asarray(rgba.split()[3]); ys, xs = np.where(a > 8)
    if not len(xs): return None
    x0, x1, y0, y1 = max(0, xs.min() - pad), min(rgba.width, xs.max() + 1 + pad), max(0, ys.min() - pad), min(rgba.height, ys.max() + 1 + pad)
    return rgba.crop((x0, y0, x1, y1))
def shrink(rgba):
    if max(rgba.size) > MAX_SIDE: rgba = rgba.copy(); rgba.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    return rgba
def inky(im):
    """photocopy / stencil / xerox: little colour, luminance piled at both ends"""
    hsv = np.asarray(im.convert("HSV")); sat = hsv[..., 1].mean() / 255; L = hsv[..., 2]
    h, _ = np.histogram(L, bins=16, range=(0, 255)); h = h / h.sum()
    return sat < 0.14 and (h[:4].sum() + h[-4:].sum()) > 0.6
def ink_cut(im):
    L = np.asarray(im.convert("L")).astype(np.float32)
    a = np.clip((150 - L) * 255 / 60, 0, 255).astype(np.uint8)     # dark = ink
    rgba = im.convert("RGBA"); rgba.putalpha(Image.fromarray(a)); return rgba

# ---- segmentation ----
def regions(im):
    r = sam()(im, device="cpu", retina_masks=True, imgsz=SAM_SIZE, conf=0.4, iou=0.9, verbose=False)[0]
    if r.masks is None: return []
    M = r.masks.data.numpy().astype(bool); H, W = M.shape[1:]; A = H * W
    out = []
    for m in M:
        area = m.sum() / A
        if area < AREA_MIN or area > AREA_MAX: continue
        ys, xs = np.where(m); x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        if (x1 - x0) < MIN_SIDE_PX or (y1 - y0) < MIN_SIDE_PX: continue
        touch = (x0 <= 2) + (y0 <= 2) + (x1 >= W - 2) + (y1 >= H - 2)
        if touch >= 3: continue                                         # that is the background, not an element
        out.append({"m": m, "box": (x0, y0, x1, y1), "area": area})
    out.sort(key=lambda d: -d["area"])
    keep = []
    for d in out:                                                       # drop near-identical masks and pieces of an already-kept region
        if any(((d["m"] & k["m"]).sum() / max(1, (d["m"] | k["m"]).sum())) > 0.8 for k in keep): continue
        if any((d["m"] & k["m"]).sum() / max(1, d["m"].sum()) > 0.7 for k in keep): continue   # mostly inside a bigger one
        keep.append(d)
        if len(keep) >= MAX_PER_IMAGE + 3: break
    return keep

def cut(im, d):
    x0, y0, x1, y1 = d["box"]; pad = 4
    x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(im.width, x1 + pad), min(im.height, y1 + pad)
    crop = im.crop((x0, y0, x1, y1)); mask = Image.fromarray((d["m"][y0:y1, x0:x1] * 255).astype(np.uint8))
    alpha = mask.filter(ImageFilter.GaussianBlur(1.1))                  # anti-aliased edge
    if d["area"] >= REFINE_FROM:                                        # main subject: matting-grade edge, bounded by the dilated SAM mask
        try:
            from rembg import remove
            fine = remove(crop, session=isnet(), only_mask=True).convert("L")
            dil = mask.filter(ImageFilter.MaxFilter(15))
            alpha = Image.fromarray((np.asarray(fine).astype(np.float32) * (np.asarray(dil) / 255.0)).astype(np.uint8))
        except Exception as e: err("isnet", e)
    rgba = crop.convert("RGBA"); rgba.putalpha(alpha)
    return rgba

# ---- main ----
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cap", type=int, default=int(os.environ.get("EXTRACT_CAP", "120"))); a = ap.parse_args()
    if not TOKEN: sys.exit("ATLAS_TOKEN missing")
    t0 = time.time()
    man = json.loads(obj_get("atlas/manifest.json") or '{"styles":{}}')
    done = json.loads(obj_get("atlas/extracted.json") or "{}")
    items = [dict(it, style=st) for st, arr in man.get("styles", {}).items() if st != "inbox" for it in arr]
    items.sort(key=lambda x: -(x.get("added") or 0))
    PRIORITY = {"free": 0.5}                     # share of each night's cap reserved for a style (free party first)
    left = [it for it in items if it["id"] not in done]; todo = []
    for sty, share in PRIORITY.items(): todo += [it for it in left if it["style"] == sty][:int(a.cap * share)]
    taken = {it["id"] for it in todo}; todo += [it for it in left if it["id"] not in taken][:a.cap - len(todo)]
    log(f"{len(items)} images, {len(done)} done, {len(todo)} this run")
    raw = obj_get("atlas/elclip.npz"); eids, E = ([], np.zeros((0, 512), np.float32))
    if raw: z = np.load(io.BytesIO(raw), allow_pickle=False); eids, E = list(z["ids"]), z["E"].astype(np.float32)
    # --- your taste: kept cut-outs are positives, discarded ones negatives (embeddings stay in elclip) ---
    els = json.loads(obj_get("atlas/elements.json") or '{"items":[]}').get("items", [])
    hist = json.loads(obj_get("atlas/elhist.json") or '{"ok":[],"dead":[]}')
    pending = {x["id"] for x in requests.get(SITE + "/api/atlas/cands", timeout=60).json().get("items", [])}
    pos_ids = {e["id"] for e in els} | set(hist.get("ok", []))
    neg_ids = set(hist.get("dead", [])) | ({i for i in eids} - pos_ids - pending)   # everything judged that is not in the library = discarded
    eidx = {i: k for k, i in enumerate(eids)}
    P = E[[eidx[i] for i in pos_ids if i in eidx]] if eids else np.zeros((0, 512), np.float32)
    Nn = E[[eidx[i] for i in neg_ids if i in eidx]] if eids else np.zeros((0, 512), np.float32)
    kind_n = {}
    for e in els: kind_n[e.get("kind") or "?"] = kind_n.get(e.get("kind") or "?", 0) + 1
    kmax = max(kind_n.values()) if kind_n else 1
    taste = len(P) >= NEED_HISTORY
    log(f"taste model: {len(P)} positives, {len(Nn)} negatives, kinds {kind_n} → {'on' if taste else 'off (generic filter)'}")
    report["taste"] = {"positives": int(len(P)), "negatives": int(len(Nn)), "on": taste}
    def taste_score(v, kind):
        pos = float((P @ v).max()) if len(P) else 0.0; neg = float((Nn @ v).max()) if len(Nn) else 0.0
        prior = (kind_n.get(kind, 0) + 1) / (kmax + 1)
        return pos - neg + 0.15 * prior, pos, neg
    batch, meta = [], []; made = 0; kinds = {}
    def flush():
        nonlocal batch, meta, made
        if not batch: return
        try:
            r = S.post(SITE + "/api/atlas/cur/cand", files=[("meta", (None, json.dumps(meta)))] + [("file", (m["id"] + ".png", b, "image/png")) for m, b in zip(meta, batch)], headers=AUTH, timeout=600); r.raise_for_status()
            made += r.json().get("added", 0)
        except Exception as e: err("upload", e)
        batch, meta = [], []
    for n, it in enumerate(todo):
        if time.time() - t0 > 50 * 60: log("time budget reached"); break
        try:
            im = load_image(S.get(SITE + it["url"], timeout=90).content)
            if max(im.size) > 2000: im.thumbnail((2000, 2000), Image.LANCZOS)
            cands = []
            for d in regions(im): cands.append((cut(im, d), "sam", float(d["area"])))
            if inky(im): cands.append((ink_cut(im), "ink", 1.0))
            pils = [trim(c[0]) for c in cands]; cands = [(p, c[1], c[2]) for p, c in zip(pils, cands) if p is not None and min(p.size) >= 48]
            count = 0
            if cands:
                V = embed([on_grey(p) for p, _, _ in cands]); _, _, _, T = clip(); PK = V @ T.T
                scored = []
                for k, (p, how, area) in enumerate(cands):
                    label = list(KINDS)[int(PK[k].argmax())]; kind = KINDS[label]
                    if kind is None and how != "ink": continue
                    if kind is None: kind = "ink"
                    if len(E) and float((E @ V[k]).max()) > DUPE: continue              # already produced this cut-out before
                    ts, pos, neg = taste_score(V[k], kind)
                    if taste:
                        if neg >= pos + 0.02: continue                                   # looks more like something you discarded
                        if pos < 0.42: continue                                          # like nothing you have ever kept
                    scored.append((ts, k, kind, how))
                scored.sort(reverse=True)
                chosen = []
                for ts, k, kind, how in scored:                                          # one per look-alike group, best first
                    if any(float(V[k] @ V[j]) > NEAR for _, j, _, _ in chosen): continue
                    chosen.append((ts, k, kind, how))
                    if len(chosen) >= MAX_PER_IMAGE: break
                for ts, k, kind, how in chosen:
                    p = shrink(cands[k][0]); buf = io.BytesIO(); p.save(buf, "PNG", optimize=True); b = buf.getvalue()
                    if len(b) > 9_000_000: continue
                    cid = hashlib.sha1(b).hexdigest()[:20]
                    batch.append(b); meta.append({"id": cid, "from": it["id"], "style": it["style"], "kind": kind, "score": round(float(ts), 3), "w": p.width, "h": p.height})
                    eids.append(cid); E = np.concatenate([E, V[k][None]]); kinds[kind] = kinds.get(kind, 0) + 1; count += 1
            done[it["id"]] = count
            if len(batch) >= 20: flush()
            if n % 10 == 0:
                log(f"{n+1}/{len(todo)} {it['style']} → {count} cut-outs ({int(time.time()-t0)}s)")
                try: obj_put("atlas/extractor.json", json.dumps({**report, "progress": f"{n+1}/{len(todo)}", "candidates": made + len(batch), "kinds": kinds, "seconds": int(time.time() - t0)}), "application/json")
                except Exception: pass
        except Exception as e: err("image " + it["id"], e); done[it["id"]] = -1
    flush()
    try:
        obj_put("atlas/extracted.json", json.dumps(done), "application/json")
        b = io.BytesIO(); np.savez_compressed(b, ids=np.array(eids), E=E.astype(np.float16)); obj_put("atlas/elclip.npz", b.getvalue())
    except Exception as e: err("state", e)
    report.update(images=len(todo), candidates=made, kinds=kinds, remaining=max(0, len(items) - len(done)), seconds=int(time.time() - t0))
    try: obj_put("atlas/extractor.json", json.dumps(report), "application/json")
    except Exception as e: err("report", e)
    md = ["# Extractor run " + time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), ""] + [f"- {k}: {v}" for k, v in report.items() if k != "errors"] + [f"- errors: {len(report['errors'])}"] + [f"  - {e}" for e in report["errors"][:20]]
    open(os.path.join(HERE, "last-extract.md"), "w").write("\n".join(md) + "\n")
    log("done", json.dumps({k: v for k, v in report.items() if k != "errors"}))

if __name__ == "__main__":
    try: main()
    except BaseException as e:
        tb = traceback.format_exc()
        try: obj_put("atlas/extractor.json", json.dumps({**report, "crashed": str(e)[:200], "traceback": tb[-1500:]}), "application/json")
        except Exception: pass
        raise
