import { isAdmin } from "./auth.js";
// Afterimage atlas — the thin part that lives on Cloudflare.
//
// Design rule: page views cost nothing. Thumbnails, the manifest and the element library are STATIC ASSETS
// (public/atlas/…, unpacked at deploy time from one tar in R2 by scripts/atlas-assets.mjs). Static asset requests
// never invoke this Worker. What remains here is a tiny write path and the full-size originals:
//
//   GET  /api/atlas/flags           {ok:[], dead:[], moves:{}, els:[...]}   public, cached — live changes since the last build
//   POST /api/atlas/flags           {ids:[], unkeep:[], kill:[], move:{id:style}}   admin (cookie or key), batched per session
//   POST /api/atlas/el              multipart file(png) + from, style, kind, tag   admin — one cut-out element
//   POST /api/atlas/el/delete       {id}                                           admin
//   GET  /api/atlas/el/<id>         element png (R2; the static copy takes over after the next build)
//   GET  /api/atlas/f/<style>/<id>.<ext>  full-size original (R2) — lightbox / cut only
//   GET  /api/atlas/stats           public health line
//   /api/atlas/cur/*                curator (GitHub Action) endpoints, Bearer ATLAS_TOKEN: obj get/put, put images, verdicts
//
// Nothing here touches D1, Workers AI or Images. Crawling, scoring, dedupe and thumbnails run in curator/ on GitHub.

const STYLES = ["inbox","crt","datamosh","analog","y2k","aero","vapor","weird","hyper","digi","witch","drain","mound","opium","goth","scene","black","death","sigil","punk","free","techno","digital","tech","factory","war","riot","drone","robot","alien","psych","meme","pop"];
const CORS = { "access-control-allow-origin": "*", "access-control-allow-methods": "POST, GET, PUT, OPTIONS", "access-control-allow-headers": "content-type, authorization" };
const json = (d, status = 200, extra = {}) => new Response(JSON.stringify(d), { status, headers: { "content-type": "application/json", ...CORS, ...extra } });
const EXT = { "image/jpeg": "jpg", "image/png": "png", "image/webp": "webp", "image/gif": "gif", "image/avif": "avif" };
const now = () => Math.floor(Date.now() / 1000);
async function sha1(buf) { const h = await crypto.subtle.digest("SHA-1", buf); return [...new Uint8Array(h)].map((b) => b.toString(16).padStart(2, "0")).join(""); }

// admin = the signed session cookie from /admin/home (auth.js); the curator job = the atlas bearer token.
// No raw password is accepted here any more (no body key, no raw-key cookie).
async function okKey(env, _key, req) {
  return (await isAdmin(req, env)) || okTok(env, req);
}
function safeEq(a, b) {
  const x = new TextEncoder().encode(String(a)), y = new TextEncoder().encode(String(b));
  if (x.length !== y.length) return false;
  let d = 0; for (let i = 0; i < x.length; i++) d |= x[i] ^ y[i]; return d === 0;
}
// served bytes are never interpreted as a document on this origin
const SAFE = { "x-content-type-options": "nosniff", "content-security-policy": "default-src 'none'; sandbox" };
const IMG_KEY = /^atlas\/[a-z0-9]+\/[a-f0-9]{20}\.(jpg|png|webp|gif|avif)$/;
function okTok(env, req) {
  const h = req.headers.get("authorization") || "";
  const t = h.startsWith("Bearer ") ? h.slice(7).trim() : "";
  return !!t && !!env.ATLAS_TOKEN && safeEq(t, env.ATLAS_TOKEN);
}

// ---------- small JSON objects in R2 ----------
async function readJson(env, key, fallback) { const o = await env.STORE.get(key); if (o) { try { const j = await o.json(); if (j) return j; } catch {} } return fallback; }
const putJson = (env, key, v) => env.STORE.put(key, JSON.stringify(v), { httpMetadata: { contentType: "application/json" } });
// read-modify-write that can't lose a concurrent update: the write only lands if nobody wrote in between, else re-read and redo
async function casJson(env, key, fallback, mutate) {
  for (let i = 0; i < 10; i++) {
    const o = await env.STORE.get(key); let v = JSON.parse(JSON.stringify(fallback));
    if (o) { try { v = (await o.json()) || v; } catch {} }
    const out = await mutate(v);
    const opts = { httpMetadata: { contentType: "application/json" } }; if (o) opts.onlyIf = { etagMatches: o.etag };
    if (await env.STORE.put(key, JSON.stringify(out), opts)) return out;
    await new Promise((r) => setTimeout(r, 30 + Math.random() * 90 * (i + 1)));
  }
  throw new Error("busy, try again");
}
const MKEY = "atlas/manifest.json", FKEY = "atlas/flags.json", EKEY = "atlas/elements.json";
const emptyFlags = () => ({ ok: [], dead: [], moves: {}, els: [] });

// The master manifest (what is stored). The curator rebuilds/extends it; the page reads the static copy.
async function readManifest(env) {
  const m = await readJson(env, MKEY, null);
  if (m && m.styles) return m;
  return rebuildManifest(env);
}
async function rebuildManifest(env) {
  const styles = {}; let cursor;
  do {
    const page = await env.STORE.list({ prefix: "atlas/", cursor, limit: 1000 });
    for (const o of page.objects) {
      const mm = /^atlas\/([a-z0-9-]+)\/([a-f0-9]{20})\.[a-z0-9]+$/.exec(o.key);
      if (!mm || !STYLES.includes(mm[1])) continue;
      (styles[mm[1]] ||= []).push({ id: mm[2], key: o.key, url: "/api/atlas/f/" + o.key.slice(6), src: "", page: "", added: Math.floor(new Date(o.uploaded).getTime() / 1000) });
    }
    cursor = page.truncated ? page.cursor : undefined;
  } while (cursor);
  for (const k of Object.keys(styles)) styles[k].sort((a, b) => b.added - a.added);
  const m = { v: 3, styles, count: 0, built: now() };
  await writeManifest(env, m); return m;
}
async function writeManifest(env, m) { m.v = 3; m.count = Object.values(m.styles).reduce((a, b) => a + b.length, 0); await putJson(env, MKEY, m); }

// ---------- flags: the only thing a triage session writes ----------
async function flagsGet(env) {
  const [f, els] = await Promise.all([readJson(env, FKEY, emptyFlags()), readJson(env, EKEY, { items: [] })]);
  return json({ ok: f.ok || [], dead: f.dead || [], moves: f.moves || {}, elok: f.elok || [], eldead: f.eldead || [], mok: f.mok || [], mdead: f.mdead || [], mseen: f.mseen || {}, mdl: f.mdl || [], mwhy: f.mwhy || {}, els: els.items || [] }, 200, { "cache-control": "public, max-age=20" });
}
async function flagsPost(env, req) {
  const b = await req.json().catch(() => null);
  if (!b || !(await okKey(env, b.key, req))) return json({ error: "not logged in" }, 403);
  const idok = (x) => /^[a-f0-9]{20}$/.test(String(x));
  // element candidates (auto cut-outs): promote to the library / discard, applied straight away
  const pro = (b.elok || []).filter(idok).map(String), dis = (b.eldead || []).filter(idok).map(String);
  let cand = null, candFailed = false;
  if (pro.length || dis.length) { try { cand = await applyCands(env, pro, dis); } catch (e) { cand = { error: String(e && e.message || e) }; candFailed = true; } }
  const f = await casJson(env, FKEY, {}, (f0) => {
    const f = Object.assign(emptyFlags(), f0);
    const ok = new Set(f.ok), dead = new Set(f.dead);
    for (const id of b.ids || []) if (idok(id)) { ok.add(String(id)); dead.delete(String(id)); }
    for (const id of b.unkeep || []) ok.delete(String(id));
    for (const id of b.kill || []) if (idok(id)) { dead.add(String(id)); ok.delete(String(id)); }
    for (const id of b.unkill || []) dead.delete(String(id));
    f.moves = f.moves || {}; for (const [id, st] of Object.entries(b.move || {})) if (idok(id) && STYLES.includes(st)) f.moves[String(id)] = st;
    const eok = new Set(f.elok || []), edead = new Set(f.eldead || []);       // only kept if applying them failed: retried by the next call or the curator
    if (candFailed) { for (const id of pro) { eok.add(id); edead.delete(id); } for (const id of dis) { edead.add(id); eok.delete(id); } }
    else { for (const id of pro) eok.delete(id); for (const id of dis) edead.delete(id); }
    for (const id of b.unelok || []) eok.delete(String(id));
    for (const id of b.uneldead || []) edead.delete(String(id));
    f.elok = [...eok]; f.eldead = [...edead];
    // montages: mok = keep, mdead = delete
    const mok = new Set(f.mok || []), mdead = new Set(f.mdead || []);
    for (const id of b.mok || []) if (idok(id)) { mok.add(String(id)); mdead.delete(String(id)); }
    for (const id of b.unmok || []) mok.delete(String(id));
    for (const id of b.mdead || []) if (idok(id)) { mdead.add(String(id)); mok.delete(String(id)); }
    for (const id of b.unmdead || []) mdead.delete(String(id));
    f.mok = [...mok]; f.mdead = [...mdead];
    // watched edits: mseen = when he last played one through, mdl = downloaded (never pruned)
    const ms = (f.mseen && typeof f.mseen === "object") ? f.mseen : {}, mdl = new Set(f.mdl || []);
    for (const id of b.mseen || []) if (idok(id)) ms[String(id)] = now();
    for (const id of b.mdl || []) if (idok(id)) mdl.add(String(id));
    const ids = Object.keys(ms); if (ids.length > 3000) for (const k of ids.sort((a, c) => ms[a] - ms[c]).slice(0, ids.length - 3000)) delete ms[k];
    f.mseen = ms; f.mdl = [...mdl].slice(-3000);
    // why an edit was deleted: song, video, both, or none (downloaded first: just tidying)
    const why = (f.mwhy && typeof f.mwhy === "object") ? f.mwhy : {};
    if (b.mwhy && typeof b.mwhy === "object") for (const [id, v] of Object.entries(b.mwhy)) if (idok(id) && ["song", "video", "both", "none"].includes(v)) why[String(id)] = v;
    const wk = Object.keys(why); if (wk.length > 3000) for (const k of wk.slice(0, wk.length - 3000)) delete why[k];
    f.mwhy = why;
    f.ok = [...ok]; f.dead = [...dead]; f.updated = now();
    return f;
  });
  return json({ ok: true, kept: f.ok.length, dead: f.dead.length, moves: Object.keys(f.moves).length, cand });
}

// ---------- elements: cut-outs for his own artwork ----------
async function elPost(env, req) {
  if (!(await okKey(env, null, req))) return json({ error: "not logged in" }, 403);     // auth before reading the body
  const fd = await req.formData();
  const f = fd.get("file"); if (!f || typeof f !== "object") return json({ error: "no file" }, 400);
  const buf = await f.arrayBuffer(); if (buf.byteLength > 12 * 1024 * 1024 || buf.byteLength < 200) return json({ error: "bad size" }, 400);
  const id = (await sha1(buf)).slice(0, 20);
  const style = STYLES.includes(fd.get("style")) ? fd.get("style") : "inbox";
  const it = { id, from: String(fd.get("from") || "").slice(0, 20), style, kind: String(fd.get("kind") || "cut").slice(0, 12), tag: String(fd.get("tag") || "").slice(0, 80),
    w: +fd.get("w") || 0, h: +fd.get("h") || 0, added: now(), url: "/api/atlas/el/" + id };
  await env.STORE.put("atlas/el/" + id + ".png", buf, { httpMetadata: { contentType: "image/png" } });
  const e = await readJson(env, EKEY, { items: [] });
  if (!e.items.some((x) => x.id === id)) e.items.unshift(it);
  await putJson(env, EKEY, e);
  return json({ ok: true, el: it });
}
async function elDelete(env, req) {
  const b = await req.json().catch(() => null);
  if (!b || !(await okKey(env, b.key, req))) return json({ error: "not logged in" }, 403);
  const e = await readJson(env, EKEY, { items: [] });
  e.items = e.items.filter((x) => x.id !== String(b.id)); await putJson(env, EKEY, e);
  await env.STORE.delete("atlas/el/" + String(b.id) + ".png").catch(() => {});
  return json({ ok: true });
}
async function elGet(env, id) {
  const o = await env.STORE.get("atlas/el/" + id + ".png");
  if (!o) return new Response("gone", { status: 404 });
  return new Response(o.body, { headers: { "content-type": "image/png", "cache-control": "public, max-age=31536000, immutable", ...SAFE, ...CORS } });
}

// ---------- element candidates (auto cut-outs awaiting review) ----------
const CKEY = "atlas/cands.json";
async function candsGet(env) {
  const c = await readJson(env, CKEY, { items: [] });
  return json(c, 200, { "cache-control": "public, max-age=20" });
}
async function candGet(env, id) {
  const o = await env.STORE.get("atlas/cand/" + id + ".png");
  if (!o) return new Response("gone", { status: 404 });
  return new Response(o.body, { headers: { "content-type": "image/png", "cache-control": "public, max-age=31536000, immutable", ...SAFE, ...CORS } });
}
// multipart: meta (JSON array, one object per file: {id, from, style, kind, score, w, h}) + file[] in the same order
async function candPut(env, req) {
  const fd = await req.formData();
  let meta = []; try { meta = JSON.parse(fd.get("meta") || "[]"); } catch {}
  const files = fd.getAll("file").slice(0, 200);
  const c = await readJson(env, CKEY, { items: [] }); const have = new Set(c.items.map((x) => x.id));
  let n = 0;
  for (let i = 0; i < files.length; i++) {
    const m = meta[i] || {}; const f = files[i]; if (!f || typeof f !== "object" || !/^[a-f0-9]{20}$/.test(String(m.id))) continue;
    if (have.has(m.id)) continue;
    const buf = await f.arrayBuffer(); if (buf.byteLength > 8 * 1024 * 1024) continue;
    await env.STORE.put("atlas/cand/" + m.id + ".png", buf, { httpMetadata: { contentType: "image/png" } });
    c.items.unshift({ id: m.id, from: String(m.from || "").slice(0, 20), style: STYLES.includes(m.style) ? m.style : "inbox", kind: String(m.kind || "").slice(0, 16), score: +m.score || 0, w: +m.w || 0, h: +m.h || 0, added: now(), url: "/api/atlas/cand/" + m.id });
    have.add(m.id); n++;
  }
  c.items = c.items.slice(0, 6000);
  await putJson(env, CKEY, c);
  return json({ ok: true, added: n, total: c.items.length });
}
// Promote reviewed candidates into the library and drop discarded ones. Called the moment a review decision
// arrives (flagsPost) and again by the curator as a backstop.
async function applyCands(env, promote, discard) {
  const ok = new Set((promote || []).map(String)), dead = new Set((discard || []).map(String));
  const c0 = await readJson(env, CKEY, { items: [] });
  if (!ok.size && !dead.size) return { promoted: 0, discarded: 0, pending: c0.items.length };
  const byId = new Map(c0.items.map((it) => [it.id, it])); const moved = [];
  for (const id of ok) {                                                  // copy the cut-out into the library (safe to repeat)
    const it = byId.get(id); if (!it) continue;
    const o = await env.STORE.get("atlas/cand/" + id + ".png");
    if (o) { await env.STORE.put("atlas/el/" + id + ".png", o.body, { httpMetadata: { contentType: "image/png" } }); }
    moved.push(it);
  }
  await casJson(env, EKEY, { items: [] }, (e) => {
    for (const it of moved) if (!e.items.some((x) => x.id === it.id)) e.items.unshift({ id: it.id, from: it.from, style: it.style, kind: it.kind, tag: it.kind, w: it.w, h: it.h, auto: 1, added: now(), url: "/api/atlas/el/" + it.id });
    return e;
  });
  let pending = 0;
  await casJson(env, CKEY, { items: [] }, (c) => { c.items = c.items.filter((it) => !ok.has(it.id) && !dead.has(it.id)); pending = c.items.length; return c; });
  for (const id of [...ok, ...dead]) await env.STORE.delete("atlas/cand/" + id + ".png").catch(() => {});
  await casJson(env, "atlas/elhist.json", { ok: [], dead: [] }, (h) => {
    const hok = new Set(h.ok), hdead = new Set(h.dead); for (const id of ok) { hok.add(id); hdead.delete(id); } for (const id of dead) { hdead.add(id); hok.delete(id); }
    h.ok = [...hok]; h.dead = [...hdead]; return h;
  });
  return { promoted: moved.length, discarded: dead.size, pending };
}
async function candApply(env, req) {
  const b = await req.json().catch(() => null); if (!b) return json({ error: "bad json" }, 400);
  const r = await applyCands(env, b.promote || [], b.discard || []);
  if (b.clearFlags) { const ok = new Set(b.promote || []), dead = new Set(b.discard || []); await casJson(env, FKEY, {}, (f0) => { const f = Object.assign(emptyFlags(), f0); f.elok = (f.elok || []).filter((id) => !ok.has(id)); f.eldead = (f.eldead || []).filter((id) => !dead.has(id)); return f; }); }
  return json({ ok: true, ...r });
}

// ---------- originals ----------
// the archive: every edit you've seen, kept small (720p) so it never fills the free storage. Served from R2 with ranges so it streams.
async function archFile(env, req, id, ext) {
  const key = "atlas/arch/" + id + "." + ext, rng = req.headers.get("range");
  let range; const m = rng && /^bytes=(\d+)-(\d*)$/.exec(rng);
  if (m) range = m[2] ? { offset: +m[1], length: +m[2] - +m[1] + 1 } : { offset: +m[1] };
  const o = await env.STORE.get(key, range ? { range } : {});
  if (!o) return new Response("gone", { status: 404 });
  const h = { "content-type": ext === "mp4" ? "video/mp4" : "image/jpeg", "accept-ranges": "bytes", "cache-control": "public, max-age=31536000, immutable", ...SAFE };
  if (range && o.range) { const off = o.range.offset || 0, len = o.range.length ?? (o.size - off); h["content-range"] = `bytes ${off}-${off + len - 1}/${o.size}`; h["content-length"] = String(len); return new Response(o.body, { status: 206, headers: h }); }
  h["content-length"] = String(o.size); return new Response(o.body, { headers: h });
}
async function file(env, rest) {
  const o = await env.STORE.get("atlas/" + rest);
  if (!o) return new Response("gone", { status: 404 });
  const ct = o.httpMetadata?.contentType || "image/jpeg";
  return new Response(o.body, { headers: { "content-type": /^image\/(jpeg|png|webp|gif|avif)$/.test(ct) ? ct : "application/octet-stream", "cache-control": "public, max-age=31536000, immutable", ...SAFE, ...CORS } });
}

const strip = (o) => { if (!o || typeof o !== "object") return o; const c = { ...o }; delete c.traceback; delete c.crashed; delete c.errors; return c; };
async function stats(env) {
  try {
    const [m, f, cur, e, c, ex] = await Promise.all([readManifest(env), readJson(env, FKEY, emptyFlags()), readJson(env, "atlas/curator.json", null), readJson(env, EKEY, { items: [] }), readJson(env, CKEY, { items: [] }), readJson(env, "atlas/extractor.json", null)]);
    const per = {}; for (const k of Object.keys(m.styles)) per[k] = m.styles[k].length;
    return json({ images: m.count, per, kept: (f.ok || []).length, dead: (f.dead || []).length, moves: Object.keys(f.moves || {}).length, elements: e.items.length, candidates: c.items.length, curator: strip(cur) || "no run yet", extractor: strip(ex) || "no run yet" }, 200, { "cache-control": "no-store" });
  } catch (err) { return json({ error: String(err && err.message || err) }, 503); }
}

// ---------- curator endpoints (GitHub Action) ----------
const OBJ_OK = /^atlas\/(clip\.(json|npz)|elclip\.npz|curator\.json|extractor\.json|flags\.json|manifest\.json|elements\.json|cands\.json|extracted\.json|seen\.json|killed\.json|qstats\.json|qtallied\.json|elhist\.json|static\.tar|static-\d+\.tar|static\.json|mont\.json|montage\.json|mont\/[a-z0-9]+\.(mp4|jpg)|arch\/[a-z0-9]+\.(mp4|jpg)|arch\.json|vid\.json|vid\/raw\/[a-f0-9]{20}\.(mp4|mov|webm|m4v)|vid\/clip\/[a-f0-9]{20}-\d{1,3}\.(mp4|jpg))$/;
async function objGet(env, key) {
  if (!OBJ_OK.test(key)) return json({ error: "key not allowed" }, 400);
  const o = await env.STORE.get(key);
  if (!o) return new Response("", { status: 404 });
  return new Response(o.body, { headers: { "content-type": o.httpMetadata?.contentType || "application/octet-stream", "cache-control": "no-store" } });
}
async function objPut(env, req, key) {
  if (!OBJ_OK.test(key)) return json({ error: "key not allowed" }, 400);
  const body = await req.arrayBuffer();
  if (body.byteLength > (/\.tar$/.test(key) ? 60 : 30) * 1024 * 1024) return json({ error: "too big" }, 413);
  await env.STORE.put(key, body, { httpMetadata: { contentType: req.headers.get("content-type") || "application/octet-stream" } });
  return json({ ok: true, bytes: body.byteLength });
}
// ---------- your own videos: resumable multipart upload straight into R2 (admin only) ----------
// The edits job later cuts each one into short 720p clips (atlas/vid/clip/) and deletes the original, so storage stays small.
const VKEY = "atlas/vid.json", VID_RAW = /^atlas\/vid\/raw\/[a-f0-9]{20}\.(mp4|mov|webm|m4v)$/, RAW_CAP = 4 * 1024 ** 3;
const VEXT = { "video/mp4": "mp4", "video/quicktime": "mov", "video/webm": "webm", "video/x-m4v": "m4v" };
async function vidStart(env, req) {
  if (!(await okKey(env, null, req))) return json({ error: "not logged in" }, 403);
  const b = await req.json().catch(() => ({})); const ext = VEXT[b.type] || (/\.(mp4|mov|webm|m4v)$/i.exec(b.name || "") || [])[1]?.toLowerCase();
  if (!ext) return json({ error: "mp4, mov or webm only" }, 400);
  const size = +b.size || 0; if (size < 1000 || size > 3 * 1024 ** 3) return json({ error: "file too big (3 GB max)" }, 413);
  const v = await readJson(env, VKEY, { items: [] }); const pending = v.items.filter((x) => x.state === "raw").reduce((a, x) => a + (x.size || 0), 0);
  if (pending + size > RAW_CAP) return json({ error: "4 GB of videos are already waiting to be cut; the next batch clears them" }, 429);
  const id = [...crypto.getRandomValues(new Uint8Array(10))].map((x) => x.toString(16).padStart(2, "0")).join("");
  const key = "atlas/vid/raw/" + id + "." + ext; const up = await env.STORE.createMultipartUpload(key, { httpMetadata: { contentType: b.type || "video/mp4" } });
  return json({ id, key, up: up.uploadId });
}
async function vidPart(env, req, url) {
  if (!(await okKey(env, null, req))) return json({ error: "not logged in" }, 403);
  const key = url.searchParams.get("key") || "", up = url.searchParams.get("up") || "", n = +url.searchParams.get("n");
  if (!VID_RAW.test(key) || !up || !(n >= 1 && n <= 10000)) return json({ error: "bad part" }, 400);
  const len = +req.headers.get("content-length") || 0; if (len > 100 * 1024 * 1024) return json({ error: "part too big" }, 413);
  const part = await env.STORE.resumeMultipartUpload(key, up).uploadPart(n, req.body);
  return json({ n: part.partNumber, etag: part.etag });
}
async function vidDone(env, req) {
  if (!(await okKey(env, null, req))) return json({ error: "not logged in" }, 403);
  const b = await req.json().catch(() => ({}));
  if (!VID_RAW.test(b.key || "") || !b.up || !Array.isArray(b.parts) || !b.parts.length) return json({ error: "bad request" }, 400);
  const parts = b.parts.map((x) => ({ partNumber: +x.n, etag: String(x.etag) })).sort((a, c) => a.partNumber - c.partNumber);
  const obj = await env.STORE.resumeMultipartUpload(b.key, b.up).complete(parts);
  const id = b.key.split("/").pop().split(".")[0];
  const it = { id, key: b.key, name: String(b.name || "").slice(0, 120), size: obj.size, added: now(), state: "raw", clips: [] };
  await casJson(env, VKEY, { items: [] }, (v) => { v.items = (v.items || []).filter((x) => x.id !== id); v.items.unshift(it); return v; });
  return json({ ok: true, it });
}
async function vidAbort(env, req) {
  if (!(await okKey(env, null, req))) return json({ error: "not logged in" }, 403);
  const b = await req.json().catch(() => ({})); if (!VID_RAW.test(b.key || "") || !b.up) return json({ error: "bad request" }, 400);
  try { await env.STORE.resumeMultipartUpload(b.key, b.up).abort(); } catch {} return json({ ok: true });
}

// Multipart upload of vetted images: style[], src[], page[], file[] in matching order.
async function putMany(env, req) {
  const fd = await req.formData();
  const files = fd.getAll("file"), styles = fd.getAll("style"), srcs = fd.getAll("src"), pages = fd.getAll("page"), qs = fd.getAll("q");
  const m = await readManifest(env);
  const have = new Set(); for (const k of Object.keys(m.styles)) for (const it of m.styles[k]) have.add(it.id);
  const out = [];
  for (let i = 0; i < files.length; i++) {
    const f = files[i]; const style = String(styles[i] || styles[0] || "inbox");
    if (!STYLES.includes(style) || !f || typeof f !== "object") { out.push({ i, error: "bad style/file" }); continue; }
    const ctype = (f.type || "application/octet-stream").toLowerCase();
    if (!EXT[ctype]) { out.push({ i, error: "not an image" }); continue; }
    const buf = await f.arrayBuffer(); if (buf.byteLength > 25 * 1024 * 1024) { out.push({ i, error: "too big" }); continue; }
    const id = (await sha1(buf)).slice(0, 20);
    if (have.has(id)) { out.push({ i, id, dup: true }); continue; }
    const key = `atlas/${style}/${id}.${EXT[ctype]}`;
    await env.STORE.put(key, buf, { httpMetadata: { contentType: ctype } });
    (m.styles[style] ||= []).unshift({ id, key, url: "/api/atlas/f/" + key.slice(6), src: String(srcs[i] || "").slice(0, 2000), page: String(pages[i] || "").slice(0, 2000), q: String(qs[i] || "").slice(0, 120), added: now() });
    have.add(id); out.push({ i, id, key });
  }
  await writeManifest(env, m);
  return json({ ok: true, stored: out.filter((o) => o.key).length, out });
}
// Curator verdicts: kills (R2 delete + manifest), moves (manifest re-file), and the flags it has applied get cleared.
async function verdicts(env, req) {
  const b = await req.json().catch(() => null);
  if (!b) return json({ error: "bad json" }, 400);
  const kill = new Set((b.kill || []).map(String)), move = b.move || {};
  let killed = 0, moved = 0;
  const m = await readManifest(env);
  for (const k of Object.keys(m.styles)) {
    const keep = [];
    for (const it of m.styles[k]) {
      if (kill.has(it.id)) { if (IMG_KEY.test(String(it.key))) await env.STORE.delete(it.key).catch(() => {}); killed++; continue; }
      const to = move[it.id]; if (to && to !== k && STYLES.includes(to)) { it.style = to; (m.styles[to] ||= []).unshift(it); moved++; continue; }
      keep.push(it);
    }
    m.styles[k] = keep;
  }
  if (b.meta) for (const k of Object.keys(m.styles)) for (const it of m.styles[k]) {
    const v = b.meta[it.id]; if (!v || typeof v !== "object") continue;
    for (const f of ["src", "page", "q", "tag"]) if (typeof v[f] === "string") it[f] = v[f].slice(0, 2000);
    for (const f of ["cs", "cb"]) if (typeof v[f] === "number" || v[f] === null) it[f] = v[f];
    if (typeof v.cb === "string" && STYLES.includes(v.cb)) it.cb = v.cb;
  }
  await writeManifest(env, m);
  if (b.clearFlags) {
    await casJson(env, FKEY, {}, (f0) => { const f = Object.assign(emptyFlags(), f0); f.dead = f.dead.filter((id) => !kill.has(id)); f.moves = f.moves || {}; for (const id of Object.keys(move)) delete f.moves[id]; return f; });
  }
  return json({ ok: true, killed, moved, count: m.count });
}

export async function handleAtlas(req, env) {
  const url = new URL(req.url); const p = url.pathname;
  if (req.method === "OPTIONS") return new Response(null, { status: 204, headers: CORS });
  if (p === "/api/atlas/flags" && req.method === "GET") return flagsGet(env);
  if ((req.method === "POST" || req.method === "PUT") && /^\/api\/atlas\/(flags|el|el\/delete|vid\/(start|part|done|abort))$/.test(p) && !okTok(env, req)) {
    const o = req.headers.get("origin"); if (o !== url.origin) return json({ error: "wrong origin" }, 403);
  }
  if (p === "/api/atlas/flags" && req.method === "POST") return flagsPost(env, req);
  if (p === "/api/atlas/el" && req.method === "POST") return elPost(env, req);
  if (p === "/api/atlas/vid/start" && req.method === "POST") return vidStart(env, req);
  if (p === "/api/atlas/vid/part" && req.method === "PUT") return vidPart(env, req, url);
  if (p === "/api/atlas/vid/done" && req.method === "POST") return vidDone(env, req);
  if (p === "/api/atlas/vid/abort" && req.method === "POST") return vidAbort(env, req);
  if (p === "/api/atlas/vid" && req.method === "GET") {
    if (!(await okKey(env, null, req))) return json({ error: "not logged in" }, 403);
    return json(await readJson(env, VKEY, { items: [] }), 200, { "cache-control": "no-store" });
  }
  if (p === "/api/atlas/el/delete" && req.method === "POST") return elDelete(env, req);
  if (p === "/api/atlas/stats" && req.method === "GET") {
    const ck = new Request(url.origin + "/api/atlas/stats"), hit = await caches.default.match(ck); if (hit) return hit;
    const r = await stats(env); const out = new Response(r.body, r); out.headers.set("cache-control", "public, max-age=60"); await caches.default.put(ck, out.clone()); return out;
  }
  if (p === "/api/atlas/cands" && req.method === "GET") return candsGet(env);
  if (p === "/api/atlas/me" && req.method === "GET") return json({ admin: await okKey(env, null, req) }, 200, { "cache-control": "no-store" });
  if (p.startsWith("/api/atlas/cur/")) {
    if (!okTok(env, req)) return json({ error: "no" }, 403);
    const sub = p.slice("/api/atlas/cur/".length);
    if (sub === "obj" && req.method === "GET") return objGet(env, (url.searchParams.get("o") || url.searchParams.get("key") || ""));
    if (sub === "obj" && req.method === "PUT") return objPut(env, req, (url.searchParams.get("o") || url.searchParams.get("key") || ""));
    if (sub === "obj" && req.method === "DELETE") {   // only pruned edits can be deleted, nothing else
      const k = url.searchParams.get("o") || ""; if (!(/^atlas\/(mont|arch)\/[a-z0-9]+\.(mp4|jpg)$/.test(k) || VID_RAW.test(k))) return json({ error: "key not allowed" }, 400);
      await env.STORE.delete(k); return json({ ok: true });
    }
    if (sub === "put" && req.method === "POST") return putMany(env, req);
    if (sub === "verdicts" && req.method === "POST") return verdicts(env, req);
    if (sub === "cand" && req.method === "POST") return candPut(env, req);
    if (sub === "cand-apply" && req.method === "POST") return candApply(env, req);
    if (sub === "rebuild" && req.method === "POST") { const m = await rebuildManifest(env); return json({ ok: true, count: m.count }); }
    // one-off migration from the old site's bucket: originals, elements and candidates only, real image types only
    if (sub === "raw" && req.method === "PUT") {
      const key = url.searchParams.get("o") || "";
      if (!(IMG_KEY.test(key) || /^atlas\/(el|cand)\/[a-f0-9]{20}\.png$/.test(key))) return json({ error: "key not allowed" }, 400);
      const ct = req.headers.get("content-type") || ""; if (!/^image\/(jpeg|png|webp|gif|avif)$/.test(ct)) return json({ error: "type not allowed" }, 400);
      const body = await req.arrayBuffer(); if (body.byteLength > 25 * 1024 * 1024) return json({ error: "too big" }, 413);
      await env.STORE.put(key, body, { httpMetadata: { contentType: ct } }); return json({ ok: true, bytes: body.byteLength });
    }
    return json({ error: "unknown" }, 404);
  }
  let m;
  if ((m = /^\/api\/atlas\/el\/([a-f0-9]{20})$/.exec(p)) && req.method === "GET") return elGet(env, m[1]);
  if ((m = /^\/api\/atlas\/cand\/([a-f0-9]{20})$/.exec(p)) && req.method === "GET") return candGet(env, m[1]);
  if ((m = /^\/api\/atlas\/arch\/([a-f0-9]{20})\.(mp4|jpg)$/.exec(p)) && req.method === "GET") return archFile(env, req, m[1], m[2]);
  if (p === "/api/atlas/arch" && req.method === "GET") { const a = await readJson(env, "atlas/arch.json", { items: [] }); return json(a, 200, { "cache-control": "public, max-age=60" }); }
  if ((m = /^\/api\/atlas\/f\/([a-z0-9-]+\/[a-f0-9]{20}\.[a-z0-9]+)$/.exec(p)) && req.method === "GET") return file(env, m[1]);
  if ((m = /^\/api\/atlas\/t\/([a-f0-9]{20})$/.exec(p)) && req.method === "GET") { // legacy thumb URL → static
    return Response.redirect(url.origin + "/atlas/t/" + m[1] + ".webp", 302);
  }
  return null;
}
