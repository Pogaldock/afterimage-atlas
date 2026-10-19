// Afterimage Atlas — standalone Worker. Static page + thumbnails are assets (free); this script only runs for /api/atlas/* and /admin.
// It has its own R2 bucket (STORE) and its own secrets (ATLAS_TOKEN for the jobs, ATLAS_ADMIN_KEY for the owner). Nothing else.
import { handleAtlas } from "./atlas.js";
import { sessionCookie, checkKey } from "./auth.js";

const SEC = { "x-content-type-options": "nosniff", "strict-transport-security": "max-age=31536000", "referrer-policy": "strict-origin-when-cross-origin", "x-frame-options": "DENY" };
const html = (body, status = 200, extra = {}) => new Response(body, { status, headers: { "content-type": "text/html; charset=utf-8", "cache-control": "no-store", "x-robots-tag": "noindex", ...SEC, ...extra } });
const page = (msg = "") => `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Afterimage admin</title></head>
<body style="background:#07070c;color:#eee;font:15px ui-monospace,monospace;display:grid;place-items:center;min-height:100vh;margin:0">
<form method="post" action="/admin" style="display:grid;gap:12px;width:min(360px,90vw)"><b>AFTERIMAGE // ADMIN</b>
<input name="key" type="password" autocomplete="current-password" placeholder="admin key" style="padding:12px;background:#111;color:#eee;border:1px solid #333;border-radius:8px">
<button style="padding:12px;background:#b8ff2e;color:#000;border:0;border-radius:8px;font:inherit;font-weight:700">ENTER</button>${msg ? `<span style="color:#ff6a6a">${msg}</span>` : ""}</form></body></html>`;

export default {
  async fetch(req, env) {
    const url = new URL(req.url);
    if (url.pathname.startsWith("/api/atlas/")) {
      const r = await handleAtlas(req, env);
      if (r) { const h = new Headers(r.headers); for (const [k, v] of Object.entries(SEC)) if (!h.has(k)) h.set(k, v); return new Response(r.body, { status: r.status, headers: h }); }
      return new Response("not found", { status: 404, headers: SEC });
    }
    if (url.pathname === "/admin") {
      if (req.method === "POST") {
        if ((req.headers.get("origin") || url.origin) !== url.origin) return html(page("wrong origin"), 403);
        const len = +(req.headers.get("content-length") || 0); if (len > 4096) return html(page("too big"), 413);
        const fd = await req.formData().catch(() => null);
        if (!checkKey(env, fd && fd.get("key"))) { await new Promise((r) => setTimeout(r, 800)); return html(page("no"), 403); }
        return new Response(null, { status: 303, headers: { location: "/atlas/", "set-cookie": await sessionCookie(env), "cache-control": "no-store", ...SEC } });
      }
      return html(page());
    }
    if (url.pathname === "/" || url.pathname === "") return Response.redirect(url.origin + "/atlas/", 302);
    return env.ASSETS.fetch(req);
  },
};
