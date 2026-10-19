// Atlas admin auth, standalone (this Worker shares nothing with the main site).
// One way in: POST /admin with ATLAS_ADMIN_KEY → signed session cookie adm=<exp>.<hmac>; the cookie never holds the key.
// Changing ATLAS_ADMIN_KEY signs everyone out.
export const SESSION_AGE = 30 * 86400;
const enc = new TextEncoder();
const now = () => Math.floor(Date.now() / 1000);
export function safeEqual(a, b) {
  const x = enc.encode(String(a ?? "")), y = enc.encode(String(b ?? ""));
  let d = x.length ^ y.length; const n = Math.max(x.length, y.length);
  for (let i = 0; i < n; i++) d |= (x[i] | 0) ^ (y[i] | 0);
  return d === 0;
}
async function sign(env, exp) {
  const k = await crypto.subtle.importKey("raw", enc.encode(env.ATLAS_ADMIN_KEY), { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  const sig = await crypto.subtle.sign("HMAC", k, enc.encode("atlas-session:v1:" + exp));
  return [...new Uint8Array(sig)].map((b) => b.toString(16).padStart(2, "0")).join("");
}
export async function sessionCookie(env) {
  const exp = now() + SESSION_AGE;
  return `adm=${exp}.${await sign(env, exp)}; Max-Age=${SESSION_AGE}; Path=/; HttpOnly; Secure; SameSite=Strict`;
}
export async function isAdmin(req, env) {
  if (!env.ATLAS_ADMIN_KEY) return false;
  const m = (req.headers.get("cookie") || "").match(/(?:^|;\s*)adm=(\d{9,11})\.([a-f0-9]{64})(?:;|$)/);
  if (!m) return false;
  const exp = parseInt(m[1], 10), t = now();
  if (!(exp > t) || exp > t + SESSION_AGE + 300) return false;
  return safeEqual(await sign(env, exp), m[2]);
}
export const checkKey = (env, given) => !!env.ATLAS_ADMIN_KEY && typeof given === "string" && given.length >= 24 && safeEqual(given, env.ATLAS_ADMIN_KEY);
