// Runs at deploy (wrangler build.command). Downloads the nightly static bundle from R2 through this Worker's curator API and
// unpacks it into public/atlas/ so thumbnails, manifest and elements ship as static assets (free, no Worker calls).
// Never fails the deploy: on any error the site deploys with whatever is already in public/atlas/.
import fs from "node:fs";
import path from "node:path";
const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
const out = path.join(root, "public", "atlas");
const site = process.env.ATLAS_SITE || "https://atlas.symbiozefpv.com";
try {
  const tok = process.env.ATLAS_TOKEN;   // Workers Builds build variable (never in the repo)
  if (!tok) throw new Error("no ATLAS_TOKEN build variable");
  const get = (key) => fetch(site + "/api/atlas/cur/obj?o=" + key, { headers: { authorization: "Bearer " + tok } });
  const ir = await get("atlas/static.json");
  let keys;
  if (ir.ok) keys = (await ir.json()).parts.map((p) => p.key);
  else if (ir.status === 404) keys = ["atlas/static.tar"];
  else throw new Error("index " + ir.status);
  let n = 0;
  for (const key of keys) {
  const r = await get(key);
  if (r.status === 404) { console.log("atlas-assets: missing", key); continue; }
  if (!r.ok) throw new Error("fetch " + key + " " + r.status);
  const buf = Buffer.from(await r.arrayBuffer());
  let off = 0;
  while (off + 512 <= buf.length) {
    const hdr = buf.subarray(off, off + 512);
    if (hdr.every((b) => b === 0)) break;
    const name = hdr.subarray(0, 100).toString("utf8").replace(/\0.*$/s, "");
    const prefix = hdr.subarray(345, 500).toString("utf8").replace(/\0.*$/s, "");
    const size = parseInt(hdr.subarray(124, 136).toString("utf8").replace(/\0.*$/s, "").trim() || "0", 8);
    const type = String.fromCharCode(hdr[156]);
    const full = (prefix ? prefix + "/" : "") + name;
    off += 512;
    const SHAPE = /^((manifest|elements|map|palette|mont)\.json|(t|el)\/[a-f0-9]{20}\.webp|mont\/[a-f0-9]{20}\.(mp4|jpg))$/;   // never html/js/svg
    if ((type === "0" || type === "\0") && full && !full.includes("..") && SHAPE.test(full)) {
      const dest = path.join(out, full);
      fs.mkdirSync(path.dirname(dest), { recursive: true });
      fs.writeFileSync(dest, buf.subarray(off, off + size)); n++;
    }
    off += Math.ceil(size / 512) * 512;
  }
  }
  console.log("atlas-assets: unpacked", n, "files from", keys.length, "part(s) into public/atlas/");
} catch (e) {
  console.log("atlas-assets: skipped —", e && e.message);
}
process.exit(0);
