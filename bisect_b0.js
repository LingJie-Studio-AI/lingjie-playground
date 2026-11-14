/* block0 内部逐算子对照（浅层定位）。 */
globalThis.window = globalThis;
require("./lj-tokenizer.js");
require("./lj-cpu.js");
const fs = require("fs");
const path = require("path");

const dbg = JSON.parse(fs.readFileSync(path.join(__dirname, "debug_b0.json"), "utf-8"));
const dl = JSON.parse(fs.readFileSync(path.join(__dirname, "debug_layers.json"), "utf-8"));
const manifest = JSON.parse(fs.readFileSync(path.join(__dirname, "weights", "manifest.json"), "utf-8"));
const buffers = manifest.parts.map(p => fs.readFileSync(path.join(__dirname, p.file)));
const w = new LJVWeights(manifest, buffers);
const cfg = manifest.cfg;
const fwd = new LJVForward(w, cfg);
const D = cfg.d_model, H = cfg.n_head, hd = D / H, inter = cfg.inter;
const ids = dl.ids, pos = ids.length - 1;

function f8(a, o) { return Array.from(a.slice(o, o + 8)).map(v => +v.toFixed(6)); }
function diff(a, b) {
  let m = 0;
  for (let i = 0; i < 8; i++) m = Math.max(m, Math.abs((a[i] || 0) - (b[i] || 0)));
  return m.toFixed(6);
}
function show(name, js, py) {
  console.log(name, "js:", f8(js, 0), "py:", py.slice(0, 8), "diff", diff(f8(js, 0), py));
}

// KV cache 先填前 9 个 token（用 fwd.step）
const st = new LJVState(cfg);
const lg = new Float32Array(cfg.vocab_size);
for (let t = 0; t < pos; t++) {
  const x = new Float32Array(D);
  x.set(w.get("tok_emb").subarray(ids[t] * D, ids[t] * D + D));
  fwd.step(x, st, lg);
}
// 最后一个 token 的 block0
let resid = new Float32Array(D);
resid.set(w.get("tok_emb").subarray(ids[pos] * D, ids[pos] * D + D));
const n1 = new Float32Array(D);
fwd.rmsnorm(resid, w.get("blocks.0.norm1.weight"), n1, 0);
show("norm1", n1, dbg.norm1);

const qkv = new Float32Array(3 * D);
fwd.gemvNT(w.get("blocks.0.attn.qkv.weight"), 0, 3 * D, D, n1, qkv, 0);
const q = new Float32Array(D), qk = new Float32Array(D);
const kc = st.cache[0].k, vc = st.cache[0].v;
fwd.ropeApply(qkv, 0, pos, qk, 0); q.set(qk);
fwd.ropeApply(qkv, D, pos, qk, 0); kc.set(qk, pos * D);
vc.set(qkv.subarray(2 * D, 3 * D), pos * D);
// q 的排列：py 是 [H,T,8] flatten → 头主序 h0 的前 8 个维度
show("q_rope", q, dbg.q_rope);
// k_rope py: kr[0,:,-1,:8].flatten → [H,8] → h0 前 8 维 = cache 行 pos 的前 8
show("k_rope", kc.subarray(pos * D), dbg.k_rope);

const att = new Float32Array(D);
const scale = 1 / Math.sqrt(hd);
for (let h = 0; h < H; h++) {
  const scores = new Float32Array(pos + 1);
  let mx = -Infinity;
  for (let t = 0; t <= pos; t++) {
    let s = 0; const base = t * D + h * hd;
    for (let d = 0; d < hd; d++) s += q[h * hd + d] * kc[base + d];
    s *= scale; scores[t] = s; if (s > mx) mx = s;
  }
  let sum = 0;
  for (let t = 0; t <= pos; t++) scores[t] = Math.exp(scores[t] - mx), sum += scores[t];
  for (let d = 0; d < hd; d++) {
    let s = 0;
    for (let t = 0; t <= pos; t++) s += scores[t] * vc[t * D + h * hd + d];
    att[h * hd + d] = s / sum;
  }
}
const o = new Float32Array(D);
fwd.gemvNT(w.get("blocks.0.attn.o.weight"), 0, D, D, att, o, 0);
show("attn_o", o, dbg.attn_o);
for (let i = 0; i < D; i++) resid[i] += o[i];
const n2 = new Float32Array(D);
fwd.rmsnorm(resid, w.get("blocks.0.norm2.weight"), n2, 0);
const router = w.get("blocks.0.moe.router.weight");
let best = 0, bp = -Infinity;
for (let e = 0; e < cfg.n_expert; e++) {
  let s = 0;
  for (let i = 0; i < D; i++) s += n2[i] * router[e * D + i];
  if (s > bp) { bp = s; best = e; }
}
console.log("expert", best, "logit", bp.toFixed(4));
const h1 = new Float32Array(inter);
fwd.gemvNT(w.get("blocks.0.moe.w1_t"), best * D * inter, inter, D, n2, h1, 0);
for (let i = 0; i < inter; i++) h1[i] = (globalThis.geluRef)(h1[i]);
const h2 = new Float32Array(D);
fwd.gemvNT(w.get("blocks.0.moe.w2_t"), best * inter * D, D, inter, h1, h2, 0);
let ex = 0;
for (let e = 0; e < cfg.n_expert; e++) {
  let s = 0;
  for (let i = 0; i < D; i++) s += n2[i] * router[e * D + i];
  ex += Math.exp(s - bp);
}
for (let i = 0; i < D; i++) resid[i] += h2[i] / ex;
show("moe_out", h2.map(v => v / ex), dbg.moe_out);
show("blk_out", resid, dbg.blk_out);
