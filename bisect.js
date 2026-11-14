/* 逐层 bisect：node 端重跑前向，与 python 每层输出对比。 */
globalThis.window = globalThis;
require("./lj-tokenizer.js");
require("./lj-cpu.js");
const fs = require("fs");
const path = require("path");

const dbg = JSON.parse(fs.readFileSync(path.join(__dirname, "debug_layers.json"), "utf-8"));
const manifest = JSON.parse(fs.readFileSync(path.join(__dirname, "weights", "manifest.json"), "utf-8"));
const buffers = manifest.parts.map(p => fs.readFileSync(path.join(__dirname, p.file)));
const w = new LJVWeights(manifest, buffers);
const cfg = manifest.cfg;
const fwd = new LJVForward(w, cfg);
const D = cfg.d_model;

function first8(arr, off) {
  return Array.from(arr.slice(off, off + 8)).map(v => +v.toFixed(6));
}
function diff(a, b) {
  let m = 0;
  for (let i = 0; i < Math.max(a.length, b.length); i++) m = Math.max(m, Math.abs((a[i] || 0) - (b[i] || 0)));
  return m;
}

const st = new LJVState(cfg);
// 逐 token 前向，但在最后一个位置逐层记录
const ids = dbg.ids;
const pos = ids.length - 1;
let resid = new Float32Array(D);
resid.set(w.get("tok_emb").subarray(ids[pos] * D, ids[pos] * D + D));
console.log("emb[-1,:8] js:", first8(resid, 0), "py:", dbg.emb, "diff", diff(first8(resid, 0), dbg.emb).toFixed(6));

// KV cache 需要先处理前面所有 token（不含最后一个），
// 且每个前面 token 也要完整走层——直接用 step 逐个跑，最后一个 token 手动展开。
for (let t = 0; t < pos; t++) {
  const x = new Float32Array(D);
  x.set(w.get("tok_emb").subarray(ids[t] * D, ids[t] * D + D));
  const lg = new Float32Array(cfg.vocab_size);
  fwd.step(x, st, lg);
}
// 最后一个 token：手动展开层
const H = cfg.n_head, hd = D / H, inter = cfg.inter;
const layerLast = [];
for (let l = 0; l < cfg.n_layer; l++) {
  const n1 = new Float32Array(D);
  fwd.rmsnorm(resid, w.get(`blocks.${l}.norm1.weight`), n1, 0);
  const qkv = new Float32Array(3 * D);
  fwd.gemvNT(w.get(`blocks.${l}.attn.qkv.weight`), 0, 3 * D, D, n1, qkv, 0);
  const q = new Float32Array(D), qk = new Float32Array(D);
  const kc = st.cache[l].k, vc = st.cache[l].v;
  fwd.ropeApply(qkv, 0, pos, qk, 0); q.set(qk);
  fwd.ropeApply(qkv, D, pos, qk, 0); kc.set(qk, pos * D);
  vc.set(qkv.subarray(2 * D, 3 * D), pos * D);
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
  fwd.gemvNT(w.get(`blocks.${l}.attn.o.weight`), 0, D, D, att, o, 0);
  for (let i = 0; i < D; i++) resid[i] += o[i];
  const n2 = new Float32Array(D);
  fwd.rmsnorm(resid, w.get(`blocks.${l}.norm2.weight`), n2, 0);
  const router = w.get(`blocks.${l}.moe.router.weight`);
  let best = 0, bp = -Infinity;
  for (let e = 0; e < cfg.n_expert; e++) {
    let s = 0;
    for (let i = 0; i < D; i++) s += n2[i] * router[e * D + i];
    if (s > bp) { bp = s; best = e; }
  }
  const h1 = new Float32Array(inter);
  fwd.gemvNT(w.get(`blocks.${l}.moe.w1_t`), best * D * inter, inter, D, n1 ? n1 : n2, h1, 0);
  for (let i = 0; i < inter; i++) h1[i] = (globalThis.geluRef)(h1[i]);
  const h2 = new Float32Array(D);
  fwd.gemvNT(w.get(`blocks.${l}.moe.w2_t`), best * inter * D, D, inter, h1, h2, 0);
  let ex = 0;
  for (let e = 0; e < cfg.n_expert; e++) {
    let s = 0;
    for (let i = 0; i < D; i++) s += n2[i] * router[e * D + i];
    ex += Math.exp(s - bp);
  }
  for (let i = 0; i < D; i++) resid[i] += h2[i] / ex;
  layerLast.push(first8(resid, 0));
}
console.log("block0 js:", layerLast[0], "py:", dbg.blocks[0],
  "diff", diff(layerLast[0], dbg.blocks[0]).toFixed(6));
for (let l = 0; l < cfg.n_layer; l++) {
  console.log("block%d diff %s", l, diff(layerLast[l], dbg.blocks[l]).toFixed(6));
}
