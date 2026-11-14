/* node 校验：分词器 vs python tokenizers；CPU 前向 vs python 参考。 */
globalThis.window = globalThis;
require("./lj-tokenizer.js");
require("./lj-cpu.js");
const fs = require("fs");
const path = require("path");

const ROOT = __dirname;
const WEB = path.join(ROOT, "webexport");

async function main() {
  const ref = JSON.parse(fs.readFileSync(path.join(__dirname, "ref_logits.json"), "utf-8"));
  const manifest = JSON.parse(fs.readFileSync(path.join(__dirname, "weights", "manifest.json"), "utf-8"));
  const buffers = [];
  for (const p of manifest.parts) {
    buffers.push(fs.readFileSync(path.join(__dirname, p.file)));
  }
  const tokJson = fs.readFileSync(path.join(__dirname, "tokenizer.json"), "utf-8");
  const tok = new LJVTokenizer(tokJson);

  // 1) 分词器
  const prompt = "<|user|>" + ref.prompt + "<|end|><|assistant|>";
  const ids = tok.encode(prompt);
  const match = JSON.stringify(ids) === JSON.stringify(ref.ids);
  console.log("[tokenizer]", match ? "PASS" : "FAIL",
    "\n  js :", ids, "\n  py :", ref.ids);
  if (!match) {
    // 定位第一个不同的位置
    for (let i = 0; i < Math.max(ids.length, ref.ids.length); i++) {
      if (ids[i] !== ref.ids[i]) { console.log("  first diff @", i, ids[i], ref.ids[i]); break; }
    }
  }

  // 2) 权重 + 前向
  const w = new LJVWeights(manifest, buffers);
  const cfg = manifest.cfg;
  const fwd = new LJVForward(w, cfg);
  const st = new LJVState(cfg);
  const logits = new Float32Array(cfg.vocab_size);
  for (const id of ref.ids) fwd.step(w.get("tok_emb").subarray(id * cfg.d_model, (id + 1) * cfg.d_model), st, logits);

  // 3) 对比
  const top = Array.from(logits).map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 8);
  const okTop = top.every((x, i) => x[1] === ref.top_ids[i] && Math.abs(x[0] - ref.top_vals[i]) < 0.15);
  const hiddenErr = 0;
  console.log("[forward]", okTop ? "PASS" : "FAIL");
  console.log("  js top :", top.map(x => [x[1], +x[0].toFixed(4)]));
  console.log("  py top :", ref.top_ids.map((id, i) => [id, ref.top_vals[i]]));
  if (!okTop) process.exit(1);
  console.log("ALL OK");
}
main().catch(e => { console.error(e); process.exit(1); });
