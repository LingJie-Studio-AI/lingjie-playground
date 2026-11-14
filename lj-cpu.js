/* LingJieLV-Chat-v2-Flash CPU 前向（fp32，与 python 参考逐算子一致）。
 * 该实现同时作为 WebGPU 不可用时的回退路径。
 * 数学：RMSNorm / 交错 RoPE / 因果 MHA / MoE top-1(GELU 精确式) / 绑定输出头。
 */
(function (root) {
  const erf = function (x) {
    // Abramowitz-Stegun 7.1.26，误差 <3e-7
    const sgn = x < 0 ? -1 : 1;
    x = Math.abs(x);
    const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741,
      a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
    const t = 1 / (1 + p * x);
    const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
    return sgn * y;
  };
  const gelu = function (x) { return 0.5 * x * (1 + erf(x * Math.SQRT1_2)); };
  const quickGelu = function (x) { return x / (1 + Math.exp(-1.702 * x)); };

  class LJVWeights {
    /* manifest + 原始字节 -> 名字到 Float32Array 的映射（fp16 解码） */
    constructor(manifest, buffers) {
      this.t = {};
      let u8full = null;
      if (buffers.length === 1) u8full = buffers[0];
      else {
        u8full = new Uint8Array(manifest.total_bytes);
        let off = 0;
        for (const b of buffers) { u8full.set(new Uint8Array(b), off); off += b.byteLength; }
      }
      const u16 = new Uint16Array(u8full.buffer, u8full.byteOffset);
      for (const [name, info] of Object.entries(manifest.tensors)) {
        const n = info.bytes >> 1;
        const start = info.offset >> 1;
        const arr = new Float32Array(n);
        for (let i = 0; i < n; i++) {
          const h = u16[start + i];
          const sgn = (h & 0x8000) >> 15, ex = (h & 0x7c00) >> 10, fr = h & 0x03ff;
          let val;
          if (ex === 0) val = fr * Math.pow(2, -24);
          else if (ex === 31) val = fr ? NaN : Infinity * (sgn ? -1 : 1);
          else val = (1 + fr / 1024) * Math.pow(2, ex - 15) * (sgn ? -1 : 1);
          arr[i] = val;
        }
        this.t[name] = { data: arr, shape: info.shape };
      }
      this.get = (n) => this.t[n].data;
      this.shape = (n) => this.t[n].shape;
    }
  }

  class LJVState {
    /* 单序列推理状态：KV cache + 位置 */
    constructor(cfg) {
      this.cfg = cfg;
      this.cache = [];
      for (let l = 0; l < cfg.n_layer; l++) {
        this.cache.push({
          k: new Float32Array(cfg.max_seq * cfg.d_model),
          v: new Float32Array(cfg.max_seq * cfg.d_model),
        });
      }
      this.pos = 0;
      this.reset();
    }
    reset() {
      this.pos = 0;
      for (const c of this.cache) c.k.fill(0), c.v.fill(0);
    }
  }

  class LJVForward {
    constructor(w, cfg) {
      this.w = w; this.cfg = cfg;
      const hd = cfg.d_model / cfg.n_head;
      // RoPE 表：cos/sin[pos][hd/2]
      this.cosT = new Float32Array(cfg.max_seq * (hd >> 1));
      this.sinT = new Float32Array(cfg.max_seq * (hd >> 1));
      for (let t = 0; t < cfg.max_seq; t++) {
        for (let j = 0; j < hd >> 1; j++) {
          const inv = Math.pow(10000, (-2 * j) / hd);
          const a = t * inv;
          this.cosT[t * (hd >> 1) + j] = Math.cos(a);
          this.sinT[t * (hd >> 1) + j] = Math.sin(a);
        }
      }
    }

    rmsnorm(x, w, out, o0) {
      let v = 0;
      for (let i = 0; i < x.length; i++) v += x[i] * x[i];
      const r = 1 / Math.sqrt(v / x.length + this.cfg.eps);
      for (let i = 0; i < x.length; i++) out[o0 + i] = x[i] * r * w[i];
    }

    gemvNT(W, Woff, N, K, x, out, o0) {
      /* out[n] = sum_k x[k]*W[n*K+k]，W 按 [N,K] 行主序 */
      for (let n = 0; n < N; n++) {
        let s = 0;
        const row = Woff + n * K;
        for (let k = 0; k < K; k++) s += x[k] * W[row + k];
        out[o0 + n] = s;
      }
    }

    gemmNN(A, Aoff, M, K, B, out, o0, N) {
      /* out[m,n] = sum_k A[m,k]*B[k,n]，B 按 [K,N] 行主序 */
      for (let m = 0; m < M; m++)
        for (let n = 0; n < N; n++) {
          let s = 0;
          for (let k = 0; k < K; k++) s += A[Aoff + m * K + k] * B[k * N + n];
          out[o0 + m * N + n] = s;
        }
    }

    ropeApply(x, xoff, pos, out, ooff) {
      /* 交错对：维度 2j/2j+1 */
      const hd = this.cfg.d_model / this.cfg.n_head;
      const H = this.cfg.n_head;
      const half = hd >> 1;
      for (let h = 0; h < H; h++) {
        const base = xoff + h * hd;
        for (let j = 0; j < half; j++) {
          const a = this.cosT[pos * half + j], b = this.sinT[pos * half + j];
          const x1 = x[base + 2 * j], x2 = x[base + 2 * j + 1];
          out[ooff + h * hd + 2 * j] = x1 * a - x2 * b;
          out[ooff + h * hd + 2 * j + 1] = x1 * b + x2 * a;
        }
      }
    }

    /* 单步（decode 或 prefill 末位）：输入 x[384]，位置 pos，返回 logits */
    step(x, st, logitsOut) {
      const cfg = this.cfg, w = this.w;
      const D = cfg.d_model, H = cfg.n_head, hd = D / H;
      let resid = Float32Array.from(x);
      const pos = st.pos;
      for (let l = 0; l < cfg.n_layer; l++) {
        // norm1
        const n1 = new Float32Array(D);
        this.rmsnorm(resid, w.get(`blocks.${l}.norm1.weight`), n1, 0);
        // qkv
        const qkv = new Float32Array(3 * D);
        this.gemvNT(w.get(`blocks.${l}.attn.qkv.weight`), 0, 3 * D, D, n1, qkv, 0);
        // rope q + 写 k/v cache
        const q = new Float32Array(D);
        const kc = st.cache[l].k, vc = st.cache[l].v;
        const qk = new Float32Array(D);
        this.ropeApply(qkv, 0, pos, qk, 0);
        q.set(qk);
        this.ropeApply(qkv, D, pos, qk, 0);
        kc.set(qk, pos * D);
        vc.set(qkv.subarray(2 * D, 3 * D), pos * D);
        // attention：每头对 0..pos
        const att = new Float32Array(D);
        const scale = 1 / Math.sqrt(hd);
        for (let h = 0; h < H; h++) {
          const scores = new Float32Array(pos + 1);
          let max = -Infinity;
          for (let t = 0; t <= pos; t++) {
            let s = 0;
            const base = t * D + h * hd;
            for (let d = 0; d < hd; d++) s += q[h * hd + d] * kc[base + d];
            s *= scale;
            scores[t] = s;
            if (s > max) max = s;
          }
          let sum = 0;
          for (let t = 0; t <= pos; t++) { scores[t] = Math.exp(scores[t] - max); sum += scores[t]; }
          for (let d = 0; d < hd; d++) {
            let s = 0;
            for (let t = 0; t <= pos; t++) s += scores[t] * vc[t * D + h * hd + d];
            att[h * hd + d] = s / sum;
          }
        }
        const o = new Float32Array(D);
        this.gemvNT(w.get(`blocks.${l}.attn.o.weight`), 0, D, D, att, o, 0);
        for (let i = 0; i < D; i++) resid[i] += o[i];
        // norm2 + MoE top-1
        const n2 = new Float32Array(D);
        this.rmsnorm(resid, w.get(`blocks.${l}.norm2.weight`), n2, 0);
        const router = w.get(`blocks.${l}.moe.router.weight`);
        let best = 0, bp = -Infinity;
        for (let e = 0; e < cfg.n_expert; e++) {
          let s = 0;
          for (let i = 0; i < D; i++) s += n2[i] * router[e * D + i];
          if (s > bp) { bp = s; best = e; }
        }
        const h1 = new Float32Array(cfg.inter);
        this.gemvNT(w.get(`blocks.${l}.moe.w1_t`), best * D * cfg.inter,
          cfg.inter, D, n2, h1, 0);
        for (let i = 0; i < cfg.inter; i++) h1[i] = gelu(h1[i]);
        const h2 = new Float32Array(D);
        this.gemvNT(w.get(`blocks.${l}.moe.w2_t`), best * cfg.inter * D,
          D, cfg.inter, h1, h2, 0);
        // 乘 top-1 概率（softmax 分母含 4 个）
        let exsum = 0;
        for (let e = 0; e < cfg.n_expert; e++) {
          let s = 0;
          for (let i = 0; i < D; i++) s += n2[i] * router[e * D + i];
          exsum += Math.exp(s - bp);
        }
        const pw = 1 / exsum;
        for (let i = 0; i < D; i++) resid[i] += h2[i] * pw;
      }
      st.pos++;
      const nf = new Float32Array(D);
      this.rmsnorm(resid, w.get("norm_f"), nf, 0);
      // logits = nf · tok_emb^T
      const E = w.get("tok_emb"), V = cfg.vocab_size;
      for (let v = 0; v < V; v++) {
        let s = 0;
        const row = v * D;
        for (let i = 0; i < D; i++) s += nf[i] * E[row + i];
        logitsOut[v] = s;
      }
    }


  /* 视觉塔（CLIP ViT，quick_gelu）+ projector，纯 CPU。
   * pixelData: f32[224*224*3]，CLIP 归一化后 → f32[49*384] */
  visionEncode(pixelData) {
    const w = this.w, cfg = this.cfg;
    const VH = cfg.v_hidden, P = cfg.n_img_token;
    const g = w.get;
    const layernorm = (x, wn, bn) => {
      const out = new Float32Array(x.length);
      const rows = x.length / VH;
      for (let r = 0; r < rows; r++) {
        let m = 0;
        for (let j = 0; j < VH; j++) m += x[r * VH + j];
        m /= VH;
        let v = 0;
        for (let j = 0; j < VH; j++) { const t = x[r * VH + j] - m; v += t * t; }
        const sc = 1 / Math.sqrt(v / VH + 1e-5);
        for (let j = 0; j < VH; j++) out[r * VH + j] = (x[r * VH + j] - m) * sc * wn[j] + bn[j];
      }
      return out;
    };
    const attnFull = (x, Wq, bq, Wk, bk, Wv, bv, Wo, bo, heads) => {
      const T = x.length / VH, hd = VH / heads;
      const q = new Float32Array(T * VH), k = new Float32Array(T * VH), vv = new Float32Array(T * VH);
      this.gemvNT(Wq, 0, VH, VH, x, q, 0);
      for (let i = 0; i < VH; i++) q[i] += bq[i];
      this.gemvNT(Wk, 0, VH, VH, x, k, 0);
      for (let i = 0; i < VH; i++) k[i] += bk[i];
      this.gemvNT(Wv, 0, VH, VH, x, vv, 0);
      for (let i = 0; i < VH; i++) vv[i] += bv[i];
      const o = new Float32Array(T * VH);
      const scale = 1 / Math.sqrt(hd);
      for (let t = 0; t < T; t++) {
        for (let h = 0; h < heads; h++) {
          const sc = new Float32Array(T);
          let mx = -1e30;
          for (let s = 0; s < T; s++) {
            let acc = 0;
            for (let d = 0; d < hd; d++) acc += q[t * VH + h * hd + d] * k[s * VH + h * hd + d];
            acc *= scale; sc[s] = acc; if (acc > mx) mx = acc;
          }
          let sum = 0;
          for (let s = 0; s < T; s++) { sc[s] = Math.exp(sc[s] - mx); sum += sc[s]; }
          for (let d = 0; d < hd; d++) {
            let acc = 0;
            for (let s = 0; s < T; s++) acc += sc[s] * vv[s * VH + h * hd + d];
            o[t * VH + h * hd + d] = acc / sum;
          }
        }
      }
      const out = new Float32Array(T * VH);
      this.gemvNT(Wo, 0, VH, VH, o, out, 0);
      for (let i = 0; i < VH; i++) out[i] += bo[i];
      return out;
    };
    // patch embedding：权重 [512,3,32,32] → [512,3072]，输入展开 [49,3072]
    const pw = g("vis.patch_embedding.weight");
    let cur = new Float32Array(P * VH);
    for (let pi = 0; pi < P; pi++) {
      const py = (pi / 7) | 0, px = pi % 7;
      for (let n = 0; n < VH; n++) {
        let s = 0;
        const woff = n * 3072;
        for (let c = 0; c < 3; c++) {
          for (let yy = 0; yy < 32; yy++) {
            for (let xx = 0; xx < 32; xx++) {
              const src = ((py * 32 + yy) * 224 + px * 32 + xx) * 3 + c;
              s += pixelData[src] * pw[woff + c * 1024 + yy * 32 + xx];
            }
          }
        }
        cur[pi * VH + n] = s;
      }
    }
    // cls + pos
    const cls = g("vis.class_embedding"), pos = g("vis.position_embedding");
    const seq = new Float32Array(50 * VH);
    for (let j = 0; j < VH; j++) seq[j] = cls[j] + pos[j];
    for (let i = 1; i <= P; i++)
      for (let j = 0; j < VH; j++) seq[i * VH + j] = cur[(i - 1) * VH + j] + pos[i * VH + j];
    cur = layernorm(seq, g("vis.pre_layrnorm.weight"), g("vis.pre_layrnorm.bias"));
    for (let l = 0; l < cfg.v_layers; l++) {
      const p = `vis.layers.${l}.`;
      const ln1 = layernorm(cur, g(p + "layer_norm1.weight"), g(p + "layer_norm1.bias"));
      const ao = attnFull(ln1, g(p + "self_attn.q_proj.weight"), g(p + "self_attn.q_proj.bias"),
        g(p + "self_attn.k_proj.weight"), g(p + "self_attn.k_proj.bias"),
        g(p + "self_attn.v_proj.weight"), g(p + "self_attn.v_proj.bias"),
        g(p + "self_attn.out_proj.weight"), g(p + "self_attn.out_proj.bias"), cfg.v_heads);
      for (let i = 0; i < ao.length; i++) cur[i] += ao[i];
      const ln2 = layernorm(cur, g(p + "layer_norm2.weight"), g(p + "layer_norm2.bias"));
      const f1 = new Float32Array(cfg.v_inter * 50);
      this.gemvNT2(ln2, g(p + "mlp.fc1.weight"), f1, cfg.v_inter, VH, 50);
      for (let i = 0; i < f1.length; i++) f1[i] += g(p + "mlp.fc1.bias")[i % cfg.v_inter];
      for (let i = 0; i < f1.length; i++) f1[i] = quickGelu(f1[i]);
      const f2 = new Float32Array(VH * 50);
      this.gemmNN2(f1, g(p + "mlp.fc2.weight"), f2, VH, cfg.v_inter, 50);
      for (let i = 0; i < f2.length; i++) f2[i] += g(p + "mlp.fc2.bias")[i % VH];
      for (let i = 0; i < f2.length; i++) cur[i] += f2[i];
    }
    cur = layernorm(cur, g("vis.post_layernorm.weight"), g("vis.post_layernorm.bias"));
    // projector：取 49 个 patch
    const pr = cur.subarray(VH, 50 * VH);
    const pc1 = new Float32Array(P * 768);
    this.gemvNT2(pr, g("proj.fc1.weight"), pc1, 768, VH, P);
    for (let i = 0; i < pc1.length; i++) pc1[i] += g("proj.fc1.bias")[i % 768];
    for (let i = 0; i < pc1.length; i++) pc1[i] = gelu(pc1[i]);
    const pc2 = new Float32Array(P * cfg.d_model);
    this.gemmNN2(pc1, g("proj.fc2.weight"), pc2, cfg.d_model, 768, P);
    for (let i = 0; i < pc2.length; i++) pc2[i] += g("proj.fc2.bias")[i % cfg.d_model];
    return pc2;
  }

  /* 批量 x[M,K] @ W[out,in]^T → out[M,out]（vision 用，行偏置后加） */
  gemvNT2(x, W, out, OUTN, K, M) {
    for (let m = 0; m < M; m++)
      for (let n = 0; n < OUTN; n++) {
        let s = 0;
        const row = n * K;
        for (let k = 0; k < K; k++) s += x[m * K + k] * W[row + k];
        out[m * OUTN + n] = s;
      }
  }

  /* 批量 x[M,K] @ B[K,N]（B 行主序） */
  gemmNN2(x, B, out, N, K, M) {
    for (let m = 0; m < M; m++)
      for (let n = 0; n < N; n++) {
        let s = 0;
        for (let k = 0; k < K; k++) s += x[m * K + k] * B[k * N + n];
        out[m * N + n] = s;
      }
  }

    /* 整段 prefill：逐 token 走 step（CPU 下等价且简单） */
    embedTokens(ids) {
      const E = this.w.get("tok_emb"), D = this.cfg.d_model;
      const out = [];
      for (const id of ids) out.push(E.subarray(id * D, id * D + D));
      return out;
    }
  }

  root.LJVWeights = LJVWeights;
  root.LJVState = LJVState;
  root.LJVForward = LJVForward;
  root.geluRef = gelu;
  root.quickGeluRef = quickGelu;
})(typeof window !== "undefined" ? window : globalThis);
