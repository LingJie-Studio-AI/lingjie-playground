/* Playground 应用逻辑（干净版）：
 * - 模型选择（Code v1 / Chat v1 转换中禁用；LV-Chat-v2-Flash 可用，眼睛 + 视觉气泡）
 * - 权重分块下载（进度条）→ GPU 自检（CPU 参考前向对比）→ 自检失败自动回退 CPU
 * - CPU 与 GPU 都维护持久 KV 状态；图像只在上传的当轮注入
 * - 采样与 serve_local.py 一致：重复惩罚 1.15/最近 24、温度、top-p 0.9、特殊符停
 */
(function () {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const MODELS = [
    { id: "code-v1", name: "LingJie-Code-v1-Pro", ready: false,
      desc: "202M JAM 架构 · 窄域代码生成", status: "WebGPU 版转换中" },
    { id: "chat-v1", name: "LingJie-Chat-v1-Flash", ready: false,
      desc: "传统 MoE 超小模型", status: "WebGPU 版转换中" },
    { id: "lv-v2", name: "LingJieLV-Chat-v2-Flash", ready: true, vision: true,
      desc: "36M MoE · 中文对话 + 轻量图文问答", status: "可用" },
  ];
  const SPECIALS = { pad: 0, end: 1, user: 2, asst: 3, img: 4 };
  const MEAN = [0.48145466, 0.4578275, 0.40821073];
  const STD = [0.26862954, 0.26130258, 0.27577711];

  const $id = (s) => document.getElementById(s);
  const el = {
    menu: $id("modelMenu"), pill: $id("modelPill"), modelName: $id("modelName"),
    modelBtn: $id("modelBtn"), modelVision: $id("modelVision"), uploadWrap: $id("uploadWrap"),
    uploadBtn: $id("uploadBtn"), fileInput: $id("fileInput"),
    chat: $id("chat"), greet: $id("greet"), status: $id("status"),
    txt: $id("txt"), send: $id("send"), preview: $id("preview"),
    overlay: $id("overlay"), bar: $id("bar"), dlmsg: $id("dlmsg"),
    dlsub: $id("dlsub"), gpuToggle: $id("gpuToggle"), backendHint: $id("backendHint"),
    miParams: $id("miParams"), miCtx: $id("miCtx"), miBackend: $id("miBackend"),
    miSpeed: $id("miSpeed"), greetSub: $id("greetSub"), ctxHint: $id("ctxHint"),
    newChat: $id("newChat"),
  };

  let tok = null, manifest = null, weights = null;
  let gpu = null;
  let selected = null;
  let loadedModel = "";      // 已加载的模型 id
  let state = null;          // CPU 推理状态（含 KV cache）
  let useGPU = true, gpuOk = false;
  let history = [];          // [{role:'user'|'assist', text}]
  let pendingImg = null;     // {pixels: f32[150528], url}
  let generating = false;

  /* ============ 模型菜单 ============ */
  function buildMenu() {
    el.menu.innerHTML = "";
    for (const m of MODELS) {
      const b = document.createElement("button");
      b.className = "mi" + (m.ready ? "" : " disabled") +
        (selected && selected.id === m.id ? " sel" : "");
      const left = document.createElement("div");
      const nm = document.createElement("div");
      nm.className = "nm"; nm.textContent = m.name;
      left.appendChild(nm);
      const sub = document.createElement("div");
      sub.className = "sub"; sub.textContent = m.desc +
        (m.ready ? "" : " · " + m.status);
      left.appendChild(sub);
      b.appendChild(left);
      if (m.vision) {
        const v = document.createElement("span");
        v.className = "mc-vision";
        v.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12s3.6-6.5 10-6.5S22 12 22 12s-3.6 6.5-10 6.5S2 12 2 12z"/><circle cx="12" cy="12" r="2.6"/></svg>';
        const t = document.createElement("span");
        t.className = "tag-soft"; t.textContent = "视觉";
        v.appendChild(t);
        b.appendChild(v);
      }
      const right = document.createElement("span");
      right.className = "right";
      right.style.cssText = "font-size:.68rem;font-family:var(--mono);color:" +
        (m.ready ? "var(--accent)" : "var(--ink3)");
      right.textContent = m.ready ? "✓ 可用" : "✗ 转换中";
      b.appendChild(right);
      if (!m.ready) b.disabled = true;
      else b.addEventListener("click", () => { el.pill.classList.remove("open"); selectModel(m); });
      el.menu.appendChild(b);
    }
  }

  function selectModel(m) {
    selected = m;
    el.modelName.textContent = m.name;
    el.modelVision.style.display = m.vision ? "inline-flex" : "none";
    el.uploadWrap.classList.toggle("show", !!m.vision);
    buildMenu();
    loadModel(m);
  }

  el.modelBtn.addEventListener("click", () => el.pill.classList.toggle("open"));
  document.addEventListener("click", (e) => {
    if (!e.target.closest("#modelPill")) el.pill.classList.remove("open");
  });

  /* ============ 权重加载 ============ */
  async function loadModel(m) {
    if (loadedModel === m.id) return;
    loadedModel = "";
    el.overlay.classList.add("show");
    el.bar.style.width = "0%";
    el.dlsub.textContent = m.name + " · 153 MB（fp16，分 2 块拉取）";
    setStatus("正在加载 " + m.name + " …");
    try {
      const [man, tokJson] = await Promise.all([
        fetch("weights/manifest.json").then(r => {
          if (!r.ok) throw new Error("manifest 404"); return r.json();
        }),
        fetch("tokenizer.json").then(r => {
          if (!r.ok) throw new Error("tokenizer 404"); return r.text();
        }),
      ]);
      manifest = man;
      tok = new LJVTokenizer(tokJson);
      const buffers = [];
      let doneBytes = 0;
      for (let i = 0; i < manifest.parts.length; i++) {
        const part = manifest.parts[i];
        const resp = await fetch(part.file);
        if (!resp.ok) throw new Error("分块下载失败：" + part.file);
        const reader = resp.body.getReader();
        const chunks = [];
        let got = 0;
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          chunks.push(value); got += value.length;
          const frac = (doneBytes + got) / manifest.total_bytes;
          el.bar.style.width = (frac * 100).toFixed(1) + "%";
          el.dlmsg.textContent = `分块 ${i + 1}/${manifest.parts.length} · ` +
            (got / 1e6).toFixed(1) + " / " + (part.bytes / 1e6).toFixed(1) + " MB";
        }
        const all = new Uint8Array(got);
        let o = 0;
        for (const c of chunks) { all.set(c, o); o += c.length; }
        buffers.push(all.buffer);
        doneBytes += part.bytes;
      }
      weights = new LJVWeights(manifest, buffers);
      await initBackend(buffers);
      el.overlay.classList.remove("show");
      loadedModel = m.id;
      el.send.disabled = false;
      el.greetSub.textContent = m.name + " 已就绪 · " +
        (useGPU ? "WebGPU 加速" : "CPU 模式") +
        (selected.vision ? " · 支持图片输入" : "");
      el.miParams.textContent = "36.20M / 激活 16.74M";
      el.miCtx.textContent = manifest.cfg.max_seq;
      setStatus("");
      addMsg("assist", "模型加载完成（后端：" + (useGPU ? "WebGPU" : "CPU") +
        "）。" + (selected.vision ? "点输入框旁的回形针上传图片，我可以看图回答。" : ""));
    } catch (e) {
      el.overlay.classList.remove("show");
      setStatus("加载失败：" + e.message, true);
    }
  }

  /* ============ 后端初始化 + GPU 自检 ============ */
  async function initBackend(buffers) {
    useGPU = el.gpuToggle.checked && !!navigator.gpu;
    gpuOk = false;
    gpu = null;
    if (el.gpuToggle.checked && !navigator.gpu) {
      setHint("此浏览器不支持 WebGPU，已回退 CPU 模式");
    }
    if (useGPU) {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) { useGPU = false; throw new Error("无可用 GPU 适配器"); }
        const device = await LJVGPU.requestDevice(adapter);
        gpu = new LJVGPU(manifest, buffers, device);
        window.__ljdbg = { gpu: gpu, weights: weights, cfg: manifest.cfg };
        await gpu.upload((f) => {
          el.bar.style.width = (100 * (0.98 + f * 0.02)).toFixed(1) + "%";
          el.dlmsg.textContent = "上传权重到 GPU … " + (f * 100).toFixed(0) + "%";
        });
        // 自检：GPU prefill vs CPU 前向（CPU 数学已用 python 参考逐层验证）
        const refIds = [2, 3214, 266, 4459, 4197, 1368, 358, 738, 1, 3];
        const gpuLg = await gpu.prefill(refIds, null);
        const cfg = manifest.cfg;
        const fwd = new LJVForward(weights, cfg);
        const st = new LJVState(cfg);
        const logits = new Float32Array(cfg.vocab_size);
        const row = new Float32Array(cfg.d_model);
        const E = weights.get("tok_emb");
        for (const id of refIds) {
          row.set(E.subarray(id * cfg.d_model, (id + 1) * cfg.d_model));
          fwd.step(row, st, logits);
        }
        let gi = 0, gv = -Infinity;
        for (let i = 0; i < cfg.vocab_size; i++) if (gpuLg[i] > gv) { gv = gpuLg[i]; gi = i; }
        let ci = 0, cv = -Infinity;
        for (let i = 0; i < cfg.vocab_size; i++) if (logits[i] > cv) { cv = logits[i]; ci = i; }
        let maxDiff = 0;
        for (let i = 0; i < 64; i++) maxDiff = Math.max(maxDiff, Math.abs(gpuLg[i] - logits[i]));
        if (gi !== ci) {
          setHint("GPU 自检未通过（top-1 " + gi + " ≠ " + ci + "），已自动回退 CPU");
        } else if (maxDiff > 0.5) {
          setHint("GPU 自检偏差过大（" + maxDiff.toFixed(3) + "），已自动回退 CPU");
        } else {
          gpuOk = true;
          setHint("WebGPU 自检通过（logit 差 " + maxDiff.toFixed(4) + "），加速已启用");
        }
        state = null;
      } catch (e) {
        gpuOk = false;
        setHint("WebGPU 初始化失败（" + e.message + "），已回退 CPU");
      }
    } else {
      setHint("WebGPU 加速已关闭，使用 CPU 模式（约 2~4 token/秒）");
    }
    el.miBackend.textContent = useGPU && gpuOk ? "WebGPU" : "CPU";
  }

  el.gpuToggle.addEventListener("change", () => {
    if (loadedModel) {
      initBackend().then(() => {
        setStatus("后端：" + (useGPU && gpuOk ? "WebGPU（自检通过）" : "CPU"));
      });
    } else setHint("选择模型后生效");
  });

  function setHint(t) { el.backendHint.textContent = t; }
  function setStatus(t, isErr) {
    el.status.textContent = t;
    el.status.classList.toggle("err", !!isErr);
  }

  /* ============ 图片 ============ */
  el.uploadBtn.addEventListener("click", () => el.fileInput.click());
  el.fileInput.addEventListener("change", () => {
    const f = el.fileInput.files[0];
    if (!f) return;
    if (!f.type.startsWith("image/")) { setStatus("只能上传图片", true); return; }
    const img = new Image();
    img.onload = () => {
      const c = document.createElement("canvas");
      c.width = 224; c.height = 224;
      const ctx = c.getContext("2d");
      ctx.drawImage(img, 0, 0, 224, 224);
      const d = ctx.getImageData(0, 0, 224, 224).data;
      const px = new Float32Array(224 * 224 * 3);
      for (let i = 0, j = 0; i < d.length; i += 4, j += 3) {
        px[j] = (d[i] / 255 - MEAN[0]) / STD[0];
        px[j + 1] = (d[i + 1] / 255 - MEAN[1]) / STD[1];
        px[j + 2] = (d[i + 2] / 255 - MEAN[2]) / STD[2];
      }
      const url = c.toDataURL("image/jpeg", 0.85);
      pendingImg = { pixels: px, url };
      el.preview.innerHTML = "";
      const chip = document.createElement("div");
      chip.className = "chip";
      const im = document.createElement("img");
      im.src = url;
      chip.appendChild(im);
      const rm = document.createElement("button");
      rm.className = "rm"; rm.textContent = "×";
      rm.addEventListener("click", () => { pendingImg = null; el.preview.innerHTML = ""; });
      chip.appendChild(rm);
      el.preview.appendChild(chip);
      URL.revokeObjectURL(img.src);
    };
    img.src = URL.createObjectURL(f);
    el.fileInput.value = "";
  });

  /* ============ 采样（与 serve_local.py 一致） ============ */
  function sample(logits, genIds, temp, topp, rep) {
    const lg = Float32Array.from(logits);
    if (genIds.length) {
      const recent = new Set(genIds.slice(-24));
      for (const t of recent) lg[t] = lg[t] > 0 ? lg[t] / rep : lg[t] * rep;
    }
    if (temp <= 0) {
      let bi = 0, bv = -Infinity;
      for (let i = 0; i < lg.length; i++) if (lg[i] > bv) { bv = lg[i]; bi = i; }
      return bi;
    }
    let mx = -Infinity;
    for (const v of lg) if (v > mx) mx = v;
    const ex = new Float32Array(lg.length);
    let sum = 0;
    for (let i = 0; i < lg.length; i++) { ex[i] = Math.exp((lg[i] - mx) / temp); sum += ex[i]; }
    const order = Array.from({ length: lg.length }, (_, i) => i)
      .sort((a, b) => ex[b] - ex[a]);
    let cum = 0;
    const kept = [];
    for (const i of order) {
      kept.push(i);
      cum += ex[i] / sum;
      if (cum >= topp) break;
    }
    let ksum = 0;
    for (const i of kept) ksum += ex[i];
    let r = Math.random() * ksum;
    for (const i of kept) { r -= ex[i]; if (r <= 0) return i; }
    return kept[kept.length - 1];
  }

  /* ============ 消息渲染 ============ */
  function addMsg(role, text, imgUrl) {
    el.greet.style.display = "none";
    const d = document.createElement("div");
    d.className = "msg " + role;
    const who = document.createElement("div");
    who.className = "who";
    who.textContent = role === "user" ? "你" : "LingJie";
    d.appendChild(who);
    const body = document.createElement("div");
    body.className = "body";
    if (imgUrl) {
      const row = document.createElement("div");
      row.className = "imgrow";
      const im = document.createElement("img");
      im.src = imgUrl; row.appendChild(im);
      body.appendChild(row);
    }
    const span = document.createElement("span");
    span.textContent = text;
    body.appendChild(span);
    d.appendChild(body);
    el.chat.appendChild(d);
    el.chat.scrollTop = el.chat.scrollHeight;
    return body;
  }

  function renderStream(span, full) {
    const m = full.match(/^\[思考\]([\s\S]*?)\[\/思考\]([\s\S]*)$/);
    if (m) {
      span.innerHTML = "";
      const th = document.createElement("div");
      th.className = "think";
      th.textContent = m[1].trim();
      span.appendChild(th);
      const rest = document.createElement("span");
      rest.textContent = m[2];
      span.appendChild(rest);
    } else {
      span.textContent = full;
    }
  }

  /* ============ 生成 ============ */
  async function send() {
    if (generating || loadedModel !== "lv-v2") return;
    const text = el.txt.value.trim();
    if (!text && !pendingImg) return;
    if (pendingImg && (!selected || !selected.vision)) {
      setStatus("当前模型不支持图像", true); return;
    }
    generating = true;
    el.send.disabled = true; el.txt.disabled = true;
    const imgUrl = pendingImg ? pendingImg.url : null;
    const imgPixels = pendingImg ? pendingImg.pixels : null;
    addMsg("user", text || "（图片）", imgUrl);
    el.txt.value = ""; pendingImg = null; el.preview.innerHTML = "";
    const body = addMsg("assist", "");
    const cur = document.createElement("span");
    const cursor = document.createElement("i");
    cursor.className = "cursor";
    body.appendChild(cur); body.appendChild(cursor);
    setStatus("生成中…");
    try {
      const cfg = manifest.cfg;
      const temp = +$id("temp").value, topp = +$id("topp").value;
      const rep = +$id("rep").value, maxNew = +$id("maxn").value;
      // 模板：多轮历史 + 本轮（图片占位只在最新 user 轮）
      let template = "";
      for (const h of history) {
        template += "<|user|>" + h.text + "<|end|><|assistant|>" + h.reply + "<|end|>";
      }
      template += "<|user|>" +
        (imgUrl ? "<|image|>".repeat(cfg.n_img_token) : "") + text +
        "<|end|><|assistant|>";
      const ids = tok.encode(template, false);
      if (ids.length > cfg.max_seq - maxNew) {
        throw new Error("上下文超限（" + ids.length + "/" + cfg.max_seq +
          "），请点击「开启新对话」");
      }
      el.ctxHint.textContent = "上下文 " + ids.length + "/" + cfg.max_seq;
      const t0 = performance.now();
      let logits;
      const useG = useGPU && gpuOk;
      if (useG) {
        logits = await gpu.prefill(ids, imgPixels);
      } else {
        const fwd = new LJVForward(weights, cfg);
        state = { fwd, st: new LJVState(cfg) };
        let emb = null;
        if (imgPixels) {
          setStatus("视觉编码中（CPU 较慢，数秒）…");
          emb = state.fwd.visionEncode(imgPixels);
        }
        const row = new Float32Array(cfg.d_model);
        const E = weights.get("tok_emb");
        let im = 0;
        for (const id of ids) {
          if (id === SPECIALS.img && emb) {
            row.set(emb.subarray(im * cfg.d_model, (im + 1) * cfg.d_model)); im++;
          } else {
            row.set(E.subarray(id * cfg.d_model, (id + 1) * cfg.d_model));
          }
          state.fwd.step(row, state.st, logits);
        }
      }
      const genIds = [];
      let full = "";
      for (let i = 0; i < maxNew; i++) {
        const nid = sample(logits, genIds, temp, topp, rep);
        if (nid === SPECIALS.pad || nid === SPECIALS.end ||
            nid === SPECIALS.user || nid === SPECIALS.asst) break;
        genIds.push(nid);
        full += tok.decode([nid]);
        renderStream(cur, full);
        el.chat.scrollTop = el.chat.scrollHeight;
        el.miSpeed.textContent = (genIds.length / ((performance.now() - t0) / 1000))
          .toFixed(1) + " tok/s";
        if (useG) {
          logits = await gpu.decode(nid);
        } else {
          const row = new Float32Array(cfg.d_model);
          row.set(weights.get("tok_emb").subarray(nid * cfg.d_model, (nid + 1) * cfg.d_model));
          state.fwd.step(row, state.st, logits);
        }
      }
      cursor.remove();
      const reply = full.replace(/<[^>]*>/g, "").trim();
      cur.textContent = reply || "（空回复）";
      history.push({ role: "user", text, reply: reply });
      setStatus("");
    } catch (e) {
      setStatus("生成失败：" + e.message, true);
      cur.textContent = "生成失败：" + e.message;
    }
    generating = false;
    el.send.disabled = false; el.txt.disabled = false;
    el.txt.focus();
  }

  el.send.addEventListener("click", send);
  el.txt.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); el.send.click(); }
  });
  el.newChat.addEventListener("click", () => {
    history = [];
    if (gpu) { /* GPU KV 在下次 prefill 时清零 */ }
    state = null;
    el.chat.innerHTML = "";
    el.chat.appendChild(el.greet);
    el.greet.style.display = "flex";
    el.ctxHint.textContent = "";
    setStatus("新对话已开启");
  });

  /* ============ 启动 ============ */
  buildMenu();
  if (navigator.gpu) setHint("WebGPU 可用 · 选择模型后自动加载并自检");
  else setHint("此浏览器不支持 WebGPU，将使用 CPU 模式");
  fetch("weights/manifest.json").then(r => r.json()).then(m => {
    el.dlsub.textContent = "LingJieLV-Chat-v2-Flash · " +
      (m.total_bytes / 1e6).toFixed(0) + " MB（fp16，分 " + m.parts.length + " 块）";
  }).catch(() => {});
})();
