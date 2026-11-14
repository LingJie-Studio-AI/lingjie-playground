/* LingJieLV-Chat-v2-Flash WebGPU 引擎（最终版）。
 *
 * 数学与 lj-cpu.js 完全一致（CPU 前向已用 python 参考逐层校验）。
 * GPU 权重 fp16 直存（显存减半，bias 拼接在权重尾部）；激活 f32。
 * 残差加法使用乒乓缓冲（WebGPU 禁止同一缓冲在同一 dispatch 内同时只读+可写）。
 * 加载后执行 GPU↔CPU 自检（top-1 + logit 差），不过自动回退 CPU。
 */
(function (root) {
  const S = {};

  S.gemm = `
enable f16;
struct Dims { m:u32, n:u32, k:u32, flags:u32 };
@group(0) @binding(0) var<storage,read> a:array<f32>;
@group(0) @binding(1) var<storage,read> b:array<f16>;
@group(0) @binding(2) var<storage,read_write> c:array<f32>;
@group(0) @binding(3) var<uniform> d:Dims;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let i = gid.x;
  if(i >= d.m*d.n){ return; }
  let m = i / d.n;
  let n = i % d.n;
  var s = 0.0;
  let K = d.k;
  for(var k=0u; k<K; k=k+1u){ s = s + a[m*K+k] * f32(b[n*K+k]); }
  if(d.flags == 1u){ s = s + f32(b[d.n*K + n]); }
  c[i] = s;
}`;

  S.rmsMean = `
@group(0) @binding(0) var<storage,read> x:array<f32>;
@group(0) @binding(1) var<storage,read_write> inv:array<f32>;
@group(0) @binding(2) var<uniform> d:vec2<u32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let r = gid.x;
  if(r >= d.x){ return; }
  var v = 0.0;
  let off = r*d.y;
  for(var i=0u; i<d.y; i=i+1u){ v = v + x[off+i]*x[off+i]; }
  inv[r] = 1.0/sqrt(v/f32(d.y) + 1e-5);
}`;

  S.rmsApply = `
enable f16;
@group(0) @binding(0) var<storage,read> x:array<f32>;
@group(0) @binding(1) var<storage,read> w:array<f16>;
@group(0) @binding(2) var<storage,read> inv:array<f32>;
@group(0) @binding(3) var<storage,read_write> y:array<f32>;
@group(0) @binding(4) var<uniform> d:vec2<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let i = gid.x;
  if(i >= d.x*d.y){ return; }
  y[i] = x[i] * inv[i/d.y] * f32(w[i%d.y]);
}`;

  S.add = `
@group(0) @binding(0) var<storage,read> a:array<f32>;
@group(0) @binding(1) var<storage,read> b:array<f32>;
@group(0) @binding(2) var<storage,read_write> y:array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  if(gid.x >= arrayLength(&a)){ return; }
  y[gid.x] = a[gid.x] + b[gid.x];
}`;

  S.mulRow = `
@group(0) @binding(0) var<storage,read> a:array<f32>;
@group(0) @binding(1) var<storage,read> w:array<f32>;
@group(0) @binding(2) var<storage,read_write> y:array<f32>;
@group(0) @binding(3) var<uniform> d:vec2<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let i = gid.x;
  if(i >= d.x*d.y){ return; }
  y[i] = a[i] * w[i/d.y];
}`;

  S.ropeKV = `
@group(0) @binding(0) var<storage,read> qkv:array<f32>;
@group(0) @binding(1) var<storage,read_write> q:array<f32>;
@group(0) @binding(2) var<storage,read_write> kc:array<f32>;
@group(0) @binding(3) var<storage,read_write> vc:array<f32>;
@group(0) @binding(4) var<uniform> d:vec4<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let i = gid.x;
  if(i >= d.x*1152u){ return; }
  let m = i / 1152u;
  let e = i % 1152u;
  let c = e / 384u;
  let rem = e % 384u;
  let h  = rem / 64u;
  let r  = (rem % 64u) / 2u;
  let odd = rem % 2u;
  let pos = d.w + m;
  let D  = d.y;
  let base = m*1152u + c*384u + h*64u + r*2u;
  let x1 = qkv[base];
  let x2 = qkv[base + 1u];
  let inv = pow(10000.0, -2.0*f32(r)/64.0);
  let ang = f32(pos)*inv;
  let cs = cos(ang); let sn = sin(ang);
  if(odd == 1u){ return; }
  let rowOff = h*64u + r*2u;
  if(c == 2u){
    vc[pos*D + rowOff] = x1;
    vc[pos*D + rowOff + 1u] = x2;
    return;
  }
  let rot0 = x1*cs - x2*sn;
  let rot1 = x1*sn + x2*cs;
  if(c == 0u){
    q[m*D + rowOff] = rot0;
    q[m*D + rowOff + 1u] = rot1;
  } else {
    kc[pos*D + rowOff] = rot0;
    kc[pos*D + rowOff + 1u] = rot1;
  }
}`;

  S.attnCausal = `
@group(0) @binding(0) var<storage,read> q:array<f32>;
@group(0) @binding(1) var<storage,read> kc:array<f32>;
@group(0) @binding(2) var<storage,read> vc:array<f32>;
@group(0) @binding(3) var<storage,read_write> y:array<f32>;
@group(0) @binding(4) var<uniform> d:vec4<u32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let mi = gid.x;
  let h  = gid.y;
  if(mi >= d.x || h >= d.z){ return; }
  let D = d.y;
  let hd = D / d.z;
  let qpos = d.w + mi;
  let qoff = mi*D + h*hd;
  var scores:array<f32,768>;
  var mx = -3e38;
  for(var t=0u; t<=qpos; t=t+1u){
    var s = 0.0;
    let koff = t*D + h*hd;
    for(var j=0u; j<hd; j=j+1u){ s = s + q[qoff+j]*kc[koff+j]; }
    s = s / sqrt(f32(hd));
    scores[t] = s;
    if(s > mx){ mx = s; }
  }
  var sum = 0.0;
  for(var t=0u; t<=qpos; t=t+1u){ scores[t] = exp(scores[t]-mx); sum = sum + scores[t]; }
  let yoff = mi*D + h*hd;
  for(var j=0u; j<hd; j=j+1u){
    var s = 0.0;
    for(var t=0u; t<=qpos; t=t+1u){ s = s + scores[t]*vc[t*D + h*hd + j]; }
    y[yoff+j] = s / sum;
  }
}`;

  S.attnFull = `
@group(0) @binding(0) var<storage,read> q:array<f32>;
@group(0) @binding(1) var<storage,read> k:array<f32>;
@group(0) @binding(2) var<storage,read> v:array<f32>;
@group(0) @binding(3) var<storage,read_write> y:array<f32>;
@group(0) @binding(4) var<uniform> d:vec4<u32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let mi = gid.x;
  let h  = gid.y;
  if(mi >= d.x || h >= d.z){ return; }
  let H = d.z;
  let hd = d.y / H;
  let T = d.w;
  let qoff = mi*d.y + h*hd;
  var scores:array<f32,64>;
  var mx = -3e38;
  for(var t=0u; t<T; t=t+1u){
    var s = 0.0;
    let koff = t*d.y + h*hd;
    for(var j=0u; j<hd; j=j+1u){ s = s + q[qoff+j]*k[koff+j]; }
    s = s / sqrt(f32(hd));
    scores[t] = s;
    if(s > mx){ mx = s; }
  }
  var sum = 0.0;
  for(var t=0u; t<T; t=t+1u){ scores[t] = exp(scores[t]-mx); sum = sum + scores[t]; }
  let yoff = mi*d.y + h*hd;
  for(var j=0u; j<hd; j=j+1u){
    var s = 0.0;
    for(var t=0u; t<T; t=t+1u){ s = s + scores[t]*v[t*d.y + h*hd + j]; }
    y[yoff+j] = s / sum;
  }
}`;

  S.gelu = `
@group(0) @binding(0) var<storage,read_write> x:array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  if(gid.x >= arrayLength(&x)){ return; }
  let v = x[gid.x];
  let sgn = select(-1.0, 1.0, v >= 0.0);
  let ax = abs(v);
  let t = 1.0/(1.0 + 0.3275911*ax);
  let y = 1.0 - (((((1.061405429*t - 1.453152027)*t) + 1.421413741)*t - 0.284496736)*t + 0.254829592)*t*exp(-ax*ax);
  x[gid.x] = 0.5*v*(1.0 + sgn*y);
}`;

  S.quickGelu = `
@group(0) @binding(0) var<storage,read_write> x:array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  if(gid.x >= arrayLength(&x)){ return; }
  x[gid.x] = x[gid.x] / (1.0 + exp(-1.702*x[gid.x]));
}`;

  S.layernorm = `
enable f16;
@group(0) @binding(0) var<storage,read> x:array<f32>;
@group(0) @binding(1) var<storage,read> w:array<f16>;
@group(0) @binding(2) var<storage,read> b:array<f16>;
@group(0) @binding(3) var<storage,read_write> y:array<f32>;
@group(0) @binding(4) var<uniform> d:vec2<u32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let r = gid.x;
  if(r >= d.x){ return; }
  let off = r*d.y;
  var m = 0.0;
  for(var i=0u; i<d.y; i=i+1u){ m = m + x[off+i]; }
  m = m / f32(d.y);
  var v = 0.0;
  for(var i=0u; i<d.y; i=i+1u){ var t=x[off+i]-m; v = v + t*t; }
  let s = 1.0/sqrt(v/f32(d.y) + 1e-5);
  for(var i=0u; i<d.y; i=i+1u){ y[off+i] = (x[off+i]-m)*s*f32(w[i]) + f32(b[i]); }
}`;

  S.gather = `
@group(0) @binding(0) var<storage,read> x:array<f32>;
@group(0) @binding(1) var<storage,read> idx:array<u32>;
@group(0) @binding(2) var<storage,read_write> y:array<f32>;
@group(0) @binding(3) var<uniform> d:vec2<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let i = gid.x;
  if(i >= d.x*d.y){ return; }
  y[i] = x[idx[i/d.y]*d.y + (i%d.y)];
}`;

  S.scatter = `
@group(0) @binding(0) var<storage,read> y:array<f32>;
@group(0) @binding(1) var<storage,read> idx:array<u32>;
@group(0) @binding(2) var<storage,read_write> x:array<f32>;
@group(0) @binding(3) var<uniform> d:vec2<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let i = gid.x;
  if(i >= d.x*d.y){ return; }
  x[idx[i/d.y]*d.y + (i%d.y)] = y[i];
}`;

  S.copyRow = `
@group(0) @binding(0) var<storage,read> x:array<f32>;
@group(0) @binding(1) var<storage,read_write> y:array<f32>;
@group(0) @binding(2) var<uniform> d:vec2<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  if(gid.x >= d.x){ return; }
  y[gid.x] = x[d.y*d.x + gid.x];
}`;

  const dim4 = (a, b, c, d) => new Uint32Array([a | 0, b | 0, c | 0, d | 0]);
  const dim2 = (a, b) => new Uint32Array([a | 0, b | 0]);

  class LJVGPU {
    constructor(manifest, buffers, device) {
      this.manifest = manifest;
      this.cfg = manifest.cfg;
      this.device = device;
      this.pipes = {};
      this.bufs = {};
      this.cpu = {};
      this._pool = {};
      this.errors = [];
      this.pos = 0;
      this.ok = false;
      device.addEventListener("uncapturederror", (e) => {
        this.errors.push(String(e.error && e.error.message || e.message));
      });
      const u8 = buffers.length === 1 ? new Uint8Array(buffers[0])
        : (() => {
          const u = new Uint8Array(manifest.total_bytes);
          let o = 0;
          for (const b of buffers) { u.set(new Uint8Array(b), o); o += b.byteLength; }
          return u;
        })();
      this.u16 = new Uint16Array(u8.buffer, u8.byteOffset);
    }

    static async requestDevice(adapter) {
      try {
        return await adapter.requestDevice({ requiredFeatures: ["shader-f16"] });
      } catch (e) {
        return await adapter.requestDevice();
      }
    }

    _f32(name) {
      if (this.cpu[name]) return this.cpu[name];
      const info = this.manifest.tensors[name];
      const n = info.bytes >> 1;
      const arr = new Float32Array(n);
      const start = info.offset >> 1;
      for (let i = 0; i < n; i++) {
        const h = this.u16[start + i];
        const ex = (h & 0x7c00) >> 10;
        let val;
        if (ex === 0) val = (h & 0x3ff) * Math.pow(2, -24);
        else val = (1 + (h & 0x3ff) / 1024) * Math.pow(2, ex - 15) * ((h & 0x8000) ? -1 : 1);
        arr[i] = val;
      }
      this.cpu[name] = arr;
      return arr;
    }

    async upload(onProgress) {
      const dev = this.device;
      const names = Object.keys(this.manifest.tensors);
      const biasOf = {};
      for (let l = 0; l < this.cfg.v_layers; l++) {
        const p = "vis.layers." + l + ".";
        for (const n of ["q_proj", "k_proj", "v_proj", "out_proj"]) {
          biasOf[p + "self_attn." + n + ".weight"] = p + "self_attn." + n + ".bias";
        }
      }
      biasOf["proj.fc1.weight"] = "proj.fc1.bias";
      biasOf["proj.fc2.weight"] = "proj.fc2.bias";
      let done = 0;
      for (const name of names) {
        const info = this.manifest.tensors[name];
        const start = info.offset >> 1, n = info.bytes >> 1;
        let u16 = this.u16.subarray(start, start + n);
        const biasName = biasOf[name];
        if (biasName) {
          const bi = this.manifest.tensors[biasName];
          const bb = this.u16.subarray(bi.offset >> 1, (bi.offset >> 1) + (bi.bytes >> 1));
          const all = new Uint16Array(n + bb.length);
          all.set(u16, 0); all.set(bb, n);
          u16 = all;
        }
        const buf = dev.createBuffer({
          size: Math.ceil(u16.length * 2 / 4) * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        dev.queue.writeBuffer(buf, 0, u16);
        this.bufs[name] = buf;
        done++;
        if (onProgress && done % 24 === 0) onProgress(done / names.length);
      }
      if (onProgress) onProgress(1);
      const D = this.cfg.d_model;
      for (let l = 0; l < this.cfg.n_layer; l++) {
        const n = this.cfg.max_seq * D;
        for (const t of ["kv_k", "kv_v"]) {
          const b = dev.createBuffer({
            size: n * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          });
          dev.queue.writeBuffer(b, 0, new Float32Array(n));
          this.bufs[`${t}_${l}`] = b;
        }
      }
      this.ready = true;
    }

    pipe(name) {
      if (!this.pipes[name]) {
        this.pipes[name] = this.device.createComputePipeline({
          layout: "auto",
          compute: { module: this.device.createShaderModule({ code: S[name] }), entryPoint: "main" },
        });
      }
      return this.pipes[name];
    }

    run(name, bufs, gx, gy, gz) {
      const p = this.pipe(name);
      const bind = this.device.createBindGroup({
        layout: p.getBindGroupLayout(0),
        entries: bufs.map((b, i) => ({ binding: i, resource: b })),
      });
      const enc = this.device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(p);
      pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(gx || 1, gy || 1, gz || 1);
      pass.end();
      this.device.queue.submit([enc.finish()]);
    }

    async readback(buf, bytes) {
      const dev = this.device;
      const st = dev.createBuffer({
        size: Math.ceil(bytes / 4) * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const enc = dev.createCommandEncoder();
      enc.copyBufferToBuffer(buf, 0, st, 0, bytes);
      dev.queue.submit([enc.finish()]);
      await st.mapAsync(GPUMapMode.READ);
      const out = new Float32Array(st.getMappedRange().slice(0));
      st.unmap();
      st.destroy();
      return out;
    }

    scratch(key, bytes) {
      bytes = Math.ceil(bytes / 4) * 4;
      this._pool = this._pool || {};
      if (!this._pool[key] || this._pool[key].size < bytes) {
        if (this._pool[key]) this._pool[key].destroy();
        this._pool[key] = this.device.createBuffer({
          size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
      }
      return this._pool[key];
    }

    uni(a, b, c, d) {
      const u = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(u, 0, dim4(a, b, c, d));
      return u;
    }
    uni2(a, b) {
      const u = this.device.createBuffer({ size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(u, 0, dim2(a, b));
      return u;
    }

    async prefill(ids, imageEmb) {
      const cfg = this.cfg, dev = this.device, D = cfg.d_model;
      const T = ids.length;
      this.pos = T;
      for (let l = 0; l < this.cfg.n_layer; l++) {
        dev.queue.writeBuffer(this.bufs[`kv_k_${l}`], 0, new Float32Array(this.cfg.max_seq * D));
        dev.queue.writeBuffer(this.bufs[`kv_v_${l}`], 0, new Float32Array(this.cfg.max_seq * D));
      }
      const x = new Float32Array(T * D);
      const E = this._f32("tok_emb");
      let im = 0;
      for (let t = 0; t < T; t++) {
        if (ids[t] === this.manifest.special.img && imageEmb) {
          x.set(imageEmb.subarray(im * D, im * D + D), t * D);
          im++;
        } else {
          x.set(E.subarray(ids[t] * D, ids[t] * D + D), t * D);
        }
      }
      let xcur = this.scratch("x", T * D * 4);
      dev.queue.writeBuffer(xcur, 0, x);
      xcur = await this._layers(xcur, T, 0);
      return await this._head(xcur, T);
    }

    async decode(id) {
      const cfg = this.cfg, dev = this.device, D = cfg.d_model;
      const T = 1, pos0 = this.pos;
      const x = new Float32Array(D);
      const E = this._f32("tok_emb");
      x.set(E.subarray(id * D, id * D + D));
      let xcur = this.scratch("xd", D * 4);
      dev.queue.writeBuffer(xcur, 0, x);
      xcur = await this._layers(xcur, T, pos0);
      this.pos = pos0 + T;
      return await this._head(xcur, T);
    }

    async _layers(xb, T, pos0) {
      const cfg = this.cfg, dev = this.device, D = cfg.d_model;
      let xcur = xb;
      if (this.dbg) {
        await dev.queue.onSubmittedWorkDone();
        const e = await this.readback(xcur, T * D * 4);
        this.dbg.push({ tag: "emb", v: Array.from(e.slice((T - 1) * D, T * D)).slice(0, 6).map(v => +v.toFixed(6)) });
      }
      for (let l = 0; l < cfg.n_layer; l++) {
        const inv1 = this.scratch(`inv1_${l}`, T * 4);
        this.run("rmsMean", [{ buffer: xcur }, { buffer: inv1 }, { buffer: this.uni2(T, D) }], T);
        const n1 = this.scratch(`n1_${l}`, T * D * 4);
        this.run("rmsApply", [
          { buffer: xcur }, { buffer: this.bufs[`blocks.${l}.norm1.weight`] },
          { buffer: inv1 }, { buffer: n1 }, { buffer: this.uni2(T, D) },
        ], Math.ceil(T * D / 64));
        const qkv = this.scratch(`qkv_${l}`, T * 3 * D * 4);
        this.run("gemm", [
          { buffer: n1 }, { buffer: this.bufs[`blocks.${l}.attn.qkv.weight`] },
          { buffer: qkv }, { buffer: this.uni(T, 3 * D, D, 0) },
        ], Math.ceil(T * 3 * D / 64));
        const q = this.scratch(`q_${l}`, T * D * 4);
        this.run("ropeKV", [
          { buffer: qkv }, { buffer: q },
          { buffer: this.bufs[`kv_k_${l}`] }, { buffer: this.bufs[`kv_v_${l}`] },
          { buffer: this.uni(T, D, cfg.n_head, pos0) },
        ], Math.ceil(T * 1152 / 64));
        const att = this.scratch(`att_${l}`, T * D * 4);
        this.run("attnCausal", [
          { buffer: q }, { buffer: this.bufs[`kv_k_${l}`] },
          { buffer: this.bufs[`kv_v_${l}`] }, { buffer: att },
          { buffer: this.uni(T, D, cfg.n_head, pos0) },
        ], T, cfg.n_head);
        const o = this.scratch(`o_${l}`, T * D * 4);
        this.run("gemm", [
          { buffer: att }, { buffer: this.bufs[`blocks.${l}.attn.o.weight`] },
          { buffer: o }, { buffer: this.uni(T, D, D, 0) },
        ], Math.ceil(T * D / 64));
        const xa = this.scratch(`xa_${l}`, T * D * 4);
        this.run("add", [{ buffer: xcur }, { buffer: o }, { buffer: xa }], Math.ceil(T * D / 64));
        xcur = xa;
        const inv2 = this.scratch(`inv2_${l}`, T * 4);
        this.run("rmsMean", [{ buffer: xcur }, { buffer: inv2 }, { buffer: this.uni2(T, D) }], T);
        const n2 = this.scratch(`n2_${l}`, T * D * 4);
        this.run("rmsApply", [
          { buffer: xcur }, { buffer: this.bufs[`blocks.${l}.norm2.weight`] },
          { buffer: inv2 }, { buffer: n2 }, { buffer: this.uni2(T, D) },
        ], Math.ceil(T * D / 64));
        const rt = this.scratch(`rt_${l}`, T * cfg.n_expert * 4);
        this.run("gemm", [
          { buffer: n2 }, { buffer: this.bufs[`blocks.${l}.moe.router.weight`] },
          { buffer: rt }, { buffer: this.uni(T, cfg.n_expert, D, 0) },
        ], Math.ceil(T * cfg.n_expert / 64));
        await dev.queue.onSubmittedWorkDone();
        const routes = await this.readback(rt, T * cfg.n_expert * 4);
        const groups = [];
        for (let e = 0; e < cfg.n_expert; e++) groups.push([]);
        for (let t = 0; t < T; t++) {
          let best = 0, bv = -Infinity;
          for (let e = 0; e < cfg.n_expert; e++) {
            if (routes[t * cfg.n_expert + e] > bv) { bv = routes[t * cfg.n_expert + e]; best = e; }
          }
          groups[best].push(t);
        }
        const moe = this.scratch(`moe_${l}`, T * D * 4);
        dev.queue.writeBuffer(moe, 0, new Float32Array(T * D));
        for (let e = 0; e < cfg.n_expert; e++) {
          const rows = groups[e];
          if (!rows.length) continue;
          const m = rows.length;
          const ib = this.scratch(`gi_${l}_${e}`, m * 4);
          dev.queue.writeBuffer(ib, 0, new Uint32Array(rows));
          const g = this.scratch(`g_${l}_${e}`, m * D * 4);
          this.run("gather", [{ buffer: n2 }, { buffer: ib }, { buffer: g },
            { buffer: this.uni2(m, D) }], Math.ceil(m * D / 64));
          const h1 = this.scratch(`h1_${l}_${e}`, m * cfg.inter * 4);
          const w1 = { buffer: this.bufs[`blocks.${l}.moe.w1_t`], offset: e * D * cfg.inter * 2, size: D * cfg.inter * 2 };
          this.run("gemm", [{ buffer: g }, w1, { buffer: h1 },
            { buffer: this.uni(m, cfg.inter, D, 0) }], Math.ceil(m * cfg.inter / 64));
          this.run("gelu", [{ buffer: h1 }], Math.ceil(m * cfg.inter / 64));
          const h2 = this.scratch(`h2_${l}_${e}`, m * D * 4);
          const w2 = { buffer: this.bufs[`blocks.${l}.moe.w2_t`], offset: e * cfg.inter * D * 2, size: cfg.inter * D * 2 };
          this.run("gemm", [{ buffer: h1 }, w2, { buffer: h2 },
            { buffer: this.uni(m, D, cfg.inter, 0) }], Math.ceil(m * D / 64));
          // 每个 token 的 softmax 概率：取被选中那个专家的
          const pw = this.scratch(`pw_${l}_${e}`, m * 4);
          const pwe = new Float32Array(m);
          for (let r = 0; r < m; r++) {
            const t = rows[r];
            let mx = -Infinity, sum = 0;
            for (let e2 = 0; e2 < cfg.n_expert; e2++) mx = Math.max(mx, routes[t * cfg.n_expert + e2]);
            for (let e2 = 0; e2 < cfg.n_expert; e2++) sum += Math.exp(routes[t * cfg.n_expert + e2] - mx);
            pwe[r] = Math.exp(routes[t * cfg.n_expert + e] - mx) / sum;
          }
          dev.queue.writeBuffer(pw, 0, pwe);
          const ob = this.scratch(`ob_${l}_${e}`, m * D * 4);
          this.run("mulRow", [{ buffer: h2 }, { buffer: pw }, { buffer: ob },
            { buffer: this.uni2(m, D) }], Math.ceil(m * D / 64));
          this.run("scatter", [{ buffer: ob }, { buffer: ib }, { buffer: moe },
            { buffer: this.uni2(m, D) }], Math.ceil(m * D / 64));
        }
        const xnext = this.scratch(`xnext_${l}`, T * D * 4);
        this.run("add", [{ buffer: xcur }, { buffer: moe }, { buffer: xnext }], Math.ceil(T * D / 64));
        xcur = xnext;
        if (this.dbg) {
          await dev.queue.onSubmittedWorkDone();
          const r = await this.readback(xcur, T * D * 4);
          this.dbg.push({ tag: "block" + l, v: Array.from(r.slice((T - 1) * D, T * D)).slice(0, 6).map(v => +v.toFixed(6)) });
        }
      }
      return xcur;
    }

    async _head(xcur, T) {
      const cfg = this.cfg, dev = this.device, D = cfg.d_model;
      const inv = this.scratch("invF", T * 4);
      this.run("rmsMean", [{ buffer: xcur }, { buffer: inv }, { buffer: this.uni2(T, D) }], T);
      const hf = this.scratch("hf", T * D * 4);
      this.run("rmsApply", [
        { buffer: xcur }, { buffer: this.bufs["norm_f"] }, { buffer: inv },
        { buffer: hf }, { buffer: this.uni2(T, D) },
      ], Math.ceil(T * D / 64));
      const last = this.scratch("last", D * 4);
      this.run("copyRow", [{ buffer: hf }, { buffer: last }, { buffer: this.uni2(D, T - 1) }], Math.ceil(D / 64));
      const lg = this.scratch("logits", cfg.vocab_size * 4);
      this.run("gemm", [
        { buffer: last }, { buffer: this.bufs["tok_emb"] },
        { buffer: lg }, { buffer: this.uni(1, cfg.vocab_size, D, 0) },
      ], Math.ceil(cfg.vocab_size / 64));
      await dev.queue.onSubmittedWorkDone();
      if (this.dbg) {
        const hfr = await this.readback(hf, T * D * 4);
        this.dbg.push({ tag: "final", v: Array.from(hfr.slice((T - 1) * D, T * D)).slice(0, 6).map(v => +v.toFixed(6)) });
      }
      return await this.readback(lg, cfg.vocab_size * 4);
    }

    /* 视觉：pixelData f32[224*224*3]（CLIP 归一化）→ f32[49*384] */
    async encodeImage(pixelData) {
      const cfg = this.cfg, dev = this.device;
      const P = 49, VH = cfg.v_hidden, VL = cfg.v_layers;
      const pw = this._f32("vis.patch_embedding.weight");
      const patches = new Float32Array(P * 3072);
      for (let py = 0; py < 7; py++) {
        for (let px = 0; px < 7; px++) {
          const pi = py * 7 + px;
          for (let c = 0; c < 3; c++) {
            for (let yy = 0; yy < 32; yy++) {
              for (let xx = 0; xx < 32; xx++) {
                const src = ((py * 32 + yy) * 224 + px * 32 + xx) * 3 + c;
                patches[pi * 3072 + c * 1024 + yy * 32 + xx] = pixelData[src];
              }
            }
          }
        }
      }
      const pb = dev.createBuffer({ size: P * 3072 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      dev.queue.writeBuffer(pb, 0, patches);
      const wb = dev.createBuffer({ size: VH * 3072 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      dev.queue.writeBuffer(wb, 0, pw);
      const pe0 = this.scratch("v_pe", P * VH * 4);
      this.run("gemm", [{ buffer: pb }, { buffer: wb }, { buffer: pe0 },
        { buffer: this.uni(P, VH, 3072, 0) }], Math.ceil(P * VH / 64));
      await dev.queue.onSubmittedWorkDone();
      const pe = await this.readback(pe0, P * VH * 4);
      const cls = this._f32("vis.class_embedding");
      const pos = this._f32("vis.position_embedding");
      const seq = new Float32Array(50 * VH);
      for (let j = 0; j < VH; j++) seq[j] = cls[j] + pos[j];
      for (let i = 1; i <= P; i++) {
        for (let j = 0; j < VH; j++) seq[i * VH + j] = pe[(i - 1) * VH + j] + pos[i * VH + j];
      }
      const sb = dev.createBuffer({ size: 50 * VH * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      dev.queue.writeBuffer(sb, 0, seq);
      let cur = this.scratch("v_pl", 50 * VH * 4);
      this.run("layernorm", [
        { buffer: sb }, { buffer: this.bufs["vis.pre_layrnorm.weight"] },
        { buffer: this.bufs["vis.pre_layrnorm.bias"] }, { buffer: cur },
        { buffer: this.uni2(50, VH) },
      ], 50);
      for (let l = 0; l < VL; l++) {
        const p = `vis.layers.${l}.`;
        const ln1 = this.scratch(`v_ln1_${l}`, 50 * VH * 4);
        this.run("layernorm", [
          { buffer: cur }, { buffer: this.bufs[p + "layer_norm1.weight"] },
          { buffer: this.bufs[p + "layer_norm1.bias"] }, { buffer: ln1 },
          { buffer: this.uni2(50, VH) },
        ], 50);
        const q = this.scratch(`v_q_${l}`, 50 * VH * 4);
        this.run("gemm", [{ buffer: ln1 }, { buffer: this.bufs[p + "self_attn.q_proj.weight"] },
          { buffer: q }, { buffer: this.uni(50, VH, VH, 1) }], Math.ceil(50 * VH / 64));
        const k = this.scratch(`v_k_${l}`, 50 * VH * 4);
        this.run("gemm", [{ buffer: ln1 }, { buffer: this.bufs[p + "self_attn.k_proj.weight"] },
          { buffer: k }, { buffer: this.uni(50, VH, VH, 1) }], Math.ceil(50 * VH / 64));
        const v = this.scratch(`v_v_${l}`, 50 * VH * 4);
        this.run("gemm", [{ buffer: ln1 }, { buffer: this.bufs[p + "self_attn.v_proj.weight"] },
          { buffer: v }, { buffer: this.uni(50, VH, VH, 1) }], Math.ceil(50 * VH / 64));
        const att = this.scratch(`v_att_${l}`, 50 * VH * 4);
        this.run("attnFull", [{ buffer: q }, { buffer: k }, { buffer: v }, { buffer: att },
          { buffer: this.uni(50, VH, cfg.v_heads, 50) }], 50, cfg.v_heads);
        const ao = this.scratch(`v_ao_${l}`, 50 * VH * 4);
        this.run("gemm", [{ buffer: att }, { buffer: this.bufs[p + "self_attn.out_proj.weight"] },
          { buffer: ao }, { buffer: this.uni(50, VH, VH, 1) }], Math.ceil(50 * VH / 64));
        const r1 = this.scratch(`v_res_${l}`, 50 * VH * 4);
        this.run("add", [{ buffer: cur }, { buffer: ao }, { buffer: r1 }], Math.ceil(50 * VH / 64));
        cur = r1;
        const ln2 = this.scratch(`v_ln2_${l}`, 50 * VH * 4);
        this.run("layernorm", [
          { buffer: cur }, { buffer: this.bufs[p + "layer_norm2.weight"] },
          { buffer: this.bufs[p + "layer_norm2.bias"] }, { buffer: ln2 },
          { buffer: this.uni2(50, VH) },
        ], 50);
        const f1 = this.scratch(`v_f1_${l}`, 50 * cfg.v_inter * 4);
        this.run("gemm", [{ buffer: ln2 }, { buffer: this.bufs[p + "mlp.fc1.weight"] },
          { buffer: f1 }, { buffer: this.uni(50, cfg.v_inter, VH, 1) }], Math.ceil(50 * cfg.v_inter / 64));
        this.run("quickGelu", [{ buffer: f1 }], Math.ceil(50 * cfg.v_inter / 64));
        const f2 = this.scratch(`v_f2_${l}`, 50 * VH * 4);
        this.run("gemm", [{ buffer: f1 }, { buffer: this.bufs[p + "mlp.fc2.weight"] },
          { buffer: f2 }, { buffer: this.uni(50, VH, cfg.v_inter, 1) }], Math.ceil(50 * VH / 64));
        const r2 = this.scratch(`v_res2_${l}`, 50 * VH * 4);
        this.run("add", [{ buffer: cur }, { buffer: f2 }, { buffer: r2 }], Math.ceil(50 * VH / 64));
        cur = r2;
      }
      const post = this.scratch("v_post", 50 * VH * 4);
      this.run("layernorm", [
        { buffer: cur }, { buffer: this.bufs["vis.post_layernorm.weight"] },
        { buffer: this.bufs["vis.post_layernorm.bias"] }, { buffer: post },
        { buffer: this.uni2(50, VH) },
      ], 50);
      await dev.queue.onSubmittedWorkDone();
      const postCpu = await this.readback(post, 50 * VH * 4);
      const pr = new Float32Array(P * VH);
      pr.set(postCpu.subarray(VH, 50 * VH));
      const prb = dev.createBuffer({ size: P * VH * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      dev.queue.writeBuffer(prb, 0, pr);
      const pc1 = this.scratch("v_pc1", P * 768 * 4);
      this.run("gemm", [{ buffer: prb }, { buffer: this.bufs["proj.fc1.weight"] },
        { buffer: pc1 }, { buffer: this.uni(P, 768, VH, 1) }], Math.ceil(P * 768 / 64));
      this.run("gelu", [{ buffer: pc1 }], Math.ceil(P * 768 / 64));
      const pc2 = this.scratch("v_pc2", P * D * 4);
      this.run("gemm", [{ buffer: pc1 }, { buffer: this.bufs["proj.fc2.weight"] },
        { buffer: pc2 }, { buffer: this.uni(P, D, 768, 1) }], Math.ceil(P * D / 64));
      await dev.queue.onSubmittedWorkDone();
      return await this.readback(pc2, P * D * 4);
    }
  }

  root.LJV_SHADERS = S;
  root.LJVGPU = LJVGPU;
})(typeof window !== "undefined" ? window : globalThis);
