    /* 视觉：pixelData 为 f32[224*224*3]（CLIP 归一化后，RGB 逐像素）→ f32[49*384] */
    async encodeImage(pixelData) {
      const cfg = this.cfg, dev = this.device;
      const P = 49, VH = cfg.v_hidden, VL = cfg.v_layers;
      // patch embedding：权重 [512,3,32,32] 展平为 [512,3072]，patch 展开为 [49,3072]
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
      // cls + position + patches
      const cls = this._f32("vis.class_embedding");
      const pos = this._f32("vis.position_embedding");
      const seq = new Float32Array(50 * VH);
      for (let j = 0; j < VH; j++) seq[j] = cls[j] + pos[j];
      for (let i = 1; i <= P; i++) {
        for (let j = 0; j < VH; j++) seq[i * VH + j] = pe[(i - 1) * VH + j] + pos[i * VH + j];
      }
      const sb = dev.createBuffer({ size: 50 * VH * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      dev.queue.writeBuffer(sb, 0, seq);
      // pre_layernorm
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
        this.run("add", [{ buffer: cur }, { buffer: ao }, { buffer: cur }], Math.ceil(50 * VH / 64));
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
        this.run("add", [{ buffer: cur }, { buffer: f2 }, { buffer: cur }], Math.ceil(50 * VH / 64));
      }
      const post = this.scratch("v_post", 50 * VH * 4);
      this.run("layernorm", [
        { buffer: cur }, { buffer: this.bufs["vis.post_layernorm.weight"] },
        { buffer: this.bufs["vis.post_layernorm.bias"] }, { buffer: post },
        { buffer: this.uni2(50, VH) },
      ], 50);
      await dev.queue.onSubmittedWorkDone();
      // 取 49 个 patch（去掉 cls）→ projector
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
