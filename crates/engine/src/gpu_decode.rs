//! Device-resident decode orchestration (T4.2).
//!
//! **One token = one command buffer.** Every pass of a 60-layer forward —
//! projections, per-head norms, RoPE, attention, the gated MLP, the residual
//! adds, the streamed `lm_head` — is encoded into a single command buffer and
//! submitted once. Before this, 471 command buffers per token
//! (411 linears × 3 passes + 60 attentions) each paid `commit` +
//! `wait_until_completed`, measured at ~0.4 ms of overhead per linear, i.e.
//! ~180 ms of a 275 ms token.
//!
//! Consequences of that choice:
//!
//! * **Activations are device-resident.** The arena is allocated once and
//!   reused; `x` is a 3-slot ring so the residual adds need no copy — layer `i`
//!   reads slot `2i % 3` and writes `(2i + 2) % 3`.
//! * **K/V live in the device-backed cache** (`KvCache::new_device`). The RoPE
//!   pass for `k` writes *straight into the cache row*, and V is copied into its
//!   row; both are `StorageModeShared`, so the host appends/rotates rows with a
//!   plain memcpy and never needs a command buffer for it.
//! * **Only two host round trips per token**: the embedding row in (bf16 → f16 ×
//!   √hidden, 10 KB) and the logits out (the ~512 KB the sampler needs anyway).
//!
//! **T3.2 batched prefill (`encode_span`)** runs `tokens.len() >= 2` in chunks
//! of at most `batch_rows`: one command buffer per chunk (host KV rotations
//! between chunks must not race with the previous chunk's dispatches), every
//! linear in `rows = m` batched-GEMM mode (weight tiles decoded once per
//! out-tile and reused across rows), per-head norms flattened to
//! `m * n_heads` rows of `head_dim`, multi-row RoPE at `pos0 + row`, and one
//! attention dispatch with `s = m`. The arena (`x` ring, `q`, `gu`, ...) is
//! sized `[batch_rows, ...]` so the single-row decode path just uses row 0.
//!
//! Norms, RoPE tables and `layer_scalar`s are uploaded once at construction.
//!
//! Numerics: every pass mirrors the CPU reference, checked pass by pass in
//! `crates/kernels` tests. The one deliberate deviation is `layer_scalar`: the
//! CPU multiplies in f64 (DE28), the kernel in f32 (Apple GPUs have no f64), so
//! a double-rounding boundary can differ by one f16 ulp. That is bounded end to
//! end by the recorded-fixture test.

use half::f16;
use kaplumba_kernels::{AttnParams, DeviceSpan, Exl3Shape};
use metal::{Buffer, CommandBufferRef, ComputeCommandEncoderRef, Device, MTLResourceOptions};

use crate::gemma4::{Gemma4Model, GemmaError, LayerKind};
use crate::kv_cache::KvCache;
use crate::metal_linear::{LinearHandles, MetalLinear};

/// Maximum head dim in this model family (full-attention layers).
const MAX_HD: usize = 512;
/// Maximum KV heads (sliding layers).
const MAX_NKV: usize = 16;

fn half_buf(device: &Device, n: usize) -> Buffer {
    device.new_buffer((n * 2).max(2) as u64, MTLResourceOptions::StorageModeShared)
}

fn f32_up(device: &Device, v: &[f32]) -> Buffer {
    device.new_buffer_with_data(
        v.as_ptr() as *const _,
        (v.len() * 4).max(4) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn span(b: &Buffer) -> DeviceSpan<'_> {
    DeviceSpan {
        buffer: b,
        byte_offset: 0,
    }
}

/// Per-layer device state: uploaded norms, `layer_scalar`, and the seven linear
/// weight handles.
struct LayerW {
    input_ln: Buffer,
    post_attn_ln: Buffer,
    pre_ff_ln: Buffer,
    post_ff_ln: Buffer,
    q_norm: Buffer,
    k_norm: Buffer,
    scalar: f32,
    head_dim: usize,
    n_kv: usize,
    kind: LayerKind,
    q: LinearHandles,
    k: LinearHandles,
    v: Option<LinearHandles>,
    o: LinearHandles,
    gate: LinearHandles,
    up: LinearHandles,
    down: LinearHandles,
}

/// Activation arena + weight staging for one model.
pub struct GpuDecoder {
    hidden: usize,
    intermediate: usize,
    n_heads: usize,
    vocab: usize,
    /// Maximum rows per prefill chunk (T3.2). Every arena buffer except
    /// `xn_final`/`logits_h` (always one row) is sized `[batch_rows, ...]`;
    /// the decode path uses row 0 of the same buffers.
    batch_rows: usize,
    /// `cache_cap` this decoder was built for: RoPE tables are sized once, so a
    /// later cache with a **larger** capacity must rebuild (`ensure_decoder`).
    pub(crate) built_cap: usize,
    /// Residual-stream ring (see module docs).
    x: [Buffer; 3],
    xn: Buffer,
    q_raw: Buffer,
    q: Buffer,
    k_raw: Buffer,
    k_norm: Buffer,
    v_raw: Buffer,
    attn: Buffer,
    o_proj: Buffer,
    o_norm: Buffer,
    gu_gate: Buffer,
    gu_up: Buffer,
    gu: Buffer,
    down: Buffer,
    down_norm: Buffer,
    xn_final: Buffer,
    logits_h: Buffer,
    lm: LinearHandles,
    final_norm: Buffer,
    rope_sliding: (Buffer, Buffer),
    rope_full: (Buffer, Buffer),
    layers: Vec<LayerW>,
}

impl GpuDecoder {
    /// Upload norms/RoPE tables and allocate the arena.
    ///
    /// `Ok(None)` means the model is outside what the kernels cover (head dim not
    /// 256/512, `Mul1` codebook, missing signs, 2-byte-misaligned span, ...) so
    /// the caller keeps the host path instead of failing.
    pub fn new(
        model: &Gemma4Model<'_>,
        gpu: &MetalLinear,
        cache_cap: usize,
    ) -> Result<Option<Self>, GemmaError> {
        let device = gpu.device();
        let cfg = &model.cfg;
        let src = model.source();
        let hidden = cfg.hidden;
        let intermediate = cfg.intermediate;
        let n_heads = cfg.n_heads;
        let vocab = cfg.vocab;

        let handles = |name: &str| -> Result<Option<LinearHandles>, GemmaError> {
            match crate::metal_linear::linear_handles(src, name) {
                Ok(h) => Ok(Some(h)),
                Err(_) => Ok(None),
            }
        };
        let Some(lm) = handles("lm_head")? else {
            return Ok(None);
        };
        if lm.shape.in_features() != hidden || lm.shape.out_features() != vocab {
            return Ok(None);
        }

        // RoPE tables for both layer kinds, sized to the context we allow.
        let cap = cache_cap.max(1);
        let mut rope_sliding = None;
        let mut rope_full = None;
        for lc in cfg.layers.iter() {
            let slot = match lc.kind {
                LayerKind::Sliding => &mut rope_sliding,
                LayerKind::Full => &mut rope_full,
            };
            if slot.is_none() {
                *slot = Some(build_rope_tables(device, lc.head_dim, &lc.rope, cap));
            }
        }
        let (Some(rope_sliding), Some(rope_full)) = (rope_sliding, rope_full) else {
            return Ok(None);
        };

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for (i, lc) in cfg.layers.iter().enumerate() {
            let p = format!("model.language_model.layers.{i}");
            let norm = |name: &str, n: usize| -> Result<Buffer, GemmaError> {
                let v = model.f32_tensor(&format!("{p}.{name}"))?;
                if v.len() != n {
                    return Err(GemmaError::Shape {
                        name: format!("{p}.{name}"),
                        msg: format!("expected {n} values, got {}", v.len()),
                    });
                }
                Ok(f32_up(device, &v))
            };
            let need =
                |name: &str, h: Option<LinearHandles>| -> Result<LinearHandles, GemmaError> {
                    h.ok_or_else(|| GemmaError::Config(format!("{p}.{name} is not GPU-eligible")))
                };
            let q = need(
                "self_attn.q_proj",
                handles(&format!("{p}.self_attn.q_proj"))?,
            )?;
            let k = need(
                "self_attn.k_proj",
                handles(&format!("{p}.self_attn.k_proj"))?,
            )?;
            let v = handles(&format!("{p}.self_attn.v_proj"))?;
            let o = need(
                "self_attn.o_proj",
                handles(&format!("{p}.self_attn.o_proj"))?,
            )?;
            let gate = need("mlp.gate_proj", handles(&format!("{p}.mlp.gate_proj"))?)?;
            let up = need("mlp.up_proj", handles(&format!("{p}.mlp.up_proj"))?)?;
            let down = need("mlp.down_proj", handles(&format!("{p}.mlp.down_proj"))?)?;
            let k_eq_v = lc.kind == LayerKind::Full && cfg.k_eq_v;
            if k_eq_v == v.is_some() {
                // Either the layer reuses k for values, or it must have v_proj.
                return Ok(None);
            }
            layers.push(LayerW {
                input_ln: norm("input_layernorm.weight", hidden)?,
                post_attn_ln: norm("post_attention_layernorm.weight", hidden)?,
                pre_ff_ln: norm("pre_feedforward_layernorm.weight", hidden)?,
                post_ff_ln: norm("post_feedforward_layernorm.weight", hidden)?,
                q_norm: norm("self_attn.q_norm.weight", lc.head_dim)?,
                k_norm: norm("self_attn.k_norm.weight", lc.head_dim)?,
                scalar: model.layer_scalar(i)? as f32,
                head_dim: lc.head_dim,
                n_kv: lc.n_kv,
                kind: lc.kind,
                q,
                k,
                v,
                o,
                gate,
                up,
                down,
            });
        }
        let final_norm = {
            let v = model.f32_tensor("model.language_model.norm.weight")?;
            f32_up(device, &v)
        };

        // Prefill chunk size (T3.2): 256 caps arena memory (~80 MB) and GEMM
        // grid height; the sliding-window reserve rejects n > window, so the
        // chunk must also fit min(window, cache_cap).
        let mut batch_rows = cache_cap.max(1).min(256);
        for lc in cfg.layers.iter() {
            if lc.kind == LayerKind::Sliding {
                batch_rows = batch_rows.min(cfg.sliding_window.max(1));
            }
        }

        let a = |n: usize| half_buf(device, n);
        Ok(Some(Self {
            hidden,
            intermediate,
            n_heads,
            vocab,
            batch_rows,
            built_cap: cache_cap.max(1),
            x: [
                a(hidden * batch_rows),
                a(hidden * batch_rows),
                a(hidden * batch_rows),
            ],
            xn: a(hidden * batch_rows),
            q_raw: a(batch_rows * n_heads * MAX_HD),
            q: a(batch_rows * n_heads * MAX_HD),
            k_raw: a(batch_rows * MAX_NKV * MAX_HD),
            k_norm: a(batch_rows * MAX_NKV * MAX_HD),
            v_raw: a(batch_rows * MAX_NKV * MAX_HD),
            attn: a(batch_rows * n_heads * MAX_HD),
            o_proj: a(hidden * batch_rows),
            o_norm: a(hidden * batch_rows),
            gu_gate: a(intermediate * batch_rows),
            gu_up: a(intermediate * batch_rows),
            gu: a(intermediate * batch_rows),
            down: a(hidden * batch_rows),
            down_norm: a(hidden * batch_rows),
            xn_final: a(hidden),
            logits_h: a(vocab),
            lm,
            final_norm,
            rope_sliding,
            rope_full,
            layers,
        }))
    }

    /// Encode one token into `cb` (which the caller commits) and return the
    /// logits once it has completed.
    pub fn encode_token(
        &self,
        model: &Gemma4Model<'_>,
        gpu: &MetalLinear,
        cb: &CommandBufferRef,
        cache: &mut KvCache,
        token: u32,
        upto_layers: usize,
    ) -> Result<Vec<f32>, GemmaError> {
        if !cache.is_device_backed() {
            return Err(GemmaError::Config(
                "gpu decode needs a device-backed kv cache".into(),
            ));
        }
        let pos = cache.len();
        let hidden = self.hidden;
        let eps = model.cfg.eps as f32;

        // Host: embedding row (bf16 → f16 ×√hidden) into ring slot 0.
        {
            let row = model.embed_row(token)?;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    row.as_ptr() as *const u8,
                    self.x[0].contents() as *mut u8,
                    hidden * 2,
                );
            }
        }

        let upto = upto_layers.min(self.layers.len());
        let enc = cb.new_compute_command_encoder();
        for i in 0..upto {
            let lw = &self.layers[i];
            let x_in = &self.x[(2 * i) % 3];
            let x_mid = &self.x[(2 * i + 1) % 3];
            let x_out = &self.x[(2 * i + 2) % 3];
            let (cos, sin) = match lw.kind {
                LayerKind::Sliding => (&self.rope_sliding.0, &self.rope_sliding.1),
                LayerKind::Full => (&self.rope_full.0, &self.rope_full.1),
            };
            let hd = lw.head_dim;

            // --- attention sublayer ---
            gpu.encode_rms(
                enc,
                span(x_in),
                Some(span(&lw.input_ln)),
                span(&self.xn),
                1,
                hidden,
                eps,
            )?;
            gpu.encode_linear(enc, &lw.q, &self.xn, &self.q_raw, 1)?;
            gpu.encode_linear(enc, &lw.k, &self.xn, &self.k_raw, 1)?;
            // q/k per-head norms, then rope. Rope reads two elements per output
            // element, so it always needs distinct in/out buffers: q ropes
            // q → q_raw (q_raw is free once the norm above has read it) and
            // attention consumes q_raw.
            gpu.encode_rms(
                enc,
                span(&self.q_raw),
                Some(span(&lw.q_norm)),
                span(&self.q),
                self.n_heads,
                hd,
                eps,
            )?;
            gpu.encode_rms(
                enc,
                span(&self.k_raw),
                Some(span(&lw.k_norm)),
                span(&self.k_norm),
                lw.n_kv,
                hd,
                eps,
            )?;
            // v: its own projection, or (k_eq_v) a weightless norm of the RAW k —
            // which is why it is computed before the roped k overwrites nothing
            // and never reads a roped buffer.
            match &lw.v {
                Some(lin) => gpu.encode_linear(enc, lin, &self.xn, &self.v_raw, 1)?,
                None => gpu.encode_rms(
                    enc,
                    span(&self.k_raw),
                    None,
                    span(&self.v_raw),
                    lw.n_kv,
                    hd,
                    eps,
                )?,
            }
            gpu.encode_rope(
                enc,
                span(&self.q),
                span(cos),
                span(sin),
                span(&self.q_raw),
                self.n_heads,
                hd,
                pos,
                1,
            )?;

            // Append this token's row: rotation is host-side (shared memory), the
            // row contents are written by the passes below/above.
            let (kv_pos0, kv_rows) = cache.reserve_row(&model.cfg, i, pos)?;
            gpu.encode_rope(
                enc,
                span(&self.k_norm),
                span(cos),
                span(sin),
                cache
                    .row_span(i, false, kv_rows - 1)
                    .ok_or_else(|| GemmaError::Config("kv k row span".into()))?,
                lw.n_kv,
                hd,
                pos,
                1,
            )?;
            gpu.encode_copy(
                enc,
                span(&self.v_raw),
                cache
                    .row_span(i, true, kv_rows - 1)
                    .ok_or_else(|| GemmaError::Config("kv v row span".into()))?,
                lw.n_kv * hd,
            )?;

            gpu.encode_attention(
                enc,
                span(&self.q_raw),
                cache
                    .layer_keys_span(i)
                    .ok_or_else(|| GemmaError::Config("kv k span".into()))?,
                cache
                    .layer_values_span(i)
                    .ok_or_else(|| GemmaError::Config("kv v span".into()))?,
                span(&self.attn),
                AttnParams {
                    s: 1,
                    kv_rows,
                    q_pos0: pos,
                    kv_pos0,
                    n_heads: self.n_heads,
                    n_kv: lw.n_kv,
                    head_dim: hd,
                    window: crate::gemma4::layer_window(lw.kind, model.cfg.sliding_window),
                },
            )?;
            gpu.encode_linear(enc, &lw.o, &self.attn, &self.o_proj, 1)?;
            gpu.encode_rms(
                enc,
                span(&self.o_proj),
                Some(span(&lw.post_attn_ln)),
                span(&self.o_norm),
                1,
                hidden,
                eps,
            )?;
            gpu.encode_add(enc, span(x_in), span(&self.o_norm), span(x_mid), hidden)?;

            // --- MLP sublayer ---
            gpu.encode_rms(
                enc,
                span(x_mid),
                Some(span(&lw.pre_ff_ln)),
                span(&self.xn),
                1,
                hidden,
                eps,
            )?;
            gpu.encode_linear(enc, &lw.gate, &self.xn, &self.gu_gate, 1)?;
            gpu.encode_linear(enc, &lw.up, &self.xn, &self.gu_up, 1)?;
            gpu.encode_gelu_mul(
                enc,
                span(&self.gu_gate),
                span(&self.gu_up),
                span(&self.gu),
                self.intermediate,
            )?;
            gpu.encode_linear(enc, &lw.down, &self.gu, &self.down, 1)?;
            gpu.encode_rms(
                enc,
                span(&self.down),
                Some(span(&lw.post_ff_ln)),
                span(&self.down_norm),
                1,
                hidden,
                eps,
            )?;
            gpu.encode_add_scale(
                enc,
                span(x_mid),
                span(&self.down_norm),
                span(x_out),
                hidden,
                lw.scalar,
            )?;
        }

        // --- final norm + lm_head (whole head in one dispatch; the CPU path
        // streams 128-column slabs only to save host memory) ---
        let last_x = &self.x[(2 * upto) % 3];
        gpu.encode_rms(
            enc,
            span(last_x),
            Some(span(&self.final_norm)),
            span(&self.xn_final),
            1,
            hidden,
            eps,
        )?;
        gpu.encode_linear(enc, &self.lm, &self.xn_final, &self.logits_h, 1)?;
        enc.end_encoding();

        // --- the single sync of the token ---
        cb.commit();
        cb.wait_until_completed();
        let logits: &[f16] = unsafe {
            std::slice::from_raw_parts(self.logits_h.contents() as *const f16, self.vocab)
        };
        let mut out: Vec<f32> = logits.iter().map(|v| v.to_f32()).collect();
        if let Some(cap) = model.cfg.softcap {
            crate::gemma4::apply_softcap(&mut out, cap);
        }
        Ok(out)
    }

    /// Encode a whole span (T3.2 batched prefill), returning the **last**
    /// token's logits. Chunks of at most `self.batch_rows` rows; **one command
    /// buffer per chunk, committed and waited before the next** — the host-side
    /// KV rotation of chunk `c+1` must not race with chunk `c`'s dispatches.
    ///
    /// Same maths as `tokens.len()` × [`Self::encode_token`] (every pass is
    /// row-independent), asserted bit-identical in the gated fixture test.
    pub fn encode_span(
        &self,
        model: &Gemma4Model<'_>,
        gpu: &MetalLinear,
        cache: &mut KvCache,
        tokens: &[u32],
        upto_layers: usize,
    ) -> Result<Vec<f32>, GemmaError> {
        if tokens.is_empty() {
            return Err(GemmaError::Config("encode_span(empty)".into()));
        }
        if !cache.is_device_backed() {
            return Err(GemmaError::Config(
                "gpu prefill needs a device-backed kv cache".into(),
            ));
        }
        let hidden = self.hidden;
        let eps = model.cfg.eps as f32;
        let upto = upto_layers.min(self.layers.len());
        let base_pos = cache.len();
        let mut offset = 0usize;

        while offset < tokens.len() {
            let m = (tokens.len() - offset).min(self.batch_rows);
            let pos0 = base_pos + offset;
            let is_last = offset + m == tokens.len();

            // Host: the chunk's embeddings (bf16 → f16 ×√hidden) into ring slot 0.
            {
                let emb = model.embed(&tokens[offset..offset + m])?;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        emb.as_ptr() as *const u8,
                        self.x[0].contents() as *mut u8,
                        m * hidden * 2,
                    );
                }
            }

            let cb = gpu.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            for i in 0..upto {
                let lw = &self.layers[i];
                let x_in = &self.x[(2 * i) % 3];
                let x_mid = &self.x[(2 * i + 1) % 3];
                let x_out = &self.x[(2 * i + 2) % 3];
                let (cos, sin) = match lw.kind {
                    LayerKind::Sliding => (&self.rope_sliding.0, &self.rope_sliding.1),
                    LayerKind::Full => (&self.rope_full.0, &self.rope_full.1),
                };
                let hd = lw.head_dim;

                // --- attention sublayer (rows = m) ---
                gpu.encode_rms(
                    enc,
                    span(x_in),
                    Some(span(&lw.input_ln)),
                    span(&self.xn),
                    m,
                    hidden,
                    eps,
                )?;
                gpu.encode_linear(enc, &lw.q, &self.xn, &self.q_raw, m)?;
                gpu.encode_linear(enc, &lw.k, &self.xn, &self.k_raw, m)?;
                // Per-head norms flattened: row r of q_raw is head r % n_heads of
                // token r / n_heads, so m*n_heads rows of head_dim norm correctly.
                gpu.encode_rms(
                    enc,
                    span(&self.q_raw),
                    Some(span(&lw.q_norm)),
                    span(&self.q),
                    m * self.n_heads,
                    hd,
                    eps,
                )?;
                gpu.encode_rms(
                    enc,
                    span(&self.k_raw),
                    Some(span(&lw.k_norm)),
                    span(&self.k_norm),
                    m * lw.n_kv,
                    hd,
                    eps,
                )?;
                match &lw.v {
                    Some(lin) => gpu.encode_linear(enc, lin, &self.xn, &self.v_raw, m)?,
                    None => gpu.encode_rms(
                        enc,
                        span(&self.k_raw),
                        None,
                        span(&self.v_raw),
                        m * lw.n_kv,
                        hd,
                        eps,
                    )?,
                }
                // q ropes q → q_raw (distinct buffers: kr_rope reads two input
                // elements per output element); attention consumes q_raw.
                gpu.encode_rope(
                    enc,
                    span(&self.q),
                    span(cos),
                    span(sin),
                    span(&self.q_raw),
                    self.n_heads,
                    hd,
                    pos0,
                    m,
                )?;

                // Reserve this layer's rows for the whole chunk *before* its k/v
                // writes (rotation is host-side; buffers are shared-memory).
                let (kv_pos0, kv_rows) = cache.reserve_rows(&model.cfg, i, pos0, m)?;
                let new_row = kv_rows - m;
                gpu.encode_rope(
                    enc,
                    span(&self.k_norm),
                    span(cos),
                    span(sin),
                    cache
                        .row_span(i, false, new_row)
                        .ok_or_else(|| GemmaError::Config("kv k row span".into()))?,
                    lw.n_kv,
                    hd,
                    pos0,
                    m,
                )?;
                gpu.encode_copy(
                    enc,
                    span(&self.v_raw),
                    cache
                        .row_span(i, true, new_row)
                        .ok_or_else(|| GemmaError::Config("kv v row span".into()))?,
                    m * lw.n_kv * hd,
                )?;

                gpu.encode_attention(
                    enc,
                    span(&self.q_raw),
                    cache
                        .layer_keys_span(i)
                        .ok_or_else(|| GemmaError::Config("kv k span".into()))?,
                    cache
                        .layer_values_span(i)
                        .ok_or_else(|| GemmaError::Config("kv v span".into()))?,
                    span(&self.attn),
                    AttnParams {
                        s: m,
                        kv_rows,
                        q_pos0: pos0,
                        kv_pos0,
                        n_heads: self.n_heads,
                        n_kv: lw.n_kv,
                        head_dim: hd,
                        window: crate::gemma4::layer_window(lw.kind, model.cfg.sliding_window),
                    },
                )?;
                gpu.encode_linear(enc, &lw.o, &self.attn, &self.o_proj, m)?;
                gpu.encode_rms(
                    enc,
                    span(&self.o_proj),
                    Some(span(&lw.post_attn_ln)),
                    span(&self.o_norm),
                    m,
                    hidden,
                    eps,
                )?;
                gpu.encode_add(enc, span(x_in), span(&self.o_norm), span(x_mid), hidden * m)?;

                // --- MLP sublayer (rows = m) ---
                gpu.encode_rms(
                    enc,
                    span(x_mid),
                    Some(span(&lw.pre_ff_ln)),
                    span(&self.xn),
                    m,
                    hidden,
                    eps,
                )?;
                gpu.encode_linear(enc, &lw.gate, &self.xn, &self.gu_gate, m)?;
                gpu.encode_linear(enc, &lw.up, &self.xn, &self.gu_up, m)?;
                gpu.encode_gelu_mul(
                    enc,
                    span(&self.gu_gate),
                    span(&self.gu_up),
                    span(&self.gu),
                    self.intermediate * m,
                )?;
                gpu.encode_linear(enc, &lw.down, &self.gu, &self.down, m)?;
                gpu.encode_rms(
                    enc,
                    span(&self.down),
                    Some(span(&lw.post_ff_ln)),
                    span(&self.down_norm),
                    m,
                    hidden,
                    eps,
                )?;
                gpu.encode_add_scale(
                    enc,
                    span(x_mid),
                    span(&self.down_norm),
                    span(x_out),
                    hidden * m,
                    lw.scalar,
                )?;
            }

            // Final norm + lm_head only on the last chunk, on the last row.
            if is_last {
                let last_x = &self.x[(2 * upto) % 3];
                let last_row = DeviceSpan {
                    buffer: last_x,
                    byte_offset: (m - 1) * hidden * 2,
                };
                gpu.encode_rms(
                    enc,
                    last_row,
                    Some(span(&self.final_norm)),
                    span(&self.xn_final),
                    1,
                    hidden,
                    eps,
                )?;
                gpu.encode_linear(enc, &self.lm, &self.xn_final, &self.logits_h, 1)?;
            }
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();

            offset += m;
        }

        // Mirror the host path: reserve_rows advanced per-layer counts; advance
        // the shared token counter once for the whole span.
        cache.end_tokens(tokens.len());

        let logits: &[f16] = unsafe {
            std::slice::from_raw_parts(self.logits_h.contents() as *const f16, self.vocab)
        };
        let mut out: Vec<f32> = logits.iter().map(|v| v.to_f32()).collect();
        if let Some(cap) = model.cfg.softcap {
            crate::gemma4::apply_softcap(&mut out, cap);
        }
        Ok(out)
    }
}

/// Build `[cap, head_dim/2]` cos/sin tables with the *same* maths as the CPU
/// reference (`rope_cos_sin`), so proportional RoPE's identity lanes match.
fn build_rope_tables(
    device: &Device,
    head_dim: usize,
    rope: &crate::gemma4::RopeCfg,
    cap: usize,
) -> (Buffer, Buffer) {
    let half = head_dim / 2;
    let inv = crate::gemma4::inv_freq(head_dim, rope);
    let mut cos = vec![0f32; cap * half];
    let mut sin = vec![0f32; cap * half];
    for p in 0..cap {
        let (c, s) = crate::gemma4::rope_cos_sin(&inv, p);
        cos[p * half..(p + 1) * half].copy_from_slice(&c[..half]);
        sin[p * half..(p + 1) * half].copy_from_slice(&s[..half]);
    }
    (f32_up(device, &cos), f32_up(device, &sin))
}

#[allow(dead_code)]
fn unused(_: &ComputeCommandEncoderRef, _: Exl3Shape) {}
