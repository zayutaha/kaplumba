//! Gemma 4 text model forward (T1.3c): naive eager CPU path built on the EXL3
//! decode/reconstruct primitives.
//!
//! Correctness before speed — this is the golden reference; Metal GEMV/GEMM
//! kernels replace it in T2/T3 and are checked against it.
//!
//! Mirrors HF `Gemma4TextModel`:
//! `embed·√hidden` → 60 decoder layers (5 sliding : 1 full, `k_eq_v` on full,
//! Gemma norm placement = norm on the sublayer output then residual add) →
//! final norm → `lm_head` (streamed in 128-col slabs) → softcap → f32 logits.
//!
//! Semantics locked in `spec/rust-metal-engine/implementation_notes.md`
//! ("T1.3c handover"); do not re-derive:
//! - attention scale is **1.0** (no `1/√d`, no text softcap);
//! - sliding layers: mask `kv ∈ [q-1023, q]`; full layers: `kv ≤ q`;
//! - full layers with `k_eq_v`: `K = rope(k_norm(k_proj(x)))`,
//!   `V = v_norm(k_proj(x))` (weightless RMSNorm on **raw** k, no rope);
//! - RoPE is neox-style over head_dim, pairs `(j, j + head_dim/2)`;
//! - all positions are computed from position 0 (naive path recomputes the
//!   whole prefix each step; incremental KV lands in T4/T5).

use half::f16;

use crate::exl3::codebook::CodebookMode;
use crate::exl3::forward::{Exl3Linear, linear_forward_reconstruct, linear_forward_slab};
use crate::exl3::reconstruct::{TrellisView, trellis_u16_view};
use crate::kv_cache::KvCache;
use crate::load::{DType, Exl3Model, TextConfig};
use kaplumba_kernels::AttnParams;

/// Output columns per streamed `lm_head` slab (one right-Hadamard block).
pub const LM_SLAB: usize = 128;

#[derive(Debug, thiserror::Error)]
pub enum GemmaError {
    #[error("missing tensor {0}")]
    Missing(String),
    #[error("tensor {name}: {msg}")]
    Shape { name: String, msg: String },
    #[error("tensor {name}: unsupported dtype {got:?}")]
    Dtype { name: String, got: DType },
    #[error("tensor {name}: unknown codebook multiplier {mult:#010x}")]
    Codebook { name: String, mult: u32 },
    #[error("config: {0}")]
    Config(String),
    #[error(transparent)]
    MetalLinear(#[from] crate::metal_linear::MetalLinearError),
    #[error(transparent)]
    Trellis(#[from] crate::exl3::trellis::TrellisError),
    #[error(transparent)]
    Reconstruct(#[from] crate::exl3::reconstruct::ReconstructError),
}

// ---------------------------------------------------------------------------
// Tensor access
// ---------------------------------------------------------------------------

/// Borrowed tensor bytes + metadata (no copy, no ownership).
#[derive(Debug, Clone, Copy)]
pub struct RawTensor<'a> {
    pub bytes: &'a [u8],
    pub dtype: DType,
    pub shape: &'a [usize],
}

impl<'a> RawTensor<'a> {
    pub fn n_elems(&self) -> usize {
        self.shape.iter().product()
    }

    fn shape_err(&self, name: &str, msg: impl Into<String>) -> GemmaError {
        GemmaError::Shape {
            name: name.to_string(),
            msg: msg.into(),
        }
    }

    /// F32 values: BF16/F16/F32 supported (norms, norms weights, embed).
    pub fn to_f32(&self, name: &str) -> Result<Vec<f32>, GemmaError> {
        let n = self.n_elems();
        let want = match self.dtype {
            DType::F32 => n * 4,
            DType::BF16 | DType::F16 => n * 2,
            _ => {
                return Err(GemmaError::Dtype {
                    name: name.to_string(),
                    got: self.dtype,
                });
            }
        };
        if self.bytes.len() != want {
            return Err(self.shape_err(name, format!("{} bytes != {want}", self.bytes.len())));
        }
        Ok(match self.dtype {
            DType::F32 => self
                .bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            DType::BF16 => self
                .bytes
                .chunks_exact(2)
                .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            DType::F16 => self
                .bytes
                .chunks_exact(2)
                .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect(),
            _ => unreachable!(),
        })
    }

    /// First element as u32 (LE); used for the scalar `mcg` codebook tag.
    pub fn first_u32(&self, name: &str) -> Result<u32, GemmaError> {
        if self.bytes.len() < 4 {
            return Err(self.shape_err(name, "expected at least 4 bytes"));
        }
        Ok(u32::from_le_bytes([
            self.bytes[0],
            self.bytes[1],
            self.bytes[2],
            self.bytes[3],
        ]))
    }
}

/// Read-only named tensor access (implemented by [`Exl3Model`] and by test
/// fixtures). `None` for absent tensors (e.g. `v_proj` on `k_eq_v` layers).
pub trait TensorSource {
    fn get(&self, name: &str) -> Option<RawTensor<'_>>;

    /// Device buffer + byte offset of a tensor, when this source owns device
    /// storage (T2.1 GPU path). `None` (the default) keeps callers on the CPU
    /// reference path.
    fn device_span(&self, _name: &str) -> Option<(&metal::Buffer, usize)> {
        None
    }
}

impl TensorSource for Exl3Model {
    fn get(&self, name: &str) -> Option<RawTensor<'_>> {
        let (bytes, info) = self.tensor_bytes(name)?;
        Some(RawTensor {
            bytes,
            dtype: info.dtype,
            shape: &info.shape,
        })
    }

    fn device_span(&self, name: &str) -> Option<(&metal::Buffer, usize)> {
        self.tensor_device_span(name)
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    Sliding,
    Full,
}

#[derive(Debug, Clone, Copy)]
pub struct RopeCfg {
    pub theta: f64,
    /// Fraction of head_dim that rotates (1.0 = full rotation).
    pub partial_rotary_factor: f64,
    /// `rope_type == "proportional"` → inv_freq is padded with zeros.
    pub proportional: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct LayerCfg {
    pub kind: LayerKind,
    pub head_dim: usize,
    pub n_kv: usize,
    pub rope: RopeCfg,
}

#[derive(Debug, Clone)]
pub struct GemmaCfg {
    pub hidden: usize,
    pub intermediate: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub vocab: usize,
    pub eps: f64,
    pub sliding_window: usize,
    pub softcap: Option<f64>,
    pub k_eq_v: bool,
    pub layers: Vec<LayerCfg>,
}

impl GemmaCfg {
    /// Synthesize the per-layer scheme from `config.json` `text_config`.
    ///
    /// `layer_types` is used when present (it is, in the target artifact);
    /// otherwise the `(i+1) % 6 == 0 → full` pattern is synthesized, which is
    /// what the artifact ships (full at 5, 11, … 59).
    pub fn from_text_config(tc: &TextConfig) -> Result<Self, GemmaError> {
        let n_layers = tc.num_hidden_layers;
        if n_layers == 0 {
            return Err(GemmaError::Config("num_hidden_layers == 0".into()));
        }
        let types: Vec<String> = if tc.layer_types.is_empty() {
            (0..n_layers)
                .map(|i| {
                    if (i + 1) % 6 == 0 {
                        "full_attention".to_string()
                    } else {
                        "sliding_attention".to_string()
                    }
                })
                .collect()
        } else {
            if tc.layer_types.len() != n_layers {
                return Err(GemmaError::Config(format!(
                    "layer_types len {} != num_hidden_layers {n_layers}",
                    tc.layer_types.len()
                )));
            }
            tc.layer_types.clone()
        };

        let mut layers = Vec::with_capacity(n_layers);
        for (i, t) in types.iter().enumerate() {
            let kind = match t.as_str() {
                "full_attention" => LayerKind::Full,
                "sliding_attention" => LayerKind::Sliding,
                other => {
                    return Err(GemmaError::Config(format!(
                        "layer {i}: unknown layer type {other:?}"
                    )));
                }
            };
            let (head_dim, n_kv) = match kind {
                LayerKind::Full => (
                    tc.global_head_dim,
                    if tc.attention_k_eq_v {
                        tc.num_global_key_value_heads
                            .unwrap_or(tc.num_key_value_heads)
                    } else {
                        tc.num_key_value_heads
                    },
                ),
                LayerKind::Sliding => (tc.head_dim, tc.num_key_value_heads),
            };
            let rope_key = match kind {
                LayerKind::Full => "full_attention",
                LayerKind::Sliding => "sliding_attention",
            };
            let rope = match tc.rope_parameters.get(rope_key) {
                Some(p) => RopeCfg {
                    theta: p.rope_theta,
                    partial_rotary_factor: p.partial_rotary_factor.unwrap_or(1.0),
                    proportional: p.rope_type == "proportional",
                },
                None => RopeCfg {
                    theta: if kind == LayerKind::Full {
                        1.0e6
                    } else {
                        1.0e4
                    },
                    partial_rotary_factor: 1.0,
                    proportional: false,
                },
            };
            if head_dim == 0 || head_dim % 2 != 0 {
                return Err(GemmaError::Config(format!(
                    "layer {i}: head_dim {head_dim} must be even and non-zero"
                )));
            }
            if n_kv == 0 || tc.num_attention_heads % n_kv != 0 {
                return Err(GemmaError::Config(format!(
                    "layer {i}: n_kv {n_kv} must divide n_heads {}",
                    tc.num_attention_heads
                )));
            }
            // Every linear output width must be a multiple of 128 (right
            // Hadamard block) — checked here so failures are loud and early.
            for (what, width) in [
                ("q_proj out", tc.num_attention_heads * head_dim),
                ("k/v_proj out", n_kv * head_dim),
                ("o_proj out", tc.hidden_size),
                ("gate/up_proj out", tc.intermediate_size),
                ("down_proj out", tc.hidden_size),
            ] {
                if width % 128 != 0 {
                    return Err(GemmaError::Config(format!(
                        "layer {i}: {what} = {width} not a multiple of 128"
                    )));
                }
            }
            layers.push(LayerCfg {
                kind,
                head_dim,
                n_kv,
                rope,
            });
        }

        if tc.hidden_size % 128 != 0 {
            return Err(GemmaError::Config(format!(
                "hidden_size {} not a multiple of 128",
                tc.hidden_size
            )));
        }
        if tc.vocab_size % 128 != 0 {
            return Err(GemmaError::Config(format!(
                "vocab_size {} not a multiple of 128",
                tc.vocab_size
            )));
        }

        Ok(Self {
            hidden: tc.hidden_size,
            intermediate: tc.intermediate_size,
            n_heads: tc.num_attention_heads,
            n_layers,
            vocab: tc.vocab_size,
            eps: tc.rms_norm_eps,
            sliding_window: tc.sliding_window.unwrap_or(usize::MAX),
            softcap: tc.final_logit_softcapping,
            k_eq_v: tc.attention_k_eq_v,
            layers,
        })
    }

    pub fn n_full_layers(&self) -> usize {
        self.layers
            .iter()
            .filter(|l| l.kind == LayerKind::Full)
            .count()
    }
}

// ---------------------------------------------------------------------------
// Elementwise / math helpers (golden-locked; see notes)
// ---------------------------------------------------------------------------

#[inline]
pub fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[inline]
pub fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

/// Exact f16 add via f64 (no double-rounding through f32) — DE28.
#[inline]
pub fn add_f16(a: f16, b: f16) -> f16 {
    f16::from_f64(a.to_f64() + b.to_f64())
}

/// Exact f16 multiply via f64 — DE28.
#[inline]
pub fn mul_f16(a: f16, b: f64) -> f16 {
    f16::from_f64(a.to_f64() * b)
}

/// `gelu_pytorch_tanh`.
#[inline]
pub fn gelu_tanh(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_56;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044_715 * x * x * x)).tanh())
}

/// RMSNorm f32 over one row of f16 values; `weight == None` → weightless.
pub fn rms_norm_row(x: &[f16], weight: Option<&[f32]>, eps: f64, out: &mut [f16]) {
    assert_eq!(x.len(), out.len(), "rms_norm_row shapes");
    if let Some(w) = weight {
        assert_eq!(w.len(), x.len(), "rms_norm_row weight len");
    }
    let mut ss = 0f32;
    for v in x {
        ss += v.to_f32() * v.to_f32();
    }
    let ms = ss / x.len() as f32 + eps as f32;
    let inv = ms.powf(-0.5);
    for i in 0..x.len() {
        let mut v = x[i].to_f32() * inv;
        if let Some(w) = weight {
            v *= w[i];
        }
        out[i] = f16::from_f32(v);
    }
}

/// RMSNorm over `rows × dim` (row-major f16).
pub fn rms_norm_rows(
    x: &[f16],
    rows: usize,
    dim: usize,
    weight: Option<&[f32]>,
    eps: f64,
) -> Vec<f16> {
    assert_eq!(x.len(), rows * dim, "rms_norm_rows shape");
    let mut out = vec![f16::ZERO; x.len()];
    for r in 0..rows {
        rms_norm_row(
            &x[r * dim..(r + 1) * dim],
            weight,
            eps,
            &mut out[r * dim..(r + 1) * dim],
        );
    }
    out
}

/// RMSNorm inside each head (rows × n_heads heads of `head_dim`).
pub fn norm_heads(
    x: &[f16],
    rows: usize,
    n_heads: usize,
    head_dim: usize,
    weight: Option<&[f32]>,
    eps: f64,
) -> Vec<f16> {
    assert_eq!(x.len(), rows * n_heads * head_dim, "norm_heads shape");
    let mut out = vec![f16::ZERO; x.len()];
    for r in 0..rows {
        for h in 0..n_heads {
            let off = (r * n_heads + h) * head_dim;
            rms_norm_row(
                &x[off..off + head_dim],
                weight,
                eps,
                &mut out[off..off + head_dim],
            );
        }
    }
    out
}

/// RoPE inverse frequencies, length `head_dim/2`.
///
/// `proportional` rotates only `partial_rotary_factor` of the head, leaving the
/// exponent denominator at the **full** head_dim (matches HF proportional RoPE):
/// the remaining lanes get inv_freq 0 → identity rotation.
pub fn inv_freq(head_dim: usize, rope: &RopeCfg) -> Vec<f32> {
    let half = head_dim / 2;
    let n_nonzero = if rope.proportional {
        ((head_dim as f64) * rope.partial_rotary_factor / 2.0).round() as usize
    } else {
        half
    }
    .min(half);
    (0..half)
        .map(|k| {
            if k < n_nonzero {
                rope.theta.powf(-2.0 * k as f64 / head_dim as f64) as f32
            } else {
                0.0
            }
        })
        .collect()
}

/// `(cos, sin)` tables of length `head_dim` for one position (neox: duplicated
/// half).
pub fn rope_cos_sin(inv: &[f32], pos: usize) -> (Vec<f32>, Vec<f32>) {
    let half = inv.len();
    let mut cos = vec![0f32; 2 * half];
    let mut sin = vec![0f32; 2 * half];
    for k in 0..half {
        let f = pos as f32 * inv[k];
        let (c, s) = (f.cos(), f.sin());
        cos[k] = c;
        cos[k + half] = c;
        sin[k] = s;
        sin[k + half] = s;
    }
    (cos, sin)
}

/// Neox rotate: `out[j] = x[j]·cos - x[j+half]·sin`,
/// `out[j+half] = x[j]·sin + x[j+half]·cos`.
pub fn rope_apply_row(x: &[f16], cos: &[f32], sin: &[f32], out: &mut [f16]) {
    let hd = x.len();
    assert_eq!(out.len(), hd);
    assert_eq!(cos.len(), hd, "rope cos len");
    assert_eq!(sin.len(), hd, "rope sin len");
    let half = hd / 2;
    for j in 0..half {
        let x1 = x[j].to_f32();
        let x2 = x[j + half].to_f32();
        let c = cos[j];
        let s = sin[j];
        out[j] = f16::from_f32(x1 * c - x2 * s);
        out[j + half] = f16::from_f32(x1 * s + x2 * c);
    }
}

/// RoPE over all heads; row index is the position (naive path starts at 0).
pub fn rope_heads(
    x: &[f16],
    rows: usize,
    n_heads: usize,
    head_dim: usize,
    rope: &RopeCfg,
) -> Vec<f16> {
    rope_heads_at(x, rows, n_heads, head_dim, rope, 0)
}

/// RoPE over all heads starting at absolute position `pos0`.
///
/// The cached decode path needs the offset: a single row is at position `pos`,
/// not 0, and Gemma 4's tables are per-position (proportional RoPE on the full
/// layers makes this load-bearing rather than a phase shift).
pub fn rope_heads_at(
    x: &[f16],
    rows: usize,
    n_heads: usize,
    head_dim: usize,
    rope: &RopeCfg,
    pos0: usize,
) -> Vec<f16> {
    assert_eq!(x.len(), rows * n_heads * head_dim, "rope_heads shape");
    let inv = inv_freq(head_dim, rope);
    let mut out = vec![f16::ZERO; x.len()];
    for r in 0..rows {
        let (cos, sin) = rope_cos_sin(&inv, pos0 + r);
        for h in 0..n_heads {
            let off = (r * n_heads + h) * head_dim;
            rope_apply_row(
                &x[off..off + head_dim],
                &cos,
                &sin,
                &mut out[off..off + head_dim],
            );
        }
    }
    out
}

/// Lowest KV index visible to query position `q` on a layer of `kind`.
pub fn attention_lo(kind: LayerKind, q: usize, window: usize) -> usize {
    match kind {
        LayerKind::Full => 0,
        LayerKind::Sliding => q.saturating_sub(window.saturating_sub(1)),
    }
}

/// Effective attention window for a layer (`0` = full causal).
pub fn layer_window(kind: LayerKind, window: usize) -> usize {
    match kind {
        LayerKind::Full => 0,
        LayerKind::Sliding => window,
    }
}

/// `tanh(x/cap)·cap` in f32 (softcap is monotone → argmax unaffected).
pub fn apply_softcap(logits: &mut [f32], cap: f64) {
    let cap = cap as f32;
    for v in logits.iter_mut() {
        *v = (*v / cap).tanh() * cap;
    }
}

/// Argmax with lowest-index tie-break.
pub fn argmax(values: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

pub struct Gemma4Model<'a> {
    pub cfg: GemmaCfg,
    src: &'a dyn TensorSource,
    gpu: Option<&'a crate::metal_linear::MetalLinear>,
    /// Lazily built device decode pipeline (T4.2). Built on the first cached
    /// forward and reused; `decoder_unavailable` records a model the kernels
    /// cannot cover so the host path is used without retrying every token.
    decoder: std::cell::RefCell<Option<crate::gpu_decode::GpuDecoder>>,
    decoder_unavailable: std::cell::Cell<bool>,
}

impl<'a> Gemma4Model<'a> {
    pub fn new(cfg: GemmaCfg, src: &'a dyn TensorSource) -> Self {
        Self {
            cfg,
            src,
            gpu: None,
            decoder: std::cell::RefCell::new(None),
            decoder_unavailable: std::cell::Cell::new(false),
        }
    }

    /// A KV cache matching this model's path: device-backed when the GPU decode
    /// pipeline is available (so kernels can read K/V in place), host otherwise.
    pub fn new_kv_cache(&self, cap: usize) -> crate::kv_cache::KvCache {
        match self.gpu {
            Some(g) if !self.decoder_unavailable.get() => {
                crate::kv_cache::KvCache::new_device(&self.cfg, cap, g.device())
            }
            _ => crate::kv_cache::KvCache::new(&self.cfg, cap),
        }
    }

    /// Build the device decode pipeline on first use (`Ok(false)` = this model or
    /// runner cannot use it, so the host path applies). Rebuilt when a later
    /// cache is **larger** than the one the pipeline was built for — the RoPE
    /// tables (and batch arena) are sized from the first `cache_cap`, so keeping
    /// a small decoder would fail the table check on a longer prompt.
    fn ensure_decoder(
        &self,
        gpu: &crate::metal_linear::MetalLinear,
        cache_cap: usize,
    ) -> Result<bool, GemmaError> {
        if self.decoder_unavailable.get() {
            return Ok(false);
        }
        {
            let d = self.decoder.borrow();
            if let Some(dec) = d.as_ref() {
                if dec.built_cap >= cache_cap {
                    return Ok(true);
                }
            }
        }
        if self.decoder.borrow().is_some() {
            // Growing capacity: drop the too-small pipeline, rebuild below.
            *self.decoder.borrow_mut() = None;
        }
        match crate::gpu_decode::GpuDecoder::new(self, gpu, cache_cap)? {
            Some(d) => {
                *self.decoder.borrow_mut() = Some(d);
                Ok(true)
            }
            None => {
                self.decoder_unavailable.set(true);
                Ok(false)
            }
        }
    }

    /// Route eligible single-row linears through the Metal kernels (T2.1).
    /// Any ineligible tensor silently stays on the CPU reference path.
    pub fn with_gpu(mut self, gpu: &'a crate::metal_linear::MetalLinear) -> Self {
        self.gpu = Some(gpu);
        self
    }

    /// True when the model will attempt the GPU linear path.
    pub fn gpu_enabled(&self) -> bool {
        self.gpu.is_some()
    }

    /// Build from a loaded artifact (`config.json` drives the scheme).
    pub fn from_model(model: &'a Exl3Model) -> Result<Self, GemmaError> {
        let cfg = GemmaCfg::from_text_config(&model.config.text_config)?;
        Ok(Self::new(cfg, model))
    }

    fn tensor<'b>(&'b self, name: &str) -> Result<RawTensor<'b>, GemmaError> {
        self.src
            .get(name)
            .ok_or_else(|| GemmaError::Missing(name.to_string()))
    }

    pub fn f32_tensor(&self, name: &str) -> Result<Vec<f32>, GemmaError> {
        self.tensor(name)?.to_f32(name)
    }

    /// Build the EXL3 linear view for `<name>.{trellis,suh,svh,mcg,bias}`.
    fn exl3_linear(&self, name: &str) -> Result<Exl3Linear<'a>, GemmaError> {
        let src = self.src; // &'a dyn TensorSource
        let trellis_name = format!("{name}.trellis");
        let t = src
            .get(&trellis_name)
            .ok_or_else(|| GemmaError::Missing(trellis_name.clone()))?;
        if t.shape.len() != 3 {
            return Err(GemmaError::Shape {
                name: trellis_name,
                msg: format!("expected [in_tiles, out_tiles, packed], got {:?}", t.shape),
            });
        }
        let view = TrellisView::new(
            trellis_u16_view(t.bytes),
            t.shape[0],
            t.shape[1],
            t.shape[2],
        )?;

        let suh = src.get(&format!("{name}.suh"));
        let svh = src.get(&format!("{name}.svh"));
        let bias = src.get(&format!("{name}.bias"));
        for (what, t) in [("suh", suh), ("svh", svh)] {
            if let Some(t) = t {
                if t.dtype != DType::F16 && t.dtype != DType::I16 {
                    return Err(GemmaError::Dtype {
                        name: format!("{name}.{what}"),
                        got: t.dtype,
                    });
                }
            }
        }
        let mcg_name = format!("{name}.mcg");
        let mcg = src
            .get(&mcg_name)
            .ok_or_else(|| GemmaError::Missing(mcg_name.clone()))?;
        let mult = mcg.first_u32(&mcg_name)?;
        let mode = CodebookMode::from_mcg_mult(mult).ok_or(GemmaError::Codebook {
            name: mcg_name,
            mult,
        })?;

        Ok(Exl3Linear {
            trellis: view,
            suh: suh.map(|t| t.bytes),
            svh: svh.map(|t| t.bytes),
            suh_is_f16: suh.map(|t| t.dtype == DType::F16).unwrap_or(true),
            svh_is_f16: svh.map(|t| t.dtype == DType::F16).unwrap_or(true),
            mode,
            bias: bias.map(|t| t.bytes),
        })
    }

    /// The artifact source, for callers that need device handles (T4.2).
    pub fn source(&self) -> &'a dyn TensorSource {
        self.src
    }

    /// One embedding row as f16 **×√hidden**, without materializing the table.
    ///
    /// The decode pipeline needs this on the host (10 KB per token) because it is
    /// the one input that cannot be produced by a kernel from device weights
    /// without another pass.
    pub fn embed_row(&self, token: u32) -> Result<Vec<f16>, GemmaError> {
        let name = "model.language_model.embed_tokens.weight";
        let t = self.tensor(name)?;
        let hidden = self.cfg.hidden;
        let vocab = t.shape.first().copied().unwrap_or(0);
        if token as usize >= vocab {
            return Err(GemmaError::Config(format!(
                "token id {token} out of range (vocab {vocab})"
            )));
        }
        let scale = (hidden as f32).sqrt();
        let row = &t.bytes[token as usize * hidden * 2..(token as usize + 1) * hidden * 2];
        let mut out = vec![f16::ZERO; hidden];
        for (j, o) in out.iter_mut().enumerate() {
            let b = bf16_bits_to_f32(u16::from_le_bytes([row[2 * j], row[2 * j + 1]]));
            *o = f16::from_f32(b * scale);
        }
        Ok(out)
    }

    /// Explicit CPU reference linear (`linear_forward_reconstruct`).
    pub fn linear_via_cpu(&self, name: &str, x: &[f16]) -> Result<Vec<f16>, GemmaError> {
        let layer = self.exl3_linear(name)?;
        Ok(linear_forward_reconstruct(&layer, x))
    }

    /// Linear dispatch: GPU when eligible (decode-shaped, device-resident),
    /// CPU reference otherwise. Both paths are the *same math*; the GPU path is
    /// verified against this one in `metal_linear`'s tests.
    fn linear(&self, name: &str, x: &[f16]) -> Result<Vec<f16>, GemmaError> {
        if let Some(gpu) = self.gpu {
            if let Some(y) = gpu.try_linear(self.src, name, x)? {
                return Ok(y);
            }
        }
        self.linear_via_cpu(name, x)
    }

    pub fn layer_scalar(&self, i: usize) -> Result<f64, GemmaError> {
        let name = format!("model.language_model.layers.{i}.layer_scalar");
        let t = self.tensor(&name)?;
        let v = t.to_f32(&name)?;
        Ok(v.first().copied().unwrap_or(1.0) as f64)
    }

    /// `embed[token] · √hidden` (BF16 table → f16 activations).
    pub fn embed(&self, tokens: &[u32]) -> Result<Vec<f16>, GemmaError> {
        let name = "model.language_model.embed_tokens.weight";
        let t = self.tensor(name)?;
        if t.shape.len() != 2 || t.shape[1] != self.cfg.hidden {
            return Err(GemmaError::Shape {
                name: name.to_string(),
                msg: format!("expected [vocab, {}], got {:?}", self.cfg.hidden, t.shape),
            });
        }
        if t.dtype != DType::BF16 {
            return Err(GemmaError::Dtype {
                name: name.to_string(),
                got: t.dtype,
            });
        }
        let hidden = self.cfg.hidden;
        let vocab = t.shape[0];
        let scale = (hidden as f32).sqrt();
        let mut out = vec![f16::ZERO; tokens.len() * hidden];
        for (i, &tok) in tokens.iter().enumerate() {
            let tok = tok as usize;
            if tok >= vocab {
                return Err(GemmaError::Config(format!(
                    "token id {tok} out of range (vocab {vocab})"
                )));
            }
            let row = &t.bytes[tok * hidden * 2..(tok + 1) * hidden * 2];
            for j in 0..hidden {
                let b = bf16_bits_to_f32(u16::from_le_bytes([row[2 * j], row[2 * j + 1]]));
                out[i * hidden + j] = f16::from_f32(b * scale);
            }
        }
        Ok(out)
    }

    /// Causal (optionally windowed) GQA attention, scale 1.0.
    ///
    /// `q`: `s × n_heads·hd`, `k`/`v`: `s × n_kv·hd`. Output `s × n_heads·hd`.
    ///
    /// Dispatches to the Metal kernel when the GPU path is enabled and the shape
    /// is covered (T3.5); the CPU path here is the reference the kernel is
    /// checked against, and stays the fallback for unsupported shapes.
    pub fn attention(
        &self,
        lc: &LayerCfg,
        q: &[f16],
        k: &[f16],
        v: &[f16],
        s: usize,
    ) -> Result<Vec<f16>, GemmaError> {
        self.attention_bases(lc, q, k, v, s, 0, 0, s)
    }

    /// Attention with explicit position bases (T5.1/T3.5).
    ///
    /// Query row `r` is at absolute position `q_pos0 + r`; key row `j` is at
    /// `kv_pos0 + j`. A prefill uses `(q_pos0=0, kv_pos0=0, kv_rows=s)`; a cached
    /// decode step uses `s=1, q_pos0=pos` with however many key rows the cache
    /// still holds — for a sliding layer that window no longer starts at 0, which
    /// is the whole point of passing `kv_pos0` instead of assuming it.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_bases(
        &self,
        lc: &LayerCfg,
        q: &[f16],
        k: &[f16],
        v: &[f16],
        s: usize,
        q_pos0: usize,
        kv_pos0: usize,
        kv_rows: usize,
    ) -> Result<Vec<f16>, GemmaError> {
        if let Some(gpu) = self.gpu {
            if let Some(out) = gpu.try_attention(
                q,
                k,
                v,
                AttnParams {
                    s,
                    kv_rows,
                    q_pos0,
                    kv_pos0,
                    n_heads: self.cfg.n_heads,
                    n_kv: lc.n_kv,
                    head_dim: lc.head_dim,
                    window: layer_window(lc.kind, self.cfg.sliding_window),
                },
            )? {
                return Ok(out);
            }
        }
        Ok(self.attention_cpu(lc, q, k, v, s, q_pos0, kv_pos0, kv_rows))
    }

    /// CPU reference attention (the naive path and the kernel's ground truth).
    #[allow(clippy::too_many_arguments)]
    pub fn attention_cpu(
        &self,
        lc: &LayerCfg,
        q: &[f16],
        k: &[f16],
        v: &[f16],
        s: usize,
        q_pos0: usize,
        kv_pos0: usize,
        kv_rows: usize,
    ) -> Vec<f16> {
        let hd = lc.head_dim;
        let nh = self.cfg.n_heads;
        let nkv = lc.n_kv;
        assert!(nkv > 0 && nh % nkv == 0, "GQA head ratio");
        assert_eq!(q.len(), s * nh * hd, "q shape");
        assert_eq!(k.len(), kv_rows * nkv * hd, "k shape");
        assert_eq!(v.len(), kv_rows * nkv * hd, "v shape");
        assert!(kv_pos0 <= q_pos0, "keys cannot start after the queries");
        let gsize = nh / nkv;
        let window = layer_window(lc.kind, self.cfg.sliding_window);

        let mut out = vec![f16::ZERO; s * nh * hd];
        let mut scores: Vec<(usize, f32)> = Vec::new();
        for h in 0..nh {
            let kvh = h / gsize;
            for qi in 0..s {
                let pos = q_pos0 + qi;
                let lo_abs = if window == 0 {
                    kv_pos0
                } else {
                    kv_pos0.max((pos + 1).saturating_sub(window))
                };
                let hi_abs = pos.min(kv_pos0 + kv_rows - 1);
                let q_off = qi * nh * hd + h * hd;
                let qrow = &q[q_off..q_off + hd];
                if lo_abs > hi_abs {
                    continue; // nothing visible: stays zero, matching the kernel
                }
                scores.clear();
                let mut max_s = f32::NEG_INFINITY;
                for a in lo_abs..=hi_abs {
                    let kv = a - kv_pos0;
                    let kv_off = kv * nkv * hd + kvh * hd;
                    let krow = &k[kv_off..kv_off + hd];
                    let mut acc = 0f32;
                    for d in 0..hd {
                        acc += qrow[d].to_f32() * krow[d].to_f32();
                    }
                    scores.push((a, acc));
                    if acc > max_s {
                        max_s = acc;
                    }
                }
                let mut sum = 0f32;
                for (_, sc) in scores.iter_mut() {
                    let e = (*sc - max_s).exp();
                    *sc = e;
                    sum += e;
                }
                let inv = 1.0 / sum;
                for d in 0..hd {
                    let mut acc = 0f32;
                    for &(a, e) in scores.iter() {
                        let p = e * inv;
                        if p == 0.0 {
                            continue;
                        }
                        let kv = a - kv_pos0;
                        acc += p * v[kv * nkv * hd + kvh * hd + d].to_f32();
                    }
                    out[q_off + d] = f16::from_f32(acc);
                }
            }
        }
        out
    }

    /// One decoder layer, in place over `x` (`s × hidden`).
    pub fn layer(&self, i: usize, x: &mut [f16], s: usize) -> Result<(), GemmaError> {
        let cfg = &self.cfg;
        let hidden = cfg.hidden;
        let lc = cfg.layers[i];
        assert_eq!(x.len(), s * hidden, "layer input shape");
        let p = format!("model.language_model.layers.{i}");
        let hd = lc.head_dim;

        // --- self-attention ---
        let res: Vec<f16> = x.to_vec();
        let ln = self.f32_tensor(&format!("{p}.input_layernorm.weight"))?;
        let xn = rms_norm_rows(x, s, hidden, Some(&ln), cfg.eps);

        let q = self.linear(&format!("{p}.self_attn.q_proj"), &xn)?;
        let k_raw = self.linear(&format!("{p}.self_attn.k_proj"), &xn)?;
        let qnorm = self.f32_tensor(&format!("{p}.self_attn.q_norm.weight"))?;
        let knorm = self.f32_tensor(&format!("{p}.self_attn.k_norm.weight"))?;

        let q = rope_heads(
            &norm_heads(&q, s, cfg.n_heads, hd, Some(&qnorm), cfg.eps),
            s,
            cfg.n_heads,
            hd,
            &lc.rope,
        );
        let k = rope_heads(
            &norm_heads(&k_raw, s, lc.n_kv, hd, Some(&knorm), cfg.eps),
            s,
            lc.n_kv,
            hd,
            &lc.rope,
        );
        let v = if lc.kind == LayerKind::Full && cfg.k_eq_v {
            // k_eq_v: V is a weightless RMSNorm of the RAW k projection.
            norm_heads(&k_raw, s, lc.n_kv, hd, None, cfg.eps)
        } else {
            self.linear(&format!("{p}.self_attn.v_proj"), &xn)?
        };

        let attn = self.attention(&lc, &q, &k, &v, s)?;
        let o = self.linear(&format!("{p}.self_attn.o_proj"), &attn)?;
        let post_ln = self.f32_tensor(&format!("{p}.post_attention_layernorm.weight"))?;
        let o = rms_norm_rows(&o, s, hidden, Some(&post_ln), cfg.eps);
        for j in 0..s * hidden {
            x[j] = add_f16(res[j], o[j]);
        }

        // --- MLP ---
        let res: Vec<f16> = x.to_vec();
        let pre_ln = self.f32_tensor(&format!("{p}.pre_feedforward_layernorm.weight"))?;
        let xn = rms_norm_rows(x, s, hidden, Some(&pre_ln), cfg.eps);
        let g = self.linear(&format!("{p}.mlp.gate_proj"), &xn)?;
        let u = self.linear(&format!("{p}.mlp.up_proj"), &xn)?;
        let mut gu = vec![f16::ZERO; g.len()];
        for j in 0..g.len() {
            gu[j] = f16::from_f32(gelu_tanh(g[j].to_f32()) * u[j].to_f32());
        }
        let d = self.linear(&format!("{p}.mlp.down_proj"), &gu)?;
        let post_ff = self.f32_tensor(&format!("{p}.post_feedforward_layernorm.weight"))?;
        let d = rms_norm_rows(&d, s, hidden, Some(&post_ff), cfg.eps);
        let scalar = self.layer_scalar(i)?;
        for j in 0..s * hidden {
            x[j] = mul_f16(add_f16(res[j], d[j]), scalar);
        }

        Ok(())
    }

    /// One decoder layer for a **single token row**, reading/writing the KV cache
    /// (T5.1/T4.2). This is the decode shape the GPU kernels target: every
    /// projection here is `rows == 1`.
    pub fn layer_step(
        &self,
        i: usize,
        x: &mut [f16],
        cache: &mut KvCache,
        pos: usize,
    ) -> Result<(), GemmaError> {
        let cfg = &self.cfg;
        let hidden = cfg.hidden;
        let lc = cfg.layers[i];
        if x.len() != hidden {
            return Err(GemmaError::Config(format!(
                "layer_step needs one row of {hidden}, got {}",
                x.len()
            )));
        }
        let p = format!("model.language_model.layers.{i}");
        let hd = lc.head_dim;

        // --- self-attention ---
        let res: Vec<f16> = x.to_vec();
        let ln = self.f32_tensor(&format!("{p}.input_layernorm.weight"))?;
        let xn = rms_norm_rows(x, 1, hidden, Some(&ln), cfg.eps);

        let q = self.linear(&format!("{p}.self_attn.q_proj"), &xn)?;
        let k_raw = self.linear(&format!("{p}.self_attn.k_proj"), &xn)?;
        let qnorm = self.f32_tensor(&format!("{p}.self_attn.q_norm.weight"))?;
        let knorm = self.f32_tensor(&format!("{p}.self_attn.k_norm.weight"))?;

        let q = rope_heads_at(
            &norm_heads(&q, 1, cfg.n_heads, hd, Some(&qnorm), cfg.eps),
            1,
            cfg.n_heads,
            hd,
            &lc.rope,
            pos,
        );
        let k = rope_heads_at(
            &norm_heads(&k_raw, 1, lc.n_kv, hd, Some(&knorm), cfg.eps),
            1,
            lc.n_kv,
            hd,
            &lc.rope,
            pos,
        );
        let v = if lc.kind == LayerKind::Full && cfg.k_eq_v {
            norm_heads(&k_raw, 1, lc.n_kv, hd, None, cfg.eps)
        } else {
            self.linear(&format!("{p}.self_attn.v_proj"), &xn)?
        };

        let (kv_pos0, kv_rows) = cache.append_layer(cfg, i, &k, &v)?;
        let attn = self.attention_bases(
            &lc,
            &q,
            cache.layer_keys(i),
            cache.layer_values(i),
            1,
            pos,
            kv_pos0,
            kv_rows,
        )?;
        let o = self.linear(&format!("{p}.self_attn.o_proj"), &attn)?;
        let post_ln = self.f32_tensor(&format!("{p}.post_attention_layernorm.weight"))?;
        let o = rms_norm_rows(&o, 1, hidden, Some(&post_ln), cfg.eps);
        for j in 0..hidden {
            x[j] = add_f16(res[j], o[j]);
        }

        // --- MLP ---
        let res: Vec<f16> = x.to_vec();
        let pre_ln = self.f32_tensor(&format!("{p}.pre_feedforward_layernorm.weight"))?;
        let xn = rms_norm_rows(x, 1, hidden, Some(&pre_ln), cfg.eps);
        let g = self.linear(&format!("{p}.mlp.gate_proj"), &xn)?;
        let u = self.linear(&format!("{p}.mlp.up_proj"), &xn)?;
        let mut gu = vec![f16::ZERO; g.len()];
        for j in 0..g.len() {
            gu[j] = f16::from_f32(gelu_tanh(g[j].to_f32()) * u[j].to_f32());
        }
        let d = self.linear(&format!("{p}.mlp.down_proj"), &gu)?;
        let post_ff = self.f32_tensor(&format!("{p}.post_feedforward_layernorm.weight"))?;
        let d = rms_norm_rows(&d, 1, hidden, Some(&post_ff), cfg.eps);
        let scalar = self.layer_scalar(i)?;
        for j in 0..hidden {
            x[j] = mul_f16(add_f16(res[j], d[j]), scalar);
        }
        Ok(())
    }

    /// Push `tokens` through the model with a KV cache, returning the **last**
    /// token's logits.
    ///
    /// Each token is one pass of `rows == 1` projections plus attention over the
    /// cached keys, so the per-token cost stops growing with the prefix (the
    /// naive [`Self::forward_seq`] recomputes it). Prefill of `tokens.len() >= 2`
    /// goes through the batched GEMM span (T3.2); a single token uses the tuned
    /// per-token pipeline (T4.2).
    pub fn forward_cached(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
    ) -> Result<Vec<f32>, GemmaError> {
        self.forward_cached_layers(cache, tokens, self.cfg.n_layers)
    }

    /// [`Self::forward_cached`] through the first `upto` decoder layers (depth
    /// limit for debugging/benching).
    pub fn forward_cached_layers(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        upto: usize,
    ) -> Result<Vec<f32>, GemmaError> {
        if tokens.is_empty() {
            return Err(GemmaError::Config("empty token sequence".into()));
        }
        // T4.2: device-resident pipeline, one command buffer per token. It needs
        // a device-backed cache; anything unsupported falls back to the host path
        // below (same maths, slower submission).
        if let Some(gpu) = self.gpu {
            if cache.is_device_backed() && self.ensure_decoder(gpu, cache.capacity())? {
                let d = self.decoder.borrow();
                let d = d.as_ref().expect("ensure_decoder built it");
                let logits = if tokens.len() == 1 {
                    let cb = gpu.new_command_buffer();
                    let l = d.encode_token(self, gpu, &cb, cache, tokens[0], upto)?;
                    // Mirror the host path: `reserve_row` advanced the per-layer
                    // row counts, this advances the shared token counter so the
                    // next token sees the right position.
                    cache.end_token();
                    l
                } else {
                    d.encode_span(self, gpu, cache, tokens, upto)?
                };
                return Ok(logits);
            }
        }
        let hidden = self.cfg.hidden;
        let upto = upto.min(self.cfg.n_layers);
        let norm_w = self.f32_tensor("model.language_model.norm.weight")?;
        let mut logits = Vec::new();
        for &t in tokens {
            let pos = cache.len();
            let mut x = self.embed(&[t])?;
            for i in 0..upto {
                self.layer_step(i, &mut x, cache, pos)?;
            }
            cache.end_token();
            let xn = rms_norm_rows(&x, 1, hidden, Some(&norm_w), self.cfg.eps);
            logits = self.logits(&xn)?;
        }
        Ok(logits)
    }

    /// Full prompt forward → final hidden states (`s × hidden` f16).
    pub fn forward_seq(&self, tokens: &[u32]) -> Result<Vec<f16>, GemmaError> {
        self.forward_seq_layers(tokens, self.cfg.n_layers)
    }

    /// Forward through the first `upto` decoder layers (+ final norm).
    ///
    /// Used to calibrate/benchmark the naive CPU path at partial depth; the
    /// truncated output is not meaningful text.
    pub fn forward_seq_layers(&self, tokens: &[u32], upto: usize) -> Result<Vec<f16>, GemmaError> {
        if tokens.is_empty() {
            return Err(GemmaError::Config("empty token sequence".into()));
        }
        let upto = upto.min(self.cfg.n_layers);
        let s = tokens.len();
        let hidden = self.cfg.hidden;
        let mut x = self.embed(tokens)?;
        for i in 0..upto {
            self.layer(i, &mut x, s)?;
        }
        let norm_w = self.f32_tensor("model.language_model.norm.weight")?;
        Ok(rms_norm_rows(&x, s, hidden, Some(&norm_w), self.cfg.eps))
    }

    /// `lm_head` → softcapped f32 logits (`rows × vocab`), streamed in 128-col
    /// slabs so the 2.8 GB f16 head is never materialized.
    ///
    /// Softcap is applied in f32 (HF uses model dtype f16 — DE33).
    pub fn logits(&self, x: &[f16]) -> Result<Vec<f32>, GemmaError> {
        let hidden = self.cfg.hidden;
        let vocab = self.cfg.vocab;
        assert_eq!(x.len() % hidden, 0, "logits input shape");
        let rows = x.len() / hidden;
        let layer = self.exl3_linear("lm_head")?;
        if layer.in_features() != hidden || layer.out_features() != vocab {
            return Err(GemmaError::Shape {
                name: "lm_head".into(),
                msg: format!(
                    "weight is {}×{} but config expects {hidden}×{vocab}",
                    layer.in_features(),
                    layer.out_features()
                ),
            });
        }
        // GPU path: the whole head in one dispatch (decode shape only).
        if let Some(gpu) = self.gpu {
            if let Some(y) = gpu.try_linear(self.src, "lm_head", x)? {
                let mut out: Vec<f32> = y.iter().map(|v| v.to_f32()).collect();
                if let Some(cap) = self.cfg.softcap {
                    apply_softcap(&mut out, cap);
                }
                return Ok(out);
            }
        }

        let mut out = vec![0f32; rows * vocab];
        for n0 in (0..vocab).step_by(LM_SLAB) {
            let n = (vocab - n0).min(LM_SLAB);
            let y = linear_forward_slab(&layer, x, n0, n);
            for r in 0..rows {
                let dst = r * vocab + n0;
                for c in 0..n {
                    out[dst + c] = y[r * n + c].to_f32();
                }
                if let Some(cap) = self.cfg.softcap {
                    apply_softcap(&mut out[dst..dst + n], cap);
                }
            }
        }
        Ok(out)
    }

    /// Greedy (temp 0) decode. Recomputes the whole prefix each step — the naive
    /// path; [`crate::generate`] uses the KV cache when asked (T5.1). EOS is
    /// included in the output.
    pub fn greedy_generate(
        &self,
        prompt: &[u32],
        max_new: usize,
        eos: &[u32],
    ) -> Result<Vec<u32>, GemmaError> {
        let params = crate::generate::GenParams {
            max_new,
            sample: crate::sampler::SampleParams::greedy(),
            seed: 0,
            eos: eos.to_vec(),
            layers: None,
            use_kv: false,
        };
        Ok(crate::generate::generate(self, prompt, &params, |_| true)?.tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- KV cache path (T5.1) ----------------------------------------------

    /// The cached path must reproduce the naive whole-prefix forward.
    ///
    /// This is the equivalence that makes the cache trustworthy: the naive path
    /// computes attention over the prompt in one block, the cached path appends
    /// one row per token and reads it back, and for a sliding layer the older
    /// rows have been rotated out. Same visible keys, different accumulation
    /// order, so the check is "same argmax + tiny f32 noise", not bit-exact.
    #[test]
    fn cached_forward_matches_naive() {
        use crate::testutil::synth_model;
        let (src, cfg) = synth_model(Some(30.0));
        let hidden = cfg.hidden;
        let model = Gemma4Model::new(cfg.clone(), &src);
        // Longer than the window (4) so the sliding layers rotate.
        let prompt: Vec<u32> = vec![2, 17, 5, 300, 42, 7, 19, 3, 88];

        let h = model.forward_seq(&prompt).expect("naive");
        let last = &h[(prompt.len() - 1) * hidden..];
        let want = model.logits(last).expect("logits");

        let mut kv = KvCache::new(&cfg, 64);
        let got = model.forward_cached(&mut kv, &prompt).expect("cached");
        assert_eq!(kv.len(), prompt.len(), "one cached row per token");
        assert_eq!(want.len(), got.len());

        let worst = want
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("cached vs naive logits: max abs {worst:.4}");
        assert_eq!(
            argmax(&want),
            argmax(&got),
            "cached path must predict the same token"
        );
        assert!(worst < 0.05, "cached logits drifted by {worst}");
    }

    /// Prefill in chunks must equal prefill in one shot, bit for bit: a token's
    /// forward depends only on the keys already cached.
    #[test]
    fn cached_prefill_is_chunk_invariant() {
        use crate::testutil::synth_model;
        let (src, cfg) = synth_model(None);
        let model = Gemma4Model::new(cfg.clone(), &src);
        let all: Vec<u32> = (0..9).map(|i| 3 + i).collect();

        let mut one = KvCache::new(&cfg, 32);
        let full = model.forward_cached(&mut one, &all).expect("one shot");

        let mut two = KvCache::new(&cfg, 32);
        model.forward_cached(&mut two, &all[..4]).expect("head");
        let split = model.forward_cached(&mut two, &all[4..]).expect("tail");

        assert_eq!(one.len(), two.len());
        assert_eq!(full, split, "chunked prefill must be bit-identical");
    }

    /// A cached step at position `pos` must match the same token pushed through
    /// the naive path with a `pos`-token prefix.
    #[test]
    fn cached_decode_step_matches_naive_prefix() {
        use crate::testutil::synth_model;
        let (src, cfg) = synth_model(Some(30.0));
        let hidden = cfg.hidden;
        let model = Gemma4Model::new(cfg.clone(), &src);
        let prefix: Vec<u32> = vec![2, 11, 13, 17, 19];
        let next = 23u32;

        let h = model
            .forward_seq(&[prefix.clone(), vec![next]].concat())
            .expect("naive");
        let want = model.logits(&h[5 * hidden..6 * hidden]).expect("logits");

        let mut kv = KvCache::new(&cfg, 32);
        model.forward_cached(&mut kv, &prefix).expect("prefill");
        let got = model.forward_cached(&mut kv, &[next]).expect("step");
        assert_eq!(kv.len(), 6);
        assert_eq!(argmax(&want), argmax(&got));
    }

    /// Position-dependent RoPE: the same token at a different position must
    /// produce different keys (guards a forgotten `pos0` offset).
    #[test]
    fn cached_step_uses_absolute_position() {
        use crate::testutil::synth_model;
        let (src, cfg) = synth_model(None);
        assert!(cfg.sliding_window > 0);
        let model = Gemma4Model::new(cfg.clone(), &src);
        let mut a = KvCache::new(&cfg, 16);
        let mut b = KvCache::new(&cfg, 16);
        model.forward_cached(&mut a, &[2, 5, 7]).expect("a");
        model
            .forward_cached(&mut b, &[2, 5, 7, 11, 13, 17])
            .expect("b");
        // Keys for the *first* token are not reused across positions: the caches
        // differ in the number of rows, and the newest row encodes position 2 vs 5.
        let la = a.layer(0);
        let lb = b.layer(0);
        assert!(la.len() <= cfg.sliding_window && lb.len() <= cfg.sliding_window);
        assert_ne!(la.start(), lb.start());
    }

    // -- pure golden constants (from the locked notes) ----------------------

    fn sliding_rope() -> RopeCfg {
        RopeCfg {
            theta: 1.0e4,
            partial_rotary_factor: 1.0,
            proportional: false,
        }
    }

    fn full_rope() -> RopeCfg {
        RopeCfg {
            theta: 1.0e6,
            partial_rotary_factor: 0.25,
            proportional: true,
        }
    }

    fn f16_of(v: &[f32]) -> Vec<f16> {
        v.iter().map(|&x| f16::from_f32(x)).collect()
    }

    #[test]
    fn inv_freq_sliding_golden() {
        let inv = inv_freq(256, &sliding_rope());
        assert_eq!(inv.len(), 128);
        let golden: [f32; 12] = [
            1.0,
            0.930_572_03,
            0.865_964_35,
            0.805_842_16,
            0.749_894_2,
            0.697_830_56,
            0.649_381_64,
            0.604_296_39,
            0.562_341_33,
            0.523_299_1,
            0.486_967_53,
            0.453_158_38,
        ];
        for (k, want) in golden.iter().enumerate() {
            let got = inv[k];
            assert!(
                (got - want).abs() <= want.abs() * 1e-6,
                "inv_freq[{k}] got={got} want={want}"
            );
        }
        assert!(inv.iter().all(|v| *v > 0.0), "sliding inv_freq all nonzero");
        // Last lane = 1e4^(-254/256) ≈ 0.0102, monotonically decreasing.
        for w in inv.windows(2) {
            assert!(w[1] < w[0], "inv_freq must decrease");
        }
    }

    #[test]
    fn inv_freq_full_proportional_golden() {
        let inv = inv_freq(512, &full_rope());
        assert_eq!(inv.len(), 256, "head_dim/2");
        let golden: [f32; 16] = [
            1.0,
            0.947_463_51,
            0.897_687_14,
            0.850_525_8,
            0.805_842_16,
            0.763_506_05,
            0.723_394_16,
            0.685_389_58,
            0.649_381_64,
            0.615_265_43,
            0.582_941_53,
            0.552_315_83,
            0.523_299_1,
            0.495_806_81,
            0.469_758_87,
            0.445_079_42,
        ];
        for (k, want) in golden.iter().enumerate() {
            let got = inv[k];
            assert!(
                (got - want).abs() <= want.abs() * 1e-6,
                "inv_freq[{k}] got={got} want={want}"
            );
        }
        let nonzero = inv.iter().filter(|v| **v != 0.0).count();
        assert_eq!(nonzero, 64, "0.25 × 512 / 2 rotated lanes");
        assert!(inv[64..].iter().all(|v| *v == 0.0));
    }

    /// Locked rope golden: dim 8, theta 1e4, pos 3, x = [1,2,3,4,8,12,16,24].
    #[test]
    fn rope_golden_bits() {
        let rope = RopeCfg {
            theta: 1.0e4,
            partial_rotary_factor: 1.0,
            proportional: false,
        };
        let inv = inv_freq(8, &rope);
        assert_eq!(inv.len(), 4);
        let (cos, sin) = rope_cos_sin(&inv, 3);
        let want_cos = [-0.989_992, 0.955_337, 0.999_55, 0.999_996];
        let want_sin = [0.141_12, 0.295_52, 0.029_995_5, 0.003];
        for k in 0..4 {
            assert!((cos[k] - want_cos[k]).abs() < 1e-5, "cos[{k}]={}", cos[k]);
            assert!((sin[k] - want_sin[k]).abs() < 1e-5, "sin[{k}]={}", sin[k]);
            assert_eq!(cos[k], cos[k + 4], "duplicated half");
            assert_eq!(sin[k], sin[k + 4], "duplicated half");
        }

        let x = f16_of(&[1.0, 2.0, 3.0, 4.0, 8.0, 12.0, 16.0, 24.0]);
        let mut y = vec![f16::ZERO; 8];
        rope_apply_row(&x, &cos, &sin, &mut y);
        // First half is bit-exact vs the reference.
        let want_bits: [u16; 4] = [0xc03d, 0xbe8b, 0x410a, 0x43db];
        for k in 0..4 {
            assert_eq!(
                y[k].to_bits(),
                want_bits[k],
                "y[{k}] got={:#06x}",
                y[k].to_bits()
            );
        }
        // Second half = x1·sin + x2·cos (x2 rotated into x1).
        let want_second = [-7.778_816, 12.055_084, 16.082_787, 24.011_904];
        for k in 0..4 {
            let got = y[k + 4].to_f32();
            assert!(
                (got - want_second[k]).abs() <= want_second[k].abs() * 1e-3,
                "y[{}] got={got} want={}",
                k + 4,
                want_second[k]
            );
        }
    }

    #[test]
    fn rope_identity_when_inv_freq_zero() {
        // All-zero inv_freq (fully padded proportional rope) → identity.
        let inv = vec![0f32; 4];
        let (cos, sin) = rope_cos_sin(&inv, 7);
        let x = f16_of(&[1.0, -2.0, 3.5, 0.25, 8.0, 12.0, 16.0, 24.0]);
        let mut y = vec![f16::ZERO; 8];
        rope_apply_row(&x, &cos, &sin, &mut y);
        assert_eq!(y, x, "cos=1,sin=0 → identity");
    }

    /// Locked rms_norm golden: dim 8, eps 1e-6,
    /// x = [1,2,3,4,8,12,16,24], w = [1.5,-2,0.25,1,0.5,-1,2,-0.75].
    #[test]
    fn rms_norm_golden_bits() {
        let x = f16_of(&[1.0, 2.0, 3.0, 4.0, 8.0, 12.0, 16.0, 24.0]);
        let w = [1.5f32, -2.0, 0.25, 1.0, 0.5, -1.0, 2.0, -0.75];
        let mut out = vec![f16::ZERO; 8];
        rms_norm_row(&x, Some(&w), 1e-6, &mut out);

        // inv = (1070/8 + 1e-6)^-0.5 = 0.086467542...
        let mut ss = 0f32;
        for v in &x {
            ss += v.to_f32() * v.to_f32();
        }
        assert_eq!(ss, 1070.0);
        let inv = (ss / 8.0 + 1e-6f32).powf(-0.5);
        assert!((inv - 0.086_467_542).abs() < 1e-8, "inv={inv}");

        let want_bits: [u16; 8] = [
            0x3027, 0xb589, 0x2c27, 0x3589, 0x3589, 0xbc27, 0x4189, 0xbe3a,
        ];
        for k in 0..8 {
            assert_eq!(
                out[k].to_bits(),
                want_bits[k],
                "out[{k}] got={:#06x} want={:#06x}",
                out[k].to_bits(),
                want_bits[k]
            );
        }
        // Weightless path drops the `* w` (used for k_eq_v V and v_norm).
        let mut wl = vec![f16::ZERO; 8];
        rms_norm_row(&x, None, 1e-6, &mut wl);
        for k in 0..8 {
            let want = x[k].to_f32() * inv;
            assert!((wl[k].to_f32() - want).abs() <= want.abs() * 1e-3);
        }
    }

    #[test]
    fn gelu_tanh_golden() {
        let cases: [(f32, f32); 7] = [
            (-3.0, -0.003_637_392_1),
            (-1.0, -0.158_808_01),
            (-0.5, -0.154_286),
            (0.0, 0.0),
            (0.5, 0.345_714),
            (1.0, 0.841_192_01),
            (3.0, 2.996_362_7),
        ];
        for (x, want) in cases {
            let got = gelu_tanh(x);
            assert!(
                (got - want).abs() <= want.abs().max(1.0) * 1e-5,
                "gelu_tanh({x}) got={got} want={want}"
            );
        }
        // Finite over a wide sweep; gelu is NOT monotone on x < 0 (it dips to
        // ≈ -0.17) but is monotone for x >= 0 and saturates to x for large x.
        let mut x = -20.0f32;
        while x <= 20.0 {
            let v = gelu_tanh(x);
            assert!(v.is_finite(), "gelu_tanh({x}) not finite");
            if x <= 0.0 {
                assert!((-0.2..=0.0).contains(&v), "gelu_tanh({x}) = {v}");
            }
            x += 0.25;
        }
        let mut prev = f32::NEG_INFINITY;
        let mut x = 0.0f32;
        while x <= 20.0 {
            let v = gelu_tanh(x);
            assert!(v >= prev, "gelu not monotone for x >= 0 at {x}");
            prev = v;
            x += 0.25;
        }
        assert!((gelu_tanh(20.0) - 20.0).abs() < 1e-3, "large-x saturation");
    }

    #[test]
    fn attention_window_bounds() {
        assert_eq!(attention_lo(LayerKind::Full, 0, 1024), 0);
        assert_eq!(attention_lo(LayerKind::Full, 9999, 1024), 0);
        assert_eq!(attention_lo(LayerKind::Sliding, 0, 1024), 0);
        assert_eq!(attention_lo(LayerKind::Sliding, 1023, 1024), 0);
        assert_eq!(attention_lo(LayerKind::Sliding, 1024, 1024), 1);
        assert_eq!(attention_lo(LayerKind::Sliding, 5000, 1024), 3977);
        // Degenerate window of 1 → self only.
        assert_eq!(attention_lo(LayerKind::Sliding, 7, 1), 7);
    }

    #[test]
    fn bf16_and_f16_helpers() {
        assert_eq!(bf16_bits_to_f32(0x3f80), 1.0);
        assert_eq!(bf16_bits_to_f32(0x0000), 0.0);
        assert_eq!(bf16_bits_to_f32(0xbf80), -1.0);
        // 0x3ccc is BF16 (not F16!) → 0.024902344 (embed[2] lane 0).
        assert!((bf16_bits_to_f32(0x3ccc) - 0.024_902_344).abs() < 1e-9);
        assert_eq!(add_f16(f16::ONE, f16::ONE), f16::from_f32(2.0));
        // 1.0 + 2^-11 rounds to-nearest-even at f16 precision.
        assert_eq!(mul_f16(f16::ONE, 0.5), f16::from_f32(0.5));
    }

    #[test]
    fn softcap_and_argmax() {
        let mut l = [0.0f32, 60.0, -60.0, 30.0];
        apply_softcap(&mut l, 30.0);
        assert!(l.iter().all(|v| v.abs() <= 30.0));
        assert!((l[0] - 0.0).abs() < 1e-6);
        assert!((l[3] - 30.0 * (1.0f32).tanh()).abs() < 1e-5);
        // Softcap is monotone → same argmax.
        assert_eq!(argmax(&l), 1);
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), 1, "lowest index wins ties");
        assert_eq!(argmax(&[f32::NEG_INFINITY; 4]), 0);
    }

    // -- synthetic end-to-end (no artifact, no Metal) ----------------------
    //
    // Fixture lives in `crate::testutil` (shared with the `generate` tests).

    use crate::testutil::synth_model;

    #[test]
    fn synth_cfg_shapes() {
        let (_src, cfg) = synth_model(Some(30.0));
        assert_eq!(cfg.n_layers, 2);
        assert_eq!(cfg.layers[0].kind, LayerKind::Sliding);
        assert_eq!(cfg.layers[1].kind, LayerKind::Full);
        assert_eq!(cfg.layers[0].n_kv, 2);
        assert_eq!(cfg.layers[1].n_kv, 2);
        assert_eq!(cfg.layers[0].rope.proportional, false);
        assert_eq!(cfg.layers[1].rope.proportional, true);
        assert_eq!(cfg.n_full_layers(), 1);
    }

    #[test]
    fn synth_forward_logits_and_greedy() {
        let (src, cfg) = synth_model(Some(30.0));
        let model = Gemma4Model::new(cfg, &src);
        // 5 tokens > window 4 so the sliding mask actually drops a position.
        let prompt = [2u32, 17, 5, 300, 42];

        let h = model.forward_seq(&prompt).expect("forward");
        assert_eq!(h.len(), prompt.len() * model.cfg.hidden);
        assert!(h.iter().all(|v| v.to_f32().is_finite()), "finite hidden");

        // Deterministic across calls.
        let h2 = model.forward_seq(&prompt).expect("forward 2");
        assert_eq!(h, h2, "forward must be deterministic");

        let last = &h[(prompt.len() - 1) * model.cfg.hidden..];
        let logits = model.logits(last).expect("logits");
        assert_eq!(logits.len(), model.cfg.vocab);
        assert!(logits.iter().all(|v| v.is_finite()));
        assert!(
            logits.iter().all(|v| v.abs() <= 30.0),
            "softcap must bound logits"
        );

        let greedy = model.greedy_generate(&prompt, 3, &[1]).expect("greedy");
        assert!(!greedy.is_empty() && greedy.len() <= 3);
        assert!(greedy.iter().all(|t| (*t as usize) < model.cfg.vocab));
        let again = model.greedy_generate(&prompt, 3, &[1]).expect("greedy 2");
        assert_eq!(greedy, again, "greedy must be deterministic");

        // Passing the sampled token as EOS stops generation immediately.
        let eos_gen = model
            .greedy_generate(&prompt, 3, &greedy)
            .expect("greedy eos");
        assert_eq!(eos_gen.len(), 1, "first sampled token is EOS → stop");
        assert_eq!(eos_gen[0], greedy[0]);
    }

    #[test]
    fn synth_partial_depth_is_consistent() {
        let (src, cfg) = synth_model(Some(30.0));
        let model = Gemma4Model::new(cfg, &src);
        let prompt = [2u32, 17, 5];
        let full = model.forward_seq(&prompt).unwrap();
        // Depth >= n_layers clamps to the full forward.
        assert_eq!(model.forward_seq_layers(&prompt, 2).unwrap(), full);
        assert_eq!(model.forward_seq_layers(&prompt, 99).unwrap(), full);
        // Depth 0 = embed + final norm only (no decoder layers).
        let zero = model.forward_seq_layers(&prompt, 0).unwrap();
        assert_ne!(zero, full, "0 layers must differ from 2");
        let norm_w = model
            .f32_tensor("model.language_model.norm.weight")
            .unwrap();
        let emb = model.embed(&prompt).unwrap();
        let want = rms_norm_rows(&emb, prompt.len(), model.cfg.hidden, Some(&norm_w), 1e-6);
        assert_eq!(zero, want, "depth 0 = embed + final norm");
    }

    #[test]
    fn synth_forward_without_softcap() {
        let (src, cfg) = synth_model(None);
        assert!(cfg.softcap.is_none());
        let model = Gemma4Model::new(cfg, &src);
        let h = model.forward_seq(&[2, 3, 4]).expect("forward");
        let last = &h[2 * model.cfg.hidden..];
        let logits = model.logits(last).expect("logits");
        // Without softcap the raw head values are unclamped (still finite).
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn synth_k_eq_v_requires_no_v_proj_on_full_layer() {
        // Loaded model has no v_proj for layer 1 (full); if the forward tried to
        // read it this errors — so a successful forward proves the k_eq_v branch.
        let (src, cfg) = synth_model(Some(30.0));
        assert!(
            src.get("model.language_model.layers.1.self_attn.v_proj.trellis")
                .is_none()
        );
        assert!(
            src.get("model.language_model.layers.0.self_attn.v_proj.trellis")
                .is_some()
        );
        let model = Gemma4Model::new(cfg, &src);
        let mut x = model.embed(&[2, 9]).unwrap();
        model.layer(1, &mut x, 2).expect("full layer forward");
    }

    #[test]
    fn synth_missing_tensor_is_a_clear_error() {
        let (mut src, cfg) = synth_model(Some(30.0));
        src.map
            .remove("model.language_model.layers.0.self_attn.q_proj.mcg");
        let model = Gemma4Model::new(cfg, &src);
        let err = model.forward_seq(&[2]).unwrap_err();
        assert!(
            matches!(&err, GemmaError::Missing(n) if n.ends_with("q_proj.mcg")),
            "got {err}"
        );

        // Bad codebook tag is rejected, not silently defaulted.
        let (mut src2, cfg2) = synth_model(Some(30.0));
        src2.map
            .get_mut("model.language_model.layers.0.self_attn.q_proj.mcg")
            .unwrap()
            .bytes = 0xDEAD_BEEFu32.to_le_bytes().to_vec();
        let model2 = Gemma4Model::new(cfg2, &src2);
        let err2 = model2.forward_seq(&[2]).unwrap_err();
        assert!(matches!(err2, GemmaError::Codebook { .. }), "got {err2}");
    }

    #[test]
    fn cfg_rejects_unaligned_and_bad_layers() {
        let base = || {
            let (_s, cfg) = synth_model(Some(30.0));
            crate::testutil::synth_text_config(&cfg)
        };

        // Non-multiple-of-128 q_proj width.
        let mut tc = base();
        tc.num_attention_heads = 3; // 3 × 64 = 192
        let err = GemmaCfg::from_text_config(&tc).unwrap_err();
        assert!(matches!(err, GemmaError::Config(_)), "got {err}");

        // Unknown layer type.
        let mut tc = base();
        tc.layer_types = vec!["sliding_attention".into(), "hybrid".into()];
        assert!(matches!(
            GemmaCfg::from_text_config(&tc).unwrap_err(),
            GemmaError::Config(_)
        ));

        // layer_types length mismatch.
        let mut tc = base();
        tc.layer_types = vec!["sliding_attention".into()];
        assert!(matches!(
            GemmaCfg::from_text_config(&tc).unwrap_err(),
            GemmaError::Config(_)
        ));

        // Missing rope_parameters → defaults (theta 1e4 sliding / 1e6 full).
        let tc = base();
        let cfg = GemmaCfg::from_text_config(&tc).expect("defaults");
        assert_eq!(cfg.layers[0].rope.theta, 1.0e4);
        assert_eq!(cfg.layers[1].rope.theta, 1.0e6);
        assert_eq!(cfg.layers[1].rope.partial_rotary_factor, 1.0);

        // Empty layer_types → synthesized 6:1 pattern over 6 layers.
        let mut tc = base();
        tc.num_hidden_layers = 6;
        tc.layer_types = vec![];
        let cfg = GemmaCfg::from_text_config(&tc).unwrap();
        let kinds: Vec<LayerKind> = cfg.layers.iter().map(|l| l.kind).collect();
        assert_eq!(
            kinds,
            vec![
                LayerKind::Sliding,
                LayerKind::Sliding,
                LayerKind::Sliding,
                LayerKind::Sliding,
                LayerKind::Sliding,
                LayerKind::Full,
            ]
        );

        // Empty sequence is rejected.
        let (src, cfg) = synth_model(Some(30.0));
        let model = Gemma4Model::new(cfg, &src);
        assert!(model.forward_seq(&[]).is_err());
        // Out-of-vocab token is rejected.
        let (src, cfg) = synth_model(Some(30.0));
        let model = Gemma4Model::new(cfg, &src);
        assert!(model.forward_seq(&[9999]).is_err());
    }

    // -- real-artifact facts (cheap: header + a few KB, no Metal) -----------

    fn artifact_dir() -> Option<std::path::PathBuf> {
        let dir = match std::env::var("KAPLUMBA_MODEL_DIR") {
            Ok(d) => std::path::PathBuf::from(d),
            Err(_) => {
                let h = std::env::var("HOME").ok()?;
                std::path::PathBuf::from(format!("{h}/.kaplumba/models/Gxma/Gxma"))
            }
        };
        dir.join("config.json").is_file().then_some(dir)
    }

    /// Read one tensor's raw bytes + dtype + shape straight from the shards
    /// (avoids a 16 GB Metal residency just to check a few facts).
    fn read_tensor(dir: &std::path::Path, name: &str) -> Option<(Vec<u8>, DType, Vec<usize>)> {
        use std::io::{Read, Seek, SeekFrom};
        let index: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("model.safetensors.index.json")).ok()?,
        )
        .ok()?;
        let shard = index["weight_map"][name].as_str()?.to_string();
        let mut f = std::fs::File::open(dir.join(&shard)).ok()?;
        let mut len_buf = [0u8; 8];
        f.read_exact(&mut len_buf).ok()?;
        let hlen = u64::from_le_bytes(len_buf) as usize;
        let mut hbuf = vec![0u8; hlen];
        f.read_exact(&mut hbuf).ok()?;
        let header: serde_json::Value = serde_json::from_slice(&hbuf).ok()?;
        let t = header.get(name)?;
        let dtype = DType::from_safetensors(t["dtype"].as_str()?);
        let shape: Vec<usize> = t["shape"]
            .as_array()?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as usize)
            .collect();
        let off = t["data_offsets"][0].as_u64()?;
        let end = t["data_offsets"][1].as_u64()?;
        f.seek(SeekFrom::Start(8 + hlen as u64 + off)).ok()?;
        let mut bytes = vec![0u8; (end - off) as usize];
        f.read_exact(&mut bytes).ok()?;
        Some((bytes, dtype, shape))
    }

    #[test]
    fn real_artifact_config_matches_locked_facts() {
        let Some(dir) = artifact_dir() else {
            eprintln!("skip: no artifact (KAPLUMBA_MODEL_DIR)");
            return;
        };
        let mc: crate::load::ModelConfig =
            serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap())
                .unwrap();
        let tc = mc.text_config.clone();
        assert_eq!(mc.model_type, "gemma4");
        assert_eq!(tc.num_hidden_layers, 60);
        let cfg = GemmaCfg::from_text_config(&tc).expect("real cfg");
        assert_eq!(cfg.n_layers, 60);
        assert_eq!(cfg.hidden, 5376);
        assert_eq!(cfg.intermediate, 21504);
        assert_eq!(cfg.vocab, 262144);
        assert_eq!(cfg.n_heads, 32);
        assert_eq!(cfg.sliding_window, 1024);
        assert_eq!(cfg.softcap, Some(30.0));
        assert_eq!(cfg.n_full_layers(), 10);
        for (i, l) in cfg.layers.iter().enumerate() {
            let want_full = (i + 1) % 6 == 0;
            assert_eq!(
                l.kind == LayerKind::Full,
                want_full,
                "layer {i} kind mismatch"
            );
            if want_full {
                assert_eq!(l.head_dim, 512, "layer {i} full head_dim");
                assert_eq!(l.n_kv, 4, "layer {i} full n_kv (k_eq_v)");
                assert!(l.rope.proportional);
                assert_eq!(l.rope.theta, 1.0e6);
                assert_eq!(l.rope.partial_rotary_factor, 0.25);
                assert_eq!(
                    inv_freq(l.head_dim, &l.rope)
                        .iter()
                        .filter(|v| **v != 0.0)
                        .count(),
                    64
                );
            } else {
                assert_eq!(l.head_dim, 256, "layer {i} sliding head_dim");
                assert_eq!(l.n_kv, 16, "layer {i} sliding n_kv");
                assert!(!l.rope.proportional);
                assert_eq!(l.rope.theta, 1.0e4);
            }
        }
        // lm_head is a 6-bit head (head_bits) → packed 96 for k=6.
        let (_, dtype, shape) = read_tensor(&dir, "lm_head.trellis").expect("lm_head.trellis");
        assert_eq!(dtype, DType::I16);
        assert_eq!(shape, vec![336, 16384, 96]);
        let k = shape[2] * 16 / 256;
        assert_eq!(k, 6);
    }

    #[test]
    fn real_artifact_embed_bos_row_and_layer_scalars() {
        let Some(dir) = artifact_dir() else {
            eprintln!("skip: no artifact (KAPLUMBA_MODEL_DIR)");
            return;
        };
        let hidden = 5376usize;
        let (bytes, dtype, shape) =
            read_tensor(&dir, "model.language_model.embed_tokens.weight").expect("embed");
        assert_eq!(dtype, DType::BF16);
        assert_eq!(shape, vec![262144, hidden]);
        // Row 2 = BOS. Read only the first 16 lanes (skip the 2.8 GB table).
        let row_off = 2 * hidden * 2;
        let mut raw = Vec::new();
        let mut f16_bits = Vec::new();
        for j in 0..16 {
            let b = u16::from_le_bytes([bytes[row_off + 2 * j], bytes[row_off + 2 * j + 1]]);
            raw.push(b);
            f16_bits.push(f16::from_f32(bf16_bits_to_f32(b)).to_bits());
        }
        // Golden = the raw BF16 bit patterns stored on disk.
        let want_raw: [u16; 16] = [
            0x3ccc, 0x3c4a, 0x3cda, 0x3cb6, 0x3cd2, 0x3c3e, 0x3c6f, 0xbbfb, 0xbc58, 0xbb4f, 0x3bc5,
            0xbc1c, 0xbbbf, 0xbc42, 0x3b49, 0xbc51,
        ];
        assert_eq!(raw, want_raw.to_vec(), "embed[2] first 16 raw BF16 bits");
        // `bf16_bits << 16 as f32` → f16 (activations are f16).
        let want_f16: [u16; 16] = [
            0x2660, 0x2250, 0x26d0, 0x25b0, 0x2690, 0x21f0, 0x2378, 0x9fd8, 0xa2c0, 0x9a78, 0x1e28,
            0xa0e0, 0x9df8, 0xa210, 0x1a48, 0xa288,
        ];
        assert_eq!(f16_bits, want_f16.to_vec(), "embed[2] first 16 f16 bits");
        // Values are pre-scale embeddings (~0.025) — √5376 inside the model.
        assert!((bf16_bits_to_f32(raw[0]) - 0.024_902_344).abs() < 1e-9);

        for (i, want) in [(0usize, 0.089_355_469f32), (5, 0.878_906_25)] {
            let (b, dtype, shape) = read_tensor(
                &dir,
                &format!("model.language_model.layers.{i}.layer_scalar"),
            )
            .expect("layer_scalar");
            assert_eq!(dtype, DType::BF16);
            assert_eq!(shape, vec![1]);
            let got = bf16_bits_to_f32(u16::from_le_bytes([b[0], b[1]]));
            assert!(
                (got - want).abs() < 1e-7,
                "layer_scalar[{i}] got={got} want={want}"
            );
        }

        // Full-layer k_proj has no v_proj sibling and a 4-head k (k_eq_v).
        assert!(
            read_tensor(
                &dir,
                "model.language_model.layers.5.self_attn.v_proj.trellis"
            )
            .is_none()
        );
        let (_, _, shape) = read_tensor(
            &dir,
            "model.language_model.layers.5.self_attn.k_proj.trellis",
        )
        .unwrap();
        assert_eq!(shape, vec![336, 128, 48], "4 × 512 / 16 out tiles");
        let (_, _, shape) = read_tensor(
            &dir,
            "model.language_model.layers.0.self_attn.k_proj.trellis",
        )
        .unwrap();
        assert_eq!(shape, vec![336, 256, 48], "16 × 256 / 16 out tiles");
    }

    /// Cheap real-weight check: partial-depth forward is bit-exact across runs
    /// and finite, on both layer kinds (layer 0 sliding, layer 5 full/k_eq_v).
    /// ~10 row-layers of CPU — opt in with `KAPLUMBA_RUN_SLOW=1`.
    #[test]
    fn real_artifact_partial_depth_deterministic_slow() {
        if std::env::var("KAPLUMBA_RUN_SLOW").is_err() {
            eprintln!("skip: set KAPLUMBA_RUN_SLOW=1");
            return;
        }
        let Some(dir) = artifact_dir() else {
            eprintln!("skip: no artifact");
            return;
        };
        let ctx = crate::metal_ctx::MetalContext::new().expect("metal context");
        let model = crate::load::load_exl3(&ctx, &dir).expect("load artifact");
        let g = Gemma4Model::from_model(&model).expect("gemma cfg");
        let prompt = [2u32, 651];

        for depth in [1usize, 6] {
            let t0 = std::time::Instant::now();
            let a = g.forward_seq_layers(&prompt, depth).expect("forward a");
            let t1 = std::time::Instant::now();
            let b = g.forward_seq_layers(&prompt, depth).expect("forward b");
            assert_eq!(a, b, "depth {depth}: real-weight forward not deterministic");
            assert!(
                a.iter().all(|v| v.to_f32().is_finite()),
                "depth {depth}: non-finite hidden state"
            );
            eprintln!(
                "depth {depth}: 2×{} rows ok in {:.1}s + {:.1}s",
                prompt.len(),
                (t1 - t0).as_secs_f64(),
                t1.elapsed().as_secs_f64()
            );
        }
    }

    /// Real artifact, cached + GPU path, against the recorded fixture.
    ///
    /// The naive path's fixture is prompt `[2]` → argmax **236773** ("S"). The
    /// cached path must reproduce it while (a) stepping one row at a time and
    /// (b) actually dispatching to the GPU — `gpu_linears == 1` was exactly the
    /// silent-CPU-fallback bug, so the count is asserted, not just logged.
    #[test]
    fn real_artifact_cached_fixture_and_gpu_slow() {
        if std::env::var("KAPLUMBA_RUN_SLOW").is_err() {
            eprintln!("skip: set KAPLUMBA_RUN_SLOW=1");
            return;
        }
        let Some(dir) = artifact_dir() else {
            eprintln!("skip: no artifact");
            return;
        };
        let ctx = crate::metal_ctx::MetalContext::new().expect("metal context");
        let model = crate::load::load_exl3(&ctx, &dir).expect("load artifact");
        let gpu = crate::metal_linear::MetalLinear::new(&ctx).expect("metal linear");
        let g = Gemma4Model::from_model(&model)
            .expect("gemma cfg")
            .with_gpu(&gpu);

        // `new_kv_cache` is device-backed when the GPU is on, which is what
        // routes this into the T4.2 single-command-buffer decoder (a host cache
        // would silently fall back to the older per-linear submission path).
        let mut kv = g.new_kv_cache(8);
        let t0 = std::time::Instant::now();
        let logits = g.forward_cached(&mut kv, &[2]).expect("cached forward");
        let dt = t0.elapsed().as_secs_f64();
        let got = argmax(&logits);
        eprintln!(
            "cached 1-token prefill: argmax {got} logit {:.4} in {dt:.1}s, {} GPU linears, {:.0}ms GPU, device_kv={}",
            logits[got],
            gpu.calls(),
            gpu.gpu_ms(),
            kv.is_device_backed()
        );
        assert!(kv.is_device_backed(), "fixture must exercise the T4.2 path");
        assert_eq!(got, 236773, "recorded naive-path fixture for prompt [2]");
        assert_eq!(kv.len(), 1);
        assert!(kv.bytes() > 0);
        assert!(
            gpu.calls() >= 400,
            "cached path must run the GPU linears ({} dispatches)",
            gpu.calls()
        );
        // T4.2 packs every dispatch of one token into a single command buffer:
        // a fresh cache must replay to the exact same logits (catches missing
        // barriers or in-place races between dependent passes).
        let mut kv2 = g.new_kv_cache(8);
        let again = g.forward_cached(&mut kv2, &[2]).expect("repeat forward");
        assert_eq!(logits, again, "T4.2 single-cb forward not deterministic");
    }

    /// T3.2: batched prefill (`encode_span`) must agree with the token loop
    /// (`encode_token` × n) on the real artifact — 300 tokens with
    /// `cache_cap = 308` forces `batch_rows = 256` → two chunks (256 + 44), so
    /// the multi-chunk command-buffer boundary is covered too.
    ///
    /// Historically this asserted **bit-identical logits**, valid while the
    /// batched GEMM accumulated each row in exactly the GEMV order (scalar
    /// RGT path). The simdgroup tensor-core GEMM (v19c port, T3.x) multiplies
    /// in f16 8×8 tiles across k-blocks: same products (f16×f16 is exact in
    /// f32), different summation order → reassociation noise at the 1e-6…
    /// 1e-5 level per linear, of which the f16 rounding in `had_out` absorbs
    /// all but boundary cases. Over 60 layers that is expected to perturb
    /// some logits slightly while leaving the token identity intact — which
    /// is what the spec actually gates (R8.1 greedy token-identity, OQ1,
    /// fixture 236773). Measured with the simdgroup GEMM path on every
    /// linear (post loader-repack, T3.6): max_abs = 4.002571e-1 over 262144
    /// logits, n_diff = 258499, argmax identical (207 = 207); before the
    /// repack (half the layers still scalar-RGT) it was 2.3743033e-1 — the
    /// drift tracks how many layers use a non-GEMV summation order, and is
    /// now at its ceiling. The per-linear bound comes from
    /// `real_artifact_batched_linear_matches_gemv_slow` (worst ~1e-3 vs
    /// per-row GEMV on real trellises — kernel proven, this is compounding).
    /// Gate: argmax must match AND max drift < 0.55 (~1.37x the measured
    /// value). Systematic row-dependence / decode bugs are caught tighter and
    /// earlier by the per-linear test above and by the replay-determinism
    /// check below; the q-RoPE race this test originally caught
    /// (`n_diff=258588/262144`, `max_abs=2.95e-1`) also violated these.
    ///
    /// Also covers the decoder rebuild: a second model builds its pipeline on a
    /// cap-8 cache first, then prefills 32 tokens with a cap-64 cache (rope
    /// tables are sized at build; `ensure_decoder` must rebuild on growth).
    #[test]
    fn real_artifact_span_matches_token_loop_slow() {
        if std::env::var("KAPLUMBA_RUN_SLOW").is_err() {
            eprintln!("skip: set KAPLUMBA_RUN_SLOW=1");
            return;
        }
        let Some(dir) = artifact_dir() else {
            eprintln!("skip: no artifact");
            return;
        };
        let ctx = crate::metal_ctx::MetalContext::new().expect("metal context");
        let model = crate::load::load_exl3(&ctx, &dir).expect("load artifact");
        let gpu = crate::metal_linear::MetalLinear::new(&ctx).expect("metal linear");
        let g = Gemma4Model::from_model(&model)
            .expect("gemma cfg")
            .with_gpu(&gpu);

        // --- decoder rebuild on growing cache capacity ---
        let g2 = Gemma4Model::from_model(&model)
            .expect("gemma cfg")
            .with_gpu(&gpu);
        let mut kv_small = g2.new_kv_cache(8);
        g2.forward_cached(&mut kv_small, &[2])
            .expect("cap-8 forward");
        assert_eq!(kv_small.len(), 1);
        let grow: Vec<u32> = (0..32u32).map(|i| i + 2).collect();
        let mut kv_big = g2.new_kv_cache(64);
        let big_logits = g2
            .forward_cached(&mut kv_big, &grow)
            .expect("decoder must rebuild for a larger cache");
        assert_eq!(kv_big.len(), 32);
        assert!(big_logits.iter().all(|v| v.is_finite()));

        // --- span (2 chunks) vs token loop: bit-identical last-token logits ---
        let tokens: Vec<u32> = (0..300u32).map(|i| (i % 997) + 2).collect();
        let mut kv_span = g.new_kv_cache(308);
        let t0 = std::time::Instant::now();
        let span_logits = g.forward_cached(&mut kv_span, &tokens).expect("span");
        let span_dt = t0.elapsed().as_secs_f64();
        assert_eq!(kv_span.len(), 300);
        assert!(span_logits.iter().all(|v| v.is_finite()));

        let mut kv_loop = g.new_kv_cache(308);
        let t1 = std::time::Instant::now();
        let mut loop_logits: Vec<f32> = Vec::new();
        for &t in &tokens {
            loop_logits = g.forward_cached(&mut kv_loop, &[t]).expect("loop step");
        }
        let loop_dt = t1.elapsed().as_secs_f64();
        assert_eq!(kv_loop.len(), 300);
        eprintln!(
            "span 300 tok: {span_dt:.2}s ({:.1} t/s) vs token loop {loop_dt:.2}s ({:.1} t/s), {} GPU linears",
            300.0 / span_dt,
            300.0 / loop_dt,
            gpu.calls()
        );
        assert!(
            gpu.calls() >= 800,
            "span must drive the GPU linears ({} dispatches)",
            gpu.calls()
        );
        let n_diff = span_logits
            .iter()
            .zip(&loop_logits)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let max_abs = span_logits
            .iter()
            .zip(&loop_logits)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let argmax_span = span_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i);
        let argmax_loop = loop_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i);
        eprintln!(
            "span vs loop: {n_diff}/{} logits differ, max_abs={max_abs:e}, argmax span={argmax_span:?} loop={argmax_loop:?}",
            span_logits.len()
        );
        assert_eq!(
            argmax_span, argmax_loop,
            "span and token loop must predict the same token (n_diff={n_diff}, max_abs={max_abs:e})"
        );
        assert!(
            max_abs < 0.55,
            "span vs loop logits drifted by {max_abs:e} (n_diff={n_diff}) — beyond the \
             measured sg-path compounding (4.003e-1); kernel itself is gated at ~1e-3 \
             per linear by real_artifact_batched_linear_matches_gemv_slow"
        );

        // Fresh-cache replay determinism (chunked cbs, host KV rotations).
        let mut kv2 = g.new_kv_cache(308);
        let again = g.forward_cached(&mut kv2, &tokens).expect("replay");
        assert_eq!(span_logits, again, "span replay not deterministic");
    }

    /// T3.2 acceptance: pp/s before (token loop) and after (batched span).
    /// Before is measured at 512 (a token-loop 8k run would take ~an hour at
    /// the recorded 1.5 t/s — its rate also falls with context, so 512 is the
    /// honest small-prompt baseline); span runs at 512 / 2048 / 8192.
    /// Gate: `KAPLUMBA_RUN_SLOW=1`, `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn prefill_throughput_slow() {
        if std::env::var("KAPLUMBA_RUN_SLOW").is_err() {
            eprintln!("skip: set KAPLUMBA_RUN_SLOW=1");
            return;
        }
        let Some(dir) = artifact_dir() else {
            eprintln!("skip: no artifact");
            return;
        };
        let ctx = crate::metal_ctx::MetalContext::new().expect("metal context");
        let model = crate::load::load_exl3(&ctx, &dir).expect("load artifact");
        let gpu = crate::metal_linear::MetalLinear::new(&ctx).expect("metal linear");
        let g = Gemma4Model::from_model(&model)
            .expect("gemma cfg")
            .with_gpu(&gpu);

        let run = |n: usize, span: bool| -> f64 {
            let tokens: Vec<u32> = (0..n as u32).map(|i| (i % 997) + 2).collect();
            let mut kv = g.new_kv_cache(n + 8);
            let t0 = std::time::Instant::now();
            if span {
                g.forward_cached(&mut kv, &tokens).expect("span prefill");
            } else {
                for &t in &tokens {
                    g.forward_cached(&mut kv, &[t]).expect("token loop");
                }
            }
            let dt = t0.elapsed().as_secs_f64();
            assert_eq!(kv.len(), n, "token accounting");
            dt
        };

        // Warm-up: shader compile + first scratch growth (also sizes the
        // decoder's rope tables/arena at the largest cap we will use).
        run(32, true);
        run(4, false);

        let before = run(512, false);
        eprintln!(
            "BEFORE @512 token-loop: {before:.1}s → {:.2} t/s (recorded baseline ~1.5)",
            512.0 / before
        );
        for n in [512usize, 2048, 8192] {
            let dt = run(n, true);
            eprintln!(
                "AFTER  @{n:5} span: {dt:.1}s → {:.1} t/s ({}x vs token-loop rate)",
                n as f64 / dt,
                (n as f64 / dt) / (512.0 / before)
            );
        }
    }

    /// Top-`n` (logit, token) pairs, descending.
    fn top_logits(logits: &[f32], n: usize) -> Vec<(usize, f32)> {
        let mut v: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        v.truncate(n);
        v
    }

    /// Real end-to-end run on the artifact. Seconds per token-row on the naive
    /// CPU path — opt in with `KAPLUMBA_RUN_SLOW=1`.
    ///
    /// Env: `KAPLUMBA_SLOW_MODE` = `calibrate` (one layer, return) | `forward`
    /// (1 full forward + logits, default) | `greedy`; `KAPLUMBA_SLOW_PROMPT`
    /// (comma-separated ids, default `2` = BOS); `KAPLUMBA_SLOW_MAX_NEW` (greedy
    /// only, default 1); `KAPLUMBA_SLOW_REPEAT=1` (run the work twice and assert
    /// bit-identical output — the T1.3d determinism fixture).
    #[test]
    fn real_artifact_greedy_slow() {
        if std::env::var("KAPLUMBA_RUN_SLOW").is_err() {
            eprintln!("skip: set KAPLUMBA_RUN_SLOW=1 to run the real-model test");
            return;
        }
        let Some(dir) = artifact_dir() else {
            eprintln!("skip: no artifact");
            return;
        };
        let mode = std::env::var("KAPLUMBA_SLOW_MODE").unwrap_or_else(|_| "forward".into());
        let prompt: Vec<u32> = std::env::var("KAPLUMBA_SLOW_PROMPT")
            .unwrap_or_else(|_| "2".into())
            .split(',')
            .map(|s| s.trim().parse().expect("token id"))
            .collect();
        let max_new: usize = std::env::var("KAPLUMBA_SLOW_MAX_NEW")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        let repeat = std::env::var("KAPLUMBA_SLOW_REPEAT").is_ok();

        let ctx = crate::metal_ctx::MetalContext::new().expect("metal context");
        let model = crate::load::load_exl3(&ctx, &dir).expect("load artifact");
        let g = Gemma4Model::from_model(&model).expect("gemma cfg");
        let tk = crate::tokenizer::GemmaTokenizer::from_dir(&dir).ok();
        let show = |id: u32| -> String {
            tk.as_ref()
                .and_then(|t| t.decode(&[id]).ok())
                .unwrap_or_default()
        };

        // Depth calibration: time 1 layer first so a full run can be estimated.
        let t0 = std::time::Instant::now();
        g.forward_seq_layers(&prompt[..1], 1).expect("1 layer");
        let one = t0.elapsed().as_secs_f64();
        eprintln!(
            "1 token × 1 layer in {one:.1}s → ~{:.0}s per token at {} layers",
            one * 60.0,
            g.cfg.n_layers
        );

        match mode.as_str() {
            "calibrate" => {}
            "greedy" => {
                let t0 = std::time::Instant::now();
                let out = g.greedy_generate(&prompt, max_new, &[1]).expect("greedy");
                eprintln!(
                    "greedy prompt={prompt:?} max_new={max_new} → {out:?} in {:.1}s",
                    t0.elapsed().as_secs_f64()
                );
                eprintln!(
                    "greedy decoded: {:?}",
                    out.iter().map(|t| show(*t)).collect::<Vec<_>>()
                );
                assert!(!out.is_empty(), "greedy produced no tokens");
                assert!(out.len() <= max_new);
                if repeat {
                    let again = g.greedy_generate(&prompt, max_new, &[1]).expect("greedy 2");
                    assert_eq!(out, again, "greedy not deterministic on real weights");
                    eprintln!("determinism: OK (repeat greedy identical)");
                }
            }
            other => {
                assert_eq!(other, "forward", "unknown KAPLUMBA_SLOW_MODE");
                let run = || {
                    let t0 = std::time::Instant::now();
                    let h = g.forward_seq(&prompt).expect("forward");
                    let last = &h[(prompt.len() - 1) * g.cfg.hidden..];
                    let logits = g.logits(last).expect("logits");
                    (h, logits, t0.elapsed().as_secs_f64())
                };
                let (h, logits, dt) = run();
                assert!(
                    h.iter().all(|v| v.to_f32().is_finite()),
                    "non-finite hidden state"
                );
                let top = top_logits(&logits, 5);
                eprintln!(
                    "prompt={prompt:?} ({:?})",
                    prompt.iter().map(|t| show(*t)).collect::<Vec<_>>()
                );
                eprintln!("full forward + lm_head in {dt:.1}s");
                for (rank, (id, v)) in top.iter().enumerate() {
                    eprintln!("  #{rank} token {id} ({:?}) logit {v:.4}", show(*id as u32));
                }
                eprintln!("argmax = {} ({:?})", top[0].0, show(top[0].0 as u32));
                // Logits are softcapped → bounded; argmax is a valid token id.
                assert!(logits.iter().all(|v| v.is_finite()));
                assert!(logits.iter().all(|v| v.abs() <= 30.0 + 1e-4));
                assert!((top[0].0 as usize) < g.cfg.vocab);
                // Recorded fixture (T1.3d): prompt [2] → 236773 "S", ~29.98.
                if let Ok(exp) = std::env::var("KAPLUMBA_SLOW_EXPECT") {
                    let exp: usize = exp.parse().expect("KAPLUMBA_SLOW_EXPECT id");
                    assert_eq!(top[0].0, exp, "argmax drifted from the recorded fixture");
                }
                if repeat {
                    let (h2, logits2, _) = run();
                    assert_eq!(h, h2, "real-weight forward not deterministic");
                    assert_eq!(logits, logits2, "real-weight logits not deterministic");
                    eprintln!("determinism: OK (forward + logits bit-identical)");
                }
            }
        }
    }

    /// T4.0 decode baseline (rows=1, device KV, T4.2 pipeline): forward-only
    /// and end-to-end tg/s plus the implied weight-stream rate. `eos=[]` so
    /// the run always reaches `KAPLUMBA_DECODE_STEPS` (default 64).
    /// Ignored by default: `KAPLUMBA_RUN_SLOW=1 cargo test -p kaplumba-engine
    /// --release --lib -- --ignored --nocapture decode_throughput_slow`.
    #[test]
    #[ignore]
    fn decode_throughput_slow() {
        if std::env::var("KAPLUMBA_RUN_SLOW").is_err() {
            eprintln!("skip: set KAPLUMBA_RUN_SLOW=1 to run the real-model test");
            return;
        }
        let Some(dir) = crate::testutil::artifact_dir() else {
            eprintln!("skip: no artifact");
            return;
        };
        let steps: usize = std::env::var("KAPLUMBA_DECODE_STEPS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(64);
        let prompt: Vec<u32> = std::env::var("KAPLUMBA_SLOW_PROMPT")
            .unwrap_or_else(|_| "2".into())
            .split(',')
            .map(|s| s.trim().parse().expect("token id"))
            .collect();

        let ctx = crate::metal_ctx::MetalContext::new().expect("metal context");
        let model = crate::load::load_exl3(&ctx, &dir).expect("load artifact");
        let gpu = crate::metal_linear::MetalLinear::new(&ctx).expect("metal linear");
        let g = Gemma4Model::from_model(&model)
            .expect("gemma cfg")
            .with_gpu(&gpu);
        assert!(g.gpu_enabled(), "decode baseline must run the GPU path");
        let tk = crate::tokenizer::GemmaTokenizer::from_dir(&dir).ok();
        let show = |ids: &[u32]| -> String {
            tk.as_ref()
                .and_then(|t| t.decode(ids).ok())
                .unwrap_or_default()
        };

        let params = crate::generate::GenParams {
            max_new: steps,
            sample: crate::sampler::SampleParams::greedy(),
            seed: 0,
            eos: vec![],
            layers: None,
            use_kv: true,
        };
        let t0 = std::time::Instant::now();
        let stats = crate::generate::generate(&g, &prompt, &params, |_| true).expect("decode run");
        let wall = t0.elapsed().as_secs_f64();

        assert!(stats.kv_cache, "device KV path required");
        assert!(stats.cache_bytes > 0, "cache actually resident");
        assert_eq!(stats.gen_tokens, steps, "eos=[] → full length");
        assert!(stats.decode_steps > 0, "need decode steps");
        assert!(
            stats.tokens.iter().all(|t| (*t as usize) < g.cfg.vocab),
            "sampled id outside vocab"
        );

        let fwd_ms = stats.decode_s / stats.decode_steps as f64 * 1e3;
        let host_s = (wall - stats.prefill_s - stats.decode_s).max(0.0);
        let host_ms = host_s / stats.decode_steps as f64 * 1e3;
        let e2e = stats.decode_steps as f64 / (stats.decode_s + host_s);
        // Resident weights per full forward: artifact shards measured 16.028 GB
        // (trellis + had/svh tables + embed + lm_head); every token streams them.
        let bytes_per_tok = 16.028e9;
        let gbs = bytes_per_tok / (fwd_ms / 1e3) / 1e9;

        eprintln!(
            "[T4.0] prefill {:.1} ms ({:.1} t/s) | prompt {:?}",
            stats.prefill_s * 1e3,
            stats.pp_tps(),
            show(&prompt)
        );
        eprintln!(
            "[T4.0] decode forward-only {fwd_ms:.1} ms/token → {:.2} t/s",
            stats.tg_tps()
        );
        eprintln!("[T4.0] host between tokens {host_ms:.2} ms/token (sample+convert+submit)");
        eprintln!("[T4.0] end-to-end decode {e2e:.2} t/s");
        eprintln!(
            "[T4.0] implied weight stream {gbs:.1} GB/s ({:.3} GB/token)",
            bytes_per_tok / 1e9
        );
        eprintln!(
            "[T4.0] decoded[0..8] {:?} | text {:?}",
            &stats.tokens[..stats.tokens.len().min(8)],
            show(&stats.tokens[..stats.tokens.len().min(32)])
        );
        assert!(
            stats.tg_tps() > 0.5,
            "decode below 0.5 t/s — pipeline broken ({:.2})",
            stats.tg_tps()
        );
    }

    /// T2.7 decode phase profile via layer-scaling: forward-only ms/token at
    /// `layers = 60 / 10 / 0`. `fwd(n) = a + b·n` where `a` = embed + final
    /// norm + lm_head + device waits (non-layer) and `b` = average per-layer
    /// cost (linears + attn + rope/rms glue). `a − lm_head` isolates the
    /// fixed overhead; comparing `60b` against the `bench_passes` linear sums
    /// shows how much of a layer is NOT trellis gemv.
    /// `KAPLUMBA_RUN_SLOW=1 cargo test -p kaplumba-engine --release --lib --
    /// --ignored --nocapture decode_layer_scaling_slow`.
    #[test]
    #[ignore]
    fn decode_layer_scaling_slow() {
        if std::env::var("KAPLUMBA_RUN_SLOW").is_err() {
            eprintln!("skip: set KAPLUMBA_RUN_SLOW=1 to run the real-model test");
            return;
        }
        let Some(dir) = crate::testutil::artifact_dir() else {
            eprintln!("skip: no artifact");
            return;
        };
        let steps: usize = std::env::var("KAPLUMBA_DECODE_STEPS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(64);
        let ctx = crate::metal_ctx::MetalContext::new().expect("metal context");
        let model = crate::load::load_exl3(&ctx, &dir).expect("load artifact");
        let gpu = crate::metal_linear::MetalLinear::new(&ctx).expect("metal linear");
        let g = Gemma4Model::from_model(&model)
            .expect("gemma cfg")
            .with_gpu(&gpu);
        assert!(g.gpu_enabled(), "layer scaling must run the GPU path");

        let mut rows: Vec<(String, Option<usize>, f64, f64)> = Vec::new();
        for (label, layers) in [
            ("full", None),
            ("L=10", Some(10usize)),
            ("L=0", Some(0usize)),
        ] {
            let params = crate::generate::GenParams {
                max_new: steps,
                sample: crate::sampler::SampleParams::greedy(),
                seed: 0,
                eos: vec![],
                layers,
                use_kv: true,
            };
            let stats =
                crate::generate::generate(&g, &[2u32], &params, |_| true).expect("decode run");
            assert_eq!(stats.gen_tokens, steps, "eos=[] → full length");
            let fwd_ms = stats.decode_s / stats.decode_steps as f64 * 1e3;
            eprintln!(
                "[T2.7] {label:<6} fwd {fwd_ms:>7.2} ms/token → {:>5.2} t/s",
                stats.tg_tps()
            );
            rows.push((
                label.to_string(),
                layers,
                fwd_ms,
                stats.decode_s / stats.decode_steps as f64 * 1e3,
            ));
            assert!(fwd_ms > 0.0 && fwd_ms < 10_000.0, "implausible timing");
        }
        let (t60, t10, t0) = (rows[0].2, rows[1].2, rows[2].2);
        let b = (t60 - t10) / 50.0;
        let a = t10 - 10.0 * b;
        eprintln!(
            "[T2.7] model fwd(n) = {a:.2} + {b:.3}·n ms | \
             a includes lm_head (~5.9 ms bench) + embed + waits | \
             60 layers = {:.1} ms | non-layer a = {:.1} ms ({:.0}%)",
            60.0 * b,
            a,
            100.0 * a / t60
        );
        assert!(
            b > 0.0,
            "per-layer cost must be positive (t60 {t60} vs t10 {t10})"
        );
        assert!(a >= 0.0, "non-layer intercept negative? t0 {t0} t10 {t10}");
    }
}
