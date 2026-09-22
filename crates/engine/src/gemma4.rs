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
use crate::load::{DType, Exl3Model, TextConfig};

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
    assert_eq!(x.len(), rows * n_heads * head_dim, "rope_heads shape");
    let inv = inv_freq(head_dim, rope);
    let mut out = vec![f16::ZERO; x.len()];
    for r in 0..rows {
        let (cos, sin) = rope_cos_sin(&inv, r);
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
}

impl<'a> Gemma4Model<'a> {
    pub fn new(cfg: GemmaCfg, src: &'a dyn TensorSource) -> Self {
        Self { cfg, src }
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

    fn f32_tensor(&self, name: &str) -> Result<Vec<f32>, GemmaError> {
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

    fn linear(&self, name: &str, x: &[f16]) -> Result<Vec<f16>, GemmaError> {
        let layer = self.exl3_linear(name)?;
        Ok(linear_forward_reconstruct(&layer, x))
    }

    fn layer_scalar(&self, i: usize) -> Result<f64, GemmaError> {
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
    pub fn attention(&self, lc: &LayerCfg, q: &[f16], k: &[f16], v: &[f16], s: usize) -> Vec<f16> {
        let hd = lc.head_dim;
        let nh = self.cfg.n_heads;
        let nkv = lc.n_kv;
        assert!(nkv > 0 && nh % nkv == 0, "GQA head ratio");
        assert_eq!(q.len(), s * nh * hd, "q shape");
        assert_eq!(k.len(), s * nkv * hd, "k shape");
        assert_eq!(v.len(), s * nkv * hd, "v shape");
        let gsize = nh / nkv;
        let window = self.cfg.sliding_window;

        let mut out = vec![f16::ZERO; s * nh * hd];
        let mut scores = vec![0f32; s];
        for h in 0..nh {
            let kvh = h / gsize;
            for qi in 0..s {
                let lo = attention_lo(lc.kind, qi, window);
                let q_off = qi * nh * hd + h * hd;
                let qrow = &q[q_off..q_off + hd];
                let mut max_s = f32::NEG_INFINITY;
                for kv in lo..=qi {
                    let kv_off = kv * nkv * hd + kvh * hd;
                    let krow = &k[kv_off..kv_off + hd];
                    let mut acc = 0f32;
                    for d in 0..hd {
                        acc += qrow[d].to_f32() * krow[d].to_f32();
                    }
                    scores[kv] = acc;
                    if acc > max_s {
                        max_s = acc;
                    }
                }
                let mut sum = 0f32;
                for sc in scores[lo..=qi].iter_mut() {
                    let e = (*sc - max_s).exp();
                    *sc = e;
                    sum += e;
                }
                let inv = 1.0 / sum;
                for d in 0..hd {
                    let mut acc = 0f32;
                    for kv in lo..=qi {
                        let p = scores[kv] * inv;
                        if p == 0.0 {
                            continue;
                        }
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

        let attn = self.attention(&lc, &q, &k, &v, s);
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
    /// path; KV-cached decode lands in T4/T5. EOS is included in the output.
    pub fn greedy_generate(
        &self,
        prompt: &[u32],
        max_new: usize,
        eos: &[u32],
    ) -> Result<Vec<u32>, GemmaError> {
        let hidden = self.cfg.hidden;
        let mut toks: Vec<u32> = prompt.to_vec();
        let mut generated = Vec::with_capacity(max_new);
        for _ in 0..max_new {
            let h = self.forward_seq(&toks)?;
            let last = &h[(toks.len() - 1) * hidden..toks.len() * hidden];
            let logits = self.logits(last)?;
            let next = argmax(&logits) as u32;
            generated.push(next);
            if eos.contains(&next) {
                break;
            }
            toks.push(next);
        }
        Ok(generated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

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

    struct SynthTensor {
        bytes: Vec<u8>,
        dtype: DType,
        shape: Vec<usize>,
    }

    #[derive(Default)]
    struct SynthSource {
        map: HashMap<String, SynthTensor>,
    }

    impl SynthSource {
        fn insert(&mut self, name: &str, bytes: Vec<u8>, dtype: DType, shape: Vec<usize>) {
            self.map.insert(
                name.to_string(),
                SynthTensor {
                    bytes,
                    dtype,
                    shape,
                },
            );
        }

        fn insert_f16(&mut self, name: &str, values: &[f16]) {
            let mut b = Vec::with_capacity(values.len() * 2);
            for v in values {
                b.extend_from_slice(&v.to_bits().to_le_bytes());
            }
            self.insert(name, b, DType::F16, vec![values.len()]);
        }

        fn insert_bf16(&mut self, name: &str, values: &[f32], shape: Vec<usize>) {
            let mut b = Vec::with_capacity(values.len() * 2);
            for v in values {
                b.extend_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
            }
            self.insert(name, b, DType::BF16, shape);
        }

        /// Deterministic synthetic EXL3 linear (`mcg`, k-bit trellis, ±1 svh).
        fn insert_linear(&mut self, name: &str, inn: usize, out: usize, k: u32, seed: u32) {
            use crate::exl3::trellis::pack_trellis_tile;
            assert_eq!(inn % 16, 0);
            assert_eq!(out % 16, 0);
            let in_tiles = inn / 16;
            let out_tiles = out / 16;
            let packed = 256 * k as usize / 16;
            let mask = ((1u32 << k) - 1) as u16;
            let mut state = seed;
            let mut data = Vec::with_capacity(in_tiles * out_tiles * packed * 2);
            for _ in 0..in_tiles * out_tiles {
                let mut fresh = [0u16; 256];
                for f in fresh.iter_mut() {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    *f = (state >> 16) as u16 & mask;
                }
                for w in pack_trellis_tile(&fresh, k).unwrap() {
                    data.extend_from_slice(&w.to_le_bytes());
                }
            }
            self.insert(
                &format!("{name}.trellis"),
                data,
                DType::I16,
                vec![in_tiles, out_tiles, packed],
            );
            let ones = vec![f16::ONE; inn];
            self.insert_f16(&format!("{name}.suh"), &ones);
            let mut svh = Vec::with_capacity(out);
            for j in 0..out {
                svh.push(if (j / 16) % 2 == 0 {
                    f16::ONE
                } else {
                    f16::NEG_ONE
                });
            }
            self.insert_f16(&format!("{name}.svh"), &svh);
            self.insert(
                &format!("{name}.mcg"),
                CodebookMode::Mcg.mcg_mult().to_le_bytes().to_vec(),
                DType::I32,
                vec![],
            );
        }
    }

    impl TensorSource for SynthSource {
        fn get(&self, name: &str) -> Option<RawTensor<'_>> {
            self.map.get(name).map(|t| RawTensor {
                bytes: &t.bytes,
                dtype: t.dtype,
                shape: &t.shape,
            })
        }
    }

    /// Tiny 2-layer model: hidden 128, intermediate 256, vocab 384, 2 heads of
    /// head_dim 64, 2 kv heads, window 4, softcap 30, k_eq_v (full layer has no
    /// v_proj). Every width is a multiple of 128 so the real kernel path runs.
    fn synth_model(softcap: Option<f64>) -> (SynthSource, GemmaCfg) {
        let tc = TextConfig {
            hidden_size: 128,
            intermediate_size: 256,
            head_dim: 64,
            global_head_dim: 64,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            num_global_key_value_heads: Some(2),
            num_hidden_layers: 2,
            vocab_size: 384,
            rms_norm_eps: 1e-6,
            sliding_window: Some(4),
            hidden_activation: Some("gelu_pytorch_tanh".into()),
            max_position_embeddings: Some(2048),
            layer_types: vec!["sliding_attention".into(), "full_attention".into()],
            attention_k_eq_v: true,
            final_logit_softcapping: softcap,
            rope_parameters: HashMap::from([
                (
                    "sliding_attention".to_string(),
                    crate::load::RopeParams {
                        rope_type: "default".into(),
                        rope_theta: 1.0e4,
                        partial_rotary_factor: None,
                    },
                ),
                (
                    "full_attention".to_string(),
                    crate::load::RopeParams {
                        rope_type: "proportional".into(),
                        rope_theta: 1.0e6,
                        partial_rotary_factor: Some(0.5),
                    },
                ),
            ]),
        };
        let cfg = GemmaCfg::from_text_config(&tc).expect("synthetic cfg");

        let h = cfg.hidden;
        let inter = cfg.intermediate;
        let mut src = SynthSource::default();
        // embed: deterministic small values, BF16.
        let embed: Vec<f32> = (0..cfg.vocab * h)
            .map(|i| ((i % 37) as f32 - 18.0) / 64.0)
            .collect();
        src.insert_bf16(
            "model.language_model.embed_tokens.weight",
            &embed,
            vec![cfg.vocab, h],
        );

        fn rng(n: usize, i: usize) -> Vec<f32> {
            (0..n)
                .map(|j| 1.0 + ((j * 7 + i) % 5) as f32 * 0.05)
                .collect()
        }

        let mut seed = 11u32;
        for i in 0..cfg.n_layers {
            let p = format!("model.language_model.layers.{i}");
            let lc = cfg.layers[i];
            for ln in [
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
            ] {
                src.insert_bf16(&format!("{p}.{ln}.weight"), &rng(h, i), vec![h]);
            }
            src.insert_bf16(
                &format!("{p}.layer_scalar"),
                &[1.0 - 0.1 * (i as f32)],
                vec![1],
            );
            src.insert_bf16(
                &format!("{p}.self_attn.q_norm.weight"),
                &rng(lc.head_dim, i),
                vec![lc.head_dim],
            );
            src.insert_bf16(
                &format!("{p}.self_attn.k_norm.weight"),
                &rng(lc.head_dim, i),
                vec![lc.head_dim],
            );
            seed += 1;
            src.insert_linear(
                &format!("{p}.self_attn.q_proj"),
                h,
                cfg.n_heads * lc.head_dim,
                3,
                seed,
            );
            seed += 1;
            src.insert_linear(
                &format!("{p}.self_attn.k_proj"),
                h,
                lc.n_kv * lc.head_dim,
                3,
                seed,
            );
            seed += 1;
            if lc.kind == LayerKind::Sliding {
                src.insert_linear(
                    &format!("{p}.self_attn.v_proj"),
                    h,
                    lc.n_kv * lc.head_dim,
                    3,
                    seed,
                );
                seed += 1;
            }
            src.insert_linear(
                &format!("{p}.self_attn.o_proj"),
                cfg.n_heads * lc.head_dim,
                h,
                3,
                seed,
            );
            seed += 1;
            src.insert_linear(&format!("{p}.mlp.gate_proj"), h, inter, 3, seed);
            seed += 1;
            src.insert_linear(&format!("{p}.mlp.up_proj"), h, inter, 3, seed);
            seed += 1;
            src.insert_linear(&format!("{p}.mlp.down_proj"), inter, h, 3, seed);
            seed += 1;
        }
        src.insert_bf16("model.language_model.norm.weight", &rng(h, 3), vec![h]);
        src.insert_linear("lm_head", h, cfg.vocab, 3, 9999);
        (src, cfg)
    }

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
            let tc = TextConfig {
                hidden_size: cfg.hidden,
                intermediate_size: cfg.intermediate,
                head_dim: cfg.layers[0].head_dim,
                global_head_dim: cfg.layers[1].head_dim,
                num_attention_heads: cfg.n_heads,
                num_key_value_heads: cfg.layers[0].n_kv,
                num_global_key_value_heads: Some(cfg.layers[1].n_kv),
                num_hidden_layers: cfg.layers.len(),
                vocab_size: cfg.vocab,
                rms_norm_eps: cfg.eps,
                sliding_window: Some(cfg.sliding_window),
                hidden_activation: None,
                max_position_embeddings: Some(2048),
                layer_types: vec!["sliding_attention".into(), "full_attention".into()],
                attention_k_eq_v: true,
                final_logit_softcapping: cfg.softcap,
                rope_parameters: HashMap::new(),
            };
            tc
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
}
