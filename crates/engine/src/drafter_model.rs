//! spec/mtp-40tps T6: drafter forward pass (single-token decode step).
//!
//! The drafter (`Gemma4AssistantForCausalLM`, text_config hidden 1024, 4
//! layers) reads the target's *shared* K/V per layer type and produces a
//! hidden state sized for the target. This module mirrors the MLX reference
//! (`mlx_vlm.speculative.drafters.gemma4_assistant`); the numerical contract
//! is pinned by `tests/drafter_forward.rs` against golden arrays dumped from
//! the reference.

use kaplumba_kernels::affine::AffineKernels;
use kaplumba_kernels::attention::{AttentionKernels, AttentionRequest};
use kaplumba_kernels::elements::{ElementKernels, RopeRequest};

use crate::drafter::DrafterWeights;

pub struct DrafterSharedKv {
    pub sliding: KvSlice,
    pub full: KvSlice,
}

pub struct KvSlice {
    /// [position][kv_head * head_dim] layout, identical to the target's cache.
    pub keys: Vec<u16>,
    pub values: Vec<u16>,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub len: usize,
    /// key position array sent to the attention kernel; `0..len` suffices when
    /// the slice holds a real contiguous prefix.
    pub positions: Vec<u32>,
}

struct Layer {
    full: bool,
    n_q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    sliding_window: usize,
    intermediate: usize,
}

pub struct DrafterModel {
    weights: DrafterWeights,
    affine: &'static AffineKernels,
    attention: AttentionKernels,
    elements: &'static ElementKernels,
    layers: [Layer; 4],
}

impl DrafterModel {
    pub fn new(weights: DrafterWeights) -> Result<Self, DrafterError> {
        let affine = AffineKernels::shared().map_err(|e| DrafterError::Kernels(e.to_string()))?;
        let attention =
            AttentionKernels::new().map_err(|e| DrafterError::Kernels(e.to_string()))?;
        let elements =
            ElementKernels::shared().map_err(|e| DrafterError::Kernels(e.to_string()))?;
        let layer_types = [false, false, false, true]; // sliding, sliding, sliding, full
        let mut layers = Vec::new();
        for i in 0..4usize {
            let full = layer_types[i];
            let head_shape = shape_of_qproj(&weights, i)?;
            let (n_q_heads, head_dim) = if full {
                (32usize, 512usize)
            } else {
                (32usize, 256usize)
            };
            debug_assert_eq!(n_q_heads * head_dim, head_shape);
            let kv_heads = if full { 4 } else { 16 };
            let intermediate = {
                let s = weights
                    .shape(&format!("model.layers.{i}.mlp.gate_proj.weight"))
                    .ok_or(DrafterError::MissingTensor)?;
                s[0] // [intermediate, hidden/8]
            };
            layers.push(Layer {
                full,
                n_q_heads,
                kv_heads,
                head_dim,
                sliding_window: if full { usize::MAX } else { 1024 },
                intermediate,
            });
        }
        let mut it = layers.into_iter();
        let layers = [
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
        ];
        Ok(Self {
            weights,
            affine,
            attention,
            elements,
            layers,
        })
    }

    /// Single-token decode step: inputs_embeds_row is the concat
    /// `[target_embed(last_token), previous_target_hidden]` of width 10752.
    /// Returns last_hidden (width 5376) and post-projection-hidden (width 1024).
    pub fn forward(
        &self,
        inputs_embeds_row: &[u16],
        shared: &DrafterSharedKv,
        position: usize,
    ) -> Result<DrafterStep, DrafterError> {
        if inputs_embeds_row.len() != 2 * 5376 {
            return Err(DrafterError::BadShape);
        }
        let mut h: Vec<u16> = self.gemv_named(
            inputs_embeds_row,
            "pre_projection",
            1024,
            inputs_embeds_row.len(),
        )?;

        for (i, layer) in self.layers.iter().enumerate() {
            let prefix = format!("model.layers.{i}");
            let hn = self.elements.rms_norm(
                &h,
                self.tensor_as_f16(&format!("{prefix}.input_layernorm.weight"))?,
                1,
                1024,
                1e-6,
            )?;
            let q = self.gemv_named(
                &hn,
                &format!("model.layers.{i}.self_attn.q_proj"),
                layer.n_q_heads * layer.head_dim,
                1024,
            )?;
            let q = self.elements.rms_norm(
                &q,
                self.tensor_as_f16(&format!("{prefix}.self_attn.q_norm.weight"))?,
                layer.n_q_heads,
                layer.head_dim,
                1e-6,
            )?;
            let q = self.elements.rope(RopeRequest {
                input: &q,
                tokens: 1,
                heads: layer.n_q_heads,
                head_dim: layer.head_dim,
                offset: position,
                theta: if layer.full { 1_000_000.0 } else { 10_000.0 },
                partial_factor: if layer.full { 0.25 } else { 1.0 },
                proportional: layer.full,
            })?;
            let kv = if layer.full { &shared.full } else { &shared.sliding };
            if kv.kv_heads * kv.head_dim != kv.keys.len() / kv.len.max(1) {
                return Err(DrafterError::BadShape);
            }
            let attn = self.attention.online_attention(AttentionRequest {
                queries: &q,
                keys: &kv.keys,
                values: &kv.values,
                query_positions: &[position as u32],
                key_positions: &kv.positions,
                q_heads: layer.n_q_heads,
                kv_heads: kv.kv_heads,
                head_dim: kv.head_dim,
                scale: 1.0,
                window: if layer.full { 0 } else { layer.sliding_window },
            })?;
            let o = self.gemv_named(
                &attn,
                &format!("model.layers.{i}.self_attn.o_proj"),
                1024,
                layer.n_q_heads * layer.head_dim,
            )?;
            let o = self.elements.rms_norm(
                &o,
                self.tensor_as_f16(&format!("{prefix}.post_attention_layernorm.weight"))?,
                1,
                1024,
                1e-6,
            )?;
            let mut out = self.elements.residual_add(&h, &o)?;

            let resid = out.clone();
            let hn = self.elements.rms_norm(
                &out,
                self.tensor_as_f16(&format!("{prefix}.pre_feedforward_layernorm.weight"))?,
                1,
                1024,
                1e-6,
            )?;
            let gate = self.gemv_named(
                &hn,
                &format!("model.layers.{i}.mlp.gate_proj"),
                layer.intermediate,
                1024,
            )?;
            let up = self.gemv_named(
                &hn,
                &format!("model.layers.{i}.mlp.up_proj"),
                layer.intermediate,
                1024,
            )?;
            let act = self.elements.gated_gelu(&gate, &up)?;
            let mlp = self.gemv_named(
                &act,
                &format!("model.layers.{i}.mlp.down_proj"),
                1024,
                layer.intermediate,
            )?;
            let mlp = self.elements.rms_norm(
                &mlp,
                self.tensor_as_f16(&format!("{prefix}.post_feedforward_layernorm.weight"))?,
                1,
                1024,
                1e-6,
            )?;
            out = self.elements.residual_add(&resid, &mlp)?;
            let scalar = layer_scalar_of(&self.weights, i)?;
            h = self.elements.layer_scale(&out, scalar)?;
        }

        let h_normed = self.elements.rms_norm(
            &h,
            self.tensor_as_f16("model.norm.weight")?,
            1,
            1024,
            1e-6,
        )?;
        let last_hidden =
            self.gemv_named(&h_normed, "post_projection", 5376, 1024)?;
        Ok(DrafterStep { last_hidden, hidden: h_normed })
    }

    fn gemv_named(
        &self,
        x: &[u16],
        base: &str,
        n: usize,
        k: usize,
    ) -> Result<Vec<u16>, DrafterError> {
        let w = self
            .weights
            .buffer(&format!("{base}.weight"))
            .ok_or(DrafterError::MissingTensor)?;
        let s = self
            .weights
            .buffer(&format!("{base}.scales"))
            .ok_or(DrafterError::MissingTensor)?;
        let b = self
            .weights
            .buffer(&format!("{base}.biases"))
            .ok_or(DrafterError::MissingTensor)?;
        let y = self.affine.alloc_u16(n).map_err(|e| DrafterError::Kernels(e.to_string()))?;
        let xb = {
            let xb = self
                .affine
                .alloc_u16(x.len())
                .map_err(|e| DrafterError::Kernels(e.to_string()))?;
            unsafe {
                std::ptr::copy_nonoverlapping(x.as_ptr(), xb.contents() as *mut u16, x.len());
            }
            xb
        };
        self.affine
            .gemv_buffers(&xb, w, s, b, &y, n, k, 64)
            .map_err(|e| DrafterError::Kernels(e.to_string()))?;
        let src = y.contents() as *const u16;
        let mut out = vec![0u16; n];
        unsafe {
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), n);
        }
        Ok(out)
    }

    fn tensor_as_f16(&self, name: &str) -> Result<&[u16], DrafterError> {
        // DrafterWeights converted bf16->f16 at load; peek at the buffer's contents.
        let buf = self
            .weights
            .buffer(name)
            .ok_or(DrafterError::MissingTensor)?;
        let n = self.weights.shape(name).ok_or(DrafterError::MissingTensor)?.iter().product::<usize>();
        let ptr = buf.contents() as *const u16;
        Ok(unsafe { std::slice::from_raw_parts(ptr, n) })
    }
}

fn shape_of_qproj(weights: &DrafterWeights, i: usize) -> Result<usize, DrafterError> {
    let s = weights
        .shape(&format!("model.layers.{i}.self_attn.q_proj.weight"))
        .ok_or(DrafterError::MissingTensor)?;
    Ok(s[0])
}

fn layer_scalar_of(weights: &DrafterWeights, i: usize) -> Result<f32, DrafterError> {
    let buf = weights
        .buffer(&format!("model.layers.{i}.layer_scalar"))
        .ok_or(DrafterError::MissingTensor)?;
    let ptr = buf.contents() as *const u16;
    Ok(half::f16::from_bits(unsafe { *ptr }).to_f32())
}

pub struct DrafterStep {
    pub last_hidden: Vec<u16>,
    pub hidden: Vec<u16>,
}

#[derive(Debug, thiserror::Error)]
pub enum DrafterError {
    #[error("kernels: {0}")]
    Kernels(String),
    #[error("missing tensor")]
    MissingTensor,
    #[error("bad shape")]
    BadShape,
}

impl From<kaplumba_kernels::exl3::KernelError> for DrafterError {
    fn from(e: kaplumba_kernels::exl3::KernelError) -> Self {
        DrafterError::Kernels(e.to_string())
    }
}
