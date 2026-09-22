//! Test-only fixtures: a synthetic EXL3 artifact (no Metal, no 16 GB), shared by
//! the `gemma4` and `generate` test modules.
//!
//! Dims are the smallest that satisfy every real-path constraint: all linear
//! **outputs** are multiples of 128 (right Hadamard block), `hidden` and
//! `vocab` are multiples of 128, and `n_kv * head_dim` is a multiple of 128.
//! The model is 2 layers (sliding + full) with `k_eq_v`, window 4 and softcap 30
//! so the sliding mask, the weightless-V branch and the softcap all execute.

use std::collections::HashMap;

use half::f16;

use crate::exl3::codebook::CodebookMode;
use crate::gemma4::{GemmaCfg, LayerKind, RawTensor, TensorSource};
use crate::load::{DType, RopeParams, TextConfig};

pub(crate) struct SynthTensor {
    pub(crate) bytes: Vec<u8>,
    pub(crate) dtype: DType,
    pub(crate) shape: Vec<usize>,
}

#[derive(Default)]
pub(crate) struct SynthSource {
    pub(crate) map: HashMap<String, SynthTensor>,
}

impl SynthSource {
    pub(crate) fn insert(&mut self, name: &str, bytes: Vec<u8>, dtype: DType, shape: Vec<usize>) {
        self.map.insert(
            name.to_string(),
            SynthTensor {
                bytes,
                dtype,
                shape,
            },
        );
    }

    pub(crate) fn insert_f16(&mut self, name: &str, values: &[f16]) {
        let mut b = Vec::with_capacity(values.len() * 2);
        for v in values {
            b.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        self.insert(name, b, DType::F16, vec![values.len()]);
    }

    pub(crate) fn insert_bf16(&mut self, name: &str, values: &[f32], shape: Vec<usize>) {
        let mut b = Vec::with_capacity(values.len() * 2);
        for v in values {
            b.extend_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
        }
        self.insert(name, b, DType::BF16, shape);
    }

    /// Deterministic synthetic EXL3 linear (`mcg`, k-bit trellis, ±1 svh).
    pub(crate) fn insert_linear(&mut self, name: &str, inn: usize, out: usize, k: u32, seed: u32) {
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

/// Real EXL3 artifact directory, when present (`KAPLUMBA_MODEL_DIR` override).
pub(crate) fn artifact_dir() -> Option<std::path::PathBuf> {
    let dir = match std::env::var("KAPLUMBA_MODEL_DIR") {
        Ok(d) => std::path::PathBuf::from(d),
        Err(_) => {
            let h = std::env::var("HOME").ok()?;
            std::path::PathBuf::from(format!("{h}/.kaplumba/models/Gxma/Gxma"))
        }
    };
    dir.join("config.json").is_file().then_some(dir)
}

pub(crate) fn norm_vals(n: usize, i: usize) -> Vec<f32> {
    (0..n)
        .map(|j| 1.0 + ((j * 7 + i) % 5) as f32 * 0.05)
        .collect()
}

/// Tiny 2-layer model: hidden 128, intermediate 256, vocab 384, 2 heads of
/// head_dim 64, 2 kv heads, window 4, softcap 30, k_eq_v (the full layer has no
/// `v_proj`).
pub(crate) fn synth_model(softcap: Option<f64>) -> (SynthSource, GemmaCfg) {
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
                RopeParams {
                    rope_type: "default".into(),
                    rope_theta: 1.0e4,
                    partial_rotary_factor: None,
                },
            ),
            (
                "full_attention".to_string(),
                RopeParams {
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
    let embed: Vec<f32> = (0..cfg.vocab * h)
        .map(|i| ((i % 37) as f32 - 18.0) / 64.0)
        .collect();
    src.insert_bf16(
        "model.language_model.embed_tokens.weight",
        &embed,
        vec![cfg.vocab, h],
    );

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
            src.insert_bf16(&format!("{p}.{ln}.weight"), &norm_vals(h, i), vec![h]);
        }
        src.insert_bf16(
            &format!("{p}.layer_scalar"),
            &[1.0 - 0.1 * (i as f32)],
            vec![1],
        );
        src.insert_bf16(
            &format!("{p}.self_attn.q_norm.weight"),
            &norm_vals(lc.head_dim, i),
            vec![lc.head_dim],
        );
        src.insert_bf16(
            &format!("{p}.self_attn.k_norm.weight"),
            &norm_vals(lc.head_dim, i),
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
    src.insert_bf16(
        "model.language_model.norm.weight",
        &norm_vals(h, 3),
        vec![h],
    );
    src.insert_linear("lm_head", h, cfg.vocab, 3, 9999);
    (src, cfg)
}

/// `TextConfig` matching `synth_model`'s dims, with the layer list overridable
/// (used to exercise config validation).
pub(crate) fn synth_text_config(cfg: &GemmaCfg) -> TextConfig {
    TextConfig {
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
    }
}
