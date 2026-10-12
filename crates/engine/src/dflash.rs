//! spec/dflash-correctness T1/T2: loader for the z-lab DFlash drafter.
//!
//! `~/.kaplumba/models/Drafter` holds a single `model.safetensors` shard of
//! EXL3-quantised weights plus BF16 RMSNorm scales. It is NOT the same format
//! as the MTP drafter (`crate::drafter`, MLX affine-4bit) — that one is
//! `.weight`/`.scales`/`.biases` triples and a completely different graph.
//!
//! Three things about this checkpoint that are easy to get wrong, all measured
//! off the file and recorded in `spec/dflash-correctness/implementation_notes.md`:
//!
//! 1. **The trellis is `[in/16, out/16, 64]` int16**, i.e. the *same* EXL3
//!    packing and mcg codebook the target model uses (`mcg_multiplier
//!    3417055213 == 0xCBAC1FED == MCG_MULTIPLIER`), so `QuantizedLinear` and
//!    the existing EXL3 gemv/gemm kernels apply unchanged.
//! 2. **The norm weights are stored WITHOUT Gemma's `+1` offset.** The
//!    repacked *target* checkpoints have it baked in (their final norm reads
//!    6.188, 8.0, 5.844...) and `rms_norm_weighted` multiplies by `weight`
//!    alone, so this loader must add `1.0` itself. Skipping it is a ~2.3x
//!    error on every residual stream and produces plausible-looking garbage.
//! 3. **The scales are already fp16** (`.suh`/`.svh`), so there is no
//!    bf16→f16 scale conversion and no packed `.su`/`.sv` path.

use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use metal::MTLResourceOptions;

use crate::exl3::{BitRate, MCG_MULTIPLIER};
use crate::model::QuantizedLinear;
use crate::safetensors::{read_shard_header, DType};
use kaplumba_kernels::attention::DflashBlockRequest;
use kaplumba_kernels::elements::RopeRequest;
use kaplumba_kernels::linear::LinearBufferRequest;

/// A drafter RMSNorm: `hidden_size` fp16 values, already offset-corrected.
pub type NormWeights = Vec<u16>;

pub struct DflashWeights {
    /// EXL3 linears, keyed by the checkpoint's own tensor stem
    /// (`fc`, `layers.0.self_attn.q_proj`, ...).
    pub lines: BTreeMap<String, QuantizedLinear>,
    /// RMSNorm scales, already `+1.0`-offset. Keyed the same way
    /// (`hidden_norm`, `norm`, `layers.0.input_layernorm`, ...).
    pub norms: BTreeMap<String, NormWeights>,
    /// Raw EXL3 scale vectors for the linears, kept so tests and the
    /// dedup check can compare against the file without re-reading it.
    pub suh: BTreeMap<String, Vec<u16>>,
    pub svh: BTreeMap<String, Vec<u16>>,
    /// Tensor names in the shard that this loader did not recognise. A
    /// non-empty list means the checkpoint carries something we are silently
    /// ignoring — the caller should refuse to run rather than load a subtly
    /// wrong model.
    pub unknown_tensors: Vec<String>,
}

/// Every tensor suffix this loader understands, and what it does with it.
///
/// Used by [`DflashWeights::unhandled_tensors`] so a checkpoint that adds a
/// tensor we silently drop fails loudly instead of loading a subtly wrong
/// model. The original DFlash release can ship `mask_embedding`,
/// `attention_sink_bias`, and per-tap `hidden_norms.*`; exllamav3's
/// `DFlashConfig` (`architecture/dflash.py:83-98`) probes for all of them and
/// this checkpoint has none of them.
const SUFFIX_RULES: &[(&str, &str)] = &[
    (".trellis", "EXL3 packed states"),
    (".mcg", "codebook selector"),
    (".suh", "input-side scale"),
    (".svh", "output-side scale"),
    (".su", "packed input-side scale"),
    (".sv", "packed output-side scale"),
    (".weight", "RMSNorm scale"),
];

#[derive(Debug, thiserror::Error)]
pub enum DflashLoadError {
    #[error("io: {0}")]
    Io(String),
    #[error("shard header: {0}")]
    Shard(String),
    #[error("missing tensor {0}")]
    Missing(String),
    #[error("{name}: {message}")]
    Tensor { name: String, message: String },
    #[error("unsupported dtype {dtype:?} for {name}")]
    DType { name: String, dtype: DType },
    #[error("{name} has {actual} elements, expected {expected}")]
    Length { name: String, actual: usize, expected: usize },
}

fn read_exact_at(
    file: &mut std::fs::File,
    offset: u64,
    len: usize,
) -> Result<Vec<u8>, DflashLoadError> {
    let mut bytes = vec![0u8; len];
    file.seek(SeekFrom::Start(offset))
        .map_err(|e| DflashLoadError::Io(e.to_string()))?;
    file.read_exact(&mut bytes)
        .map_err(|e| DflashLoadError::Io(e.to_string()))?;
    Ok(bytes)
}

impl DflashWeights {
    /// Load the drafter shard. `dir` must contain `model.safetensors`.
    ///
    /// Every tensor in the header is either consumed by one of the
    /// [`SUFFIX_RULES`] or reported by [`Self::unhandled_tensors`]; the caller
    /// is expected to check the latter, because a checkpoint we half-understand
    /// is worse than one we refuse.
    pub fn load(dir: &Path, device: &metal::Device) -> Result<Self, DflashLoadError> {
        let shard_path = dir.join("model.safetensors");
        let shard = read_shard_header(&shard_path)
            .map_err(|e| DflashLoadError::Shard(e.to_string()))?;
        let mut file = std::fs::File::open(&shard_path)
            .map_err(|e| DflashLoadError::Io(e.to_string()))?;

        let mut trellis: BTreeMap<String, (Vec<u8>, DType)> = BTreeMap::new();
        let mut mcg: BTreeMap<String, i32> = BTreeMap::new();
        let mut suh: BTreeMap<String, Vec<u16>> = BTreeMap::new();
        let mut svh: BTreeMap<String, Vec<u16>> = BTreeMap::new();
        let mut bf16_norms: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        let mut unknown: BTreeSet<String> = BTreeSet::new();

        for (name, info) in &shard.tensors {
            let (start, end) = shard
                .absolute_range(name)
                .ok_or_else(|| DflashLoadError::Missing(name.clone()))?;
            let bytes = read_exact_at(
                &mut file,
                start,
                usize::try_from(end - start).unwrap_or(0),
            )?;
            let known = SUFFIX_RULES
                .iter()
                .any(|(suffix, _)| name.ends_with(suffix));
            if !known {
                unknown.insert(name.clone());
                continue;
            }
            if name.ends_with(".trellis") {
                trellis.insert(strip(name, ".trellis"), (bytes, info.dtype));
            } else if name.ends_with(".mcg") {
                if info.dtype != DType::I32 || bytes.len() != 4 {
                    return Err(DflashLoadError::Tensor {
                        name: name.clone(),
                        message: format!(
                            "mcg must be a single i32, got {:?} of {} bytes",
                            info.dtype,
                            bytes.len()
                        ),
                    });
                }
                mcg.insert(strip(name, ".mcg"), i32::from_le_bytes(bytes[..4].try_into().unwrap()));
            } else if name.ends_with(".suh") || name.ends_with(".su") {
                expect_f16(&bytes, name, len_of(info))?;
                let values = f16_bytes(&bytes);
                suh.insert(strip_any(name, &[".suh", ".su"]), values);
            } else if name.ends_with(".svh") || name.ends_with(".sv") {
                expect_f16(&bytes, name, len_of(info))?;
                let values = f16_bytes(&bytes);
                svh.insert(strip_any(name, &[".svh", ".sv"]), values);
            } else if name.ends_with(".weight") {
                if info.dtype != DType::BF16 {
                    return Err(DflashLoadError::DType {
                        name: name.clone(),
                        dtype: info.dtype,
                    });
                }
                if bytes.len() % 2 != 0 {
                    return Err(DflashLoadError::Length {
                        name: name.clone(),
                        actual: bytes.len(),
                        expected: (bytes.len() / 2) * 2,
                    });
                }
                bf16_norms.insert(strip(name, ".weight"), bytes);
            }
        }

        let unknown = unknown.into_iter().collect::<Vec<_>>();

        let mut lines = BTreeMap::new();
        for (stem, (packed, dtype)) in trellis {
            let shape = shard
                .tensors
                .get(&format!("{stem}.trellis"))
                .map(|t| t.shape.clone())
                .ok_or_else(|| DflashLoadError::Missing(format!("{stem}.trellis")))?;
            if shape.len() != 3 {
                return Err(DflashLoadError::Tensor {
                    name: format!("{stem}.trellis"),
                    message: format!("expected [in/16, out/16, 64], got {:?}", shape),
                });
            }
            let in_tiles = shape[0];
            let out_tiles = shape[1];
            let in_features = in_tiles * 16;
            let out_features = out_tiles * 16;
            let codebook_dword = mcg
                .get(&stem)
                .copied()
                .ok_or_else(|| DflashLoadError::Missing(format!("{stem}.mcg")))?;
            if codebook_dword as u32 != MCG_MULTIPLIER {
                return Err(DflashLoadError::Tensor {
                    name: format!("{stem}.mcg"),
                    message: format!(
                        "mcg multiplier {codebook_dword:#x} is not the mcg codebook {MCG_MULTIPLIER:#x}"
                    ),
                });
            }
            let expected_words = in_tiles * out_tiles * BitRate::Integer(4).packed_words();
            if dtype != DType::I16 || packed.len() / 2 != expected_words {
                return Err(DflashLoadError::Length {
                    name: format!("{stem}.trellis"),
                    actual: if dtype == DType::I16 { packed.len() / 2 } else { packed.len() },
                    expected: expected_words,
                });
            }
            let input_signs = suh
                .get(&stem)
                .ok_or_else(|| DflashLoadError::Missing(format!("{stem}.suh")))?
                .clone();
            if input_signs.len() != in_features {
                return Err(DflashLoadError::Length {
                    name: format!("{stem}.suh"),
                    actual: input_signs.len(),
                    expected: in_features,
                });
            }
            let output_signs = svh
                .get(&stem)
                .ok_or_else(|| DflashLoadError::Missing(format!("{stem}.svh")))?
                .clone();
            if output_signs.len() != out_features {
                return Err(DflashLoadError::Length {
                    name: format!("{stem}.svh"),
                    actual: output_signs.len(),
                    expected: out_features,
                });
            }
            let packed_words = packed.len() / 2;
            let buffer = device.new_buffer(
                packed.len().max(2) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            unsafe {
                std::ptr::copy_nonoverlapping(
                    packed.as_ptr(),
                    buffer.contents() as *mut u8,
                    packed.len(),
                );
            }
            let linear = QuantizedLinear::from_metal_buffer(
                in_features,
                out_features,
                kaplumba_kernels::exl3::Exl3Rate::Integer(4),
                kaplumba_kernels::exl3::Exl3Codebook::Mcg,
                buffer,
                0,
                Some(packed_words),
                input_signs,
                output_signs,
            )
            .map_err(|e| DflashLoadError::Tensor {
                name: stem.clone(),
                message: e.to_string(),
            })?;
            lines.insert(stem, linear);
        }

        // T2. Gemma's RMSNorm is `(1 + weight) * x / rms(x)`. The engine's
        // `rms_norm_weighted` (elements.metal:110) multiplies by `weight` alone
        // and the *target* checkpoints are repacked with the offset folded in,
        // so this checkpoint's raw values (0.30-0.82 on the input norms) have
        // to be shifted here or every residual stream is scaled wrong.
        //
        // BF16 is the top 16 bits of an f32, so widening is exact; add 1.0 in
        // f32 (matching what the trainer's BF16 `(1 + w)` did before it was
        // stored) and narrow once, to f16, at the very end.
        let mut norms = BTreeMap::new();
        for (stem, bytes) in bf16_norms {
            let mut values = Vec::with_capacity(bytes.len() / 2);
            for pair in bytes.chunks_exact(2) {
                let bits = u16::from_le_bytes([pair[0], pair[1]]);
                let widened = f32::from_bits((bits as u32) << 16);
                values.push(half::f16::from_f32(widened + 1.0).to_bits());
            }
            norms.insert(stem, values);
        }

        Ok(Self {
            lines,
            norms,
            suh,
            svh,
            unknown_tensors: unknown,
        })
    }

    pub fn line(&self, name: &str) -> Result<&QuantizedLinear, DflashLoadError> {
        self.lines
            .get(name)
            .ok_or_else(|| DflashLoadError::Missing(name.to_string()))
    }

    pub fn norm(&self, name: &str) -> Result<&[u16], DflashLoadError> {
        self.norms
            .get(name)
            .map(|values| values.as_slice())
            .ok_or_else(|| DflashLoadError::Missing(name.to_string()))
    }
}

fn strip(name: &str, suffix: &str) -> String {
    name.trim_end_matches(suffix).to_string()
}

fn strip_any(name: &str, suffixes: &[&str]) -> String {
    let mut out = name.to_string();
    for suffix in suffixes {
        if let Some(stripped) = out.strip_suffix(suffix) {
            out = stripped.to_string();
        }
    }
    out
}

fn len_of(info: &crate::safetensors::TensorInfo) -> usize {
    info.shape.iter().product()
}

fn expect_f16(bytes: &[u8], name: &str, elements: usize) -> Result<(), DflashLoadError> {
    if bytes.len() != elements * 2 {
        return Err(DflashLoadError::Length {
            name: name.to_string(),
            actual: bytes.len() / 2,
            expected: elements,
        });
    }
    Ok(())
}

fn f16_bytes(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect()
}

// ---------------------------------------------------------------------------
// spec/dflash-correctness T3: the drafter's config, parsed from its
// `config.json`, and the block forward pass.
// ---------------------------------------------------------------------------

/// Parsed from `~/.kaplumba/models/Drafter/config.json`.
///
/// The file's `architectures` is `["DFlashDraftModel"]` and its `model_type` is
/// `qwen3`, which is z-lab's generic template default and is *misleading*: the
/// norm offsets (see the module doc) and the softcap make this a Gemma-family
/// checkpoint. Nothing here reads `model_type`.
#[derive(Debug, Clone)]
pub struct DflashConfig {
    pub block_size: usize,
    pub mask_token_id: u32,
    /// 0-based `DecoderLayer` indices of the target whose post-layer hidden
    /// states are the taps. The config's `target_layer_ids` are already in
    /// this space: z-lab's `extract_context_feature` uses `offset = 1`, which
    /// cancels the `hidden_states` list's leading embedding entry.
    pub target_layer_ids: Vec<usize>,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub sliding_window: usize,
    pub layer_is_sliding: Vec<bool>,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub final_logit_softcapping: f32,
    /// From `num_target_layers`; only used as a sanity bound on the taps.
    pub num_target_layers: usize,
}

impl DflashConfig {
    pub fn num_taps(&self) -> usize {
        self.target_layer_ids.len()
    }

    /// `fc`'s input width: one hidden row per tap, concatenated.
    pub fn tap_width(&self) -> usize {
        self.hidden_size * self.target_layer_ids.len()
    }

    pub fn draft_tokens(&self) -> usize {
        // A block is [anchor, mask x (block_size - 1)] and the anchor row
        // predicts a token the target has already emitted, so it is dropped.
        self.block_size - 1
    }

    pub fn load(dir: &Path) -> Result<Self, DflashLoadError> {
        let path = dir.join("config.json");
        let bytes = std::fs::read(&path).map_err(|e| DflashLoadError::Io(e.to_string()))?;
        let json: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|e| DflashLoadError::Io(format!("config.json: {e}")))?;
        let dflash = json.get("dflash_config");
        let number = |key: &str| -> Option<f64> { json.get(key).and_then(|v| v.as_f64()) };
        let text = |key: &str| json.get(key).and_then(|v| v.as_str());

        let layer_types = json
            .get("layer_types")
            .and_then(|value| value.as_array())
            .map(|values| {
                values
                    .iter()
                    .map(|value| value.as_str().unwrap_or("full_attention") == "sliding_attention")
                    .collect::<Vec<_>>()
            })
            .ok_or_else(|| DflashLoadError::Missing("layer_types".to_string()))?;
        let num_layers = require(number("num_hidden_layers"), "num_hidden_layers")?;
        if layer_types.len() != num_layers {
            return Err(DflashLoadError::Tensor {
                name: "layer_types".to_string(),
                message: format!(
                    "{} entries for {num_layers} layers",
                    layer_types.len()
                ),
            });
        }

        let target_layer_ids = dflash
            .and_then(|value| value.get("target_layer_ids"))
            .and_then(|value| value.as_array())
            .map(|values| {
                values
                    .iter()
                    .filter_map(|value| value.as_u64())
                    .map(|value| usize::try_from(value).unwrap_or(usize::MAX))
                    .collect::<Vec<_>>()
            })
            .ok_or_else(|| DflashLoadError::Missing("dflash_config.target_layer_ids".to_string()))?;
        if target_layer_ids.iter().any(|id| *id == usize::MAX) {
            return Err(DflashLoadError::Tensor {
                name: "dflash_config.target_layer_ids".to_string(),
                message: "a tap index is not a usable layer number".to_string(),
            });
        }
        let mut sorted = target_layer_ids.clone();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.len() != target_layer_ids.len() {
            return Err(DflashLoadError::Tensor {
                name: "dflash_config.target_layer_ids".to_string(),
                message: "tap layer indices must be unique".to_string(),
            });
        }
        let num_target_layers = require(number("num_target_layers"), "num_target_layers")?;
        if let Some(max) = target_layer_ids.iter().max()
            && *max >= num_target_layers
        {
            return Err(DflashLoadError::Tensor {
                name: "dflash_config.target_layer_ids".to_string(),
                message: format!("tap {max} is outside the target's {num_target_layers} layers"),
            });
        }

        let block_size = require(
            number("block_size").or_else(|| {
                dflash
                    .and_then(|value| value.get("block_size"))
                    .and_then(|v| v.as_f64())
            }),
            "block_size",
        )?;
        let mask_token_id = u32::try_from(
            dflash
                .and_then(|value| value.get("mask_token_id"))
                .and_then(|value| value.as_u64())
                .or_else(|| number("mask_token_id").map(|value| value as u64))
                .ok_or_else(|| DflashLoadError::Missing("mask_token_id".to_string()))?,
        )
        .map_err(|_| DflashLoadError::Missing("mask_token_id".to_string()))?;


        let config = Self {
            block_size,
            mask_token_id,
            target_layer_ids,
            hidden_size: require(number("hidden_size"), "hidden_size")?,
            num_layers,
            q_heads: require(number("num_attention_heads"), "num_attention_heads")?,
            kv_heads: number("num_key_value_heads")
                .map(|value| value as usize)
                .unwrap_or_else(|| require(number("num_attention_heads"), "num_attention_heads").unwrap_or(1)),
            head_dim: number("head_dim")
                .map(|value| value as usize)
                .ok_or_else(|| DflashLoadError::Missing("head_dim".to_string()))?,
            intermediate: require(number("intermediate_size"), "intermediate_size")?,
            sliding_window: number("sliding_window").map(|value| value as usize).unwrap_or(0),
            layer_is_sliding: layer_types,
            rms_norm_eps: number("rms_norm_eps").map(|value| value as f32).unwrap_or(1e-6),
            rope_theta: number("rope_theta").map(|value| value as f32).unwrap_or(10_000.0),
            final_logit_softcapping: number("final_logit_softcapping")
                .map(|value| value as f32)
                .unwrap_or(0.0),
            num_target_layers,
        };
        let _ = text;
        if config.block_size < 2 {
            return Err(DflashLoadError::Tensor {
                name: "block_size".to_string(),
                message: "a block-diffusion drafter needs at least one masked position".to_string(),
            });
        }
        if config.kv_heads == 0 || !config.q_heads.is_multiple_of(config.kv_heads) {
            return Err(DflashLoadError::Tensor {
                name: "num_key_value_heads".to_string(),
                message: format!(
                    "kv heads {} does not divide q heads {}",
                    config.kv_heads, config.q_heads
                ),
            });
        }
        Ok(config)
    }
}

fn require(value: Option<f64>, name: &str) -> Result<usize, DflashLoadError> {
    let value = value.ok_or_else(|| DflashLoadError::Missing(name.to_string()))?;
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 {
        return Err(DflashLoadError::Tensor {
            name: name.to_string(),
            message: format!("expected a non-negative integer, got {value}"),
        });
    }
    Ok(value as usize)
}


/// NaN/inf probe used by `dflash_nan_probe` to localise a bad stage, and by
/// anyone debugging the drafter. Off unless `KAPLUMBA_DFLASH_TRACE` is set; the
/// check is O(n) over rows that are at most 10752 wide, so it is not free but
/// never runs in production. Mirrors `drafter_model::trace_stage`.
fn trace_stage(name: &str, values: &[u16]) {
    if std::env::var_os("KAPLUMBA_DFLASH_TRACE").is_none() {
        return;
    }
    let mut bad = 0usize;
    let mut magnitude = 0.0_f32;
    for bits in values {
        let value = half::f16::from_bits(*bits).to_f32();
        if value.is_nan() || value.is_infinite() {
            bad += 1;
        }
        magnitude = magnitude.max(value.abs());
    }
    eprintln!(
        "[dflash] {name}: len={} bad={bad} max|v|={magnitude:.4}",
        values.len()
    );
}

/// `trace_stage` for the f32 residual stream.
///
/// The drafter's residual stream is f32 precisely *because* it exceeds fp16's
/// range, so it cannot be traced through the f16 helper: narrowing it to log it
/// would print `inf` for the very stages under investigation.
fn trace_stage_f32(name: &str, values: &[f32]) {
    if std::env::var_os("KAPLUMBA_DFLASH_TRACE").is_none() {
        return;
    }
    let mut bad = 0usize;
    let mut magnitude = 0.0_f32;
    for value in values {
        if value.is_nan() || value.is_infinite() {
            bad += 1;
        }
        magnitude = magnitude.max(value.abs());
    }
    eprintln!(
        "[dflash] {name}: len={} bad={bad} max|v|={magnitude:.4}",
        values.len()
    );
}

/// The drafter's 5 transformer layers plus the `fc`/`hidden_norm` encoder.
///
/// Mirrors z-lab's `DFlashDraftModel` (`dflash/model.py:550-607`) layer for
/// layer; the only structural difference is that this side keeps the block's
/// own K/V and the context's K/V separate and hands them to
/// [`crate::attention`]'s block kernel, where z-lab concatenates them.
pub struct DflashModel {
    weights: DflashWeights,
    config: DflashConfig,
    /// The mask token's raw embedding row (`hidden_size` fp16). Empty until
    /// [`DflashModel::set_mask_embedding`]; a block forward without it would
    /// fill all 15 masked positions with the anchor, which produces a drafter
    /// that confidentially repeats itself.
    mask_embedding: Vec<u16>,
    linear: &'static kaplumba_kernels::linear::LinearKernels,
    attention: kaplumba_kernels::attention::AttentionKernels,
    elements: &'static kaplumba_kernels::elements::ElementKernels,
}

/// What the drafter needs about the current block before it can attend.
///
/// `ctx_keys`/`ctx_values` are `[n_ctx, kv_heads * head_dim]` per layer, already
/// RoPE'd and already k_norm'd -- they are the target's fused tap features
/// projected by each layer's own k_proj/v_proj, computed once when the row was
/// appended to the cache.
pub struct BlockContext<'a> {
    pub anchor_embedding: &'a [u16],
    /// Per layer: `[n_ctx_layer, kv_heads * head_dim]`, already RoPE'd and
    /// k_norm'd.
    ///
    /// Layers may hold **different row counts**, which is why there is no single
    /// `n_ctx` field: the drafter's last layer attends the whole context while
    /// layers 0..`num_layers - 1` slide at `sliding_window`, so once the context
    /// is longer than the window the two genuinely differ. Each layer's row
    /// count is `ctx_keys[layer].len() / (kv_heads * head_dim)`.
    pub ctx_keys: &'a [Vec<u16>],
    pub ctx_values: &'a [Vec<u16>],
    /// Per layer: absolute position of that layer's **first** retained context
    /// row. A sliding layer's oldest rows are evicted, so this is not
    /// `position - n_ctx` once the ring has wrapped.
    pub ctx_first: &'a [usize],
    /// Absolute position of the first block row (== the target's
    /// `next_position`, i.e. one past the last context row).
    pub position: usize,
}

/// A draft block's outputs.
pub struct DflashBlock {
    /// Post-`norm` hidden rows for the `block_size` positions, so the caller
    /// can apply the target's head once. `[block_size, hidden_size]`.
    pub hidden: Vec<u16>,
    /// The block's own K/V per layer, RoPE'd and k_norm'd, for a caller that
    /// wants to cache or discard them.
    pub block_keys: Vec<Vec<u16>>,
    pub block_values: Vec<Vec<u16>>,
}

/// One layer's retained context K/V.
#[derive(Debug, Clone, Default)]
struct KvLayer {
    held: usize,
    /// Rows retained, or `None` for the last layer, which attends everything.
    capacity: Option<usize>,
}

/// The drafter's context K/V — the paper's "KV injection".
///
/// The drafter does not build context K/V from its own tokens. Each accepted
/// target position contributes one fused tap feature, and every layer projects
/// that feature with its own `k_proj`/`v_proj` (`project_context_kv`); this
/// struct is where those land.
///
/// Layers 0..`num_layers - 1` slide: they never need more than
/// `sliding_window` rows, so a longer context evicts the oldest. The **last**
/// layer attends the whole context and never evicts, which is why
/// [`Self::rows`] can differ per layer and [`BlockContext`] carries per-layer
/// positions rather than one `n_ctx`.
#[derive(Debug, Clone)]
pub struct DflashKvCache {
    /// One flat `held * kv_width` f16 vector per layer.
    ///
    /// Deliberately a `Vec<Vec<u16>>` and not a per-row or single-buffer layout:
    /// this is already the shape [`BlockContext`] wants, so a block forward
    /// borrows it instead of copying 5x2x[rows x kv_width] every round.
    keys: Vec<Vec<u16>>,
    values: Vec<Vec<u16>>,
    layers: Vec<KvLayer>,
    /// Context rows held, globally. Advanced once per appended chunk (after
    /// every layer has been pushed) so it stays the context length rather than
    /// a per-layer counter; [`Self::first_position`] subtracts each layer's own
    /// retained count from it.
    total: usize,
    kv_width: usize,
}

impl DflashKvCache {
    pub fn new(config: &DflashConfig) -> Self {
        let last = config.num_layers.saturating_sub(1);
        let layers = (0..config.num_layers)
            .map(|layer| KvLayer {
                capacity: if layer == last || config.sliding_window == 0 {
                    None
                } else {
                    Some(config.sliding_window)
                },
                ..KvLayer::default()
            })
            .collect();
        Self {
            keys: vec![Vec::new(); config.num_layers],
            values: vec![Vec::new(); config.num_layers],
            layers,
            total: 0,
            kv_width: config.kv_heads * config.head_dim,
        }
    }

    /// Elements of one row's key (or value) vector.
    pub fn kv_width(&self) -> usize {
        self.kv_width
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Rows currently retained for `layer`. Not uniform across layers once the
    /// context is longer than the sliding window.
    pub fn rows(&self, layer: usize) -> usize {
        self.layers[layer].held
    }

    /// Total context rows the cache represents, i.e. the absolute position one
    /// past the last one. Equals the target's `next_position`.
    pub fn total_rows(&self) -> usize {
        self.total
    }

    /// Absolute position of `layer`'s oldest retained row.
    ///
    /// For a layer that has wrapped its ring this is *greater* than
    /// `total - rows(layer)` would suggest naively, and it is exactly what the
    /// attention needs: positions are absolute and the mask is on position
    /// distance, so a ring must report where its surviving rows really are.
    pub fn first_position(&self, layer: usize) -> usize {
        self.total.saturating_sub(self.layers[layer].held)
    }

    pub fn keys(&self, layer: usize) -> &[u16] {
        &self.keys[layer]
    }

    pub fn values(&self, layer: usize) -> &[u16] {
        &self.values[layer]
    }

    /// Every layer's keys, in the form [`BlockContext`] takes. Borrowed, not
    /// copied.
    pub fn all_keys(&self) -> &[Vec<u16>] {
        &self.keys
    }

    pub fn all_values(&self) -> &[Vec<u16>] {
        &self.values
    }

    /// Per-layer first position, in the form [`BlockContext`] takes.
    pub fn all_first_positions(&self) -> Vec<usize> {
        (0..self.layers.len())
            .map(|layer| self.first_position(layer))
            .collect()
    }

    /// Append `rows` rows to one layer, evicting from the front if it slides.
    ///
    /// Layer-local: the global length advances in [`Self::commit`].
    pub fn push(
        &mut self,
        layer: usize,
        keys: &[u16],
        values: &[u16],
        rows: usize,
    ) -> Result<(), DflashError> {
        if layer >= self.layers.len() {
            return Err(DflashError::Shape(format!(
                "layer {layer} is outside the cache's {} layers",
                self.layers.len()
            )));
        }
        let expected = rows * self.kv_width;
        if keys.len() != expected || values.len() != expected {
            return Err(DflashError::Shape(format!(
                "layer {layer} push has {}/{} elements for {rows} rows of width {}",
                keys.len(),
                values.len(),
                self.kv_width
            )));
        }
        if rows == 0 {
            return Ok(());
        }
        let width = self.kv_width;
        let capacity = self.layers[layer].capacity;
        self.keys[layer].extend_from_slice(keys);
        self.values[layer].extend_from_slice(values);
        self.layers[layer].held += rows;
        if let Some(capacity) = capacity {
            let held = self.layers[layer].held;
            if held > capacity {
                let excess = (held - capacity) * width;
                self.keys[layer].drain(..excess);
                self.values[layer].drain(..excess);
                self.layers[layer].held = capacity;
            }
        }
        Ok(())
    }

    /// Advance the context length after every layer has been pushed.
    pub fn commit(&mut self, rows: usize) {
        self.total += rows;
    }

    /// Drop the last `rows` rows from every layer (a rejected speculative
    /// tail).
    ///
    /// A layer that has already evicted cannot recover those rows, so it drops
    /// only what it still holds. That is not data loss: a sliding layer only
    /// ever needs its window, and any row it evicted is older than the window
    /// the rewound context re-establishes.
    pub fn rewind(&mut self, rows: usize) -> Result<(), DflashError> {
        if rows == 0 {
            return Ok(());
        }
        if rows > self.total {
            return Err(DflashError::Shape(format!(
                "cannot rewind {rows} rows from a {}-row context",
                self.total
            )));
        }
        for layer in 0..self.layers.len() {
            let drop = rows.min(self.layers[layer].held);
            if drop == 0 {
                continue;
            }
            let elements = drop * self.kv_width;
            let length = self.keys[layer].len() - elements;
            self.keys[layer].truncate(length);
            self.values[layer].truncate(length);
            self.layers[layer].held -= drop;
        }
        self.total -= rows;
        Ok(())
    }

}

/// One draft-and-verify round's outcome (T9/R15).
pub struct DflashRound {
    /// The drafted tokens the target confirmed, in order.
    pub accepted: Vec<u32>,
    /// The target's own greedy token at the first position the drafter got
    /// wrong (or the position past the last row if it got them all right). This
    /// is the token that ends the round and seeds the next one, and it is why a
    /// round always emits at least one token.
    pub next: u32,
    /// How many tokens were drafted (the block's `draft_tokens()`), so a caller
    /// can report acceptance as `accepted.len() as f32 / drafted as f32`.
    pub drafted: usize,
}

impl DflashRound {
    /// Tokens this round emitted: the accepted prefix plus the target's own
    /// correction token.
    pub fn emitted(&self) -> usize {
        self.accepted.len() + 1
    }
}

/// One drafting round's result.
pub struct DflashDraft {
    /// `draft_tokens()` tokens in order.
    pub tokens: Vec<u32>,
    /// The block that produced them, for a caller that wants the hidden rows or
    /// the block's own K/V.
    pub block: DflashBlock,
    /// The drafter's softcapped `[block_size, vocab]` logits, one row per block
    /// row, already computed by `draft_block`. Carried so the verify pass can
    /// build its proposal distributions without re-reading the 1.06 GB target
    /// head a second time.
    pub logits: Vec<u16>,
}

/// Greedy argmax of one row of softcapped logits.
///
/// Ties go to the lowest index, matching the engine's own greedy path so a
/// draft and a verify cannot disagree about a tie.
fn argmax_row(logits: &[u16], vocab: usize, row: usize) -> u32 {
    let base = row * vocab;
    let mut best = 0usize;
    let mut best_value = half::f16::from_bits(logits[base]).to_f32();
    for index in 1..vocab {
        let value = half::f16::from_bits(logits[base + index]).to_f32();
        if value > best_value {
            best_value = value;
            best = index;
        }
    }
    best as u32
}

/// The draft tokens for a block: one greedy token per **masked** row.
///
/// The block is `[anchor, mask x (block_size - 1)]` and `forward_block` puts
/// `anchor_embedding` on row **0**. The anchor predicts a token the target has
/// already emitted, so the drafts are rows `1..=draft_rows` and never row 0.
///
/// This was `0..draft_rows` -- the anchor included and the last masked row
/// dropped -- while both `draft_tokens` and `draft_block`'s own comment said the
/// anchor is dropped. z-lab takes exactly `1..`:
/// `dflash/model.py:249` slices the drafter output with `[:, 1 - verify_size:, :]`
/// (the last `verify_size - 1` rows) and `:268` fills `block_output_ids[:, 1:]`
/// from the argmax of those. Extracted from `draft_block` so this range is
/// testable without loading either checkpoint.
fn select_draft_tokens(logits: &[u16], vocab: usize, draft_rows: usize) -> Vec<u32> {
    (1..=draft_rows)
        .map(|row| argmax_row(logits, vocab, row))
        .collect()
}

// ============================================================================
// `spec/dflash-30tps` L2: the round's distribution machinery
// ============================================================================
//
// Everything below operates on a `BTreeMap<u32, f32>` of `token -> probability`
// for a *sorted* support, which is the only shape the round's arithmetic needs:
// the accept test is a lookup, and the residual is a pointwise difference over a
// union of two sorted supports. It is also the only thing the round has to do
// 262,144 times per draft, so `acceptance` uses two sorted walks instead of a
// hash map.

/// `S(logits)`: softmax at temperature 1, truncated to the `k` largest entries,
/// renormalised.
///
/// This is the sampler the round must be exact for (`dflash_alpha_probe`'s
/// `top_k_distribution`, and z-lab's `_sampling_probs`). Passing the *sampler's*
/// distribution rather than the full softmax is what makes speculative sampling
/// exact for a truncated sampler: `draft x ~ S(q)`, accept with
/// `min(1, S_p(x) / S_q(x))`, resample from the normalised residual, and the
/// emitted marginal is exactly `S(p)`.
fn sampler_distribution(logits: &[u16], vocab: usize, k: usize) -> BTreeMap<u32, f32> {
    let values: Vec<f32> = logits
        .iter()
        .take(vocab)
        .map(|bits| half::f16::from_bits(*bits).to_f32())
        .collect();
    let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return BTreeMap::new();
    }
    let mut order: Vec<u32> = (0..vocab as u32).collect();
    order.sort_by(|a, b| {
        values[*b as usize]
            .partial_cmp(&values[*a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(b))
    });
    order.truncate(k);
    let weights: Vec<f32> = order
        .iter()
        .map(|token| (values[*token as usize] - max).exp())
        .collect();
    let total: f32 = weights.iter().sum();
    if !(total > 0.0) {
        return BTreeMap::new();
    }
    order
        .into_iter()
        .zip(weights)
        .map(|(token, weight)| (token, weight / total))
        .collect()
}

/// Draw one token from a sorted support with `uniform` in `[0, 1)`.
fn sample_sorted(support: &BTreeMap<u32, f32>, sample: &mut impl FnMut() -> f32) -> u32 {
    let threshold = sample() * support.values().sum::<f32>();
    let mut cumulative = 0.0f32;
    let mut last = 0u32;
    for (token, prob) in support {
        cumulative += *prob;
        last = *token;
        if cumulative >= threshold {
            return *token;
        }
    }
    last
}

/// The rejection step of Leviathan/Chen: resample from the normalised residual
/// `max(0, p - q)`, with the rejected proposal's mass folded back to `p` so the
/// result is a proper distribution and the emitted marginal stays exactly `p`.
///
/// `excluded` is the rejected proposal. It is removed from the residual (its
/// probability under the residual is zero because `p(x) <= q(x)` there) and
/// the remainder is renormalised; if everything cancels, fall back to `p` itself,
/// which is the only answer that keeps the marginal exact.
fn sample_residual(
    p: &BTreeMap<u32, f32>,
    q: &BTreeMap<u32, f32>,
    excluded: &u32,
    sample: &mut impl FnMut() -> f32,
) -> u32 {
    let mut residual: BTreeMap<u32, f32> = BTreeMap::new();
    for (token, prob) in p {
        if token == excluded {
            continue;
        }
        let q_prob = q.get(token).copied().unwrap_or(0.0);
        let difference = prob - q_prob;
        if difference > 0.0 {
            residual.insert(*token, difference);
        }
    }
    let total: f32 = residual.values().sum();
    if !(total > 0.0) {
        return sample_sorted(p, sample);
    }
    for prob in residual.values_mut() {
        *prob /= total;
    }
    sample_sorted(&residual, sample)
}

#[cfg(test)]
mod sampler_tests {
    use super::{sample_residual, sampler_distribution, sample_sorted};
    use half::f16;
    use std::collections::BTreeMap;

    fn row(values: &[f32]) -> Vec<u16> {
        values.iter().map(|v| f16::from_f32(*v).to_bits()).collect()
    }

    fn map(entries: &[(u32, f32)]) -> BTreeMap<u32, f32> {
        entries.iter().copied().collect()
    }

    #[test]
    fn the_distribution_is_the_sampler_not_the_softmax() {
        // A row whose 5th entry is far below the top 4: top_k = 4 must drop it
        // and renormalise, not just zero it.
        let logits = row(&[10.0, 9.0, 8.0, 7.0, 0.0, 0.0, 0.0, 0.0]);
        let dist = sampler_distribution(&logits, 8, 4);
        assert_eq!(dist.len(), 4, "top_k truncates the support");
        let total: f32 = dist.values().sum();
        assert!(
            (total - 1.0).abs() < 1e-3,
            "the truncated support is renormalised, got {total}"
        );
        assert!(dist.get(&4).is_none(), "the 5th entry is dropped");
    }

    #[test]
    fn a_degenerate_row_yields_an_empty_distribution() {
        // All -inf (or NaN) must not panic or produce NaN probabilities.
        let logits = vec![f16::NEG_INFINITY.to_bits(); 8];
        let dist = sampler_distribution(&logits, 8, 4);
        assert!(dist.is_empty(), "no finite max means no distribution");
    }

    /// Guards `sampler_distribution` against passing vacuously: a top_k larger
    /// than the vocab must return the whole row.
    #[test]
    fn a_large_top_k_returns_every_token() {
        let logits = row(&[3.0, 1.0, 0.5, 0.25]);
        let dist = sampler_distribution(&logits, 4, 64);
        assert_eq!(dist.len(), 4);
    }

    #[test]
    fn the_residual_is_zero_on_the_rejected_token_and_renormalised() {
        let p = map(&[(1, 0.5), (2, 0.5)]);
        let q = map(&[(1, 0.9), (2, 0.1)]);
        // Rejecting 1 means p(1) <= q(1) there, so the residual must exclude it
        // and put all its mass on 2.
        let mut draws = 0;
        let mut draws_mut = || {
            draws += 1;
            0.99
        };
        let token = sample_residual(&p, &q, &1u32, &mut draws_mut);
        assert_eq!(token, 2, "the residual must not resample the rejected token");
    }

    /// The exactness property the whole round rests on, checked numerically
    /// instead of symbolically: Lamperti. The emitted token's marginal under
    /// speculative sampling is
    ///
    ///   `q(t) * min(1, p(t)/q(t)) + sum_x q(x) * (1 - min(1, p(x)/q(x))) * residual(t | x)`
    ///
    /// and it must equal `p(t)` for every `t`. Both factors on the right are
    /// computed by the *shipped* code (`sample_sorted`, `sample_residual`), not
    /// by re-implementing them here — a test that re-derived the rule would only
    /// prove the test is self-consistent.
    ///
    /// `sample_sorted` is a step function of its uniform, so a stratified grid
    /// over that uniform integrates the residual exactly (up to the step's
    /// position, which the grid resolves to within 1/N).
    #[test]
    fn acceptance_is_distribution_exact() {
        for (p, q) in [
            (
                map(&[(1, 0.25), (2, 0.25), (3, 0.5)]),
                map(&[(1, 0.5), (2, 0.25), (3, 0.25)]),
            ),
            // `q` steeper than `p`: nothing is ever accepted outright, so the
            // marginal comes entirely from the residual.
            (
                map(&[(1, 0.1), (2, 0.2), (3, 0.7)]),
                map(&[(1, 0.7), (2, 0.2), (3, 0.1)]),
            ),
            // A proposal that costs more than `p` allows it: `p/q >= 1` must
            // accept with probability 1, not 1/2.
            (
                map(&[(1, 0.6), (2, 0.4)]),
                map(&[(1, 0.1), (2, 0.9)]),
            ),
        ] {
            let quadrature = 2_000usize;
            let mut tally: BTreeMap<u32, f64> = BTreeMap::new();
            for (proposal, q_prob) in &q {
                let p_prob = p.get(proposal).copied().unwrap_or(0.0);
                let accept = (p_prob / q_prob).min(1.0);
                *tally.entry(*proposal).or_insert(0.0) += *q_prob as f64 * accept as f64;
                let reject = 1.0f32 - accept;
                if reject <= 0.0 {
                    continue;
                }
                // The residual, integrated over its own uniform.
                for index in 0..quadrature {
                    let uniform = (index as f32 + 0.5) / quadrature as f32;
                    let mut state = uniform;
                    let mut next = move || {
                        let value = state;
                        state += 1.0;
                        value
                    };
                    let drawn = sample_residual(&p, &q, proposal, &mut next);
                    *tally.entry(drawn).or_insert(0.0) +=
                        *q_prob as f64 * reject as f64 / quadrature as f64;
                }
            }
            for (token, expected) in &p {
                let observed = tally.get(token).copied().unwrap_or(0.0);
                assert!(
                    (observed - *expected as f64).abs() < 5e-3,
                    "token {token}: marginal {observed} vs p {expected}"
                );
            }
        }
    }

    #[test]
    fn a_single_token_support_always_returns_it() {
        let support = map(&[(7, 1.0)]);
        let mut counter = 0.0f32;
        let mut next = || {
            counter += 0.5;
            counter
        };
        assert_eq!(sample_sorted(&support, &mut next), 7);
    }
}
#[cfg(test)]
mod tests {
    use super::select_draft_tokens;

    /// One row per block position, where row `r`'s argmax is the token id `r`
    /// held in that row's last column. `vocab` is a small placeholder: the
    /// function only indexes `row * vocab`, so the width is irrelevant to the
    /// selection it exercises.
    fn rows(block_rows: usize, vocab: usize, winner: &[u32]) -> Vec<u16> {
        let mut logits = vec![0u16; block_rows * vocab];
        for (row, token) in winner.iter().enumerate() {
            let index = row * vocab + *token as usize;
            logits[index] = half::f16::from_f32(1.0).to_bits();
        }
        logits
    }

    #[test]
    fn draft_tokens_are_the_masked_rows_not_the_anchor() {
        // block_size 16 -> 15 drafts; the anchor is row 0.
        let block_rows = 16usize;
        let vocab = 8usize;
        let draft_rows = block_rows - 1;

        // Give the anchor a distinctive winner (7) that no masked row shares,
        // so including row 0 is observable rather than a coincidence.
        let mut winner: Vec<u32> = (1..=draft_rows as u32).map(|t| t % vocab as u32).collect();
        winner.insert(0, (vocab - 1) as u32);
        let logits = rows(block_rows, vocab, &winner);

        let selected = select_draft_tokens(&logits, vocab, draft_rows);

        assert_eq!(selected.len(), draft_rows, "one draft per masked row");
        assert_eq!(
            selected,
            winner[1..].to_vec(),
            "the drafts must be rows 1..=draft_rows; taking rows 0..draft_rows would \
             put the anchor's token first and drop the last masked row"
        );
        assert_ne!(
            selected.first().copied(),
            Some((vocab - 1) as u32),
            "the anchor's own argmax must never appear as a draft"
        );
    }

    #[test]
    fn the_two_candidate_ranges_are_distinguishable() {
        // Guards the assertion above against passing vacuously: if rows 0.. and
        // 1.. happened to agree on every position, the previous test would
        // prove nothing about which range is used.
        let block_rows = 16usize;
        let vocab = 8usize;
        let draft_rows = block_rows - 1;
        let mut winner: Vec<u32> = (1..=draft_rows as u32).map(|t| t % vocab as u32).collect();
        winner.insert(0, (vocab - 1) as u32);
        let logits = rows(block_rows, vocab, &winner);

        let masked: Vec<u32> = select_draft_tokens(&logits, vocab, draft_rows);
        let with_anchor: Vec<u32> = (0..draft_rows)
            .map(|row| super::argmax_row(&logits, vocab, row))
            .collect();
        assert_ne!(masked, with_anchor, "the two ranges must differ on this input");
    }
}

#[derive(Debug, thiserror::Error)]
pub enum DflashError {
    #[error("kernels: {0}")]
    Kernels(String),
    #[error("missing tensor {0}")]
    Missing(String),
    #[error("{0}")]
    Shape(String),
    #[error("tap count {actual} does not match the config's {expected}")]
    TapCount { actual: usize, expected: usize },
}

impl From<kaplumba_kernels::exl3::KernelError> for DflashError {
    fn from(error: kaplumba_kernels::exl3::KernelError) -> Self {
        DflashError::Kernels(error.to_string())
    }
}

impl DflashModel {
    pub fn new(
        weights: DflashWeights,
        config: DflashConfig,
    ) -> Result<Self, DflashError> {
        let model = Self {
            weights,
            config,
            mask_embedding: Vec::new(),
            linear: kaplumba_kernels::linear::LinearKernels::shared()
                .map_err(|e| DflashError::Kernels(e.to_string()))?,
            attention: kaplumba_kernels::attention::AttentionKernels::new()
                .map_err(|e| DflashError::Kernels(e.to_string()))?,
            elements: kaplumba_kernels::elements::ElementKernels::shared()
                .map_err(|e| DflashError::Kernels(e.to_string()))?,
        };
        model.validate_shapes()?;
        Ok(model)
    }

    pub fn config(&self) -> &DflashConfig {
        &self.config
    }

    /// `(in_features, out_features)` of one of this drafter's linears, for a
    /// caller that wants to build a host-side reference for it.
    pub fn gemv_shape(&self, name: &str) -> (usize, usize) {
        let line = self
            .weights
            .line(name)
            .unwrap_or_else(|_| panic!("{name} is not a drafter linear"));
        (line.in_features, line.out_features)
    }

    /// Bytes of packed weights one linear's GEMM must read: the trellis plus
    /// both sign vectors.
    ///
    /// T12 turns a timing into an effective bandwidth with this, and the
    /// bandwidth is the number that answers R19's actual question: was the
    /// weight read amortised across the block's 16 rows, or paid once per row?
    /// A per-row read shows up as 1/16th of the streaming bandwidth, which
    /// nothing else in the timing would reveal.
    pub fn line_weight_bytes(&self, name: &str) -> Result<usize, DflashError> {
        let line = self
            .weights
            .line(name)
            .map_err(|_| DflashError::Missing(name.to_string()))?;
        let trellis = line.packed_len.unwrap_or(line.packed.len());
        Ok(trellis * 2 + (line.in_features + line.out_features) * 2)
    }

    fn validate_shapes(&self) -> Result<(), DflashError> {
        for layer in 0..self.config.num_layers {
            let checks = [
                (
                    format!("layers.{layer}.self_attn.q_proj"),
                    self.config.hidden_size,
                    self.config.q_heads * self.config.head_dim,
                ),
                (
                    format!("layers.{layer}.self_attn.k_proj"),
                    self.config.hidden_size,
                    self.config.kv_heads * self.config.head_dim,
                ),
                (
                    format!("layers.{layer}.self_attn.v_proj"),
                    self.config.hidden_size,
                    self.config.kv_heads * self.config.head_dim,
                ),
                (
                    format!("layers.{layer}.self_attn.o_proj"),
                    self.config.q_heads * self.config.head_dim,
                    self.config.hidden_size,
                ),
                (
                    format!("layers.{layer}.mlp.gate_proj"),
                    self.config.hidden_size,
                    self.config.intermediate,
                ),
                (
                    format!("layers.{layer}.mlp.up_proj"),
                    self.config.hidden_size,
                    self.config.intermediate,
                ),
                (
                    format!("layers.{layer}.mlp.down_proj"),
                    self.config.intermediate,
                    self.config.hidden_size,
                ),
            ];
            for (name, in_features, out_features) in checks {
                let line = self
                    .weights
                    .line(&name)
                    .map_err(|_| DflashError::Missing(name.clone()))?;
                if (line.in_features, line.out_features) != (in_features, out_features) {
                    return Err(DflashError::Shape(format!(
                        "{name} is {}x{}, the config says {in_features}x{out_features}",
                        line.in_features, line.out_features
                    )));
                }
            }
            for (name, width) in [
                (format!("layers.{layer}.input_layernorm"), self.config.hidden_size),
                (
                    format!("layers.{layer}.post_attention_layernorm"),
                    self.config.hidden_size,
                ),
                (
                    format!("layers.{layer}.self_attn.q_norm"),
                    self.config.head_dim,
                ),
                (
                    format!("layers.{layer}.self_attn.k_norm"),
                    self.config.head_dim,
                ),
            ] {
                if self.weights.norm(&name).map(|values| values.len()).unwrap_or(0) != width {
                    return Err(DflashError::Shape(format!(
                        "{name} does not have {width} values"
                    )));
                }
            }
        }
        if self.weights.norm("hidden_norm").map(|values| values.len()).unwrap_or(0)
            != self.config.hidden_size
            || self.weights.norm("norm").map(|values| values.len()).unwrap_or(0)
                != self.config.hidden_size
        {
            return Err(DflashError::Shape(
                "the encoder and final norms must be hidden_size wide".to_string(),
            ));
        }
        Ok(())
    }

    /// `out = gemm(linear, input)`, `input` being `rows * in_features` rows of
    /// fp16. Uses the engine's EXL3 prefill kernel, so `rows` is inferred from
    /// the input length.
    ///
    /// Public because the golden-reference harness (`dflash_golden`) needs to
    /// exercise the same entry point the block forward uses, and because a
    /// private copy in the test would not be the thing under test.
    pub fn gemm(
        &self,
        name: &str,
        input: &[u16],
    ) -> Result<Vec<u16>, DflashError> {
        let line = self
            .weights
            .line(name)
            .map_err(|_| DflashError::Missing(name.to_string()))?;
        if !input.len().is_multiple_of(line.in_features) {
            return Err(DflashError::Shape(format!(
                "{name} input has {} elements, not a multiple of {}",
                input.len(),
                line.in_features
            )));
        }
        let metal_buffer = line.metal_buffer.as_ref().ok_or_else(|| {
            DflashError::Kernels(format!("{name} has no GPU-resident packed buffer"))
        })?;
        Ok(self.linear.gemm_chunk_buffer(LinearBufferRequest {
            packed: metal_buffer,
            packed_offset: line.metal_offset,
            input,
            input_signs: &line.input_signs,
            output_signs: &line.output_signs,
            in_features: line.in_features,
            out_features: line.out_features,
            rate: line.rate,
            codebook: line.codebook,
        })?)
    }

    fn rms_norm(
        &self,
        name: &str,
        input: &[u16],
        rows: usize,
        width: usize,
    ) -> Result<Vec<u16>, DflashError> {
        let weight = self
            .weights
            .norm(name)
            .map_err(|_| DflashError::Missing(name.to_string()))?;
        if input.len() != rows * width {
            return Err(DflashError::Shape(format!(
                "{name} norm input has {} elements, expected {}",
                input.len(),
                rows * width
            )));
        }
        Ok(self.elements.rms_norm(
            input,
            weight,
            rows,
            width,
            self.config.rms_norm_eps,
        )?)
    }

    fn residual(&self, left: &[u16], right: &[u16]) -> Result<Vec<u16>, DflashError> {
        if left.len() != right.len() {
            return Err(DflashError::Shape(
                "residual add needs equal lengths".to_string(),
            ));
        }
        Ok(self.elements.residual_add(left, right)?)
    }

    /// `gemm` with an **f32** output, for the two projections that join the
    /// drafter's f32 residual stream.
    ///
    /// Only `o_proj` and `down_proj` need this. exllamav3 gives the attention
    /// block and the GatedMLP `out_dtype = torch.float` while leaving their
    /// inputs and the gated activation at half
    /// (`reference/exllamav3/exllamav3/architecture/dflash.py:277,308`), and
    /// that is exactly the split used here: everything upstream of those two
    /// projections stays f16, so the cheap path is unchanged.
    fn gemm_f32(&self, name: &str, input: &[u16]) -> Result<Vec<f32>, DflashError> {
        let line = self
            .weights
            .line(name)
            .map_err(|_| DflashError::Missing(name.to_string()))?;
        if !input.len().is_multiple_of(line.in_features) {
            return Err(DflashError::Shape(format!(
                "{name} input has {} elements, not a multiple of {}",
                input.len(),
                line.in_features
            )));
        }
        let metal_buffer = line.metal_buffer.as_ref().ok_or_else(|| {
            DflashError::Kernels(format!("{name} has no GPU-resident packed buffer"))
        })?;
        Ok(self.linear.gemm_chunk_buffer_f32(LinearBufferRequest {
            packed: metal_buffer,
            packed_offset: line.metal_offset,
            input,
            input_signs: &line.input_signs,
            output_signs: &line.output_signs,
            in_features: line.in_features,
            out_features: line.out_features,
            rate: line.rate,
            codebook: line.codebook,
        })?)
    }

    /// `residual` in f32. Separate rather than generic because the point is that
    /// this one does not narrow: by layer 0's MLP the stream is ~1.5e5, past
    /// fp16's 65504.
    fn residual_f32(&self, left: &[f32], right: &[f32]) -> Result<Vec<f32>, DflashError> {
        if left.len() != right.len() {
            return Err(DflashError::Shape(
                "residual add needs equal lengths".to_string(),
            ));
        }
        Ok(self.elements.residual_add_f32(left, right)?)
    }

    /// `rms_norm` reading the f32 residual stream and writing the f16 that the
    /// EXL3 gemm consumes.
    fn rms_norm_f32_in(
        &self,
        name: &str,
        input: &[f32],
        rows: usize,
        width: usize,
    ) -> Result<Vec<u16>, DflashError> {
        let weight = self
            .weights
            .norm(name)
            .map_err(|_| DflashError::Missing(name.to_string()))?;
        if input.len() != rows * width {
            return Err(DflashError::Shape(format!(
                "{name} norm input has {} elements, expected {}",
                input.len(),
                rows * width
            )));
        }
        Ok(self.elements.rms_norm_f32_in(
            input,
            weight,
            rows,
            width,
            self.config.rms_norm_eps,
        )?)
    }

    /// R4/R7: fuse the target's tap rows into the drafter's hidden width.
    ///
    /// `taps` is `target_layer_ids.len()` rows-major buffers, each
    /// `[n_ctx, hidden_size]`, in the config's tap order. z-lab concatenates
    /// along the feature axis and projects once
    /// (`DFlashDraftModel.forward`, `dflash/model.py:582`), and that order is
    /// load-bearing: the fc weights were trained on that exact concatenation.
    pub fn fuse_context(
        &self,
        taps: &[&[u16]],
        n_ctx: usize,
    ) -> Result<Vec<u16>, DflashError> {
        let hidden = self.config.hidden_size;
        if taps.len() != self.config.num_taps() {
            return Err(DflashError::TapCount {
                actual: taps.len(),
                expected: self.config.num_taps(),
            });
        }
        let expected = n_ctx * hidden;
        if taps.iter().any(|tap| tap.len() != expected) {
            return Err(DflashError::Shape(format!(
                "every tap needs {expected} elements for {n_ctx} rows"
            )));
        }
        // Row-major interleave: [row][tap][feature].
        let width = hidden * taps.len();
        let mut concat = vec![0_u16; n_ctx * width];
        for row in 0..n_ctx {
            for (tap, values) in taps.iter().enumerate() {
                let source = row * hidden;
                let target = row * width + tap * hidden;
                concat[target..target + hidden].copy_from_slice(&values[source..source + hidden]);
            }
        }
        let projected = self.gemm("fc", &concat)?;
        self.rms_norm("hidden_norm", &projected, n_ctx, hidden)
    }

    /// R7/T3: run the 16-row block through all five layers.
    pub fn forward_block(
        &self,
        context: &BlockContext<'_>,
    ) -> Result<DflashBlock, DflashError> {
        let config = &self.config;
        let rows = config.block_size;
        let hidden = config.hidden_size;
        let head_dim = config.head_dim;
        if context.anchor_embedding.len() != hidden {
            return Err(DflashError::Shape(format!(
                "anchor embedding has {} elements, expected {hidden}",
                context.anchor_embedding.len()
            )));
        }
        if context.ctx_keys.len() != config.num_layers
            || context.ctx_values.len() != config.num_layers
        {
            return Err(DflashError::Shape(format!(
                "the context needs {} layers of K/V, got {}/{}",
                config.num_layers,
                context.ctx_keys.len(),
                context.ctx_values.len()
            )));
        }
        if context.ctx_first.len() != config.num_layers {
            return Err(DflashError::Shape(format!(
                "ctx_first needs one position per layer ({})",
                config.num_layers
            )));
        }
        let kv_width = config.kv_heads * head_dim;
        // Row counts differ per layer (see `BlockContext`), so each layer is
        // validated against its own width rather than a shared `n_ctx`.
        let mut ctx_rows = Vec::with_capacity(config.num_layers);
        for layer in 0..config.num_layers {
            let keys = context.ctx_keys[layer].len();
            let values = context.ctx_values[layer].len();
            if keys % kv_width != 0 || values % kv_width != 0 || keys != values {
                return Err(DflashError::Shape(format!(
                    "layer {layer} context K/V has {keys}/{values} elements, not a \
                     matching multiple of {kv_width}"
                )));
            }
            let rows_here = keys / kv_width;
            // A sliding layer must not be handed more rows than its window.
            if config.layer_is_sliding[layer]
                && config.sliding_window > 0
                && rows_here > config.sliding_window
            {
                return Err(DflashError::Shape(format!(
                    "layer {layer} is sliding but was handed {rows_here} rows for a \
                     window of {}",
                    config.sliding_window
                )));
            }
            ctx_rows.push(rows_here);
        }

        if self.mask_embedding.is_empty() {
            return Err(DflashError::Missing(
                "the mask token's embedding row has not been supplied".to_string(),
            ));
        }
        // The block is the anchor's raw embedding followed by
        // `block_size - 1` copies of the mask token's raw embedding. z-lab
        // takes it from the target's embedding table too
        // (`_raw_input_embeddings`, `dflash/model.py:135`) with NO sqrt(hidden)
        // scale, and the caller supplies the already-looked-up anchor row.
        // f32 from here on: the residual stream exceeds fp16's range within this
        // block (see `gemm_f32`). The raw embedding rows are f16 weights, so they
        // are widened once, on entry.
        let mut hidden_state = vec![0.0_f32; rows * hidden];
        let mask_row = self.mask_embedding.as_slice();
        for row in 0..rows {
            let source = if row == 0 {
                context.anchor_embedding
            } else {
                mask_row
            };
            let target = row * hidden;
            for (offset, bits) in source.iter().enumerate() {
                hidden_state[target + offset] = half::f16::from_bits(*bits).to_f32();
            }
        }

        // Positions are built per layer inside the layer loop below: a sliding
        // layer's first retained row is not `position - n_ctx` once its ring has
        // wrapped, so the two layers genuinely differ.

        let q_width = config.q_heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut block_keys = Vec::with_capacity(config.num_layers);
        let mut block_values = Vec::with_capacity(config.num_layers);

        for layer in 0..config.num_layers {
            let prefix = format!("layers.{layer}");
            let sliding = config.layer_is_sliding[layer];
            let n_ctx = ctx_rows[layer];
            let mut positions = Vec::with_capacity(n_ctx + rows);
            for index in 0..n_ctx {
                positions.push((context.ctx_first[layer] + index) as u32);
            }
            for index in 0..rows {
                positions.push((context.position + index) as u32);
            }

            let normed = self.rms_norm_f32_in(
                &format!("{prefix}.input_layernorm"),
                &hidden_state,
                rows,
                hidden,
            )?;

            trace_stage(&format!("layer{layer}.input_norm"), &normed);
            let q = self.gemm(&format!("{prefix}.self_attn.q_proj"), &normed)?;
            trace_stage(&format!("layer{layer}.q_raw"), &q);
            let q = self.rms_norm(
                &format!("{prefix}.self_attn.q_norm"),
                &q,
                rows * config.q_heads,
                head_dim,
            )?;
            let q = self.elements.rope(RopeRequest {
                input: &q,
                tokens: rows,
                heads: config.q_heads,
                head_dim,
                offset: context.position,
                theta: config.rope_theta,
                partial_factor: 1.0,
                proportional: false,
            })?;

            trace_stage(&format!("layer{layer}.q_rope"), &q);
            let k_block = self.gemm(&format!("{prefix}.self_attn.k_proj"), &normed)?;
            trace_stage(&format!("layer{layer}.k_raw"), &k_block);
            let k_block = self.rms_norm(
                &format!("{prefix}.self_attn.k_norm"),
                &k_block,
                rows * config.kv_heads,
                head_dim,
            )?;
            let k_block = self.elements.rope(RopeRequest {
                input: &k_block,
                tokens: rows,
                heads: config.kv_heads,
                head_dim,
                offset: context.position,
                theta: config.rope_theta,
                partial_factor: 1.0,
                proportional: false,
            })?;
            let v_block = self.gemm(&format!("{prefix}.self_attn.v_proj"), &normed)?;

            let attended = self.attention.dflash_block_attention(DflashBlockRequest {
                queries: &q,
                ctx_keys: &context.ctx_keys[layer],
                ctx_values: &context.ctx_values[layer],
                block_keys: &k_block,
                block_values: &v_block,
                key_positions: &positions,
                rows,
                n_ctx,
                q_heads: config.q_heads,
                kv_heads: config.kv_heads,
                head_dim,
                scale,
                window: if sliding { config.sliding_window } else { 0 },
                bidirectional: !sliding,
            })?;

            trace_stage(&format!("layer{layer}.attended"), &attended);
            let projected = self.gemm_f32(&format!("{prefix}.self_attn.o_proj"), &attended)?;
            trace_stage_f32(&format!("layer{layer}.attn_out"), &projected);
            hidden_state = self.residual_f32(&hidden_state, &projected)?;
            trace_stage_f32(&format!("layer{layer}.residual1"), &hidden_state);

            let mlp_normed = self.rms_norm_f32_in(
                &format!("{prefix}.post_attention_layernorm"),
                &hidden_state,
                rows,
                hidden,
            )?;
            let gate = self.gemm(&format!("{prefix}.mlp.gate_proj"), &mlp_normed)?;
            let up = self.gemm(&format!("{prefix}.mlp.up_proj"), &mlp_normed)?;
            let activated = self.elements.gated_silu(&gate, &up)?;
            trace_stage(&format!("layer{layer}.activated"), &activated);
            let down = self.gemm_f32(&format!("{prefix}.mlp.down_proj"), &activated)?;
            trace_stage_f32(&format!("layer{layer}.mlp_out"), &down);
            hidden_state = self.residual_f32(&hidden_state, &down)?;
            trace_stage_f32(&format!("layer{layer}.residual2"), &hidden_state);

            block_keys.push(k_block);
            block_values.push(v_block);
        }
        let _ = q_width;

        // Back to f16 for the caller: this is the drafter's final norm, and
        // exllamav3 narrows at the same point (`.../dflash.py:316`).
        let normed = self.rms_norm_f32_in("norm", &hidden_state, rows, hidden)?;
        Ok(DflashBlock {
            hidden: normed,
            block_keys,
            block_values,
        })
    }

    /// T6's other half: project the fused context feature into one layer's K/V.
    ///
    /// This is the "KV injection" of the DFlash paper: the target's tap
    /// features, not the drafter's own token embeddings, supply the history's
    /// K and V. z-lab computes these inline inside
    /// `Qwen3DFlashAttention.forward` (`dflash/model.py:386-397`) as
    /// `k_proj(target_hidden)`; this side computes them once per accepted
    /// position and writes them to the drafter's cache.
    ///
    /// `positions` are the absolute positions of the context rows, which end at
    /// `context_position - 1`.
    pub fn project_context_kv(
        &self,
        layer: usize,
        fused: &[u16],
        n_rows: usize,
        positions: &[u32],
        context_position: usize,
    ) -> Result<(Vec<u16>, Vec<u16>), DflashError> {
        let config = &self.config;
        let head_dim = config.head_dim;
        let kv_width = config.kv_heads * head_dim;
        if layer >= config.num_layers {
            return Err(DflashError::Shape(format!(
                "layer {layer} is outside the drafter's {} layers",
                config.num_layers
            )));
        }
        if fused.len() != n_rows * config.hidden_size {
            return Err(DflashError::Shape(format!(
                "fused context has {} elements, expected {}",
                fused.len(),
                n_rows * config.hidden_size
            )));
        }
        if positions.len() != n_rows {
            return Err(DflashError::Shape(
                "one position per context row".to_string(),
            ));
        }
        let prefix = format!("layers.{layer}");
        let k = self.gemm(&format!("{prefix}.self_attn.k_proj"), fused)?;
        let k = self.rms_norm(
            &format!("{prefix}.self_attn.k_norm"),
            &k,
            n_rows * config.kv_heads,
            head_dim,
        )?;
        let k = self.elements.rope(RopeRequest {
            input: &k,
            tokens: n_rows,
            heads: config.kv_heads,
            head_dim,
            offset: positions[0] as usize,
            theta: config.rope_theta,
            partial_factor: 1.0,
            proportional: false,
        })?;
        let v = self.gemm(&format!("{prefix}.self_attn.v_proj"), fused)?;
        let _ = (kv_width, context_position);
        Ok((k, v))
    }

    /// T6/R6: fuse one chunk of target taps and append the resulting K/V to
    /// `cache`.
    ///
    /// `position` is the absolute position of the row **after** the chunk — the
    /// position the next block will start at — so the chunk's own rows occupy
    /// `[position - rows, position)`. That convention is what makes an
    /// incremental chunked prefill and a single whole-prompt call produce
    /// byte-identical caches, which is R6's invariant.
    ///
    /// `taps` is one `rows * hidden_size` F16 buffer per target tap, in
    /// `target_layer_ids` order (the order `fc` was trained on).
    pub fn append_context(
        &self,
        taps: &[&[u16]],
        rows: usize,
        position: usize,
        cache: &mut DflashKvCache,
    ) -> Result<(), DflashError> {
        if rows == 0 {
            return Ok(());
        }
        if position < rows {
            return Err(DflashError::Shape(format!(
                "a {rows}-row chunk at position {position} would start before 0"
            )));
        }
        let fused = self.fuse_context(taps, rows)?;
        let start = (position - rows) as u32;
        let positions: Vec<u32> = (0..rows).map(|index| start + index as u32).collect();
        for layer in 0..self.config.num_layers {
            let (k, v) = self.project_context_kv(layer, &fused, rows, &positions, position)?;
            cache.push(layer, &k, &v, rows)?;
        }
        cache.commit(rows);
        Ok(())
    }

    /// R13: drop the last `rows` context rows after a rejected speculative
    /// tail, so the next round drafts from the verified prefix.
    pub fn rewind_context(
        &self,
        rows: usize,
        cache: &mut DflashKvCache,
    ) -> Result<(), DflashError> {
        cache.rewind(rows)
    }

    /// T9 / `spec/dflash-30tps` L2: one full speculative round.
    ///
    /// # The shape, and why it is not the obvious one
    ///
    /// The naive round feeds `[anchor] ++ drafts` to the target at
    /// `position .. position + n - 1`. That is **wrong**. The target's cache
    /// already holds the anchor's K/V row — whatever produced `previous_logits`
    /// put it there — so re-feeding the anchor at the same position appends a
    /// *second*, identical key row beneath it. Causal attention then attends to
    /// that key twice, which does not cancel: it changes the softmax denominator
    /// and therefore the logits. The "target" being verified against is a subtly
    /// different model, and acceptance is measured against corrupted logits —
    /// silently, because nothing fails.
    ///
    /// So the anchor is never re-fed. Instead:
    ///
    ///   * `draft.tokens[0..n]` are fed at `position .. position + n - 1`,
    ///   * `L[0]` is `previous_logits` — the target's own row for the position
    ///     *before* this round's first draft, which the caller owns because each
    ///     round consumes the logits its predecessor produced,
    ///   * `L[k]` for `k >= 1` is the batched verify's row `k - 1`.
    ///
    /// `L[k]` is the target's prediction for the position `draft.tokens[k]`
    /// lands at, so it is the row that decides whether that draft is accepted.
    ///
    /// # Verification: speculative sampling, not argmax
    ///
    /// `spec/dflash-30tps` was amended on 2026-10-12 to demand coherent output at
    /// temperature 1.0 / top_k 64. **Argmax verification is not a valid verifier
    /// for that**: matching argmax makes the whole generation greedy, which is a
    /// different distribution from the one asked for. So the round uses
    /// Leviathan et al. 2023 / Chen et al. 2023 speculative sampling over the
    /// *sampler's* own distributions `S(p)` and `S(q)` (here: softmax, top-k
    /// truncation, renormalise — the same sampler the caller will apply):
    ///
    ///   draft `x ~ S(q)`; accept with `min(1, S_p(x) / S_q(x))`; on rejection
    ///   resample from the normalised residual `max(0, S_p - S_q)`.
    ///
    /// The emitted token's marginal is then exactly `S(p)`, so the speculative
    /// stream is distributionally identical to sampling from the target
    /// directly, at ~n tokens per round. `greedy` selects the argmax verifier
    /// instead — the same loop with a degenerate proposal — for the bit-identical
    /// equivalence test (R6 / AC3).
    ///
    /// # Retraction
    ///
    /// The target's cache advanced `n` rows but only `accepted.len() + 1` of them
    /// are correct after a rejection, so the difference is retracted. The
    /// drafter's own cache is the caller's business: it knows the tap schedule,
    /// and redrafting from the verified prefix is what it should do.
    ///
    /// `sample` draws uniforms in `[0, 1)`; the engine's sampler supplies it so
    /// the round and the single-token path cannot disagree about an RNG.
    ///
    /// Returns the drafted-and-verified round. `Ok(None)` when the drafter
    /// produced nothing at all (a cache that has not seen enough context yet),
    /// which the caller should treat as "take a plain target step".
    pub fn draft_and_verify(
        &mut self,
        target: &mut crate::model::TextModel,
        drafter_cache: &mut DflashKvCache,
        anchor_token: u32,
        previous_logits: &[u16],
        position: usize,
        greedy: bool,
        top_k: usize,
        mut sample: impl FnMut() -> f32,
    ) -> Result<DflashRound, DflashError> {
        let vocab = target.head().out_features;
        if previous_logits.len() < vocab {
            return Err(DflashError::Shape(format!(
                "previous_logits holds {} of {vocab}",
                previous_logits.len()
            )));
        }
        let head = target.head();
        let anchor_embedding = target.raw_embedding(&[anchor_token]);
        let draft = self.draft_block(&anchor_embedding, position, drafter_cache, head)?;
        let n = draft.tokens.len();
        if n == 0 {
            return Err(DflashError::Shape("the drafter drafted no tokens".to_string()));
        }

        // The target's row for each draft position. `L[0]` is the caller's
        // previous row and is NOT re-derived (see the header); `L[k>=1]` comes
        // off the batched verify, whose row `k-1` is the target's prediction at
        // the position `draft.tokens[k]` was fed to.
        let verify_positions: Vec<u32> =
            (0..n as u32).map(|index| position as u32 + index).collect();
        let verify_rows = target
            .forward_verify(&draft.tokens, &verify_positions)
            .map_err(|error| DflashError::Kernels(error.to_string()))?;
        if verify_rows.len() < n * vocab {
            return Err(DflashError::Shape(format!(
                "the target returned {} rows of {vocab} for {n} drafts",
                verify_rows.len() / vocab
            )));
        }

        let mut accepted: Vec<u32> = Vec::with_capacity(n);
        let mut next = 0u32;
        for k in 0..n {
            let candidate = draft.tokens[k];
            let target_row: &[u16] = if k == 0 {
                &previous_logits[..vocab]
            } else {
                let base = (k - 1) * vocab;
                &verify_rows[base..base + vocab]
            };
            if greedy {
                let target_token = argmax_row(target_row, vocab, 0);
                if target_token == candidate {
                    accepted.push(candidate);
                } else {
                    next = target_token;
                    break;
                }
            } else {
                let p = sampler_distribution(target_row, vocab, top_k);
                // The proposal for draft k is the drafter's own row k+1: row 0
                // is the anchor and predicts a token the target already
                // emitted, so it is not a proposal for anything.
                let q = if k + 1 < draft.logits.len() / vocab {
                    let base = (k + 1) * vocab;
                    sampler_distribution(&draft.logits[base..base + vocab], vocab, top_k)
                } else {
                    // The last draft has no row of its own in the block. Propose
                    // from the target's row instead: the accept test is then
                    // min(1, p/p) = 1 and the round degenerates to plain
                    // sampling for that one position, which is exact.
                    p.clone()
                };
                let p_candidate = p.get(&candidate).copied().unwrap_or(0.0);
                let q_candidate = q.get(&candidate).copied().unwrap_or(0.0);
                let accepted_now = q_candidate <= 0.0
                    || p_candidate >= q_candidate
                    || (p_candidate / q_candidate) > sample();
                if accepted_now {
                    accepted.push(candidate);
                } else {
                    next = sample_residual(&p, &q, &candidate, &mut sample);
                    break;
                }
            }
        }
        if accepted.len() == n {
            // Every draft landed. The round's correction token is then the
            // target's prediction for the position just past the last accepted
            // draft, which is the last verify row.
            let base = (n - 1) * vocab;
            let last_row = &verify_rows[base..base + vocab];
            next = if greedy {
                argmax_row(last_row, vocab, 0)
            } else {
                let p = sampler_distribution(last_row, vocab, top_k);
                sample_sorted(&p, &mut sample)
            };
        }

        // Cache accounting: the verify advanced `n` rows, `accepted.len() + 1`
        // of which are correct. Retract the rest so the next round drafts are
        // checked against the prefix the target actually emitted.
        let correct = accepted.len() + 1;
        if n > correct {
            target.retract_gpu_caches(n - correct);
        }

        Ok(DflashRound {
            accepted,
            next,
            drafted: n,
        })
    }

    /// R7/T9: draft one block from the cache.
    ///
    /// `anchor_embedding` is the *raw* target embedding row for the token at
    /// `position - 1` (no `sqrt(hidden)`; R7, and Q1 may flip this). The block is
    /// `[anchor, mask x (block_size - 1)]` and its rows are dropped: the anchor
    /// predicts a token the target has already emitted.
    ///
    /// `head` is the **target's** tied output head — the drafter has none
    /// (`tie_word_embeddings` is true and the vocabularies match), which is why
    /// this takes it as an argument rather than owning it. That is a deviation
    /// from `plan.md`'s signature, which omitted it; the plan also assumed a
    /// single `n_ctx`, which the sliding cache makes impossible.
    pub fn draft_block(
        &self,
        anchor_embedding: &[u16],
        position: usize,
        cache: &DflashKvCache,
        head: &crate::model::QuantizedLinear,
    ) -> Result<DflashDraft, DflashError> {
        let config = &self.config;
        let rows = config.block_size;
        let vocab = head.out_features;
        let first = cache.all_first_positions();
        let block = self.forward_block(&BlockContext {
            anchor_embedding,
            ctx_keys: cache.all_keys(),
            ctx_values: cache.all_values(),
            ctx_first: &first,
            position,
        })?;
        // Only the `block_size - 1` masked rows are new tokens; the anchor's own
        // logits predict a token the target has already produced.
        let logits = self.draft_logits(&block.hidden, head, config.final_logit_softcapping, rows)?;
        let draft_rows = config.draft_tokens();
        if logits.len() < draft_rows * vocab {
            return Err(DflashError::Shape(format!(
                "the head returned {} logits for {rows} rows of {vocab}",
                logits.len()
            )));
        }
        // Rows `1..=draft_rows`, NOT `0..draft_rows`. The block is
        // `[anchor, mask x (block_size - 1)]` (see `forward_block`, which puts
        // `anchor_embedding` on row 0), so row 0 is the anchor and only rows 1..
        // are masked predictions. z-lab takes exactly this range:
        // `dflash/model.py:249` slices the drafter's output with
        // `[:, 1 - verify_size :, :]` (the last `verify_size - 1` rows) and
        // `:268` fills `block_output_ids[:, 1:]` from it. Selecting `0..` instead
        // made the first draft the anchor's argmax -- a token the target had
        // already emitted -- and dropped the last masked row. No test covered
        // this; `dflash_golden` only exercises `forward_block`.
        let tokens = select_draft_tokens(&logits, vocab, draft_rows);
        Ok(DflashDraft { tokens, block, logits })
    }

    /// R11: apply the *target's* tied head and the drafter's softcap, exactly as
    /// z-lab's `compute_logits` does (`dflash/model.py:588`). The drafter has no
    /// head of its own -- `tie_word_embeddings` is true and its
    /// `vocab_size` matches the target's.
    pub fn draft_logits(
        &self,
        hidden: &[u16],
        head: &crate::model::QuantizedLinear,
        softcap: f32,
        rows: usize,
    ) -> Result<Vec<u16>, DflashError> {
        if !hidden.len().is_multiple_of(self.config.hidden_size) {
            return Err(DflashError::Shape(
                "hidden rows are not a multiple of hidden_size".to_string(),
            ));
        }
        if softcap <= 0.0 {
            return Err(DflashError::Shape(
                "the drafter's final_logit_softcapping must be positive".to_string(),
            ));
        }
        let _ = rows;
        let metal_buffer = head.metal_buffer.as_ref().ok_or_else(|| {
            DflashError::Kernels("the target head has no GPU-resident packed buffer".to_string())
        })?;
        // The target's head is offloaded into the whole second shard, so tensors
        // follow its trellis: `packed.length() - packed_offset` is 2 208 470 208 B
        // for a 1 056 964 608 B trellis, and `gemm_chunk_buffer`'s equality check
        // refuses that. `packed_len` is the real word count, so use the
        // length-aware entry point. Measured on the real checkpoint:
        //
        //   in=5376 out=262144 rate=Integer(6) packed_len=Some(528482304)
        //   metal_buffer=Some(7529545728) metal_offset=5321075520
        //   528482304 * 2 = 1056964608 == (5376/16) * (262144/16) * 96 * 2
        let packed_bytes = match head.packed_len {
            Some(words) => (words as u64) * std::mem::size_of::<u16>() as u64,
            None => metal_buffer.length().saturating_sub(head.metal_offset),
        };
        let logits = self.linear.gemm_chunk_buffer_sliced(
            LinearBufferRequest {
                packed: metal_buffer,
                packed_offset: head.metal_offset,
                input: hidden,
                input_signs: &head.input_signs,
                output_signs: &head.output_signs,
                in_features: head.in_features,
                out_features: head.out_features,
                rate: head.rate,
                codebook: head.codebook,
            },
            packed_bytes,
        )?;
        Ok(self.elements.softcap(&logits, softcap)?)
    }
}

impl DflashModel {
    /// The mask token's raw (unscaled) embedding row, `hidden_size` fp16.
    pub fn set_mask_embedding(&mut self, row: &[u16]) -> Result<(), DflashError> {
        if row.len() != self.config.hidden_size {
            return Err(DflashError::Shape(format!(
                "mask embedding has {} elements, expected {}",
                row.len(),
                self.config.hidden_size
            )));
        }
        self.mask_embedding = row.to_vec();
        Ok(())
    }
}
