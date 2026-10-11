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
    pub ctx_keys: &'a [Vec<u16>],
    pub ctx_values: &'a [Vec<u16>],
    pub n_ctx: usize,
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
        let kv_width = config.kv_heads * head_dim;
        for layer in 0..config.num_layers {
            if context.ctx_keys[layer].len() != context.n_ctx * kv_width
                || context.ctx_values[layer].len() != context.n_ctx * kv_width
            {
                return Err(DflashError::Shape(format!(
                    "layer {layer} context K/V has {}/{} elements, expected {}",
                    context.ctx_keys[layer].len(),
                    context.ctx_values[layer].len(),
                    context.n_ctx * kv_width
                )));
            }
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
        let mut hidden_state = vec![0_u16; rows * hidden];
        let mask_row = self.mask_embedding.as_slice();
        for row in 0..rows {
            let source = if row == 0 {
                context.anchor_embedding
            } else {
                mask_row
            };
            let target = row * hidden;
            hidden_state[target..target + hidden].copy_from_slice(source);
        }

        // Positions, absolute. Context rows occupy `[position - n_ctx, position)`.
        let mut positions = Vec::with_capacity(context.n_ctx + rows);
        let start = context.position.saturating_sub(context.n_ctx);
        for index in start..context.position + rows {
            positions.push(index as u32);
        }

        let q_width = config.q_heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut block_keys = Vec::with_capacity(config.num_layers);
        let mut block_values = Vec::with_capacity(config.num_layers);

        for layer in 0..config.num_layers {
            let prefix = format!("layers.{layer}");
            let sliding = config.layer_is_sliding[layer];

            let normed = self.rms_norm(
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
                n_ctx: context.n_ctx,
                q_heads: config.q_heads,
                kv_heads: config.kv_heads,
                head_dim,
                scale,
                window: if sliding { config.sliding_window } else { 0 },
                bidirectional: !sliding,
            })?;

            trace_stage(&format!("layer{layer}.attended"), &attended);
            let projected = self.gemm(&format!("{prefix}.self_attn.o_proj"), &attended)?;
            trace_stage(&format!("layer{layer}.attn_out"), &projected);
            hidden_state = self.residual(&hidden_state, &projected)?;
            trace_stage(&format!("layer{layer}.residual1"), &hidden_state);

            let mlp_normed = self.rms_norm(
                &format!("{prefix}.post_attention_layernorm"),
                &hidden_state,
                rows,
                hidden,
            )?;
            let gate = self.gemm(&format!("{prefix}.mlp.gate_proj"), &mlp_normed)?;
            let up = self.gemm(&format!("{prefix}.mlp.up_proj"), &mlp_normed)?;
            let activated = self.elements.gated_silu(&gate, &up)?;
            trace_stage(&format!("layer{layer}.activated"), &activated);
            let down = self.gemm(&format!("{prefix}.mlp.down_proj"), &activated)?;
            trace_stage(&format!("layer{layer}.mlp_out"), &down);
            hidden_state = self.residual(&hidden_state, &down)?;
            trace_stage(&format!("layer{layer}.residual2"), &hidden_state);

            block_keys.push(k_block);
            block_values.push(v_block);
        }
        let _ = q_width;

        let normed = self.rms_norm("norm", &hidden_state, rows, hidden)?;
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
        let logits = self.linear.gemm_chunk_buffer(LinearBufferRequest {
            packed: metal_buffer,
            packed_offset: head.metal_offset,
            input: hidden,
            input_signs: &head.input_signs,
            output_signs: &head.output_signs,
            in_features: head.in_features,
            out_features: head.out_features,
            rate: head.rate,
            codebook: head.codebook,
        })?;
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
