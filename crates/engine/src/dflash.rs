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
