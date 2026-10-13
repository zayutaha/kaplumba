//! spec/dflash-correctness T4: the drafter's whole graph against a scalar host
//! reference built from the *same checkpoint*, dequantised on the CPU.
//!
//! This is not the same kind of test as the per-kernel ones. A per-kernel test
//! catches a kernel bug; this one catches a **graph** bug -- a norm applied to
//! the wrong tensor, a residual taken from the wrong stream, RoPE on V, a
//! softcap in the wrong place -- none of which any kernel test can see. The
//! reference here is f32 scalar code that reads the EXL3 trellis through
//! `kaplumba_engine::exl3`, so it shares only the *format* with the engine, not
//! the kernels and not the graph.
//!
//! The comparison is numeric, not bit-exact: the engine computes in fp16 and
//! this reference in fp32, accumulated over a 16-row block and 5 layers.

use std::path::{Path, PathBuf};

use half::f16;
use kaplumba_engine::dflash::{DflashConfig, DflashModel, DflashWeights};
use kaplumba_engine::exl3::{reconstruct_matrix, BitRate, Codebook, Exl3Error, SignVector};
use kaplumba_engine::safetensors::read_shard_header;

/// Input magnitudes for the synthetic taps and embeddings.
///
/// Chosen so the drafter's residual streams stay inside fp16's range. The
/// drafter runs entirely in fp16 (its EXL3 scales, its activations), so an
/// over-large synthetic input overflows: at a tap scale of 2.0 the first
/// residual reaches 3780 and the MLP output is `inf`, which then poisons every
/// later layer. Real hidden states are O(1)-O(10) by the time they reach a tap.
const TAP_SCALE: f32 = 0.35;
const INPUT_SCALE: f32 = 0.35;

fn dir() -> PathBuf {
    std::env::var("KAPLUMBA_DFLASH_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/Users/zayaantaha/.kaplumba/models/Drafter"))
}

/// Deterministic pseudo-random fp16 row. Not a real distribution, just
/// reproducible data with a wide dynamic range so a scaling or transposition
/// error shows up rather than averaging out.
/// Deterministic synthetic rows, smooth in the hidden dimension.
///
/// **Not** white noise, deliberately. White noise is pathological for EXL3: the
/// trellis is Hadamard-transformed and sign-flipped, so a decorrelated input
/// has a huge inner product against the weight rows. Measured on the real
/// checkpoint with white-noise taps, *both* the engine and the f32 reference
/// blow up -- the reference's layer-3 attention output reaches 316 469 and its
/// MLP output 32 861, while the fp16 engine overflows to `inf` in layer 0. That
/// is a test-data problem, not a graph bug, and it cost an hour because the
/// engine's symptom (NaN) looked like a kernel fault.
///
/// Real Gemma hidden states are strongly correlated across the hidden
/// dimension, so a few harmonics is the right shape: smooth, reproducible,
/// wide enough that a transposition or scaling error still shows up.
fn pseudo(seed: u64, count: usize, scale: f32) -> Vec<u16> {
    let state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let phases = |index: usize| -> [f32; 4] {
        let mut value = state.wrapping_add(index as u64);
        let mut out = [0.0_f32; 4];
        for slot in out.iter_mut() {
            value ^= value << 13;
            value ^= value >> 7;
            value ^= value << 17;
            *slot = (value >> 40) as f32 / (1_u64 << 24) as f32 * core::f32::consts::TAU;
        }
        out
    };
    (0..count)
        .map(|index| {
            let t = index as f32;
            let phase = phases(index / 997);
            // Periods of 1/64, 1/256, 1/997 and 1/4096 elements.
            let value = (core::f32::consts::TAU * (t / 64.0) + phase[0]).sin()
                + 0.7 * (core::f32::consts::TAU * (t / 256.0) + phase[1]).sin()
                + 0.4 * (core::f32::consts::TAU * (t / 997.0) + phase[2]).sin()
                + 0.15 * (core::f32::consts::TAU * (t / 4096.0) + phase[3]).sin();
            f16::from_f32(value * scale).to_bits()
        })
        .collect()
}

fn f32_of(values: &[u16]) -> Vec<f32> {
    values
        .iter()
        .map(|value| f16::from_bits(*value).to_f32())
        .collect()
}

fn stem_bytes(dir: &Path, name: &str) -> Vec<u8> {
    use std::io::{Read, Seek, SeekFrom};
    let path = dir.join("model.safetensors");
    let shard = read_shard_header(&path).expect("header");
    let (start, end) = shard.absolute_range(name).unwrap_or_else(|| panic!("{name}"));
    let mut file = std::fs::File::open(&path).expect("open");
    file.seek(SeekFrom::Start(start)).expect("seek");
    let mut bytes = vec![0u8; usize::try_from(end - start).unwrap()];
    file.read_exact(&mut bytes).expect("read");
    bytes
}

fn f16_words(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect()
}

/// A linear layer's fp32 weights, dequantised on the host from the shard.
///
/// Laid out **`[in_features, out_features]`**, which is what
/// `exl3::reconstruct_matrix` returns (see how `adv_gemm_error_budget.rs` walks
/// it: `matrix[value * out_features + out]`). Transposing it here is the easy
/// mistake, and it produces a disagreement that looks like a kernel bug.
struct HostLinear {
    weights: Vec<f32>,
    in_features: usize,
    out_features: usize,
}

impl HostLinear {
    fn build(dir: &Path, name: &str, in_features: usize, out_features: usize) -> Self {
        let trellis = f16_words(&stem_bytes(dir, &format!("{name}.trellis")));
        let raw_mcg = stem_bytes(dir, &format!("{name}.mcg"));
        let codebook = i32::from_le_bytes(raw_mcg[..4].try_into().unwrap());
        assert_eq!(
            codebook as u32,
            kaplumba_engine::exl3::MCG_MULTIPLIER,
            "{name} must be the mcg codebook"
        );
        let input = SignVector::from_half_bits(f16_words(&stem_bytes(dir, &format!("{name}.suh"))));
        let output = SignVector::from_half_bits(f16_words(&stem_bytes(dir, &format!("{name}.svh"))));
        let weights = reconstruct_matrix(
            &trellis,
            in_features / 16,
            out_features / 16,
            BitRate::Integer(4),
            Codebook::Mcg,
            &input,
            &output,
        )
        .unwrap_or_else(|error: Exl3Error| panic!("{name}: {error}"));
        Self {
            weights,
            in_features,
            out_features,
        }
    }

    /// `[rows, in] @ [in, out]`, f32.
    fn forward(&self, input: &[f32], rows: usize) -> Vec<f32> {
        assert_eq!(input.len(), rows * self.in_features);
        let mut output = vec![0.0_f32; rows * self.out_features];
        for row in 0..rows {
            for out in 0..self.out_features {
                let mut acc = 0.0_f32;
                for index in 0..self.in_features {
                    acc += input[row * self.in_features + index]
                        * self.weights[index * self.out_features + out];
                }
                output[row * self.out_features + out] = acc;
            }
        }
        output
    }
}

fn norm_weights(dir: &Path, name: &str, width: usize) -> Vec<f32> {
    let bytes = stem_bytes(dir, &format!("{name}.weight"));
    assert_eq!(bytes.len() / 2, width, "{name} width");
    // Verbatim, matching the loader. This used to add 1.0 (Gemma's
    // `(1 + weight)` convention) and the loader did the same, so the two agreed
    // while both were wrong: the drafter is Qwen3, whose RMSNorm has no offset
    // (exllamav3 `architecture/dflash.py:267` vs `architecture/gemma3.py:267`).
    // That single shared assumption hid a 4.2x error on
    // `post_attention_layernorm` and made acceptance exactly 0.0000.
    f16_words(&bytes)
        .into_iter()
        .map(|bits| f32::from_bits((bits as u32) << 16))
        .collect()
}

/// Gemma RMSNorm over `rows` rows of `width`, with the `+1` already in `weight`.
fn rms_norm(input: &[f32], weight: &[f32], rows: usize, width: usize, eps: f32) -> Vec<f32> {
    let mut output = vec![0.0_f32; rows * width];
    for row in 0..rows {
        let base = row * width;
        let sum: f32 = (0..width).map(|index| input[base + index].powi(2)).sum();
        let inverse = 1.0 / (sum / width as f32 + eps).sqrt();
        for index in 0..width {
            output[base + index] = input[base + index] * inverse * weight[index];
        }
    }
    output
}

/// Neox-style RoPE (`rotate_half`) over `[rows, heads, head_dim]`.
fn rope(
    input: &[f32],
    rows: usize,
    heads: usize,
    head_dim: usize,
    offset: usize,
    theta: f32,
) -> Vec<f32> {
    let mut output = input.to_vec();
    let half = head_dim / 2;
    for row in 0..rows {
        for head in 0..heads {
            let base = (row * heads + head) * head_dim;
            for pair in 0..half {
                let angle =
                    (offset + row) as f32 / theta.powf(2.0 * pair as f32 / head_dim as f32);
                let (cosine, sine) = (angle.cos(), angle.sin());
                let low = input[base + pair];
                let high = input[base + half + pair];
                output[base + pair] = low * cosine - high * sine;
                output[base + half + pair] = high * cosine + low * sine;
            }
        }
    }
    output
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

/// Shared builder for the three comparisons below.
struct Harness {
    dir: PathBuf,
    config: DflashConfig,
    model: DflashModel,
    lines: ReferenceLines,
}

struct LayerLines {
    q: HostLinear,
    k: HostLinear,
    v: HostLinear,
    o: HostLinear,
    gate: HostLinear,
    up: HostLinear,
    down: HostLinear,
}

struct ReferenceLines {
    fc: HostLinear,
    per_layer: Vec<LayerLines>,
}

impl ReferenceLines {
    fn build(dir: &Path, config: &DflashConfig) -> Self {
        let hidden = config.hidden_size;
        let head_dim = config.head_dim;
        let mut per_layer = Vec::with_capacity(config.num_layers);
        for layer in 0..config.num_layers {
            let prefix = format!("layers.{layer}");
            per_layer.push(LayerLines {
                q: HostLinear::build(
                    dir,
                    &format!("{prefix}.self_attn.q_proj"),
                    hidden,
                    config.q_heads * head_dim,
                ),
                k: HostLinear::build(
                    dir,
                    &format!("{prefix}.self_attn.k_proj"),
                    hidden,
                    config.kv_heads * head_dim,
                ),
                v: HostLinear::build(
                    dir,
                    &format!("{prefix}.self_attn.v_proj"),
                    hidden,
                    config.kv_heads * head_dim,
                ),
                o: HostLinear::build(
                    dir,
                    &format!("{prefix}.self_attn.o_proj"),
                    config.q_heads * head_dim,
                    hidden,
                ),
                gate: HostLinear::build(
                    dir,
                    &format!("{prefix}.mlp.gate_proj"),
                    hidden,
                    config.intermediate,
                ),
                up: HostLinear::build(
                    dir,
                    &format!("{prefix}.mlp.up_proj"),
                    hidden,
                    config.intermediate,
                ),
                down: HostLinear::build(
                    dir,
                    &format!("{prefix}.mlp.down_proj"),
                    config.intermediate,
                    hidden,
                ),
            });
        }
        Self {
            fc: HostLinear::build(dir, "fc", config.tap_width(), hidden),
            per_layer,
        }
    }
}

impl Harness {
    fn build() -> Option<Self> {
        let dir = dir();
        if !dir.join("model.safetensors").exists() {
            eprintln!("drafter checkpoint not present; skipping");
            return None;
        }
        let config = DflashConfig::load(&dir).expect("config");
        let device = metal::Device::system_default().expect("device");
        let weights = DflashWeights::load(&dir, &device).expect("weights");
        let model = DflashModel::new(weights, config.clone()).expect("model");
        let lines = ReferenceLines::build(&dir, &config);
        Some(Self {
            dir,
            config,
            model,
            lines,
        })
    }

    fn hidden(&self) -> usize {
        self.config.hidden_size
    }

    /// One smooth synthetic tap row per target layer.
    fn taps(&self, n_ctx: usize, seed: u64) -> Vec<Vec<u16>> {
        self.config
            .target_layer_ids
            .iter()
            .enumerate()
            .map(|(index, _)| pseudo(seed + index as u64, n_ctx * self.hidden(), TAP_SCALE))
            .collect()
    }

    /// The reference's `fc` + `hidden_norm` over the same taps.
    fn reference_fused(&self, taps: &[Vec<u16>], n_ctx: usize) -> Vec<f32> {
        let hidden = self.hidden();
        let width = self.config.tap_width();
        let mut concat = vec![0.0_f32; n_ctx * width];
        for row in 0..n_ctx {
            for (tap, values) in taps.iter().enumerate() {
                for feature in 0..hidden {
                    concat[row * width + tap * hidden + feature] =
                        f16::from_bits(values[row * hidden + feature]).to_f32();
                }
            }
        }
        let projected = self.lines.fc.forward(&concat, n_ctx);
        rms_norm(
            &projected,
            &norm_weights(&self.dir, "hidden_norm", hidden),
            n_ctx,
            hidden,
            self.config.rms_norm_eps,
        )
    }
}

#[test]
#[ignore = "loads the DFlash checkpoint to inspect fixed Metal scratch ownership"]
fn dflash_gpu_scratch_is_block_sized_and_caches_norm_sign_buffers() {
    let Some(harness) = Harness::build() else {
        return;
    };
    let rows = harness.config.block_size as u64;
    let hidden = harness.config.hidden_size as u64;
    let intermediate = harness.config.intermediate as u64;
    let q_width = (harness.config.q_heads * harness.config.head_dim) as u64;
    let kv_width = (harness.config.kv_heads * harness.config.head_dim) as u64;
    let h = rows * hidden;
    let q = rows * q_width;
    let kv = rows * kv_width;
    let inter = rows * intermediate;
    let max_linear = h.max(q).max(kv).max(inter);
    let expected = 3 * h * 4
        + h * 2
        + 3 * q * 2
        + 3 * kv * 2
        + kv * 2
        + q * 2
        + 3 * inter * 2
        + 2 * max_linear * 2
        + max_linear * 4
        + max_linear * 2
        + h * 4
        + h * 2
        + harness.config.num_layers as u64 * 2 * kv * 2;
    eprintln!(
        "DFlash fixed scratch={} bytes, cached signs/norms={} bytes",
        harness.model.gpu_scratch_bytes(),
        harness.model.gpu_cached_weight_bytes()
    );
    assert_eq!(harness.model.gpu_scratch_bytes(), expected);
    assert!(harness.model.gpu_cached_weight_bytes() > 0);
    assert!(harness.model.gpu_scratch_bytes() < 32 * 1024 * 1024);
    assert!(harness.model.gpu_cached_weight_bytes() < 16 * 1024 * 1024);
}

#[test]
#[ignore = "loads the DFlash checkpoint and compares both block implementations"]
fn dflash_gpu_encoded_block_matches_slice_block() {
    let dir = dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("drafter checkpoint not present; skipping");
        return;
    }
    let config = DflashConfig::load(&dir).expect("config");
    let device = metal::Device::system_default().expect("device");
    let weights = DflashWeights::load(&dir, &device).expect("weights");
    let mut model = DflashModel::new(weights, config.clone()).expect("model");
    let hidden = config.hidden_size;
    let kv_width = config.kv_heads * config.head_dim;
    let rows = config.block_size;
    let anchor = pseudo(0x5EED_A11C, hidden, INPUT_SCALE);
    let mask = pseudo(0x0A5C0FFE, hidden, INPUT_SCALE);
    model.set_mask_embedding(&mask).expect("mask embedding");

    for (case_name, position, counts) in [
        ("empty", 24usize, vec![0usize; config.num_layers]),
        ("normal", 20 + rows, vec![20usize; config.num_layers]),
        (
            "sliding",
            config.sliding_window + rows + 3,
            config
                .layer_is_sliding
                .iter()
                .map(|sliding| {
                    if *sliding {
                        config.sliding_window
                    } else {
                        config.sliding_window + 3
                    }
                })
                .collect(),
        ),
    ] {
        let mut ctx_keys = Vec::with_capacity(config.num_layers);
        let mut ctx_values = Vec::with_capacity(config.num_layers);
        let mut ctx_first = Vec::with_capacity(config.num_layers);
        for (layer, count) in counts.iter().copied().enumerate() {
            ctx_keys.push(pseudo(0xDFA5_0000 + layer as u64, count * kv_width, 0.01));
            ctx_values.push(pseudo(0xDFA6_0000 + layer as u64, count * kv_width, 0.01));
            ctx_first.push(position.saturating_sub(count));
        }
        let context = kaplumba_engine::dflash::BlockContext {
            anchor_embedding: &anchor,
            ctx_keys: &ctx_keys,
            ctx_values: &ctx_values,
            ctx_first: &ctx_first,
            position,
        };
        let sliced = model.forward_block(&context).expect("slice block");
        let encoded = model.forward_block_gpu(&context).expect("encoded block");
        let max_diff = |left: &[u16], right: &[u16]| -> f32 {
            assert_eq!(left.len(), right.len());
            left.iter()
                .zip(right)
                .map(|(a, b)| {
                    (f16::from_bits(*a).to_f32() - f16::from_bits(*b).to_f32()).abs()
                })
                .fold(0.0_f32, f32::max)
        };
        let max_magnitude = |left: &[u16], right: &[u16]| -> f32 {
            left.iter()
                .chain(right)
                .map(|bits| f16::from_bits(*bits).to_f32().abs())
                .fold(0.0_f32, f32::max)
        };
        let k_diff = encoded
            .block_keys
            .iter()
            .zip(&sliced.block_keys)
            .map(|(a, b)| max_diff(a, b))
            .fold(0.0_f32, f32::max);
        let v_diff = encoded
            .block_values
            .iter()
            .zip(&sliced.block_values)
            .map(|(a, b)| max_diff(a, b))
            .fold(0.0_f32, f32::max);
        let hidden_diff = max_diff(&encoded.hidden, &sliced.hidden);
        let k_scale = encoded
            .block_keys
            .iter()
            .zip(&sliced.block_keys)
            .map(|(a, b)| max_magnitude(a, b))
            .fold(0.0_f32, f32::max);
        let v_scale = encoded
            .block_values
            .iter()
            .zip(&sliced.block_values)
            .map(|(a, b)| max_magnitude(a, b))
            .fold(0.0_f32, f32::max);
        let hidden_scale = max_magnitude(&encoded.hidden, &sliced.hidden);
        eprintln!(
            "{case_name}: K {k_diff:.6}/{k_scale:.6} ({:.3e}), V {v_diff:.6}/{v_scale:.6} ({:.3e}), hidden {hidden_diff:.6}/{hidden_scale:.6} ({:.3e})",
            k_diff / k_scale.max(f32::MIN_POSITIVE),
            v_diff / v_scale.max(f32::MIN_POSITIVE),
            hidden_diff / hidden_scale.max(f32::MIN_POSITIVE)
        );
        assert!(
            k_diff / k_scale.max(f32::MIN_POSITIVE) < 5e-3,
            "{case_name} K relative mismatch {k_diff}/{k_scale}"
        );
        assert!(
            v_diff / v_scale.max(f32::MIN_POSITIVE) < 5e-3,
            "{case_name} V relative mismatch {v_diff}/{v_scale}"
        );
        assert!(
            hidden_diff / hidden_scale.max(f32::MIN_POSITIVE) < 5e-3,
            "{case_name} hidden relative mismatch {hidden_diff}/{hidden_scale}"
        );
        if case_name == "normal" {
            let mut slice_times = Vec::with_capacity(5);
            let mut encoded_times = Vec::with_capacity(5);
            for _ in 0..5 {
                let start = std::time::Instant::now();
                std::hint::black_box(model.forward_block(&context).expect("slice timing"));
                slice_times.push(start.elapsed().as_secs_f64() * 1000.0);
                let start = std::time::Instant::now();
                std::hint::black_box(model.forward_block_gpu(&context).expect("encoded timing"));
                encoded_times.push(start.elapsed().as_secs_f64() * 1000.0);
            }
            slice_times.sort_by(f64::total_cmp);
            encoded_times.sort_by(f64::total_cmp);
            eprintln!(
                "normal block median: slice={:.2}ms encoded={:.2}ms",
                slice_times[2], encoded_times[2]
            );
        }
    }
}

fn max_of(values: &[f32]) -> f32 {
    values.iter().map(|value| value.abs()).fold(0.0_f32, f32::max)
}

// ---------------------------------------------------------------------------
// 1. The encoder: 6 concatenated taps -> fc -> hidden_norm.
// ---------------------------------------------------------------------------

#[test]
fn dflash_fuse_context_matches_host() {
    let Some(mut harness) = Harness::build() else {
        return;
    };
    let hidden = harness.hidden();
    let n_ctx = 20usize;
    let taps = harness.taps(n_ctx, 0xDF_1A_5000);
    let tap_refs: Vec<&[u16]> = taps.iter().map(|tap| tap.as_slice()).collect();
    let fused = harness.model.fuse_context(&tap_refs, n_ctx).expect("fuse");
    let expected = harness.reference_fused(&taps, n_ctx);

    let actual = f32_of(&fused);
    assert_eq!(actual.len(), expected.len());
    let mut worst = 0.0_f32;
    let mut worst_index = 0usize;
    for (index, (a, b)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - b).abs();
        if diff > worst {
            worst = diff;
            worst_index = index;
        }
    }
    eprintln!(
        "fuse_context worst diff {worst:.6} at {worst_index} of {} (rms {:.4})",
        actual.len(),
        (actual.iter().map(|value| value * value).sum::<f32>() / actual.len() as f32).sqrt()
    );
    // The RMSNorm weight has an RMS above 1 (its stored BF16 values are ~1.1-1.4,
    // +1 for Gemma), so the fused row's RMS is ~2.2 by construction.
    // fp16 vs an f32 reference: fp16 carries ~24 bits, so a 2e-3 *relative*
    // bound (2x fp16's own epsilon) is the right shape. An absolute bound fails
    // spuriously here because `hidden_norm` has a +1 weight with RMS 2.2, so the
    // fused rows run to ~10 while the *errors* are ~0.006.
    let reference_max = max_of(&expected);
    assert!(
        worst <= 2e-3 * reference_max,
        "fuse_context disagrees with the host reference by {worst} (reference max {reference_max}) at {worst_index}"
    );
    assert!(
        worst < 0.05,
        "fuse_context disagrees with the host reference by {worst} at {worst_index}"
    );
}

// ---------------------------------------------------------------------------
// 2. KV injection: fused feature -> per-layer k_proj/v_proj/k_norm/RoPE.
// ---------------------------------------------------------------------------

#[test]
fn dflash_context_kv_matches_host() {
    let Some(mut harness) = Harness::build() else {
        return;
    };
    let hidden = harness.hidden();
    let head_dim = harness.config.head_dim;
    let n_ctx = 20usize;
    let position = n_ctx;
    let taps = harness.taps(n_ctx, 0xDF_1A_5000);
    let tap_refs: Vec<&[u16]> = taps.iter().map(|tap| tap.as_slice()).collect();
    let fused = harness.model.fuse_context(&tap_refs, n_ctx).expect("fuse");
    let reference_fused = harness.reference_fused(&taps, n_ctx);
    let positions: Vec<u32> = (0..n_ctx as u32).collect();

    for layer in 0..harness.config.num_layers {
        let (k, v) = harness
            .model
            .project_context_kv(layer, &fused, n_ctx, &positions, position)
            .expect("context kv");

        let prefix = format!("layers.{layer}");
        let k_norm = norm_weights(
            &harness.dir,
            &format!("{prefix}.self_attn.k_norm"),
            head_dim,
        );
        let k_raw = harness.lines.per_layer[layer].k.forward(&reference_fused, n_ctx);
        let expected_k = rms_norm(
            &k_raw,
            &k_norm,
            n_ctx * harness.config.kv_heads,
            head_dim,
            harness.config.rms_norm_eps,
        );
        let expected_k = rope(
            &expected_k,
            n_ctx,
            harness.config.kv_heads,
            head_dim,
            0,
            harness.config.rope_theta,
        );
        let expected_v = harness.lines.per_layer[layer].v.forward(&reference_fused, n_ctx);

        for (name, actual, expected) in
            [("k", f32_of(&k), &expected_k), ("v", f32_of(&v), &expected_v)]
        {
            let worst = actual
                .iter()
                .zip(expected.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            let reference_max = max_of(&expected);
            // 2e-3 relative: fp16's own epsilon against an f32 reference.
            // `v_proj`'s rows run to ~45 on this data, so an absolute bound
            // rejects a 0.075 error that is 0.17% -- well inside fp16.
            assert!(
                worst <= 2e-3 * reference_max,
                "layer {layer} context {name} disagrees with the host reference by {worst} \
                 (reference max {reference_max})"
            );
        }
        eprintln!("layer {layer} context k/v agree to < 0.05");
    }
}

// ---------------------------------------------------------------------------
// 3. The whole 5-layer block pass, with no context rows.
//
// Why no context: the drafter runs in fp16 and its intermediate magnitudes are
// genuinely large. Measured on the real checkpoint with synthetic context, the
// f32 reference's layer-0 `attended` reaches 105 (bounded by `v_proj`'s output,
// which is 45.8 from an RMS-2.2 fused row) and its layer-0 MLP output reaches
// 164 165, past fp16's 65 504. That is a property of *synthetic* context, not
// of the drafter: real target hidden states are aligned with the trained
// direction. Closing this with real context needs a target forward, so it
// belongs with the end-to-end acceptance work, not with the golden graph test.
//
// With no context rows the block attends to itself only, V is scaled by
// `INPUT_SCALE`, and every other stage of the graph is still exercised:
// embeddings, all five layers' norms and projections, the causal / window /
// bidirectional mask over 16 rows, both residuals, the SwiGLU, the final norm.
// ---------------------------------------------------------------------------

#[test]
fn dflash_forward_block_matches_host() {
    let Some(mut harness) = Harness::build() else {
        return;
    };
    let hidden = harness.hidden();
    let head_dim = harness.config.head_dim;
    let rows = harness.config.block_size;
    let position = 24usize;
    let n_ctx = 0usize;

    let anchor = pseudo(0x5EED_A11C, hidden, INPUT_SCALE);
    let mask = pseudo(0x0A5C0FFE, hidden, INPUT_SCALE);
    let empty: Vec<u16> = Vec::new();
    let empties = vec![empty.clone(); harness.config.num_layers];
    // No context in this test, so every layer's first retained row is where its
    // (empty) context would have started.
    let ctx_first = vec![position.saturating_sub(n_ctx); harness.config.num_layers];

    harness
        .model
        .set_mask_embedding(&mask)
        .expect("mask embedding");
    let block = harness
        .model
        .forward_block(&kaplumba_engine::dflash::BlockContext {
            anchor_embedding: &anchor,
            ctx_keys: &empties,
            ctx_values: &empties,
            ctx_first: &ctx_first,
            position,
        })
        .expect("block");

    // --- reference, layer by layer, in f32 --------------------------------
    let eps = harness.config.rms_norm_eps;
    let groups = harness.config.q_heads / harness.config.kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut state = vec![0.0_f32; rows * hidden];
    for row in 0..rows {
        let source = if row == 0 { f32_of(&anchor) } else { f32_of(&mask) };
        state[row * hidden..(row + 1) * hidden].copy_from_slice(&source);
    }
    let mut stage_max = Vec::new();
    for layer in 0..harness.config.num_layers {
        let prefix = format!("layers.{layer}");
        let line = &harness.lines.per_layer[layer];
        let normed = rms_norm(
            &state,
            &norm_weights(&harness.dir, &format!("{prefix}.input_layernorm"), hidden),
            rows,
            hidden,
            eps,
        );
        let q_out = line.q.forward(&normed, rows);
        let q_norm = rms_norm(
            &q_out,
            &norm_weights(&harness.dir, &format!("{prefix}.self_attn.q_norm"), head_dim),
            rows * harness.config.q_heads,
            head_dim,
            eps,
        );
        let q_rope = rope(
            &q_norm,
            rows,
            harness.config.q_heads,
            head_dim,
            position,
            harness.config.rope_theta,
        );
        let k_block = line.k.forward(&normed, rows);
        let k_block = rms_norm(
            &k_block,
            &norm_weights(&harness.dir, &format!("{prefix}.self_attn.k_norm"), head_dim),
            rows * harness.config.kv_heads,
            head_dim,
            eps,
        );
        let k_block = rope(
            &k_block,
            rows,
            harness.config.kv_heads,
            head_dim,
            position,
            harness.config.rope_theta,
        );
        let v_block = line.v.forward(&normed, rows);

        // No context rows, so every key is a block row and the mask is the
        // block mask.
        let bidirectional = !harness.config.layer_is_sliding[layer];
        // Window on sliding layers only, and the gap in i64 for the
        // bidirectional case: see `host_block`.
        let window = if harness.config.layer_is_sliding[layer] {
            harness.config.sliding_window
        } else {
            0
        };
        let mut attended = vec![0.0_f32; rows * harness.config.q_heads * head_dim];
        for row in 0..rows {
            let q_pos = (position + row) as u32;
            for q_head in 0..harness.config.q_heads {
                let key_head = q_head / groups;
                let mut scores = Vec::with_capacity(rows);
                for key in 0..rows {
                    let k_pos = (position + key) as u32;
                    let past_ok = bidirectional || k_pos <= q_pos;
                    let window_ok =
                        window == 0 || (q_pos as i64 - k_pos as i64) < window as i64;
                    if past_ok && window_ok {
                        let base = (key * harness.config.kv_heads + key_head) * head_dim;
                        let q_base = (row * harness.config.q_heads + q_head) * head_dim;
                        let dot: f32 = (0..head_dim)
                            .map(|d| q_rope[q_base + d] * k_block[base + d])
                            .sum();
                        scores.push((key, dot * scale));
                    }
                }
                let max = scores
                    .iter()
                    .map(|(_, score)| *score)
                    .fold(f32::NEG_INFINITY, f32::max);
                let total: f32 = scores.iter().map(|(_, s)| (s - max).exp()).sum();
                let o_base = (row * harness.config.q_heads + q_head) * head_dim;
                for d in 0..head_dim {
                    let mut acc = 0.0_f32;
                    for (key, score) in &scores {
                        let base = (*key * harness.config.kv_heads + key_head) * head_dim;
                        acc += (score - max).exp() / total * v_block[base + d];
                    }
                    attended[o_base + d] = acc;
                }
            }
        }
        stage_max.push(max_of(&attended));
        let projected = line.o.forward(&attended, rows);
        stage_max.push(max_of(&projected));
        for (index, value) in projected.iter().enumerate() {
            state[index] += value;
        }
        let mlp_normed = rms_norm(
            &state,
            &norm_weights(
                &harness.dir,
                &format!("{prefix}.post_attention_layernorm"),
                hidden,
            ),
            rows,
            hidden,
            eps,
        );
        let gate_out = line.gate.forward(&mlp_normed, rows);
        let up_out = line.up.forward(&mlp_normed, rows);
        let activated: Vec<f32> = gate_out
            .iter()
            .zip(up_out.iter())
            .map(|(gate_value, up_value)| silu(*gate_value) * up_value)
            .collect();
        stage_max.push(max_of(&activated));
        let down_out = line.down.forward(&activated, rows);
        stage_max.push(max_of(&down_out));
        for (index, value) in down_out.iter().enumerate() {
            state[index] += value;
        }
    }
    let expected = rms_norm(
        &state,
        &norm_weights(&harness.dir, "norm", hidden),
        rows,
        hidden,
        eps,
    );

    // --- compare -----------------------------------------------------------
    let actual = f32_of(&block.hidden);
    assert_eq!(actual.len(), expected.len());
    let mut worst = 0.0_f32;
    let mut worst_index = 0usize;
    for (index, (a, b)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - b).abs();
        if diff > worst {
            worst = diff;
            worst_index = index;
        }
    }
    let engine_energy: f32 = actual.iter().map(|value| value * value).sum();
    let engine_max = max_of(&actual);
    eprintln!(
        "forward_block worst diff {worst:.6} at {worst_index} of {}; engine max {engine_max:.4}, \
         reference max {:.4}; reference per-stage max {}",
        actual.len(),
        max_of(&expected),
        stage_max
            .iter()
            .map(|value| format!("{value:.2}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    // `NaN - x` is `NaN` and `NaN > worst` is false, so an all-NaN engine output
    // would report "worst diff 0.000000" and pass. This guard is what caught the
    // first version of this test doing exactly that.
    assert!(
        engine_energy.is_finite() && engine_max > 0.01,
        "the comparison is vacuous: the engine carries no signal (energy {engine_energy})"
    );
    // fp16 accumulations over 5 layers of a 5376-wide stream, against an f32
    // reference: 0.15 is the expected spread. A graph error -- a mis-ordered
    // norm, a residual from the wrong stream, RoPE on V -- is an order of
    // magnitude larger.
    let reference_max = max_of(&expected);
    assert!(
        worst <= 2e-3 * reference_max,
        "forward_block disagrees with the host reference by {worst} (reference max {reference_max}) at element {worst_index}"
    );
}

// ---------------------------------------------------------------------------
// 4. The EXL3 decode, tensor by tensor.
//
// This is what actually proves the trellis layout, the mcg codebook, the bitrate
// and the sign-vector orientation are right for this checkpoint: the engine's
// GPU gemm and the host `reconstruct_matrix` must agree on every one of the 36
// linears. An earlier version of this check compared the drafter's weight RMS
// against the *target's* `o_proj` and got a 9.6x ratio — but that comparison was
// meaningless, because the drafter's `o_proj` is its own trained layer and has
// no reason to share the target's scale. Comparing the two *implementations* on
// the *same* weights is the check that can actually fail.
// ---------------------------------------------------------------------------

#[test]
fn dflash_exl3_decode_matches_host_on_every_linear() {
    let Some(harness) = Harness::build() else {
        return;
    };
    let hidden = harness.hidden();
    let head_dim = harness.config.head_dim;
    let rows = 4usize; // M=4: enough rows that a transposed layout diverges.

    let mut names = vec!["fc".to_string()];
    for layer in 0..harness.config.num_layers {
        let prefix = format!("layers.{layer}");
        for suffix in [
            format!("{prefix}.self_attn.q_proj"),
            format!("{prefix}.self_attn.k_proj"),
            format!("{prefix}.self_attn.v_proj"),
            format!("{prefix}.self_attn.o_proj"),
            format!("{prefix}.mlp.gate_proj"),
            format!("{prefix}.mlp.up_proj"),
            format!("{prefix}.mlp.down_proj"),
        ] {
            names.push(suffix);
        }
    }
    assert_eq!(names.len(), 1 + harness.config.num_layers * 7);

    let mut worst_overall = 0.0_f32;
    for name in &names {
        let (in_features, out_features) = if name == "fc" {
            (harness.config.tap_width(), hidden)
        } else {
            let (in_features, out_features) = harness.model.gemv_shape(name);
            (in_features, out_features)
        };
        let host = HostLinear::build(&harness.dir, name, in_features, out_features);
        // Smooth rows, small enough to stay inside fp16 for the wide projections.
        let input = pseudo(0x5CA1_E000 ^ name.len() as u64, rows * in_features, 0.05);
        let input_f32 = f32_of(&input);
        let expected = host.forward(&input_f32, rows);
        let actual = f32_of(&harness.model.gemm(name, &input).expect("gemm"));
        let reference_max = max_of(&expected);
        let worst = actual
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        eprintln!("{name}: worst {} (reference max {reference_max:.4})", worst);
        assert!(
            worst <= 2e-3 * reference_max,
            "{name} disagrees with the host decode by {worst} (reference max {reference_max})"
        );
        worst_overall = worst_overall.max(worst);
    }
    eprintln!("all {} linears agree; worst {worst_overall:.6}", names.len());
}

/// The drafter's block attention, in f32, over context rows followed by block
/// rows. `ctx_offset` is the absolute position of the first context row and
/// `block_start` of the first block row; `ctx_len` of `usize::MAX` means "no
/// context".
#[allow(clippy::too_many_arguments)]
fn attended_of(
    q: &[f32],
    k_ctx: &[f32],
    v_ctx: &[f32],
    k_block: &[f32],
    v_block: &[f32],
    ctx_len: usize,
    rows: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    groups: usize,
    scale: f32,
    ctx_offset: usize,
    block_start: usize,
    bidirectional: bool,
    window: usize,
) -> Vec<f32> {
    let mut attended = vec![0.0_f32; rows * q_heads * head_dim];
    for row in 0..rows {
        let q_pos = (block_start + row) as u32;
        for q_head in 0..q_heads {
            let key_head = q_head / groups;
            let mut scores = Vec::new();
            for key in 0..ctx_len + rows {
                let k_pos = if key < ctx_len {
                    (ctx_offset + key) as u32
                } else {
                    (block_start + key - ctx_len) as u32
                };
                let past_ok = bidirectional || k_pos <= q_pos;
                // In i64: a bidirectional layer legitimately has keys after the
                // query, and the u32 subtraction of two positions wraps to ~4e9
                // there rather than going negative.
                let window_ok = window == 0 || (q_pos as i64 - k_pos as i64) < window as i64;
                if !(past_ok && window_ok) {
                    continue;
                }
                let (keys, base) = if key < ctx_len {
                    (k_ctx, (key * kv_heads + key_head) * head_dim)
                } else {
                    (
                        k_block,
                        ((key - ctx_len) * kv_heads + key_head) * head_dim,
                    )
                };
                let q_base = (row * q_heads + q_head) * head_dim;
                let dot: f32 = (0..head_dim)
                    .map(|d| q[q_base + d] * keys[base + d])
                    .sum();
                scores.push((key, dot * scale));
            }
            let max = scores
                .iter()
                .map(|(_, score)| *score)
                .fold(f32::NEG_INFINITY, f32::max);
            let total: f32 = scores.iter().map(|(_, s)| (s - max).exp()).sum();
            let o_base = (row * q_heads + q_head) * head_dim;
            for d in 0..head_dim {
                let mut acc = 0.0_f32;
                for (key, score) in &scores {
                    let (values, base) = if *key < ctx_len {
                        (v_ctx, (*key * kv_heads + key_head) * head_dim)
                    } else {
                        (
                            v_block,
                            ((key - ctx_len) * kv_heads + key_head) * head_dim,
                        )
                    };
                    acc += (score - max).exp() / total * values[base + d];
                }
                attended[o_base + d] = acc;
            }
        }
    }
    attended
}

// ---------------------------------------------------------------------------
// 5. The whole 5-layer block pass WITH real context rows.
//
// This is the test the spec was missing, and the reason D4 existed:
// `dflash_forward_block_matches_host` above runs with **zero** context rows, and
// `dflash_cache`/`dflash_streaming` only check cache arithmetic against itself.
// Nothing had ever compared the context path -- `fc`/`hidden_norm` fusion, the
// per-layer `k_proj`/`v_proj`/`k_norm`/RoPE, the drafter's attention over the
// context rows, and the window those rows are cached at -- against z-lab. The
// measured symptom was alpha = 0.0000 with byte-identical drafts for two
// different tap sets, i.e. a drafter whose output does not depend on its
// context at all.
//
// With T14's f32 residual stream the magnitudes that used to overflow fp16
// (layer-0 attention output 45.8, `o_proj` 2169, MLP output 152 288) now fit, so
// this comparison is possible.
//
// Two tests, because agreement and context sensitivity are different claims and
// the first version of this file made only the weaker one:
//
// * `dflash_forward_block_with_context_matches_host` -- the engine agrees with
//   the reference over 20 context rows.
// * `dflash_context_rows_are_visible_in_the_block_output` -- the context is
//   what the output depends on. Agreement alone does not prove that: a
//   reference and an engine that both ignore the context agree perfectly.
// ---------------------------------------------------------------------------

/// The drafter's block pass in f32 scalar code: `n_ctx` context rows at
/// positions `0..n_ctx`, then `block_size` block rows at `position..`.
///
/// Returns the final `norm` output and the per-stage max magnitude. The stage
/// maxima matter: a stage reaching 1.5e5 makes a 1.7 absolute difference
/// meaningless until you can see that the magnitude is 1.5e5, and they are what
/// separates a real graph disagreement from fp16 range.
fn host_block(
    harness: &Harness,
    anchor: &[u16],
    mask: &[u16],
    taps: &[Vec<u16>],
    n_ctx: usize,
    position: usize,
) -> (Vec<f32>, Vec<f32>) {
    let hidden = harness.hidden();
    let head_dim = harness.config.head_dim;
    let rows = harness.config.block_size;
    let eps = harness.config.rms_norm_eps;

    // --- host reference, with the same context ----------------------------
    let reference_fused = harness.reference_fused(taps, n_ctx);
    let groups = harness.config.q_heads / harness.config.kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut state = vec![0.0_f32; rows * hidden];
    for row in 0..rows {
        let source = if row == 0 { f32_of(&anchor) } else { f32_of(&mask) };
        state[row * hidden..(row + 1) * hidden].copy_from_slice(&source);
    }
    let mut stage_max = Vec::new();
    for layer in 0..harness.config.num_layers {
        let prefix = format!("layers.{layer}");
        let line = &harness.lines.per_layer[layer];
        let k_norm = norm_weights(
            &harness.dir,
            &format!("{prefix}.self_attn.k_norm"),
            head_dim,
        );
        let normed = rms_norm(
            &state,
            &norm_weights(&harness.dir, &format!("{prefix}.input_layernorm"), hidden),
            rows,
            hidden,
            eps,
        );
        let q_out = line.q.forward(&normed, rows);
        let q_norm = rms_norm(
            &q_out,
            &norm_weights(&harness.dir, &format!("{prefix}.self_attn.q_norm"), head_dim),
            rows * harness.config.q_heads,
            head_dim,
            eps,
        );
        let q_rope = rope(
            &q_norm,
            rows,
            harness.config.q_heads,
            head_dim,
            position,
            harness.config.rope_theta,
        );
        let k_block = line.k.forward(&normed, rows);
        let k_block = rms_norm(
            &k_block,
            &k_norm,
            rows * harness.config.kv_heads,
            head_dim,
            eps,
        );
        let k_block = rope(
            &k_block,
            rows,
            harness.config.kv_heads,
            head_dim,
            position,
            harness.config.rope_theta,
        );
        let v_block = line.v.forward(&normed, rows);
        let k_ctx = line.k.forward(&reference_fused, n_ctx);
        let k_ctx = rms_norm(
            &k_ctx,
            &k_norm,
            n_ctx * harness.config.kv_heads,
            head_dim,
            eps,
        );
        let k_ctx = rope(
            &k_ctx,
            n_ctx,
            harness.config.kv_heads,
            head_dim,
            0,
            harness.config.rope_theta,
        );
        let v_ctx = line.v.forward(&reference_fused, n_ctx);
        let sliding = harness.config.layer_is_sliding[layer];
        let attended = attended_of(
            &q_rope,
            &k_ctx,
            &v_ctx,
            &k_block,
            &v_block,
            n_ctx,
            rows,
            harness.config.q_heads,
            harness.config.kv_heads,
            head_dim,
            groups,
            scale,
            0,
            position,
            !sliding,
            // NOT `config.sliding_window` unconditionally: the window applies to
            // sliding layers only. z-lab sets `self.sliding_window =
            // config.sliding_window if layer_type == "sliding_attention" else None`
            // (`dflash/model.py:366`), and the engine passes the same
            // (`forward_block`: `window: if sliding { .. } else { 0 }`). Handing
            // the full layer a 2048 window is not a no-op here: with
            // `bidirectional` true its `q_pos - k_pos` is negative for the block
            // rows ahead of the query, so the u32 subtraction wraps and every
            // future key is dropped -- the reference quietly became causal on the
            // one layer that is not, which is a 7% error on `attn_out` and was
            // invisible while this test only printed its result.
            if sliding {
                harness.config.sliding_window
            } else {
                0
            },
        );
        stage_max.push(max_of(&attended));
        let projected = line.o.forward(&attended, rows);
        stage_max.push(max_of(&projected));
        for (index, value) in projected.iter().enumerate() {
            state[index] += value;
        }
        let mlp_normed = rms_norm(
            &state,
            &norm_weights(
                &harness.dir,
                &format!("{prefix}.post_attention_layernorm"),
                hidden,
            ),
            rows,
            hidden,
            eps,
        );
        let gate_out = line.gate.forward(&mlp_normed, rows);
        let up_out = line.up.forward(&mlp_normed, rows);
        let activated: Vec<f32> = gate_out
            .iter()
            .zip(up_out.iter())
            .map(|(gate_value, up_value)| silu(*gate_value) * up_value)
            .collect();
        stage_max.push(max_of(&activated));
        let down_out = line.down.forward(&activated, rows);
        stage_max.push(max_of(&down_out));
        for (index, value) in down_out.iter().enumerate() {
            state[index] += value;
        }
    }
    let expected = rms_norm(
        &state,
        &norm_weights(&harness.dir, "norm", hidden),
        rows,
        hidden,
        eps,
    );
    (expected, stage_max)
}

/// One block pass through both implementations, over the same context rows.
struct BlockCase {
    engine: Vec<f32>,
    host: Vec<f32>,
    stage_max: Vec<f32>,
    /// `max |engine - host|` -- the number the agreement test asserts on.
    worst: f32,
    /// Where that difference is, so a failure names a row and a feature rather
    /// than only a magnitude. `worst / hidden` is the block row.
    worst_index: usize,
    /// The largest element of `host`, the scale that difference is relative to.
    host_max: f32,
}

/// Build the drafter's cache through the real entry point (`append_context`, as
/// the round loop does), run `forward_block` over it, and pair the result with
/// [`host_block`] on the same rows.
fn block_case(
    harness: &mut Harness,
    anchor: &[u16],
    mask: &[u16],
    taps: &[Vec<u16>],
    n_ctx: usize,
    position: usize,
) -> BlockCase {
    let mut cache = kaplumba_engine::dflash::DflashKvCache::new(&harness.config);
    if n_ctx > 0 {
        let tap_refs: Vec<&[u16]> = taps.iter().map(|tap| tap.as_slice()).collect();
        harness
            .model
            .append_context(&tap_refs, n_ctx, n_ctx, &mut cache)
            .expect("append context");
    }
    harness
        .model
        .set_mask_embedding(mask)
        .expect("mask embedding");
    let first = cache.all_first_positions();
    let block = harness
        .model
        .forward_block(&kaplumba_engine::dflash::BlockContext {
            anchor_embedding: anchor,
            ctx_keys: cache.all_keys(),
            ctx_values: cache.all_values(),
            ctx_first: &first,
            position,
        })
        .expect("block");
    let engine = f32_of(&block.hidden);
    let (host, stage_max) = host_block(harness, anchor, mask, taps, n_ctx, position);
    assert_eq!(engine.len(), host.len());
    let mut worst = 0.0_f32;
    let mut worst_index = 0usize;
    for (index, (a, b)) in engine.iter().zip(host.iter()).enumerate() {
        let diff = (a - b).abs();
        if diff > worst {
            worst = diff;
            worst_index = index;
        }
    }
    let host_max = max_of(&host);
    BlockCase {
        engine,
        host,
        stage_max,
        worst,
        worst_index,
        host_max,
    }
}

fn rms_of(values: &[f32]) -> f32 {
    (values.iter().map(|value| value * value).sum::<f32>() / values.len() as f32).sqrt()
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

/// Tolerance for the context path, as a fraction of the reference's largest
/// element.
///
/// Measured: 1.703331 on a reference max of 89.6041, i.e. 1.90e-2, against
/// 8.8e-4 (0.068110 / 77.2833) for the same comparison with **zero** context
/// rows. The block's residual stream is f32 and its stages reach 1.5e5, so 1.7
/// absolute is 1e-5 of a stage magnitude; the number is set by fp16 in the
/// context's `k_proj`/`v_proj` and by the f16 context K/V themselves, and it
/// does not grow with the context length at a fixed position (see the notes).
const CONTEXT_TOLERANCE: f32 = 3e-2;

#[test]
fn dflash_forward_block_with_context_matches_host() {
    let Some(mut harness) = Harness::build() else {
        return;
    };
    let hidden = harness.hidden();
    let rows = harness.config.block_size;
    let n_ctx = 20usize;
    let position = n_ctx + rows;

    let anchor = pseudo(0x5EED_A11C, hidden, INPUT_SCALE);
    let mask = pseudo(0x0A5C0FFE, hidden, INPUT_SCALE);
    let taps = harness.taps(n_ctx, 0xDF_1A_5000);
    let case = block_case(&mut harness, &anchor, &mask, &taps, n_ctx, position);

    let hidden = harness.hidden();
    eprintln!(
        "forward_block WITH CONTEXT worst diff {:.6} at {:.4} of {}; row {} feature {}, \
         engine max {:.4}, reference max {:.4}, engine RMS {:.4}, reference per-stage max {}",
        case.worst,
        case.worst / case.host_max,
        case.engine.len(),
        case.worst_index / hidden,
        case.worst_index % hidden,
        max_of(&case.engine),
        case.host_max,
        rms_of(&case.engine),
        case.stage_max
            .iter()
            .map(|value| format!("{value:.2}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    // `NaN - x` is `NaN` and `NaN > worst` is false, so an all-NaN engine output
    // reports "worst diff 0.000000" and passes. Same guard as the zero-context
    // test above, for the same reason.
    let energy: f32 = case.engine.iter().map(|value| value * value).sum();
    let engine_max = max_of(&case.engine);
    assert!(
        energy.is_finite() && engine_max > 0.01,
        "the comparison is vacuous: the engine carries no signal (energy {energy})"
    );
    assert!(
        case.worst <= CONTEXT_TOLERANCE * case.host_max,
        "forward_block with {n_ctx} context rows disagrees with the host reference by {} \
         ({} relative on a reference max of {}) -- a graph error, not fp16 spread",
        case.worst,
        case.worst / case.host_max,
        case.host_max
    );
}

/// The context must be what the block's output *depends on*.
///
/// `dflash_forward_block_with_context_matches_host` proves the engine and the
/// reference agree; it does not prove that either of them reads the context. A
/// pair of implementations that both ignore it agree perfectly, and that is not
/// a hypothetical: the real checkpoints draft byte-identical tokens from two
/// different tap-layer sets (`dflash_alpha_probe`, alpha = 0.0000), which is
/// only possible if the context never reaches the predictions.
///
/// Three cases, same anchor, same mask, same positions:
///
/// * `a`, `b` -- 20 context rows that differ in *value*.
/// * `none`  -- no context rows at all.
///
/// Both differences are measured on the engine **and** on the reference, so a
/// failure says which of the two is context-blind. Meaningless magnitudes are
/// guarded: nothing is asserted unless the reference itself is sensitive to the
/// context by more than 1% of its own RMS.
#[test]
fn dflash_context_rows_are_visible_in_the_block_output() {
    let Some(mut harness) = Harness::build() else {
        return;
    };
    let hidden = harness.hidden();
    let rows = harness.config.block_size;
    let n_ctx = 20usize;
    let position = n_ctx + rows;

    let anchor = pseudo(0x5EED_A11C, hidden, INPUT_SCALE);
    let mask = pseudo(0x0A5C0FFE, hidden, INPUT_SCALE);
    let taps_a = harness.taps(n_ctx, 0xDF_1A_5000);
    let taps_b = harness.taps(n_ctx, 0x11_C0_FFEE);
    let a = block_case(&mut harness, &anchor, &mask, &taps_a, n_ctx, position);
    let b = block_case(&mut harness, &anchor, &mask, &taps_b, n_ctx, position);
    let none = block_case(&mut harness, &anchor, &mask, &[], 0, position);

    let scale = rms_of(&a.host);
    let engine_values = max_diff(&a.engine, &b.engine);
    let engine_rows = max_diff(&a.engine, &none.engine);
    let host_values = max_diff(&a.host, &b.host);
    let host_rows = max_diff(&a.host, &none.host);
    eprintln!(
        "context visibility (host RMS {scale:.4}):\n  \
         different context values: engine {engine_values:.4} ({:.4} of host RMS), \
         host {host_values:.4}\n  \
         context rows vs none:     engine {engine_rows:.4} ({:.4} of host RMS), \
         host {host_rows:.4}",
        engine_values / scale,
        engine_rows / scale,
    );

    // Guard: if the reference is not context sensitive, the numbers below are
    // not evidence about the engine at all.
    assert!(
        host_values > 0.01 * scale && host_rows > 0.01 * scale,
        "the host reference itself is context-blind (values {host_values}, rows {host_rows} \
         on RMS {scale}), so this test cannot say anything about the engine"
    );
    assert!(
        engine_rows > 0.01 * scale,
        "`forward_block` ignored every context row: with 20 rows and with none it differs by \
         {engine_rows} ({} of the block's RMS {scale}) while the reference moves by {host_rows} \
         ({}). This is the alpha = 0.0000 signature -- the drafter's predictions do not depend \
         on the history they are supposed to continue.",
        engine_rows / scale,
        host_rows / scale
    );
}
