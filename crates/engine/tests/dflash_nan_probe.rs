//! spec/dflash-correctness T4 (diagnostic): localise a NaN to a stage.
//!
//! Exists because `dflash_golden`'s comparison is a `max(|a - b|)` over f32
//! values, and `NaN - x` is `NaN`, `NaN > worst` is false — so an all-NaN
//! engine output reports "worst diff 0.000000" and passes. The energy check in
//! that test now catches it, but this probe says *where*.

use std::path::PathBuf;

use half::f16;
use kaplumba_engine::dflash::{DflashConfig, DflashModel, DflashWeights};

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


fn bad(values: &[u16]) -> (usize, f32) {
    let mut count = 0usize;
    let mut first = f32::NAN;
    for bits in values {
        let value = f16::from_bits(*bits).to_f32();
        if value.is_nan() || value.is_infinite() {
            if count == 0 {
                first = value;
            }
            count += 1;
        }
    }
    (count, first)
}

#[test]
fn dflash_stage_probe_localises_nan() {
    let d = dir();
    if !d.join("model.safetensors").exists() {
        eprintln!("drafter checkpoint not present; skipping");
        return;
    }
    let config = DflashConfig::load(&d).expect("config");
    let device = metal::Device::system_default().expect("device");
    let weights = DflashWeights::load(&d, &device).expect("weights");
    let mut model = DflashModel::new(weights, config.clone()).expect("model");
    let hidden = config.hidden_size;
    let n_ctx = 4usize;
    let position = n_ctx;

    let taps: Vec<Vec<u16>> = config
        .target_layer_ids
        .iter()
        .enumerate()
        .map(|(index, _)| pseudo(0xDF_1A_5000 + index as u64, n_ctx * hidden, TAP_SCALE))
        .collect();
    let tap_refs: Vec<&[u16]> = taps.iter().map(|tap| tap.as_slice()).collect();
    let fused = model.fuse_context(&tap_refs, n_ctx).expect("fuse");
    eprintln!("fused   bad {:?} of {}", bad(&fused).0, fused.len());

    let positions: Vec<u32> = (0..n_ctx as u32).collect();
    let mut ctx_keys = Vec::new();
    let mut ctx_values = Vec::new();
    for layer in 0..config.num_layers {
        let (k, v) = model
            .project_context_kv(layer, &fused, n_ctx, &positions, position)
            .expect("ctx kv");
        eprintln!("layer {layer} ctx k bad {:?}, v bad {:?}", bad(&k).0, bad(&v).0);
        ctx_keys.push(k);
        ctx_values.push(v);
    }

    let anchor = pseudo(0x5EED_A11C, hidden, INPUT_SCALE);
    let mask = pseudo(0x0A5C0FFE, hidden, INPUT_SCALE);
    model.set_mask_embedding(&mask).expect("mask");
    let block = match model.forward_block(&kaplumba_engine::dflash::BlockContext {
        anchor_embedding: &anchor,
        ctx_keys: &ctx_keys,
        ctx_values: &ctx_values,
        n_ctx,
        position,
    }) {
        Ok(block) => block,
        Err(error) => {
            eprintln!("forward_block failed: {error}");
            return;
        }
    };
    eprintln!(
        "hidden  bad {:?} of {}",
        bad(&block.hidden).0,
        block.hidden.len()
    );
    for layer in 0..config.num_layers {
        eprintln!(
            "block k[{layer}] bad {:?}, v[{layer}] bad {:?}",
            bad(&block.block_keys[layer]).0,
            bad(&block.block_values[layer]).0
        );
    }

    // Walk the layers one at a time with an empty context, so the first layer
    // to produce a NaN is identifiable.
    let empty_k: Vec<u16> = Vec::new();
    let empty_v: Vec<u16> = Vec::new();
    for layer in 0..config.num_layers {
        let mut keys = Vec::with_capacity(config.num_layers);
        let mut values = Vec::with_capacity(config.num_layers);
        for index in 0..config.num_layers {
            if index <= layer {
                keys.push(ctx_keys[index].clone());
                values.push(ctx_values[index].clone());
            } else {
                keys.push(empty_k.clone());
                values.push(empty_v.clone());
            }
        }
        let _ = (&keys, &values);
    }
}
