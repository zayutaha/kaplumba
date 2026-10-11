//! spec/dflash-correctness T12 (R19): is `gemm_chunk` at M=16 fast enough?
//!
//! This is the measurement that decides whether T6's cache shape and T9's round
//! structure survive, which is why the spec puts it *before* T6.
//!
//! The concern is `fc`: it reads 32256 x 5376 x 4 bpw ~= 86.7 MB per block, and
//! the drafting pass runs it once per block. At the M4 Pro's ~250 GB/s that is a
//! ~350 ms floor if the weight read cannot be amortised across the block's 16
//! rows. The question is whether the M=16 path amortises it or repeats it 16x.
//!
//! Every one of the drafter's 36 linears plus `fc` is measured, because the
//! per-layer projections are what the round pays repeatedly; `fc` is paid once
//! per block.
//!
//! Run with:
//! ```text
//! cargo test -p kaplumba-engine --release --test dflash_gemm_m16 -- --nocapture
//! ```
//!
//! Host-side weight reads only -- no model load, no `scripts/gpuq`, no target
//! weights. It does use the GPU, so do not run it concurrently with a model
//! benchmark.

use std::path::PathBuf;
use std::time::Instant;

use half::f16;
use kaplumba_engine::dflash::{DflashConfig, DflashModel, DflashWeights};

fn dir() -> PathBuf {
    std::env::var("KAPLUMBA_DFLASH_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/Users/zayaantaha/.kaplumba/models/Drafter"))
}

/// Smooth synthetic rows, for the same reason `dflash_nan_probe` uses them:
/// white noise is pathological for a Hadamard-transformed EXL3 trellis.
fn pseudo(seed: u64, count: usize, scale: f32) -> Vec<u16> {
    let _ = seed;
    (0..count)
        .map(|index| {
            let t = index as f32;
            let value = (core::f32::consts::TAU * (t / 64.0)).sin()
                + 0.7 * (core::f32::consts::TAU * (t / 256.0)).sin()
                + 0.4 * (core::f32::consts::TAU * (t / 997.0)).sin()
                + 0.15 * (core::f32::consts::TAU * (t / 4096.0)).sin();
            f16::from_f32(value * scale).to_bits()
        })
        .collect()
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).expect("finite timings"));
    values[values.len() / 2]
}

#[test]
fn dflash_gemm_chunk_m16_throughput() {
    let dir = dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("drafter checkpoint not present at {dir:?}; skipping");
        return;
    }
    let device = metal::Device::system_default().expect("metal device");
    let weights = DflashWeights::load(&dir, &device).expect("drafter weights");
    let config = DflashConfig::load(&dir).expect("drafter config");
    let model = DflashModel::new(weights, config.clone()).expect("drafter model");

    // The block is `block_size` rows; the drafter only ever runs at that M.
    let rows = config.block_size;

    let mut names = vec!["fc".to_string()];
    for layer in 0..config.num_layers {
        for suffix in [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ] {
            names.push(format!("layers.{layer}.{suffix}"));
        }
    }

    const WARMUPS: usize = 3;
    const RUNS: usize = 9;

    let mut total_ms = 0.0_f64;
    let mut worst_ms = 0.0_f64;
    let mut worst_name = String::new();

    println!();
    println!("T12: gemm_chunk at M={rows} (median of {RUNS}, {WARMUPS} warmups)");
    println!(
        "{:<34} {:>10} {:>10} {:>12} {:>12}",
        "linear", "in x out", "MB read", "ms", "GB/s"
    );

    for name in &names {
        let (in_features, out_features) = model.gemv_shape(name);
        // What the kernel must read: the packed trellis plus both sign vectors.
        let weight_bytes = model.line_weight_bytes(name).expect("weight bytes") as f64;
        let megabytes = weight_bytes / 1e6;

        let input = pseudo(0x7A12_0000 ^ (name.len() as u64), rows * in_features, 0.05);

        for _ in 0..WARMUPS {
            model.gemm(name, &input).expect("warmup gemm");
        }
        let mut timings = Vec::with_capacity(RUNS);
        for _ in 0..RUNS {
            let start = Instant::now();
            model.gemm(name, &input).expect("gemm");
            timings.push(start.elapsed().as_secs_f64() * 1e3);
        }
        let ms = median(timings);
        let gbps = weight_bytes / (ms / 1e3) / 1e9;

        println!(
            "{name:<34} {:>5}x{:<4} {:>10.2} {:>12.3} {:>12.1}",
            in_features, out_features, megabytes, ms, gbps
        );

        if name.starts_with("layers.") {
            total_ms += ms;
        }
        if ms > worst_ms {
            worst_ms = ms;
            worst_name = name.clone();
        }
    }

    println!();
    println!("one block's 5 layers, all 35 projections: {total_ms:.2} ms");
    println!("slowest single projection: {worst_name} at {worst_ms:.3} ms");
    println!();

    // R19's decision rule. The floor for a whole drafting round is the sum of
    // the projections it pays; if that alone exceeds the token budget there is
    // no room left for the target verify pass and the shape has to change. The
    // spec's acceptance target is >= 2.4 tokens per target forward, so the
    // drafting pass must stay well under one target forward (~300 ms at 3.3
    // tok/s). 100 ms is the point at which the first-order arithmetic stops
    // working at all, so it is asserted rather than merely printed.
    assert!(
        total_ms < 100.0,
        "the 35 per-layer projections cost {total_ms:.2} ms at M={rows}, which \
         leaves nothing for the target verify pass; R19 says report the numbers \
         and stop rather than tune"
    );
}
