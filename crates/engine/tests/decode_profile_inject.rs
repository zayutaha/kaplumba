//! The injected-cost check, on the real engine decode path, in its own process.
//!
//! `tests/decode_profile_engine.rs` proves the engine's brackets enclose one
//! decode step. This proves the thing that makes those brackets worth reading: a
//! deliberately slow dispatch encoded *inside* the step moves the instrument by
//! about what it costs, with the cost established by the host clock rather than
//! by the instrument.
//!
//! AGENTS.md §3 is the reason this is a separate binary. The engine caches
//! `KAPLUMBA_DECODE_INJECT` on first read for the life of the process, so the
//! clean arm and the injected arm cannot both live in one test binary: the
//! second would inherit the first's value and the comparison would silently
//! become "injected vs injected". Splitting the process is the only way to keep
//! the two arms honest.
//!
//! The size of the injection is fixed at 24 dispatches of a dependent FMA loop.
//! It is deliberately large relative to this model's own step: a two-layer
//! synthetic step is a couple of milliseconds, so a small injection would sit
//! inside the noise and the check would pass for the wrong reason.

use std::ffi::c_void;
use std::mem::size_of;
use std::time::Instant;

use half::f16;
use kaplumba_engine::decode_profile;
use kaplumba_engine::exl3::pack_integer_states;
use kaplumba_engine::model::{
    DecoderLayer, DecoderLayerWeights, LayerConfig, QuantizedLinear, RopeMode, TextModel,
};
use kaplumba_kernels::exl3::{Exl3Codebook, Exl3Rate};
use metal::{Device, MTLResourceOptions};

const HIDDEN: usize = 128;
const INTER: usize = 256;
const VOCAB: usize = 256;
const MAX_SEQUENCE: usize = 512;
/// Dispatches added per decode step. Sized so the injection is several times the
/// clean step's cost and therefore many times its noise.
const INJECT_DISPATCHES: &str = "24";
/// Dependent-FMA trip count per injected dispatch.
const INJECT_ITERS: &str = "200000";

fn half(value: f32) -> u16 {
    f16::from_f32(value).to_bits()
}

fn value_at(index: usize, offset: f32) -> u16 {
    half(((index as f32 + offset) * 0.071).sin())
}

fn gpu_linear(
    device: &Device,
    in_features: usize,
    out_features: usize,
    seed: u32,
) -> QuantizedLinear {
    let mut packed = Vec::new();
    for tile in 0..(in_features / 16) * (out_features / 16) {
        let mut states = vec![0_u16; 256];
        let mut value = seed.wrapping_add(tile as u32 * 7919);
        for state in &mut states {
            value = value.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *state = (value & 0x7) as u16;
        }
        packed.extend(pack_integer_states(&states, 3).unwrap().0);
    }
    let input_signs = vec![half(0.01); in_features];
    let output_signs = vec![half(0.01); out_features];
    let buffer = device.new_buffer_with_data(
        packed.as_ptr() as *const c_void,
        (packed.len() * size_of::<u16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    QuantizedLinear::from_metal_buffer(
        in_features,
        out_features,
        Exl3Rate::Integer(3),
        Exl3Codebook::Mcg,
        buffer,
        0,
        Some(packed.len()),
        input_signs,
        output_signs,
    )
    .unwrap()
}

fn layer(device: &Device, full: bool, seed: u32) -> DecoderLayer {
    let config = LayerConfig {
        hidden_size: HIDDEN,
        intermediate_size: INTER,
        q_heads: 2,
        kv_heads: 1,
        head_dim: HIDDEN,
        max_sequence_len: MAX_SEQUENCE,
        full_attention: full,
        sliding_window: if full { 0 } else { 3 },
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        rope_mode: RopeMode::Default,
        rope_partial_factor: 1.0,
    };
    let q_dim = config.q_heads * config.head_dim;
    let kv_dim = config.kv_heads * config.head_dim;
    let weights = DecoderLayerWeights {
        q_proj: gpu_linear(device, HIDDEN, q_dim, seed),
        k_proj: gpu_linear(device, HIDDEN, kv_dim, seed + 1),
        v_proj: if full {
            None
        } else {
            Some(gpu_linear(device, HIDDEN, kv_dim, seed + 2))
        },
        o_proj: gpu_linear(device, q_dim, HIDDEN, seed + 3),
        gate_proj: gpu_linear(device, HIDDEN, INTER, seed + 4),
        up_proj: gpu_linear(device, HIDDEN, INTER, seed + 5),
        down_proj: gpu_linear(device, INTER, HIDDEN, seed + 6),
        q_norm: (0..config.head_dim)
            .map(|i| value_at(i, seed as f32))
            .collect(),
        k_norm: (0..config.head_dim)
            .map(|i| value_at(i, seed as f32 + 1.0))
            .collect(),
        input_norm: (0..HIDDEN)
            .map(|i| value_at(i, seed as f32 + 2.0))
            .collect(),
        post_attention_norm: (0..HIDDEN)
            .map(|i| value_at(i, seed as f32 + 3.0))
            .collect(),
        pre_ffn_norm: (0..HIDDEN)
            .map(|i| value_at(i, seed as f32 + 4.0))
            .collect(),
        post_ffn_norm: (0..HIDDEN)
            .map(|i| value_at(i, seed as f32 + 5.0))
            .collect(),
        layer_scalar: 0.94,
    };
    DecoderLayer::new(config, weights).unwrap()
}

fn synthetic_model() -> TextModel {
    let device = Device::system_default().expect("a Metal device");
    let embedding: Vec<u16> = (0..VOCAB * HIDDEN).map(|i| value_at(i, 0.5)).collect();
    let embedding_buffer = device.new_buffer_with_data(
        embedding.as_ptr() as *const c_void,
        (embedding.len() * size_of::<u16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let layers = vec![layer(&device, true, 11), layer(&device, false, 29)];
    let final_norm: Vec<u16> = (0..HIDDEN).map(|i| value_at(i, 7.5)).collect();
    let lm_head = gpu_linear(&device, HIDDEN, VOCAB, 101);
    TextModel::from_metal_parts(embedding_buffer, layers, final_norm, lm_head, 30.0, None).unwrap()
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}

#[test]
fn an_injected_dispatch_inside_a_real_decode_step_is_resolved() {
    decode_profile::configure_for_tests(Some(decode_profile::Mode::Step), Some(2));
    // Set before the first decode step, because the engine caches this on first
    // read. `Injected` is the process's whole identity.
    //
    // `unsafe` because the test harness has other threads alive. That is sound
    // here only because this binary has exactly one test and nothing reads the
    // environment concurrently: there is no other thread in this process that
    // could be looking at `environ` at this moment, and no thread is created
    // until `Device::system_default` below. A test that ran beside another
    // setting or reading an env var would not be, which is the other half of
    // why this file holds one test.
    unsafe {
        std::env::set_var("KAPLUMBA_DECODE_INJECT", INJECT_DISPATCHES);
        std::env::set_var("KAPLUMBA_DECODE_INJECT_ITERS", INJECT_ITERS);
    }
    let injected: usize = INJECT_DISPATCHES.parse().expect("dispatch count");

    let mut model = synthetic_model();
    let prompt: Vec<u32> = (0..8).map(|index| index % (VOCAB as u32 - 2) + 1).collect();
    let positions: Vec<u32> = (0..8).collect();
    model.forward_buffer(&prompt, &positions).expect("prefill");

    // A pre-roll, discarded. Without it the first measured steps carry the cold
    // clock ramp, which on this box is worth tens of percent and would be
    // indistinguishable from the injection.
    for step in 0..8 {
        let _ = Instant::now();
        model
            .forward_buffer(&[7], &[(8 + step as u32) % MAX_SEQUENCE as u32])
            .expect("warm step");
    }
    decode_profile::reset_for_tests();

    let mut spans = Vec::new();
    let mut walls = Vec::new();
    for step in 0..8 {
        let started = Instant::now();
        model
            .forward_buffer(&[7], &[(16 + step as u32) % MAX_SEQUENCE as u32])
            .expect("decode step");
        walls.push(started.elapsed().as_secs_f64() * 1e3);
        let span = decode_profile::last_span_for_tests();
        if span.is_finite() {
            spans.push(span);
        }
    }
    assert!(
        spans.len() >= 6,
        "only {} steps recorded: {spans:?}",
        spans.len()
    );
    let span_median = median(&spans);
    let wall_median = median(&walls);
    eprintln!(
        "inject: with {injected} injected dispatches of {INJECT_ITERS} iters per step, \
         span {span_median:.4} ms and host wall {wall_median:.4} ms over {} steps",
        spans.len()
    );

    // The injection has to be visible, and visible at the right size. The host
    // clock is the yardstick: the injection's cost is whatever the region took
    // on the CPU, which does not depend on the GPU timestamps at all.
    //
    // The lower bound is the one that matters for a profiler: a span that did not
    // move at all would be a span measuring something outside the step, which is
    // the original bug in a different guise. `SPAN_FLOOR` is the clean step's
    // measured cost plus room for the model's own spread, taken from
    // `tests/decode_profile_engine.rs` and the pre-roll above.
    const CLEAN_STEP_MS: f64 = 3.0;
    const SPAN_FLOOR: f64 = 2.0 * CLEAN_STEP_MS;
    assert!(
        span_median > SPAN_FLOOR,
        "the instrument reports {span_median:.4} ms for a step carrying {injected} \
         injected dispatches, which is not more than a clean step's worth: work \
         inside the brackets is not being seen"
    );
    assert!(
        span_median > 0.8 * wall_median,
        "the instrument reports {span_median:.4} ms against a host wall of \
         {wall_median:.4} ms for the same step: the injected GPU time is not \
         reaching the measurement"
    );
    // And it must not be absurd: a span far beyond the host wall would mean the
    // brackets have slipped and are now covering more than one step.
    assert!(
        span_median < 1.1 * wall_median + 1.0,
        "the instrument reports {span_median:.4} ms against a host wall of \
         {wall_median:.4} ms: the brackets cover more than the step"
    );
    let report = decode_profile::report_for_tests();
    assert!(report.contains("dp_bad_steps=0"), "{report}");
}
