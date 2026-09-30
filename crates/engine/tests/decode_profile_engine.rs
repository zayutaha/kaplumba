//! The decode instrument, on the real engine decode path.
//!
//! `tests/decode_profile_gpu.rs` proves the clock with a synthetic burn kernel.
//! This proves the *wiring*: that `TextModel::forward_buffer`'s two instrumented
//! boundaries really do bracket one decode step of a real forward, and that a
//! prefill of any length cannot leak into a decode step's span.
//!
//! The model is a two-layer synthetic in the shape of
//! `tests/forward_buffer_parity.rs` -- tens of microseconds per step. It cannot
//! stand in for a 31B step's cost, and this test does not try: a model this
//! small has nothing like the memory behaviour, dispatch count, or kernel mix of
//! a real decode step. What it can do is fail if the brackets are in the wrong
//! place, and misplaced brackets are exactly the failure this project has already
//! shipped once -- a profiler that printed the prefill's table and was read as a
//! decode table.
//!
//! One `#[test]` on purpose: the engine caches the mode and the warmup count on
//! first read for the life of the process, so a second test in this binary would
//! get this one's values. The injected arm lives in `decode_profile_inject.rs`,
//! which is a second process.
//!
//! Scaling with token count, the noise floor, and agreement with the real
//! model's 123 ms/token are measured against Gxma and belong in
//! `spec/engine-performance/implementation_notes.md`; a two-layer model has no
//! business making those claims.

use std::ffi::c_void;
use std::mem::size_of;

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

fn half(value: f32) -> u16 {
    f16::from_f32(value).to_bits()
}

fn value_at(index: usize, offset: f32) -> u16 {
    half(((index as f32 + offset) * 0.071).sin())
}

/// A quantized linear whose weights also live in a Metal buffer, which
/// `forward_buffer` requires of every projection it dispatches.
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

/// A prefill of `rows` tokens, then `steps` one-token forwards, returning the
/// instrument's most recent per-step span after each.
///
/// The KV cache is reset first: a prefill has to begin at the cache's current
/// position, so a second call in the same model would otherwise be rejected with
/// "positions must begin at the current cache position". That reset is also what
/// makes the two arms comparable -- the second one is not inheriting a longer
/// context from the first.
fn decode_after_prefill(model: &mut TextModel, rows: u32, steps: usize) -> Vec<f64> {
    model.reset_gpu_caches();
    let prompt: Vec<u32> = (0..rows)
        .map(|index| index % (VOCAB as u32 - 2) + 1)
        .collect();
    let positions: Vec<u32> = (0..rows).collect();
    model.forward_buffer(&prompt, &positions).expect("prefill");
    let mut spans = Vec::with_capacity(steps);
    for step in 0..steps {
        model
            .forward_buffer(&[7], &[(rows + step as u32)])
            .expect("decode step");
        spans.push(decode_profile::last_span_for_tests());
    }
    spans
}

#[test]
fn the_engine_brackets_a_real_decode_step_and_excludes_the_prefill() {
    decode_profile::configure_for_tests(Some(decode_profile::Mode::Step), Some(2));
    let mut model = synthetic_model();

    // 1. Every step produces a plausible interval. A zero or negative one is the
    //    shape of the original bug: boundaries committed that enclosed nothing,
    //    so the "per-kernel" table was a list of small numbers that never grew
    //    with the work.
    //
    //    `decode_after_prefill` starts each batch with a prefill, and a prefill
    //    is never instrumented, so the accumulator only sees the decode steps.
    let spans = decode_after_prefill(&mut model, 6, 8);
    let recorded: Vec<f64> = spans.iter().copied().filter(|s| s.is_finite()).collect();
    assert_eq!(
        recorded.len(),
        6,
        "6 of the 8 decode steps should be past the 2-step warmup, and every one \
         of them should be recorded: {spans:?}"
    );
    for (index, span) in spans.iter().enumerate() {
        if span.is_finite() {
            assert!(
                *span > 0.0,
                "decode step {index} measured {span} ms: the brackets enclose nothing"
            );
        }
    }
    let report = decode_profile::report_for_tests();
    assert!(
        report.contains("dp_bad_steps=0"),
        "a decode step produced an implausible GPU interval:\n{report}"
    );
    eprintln!(
        "engine: a 2-layer synthetic step is {median:.4} ms",
        median = median(&recorded)
    );

    // 2. Exclusion, checked the unconfounded way. Comparing step spans across
    //    two prefill lengths does NOT work as an exclusion probe here, and the
    //    reason is worth recording: the measured step legitimately gets more
    //    expensive when the KV window it attends over is longer (2.15 ms after 2
    //    tokens, 3.33 ms after 40), so a span that moved with prefill length
    //    would be ambiguous between "includes the prefill" and "attends over more
    //    keys". Both readings are real work, only one is a bug.
    //
    //    The unambiguous check is against the host clock for the *same* step: the
    //    span is a difference of two GPU timestamps taken at the step's own
    //    boundaries, so it cannot exceed the wall time of the region the caller
    //    timed. A prefill leaking in would push it past that, by roughly the
    //    prefill's whole cost.
    //
    //    The magnitude version -- a 466 ms block against a 38 ms decode total,
    //    0.1% shift -- is `tests/decode_profile_gpu.rs`, which can afford a
    //    prefill 12x a step. This one runs against the engine's own brackets,
    //    which is what the synthetic kernel cannot check.
    for rows in [2_u32, 40, 200] {
        model.reset_gpu_caches();
        let prefill_prompt: Vec<u32> = (0..rows)
            .map(|index| index % (VOCAB as u32 - 2) + 1)
            .collect();
        let prefill_positions: Vec<u32> = (0..rows).collect();
        let prefill_started = std::time::Instant::now();
        model
            .forward_buffer(&prefill_prompt, &prefill_positions)
            .expect("prefill");
        let prefill_ms = prefill_started.elapsed().as_secs_f64() * 1e3;

        let mut ratios = Vec::new();
        for step in 0..4 {
            let started = std::time::Instant::now();
            model
                .forward_buffer(&[7], &[(rows + step as u32)])
                .expect("decode step");
            let wall_ms = started.elapsed().as_secs_f64() * 1e3;
            let span = decode_profile::last_span_for_tests();
            if span.is_finite() {
                ratios.push((span, wall_ms));
            }
        }
        assert!(
            !ratios.is_empty(),
            "no step was recorded after a {rows}-token prefill"
        );
        let span_median = median(&ratios.iter().map(|(s, _)| *s).collect::<Vec<_>>());
        let wall_median = median(&ratios.iter().map(|(_, w)| *w).collect::<Vec<_>>());
        eprintln!(
            "engine: after a {rows}-token prefill ({prefill_ms:.3} ms of host time), a \
             step measures span {span_median:.4} ms against host wall {wall_median:.4} ms"
        );
        assert!(
            span_median <= wall_median * 1.02 + 0.2,
            "after a {rows}-token prefill the step's span is {span_median:.4} ms but \
             the host spent only {wall_median:.4} ms on that step: the span covers \
             more than the step"
        );
    }
}
