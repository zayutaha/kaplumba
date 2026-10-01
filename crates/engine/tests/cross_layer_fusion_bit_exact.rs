//! T3: the cross-layer fusion must be bit-identical to the two dispatches it
//! replaces.
//!
//! `finalize_post_ffn_and_next_input_norm_transform` fuses layer L's
//! `linear_finalize_and_post_ffn_norm_scale_sk2` with layer L+1's
//! `input_norm_and_qkv_transform`. Both halves are reductions, and reduction
//! order is observable: reassociating a float sum changes the low bits. So this
//! asserts `to_bits()` equality on every output element.
//!
//! **The reference is the two real kernels, not a CPU model.** Dispatching the
//! original pair and diffing against the fused kernel makes the comparison
//! against production code rather than against my understanding of it. An
//! earlier draft of this file reimplemented `transform_simd` in Rust; that is
//! strictly worse, because `transform_simd` is the most intricate thing in the
//! path (a 5-level shuffle butterfly plus two fixed passes) and reimplementing it
//! tests my transcription rather than the fusion.
//!
//! Two details make the bit-exactness non-obvious, and both are what this test
//! is for:
//!
//!   * Layer L's norm reduces over 1024 threads with a tree from 512, with
//!     threads 256..1023 contributing exact zeros. That is what makes it
//!     bit-identical to a 256-thread reduction.
//!   * Layer L+1's norm reduces over 256 threads with a tree from 128. NOT the
//!     same shape. Unifying them -- the obvious "simplification" -- reassociates
//!     one of the two sums.
//!
//! spec/decode-20tps: T3

use half::f16;
use kaplumba_kernels::encoder::MetalEncoder;
use kaplumba_kernels::linear::LinearKernels;
use metal::{Buffer, CommandQueue, Device, MTLResourceOptions};

/// The model's hidden width (`config.json`: hidden_size 5376). The kernels use
/// a fixed `threadgroup half[5376]`, so this is not a free parameter.
const WIDTH: usize = 5376;
const EPS: f32 = 1e-6;
/// Independent input vectors. Not tunable: see the test body for why a single
/// vector cannot detect a reduction-order change through fp16 outputs.
const SEEDS: u32 = 24;

/// `layer_scalar` is per-layer in the real model; one value is enough to catch
/// a dropped or misapplied scale.
const SCALE: f32 = 1.037_950_5;

/// Deterministic, non-trivial values. A zero or constant input would make the
/// reduction order unobservable, which is precisely the defect under test.
///
/// The exponent spans six values and the mantissa all ten bits, giving 6144
/// distinct representable magnitudes. An earlier version used a single exponent
/// and so collapsed to 2169 distinct outputs -- enough to pass bit-exactness but
/// not obviously enough to *discriminate* a reassociation, which is why the
/// caller asserts on the distinct count.
fn pattern(len: usize, seed: u32) -> Vec<u16> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            // Exponent field 12..17 keeps every value normal and well clear of
            // overflow; 12 is 2^-3 so nothing is denormal.
            let exponent = 12u16 + ((state >> 22) % 6) as u16;
            let mantissa = ((state >> 12) & 0x03FF) as u16;
            f16::from_bits((exponent << 10) | mantissa).to_bits()
        })
        .collect()
}

struct Fixtures {
    // Inputs, shared by both paths.
    raw: Buffer,
    output_signs: Buffer,
    post_ffn_norm: Buffer,
    residual: Buffer,
    next_input_norm: Buffer,
    next_q_signs: Buffer,
    // Outputs of the reference path (two dispatches).
    ref_hidden: Buffer,
    ref_normed: Buffer,
    ref_transformed: Buffer,
    // Outputs of the fused path (one dispatch).
    fused_hidden: Buffer,
    fused_normed: Buffer,
    fused_transformed: Buffer,
    host_raw: Vec<f32>,
    host_output_signs: Vec<u16>,
    host_post_ffn_norm: Vec<u16>,
    host_residual: Vec<u16>,
    host_next_input_norm: Vec<u16>,
    host_next_q_signs: Vec<u16>,
}

fn fixtures(device: &Device, seed: u32) -> Fixtures {
    let options = MTLResourceOptions::StorageModeShared;
    let u16s = |len: usize, salt: u32| -> (Buffer, Vec<u16>) {
        let host = pattern(len, seed ^ salt);
        let buffer = device.new_buffer((len * 2) as u64, options);
        unsafe {
            std::ptr::copy_nonoverlapping(host.as_ptr(), buffer.contents() as *mut u16, len);
        }
        (buffer, host)
    };

    // Split-K2 raw layout: `raw[i]` and `raw[WIDTH + i]` are the two splits,
    // which is what the `_sk2` finalize and the fused kernel both read.
    let mut host_raw = pattern(WIDTH * 2, seed ^ 0x1234_5678)
        .into_iter()
        .map(f16::from_bits)
        .map(f16::to_f32)
        .collect::<Vec<f32>>();
    for (i, value) in host_raw.iter_mut().enumerate() {
        *value *= 1.0 + (i % 977) as f32 * 1e-3;
    }
    let raw = device.new_buffer((host_raw.len() * 4) as u64, options);
    unsafe {
        std::ptr::copy_nonoverlapping(
            host_raw.as_ptr(),
            raw.contents() as *mut f32,
            host_raw.len(),
        );
    }

    let (output_signs, host_output_signs) = u16s(WIDTH, 0x9E37_79B9);
    let (post_ffn_norm, host_post_ffn_norm) = u16s(WIDTH, 0x85EB_CA6B);
    let (residual, host_residual) = u16s(WIDTH, 0xC2B2_AE35);
    let (next_input_norm, host_next_input_norm) = u16s(WIDTH, 0x27D4_EB2F);
    let (next_q_signs, host_next_q_signs) = u16s(WIDTH, 0x165667B1);
    let empty = || device.new_buffer((WIDTH * 2) as u64, options);

    Fixtures {
        raw,
        output_signs,
        post_ffn_norm,
        residual,
        next_input_norm,
        next_q_signs,
        ref_hidden: empty(),
        ref_normed: empty(),
        ref_transformed: empty(),
        fused_hidden: empty(),
        fused_normed: empty(),
        fused_transformed: empty(),
        host_raw,
        host_output_signs,
        host_post_ffn_norm,
        host_residual,
        host_next_input_norm,
        host_next_q_signs,
    }
}

fn read_u16(buffer: &Buffer) -> Vec<u16> {
    unsafe { std::slice::from_raw_parts(buffer.contents() as *const u16, WIDTH).to_vec() }
}

/// The three reference outputs and the three fused ones.
type Outputs = (
    Vec<u16>,
    Vec<u16>,
    Vec<u16>,
    Vec<u16>,
    Vec<u16>,
    Vec<u16>,
);

/// Run the two dispatches the fusion replaces, then the fused dispatch, on
/// identical inputs.
fn run(kernels: &LinearKernels, queue: &CommandQueue, fx: &Fixtures) -> Outputs {
    let mut encoder = MetalEncoder::new(queue).expect("encoder");
    encoder.set_max_dispatches(u32::MAX);

    // Path A: the production pair.
    kernels
        .encode_finalize_and_post_ffn_norm_scale_sk2(
            &mut encoder,
            &fx.raw,
            &fx.output_signs,
            &fx.post_ffn_norm,
            &fx.residual,
            &fx.ref_hidden,
            WIDTH,
            EPS,
            SCALE,
        )
        .expect("reference finalize");
    kernels
        .encode_input_norm_and_qkv_transform(
            &mut encoder,
            &fx.ref_hidden,
            &fx.next_input_norm,
            &fx.next_q_signs,
            &fx.ref_normed,
            &fx.ref_transformed,
            WIDTH,
            EPS,
        )
        .expect("reference input norm");
    encoder.commit_and_wait().expect("reference path");

    // Path B: the fused dispatch.
    let mut encoder = MetalEncoder::new(queue).expect("encoder");
    encoder.set_max_dispatches(u32::MAX);
    kernels
        .encode_cross_layer_finalize_and_norm(
            &mut encoder,
            &fx.raw,
            &fx.output_signs,
            &fx.post_ffn_norm,
            &fx.residual,
            &fx.fused_hidden,
            &fx.next_input_norm,
            &fx.next_q_signs,
            &fx.fused_normed,
            &fx.fused_transformed,
            WIDTH,
            EPS,
            SCALE,
        )
        .expect("fused dispatch");
    encoder.commit_and_wait().expect("fused path");

    (
        read_u16(&fx.ref_hidden),
        read_u16(&fx.ref_normed),
        read_u16(&fx.ref_transformed),
        read_u16(&fx.fused_hidden),
        read_u16(&fx.fused_normed),
        read_u16(&fx.fused_transformed),
    )
}

fn assert_eq_bits(label: &str, reference: &[u16], actual: &[u16]) {
    assert_eq!(reference.len(), actual.len(), "{label}: length differs");
    let mismatches: Vec<usize> = (0..reference.len())
        .filter(|&i| reference[i] != actual[i])
        .collect();
    if !mismatches.is_empty() {
        let first = mismatches[0];
        panic!(
            "{label}: {} of {} elements differ. First at {first}: \
             reference 0x{:04X} = {:?}, fused 0x{:04X} = {:?}",
            mismatches.len(),
            reference.len(),
            reference[first],
            f16::from_bits(reference[first]),
            actual[first],
            f16::from_bits(actual[first]),
        );
    }
}

/// The gate. All three fused outputs must match the two-dispatch path bit for
/// bit: the residual stream (`hidden`), the next layer's `normed`, and the
/// next layer's q-input transform.
///
/// ## Why this loops over seeds
///
/// Every observable output of this kernel is fp16. A reassociated f32 sum
/// therefore does not have to change the stored result: the rounding can absorb
/// the difference. That is not hypothetical — a sabotage which rewrote layer
/// L+1's reduction to use layer L's 1024-thread/512-tree shape (instead of its
/// own 256-thread/128-tree shape) **passed** a single-vector version of this
/// test. See the implementation notes.
///
/// So one vector proves nothing. A hundred thousand fp16 elements across
/// `SEEDS` independent vectors is a different claim, and it is the one that
/// makes "the reduction order is preserved" mean something.
#[test]
fn cross_layer_fusion_is_bit_identical_to_the_two_dispatches_it_replaces() {
    let Some(device) = Device::system_default() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let queue = device.new_command_queue();
    let kernels = LinearKernels::shared().expect("shared kernels");

    let mut worst_distinct = usize::MAX;
    for seed in 0..SEEDS {
        let fx = fixtures(&device, seed.wrapping_mul(0x9E37_79B9));
        let out = run(&kernels, &queue, &fx);
        assert_eq_bits(
            &format!("seed {seed} residual stream (hidden)"),
            &out.0,
            &out.3,
        );
        assert_eq_bits(
            &format!("seed {seed} next layer normed"),
            &out.1,
            &out.4,
        );
        assert_eq_bits(
            &format!("seed {seed} next layer q-input transform"),
            &out.2,
            &out.5,
        );

        // A guard against the test passing because both paths produced the same
        // uninformative constant. If the inputs were degenerate, a
        // reduction-order change would be invisible, which is the exact failure
        // this file exists to catch.
        let distinct = out
            .1
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>()
            .len();
        assert!(
            distinct > WIDTH / 4,
            "seed {seed}: the reference normed output has only {distinct} \
             distinct values out of {WIDTH}, so this input may not discriminate \
             a reduction-order change"
        );
        worst_distinct = worst_distinct.min(distinct);
    }
    eprintln!(
        "T3: {SEEDS} vectors x {WIDTH} elements compared bit-exactly; \
         fewest distinct normed values in any vector = {worst_distinct}"
    );
}