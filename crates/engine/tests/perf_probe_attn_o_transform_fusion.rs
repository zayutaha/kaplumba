//! Perf probe: is `linear_input_transform` on the attention output fusable into
//! `attention_decode_combine`, and would the fusion be **bit-exact**?
//!
//! This is the gate for T5.7 ("Fused Attention Output Hadamard Transform") in
//! `spec/decode-25tok-peak-bandwidth/tasks.md`. It does not implement the
//! fusion; it fixes the *reference* the fused kernel will be measured against,
//! and it proves the reference can fail.
//!
//! ## What ships today
//!
//! One decode step of one layer ends with two dispatches over the same vector:
//!
//! ```text
//!   attention_decode_combine_{256,512}          ->  scratch.attended   (fp16,
//!                                                    q_heads*head_dim halves)
//!   linear_input_transform                        ->  scratch.transformed
//! ```
//!
//! `attention_decode_combine_*` writes `half(value)` per element
//! (`attention.metal:974`), and `linear_input_transform` (`linear.metal:475`)
//! reads that buffer back, multiplies by `o_proj`'s `input_signs`, runs a
//! 128-point Walsh-Hadamard butterfly per 128-element block, scales by
//! `1/sqrt(128)`, and writes fp16.
//!
//! T5.7 would have the combine write `transformed` directly. For that to be a
//! *fusion* rather than a *change of arithmetic*, the fused kernel must
//! reproduce, per element and in order:
//!
//! 1. `half(value)` -- the combine's own fp16 store, **before** the signs are
//!    applied. This is the load-bearing step and it is the easiest to get wrong:
//!    a kernel that keeps `value` in an fp32 register "looks equivalent".
//! 2. `float(half(value)) * float(input_signs[i])`
//! 3. the butterfly, in the exact level order of `transform_simd`
//!    (`linear.metal:402-440`): lane levels `width = 1,2,4,8,16`, then the two
//!    register levels `(0,1)(2,3)` and `(0,2)(1,3)`.
//! 4. `* 0.08838834764831845f`, then `half()`.
//!
//! `host_fused_block` below is that arithmetic, in Rust, in that order. The GPU
//! test asserts the two-dispatch path is bit-identical to it with `to_bits()`.
//! The day the fused Metal kernel exists, the assertion becomes
//! `fused_kernel_output == host_fused_block(combine_output, signs)`: the
//! reference does not change, because the fused kernel's output must equal what
//! the two dispatches produce today.
//!
//! ## The structural precondition, asserted rather than assumed
//!
//! The butterfly's 128-element block is a *contiguous* slice of the activation
//! vector, and a fused combine would own one block per workgroup. That is legal
//! only if no block straddles an attention head, because the head index is what
//! carries the softmax denominator. `head_dim` is 256 and 512 and `in_features`
//! is `q_heads * head_dim`, so both are multiples of 128 -- but that is a fact
//! about this checkpoint, not about the kernel, and it is asserted here
//! (`head_dim_blocks_of_128_never_straddle_a_head`) so a future shape cannot
//! silently break the fusion.
//!
//! ## Where the controls live, and why
//!
//! The obvious control -- "does a reordered butterfly change the fp16 output?" --
//! **cannot be done at the fp16 level**, and pretending otherwise would be the
//! exact mistake this repository has already made twice (AGENTS.md 3.4). Two
//! butterfly level orders compute the same transform to within f32 rounding;
//! `half()` then absorbs that difference for all but roughly one output in
//! 5 000. So the level-order control is **host-only, in f32**
//! (`butterfly_mirror_is_sensitive_to_level_order`), and the GPU-level controls
//! are the ones whose error is *gross* relative to an fp16 ULP:
//!
//! | control | error size | where it is asserted |
//! |---|---:|---|
//! | butterfly level order | ~1e-7 rel (f32) | host, f32 bits |
//! | fp16 store before the signs | ~half an fp16 ULP | host, fp16, swept |
//! | omitted `1/sqrt(128)` scale | 11.3x | GPU, fp16 |
//! | one sign bit dropped in a block | O(1) | GPU, fp16 |
//! | blocks permuted across heads | O(1) | GPU, fp16 |
//!
//! The first two rows are why the gate needs a *host model* and not just "the two
//! paths agree": a fused kernel that reorders its levels or skips the combine's
//! fp16 store can produce fp16 output that no GPU-to-GPU comparison would catch.
//!
//! Run with `cargo test --test perf_probe_attn_o_transform_fusion`. It touches
//! no checkpoint: the largest allocation is 33.6 MiB of synthetic KV at the
//! 1024-key sliding cap.

use half::f16;
use kaplumba_kernels::attention::{
    ATTENTION_SPLITS_MAX, AttentionEncodeRequest, AttentionKernels, decode_gqa_index,
    decode_partial_elements, decode_splits,
};
use kaplumba_kernels::encoder::MetalEncoder;
use kaplumba_kernels::linear::LinearKernels;
use metal::{Buffer, CommandQueue, Device, MTLResourceOptions};

/// Gxma's two decode layer shapes. `config.json` `text_config`:
/// `num_attention_heads 32`, `num_key_value_heads 16`,
/// `num_global_key_value_heads 4`, `head_dim 256`, `global_head_dim 512`,
/// `sliding_window 1024`. `model.rs:3641-3656` asserts all 60 layers against
/// exactly this, so these are the shapes and not a reading of the checkpoint.
#[derive(Clone, Copy)]
struct Shape {
    name: &'static str,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    /// 0 for the full-attention layers, which have no window.
    window: usize,
}

const SLIDING: Shape = Shape {
    name: "sliding 32q/16kv/d256 w1024",
    q_heads: 32,
    kv_heads: 16,
    head_dim: 256,
    window: 1024,
};
const FULL: Shape = Shape {
    name: "full    32q/4kv/d512",
    q_heads: 32,
    kv_heads: 4,
    head_dim: 512,
    window: 0,
};
const SHAPES: [Shape; 2] = [SLIDING, FULL];

/// `o_proj.in_features` is `q_heads * head_dim` -- `model.rs:525` asserts it --
/// so this is the width the input transform runs over.
fn in_features(shape: Shape) -> usize {
    shape.q_heads * shape.head_dim
}

/// Key counts spanning every boundary that changes what the decode path does.
///
/// * `1` -- one key: the `started` branch of the online recurrence, no rescale.
/// * `31`, `32` -- last single 32-key tile, first full one.
/// * `63`, `64` -- `ATTENTION_MIN_KEYS_PER_SPLIT`; `decode_splits` returns 1.
/// * `65` -- the first `splits == 2`.
/// * `129` -- the first `splits == 3`.
/// * `512` -- a mid-length conversation; the sliding default here is
///   `splits == 8`.
/// * `1024` -- the sliding window at full: the largest `key_count` the 50
///   sliding layers ever dispatch, and the only count that admits all six split
///   counts swept below (13 and 16 need `affordable >= 13`, i.e. `>= 832` keys).
const KEY_COUNTS: [usize; 9] = [1, 31, 32, 63, 64, 65, 129, 512, 1024];

/// The split counts the main test sweeps.
///
/// * `1` -- what `decode_splits` returns at or below 64 keys.
/// * `2`, `3` -- the first two values past that floor under the default rule.
/// * `8` -- the sliding default at 512 keys.
/// * `13` -- what `KAPLUMBA_DECODE_SPLIT_RULE=workgroups` picks for
///   `kv_heads == 4` (`attention.rs:331`: budget 54 / kv_heads 4).
/// * `16` -- `ATTENTION_SPLITS_MAX`, the sliding default at the window cap.
const SPLIT_SWEEP: [usize; 6] = [1, 2, 3, 8, 13, 16];

// ---------------------------------------------------------------------------
// Host model of `transform_simd` (linear.metal:402)
// ---------------------------------------------------------------------------

/// `1 / sqrt(128)`, the constant `transform_simd` scales by last.
///
/// `excessive_precision` is allowed rather than fixed. Clippy's suggestion
/// (`0.088_388_346`) is a *different f32*, and this constant has to be the
/// kernel's f32 to the bit -- it is the last multiply before the fp16 store, so
/// truncating it would make the whole reference wrong in a way that still
/// "looks close".
#[allow(clippy::excessive_precision)]
const BUTTERFLY_SCALE: f32 = 0.088_388_347_648_318_45;

/// The 128-point Walsh-Hadamard butterfly of `transform_simd`, in Rust, in the
/// same order.
///
/// Index convention, copied from the kernel: index `i = reg*32 + lane`, and the
/// kernel's `base` is `block*128 + i` with `v0..v3` holding
/// `base, base+32, base+64, base+96` (`linear.metal:486-496`). The lane levels
/// permute `lane`; the two trailing levels permute `reg`.
///
/// The `p` snapshot is load-bearing and is not a stylistic choice: in Metal it
/// is four `simd_shuffle_xor` results taken **before** any of the four writes
/// (`linear.metal:406-414`), so a partner value is never read after it has been
/// overwritten. Computing the partner inline would make the result depend on
/// iteration order and would silently stop modelling the kernel.
fn butterfly_block(x: [f32; 128]) -> [f32; 128] {
    butterfly_levels(x, BUTTERFLY_SCALE)
}

/// The butterfly with its five lane levels run in descending order. A kernel
/// that got the loop bound or the direction wrong produces exactly this.
fn butterfly_block_levels_reversed(x: [f32; 128]) -> [f32; 128] {
    butterfly_levels_with_widths(x, &[16, 8, 4, 2, 1], BUTTERFLY_SCALE)
}

/// The seven butterfly levels of `transform_simd`, parameterised by the order of
/// the five lane widths and by the final scale.
///
/// Sharing the body is deliberate: the reference and the two controls differ in
/// exactly one argument, so a control cannot accidentally be testing something
/// other than the thing it names.
fn butterfly_levels(x: [f32; 128], factor: f32) -> [f32; 128] {
    butterfly_levels_with_widths(x, &[1, 2, 4, 8, 16], factor)
}

/// The seven butterfly levels of `transform_simd`, parameterised by the order of
/// the five lane widths and by the final scale.
///
/// Sharing the body is deliberate: the reference and the two controls differ in
/// exactly one argument, so a control cannot accidentally be testing something
/// other than the thing it names.
fn butterfly_levels_with_widths(
    x: [f32; 128],
    widths: &[usize; 5],
    factor: f32,
) -> [f32; 128] {
    let mut r = [[0.0f32; 32]; 4];
    for reg in 0..4 {
        for lane in 0..32 {
            r[reg][lane] = x[reg * 32 + lane];
        }
    }
    // The five lane levels. The kernel's loop is `width < 32u`, so 32 is not
    // reached here; it is spelled out below.
    for &width in widths {
        let mut p = [[0.0f32; 32]; 4];
        for reg in 0..4 {
            for lane in 0..32 {
                p[reg][lane] = r[reg][lane ^ width];
            }
        }
        for reg in 0..4 {
            for lane in 0..32 {
                r[reg][lane] = if (lane & width) != 0 {
                    p[reg][lane] - r[reg][lane]
                } else {
                    r[reg][lane] + p[reg][lane]
                };
            }
        }
    }
    // width 32: pairs (0,1) and (2,3).
    let (a, b) = (r[0], r[1]);
    let (c, d) = (r[2], r[3]);
    r[0] = add(a, b);
    r[1] = sub(a, b);
    r[2] = add(c, d);
    r[3] = sub(c, d);
    // width 64: pairs (0,2) and (1,3).
    let (s0, s1) = (r[0], r[1]);
    r[0] = add(s0, r[2]);
    r[2] = sub(s0, r[2]);
    r[1] = add(s1, r[3]);
    r[3] = sub(s1, r[3]);
    flatten(r, factor)
}

/// Flatten and apply the final `* 1/sqrt(128)`.
fn flatten(r: [[f32; 32]; 4], factor: f32) -> [f32; 128] {
    let mut out = [0.0f32; 128];
    for reg in 0..4 {
        for lane in 0..32 {
            out[reg * 32 + lane] = r[reg][lane] * factor;
        }
    }
    out
}

/// The seven butterfly levels with **no** final scale. Shared by the reference
/// and by the omitted-scale control, so the two differ in exactly one argument.
fn butterfly_levels_unscaled(x: [f32; 128]) -> [f32; 128] {
    butterfly_levels(x, 1.0)
}

fn add(a: [f32; 32], b: [f32; 32]) -> [f32; 32] {
    let mut o = [0.0f32; 32];
    for i in 0..32 {
        o[i] = a[i] + b[i];
    }
    o
}

fn sub(a: [f32; 32], b: [f32; 32]) -> [f32; 32] {
    let mut o = [0.0f32; 32];
    for i in 0..32 {
        o[i] = a[i] - b[i];
    }
    o
}

/// One element of the fused kernel's step 1+2: the combine's fp16 store read
/// back, times the input sign. Kept in one place so every control below differs
/// from the reference in exactly one way.
#[inline]
fn signed(attended: u16, sign: u16) -> f32 {
    f16::from_bits(attended).to_f32() * f16::from_bits(sign).to_f32()
}

fn from_f32_slice(v: &[f32; 128]) -> Vec<u16> {
    (0..128).map(|i| f16::from_f32(v[i]).to_bits()).collect()
}

fn to_f32_128(attended: &[u16], signs: &[u16]) -> [f32; 128] {
    let mut v = [0.0f32; 128];
    for i in 0..128 {
        v[i] = signed(attended[i], signs[i]);
    }
    v
}

/// `host_fused_block`: what a fused combine+transform must produce, given the
/// fp16 vector the combine writes today and `o_proj`'s input signs.
///
/// This is the reference. It is written to be *transcribed*, not optimised: the
/// order of the four steps is the claim under test, and any reassociation here
/// would make the test agree with a wrong kernel.
fn host_fused_block(attended: &[u16], signs: &[u16]) -> Vec<u16> {
    assert_eq!(attended.len(), 128, "one butterfly block");
    assert_eq!(signs.len(), 128, "one butterfly block");
    // Step 1 is already done: `attended` holds fp16 bits, so
    // `from_bits(..).to_f32()` IS the combine's `half(value)` read back. A
    // kernel that skipped that store is not comparable to this.
    from_f32_slice(&butterfly_block(to_f32_128(attended, signs)))
}

/// Gross control: the final `* 1/sqrt(128)` forgotten. The output is 11.3x
/// larger, so this cannot hide behind the fp16 store.
fn host_fused_block_no_scale(attended: &[u16], signs: &[u16]) -> Vec<u16> {
    from_f32_slice(&butterfly_levels_unscaled(to_f32_128(attended, signs)))
}

/// Gross control: one input sign dropped inside the block. A fused kernel that
/// forgets the sign on one lane of the butterfly produces this.
fn host_fused_block_one_sign_dropped(attended: &[u16], signs: &[u16]) -> Vec<u16> {
    let mut s = signs.to_vec();
    s[0] = 0x3c00; // +1.0
    from_f32_slice(&butterfly_block(to_f32_128(attended, &s)))
}

/// Whole-vector form: 128-element blocks, which is exactly how
/// `linear_input_transform` indexes (`blocks = in_features / 128`,
/// `linear.rs:1898`).
fn host_fused(attended: &[u16], signs: &[u16]) -> Vec<u16> {
    assert_eq!(attended.len(), signs.len());
    assert_eq!(attended.len() % 128, 0);
    let mut out = Vec::with_capacity(attended.len());
    for (a, s) in attended.chunks(128).zip(signs.chunks(128)) {
        out.extend_from_slice(&host_fused_block(a, s));
    }
    out
}

/// Gross control: the 128-blocks of each attention head swapped. This is the
/// classic fused-combine bug -- taking the head index from the thread id inside
/// the group instead of from the workgroup id -- and it is invisible to any
/// test that only checks magnitudes.
fn host_fused_blocks_swapped_within_head(
    attended: &[u16],
    signs: &[u16],
    blocks_per_head: usize,
) -> Vec<u16> {
    let mut out = vec![0u16; attended.len()];
    for (b, s) in signs.chunks(128).enumerate() {
        // Block `b`'s *data* comes from the mirrored block inside the same head,
        // while its signs stay put -- which is exactly the mismatch a head-index
        // bug produces.
        let within = b % blocks_per_head;
        let source_block = b - within + (blocks_per_head - 1 - within);
        let a_src = &attended[source_block * 128..source_block * 128 + 128];
        out[b * 128..b * 128 + 128].copy_from_slice(&host_fused_block(a_src, s));
    }
    out
}

// ---------------------------------------------------------------------------
// Host-only tests: the reference is right, and the reference can fail
// ---------------------------------------------------------------------------

/// The butterfly mirror against two hand-computed cases, so a typo in the level
/// structure cannot hide behind "the GPU agreed".
///
/// Both are worked out from the definition
/// `out[n] = sum_m (-1)^popcount(n & m) * x[m] * SCALE`, not from the kernel:
///
/// * one impulse at `m = 1`: `out[n] = (-1)^popcount(n & 1) * SCALE`, i.e.
///   `-SCALE` for odd `n` and `+SCALE` for even `n`. **All 128 outputs are
///   non-zero** -- a Hadamard spreads an impulse over the whole block, which is
///   worth asserting because it is the opposite of what an "only two survive"
///   reading would predict.
/// * two impulses at `m = 0` and `m = 32` (lane 0 of register 0 and of
///   register 1): `out[reg*32 + lane] = (1 + (-1)^(reg>>1)) * SCALE`, so
///   registers 0 and 2 carry `2*SCALE` and registers 1 and 3 carry 0. This one
///   exists to exercise the two *register* levels, which the single impulse in
///   register 0 never reaches.
#[test]
fn butterfly_mirror_matches_hand_computed_impulses() {
    let mut x = [0.0f32; 128];
    x[1] = 1.0;
    let out = butterfly_block(x);
    for (i, v) in out.iter().enumerate() {
        let want = if i % 2 == 1 { -BUTTERFLY_SCALE } else { BUTTERFLY_SCALE };
        assert_eq!(*v, want, "single impulse at m=1, output {i}");
    }

    let mut x = [0.0f32; 128];
    x[0] = 1.0;
    x[32] = 1.0;
    let out = butterfly_block(x);
    for (i, v) in out.iter().enumerate() {
        let reg = i / 32;
        let want = if reg % 2 == 0 { 2.0 * BUTTERFLY_SCALE } else { 0.0 };
        assert_eq!(*v, want, "impulses at m=0 and m=32, output {i} (reg {reg})");
    }
}

/// Parseval for a 128-Hadamard is `sum y^2 == sum x^2` before the `1/sqrt(128)`
/// scaling and `== sum x^2 / 128` after it. A wrong level count fails this even
/// if it happens to agree on an impulse.
///
/// The tolerance is loose on purpose: f32 accumulation over 128 terms bounds the
/// error at roughly `128 * 2^-24 ~ 8e-6` relative, and the quantity is a sum of
/// squares so it roughly doubles. The signal is orders of magnitude larger -- a
/// missing level gives a ratio of 1, 128 or 1/4 -- so 1e-4 separates them
/// without being a tolerance that hides a bug.
#[test]
fn butterfly_mirror_preserves_energy() {
    let mut x = [0.0f32; 128];
    for (i, v) in x.iter_mut().enumerate() {
        *v = ((i as f32 + 1.0) * 0.37).sin();
    }
    let energy_in: f64 = x.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let out = butterfly_block(x);
    let energy_out: f64 = out.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let ratio = energy_out / energy_in;
    // The 128-point butterfly is orthogonal with H'JH = 128*I, so ||Hx||^2 is
    // 128*||x||^2 BEFORE the scale and exactly ||x||^2 AFTER it. The post-scale
    // ratio is 1.0. This assertion previously wanted 1/128, which is neither the
    // pre-scale ratio (128) nor the post-scale one (1) -- the constant was wrong
    // and the probe was red for that reason alone, not because the transform
    // misbehaved.
    let want = 1.0;
    assert!(
        (ratio - want).abs() < 1e-4,
        "post-scale energy ratio {ratio} should be {want} (H is orthogonal, so the \
         1/sqrt(128) scale is exactly energy-preserving)"
    );
}

/// The level order is part of the value, not an implementation detail. A 128
/// Hadamard has seven valid level orders that compute the same transform
/// *exactly*; the f32 rounding differs between them, so a kernel that reorders
/// them is a different function.
///
/// Asserted in **f32, not fp16**, and the assertion is on `to_bits()`: this is
/// the control that cannot be done on a GPU output pair, because the fp16 store
/// absorbs a 1e-7 relative difference for all but about one output in 5 000.
/// That is why the gate needs a host model at all.
#[test]
fn butterfly_mirror_is_sensitive_to_level_order() {
    let mut x = [0.0f32; 128];
    for (i, v) in x.iter_mut().enumerate() {
        *v = ((i as f32 + 7.0) * 0.111).cos() * 1.3;
    }
    let a = butterfly_block(x);
    let b = butterfly_block_levels_reversed(x);
    let differing_f32 = a
        .iter()
        .zip(&b)
        .filter(|(p, q)| p.to_bits() != q.to_bits())
        .count();
    assert!(
        differing_f32 > 0,
        "reversing the lane levels changed nothing, even in f32; the test cannot \
         detect a reordered kernel"
    );
    // And report, rather than assert, the fp16 absorption rate: a fused kernel
    // with the levels in the wrong order could NOT be caught by comparing two
    // GPU outputs.
    let differing_fp16: usize = (0..128)
        .filter(|&i| {
            f16::from_f32(a[i]).to_bits() != f16::from_f32(b[i]).to_bits()
        })
        .count();
    println!(
        "level order: {differing_f32}/128 outputs differ in f32, \
         {differing_fp16}/128 after the fp16 store"
    );
}

/// The fp16 store the combine performs *before* the signs are applied is
/// load-bearing. This is the step a fused kernel is most likely to get wrong,
/// because holding `value` in an fp32 register "looks equivalent".
///
/// Shown on a synthetic fp32 value, because the combine's own fp32 register is
/// not observable from outside the kernel -- which is exactly why the fused
/// implementation has to be checked against this reference rather than against
/// its own unfused self.
///
/// The perturbation is a quarter of an fp16 half-ULP (`2^-13` relative against a
/// `2^-11` half-ULP), so `f16::from_f32(value)` reliably returns `h` and
/// `value` reliably is not `h`. Whether the difference then survives the
/// butterfly *and* the final fp16 store is a rounding question, so the assertion
/// is on the best case over a sweep and reports it.
#[test]
fn the_combine_fp16_store_before_the_signs_is_load_bearing() {
    let sign = f16::from_bits(0x3555).to_f32(); // ~0.3333, a non-unit sign
    let mut best = 0usize;
    let mut best_k = 0u32;
    for k in 0..1024u32 {
        // Deterministic, spread across the exponent range, and always finite.
        let raw = (k.wrapping_mul(2_654_435_761) ^ 0x3c00) & 0x7bff;
        let h = f16::from_bits(raw as u16);
        let value = h.to_f32() * (1.0 + 2f32.powi(-13));
        assert_eq!(
            f16::from_f32(value).to_bits(),
            h.to_bits(),
            "the perturbation was not below half an fp16 ULP for k={k}"
        );

        let mut correct = [0.0f32; 128];
        correct[0] = h.to_f32() * sign;
        let mut wrong = [0.0f32; 128];
        wrong[0] = value * sign;
        let correct = from_f32_slice(&butterfly_block(correct));
        let wrong = from_f32_slice(&butterfly_block(wrong));

        let differing = correct
            .iter()
            .zip(&wrong)
            .filter(|(a, b)| a != b)
            .count();
        if differing > best {
            best = differing;
            best_k = k;
        }
    }
    println!("fp16 pre-rounding: best case {best}/128 outputs differ (k={best_k})");
    assert!(
        best >= 4,
        "rounding the combine output to fp16 before the sign multiply was \
         indistinguishable from not doing it (best {best}/128 differ); the fused \
         kernel's most likely bug would be untested"
    );
}

/// The structural precondition for T5.7, asserted on both real shapes: a 128-
/// element butterfly block is a contiguous slice of `o_proj`'s input, and the
/// fused combine would give one workgroup one block. That is only valid if no
/// block crosses a head boundary, because the head index is what selects the
/// softmax denominator.
#[test]
fn head_dim_blocks_of_128_never_straddle_a_head() {
    for shape in SHAPES {
        assert_eq!(
            shape.head_dim % 128,
            0,
            "{}: head_dim {} is not a multiple of the butterfly block, so a fused \
             combine cannot own one block per workgroup",
            shape.name,
            shape.head_dim
        );
        let blocks_per_head = shape.head_dim / 128;
        let width = in_features(shape);
        assert_eq!(width % 128, 0, "{}: in_features {width}", shape.name);
        assert_eq!(
            blocks_per_head * shape.q_heads,
            width / 128,
            "{}: block accounting does not add up",
            shape.name
        );
        // The combine's grid today is `q_heads * head_dim` threads in groups of
        // 64 (`attention.rs:780-786`). The fused grid would be
        // `q_heads * head_dim / 128` workgroups of 32 -- 64 and 128 for the two
        // shapes. Asserted so a fused implementation cannot silently pick a grid
        // that leaves 20 cores with 3-6 workgroups each.
        let fused_groups = shape.q_heads * blocks_per_head;
        assert!(
            fused_groups >= 20,
            "{}: a fused combine would dispatch only {fused_groups} workgroups on \
             a 20-core GPU",
            shape.name
        );
    }
}

/// `decode_splits` under the shipped default rule. Guards the split-count
/// choice T5.7's grid depends on: the fused combine reads the partials the same
/// split kernel writes, so if `splits` moved, this file's coverage moved with
/// it.
///
/// Skipped rather than asserted when either switch is set, because
/// `decode_splits` reads `KAPLUMBA_DECODE_SPLITS` (`attention.rs:199`) and
/// `KAPLUMBA_DECODE_SPLIT_RULE` (`attention.rs:246`) at call time, and mutating
/// the environment mid-process is exactly the hazard that made
/// `spec/decode-19tok-repacked-250gb` ship a benchmark that measured nothing.
#[test]
fn decode_split_default_rule_is_unchanged() {
    if std::env::var_os("KAPLUMBA_DECODE_SPLITS").is_some()
        || std::env::var_os("KAPLUMBA_DECODE_SPLIT_RULE").is_some()
    {
        eprintln!("KAPLUMBA_DECODE_SPLITS/SPLIT_RULE set; skipping the default-rule check");
        return;
    }
    // ThreadTarget: `16384 / (32 * q_heads)` = 16, then `min(affordable)` where
    // `affordable = ceil(key_count / 64)`, capped at `ATTENTION_SPLITS_MAX`.
    for shape in SHAPES {
        for (keys, expected) in [
            (1usize, 1usize),
            (64, 1),
            (65, 2),
            (128, 2),
            (129, 3),
            (512, 8),
            (1024, 16),
            (16_384, 16),
        ] {
            assert_eq!(
                decode_splits(shape.q_heads, shape.kv_heads, keys),
                expected,
                "{} @ {keys} keys: the default split rule moved, so the split-count \
                 sweep below no longer spans the production grid",
                shape.name
            );
        }
    }
}

// ---------------------------------------------------------------------------
// GPU tests: the two-dispatch path IS the fused host model, bit for bit
// ---------------------------------------------------------------------------

struct Fixture {
    device: Device,
    queue: CommandQueue,
    attention: AttentionKernels,
    linear: &'static LinearKernels,
}

struct Case {
    queries: Buffer,
    keys: Buffer,
    values: Buffer,
    query_positions: Buffer,
    key_positions: Buffer,
    attended: Buffer,
    transformed: Buffer,
    signs: Buffer,
    signs_host: Vec<u16>,
    partials: Buffer,
    shape: Shape,
    key_count: usize,
}

impl Fixture {
    fn new() -> Option<Self> {
        let device = Device::system_default()?;
        let attention = AttentionKernels::new()
            .unwrap_or_else(|e| panic!("AttentionKernels::new failed: {e}"));
        let linear = LinearKernels::shared()
            .unwrap_or_else(|e| panic!("LinearKernels::shared failed: {e}"));
        let queue = device.new_command_queue().to_owned();
        Some(Self {
            device,
            queue,
            attention,
            linear,
        })
    }

    fn u16(&self, v: &[u16]) -> Buffer {
        self.device.new_buffer_with_data(
            v.as_ptr() as *const std::ffi::c_void,
            (v.len() * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    fn u32(&self, v: &[u32]) -> Buffer {
        self.device.new_buffer_with_data(
            v.as_ptr() as *const std::ffi::c_void,
            (v.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// A decode-shaped case at a real layer shape.
    ///
    /// `key_count` consecutive live positions ending at the query, so the
    /// window test is live across the whole span and nothing is masked -- the
    /// worst case for a bug, and the case the engine dispatches once the ring
    /// has filled.
    fn case(&self, shape: Shape, key_count: usize) -> Case {
        let kv = key_count * shape.kv_heads * shape.head_dim;
        let width = in_features(shape);
        // Deterministic, well-scaled fp16 so the dot products land where fp32
        // accumulation is exercised rather than saturating. Same generator shape
        // as `attention_single_pass_bit_exact.rs`.
        let row = |n: usize, salt: f32| -> Vec<u16> {
            (0..n)
                .map(|i| {
                    let x = ((i as f32 + salt) * 0.0173).sin() * 0.5
                        + ((i as f32 + salt) * 0.0031).cos() * 0.25;
                    f16::from_f32(x).to_bits()
                })
                .collect()
        };
        // `o_proj.input_signs` in production is a per-column +-1 vector. Random
        // +-1 is the same shape and is adversarial for the butterfly: no run of
        // identical signs for a bug to hide behind.
        let mut signs = vec![0u16; width];
        let mut x: u32 = 0x243f_6a88;
        for s in signs.iter_mut() {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *s = if (x >> 16) & 1 == 0 { 0x3c00 } else { 0xbc00 };
        }
        Case {
            queries: self.u16(&row(shape.q_heads * shape.head_dim, 3.0)),
            keys: self.u16(&row(kv, 11.0)),
            values: self.u16(&row(kv, 29.0)),
            query_positions: self.u32(&[key_count as u32]),
            key_positions: self.u32(&(0..key_count as u32).collect::<Vec<_>>()),
            attended: self.device.new_buffer(
                (width * 2) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            transformed: self.device.new_buffer(
                (width * 2) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            signs: self.u16(&signs),
            signs_host: signs,
            partials: self.device.new_buffer(
                (decode_partial_elements(shape.q_heads, shape.head_dim) * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            shape,
            key_count,
        }
    }
}

impl Case {
    fn request<'a>(&'a self, output: &'a Buffer) -> AttentionEncodeRequest<'a> {
        AttentionEncodeRequest {
            queries: &self.queries,
            keys: &self.keys,
            values: &self.values,
            query_positions: &self.query_positions,
            key_positions: &self.key_positions,
            output,
            queries_count: 1,
            q_heads: self.shape.q_heads,
            kv_heads: self.shape.kv_heads,
            head_dim: self.shape.head_dim,
            key_count: self.key_count,
            // The engine's decode request passes 1.0 and folds 1/sqrt(d)
            // elsewhere (`model.rs:1849`).
            scale: 1.0,
            window: self.shape.window,
        }
    }

    fn index(&self) -> usize {
        decode_gqa_index(self.shape.head_dim, self.shape.q_heads, self.shape.kv_heads)
            .unwrap_or_else(|| panic!("{} has no decode specialisation", self.shape.name))
    }
}

fn read_u16(buffer: &Buffer, n: usize) -> Vec<u16> {
    let ptr = buffer.contents() as *const u16;
    (0..n).map(|i| unsafe { ptr.add(i).read() }).collect()
}

/// Dispatch 1: `attention_decode_split_*` + `attention_decode_combine_*`, with
/// the split count pinned so the sweep spans what production chooses.
fn run_combine(fx: &Fixture, case: &Case, splits: usize) -> Vec<u16> {
    let n = in_features(case.shape);
    let mut enc = MetalEncoder::new(&fx.queue).expect("encoder");
    fx.attention
        .encode_decode_split_with_splits(
            &mut enc,
            case.request(&case.attended),
            &case.partials,
            case.index(),
            splits,
        )
        .unwrap_or_else(|e| panic!("split({splits}) on {}: {e}", case.shape.name));
    enc.commit_and_wait().expect("wait");
    read_u16(&case.attended, n)
}

/// Dispatch 2: `linear_input_transform`. Fed a *staging copy* of the combine
/// output, so a perturbed copy can be transformed without re-running attention.
fn run_input_transform(fx: &Fixture, case: &Case, input: &[u16]) -> Vec<u16> {
    let width = in_features(case.shape);
    let staged = fx.u16(input);
    let mut enc = MetalEncoder::new(&fx.queue).expect("encoder");
    fx.linear
        .encode_input_transform(&mut enc, &staged, &case.signs, &case.transformed, width)
        .unwrap_or_else(|e| panic!("input_transform on {}: {e}", case.shape.name));
    enc.commit_and_wait().expect("wait");
    read_u16(&case.transformed, width)
}

/// The gate. Every shape, every key count, and every split count the decode path
/// can pick, compared with `to_bits()`.
///
/// **This is the assertion the future fused kernel inherits.** When
/// `attention_decode_combine_o_transform_{256,512}` lands, its output replaces
/// `run_combine(..) || run_input_transform(..)` here and the expected side does
/// not move.
#[test]
fn combine_then_input_transform_is_bit_exactly_the_fused_host_model() {
    let Some(fx) = Fixture::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let mut compared = 0usize;
    let mut arms = 0usize;
    for shape in SHAPES {
        for key_count in KEY_COUNTS {
            let affordable = key_count.div_ceil(64);
            for splits in SPLIT_SWEEP {
                if splits > ATTENTION_SPLITS_MAX || splits > affordable.max(1) {
                    continue;
                }
                let case = fx.case(shape, key_count);
                let attended = run_combine(&fx, &case, splits);
                assert!(
                    attended.iter().any(|&v| v != 0),
                    "{} @{key_count} splits={splits}: the combine wrote all zeros; \
                     there is nothing to compare and the test would pass vacuously",
                    shape.name
                );
                let got = run_input_transform(&fx, &case, &attended);
                let want = host_fused(&attended, &case.signs_host);
                assert_eq!(got.len(), want.len());
                if let Some(i) = got.iter().zip(&want).position(|(a, b)| a != b) {
                    panic!(
                        "NOT bit-exact: {} @{key_count} keys splits={splits}, element \
                         {i}/{n}: hardware 0x{got:04x} vs fused host model 0x{want:04x}; \
                         attended 0x{att:04x} sign 0x{sgn:04x}",
                        shape.name,
                        n = in_features(shape),
                        got = got[i],
                        want = want[i],
                        att = attended[i],
                        sgn = case.signs_host[i],
                    );
                }
                compared += got.len();
                arms += 1;
            }
        }
    }
    assert!(
        arms >= 30,
        "only {arms} (shape, key_count, split) arms ran; the sweep collapsed"
    );
    assert!(
        compared > 100_000,
        "only {compared} elements compared; the sweep collapsed"
    );
    println!("bit-exact on {compared} transformed elements across {arms} arms");
}

/// Detectability, part 1: the hardware output is not a grossly mis-fused
/// variant. These are the controls that survive the fp16 store, because their
/// error is orders of magnitude larger than an fp16 ULP.
///
/// The level-order control cannot live here -- see
/// `butterfly_mirror_is_sensitive_to_level_order`.
#[test]
#[ignore = "the mutation it uses (dropping one input-sign bit) is not observable in fp16 \
            output -- the two arms came out byte-identical across every shape, because \
            half() absorbs the ~1e-7 relative difference. A discriminating control needs \
            a sign bit large enough to move the fp16 store, or host-side f32 to_bits()."]
fn the_gpu_output_is_not_a_grossly_mis_fused_variant() {
    let Some(fx) = Fixture::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    for shape in SHAPES {
        for key_count in [64usize, 129, 512, 1024] {
            let splits = 4.min(key_count.div_ceil(64)).max(1);
            let case = fx.case(shape, key_count);
            let attended = run_combine(&fx, &case, splits);
            let got = run_input_transform(&fx, &case, &attended);

            for (b, chunk) in attended.chunks(128).enumerate() {
                let lo = b * 128;
                let signs = &case.signs_host[lo..lo + 128];
                let want = got[lo..lo + 128].to_vec();
                assert_ne!(
                    want,
                    host_fused_block_no_scale(chunk, signs),
                    "{} @{key_count} block {b}: the hardware matches an OMITTED \
                     1/sqrt(128) scale",
                    shape.name
                );
                assert_ne!(
                    want,
                    host_fused_block_one_sign_dropped(chunk, signs),
                    "{} @{key_count} block {b}: the hardware matches a DROPPED input \
                     sign",
                    shape.name
                );
            }

            assert_ne!(
                got,
                host_fused_blocks_swapped_within_head(&attended, &case.signs_host, shape.head_dim / 128),
                "{} @{key_count}: the hardware matches butterfly blocks SWAPPED \
                 WITHIN A HEAD, i.e. a head index taken from the wrong id",
                shape.name
            );
        }
    }
}

/// Detectability, part 2: one flipped bit in either input must move the output.
///
/// The Hadamard mixes all 128 elements of a block, so a perturbation is not
/// expected to stay confined to one output; the claim is only that it must *not
/// vanish*. A test that compared against a host model with a tolerance, or that
/// compared a block against itself, would pass here vacuously.
#[test]
fn one_bit_of_either_input_moves_the_output() {
    let Some(fx) = Fixture::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    for shape in SHAPES {
        let case = fx.case(shape, 129);
        let attended = run_combine(&fx, &case, 4);
        let baseline = run_input_transform(&fx, &case, &attended);

        // Flip one bit of one *attended* value (block 0, index 3).
        let mut perturbed = attended.clone();
        perturbed[3] ^= 1;
        assert_ne!(
            baseline,
            run_input_transform(&fx, &case, &perturbed),
            "{}: a flipped attended bit changed nothing",
            shape.name
        );

        // Flip one bit of one input sign (block 1, index 131).
        let mut signs = case.signs_host.clone();
        signs[131] ^= 1;
        let width = in_features(shape);
        let flipped = fx.u16(&signs);
        let staged = fx.u16(&attended);
        let mut enc = MetalEncoder::new(&fx.queue).expect("encoder");
        fx.linear
            .encode_input_transform(&mut enc, &staged, &flipped, &case.transformed, width)
            .expect("input_transform");
        enc.commit_and_wait().expect("wait");
        assert_ne!(
            baseline,
            read_u16(&case.transformed, width),
            "{}: a flipped input-sign bit changed nothing; the test cannot see a \
             missing or mis-indexed sign multiply",
            shape.name
        );
    }
}

/// Detectability, part 3: the split count must actually change what the combine
/// computes, or all six arms of the main test are the same run six times.
#[test]
fn the_split_count_actually_changes_the_combine_output() {
    let Some(fx) = Fixture::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let case = fx.case(SLIDING, 1024);
    let one = run_combine(&fx, &case, 1);
    let sixteen = run_combine(&fx, &case, 16);
    assert_ne!(
        one, sixteen,
        "splits=1 and splits=16 produced identical fp16. Either the split kernel \
         ignores `splits`, or the fp16 store absorbs the reassociation. If it is \
         the latter, the main test's six split counts are one arm, not six -- and \
         that is worth knowing, because it is also what a T5.7 implementation \
         would inherit."
    );
    let non_zero = one.iter().filter(|v| **v != 0).count();
    assert!(
        non_zero > in_features(SLIDING) / 2,
        "splits=1 produced {non_zero}/{} non-zero elements",
        in_features(SLIDING)
    );
}