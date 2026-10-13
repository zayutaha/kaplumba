//! `spec/dflash-round-30tps` D1/T5 — the buffer path must honour `perm_mode`.
//!
//! Before this field existed, `LinearBufferRequest` had no way to say which
//! EXL3 tile permutation the packed tensor was written with, and
//! `gemm_chunk_buffer_with_stats_region` hardcoded `perm_mode: 0`. That is the
//! path every DFlash drafter linear and the tied head go through. The
//! consequence: `DflashModel::set_permutation_mode` could be called, would
//! update `QuantizedLinear::permutation_mode`, and would change **nothing** —
//! the drafter's tile permutation therefore went unscored through three
//! specifications, because no test could observe the knob doing anything.
//!
//! This suite is the tripwire. It compares the same synthetic packed tensor
//! under `perm_mode` 0 and 1 through `gemm_chunk_buffer`. Two permutations of
//! the trellis must give two different outputs — if they do not, the field is
//! dead again and the whole D1/D2 investigation is measuring nothing.
//!
//! Deterministic and model-free: synthetic trellis bits, one GPU dispatch per
//! arm. No checkpoint, no `scripts/gpuq`, no 16 GB load. It does use the GPU,
//! so do not run it concurrently with a model benchmark.
//!
//! Run with:
//! ```text
//! cargo test -p kaplumba-engine --release --test buffer_perm_mode -- --nocapture
//! ```

use std::ffi::c_void;
use std::mem::size_of_val;

use half::f16;
use kaplumba_kernels::exl3::{Exl3Codebook, Exl3Rate};
use kaplumba_kernels::linear::{LinearBufferRequest, LinearKernels};
use metal::{Device, MTLResourceOptions};

/// `in_features` and `out_features` are both multiples of 16, and the seed is
/// fixed so the assertion below is about the code and not about the data.
const IN_FEATURES: usize = 256;
const OUT_FEATURES: usize = 128;
const SEED: u64 = 0x51A7_0BFF_EA7A_1234;

/// Uniform random 16-bit trellis words, one word per packed slot.
///
/// The *dequantised value* distribution of a calibrated checkpoint differs from
/// this, but the tile permutation is a **data movement**: mode 0 and mode 1 read
/// different `(k, n)` elements from the same words, so a permutation that
/// reaches the kernel shows up in the output regardless of the values.
fn packed_words(tiles: usize, rate: Exl3Rate) -> Vec<u16> {
    let mut state = SEED | 1;
    (0..tiles * rate.packed_words())
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as u16
        })
        .collect()
}

/// Narrow activations in [-1, 1): a Hadamard-transformed trellis is pathological
/// for white noise, and the fp16 accumulator would spend the run in denormals.
fn activations(queries: usize) -> Vec<u16> {
    let mut state = 0x2545_F491_4F6C_DD1D_u64;
    (0..queries * IN_FEATURES)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            f16::from_f32(((state >> 40) as f32 / (1u64 << 24) as f32) - 1.0).to_bits()
        })
        .collect()
}

/// Sign vectors: an EXL3 projection is `sign * w * x`, so both are required.
/// Alternating so that a dropped vector changes the output.
fn signs(len: usize) -> Vec<u16> {
    (0..len)
        .map(|index| f16::from_f32(if index % 3 == 0 { -1.0 } else { 1.0 }).to_bits())
        .collect()
}

/// One `gemm_chunk_buffer` call with `queries` rows, under `perm_mode`.
fn run(kernels: &LinearKernels, perm_mode: u32, queries: usize, rate: Exl3Rate) -> Vec<u16> {
    let device = Device::system_default().expect("Metal device");
    let tiles = (OUT_FEATURES / 16) * (IN_FEATURES / 16);
    let packed = packed_words(tiles, rate);
    let input = activations(queries);
    let input_signs = signs(IN_FEATURES);
    let output_signs = signs(OUT_FEATURES);

    let halfs = |values: &[u16]| {
        device.new_buffer_with_data(
            values.as_ptr() as *const c_void,
            size_of_val(values) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    };
    kernels
        .gemm_chunk_buffer(LinearBufferRequest {
            packed: &halfs(&packed),
            packed_offset: 0,
            input: &input,
            input_signs: &input_signs,
            output_signs: &output_signs,
            in_features: IN_FEATURES,
            out_features: OUT_FEATURES,
            rate,
            codebook: Exl3Codebook::Mcg,
            perm_mode,
        })
        .expect("gemm_chunk_buffer")
}

/// The assertion that would have caught the bug: **the two permutations are not
/// the same matrix.** If `perm_mode` ever stops reaching the kernel this fails
/// with a named message rather than silently reporting "no difference".
#[test]
fn the_buffer_path_honours_perm_mode() {
    let kernels = LinearKernels::new().expect("linear kernels");
    // k3 MCG is the target's rate; k4 is the drafter's. Both must be sensitive,
    // because both go through the same hardcoded `perm_mode: 0` that this test
    // exists to keep honest.
    for rate in [Exl3Rate::Integer(3), Exl3Rate::Integer(4)] {
        let queries = 4;
        let row_major = run(&kernels, 0, queries, rate);
        let permuted = run(&kernels, 1, queries, rate);
        assert_eq!(row_major.len(), queries * OUT_FEATURES);

        let differing = row_major
            .iter()
            .zip(&permuted)
            .filter(|(a, b)| a != b)
            .count();
        assert!(
            differing > 0,
            "perm_mode 0 and 1 produced IDENTICAL output for rate {rate:?} through \
             gemm_chunk_buffer: the buffer path is hardcoding the permutation again \
             and `set_permutation_mode` is a no-op (spec/dflash-round-30tps D1)"
        );

        // Not just "different": a permutation must move *most* of the entries,
        // not a couple at the edges. A single differing word would mean the
        // knob reaches one tile rather than the whole tensor.
        let fraction = differing as f64 / row_major.len() as f64;
        assert!(
            fraction > 0.5,
            "only {differing}/{} output words differ between permutations: {:.1}% of the \
             tensor. A permutation that reaches one tile rather than all of them is \
             the same bug wearing a different hat",
            row_major.len(),
            fraction * 100.0
        );
    }
}

/// The negative control for the assertion above. If the two arms could never
/// differ — say because the comparison ignored the field entirely — the test
/// before this would pass vacuously. So: same output compared against itself
/// must show no difference, which proves the comparison has teeth.
#[test]
fn identical_requests_do_not_differ() {
    let kernels = LinearKernels::new().expect("linear kernels");
    let rate = Exl3Rate::Integer(3);
    let a = run(&kernels, 0, 2, rate);
    let b = run(&kernels, 0, 2, rate);
    let differing = a.iter().zip(&b).filter(|(x, y)| x != y).count();
    assert_eq!(
        differing, 0,
        "two identical requests produced {differing} differing words: the comparison \
         in `the_buffer_path_honours_perm_mode` would then be measuring noise, and \
         a hardcoded permutation could hide inside it"
    );
}

/// `perm_mode` must reach the *sliced* entry point too, because that is the one
/// the drafter's tied head uses (`draft_logits`), and a head that silently
/// ignores the permutation is exactly the "head application has no oracle"
/// problem D3 is about.
#[test]
fn the_sliced_entry_point_honours_perm_mode() {
    let kernels = LinearKernels::new().expect("linear kernels");
    let device = Device::system_default().expect("Metal device");
    let rate = Exl3Rate::Integer(4);
    let queries = 2;
    let tiles = (OUT_FEATURES / 16) * (IN_FEATURES / 16);
    let packed = packed_words(tiles, rate);
    let input = activations(queries);
    let input_signs = signs(IN_FEATURES);
    let output_signs = signs(OUT_FEATURES);
    let halfs = |values: &[u16]| {
        device.new_buffer_with_data(
            values.as_ptr() as *const c_void,
            size_of_val(values) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    };
    let run_sliced = |perm_mode: u32| {
        kernels
            .gemm_chunk_buffer_sliced(
                LinearBufferRequest {
                    packed: &halfs(&packed),
                    packed_offset: 0,
                    input: &input,
                    input_signs: &input_signs,
                    output_signs: &output_signs,
                    in_features: IN_FEATURES,
                    out_features: OUT_FEATURES,
                    rate,
                    codebook: Exl3Codebook::Mcg,
                    perm_mode,
                },
                size_of_val(packed.as_slice()) as u64,
            )
            .expect("gemm_chunk_buffer_sliced")
    };
    let row_major = run_sliced(0);
    let permuted = run_sliced(1);
    let differing = row_major
        .iter()
        .zip(&permuted)
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        differing > 0,
        "gemm_chunk_buffer_sliced ignores perm_mode (rate {rate:?}): the drafter's tied \
         head is the caller, so D3's head oracle would be built on a permutation the \
         kernel never applied"
    );
}
