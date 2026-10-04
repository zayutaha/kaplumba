//! T4 (`spec/prefill-120tps`): the GEMM-MMA tile variants against a CPU
//! reference.
//!
//! `crates/kernels/tests/tile_mma_variants.rs` proves the variants agree with
//! each other; that test cannot see a *shared* error, because all four
//! kernels could be wrong in the same way and still be identical. This file
//! is the other half: every variant is compared against `reference_matrix` +
//! an fp32 matmul built from the same EXL3 decode helpers that pack the
//! weights, and the worst normalised error is reported and bounded.
//!
//! Three things it pins, in order of how easily they are lost:
//!
//! 1. **Numerics.** Worst normalised error (`max|actual - expected| /
//!    max|expected|`) per variant must stay under `VARIANT_TOLERANCE`. The
//!    bound is set from the measured baseline, not guessed -- see the
//!    constant.
//!
//! 2. **Dispatch.** Every assertion is paired with
//!    `last_gemm_mma_variant()`, so a variant that silently fell back to the
//!    baseline reports itself instead of passing as "the variant agreed". An
//!    output-only test here is exactly the vacuous pass that shipped
//!    `8f1a863` (AGENTS.md §3).
//!
//! 3. **The canary row.** A 128-row tile that stores rows it does not own
//!    scribbles past `queries * out_features`, which reading the result back
//!    would never reveal. `run` writes a guard and checks it after every
//!    dispatch.
//!
//! Verified to fail under sabotage (recorded in `spec/prefill-120tps/
//! notes_T3_T4.md`): breaking the 4sg accumulator-to-column mapping makes
//! `every_variant_matches_the_cpu_reference` fail on the 4sg row with a
//! normalised error orders of magnitude over the bound; clearing the
//! dispatch store makes `every_variant_matches_the_cpu_reference` fail on
//! the identity assert while the numbers still match -- which is precisely
//! why the identity assert is there.

use std::ffi::c_void;
use std::mem::{size_of, size_of_val};
use std::sync::Mutex;

use half::f16;
use kaplumba_engine::exl3::{
    BitRate, Codebook, SignVector, apply_hadamard_matrix, decode_state, pack_fractional_states,
    pack_integer_states, unpack_trellis, unpermute_tile,
};
use kaplumba_kernels::encoder::MetalEncoder;
use kaplumba_kernels::exl3::{Exl3Codebook, Exl3Rate};
use kaplumba_kernels::linear::{
    GEMM_MMA_VARIANT_4SG, GEMM_MMA_VARIANT_128X64, GEMM_MMA_VARIANT_REGPIPE,
    GEMM_MMA_VARIANT_REGPIPE_K32, LinearKernels, PrefillEncodeRequest, clear_gemm_mma_variant,
    force_gemm_mma_variant, last_gemm_mma_variant,
};
use metal::{Device, MTLResourceOptions};

/// Forced-variant override and last-dispatched counter are process-global.
static SERIAL: Mutex<()> = Mutex::new(());

/// Worst normalised error allowed for a variant against the CPU reference.
///
/// The value is the *measured* baseline (`linear_gemm_mma`) error on this
/// shape set, rounded up to four significant digits: 2.1603336e-3, recorded
/// by `linear_metal::mma_prefill_gemm_matches_cpu_reference`. The T4 variants
/// claim bit-identical accumulation to that kernel, so their error is the
/// same number; the bound exists to catch a variant that is NOT identical,
/// and it is deliberately tight -- a bound with an order of magnitude of
/// headroom would pass a kernel that got the accumulation order wrong, which
/// is the only thing that can change it here.
const VARIANT_TOLERANCE: f32 = 2.1604e-3;

fn kernels() -> Option<LinearKernels> {
    if metal::Device::system_default().is_none() {
        eprintln!("no Metal device; skipping");
        return None;
    }
    match LinearKernels::new() {
        Ok(kernels) => Some(kernels),
        Err(error) => {
            eprintln!("LinearKernels unavailable ({error}); skipping");
            None
        }
    }
}

fn variant_name(variant: u8) -> &'static str {
    match variant {
        GEMM_MMA_VARIANT_128X64 => "128x64",
        GEMM_MMA_VARIANT_4SG => "4sg",
        GEMM_MMA_VARIANT_REGPIPE => "regpipe",
        GEMM_MMA_VARIANT_REGPIPE_K32 => "regpipe_k32",
        _ => "auto/unknown",
    }
}

// --- packing helpers, copied from `linear_metal.rs` -----------------------
// Integration test files are separate crates, so there is no shared module to
// import; `linear_metal.rs` must not be edited to export them.

fn integer_states(bits: u8, seed: u32) -> Vec<u16> {
    let mask = (1_u32 << bits) - 1;
    let mut value = seed;
    (0..256)
        .map(|_| {
            value = value.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (value & mask) as u16
        })
        .collect()
}

fn fractional_states(bits: u8, seed: u32) -> Vec<u8> {
    let mut value = seed;
    (0..256)
        .map(|index| {
            value = value.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let width = bits as usize + usize::from((index & 1) != 0);
            (value & ((1_u32 << width) - 1)) as u8
        })
        .collect()
}

fn matrix_packed_integer_shape(
    bits: u8,
    seed: u32,
    tile_rows: usize,
    tile_columns: usize,
) -> Vec<u16> {
    let mut packed = Vec::new();
    for tile in 0..(tile_rows * tile_columns) {
        let (words, _) =
            pack_integer_states(&integer_states(bits, seed + tile as u32 * 17), bits).unwrap();
        packed.extend(words);
    }
    packed
}

fn matrix_packed_fractional_shape(
    bits: u8,
    seed: u32,
    tile_rows: usize,
    tile_columns: usize,
) -> Vec<u16> {
    let mut packed = Vec::new();
    for tile in 0..(tile_rows * tile_columns) {
        let (words, _) = pack_fractional_states(
            &fractional_states(bits, seed + tile as u32 * 17),
            BitRate::Fractional(bits),
        )
        .unwrap();
        packed.extend(words);
    }
    packed
}

fn shape_input(queries: usize, in_features: usize) -> Vec<u16> {
    let mut input = Vec::with_capacity(queries * in_features);
    for query in 0..queries {
        for value in 0..in_features {
            input.push(
                f16::from_f32(((query * in_features + value) as f32 * 0.017).sin()).to_bits(),
            );
        }
    }
    input
}

fn shape_signs(len: usize, modulus: usize) -> Vec<u16> {
    (0..len)
        .map(|index| f16::from_f32(if index % modulus == 0 { -1.0 } else { 1.0 }).to_bits())
        .collect()
}

/// `reconstruct_matrix` without `BitRate::validate_codebook`, so fractional
/// rates can be checked against the MCG codebook `encode_gemm` insists on.
/// Same helpers, same order, same arithmetic as the production reference.
fn reference_matrix(
    packed: &[u16],
    tile_rows: usize,
    tile_columns: usize,
    rate: BitRate,
    codebook: Codebook,
    input_signs: &SignVector,
    output_signs: &SignVector,
) -> Vec<f32> {
    let rows = tile_rows * 16;
    let columns = tile_columns * 16;
    assert_eq!(packed.len(), tile_rows * tile_columns * rate.packed_words());
    let input = input_signs.decode(rows).unwrap();
    let output = output_signs.decode(columns).unwrap();
    let mut values = vec![0.0; rows * columns];
    for tile_row in 0..tile_rows {
        for tile_column in 0..tile_columns {
            let offset = (tile_row * tile_columns + tile_column) * rate.packed_words();
            let words = rate.packed_words();
            let states =
                unpack_trellis(&packed[offset..offset + words], rate).expect("unpack trellis");
            let mut lanes = [0.0_f32; 256];
            for (index, state) in states.iter().enumerate() {
                lanes[index] = decode_state(*state, codebook);
            }
            let tile = unpermute_tile(&lanes);
            for row in 0..16 {
                for column in 0..16 {
                    values[(tile_row * 16 + row) * columns + tile_column * 16 + column] =
                        tile[row * 16 + column];
                }
            }
        }
    }
    if rows.is_multiple_of(128) && columns.is_multiple_of(128) {
        apply_hadamard_matrix(&mut values, rows, columns).unwrap();
    }
    for row in 0..rows {
        for column in 0..columns {
            values[row * columns + column] *= input[row] * output[column];
        }
    }
    values
}

#[allow(clippy::too_many_arguments)]
fn expected_rows(
    packed: &[u16],
    input: &[u16],
    input_signs: &[u16],
    output_signs: &[u16],
    queries: usize,
    in_features: usize,
    out_features: usize,
    rate: BitRate,
    codebook: Codebook,
) -> Vec<f32> {
    let matrix = reference_matrix(
        packed,
        in_features / 16,
        out_features / 16,
        rate,
        codebook,
        &SignVector::from_half_bits(input_signs.to_vec()),
        &SignVector::from_half_bits(output_signs.to_vec()),
    );
    let mut output = vec![0.0; queries * out_features];
    for query in 0..queries {
        for out in 0..out_features {
            for value in 0..in_features {
                output[query * out_features + out] +=
                    f16::from_bits(input[query * in_features + value]).to_f32()
                        * matrix[value * out_features + out];
            }
        }
    }
    output
}

struct Run {
    output: Vec<f32>,
    dispatched: u8,
}

/// One `encode_gemm` under a forced variant, with the canary-row guard.
#[allow(clippy::too_many_arguments)]
fn run(
    kernels: &LinearKernels,
    variant: u8,
    packed: &[u16],
    input: &[u16],
    input_signs: &[u16],
    output_signs: &[u16],
    queries: usize,
    in_features: usize,
    out_features: usize,
    rate: Exl3Rate,
    codebook: Exl3Codebook,
) -> Run {
    // Remove, never set: the flag DISABLES the path under test. Setting it
    // made the original `linear_metal` test silently cover the scalar tiled
    // kernel, which is how the bstage overflow shipped as "6/6".
    unsafe { std::env::remove_var("KAPLUMBA_DISABLE_MMA_PREFILL") };
    force_gemm_mma_variant(variant);

    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();
    let halfs = |values: &[u16]| {
        device.new_buffer_with_data(
            values.as_ptr() as *const c_void,
            size_of_val(values) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    };
    let scratch =
        |bytes: usize| device.new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
    let packed_buffer = halfs(packed);
    let input_buffer = halfs(input);
    let input_sign_buffer = halfs(input_signs);
    let output_sign_buffer = halfs(output_signs);
    let canary = out_features;
    let transformed = scratch(queries * in_features * size_of::<u16>());
    let raw = scratch((queries + 1) * out_features * size_of::<f32>());
    let output = scratch((queries + 1) * out_features * size_of::<u16>());
    unsafe {
        let raw_words = raw.contents() as *mut u32;
        for index in 0..canary {
            *raw_words.add(queries * out_features + index) = 0xDEAD_BEEF;
        }
        let out_words = output.contents() as *mut u16;
        for index in 0..canary {
            *out_words.add(queries * out_features + index) = 0x7E7E;
        }
        std::ptr::write_bytes(raw.contents() as *mut f32, 0, queries * out_features);
    }
    let mut encoder = MetalEncoder::new(&queue).expect("Metal encoder");
    kernels
        .encode_gemm(
            &mut encoder,
            PrefillEncodeRequest {
                packed: &packed_buffer,
                packed_offset: 0,
                perm_mode: 0,
                input: &input_buffer,
                input_signs: &input_sign_buffer,
                output_signs: &output_sign_buffer,
                transformed: &transformed,
                raw: &raw,
                output: &output,
                queries,
                in_features,
                out_features,
                rate,
                codebook,
            },
        )
        .expect("encode prefill GEMM");
    encoder.commit_and_wait().expect("commit GEMM");
    let dispatched = last_gemm_mma_variant();
    unsafe {
        let raw_words = raw.contents() as *const u32;
        for index in 0..canary {
            assert_eq!(
                *raw_words.add(queries * out_features + index),
                0xDEAD_BEEF,
                "[{}] stored output row {queries}, past the end of `raw`",
                variant_name(variant)
            );
        }
        let out_words = output.contents() as *const u16;
        for index in 0..canary {
            assert_eq!(
                *out_words.add(queries * out_features + index),
                0x7E7E,
                "[{}] overwrote the canary row of `output`",
                variant_name(variant)
            );
        }
    }
    let values = output.contents() as *const u16;
    let output_values = (0..queries * out_features)
        .map(|index| f16::from_bits(unsafe { *values.add(index) }).to_f32())
        .collect();
    clear_gemm_mma_variant();
    Run {
        output: output_values,
        dispatched,
    }
}

/// `max|actual - expected| / max|expected|`. A per-element relative error is
/// useless: a dot product of K signed terms crosses zero, so single elements
/// carry no stable scale.
fn normalised_worst(actual: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(actual.len(), expected.len());
    let scale = expected
        .iter()
        .fold(0.0_f32, |peak, value| peak.max(value.abs()));
    assert!(scale > 0.0, "degenerate reference");
    actual
        .iter()
        .zip(expected)
        .fold(0.0_f32, |worst, (value, target)| {
            worst.max((value - target).abs() / scale)
        })
}

/// The main event: every variant, every rate the engine can load, against the
/// CPU reference.
#[test]
fn every_variant_matches_the_cpu_reference() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(kernels) = kernels() else { return };

    // 129 queries = one full 128-row tile plus one row of the second, so the
    // multi-tile dispatch and the row-guarded store of a partial tile are both
    // exercised. 128x128 is the smallest legal shape.
    let (queries, in_features, out_features) = (129, 128, 128);
    let input = shape_input(queries, in_features);
    let input_signs = shape_signs(in_features, 3);
    let output_signs = shape_signs(out_features, 5);
    let mut worst_by_variant: Vec<(&'static str, f32)> = Vec::new();

    let check = |variant: u8,
                 actual: &[f32],
                 expected: &[f32],
                 dispatched: u8,
                 label: String,
                 worst_by_variant: &mut Vec<(&'static str, f32)>| {
        assert_eq!(
            dispatched,
            variant,
            "{label}: forced {} but the dispatch reported {} -- the number below would \
             then be the baseline's, not this variant's",
            variant_name(variant),
            variant_name(dispatched)
        );
        let worst = normalised_worst(actual, expected);
        assert!(
            worst <= VARIANT_TOLERANCE,
            "{label}: worst normalised error {worst:e} exceeds {VARIANT_TOLERANCE:e}"
        );
        let slot = worst_by_variant
            .iter_mut()
            .find(|(name, _)| *name == variant_name(variant));
        match slot {
            Some((_, peak)) => *peak = peak.max(worst),
            None => worst_by_variant.push((variant_name(variant), worst)),
        }
    };

    // Integer 1..=8: the lane-parallel trellis window arithmetic, and the
    // word-count derivation that changes with the rate. Integer 3 is the
    // checkpoint's head rate and the only one the regpipe gate admits.
    for bits in 1..=8u8 {
        let packed = matrix_packed_integer_shape(
            bits,
            bits as u32 * 1301,
            in_features / 16,
            out_features / 16,
        );
        let expected = expected_rows(
            &packed,
            &input,
            &input_signs,
            &output_signs,
            queries,
            in_features,
            out_features,
            BitRate::Integer(bits),
            Codebook::Mcg,
        );
        for variant in [
            GEMM_MMA_VARIANT_128X64,
            GEMM_MMA_VARIANT_4SG,
            GEMM_MMA_VARIANT_REGPIPE,
            GEMM_MMA_VARIANT_REGPIPE_K32,
        ] {
            let run = run(
                &kernels,
                variant,
                &packed,
                &input,
                &input_signs,
                &output_signs,
                queries,
                in_features,
                out_features,
                Exl3Rate::Integer(bits),
                Exl3Codebook::Mcg,
            );
            // The research gate admits 3 bpw only; elsewhere all three
            // research ids (4sg included) resolve to the baseline, and
            // asserting they ran themselves would be asserting a dispatch
            // that cannot happen.
            let gate_admits = variant == GEMM_MMA_VARIANT_128X64 || bits == 3;
            let expected_variant = if gate_admits {
                variant
            } else {
                GEMM_MMA_VARIANT_128X64
            };
            check(
                expected_variant,
                &run.output,
                &expected,
                run.dispatched,
                format!("integer bits={bits} variant={}", variant_name(variant)),
                &mut worst_by_variant,
            );
        }
    }

    // Fractional 1..=3: `fractional_window`, a different word count, and the
    // same gate arithmetic. None of these reach a research variant: 4sg
    // carries only the folded-MCG arm (integer 3 bpw), so its host gate sends
    // it back to the baseline here -- and the fallback is itself asserted by
    // requiring the dispatch to report the baseline below.
    for bits in 1..=3u8 {
        let packed = matrix_packed_fractional_shape(
            bits,
            bits as u32 * 1303,
            in_features / 16,
            out_features / 16,
        );
        let expected = expected_rows(
            &packed,
            &input,
            &input_signs,
            &output_signs,
            queries,
            in_features,
            out_features,
            BitRate::Fractional(bits),
            Codebook::Mcg,
        );
        for variant in [GEMM_MMA_VARIANT_128X64, GEMM_MMA_VARIANT_4SG] {
            let run = run(
                &kernels,
                variant,
                &packed,
                &input,
                &input_signs,
                &output_signs,
                queries,
                in_features,
                out_features,
                Exl3Rate::Fractional(bits),
                Exl3Codebook::Mcg,
            );
            let expected_variant = if variant == GEMM_MMA_VARIANT_4SG {
                GEMM_MMA_VARIANT_128X64
            } else {
                variant
            };
            check(
                expected_variant,
                &run.output,
                &expected,
                run.dispatched,
                format!("fractional bits={bits} variant={}", variant_name(variant)),
                &mut worst_by_variant,
            );
        }
    }

    for (name, worst) in &worst_by_variant {
        println!("variant {name}: worst normalised error {worst:e}");
    }
    assert_eq!(
        worst_by_variant.len(),
        4,
        "not every variant was exercised: {:?}",
        worst_by_variant.iter().map(|(n, _)| *n).collect::<Vec<_>>()
    );
}

/// Non-square shapes and query counts that straddle the 128-row tile boundary,
/// for the two variants whose grid geometry differs from the baseline: 4sg
/// dispatches `out/32` groups on 128 threads, the others `out/64` on 256. A
/// grid that is wrong in x shows up here as wrong columns, not as a crash.
#[test]
fn variant_grids_cover_non_square_shapes() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(kernels) = kernels() else { return };

    let mut worst_overall = 0.0_f32;
    for (in_features, out_features) in [(128, 256), (256, 384), (384, 256), (128, 512)] {
        let input_signs = shape_signs(in_features, 3);
        let output_signs = shape_signs(out_features, 5);
        for queries in [1, 127, 128, 129, 255, 256, 257] {
            let input = shape_input(queries, in_features);
            let packed = matrix_packed_integer_shape(
                3,
                queries as u32 * 31 + in_features as u32 + out_features as u32,
                in_features / 16,
                out_features / 16,
            );
            let expected = expected_rows(
                &packed,
                &input,
                &input_signs,
                &output_signs,
                queries,
                in_features,
                out_features,
                BitRate::Integer(3),
                Codebook::Mcg,
            );
            for variant in [
                GEMM_MMA_VARIANT_128X64,
                GEMM_MMA_VARIANT_4SG,
                GEMM_MMA_VARIANT_REGPIPE,
                GEMM_MMA_VARIANT_REGPIPE_K32,
            ] {
                let run = run(
                    &kernels,
                    variant,
                    &packed,
                    &input,
                    &input_signs,
                    &output_signs,
                    queries,
                    in_features,
                    out_features,
                    Exl3Rate::Integer(3),
                    Exl3Codebook::Mcg,
                );
                assert_eq!(
                    run.dispatched,
                    variant,
                    "shape {in_features}x{out_features} queries={queries}: forced {} but \
                     dispatched {}",
                    variant_name(variant),
                    variant_name(run.dispatched)
                );
                let worst = normalised_worst(&run.output, &expected);
                assert!(
                    worst <= VARIANT_TOLERANCE,
                    "shape {in_features}x{out_features} queries={queries} variant={}: \
                     worst normalised error {worst:e} exceeds {VARIANT_TOLERANCE:e}",
                    variant_name(variant)
                );
                worst_overall = worst_overall.max(worst);
            }
        }
    }
    println!("variant_grids_cover_non_square_shapes: worst normalised error {worst_overall:e}");
}

/// Every variant's output must be bit-identical to the baseline's on the same
/// bytes. This is the claim that lets the tolerance above stay tight: if the
/// accumulation order is unchanged, the error cannot change, so a variant
/// whose error differs from the baseline's at all has reordered something.
///
/// Bit-identity is checked on the fp32 `raw` values indirectly: the output is
/// fp16, so this compares the fp16-rounded results. The kernels test does the
/// same on random bytes; here it is done on real packed data.
#[test]
fn variants_are_bit_identical_to_the_baseline() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(kernels) = kernels() else { return };

    let (queries, in_features, out_features) = (129, 128, 128);
    let input = shape_input(queries, in_features);
    let input_signs = shape_signs(in_features, 3);
    let output_signs = shape_signs(out_features, 5);
    for bits in [2u8, 3, 4] {
        let packed = matrix_packed_integer_shape(
            bits,
            bits as u32 * 4242,
            in_features / 16,
            out_features / 16,
        );
        let baseline = run(
            &kernels,
            GEMM_MMA_VARIANT_128X64,
            &packed,
            &input,
            &input_signs,
            &output_signs,
            queries,
            in_features,
            out_features,
            Exl3Rate::Integer(bits),
            Exl3Codebook::Mcg,
        );
        assert_eq!(baseline.dispatched, GEMM_MMA_VARIANT_128X64);
        for variant in [
            GEMM_MMA_VARIANT_4SG,
            GEMM_MMA_VARIANT_REGPIPE,
            GEMM_MMA_VARIANT_REGPIPE_K32,
        ] {
            let run = run(
                &kernels,
                variant,
                &packed,
                &input,
                &input_signs,
                &output_signs,
                queries,
                in_features,
                out_features,
                Exl3Rate::Integer(bits),
                Exl3Codebook::Mcg,
            );
            // All three research ids fall back at 2 and 4 bpw (the
            // folded-MCG arm is 3 bpw only); only compare when the variant
            // actually ran, and let the identity assert above catch a
            // fallback that should not have happened.
            let gate_admits = bits == 3;
            if !gate_admits {
                assert_eq!(
                    run.dispatched,
                    GEMM_MMA_VARIANT_128X64,
                    "bits={bits}: {} should have fallen back",
                    variant_name(variant)
                );
                continue;
            }
            assert_eq!(run.dispatched, variant);
            let mut first_divergence = None;
            for (index, (actual, expected)) in
                run.output.iter().zip(baseline.output.iter()).enumerate()
            {
                if actual.to_bits() != expected.to_bits() {
                    first_divergence = Some((index, *actual, *expected));
                    break;
                }
            }
            assert_eq!(
                first_divergence,
                None,
                "bits={bits} variant={}: output differs from the baseline at flat index \
                 {:?} (actual {:?}, baseline {:?}) -- the accumulation order claim is \
                 broken, and with it the tight tolerance in the test above",
                variant_name(variant),
                first_divergence.map(|(i, _, _)| i),
                first_divergence.map(|(_, a, _)| a),
                first_divergence.map(|(_, _, e)| e),
            );
        }
    }
}
