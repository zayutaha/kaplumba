//! ADVERSARY INSTRUMENT (round 2) -- DELETE BEFORE MERGING.
//!
//! `spec.md` AC5 states a worst normalised GEMM error of <= 2.16e-3. The test
//! that prints that number, `crates/engine/tests/linear_metal.rs`, gates it at
//! `MMA_TOLERANCE = 0.02` -- 9.3x looser -- so the bound is enforced by
//! nothing. The fixer reported that and did not fix it, which leaves the actual
//! question open: **is 2.16e-3 the worst error, or the worst error of the
//! shapes that test happens to cover?**
//!
//! `linear_metal.rs` covers `in_features` (K) in {128, 256, 384} and
//! `out_features` (N) in {128, 256, 384, 512}. The engine SHIPS
//! 5376->{21504, 16384, 8192, 4096} and 21504->5376. The dot product behind
//! AC5's number at K=384 is 384 terms; at K=21504 it is 56x longer, and
//! fp32 accumulation error grows with the length of the sum. So "2.16e-3 is the
//! worst normalised GEMM error" holds only if the error does NOT grow with K --
//! which is the thing to measure and the thing nobody has.
//!
//! This test is a COPY of `linear_metal.rs`'s reference path so it is
//! independent of that file's tolerances, extended with a K sweep at the
//! production shapes. It reports rather than asserts, because the finding is the
//! numbers; the assertion is that the measurement is self-consistent (a
//! deliberately perturbed result must read a larger error than an unperturbed
//! one, which is the positive control).

use std::ffi::c_void;
use std::mem::{size_of, size_of_val};

use half::f16;
use kaplumba_engine::exl3::{
    BitRate, Codebook, SignVector, apply_hadamard_matrix, decode_state, pack_integer_states,
    reconstruct_matrix, unpack_trellis, unpermute_tile,
};
use kaplumba_kernels::encoder::MetalEncoder;
use kaplumba_kernels::exl3::{Exl3Codebook, Exl3Rate};
use kaplumba_kernels::linear::{LinearKernels, PrefillEncodeRequest};
use metal::{Device, MTLResourceOptions};

// Transcribed from `linear_metal.rs:16`. Note it produces 256 STATES regardless
// of `bits` -- the states are bit indices into the codebook, not values, and
// `pack_integer_states` wants a full 256-lane table. My first version returned
// `1 << bits` states and `pack_integer_states` rejected it with
// `StateCount(8)`, which is the check working.
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
    assert_eq!(
        packed.len(),
        tile_rows * tile_columns * rate.packed_words(),
        "packed length for {tile_rows}x{tile_columns} tiles at {rate:?}"
    );
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

fn expected_mma_rows(
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

#[allow(clippy::too_many_arguments)]
fn mma_gemm_rows(
    kernels: &LinearKernels,
    packed: &[u16],
    input: &[u16],
    input_signs: &[u16],
    output_signs: &[u16],
    queries: usize,
    in_features: usize,
    out_features: usize,
    rate: Exl3Rate,
    codebook: Exl3Codebook,
) -> Result<Vec<f32>, String> {
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
    let transformed = scratch(queries * in_features * size_of::<u16>());
    let raw = scratch((queries + 1) * out_features * size_of::<f32>());
    let output = scratch((queries + 1) * out_features * size_of::<u16>());
    let mut encoder = MetalEncoder::new(&queue).expect("Metal encoder");
    kernels
        .encode_gemm(
            &mut encoder,
            PrefillEncodeRequest {
                packed: &halfs(packed),
                packed_offset: 0,
                perm_mode: 0,
                input: &halfs(input),
                input_signs: &halfs(input_signs),
                output_signs: &halfs(output_signs),
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
        .map_err(|e| format!("{e}"))?;
    encoder.commit_and_wait().expect("commit");
    let values = output.contents() as *const u16;
    Ok((0..queries * out_features)
        .map(|index| f16::from_bits(unsafe { *values.add(index) }).to_f32())
        .collect())
}

fn worst_normalised(actual: &[f32], expected: &[f32]) -> f32 {
    let scale = expected.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    actual
        .iter()
        .zip(expected.iter())
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs() / scale))
}

#[test]
fn gemm_error_grows_with_k_at_the_production_shapes() {
    unsafe { std::env::remove_var("KAPLUMBA_DISABLE_MMA_PREFILL") };
    let kernels = LinearKernels::new().expect("Metal linear kernels should compile");
    let bits = 3u8;
    let queries = 129usize;

    // (K, N, label). The first four are the shapes `linear_metal.rs` covers;
    // the rest are the engine's actual projections.
    // K = 21504 (down_proj's K) with N = 5376 is the real worst case for
    // accumulation length, but its CPU reference is 129 x 21504 x 5376 = 1.5e10
    // multiply-adds and would take minutes. 5376 x 4096 already puts K at the
    // engine's second-largest K and is 21504 / 5376 = 4x smaller. The direction
    // of the trend is what this measures, and the production shape is the last
    // row rather than the only one.
    let cases: Vec<(usize, usize, &str)> = vec![
        (128, 128, "test coverage (K=128)"),
        (384, 256, "test coverage (K=384)"),
        (512, 512, "beyond test coverage"),
        (2048, 1024, "beyond test coverage"),
        (4096, 2048, "beyond test coverage"),
        (5376, 4096, "PRODUCTION k/v_proj K"),
        (8192, 2048, "PRODUCTION o_proj K (full)"),
    ];

    println!(
        "\n{:>7} {:>7} {:>26} {:>14} {:>11} {:>9}",
        "K", "N", "shape", "worst_norm_err", "vs 2.16e-3", "verdict"
    );
    let mut by_k: Vec<(usize, f32)> = Vec::new();
    for (k, n, label) in cases {
        let input = shape_input(queries, k);
        let packed = matrix_packed_integer_shape(bits, 7 * k as u32 + n as u32, k / 16, n / 16);
        let expected = expected_mma_rows(
            &packed,
            &input,
            &shape_signs(k, 3),
            &shape_signs(n, 5),
            queries,
            k,
            n,
            BitRate::Integer(bits),
            Codebook::Mcg,
        );
        match mma_gemm_rows(
            &kernels,
            &packed,
            &input,
            &shape_signs(k, 3),
            &shape_signs(n, 5),
            queries,
            k,
            n,
            Exl3Rate::Integer(bits),
            Exl3Codebook::Mcg,
        ) {
            Ok(actual) => {
                let worst = worst_normalised(&actual, &expected);
                by_k.push((k, worst));
                println!(
                    "{k:>7} {n:>7} {:>26} {worst:>14.4e} {:>11.1} {:>9}",
                    label,
                    worst / 2.1603e-3,
                    if worst < 2.1603e-3 { "OK" } else { "OVER" }
                );
            }
            Err(e) => println!("{k:>7} {n:>7} {:>26} {:>14} {:>11} {:>9}  encode_gemm: {e}", label, "-", "-", "SKIP"),
        }
    }

    // POSITIVE CONTROL on the instrument itself. If the error metric could not
    // see a deliberate perturbation, a flat trend would be uninformative. One
    // output element is nudged by one fp32 ulp-scaled step and the metric must
    // move by strictly more than 0.
    {
        let (k, n) = (128usize, 128usize);
        let input = shape_input(queries, k);
        let packed = matrix_packed_integer_shape(bits, 7 * k as u32 + n as u32, k / 16, n / 16);
        let expected = expected_mma_rows(
            &packed,
            &input,
            &shape_signs(k, 3),
            &shape_signs(n, 5),
            queries,
            k,
            n,
            BitRate::Integer(bits),
            Codebook::Mcg,
        );
        let actual = mma_gemm_rows(
            &kernels,
            &packed,
            &input,
            &shape_signs(k, 3),
            &shape_signs(n, 5),
            queries,
            k,
            n,
            Exl3Rate::Integer(bits),
            Exl3Codebook::Mcg,
        )
        .expect("K=128 encodes");
        let clean = worst_normalised(&actual, &expected);
        let scale = expected.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        // Two perturbation sizes, because the point of a positive control is to
        // show the metric RESPONDS, not to hit an arbitrary threshold. A 1e-3
        // relative nudge on one element of `queries*N` is a smaller perturbation
        // than the clean error itself at this shape, so "the metric doubled" is
        // the expected outcome, not a failure. 1e-1 must dominate outright.
        let mut p1 = actual.clone();
        p1[0] += scale * 1e-3;
        let d1 = worst_normalised(&p1, &expected);
        let mut p2 = actual.clone();
        p2[0] += scale * 1e-1;
        let d2 = worst_normalised(&p2, &expected);
        println!(
            "\npositive control on the metric: clean {clean:.4e} -> \
             +1e-3*peak on 1 element {d1:.4e} ({:.1}x) -> \
             +1e-1*peak {d2:.4e} ({:.0}x)",
            d1 / clean.max(f32::MIN_POSITIVE),
            d2 / clean.max(f32::MIN_POSITIVE)
        );
        assert!(
            d1 > clean && d2 > d1 * 10.0,
            "the error metric is not sensitive: 1e-3 gave {d1:e} (from {clean:e}) \
             and 1e-1 gave {d2:e}"
        );
        assert!(
            (d2 - 0.1).abs() < 1e-3,
            "a 1e-1 relative perturbation of one element should read back as \
             almost exactly 1e-1, and read {d2:e}"
        );
    }

    // The finding: does the error grow with K?
    if by_k.len() >= 3 {
        let (k_small, e_small) = by_k[0];
        let (k_big, e_big) = by_k[by_k.len() - 1];
        println!(
            "\nK {k_small} -> {k_big}: worst normalised error {e_small:.4e} -> {e_big:.4e} \
             = {:.2}x",
            e_big / e_small.max(f32::MIN_POSITIVE)
        );
    }
}
