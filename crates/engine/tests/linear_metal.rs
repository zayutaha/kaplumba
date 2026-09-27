use std::ffi::c_void;
use std::mem::size_of;

use half::f16;
use kaplumba_engine::exl3::{
    BitRate, Codebook, SignVector, pack_fractional_states, pack_integer_states, reconstruct_matrix,
};
use kaplumba_kernels::exl3::{Exl3Codebook, Exl3Rate};
use kaplumba_kernels::linear::{LinearBufferRequest, LinearKernels, LinearRequest, PrefillRequest};
use metal::{Device, MTLResourceOptions};

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

fn kernel_codebook(codebook: Codebook) -> Exl3Codebook {
    match codebook {
        Codebook::Plain => Exl3Codebook::Plain,
        Codebook::Mcg => Exl3Codebook::Mcg,
        Codebook::Mul1 => Exl3Codebook::Mul1,
    }
}

fn matrix_packed_integer(bits: u8, seed: u32) -> Vec<u16> {
    let mut packed = Vec::new();
    for tile in 0..64 {
        let (words, _) =
            pack_integer_states(&integer_states(bits, seed + tile as u32 * 17), bits).unwrap();
        packed.extend(words);
    }
    packed
}

fn matrix_packed_fractional(bits: u8, seed: u32) -> Vec<u16> {
    let mut packed = Vec::new();
    for tile in 0..64 {
        let (words, _) = pack_fractional_states(
            &fractional_states(bits, seed + tile as u32 * 17),
            BitRate::Fractional(bits),
        )
        .unwrap();
        packed.extend(words);
    }
    packed
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

fn vectors() -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let input: Vec<u16> = (0..128)
        .map(|index| f16::from_f32((index as f32 * 0.07).sin()).to_bits())
        .collect();
    let input_signs: Vec<u16> = (0..128)
        .map(|index| f16::from_f32(if index % 3 == 0 { -1.0 } else { 1.0 }).to_bits())
        .collect();
    let output_signs: Vec<u16> = (0..128)
        .map(|index| f16::from_f32(if index % 5 == 0 { -1.0 } else { 1.0 }).to_bits())
        .collect();
    (input, input_signs, output_signs)
}

fn expected_gemv(
    packed: &[u16],
    input: &[u16],
    input_signs: &[u16],
    output_signs: &[u16],
    rate: BitRate,
    codebook: Codebook,
) -> Vec<f32> {
    let matrix = reconstruct_matrix(
        packed,
        8,
        8,
        rate,
        codebook,
        &SignVector::from_half_bits(input_signs.to_vec()),
        &SignVector::from_half_bits(output_signs.to_vec()),
    )
    .unwrap();
    let mut output = vec![0.0; 128];
    for out in 0..128 {
        for value in 0..128 {
            output[out] += f16::from_bits(input[value]).to_f32() * matrix[value * 128 + out];
        }
    }
    output
}

fn expected_rows(
    packed: &[u16],
    input: &[u16],
    input_signs: &[u16],
    output_signs: &[u16],
    queries: usize,
    rate: BitRate,
    codebook: Codebook,
) -> Vec<f32> {
    let matrix = reconstruct_matrix(
        packed,
        8,
        8,
        rate,
        codebook,
        &SignVector::from_half_bits(input_signs.to_vec()),
        &SignVector::from_half_bits(output_signs.to_vec()),
    )
    .unwrap();
    let mut output = vec![0.0; queries * 128];
    for query in 0..queries {
        for out in 0..128 {
            for value in 0..128 {
                output[query * 128 + out] +=
                    f16::from_bits(input[query * 128 + value]).to_f32() * matrix[value * 128 + out];
            }
        }
    }
    output
}

#[allow(clippy::too_many_arguments)]
fn expected_shape_rows(
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
    let matrix = reconstruct_matrix(
        packed,
        in_features / 16,
        out_features / 16,
        rate,
        codebook,
        &SignVector::from_half_bits(input_signs.to_vec()),
        &SignVector::from_half_bits(output_signs.to_vec()),
    )
    .unwrap();
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

#[test]
fn metal_linear_matches_cpu_integer_matrix() {
    let kernels = LinearKernels::new().expect("Metal linear kernels should compile");
    let (input, input_signs, output_signs) = vectors();
    for bits in 1..=8 {
        for codebook in [Codebook::Plain, Codebook::Mcg, Codebook::Mul1] {
            let packed = matrix_packed_integer(bits, bits as u32 * 71);
            let expected = expected_gemv(
                &packed,
                &input,
                &input_signs,
                &output_signs,
                BitRate::Integer(bits),
                codebook,
            );
            let actual = kernels
                .gemv(LinearRequest {
                    packed: &packed,
                    input: &input,
                    input_signs: &input_signs,
                    output_signs: &output_signs,
                    in_features: 128,
                    out_features: 128,
                    rate: Exl3Rate::Integer(bits),
                    codebook: kernel_codebook(codebook),
                })
                .unwrap();
            for (index, value) in actual.iter().enumerate() {
                let actual = f16::from_bits(*value).to_f32();
                assert!(
                    (actual - expected[index]).abs() < 5e-2,
                    "bits={bits} codebook={codebook:?} index={index} expected={} actual={actual}",
                    expected[index]
                );
            }
        }
    }
}

#[test]
fn metal_buffer_backed_linear_matches_slice_path() {
    let kernels = LinearKernels::new().expect("Metal linear kernels should compile");
    let packed = matrix_packed_integer(3, 811);
    let device = Device::system_default().expect("Metal device");
    let packed_buffer = device.new_buffer_with_data(
        packed.as_ptr() as *const c_void,
        (packed.len() * size_of::<u16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let (input, input_signs, output_signs) = vectors();
    let single = kernels
        .gemv_buffer(LinearBufferRequest {
            packed: &packed_buffer,
            packed_offset: 0,
            input: &input,
            input_signs: &input_signs,
            output_signs: &output_signs,
            in_features: 128,
            out_features: 128,
            rate: Exl3Rate::Integer(3),
            codebook: Exl3Codebook::Mcg,
        })
        .unwrap();
    let expected_single = expected_gemv(
        &packed,
        &input,
        &input_signs,
        &output_signs,
        BitRate::Integer(3),
        Codebook::Mcg,
    );
    for (index, value) in single.iter().enumerate() {
        let actual = f16::from_bits(*value).to_f32();
        assert!((actual - expected_single[index]).abs() < 1e-1);
    }
    let mut prefill_input = input.clone();
    prefill_input.extend(input.iter().copied());
    let prefill = kernels
        .gemm_chunk_buffer(LinearBufferRequest {
            packed: &packed_buffer,
            packed_offset: 0,
            input: &prefill_input,
            input_signs: &input_signs,
            output_signs: &output_signs,
            in_features: 128,
            out_features: 128,
            rate: Exl3Rate::Integer(3),
            codebook: Exl3Codebook::Mcg,
        })
        .unwrap();
    assert_eq!(prefill.len(), 256);
    for index in 0..128 {
        let actual = f16::from_bits(prefill[index]).to_f32();
        assert!((actual - expected_single[index]).abs() < 1e-1);
    }
}

#[test]
fn metal_linear_matches_cpu_fractional_matrix() {
    let kernels = LinearKernels::new().expect("Metal linear kernels should compile");
    let (input, input_signs, output_signs) = vectors();
    for bits in 1..=3 {
        let packed = matrix_packed_fractional(bits, bits as u32 * 73);
        let expected = expected_gemv(
            &packed,
            &input,
            &input_signs,
            &output_signs,
            BitRate::Fractional(bits),
            Codebook::Mul1,
        );
        let actual = kernels
            .gemv(LinearRequest {
                packed: &packed,
                input: &input,
                input_signs: &input_signs,
                output_signs: &output_signs,
                in_features: 128,
                out_features: 128,
                rate: Exl3Rate::Fractional(bits),
                codebook: Exl3Codebook::Mul1,
            })
            .unwrap();
        for (index, value) in actual.iter().enumerate() {
            let actual = f16::from_bits(*value).to_f32();
            assert!(
                (actual - expected[index]).abs() < 5e-2,
                "bits={bits}.5 index={index} expected={} actual={actual}",
                expected[index]
            );
        }
    }
}

fn row_input(queries: usize) -> Vec<u16> {
    let mut input = Vec::with_capacity(queries * 128);
    for query in 0..queries {
        for value in 0..128 {
            let number = ((query * 128 + value) as f32 * 0.013).sin();
            input.push(f16::from_f32(number).to_bits());
        }
    }
    input
}

#[test]
fn metal_prefill_tiles_match_cpu_across_chunks_and_variants() {
    let kernels = LinearKernels::new().expect("Metal linear kernels should compile");
    let (_, input_signs, output_signs) = vectors();
    let queries = 17;
    let input = row_input(queries);
    for bits in 1..=8 {
        for codebook in [Codebook::Plain, Codebook::Mcg, Codebook::Mul1] {
            let packed = matrix_packed_integer(bits, bits as u32 * 89);
            let expected = expected_rows(
                &packed,
                &input,
                &input_signs,
                &output_signs,
                queries,
                BitRate::Integer(bits),
                codebook,
            );
            let actual = kernels
                .gemm_chunk(PrefillRequest {
                    packed: &packed,
                    input: &input,
                    input_signs: &input_signs,
                    output_signs: &output_signs,
                    queries,
                    in_features: 128,
                    out_features: 128,
                    rate: Exl3Rate::Integer(bits),
                    codebook: kernel_codebook(codebook),
                })
                .unwrap();
            for (index, value) in actual.iter().enumerate() {
                let actual = f16::from_bits(*value).to_f32();
                assert!(
                    (actual - expected[index]).abs() < 1e-1,
                    "bits={bits} codebook={codebook:?} index={index} expected={} actual={actual}",
                    expected[index]
                );
            }
        }
    }
    for bits in 1..=3 {
        let packed = matrix_packed_fractional(bits, bits as u32 * 91);
        let expected = expected_rows(
            &packed,
            &input,
            &input_signs,
            &output_signs,
            queries,
            BitRate::Fractional(bits),
            Codebook::Mul1,
        );
        let actual = kernels
            .gemm_chunk(PrefillRequest {
                packed: &packed,
                input: &input,
                input_signs: &input_signs,
                output_signs: &output_signs,
                queries,
                in_features: 128,
                out_features: 128,
                rate: Exl3Rate::Fractional(bits),
                codebook: Exl3Codebook::Mul1,
            })
            .unwrap();
        for (index, value) in actual.iter().enumerate() {
            let actual = f16::from_bits(*value).to_f32();
            assert!(
                (actual - expected[index]).abs() < 1e-1,
                "bits={bits}.5 index={index} expected={} actual={actual}",
                expected[index]
            );
        }
    }
    for queries in [1, 15, 16, 32] {
        let input = row_input(queries);
        let packed = matrix_packed_integer(3, 271);
        let expected = expected_rows(
            &packed,
            &input,
            &input_signs,
            &output_signs,
            queries,
            BitRate::Integer(3),
            Codebook::Plain,
        );
        let actual = kernels
            .gemm_chunk(PrefillRequest {
                packed: &packed,
                input: &input,
                input_signs: &input_signs,
                output_signs: &output_signs,
                queries,
                in_features: 128,
                out_features: 128,
                rate: Exl3Rate::Integer(3),
                codebook: Exl3Codebook::Plain,
            })
            .unwrap();
        for (index, value) in actual.iter().enumerate() {
            let actual = f16::from_bits(*value).to_f32();
            assert!(
                (actual - expected[index]).abs() < 1e-1,
                "queries={queries} index={index} expected={} actual={actual}",
                expected[index]
            );
        }
    }
    let input = row_input(17);
    let packed = matrix_packed_integer(3, 401);
    let (_, stats) = kernels
        .gemm_chunk_with_stats(PrefillRequest {
            packed: &packed,
            input: &input,
            input_signs: &input_signs,
            output_signs: &output_signs,
            queries: 17,
            in_features: 128,
            out_features: 128,
            rate: Exl3Rate::Integer(3),
            codebook: Exl3Codebook::Plain,
        })
        .unwrap();
    assert_eq!(stats.query_chunks, 2);
    assert_eq!(stats.decoded_tiles, 128);
    assert_eq!(stats.weight_applications, 278_528);
    assert_eq!(stats.dispatch_groups, 16);
}

#[test]
fn metal_prefill_tiles_match_non_square_shapes() {
    let kernels = LinearKernels::new().expect("Metal linear kernels should compile");
    let in_features = 256;
    let out_features = 384;
    let queries = 19;
    let input_signs: Vec<u16> = (0..in_features)
        .map(|index| f16::from_f32(if index % 3 == 0 { -1.0 } else { 1.0 }).to_bits())
        .collect();
    let output_signs: Vec<u16> = (0..out_features)
        .map(|index| f16::from_f32(if index % 5 == 0 { -1.0 } else { 1.0 }).to_bits())
        .collect();
    let mut input = Vec::with_capacity(queries * in_features);
    for query in 0..queries {
        for value in 0..in_features {
            input.push(
                f16::from_f32(((query * in_features + value) as f32 * 0.017).sin()).to_bits(),
            );
        }
    }
    for bits in 1..=3 {
        for codebook in [Codebook::Plain, Codebook::Mcg, Codebook::Mul1] {
            let packed = matrix_packed_integer_shape(
                bits,
                bits as u32 * 503,
                in_features / 16,
                out_features / 16,
            );
            let expected = expected_shape_rows(
                &packed,
                &input,
                &input_signs,
                &output_signs,
                queries,
                in_features,
                out_features,
                BitRate::Integer(bits),
                codebook,
            );
            let actual = kernels
                .gemm_chunk(PrefillRequest {
                    packed: &packed,
                    input: &input,
                    input_signs: &input_signs,
                    output_signs: &output_signs,
                    queries,
                    in_features,
                    out_features,
                    rate: Exl3Rate::Integer(bits),
                    codebook: kernel_codebook(codebook),
                })
                .unwrap();
            for (index, value) in actual.iter().enumerate() {
                let actual = f16::from_bits(*value).to_f32();
                assert!(
                    (actual - expected[index]).abs() < 5e-1,
                    "bits={bits} codebook={codebook:?} index={index} expected={} actual={actual}",
                    expected[index]
                );
            }
        }
    }
    for bits in 1..=3 {
        let packed = matrix_packed_fractional_shape(
            bits,
            bits as u32 * 509,
            in_features / 16,
            out_features / 16,
        );
        let expected = expected_shape_rows(
            &packed,
            &input,
            &input_signs,
            &output_signs,
            queries,
            in_features,
            out_features,
            BitRate::Fractional(bits),
            Codebook::Mul1,
        );
        let actual = kernels
            .gemm_chunk(PrefillRequest {
                packed: &packed,
                input: &input,
                input_signs: &input_signs,
                output_signs: &output_signs,
                queries,
                in_features,
                out_features,
                rate: Exl3Rate::Fractional(bits),
                codebook: Exl3Codebook::Mul1,
            })
            .unwrap();
        for (index, value) in actual.iter().enumerate() {
            let actual = f16::from_bits(*value).to_f32();
            assert!(
                (actual - expected[index]).abs() < 5e-1,
                "bits={bits}.5 index={index} expected={} actual={actual}",
                expected[index]
            );
        }
    }
}

#[test]
fn linear_rejects_invalid_shapes() {
    let kernels = LinearKernels::new().expect("Metal linear kernels should compile");
    let error = kernels
        .gemv(LinearRequest {
            packed: &[0; 16],
            input: &[0; 128],
            input_signs: &[0; 128],
            output_signs: &[0; 128],
            in_features: 128,
            out_features: 128,
            rate: Exl3Rate::Integer(3),
            codebook: Exl3Codebook::Mcg,
        })
        .unwrap_err()
        .to_string();
    assert!(error.contains("packed length"));

    let prefill_error = kernels
        .gemm_chunk(PrefillRequest {
            packed: &[0; 16],
            input: &[0; 128],
            input_signs: &[0; 128],
            output_signs: &[0; 128],
            queries: 0,
            in_features: 128,
            out_features: 128,
            rate: Exl3Rate::Integer(3),
            codebook: Exl3Codebook::Mcg,
        })
        .unwrap_err()
        .to_string();
    assert!(prefill_error.contains("prefill dimensions"));

    let mismatch_error = kernels
        .gemm_chunk(PrefillRequest {
            packed: &matrix_packed_integer(3, 601),
            input: &[0; 128],
            input_signs: &[0; 127],
            output_signs: &[0; 128],
            queries: 1,
            in_features: 128,
            out_features: 128,
            rate: Exl3Rate::Integer(3),
            codebook: Exl3Codebook::Plain,
        })
        .unwrap_err()
        .to_string();
    assert!(mismatch_error.contains("prefill input or sign length"));
}
