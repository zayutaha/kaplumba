use std::ffi::c_void;
use std::mem::{size_of, size_of_val};

use half::f16;
use kaplumba_engine::exl3::{
    BitRate, Codebook, SignVector, apply_hadamard_matrix, decode_state, pack_fractional_states,
    pack_integer_states, reconstruct_matrix, unpack_trellis, unpermute_tile,
};
use kaplumba_kernels::encoder::MetalEncoder;
use kaplumba_kernels::exl3::{Exl3Codebook, Exl3Rate};
use kaplumba_kernels::linear::{
    LinearBufferRequest, LinearKernels, LinearRequest, PrefillEncodeRequest, PrefillRequest,
};
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
    eprintln!("MASTER expected[0..3] = {:?}", &expected_single[0..3]);
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
fn buffer_gemm_honors_packed_offset_for_fp16_and_fp32() {
    let kernels = LinearKernels::new().expect("Metal linear kernels should compile");
    let device = Device::system_default().expect("Metal device");
    let (row, input_signs, output_signs) = vectors();
    let input: Vec<u16> = [1.0_f32, -0.5, 0.25, 0.75]
        .into_iter()
        .flat_map(|scale| {
            row.iter().map(move |bits| {
                f16::from_f32(f16::from_bits(*bits).to_f32() * scale).to_bits()
            })
        })
        .collect();
    const PREFIX_WORDS: usize = 64;
    const SUFFIX_WORDS: usize = 32;
    const PREFIX_BYTES: u64 = (PREFIX_WORDS * size_of::<u16>()) as u64;

    let as_buffer = |words: &[u16]| {
        device.new_buffer_with_data(
            words.as_ptr() as *const c_void,
            size_of_val(words) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    };

    for bits in [4, 6] {
        let packed = matrix_packed_integer_shape(bits, 0xA13C + u32::from(bits), 8, 8);
        let mut embedded = vec![0xA55A; PREFIX_WORDS];
        embedded.extend_from_slice(&packed);
        embedded.extend(std::iter::repeat_n(0x5AA5, SUFFIX_WORDS));
        let plain_buffer = as_buffer(&packed);
        let embedded_buffer = as_buffer(&embedded);
        let packed_bytes = size_of_val(packed.as_slice()) as u64;

        let plain_half = kernels
            .gemm_chunk_buffer_sliced(
                LinearBufferRequest {
                    packed: &plain_buffer,
                    packed_offset: 0,
                    input: &input,
                    input_signs: &input_signs,
                    output_signs: &output_signs,
                    in_features: 128,
                    out_features: 128,
                    rate: Exl3Rate::Integer(bits),
                    codebook: Exl3Codebook::Mcg,
                    perm_mode: 0,
                },
                packed_bytes,
            )
            .expect("zero-offset fp16 GEMM");
        let offset_half = kernels
            .gemm_chunk_buffer_sliced(
                LinearBufferRequest {
                    packed: &embedded_buffer,
                    packed_offset: PREFIX_BYTES,
                    input: &input,
                    input_signs: &input_signs,
                    output_signs: &output_signs,
                    in_features: 128,
                    out_features: 128,
                    rate: Exl3Rate::Integer(bits),
                    codebook: Exl3Codebook::Mcg,
                    perm_mode: 0,
                },
                packed_bytes,
            )
            .expect("offset fp16 GEMM");
        assert_eq!(
            offset_half, plain_half,
            "fp16 buffer GEMM read prefix canaries instead of the rate-{bits} tensor at byte offset {PREFIX_BYTES}"
        );

        let plain_f32 = kernels
            .gemm_chunk_buffer_f32_sliced(
                LinearBufferRequest {
                    packed: &plain_buffer,
                    packed_offset: 0,
                    input: &input,
                    input_signs: &input_signs,
                    output_signs: &output_signs,
                    in_features: 128,
                    out_features: 128,
                    rate: Exl3Rate::Integer(bits),
                    codebook: Exl3Codebook::Mcg,
                    perm_mode: 0,
                },
                packed_bytes,
            )
            .expect("zero-offset fp32 GEMM");
        let offset_f32 = kernels
            .gemm_chunk_buffer_f32_sliced(
                LinearBufferRequest {
                    packed: &embedded_buffer,
                    packed_offset: PREFIX_BYTES,
                    input: &input,
                    input_signs: &input_signs,
                    output_signs: &output_signs,
                    in_features: 128,
                    out_features: 128,
                    rate: Exl3Rate::Integer(bits),
                    codebook: Exl3Codebook::Mcg,
                    perm_mode: 0,
                },
                packed_bytes,
            )
            .expect("offset fp32 GEMM");
        assert_eq!(
            offset_f32, plain_f32,
            "fp32 buffer GEMM read prefix canaries instead of the rate-{bits} tensor at byte offset {PREFIX_BYTES}"
        );

        let bad_len = packed_bytes - size_of::<u16>() as u64;
        for (output_kind, result) in [
            (
                "fp16",
                kernels.gemm_chunk_buffer_sliced(
                    LinearBufferRequest {
                        packed: &embedded_buffer,
                        packed_offset: PREFIX_BYTES,
                        input: &input,
                        input_signs: &input_signs,
                        output_signs: &output_signs,
                        in_features: 128,
                        out_features: 128,
                        rate: Exl3Rate::Integer(bits),
                        codebook: Exl3Codebook::Mcg,
                        perm_mode: 0,
                    },
                    bad_len,
                )
                .map(|_| ()),
            ),
            (
                "fp32",
                kernels
                    .gemm_chunk_buffer_f32_sliced(
                        LinearBufferRequest {
                            packed: &embedded_buffer,
                            packed_offset: PREFIX_BYTES,
                            input: &input,
                            input_signs: &input_signs,
                            output_signs: &output_signs,
                            in_features: 128,
                            out_features: 128,
                            rate: Exl3Rate::Integer(bits),
                            codebook: Exl3Codebook::Mcg,
                            perm_mode: 0,
                        },
                        bad_len,
                    )
                    .map(|_| ()),
            ),
        ] {
            assert!(
                result.is_err(),
                "{output_kind} sliced GEMM accepted a packed region one word too short at rate {bits}"
            );
        }

        let out_of_range = kernels.gemm_chunk_buffer_sliced(
            LinearBufferRequest {
                packed: &embedded_buffer,
                packed_offset: embedded.len() as u64,
                input: &input,
                input_signs: &input_signs,
                output_signs: &output_signs,
                in_features: 128,
                out_features: 128,
                rate: Exl3Rate::Integer(bits),
                codebook: Exl3Codebook::Mcg,
                perm_mode: 0,
            },
            packed_bytes,
        );
        assert!(
            out_of_range.is_err(),
            "fp16 sliced GEMM accepted a rate-{bits} tensor range past the end of its Metal buffer"
        );
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

/// `reconstruct_matrix` without `BitRate::validate_codebook`, so the fractional
/// rates can be checked against the MCG codebook that `encode_gemm` insists on
/// (`validate_encode`). Same helpers, same order, same arithmetic as the
/// production reference: `unpack_trellis` -> `decode_state` -> `unpermute_tile`
/// per 16x16 tile, then the Hadamard matrix, then the sign scaling.
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

#[allow(clippy::too_many_arguments)]
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

/// Run one prefill GEMM through the production entry point
/// (`LinearKernels::encode_gemm`, i.e. exactly what `model.rs` calls), with
/// `KAPLUMBA_DISABLE_MMA_PREFILL=1` set by the caller so the dispatch selects
/// `linear_gemm_mma` rather than the tiled kernel.
///
/// `encode_gemm` is the only public surface that reaches the MMA path:
/// `gemm_chunk` / `gemm_chunk_buffer` hardcode `gemv_tiled_pipeline`, which is
/// why 8f1a863 could claim "6/6" while shipping a kernel with no test coverage
/// at all. The gate also requires `in_features % 32 == 0` and
/// `out_features % 32 == 0`, which every shape below satisfies with room to
/// spare (`validate_encode` demands multiples of 128 and the MCG codebook).
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
) -> Vec<f32> {
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
    // One canary row past the end of both output buffers. `linear_gemm_mma`
    // processes a whole 128-row tile whatever `queries` is and must NOT store the
    // rows it does not own -- there is no `simdgroup_store` row mask, so that
    // guard is the only thing between a partial tile and a write outside the
    // buffer. Reading `queries * out_features` elements back would never see
    // such a write, so the canary is what makes the guard testable.
    let canary = out_features;
    let transformed = scratch(queries * in_features * size_of::<u16>());
    let raw = scratch((queries + 1) * out_features * size_of::<f32>());
    let output = scratch((queries + 1) * out_features * size_of::<u16>());
    {
        let raw_words = raw.contents() as *mut u32;
        for index in 0..canary {
            unsafe { *raw_words.add(queries * out_features + index) = 0xDEAD_BEEFu32 };
        }
        let out_words = output.contents() as *mut u16;
        for index in 0..canary {
            unsafe { *out_words.add(queries * out_features + index) = 0x7E7Eu16 };
        }
    }
    // `linear_gemm_mma` stores (not accumulates) every output block, so `raw`
    // does not strictly need clearing. Zero it anyway so the comparison stays
    // meaningful if the gate ever falls back to the tiled kernel, which
    // accumulates into `raw`.
    unsafe { std::ptr::write_bytes(raw.contents() as *mut f32, 0, queries * out_features) };
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
        .expect("encode MMA prefill GEMM");
    encoder.commit_and_wait().expect("commit MMA GEMM");
    {
        let raw_words = raw.contents() as *const u32;
        for index in 0..canary {
            assert_eq!(
                unsafe { *raw_words.add(queries * out_features + index) },
                0xDEAD_BEEFu32,
                "the MMA kernel stored output row {queries}, past the end of `raw`"
            );
        }
        let out_words = output.contents() as *const u16;
        for index in 0..canary {
            assert_eq!(
                unsafe { *out_words.add(queries * out_features + index) },
                0x7E7Eu16,
                "the MMA kernel overwrote the canary row of `output`"
            );
        }
    }
    let values = output.contents() as *const u16;
    (0..queries * out_features)
        .map(|index| f16::from_bits(unsafe { *values.add(index) }).to_f32())
        .collect()
}

/// Compare one MMA GEMM result against the CPU reference built from the same
/// EXL3 decode helpers (`reconstruct_matrix`, which is what packs the trellis
/// the kernel dequantizes) and report the worst normalised error.
///
/// Metric: `max|actual - expected| / max|expected|`, i.e. the largest absolute
/// deviation anywhere in the result as a fraction of the largest magnitude the
/// reference produces. A per-element relative test is useless here because
/// dot products of K signed terms cross zero, so single elements carry no
/// meaningful scale; the reference's peak magnitude is the one stable scale
/// available.
fn assert_mma_close(label: &str, actual: &[f32], expected: &[f32], worst_so_far: &mut f32) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    let scale = expected
        .iter()
        .fold(0.0_f32, |peak, value| peak.max(value.abs()));
    assert!(scale > 0.0, "{label}: degenerate reference");
    let mut worst = 0.0_f32;
    for (index, value) in actual.iter().enumerate() {
        worst = worst.max((value - expected[index]).abs() / scale);
    }
    *worst_so_far = worst_so_far.max(worst);
    assert!(
        worst < MMA_TOLERANCE,
        "{label}: worst normalised error {worst:e} exceeds {MMA_TOLERANCE:e}"
    );
}

/// Bit-exactness is NOT achievable and is NOT claimed: the tensor path rounds
/// every dequantized weight to fp16 (`half(decode_state(...))`, and
/// `decode_three_inst` sums two fp16 values, so the sum need not be fp16), and
/// accumulates 8x8 half products in a different order than the fp32 reference
/// (64-wide chunks, four accumulator chains vs. one ascending loop). Both are
/// documented in the kernel header (Q10, user decision 2026-09-27).
///
/// **This was 0.02 and enforced almost nothing** (`spec/prefill-24k-90tps` A5):
/// AC5 states a 2.16e-3 bound, so the gate sat **9.3x above the spec's own
/// limit**. A gate that loose cannot fail on anything short of a gross break.
///
/// **3e-3 is set against this test's own measured worst, not against a
/// remembered figure.** The record, because two tighter values were tried and
/// failed, which is what makes this a gate and not a decoration:
///
/// | `MMA_TOLERANCE` | result |
/// |---|---|
/// | 0.02 (was) | passes; 9.3x above the worst this test observes |
/// | 5e-4 | **FAILS**: `integer bits=2: worst normalised error 5.628729e-4` |
/// | 1e-3 | **FAILS**: `shape 128x256 queries=1: worst normalised error 2.1603336e-3` |
/// | **3e-3** | passes; **1.39x** above the worst |
///
/// **AC5's 2.16e-3 is not stale -- it is this test's own worst**, at the small
/// `128x256, queries=1` shape, and it was read off this test's output. So the
/// correction to AC5 is not "the number is wrong"; it is that the gate was 9.3x
/// above it.
///
/// The worst error at the shapes the engine actually **ships** is far lower.
/// Measured through the production entry point
/// (`spec/prefill-24k-90tps/adversary_evidence2/a5_gemm_error_budget.log`):
///
/// | K | N | shape | worst normalised error |
/// |---|---|---|---|
/// | 128 | 128 | test coverage | 5.05e-4 |
/// | 384 | 256 | test coverage | 4.87e-4 |
/// | 5376 | 4096 | production k/v_proj K | 3.14e-4 |
/// | 8192 | 2048 | production o_proj K | **4.55e-4** |
///
/// so production sits **4.7x inside** AC5's bound and the error does **not** grow
/// with K (0.90x from K=128 to K=8192 -- the metric normalises by the
/// reference's peak magnitude, which grows with the dot product too). The
/// residual is the fp16 rounding of every dequantized weight, which is a property
/// of the weight format and not of K.
///
/// 3e-3 keeps 1.39x of headroom over the observed worst -- enough that a
/// different compiler's rounding does not make the gate red for no reason -- and
/// is **6.7x tighter** than what it replaced.
const MMA_TOLERANCE: f32 = 3e-3;

#[test]
fn mma_prefill_gemm_matches_cpu_reference() {
    // The gate is process-global, but `encode_gemm` is the only reader and only
    // this test calls it, so there is no concurrent `getenv` to race with. Set
    // it before any dispatch; it is read per `encode_gemm` call, not cached.
    //
    // This must be *removed* rather than set: the flag DISABLES the MMA path,
    // and the variable default is to USE it. Setting it here made this test
    // silently cover the scalar tiled kernel instead, which is how the original
    // bstage overflow shipped with a "6/6" claim. With the flag cleared, a
    // deliberately sabotaged `linear_gemm_mma` fails this test.
    unsafe { std::env::remove_var("KAPLUMBA_DISABLE_MMA_PREFILL") };
    let kernels = LinearKernels::new().expect("Metal linear kernels should compile");
    let mut worst = 0.0_f32;

    // Every rate the engine can load: integer 1..=8 and fractional 1..=3. The
    // integer path exercises the lane-parallel trellis window arithmetic, the
    // fractional path `fractional_window`, and the word-count derivations
    // differ. 129 queries = one FULL 128-row tile plus one row of the second, so
    // both the multi-tile dispatch and the row-guarded store of a partial tile
    // are exercised; 129 is also the smallest count that proves the dispatch
    // grid, not just the kernel, grew with the tile.
    let (queries, in_features, out_features) = (129, 128, 128);
    let input = shape_input(queries, in_features);
    let input_signs = shape_signs(in_features, 3);
    let output_signs = shape_signs(out_features, 5);
    for bits in 1..=8 {
        let packed = matrix_packed_integer_shape(
            bits,
            bits as u32 * 1301,
            in_features / 16,
            out_features / 16,
        );
        let expected = expected_mma_rows(
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
        let actual = mma_gemm_rows(
            &kernels,
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
        assert_mma_close(
            &format!("integer bits={bits}"),
            &actual,
            &expected,
            &mut worst,
        );
    }
    for bits in 1..=3 {
        let packed = matrix_packed_fractional_shape(
            bits,
            bits as u32 * 1303,
            in_features / 16,
            out_features / 16,
        );
        let expected = expected_mma_rows(
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
        let actual = mma_gemm_rows(
            &kernels,
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
        assert_mma_close(
            &format!("fractional bits={bits}"),
            &actual,
            &expected,
            &mut worst,
        );
    }

    // Non-square shapes and query counts that straddle the 128-row MMA tile
    // boundary: 1 (one partial tile), 127 (partial), 128 (exactly one), 129
    // (one row of a second tile), 255 and 256 (two tiles, the second exactly
    // full then one row over), 257 (three tiles). 256x384 is the real gate_proj
    // geometry class; 384x256 covers out_features < in_features; 128x512 covers
    // out_features > 2 tiles so the x grid is exercised too. Every count below
    // 128 wastes most of a tile, which is the tail the row-guarded store exists
    // for: a tile that stored unconditionally would scribble past the output.
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
            let expected = expected_mma_rows(
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
            let actual = mma_gemm_rows(
                &kernels,
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
            assert_mma_close(
                &format!("shape {in_features}x{out_features} queries={queries}"),
                &actual,
                &expected,
                &mut worst,
            );
        }
    }

    println!("mma_prefill_gemm_matches_cpu_reference: worst normalised error {worst:e}");
}
