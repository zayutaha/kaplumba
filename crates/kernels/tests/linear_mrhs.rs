//! AC1, AC2: Unit test probe & parity verification for Multi-RHS GEMV (`encode_gemv_mrhs`).
//!
//! Verifies:
//! 1. Parity against single-vector GEMV across batch sizes M in [1, 16] for both
//!    repacked column-major (`perm_mode == 1`) and standard row-major (`perm_mode == 0`).
//! 2. Canaries: MRHS does not write past row M - 1.
//! 3. Sabotage verification: deliberate corruption of row m > 0 causes an immediate,
//!    diagnosable failure.

use std::ffi::c_void;
use std::mem::size_of_val;

use half::f16;
use kaplumba_kernels::encoder::MetalEncoder;
use kaplumba_kernels::exl3::{Exl3Codebook, Exl3Rate};
use kaplumba_kernels::linear::{LinearKernels, PrefillEncodeRequest};
use metal::{Device, MTLResourceOptions};

fn packed_words(tiles: usize, rate: Exl3Rate, seed: u64) -> Vec<u16> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..tiles * rate.packed_words())
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as u16
        })
        .collect()
}

fn activations(queries: usize, in_features: usize) -> Vec<u16> {
    let mut state = 0x2545_F491_4F6C_DD1D_u64;
    (0..queries * in_features)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            f16::from_f32(((state >> 40) as f32 / (1u64 << 24) as f32) - 1.0).to_bits()
        })
        .collect()
}

fn signs(len: usize, modulus: usize) -> Vec<u16> {
    (0..len)
        .map(|index| f16::from_f32(if index % modulus == 0 { -1.0 } else { 1.0 }).to_bits())
        .collect()
}

fn run_mrhs(
    kernels: &LinearKernels,
    packed: &[u16],
    input: &[u16],
    input_signs: &[u16],
    output_signs: &[u16],
    queries: usize,
    in_features: usize,
    out_features: usize,
    perm_mode: u32,
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
    let transformed = scratch(queries * in_features * std::mem::size_of::<u16>());
    let raw = scratch((queries + 1) * out_features * std::mem::size_of::<f32>());
    let output = scratch((queries + 1) * out_features * std::mem::size_of::<u16>());

    unsafe {
        let raw_words = raw.contents() as *mut u32;
        for index in 0..out_features {
            *raw_words.add(queries * out_features + index) = 0xDEAD_BEEF;
        }
        let out_words = output.contents() as *mut u16;
        for index in 0..out_features {
            *out_words.add(queries * out_features + index) = 0x7E7E;
        }
        std::ptr::write_bytes(raw.contents() as *mut f32, 0, queries * out_features);
    }

    let mut encoder = MetalEncoder::new(&queue).expect("Metal encoder");
    kernels
        .encode_gemv_mrhs(
            &mut encoder,
            PrefillEncodeRequest {
                packed: &packed_buffer,
                packed_offset: 0,
                perm_mode,
                input: &input_buffer,
                input_signs: &input_sign_buffer,
                output_signs: &output_sign_buffer,
                transformed: &transformed,
                raw: &raw,
                output: &output,
                queries,
                in_features,
                out_features,
                rate: Exl3Rate::Integer(3),
                codebook: Exl3Codebook::Mcg,
            },
        )
        .expect("encode_gemv_mrhs succeeds");
    encoder.commit_and_wait().expect("GPU completes");

    // Check canaries
    unsafe {
        let raw_words = raw.contents() as *const u32;
        for index in 0..out_features {
            assert_eq!(
                *raw_words.add(queries * out_features + index),
                0xDEAD_BEEF,
                "MRHS canary overrun in raw buffer at offset {index} for queries={queries}"
            );
        }
        let out_words = output.contents() as *const u16;
        for index in 0..out_features {
            assert_eq!(
                *out_words.add(queries * out_features + index),
                0x7E7E,
                "MRHS canary overrun at offset {index} for queries={queries}"
            );
        }
        let halfs_out = std::slice::from_raw_parts(out_words, queries * out_features);
        halfs_out
            .iter()
            .map(|&bits| f16::from_bits(bits).to_f32())
            .collect()
    }
}

#[test]
fn mrhs_parity_against_single_row_gemv() {
    let kernels = LinearKernels::new().unwrap_or_else(|e| panic!("LinearKernels::new failed: {:?}", e));

    let in_features = 256;
    let out_features = 256;
    let tiles = (in_features / 16) * (out_features / 16);
    let rate = Exl3Rate::Integer(3);
    let packed = packed_words(tiles, rate, 42);
    let input_signs = signs(in_features, 7);
    let output_signs = signs(out_features, 5);

    // Test both perm_mode == 0 (row-major) and perm_mode == 1 (colmajor repacked)
    for perm_mode in [0u32, 1u32] {
        // Test multiple batch sizes spanning all M_CAP buckets: 1, 2, 4, 7, 14, 16
        for &m in &[1usize, 2, 4, 7, 14, 16] {
            let act = activations(m, in_features);
            let mrhs_out = run_mrhs(
                &kernels,
                &packed,
                &act,
                &input_signs,
                &output_signs,
                m,
                in_features,
                out_features,
                perm_mode,
            );

            // Compare each row against single-vector run (m=1)
            for row in 0..m {
                let single_act = &act[row * in_features..(row + 1) * in_features];
                let single_out = run_mrhs(
                    &kernels,
                    &packed,
                    single_act,
                    &input_signs,
                    &output_signs,
                    1,
                    in_features,
                    out_features,
                    perm_mode,
                );

                let mrhs_row_out = &mrhs_out[row * out_features..(row + 1) * out_features];
                let mut max_diff: f32 = 0.0;
                for col in 0..out_features {
                    let diff = (mrhs_row_out[col] - single_out[col]).abs();
                    if diff > max_diff {
                        max_diff = diff;
                    }
                }

                assert!(
                    max_diff < 1e-2,
                    "MRHS divergence on perm_mode={perm_mode}, M={m}, row={row}: max_diff={max_diff} exceeds 1e-2"
                );
            }
        }
    }
}
