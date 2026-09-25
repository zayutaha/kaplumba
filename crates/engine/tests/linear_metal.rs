use half::f16;
use kaplumba_engine::exl3::{
    BitRate, Codebook, SignVector, pack_fractional_states, pack_integer_states, reconstruct_matrix,
};
use kaplumba_kernels::exl3::{Exl3Codebook, Exl3Rate};
use kaplumba_kernels::linear::{LinearKernels, LinearRequest};

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
}
