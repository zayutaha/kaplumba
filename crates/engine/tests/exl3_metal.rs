use half::f16;
use kaplumba_engine::exl3::{
    BitRate, Codebook, decode_tile, hadamard_128, pack_fractional_states, pack_integer_states,
};
use kaplumba_kernels::exl3::{Exl3Codebook, Exl3Kernels, Exl3Rate};

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

#[test]
fn metal_codec_matches_cpu_matrix() {
    let kernels = Exl3Kernels::new().expect("Metal kernels should compile");
    for bits in 1..=8 {
        for codebook in [Codebook::Plain, Codebook::Mcg, Codebook::Mul1] {
            let (packed, _) =
                pack_integer_states(&integer_states(bits, bits as u32 * 37), bits).unwrap();
            let expected = decode_tile(&packed, BitRate::Integer(bits), codebook).unwrap();
            let actual = kernels
                .decode_tile(&packed, Exl3Rate::Integer(bits), kernel_codebook(codebook))
                .unwrap();
            for (index, (expected, actual)) in expected.into_iter().zip(actual).enumerate() {
                assert!(
                    (expected - actual).abs() < 1e-5,
                    "bits={bits} codebook={codebook:?} index={index} expected={expected} actual={actual}"
                );
            }
        }
    }
    for bits in 1..=3 {
        let (packed, _) = pack_fractional_states(
            &fractional_states(bits, bits as u32 * 53),
            BitRate::Fractional(bits),
        )
        .unwrap();
        let expected = decode_tile(&packed, BitRate::Fractional(bits), Codebook::Mul1).unwrap();
        let actual = kernels
            .decode_tile(&packed, Exl3Rate::Fractional(bits), Exl3Codebook::Mul1)
            .unwrap();
        for (expected, actual) in expected.into_iter().zip(actual) {
            assert!((expected - actual).abs() < 1e-5);
        }
    }
}

#[test]
fn metal_hadamard_matches_cpu() {
    let kernels = Exl3Kernels::new().expect("Metal kernels should compile");
    let mut input = Vec::with_capacity(256);
    let mut expected = vec![0.0; 256];
    for (index, expected_value) in expected.iter_mut().enumerate() {
        let value = ((index as f32) * 0.03125 - 3.0).sin();
        input.push(f16::from_f32(value).to_bits());
        *expected_value = f16::from_f32(value).to_f32();
    }
    hadamard_128(&mut expected[..128]).unwrap();
    hadamard_128(&mut expected[128..]).unwrap();
    let actual = kernels.hadamard_128(&input).unwrap();
    for (index, value) in actual.iter().enumerate() {
        let actual = f16::from_bits(*value).to_f32();
        let expected = f16::from_f32(expected[index]).to_f32();
        assert!((actual - expected).abs() < 1e-5);
    }
}
