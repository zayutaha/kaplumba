use half::f16;
use kaplumba_engine::exl3::{
    BitRate, Codebook, SignVector, decode_state, decode_tile, hadamard_128, pack_fractional_states,
    pack_integer_states, pack_signs, pack_trellis, tensor_core_permutation,
};

fn states(bits: u8, seed: u32) -> Vec<u16> {
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

fn signs(length: usize, seed: u32) -> (SignVector, SignVector) {
    let input: Vec<f16> = (0..length)
        .map(|index| {
            f16::from_f32(if (index + seed as usize).is_multiple_of(2) {
                1.0
            } else {
                -1.0
            })
        })
        .collect();
    let output: Vec<f16> = (0..length)
        .map(|index| {
            f16::from_f32(if (index + seed as usize + 1).is_multiple_of(2) {
                1.0
            } else {
                -1.0
            })
        })
        .collect();
    let input_packed = pack_signs(&input).unwrap();
    (
        SignVector::from_packed_bits(input_packed),
        SignVector::from_half_bits(output.iter().map(|value| value.to_bits()).collect()),
    )
}

#[test]
fn codebook_vectors_match_format() {
    let expected = [
        (Codebook::Plain, [0x3a25, 0xbb5b, 0xc066, 0xb7f8, 0xb110]),
        (Codebook::Mcg, [0x3f60, 0x304e, 0x3e0f, 0xb915, 0x3acd]),
        (Codebook::Mul1, [0xc2e8, 0x3921, 0xb89e, 0xbaf2, 0xb936]),
    ];
    for (codebook, values) in expected {
        for (state, expected_bits) in [0_u16, 1, 0x1234, 0x8000, 0xffff].into_iter().zip(values) {
            assert_eq!(
                decode_state(state, codebook),
                f16::from_bits(expected_bits).to_f32()
            );
        }
    }
}

#[test]
fn integer_variant_matrix() {
    for bits in 1..=8 {
        for codebook in [Codebook::Plain, Codebook::Mcg, Codebook::Mul1] {
            let (packed, windows) =
                pack_integer_states(&states(bits, bits as u32 * 101), bits).unwrap();
            assert_eq!(packed.len(), 16 * bits as usize);
            let decoded = decode_tile(&packed, BitRate::Integer(bits), codebook).unwrap();
            let (input_signs, output_signs) = signs(16, bits as u32);
            let matrix = kaplumba_engine::exl3::reconstruct_matrix(
                &packed,
                1,
                1,
                BitRate::Integer(bits),
                codebook,
                &input_signs,
                &output_signs,
            )
            .unwrap();
            for row in 0..16 {
                for column in 0..16 {
                    let input = if (row + bits as usize).is_multiple_of(2) {
                        1.0
                    } else {
                        -1.0
                    };
                    let output = if (column + bits as usize + 1).is_multiple_of(2) {
                        1.0
                    } else {
                        -1.0
                    };
                    let expected = decoded[row * 16 + column] * input * output;
                    assert!((matrix[row * 16 + column] - expected).abs() < 1e-6);
                }
            }
            assert_eq!(windows.len(), 256);
            assert!(decode_state(windows[0], codebook).is_finite());
        }
    }
}

#[test]
fn fractional_mul1_matrix() {
    for bits in 1..=3 {
        let (packed, windows) = pack_fractional_states(
            &fractional_states(bits, bits as u32 * 211),
            BitRate::Fractional(bits),
        )
        .unwrap();
        assert_eq!(packed.len(), 16 * bits as usize + 8);
        assert_eq!(windows.len(), 256);
        let tile = decode_tile(&packed, BitRate::Fractional(bits), Codebook::Mul1).unwrap();
        assert!(tile.iter().all(|value| value.is_finite()));
        for codebook in [Codebook::Plain, Codebook::Mcg] {
            assert!(decode_tile(&packed, BitRate::Fractional(bits), codebook).is_err());
        }
    }
}

/// The trellis packer is fixture-only and deliberately *not* an inverse of the
/// reader, so this asserts the two things it does promise: the ring is the length
/// the reader expects, and the reader accepts it. Round-trip fidelity is not
/// claimed because the reader is lossy at the ring wrap; see `pack_integer_windows`.
#[test]
fn pack_trellis_produces_a_readable_ring() {
    for bits in 1..=8_u8 {
        let (_, windows) = pack_integer_states(&states(bits, 17), bits).unwrap();
        let packed = pack_trellis(&windows, BitRate::Integer(bits)).unwrap();
        assert_eq!(packed.len(), 16 * bits as usize, "bits={bits}");
        let decoded =
            kaplumba_engine::exl3::unpack_trellis(&packed, BitRate::Integer(bits)).unwrap();
        assert_eq!(decoded.len(), 256, "bits={bits}");
    }
}

#[test]
fn signs_and_transforms_are_public_and_stable() {
    let values: Vec<f16> = (0..32)
        .map(|index| f16::from_f32(if index % 2 == 0 { 1.0 } else { -1.0 }))
        .collect();
    let packed = pack_signs(&values).unwrap();
    let unpacked = SignVector::from_packed_bits(packed).decode(32).unwrap();
    assert_eq!(
        unpacked,
        values
            .iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>()
    );

    let permutation = tensor_core_permutation();
    assert_eq!(permutation.len(), 256);
    let mut transformed = [0.0; 128];
    for (index, value) in transformed.iter_mut().enumerate() {
        *value = index as f32 + 1.0;
    }
    hadamard_128(&mut transformed).unwrap();
    assert!(transformed.iter().all(|value| value.is_finite()));
}

#[test]
fn malformed_shapes_and_rates_fail() {
    assert!(pack_trellis(&[0; 255], BitRate::Integer(3)).is_err());
    assert!(kaplumba_engine::exl3::unpack_trellis(&[0; 2], BitRate::Integer(3)).is_err());
    assert!(BitRate::from_rate(4.5).is_err());
    assert!(BitRate::from_rate(0.5).is_err());
}

#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL"]
fn decodes_real_current_mcg_tile() {
    use std::fs::File;
    use std::io::{Read, Seek, SeekFrom};
    use std::path::PathBuf;

    let model = PathBuf::from(
        std::env::var("KAPLUMBA_TEST_MODEL")
            .expect("KAPLUMBA_TEST_MODEL must point to the checkpoint"),
    );
    let metadata = kaplumba_engine::safetensors::inspect_model_dir(&model).unwrap();
    let name = "model.language_model.layers.0.self_attn.q_proj.trellis";
    let shard = metadata.selection.shard_for(name).unwrap();
    let (start, end) = metadata.selection.absolute_range(name).unwrap();
    let mut file = File::open(model.join(shard)).unwrap();
    file.seek(SeekFrom::Start(start)).unwrap();
    let mut bytes = vec![0_u8; (end - start) as usize];
    file.read_exact(&mut bytes).unwrap();
    let words: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect();
    let tile_words = &words[..48];
    let tile = decode_tile(tile_words, BitRate::Integer(3), Codebook::Mcg).unwrap();
    assert!(tile.iter().all(|value| value.is_finite()));
}
