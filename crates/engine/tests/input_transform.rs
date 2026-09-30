//! Bit-exactness of `linear_input_transform`, the decode-path EXL3 input
//! butterfly, against an independent CPU reference.
//!
//! This kernel is the only place in decode where 240 dispatches per token add up
//! to milliseconds of pure overhead, so it was rewritten from "one thread holds
//! 128 floats" to "one 32-lane simdgroup holds four floats per lane". Float
//! addition is not associative and the butterfly is not a free reordering, so the
//! only acceptable verdict is *bit-identical output*, not a tolerance: every
//! element is compared as raw `f16` bits.
//!
//! The reference below is written from the butterfly's definition rather than
//! copied from the kernel, and derives its scale from `1 / sqrt(128)` so a
//! wrong constant in the kernel is a failure rather than a shared bug.
//!
//! The shapes deliberately include `in_features / 128` block counts on both sides
//! of 32, because the kernel is dispatched with 64-thread groups (two simdgroups)
//! and the lane index is `thread_index_in_simdgroup`, not `gid`. Getting that
//! wrong shows up only at block counts above 32.

use std::ffi::c_void;
use std::mem::size_of_val;

use half::f16;
use kaplumba_kernels::encoder::MetalEncoder;
use kaplumba_kernels::linear::LinearKernels;
use metal::{Device, MTLResourceOptions};

/// xorshift, so the data is not a small-integer LCG with a short period (an
/// earlier bit-exactness check in this repo passed vacuously for that reason).
fn noise(count: usize, seed: u32) -> Vec<u32> {
    let mut state = seed | 1;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        })
        .collect()
}

/// A half whose exponent is a few notches either side of 1.0, so the butterfly
/// sees a spread of magnitudes and produces neither exact zeros nor overflow.
fn spread_half(words: &[u32]) -> u16 {
    // sign | exponent in [0x2c00, 0x3400] | 10-bit mantissa
    let exponent = 0x2E00u32 + (words[0] % 0x600u32);
    let mantissa = words[1] & 0x3FF;
    ((words[2] & 1) << 15 | exponent | mantissa) as u16
}

/// The butterfly as its definition states it: `values[i]` and `values[i ^ width]`
/// become their sum and difference, widths 1, 2, 4 ... 64, then scale by
/// `1 / sqrt(128)`.
fn reference(input: &[u16], input_signs: &[u16]) -> Vec<u16> {
    assert_eq!(input.len(), input_signs.len());
    assert_eq!(input.len() % 128, 0);
    let mut values: Vec<f32> = (0..input.len())
        .map(|index| {
            f16::from_bits(input[index]).to_f32() * f16::from_bits(input_signs[index]).to_f32()
        })
        .collect();
    for block in values.chunks_mut(128) {
        let mut width = 1usize;
        while width < 128 {
            let mut start = 0usize;
            while start < 128 {
                for offset in 0..width {
                    let left = block[start + offset];
                    let right = block[start + offset + width];
                    block[start + offset] = left + right;
                    block[start + offset + width] = left - right;
                }
                start += width * 2;
            }
            width *= 2;
        }
        let scale = 1.0f32 / (128.0f32).sqrt();
        for value in block.iter_mut() {
            *value *= scale;
        }
    }
    values
        .iter()
        .map(|value| f16::from_f32(*value).to_bits())
        .collect()
}

/// Run the kernel over `in_features / 128` blocks and return the output bits.
///
/// `input_signs` is allocated at exactly `in_features` halfs -- the length the
/// model gives it (`model.rs`, `input_signs.len() != in_features` is rejected at
/// load) -- so a kernel that indexed it per block instead of per element walks
/// off the end of the buffer and is caught by the comparison against
/// `reference`, not by luck.
fn gpu(input: &[u16], input_signs: &[u16], in_features: usize) -> Vec<u16> {
    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();
    let exact = |values: &[u16]| {
        device.new_buffer_with_data(
            values.as_ptr() as *const c_void,
            size_of_val(values) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    };
    let input_buffer = exact(input);
    let sign_buffer = exact(input_signs);
    // One canary block past the end, so a kernel that stores past `in_features`
    // is reported as such instead of quietly succeeding on a padded buffer.
    let canary_blocks = 2;
    let transformed = device.new_buffer(
        ((in_features / 128 + canary_blocks) * 128 * size_of::<u16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let canary: u16 = 0x7E7E;
    unsafe {
        let words = transformed.contents() as *mut u16;
        for index in 0..(canary_blocks * 128) {
            *words.add(in_features + index) = canary;
        }
    }

    let kernels = LinearKernels::shared().expect("linear kernels");
    let mut encoder = MetalEncoder::new(&queue).expect("Metal encoder");
    kernels
        .encode_input_transform(
            &mut encoder,
            &input_buffer,
            &sign_buffer,
            &transformed,
            in_features,
        )
        .expect("encode input transform");
    encoder.commit_and_wait().expect("commit");

    let words = transformed.contents() as *const u16;
    for index in 0..(canary_blocks * 128) {
        assert_eq!(
            unsafe { *words.add(in_features + index) },
            canary,
            "linear_input_transform stored element {}/{in_features}, past the end \
             of the output buffer",
            in_features + index,
        );
    }
    (0..in_features)
        .map(|index| unsafe { *words.add(index) })
        .collect()
}

fn check(in_features: usize, seed: u32) {
    let words = noise(in_features * 4, seed);
    let input: Vec<u16> = (0..in_features)
        .map(|index| spread_half(&words[index * 4..index * 4 + 4]))
        .collect();
    let input_signs: Vec<u16> = (0..in_features)
        .map(|index| {
            f16::from_f32(if words[index * 4 + 3] % 2 == 0 {
                -1.0
            } else {
                1.0
            })
            .to_bits()
        })
        .collect();

    let expected = reference(&input, &input_signs);
    let actual = gpu(&input, &input_signs, in_features);

    let mut worst: Option<(usize, u16, u16)> = None;
    for (index, (want, got)) in expected.iter().zip(actual.iter()).enumerate() {
        if want != got {
            let record = (index, *want, *got);
            let replace = match worst {
                None => true,
                // Report the *largest* deviation, not the first: a butterfly that
                // has the pass order wrong differs everywhere, and the first
                // mismatch says nothing the largest one does not.
                Some(current) => record.1.abs_diff(record.2) > current.1.abs_diff(current.2),
            };
            if replace {
                worst = Some(record);
            }
        }
    }
    let Some((index, want, got)) = worst else {
        return;
    };
    let differing = expected
        .iter()
        .zip(actual.iter())
        .filter(|(want, got)| want != got)
        .count();
    panic!(
        "linear_input_transform at in_features={in_features} is not bit-exact: \
         {differing} of {in_features} elements differ; worst at {index}/{in_features}: \
         expected 0x{want:04x} ({:e}), got 0x{got:04x} ({:e})",
        f16::from_bits(want).to_f32(),
        f16::from_bits(got).to_f32(),
    );
}

/// The decode shape: 5376 features is 42 blocks, which spans both simdgroups of a
/// 64-thread dispatch group and stops 10 lanes into the second.
#[test]
fn input_transform_is_bit_exact_at_the_decode_shape() {
    check(5376, 0x1234_5678);
}

/// One block, and a block count that exactly fills one simdgroup.
#[test]
fn input_transform_is_bit_exact_for_one_and_32_blocks() {
    check(128, 0x0BAD_C0DE);
    check(4096, 0x0BAD_C0DF);
}

/// Block counts 33 and 65: the 33rd and 65th blocks are the first lane of the
/// second and third simdgroups, so a kernel that derived the block index from the
/// lane rather than from `gid` fails here and nowhere else.
#[test]
fn input_transform_is_bit_exact_across_simdgroup_boundaries() {
    check(4224, 0x00C0_FFEE);
    check(8320, 0x00C0_FFEF);
}

/// Two identical inputs must give two identical outputs: a shuffle with the wrong
/// lane mask is data-dependent, so this is a second, independent witness of the
/// same class of bug.
#[test]
fn input_transform_is_deterministic_and_block_local() {
    let in_features = 5376;
    let words = noise(in_features * 4, 0x5EED_1234);
    let input: Vec<u16> = (0..in_features)
        .map(|index| spread_half(&words[index * 4..index * 4 + 4]))
        .collect();
    let input_signs: Vec<u16> = (0..in_features)
        .map(|index| {
            f16::from_f32(if words[index * 4 + 3] % 2 == 0 {
                -1.0
            } else {
                1.0
            })
            .to_bits()
        })
        .collect();
    assert_eq!(
        gpu(&input, &input_signs, in_features),
        gpu(&input, &input_signs, in_features)
    );

    // Block locality: changing one sign in block 20 must leave every other
    // block's output bits alone. A kernel whose lane mapping crossed blocks
    // (e.g. treating the block index as `gid / 32`) would smear that change
    // across the whole buffer.
    let mut perturbed = input_signs.clone();
    perturbed[20 * 128 + 7] ^= 0x8000;
    let before = gpu(&input, &input_signs, in_features);
    let after = gpu(&input, &perturbed, in_features);
    let changed: Vec<usize> = (0..in_features)
        .filter(|i| before[*i] != after[*i])
        .collect();
    assert_eq!(
        changed
            .iter()
            .filter(|i| **i < 20 * 128 || **i >= 21 * 128)
            .count(),
        0,
        "flipping one sign inside block 20 changed {changed:?}, which is outside it",
    );
    assert_eq!(
        reference(&input, &perturbed),
        after,
        "perturbed run not bit-exact"
    );
}
