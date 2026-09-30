//! In-place `rope` must be bit-identical to out-of-place, and must not race.
//!
//! `rope` is called in place on q and k, and it rotates column `c` with column
//! `c + head_dim/2`. With one thread per column, the thread for `c` could write
//! `output[c]` before the thread for `c + head_dim/2` had read `input[c]` — a
//! cross-warp race whose winner depended on scheduling. That made greedy decode
//! nondeterministic in ~2 of 8 runs of a 64-token prompt, and
//! `KAPLUMBA_MAX_IN_FLIGHT=1` could not mask it because the hazard is inside a
//! single dispatch, not between command buffers.
//!
//! This test is cheap and un-ignored on purpose: no checkpoint, so it runs in
//! milliseconds on every build. The full determinism check costs a model load
//! plus 16 prefills, so it is `#[ignore]`d and a reviewer can skip it. This file
//! is what stops the race from coming back unnoticed.

use half::f16;
use kaplumba_kernels::elements::{ElementKernels, RopeBufferRequest};
use kaplumba_kernels::encoder::MetalEncoder;
use metal::{Device, MTLResourceOptions};

/// Run `rope` `passes` times in place and return the resulting half bits.
fn rope_in_place(
    device: &Device,
    input: &[u16],
    tokens: usize,
    heads: usize,
    head_dim: usize,
    offset: usize,
    passes: usize,
) -> Vec<u16> {
    let kernels = ElementKernels::shared().expect("element kernels");
    let q = device.new_buffer(
        (input.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    unsafe {
        std::ptr::copy_nonoverlapping(input.as_ptr(), q.contents() as *mut u16, input.len());
    }
    for _ in 0..passes {
        let queue = device.new_command_queue();
        let mut encoder = MetalEncoder::new(&queue).expect("encoder");
        kernels
            .encode_rope(
                &mut encoder,
                &q,
                &q,
                RopeBufferRequest {
                    tokens,
                    heads,
                    head_dim,
                    offset,
                    theta: 10000.0,
                    partial_factor: 1.0,
                    proportional: false,
                },
            )
            .expect("encode_rope");
        encoder.commit_and_wait().expect("commit");
    }
    let mut out = Vec::with_capacity(input.len());
    for i in 0..input.len() {
        out.push(unsafe { std::ptr::read_volatile((q.contents() as *const u16).add(i)) });
    }
    out
}

/// Run `rope` exactly once, out of place, and return the resulting half bits.
fn rope_out_of_place(
    device: &Device,
    input: &[u16],
    tokens: usize,
    heads: usize,
    head_dim: usize,
    offset: usize,
) -> Vec<u16> {
    let kernels = ElementKernels::shared().expect("element kernels");
    let src = device.new_buffer(
        (input.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let dst = device.new_buffer(
        (input.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    unsafe {
        std::ptr::copy_nonoverlapping(input.as_ptr(), src.contents() as *mut u16, input.len());
    }
    let queue = device.new_command_queue();
    let mut encoder = MetalEncoder::new(&queue).expect("encoder");
    kernels
        .encode_rope(
            &mut encoder,
            &src,
            &dst,
            RopeBufferRequest {
                tokens,
                heads,
                head_dim,
                offset,
                theta: 10000.0,
                partial_factor: 1.0,
                proportional: false,
            },
        )
        .expect("encode_rope");
    encoder.commit_and_wait().expect("commit");
    let mut out = Vec::with_capacity(input.len());
    for i in 0..input.len() {
        out.push(unsafe { std::ptr::read_volatile((dst.contents() as *const u16).add(i)) });
    }
    out
}

#[test]
fn element_rope_in_place_matches_out_of_place_bitwise() {
    let device = Device::system_default().expect("Metal device");
    // Both of the model's head dims, plus a token count that leaves a partial
    // warp. The racy per-column mapping loses this within a single pass because
    // every rotation pair is split across two lanes of the same dispatch, and
    // with enough threads the write and the read are not ordered.
    for (tokens, heads, head_dim) in [(1usize, 4usize, 256usize), (33, 4, 256), (7, 2, 512)] {
        let n = tokens * heads * head_dim;
        let mut x: u32 = 0x1234_5678;
        let input: Vec<u16> = (0..n)
            .map(|_| {
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                // A spread across [-1, 1) so cos/sin genuinely mix the halves.
                f16::from_f32(((x >> 8) as f32 / 8_388_608.0) - 1.0).to_bits()
            })
            .collect();

        // ONE pass. In-place aliasing matters within a single pass -- the race
        // is a thread writing output[c] before its partner reads input[c] -- so
        // one pass is where a correct and a racy kernel differ. Aliasing many
        // times is NOT a valid comparison: a rotation is not idempotent, so N
        // aliased passes legitimately differ from one out-of-place pass.
        let want = rope_out_of_place(&device, &input, tokens, heads, head_dim, 4);
        let got = rope_in_place(&device, &input, tokens, heads, head_dim, 4, 1);

        let mut differing = 0usize;
        let mut worst_at = 0usize;
        for i in 0..n {
            if got[i] != want[i] {
                differing += 1;
                worst_at = i;
            }
        }
        assert_eq!(
            differing, 0,
            "in-place rope diverges from out-of-place at {tokens}x{heads}x{head_dim} \
             after one aliased pass: {differing} of {n} elements differ, first at {worst_at} \
             (expected bits {:#06x}, got {:#06x}). This is the cross-warp race: a thread \
             writing output[c] before its partner read input[c].",
            want[worst_at], got[worst_at],
        );
    }
}
