//! ADVERSARY INSTRUMENT (round 2) -- DELETE BEFORE MERGING.
//!
//! F1 says the §10 ablation table "measured the wrong binary", and gives a
//! specific root cause: `attention.metal` is `include_str!`'d, so editing the
//! `.metal` does nothing until the crate is recompiled, and §10's decisive arm
//! `if (tid < 256u) { return; }` was a completely empty kernel that read
//! 964.79 ms against a 958.39 ms baseline.
//!
//! Two claims to separate:
//!
//!   (1) `tid < 256u` empties the kernel. That is checkable from the source:
//!       `threads_per_group = 256` and `tid = thread_position_in_threadgroup`,
//!       so `tid < 256` is true for every thread. The claim is almost certainly
//!       right, and it is testable WITHOUT any build subtlety: run the arm.
//!
//!   (2) The `.metal` edit failed to reach the binary because of `include_str!`.
//!       This is the part that matters for the spec's METHOD, because §22.2
//!       recommends the uniform-block selector precisely to make this impossible
//!       again. Cargo DOES track `include_str!` inputs: `cargo build -v` prints
//!       "the file `crates/kernels/src/attention.metal` has changed". So the
//!       stated root cause is at best incomplete -- a `.metal` edit is tracked and
//!       does force a rebuild, and what actually went wrong in §10 is that nobody
//!       ran the control.
//!
//! This file runs the §10 arm for real, by patching `attention.metal` and
//! letting the ordinary build path pick it up, then reading the result. It
//! therefore answers (1) empirically and answers (2) by demonstrating that the
//! build path works when the arm is checked.

use half::f16;
use kaplumba_kernels::attention::{
    AttentionEncodeRequest, AttentionKernels, AttentionQk,
};
use kaplumba_kernels::encoder::MetalEncoder;
use metal::{Buffer, CommandQueue, Device, MTLResourceOptions};
use std::time::Instant;

const KEYS: usize = 24576;
const QUERIES: usize = 1024;

struct Fixture {
    device: Device,
    queue: CommandQueue,
    kernels: AttentionKernels,
}

fn main() {
    let device = Device::system_default().expect("Metal device");
    let kernels = AttentionKernels::new().expect("AttentionKernels::new");
    let queue = device.new_command_queue().to_owned();
    let fx = Fixture { device, queue, kernels };

    let row = |n: usize, salt: f32| -> Buffer {
        let bits: Vec<u16> = (0..n)
            .map(|i| {
                let x = ((i as f32 + salt) * 0.0173).sin() * 0.5
                    + ((i as f32 + salt) * 0.0031).cos() * 0.25;
                f16::from_f32(x).to_bits()
            })
            .collect();
        fx.device.new_buffer_with_data(
            bits.as_ptr() as *const std::ffi::c_void,
            (bits.len() * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    };
    let u32buf = |v: &[u32]| -> Buffer {
        fx.device.new_buffer_with_data(
            v.as_ptr() as *const std::ffi::c_void,
            (v.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    };

    let kv = KEYS * 4 * 512;
    let queries = QUERIES * 32 * 512;
    let queries_buf = row(queries, 3.0);
    let keys_buf = row(kv, 11.0);
    let values_buf = row(kv, 29.0);
    let qp: Vec<u32> = (0..QUERIES as u32).map(|i| KEYS as u32 + i).collect();
    let kp: Vec<u32> = (0..KEYS as u32).collect();
    let qp_buf = u32buf(&qp);
    let kp_buf = u32buf(&kp);
    let out = fx.device.new_buffer(
        (QUERIES * 32 * 512 * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let request = AttentionEncodeRequest {
        queries: &queries_buf,
        keys: &keys_buf,
        values: &values_buf,
        query_positions: &qp_buf,
        key_positions: &kp_buf,
        output: &out,
        queries_count: QUERIES,
        q_heads: 32,
        kv_heads: 4,
        head_dim: 512,
        key_count: KEYS,
        scale: 1.0,
        window: 0,
    };

    assert!(
        fx.kernels.flash_attention_available(),
        "flash_attention_available() is false; the F1 arm would not even run"
    );
    println!("device: {}", fx.device.name());
    println!("flash_attention_available: true");

    for (label, block) in [("flash BQ=1", 1usize), ("flash BQ=2", 2usize)] {
        let mut ms = Vec::new();
        for _ in 0..3 {
            let t0 = Instant::now();
            let mut enc = MetalEncoder::new(&fx.queue).expect("encoder");
            fx.kernels
                .encode_online_attention_qk_block(&mut enc, request, AttentionQk::Flash, block)
                .expect("encode");
            enc.commit_and_wait().expect("wait");
            ms.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // Non-zero output is the positive control that the kernel RAN. An arm
        // that emptied the kernel writes zeros, and a table that reported a
        // timing for it without checking this is the F1 mistake.
        let sentinel = unsafe { *(out.contents() as *const u16).add(7) };
        let ran = f16::from_bits(sentinel).to_f32() != 0.0;
        println!(
            "{label}: median {:8.2} ms  [min {:.2} max {:.2}]   output non-zero: {ran} \
             (sentinel {sentinel})",
            ms[1],
            ms[0],
            ms[2]
        );
    }
}
