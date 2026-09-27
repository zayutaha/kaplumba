//! Sustained Apple-Silicon tensor-core throughput for the prefill GEMM.
//!
//! `spec/engine-performance` guesses "30-50+ TFLOPS" for Metal
//! `simdgroup_multiply_accumulate` but never measured it, and the whole
//! tensor-core prefill plan rests on that number. This turns it into a figure.
//!
//! `mma_alu` keeps a, b and c in registers and runs `iters` 8x8x8 MMAs with no
//! memory traffic, so it is the hardware ceiling for the tensor units. Real GEMM
//! throughput will be lower once operands have to be fed from memory.

use metal::{Device, MTLResourceOptions};
use std::time::Instant;

const SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Pure ALU throughput. The third argument to simdgroup_load/store is the leading
// dimension (elements per row), NOT a row stride or element stride -- passing 0
// there was the bug that made the 8x8 probe look broken.
// Sixteen independent accumulators, written as named scalars. This is not
// style: an array `simdgroup_half8x8 acc[32]` with an unrolled index loop made
// the compiler spill the accumulators to local memory and collapsed throughput
// from 15.49 to 0.74 TFLOPS. Only named locals stay in registers.
//
// Chain count matters a lot, because dependent MMAs measure latency, not
// throughput:
//
// | chains | measured |
// |---|---|
// | 1  |  5.75 TFLOPS |
// | 8  |  7.65 TFLOPS |
// | 16 | 15.49 TFLOPS |
// | 32 (array, spilled) | 0.74 TFLOPS |

kernel void mma_alu(device half *out [[buffer(0)]],
                    constant uint &iters [[buffer(1)]],
                    uint gid [[thread_position_in_grid]]) {
    constexpr int LD = 8;
    simdgroup_half8x8 a, b;
    simdgroup_half8x8 c0, c1, c2, c3, c4, c5, c6, c7;
    simdgroup_half8x8 d0, d1, d2, d3, d4, d5, d6, d7;
    simdgroup_load(a, out, LD);
    simdgroup_load(b, out, LD);
    simdgroup_load(c0, out, LD);
    simdgroup_load(c1, out, LD);
    simdgroup_load(c2, out, LD);
    simdgroup_load(c3, out, LD);
    simdgroup_load(c4, out, LD);
    simdgroup_load(c5, out, LD);
    simdgroup_load(c6, out, LD);
    simdgroup_load(c7, out, LD);
    simdgroup_load(d0, out, LD);
    simdgroup_load(d1, out, LD);
    simdgroup_load(d2, out, LD);
    simdgroup_load(d3, out, LD);
    simdgroup_load(d4, out, LD);
    simdgroup_load(d5, out, LD);
    simdgroup_load(d6, out, LD);
    simdgroup_load(d7, out, LD);
    for (uint i = 0; i < iters; ++i) {
        simdgroup_multiply_accumulate(c0, a, b, c0);
        simdgroup_multiply_accumulate(c1, a, b, c1);
        simdgroup_multiply_accumulate(c2, a, b, c2);
        simdgroup_multiply_accumulate(c3, a, b, c3);
        simdgroup_multiply_accumulate(c4, a, b, c4);
        simdgroup_multiply_accumulate(c5, a, b, c5);
        simdgroup_multiply_accumulate(c6, a, b, c6);
        simdgroup_multiply_accumulate(c7, a, b, c7);
        simdgroup_multiply_accumulate(d0, a, b, d0);
        simdgroup_multiply_accumulate(d1, a, b, d1);
        simdgroup_multiply_accumulate(d2, a, b, d2);
        simdgroup_multiply_accumulate(d3, a, b, d3);
        simdgroup_multiply_accumulate(d4, a, b, d4);
        simdgroup_multiply_accumulate(d5, a, b, d5);
        simdgroup_multiply_accumulate(d6, a, b, d6);
        simdgroup_multiply_accumulate(d7, a, b, d7);
    }
    simdgroup_store(c0, out, LD);
    simdgroup_store(c1, out, LD);
    simdgroup_store(c2, out, LD);
    simdgroup_store(c3, out, LD);
    simdgroup_store(c4, out, LD);
    simdgroup_store(c5, out, LD);
    simdgroup_store(c6, out, LD);
    simdgroup_store(c7, out, LD);
}
"#;

fn main() {
    let device = Device::system_default().expect("no Metal device");
    println!("device: {}", device.name());
    let library = device
        .new_library_with_source(SRC, &metal::CompileOptions::new())
        .expect("compile");
    let function = library.get_function("mma_alu", None).expect("fn");
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .expect("pipeline");

    let groups: u64 = 2048; // 2048 simdgroups = 65536 threads
    let iters: u32 = 20000;
    let buf = device.new_buffer(128, MTLResourceOptions::StorageModeShared);
    let iters_buf = device.new_buffer_with_data(
        &iters as *const u32 as *const std::ffi::c_void,
        4,
        MTLResourceOptions::StorageModeShared,
    );

    let queue = device.new_command_queue();
    let run = || {
        let cb = queue.new_command_buffer();
        let ce = cb.new_compute_command_encoder();
        ce.set_compute_pipeline_state(&pipeline);
        ce.set_buffer(0, Some(&buf), 0);
        ce.set_buffer(1, Some(&iters_buf), 0);
        ce.dispatch_thread_groups(
            metal::MTLSize::new(groups, 1, 1),
            metal::MTLSize::new(32, 1, 1),
        );
        ce.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    };

    run(); // warm up / JIT
    let start = Instant::now();
    run();
    let elapsed = start.elapsed().as_secs_f64();

    // 8x8x8 MMA = 8*8*8 = 512 MACs = 1024 FLOPs, times 16 chains per loop body.
    let flops = groups as f64 * iters as f64 * 16.0 * 1024.0;
    println!(
        "simdgroup MMAs: {} groups x {} iters = {:.3e} FLOPs in {:.3}s",
        groups, iters, flops, elapsed
    );
    println!("tensor-core peak: {:.2} TFLOPS", flops / elapsed / 1e12);

    // What that means for Gemma 4 31B prefill, whose GEMM is 5.75e10 FLOP/token.
    let gemm_flops_per_token = 5.75e10_f64;
    let tflops = flops / elapsed / 1e12;
    println!(
        "GEMM-only ceiling at this rate: {:.1} tok/s (attention at 0 ms)",
        tflops * 1e12 / gemm_flops_per_token
    );
    // And at 50% of peak, which is a realistic GEMM efficiency.
    let half = tflops * 0.5;
    println!(
        "GEMM-only ceiling at 50% of peak: {:.1} tok/s",
        half * 1e12 / gemm_flops_per_token
    );
}
