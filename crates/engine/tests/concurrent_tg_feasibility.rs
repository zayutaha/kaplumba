//! T1: Verify whether Metal schedules concurrent threadgroups within a single
//! command buffer.
//!
//! Dispatches two compute kernels sequentially into the same command encoder:
//!   Kernel A: writes 1 to an atomic flag after a busy-wait loop (~1 ms of work)
//!   Kernel B: spin-waits on the flag, then writes a marker to output
//!
//! If output[0] == 0xCAFE within a reasonable time, Metal ran them concurrently.
//! If the test times out (> 5 s), Metal serializes dispatches within a command
//! buffer and the warmstart approach (T2a) is not feasible.
//!
//! spec/decode-20tps: T1

use metal::{
    CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};
use std::time::Instant;

const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Kernel A: does ~1 ms of busy work, then writes flag = 1
kernel void signal_after_work(
    device atomic_uint& flag           [[buffer(0)]],
    device float*       scratch        [[buffer(1)]],
    constant uint&      iterations     [[buffer(2)]],
    uint                tid            [[thread_position_in_grid]]
) {
    // Busy loop to simulate ~1 ms of work
    float acc = float(tid);
    for (uint i = 0; i < iterations; i++) {
        acc = fma(acc, 1.0001f, 0.0001f);
    }
    // Prevent dead code elimination
    scratch[tid] = acc;
    // Only thread 0 signals
    if (tid == 0u) {
        atomic_store_explicit(&flag, 1u, memory_order_relaxed);
    }
}

// Kernel B: spin-waits on flag, then writes marker
kernel void wait_and_mark(
    device const atomic_uint& flag     [[buffer(0)]],
    device uint*              output   [[buffer(1)]],
    uint                      tid      [[thread_position_in_grid]]
) {
    if (tid == 0u) {
        // Record start: write 0xDEAD to prove we ran
        output[1] = 0xDEADu;

        // Spin-wait for flag
        uint spins = 0u;
        while (atomic_load_explicit(&flag, memory_order_relaxed) == 0u) {
            spins++;
            // Safety valve: don't spin forever on the GPU
            if (spins > 100000000u) {
                output[0] = 0xTIMEu;  // timed out on GPU
                return;
            }
        }
        output[0] = 0xCAFEu;  // success marker
        output[2] = spins;     // how many spins it took
    }
}
"#;

// Fix the 0xTIME issue - use a valid hex literal
const SOURCE_FIXED: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Kernel A: does ~1 ms of busy work, then writes flag = 1.
// Dispatched with 256 threads to occupy one threadgroup.
kernel void signal_after_work(
    device atomic_uint& flag           [[buffer(0)]],
    device float*       scratch        [[buffer(1)]],
    constant uint&      iterations     [[buffer(2)]],
    uint                tid            [[thread_position_in_grid]]
) {
    float acc = float(tid + 1u);
    for (uint i = 0; i < iterations; i++) {
        acc = fma(acc, 1.0001f, 0.0001f);
    }
    scratch[tid] = acc;
    if (tid == 0u) {
        atomic_store_explicit(&flag, 1u, memory_order_relaxed);
    }
}

// Kernel B: dispatched in a separate threadgroup.
// Spin-waits on the flag, then writes a success marker.
kernel void wait_and_mark(
    device const atomic_uint& flag     [[buffer(0)]],
    device uint*              output   [[buffer(1)]],
    uint                      tid      [[thread_position_in_grid]]
) {
    if (tid == 0u) {
        output[1] = 0xDEADu;
        uint spins = 0u;
        while (atomic_load_explicit(&flag, memory_order_relaxed) == 0u) {
            spins++;
            if (spins > 500000000u) {
                output[0] = 0xBADDu;
                return;
            }
        }
        output[0] = 0xCAFEu;
        output[2] = spins;
    }
}
"#;

fn compile_pipeline(device: &Device, source: &str, name: &str) -> ComputePipelineState {
    let options = CompileOptions::new();
    let lib = device
        .new_library_with_source(source, &options)
        .expect("Metal shader compilation");
    let func = lib.get_function(name, None).expect("function lookup");
    device
        .new_compute_pipeline_state_with_function(&func)
        .expect("pipeline creation")
}

#[test]
fn t1_concurrent_threadgroup_scheduling() {
    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();

    let signal_pipeline = compile_pipeline(&device, SOURCE_FIXED, "signal_after_work");
    let wait_pipeline = compile_pipeline(&device, SOURCE_FIXED, "wait_and_mark");

    // Shared atomic flag (4 bytes, init to 0)
    let flag_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    unsafe {
        *(flag_buf.contents() as *mut u32) = 0;
    }

    // Scratch for kernel A (256 floats)
    let scratch_buf = device.new_buffer(256 * 4, MTLResourceOptions::StorageModeShared);

    // Iterations param — enough to keep kernel A busy for ~1 ms
    // 256 threads × 100000 iterations ≈ 25.6M FMAs ≈ 1-2 ms on M4 Pro
    let iterations: u32 = 100_000;
    let iter_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    unsafe {
        *(iter_buf.contents() as *mut u32) = iterations;
    }

    // Output for kernel B (3 u32s: marker, ran-flag, spin-count)
    let output_buf = device.new_buffer(12, MTLResourceOptions::StorageModeShared);
    unsafe {
        let ptr = output_buf.contents() as *mut u32;
        *ptr = 0;
        *ptr.add(1) = 0;
        *ptr.add(2) = 0;
    }

    let start = Instant::now();

    // Encode BOTH dispatches into the SAME command buffer
    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();

    // Dispatch A: signal_after_work — 1 threadgroup of 256 threads
    enc.set_compute_pipeline_state(&signal_pipeline);
    enc.set_buffer(0, Some(&flag_buf), 0);
    enc.set_buffer(1, Some(&scratch_buf), 0);
    enc.set_buffer(2, Some(&iter_buf), 0);
    enc.dispatch_thread_groups(
        MTLSize::new(1, 1, 1),
        MTLSize::new(256, 1, 1),
    );

    // Dispatch B: wait_and_mark — 1 threadgroup of 32 threads
    enc.set_compute_pipeline_state(&wait_pipeline);
    enc.set_buffer(0, Some(&flag_buf), 0);
    enc.set_buffer(1, Some(&output_buf), 0);
    enc.dispatch_thread_groups(
        MTLSize::new(1, 1, 1),
        MTLSize::new(32, 1, 1),
    );

    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    let elapsed = start.elapsed();

    // Read results
    let marker = unsafe { *(output_buf.contents() as *const u32) };
    let ran = unsafe { *(output_buf.contents() as *const u32).add(1) };
    let spins = unsafe { *(output_buf.contents() as *const u32).add(2) };

    eprintln!("=== T1: Concurrent Threadgroup Scheduling Test ===");
    eprintln!("  Wall time:    {:.1} ms", elapsed.as_secs_f64() * 1000.0);
    eprintln!("  Marker:       0x{marker:04X}");
    eprintln!("  Kernel B ran: {}", if ran == 0xDEAD { "yes" } else { "no" });
    eprintln!("  Spin count:   {spins}");

    if marker == 0xCAFE {
        eprintln!("  RESULT: ✅ CONCURRENT — Metal scheduled both threadgroups simultaneously.");
        eprintln!("  → Proceed with T2a (warmstart kernel with atomic flag).");
    } else if marker == 0xBADD {
        eprintln!("  RESULT: ❌ GPU TIMEOUT — Kernel B spin-waited too long.");
        eprintln!("  Metal likely serializes dispatches within a command buffer.");
        eprintln!("  → Fall back to T2b (dual-queue with MTLEvent).");
    } else if marker == 0 && ran == 0 {
        eprintln!("  RESULT: ❌ SERIALIZED — Kernel B never ran (Metal serialized).");
        eprintln!("  → Fall back to T2b (dual-queue with MTLEvent).");
    } else {
        eprintln!("  RESULT: ❓ UNEXPECTED marker=0x{marker:04X} ran=0x{ran:04X}");
    }

    // The test passes either way — it's a feasibility probe, not a correctness test.
    // We just need the answer, not a specific outcome.
    assert!(
        elapsed.as_secs() < 10,
        "Test took too long ({:.1}s) — possible GPU hang",
        elapsed.as_secs_f64()
    );
}
