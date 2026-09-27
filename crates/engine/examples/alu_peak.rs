//! Measured peak ALU throughput of this GPU, so the EXL3 decode roofline stops
//! being a guess. The previous estimate assumed a 2.4 GHz clock, which may be
//! wrong for an M4 Pro, and the whole "is 20 tok/s reachable" question turns on
//! this number.

use metal::{Device, MTLResourceOptions};
use std::time::Instant;

const SRC: &str = include_str!("../../kernels/src/alu.metal");

fn run(label: &str, threads: u64, groups: u64, iters: u32, f16: bool) {
    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();
    let lib = device.new_library_with_source(SRC, &metal::CompileOptions::new()).expect("compile");
    let name = if f16 { "alu_half" } else { "alu_int" };
    let st = device
        .new_compute_pipeline_state_with_function(&lib.get_function(name, None).unwrap())
        .unwrap();
    let n = 1 << 20;
    let a = device.new_buffer((n * 2) as u64, MTLResourceOptions::StorageModeShared);
    let b = device.new_buffer((n * 2) as u64, MTLResourceOptions::StorageModeShared);
    for _ in 0..3 {
        let cb = queue.new_command_buffer();
        let e = cb.new_compute_command_encoder();
        e.set_compute_pipeline_state(&st);
        e.set_buffer(0, Some(&a), 0);
        e.set_buffer(1, Some(&b), 0);
        e.set_bytes(2, 4, &iters as *const _ as *const std::ffi::c_void);
        e.dispatch_thread_groups(metal::MTLSize::new(groups, 1, 1), metal::MTLSize::new(256, 1, 1));
        e.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }
    let mut best = f64::MAX;
    for _ in 0..8 {
        let cb = queue.new_command_buffer();
        let e = cb.new_compute_command_encoder();
        e.set_compute_pipeline_state(&st);
        e.set_buffer(0, Some(&a), 0);
        e.set_buffer(1, Some(&b), 0);
        e.set_bytes(2, 4, &iters as *const _ as *const std::ffi::c_void);
        e.dispatch_thread_groups(metal::MTLSize::new(groups, 1, 1), metal::MTLSize::new(256, 1, 1));
        e.end_encoding();
        let t = Instant::now();
        cb.commit();
        cb.wait_until_completed();
        best = best.min(t.elapsed().as_secs_f64());
    }
    // alu_int: 4 int32 ops per iteration. alu_half: ~6 per iteration.
    let ops_per = if f16 { 12.0 } else { 8.0 };
    let total = threads as f64 * iters as f64 * ops_per;
    println!("  {label:<34} {:>7.2} e12 ops/s", total / best / 1e12);
}

fn main() {
    println!("measured peak, {threads} threads", threads = 20 * 2048);
    for (label, threads) in [("20k threads", 20_480u64), ("80k threads", 81_920)] {
        println!("{label}:");
        run("int32 (8 ops/iter, 4 chains)", threads, 256, 20_000, false);
        run("fp16 codebook (12 ops/iter)", threads, 256, 20_000, true);
    }
}
