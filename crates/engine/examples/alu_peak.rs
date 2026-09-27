//! Per-op-class peak throughput. A multiply-limited loop cannot stand in for
//! general ALU, and fp16 may issue as half2, so each class is measured alone.
use metal::{Device, MTLResourceOptions};
use std::time::Instant;

const SRC: &str = include_str!("../../kernels/src/alu.metal");

fn main() {
    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();
    let lib = device.new_library_with_source(SRC, &metal::CompileOptions::new()).expect("compile");
    let threads = 81_920u64;
    let groups = 320u64;
    let iters = 20_000u32;

    for (name, ops_per_iter, note) in [
        ("alu_logic", 12.0, "shift/and/xor only (extraction mix)"),
        ("alu_mul",   12.0, "1 mul + 1 add per chain (MCG-style)"),
        ("alu_half2", 12.0, "fp16 as half2, 2 codes per lane"),
        ("alu_fma",    8.0, "fp32 FMA (calibration)"),
    ] {
        let st = device
            .new_compute_pipeline_state_with_function(&lib.get_function(name, None).unwrap())
            .unwrap();
        let a = device.new_buffer(1 << 21, MTLResourceOptions::StorageModeShared);
        let b = device.new_buffer(1 << 21, MTLResourceOptions::StorageModeShared);
        let mut best = f64::MAX;
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
        let ops = threads as f64 * iters as f64 * ops_per_iter;
        let rate = ops / best / 1e12;
        println!("  {name:<10} {rate:>6.2} e12 ops/s   {note}");
    }
}
