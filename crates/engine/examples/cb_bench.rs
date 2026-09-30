//! Codebook cost: hash versus table lookup, at the real rate.
use metal::{Device, MTLResourceOptions};
use std::time::Instant;
const SRC: &str = include_str!("../../kernels/src/cb.metal");
fn main() {
    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();
    let lib = device
        .new_library_with_source(SRC, &metal::CompileOptions::new())
        .expect("compile");
    // 32.2e9 codes per token is one token's worth; use 256e6 as a tractable proxy
    // and report rates normalised to weights/second.
    let n = 256_000_000u32;
    let a = device.new_buffer((n as u64) * 4, MTLResourceOptions::StorageModeShared);
    let b = device.new_buffer(81_920 * 4, MTLResourceOptions::StorageModeShared);
    unsafe {
        let p = a.contents() as *mut u32;
        for i in 0..n as usize {
            *p.add(i) = (i as u32).wrapping_mul(2654435761);
        }
    }
    for (name, extra) in [("cb_hash", false), ("cb_lut", true)] {
        let st = device
            .new_compute_pipeline_state_with_function(&lib.get_function(name, None).unwrap())
            .unwrap();
        let mut best = f64::MAX;
        for _ in 0..3 {
            let cb = queue.new_command_buffer();
            let e = cb.new_compute_command_encoder();
            e.set_compute_pipeline_state(&st);
            e.set_buffer(0, Some(&a), 0);
            e.set_buffer(1, Some(&b), 0);
            e.set_bytes(2, 4, &n as *const _ as *const std::ffi::c_void);
            if extra {
                e.set_bytes(3, 4, &(16u32 - 3) as *const _ as *const std::ffi::c_void);
            }
            e.dispatch_thread_groups(
                metal::MTLSize::new(320, 1, 1),
                metal::MTLSize::new(256, 1, 1),
            );
            e.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        for _ in 0..6 {
            let cb = queue.new_command_buffer();
            let e = cb.new_compute_command_encoder();
            e.set_compute_pipeline_state(&st);
            e.set_buffer(0, Some(&a), 0);
            e.set_buffer(1, Some(&b), 0);
            e.set_bytes(2, 4, &n as *const _ as *const std::ffi::c_void);
            if extra {
                e.set_bytes(3, 4, &(16u32 - 3) as *const _ as *const std::ffi::c_void);
            }
            e.dispatch_thread_groups(
                metal::MTLSize::new(320, 1, 1),
                metal::MTLSize::new(256, 1, 1),
            );
            e.end_encoding();
            let t = Instant::now();
            cb.commit();
            cb.wait_until_completed();
            best = best.min(t.elapsed().as_secs_f64());
        }
        let per_s = n as f64 / best;
        println!(
            "  {:<8} {:>6.2} e9 codes/s -> {:>5.1} tok/s ceiling for 32.2e9 codes",
            name,
            per_s / 1e9,
            per_s / 32.2e9
        );
    }
}
