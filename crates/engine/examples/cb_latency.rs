//! Per-command-buffer overhead: commit a trivial kernel and wait, in a loop.
//! If this costs ~100 ms per iteration on this machine, every per-layer-CB
//! scheme is doomed and the "machine shift" in the notes is explained; if it
//! is ~0.1 ms, the overhead is content-dependent (CB size / resource count).

use metal::{Device, MTLResourceOptions};
use std::time::Instant;

const SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void trivial(device float *out [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
    if (gid < 1024) out[gid] = gid;
}
kernel void heavy(device float *out [[buffer(0)]], constant uint &iters [[buffer(1)]],
                  uint gid [[thread_position_in_grid]]) {
    float x = float(gid);
    for (uint i = 0; i < iters; ++i) x = fma(x, 1.0000001f, 0.0000001f);
    if (gid < 1024) out[gid] = x;
}
"#;

fn main() {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();
    let lib = device
        .new_library_with_source(SRC, &metal::CompileOptions::new())
        .unwrap();
    let out = device.new_buffer(8192, MTLResourceOptions::StorageModeShared);

    // 1. trivial kernel, one CB per dispatch, wait each time
    {
        let f = lib.get_function("trivial", None).unwrap();
        let p = device.new_compute_pipeline_state_with_function(&f).unwrap();
        // warm
        for _ in 0..5 {
            let cb = queue.new_command_buffer();
            let e = cb.new_compute_command_encoder();
            e.set_compute_pipeline_state(&p);
            e.set_buffer(0, Some(&out), 0);
            e.dispatch_thread_groups(metal::MTLSize::new(1, 1, 1), metal::MTLSize::new(32, 1, 1));
            e.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let n = 200;
        let t = Instant::now();
        for _ in 0..n {
            let cb = queue.new_command_buffer();
            let e = cb.new_compute_command_encoder();
            e.set_compute_pipeline_state(&p);
            e.set_buffer(0, Some(&out), 0);
            e.dispatch_thread_groups(metal::MTLSize::new(1, 1, 1), metal::MTLSize::new(32, 1, 1));
            e.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        println!(
            "trivial commit+wait: {:.4} ms/CB",
            t.elapsed().as_secs_f64() * 1e3 / n as f64
        );
    }

    // 2. same but waitless: commit, no wait (pure submit cost)
    {
        let f = lib.get_function("trivial", None).unwrap();
        let p = device.new_compute_pipeline_state_with_function(&f).unwrap();
        let n = 200;
        let cbs: Vec<_> = (0..n).map(|_| queue.new_command_buffer()).collect();
        let t = Instant::now();
        for cb in &cbs {
            let e = cb.new_compute_command_encoder();
            e.set_compute_pipeline_state(&p);
            e.set_buffer(0, Some(&out), 0);
            e.dispatch_thread_groups(metal::MTLSize::new(1, 1, 1), metal::MTLSize::new(32, 1, 1));
            e.end_encoding();
            cb.commit();
        }
        println!(
            "trivial commit-only: {:.4} ms/CB",
            t.elapsed().as_secs_f64() * 1e3 / n as f64
        );
        for cb in &cbs {
            cb.wait_until_completed();
        }
    }

    // 3. heavy kernel (~30 ms), one CB per dispatch, wait each
    {
        let f = lib.get_function("heavy", None).unwrap();
        let p = device.new_compute_pipeline_state_with_function(&f).unwrap();
        let iters: u32 = 2_000_000;
        let iters_buf = device.new_buffer_with_data(
            &iters as *const u32 as *const std::ffi::c_void,
            4,
            MTLResourceOptions::StorageModeShared,
        );
        let n = 20;
        let mut times = Vec::new();
        for _ in 0..n {
            let t = Instant::now();
            let cb = queue.new_command_buffer();
            let e = cb.new_compute_command_encoder();
            e.set_compute_pipeline_state(&p);
            e.set_buffer(0, Some(&out), 0);
            e.set_buffer(1, Some(&iters_buf), 0);
            e.dispatch_thread_groups(
                metal::MTLSize::new(2048, 1, 1),
                metal::MTLSize::new(32, 1, 1),
            );
            e.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            times.push(t.elapsed().as_secs_f64() * 1e3);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "heavy (~ms scale) commit+wait: min {:.2} med {:.2} max {:.2} ms",
            times[0],
            times[n / 2],
            times[n - 1]
        );
    }
}
