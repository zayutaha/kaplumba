//! What does this machine actually stream at?
//!
//! Every ceiling in this project has been computed from the 273 GB/s spec sheet.
//! This measures a plain strided read over a buffer far larger than any cache,
//! with nothing but an add in the loop, so the roofs stop being arithmetic.
use metal::{Device, MTLResourceOptions};
use std::time::Instant;
const SRC: &str = include_str!("../../kernels/src/bw.metal");
fn main() {
    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();
    let lib = device
        .new_library_with_source(SRC, &metal::CompileOptions::new())
        .expect("compile");
    let st = device
        .new_compute_pipeline_state_with_function(&lib.get_function("stream_sum", None).unwrap())
        .unwrap();
    let threads = 20 * 1024u64;
    for gib in [2u64, 6] {
        let bytes = gib * 1_000_000_000u64;
        let words = (bytes / 4) as u32;
        let data = device.new_buffer(bytes, MTLResourceOptions::StorageModeShared);
        let out = device.new_buffer(threads * 4, MTLResourceOptions::StorageModeShared);
        let mut best = f64::MAX;
        for round in 0..10 {
            let cb = queue.new_command_buffer();
            let e = cb.new_compute_command_encoder();
            e.set_compute_pipeline_state(&st);
            e.set_buffer(0, Some(&data), 0);
            e.set_buffer(1, Some(&out), 0);
            e.set_bytes(2, 4, &words as *const _ as *const std::ffi::c_void);
            e.dispatch_threads(
                metal::MTLSize::new(threads, 1, 1),
                metal::MTLSize::new(256, 1, 1),
            );
            e.end_encoding();
            let t = Instant::now();
            cb.commit();
            cb.wait_until_completed();
            if round >= 3 {
                best = best.min(t.elapsed().as_secs_f64());
            }
        }
        let gbps = bytes as f64 / best / 1e9;
        println!(
            "  stream {gib:>2} GB: {gbps:>6.1} GB/s  ({:>4.0}% of 273)  -> {:.1} tok/s on 12.06 GB",
            gbps / 273.0 * 100.0,
            gbps / 12.06
        );
    }
}
