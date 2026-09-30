//! Memory bandwidth probe targeting 250+ GB/s on Apple Silicon M4 Pro
//!
//! Explores vector width (uint4 = 16 bytes), threadgroup sizing,
//! grid concurrency (40k to 81k threads), and layer-chunked streaming
//! over 2 GB, 6 GB, and 12 GB buffers.
//!
//! Full entropy buffer filling ensures every cacheline is dirty and non-compressible.
//! Includes fail-first sabotage verification per repository rules.

use metal::{CompileOptions, Device, MTLResourceOptions, MTLSize};
use std::time::Instant;

const SRC: &str = include_str!("../../kernels/src/bw_probe.metal");

fn fill_high_entropy_buffer(data: &metal::Buffer, bytes: usize) {
    let ptr = data.contents() as *mut u8;
    let pattern_bytes = 64 * 1024 * 1024;
    let mut pattern = vec![0u8; pattern_bytes];
    let mut x: u32 = 0x9e37_79b9;
    for chunk in pattern.chunks_exact_mut(4) {
        x = x.wrapping_mul(1664525).wrapping_add(1013904223);
        chunk.copy_from_slice(&x.to_le_bytes());
    }

    let mut offset = 0;
    while offset < bytes {
        let chunk_size = std::cmp::min(pattern_bytes, bytes - offset);
        unsafe {
            std::ptr::copy_nonoverlapping(pattern.as_ptr(), ptr.add(offset), chunk_size);
        }
        offset += chunk_size;
    }
}

fn run_probe_for_kernel(
    device: &Device,
    queue: &metal::CommandQueue,
    lib: &metal::Library,
    kernel_name: &str,
    threads: u64,
    tg_size: u64,
    gib: u64,
    sabotage: bool,
) -> Result<(f64, f64, f64, u32), String> {
    let pipeline = device
        .new_compute_pipeline_state_with_function(&lib.get_function(kernel_name, None).unwrap())
        .expect("pipeline");

    let bytes = gib * 1_000_000_000u64;
    let mut count = (bytes / 16) as u32;

    if sabotage {
        // Sabotage: artificially halve the count to simulate partial load bug
        count /= 2;
    }

    let res_opts = MTLResourceOptions::StorageModeShared
        | MTLResourceOptions::HazardTrackingModeUntracked;
    let data = device.new_buffer(bytes, res_opts);
    fill_high_entropy_buffer(&data, bytes as usize);

    let out = device.new_buffer(threads * 16, res_opts);

    let mut times: Vec<f64> = Vec::new();
    let mut last_gpu_sum = 0u32;

    for round in 0..10 {
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&data), 0);
        enc.set_buffer(1, Some(&out), 0);
        enc.set_bytes(2, 4, &count as *const _ as *const std::ffi::c_void);

        enc.dispatch_threads(
            MTLSize::new(threads, 1, 1),
            MTLSize::new(tg_size, 1, 1),
        );
        enc.end_encoding();

        let t0 = Instant::now();
        cb.commit();
        cb.wait_until_completed();
        let elapsed = t0.elapsed().as_secs_f64();

        if round >= 3 {
            times.push(elapsed);
        }

        let out_ptr = out.contents() as *const u32;
        let mut sum = 0u32;
        for i in 0..(threads as usize * 4) {
            unsafe {
                sum = sum.wrapping_add(*out_ptr.add(i));
            }
        }
        last_gpu_sum = sum;
    }

    // Verification check: ensure sum is not zero
    if last_gpu_sum == 0 {
        return Err("Verification failed: GPU sum is 0 (no data read)".to_string());
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let best = *times.first().unwrap();
    let median = times[times.len() / 2];
    let worst = *times.last().unwrap();

    let best_gbps = bytes as f64 / best / 1e9;
    let med_gbps = bytes as f64 / median / 1e9;
    let worst_gbps = bytes as f64 / worst / 1e9;

    Ok((med_gbps, best_gbps, worst_gbps, last_gpu_sum))
}

fn test_sabotage(device: &Device, queue: &metal::CommandQueue, lib: &metal::Library) {
    println!("\n--- T2: Fail-First Sabotage Verification ---");
    let normal = run_probe_for_kernel(device, queue, lib, "stream_uint4", 40960, 256, 1, false).unwrap();
    let sabotaged = run_probe_for_kernel(device, queue, lib, "stream_uint4", 40960, 256, 1, true).unwrap();

    println!("Normal sum:    {:#010x}", normal.3);
    println!("Sabotaged sum: {:#010x}", sabotaged.3);

    assert_ne!(
        normal.3, sabotaged.3,
        "Sabotage must alter checksum! Failed to detect halved read count."
    );
    println!("Sabotage detected successfully: checksums differ as expected. Fail-first passed.\n");
}

fn main() {
    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();
    let opts = CompileOptions::new();
    let lib = device.new_library_with_source(SRC, &opts).expect("compile metal");

    // Run T2 Fail-First Sabotage Check
    test_sabotage(&device, &queue, &lib);

    println!("=== M4 Pro Bandwidth Probe (Untracked, Full Entropy) ===");

    let configs = [
        (40960u64, 256u64),
        (81920u64, 256u64),
        (81920u64, 512u64),
    ];

    for &gib in &[2u64, 6, 12] {
        println!("\n--- Testing Buffer Size: {} GB ---", gib);
        for &(threads, tg_size) in &configs {
            let res = run_probe_for_kernel(
                &device, &queue, &lib, "stream_uint4", threads, tg_size, gib, false
            ).unwrap();
            let (med, best, worst, sum) = res;
            let pct = (med / 273.0) * 100.0;
            println!(
                "stream_uint4 | th={:>5}, tg={:>3} | med: {:>5.1} GB/s ({:>4.1}%) | best: {:>5.1} | worst: {:>5.1} | sum: {:#010x}",
                threads, tg_size, med, pct, best, worst, sum
            );
        }
    }
}
