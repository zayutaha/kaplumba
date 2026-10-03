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

/// `elem_bytes` is how many bytes one load instruction moves, and it fixes the
/// element count: `count = bytes / elem_bytes`. `out_words_per_thread` is how
/// many `u32` the kernel stores per thread, which differs between the uint4
/// arms (4) and the uint1/uint2 arms (1).
///
/// Both are parameters rather than constants so that the load-width arms read
/// *exactly* the same total bytes as the original `stream_uint4` arm and are
/// therefore directly comparable; `load_width_sensitivity` asserts the three
/// checksums agree, which is only true if they did.
fn run_probe_for_kernel(
    device: &Device,
    queue: &metal::CommandQueue,
    lib: &metal::Library,
    kernel_name: &str,
    threads: u64,
    tg_size: u64,
    gib: u64,
    elem_bytes: u64,
    out_words_per_thread: u64,
    sabotage: bool,
) -> Result<(f64, f64, f64, u32), String> {
    let pipeline = device
        .new_compute_pipeline_state_with_function(&lib.get_function(kernel_name, None).unwrap())
        .expect("pipeline");

    let bytes = gib * 1_000_000_000u64;
    let mut count = (bytes / elem_bytes) as u32;

    if sabotage {
        // Sabotage: artificially halve the count to simulate partial load bug
        count /= 2;
    }

    let res_opts = MTLResourceOptions::StorageModeShared
        | MTLResourceOptions::HazardTrackingModeUntracked;
    let data = device.new_buffer(bytes, res_opts);
    fill_high_entropy_buffer(&data, bytes as usize);

    let out = device.new_buffer(threads * out_words_per_thread * 4, res_opts);

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
        for i in 0..(threads as usize * out_words_per_thread as usize) {
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
    let normal = run_probe_for_kernel(device, queue, lib, "stream_uint4", 40960, 256, 1, 16, 4, false).unwrap();
    let sabotaged = run_probe_for_kernel(device, queue, lib, "stream_uint4", 40960, 256, 1, 16, 4, true).unwrap();

    println!("Normal sum:    {:#010x}", normal.3);
    println!("Sabotaged sum: {:#010x}", sabotaged.3);

    assert_ne!(
        normal.3, sabotaged.3,
        "Sabotage must alter checksum! Failed to detect halved read count."
    );
    println!("Sabotage detected successfully: checksums differ as expected. Fail-first passed.\n");
}

/// Is this box's streaming throughput sensitive to load-INSTRUCTION count at
/// constant bytes?
///
/// This is the falsifier for the wide-lane-window conclusion (`findings/F6`).
/// The wide window moves **no** DRAM traffic -- it reads the same tiles -- it
/// only removes load instructions. So if throughput is flat from 4-byte to
/// 16-byte loads, load-instruction count is not what limits this kernel and the
/// window cannot pay, whatever its arithmetic is worth.
///
/// Three arms, **the same bytes in the same grid-stride order**, differing only
/// in the width of one load:
///   stream_uint1  4 B/load    1 load per 4 bytes
///   stream_uint2  8 B/load    1 load per 8 bytes
///   stream_uint4 16 B/load    1 load per 16 bytes
///
/// Arithmetic intensity per byte is identical (one u32 add per four bytes), so
/// the only variable is how many load instructions issue.
///
/// **The gate.** All three checksums must be equal. Adding the same multiset of
/// u32s in groups of 1, 2 or 4 is the same sum, so an arm that read a different
/// number of bytes, or a different range, cannot produce it. That equality is
/// what makes this a discriminator rather than three unrelated numbers.
fn load_width_sensitivity(device: &Device, queue: &metal::CommandQueue, lib: &metal::Library) {
    const GIB: u64 = 6;
    const THREADS: u64 = 40960;
    const TG: u64 = 256;
    const ROUNDS: usize = 9;

    let arms: [(&str, u64, u64); 3] = [
        ("stream_uint1", 4, 1),   // (kernel, bytes per load, u32 written per thread)
        ("stream_uint2", 8, 1),
        ("stream_uint4", 16, 4),
    ];

    let bytes = GIB * 1_000_000_000u64;
    let res_opts = MTLResourceOptions::StorageModeShared
        | MTLResourceOptions::HazardTrackingModeUntracked;

    // One buffer, filled once, shared by every arm. The original harness
    // reallocates and refills per call, which at 6 GB costs seconds; the point
    // here is an interleaved comparison, so the buffer must be constant.
    let data = device.new_buffer(bytes, res_opts);
    fill_high_entropy_buffer(&data, bytes as usize);
    let out = device.new_buffer(THREADS * 4 * 4, res_opts);

    let pipes: Vec<metal::ComputePipelineState> = arms
        .iter()
        .map(|(name, _, _)| {
            device
                .new_compute_pipeline_state_with_function(
                    &lib.get_function(name, None).unwrap_or_else(|e| panic!("{name}: {e}")),
                )
                .unwrap_or_else(|e| panic!("pipeline {name}: {e}"))
        })
        .collect();

    let mut best = [f64::MAX; 3];
    let mut sums = [0u32; 3];

    // Warm every arm once before timing anything (AGENTS.md 10.5: cold clocks).
    for ai in 0..3 {
        let count = (bytes / arms[ai].1) as u32;
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipes[ai]);
        enc.set_buffer(0, Some(&data), 0);
        enc.set_buffer(1, Some(&out), 0);
        enc.set_bytes(2, 4, &count as *const u32 as *const std::ffi::c_void);
        enc.dispatch_threads(MTLSize::new(THREADS, 1, 1), MTLSize::new(TG, 1, 1));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    for _ in 0..ROUNDS {
        for ai in 0..3 {
            let count = (bytes / arms[ai].1) as u32;
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipes[ai]);
            enc.set_buffer(0, Some(&data), 0);
            enc.set_buffer(1, Some(&out), 0);
            enc.set_bytes(2, 4, &count as *const u32 as *const std::ffi::c_void);
            enc.dispatch_threads(MTLSize::new(THREADS, 1, 1), MTLSize::new(TG, 1, 1));
            enc.end_encoding();
            let t0 = Instant::now();
            cb.commit();
            cb.wait_until_completed();
            let el = t0.elapsed().as_secs_f64();
            if el < best[ai] {
                best[ai] = el;
            }

            let out_ptr = out.contents() as *const u32;
            let mut s = 0u32;
            for i in 0..(THREADS as usize * arms[ai].2 as usize) {
                unsafe {
                    s = s.wrapping_add(*out_ptr.add(i));
                }
            }
            sums[ai] = s;
        }
    }

    println!("\n=== LOAD-WIDTH SENSITIVITY ({GIB} GB, th={THREADS} tg={TG}, {ROUNDS} interleaved rounds, min) ===");
    assert_eq!(
        sums[0], sums[1],
        "stream_uint1 and stream_uint2 disagree on the checksum ({:#010x} vs {:#010x}). \
         They read the same bytes in the same order, so they MUST agree. An arm is not \
         reading what this comparison assumes and the numbers below are void.",
        sums[0], sums[1]
    );
    assert_eq!(
        sums[0], sums[2],
        "stream_uint1 and stream_uint4 disagree on the checksum ({:#010x} vs {:#010x}). \
         Same bytes, same order, so they MUST agree; the comparison is void.",
        sums[0], sums[2]
    );
    println!("  GATE: all three arms read the same bytes (checksum {:#010x})\n", sums[0]);

    let mut gbps = [0.0f64; 3];
    println!(
        "  {:<16} {:>10} {:>12} {:>10}",
        "arm", "B/load", "GB/s", "loads/GB"
    );
    for ai in 0..3 {
        gbps[ai] = bytes as f64 / best[ai] / 1e9;
        println!(
            "  {:<16} {:>10} {:>12.1} {:>10.3}",
            arms[ai].0,
            arms[ai].1,
            gbps[ai],
            1e9 / arms[ai].1 as f64
        );
    }

    let spread = gbps[2].max(gbps[0]) / gbps[2].min(gbps[0]).min(gbps[1]);
    println!(
        "\n  uint2 / uint1 = {:+.1}%   (half the load instructions, same bytes)",
        (gbps[1] - gbps[0]) / gbps[0] * 100.0
    );
    println!(
        "  uint4 / uint1 = {:+.1}%   (a quarter of the load instructions, same bytes)",
        (gbps[2] - gbps[0]) / gbps[0] * 100.0
    );

    let gain = (gbps[2] - gbps[0]) / gbps[0] * 100.0;
    if gain.abs() < 5.0 {
        println!(
            "  => **LOAD-INSTRUCTION COUNT IS NOT THE WALL** (widening 4 B -> 16 B is worth \
             {gain:+.1}%). A kernel change that only removes load instructions -- the wide lane \
             window, `uint4`/`uint2` widening -- cannot pay, whatever its arithmetic."
        );
    } else {
        println!(
            "  => **LOAD-INSTRUCTION COUNT IS A REAL LEVER** ({gain:+.1}% for a 4x reduction). \
             The wide lane window is worth pursuing; it removes about a quarter of the weight \
             load instructions."
        );
    }
    let _ = spread;
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
                &device, &queue, &lib, "stream_uint4", threads, tg_size, gib, 16, 4, false
            ).unwrap();
            let (med, best, worst, sum) = res;
            let pct = (med / 273.0) * 100.0;
            println!(
                "stream_uint4 | th={:>5}, tg={:>3} | med: {:>5.1} GB/s ({:>4.1}%) | best: {:>5.1} | worst: {:>5.1} | sum: {:#010x}",
                threads, tg_size, med, pct, best, worst, sum
            );
        }
    }

    load_width_sensitivity(&device, &queue, &lib);
}
