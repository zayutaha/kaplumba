//! Isolated GEMV bandwidth microbenchmark at production shapes.
//!
//! Why this exists: `o_proj` (K=8192) measured ~100 GB/s while `down_proj`
//! (K=21504) measured ~200 GB/s **through the same `colmajor_2col_sk2`
//! kernel**. End-to-end tok/s cannot separate "this kernel is slow" from
//! "this shape is small", and a 7.5 s model load per data point makes
//! iterating on kernel variants impractical. This runs the real dispatch
//! paths on synthetic buffers and reads the GPU's own clock, so a point
//! costs milliseconds and K can be swept against N.
//!
//! Three things make the numbers mean something:
//!
//! * **GPU clock, not wall clock.** The first version timed wall clock and
//!   found every point launch-bound: ~44 us of CPU per dispatch against
//!   ~50 us of GPU work, so it measured the CPU. metal-rs 0.33 has no
//!   timing-info descriptor, so this uses the engine's own
//!   [`CommandBufferGpuTiming`] trait, which reaches `GPUStartTime` /
//!   `GPUEndTime` through the ObjC runtime. One command buffer per
//!   measurement, so the span is GPU-only and launch latency is excluded.
//! * **No cache reuse.** Production streams 10.78 GB of distinct weights per
//!   token; re-reading one 43 MB matrix repeatedly can be served partly from
//!   L2 and reports bandwidth the real workload never sees. Rotating over
//!   several buffers keeps the working set far larger than any plausible L2.
//! * **Long runs.** A 4.7 ms measurement reported a 3.5x spread between
//!   adjacent K values -- clock ramp, not the kernel. Each point calibrates
//!   then runs for tens of ms.
//!
//! Buffer contents are pseudo-random `u16`, not valid codes: the kernel
//! extracts bit patterns and this measures streaming bandwidth, not
//! correctness. Correctness is covered by `linear_metal.rs` and
//! `live_kernel_bit_exact.rs`.
//!
//! Run with:
//!   cargo test --release --test gemv_bandwidth -- --ignored --nocapture

use std::ffi::c_void;
use std::mem::size_of;
use std::time::Instant;

use kaplumba_kernels::encoder::{CommandBufferGpuTiming, MetalEncoder};
use kaplumba_kernels::exl3::{Exl3Codebook, Exl3Rate};
use kaplumba_kernels::linear::LinearKernels;
use metal::{Buffer, CommandQueue, Device, MTLResourceOptions};

/// Words per 16x16 trellis tile at 3 bpw: 256 codes * 3 bits = 768 bits = 24
/// `u32` = 48 `u16`. Mirrors `RING/WORDS` in `linear_repacked.metal`.
const U16_PER_TILE_3BPW: usize = 48;

/// Aim for this much GPU time per data point: long enough that clock ramp and
/// dispatch startup are a rounding error.
const TARGET_MS: f64 = 40.0;
/// Working-set bounds for the buffer rotation, so the sweep cannot exhaust
/// memory and cannot accidentally fit in cache.
const MAX_WORKING_SET: usize = 768 << 20;
const MIN_WORKING_SET: usize = 192 << 20;

fn random_buffer(device: &Device, bytes: usize, seed: u32) -> Buffer {
    let mut state = seed | 1;
    let words = bytes / 2;
    let mut data: Vec<u16> = Vec::with_capacity(words);
    for _ in 0..words {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        data.push(state as u16);
    }
    // `new_buffer_with_data` copies, so `data` is free to drop here.
    device.new_buffer_with_data(
        data.as_ptr() as *const c_void,
        (data.len() * size_of::<u16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

struct Case {
    label: &'static str,
    k: usize,
    n: usize,
    splits: usize,
    use_sk: bool,
}

impl Case {
    fn shape_bytes(&self) -> usize {
        self.k * self.n * 3 / 8
    }
}

#[allow(clippy::too_many_arguments)]
fn dispatch(
    kernels: &LinearKernels,
    enc: &mut MetalEncoder,
    packed: &Buffer,
    transformed: &Buffer,
    raw: &Buffer,
    case: &Case,
) {
    let rate = Exl3Rate::Integer(3);
    let codebook = Exl3Codebook::Mcg;
    let (k, n) = (case.k, case.n);
    if !case.use_sk {
        kernels
            .encode_gemv_raw(enc, packed, 0, 1, rate, codebook, k, n, transformed, raw)
            .expect("gemv_raw");
    } else if case.splits == 2 {
        kernels
            .encode_gemv_raw_sk2(enc, packed, 0, 1, rate, k, n, transformed, raw)
            .expect("gemv_raw_sk2");
    } else {
        kernels
            .encode_gemv_raw_sk4(enc, packed, 0, 1, rate, k, n, transformed, raw)
            .expect("gemv_raw_sk4");
    }
}

/// Run one case and return its achieved GB/s.
fn run(
    kernels: &LinearKernels,
    device: &Device,
    queue: &CommandQueue,
    case: &Case,
    fixed_reps: Option<usize>,
) -> f64 {
    let shape_bytes = case.shape_bytes();
    let nbuf = (MAX_WORKING_SET / shape_bytes.max(1))
        .clamp(2, 8)
        .min((MIN_WORKING_SET / shape_bytes.max(1)).max(2));
    let packed: Vec<Buffer> = (0..nbuf)
        .map(|i| {
            random_buffer(
                device,
                (shape_bytes / 4) * 2,
                0x1234_5678u32.wrapping_add(i as u32 * 7919),
            )
        })
        .collect();
    let transformed = random_buffer(device, case.k * 2, 0x9abc_def0);
    let raw = random_buffer(device, (case.n * case.splits.max(1)) * 4, 0xfeed_beef);

    let encode = |enc: &mut MetalEncoder, i: usize| {
        dispatch(
            kernels,
            enc,
            &packed[i % packed.len()],
            &transformed,
            &raw,
            case,
        );
    };

    // Calibrate on 4 reps of wall time to size the real run.
    let reps = match fixed_reps {
        Some(r) => r,
        None => {
            let mut enc = MetalEncoder::new(queue).expect("encoder");
            enc.set_max_dispatches(u32::MAX);
            let t0 = Instant::now();
            for i in 0..4 {
                encode(&mut enc, i);
            }
            enc.commit_and_wait().expect("cal");
            let per = (t0.elapsed().as_secs_f64() * 1e3 / 4.0).max(1e-4);
            ((TARGET_MS / per).round() as usize).clamp(4, 2000)
        }
    };

    // Warm the page tables and the clocks; discarded.
    {
        let mut enc = MetalEncoder::new(queue).expect("encoder");
        for i in 0..reps.min(8) {
            encode(&mut enc, i);
        }
        enc.commit_and_wait().expect("warmup");
    }

    // One command buffer for the whole run, so its GPU span covers exactly
    // `reps` dispatches and nothing else.
    let mut enc = MetalEncoder::new(queue).expect("encoder");
    // The encoder rolls to a fresh command buffer every `max_dispatches`
    // (default well under the counts used here), and `commit_stamped` only
    // parks the *current* buffer -- so without this the GPU span covers a
    // fraction of the work and reports impossible bandwidth.
    enc.set_max_dispatches(u32::MAX);
    for i in 0..reps {
        encode(&mut enc, i);
    }
    enc.commit_stamped(case.label).expect("commit");
    // `take_gpu_stamps` waits on the parked buffer before reading it.
    let gpu_ms: f64 = enc.take_gpu_stamps().iter().map(|s| s.own_ms).sum();
    enc.commit_and_wait().ok();

    let gbps = (shape_bytes as f64 * reps as f64) / (gpu_ms.max(1e-9) * 1e-3) / 1e9;
    let note = if gpu_ms <= 0.0 {
        "  <-- NO GPU TIMING (set KAPLUMBA_DECODE_PROFILE=kernel or check trait)"
    } else {
        ""
    };
    println!(
        "{label:<22} K={k:<6} N={n:<6} sk={splits}  bufs={nbuf:<2} reps={reps:<5} gpu={gpu_ms:8.2} ms  = {gbps:7.1} GB/s{note}",
        label = case.label,
        k = case.k,
        n = case.n,
        splits = case.splits,
    );
    gbps
}

/// Sweep K at fixed N, anchored on the two production shapes that showed the
/// 2x discrepancy.
#[test]
#[ignore = "microbenchmark; run explicitly"]
fn gemv_bandwidth_sweep() {
    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();
    let kernels = LinearKernels::new().expect("linear kernels compile");

    println!("\n=== production anchor: N=5376 (o_proj, down_proj), sweeping K ===");
    for (label, splits, use_sk) in [
        ("colmajor_2col_sk2", 2usize, true),
        ("colmajor_2col", 1, false),
        ("colmajor_2col_sk4", 4, true),
    ] {
        for k in [4096usize, 8192, 16384, 21504, 32768] {
            run(
                &kernels,
                &device,
                &queue,
                &Case { label, k, n: 5376, splits, use_sk },
                None,
            );
        }
    }

    println!("\n=== K=5376 (qkv / gate_up input width), sweeping N ===");
    for nn in [4096usize, 8192, 21504, 43008] {
        run(
            &kernels,
            &device,
            &queue,
            &Case { label: "colmajor_2col_sk2", k: 5376, n: nn, splits: 2, use_sk: true },
            None,
        );
    }
}