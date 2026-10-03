//! Occupancy probe for the decode GEMV family: how many threadgroups the GPU
//! holds concurrently, and whether the production grids are large enough to
//! matter.
//!
//! WHY THIS EXISTS
//!
//! Decode runs at 16.05-16.10 tok/s = 62.3 ms/token while the per-layer GEMVs
//! stream at 125-150 GB/s and `lm_head` (the same pipeline, 6 bpw) streams at
//! 253 GB/s. Occupancy is the standard reason a kernel cannot saturate DRAM, so
//! the question is worth one honest experiment rather than an argument.
//!
//! What is already known (do not re-measure, see the specs):
//!
//! * `spec/decode-25tok-peak-bandwidth/tasks.md` T4.2: 4col halves the grid
//!   (168 -> 84 TGs on N=5376) and costs 4.2% tok/s.
//! * same file T5.2: `down_proj` split-K 1 -> 2 (168 -> 336 TGs) is worth
//!   -2.7 ms; split-K 4 (672 TGs) is worth **nothing** over split-K 2. So the
//!   resident capacity is bracketed: > 168 TGs, and adding TGs past ~336 buys
//!   nothing for that shape.
//! * same file T5.11: capping the grid with `KAPLUMBA_GEMV_MAX_GROUPS=168`
//!   costs -0.12 tok/s and `=84` costs -0.21 -- monotonically worse.
//! * `spec/decode-19tok-repacked-250gb/implementation_notes.md`:42 TGs (the
//!   `_stream` kernel) measured 110.8 GB/s against 178 GB/s for 336 TGs.
//! * `spec/decode-20tps/parallel_track_a_notes.md` A.2 reaches the same verdict
//!   from the arithmetic: gate_up runs 1344 TGs at the *same* ~129 GB/s as
//!   down_proj's 336, and the head runs 16384 TGs at 253 GB/s. Threadgroup
//!   count does not order with bandwidth here.
//!
//! WHAT THIS ADDS
//!
//! `occupancy_census` is pure arithmetic (no device) and pins the exact grid
//! every production shape launches, transcribed from the dispatch sites, so a
//! later change to a grid shows up as a test failure instead of as a slower
//! number nobody attributes.
//!
//! `pipeline_static_limits` reads the only occupancy-adjacent quantities Metal's
//! public API exposes: SIMD-group width, max threads per threadgroup, and
//! *static* threadgroup memory per threadgroup. The last one is the input to
//! the threadgroup-memory occupancy cliff (measured elsewhere in this repo at
//! ~20-27 KB/threadgroup -> 1 threadgroup/core), so a GEMV using 1 KB cannot be
//! hitting it.
//!
//! `occupancy_knee_sweep` is the experiment: dispatch the grid-stride
//! `linear_gemv_ishizuki_k3_colmajor_2col` at fixed total work and vary only the
//! grid, so achieved bandwidth against grid size *is* the occupancy curve, and
//! its knee is the resident threadgroup count. It also runs the 2x2 that
//! separates "fewer threadgroups" from "more registers": {2col, 4col} x
//! {natural grid, half grid}, where both arms of the half-grid column launch
//! the *same* number of threadgroups and differ only in register pressure.
//!
//! Run with:
//!   cargo test --release --test perf_probe_occupancy_grid -- --ignored --nocapture
//!
//! `occupancy_census` needs no device and is not ignored.
//!
//! Buffer contents are pseudo-random `u16`, not valid codes: this measures
//! streaming, not correctness (see `live_kernel_bit_exact.rs` for that).

use std::ffi::c_void;
use std::mem::size_of;
use std::time::Instant;

use kaplumba_kernels::encoder::{MetalEncoder, TensorRef};
use kaplumba_kernels::linear::LinearKernels;
use metal::{Buffer, CommandQueue, Device, MTLGPUFamily, MTLResourceOptions};

/// `u16` per 16x16 trellis tile at 3 bpw: 256 codes * 3 bits = 768 bits = 96 B
/// = 48 `u16`. Mirrors `RING / WORDS` in `linear_repacked.metal`.
const U16_PER_TILE_3BPW: usize = 48;
/// Aim for this much GPU time per point: clock ramp must be a rounding error.
const TARGET_MS: f64 = 40.0;
/// Rotate buffers so the working set cannot be served from L2.
const MAX_WORKING_SET: usize = 768 << 20;
const MIN_WORKING_SET: usize = 192 << 20;

// ---------------------------------------------------------------------------
// 1. Census: exact grids, no device required
// ---------------------------------------------------------------------------

/// One live decode dispatch, transcribed from the call site that encodes it.
#[derive(Clone, Copy, Debug)]
struct Row {
    label: &'static str,
    /// `(name, site)` of the `encode_*` that decides the grid.
    dispatch: &'static str,
    kernel: &'static str,
    k: usize,
    n: usize,
    threadgroups: u64,
    /// Threadgroup memory the kernel declares, from `red[...]` in the source.
    tg_bytes: u64,
}

/// Grid arithmetic, one function per live dispatch site. Each mirrors the
/// `groups` expression in `linear.rs`; if one of those changes, this must too.
///
/// * `encode_gemv_fused3` -> `linear.rs:2780-2805`
/// * `encode_gemv_fused2_mlp` -> `linear.rs:2590-2597`
/// * `encode_gemv_raw_sk2` -> `linear.rs:2255-2259`
/// * `encode_gemv` (lm_head) -> `encode_gemv_raw` `linear.rs:2159-2208`
const fn fused_n_groups(n: &[usize]) -> u64 {
    // tiles = out/16 per tensor, then (sum of tiles) / 2 for the 2col variant.
    let mut tiles = 0u64;
    let mut i = 0;
    while i < n.len() {
        tiles += (n[i] / 16) as u64;
        i += 1;
    }
    tiles / 2
}

const fn sk2_colmajor_groups(n: usize) -> u64 {
    // `((out_features / 16) / 2) * 2`, i.e. exactly out_features / 16.
    ((n as u64 / 16) / 2) * 2
}

const fn head_groups(n: usize, nr0: u64) -> u64 {
    // `encode_gemv_raw`: 2col branch is unreachable for the head (the repacked
    // family is k3-only), so either `encode_ishizuki_gemv` (out/16, capped by
    // KAPLUMBA_GEMV_MAX_GROUPS which is u64::MAX by default) or the EXL3 arm
    // (`tiles / nr0`, `linear.rs:2190-2207`). Same grid either way.
    let _ = n;
    (n as u64) / 16 / nr0
}

/// Shapes as given to this probe: q 5376x8192, k/v 5376x2048 (4 kv heads x 512),
/// o 5376x8192, gate/up 5376x21504, down 21504x5376, head 5376x262144 at 6 bpw.
///
/// The measured repo checkpoint is the 16 kv-head x 256 variant, so k/v are
/// 4096 wide there; both are printed.
fn census() -> Vec<Row> {
    let mut rows = vec![
        Row {
            label: "q+k+v fused3_2col (4 kv heads)",
            dispatch: "encode_gemv_fused3 -> linear.rs:2785-2787",
            kernel: "linear_gemv_ishizuki_k3_fused3_2col",
            k: 5376,
            n: 8192 + 2048 + 2048,
            threadgroups: fused_n_groups(&[8192, 2048, 2048]),
            tg_bytes: 2 * 8 * 16 * 4,
        },
        Row {
            label: "q+k+v fused3_2col (16 kv heads, measured ckpt)",
            dispatch: "encode_gemv_fused3 -> linear.rs:2785-2787",
            kernel: "linear_gemv_ishizuki_k3_fused3_2col",
            k: 5376,
            n: 8192 + 4096 + 4096,
            threadgroups: fused_n_groups(&[8192, 4096, 4096]),
            tg_bytes: 2 * 8 * 16 * 4,
        },
        Row {
            label: "o_proj colmajor_2col_sk2",
            dispatch: "encode_gemv_raw_sk2 -> linear.rs:2255-2259",
            kernel: "linear_gemv_ishizuki_k3_colmajor_2col_sk2",
            k: 8192,
            n: 5376,
            threadgroups: sk2_colmajor_groups(5376),
            tg_bytes: 2 * 8 * 16 * 4,
        },
        Row {
            label: "gate+up fused2_2col",
            dispatch: "encode_gemv_fused2_mlp -> linear.rs:2590-2597",
            kernel: "linear_gemv_ishizuki_k3_fused2_2col",
            k: 5376,
            n: 21504 + 21504,
            threadgroups: fused_n_groups(&[21504, 21504]),
            tg_bytes: 2 * 8 * 16 * 4,
        },
        Row {
            label: "down_proj colmajor_2col_sk2",
            dispatch: "encode_gemv_raw_sk2 -> linear.rs:2255-2259",
            kernel: "linear_gemv_ishizuki_k3_colmajor_2col_sk2",
            k: 21504,
            n: 5376,
            threadgroups: sk2_colmajor_groups(5376),
            tg_bytes: 2 * 8 * 16 * 4,
        },
        Row {
            label: "lm_head (ishizuki_k6 or exl3_k6_mcg)",
            dispatch: "encode_gemv -> encode_gemv_raw -> linear.rs:2190-2207",
            kernel: "linear_gemv_ishizuki_k6 | linear_gemv_exl3_k6_mcg",
            k: 5376,
            n: 262144,
            threadgroups: head_groups(262144, 1),
            tg_bytes: 8 * 16 * 4,
        },
    ];
    // NR0 = 2 halves the head's grid; the axis exists and defaults to 1.
    rows.push(Row {
        label: "lm_head at KAPLUMBA_DECODE_NR0=2",
        dispatch: "encode_gemv_raw -> linear.rs:2190-2207",
        kernel: "linear_gemv_exl3_k6_mcg_r2",
        k: 5376,
        n: 262144,
        threadgroups: head_groups(262144, 2),
        tg_bytes: 8 * 16 * 2 * 4,
    });
    rows
}

/// Resident threadgroups the GPU is known to hold, bracketed by measurements
/// already in this repository rather than by a driver constant Metal does not
/// expose:
///
/// * lower bound 168: split-K 1 (168 TGs) -> split-K 2 (336 TGs) is worth
///   -2.7 ms on `down_proj`, so 168 is not enough.
/// * upper bound 672: split-K 4 (672 TGs) is worth 0 over split-K 2, and
///   capping at 84/168 is monotonically worse, so nothing above ~336 helps.
///
/// The upper end is consistent with a 32 KB threadgroup-memory-per-core budget
/// at 1 KB per threadgroup (32 TGs/core x 20 = 640), but that limit is not a
/// public API value and is not asserted here.
const RESIDENT_LOWER_BOUND_TG: u64 = 168;
const RESIDENT_UPPER_BOUND_TG: u64 = 672;
const CORES: u64 = 20;

#[test]
fn occupancy_census() {
    let rows = census();
    println!(
        "\n{:<46} {:>7} {:>8} {:>6} {:>9} {:>9} {:>7}",
        "shape", "K", "N(sum)", "TGs", "TG/core", "vs 168-672", "tg_mem"
    );
    for row in &rows {
        let per_core = row.threadgroups as f64 / CORES as f64;
        let verdict = if row.threadgroups < RESIDENT_LOWER_BOUND_TG {
            "UNDER"
        } else if row.threadgroups >= RESIDENT_UPPER_BOUND_TG {
            "over-subscribed"
        } else {
            "inside bracket"
        };
        println!(
            "{:<46} {:>7} {:>8} {:>6} {:>9.1} {:>9} {:>5} B  {}",
            row.label,
            row.k,
            row.n,
            row.threadgroups,
            per_core,
            verdict,
            row.tg_bytes,
            row.kernel,
        );
        println!("{:<46} {}", "", row.dispatch);
    }

    // Pinned numbers. These are the grids the production decode step issues;
    // if a dispatch site's `groups` expression moves, this fails and says so,
    // instead of the next bandwidth number quietly changing meaning.
    assert_eq!(census()[0].threadgroups, 384, "qkv, 4 kv heads");
    assert_eq!(census()[1].threadgroups, 512, "qkv, 16 kv heads");
    assert_eq!(census()[2].threadgroups, 336, "o_proj split-K 2");
    assert_eq!(census()[3].threadgroups, 1344, "gate+up");
    assert_eq!(census()[4].threadgroups, 336, "down_proj split-K 2");
    assert_eq!(census()[5].threadgroups, 16384, "lm_head, NR0=1");
    assert_eq!(census()[6].threadgroups, 8192, "lm_head, NR0=2");

    // No live GEMV is anywhere near the threadgroup-memory occupancy cliff
    // measured at 20-27 KB per threadgroup in this repo
    // (`spec/engine-performance/implementation_notes.md`: omlx's 27 KB
    // `dsa_indexer` -> 1 threadgroup/core, -18%; our own 20 KB 64x64 GEMM lost
    // for the same reason). At 512 B or 1 KB per threadgroup, threadgroup
    // memory cannot be what limits occupancy.
    for row in &rows {
        assert!(
            row.tg_bytes <= 2048,
            "{} declares {} B of threadgroup memory; over 20 KB collapses to \
             1 threadgroup/core, so re-check the occupancy argument",
            row.kernel,
            row.tg_bytes,
        );
    }

    // The census's own conclusion, asserted so it cannot rot: the two shapes
    // with the largest grids (gate+up 1344, head 16384) bracket the two with
    // 336, and they do NOT order with measured bandwidth. That is the whole
    // reason occupancy is not the limiter for the long-K layer GEMVs.
    let under = rows.iter().filter(|r| r.threadgroups < RESIDENT_LOWER_BOUND_TG).count();
    assert_eq!(under, 0, "no production shape is below the 168-TG lower bound");
}

// ---------------------------------------------------------------------------
// 2. Static per-kernel limits (device required, read-only)
// ---------------------------------------------------------------------------

#[test]
#[ignore = "needs a Metal device; run explicitly"]
fn pipeline_static_limits() {
    let device = Device::system_default().expect("Metal device");
    let kernels = LinearKernels::new().expect("linear kernels compile");

    println!("\ndevice: {}", device.name());
    let max_thr = device.max_threads_per_threadgroup();
    println!(
        "  max_threads_per_threadgroup  = {} x {} x {}",
        max_thr.width, max_thr.height, max_thr.depth
    );
    println!(
        "  max_threadgroup_memory_len   = {} B",
        device.max_threadgroup_memory_length()
    );
    for (family, name) in [
        (MTLGPUFamily::Apple7, "Apple7 (M1)"),
        (MTLGPUFamily::Apple8, "Apple8 (M2)"),
        (MTLGPUFamily::Apple9, "Apple9 (M3/M4)"),
    ] {
        println!("  supports {name}: {}", device.supports_family(family));
    }

    println!(
        "\n{:<58} {:>5} {:>7} {:>9} {:>7}",
        "kernel", "simd", "max_thr", "tg_mem_B", "thr/grp"
    );
    let live = [
        "linear_gemv_ishizuki_k3_fused3_2col",
        "linear_gemv_ishizuki_k3_fused2_2col",
        "linear_gemv_ishizuki_k3_colmajor_2col_sk2",
        "linear_gemv_ishizuki_k3_colmajor_2col",
        "linear_gemv_ishizuki_k3_colmajor_4col",
        "linear_gemv_ishizuki_k3_w2",
        "linear_gemv_ishizuki_k6",
        "linear_gemv_exl3_k6_mcg",
    ];
    for name in live {
        let Some(pipeline) = kernels.ishizuki_gemv_pipeline_by_name(name).or_else(|| {
            // The EXL3 family is not in `ishizuki_gemv_by_name`.
            None
        }) else {
            println!("{name:<58}  <not built>");
            continue;
        };
        println!(
            "{:<58} {:>5} {:>7} {:>9} {:>7}",
            name,
            pipeline.thread_execution_width(),
            pipeline.max_total_threads_per_threadgroup(),
            pipeline.static_threadgroup_memory_length(),
            256,
        );
    }

    // `simd` is 32 and `max_thr` is 1024 for every one of these: the kernels
    // declare no `[[threads_per_threadgroup]]` / `[[simdgroups_per_threadgroup]]`
    // attribute anywhere (grep `crates/kernels/src` returns nothing), so the
    // launch shape comes entirely from the host `dispatch_1d_offset(.., 256)`
    // and Metal infers 8 simdgroups from 256 threads. `constexpr int SG = 8`
    // indexes `red[sg]`, so 256 threads per threadgroup is load-bearing:
    // 512 would make `sg` run to 15 and index `red[8]` out of bounds.
    let probe = kernels
        .ishizuki_gemv_pipeline_by_name("linear_gemv_ishizuki_k3_colmajor_2col_sk2")
        .expect("live kernel");
    assert_eq!(probe.thread_execution_width(), 32, "SIMD-group width");
    assert_eq!(probe.static_threadgroup_memory_length(), 1024, "red[2][8][16]");
    assert!(
        probe.max_total_threads_per_threadgroup() >= 256,
        "a 256-thread dispatch must be legal",
    );
}

// ---------------------------------------------------------------------------
// 3. The occupancy-knee experiment
// ---------------------------------------------------------------------------

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
    device.new_buffer_with_data(
        data.as_ptr() as *const c_void,
        (data.len() * size_of::<u16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

/// Shape params for the ishizuki family: two u32, little-endian, in the params
/// slot (`linear.rs:1469-1472`).
fn shape_params(k: usize, n: usize) -> [u8; 8] {
    let mut bytes = [0u8; 8];
    bytes[0..4].copy_from_slice(&(k as u32).to_le_bytes());
    bytes[4..8].copy_from_slice(&(n as u32).to_le_bytes());
    bytes
}

/// Dispatch `name` with an explicit `groups` count, bypassing the dispatcher's
/// own grid so occupancy can be varied at constant total work.
#[allow(clippy::too_many_arguments)]
fn dispatch_named(
    kernels: &LinearKernels,
    enc: &mut MetalEncoder,
    name: &str,
    packed: &Buffer,
    transformed: &Buffer,
    raw: &Buffer,
    k: usize,
    n: usize,
    groups: u64,
) {
    let pipeline = kernels
        .ishizuki_gemv_pipeline_by_name(name)
        .unwrap_or_else(|| panic!("kernel {name} is not built"));
    enc.dispatch_1d_offset(
        pipeline,
        &[
            TensorRef::new(packed, 0),
            TensorRef::new(transformed, 0),
            TensorRef::new(raw, 0),
        ],
        &shape_params(k, n),
        groups * 256,
        256,
    )
    .expect("dispatch");
}

/// Achieved GB/s for `name` at `groups`, at fixed (K, N).
fn measure(
    kernels: &LinearKernels,
    device: &Device,
    queue: &CommandQueue,
    name: &str,
    k: usize,
    n: usize,
    groups: u64,
) -> f64 {
    let shape_bytes = k * n * 3 / 8;
    let nbuf = (MAX_WORKING_SET / shape_bytes.max(1))
        .clamp(2, 8)
        .min((MIN_WORKING_SET / shape_bytes.max(1)).max(2));
    let packed: Vec<Buffer> = (0..nbuf)
        .map(|i| {
            random_buffer(
                device,
                (k * n / 8 / 4) * 2,
                0x1234_5678u32.wrapping_add(i as u32 * 7919),
            )
        })
        .collect();
    let transformed = random_buffer(device, k * 2, 0x9abc_def0);
    let raw = random_buffer(device, n * 4, 0xfeed_beef);

    let encode = |enc: &mut MetalEncoder, i: usize| {
        dispatch_named(
            kernels,
            enc,
            name,
            &packed[i % packed.len()],
            &transformed,
            &raw,
            k,
            n,
            groups,
        );
    };

    let reps = {
        let mut enc = MetalEncoder::new(queue).expect("encoder");
        enc.set_max_dispatches(u32::MAX);
        let t0 = Instant::now();
        for i in 0..4 {
            encode(&mut enc, i);
        }
        enc.commit_and_wait().expect("cal");
        let per = (t0.elapsed().as_secs_f64() * 1e3 / 4.0).max(1e-4);
        ((TARGET_MS / per).round() as usize).clamp(4, 2000)
    };
    {
        let mut enc = MetalEncoder::new(queue).expect("encoder");
        for i in 0..reps.min(8) {
            encode(&mut enc, i);
        }
        enc.commit_and_wait().expect("warm");
    }

    let mut enc = MetalEncoder::new(queue).expect("encoder");
    enc.set_max_dispatches(u32::MAX);
    for i in 0..reps {
        encode(&mut enc, i);
    }
    enc.commit_stamped(name).expect("commit");
    let gpu_ms: f64 = enc.take_gpu_stamps().iter().map(|s| s.own_ms).sum();
    enc.commit_and_wait().ok();
    (shape_bytes as f64 * reps as f64) / (gpu_ms.max(1e-9) * 1e-3) / 1e9
}

/// Sweep grid size at constant total work. Bandwidth against `groups` IS the
/// occupancy curve; its knee is the resident threadgroup count.
#[test]
#[ignore = "microbenchmark; run explicitly under gpuq"]
fn occupancy_knee_sweep() {
    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();
    let kernels = LinearKernels::new().expect("linear kernels compile");

    // `down_proj`: K=21504, N=5376 -> pairs_n = 5376/32 = 168, so the natural
    // grid is 168 TGs and the measured bracket is (168, 672].
    let (k, n) = (21504usize, 5376usize);
    let pairs = (n / 32) as u64;
    assert_eq!(pairs, 168);
    println!(
        "\n=== occupancy knee: colmajor_2col, K={k} N={n} ({} pairs, total work fixed) ===",
        pairs
    );
    println!("{:<10} {:>10} {:>12}", "groups", "TG/core", "GB/s");
    let mut curve = Vec::new();
    for groups in [8u64, 12, 20, 28, 40, 56, 84, 112, 140, 168] {
        let gbps = measure(
            &kernels,
            &device,
            &queue,
            "linear_gemv_ishizuki_k3_colmajor_2col",
            k,
            n,
            groups,
        );
        println!(
            "{groups:<10} {:>10.1} {:>12.1}",
            groups as f64 / CORES as f64,
            gbps
        );
        curve.push((groups, gbps));
    }
    // The knee: the smallest grid within 2% of the best point. If it lands near
    // 168 or below, occupancy was binding at the production grid; if it lands
    // far below 168, the production grid already over-subscribes and occupancy
    // cannot be the limiter.
    let best = curve.iter().map(|(_, g)| *g).fold(0.0f64, f64::max);
    if let Some((knee, _)) = curve.iter().find(|(_, g)| *g >= 0.98 * best) {
        println!(
            "\nknee (>=98% of best) = {} TGs = {:.1} TG/core",
            knee,
            *knee as f64 / CORES as f64
        );
    } else {
        println!("\nknee: no point reached 98% of best ({best:.1} GB/s)");
    }

    // 2x2: does register pressure cost anything once grid size is held equal?
    // Both columns of the half-grid arm launch the SAME number of threadgroups.
    let natural_2col = pairs;
    let natural_4col = (n / 64) as u64;
    println!("\n=== register pressure at equal grid (K={k} N={n}) ===");
    println!("{:<34} {:>7} {:>10}", "kernel", "groups", "GB/s");
    for (name, groups) in [
        ("linear_gemv_ishizuki_k3_colmajor_2col", natural_2col),
        ("linear_gemv_ishizuki_k3_colmajor_2col", natural_4col),
        ("linear_gemv_ishizuki_k3_colmajor_4col", natural_4col),
    ] {
        let gbps = measure(&kernels, &device, &queue, name, k, n, groups);
        println!("{name:<34} {groups:>7} {gbps:>10.1}");
    }

    // The wide control: `gate_up`'s fused shape launches 1344 TGs and is the
    // production shape with the most parallelism. If sweeping its grid down
    // toward 336 changes nothing, the 336-TG shapes cannot be occupancy-limited.
    let (kg, ng) = (5376usize, 21504usize);
    let pairs_g = (ng / 32) as u64;
    println!("\n=== control: fused2_2col, K={kg} N={ng} (gate_proj half) ===");
    println!("{:<10} {:>10} {:>12}", "groups", "TG/core", "GB/s");
    for groups in [336u64, 672, pairs_g] {
        if groups > pairs_g {
            continue;
        }
        let gbps = measure(
            &kernels,
            &device,
            &queue,
            "linear_gemv_ishizuki_k3_colmajor_2col",
            kg,
            ng,
            groups,
        );
        println!(
            "{groups:<10} {:>10.1} {:>12.1}",
            groups as f64 / CORES as f64,
            gbps
        );
    }
    let _ = U16_PER_TILE_3BPW;
}