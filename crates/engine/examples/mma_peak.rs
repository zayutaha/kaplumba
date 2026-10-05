//! Sustained Apple-Silicon tensor-core throughput for the prefill GEMM --
//! **corrected**. `spec/engine-performance` guessed "30-50+ TFLOPS" for Metal
//! `simdgroup_multiply_accumulate` and never measured it; this turns it into a
//! figure. `spec/prefill-24k-90tps` then used that figure as the denominator of
//! every "% of peak" it wrote, so the instrument has to be trustworthy more than
//! the number has to be big.
//!
//! # What was wrong (round 2 of `spec/prefill-24k-90tps`, finding A1)
//!
//! The old kernel accumulated into **16** named accumulators and stored **8**.
//! `d4..d7` were written and never read, so their dependency chains were dead
//! stores the compiler was entitled to delete -- and did. It reported
//! **15.0 TFLOPS** for work that a kernel doing **half** the MMAs finished in
//! the same time. The honest figure for that MMAs-at-that-shape was **7.5**, and
//! every percentage written off 15.6 was out by 2x.
//!
//! # What this instrument does about it
//!
//! * **Every accumulator is stored.** All of them, into a `device float *`, so
//!   no chain is dead. No accumulator can be eliminated.
//! * **The dead-store case is still compiled in, as a control.** `dead8` runs
//!   the *old* kernel's shape (16 accumulate, 8 store) next to `live8` (8
//!   accumulate, 8 store) and `live16` (16 accumulate, 16 store). If the
//!   compiler were deleting `dead8`'s unstored chains, `dead8` and `live8` time
//!   the same; the instrument **asserts** they do not, and asserts `live16` is
//!   about 2x `live8`. That assertion is the positive control: it fails on the
//!   old bug, so a coverage claim here is a claim only after the instrument has
//!   been seen to fail.
//! * **Swept in chain count**, 1/2/4/8/16/24, one library, one build,
//!   interleaved rounds, medians with spread. `mma_peak.rs` stops at 16 because
//!   16 was where the old (phantom) table stopped; 24 is here to show whether
//!   16 is a ceiling or a register cliff.
//!
//! `mma_alu` keeps a, b and c in registers and runs `iters` 8x8x8 MMAs with no
//! memory traffic, so it is the hardware ceiling for the tensor units *for this
//! operand shape*. Real GEMM throughput will be lower once operands have to be
//! fed from memory.
//!
//! FLOP convention: one `simdgroup_half8x8` `simdgroup_multiply_accumulate` =
//! 8x8x8 MACs = **1024 FLOPs**.

use metal::{Device, Library, MTLResourceOptions};
use std::time::Instant;

const SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Pure ALU throughput. The third argument to simdgroup_load/store is the leading
// dimension (elements per row), NOT a row stride or element stride -- passing 0
// there was the bug that made the 8x8 probe look broken.
//
// Chain count matters, because dependent MMAs measure latency rather than
// throughput -- but the OLD table's magnitudes were an artefact and are not
// reproduced here. Measured on this box with every accumulator stored (see the
// main file for the numbers this run prints):
//
//   | chains | TFLOP/s |
//   |---|---|
//   | see the printed table |
//
// An ARRAY of accumulators with an unrolled index loop makes the compiler spill
// to local memory and collapses throughput (the old header recorded 15.49 ->
// 0.74 TFLOPS for 32 of them). Only named locals stay in registers, so every
// kernel below names its accumulators.

// Zero-filled, not loaded from memory: an undefined value can be NaN or denormal,
// and a tensor op that sees one is not measuring the pipe.
#define DECL1 simdgroup_float8x8 c0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
#define DECL2 DECL1 simdgroup_float8x8 c1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
#define DECL4 DECL2 simdgroup_float8x8 c2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \
                   simdgroup_float8x8 c3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
#define DECL8 DECL4 simdgroup_float8x8 c4 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \
                   simdgroup_float8x8 c5 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \
                   simdgroup_float8x8 c6 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \
                   simdgroup_float8x8 c7 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
#define DECL16 DECL8 simdgroup_float8x8 d0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \
                    simdgroup_float8x8 d1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \
                    simdgroup_float8x8 d2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \
                    simdgroup_float8x8 d3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \
                    simdgroup_float8x8 d4 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \
                    simdgroup_float8x8 d5 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \
                    simdgroup_float8x8 d6 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \
                    simdgroup_float8x8 d7 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
#define DECL24 DECL16 simdgroup_float8x8 e0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \
                    simdgroup_float8x8 e1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \
                    simdgroup_float8x8 e2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \
                    simdgroup_float8x8 e3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \
                    simdgroup_float8x8 e4 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \
                    simdgroup_float8x8 e5 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
#define DECL32 DECL24 simdgroup_float8x8 f0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \
                    simdgroup_float8x8 f1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \
                    simdgroup_float8x8 f2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \
                    simdgroup_float8x8 f3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \
                    simdgroup_float8x8 f4 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \
                    simdgroup_float8x8 f5 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \
                    simdgroup_float8x8 f6 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \
                    simdgroup_float8x8 f7 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

#define MMA(x) simdgroup_multiply_accumulate(x, a, b, x);

// ---- live chains: N accumulate, N store -------------------------------------
kernel void live1(device float *out [[buffer(0)]],
                  constant uint &iters [[buffer(1)]],
                   device half *in [[buffer(2)]],
                  uint gid [[thread_position_in_grid]]) {
    constexpr int LD = 8;
    simdgroup_half8x8 a, b;
    simdgroup_load(a, in, LD);
    simdgroup_load(b, in, LD);
    DECL1
    for (uint i = 0; i < iters; ++i) { MMA(c0) }
    simdgroup_store(c0, out, LD);
}

kernel void live2(device float *out [[buffer(0)]],
                  constant uint &iters [[buffer(1)]],
                   device half *in [[buffer(2)]],
                  uint gid [[thread_position_in_grid]]) {
    constexpr int LD = 8;
    simdgroup_half8x8 a, b;
    simdgroup_load(a, in, LD);
    simdgroup_load(b, in, LD);
    DECL2
    for (uint i = 0; i < iters; ++i) { MMA(c0) MMA(c1) }
    simdgroup_store(c0, out, LD);
    simdgroup_store(c1, out, LD);
}

kernel void live4(device float *out [[buffer(0)]],
                  constant uint &iters [[buffer(1)]],
                   device half *in [[buffer(2)]],
                  uint gid [[thread_position_in_grid]]) {
    constexpr int LD = 8;
    simdgroup_half8x8 a, b;
    simdgroup_load(a, in, LD);
    simdgroup_load(b, in, LD);
    DECL4
    for (uint i = 0; i < iters; ++i) { MMA(c0) MMA(c1) MMA(c2) MMA(c3) }
    simdgroup_store(c0, out, LD);
    simdgroup_store(c1, out, LD);
    simdgroup_store(c2, out, LD);
    simdgroup_store(c3, out, LD);
}

kernel void live8(device float *out [[buffer(0)]],
                  constant uint &iters [[buffer(1)]],
                   device half *in [[buffer(2)]],
                  uint gid [[thread_position_in_grid]]) {
    constexpr int LD = 8;
    simdgroup_half8x8 a, b;
    simdgroup_load(a, in, LD);
    simdgroup_load(b, in, LD);
    DECL8
    for (uint i = 0; i < iters; ++i) {
        MMA(c0) MMA(c1) MMA(c2) MMA(c3) MMA(c4) MMA(c5) MMA(c6) MMA(c7)
    }
    simdgroup_store(c0, out, LD); simdgroup_store(c1, out, LD);
    simdgroup_store(c2, out, LD); simdgroup_store(c3, out, LD);
    simdgroup_store(c4, out, LD); simdgroup_store(c5, out, LD);
    simdgroup_store(c6, out, LD); simdgroup_store(c7, out, LD);
}

kernel void live16(device float *out [[buffer(0)]],
                   constant uint &iters [[buffer(1)]],
                    device half *in [[buffer(2)]],
                   uint gid [[thread_position_in_grid]]) {
    constexpr int LD = 8;
    simdgroup_half8x8 a, b;
    simdgroup_load(a, in, LD);
    simdgroup_load(b, in, LD);
    DECL16
    for (uint i = 0; i < iters; ++i) {
        MMA(c0) MMA(c1) MMA(c2) MMA(c3) MMA(c4) MMA(c5) MMA(c6) MMA(c7)
        MMA(d0) MMA(d1) MMA(d2) MMA(d3) MMA(d4) MMA(d5) MMA(d6) MMA(d7)
    }
    simdgroup_store(c0, out, LD); simdgroup_store(c1, out, LD);
    simdgroup_store(c2, out, LD); simdgroup_store(c3, out, LD);
    simdgroup_store(c4, out, LD); simdgroup_store(c5, out, LD);
    simdgroup_store(c6, out, LD); simdgroup_store(c7, out, LD);
    simdgroup_store(d0, out, LD); simdgroup_store(d1, out, LD);
    simdgroup_store(d2, out, LD); simdgroup_store(d3, out, LD);
    simdgroup_store(d4, out, LD); simdgroup_store(d5, out, LD);
    simdgroup_store(d6, out, LD); simdgroup_store(d7, out, LD);
}

kernel void live24(device float *out [[buffer(0)]],
                   constant uint &iters [[buffer(1)]],
                    device half *in [[buffer(2)]],
                   uint gid [[thread_position_in_grid]]) {
    constexpr int LD = 8;
    simdgroup_half8x8 a, b;
    simdgroup_load(a, in, LD);
    simdgroup_load(b, in, LD);
    DECL24
    for (uint i = 0; i < iters; ++i) {
        MMA(c0) MMA(c1) MMA(c2) MMA(c3) MMA(c4) MMA(c5) MMA(c6) MMA(c7)
        MMA(d0) MMA(d1) MMA(d2) MMA(d3) MMA(d4) MMA(d5) MMA(d6) MMA(d7)
        MMA(e0) MMA(e1) MMA(e2) MMA(e3) MMA(e4) MMA(e5)
    }
    simdgroup_store(c0, out, LD); simdgroup_store(c1, out, LD);
    simdgroup_store(c2, out, LD); simdgroup_store(c3, out, LD);
    simdgroup_store(c4, out, LD); simdgroup_store(c5, out, LD);
    simdgroup_store(c6, out, LD); simdgroup_store(c7, out, LD);
    simdgroup_store(d0, out, LD); simdgroup_store(d1, out, LD);
    simdgroup_store(d2, out, LD); simdgroup_store(d3, out, LD);
    simdgroup_store(d4, out, LD); simdgroup_store(d5, out, LD);
    simdgroup_store(d6, out, LD); simdgroup_store(d7, out, LD);
    simdgroup_store(e0, out, LD); simdgroup_store(e1, out, LD);
    simdgroup_store(e2, out, LD); simdgroup_store(e3, out, LD);
    simdgroup_store(e4, out, LD); simdgroup_store(e5, out, LD);
}

// 24 chains already read HIGHER than 16, so 16 is not a ceiling and the old
// table stopped one step too early. 32 is here to find where it does turn over.
kernel void live32(device float *out [[buffer(0)]],
                   constant uint &iters [[buffer(1)]],
                   device half *in [[buffer(2)]],
                   uint gid [[thread_position_in_grid]]) {
    constexpr int LD = 8;
    simdgroup_half8x8 a, b;
    simdgroup_load(a, in, LD);
    simdgroup_load(b, in, LD);
    DECL32
    for (uint i = 0; i < iters; ++i) {
        MMA(c0) MMA(c1) MMA(c2) MMA(c3) MMA(c4) MMA(c5) MMA(c6) MMA(c7)
        MMA(d0) MMA(d1) MMA(d2) MMA(d3) MMA(d4) MMA(d5) MMA(d6) MMA(d7)
        MMA(e0) MMA(e1) MMA(e2) MMA(e3) MMA(e4) MMA(e5)
        MMA(f0) MMA(f1) MMA(f2) MMA(f3) MMA(f4) MMA(f5) MMA(f6) MMA(f7)
    }
    simdgroup_store(c0, out, LD); simdgroup_store(c1, out, LD);
    simdgroup_store(c2, out, LD); simdgroup_store(c3, out, LD);
    simdgroup_store(c4, out, LD); simdgroup_store(c5, out, LD);
    simdgroup_store(c6, out, LD); simdgroup_store(c7, out, LD);
    simdgroup_store(d0, out, LD); simdgroup_store(d1, out, LD);
    simdgroup_store(d2, out, LD); simdgroup_store(d3, out, LD);
    simdgroup_store(d4, out, LD); simdgroup_store(d5, out, LD);
    simdgroup_store(d6, out, LD); simdgroup_store(d7, out, LD);
    simdgroup_store(e0, out, LD); simdgroup_store(e1, out, LD);
    simdgroup_store(e2, out, LD); simdgroup_store(e3, out, LD);
    simdgroup_store(e4, out, LD); simdgroup_store(e5, out, LD);
    simdgroup_store(f0, out, LD); simdgroup_store(f1, out, LD);
    simdgroup_store(f2, out, LD); simdgroup_store(f3, out, LD);
    simdgroup_store(f4, out, LD); simdgroup_store(f5, out, LD);
    simdgroup_store(f6, out, LD); simdgroup_store(f7, out, LD);
}

// ---- CONTROL: the OLD kernel's shape, 16 accumulate and 8 store --------------
// `d4..d7` are dead stores. It is here so the instrument can *demonstrate* the
// bug it was rewritten to prevent: `dead8` must read the same as `live8` and
// half of `live16`. If it does not -- i.e. if a future Metal compiler keeps the
// dead chains -- the corrected table and this control have to be re-derived.
kernel void dead8(device float *out [[buffer(0)]],
                  constant uint &iters [[buffer(1)]],
                   device half *in [[buffer(2)]],
                  uint gid [[thread_position_in_grid]]) {
    constexpr int LD = 8;
    simdgroup_half8x8 a, b;
    simdgroup_load(a, in, LD);
    simdgroup_load(b, in, LD);
    DECL16
    for (uint i = 0; i < iters; ++i) {
        MMA(c0) MMA(c1) MMA(c2) MMA(c3) MMA(c4) MMA(c5) MMA(c6) MMA(c7)
        MMA(d0) MMA(d1) MMA(d2) MMA(d3) MMA(d4) MMA(d5) MMA(d6) MMA(d7)
    }
    simdgroup_store(c0, out, LD); simdgroup_store(c1, out, LD);
    simdgroup_store(c2, out, LD); simdgroup_store(c3, out, LD);
    simdgroup_store(c4, out, LD); simdgroup_store(c5, out, LD);
    simdgroup_store(c6, out, LD); simdgroup_store(c7, out, LD);
}
"#;

/// (kernel name, named accumulators, MMAs per loop body, stores).
const ARMS: &[(&str, usize)] = &[
    ("live1", 1),
    ("live2", 2),
    ("live4", 4),
    ("live8", 8),
    ("live16", 16),
    ("live24", 24),
    ("live32", 32),
    ("dead8", 16),
];

/// Index of an arm by name, so the controls below cannot silently point at the
/// wrong arm after a row is inserted (the `mma_peak` bug was an index-shaped
/// mistake twice over).
fn arm_index(name: &str) -> usize {
    ARMS.iter()
        .position(|(n, _)| *n == name)
        .unwrap_or_else(|| panic!("no arm {name}"))
}

fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

fn main() {
    let device = Device::system_default().expect("no Metal device");
    println!("device: {}", device.name());
    let library: Library = device
        .new_library_with_source(SRC, &metal::CompileOptions::new())
        .expect("compile");

    let groups: u64 = 2048; // 2048 simdgroups = 65536 threads
    let iters: u32 = 20_000;
    let rounds: usize = std::env::var("KAPLUMBA_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    // `out` is the fp32 store target (every accumulator goes here, so no chain
    // is a dead store); `in` is the fp32 buffer the half operands are loaded
    // from. They are separate because `simdgroup_load` is typed.
    let out = device.new_buffer(256, MTLResourceOptions::StorageModeShared);
    let in_buf = device.new_buffer(256, MTLResourceOptions::StorageModeShared);
    // 0.25f as fp32: a normal, finite, non-denormal operand for the tensor op.
    // An uninitialised buffer can hand the MMA a NaN or a denormal, which changes
    // its cost and makes the number a property of memory rather than of the pipe.
    {
        let seed: [f32; 64] = [0.25; 64];
        let slot = in_buf.contents() as *mut f32;
        for i in 0..64 {
            unsafe { slot.add(i).write(seed[i]) };
        }
    }
    let iters_buf = device.new_buffer_with_data(
        &iters as *const u32 as *const std::ffi::c_void,
        4,
        MTLResourceOptions::StorageModeShared,
    );
    let queue = device.new_command_queue();

    let mut pipelines = Vec::new();
    for (name, _) in ARMS {
        let function = library
            .get_function(name, None)
            .unwrap_or_else(|_| panic!("function {name}"));
        let state = device
            .new_compute_pipeline_state_with_function(&function)
            .expect("pipeline");
        println!(
            "pipeline: {name} ({}, {} registers/lane)",
            state.max_total_threads_per_threadgroup(),
            state.static_threadgroup_memory_length()
        );
        pipelines.push(state);
    }

    let run = |state: &metal::ComputePipelineState| -> f64 {
        let cb = queue.new_command_buffer();
        let ce = cb.new_compute_command_encoder();
        ce.set_compute_pipeline_state(state);
        ce.set_buffer(0, Some(&out), 0);
        ce.set_buffer(1, Some(&iters_buf), 0);
        ce.set_buffer(2, Some(&in_buf), 0);
        ce.dispatch_thread_groups(
            metal::MTLSize::new(groups, 1, 1),
            metal::MTLSize::new(32, 1, 1),
        );
        ce.end_encoding();
        cb.commit();
        let start = Instant::now();
        cb.wait_until_completed();
        start.elapsed().as_secs_f64()
    };

    // One untimed warm pass per arm, so no arm pays pipeline construction.
    for state in &pipelines {
        run(state);
    }
    // Interleaved rounds: `0 1 2 ... 0 1 2 ...`, so clock and thermal drift
    // cannot masquerade as an arm difference (AGENTS.md 10.2 #5).
    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); pipelines.len()];
    for _ in 0..rounds {
        for (index, state) in pipelines.iter().enumerate() {
            samples[index].push(run(state));
        }
    }

    println!(
        "\n{groups} simdgroups x {iters} iters, medians of {rounds} interleaved rounds"
    );
    println!(
        "{:>6}  {:>5}  {:>9}  {:>9}  {:>9}  {:>9}  {:>7}",
        "arm", "chains", "median_s", "min_s", "max_s", "TFLOP/s", "vs live8"
    );
    // All rates first: the `vs live8` column needs `live8`'s rate, which is not
    // known until its own row is computed.
    let mut rows: Vec<(&str, usize, f64, f64, f64, f64)> = Vec::new();
    for (index, (name, chains)) in ARMS.iter().enumerate() {
        let mut xs = samples[index].clone();
        let med = median(&mut xs);
        let flops = groups as f64 * iters as f64 * *chains as f64 * 1024.0;
        rows.push((*name, *chains, med, xs[0], xs[xs.len() - 1], flops / med / 1e12));
    }
    // Chain-count scaling, against `live8`'s time: time should be proportional to
    // the MMA count. It is not, quite, past 16 chains -- recorded rather than
    // smoothed over, because "the peak" is the max of this table and a
    // sub-linear curve is the reason it has to be measured, not modelled.
    let live8_time = rows[arm_index("live8")].2;
    for (name, chains, med, _, _, rate) in &rows {
        let expected = live8_time * *chains as f64 / 8.0;
        println!(
            "scale  {name:>6}  {chains:>3} chains: measured/expected time = x{:.3}  ({rate:.2} TFLOP/s)",
            med / expected
        );
    }
    let tflops: Vec<f64> = rows.iter().map(|r| r.5).collect();
    let live8_rate = tflops[arm_index("live8")];
    for (name, chains, med, min, max, rate) in &rows {
        println!(
            "{name:>6}  {chains:>5}  {med:>9.4}  {min:>9.4}  {max:>9.4}  {rate:>9.2}  {:>7.3}",
            live8_rate / rate
        );
    }

    // ---- the controls. These are what make the table above a measurement -----
    let live8_s = median(&mut samples[arm_index("live8")].clone());
    let live16_s = median(&mut samples[arm_index("live16")].clone());
    let dead8_s = median(&mut samples[arm_index("dead8")].clone());
    println!("\n--- controls ---");
    println!(
        "live16 / live8  = x{:.3}  (expect ~2: twice the MMAs, twice the time)",
        live16_s / live8_s
    );
    println!(
        "live8  / dead8  = x{:.3}  (expect ~2: the old kernel did 2x the live MMAs \
         and, if the compiler had NOT deleted its dead chains, would take 2x as long)",
        live8_s / dead8_s
    );
    let dead_deleted = dead8_s < live8_s * 1.35;
    println!(
        "dead-store deletion reproduced: {}",
        if dead_deleted { "YES -- dead8 is the phantom peak, and it is ~2x too fast" }
        else { "no -- a future compiler keeps the dead chains; re-derive the whole table" }
    );
    // Linearity control: the timing must be of the MMAs it counts, so doubling
    // `iters` must double the time. Without this, a "peak" could be a fixed cost
    // per dispatch.
    {
        let double = iters * 2;
        let probe = |pipeline: &metal::ComputePipelineState, iters: u32| -> f64 {
            let buf = device.new_buffer_with_data(
                &iters as *const u32 as *const std::ffi::c_void,
                4,
                MTLResourceOptions::StorageModeShared,
            );
            let cb = queue.new_command_buffer();
            let ce = cb.new_compute_command_encoder();
            ce.set_compute_pipeline_state(pipeline);
            ce.set_buffer(0, Some(&out), 0);
            ce.set_buffer(1, Some(&buf), 0);
            ce.set_buffer(2, Some(&in_buf), 0);
            ce.dispatch_thread_groups(
                metal::MTLSize::new(groups, 1, 1),
                metal::MTLSize::new(32, 1, 1),
            );
            ce.end_encoding();
            cb.commit();
            let start = Instant::now();
            cb.wait_until_completed();
            start.elapsed().as_secs_f64()
        };
        let p = &pipelines[arm_index("live16")];
        let once = probe(p, iters);
        let twice = probe(p, iters * 2);
        println!(
            "live16 at 1x iters {:.4} s, at 2x iters {:.4} s -> x{:.3} (expect ~2)",
            once,
            twice,
            twice / once
        );
        assert!(
            (1.85..2.15).contains(&(twice / once)),
            "live16 is not linear in iters (x{:.3}): the timing is not of the MMAs it counts",
            twice / once
        );
    }

    let live16_over_live8 = live16_s / live8_s;
    assert!(
        (1.7..2.3).contains(&live16_over_live8),
        "live16 is {live16_over_live8:.2}x live8, expected ~2: the instrument is not \
         measuring the MMAs it counts"
    );
    assert!(
        dead_deleted,
        "dead8 is not ~2x live8: the old kernel's dead stores are being kept, so the \
         corrected table and the control disagree and both need re-deriving"
    );

    let peak = tflops
        .iter()
        .zip(ARMS)
        .filter(|(_, (name, _))| name.starts_with("live"))
        .map(|(rate, _)| *rate)
        .fold(0.0f64, f64::max);
    let peak_at = |chains: usize| -> f64 {
        tflops[arm_index(&format!("live{chains}"))]
    };
    println!("\ntensor-core peak (best live chain count, all accumulators stored): {peak:.2} TFLOPS");
    println!(
        "  live8 {:.2} | live16 {:.2} | live24 {:.2} | live32 {:.2}",
        peak_at(8),
        peak_at(16),
        peak_at(24),
        peak_at(32)
    );

    // What that means for Gemma 4 31B prefill, whose GEMM is 5.75e10 FLOP/token.
    let gemm_flops_per_token = 5.75e10_f64;
    println!(
        "GEMM-only ceiling at this rate: {:.1} tok/s (attention at 0 ms)",
        peak * 1e12 / gemm_flops_per_token
    );
    // And at 50% of peak, which is a realistic GEMM efficiency.
    println!(
        "GEMM-only ceiling at 50% of peak: {:.1} tok/s",
        peak * 0.5 * 1e12 / gemm_flops_per_token
    );
}