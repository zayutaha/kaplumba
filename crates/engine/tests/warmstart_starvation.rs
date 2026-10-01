//! T2a gate: can a full-grid spinning kernel coexist with the dispatch that
//! releases it?
//!
//! The warm-start design puts a spin-wait on the critical path of *every*
//! threadgroup of the `o_proj` GEMV -- 336 threadgroups of 256 threads on this
//! checkpoint -- and dispatches the kernel that sets the flag AFTER them. If the
//! GPU dedicates all of its threadgroup slots to the spinners, the producer is
//! never resident, the flag never arrives, and the command buffer dies. That is
//! not a slow run; it is a GPU-level timeout, which is exactly what the engine
//! reported as `Metal command buffer failed during warmup decode`.
//!
//! T1 (`concurrent_tg_feasibility.rs`) does NOT answer this question. It
//! dispatched one threadgroup of 256 spinners and one of 32, so it had 96% of the
//! machine free and would have passed under either scheduling policy. Its result
//! -- `spin_count=0`, flag already set -- is equally consistent with the
//! producer having run to completion first, i.e. with full serialisation. The
//! test as written cannot distinguish the two, which is why this one exists.
//!
//! The dispatch order is the production order: spin-waiters first, producer
//! after. `SPIN_THREADGROUPS` and `SPIN_THREADS` are the production values for
//! `o_proj` (K=8192, N=5376, split-K 2 -> `(5376/16/2)*2 = 336` groups of 256).
//!
//! # Reading the numbers
//!
//! `SPIN_LIMIT` here is 2e6 iterations, not the kernel's 2e8. That is a
//! deliberate difference, and it is the one thing that makes this test usable: a
//! starved producer is detected in milliseconds instead of the ~900 s the first
//! version of this test took to time out. The engine cannot use the same trick,
//! because its spin is on the critical path and the bound has to cover a real
//! attention pass -- which is exactly why the engine's version of this failure
//! surfaced as a command-buffer error rather than as a number.
//!
//! spec/decode-20tps: T2a

use std::time::Instant;

use metal::{
    CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};

const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Spin bound. Deliberately MUCH smaller than the kernel's
// WARMSTART_SPIN_LIMIT (2e8): this test's job is to reach a verdict in
// milliseconds, and a starvation that would take the production bound
// hundreds of millions of iterations to reveal takes a few million here. The
// engine cannot use this trick -- there, the spin sits on the critical path and
// the bound has to cover a real attention pass -- which is precisely why the
// engine's failure showed up as a 900 s test timeout rather than a number.
static constant uint SPIN_LIMIT = 2000000u;

kernel void spin_on_flag(
    device const atomic_uint *flag     [[buffer(0)]],
    device atomic_uint       *timeout  [[buffer(1)]],
    constant const uint      *params   [[buffer(2)]],
    uint lid  [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    // One store per threadgroup, by lane 0, so the flag is set once the whole
    // threadgroup has executed -- the same "dispatch finished" semantics the
    // engine's producer kernel has.
    if (lane == 0u) {
        uint spins = 0u;
        while (atomic_load_explicit(flag, memory_order_relaxed) != params[0]) {
            if (++spins > SPIN_LIMIT) {
                atomic_store_explicit(timeout, 1u, memory_order_relaxed);
                return;
            }
        }
    }
}

kernel void produce_flag(
    device atomic_uint  *flag   [[buffer(0)]],
    device atomic_uint  *ran    [[buffer(1)]],
    constant const uint *params [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid == 0u) {
        atomic_store_explicit(ran, 1u, memory_order_relaxed);
        atomic_store_explicit(flag, params[0], memory_order_relaxed);
    }
}
"#;

/// `o_proj`'s production grid: 336 threadgroups of 256 threads.
const SPIN_THREADGROUPS: u64 = 336;
const SPIN_THREADS: u64 = 256;

struct Ctx {
    device: metal::Device,
    queue: metal::CommandQueue,
    spin: ComputePipelineState,
    produce: ComputePipelineState,
    flag: metal::Buffer,
    timeout: metal::Buffer,
    ran: metal::Buffer,
    params: metal::Buffer,
}

fn context() -> Option<Ctx> {
    let device = Device::system_default()?;
    let library = device
        .new_library_with_source(SOURCE, &CompileOptions::new())
        .ok()?;
    let spin = library.get_function("spin_on_flag", None).ok()?;
    let produce = library.get_function("produce_flag", None).ok()?;
    let spin = device
        .new_compute_pipeline_state_with_function(&spin)
        .ok()?;
    let produce = device
        .new_compute_pipeline_state_with_function(&produce)
        .ok()?;
    let options = MTLResourceOptions::StorageModeShared;
    let flag = device.new_buffer(4, options);
    let timeout = device.new_buffer(4, options);
    let ran = device.new_buffer(4, options);
    let params = device.new_buffer(4, options);
    unsafe {
        *(flag.contents() as *mut u32) = 0;
        *(timeout.contents() as *mut u32) = 0;
        *(ran.contents() as *mut u32) = 0;
        *(params.contents() as *mut u32) = 1;
    }
    let queue = device.new_command_queue();
    Some(Ctx {
        device,
        queue,
        spin,
        produce,
        flag,
        timeout,
        ran,
        params,
    })
}

/// Dispatch `spinners` threadgroups of 256 threads that wait on the flag, then
/// the producer that sets it -- both in ONE command buffer, in that order.
/// Returns `(timed_out, elapsed)`.
fn run(spinners: u64, ctx: &Ctx) -> (bool, f64) {
    unsafe {
        *(ctx.flag.contents() as *mut u32) = 0;
        *(ctx.timeout.contents() as *mut u32) = 0;
        *(ctx.ran.contents() as *mut u32) = 0;
    }
    let started = Instant::now();
    let buffer = ctx.queue.new_command_buffer();
    let enc = buffer.new_compute_command_encoder();

    enc.set_compute_pipeline_state(&ctx.spin);
    enc.set_buffer(0, Some(&ctx.flag), 0);
    enc.set_buffer(1, Some(&ctx.timeout), 0);
    enc.set_buffer(2, Some(&ctx.params), 0);
    enc.dispatch_thread_groups(
        MTLSize::new(spinners, 1, 1),
        MTLSize::new(SPIN_THREADS, 1, 1),
    );

    enc.set_compute_pipeline_state(&ctx.produce);
    enc.set_buffer(0, Some(&ctx.flag), 0);
    enc.set_buffer(1, Some(&ctx.ran), 0);
    enc.set_buffer(2, Some(&ctx.params), 0);
    enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(32, 1, 1));

    enc.end_encoding();
    buffer.commit();
    buffer.wait_until_completed();

    let timed_out = unsafe { *(ctx.timeout.contents() as *const u32) != 0 };
    (timed_out, started.elapsed().as_secs_f64() * 1e3)
}

/// The gate. A full-grid spinner that starves its own producer would make the
/// warm-start design a GPU hang, and the engine reported exactly that. This
/// asserts the producer is still reachable at the production grid size.
#[test]
fn a_full_o_proj_grid_of_spinners_does_not_starve_its_producer() {
    let Some(ctx) = context() else {
        eprintln!("no Metal device; skipping");
        return;
    };

    let (timed_out, ms) = run(SPIN_THREADGROUPS, &ctx);
    let produced = unsafe { *(ctx.ran.contents() as *const u32) != 0 };

    eprintln!("=== T2a starvation gate ===");
    eprintln!("  spinners:         {SPIN_THREADGROUPS} groups x {SPIN_THREADS} threads");
    eprintln!("  producer ran:     {produced}");
    eprintln!("  spin timed out:   {timed_out}");
    eprintln!("  wall:             {ms:.1} ms");

    assert!(
        produced,
        "the producer never ran: {SPIN_THREADGROUPS} spinning threadgroups starve it, \
         so dispatching the o_proj warm-start GEMV ahead of the dispatch that \
         releases it deadlocks the command buffer"
    );
    assert!(
        !timed_out,
        "the spinner exhausted WARMSTART_SPIN_LIMIT at the production grid size"
    );
}

/// Sweep the grid size, so the boundary is a measurement rather than an
/// assumption. Printed, not asserted: the exact crossover is a property of this
/// GPU's occupancy, and asserting a specific number would make the test a
/// calibration of Metal rather than of this design.
#[test]
fn starvation_boundary_is_measured() {
    let Some(ctx) = context() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    eprintln!("=== T2a starvation sweep ===");
    eprintln!("  groups   timed_out   produced   wall_ms");
    for groups in [1_u64, 8, 32, 64, 128, 168, 336, 672] {
        let (timed_out, ms) = run(groups, &ctx);
        let produced = unsafe { *(ctx.ran.contents() as *const u32) != 0 };
        eprintln!("  {groups:>6}   {timed_out:>9}   {produced:>8}   {ms:>7.1}");
    }
}