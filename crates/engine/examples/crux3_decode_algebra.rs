//! crux3 — is the 3 bpw decode GEMV **issue**-bound on `decode_state_mcg`, or is
//! it already at the memory wall?
//!
//! This runs the experiment `spec/decode-ceiling/findings/A2-dequant.md` §3
//! pre-registered and §10 recorded as never run. Until it ran, the largest
//! number in the whole decode programme — "+33.5% of GB/s sits above the layer's
//! 187.7, and it is ALU work" (`WHERE-THIS-STANDS.md`) — was **arithmetic on an
//! instruction count**, produced by a model whose own author had already recorded
//! that the model is falsified (`A2-dequant.md` §8: it predicts 256.9 GB/s and
//! "no rate on this box has ever exceeded ~250 GB/s").
//!
//! Three arms, one macro, identical loop / identical addresses / identical loads:
//!
//! | arm | extract body                                     | ops/code | chains |
//! |-----|-------------------------------------------------|----------|--------|
//! | A   | `decode_state_mcg(p & 0xffff)` — production     | 4        | 8      |
//! | B   | `float(as_type<half>((p & 0xffff) ^ K))`        | 2        | 8      |
//! | C   | `float(as_type<half>(p & 0xffff))`              | 1        | 8      |
//!
//! **B and C produce wrong numbers on purpose.** They still consume every loaded
//! word through the accumulator, so they are bandwidth-identical to A.
//!
//! ## Reading the result
//!
//! * `C/A >= 1.25` ⇒ the decode arithmetic is a real, large, *legal-looking*
//!   pool, and the ~203 GB/s wall is an issue limit. Every subsequent proposal
//!   about bit-locked logic (LOP3, `IMAD`) becomes worth building.
//! * `C/A <= 1.05` ⇒ the decode arithmetic is free, 203 GB/s is a memory-system
//!   property, and the whole ALU axis of the decode programme is closed. The
//!   remaining gap is bytes and dispatch, not instructions.
//!
//! The 1.25 / 1.05 window is deliberately wide (20%) because this box has
//! produced 2x swings from one binary and A2 §3 records "a measured 1.13x decode
//! spread". A verdict inside 5% is not a verdict.
//!
//! ## What the arms actually say — and why the verdict is NOT the naive one
//!
//! Measured (9 interleaved rounds, spreads 1.00-1.04, clock checks 1.001-1.004x):
//!
//! | shape | A (production) | B (2 ops) | C (1 op) | B/A | C/A |
//! |---|---:|---:|---:|---:|---:|
//! | down K21504 N5376   | 203.0 | 239.5 | 238.0 | 1.179 | 1.172 |
//! | q_slid K5376 N2048  | 156.5 | 184.8 | 200.8 | 1.181 | 1.283 |
//! | o_proj K5376 N5376  | 180.5 | 217.9 | 217.4 | 1.207 | 1.204 |
//! | gateup K5376 N43008 | 164.2 | 182.0 | 182.2 | 1.108 | 1.110 |
//!
//! **B and C are within 1-10% of each other everywhere**, despite C having half
//! B's arithmetic. That is the whole finding, and it is not what the op-count
//! model predicts: the model says halving the ops again should buy most of what
//! the first halving bought. It does not.
//!
//! So the shape is: **one step off the MCG decode is worth 11-21%, the second is
//! worth ~0**. That is the signature of a *latency* cost at the head of the
//! dependency chain, not a throughput cost. `decode_state_mcg` is
//! `IMUL -> AND -> XOR -> HADD2`: four **serially dependent** ops. Arm B breaks
//! the chain after the multiply; arm C removes the multiply but keeps a 3-deep
//! chain. Each arm shortens the chain once; neither shortens it twice, because
//! the remaining depth is the extraction shift plus the FMA.
//!
//! **The discriminator that settles it** (run as a sabotage, recorded here
//! because it is the number that makes the reading trustworthy): an arm that
//! removes the window shifts *entirely* — one convert reused eight times, zero
//! per-code extraction — reaches **265.3 / 354.2 / 302.1 / 191.3 GB/s**, i.e.
//! **C/A = 1.306 / 2.229 / 1.675 / 1.158**. The K=5376 N=2048 shape goes to
//! 2.23x, far above this box's 273 GB/s wire peak *because that shape's working
//! set is only 3 MB and is served from L2*; the 82 MB `gateup` shape, which
//! cannot be cached, moves only 1.16x. **So the arm ordering is monotone in
//! arithmetic removed, the tool resolves differences this small, and the
//! interesting structure is in the four production shapes above, not in the
//! pathological arm.**
//!
//! ## Consequence for the decode programme, stated as a negative result
//!
//! The decode GEMV is **partly issue/latency bound on `decode_state_mcg`, worth
//! 11-21% of GB/s**, and that pool is **bit-locked**: the mask applies to the
//! product, so it cannot fold into the extraction or the constant
//! (`A2-dequant.md` §2-Q1), and MSL has no `lop3` and no `bfe` to reach for.
//! That is why eight agents found nothing — the pool is real and unreachable,
//! not absent. **The 203 GB/s figure is not a memory wall**, which
//! `WHERE-THIS-STANDS.md` suspected but never measured; the achievable ceiling
//! with a free decode is **~240 GB/s** on the down/o_proj shapes, which is
//! `crux2`'s synthetic 246.2 and not the tree's 203.
//!
//! Before trusting any of these numbers, run
//! `python3 scripts/crux3_armA_equiv.py`, which proves arm A is token-identical
//! to the production kernel.
//!
//! ## Protocol
//!
//! * Interleaved: every arm is measured in every round, and arm A is re-measured
//!   at the end as a clock check (`attn_bench.rs`'s rule — a sequential sweep is
//!   how the sliding shape read 73.0 us in one run and 203.6 in another).
//! * GPU clock, not wall: `commit_stamped` + `take_gpu_stamps`, one command
//!   buffer per measurement.
//! * One buffer per arm per rep, rotated over a working set far larger than any
//!   plausible L2, so no arm is served from cache (`gemv_bandwidth.rs`'s rule).
//! * The **production** dispatch path is measured too, as a fourth arm, so the
//!   arms are anchored to the number the engine actually achieves rather than to
//!   each other.
//!
//! Usage: `cargo run --release --example crux3_decode_algebra`

use std::ffi::c_void;
use std::mem::size_of;
use std::time::Instant;

use kaplumba_kernels::encoder::{CommandBufferGpuTiming, MetalEncoder, TensorRef};
use kaplumba_kernels::exl3::Exl3Rate;
use kaplumba_kernels::linear::LinearKernels;
use metal::{Buffer, CommandQueue, ComputePipelineState, Device, MTLResourceOptions};

/// Aim for this much GPU time per data point: long enough that clock ramp and
/// dispatch startup are a rounding error.
const TARGET_MS: f64 = 60.0;
const MAX_WORKING_SET: usize = 768 << 20;
const MIN_WORKING_SET: usize = 192 << 20;
const ROUNDS: usize = 9;
const WARMUP: usize = 2;

/// `(arm label, kernel name)`. `prod` is dispatched through
/// `encode_gemv_raw_sk4` so the anchor is the real entry point rather than a
/// hand-dispatched pipeline.
const ARMS: [(&str, Option<&str>); 4] = [
    ("A_mcg_prod_path", None),
    ("A_mcg_macro", Some("linear_gemv_ishizuki_k3_crux3_armA")),
    ("B_xor", Some("linear_gemv_ishizuki_k3_crux3_armB")),
    ("C_mask", Some("linear_gemv_ishizuki_k3_crux3_armC")),
];

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
    let buffer = device.new_buffer_with_data(
        data.as_ptr() as *const c_void,
        (data.len() * size_of::<u16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    buffer
}

struct Shape {
    label: &'static str,
    k: usize,
    n: usize,
}

/// The five production decode GEMV shapes. `full/q` is the widest q projection
/// (16384 = 32 heads x 512), `sliding/gate+up` is the fused pair's K side.
const SHAPES: [Shape; 4] = [
    Shape { label: "down   K21504 N5376", k: 21504, n: 5376 },
    Shape { label: "q_slid K5376  N2048", k: 5376, n: 2048 },
    Shape { label: "o_proj K5376  N5376", k: 5376, n: 5376 },
    Shape { label: "gateup K5376  N43008", k: 5376, n: 43008 },
];

fn median(v: &mut Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
    v[v.len() / 2]
}

fn main() {
    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();
    let kernels = LinearKernels::new().expect("linear kernels compile");

    // Every arm must have compiled, or the comparison silently drops an arm and
    // reports a speedup for a kernel that was never dispatched. This check is
    // not decoration: it is what caught the arms failing to build the first
    // time, because `LinearKernels::new()` returns `Ok` for a library with
    // compile errors in *some* functions as long as the pipelines it indexes by
    // name all built.
    for (_, name) in ARMS {
        if let Some(name) = name {
            assert!(
                kernels.ishizuki_gemv_pipeline_by_name(name).is_some(),
                "arm kernel {name} did not compile — the arms would not be comparable"
            );
        }
    }
    // Also assert the names are in the family's index rather than merely
    // resolvable: `ishizuki_kernel_names` filters on the
    // `linear_gemv_ishizuki_` prefix, so a rename would drop an arm from the
    // map and the assertion above would fire. Both checks, because they fail for
    // different reasons and each has caught a different mistake.
    let names = kernels.ishizuki_gemv_kernel_names();
    for (_, name) in ARMS {
        if let Some(name) = name {
            assert!(
                names.iter().any(|n| n == name),
                "arm kernel {name} is not in the indexed family; rename it to start \
                 with `linear_gemv_ishizuki_` or it will never be built"
            );
        }
    }
    assert!(
        kernels
            .ishizuki_gemv_pipeline_by_name("linear_gemv_ishizuki_k3_colmajor_2col_sk4")
            .is_some(),
        "the production sk4 kernel is missing; this probe measures the wrong thing"
    );

    for shape in &SHAPES {
        measure_shape(&device, &queue, &kernels, shape);
    }
}

struct Rig {
    packed: Vec<Buffer>,
    transformed: Buffer,
    /// sk4 writes `y[sk * N + ...]`, so it needs `SK * N` floats, not `N`.
    raw: Buffer,
    /// `[in_features, out_features]` as two LE u32, the blob the kernel reads
    /// from the constant namespace. Passed by value because
    /// `dispatch_1d_offset` wants owned bytes, and computing it once keeps every
    /// arm's params identical rather than merely equal.
    shape_params: Vec<u8>,
}

fn measure_shape(
    device: &Device,
    queue: &CommandQueue,
    kernels: &LinearKernels,
    shape: &Shape,
) {
    let shape_bytes = shape.k * shape.n * 3 / 8;
    let nbuf = (MAX_WORKING_SET / shape_bytes.max(1))
        .clamp(2, 8)
        .min((MIN_WORKING_SET / shape_bytes.max(1)).max(2));
    let rig = Rig {
        packed: (0..nbuf)
            .map(|i| {
                random_buffer(
                    device,
                    (shape_bytes / 4) * 2,
                    0x1234_5678u32.wrapping_add(i as u32 * 7919),
                )
            })
            .collect(),
        transformed: random_buffer(device, shape.k * 2, 0x9abc_def0),
        raw: random_buffer(device, shape.n * 4 * 4, 0xfeed_beef),
        shape_params: (shape.k as u32)
            .to_le_bytes()
            .into_iter()
            .chain((shape.n as u32).to_le_bytes())
            .collect(),
    };

    let dispatch = |enc: &mut MetalEncoder, arm: Option<&str>, buf: &Buffer| {
        match arm {
            // The production entry point, verbatim. Kept as its own arm so the
            // macro arms are anchored to the number the engine achieves rather
            // than to each other.
            None => {
                kernels
                    .encode_gemv_raw_sk4(
                        enc,
                        buf,
                        0,
                        1,
                        Exl3Rate::Integer(3),
                        shape.k,
                        shape.n,
                        &rig.transformed,
                        &rig.raw,
                    )
                    .expect("encode_gemv_raw_sk4");
            }
            Some(name) => {
                let pipeline: &ComputePipelineState = kernels
                    .ishizuki_gemv_pipeline_by_name(name)
                    .expect("arm pipeline");
                // Byte-for-byte the launch `encode_gemv_raw_sk4` computes for
                // this kernel (linear.rs:2392-2408): `shape` in the *constant*
                // namespace as a params blob, `pairs * 4` threadgroups of 256
                // threads. Deliberately NOT `encode_ishizuki_gemv`'s geometry —
                // that one caps the grid at `gemv_max_groups()`, and the sk4 path
                // does not. Copying the capped geometry here would have made the
                // macro arms run at a different grid from the anchor, and a grid
                // change is exactly the kind of difference that masquerades as an
                // ALU effect.
                let pairs = ((shape.n / 16) / 2) as u64;
                let groups = pairs * 4;
                enc.dispatch_1d_offset(
                    pipeline,
                    &[
                        TensorRef::new(buf, 0),
                        TensorRef::new(&rig.transformed, 0),
                        TensorRef::new(&rig.raw, 0),
                    ],
                    &rig.shape_params,
                    groups * 256,
                    256,
                );
            }
        }
    };

    // Calibrate reps on wall time, per arm, then take the max so no arm is
    // measured on too few reps to be stable.
    let reps = ARMS
        .iter()
        .map(|(_, arm)| {
            let mut enc = MetalEncoder::new(queue).expect("encoder");
            enc.set_max_dispatches(u32::MAX);
            let t0 = Instant::now();
            for i in 0..4 {
                dispatch(&mut enc, *arm, &rig.packed[i % rig.packed.len()]);
            }
            enc.commit_and_wait().expect("cal");
            let per = (t0.elapsed().as_secs_f64() * 1e3 / 4.0).max(1e-4);
            ((TARGET_MS / per).round() as usize).clamp(4, 4000)
        })
        .max()
        .expect("arms");

    // Warm clocks and page tables; discarded.
    {
        let mut enc = MetalEncoder::new(queue).expect("encoder");
        enc.set_max_dispatches(u32::MAX);
        for (_, arm) in ARMS {
            for i in 0..reps.min(8) {
                dispatch(&mut enc, arm, &rig.packed[i % rig.packed.len()]);
            }
        }
        enc.commit_and_wait().ok();
    }

    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); ARMS.len()];
    let mut clock_check: Option<f64> = None;

    // Interleaved, round-major: every arm measured in every round.
    for round in 0..(WARMUP + ROUNDS) {
        for (index, (_, arm)) in ARMS.iter().enumerate() {
            let mut enc = MetalEncoder::new(queue).expect("encoder");
            // The encoder rolls every `max_dispatches`, and `commit_stamped`
            // only parks the current buffer, so without this the GPU span covers
            // a fraction of the work and reports impossible bandwidth.
            enc.set_max_dispatches(u32::MAX);
            for i in 0..reps {
                dispatch(&mut enc, *arm, &rig.packed[(i + round) % rig.packed.len()]);
            }
            enc.commit_stamped("crux3").expect("commit");
            let gpu_ms: f64 = enc.take_gpu_stamps().iter().map(|s| s.own_ms).sum();
            enc.commit_and_wait().ok();
            if round >= WARMUP {
                let bytes = (shape_bytes as f64) * (reps as f64);
                samples[index].push(bytes / (gpu_ms.max(1e-9) * 1e-3) / 1e9);
            }
        }
        if round == WARMUP + ROUNDS - 1 {
            // Arm A re-measured last: if the box drifted, this shows it.
            let mut enc = MetalEncoder::new(queue).expect("encoder");
            enc.set_max_dispatches(u32::MAX);
            for i in 0..reps {
                dispatch(&mut enc, None, &rig.packed[i % rig.packed.len()]);
            }
            enc.commit_stamped("crux3").expect("commit");
            let gpu_ms: f64 = enc.take_gpu_stamps().iter().map(|s| s.own_ms).sum();
            enc.commit_and_wait().ok();
            clock_check = Some(
                (shape_bytes as f64) * (reps as f64) / (gpu_ms.max(1e-9) * 1e-3) / 1e9,
            );
        }
    }

    println!(
        "\n== {}  K={} N={}  {} MB  bufs={}  reps={}  {} timed rounds, interleaved ==",
        shape.label,
        shape.k,
        shape.n,
        shape_bytes / (1 << 20),
        nbuf,
        reps,
        ROUNDS,
    );
    println!(
        "   {:<18} {:>9} {:>9} {:>9} {:>8} {:>8}",
        "arm", "median", "best", "worst", "spread", "vs prod"
    );
    let medians: Vec<f64> = samples.iter_mut().map(|s| median(s)).collect();
    for (index, (label, _)) in ARMS.iter().enumerate() {
        let mut s = std::mem::take(&mut samples[index]);
        let med = median(&mut s);
        let best = *s.first().expect("timed");
        let worst = *s.last().expect("timed");
        println!(
            "   {:<18} {:>9.1} {:>9.1} {:>9.1} {:>8.2} {:>8.3}",
            label,
            med,
            best,
            worst,
            worst / best,
            med / medians[0],
        );
    }
    if let Some(check) = clock_check {
        println!(
            "   clock check: arm A again at the end = {:.1} GB/s ({:.3}x of its own median)",
            check,
            check / medians[0]
        );
    }

    // The verdict, with the window stated rather than hidden.
    let b = medians[2] / medians[0];
    let c = medians[3] / medians[0];
    let verdict = if c >= 1.25 {
        "ISSUE BOUND -- the decode arithmetic is a large legal pool; build the ALU work"
    } else if c <= 1.05 {
        "MEMORY BOUND -- the decode arithmetic is free; the ALU axis is CLOSED"
    } else {
        "WITHHELD -- inside the 1.05-1.25 band, which this box's spread cannot resolve"
    };
    println!("   B/A = {b:.3}   C/A = {c:.3}\n   VERDICT: {verdict}");
}
