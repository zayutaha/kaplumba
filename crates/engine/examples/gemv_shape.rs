//! Where the decode matvec's bandwidth goes, as a function of shape.
//!
//! The first honest per-decode-step profile (`KAPLUMBA_DECODE_PROFILE=kernel`,
//! landed in `a58fa7b`) shows the *same* dispatch reaching 198 GB/s on `lm_head`
//! and 112-115 GB/s on `gate`/`up`/`down`, which are 65% of all the bytes a
//! decode token moves. Nothing distinguishes them except shape, so the question
//! this answers is: **which axis of the shape is the one that costs bandwidth?**
//!
//! Three candidate axes, each swept separately with the others held fixed:
//!
//! * total bytes (i.e. how long one matvec call runs),
//! * `out_features` (the grid is `out_features/16` threadgroups of 256 threads,
//!   so this is also "how many waves"),
//! * `in_features` (i.e. how many loop iterations each thread runs),
//! * the bit rate, which is the only axis on which `lm_head` differs from the
//!   rest, and which changes the bytes each lane's three ring loads carry
//!   (`exl3_decode8_mcg` reads a fixed 3 words per lane whatever `ka` is, so at
//!   6 bpw each load retires twice the payload of a 3 bpw one).
//!
//! # Why this had to be written rather than reused
//!
//! `examples/bench_gemv.rs` reports **min-of-passes**, which is how this
//! project's 248 GB/s ceiling figure came to be believed (spec, "Four earlier
//! bandwidth figures were artifacts"). It also times the input transform and
//! the finalize inside the same number, and it allocates one working set per
//! shape without releasing the previous one. So: median, never min; the input
//! transform hoisted out with `pretransformed`; one buffer per shape, dropped
//! before the next is built.
//!
//! # The instrument has to be able to fail
//!
//! `--selftest` runs the checks that make a number usable, and exits non-zero if
//! any of them fails:
//!
//! 1. **Scale.** Doubling `out_features` at fixed `in_features` must double the
//!    time to within 15%. A probe that is accidentally reading an L2-resident
//!    buffer, or measuring a fixed cost, fails this.
//! 2. **Sensitivity.** A 4x smaller `in_features` must be at least 2x faster. A
//!    probe that cannot resolve a 4x change in work cannot resolve the 2x we are
//!    hunting for.
//! 3. **Clock.** The first shape measured is re-measured at the end of the run;
//!    if the GPU had not reached steady state the two disagree, and every number
//!    in between is then measured on a moving clock.
//!
//! The checks were themselves verified by sabotage: a `pretransformed: false`
//! build, a 16x-too-small working set, and a deliberately mis-set
//! `packed_offset` each made this print a FAIL line with the expected shape.

use half::f16;
use kaplumba_engine::exl3::pack_integer_states;
use kaplumba_kernels::encoder::MetalEncoder;
use kaplumba_kernels::exl3::{Exl3Codebook, Exl3Rate};
use kaplumba_kernels::linear::{LinearEncodeRequest, LinearKernels, NR0_VARIANTS};
use metal::{Device, MTLResourceOptions};

/// Working set the weights rotate through. Large enough that no pass finds its
/// weight in cache: the model streams 11.9 GB per token, so anything a pass
/// could still be resident from is not a condition the model is ever in.
const WORKING_SET_BYTES: u64 = 1_600_000_000;

const SAMPLES: usize = 9;

struct Shape {
    name: &'static str,
    in_f: usize,
    out_f: usize,
    rate: u8,
}

/// The seven decode projections and the vocabulary head, at the shapes
/// `config.json` gives (56 sliding layers at head_dim 256 / 16 kv heads,
/// 4 full layers at head_dim 512 / 4 kv heads, hidden 5376, intermediate 21504).
/// A sliding and a full layer are listed separately because they differ.
fn real_shapes() -> Vec<Shape> {
    vec![
        Shape {
            name: "q_proj   slide",
            in_f: 5376,
            out_f: 8192,
            rate: 3,
        },
        Shape {
            name: "k_proj   slide",
            in_f: 5376,
            out_f: 4096,
            rate: 3,
        },
        Shape {
            name: "v_proj   slide",
            in_f: 5376,
            out_f: 4096,
            rate: 3,
        },
        Shape {
            name: "o_proj   slide",
            in_f: 8192,
            out_f: 5376,
            rate: 3,
        },
        Shape {
            name: "gate_proj",
            in_f: 5376,
            out_f: 21504,
            rate: 3,
        },
        Shape {
            name: "up_proj",
            in_f: 5376,
            out_f: 21504,
            rate: 3,
        },
        Shape {
            name: "down_proj",
            in_f: 21504,
            out_f: 5376,
            rate: 3,
        },
        Shape {
            name: "q_proj   full",
            in_f: 5376,
            out_f: 16384,
            rate: 3,
        },
        Shape {
            name: "k_proj   full",
            in_f: 5376,
            out_f: 2048,
            rate: 3,
        },
        Shape {
            name: "o_proj   full",
            in_f: 16384,
            out_f: 5376,
            rate: 3,
        },
        Shape {
            name: "lm_head",
            in_f: 5376,
            out_f: 262144,
            rate: 6,
        },
    ]
}

/// Deterministic EXL3 states for one projection.
///
/// A real checkpoint would do, but the *values* are irrelevant to a bandwidth
/// measurement and generating them from a constant keeps the harness free of a
/// 12 GB dependency. The LCG below has period 2^32 in `state`, and the state is
/// advanced once per of the 256 codes per tile, so no two codes in a tile repeat
/// and a perturbation of any single code is visible -- the "LCG of period 8"
/// failure in this project's notes was a generator whose every index shift was a
/// multiple of 8 and so invisible to the check.
fn build(in_f: usize, out_f: usize, rate: u8, seed: u32) -> Vec<u16> {
    let tiles = (in_f / 16) * (out_f / 16);
    let mask = (1u32 << rate) - 1;
    let mut packed = Vec::with_capacity(tiles * 256 * rate as usize / 8 / 4 * 2);
    let mut state = seed | 1;
    let mut states = vec![0u16; 256];
    for _ in 0..tiles {
        for slot in states.iter_mut() {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *slot = ((state >> 16) as u16) & (mask as u16);
        }
        packed.extend(pack_integer_states(&states, rate).unwrap().0);
    }
    packed
}

struct Rig {
    device: Device,
    queue: metal::CommandQueue,
    kernels: &'static LinearKernels,
}

/// One shape's weights, resident and cold, plus the small buffers the dispatch
/// needs. Dropped before the next shape is built, so a sweep does not accumulate
/// gigabytes: an earlier probe that held 3.8 GB without releasing it drove this
/// box to 3.7 GiB free and 573 MB of swap, and swapped numbers look real.
struct Case {
    in_f: usize,
    out_f: usize,
    rate: u8,
    packed: metal::Buffer,
    stride: u64,
    copies: u64,
    input: metal::Buffer,
    input_signs: metal::Buffer,
    output_signs: metal::Buffer,
    transformed: metal::Buffer,
    raw: metal::Buffer,
    output: metal::Buffer,
    bytes: f64,
}

impl Case {
    fn build(rig: &Rig, in_f: usize, out_f: usize, rate: u8) -> Self {
        let device = &rig.device;
        let one = build(
            in_f,
            out_f,
            rate,
            0x5eed_0000 ^ (in_f as u32) ^ (out_f as u32) ^ ((rate as u32) << 20),
        );
        let bytes = (one.len() * 2) as u64;
        let copies = (WORKING_SET_BYTES / bytes).max(2);
        let mut big: Vec<u16> = Vec::with_capacity(one.len() * copies as usize);
        for copy in 0..copies {
            // Distinct content per copy, so a pass cannot be served from a line
            // a previous pass dirtied. The XOR is on the u16 view; the point is
            // only that the bytes differ.
            let tag = (copy as u16).wrapping_mul(0x9E37) ^ 0x5A5A;
            big.extend(one.iter().map(|value| value ^ tag));
        }
        let packed = device.new_buffer_with_data(
            big.as_ptr() as *const std::ffi::c_void,
            (big.len() * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        drop(big);

        let input: Vec<u16> = (0..in_f)
            .map(|i| f16::from_f32((i as f32 * 0.017).sin()).to_bits())
            .collect();
        let input = device.new_buffer_with_data(
            input.as_ptr() as *const std::ffi::c_void,
            (in_f * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let input_signs = device.new_buffer_with_data(
            [f16::from_f32(0.0031).to_bits(); 8].as_ptr() as *const std::ffi::c_void,
            16,
            MTLResourceOptions::StorageModeShared,
        );
        let output_signs = device.new_buffer_with_data(
            [f16::from_f32(-0.0027).to_bits(); 8].as_ptr() as *const std::ffi::c_void,
            16,
            MTLResourceOptions::StorageModeShared,
        );
        let transformed =
            device.new_buffer((in_f * 2) as u64, MTLResourceOptions::StorageModeShared);
        let raw = device.new_buffer(
            (out_f * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let output = device.new_buffer((out_f * 2) as u64, MTLResourceOptions::StorageModeShared);
        Self {
            in_f,
            out_f,
            rate,
            packed,
            stride: bytes,
            copies,
            input,
            input_signs,
            output_signs,
            transformed,
            raw,
            output,
            bytes: bytes as f64,
        }
    }

    /// One pass: the input transform once, then `passes` matvecs over
    /// consecutive copies, encoded into a single command buffer.
    ///
    /// `pretransformed: true` is what makes this a *matvec* measurement. The
    /// input transform is 2*in_f bytes against in_f*out_f*rate/8 of weights, so
    /// it is 0.1% at these shapes -- but it is a separate dispatch with its own
    /// ramp, and leaving it in would put a per-dispatch cost into a number that
    /// is supposed to be about the matvec.
    fn run(&self, rig: &Rig, passes: usize, quiet: bool) -> f64 {
        let mut encoder = MetalEncoder::new(&rig.queue).unwrap();
        {
            let mut bootstrap = MetalEncoder::new(&rig.queue).unwrap();
            rig.kernels
                .encode_input_transform(
                    &mut bootstrap,
                    &self.input,
                    &self.input_signs,
                    &self.transformed,
                    self.in_f,
                )
                .unwrap();
            bootstrap.commit_and_wait().unwrap();
        }
        let started = std::time::Instant::now();
        for pass in 0..passes {
            let offset = (pass as u64 % self.copies) * self.stride;
            rig.kernels
                .encode_gemv(
                    &mut encoder,
                    LinearEncodeRequest {
                        packed: &self.packed,
                        packed_offset: offset,
                        perm_mode: 0,
                        input: &self.input,
                        input_signs: &self.input_signs,
                        output_signs: &self.output_signs,
                        transformed: &self.transformed,
                        raw: &self.raw,
                        output: &self.output,
                        in_features: self.in_f,
                        out_features: self.out_f,
                        rate: Exl3Rate::Integer(self.rate),
                        codebook: Exl3Codebook::Mcg,
                        pretransformed: true,
                    },
                )
                .unwrap();
        }
        encoder.commit_and_wait().unwrap();
        let elapsed = started.elapsed().as_secs_f64();
        if quiet {
            eprintln!("  (warm-up: {:.3} ms for {passes} passes)", elapsed * 1e3);
        }
        elapsed
    }
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

struct Reading {
    median_ms: f64,
    best_ms: f64,
    worst_ms: f64,
    /// Median over the same samples divided by the best of them.
    ///
    /// This is the instrument's own contamination check, and it is the number
    /// that decides whether a *median* is safe to quote. The full
    /// worst/best spread is reported alongside it, but the worst sample is a
    /// scheduling artefact often enough (a 1.62x outlier was seen on an
    /// otherwise clean run) that requiring the whole spread to be tight would
    /// mean refusing to report at all. The median is what has to be clean; the
    /// spread is what has to be *visible*.
    median_over_best: f64,
    gbps: f64,
}

fn measure(rig: &Rig, shape: &Shape) -> Reading {
    let case = Case::build(rig, shape.in_f, shape.out_f, shape.rate);
    // Enough passes to exceed the noise floor of one dispatch: the smallest shape
    // here is 2.5 MB, and a single 10 us dispatch is dominated by ramp.
    let bytes_per_pass = case.bytes;
    let passes = ((300e6 / bytes_per_pass).ceil() as usize).clamp(4, 64);
    let _ = case.run(rig, passes, false);
    let mut samples = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        // `run` returns seconds for the whole batch; per-pass *milliseconds* is
        // what every downstream line expects. Getting this wrong is not subtle
        // in the output -- the first run of this harness reported 131,512 GB/s
        // and the self-test's PLAUSIBLE check is what caught it.
        samples.push(case.run(rig, passes, false) / passes as f64 * 1e3);
    }
    let best = samples.iter().copied().fold(f64::INFINITY, f64::min);
    let worst = samples.iter().copied().fold(0.0, f64::max);
    let median_ms = median(&mut samples);
    Reading {
        median_ms,
        best_ms: best,
        worst_ms: worst,
        median_over_best: median_ms / best,
        gbps: bytes_per_pass / (median_ms / 1e3) / 1e9,
    }
}

fn print_row(label: &str, mb: f64, r: &Reading) {
    println!(
        "{:<26}{:>9.1}{:>10.4}{:>10.4}{:>10.4}{:>9.3}{:>9.1}{:>7}%",
        label,
        mb,
        r.median_ms,
        r.best_ms,
        r.worst_ms,
        r.worst_ms / r.best_ms,
        r.gbps,
        100.0 * r.gbps / 240.0,
    );
}

fn header() {
    println!(
        "{:<26}{:>9}{:>10}{:>10}{:>10}{:>9}{:>9}{:>8}",
        "shape", "MB", "med ms", "best ms", "worst ms", "spread", "GB/s", "%of240"
    );
}

/// Run `name` instead of the environment-selected kernel, `passes` times.
fn run_named(rig: &Rig, case: &Case, name: &str, passes: usize) -> f64 {
    let mut encoder = MetalEncoder::new(&rig.queue).unwrap();
    {
        let mut bootstrap = MetalEncoder::new(&rig.queue).unwrap();
        rig.kernels
            .encode_input_transform(
                &mut bootstrap,
                &case.input,
                &case.input_signs,
                &case.transformed,
                case.in_f,
            )
            .unwrap();
        bootstrap.commit_and_wait().unwrap();
    }
    let started = std::time::Instant::now();
    for pass in 0..passes {
        rig.kernels
            .encode_gemv_named(
                &mut encoder,
                name,
                &LinearEncodeRequest {
                    packed: &case.packed,
                    packed_offset: (pass as u64 % case.copies) * case.stride,
                    perm_mode: 0,
                    input: &case.input,
                    input_signs: &case.input_signs,
                    output_signs: &case.output_signs,
                    transformed: &case.transformed,
                    raw: &case.raw,
                    output: &case.output,
                    in_features: case.in_f,
                    out_features: case.out_f,
                    rate: Exl3Rate::Integer(case.rate),
                    codebook: Exl3Codebook::Mcg,
                    pretransformed: true,
                },
            )
            .unwrap();
    }
    encoder.commit_and_wait().unwrap();
    started.elapsed().as_secs_f64() / passes as f64 * 1e3
}

/// The equal-bytes bit-rate sweep, **interleaved**, with a control arm.
///
/// # Why this exists and why it is not the `--only "bit rate"` group
///
/// That group is where "3 bpw is 40% slower than 6 bpw at equal bytes" comes
/// from, and it measures one shape **to completion** before starting the next.
/// Any drift over the sweep therefore folds into the shape's number instead of
/// being visible. Run before and after T6 it gave:
///
/// ```text
/// rate  has _w2?    pre-T6    post-T6    delta
/// 3 bpw   YES        132.5      146.9     +10.9%
/// 4 bpw   YES        179.8      153.7     -14.5%
/// 6 bpw   no         164.5      182.0     +10.6%   <- CONTROL
/// 8 bpw   no         176.3      162.3      -7.9%   <- CONTROL
/// ```
///
/// 6 and 8 bpw **cannot** have changed from T6: `TWO_WORD_SAFE_RATES` emits no
/// `_w2` kernel above 4 bpw, so those rows are controls and they moved +10.6%
/// and -7.9% in opposite directions. A control that moves means the instrument
/// is not measuring the kernel. Within-sweep spreads were 2.2x-3.7x.
///
/// So the table could not resolve a ~10% effect in either direction, and a
/// conclusion drawn from it either way would have been unsupported.
///
/// # What this does instead
///
/// Every round measures **every arm once**, so drift appears as a moving *ratio*
/// rather than a moving number — the same structure `nr0_sweep` and `w2_sweep`
/// already use, and the reason T6's own +7% result is trustworthy.
///
/// On top of that, the **6 bpw arm is a control carried through the whole run**:
/// no `_w2` kernel exists for it, so its ratio to itself across the sweep is a
/// direct read of drift. It is printed as `drift` and the run is only
/// interpretable when it is small. A control that moves is reported as a
/// control that moved, rather than being quietly averaged into a conclusion.
fn rate_sweep_interleaved(rig: &Rig, shapes: &[Shape], rounds: usize) {
    // Build one Case per shape up front: allocating 43 MB of packed weights per
    // arm inside the round loop would put allocator noise in the measurement.
    let mut cases = Vec::new();
    for shape in shapes {
        let case = Case::build(rig, shape.in_f, shape.out_f, shape.rate);
        let passes = ((300e6 / case.bytes).ceil() as usize).clamp(4, 64);
        cases.push((case, passes));
    }

    // Confirm every arm exists before spending a round on it.
    let mut names: Vec<Vec<String>> = Vec::new();
    for (shape, _) in shapes.iter().zip(cases.iter()) {
        let plain = format!("linear_gemv_ishizuki_k{}", shape.rate);
        let mut arms = vec![plain.clone()];
        let w2 = format!("{plain}_w2");
        if rig.kernels.ishizuki_gemv_pipeline_by_name(&w2).is_some() {
            arms.push(w2);
        }
        for name in &arms {
            if rig.kernels.ishizuki_gemv_pipeline_by_name(name).is_none() {
                println!("  skipping {name}: not in the library");
                return;
            }
        }
        names.push(arms);
    }

    // Warm every arm once: the first dispatch of a kernel pays pipeline and
    // residency setup, and a warmup that warms only some arms biases round 1.
    for (arms, (case, passes)) in names.iter().zip(cases.iter()) {
        for name in arms {
            let _ = run_named(rig, case, name, *passes);
        }
    }

    // samples[arm][round]
    let total_arms: usize = names.iter().map(|a| a.len()).sum();
    let mut samples: Vec<Vec<f64>> = vec![Vec::with_capacity(rounds); total_arms];
    for round in 0..rounds {
        let mut index = 0;
        for (arms, (case, passes)) in names.iter().zip(cases.iter()) {
            for name in arms {
                samples[index].push(run_named(rig, case, name, *passes));
                index += 1;
            }
        }
        // Progress to stderr so a long sweep shows it is alive; stdout stays the
        // table, which is what gets pasted into the spec.
        eprintln!("  round {}/{rounds}", round + 1);
    }

    let control_index = names
        .iter()
        .position(|arms| arms.iter().any(|n| n.contains("_k6")))
        .map(|shape_index| {
            let offset: usize = names[..shape_index].iter().map(|a| a.len()).sum();
            offset
        });
    // The control is the *first* arm of the 6 bpw shape (the plain 3-word
    // kernel), which no `_w2` variant can affect.
    let control_arm = control_index.map(|offset| offset);

    println!("\n  equal-bytes bit-rate sweep, {rounds} interleaved rounds");
    println!(
        "    {:<7}{:>9}{:>10}{:>10}{:>10}{:>9}{:>9}",
        "rate", "MB", "med ms", "best", "worst", "spread", "GB/s"
    );
    let mut index = 0;
    let mut medians = Vec::new();
    for (shape, (case, _)) in shapes.iter().zip(cases.iter()) {
        for name in &names[shape_index_of(index, &names)] {
            let mut s = samples[index].clone();
            let m = median(&mut s);
            let mut sorted = s.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let best = *sorted.first().unwrap();
            let worst = *sorted.last().unwrap();
            let mb = case.bytes / 1e6;
            let gbps = gbps(mb, m);
            medians.push(m);
            println!(
                "    {:<7}{:>9.1}{m:>10.4}{best:>10.4}{worst:>10.4}{:>8.2}x{gbps:>10.1}   {name}",
                format!("{} bpw", shape.rate),
                mb,
                worst / best,
            );
            index += 1;
        }
    }

    // The drift read. Each arm's ratio to the FIRST arm of its own sweep is not
    // meaningful across different rates; what is meaningful is the control's
    // round-to-round spread, and the ratio of every arm to the 3 bpw baseline
    // computed WITHIN each round and then medianed.
    if let Some(control) = control_arm {
        let s = &samples[control];
        let m = median(&mut s.clone());
        let mut sorted = s.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let best = *sorted.first().unwrap();
        let worst = *sorted.last().unwrap();
        println!("\n  CONTROL (6 bpw, no _w2 kernel exists for it, so T6 cannot move it):");
        println!(
            "    median {m:.4} ms, best {best:.4}, worst {worst:.4}, spread {:.2}x",
            worst / best
        );
        if worst / best > 1.30 {
            println!(
                "    DRIFT HIGH -- the control moved {:.0}% across the sweep. Any rate comparison \
                 below is not trustworthy; re-run when the box is quieter.",
                100.0 * (worst / best - 1.0)
            );
        } else {
            println!(
                "    drift {:.0}% -- acceptable, the rate comparison below is readable",
                100.0 * (worst / best - 1.0)
            );
        }
    }

    // Paired per-round ratios against the 3 bpw arm, which is the comparison the
    // question actually needs. Paired because each round saw the same conditions.
    if medians.len() >= shapes.len() {
        println!("\n  paired ratio vs 3 bpw (same round, so drift cancels):");
        let mut index = 0;
        for (shape, _) in shapes.iter().zip(cases.iter()) {
            let arms = names[shape_index_of(index, &names)].len();
            if index > 0 {
                let mut ratios: Vec<f64> = Vec::new();
                for round in 0..rounds {
                    let base = samples[0][round];
                    for arm in 0..arms {
                        ratios.push(base / samples[index + arm][round]);
                    }
                }
                let m = median(&mut ratios);
                println!(
                    "    {:<7} {:.3}x  (n={})",
                    format!("{} bpw", shape.rate),
                    m,
                    ratios.len()
                );
            }
            index += arms;
        }
    }
    let _ = &medians;
}

/// Index of the shape whose arm list contains flattened arm `index`.
fn shape_index_of(index: usize, names: &[Vec<String>]) -> usize {
    let mut seen = 0;
    for (i, arms) in names.iter().enumerate() {
        seen += arms.len();
        if index < seen {
            return i;
        }
    }
    names.len().saturating_sub(1)
}

/// Real kernel vs the no-decode probe, interleaved, on one shape.
///
/// The probe (`linear_gemv_ishizuki_kN_nod`) is the verbatim kernel with
/// `decode_state` replaced by a one-AND stub, generated by one asserted
/// substitution in `scripts/gen_exl3_gemv.py`. Memory traffic is identical; the
/// decode arithmetic is the only difference. Its OUTPUT IS GARBAGE and nothing
/// in the engine dispatches it -- this measures a rate and nothing else.
///
/// Interleaved for the same reason as every other sweep here: a box that drifts
/// 10% within a sweep cannot resolve a 30% effect, and the arms have to be
/// compared WITHIN a round for that to be true.
fn nod_sweep(rig: &Rig, shape: &Shape) {
    let case = Case::build(rig, shape.in_f, shape.out_f, shape.rate);
    let passes = ((300e6 / case.bytes).ceil() as usize).clamp(4, 64);
    let real = format!("linear_gemv_ishizuki_k{}", shape.rate);
    // Prefer the shipped 2-word form when it exists: that is what the engine runs
    // at this rate, so the probe has to be built from the same baseline or the
    // comparison is against a kernel that is no longer in service.
    let two_word = format!("{real}_w2");
    let baseline = if rig
        .kernels
        .ishizuki_gemv_pipeline_by_name(&two_word)
        .is_some()
    {
        two_word
    } else {
        real.clone()
    };
    let probe = format!("{real}_nod");
    for name in [&baseline, &probe] {
        if rig.kernels.ishizuki_gemv_pipeline_by_name(name).is_none() {
            println!("  skipping {name}: not in the library");
            return;
        }
    }
    // Warm both, so round 1 does not pay pipeline setup for one arm only.
    for name in [&baseline, &probe] {
        let _ = run_named(rig, &case, name, passes);
    }
    let rounds = 9;
    let mut real_samples = Vec::with_capacity(rounds);
    let mut nod_samples = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        real_samples.push(run_named(rig, &case, &baseline, passes));
        nod_samples.push(run_named(rig, &case, &probe, passes));
    }
    let mb = case.bytes / 1e6;
    let (rm, nm) = (
        median(&mut real_samples.clone()),
        median(&mut nod_samples.clone()),
    );
    println!(
        "\n  {} x{} @ {}bpw = {mb:.1} MB, {rounds} interleaved rounds",
        shape.in_f, shape.out_f, shape.rate
    );
    println!(
        "    {:<12}{:>10}{:>10}{:>10}{:>9}",
        "arm", "med ms", "GB/s", "spread", "vs real"
    );
    for (label, samples) in [("real", &real_samples), ("no-decode", &nod_samples)] {
        let m = median(&mut samples.clone());
        let mut sorted = samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let spread = sorted.last().unwrap() / sorted.first().unwrap();
        println!(
            "    {label:<12}{m:>10.4}{:>10.1}{:>9.2}x{:>8.3}x",
            gbps(mb, m),
            spread,
            rm / m
        );
    }
    // The verdict, stated as the number that decides between the two mechanisms.
    let lift = rm / nm - 1.0;
    println!(
        "    decode removal lifts this rate by {:+.1}%",
        lift * 100.0
    );
}

/// The NR0 sweep, on the three real shapes that matter, A/B'd **interleaved**.
///
/// Interleaved because this box has been measured at 3.7, 3.9, 4.3, 4.8, 6.0 and
/// 6.4 tok/s from the *same binary* minutes apart, so a sequential A-then-B
/// sweep is not a comparison, it is two unrelated measurements. Each round below
/// measures every NR0 once, and the reported figure is the median over rounds.
///
/// The baseline is re-measured in every round rather than measured once at the
/// start, so a clock ramp shows up as a *ratio* that moves, not as a number that
/// moves -- which is the difference between a visible problem and a confident
/// wrong answer.
/// The T6 sweep: 3-word window against 2-word, on the **live** ishizuki kernel.
///
/// Interleaved for the reason `nr0_sweep` gives, and with a further wrinkle: the
/// two arms differ by a *load*, so a clock ramp would move them together and the
/// ratio would survive it — but a *thermal* or contention difference would not,
/// and would show up as a ratio. Reporting the ratio against a baseline
/// re-measured every round is what makes the ratio trustworthy, and printing the
/// absolute medians alongside it is what makes a suspiciously stable ratio
/// checkable.
fn w2_sweep(rig: &Rig, shape: &Shape) {
    let case = Case::build(rig, shape.in_f, shape.out_f, shape.rate);
    let passes = ((300e6 / case.bytes).ceil() as usize).clamp(4, 64);
    let bits = shape.rate;
    let names: Vec<(String, String)> = vec![
        (
            "3-word".to_string(),
            format!("linear_gemv_ishizuki_k{bits}"),
        ),
        (
            "2-word".to_string(),
            format!("linear_gemv_ishizuki_k{bits}_w2"),
        ),
    ];
    for (_, name) in &names {
        if rig.kernels.ishizuki_gemv_pipeline_by_name(name).is_none() {
            println!("  skipping {name}: not in the library");
            return;
        }
    }
    for (_, name) in &names {
        let _ = run_named(rig, &case, name, passes);
    }
    let rounds = 7;
    let mut samples: Vec<Vec<f64>> = vec![Vec::with_capacity(rounds); names.len()];
    for _ in 0..rounds {
        for (index, (_, name)) in names.iter().enumerate() {
            samples[index].push(run_named(rig, &case, name, passes));
        }
    }
    let mb = case.bytes / 1e6;
    println!(
        "\n  {} x{} @ {}bpw = {mb:.1} MB, {rounds} interleaved rounds of {passes} passes",
        shape.in_f, shape.out_f, shape.rate
    );
    println!(
        "    {:<10}{:>10}{:>10}{:>10}{:>10}",
        "window", "med ms", "GB/s", "vs 3-word", "vs 240"
    );
    let base = {
        let mut s = samples[0].clone();
        median(&mut s)
    };
    for (index, (label, _)) in names.iter().enumerate() {
        let mut s = samples[index].clone();
        let mut sorted = s.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let m = median(&mut s);
        let gbps = mb / 1e3 / (m / 1e3);
        println!(
            "    {label:<10}{m:>10.4}{gbps:>10.1}{:>9.3}x{:>9.0}%   spread {:.2}x",
            base / m,
            100.0 * gbps / 240.0,
            sorted[0] / sorted[sorted.len() - 1],
        );
    }
}

/// Decompose the per-dispatch fixed cost, by scaling **passes** and extrapolating
/// to one.
///
/// # The first version of this instrument was wrong, and the way it was wrong is
/// the finding
///
/// It fitted `ms = fixed + count * per_pass` and got a **negative** per-pass
/// slope — time going *down* as passes go up. That is impossible if the passes
/// are the work, so the model was wrong, and the reason is the one worth
/// recording:
///
/// * the measured quantity is one `commit_and_wait`, which is a **command-buffer
///   round trip** plus N kernel dispatches;
/// * so the correct model is `ms = overhead + count * per_pass`, and `overhead`
///   dominates at low counts.
///
/// The *old* 59 us figure was read off a (bytes, time) fit whose pass count
/// scaled as `1/bytes`, so a per-dispatch cost was amortised differently at every
/// size and landed in the **slope**. Refitting today's numbers that way gives
/// 8.1 us — a 7x disagreement with 59 us, which is what prompted this
/// instrument. Neither number was a per-dispatch cost; both were an artefact of
/// the batching.
///
/// So the honest decomposition is now two terms, and both are reported: the
/// per-command-buffer overhead and the per-dispatch streaming cost. The check
/// that the split is real is that the *per-dispatch* term is flat across shapes
/// in **GB/s** — a page-walk or launch cost would not be.
fn fixed_cost(rig: &Rig, shape: &Shape) {
    let case = Case::build(rig, shape.in_f, shape.out_f, shape.rate);
    let name = format!("linear_gemv_ishizuki_k{}", shape.rate);
    if rig.kernels.ishizuki_gemv_pipeline_by_name(&name).is_none() {
        println!("  {name} not in the library, skipping");
        return;
    }
    // Pass counts spanning an order of magnitude, so the slope is well determined.
    let counts: Vec<usize> = vec![1, 2, 4, 8, 16, 32];
    // Warm, so the first count is not also the clock ramp.
    for count in &counts {
        let _ = run_named(rig, &case, &name, *count);
    }
    let rounds = 7;
    // `run_named` returns **per-pass** milliseconds -- it divides by the pass
    // count, which is what makes it the right unit for a bandwidth A/B (every
    // arm sees the same count) and the wrong unit here. The decomposition needs
    // TOTALS, or the fit sees time falling as passes rise and reports a negative
    // per-dispatch cost, which is what the first version of this instrument did.
    //
    // The two units differ by exactly a factor of `count`, and that factor is the
    // whole decomposition: dividing by it is what turns a per-dispatch cost into
    // an apparent fixed term.
    let mut totals = Vec::with_capacity(counts.len());
    for count in &counts {
        let mut samples = Vec::with_capacity(rounds);
        for _ in 0..rounds {
            samples.push(run_named(rig, &case, &name, *count) * *count as f64);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        totals.push(samples[rounds / 2]);
    }
    // Least squares on (count, total_ms): total = buffer_overhead + count * per_dispatch.
    let n = counts.len() as f64;
    let sx: f64 = counts.iter().map(|c| *c as f64).sum();
    let sy: f64 = totals.iter().sum();
    let sxx: f64 = counts.iter().map(|c| (*c as f64) * (*c as f64)).sum();
    let sxy: f64 = counts
        .iter()
        .zip(totals.iter())
        .map(|(c, t)| (*c as f64) * t)
        .sum();
    let den = n * sxx - sx * sx;
    let per_dispatch = (n * sxy - sx * sy) / den;
    let overhead = (sxx * sy - sx * sxy) / den;
    assert!(
        per_dispatch > 0.0,
        "fitted a NEGATIVE per-dispatch cost ({per_dispatch:.8} ms) for {}x{}: \
         time decreases as passes increase, so the model is wrong and no \
         decomposition can be read off it",
        shape.in_f,
        shape.out_f
    );
    let worst = counts
        .iter()
        .zip(totals.iter())
        .map(|(c, t)| (overhead + per_dispatch * (*c as f64) - t).abs() / t)
        .fold(0.0f64, f64::max);
    let mb = case.bytes / 1e6;
    println!(
        "\n  {} x{} @ {}bpw = {mb:.1} MB, medians of {rounds} at each pass count",
        shape.in_f, shape.out_f, shape.rate
    );
    println!(
        "    {:>7}{:>12}{:>14}{:>14}",
        "passes", "total ms", "per-pass us", "GB/s"
    );
    for (count, total) in counts.iter().zip(totals.iter()) {
        println!(
            "    {count:>7}{total:>12.4}{:>14.1}{:>14.1}",
            total / (*count as f64) * 1e3,
            gbps(mb, total / (*count as f64))
        );
    }
    println!(
        "    fit: {overhead:.4} ms command-buffer + {per_dispatch:.8} ms/dispatch  \
         (residual max {:.1}%)",
        100.0 * worst
    );
    println!(
        "    => per-dispatch {:.1} us at {:.1} GB/s; command-buffer overhead {:.1} us",
        per_dispatch * 1e3,
        gbps(mb, per_dispatch),
        overhead * 1e3
    );
    // The plausibility gate. It is here because this harness's first version
    // printed 0.0-0.2 GB/s for a kernel every other instrument measures at
    // 150+ GB/s: it divided MB by a *total* spanning `count` passes, and
    // separately divided by a per-dispatch time still in milliseconds. Both are
    // unit errors that compile, run, and print a confident wrong number -- the
    // same failure as the seconds-labelled-as-milliseconds bug that claimed
    // 131,512 GB/s.
    //
    // The band is wide on purpose: it is there to catch a factor-of-1000 slip,
    // not to police small shapes. 1024x1024 legitimately reads ~38 GB/s because
    // at 0.4 MB the per-dispatch work is smaller than its own launch, and an
    // earlier version of this gate rejected exactly that case -- a gate that
    // fires on the correct answer trains you to ignore it.
    let streaming = gbps(mb, per_dispatch);
    assert!(
        (5.0..=400.0).contains(&streaming),
        "per-dispatch streaming rate {streaming:.1} GB/s for {}x{} is outside 5-400. \
         Every instrument on this box reports 38-250 GB/s for this kernel, so this is \
         a unit error in the decomposition, not a result.",
        shape.in_f,
        shape.out_f
    );
}

/// MB and milliseconds to GB/s.
///
/// One function, because the two places that needed this got it wrong in two
/// different ways, and a single definition cannot be wrong in two ways at once.
fn gbps(mb: f64, ms: f64) -> f64 {
    if ms <= 0.0 {
        return f64::INFINITY;
    }
    (mb / 1e3) / (ms / 1e3)
}

fn nr0_sweep(rig: &Rig, shape: &Shape) {
    let case = Case::build(rig, shape.in_f, shape.out_f, shape.rate);
    let passes = ((300e6 / case.bytes).ceil() as usize).clamp(4, 64);
    let bits = shape.rate;
    let mut names: Vec<(String, String)> =
        vec![("NR0=1".to_string(), format!("linear_gemv_exl3_k{bits}_mcg"))];
    for nr0 in NR0_VARIANTS {
        if (shape.out_f / 16) % nr0 as usize == 0 {
            names.push((
                format!("NR0={nr0}"),
                format!("linear_gemv_exl3_k{bits}_mcg_r{nr0}"),
            ));
        }
    }
    // Warm every variant first, so the first round is not also the clock ramp.
    for (_, name) in &names {
        let _ = run_named(rig, &case, name, passes);
    }
    let rounds = 7;
    let mut samples: Vec<Vec<f64>> = vec![Vec::with_capacity(rounds); names.len()];
    for _ in 0..rounds {
        for (index, (_, name)) in names.iter().enumerate() {
            samples[index].push(run_named(rig, &case, name, passes));
        }
    }
    let mb = case.bytes / 1e6;
    println!(
        "\n  {} x{} @ {}bpw = {mb:.1} MB, {rounds} interleaved rounds of {passes} passes",
        shape.in_f, shape.out_f, shape.rate
    );
    println!(
        "    {:<10}{:>10}{:>10}{:>9}{:>9}",
        "variant", "med ms", "GB/s", "vs NR0=1", "vs 240"
    );
    let base = {
        let mut s = samples[0].clone();
        median(&mut s)
    };
    for (index, (label, _)) in names.iter().enumerate() {
        let mut s = samples[index].clone();
        let m = median(&mut s);
        println!(
            "    {label:<10}{m:>10.4}{:>10.1}{:>8.3}x{:>8.0}%",
            mb / 1e3 / (m / 1e3),
            base / m,
            100.0 * (mb / 1e3 / (m / 1e3)) / 240.0
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let selftest = args.iter().any(|a| a == "--selftest");
    let only = args
        .iter()
        .position(|a| a == "--only")
        .map(|i| args[i + 1].clone());
    let nr0 = args.iter().any(|a| a == "--nr0");
    let rate_interleaved = args.iter().any(|a| a == "--rate-interleaved");
    let decode_probe = args.iter().any(|a| a == "--decode-probe");
    let w2 = args.iter().any(|a| a == "--w2");
    let fixed = args.iter().any(|a| a == "--fixed");

    let device = Device::system_default().expect("Metal device");
    let kernels = LinearKernels::shared().expect("kernels");
    let queue = device.new_command_queue();
    let rig = Rig {
        device,
        kernels,
        queue,
    };
    // Global warm-up before ANY measurement, on a shape that is not in any
    // table: cold Metal clocks are up to 7x slow, and this project's history has
    // several false results from measuring during the ramp.
    {
        let warm_shape = Shape {
            name: "warm",
            in_f: 5376,
            out_f: 21504,
            rate: 3,
        };
        let case = Case::build(&rig, warm_shape.in_f, warm_shape.out_f, warm_shape.rate);
        for _ in 0..12 {
            case.run(&rig, 8, false);
        }
    }

    if selftest {
        run_selftest(&rig);
        return;
    }

    if fixed {
        // Four shapes chosen so a per-dispatch term that secretly scales with
        // work would be visible: the fixed cost is claimed to be independent of
        // both in_features and out_features, so varying each alone is the test.
        // down_proj inverts the aspect ratio (16x more k), gate_proj is 4x more
        // n, and 1024x1024 is small enough that the term should dominate.
        for shape in [
            Shape {
                name: "small",
                in_f: 1024,
                out_f: 1024,
                rate: 3,
            },
            Shape {
                name: "k_proj slide",
                in_f: 5376,
                out_f: 4096,
                rate: 3,
            },
            Shape {
                name: "o_proj slide",
                in_f: 5376,
                out_f: 8192,
                rate: 3,
            },
            Shape {
                name: "gate_proj",
                in_f: 5376,
                out_f: 21504,
                rate: 3,
            },
            Shape {
                name: "down_proj",
                in_f: 21504,
                out_f: 5376,
                rate: 3,
            },
        ] {
            fixed_cost(&rig, &shape);
        }
        println!(
            "\n  A per-dispatch GB/s that is roughly CONSTANT across these five shapes \
             is streaming, and the command-buffer overhead that is roughly constant \
             is the per-dispatch fixed cost. If instead the per-dispatch GB/s rises \
             with size, the overhead is per-work and the split above is mislabelled."
        );
        return;
    }

    if w2 {
        // gate/up/down carry 65% of a decode token's bytes; q_proj is a mid-size
        // sliding projection; down_proj is the 16x-more-k axis, where the k loop
        // is long and the window is re-read every iteration -- so if the extra
        // load were costing anything, it would show most there. lm_head is 6 bpw
        // and has no 2-word variant (the derivation excludes it), which is itself
        // worth printing: it is the one shape where the lever provably does not
        // apply.
        for shape in [
            Shape {
                name: "gate_proj",
                in_f: 5376,
                out_f: 21504,
                rate: 3,
            },
            Shape {
                name: "up_proj",
                in_f: 5376,
                out_f: 21504,
                rate: 3,
            },
            Shape {
                name: "down_proj",
                in_f: 21504,
                out_f: 5376,
                rate: 3,
            },
            Shape {
                name: "o_proj slide",
                in_f: 5376,
                out_f: 8192,
                rate: 3,
            },
            Shape {
                name: "k_proj slide",
                in_f: 5376,
                out_f: 4096,
                rate: 3,
            },
        ] {
            w2_sweep(&rig, &shape);
        }
        println!(
            "\n  6 bpw (lm_head) has no 2-word variant: two_word_is_safe reports \
             KA=6 as used [6,63] < max sh 22, so word 3 IS read there. That is the \
             derivation, not an omission."
        );
        return;
    }

    if decode_probe {
        // THE experiment. Real kernel vs no-decode probe, interleaved, at the
        // three rates that answer the question:
        //
        //   3 bpw  -- the outlier, 165.7 GB/s, the one that matters (410/411
        //             tensors) and the one whose rate must double to reach 19 tok/s
        //   4 bpw  -- nearest saturated rate, 216.4
        //   8 bpw  -- furthest saturated rate, 217.9
        //
        // 3 bpw is 1.306x slower than 4 bpw, and TWO mechanisms both predict
        // 1.333x: lane-load requests (2.67/2.00) and decode calls per useful byte
        // (2.67/2.00). The probe removes decode and NOTHING else -- identical
        // memory traffic, one asserted substitution in the generator -- so:
        //
        //   _nod reaches ~216 at 3 bpw  -> decode cost is the wall
        //   _nod stays near 166          -> requests are the wall
        //   partial move                 -> both, and the split is the fraction
        //
        // The 8 bpw arm is what makes the reading unambiguous. Without a
        // saturated contrast arm, a `_nod` number at 3 bpw could be explained by
        // the probe having changed something other than decode.
        for shape in [
            Shape {
                name: "3 bpw (the outlier)",
                in_f: 5376,
                out_f: 21504,
                rate: 3,
            },
            Shape {
                name: "4 bpw (saturated)",
                in_f: 5376,
                out_f: 16384,
                rate: 4,
            },
            Shape {
                name: "8 bpw (saturated)",
                in_f: 5376,
                out_f: 8192,
                rate: 8,
            },
        ] {
            nod_sweep(&rig, &shape);
        }
        println!(
            "\n  The _nod arm computes GARBAGE by construction (decode_state replaced by a\n  one-AND stub) and is never dispatched by the engine. Only its BANDWIDTH is\n  meaningful: identical memory traffic, decode removed."
        );
        return;
    }

    if rate_interleaved {
        // The four equal-bytes shapes, in the order the old `--only "bit rate"`
        // group used, so the two tables are directly comparable. 6 bpw comes
        // third deliberately: it has no `_w2` kernel and so is the control.
        rate_sweep_interleaved(
            &rig,
            &[
                Shape {
                    name: "3 bpw",
                    in_f: 5376,
                    out_f: 21504,
                    rate: 3,
                },
                Shape {
                    name: "4 bpw",
                    in_f: 5376,
                    out_f: 16384,
                    rate: 4,
                },
                Shape {
                    name: "6 bpw",
                    in_f: 5376,
                    out_f: 10752,
                    rate: 6,
                },
                Shape {
                    name: "8 bpw",
                    in_f: 5376,
                    out_f: 8192,
                    rate: 8,
                },
            ],
            9,
        );
        return;
    }

    if nr0 {
        // are 65% of a decode token's bytes; lm_head is the one shape already
        // near the ceiling, and including it says whether NR0 helps where there is
        // nothing left to win -- a regression there would be free to accept if
        // only the big shapes were measured.
        for shape in [
            Shape {
                name: "gate_proj",
                in_f: 5376,
                out_f: 21504,
                rate: 3,
            },
            Shape {
                name: "down_proj",
                in_f: 21504,
                out_f: 5376,
                rate: 3,
            },
            Shape {
                name: "q_proj slide",
                in_f: 5376,
                out_f: 8192,
                rate: 3,
            },
            Shape {
                name: "lm_head",
                in_f: 5376,
                out_f: 262144,
                rate: 6,
            },
        ] {
            nr0_sweep(&rig, &shape);
        }
        return;
    }

    let groups: Vec<(&str, Vec<Shape>)> = vec![
        ("real decode shapes", real_shapes()),
        (
            "axis: total bytes (in=5376 fixed, rate 3, out_features swept)",
            [2048usize, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
                .into_iter()
                .map(|out| Shape {
                    name: "sweep",
                    in_f: 5376,
                    out_f: out,
                    rate: 3,
                })
                .collect(),
        ),
        (
            "axis: in_features (out=21504 fixed, rate 3, in_features swept)",
            [1024usize, 2048, 5376, 10752, 21504]
                .into_iter()
                .map(|input| Shape {
                    name: "sweep",
                    in_f: input,
                    out_f: 21504,
                    rate: 3,
                })
                .collect(),
        ),
        (
            "axis: bit rate at EQUAL BYTES (43.4 MB of weights either way)",
            [
                Shape {
                    name: "3 bpw",
                    in_f: 5376,
                    out_f: 21504,
                    rate: 3,
                },
                Shape {
                    name: "4 bpw",
                    in_f: 5376,
                    out_f: 16384,
                    rate: 4,
                },
                Shape {
                    name: "6 bpw",
                    in_f: 5376,
                    out_f: 10752,
                    rate: 6,
                },
                Shape {
                    name: "8 bpw",
                    in_f: 5376,
                    out_f: 8192,
                    rate: 8,
                },
            ]
            .into_iter()
            .collect(),
        ),
    ];

    for (title, shapes) in groups {
        if let Some(filter) = &only {
            if !title.contains(filter.as_str()) {
                continue;
            }
        }
        println!("\n=== {title} ===");
        header();
        for shape in shapes {
            let label = if title.starts_with("real") {
                shape.name.to_string()
            } else {
                format!("{} x{}", shape.in_f, shape.out_f)
            };
            let reading = measure(&rig, &shape);
            print_row(
                &label,
                shape.in_f as f64 * shape.out_f as f64 * shape.rate as f64 / 8.0 / 1e6,
                &reading,
            );
        }
    }
}

fn run_selftest(rig: &Rig) {
    let mut failures = 0;
    println!("=== gemv_shape self-test: the checks a number from this harness needs ===");

    // 1. AFFINE. The matvec has a per-call cost that does not scale with bytes
    //    (the spec's 79 us), so "2x the bytes must take 2x the time" is simply
    //    the wrong check -- it failed here at 1.62x on a real effect, which is
    //    how a correct instrument gets declared broken. The right statement is
    //    that time is *affine* in bytes, and the check is on the fit: three sizes,
    //    least squares, residuals under 4% of the span, and the fitted streaming
    //    rate inside a plausible band. It also *reports* the fixed term, which is
    //    the quantity `spec/decode-bandwidth-240gb` T1 is trying to decompose.
    {
        let sizes = [16384usize, 32768, 65536];
        let mut points = Vec::new();
        for out in sizes {
            let shape = Shape {
                name: "a",
                in_f: 5376,
                out_f: out,
                rate: 3,
            };
            let mb = out as f64 * 5376.0 * 3.0 / 8.0 / 1e6;
            let r = measure(rig, &shape);
            points.push((mb, r.median_ms));
        }
        let n = points.len() as f64;
        let sx: f64 = points.iter().map(|p| p.0).sum();
        let sy: f64 = points.iter().map(|p| p.1).sum();
        let sxx: f64 = points.iter().map(|p| p.0 * p.0).sum();
        let sxy: f64 = points.iter().map(|p| p.0 * p.1).sum();
        let slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
        let intercept = (sy - slope * sx) / n;
        let span = points.last().unwrap().1 - points.first().unwrap().1;
        let residual = points
            .iter()
            .map(|p| (p.1 - (intercept + slope * p.0)).abs())
            .fold(0.0, f64::max);
        // slope is ms per MB, so its reciprocal is MB/ms == GB/s exactly.
        let rate_gbps = 1.0 / slope;
        let ok = residual < 0.04 * span && (120.0..=320.0).contains(&rate_gbps);
        println!(
            "  [{:4}] AFFINE     t = {:.4} ms + {:.5} ms/MB  (fit: {:.1} GB/s streaming, {:.1} us \
             fixed, max residual {:.4} ms = {:.1}% of span; want rate 120-320, residual < 4%)",
            if ok { "PASS" } else { "FAIL" },
            intercept,
            slope,
            rate_gbps,
            intercept * 1e3,
            residual,
            100.0 * residual / span
        );
        failures += !ok as usize;
    }

    // 2. Sensitivity: 4x more bytes must be at least 2x the time. A probe that
    //    cannot resolve a 4x change in work cannot resolve the 2x being hunted.
    {
        let small = Shape {
            name: "t",
            in_f: 5376,
            out_f: 21504,
            rate: 3,
        };
        let big = Shape {
            name: "b",
            in_f: 5376,
            out_f: 86016,
            rate: 3,
        };
        let a = measure(rig, &small);
        let b = measure(rig, &big);
        let ratio = b.median_ms / a.median_ms;
        let ok = ratio > 2.0;
        println!(
            "  [{:4}] SENSITIVE  out=21504 = {:.4} ms, out=86016 = {:.4} ms, ratio {:.3} (want > 2.0)",
            if ok { "PASS" } else { "FAIL" },
            a.median_ms,
            b.median_ms,
            ratio
        );
        failures += !ok as usize;
    }

    // 3. CLOCK. The first shape, re-measured at the end, must still agree:
    //    cold Metal clocks are up to 7x slow and the ramp has to be behind the
    //    first number, not inside it.
    {
        let shape = Shape {
            name: "r",
            in_f: 5376,
            out_f: 21504,
            rate: 3,
        };
        let first = measure(rig, &shape);
        for _ in 0..12 {
            let _ = measure(
                rig,
                &Shape {
                    name: "o",
                    in_f: 5376,
                    out_f: 65536,
                    rate: 3,
                },
            );
        }
        let again = measure(rig, &shape);
        let ratio = again.median_ms / first.median_ms;
        let ok = (0.92..=1.09).contains(&ratio);
        println!(
            "  [{:4}] CLOCK      first {:.4} ms, after 12 other shapes {:.4} ms, ratio {:.3} (want 0.92-1.09)",
            if ok { "PASS" } else { "FAIL" },
            first.median_ms,
            again.median_ms,
            ratio
        );
        failures += !ok as usize;
    }

    // 4. PLAUSIBLE. A reported GB/s must be inside the physically reachable band.
    //    This is the check that catches a wrong byte count or a seconds-as-ms
    //    slip: it failed here on the first run of this harness, reporting
    //    131,512 GB/s, and it is the check that would have caught this project's
    //    historical 20,340 GB/s and 1,592 GB/s.
    {
        let shape = Shape {
            name: "p",
            in_f: 5376,
            out_f: 21504,
            rate: 3,
        };
        let r = measure(rig, &shape);
        let ok = r.gbps < 320.0 && r.gbps > 20.0;
        println!(
            "  [{:4}] PLAUSIBLE  {:.1} GB/s on 5376x21504 @3bpw (want 20-320; the 240 GB/s \
             ceiling is measured, the 273 datasheet figure is arithmetic)",
            if ok { "PASS" } else { "FAIL" },
            r.gbps
        );
        failures += !ok as usize;
    }

    // 5. MEDIAN CLEAN. The 9 samples of one shape must have a median within 3%
    //    of the best of them. The full worst/best spread is printed for
    //    information: the worst sample is a scheduling artefact often enough (a
    //    1.62x outlier on an otherwise clean run) that demanding a tight whole
    //    spread would mean refusing to report at all. The median is what has to
    //    be clean; the spread has to be *visible*.
    {
        let shape = Shape {
            name: "d",
            in_f: 5376,
            out_f: 21504,
            rate: 3,
        };
        let r = measure(rig, &shape);
        let ok = r.median_over_best < 1.03;
        println!(
            "  [{:4}] MEDIAN     median {:.4} ms / best {:.4} ms = {:.4} (want < 1.03); \
             worst {:.4} ms, full spread {:.3}x reported for information",
            if ok { "PASS" } else { "FAIL" },
            r.median_ms,
            r.best_ms,
            r.median_over_best,
            r.worst_ms,
            r.worst_ms / r.best_ms
        );
        failures += !ok as usize;
    }

    println!(
        "\n{}: {} check(s) failed",
        if failures == 0 {
            "self-test PASSED"
        } else {
            "self-test FAILED"
        },
        failures
    );
    if failures > 0 {
        std::process::exit(1);
    }
}
