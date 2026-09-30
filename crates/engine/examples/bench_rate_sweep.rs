//! Is the 3 bpw colmajor GEMV's ~188 GB/s the EXL3 *rate*, or a kernel defect?
//!
//! # The question
//!
//! `spec/decode-19tok-repacked-250gb` measured `gate_up_gelu` at **187.7 GB/s**
//! and fitted it, to 0.1%, with
//!
//! ```text
//! predicted 3 bpw ceiling = 250.5 x 2.00 / 2.667 = 187.9 GB/s
//! ```
//!
//! where 2.667 and 2.00 are *lane-load amplifications*: on the colmajor
//! `_contig4` path each of 32 lanes loads two 32-bit words = 8 B to consume
//! `8 * KA` bits of tile, so amplification is `8 / KA` -- 2.667 at 3 bpw, 2.000
//! at 4 bpw, 1.333 at 6 bpw. A good fit is not a mechanism. The competing
//! explanation is a kernel defect that would cost the same factor at every rate,
//! and both predict one number for 3 bpw.
//!
//! They can only be separated by moving the rate and holding everything else
//! fixed. That is what this does, and it is why the 4 bpw and 6 bpw kernels
//! exist (`linear_gemv_ishizuki_k{4,6}_colmajor_contig4_bench`): they are
//! `linear_gemv_ishizuki_k3_colmajor_contig4` with `KA` changed, character for
//! character otherwise. Nothing in the engine can dispatch them; see
//! `crates/engine/tests/rate4_colmajor_bit_exact.rs`.
//!
//! | outcome | reading |
//! |---|---|
//! | 4 bpw ~220-250, 6 bpw ~250 | the rate is the constraint; the kernel is sound; 187.7 is a format property |
//! | all three near 187.7 | rate-independent defect; the amplification fit is coincidence |
//! | in between, varying by shape | a shape-dependent ratio localises the defect |
//!
//! # Why a separate instrument from `bench_residency`
//!
//! `bench_residency` answers a different question -- whether the rate survives
//! file-backed weights -- and it measures one shape at a time to completion. That
//! is exactly the arrangement `spec/decode-bandwidth-240gb` retracted: its
//! within-sweep spreads were 2.2x-3.7x and its two control rows (6 and 8 bpw,
//! which cannot have moved) moved +10.6% and -7.9% in *opposite* directions, so
//! it could not resolve a 10% effect. A drift-proof ratio needs the arms
//! adjacent in time.
//!
//! So every round here measures **every** arm of **every** shape, and the order
//! is reversed on alternate rounds so neither rate systematically owns the start
//! of a round. The 3 bpw arm is not a control bolted on afterwards -- it is the
//! production kernel, re-measured in the same run as the new ones, so the
//! headline ratio cannot be an artefact of drift between two sessions.
//!
//! # Accounting
//!
//! Bandwidth is always against **useful weight bytes** -- the packed checkpoint
//! bytes, `16 * bits` u16 per 16x16 tile -- never against bytes requested.
//! Requested bytes are printed separately as `amp`, because `amp` is the thing
//! under test: if the kernel were defective, GB/s against useful bytes would be
//! flat across rates while `amp` fell, and the table has to be able to see that.
//!
//! Nothing here runs a model, and no weight file is read: every arm's weights
//! are synthesised by `build_rowmajor`, a fixed-seed LCG, so two runs measure
//! the same bytes. `--check-determinism` prints an FNV-1a digest per arm; run
//! it twice and the digests must match (`spec` AC6's "generator twice -> clean
//! diff" in the form that applies to a measurement harness, whose output is
//! data rather than source).
//!
//! Usage:
//! ```text
//! bench_rate_sweep [--rounds N] [--shapes a,b] [--rates 3,4,6] [--round-ms MS]
//! bench_rate_sweep --check-determinism
//! ```

use half::f16;
use kaplumba_engine::bench::{Shape, SHAPES};
use kaplumba_engine::exl3::pack_integer_states;
use kaplumba_kernels::encoder::MetalEncoder;
use kaplumba_kernels::exl3::{Exl3Codebook, Exl3Rate};
use kaplumba_kernels::linear::{LinearEncodeRequest, LinearKernels};
use kaplumba_engine::bench::summarize;
use metal::{Buffer, Device, MTLResourceOptions};
use std::time::Instant;

/// Target wall time for one (arm, round) sample. Long enough that per-dispatch
/// encoding is not a measurable fraction, short enough that 22 arms x 9 rounds
/// stays inside a few minutes.
const ROUND_TARGET_MS: f64 = 40.0;
const MIN_ITERS: u64 = 3;
const MAX_ITERS: u64 = 512;

/// Fixed seed base. Constant, not a clock: the whole point is that two runs
/// measure the same bytes, so that a difference between them is the machine and
/// not the data.
const SEED_BASE: u32 = 0x5EED_1234;

/// `linear_gemv_ishizuki_k3_colmajor_contig4` is the kernel the engine
/// dispatches for every 3 bpw colmajor layer linear (`KAPLUMBA_TRACE_GEMV=1`,
/// `model.rs` sets `perm_mode = 1` on the seven per-layer linears). The other
/// two are that body at KA = 4 and KA = 6, measurement-only.
const KERNEL_3: &str = "linear_gemv_ishizuki_k3_colmajor_contig4";
const KERNEL_4: &str = "linear_gemv_ishizuki_k4_colmajor_contig4_bench";
const KERNEL_6: &str = "linear_gemv_ishizuki_k6_colmajor_contig4_bench";

fn kernel_for(rate: u8) -> &'static str {
    match rate {
        3 => KERNEL_3,
        4 => KERNEL_4,
        6 => KERNEL_6,
        other => panic!("no colmajor contig4 kernel exists at {other} bpw"),
    }
}

/// 32-bit words each lane loads per tile, on the colmajor `_contig4` body.
///
/// **Two** where the two-word window is legal, **three** where it is not. The
/// boundary is `max_lane sh <= 48 - 7*KA` (`two_word_is_safe` in
/// `scripts/gen_exl3_gemv.py`; measured at every rate in
/// `tests/rate4_colmajor_bit_exact.rs`), which holds for KA 1..=4 -- and 3 and 4
/// sit exactly on it. KA = 6 does not hold, so the 6 bpw kernel loads a third
/// word and `..._k4_...` would have decoded six bits of zeros per lane as
/// weight.
fn words_per_lane(rate: u8) -> f64 {
    if rate <= 4 { 2.0 } else { 3.0 }
}

/// Lane-load amplification on this path: 32 lanes x `words_per_lane` words x 4 B
/// requested per tile, against `32 * KA` bytes of tile. So
///
/// ```text
/// rate   words/lane   amplification
/// 3 bpw      2            2.667     <- the engine's decode path
/// 4 bpw      2            2.000
/// 6 bpw      3            2.000
/// ```
///
/// Note that 4 bpw and 6 bpw are at the *same* amplification. The 6 bpw arm is
/// therefore not a lower-amplification point; it is a same-amplification /
/// different-rate control, and it is the one that separates "the requests are the
/// limit" from "the rate is the limit".
///
/// Derived in `linear_repacked.metal` at `..._k4_colmajor_contig4_bench`; printed
/// here so the table states the prediction it is testing rather than leaving it in
/// a comment.
fn amplification(rate: u8) -> f64 {
    words_per_lane(rate) * 4.0 / f64::from(rate)
}

/// 32-bit words of packed trellis per 16x16 tile: `256 * KA / 32`.
fn words_per_tile(rate: u8) -> usize {
    8 * rate as usize
}

/// Transpose the tile grid, for any integer rate.
///
/// `exl3::reorder_tensor_to_colmajor` hardcodes 48 u16 per tile and therefore
/// only accepts 3 bpw, which is precisely why a 4 bpw colmajor arm needs its
/// own repack. The transposition is rate-agnostic -- tile `(tk, tn)` moves from
/// row-major offset `(tk * tiles_n + tn) * W` to colmajor offset
/// `(tn * tiles_k + tk) * W`, with a tile's `W` words copied verbatim -- and
/// `tests/rate4_colmajor_bit_exact.rs` asserts this agrees with the production
/// function exactly at 3 bpw, so the two arms are not different layouts.
fn reorder_to_colmajor(packed: &[u16], in_features: usize, out_features: usize, rate: u8) -> Vec<u16> {
    let tiles_k = in_features / 16;
    let tiles_n = out_features / 16;
    let w = words_per_tile(rate) * 2;
    assert_eq!(
        packed.len(),
        tiles_k * tiles_n * w,
        "packed must be {tiles_k}x{tiles_n} tiles of {w} u16"
    );
    let mut out = vec![0u16; packed.len()];
    for tk in 0..tiles_k {
        for tn in 0..tiles_n {
            let src = (tk * tiles_n + tn) * w;
            let dst = (tn * tiles_k + tk) * w;
            out[dst..dst + w].copy_from_slice(&packed[src..src + w]);
        }
    }
    out
}

/// Row-major packed tiles, then colmajor. A pure function of
/// `(in_features, out_features, rate, seed)`.
///
/// Splitmix64 over `(tile, code)`, not a running LCG. The LCG that the other
/// harnesses in this directory use has a period-16 low-3-bits cycle and a tile
/// is 256 draws, so every tile comes out byte-identical and the whole tensor is
/// one tile repeated -- which makes a colmajor repack the identity and would
/// leave the layout half of the A/B untested. Bandwidth would be unaffected
/// (same bytes, same addresses); correctness-of-intent would not.
///
/// The same seed is used for every rate, so the 3/4/6 bpw arms are the same
/// matrix at three rates, which is what "the same matrix" has to mean for the
/// ratio to mean anything.
fn build_colmajor(shape: &Shape, rate: u8, seed: u32) -> Vec<u16> {
    let tiles = (shape.in_features / 16) * (shape.out_features / 16);
    let mask = (1u32 << rate) - 1;
    let mut row = Vec::with_capacity(tiles * 16 * rate as usize);
    for tile in 0..tiles {
        let mut states = [0u16; 256];
        for (code, slot) in states.iter_mut().enumerate() {
            let mut z = u64::from(seed)
                ^ (tile as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ (code as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            *slot = (z as u32 & mask) as u16;
        }
        row.extend(pack_integer_states(&states, rate).expect("pack").0);
    }
    reorder_to_colmajor(&row, shape.in_features, shape.out_features, rate)
}

/// FNV-1a over the packed words. `--check-determinism` prints this per arm so
/// "the generator is deterministic" is observable rather than promised.
fn digest(words: &[u16]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for w in words {
        h ^= u64::from(*w);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Packed weight bytes for `shape` at `rate`. This is the denominator of every
/// bandwidth number below and it is deliberately the *checkpoint* size, not
/// `16 * bits * 2` times something else: `Shape::weight_bytes()` at 3 bpw equals
/// the checkpoint's own `n_bytes` for all eleven tensors, which
/// `bench.rs::shapes_match_the_checkpoint_byte_counts` enforces.
fn useful_bytes(shape: &Shape, rate: u8) -> u64 {
    let tiles = (shape.in_features / 16) * (shape.out_features / 16);
    (tiles * 16 * rate as usize * 2) as u64
}

struct Ctx {
    queue: metal::CommandQueue,
    input: Buffer,
    input_signs: Buffer,
    output_signs: Buffer,
    transformed: Buffer,
    raw: Buffer,
    output: Buffer,
    in_features: usize,
    out_features: usize,
    rate: u8,
}

fn upload_words(device: &Device, v: &[u16]) -> Buffer {
    device.new_buffer_with_data(
        v.as_ptr() as *const std::ffi::c_void,
        (v.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn upload_f16(device: &Device, v: &[f16]) -> Buffer {
    device.new_buffer_with_data(
        v.as_ptr() as *const std::ffi::c_void,
        (v.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn make_ctx(device: &Device, queue: &metal::CommandQueue, shape: &Shape, rate: u8) -> Ctx {
    let input: Vec<f16> = (0..shape.in_features)
        .map(|i| f16::from_f32((i as f32 * 0.017).sin()))
        .collect();
    // Signs matter for correctness, not for bandwidth, but they are set to
    // something other than 1.0 so a harness that silently skipped the input
    // transform would still be measuring the same amount of work.
    let input_signs: Vec<f16> = (0..shape.in_features)
        .map(|i| f16::from_f32(if i % 3 == 0 { 0.0021 } else { -0.0017 }))
        .collect();
    let output_signs: Vec<f16> = (0..shape.out_features)
        .map(|i| f16::from_f32(if i % 5 == 0 { -0.0031 } else { 0.0023 }))
        .collect();
    Ctx {
        queue: queue.clone(),
        input: upload_f16(device, &input),
        input_signs: upload_f16(device, &input_signs),
        output_signs: upload_f16(device, &output_signs),
        transformed: device.new_buffer(
            (shape.in_features * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        ),
        // Matches the engine's scratch: `out_features * 4 * 4`.
        raw: device.new_buffer(
            (shape.out_features * 4 * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ),
        output: device.new_buffer(
            (shape.out_features * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        ),
        in_features: shape.in_features,
        out_features: shape.out_features,
        rate,
    }
}

/// Seconds per dispatch of `kernel` over `packed`, over `iters` dispatches in one
/// command buffer.
///
/// One encoder, `iters` dispatches, `commit_and_wait`, divide -- the same shape
/// as `bench_residency`, so the two instruments are comparable. Encoding is on
/// the critical path here, which is why `iters` is calibrated per arm to reach
/// `ROUND_TARGET_MS` rather than fixed: at 3 bpw a 16 MB shape dispatches in
/// ~70 us and at 6 bpw `lm_head` takes ~4 ms, a 60x range where a fixed `iters`
/// would make the small arms nothing but encoder overhead.
fn seconds_per_dispatch(
    kernels: &LinearKernels,
    ctx: &Ctx,
    kernel: &str,
    packed: &Buffer,
    iters: u64,
) -> f64 {
    let mut encoder = MetalEncoder::new(&ctx.queue).expect("encoder");
    let started = Instant::now();
    for _ in 0..iters {
        kernels
            .encode_gemv_named(
                &mut encoder,
                kernel,
                &LinearEncodeRequest {
                    packed,
                    packed_offset: 0,
                    perm_mode: 1,
                    input: &ctx.input,
                    input_signs: &ctx.input_signs,
                    output_signs: &ctx.output_signs,
                    transformed: &ctx.transformed,
                    raw: &ctx.raw,
                    output: &ctx.output,
                    in_features: ctx.in_features,
                    out_features: ctx.out_features,
                    rate: Exl3Rate::Integer(ctx.rate),
                    codebook: Exl3Codebook::Mcg,
                    pretransformed: false,
                },
            )
            .expect("encode");
    }
    encoder.commit_and_wait().expect("wait");
    started.elapsed().as_secs_f64() / iters as f64
}

struct Arm {
    shape_idx: usize,
    rate: u8,
    kernel: &'static str,
    bytes: u64,
    packed: Buffer,
    ctx: Ctx,
    iters: u64,
    /// GB/s per round. Higher is better, so `summarize`'s `worst/best` spread is
    /// "lower is more repeatable", which is the convention `bench.rs` fixes.
    samples: Vec<f64>,
}


/// `ru_maxrss`, which on Darwin counts resident pages only and misses the Metal
/// buffer pages. AGENTS.md 10.3 rejects it as a memory figure, and
/// `peak_resident_pages_bytes` in the engine's `--metrics` is the same number
/// under a name that once made a 93%-of-budget situation look like 57%.
///
/// Printed anyway, and labelled, because the number that matters for *this* tool
/// is whether it disturbed anybody: allocating GiBs of synthesised weights while
/// another agent had a model loaded is exactly the AGENTS.md 10.1 failure, and the
/// only way a reader of these numbers can tell is if the tool says how much it was
/// holding.
fn max_rss_bytes() -> u64 {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe {
        libc::getrusage(libc::RUSAGE_SELF, &mut usage);
    }
    usage.ru_maxrss as u64
}

/// `(rounds, shape name prefixes, rates, target round ms, determinism only)`.
fn parse_args() -> (usize, Vec<String>, Vec<u8>, f64, bool) {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut rounds = 9usize;
    let mut shapes: Vec<String> = Vec::new();
    let mut rates: Vec<u8> = vec![3, 4];
    let mut round_ms = ROUND_TARGET_MS;
    let mut determinism = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--rounds" => {
                i += 1;
                rounds = args[i].parse().expect("--rounds N");
            }
            "--shapes" => {
                i += 1;
                shapes = args[i].split(',').map(str::to_string).collect();
            }
            "--rates" => {
                i += 1;
                rates = args[i]
                    .split(',')
                    .map(|s| s.parse().expect("--rates 3,4,6"))
                    .collect();
            }
            "--round-ms" => {
                i += 1;
                round_ms = args[i].parse().expect("--round-ms MS");
            }
            "--check-determinism" => determinism = true,
            "--quiet" => {} // consumed below; harmless in the positional walk
            other => panic!("unknown flag {other}"),
        }
        i += 1;
    }
    (rounds, shapes, rates, round_ms, determinism)
}

/// One (shape, rate) measurement, retained across chunks so the final table and
/// aggregate cover every shape even though the buffers did not all exist at once.
struct Result_ {
    shape_idx: usize,
    rate: u8,
    bytes: u64,
    samples: Vec<f64>,
}

/// Build, warm up and measure every rate of one shape. All rates of a shape are
/// resident simultaneously and interleaved within every round, which is what the
/// ratio needs; shapes are handled one chunk at a time so peak memory is a
/// fraction of the full sweep.
///
/// Chunking by shape rather than by arm is deliberate. The quantity under test is
/// a *ratio between rates of the same shape*, so drift has to be controlled
/// within a shape, and it is. Drift *between* shapes cannot contaminate a
/// per-shape ratio because each shape's rates are measured adjacent in time.
#[allow(clippy::too_many_lines)]
fn run_shape(
    device: &Device,
    queue: &metal::CommandQueue,
    kernels: &LinearKernels,
    si: usize,
    rates: &[u8],
    rounds: usize,
    round_ms: f64,
    verbose: bool,
) -> Vec<Result_> {
    let shape = &SHAPES[si];
    let mut out: Vec<Result_> = Vec::with_capacity(rates.len());
    let mut arms: Vec<Arm> = Vec::new();

    for &rate in rates {
        let seed = SEED_BASE.wrapping_add(si as u32 * 131).wrapping_add(rate as u32);
        let words = build_colmajor(shape, rate, seed);
        let bytes = useful_bytes(shape, rate);
        let kernel = kernel_for(rate);
        let packed = upload_words(device, &words);
        drop(words);
        let ctx = make_ctx(device, queue, shape, rate);
        // Calibrate off the timed path: the first sample runs cold and short, and
        // an `iters` chosen from a cold measurement is the kind of thing that
        // quietly halves a run's resolution.
        let secs = seconds_per_dispatch(kernels, &ctx, kernel, &packed, MIN_ITERS);
        let iters = (((round_ms * 1e-3) / secs).round() as u64).clamp(MIN_ITERS, MAX_ITERS);
        let secs = seconds_per_dispatch(kernels, &ctx, kernel, &packed, iters);
        let iters = ((round_ms * 1e-3) / secs).round() as u64;
        // One more untimed pass at the chosen count, so the first *timed* round is
        // not the one that pays for the ramp.
        let _ = seconds_per_dispatch(kernels, &ctx, kernel, &packed, iters);
        arms.push(Arm {
            shape_idx: si,
            rate,
            kernel,
            bytes,
            packed,
            ctx,
            iters: iters.clamp(MIN_ITERS, MAX_ITERS),
            samples: Vec::with_capacity(rounds),
        });
    }

    for round in 0..rounds {
        let order: Vec<usize> = if round % 2 == 0 {
            (0..arms.len()).collect()
        } else {
            (0..arms.len()).rev().collect()
        };
        for ai in order {
            let arm = &mut arms[ai];
            let secs = seconds_per_dispatch(kernels, &arm.ctx, arm.kernel, &arm.packed, arm.iters);
            arm.samples.push(arm.bytes as f64 / secs / 1e9);
        }
        if verbose {
            let line: Vec<String> = arms
                .iter()
                .map(|a| {
                    let s = summarize(&a.samples).expect("at least one sample");
                    format!("{}bpw {:.1}", a.rate, s.median)
                })
                .collect();
            println!("  {} round {round}: {}", shape.name, line.join("  "));
        }
    }

    for arm in arms {
        out.push(Result_ {
            shape_idx: arm.shape_idx,
            rate: arm.rate,
            bytes: arm.bytes,
            samples: arm.samples,
        });
    }
    out
}

fn main() {
    let (rounds, wanted, mut rates, round_ms, determinism) = parse_args();

    let shape_idxs: Vec<usize> = (0..SHAPES.len())
        .filter(|i| {
            wanted.is_empty() || wanted.iter().any(|w| SHAPES[*i].name.starts_with(w.as_str()))
        })
        .collect();
    assert!(
        !shape_idxs.is_empty(),
        "no shape matched {wanted:?}; `bench::SHAPES` has {:?}",
        SHAPES.iter().map(|s| s.name).collect::<Vec<_>>()
    );
    rates.sort_unstable();

    println!("=== bench_rate_sweep: colmajor _contig4, rate 3 vs 4 (and 6) ===");
    println!(
        "{} shapes x {} rates, {rounds} interleaved rounds of ~{round_ms:.0} ms per \
         (shape, rate), arm order reversed on alternate rounds.\n\
         Bytes are USEFUL WEIGHT BYTES (the checkpoint's own packed size). `amp` is \
         lane-load amplification = 8/KA at 3 and 4 bpw (2 words/lane) and 12/KA at \
         6 bpw (3 words/lane -- see linear_repacked.metal).\n\
         Spread is worst/best, so LOWER is more repeatable. Pass --quiet to drop the \
         per-round lines.\n",
        shape_idxs.len(),
        rates.len(),
    );

    if determinism {
        // No Metal device, no pipeline, no buffer. This mode hashes the generator's
        // output and must be runnable while the GPU is busy.
        for &si in &shape_idxs {
            for &rate in &rates {
                let seed = SEED_BASE.wrapping_add(si as u32 * 131).wrapping_add(rate as u32);
                let words = build_colmajor(&SHAPES[si], rate, seed);
                println!(
                    "digest {:>12}  {:<13} @{}bpw  {:>10} B  {} words/tile  kernel {}",
                    digest(&words),
                    SHAPES[si].name,
                    rate,
                    words.len() as u64 * 2,
                    words_per_tile(rate),
                    kernel_for(rate),
                );
            }
        }
        println!(
            "\nre-run and every digest must be identical. No file read, no Metal \
             buffer allocated, no GPU touched."
        );
        return;
    }

    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue().to_owned();
    let kernels = LinearKernels::new().expect("LinearKernels");

    // Every rate must have a kernel, checked before any buffer is allocated. A
    // benchmark that silently skips a rate it was asked for is worse than one that
    // refuses.
    for rate in &rates {
        let kernel = kernel_for(*rate);
        assert!(
            kernels.ishizuki_gemv_pipeline_by_name(kernel).is_some(),
            "{kernel} is not in the library. The 3 bpw arm is the engine's live \
             decode kernel; the 4/6 bpw arms are measurement kernels added by this \
             spec. If one is missing, this harness would measure 3 bpw and print a \
             ratio against nothing."
        );
        assert_eq!(
            LinearKernels::ishizuki_rate_of_name(kernel),
            Some(*rate),
            "{kernel} does not name a {rate} bpw rate; `encode_gemv_named` rejects a \
             name/rate mismatch, so every sample would be an error"
        );
    }

    let verbose = !std::env::args().any(|a| a == "--quiet");
    let mut results: Vec<Result_> = Vec::new();
    for &si in &shape_idxs {
        println!("--- {} {}x{} ---", SHAPES[si].name, SHAPES[si].in_features, SHAPES[si].out_features);
        results.extend(run_shape(
            &device, &queue, &kernels, si, &rates, rounds, round_ms, verbose,
        ));
    }

    // --- report
    println!("\n=== per shape: medians of {rounds}, GB/s of USEFUL weight bytes ===");
    println!(
        "{:<13} {:>4} {:>8} {:>8} {:>8} {:>6}  {}",
        "shape", "bpw", "MB", "median", "spread", "amp", "ratio vs ref"
    );
    let get = |si: usize, rate: u8| -> Option<kaplumba_engine::bench::Summary> {
        results
            .iter()
            .find(|r| r.shape_idx == si && r.rate == rate)
            .and_then(|r| summarize(&r.samples))
    };
    for &si in &shape_idxs {
        let base = get(si, rates[0]).expect("the reference rate is always measured");
        for &rate in &rates {
            let Some(s) = get(si, rate) else { continue };
            let bytes = results
                .iter()
                .find(|r| r.shape_idx == si && r.rate == rate)
                .map(|r| r.bytes)
                .unwrap_or(0);
            println!(
                "{:<13} {rate:>4} {:>8.1} {:>8.1} {:>8.3} {:>6.3}  {}",
                SHAPES[si].name,
                bytes as f64 / 1e6,
                s.median,
                s.spread,
                amplification(rate),
                if rate == rates[0] {
                    format!("1.000 (ref, {:.1})", s.median)
                } else {
                    format!(
                        "{:.3}x   amp predicts {:.3}x   residual {:.3}",
                        s.median / base.median,
                        amplification(rates[0]) / amplification(rate),
                        (s.median / base.median)
                            / (amplification(rates[0]) / amplification(rate)),
                    )
                }
            );
        }
    }

    // --- aggregate, byte-weighted: the figure comparable to the 187.7 GB/s the
    // spec recorded for `gate_up_gelu`.
    println!("\n=== aggregate over all shapes, byte-weighted ===");
    let mut aggregates: Vec<(u8, f64, f64, f64)> = Vec::new();
    for &rate in &rates {
        let mut total_bytes = 0f64;
        let mut seconds = 0f64;
        let mut worst_spread = 0f64;
        for &si in &shape_idxs {
            let Some(s) = get(si, rate) else { continue };
            let bytes = results
                .iter()
                .find(|r| r.shape_idx == si && r.rate == rate)
                .map(|r| r.bytes)
                .unwrap_or(0);
            total_bytes += bytes as f64;
            seconds += bytes as f64 / s.median / 1e9;
            worst_spread = worst_spread.max(s.spread);
        }
        aggregates.push((rate, total_bytes / seconds / 1e9, amplification(rate), worst_spread));
    }
    for (rate, gbps, amp, worst_spread) in &aggregates {
        println!(
            "  {rate} bpw: {:>7.1} GB/s   amp {amp:.3}   worst per-shape spread \
             {worst_spread:.3}",
            gbps
        );
    }
    let (ref_rate, ref_gbps, ref_amp, _) = aggregates[0];
    for (rate, gbps, amp, _) in aggregates.iter().skip(1) {
        let ratio = gbps / ref_gbps;
        let predicted = ref_amp / amp;
        println!(
            "\n  VERDICT  {rate}bpw / {ref_rate}bpw = {ratio:.3}x   \
             amplification model predicts {predicted:.3}x   residual {residual:.3}",
            residual = ratio / predicted,
        );
        println!(
            "           {}",
            if ratio >= predicted * 0.97 {
                "the rate's lane-load amplification accounts for the gap: the kernel \
                 is sound at this rate and the 3 bpw ceiling is a FORMAT property"
            } else if ratio <= 1.02 {
                "the stall is rate-INDEPENDENT: this rate moves the amplification and \
                 nothing else moves, so the cost is in the kernel, not the format"
            } else {
                "BETWEEN: between the model's prediction and no effect -- look at the \
                 per-shape table, a shape-dependent ratio localises the defect"
            }
        );
    }

    println!(
        "\nheld {:.2} GiB peak (ru_maxrss; resident pages only, NOT a footprint per \
         AGENTS.md 10.3). If a model was resident while this ran, these numbers are \
         contention-limited and must be re-taken on an idle box.",
        max_rss_bytes() as f64 / (1u64 << 30) as f64,
    );
}
