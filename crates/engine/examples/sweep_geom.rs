//! Decode dispatch-geometry sweep: (K-split x threads-per-threadgroup) per shape.
//!
//! The gap this measures is the one TensorFold's own note describes: its
//! `plan()` picks a `(SK, warps)` pair from the shape alone, and on the 3-bit
//! rows of their 27B packs "the split a shape alone picks is 3-19% off the best
//! cell of the (K split, warps) grid" (`linear.py:30-45`). That grid has never
//! been built here, and the one axis of it this repository has measured at all
//! is the `SK` half, once, on three shapes (T7.6.9 — `gate_up 168.1 vs 188.2
//! GB/s at SK4`), after which split-K was gated behind `KAPLUMBA_USE_SK=1`.
//!
//! So: 4 K-splits x 3 threadgroup widths over every shape, each measured over a
//! working set far larger than any cache, interleaved, median with spread, and
//! with the split-K reduction *inside* the timed region so a cell cannot look
//! cheap by leaving its partials unreduced. The output is a table plus a
//! per-token projection; the recommendation that follows is arithmetic on those.
//!
//! Usage:
//! ```text
//! cargo run --release -p kaplumba-engine --example sweep_geom -- [rounds] [working_set_gb] [shape-filter...]
//! ```
//!
//! `rounds` defaults to 7 (AGENTS.md 10.5: one reading is not a result),
//! `working_set_gb` to 6.0, and an empty filter means every shape.
//!
//! Do not run this while a model is resident (AGENTS.md 10). Use `gpuq try`.

use half::f16;
use kaplumba_engine::bench::{summarize, Layout, Shape, Summary, SHAPES};
use kaplumba_engine::exl3::{
    decode_state, extract_ishizuki_lane_codes, pack_integer_states, reorder_tensor_to_colmajor, Codebook,
};
use kaplumba_kernels::encoder::MetalEncoder;
use kaplumba_kernels::sweep::{cells, Cell, SweepKernels, SweepRequest};
use metal::{Buffer, CommandQueue, Device, MTLResourceOptions};
use std::collections::BTreeMap;
use std::time::Instant;

/// A sample long enough that a clock tick is noise, short enough that a
/// round-robin over 12 cells still fits in one sitting. `bench_residency.rs`
/// uses the same 40 ms.
const ROUND_TARGET_MS: f64 = 40.0;
const MIN_ITERS: u64 = 4;
/// 16 rather than 4: at four dispatches the per-launch cost dominates and the
/// calibration hands the timed sample an iteration count ~3x too high, which on
/// the smallest shape means a 17 ms "40 ms" round.
const CALIBRATION_ITERS: u64 = 16;
const MAX_ITERS: u64 = 4096;
/// Tile rows the CPU reference covers. 256 of them is 4096 output values,
/// spread across the whole index space, and costs ~2 s on the widest shape --
/// a full-tensor reference would cost ten times that and pin the same mapping.
const REFERENCE_TILE_ROWS: usize = 256;
/// `Summary::is_repeatable`'s tolerance: a cell's worst round must land within
/// this fraction of its best before the table will call its median a result.
/// Two cells measured while another agent's model was loading produced spreads
/// of 0.17 and 0.45 and a nominal "+5.5% gap" that was pure contention, so the
/// floor is not decoration.
const RESOLVE_TOLERANCE: f64 = 0.92;

/// How many times each shape is read per token: 50 sliding + 10 full-attention
/// layers, plus `lm_head` once. `bench::per_token_weight_bytes` is the
/// cross-check, asserted at the end of `main`.
fn reads_per_token(name: &str) -> u64 {
    match name {
        "sliding/q" | "sliding/k" | "sliding/v" | "sliding/o" => 50,
        "sliding/gate" | "sliding/up" | "sliding/down" => 60,
        "full/q" | "full/kv" | "full/o" => 10,
        "lm_head" => 1,
        other => panic!("{other} is not in SHAPES; bench.rs and this table must agree"),
    }
}

struct Ctx {
    queue: CommandQueue,
    in_features: usize,
    out_features: usize,
    /// fp16 activations, as every colmajor kernel wants them. Raw (untransformed):
    /// the sweep is about dispatch geometry, so the input transform is not part
    /// of it, and the CPU reference multiplies by these exact values.
    x: Buffer,
    partials: Buffer,
    y: Buffer,
}

fn upload(device: &Device, v: &[u16]) -> Buffer {
    device.new_buffer_with_data(
        v.as_ptr() as *const std::ffi::c_void,
        (v.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn upload_f32(device: &Device, v: &[f32]) -> Buffer {
    device.new_buffer_with_data(
        v.as_ptr() as *const std::ffi::c_void,
        (v.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

/// Deterministic packed trellis: mixed xorshift per tile. An LCG of period 8 has
/// bitten a generator in this repo before (`decode_nr0.rs:30`).
fn build(bits: u8, in_f: usize, out_f: usize, seed: u32) -> Vec<u16> {
    let tiles = (in_f / 16) * (out_f / 16);
    let mut packed = Vec::with_capacity(tiles * 16 * bits as usize);
    let mut state = seed | 1;
    for _ in 0..tiles {
        let mut states = [0u16; 256];
        for st in states.iter_mut() {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            *st = (state & ((1u32 << bits) - 1)) as u16;
        }
        let (words, _) = pack_integer_states(&states, bits).expect("pack");
        packed.extend(words);
    }
    packed
}

/// The sweep library carries its own `decode_state_mcg` (a copy, because MSL has
/// no include path through `new_library_with_source`). Proving the copy is the
/// engine's function is the difference between a table about this engine's
/// codebook and a table about a plausible-looking stranger.
///
/// The tolerance is 4 fp16 ULP, and it is not a fudge: `decode_state_mcg` sums
/// its two halves as **half2**, i.e. in fp16, before widening
/// (`linear.metal:121-126`), while `exl3::decode_state` sums them in f32. So
/// the shipped kernel already differs from the CPU reference by up to one fp16
/// ULP on some states, and this probe reproduces exactly that much and no
/// more — measured: 12 of 256 states differ, worst 1 ULP (0.00098 absolute at
/// state 3, 0.8396 against 0.8398). A wrong multiplier or a dropped xor term
/// would be off by O(1), four orders of magnitude outside this bound.
///
/// The end-to-end proof is still the per-cell comparison against the engine's
/// own GEMV output; this probe exists to name the function if it drifts.
fn check_codebook_copy(device: &Device, kernels: &SweepKernels) {
    /// fp16 has a 10-bit significand, so one ULP is a relative 2^-10.
    const FP16_ULP: f32 = 1.0 / 1024.0;
    let out = upload_f32(device, &vec![0.0f32; 256]);
    let queue = device.new_command_queue();
    let mut encoder = MetalEncoder::new(&queue).expect("encoder");
    kernels.dispatch_mcg_probe(&mut encoder, &out).expect("probe");
    encoder.commit_and_wait().expect("wait");
    let got = unsafe { std::slice::from_raw_parts(out.contents() as *const f32, 256) };
    let mut differing = 0usize;
    let mut worst_ulps = 0.0f32;
    let mut worst = None;
    for (state, gpu) in got.iter().enumerate() {
        let want = decode_state(state as u16, Codebook::Mcg);
        let ulps = (gpu - want).abs() / (want.abs().max(1e-3) * FP16_ULP);
        if ulps > 0.5 {
            differing += 1;
        }
        if ulps > worst_ulps {
            worst_ulps = ulps;
            worst = Some((state, *gpu, want));
        }
    }
    assert!(
        worst_ulps <= 4.0,
        "the sweep library's decode_state_mcg is not the engine's: state {:?} decoded {:?} \
         against {:?}, {:.1} fp16 ULP out. Every bandwidth below would be a measurement of \
         a kernel nobody ships.",
        worst.map(|w| w.0),
        worst.map(|w| w.1),
        worst.map(|w| w.2),
        worst_ulps,
    );
    println!(
        "codebook copy check: {differing}/256 states differ from the f32 CPU reference, \
         worst {:.2} fp16 ULP (state {}) — that is the shipped kernel's own fp16 add, \
         not drift",
        worst_ulps,
        worst.map(|w| w.0).unwrap_or(0),
    );
}

struct Arm {
    bytes: u64,
    buffer: Buffer,
    /// Whole copies of the tensor packed into `buffer`, and one copy's byte
    /// length. A dispatch reads copy `w % copies`, so consecutive dispatches
    /// stream different DRAM.
    copies: u64,
    stride: u64,
    /// The first copy, kept for the CPU reference. The GPU only ever sees
    /// `buffer`.
    first_copy: Vec<u16>,
}

/// `target_gb` of packed weights, as whole copies of one column-major tensor so
/// a dispatch can cycle the offset and stream past any cache. At least 2 copies,
/// so the cycling is real even for the widest shape.
///
/// Filled through `contents()` rather than `new_buffer_with_data` of a 6 GB host
/// vector: that vector is a second 6 GB allocation, and on a 24 GiB box with a
/// model loaded the difference is between measuring and evicting someone else's
/// run.
fn build_arm(device: &Device, shape: &Shape, target_gb: f64, seed: u32) -> Arm {
    let colmajor = reorder_tensor_to_colmajor(
        &build(shape.bits, shape.in_features, shape.out_features, seed),
        shape.in_features,
        shape.out_features,
    )
    .expect("colmajor");
    let stride = (colmajor.len() * 2) as u64;
    assert_eq!(
        stride,
        shape.weight_bytes(),
        "{}: built {} bytes, checkpoint says {}",
        shape.name,
        stride,
        shape.weight_bytes()
    );
    let copies = ((target_gb * 1e9) as u64 / stride).max(2);
    let total = stride * copies;
    let buffer = device.new_buffer(
        total,
        MTLResourceOptions::StorageModeShared | MTLResourceOptions::HazardTrackingModeUntracked,
    );
    {
        let base = buffer.contents() as *mut u16;
        for c in 0..copies {
            let tag = (c as u16).wrapping_mul(0x9E37);
            let dst = unsafe { base.add(c as usize * colmajor.len()) };
            for (i, word) in colmajor.iter().enumerate() {
                unsafe { *dst.add(i) = *word ^ tag };
            }
        }
    }
    Arm {
        bytes: shape.weight_bytes(),
        buffer,
        copies,
        stride,
        first_copy: colmajor,
    }
}

/// The 8 window states a lane owns, as the *16-bit* values the kernel decodes.
///
/// Not the same thing as `extract_ishizuki_lane_codes`, which masks each state
/// down to `& 0x7` because its caller is the 3-bit bit-plane repacker. The GEMV
/// hands `decode_state_mcg` the full 16 bits, and that matters: the codebook
/// keeps bits 19-12 and 7-0 of `state * M`, so the bits the mask throws away
/// reach the result *through the multiply*. Using the masked codes here gives a
/// reference that is 100% wrong and presents as a broken kernel.
///
/// The link to `exl3.rs` is kept where it is checkable: every state here is
/// asserted to equal `extract_ishizuki_lane_codes` in its low 3 bits, so if this
/// window arithmetic ever drifts from the engine's copy it panics instead of
/// quietly measuring the wrong function.
fn lane_states(words: &[u32; 24], lane: usize) -> [u16; 8] {
    let ka = 3u32;
    let ring = 256u32 * ka;
    let words_count = (ring / 32) as usize;
    let lo0 = (lane as u32 * 8 + 1) * ka + ring - 16;
    let sh0 = lo0 % 32;
    let hi0 = (lo0 / 32) as usize;
    let j0 = if hi0 >= words_count { hi0 - words_count } else { hi0 };
    let j1 = if j0 + 1 >= words_count { j0 + 1 - words_count } else { j0 + 1 };
    let a0 = (((words[j0] as u64) << 32) | (words[j1] as u64)) << sh0;
    let mut states = [0u16; 8];
    for (i, slot) in states.iter_mut().enumerate() {
        *slot = ((a0 >> (64 - i as u32 * ka - 16)) as u32 & 0xffff) as u16;
    }
    let codes = extract_ishizuki_lane_codes(words, lane);
    for (i, code) in codes.iter().enumerate() {
        assert_eq!(
            states[i] & 0x7,
            *code as u16,
            "lane {lane} code {i}: this window arithmetic disagrees with exl3.rs"
        );
    }
    states
}

/// An independent CPU reference for the output tile-rows in `tile_rows`.
///
/// Transcribed from the kernel's own thread mapping, so the only thing being
/// pinned is the part nothing else checks: which `(row, col)` of the 16x16 tile
/// each of a lane's 8 states belongs to. Read that mapping wrong and every cell
/// agrees with every other cell while all of them compute a different function.
///
/// Why a CPU reference and not the engine's own GEMV: it is independent of
/// `LinearKernels` (which is not always loadable while the tree is mid-edit), and
/// it is a stronger claim — "this cell computes `W x`" rather than "this cell
/// agrees with another cell".
fn cpu_reference(arm: &Arm, in_f: usize, tile_rows: &[usize]) -> Vec<f32> {
    let tiles_k = in_f / 16;
    let packed = &arm.first_copy;
    let x: Vec<f32> = (0..in_f)
        .map(|i| f16::from_f32((i as f32 * 0.017).sin()).to_f32())
        .collect();
    let mut out = vec![0.0f32; tile_rows.len() * 16];
    for (block, tn) in tile_rows.iter().enumerate() {
        let acc = &mut out[block * 16..(block + 1) * 16];
        for tk in 0..tiles_k {
            let tile = &packed[(*tn * tiles_k + tk) * 48..][..48];
            let mut words = [0u32; 24];
            for (w, slot) in words.iter_mut().enumerate() {
                *slot = tile[w * 2] as u32 | (tile[w * 2 + 1] as u32) << 16;
            }
            for lane in 0..32usize {
                let q = lane / 4;
                let r0 = (lane % 4) * 2;
                for (i, state) in lane_states(&words, lane).iter().enumerate() {
                    // acc0 covers states 0..4 at columns r0 + {0,1,8,9} and output
                    // rows q; acc1 covers states 4..8 at rows q + 8. Same columns,
                    // which is why the mapping is a stride on `i` and not a
                    // separate case.
                    let col = r0 + match i % 4 {
                        0 => 0,
                        1 => 1,
                        2 => 8,
                        _ => 9,
                    };
                    let row = q + if i >= 4 { 8 } else { 0 };
                    acc[row] += decode_state(*state, Codebook::Mcg) * x[tk * 16 + col];
                }
            }
        }
    }
    out
}

/// Evenly spaced tile rows, so the reference spans the whole output index space
/// rather than only its first few thousand entries.
fn reference_tile_rows(tiles_n: usize, wanted: usize) -> Vec<usize> {
    let count = wanted.min(tiles_n);
    (0..count)
        .map(|i| i * tiles_n / count)
        .collect::<Vec<usize>>()
        .tap_dedup()
}

/// `Vec::dedup` that tolerates a repeated index (it cannot happen at `count > 1`,
/// but a zero-length tile axis could produce one and a silent duplicate block
/// would make the comparison read the wrong slice).
trait TapDedup {
    fn tap_dedup(self) -> Vec<usize>;
}

impl TapDedup for Vec<usize> {
    fn tap_dedup(self) -> Vec<usize> {
        let mut out: Vec<usize> = Vec::with_capacity(self.len());
        for v in self {
            if out.last() != Some(&v) {
                out.push(v);
            }
        }
        out
    }
}

fn ctx_for(device: &Device, queue: &CommandQueue, shape: &Shape) -> Ctx {
    let in_f = shape.in_features;
    let out_f = shape.out_features;
    let x: Vec<u16> = (0..in_f)
        .map(|i| f16::from_f32((i as f32 * 0.017).sin()).to_bits())
        .collect();
    Ctx {
        queue: queue.clone(),
        in_features: in_f,
        out_features: out_f,
        x: upload(device, &x),
        partials: device.new_buffer((8 * out_f * 4) as u64, MTLResourceOptions::StorageModeShared),
        y: device.new_buffer((out_f * 4) as u64, MTLResourceOptions::StorageModeShared),
    }
}

/// `(max |a-b|, max |a-b| / max(1, max|b|))` against `reference`.
fn deviation(got: &[f32], reference: &[f32]) -> (f32, f32) {
    let scale = reference.iter().fold(0.0f32, |acc, v| acc.max(v.abs())).max(1.0);
    let worst = got
        .iter()
        .zip(reference)
        .fold(0.0f32, |acc, (a, b)| acc.max((a - b).abs()));
    (worst, worst / scale)
}

fn request<'a>(ctx: &'a Ctx, arm: &'a Arm, dispatch_index: u64) -> SweepRequest<'a> {
    SweepRequest {
        packed: &arm.buffer,
        packed_offset: (dispatch_index % arm.copies) * arm.stride,
        x: &ctx.x,
        partials: &ctx.partials,
        y: &ctx.y,
        in_features: ctx.in_features,
        out_features: ctx.out_features,
    }
}

/// One timed sample: `iters` dispatches (and, for `sk > 1`, `iters` reductions)
/// inside one command buffer.
fn sample(
    kernels: &SweepKernels,
    ctx: &Ctx,
    cell: Cell,
    arm: &Arm,
    iters: u64,
    first_copy_index: u64,
) -> f64 {
    let mut encoder = MetalEncoder::new(&ctx.queue).expect("encoder");
    let started = Instant::now();
    for w in 0..iters {
        kernels
            .dispatch(&mut encoder, cell, &request(ctx, arm, first_copy_index + w))
            .expect("sweep dispatch");
    }
    encoder.commit_and_wait().expect("wait");
    let elapsed = started.elapsed().as_secs_f64();
    if std::env::var("KAPLUMBA_SWEEP_DEBUG").is_ok() {
        eprintln!(
            "  [dbg] {:<10} iters {iters:>4}  total {:.1} ms  per-dispatch {:.1} us  -> {:.1} GB/s",
            cell.label(),
            elapsed * 1e3,
            elapsed / iters as f64 * 1e6,
            arm.bytes as f64 / (elapsed / iters as f64) / 1e9
        );
    }
    arm.bytes as f64 / (elapsed / iters as f64) / 1e9
}

/// A single dispatch, output left in `ctx.y` for comparison.
fn once(kernels: &SweepKernels, ctx: &Ctx, cell: Cell, arm: &Arm) {
    let mut encoder = MetalEncoder::new(&ctx.queue).expect("encoder");
    kernels
        .dispatch(&mut encoder, cell, &request(ctx, arm, 0))
        .expect("sweep dispatch");
    encoder.commit_and_wait().expect("wait");
}

fn read_y(ctx: &Ctx) -> Vec<f32> {
    unsafe { std::slice::from_raw_parts(ctx.y.contents() as *const f32, ctx.out_features) }.to_vec()
}

/// TensorFold's `plan()` and its `min_tiles` guard, transcribed from
/// `linear.py:30-45`. Reported, never used to filter a cell out: the point of
/// the sweep is to see whether the heuristic's answer is its grid's best cell,
/// so the cells the heuristic rejects still have to be in the table.
struct Plan {
    sk: u32,
    warps: u32,
    /// The split it would choose if `nb * sk` alone satisfied it, i.e. before
    /// `min_tiles` walks it back.
    wanted_sk: u32,
}

fn plan(k: usize, n: usize, blocks: u32, min_tiles: u32) -> Plan {
    let kt = (k / 16) as u32;
    let nb = (n / 128) as u32;
    let mut sk = 1u32;
    let mut wk = if nb >= 64 { 8 } else { 4 };
    if nb < 64 {
        while nb * sk < blocks && sk < 64 {
            sk *= 2;
        }
    }
    let wanted_sk = sk;
    while (wk > 4 || sk > 1) && (kt % (sk * wk) != 0 || kt / (sk * wk) < min_tiles) {
        if wk > 4 {
            wk = 4;
        } else if sk > 1 {
            sk /= 2;
        } else {
            break;
        }
    }
    Plan {
        sk,
        warps: wk,
        wanted_sk,
    }
}

/// Our split rule, transcribed from `contig4_sk2` / `contig4_sk4`:
/// `k_per_split = ((tiles_k / SK) / 32u) * 32u` — rounded *down*, with no floor.
/// Returns each split's `[start, end)`, so a degenerate tail shows up as a
/// zero-width split or a last split several times the others.
fn our_k_splits(tiles_k: u32, sk: u32) -> Vec<(u32, u32)> {
    (0..sk)
        .map(|s| {
            let per = ((tiles_k / sk) / 32) * 32;
            let start = s * per;
            let end = if s == sk - 1 { tiles_k } else { start + per };
            (start, end)
        })
        .collect()
}

/// One line per shape: what TensorFold's `min_tiles=8` asks for, what our split
/// rule actually hands each threadgroup, and whether any of it is degenerate.
fn guard_row(name: &str, tiles_k: u32, p: &Plan) -> String {
    let mut parts = vec![format!(
        "{name:<13} tiles_k {tiles_k:>4}  plan SK {} x {} warps",
        p.sk, p.warps
    )];
    for sk in [2u32, 4, 8] {
        let widths: Vec<u32> = our_k_splits(tiles_k, sk)
            .iter()
            .map(|(a, b)| b - a)
            .collect();
        let min = widths.iter().copied().min().unwrap_or(0);
        let max = widths.iter().copied().max().unwrap_or(0);
        parts.push(format!(
            "SK{sk} {widths:?} (widest/narrowest {:.1}x, zero-width split: {})",
            max as f64 / min.max(1) as f64,
            min == 0
        ));
    }
    // The narrowest split at the widest threadgroup: 16 simdgroups.
    let narrowest = our_k_splits(tiles_k, 8)
        .iter()
        .map(|(a, b)| b - a)
        .min()
        .unwrap_or(0);
    parts.push(format!(
        "tiles/simdgroup at SK=8,TG=512: {:.1} ({})",
        narrowest as f64 / 16.0,
        if narrowest / 16 >= 8 {
            ">= min_tiles"
        } else {
            "BELOW min_tiles"
        }
    ));
    parts.join("  |  ")
}

struct Row {
    cell: Cell,
    /// Max absolute and scale-relative deviation from the CPU reference.
    deviation: (f32, f32),
    iters: u64,
}

/// Where one shape's measurements are cached between runs, so a sweep can be
/// assembled across gaps.
///
/// This box is shared: AGENTS.md §10 forbids overlapping GPU jobs, and in
/// practice that means waiting out model runs of minutes at a time.
/// Re-measuring all ten shapes because the seventh was interrupted is the wrong
/// trade when one shape is a two-second job. The cache is keyed by
/// `(shape, rounds, working set)` and a mismatched key is a miss rather than a
/// silent join — a table stitched from two different round counts is worse than
/// no table, because the spread column stops meaning anything.
///
/// The correctness pass is deliberately *not* cached: a cached row cannot claim a
/// deviation it did not measure, so `print_shape` says so instead.
fn cache_path(shape: &str, rounds: usize, target_gb: f64) -> std::path::PathBuf {
    std::env::var("KAPLUMBA_SWEEP_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
            std::path::PathBuf::from(format!("{home}/.kaplumba/scratch"))
        })
        .join(format!(
            "sweep_geom_{}_{}_{:.1}.txt",
            shape.replace(['/', ' '], "_"),
            rounds,
            target_gb
        ))
}

/// A deliberately trivial format. It is read by this same file's parser and by
/// nothing else, so a hand edit is a way to break the run, not a feature.
fn write_cache(shape: &Shape, rounds: usize, target_gb: f64, cells: &BTreeMap<String, Summary>) {
    let path = cache_path(shape.name, rounds, target_gb);
    if std::fs::create_dir_all(path.parent().expect("parent")).is_err() {
        return;
    }
    let mut text = format!("rounds {rounds} target_gb {target_gb:.1}\n");
    for (label, summary) in cells {
        text.push_str(&format!(
            "{label} {} {} {} {} {}\n",
            summary.n, summary.median, summary.best, summary.worst, summary.spread
        ));
    }
    let _ = std::fs::write(path, text);
}

fn read_cache(shape: &Shape, rounds: usize, target_gb: f64) -> Option<BTreeMap<String, Summary>> {
    let text = std::fs::read_to_string(cache_path(shape.name, rounds, target_gb)).ok()?;
    let mut lines = text.lines();
    let header = lines.next()?;
    if header != format!("rounds {rounds} target_gb {target_gb:.1}") {
        return None;
    }
    let mut out = BTreeMap::new();
    for line in lines {
        let mut fields = line.split_whitespace();
        let label = fields.next()?.to_string();
        let values: Vec<f64> = (0..5)
            .map(|_| fields.next()?.parse::<f64>().ok())
            .collect::<Option<Vec<f64>>>()?;
        out.insert(
            label,
            Summary {
                n: values[0] as usize,
                median: values[1],
                best: values[2],
                worst: values[3],
                spread: values[4],
            },
        );
    }
    (!out.is_empty()).then_some(out)
}

/// Print one shape's cells, best first, with the baseline marked and the gap to
/// it stated — the gap is the deliverable, so it is computed here rather than
/// left to whoever reads the table.
fn print_shape(cells: &BTreeMap<String, Summary>, rows: &[Row]) {
    let baseline = cells
        .get(&Cell::baseline().label())
        .expect("baseline cell is always in the grid")
        .median;
    let mut printed: Vec<(&String, &Summary)> = cells.iter().collect();
    printed.sort_by(|a, b| b.1.median.partial_cmp(&a.1.median).expect("finite"));
    for (label, summary) in &printed {
        let deviation = rows
            .iter()
            .find(|r| r.cell.label() == **label)
            .map(|r| r.deviation.1);
        println!(
            "  {label:<10} median {:>7.1} GB/s  spread {:.3}  gap {:+6.1}%{}{}",
            summary.median,
            summary.spread,
            (summary.median / baseline - 1.0) * 100.0,
            match deviation {
                Some(dev) => format!("  rel dev {dev:.1e}"),
                // Only reachable from the cache path, which does not re-run the
                // correctness pass. Said so rather than printed as a real number.
                None => "  (correctness not re-checked: cached)".to_string(),
            },
            if **label == Cell::baseline().label() {
                "   <- engine default"
            } else {
                ""
            },
        );
    }
    let (best_label, best) = printed[0];
    let gap = (best.median / baseline - 1.0) * 100.0;

    // A gap smaller than the noise is not a gap. AGENTS.md 10.5 records decode
    // moving 3.7-6.4 tok/s from one binary, and §3.4 requires a claim to survive
    // being broken; the same discipline applies here, so a cell whose worst
    // round fell below `baseline - gap` cannot support a claim, and the table
    // says so rather than quoting a number.
    let resolved = best.spread >= RESOLVE_TOLERANCE
        && cells
            .get(&Cell::baseline().label())
            .map(|s| s.spread >= RESOLVE_TOLERANCE)
            .unwrap_or(false);
    println!(
        "  best cell {best_label} at {:.1} GB/s: {gap:+.1}% vs the engine default{}",
        best.median,
        if resolved {
            ""
        } else {
            "  [NOT RESOLVED: a cell's spread exceeds the gap; re-measure with the GPU idle]"
        }
    );
    if let Some(noisy) = cells
        .iter()
        .filter(|(_, s)| s.spread < RESOLVE_TOLERANCE)
        .map(|(label, s)| format!("{label} {:.3}", s.spread))
        .collect::<Vec<_>>()
        .first()
    {
        println!("    cells below the {RESOLVE_TOLERANCE} repeatability floor: {noisy} (and possibly others)");
    }
    println!();
}

fn print_cached(cells: &BTreeMap<String, Summary>) {
    print_shape(cells, &[]);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let rounds: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(7);
    let target_gb: f64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(6.0);
    let filter: Vec<String> = args[3..].to_vec();

    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue().to_owned();
    let kernels = SweepKernels::new().expect("SweepKernels (MSL compile)");
    check_codebook_copy(&device, &kernels);

    let shapes: Vec<&Shape> = SHAPES
        .iter()
        .filter(|s| filter.is_empty() || filter.iter().any(|f| s.name.starts_with(f.as_str())))
        .collect();
    let skipped: Vec<&Shape> = shapes
        .iter()
        .filter(|s| s.layout != Layout::Colmajor)
        .copied()
        .collect();

    println!("=== sweep_geom: decode dispatch geometry (K-split x threads/threadgroup) ===");
    println!(
        "{rounds} interleaved rounds per cell, {target_gb:.1} GB working set, \
         reduction inside the timed region\n"
    );
    if !skipped.is_empty() {
        println!("Shapes this table cannot cover:");
        for shape in &skipped {
            println!(
                "  {}: {} bpw, {} layout -- a different kernel family, and {} tile rows is \
                 already {:.0} waves at SK=1 (20 GPU cores). TensorFold's own rule sends a \
                 shape that wide to SK=1 for the same reason.",
                shape.name,
                shape.bits,
                match shape.layout {
                    Layout::Colmajor => "column-major",
                    Layout::RowMajor => "row-major",
                },
                shape.tile_rows(),
                shape.tile_rows() as f64 / 20.0,
            );
        }
        println!();
    }

    let grid = cells();
    let mut table: BTreeMap<String, BTreeMap<String, Summary>> = BTreeMap::new();
    let mut guard_rows: Vec<String> = Vec::new();

    for (index, shape) in shapes.iter().enumerate() {
        if shape.layout != Layout::Colmajor {
            continue;
        }
        let tiles_k = (shape.in_features / 16) as u32;
        let p = plan(shape.in_features, shape.out_features, 192, 8);
        guard_rows.push(guard_row(shape.name, tiles_k, &p));

        if let Some(cached) = read_cache(shape, rounds, target_gb) {
            if cached.len() == grid.len() {
                println!("--- {}  (from cache, not re-measured)\n", shape.name);
                print_cached(&cached);
                table.insert(shape.name.to_string(), cached);
                continue;
            }
        }

        let ctx = ctx_for(&device, &queue, shape);
        let arm = build_arm(&device, shape, target_gb, 0x9E3779B9 ^ (index as u32 + 1));
        let tiles_n = shape.tile_rows();
        let ref_rows = reference_tile_rows(tiles_n, REFERENCE_TILE_ROWS);
        let reference = cpu_reference(&arm, shape.in_features, &ref_rows);
        assert!(
            reference.iter().any(|v| v.abs() > 1e-3),
            "{}: the CPU reference produced only zeros, so every comparison below is vacuous",
            shape.name
        );

        let tiles_k = (shape.in_features / 16) as u32;
        let p = plan(shape.in_features, shape.out_features, 192, 8);
        println!(
            "--- {}  {}x{} @{}bpw  ({:.1} MB, {} tile rows x {} K-tiles)\n\
             \x20   TensorFold plan(): SK={} x {} warps ({} threads); it wanted SK={}\n",
            shape.name,
            shape.in_features,
            shape.out_features,
            shape.bits,
            shape.weight_bytes() as f64 / 1e6,
            shape.tile_rows(),
            tiles_k,
            p.sk,
            p.warps,
            p.warps * 32,
            p.wanted_sk,
        );

        // Per cell: prove it against the CPU reference, then calibrate its
        // iteration count. A cell that reads the wrong tiles is fast for the
        // wrong reason and must not reach the table.
        //
        // Two tolerances, and both are needed. The codebook's fp16 add is worth
        // about 1e-6 relative even before any reassociation, so `1e-3` is the
        // "right answer" gate. `20x` that baseline-to-baseline number is the
        // "agrees with the shipped kernel up to reduction order" gate, and it is
        // what catches a split that drops or double-counts K-tiles: those are
        // off by a factor, not by an ULP.
        let mut rows: Vec<Row> = Vec::new();
        for cell in &grid {
            once(&kernels, &ctx, *cell, &arm);
            let got = read_y(&ctx);
            // Block-major, matching `cpu_reference`'s output layout.
            let checked: Vec<f32> = ref_rows
                .iter()
                .flat_map(|tn| got[tn * 16..tn * 16 + 16].iter().copied())
                .collect();
            let dev = deviation(&checked, &reference);
            let coarse = sample(&kernels, &ctx, *cell, &arm, CALIBRATION_ITERS, 0);
            let secs = arm.bytes as f64 / coarse / 1e9;
            let iters = ((ROUND_TARGET_MS * 1e-3) / secs)
                .round()
                .clamp(MIN_ITERS as f64, MAX_ITERS as f64) as u64;
            rows.push(Row {
                cell: *cell,
                deviation: dev,
                iters,
            });
        }
        let baseline_dev = rows
            .iter()
            .find(|r| r.cell == Cell::baseline())
            .expect("baseline cell")
            .deviation
            .1;
        assert!(
            baseline_dev < 1e-3,
            "{}: the SK=1/TG=256 cell disagrees with the CPU reference by {:.2e} relative, \
             so the lane->(row,col) mapping this file assumes is wrong and no cell may be \
             timed",
            shape.name,
            baseline_dev
        );
        let ceiling = baseline_dev.max(1e-7) * 20.0;
        for row in &rows {
            assert!(
                row.deviation.1 < ceiling,
                "{}: cell {} is {:.2e} from the CPU reference, more than 20x the baseline \
                 cell's {:.2e}. A split that drops or double-counts K-tiles looks exactly \
                 like this.",
                shape.name,
                row.cell.label(),
                row.deviation.1,
                baseline_dev
            );
        }

        // Warm-up round, discarded.
        //
        // Not defensive padding: without it the first three cells' first sample
        // came in at 58.1 / 86.9 / 110.5 GB/s against a steady-state of 126 /
        // 134 / 135, and nothing else in the run looked like that. Those are
        // slots 0-2 of round 0, i.e. the first command buffers this process ever
        // commits — GPU clocks and first-use pipeline costs, not contention.
        // It also produced the only two spreads below the repeatability floor in
        // the whole sweep, which is what the floor is for.
        for slot in 0..rows.len() {
            sample(&kernels, &ctx, grid[slot], &arm, rows[slot].iters, 0);
        }

        // Interleaved, with the grid rotated one cell per round so a clock or
        // thermal drift lands on every cell rather than on the first ones.
        let mut samples: Vec<Vec<f64>> = vec![Vec::new(); rows.len()];
        for round in 0..rounds {
            for offset in 0..grid.len() {
                let slot = (offset + round) % grid.len();
                // Each round starts the copy cycle where the previous one left
                // off, so a cell cannot be handed a warm region by the cell that
                // ran before it in the round-robin.
                samples[slot].push(sample(
                    &kernels,
                    &ctx,
                    grid[slot],
                    &arm,
                    rows[slot].iters,
                    (round as u64 * 7 + slot as u64) * rows[slot].iters,
                ));
            }
        }

        if std::env::var("KAPLUMBA_SWEEP_SAMPLES").is_ok() {
            for (row, s) in rows.iter().zip(samples.iter()) {
                eprintln!(
                    "  [samples] {:<10} {}",
                    row.cell.label(),
                    s.iter().map(|v| format!("{v:.1}")).collect::<Vec<_>>().join(" ")
                );
            }
        }
        let shape_cells: BTreeMap<String, Summary> = rows
            .iter()
            .zip(samples.iter())
            .map(|(row, s)| {
                (
                    row.cell.label(),
                    summarize(s).unwrap_or_else(|| panic!("{}: no samples", shape.name)),
                )
            })
            .collect();
        print_shape(&shape_cells, &rows);
        write_cache(shape, rounds, target_gb, &shape_cells);
        table.insert(shape.name.to_string(), shape_cells);
    }

    println!("\n=== the min_tiles guard, against our split rule ===");
    println!(
        "\nOur rule (`contig4_sk2`/`contig4_sk4`): k_per = ((tiles_k / SK) / 32) * 32 -- rounded\n\
         DOWN to a multiple of 32, with no floor. Two ways that goes wrong:\n\
         \x20 * a width of 0 (tiles_k / SK < 32), so every split but the last reads nothing and\n\
         \x20   one threadgroup does the whole reduction -- correct, and pure overhead;\n\
         \x20 * the last split inheriting the rounding remainder, so it runs 2-5x the others and\n\
         \x20   the split-K cell is bounded by its slowest threadgroup.\n\
         'tiles/sg' is the narrowest split divided by 16 simdgroups, i.e. the TG=512 cell; below\n\
         8 is the direct analogue of TensorFold's `min_tiles=8`.\n"
    );
    for row in &guard_rows {
        println!("  {row}");
    }
    report_aggregate(&table);
    assert!(
        table.len() >= 10,
        "only {} shapes were measured; the table is not the one that was asked for",
        table.len()
    );
}

/// Per-token projection from the table, and the recommendation it implies.
///
/// * `(a)` one global configuration for every shape — the engine's today.
/// * `(b)` every shape on its own best cell from this table.
///
/// The ratio is the whole argument for or against maintaining a table, so it is
/// computed here rather than in prose where it cannot be checked.
fn report_aggregate(table: &BTreeMap<String, BTreeMap<String, Summary>>) {
    let mut global_ms = 0.0f64;
    let mut best_ms = 0.0f64;
    let mut widest_gap: Option<(String, f64)> = None;
    let mut split_k_wins = Vec::new();
    for shape in SHAPES {
        let (Some(cells), Some(base)) = (
            table.get(shape.name),
            table.get(shape.name).and_then(|c| c.get(&Cell::baseline().label())),
        ) else {
            continue;
        };
        let (best_label, best) = cells
            .iter()
            .max_by(|a, b| a.1.median.partial_cmp(&b.1.median).expect("finite"))
            .expect("non-empty");
        if best_label != &Cell::baseline().label() {
            split_k_wins.push(format!("{} -> {}", shape.name, best_label));
        }
        let reads = reads_per_token(shape.name) as f64;
        let bytes = shape.weight_bytes() as f64;
        global_ms += reads * bytes / base.median / 1e6;
        best_ms += reads * bytes / best.median / 1e6;
        let gap = best.median / base.median - 1.0;
        if widest_gap.as_ref().is_none_or(|(_, g)| gap > *g) {
            widest_gap = Some((shape.name.to_string(), gap));
        }
    }
    println!("\n=== aggregate: what per-shape autotuning would buy ===\n");
    println!("  (a) one global config (SK=1, 256 threads): {global_ms:.2} ms/token of matvec");
    println!("  (b) per-shape best cell from this table:   {best_ms:.2} ms/token");
    println!(
        "  difference: {:.2}% of matvec time",
        (1.0 - best_ms / global_ms) * 100.0
    );
    if let Some((name, gap)) = widest_gap {
        println!("  largest single-shape gap: {name} {gap:+.1}%");
    }
    println!(
        "  shapes whose best cell is not the default: {}",
        if split_k_wins.is_empty() {
            "none".to_string()
        } else {
            split_k_wins.join(", ")
        }
    );
    println!(
        "  NOTE: these are isolated GEMV launches. The engine's 60 layers are a dependent\n\
         \x20 chain, so the end-to-end gain is smaller than this ratio."
    );
}