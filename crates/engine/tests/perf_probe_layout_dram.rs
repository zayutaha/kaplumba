//! Physical DRAM access pattern of each EXL3 decode layout, and whether layout
//! can close the 203 -> 236.7 GB/s gap.
//!
//! # The question
//!
//! Decode reads 12.04 GB of weights per token. At the measured 62.3 ms/step that
//! is 193 GB/s of useful weight traffic; 20 tok/s needs 236.7 GB/s. The best
//! sustained 3-bpw GEMV kernel in this tree reaches 203 GB/s, so the target sits
//! *above* the best kernel. The hypothesis under test is that the shortfall is a
//! memory-system problem -- page hits, row-buffer locality, prefetch
//! friendliness -- and therefore addressable by changing the byte layout the
//! GEMV walks.
//!
//! # What the geometry actually is
//!
//! An EXL3 3-bpw tile is 16x16 codes x 3 bits = 768 bits = **96 bytes** = 24
//! `u32` (`RING = 256u * KA`, `WORDS = RING / 32u` in
//! `crates/kernels/src/linear_repacked.metal:1980-1983`). Every layout below is a
//! permutation of the same tiles; only the tile *order* differs.
//!
//! Row-major (`reorder_tensor_to_colmajor`'s input, read by
//! `linear_gemv_ishizuki_k3_w2`, `crates/kernels/src/linear.metal:8740-8741`):
//!
//! ```text
//! tile (tk, tn) at (tk * tiles_n + tn) * 96
//! ```
//!
//! Column-major (`reorder_tensor_to_colmajor`,
//! `crates/engine/src/exl3.rs:748-755`, read by
//! `linear_gemv_ishizuki_k3_colmajor_contig4`,
//! `crates/kernels/src/linear_repacked.metal:1999-2013`):
//!
//! ```text
//! tile (tk, tn) at (tn * tiles_k + tk) * 96
//! ```
//!
//! The grid is one threadgroup per output tile (`encode_ishizuki_gemv`,
//! `crates/kernels/src/linear.rs:1479-1480`: `groups =
//! tiles.min(gemv_max_groups())`; with the cap unset `ntg == tiles_n`, so
//! `for (tn = tgid; tn < tiles_n; tn += ntg)` at line 1999 runs exactly once).
//! Threadgroup `tgid` owns column `tgid` and nothing else.
//!
//! ## Column-major: one contiguous run per threadgroup
//!
//! A column is `tiles_k` tiles at 96 B stride with no gaps, so the threadgroup
//! walks a single ascending contiguous run of `tiles_k * 96` bytes. For K=5376
//! that is 336 tiles = **32,256 B = 252 x 128 B lines exactly**. Within one
//! unrolled iteration simdgroup `sg` takes tiles `[4sg, 4sg+4)` and `tk += 32`,
//! so the 8 simdgroups together cover tiles `[32i, 32i+32)` = **3,072 contiguous
//! bytes per iteration, advancing by exactly 3,072**. No gap, no overlap, no
//! revisitation: every byte in the column is consumed once, in order. That is
//! the ideal DRAM pattern.
//!
//! One honest exception, which
//! `contig4_partitions_the_column_without_gap_or_overlap` pins rather than
//! hides: `tiles_k % 32 == 16` at K=5376, so the last 16 tiles (1,536 B,
//! **4.8%** of the column) fall to the tail loop at
//! `linear_repacked.metal:2041`, which takes **one** tile per simdgroup at
//! `SG`-tile spacing -- eight 96-byte reads 768 B apart. Everything before it is
//! a gapless sequential run.
//!
//! Each *lane* reads 8 bytes (`t_i[j0]`, `t_i[j1]`, `WORDS = 24`) at a
//! lane-specific offset inside the 96-byte tile: `j0` runs over all 24 words as
//! `lane` runs 0..31, so the 32 lanes issue 64 word-reads covering 24 distinct
//! words. The over-fetch is entirely intra-tile and absorbed by L1; it never
//! reaches DRAM.
//!
//! ## Row-major: 96 bytes every 126 KB
//!
//! `stride = tiles_n * 96`, so a lane that finishes tile `tk` finds tile `tk+1`
//! **129,024 bytes away** at N=21504 (32,256 bytes at N=5376). Worse, the 8
//! simdgroups of a threadgroup are deliberately spread `SG` tiles apart in K
//! (`linear.metal:8742`, `tk = sg`), so at any instant one threadgroup is
//! touching 8 tiles that are 32 KB to 126 KB from each other. Per-threadgroup
//! DRAM locality collapses from a 3,072-byte sequential run to **eight 96-byte
//! bursts, stride `tiles_n * 96`** -- a 1:1344 stride/run ratio.
//!
//! Across the *grid* row-major is not hopeless: at a fixed `tk`, tile `(tk, tn)`
//! for `tn = 0..tiles_n-1` is itself contiguous. But exploiting that requires all
//! 1344 threadgroups to march through K in lockstep, which is a scheduling
//! accident, not a property of the layout. Column-major needs no cooperation.
//!
//! That asymmetry is the whole of the ~1 tok/s the repack bought (14 -> 16.05).
//! It is a real effect, it is small, and the reason it is small is below.
//!
//! # The finding that decides the question
//!
//! `spec/decode-19tok-repacked-250gb/implementation_notes.md:1182,1199,1233-1237`
//! records arm (d) of the load/ALU partition: `tn -> tn & 3`, which keeps every
//! load, every address computation and every ALU op but folds the footprint from
//! 2.6 GB of DRAM to a **129 KB L2-resident window**
//! (`crates/kernels/probe/contig4_load_alu.metal:852-853`). Measured `-0.8%`.
//!
//! That is the ceiling on what any layout change can buy. Layout only decides
//! *where the bytes come from*; if making 99.995% of them come from L2 instead
//! of DRAM is worth under 1%, then no reordering, blocking, alignment or
//! co-location scheme is worth more. Arm (b) -- remove *every* weight load -- is
//! 7.8%, which is the entire load-issue + memory budget. The target needs
//! **+16.6%**.
//!
//! **Conclusion: layout cannot close the gap. 203 GB/s is not a bandwidth wall;
//! it is an ALU wall** -- the 64-bit variable barrel shift
//! `uint(A >> (64u - rad))` in `decode_state_mcg`, measured at 65.6% of the
//! kernel (same file, line 1226).
//!
//! Two corollaries this file pins, because both have been got wrong here before:
//!
//! * The withdrawn "1.333x 96-vs-128 burst amplification"
//!   (`implementation_notes.md:90-106`) is refuted by the geometry: a 32,256-byte
//!   column is 252 whole 128-byte lines, fully consumed, and the whole tensor is
//!   contiguous, so every byte is fetched exactly once.
//! * The 512 MB arena segment size and cross-tensor interleaving cannot matter,
//!   because a decode GEMV reads every weight byte exactly once per token. There
//!   is no reuse for co-location to protect.
//!
//! # What these tests do
//!
//! * `colmajor_column_is_a_gapless_contiguous_run` and friends **pin the geometry
//!   on the host**, no GPU. They recompute the kernels' address arithmetic and
//!   assert the properties the analysis rests on, so a future edit to
//!   `linear_repacked.metal` that breaks contiguity fails a test instead of
//!   silently invalidating the conclusion.
//! * `row_major_and_col_major_are_the_same_math` proves the two layouts compute
//!   identical results, which is what makes a bandwidth A/B attributable to
//!   layout alone.
//! * `layout_bandwidth_against_a_streaming_ceiling` (`--ignored`, needs the GPU,
//!   ~1.3 GB of buffers) is the measurement: it rotates a working set far larger
//!   than any cache, times both layouts with the real production kernels in ONE
//!   command buffer so the GPU span covers exactly the work, and times a pure
//!   read-only `uint4` stream over comparable bytes as the hardware ceiling for
//!   this working set. It also A/Bs a 128-byte-aligned column base against a
//!   deliberately misaligned one -- the only layout knob the arena leaves open,
//!   since `WeightArena::allocate` bump-allocates with no alignment padding
//!   (`crates/engine/src/weights.rs:180-187`).
//!
//! Run with:
//!   cargo test --release --test perf_probe_layout_dram -- --ignored --nocapture

use half::f16;
use kaplumba_engine::exl3::{pack_integer_states, reorder_tensor_to_colmajor};
use kaplumba_kernels::encoder::MetalEncoder;
use kaplumba_kernels::exl3::{Exl3Codebook, Exl3Rate};
use kaplumba_kernels::linear::{LinearEncodeRequest, LinearKernels};
use metal::{Buffer, ComputePipelineState, Device, MTLResourceOptions};

const BITS: u8 = 3;
/// Bytes per 16x16 tile at 3 bpw: 256 codes * 3 bits / 8. Mirrors `RING / 8`
/// in `linear_repacked.metal`.
const TILE_BYTES: u64 = 96;
/// `WORDS` in `linear_repacked.metal:1983`: `RING / 32`.
const WORDS: u64 = 24;
/// Simdgroups per threadgroup (`SG` in every ishizuki kernel).
const SG: u64 = 8;

// ---------------------------------------------------------------------------
// Host-side geometry. Deliberately a re-implementation of the kernels' address
// arithmetic: the point is to assert what the kernels do, not to call them.
// ---------------------------------------------------------------------------

/// Bytes one threadgroup consumes per unrolled iteration of
/// `linear_gemv_ishizuki_k3_colmajor_contig4`: simdgroup `sg` takes tiles
/// `[4sg, 4sg+4)` (`linear_repacked.metal:2008-2013`) and the 8 simdgroups tile
/// `[0, 32)` exactly once.
fn colmajor_threadgroup_iteration_bytes() -> u64 {
    SG * 4 * TILE_BYTES
}

/// Byte stride between consecutive K-tiles in the row-major kernel
/// (`linear.metal:8741`, `stride = tiles_n * WORDS` words).
fn rowmajor_k_tile_stride(n: usize) -> u64 {
    (n / 16) as u64 * WORDS * 4
}

/// A colmajor column is `tiles_k` tiles at 96 B, no gaps (`linear_repacked.metal:2002`,
/// `column = packed + tn * (tiles_k * WORDS)`).
fn colmajor_column_bytes(k: usize) -> u64 {
    (k / 16) as u64 * TILE_BYTES
}

/// What a single lane loads per tile: `t0[j0]`, `t0[j1]` -- two u32.
const ROWMAJOR_LANE_RUN_BYTES: u64 = 8;

// ---------------------------------------------------------------------------
// Geometry tests. No GPU.
// ---------------------------------------------------------------------------

#[test]
fn colmajor_column_is_a_gapless_contiguous_run() {
    for k in [5376usize, 21504, 1280] {
        let column = colmajor_column_bytes(k);
        assert_eq!(
            column / TILE_BYTES,
            (k / 16) as u64,
            "K={k}: column length disagrees with tile count"
        );
    }
}

#[test]
fn colmajor_columns_are_exact_multiples_of_128_bytes() {
    // This is what kills the withdrawn 1.333x burst-amplification story
    // (`implementation_notes.md:90-106`): a column is a whole number of 128-byte
    // lines, so every fetched line is fully consumed and no byte is fetched
    // twice. It holds because 96 * tiles_k is a multiple of 128 exactly when
    // tiles_k is a multiple of 4, i.e. K is a multiple of 64.
    for k in [5376usize, 21504, 43008] {
        assert_eq!(k % 64, 0, "K={k} is not a multiple of 64");
        assert_eq!(
            colmajor_column_bytes(k) % 128,
            0,
            "K={k}: column of {} B is not a whole number of 128 B lines",
            colmajor_column_bytes(k)
        );
    }
    // The whole tensor is contiguous, so adjacent columns share no lines and the
    // byte count is exactly the packed size.
    let (k, n) = (5376usize, 21504usize);
    assert_eq!(
        colmajor_column_bytes(k) * (n / 16) as u64,
        (k * n * 3 / 8) as u64,
        "column arithmetic lost or invented bytes"
    );
}

#[test]
fn contig4_partitions_the_column_without_gap_or_overlap() {
    let span = colmajor_threadgroup_iteration_bytes();
    assert_eq!(
        span, 3072,
        "per-iteration span changed; re-derive the analysis"
    );
    assert_eq!(span % 128, 0);

    // Model both loops of `linear_gemv_ishizuki_k3_colmajor_contig4` as written:
    // the unrolled loop (`:2009`, `tk = sg*4`, `tk += 32`, four tiles at a time)
    // and the tail (`:2041`, `tk = k_unroll_end + sg`, `tk += 8`, one tile).
    // The tail starts from `k_unroll_end + sg`, NOT from wherever the unrolled
    // loop's own `tk` stopped, so the two regions must partition [0, tiles_k).
    let k = 5376usize;
    let tiles_k = (k / 16) as u64;
    let step_tiles = SG * 4;
    let k_unroll_end = (tiles_k / step_tiles) * step_tiles;
    assert_eq!(k_unroll_end, 320, "K={k} tiles_k={tiles_k}");

    let mut visits = vec![0u32; tiles_k as usize];
    for sg in 0..SG {
        let mut tk = sg * 4;
        while tk < k_unroll_end {
            for q in 0..4 {
                visits[(tk + q) as usize] += 1;
            }
            tk += step_tiles;
        }
    }
    for sg in 0..SG {
        let mut tk = k_unroll_end + sg;
        while tk < tiles_k {
            visits[tk as usize] += 1;
            tk += SG;
        }
    }
    assert!(
        visits.iter().all(|&v| v == 1),
        "a K-tile is visited more than once, or not at all: {:?}",
        visits
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v != 1)
            .map(|(i, _)| i)
            .collect::<Vec<_>>()
    );

    // Honest caveat the iteration figure above hides: the unrolled region is
    // 3072 contiguous bytes per step, but the 16-tile tail
    // (`tiles_k % step_tiles == 16`) is one tile per simdgroup at `SG`-tile
    // spacing, i.e. eight 96-byte reads 768 B apart. That is 1,536 of the
    // column's 32,256 bytes -- 4.8% -- and it is the only part of a colmajor
    // column that is not a gapless sequential run.
    let tail_bytes = (tiles_k - k_unroll_end) * TILE_BYTES;
    assert_eq!(tail_bytes, 1536);
    assert_eq!(tail_bytes * 100 / colmajor_column_bytes(k), 4); // floor of 4.75%
}

#[test]
fn row_major_walks_a_huge_stride_per_short_run() {
    // The shape in the brief: K=5376, N=21504 (gate/up_proj).
    let (k, n) = (5376usize, 21504usize);
    let stride = rowmajor_k_tile_stride(n);
    assert_eq!(n / 16, 1344, "tiles_n for N=21504");
    assert_eq!(stride, 129_024, "row-major K stride changed");
    assert_eq!(
        stride / TILE_BYTES,
        1344,
        "the stride is no longer tiles_n tiles"
    );
    // A lane reads 8 bytes and then jumps 129,024: a 1:16,128 stride/run ratio.
    assert!(
        stride > ROWMAJOR_LANE_RUN_BYTES * 1000,
        "row-major is supposed to be the strided layout; it is not"
    );

    // The 8 simdgroups of one threadgroup sit `SG` tiles apart in K
    // (`linear.metal:8742`), so a threadgroup's instantaneous footprint is 8
    // separate 96-byte tiles `stride` apart, spanning 7 * stride + 96.
    assert_eq!(7 * stride + TILE_BYTES, 7 * 129_024 + 96);

    // down_proj is the transposed shape (K=21504, N=5376). The stride is
    // `tiles_n` tiles, so it scales with N, not with K.
    assert_eq!(rowmajor_k_tile_stride(5376), 32_256);

    // Both layouts must describe the same byte count.
    assert_eq!(
        (k * n * 3 / 8) as u64,
        colmajor_column_bytes(k) * (n / 16) as u64,
        "row- and column-major must describe the same byte count"
    );
    assert_eq!(
        (21504 * 5376 * 3 / 8) as u64,
        colmajor_column_bytes(21504) * (5376 / 16) as u64
    );
}

/// A lane's window. `j0` from `linear_repacked.metal:1992-1997`:
/// `lo0 = (lane*8+1)*3 + RING - 16`, `hi0 = lo0 / 32`, `j0 = hi0 mod WORDS`.
///
/// The 32 lanes of a simdgroup must cover all 24 words of the 96-byte tile: if
/// they do not, some codes are never fetched, and if they cover much more, the
/// redundancy has escaped L1 and become real DRAM traffic.
#[test]
fn a_simdgroup_covers_exactly_one_tile() {
    const RING: u32 = 256 * 3;
    let mut covered = [false; WORDS as usize];
    let mut reads = 0u32;
    for lane in 0..32u32 {
        let lo0 = (lane * 8 + 1) * 3 + RING - 16;
        let hi0 = lo0 / 32;
        let words_u32 = WORDS as u32;
        let j0 = if hi0 >= words_u32 { hi0 - words_u32 } else { hi0 };
        let j1 = if j0 + 1 >= words_u32 {
            j0 + 1 - words_u32
        } else {
            j0 + 1
        };
        assert!(j0 < words_u32 && j1 < words_u32, "lane {lane} left the tile");
        covered[j0 as usize] = true;
        covered[j1 as usize] = true;
        reads += 2;
    }
    assert!(
        covered.iter().all(|&c| c),
        "the 32 lanes do not cover all {WORDS} words of the 96-byte tile"
    );
    // 64 word-reads over 24 distinct words inside one tile: 2.67x intra-tile
    // redundancy, all of it inside a span L1 serves. Nothing here reaches DRAM.
    assert_eq!(reads, 64);
    assert_eq!(reads as u64 * 4, 256);
    assert_eq!(TILE_BYTES, 96);
}

// ---------------------------------------------------------------------------
// GPU: correctness of the A/B, then the measurement.
// ---------------------------------------------------------------------------

fn build_rowmajor(in_f: usize, out_f: usize, seed: u32) -> Vec<u16> {
    let tiles = (in_f / 16) * (out_f / 16);
    let mut packed = Vec::with_capacity(tiles * 48);
    let mut s = seed;
    for _ in 0..tiles {
        let mut states = [0u16; 256];
        for st in &mut states {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *st = (s & ((1u32 << BITS) - 1)) as u16;
        }
        let (w, _) = pack_integer_states(&states, BITS).expect("pack");
        packed.extend(w);
    }
    assert_eq!(packed.len(), tiles * 48, "row-major tile is not 48 u16");
    packed
}

fn upload(device: &Device, v: &[u16]) -> Buffer {
    device.new_buffer_with_data(
        v.as_ptr() as *const std::ffi::c_void,
        (v.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

struct Sig {
    input: Buffer,
    input_signs: Buffer,
    output_signs: Buffer,
    transformed: Buffer,
    raw: Buffer,
    output: Buffer,
}

fn sig(device: &Device, in_f: usize, out_f: usize) -> Sig {
    let input_v: Vec<u16> = (0..in_f)
        .map(|i| f16::from_f32((i as f32 * 0.011).sin()).to_bits())
        .collect();
    let ones = upload(device, &vec![f16::from_f32(1.0).to_bits(); in_f.max(1)]);
    Sig {
        input: upload(device, &input_v),
        input_signs: ones,
        output_signs: upload(device, &vec![0u16; out_f]),
        transformed: device.new_buffer((in_f * 2) as u64, MTLResourceOptions::StorageModeShared),
        raw: device.new_buffer((out_f * 4) as u64, MTLResourceOptions::StorageModeShared),
        output: device.new_buffer((out_f * 2) as u64, MTLResourceOptions::StorageModeShared),
    }
}

/// Encode into the caller's encoder. Deliberately does *not* commit: the timing
/// harness needs every rep inside one command buffer, and a per-rep commit would
/// make it measure launch latency instead of a GPU span.
fn encode_gemv(
    kernels: &LinearKernels,
    enc: &mut MetalEncoder,
    kernel: &str,
    packed: &Buffer,
    packed_offset: u64,
    in_f: usize,
    out_f: usize,
    s: &Sig,
) {
    kernels
        .encode_gemv_named(
            enc,
            kernel,
            &LinearEncodeRequest {
                packed,
                packed_offset,
                perm_mode: 1,
                input: &s.input,
                input_signs: &s.input_signs,
                output_signs: &s.output_signs,
                transformed: &s.transformed,
                raw: &s.raw,
                output: &s.output,
                in_features: in_f,
                out_features: out_f,
                rate: Exl3Rate::Integer(3),
                codebook: Exl3Codebook::Mcg,
                pretransformed: false,
            },
        )
        .unwrap_or_else(|e| panic!("{kernel} {in_f}x{out_f} @{packed_offset}: {e}"));
}

fn run_once(
    kernels: &LinearKernels,
    queue: &metal::CommandQueue,
    kernel: &str,
    packed: &Buffer,
    packed_offset: u64,
    in_f: usize,
    out_f: usize,
    s: &Sig,
) -> Vec<u16> {
    let mut enc = MetalEncoder::new(queue).expect("encoder");
    encode_gemv(kernels, &mut enc, kernel, packed, packed_offset, in_f, out_f, s);
    enc.commit_and_wait().expect("wait");
    let p = s.output.contents() as *const u16;
    (0..out_f).map(|i| unsafe { p.add(i).read() }).collect()
}

/// The A/B is only meaningful if both layouts compute the same thing. They
/// differ only in tile order, so they must agree bit-for-bit on the fp16 output
/// the sampler reads. If this fails, a bandwidth comparison between them is
/// measuring a correctness difference, not a layout difference.
#[test]
fn row_major_and_col_major_are_the_same_math() {
    let Some(device) = Device::system_default() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let kernels = LinearKernels::new().expect("kernels compile");
    let queue = device.new_command_queue().to_owned();

    // One small shape for speed, then two production shapes.
    for (in_f, out_f) in [(1280usize, 1152usize), (5376, 5376), (5376, 21504)] {
        assert_eq!(in_f % 128, 0, "in_features must be a multiple of 128");
        assert_eq!(out_f % 128, 0, "out_features must be a multiple of 128");
        let row = build_rowmajor(in_f, out_f, 0x51ed_0000 ^ in_f as u32);
        let col = reorder_tensor_to_colmajor(&row, in_f, out_f).expect("colmajor");
        assert_eq!(row.len(), col.len());
        let s = sig(&device, in_f, out_f);

        let row_buf = upload(&device, &row);
        let row_out = run_once(
            &kernels,
            &queue,
            "linear_gemv_ishizuki_k3_w2",
            &row_buf,
            0,
            in_f,
            out_f,
            &s,
        );

        let col_buf = upload(&device, &col);
        let col_out = run_once(
            &kernels,
            &queue,
            "linear_gemv_ishizuki_k3_colmajor_contig4",
            &col_buf,
            0,
            in_f,
            out_f,
            &s,
        );

        let diffs = row_out.iter().zip(&col_out).filter(|(a, b)| a != b).count();
        assert_eq!(
            diffs, 0,
            "{in_f}x{out_f}: row-major and col-major disagree on {diffs}/{out_f} \
             outputs, so a bandwidth A/B between them would not be about layout"
        );
    }
}

// ---------------------------------------------------------------------------
// The measurement.
// ---------------------------------------------------------------------------

/// Read-only streaming reference: the ceiling this working set delivers, with no
/// dequantization and no layout. Compiled from a source string so this probe
/// adds no kernel to the shipped library.
const STREAM_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Block-contiguous, like a colmajor column: each thread owns one long
// sequential run rather than a grid-stride slice.
kernel void stream_read(const device uint4 *data [[buffer(0)]],
                        device uint4 *out [[buffer(1)]],
                        constant uint &count [[buffer(2)]],
                        uint gid [[thread_position_in_grid]],
                        uint total [[threads_per_grid]]) {
    uint4 sum = uint4(0);
    uint per = count / total;
    uint start = gid * per;
    uint end = min(start + per, count);
    for (uint i = start; i < end; ++i) { sum += data[i]; }
    out[gid] = sum;
}
"#;

const STREAM_THREADS: u64 = 1 << 16;
/// Distinct bytes read per rep by the GEMV arms: far above any plausible L2
/// (M4 Pro's SLC is single-digit MB), so nothing is served from cache.
const WORKING_SET: usize = 768 << 20;
/// GPU time per point, so clock ramp is a rounding error.
const TARGET_MS: f64 = 60.0;

fn stream_pipeline(device: &Device) -> ComputePipelineState {
    let lib = device
        .new_library_with_source(STREAM_SRC, &metal::CompileOptions::new())
        .expect("stream library compiles");
    let f = lib.get_function("stream_read", None).expect("stream_read");
    device
        .new_compute_pipeline_state_with_function(&f)
        .expect("stream pipeline")
}

struct Reading {
    gbps: f64,
    gpu_ms: f64,
    reps: usize,
}

/// Time `reps` dispatches that each stream `shape_bytes` from a rotating set,
/// on the GPU's own clock.
///
/// Calibrate, warm, then encode every rep into ONE command buffer
/// (`set_max_dispatches(u32::MAX)`) so the stamped span covers exactly the work.
/// Per-rep commits here would measure launch latency: `gemv_bandwidth.rs:176-187`
/// records that this exact mistake reports impossible bandwidth, and
/// `gemv_bandwidth.rs:12-19` records the wall-clock version measuring the CPU.
fn measure(
    queue: &metal::CommandQueue,
    label: &str,
    shape_bytes: usize,
    encode: impl Fn(&mut MetalEncoder, usize),
) -> Reading {
    let mut probe = MetalEncoder::new(queue).expect("encoder");
    probe.set_max_dispatches(u32::MAX);
    let t0 = std::time::Instant::now();
    for i in 0..4 {
        encode(&mut probe, i);
    }
    probe.commit_and_wait().expect("cal");
    let per = (t0.elapsed().as_secs_f64() * 1e3 / 4.0).max(1e-4);
    let reps = ((TARGET_MS / per).round() as usize).clamp(4, 4000);

    {
        let mut enc = MetalEncoder::new(queue).expect("encoder");
        enc.set_max_dispatches(u32::MAX);
        for i in 0..reps.min(8) {
            encode(&mut enc, i);
        }
        enc.commit_and_wait().expect("warmup");
    }

    let mut enc = MetalEncoder::new(queue).expect("encoder");
    enc.set_max_dispatches(u32::MAX);
    for i in 0..reps {
        encode(&mut enc, i);
    }
    enc.commit_stamped(label).expect("commit");
    let gpu_ms: f64 = enc.take_gpu_stamps().iter().map(|s| s.own_ms).sum();
    enc.commit_and_wait().ok();

    assert!(
        gpu_ms > 0.0,
        "{label}: no GPU timing came back (the CommandBufferGpuTiming trait is \
         not bound in this build) -- the ratio printed below would be meaningless"
    );
    Reading {
        gbps: (shape_bytes as f64 * reps as f64) / (gpu_ms * 1e-3) / 1e9,
        gpu_ms,
        reps,
    }
}

/// Achieved GB/s per layout, against a pure read-only stream.
///
/// What it decides:
///
/// * If row-major and column-major land close together, the strided layout is
///   not costing what its geometry suggests, and DRAM locality is not what
///   separates this kernel from 236.7 GB/s.
/// * If both land far above the streaming ceiling, the ceiling probe is broken
///   (not enough concurrency, or `per` too small) and its number is not a
///   ceiling. Check this before believing any ratio below.
/// * Either way, the ceiling number is the number to compare 236.7 against. If
///   the ceiling is below 236.7, the target is out of reach on this machine for
///   reasons no layout can fix; if it is above, the kernel is leaving the
///   available bandwidth on the table for a reason that arm (d) localises to
///   the ALU.
#[test]
#[ignore = "microbenchmark; needs the GPU and ~1.3 GB of buffers. Run explicitly"]
fn layout_bandwidth_against_a_streaming_ceiling() {
    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();
    let kernels = LinearKernels::new().expect("kernels compile");

    // The shape in the brief: 5376 in, 21504 out (gate/up_proj).
    let (k, n) = (5376usize, 21504usize);
    let shape_bytes = (k * n * 3 / 8) as usize;
    assert_eq!(shape_bytes as u64, colmajor_column_bytes(k) * (n / 16) as u64);

    let row = build_rowmajor(k, n, 0xa11ce);
    let col = reorder_tensor_to_colmajor(&row, k, n).expect("colmajor");
    let s = sig(&device, k, n);

    let nbuf = (WORKING_SET / shape_bytes).clamp(2, 6);
    println!(
        "\nshape {k}x{n} = {:.1} MB at 3 bpw, {nbuf} rotating buffers ({:.0} MB)",
        shape_bytes as f64 / 1e6,
        (nbuf * shape_bytes) as f64 / 1e6
    );
    assert!(nbuf >= 2, "a single buffer would report cache bandwidth");

    let row_bufs: Vec<Buffer> = (0..nbuf).map(|_| upload(&device, &row)).collect();
    let col_bufs: Vec<Buffer> = (0..nbuf).map(|_| upload(&device, &col)).collect();

    let mut readings: Vec<(&str, Reading)> = Vec::new();

    let row_r = measure(&queue, "row-major", shape_bytes, |enc, i| {
        encode_gemv(
            &kernels,
            enc,
            "linear_gemv_ishizuki_k3_w2",
            &row_bufs[i % nbuf],
            0,
            k,
            n,
            &s,
        )
    });
    readings.push(("row-major  (_w2)", row_r));

    let col_r = measure(&queue, "colmajor", shape_bytes, |enc, i| {
        encode_gemv(
            &kernels,
            enc,
            "linear_gemv_ishizuki_k3_colmajor_contig4",
            &col_bufs[i % nbuf],
            0,
            k,
            n,
            &s,
        )
    });
    readings.push(("colmajor (contig4)", col_r));

    // Alignment arm: the only layout knob the arena leaves open. Metal aligns
    // buffers to at least 256 B, so a tensor at `packed_offset == 0` has
    // 128-aligned column bases; 64 bytes of leading pad makes every column base
    // 128-unaligned, which is exactly what a bump-allocated arena tensor with an
    // arbitrary offset produces.
    let mut padded = vec![0u16; 32];
    padded.extend_from_slice(&col);
    let mis_buf = upload(&device, &padded);
    assert_eq!(
        mis_buf.contents() as usize % 128,
        0,
        "Metal must hand back a 128-aligned buffer for this arm to mean anything"
    );
    let mis_r = measure(&queue, "colmajor-misaligned", shape_bytes, |enc, _| {
        encode_gemv(
            &kernels,
            enc,
            "linear_gemv_ishizuki_k3_colmajor_contig4",
            &mis_buf,
            64,
            k,
            n,
            &s,
        )
    });
    readings.push(("colmajor misaligned", mis_r));
    drop(padded);

    // The ceiling: read bytes and add them, same shape of work, no layout.
    let stream_bytes = WORKING_SET;
    let nvec = stream_bytes / 16;
    let words: Vec<u32> = (0..(nvec * 4) as u64).map(|i| i as u32).collect();
    let dbuf = device.new_buffer_with_data(
        words.as_ptr() as *const std::ffi::c_void,
        (words.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    drop(words);
    let obuf = device.new_buffer(
        STREAM_THREADS * 16,
        MTLResourceOptions::StorageModeShared,
    );
    let pipe = stream_pipeline(&device);
    let count = (nvec as u32).to_le_bytes();
    let ceil = measure(&queue, "stream", stream_bytes, |enc, _| {
        enc.dispatch_1d(
            &pipe,
            &[Some(&dbuf), Some(&obuf)],
            &count,
            STREAM_THREADS,
            256,
        )
        .expect("stream dispatch");
    });

    println!("\n=== achieved ===");
    for (label, r) in readings.iter() {
        println!(
            "{label:<22} reps={:<5} gpu={:8.2} ms  = {:7.1} GB/s",
            r.reps, r.gpu_ms, r.gbps
        );
    }
    println!(
        "{:<22} reps={:<5} gpu={:8.2} ms  = {:7.1} GB/s   (ceiling)",
        "pure uint4 read",
        ceil.reps,
        ceil.gpu_ms,
        ceil.gbps
    );

    let (row_gbps, col_gbps, mis_gbps) = (
        readings[0].1.gbps,
        readings[1].1.gbps,
        readings[2].1.gbps,
    );
    println!("\n=== ratios ===");
    println!("colmajor / row-major     = {:.3}", col_gbps / row_gbps);
    println!("colmajor / streaming     = {:.3}", col_gbps / ceil.gbps);
    println!("row-major  / streaming   = {:.3}", row_gbps / ceil.gbps);
    println!("misaligned / aligned col = {:.3}", mis_gbps / col_gbps);
    println!(
        "\ntarget 236.7 GB/s is {:.3}x the measured ceiling of {:.1} GB/s \
         (273 GB/s wire = {:.3}x of wire).",
        236.7 / ceil.gbps,
        ceil.gbps,
        ceil.gbps / 273.0
    );
    println!(
        "Read this alongside arm (d) in \
         spec/decode-19tok-repacked-250gb/implementation_notes.md:1199 -- loads \
         kept,\naddresses kept, 2.6 GB of DRAM swapped for a 129 KB L2 window \
         for -0.8%.\nIf that holds, no layout here can reach the target, and \
         the wall is `decode_state_mcg`'s\n64-bit variable barrel shift \
         (65.6% of the kernel), not DRAM."
    );
}