//! Lever 1 for `spec/prefill-24k-90tps`: **the operand-issue finding.** The
//! flash prefill kernel issues **two fragment loads per MMA** in its QK phase
//! (`kf` from device + `qf` from `qstage`) against one per MMA in PV, and
//! `pragma_peak_report.md` T6 measures all four fragment loads together at
//! **52.4%** of the kernel with the MMA count untouched.
//!
//! This instrument builds, in **one process and one shader library per arm**,
//! variants of the QK loop that change the *loads-per-MMA* ratio, and checks
//! every arm is **bit-identical** to the shipping kernel before it prints a
//! single timing.
//!
//! # The mechanism
//!
//! `Q` does not change across the key loop: for a given `d8` the `qf` fragment
//! is the same 128 B in every key tile. It is therefore loop-invariant, and at
//! `BQ = 1` it is loaded once and consumed by exactly **one** MMA -- two
//! fragment loads per MMA, against one per MMA in PV.
//!
//! Give each simdgroup **MSUB** m-tiles (`sg`, `sg + NSUB`, `sg + 2*NSUB`, ...)
//! over a `MSUB * 64`-key QK window and one `qf` feeds MSUB MMAs:
//!
//! | MSUB | QK window | `sth` | loads/MMA in QK | whole-kernel loads/MMA |
//! |---|---|---|---|---|
//! | 1 (shipping) | 64 keys | 2 KB | 2.000 | 1.563 |
//! | 2 | 128 keys | 4 KB | 1.500 | 1.313 |
//! | 4 | 256 keys | 8 KB | 1.250 | 1.188 |
//!
//! **The loop nest has to be swapped, and that is the whole trick.** Widening the
//! QK window with the shipping order -- `for m-tile { for d8 { load qf; MMA } }`
//! -- leaves `qf` used once per load and changes the loads/MMA by **nothing**.
//! Only `for d8 { load qf once; for m-tile { load kf; MMA } }` reuses it. That
//! costs MSUB live accumulators (2 registers/lane each: 8 at MSUB = 4, against
//! the 16 `acc[BQ][NPER]` already holds) and it does **not** hoist Q into
//! registers -- that was `AblationScaffold`'s arm 7 at -92%, 64 registers/lane of
//! `qreg`.
//!
//! **Why it is bit-identical and not merely close.** The QK reduction order per
//! accumulator is unchanged (`d8 = 0, 8, ... HEAD_DIM-8`, one accumulator per
//! m-tile), the `kf`/`qf` *values* are unchanged, and the softmax and PV still run
//! per 64-key half in the shipping order, so the rescale points, the running
//! `(m, s)` sequence and every `wq` write are exactly the shipping ones. The
//! `msub1` arm is the refactor's own control: it is the shipping arithmetic
//! through this new code, and it must also be byte-identical.
//!
//! # The `qt` arms
//!
//! `qf` is read with `trans = true` from `qstage` at stride `HEAD_DIM`, i.e. 32
//! lanes gather 16 B each from 8 rows 1 KB apart. Staging Q **transposed** makes
//! it 128 B contiguous and drops the transpose, at the cost of a scattered
//! one-time staging write. Pure data movement: bit-identical, and it is the
//! "stage fragments so the load is cheap" option in its only form that does not
//! pay in registers.
//!
//! # Controls, asserted not trusted
//!
//! `spin` (a 3 000 000-iteration threadgroup spin, added at the top of the body)
//! must move the timer by >1.5x or every ratio here is void. `empty`
//! (`if (tid < 256u) { return; }` in a 256-thread dispatch, i.e. a genuinely
//! empty kernel -- §10's arm) must collapse below 5% of the baseline or this is
//! not timing the body.
//!
//! Interleaved A/B/A/B, `ROUNDS` rounds, medians with min/max, ratios taken
//! within one process only (cross-run drift on this box is ~14% on an unmodified
//! kernel, `pragma_peak_report.md` T5).
//!
//! ```text
//! cargo run --release --example flash_operand_issue
//! KAPLUMBA_ARMS=msub1,msub2,spin,empty KAPLUMBA_KEYS=24576 KAPLUMBA_BQ=1
//! KAPLUMBA_DUMP=dir   # write each arm's shader source out for diffing
//! ```

use half::f16;
use metal::{Buffer, Device, Library, MTLResourceOptions, MTLSize};
use std::path::PathBuf;
use std::time::Instant;

/// The shipping shader. `include_str!` so a `.metal` edit rebuilds this example
/// (cargo tracks `include_str!` inputs -- `implementation_notes.md` §22.1b).
const SHIPPING: &str = include_str!("../../kernels/src/attention.metal");

/// The restructured key-tile loop, shared by every `msub`/`qt` arm. It replaces
/// the shipping block from `for (uint base = ...; base += TILE)` up to (not
/// including) the epilogue, and is written so that `MSUB == 1` is the shipping
/// arithmetic verbatim.
const MSUB_BLOCK: &str = include_str!("blocks/flash_qk_msub_block.metal");

// ---------------------------------------------------------------------------
// anchors. Every one must match exactly the stated number of times, or the
// instrument aborts: an unmatched anchor would leave an arm byte-identical to
// the shipping kernel and produce a beautiful, meaningless x1.000 row.
// ---------------------------------------------------------------------------

/// Start of the block the `msub` arms replace.
const LOOP_START: &str = "    for (uint base = range_begin; base < range_end; base += TILE) {";
/// End of the block: the epilogue's own comment. The replaced region is
/// everything strictly between `LOOP_START` and this.
const LOOP_END: &str = "    // Decode: emit the unnormalised `(m, s, O)` partial for this split in the";
/// The `TILE` declaration the `msub` arms extend with the QK span.
const TILE_DECL: &str = "    constexpr uint TILE = 64u;";
/// The body template's parameter list.
const TEMPLATE_DECL: &str =
    "template <int HEAD_DIM, int NSG, int BQ, bool PARTIALS, bool PASSES>";
/// The body template's invocation from the prefill prefill-macro (2 sites: the
/// one-dispatch macro and the OQ9 pass macro).
const CALL_PLAIN: &str = "attention_prefill_flash_body<HEAD_DIM, NSG, BQ, false, false>(";
const CALL_PASSES: &str = "attention_prefill_flash_body<HEAD_DIM, NSG, BQ, false, true>(";
/// The decode (PARTIALS) invocation, which stays at `MSUB = 1`: it writes
/// `(m, s, O)` partials for one query, so a wider QK span buys it nothing.
const CALL_DECODE: &str = "attention_prefill_flash_body<HEAD_DIM, NSG, 1, true, false>(";
/// `sth`'s declaration, in both prefill macros. `64` is the TILE-key width.
const STH_DECL: &str = "threadgroup float sth[(BQ) * 64 * 8];";
/// The Q staging loop.
const Q_STAGE: &str = "\
        for (uint c = lane * 8u; c < uint(HEAD_DIM); c += 256u) {
            *reinterpret_cast<threadgroup uint4 *>(qdst + c) =
                *reinterpret_cast<const device uint4 *>(qsrc + c);
        }";
/// The Q fragment load, **as `MSUB_BLOCK` writes it**. Deliberately not the
/// shipping indentation: the `qt` arms replace the block before this anchor is
/// applied, and an anchor written against the shipping text would match zero
/// times and abort (which is the intended behaviour for a drifted anchor, but
/// it is not what should happen on a fresh edit).
const QF_LOAD: &str = "\
                    simdgroup_half8x8 qf;
                    if ((ABL & 2u) != 0u) {
                        qf = make_filled_simdgroup_matrix<half, 8, 8>(half(0.5f));
                    } else {
                        simdgroup_load(qf,
                                       qstage + (b * uint(NSG)) * uint(HEAD_DIM) + d8,
                                       HEAD_DIM, ulong2(0, 0), true);
                    }";
/// Positive control.
const BODY_ANCHOR: &str = "    constexpr float NEG = -1.701411733864e+38f;\n    constexpr uint TILE = 64u;";

/// Lines in the shipping region between [`LOOP_START`] and [`LOOP_END`]. A
/// tripwire on the replacement staying pointed at the same block: if the
/// shipping loop is edited, this fails rather than silently replacing the wrong
/// span. Checked by hand against `attention.metal` and re-checked whenever it
/// fires.
const REPLACED_LINES: usize = 192;

/// What an arm's output is expected to do, and therefore how it is judged.
#[derive(Clone, Copy, PartialEq)]
enum Expectation {
    /// Bit-identical to the shipping kernel.
    Exact,
    /// Must differ: real data replaced by constants, or work removed.
    Differs,
}

fn sub_exact(src: &str, find: &str, replacement: &str, want: usize) -> String {
    let hits = src.matches(find).count();
    assert_eq!(hits, want, "anchor occurs {hits} times, expected {want}: {find:?}");
    src.replace(find, replacement)
}

/// What an arm does, and therefore whether its output must match the shipping
/// kernel byte for byte.
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    /// The shipping source, verbatim.
    Baseline,
    /// The restructured QK loop at `MSUB` m-tiles per simdgroup, 1..=4.
    Msub(u32),
    /// `MSUB` m-tiles plus transposed Q staging (contiguous `qf` load).
    Qt(u32),
    /// Positive control: a spin the instrument must resolve.
    Spin,
    /// `if (tid < 256u) { return; }` in a 256-thread dispatch: an empty kernel.
    Empty,
    /// The restructured loop with an **ablation mask**, for the marginal-cost
    /// question "is it the loads or the MMAs?". Bits:
    ///
    /// | bit | removes | keeps |
    /// |---|---|---|
    /// | 1 | the QK MMAs | the QK `kf`/`qf` loads |
    /// | 2 | the QK `kf`/`qf` loads (filled constant) | the QK MMAs |
    /// | 4 | the PV MMAs | the PV `af`/`bf` loads |
    /// | 8 | the PV `af`/`bf` loads (filled constant) | the PV MMAs |
    ///
    /// These are the arms `pragma_peak_report.md` T6 read as a **share** and
    /// then had to reconcile with arms that do not sum. Read here as a
    /// **marginal cost**: removing the loads while holding the MMA count fixed
    /// is the cost of feeding the tensor pipe, which is the number that decides
    /// whether raising MMAs-per-load (the `MSUB` arms) can pay at all.
    Abl(u32),
}

#[derive(Clone, Copy)]
struct Arm {
    name: &'static str,
    kind: Kind,
}

fn arms() -> Vec<Arm> {
    let all = vec![
        Arm { name: "shipped", kind: Kind::Baseline },
        Arm { name: "msub1", kind: Kind::Msub(1) },
        Arm { name: "msub2", kind: Kind::Msub(2) },
        Arm { name: "msub4", kind: Kind::Msub(4) },
        Arm { name: "qt1", kind: Kind::Qt(1) },
        Arm { name: "qt2", kind: Kind::Qt(2) },
        Arm { name: "qt4", kind: Kind::Qt(4) },
        Arm { name: "abl-noloads", kind: Kind::Abl(2 | 8) },
        Arm { name: "abl-qk-mma", kind: Kind::Abl(1) },
        Arm { name: "abl-pv-mma", kind: Kind::Abl(4) },
        Arm { name: "abl-no-mma", kind: Kind::Abl(1 | 4) },
        Arm { name: "spin", kind: Kind::Spin },
        Arm { name: "empty", kind: Kind::Empty },
    ];
    let want = std::env::var("KAPLUMBA_ARMS").ok();
    match want {
        None => all,
        Some(spec) => {
            let picked: Vec<Arm> = spec
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(|token| {
                    all.iter()
                        .find(|a| a.name == token)
                        .unwrap_or_else(|| panic!("unknown arm {token:?}"))
                        .clone()
                })
                .collect();
            assert!(!picked.is_empty(), "KAPLUMBA_ARMS selected nothing");
            picked
        }
    }
}

impl Arm {
    /// The arm's shader source. `shipped` returns `SHIPPING` verbatim.
    fn source(&self) -> String {
        let (msub, abl) = match self.kind {
            Kind::Baseline => return SHIPPING.to_string(),
            Kind::Spin | Kind::Empty => (1, 0),
            Kind::Msub(n) | Kind::Qt(n) => (n, 0),
            Kind::Abl(a) => (1, a),
        };
        assert!((1..=4).contains(&msub), "MSUB must be 1..=4, got {msub}");
        assert!(abl <= 0xF, "the ablation mask is 4 bits, got {abl}");

        // ---- 1. threadgroup memory: `sth` spans the QK window ---------------
        // `wq`, `tail`, `qstage` are unchanged: `wq` holds one TILE-key half at
        // a time (the halves are consumed in order and separated by barriers)
        // and `tail` still stages the single straddling m-tile of the span,
        // because only the *last* live m-tile can straddle.
        let mut out = sub_exact(SHIPPING, STH_DECL, "threadgroup float sth[(BQ) * 64 * 8 * KAPLUMBA_MSUB];", 2);

        // ---- 2. the `MSUB` constant, and the body template ------------------
        out = sub_exact(
            &out,
            "using namespace metal;",
            "using namespace metal;\n\n// Instrument: QK m-tiles per simdgroup. See flash_operand_issue.rs.\n#define KAPLUMBA_MSUB 1\n// Instrument: ablation mask (1 QK MMA, 2 QK loads, 4 PV MMA, 8 PV loads).\n#define KAPLUMBA_ABL 0u",
            1,
        );
        out = sub_exact(
            &out,
            TEMPLATE_DECL,
            "template <int HEAD_DIM, int NSG, int BQ, bool PARTIALS, bool PASSES, int MSUB, uint ABL>",
            1,
        );
        out = sub_exact(&out, CALL_PLAIN, "attention_prefill_flash_body<HEAD_DIM, NSG, BQ, false, false, KAPLUMBA_MSUB, KAPLUMBA_ABL>(", 1);
        out = sub_exact(&out, CALL_PASSES, "attention_prefill_flash_body<HEAD_DIM, NSG, BQ, false, true, KAPLUMBA_MSUB, KAPLUMBA_ABL>(", 1);
        // Decode keeps MSUB = 1 and ABL = 0: it emits one query's partial, so a
        // wider QK window has nothing to amortise, and the ablations must never
        // reach the decode path.
        out = sub_exact(&out, CALL_DECODE, "attention_prefill_flash_body<HEAD_DIM, NSG, 1, true, false, 1, 0u>(", 1);
        out = sub_exact(&out, "#define KAPLUMBA_MSUB 1", &format!("#define KAPLUMBA_MSUB {msub}u"), 1);
        out = sub_exact(&out, "#define KAPLUMBA_ABL 0u", &format!("#define KAPLUMBA_ABL {abl}u"), 1);

        // ---- 3. the key-tile loop itself ------------------------------------
        out = sub_exact(
            &out,
            TILE_DECL,
            "    constexpr uint TILE = 64u;\n    // QK window: MSUB tiles of TILE keys, so each simdgroup owns MSUB\n    // m-tiles and one `qf` fragment feeds MSUB MMAs. The softmax and PV still\n    // run per TILE keys, in order, so the arithmetic is the shipping one.\n    constexpr uint TILEQ = uint(MSUB) * TILE;\n    constexpr uint NSUBQ = uint(MSUB) * NSUB;",
            1,
        );
        let start = out
            .find(LOOP_START)
            .unwrap_or_else(|| panic!("LOOP_START not found"));
        assert_eq!(out.matches(LOOP_START).count(), 1, "LOOP_START is not unique");
        let end = out
            .find(LOOP_END)
            .unwrap_or_else(|| panic!("LOOP_END not found"));
        assert!(end > start, "LOOP_END precedes LOOP_START");
        let replaced = &out[start..end];
        // The replaced region must be the shipping loop, not something else:
        // these four markers are what makes the span identifiable.
        for marker in [
            "const uint partial_sub =",
            "sth[(b * TILE + sg * 8u + fm) * 8u + fn]",
            "const uint ksteps = (live_in_tile + 7u) / 8u;",
            "simdgroup_multiply_accumulate(acc[b][j], af[b], bf, acc[b][j]);",
        ] {
            assert!(
                replaced.contains(marker),
                "the region being replaced does not contain {marker:?}; the anchor is \
                 pointing at the wrong block"
            );
        }
        assert_eq!(
            replaced.lines().count(),
            REPLACED_LINES,
            "the region being replaced has {} lines, expected {REPLACED_LINES}: the shipping loop \
             changed and this instrument's block is no longer its replacement",
            replaced.lines().count()
        );
        let mut next = String::with_capacity(out.len() + MSUB_BLOCK.len());
        next.push_str(&out[..start]);
        next.push_str(MSUB_BLOCK);
        next.push_str(&out[end..]);
        let mut out = next;

        // ---- 4. the `qt` arms: transposed Q staging ------------------------
        if matches!(self.kind, Kind::Qt(_)) {
            out = sub_exact(
                &out,
                Q_STAGE,
                "\
        for (uint c = lane * 8u; c < uint(HEAD_DIM); c += 256u) {
            // Transposed: `qstage` becomes [query][dim * 8 + head], so the `qf`
            // fragment (dim x head) is 128 B contiguous and needs no transpose on
            // load. The slot index is this simdgroup's own head, `sg`, which is
            // why every head stages into the SAME region rather than its own.
            threadgroup half *qt = qstage + b * uint(HEAD_DIM) * 8u;
#pragma unroll
            for (uint j = 0; j < 8u; ++j) {
                qt[(c + j) * 8u + sg] = qsrc[c + j];
            }
        }",
                1,
            );
            out = sub_exact(
                &out,
                QF_LOAD,
                "\
                    simdgroup_half8x8 qf;
                    if ((ABL & 2u) != 0u) {
                        qf = make_filled_simdgroup_matrix<half, 8, 8>(half(0.5f));
                    } else {
                        simdgroup_load(qf,
                                       qstage + (b * uint(NSG)) * uint(HEAD_DIM) + d8 * 8u,
                                       8);
                    }",
                1,
            );
        }

        // ---- 5. the two controls -------------------------------------------
        match self.kind {
            Kind::Spin => {
                out = sub_exact(
                    &out,
                    BODY_ANCHOR,
                    &format!("{BODY_ANCHOR}\n    // POSITIVE CONTROL: a spin the instrument must resolve.\n    for (uint c = 0; c < 3'000'000u; ++c) {{\n        if (tid == 0u) {{ msb[0] += float(c); }}\n    }}"),
                    1,
                );
            }
            Kind::Empty => {
                out = sub_exact(
                    &out,
                    BODY_ANCHOR,
                    &format!("{BODY_ANCHOR}\n    if (tid < 256u) {{ return; }}"),
                    1,
                );
            }
            _ => {}
        }
        out
    }

    /// How this arm's output is judged.
    ///
    /// * `exact` -- must be **bit-identical** to the shipping kernel. The `msub`
    ///   and `qt` arms change which registers a value travels through and how Q
    ///   is laid out in threadgroup memory, never which arithmetic is performed.
    /// * `differs` -- must **differ**. The ablations replace real data with
    ///   constants or remove work; if one were bit-identical its edit did not
    ///   apply and its timing is a second copy of the baseline.
    /// * `void_output` -- the `empty` arm must differ, but it is *expected* to
    ///   write essentially nothing, so it is exempt from the non-zero check that
    ///   guards the other arms (there is none for it: it is arm 0's check that
    ///   matters, and `empty` is never arm 0).
    fn expectation(&self) -> Expectation {
        match self.kind {
            Kind::Baseline | Kind::Msub(_) | Kind::Qt(_) | Kind::Spin => Expectation::Exact,
            Kind::Abl(_) | Kind::Empty => Expectation::Differs,
        }
    }
}

/// The shipping `AttentionParams`, byte for byte.
#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    queries: u32,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    key_count: u32,
    scale: f32,
    window: u32,
    splits: u32,
    keys_per_split: u32,
    pass_begin: u32,
    pass_keys: u32,
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn row(device: &Device, n: usize, salt: f32) -> Buffer {
    let bits: Vec<u16> = (0..n)
        .map(|i| {
            let x = ((i as f32 + salt) * 0.0173).sin() * 0.5
                + ((i as f32 + salt) * 0.0031).cos() * 0.25;
            f16::from_f32(x).to_bits()
        })
        .collect();
    device.new_buffer_with_data(
        bits.as_ptr() as *const std::ffi::c_void,
        (bits.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn u32s(device: &Device, v: &[u32]) -> Buffer {
    device.new_buffer_with_data(
        v.as_ptr() as *const std::ffi::c_void,
        (v.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn median_f64(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

fn main() {
    let dump = std::env::var("KAPLUMBA_DUMP").ok().map(PathBuf::from);
    if let Some(dir) = &dump {
        std::fs::create_dir_all(dir).expect("dump dir");
    }

    // `KAPLUMBA_EMIT_ONLY=1` writes each arm's shader source and exits without
    // touching the GPU, so the sources can be compiled with `xcrun metal` while
    // the GPU is busy with someone else's model run. Every anchor assertion in
    // `Arm::source` runs on this path, so an arm whose edit does not apply is
    // still caught here.
    if std::env::var("KAPLUMBA_EMIT_ONLY").is_ok() {
        for arm in arms() {
            let src = arm.source();
            if matches!(arm.kind, Kind::Baseline) {
                assert_eq!(src, SHIPPING, "arm {} must be the shipping bytes", arm.name);
            } else {
                assert!(src != SHIPPING, "arm {} produced the shipping source unchanged", arm.name);
            }
            println!("  arm {:<10} {} lines", arm.name, src.lines().count());
            if let Some(dir) = &dump {
                std::fs::write(dir.join(format!("{}.metal", arm.name)), &src)
                    .expect("write arm source");
            }
        }
        println!("emitted; no GPU was touched");
        return;
    }

    let device = Device::system_default().expect("no Metal device");
    let queue = device.new_command_queue().to_owned();

    let key_count = env_usize("KAPLUMBA_KEYS", 24576);
    let queries_count = env_usize("KAPLUMBA_QUERIES", 1024);
    let bq = env_usize("KAPLUMBA_BQ", 1);
    let rounds = env_usize("KAPLUMBA_ROUNDS", 3);
    let (q_heads, kv_heads, head_dim) = (32usize, 4usize, 512usize);

    println!(
        "device: {} | keys={key_count} queries={queries_count} BQ={bq} rounds={rounds}",
        device.name()
    );

    let arms = arms();
    let mut built = Vec::new();
    for arm in &arms {
        let src = arm.source();
        let inserted = src.lines().count() as i64 - SHIPPING.lines().count() as i64;
        if matches!(arm.kind, Kind::Baseline) {
            assert_eq!(src, SHIPPING, "arm {} must be the shipping bytes", arm.name);
        } else {
            assert!(src != SHIPPING, "arm {} produced the shipping source unchanged", arm.name);
        }
        println!("  arm {:<10} lines changed vs shipping: {inserted:+}", arm.name);
        if let Some(dir) = &dump {
            std::fs::write(dir.join(format!("{}.metal", arm.name)), &src).expect("write arm source");
        }
        let lib: Library = device
            .new_library_with_source(&src, &metal::CompileOptions::new())
            .unwrap_or_else(|e| panic!("compile arm {}: {e}", arm.name));
        let get = |n: &str| -> metal::ComputePipelineState {
            let f = lib
                .get_function(n, None)
                .unwrap_or_else(|_| panic!("{n} missing in {}", arm.name));
            device
                .new_compute_pipeline_state_with_function(&f)
                .unwrap_or_else(|e| panic!("pipeline {n}: {e}"))
        };
        // The threadgroup-memory footprint the dispatch will ask for, so a
        // variant that would not launch is visible before it is timed.
        let tg = get("online_attention_flash_bq1_512_8").max_total_threads_per_threadgroup();
        println!(
            "  {:<10} max threads/threadgroup = {tg}  (256 threads dispatched)",
            arm.name
        );
        assert!(tg >= 256, "arm {} cannot run a 256-thread dispatch", arm.name);
        built.push((
            arm.name,
            get("online_attention_flash_bq1_512_8"),
            get("online_attention_flash_512_8"),
        ));
    }

    let kv = key_count * kv_heads * head_dim;
    let queries_buf = row(&device, queries_count * q_heads * head_dim, 3.0);
    let keys_buf = row(&device, kv, 11.0);
    let values_buf = row(&device, kv, 29.0);
    let query_positions: Vec<u32> =
        (0..queries_count as u32).map(|i| key_count as u32 + i).collect();
    let key_positions: Vec<u32> = (0..key_count as u32).collect();
    let query_positions_buf = u32s(&device, &query_positions);
    let key_positions_buf = u32s(&device, &key_positions);
    let params = Params {
        queries: queries_count as u32,
        q_heads: q_heads as u32,
        kv_heads: kv_heads as u32,
        head_dim: head_dim as u32,
        key_count: key_count as u32,
        scale: 1.0,
        window: 0,
        splits: 0,
        keys_per_split: 0,
        pass_begin: 0,
        pass_keys: 0,
    };
    let out_bytes = (queries_count * q_heads * head_dim * 2) as u64;
    let outs: Vec<Buffer> = built
        .iter()
        .map(|_| device.new_buffer(out_bytes, MTLResourceOptions::StorageModeShared))
        .collect();
    let groups = (queries_count.div_ceil(bq) * kv_heads) as u64;

    let dispatch = |i: usize| -> f64 {
        let (_, bq1, bq2) = &built[i];
        let pipeline = if bq == 1 { bq1 } else { bq2 };
        let cb = queue.new_command_buffer();
        let ce = cb.new_compute_command_encoder();
        ce.set_compute_pipeline_state(pipeline);
        for (index, buf) in [
            &queries_buf, &keys_buf, &values_buf, &query_positions_buf,
            &key_positions_buf, &outs[i],
        ]
        .iter()
        .enumerate()
        {
            ce.set_buffer(index as u64, Some(buf), 0);
        }
        ce.set_bytes(
            6,
            std::mem::size_of::<Params>() as u64,
            &params as *const Params as *const std::ffi::c_void,
        );
        ce.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(256, 1, 1));
        ce.end_encoding();
        cb.commit();
        let t0 = Instant::now();
        cb.wait_until_completed();
        t0.elapsed().as_secs_f64() * 1e3
    };

    // ---- correctness FIRST, before any timing is printed -------------------
    println!("\n--- correctness (bitwise vs the shipped kernel, before any timing) ---");
    for i in 0..built.len() {
        dispatch(i);
    }
    let reference = outs[0].contents() as *const u8;
    let len = out_bytes as usize;
    let nonzero = (0..len / 2)
        .filter(|i| unsafe { *((reference as *const u16).add(*i)) != 0 })
        .count();
    assert!(
        nonzero > len / 4,
        "the shipped kernel wrote only {nonzero} non-zero halves of {}: the instrument timed an \
         empty kernel",
        len / 2
    );
    println!("  {:<10} baseline | non-zero u16: {nonzero} of {}", built[0].0, len / 2);
    for i in 1..built.len() {
        let bytes = outs[i].contents() as *const u8;
        let diff = (0..len).filter(|k| unsafe { *bytes.add(*k) != *reference.add(*k) }).count();
        match arms[i].expectation() {
            Expectation::Exact => assert_eq!(
                diff, 0,
                "arm {}: {diff} bytes differ from the shipped kernel, but it must be \
                 BIT-IDENTICAL -- this arm changes register lifetime and threadgroup layout, \
                 not arithmetic",
                built[i].0
            ),
            Expectation::Differs => assert_ne!(
                diff, 0,
                "arm {}: byte-identical to the shipped kernel, so its edit did not do what its \
                 name says",
                built[i].0
            ),
        }
        println!(
            "  {:<12} bytes differing: {diff:>10}  ({})",
            built[i].0,
            match arms[i].expectation() {
                Expectation::Exact => "exact, as required",
                Expectation::Differs => "differs, as required",
            }
        );
    }

    // ---- interleaved timing ------------------------------------------------
    println!("\n--- timing, {rounds} interleaved rounds, median [min max] ms ---");
    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); built.len()];
    for i in 0..built.len() {
        dispatch(i);
    }
    for _ in 0..rounds {
        for i in 0..built.len() {
            samples[i].push(dispatch(i));
        }
    }
    let flops = (queries_count * q_heads * key_count * head_dim * 4) as f64;
    let mut medians = Vec::new();
    for (i, (name, _, _)) in built.iter().enumerate() {
        let mut xs = samples[i].clone();
        let med = median_f64(&mut xs);
        medians.push(med);
        println!(
            "  {name:<10} {med:9.2} ms  [{:8.2} {:8.2}]  {:6.2} TFLOP/s  {:5.1}% of 8.30{}",
            xs[0],
            xs[xs.len() - 1],
            flops / (med * 1e-3) / 1e12,
            100.0 * flops / (med * 1e-3) / 1e12 / 8.30,
            if i == 0 { "  baseline" } else { "" },
        );
    }
    println!("\n--- ratios vs the shipped kernel (same process, same rounds) ---");
    for (i, (name, _, _)) in built.iter().enumerate().skip(1) {
        println!("  {name:<10} x{:>6.3}   {:>9.2} ms", medians[0] / medians[i], medians[i]);
    }

    let at = |name: &str| -> Option<f64> {
        built.iter().position(|b| b.0 == name).map(|i| medians[i])
    };
    if let Some(spin) = at("spin") {
        let moved = spin.max(medians[0]) / spin.min(medians[0]);
        println!(
            "\ncontrol `spin`: {spin:9.2} ms vs {:9.2} ms = x{moved:.3} {}",
            medians[0],
            if moved > 1.5 { "(the timer resolves a real change)" } else { "(VOID: every ratio here is void)" }
        );
        assert!(moved > 1.5, "the spin control did not move the timer: x{moved:.3}");
    }
    if let Some(empty) = at("empty") {
        let share = empty / medians[0];
        println!(
            "control `empty` (§10's arm): {empty:.3} ms = {:.2}% of the baseline -- the timing IS \
             of the kernel body",
            share * 100.0
        );
        assert!(share < 0.05, "`empty` read {share:.1}% of the baseline: this instrument is not measuring the body");
    }
}
