//! Task-1 instrument for `spec/prefill-24k-90tps`: **does `#pragma unroll` on
//! the shipping flash QK loop change the shipping flash kernel's speed?**
//!
//! `adversary_report2.md` (A2/A3) reports a 2.3x swing on a *standalone
//! transcription* of `attention.metal:1632`'s `d8` loop: transcribed as-is (no
//! pragma) 45.6 ms / 4.52 TFLOP/s, with the unroll forced 105.5 ms / 1.95
//! TFLOP/s, bit-identical. That is a large enough claim to be worth checking on
//! the real kernel, and it is cheap to check there.
//!
//! **This instrument uses the shipping shader source itself.** It
//! `include_str!`s `crates/kernels/src/attention.metal` -- the same bytes
//! `kaplumba-kernels` compiles -- and for each arm applies a small, anchored
//! textual edit to it. Every anchor must match exactly once (asserted), arm 0
//! with no edit is the file **verbatim** (asserted), every arm must change the
//! line count (asserted), and `KAPLUMBA_DUMP=dir` writes each arm's source out
//! to be diffed against HEAD.
//!
//! Everything else is the real dispatch: the real `online_attention_flash_*`
//! kernels, the real grid (`ceil(queries/BQ) * kv_heads` threadgroups of 256),
//! the real `AttentionParams` layout, real K/V/Q buffers, the prefill geometry
//! (1024 queries, 8 192 / 24 576 keys).
//!
//! **Correctness is checked before any timing is printed**, and the check is
//! bidirectional: an arm that must be bit-identical to the shipped kernel and is
//! not has changed the arithmetic, and an ablation that must differ and does not
//! has not done what its name says. Either aborts.
//!
//! **Two controls, both asserted.** `spin` adds a 3 000 000-iteration
//! threadgroup-memory spin at the top of the body and must move the timer by
//! >1.5x, or the timer resolves nothing and every ratio here is void. `empty` is
//! §10's arm -- `if (tid < 256u) { return; }` in a 256-thread dispatch, i.e. a
//! completely empty kernel -- and must collapse by >100x, or this timing is not
//! of the body at all. §10 recorded 964.79 ms for that arm; in this process it
//! reads a fraction of a millisecond.
//!
//! **Also here, because the corrected tensor peak (see `mma_peak.rs`) made it the
//! question that matters:** the ablation arms (`noqk`, `nopv`, `loads-l1`,
//! `noqf`, `noaf`) and the compile-time chain arms (`chains2`, `chains4`). §22's
//! arms 11/12 tried to break the accumulator chain with a `((d8 >> 3) & 1)`
//! selector on a **rolled** loop -- a runtime branch inside the loop, which
//! cannot dual-issue, so whatever they measured it was not chain ILP. These
//! carry `#pragma unroll`, so `d8` is a compile-time constant and the selector
//! resolves to straight-line code with real independent chains.
//!
//! Interleaved A/B/A/B (`ROUNDS` rounds, arms cycled inside each round), medians
//! with min/max, ratios taken within one round-set only. Usage:
//!
//! ```text
//! cargo run --release --example flash_qk_pragma_arms
//! KAPLUMBA_ARMS=0-shipped,1-unroll,spin,empty ...
//! KAPLUMBA_KEYS=8192 KAPLUMBA_QUERIES=1024 KAPLUMBA_BQ=2 ...
//! ```
//!
//! Writes `KAPLUMBA_DUMP=dir` per-arm sources there.

use half::f16;
use metal::{Buffer, Device, Library, MTLResourceOptions, MTLSize};
use std::path::PathBuf;
use std::time::Instant;

/// The shipping shader. `include_str!` so a `.metal` edit rebuilds this example
/// (cargo tracks `include_str!` inputs -- see §22.1, corrected in A11).
const SHIPPING: &str = include_str!("../../kernels/src/attention.metal");

/// Anchor: the flash body's QK loop. Unique in the file -- the other three
/// `d8 < HEAD_DIM` loops are in `attention_gqa_mma_body`, which has no `staged`
/// / `k0` locals.
const ANCHOR: &str = "\
            if (staged || k0 + 8u <= range_end) {
                for (int d8 = 0; d8 < HEAD_DIM; d8 += 8) {";

/// Anchor for the positive control: the top of the flash body.
const BODY_ANCHOR: &str = "    constexpr float NEG = -1.701411733864e+38f;\n    constexpr uint TILE = 64u;";

/// One arm: a name, the pragma to insert before the QK `d8` loop (if any), and
/// a recipe for the rest of the edit. Every arm is the shipping source with its
/// edits applied; arm 0 with no recipe is the shipping bytes verbatim.
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    /// Shipping.
    Baseline,
    /// Insert a `#pragma unroll` line before the QK `d8` loop.
    Pragma,
    /// `spin`: the positive control. A spin the instrument must resolve.
    Spin,
    /// Return before any work. The control §10.3 never ran.
    Empty,
    /// Skip the whole QK block: the `kf`/`qf` loads and the QK MMAs.
    NoQk,
    /// `ksteps = 0`: no PV MMAs, no `af`/`bf` loads, no rescale.
    NoPv,
    /// Both of the above: everything but the softmax, barriers and epilogue.
    NoQkNoPv,
    /// Every fragment load from an L1-resident fixed row: same instruction
    /// count, same MMA count, no device path.
    AllLoadsL1,
    /// The Q fragment from a filled constant: no `qstage` read at all.
    NoQfLoad,
    /// The P fragment from a filled constant: no `wq` read at all.
    NoAfLoad,
    /// **Every** fragment load replaced by a filled constant: `kf`, `qf`, `af`
    /// and `bf`. The MMA count and the instruction mix are otherwise untouched.
    ///
    /// This is the arm that identifies the limiter, because it is the only one
    /// comparable to `mma_peak.rs`: that instrument runs MMAs whose operands are
    /// already in registers and reaches 7.6-8.3 TFLOPS. This arm runs the same
    /// MMAs with every operand fetched from memory. The ratio between them is the
    /// cost of feeding the tensor pipe, which is what a "% of peak" number has to
    /// be read against.
    NoLoads,
    /// §22's ablation scaffolding with **every arm inactive** -- i.e. what
    /// `1a30427`'s arm 0 actually compiled.
    ///
    /// That harness added, to *every* arm including arm 0:
    ///
    /// * `simdgroup_float8x8 dots2[BQ], dots3[BQ], dots4[BQ]`, each
    ///   initialised on every tile (3 x 8 registers/lane at BQ=1), and
    /// * `simdgroup_half8x8 qreg[BQ][NTILES]` -- **64 registers/lane** at BQ=1,
    ///   128 at BQ=2 -- declared unconditionally, plus
    /// * a runtime `ABL` load and branch at each of nine sites.
    ///
    /// So §22's arm-0 baseline is not the shipping kernel; it is the shipping
    /// kernel with 70-190 extra registers/lane of pressure. This arm measures
    /// what that costs, against the shipping bytes.
    AblationScaffold,
    /// `N` independent QK accumulator chains, selected at COMPILE time.
    ///
    /// This is the mechanism §22's arms 11/12 tried and could not: there, the
    /// chain index was `if ((d8 >> 3) & 1)` on a **rolled** loop, i.e. a runtime
    /// branch inside the loop, which cannot dual-issue. Here the loop carries
    /// `#pragma unroll`, so `d8` is a compile-time constant and the selector is
    /// resolved at compile time into straight-line code.
    Chains(u32),
}

#[derive(Clone, Copy)]
struct Arm {
    name: &'static str,
    pragma: Option<&'static str>,
    kind: Kind,
}

fn arms() -> Vec<Arm> {
    let all = vec![
        Arm { name: "0-shipped", pragma: None, kind: Kind::Baseline },
        Arm { name: "1-unroll", pragma: Some("#pragma unroll\n"), kind: Kind::Pragma },
        Arm { name: "2-unroll2", pragma: Some("#pragma unroll(2)\n"), kind: Kind::Pragma },
        Arm { name: "3-unroll4", pragma: Some("#pragma unroll(4)\n"), kind: Kind::Pragma },
        Arm { name: "4-unroll8", pragma: Some("#pragma unroll(8)\n"), kind: Kind::Pragma },
        Arm { name: "5-unroll16", pragma: Some("#pragma unroll(16)\n"), kind: Kind::Pragma },
        Arm { name: "6-unroll32", pragma: Some("#pragma unroll(32)\n"), kind: Kind::Pragma },
        Arm { name: "spin", pragma: None, kind: Kind::Spin },
        Arm { name: "empty", pragma: None, kind: Kind::Empty },
        Arm { name: "noqk", pragma: None, kind: Kind::NoQk },
        Arm { name: "nopv", pragma: None, kind: Kind::NoPv },
        Arm { name: "noqk-nopv", pragma: None, kind: Kind::NoQkNoPv },
        Arm { name: "loads-l1", pragma: None, kind: Kind::AllLoadsL1 },
        Arm { name: "noqf", pragma: None, kind: Kind::NoQfLoad },
        Arm { name: "noaf", pragma: None, kind: Kind::NoAfLoad },
        Arm { name: "noloads", pragma: None, kind: Kind::NoLoads },
        Arm { name: "scaffold", pragma: None, kind: Kind::AblationScaffold },
        Arm { name: "chains2", pragma: Some("#pragma unroll\n"), kind: Kind::Chains(2) },
        Arm { name: "chains4", pragma: Some("#pragma unroll\n"), kind: Kind::Chains(4) },
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

/// The QK loop's `kf` device load, as the shipping source writes it.
const QK_KF_DEVICE: &str = "\
                    } else {
                        simdgroup_load(kf, keys + k0 * kv_stride + head_base + d8,
                                       kv_stride);
                    }";
/// The QK loop's `qf` load.
const QK_QF_LOAD: &str = "\
                        simdgroup_load(qf,
                                       qstage + (b * uint(NSG)) * uint(HEAD_DIM) + d8,
                                       HEAD_DIM, ulong2(0, 0), true);
                        simdgroup_multiply_accumulate(dots[b], kf, qf, dots[b]);";
/// The QK accumulator declaration block.
const QK_DOTS_DECL: &str = "\
            simdgroup_float8x8 dots[BQ];
#pragma unroll
            for (uint b = 0; b < uint(BQ); ++b) {
                dots[b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            }";
/// The QK MMA line, alone.
const QK_MMA: &str = "simdgroup_multiply_accumulate(dots[b], kf, qf, dots[b]);";
/// The publish block that folds the chains together.
const QK_PUBLISH: &str = "\
            // Publish [query][key in tile][head]. Keys past `range_end` are
            // forced to NEG by the `live` test below, so a stale or zero slot is
            // harmless.
#pragma unroll";
/// The PV `ksteps` line.
const PV_KSTEPS: &str = "const uint ksteps = (live_in_tile + 7u) / 8u;";
/// The PV `bf` device load.
const PV_BF_DEVICE: &str = "\
                } else {
                    simdgroup_load(bf, values + (base + kt * 8u) * kv_stride
                                       + head_base + nt * 8u, kv_stride);
                }";
/// The PV `af` load.
const PV_AF_LOAD: &str = "\
                simdgroup_load(af[b], wq + (b * uint(NSG)) * TILE + kt * 8u, TILE);";

/// How an arm's output is expected to differ, if at all. See
/// [`Arm::expectation`].
#[derive(Clone, Copy, PartialEq)]
enum Expectation {
    /// Bit-identical to the shipped kernel.
    Exact,
    /// Different: the arm replaced real data with a constant, or removed work.
    Differs,
    /// Different but bounded: a reassociation of the same sum.
    Reassociates,
}

/// Largest relative difference allowed for a `Reassociates` arm, measured as
/// `max|arm - shipped| / max|shipped|` over the output. Chained dot products are
/// summed in a different order, so the two differ by fp32 rounding and nothing
/// else; 1e-3 is ~4 orders of magnitude above the rounding of a 512-term fp32 sum
/// and still three orders below anything a real bug would produce.
const REASSOC_BOUND: f32 = 1e-3;

/// Replace `find` with `replacement`, asserting it occurs exactly once. A
/// silently unmatched anchor would make the arm a copy of arm 0 and produce a
/// beautiful, meaningless table, which is the failure mode this instrument
/// exists to make impossible.
fn sub_once(src: &str, find: &str, replacement: &str) -> String {
    let hits = src.matches(find).count();
    assert_eq!(hits, 1, "anchor occurs {hits} times, expected 1: {find:?}");
    src.replacen(find, replacement, 1)
}

impl Arm {
    /// The arm's shader source: the shipping bytes plus this arm's edits.
    ///
    /// Arm 0 with no pragma returns `SHIPPING` **verbatim**, so the baseline is
    /// the shipping bytes and not a whitespace-different copy of them.
    fn source(&self) -> String {
        let hits = SHIPPING.matches(ANCHOR).count();
        assert_eq!(hits, 1, "anchor found {hits} times, expected 1");
        if self.pragma.is_none() && self.kind == Kind::Baseline {
            return SHIPPING.to_string();
        }
        let mut out = SHIPPING.to_string();
        // The pragma edit, if any, comes first so later anchors still match the
        // shipping text (they do not overlap the QK loop header).
        if let Some(pragma) = self.pragma {
            let replacement = format!(
                "            if (staged || k0 + 8u <= range_end) {{\n{pragma}\
                 \x20               for (int d8 = 0; d8 < HEAD_DIM; d8 += 8) {{"
            );
            out = sub_once(&out, ANCHOR, &replacement);
        }
        match self.kind {
            Kind::Baseline | Kind::Pragma => {}
            Kind::Spin => {
                out = sub_once(
                    &out,
                    BODY_ANCHOR,
                    &format!(
                        "{BODY_ANCHOR}\n    // POSITIVE CONTROL: a spin the instrument must \
                         resolve.\n    for (uint c = 0; c < 3'000'000u; ++c) {{\n        \
                         if (tid == 0u) {{ msb[0] += float(c); }}\n    }}"
                    ),
                );
            }
            Kind::Empty => {
                // `tid` is `thread_position_in_threadgroup` and the prefill flash
                // path dispatches 256 threads per threadgroup, so `tid < 256u` is
                // true for EVERY thread: this is an empty kernel, not a partial
                // one. That is §10's arm, reproduced as a control here.
                out = sub_once(&out, BODY_ANCHOR, &format!("{BODY_ANCHOR}\n    if (tid < 256u) {{ return; }}"));
            }
            // Matched before the single-edit arms so `noqk-nopv` gets both.
            Kind::NoQkNoPv => {
                out = sub_once(
                    &out,
                    "            if (staged || k0 + 8u <= range_end) {",
                    "            if (false) {",
                );
                out = sub_once(&out, PV_KSTEPS, "const uint ksteps = 0u;");
            }
            Kind::NoQk => {
                out = sub_once(
                    &out,
                    "            if (staged || k0 + 8u <= range_end) {",
                    "            if (false) {",
                );
            }
            Kind::NoPv => {
                out = sub_once(&out, PV_KSTEPS, "const uint ksteps = 0u;");
            }
            Kind::AllLoadsL1 => {
                out = sub_once(
                    &out,
                    QK_KF_DEVICE,
                    "                    } else {\n                        simdgroup_load(kf, keys + head_base + d8, kv_stride);\n                    }",
                );
                out = sub_once(
                    &out,
                    PV_BF_DEVICE,
                    "                } else {\n                    simdgroup_load(bf, values + head_base + nt * 8u, kv_stride);\n                }",
                );
            }
            Kind::NoQfLoad => {
                out = sub_once(
                    &out,
                    QK_QF_LOAD,
                    "                        qf = make_filled_simdgroup_matrix<half, 8, 8>(half(0.5f));\n                        simdgroup_multiply_accumulate(dots[b], kf, qf, dots[b]);",
                );
            }
            Kind::NoAfLoad => {
                out = sub_once(
                    &out,
                    PV_AF_LOAD,
                    "                af[b] = make_filled_simdgroup_matrix<half, 8, 8>(half(0.5f));",
                );
            }
            Kind::NoLoads => {
                // `kf`: both sources (the staged tail and the device).
                out = sub_once(
                    &out,
                    "                    if (staged) {\n                        simdgroup_load(kf, tail + d8, HEAD_DIM);\n                    } else {",
                    "                    if (staged) {\n                        kf = make_filled_simdgroup_matrix<half, 8, 8>(half(0.5f));\n                    } else {",
                );
                out = sub_once(
                    &out,
                    QK_KF_DEVICE,
                    "                    } else {\n                        kf = make_filled_simdgroup_matrix<half, 8, 8>(half(0.5f));\n                    }",
                );
                // `qf`.
                out = sub_once(
                    &out,
                    QK_QF_LOAD,
                    "                        qf = make_filled_simdgroup_matrix<half, 8, 8>(half(0.5f));\n                        simdgroup_multiply_accumulate(dots[b], kf, qf, dots[b]);",
                );
                // `af` and `bf`, staged and device.
                out = sub_once(
                    &out,
                    PV_AF_LOAD,
                    "                af[b] = make_filled_simdgroup_matrix<half, 8, 8>(half(0.5f));",
                );
                out = sub_once(
                    &out,
                    "                    simdgroup_load(bf, tail + nt * 8u, HEAD_DIM);",
                    "                    bf = make_filled_simdgroup_matrix<half, 8, 8>(half(0.5f));",
                );
                out = sub_once(
                    &out,
                    PV_BF_DEVICE,
                    "                } else {\n                    bf = make_filled_simdgroup_matrix<half, 8, 8>(half(0.5f));\n                }",
                );
            }
            // §22's arm 0: the shipping body plus the whole ablation harness,
            // with `ABL` fixed at 0 so no arm fires. Reproduces
            // `1a30427`'s declaration and initialisation additions verbatim.
            Kind::AblationScaffold => {
                out = sub_once(
                    &out,
                    "    threadgroup half *qstage,\n    threadgroup float *sth,",
                    "    threadgroup half *qstage,\n    threadgroup float *sth,",
                );
                // (a) the four-dot accumulator set, initialised on every tile.
                out = sub_once(
                    &out,
                    QK_DOTS_DECL,
                    "            simdgroup_float8x8 dots[BQ];\n\
                     \x20           simdgroup_float8x8 dots2[BQ];\n\
                     \x20           simdgroup_float8x8 dots3[BQ];\n\
                     \x20           simdgroup_float8x8 dots4[BQ];\n\
                     #pragma unroll\n            for (uint b = 0; b < uint(BQ); ++b) {\n\
                     \x20               dots[b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n\
                     \x20               dots2[b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n\
                     \x20               dots3[b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n\
                     \x20               dots4[b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n\
                     \x20           }",
                );
                // (b) the qreg array, declared unconditionally and left untouched
                // because `ABL != 7`.
                out = sub_once(
                    &out,
                    "    for (uint base = range_begin; base < range_end; base += TILE) {",
                    "    threadgroup half *qreg_scratch = qstage;\n\
                     \x20   simdgroup_half8x8 qreg[BQ][NTILES];\n\
                     \x20   if (params.pass_keys == 0xFFFFFFFFu) {  // never taken; ABL == 0\n\
                     #pragma unroll\n        for (uint b = 0; b < uint(BQ); ++b) {\n\
                     #pragma unroll\n            for (uint t = 0; t < NTILES; ++t) {\n\
                     \x20               simdgroup_load(qreg[b][t], qstage + (b * uint(NSG)) * uint(HEAD_DIM)\n\
                     \x20                                   + t * 8u, HEAD_DIM, ulong2(0, 0), true);\n\
                     \x20           }\n        }\n\
                     \x20       (void)qreg_scratch;\n    }\n\
                     \x20   for (uint base = range_begin; base < range_end; base += TILE) {",
                );
                // (c) the chain-select branch, compiled in with `ABL == 0` so it
                // takes the `else` (shipping) arm at runtime.
                out = sub_once(
                    &out,
                    QK_MMA,
                    "const uint ABL = 0u;\n                        \
                     if (ABL == 11u) {\n                            \
                     simdgroup_multiply_accumulate(dots2[b], kf, qf, dots2[b]);\n                        \
                     } else if (ABL == 12u) {\n                            \
                     simdgroup_multiply_accumulate(dots3[b], kf, qf, dots3[b]);\n                        \
                     } else {\n                            \
                     simdgroup_multiply_accumulate(dots[b], kf, qf, dots[b]);\n                        }",
                );
            }
            Kind::Chains(n) => {
                // Compile-time chain selection: requires the pragma arm, which is
                // why every `Chains` arm carries one.
                assert!(
                    self.pragma.is_some(),
                    "arm {} selects chains at compile time and needs the unroll",
                    self.name
                );
                out = sub_once(
                    &out,
                    QK_DOTS_DECL,
                    &format!(
                        "            simdgroup_float8x8 dots[BQ];\n\
                         threadgroup float *chains_scratch = sth;\n\
                         \x20           simdgroup_float8x8 dots2[BQ];\n\
                         #pragma unroll\n            for (uint b = 0; b < uint(BQ); ++b) {{\n        \
                         dots[b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n        \
                         dots2[b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n    \
                         \x20           }}\n            (void)chains_scratch;"
                    ),
                );
                // With the loop unrolled, `(d8 >> 3) % N` is a compile-time
                // constant, so this resolves to straight-line code.
                out = sub_once(
                    &out,
                    QK_MMA,
                    &format!(
                        "if (((d8 >> 3) % {n}u) == 0u) {{ \
                         simdgroup_multiply_accumulate(dots[b], kf, qf, dots[b]); }} else {{ \
                         simdgroup_multiply_accumulate(dots2[b], kf, qf, dots2[b]); }}"
                    ),
                );
                out = sub_once(
                    &out,
                    QK_PUBLISH,
                    "            // Publish [query][key in tile][head]. Chains folded back\n\
                     \x20           // into `dots` first; the fold order is fixed so the sum\n\
                     \x20           // is the same every dispatch.\n\
                     #pragma unroll\n            for (uint b = 0; b < uint(BQ); ++b) {\n\
                     \x20               dots[b].thread_elements()[0] += dots2[b].thread_elements()[0];\n\
                     \x20               dots[b].thread_elements()[1] += dots2[b].thread_elements()[1];\n\
                     \x20           }\n#pragma unroll",
                );
            }
        }
        out
    }

    /// How many lines this arm's source differs from the shipping source by.
    /// A tripwire, not a limit: it catches an edit that silently did nothing and
    /// it is printed per arm.
    fn inserted_lines(&self, src: &str) -> i64 {
        src.lines().count() as i64 - SHIPPING.lines().count() as i64
    }

    /// What the arm is expected to do to the output, and therefore how its
    /// correctness is judged. Every arm falls into exactly one of these, and an
    /// arm that lands in the wrong one aborts the run.
    fn expectation(&self) -> Expectation {
        match self.kind {
            // Same arithmetic, different scheduling hint: must be BIT-IDENTICAL.
            Kind::Baseline | Kind::Pragma => Expectation::Exact,
            // Real data replaced by constants or by nothing: must DIFFER.
            Kind::NoQk
            | Kind::NoPv
            | Kind::NoQkNoPv
            | Kind::NoQfLoad
            | Kind::NoAfLoad
            | Kind::AllLoadsL1
            | Kind::NoLoads
            | Kind::Empty => Expectation::Differs,
            // The scaffold adds declarations and branches but changes no
            // arithmetic: it must be BIT-IDENTICAL. If it is not, the harness
            // itself altered the kernel and every §22 ratio needs re-deriving.
            Kind::AblationScaffold => Expectation::Exact,
            // The spin writes `msb[0]`, which the rescale reads only for
            // `(b = 0, sg = 0)`'s own lane mapping and which is overwritten before
            // any consumer sees it -- so it is a pure timing control and the
            // output must be bit-identical. That is the useful property: a
            // control that also changed the answer would not isolate the timer.
            Kind::Spin => Expectation::Exact,
            // Folding two chains together reassociates the dot product, so the
            // result differs in the last bits and must be CLOSE. This is the one
            // category that is not bit-identical and not arbitrary: it is
            // bounded-ulp, and the bound is asserted here.
            Kind::Chains(_) => Expectation::Reassociates,
        }
    }
}

/// The shipping `AttentionParams`, byte-for-byte. Must stay in lockstep with
/// `attention.rs`'s struct and `attention.metal`'s.
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

struct ArmKernels {
    name: String,
    bq1: metal::ComputePipelineState,
    bq2: metal::ComputePipelineState,
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
    let device = Device::system_default().expect("no Metal device");
    let queue = device.new_command_queue().to_owned();
    let dump = std::env::var("KAPLUMBA_DUMP").ok().map(PathBuf::from);
    if let Some(dir) = &dump {
        std::fs::create_dir_all(dir).expect("dump dir");
    }

    let key_count = env_usize("KAPLUMBA_KEYS", 24576);
    let queries_count = env_usize("KAPLUMBA_QUERIES", 1024);
    let bq = env_usize("KAPLUMBA_BQ", 1);
    let rounds = env_usize("KAPLUMBA_ROUNDS", 3);
    // The engine's full-attention shape: 32 q heads / 4 kv heads / head_dim 512.
    let (q_heads, kv_heads, head_dim) = (32usize, 4usize, 512usize);

    println!(
        "device: {} | keys={key_count} queries={queries_count} BQ={bq} rounds={rounds}",
        device.name()
    );

    // ---- compile every arm's library up front (compile time is not measured) --
    let arms = arms();
    let mut built: Vec<ArmKernels> = Vec::new();
    for arm in &arms {
        let src = arm.source();
        let inserted = arm.inserted_lines(&src);
        if arm.kind == Kind::Baseline && arm.pragma.is_none() {
            assert_eq!(
                src, SHIPPING,
                "arm {} is supposed to be the shipping source verbatim",
                arm.name
            );
        }
        let is_baseline = arm.kind == Kind::Baseline && arm.pragma.is_none();
        // An arm that produced the shipping source unchanged would be a silent
        // no-op: its timing would be a second copy of the baseline and its
        // "ratio" would be 1.000 for the wrong reason.
        assert!(
            is_baseline || src != SHIPPING,
            "arm {} produced the shipping source unchanged: its edit did not apply",
            arm.name
        );
        println!(
            "  arm {:<14} lines changed vs shipping: {inserted:+}",
            arm.name
        );
        if let Some(dir) = &dump {
            let path = dir.join(format!("{}.metal", arm.name.replace(' ', "_")));
            std::fs::write(&path, &src).expect("write arm source");
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
        built.push(ArmKernels {
            name: arm.name.to_string(),
            bq1: get("online_attention_flash_bq1_512_8"),
            bq2: get("online_attention_flash_512_8"),
        });
    }

    // ---- buffers, exactly as `attention_qk_mma_bench` builds them -------------
    let kv = key_count * kv_heads * head_dim;
    let queries = queries_count * q_heads * head_dim;
    let queries_buf = row(&device, queries, 3.0);
    let keys_buf = row(&device, kv, 11.0);
    let values_buf = row(&device, kv, 29.0);
    // Every query sits at or past `key_count`, so nothing is causally masked and
    // the FLOP count is the kernel's actual work.
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
    // The params blob is set inline per dispatch (`set_bytes`), not from a
    // buffer: it is a `constant` binding and is read once per threadgroup.
    let out_bytes = (queries_count * q_heads * head_dim * 2) as u64;
    let outs: Vec<Buffer> = built
        .iter()
        .map(|_| device.new_buffer(out_bytes, MTLResourceOptions::StorageModeShared))
        .collect();

    let groups = (queries_count.div_ceil(bq) * kv_heads) as u64;

    let dispatch = |arm: &ArmKernels, out: &Buffer| -> f64 {
        let pipeline = if bq == 1 { &arm.bq1 } else { &arm.bq2 };
        let cb = queue.new_command_buffer();
        let ce = cb.new_compute_command_encoder();
        ce.set_compute_pipeline_state(pipeline);
        for (index, buf) in [
            &queries_buf, &keys_buf, &values_buf, &query_positions_buf,
            &key_positions_buf, out,
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

    // ---- correctness FIRST, before any timing is printed ----------------------
    // Each arm's output is compared byte-for-byte with arm 0's. An arm that is
    // *expected* to be exact and is not has changed the kernel's arithmetic and
    // its timing is meaningless; an arm that is *expected* to differ (the
    // ablations, which replace real data with constants) and does not has not
    // done what its name says. Both are failures.
    println!("\n--- correctness (bitwise vs arm 0, before any timing) ---");
    for (index, arm) in built.iter().enumerate() {
        dispatch(arm, &outs[index]);
    }
    let reference = outs[0].contents() as *const u8;
    let reference_len = out_bytes as usize;
    let arm0_nonzero = (0..reference_len / 2)
        .filter(|i| unsafe { *(reference as *const u16).add(*i) != 0 })
        .count();
    assert!(
        arm0_nonzero > reference_len / 4,
        "arm 0 wrote only {arm0_nonzero} non-zero halves of {}: the instrument timed an \
         empty kernel",
        reference_len / 2
    );
    println!(
        "  {:<24} baseline | arm-0 non-zero u16: {arm0_nonzero} of {}",
        built[0].name,
        reference_len / 2
    );
    let reference_half = reference as *const u16;
    let scale = (0..reference_len / 2)
        .map(|i| {
            let bits = unsafe { *reference_half.add(i) };
            f32::from(half::f16::from_bits(bits)).abs()
        })
        .fold(0.0f32, f32::max);
    for index in 1..built.len() {
        let bytes = outs[index].contents() as *const u8;
        let diff = (0..reference_len)
            .filter(|i| unsafe { *bytes.add(*i) != *reference.add(*i) })
            .count();
        let halfs = outs[index].contents() as *const u16;
        let worst = if scale > 0.0 {
            (0..reference_len / 2)
                .map(|i| {
                    let a = f32::from(half::f16::from_bits(unsafe { *halfs.add(i) }));
                    let b = f32::from(half::f16::from_bits(unsafe {
                        *reference_half.add(i)
                    }));
                    (a - b).abs() / scale
                })
                .fold(0.0f32, f32::max)
        } else {
            f32::INFINITY
        };
        match arms[index].expectation() {
            Expectation::Exact => assert_eq!(
                diff, 0,
                "arm {}: {} bytes differ from the shipped kernel but it must be \
                 bit-identical",
                built[index].name,
                diff
            ),
            Expectation::Differs => assert_ne!(
                diff, 0,
                "arm {}: byte-identical to the shipped kernel, so it did not do what \
                 its name says",
                built[index].name
            ),
            Expectation::Reassociates => assert!(
                worst <= REASSOC_BOUND,
                "arm {}: worst normalised difference {worst:e} exceeds the \
                 reassociation bound {REASSOC_BOUND:e}",
                built[index].name
            ),
        }
        println!(
            "  {:<24} bytes differing: {diff:>10}  worst norm diff: {worst:.3e}  ({})",
            built[index].name,
            match arms[index].expectation() {
                Expectation::Exact => "exact, as required",
                Expectation::Differs => "differs, as required",
                Expectation::Reassociates => "bounded-ulp, as required",
            }
        );
    }

    // ---- interleaved timing --------------------------------------------------
    println!("\n--- timing, {rounds} interleaved rounds, median [min max] ms ---");
    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); built.len()];
    // One untimed warm pass per arm, so no arm pays first-touch.
    for (index, arm) in built.iter().enumerate() {
        dispatch(arm, &outs[index]);
    }
    for _ in 0..rounds {
        for (index, arm) in built.iter().enumerate() {
            samples[index].push(dispatch(arm, &outs[index]));
        }
    }
    let flops = (queries_count * q_heads * key_count * head_dim * 4) as f64;
    let mut medians = Vec::new();
    for (index, arm) in built.iter().enumerate() {
        let mut xs = samples[index].clone();
        let med = median_f64(&mut xs);
        medians.push(med);
        println!(
            "  {:<24} {med:9.2} ms  [{:8.2} {:8.2}]  {:6.2} TFLOP/s  {}",
            arm.name,
            xs[0],
            xs[xs.len() - 1],
            flops / (med * 1e-3) / 1e12,
            if index == 0 { "baseline" } else { "" },
        );
    }
    println!("\n--- ratios vs arm 0 (same process, same rounds) ---");
    for (index, arm) in built.iter().enumerate().skip(1) {
        println!(
            "  {:<24} x{:>6.3}   {:>7.2} ms",
            arm.name,
            medians[0] / medians[index],
            medians[index]
        );
    }
    // The controls, checked here rather than trusted.
    //
    // `spin` moves the instrument by construction (an added spin), which shows
    // the timer resolves a real change. `empty` is §10's arm and the real test
    // of "is this timing of the kernel": it must collapse to the dispatch cost.
    // Without `empty`, an arm that accidentally emptied the whole body would read
    // as "the change did nothing" -- which is exactly the failure this spec has
    // already made twice.
    let at = |name: &str| -> Option<f64> {
        built.iter().position(|a| a.name == name).map(|i| medians[i])
    };
    if let Some(spin) = at("spin") {
        let moved = spin.max(medians[0]) / spin.min(medians[0]);
        println!(
            "\ncontrol `spin`: {:9.2} ms vs {:9.2} ms = x{moved:.3} {}",
            medians[0],
            spin,
            if moved > 1.5 {
                "(the timer resolves a real change)"
            } else {
                "(THE TIMER DID NOT MOVE -- every ratio here is void)"
            }
        );
        assert!(moved > 1.5, "the spin control did not move the timer: x{moved:.3}");
    }
    if let Some(empty) = at("empty") {
        let collapsed = medians[0] / empty;
        let share = empty / medians[0];
        println!(
            "control `empty` (§10's arm): {empty:.3} ms = {:.2}% of the baseline (x{collapsed:.0}) \
             -- the timing IS of the kernel body",
            share * 100.0
        );
        // The right form of this assertion is a *share*, not a ratio: the collapse
        // factor scales with how much work the geometry has, so a fixed ratio
        // would fail on a small shape and pass on a large one for the wrong
        // reason. An emptied kernel must cost the dispatch and nothing else.
        assert!(
            share < 0.05,
            "`empty` read {empty:.2} ms = {:.1}% of a {:.2} ms baseline: if an emptied kernel does \
             not collapse to dispatch cost, this instrument is not measuring the kernel it \
             claims to",
            share * 100.0,
            medians[0]
        );
    }
}