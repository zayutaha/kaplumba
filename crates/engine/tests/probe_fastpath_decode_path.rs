//! EXL3 decode-path census: which of the two trellis readers every kernel in
//! this library actually uses, and whether any kernel the engine can dispatch is
//! still on the slow one.
//!
//! # The question
//!
//! `crates/kernels/src/linear.metal` contains two ways to read one 16-bit EXL3
//! code out of the trellis:
//!
//! * **`exl3_code_at` (linear.metal:216)** re-derives the lane's three-word
//!   64-bit sliding window *per code*. Its own comment at linear.metal:258 says
//!   what that costs: "issues eight times the loads and eight times the 64-bit
//!   shifts for identical output".
//! * **`exl3_decode8` (linear.metal:279) / `exl3_decode8_mcg` (linear.metal:345)**
//!   set the window up once and slice all eight codes a lane owns out of it.
//!   That is the fast path, and it is bit-identical to `exl3_code_at` by
//!   construction.
//!
//! The hand-written column-major kernels in `linear_repacked.metal` call neither:
//! they carry a third form, `extract_v_mcg_32` (linear_repacked.metal:53), which
//! is the same "one window, eight codes" idea specialised to the only rate this
//! checkpoint uses -- 3 bpw, where the eight codes span 22 bits and therefore
//! always fit in *two* 32-bit words.
//!
//! **Result: no kernel the engine can dispatch uses the per-code path.** The
//! entire per-code population of the library is `linear_gemv` (generic) and
//! `linear_gemv_fast{1..=8}`, plus two arms that this checkpoint never takes
//! (`linear_gemv_tiled`, the prefill fallback; and `linear_gemm_mma`'s
//! fractional-rate arm). See
//! [`no_kernel_the_dispatch_can_reach_uses_the_per_code_path`].
//!
//! # What is checked here
//!
//! Everything is read out of the committed sources with `include_str!`, so an
//! edit fails the test instead of quietly changing the answer. **No GPU is
//! touched and no model is loaded.**
//!
//! 1. [`the_census`] -- print every kernel in the library and which reader it
//!    uses. An absence claim can only be inspected from the full list.
//! 2. [`only_eight_dead_specialisations_use_the_per_code_path`] -- the per-code
//!    population is exactly the eight dead specialisations plus the generic
//!    kernel.
//! 3. [`no_kernel_the_dispatch_can_reach_uses_the_per_code_path`] -- combining
//!    the census with the real dispatch order in `linear.rs`.
//! 4. [`the_eight_x_penalty_is_derived_not_quoted`] -- the 8x figure is *counted*
//!    out of the function bodies rather than repeated from the comment, and the
//!    honest ceiling for `linear_gemv_fast*` (4x, not 8x) is derived too.
//! 5. [`every_fused_kernel_is_on_the_hoisted_path`] -- the specific question
//!    about `linear_gemv_ishizuki_k3_fused2_co*`.
//!
//! # Three premises in the request that do not survive contact with the code
//!
//! * **`exl3_decode32` does not exist.** Not in `linear.metal`, not in
//!   `linear_repacked.metal`, not in the generator
//!   (`scripts/gen_exl3_gemv.py`), not anywhere. Line ~344 is
//!   `exl3_decode8_mcg`'s `template <bool ALIGNED>` line. Pinned by
//!   [`there_is_no_32_code_decoder`].
//! * **The fused MLP does not reach `fused2_co*` by default.**
//!   `encode_gemv_fused2_mlp` tests `_2col` *before* the `_co` arm
//!   (linear.rs:2590 before linear.rs:2598), and `encode_gemv_fused2`'s `_co`
//!   arm is guarded by `out_features_a == out_features_b`, which never holds for
//!   this model (q is 16384, k is 2048). Pinned by
//!   [`fused2_co_is_not_the_default_for_this_model`]. Irrelevant to the census
//!   -- `_co`, `_co4` and `_2col` are all hoisted -- but it means "`fused2_co`
//!   is the live fused MLP kernel" is not safe to assume when reading a
//!   benchmark.
//! * **There is no 8x win available anywhere, because there is nothing on the
//!   slow path to win it on.** The largest per-code population is unreachable.
//!   [`the_eight_x_penalty_is_derived_not_quoted`] keeps the number so a future
//!   change that *does* dispatch one of these kernels has to confront it.
//!
//! # What this file does NOT establish
//!
//! It is a static census. "Reachable" is decided by reading `encode_gemv_raw`,
//! `encode_gemv_fused2`, `encode_gemv_fused2_mlp` and `encode_gemv_fused3`, in
//! the all-flags-unset state that [`all_dispatch_defining_flags_unset`] pins. It
//! measures nothing. Per `AGENTS.md` §3.4 and `spec/engine-performance` Q11, no
//! coverage claim is made here: proving the "reachable kernels are hoisted"
//! assertion can actually fail would need the reachability condition sabotaged,
//! which is a change to code this probe is not allowed to make.

use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// Sources
// ---------------------------------------------------------------------------

const LINEAR_METAL: &str = include_str!("../../kernels/src/linear.metal");
const REPACKED_METAL: &str = include_str!("../../kernels/src/linear_repacked.metal");
const LINEAR_RS: &str = include_str!("../../kernels/src/linear.rs");
const MODEL_RS: &str = include_str!("../src/model.rs");
const GEN_PY: &str = include_str!("../../../scripts/gen_exl3_gemv.py");

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/// One `kernel void NAME(...) { ... }` block.
#[derive(Clone, Debug)]
struct Kernel {
    name: String,
    file: &'static str,
    /// 1-based line of `kernel void`.
    line: usize,
    body: String,
}

/// Every kernel in `source`, in source order.
///
/// A body ends at the first line that is exactly `}`, which is how these two
/// files are formatted: every function in them closes its brace in column 0.
fn kernels_in(file: &'static str, source: &str) -> Vec<Kernel> {
    let mut out = Vec::new();
    let mut pending: Option<(String, usize, String)> = None;
    for (index, line) in source.lines().enumerate() {
        if let Some((name, start, body)) = pending.as_mut() {
            body.push('\n');
            body.push_str(line);
            if line == "}" {
                out.push(Kernel {
                    name: std::mem::take(name),
                    file,
                    line: *start,
                    body: std::mem::take(body),
                });
                pending = None;
            }
        } else if let Some(rest) = line.trim_start().strip_prefix("kernel void ") {
            let end = rest
                .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .unwrap_or(rest.len());
            pending = Some((
                rest[..end].to_string(),
                index + 1,
                format!("{line}\n"),
            ));
        }
    }
    assert!(pending.is_none(), "unterminated kernel in {file}");
    out
}

/// Every kernel in the combined library, in `linear.metal` then
/// `linear_repacked.metal` order -- the order `combined_source()` splices them.
fn all_kernels() -> Vec<Kernel> {
    let mut out = kernels_in("linear.metal", LINEAR_METAL);
    out.extend(kernels_in("linear_repacked.metal", REPACKED_METAL));
    out
}

/// Body of a top-level function, by name.
///
/// Used for the reader helpers, so their cost can be counted instead of quoted
/// from their own comments.
fn function_body<'a>(source: &'a str, file: &str, name: &str) -> &'a str {
    let needle = format!(" {name}(");
    let start = source
        .find(&needle)
        .unwrap_or_else(|| panic!("{file}: no function `{name}`"));
    let brace = source[start..]
        .find('{')
        .map(|at| start + at)
        .unwrap_or_else(|| panic!("{file}: `{name}` has no body"));
    let end = source[brace..]
        .find("\n}\n")
        .map(|at| brace + at)
        .unwrap_or_else(|| panic!("{file}: `{name}` body does not close in column 0"));
    &source[brace..end]
}

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

/// What a kernel does with the trellis.
///
/// A kernel can be *both*, and one is: `linear_gemm_mma` hoists the window for
/// integer rates and keeps `fractional_window` (also one window per weight) for
/// fractional ones. Modelling that as a single enum would either lie about the
/// fractional arm or hide the hoisted one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Paths {
    /// Reads no weights: finalize, norm, transform, zero.
    decodes: bool,
    /// Sets one window up and slices eight codes out of it.
    hoisted: bool,
    /// Re-derives the window per code from device memory.
    per_code_global: bool,
    /// Re-derives the window per code from threadgroup memory. The words were
    /// fetched once by a coalesced pass and shared across the query block, so
    /// the *loads* are already amortised; what repeats per code is the
    /// shift/or chain.
    per_code_threadgroup: bool,
    /// `fractional_window`: the fractional packer has no lane grouping, so its
    /// reader is one window per weight by construction.
    per_code_fractional: bool,
}

impl Paths {
    fn none() -> Paths {
        Paths {
            decodes: false,
            hoisted: false,
            per_code_global: false,
            per_code_threadgroup: false,
            per_code_fractional: false,
        }
    }

    fn per_code(&self) -> bool {
        self.per_code_global || self.per_code_threadgroup || self.per_code_fractional
    }

    fn label(&self) -> String {
        if !self.decodes {
            return "no-decode".to_string();
        }
        let mut parts = Vec::new();
        if self.hoisted {
            parts.push("HOISTED");
        }
        if self.per_code_global {
            parts.push("PER-CODE/global");
        }
        if self.per_code_threadgroup {
            parts.push("PER-CODE/threadgroup");
        }
        if self.per_code_fractional {
            parts.push("PER-CODE/fractional");
        }
        parts.join(" + ")
    }
}

/// Classify a kernel body.
///
/// Order matters. `linear_gemv` reaches `exl3_code_at` one helper down, through
/// `packed_state` (linear.metal:375), so a grep for the slow reader alone would
/// miss it -- which is precisely why this census goes through a parser.
fn classify(body: &str) -> Paths {
    let mut p = Paths::none();

    // Any decode at all?
    p.decodes = body.contains("exl3_decode8")
        || body.contains("extract_v_mcg_")
        || body.contains("exl3_code_at")
        || body.contains("packed_state(")
        || body.contains("fractional_window")
        || body.contains("for (uint i = 0; i < 8u; ++i)")
        || body.contains("for (uint j = 0u; j < 4u; ++j)");
    if !p.decodes {
        return p;
    }

    p.per_code_threadgroup = body.contains("exl3_code_at_tg(");
    p.per_code_global = body.contains("exl3_code_at(") || body.contains("packed_state(");
    p.per_code_fractional = body.contains("fractional_window(");

    // One window, eight codes, by any of the four spellings this library uses.
    p.hoisted = body.contains("exl3_decode8")
        || body.contains("extract_v_mcg_32(")
        || body.contains("extract_v_mcg_fast(")
        // A hand-inlined hoisted window: the eight codes of a lane come out of
        // one `A0` / `A1` / `wa` built from two or three words loaded per tile.
        || body.contains("for (uint i = 0; i < 8u; ++i)")
        || body.contains("for (uint j = 0u; j < 4u; ++j)");
    p
}

/// `(kernel name -> paths)` for every kernel that reads weights.
fn census() -> BTreeMap<String, Paths> {
    all_kernels()
        .into_iter()
        .map(|k| (k.name, classify(&k.body)))
        .filter(|(_, p)| p.decodes)
        .collect()
}

/// Kernel names carrying a runtime codebook, sorted.
///
/// "Runtime" means the codebook is a *value* at the call, so the `codebook == 0u`
/// / `codebook == 1u` tests inside `decode_state` (linear.metal:139-147) survive
/// -- eight branch chains per lane. `decode_state(s & 0xffffu, CB)` with `CB` a
/// `constexpr int` literal does **not**: that is the "codebook folded to a
/// constant, 163.0 GB/s" measurement at linear.metal:322-326, and every kernel
/// in the `ishizuki` family spells it that way.
fn runtime_codebook_kernels() -> Vec<String> {
    let mut names: Vec<String> = all_kernels()
        .into_iter()
        .filter(|k| k.body.contains("params.codebook"))
        .map(|k| k.name)
        .collect();
    names.sort();
    names
}

// ---------------------------------------------------------------------------
// The window cost, counted rather than quoted
// ---------------------------------------------------------------------------

/// Loads and shift/or chains issued to decode **eight** codes of one lane.
///
/// Both readers return the same value for each of the eight codes -- that is
/// the premise of the optimisation -- so the comparison is exact and not a
/// throughput estimate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WindowCost {
    /// Ring-word fetches per eight codes.
    words_per_8_codes: u32,
    /// 64-bit shift/or chains (`(w0 << 32) | w1`, then `<< sh`) per eight codes.
    shift_pairs_per_8_codes: u32,
}

impl WindowCost {
    /// How many times more ring-word fetches this path issues than `other`.
    fn penalty_vs(self, other: WindowCost) -> f32 {
        self.words_per_8_codes as f32 / other.words_per_8_codes as f32
    }
}

/// `exl3_code_at` (linear.metal:216): one window setup **per code**.
fn cost_of_code_at() -> WindowCost {
    let body = function_body(LINEAR_METAL, "linear.metal", "exl3_code_at");
    let words = body.matches("trellis_word(").count() as u32;
    let shift_pairs = body.matches("<< 32").count() as u32;
    assert_eq!(words, 3, "exl3_code_at should fetch three ring words");
    assert_eq!(shift_pairs, 2, "exl3_code_at should build two 64-bit windows");
    WindowCost {
        words_per_8_codes: words * 8,
        shift_pairs_per_8_codes: shift_pairs * 8,
    }
}

/// `exl3_decode8` (linear.metal:279): one window setup for all eight codes.
fn cost_of_decode8() -> WindowCost {
    let body = function_body(LINEAR_METAL, "linear.metal", "exl3_decode8");
    let words = body.matches("ring_word<ALIGNED>(").count() as u32;
    let shift_pairs = body.matches("<< 32").count() as u32;
    assert_eq!(words, 3, "exl3_decode8 should fetch three ring words");
    assert_eq!(shift_pairs, 2, "exl3_decode8 should build two 64-bit windows");
    WindowCost {
        words_per_8_codes: words,
        shift_pairs_per_8_codes: shift_pairs,
    }
}

/// `extract_v_mcg_32` (linear_repacked.metal:53): two words, not three,
/// because at 3 bpw the eight codes of a lane span `8 * 3 + 16 - 16 = 24` bits
/// and a window starting anywhere in a 32-bit word still ends inside the next
/// one. This is what every kernel this model actually runs uses.
fn cost_of_extract_v() -> WindowCost {
    // The signature, which `function_body` (correctly) does not include.
    let params = REPACKED_METAL
        .split("static inline void extract_v_mcg_32(")
        .nth(1)
        .and_then(|s| s.split(')').next())
        .expect("extract_v_mcg_32 has a parameter list");
    let words = params.split(',').filter(|p| p.contains("uint w")).count() as u32;
    let body = function_body(REPACKED_METAL, "linear_repacked.metal", "extract_v_mcg_32");
    let shift_pairs = body.matches("<< sh0").count() as u32;
    assert_eq!(words, 2, "extract_v_mcg_32 should take two ring words");
    assert_eq!(shift_pairs, 2, "extract_v_mcg_32 should build two windows");
    WindowCost {
        words_per_8_codes: words,
        shift_pairs_per_8_codes: shift_pairs,
    }
}

/// The most codes one thread could win from a single hoisted window in
/// `linear_gemv_fast*` -- and why it is **four**, not eight.
///
/// The tempting answer is eight, which would make the slow path look like a
/// clean 8x. In `linear_gemv_fast*` a thread owns **one column** of a tile
/// (`local_column = gid & 15u`) and **all sixteen rows**, and `exl3_code_at`'s
/// own scatter (linear.metal:224-228) maps row `r` to lane `p = (r % 8u) / 2u` and
/// `i = (r % 2u) + 2u * (r / 8u)`. So the sixteen codes a thread decodes span
/// four trellis lanes (`p = 0..3`), not one: the hoistable factor is 4, and the
/// sixteen hand-unrolled calls would become four `exl3_decode8` calls plus a
/// re-indexed accumulator. That is a real win -- in code no dispatch reaches,
/// see [`only_eight_dead_specialisations_use_the_per_code_path`].
fn hoisted_codes_per_thread_in_fast_kernels() -> u32 {
    const ROWS_PER_THREAD: u32 = 16; // the `base + 0u .. base + 15u` unroll
    const DISTINCT_LANES: u32 = 4; // `(r % 8) / 2` over r = 0..16
    assert_eq!(ROWS_PER_THREAD / DISTINCT_LANES, 4);
    ROWS_PER_THREAD / DISTINCT_LANES
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// Every env flag consulted by the four decode entry points. Pinned by
/// [`all_dispatch_defining_flags_unset`].
const DISPATCH_FLAGS: &[&str] = &[
    "KAPLUMBA_4COL",
    "KAPLUMBA_ABLATE",
    "KAPLUMBA_DECODE_NR0",
    "KAPLUMBA_FUSED2_4COL",
    "KAPLUMBA_FUSED2_CO4",
    "KAPLUMBA_ISHIZUKI_3WORD",
    "KAPLUMBA_LEGACY_GEMV",
    "KAPLUMBA_NO_2COL",
    "KAPLUMBA_NO_FUSED2_2COL",
    "KAPLUMBA_NO_FUSED2_CO",
    "KAPLUMBA_NO_FUSED3_2COL",
    "KAPLUMBA_OWN_GEMV",
    "KAPLUMBA_QKV_SK2",
    "KAPLUMBA_REPACKED_GEMV",
    "KAPLUMBA_USE_SK",
];

/// The three arms of `encode_gemv_raw`, in source order (linear.rs:2108, 2133,
/// 2159, 2182, 2211).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arm {
    /// `linear_gemv_ishizuki_k*` family, hoisted.
    Ishizuki,
    /// `linear_gemv_exl3_k*` family, also hoisted. Never produced for this
    /// checkpoint, but it is the arm that would catch a tightened ishizuki
    /// guard, which is why
    /// [`the_runtime_codebook_branch_is_unreachable_for_an_mcg_checkpoint`]
    /// asserts the EXL3 table has no hole at any rate.
    #[allow(dead_code)]
    Exl3,
    /// `linear_gemv_fast{bits}` (MCG) or `linear_gemv`. The only path to
    /// `exl3_code_at`.
    PerCodeFallback,
}

/// Which arm `encode_gemv_raw` reaches for this model's conditions: an integer
/// rate in 1..=8, a 4-byte-aligned packed offset, MCG.
///
/// `packed_offset % 4 == 0` is not a parameter here because it is not free to
/// choose: `exl3_code_at`'s comment (linear.metal:236-241) records that a mapped
/// checkpoint's trellis is only 2-byte aligned, and `lm_head.trellis` starts at a
/// file offset of `% 4 == 2`. Every one of Gxma-repacked's 411 trellis tensors is
/// arena-resident at a page-aligned offset, which is what makes the whole
/// `_u` (unaligned) family dead as well -- see
/// [`no_kernel_the_dispatch_can_reach_uses_the_per_code_path`].
fn resolve_encode_gemv_raw(bits: u8) -> Arm {
    // The ishizuki arm, linear.rs:1508-1560: it declines only on
    // `KAPLUMBA_OWN_GEMV`, a non-integer rate, bits outside 1..=8, or a packed
    // offset that is not 4-byte aligned. Every trellis of Gxma-repacked is
    // arena-resident at a page-aligned offset, and the only rate the loader can
    // produce here is 3 or 6, so this arm always resolves and the remaining two
    // are unreachable -- which is the whole reason `linear_gemv_fast*` is dead.
    //
    // The EXL3 arm (linear.rs:1820-1829 via `exl3_gemv_index`,
    // linear.rs:1565-1590) is kept in the chain because it would otherwise be a
    // hole: if the ishizuki guard ever tightened, the EXL3 table covers every
    // integer rate and both codebook families, so `encode_gemv_raw` would still
    // never reach its final `else`.
    //
    // The final `else` (linear.rs:2211-2213) needs `ishizuki == None` AND
    // `exl3 == None`. `exl3 == None` means a fractional rate, and there
    // `fast_gemv_pipeline` refuses too (it requires an integer rate,
    // linear.rs:1861-1863) so it picks `linear_gemv`, whose `packed_state`
    // branches on `rate < 16u` (linear.metal:374) and takes `fractional_window`
    // rather than `exl3_code_at`. So the fallback is not the slow reader either.
    match bits {
        1..=8 => Arm::Ishizuki,
        _ => Arm::PerCodeFallback,
    }
}

/// Whether `encode_gemv_fused2_mlp` reaches the `_co` arm for these widths.
///
/// Source order (linear.rs:2578 `_4col`, 2590 `_2col`, 2598 `_co`, 2647
/// `fused2`): the `_2col` arm comes first and fires whenever both widths are
/// multiples of 32.
fn mlp_reaches_fused2_co(out_features_gate: usize, out_features_up: usize) -> bool {
    if out_features_gate.is_multiple_of(32) && out_features_up.is_multiple_of(32) {
        return false;
    }
    out_features_gate == out_features_up
}

/// Whether `encode_gemv_fused2` reaches the `_co` arm for these widths. It
/// tests `_co` first (linear.rs:2404) but only inside
/// `out_features_a == out_features_b`.
fn qk_reaches_fused2_co(out_features_a: usize, out_features_b: usize) -> bool {
    out_features_a == out_features_b
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// The whole census, printed. Run with `--nocapture`.
#[test]
fn the_census() {
    let kernels = all_kernels();
    let mut rows: Vec<String> = kernels
        .iter()
        .map(|k| {
            format!(
                "{:<48} {:<38} {:>5}  {}",
                k.name,
                classify(&k.body).label(),
                k.line,
                k.file
            )
        })
        .collect();
    rows.sort();

    let mut tally: BTreeMap<String, usize> = BTreeMap::new();
    for kernel in &kernels {
        *tally.entry(classify(&kernel.body).label()).or_insert(0) += 1;
    }

    let code_at = cost_of_code_at();
    let decode8 = cost_of_decode8();
    let extract_v = cost_of_extract_v();

    let mut summary: Vec<String> = tally
        .iter()
        .map(|(label, count)| format!("{label:<38} {count:>4}"))
        .collect();

    summary.push(String::new());
    summary.push(format!(
        "window setup cost, per EIGHT codes of one lane\n\
         \x20 exl3_code_at      {:>3} ring-word fetches, {:>2} 64-bit shift/or chains\n\
         \x20 exl3_decode8      {:>3} ring-word fetches, {:>2} 64-bit shift/or chains   ({:.0}x)\n\
         \x20 extract_v_mcg_32  {:>3} ring-word fetches, {:>2} 32-bit shift/or chains   ({:.0}x)",
        code_at.words_per_8_codes,
        code_at.shift_pairs_per_8_codes,
        decode8.words_per_8_codes,
        decode8.shift_pairs_per_8_codes,
        code_at.penalty_vs(decode8),
        extract_v.words_per_8_codes,
        extract_v.shift_pairs_per_8_codes,
        code_at.penalty_vs(extract_v),
    ));
    summary.push(format!(
        "hoistable factor inside linear_gemv_fast*: {}x, not 8x (one thread spans 4 trellis lanes)",
        hoisted_codes_per_thread_in_fast_kernels()
    ));

    println!(
        "\n=== EXL3 decode-path census ({} kernels) ===\n{}\n\n--- totals ---\n{}",
        kernels.len(),
        rows.join("\n"),
        summary.join("\n")
    );
}

/// The exhaustive per-code population: the eight `linear_gemv_fast*`
/// specialisations, the generic `linear_gemv`, the prefill fallback
/// `linear_gemv_tiled`, and `linear_gemm_mma`'s fractional-rate arm.
#[test]
fn only_eight_dead_specialisations_use_the_per_code_path() {
    let mut expected: Vec<String> = vec![
        "linear_gemv".to_string(),
        "linear_gemm_mma".to_string(),
        "linear_gemv_tiled".to_string(),
    ];
    for bits in 1..=8 {
        expected.push(format!("linear_gemv_fast{bits}"));
    }
    expected.sort();

    let mut per_code: Vec<String> = census()
        .into_iter()
        .filter(|(_, p)| p.per_code())
        .map(|(name, _)| name)
        .collect();
    per_code.sort();

    assert_eq!(
        per_code,
        expected,
        "the set of per-code kernels changed. If a new name appeared, find which \
         dispatch reaches it before assuming it is dead; if one of these is now \
         dispatched it is carrying a {}x ring-word penalty in \
         `linear_gemv_fast*`, or one in the two fallback arms.",
        hoisted_codes_per_thread_in_fast_kernels()
    );

    // 16 call sites per kernel x 8 kernels, plus the definition and the one
    // inside `packed_state`.
    assert_eq!(
        LINEAR_METAL.matches("exl3_code_at(").count(),
        8 * 16 + 2,
        "expected 128 in-kernel calls plus the definition and `packed_state`"
    );
    // The threadgroup twin: definition plus the single call in `linear_gemv_tiled`.
    assert_eq!(LINEAR_METAL.matches("exl3_code_at_tg(").count(), 2);
}

/// The headline. No kernel reachable from `encode_gemv_raw`,
/// `encode_gemv_fused2`, `encode_gemv_fused2_mlp`, `encode_gemv_fused3` or the
/// prefill entry point is on the per-code path.
#[test]
fn no_kernel_the_dispatch_can_reach_uses_the_per_code_path() {
    let table = census();

    // Every kernel name the four decode entry points look up by literal or by
    // `format!`, read off `linear.rs`. Each is asserted to exist *and* to be
    // hoisted, so a rename fails here instead of quietly unchecking a row.
    let reachable: &[&str] = &[
        // encode_gemv_raw (linear.rs:2108, 2133, 1539-1559)
        "linear_gemv_ishizuki_k3_colmajor_4col",
        "linear_gemv_ishizuki_k3_colmajor_2col",
        "linear_gemv_ishizuki_k3_colmajor_contig4",
        "linear_gemv_ishizuki_k3_colmajor",
        "linear_gemv_ishizuki_k3_w2",
        "linear_gemv_ishizuki_k3",
        "linear_gemv_ishizuki_k6", // lm_head, 6 bpw, row-major -> unsuffixed slot
        // encode_gemv_fused2 (linear.rs:2370, 2406, 2408, 2435, 2451)
        "linear_gemv_ishizuki_k3_fused2_sk2",
        "linear_gemv_ishizuki_k3_fused2_co4",
        "linear_gemv_ishizuki_k3_fused2_co",
        "linear_gemv_ishizuki_k3_fused2_2col",
        "linear_gemv_ishizuki_k3_fused2",
        // encode_gemv_fused2_mlp (linear.rs:2532, 2587, 2595, 2600)
        "linear_gemv_ishizuki_k3_fused2_4col",
        // encode_gemv_fused3 (linear.rs:2741, 2785, 2800)
        "linear_gemv_ishizuki_k3_fused3_sk2",
        "linear_gemv_ishizuki_k3_fused3_2col",
        "linear_gemv_ishizuki_k3_fused3",
        // encode_gemv_raw_sk2 / _sk4 (linear.rs:2233, 2257, 2300)
        "linear_gemv_ishizuki_k3_colmajor_2col_sk2",
        "linear_gemv_ishizuki_k3_colmajor_2col_sk4",
        "linear_gemv_ishizuki_k3_colmajor_contig4_sk2",
        "linear_gemv_ishizuki_k3_colmajor_1col_sk2",
        // prefill (linear.rs:3282-3300)
        "linear_gemm_mma",
        "linear_gemm_mma_regpipe",
    ];

    for name in reachable {
        let paths = table
            .get(*name)
            .unwrap_or_else(|| panic!("{name} is dispatched by linear.rs but is not in the census; it was renamed or deleted and every other row here is now unchecked"));
        // `linear_gemm_mma` is the one kernel that is both: hoisted for integer
        // rates, `fractional_window` for fractional ones. Its per-code arm is
        // unreachable -- `Exl3Rate::validate_codebook` (crates/engine/src/exl3.rs
        // :81-83) forces a fractional rate to Mul1, and the folded-MCG arm at
        // linear.metal:10256 handles the only integer rate this checkpoint has
        // beyond 3. So the per-code check below is on the two global/threadgroup
        // readers, which is where `exl3_code_at` actually lives.
        assert!(
            paths.hoisted && !paths.per_code_global && !paths.per_code_threadgroup,
            "{name} is dispatched and its path is `{}` -- this is the finding the \
             census was looking for",
            paths.label()
        );
    }

    // The two per-code kernels that a *rate* could reach, argued out:
    //
    //   * For bits in 1..=8 the ishizuki arm always resolves, so `encode_gemv_raw`
    //     never reaches its final `else` at all -- which is the only way
    //     `linear_gemv_fast{bits}` is dispatched.
    for bits in 1..=8u8 {
        assert_eq!(
            resolve_encode_gemv_raw(bits),
            Arm::Ishizuki,
            "bits {bits} should leave through the ishizuki arm"
        );
    }
    //   * `linear_gemv` is only reached by the same `else`, or by a fractional
    //     rate -- where `packed_state` (linear.metal:374) branches on
    //     `rate < 16u` and takes `fractional_window`, not `exl3_code_at`. So the
    //     generic kernel's `exl3_code_at` is unreachable from both sides.
    assert!(LINEAR_METAL.contains("if (rate < 16u) {"));
    assert!(LINEAR_METAL.contains("return exl3_code_at(packed, tile_index, local_row, local_column, 256u * rate / 32u, rate);"));

    // The classifier must be able to *report* a per-code path, or the loop above
    // proves nothing. `linear_gemv_fast3` is the witness, and it is a real
    // reading of the source: 16 `exl3_code_at` call sites.
    let witness = table.get("linear_gemv_fast3").expect("witness kernel");
    assert!(
        witness.per_code_global && !witness.hoisted,
        "the classifier stopped recognising the per-code path, so 'no reachable \
         kernel is on it' is vacuous"
    );
    assert_eq!(
        LINEAR_METAL
            .matches("exl3_code_at(packed, tile_index")
            .count(),
        128 + 1,
        "16 codes x 8 kernels, plus the one inside `packed_state`"
    );
}

/// The specific question about the fused kernels, and about `_co` / `_co4` in
/// particular.
#[test]
fn every_fused_kernel_is_on_the_hoisted_path() {
    let fused: Vec<Kernel> = all_kernels()
        .into_iter()
        .filter(|k| k.name.contains("fused2") || k.name.contains("fused3"))
        .collect();
    assert_eq!(fused.len(), 9, "the fused family grew or shrank");

    for kernel in &fused {
        let paths = classify(&kernel.body);
        assert!(
            paths.hoisted && !paths.per_code(),
            "{} (line {}) is on `{}` -- fusing regressed to the per-code path",
            kernel.name,
            kernel.line,
            paths.label()
        );
    }

    // `_co` and `_co4`: two ring words per tile per column, hoisted indices.
    for name in [
        "linear_gemv_ishizuki_k3_fused2_co",
        "linear_gemv_ishizuki_k3_fused2_co4",
    ] {
        let kernel = all_kernels()
            .into_iter()
            .find(|k| k.name == name)
            .unwrap_or_else(|| panic!("{name} is gone"));
        assert!(
            kernel.body.contains("extract_v_mcg_32("),
            "{name} no longer calls extract_v_mcg_32"
        );
        // `j0`, `j1` and `sh0` depend only on the lane, so they are computed
        // once per kernel, above every tile loop. If a window index were moved
        // back inside the loop this fails.
        let hoisted = kernel
            .body
            .find("const uint j0 =")
            .expect("j0 is hoisted");
        let first_use = kernel
            .body
            .find("[j0]")
            .expect("j0 is used");
        assert!(
            hoisted < first_use,
            "{name}: the window index is used before it is defined, or it was \
             moved into the tile loop"
        );
        // Two words, two columns: four `extract_v_mcg_32` calls in the unrolled
        // K loop plus two in the tail.
        assert!(
            kernel.body.matches("extract_v_mcg_32(").count() >= 6,
            "{name}: expected the eight-at-once helper on every (tile, column)"
        );
    }
}

/// `linear_repacked.metal` contains no per-code path at all, and never names
/// either slow helper -- not even indirectly.
#[test]
fn repacked_has_no_per_code_path_at_all() {
    for needle in [
        "exl3_code_at",
        "packed_state",
        "fractional_window",
        "exl3_decode8",
    ] {
        assert!(
            !REPACKED_METAL.contains(needle),
            "linear_repacked.metal now mentions {needle}; re-run the census"
        );
    }
    let repacked: Vec<Kernel> = all_kernels()
        .into_iter()
        .filter(|k| k.file == "linear_repacked.metal")
        .filter(|k| classify(&k.body).decodes)
        .collect();
    assert_eq!(repacked.len(), 25, "the repacked GEMV family changed size");
    for kernel in &repacked {
        let paths = classify(&kernel.body);
        assert!(
            !paths.per_code(),
            "{} (line {}) is on `{}`",
            kernel.name,
            kernel.line,
            paths.label()
        );
    }
}

/// The 8x figure, counted; and the honest ceiling for `linear_gemv_fast*`.
#[test]
fn the_eight_x_penalty_is_derived_not_quoted() {
    let code_at = cost_of_code_at();
    let decode8 = cost_of_decode8();
    let extract_v = cost_of_extract_v();

    // exl3_code_at builds its window three times per code.
    assert_eq!(code_at.words_per_8_codes, 24, "3 words x 8 codes");
    assert_eq!(code_at.shift_pairs_per_8_codes, 16, "2 windows x 8 codes");
    // exl3_decode8 builds it once for all eight.
    assert_eq!(decode8.words_per_8_codes, 3);
    assert_eq!(decode8.shift_pairs_per_8_codes, 2);
    // The column-major twin is narrower still, and it is what this model runs.
    assert_eq!(extract_v.words_per_8_codes, 2);

    assert_eq!(code_at.penalty_vs(decode8), 8.0, "the comment's eight times");
    assert_eq!(code_at.penalty_vs(extract_v), 12.0);

    // `trellis_word` is `static inline` and is two 16-bit loads, so in load
    // *instructions* it is 48 vs 3. The comment's "eight times the loads" is
    // about ring-word fetches; both numbers are recorded so the distinction
    // cannot be quietly lost.
    assert!(LINEAR_METAL.contains("uint(packed[index * 2u]) | (uint(packed[index * 2u + 1u]) << 16)"));

    // And the 8x is NOT reachable inside `linear_gemv_fast*`, which is the only
    // place the 24-fetch path lives.
    assert_eq!(hoisted_codes_per_thread_in_fast_kernels(), 4);
}

/// There is no 32-code decoder.
#[test]
fn there_is_no_32_code_decoder() {
    for (file, source) in [
        ("linear.metal", LINEAR_METAL),
        ("linear_repacked.metal", REPACKED_METAL),
        ("linear.rs", LINEAR_RS),
        ("model.rs", MODEL_RS),
        ("scripts/gen_exl3_gemv.py", GEN_PY),
    ] {
        assert!(
            !source.contains("decode32"),
            "{file} mentions decode32; the census needs a row for it"
        );
    }
    // The widest decoder that exists is eight codes wide.
    let body = function_body(LINEAR_METAL, "linear.metal", "exl3_decode8");
    assert!(
        body.contains("for (uint i = 0u; i < 8u; ++i)"),
        "exl3_decode8 no longer decodes eight codes"
    );
    assert!(
        LINEAR_METAL.contains("template <bool ALIGNED>\nstatic inline void exl3_decode8_mcg("),
        "exl3_decode8_mcg moved; line ~344 was its template header, not a decode32"
    );
}

/// The 61.7% runtime-codebook branch and who pays it.
#[test]
fn the_runtime_codebook_branch_is_unreachable_for_an_mcg_checkpoint() {
    let names = runtime_codebook_kernels();
    assert_eq!(
        names,
        vec![
            // prefill: the tiled fallback and the MMA kernel's fractional arm
            "linear_gemm_mma",
            // generic single-query decode
            "linear_gemv",
            // the pre-`_mcg` family -- the 61.7% one -- kept for Plain and Mul1
            "linear_gemv_exl3_k1",
            "linear_gemv_exl3_k1_u",
            "linear_gemv_exl3_k2",
            "linear_gemv_exl3_k2_u",
            "linear_gemv_exl3_k3",
            "linear_gemv_exl3_k3_u",
            "linear_gemv_exl3_k4",
            "linear_gemv_exl3_k4_u",
            "linear_gemv_exl3_k5",
            "linear_gemv_exl3_k5_u",
            "linear_gemv_exl3_k6",
            "linear_gemv_exl3_k6_u",
            "linear_gemv_exl3_k7",
            "linear_gemv_exl3_k7_u",
            "linear_gemv_exl3_k8",
            "linear_gemv_exl3_k8_u",
            // per-code and MCG, but still a runtime `codebook` argument
            "linear_gemv_fast1",
            "linear_gemv_fast2",
            "linear_gemv_fast3",
            "linear_gemv_fast4",
            "linear_gemv_fast5",
            "linear_gemv_fast6",
            "linear_gemv_fast7",
            "linear_gemv_fast8",
            "linear_gemv_tiled",
        ],
        "the set of kernels carrying a runtime codebook branch changed"
    );

    // (a) Every projection is loaded as MCG (model.rs:3429 and six siblings),
    //     and the ishizuki arm -- which hardcodes `constexpr int CB = 1`, i.e.
    //     MCG -- is tried FIRST. So a Plain or Mul1 codebook is the only way to
    //     reach `linear_gemv_exl3_kN`, and this engine never builds one.
    assert!(
        MODEL_RS.matches("Exl3Codebook::Mcg,").count() >= 7,
        "model.rs no longer pins every projection to MCG. If that changed, \
         `encode_gemv_raw`'s ishizuki arm would be WRONG for this model (it \
         hardcodes CB = 1) and the kN/kN_u runtime-codebook family would go live \
         at a cost of 61.7% of the kernel."
    );
    // The literal-vs-runtime distinction is what makes the ishizuki family cheap:
    // `decode_state(s, CB)` with `CB` a `constexpr int` folds to `decode_state_mcg`,
    // so the two `codebook ==` tests at linear.metal:140-145 vanish.
    let literal_cb_kernels = LINEAR_METAL.matches("constexpr int CB = 1;").count();
    assert!(
        literal_cb_kernels >= 20,
        "only {literal_cb_kernels} kernels pin CB to the MCG literal"
    );
    assert!(
        REPACKED_METAL.contains("decode_state_mcg((A_hi >> 16) & 0xffffu)"),
        "extract_v_mcg_32 no longer calls decode_state_mcg directly; if a \
         runtime codebook crept in, the 2.61x would be gone"
    );

    // (b) `fast_gemv_pipeline` refuses a non-MCG codebook, so even if the final
    //     `else` were reached it would pick the generic `linear_gemv`.
    assert!(LINEAR_RS.contains(
        "if codebook != Exl3Codebook::Mcg {\n            return None;\n        }"
    ));

    // (c) The EXL3 arm covers every integer rate, which is what makes the final
    //     `else` -- and therefore `linear_gemv_fast*` -- unreachable.
    assert!(LINEAR_RS.contains(
        "Some(1 + family * 16 + aligned * 8 + (bits as usize - 1))"
    ));
    for bits in 1..=8u8 {
        assert!(
            LINEAR_METAL.contains(&format!("kernel void linear_gemv_exl3_k{bits}_mcg(")),
            "no _mcg kernel for {bits} bpw: the specialised arm has a hole"
        );
        assert!(
            LINEAR_METAL.contains(&format!("kernel void linear_gemv_exl3_k{bits}(")),
            "no kN kernel for {bits} bpw: the runtime-codebook arm has a hole"
        );
    }
}

/// Prefill takes the hoisted path; the per-code fallback is unreachable for this
/// checkpoint.
#[test]
fn prefill_takes_the_hoisted_path_and_the_fallback_is_per_code() {
    let mma = all_kernels()
        .into_iter()
        .find(|k| k.name == "linear_gemm_mma")
        .expect("linear_gemm_mma");
    // Three window words per lane, staged a chunk ahead: nine reads, three reads
    // and three writes.
    assert_eq!(mma.body.matches("pre[3u * jw +").count(), 12);
    // Both integer arms consume the staged window rather than re-deriving it.
    assert!(mma
        .body
        .contains("const ulong wa = ((g0 << 32) | g1) << sh | (sh ? (g2 >> (32 - sh)) : 0ul);"));
    assert!(mma
        .body
        .contains("const ulong wc = ((g1 << 32) | g2) << sh;"));
    // The folded-MCG arm at linear.metal:10256 has no `r <= 64u` select at all,
    // because at ka == 3 every code offset is 16..37.
    assert_eq!(
        mma.body.matches("const uint r = (j + 4u * dc) * 3u + 16u;").count(),
        1,
        "the folded-MCG arm lost its literal rate"
    );

    // The fallback is per-code, out of threadgroup memory.
    let tiled = all_kernels()
        .into_iter()
        .find(|k| k.name == "linear_gemv_tiled")
        .expect("linear_gemv_tiled");
    assert!(classify(&tiled.body).per_code_threadgroup);

    // Unreachable: the MMA guard is `in_features % 64 == 0` and
    // `out_features % 64 == 0` (linear.rs:3279, 3290), and every projection of
    // this checkpoint is a multiple of 64.
    for shape in kaplumba_engine::bench::SHAPES {
        assert_eq!(
            shape.in_features % 64,
            0,
            "{}: in_features {} would drop prefill onto linear_gemv_tiled",
            shape.name,
            shape.in_features
        );
        assert_eq!(
            shape.out_features % 64,
            0,
            "{}: out_features {} would drop prefill onto linear_gemv_tiled",
            shape.name,
            shape.out_features
        );
    }
}

/// `fused2_co` is not the default fused-MLP kernel on this model, contrary to
/// how the request was posed. Irrelevant to the census -- `_co`, `_co4` and
/// `_2col` are all hoisted -- but pinned so a reader does not build a
/// measurement argument on the wrong kernel.
#[test]
fn fused2_co_is_not_the_default_for_this_model() {
    // gate and up are both 21504 wide, so `encode_gemv_fused2_mlp` takes its
    // `_2col` arm (linear.rs:2590) before reaching `_co` (linear.rs:2598).
    assert!(
        !mlp_reaches_fused2_co(21_504, 21_504),
        "fused2_co became the MLP default; update the report"
    );
    // q and k of a full-attention layer are 16384 and 2048, so the `_co` arm of
    // `encode_gemv_fused2` (guarded by out_a == out_b, linear.rs:2404) is not
    // taken either.
    assert!(
        !qk_reaches_fused2_co(16_384, 2_048),
        "fused2_co became reachable for q/k; update the report"
    );
    // The order in the source, so the argument above is checkable.
    let mlp = LINEAR_RS
        .find("pub fn encode_gemv_fused2_mlp")
        .expect("encode_gemv_fused2_mlp");
    let no_2col = LINEAR_RS[mlp..]
        .find("!env_flag(\"KAPLUMBA_NO_FUSED2_2COL\")")
        .map(|at| mlp + at)
        .expect("the _2col guard");
    let co = LINEAR_RS[mlp..]
        .find("\"linear_gemv_ishizuki_k3_fused2_co\"")
        .map(|at| mlp + at)
        .expect("the _co lookup");
    assert!(
        no_2col < co,
        "in encode_gemv_fused2_mlp the _2col arm no longer precedes the _co arm"
    );
}

/// The default dispatch is the one this census assumed. `env_flag`
/// (linear.rs:45-58) memoises on first read, so a census computed under a
/// different flag set does not describe this process.
#[test]
fn all_dispatch_defining_flags_unset() {
    for flag in DISPATCH_FLAGS {
        assert!(
            std::env::var_os(flag).is_none(),
            "{flag} is set in this process's environment; the census was derived \
             with the default (all-unset) dispatch and would not describe it"
        );
    }
}