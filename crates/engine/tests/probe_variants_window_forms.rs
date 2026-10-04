//! PROBE — the three window forms of the ishizuki EXL3 decode matvec: plain,
//! `_w2` (two-word) and `_nod` (no-decode).
//!
//! **Question.** `crates/kernels/src/linear.metal` carries three variants of
//! `linear_gemv_ishizuki_k{N}`, emitted by `scripts/gen_exl3_gemv.py`:
//!
//! | form | emitted at | dispatched by the engine? | what it changes |
//! |---|---|---|---|
//! | `linear_gemv_ishizuki_k{1..8}` | every rate | yes, as the per-rate fallback slot | — |
//! | `linear_gemv_ishizuki_k{1..4}_w2` | `TWO_WORD_SAFE_RATES` | **yes, by name lookup** (`linear.rs:1551-1558`) | drops one ring load per lane per tile |
//! | `linear_gemv_ishizuki_k{1..8}_nod` | `NO_DECODE_PROBE_RATES = tuple(range(1,9))` | **no** | replaces `decode_state` with a one-AND stub |
//!
//! Four things are ambiguous and worth settling without a GPU:
//!
//! 1. Is the `two_word_is_safe` derivation *correct*, or merely self-consistent?
//! 2. Could a 2-word form be made safe at 6 bpw (the head rate) or 8 bpw?
//! 3. For which of the model's rates does the dispatch fall through to the
//!    3-word slot — and is the head silently on the slow form?
//! 4. Is the `_nod` family live instrumentation or finished, dead generated
//!    code, and what does it cost?
//!
//! **What this file is.** Two things, both GPU-free.
//!
//! 1. **A CPU port of the window arithmetic** — `lo0/sh0/hi0/j0/j1/j2`, `A0`,
//!    `B0`, and the `(r <= 64u) ? uint(A0 >> (64u - r)) : uint(B0 >> (96u - r))`
//!    extraction — with the `KA` constant and the `lo0` multiplier read out of
//!    each kernel's own source text. That is what makes the derivation a claim
//!    about the shipped file rather than a restatement of it.
//! 2. **Source-level censuses** via `include_str!`, for the same reason
//!    `perf_probe_host_ops.rs:60` gives.
//!
//! **What this file is NOT.** It measures no cycle and loads no weights. The
//! tok/s figures quoted in the comments are the ones already recorded in
//! `spec/decode-bandwidth-240gb/`; nothing here produced them and nothing here
//! may be quoted as if it did.
//!
//! **Sabotage discipline** (AGENTS.md §3.4). `the_matchers_can_fail` mutates
//! in-memory copies of the sources and asserts each matcher flips. A coverage
//! claim whose branches were never observed failing is not a claim.

use kaplumba_kernels::linear::LinearKernels;

const LINEAR_METAL: &str = include_str!("../../kernels/src/linear.metal");
const LINEAR_REPACKED: &str = include_str!("../../kernels/src/linear_repacked.metal");
const LINEAR_RS: &str = include_str!("../../kernels/src/linear.rs");
const GENERATOR: &str = include_str!("../../../scripts/gen_exl3_gemv.py");
const GEMV_SHAPE: &str = include_str!("../examples/gemv_shape.rs");

// ===========================================================================
// 0. Source-level helpers
// ===========================================================================

/// `(1-based line, name, body)` for every `kernel void` in `src`.
///
/// The body ends at the first line that is exactly `}`, so a count taken over
/// a whole file cannot pick up the next kernel's epilogue — the scoping
/// argument `probe_reduce_epilogue.rs:119-122` gives.
fn kernels(src: &str) -> Vec<(usize, String, String)> {
    let lines: Vec<&str> = src.split_inclusive('\n').collect();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < lines.len() {
        if let Some(rest) = lines[i].strip_prefix("kernel void ") {
            let name = rest.split('(').next().unwrap_or("").trim().to_string();
            let mut j = i;
            while j < lines.len() && lines[j].trim_end() != "}" {
                j += 1;
            }
            let end = j.min(lines.len() - 1);
            out.push((i + 1, name, lines[i..=end].concat()));
            i = end + 1;
        } else {
            i += 1;
        }
    }
    out
}

/// The two halves of the *linear* library, as `LinearKernels::new` sees them:
/// `combined_source()` is `linear.metal` + `\n` + `linear_repacked.metal`
/// (`linear.rs:24-30`).
fn linear_library() -> Vec<(usize, String, String, &'static str)> {
    let mut out = Vec::new();
    for (file, src) in [
        ("linear.metal", LINEAR_METAL),
        ("linear_repacked.metal", LINEAR_REPACKED),
    ] {
        for (line, name, body) in kernels(src) {
            out.push((line, name, body, file));
        }
    }
    out
}

/// The lines of `body` that are code, i.e. with comments, blanks and the
/// `kernel void NAME(` declaration line dropped.
///
/// Used to compare a variant against its source kernel: "identical apart from
/// the substitution" is a much sharper claim than "looks the same", and it
/// cannot be satisfied by a kernel that differs anywhere else. The declaration
/// line is dropped because a variant's *name* differs by construction — that
/// is the point of the variant, not a change to its arithmetic.
fn code_lines(body: &str) -> Vec<String> {
    body.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with("//") && !l.starts_with("kernel void "))
        .map(str::to_string)
        .collect()
}

/// `(name, start byte, end byte)` for every `kernel void` in `src`, where the
/// extent runs to the next kernel's declaration.
///
/// Byte extents rather than body lengths, because the question the census asks
/// is how many bytes of *source* a family contributes to the string handed to
/// `new_library_with_source`.
fn kernel_spans(src: &str) -> Vec<(String, usize, usize)> {
    let all = kernels(src);
    let mut out = Vec::new();
    for (i, (_, name, body)) in all.iter().enumerate() {
        let start = src.find(body.as_str()).unwrap_or(0);
        let end = all
            .get(i + 1)
            .and_then(|(_, _, b)| src.find(b.as_str()))
            .unwrap_or(src.len())
            .max(start);
        out.push((name.clone(), start, end));
    }
    out
}

/// Every ring-word load in `body`, as `(variable, index expression)`.
///
/// The traffic property the `_nod` claim rests on, counted rather than
/// searched for: a probe that dropped one of its three loads would still
/// contain `w0c` somewhere, but it would not have the same load *list*.
fn ring_loads(body: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for code in code_lines(body) {
        let rest = code
            .strip_prefix("const ulong ")
            .or_else(|| code.strip_prefix("uint "))
            .unwrap_or(&code);
        let mut names: Vec<(&str, &str)> = Vec::new();
        for part in rest.split(',') {
            let part = part.trim();
            if let Some((lhs, rhs)) = part.split_once('=') {
                let (lhs, rhs) = (lhs.trim(), rhs.trim());
                if rhs.starts_with('t') && rhs.contains('[') {
                    let idx = rhs[rhs.find('[').unwrap()..].to_string();
                    names.push((lhs, Box::leak(idx.into_boxed_str())));
                }
            }
        }
        for (lhs, idx) in names {
            out.push((lhs.to_string(), idx.to_string()));
        }
    }
    out
}

/// 1-based line number of the first line of `src` containing `needle`, or `None`.
fn line_of(src: &str, needle: &str) -> Option<usize> {
    src.lines()
        .enumerate()
        .find(|(_, l)| l.contains(needle))
        .map(|(n, _)| n + 1)
}

/// `const uint KA = 3u;` — the rate the kernel was specialised for.
fn ka_of(body: &str) -> Option<u8> {
    let at = body.find("constexpr uint KA = ")?;
    let rest = &body[at + "constexpr uint KA = ".len()..];
    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

/// The literal multiplier in `const uint lo0 = (lane * 8u + 1u) * Nu + RING - 16u;`
///
/// Read separately from `ka_of` so the two cannot silently agree by
/// construction: the model below computes `lo0` from *this* number, so if
/// someone edits the `lo0` formula the model follows the edit and
/// `the_ka_constant_and_the_lo0_multiplier_agree` is what notices.
fn lo0_multiplier_of(body: &str) -> Option<u32> {
    let at = body.find("(lane * 8u + 1u) * ")?;
    let rest = &body[at + "(lane * 8u + 1u) * ".len()..];
    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

// ===========================================================================
// 1. The window arithmetic, as a CPU model
// ===========================================================================

/// `linear_gemv_ishizuki_k{N}`'s window, ported to the host.
///
/// `RING = 256u * KA`, `WORDS = RING / 32u` (`linear.metal:1833-1835`) and the
/// per-lane setup is `linear.metal:1853-1858`:
///
/// ```text
/// lo0 = (lane * 8 + 1) * KA + RING - 16
/// sh0 = lo0 % 32;  hi0 = lo0 / 32
/// j0 = hi0 >= WORDS ? hi0 - WORDS : hi0      // and +1 for j1, j2
/// ```
#[derive(Clone, Copy, Debug)]
struct Window {
    bits: u32,
    ring: u32,
    words: u32,
}

impl Window {
    fn new(bits: u32) -> Self {
        let ring = 256 * bits;
        Self {
            bits,
            ring,
            words: ring / 32,
        }
    }

    /// `(lo0, sh0, hi0, [j0, j1, j2])`.
    fn indices(&self, lane: u32) -> (u32, u32, u32, [u32; 3]) {
        let lo0 = (lane * 8 + 1) * self.bits + self.ring - 16;
        let sh0 = lo0 % 32;
        let hi0 = lo0 / 32;
        let wrap = |j: u32| if j >= self.words { j - self.words } else { j };
        let j0 = wrap(hi0);
        let j1 = wrap(j0 + 1);
        let j2 = wrap(j1 + 1);
        (lo0, sh0, hi0, [j0, j1, j2])
    }

    /// `r = i * KA + 16`, the largest being `7 * KA + 16`.
    fn r_max(&self) -> u32 {
        7 * self.bits + 16
    }

    /// The lowest A0 bit any code reads: `64 - r_max = 48 - 7*KA`.
    ///
    /// Signed, because it goes negative at KA = 7 and 8 — which is itself
    /// information: "the used bits start below bit 0" is exactly what makes
    /// those rates unservable from a 64-bit window.
    fn used_lo(&self) -> i64 {
        48 - 7 * self.bits as i64
    }

    /// `A0 = ((w0a << 32) | w0b) << sh0 | (sh0 ? (w0c >> (32 - sh0)) : 0)`.
    fn a0_three(&self, w: [u32; 3], sh0: u32) -> u64 {
        let pair = ((w[0] as u64) << 32) | w[1] as u64;
        let tail = if sh0 != 0 {
            (w[2] >> (32 - sh0)) as u64
        } else {
            0
        };
        (pair << sh0) | tail
    }

    /// `A0 = ((w0a << 32) | w0b) << sh0` — the `_w2` form, which has no `w0c`.
    ///
    /// The `& u64::MAX` is written out rather than left implicit: Metal's
    /// `ulong` shift is the C++ unsigned shift, i.e. modulo 2^64, and spelling
    /// the truncation is what makes the port auditable rather than relying on
    /// the reader knowing Rust agrees.
    #[allow(clippy::identity_op)]
    fn a0_two(&self, w: [u32; 2], sh0: u32) -> u64 {
        ((((w[0] as u64) << 32) | w[1] as u64) << sh0) & u64::MAX
    }

    /// `B0 = ((w0b << 32) | w0c) << sh0`.
    #[allow(clippy::identity_op)]
    fn b0(&self, w: [u32; 3], sh0: u32) -> u64 {
        ((((w[1] as u64) << 32) | w[2] as u64) << sh0) & u64::MAX
    }
}

/// `(r <= 64u) ? uint(A0 >> (64u - r)) : uint(B0 >> (96u - r))`, masked to 16 bits.
///
/// `uint(...)` truncates the 64-bit shift to 32 and `& 0xffffu` then keeps the
/// low 16, so code `i` reads A0 bits `[64-r, 79-r]` (or B0 bits `[96-r, 111-r]`).
fn extract(a0: u64, b0: u64, bits: u32, i: u32) -> u32 {
    let r = i * bits + 16;
    let raw = if r <= 64 {
        a0 >> (64 - r)
    } else {
        b0 >> (96 - r)
    };
    (raw & 0xffff) as u32
}

/// Deterministic ring contents. A fixed LCG rather than `rand`, so a failure is
/// reproducible from the seed in the panic message alone.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
}

/// The two-word verdict at `bits`, computed exactly as `two_word_is_safe`
/// (`scripts/gen_exl3_gemv.py:107-144`) does.
fn two_word_verdict(bits: u32) -> (bool, u32, i64, u32, String) {
    let w = Window::new(bits);
    let r_max = w.r_max();
    let used_lo = w.used_lo();
    let sh_max = (0..32).map(|lane| w.indices(lane).1).max().unwrap();
    let why = if r_max > 64 {
        format!("r_max={r_max} > 64, so the B0 path reads w0c")
    } else if used_lo < sh_max as i64 {
        format!("lowest used A0 bit {used_lo} < max sh {sh_max}, so some lane reads w0c")
    } else {
        format!("used bits start at {used_lo} >= every lane's sh (max {sh_max}), and r_max={r_max} <= 64")
    };
    (
        r_max <= 64 && used_lo >= sh_max as i64,
        r_max,
        used_lo,
        sh_max,
        why,
    )
}

// ===========================================================================
// 2. The derivation, recomputed from the shipped source
// ===========================================================================

/// The rate constants and the `lo0` multiplier must agree in every kernel of
/// the family, because the model in §1 takes `lo0`'s multiplier from the source
/// and the rate from `KA`. If they diverge the model is measuring a kernel that
/// does not exist.
#[test]
fn the_ka_constant_and_the_lo0_multiplier_agree() {
    // Every kernel carrying BOTH a `constexpr uint KA` and the
    // `(lane * 8u + 1u) * N + RING - 16u` window setup: the three forms the
    // generator emits from that preamble, plus every repacked variant derived
    // from them.
    let family: Vec<_> = linear_library()
        .into_iter()
        .filter(|(_, _, body, _)| ka_of(body).is_some() && lo0_multiplier_of(body).is_some())
        .collect();
    assert_eq!(family.len(), 43, "the window-carrying kernel count moved");
    let with_w2 = family
        .iter()
        .filter(|(_, n, _, _)| n.ends_with("_w2"))
        .count();
    let with_nod = family
        .iter()
        .filter(|(_, n, _, _)| n.ends_with("_nod"))
        .count();
    assert_eq!(
        (with_w2, with_nod),
        (4, 8),
        "the emitted window-form families moved"
    );
    for (line, name, body, file) in &family {
        let ka =
            ka_of(body).unwrap_or_else(|| panic!("{file}:{line} {name}: no `constexpr uint KA`"));
        let mult = lo0_multiplier_of(body)
            .unwrap_or_else(|| panic!("{file}:{line} {name}: no `(lane * 8u + 1u) * N` in lo0"));
        assert_eq!(
            u32::from(ka),
            mult,
            "{file}:{line} {name}: KA = {ka} but lo0 multiplies by {mult}; the CPU model \
             would be measuring a kernel that is not this one"
        );
        assert!(
            body.contains("RING = 256u * KA"),
            "{file}:{line} {name}: RING is not 256*KA"
        );
        assert!(
            body.contains("WORDS = RING / 32u"),
            "{file}:{line} {name}: WORDS is not RING/32"
        );
    }
}

/// The verdict at every rate, with both failure modes distinguished.
///
/// The table is the answer to question 2. Note that **the family stops at k4
/// because of the shift condition, not the span condition** — `r_max <= 64`
/// would allow KA = 6. `crates/engine/tests/rate4_8_kernel_coverage.rs:20-21`
/// describes the derivation as "valid for KA <= 6", which is condition (a)
/// alone and is wrong; `implementation_notes_rate4_appendix.md:75-80` records
/// the hardware consequence of believing it (1023 of 1024 fp16 outputs wrong).
#[test]
fn two_word_safety_is_derived_from_the_source_and_stops_at_four() {
    let mut safe = Vec::new();
    let mut rows = Vec::new();
    for bits in 1..=8u32 {
        let (ok, r_max, used_lo, sh_max, why) = two_word_verdict(bits);
        rows.push(format!(
            "KA={bits}  r_max={r_max:2} (<=64: {})  used_lo={used_lo:3}  max_sh={sh_max:2}  \
             (<=: {})  -> {:<5} {why}",
            r_max <= 64,
            used_lo >= sh_max as i64,
            if ok { "SAFE" } else { "unsafe" }
        ));
        if ok {
            safe.push(bits as u8);
        }
    }
    println!("{:#?}", rows);
    assert_eq!(
        safe,
        vec![1, 2, 3, 4],
        "the derived two-word set moved; the emitted family and `\
        decode_matvec_golden.rs:106-112` must move with it"
    );
    // The two failure modes are distinguishable, so "unsafe at 5..8" is not a
    // single undifferentiated fact.
    let span = (1..=6u32).all(|b| two_word_verdict(b).1 <= 64);
    assert!(span, "r_max = 7*KA+16 <= 64 holds for KA <= 6");
    assert!(two_word_verdict(7).1 > 64 && two_word_verdict(8).1 > 64);
    assert!(
        two_word_verdict(5).2 < two_word_verdict(5).3 as i64,
        "KA=5 must fail the shift condition"
    );
    assert!(
        two_word_verdict(6).2 < two_word_verdict(6).3 as i64,
        "KA=6 must fail the shift condition"
    );
}

/// 3 and 4 bpw sit **exactly** on the boundary, 1 and 2 have 16 bits of margin.
/// This is the load-bearing detail behind the "do not assume it extends" note
/// (`implementation_notes_rate4_appendix.md:84-88`).
#[test]
fn three_and_four_bpw_sit_exactly_on_the_condition() {
    for (bits, margin) in [(1u32, 16i64), (2, 16), (3, 0), (4, 0)] {
        let (_, _, used_lo, sh_max, _) = two_word_verdict(bits);
        assert_eq!(
            used_lo - sh_max as i64,
            margin,
            "KA={bits}: margin moved to {margin}"
        );
    }
}

/// The decisive test: run the real window arithmetic on real (pseudo-random)
/// ring contents and compare the 3-word extraction with the 2-word one, for
/// every lane and every code.
///
/// It asserts equality where the derivation says equality holds **and**
/// inequality at 5 and 6 bpw. The second half is what stops the first half
/// being vacuous — a comparison that cannot fail proves nothing.
#[test]
fn the_two_word_extraction_is_bit_identical_exactly_where_the_derivation_says() {
    for bits in 1..=6u32 {
        let w = Window::new(bits);
        let (derived_safe, ..) = two_word_verdict(bits);
        let mut differing_lanes = 0usize;
        let mut differing_codes = 0usize;
        let mut first_bad_lane = None;
        let mut rng = Lcg(0x5eed_0000 + bits as u64);
        let ring: Vec<u32> = (0..w.words).map(|_| rng.next()).collect();
        for lane in 0..32u32 {
            let (_, sh0, _, j) = w.indices(lane);
            let three = [
                ring[j[0] as usize],
                ring[j[1] as usize],
                ring[j[2] as usize],
            ];
            let a0 = w.a0_three(three, sh0);
            let b0 = w.b0(three, sh0);
            let a0_two = w.a0_two([three[0], three[1]], sh0);
            let mut lane_differs = false;
            for i in 0..8u32 {
                let c3 = extract(a0, b0, bits, i);
                let c2 = extract(a0_two, 0, bits, i);
                if c3 != c2 {
                    lane_differs = true;
                    differing_codes += 1;
                }
            }
            if lane_differs {
                differing_lanes += 1;
                first_bad_lane.get_or_insert(lane);
            }
        }
        let observed_safe = differing_lanes == 0;
        assert_eq!(
            observed_safe,
            derived_safe,
            "KA={bits}: derivation says {derived_safe} but {differing_lanes}/32 lanes \
             disagree ({differing_codes} codes), first at lane {first_bad_lane:?} (sh0={})",
            first_bad_lane.map(|l| w.indices(l).1).unwrap_or(0)
        );
        if bits == 5 || bits == 6 {
            assert_eq!(
                differing_lanes, 16,
                "KA={bits}: exactly half the lanes need the third word; if this moved, the \
                 6 bpw discussion in question 2 has to be rewritten"
            );
        }
    }
    // 7 and 8 bpw: the 2-word extraction does not exist, so the test is not
    // allowed to reach them — `r_max > 64` means `A0 >> (64 - r)` would be a
    // negative shift count.
    for bits in [7u32, 8] {
        assert!(
            two_word_verdict(bits).1 > 64,
            "KA={bits}: r_max should exceed 64"
        );
    }
}

/// Why 6 bpw cannot join the family even though `r_max = 58 <= 64`.
///
/// The third word is not dead at 6 bpw because **half** the lanes still read it:
/// `sh0 = (16*lane + 22) % 32`, so even lanes sit at 22 and odd lanes at 6,
/// against `used_lo = 48 - 42 = 6`. The 16 lanes at `sh0 = 6` are fine; the 16
/// at `sh0 = 22` read A0 bits `[6, 21]`, which only `w0c` reaches
/// (`A0`'s low `sh0` bits are the *only* contribution of `w0c`).
///
/// A `sh0`-keyed branch cannot rescue it either: `sh0` differs *within* a
/// 32-lane simdgroup, so both arms execute and every lane still issues three
/// loads. And it would be worth nothing — 6 bpw is already at the DRAM ceiling
/// (215.7 GB/s against 216.4 at 4 bpw, recorded at
/// `spec/decode-bandwidth-240gb/implementation_notes.md:444-445`), and for the
/// shipped checkpoint the head never reaches the ishizuki arm at all: its
/// trellis starts at `packed_offset % 4 == 2`, so `linear.rs:1520` returns
/// `None` before the `_w2` lookup is even reached.
#[test]
fn a_two_word_form_at_six_bpw_would_only_serve_half_the_simdgroup() {
    let w = Window::new(6);
    let used_lo = w.used_lo();
    let ok: Vec<u32> = (0..32)
        .filter(|l| used_lo >= w.indices(*l).1 as i64)
        .collect();
    let bad: Vec<u32> = (0..32)
        .filter(|l| used_lo < w.indices(*l).1 as i64)
        .collect();
    assert_eq!(used_lo, 6);
    assert_eq!(
        ok.len(),
        16,
        "the lanes a 2-word window could serve at 6 bpw"
    );
    assert_eq!(bad.len(), 16, "the lanes that need the third word at 6 bpw");
    assert!(
        (0..32).all(|l| w.indices(l).1 == 6 || w.indices(l).1 == 22),
        "the 6 bpw lane shifts are 6 and 22; a change here invalidates the half-and-half claim"
    );
    // Both values occur inside one simdgroup, so the split is divergent and a
    // branch would not remove a load from any lane.
    assert!(
        (0..32).any(|l| w.indices(l).1 == 6) && (0..32).any(|l| w.indices(l).1 == 22),
        "a uniform lane set would make the 6 bpw answer different"
    );
    // And the shipped head is not on this path anyway.
    let guard = line_of(LINEAR_RS, "packed_offset % 4 != 0").expect("the alignment guard");
    let lookup = line_of(LINEAR_RS, "_w2\"").expect("the _w2 name lookup");
    assert!(
        guard < lookup,
        "the `packed_offset % 4` guard is at linear.rs:{guard} but the `_w2` lookup is at \
         :{lookup}; if the lookup moved above the guard, an unaligned tensor could reach it"
    );
}

// ===========================================================================
// 3. What is actually emitted, and who dispatches it
// ===========================================================================

/// The emitted `_w2` set is exactly the derived set, and each `_w2` body is the
/// plain kernel with the third word's contribution gone: no `w?c`/`a2` load, no
/// `B?` local, no extraction ternary, two ring loads per tile instead of three.
#[test]
fn the_two_word_family_is_exactly_the_derived_rates_and_really_is_two_words() {
    let plain_names: Vec<String> = (1..=8u8)
        .map(|b| format!("linear_gemv_ishizuki_k{b}"))
        .collect();
    let plain: Vec<(String, String)> = linear_library()
        .into_iter()
        .filter(|(_, name, _, _)| plain_names.contains(name))
        .map(|(_, n, b, _)| (n, b))
        .collect();
    assert_eq!(
        plain.len(),
        8,
        "the unsuffixed per-rate kernels are the dispatch slots"
    );
    let w2: Vec<(String, String)> = linear_library()
        .into_iter()
        .filter(|(_, name, _, _)| name.ends_with("_w2"))
        .map(|(_, n, b, _)| (n, b))
        .collect();
    assert_eq!(w2.len(), 4, "expected the two-word family at 4 rates");
    let mut rates: Vec<u8> = w2.iter().filter_map(|(_, body)| ka_of(body)).collect();
    rates.sort();
    assert_eq!(rates, vec![1, 2, 3, 4]);
    // The RATES agreeing is not the NAMES agreeing, and a renamed kernel is a
    // silent fall-through at dispatch, so the names are pinned too.
    let names: Vec<String> = w2.iter().map(|(n, _)| n.clone()).collect();
    assert_eq!(
        names,
        vec![
            "linear_gemv_ishizuki_k1_w2",
            "linear_gemv_ishizuki_k2_w2",
            "linear_gemv_ishizuki_k3_w2",
            "linear_gemv_ishizuki_k4_w2",
        ],
        "the two-word family's NAMES moved, not just its rates"
    );

    for (name, body) in &w2 {
        let base = name.trim_end_matches("_w2").to_string();
        let original = plain
            .iter()
            .find(|(n, _)| *n == base)
            .map(|(_, b)| b)
            .unwrap_or_else(|| panic!("no plain kernel for {name}"));
        assert!(
            !body.contains("w0c") && !body.contains("w1c"),
            "{name} still loads the third word"
        );
        assert!(
            !body.contains("a2"),
            "{name} still loads the tail third word"
        );
        assert!(
            !body.contains("const ulong B"),
            "{name} kept a dead B? local"
        );
        assert!(
            !body.contains("(r <= 64u) ?"),
            "{name} kept the extraction ternary"
        );
        // Two ring loads per tile, against the plain kernel's three.
        let loads_w2 = body.matches("[j").count();
        let loads_plain = original.matches("[j").count();
        assert_eq!(
            loads_w2 * 3,
            loads_plain * 2,
            "{name}: {loads_w2} vs {loads_plain} ring loads"
        );
    }
    // No kernel exists above the derived set, which is the fact the dispatch
    // fall-through depends on.
    for bits in 5..=8u8 {
        let name = format!("linear_gemv_ishizuki_k{bits}_w2");
        assert!(
            !LINEAR_METAL.contains(&format!("kernel void {name}(")),
            "{name} exists; TWO_WORD_SAFE_RATES stops at 4 and the derivation says why"
        );
    }
}

/// The work `_w2` saves, stated exactly: **one 32-bit ring load per lane per
/// tile**, i.e. 32 loads per tile out of 96.
///
/// Amplification is lane-loads per ring word, and it is the quantity the
/// recorded measurements attribute 3 bpw's deficit to (2.67x against 2.00x at
/// 4 bpw, `implementation_notes.md:459-461`).
#[test]
fn two_word_cuts_lane_loads_per_tile_by_a_third_where_it_is_legal() {
    println!("\n  rate  words/tile  lane-loads/tile (3-word -> 2-word)   amplification");
    for bits in 1..=8u32 {
        let w = Window::new(bits);
        let words = f64::from(w.words);
        let three = 32.0 * 3.0 / words;
        let two = 32.0 * 2.0 / words;
        let (legal, r_max, used_lo, sh_max, why) = two_word_verdict(bits);
        println!(
            "  {bits:>4}  {:>10}  {:>17.3} ->{two:>8.3}   legal={legal:<5}  r_max={r_max:2} \
             used_lo={used_lo:3} max_sh={sh_max:2}  {why}",
            w.words, three
        );
        // The arithmetic saving is the same everywhere a variant could exist:
        // exactly one third of the lane loads.
        assert!(
            (three / two - 1.5).abs() < 1e-9,
            "KA={bits}: the ratio is not exactly 1.5"
        );
        // But it is only REACHABLE where the derivation says so. At 5 and 6 the
        // load count would improve and the arithmetic is wrong; at 7 and 8 the
        // two-word extraction does not exist at all.
        assert!(
            two < three,
            "KA={bits}: a 2-word window always issues fewer lane loads"
        );
        if !legal {
            assert!(
                bits >= 5 || r_max > 64,
                "KA={bits} is unsafe but for an unstated reason"
            );
        }
    }
    // The headline number, from the recorded interleaved sweep
    // (`spec/decode-bandwidth-240gb/implementation_notes.md:459-461`):
    // post-T6 amplification at 3 bpw is 2.67x against 2.00x at 4 bpw.
    let w3 = Window::new(3);
    assert_eq!(w3.words, 24);
    assert!(
        (64.0_f64 / 24.0 - 2.6667).abs() < 1e-3,
        "3 bpw two-word amplification"
    );
    let w4 = Window::new(4);
    assert_eq!(w4.words, 32);
    assert!(
        (64.0_f64 / 32.0 - 2.0).abs() < 1e-9,
        "4 bpw two-word amplification"
    );
}

/// Which of the model's rates falls through to the 3-word slot.
///
/// `ishizuki_gemv_pipeline` (`linear.rs:1508-1560`) tries, in order:
/// `KAPLUMBA_OWN_GEMV` -> `None`; non-integer / out-of-range /
/// `packed_offset % 4 != 0` -> `None`; `perm_mode == 1 || KAPLUMBA_REPACKED_GEMV`
/// -> `_colmajor_contig4` then `_colmajor`; then, unless
/// `KAPLUMBA_ISHIZUKI_3WORD`, a **name lookup** of
/// `linear_gemv_ishizuki_k{bits}_w2`; and finally the per-rate 3-word slot.
///
/// So the fall-through is exactly rates 5..=8, and for the shipped checkpoint
/// (3 bpw on 410 tensors, 6 bpw on `lm_head`) the only model tensor that hits
/// it is the head — which **does** silently use the 3-word form, correctly.
#[test]
fn the_dispatch_takes_w2_for_one_through_four_and_falls_through_for_five_through_eight() {
    let emitted: Vec<u8> = (1..=8u8)
        .filter(|b| LINEAR_METAL.contains(&format!("kernel void linear_gemv_ishizuki_k{b}_w2(")))
        .collect();
    assert_eq!(emitted, vec![1, 2, 3, 4]);

    // The selection is a lookup, and it is gated on an env flag that only an
    // attribution run would set.
    let lookup = line_of(LINEAR_RS, "format!(\"linear_gemv_ishizuki_k{bits}_w2\")")
        .expect("linear.rs must build the _w2 name");
    let gate = line_of(LINEAR_RS, "KAPLUMBA_ISHIZUKI_3WORD").expect("the 3-word override");
    let fallback = line_of(
        LINEAR_RS,
        "self.ishizuki_gemv_pipelines[bits as usize].as_ref()",
    )
    .expect("the 3-word per-rate slot");
    assert!(
        gate < lookup && lookup < fallback,
        "dispatch order moved: gate {gate}, lookup {lookup}, fallback {fallback}"
    );

    // The fall-through is decided by absence from the name map, so the rate
    // parser WOULD accept a name for a rate that has no such kernel. That is
    // the direction the `linear.rs:1526-1532` comment claims the lookup
    // closes and does not: it fails *closed* for a missing kernel, but a
    // wrongly emitted `_k6_w2` would take over the head silently.
    for bits in 5..=8u8 {
        let name = format!("linear_gemv_ishizuki_k{bits}_w2");
        assert_eq!(
            LinearKernels::ishizuki_rate_of_name(&name),
            Some(bits),
            "{name} must still parse as rate {bits}; the fall-through is absence-only"
        );
    }
    // What production actually reaches, for the two rates the model has.
    assert_eq!(
        LinearKernels::ishizuki_rate_of_name("linear_gemv_ishizuki_k3_w2"),
        Some(3)
    );
    assert_eq!(
        LinearKernels::ishizuki_rate_of_name("linear_gemv_ishizuki_k3_nod"),
        Some(3)
    );
    assert_eq!(
        LinearKernels::ishizuki_rate_of_name("linear_gemv_ishizuki_k34_w2"),
        None
    );
    assert!(
        LINEAR_RS.contains("bit-exactness"),
        "the file was replaced wholesale"
    );
}

// ===========================================================================
// 4. The `_nod` probe
// ===========================================================================

/// `_nod` = **no decode**. One asserted substitution replaces every
/// `decode_state(s & 0xffffu, CB)` with `nod_stub(s)`, and nothing else:
/// stripping comments from both bodies must leave code that differs *only* at
/// those sites.
///
/// The decode step it skips is the per-code state lookup — the MCG codebook
/// dequantisation, 24 codes per lane per unrolled iteration — not a memory
/// step. The ring loads, `A0`/`B0`, the unroll and the reduction are untouched,
/// which is the whole point: memory traffic is byte-identical and only the
/// arithmetic differs.
///
/// There is no "proven-safe condition" because there is no correctness
/// property: the output is garbage by construction. That is why
/// `NO_DECODE_PROBE_RATES` is a plain list rather than a derivation
/// (`gen_exl3_gemv.py:155-165`), and why the only invariant asserted is that
/// *every* decode site was replaced (`n >= 3`, `gen:395`).
#[test]
fn no_decode_differs_from_the_real_kernel_only_at_the_decode_sites() {
    let by_name: Vec<(String, String)> = linear_library()
        .into_iter()
        .map(|(_, n, b, _)| (n, b))
        .collect();
    let nod: Vec<(String, String)> = by_name
        .iter()
        .filter(|(n, _)| n.ends_with("_nod"))
        .cloned()
        .collect();
    assert_eq!(nod.len(), 8, "NO_DECODE_PROBE_RATES is tuple(range(1,9))");
    for bits in 1..=8u8 {
        let base = format!("linear_gemv_ishizuki_k{bits}");
        let probe = format!("{base}_nod");
        let original = by_name
            .iter()
            .find(|(n, _)| *n == base)
            .map(|(_, b)| b)
            .unwrap_or_else(|| panic!("no plain kernel for {bits} bpw"));
        let body = &nod
            .iter()
            .find(|(n, _)| *n == probe)
            .map(|(_, b)| b)
            .unwrap_or_else(|| panic!("no probe for {bits} bpw"));

        let stubbed = code_lines(&original.replace("decode_state(s & 0xffffu, CB)", "nod_stub(s)"));
        assert_eq!(
            code_lines(body),
            stubbed,
            "{probe}: the probe differs from {base} somewhere \
            other than the decode call, so its bandwidth is not attributable to decode"
        );
        // And the substitution is real: unstubbed, the bodies do NOT match.
        assert_ne!(
            code_lines(body),
            code_lines(original),
            "{probe} is a verbatim copy of {base}, \
            so it measures the real kernel and not the probe"
        );
        // Every decode site is gone and the traffic is untouched. Checked on the
        // CODE lines only: the probe's own banner says "`decode_state` is
        // replaced by a one-AND stub", so a raw substring search over the body
        // would match the comment that documents the substitution.
        assert!(
            code_lines(body).iter().all(|l| !l.contains("decode_state")),
            "{probe} still calls decode_state in code"
        );
        // Traffic identity, stated as the thing that actually determines it:
        // the same number of `t?c[jN]` ring-word loads at the same indices.
        // Counting `[j` occurrences rather than searching for `w0c`, because a
        // probe that dropped only ONE of its three loads would still contain
        // `w0c` somewhere and pass a substring check.
        assert_eq!(
            ring_loads(body),
            ring_loads(original),
            "{probe} issues a different number of ring-word loads than {base}, so its \
             bandwidth is not that of the real kernel"
        );
        assert!(body.contains("NO-DECODE PROBE. OUTPUT IS GARBAGE BY CONSTRUCTION"));
    }
    assert_eq!(LINEAR_METAL.matches("static float nod_stub").count(), 1);
}

/// Is `_nod` live, or finished dead code?
///
/// * **Not dispatched by the engine.** No dispatch path in `linear.rs` builds or
///   looks up a `_nod` name; the selector's only name is `_w2`.
/// * **Not referenced by any test.** `probe_reduce_epilogue.rs:308` names
///   `linear_gemv_ishizuki_k3_nod` only as a barrier-counting example.
/// * **Reachable only from the experiment harness**,
///   `crates/engine/examples/gemv_shape.rs --decode-probe`, whose question has
///   been answered and recorded: removing decode made every rate *slower*
///   (3 bpw 167.3 -> 122.7 GB/s, **-26.7%**; 4 bpw -5.0%; 8 bpw -1.5%;
///   `spec/decode-bandwidth-240gb/implementation_notes.md:473-490`). The
///   conclusion recorded there is that the 3 bpw gap is request/latency-bound,
///   not decode-bound.
///
/// So it is finished generated code, kept alive only by the harness that
/// produced that answer.
#[test]
fn no_decode_is_never_dispatched_and_only_the_probe_harness_names_it() {
    assert!(
        !LINEAR_RS.contains("_nod"),
        "linear.rs names a _nod kernel, so the engine can dispatch it"
    );
    assert!(
        !LINEAR_RS.contains("nod_stub"),
        "linear.rs mentions nod_stub"
    );
    assert!(GEMV_SHAPE.contains("linear_gemv_ishizuki_k{}_nod") || GEMV_SHAPE.contains("_nod"));
    assert!(GEMV_SHAPE.contains("--decode-probe"));
    assert!(
        GEMV_SHAPE.contains("is never dispatched by the engine"),
        "the harness must still say the probe is never dispatched"
    );
    // The generator, too, says it.
    assert!(GENERATOR
        .contains("**This kernel computes garbage and is never dispatched by the engine.**"));
}

/// It is not free. `LinearKernels::new` builds a pipeline for **every**
/// `linear_gemv_ishizuki_*` in the library (`linear.rs:456-468`), so all eight
/// probes are compiled at load even though none is ever dispatched.
#[test]
fn the_probe_family_costs_eight_pipelines_and_eight_percent_of_the_source() {
    let probes: Vec<(usize, String, String, &'static str)> = linear_library()
        .into_iter()
        .filter(|(_, n, _, _)| n.ends_with("_nod"))
        .collect();
    assert_eq!(probes.len(), 8);

    // Source weight, measured against the string `LinearKernels::new` actually
    // compiles: `combined_source()` is `linear.metal` + `\n` +
    // `linear_repacked.metal` (`linear.rs:24-30`).
    let combined_len = LINEAR_METAL.len() + 1 + LINEAR_REPACKED.len();
    let mut probe_bytes = 0usize;
    let mut w2_bytes = 0usize;
    let mut probe_bytes_metal = 0usize;
    for src in [LINEAR_METAL, LINEAR_REPACKED] {
        for (name, start, end) in kernel_spans(src) {
            let span = end - start;
            if name.ends_with("_nod") {
                probe_bytes += span;
                if src == LINEAR_METAL {
                    probe_bytes_metal += span;
                }
            } else if name.ends_with("_w2") {
                w2_bytes += span;
            }
        }
    }
    let nod_share = probe_bytes as f64 / combined_len as f64 * 100.0;
    let nod_share_metal = probe_bytes_metal as f64 / LINEAR_METAL.len() as f64 * 100.0;
    let w2_share = w2_bytes as f64 / combined_len as f64 * 100.0;
    println!(
        "\n  combined_source() = {combined_len} bytes (linear.metal {} + linear_repacked.metal {})\n  \
           8 _nod kernels    = {probe_bytes} bytes = {nod_share:.1}% of the compiled source \
           ({nod_share_metal:.1}% of linear.metal)\n  \
           4 _w2  kernels    = {w2_bytes} bytes = {w2_share:.1}% — and these ARE dispatched",
        LINEAR_METAL.len(),
        LINEAR_REPACKED.len()
    );
    assert!(
        (5.5..6.5).contains(&nod_share),
        "the probe family's share of combined_source moved to {nod_share:.1}%"
    );
    assert!(
        (7.5..8.5).contains(&nod_share_metal),
        "the probe family's share of linear.metal moved to {nod_share_metal:.1}%"
    );
    assert!(
        (2.5..3.5).contains(&w2_share),
        "the two-word family's share moved to {w2_share:.1}%"
    );

    // Pipeline builds: the recorded audit is 173 builds / 157 distinct kernels
    // / 10.3 ms, all at load (`spec/decode-19tok-repacked-250gb/implementation_notes.md:1381`).
    // Eight of 157 distinct is 5.1%; at the measured 59.5 us mean build, ~0.5 ms.
    assert!(LINEAR_RS.contains("for name in ishizuki_kernel_names(&library)"));
    assert!(LINEAR_RS.contains("pipeline_audit::build(&device, &library, &name)"));
}

// ===========================================================================
// 5. Library census, and whether I-cache pressure is real
// ===========================================================================

/// Kernel census, and the honest answer to "is library size a decode cost?".
///
/// **No.** The number of *resident* kernels at steady state is what an
/// instruction cache sees, and an un-dispatched pipeline is never resident:
/// Metal loads a pipeline's code into GPU instruction memory on first use. A
/// decode step dispatches ~10-20 distinct kernels (`perf_probe_dispatch_census`
/// lands on 750 dispatches over 741 layer + 9 head), and the single hottest
/// kernel — `linear_gemv_ishizuki_k3_w2` — is dispatched 420 times per token
/// (7 matvecs x 60 layers). Re-running one kernel 420 times is the opposite of
/// I-cache-hostile.
///
/// What library size *does* cost is one-off and all of it at load: the recorded
/// audit is 173 pipeline builds / 157 distinct kernels / 10.3 ms, plus whatever
/// `new_library_with_source` charges for the source, both inside model load and
/// neither on the decode path.
///
/// The 546 in circulation is not a dispatch count: it is the number of
/// `GpuProfile::stamp` labels (9 per layer + 6 in the head epilogue), which
/// exist only under `KAPLUMBA_GPU_PROFILE=1`
/// (`perf_probe_dispatch_census.rs:10-18`). The dispatch count is 750.
#[test]
fn library_census_and_the_honest_answer_on_instruction_cache_pressure() {
    let lib = linear_library();
    let ishizuki: Vec<&String> = lib
        .iter()
        .filter(|(_, n, _, _)| n.starts_with("linear_gemv_ishizuki_"))
        .map(|(_, n, _, _)| n)
        .collect();
    let nod = ishizuki.iter().filter(|n| n.ends_with("_nod")).count();
    let w2 = ishizuki.iter().filter(|n| n.ends_with("_w2")).count();
    let variants = ishizuki.len() - 8; // the eight unsuffixed per-rate slots
    println!(
        "\n  linear library          : {} kernels ({} ishizuki: 8 per-rate + {variants} variants)\n  \
         of which _nod            : {nod}  ({:.1}% of the library, {:.1}% of ishizuki)\n  \
         of which _w2             : {w2}\n  ishizuki variants never dispatched by the shipped \
         checkpoint (410 tensors at 3 bpw, aligned) : {}",
        lib.len(),
        ishizuki.len(),
        lib.len() as f64 * 100.0 / lib.len() as f64 * nod as f64 / lib.len() as f64,
        ishizuki.len() as f64 * 100.0 / ishizuki.len() as f64 * nod as f64 / ishizuki.len() as f64,
        variants - 1,
    );
    assert_eq!(lib.len(), 140, "the linear library's kernel count moved");
    assert_eq!(ishizuki.len(), 45);
    assert_eq!((nod, w2), (8, 4));

    // The dispatch census is 750 dispatches, and 546 is the profile-label count.
    let census = include_str!("perf_probe_dispatch_census.rs");
    assert!(census.contains("assert_eq!(census.layer_dispatches, 741"));
    assert!(census.contains("assert_eq!(census.head_dispatches, 9"));
    assert!(census.contains("census.profile_boundaries, 546"));
}

/// The production dispatch set for the shipped checkpoint, as source text.
///
/// 410 tensors are 3 bpw and 4-byte aligned, so they reach
/// `ishizuki_gemv_pipeline`, take the `_w2` name lookup, and land on
/// `linear_gemv_ishizuki_k3_w2` — the only ishizuki kernel production dispatches
/// at all. `lm_head` is 6 bpw with `packed_offset % 4 == 2`, so it returns
/// `None` at the alignment guard and falls to `linear_gemv_exl3_k6_u_mcg`.
#[test]
fn production_dispatches_exactly_one_ishizuki_kernel() {
    let guard = line_of(LINEAR_RS, "packed_offset % 4 != 0").unwrap();
    let w2 = line_of(LINEAR_RS, "format!(\"linear_gemv_ishizuki_k{bits}_w2\")").unwrap();
    assert!(
        guard < w2,
        "the alignment guard must precede the _w2 lookup"
    );
    let exl3 =
        line_of(LINEAR_RS, "linear_gemv_exl3_k{bits}{suffix}{family}").expect("the EXL3 slot loop");
    assert!(
        exl3 > 0,
        "the EXL3 fallback family must still exist for the unaligned head"
    );
    assert!(
        LINEAR_METAL.contains("kernel void linear_gemv_exl3_k6_u_mcg("),
        "the head's actual kernel is missing"
    );
    println!(
        "\n  live ishizuki kernel : linear_gemv_ishizuki_k3_w2  (3 bpw, aligned, perm_mode 0)\n  \
         live head kernel      : linear_gemv_exl3_k6_u_mcg      (packed_offset % 4 == 2)\n  \
         3-word fall-through   : rates 5,6,7,8 — of which only the head's rate is in this model, \
         and it never reaches the fall-through"
    );
}

// ===========================================================================
// 6. Sabotage: every matcher must be able to fail (AGENTS.md §3.4)
// ===========================================================================

/// Mutate copies of the sources and confirm each census below flips. Without
/// this, `two_word_safety_is_derived_from_the_source_and_stops_at_four` could be
/// passing because the comparison is vacuous.
#[test]
fn the_matchers_can_fail() {
    // 1. A `lo0` multiplier that disagrees with `KA`. The CPU model takes
    //    `lo0`'s multiplier from the source, so this is the failure mode the
    //    agreement test exists to catch.
    let kernel3 = |src: &str| -> String {
        let start = src
            .find("kernel void linear_gemv_ishizuki_k3(")
            .expect("plain k3");
        let end = src[start..]
            .find("\n}\n")
            .map(|i| start + i)
            .expect("body end");
        src[start..end].to_string()
    };
    let real = kernel3(LINEAR_METAL);
    assert_eq!(ka_of(&real).map(u32::from), lo0_multiplier_of(&real));
    let sabotaged = real.replace(
        "(lane * 8u + 1u) * 3u + RING - 16u",
        "(lane * 8u + 1u) * 4u + RING - 16u",
    );
    assert_ne!(real, sabotaged, "the sabotage did not apply");
    assert_ne!(
        ka_of(&sabotaged).map(u32::from),
        lo0_multiplier_of(&sabotaged),
        "a KA/lo0 mismatch must be visible to the model, or the model measures a different kernel"
    );

    // 2. A `_w2` kernel that kept the third load is not a two-word kernel.
    let body_of = |src: &str, name: &str| -> String {
        let start = src
            .find(&format!("kernel void {name}("))
            .unwrap_or_else(|| panic!("no kernel {name}"));
        let end = src[start..]
            .find("\n}\n")
            .map(|i| start + i)
            .expect("body end");
        src[start..end].to_string()
    };
    let w2 = body_of(LINEAR_METAL, "linear_gemv_ishizuki_k3_w2");
    assert!(
        !w2.contains("w0c = t0[j2]"),
        "control: the shipped _w2 has no third load"
    );
    let leaky = w2.replacen(
        "const ulong w0a = t0[j0], w0b = t0[j1];",
        "const ulong w0a = t0[j0], w0b = t0[j1], w0c = t0[j2];",
        1,
    );
    assert!(leaky.contains("w0c = t0[j2]"), "the sabotage did not apply");
    assert!(
        leaky.matches("[j").count() * 3
            != body_of(LINEAR_METAL, "linear_gemv_ishizuki_k3")
                .matches("[j")
                .count()
                * 2,
        "a leaky _w2 must break the 2-of-3 load ratio"
    );

    // 3. A `_nod` probe that also dropped a ring load is not traffic-identical.
    let nod = body_of(LINEAR_METAL, "linear_gemv_ishizuki_k3_nod");
    // The exact matcher the real test uses, run on the shipped body (control)
    // and on a body with one ring load deleted (must flip).
    // Three sites: two in the unrolled-by-2 loop (w0a/w0b/w0c and
    // w1a/w1b/w1c) and one in the tail loop (a0/a1/a2). Nine loads, not six.
    assert_eq!(
        ring_loads(&nod).len(),
        9,
        "control: k3 loads 3 ring words at each of 3 sites"
    );
    let traffic_broken = nod.replacen("w0c = t0[j2]", "0u", 1);
    assert_ne!(nod, traffic_broken, "the sabotage did not apply");
    assert_ne!(
        ring_loads(&traffic_broken).len(),
        ring_loads(&nod).len(),
        "the traffic check must be able to fail when a load is removed"
    );

    // 4. A `_nod` probe that changes something OTHER than the decode call must
    //    break the equality, which is the whole "attributable" claim.
    let original = body_of(LINEAR_METAL, "linear_gemv_ishizuki_k3");
    let stubbed = code_lines(&original.replace("decode_state(s & 0xffffu, CB)", "nod_stub(s)"));
    assert_eq!(
        code_lines(&nod),
        stubbed,
        "control: the shipped probe matches its substitution"
    );
    let extra = nod.replacen("acc0 += p00 * v[0]", "acc0 += 2.0f * p00 * v[0]", 1);
    assert_ne!(
        code_lines(&extra),
        stubbed,
        "an extra change must break the comparison"
    );

    // 5. The derived verdict is sensitive, and the extraction comparison is
    //    sensitive to a single flipped input bit.
    assert!(!two_word_verdict(5).0 && two_word_verdict(4).0);
    let w = Window::new(3);
    let three = [0xdead_beefu32, 0x1234_5678, 0x9abc_def0];
    let sh0 = w.indices(0).1;
    let a0 = w.a0_three(three, sh0);
    let b0 = w.b0(three, sh0);
    assert_eq!(
        extract(a0, b0, 3, 0),
        extract(w.a0_two([three[0], three[1]], sh0), 0, 3, 0)
    );
    assert_ne!(
        extract(a0, b0, 3, 0),
        extract(w.a0_two([three[0] ^ 1, three[1]], sh0), 0, 3, 0),
        "flipping one input bit must change the extracted code, or the comparison proves nothing"
    );

    // 6. The dispatch-order claims are about line positions, so they can fail
    //    by the guard moving below the lookup. Rebuild the source with the
    //    guard line moved to just after the lookup line and re-measure.
    let guard_line = line_of(LINEAR_RS, "packed_offset % 4 != 0").unwrap();
    let lookup_line = line_of(LINEAR_RS, "format!(\"linear_gemv_ishizuki_k{bits}_w2\")").unwrap();
    assert!(
        guard_line < lookup_line,
        "control: the guard precedes the lookup in linear.rs"
    );
    let byte_of_line = |n: usize| -> usize {
        LINEAR_RS
            .lines()
            .take(n - 1)
            .map(|l| l.len() + 1)
            .sum::<usize>()
    };
    let guard_text: String = LINEAR_RS
        .lines()
        .nth(guard_line - 1)
        .expect("guard line")
        .to_string();
    let reordered = format!(
        "{}{}\n{}",
        // Everything before the guard, then everything from the lookup line on
        // (which deletes the guard from its original position), then the guard
        // itself at the very end — so `line_of` can only find it after.
        &LINEAR_RS[..byte_of_line(guard_line)],
        &LINEAR_RS[byte_of_line(lookup_line)..],
        guard_text
    );
    assert_ne!(reordered, LINEAR_RS, "the sabotage did not apply");
    assert!(
        line_of(&reordered, "packed_offset % 4 != 0").unwrap()
            > line_of(&reordered, "linear_gemv_ishizuki_k{bits}_w2").unwrap(),
        "moving the guard below the lookup must make the ordering assertion fail"
    );
}
