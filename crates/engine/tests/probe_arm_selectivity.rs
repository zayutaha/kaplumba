//! Probe: is the decode-matvec dispatch order (`ishizuki` first, `EXL3` second)
//! actually *selective*, or is the second arm dead?
//!
//! This is a **static, no-GPU** probe. It parses `linear.metal` and `linear.rs`
//! as text and asserts structural claims about the two kernel families. It
//! allocates no `MTLDevice`, loads no weights, and measures nothing, so it is
//! safe to run while another agent holds the GPU lock -- which is why it is
//! worth having at all: the claims it checks were previously only checkable by
//! running the model.
//!
//! Read-only. Deletes nothing; modifies nothing.
//!
//! Scope, and the three things it is actually for:
//!
//! 1. **Instruction-level delta between the families at the same rate.** Both
//!    are 256-thread / 8-simdgroup / 16-output-tile kernels with the *same*
//!    `simd_shuffle_xor(1,2)` butterfly and the same `red[8][16]` threadgroup
//!    sum. The delta is three things and nothing else: the ring-window width
//!    (2 vs 3 loads), the loop unroll (2 vs 1), and the codebook call site.
//! 2. **The r-variant coverage**, computed from the real checkpoint shapes and
//!    compared against what the dispatch table can actually name. `linear.rs`
//!    asserts "only the ALIGNED MCG family has r-variants"; the question this
//!    answers is whether that restriction *binds* on any shape the model uses.
//! 3. **Whether the EXL3 arm is reachable at all** for an aligned integer-rate
//!    tensor, which is the thing `linear.rs:1490-1502` warns readers to get
//!    wrong when attributing an EXL3 measurement.
//!
//! A note on the `two_word_is_safe` derivation, re-implemented here
//! independently of `scripts/gen_exl3_gemv.py` (which is Python, and AGENTS.md
//! §8 forbids Python in the project -- so a Rust re-derivation is the only way
//! for a Rust test to pin it). It agrees bit-for-bit on all 8 rates; if the two
//! ever disagree, this test fails rather than trusting either.

use std::collections::BTreeSet;

const LINEAR_METAL: &str = include_str!("../../kernels/src/linear.metal");
const LINEAR_RS: &str = include_str!("../../kernels/src/linear.rs");

// ---------------------------------------------------------------------------
// Source extraction
// ---------------------------------------------------------------------------

/// The text of `kernel void <name>(` up to the start of the next top-level
/// `kernel void`, plus its line number in `linear.metal`.
///
/// Slice from the *declaration line* rather than the line after it: the
/// parameter list is part of the text, and a slice starting one line late would
/// silently drop the signature -- which for this file means losing the `params`
/// vs `shape` buffer-3 distinction that the whole arm comparison turns on.
fn kernel_body(name: &str) -> (usize, String) {
    let start = kernel_start(name);
    let line = LINEAR_METAL[..start].lines().count() + 1;
    let rest = &LINEAR_METAL[start..];
    let end = rest.find("\nkernel void ").unwrap_or(rest.len());
    (line, rest[..end].to_string())
}

fn kernel_start(name: &str) -> usize {
    let needle = format!("kernel void {name}(");
    LINEAR_METAL
        .find(&needle)
        .unwrap_or_else(|| panic!("{name} not found in linear.metal"))
}

/// The text of a *non-kernel* function (`static inline void <name>(`).
///
/// Needed because `exl3_decode8_mcg` is where EXL3's ring load actually lives --
/// the generated `linear_gemv_exl3_k*_mcg` kernels just call it -- so counting
/// loads by looking for a `kernel void` of that name finds nothing, which is
/// exactly what the first run of this test reported.
fn function_body(name: &str) -> (usize, String) {
    let needle = format!("void {name}(");
    let start = LINEAR_METAL
        .find(&needle)
        .unwrap_or_else(|| panic!("{name} not found in linear.metal"));
    let line = LINEAR_METAL[..start].lines().count() + 1;
    let rest = &LINEAR_METAL[start..];
    // Up to the next `}` at column 0, which is how a free function ends here.
    let end = rest.find("\n}\n").map(|i| i + 3).unwrap_or(rest.len());
    (line, rest[..end].to_string())
}

fn all_kernel_names() -> Vec<String> {
    let mut names = Vec::new();
    for line in LINEAR_METAL.lines() {
        if let Some(rest) = line.strip_prefix("kernel void ")
            && let Some(paren) = rest.find('(')
        {
            names.push(rest[..paren].to_string());
        }
    }
    names
}

// ---------------------------------------------------------------------------
// 1. Instruction-level delta: `linear_gemv_ishizuki_k3_w2` vs
//    `linear_gemv_exl3_k3_mcg` (and its r-variants)
// ---------------------------------------------------------------------------

/// Ring-word loads per tile in an ishizuki K-loop body, as `(site, words)`.
///
/// Both the unrolled (`w0a`/`w0b`/`w0c`) and tail (`a0`/`a1`/`a2`) forms are
/// matched, because they use different names for the same thing -- the exact
/// asymmetry that made the generator's substitution indices differ. Both the
/// 3-word (`linear_gemv_ishizuki_k3`) and 2-word (`_k3_w2`) spellings are
/// matched for the same reason: the whole point is the *width*, so a matcher
/// that only knew one of the two spellings would report 0 for the other.
fn ishizuki_ring_load_sites(body: &str) -> Vec<usize> {
    let three = [
        (", w0b = t0[j1], w0c = t0[j2]", 3usize),
        (", w1b = t1[j1], w1c = t1[j2]", 3),
        (", a1 = t0[j1], a2 = t0[j2]", 3),
    ];
    let two = [
        (", w0b = t0[j1];", 2usize),
        (", w1b = t1[j1];", 2),
        (", a1 = t0[j1];", 2),
    ];
    three
        .iter()
        .chain(two.iter())
        .filter(|(pat, _)| body.contains(pat))
        .map(|(_, w)| *w)
        .collect()
}

#[test]
fn ishizuki_w2_has_two_ring_loads_and_exl3_has_three() {
    // The live kernel for all 410 aligned 3 bpw decoder linears.
    let (ish_line, ish) = kernel_body("linear_gemv_ishizuki_k3_w2");
    // The EXL3 arm's aligned, MCG, NR0=1 kernel.
    let (_, exl) = kernel_body("linear_gemv_exl3_k3_mcg");

    // `_w2`: 2 loads per tile, at three sites (two in the unrolled body, one in
    // the tail). Uniform width is the invariant -- a site of any other width
    // would mean the generator's substitutions did not all match.
    let sites = ishizuki_ring_load_sites(&ish);
    assert_eq!(
        sites,
        vec![2, 2, 2],
        "expected every ring-load site in _w2 to read two words; got {sites:?} \
         (linear.metal:{ish_line})"
    );

    // `_w2` must carry no third-word term anywhere.
    assert!(
        !ish.contains("w0c") && !ish.contains("a2]"),
        "_w2 body still references the third ring word (linear.metal:{ish_line})"
    );

    // EXL3's ring load lives in `exl3_decode8_mcg`, a `static inline` helper the
    // generated kernels call -- NOT in a `kernel void` of that name.
    let (dec_line, dec) = function_body("exl3_decode8_mcg");
    assert_eq!(
        dec.matches("ring_word<ALIGNED>(packed, base + j").count(),
        3,
        "exl3_decode8_mcg should read exactly three ring words (linear.metal:{dec_line})"
    );
    // The third word `w2` is read, and it is folded into BOTH `a` and `c` -- the
// `c` fold is the `B0` arm, live at 6/7/8 bpw and dead at 3 bpw, but EXL3 has
    // no rate-specialised build so it always pays for it.
    assert!(
        dec.contains("const ulong a = ((w0 << 32) | w1) << sh | (sh ? (w2 >> (32 - sh)) : 0ul);"),
        "exl3_decode8_mcg lost the third word's contribution to A0 (linear.metal:{dec_line})"
    );
    assert!(
        dec.contains("const ulong c = ((w1 << 32) | w2) << sh;"),
        "exl3_decode8_mcg lost the B0 arm (linear.metal:{dec_line})"
    );

    // The reduction must be IDENTICAL. If either family changes its butterfly or
    // its threadgroup sum, the two arms stop being comparable and every
    // "family A beat family B" number in the notes becomes a comparison of two
    // different reductions.
    // The two families spell the butterfly differently -- ishizuki uses the
    // scalar `acc0`/`acc1`, EXL3 the NR0-indexed `acc[q][0]`/`acc[q][1]` -- so
    // count the *passes* by their shuffle width argument, not by variable name.
    // A matcher keyed on `acc0` finds 0 in the EXL3 body and reads as "the
    // reduction is gone", which is what two earlier runs of this test reported.
    let shuffle_widths = |body: &str| -> Vec<u32> {
        body.lines()
            .filter(|l| l.contains("simd_shuffle_xor"))
            .filter_map(|l| {
                // `acc0 += simd_shuffle_xor(acc0, 1u);` -- the text after the
                // last comma is " 1u);", so trim before reading the digits or
                // the take_while stops on the leading space and finds none.
                let tail = l.rsplit_once(',')?.1.trim_start();
                let n: String = tail.chars().take_while(char::is_ascii_digit).collect();
                n.parse::<u32>().ok()
            })
            .collect()
    };

    for (label, body) in [("ishizuki", &ish), ("exl3", &exl)] {
        let widths = shuffle_widths(body);
        // Two accumulators, each reduced over four lanes by a width-1 then a
        // width-2 xor -- so the sequence is [1, 2, 1, 2], not [1, 1, 2, 2]. The
        // families agree on this, which is the point: it is why their outputs
        // are comparable at all.
        assert_eq!(
            widths,
            vec![1, 2, 1, 2],
            "{label}'s butterfly passes changed: {widths:?} (expected a width-1 \\
             then width-2 pass for each of the two accumulators)"
        );
    }
    // And the two families are byte-identical in their reduction *shape*.
    assert_eq!(
        shuffle_widths(&ish),
        shuffle_widths(&exl),
        "the two families no longer perform the same four-lane reduction"
    );
    assert!(
        ish.contains("threadgroup float red[SG][16]"),
        "ishizuki's threadgroup reduction array changed shape"
    );
    assert!(
        exl.contains("threadgroup float red[8][16 * NR0]"),
        "exl3's threadgroup reduction array changed shape"
    );
}

#[test]
fn only_ishizuki_unrolls_the_k_loop_by_two() {
    let (_, ish) = kernel_body("linear_gemv_ishizuki_k3_w2");
    let (_, exl) = kernel_body("linear_gemv_exl3_k3_mcg");

    assert!(
        ish.contains("tk += 2u * SG"),
        "ishizuki_k3_w2 lost its 2x K-loop unroll (linear.metal:1)"
    );
    assert!(
        exl.contains("tk += simdgroups"),
        "exl3_k3_mcg gained or lost its K stride (linear.metal:1)"
    );
    // Words in flight per lane, computed from the unroll factor and the window
    // width rather than asserted as constants. `_w2`: 2 tiles x 2 words = 4.
    // EXL3: 1 tile x 3 words = 3.
    let unroll_tiles = if ish.contains("tk += 2u * SG") { 2 } else { 1 };
    let window_words = ishizuki_ring_load_sites(&ish)
        .first()
        .copied()
        .expect("no ring-load sites in _w2");
    let in_flight = unroll_tiles * window_words;
    assert_eq!(
        in_flight, 4,
        "_w2 should keep four weight words in flight (2 tiles x 2 words); \
         unroll_tiles={unroll_tiles}, window_words={window_words}"
    );
    // EXL3 reads one tile of three words per iteration; its window lives in
    // `exl3_decode8_mcg`, so it is NOT visible in the kernel body and is
    // asserted in `ishizuki_w2_has_two_ring_loads_and_exl3_has_three` instead.
    // Do not re-derive it here: `ishizuki_ring_load_sites` on the EXL3 body
    // returns empty, and `empty.first().unwrap_or(0).max(3) == 3` would assert
    // `3 == 3` -- true no matter what the kernel does.
    assert!(
        in_flight > 3,
        "the unroll should beat EXL3's single-tile window ({in_flight} vs 3)"
    );

    // The phrase spans a line break in the source ("keeps six weight words" /
    // "in flight per lane"), so the comment is comment-markers-stripped and
    // whitespace-collapsed before matching.
    //
    // Both halves are necessary, and both were discovered by the assertion
    // failing on text that is present:
    //   * collapsing whitespace alone yields "six weight words // in flight",
    //     because every line keeps its leading `//`, so the marker must go
    //     first;
    //   * matching the raw text yields nothing at all, because the phrase is
    //     split across the newline.
    let squash = |s: &str| {
        s.lines()
            .map(|l| l.trim().trim_start_matches("//"))
            .collect::<Vec<_>>()
            .join(" ")
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    };
    let (_, ish3) = kernel_body("linear_gemv_ishizuki_k3");
    assert!(
        squash(&ish3).contains("six weight words in flight"),
        "the 3-word kernel's in-flight comment changed"
    );
    assert!(
        squash(&ish).contains("six weight words in flight"),
        "expected `_w2` to still carry the 3-word kernel's 'six weight words in \\
         flight' comment. It is STALE -- `_w2` reads four words in flight, not \\
         six. If this now reads correctly, the comment was fixed and this \\
         assertion should be replaced with the accurate figure."
    );
}

#[test]
fn ishizuki_folds_the_codebook_to_a_constant_and_exl3_calls_the_mcg_form() {
    let (_, ish) = kernel_body("linear_gemv_ishizuki_k3_w2");
    assert!(
        ish.contains("constexpr int CB = 1;"),
        "ishizuki lost `constexpr int CB = 1` -- this is the compile-time-codebook \
         property that commit 3439d41 claimed as its 8.1->9.2 tok/s step, and it \
         was ALREADY TRUE in the live kernel before that commit"
    );
    // CB == 1 is MCG (`Exl3Codebook::encoded`), so the generic `decode_state`
    // folds to `decode_state_mcg` at compile time. If the encoding ever changes,
    // the ishizuki arm silently becomes the WRONG codebook.
    let exl_rs = include_str!("../../kernels/src/exl3.rs");
    let mcg_is_one = exl_rs
        .lines()
        .zip(exl_rs.lines().skip(1))
        .any(|(a, b)| a.contains("Self::Mcg =>") && b.contains('1'));
    assert!(
        mcg_is_one,
        "Exl3Codebook::Mcg is no longer encoded as 1, so ishizuki's `CB = 1` \
         no longer means MCG"
    );

    // The EXL3 arm gets the same effect via a *different* mechanism: a separate
    // function per codebook, called from a `CODEBOOK` template. That is the
    // property 3439d41 actually added, and it is what the 163.0 GB/s ablation
    // line measured -- on the EXL3 arm, which was not the live one.
    let (_, dec) = function_body("exl3_decode8_mcg");
    assert!(
        dec.contains("decode_state_mcg(s & 0xffffu)"),
        "exl3_decode8_mcg no longer calls the MCG-specialised decoder directly"
    );
    let (_, ish_k3) = kernel_body("linear_gemv_ishizuki_k3");
    assert!(
        ish_k3.contains("constexpr int CB = 1;"),
        "the 3-word ishizuki k3 also needs CB = 1 for the attribution argument"
    );
}

// ---------------------------------------------------------------------------
// 2. The `two_word_is_safe` derivation, re-derived in Rust
// ---------------------------------------------------------------------------

/// Independent re-implementation of `two_word_is_safe` in
/// `scripts/gen_exl3_gemv.py`. Returns `(safe, why)`.
///
/// `A0 = ((w0a << 32) | w0b) << sh0 | (sh0 ? (w0c >> (32 - sh0)) : 0)`, and code
/// `i` reads `A0 >> (64 - (i*KA + 16))` masked to 16 bits. So the lowest A0 bit
/// any code can touch is `64 - (7*KA + 16) = 48 - 7*KA`, and `w0c` only ever
/// contributes to A0 bits `[0, sh)`. Dead iff no lane needs a bit below its own
/// `sh`, and no code takes the `B0` arm (which is built from w0b/w0c).
fn two_word_is_safe(bits: u32) -> (bool, String) {
    let ring = 256 * bits;
    let r_max = 7 * bits + 16;
    let used_lo = 48i64 - 7 * bits as i64;
    let sh_max = (0..32u32)
        .map(|lane| ((lane * 8 + 1) * bits + ring - 16) % 32)
        .max()
        .expect("32 lanes");
    if r_max > 64 {
        return (false, format!("r_max={r_max} > 64, so the B0 path reads w0c"));
    }
    if used_lo < sh_max as i64 {
        return (
            false,
            format!("lowest used A0 bit {used_lo} < max sh {sh_max}, so some lane reads w0c"),
        );
    }
    (
        true,
        format!("used [{used_lo},63] >= max sh {sh_max}, and r_max={r_max} <= 64"),
    )
}

#[test]
fn two_word_safe_rates_are_exactly_1_to_4_and_the_kernels_agree() {
    let derived: Vec<u32> = (1..=8).filter(|b| two_word_is_safe(*b).0).collect();
    assert_eq!(
        derived,
        vec![1, 2, 3, 4],
        "the derivation moved; _w2 emission and its safety argument must move with it"
    );

    // The emitted kernels must match the derivation exactly -- no `_w2` above 4,
    // none missing at or below 4.
    let emitted: BTreeSet<u32> = all_kernel_names()
        .iter()
        .filter_map(|n| n.strip_prefix("linear_gemv_ishizuki_k")?.strip_suffix("_w2"))
        .filter_map(|s| s.parse::<u32>().ok())
        .collect();
    let want: BTreeSet<u32> = derived.iter().copied().collect();
    assert_eq!(
        emitted, want,
        "the `_w2` kernels in linear.metal do not match the derived safe set"
    );

    // **Zero margin at 3 bpw.** `used_lo == sh_max == 27`: the derivation holds
    // by equality, not by slack. Any change to the ring geometry -- a different
    // `+16`, a different `+1`, a different lane->bit map -- breaks 3 bpw
    // silently, and 3 bpw is 410 of 411 tensors. This is the single most
    // fragile invariant in the live decode kernel.
    let (_, why3) = two_word_is_safe(3);
    let (_, why4) = two_word_is_safe(4);
    assert!(why3.contains("used [27,63] >= max sh 27"), "3 bpw margin changed: {why3}");
    assert!(why4.contains("used [20,63] >= max sh 20"), "4 bpw margin changed: {why4}");

    // 6 bpw is `lm_head`, and it is NOT safe -- which is why `lm_head` runs the
    // EXL3 `_u_mcg` arm and why that is not a loss.
    let (ok6, why6) = two_word_is_safe(6);
    assert!(!ok6, "6 bpw must not be 2-word safe: {why6}");
}

// ---------------------------------------------------------------------------
// 3. r-variant coverage against the real checkpoint shapes
// ---------------------------------------------------------------------------

/// The shapes, transcribed from
/// `~/.kaplumba/models/Gxma/Gxma/quantization_config.json` (`trellis.shape` =
/// `[tiles_k, tiles_n, ushort_words]`, so `K = tiles_k*16`, `N = tiles_n*16`,
/// `KA = words/16`). Transcribed rather than parsed because the checkpoint lives
/// outside the repository and a test may not depend on `~/.kaplumba`.
///
/// Regenerate with:
///   python3 -c 'import json,collections; ...'   (see implementation notes)
///
/// counts sum to 411 quantized tensors: 410 at 3 bpw, 1 at 6 bpw.
const MODEL_SHAPES: &[(u32, u32, u32, u32)] = &[
    // bits, K, N, count
    (3, 5376, 2048, 10),   // k_proj, full-attention layers (every 6th)
    (3, 5376, 4096, 100),  // k_proj / v_proj, sliding layers
    (3, 8192, 5376, 50),   // o_proj, sliding layers
    (3, 16384, 5376, 10),  // o_proj, full-attention layers
    (3, 21504, 5376, 60),  // down_proj
    (3, 5376, 8192, 50),   // q_proj, sliding layers
    (3, 5376, 16384, 10),  // q_proj, full-attention layers
    (3, 5376, 21504, 120), // gate_proj + up_proj
    (6, 5376, 262144, 1),  // lm_head
];

#[test]
fn the_transcribed_model_shapes_are_self_consistent() {
    let total: usize = MODEL_SHAPES.iter().map(|s| s.3 as usize).sum();
    assert_eq!(total, 411, "quantized tensor count moved; re-transcribe");
    let at3: usize = MODEL_SHAPES.iter().filter(|s| s.0 == 3).map(|s| s.3 as usize).sum();
    let at6: usize = MODEL_SHAPES.iter().filter(|s| s.0 == 6).map(|s| s.3 as usize).sum();
    assert_eq!((at3, at6), (410, 1), "the 410/1 rate split moved");
    // Every shape must satisfy the kernels' own divisibility requirement.
    for &(bits, k, n, _) in MODEL_SHAPES {
        assert!(k % 16 == 0 && n % 16 == 0, "shape {k}x{n} is not tile-aligned");
        if bits == 3 {
            assert!(
                two_word_is_safe(3).0,
                "3 bpw shapes rely on the zero-margin 2-word window"
            );
        }
    }
}

#[test]
fn every_model_shape_admits_every_r_variant() {
    // `linear.rs:1804` says only the ALIGNED MCG family has r-variants. The
    // question this answers is whether that restriction *binds*: if every shape
    // in the model satisfies N/16 % NR0 == 0, then the restriction excludes
    // nothing and the r-variants are a dead axis for a second reason (the EXL3
    // arm is never reached), not a third.
    let mut never = Vec::new();
    for &(bits, k, n, count) in MODEL_SHAPES {
        let tiles = n / 16;
        for nr0 in [2u32, 4, 8] {
            if tiles % nr0 != 0 {
                never.push((bits, k, n, count, nr0));
            }
        }
    }
    assert!(
        never.is_empty(),
        "some model shape would reject an r-variant under the \
         `out_features/16 % NR0 == 0` guard in encode_gemv: {never:?}"
    );
}

#[test]
fn nr0_shapes_that_actually_exist_in_the_metal_file() {
    // All 8 rates x {2,4,8} = 24 r-variants, aligned + MCG only. No `_u`.
    let names = all_kernel_names();
    let mut found = BTreeSet::new();
    for n in &names {
        for nr0 in [2u32, 4, 8] {
            if n.starts_with("linear_gemv_exl3_k")
                && n.ends_with(&format!("_mcg_r{nr0}"))
                && !n.contains("_u")
            {
                found.insert(n.clone());
            }
        }
    }
    assert_eq!(
        found.len(),
        24,
        "expected 8 rates x 3 NR0 values of aligned MCG r-variants, found {}: {found:?}",
        found.len()
    );
    // And the unsuffixed kernels are NR0=1.
    for bits in 1..=8u32 {
        let (line, body) = kernel_body(&format!("linear_gemv_exl3_k{bits}_mcg"));
        assert!(
            body.contains("const uint NR0 = 1;"),
            "linear_gemv_exl3_k{bits}_mcg is not NR0=1 (linear.metal:{line})"
        );
    }
}

// ---------------------------------------------------------------------------
// 4. Reachability: does the dispatch order ever *choose* the EXL3 arm?
// ---------------------------------------------------------------------------

#[test]
fn ishizuki_is_unconditional_for_aligned_integer_rates() {
    // `ishizuki_gemv_pipeline` returns None only for: OWN_GEMV set, a
    // non-Integer rate, bits outside 1..=8, or a non-4-byte-aligned offset.
    // Everything else reaches the ishizuki arm, so the EXL3 arm is reachable
    // for a `(Integer(1..=8), offset%4==0)` tensor ONLY via KAPLUMBA_OWN_GEMV.
    assert!(
        LINEAR_RS.contains("if env_flag(\"KAPLUMBA_OWN_GEMV\")"),
        "KAPLUMBA_OWN_GEMV no longer gates the ishizuki arm; the A/B switch moved"
    );
    assert!(
        LINEAR_RS.contains("if bits < 1 || bits > 8 || packed_offset % 4 != 0"),
        "the ishizuki eligibility guard changed; re-derive which tensors fall \
         through to the EXL3 arm"
    );
    // The order itself: ishizuki first.
    //
    // The arms are located inside `encode_gemv_raw`, NOT `encode_gemv`:
    // `encode_gemv` (linear.rs:1960) handles only the sk2/sk4 colmajor
    // fast paths and then delegates to `encode_gemv_raw` (linear.rs:2062) for
    // everything else, and it is `encode_gemv_raw` that holds the
    // ishizuki-then-EXL3 `if/else if` chain (linear.rs:2159 / 2182).
    //
    // A *global* `find` on the `decode_gemv_pipeline` string is worse than
    // useless here: it hits `gemv_buffer` at linear.rs:864, ~1300 lines
    // earlier, and comparing that against the ishizuki site reports the order
    // flipped when it is not. That is the same class of error as the one
    // `linear.rs:1580-1584` documents -- a lookup that answers a nearby
    // question instead of the intended one.
    let enc = LINEAR_RS
        .find("pub fn encode_gemv_raw(")
        .expect("encode_gemv_raw");
    let enc_end = LINEAR_RS[enc..]
        .find("\n    pub fn ")
        .map(|i| enc + i)
        .expect("end of encode_gemv_raw");
    let enc_body = &LINEAR_RS[enc..enc_end];

    let ish_at = enc_body
        .find("if let Some(pipeline) = self.ishizuki_gemv_pipeline(rate, packed_offset, perm_mode)")
        .expect("ishizuki arm in encode_gemv_raw");
    let exl_at = enc_body
        .find("else if let Some(pipeline) = self.decode_gemv_pipeline(rate, packed_offset, codebook)")
        .expect("exl3 arm in encode_gemv_raw");
    assert!(
        ish_at < exl_at,
        "the dispatch order flipped inside encode_gemv_raw: EXL3 would now run first"
    );
    // And the EXL3 arm really is an `else if` on the ishizuki result, not a
    // sibling reached independently.
    assert!(
        enc_body.contains("} else if let Some(pipeline) = self.decode_gemv_pipeline("),
        "the EXL3 arm is no longer gated on ishizuki declining"
    );
    // And `encode_gemv` really does delegate here, so this is the decode path.
    let pub_at = LINEAR_RS.find("pub fn encode_gemv(").expect("encode_gemv");
    let pub_end = LINEAR_RS[pub_at..]
        .find("\n    pub fn ")
        .map(|i| pub_at + i)
        .expect("end of encode_gemv");
    assert!(
        LINEAR_RS[pub_at..pub_end].contains("self.encode_gemv_raw("),
        "encode_gemv no longer delegates to encode_gemv_raw; the dispatch order \
         checked above is no longer on the decode path"
    );
}

#[test]
fn nr0_knob_is_unreachable_without_own_gemv() {
    // `decode_gemv_kernel_name` builds `{stem}_mcg_r{nr0}` from the EXL3 table,
    // and `decode_gemv_pipeline` -- the only consumer -- sits in the `else if`
    // behind `ishizuki_gemv_pipeline`. So for all 410 aligned 3 bpw linears,
    // KAPLUMBA_DECODE_NR0 has NO effect unless KAPLUMBA_OWN_GEMV is also set.
    //
    // This is worth pinning because the knob looks live: it is documented, it
    // has 24 generated kernels, and `decode_gemv_kernel_name` will happily
    // *name* an r-variant for a (3 bpw, aligned) tensor. Naming it is not
    // reaching it.
    assert!(
        LINEAR_RS.contains("let base = self.exl3_gemv_kernel_name(rate, packed_offset, codebook)?;"),
        "decode_gemv_kernel_name no longer resolves through the EXL3 table"
    );
    assert!(
        LINEAR_RS.contains("if !base.ends_with(\"_mcg\") || base.contains(\"_u\")"),
        "the aligned-MCG-only r-variant restriction changed"
    );
    // Both switches must be consulted by the same caller for the sweep to mean
    // anything; assert the strings exist and are independent.
    for flag in [
        "KAPLUMBA_DECODE_NR0",
        "KAPLUMBA_OWN_GEMV",
        "KAPLUMBA_ISHIZUKI_3WORD",
        "KAPLUMBA_GEMV_MAX_GROUPS",
        "KAPLUMBA_REPACKED_GEMV",
        "KAPLUMBA_LEGACY_GEMV",
    ] {
        assert!(
            LINEAR_RS.contains(flag),
            "environment switch {flag} disappeared from linear.rs"
        );
    }
}

#[test]
fn gemv_max_groups_defaults_to_uncapped_despite_the_comment() {
    // `linear.rs:1478-1484` documents "One threadgroup per 16 outputs, capped ...
    // At N=21504 that is 1344 threadgroups on 20 cores -- about 1.4 waves".
    // But `gemv_max_groups()` returns `u64::MAX` when
    // KAPLUMBA_GEMV_MAX_GROUPS is unset, so the cap is *inert by default* and
    // the grid is always the full `out_features/16`. The documented tail
    // flattening therefore does not happen in the default configuration.
    let fn_at = LINEAR_RS.find("fn gemv_max_groups").expect("gemv_max_groups");
    let body: String = LINEAR_RS[fn_at..].chars().take(600).collect();
    assert!(
        body.contains(".unwrap_or(u64::MAX)"),
        "gemv_max_groups no longer defaults to uncapped; the linear.rs:1478 \
         comment's 'capped' grid needs re-checking against the code"
    );
    // And the ishizuki dispatch still clamps by it, so setting the env var is
    // the only way to get the documented behaviour.
    assert!(
        LINEAR_RS.contains("let groups = tiles.min(gemv_max_groups()).max(1);"),
        "encode_ishizuki_gemv no longer clamps its grid by gemv_max_groups"
    );
}

#[test]
fn the_nod_probes_are_unreachable() {
    // `linear_gemv_ishizuki_k{1..8}_nod` exist (8 kernels, every rate) and are
    // named nowhere in any Rust source: `KAPLUMBA_*` never selects them. They
    // are A/B-only, and `KAPLUMBA_REPACKED_GEMV` -- the only flag that reaches
    // the ishizuki by-name map with a non-`_w2` suffix -- selects
    // `_colmajor_contig4` / `_colmajor`, never `_nod`.
    //
    // Worth stating because they look like a decode-no-decode control that could
    // be run today, and they cannot be, at any cost.
    let nod: Vec<String> = all_kernel_names()
        .into_iter()
        .filter(|n| n.ends_with("_nod"))
        .collect();
    assert_eq!(nod.len(), 8, "expected one _nod probe per rate, got {nod:?}");
    for n in &nod {
        // Assert on the FULL name only. Trimming the `_nod` suffix yields
        // `linear_gemv_ishizuki_k3`, which is a real and reachable kernel (the
        // `KAPLUMBA_ISHIZUKI_3WORD` fallback), so a stem check fails for a
        // reason that has nothing to do with the probe -- the first run of this
        // test reported exactly that false positive.
        assert!(
            !LINEAR_RS.contains(n.as_str()),
            "{n} is now referenced by linear.rs; update the unreachable claim"
        );
    }
    // The only ishizuki names linear.rs can *build* are the unsuffixed `kN`
    // fallback and a fixed set of colmajor/sk-suffixed spellings -- none of them
    // a `_nod`. Pin the accepted set so that a future probe is added *reachable*
    // on purpose rather than by accident. Derived from linear.rs:1542-1556,
    // 2003-2035, 2110-2135, 2247-2301, 2370.
    const REACHABLE_ISHIZUKI_SUFFIXES: &[&str] = &[
        "",
        "_w2",
        "_colmajor",
        "_colmajor_2col",
        "_colmajor_4col",
        "_colmajor_contig4",
        "_colmajor_contig4_sk2",
        "_colmajor_contig4_sk4",
        "_colmajor_1col_sk2",
        "_colmajor_2col_sk2",
        "_colmajor_2col_sk4",
        "_fused2_sk2",
    ];
    for line in LINEAR_RS.lines() {
        // Match the `{bits}`-templated spellings: `linear_gemv_ishizuki_k{...}`
        // followed by a literal suffix.
        let mut rest = line;
        while let Some(at) = rest.find("linear_gemv_ishizuki_k{") {
            let after = &rest[at + "linear_gemv_ishizuki_k{".len()..];
            let close = after.find('}').expect("unterminated format placeholder");
            let suffix: String = after[close + 1..]
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            assert!(
                REACHABLE_ISHIZUKI_SUFFIXES.contains(&suffix.as_str()),
                "linear.rs can now build ishizuki variant suffix `{suffix}`, which \\
                 this probe has not classified; if it is `_nod` the probe is reachable"
            );
            rest = &after[close + 1..];
        }
    }
}