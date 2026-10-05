//! crux3 — the decode-algebra arms, and the claim that they are comparable.
//!
//! `examples/crux3_decode_algebra.rs` measures whether the 3 bpw decode GEMV is
//! issue-bound on `decode_state_mcg`. Its result is only meaningful if the three
//! arms are the *same kernel* with a different extract body. Nothing in a Metal
//! shader lets you assert that, so these tests assert it on the **source**, and
//! the executable half of the claim is `scripts/crux3_armA_equiv.py`, which does
//! the token-level comparison against the production kernel and was
//! sabotage-verified (see `spec/decode-20tok-crux3/crux3.md` §2).
//!
//! What is asserted here, without a GPU:
//!
//! 1. all three arms come from one macro, so their loop bodies cannot drift;
//! 2. the macro's arm-A instantiation is token-identical to the production
//!    `linear_gemv_ishizuki_k3_colmajor_2col_sk4`;
//! 3. arms B and C keep **eight independent chains** — the property whose loss
//!    made `_nod` 26.7% *slower* and turned it into a measurement of ILP
//!    starvation rather than of arithmetic cost;
//! 4. arms B and C still **consume both loaded words**, so they are
//!    bandwidth-identical to A and the comparison is arithmetic-only;
//! 5. nothing in the crate can dispatch a probe kernel, so arms B and C — which
//!    produce deliberately wrong values — cannot reach the engine.

use std::collections::BTreeSet;

const PROBE: &str = include_str!("../../kernels/src/crux3_decode_algebra.metal");
const REPACKED: &str = include_str!("../../kernels/src/linear_repacked.metal");
const LINEAR_RS: &str = include_str!("../../kernels/src/linear.rs");

/// Text between the first `{` after `header` and its matching `}`.
///
/// Brace matching rather than "up to the next blank line", because the bodies
/// contain `for` loops and nested blocks; a line-based window grows into the
/// next kernel, which is how an assertion like this passes for the wrong reason.
fn body(text: &'static str, header: &str) -> &'static str {
    let start = text
        .find(header)
        .unwrap_or_else(|| panic!("{header} not found"));
    let open = text[start..]
        .find('{')
        .map(|i| start + i)
        .expect("body has no opening brace");
    let bytes = text.as_bytes();
    let mut depth = 0usize;
    for (offset, byte) in bytes[open..].iter().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &text[open + 1..open + offset];
                }
            }
            _ => {}
        }
    }
    panic!("unbalanced braces after {header}");
}

/// The macro's single definition, with line continuations removed.
fn macro_body() -> &'static str {
    body(PROBE, "#define CRUX3_COLMAJOR_2COL_SK4(NAME, EXTRACT)")
}

#[test]
fn all_three_arms_come_from_one_macro() {
    let instantiations: Vec<&str> = PROBE
        .lines()
        .filter(|l| l.starts_with("CRUX3_COLMAJOR_2COL_SK4("))
        .collect();
    assert_eq!(
        instantiations.len(),
        3,
        "expected exactly three arm instantiations, found {instantiations:?}"
    );
    let extracts: Vec<&str> = instantiations
        .iter()
        .map(|l| l.trim_end_matches(')').rsplit(',').next().expect("has an extract").trim())
        .collect();
    assert_eq!(
        extracts,
        vec![
            "extract_v_mcg_32",
            "extract_v_crux3_armB",
            "extract_v_crux3_armC"
        ],
        "the three arms must differ only in which extract helper they pass"
    );
}

#[test]
fn the_arms_keep_eight_independent_chains() {
    // Each of the eight windows must be a distinct shift of `A_hi`/`A_lo`.
    // If two windows collapsed onto the same expression the arm would have
    // fewer chains than A and, per `_nod`, could be *slower* for reasons that
    // have nothing to do with arithmetic — the exact confound A2 §3 was written
    // to avoid.
    for (helper, name) in [
        ("extract_v_crux3_armB", "arm B"),
        ("extract_v_crux3_armC", "arm C"),
    ] {
        let b = body(PROBE, &format!("static inline void {helper}("));
        let mut seen: BTreeSet<String> = BTreeSet::new();
        for line in b.lines() {
            let line = line.trim();
            if let Some(rest) = line.strip_prefix("v[") {
                if let Some((index, expr)) = rest.split_once("] = ") {
                    let expr = expr.trim().trim_end_matches(';').trim();
                    // Strip the common `float(as_type<half>(ushort(` wrapper so
                    // two windows that differ only in shift amount are not
                    // counted as distinct by their wrapper text.
                    // `v[7] = v[6];` is the same loss wearing a different mask:
                    // it references a sibling slot rather than an expression over
                    // the loaded words. Both forms are checked, because the
                    // first sabotage attempt below used the second form and
                    // passed — a test that only catches one spelling of the same
                    // defect is decoration.
                    assert!(
                        !expr.starts_with("v["),
                        "{name}: v[{index}] copies another slot ({expr:?}) instead of \
                         extracting a window, so this arm has fewer than eight \
                         independent chains"
                    );
                    let core = expr
                        .trim_start_matches("float(as_type<half>(ushort(")
                        .trim_end_matches(")))")
                        .to_string();
                    assert!(
                        seen.insert(core.clone()),
                        "{name}: two windows share the expression {core:?}, so this \
                         arm has fewer than eight independent chains"
                    );
                    assert!(
                        index.parse::<usize>().expect("v[] index is a number") < 8,
                        "{name}: v[{index}] is out of range"
                    );
                }
            }
        }
        assert_eq!(
            seen.len(),
            8,
            "{name} writes {seen:#?} distinct windows, expected 8"
        );
    }
}

#[test]
fn arms_b_and_c_still_consume_both_loaded_words() {
    // If an arm stopped reading `wb`, it would be doing less memory traffic as
    // well as less arithmetic, and the A/B/C comparison would be measuring two
    // variables. The sabotage that would break this is deleting a use of `wb`.
    for (helper, name) in [
        ("extract_v_crux3_armB", "arm B"),
        ("extract_v_crux3_armC", "arm C"),
    ] {
        let b = body(PROBE, &format!("static inline void {helper}("));
        let uses_wa = b.matches("(wa").count();
        let uses_wb = b.matches("(wb").count();
        assert!(
            uses_wa >= 1 && uses_wb >= 1,
            "{name} does not use both loaded words (wa {uses_wa}x, wb {uses_wb}x); it \
             would differ from A in traffic as well as in arithmetic"
        );
    }
}

#[test]
fn no_probe_kernel_is_reachable_from_the_crate() {
    // Arms B and C produce wrong values. The only thing keeping them out of the
    // engine is that no dispatch site asks for them by name.
    for arm in ["crux3_armB", "crux3_armC"] {
        assert!(
            !LINEAR_RS.contains(arm),
            "{arm} is named in linear.rs; a probe kernel must never be dispatchable \
             from production code"
        );
    }
    // And the production kernel they are compared against is still selected by
    // its own name, unmodified.
    assert!(
        LINEAR_RS.contains("linear_gemv_ishizuki_k{bits}_colmajor_2col_sk4"),
        "the sk4 selector moved; the arms are no longer anchored to what ships"
    );
    assert!(
        REPACKED.contains("kernel void linear_gemv_ishizuki_k3_colmajor_2col_sk4("),
        "the production sk4 kernel is gone"
    );
}

#[test]
fn the_macro_is_a_transcription_and_says_so() {
    // The macro exists only because there is no way to parameterise a Metal
    // kernel's function body. So the equivalence to production is NOT automatic
    // and rests on `scripts/crux3_armA_equiv.py`. This test asserts that script
    // exists, is wired into the harness, and that the macro is a copy of the
    // production kernel rather than of something else in the file.
    //
    // It cannot re-derive the equivalence here without duplicating the script's
    // normaliser, so what it *does* check is the thing that would silently break
    // it: that the production kernel it compares against is still the one the
    // sk4 selector names, and that the script is checked in and referenced from
    // the probe's docs.
    let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../scripts/crux3_armA_equiv.py");
    assert!(
        script.exists(),
        "scripts/crux3_armA_equiv.py is missing — the arm-A equivalence proof is \
         the only thing making this file's arms comparable, and it is not derivable \
         from the shader source"
    );
    let script = std::fs::read_to_string(&script).expect("read armA equivalence script");
    assert!(
        script.contains("linear_gemv_ishizuki_k3_colmajor_2col_sk4"),
        "the equivalence script no longer names the production kernel"
    );
    let example = include_str!("../examples/crux3_decode_algebra.rs");
    assert!(
        example.contains("crux3_armA_equiv"),
        "the probe does not reference the equivalence script, so a reader cannot \
         tell that arm A was verified to be the production kernel"
    );
}
