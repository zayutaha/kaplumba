//! The decode matvec's output must not move when `linear.metal` is edited.
//!
//! `tests/decode_nr0.rs` compares NR0 variants *against each other*. That is the
//! wrong question on its own: a change that re-associates the accumulator in
//! every kernel at once moves every variant together, so all of them still agree
//! with each other and the test passes while the model outputs different numbers.
//! Verified, not assumed -- re-ordering the four-lane butterfly so that
//! `shuffle_xor 1` comes after `shuffle_xor 2` instead of before is a *double*
//! accumulation, so it changes almost every output, 79325 of them across this
//! harness's cases, and `decode_nr0.rs` passed it without complaint because its
//! baseline was rebuilt from the same edited source.
//!
//! So the check that matters is against a fixed reference: the output the
//! **committed** kernel produces. This test holds that reference as raw bits.
//!
//! # What the reference covers
//!
//! See [`common::matvec_golden::CASES`]: 3 bpw -- 410 of this checkpoint's 411
//! tensors -- at five shapes, and 6 bpw, `lm_head`'s rate, at one. The other six
//! rates are unreachable from this checkpoint, so referencing them would pin
//! numbers nobody could act on.
//!
//! The 8 bpw kernel is the one place the explicit `fma` chain changed the output,
//! by one fp16 ulp in 2 of 21504 outputs; see `scripts/gen_exl3_gemv.py`. 8 bpw is
//! not dispatchable here, and 6 bpw -- which is, and where a 1-ulp logit change
//! can flip a token -- is bit-exact against the pre-NR0 kernel.
//!
//! # Regenerating
//!
//! Only when a `linear.metal` change is *meant* to move these bits, which under
//! this project's constraints means essentially never:
//!
//! ```text
//! KAPLUMBA_ACCEPT_MATVEC_GOLDEN=1 cargo test --release -p kaplumba-engine \
//!     --test decode_matvec_golden -- --nocapture
//! ```
//!
//! The env var makes the test print the new digest instead of failing, so the diff
//! is reviewed rather than rubber-stamped.

#[path = "common/matvec_golden.rs"]
mod matvec_golden;

use half::f16;
use matvec_golden::{CASES, compute, parse, render};

const GOLDEN: &str = include_str!("data/decode_matvec_golden.txt");

#[test]
fn the_decode_matvec_matches_its_committed_output() {
    let reference = parse(GOLDEN);
    assert_eq!(
        reference.len(),
        CASES.len(),
        "the golden file holds {} cases, this test runs {}; `CASES` changed without the \
         reference being regenerated",
        reference.len(),
        CASES.len()
    );

    if std::env::var_os("KAPLUMBA_ACCEPT_MATVEC_GOLDEN").is_some() {
        let cases = compute();
        print!("{}", render(&cases));
        eprintln!(
            "\nKAPLUMBA_ACCEPT_MATVEC_GOLDEN was set. The digest above is what belongs in \
             crates/engine/tests/data/decode_matvec_golden.txt. Review it before committing: \
             this is the model's arithmetic, and a 1-ulp change in a logit can flip a token."
        );
        return;
    }

    let got = compute();
    let failures: Vec<String> = got
        .iter()
        .zip(reference.iter())
        .filter_map(|(case, (want_header, want))| {
            if &case.header != want_header {
                return Some(format!(
                    "case order changed: this run produced '{}', the reference has '{want_header}'",
                    case.header
                ));
            }
            if case.values.len() != want.len() {
                return Some(format!(
                    "{}: {} outputs now, the reference has {}",
                    case.header,
                    case.values.len(),
                    want.len()
                ));
            }
            let bad: Vec<usize> = case
                .values
                .iter()
                .zip(want.iter())
                .enumerate()
                .filter(|(_, (a, b))| a != b)
                .map(|(i, _)| i)
                .collect();
            if bad.is_empty() {
                return None;
            }
            let sample: Vec<String> = bad
                .iter()
                .take(4)
                .map(|i| {
                    format!(
                        "col {i}: {:04x} now vs {:04x} in the reference (delta {:+e})",
                        case.values[*i],
                        want[*i],
                        f16::from_bits(case.values[*i]).to_f32()
                            - f16::from_bits(want[*i]).to_f32()
                    )
                })
                .collect();
            Some(format!(
                "{}: {}/{} outputs differ\n    {}",
                case.header,
                bad.len(),
                case.values.len(),
                sample.join("\n    ")
            ))
        })
        .collect();

    assert!(
        failures.is_empty(),
        "the decode matvec's output moved. Decode is bit-exact by project constraint (AGENTS.md \
         §8) and a 1-ulp change in a logit can flip a generated token, so this is a quality \
         change and not a refactor.\n\n{}\n\nIf the change was genuinely intended, regenerate \
         with KAPLUMBA_ACCEPT_MATVEC_GOLDEN=1 and review the diff.\n\nThe likeliest cause is FMA \
         contraction. The accumulator is an explicit `fma` chain precisely so the compiler \
         cannot re-associate it -- see the comment in `scripts/gen_exl3_gemv.py` -- so a \
         re-association here means that chain was edited, reformatted past the generator's \
         normalisation, or bypassed by an `_mcg` kernel that was hand-written.",
        failures.join("\n\n")
    );
}
