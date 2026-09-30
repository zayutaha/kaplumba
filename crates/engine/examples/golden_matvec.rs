//! Print the decode matvec's output digest, for a human to diff.
//!
//! The reference itself lives in `tests/decode_matvec_golden.rs`, which holds it
//! as `tests/data/decode_matvec_golden.txt` and compares against it. This example
//! exists so the diff can be *looked at* rather than only pass/failed: it prints
//! every output word, so a 1-ulp change is visible as a specific column rather
//! than as a failure message.
//!
//! ```text
//! cargo run --release --example golden_matvec > /tmp/after.txt
//! diff crates/engine/tests/data/decode_matvec_golden.txt /tmp/after.txt
//! ```
//!
//! The generator is shared with the test (`tests/common/matvec_golden.rs`) so the
//! two cannot drift: a reference can only be produced by the same code that
//! checks it.

#[path = "../tests/common/matvec_golden.rs"]
mod matvec_golden;

fn main() {
    print!("{}", matvec_golden::render(&matvec_golden::compute()));
}
