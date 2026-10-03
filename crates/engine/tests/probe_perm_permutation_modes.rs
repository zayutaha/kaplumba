//! `permutation_mode`: what the ten modes mean, which of them the engine can
//! reach, and what "the coherence probe searches" is.
//!
//! READ-ONLY PROBE. No Metal device, no model, no `gpuq`, no `TextModel`. Every
//! claim below is either a pure function of the committed sources
//! (`include_str!`, so the audit cannot be green against a stale checkout) or a
//! pure-Rust re-derivation of a formula transcribed out of them.
//!
//! ```text
//! cargo test -p kaplumba-engine --test probe_perm_permutation_modes
//! ```
//!
//! # What the field actually is
//!
//! `QuantizedLinear::permutation_mode` is documented as "Tile permutation
//! selector. 0 is the row-major default implied by the reference packer; other
//! values are the candidate fragment shuffles that the coherence probe
//! searches" (`model.rs:309-312`). **The second clause is false**, for three
//! separate reasons, each pinned by its own test:
//!
//! 1. `permutation_index` (`linear.metal:153-202`) is the ten-mode function, and
//!    it is reached from exactly one place -- `packed_state`
//!    (`linear.metal:373-379`) -- which returns early for every rate the engine
//!    can produce. `packed_state` is itself called from exactly one kernel,
//!    `linear_gemv` (`linear.metal:9918`), and the `LinearParams` that kernel
//!    is handed hardcodes `perm_mode: 0u32` (`linear.rs:719`).
//!    => **modes 1..=9 are unreachable.** See
//!    [`only_mode_zero_can_reach_permutation_index`].
//! 2. Every predicate in Rust that reads the field compares it to `1` (once,
//!    `!= 1`). Nothing anywhere branches on 2..=9. On the Rust side
//!    `permutation_mode` is a **boolean** meaning "this checkpoint's trellis tiles
//!    are stored column-major", not a shuffle selector. See
//!    [`the_rust_side_treats_permutation_mode_as_a_boolean`].
//! 3. The number `1` means two unrelated things, which is why the field reads as
//!    a shuffle selector when it is a layout flag. `permutation_index`'s mode 1 is
//!    an **intra**-tile transpose (`col * 16 + row`, `linear.metal:159-161`);
//!    `permutation_mode == 1` selects kernels for an **inter**-tile transpose
//!    (`exl3::reorder_tensor_to_colmajor`, `exl3.rs:733-757`: tile `(tk, tn)`
//!    moves from `(tk * tiles_n + tn) * 48` to `(tn * tiles_k + tk) * 48`).
//!    Nothing on the decode path executes code 1 of `permutation_index` at all.
//!    See [`mode_one_means_two_different_transposes`].
//!
//! # What the "coherence probe" is
//!
//! `real_model_permutation_sweep`, `crates/engine/tests/real_forward_buffer.rs:
//! 293-341`. It is a `#[test]` marked `#[ignore = "requires
//! KAPLUMBA_TEST_MODEL; permutation sweep"]`, it calls `TextModel::load_from_dir`
//! once per mode for ten modes, and it scores each mode on *knowledge*, not
//! speed: `"The capital of France is"` -> `" Paris"` and `"Count: 1 2 3 4 5 6 7
//! 8 9"` -> `" 10"`, scored by top-1 softmax mass. So it costs ~10 model loads
//! and answers "which shuffle decodes the weights correctly" -- never "which
//! shuffle is faster". It runs **only** under `--ignored` with
//! `KAPLUMBA_TEST_MODEL` set; there is no load-time or first-token cost. Pinned
//! by [`the_coherence_probe_is_an_ignored_test_with_no_production_caller`].
//!
//! Its recorded result, `spec/gemma4-metal-engine/implementation_notes.md:347`,
//! is one sentence: *"swept 10 candidate modes end to end
//! (`real_model_permutation_sweep`); none produced "Paris" or "10", so the
//! permutation is NOT the defect."* **Zero of ten modes were correct**, so there
//! was never a faster candidate to adopt -- and the test could not have found
//! one, because it never measures time. Pinned by
//! [`no_adopted_or_rejected_faster_mode_is_recorded_anywhere`].

const LINEAR_METAL: &str = include_str!("../../kernels/src/linear.metal");
const REPACKED_METAL: &str = include_str!("../../kernels/src/linear_repacked.metal");
const EXL3_METAL: &str = include_str!("../../kernels/src/exl3.metal");
const LINEAR_RS: &str = include_str!("../../kernels/src/linear.rs");
const MODEL_RS: &str = include_str!("../src/model.rs");
const SAFETENSORS_RS: &str = include_str!("../src/safetensors.rs");
const ENGINE_EXL3_RS: &str = include_str!("../src/exl3.rs");
const SWEEP_RS: &str = include_str!("real_forward_buffer.rs");
const BENCH_COLMAJOR_RS: &str = include_str!("../examples/bench_colmajor.rs");
const GEMMA4_NOTES: &str =
    include_str!("../../../spec/gemma4-metal-engine/implementation_notes.md");
const AMPLIFICATION_NOTES: &str =
    include_str!("../../../spec/decode-3bpw-amplification/implementation_notes.md");

/// From the first occurrence of `needle`, the text up to (not including) the
/// first following `close`.
///
/// `close` is the closing brace *with its own newline and indentation*, e.g.
/// `"\n}\n"` for a free function at column 0. Doing it in one place is the point:
/// hand-rolled `src[a..a+src[a..].find(..)]` slicing silently produced four
/// out-of-range slices while this file was being written, and each one read as
/// "the tripwire moved" rather than as "the arithmetic is wrong".
fn block_from<'a>(haystack: &'a str, needle: &str, close: &str) -> &'a str {
    let at = haystack
        .find(needle)
        .unwrap_or_else(|| panic!("tripwire not found in source: {needle:?}"));
    let rest = &haystack[at..];
    let end = rest
        .find(close)
        .unwrap_or_else(|| panic!("no {close:?} after {needle:?}"));
    &rest[..end]
}

/// How many characters into `haystack` the first `needle` sits. For messages.
#[allow(dead_code)]
fn offset_of(haystack: &str, needle: &str) -> usize {
    haystack.find(needle).unwrap_or_else(|| panic!("{needle:?}"))
}

// ---------------------------------------------------------------------------
// 1. The ten modes, transcribed
// ---------------------------------------------------------------------------

/// `permutation_index` from `linear.metal:153-202`, transcribed.
///
/// Maps a logical `(row, col)` of a 16x16 tile to its state index in the
/// trellis; `linear` is `row * 16 + col`, the row-major index.
///
/// The Metal function is `static`, so without a device the only faithful check
/// is the source tripwires in
/// [`the_metal_chain_is_exactly_modes_zero_through_nine_in_order`] plus the
/// closed-form behaviour pinned here. Every value below was cross-checked
/// against an independent Python transcription of the same source text.
fn permutation_index(linear: u32, mode: u32) -> u32 {
    let row = linear / 16;
    let col = linear % 16;
    match mode {
        0 => linear,
        // intra-tile transpose
        1 => col * 16 + row,
        // CUDA reconstruct shuffle: r0=(lane%4)*2, c0=lane/8, c1=c0+4.
        2 => {
            let p = (row % 8) / 2;
            let sub = (row % 8) % 2 + 2 * (row / 8);
            let q = col % 4;
            let csel = col / 4;
            64 * q + 8 * p + sub + 4 * csel
        }
        // original in-tree shuffle
        3 => {
            let t = linear / 8;
            let j = linear % 8;
            let r = ((t & 3) * 2) + (j & 1) + (((j >> 1) & 1) * 8);
            let c = (t >> 2) + (((j >> 2) & 1) * 8);
            r * 16 + c
        }
        4 => row * 16 + ((col + 4) & 15),
        5 => row * 16 + (col % 4) * 4 + col / 4,
        6 => row * 16 + (col % 8) * 2 + col / 8,
        7 => {
            // bit-reverse the 8-bit index
            let mut v = linear;
            let mut r = 0u32;
            for _ in 0..8 {
                r = (r << 1) | (v & 1);
                v >>= 1;
            }
            r
        }
        // See the finding below: this evaluates to `row * 17`.
        8 => ((row * 16 + col) & 0xF0) | (((row * 16 + col) >> 4) & 0x0F),
        9 => row * 16 + ((col + 8) & 15),
        _ => linear,
    }
}

/// The chain in `linear.metal` is exactly `mode == 0u ..= mode == 9u` plus a
/// `return linear` fallback, in the order read. And **two of the ten arms are not
/// permutations of the tile at all** -- the sharpest single fact about this
/// candidate list, invisible until you compute an image instead of reading
/// arithmetic:
///
/// * **mode 2** ("CUDA reconstruct shuffle") -> **160** distinct states out of
///   256. 96 logical `(row, col)` pairs read the *same* trellis state and 96
///   states are never read. A lossy 160-of-256 remap.
/// * **mode 8** -> **16** distinct states out of 256, one per row, every column
///   of a row aliased. The intended "swap the nibbles of the 8-bit index" does
///   not survive contact with the decomposition: `row * 16 + col` has `row` in
///   *both* nibbles after `>> 4`, so the arm evaluates to `row * 17`.
///
/// So "swept 10 candidate modes" was eight real shuffles plus two degenerate
/// maps. It does not weaken the sweep's recorded conclusion -- mode 0 is the
/// identity the reference packer implies and it works -- but the list was never
/// ten serious candidates.
#[test]
fn the_metal_chain_is_exactly_modes_zero_through_nine_in_order() {
    let body = block_from(
        LINEAR_METAL,
        "static uint permutation_index(uint linear, uint mode) {",
        "\n}\n",
    );

    let mut last = 0usize;
    for mode in 0..=9u32 {
        let needle = format!("if (mode == {mode}u) {{");
        let at = body.find(&needle).unwrap_or_else(|| {
            panic!(
                "mode {mode} arm is gone from permutation_index; the mode table in \
                 this file is stale"
            )
        });
        assert!(at > last, "mode {mode} is out of order -- the arms were reordered");
        last = at;
    }

    // Pin each arm's *return expression* against the source, so the Rust
    // transcription above cannot drift from the Metal it claims to transcribe.
    // Without this the image-size table below was asserting properties of a
    // hand-written reimplementation: a mutation that rewrote mode 8 in
    // linear.metal into a bijective nibble swap left the whole file green.
    const ARM_BODIES: &[(u32, &str)] = &[
        (0, "return linear;"),
        (1, "return col * 16u + row;"),
        (2, "return 64u * q + 8u * p + sub + 4u * csel;"),
        (3, "return r * 16u + c;"),
        (4, "return row * 16u + ((col + 4u) & 15u);"),
        (5, "return row * 16u + (col % 4u) * 4u + (col / 4u);"),
        (6, "return row * 16u + (col % 8u) * 2u + (col / 8u);"),
        (7, "for (uint i = 0u; i < 8u; ++i) { r = (r << 1u) | (v & 1u); v >>= 1u; }"),
        (8, "return ((row * 16u + col) & 0xF0u) | (((row * 16u + col) >> 4u) & 0x0Fu);"),
        (9, "return row * 16u + ((col + 8u) & 15u);"),
    ];
    for (mode, expected) in ARM_BODIES {
        let arm_start = body
            .find(&format!("if (mode == {mode}u) {{"))
            .expect("arm present");
        let arm_end = body[arm_start..]
            .find("return ")
            .map(|i| arm_start + i)
            .expect("arm has a return");
        let arm_end = body[arm_end..]
            .find('\n')
            .map(|i| arm_end + i)
            .unwrap_or(body.len());
        let arm = &body[arm_start..arm_end];
        assert!(
            arm.contains(expected),
            "mode {mode}'s arm no longer contains `{expected}`; it reads:\n{arm}\n\
             The Rust transcription at the top of this file must be updated with it, \
             or the image-size table below is asserting properties of a reimplementation \
             that no longer matches the Metal."
        );
    }
    assert_eq!(
        body.matches("if (mode == ").count(),
        10,
        "an `if (mode ==` arm appeared or vanished; enumerate it and say whether it \
         is reachable"
    );
    assert!(
        body.trim_end().ends_with("return linear;"),
        "the out-of-range fallback is no longer `return linear`, so an unrecognised \
         mode no longer behaves like mode 0"
    );

    /// Pinned image sizes. Modes 2 and 8 are the degenerate ones; everything else
    /// is a genuine permutation of 0..=255.
    const IMAGE_SIZES: [usize; 10] = [256, 256, 160, 256, 256, 256, 256, 256, 16, 256];
    for mode in 0..=9u32 {
        let image: std::collections::HashSet<u32> =
            (0..256).map(|l| permutation_index(l, mode)).collect();
        assert_eq!(
            image.len(),
            IMAGE_SIZES[mode as usize],
            "mode {mode}'s image size moved. If a degenerate arm was fixed that is a \
             good change -- but modes 2 and 8 must stay unreachable dead code (see \
             only_mode_zero_can_reach_permutation_index) or be given a caller."
        );
        assert!(
            image.iter().all(|v| *v < 256),
            "mode {mode} leaves the 16x16 tile"
        );
    }
    // The two degenerate arms as arithmetic, so a failure names which broke.
    for col in 0..16u32 {
        assert_eq!(
            permutation_index(5 * 16 + col, 8),
            5 * 16 + 5,
            "mode 8 is no longer column-blind (it used to collapse to `row * 17`)"
        );
    }
    assert_eq!(
        (0..256)
            .map(|l| permutation_index(l, 2))
            .collect::<std::collections::HashSet<u32>>()
            .len(),
        160,
        "mode 2 is no longer a lossy 160-of-256 remap"
    );
    // Mode 0 is the identity the reference packer implies; mode 1 is the only
    // arm that moves a whole row at once -- it is a transpose, not a shuffle.
    assert_eq!(permutation_index(0, 0), 0);
    assert_eq!(permutation_index(1, 1), 16);
    // The three arms whose order is self-evident from the code.
    assert_eq!(permutation_index(0b0000_0001, 7), 0b1000_0000, "mode 7 bit-reverses");
    assert_eq!(permutation_index(0b0000_0001, 4), 0b0000_0101, "mode 4 rotates by 4");
    assert_eq!(permutation_index(0b0000_0001, 9), 0b0000_1001, "mode 9 rotates by 8");
    // 4 and 9 really are row rotations, exhaustively.
    for row in 0..16u32 {
        for col in 0..16u32 {
            let linear = row * 16 + col;
            assert_eq!(permutation_index(linear, 4), (col + 4) % 16 + row * 16);
            assert_eq!(permutation_index(linear, 9), (col + 8) % 16 + row * 16);
        }
    }
    // ... and every non-degenerate, non-identity arm actually moves something.
    for mode in [1u32, 3, 4, 5, 6, 7, 9] {
        assert!(
            (0..256).any(|l| permutation_index(l, mode) != l),
            "mode {mode} is the identity; it is not a candidate"
        );
    }
}

// ---------------------------------------------------------------------------
// 2. Reachability: only mode 0, and only on a rate no checkpoint can have
// ---------------------------------------------------------------------------

/// Modes 1..=9 of `permutation_index` are dead code. Three independent locks,
/// any one of which suffices; all three are asserted so that relaxing one cannot
/// silently resurrect them.
///
/// This is the test that makes the doc comment on `permutation_mode`
/// falsifiable. If it ever fails, the field *becomes* a selector and this file's
/// framing is wrong -- so the failure message says exactly that.
#[test]
fn only_mode_zero_can_reach_permutation_index() {
    // (a) The `rate < 16u` early return sits *before* the `permutation_index`
    //     call, so every integer rate returns without ever consulting `mode`.
    let body = block_from(LINEAR_METAL, "static uint packed_state(", "\n}\n");
    let guard = body
        .find("if (rate < 16u) {")
        .expect("packed_state still short-circuits on rate < 16");
    let call = body
        .find("permutation_index(local_row * 16u + local_column, mode)")
        .expect("packed_state still calls permutation_index");
    assert!(
        guard < call,
        "the `rate < 16u` guard no longer precedes the permutation_index call, so \
         modes 1..=9 may now be reachable. Re-derive reachability before trusting \
         any other test in this file."
    );

    // (b) The only rate that gets past that guard is `16 + bits`, i.e.
    //     `Exl3Rate::Fractional`. The engine's loader refuses to build one:
    //     `integer_rate` rejects any rate with a fractional part.
    let ir = block_from(
        MODEL_RS,
        "fn integer_rate(bits: f32, field: &str)",
        "\n}\n",
    );
    assert!(
        ir.contains("bits.fract() != 0.0 || !(1.0..=8.0).contains(&bits)"),
        "integer_rate no longer refuses fractional rates, so an Exl3Rate::Fractional \
         -- and therefore permutation_index modes 1..=9 -- is now reachable from a \
         checkpoint"
    );
    // `encoded()` maps Integer(bits) -> bits. With bits in 1..=8 the largest value
    // the loader can produce is 8, so `rate < 16u` holds for every tensor a
    // checkpoint can build.
    for bits in 1u8..=8 {
        assert!(
            (bits as u32) < 16,
            "an integer rate of {bits} encodes to {bits}, which is >= 16 and would \
             take the permutation_index branch"
        );
    }
    assert!(
        MODEL_RS.contains("Ok(Exl3Rate::Integer(bits as u8))"),
        "integer_rate no longer yields Exl3Rate::Integer"
    );

    // (c) The one kernel that calls `packed_state` -- so the only kernel that can
    //     ever execute `permutation_index` -- is handed a hardcoded zero.
    assert_eq!(
        LINEAR_METAL.matches("packed_state(").count(),
        2,
        "packed_state's definition+call-site count moved from 2; re-derive which \
         kernels can run permutation_index"
    );
    assert_eq!(
        LINEAR_METAL.matches("packed_state(packed,").count(),
        1,
        "packed_state gained a second call site; re-derive which kernels can run \
         permutation_index"
    );
    assert_eq!(
        LINEAR_METAL.matches("static uint permutation_index(").count(),
        1,
        "linear.metal now defines permutation_index more than once"
    );
    assert_eq!(
        LINEAR_METAL.matches("uint lane = permutation_index(").count(),
        1,
        "the single permutation_index call site moved out of packed_state"
    );
    assert_eq!(
        LINEAR_METAL.matches("permutation_index(").count(),
        2,
        "permutation_index has gained or lost a definition or a call"
    );
    assert!(
        LINEAR_RS.matches("            perm_mode: 0u32,").count() >= 1,
        "no path hardcodes `perm_mode: 0u32` any more. `gemv()` was one such path: \
         if `LinearRequest` grew a perm_mode field, modes 1..=9 are reachable."
    );
    // `LinearRequest` -- the struct `gemv()` takes -- carries no perm_mode at all,
    // which is the structural reason (c) holds.
    assert!(
        !block_from(LINEAR_RS, "pub struct LinearRequest<'a> {", "\n}\n").contains("perm_mode"),
        "LinearRequest gained a perm_mode field, so gemv() can be handed a non-zero \
         mode and permutation_index modes 1..=9 become reachable"
    );
    // The two encode paths that *do* take a caller-supplied perm_mode
    // (`LinearEncodeRequest`, `PrefillEncodeRequest`) are only ever called by
    // `model.rs` with `linear.permutation_mode`, which is 0 or 1.
    for ty in ["pub struct LinearEncodeRequest<'a> {", "pub struct PrefillEncodeRequest<'a> {"] {
        assert!(
            block_from(LINEAR_RS, ty, "\n}\n").contains("pub perm_mode: u32"),
            "{ty} no longer carries a perm_mode field"
        );
    }
    assert!(
        MODEL_RS.contains("perm_mode: linear.permutation_mode,"),
        "model.rs no longer forwards the field verbatim to both request types; if a \
         literal was substituted, modes 2..=9 may be reachable"
    );
    assert_eq!(
        MODEL_RS.matches("perm_mode: linear.permutation_mode,").count(),
        2,
        "one of the two forward sites (gemv, gemm) changed"
    );

    // And the second `permutation_index` in the tree -- the one `exl3_decode_tile`
    // actually calls -- is a stub that ignores its argument entirely, so even the
    // Fractional path cannot vary by mode through it.
    let stub = block_from(
        EXL3_METAL,
        "static uint permutation_index(uint row, uint mode) {",
        "\n}\n",
    );
    assert_eq!(
        stub.matches("return row;").count(),
        1,
        "exl3.metal's permutation_index is no longer the identity stub; it is now a \
         second, independent mode table that this file does not enumerate"
    );
    assert!(
        !stub.contains("mode =="),
        "exl3.metal's permutation_index now branches on mode; that is a THIRD mode \
         selector and it needs enumerating here"
    );
    assert!(
        EXL3_METAL.contains("permutation_index(gid, 0u)"),
        "exl3_decode_tile no longer passes mode 0; the argument above needs re-reading"
    );
}

// ---------------------------------------------------------------------------
// 3. On the Rust side it is a boolean, not a selector
// ---------------------------------------------------------------------------

/// Every comparison against the field is against `1` (once, `!= 1`). Nothing
/// branches on 2..=9, so `permutation_mode` is a boolean "the trellis is stored
/// column-major" and the field's doc comment over-promises.
///
/// Note *how much* is gated on it: 15 comparisons in `model.rs` alone, covering
/// the q/k/v, gate/up, o/down and cross-layer fusions. A row-major checkpoint
/// does not merely run a different tile shuffle, it runs a **different decode
/// graph** -- and no fused row-major kernel exists in the tree at all.
#[test]
fn the_rust_side_treats_permutation_mode_as_a_boolean() {
    let mut operands: Vec<(usize, String, String)> = Vec::new();
    for src in [LINEAR_RS, MODEL_RS] {
        for (i, line) in src.lines().enumerate() {
            let line = line.trim();
            for field in ["perm_mode", "permutation_mode"] {
                for op in [" == ", " != "] {
                    let pat = format!("{field}{op}");
                    if let Some(at) = line.find(pat.as_str()) {
                        let operand: String = line[at + pat.len()..]
                            .chars()
                            .take_while(|c| c.is_ascii_digit() || *c == 'u')
                            .collect();
                        let operand = operand.trim_end_matches('u');
                        if !operand.is_empty() {
                            operands.push((i + 1, line.to_string(), operand.to_string()));
                        }
                    }
                }
            }
        }
    }
    assert!(
        operands.len() >= 20,
        "only {} perm_mode comparisons found, expected the 22 audited (7 in \
         linear.rs, 15 in model.rs). Either the field moved or the two sources have \
         stopped consulting it, and this test would then be auditing nothing.",
        operands.len()
    );
    let exotic: Vec<_> = operands
        .iter()
        .filter(|(_, _, operand)| operand != "1" && operand != "0")
        .collect();
    assert!(
        exotic.is_empty(),
        "perm_mode is now compared against something other than 0 or 1: {exotic:?}. \
         If that is a real selector, enumerate it here -- the field's doc comment is \
         only wrong while there is exactly one interesting value."
    );

    // Comparing is not the whole surface: the field can also be *written*. A
    // mutation that appended `if linear.in_features == 5376 { linear.permutation_mode
    // = 2; }` to the colmajor block was CAUGHT by neither the comparison scan nor
    // the "one `= 1;` write" pin, because it is a comparison-free assignment to a
    // different literal. So scan assignments too. `set_permutation_mode`'s
    // `= mode;` is the one legitimate variable write and is excluded by name.
    let mut writes: Vec<(usize, String)> = Vec::new();
    for (i, line) in MODEL_RS.lines().enumerate() {
        for pat in [
            ".permutation_mode = ",
            "permutation_mode = ",
            ".perm_mode = ",
        ] {
            if let Some(at) = line.find(pat) {
                let rest = line[at + pat.len()..].trim_start();
                if let Some(lit) = rest.strip_prefix(|c: char| c.is_ascii_digit()) {
                    let digits: String =
                        rest.chars().take_while(char::is_ascii_digit).collect();
                    writes.push((i + 1, format!("{pat}{digits}: {line}")));
                    let _ = lit;
                }
            }
        }
    }
    for (line_no, text) in &writes {
        assert!(
            text.contains("= 1;") || text.contains("= 0;"),
            "the field is assigned a literal other than 0 or 1 at model.rs:{line_no}: \
             {text:?}. A write is how permutation_mode becomes a real selector, and \
             a comparison-free write is invisible to the comparison scan above."
        );
    }
    assert_eq!(
        writes.len(),
        2,
        "expected exactly two literal writes to the field in model.rs (0 in \
         QuantizedLinear::new/from_metal_buffer, 1 in the colmajor block); found {writes:?}"
    );
    assert!(
        LINEAR_RS.contains("perm_mode == 1"),
        "kernels/src/linear.rs no longer branches on perm_mode == 1"
    );
    assert_eq!(
        MODEL_RS.matches("permutation_mode == 1").count(),
        15,
        "the count of `permutation_mode == 1` gates in model.rs moved from 15. These \
         are the fusion gates a colmajor checkpoint gets and a row-major one does \
         not, so re-enumerate them."
    );

    // The colmajor decision itself: one global write, one key, seven per-layer
    // projections, NOT the head.
    assert_eq!(
        MODEL_RS.matches("linear.permutation_mode = 1;").count(),
        1,
        "the colmajor assignment is no longer a single unconditional write; re-audit \
         which tensors it covers"
    );
    let load = block_from(
        MODEL_RS,
        "if metadata.quantization.layout.as_deref() == Some(\"colmajor\") {",
        "\n        }\n",
    );
    for proj in [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ] {
        assert!(load.contains(proj), "the colmajor block no longer covers {proj}");
    }
    assert!(
        !load.contains("lm_head"),
        "the colmajor block now covers lm_head, but lm_head.trellis is NOT \
         transposed (shape [336, 16384, 96] in both checkpoints). Such a checkpoint \
         would load and then decode garbage."
    );
    assert!(
        SAFETENSORS_RS.contains(
            "if quantization.layout.as_deref() == Some(\"colmajor\") && module != \"lm_head\" {"
        ),
        "the trellis-shape validator no longer exempts lm_head from the colmajor \
         shape, so a colmajor checkpoint's head would be rejected at load"
    );

    // `set_permutation_mode` is the footgun this test was written to find: it
    // writes the head, the loader's colmajor block does not. So on the only
    // colmajor checkpoint in the tree EVERY value of `mode` corrupts one half or
    // the other -- 0 mis-indexes the seven layers, 1 mis-indexes lm_head. There is
    // no value at all that is correct for a colmajor checkpoint.
    let setter = block_from(
        MODEL_RS,
        "pub fn set_permutation_mode(&mut self, mode: u32) {",
        "\n    }\n",
    );
    assert!(
        setter.contains("self.lm_head.permutation_mode = mode;"),
        "set_permutation_mode no longer writes lm_head. If that changed, re-audit \
         whether calling it on a colmajor checkpoint is still a footgun."
    );
    assert!(
        setter.contains("linear.permutation_mode = mode;"),
        "set_permutation_mode no longer writes the layer projections"
    );
}

// ---------------------------------------------------------------------------
// 4. Mode 1 means two unrelated transposes
// ---------------------------------------------------------------------------

/// `permutation_index`'s mode 1 is an intra-tile transpose. The `layout:
/// "colmajor"` decision is an inter-tile transpose. They share the number 1 and
/// nothing else, and this is the entire reason the field reads as a shuffle
/// selector when it is a layout flag.
#[test]
fn mode_one_means_two_different_transposes() {
    // (a) intra-tile: logical (row, col) of a 16x16 tile -> state index.
    assert_eq!(permutation_index(5 * 16 + 9, 1), 9 * 16 + 5);
    // (b) inter-tile: the trellis is [tiles_k, tiles_n, 48] u16 row-major and
    //     [tiles_n, tiles_k, 48] col-major. Same length, same bytes, same codes --
    //     the trellis is not re-encoded, only reordered.
    assert!(
        ENGINE_EXL3_RS.contains(
            "/// In row-major, tile (tk, tn) is at index (tk * tiles_n + tn) * 48."
        ),
        "reorder_tensor_to_colmajor's contract comment moved (exl3.rs:731)"
    );
    assert!(
        ENGINE_EXL3_RS.contains(
            "/// In column-major, tile (tk, tn) is at index (tn * tiles_k + tk) * 48."
        ),
        "reorder_tensor_to_colmajor's contract comment moved (exl3.rs:732)"
    );
    assert!(
        ENGINE_EXL3_RS.contains("let expected = tiles_k * tiles_n * 48;"),
        "the colmajor repacker is no longer 48 u16 per tile, so the byte layout it \
         produces -- and the alignment argument below -- has changed"
    );
    // The tile stride is 48 u16 = 96 bytes at 3 bpw, a multiple of 4, so the
    // repack cannot change any tensor's 4-byte alignment. That is why the
    // alignment gate behaves identically on both layouts -- see
    // `perm_mode_cannot_rescue_a_misaligned_tensor`.
    assert!(
        ENGINE_EXL3_RS.contains("let src_offset = (tk * tiles_n + tn) * 48;"),
        "the row-major source offset moved (exl3.rs:750)"
    );
    assert!(
        ENGINE_EXL3_RS.contains("let dst_offset = (tn * tiles_k + tk) * 48;"),
        "the colmajor destination offset moved (exl3.rs:751)"
    );

    // ... and the decode path really does use (b), never (a): the three
    // `tile_index` transpositions that carry `perm_mode == 1` into the kernels.
    let sites: usize = [
        "uint tile_index = (params.perm_mode == 1u)",
        "uint tile_idx = (params.perm_mode == 1u)",
    ]
    .iter()
    .map(|needle| LINEAR_METAL.matches(needle).count())
    .sum();
    assert_eq!(
        sites, 3,
        "expected the inter-tile transposition at the three LinearParams sites \
         (linear.metal:9915, 10042, 10167); found {sites}"
    );
    assert!(
        LINEAR_METAL.contains("tile_column * (params.in_features / 16u) + tile_row"),
        "the colmajor tile_index is no longer `tile_column * tiles_k + tile_row`"
    );
    // The colmajor kernels themselves never call permutation_index: they are a
    // separate file with their own extractors.
    assert!(
        !REPACKED_METAL.contains("permutation_index"),
        "linear_repacked.metal now calls permutation_index, so the colmajor path DOES \
         execute a mode after all and this file's framing is wrong"
    );
    for needle in ["extract_v_mcg_32", "extract_v_mcg_fast"] {
        assert!(
            REPACKED_METAL.contains(needle),
            "{needle} is gone from linear_repacked.metal; the colmajor kernels no \
             longer decode through the shared mcg extractors"
        );
    }
    // And there is no row-major fused kernel to compare against: every
    // `linear_gemv_ishizuki_k3_fused*` lives in linear_repacked.metal and is
    // colmajor by construction. A row-major checkpoint cannot use any of the six
    // decode fusions.
    assert!(
        !LINEAR_METAL.contains("kernel void linear_gemv_ishizuki_k3_fused"),
        "a fused GEMV appeared in linear.metal, so a row-major checkpoint may now \
         have a fused path; re-audit what perm_mode 0 costs"
    );
}

// ---------------------------------------------------------------------------
// 5. The coherence probe
// ---------------------------------------------------------------------------

/// "the coherence probe" is one `#[ignore]`d integration test. No production
/// caller, no benchmark harness, and it runs only under `--ignored` with
/// `KAPLUMBA_TEST_MODEL` set, so it costs nothing at load or on the first token.
/// It also cannot find a *faster* mode: it scores knowledge, not time.
#[test]
fn the_coherence_probe_is_an_ignored_test_with_no_production_caller() {
    let body = block_from(SWEEP_RS, "fn real_model_permutation_sweep()", "\n}\n");
    let window = {
        let at = offset_of(SWEEP_RS, "fn real_model_permutation_sweep()");
        &SWEEP_RS[at - SWEEP_RS[..at].rfind("#[test]").expect("annotated")..at]
    };
    assert!(
        window.contains("#[ignore = \"requires KAPLUMBA_TEST_MODEL; permutation sweep\"]"),
        "real_model_permutation_sweep is no longer #[ignore]d, so it would start \
         costing a model load in every `cargo test` run"
    );
    // Ten modes, one model load each. This is why it is expensive when run, and
    // also why it is not a load-time cost anyone should worry about.
    assert!(
        body.contains("for mode in 0..=9_u32 {"),
        "the sweep no longer enumerates modes 0..=9 (was real_forward_buffer.rs:303)"
    );
    assert_eq!(
        body.matches("TextModel::load_from_dir(path)").count(),
        1,
        "the sweep stopped reloading per mode. If it now loads once and re-permutes \
         in place, that is a behavioural change worth recording -- the loader is \
         what sets permutation_mode in the first place."
    );
    // It scores on the answer, not on the clock.
    assert!(
        body.contains("\"The capital of France is\", \" Paris\""),
        "the first probe prompt changed; the sweep is a knowledge probe"
    );
    assert!(
        body.contains("\"Count: 1 2 3 4 5 6 7 8 9\", \" 10\""),
        "the second probe prompt changed; the sweep is a knowledge probe"
    );
    assert!(
        !body.contains("Instant::") && !body.contains("elapsed("),
        "the sweep started timing. If so it is a performance probe, and the notes' \
         \"none produced Paris\" is not the whole result."
    );
    // `set_permutation_mode` -- the only way to write a mode other than 0 or 1 --
    // has no caller in `src/` at all: it appears exactly once in model.rs, as its
    // own definition.
    assert_eq!(
        MODEL_RS.matches("set_permutation_mode").count(),
        1,
        "src/model.rs mentions set_permutation_mode {} time(s) beyond its \
         definition, so the sweep is no longer the only way to reach a mode other \
         than 0 or 1 and modes 2..=9 would need a reachability argument",
        MODEL_RS.matches("set_permutation_mode").count() - 1
    );
    assert!(
        !LINEAR_RS.contains("set_permutation_mode"),
        "kernels/src now calls set_permutation_mode"
    );
    // The sweep is the sole caller, and it is in a test file.
    assert!(
        SWEEP_RS.contains("model.set_permutation_mode(mode);"),
        "the sweep no longer drives set_permutation_mode"
    );
}

/// The recorded outcome is one sentence in one spec folder, it says **none of
/// the ten worked**, and no table of per-mode timings exists anywhere -- because
/// the sweep never measured time. So there is no "found but not adopted" mode,
/// and there never was a candidate to adopt.
#[test]
fn no_adopted_or_rejected_faster_mode_is_recorded_anywhere() {
    let line = GEMMA4_NOTES
        .lines()
        .find(|l| l.contains("real_model_permutation_sweep"))
        .unwrap_or_else(|| {
            panic!(
                "spec/gemma4-metal-engine/implementation_notes.md no longer records \
                 the sweep. If its result moved, move this assertion with it -- the \
                 answer to \"was a faster mode found and not adopted?\" lives in that \
                 line and nowhere else."
            )
        });
    // The sentence is a single bullet in the notes; make sure it is still that
    // bullet rather than a passing mention in prose.
    assert!(
        line.contains("permutation_index is now a runtime selector")
            || line.contains("runtime selector"),
        "the sweep bullet no longer states that permutation_index became a runtime \
         selector: {line:?}"
    );
    assert!(
        line.contains("none produced"),
        "the sweep's recorded outcome changed: {line:?}. Re-read it before claiming \
         that no candidate mode was found."
    );
    assert!(
        line.contains("permutation_index") && line.contains("set_permutation_mode"),
        "the note no longer says permutation_index became a runtime selector: {line:?}"
    );
    // Pin the doc comment on the field, which is the thing that misleads. When it
    // is corrected, these two assertions have to go *deliberately*, together with
    // the framing in the file header.
    assert!(
        MODEL_RS.contains(
            "    /// Tile permutation selector. 0 is the row-major default implied by the"
        ),
        "the permutation_mode doc comment's first sentence was reworded. If it now \
         describes the field honestly -- \"selector\" -> \"column-major flag\" -- \
         record that here, because that is the same edit this test is asking for."
    );
    assert!(
        MODEL_RS.contains("the coherence probe searches."),
        "the \"candidate fragment shuffles that the coherence probe searches\" clause \
         is gone from permutation_mode's doc comment. Good -- and \
         only_mode_zero_can_reach_permutation_index independently shows modes 1..=9 \
         are unreachable, so the remaining half of that sentence (\"selector\") should \
         go in the same edit. Record that here when it does."
    );
}

// ---------------------------------------------------------------------------
// 6. perm_mode vs the alignment gate
// ---------------------------------------------------------------------------

/// `perm_mode` does **not** relax the gate at `linear.rs:1520`
/// (`packed_offset % 4 != 0 -> None`): the gate is unconditional and comes first
/// inside `ishizuki_gemv_pipeline`. But the premise "a misaligned tensor refuses
/// ishizuki entirely" is only true for `perm_mode != 1`, because the two
/// `perm_mode == 1` arms in `encode_gemv_raw` sit *above* the selector and return
/// before it is ever called.
#[test]
fn perm_mode_cannot_rescue_a_misaligned_tensor() {
    let body = block_from(LINEAR_RS, "fn ishizuki_gemv_pipeline(", "\n    }\n");
    let gate = body
        .find("if bits < 1 || bits > 8 || packed_offset % 4 != 0 {")
        .unwrap_or_else(|| {
            panic!(
                "the alignment gate is gone from ishizuki_gemv_pipeline \
                 (was linear.rs:1520). Re-audit: what refuses an unaligned tensor \
                 now, and is perm_mode consulted before it?"
            )
        });
    let colmajor = body
        .find("if perm_mode == 1 || env_flag(\"KAPLUMBA_REPACKED_GEMV\") {")
        .unwrap_or_else(|| {
            panic!("the colmajor lookup is gone from ishizuki_gemv_pipeline (was 1539)")
        });
    assert!(
        gate < colmajor,
        "the colmajor lookup now precedes the alignment gate, so perm_mode CAN \
         rescue an unaligned tensor; update the finding in the file header"
    );
    assert!(
        body[gate..].starts_with(
            "if bits < 1 || bits > 8 || packed_offset % 4 != 0 {\n            return None;"
        ),
        "the alignment gate no longer returns None immediately"
    );
    // Ordering alone is not the invariant; the *semantic* one is that the gate
    // does not consult perm_mode at all. Reordering it after the colmajor lookup
    // is one way to break it; making it `... && perm_mode != 1` is another, and
    // the second leaves the textual order untouched. Caught: a mutation that
    // inserted `&& perm_mode != 1` into the gate condition left this test green
    // while a misaligned colmajor tensor started reaching a colmajor kernel.
    let gate_line = body[gate..].lines().next().unwrap_or("");
    assert!(
        !gate_line.contains("perm_mode"),
        "the alignment gate now consults perm_mode: {gate_line:?}. A misaligned \
         colmajor tensor would then be dispatched instead of refused, which is the \
         silent-gibberish failure recorded in \
         spec/decode-19tok-repacked-250gb/implementation_notes.md."
    );

    // ... and the two perm_mode == 1 arms above it, which is the real answer:
    // a colmajor 3 bpw tensor is dispatched without the gate ever being consulted.
    let raw = offset_of(LINEAR_RS, "pub fn encode_gemv_raw(");
    let sk2 = raw + offset_of(&LINEAR_RS[raw..], "pub fn encode_gemv_raw_sk2(");
    for (needle, what) in [
        (
            "if perm_mode == 1 && out_features % 32 == 0 && !env_flag(\"KAPLUMBA_NO_2COL\") {",
            "2col",
        ),
        (
            "if perm_mode == 1 && out_features % 64 == 0 && env_flag(\"KAPLUMBA_4COL\") {",
            "4col",
        ),
    ] {
        let at = offset_of(LINEAR_RS, needle);
        assert!(
            raw < at && at < sk2,
            "the perm_mode == 1 {what} arm moved out of encode_gemv_raw; re-audit \
             which encode entry point can bypass ishizuki_gemv_pipeline's gate"
        );
    }
    let selector = offset_of(LINEAR_RS, "if let Some(pipeline) = self.ishizuki_gemv_pipeline(rate, packed_offset, perm_mode)");
    assert!(
        sk2 > selector && selector > raw,
        "the ishizuki_gemv_pipeline call inside encode_gemv_raw moved; the ordering \
         argument above depends on it coming last"
    );

    // The concrete hazard this creates: those arms bind `packed_offset`, and the
    // colmajor kernels cast the pointer to `uint *`, which a % 4 != 0 byte offset
    // invalidates. `ring_word<ALIGNED>` (`linear.metal:267`) exists for exactly
    // this on the other path and is not used here. Untriggerable on this
    // checkpoint -- 410 of 411 tensors are 4-byte aligned
    // (`spec/decode-bandwidth-handover/spec.md:34`) -- recorded so it is not
    // rediscovered as a mystery.
    assert!(
        REPACKED_METAL.contains("(const device uint *)packed"),
        "the colmajor kernels no longer cast packed to uint*, so the alignment \
         asymmetry described above no longer applies"
    );
    assert!(
        LINEAR_METAL.contains("template <bool ALIGNED>"),
        "ring_word<ALIGNED> is gone from linear.metal"
    );
    assert!(
        LINEAR_METAL.contains("return ((const device uint *)packed)[index];"),
        "ring_word's ALIGNED fast path changed; re-read the alignment note at \
         linear.metal:266-276"
    );
}

// ---------------------------------------------------------------------------
// 7. Was the colmajor gain measured?
// ---------------------------------------------------------------------------

/// It was measured **once, as a kernel microbenchmark, against a harness that no
/// longer has a row-major arm**, and never end to end. This test pins that,
/// because the numbers are still quoted as a layout result.
///
/// The evidence:
///   * `spec/decode-19tok-repacked-250gb/implementation_notes.md:36-43` quotes
///     `q_proj 94.8 -> 101.8 GB/s`, `up_proj 151.5 -> 180.5`, `down 148.8 ->
///     177.8`. Single readings; `min`-of-5 batches of 20 in one process; no
///     interleaving with the other arm; and against `_colmajor`, which is not a
///     kernel the engine dispatches today
///     (`perf_probe_dispatchsel_columns.rs:71`, finding 2: "`_colmajor_contig4`
///     never dispatches").
///   * `crates/engine/examples/bench_colmajor.rs:45-46` still *builds* the
///     row-major buffer and immediately discards it: `packed_orig` is consumed
///     only by `reorder_tensor_to_colmajor`. All five kernels in its list are
///     `_colmajor*`. There is no row-major arm left to reproduce the left-hand
///     column of that arrow.
///   * No spec records an end-to-end decode rate for a row-major checkpoint at
///     any HEAD. The last one, `spec/decode-bandwidth-handover/spec.md:19`
///     (~11.2 tok/s), predates the repacker *and* every kernel change since, so
///     it cannot be differenced against the colmajor figures.
#[test]
fn the_row_major_arm_of_the_colmajor_comparison_no_longer_exists() {
    // (a) The harness has no row-major arm.
    let build = offset_of(BENCH_COLMAJOR_RS, "let packed_orig = build(in_f, out_f, bits, 7);");
    let repack = offset_of(
        BENCH_COLMAJOR_RS,
        "let packed = reorder_tensor_to_colmajor(&packed_orig, in_f, out_f)",
    );
    assert!(build < repack, "bench_colmajor's construction order changed");
    assert_eq!(
        BENCH_COLMAJOR_RS.matches("packed_orig").count(),
        2,
        "packed_orig is now used more than the build and the repack. If that is a \
         row-major arm being restored, bench_colmajor DOES have one, \
         decode-19tok-repacked-250gb/implementation_notes.md:36-43 becomes \
         reproducible, and this test should be INVERTED -- not relaxed."
    );
    let kern = offset_of(BENCH_COLMAJOR_RS, "for (k_name, label) in [");
    let names = &BENCH_COLMAJOR_RS[kern..kern + offset_of(&BENCH_COLMAJOR_RS[kern..], "] {")];
    let rows: Vec<&str> = names
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with('(') && l.contains("\"linear_gemv"))
        .collect();
    assert_eq!(
        rows.len(),
        5,
        "bench_colmajor's kernel list no longer has the 5 audited colmajor variants; \
         it now reads {rows:?}"
    );
    assert!(
        rows.iter().all(|n| n.contains("_colmajor")),
        "bench_colmajor now benchmarks a non-colmajor kernel: {rows:?}. If that is a \
         row-major arm being restored, the left-hand side of \
         decode-19tok-repacked-250gb/implementation_notes.md:36-43 becomes \
         reproducible and this test should be inverted, not relaxed."
    );
    // `perm_mode: 0` is passed to `encode_gemv_named` throughout, i.e. the harness
    // bypasses the selector by kernel name, so `LinearParams.perm_mode` is 0 even
    // while a colmajor kernel runs. Worth knowing before anyone reads this harness
    // as a perm_mode experiment.
    assert!(
        BENCH_COLMAJOR_RS.matches("perm_mode: 0,").count() >= 1,
        "bench_colmajor no longer hardcodes perm_mode: 0"
    );
    assert!(
        !BENCH_COLMAJOR_RS.contains("perm_mode: 1,"),
        "bench_colmajor now passes perm_mode: 1; if so it is exercising the \
         selector, not just the kernels, and the audit needs re-reading"
    );

    // (b) The A/B pair is a *perfect* controlled pair on data and has simply never
    //     been timed. Same shards, same tensor names, same dtype, same element
    //     count; only the two leading axes of each per-layer trellis are
    //     transposed, and lm_head is untouched. Pinned against the spec table
    //     rather than the checkpoints, so the test needs no model on disk.
    assert!(
        AMPLIFICATION_NOTES
            .contains("| `Gxma/Gxma` | 16,028,050,244 | none (row-major) | 3.0 | 6 | mcg |"),
        "the row-major/colmajor model-directory table moved or changed shape; \
         decode-3bpw-amplification/implementation_notes.md:258 is the record that \
         the two checkpoints differ ONLY by layout"
    );
    assert!(
        AMPLIFICATION_NOTES
            .contains("| `Gxma-repacked` | 16,028,388,994 | `colmajor` | 3.0 | 6 | mcg |"),
        "the colmajor half of that table moved (was \
         decode-3bpw-amplification/implementation_notes.md:259)"
    );
    assert!(
        AMPLIFICATION_NOTES.contains("**the 15.3 tok/s baseline**"),
        "the colmajor checkpoint's current baseline is no longer quoted as 15.3 tok/s. \
         The row-major half is still unmeasured, so whatever replaced that number is \
         the one to difference against."
    );
    assert!(
        !AMPLIFICATION_NOTES.contains("row-major vs colmajor"),
        "decode-3bpw-amplification now claims a row-major vs colmajor comparison. If \
         so, read it: it would make this test wrong and the answer to question 4 a \
         'yes'."
    );
}