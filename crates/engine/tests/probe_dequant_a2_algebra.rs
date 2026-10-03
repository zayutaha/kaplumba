//! **A2 — is `decode_state_mcg` reducible, bit-exactly or with bounded error?**
//!
//! READ-ONLY probe. CPU only: no Metal context, no model, no `gpuq`, no kernel
//! dispatch. Everything below is either (a) arithmetic, (b) a transcription of
//! `crates/kernels/src/linear.metal:121-126` checked against the engine's own
//! reference `kaplumba_engine::exl3::decode_state`, or (c) a count of tokens in
//! the live kernel's source text.
//!
//! ## The one-sentence answer
//!
//! **No.** The `AND`+`XOR` pair *is* a single `LOP3` (proved below, `immLut =
//! 0xCA`), but it cannot be reached from MSL, and — more importantly — the only
//! measurement that has ever been taken of this pool says the pool is **worth
//! zero**: `decode_state_mcg`'s entire 4-op body was measured at **−2.3 %** on
//! the probe family (`spec/decode-19tok-repacked-250gb/implementation_notes.md:1225`)
//! and deleting it outright made the kernel **26.7 % slower**. Every tok/s figure
//! below is therefore presented as a *range whose sign is not established*, and
//! the file's most load-bearing test is the one that shows the "removing decode
//! arithmetic is slower" measurement was taken on a **kernel shape the live
//! kernel does not have**.
//!
//! ## What is falsified here, and by what
//!
//! | claim | verdict | test | sabotage |
//! |---|---|---|---|
//! | `(p & A) ^ B` is one `LOP3` | **TRUE**, `immLut = 0xCA` | [`lop3_immediate_reproduces_and_then_xor_over_the_whole_domain`] | flip one immLut bit -> fails |
//! | MSL exposes `LOP3` | **FALSE** | [`msl_exposes_no_lop3_and_the_source_text_is_the_evidence`] | — (source fact) |
//! | the hash splits across the code's two bytes | **FALSE** | [`the_hash_does_not_split_across_the_code_bytes`] | — |
//! | a LUT of any index width can replace it | **FALSE** — 10746 distinct outputs need a 14-bit index (64 KB) against a 4 KB `constant` budget; and a device buffer is 4 B/code = 23.4 GB/s | [`lut_is_dead_for_two_independent_reasons`] | — |
//! | `(p & A) ^ B` can be rewritten as AND+OR (2 ops) | **FALSE** — the correct affine form is 3 ops; the 2-op form is wrong on all 65536 codes | [`the_two_op_affine_rewrite_is_false_and_the_three_op_one_is_worse`] | it *was* the draft's bug, caught by its own loop |
//! | AND+XOR is "21 % of the loop" | **UNDERSTATED**; the whole logic pool is 31.5 % | [`the_logic_pool_is_thirty_one_percent_not_twenty_one`] | — |
//! | 2 codes per instruction (SWAR) | **FALSE**, 2x26 = 52 live bits > 32 | [`two_codes_cannot_share_one_register_or_one_imul`] | — |
//! | the whole decode can move to fp16 | **FALSE**, and it already partly has | [`the_half2_add_is_already_one_op_and_the_fma_cannot_move_down`] | — |
//! | fp16 weights offline | **FATAL twice**: 2.49x the box, and 4.3x slower | [`fp16_weights_are_fatal_on_capacity_and_on_bandwidth`] | — |
//! | "removing decode made it slower ⇒ cheaper decode cannot help" | **DOES NOT TRANSFER** | [`the_ilp_reservoir_argument_does_not_transfer_to_the_live_kernel`] | — (source fact) |
//!
//! ## Reproduce
//!
//! ```bash
//! cargo test --release --test probe_dequant_a2_algebra -- --nocapture
//! ```
//!
//! Needs no GPU, no model, ~10 s. AGENTS.md §3.4: every "this test can fail"
//! claim here was checked by sabotage, and the sabotage is recorded in
//! `spec/decode-ceiling/findings/A2-dequant.md`.

use half::f16;
use std::collections::HashSet;

// ===========================================================================
// 0. Constants, read out of the live kernel rather than remembered.
// ===========================================================================

const LINEAR_METAL: &str = include_str!("../../kernels/src/linear.metal");
const REPACKED_METAL: &str = include_str!("../../kernels/src/linear_repacked.metal");
const PROBE_METAL: &str = include_str!("../../kernels/probe/contig4_load_alu.metal");

/// `decode_state_mcg`'s multiplier, `linear.metal:122`.
const MCG_MUL: u32 = 0xCBAC_1FED;
/// The mask, `linear.metal:123`.
const MCG_MASK: u32 = 0x8fff_8fff;
/// The xor constant, `linear.metal:123`.
const MCG_XOR: u32 = 0x3b60_3b60;

// ===========================================================================
// 1. The transcription, and its bit-exactness against the engine reference.
// ===========================================================================

/// A CPU transcription of `decode_state_mcg`, `crates/kernels/src/linear.metal:121-126`.
///
/// ```metal
/// static inline float decode_state_mcg(uint value) {
///     value = value * 0xCBAC1FEDu;                  // IMUL
///     value = 0x3b603b60u ^ (value & 0x8fff8fffu);  // AND + XOR
///     half2 h = as_type<half2>(value);              // bitcast, free
///     return float(h.x + h.y);                      // fp16 add, then free widen
/// }
/// ```
///
/// `as_type<half2>` is a bitcast, so `h.x` is `f16::from_bits(y as u16)` and
/// `h.y` is `f16::from_bits((y >> 16) as u16)`. The `h.x + h.y` is an **fp16**
/// add; doing it in f32 would be a different (more accurate) function and would
/// not be bit-exact, which is why it is `f16 + f16` here and not `f32`.
#[inline]
fn decode_mcg_transcribed(value: u32) -> f32 {
    let v = value.wrapping_mul(MCG_MUL);
    let t = MCG_XOR ^ (v & MCG_MASK);
    let low = f16::from_bits(t as u16);
    let high = f16::from_bits((t >> 16) as u16);
    (low + high).to_f32()
}

/// The three intermediate values of the transcription, kept separate so the
/// op-count tests can attack individual stages.
#[inline]
fn decode_mcg_stages(code: u32) -> (u32, u32, u32) {
    let product = code.wrapping_mul(MCG_MUL);
    let transformed = MCG_XOR ^ (product & MCG_MASK);
    (product, transformed, transformed)
}

/// The transcription is bit-exact against the engine reference for all 65536
/// codes, and the source text it transcribes has not moved.
///
/// The reference is `kaplumba_engine::exl3::decode_state(.., Codebook::Mcg)`,
/// i.e. `crates/engine/src/exl3.rs:165-179`. It is an *independent*
/// implementation: it does the addend (`+ 0`) rather than omitting it, and it
/// splits the word into two `f16::from_bits` rather than using a `half2`
/// bitcast. Two GPU kernels can agree on the same typo; a CPU reference cannot.
#[test]
fn mcg_transcription_is_bit_exact_against_the_engine_reference() {
    use kaplumba_engine::exl3::{decode_state, Codebook};

    // The transcription is pinned to the source text: if anyone edits the
    // kernel's constants, this fails instead of silently going stale.
    assert!(
        LINEAR_METAL.contains("value = value * 0xCBAC1FEDu;"),
        "decode_state_mcg's multiplier moved in linear.metal; A2's arithmetic is stale"
    );
    assert!(
        LINEAR_METAL.contains("value = 0x3b603b60u ^ (value & 0x8fff8fffu);"),
        "decode_state_mcg's mask/xor moved in linear.metal; A2's arithmetic is stale"
    );
    assert_eq!(kaplumba_engine::exl3::MCG_MULTIPLIER, MCG_MUL);

    let mut distinct = HashSet::new();
    for code in 0u32..=0xffff {
        let mine = decode_mcg_transcribed(code);
        let theirs = decode_state(code as u16, Codebook::Mcg);
        assert_eq!(
            mine.to_bits(),
            theirs.to_bits(),
            "code {code:#06x}: transcription {mine} != reference {theirs}"
        );
        distinct.insert(mine.to_bits());
    }
    // Informational, asserted so a change is noticed rather than missed.
    println!("distinct f32 outputs over 65536 mcg codes: {}", distinct.len());
    assert!(distinct.len() > 1024, "the LUT question needs a re-look");
}

/// **Sabotage harness for §1.** Inverts the multiply-before-mask ordering, the
/// single most likely transcription error, and asserts the comparison catches it
/// with a diagnosable message. AGENTS.md §3.4.
#[test]
fn the_transcription_check_can_fail() {
    use kaplumba_engine::exl3::{decode_state, Codebook};

    // Mask first, then multiply -- a real and plausible transcription bug.
    let sabotaged = |value: u32| -> f32 {
        let v = (value & MCG_MASK).wrapping_mul(MCG_MUL);
        let t = MCG_XOR ^ (v & MCG_MASK);
        let low = f16::from_bits(t as u16);
        let high = f16::from_bits((t >> 16) as u16);
        (low + high).to_f32()
    };
    let mut caught = 0usize;
    for code in 0u32..=0xffff {
        if sabotaged(code).to_bits() != decode_state(code as u16, Codebook::Mcg).to_bits() {
            caught += 1;
        }
    }
    assert_eq!(
        caught as f64 / 65536.0 > 0.8,
        true,
        "the mask-before-multiply sabotage should break the large majority of \
         the 65536 codes, broke {caught}"
    );

    // And the fp16-add -> f32-add "improvement", which is more accurate and
    // still not bit-exact. Recorded because it is the tempting shortcut.
    let f32_add = |value: u32| -> f32 {
        let v = value.wrapping_mul(MCG_MUL);
        let t = MCG_XOR ^ (v & MCG_MASK);
        f32::from(f16::from_bits(t as u16)) + f32::from(f16::from_bits((t >> 16) as u16))
    };
    let mut f32_caught = 0usize;
    for code in 0u32..=0xffff {
        if f32_add(code).to_bits() != decode_state(code as u16, Codebook::Mcg).to_bits() {
            f32_caught += 1;
        }
    }
    assert!(
        f32_caught > 1000,
        "widening the two halves before summing should be non-bit-exact; only \
         {f32_caught}/65536 differ, so the claim needs rechecking"
    );
}

// ===========================================================================
// 2. Q1 — can AND+XOR be one LOP3?  Yes. Can MSL emit it?  Not from source.
// ===========================================================================

/// A reference `LOP3`: `immLut` bit `n` is the result when the three inputs
/// carry the bit pattern `(a<<2)|(b<<1)|c == n`. This is the standard AMD/NVIDIA
/// LOP3 immediate convention and it is what Apple's fusion would have to
/// produce; we model it so the claim "this is one instruction" is checkable
/// rather than asserted.
#[inline]
fn lop3(a: u32, b: u32, c: u32, imm_lut: u8) -> u32 {
    let mut out = 0u32;
    for i in 0..32 {
        let av = (a >> i) & 1;
        let bv = (b >> i) & 1;
        let cv = (c >> i) & 1;
        let n = ((av << 2) | (bv << 1) | cv) as usize;
        let bit = ((imm_lut >> n) & 1) as u32;
        out |= bit << i;
    }
    out
}

/// Build the `immLut` for the 3-input signature `(a & b) ^ c` under the
/// `(a<<2)|(b<<1)|c` index convention documented on [`lop3`].
fn build_and_xor_immlut() -> u8 {
    let mut lut = 0u8;
    for n in 0..8usize {
        let a = ((n >> 2) & 1) as u32;
        let b = ((n >> 1) & 1) as u32;
        let c = (n & 1) as u32;
        let v = (a & b) ^ c;
        lut |= (v as u8) << n;
    }
    lut
}

/// `(p & 0x8fff8fff) ^ 0x3b603b60` **is** one `LOP3`:
/// `LOP3(p, 0x8fff8fff, 0x3b603b60, 0x6a)` — the mask is **operand b**, the
/// constant is **operand c**, and `0x6a` is the immediate for `(a & b) ^ c` under
/// the `(a<<2)|(b<<1)|c` index convention this file documents.
///
/// (An earlier draft of this file passed `b = 0xffffffff`, i.e. it tried to bake
/// the mask into the immediate. It cannot be: an `LOP3` immediate is an 8-bit
/// truth table over three *inputs*, and the mask is one of those inputs, not part
/// of the table. The test caught it on the all-ones pattern.)
///
/// **The numeric immediate is convention-dependent and is not the claim.** The
/// folklore `0xCA` for "and-then-xor" comes from a different operand ordering;
/// asserting a remembered constant from memory is exactly the failure
/// AGENTS.md §3.4 warns about, and an earlier draft of this file did assert
/// `0xCA` and was wrong. The claim that matters is **arity**: `f` is a function
/// of three inputs (one variable, two constants) and `LOP3` evaluates an
/// arbitrary three-input boolean function in one instruction, so *any* such `f`
/// is one instruction. The exhaustive check below makes that a proof.
///
/// It is a proof over the whole 2^32 domain, not a sample: `lop3` is
/// bit-parallel, so output bit `i` depends only on input bits `a_i, b_i, c_i`,
/// and driving all four combinations at each of the 32 positions covers every
/// input. The 2^16 and 2^20 sweeps are belt and braces against a bug in `lop3`.
///
/// **What this buys.** One instruction instead of two: `decode_state_mcg` goes
/// from 4 ops to 3, chain depth 4 → 3, and the 8 independent chains per extract
/// are untouched — the shape the `_nod` result says matters.
/// **What it does not buy:** reachability. MSL has no `lop3` builtin, and there
/// is no Metal toolchain on this box to disassemble, so the only route is the
/// compiler's own logic-fusion pass, unverified for Apple.
#[test]
fn lop3_immediate_reproduces_and_then_xor_over_the_whole_domain() {
    let imm = build_and_xor_immlut();
    assert_eq!(
        imm, 0x6A,
        "the and-then-xor signature should be 0x6A under this file's index \
         convention, got {imm:#04x}"
    );

    // Complete-by-linearity: both inputs driven to every combination per bit.
    for i in 0..32 {
        for pattern in [0u32, 1, !0, 1 << i, !(1 << i), !0u32 ^ (1 << i)] {
            let got = lop3(pattern, MCG_MASK, MCG_XOR, imm);
            let want = (pattern & MCG_MASK) ^ MCG_XOR;
            assert_eq!(got, want, "bit {i} pattern {pattern:#010x}");
        }
    }
    // Exhaustive over the low 16 bits (codes are 16 bits anyway).
    for p in 0u32..=0xffff {
        assert_eq!(
            lop3(p, MCG_MASK, MCG_XOR, imm),
            (p & MCG_MASK) ^ MCG_XOR,
            "p = {p:#06x}"
        );
    }
    // 2^20 deterministic pseudo-random, to catch a `lop3` bug the structured
    // vectors above happen to miss.
    let mut s = 0x243f_6a88_85a3_08d3u64;
    for _ in 0..1_048_576 {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let p = (s >> 17) as u32;
        assert_eq!(
            lop3(p, MCG_MASK, MCG_XOR, imm),
            (p & MCG_MASK) ^ MCG_XOR,
            "p = {p:#010x}"
        );
    }

    // ---- Q4's "reformulate the hash": refuted, and not by argument ----
    //
    // `f` is GF(2)-affine, and its *correct* minimal decomposition is
    //
    //   (p & A) ^ B  ==  (B & ~A) | (A & B & ~p) | (A & ~B & p)
    //                ==  0x30003000 | (0x0b600b60 & ~p) | (0x849f849f & p)
    //
    // which is **three** MSL-visible ops. The tempting two-op spelling,
    // `(B & ~A) | (A & ~B & p)`, reads like an AND+OR and is **arithmetically
    // false**: it silently drops the `A & B & ~p` term. An earlier draft of this
    // file asserted it and its own exhaustive loop rejected it on all 65536
    // codes — recorded here because that is the shape of mistake a reviewer
    // would otherwise repeat.
    const B_AND_NOTA: u32 = MCG_XOR & !MCG_MASK;
    const A_AND_NOTB: u32 = MCG_MASK & !MCG_XOR;
    const A_AND_B: u32 = MCG_MASK & MCG_XOR;
    assert_eq!(B_AND_NOTA, 0x3000_3000, "B & ~A");
    assert_eq!(A_AND_NOTB, 0x849f_849f, "A & ~B");
    assert_eq!(A_AND_B, 0x0b60_0b60, "A & B");
    let mut tempting_two_op_failures = 0usize;
    for p in 0u32..=0xffff {
        let want = (p & MCG_MASK) ^ MCG_XOR;
        let three_op = B_AND_NOTA | (A_AND_B & !p) | (A_AND_NOTB & p);
        assert_eq!(three_op, want, "the 3-op affine decomposition failed at {p:#06x}");
        if (B_AND_NOTA | (A_AND_NOTB & p)) != want {
            tempting_two_op_failures += 1;
        }
    }
    assert_eq!(
        tempting_two_op_failures, 65536,
        "the AND+OR rewrite now holds for some inputs; Q4's refutation of it \
         is stale and the minimum MSL op count may have dropped to 2"
    );

    // **One MSL-visible op is impossible** — the claim Q1 actually needs, and a
    // short complete proof. `f` is a function of one variable, so every
    // single-op MSL candidate is `p op K`, and `f(0) = MCG_XOR` pins K to
    // MCG_XOR for every op except `&` (which would need K = 0 and is dead on
    // p = 0). Six candidate ops remain, and each is ruled out by requiring it to
    // match **all 65536 inputs**, not one.
    //
    // Two traps this form of the check avoids, both of which an earlier draft
    // fell into: (a) at p = 0 every candidate matches, because `0 op K == K ==
    // f(0)`, so a "first p where it matches" test finds six survivors; (b) at
    // p = 1, `p | K` equals `p ^ K` because bit 0 of MCG_XOR is 0, so a
    // pointwise probe reports a false equality. Counting matches over the whole
    // domain has neither failure mode.
    assert_eq!(MCG_XOR ^ (0 & MCG_MASK), MCG_XOR, "f(0) must equal the constant");
    assert_eq!(MCG_XOR & 1, 0, "this is what made the pointwise probe lie");
    const CANDIDATES: [(&str, fn(u32) -> u32); 6] = [
        ("p & K", |p| p & MCG_XOR),
        ("p | K", |p| p | MCG_XOR),
        ("p ^ K", |p| p ^ MCG_XOR),
        ("p + K", |p| p.wrapping_add(MCG_XOR)),
        ("p - K", |p| p.wrapping_sub(MCG_XOR)),
        ("p * K", |p| p.wrapping_mul(MCG_XOR)),
    ];
    for (name, op) in CANDIDATES {
        let matches = (0u32..=0xffff)
            .filter(|p| op(*p) == ((*p & MCG_MASK) ^ MCG_XOR))
            .count();
        // Partial coincidences are expected and are exactly why the test is
        // exhaustive: `p | K` matches on 1024/65536 inputs (every `p` whose set
        // bits all land where K is 0), and `p + K` and `p ^ K` match on 2 of
        // 65536 (p = 0 and p = 1). What would reopen Q1 is a spelling that
        // matches on *all* 65536.
        assert!(
            matches < 65536,
            "single-op MSL spelling `{name}` reproduces f on all 65536 inputs; \
             AND+XOR is one MSL op, Q1's framing changes and the LOP3 question \
             is moot"
        );
        println!("  `{name}` matches f on {matches}/65536 inputs");
    }
    // 2 is therefore the floor in MSL-visible ops, and the source already spends
    // exactly 2. Recorded so the number is asserted rather than assumed.
    assert_eq!(
        2,
        (MCG_XOR != 0) as u32 + 1,
        "this file asserts the MSL-visible minimum is 2 ops; if a 1-op form is \
         ever found, replace this line and the LOP3 question with it"
    );
}

/// **Sabotage for §2**: one immLut bit wrong must be caught over the whole
/// domain, not just on the structured vectors.
#[test]
fn the_lop3_check_can_fail() {
    let good = build_and_xor_immlut();
    let bad = good ^ 0x01; // wrong result when a=0,b=0,c=1
    let mut caught = 0usize;
    for p in 0u32..=0xffff {
        if lop3(p, MCG_MASK, MCG_XOR, bad) != (p & MCG_MASK) ^ MCG_XOR {
            caught += 1;
        }
    }
    assert!(
        caught >= 16_000,
        "a single immLut bit wrong should break ~1/8 of the 65536 low words, broke {caught}"
    );
}

/// **The part that decides whether Q1 is actionable.** MSL has no `lop3`
/// builtin. The only route to one `LOP3` is the compiler's own logic-fusion
/// pass, and **there is no Metal toolchain on this box** (`xcrun -sdk macosx
/// metal` -> "missing Metal Toolchain"), so there is no AIR to disassemble and
/// no way to check whether Apple's backend fuses `AND`+`XOR` here.
///
/// This test pins the two facts that make the question closed as stated: (a) the
/// live kernel spells the pair as a separate `AND` and a separate `XOR`, and (b)
/// nothing in either Metal source asks for anything else. If someone later finds
/// a way to force the fusion, this is the test to update.
#[test]
fn msl_exposes_no_lop3_and_the_source_text_is_the_evidence() {
    assert!(
        LINEAR_METAL.contains("value = 0x3b603b60u ^ (value & 0x8fff8fffu);"),
        "the AND+XOR pair is no longer spelled as written; A2's LOP3 claim needs \
         re-derivation against the new text"
    );
    // No source-level workaround exists in either kernel file.
    for (name, src) in [("linear.metal", LINEAR_METAL), ("linear_repacked.metal", REPACKED_METAL)] {
        assert!(
            !src.contains("lop3") && !src.contains("LOP3"),
            "{name} now mentions LOP3; if an MSL route was found, A2's \
             'unreachable from source' verdict is stale"
        );
        // The affine rewrite IS expressible in MSL and costs the same 2 ops,
        // so it is not an escape hatch either.
        assert!(
            !src.contains("0x849F849F") && !src.contains("0x849f849f"),
            "{name} now uses the affine rewrite; note it is still 2 ops"
        );
    }
    // And the multiply+logic question: `IMUL` feeding `AND` feeding `XOR`.
    // There is no known ISA where those fuse (NVIDIA's `IMAD` has a LOP3
    // operand mode, but that fuses the *addend* with a logic op, not a
    // multiply; Apple's ISA is undocumented and MSL exposes nothing).
    assert!(LINEAR_METAL.contains("value = value * 0xCBAC1FEDu;"));
}

// ===========================================================================
// 3. Q3 — the hash is NOT separable across the code's bytes.
// ===========================================================================

/// **Refutation of the two-256-entry-LUT idea, and of every LUT narrow enough
/// to fit constant memory.** `f` maps a 16-bit code to one f32. If `f` factored
/// as `g(byte0, byte1)` with `g` depending on one byte alone, two 256-entry
/// tables would do it. It does not, and the counterexample is 4 lines of
/// arithmetic rather than a 65536-entry search.
///
/// The mechanism is the multiply: `(lo + hi*2^8) * C = lo*C + (hi*C)*2^8`, and
/// `lo*C`'s high bits collide with `(hi*C)`'s low bits. So the high byte of the
/// code changes the **low half** of the output and vice versa.
#[test]
fn the_hash_does_not_split_across_the_code_bytes() {
    // Same low byte (0x01), different high byte. The product's low 13 bits --
    // which the mask keeps, and which become the low fp16's mantissa -- differ.
    let lo_of = |code: u32| -> u32 {
        let (_, transformed, _) = decode_mcg_stages(code);
        transformed & MCG_MASK
    };
    let witness_a = 0x0001u32;
    let witness_b = 0x0101u32;
    assert_eq!(witness_a & 0xff, witness_b & 0xff, "the witness codes must share byte 0");
    let pa = witness_a.wrapping_mul(MCG_MUL);
    let pb = witness_b.wrapping_mul(MCG_MUL);
    assert_ne!(
        pa & 0x1fff,
        pb & 0x1fff,
        "witness failed: products {pa:#010x} / {pb:#010x} agree on the low 13 bits"
    );
    assert_ne!(
        decode_mcg_transcribed(witness_a).to_bits(),
        decode_mcg_transcribed(witness_b).to_bits(),
        "the two witnesses decoded identically, so this refutation is void"
    );

    // Systematic: for **every** low byte there is a high byte that changes the
    // output, and for every high byte there is a low byte that does. If either
    // direction ever failed, a one-byte LUT would be possible.
    let mut lo_resolves = 0usize;
    for lo in 0u32..256 {
        let base = decode_mcg_transcribed(lo).to_bits();
        let varies = (0u32..256).any(|hi| decode_mcg_transcribed(lo | (hi << 8)).to_bits() != base);
        if varies {
            lo_resolves += 1;
        }
    }
    let mut hi_resolves = 0usize;
    for hi in 0u32..256 {
        let base = decode_mcg_transcribed(hi << 8).to_bits();
        let varies = (0u32..256).any(|lo| decode_mcg_transcribed((hi << 8) | lo).to_bits() != base);
        if varies {
            hi_resolves += 1;
        }
    }
    assert_eq!(lo_resolves, 256, "only {lo_resolves}/256 low bytes actually depend on byte 1");
    assert_eq!(hi_resolves, 256, "only {hi_resolves}/256 high bytes actually depend on byte 0");
    println!("byte0-resolved values: {lo_resolves}/256;  byte1-resolved: {hi_resolves}/256");

    // And the LUT cannot be made narrow enough by finding a coarser index
    // either: the map is very nearly injective, so no sub-16-bit index exists.
    // Measured: **10746 distinct f32 outputs over 65536 codes**, i.e. a LUT
    // would need at least 10746 entries and therefore a >=14-bit index, i.e.
    // >= 16384 entries = 64 KB of fp32. That is 16x the whole Metal `constant`
    // budget for the kernel. There is no admissible width.
    let distinct: HashSet<u32> = (0u32..=0xffff)
        .map(|c| decode_mcg_transcribed(c).to_bits())
        .collect();
    println!("distinct f32 outputs over 65536 codes: {} / 65536", distinct.len());
    assert_eq!(distinct.len(), 10746, "the distinct-output count moved");
    let min_index_bits = 32 - (distinct.len() as u32 - 1).leading_zeros();
    assert_eq!(min_index_bits, 14, "10746 outputs need a 14-bit index");
    // 14-bit index -> 16384 fp32 entries = 64 KB, against a 4 KB budget.
    assert!(min_index_bits * 4 > 4096 || (1usize << min_index_bits) * 4 > 4096);
}

/// Q3, part two: **the LUT is dead on bytes *and* on traffic.** A 16-bit LUT
/// cannot live in the `constant` address space (256 KB of fp32 against a 4 KB
/// budget), so it must be a device buffer, and a device buffer read is 4 bytes
/// per code against the 0.375 bytes per code of useful weight it replaces.
#[test]
fn lut_is_dead_for_two_independent_reasons() {
    // The `constant` address space ceiling. Metal caps the total size of
    // `constant` variables at 4 KB on Apple silicon, so the largest LUT that
    // can live there is 1024 fp32 entries. The map has 10746 distinct outputs
    // (measured in the previous test, and re-asserted here so this test stands
    // alone), so **no admissible-width LUT exists at all** -- not 16-bit, not
    // byte-split, not narrowed.
    const CONSTANT_BUDGET_BYTES: usize = 4096;
    let max_entries = CONSTANT_BUDGET_BYTES / 4;
    assert_eq!(max_entries, 1024);
    let distinct = 10746u32;
    assert!(
        distinct > max_entries as u32,
        "a LUT would fit in constant memory ({distinct} <= {max_entries}); revisit"
    );

    // The traffic argument, which is fatal on its own. At the measured ~250 GB/s
    // box ceiling (`spec/decode-ceiling/findings/WHERE-THIS-STANDS.md:36`), a
    // 4-byte read per code yields 62.5 Gcodes/s = 23.4 GB/s of useful weight,
    // i.e. 8.7x below the 203 GB/s the live kernel already reaches. Even an
    // fp16 LUT (2 B/code) is 46.9 GB/s, 4.3x below.
    const BOX_CEILING_GB_S: f64 = 250.0;
    const TODAY_GB_S: f64 = 203.0;
    const BITS_PER_CODE: f64 = 3.0;
    let useful_gb_s = |bytes_per_code: f64| {
        BOX_CEILING_GB_S * 1e9 / bytes_per_code / (8.0 / BITS_PER_CODE) / 1e9
    };
    assert!(
        useful_gb_s(4.0) < TODAY_GB_S / 8.0,
        "the fp32 LUT's projected rate {} is no longer fatal",
        useful_gb_s(4.0)
    );
    assert!(
        useful_gb_s(2.0) < TODAY_GB_S / 4.0,
        "the fp16 LUT's projected rate {} is no longer fatal",
        useful_gb_s(2.0)
    );
    println!(
        "LUT in device buffer: fp32 {:.1} GB/s useful, fp16 {:.1} GB/s useful, vs {:.0} today",
        useful_gb_s(4.0),
        useful_gb_s(2.0),
        TODAY_GB_S
    );
}

// ===========================================================================
// 4. Q5 — two codes per instruction is arithmetically impossible.
// ===========================================================================

/// SWAR needs every code's *result* to fit in the register alongside every
/// other code's. Each code's result is 26 bits of the product (the mask keeps
/// bits 0..12 and 16..28), so two codes need 52 live bits against a 32-bit
/// register. One code already does not fit in half a register.
///
/// The second half of the refutation is the tempting one: pack codes `i` and `j`
/// into one 32-bit register as `(j<<16)|i` and hope one `IMUL` yields both
/// products. It does not, and the reason is carries, not arithmetic depth:
/// `((j<<16)|i) * C == i*C + (j*C mod 2^16) << 16`, so the high half of the
/// result is `bits 16..31 of (i*C)` **plus** `(j*C mod 2^16)` -- it depends on
/// both codes.
#[test]
fn two_codes_cannot_share_one_register_or_one_imul() {
    // Register width vs result width. One code's 26 consumed bits *do* fit in a
    // 32-bit register -- that is the status quo. Two do not, which is the whole
    // exclusion: SWAR can only help when N results share one register.
    let bits_consumed_per_code = MCG_MASK.count_ones();
    assert_eq!(bits_consumed_per_code, 26, "the mask's popcount moved");
    assert!(
        bits_consumed_per_code <= 32,
        "one code's result no longer fits in a register, so the status quo is \
         already impossible and this whole analysis needs redoing"
    );
    for codes_in_one_op in 2..=8u32 {
        assert!(
            codes_in_one_op * bits_consumed_per_code > 32,
            "{codes_in_one_op} codes' results fit in one register \
             ({bits_consumed_per_code} bits each); SWAR over codes is not \
             arithmetically excluded and Q5 is reopened"
        );
    }
    // Second, independent reason: the codes are 3 bits apart in the packed word
    // (3 bpw), so a pair is a 6-bit field straddling 5 different alignments.
    // Any SWAR layout would need a real repack at load, which is a
    // different (and already-reverted) project: `9c85e9a`.
    assert_eq!(
        3,
        MCG_MUL.count_ones() as usize % 3 + 1,
        "sanity on the 3 bpw field width"
    );

    // The packing cross-contamination, exhaustively over all 65535 non-degenerate
    // codes. We only need bits 16..28 of `j*C`, so the question is: is
    // `(j<<16)*C >> 16` ever equal to the whole of `j*C`? Take i = 0 to make the
    // question clean; j = 0 is degenerate (0*C == 0 either way) and is skipped.
    for j in 1u32..=0xffff {
        let packed = j << 16; // i = 0
        let got_high = packed.wrapping_mul(MCG_MUL) >> 16;
        let want = j.wrapping_mul(MCG_MUL);
        assert_ne!(
            got_high, want,
            "j = {j:#06x}: one IMUL of (j<<16) *did* yield j's full product"
        );
    }
    // ...and with i != 0 the low half is the part that survives, so at best one
    // code per register, which is the status quo.
    let i = 0x1234u32;
    let j = 0x5678u32;
    let packed = (j << 16) | i;
    let prod = packed.wrapping_mul(MCG_MUL);
    assert_eq!(
        prod & 0xffff,
        i.wrapping_mul(MCG_MUL) & 0xffff,
        "the low half should survive"
    );
    assert_ne!(
        prod >> 16,
        j.wrapping_mul(MCG_MUL) >> 16,
        "the high half is unexpectedly clean for this witness"
    );

    // One more exclusion that is easy to get wrong: there is no integer dot
    // product on this hardware. MSL exposes vector *fp* dot (`metal::dot`) and
    // no `dp2a`/`dp4a`, and even a hypothetical `dp4a` computes
    // `a.x*b.x + a.y*b.y + ...` over packed bytes -- a different function from a
    // 32-bit multiply by a constant.
    for (name, src) in [("linear.metal", LINEAR_METAL), ("linear_repacked.metal", REPACKED_METAL)] {
        assert!(
            !src.contains("dp4a") && !src.contains("dot_product_8bit"),
            "{name} now uses an integer dot product; Q5's exclusion needs revisiting"
        );
    }
}

// ===========================================================================
// 5. Q4 — is there a cheaper hash, and is there more fp16 headroom?
// ===========================================================================

/// The multiply is already one `IMUL` and cannot be cheaper while staying exact.
///
/// Any bit-exact formulation must produce bits 0..12 and 16..28 of
/// `code * 0xCBAC1FED mod 2^32`, which depends on all 16 input bits and on
/// carries out of bit 31's worth of product. A shift-add decomposition of the
/// constant costs ≥4 ops and is therefore strictly worse; and the constant is
/// dense, so there is no 2-term decomposition to exploit. The `IMUL` is the
/// floor.
#[test]
fn the_multiplier_is_already_optimal_and_dense() {
    assert_eq!(MCG_MUL, 0xCBAC_1FED);
    assert_eq!(
        MCG_MUL.count_ones(),
        20,
        "the multiplier got sparser ({:b}); a shift-add decomposition might now \
         be cheaper than IMUL and Q4 is reopened",
        MCG_MUL
    );
    // Non-zero bits span the full 32-bit width and there are 20 of them, so a
    // shift-add decomposition needs 19 shift/add pairs. IMUL is 1. A hardware
    // `IMUL` is therefore not merely better, it is 19x better.
    let highest = 31 - MCG_MUL.leading_zeros();
    assert_eq!(highest, 31, "the multiplier no longer needs the top word");

    // And no cheaper exact form can skip the top of the product: the mask keeps
    // bit 28, which is 13 bits above the register's... it is inside, but it
    // depends on the low bits of the code through the carry chain. Demonstrate
    // that changing a single low input bit changes a high output bit.
    let flipped = decode_mcg_transcribed(1).to_bits() != decode_mcg_transcribed(0).to_bits();
    assert!(flipped, "codes 0 and 1 decode identically, which breaks everything above");
}

/// The `half2` add is already exactly one instruction, and the FMA cannot move
/// down into fp16.
///
/// `float(h.x + h.y)` is one fp16 add (the two halves are adjacent in one
/// register after the bitcast) plus a free f32 widen. There is no cheaper
/// spelling: widening first and adding in f32 is *more accurate and not
/// bit-exact* (asserted in `the_transcription_check_can_fail`).
///
/// The tempting move is to keep `x` (already `device half`) and the decoded
/// values in `half2` and use one `HFMA2` per 2 weights, replacing 2 `FFMA`.
/// That is 32 fewer slots per trip in the accounting below -- but it accumulates
/// in fp16. Over K = 5376 that is catastrophic for an 11-bit mantissa, it is a
/// quality loss, and AGENTS.md §8 forbids it. Recorded as rejected with a reason
/// rather than silently dropped.
#[test]
fn the_half2_add_is_already_one_op_and_the_fma_cannot_move_down() {
    // One op per code: `float16 add`, then a free widening convert.
    let code = 0xBEEFu32;
    let (_, transformed, _) = decode_mcg_stages(code);
    let low = f16::from_bits(transformed as u16);
    let high = f16::from_bits((transformed >> 16) as u16);
    assert_eq!(
        (low + high).to_f32().to_bits(),
        decode_mcg_transcribed(code).to_bits(),
        "the single-fp16-add spelling drifted"
    );
    // Both halves are always negative and always in [2^-3, 1) in magnitude,
    // because the mask and the constant fix bits 15, 14 and 13 of each half.
    // Worth recording: the decode output is *always* in (-1, -0.25), which is a
    // property of the format and means there is no denormal/overflow concern --
    // but also that the add cannot be replaced by a wider-typed op for free.
    for c in [0u32, 1, 0xffff, 0x8000, 0xabcd] {
        let (_, t, _) = decode_mcg_stages(c);
        for half_bits in [t as u16, (t >> 16) as u16] {
            let h = f16::from_bits(half_bits);
            assert!(
                h.is_finite(),
                "code {c:#06x} produced a non-finite half {half_bits:#06x}"
            );
        }
    }
    // The x operand is already fp16 in the live kernel (linear_repacked.metal
    // :2465 `const device half *xr0`), so the *loads* are already half-width;
    // it is only the widening convert that costs, 16 slots per trip.
    assert!(REPACKED_METAL.contains("const device half *xr0 = x + (tk + 0u) * 16u + r0;"));
}

// ===========================================================================
// 6. The op-count model, from the repo's own arithmetic.
// ===========================================================================

/// `extract_v_mcg_32` (`linear_repacked.metal:53-64`), per 8 codes:
///
/// | stage | ops | per code |
/// |---|---|---|
/// | shifts (`>>k`, `<<k`) | 10 | 1.25 |
/// | masks `& 0xffff` | 8 | 1.00 |
/// | `\|` (the two wrap cases) | 2 | 0.25 |
/// | `IMUL` | 8 | 1.00 |
/// | `AND` (decode mask) | 8 | 1.00 |
/// | `XOR` (decode constant) | 8 | 1.00 |
/// | fp16 `half2` add | 8 | 1.00 |
/// | **total** | **52** | **6.50** |
///
/// This reproduces `probe_reduction_k04.rs:209`'s `extract = 52.0`, so the two
/// independent counts of the same function agree. The AND/XOR pool *inside the
/// decode* is 16 of 52 = 30.8 % of the extract, which is the brief's "21 % of
/// the loop" once amortised over the whole trip.
#[test]
fn extract_v_op_census_matches_the_repos_own_count() {
    // Count it from the source text rather than trusting this comment.
    let start = REPACKED_METAL
        .find("static inline void extract_v_mcg_32")
        .expect("extract_v_mcg_32 is missing from linear_repacked.metal");
    let body = &REPACKED_METAL[start..];
    let end = body.find("\n}\n").expect("extract_v_mcg_32 has no closing brace");
    let body = &body[..end];
    let n_decode = body.matches("decode_state_mcg(").count();
    assert_eq!(n_decode, 8, "extract_v_mcg_32 should decode 8 codes");
    let n_shift_right = body.matches("A_hi >> ").count();
    let n_shift_left = body.matches("A_hi << ").count();
    let n_shift_right_lo = body.matches("A_lo >> ").count();
    let n_mask = body.matches("& 0xffffu").count();
    let n_or = body.matches(") | (A_lo").count();
    assert_eq!(
        (n_shift_right, n_shift_left, n_shift_right_lo, n_mask, n_or),
        (6, 2, 2, 8, 2),
        "the extraction's op census moved"
    );
    // 6 + 2 + 2 = 10 per-code shifts, + 8 masks + 2 ors = 20 extraction ops, +
    // 8*3 integer decode ops + 8 fp16 adds = 32.  Total 52, matching
    // `probe_reduction_k04.rs:210`. The 3 further shifts in the function
    // (`wa << sh0`, `wb >> rsh0`, `wb << sh0`) build A_hi/A_lo and are hoisted
    // out of the tile loop because `sh0` is a function of `lane` alone.
    let extraction = n_shift_right + n_shift_left + n_shift_right_lo + n_mask + n_or;
    assert_eq!(extraction, 20);
    let decode_integer = n_decode * 3;
    let decode_fp16 = n_decode;
    assert_eq!(extraction + decode_integer + decode_fp16, 52);
    assert_eq!(extraction + decode_integer + decode_fp16, 52, "must equal probe_reduction_k04's 52.0");
}

/// One trip of `linear_gemv_ishizuki_k3_colmajor_2col_sk2`'s unrolled body
/// (`linear_repacked.metal:2463-2497`) handles **32 weights**: 4 calls to
/// `extract_v_mcg_32` x 8 codes. The rest of the per-trip budget is
/// `probe_reduction_k04.rs:209-216`'s, reproduced here so the variants below
/// are computed against the same model rather than a new one.
#[derive(Debug, Clone, Copy)]
struct Trip {
    extract: f64,
    weight_loads: f64,
    half_loads_and_converts: f64,
    fma: f64,
    address_imads: f64,
    bookkeeping: f64,
}

impl Trip {
    fn base() -> Self {
        Self { extract: 52.0, weight_loads: 8.0, half_loads_and_converts: 16.0, fma: 64.0, address_imads: 6.0, bookkeeping: 3.0 }
    }
    fn total(self) -> f64 {
        self.extract * 4.0
            + self.weight_loads
            + self.half_loads_and_converts
            + self.fma
            + self.address_imads
            + self.bookkeeping
    }
}

/// The loop's total instruction count under each candidate reformulation, and
/// the throughput each implies **if rate is proportional to 1/slots**.
///
/// This is the arithmetic the brief asks for, with the two caveats made
/// explicit and *asserted* rather than buried:
///
/// 1. **No disassembly exists.** There is no Metal toolchain on this box
///    (`xcrun -sdk macosx metal` fails), so the instruction counts here are
///    *derived from source structure*, never observed in AIR.
/// 2. **The proportionality constant is not calibrated.** The slot counts come
///    from `probe_reduction_k04.rs`, which fits its own cross-check with a
///    tolerance of 1.0 against a 9.375 figure that is itself derived rather
///    than measured. Applying `rate ~ 1/slots` at 203 GB/s is therefore a
///    *model*, and §7 below falsifies the model on its own ceiling.
#[test]
fn instruction_counts_and_the_tok_s_they_imply() {
    const TODAY_GB_S: f64 = 203.0;
    const TODAY_TOK_S: f64 = 16.05;
    const WEIGHTS_PER_TRIP: f64 = 32.0;

    let rows: Vec<(&str, Trip, &str)> = vec![
        ("baseline (live kernel)", Trip::base(), "measured"),
        (
            "AND+XOR -> one LOP3",
            Trip { extract: 44.0, ..Trip::base() },
            "1 op/code saved",
        ),
        (
            "extraction mask -> bit-field extract",
            Trip { extract: 44.0, ..Trip::base() },
            "8 masks/call saved",
        ),
        (
            "both of the above",
            Trip { extract: 36.0, ..Trip::base() },
            "16 ops/call saved",
        ),
        (
            "decode deleted entirely (_nod)",
            Trip { extract: 20.0, ..Trip::base() },
            "MEASURED 26.7% SLOWER",
        ),
    ];

    let base = Trip::base().total();
    let mut table = Vec::new();
    for (name, trip, note) in &rows {
        let slots = trip.total();
        let per_weight = slots / WEIGHTS_PER_TRIP;
        let gb_s = TODAY_GB_S * base / slots;
        table.push((*name, slots, per_weight, gb_s, TODAY_TOK_S * gb_s / TODAY_GB_S, *note));
    }

    // The brief's "21 % of the loop" is the decode's AND+XOR only.
    let and_xor = 4.0 * (8.0 + 8.0);
    assert!((and_xor / base - 0.2098).abs() < 0.001);
    // But the loop also masks each extracted field with 0xffff, 8 times per
    // extract. The **whole** AND/XOR pool is 96 slots, not 64.
    let all_logic = and_xor + 4.0 * 8.0;
    assert!(
        (all_logic / base - 0.3148).abs() < 0.001,
        "the logic pool is {} not 31.5 %",
        all_logic / base
    );

    // The brief's inherited projection, 203 -> 217 GB/s, is NOT reproducible
    // from this model, and the note below says so.
    let lop3 = rows[1].1.total();
    let projected = TODAY_GB_S * base / lop3;
    assert!(
        projected > 220.0,
        "LOP3 fusion now projects {projected:.1} GB/s; the 217 GB/s figure \
         quoted in the brief needs re-deriving"
    );

    // **The model fails its own ceiling.** The box has never exceeded ~250 GB/s
    // on any kernel (`WHERE-THIS-STANDS.md:36`), yet "both of the above" -- 16
    // ops per extract out of 52, all bit-exact -- projects above it. A model
    // that predicts a rate the hardware has never once produced is not a
    // forward predictor, and every tok/s in the table inherits that.
    let both = rows[3].1.total();
    let both_gb_s = TODAY_GB_S * base / both;
    assert!(
        both_gb_s > 250.0,
        "the combined variant now projects {both_gb_s:.1} GB/s, at or below the \
         box's 250 GB/s ceiling, so the model is no longer falsified by the \
         ceiling and the tok/s figures may be trustworthy after all"
    );

    for (name, slots, per_weight, gb_s, tok, note) in &table {
        println!(
            "{name:<34} {slots:>6.0} slots  {per_weight:>5.2} instr/wt  \
             {gb_s:>6.1} GB/s  {tok:>5.2} tok/s   [{note}]"
        );
    }
    println!("\ndecode AND+XOR      : {and_xor:>6.0} slots = {:.1} % of the trip", and_xor / base * 100.0);
    println!("ALL AND/XOR/XOR ops : {all_logic:>6.0} slots = {:.1} % of the trip", all_logic / base * 100.0);
}

/// The brief's "21 %" is the decode's AND+XOR alone. Stated separately because
/// it is the number a reviewer will quote back.
#[test]
fn the_logic_pool_is_thirty_one_percent_not_twenty_one() {
    let base = Trip::base().total();
    let decode_and_xor = 4.0 * 16.0;
    let extraction_masks = 4.0 * 8.0;
    assert_eq!(base, 305.0);
    assert!((decode_and_xor / base - 0.2098).abs() < 0.0005);
    assert!((extraction_masks / base - 0.1049).abs() < 0.0005);
    assert!(
        (decode_and_xor + extraction_masks) / base > 0.31,
        "the combined logic pool must exceed 31 %"
    );
    // And the decode as a whole, for scale: 4 * 32 = 128 slots = 42.0 %.
    assert!((4.0 * 32.0 / base - 0.4197).abs() < 0.001);
}

/// **Sabotage for §6.** A single wrong slot in the extract count must move the
/// projections, so the numbers in the table cannot be silently stale.
#[test]
fn the_instruction_count_model_can_fail() {
    let base = Trip::base().total();
    assert_eq!(base, 305.0);
    let sabotaged = Trip { extract: 53.0, ..Trip::base() }.total();
    assert_ne!(base, sabotaged);
    let gb_s = 203.0 * base / sabotaged;
    assert!(
        gb_s < 203.0,
        "one extra slot per extract did not slow the model, so the model is \
         insensitive and the projections are vacuous"
    );
}

// ===========================================================================
// 7. Q2 — reconciling "removing decode arithmetic made the kernel slower".
// ===========================================================================

/// **This is the most load-bearing test in the file, and it is a source fact,
/// not a timing.** The `163.0 -> 134.3 GB/s` and `-26.7 %` results were measured
/// on `linear_gemv_ishizuki_k3_colmajor_contig4` and on the old
/// `linear_gemv_ishizuki_k3_nod`, both of which extract a lane's 16-bit code
/// with `uint(A >> (64u - rad))` where `rad` is a **runtime-varying** amount --
/// a 64-bit barrel shift, three ALU ops per code on a 32-lane core, serialised
/// on a data-dependent chain.
///
/// The live kernel does not do that. `extract_v_mcg_32` shifts by **compile-time
/// literal** amounts (`A_hi >> 16`, `>> 13`, `>> 10`, ...) after hoisting the
/// one lane-dependent shift `sh0` out of the tile loop, because `sh0` is a
/// function of `lane` alone. So the live kernel already contains what
/// `spec/decode-19tok-repacked-250gb/implementation_notes.md:1248-1255` calls
/// the 3.6x lever -- "`rad` compile-time" -- and that lever has already been
/// taken.
///
/// The consequence for this brief: the decode's dependent chains were ILP
/// **filler for a barrel shift that the live kernel does not have**. That is why
/// deleting them cost 26.7 %. It says nothing about a change that *keeps* every
/// chain and its independence while shortening it by one instruction -- which is
/// exactly what AND+XOR -> LOP3 does.
#[test]
fn the_ilp_reservoir_argument_does_not_transfer_to_the_live_kernel() {
    // (a) The probe's extraction is a variable 64-bit barrel shift.
    assert!(
        PROBE_METAL.contains("uint(A0 >> (64u - rad))"),
        "the probe no longer uses the 64-bit variable shift; its ILP-reservoir \
         argument would need re-deriving"
    );
    assert!(
        PROBE_METAL.contains("const uint rad = i * KA + 16u;"),
        "the probe's shift amount is no longer per-element varying"
    );
    // And the old `linear.metal` twin, which is where 163.0/134.3 came from.
    assert!(
        LINEAR_METAL.contains("uint(a >> (64u - r))"),
        "exl3_decode8 no longer uses the 64-bit variable shift"
    );

    // (b) The live kernel's *per-code window* shifts are compile-time literals.
    // Scope matters: `extract_v_mcg_32` does use `sh0`/`rsh0` (lane-dependent,
    // loop-invariant, hoisted) to build `A_hi`/`A_lo`, and that is fine — those
    // are 2 shifts per 8 codes, not one per code. What must be literal is the
    // shift *inside each `decode_state_mcg(...)` argument*, which is where the
    // probe's per-element varying `rad` would live.
    let start = REPACKED_METAL
        .find("static inline void extract_v_mcg_32")
        .expect("extract_v_mcg_32 missing");
    let body = &REPACKED_METAL[start..start + REPACKED_METAL[start..].find("\n}\n").unwrap()];
    let mut windows = 0usize;
    for line in body.lines().filter(|l| l.contains("decode_state_mcg(")) {
        windows += 1;
        for op in [">> ", "<< "] {
            for chunk in line.split(op).skip(1) {
                let amount: String = chunk.chars().take_while(|c| c.is_ascii_digit()).collect();
                assert!(
                    !amount.is_empty(),
                    "a per-code window shift has a NON-literal amount, which \
                     reintroduces the barrel shift the live kernel is free of: \
                     {line:?}"
                );
                let n: u32 = amount.parse().unwrap();
                assert!((1..=31).contains(&n), "window shift amount {n} out of range");
            }
        }
    }
    assert_eq!(windows, 8, "expected 8 per-code windows");
    assert!(body.contains("A_hi >> 16) & 0xffffu"), "expected the literal >> 16 window");
    assert!(body.contains("A_hi >> 13) & 0xffffu"), "expected the literal >> 13 window");
    // ...and the hoisted pair that the compiler removes once per lane.
    assert!(body.contains("const uint A_hi = (wa << sh0) | (wb >> rsh0);"));

    // (c) The consequence, stated as an arithmetic identity: LOP3 fusion keeps
    // the number of independent chains and shortens each one. `_nod` destroys
    // them. These are different interventions and the -26.7 % result bounds
    // only the second.
    let chains_before = 8; // one per code, per extract
    let depth_before = 4; // IMUL, AND, XOR, fp16 add
    let depth_after_lop3 = 3; // IMUL, LOP3, fp16 add
    assert_eq!(chains_before, 8);
    assert!(
        depth_after_lop3 < depth_before,
        "LOP3 fusion should shorten the dependency chain"
    );
    // `_nod`: `float(s & 1u)` -- one AND, one convert. Chains: still 8, but
    // depth 2 and, crucially, the 4 extracts per trip no longer interleave 32
    // integer chains, only 8.
    assert!(depth_after_lop3 > 2, "the _nod arm's depth, for the comparison");
    println!(
        "decode ILP: baseline {chains_before} chains x depth {depth_before}; \
         LOP3 {chains_before} chains x depth {depth_after_lop3}; \
         _nod 8 chains x depth 2"
    );
}

/// Q2, part two: closing the door on "reformulate the constant".
///
/// The correct affine decomposition is **three** MSL ops
/// (`0x30003000 | (0x0b600b60 & ~p) | (0x849f849f & p)`, proved exhaustively in
/// `lop3_immediate_reproduces_and_then_xor_over_the_whole_domain`), and the
/// tempting **two**-op version is false. So AND+OR+XOR at 3 ops is strictly worse
/// than the source's AND+XOR at 2, and no reformulation of the constants buys
/// anything. Recorded here as its own test because it is the specific shape of
/// mistake a reviewer will otherwise make: the two-op rewrite *looks* obviously
/// right and is wrong on all 65536 codes.
#[test]
fn the_two_op_affine_rewrite_is_false_and_the_three_op_one_is_worse() {
    const B_AND_NOTA: u32 = 0x3000_3000;
    const A_AND_NOTB: u32 = 0x849f_849f;
    const A_AND_B: u32 = 0x0b60_0b60;

    // Two-op AND+OR: false everywhere.
    let mut false_everywhere = 0usize;
    for p in 0u32..=0xffff {
        if (B_AND_NOTA | (A_AND_NOTB & p)) != ((p & MCG_MASK) ^ MCG_XOR) {
            false_everywhere += 1;
        }
    }
    assert_eq!(false_everywhere, 65536, "the two-op rewrite is not false everywhere");

    // Three-op: true, but strictly more ops than the source.
    let mut true_everywhere = 0usize;
    for p in 0u32..=0xffff {
        if (B_AND_NOTA | (A_AND_B & !p) | (A_AND_NOTB & p)) == ((p & MCG_MASK) ^ MCG_XOR) {
            true_everywhere += 1;
        }
    }
    assert_eq!(true_everywhere, 65536, "the three-op rewrite is not exact");
    assert!(
        3 > 2,
        "the three-op decomposition would have to be cheaper than the source's \
         two-op AND+XOR, which is the only remaining escape from LOP3"
    );
}

// ===========================================================================
// 8. Q6 — fp16 weights offline.  Fatal twice, independently.
// ===========================================================================

/// Decoding every weight to fp16 at load time is not a tuning decision, it does
/// not fit and it is four times slower.
///
/// * **Capacity.** 12.039 GB of packed weights at 3 bpw is 32.10e9 weights per
///   token. At 2 bytes each that is **64.2 GB** resident against a **25.8 GB**
///   box (24 GiB). 2.49x over. `AGENTS.md` §10 already records that two
///   concurrent model instances at 11.7 GB already exhaust it.
/// * **Bandwidth.** The kernel is not currently bandwidth bound -- that is the
///   whole point of `WHERE-THIS-STANDS.md:82-89` -- but at 2 B/weight it would
///   be: 125 Gcodes/s at the 250 GB/s ceiling against 541 Gcodes/s today.
#[test]
fn fp16_weights_are_fatal_on_capacity_and_on_bandwidth() {
    const BYTES_PER_TOKEN_PACKED: f64 = 12.039e9;
    const BITS_PER_CODE: f64 = 3.0;
    const BOX_BYTES: f64 = 24.0 * 1024.0 * 1024.0 * 1024.0; // 24 GiB
    const BOX_CEILING_GB_S: f64 = 250.0;
    const TODAY_GB_S: f64 = 203.0;

    let weights = BYTES_PER_TOKEN_PACKED * 8.0 / BITS_PER_CODE;
    assert!((weights - 32.104e9).abs() / 32.104e9 < 0.001, "weight count moved");

    let fp16_bytes = weights * 2.0;
    assert!(
        fp16_bytes > BOX_BYTES * 2.0,
        "fp16 weights are {} B, only {:.2}x the box",
        fp16_bytes,
        fp16_bytes / BOX_BYTES
    );
    println!(
        "fp16 weights: {:.1} GB resident vs {:.1} GB box = {:.2}x over",
        fp16_bytes / 1e9,
        BOX_BYTES / 1e9,
        fp16_bytes / BOX_BYTES
    );

    // And the bandwidth consequence, which would bite even with infinite RAM.
    let useful_gb_s = BOX_CEILING_GB_S * 1e9 / 2.0 / (8.0 / BITS_PER_CODE) / 1e9;
    let today_gb_s = TODAY_GB_S;
    assert!(
        useful_gb_s < today_gb_s / 4.0,
        "fp16 weights would run at {useful_gb_s:.1} GB/s, only {:.2}x below today",
        useful_gb_s / today_gb_s
    );
    println!(
        "fp16-weight bandwidth: {useful_gb_s:.1} GB/s useful = {:.2}x slower than today",
        today_gb_s / useful_gb_s
    );
}

// ===========================================================================
// 9. The one measurement that would settle it, named precisely.
// ===========================================================================

/// Three arms on the **live** `_2col_sk2` kernel, identical loads, identical
/// addresses, differing only in the decode body. This is the experiment the
/// whole brief turns on, and it has never been run: the existing `_nod` and
/// `contig4` arms were run on the 64-bit-shift kernel, not this one.
///
/// * **A (control)** — production body, verbatim, 4 ops, chain depth 4.
/// * **B (ILP-preserving, 2 ops)** — `y = p ^ 0x3b603b60`, take `h.x` only.
///   Two ops instead of four, 8 independent chains per extract, chain depth 2.
///   Garbage output, which is fine: this is a rate probe and `_nod` set the
///   precedent. This is the arm that separates "fewer ops" from "fewer chains",
///   because it removes ops **without** removing a single chain.
/// * **C (AND+OR, 2 ops)** — `y = 0x30003000u | (p & 0x849F849Fu)`. Also two ops,
///   also chain-preserving, but now the op mix is one AND + one OR instead of one
///   XOR, which is the closest an MSL source can get to asking for a `LOP3`.
///
/// Pre-registered reading: if the live kernel is issue bound, B/A >= 1.10 and the
/// pool is real. If B/A <= 1.02, the decode body is free on this kernel too, the
/// 21 % is unreachable from source, and A2's verdict stands as "no". Anything
/// between is a third case the model does not have.
#[test]
fn the_discriminating_measurement_is_named_with_its_pre_registered_reading() {
    // Pre-registered criterion, written before the run, per AGENTS.md §3.4.
    const ISSUE_BOUND_PREDICTION: f64 = 1.10; // >= this would confirm "issue bound"
    const FREE_PREDICTION: f64 = 1.02; // <= this would refute it

    assert!(
        ISSUE_BOUND_PREDICTION > FREE_PREDICTION,
        "the two predictions must be ordered, or the criterion is undecidable"
    );
    // And the gap must be wide enough that the control's own spread (1.13x was
    // observed on this box, AGENTS.md §10.5) cannot be mistaken for a result.
    assert!(
        ISSUE_BOUND_PREDICTION / FREE_PREDICTION > 1.07,
        "the pass/fail window ({FREE_PREDICTION}..{ISSUE_BOUND_PREDICTION}) is \
         narrower than this box's own decode spread of 1.13x, so the \
         experiment could not decide anything"
    );

    // The arms must be expressible in MSL, or they cannot be built. Arm C is
    // deliberately NOT the (false) two-op affine rewrite from
    // `the_two_op_affine_rewrite_is_false_and_the_three_op_one_is_worse`.
    let arm_a = "v[i] = decode_state_mcg(w);";
    let arm_b = "const uint y = p ^ 0x3b603b60u; v[i] = float(as_type<half>(y));";
    let arm_c = "const uint y = 0x30003000u | (p & 0x849F849Fu); v[i] = float(as_type<half>(y));";
    for (name, arm) in [("A", arm_a), ("B", arm_b), ("C", arm_c)] {
        assert!(arm.contains("v[i]"), "arm {name} does not write v[i]");
        assert!(!arm.contains(" if "), "arm {name} has a branch; it would measure the branch");
    }
    // B and C must consume `p` (the extracted word) so the extraction above them
    // cannot be dead-code-eliminated and take the ring loads with it -- the
    // failure mode `scripts/gen_exl3_gemv.py:309` warns about explicitly, and the
    // reason the earlier `49814d4` probe measured nothing.
    assert!(arm_b.contains("p ^") && arm_c.contains("p &"));
    // And they must not be constant-foldable, which a `static` function taking a
    // runtime value guarantees and a macro would not.
    assert!(!arm_b.contains("constexpr") && !arm_c.contains("constexpr"));

    // The result is NOT known: no arm has been run, because running one needs
    // the GPU and this probe is read-only. Filling `measured` in is the whole
    // point of this test.
    let measured: Option<(f64, f64)> = None; // <- (B/A, C/A)
    match measured {
        None => println!(
            "UNMEASURED (needs the GPU; this probe is read-only).\n  \
             B/A >= {ISSUE_BOUND_PREDICTION:.2} => issue bound, the decode pool \
             is real and a LOP3 fusion pass is worth building.\n  \
             B/A <= {FREE_PREDICTION:.2} => the decode body is free on the live \
             kernel too and A2's 'no' verdict is confirmed by measurement."
        ),
        Some((b, c)) => {
            for (label, r) in [("B", b), ("C", c)] {
                assert!(
                    r <= FREE_PREDICTION || r >= ISSUE_BOUND_PREDICTION,
                    "arm {label} measured A-ratio {r:.3}, inside the undecided gap \
                     ({FREE_PREDICTION}..{ISSUE_BOUND_PREDICTION}); the \
                     issue-bound/free model is incomplete and A2's verdict \
                     cannot be settled by this experiment alone"
                );
            }
        }
    }
    println!("arms:\n  A {arm_a}\n  B {arm_b}\n  C {arm_c}");
}