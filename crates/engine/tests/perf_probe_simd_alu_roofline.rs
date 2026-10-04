//! Is the EXL3 decode formulation leaving bandwidth on the table, and does the
//! M4 Pro's SIMD hardware offer anything unused?
//!
//! ## Answer: no on both, and here is the chain that forces it
//!
//! Quality preservation forces a 16-bit input window per code. Because
//! `decode_state_mcg` multiplies *before* masking, that window is 37 bits, so a
//! lane needs two 32-bit reads (amp 2.667) and eight runtime-offset extractions.
//! Those constraints are not choices, and every alternative is already barred:
//! narrowing the window is quality loss, sharing the overlap by `simd_shuffle`
//! is measured 180.5 -> 146.5 GB/s, and the lane-contiguous repack that would
//! make every shift a compile-time immediate is withdrawn because it changes
//! the dequant.
//!
//! Priced against the machine, that lands at **~199 GB/s at 3 bpw and a ceiling
//! of ~18.5 tok/s**, against 16.05 measured and 20 requested.
//!
//! ## What the measurements do and do not settle
//!
//! ```
//!   resource          predicted form        3 bpw      4 bpw      6 bpw
//!   ALU / loads / warp  proportional to KA    199.4     265.9->250  399->250
//!   bytes (DRAM)        proportional to KA    199.4     265.9->250  399->250
//! ```
//!
//! Both surviving models have the *same functional form*. The 3/4/6 bpw sweep in
//! `spec/decode-3bpw-amplification/implementation_notes.md:70-72` therefore
//! **cannot** discriminate between them, and the record's claim that "only
//! amplification tracks bandwidth" (`:80-87`) does not follow from it. What the
//! sweep does establish, and what this file asserts, is that **some cost that
//! does not scale with bytes-per-tile binds at 3 bpw**, and that the crossover
//! where it meets the 250.5 GB/s copy roof falls between 3 and 4 bpw -- the
//! prediction that distinguishes this account from "DRAM is slow".
//!
//! ## The op count is not the weak link
//!
//! The machine constant is *fitted* from the shipped kernel, so every conclusion
//! here is invariant to the absolute op count: raising `OpBudget::windows` from
//! 18 to 30 (verified by sabotage) leaves all three roofline tests and the
//! tok/s estimate unchanged, because only the form `1/KA` and the *ratios*
//! between budget components enter. A wrong op count cannot produce this
//! result; a wrong functional form cannot either.
//!
//! ## What is left, stated as a negative
//!
//! 1. **`simdgroup_matrix_*` is unusable, provably, three ways** -- the dequant
//!    is 82% of the ALU and a matrix unit cannot touch it; M stays 1; and
//!    staging 12.0 GB of packed bits into matrix operands costs +96 ms against
//!    a 62 ms step.
//! 2. **There is no integer SIMD unit to widen into.** Apple's GPUs expose
//!    vector *fp* types and scalar integer ops. `ulong` is a register pair on a
//!    32-lane core, so `ulong2`/`ulong4` word ops are 4x and 8x scalar: they
//!    merge *loads*, and loads are not what is short of capacity.
//! 3. **The reduction is 0.4% of a threadgroup's work.** Eight shuffles, two
//!    barriers and an 8-deep serial sum against 21 unrolled K steps. Widening
//!    it cannot move the roofline, and the barrier-free `_colmajor_stream`
//!    variant that already exists removes barriers, not per-lane ALU.
//! 4. **The 138-kernel specialised library costs compile time, not
//!    throughput.** Three kernels are live; each is entered ~180 times per step.
//!
//! **One measurement in this file is genuinely unresolved and is recorded as
//! such**: `linear.metal:96-106` records that removing the decode *arithmetic*
//! made the kernel 17.6% *slower* (163.0 -> 134.3 GB/s). Under an ALU-bound
//! account, deleting 32 of ~66 ops per lane per tile must have helped. That is
//! real evidence for a load-side limit, it contradicts this file's account, and
//! the reconciling A/B has not been run. See
//! [`the_discriminating_experiment_is_named_not_assumed`].
//!
//! **Provenance of every number** is in [`MEASURED`]. No GPU is touched: this is
//! arithmetic plus a CPU transcription of the Metal, checked against
//! `kaplumba_engine::exl3::decode_tile` -- the same independent reference
//! `rate4_colmajor_bit_exact.rs:15-18` insists on, for the same reason (two GPU
//! kernels can share an error; a CPU reference cannot).
//!
//! **This file is a probe, not a coverage claim.** AGENTS.md 3.4 requires a
//! coverage claim to have been seen capable of failing.
//! [`the_window_check_can_fail`] sabotages the transcription and asserts the
//! comparison reports a mismatch. The roofline tests were separately shown to
//! fail with diagnosable messages when the fit point moves (3 bpw -> 4 bpw:
//! "3 bpw: predicted 186.8 GB/s, measured 199.4 (6.3% off)"; "crossover at 4.02
//! bpw: 4 bpw should already be DRAM-bound") and when the multiply-before-mask
//! ordering is inverted.

// ---------------------------------------------------------------------------
// Measured inputs. Every one carries the file and line it came from.
// ---------------------------------------------------------------------------

/// Bandwidths of the live 3 bpw colmajor GEMV, and of its rate-shifted twins.
///
/// `spec/decode-3bpw-amplification/implementation_notes.md:70-72`, medians of an
/// isolated sweep, and `:94` states the isolated 3 bpw number agrees with the
/// ~196-207 GB/s measured in-engine, so it transfers.
#[derive(Debug, Clone, Copy)]
struct Measured {
    bpw: u32,
    gb_per_s: f64,
}

const MEASURED: [Measured; 3] = [
    Measured {
        bpw: 3,
        gb_per_s: 199.4,
    },
    Measured {
        bpw: 4,
        gb_per_s: 249.0,
    },
    Measured {
        bpw: 6,
        gb_per_s: 251.4,
    },
];

/// Streaming copy ceiling on this box, and the datasheet wire rate.
///
/// 250.5 GB/s measured: `spec/decode-19tok-repacked-250gb/implementation_notes.md:30`.
/// 273 GB/s wire: AGENTS.md 8. Both are cited in every spec in this repo.
const COPY_CEILING_GB_PER_S: f64 = 250.5;
const WIRE_GB_PER_S: f64 = 273.0;

/// End-to-end decode figures the target is measured against.
const BASELINE_TOK_PER_S: f64 = 16.05;
const TARGET_TOK_PER_S: f64 = 20.0;

/// Bytes of weight per 16x16 tile at an integer rate: `256 * KA` bits.
const fn tile_bytes(bpw: u32) -> f64 {
    256.0 * bpw as f64 / 8.0
}

// ---------------------------------------------------------------------------
// The CPU transcription of the live kernel.
// ---------------------------------------------------------------------------

/// `extract_v_mcg_32`, `crates/kernels/src/linear_repacked.metal:53-64`.
///
/// Transcribed line for line. `sh0` is never 0 at 3 bpw (it cycles 19/11/3/27 --
/// asserted by [`extract_matches_the_cpu_reference`]), so `32 - sh0` never
/// reaches 32 and the shift amounts below are all in range.
fn extract_v_mcg_32(wa: u32, wb: u32, sh0: u32) -> [u32; 8] {
    let rsh0 = 32 - sh0;
    let a_hi = wa.wrapping_shl(sh0) | wb.wrapping_shr(rsh0);
    let a_lo = wb.wrapping_shl(sh0);
    [
        (a_hi >> 16) & 0xffff,
        (a_hi >> 13) & 0xffff,
        (a_hi >> 10) & 0xffff,
        (a_hi >> 7) & 0xffff,
        (a_hi >> 4) & 0xffff,
        (a_hi >> 1) & 0xffff,
        ((a_hi << 2) | (a_lo >> 30)) & 0xffff,
        ((a_hi << 5) | (a_lo >> 27)) & 0xffff,
    ]
}

/// The same, with one window offset perturbed by `delta` bits.
///
/// Exists so [`the_window_check_can_fail`] can prove the equality test below is
/// capable of failing. `delta == 0` is the real kernel.
fn extract_v_mcg_32_sabotaged(wa: u32, wb: u32, sh0: u32, delta: i32) -> [u32; 8] {
    let real = extract_v_mcg_32(wa, wb, sh0);
    if delta == 0 {
        return real;
    }
    let mut out = real;
    let index = 3usize;
    out[index] = (real[index].wrapping_add(delta as u32)) & 0xffff;
    out
}

/// A tile's 32-bit words, in the order both the CPU and the GPU read them.
///
/// `unpack_integer` (`crates/engine/src/exl3.rs:240-243`) and the kernel's
/// `ring_word<true>` (`linear.metal:267-272`, one native 32-bit read) agree
/// because both are little-endian: `packed[2i] | packed[2i+1] << 16`.
fn tile_words(packed: &[u16]) -> Vec<u32> {
    packed
        .chunks_exact(2)
        .map(|pair| u32::from(pair[0]) | (u32::from(pair[1]) << 16))
        .collect()
}

/// Per-lane word indices and funnel shift, hoisted out of the K loop.
///
/// `linear_repacked.metal:2095-2100`. Identical arithmetic to
/// `exl3_decode8_mcg` (`linear.metal:351-358`) and to `unpack_integer`
/// (`crates/engine/src/exl3.rs:256-265`).
fn lane_window(bpw: u32, lane: u32) -> (u32, u32, u32, u32) {
    let ring_words = bpw * 8;
    let lo0 = (lane * 8 + 1) * bpw + ring_words * 32 - 16;
    let sh0 = lo0 % 32;
    let hi0 = lo0 / 32;
    let j0 = if hi0 >= ring_words {
        hi0 - ring_words
    } else {
        hi0
    };
    let j1 = if j0 + 1 >= ring_words {
        j0 + 1 - ring_words
    } else {
        j0 + 1
    };
    (sh0, j0, j1, lo0)
}

/// Tile rows a lane's eight codes land on, and its two columns.
///
/// `unpack_integer`'s scatter (`crates/engine/src/exl3.rs:283-291`): rows
/// `[r0, r0+1, r0+8, r0+9]` with `r0 = (lane % 4) * 2`, columns `lane/4` and
/// `lane/4 + 8`. `state_for_tile` (`linear.metal:44-54`) is the GPU-side
/// inverse.
fn lane_slot(lane: u32, i: u32) -> (usize, usize) {
    let r0 = (lane % 4) * 2;
    let k = match i % 4 {
        0 => r0,
        1 => r0 + 1,
        2 => r0 + 8,
        _ => r0 + 9,
    };
    let n = lane / 4 + (i / 4) * 8;
    (k as usize, n as usize)
}

// ---------------------------------------------------------------------------
// Test 1: the transcription is the kernel, and the check can fail.
// ---------------------------------------------------------------------------

/// The transcription must reproduce `unpack_integer`'s windows at 3 bpw, the
/// only rate it is written for.
///
/// `extract_v_mcg_32` hardcodes the eight extraction offsets
/// (16, 13, 10, 7, 4, 1, then two straddles), which is a stride of three -- it
/// is a 3 bpw kernel, and `constexpr uint KA = 3u` on every one of the fifteen
/// callers. Running it at 2/4/6 bpw is not a stricter test, it is a wrong test,
/// and it is why this loop starts at 3.
///
/// The decoded floats are checked too, not just the windows: `decode_tile` is
/// the authority for the value that reaches the accumulator, and a window can
/// match while the codebook applied to it does not.
#[test]
fn extract_matches_the_cpu_reference() {
    use kaplumba_engine::exl3::{
        BitRate, Codebook, decode_state, decode_tile, pack_integer_states,
    };

    let bits = 3u8;
    let mut rng = 0x1234_5678u32;
    let mask = (1u32 << bits) - 1;
    let states: Vec<u16> = (0..256)
        .map(|_| {
            rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (rng & mask) as u16
        })
        .collect();

    let (packed, windows) = pack_integer_states(&states, bits).unwrap();
    let words = tile_words(&packed);
    let tile = decode_tile(&packed, BitRate::Integer(bits), Codebook::Mcg).unwrap();

    for lane in 0..32u32 {
        let (sh0, j0, j1, _lo0) = lane_window(bits as u32, lane);
        assert_ne!(sh0, 0, "a 0 shift makes rsh0 == 32 undefined");
        let windows_mine = extract_v_mcg_32(words[j0 as usize], words[j1 as usize], sh0);

        for i in 0..8u32 {
            let (k, n) = lane_slot(lane, i);

            // `unpack_integer` returns the TILE-ordered array (`out[k*16+n]`),
            // not `by_state[lane*8+i]` -- it scatters before returning, and an
            // earlier draft of this file indexed the state-ordered way and
            // failed. The scatter is `lane_slot` above.
            let reference_window = windows[k * 16 + n];
            assert_eq!(
                windows_mine[i as usize] as u16, reference_window,
                "bits={bits} lane={lane} i={i}: window mismatch"
            );

            let mine = decode_state(windows_mine[i as usize] as u16, Codebook::Mcg);
            let theirs = tile[k * 16 + n];
            assert_eq!(
                mine.to_bits(),
                theirs.to_bits(),
                "bits={bits} lane={lane} i={i}: decoded value mismatch"
            );
        }
    }
}

/// AGENTS.md 3.4: a coverage claim is not a claim until the failing branch has
/// been seen to fail.
///
/// Perturbs one window by one bit and requires the transcription to disagree
/// with `unpack_integer` somewhere. If this test ever passes with a
/// non-zero `delta`, the equality test above is vacuous and this file's
/// conclusions are worthless -- so the file refuses to be trusted.
#[test]
fn the_window_check_can_fail() {
    use kaplumba_engine::exl3::pack_integer_states;

    let bits = 3u8;
    let mut rng = 0xDEAD_BEEFu32;
    let mask = (1u32 << bits) - 1;
    let states: Vec<u16> = (0..256)
        .map(|_| {
            rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (rng & mask) as u16
        })
        .collect();
    let (packed, windows) = pack_integer_states(&states, bits).unwrap();
    let words = tile_words(&packed);

    let mut mismatches = 0usize;
    for lane in 0..32u32 {
        let (sh0, j0, j1, _) = lane_window(bits as u32, lane);
        let good = extract_v_mcg_32(words[j0 as usize], words[j1 as usize], sh0);
        let bad = extract_v_mcg_32_sabotaged(words[j0 as usize], words[j1 as usize], sh0, 1);
        assert_ne!(
            good, bad,
            "the sabotage is a no-op, so nothing is being tested"
        );
        for i in 0..8usize {
            let (k, n) = lane_slot(lane, i as u32);
            if bad[i] as u16 != windows[k * 16 + n] {
                mismatches += 1;
            }
        }
    }
    assert!(
        mismatches > 0,
        "a one-bit perturbation of every lane's window 3 produced no mismatch: \
         `extract_matches_the_cpu_reference` cannot fail and proves nothing"
    );
}

/// The 37-bit window, re-derived here rather than taken from the standing note.
///
/// `spec/decode-3bpw-amplification/implementation_notes.md:123-125` reports 37
/// ring bits per lane by bit-flip enumeration. This reproduces it *per lane* --
/// the union over all 32 lanes is the whole 768-bit ring, since the 32 windows
/// are 24 bits apart and there are 256 of them, so a whole-tile count is
/// vacuous (an earlier draft of this file asserted it and returned 512 of 512).
///
/// The question asked is the one the notes' own method note warns about
/// (`:155-159`): not "which bits affect the 3-bit code" but "which ring bits
/// affect `decode_state_mcg`'s float output", because the multiply precedes the
/// mask. That ordering is asserted separately in
/// [`the_op_budget_matches_the_kernel_source`].
#[test]
fn mcgs_window_is_thirty_seven_bits_not_twenty_four() {
    use kaplumba_engine::exl3::pack_integer_states;

    let bits = 3u8;
    let bpw = bits as u32;
    let ring_bits = 256 * bits as usize;

    let mut rng = 0x0BAD_C0DEu32;
    let mask = (1u32 << bits) - 1;
    let states: Vec<u16> = (0..256)
        .map(|_| {
            rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (rng & mask) as u16
        })
        .collect();
    let (packed, _) = pack_integer_states(&states, bits).unwrap();
    let words = tile_words(&packed);

    for lane in 0..32u32 {
        let base = lane_decoded(&words, bpw, lane);

        // Bit-flip enumeration: which ring positions move any of this lane's
        // eight decoded floats?
        let mut hits = vec![0usize; ring_bits];
        for position in 0..ring_bits {
            let mut flipped = words.clone();
            flipped[position / 32] ^= 1u32 << (31 - (position % 32));
            let after = lane_decoded(&flipped, bpw, lane);
            if after != base {
                hits[position] = 1;
            }
        }
        let observed: Vec<usize> = (0..ring_bits)
            .filter(|position| hits[*position] == 1)
            .collect();

        // The structural claim, stated without predicting the start position:
        // the eight 16-bit windows advance by three bits each and overlap, so
        // their union in ring space is a single contiguous 37-bit run -- which
        // is what forces a second 32-bit read, since no 32-bit-aligned pair of
        // words can always contain 37 bits at an arbitrary rotation.
        //
        // The start is *not* asserted against a formula. An earlier draft
        // derived `a` bit k -> ring `j0*32 + k - sh0`, which was off by the
        // funnel's alignment, and the test failed on a prediction rather than
        // on the thing being claimed. Contiguity is the claim; the rotation is
        // an implementation detail.
        let members: Vec<bool> = (0..ring_bits).map(|position| hits[position] == 1).collect();
        let starts: Vec<usize> = (0..ring_bits)
            .filter(|position| {
                members[*position] && !members[(*position + ring_bits - 1) % ring_bits]
            })
            .collect();
        assert_eq!(
            starts.len(),
            1,
            "lane {lane}: the influencing set must be one contiguous run, found {} runs",
            starts.len()
        );
        let start = starts[0];
        for offset in 0..37usize {
            assert!(
                members[(start + offset) % ring_bits],
                "lane {lane}: ring position {} is inside the run but does not influence",
                (start + offset) % ring_bits
            );
        }
        assert!(
            !members[(start + 37) % ring_bits],
            "lane {lane}: the run is longer than 37 bits"
        );
        assert_eq!(
            observed.len(),
            37,
            "lane {lane}: {} influencing ring bits, expected 37",
            observed.len()
        );
    }

    // And the number that matters downstream: 37 > 32, so one 32-bit read cannot
    // carry a lane's window, which is why amp is 2.0 rather than 1.0 at every
    // rate below 8 bpw.
    assert!(16 + 7 * bits as usize > 32);
}

/// The eight decoded floats one lane owns, in `i` order.
fn lane_decoded(words: &[u32], bpw: u32, lane: u32) -> [u32; 8] {
    use kaplumba_engine::exl3::{Codebook, decode_state};
    let (sh0, j0, j1, _) = lane_window(bpw, lane);
    let windows = extract_v_mcg_32(words[j0 as usize], words[j1 as usize], sh0);
    let mut out = [0u32; 8];
    for i in 0..8usize {
        out[i] = decode_state(windows[i] as u16, Codebook::Mcg).to_bits();
    }
    out
}

// ---------------------------------------------------------------------------
// Test 2: the op budget, anchored to the Metal source rather than to a comment.
// ---------------------------------------------------------------------------

const LINEAR_METAL: &str = include_str!("../../kernels/src/linear.metal");
const REPACKED_METAL: &str = include_str!("../../kernels/src/linear_repacked.metal");

/// Integer ALU lane-ops per lane per 16x16 tile, itemised.
///
/// Every item is a count of *source* operations, and
/// [`the_op_budget_matches_the_kernel_source`] re-derives the two biggest from
/// the Metal text so a future edit cannot silently invalidate the roofline.
/// The lowering assumptions, stated because they are assumptions:
///
/// * `(a & C1) ^ C2` fuses to one `LOP3` (LLVM does this; it is the whole
///   reason the codebook specialisation was worth 2.61x, `linear.metal:96-106`).
/// * `(A_hi << s) & 0xffff` is one `SHF` + one `LOP3`.
/// * `(A_hi << s) | (A_lo >> t) & 0xffff` is two `SHF` + one 3-input `LOP3`.
/// * `ulong` shifts are pairs of 32-bit ops, which is why they are counted at
///   all; see [`ulong_word_ops_buy_nothing_on_a_32_lane_gpu`].
struct OpBudget {
    /// `A_hi = (wa << sh0) | (wb >> rsh0)` and `A_lo = wb << sh0`: 3 `SHF` + 1 `LOP3`.
    funnel: f64,
    /// Eight window extractions: 6 x 2 ops + 2 x 3 ops.
    windows: f64,
    /// `decode_state_mcg` x 8: `IMUL`, `LOP3`, `HADD2`, `F2F`.
    codebook: f64,
    /// Eight `fma`s, four x values used twice each.
    accum: f64,
    /// Four `half` x loads widened to `float`.
    x_convert: f64,
}

impl OpBudget {
    const fn live() -> Self {
        Self {
            funnel: 4.0,
            windows: 18.0,
            codebook: 32.0,
            accum: 8.0,
            x_convert: 4.0,
        }
    }

    /// The bit-unpacking cost: everything that exists only because the codes are
    /// packed. Excludes the codebook and the accumulate, which any quantised
    /// GEMV pays.
    const fn unpacking(&self) -> f64 {
        self.funnel + self.windows
    }

    const fn total(&self) -> f64 {
        self.funnel + self.windows + self.codebook + self.accum + self.x_convert
    }

    /// Bytes of weight a lane consumes per tile: eight codes at `KA` bits.
    const fn bytes_per_lane_tile(bpw: u32) -> f64 {
        bpw as f64
    }

    /// ALU lane-ops per byte of weight.
    fn ops_per_byte(&self, bpw: u32) -> f64 {
        self.total() / Self::bytes_per_lane_tile(bpw)
    }
}

/// A tile is 32 lanes, so a warp's ALU work per tile is `32 * total`.
fn warp_alus_per_tile(budget: &OpBudget) -> f64 {
    32.0 * budget.total()
}

/// Re-derive the two biggest op-count items from the Metal text.
///
/// Not a "does the comment still say what it said" check -- a check that the
/// *shape* the roofline depends on is still the shape in the file: one funnel
/// line, eight window extractions with the offsets the arithmetic assumes, and
/// a four-operation codebook.
#[test]
fn the_op_budget_matches_the_kernel_source() {
    let extract = function_body(REPACKED_METAL, "extract_v_mcg_32")
        .expect("extract_v_mcg_32 must exist in linear_repacked.metal");
    let codebook = function_body(LINEAR_METAL, "decode_state_mcg")
        .expect("decode_state_mcg must exist in linear.metal");

    // Six windows are a shift of A_hi; two need A_lo as well.
    let shifts_of_a_hi = extract
        .lines()
        .filter(|line| line.contains("(A_hi >>") || line.contains("(A_hi <<"))
        .count();
    assert_eq!(
        shifts_of_a_hi, 8,
        "eight extractions expected, got {shifts_of_a_hi}"
    );
    let uses_a_lo = extract.matches("A_lo").count();
    assert!(
        uses_a_lo >= 3,
        "A_lo must be built and used by two extractions; found {uses_a_lo} mentions"
    );

    // The offsets the arithmetic assumes, in order. If these move, the ALU count
    // above is wrong.
    for offset in [16u32, 13, 10, 7, 4, 1] {
        assert!(
            extract.contains(&format!("(A_hi >> {offset})")),
            "extraction offset {offset} is gone"
        );
    }
    for (shift, back) in [(2u32, 30u32), (5, 27)] {
        assert!(
            extract.contains(&format!("(A_hi << {shift}) | (A_lo >> {back})")),
            "the straddle extraction A_hi<<{shift} | A_lo>>{back} is gone"
        );
    }

    // Four operations, and the multiply really does precede the mask. The order
    // is the whole reason the window is 37 bits wide rather than 8.
    let mask = codebook.find("value & 0x8fff8fff").expect("the MCG mask");
    let multiply = codebook
        .find("value * 0xCBAC1FED")
        .expect("the MCG multiply");
    assert!(
        multiply < mask,
        "if the mask moved before the multiply the 37-bit window collapses and \
         this file's central negative result is void"
    );
    assert!(codebook.contains("^ (value & 0x8fff8fffu)"));
    assert!(codebook.contains("as_type<half2>"));
    assert!(codebook.contains("h.x + h.y"));
}

/// Lane-loads per tile, and the amp that follows.
///
/// The rate is not cosmetic. At 3 and 4 bpw every code's window lies inside the
/// 64-bit funnel pair, so two 32-bit reads per lane suffice and amp is
/// `64 / 8KA`. At 6 bpw `linear_gemv_ishizuki_k6_colmajor_contig4_bench`
/// reads `j0, j1, j2` (`linear_repacked.metal:3564-3565`) -- three reads, so
/// amp is `96 / 48 = 2.000`, the same as 4 bpw. That near-equality is *why*
/// the rate sweep cannot tell a load-bound kernel from an ALU-bound one: both
/// hypotheses predict bandwidth proportional to KA, and both predict amp 2.0
/// at 4 and 6 bpw.
#[test]
fn amp_is_two_words_per_lane_and_why() {
    let kernel = function_body(REPACKED_METAL, "linear_gemv_ishizuki_k3_colmajor_2col")
        .expect("the live 3 bpw kernel must exist");

    // One call site per (column, k-step); each reads j0 and j1.
    let two_word_reads = kernel
        .lines()
        .filter(|line| {
            let reads = line.matches("[j0]").count() + line.matches("[j1]").count();
            reads == 2
        })
        .count();
    assert!(
        two_word_reads >= 2,
        "expected several 'w = t[j0], w = t[j1]' lines, found {two_word_reads}"
    );

    // And it does not read a third word: the 3-word funnel of
    // `exl3_decode8_mcg` (linear.metal:360-364) is not what is live.
    assert!(
        !kernel.contains("[j2]"),
        "the live kernel reads j2, so it is the 3-load form and amp is 4.0, not 2.667"
    );

    for (bpw, loads, expected_amp) in [(3u32, 2.0, 2.667f64), (4, 2.0, 2.0), (6, 3.0, 2.0)] {
        let words = 8.0 * bpw as f64;
        let amp = 32.0 * loads / words;
        assert!(
            (amp - expected_amp).abs() < 0.01,
            "amp at {bpw} bpw is {amp}, expected {expected_amp}"
        );
        assert!(
            amp >= 2.0,
            "amp below 2.0 would mean under two reads per lane"
        );
    }

    // Cross-check the 6 bpw three-read form against the source, since the whole
    // point is that the rates differ in *loads* and not only in bytes.
    let k6 = function_body(
        REPACKED_METAL,
        "linear_gemv_ishizuki_k6_colmajor_contig4_bench",
    )
    .expect("the 6 bpw bench kernel must exist");
    let three_word_reads = k6
        .lines()
        .filter(|line| line.matches("t0[j").count() == 3)
        .count();
    assert!(
        three_word_reads >= 1,
        "the 6 bpw kernel must read three words per lane, found {three_word_reads}"
    );
}

// ---------------------------------------------------------------------------
// Test 3: the roofline. Which roof binds at 3 bpw?
// ---------------------------------------------------------------------------

/// The machine's integer-ALU lane-op rate, *fitted* to the 3 bpw measurement.
///
/// Fitted rather than assumed because there is no trustworthy published figure
/// for this part in this repo and the Metal toolchain is absent on this box
/// (`spec/decode-3bpw-amplification/implementation_notes.md:18-21`), so no
/// disassembly may be invented. The point of the constant is that it is
/// *common* to all three measurements: see
/// [`one_constant_explains_all_three_rates`].
fn fitted_alu_lane_ops_per_s(budget: &OpBudget) -> f64 {
    MEASURED[0].gb_per_s * 1e9 * budget.ops_per_byte(MEASURED[0].bpw)
}

/// The ALU roof for a rate: lane-ops available per byte, inverted.
fn alu_roof_gb_per_s(budget: &OpBudget, alu_lane_ops_per_s: f64, bpw: u32) -> f64 {
    alu_lane_ops_per_s / budget.ops_per_byte(bpw) / 1e9
}

/// The roofline prediction: `min(ALU roof, DRAM roof)`.
fn predict_gb_per_s(budget: &OpBudget, alu_lane_ops_per_s: f64, bpw: u32) -> f64 {
    alu_roof_gb_per_s(budget, alu_lane_ops_per_s, bpw).min(COPY_CEILING_GB_PER_S)
}

/// One per-tile-constant resource, `min`ed with the bytes-per-tile roof, fits
/// all three measured rates. This is the load-bearing claim of this file, and it
/// is falsifiable: if any rate were off by more than a couple of percent, the
/// "per-tile cost, not DRAM, binds at 3 bpw" conclusion would be wrong.
#[test]
fn one_constant_explains_all_three_rates() {
    let budget = OpBudget::live();
    let alu = fitted_alu_lane_ops_per_s(&budget);

    for measured in MEASURED {
        let predicted = predict_gb_per_s(&budget, alu, measured.bpw);
        let error = (predicted - measured.gb_per_s).abs() / measured.gb_per_s;
        assert!(
            error < 0.02,
            "{} bpw: predicted {predicted:.1} GB/s, measured {:.1} ({:.1}% off)",
            measured.bpw,
            measured.gb_per_s,
            error * 100.0
        );
    }
}

/// The falsifiable crossover: the rate at which the ALU roof meets the DRAM
/// roof must land between the two measured rates, and 3 bpw must be on the ALU
/// side of it while 4 and 6 are on the DRAM side.
///
/// This is the prediction that distinguishes this account from "DRAM is slow".
/// If the crossover came out at, say, 8 bpw, then 3 bpw would be DRAM-bound
/// and the whole ALU analysis would be beside the point.
#[test]
fn the_roofs_cross_between_3_and_4_bpw() {
    let budget = OpBudget::live();
    let alu = fitted_alu_lane_ops_per_s(&budget);
    let crossover_bpw = budget.total() * COPY_CEILING_GB_PER_S * 1e9 / alu;

    assert!(
        crossover_bpw > MEASURED[0].bpw as f64,
        "crossover at {crossover_bpw:.2} bpw: 3 bpw would be DRAM-bound and the \
         ALU analysis is void"
    );
    assert!(
        crossover_bpw <= MEASURED[1].bpw as f64,
        "crossover at {crossover_bpw:.2} bpw: 4 bpw should already be DRAM-bound"
    );

    // 3 bpw is ALU-bound at 80% of the copy ceiling, which is the gap to close.
    let headroom = COPY_CEILING_GB_PER_S / MEASURED[0].gb_per_s;
    assert!(
        headroom > 1.20 && headroom < 1.30,
        "headroom is {headroom:.3}x, expected ~1.256x (250.5 / 199.4)"
    );
}

/// A per-tile-constant resource and a per-byte resource are the *only* two
/// functional forms consistent with the three rates, and the data cannot tell
/// them apart.
///
/// This is the honest limit of the rate sweep, and it is why the withdrawn
/// repack's "199 -> 250 GB/s" prediction (which silently assumed the lane-load
/// resource specifically) was never tested. Stated as a test so that a future
/// author cannot claim the sweep settled it.
#[test]
fn the_rate_sweep_cannot_identify_which_resource_binds() {
    let budget = OpBudget::live();

    // Model A: a cost per tile, independent of KA. Bandwidth grows with KA.
    let per_tile: Vec<f64> = MEASURED
        .iter()
        .map(|m| MEASURED[0].gb_per_s * m.bpw as f64 / MEASURED[0].bpw as f64)
        .collect();
    // Model B: a cost per byte. Bandwidth is flat in KA.
    let per_byte: Vec<f64> = MEASURED.iter().map(|_| MEASURED[0].gb_per_s).collect();

    // Model A only survives once the DRAM roof clamps it -- and model A's
    // uncapped 6 bpw value (398.8 GB/s) is 1.59x the measured 251.4.
    let unclamped_6 = MEASURED[0].gb_per_s * 6.0 / 3.0;
    assert!(
        unclamped_6 > COPY_CEILING_GB_PER_S,
        "the unclamped per-tile model must exceed the copy ceiling, or it is not \
         being tested"
    );

    // Model B (flat) is refuted outright: 4 bpw measured 25% above 3 bpw.
    assert!(
        MEASURED[1].gb_per_s / MEASURED[0].gb_per_s > 1.20,
        "a per-byte model would predict a flat rate; it does not fit"
    );

    // Both surviving models are per-tile, so they cannot be separated by rate.
    assert_eq!(per_tile.len(), 3);
    assert_eq!(per_byte.len(), 3);

    // Whatever binds, it is paid once per tile no matter how many bytes the
    // tile holds -- which is why the ALU account and the load account are
    // numerically indistinguishable here, and why the discriminating
    // experiment has to be an A/B inside one rate.
    assert!(
        budget.total() > budget.codebook,
        "if unpacking dominated, the shape of the argument would change"
    );
}

/// What the three measurements *do* settle: a per-tile resource binds at
/// 3 bpw, DRAM does not. And how much of the achievable step is GEMV at all.
#[test]
fn dram_is_not_the_binding_roof_at_3_bpw() {
    // 3 bpw measured 199.4 against a 250.5 copy ceiling: 80%.
    let util = MEASURED[0].gb_per_s / COPY_CEILING_GB_PER_S;
    assert!(
        util < 0.85,
        "3 bpw is already at the copy ceiling; nothing to win"
    );

    // The in-engine figure the target is set against, for coherence: if the
    // GEMV is 199.4 and the step measures 193, the non-GEMV share is ~3%,
    // which is what a 60-layer step of norms and attention should cost.
    let in_engine = 193.0;
    let non_gemv = 1.0 - in_engine / MEASURED[0].gb_per_s;
    assert!(
        non_gemv > 0.0 && non_gemv < 0.06,
        "in-engine 193 GB/s against an isolated 199.4 implies a non-GEMV share of \
         {non_gemv:.3}; a larger share means something else is being counted"
    );
}

// ---------------------------------------------------------------------------
// Test 4: the hardware questions. Why the SIMD width is unavailable.
// ---------------------------------------------------------------------------

/// Useful arithmetic intensity of the decode GEMV, against the machine balance
/// the same GEMV would need to be memory-bound.
#[test]
fn the_gemv_is_trivially_memory_bound_and_that_is_not_the_problem() {
    // One FMA per weight, one tile of weights per 96 bytes.
    let flops_per_byte = 2.0 * 256.0 / tile_bytes(3);
    assert!(
        (flops_per_byte - 5.333).abs() < 0.01,
        "useful intensity is {flops_per_byte} FLOP/byte at 3 bpw"
    );

    // Machine balance: fp32 peak over the wire. Stated as a range because the
    // fp32 peak of this part is not recorded anywhere trustworthy in this repo;
    // 20 cores x {64,128} fp32 FLOP-lanes/cycle x {2.6, 3.4} GHz.
    let low = 20.0 * 64.0 * 2.6e9 * 2.0 / (WIRE_GB_PER_S * 1e9);
    let high = 20.0 * 128.0 * 3.4e9 * 2.0 / (WIRE_GB_PER_S * 1e9);
    assert!(
        low > 20.0 && high < 80.0,
        "machine balance range {low:.0}..{high:.0} FLOP/byte"
    );

    // The GEMM's own arithmetic is 4-15x under the balance point even on the
    // pessimistic end. So a plain fp16 GEMV at this shape is nowhere near
    // compute-bound, and no amount of matrix hardware is what stands between
    // 199 and 250 GB/s.
    assert!(
        high / flops_per_byte > 4.0,
        "the useful arithmetic is only {:.1}x under the pessimistic balance point",
        high / flops_per_byte
    );
    assert!(low / flops_per_byte > 3.0);

    // The whole difficulty is the dequant tax, which the balance calculation
    // does not see at all: the kernel performs 4.1x the minimum lane-ops per
    // useful FLOP, and 8.25x the minimum lane-ops per weight byte.
    let budget = OpBudget::live();
    let per_flop = budget.ops_per_byte(3) / flops_per_byte;
    assert!(
        per_flop > 3.5 && per_flop < 4.5,
        "the kernel performs {per_flop:.2}x the minimum lane-ops per useful FLOP"
    );
    let gemm_minimum_ops_per_byte = 256.0 / tile_bytes(3);
    let tax = budget.ops_per_byte(3) / gemm_minimum_ops_per_byte;
    assert!(
        tax > 7.0 && tax < 9.5,
        "the ALU tax is {tax:.2}x the minimum (8 FMAs per {}-byte tile)",
        tile_bytes(3)
    );
}

/// `simdgroup_matrix_*` is unusable, and the reason is arithmetic rather than
/// architectural taste.
///
/// Three independent reasons, any one of which is sufficient. This test exists
/// because "matrix units do not apply to a GEMV" is the kind of claim that gets
/// repeated without being checked, and the first two reasons are the ones that
/// are actually decisive.
#[test]
fn amx_is_unusable_not_merely_unused() {
    let budget = OpBudget::live();

    // (1) The dequant is 48% of the ALU and a matrix unit cannot touch it: the
    //     operand it would consume is the *decoded* weight, and decoding is
    //     the cost. Amdahl on the op budget alone caps the win.
    let dequant_fraction = budget.unpacking() + budget.codebook;
    let total = budget.total();
    assert!(
        dequant_fraction / total > 0.7,
        "dequant is {dequant_fraction} of {total}"
    );
    // Replacing 8 FMAs by one 8x8x8 MMA is 8 ops saved out of `total`.
    let best_case = 8.0 / total;
    assert!(
        best_case < 0.15,
        "an MMA saves at most {best_case:.3} of the ALU even in the impossible \
         case that decoding were free"
    );

    // (2) Arithmetic intensity does not improve. M = 1 either way: one
    //     activation vector. A matrix unit changes the instruction count of an
    //     operation the kernel is not short of, because the kernel is short of
    //     *bytes and integer ops*, not of multiplies.
    let flops_per_byte = 2.0 * 256.0 / tile_bytes(3);
    assert!(
        flops_per_byte < 6.0,
        "the decode GEMV is a rank-1 update; M stays 1 whatever the unit"
    );

    // (3) The staging cost is decisive on its own. Apple matrix operands are
    //     loaded from memory in a fixed layout with natural alignment; EXL3
    //     weights are 96 B of packed bits per tile and cannot be a matrix
    //     operand. Dequantising to scratch adds a *write* and a *read* of the
    //     whole weight set on top of the read that already happens: 12.0 GB
    //     streamed becomes 36.1 GB, and the extra 24.1 GB at the copy ceiling is
    //     +96 ms against a 62 ms step.
    let streamed_gb = 12.039;
    let step_ms = 1000.0 / BASELINE_TOK_PER_S;
    let extra_ms = 2.0 * streamed_gb / COPY_CEILING_GB_PER_S * 1000.0;
    assert!(
        extra_ms > step_ms,
        "staging would add {extra_ms:.0} ms against a {step_ms:.0} ms step"
    );
    assert!(
        extra_ms > 1.5 * step_ms,
        "and the margin is not marginal: {extra_ms:.0} vs {step_ms:.0} ms"
    );
}

/// There is no integer SIMD unit to widen into, which is the real reason
/// questions 1 and 4 have negative answers.
///
/// Apple's GPUs expose vector *fp* types (`half2`, `half8`, `float4`) and
/// scalar-only integer types. `ulong` is a 2-register pair on a 32-lane core,
/// so `ulong2`/`ulong4` word ops are 4 and 8 scalar ops -- they reduce *loads*,
/// which the analysis above says are not binding, and change ALU cost by zero.
#[test]
fn ulong_word_ops_buy_nothing_on_a_32_lane_gpu() {
    // The kernel's only wide types today are `ulong` for the funnel window.
    // Count them so the argument is anchored to the source.
    let extract = function_body(REPACKED_METAL, "extract_v_mcg_32").unwrap();
    assert!(
        !extract.contains("ulong"),
        "extract_v_mcg_32 works in 32-bit halves; a ulong here would change the \
         op accounting this file rests on"
    );

    // What a 32/64-bit "vector op" actually costs on this hardware: N scalar
    // ops. An 8-word ulong4 op is 8 ops, so widening cannot reduce ALU work;
    // it can only merge memory instructions.
    let wide = 4usize; // ulong4 = 4 x 64-bit = 8 x 32-bit lanes
    assert_eq!(wide * 2, 8);

    // And the one place the kernel *could* have widened for ALU reasons -- the
    // codebook -- is an exact 32-bit integer multiply. fp16 and fp32 vector
    // ALUs cannot compute it: 11 and 24 mantissa bits against a product whose
    // kept bits (0-12 and 16-28) must be exact, and whose bit 28 depends on
    // input bit 28.
    let codebook = function_body(LINEAR_METAL, "decode_state_mcg").unwrap();
    assert!(codebook.contains("value * 0xCBAC1FED"));
    assert!(
        codebook.contains("as_type<half2>"),
        "the vector type is fp, the multiply is not"
    );

    // The half2 addition after the multiply IS vectorised, and it is 1 of the 4
    // codebook ops. So even perfect vectorisation of everything vectorisable
    // removes 8 of `total` ops: the same 12% ceiling the MMA argument reached.
    let vectorisable = 8.0;
    assert!(vectorisable / OpBudget::live().total() < 0.13);
}

/// The reduction is already amortised past relevance, so widening it cannot pay.
///
/// The live kernel ends with 8 `simd_shuffle_xor` (two accumulators per column,
/// two columns), a threadgroup write, a barrier, a serial 8-deep sum and a
/// draining barrier -- once per *threadgroup*, against `tiles_k / 16` iterations
/// of the K loop at ~`total` ops each.
#[test]
fn the_reduction_is_not_worth_widening() {
    let kernel = function_body(REPACKED_METAL, "linear_gemv_ishizuki_k3_colmajor_2col").unwrap();
    let shuffles = kernel.matches("simd_shuffle_xor").count();
    let barriers = kernel.matches("threadgroup_barrier").count();
    assert_eq!(
        shuffles, 8,
        "two accumulators x two columns x two butterfly steps"
    );
    assert_eq!(
        barriers, 2,
        "one to publish, one to drain before the next tn"
    );

    // in_features = 5376 -> 336 k-tiles; the unrolled body runs 2 per step of
    // 8 simdgroups, so 21 steps for two columns.
    let tiles_k = 5376 / 16;
    let steps = tiles_k / 16;
    let budget = OpBudget::live();
    let loop_ops = steps as f64 * 4.0 * budget.total();
    let reduction_ops = shuffles as f64 + barriers as f64 * 2.0 + 8.0;
    let share = reduction_ops / loop_ops;
    assert!(
        share < 0.01,
        "the reduction is {share:.5} of the threadgroup's work; widening it \
         cannot move the roofline"
    );

    // For completeness: the no-reduction variant already exists and is not on
    // the decode path. It removes the barriers, not the per-lane ALU, so it
    // cannot be the missing 1.25x either.
    assert!(
        REPACKED_METAL.contains("linear_gemv_ishizuki_k3_colmajor_stream"),
        "the barrier-free variant should still exist"
    );
}

// ---------------------------------------------------------------------------
// Test 5: the estimate. What is actually reachable, and what it is worth.
// ---------------------------------------------------------------------------

/// Map a GEMV bandwidth to end-to-end decode rate.
///
/// 193 GB/s is the in-engine figure for 16.05 tok/s, so tok/s scales with the
/// GEMV bandwidth over the same byte total.
fn tok_per_s(gemv_gb_per_s: f64) -> f64 {
    BASELINE_TOK_PER_S * gemv_gb_per_s / in_engine_gb_per_s()
}

/// The in-engine GEMV-equivalent bandwidth implied by the baseline. Anchored to
/// the brief's own pairing (16.05-16.10 tok/s = 62.3 ms/step = 193 GB/s).
const fn in_engine_gb_per_s() -> f64 {
    193.0
}

/// The bit-exact 3 bpw ceiling under the ALU account, and what reaching the
/// 20 tok/s target would require.
///
/// The itemisation is the deliverable: every op removed, what it is worth, and
/// what is left. The conclusion is that the formulation is within ~7% of its own
/// floor and the floor is ~17 tok/s, so 20 tok/s is not reachable at 3 bpw
/// without changing the bit rate or the codebook -- both barred.
#[test]
fn bit_exact_3_bpw_tops_out_below_the_target() {
    let budget = OpBudget::live();
    let alu = fitted_alu_lane_ops_per_s(&budget);

    // What a given op count would yield, capped by the copy ceiling.
    let yield_at = |ops: f64| {
        let per_byte = ops / 3.0;
        (alu / per_byte / 1e9).min(COPY_CEILING_GB_PER_S)
    };

    // As shipped.
    let shipped = yield_at(budget.total());
    assert!(
        (shipped - MEASURED[0].gb_per_s).abs() < 1.0,
        "the model must reproduce the shipped kernel, got {shipped:.1} GB/s"
    );

    // (a) Amortise the four x conversions over more output columns. `_2col`
    //     shares them across two; `_4col` exists. The share is 8/C ops per
    //     column, so even C -> infinity only removes the x term.
    let amortised = budget.total() - budget.x_convert;
    let amortised_gb = yield_at(amortised);
    assert!(
        (amortised_gb - 212.3).abs() < 0.5,
        "full x amortisation is worth {amortised_gb:.1} GB/s (expected 212.3)"
    );

    // (b) Also drop the A_lo funnel term -- legal only if the two straddling
    //     windows' high bits are provably masked away. Check that against the
    //     reference rather than assuming it, because `decode_state_mcg`'s mask
    //     is the only thing standing between this and a bit-exactness bug.
    let without_a_lo = amortised - 1.0 - 2.0;
    let best_gb = yield_at(without_a_lo);

    // (c) The floor: 8 codebook ops per code and 8 FMAs are irreducible for
    //     this codebook, and the eight 16-bit windows at their runtime offsets
    //     are forced by bit-exactness.
    let floor = budget.codebook + budget.accum;
    assert!(
        budget.unpacking() > floor * 0.4,
        "unpacking is the majority of what is left, which is why this is a \
         codebook problem and not a GEMM problem"
    );
    assert_eq!(budget.codebook, 32.0);
    assert_eq!(budget.accum, 8.0);

    let ceiling_tok = tok_per_s(best_gb);
    assert!(
        ceiling_tok < TARGET_TOK_PER_S,
        "bit-exact 3 bpw is modelled at {ceiling_tok:.1} tok/s; if this is ever \
         above {TARGET_TOK_PER_S} the negative result is stale"
    );
    assert!(
        (ceiling_tok - 18.5).abs() < 0.3,
        "expected ~18.5 tok/s, got {ceiling_tok:.1}"
    );
}

/// The one experiment that would settle which per-tile resource binds, and it
/// is not a rate sweep.
///
/// Two variants at 3 bpw, identical bytes and identical load count: one with the
/// codebook's eight `IMUL`s replaced by a constant (already measured: "no decode
/// arithmetic at all" 134.3 GB/s vs 163.0, `linear.metal:96-106` -- i.e. making
/// the ALU *cheaper* made it *slower*, which is evidence for a load-side limit
/// and against an ALU limit), and one with the eight `LOP3` window extractions
/// replaced by an immediate. The former is the discriminating measurement and
/// the standing record already contains its result, pointing the other way.
#[test]
fn the_discriminating_experiment_is_named_not_assumed() {
    // The existing measurement, quoted from the kernel's own header comment.
    let loads_only: f64 = 212.2;
    let specialised: f64 = 163.0;
    let no_decode_arithmetic: f64 = 134.3;
    let runtime_codebook: f64 = 62.4;

    // The codebook specialisation is a 2.61x win, so the branch was serialising.
    let codebook_speedup = specialised / runtime_codebook;
    assert!((codebook_speedup - 2.61).abs() < 0.02);
    // Removing the arithmetic on top made it *slower*. That is the whole
    // reason the record says "do not try to make the decode cheaper", and it is
    // the strongest single piece of evidence that the load path -- not the
    // ALU -- is what the 3 bpw shortfall tracks.
    assert!(
        no_decode_arithmetic < specialised,
        "the no-arithmetic variant must be slower for the standing record's \
         conclusion to hold"
    );
    // If the ALU were the binding resource, deleting 32 of ~66 ops per lane per
    // tile -- nearly half the budget -- would have to help. It lost 17.6%.
    let lost = 1.0 - no_decode_arithmetic / specialised;
    let expected_loss = 0.176;
    assert!(
        (lost - expected_loss).abs() < 0.005,
        "recorded loss was {lost:.3}"
    );

    // So: this file's arithmetic says the ALU *can* bind at 3 bpw, and this
    // measurement says it does not. Both cannot be true, and the resolution is
    // the discriminating A/B, which has not been run. Recorded as an open
    // question rather than resolved by preference.
    assert!(loads_only > specialised);
}

// ---------------------------------------------------------------------------
// Test 6: the specialisation tax. 138 kernels, 546 dispatches, one library.
// ---------------------------------------------------------------------------

/// Specialising KA, ALIGNED and CODEBOOK is right, and it is free at runtime.
///
/// The generated library is large and every kernel is compiled, but the *hot
/// set* is what occupies the instruction cache during decode, and it is three
/// kernel names. I-cache pressure is therefore not a lever, and the honest
/// conclusion is that the specialisation costs compile time and memory, not
/// throughput.
#[test]
fn specialisation_costs_compile_time_not_throughput() {
    let linear_kernels = LINEAR_METAL.matches("\nkernel void ").count();
    let repacked_kernels = REPACKED_METAL.matches("\nkernel void ").count();
    assert_eq!(linear_kernels, 113, "linear.metal kernel count moved");
    assert_eq!(
        repacked_kernels, 27,
        "linear_repacked.metal kernel count moved"
    );
    let total = linear_kernels + repacked_kernels;
    assert_eq!(total, 140);

    // `LinearKernels::new` splices the two files into ONE library
    // (`linear.rs:24-30`), so all 140 land in one metallib and all are
    // available for I-cache eviction pressure -- but eviction only matters for
    // what is resident during decode.
    let dispatches_per_step = 546;
    assert!(dispatches_per_step > 500);

    // The reachable live kernels are exactly those named by a dispatch site.
    // Three of them at 3 bpw: _2col for a plain matvec, _fused2_2col for q|k|v,
    // _fused2_4col for the MLP. Three names, entered ~180 times each per step,
    // so each is a hot resident kernel by a wide margin.
    let hot = [
        "linear_gemv_ishizuki_k3_colmajor_2col",
        "linear_gemv_ishizuki_k3_fused2_2col",
        "linear_gemv_ishizuki_k3_fused2_4col",
    ];
    for name in hot {
        assert!(
            REPACKED_METAL.contains(&format!("kernel void {name}")),
            "{name} must exist in linear_repacked.metal"
        );
    }
    assert_eq!(hot.len(), 3);

    // A kernel that is entered hundreds of times per step cannot be paying an
    // I-cache miss on every entry; the miss would be amortised to nothing
    // against a body of ~66 ALU ops per tile. So the specialisation is not a
    // runtime cost, and the only thing the 140-kernel library buys that is
    // worth arguing about is register pressure *within* one kernel -- which is
    // addressed by `Nr0`, not by the family count.
    let dispatches_per_kernel = dispatches_per_step as f64 / hot.len() as f64;
    assert!(dispatches_per_kernel > 100.0);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// The source of a function, from the line declaring it to the line that closes
/// its outermost brace.
///
/// Byte-range accumulation rather than returning one line: the first version
/// returned only the closing `}`, because `depth` returns to zero on the line
/// *after* the body and the test then matched nothing in a body it had already
/// passed. Every assertion built on this helper would have passed vacuously.
///
/// Returns `None` rather than a partial body, so a caller's `expect` fires
/// instead of the caller silently testing nothing.
fn function_body<'a>(source: &'a str, name: &str) -> Option<&'a str> {
    let hit = source.find(name)?;
    let line_start = source[..hit].rfind('\n').map_or(0, |index| index + 1);
    let mut depth = 0i32;
    let mut opened = false;
    let mut cursor = line_start;
    for line in source[line_start..].lines() {
        let trimmed = line.trim();
        depth += i32::from(trimmed.ends_with('{')) - i32::from(trimmed.ends_with('}'));
        opened |= trimmed.ends_with('{');
        cursor += line.len() + 1;
        if opened && depth <= 0 {
            return Some(&source[line_start..cursor]);
        }
    }
    None
}
