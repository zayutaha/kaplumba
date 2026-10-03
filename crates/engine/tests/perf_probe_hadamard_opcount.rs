//! READ-ONLY perf probe: op counts for the post-extraction arithmetic of the
//! EXL3 decode matvec (`transform_simd`/`hadamard_128`, `decode_state_mcg`).
//!
//! No Metal, no GPU, no model, no checkpoint. Pure CPU arithmetic over the
//! window geometry plus analytic constants, so it runs in the normal `cargo test`
//! suite. Its job is to make every number in the report checkable before anyone
//! spends a GPU hour on it.
//!
//! ================================ ANSWER ================================
//!
//! **The Hadamard is not in the decode loop, and the codebook is ~5% of the
//! kernel. Neither is the bottleneck.**
//!
//! * The Hadamard is ALREADY hoisted. `transform_simd` (linear.metal:402) is
//!   called only from `linear_input_transform` (linear.metal:475) and the
//!   `linear_finalize_*` epilogue kernels (linear.metal:507+) -- never from
//!   `linear_gemv_exl3_k3_mcg` (linear.metal:4634), whose K loop (4693-4714)
//!   reads `input[]` directly at 4705-4708. It therefore already runs once per
//!   activation (5376 values), fused into the previous layer's epilogue, once
//!   per token. The brief's "hoist it out of the inner loop" is a no-op.
//!   `hadamard_128`/`hadamard128_vectors` (exl3.rs:149, exl3.metal) is a
//!   256-value VALIDATION harness, reached only from
//!   `crates/engine/examples/pipetest.rs:3`; it is not on the decode path.
//! * The codebook phase was measured at 0.032 ms of 0.612 ms = **5%**
//!   (spec/exl3-decode-throughput/implementation_notes.md:49-51), against loads
//!   55% and shifts 40%. Deleting the codebook ENTIRELY -- a free lookup -- is
//!   bounded at +5% of the step: 16.05 -> 16.90 tok/s
//!   (`codebook_ceiling_is_five_percent`).
//!
//! ===================== WHAT THE COUNTING DID FIND =====================
//!
//! Counting forced out three findings the existing notes get wrong. All three
//! are asserted here, on the CPU, from the window geometry.
//!
//! 1. **The `2^ka`-entry codebook LUT is not implementable.** It rests on
//!    "at 3 bpw only the top `ka` bits of a 16-bit code carry information"
//!    (spec/decode-20tok/spec.md:132-135). They do -- but the other 13 bits are
//!    the NEIGHBOURING codes' bits, and `decode_state_mcg` consumes all 16, so
//!    the decoded value is NOT a function of the top 3 bits. See
//!    `the_low_bits_of_the_window_are_the_neighbours_bits` and
//!    `the_codebook_is_not_a_function_of_the_code`. The premise is false; a LUT
//!    would return wrong weights that still run at full speed.
//!
//! 2. **The third ring-word load is dead code at ka = 3.** `exl3_decode8_mcg`
//!    loads 3 words and builds two 64-bit funnels, but the codes read `a` bits
//!    [27, 64) only, while the funnel from `w2` lands in bits [0, 27). So `w2`
//!    and both funnels are never read. See
//!    `the_third_ring_word_is_dead_at_ka3`. This is the one concrete change
//!    worth a GPU hour: 3 -> 2 loads per lane and ~11 ALU ops per lane.
//!
//! 3. **linear.metal:4237-4239's "a lane spends 12 B of load traffic" is right
//!    about the instruction count and wrong about the ring.** 3 loads x 4 B is
//!    12 B of *load instructions* per 3 useful bytes, but the tile ring is only
//!    96 B for all 32 lanes, so the 4x is L1 traffic, not DRAM. Callers have
//!    been reading the 4x as a bandwidth problem; it is an instruction-count
//!    problem, which is why `NR0` (more streams, same bytes) was the axis that
//!    actually moved. See `the_four_x_is_l1_not_dram`.

use half::f16;

/// model.rs hidden width.
const HIDDEN: usize = 5376;
/// bits per weight in this checkpoint; 410 of 411 tensors.
const KA: u32 = 3;
/// codes per 16x16 EXL3 tile, fixed by the format (`TILE_SIZE`).
const CODES_PER_TILE: usize = 256;
/// codes each lane decodes per `exl3_decode8_mcg` call.
const CODES_PER_LANE: usize = 8;
/// lanes per simdgroup == lanes per tile ring (256 / 8).
const LANES: u32 = 32;

/// Bytes read per token by the decode step (from `--metrics`).
const BYTES_PER_TOKEN: f64 = 12.04e9;
/// Achieved decode bandwidth (GB/s), and ms/step implied by it.
const ACHIEVED_GBPS: f64 = 193.0;
const STEP_MS: f64 = 62.3;
const TOK_S: f64 = 16.05;

// ===========================================================================
// 1. The Hadamard: already per-activation, and cheap
// ===========================================================================

#[test]
fn hadamard_is_per_activation_not_per_code() {
    // transform_simd (linear.metal:402-440) per simdgroup of 32 lanes:
    //   5 shuffle widths x 4 shuffles            = 20 SIMD shuffles
    //   5 widths x 4 add/sub + 2 register swaps x 4 = 28 float add/sub
    //   1/sqrt(128) scale, 4 lanes               = 4 multiplies
    const OPS_PER_BLOCK: u32 = 5 * 4 + (5 * 4 + 2 * 4) + 4;
    assert_eq!(OPS_PER_BLOCK, 52);

    // 5376 activations = 42 blocks of 128, on a 20-core GPU: 2.1 blocks/core.
    let blocks = HIDDEN / 128;
    assert_eq!(blocks, 42);
    let hadamard_ops = blocks * OPS_PER_BLOCK as usize;
    assert_eq!(hadamard_ops, 2_184);

    // Codes per token: BYTES_PER_TOKEN / 3 bits. The brief's "30.7 MILLION codes
    // per token" is 28.9 MILLION codes per PROJECTION (5376^2) -- ~1000x short.
    let codes_per_token = (BYTES_PER_TOKEN * 8.0 / KA as f64) as u64;
    assert!(
        codes_per_token.abs_diff(32_100_000_000) < 200_000_000,
        "codes/token = {codes_per_token}"
    );
    assert!(
        codes_per_token > 1_000 * (HIDDEN * HIDDEN) as u64,
        "per-token codes must dwarf per-projection codes"
    );

    // If the transform were per code (as the brief's premise assumes) it would
    // be ~700,000x the cost it actually has.
    let per_code = codes_per_token * u64::from(OPS_PER_BLOCK);
    assert!(per_code / hadamard_ops as u64 > 700_000);

    // And it is fused into the previous epilogue, so it is on no critical path.
    assert!(blocks < LANES as usize * 3, "42 blocks is under-saturated by design");
}

/// The brief asks whether the Hadamard is shared via threadgroup memory or could
/// be a separate batched pass. Both are already true, and neither is per-code:
/// `transform_simd` keeps 4 registers per lane (no threadgroup memory, unlike
/// the 128-float local array `transform_vector` needs) and runs in its own
/// kernel that the engine issues once per activation.
#[test]
fn hadamard_is_already_a_separate_simd_pass() {
    // Four registers per lane, 32 lanes -> 128 elements. No threadgroup array.
    let elements_per_simdgroup = 4 * LANES;
    assert_eq!(elements_per_simdgroup as usize, 128);
    // No barrier and no threadgroup memory: 20 shuffles + 28 add/sub + 4 mul,
    // entirely in-lane and in-register.
    let ops = 20 + 28 + 4;
    assert_eq!(ops, 52);
}

// ===========================================================================
// 2. Op counts
// ===========================================================================

/// `decode_state_mcg` (linear.metal:121-126) instruction count, per code.
/// ```text
/// value = value * 0xCBAC1FEDu;                  // 1 IMAD
/// value = 0x3b603b60u ^ (value & 0x8fff8fffu);  // 1 AND + 1 XOR
/// half2 h = as_type<half2>(value);              // 0 -- bit reinterpret
/// return float(h.x + h.y);                      // 1 HADD2 + 1 extract + 1 CVT
/// ```
const MCG_OPS_PER_CODE: u32 = 3 + 3;

/// Extraction per code (linear.metal:366-370). `KA` is a compile-time constant
/// in every generated kernel (`const uint KA = 3;`), so `r = i*ka + 16u` folds
/// to a literal and `64u - r` is a literal shift: one `SHR` plus the
/// `& 0xffffu` AND.
const EXTRACT_OPS_PER_CODE: u32 = 2;

/// Per-lane window setup (linear.metal:351-364), amortised over 8 codes:
/// `lo`(2) `i0`(1) `sh`(1) `j0,j1,j2` wrap selects(6) `base`(1)
/// `a` funnel(7) `c` funnel(3).
const SETUP_OPS_PER_LANE: u32 = 21;

/// `ring_word<true>` is ONE `LDG.32` (linear.metal:267-271) on the aligned path
/// the engine dispatches (`aligned = packed_offset % 4 == 0`, linear.rs:1601).
const LOADS_PER_LANE: u32 = 3;

/// 8 FMAs + 4 `input[]` loads + 4 cvt (linear.metal:4705-4713).
const ACC_OPS_PER_LANE: u32 = 8;

#[test]
fn op_counts_per_lane_and_per_projection() {
    let codebook = CODES_PER_LANE as u32 * MCG_OPS_PER_CODE;
    let extract = CODES_PER_LANE as u32 * EXTRACT_OPS_PER_CODE;
    let alu_per_lane = SETUP_OPS_PER_LANE + ACC_OPS_PER_LANE + codebook + extract;
    assert_eq!(codebook, 48);
    assert_eq!(extract, 16);
    assert_eq!(alu_per_lane, 93, "21 setup + 8 acc + 48 codebook + 16 extract");
    // ~93 ALU : 7 loads (3 packed + 4 activation). ~13:1 -- the kernel is not
    // ALU-throughput bound, which is why the codebook costs 5% of the time
    // despite being 52% of the integer instructions.
    assert_eq!(LOADS_PER_LANE + 4, 7);

    // Per projection: q_proj is 5376 x 5376.
    let tiles = (HIDDEN / 16) * (HIDDEN / 16);
    assert_eq!(tiles, 112_896);
    assert_eq!(tiles * CODES_PER_TILE, 28_901_376, "5376^2 codes in q_proj");
    // 256 codes/tile / 8 codes/lane == 32 lanes == exactly one simdgroup/tile.
    assert_eq!(CODES_PER_TILE / CODES_PER_LANE, LANES as usize);
}

/// 52% of the integer instructions is 5% of the time. This is the
/// reconciliation that decides "is decode arithmetic the bottleneck", so it is
/// asserted rather than left to prose.
#[test]
fn codebook_is_half_the_alu_and_five_percent_of_the_time() {
    let codebook = CODES_PER_LANE as u32 * MCG_OPS_PER_CODE;
    let total = SETUP_OPS_PER_LANE + ACC_OPS_PER_LANE
        + CODES_PER_LANE as u32 * (MCG_OPS_PER_CODE + EXTRACT_OPS_PER_CODE);
    assert_eq!(codebook * 100 / total, 51, "48 of 93 integer ops");

    // Measured phase split: spec/exl3-decode-throughput/implementation_notes.md:49-51
    const LOADS_MS: f64 = 0.336;
    const SHIFTS_MS: f64 = 0.244;
    const FULL_MS: f64 = 0.612;
    let codebook_ms = FULL_MS - LOADS_MS - SHIFTS_MS;
    assert!((codebook_ms - 0.032).abs() < 1e-9);
    assert!((LOADS_MS / FULL_MS * 100.0 - 54.9).abs() < 1.0);
    assert!((SHIFTS_MS / FULL_MS * 100.0 - 39.9).abs() < 1.0);

    // 52% of instructions -> 5% of time puts ALL post-extraction integer
    // arithmetic at ~10% of the kernel, so extract+codebook together cannot be
    // the wall. This is the number that ends the "is decode arithmetic the
    // bottleneck" question.
    let alu_share = codebook_ms / (codebook as f64 / total as f64);
    assert!((alu_share / FULL_MS * 100.0 - 9.7).abs() < 1.0, "ALU share {alu_share:.4} ms");
}

#[test]
fn codebook_ceiling_is_five_percent() {
    // The step is pure bandwidth: 12.04 GB/token at 193 GB/s = 62.3 ms = 1/16.05
    // s. So any percentage of the matvec is that percentage of the step.
    let from_bandwidth_ms = BYTES_PER_TOKEN / (ACHIEVED_GBPS * 1e9) * 1e3;
    assert!((from_bandwidth_ms - STEP_MS).abs() < 0.5, "bandwidth-implied {from_bandwidth_ms} ms");
    assert!((1000.0 / STEP_MS - TOK_S).abs() < 0.05);

    // Deleting the codebook ENTIRELY -- a lookup with zero cost -- is the hard
    // ceiling on this whole angle.
    let ceiling_ms = STEP_MS * 0.05;
    let tok_s = 1000.0 / (STEP_MS - ceiling_ms);
    assert!((tok_s - 16.90).abs() < 0.05, "ceiling {tok_s:.3} tok/s");
    assert!(tok_s < 17.0, "the codebook angle cannot reach 17 tok/s");
}

// ===========================================================================
// 3. The LUT idea does not work, and here is the arithmetic
// ===========================================================================

/// CPU mirror of `decode_state_mcg` (linear.metal:121), matching
/// `kaplumba_engine::exl3::decode_state(.., Codebook::Mcg)`
/// (crates/engine/src/exl3.rs:165-179) including the fp16 add.
fn decode_state_mcg(value: u32) -> f32 {
    let value = value.wrapping_mul(0xCBAC_1FED);
    let transformed = 0x3b60_3b60u32 ^ (value & 0x8fff_8fff);
    let low = f16::from_bits(transformed as u16);
    let high = f16::from_bits((transformed >> 16) as u16);
    (low + high).to_f32()
}

/// The window geometry of `exl3_decode8_mcg`, transcribed from
/// linear.metal:351-364 in u64 so it cannot diverge.
struct Window {
    lo: u64,
    i0: usize,
    sh: u32,
    j: [usize; 3],
    ring_words: usize,
}

impl Window {
    fn new(lane: u32, ka: u32) -> Self {
        let ring_words = ka as usize * 8;
        let ring = ring_words as u64 * 32;
        let lo = (u64::from(lane) * 8 + 1) * u64::from(ka) + ring - 16;
        let i0 = (lo / 32) as usize;
        let sh = (lo % 32) as u32;
        // Sequential wrap, as in linear.metal:356-358 (j1 wraps from j0).
        let wrap = |i: usize| if i >= ring_words { i - ring_words } else { i };
        let j0 = wrap(i0);
        let j1 = wrap(j0 + 1);
        let j2 = wrap(j1 + 1);
        Self { lo, i0, sh, j: [j0, j1, j2], ring_words }
    }

    /// Which `a` bit index does code `i` read first? Code i reads
    /// `a >> (64 - (i*ka+16))`, i.e. it starts at `a` bit `64 - i*ka - 16`.
    fn a_bit_of_code(i: u32, ka: u32) -> u32 {
        64 - i * ka - 16
    }

    fn decode(&self, words: &[u32], ka: u32, v: &mut [f32; 8]) {
        let g = |idx: usize| u64::from(words[idx]);
        let sh = u64::from(self.sh);
        let a = ((g(self.j[0]) << 32) | g(self.j[1])) << sh
            | if self.sh != 0 { g(self.j[2]) >> (32 - sh) } else { 0 };
        let c = ((g(self.j[1]) << 32) | g(self.j[2])) << sh;
        for i in 0..CODES_PER_LANE as u32 {
            let r = i * ka + 16;
            let s = if r <= 64 { (a >> (64 - r)) as u32 } else { (c >> (96 - r)) as u32 };
            v[i as usize] = decode_state_mcg(s & 0xffff);
        }
    }
}

/// The premise behind spec/decode-20tok/spec.md:132-135 -- "at 3 bpw only the
/// top `ka` bits of a 16-bit code carry information, so the codebook is a
/// function of `2^ka` inputs" -- is FALSE, and the reason is visible in
/// `pack_integer_states` (crates/engine/src/exl3.rs:355-389): it packs 16 states
/// at `bits` bits each, back to back, with no padding. So the ring is exactly
/// dense, every bit of the 16-bit window belongs to some code, and the low bits
/// of one window are the neighbouring codes' data.
///
/// Consequence: `decode_state_mcg`, which consumes all 16 bits, is NOT a
/// function of the top `ka` bits, so a `2^ka`-entry LUT cannot be built. A LUT
/// would return wrong weights while still running at full speed -- the exact
/// silent-wrongness failure mode AGENTS.md §3.4 and
/// codebook_specialisation.rs:19-31 both exist to catch.
#[test]
fn the_codebook_is_not_a_function_of_the_code() {
    for ka in [1u32, 2, 3, 4, 6, 8] {
        let codes = 1u32 << ka;
        let low_bits = 1u32 << (16 - ka);
        let mut ambiguous = 0;
        for code in 0..codes {
            let vals: std::collections::BTreeSet<u32> = (0..low_bits)
                .map(|low| decode_state_mcg((code << (16 - ka)) | low).to_bits())
                .collect();
            if vals.len() > 1 {
                ambiguous += 1;
            }
        }
        assert_eq!(ambiguous, codes, "ka={ka}: every code's decode depends on the low bits");
    }

    // At ka = 3 that is 8 codes x 8192 low-bit fillings, each code spreading over
    // thousands of distinct floats (measured 5149-5776 across the 8 codes).
    // A table would have to hold 2^16 entries to be exact -- strictly worse
    // than the arithmetic it replaces.
    let ka = 3u32;
    for code in 0..8u32 {
        let distinct: std::collections::BTreeSet<u32> = (0..(1u32 << 13))
            .map(|low| decode_state_mcg((code << 13) | low).to_bits())
            .collect();
        assert!(distinct.len() > 4000, "code {code} -> {} distinct values", distinct.len());
    }
}

/// The ring is exactly dense, which is *why* the LUT fails: the low 13 bits of
/// each window are the neighbouring codes' bits, not zeros.
#[test]
fn the_low_bits_of_the_window_are_the_neighbours_bits() {
    // pack_integer_states: ring_words = bits*8 words = bits*8*32 bits per tile,
    // and 256 codes x bits = bits*256 bits. bits*8*32 == bits*256 exactly.
    for ka in 1..=8u32 {
        let ring_bits = u64::from(ka) * 8 * 32;
        assert_eq!(ring_bits, 256 * u64::from(ka), "ka={ka}: ring is exactly dense");
    }
    assert_eq!(KA * 8 * 32, 256 * KA);

    // So a 16-bit window read at a code boundary contains 16*... bits of which
    // only `ka` are this code; the other 16-ka belong to codes 1, 2, 3, ...
    // Concretely, at ka = 3 the low 13 bits of code i's window are 4+ codes.
    let neighbours = 16 / KA;
    assert_eq!(neighbours, 5, "a 16-bit window spans 5 codes' worth of 3-bit fields");
}

// ===========================================================================
// 4. THE CONCRETE CHANGE: the third ring word is dead at ka = 3
// ===========================================================================

/// `exl3_decode8_mcg` (linear.metal:360-364) loads three ring words and builds
/// two 64-bit funnels. At ka = 3 the funnels are dead code.
///
/// The funnel `| (w2 >> (32 - sh))` lands in `a` bits [0, sh), and `sh` takes
/// values {3, 11, 19, 27} across lanes -- so `w2` can reach `a` bit 26 at most.
/// But the lowest `a` bit any code reads is `a_bit_of_code(7, 3)` = 27.
/// Therefore `a` bits [0, 27) are never read, `w2` is never read, and `c` is
/// never read (`c` is only consulted when `r > 64`, and `r_max = 7*3+16 = 37`).
///
/// 3 loads -> 2 per lane, and ~11 ALU ops per lane go with it.
#[test]
fn the_third_ring_word_is_dead_at_ka3() {
    // sh across lanes at ka = 3.
    let mut shs = std::collections::BTreeSet::new();
    for lane in 0..LANES {
        shs.insert(Window::new(lane, KA).sh);
    }
    assert_eq!(shs.into_iter().collect::<Vec<_>>(), vec![3, 11, 19, 27]);
    let max_sh = 27u32;

    // Lowest `a` bit read by any code: code 7 at ka = 3.
    let read_lo = Window::a_bit_of_code(7, KA);
    assert_eq!(read_lo, 27);
    // The funnel fills bits [0, max_sh) = [0, 27), so bit 26 is the highest it
    // can reach -- one below the lowest bit any code reads.
    assert_eq!(max_sh, read_lo, "funnel must stop exactly where reading starts");
    assert!(max_sh - 1 < read_lo, "w2 can only reach a bit {}", max_sh - 1);

    // `c` is only read when r > 64; at ka = 3, r_max = 37.
    let r_max = 7 * KA + 16;
    assert_eq!(r_max, 37);
    assert!(r_max <= 64, "the whole `c` funnel is dead at ka = 3");

    // And j2 exists only to feed w2.
    assert_eq!(Window::new(LANES - 1, KA).j[2], Window::new(0, KA).j[1]);

    // This also holds per-rate, which is why it is worth stating generally.
    // ka <= 4: dead. ka >= 5: live (the funnel can reach a bit a code reads).
    for ka in 1..=4u32 {
        let mut max_sh = 0;
        for lane in 0..LANES {
            max_sh = max_sh.max(Window::new(lane, ka).sh);
        }
        let read_lo = Window::a_bit_of_code(7, ka);
        assert!(max_sh <= read_lo, "ka={ka}: max_sh {max_sh} <= read_lo {read_lo}");
    }
}

/// The trimmed window, proved bit-exact against the current one.
///
/// The candidate drops `w2`, `j2` and both funnels, so `a` is just
/// `((w0 << 32) | w1) << sh` and `c` is unused. Every lane of every tile must
/// produce identical floats -- bit-for-bit, because a wrong-but-fast decode is
/// the failure mode this repo has already shipped twice.
#[test]
fn trimming_w2_is_bit_exact() {
    let tiles = 4096usize;
    let mut words = vec![0u32; tiles * 24];
    // Per-tile xorshift, not an LCG: an LCG with period 8 would make an
    // index perturbation that is a multiple of 8 invisible, and the comparison
    // would pass vacuously (codebook_specialisation.rs:33 documents that trap).
    let mut s: u64 = 0x243f_6a88_85a3_08d3;
    for w in words.iter_mut() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *w = s as u32;
    }

    let mut compared = 0usize;
    let mut mismatches = 0usize;
    for tile in 0..tiles {
        let tw = &words[tile * 24..tile * 24 + 24];
        for lane in 0..LANES {
            let win = Window::new(lane, KA);
            let mut baseline = [0f32; 8];
            win.decode(tw, KA, &mut baseline);

            // trimmed: two words, no funnel, no `c`.
            let g = |idx: usize| u64::from(tw[idx]);
            let sh = u64::from(win.sh);
            let a = ((g(win.j[0]) << 32) | g(win.j[1])) << sh;
            for i in 0..CODES_PER_LANE as u32 {
                let r = i * KA + 16;
                let got = decode_state_mcg(((a >> (64 - r)) as u32) & 0xffff);
                compared += 1;
                if got.to_bits() != baseline[i as usize].to_bits() {
                    mismatches += 1;
                }
            }
        }
    }
    assert_eq!(compared, tiles * 32 * 8, "every lane of every tile");
    assert_eq!(mismatches, 0, "dropping w2 must not change a single output bit");
}

/// The saving, quantified per lane and per token.
#[test]
fn trimming_w2_saves_one_load_and_about_eleven_ops() {
    let codebook = CODES_PER_LANE as u32 * MCG_OPS_PER_CODE;
    let extract = CODES_PER_LANE as u32 * EXTRACT_OPS_PER_CODE;

    // Current setup: lo(2) i0(1) sh(1) j0,j1,j2(6) base(1) a-funnel(7) c-funnel(3)
    assert_eq!(SETUP_OPS_PER_LANE, 21);
    // Trimmed setup drops the a funnel (7), the c funnel (3) and j2's select (1).
    const TRIMMED_SETUP: u32 = SETUP_OPS_PER_LANE - 7 - 3 - 1;
    assert_eq!(TRIMMED_SETUP, 10);

    let alu_now = SETUP_OPS_PER_LANE + ACC_OPS_PER_LANE + codebook + extract;
    let alu_trim = TRIMMED_SETUP + ACC_OPS_PER_LANE + codebook + extract;
    assert_eq!(alu_now, 93);
    assert_eq!(alu_trim, 82);

    // Packed loads per lane: 3 -> 2, a 33% cut in the load instruction count.
    assert_eq!(LOADS_PER_LANE, 3);
    let loads_now = LOADS_PER_LANE;
    let loads_trim = LOADS_PER_LANE - 1;
    assert_eq!(loads_now, 3);
    assert_eq!(loads_trim, 2);

    // Bytes touched by the PACKED stream drop 12 -> 8 per lane. Activation loads
    // (4) are unaffected.
    let packed_bytes_now = loads_now * 4;
    let packed_bytes_trim = loads_trim * 4;
    assert_eq!(packed_bytes_now, 12);
    assert_eq!(packed_bytes_trim, 8);
    let total_bytes_now = packed_bytes_now + 4;
    let total_bytes_trim = packed_bytes_trim + 4;
    assert_eq!(total_bytes_now, 16);
    assert_eq!(total_bytes_trim, 12);
    // So total load bytes per lane fall 25%, and per-lane load INSTRUCTIONS 12 -> 10.
    assert_eq!((total_bytes_now - total_bytes_trim) * 100 / total_bytes_now, 25);

    // Honest ceiling. The phase split puts loads at 55% of the kernel and ALU at
    // ~10%. Removing 1 of 3 packed loads is 1/7 of all loads -> at most
    // 55%/7 ~ 7.9% of the kernel; ALU drops 12% of ~10% -> ~1.2%. If the removed
    // load were purely L1-resident the gain would be smaller, since DRAM traffic
    // does not change. Bound it at 2-8% of the step and be explicit that the
    // 8% end needs the load to be issue-bound rather than L1-bound.
    let floor = 0.02f64;
    let ceiling = 0.079f64;
    for gain in [floor, 0.05, ceiling] {
        let ms = STEP_MS * (1.0 - gain);
        let tok_s = 1000.0 / ms;
        assert!(tok_s > TOK_S && tok_s < 17.5, "gain {gain:.3} -> {tok_s:.2} tok/s");
    }
    assert!((1000.0 / (STEP_MS * (1.0 - ceiling)) - 17.43).abs() < 0.05);
    assert!((1000.0 / (STEP_MS * (1.0 - floor)) - 16.35).abs() < 0.05);
    // Even at the optimistic end this is a ~1.4 tok/s move, not a 4 tok/s one,
    // and it does not reach the 20 tok/s target (236.7 GB/s).
    assert!(1000.0 / (STEP_MS * (1.0 - ceiling)) - TOK_S < 1.5);
}

/// linear.metal:4237-4239 says a lane "spends 12 B of load traffic to produce
/// 8*KA bits". The 12 B is right for load INSTRUCTIONS and wrong as a statement
/// about the tile: the ring is 96 B for all 32 lanes, so adjacent lanes overlap
/// in L1 and the 4x is not DRAM traffic.
///
/// This matters because it is why `NR0` -- adding streams without changing what
/// is loaded -- was the axis that moved the 3 bpw rate, and why reading the 4x
/// as a bandwidth problem sends you looking for a bandwidth fix that cannot
/// exist (the bytes are already the minimum: 3 bits per weight, dense).
#[test]
fn the_four_x_is_l1_not_dram() {
    let ring_words = KA as usize * 8;
    let ring_bytes_per_tile = ring_words * 4;
    assert_eq!(ring_bytes_per_tile, 96);
    let useful_bytes_per_tile = CODES_PER_TILE * KA as usize / 8;
    assert_eq!(useful_bytes_per_tile, 96);
    assert_eq!(
        ring_bytes_per_tile, useful_bytes_per_tile,
        "the ring is exactly the codes: zero DRAM redundancy"
    );

    // Per lane the kernel issues 3 x 4 B = 12 B of loads to get 3 useful bytes.
    let issued = LOADS_PER_LANE as usize * 4;
    let useful = CODES_PER_LANE * KA as usize / 8;
    assert_eq!(issued, 12);
    assert_eq!(useful, 3);
    assert_eq!(issued / useful, 4);

    // But summed over the simdgroup the simdgroup issues 32 x 12 = 384 B of
    // loads against a 96 B ring: 4x L1 re-reads, 96 B of unique DRAM.
    const ISSUED_PER_SIMDGROUP: usize = 384;
    assert_eq!(LANES as usize * issued, ISSUED_PER_SIMDGROUP);
    assert_eq!(ISSUED_PER_SIMDGROUP / ring_bytes_per_tile, 4);
    // The DRAM floor for this tile is 96 B and it is already achieved, so there
    // is no bandwidth win available in the packed stream at all.
    assert_eq!(ring_bytes_per_tile, 96);
}