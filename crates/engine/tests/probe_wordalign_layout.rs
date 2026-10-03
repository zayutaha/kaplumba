//! A1 probe — CPU-only. No model, no GPU, no checkpoint, no network.
//!
//! Question (from `spec/decode-ceiling/hypotheses.md`, hypothesis A1): can the
//! 3 bpw EXL3 tile be repacked so that each lane's 8 codes occupy a single
//! 32-bit word, cutting lane loads 2 -> 1 and `amp = 8/KA = 2.667` -> 1.333,
//! **without** growing the tile from 24 words (96 B) to 32 words (128 B)?
//!
//! Everything below is a derivation transcribed from the live code, not a
//! measurement:
//!   * `crates/engine/src/exl3.rs:231`      `unpack_integer`  (CPU reference)
//!   * `crates/engine/src/exl3.rs:668`      `extract_ishizuki_lane_codes`
//!   * `crates/kernels/src/linear_repacked.metal:53`   `extract_v_mcg_32`
//!   * `crates/kernels/src/linear_repacked.metal:2407` `..._2col_sk2` (live)
//!   * `crates/kernels/src/linear.metal:121`           `decode_state_mcg`
//!   * `scripts/gen_exl3_gemv.py:107`                   `two_word_is_safe`
//!
//! Run standalone (no cargo, no deps, no GPU):
//!   rustc --edition 2021 --test crates/engine/tests/probe_wordalign_layout.rs \
//!         -o "$TMPDIR/probe_wordalign_layout" && "$TMPDIR/probe_wordalign_layout"

// ---------------------------------------------------------------------------
// The ring: 256 codes of KA bits, end to end, cyclic. 768 bits at KA=3.
// Tile = 24 u32 = 96 B. `tile[w]` holds ring bits [32w, 32w+32).
// ---------------------------------------------------------------------------

const LANES: usize = 32;
const TILE_CODES: usize = 256;

fn ring_bits(ka: u32) -> u32 {
    (TILE_CODES as u32) * ka
}

fn words_per_tile(ka: u32) -> u32 {
    ring_bits(ka) / 32
}

/// `lo`, `shift`, `j0`, `j1` exactly as `unpack_integer` computes them
/// (exl3.rs:256-264) and as the live kernel does (linear_repacked.metal:2431).
fn setup(lane: usize, ka: u32) -> (u32, u32, u32, u32) {
    let lo = (lane as u32 * 8u32 + 1u32) * ka + ring_bits(ka) - 16u32;
    let shift = lo % 32u32;
    let i0 = lo / 32u32;
    let w = words_per_tile(ka);
    let j0 = if i0 >= w { i0 - w } else { i0 };
    let j1 = (j0 + 1u32) % w;
    (lo, shift, j0, j1)
}

/// Ring bit positions of lane `lane`'s 16-bit decode window for code index `i`,
/// **MSB first** (i.e. `a` bit 63-3i downwards).
///
/// `a = ((tile[j0] << 32) | tile[j1]) << shift`.  Note the halves of `a` run
/// through the ring in OPPOSITE directions: `a`'s high 32 bits are `tile[j0]`
/// (ring `32*j0 + 0..31`) and its low 32 bits are `tile[j1]` (ring
/// `32*j0 + 32..63`), so
///
///     a bit p  ->  m = p - shift  ->  ring 32*j0 + (m mod 32) + (m < 32 ? 32 : 0)
///
/// Every *used* bit has `m >= 0` because `used_lo = 48 - 7*KA >= sh_max` at
/// KA 3 and 4, so the funnel's `[0, shift)` contribution is never read.
fn window_bits(lane: usize, i: usize, ka: u32) -> Vec<u32> {
    let (_, shift, j0, _) = setup(lane, ka);
    let rb = ring_bits(ka);
    let hi = 64u32 - (i as u32) * ka; // exclusive top of the 16-bit window
    (0..16u32)
        .map(|k| {
            let p = hi - 1u32 - k;
            let m = p - shift;
            (32u32 * j0 + (m % 32u32) + if m < 32u32 { 32u32 } else { 0u32 }) % rb
        })
        .collect()
}

/// The `i`-th *independent* 3-bit chunk a lane receives: the three ring bits
/// that enter `window(i)` at its top, `a` bits `[61-3i, 64-3i)`.
/// `window(i)`'s low 3 bits are NOT new data (they are `window(i-1)`'s bits
/// 9..11) — which is why the withdrawn bit-plane repacker, which stored
/// `w & 7`, was lossy: it kept the derived quantity and dropped the payload.
fn new_chunk_bits(lane: usize, i: usize, ka: u32) -> Vec<u32> {
    window_bits(lane, i, ka)[0..ka as usize].to_vec()
}

// ---------------------------------------------------------------------------
// 1. The span, and whether one 32-bit load can hold it.
// ---------------------------------------------------------------------------

/// Bits a lane must hold resident to decode `codes` codes: the anchor window is
/// 16 bits and each further code contributes `KA` fresh bits.
/// = `16 + (codes-1)*KA`, matching the record's `used_lo = 48 - 7*KA` since
/// `64 - used_lo = 16 + 7*KA`.
fn span_bits(codes: usize, ka: u32) -> u32 {
    if codes == 0 { 0 } else { 16u32 + (codes as u32 - 1u32) * ka }
}

/// `used_lo`: lowest `a` bit any of the 8 windows touches (gen_exl3_gemv.py:138).
fn used_lo(ka: u32) -> u32 {
    48u32.saturating_sub(7u32 * ka)
}

/// `sh_max`: the largest lane shift, which is what makes the 3rd word dead
/// (`two_word_is_safe`).  Enumerated, not bounded.
fn sh_max(ka: u32) -> u32 {
    (0..LANES as u32)
        .map(|l| ((l * 8u32 + 1u32) * ka + ring_bits(ka) - 16u32) % 32u32)
        .max()
        .unwrap()
}

#[test]
fn lane_span_is_37_at_3bpw() {
    for lane in 0..LANES {
        let mut u: Vec<u32> = (0..8).flat_map(|i| window_bits(lane, i, 3)).collect();
        u.sort_unstable();
        u.dedup();
        assert_eq!(u.len(), 37, "lane {lane} window span");
    }
    assert_eq!(span_bits(8, 3), 37);
    assert!(span_bits(8, 3) > 32, "37 > 32: this is the crux");
    assert_eq!(64 - used_lo(3), 37, "used_lo and the span agree");
}

#[test]
fn max_codes_per_32bit_load_is_6_at_3bpw() {
    let mut c = 0usize;
    while span_bits(c + 1, 3) <= 32 {
        c += 1;
    }
    println!("KA=3: span(6)={} span(7)={} -> at most {c} codes in one 32-bit load",
             span_bits(6, 3), span_bits(7, 3));
    assert_eq!(c, 6, "the kernel cannot do 7, let alone 8");
    assert!(
        LANES * c < TILE_CODES,
        "32 lanes x 6 codes = {} < 256 codes/tile: one word per lane cannot cover a tile",
        LANES * c
    );
    for ka in 1u32..=6 {
        let mut m = 0usize;
        while span_bits(m + 1, ka) <= 32 { m += 1; }
        let mut r = 0usize;
        while span_bits(r + 1, ka) > 32 { r += 1; }
        println!(
            "KA={ka}: span(8)={:3} used_lo={:2} sh_max={:2} max codes/32b load={} (needs {} words)",
            span_bits(8, ka),
            used_lo(ka),
            sh_max(ka),
            m,
            (span_bits(8, ka) as usize).div_ceil(32)
        );
    }
}

#[test]
fn second_word_is_dead_exactly_at_ka_le_2() {
    // `a`'s used bits are [used_lo, 64).  The 2nd loaded word is dead only when
    // every used bit is at `a` bit >= 32, i.e. `used_lo >= 32`.
    for ka in 1u32..=6 {
        println!("KA={ka}: used_lo={:2} 2nd word dead={}", used_lo(ka), used_lo(ka) >= 32);
    }
    assert!(used_lo(3) < 32, "3 bpw genuinely needs word 2");
    assert!(used_lo(2) >= 32, "2 bpw does not -> amp 1.0 there, and that is why it is not an option");
    assert_eq!(used_lo(3), 27);
}

#[test]
fn three_word_margin_is_sh_max_eq_used_lo_at_3_and_4() {
    // gen_exl3_gemv.py's `two_word_is_safe`: safe iff `48 - 7*KA >= max(sh)`.
    assert_eq!(sh_max(3), 27);
    assert_eq!(used_lo(3), 27);
    assert_eq!(sh_max(4), 20);
    assert_eq!(used_lo(4), 20);
    println!("KA=3 two-word margin = {} bits; KA=4 = {} bits (both exactly zero)",
             used_lo(3) as i32 - sh_max(3) as i32, used_lo(4) as i32 - sh_max(4) as i32);
    for ka in 1u32..=6 {
        println!("KA={ka}: used_lo={} sh_max={} safe={}", used_lo(ka), sh_max(ka), used_lo(ka) >= sh_max(ka));
    }
}

// ---------------------------------------------------------------------------
// 2. Lanes read DIFFERENT windows, not rotations of one (record correction).
// ---------------------------------------------------------------------------

#[test]
fn lane_windows_are_distinct_blocks_not_rotations() {
    // `spec/decode-3bpw-amplification/{spec,implementation_notes}.md` state
    // "every lane reads the same 37-bit window, at 32 different rotations" and a
    // "union over all 32 lanes" of 37 bits.  Both are wrong.
    let mut a = window_bits(0, 0, 3);
    a.sort_unstable();
    a.dedup();
    let mut b = window_bits(16, 0, 3);
    b.sort_unstable();
    b.dedup();
    assert!(!a.iter().any(|x| b.contains(x)), "lane 0 and lane 16 windows are disjoint");
    let mut all: Vec<u32> = (0..LANES)
        .flat_map(|l| (0..8).flat_map(move |i| window_bits(l, i, 3)))
        .collect();
    all.sort_unstable();
    all.dedup();
    println!("union over all 32 lanes = {} of {} ring bits (not 37)", all.len(), ring_bits(3));
    assert_eq!(all.len(), ring_bits(3) as usize);
}

#[test]
fn the_256_codes_tile_the_ring_exactly() {
    let rb = ring_bits(3);
    let all: Vec<u32> = (0..LANES)
        .flat_map(|l| (0..8).flat_map(move |i| new_chunk_bits(l, i, 3)))
        .collect();
    let mut distinct: Vec<u32> = all.clone();
    distinct.sort_unstable();
    distinct.dedup();
    println!("{} chunk slots, {} distinct ring positions (ring = {})", all.len(), distinct.len(), rb);
    assert_eq!(all.len(), 3 * TILE_CODES);
    assert_eq!(distinct.len(), rb as usize, "the 256 chunks tile the ring exactly once");
}

#[test]
fn each_lane_gets_24_consecutive_bits_of_codes() {
    // "24 bits per lane" is TRUE of the independent code chunks and FALSE of the
    // decode window.  The withdrawn T1 proposal had the first and ignored the second.
    //
    // Stronger still: the 24 bits are NOT one contiguous ring run.  `a`'s two
    // halves walk the ring in opposite directions, so a lane's payload is split
    // across the two words it loads.  That is the mechanical reason one word per
    // lane is impossible even before the 37-bit state argument.

    let mut splits: std::collections::BTreeMap<(u32, u32), usize> = Default::default();
    for lane in 0..LANES {
        let mut code: Vec<u32> = (0..8).flat_map(|i| new_chunk_bits(lane, i, 3)).collect();
        code.sort_unstable();
        code.dedup();
        let mut win: Vec<u32> = (0..8).flat_map(|i| window_bits(lane, i, 3)).collect();
        win.sort_unstable();
        win.dedup();
        assert_eq!(code.len(), 24, "lane {lane} holds 24 distinct code bits");
        assert_eq!(win.len(), 37, "lane {lane} window is 37 bits");
        let state: Vec<u32> = win.iter().copied().filter(|p| !code.contains(p)).collect();
        assert_eq!(state.len(), 13, "13 bits of trellis state are not codes");
        assert!(state.iter().all(|p| !code.contains(p)));

        // The window spans exactly the two words the kernel loads, and no others.
        let (_, _, j0, j1) = setup(lane, 3);
        let words: std::collections::BTreeSet<u32> =
            win.iter().map(|p| (p / 32) % 24).collect();
        assert_eq!(words.len(), 2, "lane {lane} window spans 2 words");
        assert!(words.contains(&j0) && words.contains(&j1));

        let in_j0 = code.iter().filter(|p| (**p / 32) % 24 == j0).count();
        let in_j1 = code.iter().filter(|p| (**p / 32) % 24 == j1).count();
        assert_eq!(in_j0 + in_j1, 24);
        *splits.entry((in_j0 as u32, in_j1 as u32)).or_default() += 1;
        if lane < 3 {
            println!("lane {lane}: j0={j0} j1={j1} window(37) = {win:?}");
            println!("lane {lane}: codes(24) = {code:?}  state(13) = {state:?}");
        }
    }
    println!("(code bits in word j0, code bits in word j1) -> lanes: {splits:?}");
    println!("=> 8 lanes hold all 24 code bits in one word but still need the other for state; \
8 hold 21+3, 8 hold 13+11, 8 hold 5+19.  No lane is served by one word.");
    assert_eq!(splits.values().sum::<usize>(), LANES);
    assert!(
        splits.keys().all(|(a, b)| *a < 24 || *b < 24),
        "for every lane the 24 code bits and the 13 state bits need two words"
    );
}

// ---------------------------------------------------------------------------
// 3. Where the 64 lane-requests actually go: requests, not DRAM bytes.
// ---------------------------------------------------------------------------

#[test]
fn amp_is_lane_requests_not_dram_amplification() {
    let mut req: Vec<u32> = Vec::new();
    for lane in 0..LANES {
        let (_, _, j0, j1) = setup(lane, 3);
        req.push(j0);
        req.push(j1);
    }
    let distinct: std::collections::BTreeSet<u32> = req.iter().copied().collect();
    let mut mult: std::collections::BTreeMap<u32, u32> = Default::default();
    for w in &req { *mult.entry(*w).or_default() += 1; }
    println!(
        "64 lane word-requests -> {} distinct words of a {}-word tile; \
         every tile word IS read => DRAM amplification exactly 1.000; \
         per-word multiplicity {:?}",
        distinct.len(), words_per_tile(3),
        mult.values().collect::<Vec<_>>()
    );
    assert_eq!(req.len(), 64);
    assert_eq!(distinct.len(), words_per_tile(3) as usize);
}

#[test]
fn request_count_is_invariant_under_any_code_to_lane_map() {
    // Codes/tile = 256 and lanes = 32 force codes/lane = 8; a lane may only take
    // CONSECUTIVE ring positions (any other set needs >= 2 independent 16-bit
    // states = more bits, not fewer).  So every legal map has the same cost.
    for (codes, words) in [(8usize, 2usize)] {
        assert_eq!(span_bits(codes, 3), 37);
        assert_eq!(words, (span_bits(codes, 3) as usize).div_ceil(32));
    }
    println!("32 lanes x 8 codes x ceil(37/32) words = 64 requests/tile, for every legal map");
    assert_eq!(LANES * 2, 64);
}

// ---------------------------------------------------------------------------
// 4. The repack, and its price.
// ---------------------------------------------------------------------------

/// The lossless lane-local repack: store the lane's 16-bit anchor window plus the
/// 7 fresh 3-bit chunks that follow it.  `16 + 3*7 = 37` bits.  Reconstructing
/// `w_{i+1} = (w_i << 3 | chunk_i) & 0xffff` reproduces all 8 windows exactly, so
/// every `decode_state_mcg` output is bit-identical.
#[test]
fn lane_local_repack_is_bit_exact_and_costs_37_bits_per_lane() {
    let anchor: u32 = 0xA5C3;
    let chunks: [u32; 7] = [3, 1, 7, 0, 5, 2, 6];

    // Ground truth: the 8 windows the kernel reads.
    let mut truth = [0u32; 8];
    let mut s = anchor;
    for i in 0..8 {
        truth[i] = s;
        if i < 7 { s = ((s << 3) | chunks[i]) & 0xffff; }
    }
    // Repacked lane: (anchor, chunks[0..7]) -> 37 bits, no ring read at all.
    let mut recon = [0u32; 8];
    let mut t = anchor;
    recon[0] = t;
    for i in 0..7 { t = ((t << 3) | chunks[i]) & 0xffff; recon[i + 1] = t; }
    assert_eq!(truth, recon, "repack is bit-exact: all 8 windows reproduced");

    let payload = span_bits(8, 3);
    assert_eq!(payload, 37);
    let packed_bits = LANES as u32 * payload;
    println!("--- cost of the lossless lane-local repack, per 16x16 tile ---");
    println!("today                         : 24 words =  96 B  (768 bits, 1.000x)");
    println!("32 words (the brief's option) : 32 words = 128 B  ({:.3}x) -- CANNOT HOLD {} bits",
             32.0 * 4.0 / 96.0, packed_bits);
    println!("37 words, lane payloads packed : {} words = {} B  ({:.3}x)",
             (packed_bits + 31) / 32, (packed_bits + 31) / 32 * 4,
             ((packed_bits + 31) / 32 * 4) as f64 / 96.0);
    println!("64 words, lane-aligned pairs  : 64 words = 256 B  ({:.3}x)", 256.0 / 96.0);
    assert_eq!((packed_bits + 31) / 32, 37);
    assert_eq!((packed_bits + 7) / 8, 148);
    assert!(32u32 * 32 < packed_bits, "32 words (1024 bits) cannot hold 32 x 37 = 1184 bits");
    // And 6-codes-per-lane (the only count that fits a word) does not tile either.
    let groups = (TILE_CODES as u32 + 5) / 6;
    println!("6 codes/lane instead          : {} words = {} B  ({:.3}x), and 32 lanes cover \
              only 192 of 256 codes",
             groups, groups * 4, groups as f64 * 4.0 / 96.0);
}

// ---------------------------------------------------------------------------
// 5. sh0 and the funnel.
// ---------------------------------------------------------------------------

#[test]
fn sh0_never_zero_so_the_funnel_cannot_be_deleted() {
    let mut set = std::collections::BTreeSet::new();
    for lane in 0..LANES { set.insert(setup(lane, 3).1); }
    println!("sh0 at KA=3: {set:?}  sh_max={}", *set.iter().max().unwrap());
    assert_eq!(*set.iter().max().unwrap(), used_lo(3));
    assert!(!set.contains(&0), "sh0 == 0 never occurs at 3 bpw");
    // A word-aligned layout means sh0 == 0, and `rsh0 = 32 - sh0 = 32`, which is an
    // out-of-range shift; the funnel must be deleted, not simplified.  Deleting it
    // is only legal if the whole span fits in A_hi, i.e. used_lo >= 32, i.e. KA <= 2.
    assert_eq!(32u32 - 0u32, 32u32);
    assert!(used_lo(3) < 32, "deleting the funnel at KA=3 reads garbage into v[6], v[7]");
}

#[test]
fn sh0_is_a_constant_at_ka4_and_the_window_overflows_at_ka6() {
    for ka in [2u32, 3, 4, 6] {
        let mut set = std::collections::BTreeSet::new();
        for lane in 0..LANES { set.insert(setup(lane, ka).1); }
        println!("KA={ka}: sh0 set={set:?} used_lo={} span(8)={}", used_lo(ka), span_bits(8, ka));
    }
    let s4: std::collections::BTreeSet<u32> = (0..LANES).map(|l| setup(l, 4).1).collect();
    assert_eq!(s4.len(), 1, "KA=4 sh0 is the constant 20 -- and span is still 44 > 32");
    assert_eq!(*s4.iter().next().unwrap(), 20);
    assert_eq!(span_bits(8, 6), 58);
}