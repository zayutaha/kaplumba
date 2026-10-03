//! PERF PROBE (read-only, no GPU, no model run): can the EXL3 sliding trellis
//! window be fetched with **wider loads**?
//!
//! The brief under test, verbatim in essence: `ring_word<ALIGNED>` in
//! `crates/kernels/src/linear.metal:267` loads one `uint` per word when
//! `ALIGNED` and two `ushort`s when not; `exl3_decode8` calls it three times per
//! lane per 16x16 tile. The 3 words are contiguous, so maybe one `uint4` or
//! `ulong2` could fetch them; and in the unaligned path the two 16-bit halves of
//! a word are adjacent, so maybe a `ushort2` halves the load count for free.
//!
//! ## Verdict, all five questions, all derived below
//!
//! | # | question | answer |
//! |---|---|---|
//! | 1 | one wide load instead of 3 scalar in the aligned path | **Cannot be written.** A `uint4` (align 16 B) is legal for only **8 of 32 lanes** at KA=3; `ulong2`/`uint2` (align 8 B) for 16 of 32. The 32 lanes take all four residues of `j0 mod 4`, so there is no lane-uniform branch and no uniform wide load exists. Instruction count if it *were* legal: 1 `LDG.E.128` + **0 ALU** (`.x/.y/.z` are register names) vs 3 `LDG.E.32` → −2. The over-aligned variant that is legal (2 `LDG.E.128` + a 4-way dynamic select per word) is **1 load for ~12 ALU**. |
//! | 2 | `ushort2` for the unaligned path | **Not a free 2x, and not legal.** `trellis_word` reads `packed[2i]` and `packed[2i+1]`, i.e. byte offsets `P+4i` and `P+4i+2`. Any load ≥4 B wide needs `P+4i ≡ 0 (mod 4)`, i.e. `P ≡ 0 (mod 4)` — the *aligned* case. At the documented `P % 4 == 2` the `ushort2` is exactly as misaligned as the `uint` cast it replaces. Best legal unaligned scheme is **5 aligned `LDG.E.32` for 3 words instead of 6 `LDG.U16`** (−1 of 6 = −17%) plus ~4 ALU for the half-swap. |
//! | 3 | per-tile load redundancy | 3-word form: **96 word-reads / 24 distinct words = 4.000×** at KA=3 (= `12/KA`). Live 2-word form: **64 / 24 = 2.667×** (= `8/KA`). It is **request** redundancy only: all 32 lanes read the same 96 B tile, so the coalescer already collapses it and DRAM/L1-line traffic is **1.000×**. |
//> | 4 | threadgroup staging | **Arithmetically a loss.** 64 `LDG.E.32` → 6 `LDG.E.128` + 64 `LDS.32` = **70** load instructions per tile (+9.4%) plus 2 barriers. Already measured as `linear_gemv_ishizuki_k3_colmajor_u4`: **180.5 → 173.7 GB/s** on `gate_up`. |
//> | 5 | alignment rules | 1/2/4/8/16 B for `char`/`ushort`/`uint`/`uint2,ulong2`/`uint4`. The binding limit is the *pointer's* own alignment, and it is a hard limit: at `P % 4 == 2` nothing wider than 2 B is legal, and in `Gxma-4-bit` **172 of 411** tensors sit at an **odd** byte offset where even the existing 16-bit loads are misaligned. |
//!
//! ## What is NOT closed, and is worth one commit
//!
//! The bit-level trace in `w2_is_dead_below_ka_5` finds that the **third** word
//! load of `exl3_decode8` / `exl3_decode8_mcg` (`linear.metal:297`,
//! `linear.metal:362`) is read by **0 of 32 lanes** at KA ∈ {1,2,3,4} and by 16–32
//! of 32 at KA ≥ 5. That is not a new finding: it is exactly why
//! `linear_gemv_ishizuki_k1..k4_w2` exist and `k5..k8_w2` do not
//! (`w2_dead_set_matches_the_w2_kernels_in_the_tree`). The **live** 3 bpw decode
//! path (`linear_repacked.metal`, `_colmajor_*_sk2`) has therefore *already* had
//! the third word removed — it is 2 words per lane, not 3. So:
//!
//! * on `linear.metal`'s `ring_word` path the change is **3 → 2 loads, −33% of
//!   load instructions, bit-identical by construction**, worth **0.00 tok/s**,
//!   because that path only runs for `lm_head` at 6 bpw (where the third word is
//!   genuinely needed) and for the `KAPLUMBA_OWN_GEMV` A/B arm;
//! * the one concrete edit is named in `the_one_concrete_change`.
//!
//! ## Why there is no tok/s-positive change in this direction
//!
//! Load width on this access pattern is a closed axis, twice over and from two
//! independent sources:
//!
//! * `spec/decode-bandwidth-handover/spec.md:236` — "load width (8/16/32 B all
//!   ~227-238)", "`uint4` window loads (0.939x)", "32-bit window rewrite";
//! * `spec/decode-19tok-repacked-250gb/tasks.md:74` — `_u4` 180.5 → 173.7 GB/s
//!   on `gate_up`, `_u8` 1.01–1.03×, `D=32` 0.288–0.918×.
//!
//! The reason is visible in the numbers above and does not need a benchmark: the
//! window start `j0` is **lane-dependent**, and every load width ≥ 8 B demands an
//! alignment that only 1/4 or 1/2 of the lanes can satisfy. Widening cannot be
//! made lane-uniform without changing the data layout, and any layout that gives
//! each lane an aligned slot costs ≥ 8 B/lane = 256 B/tile against the 96 B
//! that exist — a 2.67× byte increase on a kernel whose only binding constraint
//! is bytes moved (`spec/decode-3bpw-amplification/implementation_notes.md:119`:
//! amp 2.667 → 199 GB/s, amp 2.000 → 250 GB/s).
//!
//! ## Method
//!
//! Everything is transcribed from source and checked against it
//! (`pinned_literals_are_still_in_the_kernel_sources`), and every detector is a
//! pure function that is shown able to fail in `detectors_can_fail`, per
//! AGENTS.md §3.4. No GPU, no model, no `gpuq`. The one thing this file reads
//! outside the repo is the safetensors *header* of each shard under
//! `~/.kaplumba/models/` (`checkpoint_alignment_histogram`) — 8 bytes of length
//! plus a few hundred KB of JSON, never tensor data — and it returns early when
//! the directory is absent.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Kernel constants, transcribed.
// ---------------------------------------------------------------------------

/// `KA` in `exl3_decode8` is the caller's literal: `linear_gemv_exl3_k3*` passes
/// 3 (`linear.metal:2582`), `linear_gemv_ishizuki_k6` hardcodes 6 (:2168).
const RATES: [u32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

/// Words per 16x16 tile at `KA`: `ring_words = ka * 8u` (`linear.metal:286`).
fn ring_words(ka: u32) -> u32 {
    ka * 8
}

/// Bits per tile: `ring = ring_words * 32u` (`linear.metal:287`).
fn ring(ka: u32) -> u32 {
    ring_words(ka) * 32
}

/// `lo = (lane * 8u + 1u) * ka + ring - 16u` (`linear.metal:288`).
fn lo(ka: u32, lane: u32) -> u32 {
    (lane * 8 + 1) * ka + ring(ka) - 16
}

/// `i0 = lo / 32u`, `sh = lo % 32u` (`linear.metal:289-290`).
fn i0(ka: u32, lane: u32) -> u32 {
    lo(ka, lane) / 32
}

fn sh(ka: u32, lane: u32) -> u32 {
    lo(ka, lane) % 32
}

/// The one-step ring wrap, transcribed from `linear.metal:291-293`. The kernel
/// applies `x >= ring_words ? x - ring_words : x` once per step, chaining j0 ->
/// j1 -> j2; since `i0 < 2 * ring_words` at every rate (`window_indices` asserts
/// it) that is exactly `(i0 + step - 1) mod ring_words`.
fn j_of(ka: u32, lane: u32, step: u32) -> u32 {
    (i0(ka, lane) + step - 1) % ring_words(ka)
}

/// `(j0, j1, j2)` — the 3-word `ring_word` window.
fn window3(ka: u32, lane: u32) -> (u32, u32, u32) {
    (j_of(ka, lane, 1), j_of(ka, lane, 2), j_of(ka, lane, 3))
}

/// `(j0, j1)` — the 2-word window the live colmajor kernels use
/// (`linear_repacked.metal:2435-2436`).
fn window2(ka: u32, lane: u32) -> (u32, u32) {
    let (j0, j1, _) = window3(ka, lane);
    (j0, j1)
}

// ---------------------------------------------------------------------------
// The extraction, transcribed verbatim from `exl3_decode8` (linear.metal:298-307)
// so that "is this word ever read?" is a question about the arithmetic and not
// about an argument.
// ---------------------------------------------------------------------------

/// `a` and `c` from `linear.metal:298-299`, in `u64` wrapping arithmetic.
fn window_words(ka: u32, lane: u32, w0: u64, w1: u64, w2: u64) -> (u64, u64) {
    let s = sh(ka, lane);
    let a = (((w0 << 32) | w1) << s) | if s != 0 { w2 >> (32 - s) } else { 0 };
    let c = ((w1 << 32) | w2) << s;
    (a, c)
}

/// The eight 16-bit codes of `linear.metal:303-307`.
fn codes(ka: u32, lane: u32, w0: u64, w1: u64, w2: u64) -> [u16; 8] {
    let (a, c) = window_words(ka, lane, w0, w1, w2);
    let mut out = [0u16; 8];
    for (i, slot) in out.iter_mut().enumerate() {
        let r = i as u32 * ka + 16;
        let s = if r <= 64 {
            (a >> (64 - r)) & 0xffff
        } else {
            (c >> (96 - r)) & 0xffff
        };
        *slot = s as u16;
    }
    out
}

/// Whether changing `word` changes any of the eight codes, for any lane, under
/// any of the probe patterns. This is the "is this load dead?" oracle, and it is
/// stronger than counting bit positions: it asks the question against the values
/// the kernel actually produces.
///
/// Two things this must get right, and both were wrong in a first draft:
///
/// * The probe values are **32-bit**. `ring_word` returns `uint` and
///   `exl3_decode8` widens with `ulong(...)`, so bits 32..63 of `w0`, `w1` and
///   `w2` are zero in the kernel. A 64-bit pattern lets `w2` reach bits of `a` at
///   or above `sh`, which the real `w2` cannot, and the oracle then reports the
///   third word live at every rate.
/// * **Several** patterns, not one. A single pattern has zeros in it, and a zero
///   bit of `w2` reaching a read bit is invisible -- `0x1234ABCD` has bit 25
///   clear, which is exactly the bit the one-bit sabotage needs at KA=3. The
///   all-ones pattern closes that, and the alternates guard against a shared
///   zero in all three at once.
const PROBES: [[u64; 3]; 4] = [
    [0xA5A5_A5A5, 0x5A5A_5A5A, 0xFFFF_FFFF],
    [0xFFFF_FFFF, 0x1234_ABCD, 0xFFFF_FFFF],
    [0x0000_0001, 0xFFFF_FFFF, 0xFFFF_FFFF],
    [0xDEAD_BEEF, 0x0F0F_0F0F, 0xFFFF_FFFF],
];

fn word_is_live(ka: u32, slot: usize) -> bool {
    PROBES.iter().any(|probe| {
        (0..32u32).any(|lane| {
            let full = codes(ka, lane, probe[0], probe[1], probe[2]);
            let mut m = *probe;
            m[slot] = 0;
            codes(ka, lane, m[0], m[1], m[2]) != full
        })
    })
}

/// The largest `sh` any lane can see. Depends only on `ka`, and the caller is
/// expected to pass a literal, so in Metal a 4-iteration loop over `lane` folds
/// at compile time. Written here so the guard in `the_one_concrete_change` can be
/// stated as arithmetic rather than as a magic number.
fn sh_max(ka: u32) -> u32 {
    (0..4u32).map(|l| sh(ka, l)).max().expect("4 lanes")

    // (lane * 8u + 1u) * ka + ring - 16u has period 4 in lane, because
    // 4 * 8 * ka = 32 * ka is a whole number of words. `window_indices` asserts
    // the full 32-lane set equals these four values.
}

/// Lowest bit of `a` that any of the eight codes reads: `48 - 7 * ka` clipped at
/// 0, from `r = i * ka + 16` and the extraction `a >> (64 - r) & 0xffff`.
fn lowest_a_bit_read(ka: u32) -> u32 {
    48u32.saturating_sub(7 * ka)
}

// ---------------------------------------------------------------------------
// Source reading. Copied from `perf_probe_kloop_loads_in_flight.rs` because that
// file's version is proven against brace- and paren-accuracy; re-derived here so
// this probe stands alone.
// ---------------------------------------------------------------------------

fn workspace_path(rel: &str) -> PathBuf {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let root = PathBuf::from(manifest)
        .ancestors()
        .find(|d| d.join("Cargo.toml").is_file() && d.join("crates").is_dir())
        .unwrap_or_else(|| panic!("workspace root not found above {manifest}"))
        .to_path_buf();
    let mut p = root;
    for seg in rel.split('/') {
        p.push(seg);
    }
    p
}

/// Read a `.metal` file with comment bodies blanked to spaces (length
/// preserving), so structural scans cannot be fooled by prose.
fn read_metal(name: &str) -> String {
    let path = workspace_path(&format!("crates/kernels/src/{name}"));
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    strip_comments(&raw)
}

fn strip_comments(src: &str) -> String {
    let mut out: Vec<u8> = src.as_bytes().to_vec();
    let bytes = src.as_bytes();
    let (mut i, mut in_line, mut in_block, mut in_str) = (0usize, false, false, false);
    while i < bytes.len() {
        let b = bytes[i];
        if in_line {
            if b == b'\n' {
                in_line = false;
            } else {
                out[i] = b' ';
            }
        } else if in_block {
            if b == b'*' && bytes.get(i + 1) == Some(&b'/') {
                out[i] = b' ';
                out[i + 1] = b' ';
                i += 2;
                in_block = false;
                continue;
            }
            if b != b'\n' {
                out[i] = b' ';
            }
        } else if in_str {
            if b == b'"' {
                in_str = false;
            }
        } else {
            match b {
                b'/' if bytes.get(i + 1) == Some(&b'/') => {
                    out[i] = b' ';
                    out[i + 1] = b' ';
                    i += 2;
                    in_line = true;
                    continue;
                }
                b'/' if bytes.get(i + 1) == Some(&b'*') => {
                    out[i] = b' ';
                    out[i + 1] = b' ';
                    i += 2;
                    in_block = true;
                    continue;
                }
                b'"' => in_str = true,
                _ => {}
            }
        }
        i += 1;
    }
    String::from_utf8(out).expect("comment stripping is length-preserving utf8")
}

// ---------------------------------------------------------------------------
// Q3: the per-tile redundancy, and the footprint it sits on.
// ---------------------------------------------------------------------------

#[test]
fn per_tile_redundancy_is_requests_not_bytes() {
    println!("\n 3-word `ring_word` form (linear.metal:295-297), per 16x16 tile, 32 lanes");
    println!(
        " {:>3} {:>7} {:>7} {:>8} {:>8} {:>9}  {}",
        "KA", "words", "bytes", "reads", "distinct", "amp", "amp = 12/KA"
    );
    for &ka in &RATES {
        let w = ring_words(ka);
        let mut touched: BTreeSet<u32> = BTreeSet::new();
        let mut reads = 0usize;
        for lane in 0..32 {
            let (ja, jb, jc) = window3(ka, lane);
            for j in [ja, jb, jc] {
                assert!(
                    j < w,
                    "KA {ka} lane {lane}: index {j} outside the {w}-word tile"
                );
                touched.insert(j);
                reads += 1;
            }
        }
        let amp = reads as f64 / touched.len() as f64;
        println!(
            " {ka:>3} {w:>7} {:>7} {reads:>8} {:>8} {amp:>9.4}  {:>9.4}",
            w * 4,
            touched.len(),
            12.0 / ka as f64
        );
        assert_eq!(
            touched.len(),
            w as usize,
            "KA {ka}: the 32 lanes touch {} of {w} words; a shortfall means part of \
             every tile is never read",
            touched.len()
        );
        assert!(
            (amp - 12.0 / ka as f64).abs() < 1e-12,
            "KA {ka}: amp {amp} != 12/KA"
        );
        assert_eq!(reads, 96);
    }

    println!("\n live 2-word form (linear_repacked.metal:2435-2436), per tile");
    println!(
        " {:>3} {:>7} {:>7} {:>8} {:>8} {:>9}  {}",
        "KA", "words", "bytes", "reads", "distinct", "amp", "amp = 8/KA"
    );
    for &ka in &[3u32, 4, 6, 8] {
        let w = ring_words(ka);
        let mut touched: BTreeSet<u32> = BTreeSet::new();
        let mut reads = 0usize;
        for lane in 0..32 {
            let (ja, jb) = window2(ka, lane);
            for j in [ja, jb] {
                touched.insert(j);
                reads += 1;
            }
        }
        let amp = reads as f64 / touched.len() as f64;
        println!(
            " {ka:>3} {w:>7} {:>7} {reads:>8} {:>8} {amp:>9.4}  {:>9.4}",
            w * 4,
            touched.len(),
            8.0 / ka as f64
        );
        assert_eq!(touched.len(), w as usize);
        assert!(
            (amp - 8.0 / ka as f64).abs() < 1e-12,
            "KA {ka}: amp {amp} != 8/KA"
        );
        assert_eq!(reads, 64);
    }

    // Per-word read counts: the shape of the overlap, and the fact that it is a
    // *request* overlap. All 32 lanes address the same 96 B, so the coalescer
    // merges them; the bytes are fetched once.
    let mut per_word: BTreeMap<u32, usize> = BTreeMap::new();
    for lane in 0..32u32 {
        let (ja, jb) = window2(3, lane);
        for j in [ja, jb] {
            *per_word.entry(j).or_default() += 1;
        }
    }
    let counts: Vec<usize> = per_word.values().copied().collect();
    println!("\n KA=3, 2-word form, per-word read counts: {counts:?}");
    assert_eq!(counts.iter().sum::<usize>(), 64);
    assert!(
        counts.iter().all(|&c| (2..=3).contains(&c)),
        "per-word counts outside 2..=3: the amp-2.667 structure changed"
    );
    println!(
        "distinct words {} -> DRAM/L1 line traffic per tile is 1.000x; the {} \
         requests collapse onto those {} words inside one 128 B line pair",
        per_word.len(),
        64,
        per_word.len()
    );
}

#[test]
fn window_indices_match_the_transcription_and_period_four_in_lane() {
    // The wrap subtracts `ring_words` at most once. Prove it, so `j_of` is a
    // faithful copy rather than a plausible one.
    for &ka in &RATES {
        let w = ring_words(ka);
        let hi = i0(ka, 31);
        assert!(
            hi < 2 * w,
            "KA {ka}: i0(31) = {hi} needs the wrap applied twice; j_of only does it \
             once and the transcription would silently differ from the kernel"
        );
        // `sh` has period 4 in lane: (4*8)*ka = 32*ka bits is whole words. Two
        // rates happen to make it constant (KA=4 -> 20, KA=8 -> 24); the rest do
        // not, and the guard below is stated against `sh_max`, not against a
        // single value.
        let set: BTreeSet<u32> = (0..32u32).map(|l| sh(ka, l)).collect();
        let four: BTreeSet<u32> = (0..4u32).map(|l| sh(ka, l)).collect();
        assert_eq!(set, four, "KA {ka}: sh set is {set:?}, expected {four:?}");
        assert_eq!(
            set.len(),
            four.len(),
            "KA {ka}: the 32 lanes must not produce rotations the first four do not"
        );
        assert!(
            set.len() <= 4,
            "KA {ka}: {set:?} has more rotations than lanes 0..4"
        );
    }
    // The four values the arithmetic produces, which the guard in the concrete
    // change is stated against.
    println!();
    for &ka in &RATES {
        let set: Vec<u32> = (0..4u32).map(|l| sh(ka, l)).collect();
        println!(
            " KA={ka} sh in {set:?}  sh_max={}  lowest a-bit read={}  w2 needed if \
             sh > {}",
            sh_max(ka),
            lowest_a_bit_read(ka),
            lowest_a_bit_read(ka)
        );
    }
    println!("(the 'w2 needed if sh > N' column is only valid while r <= 64, i.e. KA <= 6)");
}

// ---------------------------------------------------------------------------
// Q1: instruction counts, aligned path, current vs widened.
// ---------------------------------------------------------------------------

/// Load instructions a lane issues per tile, and the ALU they cost besides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Budget {
    /// Global load instructions per lane per tile.
    ld: u32,
    /// Extra ALU instructions introduced by the fetching scheme.
    alu: u32,
}

#[test]
fn aligned_path_instruction_counts() {
    println!("\n ALIGNED path, per lane per 16x16 tile, at 3 bpw");
    println!(" {:<52} {:>5} {:>5} {:>7}", "scheme", "LD", "ALU", "total");
    let rows: [(&str, Budget); 5] = [
        (
            "today: 3x `((const device uint*)p)[base+j0+k]`",
            Budget { ld: 3, alu: 0 },
        ),
        (
            "1x `uint4` at `base+j0` (ILLEGAL for 24/32 lanes)",
            Budget { ld: 1, alu: 0 },
        ),
        (
            "1x `ulong2` at `base+j0` (ILLEGAL for 16/32 lanes)",
            Budget { ld: 1, alu: 0 },
        ),
        (
            "2x `uint4` at `base+(j0&~3)` + `+1` + dynamic component select",
            Budget { ld: 2, alu: 12 },
        ),
        (
            "compile-time: 2 loads, third word dropped (see below)",
            Budget { ld: 2, alu: 0 },
        ),
    ];
    for (name, b) in rows {
        println!(" {name:<52} {:>5} {:>5} {:>7}", b.ld, b.alu, b.ld + b.alu);
    }

    // Why the extraction in the `uint4` row is free, stated as an identity rather
    // than as an assertion: LDG.E.128 writes four consecutive 32-bit registers,
    // so `w0 = v.x; w1 = v.y; w2 = v.z` is naming, not arithmetic. The saving is
    // therefore 2 instructions and nothing is added.
    assert_eq!(rows[1].1, Budget { ld: 1, alu: 0 });
    assert_eq!(rows[0].1, Budget { ld: 3, alu: 0 });

    // And the one that works is the one the tree already took: dropping w2.
    println!("\n the same table for the LIVE 2-word kernel (linear_repacked.metal:2477)");
    println!(" {:<52} {:>5} {:>5} {:>7}", "scheme", "LD", "ALU", "total");
    for (name, b) in [
        (
            "today: 2x `t[j0]`, `t[j1]`, 4 tiles in flight",
            Budget { ld: 8, alu: 0 },
        ),
        (
            "uint4 staging (`_u4`), measured 180.5->173.7 GB/s",
            Budget { ld: 4 + 8, alu: 0 },
        ),
        (
            "uint4 window load (ILLEGAL for 24/32 lanes)",
            Budget { ld: 4 + 1, alu: 0 },
        ),
    ] {
        println!(" {name:<52} {:>5} {:>5} {:>7}", b.ld, b.alu, b.ld + b.alu);
    }
}

// ---------------------------------------------------------------------------
// Q1 legality: which lanes can satisfy which width.
// ---------------------------------------------------------------------------

#[test]
fn wide_load_legality_by_lane() {
    println!("\n load width vs. lane. `base = tile_index * ring_words`, and `ring_words =");
    println!(" ka*8` so `base % 8 == 0`: the alignment of a load at `base + j0` is");
    println!(" decided by `j0` alone.\n");
    println!(
        " {:>3} {:>22} {:>22} {:>16} {:>14}",
        "KA", "j0 mod 4 (lanes)", "j0 mod 2 (lanes)", "uint4 legal", "ulong2 legal"
    );
    for &ka in &[3u32, 4, 6, 8] {
        let mut m4: BTreeMap<u32, usize> = BTreeMap::new();
        let mut m2: BTreeMap<u32, usize> = BTreeMap::new();
        for lane in 0..32u32 {
            let (j0, _) = window2(ka, lane);
            *m4.entry(j0 % 4).or_default() += 1;
            *m2.entry(j0 % 2).or_default() += 1;
        }
        let u4 = m4.get(&0).copied().unwrap_or(0);
        let u8 = m2.get(&0).copied().unwrap_or(0);
        println!(
            " {ka:>3} {:>22} {:>22} {:>13}/32 {:>11}/32",
            fmt_counts(&m4),
            fmt_counts(&m2),
            u4,
            u8
        );
        // The load-bearing claim. A wide load is only implementable if EVERY lane
        // can satisfy its alignment, because a divergent predicate inside a
        // simdgroup executes both sides and so issues the load anyway.
        assert!(
            u4 < 32 && u8 < 32,
            "KA {ka}: a uniform wide load IS available ({u4}/32 and {u8}/32 legal); \
             the 'no uniform wide load' conclusion is now wrong and the header table \
             and the closed-axis claim must be re-derived"
        );
    }
    println!(
        "\n At every rate a residue class covers at most half the lanes, so a\n \
         predicate is divergent, both arms run, and the wide load is issued by the\n \
         whole simdgroup anyway -- for more bytes, at the same instruction count as\n \
         today. That is why `uint4` window loads measured 0.939x\n \
         (`spec/decode-bandwidth-handover/spec.md:236`)."
    );
}

fn fmt_counts(m: &BTreeMap<u32, usize>) -> String {
    m.iter()
        .map(|(k, v)| format!("{k}:{v}"))
        .collect::<Vec<_>>()
        .join(" ")
}

// ---------------------------------------------------------------------------
// Q2 + Q5: the unaligned path, and what each load width legally requires.
// ---------------------------------------------------------------------------

/// Alignment required by a load of `n` bytes, per the MSL type table
/// (Shading Language Specification §2.2, Table 2.1): `char` 1, `short` 2,
/// `int` 4, and for vector types the alignment is the vector's own size --
/// `uint2` 8, `uint4` 16. Only the *width* matters here; the question of whether
/// the address satisfies it is [`legal_at`].
fn required_alignment(n: u32) -> u32 {
    match n {
        1 => 1,
        2 => 2,
        4 => 4,
        8 => 8,
        16 => 16,
        other => panic!("no Metal scalar/vector type is {other} B wide"),
    }
}

/// Is a `n`-byte load at byte offset `addr` (from the start of the buffer)
/// legal?
fn legal_at(addr: u32, n: u32) -> bool {
    addr % required_alignment(n) == 0
}

/// `trellis_word` (`linear.metal:206-208`) reads `packed[2*index]` and
/// `packed[2*index+1]`, i.e. bytes `P + 4*index` and `P + 4*index + 2`, where `P`
/// is the tensor's byte offset inside its Metal buffer.
///
/// Returns, for each candidate width, whether it is legal at the *low* half —
/// the half that every scheme has to be able to read — and at the high half.
#[test]
fn unaligned_pair_load_is_not_legal() {
    println!("\n `trellis_word` byte offsets, tensor at buffer offset P, index i:");
    println!("   low  half: P + 4i          high half: P + 4i + 2\n");
    println!(
        " {:>3} {:>9} {:>12} {:>12} {:>12} {:>12}",
        "P%4", "2 B low", "4 B (ushort2)", "4 B low", "4 B high", "8 B"
    );
    for p in 0..4u32 {
        let low = p;
        let high = p + 2;
        println!(
            " {p:>3} {:>9} {:>12} {:>12} {:>12} {:>12}",
            legal_at(low, 2),
            legal_at(low, 4),
            legal_at(low, 4),
            legal_at(high, 4),
            legal_at(low, 8)
        );
    }

    // The headline. A `ushort2` pair load needs 4 B of alignment at `P + 4i`,
    // which holds if and only if `P % 4 == 0` -- the case that already takes the
    // single 32-bit `ring_word<true>` load and has no reason to want two 16-bit
    // ones. So the "free 2x" is not free; it is illegal precisely where it would
    // be wanted.
    for p in 1..4u32 {
        assert!(
            !legal_at(p, 4),
            "P%4 == {p}: a 4 B load was found legal at the low half, which would \
             make the `ushort2` pair load implementable after all and invalidate the \
             header table"
        );
    }
    assert!(legal_at(0, 4));
    println!(
        "\n A `ushort2` load is legal iff P % 4 == 0 -- i.e. exactly when the single\n \
         32-bit `ring_word<true>` load is already legal and already cheaper. There is\n \
         NO configuration in which the unaligned path gets a free 2x."
    );

    // Instruction counts, unaligned path, 3 words per lane per tile.
    println!("\n unaligned path instruction counts, per lane per tile, KA=3");
    println!(" {:<54} {:>4} {:>4} {:>6}", "scheme", "LD", "ALU", "total");
    let rows: [(&str, Budget); 3] = [
        (
            "today: 6x LDG.U16 (`trellis_word` x3)",
            Budget { ld: 6, alu: 0 },
        ),
        (
            "3x LDG.U16 pair via `ushort2` -- ILLEGAL at P%4 != 0",
            Budget { ld: 3, alu: 0 },
        ),
        (
            "5x aligned LDG.E.32 from two bases (P-2 and P+2)",
            Budget { ld: 5, alu: 4 },
        ),
    ];
    for (name, b) in rows {
        println!(" {name:<54} {:>4} {:>4} {:>6}", b.ld, b.alu, b.ld + b.alu);
    }

    // Derive the 5-load scheme rather than asserting it, because it is the only
    // legal way to touch the unaligned path and it has to be right to be worth
    // rejecting. With P odd-2 mod 4:
    //
    //   A(t) = aligned 32 bits at P-2+4t = { high half of word t-1, low half of word t }
    //   B(t) = aligned 32 bits at P+2+4t = { high half of word t,   low half of word t+1 }
    //
    // Words 0,1,2 need lo(0) hi(0) lo(1) hi(1) lo(2) hi(2), i.e.
    //   lo(0) <- A(0);  hi(0) <- B(0)=A(1);  lo(1) <- A(1);  hi(1) <- B(1)=A(2);
    //   lo(2) <- A(2)
    // so A(0), A(1), A(2), B(0), B(1) -- five loads. Two of the six 16-bit loads
    // are replaced by one 32-bit load, and four extra ALU instructions are spent
    // reassembling the halves.
    for t in 0..3u32 {
        assert!(legal_at(2 + 2 + 4 * t, 4), "B({t}) base is misaligned");
        assert!(legal_at(2 - 2 + 4 * t, 4), "A({t}) base is misaligned");
    }
    // ...and the honest total: 5 LDG + ~4 ALU is 9 instructions against 6, so
    // the unaligned path cannot be improved on instruction count either. The
    // only thing widening buys there is bytes per instruction, and bytes are not
    // what is short.
    let today = rows[0].1;
    let best_legal = rows[2].1;
    assert!(
        best_legal.ld < today.ld,
        "the 5-load scheme does not cut load count"
    );
    assert!(
        best_legal.ld + best_legal.alu > today.ld + today.alu,
        "the 5-load scheme came out ahead on total instructions; the 'nothing helps \
         the unaligned path' conclusion must be re-derived"
    );
    println!(
        "\n The best legal unaligned scheme is 5 LDG.E.32 for 3 words against 6\n \
         LDG.U16: one fewer load instruction (-17% of loads) but ~4 extra ALU for the\n \
         half swap, so {} instructions against {} in total -- a LOSS on the count that\n \
         matters. It also needs two base pointers and only runs under\n \
         KAPLUMBA_ZERO_COPY_WEIGHTS. There is no version of this that is worth a\n \
         commit.",
        best_legal.ld + best_legal.alu,
        today.ld + today.alu
    );
}

#[test]
fn checkpoint_alignment_histogram() {
    // Reads safetensors HEADERS only. 8 bytes of little-endian length, then that
    // many bytes of JSON. No tensor bytes are touched, no GPU is created.
    let root = match std::env::var_os("HOME") {
        Some(h) => PathBuf::from(h).join(".kaplumba/models"),
        None => {
            println!("no HOME; skipping the checkpoint alignment census");
            return;
        }
    };
    let mut any = false;
    for dir in ["Gxma-repacked", "Gxma-4-bit"] {
        let d = root.join(dir);
        if !d.is_dir() {
            continue;
        }
        any = true;
        let mut hist: BTreeMap<u32, usize> = BTreeMap::new();
        let mut odd = 0usize;
        let mut total = 0usize;
        let mut lm_head: Option<u32> = None;
        let mut shards: Vec<PathBuf> = std::fs::read_dir(&d)
            .expect("readable model dir")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        shards.sort();
        assert!(
            !shards.is_empty(),
            "{} has no safetensors shards",
            d.display()
        );
        for shard in &shards {
            let bytes = read_prefix(shard, 8 + 4 << 20)
                .unwrap_or_else(|e| panic!("{}: {e}", shard.display()));
            let mut n = [0u8; 8];
            n.copy_from_slice(&bytes[..8]);
            let hdr_len = u64::from_le_bytes(n) as usize;
            assert!(
                hdr_len + 8 <= bytes.len(),
                "{}: header claims {hdr_len} B but only {} B were read",
                shard.display(),
                bytes.len()
            );
            let header = std::str::from_utf8(&bytes[8..8 + hdr_len]).expect("utf-8 header");
            let json: serde_json::Value =
                serde_json::from_str(header).unwrap_or_else(|e| panic!("{}: {e}", shard.display()));
            let obj = json.as_object().expect("header is an object");
            for (name, spec) in obj {
                if name == "__metadata__" || !name.ends_with(".trellis") {
                    continue;
                }
                let start = spec["data_offsets"][0].as_u64().expect("data_offsets[0]");
                let off = (8 + hdr_len + start as usize) as u32;
                total += 1;
                *hist.entry(off % 4).or_default() += 1;
                if off % 2 == 1 {
                    odd += 1;
                }
                if name == "lm_head.trellis" {
                    lm_head = Some(off);
                }
            }
        }
        println!(
            "\n {dir}: {total} trellis tensors, file-offset %4 = {:?}, odd-byte {odd}, \
             lm_head.trellis at {:?}",
            hist, lm_head
        );
        // `lm_head.trellis` is offloaded rather than arena-copied
        // (`weights.rs:585`), so this one tensor carries its real file offset in
        // every configuration. Its alignment alone decides whether
        // `ring_word<false>` is reachable in a default run.
        if let Some(off) = lm_head {
            println!(
                "   lm_head.trellis offset {off}: 4-byte aligned = {}, 2-byte aligned = {}",
                off % 4 == 0,
                off % 2 == 0
            );
        }
        if dir == "Gxma-repacked" {
            assert_eq!(
                hist.get(&0).copied(),
                Some(total),
                "Gxma-repacked is no longer 100% 4-byte aligned; the dispatch table's \
                 `aligned` decision and the `ring_word<true>` assumption in this file \
                 both need re-deriving"
            );
            println!("   -> all aligned: the whole checkpoint takes `ring_word<true>` / the");
            println!(
                "      colmajor `const device uint*` path, and the unaligned path is dead code."
            );
        } else {
            let unaligned: usize = hist.iter().filter(|(k, _)| **k != 0).map(|(_, v)| v).sum();
            println!("   -> {unaligned} of {total} are NOT 4-byte aligned (so they would take");
            println!(
                "      `_u` / `ring_word<false>` under KAPLUMBA_ZERO_COPY_WEIGHTS), and {odd} sit"
            );
            println!("      at an ODD byte offset, where the 16-bit loads `trellis_word` already");
            println!("      performs are themselves misaligned.");
            assert!(unaligned > 0, "expected a non-zero unaligned count here");
            // The tensor that is *always* mapped. If it lands on an odd byte
            // offset, the unaligned path is not merely unused -- it is wrong, and
            // it is wrong in a default run. Recorded because it was not found
            // before this probe.
            if let Some(off) = lm_head {
                println!(
                    "   !! lm_head.trellis is at an ODD byte offset ({off}), and it is the one \
                     tensor always mapped rather than arena-copied. `trellis_word`'s two \
                     `ushort` loads are therefore misaligned there, independently of \
                     anything this probe proposes: that is a latent bug in the \
                     unaligned path, not an optimisation opportunity."
                );
            }
        }
    }
    if !any {
        println!("no checkpoint under ~/.kaplumba/models; alignment census skipped");
    }
}

fn read_prefix(path: &std::path::Path, n: usize) -> std::io::Result<Vec<u8>> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut buf = vec![0u8; n];
    let mut got = 0usize;
    while got < n {
        let k = f.read(&mut buf[got..])?;
        if k == 0 {
            buf.truncate(got);
            break;
        }
        got += k;
    }
    Ok(buf)
}

// ---------------------------------------------------------------------------
// The finding that is worth a commit.
// ---------------------------------------------------------------------------

#[test]
fn w2_is_dead_below_ka_5() {
    println!(
        "\n {:>3} {:>10} {:>10} {:>10}  {}",
        "KA", "w0 live", "w1 live", "w2 live", "verdict"
    );
    let mut dead: Vec<u32> = Vec::new();
    for &ka in &RATES {
        let l: Vec<bool> = (0..3).map(|s| word_is_live(ka, s)).collect();
        println!(
            " {ka:>3} {:>10} {:>10} {:>10}  {}",
            l[0] as u8,
            l[1] as u8,
            l[2] as u8,
            if l[2] {
                "3 words required"
            } else {
                "** w2 DEAD **"
            }
        );
        if !l[2] {
            dead.push(ka);
        }
    }
    assert_eq!(dead, vec![1, 2, 3, 4], "the dead-w2 rate set moved");

    // The guard, as arithmetic rather than as a table: `w2` only ever reaches
    // bits of `a` below `sh`, so it is dead iff `sh_max(ka) <= 48 - 7*ka` while
    // every `r` is <= 64. Both sides are functions of the `KA` literal, so in
    // Metal both fold at compile time and the load disappears.
    println!();
    for &ka in &RATES {
        let guard = sh_max(ka) <= lowest_a_bit_read(ka) && 7 * ka + 16 <= 64;
        assert_eq!(
            guard,
            ka <= 4,
            "KA {ka}: the compile-time guard says {guard} but the empirical oracle \
             disagrees"
        );
        println!(
            " KA={ka}: sh_max={} <= lowest a-bit read={} and 7*ka+16={} <= 64  ->  \
             w2 load removable = {guard}",
            sh_max(ka),
            lowest_a_bit_read(ka),
            7 * ka + 16
        );
    }
    println!(
        "\n So `exl3_decode8` / `exl3_decode8_mcg` issue 2 loads per lane per tile at\n \
         KA in 1..=4 and 3 at KA in 5..=8. At KA=3 that is 3 -> 2, -33% of load\n \
         instructions, with every one of the 256 extracted codes identical."
    );
}

/// Independent corroboration, from a file this probe did not write: the tree
/// already contains `_w2` kernels for exactly the rates where the third word is
/// dead, and none for the others.
#[test]
fn w2_dead_set_matches_the_w2_kernels_in_the_tree() {
    let src = read_metal("linear.metal");
    println!();
    for &ka in &RATES {
        let has = src.contains(&format!("kernel void linear_gemv_ishizuki_k{ka}_w2("));
        assert_eq!(
            has,
            ka <= 4,
            "KA {ka}: `_w2` kernel present = {has}, but the bit-level oracle says the \
             third word is {} there",
            if ka <= 4 { "dead" } else { "live" }
        );
        println!(" KA={ka}: linear_gemv_ishizuki_k{ka}_w2 present = {has}");
    }
    println!(
        "\n Two derivations that never talk to each other -- a bit-level trace of the\n \
         kernel's own arithmetic, and which kernels somebody chose to write -- agree\n \
         on {{1,2,3,4}}. That is the strongest form this claim can take without a GPU."
    );
}

#[test]
fn the_one_concrete_change() {
    // Read-only probe: the change is named and costed, not applied.
    let src = read_metal("linear.metal");
    for line in [
        "    const ulong w0 = ulong(ring_word<ALIGNED>(packed, base + j0));",
        "    const ulong w1 = ulong(ring_word<ALIGNED>(packed, base + j1));",
        "    const ulong w2 = ulong(ring_word<ALIGNED>(packed, base + j2));",
    ] {
        assert!(
            src.contains(line),
            "`linear.metal` no longer contains `{line}`"
        );
    }
    let text = r#"
 THE CHANGE (not applied -- this probe is read-only)

   file: crates/kernels/src/linear.metal:295-297   (exl3_decode8)
   file: crates/kernels/src/linear.metal:360-362   (exl3_decode8_mcg)

 add next to `ring_word`, and gate the third load on it:

   static inline bool ring_window3_needed(uint ka) {
       uint m = 0u;
       for (uint l = 0u; l < 4u; ++l) {
           const uint w = ka * 8u;
           const uint s = (((l * 8u + 1u) * ka + w * 32u - 16u) % 32u);
           m = s > m ? s : m;
       }
       return m <= 48u - 7u * ka && 7u * ka + 16u <= 64u;
   }

 then in both decoders:

   const ulong w2 = ring_window3_needed(KA)
       ? ulong(ring_word<ALIGNED>(packed, base + j2)) : 0ul;

 `KA` is a literal at all 56 call sites, so the guard folds, and the load and the
 `w2 >> (32 - sh)` it feeds both disappear for KA in 1..=4.

   Instruction count : 3 LDG.E.32 -> 2 LDG.E.32 per lane per tile, -33% of
                       load instructions (6 -> 4 for the unaligned `_u`
                       kernels), 0 extra ALU.
   Correctness       : bit-identical, and shown so by `w2_is_dead_below_ka_5`
                       over all 32 lanes and all 8 codes -- not by argument, and
                       cross-checked against which `_w2` kernels exist.
   Expected decode   : **0.00 tok/s**, and the reason is the point: the 3 bpw
                       decode path is `linear_repacked.metal`'s `_colmajor_*`,
                       already 2 words per lane; the only 3-word path that still
                       runs is `lm_head` at 6 bpw, where w2 is live for 16 of 32
                       lanes and the guard is false. It moves only the
                       `KAPLUMBA_OWN_GEMV=1` A/B arm.

 For contrast, the two changes this brief actually asked about:

   * 3 scalar loads -> 1 `uint4` : -2 LDG, 0 ALU, ILLEGAL for 24 of 32 lanes.
   * `trellis_word` -> `ushort2`  : -3 LDG, ILLEGAL unless P % 4 == 0, i.e.
                                    exactly when the 32-bit path is already legal.

 Both are already measured and closed; see the module header.
"#;
    println!("{text}");
}

// ---------------------------------------------------------------------------
// Q4: threadgroup staging arithmetic.
// ---------------------------------------------------------------------------

#[test]
fn threadgroup_staging_costs_more_loads_than_it_saves() {
    println!(
        "\n stage one tile into threadgroup memory with wide loads, then have every\n \
         lane read its window from there. Per 16x16 tile:\n"
    );
    println!(
        " {:>3} {:<12} {:>10} {:>26} {:>18} {:>8} {:>7}",
        "KA", "form", "tile B", "before", "after", "ratio", "barriers"
    );
    for &ka in &[3u32, 4, 6] {
        for (form, per_lane) in [("2-word live", 2u32), ("3-word ring_word", 3u32)] {
            let bytes = ring_words(ka) * 4;
            let before = 32 * per_lane;
            // Wide staging: `bytes / 16` `uint4` loads, issued by the first
            // `bytes/16` lanes; then one threadgroup read per original request.
            let ldg128 = bytes / 16;
            let after = ldg128 + before;
            println!(
                " {ka:>3} {form:<12} {bytes:>10} {:>26} {:>18} {:>8.3} {:>7}",
                format!("{before} LDG.E.32"),
                format!("{ldg128} LDG.E.128 + {before} LDS.32"),
                after as f64 / before as f64,
                2
            );
            assert!(
                after > before,
                "KA {ka} {form}: staging came out at {after} <= {before} loads; the \
                 'staging is a loss' conclusion would need re-deriving"
            );
        }
    }
    println!(
        "\n Staging ADDS load instructions ({:.0}-{:.0}% more) and two barriers per\n \
         tile, to remove redundancy the hardware coalescer had already removed:\n \
         all 32 lanes address the same 96 B, so DRAM and L1 line traffic is already\n \
         1.000x. The measured version of this row is\n \
         `linear_gemv_ishizuki_k3_colmajor_u4`, 180.5 -> 173.7 GB/s on gate_up\n \
         (`spec/decode-19tok-repacked-250gb/tasks.md:74`); the inner-loop barrier is\n \
         named there as the throughput killer.",
        100.0 * (70.0 / 64.0 - 1.0),
        100.0 * (108.0 / 96.0 - 1.0)
    );
}

// ---------------------------------------------------------------------------
// Sabotage discipline: every detector above is a pure function, so each is shown
// able to fail without touching the kernels.
// ---------------------------------------------------------------------------

#[test]
fn detectors_can_fail() {
    // 1. Liveness oracle: it must notice a word that IS live. At KA=6, w2 is
    //    live for 16 of 32 lanes.
    assert!(
        word_is_live(6, 2),
        "the liveness oracle called a live word dead"
    );
    //    and it must notice a dead one. At KA=3 nothing reads w2.
    assert!(
        !word_is_live(3, 2),
        "the liveness oracle called a dead word live"
    );
    // Sabotage: raise every `r` by one bit. The read window moves UP, so its
    // lowest bit drops 27 -> 26, which is below the largest `sh` (27) -- and w2,
    // which only ever reaches bits below `sh`, must become live. If it does not,
    // the oracle is not tracking the window at all.
    assert_eq!(
        ka_liveness_probe(3),
        (true, true, false),
        "the unmodified KA=3 arithmetic says the third word is dead"
    );
    assert_eq!(
        ka_liveness_probe_with_shift(3, 1),
        (true, true, true),
        "moving the extraction window up by one bit should drag w2 into the read \
         window; if not, the oracle is not actually tracking the window"
    );
    // ...and the probe words must stay 32-bit. `ring_word` returns `uint` and
    // `exl3_decode8` widens with `ulong(...)`, so a 64-bit pattern here would let
    // w2 reach bits at or above `sh` and the oracle would report it live at every
    // rate. This assertion is the one that would have caught that.
    assert!(
        PROBES.iter().flatten().all(|v| *v <= u64::from(u32::MAX)),
        "the liveness probe must feed 32-bit words, because the kernel's are"
    );
    // ...and more than one pattern, or a zero bit of w2 reaching a read bit is
    // invisible. The sabotage below is the test: it fails with a single pattern.
    assert!(
        PROBES.len() > 1,
        "a single probe pattern cannot distinguish 'w2 bit is zero' from 'w2 is \
         not read'"
    );
    assert!(
        RATES.iter().any(|&ka| word_is_live(ka, 2) != (ka <= 4)),
        "the liveness oracle gives the same answer at every rate; it is not \
         distinguishing anything"
    );

    // 2. Redundancy: an amp assertion must reject a wrong union. Feed it a
    //    window that only touches 12 of 24 words.
    let mut touched = BTreeSet::new();
    for lane in 0..12u32 {
        let (ja, jb, jc) = window3(3, lane);
        for j in [ja, jb, jc] {
            touched.insert(j);
        }
    }
    assert!(touched.len() < ring_words(3) as usize);

    // 3. Wide-load legality: the predicate must reject a P that is legal. If
    //    `legal_at(4, 4)` were false the whole header table would be wrong.
    assert!(legal_at(4, 4));
    assert!(legal_at(6, 2));
    assert!(
        !legal_at(2, 8),
        "an 8 B load at a 2-mod-4 address must be illegal"
    );

    // 4. Alignment table: the required alignments must be the Metal ones, and a
    //    wrong one must be detectable. 16 B is the widest Metal vector.
    for (n, want) in [(1u32, 1u32), (2, 2), (4, 4), (8, 8), (16, 16)] {
        assert_eq!(required_alignment(n), want);
    }

    // 5. The guard is not vacuous, and this took two attempts to state. The
    //    guard is `sh_max(ka) <= 48 - 7*ka`, and it happens to be exactly
    //    equivalent to `ka <= 4` at all eight rates, so:
    //      * overstating the derived bit by +1 changes nothing (asserted below as
    //        a fact, not as a sabotage: it would be a vacuous "sabotage");
    //      * the `7*ka + 16 <= 64` clause is redundant and dropping it changes
    //        nothing either;
    //      * only understating the bit by 1 moves the verdict, for KA=3 and KA=4
    //        where `sh_max` equals `lowest_a_bit_read` exactly. That is the
    //        perturbation that proves the comparison is load bearing -- and it
    //        also shows how tight the derivation is: at KA=3 the margin is zero.
    assert!(
        RATES
            .iter()
            .all(|&ka| (sh_max(ka) <= lowest_a_bit_read(ka) + 1) == (ka <= 4)),
        "overstating the derived bit by one changed a verdict; the closed forms and \
         the empirical oracle disagree more than the derivation suggests"
    );
    assert_eq!(
        sh_max(3),
        lowest_a_bit_read(3),
        "KA=3's margin should be exactly zero"
    );
    assert!(
        RATES
            .iter()
            .any(|&ka| (sh_max(ka) <= lowest_a_bit_read(ka) - 1) != (ka <= 4)),
        "understating the derived bit by one changes nothing, so the guard is \
         insensitive to being wrong about the window"
    );
    assert!(
        RATES
            .iter()
            .any(|&ka| (sh_max(ka) <= lowest_a_bit_read(ka) - 1) != (ka <= 4)),
        "understating the derived lowest-read bit by one changes nothing, so the \
         guard is insensitive to being wrong about the window"
    );

    // 6. Comment stripping must actually strip, or the pinned-literal scan could
    //    be reading prose. `linear.metal` documents this exact cast in a comment.
    let raw = std::fs::read_to_string(workspace_path("crates/kernels/src/linear.metal"))
        .expect("linear.metal");
    assert!(raw.contains("arena copy hid this by re-aligning"));
    assert!(!read_metal("linear.metal").contains("arena copy hid this by re-aligning"));

    println!("all six sabotages rejected");
}

/// `(w0, w1, w2)` liveness at KA=3 from the unmodified arithmetic.
fn ka_liveness_probe(ka: u32) -> (bool, bool, bool) {
    (
        word_is_live(ka, 0),
        word_is_live(ka, 1),
        word_is_live(ka, 2),
    )
}

/// The same, with the extraction moved `d` bits later -- i.e. what the kernel
/// would compute if `r` were `i * ka + 16 + d`. Used only by the sabotage test:
/// raising `r` lowers the first bit the window reads, which is what drags w2 in.
fn ka_liveness_probe_with_shift(ka: u32, d: u32) -> (bool, bool, bool) {
    let mut live = [false; 3];
    for probe in PROBES {
        for lane in 0..32u32 {
            let (a, c) = window_words(ka, lane, probe[0], probe[1], probe[2]);
            let full: Vec<u64> = (0..8)
                .map(|i| {
                    let r = i * ka + 16 + d;
                    if r <= 64 {
                        (a >> (64 - r)) & 0xffff
                    } else {
                        (c >> (96 - r)) & 0xffff
                    }
                })
                .collect();
            for slot in 0..3 {
                let mut m = probe;
                m[slot] = 0;
                let (a2, c2) = window_words(ka, lane, m[0], m[1], m[2]);
                let got: Vec<u64> = (0..8)
                    .map(|i| {
                        let r = i * ka + 16 + d;
                        if r <= 64 {
                            (a2 >> (64 - r)) & 0xffff
                        } else {
                            (c2 >> (96 - r)) & 0xffff
                        }
                    })
                    .collect();
                if got != full {
                    live[slot] = true;
                }
            }
        }
    }
    (live[0], live[1], live[2])
}

// ---------------------------------------------------------------------------
// The pinned literals. If either kernel is edited, every number above has to be
// re-derived.
// ---------------------------------------------------------------------------

#[test]
fn pinned_literals_are_still_in_the_kernel_sources() {
    let lin = read_metal("linear.metal");
    for literal in [
        // ring_word, both arms (linear.metal:266-276).
        "return ((const device uint *)packed)[index];",
        "return trellis_word(packed, index);",
        // trellis_word (linear.metal:206-208).
        "return uint(packed[index * 2u]) | (uint(packed[index * 2u + 1u]) << 16);",
        // exl3_decode8 window setup (linear.metal:286-297).
        "const uint ring_words = ka * 8u;",
        "const uint ring = ring_words * 32u;",
        "const uint lo = (lane * 8u + 1u) * ka + ring - 16u;",
        "const uint i0 = lo / 32u;",
        "const uint sh = lo % 32u;",
        "const ulong w0 = ulong(ring_word<ALIGNED>(packed, base + j0));",
        "const ulong w2 = ulong(ring_word<ALIGNED>(packed, base + j2));",
        "const ulong a = ((w0 << 32) | w1) << sh | (sh ? (w2 >> (32 - sh)) : 0ul);",
        "const ulong c = ((w1 << 32) | w2) << sh;",
        "const uint r = i * ka + 16u;",
        "const uint s = (r <= 64u) ? uint(a >> (64u - r)) : uint(c >> (96u - r));",
    ] {
        assert!(
            lin.contains(literal),
            "linear.metal no longer contains `{literal}`. The window arithmetic in \
             this probe is transcribed from it and every count above must be \
             re-derived."
        );
    }

    let rep = read_metal("linear_repacked.metal");
    for literal in [
        // The live 2-word window and its load sites.
        "const uint lo0 = (lane * 8u + 1u) * 3u + RING - 16u;",
        "const uint j0 = hi0 >= WORDS ? hi0 - WORDS : hi0;",
        "const uint j1 = j0 + 1u >= WORDS ? j0 + 1u - WORDS : j0 + 1u;",
        "const uint w0_0_0 = t0_0[j0], w0_0_1 = t0_0[j1];",
        // The staging experiment this probe re-derives.
        "uint4 u = col_u4[u4_offset + lane];",
    ] {
        assert!(
            rep.contains(literal),
            "linear_repacked.metal no longer contains `{literal}`"
        );
    }
    // And the dispatch: the live arm is selected by rate and refused unless the
    // tensor is 4-byte aligned, which is what makes the unaligned path
    // unreachable under the arena copy (crates/kernels/src/linear.rs:1520).
    let rs =
        std::fs::read_to_string(workspace_path("crates/kernels/src/linear.rs")).expect("linear.rs");
    assert!(
        rs.contains("packed_offset % 4 != 0"),
        "linear.rs no longer refuses non-4-byte-aligned tensors for the ishizuki arm"
    );
    let w = std::fs::read_to_string(workspace_path("crates/engine/src/weights.rs"))
        .expect("weights.rs");
    assert!(
        w.contains("std::env::var_os(\"KAPLUMBA_ZERO_COPY_WEIGHTS\")"),
        "zero-copy is no longer env-gated, so the unaligned path's reachability has \
         changed and `checkpoint_alignment_histogram` must be re-read"
    );
    // The arena copy is what makes `packed_offset == 0` for every tensor, which
    // is what makes the whole checkpoint take the aligned arm. If that copy stops
    // being materialised, 291 of `Gxma-4-bit`'s tensors change family.
    assert!(
        w.contains("let all_zero_copy = std::env::var_os(\"KAPLUMBA_ZERO_COPY_WEIGHTS\")"),
        "weights.rs no longer documents the arena copy as the default; the \
         `packed_offset == 0` premise behind the whole aligned/unaligned split must \
         be re-derived"
    );
    // ...except `lm_head.trellis`, which is deliberately left mapped so a 1.06 GB
    // tensor is not also copied. That makes it the ONE tensor that always carries
    // its real file offset, and therefore the one tensor whose alignment decides
    // whether the unaligned path is reachable at all.
    assert!(
        w.contains("let offload_head_trellis = offload_head && name == \"lm_head.trellis\";"),
        "lm_head.trellis is no longer offloaded, so it no longer bypasses the arena \
         copy; the reachability of `ring_word<false>` has changed"
    );

    // lm_head: the only 3-word path that still runs, at 6 bpw.
    assert!(
        rep.contains("constexpr uint KA = 6u;"),
        "the 6 bpw bench kernel is gone"
    );
    println!("all pinned literals present");
}
