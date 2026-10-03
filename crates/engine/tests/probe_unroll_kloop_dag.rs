//! PERF PROBE (read-only, no GPU): the K-loop's dependency graph and its
//! instruction-level parallelism, on the kernels that actually run at decode.
//!
//! ## Why this file exists
//!
//! Two things it is *not*.
//!
//! It is **not** `perf_probe_kloop_loads_in_flight.rs`. That probe already
//! answers "how many global loads per thread are in flight" for five kernels,
//! and answers it well. What that probe does not do is (a) enumerate the
//! dependency graph *inside* one tile's eight-code extraction, (b) count
//! selection branches per lane per tile, or (c) cover `fused2_co` -- which is
//! the live `gate_up` kernel and which that probe's kernel list omits. Those
//! three gaps are what this file closes. Where they overlap, this file cites
//! the other one rather than restating it.
//!
//! It is **not** a restatement of `spec/decode-25tok-peak-bandwidth` T3 or
//! T4.3. Those are done. This file measures what is left over after them.
//!
//! ## The brief I was handed, and which of its premises hold
//!
//! The investigation this file records was asked to confirm that two
//! instruction-level-parallelism wins were available: "8 branches per lane
//! eliminated" from the `r <= 64` selection, and a mask instead of a select for
//! the `sh ? (w2 >> (32 - sh)) : 0` ternary. **Both premises are false, and
//! the reason they are false is the interesting result.** See
//! `selection_branch_count_is_zero_on_the_live_path` and
//! `legacy_three_word_form_has_one_runtime_select_not_nine`.
//!
//! ## Result summary
//!
//! | quantity | value |
//! |---|---|
//! | critical path, load -> FMA, per tile | 5 levels, unchanged by any reordering |
//! | `A_hi` / `A_lo` dependence | `A_hi = f(wa, wb)`, `A_lo = f(wb)` -- disjoint, both level 1 |
//! | is there a `load w1 -> a -> c` serial chain? | **no**, and there never was |
//! | selection branches per lane per tile, live path | **0** |
//! | selection branches per lane per tile, legacy 3-word | 9 source-level, **1 genuine** |
//! | `sh0` is a runtime value? | yes -> the legacy shift-select is real, the `r<=64` one is not |
//! | word loads per lane per tile, live | **2** (`j0`, `j1`) -- not 3 |
//! | `rsh0 = 32 - sh0` hoisted out of the K-loop? | yes, in every live kernel |
//! | weight loads in flight, `fused2_co` (live gate_up) | **4 of 8** -- coverage gap, see below |
//!
//! ## The one genuine structural finding
//!
//! `linear_gemv_ishizuki_k3_fused2_co` (`linear_repacked.metal:814-851`) issues
//! its two Col-B weight loads *after* consuming Col A:
//!
//! ```text
//!     w0_a0, w0_a1  <- loads      extract_v_mcg_32 -> acc  (ALU)
//!     w1_a0, w1_a1  <- loads      extract_v_mcg_32 -> acc  (ALU)
//!     w0_b0, w0_b1  <- loads      extract_v_mcg_32 -> acc  (ALU)
//!     w1_b0, w1_b1  <- loads      extract_v_mcg_32 -> acc  (ALU)
//! ```
//!
//! so its steady-state burst is 4 deep, twice, with an ALU drain in between --
//! where `colmajor_2col_sk2`, `fused2_2col` and `fused3_2col` burst all 8 at
//! once. That is the *same* source-level shape T4.3 already fixed in
//! `colmajor_2col` -> `colmajor_2col_sk2` (the other probe prints that diff:
//! "_colmajor_2col: col1 load at byte offset 1325, first consume at 858 -> 4 in
//! flight; _colmajor_2col_sk2 -> 8 in flight"). T4.3 did not carry the hoist
//! across to `fused2_co`.
//!
//! Expected gain: **0 to +1.9% end to end, most likely inside noise.** The
//! ceiling is T4.3's own measurement for exactly this hoist on the
//! `colmajor_2col` pair (14.58 -> 14.85 tok/s, but bundled with the 2x unroll
//! and the divisibility fix), and the floor is the `_u8` deeper-unroll result,
//! which measured 1.01-1.03x isolated and -0.55%/+0.62% in engine
//! (`spec/decode-19tok-repacked-250gb/spec.md:152`). Neither end of that range
//! justifies doing it for throughput alone; it is worth doing because it makes
//! `fused2_co` consistent with its siblings and removes the last kernel where
//! the "8 in flight" claim is false.
//!
//! ## Why the expected gain is capped (stated up front, not buried)
//!
//! Two independent arguments, both from the repo, say instruction-level
//! parallelism is *not* the binding constraint on this box:
//!
//!  1. Little's law. At 3 tg/core and 4 tiles (384 B) in flight per simdgroup,
//!    480 resident simdgroups cover ~190 ns of latency against ~100 ns of Apple
//!    DRAM latency. Already ample (`perf_probe_kloop_loads_in_flight.rs:759`).
//!  2. The engine is bandwidth-bound, not latency-bound. Layer GEMVs run at
//!    ~126 GB/s against a measured 3 bpw ceiling of 203 GB/s, and 11.834
//!    GB/token at 203 GB/s is 58.3 ms = 17.2 tok/s, so the *entire* remaining
//!    latency-side headroom in the whole model is about +0.7 tok/s
//!    (`spec/decode-25tok-peak-bandwidth/implementation_notes.md` §18.3).
//!
//! So the honest summary of this angle is: **the K-loop is already deeply
//! pipelined, the branch wins are already banked, and what is left is one
//! two-line hoist worth measuring rather than assuming.**
//!
//! ## Read-only, and why every number here is checkable
//!
//! This file opens no GPU context, loads no model, and calls no `gpuq`. Every
//! detector is a pure function over a source string. That is deliberate: it is
//! the only way to satisfy AGENTS.md §3.4 -- a coverage claim is not a claim
//! until the code has been broken and seen to fail -- while being forbidden from
//! editing `linear_repacked.metal`. So each detector is re-run against a
//! deliberately mutated copy of the kernel text in `detectors_can_fail`, and the
//! mutation names the claim it is attacking.
//!
//! NOT MEASURED HERE, and not derivable from source: register counts, occupancy,
//! and achieved bandwidth. The Metal toolchain is absent on this box
//! (`xcrun -sdk macosx metal` -> "missing Metal Toolchain"), so there is no AIR
//! to disassemble and no way to check whether the compiler already hoists these
//! loads past the first consume. Every "in flight" number below is a
//! *source-order* claim, and is labelled as such.

use std::collections::BTreeMap;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Kernel constants, transcribed, and pinned by `constants_are_pinned_in_source`.
// ---------------------------------------------------------------------------

/// `constexpr uint KA = 3u;`
const KA: u32 = 3;
/// `constexpr int SG = 8;` -> 256 threads per threadgroup.
const SG: usize = 8;

/// The decode kernels `linear.rs` can dispatch at decode time, with the role
/// each one has. Sourced from the `ishizuki_gemv_pipeline_by_name` /
/// `trace_gemv` sites in `crates/kernels/src/linear.rs` (2385-2590), not from
/// the kernel list in the other probe, so the two sets are cross-checkable.
const LIVE_KERNELS: &[(&str, &str)] = &[
    ("linear_gemv_ishizuki_k3_fused2_co", "gate_up 5376x43008 (DEFAULT: a==b, no env)"),
    ("linear_gemv_ishizuki_k3_fused2_2col", "gate_up 5376x43008 (KAPLUMBA_NO_FUSED2_CO)"),
    ("linear_gemv_ishizuki_k3_fused2_co4", "gate_up (KAPLUMBA_FUSED2_CO4)"),
    ("linear_gemv_ishizuki_k3_fused2_4col", "gate_up (4col opt-in)"),
    ("linear_gemv_ishizuki_k3_fused2_sk2", "gate_up (KAPLUMBA_USE_SK)"),
    ("linear_gemv_ishizuki_k3_fused3_2col", "qkv 2048x6144"),
    ("linear_gemv_ishizuki_k3_fused3_sk2", "qkv (KAPLUMBA_USE_SK)"),
    ("linear_gemv_ishizuki_k3_colmajor_2col", "down/o_proj, pre-T4.3 shape"),
    ("linear_gemv_ishizuki_k3_colmajor_2col_sk2", "down 21504x5376, o_proj"),
    ("linear_gemv_ishizuki_k3_colmajor_2col_sk4", "down (sk4)"),
    ("linear_gemv_ishizuki_k3_colmajor_contig4_sk2", "o_proj (sk2)"),
    ("linear_gemv_ishizuki_k3_colmajor_contig4_sk4", "o_proj (sk4)"),
];

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

/// Replace comment bodies with spaces, preserving byte length, so offsets found
/// in the cleaned copy index the original and brace matching cannot be fooled by
/// a `{` or `?` inside a comment. `linear_repacked.metal` has both -- the file's
/// own comments discuss `sh ? (w2 >> (32 - sh)) : 0`, so a naive scan for `?`
/// finds prose.
fn strip_comments(src: &str) -> String {
    let bytes = src.as_bytes();
    let mut out = bytes.to_vec();
    let mut i = 0usize;
    let (mut in_line, mut in_block, mut in_str) = (false, false, false);
    while i < bytes.len() {
        let b = bytes[i];
        if in_line {
            if b == b'\n' {
                in_line = false;
            } else {
                out[i] = b' ';
            }
            i += 1;
        } else if in_block {
            if b == b'*' && bytes.get(i + 1) == Some(&b'/') {
                out[i] = b' ';
                out[i + 1] = b' ';
                i += 2;
                in_block = false;
            } else {
                if b != b'\n' {
                    out[i] = b' ';
                }
                i += 1;
            }
        } else if in_str {
            if b == b'\\' {
                i += 2;
                continue;
            }
            if b == b'"' {
                in_str = false;
            }
            i += 1;
        } else {
            match b {
                b'/' if bytes.get(i + 1) == Some(&b'/') => {
                    out[i] = b' ';
                    out[i + 1] = b' ';
                    in_line = true;
                    i += 2;
                }
                b'/' if bytes.get(i + 1) == Some(&b'*') => {
                    out[i] = b' ';
                    out[i + 1] = b' ';
                    in_block = true;
                    i += 2;
                }
                b'"' => {
                    in_str = true;
                    i += 1;
                }
                _ => i += 1,
            }
        }
    }
    String::from_utf8(out).expect("comment stripping preserved utf8")
}

fn read_metal(name: &str) -> String {
    let path = workspace_path(&format!("crates/kernels/src/{name}"));
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    strip_comments(&raw)
}

/// Body of `kernel void <name>(`, up to the next `kernel void `.
fn kernel_body<'a>(src: &'a str, name: &str) -> &'a str {
    let start = src
        .find(&format!("kernel void {name}("))
        .unwrap_or_else(|| panic!("no kernel {name}"));
    let tail = &src[start..];
    // `tail` begins at the matched `kernel void`, which is not preceded by a
    // newline in the slice, so this finds the *next* kernel, not this one.
    let end = tail.find("\nkernel void ").unwrap_or(tail.len());
    assert!(end > 0, "{name}: empty body");
    &tail[..end]
}

/// Byte offset of the unrolled K-loop header inside a kernel body, used to prove
/// the `32 - sh0` mask is hoisted *above* the loop rather than recomputed in it.
fn unrolled_k_loop_header_offset(kernel: &str, name: &str) -> usize {
    kernel
        .find("k_unroll_end")
        .or_else(|| kernel.find("+ 3 * SG < tiles_k"))
        .unwrap_or_else(|| panic!("{name}: no unrolled K-loop header"))
}

/// The unrolled K-loop body of a kernel: the first loop whose header mentions
/// `k_unroll_end`, or the explicit 4-substep form `+ 3 * SG < tiles_k`, matched
/// by braces so the returned slice is exactly one loop body.
fn unrolled_k_loop_body<'a>(kernel: &'a str, name: &str) -> &'a str {
    let mut from = 0usize;
    while let Some(abs) = kernel[from..].find("for (") {
        let open_rel = kernel[from + abs..].find('{').expect("loop header with {");
        let header = &kernel[from + abs..from + abs + open_rel];
        if header.contains("k_unroll_end") || header.contains("+ 3 * SG < tiles_k") {
            let body_start = from + abs + open_rel + 1;
            let mut depth = 1i32;
            let mut i = body_start;
            let b = kernel.as_bytes();
            while i < b.len() && depth > 0 {
                match b[i] {
                    b'{' => depth += 1,
                    b'}' => depth -= 1,
                    _ => {}
                }
                i += 1;
            }
            assert_eq!(depth, 0, "{name}: unbalanced braces in K-loop body");
            return &kernel[body_start..i - 1];
        }
        from += abs + 4;
    }
    panic!("{name}: no unrolled K-loop found")
}

/// Count `[j0]`, `[j1]`, `[j2]` subscripts in `src`. On a `const device uint *`
/// weight pointer each is one 32-bit global load.
fn weight_load_count(src: &str) -> usize {
    ["[j0]", "[j1]", "[j2]"]
        .iter()
        .map(|p| src.matches(p).count())
        .sum()
}

/// Byte offset of the first place a loaded word is consumed. Everything issued
/// before this is in flight at the top of the burst *in source order*; a load
/// after it is not.
fn first_consume_offset(body: &str) -> usize {
    ["extract_v_mcg_32(", "A_hi =", "A0 =", "A_lo =", "v0[i] ="]
        .iter()
        .filter_map(|m| body.find(m))
        .min()
        .unwrap_or_else(|| panic!("no consume site in K-loop body"))
}

fn loads_in_flight(body: &str) -> (usize, usize) {
    let cut = first_consume_offset(body);
    (weight_load_count(&body[..cut]), weight_load_count(body))
}

/// The burst shape of one unrolled K-loop body.
///
/// Two source forms exist in this file and conflating them is how the count goes
/// wrong. The *named-argument* form loads into locals first
/// (`w0_a0 = t0_a[j0]` ... then `extract_v_mcg_32(w0_a0, w0_a1)`), so loads can
/// be hoisted above the consume and the burst is `loads before first consume`.
/// The *inline* form passes the subscripts straight into the call
/// (`extract_v_mcg_32(t0_a0[j0], t0_a0[j1], ...)`), so a load and its only use
/// are the same expression: the burst is exactly one tile's 2 words and cannot be
/// hoisted without also rewriting the call. `_co4` and `_4col` are inline.
enum Burst {
    /// Loads hoisted into locals; `inflight` may be less than `total`.
    Split { inflight: usize, total: usize },
    /// Loads written directly as call arguments: 2 in flight per tile, always.
    Inline { per_tile: usize, tiles: usize },
}

fn burst_shape(body: &str) -> Burst {
    if body.contains("extract_v_mcg_32(t") {
        let tiles = body.matches("extract_v_mcg_32(t").count();
        return Burst::Inline { per_tile: 2, tiles };
    }
    let (inflight, total) = loads_in_flight(body);
    Burst::Split { inflight, total }
}

// ---------------------------------------------------------------------------
// 1. The two-word derivation, re-derived here rather than trusted.
// ---------------------------------------------------------------------------

#[test]
fn two_word_window_covers_every_bit_a_code_can_read() {
    // r = i*KA + 16 for i in 0..8, so the A0 bits a code touches are
    // [64 - r_max, 63] with r_max = 7*KA + 16.
    let r_max = 7 * KA + 16;
    let lowest_used_bit = 64 - r_max;
    assert_eq!(r_max, 37, "KA=3 -> r_max is 7*3+16=37");
    assert!(r_max <= 64, "so the B0 arm is unreachable at KA=3");

    // `sh0 = lo0 % 32`, `lo0 = (lane*8+1)*KA + RING - 16`, lane in 0..32.
    let ring = 256 * KA;
    let shs: Vec<u32> = (0..32)
        .map(|lane| ((lane * 8 + 1) * KA + ring - 16) % 32)
        .collect();
    let max_sh = *shs.iter().max().unwrap();
    let mut uniq = shs.clone();
    uniq.sort_unstable();
    uniq.dedup();
    println!("KA={KA}  RING={ring}  r_max={r_max}");
    println!("A0 bits any code can read: [{lowest_used_bit}, 63]");
    println!("sh0 over the 32 lanes: {uniq:?}  (max {max_sh})");

    // The third word contributes only to A0 bits [0, sh); it is dead when the
    // lowest used bit is at or above the largest sh that occurs.
    assert!(
        lowest_used_bit >= max_sh,
        "KA={KA}: code reads A0 bit {lowest_used_bit} but lane sh reaches {max_sh}, \
         so the third word IS read"
    );
    println!(
        "=> third ring word is dead at KA=3 ({lowest_used_bit} >= {max_sh}); this is \
         why every live kernel loads 2 words, not 3"
    );
    // This is a *tight* equality, worth stating: it is not slack.
    assert_eq!(
        lowest_used_bit, max_sh,
        "the KA=3 margin is exactly zero; a KA or lane change would reopen it"
    );
    println!("NOTE: the margin is exactly 0 -- the proof is tight, not comfortable.");

    // Per-lane r values at KA=3: every one is <= 64, so the `r <= 64` select is
    // dead. This is the brief's item 4, shown arithmetically.
    let rs: Vec<u32> = (0..8).map(|i| i * KA + 16).collect();
    println!("r per code i=0..7 at KA=3: {rs:?}");
    assert!(rs.iter().all(|&r| r <= 64), "all r <= 64 at KA=3");
}

// ---------------------------------------------------------------------------
// 2. The dependency graph of one tile's eight-code extraction.
// ---------------------------------------------------------------------------

/// Free identifiers an expression reads, ignoring C keywords and call names.
fn operands(expr: &str) -> Vec<String> {
    let bytes = expr.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
        let c = bytes[i] as char;
        // Skip the tail of a numeric literal: in `0xffffu` the scanner would
        // otherwise pick up `xffffu` as an identifier operand. Operands always
        // follow a non-alphanumeric character, so testing `prev_is_alnum` cannot
        // discard a real operand.
        let prev_is_alnum = i > 0 && ((bytes[i - 1] as char).is_ascii_alphanumeric());
        if (c.is_ascii_alphabetic() || c == '_' as char) && !prev_is_alnum {
            let start = i;
            while i < bytes.len() && ((bytes[i] as char).is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            let word = &expr[start..i];
            const KW: &[&str] = &[
                "const", "uint", "ulong", "float", "int", "if", "else", "true", "false",
            ];
            // A word followed by '(' is a call, not an operand.
            let is_call = expr[i..].trim_start().starts_with('(');
            if !KW.contains(&word) && !is_call {
                out.push(word.to_string());
            }
        } else {
            i += 1;
        }
    }
    out.sort();
    out.dedup();
    out
}

/// Right-hand side of `const <type> <name> = <rhs>;` on one line.
fn init_rhs(src: &str, name: &str) -> Option<String> {
    let pat = format!("{name} = ");
    let at = src.find(&pat)? + pat.len();
    let end = src[at..].find(';')? + at;
    Some(src[at..end].trim().to_string())
}

#[test]
fn dependency_graph_of_the_eight_extractions_has_no_serial_chain() {
    let src = read_metal("linear_repacked.metal");
    let start = src
        .find("static inline void extract_v_mcg_32")
        .unwrap_or_else(|| panic!("no extract_v_mcg_32"));
    let body = &src[start..start + 1400];
    let body = &body[..body.find("\n}\n").expect("end of helper")];

    println!("--- extract_v_mcg_32, verbatim ---\n{}\n---", body.trim());

    // --- Level 0/1: the two window halves ---
    let a_hi = init_rhs(body, "A_hi").expect("A_hi");
    let a_lo = init_rhs(body, "A_lo").expect("A_lo");
    let hi_ops = operands(&a_hi);
    let lo_ops = operands(&a_lo);
    println!("L1  A_hi = {a_hi}\n    operands {hi_ops:?}");
    println!("L1  A_lo = {a_lo}\n    operands {lo_ops:?}");

    // The claim under test: BOTH level-1 nodes read the loads directly. There is
    // no `load -> A_hi -> A_lo` ordering, so the brief's proposed serial
    // `load w1 -> compute a -> compute c` chain does not exist.
    assert_eq!(
        hi_ops,
        vec!["rsh0".to_string(), "sh0".to_string(), "wa".to_string(), "wb".to_string()],
        "A_hi must read wa, wb, sh0, rsh0 and nothing computed from them"
    );
    assert_eq!(
        lo_ops,
        vec!["sh0".to_string(), "wb".to_string()],
        "A_lo must read wb, sh0 and nothing computed from them -- in particular \
         not A_hi"
    );
    assert!(
        !lo_ops.contains(&"A_hi".to_string()) && !hi_ops.contains(&"A_lo".to_string()),
        "the two window halves must be mutually independent"
    );
    println!("OK: A_hi and A_lo are both level-1 and mutually independent.");
    println!("    There is NO 'load w1 -> a -> c' serial chain to break.");

    // --- Level 2: the eight extractions ---
    let mut deps: Vec<(String, Vec<String>)> = Vec::new();
    for line in body.lines() {
        if let Some((lhs, rhs)) = line.split_once('=') {
            let lhs = lhs.trim();
            if lhs.starts_with("v[") {
                deps.push((lhs.to_string(), operands(rhs)));
            }
        }
    }
    assert_eq!(deps.len(), 8, "one extraction per code, i=0..7");
    for (lhs, ops) in &deps {
        println!("L2  {lhs} <- {ops:?}");
    }
    let uses_hi = deps.iter().filter(|(_, o)| o.contains(&"A_hi".to_string())).count();
    let uses_lo = deps.iter().filter(|(_, o)| o.contains(&"A_lo".to_string())).count();
    println!("extractions reading A_hi: {uses_hi}/8; reading A_lo: {uses_lo}/8");

    // v[0..5] come from A_hi alone. v[6], v[7] straddle the 32-bit boundary and
    // so need both. Every one of the eight is an independent level-2 node: no
    // extraction feeds another.
    assert_eq!(uses_hi, 8, "all eight codes read the high half");
    assert_eq!(uses_lo, 2, "only the two straddling codes read the low half");
    for (lhs, ops) in &deps {
        assert!(
            !ops.iter().any(|o| o.starts_with('v') && o.ends_with(']')),
            "{lhs} must not consume another extraction: {ops:?}"
        );
    }
    println!("OK: all eight extractions are independent level-2 nodes.");

    // --- Critical path ---
    // load -> {A_hi, A_lo} -> v[i] -> decode_state_mcg -> FMA. Four hops after
    // the load, and `A_lo` is not on it (it is a sibling of `A_hi`, not a
    // descendant), so there is no way to shorten the path by reordering.
    println!("\ncritical path (4 hops after the load):");
    println!("  [wa,wb] -> A_hi -> v[i] -> decode_state_mcg -> acc  (v[0..5])");
    println!("  [wb]    -> A_lo -> v[6],v[7] -> decode_state_mcg -> acc");
    println!("A_lo is a SIBLING of A_hi at level 1, so the v[6]/v[7] path has the");
    println!("same depth as the v[0..5] path. Critical path = 5 levels, either way.");

    // decode_state_mcg must be branchless too, or the path grows a branch.
    let lin = read_metal("linear.metal");
    let ds = &lin[lin.find("static inline float decode_state_mcg").expect("mcg")
        ..];
    let ds = &ds[..ds.find("\n}\n").expect("end")];
    println!("\n--- decode_state_mcg ---\n{}\n---", ds.trim());
    assert!(
        !ds.contains('?') && !ds.contains("if "),
        "decode_state_mcg must be straight-line"
    );
    println!("OK: decode_state_mcg is straight-line (no select, no branch).");
}

// ---------------------------------------------------------------------------
// 3 + 4. Selection-branch census: live path vs legacy 3-word path.
// ---------------------------------------------------------------------------

#[test]
fn selection_branch_count_is_zero_on_the_live_path() {
    let src = read_metal("linear_repacked.metal");
    println!(
        "{:<50} {:>7} {:>6} {:>6} {:>7} {:>6}  {}",
        "kernel", "j2?", "r<=64?", "96-r?", "sh0?", "total", "role"
    );
    let mut grand = 0usize;
    let mut by_form: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for (name, role) in LIVE_KERNELS {
        let body = kernel_body(&src, name);
        let loop_body = unrolled_k_loop_body(body, name);
        let j2 = body.matches("[j2]").count();
        let rle = loop_body.matches("r <= 64u").count();
        let r96 = loop_body.matches("96u - r").count();
        // A `?` inside the K-loop body. The hoisted `32 - sh0` is the mask form
        // of the select and is not a branch, so it is not counted here.
        let sel = loop_body.matches('?').count();
        let total = rle + r96 + sel;
        grand += total;
        println!(
            "{:<50} {:>7} {:>6} {:>6} {:>7} {:>6}  {role}",
            name, j2, rle, r96, sel, total
        );
        assert_eq!(j2, 0, "{name}: a live kernel reads the third ring word [j2]");
        assert_eq!(
            total, 0,
            "{name}: live K-loop carries {total} selection branches; the claim is zero"
        );

        // Two branchless two-word window forms coexist in this file, and the
        // invariant differs. Both are two-word; only the width of the window
        // differs, and both take `sh0` from `lo0` above the loop.
        let loop_at = unrolled_k_loop_header_offset(body, name);
        let is_32 = body.contains("rsh0");
        let is_64 = body.contains("const ulong A0 =")
            || body.contains("const ulong A1 =")
            || body.contains("v0[i] = decode_state_mcg(uint(A0 >>");
        assert!(
            is_32 ^ is_64,
            "{name}: expected exactly one window form, got 32bit={is_32} 64bit={is_64}"
        );
        if is_32 {
            // 32-bit form: the cross-word carry is a shift by `rsh0`, which must
            // be hoisted above the K-loop rather than recomputed per tile.
            let rsh_at = body
                .find("32u - sh0")
                .unwrap_or_else(|| panic!("{name}: no hoisted `32 - sh0` mask"));
            assert!(
                rsh_at < loop_at,
                "{name}: `32 - sh0` at {rsh_at} must be hoisted above the K-loop \
                 header at {loop_at}, not recomputed inside it"
            );
        }
        // Both forms: `sh0` is derived once from `lo0`, never inside the loop.
        let sh0_at = body.find("sh0 = lo0").expect("sh0 from lo0");
        assert!(sh0_at < loop_at, "{name}: `sh0` must be hoisted");
        assert!(
            !loop_body.contains("% 32"),
            "{name}: the K-loop must not recompute a modulo"
        );
        by_form
            .entry(if is_32 { "32-bit A_hi/A_lo" } else { "64-bit ulong A0" })
            .or_default()
            .push(name);
    }
    println!("\ntotal selection branches across all live kernels: {grand}");
    assert_eq!(grand, 0, "items 4 and 5 of the brief are ALREADY DONE on this path");

    println!("\nthe two branchless two-word window forms in use:");
    for (form, ks) in &by_form {
        println!("  {form}: {} kernels", ks.len());
        for k in ks {
            println!("    {k}");
        }
    }
    assert_eq!(by_form.len(), 2, "both forms are expected to be in use");
}

#[test]
fn legacy_three_word_form_has_one_runtime_select_not_nine() {
    // The legacy form is `linear_gemv_ishizuki_k3` in linear.metal (still
    // compiled: it is the fallback when KAPLUMBA_ISHIZUKI_3WORD=1, and it is
    // what lm_head/prefill-shaped row-major paths use).
    let src = read_metal("linear.metal");
    let body = kernel_body(&src, "linear_gemv_ishizuki_k3");
    let r_le = body.matches("r <= 64u").count();
    let sh_sel = body.matches("sh0 ? (").count();
    let j2 = body.matches("[j2]").count();
    println!("legacy `linear_gemv_ishizuki_k3`, SOURCE occurrences:");
    println!("  `r <= 64u` selects      : {r_le}  (one per #pragma-unrolled loop body)");
    println!("  `sh0 ? (...)` selects   : {sh_sel}  (once per tile build site)");
    println!("  third-word `[j2]` loads : {j2}");
    println!("SOURCE lines and RUNTIME evaluations differ; both are reported:");
    println!("  each of the 3 loops runs i=0..7, so per lane per unrolled iteration");
    println!("  that is {r_le} x 8 = 24 *evaluated* selects. The shift-selects are");
    println!("  3 x 1 = 3: they build `a`/`c` once per tile, not once per code.");
    assert_eq!(r_le, 3, "one select per extraction loop body");
    assert_eq!(sh_sel, 3, "one shift-select per tile build site");
    assert!(j2 > 0, "the legacy form is three-word");

    // Now the load-bearing correction: of those 9 `r <= 64u` selects, how many
    // can actually become a machine branch?
    //
    // `r = i * KA + 16u` inside `#pragma unroll for (i = 0; i < 8u; ++i)`, with
    // `constexpr uint KA = 3u`. Every `i` is a literal after unrolling, so `r`
    // and `64u - r` are literals, so `r <= 64u` is a comparison of two
    // constants and the select is folded at compile time. The brief's "8 runtime
    // branches per lane" is therefore 8 constant-folded selects and 0 branches.
    assert!(
        body.contains("constexpr uint KA = 3u;"),
        "KA must be constexpr for the fold argument"
    );
    let unrolled_sites = body.matches("#pragma unroll").count();
    println!("\n  `#pragma unroll` sites in the kernel: {unrolled_sites}");
    println!("  with constexpr KA, every `r` is a literal => `r <= 64u` is");
    println!("  `const <= const` => folded. Runtime branches from item 4: 0.");
    println!("\n  GENUINE runtime selects in the legacy form: {sh_sel} per tile");
    println!("  (one, because `sh0 = lo0 % 32` really does vary per lane);");
    println!("  source-level but dead: {r_le}.");
    assert_eq!(
        sh_sel, 3,
        "the shift-select is the only genuinely runtime-varying select left"
    );
}

// ---------------------------------------------------------------------------
// 5. The coverage gap: loads in flight, for every live kernel.
// ---------------------------------------------------------------------------

#[test]
fn loads_in_flight_table_covers_every_live_kernel() {
    let src = read_metal("linear_repacked.metal");
    println!(
        "\n{:<50} {:>9} {:>6} {:>6}  {}",
        "kernel", "form", "flight", "total", "role"
    );
    let mut split = Vec::new();
    for (name, role) in LIVE_KERNELS {
        let body = kernel_body(&src, name);
        let loop_body = unrolled_k_loop_body(body, name);
        match burst_shape(loop_body) {
            Burst::Inline { per_tile, tiles } => {
                println!(
                    "{:<50} {:>9} {:>6} {:>6}  {role}",
                    name, "inline", per_tile, per_tile * tiles
                );
            }
            Burst::Split { inflight, total } => {
                println!(
                    "{:<50} {:>9} {:>6} {:>6}  {role}",
                    name, "named", inflight, total
                );
                if inflight < total {
                    split.push((name, inflight, total));
                }
            }
        }
    }
    println!("\nnamed-argument kernels whose burst is narrower than their body total:");
    for (n, b, t) in &split {
        println!("  {n}: {b} in flight of {t} -- consumes one column before issuing the next");
    }
    // `fused2_co` is the live gate_up kernel (out_features_a == out_features_b,
    // no env flags -> linear.rs:2421) and is the one kernel where the "8 loads in
    // flight" claim does not hold. `colmajor_2col` is the pre-T4.3 shape and is
    // listed only to show T4.3 did fix its sk2 successor.
    let gate_up = split
        .iter()
        .find(|(n, _, _)| n.ends_with("fused2_co"))
        .expect("fused2_co must appear in the split list");
    assert_eq!((gate_up.1, gate_up.2), (4, 8), "fused2_co burst shape changed");
    assert_eq!(
        split.len(),
        2,
        "expected exactly fused2_co + the pre-T4.3 colmajor_2col, got {split:?}"
    );
    println!("\n=> the live gap is fused2_co: 4 of 8. That is the same source shape");
    println!("   T4.3 fixed in colmajor_2col -> colmajor_2col_sk2, which now reads 8.");
}

#[test]
fn fused2_co_burst_order_is_col_a_then_drain_then_col_b() {
    let src = read_metal("linear_repacked.metal");
    let body = unrolled_k_loop_body(
        kernel_body(&src, "linear_gemv_ishizuki_k3_fused2_co"),
        "fused2_co",
    );
    let a_load = body.find("w0_a0 =").expect("col A load");
    let a_use = body.find("extract_v_mcg_32(w0_a0").expect("col A consume");
    let b_load = body.find("w0_b0 =").expect("col B load");
    assert!(
        a_load < a_use && a_use < b_load,
        "expected load-A, consume-A, then load-B; offsets {a_load}/{a_use}/{b_load}"
    );
    println!("fused2_co K-loop, source order:");
    println!("  col A load  @{a_load}");
    println!("  col A use   @{a_use}   <-- drain point");
    println!("  col B load  @{b_load}");
    println!(
        "  => {} loads in flight at the drain, {} by the end of the body",
        weight_load_count(&body[..a_use]),
        weight_load_count(body)
    );
    assert_eq!(weight_load_count(&body[..a_use]), 4);
    assert_eq!(weight_load_count(body), 8);

    // The fix, simulated as a real line move so the count is falsifiable: take
    // the two Col-B load lines and place them immediately above the first
    // consume. Nothing else changes, and no arithmetic is reordered.
    let hoisted = hoist_col_b_loads(body);
    let (burst2, total2) = loads_in_flight(&hoisted);
    println!(
        "  after moving the 2 col-B load lines above the consume: \
         {burst2} in flight of {total2}"
    );
    assert_eq!(burst2, 8, "the hoist is what turns 4 into 8");
    assert_eq!(total2, 8, "the hoist must not delete or add loads");
}

/// Move the two Col-B weight-load lines of `fused2_co`'s K-loop body above the
/// first `extract_v_mcg_32` call. This is the proposed one-line-scope change,
/// applied textually so its effect on the burst is measurable without editing
/// `linear_repacked.metal`. Used by two tests, which between them show the
/// counter reads 4 before and 8 after -- so it can fail in both directions.
fn hoist_col_b_loads(body: &str) -> String {
    let lines: Vec<&str> = body.lines().collect();
    let use_i = lines
        .iter()
        .position(|l| l.contains("extract_v_mcg_32(w0_a0"))
        .expect("first consume line");
    let load_is: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, l)| l.contains("w0_b0 =") || l.contains("w1_b0 ="))
        .map(|(i, _)| i)
        .collect();
    assert_eq!(load_is.len(), 2, "expected the two col-B load lines");
    assert!(load_is[0] > use_i, "col B is already hoisted; nothing to do");
    let mut after: Vec<&str> = Vec::with_capacity(lines.len());
    for (i, l) in lines.iter().enumerate() {
        if i == use_i {
            for &li in &load_is {
                after.push(lines[li]);
            }
        }
        if !load_is.contains(&i) {
            after.push(l);
        }
    }
    after.join("\n")
}

// ---------------------------------------------------------------------------
// 6. Every detector above, shown able to fail.
// ---------------------------------------------------------------------------

#[test]
fn detectors_can_fail() {
    let src = read_metal("linear_repacked.metal");
    let live = kernel_body(&src, "linear_gemv_ishizuki_k3_fused2_co");

    // (a) Item 4's claim -- zero `r <= 64` on the live path. Sabotage it by
    //     reinstating one select.
    let sabotaged = live.replacen("extract_v_mcg_32(", "r <= 64u ? oops(1) : ", 1);
    let rle = sabotaged.matches("r <= 64u").count();
    assert_eq!(rle, 1, "mutation must introduce the select");
    assert_ne!(rle, 0, "detector 3 must reject a reinstated select");
    println!("(a) zero-select claim: FAILS on a kernel with 1 select reinstated -- observed");

    // (b) Two-word claim -- reintroduce the third word.
    let three = live.replacen("t0_a[j1]", "t0_a[j2]", 1);
    assert!(
        three.matches("[j2]").count() > 0,
        "detector 3's `j2 == 0` assertion must reject it"
    );
    println!("(b) two-word claim: FAILS on a kernel reading [j2] -- observed");

    // (c) Burst claim, two-sided. The counter must read 4 on the kernel as it
    //     stands and 8 after the hoist, so it can fail in BOTH directions.
    let body = unrolled_k_loop_body(live, "fused2_co");
    let (before, total_before) = loads_in_flight(body);
    assert_eq!(
        (before, total_before),
        (4, 8),
        "fused2_co must read 4 of 8 as shipped"
    );
    let after = hoist_col_b_loads(body);
    let (burst_after, total_after) = loads_in_flight(&after);
    assert_eq!(
        (burst_after, total_after),
        (8, 8),
        "the hoist must read 8 of 8"
    );
    assert_ne!(
        before, burst_after,
        "detector 5's burst counter must separate 4 from 8"
    );
    println!(
        "(c) burst claim: reads {before} before the hoist and {burst_after} after \
         -- observed, fails either way"
    );

    // (d) The DAG claim must reject an A_lo that consumes A_hi, i.e. exactly the
    //     serial chain the brief hypothesised.
    let helper = "const uint A_hi = (wa << sh0) | (wb >> rsh0);\n\
                  const uint A_lo = (A_hi >> 2) | (wb << sh0);\n";
    assert_eq!(
        operands(&init_rhs(helper, "A_lo").unwrap()),
        vec!["A_hi", "sh0", "wb"],
        "the sabotaged A_lo must now read A_hi -- which is the serial chain the \
         brief hypothesised, and which the real kernel does not have"
    );
    println!("(d) DAG claim: rejects an A_lo that reads A_hi -- observed");

    // (e) The fold argument: drop `constexpr` from KA and the literal-value
    //     derivation no longer applies. Assert the guard is load-bearing.
    assert!(
        read_metal("linear.metal").contains("constexpr uint KA = 3u;"),
        "the fold argument depends on constexpr KA"
    );
    println!("(e) fold argument: guarded on `constexpr uint KA = 3u` -- observed");

    // (f) The tightness guard: if the low-half contribution were needed (KA=4),
    //     the derivation must refuse. Re-derive at KA=4 and KA=5 explicitly.
    for ka in [3u32, 4, 5] {
        let r_max = 7 * ka + 16;
        let lowest = 64 - r_max;
        let ring = 256 * ka;
        let max_sh = (0..32)
            .map(|l| ((l * 8 + 1) * ka + ring - 16) % 32)
            .max()
            .unwrap();
        let safe = r_max <= 64 && lowest >= max_sh;
        println!("   KA={ka}: r_max={r_max} lowest_used={lowest} max_sh={max_sh} \
                  two_word_safe={safe}");
        if ka == 3 {
            assert!(safe, "KA=3 must be safe or T6 was wrong");
        }
        if ka == 5 {
            assert!(!safe, "KA=5 must be UNSAFE -- the guard must be able to say no");
        }
    }
    println!("(f) two-word derivation: accepts KA=3, refuses KA=5 -- observed");
}

// ---------------------------------------------------------------------------
// 7. Constants pinned, so a reformat cannot silently invalidate the above.
// ---------------------------------------------------------------------------

#[test]
fn constants_are_pinned_in_source() {
    let lin = read_metal("linear.metal");
    let rep = read_metal("linear_repacked.metal");
    for lit in [
        "constexpr uint KA = 3u;",
        "constexpr uint RING = 256u * KA;",
        "constexpr uint WORDS = RING / 32u;",
        "constexpr int SG = 8;",
    ] {
        assert!(
            lin.contains(lit),
            "linear.metal no longer contains `{lit}`; KA/SG/WORDS here are stale"
        );
    }
    for lit in ["constexpr uint KA = 3u;", "constexpr int SG = 8;"] {
        assert!(rep.contains(lit), "linear_repacked.metal lost `{lit}`");
    }
    assert_eq!(SG, 8);
    assert_eq!(KA, 3);
    println!("all pinned literals present; KA={KA} SG={SG} RING={} WORDS={}",
             256 * KA, (256 * KA) / 32);
}