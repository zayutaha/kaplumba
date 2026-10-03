//! PERF PROBE (read-only, no GPU): the memory pipeline inside one GEMV's K-loop.
//!
//! Scope: `linear_repacked.metal`'s live 3 bpw column-major decode kernels. The
//! question this file answers is narrow and is answered from source, not from a
//! benchmark: **how many global loads does each thread have in flight inside the
//! K-loop, and how many distinct bytes do those loads actually touch?**
//!
//! It touches no GPU and no model. Everything below is either (a) a transcription
//! of index arithmetic copied out of the kernel, checked against the kernel's own
//! text, or (b) arithmetic on constants copied out of the kernel. That is
//! deliberate: AGENTS.md §3.4 requires that a claim be shown able to fail, and the
//! only way to show that without editing `linear_repacked.metal` (this probe is
//! read-only) is to make every detector a pure function over a source string and
//! then feed it a mutated copy of that string. `detectors_can_fail` does exactly
//! that, per detector, at the bottom of this file.
//!
//! ## Result summary (all of it derived below, all of it printed)
//!
//! | quantity | value |
//! |---|---|
//! | **global weight loads in flight per thread, every live K-loop** | **8** |
//! | activation (`x`) loads in flight in the same window | 8 (16 in `_fused2_sk2`) |
//! | **cross-iteration software pipelining (prefetch next tile)** | **none** |
//! | distinct words touched per 16x16 tile by the 32 lanes | **24 of 24** |
//! | lane-requests per tile | **64** |
//! | **lane-load amplification** | **64/24 = 2.667** |
//! | **DRAM line traffic per column** | **1.000x** (one partial line) |
//! | words per lane | **2** (`j0`, `j1`) -- not 3 |
//! | **redundant reads between adjacent columns `tn`, `tn+1`** | **0** |
//! | threadgroup staging of a tile in the live kernels | **none** |
//! | `simd_shuffle` used to share *loaded words* | **none** (accumulators only) |
//! | latency the in-flight bytes can hide, Little's law | 675 ns @ 3 tg/core |
//!
//! ## The one concrete change this points at, and why it is small
//!
//! **Cross-iteration register prefetch** in the two heaviest loops, i.e. hoist
//! iteration `tk + step`'s eight weight loads to the tail of iteration `tk`'s
//! body so the steady state is 16 in flight instead of a burst of 8 followed by
//! an ALU half with 0 in flight. The sites are
//! `linear_repacked.metal:2477-2480` (`colmajor_2col_sk2`, down_proj + o_proj)
//! and `:506-509` (`fused2_2col`, gate_up), with `:2463-2497` / `:491-528` as the
//! loops to rewrite.
//!
//! **Estimated effect: +0.0 to +0.3 tok/s (0 to +0.5%), i.e. at or below the
//! harness noise floor.** The reasoning, so the estimate can be argued with
//! rather than taken on faith:
//!
//! * Little's law says latency is already covered ~7x. 384 B of distinct bytes
//!   per simdgroup x 480 resident simdgroups = 180 KB, which is 675 ns of
//!   latency headroom at the 273 GB/s wire rate. Apple DRAM latency is ~100 ns.
//!   Going to 16 in flight moves 675 ns to 1350 ns, and nothing observable sits
//!   between 100 ns and 1350 ns.
//! * The closest existing experiment already ran and was flat: deeper K-unroll
//!   (`_u8`, more loads in flight) is **1.01-1.03x** isolated and
//!   **-0.55%/+0.62%** in engine (`decode-19tok-repacked-250gb/spec.md:152`).
//! * It costs 8 registers on a kernel whose occupancy T4.3 bought by cutting to
//!   <=30 registers (`decode-25tok-peak-bandwidth/...:145`), on a kernel where
//!   T4.2's 4col lost 4.2% purely from a second accumulator pair (same file
//!   :122-128). Occupancy, not MLP, is the binding resource here, and prefetch
//!   spends it.
//!
//! If it is built, it must be A/B'd at N>=16 interleaved pairs: the noise floor
//! of this harness is 0.01-0.08 tok/s and the expected effect is inside it
//! (`decode-25tok-peak-bandwidth/...:315`). Anything above 0.1 tok/s is signal.
//!
//! ## What is NOT worth doing (all measured, all losses)
//!
//! * **Share loaded words between lanes with `simd_shuffle`.** Every lane reads
//!   the *same* 37-bit window at 32 different rotations (`decode-3bpw-
//!   amplification/implementation_notes.md` §3a:119-138), so there is a real
//!   sharing opportunity -- 8.667 distinct lane-windows per 64 requests. It was
//!   built: **180.5 -> 146.5 GB/s**. The crossbar serialises on `j0`, which
//!   differs per thread (`decode-19tok-repacked-250gb/...:64-67`).
//! * **Stage a whole tile in threadgroup memory** (`linear_gemv_ishizuki_k3_
//! colmajor_u4`, `linear_repacked.metal:3171-3276`, `tg_tiles[SG][96]`, 2
//!   barriers per 4 tiles). This *is* the tile-local-shared-memory idea in the
//!   brief, it exists in the tree, and it was measured: **180.5 -> 173.7 GB/s on
//!   `gate_up`** (`decode-19tok-repacked-250gb/tasks.md:74`). Threadgroup ring
//!   staging was a wash (118.8 vs 119.5). It costs a barrier per tile, and the
//!   96 B tile is one or two cache lines either way.
//! * **Anything that reduces requests at fixed bytes.** The two-word window is
//!   forced: `decode_state_mcg` multiplies *before* it masks, so all 16 input
//!   bits affect the output and a lane cannot be reconstructed from its own 8
//!   codes. `D=32` (which halves amplification to 1.333 directly) ran
//!   **0.288-0.918x on all 11 shapes** (`decode-19tok-repacked-250gb/...:468-477`).
//! * **Fewer output columns per threadgroup.** 4col lost 4.2% to occupancy
//!   (T4.2), 1col lost 0.16 tok/s to 2col (T5.5), threadgroup width is flat
//!   within -5.1%..+1.9% across a 120-cell (SK x TG) grid.
//!
//! The three findings that matter:
//!
//! 1. **DRAM traffic is already 1.000x.** The lanes' 2.667x is *request*
//!    redundancy, and adjacent tiles share 128-byte lines, so there are no wasted
//!    bytes to reclaim. Anything that tries to reclaim them has to remove
//!    *requests*, and every such attempt in this tree has already been measured
//!    (see the header of `spec/decode-3bpw-amplification/implementation_notes.md`
//!    §3a and `spec/decode-19tok-repacked-250gb/implementation_notes.md:451-453`).
//! 2. **8 weight loads in flight, with no cross-iteration prefetch.** The burst
//!    is issued at the top of the loop body and consumed in the same body, so the
//!    steady state is "8 in flight, then 0 in flight for the whole ALU half".
//!    This is the only genuine structural gap found.
//! 3. **Raising it is probably not worth it, and here is the number.** See
//!    `little_law_says_the_current_mlp_is_ample` and the concrete-change section
//!    below: the register file needed for a deeper prefetch is the same resource
//!    T4.2/T4.3 identify as the binding one.
//!
//! A fourth, incidental but load-bearing finding: **T4.3's "burst loading" is a
//! 4 -> 8 loads-in-flight hoist, and the pre-T4.3 shape is still in the tree.**
//! `linear_gemv_ishizuki_k3_colmajor_2col` consumes column 0's four words before
//! it has issued column 1's, while `_colmajor_2col_sk2` (and every other live
//! kernel) hoists all eight above the first consume. That is the whole
//! difference between the two, and it is the only thing T4.3 changed about the
//! memory pipeline.
//!
//! T4.3's reported result is **14.58 -> 14.85 tok/s**, 32-token greedy decode
//! (`decode-25tok-peak-bandwidth/implementation_notes.md` §9:149), i.e. +1.9%.
//! Two caveats, both from that same spec file: the edit also changed
//! `step` to `SG * 2u`, so the hoist was never A/B'd on its own; and §16.1
//! (:315-320) later established that this spec's before/after pairs are in the
//! noise -- two arms of the *same* binary differ by 0.010 tok/s, so the history's
//! 15.33/15.41/... readings were different code, and "T5.2's claimed
//! 15.33 -> 15.75 was not established". +0.27 tok/s is 3-27x the 0.01-0.08 noise
//! floor, so unlike T5.2's this one is probably real, but it is a single
//! unreplicated pair from a session whose other pairs were later retracted.
//!
//! ## Provenance of the "already measured" numbers quoted above
//!
//! Not measured here. Cited, with the measurement that produced each:
//!
//! * lane-load amp is structurally 2.667 at 3 bpw -- three independent
//!   derivations in `spec/decode-3bpw-amplification/implementation_notes.md`
//!   §3a (lines 119-138): `decode_state_mcg` multiplies before it masks, so each
//!   lane's 16-bit window depends on 37 ring bits, and the union over all 32 lanes
//!   is also 37; 37 > 32 forces 2 words per lane.
//! * `simd_shuffle` cooperative loads: 180.5 -> 146.5 GB/s
//!   (`spec/decode-19tok-repacked-250gb/implementation_notes.md:64-67`).
//! * threadgroup staging (`linear_gemv_ishizuki_k3_colmajor_u4`): 180.5 -> 173.7
//!   GB/s on `gate_up` (`.../tasks.md:74`); threadgroup ring staging 118.8 vs
//!   119.5 = no change (`.../implementation_notes.md:453`).
//! * deeper K-unroll (`_u8`, i.e. *more loads in flight*): 1.01-1.03x isolated,
//!   -0.55%/+0.62% in engine (`spec/decode-19tok-repacked-250gb/spec.md:152`).
//! * T4.3 itself: 14.58 -> 14.85 tok/s, 32-token greedy decode
//!   (`spec/decode-25tok-peak-bandwidth/implementation_notes.md` §9:138-150).
//! * DRAM line traffic is 1x at every rate: asserted by the existing test
//!   `crates/engine/tests/rate4_colmajor_bit_exact.rs:746`. Re-derived here
//!   independently as `dram_line_traffic_per_column_is_one_x`.

use std::collections::BTreeMap;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Kernel constants, transcribed. KA/RING/WORDS from `linear_repacked.metal`
// :2417-2420 (`colmajor_2col_sk2`) and :446-449 (`fused2_2col`); `SG` from
// :2426. `checked_against_source` below asserts each literal is still present.
// ---------------------------------------------------------------------------

/// EXL3 bit rate. `constexpr uint KA = 3u;`
const KA: u32 = 3;
/// Bits per 16x16 tile: 256 codes x KA. `constexpr uint RING = 256u * KA;`
const RING: u32 = 256 * KA;
/// 32-bit words per tile. `constexpr uint WORDS = RING / 32u;`
const WORDS: usize = (RING / 32) as usize;
/// Bytes per tile: 96 at KA=3, 128 at KA=4, 192 at KA=6.
const TILE_BYTES: usize = WORDS * 4;
/// Simdgroups per threadgroup. `constexpr int SG = 8;` -> 256 threads.
const SG: usize = 8;
/// Threads per threadgroup, as dispatched (`linear.rs:1250-1258`, `groups * 256,
/// 256`).
const THREADS_PER_TG: usize = SG * 32;
/// GPU cores on the target box (AGENTS.md §8).
const CORES: usize = 20;
/// Wire bandwidth, AGENTS.md §8.
const WIRE_GBPS: f64 = 273.0;

/// A path relative to the workspace root.
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

/// Read a `.metal` file with comments blanked out (length-preserving), so every
/// structural scan below is brace- and paren-accurate and cannot be fooled by
/// prose in a comment block.
fn read_metal(name: &str) -> String {
    let path = workspace_path(&format!("crates/kernels/src/{name}"));
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    strip_comments(&raw)
}

/// Replace comment bodies with spaces, preserving byte length, so that offsets
/// found in the cleaned copy index the original. Brace matching must not be
/// confused by a `{` inside a comment -- and `linear_repacked.metal` has many.
fn strip_comments(src: &str) -> String {
    let mut out: Vec<u8> = src.as_bytes().to_vec();
    let bytes = src.as_bytes();
    let mut i = 0usize;
    let mut in_line = false;
    let mut in_block = false;
    let mut in_str = false;
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

/// Byte span of the `kernel void <name>(` ... matching `}` region, braces matched
/// so a body cannot run past its own end into the next kernel. `src` must already
/// have been through `strip_comments`.
fn kernel_body<'a>(src: &'a str, kernel: &str) -> &'a str {
    let open = format!("kernel void {kernel}(");
    let start = src
        .find(&open)
        .unwrap_or_else(|| panic!("kernel {kernel} not found"));
    let bytes = src.as_bytes();

    // Close the signature.
    let mut paren = 0i32;
    let mut sig_end = None;
    for (i, &b) in bytes[start..].iter().enumerate() {
        match b {
            b'(' => paren += 1,
            b')' => {
                paren -= 1;
                if paren == 0 {
                    sig_end = Some(start + i);
                    break;
                }
            }
            _ => {}
        }
    }
    let sig_end = sig_end.unwrap_or_else(|| panic!("{kernel}: unbalanced signature"));
    let brace = src[sig_end..]
        .find('{')
        .map(|i| sig_end + i)
        .unwrap_or_else(|| panic!("{kernel}: no body"));

    let mut depth = 0i32;
    for (i, &b) in bytes[brace..].iter().enumerate() {
        match b {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &src[brace..=brace + i];
                }
            }
            _ => {}
        }
    }
    panic!("{kernel}: unbalanced body braces");
}

/// Body of the main unrolled K-loop: from `for (; tk < k_unroll_end;` to the brace
/// that closes it. Braces are matched rather than indentation-delimited, because
/// the loop sits at different indentation depths in different kernels
/// (`_fused2_sk2` is straight-line code at 4 spaces, `_2col` is inside a `tg`
/// loop at 8) and an indentation rule silently truncates one of them.
fn unrolled_loop_body<'a>(kernel_src: &'a str, kernel: &str) -> &'a str {
    let body = kernel_body(kernel_src, kernel);
    let anchor = "for (; tk < k_unroll_end;";
    let start = body
        .find(anchor)
        .unwrap_or_else(|| panic!("{kernel}: unrolled loop `{anchor}` not found"));
    let brace = body[start..]
        .find('{')
        .map(|i| start + i)
        .unwrap_or_else(|| panic!("{kernel}: unrolled loop has no body"));
    let mut depth = 0i32;
    for (i, &b) in body.as_bytes()[brace..].iter().enumerate() {
        match b {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &body[start..=brace + i];
                }
            }
            _ => {}
        }
    }
    panic!("{kernel}: unrolled loop braces unbalanced");
}

// ---------------------------------------------------------------------------
// Detector 1: global word-loads in flight per thread.
// ---------------------------------------------------------------------------

/// True only for a line that *declares* threadgroup memory, i.e.
/// `threadgroup <type> name[...]`. Not for `threadgroup_barrier` and not for the
/// `[[thread_position_in_threadgroup]]` attribute.
fn declares_threadgroup_memory(line: &str) -> bool {
    ["threadgroup float", "threadgroup uint", "threadgroup int", "threadgroup half"]
        .iter()
        .any(|p| line.contains(p))
}

/// True when the last `acc` in the loop body comes after the last weight-load
/// subscript, i.e. the body's final act is arithmetic on words it already holds.
fn last_acc_after_loads(loop_body: &str) -> bool {
    let acc = loop_body.rfind("acc").expect("accumulation in loop");
    let load = loop_body.rfind("[j0]").or_else(|| loop_body.rfind("[j1]"));
    match load {
        Some(l) => acc > l,
        None => false,
    }
}

/// What a subscript in the K-loop reads from.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum Sub {
    /// `t0_0[j0]`, `column[j1]` -- a 32-bit EXL3 weight word. One global load.
    Weight,
    /// `xr0[0]`, `xr0[8]` -- an activation half. L1/L2 resident.
    Activation,
    /// `v[3]`, `v_0[5]` -- the decoded-weight scratch array, not a load at all.
    Scratch,
    /// Anything else, e.g. `red[0][sg][lane]`.
    Other,
}

/// Classify every `[...]` subscript in `src` by the identifier in front of it.
fn classify_subscripts(src: &str) -> Vec<(usize, Sub)> {
    let bytes = src.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] != b'[' {
            i += 1;
            continue;
        }
        let close = src[i..]
            .find(']')
            .map(|k| i + k)
            .unwrap_or_else(|| panic!("unbalanced '[' at byte {i}"));
        let idx = src[i + 1..close].trim();

        // Walk back over the identifier immediately before the '['.
        let mut j = i;
        while j > 0 && (bytes[j - 1].is_ascii_alphanumeric() || bytes[j - 1] == b'_') {
            j -= 1;
        }
        let ident = &src[j..i];
        let kind = if idx == "j0" || idx == "j1" || idx == "j2" {
            Sub::Weight
        } else if ident.starts_with('x') {
            Sub::Activation
        } else if ident.starts_with('v') {
            Sub::Scratch
        } else {
            Sub::Other
        };
        out.push((i, kind));
        i = close + 1;
    }
    out
}

/// Byte offset of the first place a loaded word is *consumed* -- the first
/// `extract_v_mcg_32` in the `_2col` family, the first `decode_state_mcg` in
/// `_fused2_sk2`. Everything issued before this offset is in flight at the top of
/// the burst. `A0 =` is deliberately NOT the cut point: assembling the 64-bit
/// window `((w0a << 32) | w0b) << sh0` consumes nothing, it only narrows words the
/// lane already holds.
fn first_consume_offset(loop_body: &str) -> usize {
    loop_body
        .find("extract_v_mcg_32(")
        .or_else(|| loop_body.find("decode_state_mcg("))
        .unwrap_or_else(|| panic!("no consume site in the loop body"))
}

/// A `tN_M[jK]` subscript on a `const device uint *` weight pointer is one 32-bit
/// global load. Count the ones issued *before* the first consume, because a load
/// after the first consume is not in flight at the top of the burst.
fn weight_loads_before_first_consume(loop_body: &str) -> usize {
    let cut = first_consume_offset(loop_body);
    classify_subscripts(&loop_body[..cut])
        .iter()
        .filter(|(_, k)| *k == Sub::Weight)
        .count()
}

/// Activation loads (`xr0[0]`, `xr0[1]`, `xr0[8]`, `xr0[9]`, ...) issued before the
/// first consume. These are L1/L2-resident and are not the bandwidth question;
/// they are counted because they occupy the same LSU issue slots.
fn activation_loads_before_first_consume(loop_body: &str) -> usize {
    let cut = first_consume_offset(loop_body);
    classify_subscripts(&loop_body[..cut])
        .iter()
        .filter(|(_, k)| *k == Sub::Activation)
        .count()
}

/// Every weight subscript in the loop body, wherever it sits. Comparing this to
/// `weight_loads_before_first_consume` is what distinguishes "issue all, then
/// consume" (equal) from software pipelining (body total > head).
fn weight_loads_total(loop_body: &str) -> usize {
    classify_subscripts(loop_body)
        .iter()
        .filter(|(_, k)| *k == Sub::Weight)
        .count()
}

#[test]
fn loads_in_flight_is_eight_per_thread_in_every_live_k_loop() {
    // The five kernels that actually run at decode time, with the shapes they run on.
    // `SK`/`TG` come from `linear.rs:1241-1258` (`groups * 256, 256`).
    // (kernel, role, expected weight loads in flight)
    //
    // The last entry is the T4.3 "before" picture and is still in the tree: it is
    // the only live-shaped kernel that consumes col 0's four words *before* it has
    // issued col 1's. Every T4.3-edited kernel hoisted all 8 above the first
    // consume. That hoist -- 4 loads in flight -> 8 -- IS "burst loading".
    let live: &[(&str, &str, usize)] = &[
        (
            "linear_gemv_ishizuki_k3_colmajor_2col_sk2",
            "down_proj 21504x5376, o_proj 4096x5376 (linear.rs:2257)",
            8,
        ),
        (
            "linear_gemv_ishizuki_k3_fused2_2col",
            "gate_up 5376x43008 (encode_gemv_fused2_mlp)",
            8,
        ),
        (
            "linear_gemv_ishizuki_k3_fused2_sk2",
            "gate_up 5376x43008 under KAPLUMBA_USE_SK (linear.rs:2532)",
            8,
        ),
        (
            "linear_gemv_ishizuki_k3_fused3_2col",
            "qkv (encode_gemv_fused3, model.rs:1504)",
            8,
        ),
        (
            "linear_gemv_ishizuki_k3_colmajor_2col",
            "pre-T4.3 shape: consumes col0 before issuing col1 (linear.rs:1397)",
            4,
        ),
    ];

    let src = read_metal("linear_repacked.metal");
    println!(
        "\nKA={KA} RING={RING} WORDS={WORDS} tile={TILE_BYTES} B \
         threads/tg={THREADS_PER_TG} simdgroups/tg={SG}\n"
    );
    println!(
        "{:<48} {:>8} {:>8} {:>7} {:>7}  role",
        "kernel", "weights", "x-loads", "total", "expect"
    );

    for (name, role, want) in live {
        let loop_body = unrolled_loop_body(&src, name);
        let w = weight_loads_before_first_consume(loop_body);
        let a = activation_loads_before_first_consume(loop_body);
        println!(
            "{name:<48} {w:>8} {a:>8} {:>7} {want:>7}  {role}",
            w + a
        );
        assert_eq!(
            w, *want,
            "{name}: expected {want} weight word-loads issued before the first \
             consume in the unrolled K-loop body, counted {w}. A change here is a \
             change to loads-in-flight and every bandwidth number derived from it."
        );
    }

    // Every T4.3-edited loop is 4 tiles x 2 words. Assert that decomposition
    // directly, so a kernel that reached 8 loads some other way (8 tiles x 1 word)
    // would be distinguishable from one that is structurally the same shape.
    let body = unrolled_loop_body(&src, "linear_gemv_ishizuki_k3_colmajor_2col_sk2");
    for t in ["t0_0", "t1_0", "t0_1", "t1_1"] {
        let hits = body.matches(&format!("{t}[j0]")).count()
            + body.matches(&format!("{t}[j1]")).count();
        assert_eq!(hits, 2, "{t}: expected both words loaded exactly once");
    }

    // The T4.3 hoist, stated as a diff in this file's terms: in the pre-T4.3
    // kernel the col-1 loads sit *after* the col-0 consume.
    let pre = unrolled_loop_body(&src, "linear_gemv_ishizuki_k3_colmajor_2col");
    let col1_load = pre.find("const uint w0_1_0").expect("col1 load in _2col");
    let first_consume = pre.find("extract_v_mcg_32(").expect("consume in _2col");
    assert!(
        col1_load > first_consume,
        "_colmajor_2col no longer looks pre-T4.3; if it has been hoisted, the \
         4->8 loads-in-flight comparison in the header no longer holds"
    );
    let post = unrolled_loop_body(&src, "linear_gemv_ishizuki_k3_colmajor_2col_sk2");
    assert!(
        post.find("const uint w0_1_0").unwrap() < post.find("extract_v_mcg_32(").unwrap(),
        "_2col_sk2 should hoist all 8 loads above the first consume"
    );
    println!("\nT4.3's hoist, visible as a source diff between the two kernels:");
    println!("  _colmajor_2col      col1 load at byte offset {col1_load}, first consume at {first_consume}  -> 4 in flight");
    println!("  _colmajor_2col_sk2  col1 load before the first consume                                  -> 8 in flight");
}

// ---------------------------------------------------------------------------
// Detector 2: index arithmetic, transcribed and checked.
// ---------------------------------------------------------------------------

/// `lo0 = (lane * 8u + 1u) * KA + RING - 16u` (linear_repacked.metal:2431).
fn lo0(lane: u32) -> u32 {
    (lane * 8 + 1) * KA + RING - 16
}

/// `sh0 = lo0 % 32u` (:2432).
fn sh0(lane: u32) -> u32 {
    lo0(lane) % 32
}

/// `hi0 = lo0 / 32u` (:2434).
fn hi0(lane: u32) -> u32 {
    lo0(lane) / 32
}

/// `j0 = hi0 >= WORDS ? hi0 - WORDS : hi0` (:2435).
fn j0(lane: u32) -> u32 {
    let h = hi0(lane);
    if h as usize >= WORDS {
        h - WORDS as u32
    } else {
        h
    }
}

/// `j1 = j0 + 1u >= WORDS ? j0 + 1u - WORDS : j0 + 1u` (:2436).
fn j1(lane: u32) -> u32 {
    let a = j0(lane) + 1;
    if a as usize >= WORDS {
        a - WORDS as u32
    } else {
        a
    }
}

/// Byte offset of the column-`tn` base: `packed + tn * (tiles_k * WORDS)`
/// (:2455-2456). Two different `tn` therefore never overlap.
fn column_byte_offset(tn: u32, tiles_k: u32) -> u64 {
    u64::from(tn) * u64::from(tiles_k) * TILE_BYTES as u64
}

#[test]
fn the_lane_window_is_two_words_and_covers_every_word_exactly() {
    let mut touched: BTreeMap<u32, u32> = BTreeMap::new();
    let mut requests = 0usize;
    let mut sh_set = std::collections::BTreeSet::new();

    println!("\n lane   lo0  sh0  hi0  j0  j1");
    for lane in 0..32u32 {
        sh_set.insert(sh0(lane));
        *touched.entry(j0(lane)).or_default() += 1;
        *touched.entry(j1(lane)).or_default() += 1;
        requests += 2;
        println!(
            "{lane:>5} {:>5} {:>4} {:>4} {:>3} {:>3}",
            lo0(lane),
            sh0(lane),
            hi0(lane),
            j0(lane),
            j1(lane)
        );
    }

    // (a) Footprint: the union of every address any lane reads is the whole tile.
    //     This is the load-bearing claim for "there are no wasted bytes".
    assert_eq!(
        touched.len(),
        WORDS,
        "the 32 lanes touch {} distinct words of {WORDS}; a shortfall means part \
         of every tile is never read, which is a correctness bug a bandwidth \
         number would hide",
        touched.len()
    );
    assert!(
        (0..WORDS as u32).all(|w| touched.contains_key(&w)),
        "the touched set is not contiguous 0..WORDS"
    );

    // (b) Requests: 32 lanes x 2 words, each word read 2 or 3 times.
    assert_eq!(requests, 64);
    let per_word: Vec<u32> = (0..WORDS as u32).map(|w| touched[&w]).collect();
    assert!(
        per_word.iter().all(|&c| c == 2 || c == 3),
        "per-word load counts {per_word:?} outside 2..=3; the amp-2.667 structure \
         depends on it"
    );
    assert_eq!(per_word.iter().sum::<u32>(), requests as u32);

    // (c) Amplification = requests / useful words = 8/KA exactly.
    let amp = requests as f64 / WORDS as f64;
    assert!((amp - 8.0 / KA as f64).abs() < 1e-12, "amp {amp} != 8/KA");
    // Cross-check against the value the tree already asserts
    // (rate4_colmajor_bit_exact.rs:687).
    assert!((amp - 64.0 / 24.0).abs() < 1e-12);

    // (d) Only four distinct rotation amounts exist, cycling with period 4 lanes.
    let sh: Vec<u32> = (0..32).map(sh0).collect();
    assert_eq!(sh.len(), 32);
    assert_eq!(
        sh_set.iter().copied().collect::<Vec<_>>(),
        vec![3, 11, 19, 27],
        "sh0 set drifted"
    );
    for lane in 0..32usize {
        assert_eq!(sh0(lane as u32), sh0((lane % 4) as u32));
    }

    println!("\nwords/tile {} ({} B), lane-requests {} -> amp {amp:.4}", WORDS, TILE_BYTES, requests);
    println!("per-word load counts: {per_word:?}");
    println!("sh0 = {{3, 11, 19, 27}}, period 4 lanes");
    println!("\nNote: the 3-word `ring_word` window in the brief is NOT on this path.");
    println!("`exl3_decode8` (linear.metal:279-308, `j0/j1/j2`) has 42 call sites,");
    println!("all inside the row-major `linear_gemv_exl3_k*` generators, i.e. lm_head");
    println!("(6 bpw) and prefill. The live colmajor decode path is 2 words/lane.");
}

#[test]
fn the_three_word_window_belongs_to_the_row_major_path_only() {
    // The brief describes the window as "3 consecutive uint per lane". That is
    // `exl3_decode8` / `ring_word<ALIGNED>` in linear.metal:267-308, and it is the
    // row-major generator family. Check that the colmajor decode kernels really do
    // stop at two words, so a future edit that widens the window is caught here
    // rather than as a bandwidth regression.
    let repacked = read_metal("linear_repacked.metal");
    for kernel in [
        "linear_gemv_ishizuki_k3_colmajor_2col_sk2",
        "linear_gemv_ishizuki_k3_fused2_2col",
        "linear_gemv_ishizuki_k3_fused2_sk2",
        "linear_gemv_ishizuki_k3_fused3_2col",
        "linear_gemv_ishizuki_k3_colmajor_2col",
    ] {
        let body = kernel_body(&repacked, kernel);
        assert!(
            !body.contains("j2"),
            "{kernel} now reads a third word (j2). At 3 bpw the two-word form is \
             legal exactly on the boundary (max sh = 27, 48 - 7*KA = 27 -- see \
             linear_repacked.metal:3321-3335), so a third word is not an \
             optimisation, it is a layout/derivation change that needs its own \
             bit-exact test."
        );
        assert!(
            !body.contains("ring_word"),
            "{kernel} calls ring_word; that is the row-major path's accessor."
        );
    }

    // And the row-major family does use three, so the split is real and not an
    // artefact of this probe looking at one file.
    let linear = read_metal("linear.metal");
    let n_decode8 = linear.matches("exl3_decode8<").count();
    let n_decode8_mcg = linear.matches("exl3_decode8_mcg<").count();
    println!(
        "row-major `linear.metal`: {n_decode8} `exl3_decode8<...>` and \
         {n_decode8_mcg} `exl3_decode8_mcg<...>` call sites, all 3-word\n\
         colmajor `linear_repacked.metal`: 0 -- every live kernel is 2 words/lane\n\
         amp at 3 bpw: row-major 96/24 = 4.000, colmajor 64/24 = 2.667\n\
         amp at 6 bpw: row-major 96/48 = 2.000 -- the lm_head/head-gemv path"
    );
    assert!(
        n_decode8 + n_decode8_mcg >= 40,
        "expected the generated row-major family to still call exl3_decode8, found \
         {n_decode8} + {n_decode8_mcg}"
    );
    // `linear_repacked.metal` does contain a 3-word kernel, and it is a
    // measurement-only arm for 6 bpw -- so the two-word rule above is a rule
    // about the *live* kernels, not about the file. Pin which one it is.
    let three_word: Vec<&str> = repacked
        .split("kernel void ")
        .skip(1)
        .filter(|k| k.contains("j2"))
        .map(|k| k.split(['(', ' ']).next().unwrap_or("?"))
        .collect();
    println!("3-word kernels in linear_repacked.metal: {three_word:?}");
    assert!(
        three_word.iter().all(|k| k.ends_with("_bench")),
        "a non-bench colmajor kernel now reads three words: {three_word:?}. At 6 bpw \
         that is required (max sh = 22 > 48 - 7*KA = 6, linear_repacked.metal \
         :3489-3501), so a 3-word kernel outside the `_bench` family means a rate \
         changed."
    );
    assert_eq!(
        repacked.matches("ring_word<").count(),
        0,
        "ring_word is the row-major accessor and has no business in the colmajor \
         file"
    );
}

#[test]
fn adjacent_output_columns_share_no_bytes() {
    // Answers "does column tn+1 re-read words column tn already loaded?"
    // No: the repack is `[tiles_n, tiles_k, 48]`, so a column is one contiguous
    // run and `tn` and `tn+1` are `tiles_k * TILE_BYTES` apart with no overlap.
    // `reorder_tensor_to_colmajor` is the thing that makes this true --
    // spec/decode-19tok-repacked-250gb/implementation_notes.md:35.
    for tiles_k in [336u32, 1024, 1344] {
        for tn in [0u32, 1, 7, 335, 336] {
            let a = column_byte_offset(tn, tiles_k);
            let b = column_byte_offset(tn + 1, tiles_k);
            assert_eq!(
                b - a,
                u64::from(tiles_k) * TILE_BYTES as u64,
                "columns {tn}/{}+1 must be exactly one column apart",
                tn
            );
            assert!(
                a + u64::from(tiles_k) * TILE_BYTES as u64 <= b,
                "columns {tn} and {} overlap",
                tn + 1
            );
        }
    }
    // Tile-local shared memory therefore has nothing to de-duplicate ACROSS
    // columns. It could only de-duplicate the 2.667x WITHIN a tile (lanes
    // re-reading a neighbour's word), which is the `simd_shuffle` / threadgroup
    // staging experiment -- both already measured, both losses. See the header.
    println!("inter-column redundancy: 0.000x at tiles_k in [336, 1024, 1344]");
}

#[test]
fn dram_line_traffic_per_column_is_one_x() {
    // Independent re-derivation of `rate4_colmajor_bit_exact.rs:746`. The 2.667
    // is requests; the bytes are 1.000x, because consecutive 96 B tiles share
    // 128 B lines for the whole length of a column.
    const LINE: usize = 128;
    println!();
    for tiles_k in [336usize, 1024, 1344] {
        let column_bytes = tiles_k * TILE_BYTES;
        let lines = column_bytes.div_ceil(LINE);
        let ratio = lines as f64 * LINE as f64 / column_bytes as f64;
        println!(
            "tiles_k {tiles_k:>5}: column {column_bytes:>9} B = {lines:>8} lines -> \
             {ratio:.4}x useful bytes, {amp:.4}x requests",
            amp = 64.0 / WORDS as f64
        );
        assert!(
            ratio >= 1.0,
            "line traffic {ratio:.4}x < useful bytes: the kernel would be reading \
             less than the matrix holds"
        );
        assert!(
            ratio <= 1.0 + LINE as f64 / column_bytes as f64,
            "line traffic {ratio:.4}x exceeds useful by more than one line, so \
             there ARE wasted bytes and the 1x claim is false"
        );
    }
    println!("DRAM line traffic is 1.000x up to one partial line per column.");
}

// ---------------------------------------------------------------------------
// Detector 3: the loop shape, and Little's law on the memory-level parallelism.
// ---------------------------------------------------------------------------

#[test]
fn the_loop_is_over_k_tiles_and_each_simdgroup_stream_is_forward_sequential() {
    let src = read_metal("linear_repacked.metal");
    let body = unrolled_loop_body(&src, "linear_gemv_ishizuki_k3_colmajor_2col_sk2");

    // Loop variable and stride.
    assert!(
        body.contains("for (; tk < k_unroll_end; tk += step)"),
        "the unrolled loop is not `for (; tk < k_unroll_end; tk += step)`"
    );
    let kernel = kernel_body(&src, "linear_gemv_ishizuki_k3_colmajor_2col_sk2");
    assert!(kernel.contains("const uint step = SG * 2u;"));
    assert!(
        kernel.contains("uint tk = k_start + sg * 2u;"),
        "the per-simdgroup start is not `k_start + sg * 2u`"
    );

    // Per-thread access: `col0 + (tk + u) * stride` with stride == WORDS, and
    // col1 at a *disjoint* base. Exact index arithmetic:
    //
    //   address(tk, sub, col) = col * tiles_k * 24 + (tk + sub) * 24   [words]
    //
    // so within one iteration the two sub-steps are 24 words apart (adjacent
    // tiles, forward), and between iterations this simdgroup advances
    // step * 24 = 384 words = 1536 B. Across the threadgroup, the 8 simdgroups
    // take tk offsets 0,2,4,...,14, so together they cover 16 consecutive tiles
    // = one contiguous 1536 B run per column. Forward-sequential at the
    // threadgroup granularity, strided at the lane granularity. That is the
    // arrangement the hardware streaming prefetcher wants, and it is the thing
    // T4.3/"Contiguous Tile Load Grouping" arranged (decode-25tok-peak-bandwidth
    // /implementation_notes.md:101-103).
    let tiles_k_down = 21504u32 / 16;
    let tiles_k_gate = 5376u32 / 16;
    for (tk, sub, col) in [(100u32, 0u32, 0u32), (100, 1, 0), (100, 0, 1), (116, 1, 1)] {
        let addr_words =
            u64::from(col) * u64::from(tiles_k_down) * WORDS as u64 + u64::from(tk + sub) * WORDS as u64;
        println!("col {col} tile {tk}+{sub} -> word {addr_words}, byte {}", addr_words * 4);
        assert!(addr_words < u64::from(tiles_k_down) * WORDS as u64 * 2);
    }
    // Step in bytes, per simdgroup, between consecutive iterations.
    let per_iter_bytes = (SG as u32 * 2) * WORDS as u32 * 4;
    println!(
        "\nper-simdgroup stride between iterations: {} B; threadgroup window per \
         iteration: {per_iter_bytes} B contiguous per column",
        per_iter_bytes
    );
    // And it is the same at gate_up's tiles_k, i.e. the stride is shape-independent.
    assert_eq!(
        column_byte_offset(1, tiles_k_down) - column_byte_offset(0, tiles_k_down),
        129024,
        "down_proj column stride = tiles_k(1344) * 96 B"
    );
    assert_eq!(
        column_byte_offset(1, tiles_k_gate) - column_byte_offset(0, tiles_k_gate),
        32256,
        "gate_up column stride = tiles_k(336) * 96 B"
    );
    println!(
        "column strides: down_proj {} B, gate_up {} B (tiles_k * {TILE_BYTES})",
        tiles_k_down * TILE_BYTES as u32,
        tiles_k_gate * TILE_BYTES as u32
    );
}

#[test]
fn there_is_no_cross_iteration_pipelining_and_no_shared_memory_staging() {
    let src = read_metal("linear_repacked.metal");
    for kernel in [
        "linear_gemv_ishizuki_k3_colmajor_2col_sk2",
        "linear_gemv_ishizuki_k3_fused2_2col",
        "linear_gemv_ishizuki_k3_fused2_sk2",
        "linear_gemv_ishizuki_k3_fused3_2col",
    ] {
        let body = kernel_body(&src, kernel);

        // No prefetch primitive of any kind.
        for banned in [
            "prefetch",
            "Prefetch",
            "async_read",
            "mem_flags::mem_device",
            "__builtin",
        ] {
            assert!(
                !body.contains(banned),
                "{kernel} contains `{banned}`; the loads-in-flight count in \
                 `loads_in_flight_is_eight_per_thread_in_every_live_k_loop` is \
                 derived from the source and would now be wrong"
            );
        }

        // No threadgroup tile buffer. `red[][]` is the accumulator reduction only.
        for line in body.lines() {
            if declares_threadgroup_memory(line) {
                let found = line.trim().to_string();
                assert!(
                    line.contains("float red"),
                    "{kernel}: unexpected threadgroup storage `{found}`. \
                     A tile buffer would change both the barrier count and the \
                     loads-in-flight arithmetic."
                );
            }
        }

        // simd_shuffle is used, but only on accumulators -- never on a loaded
        // word. That distinction is the whole content of the question: sharing
        // loaded words between lanes via the crossbar was measured at
        // 180.5 -> 146.5 GB/s (decode-19tok-repacked-250gb/
        // implementation_notes.md:64-67).
        let shuffle_lines: Vec<&str> = body
            .lines()
            .filter(|l| l.contains("simd_shuffle"))
            .collect();
        assert!(
            !shuffle_lines.is_empty(),
            "{kernel}: expected the accumulator butterfly"
        );
        let n_shuffles = shuffle_lines.len();
        for line in &shuffle_lines {
            assert!(
                line.contains("acc"),
                "{kernel}: simd_shuffle on something other than an accumulator: \
                 `{}`",
                line.trim()
            );
        }

        // Loads are issued at the top of the body and consumed in the same body:
        // no register is carried across the `tk += step` boundary. The structural
        // statement is "every word the body loads is loaded before the body
        // consumes anything", which `weight_loads_before_first_consume` measures
        // directly -- there is no need to name a particular decode call, because
        // the two live shapes spell the consume differently (`extract_v_mcg_32` in
        // the `_2col` family, `const ulong A0 = ...` in `_fused2_sk2`).
        let loop_body = unrolled_loop_body(&src, kernel);
        let head_loads = weight_loads_before_first_consume(loop_body);
        let total_loads = weight_loads_total(loop_body);
        assert_eq!(
            head_loads, total_loads,
            "{kernel}: {total_loads} weight loads in the body but only {head_loads} \
             before the first consume, so the loop is NOT issue-all-then-consume. \
             That is what software pipelining would look like, so either the \
             loads-in-flight table or this claim is now wrong."
        );
        assert!(
            last_acc_after_loads(loop_body),
            "{kernel}: an accumulation precedes the last load"
        );
        println!(
            "{kernel:<48} no prefetch, {n_shuffles} shuffles (accumulators only), \
             threadgroup storage = red only"
        );
    }
    println!(
        "\nLoads in flight = 8 (burst), 0 (ALU half). Steady-state MLP is provided \
         entirely by warp-level parallelism across threadgroups, not by \
         software pipelining inside the loop."
    );
}

#[test]
fn little_law_says_the_current_mlp_is_ample() {
    // Distinct bytes in flight per simdgroup:
    //   4 tiles x TILE_BYTES. The 8 in-flight loads per lane cover 24 distinct
    //   words each, but lanes overlap, so the simdgroup's 8 loads touch exactly
    //   4 tiles = 4 * 96 = 384 B.
    let bytes_in_flight_per_simdgroup = 4 * TILE_BYTES;
    // Resident simdgroups per core: threadgroup size was swept 128/256/512 and is
    // flat within -5.1%..+1.9% (decode-19tok-repacked-250gb/
    // implementation_notes.md:1548-1552), so take the conservative 3 threadgroups
    // per core (the minimum consistent with "100% occupancy" at 256 threads on a
    // core that retires >=24 warps) and also show 6.
    println!();
    for tgs_per_core in [3usize, 6] {
        let simdgroups_resident = CORES * tgs_per_core * SG;
        let bytes_in_flight =
            simdgroups_resident as f64 * bytes_in_flight_per_simdgroup as f64;
        // Little's law: sustained B/s = bytes_in_flight / latency. Invert it:
        // the latency this much in-flight data can hide while still moving
        // WIRE_GBPS is bytes_in_flight / WIRE_GBPS nanoseconds.
        let ns_hidden = bytes_in_flight / WIRE_GBPS;
        println!(
            "{tgs_per_core:>2} tg/core: {simdgroups_resident:>4} simdgroups resident, \
             {:>7.0} KB in flight -> covers {ns_hidden:.0} ns of latency while \
             sustaining {WIRE_GBPS:.0} GB/s",
            bytes_in_flight / 1024.0,
        );
        assert!(
            ns_hidden > 100.0,
            "at {tgs_per_core} tg/core the in-flight bytes only cover {ns_hidden:.0} \
             ns, which is inside DRAM latency: the kernel WOULD be latency bound"
        );
    }
    println!(
        "Even the conservative 3 tg/core covers {:.0} ns of latency against the \
         wire rate. Apple DRAM latency is ~100 ns. So memory-level parallelism is \
         NOT the constraint.",
        CORES as f64 * 3.0 * SG as f64 * bytes_in_flight_per_simdgroup as f64 / WIRE_GBPS
    );
    println!(
        "Doubling loads-in-flight (cross-iteration prefetch, 8 -> 16) would move \
         that number to ~{:.0} ns. Little's law does not distinguish 100 ns from \
         200 ns; both are covered.",
        CORES as f64 * 3.0 * SG as f64 * 2.0 * bytes_in_flight_per_simdgroup as f64
            / WIRE_GBPS
    );
    println!(
        "\nThe resource that WOULD be consumed by a deeper prefetch is the \
         register file. T4.3 got this kernel to <=30 registers to hold 100% \
         occupancy (decode-25tok-peak-bandwidth/implementation_notes.md:145), \
         and T4.2's 4col lost 4.2% purely by doubling accumulators to 8 floats \
         (same file :122-128). On this kernel occupancy is the binding \
         constraint, so the direction that spends registers is the expensive one."
    );
}

// ---------------------------------------------------------------------------
// Sabotage discipline: every detector above is a pure function over a source
// string, so each one can be shown able to fail without editing the kernel.
// ---------------------------------------------------------------------------

#[test]
fn detectors_can_fail() {
    let src = read_metal("linear_repacked.metal");
    let name = "linear_gemv_ishizuki_k3_colmajor_2col_sk2";
    let real = unrolled_loop_body(&src, name);

    // 1. load counter: delete ONE weight load -> 8 becomes 7.
    let target = "const uint w1_1_0 = t1_1[j0], w1_1_1 = t1_1[j1];";
    assert!(real.contains(target), "sabotage 1 target not in the source");
    let sabotaged = real.replacen(target, "const uint w1_1_0 = t1_1[j0], w1_1_1 = 0u;", 1);
    assert_ne!(sabotaged, real, "sabotage 1 did not apply");
    assert_eq!(
        weight_loads_before_first_consume(&sabotaged),
        7,
        "the load counter did not notice a removed load"
    );

    // 2. load counter: halve the burst by consuming earlier -> still 8 (the
    //    detector counts the head, so this proves the counter tracks the burst
    //    boundary rather than total subscripts in the body).
    let earlier = real.replacen("float v[8];", "float v[8]; extract_v_mcg_32(0u,0u,sh0,rsh0,v);", 1);
    assert!(
        weight_loads_before_first_consume(&earlier) < 8,
        "the counter must be sensitive to where the first consume sits"
    );

    // 3. index arithmetic: a KA typo in the transcription must change the
    //    footprint. KA=4 is the real 4 bpw body; its footprint is 32 of 32.
    assert_eq!(
        (0..32u32)
            .map(|l| {
                let h = ((l * 8 + 1) * 4 + 256 * 4 - 16) / 32;
                (if h as usize >= 32 { h - 32 } else { h }) as usize
            })
            .chain(
                (0..32u32).map(|l| {
                    let j = {
                        let h = ((l * 8 + 1) * 4 + 256 * 4 - 16) / 32;
                        if h as usize >= 32 { h - 32 } else { h }
                    };
                    (if j + 1 >= 32 { j + 1 - 32 } else { j + 1 }) as usize
                }),
            )
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        32,
        "at KA=4 the same arithmetic must cover 32 of 32 words; if this fails the \
         arithmetic being checked is not the arithmetic the kernel runs"
    );

    // 4. staging detector: a threadgroup tile buffer must be rejected.
    let with_buffer = kernel_body(&src, name).replacen(
        "threadgroup float red[2][SG][16];",
        "threadgroup float red[2][SG][16];\n    threadgroup uint tg_tiles[24];",
        1,
    );
    let rejected = with_buffer
        .lines()
        .filter(|l| declares_threadgroup_memory(l))
        .any(|l| !l.contains("float red"));
    assert!(rejected, "the staging detector accepted a tile buffer");

    // 5. shuffle detector: a shuffle of a loaded word must be rejected.
    let shuffled = real.replace(
        "const uint w0_0_0 = t0_0[j0], w0_0_1 = t0_0[j1];",
        "const uint w0_0_0 = t0_0[j0], w0_0_1 = simd_shuffle(t0_0[j1], j1);",
    );
    let rejected = shuffled
        .lines()
        .filter(|l| l.contains("simd_shuffle"))
        .any(|l| !l.contains("acc"));
    assert!(rejected, "the shuffle detector accepted word sharing");

    println!("all five detectors reject their sabotaged copy");
}

// ---------------------------------------------------------------------------
// The pinned literals. If `linear_repacked.metal` is edited, these fail and the
// numbers in the header table have to be re-derived.
// ---------------------------------------------------------------------------

#[test]
fn the_constants_this_probe_is_built_on_are_still_in_the_kernel() {
    let src = read_metal("linear_repacked.metal");
    let body = kernel_body(&src, "linear_gemv_ishizuki_k3_colmajor_2col_sk2");
    for literal in [
        "constexpr uint KA = 3u;",
        "constexpr uint RING = 256u * KA;",
        "constexpr uint WORDS = RING / 32u;",
        "constexpr int SG = 8;",
        "const uint lo0 = (lane * 8u + 1u) * 3u + RING - 16u;",
        "const uint sh0 = lo0 % 32u;",
        "const uint hi0 = lo0 / 32u;",
        "const uint j0 = hi0 >= WORDS ? hi0 - WORDS : hi0;",
        "const uint j1 = j0 + 1u >= WORDS ? j0 + 1u - WORDS : j0 + 1u;",
    ] {
        assert!(
            body.contains(literal),
            "linear_repacked.metal no longer contains `{literal}`. The arithmetic \
             in this probe is transcribed from it and must be re-derived."
        );
    }
    // The dispatch really is 256 threads and 8 simdgroups.
    let rs_path = workspace_path("crates/kernels/src/linear.rs");
    let rs = std::fs::read_to_string(&rs_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", rs_path.display()));
    assert!(
        rs.contains("groups * 256,\n                256,"),
        "encode_gemv_* no longer dispatches 256-thread threadgroups; the \
         loads-in-flight and occupancy arithmetic assume 8 simdgroups/TG"
    );
    println!("all pinned literals present");
}
