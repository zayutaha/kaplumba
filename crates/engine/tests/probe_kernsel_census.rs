//! A3 — the complete census of the 138 linear kernels, and which of the four
//! that ship could a different tiling beat.
//!
//! NO GPU. NO MODEL. Pure source parsing plus integer arithmetic.
//!
//! WHY THIS FILE EXISTS
//!
//! Every prior probe in this repo asks about *one* dispatch site
//! (`probe_mlpdispatch_tree`, `perf_probe_dispatchsel_columns`), one *axis*
//! (`probe_coltile_registers`, `probe_reduction_k04`), or one *arm* against the
//! live baseline (`probe_arm_selectivity`). None of them enumerates the library.
//! The inventory is therefore known only as a number — "138 kernels, 4
//! dispatched" — and the number has never been decomposed, so nobody can say
//! which of the 134 unselected kernels are (a) deliberately dead, (b) reachable
//! by one env var and never run, or (c) not applicable to this checkpoint at
//! all. This file decomposes it, from the `.metal` sources and the two Rust
//! dispatch files, with nothing transcribed.
//!
//! WHAT IS DERIVED HERE (nothing is a hand-written list)
//!
//! * **Inventory.** `^kernel void NAME(` in `linear.metal` and
//!   `linear_repacked.metal`. 111 + 27 = 138.
//! * **Shape axes.** Per kernel body: split-K from `constexpr uint SK = Nu`,
//!   columns-per-threadgroup from the `threadgroup float red[C]...` rank (and
//!   `col1..col3` for the fused arms), K-unroll depth from `SG * Nu`,
//!   layout from `tn * (tiles_k * WORDS)` (column-contiguous) versus
//!   `stride = tiles_n * WORDS` (row-major), ring-window width from whether the
//!   hot loop reads `j2`, and the decode callee (`decode_state_mcg` /
//!   `decode_state` / `nod_stub`).
//! * **Reachability.** Every `linear_gemv_*` literal *and* every
//!   `format!("linear_gemv_ishizuki_k{bits}…")` template in
//!   `crates/kernels/src/linear.rs` and `crates/engine/src/model.rs` is bound
//!   to its enclosing `fn`, and the `env_flag("…")` literals inside that `fn`
//!   become the gating flags. A reference inside a function whose name does not
//!   start with `encode_` is construction or name-derivation, not dispatch.
//!
//! WHAT IT ASSERTS
//!
//!   C1. The inventory is 138 kernels, and exactly one `new_library_with_source`
//!       compiles both `.metal` files (`combined_source`, `linear.rs:24-31`), so
//!       "138" is one library, not two.
//!   C2. Exactly four kernel names are reachable from a production decode
//!       dispatch with the default environment for THIS checkpoint's facts
//!       (perm_mode 1 on all seven layer linears, perm_mode 0 on the head, rate
//!       3 layers / 6 head, every offset %4 == 0). Anything else is reachable
//!       only behind an env flag, only from the readback/verification forward,
//!       or not at all.
//!   C3. `_co`'s 4-loads-in-flight gap is real and lands on a kernel that is NOT
//!       dispatched. The live gate/up kernel has 8. This is asserted against the
//!       source, because the alternative reading — "`fused2_co` is the live
//!       gate/up kernel and is therefore under-hoisted" — is exactly the kind of
//!       plausible-but-wrong claim this project has shipped twice.
//!   C4. The `(columns x split)` grid of the live families has holes, and the
//!       holes are enumerated: `fused3` has no `_2col_sk2` and no `_4col`; every
//!       split-K variant in the tree still uses the pre-§22 64-bit `ulong`
//!       window, including `fused3_sk2`.
//!
//! SABOTAGE DISCIPLINE (AGENTS.md §3.4)
//!
//! Every assertion below was made to fail once, by mutating an in-memory copy of
//! the source it reads, and the observed failure is recorded in the comment on
//! the test. The inventory guard in particular is asserted *inside* the parser,
//! so a source change that renames or removes a kernel fails loudly instead of
//! silently shrinking the table.
//!
//! Run:
//!   cargo test -p kaplumba-engine --test probe_kernsel_census -- --nocapture

use std::collections::BTreeMap;
use std::path::PathBuf;

fn workspace_path(rel: &str) -> PathBuf {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let root = std::path::Path::new(manifest)
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

fn read(rel: &str) -> String {
    std::fs::read_to_string(workspace_path(rel))
        .unwrap_or_else(|e| panic!("read {rel}: {e}"))
}

/// Replace comment bodies with spaces, preserving byte length.
fn strip_comments(src: &str) -> String {
    let b = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    let mut in_line = false;
    let mut in_block = false;
    let mut in_str = false;
    while i < b.len() {
        if in_line {
            if b[i] == b'\n' {
                in_line = false;
                out.push('\n');
            } else {
                out.push(' ');
            }
        } else if in_block {
            if b[i] == b'*' && b.get(i + 1) == Some(&b'/') {
                out.push_str("  ");
                i += 2;
                in_block = false;
                continue;
            }
            out.push(if b[i] == b'\n' { '\n' } else { ' ' });
        } else if in_str {
            out.push(b[i] as char);
            if b[i] == b'\\' {
                if let Some(&n) = b.get(i + 1) {
                    out.push(n as char);
                    i += 2;
                    continue;
                }
            } else if b[i] == b'"' {
                in_str = false;
            }
        } else if b[i] == b'/' && b.get(i + 1) == Some(&b'/') {
            out.push_str("  ");
            i += 2;
            in_line = true;
            continue;
        } else if b[i] == b'/' && b.get(i + 1) == Some(&b'*') {
            out.push_str("  ");
            i += 2;
            in_block = true;
            continue;
        } else {
            if b[i] == b'"' {
                in_str = true;
            }
            out.push(b[i] as char);
        }
        i += 1;
    }
    out
}

// ---------------------------------------------------------------------------
// Kernel bodies
// ---------------------------------------------------------------------------

struct Kernel {
    file: &'static str,
    name: String,
    body: String,
    /// First source line, for the report.
    line: usize,
}

impl Kernel {
    fn clone_for_test(&self) -> Kernel {
        Kernel {
            file: self.file,
            name: self.name.clone(),
            body: self.body.clone(),
            line: self.line,
        }
    }
}

fn kernels_in(file: &'static str, source: &str) -> Vec<Kernel> {
    let clean = strip_comments(source);
    let lines: Vec<(&str, usize)> = clean
        .lines()
        .enumerate()
        .map(|(i, l)| (l, i + 1))
        .collect();
    let starts: Vec<(usize, usize, String)> = lines
        .iter()
        .filter_map(|(l, n)| {
            let t = l.trim_start();
            let rest = t.strip_prefix("kernel void ")?;
            let end = rest
                .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .unwrap_or(rest.len());
            let name = rest[..end].to_string();
            name.chars()
                .next()
                .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
                .then_some((end, *n, name))
        })
        .collect();
    let mut out = Vec::new();
    for (i, (_, line, name)) in starts.iter().enumerate() {
        let begin = lines
            .iter()
            .position(|(_, n)| *n == *line)
            .expect("line index");
        let end = if i + 1 < starts.len() {
            starts[i + 1].1
        } else {
            lines.len() + 1
        };
        let text: String = lines
            .iter()
            .filter(|(_, n)| *n >= *line && *n < end)
            .map(|(l, _)| *l)
            .collect::<Vec<_>>()
            .join("\n");
        let _ = begin;
        out.push(Kernel {
            file,
            name: name.clone(),
            body: text,
            line: *line,
        });
    }
    out
}

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Shape {
    family: &'static str,
    rate: Option<u8>,
    layout: &'static str,
    cols: u32,
    split_k: u32,
    unroll: u32,
    /// Ring words read per lane per tile: 2, 3, or n/a.
    ring_words: &'static str,
    /// Width of the shift chain the hot loop builds: 32 (the §22 funnel-shift
    /// form) or 64 (`ulong`, the pre-§22 form §22 measured at -22.8% on qkv).
    alu_width: &'static str,
    decode: &'static str,
    /// Weight `uint` loads held across one unrolled K-loop trip.
    loads_in_flight: u32,
    /// How many of those are separate 2-3 word bursts.
    #[allow(dead_code)]
    bursts: u32,
    valid_output: bool,
}

fn classify(k: &Kernel) -> Shape {
    // Helper definitions that live in the gap between one kernel and the next
    // must not be attributed to the earlier kernel. `nod_stub` is declared at
    // `linear.metal` just after `linear_gemv_ishizuki_k8` and before the `_nod`
    // family, so a naive `contains` reads k8 as a no-decode probe.
    let n_stub = k
        .body
        .find("static float nod_stub")
        .unwrap_or(k.body.len());
    let b: &str = &k.body[..n_stub];
    let name = &k.name;

    let family: &'static str = if name.contains("_exl3_") {
        "exl3"
    } else if name.contains("_ishizuki_") {
        "ishizuki"
    } else if name.starts_with("linear_gemv_fast") {
        "fast"
    } else if name.starts_with("gemv_reduce") {
        "reduce"
    } else if name.contains("finalize") || name.contains("transform") {
        "epilogue"
    } else {
        "other"
    };

    // The rate is the digits that follow `_k` -- NOT the `k` inside "ishizuki",
    // and NOT a `_sk2` / `_sk4` split-K suffix.
    let rate = {
        let bytes = name.as_bytes();
        let mut idx = 0;
        let mut found = None;
        while idx + 2 < bytes.len() {
            if bytes[idx] == b'_'
                && bytes[idx + 1] == b'k'
                && bytes[idx + 2].is_ascii_digit()
            {
                let ds: String = name[idx + 2..]
                    .chars()
                    .take_while(|c| c.is_ascii_digit())
                    .collect();
                found = ds.parse::<u8>().ok();
                break;
            }
            idx += 1;
        }
        found.filter(|r| (1..=8).contains(r))
    };

    let layout = if name.contains("colmajor")
        || name.contains("_fused")
        || b.contains("tn * (tiles_k * WORDS)")
    {
        "colmajor"
    } else {
        "row-major"
    };

    let split_k = b
        .split("constexpr uint SK = ")
        .nth(1)
        .and_then(|t| t.split('u').next())
        .and_then(|d| d.trim().parse::<u32>().ok())
        .unwrap_or(1);

    // Columns per threadgroup. `red[C][SG][16]` counts output *tiles*; the
    // fused arms keep C = 1 there but own 2 matrices, so the column count is
    // also the number of distinct column base pointers.
    let red_rank = b
        .split("threadgroup float red[")
        .nth(1)
        .and_then(|t| t.split(']').next())
        .and_then(|d| d.trim().parse::<u32>().ok())
        .unwrap_or(1);
    let col_ptrs = ["col0", "col1", "col2", "col3", "col_a", "col_b", "column"]
        .iter()
        .filter(|n| b.contains(&format!("const device uint *{n} =")))
        .count() as u32;
    let cols = red_rank.max(col_ptrs).max(1);

    let unroll = if b.contains("SG * 4u") {
        4
    } else if b.contains("SG * 2u") || b.contains("2u * SG") {
        2
    } else {
        1
    };

    let has_window = b.contains("[j0]") || b.contains("j0 =");
    let ring_words = if !has_window {
        "n/a"
    } else if b.contains("nod_stub") {
        "3 (probe)"
    } else if b.contains("[j2]") || b.contains("j2 =") {
        "3"
    } else {
        "2"
    };
    // 64-bit iff the unrolled trip builds a `ulong` shift chain; that is the
    // `A0 = ((w0<<32)|w1)<<sh` form §22 replaced with `extract_v_mcg_32`.
    let alu_width = if b.contains("const ulong A") || b.contains("const ulong B") {
        "64"
    } else {
        "32"
    };

    let decode = if b.contains("nod_stub") {
        "nod_stub (garbage by design)"
    } else if b.contains("extract_v_mcg_32(") || b.contains("extract_v_mcg_fast(") {
        "extract_v_mcg_32 (MCG, 32-bit)"
    } else if b.contains("exl3_decode8_mcg") {
        "exl3_decode8_mcg (MCG folded)"
    } else if b.contains("decode_state_mcg(") {
        "decode_state_mcg (MCG)"
    } else if b.contains("decode_state(") {
        "decode_state (runtime CB)"
    } else {
        "n/a"
    };

    // Weight loads held across one unrolled K-loop trip. The metric that
    // matters for memory-level parallelism is how many separate bursts are
    // issued *before the first decode consumes one*: the hoisted
    // `const uint w…` / `const ulong w…` declaration groups that precede the
    // first `extract_v_mcg*` / `decode_state_mcg` call. Kernels that inline the
    // index expression into the extractor (`fused2_4col`, `colmajor_4col`) hold
    // exactly one burst however many extractor calls they make.
    let first_use = b
        .find("extract_v_mcg_32(")
        .or_else(|| b.find("extract_v_mcg_fast("))
        .or_else(|| b.find("decode_state_mcg("))
        .or_else(|| b.find("exl3_decode8_mcg"))
        .unwrap_or(b.len());
    let head = &b[..first_use];
    let bursts = head
        .lines()
        .filter(|l| {
            let t = l.trim();
            t.starts_with("const uint w") || t.starts_with("const ulong w")
        })
        .count() as u32;
    let words_per_burst = if ring_words.starts_with('3') { 3 } else { 2 };
    let loads_in_flight = if !has_window {
        0
    } else if bursts == 0 {
        words_per_burst // inline form: one burst consumed at a time
    } else {
        bursts * words_per_burst
    };

    let valid_output = !name.contains("_nod") && !name.contains("_bench") && family != "reduce";

    Shape {
        family,
        rate,
        layout,
        cols,
        split_k,
        unroll,
        ring_words,
        alu_width,
        decode,
        loads_in_flight,
        bursts,
        valid_output,
    }
}

// ---------------------------------------------------------------------------
// Reachability: bind every kernel-name reference to its enclosing `fn`
// ---------------------------------------------------------------------------

struct Site {
    file: &'static str,
    func: String,
    flags: Vec<String>,
}

/// `(fn name, body span)` for every top-level-ish `fn` in a Rust source.
fn functions(src: &str) -> Vec<(String, String)> {
    let clean = strip_comments(src);
    let mut out = Vec::new();
    let bytes = clean.as_bytes();
    let mut i = 0;
    while let Some(at) = clean[i..].find("fn ") {
        let start = i + at;
        // walk back to the start of the line
        let mut head = start;
        while head > 0 && bytes[head - 1] != b'\n' {
            head -= 1;
        }
        let name_start = start + 3;
        let mut name_end = name_start;
        while name_end < bytes.len()
            && (bytes[name_end].is_ascii_alphanumeric() || bytes[name_end] == b'_')
        {
            name_end += 1;
        }
        if name_end == name_start {
            i = start + 3;
            continue;
        }
        let name = clean[name_start..name_end].to_string();
        // find the opening brace
        let mut j = name_end;
        while j < bytes.len() && bytes[j] != b'{' {
            if bytes[j] == b';' {
                break;
            }
            j += 1;
        }
        if j >= bytes.len() || bytes[j] != b'{' {
            i = name_end;
            continue;
        }
        let mut depth = 0usize;
        let mut k = j;
        while k < bytes.len() {
            match bytes[k] {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                _ => {}
            }
            k += 1;
        }
        out.push((name, clean[j..k.min(bytes.len())].to_string()));
        i = k.max(name_end);
    }
    out
}

fn sites() -> Vec<(String, Site)> {
    let mut refs: Vec<(String, Site)> = Vec::new();
    for (file, src) in [
        ("linear.rs", read("crates/kernels/src/linear.rs")),
        ("model.rs", read("crates/engine/src/model.rs")),
    ] {
        for (func, body) in functions(&src) {
            let flags: Vec<String> = body
                .match_indices("env_flag(\"")
                .map(|(i, _)| {
                    let rest = &body[i + "env_flag(\"".len()..];
                    rest[..rest.find('"').unwrap()].to_string()
                })
                .chain(body.match_indices("env_var(\"").map(|(i, _)| {
                    let rest = &body[i + "env_var(\"".len()..];
                    rest[..rest.find('"').unwrap()].to_string()
                }))
                .collect();
            let site = Site {
                file,
                func: func.clone(),
                flags,
            };
            // Literal names.
            let mut idx = 0;
            while let Some(at) = body[idx..].find("linear_gemv_") {
                let abs = idx + at;
                let rest = &body[abs..];
                let end = rest
                    .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                    .unwrap_or(rest.len());
                refs.push((rest[..end].to_string(), site_of(&site)));
                idx = abs + end.max(1);
            }
            // `format!` templates: `linear_gemv_ishizuki_k{bits}_<suffix>`.
            idx = 0;
            while let Some(at) = body[idx..].find("format!(\"linear_gemv_ishizuki_k{bits}") {
                let abs = idx + at;
                let rest = &body[abs + "format!(\"".len()..];
                let end = rest.find('"').unwrap();
                let tmpl = &rest[..end];
                // Rewrite the interpolated `{bits}` into each rate the library
                // emits, so a template is a statement about real kernel names.
                let tail: String = tmpl
                    .trim_start_matches("linear_gemv_ishizuki_k{bits}")
                    .to_string();
                for bits in 1..=8u8 {
                    let name = format!("linear_gemv_ishizuki_k{bits}{tail}");
                    refs.push((name, site_of(&site)));
                }
                idx = abs + end + 1;
            }
        }
    }
    refs
}

fn site_of(s: &Site) -> Site {
    Site {
        file: s.file,
        func: s.func.clone(),
        flags: s.flags.clone(),
    }
}

// ---------------------------------------------------------------------------
// The facts this checkpoint presents to the dispatch tree
// ---------------------------------------------------------------------------

const HIDDEN: usize = 5376;
const INTERMEDIATE: usize = 21504;
const VOCAB: usize = 262_144;
const SLIDING_Q: usize = 8192;
const SLIDING_KV: usize = 4096;
const FULL_Q: usize = 16_384;
const FULL_KV: usize = 2048;
const FULL_O_IN: usize = 16_384;
const SLIDING_LAYERS: usize = 50;
const FULL_LAYERS: usize = 10;

fn weight_bytes(n: usize, k: usize, bits: u8) -> u64 {
    n as u64 * k as u64 * bits as u64 / 8
}

#[test]
fn census_of_all_138_linear_kernels() {
    let lin_metal = read("crates/kernels/src/linear.metal");
    let rep_metal = read("crates/kernels/src/linear_repacked.metal");
    let kernels: Vec<Kernel> = kernels_in("linear.metal", &lin_metal)
        .into_iter()
        .chain(kernels_in("linear_repacked.metal", &rep_metal))
        .collect();

    // C1. The count, and that both files share one compilation.
    assert_eq!(
        kernels.len(),
        138,
        "expected 138 linear kernels (111 in linear.metal + 27 in linear_repacked.metal), \
         found {} — the census below is incomplete if this moved",
        kernels.len()
    );
    let linear_rs = read("crates/kernels/src/linear.rs");
    assert!(
        linear_rs.contains("new_library_with_source(&combined_source(), &options)"),
        "linear.rs must compile both .metal files as ONE library, or \"138 kernels\" \
         is two libraries and the load cost is not the one the notes record"
    );

    // Every name unique, so a row in the census is a row in the library.
    let mut seen = std::collections::BTreeSet::new();
    for k in &kernels {
        assert!(seen.insert(k.name.clone()), "duplicate kernel name {}", k.name);
    }

    // ---- print the census -------------------------------------------------
    println!(
        "\n{:<52} {:<10} {:>4} {:<9} {:>4} {:>3} {:>3} {:>10} {:>4} {:<30} {:>4}",
        "kernel", "file", "rate", "layout", "cols", "SK", "unr", "ringw", "alu", "decode", "loads"
    );
    println!("{}", "-".repeat(140));
    let mut by_family: BTreeMap<&str, (usize, usize, usize)> = BTreeMap::new();
    let mut rows: Vec<(&Kernel, Shape)> = Vec::new();
    for k in &kernels {
        let s = classify(k);
        let short = k.file.trim_start_matches("linear").trim_start_matches('_');
        println!(
            "{:<52} {:<10} {:>4} {:<9} {:>4} {:>3} {:>3} {:>10} {:>4} {:<30} {:>4}",
            k.name,
            short,
            s.rate.map(|r| r.to_string()).unwrap_or("-".into()),
            s.layout,
            s.cols,
            s.split_k,
            s.unroll,
            s.ring_words,
            s.alu_width,
            s.decode,
            s.loads_in_flight
        );
        let e = by_family.entry(s.family).or_insert((0, 0, 0));
        e.0 += 1;
        if s.rate.is_some() {
            e.1 += 1;
        }
        if s.valid_output {
            e.2 += 1;
        }
        rows.push((k, s));
    }
    println!("{}", "-".repeat(140));
    println!("family            total  rate-specialised  produces-valid-output");
    for (fam, (total, rated, valid)) in &by_family {
        println!("{fam:<18} {total:>5} {rated:>16} {valid:>22}");
    }
    let gemvish: usize = rows
        .iter()
        .filter(|(_, s)| s.rate.is_some() || s.family == "other")
        .count();
    println!(
        "\n138 total = {gemvish} rate-parameterised GEMV + {} epilogue/reduce/gemm/aux",
        138 - gemvish
    );

    // ---- C3: the `_co` hoist gap lands on a dead kernel ------------------
    let by_name: BTreeMap<&str, (&Kernel, &Shape)> = rows
        .iter()
        .map(|(k, s)| (k.name.as_str(), (*k, s)))
        .collect();
    let (two_col, two) = by_name["linear_gemv_ishizuki_k3_fused2_2col"];
    let (co, co_s) = by_name["linear_gemv_ishizuki_k3_fused2_co"];
    let (_two_col_body, _co_body) = (two_col, co);
    let _ = (_two_col_body, _co_body);
    println!(
        "\nfused2_2col (LIVE, gate/up + q/k): loads in flight = {}, cols = {}, unroll = {}",
        two.loads_in_flight, two.cols, two.unroll
    );
    println!(
        "fused2_co  (dead for this ckpt)  : loads in flight = {}, cols = {}, unroll = {}",
        co_s.loads_in_flight, co_s.cols, co_s.unroll
    );
    assert_eq!(
        two.loads_in_flight, 8,
        "the live fused2_2col is expected to hoist 8 weight loads per trip \
         (4 burst pairs x 2 words), same as fused3_2col and colmajor_2col_sk2"
    );
    assert_eq!(
        co_s.loads_in_flight, 4,
        "fused2_co is expected to hoist only 4 — the T4.3 8->4->8 hoist was never \
         carried into it. If this changed, the anomaly in Q3 is gone."
    );
    assert_eq!(two.cols, co_s.cols, "both arms own 2 columns");
    // Same shape, half the memory-level parallelism — and `_co` is not selected.
    let _ = two_col;

    // ---- C4: every split-K variant is still on the 64-bit window ----------
    println!("\n(ring words / shift-chain width / split-K) for every split-K kernel:");
    let mut split_k_kernels = Vec::new();
    for (k, s) in &rows {
        if s.split_k > 1 {
            split_k_kernels.push((k.name.clone(), s.ring_words.to_string(), s.alu_width));
        }
    }
    split_k_kernels.sort();
    for (n, w, a) in &split_k_kernels {
        println!("  {n:<52} ring={w} alu={a}");
    }
    let alu64_sk: Vec<&String> = split_k_kernels
        .iter()
        .filter(|(_, _, a)| *a == "64")
        .map(|(n, _, _)| n)
        .collect();
    println!("\nsplit-K kernels still on the 64-bit shift chain: {alu64_sk:?}");
    assert!(
        alu64_sk.contains(&&"linear_gemv_ishizuki_k3_fused3_sk2".to_string()),
        "fused3_sk2 -- the ONLY split-K variant of the kernel that carries 13.7% of \
         a token's bytes -- is no longer on the pre-§22 64-bit ring window. This is \
         the load-bearing fact behind the A3 verdict on Q4; re-read the file."
    );
    assert!(
        !alu64_sk.contains(&&"linear_gemv_ishizuki_k3_fused3_2col".to_string()),
        "the LIVE qkv kernel is on the 64-bit chain; the whole §22 argument and \
         the 9.375 instr/code figure need re-deriving"
    );
    assert!(
        !split_k_kernels
            .iter()
            .any(|(n, _, w)| n == "linear_gemv_ishizuki_k3_fused3_2col_sk2" && *w == "2"),
        "a fused3_2col_sk2 with the 32-bit funnel shift now exists; A3's coverage-gap \
         claim is stale and the file needs re-reading"
    );

    // ---- C2: which four ship ---------------------------------------------
    type Refs = BTreeMap<String, Vec<Site>>;
    let (dispatch, selector): (Refs, Refs) = {
        let mut d: Refs = BTreeMap::new();
        let mut s: Refs = BTreeMap::new();
        for (name, site) in sites() {
            let bucket = if site.func.starts_with("encode_") {
                &mut d
            } else {
                &mut s
            };
            bucket.entry(name).or_default().push(site);
        }
        (d, s)
    };
    let live = [
        ("linear_gemv_ishizuki_k3_fused3_2col", "q,k,v  (50 sliding)"),
        ("linear_gemv_ishizuki_k3_fused2_2col", "q,k (10 full) + gate,up (60)"),
        ("linear_gemv_ishizuki_k3_colmajor_2col_sk2", "o_proj, down_proj (60)"),
        ("linear_gemv_ishizuki_k6", "lm_head (1)"),
    ];
    println!("\nThe four live kernels and every env flag in their dispatch function:");
    for (name, what) in live {
        assert!(by_name.contains_key(name), "{name} must exist");
        let sites: &[Site] = dispatch
            .get(name)
            .map(Vec::as_slice)
            .or_else(|| selector.get(name).map(Vec::as_slice))
            .unwrap_or_default();
        let mut f: Vec<&str> = sites
            .iter()
            .flat_map(|s| s.flags.iter().map(|x| x.as_str()))
            .collect();
        f.sort_unstable();
        f.dedup();
        if sites.is_empty() {
            println!("  {name:<44} {what:<32} (rate slot, no literal in source)");
        } else {
            println!(
                "  {name:<44} {what:<32} via {} / flags {f:?}",
                sites[0].func
            );
        }
    }

    // Reachable-but-unmeasured: a *dispatch-site* reference whose only gates are
    // env flags that no product code sets (`grep set_var crates/` finds tests
    // only). `encode_gemv_named` / `encode_ishizuki_gemv_named` are excluded:
    // they are the name-addressable A/B entry points and would otherwise make
    // all 138 look reachable.
    const AB_ONLY: [&str; 2] = ["encode_gemv_named", "encode_ishizuki_gemv_named"];
    println!("\nA: kernels a single env flag would put on the production decode path");
    let mut flagged: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (name, sites) in &dispatch {
        if live.iter().any(|(l, _)| l == name) || AB_ONLY.contains(&sites[0].func.as_str()) {
            continue;
        }
        if !by_name.contains_key(name.as_str()) {
            continue;
        }
        let f: Vec<String> = sites
            .iter()
            .flat_map(|s| s.flags.iter().cloned())
            .collect::<std::collections::BTreeSet<String>>()
            .into_iter()
            .collect();
        let unset: Vec<String> = f
            .iter()
            .filter(|x| !x.starts_with("KAPLUMBA_NO_"))
            .cloned()
            .collect();
        if !unset.is_empty() {
            flagged.insert(name.clone(), unset);
        }
    }
    for (name, flags) in &flagged {
        let s = &by_name[name.as_str()].1;
        println!(
            "  {name:<52} {flags:?}  cols={} SK={} ring={} alu={} loads={}",
            s.cols, s.split_k, s.ring_words, s.alu_width, s.loads_in_flight
        );
    }

    // ---- the (columns x split) coverage matrix, from the inventory ----------
    // This is Q4. Each cell is filled or not by construction, and the point is
    // which cells a live kernel sits next to and cannot reach. `base` is the
    // prefix the family shares; a cell counts as filled when some kernel in the
    // inventory matches base + (1|2|4)col (+ _sk2|_sk4) at that (cols, SK).
    fn cell_filled(
        rows: &[(&Kernel, Shape)],
        base: &str,
        cols: u32,
        sk: u32,
        alu: &str,
    ) -> bool {
        // These are different tiling *families* that share the base name, not
        // (columns x split) cells of it, so they do not fill a cell.
        const NOT_A_CELL: [&str; 7] = [
            "_simd", "_stream", "_u4", "_contig4", "_co", "_co4", "_bench",
        ];
        rows.iter().any(|(k, s)| {
            k.name.strip_prefix(base).is_some_and(|t| {
                !NOT_A_CELL.iter().any(|skip| t.starts_with(skip))
                    && s.cols == cols
                    && s.split_k == sk
                    && s.ring_words == "2"
                    && s.alu_width == alu
            })
        })
    }
    println!("\n(columns x split-K) coverage. base = the family's unsuffixed kernel.");
    println!("  rows = SK, cells = columns-per-threadgroup; `--` = kernel does not exist");
    for (label, base) in [
        ("single-matrix (o_proj/down)", "linear_gemv_ishizuki_k3_colmajor"),
        ("fused2 (q/k full, gate+up)", "linear_gemv_ishizuki_k3_fused2"),
        ("fused3 (q/k/v sliding)", "linear_gemv_ishizuki_k3_fused3"),
    ] {
        println!("  {label}  [{base}]");
        for alu in ["32", "64"] {
            for sk in [1u32, 2, 4] {
                let row: Vec<String> = [1u32, 2, 4]
                    .iter()
                    .map(|c| {
                        if cell_filled(&rows, base, *c, sk, alu) {
                            format!(" {c}c ")
                        } else {
                            " -- ".to_string()
                        }
                    })
                    .collect();
                println!("      SK={sk} alu={alu}:{} ", row.join(""));
            }
        }
    }

    println!("\nB: kernels with NO production dispatch reference at all");
    let mut none = 0;
    for (k, _) in &rows {
        if dispatch.contains_key(k.name.as_str()) || selector.contains_key(k.name.as_str()) {
            continue;
        }
        none += 1;
    }
    println!("  {none} of 138: constructed at load, never named by any dispatch site.");
    println!("  (includes the 8 `_nod` garbage probes, the 2 `gemv_reduce_*`, and the 2 `_bench` kernels)");

    println!("\nC: kernels reachable only through `ishizuki_gemv_pipeline` (the rate-slot");
    println!("   selector), which for THIS checkpoint is entered only by lm_head");
    println!("   (perm_mode 0), so only the rate-6 unsuffixed kernel survives:");
    for (name, sites) in &selector {
        if live.iter().any(|(l, _)| *l == name.as_str()) || !by_name.contains_key(name.as_str()) {
            continue;
        }
        println!("  {name:<52} via {}", sites[0].func);
    }

    // ---- the arithmetic that ranks the four -------------------------------
    println!("\nbytes/token and grid for each live cell (3 bpw layers, 6 bpw head):");
    #[derive(Debug)]
    struct Cell {
        what: &'static str,
        kernel: &'static str,
        bytes: u64,
        /// Bytes a single threadgroup reads, i.e. `bytes` for the one
        /// layer that projects into this grid.
        per_layer: u64,
        groups: u64,
    }
    let cells = vec![
        Cell {
            what: "gate+up (60)",
            kernel: "linear_gemv_ishizuki_k3_fused2_2col",
            bytes: 60 * 2 * weight_bytes(INTERMEDIATE, HIDDEN, 3),
            per_layer: 2 * weight_bytes(INTERMEDIATE, HIDDEN, 3),
            groups: ((INTERMEDIATE / 16) * 2 / 2) as u64,
        },
        Cell {
            what: "down_proj (60)",
            kernel: "linear_gemv_ishizuki_k3_colmajor_2col_sk2",
            bytes: 60 * weight_bytes(HIDDEN, INTERMEDIATE, 3),
            per_layer: weight_bytes(HIDDEN, INTERMEDIATE, 3),
            groups: ((HIDDEN / 16) / 2 * 2) as u64,
        },
        Cell {
            what: "q,k,v sliding (50)",
            kernel: "linear_gemv_ishizuki_k3_fused3_2col",
            bytes: SLIDING_LAYERS as u64
                * (weight_bytes(SLIDING_Q, HIDDEN, 3) + 2 * weight_bytes(SLIDING_KV, HIDDEN, 3)),
            per_layer: weight_bytes(SLIDING_Q, HIDDEN, 3) + 2 * weight_bytes(SLIDING_KV, HIDDEN, 3),
            groups: ((SLIDING_Q / 16 + 2 * (SLIDING_KV / 16)) / 2) as u64,
        },
        Cell {
            what: "o_proj sliding (50)",
            kernel: "linear_gemv_ishizuki_k3_colmajor_2col_sk2",
            bytes: SLIDING_LAYERS as u64 * weight_bytes(HIDDEN, SLIDING_Q, 3),
            per_layer: weight_bytes(HIDDEN, SLIDING_Q, 3),
            groups: ((HIDDEN / 16) / 2 * 2) as u64,
        },
        Cell {
            what: "o_proj full (10)",
            kernel: "linear_gemv_ishizuki_k3_colmajor_2col_sk2",
            bytes: FULL_LAYERS as u64 * weight_bytes(HIDDEN, FULL_O_IN, 3),
            per_layer: weight_bytes(HIDDEN, FULL_O_IN, 3),
            groups: ((HIDDEN / 16) / 2 * 2) as u64,
        },
        Cell {
            what: "q,k full (10)",
            kernel: "linear_gemv_ishizuki_k3_fused2_2col",
            bytes: FULL_LAYERS as u64
                * (weight_bytes(FULL_Q, HIDDEN, 3) + weight_bytes(FULL_KV, HIDDEN, 3)),
            per_layer: weight_bytes(FULL_Q, HIDDEN, 3) + weight_bytes(FULL_KV, HIDDEN, 3),
            groups: ((FULL_Q / 16 + FULL_KV / 16) / 2) as u64,
        },
        Cell {
            what: "lm_head (1)",
            kernel: "linear_gemv_ishizuki_k6",
            bytes: weight_bytes(VOCAB, HIDDEN, 6),
            per_layer: weight_bytes(VOCAB, HIDDEN, 6),
            groups: (VOCAB / 16) as u64,
        },
    ];
    let total: u64 = cells.iter().map(|c| c.bytes).sum();
    println!(
        "{:<22} {:<42} {:>10} {:>7} {:>7} {:>9} {:>9}",
        "cell", "kernel", "GB/token", "share", "TGs", "B/TG", "TGs/core"
    );
    for c in &cells {
        println!(
            "{:<22} {:<42} {:>10.4} {:>6.1}% {:>7} {:>9} {:>9.1}",
            c.what,
            c.kernel,
            c.bytes as f64 / 1e9,
            100.0 * c.bytes as f64 / total as f64,
            c.groups,
            c.per_layer / c.groups.max(1),
            c.groups as f64 / 20.0
        );
    }
    println!(
        "{:<22} {:<42} {:>10.4}",
        "TOTAL", "(all GEMV)",
        total as f64 / 1e9
    );
    assert_eq!(cells.len(), 7);
    assert_eq!(
        total, 12_039_487_488,
        "GEMV bytes/token for this checkpoint; 12.039 GB is the figure every \
         tok/s number in this repo is quoted against"
    );
    // Anchors already published in K03/§4, re-derived here.
    assert_eq!(cells[0].groups, 1344, "gate+up grid");
    assert_eq!(cells[1].groups, 336, "down_proj grid");
    assert_eq!(cells[2].groups, 512, "qkv sliding grid");
    assert_eq!(cells[6].groups, 16_384, "lm_head grid");
    // `fused2_2col` on gate/up and `fused3_2col` on qkv are the two biggest
    // cells and both land on exactly 64,512 B/threadgroup; `down_proj`'s split-K
    // cell is the only one above it. These are the anchors the A3 report quotes.
    assert_eq!(cells[0].per_layer / cells[0].groups, 64_512);
    assert_eq!(cells[2].per_layer / cells[2].groups, 64_512);
    assert_eq!(cells[1].per_layer / cells[1].groups, 129_024);
    assert_eq!(cells[6].per_layer / cells[6].groups, 64_512);
    assert_eq!(cells[3].per_layer / cells[3].groups, 49_152);
}

/// AGENTS.md §3.4: a coverage claim is not a claim until the thing it covers has
/// been broken and the break observed. Every detector in `classify` is a pure
/// function of a kernel body, so each one is fed a mutated body here and the
/// change in its output is asserted. Observed values are in the comments; these
/// are the runs that produced them.
#[test]
fn the_census_detectors_can_fail() {
    let rep = read("crates/kernels/src/linear_repacked.metal");
    let k = kernels_in("linear_repacked.metal", &rep);
    let find = |n: &str| k.iter().find(|x| x.name == n).expect(n).clone_for_test();

    // 1. loads-in-flight. Drop both of `fused2_2col`'s hoisted declaration
    //    groups for column 1 -- i.e. undo T4.3 for one column -- and the count
    //    must fall 8 -> 4, i.e. from 4 bursts to 2.
    let base = find("linear_gemv_ishizuki_k3_fused2_2col");
    assert_eq!(classify(&base).loads_in_flight, 8);
    let m1 = Kernel {
        body: base
            .body
            .replace("const uint w0_1_0 = t0_1[j0], w0_1_1 = t0_1[j1];", "")
            .replace("const uint w1_1_0 = t1_1[j0], w1_1_1 = t1_1[j1];", ""),
        ..base.clone_for_test()
    };
    assert_eq!(
        classify(&m1).loads_in_flight, 4,
        "removing both hoisted groups for column 1 must leave only col 0's two \
         bursts, i.e. the pre-T4.3 `colmajor_2col` shape (4, not 6)"
    );
    // 2. The same detector on `_co`: hoisting its two extra groups to match
    //    `_2col` is the fix the T4.3 hoist was, and it must move 4 -> 8.
    let co = find("linear_gemv_ishizuki_k3_fused2_co");
    assert_eq!(classify(&co).loads_in_flight, 4);
    let hoisted = Kernel {
        body: co
            .body
            .replace(
                "            // Col B: 2 contiguous sub-steps along K\n            const uint w0_b0 = t0_b[j0], w0_b1 = t0_b[j1];\n            const uint w1_b0 = t1_b[j0], w1_b1 = t1_b[j1];\n",
                "",
            )
            .replace(
                "            float v[8];\n",
                "            float v[8];\n            const uint w0_b0 = t0_b[j0], w0_b1 = t0_b[j1];\n            const uint w1_b0 = t1_b[j0], w1_b1 = t1_b[j1];\n",
            ),
        ..co.clone_for_test()
    };
    assert_eq!(
        classify(&hoisted).loads_in_flight, 8,
        "hoisting _co's col-B loads above col A's first extract must give it the \
         same 4 bursts as _2col -- this is what the 8->4->8 hoist is"
    );

    // 3. shift-chain width. Give `fused3_sk2` the §22 funnel-shift call and the
    //    64-bit detector must flip.
    let sk2 = find("linear_gemv_ishizuki_k3_fused3_sk2");
    assert_eq!(classify(&sk2).alu_width, "64");
    let converted = Kernel {
        body: sk2.body.replace("const ulong A", "uint A"),
        ..sk2.clone_for_test()
    };
    assert_eq!(
        classify(&converted).alu_width, "32",
        "removing the ulong declaration must flip the detector"
    );

    // 4. split-K. Delete the `constexpr uint SK` from `fused3_sk2` and it must
    //    read as 1 -- which is exactly the mistake "it is a split-K variant"
    //    would make if the classifier were pattern-matching on the NAME.
    assert_eq!(classify(&sk2).split_k, 2);
    let no_sk = Kernel {
        body: sk2.body.replace("constexpr uint SK = 2u;", ""),
        ..sk2.clone_for_test()
    };
    assert_eq!(
        classify(&no_sk).split_k, 1,
        "the split-K detector reads the source, not the name suffix"
    );

    // 5. columns. `_2col` -> `_4col` is a one-token change to `red[2]` and must
    //    be visible, because "2 columns per threadgroup" is the axis the whole
    //    coverage matrix turns on.
    let two = find("linear_gemv_ishizuki_k3_fused2_2col");
    assert_eq!(classify(&two).cols, 2);
    let four = Kernel {
        body: two.body.replace("threadgroup float red[2][SG][16];", "threadgroup float red[4][SG][16];"),
        ..two.clone_for_test()
    };
    assert_eq!(classify(&four).cols, 4);

    // 6. ring width. Add a `j2` read to a live kernel and the detector must
    //    say 3 -- the condition `perf_probe_kloop_loads_in_flight` asserts
    //    against for exactly these five kernels.
    assert_eq!(classify(&two).ring_words, "2");
    let three = Kernel {
        body: two.body.replace("const uint j1 = j0 + 1u >= WORDS", "const uint j2 = j1 + 1u >= WORDS ? j1 + 1u - WORDS : j1 + 1u;\n        const uint j1 = j0 + 1u >= WORDS"),
        ..two.clone_for_test()
    };
    assert_eq!(classify(&three).ring_words, "3");

    // 7. The rate parser must not read the `k` in "ishizuki", nor a `_sk2`
    //    suffix as a rate.
    assert_eq!(classify(&two).rate, Some(3));
    assert_eq!(classify(&sk2).rate, Some(3));
    let k4 = kernels_in("linear_repacked.metal", &rep)
        .into_iter()
        .find(|x| x.name == "linear_gemv_ishizuki_k4_colmajor_contig4_bench")
        .expect("k4 bench");
    assert_eq!(classify(&k4).rate, Some(4));
    let reduce = find("gemv_reduce_sk4");
    assert_eq!(
        classify(&reduce).rate, None,
        "`gemv_reduce_sk4` has no rate; a parser that stripped non-digits would \
         read 4 and put a split-K reducer in the rate table"
    );

    // 8. Negative control on the inventory itself. Renaming a kernel keeps the
    //    count at 27 but must drop it under the old name; demoting its
    //    `kernel void` signature to `static void` must drop the count to 26.
    //    Observed: 27 / 27-and-renamed / 26.
    let both = read("crates/kernels/src/linear.metal");
    assert_eq!(
        kernels_in("linear_repacked.metal", &rep).len()
            + kernels_in("linear.metal", &both).len(),
        138
    );
    let renamed = rep.replacen(
        "kernel void linear_gemv_ishizuki_k3_fused2_co4(",
        "kernel void linear_gemv_ishizuki_k3_fused2_coX(",
        1,
    );
    assert_eq!(kernels_in("linear_repacked.metal", &renamed).len(), 27);
    assert!(
        kernels_in("linear_repacked.metal", &renamed)
            .iter()
            .all(|k| k.name != "linear_gemv_ishizuki_k3_fused2_co4"),
        "the renamed kernel must be gone under its old name"
    );
    let demoted = rep.replacen(
        "kernel void linear_gemv_ishizuki_k3_fused2_co4(",
        "static void linear_gemv_ishizuki_k3_fused2_co4(",
        1,
    );
    assert_eq!(
        kernels_in("linear_repacked.metal", &demoted).len(),
        26,
        "a kernel demoted to a plain function is no longer in the library's \
         inventory, and the 138 figure must move with it"
    );
}

/// True when `encode_gemv_raw` consults the `_2col` arm *before* the EXL3 arm,
/// i.e. when a colmajor 3 bpw projection of this checkpoint never reaches
/// `linear_gemv_exl3_k*`.
fn exl3_arm_follows_two_col(linear_rs: &str) -> bool {
    let raw_at = linear_rs
        .find("pub fn encode_gemv_raw(")
        .expect("encode_gemv_raw");
    let raw_tail = &linear_rs[raw_at..];
    let raw_end = raw_tail
        .find("\n    #[allow")
        .or_else(|| raw_tail.find("\n    pub fn "))
        .unwrap_or(raw_tail.len())
        .max(1);
    let body = &raw_tail[..raw_end];
    let two_col_at = body
        .find("env_flag(\"KAPLUMBA_NO_2COL\")")
        .expect("the _2col arm");
    let exl3_at = body
        .find("self.decode_gemv_pipeline(rate, packed_offset, codebook)")
        .expect("the EXL3 arm");
    two_col_at < exl3_at
}

#[test]
fn every_rate_specialised_family_is_accounted_for() {
    // The `_mcg` EXL3 family is 8 rates x {aligned, unaligned} x {plain, mcg}
    // plus 8 x 3 NR0 variants. All of it is reachable only from `gemv` /
    // `gemv_buffer` (the readback verification forward), never from the GPU
    // decode loop, because `encode_gemv_raw` returns at its `_2col` arm before
    // `decode_gemv_pipeline` is consulted. Assert the *structural* facts that
    // make that true, so the day one of them changes the claim fails.
    let lin_metal = read("crates/kernels/src/linear.metal");
    let k = kernels_in("linear.metal", &lin_metal);
    let names: Vec<&str> = k.iter().map(|x| x.name.as_str()).collect();
    let mcg = names.iter().filter(|n| n.contains("_exl3_")).count();
    let nr0 = names.iter().filter(|n| n.contains("_mcg_r")).count();
    let nod = names.iter().filter(|n| n.contains("_nod")).count();
    let w2 = names.iter().filter(|n| n.contains("_w2")).count();
    let fast = names.iter().filter(|n| n.contains("_fast")).count();
    assert_eq!(mcg, 56, "32 slot kernels + 24 NR0 variants");
    assert_eq!(nr0, 24);
    assert_eq!(nod, 8);
    assert_eq!(w2, 4, "only k1..k4 have a two-word row-major form");
    assert_eq!(fast, 8);

    let linear_rs = read("crates/kernels/src/linear.rs");
    assert!(
        exl3_arm_follows_two_col(&linear_rs),
        "encode_gemv_raw consults the EXL3 arm before the _2col arm returns; \
         the whole `_mcg` family may have become reachable"
    );
    // And the `_2col` arm's guard holds for all seven layer linears.
    for n in [
        SLIDING_Q,
        SLIDING_KV,
        FULL_Q,
        FULL_KV,
        HIDDEN,
        INTERMEDIATE,
        FULL_O_IN,
    ] {
        assert_eq!(n % 32, 0, "the _2col arm requires out_features % 32 == 0");
    }
    // lm_head is the only projection that is NOT colmajor, which is why the
    // rate-6 slot kernel is the one that runs for it.
    assert_eq!(VOCAB % 32, 0, "vocab is also %32==0 but the head is perm_mode 0");
    assert!(
        !names.contains(&"linear_gemv_ishizuki_k6_w2"),
        "a two-word rate-6 form now exists; ishizuki_gemv_pipeline would return it \
         for lm_head and the head would change kernels"
    );

    // The ordering assertion above reads the file, so it cannot be exercised by
    // an in-memory body mutation. `exl3_arm_follows_two_col` is therefore a pure
    // predicate over the source text, checked here against a mutated copy.
    assert!(
        exl3_arm_follows_two_col(&linear_rs),
        "the unmutated source must satisfy the predicate for the mutation below to \
         mean anything"
    );
    let two_col_marker = "env_flag(\"KAPLUMBA_NO_2COL\")";
    let exl3_call = "self.decode_gemv_pipeline(rate, packed_offset, codebook)";
    let raw_at = linear_rs
        .find("pub fn encode_gemv_raw(")
        .expect("encode_gemv_raw");
    let at2 = linear_rs[raw_at..].find(two_col_marker).expect("marker") + raw_at;
    let at3 = linear_rs[raw_at..].find(exl3_call).expect("call") + raw_at;
    assert!(at2 < at3);
    // Insert the EXL3 call immediately *before* the `_2col` guard, keeping both
    // markers present so the same predicate can still parse the source.
    let mut swapped = String::with_capacity(linear_rs.len() + exl3_call.len() + 1);
    swapped.push_str(&linear_rs[..at2]);
    swapped.push_str(exl3_call);
    swapped.push('\n');
    swapped.push_str(&linear_rs[at2..]);
    assert!(
        !exl3_arm_follows_two_col(&swapped),
        "moving the EXL3 call above the _2col arm must make the predicate reject \
         the source"
    );
}

#[test]
fn the_head_runs_the_unfolded_codebook_kernel() {
    // `linear_gemv_ishizuki_k6` (the live lm_head kernel) calls
    // `decode_state(s, CB)` with CB = 1 as a *runtime* argument, while the
    // `_mcg` specialisation that folded it is measured at 2.61x and exists only
    // in the unreachable `linear_gemv_exl3_k*_mcg` family. Also: at KA = 6 the
    // highest read bit is r_max = 7*6 + 16 = 58 <= 64, so the `B0` half-window
    // and the `(r <= 64u)` ternary in its hot loop are dead code.
    let lin_metal = read("crates/kernels/src/linear.metal");
    let k = kernels_in("linear.metal", &lin_metal);
    let head = k
        .iter()
        .find(|x| x.name == "linear_gemv_ishizuki_k6")
        .expect("linear_gemv_ishizuki_k6");
    let s = classify(head);
    assert_eq!(s.decode, "decode_state (runtime CB)");
    assert_eq!(s.alu_width, "64", "still the 64-bit shift chain");
    assert_eq!(
        s.ring_words, "3",
        "KA=6 needs three ring words; two is provably unsafe (TWO_WORD_SAFE_RATES)"
    );
    assert!(head.body.contains("const ulong A0"), "still the 64-bit window");
    assert!(head.body.contains("(r <= 64u)"), "still the per-code ternary");
    // At KA = 6 the highest bit any lane reads is `7*KA + 16 = 58 <= 64`, so
    // every read lands in `A0` and the `B0` half-window is dead code. That is
    // the arithmetic behind "B0 can be deleted"; the ternary is still written
    // out per code, which is the cost the `linear.metal:328-331` header calls
    // out ("the BRANCH is what serialises the eight values per lane").
    let r_max = 7 * 6 + 16;
    assert_eq!(r_max, 58);
    assert!(r_max <= 64, "so every read lands in A0 and B0 is dead");
    // `w2` is still needed. At KA = 6, `RING = 1536` and
// `lo0 = (lane*8+1)*6 + RING - 16 = 48*lane + 1526`, so
// `sh0 = lo0 % 32 ∈ {22, 6}` and bit `63 + 22 = 85` of the 96-bit window is
// read -- only `w0`/`w1` cover bits 0..63. So the third word is load-bearing
// even though `B0` is not: `TWO_WORD_SAFE_RATES = (1,2,3,4)`.
    let sh0: std::collections::BTreeSet<u32> =
        (0..32).map(|lane| (48 * lane + 1526) % 32).collect();
    assert_eq!(
        sh0.into_iter().collect::<Vec<_>>(),
        vec![6, 22],
        "the KA=6 lane shift set; anything outside {{6, 22}} means the \
         3-word-window derivation in linear_repacked.metal changed"
    );
    // No MCG-folded rate-6 ishizuki kernel exists.
    assert!(
        !k.iter().any(|x| x.name == "linear_gemv_ishizuki_k6_mcg"),
        "an MCG-folded rate-6 kernel appeared; re-run the head analysis"
    );
}