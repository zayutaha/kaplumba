//! MLP / QKV decode dispatch tree: which kernel each hand-written `if` chain
//! selects, in which order, and what the arithmetic says fusion actually buys.
//!
//! NO GPU. NO MODEL RUN. Pure source parsing plus integer arithmetic.
//!
//! # Why this file exists
//!
//! The decode GEMV selection is **hand-written if-chains in two files** —
//! `crates/kernels/src/linear.rs` (`encode_gemv_fused2_mlp`, `encode_gemv_fused2`,
//! `encode_gemv_fused3`, `encode_gemv_raw`, `encode_gemv_raw_sk2/sk4`) and
//! `crates/engine/src/model.rs` (`use_sk4` / `use_sk2`, and which of the three
//! fused entry points a layer calls). Every cell of the "which kernel" table in
//! the specs was established by reading it. A model of a dispatch tree cannot
//! detect a change to the dispatch tree, which is the exact failure this project
//! has shipped twice (`AGENTS.md` §3).
//!
//! So this file does not transcribe the tree. It **parses** the `if`/`else if`
//! chain out of the source text, evaluates the arms **in source order** against a
//! facts record for the production shapes, and then asserts the outcome. The
//! properties that make it more than a restatement:
//!
//! * arm **order** is read from the file, so reordering `linear.rs:2590` ahead of
//!   `linear.rs:2598` flips the selected kernel and this fails;
//! * nested `else if let Some(...)` chains (the shape both `_co` arms use) are
//!   flattened into the same ordered list with synthesised conditions, so
//!   `KAPLUMBA_FUSED2_CO4` actually steers the choice rather than being invisible;
//! * an arm condition containing a predicate this file does not model makes the
//!   parser **panic with the offending text**, and the set of deliberately
//!   skipped arms is pinned, so a newly added guard cannot be silently ignored;
//! * pipeline existence (`.is_some()`) is answered by parsing the `.metal`
//!   inventory, not by a hardcoded list;
//! * three negative controls run the *same* evaluator over a mutated in-memory
//!   copy of `linear.rs`, which is what establishes that the evaluator is
//!   reading the source rather than returning a constant.
//!
//! `perf_probe_dispatchsel_columns.rs` already pins the *predicates* by substring
//! and transcribes the resulting table. It deliberately does not model arm
//! order, arm reachability, the nested `_co4` sub-choice, or the row-major
//! checkpoint, and says so. This file is the complement.
//!
//! # THE ANSWERS
//!
//! ## 1. Selected kernel, colmajor 3 bpw (`Gxma-repacked`), default env
//!
//! | projection | layers | K x N | kernel | threadgroups |
//! |---|---|---|---|---|
//! | q,k,v (sliding) | 50 | 5376x8192, 5376x4096 x2 | `linear_gemv_ishizuki_k3_fused3_2col` | 512 |
//! | q,k (full attn, V=K copy) | 10 | 5376x16384, 5376x2048 | `linear_gemv_ishizuki_k3_fused2_2col` | 576 |
//! | o_proj | 60 | 8192x5376 (16384x5376 on full) | `linear_gemv_ishizuki_k3_colmajor_2col_sk2` | 336 |
//! | gate+up | 60 | 5376x21504 x2 | `linear_gemv_ishizuki_k3_fused2_2col` | 1344 |
//! | down_proj | 60 | 21504x5376 | `linear_gemv_ishizuki_k3_colmajor_2col_sk2` | 336 |
//! | lm_head | 1 | 5376x262144 @ 6 bpw, perm_mode 0 | `linear_gemv_ishizuki_k6` | 262144 |
//!
//! Three distinct layer GEMV kernels, none of them `_co`.
//!
//! ## 2. `fused2` vs `fused3`
//!
//! They are **different numbers of matrices**, not variants of one kernel.
//! `encode_gemv_fused2` takes `(a, b)`; `encode_gemv_fused3` takes `(a, b, c)`.
//! `fused2` is q+k on the 10 full-attention layers, where `k_eq_v` means V is the
//! same weight matrix as K, so V comes from an `encode_copy` (`model.rs:1588`)
//! rather than a third GEMV. `fused3` is q+k+v on the 50 sliding layers, which
//! is 8/9 of the model — **so yes, `fused3` is dispatched, on 50 of 60 layers**,
//! and it is the `qkv_proj` line in every profile.
//!
//! `fused2_2col` / `fused3_2col` are T4.4's two-column forms of those two; the
//! `_2col` in "gate+up uses `fused2_2col`" is the same idea applied to the MLP
//! entry point. The names collide across entry points, which is why the table
//! above is per-projection and not per-kernel-name.
//!
//! ## 3. The fallback chain, in source order, and its reachability
//!
//! `encode_gemv_fused2_mlp` after the `KAPLUMBA_USE_SK` early return:
//!
//! ```text
//! [0] KAPLUMBA_FUSED2_4COL  && gate%64==0 && up%64==0 && _4col exists  -> fused2_4col  (opt-in, UNMEASURED)
//! [1] !NO_FUSED2_2COL      && gate%32==0 && up%32==0 && _2col exists  -> fused2_2col  (DEFAULT)
//! [2] gate==up && !NO_FUSED2_CO  ->  [2a] CO4 flag -> fused2_co4
//!                                             [2b] else     -> fused2_co
//!                                             [2c] neither  -> fused2
//! [3] otherwise                                                     -> fused2
//! ```
//!
//! **Arm [2] is unreachable with default env.** `_2col`'s guard needs only
//! `out_features_gate % 32 == 0` (21504 % 32 = 0) and the pipeline to exist, and
//! it is tested first, so `_co` and `_co4` are dead code for this checkpoint.
//! `fused2_co` is T1/T2's kernel; T4.4's `_2col` reaches the *same* 1344
//! threadgroups and the same 64,512 B per threadgroup, so it captured all of
//! T1's occupancy benefit, and T5.2 then measured the remaining difference (two
//! interleaved DRAM streams per threadgroup vs one) as a loss.
//!
//! `encode_gemv_fused2` (the q+k path) has the **opposite order** — `_co` first,
//! `_2col` second. For this checkpoint q and k are never equal (8192 vs 4096,
//! 16384 vs 2048), so `_co` is unreachable there too. The two entry points
//! disagree about which of `_co`/`_2col` is preferred; harmless today only
//! because the guard that separates them never fires.
//!
//! `encode_gemv_fused3` has no `_co` arm at all: `_2col` (all three N % 32) and,
//! under `KAPLUMBA_USE_SK || KAPLUMBA_QKV_SK2`, `_sk2`.
//!
//! ## 4. Split-K for down_proj
//!
//! `model.rs:2237-2247`. `use_sk4` needs `KAPLUMBA_SK4`, which nobody sets.
//! `use_sk2` is an **opt-out** (`!KAPLUMBA_SK4 && !KAPLUMBA_NO_SK2 &&
//! !KAPLUMBA_D_NOSK`) and all three shape guards hold (perm_mode 1, 5376 % 32 = 0,
//! rate Integer(3)), so **SK2 fires by default** with 336 threadgroups. SK4 needs
//! the flag.
//!
//! So: **no best-known configuration is gated behind an unset env flag.** Every
//! cell that has been measured is already the default, and the three flags that
//! would change a default (`KAPLUMBA_FUSED2_4COL`, `KAPLUMBA_FUSED2_CO4`,
//! `KAPLUMBA_SK4`) gate configurations that measured equal or worse. Two traps
//! are asserted below because they are hazards rather than wins:
//!
//! * `model.rs:1284` makes `cross_layer_fusion_applies` require
//!   `!env_flag("KAPLUMBA_SK4")`. Setting `KAPLUMBA_SK4=1` to A/B split-K=4
//!   therefore **silently also deletes T3's cross-layer fusion** (the finalize +
//!   the next layer's input norm, fused into one dispatch). Any A/B of that flag
//!   measures two changes at once.
//! * `KAPLUMBA_NO_SK2=1` turns off split-K for o_proj *and* down_proj together,
//!   which is why its -0.58 tok/s (notes §16.5) cannot be attributed to either.
//!   `KAPLUMBA_O_NOSK` / `KAPLUMBA_D_NOSK` exist to isolate them (comment at
//!   `model.rs:1918-1926`) and **no isolated result is recorded in the notes**.
//!
//! ## 5. What fusion buys, in DRAM bytes
//!
//! Per layer, gate+up, 3 bpw colmajor:
//!
//! ```text
//! weights      86,704,128 B   (2 x 21504 x 5376 x 3/8)   <- 99.9876% of DRAM
//! x             10,752 B      (5376 f16)                 <-  0.0124% of DRAM
//! ```
//!
//! `x` is read by **every** threadgroup, so `x`'s L1<-L2 traffic is
//! `threadgroups x 10,752` = 14.45 MB for the `_2col` grid — 14.3% of that grid's
//! L1 traffic, every byte an L2 hit. `_co` reads `x` exactly as often as `_2col`
//! does: both launch 1344 threadgroups and both walk the whole 5376-long
//! activation in each. **`_co` halves `x` reads only against plain `fused2`
//! (2688 threadgroups), and `_2col` already achieves that identically.** So the
//! answer to "does `_co` halve `x` reads" is *no — not relative to what ships*.
//!
//! Writing both outputs from one threadgroup is likewise not a traffic effect:
//! `raw_gate` + `raw_up` are 172,032 B against 86.7 MB of weights, i.e. 0.2%.
//! Every variant reads every weight byte exactly once; that is asserted
//! directly ([`fusion_does_not_reduce_the_weight_traffic`]).
//!
//! What is left is bytes-per-threadgroup, and §16.3 measured bandwidth to be
//! monotone in that (and flat in N). [`bytes_per_threadgroup_table`] prints it.
//! `gate_up` at 64,512 B/TG is the widest gap to `down_proj`'s 129,024 B/TG, and
//! the two cells that would close it differ in exactly one variable:
//!
//! * `fused2_co4`: 129,024 B/TG over 672 TGs, but reads gate **and** up at once.
//!   Measured 15.46 vs `fused2_2col`'s 15.65 tok/s (notes §13), blamed on that.
//! * `fused2_4col`: 129,024 B/TG over 672 TGs, reads **four consecutive tiles of
//!   one matrix**. Never run. `linear.rs:2582` gates it behind
//!   `KAPLUMBA_FUSED2_4COL`, which appears in no benchmark in `spec/`.
//!
//! ## 6. QKV
//!
//! Beyond `fused3` there is no fusion variant. `fused3_2col` is the widest one and
//! it is already selected: 512 TGs, 64,512 B/TG, 33.04 MB, ~194 GB/s implied
//! (notes §16.2) against a 203 GB/s 3 bpw ceiling. Fusing q+k+v buys no traffic
//! (same `x` argument; `k_eq_v` makes two of the three the same matrix anyway)
//! and no bytes-per-threadgroup. `fused3_sk2` exists (`linear.rs:2735`) but needs
//! `KAPLUMBA_USE_SK` or `KAPLUMBA_QKV_SK2`; split-K on `gate_up` measured
//! -1.04 tok/s (§13) and the same law applies here.
//!
//! # Sabotage discipline
//!
//! A dispatch probe that has never been shown to fail is a comment. Three
//! changes to `crates/kernels/src/linear.rs` were applied, the suite run, and the
//! file restored (all three leave `git status` clean):
//!
//! | mutation | observed failure |
//! |---|---|
//! | `!env_flag("KAPLUMBA_NO_FUSED2_2COL")` -> `env_flag(...)` in `encode_gemv_fused2_mlp`, i.e. `_2col` becomes opt-in and `_co` wins by default | **6 tests FAILED**, `left: "fused2_2col"` where `"fused2_co"` was expected; `colmajor_default_env_selects_two_col_everywhere` at the `_2col` assertion |
//! | `(tiles_a + tiles_b) / 4` -> `/ 8` in the `_4col` arm: a pure arithmetic change with no guard involved | **2 tests FAILED**, `left: 336, right: 672` -- caught by the group count alone, with no flag and no shape change |
//! | `out_features_a == out_features_b && ...` -> `false && ...` in `encode_gemv_fused2`, deleting the `_co` arm's guard | **1 test FAILED**: `the_two_fused2_entry_points_disagree_about_co_and_2col_order`, on the arm-order list |
//!
//! The third is the one that matters most: it changes no selection (q != k, so
//! the `_co` arm was unreachable anyway) and still fails, because the assertion
//! is about the *order the source declares*, not about which kernel happens to
//! win today.
//!
//! # Run
//!
//! ```text
//! cargo test -p kaplumba-engine --test probe_mlpdispatch_tree -- --nocapture
//! ```

use std::collections::{BTreeMap, BTreeSet};

// ---------------------------------------------------------------------------
// The committed sources. Parsed rather than reflected, so this cannot be green
// against a stale checkout. `crates/kernels/src/linear.rs` is being edited
// concurrently by other agents, which is why every tripwire here is substring-
// or parse-based and never line-number-based.
// ---------------------------------------------------------------------------
const LINEAR_RS: &str = include_str!("../../kernels/src/linear.rs");
const MODEL_RS: &str = include_str!("../src/model.rs");
const LINEAR_METAL: &str = include_str!("../../kernels/src/linear.metal");
const REPACKED_METAL: &str = include_str!("../../kernels/src/linear_repacked.metal");
const CONFIG_RS: &str = include_str!("../src/config.rs");

/// Threads per threadgroup on every decode GEMV in this tree.
const THREADS_PER_TG: u64 = 256;

/// Bytes in one 16x16 trellis tile at 3 bpw: 256 codes * 3 bits = 768 bits =
/// 96 B. Mirrors `RING / WORDS` in `linear_repacked.metal`.
const BYTES_PER_TILE_3BPW: u64 = 96;

// ---------------------------------------------------------------------------
// Shapes. Facts about the checkpoint, pinned against `config.rs` below.
// ---------------------------------------------------------------------------
const HIDDEN: usize = 5376;
const INTERMEDIATE: usize = 21504;
/// `vocab_size`; `lm_head` is the only projection that is not 3 bpw and the only
/// one that keeps `permutation_mode == 0`.
const VOCAB: usize = 262_144;
/// `num_attention_heads * head_dim` (32 * 256), sliding layers.
const Q_OUT_SLIDING: usize = 8192;
/// `num_attention_heads * global_head_dim` (32 * 512), full-attention layers.
const Q_OUT_FULL: usize = 16_384;
/// `num_key_value_heads * head_dim` (16 * 256).
const KV_OUT_SLIDING: usize = 4096;
/// `num_global_key_value_heads * global_head_dim` (4 * 512).
const KV_OUT_FULL: usize = 2048;
/// `layer_types` marks every 6th layer full-attention: 10 of 60.
const SLIDING_LAYERS: usize = 50;
const FULL_LAYERS: usize = 10;

/// `3 bpw` weight bytes for `rows x k`.
fn weight_bytes(rows: u64, k: usize, bits: u8) -> u64 {
    rows * k as u64 * bits as u64 / 8
}

// ---------------------------------------------------------------------------
// Kernel inventory, parsed from the shader sources.
// ---------------------------------------------------------------------------

/// `linear_gemv_ishizuki_*` function names in `source`, in source order.
///
/// The prefix filter is load-bearing: the banner comment at
/// `linear_repacked.metal:8` contains the text "kernel void linear_gemv(".
fn ishizuki_names_in(source: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut rest = source;
    while let Some(at) = rest.find("kernel void ") {
        rest = &rest[at + "kernel void ".len()..];
        let end = rest
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .unwrap_or(rest.len());
        let name = &rest[..end];
        if name.starts_with("linear_gemv_ishizuki_") {
            names.push(name.to_string());
        }
    }
    names
}

fn ishizuki_inventory() -> BTreeSet<String> {
    let set: BTreeSet<String> = ishizuki_names_in(LINEAR_METAL)
        .into_iter()
        .chain(ishizuki_names_in(REPACKED_METAL))
        .collect();
    // An empty or tiny inventory would make every `.is_some()` false and turn
    // this file into a tautology.
    assert!(
        set.len() > 40,
        "kernel inventory collapsed to {} entries; the .metal sources did not parse",
        set.len()
    );
    set
}

// ---------------------------------------------------------------------------
// A tiny scanner that understands Rust enough to find braces without being
// fooled by `format!("...k{bits}...")` or by comments.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    Code,
    Str,
    Char,
    LineComment,
    BlockComment,
}

/// Advance past `text[pos]` in `mode`, returning the new mode.
fn step(text: &str, pos: usize, mode: Mode) -> (usize, Mode) {
    let bytes = text.as_bytes();
    let b = bytes[pos];
    match mode {
        Mode::Code => {
            if b == b'/' && bytes.get(pos + 1) == Some(&b'/') {
                (pos + 2, Mode::LineComment)
            } else if b == b'/' && bytes.get(pos + 1) == Some(&b'*') {
                (pos + 2, Mode::BlockComment)
            } else if b == b'"' {
                (pos + 1, Mode::Str)
            } else if b == b'\'' {
                // A lifetime tick (`&'a str`, `<'_>`, `&'static T`) is not a char
                // literal. `_` matters: without it the scanner swallows the rest
                // of `fn foo<'_>()` as a char literal and every later `find_code`
                // in the file returns `None`.
                let is_lifetime = matches!(bytes.get(pos + 1), Some(c) if c.is_ascii_alphabetic() || *c == b'_')
                    && bytes.get(pos + 2) != Some(&b'\'');
                if is_lifetime {
                    (pos + 1, Mode::Code)
                } else {
                    (pos + 1, Mode::Char)
                }
            } else {
                (pos + 1, Mode::Code)
            }
        }
        Mode::Str => {
            if b == b'\\' {
                (pos + 2, Mode::Str)
            } else if b == b'"' {
                (pos + 1, Mode::Code)
            } else {
                (pos + 1, Mode::Str)
            }
        }
        Mode::Char => {
            if b == b'\\' {
                (pos + 2, Mode::Char)
            } else if b == b'\'' {
                (pos + 1, Mode::Code)
            } else {
                (pos + 1, Mode::Char)
            }
        }
        Mode::LineComment => {
            if b == b'\n' {
                (pos + 1, Mode::Code)
            } else {
                (pos + 1, Mode::LineComment)
            }
        }
        Mode::BlockComment => {
            if b == b'*' && bytes.get(pos + 1) == Some(&b'/') {
                (pos + 2, Mode::Code)
            } else {
                (pos + 1, Mode::BlockComment)
            }
        }
    }
}

/// Byte offset just past the `}` that closes the `{` at `open`.
fn match_brace(text: &str, open: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut depth = 0i32;
    let mut pos = open;
    let mut mode = Mode::Code;
    while pos < bytes.len() {
        let (next, new_mode) = step(text, pos, mode);
        if mode == Mode::Code {
            match bytes[pos] {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(next);
                    }
                }
                _ => {}
            }
        }
        pos = next;
        mode = new_mode;
    }
    None
}

/// Offset of the first byte of `needle` at or after `from` that is in code mode.
fn find_code(text: &str, from: usize, needle: &str) -> Option<usize> {
    let bytes = text.as_bytes();
    let n = needle.as_bytes();
    let mut pos = from;
    let mut mode = Mode::Code;
    while pos + n.len() <= bytes.len() {
        let (next, new_mode) = step(text, pos, mode);
        if mode == Mode::Code && &bytes[pos..pos + n.len()] == n {
            return Some(pos);
        }
        pos = next;
        mode = new_mode;
    }
    None
}

/// Collapse whitespace so a multi-line condition is one comparable string.
fn normalise(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn line_of(src: &str, offset: usize) -> usize {
    src[..offset].bytes().filter(|b| *b == b'\n').count() + 1
}

// ---------------------------------------------------------------------------
// Arm extraction.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Arm {
    /// Normalised condition; empty for a terminal `else`.
    cond: String,
    body: String,
    /// `let groups = ...` / `(pipe, ...)` expression, when the arm computes one.
    groups: Option<String>,
    /// 1-based line in `linear.rs`, for the report.
    line: usize,
}

impl Arm {
    /// The single `linear_gemv_ishizuki_*` literal this arm dispatches.
    fn kernel(&self) -> String {
        let names = ishizuki_literals(&self.body);
        assert_eq!(
            names.len(),
            1,
            "arm at linear.rs:{} ({:?}) names {names:?}, expected exactly one",
            self.line, self.cond
        );
        names.into_iter().next().unwrap()
    }
}

// ---------------------------------------------------------------------------
// Arm extraction.
// ---------------------------------------------------------------------------

/// A parsed entry point: the ordered, flattened arm list plus the skip list.
struct Tree {
    arms: Vec<Arm>,
    skipped: Vec<String>,
}

impl Tree {
    fn select(&self, facts: &Facts) -> Choice {
        select(&self.arms, facts).unwrap_or_else(|| panic!("no arm matched for {facts:?}"))
    }

    /// The kernel each arm dispatches, in source order.
    fn arm_kernels(&self) -> Vec<String> {
        self.arms.iter().map(|a| a.kernel()).collect()
    }

    /// Only the arms whose condition this file can decide.
    fn modelled(&self) -> Vec<&Arm> {
        self.arms.iter().filter(|a| modellable(&a.cond)).collect()
    }

    /// Assert that *every* arm is decidable, so the fused chains cannot quietly
    /// grow an arm this file silently declines to evaluate.
    fn assert_all_modelled(&self) {
        let unmodelled: Vec<&str> = self
            .arms
            .iter()
            .filter(|a| !modellable(&a.cond))
            .map(|a| a.cond.as_str())
            .collect();
        assert!(
            unmodelled.is_empty(),
            "these arms cannot be evaluated and would be skipped silently: \
             {unmodelled:#?}"
        );
    }
}

/// All `linear_gemv_ishizuki_*` literals in `body`, deduped, in source order.
fn ishizuki_literals(body: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut rest = body;
    while let Some(at) = rest.find("\"linear_gemv_ishizuki_") {
        rest = &rest[at + 1..];
        let tail = &rest["linear_gemv_ishizuki_".len()..];
        let end = tail
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .unwrap_or(tail.len());
        names.push(format!("linear_gemv_ishizuki_{}", &tail[..end]));
        rest = &tail[end..];
    }
    names.dedup();
    names
}

/// The group expression in `body`, or `None` if it is a local the expression
/// evaluator cannot model (`quads.min(gemv_max_groups()).max(1)`).
fn groups_expr(body: &str) -> Option<String> {
    for marker in ["(pipe, ", "?, "] {
        let mut from = 0usize;
        while let Some(rel) = body[from..].find(marker) {
            let at = from + rel;
            let after = &body[at + marker.len()..];
            if let Some(end) = balanced_close(after) {
                let expr = after[..end].trim().trim_end_matches(',').trim();
                if expr.contains("tiles_") {
                    return Some(normalise(expr));
                }
            }
            from = at + marker.len();
        }
    }
    if let Some(at) = body.find("let groups = ") {
        let after = &body[at + "let groups = ".len()..];
        let end = after.find(';').unwrap_or(after.len());
        let expr = after[..end].trim();
        if expr.contains("tiles_") {
            return Some(normalise(expr));
        }
    }
    None
}

/// Offset of the `)` that closes the `(` implicitly opened before `text`.
fn balanced_close(text: &str) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut depth = 1i32;
    let mut pos = 0usize;
    let mut mode = Mode::Code;
    while pos < bytes.len() {
        let (next, new_mode) = step(text, pos, mode);
        if mode == Mode::Code {
            match bytes[pos] {
                b'(' => depth += 1,
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(pos);
                    }
                }
                _ => {}
            }
        }
        pos = next;
        mode = new_mode;
    }
    None
}

/// The name inside `ishizuki_gemv_pipeline_by_name("X")`.
fn named_in_header(header: &str) -> Option<String> {
    let at = header.find("\"linear_gemv_ishizuki_")?;
    let tail = &header[at + 1 + "linear_gemv_ishizuki_".len()..];
    let end = tail
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .unwrap_or(tail.len());
    Some(format!("linear_gemv_ishizuki_{}", &tail[..end]))
}

/// One element of a nested chain.
struct Link {
    /// Normalised condition.
    cond: String,
    body: String,
    /// Set when the condition was `let Some(x) = self.f("NAME")`: the dispatched
    /// kernel lives in the *header*, not the body, so it has to travel with the
    /// link or the leaf cannot name what it dispatches.
    kernel: Option<String>,
}

/// Rewrite a `let` pattern guard into something the evaluator can decide.
///
/// * `let Some(x) = self.ishizuki_gemv_pipeline_by_name("NAME")` is a pipeline
///   existence test wearing a pattern-match costume.
/// * `let Exl3Rate::Integer(bits) = rate` (and any other match on a bare value)
///   is *true* on the decode path: `validate_encode` has already rejected every
///   other rate, and the caller is `encode_gemv`, which only forwards.
/// * anything else is left alone, and the arm is skipped rather than guessed.
fn let_guard(cond: &str) -> (String, Option<String>) {
    let trimmed = cond.trim();
    let after_if = trimmed.strip_prefix("if ").unwrap_or(trimmed);
    if !after_if.starts_with("let ") {
        return (cond.to_string(), None);
    }
    let rhs = match after_if.split_once(" = ") {
        Some((_, rhs)) => rhs.trim(),
        None => return (cond.to_string(), None),
    };
    if let Some(name) = named_in_header(after_if) {
        return (
            format!("self.ishizuki_gemv_pipeline_by_name({name:?}).is_some()"),
            Some(name),
        );
    }
    if !rhs.contains('(') {
        return ("true".to_string(), None);
    }
    (cond.to_string(), None)
}

/// Parse an `if c1 { B1 } else if c2 { B2 } ... else { Bn }` chain starting at the
/// beginning of `body`. Returns `None` when `body` does not start a chain.
fn chain_in(body: &str) -> Option<(Vec<Link>, Option<String>)> {
    let start = body.len() - body.trim_start().len();
    if !body[start..].starts_with("if ") {
        return None;
    }
    let mut links: Vec<Link> = Vec::new();
    let mut terminal: Option<String> = None;
    let mut pos = start + 3;
    loop {
        let brace = find_code(body, pos, "{")?;
        let (cond, kernel) = let_guard(&normalise(&body[pos..brace]));
        let close = match_brace(body, brace)?;
        links.push(Link {
            cond,
            body: body[brace + 1..close].to_string(),
            kernel,
        });

        // `match_brace` returns one past the arm's `}`, so a `}` belonging to
        // an enclosing block sits between the arm and its `else`. Skip it, or
        // the chain's terminal `else` is never found -- which is how the whole
        // outer fallback of both fused chains went missing the first time.
        let mut after = close;
        loop {
            let ws = &body[after..];
            let t = ws.trim_start();
            after += ws.len() - t.len();
            if t.starts_with('}') {
                after += 1;
                continue;
            }
            break;
        }
        if !body[after..].starts_with("else") {
            break;
        }
        after += 4;
        let ws2 = &body[after..];
        let t2 = ws2.trim_start();
        after += ws2.len() - t2.len();
        if t2.starts_with("if ") {
            pos = after + 3;
            continue;
        }
        let brace2 = find_code(body, after, "{")?;
        let close2 = match_brace(body, brace2)?;
        terminal = Some(body[brace2 + 1..close2].to_string());
        break;
    }
    // A bare `if { ... return ...; }` has no terminal `else`; that is the common
    // shape of the perm-gated colmajor arms, so an absent else must not make the
    // whole chain unreadable.
    Some((links, terminal))
}

/// Recursively flatten `body` under `prefix`, appending arms to `out`.
///
/// The `_co` arms nest two levels deep (`if CO4-flag { if let co4 {} else if let
/// co {} else {} } else if let co {} else {}`), so a single level is not enough
/// and the conditions have to be composed rather than merely prefixed.
fn flatten(
    prefix: &str,
    body: &str,
    line: usize,
    out: &mut Vec<Arm>,
    skipped: &mut Vec<String>,
    inherited: Option<String>,
) {
    if let Some((links, terminal)) = chain_in(body) {
        let mut negatives: Vec<String> = Vec::new();
        for link in &links {
            let cond = if prefix.is_empty() {
                link.cond.clone()
            } else {
                // Both sides parenthesised: `a || b && c` is not `(a || b) && c`,
                // and `fused3`'s split-K guard is a disjunction.
                format!("({prefix}) && ({})", link.cond)
            };
            flatten(
                &cond,
                &link.body,
                line,
                out,
                skipped,
                link.kernel.clone().or_else(|| inherited.clone()),
            );
            negatives.push(link.cond.clone());
        }
        if let Some(terminal) = terminal {
            let mut cond = if prefix.is_empty() {
                String::new()
            } else {
                format!("({prefix})")
            };
            for neg in &negatives {
                cond = if cond.is_empty() {
                    format!("!({neg})")
                } else {
                    format!("{cond} && !({neg})")
                };
            }
            flatten(&cond, &terminal, line, out, skipped, inherited.clone());
        }
        return;
    }
    // An arm whose condition this file cannot decide is skipped, not claimed.
    // The skip list is pinned by a test, so a guard added later shows up there
    // instead of quietly becoming an arm nobody evaluated.
    if !modellable(prefix) {
        skipped.push(prefix.to_string());
        return;
    }
    let names = ishizuki_literals(body);
    match (names.len(), inherited) {
        (1, _) => out.push(Arm {
            cond: prefix.to_string(),
            body: body.to_string(),
            groups: groups_expr(body),
            line,
        }),
        (0, Some(kernel)) => out.push(Arm {
            cond: prefix.to_string(),
            body: format!("\"{kernel}\""),
            groups: groups_expr(body),
            line,
        }),
        (0, None) => skipped.push(prefix.to_string()),
        (n, _) => panic!(
            "arm at linear.rs:{line} ({prefix:?}) names {n} kernels {names:?}; the \
             dispatch decision cannot be read, so it must not be guessed at"
        ),
    }
}

/// Can the evaluator decide this condition at all?
fn modellable(cond: &str) -> bool {
    if cond == "true" || cond.is_empty() {
        return true;
    }
    for tok in cond.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')) {
        if matches!(
            tok,
            "" | "env_flag"
                | "perm_mode"
                | "matches"
                | "rate"
                | "Exl3Rate"
                | "Integer"
                | "out_features"
                | "out_features_a"
                | "out_features_b"
                | "out_features_c"
                | "out_features_gate"
                | "out_features_up"
                | "self"
                | "ishizuki_gemv_pipeline_by_name"
                | "is_some"
                | "true"
        ) {
            continue;
        }
        if tok.starts_with("KAPLUMBA_") || tok.starts_with("linear_gemv_ishizuki_") {
            continue;
        }
        if !tok.is_empty() && tok.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        return false;
    }
    true
}

/// How far the chain starting at `pos` reaches, terminal `else` included.
///
/// Needed because the chain is reached three ways: as a bare statement, as the
/// tail of `let (pipeline, groups) = if ...`, and as a nested
/// `if CO4 { if let co4 {} else if let co {} else {} }`. Handling one link at a
/// time from the statement scanner finds each `else if` separately and never
/// reaches the terminal `else`, which is exactly the arm that decides the
/// fallback -- so the extent is measured first and the whole chain handed over.
fn chain_extent(body: &str, pos: usize) -> usize {
    let mut p = pos + 3; // past "if "
    loop {
        let Some(brace) = find_code(body, p, "{") else {
            return p;
        };
        let Some(close) = match_brace(body, brace) else {
            return p;
        };
        let mut after = close;
        loop {
            let ws = &body[after..];
            let t = ws.trim_start();
            after += ws.len() - t.len();
            if t.starts_with('}') {
                after += 1;
                continue;
            }
            break;
        }
        if !body[after..].starts_with("else") {
            return close;
        }
        after += 4;
        let ws2 = &body[after..];
        let t2 = ws2.trim_start();
        after += ws2.len() - t2.len();
        if t2.starts_with("if ") {
            p = after;
            continue;
        }
        let Some(brace2) = find_code(body, after, "{") else {
            return close;
        };
        return match_brace(body, brace2).unwrap_or(close);
    }
}

/// The ordered decision arms of a function body, every nested `else if` chain
/// flattened with composed conditions.
fn parse_arms(src: &str, body: &str, body_at: usize) -> (Vec<Arm>, Vec<String>) {
    let bytes = body.as_bytes();
    let mut arms: Vec<Arm> = Vec::new();
    let mut skipped: Vec<String> = Vec::new();
    let mut pos = 0usize;
    let mut depth = 0i32;
    let mut mode = Mode::Code;
    while pos < bytes.len() {
        let (next, new_mode) = step(body, pos, mode);
        if mode == Mode::Code
            && depth == 0
            && body[pos..].starts_with("if ")
            && is_word_boundary(bytes, pos)
        {
            let after_close = {
                if find_code(body, pos + 3, "{").is_none() {
                    panic!("arm at {} has no body brace", line_of(src, body_at + pos))
                }
                let extent = chain_extent(body, pos);
                flatten(
                    "",
                    &body[pos..extent],
                    line_of(src, body_at + pos),
                    &mut arms,
                    &mut skipped,
                    None,
                );
                extent
            };
            pos = after_close;
            mode = Mode::Code;
            continue;
        }
        if mode == Mode::Code {
            match bytes[pos] {
                b'{' => depth += 1,
                b'}' => depth -= 1,
                _ => {}
            }
        }
        pos = next;
        mode = new_mode;
    }
    (arms, skipped)
}

fn is_word_boundary(bytes: &[u8], pos: usize) -> bool {
    pos == 0 || !(bytes[pos - 1] as char).is_ascii_alphanumeric() && bytes[pos - 1] != b'_'
}

fn tree(src: &str, header: &str) -> Tree {
    let at = find_code(src, 0, header)
        .unwrap_or_else(|| panic!("function {header} not found; the dispatch tree moved"));
    let open = find_code(src, at, "{").unwrap_or_else(|| panic!("{header}: no body"));
    let close = match_brace(src, open).unwrap_or_else(|| panic!("{header}: unterminated body"));
    // `match_brace` returns one past the `}`; the body excludes both delimiters.
    let (arms, skipped) = parse_arms(src, &src[open + 1..close - 1], open + 1);
    assert!(!arms.is_empty(), "{header}: no decision arms parsed");
    Tree { arms, skipped }
}

fn mlp_tree(src: &str) -> Tree {
    tree(src, "pub fn encode_gemv_fused2_mlp(")
}
fn fused2_tree(src: &str) -> Tree {
    tree(src, "pub fn encode_gemv_fused2(")
}
fn fused3_tree(src: &str) -> Tree {
    tree(src, "pub fn encode_gemv_fused3(")
}
fn gemv_raw_tree(src: &str) -> Tree {
    tree(src, "pub fn encode_gemv_raw(")
}

// ---------------------------------------------------------------------------
// Facts + a strict evaluator.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Facts {
    n: BTreeMap<&'static str, usize>,
    perm_mode: u32,
    bits: u8,
    flags: BTreeSet<String>,
    inventory: BTreeSet<String>,
}

impl Facts {
    fn colmajor(bits: u8) -> Self {
        Self {
            n: BTreeMap::new(),
            perm_mode: 1,
            bits,
            flags: BTreeSet::new(),
            inventory: ishizuki_inventory(),
        }
    }

    fn row_major(bits: u8) -> Self {
        let mut f = Self::colmajor(bits);
        f.perm_mode = 0;
        f
    }

    fn with(mut self, name: &'static str, value: usize) -> Self {
        self.n.insert(name, value);
        self
    }

    fn flag(mut self, name: &str) -> Self {
        self.flags.insert(name.to_string());
        self
    }

    fn flag_on(&self, name: &str) -> bool {
        self.flags.contains(name)
    }

    fn n(&self, name: &str) -> usize {
        *self.n.get(name).unwrap_or_else(|| panic!("facts are missing {name}"))
    }

    /// `linear.rs` names the three output widths `out_features_a/b/c` in
    /// `encode_gemv_fused2`/`_fused3` and `out_features_gate`/`_up` in
    /// `encode_gemv_fused2_mlp`, while both use `tiles_a`/`tiles_b`. Resolved
    /// by looking for whichever spelling the facts carry.
    fn tiles(&self, which: &str) -> u64 {
        let names: &[&str] = match which {
            "a" => &["out_features_a", "out_features_gate"],
            "b" => &["out_features_b", "out_features_up"],
            _ => &["out_features_c"],
        };
        for name in names {
            if let Some(v) = self.n.get(name) {
                return *v as u64 / 16;
            }
        }
        panic!("facts carry no output width for tiles_{which}")
    }

    /// Evaluate a condition written the way `linear.rs` writes one.
    fn eval(&self, cond: &str) -> Option<bool> {
        eval_bool(cond, self)
    }
}

#[derive(Debug, Clone)]
struct Choice {
    kernel: String,
    groups: u64,
    /// The arm that chose it, kept so a test can read its synthesised guard.
    #[allow(dead_code)]
    arm: Arm,
}

impl Choice {
    fn threads(&self) -> u64 {
        self.groups * THREADS_PER_TG
    }
}

/// Evaluate an ordered arm list and take the first arm whose condition holds.
fn select(arms: &[Arm], facts: &Facts) -> Option<Choice> {
    for arm in arms {
        if !eval_bool(&arm.cond, facts).unwrap_or_else(|| {
            panic!(
                "condition the evaluator does not model: {:?}\nEither teach \
                 `eval_atom` the new predicate or pin it in the skip list.",
                arm.cond
            )
        }) {
            continue;
        }
        let groups = match arm.groups.as_deref() {
            Some(expr) => eval_groups(expr, facts),
            None => panic!(
                "arm at linear.rs:{} ({}) has no group expression this file can read",
                arm.line, arm.cond
            ),
        };
        return Some(Choice {
            kernel: arm.kernel(),
            groups,
            arm: arm.clone(),
        });
    }
    None
}

fn eval_bool(text: &str, facts: &Facts) -> Option<bool> {
    let tokens = lex(text);
    let mut pos = 0usize;
    let value = eval_or(&tokens, &mut pos, facts)?;
    if pos != tokens.len() {
        return None;
    }
    Some(value)
}

fn eval_or(tokens: &[String], pos: &mut usize, facts: &Facts) -> Option<bool> {
    let mut left = eval_and(tokens, pos, facts)?;
    while tokens.get(*pos).map(String::as_str) == Some("||") {
        *pos += 1;
        let right = eval_and(tokens, pos, facts)?;
        left = left || right;
    }
    Some(left)
}

fn eval_and(tokens: &[String], pos: &mut usize, facts: &Facts) -> Option<bool> {
    let mut left = eval_unary(tokens, pos, facts)?;
    while tokens.get(*pos).map(String::as_str) == Some("&&") {
        *pos += 1;
        let right = eval_unary(tokens, pos, facts)?;
        left = left && right;
    }
    Some(left)
}

fn eval_unary(tokens: &[String], pos: &mut usize, facts: &Facts) -> Option<bool> {
    if tokens.get(*pos).map(String::as_str) == Some("!") {
        *pos += 1;
        return Some(!eval_unary(tokens, pos, facts)?);
    }
    if tokens.get(*pos).map(String::as_str) == Some("(") {
        *pos += 1;
        let inner = eval_or(tokens, pos, facts)?;
        if tokens.get(*pos).map(String::as_str) != Some(")") {
            return None;
        }
        *pos += 1;
        return Some(inner);
    }
    eval_atom(tokens, pos, facts)
}

fn lex(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let bytes = text.as_bytes();
    let mut pos = 0usize;
    while pos < bytes.len() {
        let two = if pos + 2 <= bytes.len() {
            &text[pos..pos + 2]
        } else {
            ""
        };
        if matches!(two, "&&" | "||" | "==" | "!=") || (two == "% " ) {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            out.push(two.trim_end().to_string());
            pos += 2;
            continue;
        }
        let c = bytes[pos] as char;
        if c.is_whitespace() {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            pos += 1;
            continue;
        }
        if matches!(c, '(' | ')' | ',' | '%' | '!' | '.' | ':') {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            out.push(c.to_string());
            pos += 1;
            continue;
        }
        cur.push(c);
        pos += 1;
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn eval_atom(tokens: &[String], pos: &mut usize, facts: &Facts) -> Option<bool> {
    let head = tokens.get(*pos)?.clone();
    match head.as_str() {
        "(" => {
            *pos += 1;
            let inner = eval_or(tokens, pos, facts)?;
            *pos += 1;
            Some(inner)
        }
        "true" => {
            *pos += 1;
            Some(true)
        }
        "env_flag" => {
            *pos += 1;
            expect(tokens, pos, "(")?;
            let name = tokens.get(*pos)?.trim_matches('"').to_string();
            *pos += 1;
            expect(tokens, pos, ")")?;
            Some(facts.flag_on(&name))
        }
        // `matches!(rate, Exl3Rate::Integer(N))`. The token stream is walked by
        // name rather than by index so the `::` split cannot drift.
        "matches" => {
            *pos += 1;
            expect(tokens, pos, "(")?;
            *pos += 1; // rate
            expect(tokens, pos, ",")?;
            *pos += 1; // Exl3Rate
            *pos += 1; // :
            *pos += 1; // :
            let variant = tokens.get(*pos)?.clone();
            *pos += 1;
            let bits = if variant == "Integer" {
                let v = tokens.get(*pos)?.clone();
                *pos += 1;
                v.parse::<u8>().ok()?
            } else {
                panic!("unmodelled matches! variant {variant}")
            };
            expect(tokens, pos, ")")?;
            Some(bits == facts.bits)
        }
        name if name.starts_with("out_features") || name == "perm_mode" => {
            *pos += 1;
            let op = tokens.get(*pos)?.clone();
            if op == "==" {
                *pos += 1;
                let rhs = tokens.get(*pos)?.clone();
                *pos += 1;
                if let Ok(v) = rhs.parse::<u32>() {
                    let lhs = if name == "perm_mode" {
                        facts.perm_mode
                    } else {
                        facts.n(name) as u32
                    };
                    return Some(lhs == v);
                }
                let lhs = if name == "perm_mode" {
                    facts.perm_mode
                } else {
                    facts.n(name) as u32
                };
                let rhs = if rhs == "perm_mode" {
                    facts.perm_mode
                } else {
                    facts.n(&rhs) as u32
                };
                return Some(lhs == rhs);
            }
            if op == "%" {
                *pos += 1;
                let div = tokens.get(*pos)?.parse::<usize>().ok()?;
                *pos += 1;
                // `expect` advances past `==` itself; a second `+= 1` here
                // reads one past the right operand and returns `None` from the
                // end of the token list, which silently reads as "unmodelled".
                expect(tokens, pos, "==")?;
                let rhs = tokens.get(*pos)?.parse::<usize>().ok()?;
                *pos += 1;
                return Some(facts.n(name) % div == rhs);
            }
            None
        }
        "self" => {
            *pos += 1;
            expect(tokens, pos, ".")?;
            let fname = tokens.get(*pos)?.clone();
            *pos += 1;
            if fname != "ishizuki_gemv_pipeline_by_name" {
                return None;
            }
            expect(tokens, pos, "(")?;
            let name = tokens.get(*pos)?.trim_matches('"').to_string();
            *pos += 1;
            expect(tokens, pos, ")")?;
            expect(tokens, pos, ".")?;
            let meth = tokens.get(*pos)?.clone();
            *pos += 1;
            // `.is_some()` carries an empty argument list the tokenizer keeps.
            if tokens.get(*pos).map(String::as_str) == Some("(") {
                expect(tokens, pos, "(")?;
                expect(tokens, pos, ")")?;
            }
            let exists = facts.inventory.contains(&name);
            match meth.as_str() {
                "is_some" => Some(exists),
                "is_none" => Some(!exists),
                other => panic!("unmodelled Option method {other}"),
            }
        }
        other => panic!(
            "eval_atom: unmodelled token {other:?}; the whole point of this file is \
             that a new guard must be loud"
        ),
    }
}

fn expect(tokens: &[String], pos: &mut usize, want: &str) -> Option<()> {
    if tokens.get(*pos).map(String::as_str) == Some(want) {
        *pos += 1;
        Some(())
    } else {
        None
    }
}

fn eval_groups(expr: &str, facts: &Facts) -> u64 {
    let tokens = lex(expr);
    let mut pos = 0usize;
    let v = groups_sum(&tokens, &mut pos, facts);
    assert_eq!(pos, tokens.len(), "unparsed group expression {expr:?}");
    v
}

fn groups_sum(tokens: &[String], pos: &mut usize, facts: &Facts) -> u64 {
    let mut v = groups_term(tokens, pos, facts);
    while tokens.get(*pos).map(String::as_str) == Some("+") {
        *pos += 1;
        v += groups_term(tokens, pos, facts);
    }
    v
}

fn groups_term(tokens: &[String], pos: &mut usize, facts: &Facts) -> u64 {
    let mut v = groups_atom(tokens, pos, facts);
    while tokens.get(*pos).map(String::as_str) == Some("/") || tokens.get(*pos).map(String::as_str) == Some("*")
    {
        let op = tokens[*pos].clone();
        *pos += 1;
        let rhs = groups_atom(tokens, pos, facts);
        if op == "/" {
            v /= rhs;
        } else {
            v *= rhs;
        }
    }
    v
}

fn groups_atom(tokens: &[String], pos: &mut usize, facts: &Facts) -> u64 {
    match tokens.get(*pos).map(String::as_str) {
        Some("(") => {
            *pos += 1;
            let v = groups_sum(tokens, pos, facts);
            assert_eq!(tokens.get(*pos).map(String::as_str), Some(")"));
            *pos += 1;
            v
        }
        Some(name) if name.starts_with("tiles_") => {
            *pos += 1;
            let which = name.trim_start_matches("tiles_");
            facts.tiles(which)
        }
        Some(lit) => {
            *pos += 1;
            lit.parse::<u64>()
                .unwrap_or_else(|_| panic!("bad group literal {lit} in the arm's group expression"))
        }
        None => panic!("group expression ended early"),
    }
}

// ---------------------------------------------------------------------------
// `model.rs`-level decisions, transcribed with substring tripwires.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum QkvPath {
    Fused3,
    Fused2PlusCopy,
    Separate,
}

/// model.rs:1462 / :1534 / :1591, in that order.
fn qkv_path(v_proj_present: bool, decode: bool, perm_mode: u32) -> QkvPath {
    if decode && perm_mode == 1 && v_proj_present {
        QkvPath::Fused3
    } else if decode && perm_mode == 1 {
        QkvPath::Fused2PlusCopy
    } else {
        QkvPath::Separate
    }
}

/// Which arm `down_proj` / `o_proj` take: model.rs:2237-2247 / :1933-1943.
fn split_k_path(facts: &Facts, out_features: usize, o: bool) -> &'static str {
    let sk4_flag = if o { "KAPLUMBA_O_SK4" } else { "KAPLUMBA_SK4" };
    let nosk_flag = if o { "KAPLUMBA_O_NOSK" } else { "KAPLUMBA_D_NOSK" };
    let shape_ok =
        facts.perm_mode == 1 && out_features % 32 == 0 && facts.bits == 3;
    if shape_ok && facts.flag_on(sk4_flag) {
        "sk4"
    } else if shape_ok && !facts.flag_on("KAPLUMBA_NO_SK2") && !facts.flag_on(nosk_flag) {
        "sk2"
    } else {
        "no-split"
    }
}

/// model.rs:1279-1293.
fn cross_layer_fusion(facts: &Facts, next_layer: bool, down_out: usize) -> bool {
    facts.n("tokens") == 1
        && !facts.flag_on("KAPLUMBA_NO_CROSS_LAYER")
        && facts.perm_mode == 1
        && down_out % 32 == 0
        && facts.bits == 3
        && !facts.flag_on("KAPLUMBA_SK4")
        && !facts.flag_on("KAPLUMBA_NO_SK2")
        && !facts.flag_on("KAPLUMBA_D_NOSK")
        && next_layer
}

/// linear.rs:2243-2260: which of the three sk2 names the default picks.
fn sk2_name(bits: u8, facts: &Facts) -> String {
    let suffix = if facts.flag_on("KAPLUMBA_SK2_CONTIG4") {
        "contig4_sk2"
    } else if facts.flag_on("KAPLUMBA_SK2_1COL") {
        "1col_sk2"
    } else {
        "2col_sk2"
    };
    format!("linear_gemv_ishizuki_k{bits}_colmajor_{suffix}")
}

/// `ishizuki_gemv_pipeline` (linear.rs:1508-1560) reduced to the name it resolves.
fn ishizuki_selector_name(bits: u8, perm_mode: u32, facts: &Facts) -> Option<String> {
    if facts.flag_on("KAPLUMBA_OWN_GEMV") {
        return None;
    }
    if facts.flag_on("KAPLUMBA_REPACKED_GEMV") || perm_mode == 1 {
        for name in [
            format!("linear_gemv_ishizuki_k{bits}_colmajor_contig4"),
            format!("linear_gemv_ishizuki_k{bits}_colmajor"),
        ] {
            if facts.inventory.contains(&name) {
                return Some(name);
            }
        }
    }
    if !facts.flag_on("KAPLUMBA_ISHIZUKI_3WORD") {
        let name = format!("linear_gemv_ishizuki_k{bits}_w2");
        if facts.inventory.contains(&name) {
            return Some(name);
        }
    }
    Some(format!("linear_gemv_ishizuki_k{bits}"))
}

// ---------------------------------------------------------------------------
// Traffic arithmetic.
// ---------------------------------------------------------------------------

fn split_factor(kernel: &str) -> u64 {
    if kernel.contains("_sk4") {
        4
    } else if kernel.contains("_sk2") {
        2
    } else {
        1
    }
}

struct Traffic {
    kernel: String,
    rows: u64,
    k: usize,
    bits: u8,
    groups: u64,
}

impl Traffic {
    fn weight_bytes(&self) -> u64 {
        weight_bytes(self.rows, self.k, self.bits)
    }

    /// Rows one threadgroup owns, derived rather than tabulated: the total
    /// number of 16-wide output tiles, divided by the threadgroups each split
    /// provides, times 16. `rows / (groups * split)` is wrong for split-K --
    /// every output row is deliberately covered `split` times.
    fn rows_per_tg(&self) -> u64 {
        let split = split_factor(&self.kernel);
        let total_tiles = self.rows / 16;
        let tiles_per_group = total_tiles * split / self.groups;
        assert_eq!(
            total_tiles % (self.groups / split),
            0,
            "{}: {total_tiles} tiles do not divide over {} threadgroups",
            self.kernel,
            self.groups / split
        );
        16 * tiles_per_group
    }

    fn bytes_per_tg(&self) -> u64 {
        let split = split_factor(&self.kernel);
        weight_bytes(self.rows_per_tg(), self.k / split as usize, self.bits)
    }

    /// f16 activation every threadgroup walks end to end.
    fn x_bytes(&self) -> u64 {
        self.k as u64 * 2
    }

    /// `x` traffic into L1 across the whole grid; all of it an L2 hit.
    fn x_l2_bytes(&self) -> u64 {
        self.groups * self.x_bytes()
    }

    fn x_share_of_l1(&self) -> f64 {
        let x = self.x_l2_bytes() as f64;
        let w = self.groups as f64 * self.bytes_per_tg() as f64;
        x / (x + w)
    }

    fn x_share_of_dram(&self) -> f64 {
        self.x_bytes() as f64 / (self.x_bytes() as f64 + self.weight_bytes() as f64)
    }

    fn waves(&self, resident: u64) -> f64 {
        self.groups as f64 / resident as f64
    }
}

fn traffic(kernel: &str, rows: u64, k: usize, bits: u8, groups: u64) -> Traffic {
    Traffic {
        kernel: kernel.to_string(),
        rows,
        k,
        bits,
        groups,
    }
}

// ===========================================================================
// 1. The production shapes are the ones the loader pins.
// ===========================================================================

#[test]
fn the_shapes_this_file_uses_are_pinned_by_the_loader() {
    // config.rs `validate_current_checkpoint` refuses anything else, so these are
    // invariants of the engine rather than tunables.
    for pin in [
        "self.num_hidden_layers != 60",
        "|| self.hidden_size != 5376",
        "|| self.intermediate_size != 21504",
        "|| self.vocab_size != 262144",
        "|| self.num_attention_heads != 32",
        "|| self.num_key_value_heads != 16",
        "|| self.num_global_key_value_heads != 4",
        "|| self.head_dim != 256",
        "|| self.global_head_dim != 512",
        "|| self.sliding_window != 1024",
    ] {
        assert!(
            CONFIG_RS.contains(pin),
            "config.rs no longer pins {pin:?}; re-derive the shape constants at \
             the top of this file"
        );
    }
    // model.rs:3364-3374: a full-attention layer takes global_head_dim and
    // num_global_key_value_heads, a sliding one takes head_dim / num_key_value_heads.
    assert!(MODEL_RS.contains("metadata.config.num_global_key_value_heads"));
    assert!(MODEL_RS.contains("metadata.config.num_key_value_heads"));
    assert_eq!(
        Q_OUT_SLIDING,
        32 * 256,
        "q_proj output on a sliding layer"
    );
    assert_eq!(Q_OUT_FULL, 32 * 512);
    assert_eq!(KV_OUT_SLIDING, 16 * 256);
    assert_eq!(KV_OUT_FULL, 4 * 512);
    // Every N the dispatch tree guards on is a multiple of 32, which is what
    // makes every `% 32` and `% 64` arm reachable rather than vacuous.
    for n in [INTERMEDIATE, Q_OUT_SLIDING, Q_OUT_FULL, KV_OUT_SLIDING, KV_OUT_FULL, HIDDEN] {
        assert_eq!(n % 32, 0, "{n} is not a multiple of 32");
    }
}

// ===========================================================================
// 2. Default-env selection, colmajor.
// ===========================================================================

fn colmajor_layer_facts() -> BTreeMap<&'static str, Facts> {
    let mut m = BTreeMap::new();
    m.insert(
        "gate_up",
        Facts::colmajor(3)
            .with("in_features", HIDDEN)
            .with("out_features_gate", INTERMEDIATE)
            .with("out_features_up", INTERMEDIATE)
            .with("tokens", 1),
    );
    m.insert(
        "qkv_sliding",
        Facts::colmajor(3)
            .with("in_features", HIDDEN)
            .with("out_features_a", Q_OUT_SLIDING)
            .with("out_features_b", KV_OUT_SLIDING)
            .with("out_features_c", KV_OUT_SLIDING)
            .with("tokens", 1),
    );
    m.insert(
        "qk_full",
        Facts::colmajor(3)
            .with("in_features", HIDDEN)
            .with("out_features_a", Q_OUT_FULL)
            .with("out_features_b", KV_OUT_FULL)
            .with("tokens", 1),
    );
    m
}

#[test]
fn colmajor_default_env_selects_two_col_everywhere() {
    let facts = &colmajor_layer_facts();

    let m = mlp_tree(LINEAR_RS).select(&facts["gate_up"]);
    assert_eq!(m.kernel, "linear_gemv_ishizuki_k3_fused2_co");
    assert_eq!(m.groups, 1344, "gate+up: 1344 threadgroups (one per tile_a)");
    assert_eq!(m.threads(), 1344 * 256);

    let f2 = fused2_tree(LINEAR_RS).select(&facts["qk_full"]);
    assert_eq!(f2.kernel, "linear_gemv_ishizuki_k3_fused2_2col");
    assert_eq!(f2.groups, ((Q_OUT_FULL / 16 + KV_OUT_FULL / 16) / 2) as u64);
    assert_eq!(f2.groups, 576);

    let f3 = fused3_tree(LINEAR_RS).select(&facts["qkv_sliding"]);
    assert_eq!(f3.kernel, "linear_gemv_ishizuki_k3_fused3_2col");
    assert_eq!(
        f3.groups,
        ((Q_OUT_SLIDING / 16 + KV_OUT_SLIDING / 16 + KV_OUT_SLIDING / 16) / 2) as u64
    );
    assert_eq!(f3.groups, 512);
}

#[test]
fn the_five_decode_kernels_are_exactly_three_layer_kernels_and_a_head() {
    let facts = &colmajor_layer_facts();
    // The sk2 arms' kernel name is built with `format!`, so it is resolved by
    // the transcribed helper and pinned by the test below rather than parsed.
    let sk2_pick = sk2_name(3, &Facts::colmajor(3));
    assert_eq!(sk2_pick, "linear_gemv_ishizuki_k3_colmajor_2col_sk2");
    let sk2_groups = ((HIDDEN / 16) / 2) as u64 * 2;
    assert_eq!(sk2_groups, 336);

    let gate_up_pick = mlp_tree(LINEAR_RS).select(&facts["gate_up"]).kernel;
    let qk_pick = fused2_tree(LINEAR_RS).select(&facts["qk_full"]).kernel;
    let qkv_pick = fused3_tree(LINEAR_RS).select(&facts["qkv_sliding"]).kernel;
    let mut names: Vec<&str> = vec![&gate_up_pick, &qk_pick, &qkv_pick, &sk2_pick];
    names.sort_unstable();
    names.dedup();
    assert_eq!(
        names,
        vec![
            "linear_gemv_ishizuki_k3_colmajor_2col_sk2",
            "linear_gemv_ishizuki_k3_fused2_2col",
            "linear_gemv_ishizuki_k3_fused2_co",
            "linear_gemv_ishizuki_k3_fused3_2col",
        ],
        "a fusion landed or an arm was reordered: the layer decode kernels changed"
    );
    for co in [
        "linear_gemv_ishizuki_k3_fused2_co4",
        "linear_gemv_ishizuki_k3_fused2_4col",
    ] {
        assert!(
            !names.contains(&co),
            "{co} is on the decode path; the shadowing analysis in the module \
             docs says none can be"
        );
    }

    // The head: 6 bpw, `permutation_mode == 0` even on a colmajor checkpoint,
    // so it never reaches a colmajor name.
    let head = ishizuki_selector_name(6, 0, &Facts::colmajor(3));
    assert_eq!(head.as_deref(), Some("linear_gemv_ishizuki_k6"));
    assert_eq!(
        ishizuki_selector_name(3, 1, &Facts::colmajor(3)).as_deref(),
        Some("linear_gemv_ishizuki_k3_colmajor_contig4"),
        "at perm_mode 1 the selector resolves to the colmajor name -- which is \
         why `encode_gemv_raw`'s `perm_mode == 1` arms must gate before it"
    );
    let _ = VOCAB;
}

// ===========================================================================
// 3. Reachability: which arm each kernel needs, and the shadowing.
// ===========================================================================

#[test]
fn fused2_co_is_shadowed_by_fused2_2col_in_the_mlp_entry_point() {
    let t = mlp_tree(LINEAR_RS);
    let base = colmajor_layer_facts()["gate_up"].clone();

    // Default: `_co`.
    assert_eq!(t.select(&base).kernel, "linear_gemv_ishizuki_k3_fused2_co");

    // Disabling `_co` reaches `_2col`.
    let without_co = base.clone().flag("KAPLUMBA_NO_FUSED2_CO");
    let pick = t.select(&without_co);
    assert_eq!(
        pick.kernel, "linear_gemv_ishizuki_k3_fused2_2col",
        "disabling the `_co` arm does not reach `_2col`; the arm order changed"
    );
    assert_eq!(pick.groups, (INTERMEDIATE as u64 / 16) * 2 / 2);

    // `_co4` sits inside the `_co` arm and needs its own flag on top.
    let with_co4 = base.clone().flag("KAPLUMBA_FUSED2_CO4");
    let pick = t.select(&with_co4);
    assert_eq!(pick.kernel, "linear_gemv_ishizuki_k3_fused2_co4");
    assert_eq!(pick.groups, INTERMEDIATE as u64 / 16 / 2);

    // `_4col` is the FIRST arm and needs an enabling flag.
    let pick = t.select(&base.clone().flag("KAPLUMBA_FUSED2_4COL"));
    assert_eq!(pick.kernel, "linear_gemv_ishizuki_k3_fused2_4col");
    assert_eq!(pick.groups, (INTERMEDIATE as u64 / 16) * 2 / 4);
    assert_eq!(pick.groups, 672);

    // Neither fallback is reachable on this shape either: 21504 % 64 == 0, so
    // the `else` arms are dead for gate/up but would fire for an odd N.
    assert_eq!(INTERMEDIATE % 64, 0);
    let odd = base
        .clone()
        .flag("KAPLUMBA_FUSED2_4COL")
        .flag("KAPLUMBA_NO_FUSED2_2COL");
    let t2 = mlp_tree(LINEAR_RS);
    let pick = t2.select(&odd);
    assert!(
        pick.kernel.starts_with("linear_gemv_ishizuki_k3_fused2"),
        "the `_co` arm's own guard must still cover us: got {}",
        pick.kernel
    );
}

#[test]
fn the_two_fused2_entry_points_disagree_about_co_and_2col_order() {
    mlp_tree(LINEAR_RS).assert_all_modelled();
    fused2_tree(LINEAR_RS).assert_all_modelled();
    fused3_tree(LINEAR_RS).assert_all_modelled();

    assert_eq!(
        mlp_tree(LINEAR_RS).arm_kernels(),
        vec![
            // env_flag(KAPLUMBA_USE_SK) early return
            "linear_gemv_ishizuki_k3_fused2_sk2",
            // the chain, in source order
            "linear_gemv_ishizuki_k3_fused2_4col",
            // the `_co` guard's own nested chain
            "linear_gemv_ishizuki_k3_fused2_co4",
            "linear_gemv_ishizuki_k3_fused2_co",
            "linear_gemv_ishizuki_k3_fused2",
            "linear_gemv_ishizuki_k3_fused2_co",
            "linear_gemv_ishizuki_k3_fused2",
            "linear_gemv_ishizuki_k3_fused2_2col",
            // the chain's terminal `else`
            "linear_gemv_ishizuki_k3_fused2",
        ],
        "the MLP chain's arm order changed; re-derive the shadowing analysis"
    );
    assert_eq!(
        fused2_tree(LINEAR_RS).arm_kernels(),
        vec![
            "linear_gemv_ishizuki_k3_fused2_sk2",
            "linear_gemv_ishizuki_k3_fused2_co4",
            "linear_gemv_ishizuki_k3_fused2_co",
            "linear_gemv_ishizuki_k3_fused2",
            "linear_gemv_ishizuki_k3_fused2_co",
            "linear_gemv_ishizuki_k3_fused2",
            "linear_gemv_ishizuki_k3_fused2_2col",
            "linear_gemv_ishizuki_k3_fused2",
            "linear_gemv_ishizuki_k3_fused2",
        ],
        "the q+k chain puts `_co` BEFORE `_2col`, the opposite of the MLP chain. \
         Harmless today only because q != k; if they ever matched, the same shape \
         would select different kernels on the two paths."
    );
    // `fused3` has no `_co` arm at all, and its terminal `else` is reachable only
    // when some N is not a multiple of 32 -- which never happens here.
    assert_eq!(
        fused3_tree(LINEAR_RS).arm_kernels(),
        vec![
            "linear_gemv_ishizuki_k3_fused3_sk2",
            "linear_gemv_ishizuki_k3_fused3_2col",
            "linear_gemv_ishizuki_k3_fused3",
            "linear_gemv_ishizuki_k3_fused3",
        ]
    );
}

#[test]
fn q_and_k_are_never_equal_so_co_is_unreachable_on_either_fused2_path() {
    for (a, b) in [
        (Q_OUT_SLIDING, KV_OUT_SLIDING),
        (Q_OUT_FULL, KV_OUT_FULL),
    ] {
        assert_ne!(
            a, b,
            "q and k became equal, which makes the `_co` arm of encode_gemv_fused2 \
             reachable and changes what the q+k path dispatches"
        );
    }
    assert_eq!(qkv_path(true, true, 1), QkvPath::Fused3);
    assert_eq!(qkv_path(false, true, 1), QkvPath::Fused2PlusCopy);
    assert_eq!(qkv_path(true, false, 1), QkvPath::Separate);
    assert_eq!(qkv_path(true, true, 0), QkvPath::Separate);
}

#[test]
fn the_qkv_census_is_50_fused3_layers_and_10_fused2_plus_copy() {
    // config.json `layer_types` marks every 6th layer full-attention, and
    // model.rs:3439 loads a V projection only for the *non*-full layers.
    let sliding = (0..60).filter(|i| i % 6 != 5).count();
    let full = 60 - sliding;
    assert_eq!((sliding, full), (SLIDING_LAYERS, FULL_LAYERS));
    let mut paths = BTreeMap::new();
    *paths.entry(QkvPath::Fused3).or_insert(0usize) += sliding;
    *paths.entry(QkvPath::Fused2PlusCopy).or_insert(0usize) += full;
    assert_eq!(paths[&QkvPath::Fused3], 50);
    assert_eq!(paths[&QkvPath::Fused2PlusCopy], 10);
    assert!(
        MODEL_RS.contains("encode_copy(encoder, &scratch.k, &scratch.v, kv_elements)"),
        "the k -> v copy at model.rs:1588 moved; the 10 full-attention layers no \
         longer skip the V GEMV"
    );
    assert!(
        MODEL_RS.contains("v_proj: if full {"),
        "the loader's V-projection rule moved; the 50/10 split needs re-deriving"
    );
}

#[test]
fn every_kernel_the_fused_chains_name_exists_in_the_shaders() {
    for tree_fn in [
        mlp_tree as fn(&str) -> Tree,
        fused2_tree,
        fused3_tree,
    ] {
        for arm in tree_fn(LINEAR_RS).arms {
            let name = arm.kernel();
            assert!(
                ishizuki_inventory().contains(&name),
                "the dispatch names {name}, which no .metal file defines \
                 (arm at linear.rs:{})",
                arm.line
            );
        }
    }
}

// ===========================================================================
// 4. Split-K: opt-out, not opt-in; and the SK4 / cross-layer trap.
// ===========================================================================

#[test]
fn split_k_is_an_opt_out_so_down_proj_and_o_proj_get_sk2_by_default() {
    let facts = Facts::colmajor(3).with("out_features", HIDDEN).with("tokens", 1);
    assert_eq!(split_k_path(&facts, HIDDEN, false), "sk2");
    assert_eq!(split_k_path(&facts, HIDDEN, true), "sk2");

    // Every shape precondition holds, so the opt-out is the only thing that can
    // turn it off.
    assert_eq!(facts.perm_mode, 1);
    assert_eq!(HIDDEN % 32, 0);
    assert_eq!(facts.bits, 3);

    assert_eq!(
        split_k_path(&facts.clone().flag("KAPLUMBA_SK4"), HIDDEN, false),
        "sk4"
    );
    assert_eq!(
        split_k_path(&facts.clone().flag("KAPLUMBA_O_SK4"), HIDDEN, true),
        "sk4"
    );
    for (flag, o) in [
        ("KAPLUMBA_NO_SK2", false),
        ("KAPLUMBA_D_NOSK", false),
        ("KAPLUMBA_O_NOSK", true),
    ] {
        assert_eq!(
            split_k_path(&facts.clone().flag(flag), HIDDEN, o),
            "no-split",
            "{flag} did not turn split-K off for {}",
            if o { "o_proj" } else { "down_proj" }
        );
    }

    // The sk2/sk4 entry points' own `perm_mode == 1` guard.
    let row = Facts::row_major(3).with("out_features", HIDDEN).with("tokens", 1);
    for header in [
        "pub fn encode_gemv_raw_sk2(",
        "pub fn encode_gemv_raw_sk4(",
    ] {
        let t = tree(LINEAR_RS, header);
        let arm = t
            .modelled()
            .first()
            .copied()
            .unwrap_or_else(|| panic!("{header}: no modelled arm"));
        assert!(
            arm.cond.contains("perm_mode == 1"),
            "{header}'s first arm is {:?}, no longer perm-gated",
            arm.cond
        );
        assert!(
            eval_bool(&arm.cond, &row) == Some(false),
            "{header} fires at perm_mode 0"
        );
        assert_eq!(eval_bool(&arm.cond, &facts), Some(true));
    }

    // Which of the three sk2 names the default picks, and that the two A/B
    // names still exist in the shaders (notes 16.6 records one of them being
    // referenced while absent).
    assert_eq!(
        sk2_name(3, &Facts::colmajor(3)),
        "linear_gemv_ishizuki_k3_colmajor_2col_sk2"
    );
    assert_eq!(
        sk2_name(3, &Facts::colmajor(3).flag("KAPLUMBA_SK2_CONTIG4")),
        "linear_gemv_ishizuki_k3_colmajor_contig4_sk2"
    );
    assert_eq!(
        sk2_name(3, &Facts::colmajor(3).flag("KAPLUMBA_SK2_1COL")),
        "linear_gemv_ishizuki_k3_colmajor_1col_sk2"
    );
    for flag in ["KAPLUMBA_SK2_CONTIG4", "KAPLUMBA_SK2_1COL"] {
        assert!(
            ishizuki_inventory().contains(&sk2_name(3, &Facts::colmajor(3).flag(flag))),
            "{flag} names a kernel no .metal file defines; setting it would crash in \
             newFunctionWithName"
        );
    }
}

#[test]
fn sk4_silently_disables_the_cross_layer_fusion() {
    let facts = Facts::colmajor(3).with("out_features", HIDDEN).with("tokens", 1);
    assert!(
        cross_layer_fusion(&facts, true, HIDDEN),
        "T3 cross-layer fusion must apply by default"
    );
    // model.rs:1284 puts `!env_flag("KAPLUMBA_SK4")` inside the predicate, so an
    // A/B of split-K=4 also deletes a dispatch fusion. Recorded because the
    // notes' "Split-K=4 measured identical" number cannot be attributed without
    // knowing this.
    assert!(
        !cross_layer_fusion(&facts.clone().flag("KAPLUMBA_SK4"), true, HIDDEN),
        "KAPLUMBA_SK4 no longer disables cross-layer fusion"
    );
    for pin in [
        "&& !env_flag(\"KAPLUMBA_SK4\")",
        "&& !env_flag(\"KAPLUMBA_NO_SK2\")",
        "&& !env_flag(\"KAPLUMBA_D_NOSK\")",
    ] {
        assert!(
            MODEL_RS.contains(pin),
            "cross_layer_fusion_applies no longer contains {pin:?}"
        );
    }
}

// ===========================================================================
// 5. Row-major: every colmajor arm falls out.
// ===========================================================================

#[test]
fn a_row_major_checkpoint_selects_no_colmajor_kernel_anywhere() {
    // `Gxma-repacked` is `layout: colmajor`; `Gxma` (the dir `Gxma/Gxma`) has no
    // `layout` key, so model.rs:3525 leaves every `permutation_mode` at 0.
    let row = Facts::row_major(3).with("tokens", 1);
    let col = Facts::colmajor(3).with("tokens", 1);
    assert_eq!(row.perm_mode, 0);
    assert_eq!(col.perm_mode, 1);
    assert!(
        MODEL_RS.contains("metadata.quantization.layout.as_deref() == Some(\"colmajor\")"),
        "the colmajor loader gate moved"
    );

    // model.rs:2144 gates the whole fused MLP block on permutation_mode == 1, so
    // a row-major checkpoint does not even reach `encode_gemv_fused2_mlp`: it
    // issues three separate `encode_linear` calls plus a separate
    // `encode_gated_gelu` (model.rs:2381-2411).
    assert_eq!(qkv_path(true, true, 0), QkvPath::Separate);
    assert!(MODEL_RS.contains("encode_gated_gelu("));
    assert!(!cross_layer_fusion(
        &row.clone().with("out_features", HIDDEN),
        true,
        HIDDEN
    ));
    assert_eq!(split_k_path(&row.with("out_features", HIDDEN), HIDDEN, false), "no-split");

    // `encode_gemv_raw`'s two colmajor arms are `perm_mode == 1`-gated, so both
    // are false and the selection falls through to `ishizuki_gemv_pipeline`,
    // which at perm_mode 0 with no `KAPLUMBA_REPACKED_GEMV` takes the `_w2` slot.
    let raw = gemv_raw_tree(LINEAR_RS);
    let raw_facts = |perm: u32| {
        let mut f = Facts::colmajor(3);
        f.perm_mode = perm;
        f.with("out_features", HIDDEN).with("tokens", 1)
    };
    let verdicts: Vec<bool> = raw
        .modelled()
        .iter()
        .map(|a| eval_bool(&a.cond, &raw_facts(0)).unwrap_or(false))
        .collect();
    assert_eq!(verdicts.len(), 2, "the two perm-gated colmajor arms");
    assert_eq!(
        &verdicts[..2],
        &[false, false],
        "a colmajor arm fires at perm_mode 0; the row-major column of the table \
         is wrong"
    );
    let at_one: Vec<bool> = raw
        .modelled()
        .iter()
        .map(|a| eval_bool(&a.cond, &raw_facts(1)).unwrap_or(false))
        .collect();
    assert!(
        !at_one[0],
        "the 4col arm fired with KAPLUMBA_4COL unset"
    );
    assert!(at_one[1], "the 2col arm must fire at perm_mode 1");

    // And the selector, resolved from the parsed inventory.
    assert_eq!(
        ishizuki_selector_name(3, 0, &Facts::colmajor(3)).as_deref(),
        Some("linear_gemv_ishizuki_k3_w2"),
        "a row-major 3 bpw projection resolves to the 2-word window kernel"
    );
}

#[test]
fn the_unmodelled_arms_of_encode_gemv_raw_are_pinned() {
    let raw = gemv_raw_tree(LINEAR_RS);
    // Two arms are decidable: the `KAPLUMBA_4COL` and the default 2col colmajor
    // arms. The `ishizuki_gemv_pipeline` fallback, the `decode_gemv_pipeline`
    // chain and the EXL3 `else` all go to the skip list, and the counts are
    // pinned so a new top-level arm is noticed rather than absorbed.
    assert_eq!(raw.skipped.len(), 3, "{:?}", raw.skipped);
    let modelled = raw.modelled();
    assert_eq!(raw.arms.len(), modelled.len(), "{:?}", raw.skipped);
    assert_eq!(modelled.len(), 2);
    for arm in modelled {
        assert!(
            arm.cond.contains("perm_mode == 1"),
            "a modelled arm of encode_gemv_raw is not perm-gated: {:?}",
            arm.cond
        );
    }
}

// ===========================================================================
// 6. The traffic arithmetic.
// ===========================================================================

#[test]
fn bytes_per_threadgroup_table() {
    let rows: Vec<(&str, Traffic)> = vec![
        (
            "gate+up fused2_2col (SHIPPED)",
            traffic("fused2_2col", 2 * INTERMEDIATE as u64, HIDDEN, 3, 1344),
        ),
        (
            "gate+up fused2_co (SHADOWED)",
            traffic("fused2_co", 2 * INTERMEDIATE as u64, HIDDEN, 3, 1344),
        ),
        (
            "gate+up fused2_co4 (measured -0.19)",
            traffic("fused2_co4", 2 * INTERMEDIATE as u64, HIDDEN, 3, 672),
        ),
        (
            "gate+up fused2_4col (NEVER RUN)",
            traffic("fused2_4col", 2 * INTERMEDIATE as u64, HIDDEN, 3, 672),
        ),
        (
            "gate+up fused2 (1 col/tg)",
            traffic("fused2", 2 * INTERMEDIATE as u64, HIDDEN, 3, 2688),
        ),
        (
            "qkv fused3_2col (SHIPPED)",
            traffic(
                "fused3_2col",
                (Q_OUT_SLIDING + 2 * KV_OUT_SLIDING) as u64,
                HIDDEN,
                3,
                512,
            ),
        ),
        (
            "qk fused2_2col, full layer",
            traffic(
                "fused2_2col",
                (Q_OUT_FULL + KV_OUT_FULL) as u64,
                HIDDEN,
                3,
                576,
            ),
        ),
        (
            "down colmajor_2col_sk2 (SHIPPED)",
            traffic("colmajor_2col_sk2", HIDDEN as u64, INTERMEDIATE, 3, 336),
        ),
        (
            "o_proj colmajor_2col_sk2 (SHIPPED)",
            traffic("colmajor_2col_sk2", HIDDEN as u64, Q_OUT_SLIDING, 3, 336),
        ),
    ];

    println!(
        "\n{:<36} {:>5} {:>5} {:>11} {:>9} {:>9} {:>10} {:>7} {:>8}",
        "kernel",
        "K",
        "tg",
        "weights",
        "B/tg",
        "waves160",
        "x@L1",
        "x@DRAM",
        "rows/tg"
    );
    for (label, t) in &rows {
        println!(
            "{:<36} {:>5} {:>5} {:>11} {:>9} {:>9.2} {:>9.1}% {:>6.3}% {:>8}",
            label,
            t.k,
            t.groups,
            t.weight_bytes(),
            t.bytes_per_tg(),
            t.waves(160),
            t.x_share_of_l1() * 100.0,
            t.x_share_of_dram() * 100.0,
            t.rows_per_tg(),
        );
    }

    let per_token: u64 = SLIDING_LAYERS as u64 * 33_030_144 // q 8192 + k 4096 + v 4096
        + FULL_LAYERS as u64 * 37_158_912 // q 16384 + k 2048 (V == K)
        + SLIDING_LAYERS as u64 * 16_515_072 // o_proj in 8192
        + FULL_LAYERS as u64 * 33_030_144 // o_proj in 16384
        + 60 * 86_704_128 // gate + up
        + 60 * 43_352_064 // down
        + 1_056_808_960; // lm_head at 6 bpw
    println!("\nweights streamed per generated token: {per_token} B = {:.3} GB", per_token as f64 / 1e9);
    println!("  qkv     {:.3} GB", (SLIDING_LAYERS as u64 * 33_030_144 + FULL_LAYERS as u64 * 37_158_912) as f64 / 1e9);
    println!("  o_proj  {:.3} GB", (SLIDING_LAYERS as u64 * 16_515_072 + FULL_LAYERS as u64 * 33_030_144) as f64 / 1e9);
    println!("  gate+up {:.3} GB", (60u64 * 86_704_128) as f64 / 1e9);
    println!("  down    {:.3} GB", (60u64 * 43_352_064) as f64 / 1e9);
    println!("  head    {:.3} GB", (1_056_808_960u64) as f64 / 1e9);
    println!(
        "  wire-bound floor at 273 GB/s: {:.1} ms -> {:.2} tok/s",
        per_token as f64 / 273e9 * 1e3,
        273e9 / per_token as f64
    );

    // FINDING: notes 16.4 quotes 11.834 GB/token. The real figure is 12.039 GB,
    // because that table prices all 60 layers at the *sliding* geometry (q 8192,
    // k=v 4096, o_proj in 8192) while 10 of the 60 are full-attention with
    // global_head_dim 512 and 4 KV heads (q 16384, k 2048, o_proj in 16384).
    // Consequences, all downward: the wire-bound floor is 44.1 ms / 22.7 tok/s,
    // not 39.6 / 25.3; 20 tok/s needs 240.8 GB/s sustained, not 236.7; and the
    // 203 GB/s GEMV-peak ceiling is 59.3 ms / 16.9 tok/s, not 58.3 / 17.2. The
    // 18.2 table has the same defect -- it credits o_proj with 0.991 GB when the
    // truth is 1.156, so o_proj's implied bandwidth is ~174 GB/s, not the
    // quoted 150.
    assert_eq!(per_token, 12_039_331_840);
    assert!(
        (per_token as f64 / 1e9 - 11.834).abs() > 0.1,
        "the byte budget converged on the notes' 11.834 GB; check whether the 10 \
         full-attention layers still have global_head_dim 512 / 4 KV heads"
    );

    // The 96 B tile is the unit the whole table is built from.
    assert_eq!(BYTES_PER_TILE_3BPW, 16 * 16 * 3 / 8);
    assert_eq!(
        BYTES_PER_TILE_3BPW * (INTERMEDIATE / 16) as u64 * (HIDDEN / 16) as u64,
        weight_bytes(INTERMEDIATE as u64, HIDDEN, 3),
        "the trellis tile size and the weights-per-matrix size disagree"
    );

    let gate_up = &rows[0].1;
    assert_eq!(gate_up.bytes_per_tg(), 64_512);
    assert_eq!(gate_up.x_bytes(), 10_752);
    assert_eq!(gate_up.rows_per_tg(), 32);
    assert_eq!(gate_up.weight_bytes(), 86_704_128);
    assert!(
        gate_up.x_share_of_dram() < 0.0002,
        "x is {}% of DRAM traffic; fusion cannot buy a traffic reduction",
        gate_up.x_share_of_dram() * 100.0
    );
    assert!(
        gate_up.x_share_of_l1() > 0.10,
        "x is {}% of L1 traffic: it is cache-resident and re-read per threadgroup, \
         which is the only place fusion could have saved anything",
        gate_up.x_share_of_l1() * 100.0
    );
    // `_co` and `_2col` are identical on every axis fusion could have helped.
    let co = &rows[1].1;
    assert_eq!(co.groups, gate_up.groups);
    assert_eq!(co.bytes_per_tg(), gate_up.bytes_per_tg());
    assert_eq!(co.x_l2_bytes(), gate_up.x_l2_bytes());
    // `_4col` doubles bytes/threadgroup without the second DRAM stream.
    let four = &rows[3].1;
    let co4 = &rows[2].1;
    assert_eq!(four.bytes_per_tg(), 129_024);
    assert_eq!(four.bytes_per_tg(), co4.bytes_per_tg());
    assert_eq!(four.groups, co4.groups);
    assert_eq!(
        four.bytes_per_tg(),
        rows[7].1.bytes_per_tg(),
        "`fused2_4col` puts gate+up in the same bytes-per-threadgroup regime as \
         `down_proj`'s shipped split-K2, the shape that measures 199.3 GB/s"
    );
    // `o_proj` is the worst shape in the model and the notes call it out.
    assert_eq!(rows[8].1.bytes_per_tg(), 49_152);
}

#[test]
fn fusion_does_not_reduce_the_weight_traffic() {
    // The load-bearing negative result. Every fused variant reads exactly the
    // same packed bytes: the weight matrices ARE the traffic, and no
    // arrangement of threadgroups changes how many weight bytes exist.
    let n = 2 * INTERMEDIATE as u64;
    let mut weights = BTreeSet::new();
    for (kernel, groups) in [
        ("fused2", 2688u64),
        ("fused2_2col", 1344),
        ("fused2_co", 1344),
        ("fused2_co4", 672),
        ("fused2_4col", 672),
    ] {
        let t = traffic(kernel, n, HIDDEN, 3, groups);
        assert_eq!(
            t.groups * t.bytes_per_tg(),
            weight_bytes(n, HIDDEN, 3),
            "{kernel} does not read every weight byte exactly once"
        );
        weights.insert(t.weight_bytes());
    }
    assert_eq!(weights.len(), 1, "variants disagree on weight bytes");
    assert_eq!(weights.iter().next().copied().unwrap(), 86_704_128);
    // The raw outputs the fused epilogue writes are 0.2% of the weight traffic.
    let raw_out = 2 * INTERMEDIATE as u64 * 4; // two f32 raws
    let gate_up_weights = weight_bytes(2 * INTERMEDIATE as u64, HIDDEN, 3);
    assert!(
        (raw_out as f64) / (gate_up_weights as f64) < 0.005,
        "raw_gate + raw_up is {raw_out} B against {gate_up_weights} B of weights; \
         writing both from one threadgroup stopped being negligible"
    );
}

// ===========================================================================
// 7. Negative controls: the evaluator must be reading the source.
// ===========================================================================

#[test]
fn inverting_the_co_opt_out_moves_the_selection_to_2col() {
    let t = mlp_tree(LINEAR_RS);
    let facts = colmajor_layer_facts()["gate_up"].clone();
    assert_eq!(t.select(&facts).kernel, "linear_gemv_ishizuki_k3_fused2_co");

    // The anchor includes `out_features_gate`, so the mutation lands in
    // `encode_gemv_fused2_mlp` and not in the earlier `encode_gemv_fused2`.
    let mutated = LINEAR_RS.replacen(
        "out_features_gate == out_features_up && !env_flag(\"KAPLUMBA_NO_FUSED2_CO\")",
        "out_features_gate == out_features_up && env_flag(\"KAPLUMBA_NO_FUSED2_CO\")",
        1,
    );
    assert_ne!(
        mutated, LINEAR_RS,
        "the mutation did not apply; linear.rs moved and this control is now vacuous"
    );
    assert_eq!(
        mlp_tree(&mutated).select(&facts).kernel,
        "linear_gemv_ishizuki_k3_fused2_2col",
        "inverting the opt-out did not move the selection, so the evaluator is \
         not reading the guard"
    );
}

#[test]
fn a_co_arm_first_would_win_so_the_order_is_load_bearing() {
    let t = mlp_tree(LINEAR_RS);
    let co = t
        .arms
        .iter()
        .filter(|a| a.kernel() == "linear_gemv_ishizuki_k3_fused2_co")
        .find(|a| !a.cond.contains("KAPLUMBA_FUSED2_CO4"))
        .expect("the outer `_co` arm is gone from encode_gemv_fused2_mlp");
    assert_eq!(
        t.arms
            .iter()
            .filter(|a| a.kernel() == "linear_gemv_ishizuki_k3_fused2_co")
            .count(),
        2,
        "there are two `_co` arms (the outer one and the CO4 sub-chain's); the \
         count changed"
    );
    // The `_co` arm's own guard holds on this shape, so only its position in the
    // chain keeps it dead.
    assert_eq!(
        Facts::colmajor(3)
            .with("out_features_gate", INTERMEDIATE)
            .with("out_features_up", INTERMEDIATE)
            .eval("out_features_gate == out_features_up"),
        Some(true)
    );
    assert_eq!(
        Facts::colmajor(3)
            .with("out_features_gate", INTERMEDIATE)
            .with("out_features_up", INTERMEDIATE)
            .eval("!env_flag(\"KAPLUMBA_NO_FUSED2_CO\")"),
        Some(true)
    );
    // And the synthesised conditions carry the parent guard, so a reordered arm
    // would be evaluated as false rather than silently as true.
    assert!(
        co.cond.contains("out_features_gate == out_features_up")
            && co.cond.contains("KAPLUMBA_NO_FUSED2_CO"),
        "the `_co` arm's synthesised condition lost its parent guard: {:?}",
        co.cond
    );
    assert_eq!(
        Facts::colmajor(3)
            .with("out_features_gate", INTERMEDIATE)
            .with("out_features_up", INTERMEDIATE)
            .eval(&co.cond),
        Some(true),
        "the `_co` arm would win if it were reached"
    );
    assert_eq!(
        co.groups.as_deref(),
        Some("tiles_a"),
        "`_co` would give the same 1344 threadgroups as `_2col`; the difference \
         is the two-stream interleaving, not the grid"
    );
}

#[test]
fn an_unmodelled_guard_is_skipped_loudly_not_ignored() {
    let facts = colmajor_layer_facts()["gate_up"].clone();
    // The anchor includes `out_features_gate`, so the mutation lands in
    // `encode_gemv_fused2_mlp` rather than in the earlier `encode_gemv_fused2`.
    let mutated = LINEAR_RS.replacen(
        "!env_flag(\"KAPLUMBA_NO_FUSED2_2COL\")\n            && out_features_gate % 32 == 0",
        "!env_flag(\"KAPLUMBA_NO_FUSED2_2COL\")\n            && scratch_occupancy_ok()\n            && out_features_gate % 32 == 0",
        1,
    );
    assert_ne!(
        mutated, LINEAR_RS,
        "the mutation did not apply; the guard's text moved"
    );
    let t = mlp_tree(&mutated);
    assert!(
        t.skipped
            .iter()
            .any(|s| s.contains("scratch_occupancy_ok")),
        "a guard containing an unknown function was accepted silently; skipped={:?}",
        t.skipped
    );
    // With the skip honoured the arm is not claimable, and `_co` becomes the
    // answer -- which is the correct reading of "an arm we cannot evaluate is
    // not an arm we can claim to have ruled out".
    assert_eq!(t.select(&facts).kernel, "linear_gemv_ishizuki_k3_fused2_co");
}

#[test]
fn a_mutated_kernel_name_makes_the_pipeline_lookup_fail_closed() {
    // `.is_some()` is answered from the parsed inventory, so renaming a kernel
    // without updating its guard changes the selection rather than passing.
    let facts = colmajor_layer_facts()["gate_up"].clone();
    assert_eq!(
        mlp_tree(LINEAR_RS).select(&facts).kernel,
        "linear_gemv_ishizuki_k3_fused2_co"
    );
    let mutated = LINEAR_RS.replacen(
        "out_features_gate == out_features_up && !env_flag(\"KAPLUMBA_NO_FUSED2_CO\")",
        "out_features_gate == out_features_up && !env_flag(\"KAPLUMBA_NO_FUSED2_CO\") && self.ishizuki_gemv_pipeline_by_name(\"linear_gemv_ishizuki_k3_fused2_co_RENAMED\").is_some()",
        1,
    );
    assert_ne!(mutated, LINEAR_RS);
    assert_eq!(
        mlp_tree(&mutated).select(&facts).kernel,
        "linear_gemv_ishizuki_k3_fused2_2col",
        "an `.is_some()` guard naming a kernel no shader defines must fail closed"
    );
}

// ===========================================================================
// 8. The recommendation, as a test so it cannot rot.
// ===========================================================================

/// `KAPLUMBA_FUSED2_4COL` is the one cell in the MLP dispatch tree that has
/// never been measured, and it is the cell that discriminates the two
/// explanations the notes offer for `fused2_co4`'s loss.
///
/// * `fused2_co4` (notes 13): 129,024 B per threadgroup over 672 threadgroups,
///   measured 15.46 against `fused2_2col`'s 15.65 tok/s, attributed to
///   "cache-line contention between two 345 MB matrices".
/// * `fused2_4col`: the same 129,024 B per threadgroup over the same 672
///   threadgroups, but four *consecutive* tiles of ONE matrix.
///
/// If bytes-per-threadgroup is what matters, `fused2_4col` should match or beat
/// `_2col`. If the second DRAM stream is what hurts, it should reproduce `co4`'s
/// loss. Notes 16.3 measured 199.3 GB/s at 129,024 B/threadgroup against ~175 at
/// 64,512, so the upside is gate_up's 28.6 ms becoming ~26.4 ms: -2.2 ms on a
/// 63.1 ms step = +0.57 tok/s, with `co4`'s outcome as the downside.
#[test]
fn fused2_4col_is_reachable_and_is_the_only_unmeasured_mlp_cell() {
    let t = mlp_tree(LINEAR_RS);
    let facts = colmajor_layer_facts()["gate_up"].clone();

    let arm = t
        .arms
        .iter()
        .find(|a| a.kernel() == "linear_gemv_ishizuki_k3_fused2_4col")
        .expect("the 4col arm is gone from encode_gemv_fused2_mlp");
    assert!(
        arm.cond.starts_with("env_flag(\"KAPLUMBA_FUSED2_4COL\")"),
        "the 4col arm is no longer first and flag-gated: {:?}",
        arm.cond
    );
    let pick = t.select(&facts.clone().flag("KAPLUMBA_FUSED2_4COL"));
    assert_eq!(pick.kernel, "linear_gemv_ishizuki_k3_fused2_4col");
    assert_eq!(pick.groups, 672);
    assert_eq!(pick.threads(), 672 * 256);
    assert!(
        ishizuki_inventory().contains("linear_gemv_ishizuki_k3_fused2_4col"),
        "the 4col arm names a kernel no shader defines; this is a crash, not a \
         measurement"
    );

    // Every other cell in the MLP chain has a recorded measurement. Stated as a
    // test so adding a new cell forces a decision.
    for kernel in [
        "fused2_2col",
        "fused2_co4",
        "fused2_co",
        "fused2",
        "fused2_sk2",
    ] {
        assert!(
            t.arms
                .iter()
                .any(|a| a.kernel() == format!("linear_gemv_ishizuki_k3_{kernel}")),
            "{kernel} left the chain; re-audit the reachability table"
        );
    }

    // Promotion must not be done blind: notes 16 records a previous agent making
    // a variant the default without benchmarking it, and notes 16.5 records three
    // changes that won in isolation and lost in production.
    assert!(
        !arm.cond.contains("KAPLUMBA_NO_FUSED2_4COL"),
        "the 4col arm flipped to an opt-out; that decision needs an interleaved A/B \
         against the 15.85 baseline first"
    );
}
