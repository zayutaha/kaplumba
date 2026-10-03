//! Column-tiling register-budget probe (spec/decode-ceiling, K03).
//!
//! WHAT THIS IS
//!
//! K03 asks why `linear_gemv_ishizuki_k3_colmajor_4col` cost 4.2% tok/s while
//! `colmajor_2col` was the baseline, and whether the `_co` / `_co4` "co-compute"
//! family is a cheaper way to get two output columns out of one input tile.
//! Both are questions about the *live state of the innermost K-loop*, and that
//! state is a property of the source, not of a benchmark.
//!
//! So this probe reads `crates/kernels/src/linear_repacked.metal` and counts it.
//! Nothing here is transcribed from a reading of the file: every number is
//! produced by scanning the kernel body, so if someone edits a kernel the table
//! moves with it and the two structural assertions below fail loudly.
//!
//! WHY COUNTS, NOT REGISTERS
//!
//! Metal's public API does not expose a register count for a
//! `ComputePipelineState` (there is no `maxRegistersPerThread` equivalent of
//! `maxTotalThreadsPerThreadgroup`), so an exact occupancy number is not
//! obtainable from inside a test. What *is* obtainable, and what the 4col/2col
//! difference actually turns on, is the count of simultaneously-live 32-bit
//! values in the unrolled K-loop:
//!
//!   accumulators      1 float per (column x output-half)
//!   activation scalars 4 floats per K sub-step   (p00..p03 per tile)
//!   weight pointers   1 x 64-bit per K sub-step x column
//!   decode buffer     1 x float[8], shared and pipelined since T3
//!
//! These are lower bounds on the real allocation (address arithmetic, the
//! 64-bit shift chain inside `extract_v_mcg_32`, and the scheduler's own
//! renaming all add more), which is why the comparison between variants is the
//! useful output and the absolute figure is not.
//!
//! WHAT IT ASSERTS
//!
//!   A1. `_co` and `_2col` have *identical* live-state counts. The "co-compute"
//!       family is therefore not a register-saving variant of 2-column tiling;
//!       it is the same loop shape with a different choice of which two columns
//!       a threadgroup owns. Any speed difference between them is memory-layout,
//!       not occupancy.
//!   A2. `_co4` and `_4col` have *identical* live-state counts, so `_co4`
//!       inherits 4col's cost exactly and cannot be a way around it.
//!
//! Run with:
//!   cargo test --release --test probe_coltile_registers -- --nocapture
//!
//! No Metal device is required and no model is loaded; this is pure text
//! analysis plus arithmetic.

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

/// Replace comment bodies with spaces, preserving byte length, so the brace
/// matcher below cannot be led astray by a brace inside a comment.
fn strip_comments(src: &str) -> String {
    let bytes = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
            while i < bytes.len() && bytes[i] != b'\n' {
                out.push(' ');
                i += 1;
            }
        } else if bytes[i] == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
            while i < bytes.len() && !(bytes[i] == b'*' && i + 1 < bytes.len() && bytes[i + 1] == b'/')
            {
                if bytes[i] != b'\n' {
                    out.push(' ');
                } else {
                    out.push('\n');
                }
                i += 1;
            }
            for _ in 0..2 {
                if i < bytes.len() {
                    out.push(' ');
                    i += 1;
                }
            }
        } else {
            let ch = src[i..].chars().next().expect("char boundary");
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    out
}

/// Body of `kernel void <name>(...)`, brace-matched from the opening `{` that
/// follows the signature's closing `)`.
fn kernel_body<'a>(src: &'a str, name: &str) -> &'a str {
    let needle = format!("kernel void {name}(");
    let start = src
        .find(&needle)
        .unwrap_or_else(|| panic!("kernel {name} not found in linear_repacked.metal"));
    let open = src[start..]
        .find('{')
        .unwrap_or_else(|| panic!("no opening brace for {name}"))
        + start;
    let bytes = src.as_bytes();
    let mut depth = 0i32;
    let mut i = open;
    while i < bytes.len() {
        match bytes[i] {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &src[open + 1..i];
                }
            }
            _ => {}
        }
        i += 1;
    }
    panic!("unbalanced braces for {name}");
}

/// The `for (; tk < k_unroll_end; ...)` body -- the fully-unrolled hot loop that
/// sets the register budget.
fn unrolled_loop(src: &str, name: &str) -> String {
    let body = kernel_body(src, name);
    let anchor = body
        .find("for (; tk < k_unroll_end")
        .unwrap_or_else(|| panic!("{name} has no unrolled K-loop anchor"));
    let rest = &body[anchor..];
    let open = rest.find('{').expect("loop brace");
    let bytes = rest.as_bytes();
    let mut depth = 0i32;
    let mut i = open;
    while i < bytes.len() {
        match bytes[i] {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return rest[open + 1..i].to_string();
                }
            }
            _ => {}
        }
        i += 1;
    }
    panic!("unbalanced braces in {name} unrolled loop");
}

/// Distinct identifiers matching a prefix inside a block. Crude on purpose: the
/// Metal in question uses single-token names (`p00`, `t0_1`, `acc0_0`), so a
/// prefix scan is exact here and does not need a parser.
fn count_identifiers(block: &str, prefix: &str) -> usize {
    let mut names: Vec<&str> = Vec::new();
    let bytes = block.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if (bytes[i] as char).is_ascii_alphabetic() || bytes[i] == b'_' {
            let start = i;
            while i < bytes.len()
                && ((bytes[i] as char).is_ascii_alphanumeric() || bytes[i] == b'_')
            {
                i += 1;
            }
            let word = &block[start..i];
            if word.starts_with(prefix) && !names.contains(&word) {
                names.push(word);
            }
        } else {
            i += 1;
        }
    }
    names.len()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Budget {
    /// K sub-steps unrolled per hot-loop iteration (`step = SG * Nu`).
    unroll: usize,
    /// Output columns (tile-columns of 16) a threadgroup owns.
    columns: usize,
    /// Distinct `acc*` float locals: 2 per column (the two output halves).
    accumulators: usize,
    /// Distinct `p00..pNN` activation floats live at once: 4 per K sub-step.
    activation: usize,
    /// Distinct `const device uint *t*` tile pointers in the loop: 1 per
    /// (column x sub-step), 64-bit each.
    tile_pointers: usize,
    /// Loop-invariant column base pointers (`col*`, `column`): 1 per column.
    column_bases: usize,
    /// `float v[N]` decode buffer, shared across columns since T3.
    decode_buffer: usize,
}

impl Budget {
    /// Lower bound on 32-bit live values per thread in the hot loop.
    /// 64-bit pointers are counted as two words.
    fn lower_bound_words(&self) -> usize {
        self.accumulators
            + self.activation
            + 2 * self.tile_pointers
            + 2 * self.column_bases
            + self.decode_buffer
    }

}

fn scan(src: &str, name: &str) -> Budget {
    let body = kernel_body(src, name);
    let loop_block = unrolled_loop(src, name);

    // `const uint step = SG * 4u;` -> unroll 4.
    let unroll = body
        .split("const uint step = SG *")
        .nth(1)
        .and_then(|rest| {
            rest.chars()
                .find(|c| c.is_ascii_digit())
                .and_then(|d| d.to_digit(10))
        })
        .unwrap_or_else(|| panic!("{name}: no `const uint step = SG * Nu`")) as usize;

    // A column base is `col0/col1/col_a/col_b/column` declared before the loop.
    let pre_loop = body.split("for (; tk < k_unroll_end").next().unwrap_or(body);
    let column_bases = count_identifiers(pre_loop, "col") + count_identifiers(pre_loop, "column");

    let accumulators = count_identifiers(body, "acc");
    let activation = count_identifiers(&loop_block, "p0") + count_identifiers(&loop_block, "p1")
        + count_identifiers(&loop_block, "p2")
        + count_identifiers(&loop_block, "p3");
    let tile_pointers = count_identifiers(&loop_block, "t0_")
        + count_identifiers(&loop_block, "t1_")
        + count_identifiers(&loop_block, "t2_")
        + count_identifiers(&loop_block, "t3_");
    let decode_buffer = if loop_block.contains("float v[8]") { 8 } else { 0 };

    let columns = accumulators / 2;
    assert!(columns >= 1, "{name}: {accumulators} accumulators is not a whole number of columns");

    Budget {
        unroll,
        columns,
        accumulators,
        activation,
        tile_pointers,
        column_bases,
        decode_buffer,
    }
}

#[test]
fn coltile_register_budget() {
    let path = workspace_path("crates/kernels/src/linear_repacked.metal");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let src = strip_comments(&raw);

    // Every column-tiling variant, plus the two non-tiled colmajor kernels for
    // reference: `contig4` is the un-unrolled-column form (1 column, unroll 4)
    // that `ishizuki_gemv_pipeline` selects for a single colmajor projection.
    let variants = [
        "linear_gemv_ishizuki_k3_colmajor_contig4",
        "linear_gemv_ishizuki_k3_colmajor_1col",
        "linear_gemv_ishizuki_k3_colmajor_2col",
        "linear_gemv_ishizuki_k3_colmajor_4col",
        "linear_gemv_ishizuki_k3_fused2",
        "linear_gemv_ishizuki_k3_fused2_co",
        "linear_gemv_ishizuki_k3_fused2_2col",
        "linear_gemv_ishizuki_k3_fused2_4col",
        "linear_gemv_ishizuki_k3_fused2_co4",
        "linear_gemv_ishizuki_k3_fused3",
        "linear_gemv_ishizuki_k3_fused3_2col",
    ];

    let budgets: Vec<(&str, Budget)> = variants.iter().map(|n| (*n, scan(&src, n))).collect();

    println!(
        "\n{:<48} {:>6} {:>4} {:>5} {:>5} {:>5} {:>5} {:>6}",
        "kernel", "unroll", "cols", "acc", "act", "ptr", "base", "words"
    );
    for (name, b) in &budgets {
        println!(
            "{:<48} {:>6} {:>4} {:>5} {:>5} {:>5} {:>5} {:>6}",
            name,
            b.unroll,
            b.columns,
            b.accumulators,
            b.activation,
            b.tile_pointers,
            b.column_bases,
            b.lower_bound_words()
        );
    }

    let get = |n: &str| *budgets.iter().find(|(k, _)| *k == n).map(|(_, b)| b).unwrap();
    let two = get("linear_gemv_ishizuki_k3_colmajor_2col");
    let four = get("linear_gemv_ishizuki_k3_colmajor_4col");
    let co = get("linear_gemv_ishizuki_k3_fused2_co");
    let co4 = get("linear_gemv_ishizuki_k3_fused2_co4");
    let fused2_two = get("linear_gemv_ishizuki_k3_fused2_2col");
    let contig4 = get("linear_gemv_ishizuki_k3_colmajor_contig4");

    // A1: `_co` is NOT a register-saving variant of `_2col`. Same loop shape.
    assert_eq!(co.accumulators, two.accumulators, "_co accumulators != _2col");
    assert_eq!(co.activation, two.activation, "_co activation != _2col");
    assert_eq!(co.tile_pointers, two.tile_pointers, "_co tile ptrs != _2col");
    assert_eq!(co.unroll, two.unroll, "_co unroll != _2col");
    // And the fused2 form of 2col tiling, which is the one that actually runs
    // for gate+up, matches too.
    assert_eq!(fused2_two.accumulators, two.accumulators);
    assert_eq!(fused2_two.unroll, two.unroll);

    // A2: `_co4` inherits `_4col`'s cost exactly.
    assert_eq!(co4.accumulators, four.accumulators, "_co4 acc != _4col acc");
    assert_eq!(co4.activation, four.activation, "_co4 activation != _4col");
    assert_eq!(co4.tile_pointers, four.tile_pointers, "_co4 ptrs != _4col");
    assert_eq!(co4.unroll, four.unroll, "_co4 unroll != _4col");

    // The two anchors the report rests on.
    assert_eq!(two.columns, 2, "_2col owns 2 tile-columns");
    assert_eq!(two.unroll, 2, "_2col unrolls 2 K sub-steps");
    assert_eq!(four.columns, 4, "_4col owns 4 tile-columns");
    assert_eq!(four.unroll, 4, "_4col unrolls 4 K sub-steps");
    assert_eq!(contig4.columns, 1, "contig4 owns 1 tile-column");
    assert_eq!(contig4.unroll, 4, "contig4 unrolls 4 K sub-steps");

    // The growth that 4col pays: every term except the base pointers is 2x
    // between _2col and _4col, because unroll and column count both double.
    assert!(
        four.lower_bound_words() > 2 * two.lower_bound_words(),
        "4col is {} words vs 2col's {}: the doubling claim in K03 is wrong, \
         fix the report",
        four.lower_bound_words(),
        two.lower_bound_words()
    );
    println!(
        "\n_2col {:>3} words, _4col {:>3} words ({:.2}x);  contig4 (1 col, unroll 4) {:>3} words",
        two.lower_bound_words(),
        four.lower_bound_words(),
        four.lower_bound_words() as f64 / two.lower_bound_words() as f64,
        contig4.lower_bound_words()
    );

    // What the unroll costs on its own, holding columns fixed at 1: that is the
    // only knob `_4col` could have turned down, and it did not.
    assert!(contig4.lower_bound_words() < two.lower_bound_words(),
        "1 column at unroll 4 should be cheaper than 2 columns at unroll 2");
}

// ---------------------------------------------------------------------------
// Grid arithmetic for the seven production projections
// ---------------------------------------------------------------------------

/// GPU cores on the target box (AGENTS.md §8). Every kernel here is dispatched
/// at 256 threads (`groups * 256, 256`) with `constexpr int SG = 8`.
const CORES: u64 = 20;

/// `groups` expressions transcribed from the three dispatch sites, because the
/// point of the table is the grid, and the grid is what halved when 4col won.
/// `perf_probe_occupancy_grid.rs` pins the same numbers for the same sites.
fn grid_at_two_col(out_features: usize) -> u64 {
    // encode_gemv_raw_sk2 -> linear.rs:2255-2259: ((out/16)/2)*2
    (((out_features / 16) / 2) * 2) as u64
}

fn grid_at_four_col(out_features: usize) -> u64 {
    // encode_gemv_raw (4col arm) -> linear.rs:2117-2118: out/64
    (out_features / 64) as u64
}

fn grid_fused2_2col(n_a: usize, n_b: usize) -> u64 {
    // encode_gemv_fused2_mlp -> linear.rs:2597: (tiles_a + tiles_b) / 2
    (((n_a / 16) + (n_b / 16)) / 2) as u64
}

fn grid_fused2_co(n: usize) -> u64 {
    // encode_gemv_fused2_mlp -> linear.rs:2619: tiles_a
    (n / 16) as u64
}

fn grid_fused2_co4(n: usize) -> u64 {
    // encode_gemv_fused2_mlp -> linear.rs:2602: tiles_a / 2
    ((n / 16) / 2) as u64
}

fn grid_fused3_2col(n: &[usize]) -> u64 {
    // encode_gemv_fused3 -> linear.rs:2787: (tiles_a + tiles_b + tiles_c) / 2
    (n.iter().map(|v| v / 16).sum::<usize>() / 2) as u64
}

#[test]
fn coltile_grid_arithmetic() {
    // Shapes: the measured checkpoint is q 8192, k/v 4096, o out 5376,
    // gate/up 21504, down out 5376, head 262144 @ 6 bpw row-major. The brief's
    // 4-kv-head variant (k/v 2048) is printed too because that is the shape the
    // T4.2 4col number was taken against.
    println!(
        "\n{:<34} {:>8} {:>10} {:>9} {:>12}",
        "projection", "TGs(2col)", "TG/core", "TGs(4col)", "TG/core"
    );
    let mut rows: Vec<(&str, u64, u64)> = vec![
        ("o_proj   N=5376 (2col_sk2)", grid_at_two_col(5376), grid_at_four_col(5376)),
        ("o_proj   N=8192 (2col_sk2)", grid_at_two_col(8192), grid_at_four_col(8192)),
        ("down     N=5376 (2col_sk2)", grid_at_two_col(5376), grid_at_four_col(5376)),
        ("qkv      16 kv heads", grid_fused3_2col(&[8192, 4096, 4096]), 0),
        ("qkv       4 kv heads", grid_fused3_2col(&[8192, 2048, 2048]), 0),
        ("gate+up  fused2_2col", grid_fused2_2col(21504, 21504), grid_fused2_co4(21504)),
        ("gate+up  fused2_co  (unreachable)", grid_fused2_co(21504), grid_fused2_co4(21504)),
    ];
    for (label, two, four) in rows.drain(..) {
        println!(
            "{:<34} {:>8} {:>10.1} {:>9} {:>12.1}",
            label,
            two,
            two as f64 / CORES as f64,
            four,
            four as f64 / CORES as f64
        );
    }

    // Anchors, matching `perf_probe_occupancy_grid::census`.
    assert_eq!(grid_at_two_col(5376), 336, "o_proj/down split-K2 grid");
    assert_eq!(grid_fused3_2col(&[8192, 4096, 4096]), 512, "qkv grid");
    assert_eq!(grid_fused2_2col(21504, 21504), 1344, "gate+up grid");
    assert_eq!(grid_at_four_col(5376), 84, "the 84-threadgroup grid T4.2 cites");
    assert_eq!(grid_at_four_col(8192), 128, "4col grid for the 8192-wide shape");
    // `_co` and `_2col` launch the SAME grid for the MLP, so the choice between
    // them cannot be an occupancy question at all.
    assert_eq!(grid_fused2_co(21504), grid_fused2_2col(21504, 21504));
}
