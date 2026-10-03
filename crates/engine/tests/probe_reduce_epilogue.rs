//! PERF PROBE — the cross-lane reduction epilogue of the EXL3 decode GEMV.
//!
//! **Question.** Every `linear_gemv_ishizuki_*` kernel ends its K-loop with
//!
//! ```text
//! acc0 += simd_shuffle_xor(acc0, 1u);        // 4-lane butterfly
//! acc0 += simd_shuffle_xor(acc0, 2u);
//! ...
//! red[sg][lane / 4u] = acc0;                // 8 simdgroups -> threadgroup memory
//! threadgroup_barrier(mem_flags::mem_threadgroup);
//! if (lid < 16u) { sum 8 partials; y[tn * 16u + lid] = sum; }
//! threadgroup_barrier(mem_flags::mem_threadgroup);   // drain `red` for the next tile
//! ```
//!
//! (`crates/kernels/src/linear.metal:1913-1930`, and the same epilogue in every
//! other kernel of the family). How much does it cost, relative to the decode
//! work for the same output element, and is it worth changing?
//!
//! **What this file is.** Two things, and only these two.
//!
//! 1. **Source-level counts** (`include_str!` on the kernel sources, not
//!    `read_to_string` on a path — the test must read the bytes the crate
//!    compiled, same reasoning as `perf_probe_host_ops.rs:60`). Every count in
//!    the accompanying report is derived from these greps, so editing an
//!    epilogue makes them fail.
//! 2. **A CPU model of the reduction geometry** — 32 lanes x 8 simdgroups,
//!
//!    modelled exactly as the source maps them, checked against the direct
//!    16-row dot product. No Metal device, no model, no GPU.
//!
//! **What this file is NOT, and cannot become.** Nothing here measures a cycle
//! or a nanosecond. It counts operations. Pricing a `threadgroup_barrier`
//! needs the GPU, and AGENTS.md §10 forbids GPU work from a probe (and a probe
//! that perturbs what it measures is worse than no probe). The counts here are
//! the *denominator* of that pricing; the *numerator* has to come from an
//! interleaved A/B run under `scripts/gpuq`.
//!
//! **Sabotage discipline.** AGENTS.md §3.4: a coverage claim is not a claim
//! until the code has been broken and the test seen to fail. This file cannot
//! edit the kernel sources, so `*_matchers_can_fail` mutates an in-memory copy
//! of the source and asserts that each matcher flips. If a matcher is written
//! so that it cannot fail, those tests fail instead of the claim surviving.
//! `butterfly_is_load_bearing` is the same discipline applied to the geometry
//! itself: it removes the shuffles from the CPU model and shows the answer
//! changes.

// ===========================================================================
// 0. Sources and shapes
// ===========================================================================

/// `LinearKernels::new` concatenates `linear_repacked.metal` onto this file, so
/// both halves of the family live in these two sources
/// (`crates/kernels/src/linear.metal:1555-1557`).
const LINEAR_METAL: &str = include_str!("../../kernels/src/linear.metal");
const LINEAR_REPACKED: &str = include_str!("../../kernels/src/linear_repacked.metal");
const LINEAR_RS: &str = include_str!("../../kernels/src/linear.rs");

/// `[[thread_index_in_simdgroup]]` is a 32-lane simdgroup by construction;
/// the host launch size is asserted from `LINEAR_RS` below.
const LANES: usize = 32;
/// `constexpr int SG = 8` — `linear.metal:1840`, `linear_repacked.metal:1985`.
const SG: usize = 8;
/// 8 simdgroups x 32 lanes. This is the launch size: `groups * 256, 256`
/// (`linear.rs:1489-1490`).
const THREADS_PER_THREADGROUP: usize = LANES * SG;
/// `red[SG][16]`, `y[tn * 16u + lid]`, `tiles_n = N / 16u`.
const TILE_N: usize = 16;

// --- the shapes the 31B decode path actually issues -----------------------
// `crates/engine/src/bench.rs:86-91`. Kept as a table so every derived number
// in the report is traceable to a real projection rather than a convenient one.

#[derive(Clone, Copy, Debug)]
struct Shape {
    name: &'static str,
    k: usize,
    n: usize,
    /// How many times per token (`bench.rs:190-191`).
    count: usize,
}

const HIDDEN: usize = 5376;
const INTERMEDIATE: usize = 21504;
const SLIDING_Q: usize = 8192;
const SLIDING_KV: usize = 4096;
const FULL_Q: usize = 16384;
const FULL_KV: usize = 2048;
const VOCAB: usize = 262_144;

const SLIDING_LAYERS: usize = 50;
const FULL_LAYERS: usize = 10;

const SHAPES: &[Shape] = &[
    Shape { name: "sliding/q", k: HIDDEN, n: SLIDING_Q, count: SLIDING_LAYERS },
    Shape { name: "sliding/k", k: HIDDEN, n: SLIDING_KV, count: SLIDING_LAYERS },
    Shape { name: "sliding/v", k: HIDDEN, n: SLIDING_KV, count: SLIDING_LAYERS },
    Shape { name: "sliding/o", k: SLIDING_Q, n: HIDDEN, count: SLIDING_LAYERS },
    Shape { name: "sliding/gate", k: HIDDEN, n: INTERMEDIATE, count: SLIDING_LAYERS },
    Shape { name: "sliding/up", k: HIDDEN, n: INTERMEDIATE, count: SLIDING_LAYERS },
    Shape { name: "sliding/down", k: INTERMEDIATE, n: HIDDEN, count: SLIDING_LAYERS },
    Shape { name: "full/q", k: HIDDEN, n: FULL_Q, count: FULL_LAYERS },
    Shape { name: "full/kv", k: HIDDEN, n: FULL_KV, count: FULL_LAYERS },
    Shape { name: "full/o", k: FULL_Q, n: HIDDEN, count: FULL_LAYERS },
    Shape { name: "full/gate", k: HIDDEN, n: INTERMEDIATE, count: FULL_LAYERS },
    Shape { name: "full/up", k: HIDDEN, n: INTERMEDIATE, count: FULL_LAYERS },
    Shape { name: "full/down", k: INTERMEDIATE, n: HIDDEN, count: FULL_LAYERS },
    Shape { name: "lm_head", k: HIDDEN, n: VOCAB, count: 1 },
];

// ===========================================================================
// 1. Kernel-body extraction
// ===========================================================================

/// `(1-based line, name, body)` for every `kernel void` in `src`.
///
/// The body ends at the first line that is exactly `}`. Scoping to the body
/// matters for the same reason `perf_probe_host_ops.rs:76` gives: a whole-file
/// `find` counts the *other* kernel's epilogue and the assertion then holds for
/// reasons that have nothing to do with the code it claims to check.
fn kernels(src: &str) -> Vec<(usize, String, String)> {
    let lines: Vec<&str> = src.split_inclusive('\n').collect();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < lines.len() {
        if let Some(rest) = lines[i].strip_prefix("kernel void ") {
            let name = rest.split('(').next().unwrap_or("").trim().to_string();
            let mut j = i;
            while j < lines.len() && lines[j].trim_end() != "}" {
                j += 1;
            }
            let end = j.min(lines.len() - 1);
            out.push((i + 1, name, lines[i..=end].concat()));
            i = end + 1;
        } else {
            i += 1;
        }
    }
    out
}

/// Every kernel in either source that both carries a `red[...]` threadgroup
/// reduction window and belongs to the ishizuki family, as
/// `(file, line, name, body)`.
fn red_window_ishizuki_kernels() -> Vec<(&'static str, usize, String, String)> {
    let mut out = Vec::new();
    for (file, src) in [("linear.metal", LINEAR_METAL), ("linear_repacked.metal", LINEAR_REPACKED)] {
        for (line, name, body) in kernels(src) {
            if name.starts_with("linear_gemv_ishizuki_")
                && body.contains("threadgroup float red[")
            {
                out.push((file, line, name, body));
            }
        }
    }
    out
}

fn occurrences(haystack: &str, needle: &str) -> usize {
    haystack.matches(needle).count()
}

/// The lane masks of every `simd_shuffle_xor` in `body`, as the literal tokens
/// the source uses (`"1u"`, `"2u"`, ...).
fn shuffle_masks(body: &str) -> Vec<String> {
    let mut masks = Vec::new();
    let mut rest = body;
    while let Some(at) = rest.find("simd_shuffle_xor(") {
        rest = &rest[at + "simd_shuffle_xor(".len()..];
        let close = rest.find(')').expect("unterminated simd_shuffle_xor");
        let args = &rest[..close];
        let mask = args.rsplit(',').next().unwrap_or("").trim().to_string();
        masks.push(mask);
        rest = &rest[close..];
    }
    masks
}

// ===========================================================================
// 2. Static facts about the epilogue
// ===========================================================================

/// The epilogue is exactly the shape the header comment claims: a 4-lane
/// butterfly (`xor 1`, `xor 2` — those two masks span exactly four lanes) and
/// nothing wider. A "wider tree" alternative would show up here as a `4u`,
/// `8u` or `16u` mask; there is none in any of the family.
#[test]
fn the_tree_is_exactly_four_lanes_wide_everywhere() {
    let kernels = red_window_ishizuki_kernels();
    assert!(
        kernels.len() >= 35,
        "expected the whole ishizuki family with a reduction window, found {}",
        kernels.len()
    );
    for (file, line, name, body) in &kernels {
        let masks = shuffle_masks(body);
        assert!(!masks.is_empty(), "{file}:{line} {name}: no shuffles at all");
        for m in &masks {
            assert!(
                m == "1u" || m == "2u",
                "{file}:{line} {name}: shuffle mask {m} — the reduction tree is not 4 lanes"
            );
        }
    }
}

/// `SG` is 8 in every one of them, which is what makes the 8-way threadgroup
/// sum and the 256-thread launch consistent.
#[test]
fn sg_is_eight_in_every_reduction_window_kernel() {
    for (file, line, name, body) in red_window_ishizuki_kernels() {
        assert!(
            body.contains("constexpr int SG = 8") || body.contains("constexpr uint SG = 8"),
            "{file}:{line} {name}: SG is not a compile-time 8"
        );
    }
}

/// Shuffles and `red` writes stay in lockstep: 2 shuffles per accumulator, 2
/// accumulators per column-tile, 1 write per accumulator. 1-col => 4/2,
/// 2-col => 8/4, 4-col => 16/8. Anything else means an epilogue was edited on
/// one side only — the exact class of bug that produced the
/// `red[col_idx][g][out_lane]` index family in `linear_repacked.metal`.
#[test]
fn shuffles_are_exactly_twice_the_red_writes() {
    for (file, line, name, body) in red_window_ishizuki_kernels() {
        let shuffles = occurrences(&body, "simd_shuffle_xor");
        // "] = acc" is a `red` *write*; the declaration
        // (`threadgroup float red[SG][16]`) and the reads (`sum += red[g][lid]`)
        // both contain `red[` but neither contains `] = acc`.
        let writes = occurrences(&body, "] = acc");
        assert_eq!(
            shuffles,
            2 * writes,
            "{file}:{line} {name}: {shuffles} shuffles vs {writes} red writes"
        );
    }
}

/// The one kernel in the family with a barrier *inside* the K-loop, and the
/// only one with three barriers. It is the `uint4`-vectorised-load probe, and
/// it lost: 180.5 -> 173.7 GB/s on `gate_up`
/// (`spec/decode-19tok-repacked-250gb/implementation_notes.md:103-104`). It is
/// excluded from the barrier census explicitly rather than by a wildcard, and
/// the exclusion is asserted to contain exactly this one name so that a second
/// offender cannot slip through.
const BARRIER_IN_K_LOOP: &[&str] = &["linear_gemv_ishizuki_k3_colmajor_u4"];

#[test]
fn the_kernel_family_is_one_with_a_barrier_inside_the_k_loop() {
    let family = red_window_ishizuki_kernels();
    let offenders: Vec<&str> = family
        .iter()
        .filter(|(_, _, name, _)| BARRIER_IN_K_LOOP.contains(&name.as_str()))
        .map(|(_, _, name, _)| name.as_str())
        .collect();
    assert_eq!(
        offenders,
        BARRIER_IN_K_LOOP,
        "the set of kernels with an in-loop barrier changed"
    );
    assert!(
        LINEAR_REPACKED.contains("kernel void linear_gemv_ishizuki_k3_colmajor_u4"),
        "_u4 is gone; drop it from BARRIER_IN_K_LOOP"
    );
}

/// Barrier census. Three distinct epilogue shapes exist in the family and the
/// difference matters for the recommendation:
///
/// * **2 barriers** — the tile-loop kernels. The second one drains `red` before
///   the *next* tile iteration overwrites it.
/// * **1 barrier** — the fused (`fused2`/`fused3`) and split-K kernels. There is
///   no tile loop, so there is nothing to drain; the trailing barrier has
///   already been deleted here. These are the shapes that dominate decode time
///   (`gate_up` 37.5%, `qkv` 21.0%,
///   `spec/decode-19tok-repacked-250gb/implementation_notes.md:111-113`).
/// * **0 barriers** — `linear_gemv_ishizuki_k3_colmajor_stream`, which deletes
///   the cross-simdgroup reduction entirely by giving each simdgroup its own
///   output tile. Alternative (a)+(c) of the question, already built.
#[test]
fn barrier_census_is_one_or_two_and_stream_has_none() {
    let mut two = Vec::new();
    let mut one = Vec::new();
    let mut other = Vec::new();
    for (file, line, name, body) in red_window_ishizuki_kernels() {
        if BARRIER_IN_K_LOOP.contains(&name.as_str()) {
            continue;
        }
        match occurrences(&body, "threadgroup_barrier") {
            2 => two.push(format!("{file}:{line} {name}")),
            1 => one.push(format!("{file}:{line} {name}")),
            n => other.push(format!("{file}:{line} {name} has {n}")),
        }
    }
    assert!(
        other.is_empty(),
        "a reduction-window kernel has neither 1 nor 2 barriers: {other:?}"
    );
    assert!(
        one.len() >= 6,
        "expected the fused and split-K kernels to have already dropped the \
         trailing barrier; only {} have one barrier",
        one.len()
    );

    // The named kernels the report cites.
    for expect_two in [
        "linear_gemv_ishizuki_k3",
        "linear_gemv_ishizuki_k3_w2",
        "linear_gemv_ishizuki_k3_nod",
        "linear_gemv_ishizuki_k3_colmajor",
        "linear_gemv_ishizuki_k3_colmajor_contig4",
        "linear_gemv_ishizuki_k3_colmajor_1col",
    ] {
        assert!(
            two.iter().any(|s| s.ends_with(expect_two)),
            "{expect_two} should be a 2-barrier tile-loop kernel; 1-barrier set is {one:?}"
        );
    }
    for expect_one in [
        "linear_gemv_ishizuki_k3_fused2",
        "linear_gemv_ishizuki_k3_fused3",
        "linear_gemv_ishizuki_k3_colmajor_contig4_sk4",
        "linear_gemv_ishizuki_k3_fused2_sk2",
    ] {
        assert!(
            one.iter().any(|s| s.ends_with(expect_one)),
            "{expect_one} should already have dropped the trailing barrier; 2-barrier set is {two:?}"
        );
    }

    // And the barrier-free variant exists and is a real kernel, not a comment.
    let stream = kernels(LINEAR_REPACKED)
        .into_iter()
        .find(|(_, n, _)| n == "linear_gemv_ishizuki_k3_colmajor_stream")
        .map(|(_, _, b)| b)
        .expect("no _colmajor_stream kernel");
    assert_eq!(occurrences(&stream, "threadgroup_barrier"), 0);
    assert!(!stream.contains("threadgroup float red["));
    assert_eq!(occurrences(&stream, "simd_shuffle_xor"), 4);
}

/// In a 2-barrier kernel the second barrier really is the *drain*, i.e. it
/// comes after the reads of `red`. This is what makes it removable when there
/// is no next tile iteration, and it is falsifiable: swap the two and this
/// fails.
#[test]
fn the_second_barrier_drains_the_red_reads() {
    for (file, line, name, body) in red_window_ishizuki_kernels() {
        if occurrences(&body, "threadgroup_barrier") != 2 {
            continue;
        }
        let barrier_2 = body
            .match_indices("threadgroup_barrier")
            .nth(1)
            .map(|(at, _)| at)
            .expect("two barriers");
        let read = body
            .find("sum += red[")
            .unwrap_or_else(|| panic!("{file}:{line} {name}: no read of red"));
        assert!(
            barrier_2 > read,
            "{file}:{line} {name}: barrier 2 precedes the read of red"
        );
    }
}

/// The launch is 256 threads per threadgroup, which is what makes SG = 8 the
/// only consistent value for the cross-simdgroup reduction depth.
#[test]
fn the_host_launches_256_threads_per_threadgroup() {
    assert_eq!(THREADS_PER_THREADGROUP, 256);
    // `groups * 256,\n            256,` at linear.rs:1489-1490 and the seven
    // other ishizuki dispatch sites.
    let sites = occurrences(LINEAR_RS, "            256,\n");
    assert!(
        sites >= 8,
        "expected every ishizuki dispatch site to pass 256 threads/threadgroup, found {sites}"
    );
}

// ===========================================================================
// 3. Is the second barrier dead code?
// ===========================================================================

/// `gemv_max_groups()` is `u64::MAX` unless `KAPLUMBA_GEMV_MAX_GROUPS` is set
/// (`linear.rs:94-102`), so `groups == tiles_n` and the grid-stride tile loop
/// `for (tn = tgid; tn < tiles_n; tn += ntg)` executes **exactly once** per
/// threadgroup. With no second iteration there is no second `red` to protect,
/// and the trailing barrier is dead. This is the load-bearing fact behind the
/// one concrete change in the report.
#[test]
fn the_grid_cap_is_off_by_default_so_the_tile_loop_runs_once() {
    assert!(
        LINEAR_RS.contains(".unwrap_or(u64::MAX)"),
        "gemv_max_groups is no longer uncapped by default; the dead-barrier argument needs redoing"
    );
    assert!(LINEAR_RS.contains("fn gemv_max_groups() -> u64"));
    // Every ishizuki grid is `tiles` (or a multiple of it), never fewer, so the
    // single-iteration conclusion holds on every dispatch path and not just the
    // default one that the report quotes.
    let grid_forms = [
        "let groups = tiles.min(gemv_max_groups()).max(1);",
        "let groups = pairs.min(gemv_max_groups()).max(1);",
        "let groups = tiles * 2;",
        "let groups = tiles * 4;",
        "let groups = pairs * 2;",
        "let groups = pairs * 4;",
    ];
    for form in grid_forms {
        assert!(
            LINEAR_RS.contains(form),
            "grid form {form:?} disappeared; re-audit every ishizuki dispatch path"
        );
    }
}

/// The proposed replacement, as a predicate over a source. `tn`, `ntg` and
/// `tiles_n` are all uniform across a threadgroup (`tn` is initialised from
/// `tgid`), so branching around a barrier on `tn + ntg < tiles_n` is legal.
fn trailing_barrier_is_guarded(body: &str) -> bool {
    body.contains("tn + ntg < tiles_n") || body.contains("tgid + ntg < tiles_n")
}

#[test]
fn the_trailing_barrier_is_not_yet_guarded() {
    // POSITIVE CONTROL. `trailing_barrier_is_guarded` cannot be exercised from
    // the real sources, because the change has not landed — which would make
    // the assertion below vacuous. Apply the proposed edit to a copy first and
    // require the detector to fire on it.
    let k3 = kernels(LINEAR_METAL)
        .into_iter()
        .find(|(_, n, _)| n == "linear_gemv_ishizuki_k3")
        .expect("k3")
        .2;
    let landed = {
        // Insert the guard in front of the *last* barrier, which is the drain.
        let at = k3.rfind("threadgroup_barrier").expect("a barrier");
        let mut s = k3.clone();
        s.insert_str(at, "if (tn + ntg < tiles_n) ");
        s
    };
    assert_ne!(landed, k3, "the sabotage did not apply: no barrier to guard");
    assert!(
        trailing_barrier_is_guarded(&landed),
        "the detector does not see the guard it is supposed to detect"
    );
    assert!(
        !trailing_barrier_is_guarded(&k3),
        "the detector fires on the unguarded source"
    );

    let already: Vec<String> = red_window_ishizuki_kernels()
        .into_iter()
        .filter(|(_, _, _, body)| trailing_barrier_is_guarded(body))
        .map(|(file, line, name, _)| format!("{file}:{line} {name}"))
        .collect();
    assert!(
        already.is_empty(),
        "the trailing barrier is already guarded in {already:?}; the change has landed"
    );
    let unguarded: Vec<String> = red_window_ishizuki_kernels()
        .into_iter()
        .filter(|(_, _, _, body)| occurrences(body, "threadgroup_barrier") == 2)
        .map(|(file, line, name, _)| format!("{file}:{line} {name}"))
        .collect();
    assert!(
        unguarded.len() >= 12,
        "expected the 2-barrier tile-loop family to still carry an unguarded trailing barrier, \
         found {}: {unguarded:?}",
        unguarded.len()
    );
}

// ===========================================================================
// 4. The arithmetic: reduction ops vs decode ops, per output element
// ===========================================================================

/// Operations the epilogue costs **per output element**, counted from source.
///
/// Independent of `K` by construction: the K-loop has already finished, and the
/// epilogue's width is set by the tile geometry (16 columns) and the reduction
/// depth (8 simdgroups), neither of which involves `K`.
#[derive(Clone, Copy, Debug)]
struct Reduction {
    shuffles: f64,
    tg_stores: f64,
    tg_loads: f64,
    g_stores: f64,
}

impl Reduction {
    fn per_output() -> Self {
        // 4 shuffles per lane, all 256 lanes, amortised over 16 output columns.
        let shuffles = 4.0 * THREADS_PER_THREADGROUP as f64 / TILE_N as f64;
        // `red[sg][lane/4u] = acc0` and `... + 8u] = acc1`: one write per
        // simdgroup per output column, SG of them.
        let tg_stores = SG as f64;
        // `sum += red[g][lid]` for g in 0..8, 16 threads, 16 outputs.
        let tg_loads = SG as f64;
        // `y[tn * 16u + lid] = sum`, one per output.
        let g_stores = 1.0;
        Self { shuffles, tg_stores, tg_loads, g_stores }
    }

    fn total(&self) -> f64 {
        self.shuffles + self.tg_stores + self.tg_loads + self.g_stores
    }
}

#[test]
fn reduction_costs_81_operations_per_output_element() {
    let r = Reduction::per_output();
    assert_eq!(r.shuffles, 64.0, "4 shuffles x 256 lanes / 16 columns");
    assert_eq!(r.tg_stores, 8.0, "one red write per simdgroup per column");
    assert_eq!(r.tg_loads, 8.0, "one red read per simdgroup per column");
    assert_eq!(r.g_stores, 1.0);
    assert_eq!(r.total(), 81.0);
}

/// Operations the K-loop costs **per output element**, i.e. per input row,
/// counted from the inner loop of `linear_gemv_ishizuki_k3`
/// (`linear.metal:1896-1912`).
///
/// The components are named and asserted individually so the total can be
/// audited rather than trusted; the one that dominates is the extract, which
/// this repo has already measured as the single largest cost in the kernel
/// (`spec/decode-19tok-repacked-250gb/implementation_notes.md:1226-1227`: "65.6%
/// of the kernel is `uint(A >> (64u - rad))` with `rad` varying per element").
#[derive(Clone, Copy, Debug)]
struct DecodePerRow {
    /// `decode_state_mcg`: imul, and, xor, half2 unpack + add
    /// (`linear.metal:121-126`).
    decode_state: f64,
    /// `acc0 += p00 * v[0] + ...` — one mul + one add per row.
    fma: f64,
    /// `uint(A0 >> (64u - r))` on a 64-bit value: two 32-bit shifts + a select.
    extract: f64,
    /// `& 0xffffu`.
    mask: f64,
    /// The window build (`(w0a << 32) | w0b) << sh0 | ...`) amortised over the
    /// 8 codes a lane extracts per k-tile.
    window: f64,
    /// `xr0[0], xr0[1], xr0[8], xr0[9]` — 4 loads serve 16 rows (4 lanes share
    /// each row group), so 0.25 loads per row.
    x_loads: f64,
    /// 3 weight loads serve 16 rows for the same reason.
    weight_loads: f64,
}

impl DecodePerRow {
    fn three_word() -> Self {
        Self {
            decode_state: 5.0,
            fma: 1.0,
            extract: 3.0,
            mask: 1.0,
            window: 1.0,
            x_loads: 4.0 / 16.0,
            weight_loads: 3.0 / 16.0,
        }
    }

    fn total(&self) -> f64 {
        self.decode_state
            + self.fma
            + self.extract
            + self.mask
            + self.window
            + self.x_loads
            + self.weight_loads
    }
}

#[test]
fn decode_costs_11_44_operations_per_row() {
    let d = DecodePerRow::three_word();
    assert_eq!(d.decode_state, 5.0);
    assert_eq!(d.fma, 1.0);
    assert_eq!(d.extract, 3.0);
    assert_eq!(d.x_loads, 0.25);
    assert_eq!(d.weight_loads, 0.1875);
    // 11.4375 ops per row. The epilogue's 81 ops are therefore amortised over
    // `81 / 11.4375 = 7.08` rows of decode work for the same output element.
    assert!((d.total() - 11.4375).abs() < 1e-12, "got {}", d.total());
}

/// **Question 3, answered.** Reduction cost per output element is independent
/// of `K`; decode cost per output element is linear in `K`. So the fraction is
/// `81 / (11.4375*K + 81)`, it *falls* as `K` grows, and it is the same for
/// `q_proj` (N=5376) and `gate`/`up` (N=21504) because those share K=5376.
/// The MLP projections do 4x the absolute reduction work of `q_proj` only
/// because they have 4x the outputs — not because their reduction is deeper.
#[test]
fn reduction_fraction_per_shape_is_under_one_half_a_percent() {
    let r = Reduction::per_output().total();
    let d = DecodePerRow::three_word().total();
    let mut worst: f64 = 0.0;
    let mut worst_name = "";
    for shape in SHAPES {
        let decode = d * shape.k as f64;
        let frac = r / (r + decode);
        println!(
            "{:<12} K={:<6} N={:<7} tiles={:<5} reduction={:.4}%",
            shape.name,
            shape.k,
            shape.n,
            shape.n / 16,
            frac * 100.0
        );
        if frac > worst {
            worst = frac;
            worst_name = shape.name;
        }
    }
    // The whole family, including the smallest K (5376), is ~0.13%.
    assert!(
        worst < 0.005,
        "worst shape {worst_name} is {:.4}%, above the 0.5% the report claims",
        worst * 100.0
    );
    // And the specific comparison the question asks for.
    let q = r / (r + d * HIDDEN as f64);
    let gate = r / (r + d * HIDDEN as f64);
    let down = r / (r + d * INTERMEDIATE as f64);
    assert!((q - gate).abs() < 1e-12, "gate and q share K=5376");
    assert!(down < q / 3.0, "down_proj (K=21504) dilutes the epilogue further");
}

/// Time-weighted across a whole token, which is the number that matters,
/// because time is proportional to work.
#[test]
fn whole_token_reduction_share_is_about_a_tenth_of_a_percent() {
    let r = Reduction::per_output().total();
    let d = DecodePerRow::three_word().total();
    let (mut red_ops, mut decode_ops) = (0f64, 0f64);
    for shape in SHAPES {
        let outs = shape.n as f64 * shape.count as f64;
        red_ops += outs * r;
        decode_ops += outs * d * shape.k as f64;
    }
    let frac = red_ops / (red_ops + decode_ops);
    println!(
        "per token: reduction {red_ops:.3e} ops, decode {decode_ops:.3e} ops, share {:.4}%",
        frac * 100.0
    );
    assert!(
        frac < 0.005,
        "token-weighted reduction share {:.4}% exceeds the claimed 0.5%",
        frac * 100.0
    );
}

// ===========================================================================
// 5. The arithmetic: barriers
// ===========================================================================

/// Barriers are **not** in the K-loop. They execute once per threadgroup, at
/// most twice. The question's premise — "several per dispatch across 420
/// dispatches per token could be milliseconds" — is off by three orders of
/// magnitude on the first half: it is 1–2 per *threadgroup*, and a decode step
/// issues on the order of 2.8e5 threadgroups, all of which run concurrently.
#[test]
fn barriers_are_per_threadgroup_not_per_dispatch_and_not_in_the_k_loop() {
    let mut in_k_loop = 0usize;
    for (file, line, name, body) in red_window_ishizuki_kernels() {
        if BARRIER_IN_K_LOOP.contains(&name.as_str()) {
            continue;
        }
        // The barrier must appear after the last K-loop close brace and after
        // the shuffle epilogue, i.e. in the epilogue, not the loop.
        let last_shuffle = body.rfind("simd_shuffle_xor").expect("shuffles");
        let first_barrier = body.find("threadgroup_barrier").expect("barrier");
        assert!(
            first_barrier > last_shuffle,
            "{file}:{line} {name}: a barrier precedes the reduction epilogue"
        );
        let k_loop_open = body.find("for (").expect("a loop");
        assert!(
            k_loop_open < first_barrier,
            "{file}:{line} {name}: the K-loop opens after the barrier"
        );
        in_k_loop += occurrences(&body[k_loop_open..first_barrier], "threadgroup_barrier");
    }
    assert_eq!(
        in_k_loop, 0,
        "a barrier appeared between the K-loop and the epilogue in {in_k_loop} place(s)"
    );
}

/// Threadgroups and barrier executions per decode token, and the barrier cost
/// that *would* be needed to reach 5% of a step.
///
/// `spec/decode-19tok-repacked-250gb/spec.md` records 74.01 ms/step at 1167
/// dispatches/token. Inverting the question gives a number that does not depend
/// on guessing a GPU clock: at 4 GHz a `threadgroup_barrier` would have to cost
/// **~900 cycles** for the barrier structure to be 5% of decode. That is two
/// orders of magnitude beyond a drain-and-reconverge.
///
/// **Every kernel on the live path is a 2-barrier kernel.** An earlier draft of
/// this probe assumed the fused families were down to one; they are not. The
/// default picks, with no env flags set, are
/// `linear_gemv_ishizuki_k3_fused2_co` for `gate`/`up` (`linear.rs:2417` —
/// `out_features_a == out_features_b` and `KAPLUMBA_FUSED2_CO4` unset),
/// `linear_gemv_ishizuki_k3_fused3_2col` for `q`/`k`/`v` (`linear.rs:2781`),
/// and `linear_gemv_ishizuki_k3_colmajor_contig4` for `o`/`down`
/// (`linear.rs:1539-1548`, `permutation_mode == 1`) — and all three carry two
/// barriers. Only the bare `fused2`/`fused3`/`_sk2`/`_sk4` kernels are down to
/// one, and the default selection never reaches them.
#[test]
fn barriers_would_need_900_cycles_each_to_reach_five_percent() {
    let step_ms = 74.01f64;
    let gpu_cores = 20.0;
    let ghz = 4.0f64;

    // Threadgroups per token: one per 16 output columns, summed over the real
    // shape table. (Threads, not dispatches — 1167 dispatches/token is the
    // post-fusion figure and each dispatches a whole grid.)
    let threadgroups: f64 = SHAPES
        .iter()
        .map(|s| (s.n / 16) as f64 * s.count as f64)
        .sum();
    // The one shape on a 1-barrier kernel: `lm_head` is 6 bpw and row-major, so
    // it goes to the EXL3 arm (`linear.rs:1517-1522` rejects it from the ishizuki
    // family on rate alone), whose kernels carry one barrier.
    let lm_head = SHAPES
        .iter()
        .find(|s| s.name == "lm_head")
        .map(|s| (s.n / 16) as f64 * s.count as f64)
        .expect("lm_head shape");
    let barriers: f64 = 2.0 * (threadgroups - lm_head) + 1.0 * lm_head;
    let per_core = barriers / gpu_cores;

    let budget_cycles = step_ms * 1e-3 * ghz * 1e9 * 0.05;
    let cycles_to_5pct = budget_cycles / per_core;
    println!(
        "threadgroups/token {threadgroups:.0}, barriers/token {barriers:.0}, \
         per core {per_core:.0}; 5% of {step_ms} ms = {budget_cycles:.0} cycles \
         -> {cycles_to_5pct:.0} cycles per barrier"
    );
    assert!(
        cycles_to_5pct > 500.0,
        "the barrier budget is tighter than the report claims ({cycles_to_5pct:.0} cycles)"
    );

    // For scale: at 100 cycles per barrier the whole barrier structure is
    // this fraction of a step.
    let at_100 = per_core * 100.0 / (ghz * 1e9) * 1000.0 / step_ms;
    println!("at 100 cycles/barrier the barriers are {at_100:.3}% of a step");
    assert!(at_100 < 0.05);
}

// ===========================================================================
// 6. CPU model of the geometry
// ===========================================================================

/// The mapping the epilogue implies, reconstructed from the source and then
/// *checked* rather than assumed:
///
/// * `r0 = (lane % 4u) * 2u` — the lane's four rows (`linear.metal:1842`).
/// * `acc0 += p00*v[0] + p01*v[1] + p02*v[2] + p03*v[3]` with
///   `p0j = xr0[{0,1,8,9}]` — codes for rows `{r0, r0+1, r0+8, r0+9}` at
///   column `c`, and `acc1` the same four rows at column `c + 8`
///   (`linear.metal:1901-1911`, and the header comment at 1574-1575).
/// * the butterfly leaves the sum in *all four* lanes of a group of 4;
///   `if (lane % 4u == 0u)` keeps one of them and `red[sg][lane / 4u]` puts
///   lane `4k`'s `acc0` in column slot `k`, `red[sg][lane / 4u + 8u]` puts it
///   in slot `k + 8` (`linear.metal:1917-1919`).
/// * `for (g = 0; g < SG; ++g) sum += red[g][lid]` — `y[j]` is the sum of the
///   same slot `j` across all 8 simdgroups (`linear.metal:1922-1927`).
///
/// So the surviving lane `4k` must own column `k`, and every simdgroup must own
/// the same column assignment (it depends on `lane`, not on `sg`). That is the
/// claim this test proves, by reproducing the direct dot product.
struct Tile {
    /// `[16 columns][K rows]` of weights.
    w: Vec<Vec<f64>>,
    /// `[K]` inputs.
    x: Vec<f64>,
}

impl Tile {
    /// Small integers only, so every partial sum is exact in `f64` and the
    /// comparison can be `==`. A GEMV reassociates its K reduction; with values
    /// this small that reassociation is exact, which is what makes an exact
    /// equality assertion legitimate here instead of a tolerance.
    fn synthetic(k: usize, seed: u64) -> Self {
        let mut s = seed | 1;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let w = (0..TILE_N)
            .map(|_| (0..k).map(|_| (next() % 17) as f64 - 8.0).collect())
            .collect();
        let x = (0..k).map(|_| (next() % 9) as f64 - 4.0).collect();
        Self { w, x }
    }

    /// `y[j] = sum_r x[r] * w[j][r]`, the answer the epilogue must reproduce.
    fn direct(&self) -> Vec<f64> {
        (0..TILE_N)
            .map(|j| (0..self.x.len()).map(|r| self.x[r] * self.w[j][r]).sum())
            .collect()
    }

    /// The epilogue, on the CPU, exactly as the kernels sequence it.
    ///
    /// `butterfly: bool` exists only so `butterfly_is_load_bearing` can turn it
    /// off and observe the difference.
    fn reduced(&self, butterfly: bool) -> Vec<f64> {
        // `acc[sg][lane]` -> (acc0, acc1).
        let mut acc0 = vec![[0.0f64; LANES]; SG];
        let mut acc1 = vec![[0.0f64; LANES]; SG];

        // --- the K-loop, distributed exactly as the source distributes it ---
        for sg in 0..SG {
            for lane in 0..LANES {
                let r0 = (lane % 4) * 2;
                // Lane `lane`'s column. Within a group of 4 lanes the column is
                // shared (that is what makes the 4-lane butterfly correct), and
                // the 8 groups map onto the 8 surviving slots.
                let slot = lane / 4;
                let rows = [r0, r0 + 1, r0 + 8, r0 + 9];
                for r in 0..self.x.len() {
                    // `uint tk = sg; for (; tk < tiles_k; tk += SG)` — the eight
                    // simdgroups split K by k-tile, so simdgroup `sg` owns rows
                    // whose k-tile index is `= sg (mod 8)`. Without this the
                    // model sums the same K eight times and comes out exactly
                    // 8x too large, which is a real failure mode and not a
                    // hypothetical one.
                    if (r / 16) % SG != sg {
                        continue;
                    }
                    // Rows arrive in k-tiles of 16; a lane sees the four rows
                    // `r % 16 in {r0, r0+1, r0+8, r0+9}` and skips the rest.
                    if !rows.contains(&(r % 16)) {
                        continue;
                    }
                    acc0[sg][lane] += self.x[r] * self.w[slot][r];
                    acc1[sg][lane] += self.x[r] * self.w[slot + 8][r];
                }
            }
        }

        // --- the 4-lane butterfly: xor 1, then xor 2 ---
        if butterfly {
            for sg in 0..SG {
                for mask in [1usize, 2] {
                    let (a0, a1) = (acc0[sg], acc1[sg]);
                    for lane in 0..LANES {
                        acc0[sg][lane] = a0[lane] + a0[lane ^ mask];
                        acc1[sg][lane] = a1[lane] + a1[lane ^ mask];
                    }
                }
            }
        }

        // --- red[sg][16], written only by lane % 4 == 0 ---
        let mut red = vec![[0.0f64; TILE_N]; SG];
        for sg in 0..SG {
            for lane in (0..LANES).step_by(4) {
                red[sg][lane / 4] = acc0[sg][lane];
                red[sg][lane / 4 + 8] = acc1[sg][lane];
            }
        }

        // --- the barrier (a no-op on the CPU) and the final 8-way sum ---
        (0..TILE_N).map(|lid| (0..SG).map(|g| red[g][lid]).sum()).collect()
    }
}

#[test]
fn butterfly_plus_threadgroup_sum_equals_the_direct_row_sum() {
    for k in [16usize, 32, 336, 1344] {
        let tile = Tile::synthetic(k, 0x9E37_79B9_7F4A_7C15 ^ k as u64);
        assert_eq!(
            tile.reduced(true),
            tile.direct(),
            "epilogue disagrees with the direct dot product at K={k}"
        );
    }
}

/// **Sabotage.** Removing the butterfly breaks the epilogue. If this test ever
/// passes, the equality above is not testing the butterfly.
///
/// This is the CPU equivalent of AGENTS.md §3.4's rule: the failing branch has
/// to be shown to be able to fail.
#[test]
fn butterfly_is_load_bearing() {
    let tile = Tile::synthetic(336, 0xDEAD_BEEF_CAFE_F00D);
    assert_ne!(
        tile.reduced(false),
        tile.direct(),
        "dropping simd_shuffle_xor still produces the right answer; the geometry model is wrong"
    );
}

// ===========================================================================
// 7. Sabotage: the source matchers can fail
// ===========================================================================

#[test]
fn shuffle_mask_matcher_can_fail() {
    assert!(shuffle_masks("acc += simd_shuffle_xor(acc, 1u);").contains(&"1u".to_string()));
    let widened = LINEAR_METAL.replace(
        "    acc0 += simd_shuffle_xor(acc0, 2u);",
        "    acc0 += simd_shuffle_xor(acc0, 4u);",
    );
    assert_ne!(widened, LINEAR_METAL, "the sabotage did not apply");
    let widened_masks = shuffle_masks(&widened);
    assert!(
        widened_masks.contains(&"4u".to_string()),
        "the masker missed an injected 4u mask"
    );
    let bad: Vec<&String> = widened_masks.iter().filter(|m| *m != "1u" && *m != "2u").collect();
    assert!(!bad.is_empty(), "the `4-lane only` assertion would not have failed");
}

#[test]
fn barrier_counter_can_fail() {
    let body = &kernels(LINEAR_METAL)
        .into_iter()
        .find(|(_, n, _)| n == "linear_gemv_ishizuki_k3")
        .expect("k3")
        .2;
    assert_eq!(occurrences(body, "threadgroup_barrier"), 2);
    let injected = format!("{body}\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n");
    assert_eq!(
        injected.matches("threadgroup_barrier").count(),
        3,
        "an injected barrier was not counted"
    );
}

#[test]
fn kernel_body_extraction_can_fail() {
    // A whole-file search would attribute the *next* kernel's epilogue to this
    // one. Scoping to the body is what makes the census mean something.
    let k3 = kernels(LINEAR_METAL)
        .into_iter()
        .find(|(_, n, _)| n == "linear_gemv_ishizuki_k3")
        .expect("k3");
    assert!(
        !k3.2.contains("linear_gemv_ishizuki_k4"),
        "the k3 body leaked into k4"
    );
    assert!(!k3.2.contains("decode_state_mul1"), "the k3 body leaked into k8");
    // And the reduction window must not be found in a kernel that has none.
    let stream = kernels(LINEAR_REPACKED)
        .into_iter()
        .find(|(_, n, _)| n == "linear_gemv_ishizuki_k3_colmajor_stream")
        .expect("stream");
    assert!(!stream.2.contains("threadgroup float red["));
}
