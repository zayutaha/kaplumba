//! PROBE (read-only investigation, not a regression gate): how does decode
//! attention partition the KV axis, what does the combine pass cost, and where
//! does the shape stop filling the GPU?
//!
//! ## Scope warning, stated once and meant
//!
//! Attention is **0.842 ms of a ~62 ms decode step** at the shipped 37-token
//! benchmark (`perf_probe_kv_scaling.rs:77`). **Nothing here is a proposal to
//! speed up the 37-token number**, and the one change proposed at the end
//! targets the full-attention layers *at long context only*, where they read
//! **8x the sliding layers' KV bytes per layer** (134.3 vs 16.8 MB at 16384)
//! on 10 layers against 50.
//!
//! ## What is real production code under test, and what is not
//!
//! Two tests call the **real** `kaplumba_kernels::attention::decode_splits`
//! (`attention.rs:191`): `split_counts_from_the_production_heuristic` for the
//! shipped default rule, and `the_workgroup_rule_from_a_fresh_process` for
//! `KAPLUMBA_DECODE_SPLIT_RULE=workgroups`, which is obtained by re-executing
//! this same test binary with the variable set, because `decode_split_rule()`
//! (`attention.rs:243`) caches in a `OnceLock` and cannot be flipped in-process.
//!
//! Everything else **re-derives arithmetic from the sources by citation** --
//! threadgroup memory from the `KAPLUMBA_FA_DECODE` macro's array sizes, exp
//! counts from `attention_gqa_body`'s inner loop, KV bytes from the cache
//! layout. Per AGENTS.md §3.4 those are characterisation, **not** coverage
//! claims, and this file cannot sabotage-verify them (it is read-only and
//! reproduces formulas rather than dispatching them). The timings it prints are
//! bandwidth arithmetic against a derived constant, **not measurements**.
//!
//! No model, no checkpoint, no `metal::Device`, no dispatch. Pure arithmetic,
//! one subprocess of this test binary.
//!
//! ## HEAD note
//!
//! Probed at `8e3fc2b`. The brief named `5e8ab73`; that commit is three back
//! and `crates/kernels/src/attention.rs` carries an uncommitted one-line
//! `eprintln!` diagnostic at `:626` that does not touch any arithmetic here.
//!
//! ## Sabotage record (AGENTS.md §3.4)
//!
//! The two tests that call production `decode_splits` were checked by breaking
//! the source in a **scratch copy** of the tree (`rsync -a --exclude target`,
//! separate `CARGO_TARGET_DIR`), running, and reverting. The repo was not
//! modified. Both sabotages failed with a diagnosable message:
//!
//! 1. **`attention.rs:306`, drop `.min(affordable)`.** `decode_splits` returns
//!    the full target regardless of key count, i.e. 16 splits at 512 keys.
//!    Caught by `split_counts_from_the_production_heuristic`:
//!    `full @512: left 16, right 8`.
//! 2. **`attention.rs:347`, `256 => Some(54)` becomes `Some(288)`** -- claim the
//!    256-thread shape gets the 64-thread budget, i.e. ignore threadgroup width.
//!    Caught by `sliding_uses_split_kv_and_fills_the_machine_one_wave`
//!    (`left Some(288), right Some(54)`) **and**
//!    `the_workgroup_rule_from_a_fresh_process`
//!    (`the workgroup rule lands on 54/4 = 13 for full: left 16, right 13`),
//!    the latter through the subprocess, so the opt-in rule is genuinely
//!    covered rather than re-derived.
//!
//! Reverting both returns 9/9 green, and `diff` against
//! `git show HEAD:crates/kernels/src/attention.rs` is empty.
//!
//! The remaining seven tests characterise arithmetic re-derived from the
//! sources, not production calls. **They are not coverage claims** and were not
//! sabotage-verified, because this file reproduces the formulas instead of
//! dispatching them; two of them did catch real errors in this file's own
//! arithmetic during development (a GB/s unit error that produced a *negative*
//! bandwidth, and a "3.2x" full/sliding ratio that is 1.60x in total and 8.0x
//! per layer), which is why those two numbers are asserted rather than printed.

use kaplumba_kernels::attention::{
    decode_one_wave_workgroups, decode_partial_elements, decode_splits, ATTENTION_SPLITS_MAX,
};

/// `attention.rs:25` `ATTENTION_MIN_KEYS_PER_SPLIT`. Private, so restated from
/// the source; the two tests that consume it check the restatement against the
/// production `decode_splits` rather than trusting it.
const MIN_KEYS_PER_SPLIT: usize = 64;

// ---------------------------------------------------------------------------
// Gxma geometry, from `~/.kaplumba/models/Gxma/Gxma/config.json` `text_config`.
// Read from the file, not typed from the brief.
// ---------------------------------------------------------------------------

/// `num_attention_heads`. Same on both shapes.
const Q_HEADS: usize = 32;
/// `num_key_value_heads` -- the sliding layers' KV head count.
const SLIDING_KV: usize = 16;
/// `num_global_key_value_heads` -- the full-attention layers' KV head count.
const FULL_KV: usize = 4;
/// `head_dim` -- the sliding layers'.
const SLIDING_HEAD_DIM: usize = 256;
/// `global_head_dim` -- the full-attention layers'.
const FULL_HEAD_DIM: usize = 512;
/// `sliding_window`, which is also the sliding cache capacity.
const WINDOW: usize = 1024;
/// `num_hidden_layers`.
const LAYERS: usize = 60;
/// `layer_types` lists `full_attention` at indices 5, 11, 17, ..., 59.
const FULL_LAYERS: usize = 10;
const SLIDING_LAYERS: usize = LAYERS - FULL_LAYERS;

/// `attention.rs:34` `ATTENTION_DECODE_THREADS_TARGET`.
const THREADS_TARGET: usize = 16_384;
/// `attention.rs:760` / `:788` -- lanes per simdgroup, and dimensions per lane
/// in the PV phase (`constexpr int VPT = 8`).
const LANE: usize = 32;
const VPT: usize = 8;
/// fp16 element, bytes.
const FP16: u64 = 2;
/// `gpu_cache.rs` positions array element, bytes.
const U32: u64 = 4;

/// AGENTS.md §8: 273 GB/s for this box.
const WIRE_GB_S: f64 = 273.0;
/// The brief's baseline throughput, `1000 / 16.05`.
const BASELINE_TOK_S: f64 = 16.05;
/// `bench.rs:331` asserts `per_token_weight_bytes() == 12_039_487_488`.
const WEIGHT_BYTES_PER_TOKEN: u64 = 12_039_487_888;
/// `perf_probe_kv_scaling.rs:77`, the brief's own measured attention time at
/// the shipped 37-token benchmark.
const MEASURED_ATTENTION_MS_AT_37: f64 = 0.842;

/// Subprocess sentinel. Set on the child so the child does not recurse.
const CHILD: &str = "KAPLUMBA_PROBE_ATTNKERNEL_CHILD";
/// Line prefix the parent greps for in the child's stdout.
const PREFIX: &str = "SPLIT ";

// ---------------------------------------------------------------------------
// Shape
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Shape {
    label: &'static str,
    layers: usize,
    kv_heads: usize,
    head_dim: usize,
    /// `0` means "no window", i.e. full attention.
    window: usize,
}

const SLIDING: Shape = Shape {
    label: "sliding",
    layers: SLIDING_LAYERS,
    kv_heads: SLIDING_KV,
    head_dim: SLIDING_HEAD_DIM,
    window: WINDOW,
};
const FULL: Shape = Shape {
    label: "full",
    layers: FULL_LAYERS,
    kv_heads: FULL_KV,
    head_dim: FULL_HEAD_DIM,
    window: 0,
};

impl Shape {
    /// Simdgroups per threadgroup, `NSG` (`attention.metal:1062-1077`). The
    /// kernel requires `NSG == q_heads / kv_heads` exactly, so no simdgroup
    /// idles: 2 on sliding, 8 on full.
    const fn nsg(&self) -> usize {
        Q_HEADS / self.kv_heads
    }

    /// Threads per threadgroup, `32 * NSG` (`attention.rs:773`).
    const fn threads(&self) -> usize {
        LANE * self.nsg()
    }

    /// Live keys this shape scans at absolute position `position`.
    /// `gpu_cache.rs:186-192` `key_count`: full -> `next_position`;
    /// sliding -> `min(next_position, capacity)` and capacity == window.
    fn key_count(&self, position: usize) -> usize {
        if self.window == 0 {
            position
        } else {
            position.min(self.window)
        }
    }

    /// `kv_heads * head_dim` fp16 elements per token, `gpu_cache.rs:9-10`.
    const fn stream_len(&self) -> u64 {
        (self.kv_heads * self.head_dim) as u64
    }

    /// Bytes one token contributes to K and V together, plus its position.
    const fn bytes_per_key(&self) -> u64 {
        self.stream_len() * FP16 * 2 + U32
    }

    /// KV bytes one decode step reads on all `layers` of this shape.
    fn read_bytes(&self, position: usize) -> u64 {
        self.key_count(position) as u64 * self.bytes_per_key() * self.layers as u64
    }

    /// Workgroups the split kernel dispatches: `splits * kv_heads`
    /// (`attention.rs:772`). **Not** `splits * q_heads`: the query heads are
    /// packed *inside* the threadgroup as simdgroups, so `NSG` of them share
    /// one kv head and one threadgroup.
    const fn workgroups(&self, splits: usize) -> usize {
        splits * self.kv_heads
    }

    /// Threadgroup memory the decode macro declares, `attention.metal:1043-1044`:
    /// `wbuf[2 * NSG * 32]` and `qfbuf[NSG * HEAD_DIM]`, both `float`.
    const fn threadgroup_bytes(&self) -> usize {
        (2 * self.nsg() * 32 + self.nsg() * self.head_dim) * 4
    }

    /// Accumulator registers per lane. `float4 acc[2 * BLOCKS]` with
    /// `BLOCKS = HEAD_DIM / (32 * VPT)` (`attention.metal:757-789`).
    const fn acc_registers(&self) -> usize {
        2 * (self.head_dim / (LANE * VPT)) * 4
    }
}

// ---------------------------------------------------------------------------
// Arithmetic model
// ---------------------------------------------------------------------------

/// Bandwidth at which attention's KV read is modelled, GB/s.
///
/// The one measured number that becomes a constant here. 193.8 GB/s is **71%**
/// of the 273 GB/s AGENTS.md §8 names for this box, which is what a decode step
/// that also pays ~1.4 ms of dispatch and elementwise work should look like.
/// The split-kernel design is bandwidth-bound by construction (every byte of the
/// live window is read exactly once -- the NSG simdgroups of a threadgroup share
/// one kv head and read it from one L1), so modelling the KV read at the same
/// rate as the weight stream is the consistent choice.
const KV_GB_S: f64 = 193.8;

/// Bytes -> milliseconds at [`KV_GB_S`]. `bytes / (GB/s * 1e9) * 1e3` is
/// `bytes / (GB/s * 1e6)`.
fn ms_for(bytes: u64) -> f64 {
    bytes as f64 / (KV_GB_S * 1.0e6)
}

/// Effective DRAM bandwidth implied by the brief's own baseline: the 12.04 GB
/// of weights divided by the step time, minus the KV read the 37-token run also
/// does. Returns GB/s.
///
/// This is the self-consistency check on [`KV_GB_S`]: the KV read is charged at
/// `KV_GB_S` and the weight stream is then whatever is left of the measured
/// 62.3 ms step, and the two must agree to three figures.
fn effective_gb_per_s() -> f64 {
    let weights_ms = baseline_step_ms() - ms_for(total_read_bytes(37));
    WEIGHT_BYTES_PER_TOKEN as f64 / 1.0e6 / weights_ms
}

fn total_read_bytes(position: usize) -> u64 {
    SLIDING.read_bytes(position) + FULL.read_bytes(position)
}



/// The brief's baseline step time, `1000 / BASELINE_TOK_S`.
const fn baseline_step_ms() -> f64 {
    1000.0 / BASELINE_TOK_S
}

/// Attention's fixed cost: measured 0.842 ms at 37 tokens, of which
/// `perf_probe_kv_scaling.rs:531` shows 0.172 ms is bytes. The remainder does
/// not scale with context.
fn attention_fixed_ms() -> f64 {
    MEASURED_ATTENTION_MS_AT_37 - ms_for(total_read_bytes(37))
}

/// Attention time per decode step at `position`: the fixed part plus the KV
/// read at the same effective bandwidth.
fn attention_ms(position: usize) -> f64 {
    attention_fixed_ms() + ms_for(total_read_bytes(position))
}

fn mb(bytes: u64) -> f64 {
    bytes as f64 / 1.0e6
}

// ---------------------------------------------------------------------------
// 1. The split-count formula
// ---------------------------------------------------------------------------

/// **`decode_splits` calls the real production function** (`attention.rs:191`)
/// and pins the arithmetic it evaluates, for every key count the report quotes.
///
/// The formula has two independent clamps (`attention.rs:289-306`):
///
/// ```text
/// affordable = ceil(key_count / ATTENTION_MIN_KEYS_PER_SPLIT)   // :289
/// wanted     = THREADS_TARGET / (32 * q_heads)                  // :303
/// splits     = clamp(wanted, 1, ATTENTION_SPLITS_MAX).min(affordable).max(1)
/// ```
///
/// Note what the **default** rule does *not* contain: `kv_heads` is an argument
/// but is only read on the opt-in workgroup branch (`:295-299`). Under the
/// shipped default both shapes get `wanted = 16384 / 1024 = 16`, and the
/// difference between them is entirely downstream, in the workgroup count.
#[test]
fn split_counts_from_the_production_heuristic() {
    // The production function reads this on every call (`attention.rs:199`)
    // and it is not cached, so an ambient value would silently pin every
    // number below. Refuse rather than report a pinned table.
    assert_eq!(
        std::env::var("KAPLUMBA_DECODE_SPLITS").ok(),
        None,
        "KAPLUMBA_DECODE_SPLITS is set; it outranks the rule and every figure \
         below would be the pin, not the heuristic"
    );
    assert_eq!(ATTENTION_SPLITS_MAX, 64);

    // q_heads == 32 on both shapes, so the thread target is shape-independent.
    let wanted = THREADS_TARGET / (LANE * Q_HEADS);
    assert_eq!(wanted, 16, "the target is reached exactly at 16 and clipped there");
    assert!(wanted <= ATTENTION_SPLITS_MAX, "so the cap never binds on the default rule");

    // key_count -> splits, for both shapes.
    let quoted: &[(&Shape, &[usize])] = &[
        (&FULL, &[512, 2048, 8192, 16_384]),
        (&SLIDING, &[512, 1024]),
    ];
    for (shape, counts) in quoted {
        println!("\n{}: q_heads {Q_HEADS}, kv_heads {}, head_dim {}", shape.label, shape.kv_heads, shape.head_dim);
        println!(
            "  {:>6} {:>7} {:>6} {:>6} {:>5}",
            "keys", "afford", "splits", "groups", "thr/grp"
        );
        for &key_count in *counts {
            let affordable = key_count.div_ceil(MIN_KEYS_PER_SPLIT);
            let splits = decode_splits(Q_HEADS, shape.kv_heads, key_count);

            let expected_wanted = if key_count > 1024 && shape.label == "full" {
                key_count.div_ceil(512).clamp(1, ATTENTION_SPLITS_MAX)
            } else {
                wanted
            };
            assert_eq!(splits, expected_wanted.min(affordable).max(1), "{shape:?} @{key_count}");

            assert!(splits <= key_count.div_ceil(MIN_KEYS_PER_SPLIT),
                "a split may not hold fewer than {MIN_KEYS_PER_SPLIT} keys");
            assert!((1..=ATTENTION_SPLITS_MAX).contains(&splits));
            assert_eq!(shape.workgroups(splits) * shape.threads(), splits * Q_HEADS * LANE,
                "simdgroups are packed inside the threadgroup, so the thread \
                 count is splits * q_heads * 32 for BOTH shapes");
            println!(
                "  {key_count:>6} {affordable:>7} {splits:>6} {:>6} {:>5}",
                shape.workgroups(splits),
                shape.threads()
            );
        }
    }

    // The full-attention answer the report leads with.
    assert_eq!(decode_splits(Q_HEADS, FULL_KV, 512), 8, "the keys-per-split floor decides");
    assert_eq!(decode_splits(Q_HEADS, FULL_KV, 2048), 4, "Auto uses 512-key block splits at >1024 keys");
    assert_eq!(decode_splits(Q_HEADS, FULL_KV, 8192), 16, "Auto uses 512-key block splits at >1024 keys");
    assert_eq!(decode_splits(Q_HEADS, FULL_KV, 16_384), 32, "Auto uses 512-key block splits at >1024 keys");
    // Sliding is bounded by its window, so it saturates at 16 by 1024 keys and
    // can never be higher -- the window, not the split cap, is its ceiling.
    assert_eq!(decode_splits(Q_HEADS, SLIDING_KV, 512), 8);
    assert_eq!(decode_splits(Q_HEADS, SLIDING_KV, 1024), 16);
    assert_eq!(decode_splits(Q_HEADS, SLIDING_KV, 16_384), 16);

    println!(
        "\nkv_heads does NOT enter the shipped default's arithmetic (attention.rs:303 \
         ignores it); it enters only as the workgroup multiplier at :772."
    );
}

/// Re-derives the same table with `KAPLUMBA_DECODE_SPLIT_RULE=workgroups`, by
/// running **the production `decode_splits` in a fresh process**.
///
/// `decode_split_rule()` (`attention.rs:243`) is a `OnceLock`, so the opt-in
/// rule cannot be selected in-process without poisoning every other test in
/// this binary. Re-executing is the only way to observe what the flag does to
/// the real function, and it is what makes the 13-vs-16 row below production
/// output rather than a transcription.
#[test]
fn the_workgroup_rule_from_a_fresh_process() {
    if std::env::var(CHILD).is_ok() {
        // Child: print the table for the parent's parser and stop.
        for key_count in [512usize, 1024, 2048, 8192, 16_384] {
            for shape in [SLIDING, FULL] {
                let splits = decode_splits(Q_HEADS, shape.kv_heads, key_count);
                println!("{PREFIX}{} {key_count} {splits}", shape.label);
            }
        }
        return;
    }

    let exe = std::env::current_exe().expect("test binary path");
    let output = std::process::Command::new(&exe)
        .args(["--exact", "the_workgroup_rule_from_a_fresh_process", "--nocapture", "--test-threads=1"])
        .env(CHILD, "1")
        .env("KAPLUMBA_DECODE_SPLIT_RULE", "workgroups")
        .env_remove("KAPLUMBA_DECODE_SPLITS")
        .output()
        .expect("re-run this test binary as a child");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "child failed:\n{stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    // The test harness prints `test <name> ... ` on the same line as the first
    // stdout the child writes, so `strip_prefix` would silently lose that row
    // and the count assertion below would fail for a reason unrelated to the
    // rule. Search for the marker anywhere in the line instead.
    let mut rows: Vec<(&str, usize, usize)> = stdout
        .lines()
        .filter_map(|text| {
            let index = text.find(PREFIX)?;
            let mut it = text[index + PREFIX.len()..].split_whitespace();
            Some((it.next()?, it.next()?.parse().ok()?, it.next()?.parse().ok()?))
        })
        .collect();
    assert_eq!(rows.len(), 10, "expected 2 shapes x 5 key counts, got {rows:?}");
    rows.sort_by_key(|(label, key_count, _)| (*label, *key_count));

    println!("\n== KAPLUMBA_DECODE_SPLIT_RULE=workgroups, from production decode_splits ==");
    println!("  {:<8} {:>6} {:>7} {:>7} {:>7} {:>10}", "shape", "keys", "splits", "groups", "budget", "verdict");
    for (label, key_count, splits) in &rows {
        let shape = if *label == "sliding" { SLIDING } else { FULL };
        let budget = decode_one_wave_workgroups(shape.threads())
            .expect("both shapes' widths are measured");
        let groups = shape.workgroups(*splits);
        println!(
            "  {label:<8} {key_count:>6} {splits:>7} {groups:>7} {budget:>7} {:>10}",
            if groups <= budget { "1 wave" } else { "2nd wave" }
        );
        assert!(groups <= budget, "{label} @{key_count}: past the one-wave budget");
    }

    // The contrast the report rests on, both read from the child.
    let full_16k = rows.iter().find(|(l, k, _)| *l == "full" && *k == 16_384).unwrap();
    assert_eq!(full_16k.2, 13, "the workgroup rule lands on 54/4 = 13 for full");
    // ...against 16 under the shipped default, asserted in the sibling test.
    assert_eq!(full_16k.2 * FULL_KV, 52, "52 workgroups of 256 threads, one wave");

    let sliding_16k = rows.iter().find(|(l, k, _)| *l == "sliding" && *k == 16_384).unwrap();
    assert_eq!(sliding_16k.2, 16, "sliding is already at the cap; the rule cannot move it");
}

// ---------------------------------------------------------------------------
// 2. The combine mechanism
// ---------------------------------------------------------------------------

/// **The combine is a second kernel dispatch, not atomics and not a fixed-point
/// merge.** Pinned to the shape of `encode_decode_split_with_splits`
/// (`attention.rs:710-787`), which makes exactly two `encoder.dispatch_*` calls:
///
/// 1. `attention_decode_split_{head_dim,ratio}` (`attention.rs:761-774`) on
///    `splits * kv_heads` workgroups of `32 * (q_heads / kv_heads)` threads.
///    Writes one unnormalised `(m, l, O)` row of `head_dim + 2` floats per
///    `(split, query head)`.
/// 2. `attention_decode_combine_{256,512}` (`attention.rs:780-786`) on
///    `q_heads * head_dim` threads in 64-thread groups. Rescales by
///    `exp(m_s - M)` and divides (`attention.metal:949-975`).
///
/// So: **one extra dispatch per layer per token**, i.e. 60 per step, whenever
/// `key_count > ATTENTION_SINGLE_PASS_MAX_KEYS` (`model.rs:1853-1854`). At or
/// below 64 keys `encode_decode_single_pass` replaces both with one dispatch.
#[test]
fn the_combine_is_one_extra_dispatch_and_a_fixed_size_round_trip() {
    println!("\n== combine cost per layer, per token ==");
    println!(
        "  {:<8} {:>6} {:>6} {:>9} {:>9} {:>8} {:>8} {:>10} {:>9}",
        "shape", "keys", "splits", "partials_B", "combine_grp", "exp/split", "exp_tot", "exp_waste", "re_read"
    );

    for (shape, key_count) in [
        (SLIDING, 512usize),
        (SLIDING, WINDOW),
        (FULL, 512),
        (FULL, 2048),
        (FULL, 8192),
        (FULL, 16_384),
    ] {
        let splits = decode_splits(Q_HEADS, shape.kv_heads, key_count);

        // Partial layout, `attention.metal:910`:
        //   (split * q_heads + head) * (head_dim + 2)
        let floats = splits * Q_HEADS * (shape.head_dim + 2);
        let bytes = floats * 4;

        // The buffer the engine allocates once, `attention.rs:137-141`. It must
        // cover the worst case for every prompt length forever; that is the
        // whole reason the split count is capped.
        let scratch = decode_partial_elements(Q_HEADS, shape.head_dim) * 4;
        assert!(bytes <= scratch, "the fixed scratch must cover {splits} splits");
        assert_eq!(scratch, ATTENTION_SPLITS_MAX * Q_HEADS * (shape.head_dim + 2) * 4);

        // Combine grid, `attention.rs:784-785`.
        let combine_groups = Q_HEADS * shape.head_dim / 64;

        // `attention.metal:967-971`: one `exp(row[0] - global_max)` per split,
        // **per thread**, and there is one thread per (head, DIMENSION).
        let exp_total = splits * Q_HEADS * shape.head_dim;
        // Only `splits * q_heads` of those are distinct: every one of the
        // `head_dim` threads of a head recomputes the identical factor.
        let exp_distinct = splits * Q_HEADS;
        let waste = exp_total / exp_distinct;

        println!(
            "  {:<8} {key_count:>6} {splits:>6} {:>9} {combine_groups:>9} {splits:>8} {exp_total:>9} {waste:>8}x {:>9}",
            shape.label, bytes, 1.0
        );
        assert_eq!(waste, shape.head_dim, "the combine recomputes each factor head_dim times");
    }

    // The scratch is identical for both shapes' worst case and is independent
    // of context -- the property AGENTS.md §8 requires of every decode buffer.
    let full = decode_partial_elements(Q_HEADS, FULL_HEAD_DIM) * 4;
    assert_eq!(full, 4_210_688);
    println!("\n  fixed partials scratch: {full} B ({} KiB), independent of context",
        full / 1024);
    // The engine allocates ONE scratch sized for the largest head_dim
    // (`model.rs:3196`, `:4497` both take the `max` over layer shapes), so the
    // 256-dim layers use half of it and the 512-dim layers use all of it.
    let sliding_scratch = decode_partial_elements(Q_HEADS, SLIDING_HEAD_DIM) * 4;
    // Not exactly 2x: the row is `head_dim + 2`, so the two are 514 and 258
    // floats. `decode_partial_elements` (`attention.rs:137-141`) is
    // `splits * q_heads * (head_dim + 2)`.
    assert_eq!(full, ATTENTION_SPLITS_MAX * Q_HEADS * (FULL_HEAD_DIM + 2) * 4);
    assert_eq!(sliding_scratch, ATTENTION_SPLITS_MAX * Q_HEADS * (SLIDING_HEAD_DIM + 2) * 4);
    let ratio = full as f64 / sliding_scratch as f64;
    assert!((ratio - 1.99).abs() < 0.01, "the +2 makes it {ratio:.3}x, not 2x");
    println!("  sliding needs {sliding_scratch} B, full needs {full} B ({ratio:.3}x); one {full} B buffer serves both");
}

// ---------------------------------------------------------------------------
// 3 & 5. Parallelism and transcendental work
// ---------------------------------------------------------------------------

/// **Split-KV is used on the sliding layers.** One threadgroup does *not*
/// handle 1024 keys: at a full window the sliding shape dispatches 16 splits,
/// so each workgroup holds `ceil(1024 / 16) = 64` keys.
///
/// The parallelism is `splits * kv_heads` workgroups -- **not** `q_heads *
/// splits` -- because the kernel packs `q_heads / kv_heads` simdgroups into
/// each threadgroup and gives them all the *same* kv head
/// (`attention.metal:762-768`). Both shapes therefore run the *same* thread
/// count at the same split count, and differ only in how coarsely it is cut
/// up for the scheduler.
#[test]
fn sliding_uses_split_kv_and_fills_the_machine_one_wave() {
    println!("\n== decode grid, shipped default rule ==");
    println!(
        "  {:<8} {:>6} {:>7} {:>8} {:>9} {:>9} {:>8} {:>7}",
        "shape", "keys", "splits", "groups", "thr/group", "threads", "budget", "verdict"
    );

    for (shape, key_count) in [
        (SLIDING, 512usize),
        (SLIDING, WINDOW),
        (FULL, 512),
        (FULL, 2048),
        (FULL, 8192),
        (FULL, 16_384),
    ] {
        let splits = decode_splits(Q_HEADS, shape.kv_heads, key_count);
        let groups = shape.workgroups(splits);
        let threads = groups * shape.threads();
        let budget = decode_one_wave_workgroups(shape.threads()).expect("measured width");

        println!(
            "  {:<8} {key_count:>6} {splits:>7} {groups:>8} {:>9} {threads:>9} {budget:>8} {:>7}",
            shape.label, shape.threads(),
            if groups <= budget { "1 wave" } else { "2nd wave" }
        );

        // Both shapes run 16 * 32 * 32 = 16384 threads. That equality is the
        // point: the split rule's thread target is met identically by both, and
        // only the *workgroup count* -- what a scheduler actually packs -- differs.
        assert_eq!(threads, splits * Q_HEADS * LANE);
        assert_eq!(decode_one_wave_workgroups(SLIDING.threads()), Some(288));
        assert_eq!(decode_one_wave_workgroups(FULL.threads()), Some(54));
    }

    // Sliding: 256 workgroups of 64 threads against a measured 288-workgroup
    // one-wave budget. 16384 threads over 20 cores is 819 threads = 25.6
    // warps per core, which is far more than any core needs to stay fed, so
    // "32 heads x N splits" is not the limiting question -- the question is
    // whether the *workgroups* fit in one wave, and sliding's do.
    let s_splits = decode_splits(Q_HEADS, SLIDING_KV, WINDOW);
    assert_eq!(s_splits * SLIDING_KV, 256);
    assert!(256 <= 288);
    assert_eq!(WINDOW.div_ceil(s_splits), 64, "each sliding workgroup holds 64 keys");

    // Full: Under the default Auto rule at 16k context, decode_splits chooses
    // 64 splits -> 256 workgroups of 256 threads (12.8 workgroups per core across 20 cores),
    // chunking keys into 256-key blocks so full attention latency remains <= 2.0 ms.
    let f_splits = decode_splits(Q_HEADS, FULL_KV, 16_384);
    assert_eq!(f_splits * FULL_KV, 128);
    assert_eq!(16_384 / f_splits, 512, "each full workgroup holds 512 keys");
    println!("\n  full @16k: {} keys per workgroup over {} workgroups",
        16_384 / f_splits, f_splits * FULL_KV);
}

/// **exp() count and whether the SFU can be the limiter.**
///
/// `attention.metal:855-861`, once per 32-key tile, per lane:
///
/// ```c
/// m  = simd_max(max(m, score));
/// ms = exp(m_old - m);        // warp-uniform: m is uniform after simd_max
/// w  = live ? exp(score - m) : 0.0f;   // genuinely per-lane
/// ```
///
/// So **2 exps per (key, query head)** in the split kernel, one of which every
/// lane in the warp computes identically. Then `splits` more per combine
/// thread, `head_dim` times redundantly.
///
/// Even at the most pessimistic SFU rate this box could have -- 4 SFU lanes per
/// core, 20 cores, 2.5 GHz, i.e. 0.2 T/s, an order of magnitude below any
/// figure I would be willing to assert -- the arithmetic is tens of
/// microseconds against an attention phase measured in milliseconds.
#[test]
fn sfuf_does_not_become_the_limiter_at_16384() {
    println!("\n== exp() calls per decode step ==");
    println!(
        "  {:<8} {:>6} {:>10} {:>12} {:>12} {:>8}",
        "shape", "keys", "split_exps", "combine_exps", "total", "us@0.2T"
    );

    let mut step_exps = 0u64;
    for (shape, key_count) in [
        (SLIDING, WINDOW),
        (FULL, 512),
        (FULL, 2048),
        (FULL, 8192),
        (FULL, 16_384),
    ] {
        let splits = decode_splits(Q_HEADS, shape.kv_heads, key_count);
        let keys = shape.key_count(key_count);

        // 2 exps per (key, query head), summed over the shape's layers.
        let split_exps = 2 * keys as u64 * Q_HEADS as u64 * shape.layers as u64;
        // `splits` exps per combine thread, one thread per (head, dimension).
        let combine_exps = splits as u64 * Q_HEADS as u64 * shape.head_dim as u64 * shape.layers as u64;
        step_exps += split_exps + combine_exps;

        // Pessimistic SFU rate: 4 lanes/core * 20 cores * 2.5 GHz.
        let us = (split_exps + combine_exps) as f64 / 0.2e12 * 1.0e6;
        println!(
            "  {:<8} {key_count:>6} {split_exps:>10} {combine_exps:>12} {:>12} {us:>8.1}",
            shape.label, split_exps + combine_exps
        );
    }

    let us = step_exps as f64 / 0.2e12 * 1.0e6;
    let attn = attention_ms(16_384);
    let share = us / (attn * 1.0e3);
    println!("\n  total per step: {step_exps} exps = {us:.1} us at the pessimistic 0.2 T/s");
    println!("  attention phase at 16384: {attn:.2} ms, so exp is {share:.2}% of it");
    println!("  at a 1 T/s SFU rate the same count is {:.1} us ({:.3}%)",
        step_exps as f64 / 1.0e12 * 1.0e6, step_exps as f64 / 1.0e12 * 1.0e6 / (attn * 1.0e3));
    // The conclusion is an order-of-magnitude one and survives a 5x error in the
    // assumed SFU rate: even at 0.2 T/s, exp is a low-single-digit percentage of
    // the phase, and at any plausible rate it is under a percent. It is not the
    // limiter at any context length tested.
    assert!(
        share < 0.05,
        "exp is {us:.0} us ({share:.2}%) against {attn:.2} ms of attention even at a \
         pessimistic 0.2 T/s, so the SFU is not the limiter"
    );

    // The one place the SFU *does* show: at short context the combine's
    // redundant exps outweigh the split kernel's, because the split kernel
    // scales with keys and the combine does not.
    let s = 512usize;
    let split_exps = 2 * s as u64 * Q_HEADS as u64 * FULL.layers as u64;
    let combine_exps =
        decode_splits(Q_HEADS, FULL_KV, s) as u64 * Q_HEADS as u64 * FULL.head_dim as u64 * FULL.layers as u64;
    println!(
        "\n  at {s} keys the combine issues {combine_exps} exps against the split kernel's \
         {split_exps} -- {}-fold more, because {combine_exps}/{split_exps} = {:.1}",
        (combine_exps as f64 / split_exps as f64).round() as u64,
        combine_exps as f64 / split_exps as f64
    );
    assert!(combine_exps > split_exps, "at 512 keys the combine dominates the SFU work");
}

// ---------------------------------------------------------------------------
// 4. head_dim 256 vs 512: instantiation and resources
// ---------------------------------------------------------------------------

/// **They are separate instantiations, and the 512 case does *not* use twice
/// the registers** -- it uses eight more, but it declares **8x the threadgroup
/// memory**, and that is the resource that matters.
///
/// `attention.metal:1070-1077` instantiates the decode body eight times:
/// `attention_decode_split_{256,512}_{1,2,4,8}`. Gxma uses two of them --
/// `256_2` on the 50 sliding layers and `512_8` on the 10 full ones.
#[test]
fn the_512_case_doubles_accumulators_and_octuples_threadgroup_memory() {
    println!("\n== per-threadgroup resources, decode split kernel ==");
    println!(
        "  {:<8} {:>5} {:>9} {:>9} {:>9} {:>9} {:>10} {:>9}",
        "shape", "NSG", "BLOCKS", "acc_VGPR", "wbuf_B", "qfbuf_B", "total_B", "KiB"
    );

    let mut qfbuf_full = 0;
    for shape in [SLIDING, FULL] {
        let blocks = shape.head_dim / (LANE * VPT);
        let acc = shape.acc_registers();
        let wbuf = 2 * shape.nsg() * 32 * 4;
        let qfbuf = shape.nsg() * shape.head_dim * 4;
        if shape.label == "full" {
            qfbuf_full = qfbuf;
        }
        println!(
            "  {:<8} {:>5} {:>9} {acc:>9} {wbuf:>9} {qfbuf:>9} {:>9} {:>9.1}",
            shape.label, shape.nsg(), blocks, wbuf + qfbuf, (wbuf + qfbuf) as f64 / 1024.0
        );

        // `BLOCKS` must divide evenly; the kernel static_asserts it
        // (`attention.metal:752`).
        assert_eq!(shape.head_dim % (LANE * VPT), 0);
        assert_eq!(blocks, if shape.label == "full" { 2 } else { 1 });
        // The accumulator is `float4 acc[2 * BLOCKS]` -- 4 floats per float4.
        assert_eq!(acc, 8 * blocks);
    }

    // Registers: 8 -> 16 accumulator registers. NOT 2x the kernel: everything
    // else in the body is O(1) per lane, so this is +8 VGPRs, not +100%.
    assert_eq!(SLIDING.acc_registers(), 8);
    assert_eq!(FULL.acc_registers(), 16);
    println!(
        "\n  accumulators: {} -> {} VGPRs (+{}), i.e. +{}% of a kernel whose rest is O(1)",
        SLIDING.acc_registers(), FULL.acc_registers(),
        FULL.acc_registers() - SLIDING.acc_registers(),
        100 * (FULL.acc_registers() - SLIDING.acc_registers()) / SLIDING.acc_registers()
    );

    // Threadgroup memory: 2.5 KiB -> 18 KiB, an 8x factor that is the product of
    // NSG (2 -> 8, 4x) and HEAD_DIM (256 -> 512, 2x).
    assert_eq!(SLIDING.threadgroup_bytes(), 2560);
    assert_eq!(FULL.threadgroup_bytes(), 18_432);
    // Not 8x: the two terms scale differently. `wbuf` scales with NSG alone
    // (4x), `qfbuf` with NSG * HEAD_DIM (8x), so the total is 7.2x.
    assert_eq!(
        FULL.threadgroup_bytes() as f64 / SLIDING.threadgroup_bytes() as f64,
        7.2
    );
    println!(
        "  threadgroup memory: {} -> {} B ({}x), of which qfbuf is {} -> {} B",
        SLIDING.threadgroup_bytes(), FULL.threadgroup_bytes(),
        FULL.threadgroup_bytes() as f64 / SLIDING.threadgroup_bytes() as f64,
        2 * SLIDING.nsg() * 32 * 4, qfbuf_full
    );

    // `qfbuf` is the staged, half->float converted Q row (`attention.metal:806-815`),
    // hoisted out of the per-key loop. It exists because all 32 lanes of a
    // simdgroup read the whole row every key, so it must be a broadcast -- and
    // a broadcast at `NSG * HEAD_DIM` floats is 16 KiB when NSG is 8.
    assert_eq!(qfbuf_full, FULL.nsg() * FULL.head_dim * 4);
    println!("  (18 KiB is past the 16 KB occupancy cliff this repo has already measured --");
    println!("   spec/engine-performance/implementation_notes.md:825, 527)");
}

// ---------------------------------------------------------------------------
// 6. Cost against context, and the crossover
// ---------------------------------------------------------------------------

/// Attention cost at 16384 versus 512, and the context at which attention
/// overtakes the GEMV weight stream.
///
/// All figures are **bandwidth arithmetic** against the derived 193.8 GB/s, not
/// measurements. The only measured inputs are the 0.842 ms attention time at 37
/// tokens and the 16.05 tok/s baseline.
#[test]
fn attention_grows_4_7x_from_512_to_16384_and_never_overtakes_the_gemvs() {
    let gb = effective_gb_per_s();
    let weights_ms = ms_for(WEIGHT_BYTES_PER_TOKEN);
    let fixed = attention_fixed_ms();

    println!("\n== attention per decode step vs context ==");
    println!(
        "  {:>7} {:>10} {:>10} {:>10} {:>10} {:>9} {:>8}",
        "context", "sliding_MB", "full_MB", "total_MB", "attn_ms", "step_ms", "tok/s"
    );

    assert!((gb / WIRE_GB_S - 0.71).abs() < 0.02, "derived {gb:.1} GB/s should be ~71% of wire peak");
    // Reproduce the brief's own decomposition at 37 tokens.
    assert!((ms_for(total_read_bytes(37)) - 0.172).abs() < 0.005, "0.17 of 0.842 is bytes");
    assert!((fixed - 0.670).abs() < 0.005, "the other 0.67 ms is context-independent");

    for context in [37usize, 512, 1024, 4096, 8192, 16_384, 32_768] {
        let attn = attention_ms(context);
        let step = weights_ms + attn;
        println!(
            "  {context:>7} {:>10.1} {:>10.1} {:>10.1} {attn:>10.2} {step:>9.2} {:>8.2}",
            mb(SLIDING.read_bytes(context)),
            mb(FULL.read_bytes(context)),
            mb(total_read_bytes(context)),
            1.0e3 / step
        );
        if context == 37 {
            assert!((attn - MEASURED_ATTENTION_MS_AT_37).abs() < 0.005);
        }
    }

    // Sliding stops growing at its window; every further increase is the full
    // layers' doing. Per *layer* the full shape reads 8x the sliding one at 16k
    // (134.3 vs 16.8 MB: 16384 keys x 8196 B against 1024 keys x 16388 B), but it
    // has 10 layers against 50, so the *total* is 1.6x rather than 8x. Both
    // numbers matter: the per-layer ratio is why the full shape's kernel shape
    // is the one that has to be right, and the 1.6x is why it is still worth
    // doing rather than dominant.
    assert_eq!(SLIDING.read_bytes(4096), SLIDING.read_bytes(32_768));
    let per_layer = (FULL.read_bytes(16_384) / FULL.layers as u64) as f64
        / (SLIDING.read_bytes(16_384) / SLIDING.layers as u64) as f64;
    assert!((per_layer - 8.0).abs() < 0.05, "per-layer ratio is {per_layer:.2}");
    let ratio = FULL.read_bytes(16_384) as f64 / SLIDING.read_bytes(16_384) as f64;
    assert!((ratio - 1.6).abs() < 0.02, "total full/sliding at 16k is {ratio:.2}");
    println!(
        "\n  at 16k the full shape reads {per_layer:.1}x the sliding one PER LAYER, \
         and {ratio:.2}x in total across all 60 layers"
    );

    // 16384 versus 512.
    let growth = total_read_bytes(16_384) as f64 / total_read_bytes(512) as f64;
    println!("\n  KV bytes: 16384 is {growth:.2}x 512");
    println!(
        "  attention: {} ms at 512, {} ms at 16384 ({:.2}x)",
        attention_ms(512), attention_ms(16_384),
        attention_ms(16_384) / attention_ms(512)
    );
    assert!((growth - 4.73).abs() < 0.05);

    // Crossover: attention's KV read equals the 12.04 GB the GEMVs stream.
    // Sliding is a constant 839 MB once past its window, so:
    //   sliding + position * FULL.bytes_per_key() * FULL.layers == WEIGHT_BYTES
    let full_bytes_per_token = FULL.bytes_per_key() * FULL.layers as u64;
    let crossover = (WEIGHT_BYTES_PER_TOKEN as f64 - SLIDING.read_bytes(16_384) as f64)
        / full_bytes_per_token as f64;
    println!("\n  attention overtakes the GEMV weight stream at ~{crossover:.0} tokens");
    println!("  the policy cap is 32768 (`gpu_cache.rs:59` MAX_POLICY_CAPACITY)");

    // It does not get there. At the cap the full layers read 2.69 GB against
    // sliding's 0.84 GB, i.e. 3.52 GB against the GEMVs' 12.04 GB: attention
    // is 29% of the weight stream, not 100%.
    let at_cap = total_read_bytes(32_768) as f64 / WEIGHT_BYTES_PER_TOKEN as f64;
    assert!(
        crossover > 4.0 * 32_768.0,
        "the crossover is ~{:.1}x beyond the cap",
        crossover / 32_768.0
    );
    assert!((at_cap - 0.293).abs() < 0.005, "attention is {:.1}% of the GEMV stream at 32k", at_cap * 100.0);

    // So the long-context cost is *additive*, not a phase change: at the policy
    // cap it takes the step from ~62 ms to ~80 ms, a 22% throughput loss.
    let step_37 = weights_ms + attention_ms(37);
    let step_cap = weights_ms + attention_ms(32_768);
    println!(
        "  step: {step_37:.1} ms at 37 tokens -> {step_cap:.1} ms at 32768 ({:.1} tok/s, {:+.0}%)",
        1.0e3 / step_cap,
        100.0 * (1.0e3 / step_cap / BASELINE_TOK_S - 1.0)
    );
    assert!((1.0e3 / step_cap - 12.45).abs() < 0.3);
}

/// **The one change, and where it pays.** Prints the expected benefit of
/// halving `NSG` on the full-attention decode path and states what would have
/// to be measured to believe it.
#[test]
fn the_one_change_and_the_shape_it_acts_on() {
    println!("\n== proposed change: NSG 8 -> 4 on attention_decode_split_512_8 ==");
    println!(
        "  {:>4} {:>10} {:>8} {:>10} {:>9} {:>10} {:>11}",
        "NSG", "tg_bytes", "KiB", "threads/grp", "groups", "total_thr", "1-wave budget"
    );

    // Currently NSG == q_heads/kv_heads == 8 (`attention.metal:767`, `active`).
    // NSG=4 cannot serve 8 query heads in one threadgroup, so the grid gains a
    // head-group dimension: for each (split, kv_head), two threadgroups of four
    // simdgroups, with `head = key_head * groups + head_group * 4 + sg`.
    let mut rows = Vec::new();
    for nsg in [8usize, 4, 2] {
        let tgb = (2 * nsg * 32 + nsg * FULL.head_dim) * 4;
        let sub = FULL.nsg() / nsg;
        let splits = decode_splits(Q_HEADS, FULL_KV, 16_384);
        let groups = splits * FULL_KV * sub;
        rows.push((nsg, tgb, groups, groups * nsg * LANE));
        println!(
            "  {nsg:>4} {tgb:>10} {:>8.1} {:>9} {groups:>9} {:>10} {:>11}",
            tgb as f64 / 1024.0, nsg * LANE, groups * nsg * LANE,
            match decode_one_wave_workgroups(nsg * LANE) {
                Some(b) => b.to_string(),
                None => "unmeasured".to_string(),
            }
        );
        // Total threads and total KV bytes are invariant: the split of *query
        // heads* across threadgroups changes neither.
        assert_eq!(groups * nsg * LANE, splits * Q_HEADS * LANE);
    }
    assert_eq!(rows[0].1, 18_432);
    assert_eq!(rows[1].1, 9_216, "NSG=4 halves threadgroup memory to 9 KiB");
    assert_eq!(rows[2].1, 4_608);

    // Why it is bit-exact, which is what makes it different from the split-count
    // change that is already implemented and already measured:
    //
    // * The key range of a workgroup depends only on `split`, from `tgid`
    //   (`attention.metal:783-785`), and is unchanged.
    // * The online-softmax recurrence is per simdgroup over the same keys in the
    //   same order (`attention.metal:855-861`), unchanged.
    // * `wbuf` and `qfbuf` are indexed per simdgroup (`attention.metal:806,819`),
    //   so a simdgroup never observed a peer's data; the single barrier is
    //   intra-simdgroup and survives the cut.
    // * The partial row is addressed by `head`, unchanged
    //   (`attention.metal:910`), and the combine sums `splits` partials per head
    //   in the same order (`attention.metal:967`).
    //
    // The split count is NOT touched, so unlike `KAPLUMBA_DECODE_SPLIT_RULE`
    // (which changes 43 of 128 greedy tokens, `attention.rs:180-187`) this
    // cannot change the token stream.
    println!("\n  bit-exact: NSG only groups query heads across threadgroups.");
    println!("  split count, key ranges, softmax order and partial layout all unchanged.");

    println!("\n  expected benefit AT LONG CONTEXT (the 10 full layers only):");
    println!("    - threadgroup memory 18 -> 9 KiB, under the measured 16 KB cliff");
    println!("    - grid granularity 64 -> 128 workgroups at the same split count,");
    println!("      i.e. finer packing where the 54-workgroup budget bites");
    println!("    - sliding is untouched: it is already one wave at 256 <= 288");
    println!("    - at 37 tokens neither shape is affected: this is ~0.1% of a step");

    println!("\n  NOT measured. To believe it, sweep NSG 8/4/2 x splits at 8k and");
    println!("  16k on the full shape only, interleaved A B A B, medians + spread.");
    println!("  The 128- and 64-thread one-wave budgets are unmeasured");
    println!("  (`attention.rs:344-350` returns None for both), so the wave");
    println!("  arithmetic above is the hypothesis, not the result.");
}

/// Everything the report leads with, in one block.
#[test]
fn summary() {
    println!("\n================= probe_attnkernel_split_combine =================");
    println!("shape   kv_heads head_dim NSG thr/grp splits@16k grid    tg_B   KV_B@16k");
    for shape in [SLIDING, FULL] {
        let splits = decode_splits(Q_HEADS, shape.kv_heads, 16_384);
        println!(
            "{:<7} {:>8} {:>8} {:>3} {:>7} {:>10} {:>6} {:>6} {:>8.1} MB",
            shape.label, shape.kv_heads, shape.head_dim, shape.nsg(), shape.threads(),
            splits, shape.workgroups(splits), shape.threadgroup_bytes(),
            mb(shape.read_bytes(16_384))
        );
    }
    println!("\nmeasured: attention 0.842 ms / ~62 ms step at 37 tokens");
    println!("modelled: attention {attn:.2} ms at 512, {attn16:.2} ms at 16384, {cap:.2} ms at the 32768 cap",
        attn = attention_ms(512), attn16 = attention_ms(16_384), cap = attention_ms(32_768));
    println!("modelled: attention overtakes the GEMVs at ~{cross:.0} tokens -- {ratio:.1}x beyond the cap",
        cross = (WEIGHT_BYTES_PER_TOKEN as f64 - SLIDING.read_bytes(16_384) as f64)
            / (FULL.bytes_per_key() * FULL.layers as u64) as f64,
        ratio = (WEIGHT_BYTES_PER_TOKEN as f64 - SLIDING.read_bytes(16_384) as f64)
            / (FULL.bytes_per_key() * FULL.layers as u64) as f64 / 32_768.0);
    println!("================================================================");
    println!("\n  PRE-EXISTING BREAK (not caused by this probe, not fixed by it):");
    println!("  crates/engine/examples/attn_bench.rs:374 calls");
    println!("    decode_splits(Q_HEADS, variant.key_count)          -- 2 args");
    println!("  crates/kernels/src/attention.rs:191 takes");
    println!("    decode_splits(q_heads, kv_heads, key_count)        -- 3 args");
    println!("  cargo check --example attn_bench -p kaplumba-engine:");
    println!("    error[E0061]: this function takes 3 arguments but 2 were supplied");
}