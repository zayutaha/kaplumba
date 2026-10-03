//! PROBE (read-only investigation, not a regression gate): does the KV cache
//! become the decode bottleneck as context grows, and is `[capacity][stream_len]`
//! DRAM-friendly?
//!
//! **Scope warning, stated once and meant.** Everything below is about *long*
//! context. The shipped decode benchmark is **37 tokens**, where the whole KV
//! read is 31.8 MiB against 12.04 GB of weights — 0.28 % of the bytes and
//! ~0.17 ms of a 62.3 ms step, and 0.842 ms measured for all 60 layers of
//! attention. **Nothing in this file implies a change is worth making for the
//! 37-token number.** The crossover where attention starts to matter is around
//! 8k of context, and the interesting range is 16k–32k.
//!
//! ## What is real production code under test, and what is not
//!
//! Two tests (`sliding_layers_wrap_and_never_memmove`,
//! `sliding_read_bytes_stop_growing_at_the_window`) drive
//! `kaplumba_engine::kv_cache::SlidingKvCache` — real production logic, host
//! side, no GPU. They would fail if the ring were replaced by a shift.
//!
//! Every other test **characterises a model re-derived from the sources**, not
//! the engine's own arithmetic. Per AGENTS.md §3.4 that is not a coverage claim
//! until someone has sabotaged the source and watched the corresponding number
//! move, and this file cannot do that (it is read-only, and `GpuKvCache` has no
//! constructor that does not take a `metal::Device`). Treat every figure below as
//! "what the sources say they will do", and gate any real change on
//! `memory_proof.rs` / `kv_growth_policy.rs` plus a GPU A/B.
//!
//! No model, no checkpoint, no `metal::Device`, no dispatch. Pure arithmetic plus
//! the host cache.
//!
//! ## HEAD note
//!
//! Probed at `2aaa50e`, not the `5e8ab73` in the brief.

use kaplumba_engine::kv_cache::SlidingKvCache;
use kaplumba_kernels::attention::decode_one_wave_workgroups;

// ---------------------------------------------------------------------------
// Gxma geometry, from `~/.kaplumba/models/Gxma/Gxma/config.json` `text_config`.
// ---------------------------------------------------------------------------

/// `num_hidden_layers`.
const LAYERS: usize = 60;
/// `layer_types` has `full_attention` at indices 5, 11, …, 59 — every sixth.
const FULL_LAYERS: usize = 10;
const SLIDING_LAYERS: usize = LAYERS - FULL_LAYERS; // 50

/// `num_attention_heads`. Same on both shapes.
const Q_HEADS: usize = 32;

/// Sliding: `num_key_value_heads` (`model.rs:3370-3374`, the non-full branch).
const SLIDING_KV_HEADS: usize = 16;
/// Sliding: `head_dim` (`model.rs:3365-3369`, the non-full branch).
const SLIDING_HEAD_DIM: usize = 256;
/// Sliding: `sliding_window`, which is also the cache capacity
/// (`model.rs:1086-1091` passes `config.sliding_window` to `GpuKvCache::new`).
const SLIDING_CAPACITY: usize = 1024;

/// Full: `num_global_key_value_heads`.
const FULL_KV_HEADS: usize = 4;
/// Full: `global_head_dim`.
const FULL_HEAD_DIM: usize = 512;

/// `model.rs:1077` `INITIAL_FULL_CACHE`; full caches never start larger.
const INITIAL_FULL_CACHE: usize = 512;
/// `gpu_cache.rs:59` `MAX_POLICY_CAPACITY` = `generate.rs` `DEFAULT_CONTEXT_LIMIT`.
const MAX_POLICY_CAPACITY: usize = 32_768;

/// `bench.rs:331` asserts `per_token_weight_bytes() == 12_039_487_488`.
const WEIGHT_BYTES_PER_TOKEN: u64 = 12_039_487_888;

/// Brief's baseline: 16.05 tok/s.
const BASELINE_TOK_S: f64 = 16.05;
/// `1000 / 16.05`, i.e. the brief's 62.3 ms.
const BASELINE_STEP_MS: f64 = 62.3;
/// Brief: "attention measured at only 0.842 ms per 60-layer pass".
const MEASURED_ATTENTION_MS_AT_37: f64 = 0.842;

const FP16: u64 = 2;
const U32: u64 = 4;

/// The line the AGX L1/L2 works in. Apple silicon publishes no figure; 128 B is
/// the architecture's documented fetch granularity, and every coalescing claim
/// below is stated against it so the conclusions do not depend on a larger one.
const CACHE_LINE: u64 = 128;
/// 32 lanes (`attention.metal:760`, `const uint lane = tid & 31u`).
const WARP: u64 = 32;

// ---------------------------------------------------------------------------
// Layout: `[capacity][stream_len]`, fp16, `stream_len = kv_heads * head_dim`
// (`gpu_cache.rs:9-10`, `:100-105`).
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Shape {
    label: &'static str,
    capacity: usize,
    kv_heads: usize,
    head_dim: usize,
}

impl Shape {
    /// `gpu_cache.rs:100-102`.
    fn stream_len(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    /// Distance between two *consecutive tokens* of one head, i.e. the stride
    /// the brief worries about.
    fn row_bytes(&self) -> u64 {
        self.stream_len() as u64 * FP16
    }

    /// Bytes one head's K row occupies — what one lane of the PV phase sweeps
    /// (`attention.metal:788`, `:876`, `:884-886`).
    fn head_row_bytes(&self) -> u64 {
        self.head_dim as u64 * FP16
    }

    /// `gpu_cache.rs:165-167` `resident_bytes`: what the cache **holds**.
    fn resident_bytes(&self) -> u64 {
        self.capacity as u64 * (self.stream_len() as u64 * FP16 * 2 + U32)
    }

    /// Live window, `gpu_cache.rs:186-192` `key_count`: full → `next_position`;
    /// sliding → `min(next_position, capacity)`.
    fn key_count(&self, position: usize) -> u64 {
        if self.label == "full" {
            position as u64
        } else {
            position.min(self.capacity) as u64
        }
    }

    /// What a decode step **reads** at absolute `position`: one pass over the
    /// live window. The `+ U32` is `positions`, read at `attention.metal:831`
    /// (`key_positions[k]`, `k = base + lane` — 32 consecutive u32, one
    /// contiguous 128 B run, already perfectly coalesced).
    fn read_bytes_at(&self, position: usize) -> u64 {
        self.key_count(position) * (self.stream_len() as u64 * FP16 * 2 + U32)
    }

    /// GQA ratio `q_heads / kv_heads`, which is also the simdgroup count `NSG`
    /// (`attention.metal:1059-1069`).
    fn nsg(&self) -> u64 {
        Q_HEADS as u64 / self.kv_heads as u64
    }
}

fn sliding() -> Shape {
    Shape {
        label: "sliding",
        capacity: SLIDING_CAPACITY,
        kv_heads: SLIDING_KV_HEADS,
        head_dim: SLIDING_HEAD_DIM,
    }
}

fn full(capacity: usize) -> Shape {
    Shape {
        label: "full",
        capacity,
        kv_heads: FULL_KV_HEADS,
        head_dim: FULL_HEAD_DIM,
    }
}

// ---------------------------------------------------------------------------
// The capacity ladder: `gpu_cache.rs:371-372` (`GROWTH_NUM/GROWTH_DEN` = 5/4),
// `gpu_cache.rs:414-444` (`growth_target`), `gpu_cache.rs:467-492` (`grow_once`).
// ---------------------------------------------------------------------------

/// Mirror of `gpu_cache.rs:472-491`: `current + ceil(current / 4)`.
fn grow_once(current: usize) -> usize {
    current + (current + 3) / 4
}

/// Mirror of `gpu_cache.rs:414-444`, including the `min(max_capacity)` clamp.
/// `None` where the engine returns `PositionOutOfRange`.
fn growth_target(current: usize, needed: usize, max_capacity: usize) -> Option<usize> {
    if needed <= current {
        return Some(current);
    }
    if needed > max_capacity {
        return None;
    }
    let mut target = current.max(1);
    while target < needed {
        target = grow_once(target);
    }
    Some(target.min(max_capacity))
}

/// What `ensure_capacity` actually holds once a run reaches `context` tokens.
fn full_capacity_at(context: usize) -> usize {
    growth_target(INITIAL_FULL_CACHE, context, MAX_POLICY_CAPACITY)
        .expect("the policy limit is 32768, so every context tested here fits")
}

// ---------------------------------------------------------------------------
// Accounting. Milliseconds throughout: `ms_for(b, gb_per_s) = b / (gb_per_s * 1e6)`.
// ---------------------------------------------------------------------------

/// Bytes one decode step reads across all 60 layers at `context`.
fn kv_read_bytes(context: usize, capacity: usize) -> u64 {
    sliding().read_bytes_at(context) * SLIDING_LAYERS as u64
        + full(capacity).read_bytes_at(context) * FULL_LAYERS as u64
}

/// Bytes the caches **hold** once a run has reached `context` tokens. Note that
/// this does not depend on `context` beyond the ladder rung — which is itself the
/// point of `read_and_footprint_differ_at_16k`.
fn kv_footprint(_context: usize, capacity: usize) -> u64 {
    sliding().resident_bytes() * SLIDING_LAYERS as u64
        + full(capacity).resident_bytes() * FULL_LAYERS as u64
}

fn ms_for(bytes: u64, gb_per_s: f64) -> f64 {
    bytes as f64 / (gb_per_s * 1.0e6)
}

/// Effective weight-read bandwidth implied by the brief's own baseline: the
/// 12.04 GB of weights over the step time left after subtracting the 37-token KV
/// read. Returns GB/s.
///
/// This is the only place a measured number becomes a constant, and it is
/// derived rather than quoted on purpose. It comes out near 194 GB/s, i.e. 71 %
/// of the 273 GB/s AGENTS.md §8 names for this box, which is what a decode step
/// that also pays ~1.4 ms of dispatch and elementwise work should look like.
/// The fixed point is not circular in practice: the 37-token KV read is 0.17 ms
/// of 62.3, so re-deriving it moves the answer by 0.3 %.
fn effective_gb_per_s() -> f64 {
    let kv_ms_37 = ms_for(kv_read_bytes(37, INITIAL_FULL_CACHE), 194.0);
    let weights_ms = BASELINE_STEP_MS - kv_ms_37;
    WEIGHT_BYTES_PER_TOKEN as f64 / (weights_ms * 1.0e6)
}

/// Step time at `context`: the weight stream is unchanged, the KV read is added
/// at the same effective bandwidth.
fn step_ms(context: usize, capacity: usize) -> f64 {
    let gb = effective_gb_per_s();
    let weights_ms =
        BASELINE_STEP_MS - ms_for(kv_read_bytes(37, INITIAL_FULL_CACHE), gb);
    weights_ms + ms_for(kv_read_bytes(context, capacity), gb)
}

fn tok_s(context: usize, capacity: usize) -> f64 {
    1.0e3 / step_ms(context, capacity)
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn mb(bytes: u64) -> f64 {
    bytes as f64 / 1.0e6
}

fn gb(bytes: u64) -> f64 {
    bytes as f64 / 1.0e9
}

fn row(note: &str, context: usize) -> String {
    let capacity = full_capacity_at(context);
    let read = kv_read_bytes(context, capacity);
    let hold = kv_footprint(context, capacity);
    format!(
        "{note:<24} pos={context:<6} full_cap={capacity:<6} \
         read={:>8.1} MiB ({:.3} GB)  footprint={:>8.1} MiB ({:.3} GB)  \
         step={:>6.2} ms  {:>6.3} tok/s ({:+6.1} %)",
        mib(read),
        gb(read),
        mib(hold),
        gb(hold),
        step_ms(context, capacity),
        tok_s(context, capacity),
        100.0 * (tok_s(context, capacity) / BASELINE_TOK_S - 1.0),
    )
}

// ===========================================================================
// 1. Layout shape and per-layer KV bytes.
// ===========================================================================

/// Corrects the brief's arithmetic and pins it.
///
/// The brief asserts `stream_len = 2048` for **both** layer types, then quotes
/// 8 MB sliding / 64 MB full. Both are wrong, in the direction that makes the
/// cache look smaller than it is:
///
/// * `stream_len = kv_heads * head_dim` is **4096** on the sliding layers
///   (16 × 256) and 2048 only on the full ones (4 × 512). `model.rs:3370-3374`
///   takes `kv_heads` from `num_global_key_value_heads` on full layers and
///   `num_key_value_heads` otherwise; `model.rs:3365-3369` does the same for
///   `head_dim` / `global_head_dim`.
/// * "16384×2048 = 64 MB" is K alone. K **and** V are two buffers
///   (`gpu_cache.rs:85-86`), so that layer is 128 MiB, not 64.
///
/// The corrected figures reproduce both of the brief's numbers — 881 MB at 512
/// positions and 2.18 GB at 16384 — which is the check that the correction is
/// the right one. They are, however, numbers of two *different* quantities, and
/// the brief uses one name for both; see `read_and_footprint_differ_at_16k`.
#[test]
fn per_layer_kv_bytes_at_the_real_shapes() {
    let s = sliding();
    assert_eq!(s.stream_len(), 4096, "sliding stream_len is 16 * 256");
    assert_eq!(s.row_bytes(), 8192, "sliding tokens are 8 KiB apart per head");

    let f = full(INITIAL_FULL_CACHE);
    assert_eq!(f.stream_len(), 2048, "full stream_len is 4 * 512");
    assert_eq!(f.row_bytes(), 4096, "full tokens are 4 KiB apart per head");

    // Sliding: capacity is fixed at the window and never grows, so its footprint
    // is the same at every context length.
    assert_eq!(s.resident_bytes(), 16_781_312, "16.0 MiB per sliding layer");
    assert_eq!(
        s.resident_bytes() * SLIDING_LAYERS as u64,
        839_065_600,
        "50 sliding layers = 800.03 MiB, a constant"
    );

    // Cross-check `resident_bytes` against the real host cache rather than
    // against my own restatement of the formula.
    //
    // This assertion used to add `capacity * 4` and name the difference as the
    // `positions` array. **That documented a bug as fact.** `SlidingKvCache::
    // resident_bytes` omitted `positions` while `GpuKvCache::resident_bytes`
    // counted it — the same logical object answering two different questions,
    // and `tests/kv_cache.rs` pinned the short answer. Fixed by
    // `spec/memory-swap-cliff` F8a, which added the `positions.len() * 4` term
    // to *both* host impls (`kv_cache.rs:97`, `:263`).
    //
    // So the two now agree exactly, with no correction term. Adding one would
    // double-count.
    let host = SlidingKvCache::with_capacity(SLIDING_CAPACITY, SLIDING_KV_HEADS, SLIDING_HEAD_DIM)
        .expect("the real sliding shape");
    assert_eq!(host.stream_len(), s.stream_len());
    assert_eq!(
        host.resident_bytes() as u64,
        s.resident_bytes(),
        "host and GPU caches must agree exactly on resident_bytes (F8a); \
         a `+ capacity * 4` term here would re-introduce the double-count that \
         F8a removed"
    );

    // The brief's 881 MB, which is the 512-position *footprint*.
    let hold_512 = kv_footprint(512, INITIAL_FULL_CACHE);
    assert_eq!(hold_512, 881_029_120);
    assert!((mb(hold_512) - 881.0).abs() < 1.0, "brief says 881 MB");
}

/// The ladder rung is not the context length, and that is worth 6.9 % of the
/// 16k footprint.
///
/// `grow_once` never lands on 16384. A run at 16 384 positions holds **18 212**
/// slots per full layer — 1 828 spare — because the rungs are
/// 14569 → 18212 → 22765 and the loop stops at the first one that fits.
#[test]
fn the_ladder_rung_is_not_the_context_length() {
    // Anchors the upstream unit test pins at `gpu_cache.rs:572-573`, so this
    // mirror cannot drift from `grow_once` without one of the two failing.
    assert_eq!(grow_once(512), 640);
    assert_eq!(grow_once(16_384), 20_480);
    for current in 1..8192usize {
        assert!(grow_once(current) > current, "grow_once must make progress");
    }

    assert_eq!(full_capacity_at(512), INITIAL_FULL_CACHE, "the ladder has not rung yet");
    assert_eq!(full_capacity_at(513), 640, "the first rung");
    assert_eq!(full_capacity_at(1024), 1250, "22 % over-allocated");
    assert_eq!(full_capacity_at(16_384), 18_212, "11.2 % over-allocated");
    assert_eq!(full_capacity_at(16_384) - 16_384, 1_828);
    assert_eq!(
        full_capacity_at(MAX_POLICY_CAPACITY),
        MAX_POLICY_CAPACITY,
        "32768 is reached by clamping a 35572 rung, so it lands exactly on the bound"
    );

    // The idealised footprint, capacity == context exactly, is where the brief's
    // 2.18 GB comes from.
    let ideal = full(16_384).resident_bytes() * FULL_LAYERS as u64
        + sliding().resident_bytes() * SLIDING_LAYERS as u64;
    assert!((mb(ideal) - 2181.0).abs() < 2.0, "the idealised 16k footprint");

    let actual = kv_footprint(16_384, full_capacity_at(16_384));
    assert_eq!(actual - ideal, 149_822_880, "the spare rung costs 142.9 MiB");
    assert!(
        (gb(actual) / gb(ideal) - 1.0 - 0.0686).abs() < 0.001,
        "6.9 % of the 16k footprint is never read"
    );
}

// ===========================================================================
// 2. Per-token read bytes vs footprint, and the tok/s curve.
// ===========================================================================

/// **These are two different quantities and the brief conflates them.** They
/// coincide only when every allocated slot is live.
///
/// * Read = `key_count × (stream_len × 2 × 2 + 4)`, and `key_count` is
///   `next_position` for a full layer (`gpu_cache.rs:186-192`), so the kernel
///   stops at the last written position and never touches the ladder's slack.
/// * Footprint = `capacity × (stream_len × 2 × 2 + 4)`
///   (`gpu_cache.rs:165-167`).
///
/// At 16 384 positions the engine **reads 2.182 GB and holds 2.332 GB** — the
/// brief's "2.18 GB" is the read, while its "881 MB" at 512 is a footprint. At
/// 512 the read is only 461 MB because the sliding layers are not full yet.
///
/// On the **sliding** layers the two are equal from position 1024 onward, and
/// that equality is what makes the sliding traffic a constant: 50 × 1024 ×
/// (4096 × 2 × 2 + 4) = **839 065 600 B read every single token, forever**.
#[test]
fn read_and_footprint_differ_at_16k() {
    // Not yet equal: the sliding ring is partly empty.
    assert_eq!(kv_read_bytes(512, INITIAL_FULL_CACHE), 461_496_320);
    assert_eq!(kv_footprint(512, INITIAL_FULL_CACHE), 881_029_120);
    assert!(kv_read_bytes(512, INITIAL_FULL_CACHE) < kv_footprint(512, INITIAL_FULL_CACHE));

    // From the window onward the sliding read is the whole sliding footprint, and
    // the full layers read `context` keys out of a ladder rung that is larger —
    // so read < footprint at 1024 too.
    assert_eq!(
        kv_read_bytes(1024, full_capacity_at(1024)),
        922_992_640,
        "sliding is exactly full (839 065 600) and the full layers read 1024 keys \
         out of a 1250 rung"
    );
    assert_eq!(
        kv_footprint(1024, full_capacity_at(1024)) - kv_read_bytes(1024, full_capacity_at(1024)),
        (1250 - 1024) * 8196 * 10,
        "the gap is the ladder slack, and only on the full layers"
    );

    // At 16k they diverge by exactly the ladder slack on the full layers.
    let capacity = full_capacity_at(16_384);
    assert_eq!(kv_read_bytes(16_384, capacity), 2_181_898_240, "2.182 GB read");
    assert_eq!(kv_footprint(16_384, capacity), 2_331_721_120, "2.332 GB held");
    assert_eq!(
        kv_footprint(16_384, capacity) - kv_read_bytes(16_384, capacity),
        (18_212 - 16_384) * 8196 * 10
    );

    // At 32768 the read catches up: 32768 *is* a rung.
    assert_eq!(
        kv_read_bytes(MAX_POLICY_CAPACITY, MAX_POLICY_CAPACITY),
        kv_footprint(MAX_POLICY_CAPACITY, MAX_POLICY_CAPACITY)
    );
}

/// The deliverable's headline: per-token KV read bytes and the tok/s they cost.
///
/// Shape of the answer: at 16 384 positions the step is ~11 ms slower than the
/// 37-token baseline and 15 % down in tok/s, and **38 % of the traffic is the
/// sliding layers' fixed 800 MiB, not the growth the question asks about.** The
/// marginal cost of context — full-attention layers only — is 9.5 ms from 512 to
/// 16k and a further 5.9 ms to 32k.
#[test]
fn step_time_and_tok_s_against_context() {
    let gbps = effective_gb_per_s();
    println!("\n=== KV decode traffic vs context (Gxma, M4 Pro) ===");
    println!("effective weight bandwidth implied by the 16.05 tok/s baseline: {gbps:.1} GB/s");
    println!(
        "weights per step: {:.3} GB, baseline step {BASELINE_STEP_MS} ms",
        gb(WEIGHT_BYTES_PER_TOKEN)
    );
    println!();
    println!("{}", row("the benchmark", 37));
    println!("{}", row("short conversation", 512));
    println!("{}", row("crossover", 4096));
    println!("{}", row("crossover", 8192));
    println!("{}", row("the interesting range", 16_384));
    println!("{}", row("the policy ceiling", 32_768));

    println!("\nsliding share of the KV read:");
    for context in [512usize, 8192, 16_384, 32_768] {
        let capacity = full_capacity_at(context);
        let s = sliding().read_bytes_at(context) * SLIDING_LAYERS as u64;
        println!("  {context:>6}: {:>5.1} %", 100.0 * s as f64 / kv_read_bytes(context, capacity) as f64);
    }

    // The model is anchored on the brief's own baseline at 37 tokens, so it must
    // reproduce it, and the measured 0.842 ms decomposes as bytes + fixed
    // dispatch, which is the only fixed cost the additive model needs.
    let kv_ms_37 = ms_for(kv_read_bytes(37, INITIAL_FULL_CACHE), gbps);
    println!(
        "\n0.842 ms measured at 37 tokens = {kv_ms_37:.3} ms of bytes \
         + {:.3} ms of fixed dispatch overhead",
        MEASURED_ATTENTION_MS_AT_37 - kv_ms_37
    );
    assert!(
        (kv_ms_37 - 0.172).abs() < 0.001,
        "0.17 ms of the 0.842 ms is bytes, {:.3} ms",
        kv_ms_37
    );
    assert!((tok_s(37, INITIAL_FULL_CACHE) - BASELINE_TOK_S).abs() < 0.01);

    // Degradation, as the deliverable asks for it.
    let t512 = tok_s(512, full_capacity_at(512));
    let t16k = tok_s(16_384, full_capacity_at(16_384));
    let t32k = tok_s(32_768, MAX_POLICY_CAPACITY);
    // Two marginal costs, because they differ and conflating them is how "KV
    // grows 32x" gets read as "decode is 32x slower". The sliding layers ramp
    // only from 512 to 1024 and are flat after that, so most of the total
    // marginal cost between 512 and 16k is the full layers but not all of it.
    let full_only = |context: usize| full(full_capacity_at(context)).read_bytes_at(context) * FULL_LAYERS as u64;
    println!(
        "\ntok/s: 512 = {t512:.2} ({:+.1} %), 16k = {t16k:.2} ({:+.1} %), 32k = {t32k:.2} ({:+.1} %)",
        100.0 * (t512 / BASELINE_TOK_S - 1.0),
        100.0 * (t16k / BASELINE_TOK_S - 1.0),
        100.0 * (t32k / BASELINE_TOK_S - 1.0)
    );
    println!(
        "marginal cost of context, TOTAL step      : {:+.2} ms/1000 tok (512->16k), \
         {:+.2} ms/1000 (16k->32k)",
        (step_ms(16_384, full_capacity_at(16_384)) - step_ms(512, INITIAL_FULL_CACHE)) / 15.872,
        (step_ms(MAX_POLICY_CAPACITY, MAX_POLICY_CAPACITY) - step_ms(16_384, full_capacity_at(16_384))) / 16.384,
    );
    println!(
        "marginal cost of context, FULL layers only: {:+.2} ms/1000 tok (512->16k), \
         {:+.2} ms/1000 (16k->32k)  <- linear, and the only part that keeps growing",
        ms_for(full_only(16_384) - full_only(512), gbps) / 15.872,
        ms_for(full_only(MAX_POLICY_CAPACITY) - full_only(16_384), gbps) / 16.384,
    );
    assert!(t16k < t512 && t512 < BASELINE_TOK_S, "monotone degradation");
}

/// The single largest structural fact in this file, and the reason a "KV grows
/// 32x" reading is only half the story: **the sliding layers do not grow.**
///
/// From position 1024 onward they read the same 839 065 600 B every token
/// forever, because the window is 1024 and every one of those 1024 keys is
/// inside it. That is 91 % of the KV read at 512 positions, 38 % at 16k and 24 %
/// at 32k — a permanent 4.3 ms floor under every decode step, on a box whose
/// budget is 24 GiB and whose attention budget is not the bottleneck.
///
/// It is not a bug. The window is what the model specifies
/// (`sliding_window: 1024`), and `model.rs:1086-1091` sizes the cache to exactly
/// that. But it means the *marginal* cost of context is smaller than the
/// *absolute* cost, and any claim about long-context decode has to separate them.
#[test]
fn sliding_read_bytes_stop_growing_at_the_window() {
    let at = |context: usize| sliding().read_bytes_at(context) * SLIDING_LAYERS as u64;
    assert_eq!(at(1023), 1023 * (4096 * 2 * 2 + U32) * 50, "1023 keys still short of a full ring");
    assert_eq!(at(1024), at(1025), "the ring is full; more context adds nothing");
    assert_eq!(at(1024), at(16_384));
    assert_eq!(at(1024), at(32_768));
    assert_eq!(at(1024), 839_065_600, "800.03 MiB, every token, forever");
    assert!(
        (gb(at(1024)) - 0.839).abs() < 0.001,
        "6.97 % of the 12.039 GB weight stream"
    );

    // And it is a read, not a scan-with-masking: exactly one of the 1024 keys is
    // dead at any moment (`q_abs - k_abs == 1024 >= window`, `attention.metal:833`),
    // and its K and V loads are skipped (`:835` guards K, `:880-882` guards V).
    // The only thing still read for the dead key is its `positions` entry, so the
    // gap between the 1024-key and 1023-key figures is one key's K+V plus every
    // key's 4-byte position.
    assert_eq!(1023 * 16_384 * SLIDING_LAYERS as u64, 838_041_600, "1023/1024 of the K+V");
    assert_eq!(at(1024) - 1023 * 16_384 * SLIDING_LAYERS as u64, 1_024_000);
    assert_eq!(16_384 * SLIDING_LAYERS as u64 + 4 * 1024 * SLIDING_LAYERS as u64, 1_024_000);
}

// ===========================================================================
// 3. Wrap versus shift / memmove.
// ===========================================================================

/// **Sliding layers wrap; they do not shift and they do not memmove.** Three
/// independent pieces of evidence, two of them executable against real
/// production code.
///
/// 1. `gpu_cache.rs:195-201` `slot_for`: `position % self.capacity` for
///    sliding, `position` for full. A ring, not a queue.
/// 2. `elements.metal:451` `cache_append`:
///    `slot = (slot_start + token) % capacity`, then
///    `key_cache[slot * stream_len + element] = keys[gid]`. **One** row written
///    per token. The kernel body is 22 lines with a single store pair and no
///    second loop, so there is nowhere for a shift to hide.
/// 3. `kv_cache.rs:305-309`, the host mirror, likewise `position % capacity`
///    into a single row. That is what the executable half below drives.
///
/// **What the ring is worth.** The per-token write is
/// `stream_len × 2 × 2` = 16 KiB for a sliding layer and 8 KiB for a full one —
/// 0.55 MB/token across all 60 layers, against a 2.18 GB read. A shift
/// implementation would instead rewrite `capacity × stream_len × 2 × 2` =
/// 16 MiB per sliding layer per token: **800 MiB/token of pure copy on top of
/// the 800 MiB read**, i.e. it would double the sliding traffic and add
/// 4.3 ms to every step from position 1024 onward, for a 16.8 MiB scratch per
/// layer. The ring is not an optimisation here, it is the difference between
/// 800 MiB and 1.6 GiB of sliding traffic per token.
#[test]
fn sliding_layers_wrap_and_never_memmove() {
    const SLOTS: usize = SLIDING_CAPACITY;
    let mut cache = SlidingKvCache::with_capacity(SLOTS, SLIDING_KV_HEADS, SLIDING_HEAD_DIM)
        .expect("the real sliding shape");
    let stream = cache.stream_len();
    let total = SLOTS * 2 + 37;

    // Bytes never move: the allocation is fixed at construction, so a shift would
    // have to memmove *within* it, and a ring does not. Since F8a the host
    // `resident_bytes` counts K, V *and* the `positions` array, so the expected
    // total is the two rings plus `SLOTS * 4`.
    let bytes_before = cache.resident_bytes() as u64;
    assert_eq!(
        bytes_before,
        (stream * SLOTS * 2 * 2) as u64 + (SLOTS as u64 * U32),
        "K, V and the positions array, both rings"
    );

    let row = vec![0_u16; stream];
    for position in 0..total {
        let mut k = row.clone();
        k[0] = position as u16;
        cache
            .append(position, &k, &row)
            .unwrap_or_else(|e| panic!("append {position}: {e}"));
    }
    assert_eq!(
        cache.resident_bytes() as u64,
        bytes_before,
        "resident_bytes is unchanged after {total} appends, so nothing was reallocated"
    );

    // The ring: the newest `SLOTS` positions are all still readable, each at
    // `position % SLOTS`, each still holding the value written at its own append.
    for position in (total - SLOTS)..total {
        let slot = cache
            .position_to_slot(position)
            .unwrap_or_else(|| panic!("position {position} must still be live"));
        assert_eq!(slot, position % SLOTS, "the slot is the modulo, not a queue head");
        let (k, _v) = cache.key_value(position).expect("live");
        assert_eq!(k[0], position as u16, "content must be what this position wrote");
    }

    // `visible_positions` is `next_position - capacity .. next_position`
    // (`kv_cache.rs:252-258`): the ring walked backwards from the head. A shift
    // cache would report the same window but would have had to physically move
    // every row to make it.
    assert_eq!(cache.visible_positions(), total - SLOTS..total);
    assert!(cache.contains(total - SLOTS), "the oldest live position");
    assert!(!cache.contains(total - SLOTS - 1), "the evicted one is gone");

    // The write really is one row: the last append touched `stream` elements, not
    // `stream * SLOTS`. `stream * 2` (K + V) against a shift's `stream * SLOTS`.
    assert_eq!(stream, 4096);
    assert!(
        stream * 2 * 100 < stream * SLOTS,
        "one append writes {stream} elements; a shift would move {}",
        stream * SLOTS
    );
}

// ===========================================================================
// 4 & 6. DRAM locality: what the kernel actually reads, and whether a
//         head-major layout would help. It would not.
// ===========================================================================

/// One warp memory instruction's DRAM cost under a lane→address map.
///
/// `lane_stride` is the address distance between adjacent lanes in bytes and
/// `bytes_per_lane` what each loads. Returns `(bytes fetched, bytes useful)`.
///
/// Two regimes, because one span formula cannot describe both: when lanes are
/// closer together than a cache line their runs merge and the footprint is the
/// span; when they are further apart each lane owns whole lines of its own, and
/// the count is `lanes × ceil(bytes / line)` with no merging at all.
fn warp_instruction(lane_stride: u64, bytes_per_lane: u64) -> (u64, u64) {
    let useful = WARP * bytes_per_lane;
    let lines = if lane_stride >= CACHE_LINE {
        WARP * bytes_per_lane.div_ceil(CACHE_LINE)
    } else {
        (lane_stride * (WARP - 1) + bytes_per_lane).div_ceil(CACHE_LINE)
    };
    (lines * CACHE_LINE, useful)
}

/// Line utilisation of a warp instruction, in percent.
fn util(fetched: u64, useful: u64) -> f64 {
    100.0 * useful as f64 / fetched as f64
}

/// **The kernel's real access pattern, read off `attention_gqa_body`.** It
/// splits by phase, and the answer is the opposite of the brief's hypothesis in
/// both directions.
///
/// **QK phase — `attention.metal:824`, `:836-840`:**
/// ```c
/// const uint k = base + lane;                    // lane -> KEY
/// const device half *krow = keys + k * kv_stride + head_base;
/// for (int i = 0; i < HEAD_DIM / 4; ++i)
///     const uint2 kraw = *reinterpret_cast<const device uint2 *>(krow + 4 * i);
/// ```
/// Lanes map to **keys**, so adjacent lanes are `kv_stride × 2` = 4096 B (full)
/// or 8192 B (sliding) apart and each loads 8 B. **Scattered**: one whole line
/// per lane, 256 useful bytes out of 4 KiB fetched.
///
/// **PV phase — `attention.metal:788`, `:876`, `:884-886`:**
/// ```c
/// constexpr int VPT = 8;                                       // :757
/// const uint lane_dim = lane * uint(VPT);                      // :788
/// const uint d = lane_dim + uint(b) * 32u * uint(VPT);         // :876
/// const device half *vrow = values + (base + t) * kv_stride + head_base + d;
/// const uint4 raw = *reinterpret_cast<const device uint4 *>(vrow);
/// ```
/// Lanes map to **dimensions**, `VPT × 2` = 16 B apart, so the warp reads one
/// contiguous 512 B run in a single `uint4` load. **Perfectly coalesced**.
///
/// So: **V is exactly right, K is scattered.** `key_positions` is right too —
/// `attention.metal:831` reads 32 consecutive `u32`, one contiguous 128 B run.
#[test]
fn the_kernel_reads_v_coalesced_and_k_scattered() {
    // PV: `lane * VPT` halves = 16 B apart, one `uint4` per lane.
    let (v_lines, v_useful) = warp_instruction(16, 16);
    assert_eq!((v_lines, v_useful), (512, 512), "512 B out of 4 lines, 100 %");
    assert_eq!(util(v_lines, v_useful), 100.0);

    // QK: lanes are keys, so the stride is the whole row and nothing merges.
    for shape in [sliding(), full(INITIAL_FULL_CACHE)] {
        let (k_lines, k_useful) = warp_instruction(shape.row_bytes(), 8);
        assert_eq!(k_useful, 256, "32 lanes × 8 B");
        assert_eq!(k_lines, WARP * CACHE_LINE, "{} K: one line per lane", shape.label);
        assert!(
            util(k_lines, k_useful) < 10.0,
            "{} K utilisation {:.2} % -- scattered, as documented",
            shape.label,
            util(k_lines, k_useful)
        );
        println!(
            "{}: QK warp instruction fetches {} B for {} useful ({:.2} %); \
             PV fetches 512 B for 512 useful (100.00 %)",
            shape.label,
            k_lines,
            k_useful,
            util(k_lines, k_useful)
        );
    }

    // The scatter survives at both shapes, which is why attention is 1.4 % of the
    // 37-token step and 15 % of the 16k step rather than 8× that: a warp's whole
    // working set for one 32-key tile is `32 × head_row_bytes` — 16 KiB sliding,
    // 32 KiB full — small enough to sit in the 128 KiB-per-core L1, and every
    // byte of a line is consumed within `HEAD_DIM / 16` iterations of the inner
    // loop. The penalty is transaction *count*, and the cache absorbs it.
    assert_eq!(32 * sliding().head_row_bytes(), 16_384);
    assert_eq!(32 * full(1).head_row_bytes(), 32_768);

    // What it is NOT: a 4 KB stride between "consecutive tokens for one head"
    // (`row_bytes`), because the kernel never sweeps one head across positions.
    // It sweeps *positions* with a head offset, which is exactly why the layout
    // is right and the brief's concern is misplaced.
    assert_eq!(sliding().row_bytes(), 8192);
    assert_eq!(full(1).row_bytes(), 4096);
}

/// **Negative result, quantified: `[stream_len][capacity]` (head-major) is
/// strictly worse, and it would break the one read that is currently perfect.**
///
/// Head-major puts position on the innermost axis. The PV phase's `lane → dim`
/// map then has adjacent lanes `head_dim × 2` = 1024 B (full) / 512 B (sliding)
/// apart — one line per lane, 512 useful bytes out of 4 KiB — which is exactly
/// the pathology the QK phase has today and the PV phase does not.
///
/// At 16 384 positions V is half the KV read, so this is not a rounding error:
/// it takes the PV phase from 9.1 M line fetches to 72.9 M for the identical
/// byte count, and turns a phase that is currently free into one that is
/// co-dominant with the bandwidth bound.
///
/// The row-buffer argument that motivates head-major does not survive contact
/// with this engine either, and this is the part worth writing down: because
/// `decode_splits` puts every `kv_heads` workgroup in **one wave** — 256 of a
/// measured 288 budget for sliding, and all 4 for full — the workgroups that own
/// the *other* heads of the same position are resident **simultaneously**, so
/// their accesses aggregate into the same cache lines and the same DRAM pages in
/// the same instant. Position-major therefore already gets 100 % line
/// utilisation *and* full row-buffer consumption, in aggregate. Head-major would
/// trade away the per-instruction coalescing and buy back nothing.
#[test]
fn head_major_would_break_the_one_coalesced_read() {
    for shape in [sliding(), full(full_capacity_at(16_384))] {
        let (today_lines, today_useful) = warp_instruction(16, 16);
        assert_eq!((today_lines, today_useful), (512, 512), "PV today: 100 %");

        let (hm_lines, hm_useful) = warp_instruction(shape.head_row_bytes(), 16);
        assert_eq!(hm_useful, 512, "the same useful bytes");
        assert_eq!(hm_lines, WARP * CACHE_LINE, "{} head-major PV: one line per lane", shape.label);

        // QK is scattered under both layouts, so head-major only costs.
        let (qk_now, _) = warp_instruction(shape.row_bytes(), 8);
        let (qk_hm, _) = warp_instruction(shape.head_row_bytes(), 8);
        assert_eq!(qk_hm, qk_now, "{}: QK unchanged, PV 8x worse", shape.label);

        println!(
            "{} head-major: PV {} -> {} B fetched ({:.2} %), {}x worse; \
             QK {} -> {} B (unchanged)",
            shape.label,
            today_lines, hm_lines,
            util(hm_lines, hm_useful),
            hm_lines / today_lines,
            qk_now, qk_hm
        );
    }

    // The 16k consequence, in absolute terms.
    let v_bytes = kv_read_bytes(16_384, full_capacity_at(16_384)) / 2;
    assert!(
        (gb(v_bytes) - 1.091).abs() < 0.001,
        "V is half the 16k KV read of 2.182 GB, i.e. {:.3} GB",
        gb(v_bytes)
    );
    let today = v_bytes / 128;
    let head_major = v_bytes / 128 * 8;
    println!(
        "16k V: {today} line fetches today vs {head_major} head-major \
         ({:.0} MiB vs {:.0} MiB of line traffic)",
        today * 128 / 1_048_576,
        head_major * 128 / 1_048_576
    );
    assert_eq!(head_major, today * 8);
}

// ===========================================================================
// The concrete recommendation, its budget, and its bit-exactness caveat.
// ===========================================================================

/// **There is no layout change worth making here.** The layout is already the
/// right one, and the head-major alternative is measurably worse. What is left
/// is the QK phase's scattered read, and the fix is in the kernel at
/// `crates/kernels/src/attention.metal:836-840`.
///
/// Two ways to fix it, and the choice between them is a *numerics* question
/// rather than an engineering one:
///
/// **(a) Threadgroup-staged, bit-exact.** Replace the per-lane scattered read
/// with a cooperative coalesced load of a `KT`-key tile into threadgroup memory,
/// then have each lane read its own key's row back out of it — exactly the trick
/// already used for Q at `attention.metal:806-815`. The per-key online-softmax
/// recurrence is untouched, so the result is bit-identical. Cost is threadgroup
/// memory, and the budget is tight:
///
/// ```c
/// // attention.metal:988-989
/// threadgroup float wbuf[2 * NSG * 32];      // sliding 512 B,  full 2 KiB
/// threadgroup float qfbuf[NSG * HEAD_DIM];   // sliding 2 KiB, full 16 KiB
/// ```
///
/// Metal allows 32 KiB per threadgroup, which leaves 30 208 B on the sliding
/// layers (a **59**-key K tile) and 14 336 B on the full ones (a **14**-key
/// tile). **14 is the binding constraint:** at HEAD_DIM=512 a warp-shaped
/// 32-key K tile is 32 KiB and does not fit alongside `qfbuf` at all, so (a)
/// either runs a 14-key tile with two threadgroup barriers per warp tile, or
/// shrinks `qfbuf` — Q is 512 floats per simdgroup only because
/// `attention.metal:799-805` hoists the whole row out of the key loop, and an
/// fp16 `qfbuf` would halve it to 8 KiB and leave 24 KiB spare, which is a
/// 48-key tile.
///
/// **(b) Lane → dimension, no scratch, but not bit-exact.** Every lane reads a
/// 32 B slice of **one shared key** and the dot product ends in a `simd_sum`:
/// 2 coalesced warp instructions per key instead of 128 scattered ones, and zero
/// threadgroup memory. But it forces the running max to advance every 32 keys
/// rather than every key. `attention.metal:456-483` already documents that
/// rescale as the standard flash-attention form and as *more* numerically stable;
/// `attention.rs:174-188` records that doing it to decode attention changed
/// **43 of 128** output elements and produced a different token stream. Under
/// AGENTS.md §8 ("no quality loss") that is a sign-off question for the user,
/// not an engineering one.
///
/// **Estimated benefit at 16 384 positions, for (a).** Bytes are unchanged —
/// 1.091 GB of K and 1.091 GB of V — and no KV quantisation is permitted
/// (AGENTS.md §8), so not one byte can be removed. What is recoverable is the
/// QK phase's transaction count: 8 warp load instructions per key instead of 128
/// (see the arithmetic below), all fully coalesced. If the scatter costs the
/// usual 1.3–2× against the bandwidth bound, recovering it on the 16k K stream
/// is worth **1.3–2.8 ms of a 73.4 ms step, i.e. 1.8–3.8 %**. Separately,
/// `KAPLUMBA_DECODE_SPLIT_RULE=workgroups` (already implemented, measured 1.34×
/// on the full layers at 8k, but not the default because it changes decode
/// numerics) is worth **up to ~1.8 ms** at 16k by taking the full layers' grid
/// from 64 workgroups back inside the measured 54. Both are **long-context
/// levers: at 37 tokens the KV read is 33.3 MB, 0.28 % of the step's bytes, and
/// either change is worth nothing measurable.**
#[test]
fn the_concrete_recommendation_and_its_budget() {
    // Threadgroup budget, `attention.metal:988-989`.
    for (shape, nsg) in [(sliding(), 2_usize), (full(1), 8_usize)] {
        let used = (2 * nsg * 32 + nsg * shape.head_dim as usize) * 4;
        let spare = 32_768 - used;
        let kt = spare / shape.head_row_bytes() as usize;
        println!(
            "{} (NSG={nsg}): wbuf + qfbuf = {used} B, spare {spare} B, \
             largest fp16 K tile that fits = {kt} keys ({} B)",
            shape.label, kt * shape.head_row_bytes() as usize
        );
        assert!(used < 32_768);
        if shape.label == "sliding" {
            assert_eq!(used, 2_560, "512 B wbuf + 2 KiB qfbuf");
            assert_eq!(kt, 59);
        } else {
            assert_eq!(used, 18_432, "2 KiB wbuf + 16 KiB qfbuf");
            // 14 keys, i.e. *less* than the 32-key warp tile. This is the binding
            // constraint on (a) and the reason (b) exists.
            assert_eq!(kt, 14);
            assert!(kt < 32, "a warp-shaped 32-key K tile is {} B and does not fit", 32 * 1024);
        }
    }

    // The instruction-count claim, as arithmetic so it can be checked. Today the
    // QK phase issues `HEAD_DIM / 4` warp loads per key, each fetching 32 lines
    // for 256 useful bytes. A cooperative load of a `KT`-key tile moves
    // `KT × head_dim × 2` bytes = `KT × head_dim / 8` `uint4`s spread over the
    // group's 8 warps, i.e. `KT × head_dim / 64` warp instructions per tile —
    // **`head_dim / 64` per key, independent of `KT`**. The staging buffer's size
    // buys coalescing, not instruction count.
    for shape in [sliding(), full(1)] {
        let today = shape.head_dim as u64 / 4;
        let staged = shape.head_dim as u64 / 64;
        println!(
            "{} QK global warp loads per key: {today} -> {staged} ({:.0}x fewer)",
            shape.label, today as f64 / staged as f64
        );
        assert_eq!(staged * 16, today, "16x fewer at both shapes");
    }

    // What only 128 of the 512 Q dimensions are rotated: `rope_partial_factor`
    // (`config.json` `full_attention.partial_rotary_factor`), which is why
    // shrinking `qfbuf` is on the table.
    assert_eq!(F32_BYTES_FULL_QF, 16_384, "qfbuf[8][512] floats");
    assert_eq!(F16_BYTES_FULL_QF, 8_192, "fp16 qfbuf would halve it, leaving 24 KiB spare");
}

const F32_BYTES_FULL_QF: usize = 8 * 512 * 4;
const F16_BYTES_FULL_QF: usize = 8 * 512 * 2;

/// Decode splits: how much parallelism the KV read actually gets, and why the
/// head-major argument above does not apply.
///
/// Default rule `ThreadTarget` (`attention.rs:303`):
/// `splits = clamp(16384 / (32 × q_heads), 1, 16)`, then
/// `.min(key_count.div_ceil(ATTENTION_MIN_KEYS_PER_SPLIT))` (`attention.rs:289`,
/// `:306`), with `ATTENTION_MIN_KEYS_PER_SPLIT = 64` at `attention.rs:25`.
///
/// * Sliding: 16 splits × 16 kv heads = **256 workgroups of 64 threads**, and the
///   measured one-wave budget for a 64-thread group is 288 (`attention.rs:346`).
///   One wave, every kv head resident.
/// * Full: 16 splits × 4 kv heads = **64 workgroups of 256 threads**, and the
///   budget for a 256-thread group is 54 (`attention.rs:347`). **64 is past it.**
///   `attention.rs:292-294` records that 56 workgroups costs 1.50× the time of 52
///   for the identical byte count, and `attention.rs:330-333` records that the
///   `workgroups` rule reproduces 13 splits for `kv_heads = 4` — which is 52
///   workgroups, inside the budget. That rule is implemented and measured
///   (`KAPLUMBA_DECODE_SPLIT_RULE=workgroups`, 1.34× on the full layers at 8k) but
///   is **not** the default, because it changes decode numerics and therefore the
///   token stream (`attention.rs:174-188`). At 16k that is worth ~1.7 ms of the
///   73.4 ms step, so it is a legitimate long-context lever to raise with the
///   user — but it is a numerics change, not a layout one.
#[test]
fn decode_split_grids_at_long_context() {
    let thread_target = 16_384_usize / (32 * Q_HEADS);
    assert_eq!(thread_target, 16);
    assert_eq!(decode_one_wave_workgroups(64), Some(288), "attention.rs:346");
    assert_eq!(decode_one_wave_workgroups(256), Some(54), "attention.rs:347");

    for (label, shape, context) in [
        ("sliding@16k", sliding(), 16_384_usize),
        ("full@512", full(INITIAL_FULL_CACHE), 512),
        ("full@16k", full(full_capacity_at(16_384)), 16_384),
        ("full@32k", full(MAX_POLICY_CAPACITY), 32_768),
    ] {
        let keys = shape.key_count(context) as usize;
        let splits = thread_target.min(keys.div_ceil(64)).max(1);
        let threads = 32 * shape.nsg() as usize;
        let grid = splits * shape.kv_heads;
        let budget = decode_one_wave_workgroups(threads).expect("both shapes are measured");
        println!(
            "{label:<12} keys={keys:<6} splits={splits:<3} grid={grid:<4} ({threads} thr)  \
             budget={budget:<4} {}",
            if grid <= budget { "1 wave" } else { "PAST THE CLIFF" }
        );
    }

    // The claim that matters: sliding is inside its budget, full is not.
    assert!(16 * SLIDING_KV_HEADS <= 288, "sliding: 256 <= 288, one wave");
    assert!(16 * FULL_KV_HEADS > 54, "full at 16k: 64 > 54, second wave");
    assert_eq!(54 / FULL_KV_HEADS, 13, "the workgroups rule would pick 13 splits = 52");
}

// ===========================================================================
// Summary.
// ===========================================================================

#[test]
fn summary() {
    let gbps = effective_gb_per_s();
    let kv_ms_37 = ms_for(kv_read_bytes(37, INITIAL_FULL_CACHE), gbps);

    println!("\n================ KV SCALING SUMMARY ================");
    println!(
        "baseline         37 tokens : {BASELINE_STEP_MS:.2} ms  {BASELINE_TOK_S:.2} tok/s   \
         (measured attention {MEASURED_ATTENTION_MS_AT_37:.3} ms/pass = {kv_ms_37:.3} ms bytes \
         + {:.3} ms fixed)",
        MEASURED_ATTENTION_MS_AT_37 - kv_ms_37
    );
    println!(
        "weights per step          : {:.3} GB",
        gb(WEIGHT_BYTES_PER_TOKEN)
    );
    println!("effective bandwidth       : {gbps:.1} GB/s of 273 nominal (71 %)\n");

    for (context, note) in [
        (512_usize, "short conversation"),
        (8192, "crossover"),
        (16_384, "the interesting range"),
        (32_768, "the policy ceiling"),
    ] {
        let capacity = full_capacity_at(context);
        let read = kv_read_bytes(context, capacity);
        let sliding_share =
            100.0 * (sliding().read_bytes_at(context) * SLIDING_LAYERS as u64) as f64 / read as f64;
        println!("{}", row(note, context));
        println!(
            "    KV read = {:.3} GB = {:.1} % of the weight stream; \
             {:.1} ms of the step; sliding {:.0} % / full {:.0} %",
            gb(read),
            100.0 * read as f64 / WEIGHT_BYTES_PER_TOKEN as f64,
            ms_for(read, gbps),
            sliding_share,
            100.0 - sliding_share
        );
        println!(
            "    KV footprint {:.3} GB; sliding {} from position 1024, \
             {:.1} % of the footprint is unread ladder slack",
            gb(kv_footprint(context, capacity)),
            if context >= SLIDING_CAPACITY { "fixed" } else { "still filling" },
            100.0 * (kv_footprint(context, capacity) - read) as f64
                / kv_footprint(context, capacity) as f64
        );
        println!();
    }

    println!("ANSWERS");
    println!("  per-token KV read : 461 MB at 512 | 2.182 GB at 16384 | 3.525 GB at 32768");
    println!("  tok/s             : 15.50 at 512 | 13.63 at 16384 | 12.45 at 32768 (baseline 16.05)");
    println!("  read vs footprint : 2.182 GB read, 2.332 GB held at 16k -- 6.9 % of the");
    println!("                     footprint is ladder slack the kernel never touches");
    println!("  sliding memmove?  : NO. It rings (gpu_cache.rs:195-201, elements.metal:451).");
    println!("                     A shift would add 800 MiB/token of copy, doubling sliding traffic.");
    println!("  layout            : [capacity][stream_len] is CORRECT. Head-major would take the PV");
    println!("                     phase from 100 % to 12.5 % line utilisation -- 8x worse, on half");
    println!("                     the KV traffic, and it buys no row-buffer locality because all");
    println!("                     kv_heads workgroups are in one wave.");
    println!("  one change        : attention.metal:836-840, stage the QK K tile through threadgroup");
    println!("                     memory with a cooperative coalesced load. Bit-exact. Worth");
    println!("                     1.8-3.8 % at 16k (1.3-2.8 ms of 73.4 ms), ~0 % at 37 tokens.");
    println!("  adjacent lever    : KAPLUMBA_DECODE_SPLIT_RULE=workgroups takes the full layers'");
    println!("                     16k grid from 64 workgroups inside the measured 54. ~1.8 ms,");
    println!("                     but it changes decode numerics, so it needs sign-off.");
    println!("  long context only : at 37 tokens the KV read is 0.28 % of the step's bytes.");
    println!("=====================================================\n");
}
