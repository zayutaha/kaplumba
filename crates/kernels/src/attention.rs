use std::ffi::c_void;
use std::mem::{size_of, size_of_val};
use std::sync::OnceLock;

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, Library,
    MTLCommandBufferStatus, MTLResourceOptions, MTLSize,
};

pub use crate::exl3::KernelError;

const SOURCE: &str = include_str!("attention.metal");
const MAX_HEAD_DIM: usize = 512;

/// Most splits of the KV axis the decode path ever takes.
///
/// This is a hard cap, and it is what makes the decode scratch *fixed-size*: the
/// engine allocates `ATTENTION_SPLITS_MAX * q_heads * (head_dim + 2)` floats
/// once, at construction, and no prompt length can change it. 16 is what fills
/// the GPU at Gxma's 32 query heads -- `16 * 32 * 32` = 16,384 threads, ~6 waves
/// over a 20-core M4 Pro's resident thread capacity.
pub const ATTENTION_SPLITS_MAX: usize = 16;

/// Fewest keys a split may be given. Below this the combine pass's extra read
/// of the partials costs more than the parallelism buys.
const ATTENTION_MIN_KEYS_PER_SPLIT: usize = 64;

/// Threads the decode partial kernel aims to run, before the per-split cap.
///
/// **Retained only as the fallback for threadgroup widths whose one-wave
/// workgroup budget was not measured** -- see [`decode_splits`] and
/// [`decode_one_wave_workgroups`]. It is a thread target, and a thread target
/// is the wrong unit: what quantises is workgroups, and the two are not the
/// same thing once `kv_heads` differs between layer shapes.
const ATTENTION_DECODE_THREADS_TARGET: usize = 16_384;

/// Key counts at or below which the decode path prefers one single-pass dispatch
/// to split+combine, when `KAPLUMBA_DECODE_SINGLE_PASS` is set to a truthy value
/// that is not a number of keys. `None` (the default) means the production path
/// is untouched.
///
/// MEASURED, not the intersection of two cost models. See
/// `spec/decode-19tok-repacked-250gb/implementation_notes.md` ("single-pass
/// decode attention") for the sweep, the interleaved A/B medians and the negative
/// results. The short version: the two paths move *exactly the same DRAM bytes* --
/// a workgroup owns one kv head and its `q_heads / kv_heads` simdgroups read that
/// head's K and V out of the same L1, so a split never duplicates a byte. The
/// single-pass path therefore saves one dispatch and one
/// `splits * q_heads * (head_dim + 2)` float round trip, and pays for it with the
/// entire `splits` factor of parallelism.
///
/// MEASURED: **64 keys**, which is exactly `ATTENTION_MIN_KEYS_PER_SPLIT` and
/// therefore exactly where `decode_splits` already returns 1. That the crossover
/// lands on that number is structural, not a coincidence, and it is worth being
/// explicit about because it means the window is as narrow as it looks:
///
///   * At `splits == 1` both paths launch `kv_heads` workgroups of
///     `32 * (q_heads / kv_heads)` threads over the identical key range. The
///     *only* difference is the second dispatch and the
///     `q_heads * (head_dim + 2)` float round trip through the partials buffer,
///     so single-pass wins by exactly that: measured 1.059x (sliding) and
///     1.034x (full) at 64 keys, and 1.14x / 1.10x at 16 keys where the combine
///     is a larger share of a smaller kernel.
///   * At `splits == 2` the split path has twice the workgroups for the identical
///     byte count, and it wins immediately: 0.68x (sliding) and 0.63x (full) at 65
///     keys, falling to 0.13x and 0.14x by the shapes' maxima. The gap widens
///     monotonically with `splits` because the loss is the whole `splits` factor
///     of occupancy.
///
/// So this is a real but very short win: the first 64 tokens of a conversation,
/// worth 0.097 ms/token summed over 60 layers (0.08 sliding + 0.017 full), and
/// nothing at all after that. See
/// `spec/decode-19tok-repacked-250gb/implementation_notes.md` ("single-pass
/// decode attention") for the full interleaved tables and the negative results.
pub const ATTENTION_SINGLE_PASS_MAX_KEYS: usize = 64;

/// Highest key count `KAPLUMBA_DECODE_SINGLE_PASS=<n>` may be swept to. Far above
/// any crossover, and deliberately so: the sweep has to be able to reach the point
/// where single-pass is *much* worse, or it cannot see the shape of the curve.
pub const ATTENTION_SINGLE_PASS_SWEEP_MAX_KEYS: usize = 65_536;

/// Queries per threadgroup in the flash prefill attention kernel (T4,
/// `spec/prefill-24k-90tps`). At 24k tokens each query would otherwise re-read
/// its whole causal K/V prefix -- 23.5 TB over the 10 full layers, ~86 s of pure
/// 273 GB/s streaming against ~31 s of tensor work. Blocking `BQ` queries per
/// threadgroup divides that byte traffic by `BQ`. Must equal the `BQ` argument of
/// `KAPLUMBA_FA_FLASH(online_attention_flash_512_8, 512, 8, BQ)` in
/// `attention.metal`, which also sizes the threadgroup memory.
pub const FLASH_QUERY_BLOCK: usize = 2;

/// Key count at or above which the flash prefill kernel uses
/// [`FLASH_QUERY_BLOCK`] queries per threadgroup instead of 1. Query blocking
/// halves the K/V device traffic but doubles the per-threadgroup critical path,
/// so it only pays when attention is bandwidth-bound rather than latency-bound.
/// Overridable with `KAPLUMBA_FLASH_BQ=1|2` for A/B.
pub const FLASH_QUERY_BLOCK_MIN_KEYS: usize = 16_384;

/// 8-key m-tiles per simdgroup in the shipped flash kernel: **1**, i.e. a
/// 64-key pass with each simdgroup owning one m-tile. This is a record of the
/// shipped configuration, not a template parameter -- the flash body is not
/// currently instantiated with a key-subtile count.
///
/// A `KSUB = 4` variant (a 256-key pass, 4 m-tiles per simdgroup) was built to
/// halve the QK's `simdgroup_load`s per MMA; it needs 28 KB of threadgroup memory
/// instead of 19.5 KB and measured **13% slower** in the engine at 8k (attention
/// 20.6 s -> 23.4 s) as well as slower on the micro-bench at 24 576 keys
/// (1118 -> 1318 ms). It was reverted, so this constant has no `attention.metal`
/// counterpart to match.
pub const FLASH_KEY_SUBTILES: usize = 1;

/// `KAPLUMBA_FLASH_BQ=1|2` forces the flash query block; any other value is
/// ignored. Read once (same `OnceLock` idiom as [`attention_qk`]).
fn flash_query_block_override() -> Option<usize> {
    static CACHE: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| match std::env::var("KAPLUMBA_FLASH_BQ") {
        Ok(raw) => match raw.trim().parse::<usize>() {
            Ok(b) if b == 1 || b == FLASH_QUERY_BLOCK => Some(b),
            _ => None,
        },
        Err(_) => None,
    })
}

/// Query block for one flash dispatch, from the key count and the override.
pub fn flash_query_block_for(key_count: usize) -> usize {
    if let Some(forced) = flash_query_block_override() {
        return forced;
    }
    if key_count >= FLASH_QUERY_BLOCK_MIN_KEYS {
        FLASH_QUERY_BLOCK
    } else {
        1
    }
}

/// Key count from which the decode step uses the MMA flash kernel instead of the
/// scalar split kernel.
///
/// Defaults to 4096, the same threshold oMLX ships (`OMLX_MIMO_DECODE_FLASH_MIN_KEYS`,
/// `omlx/patches/mimo_v2/sdpa_flash.py`, v0.7.0): below it a decode split is
/// short enough that the scalar path's lack of MMA does not dominate, and the
/// existing `attention_single_pass_bit_exact` guarantees are worth more than the
/// speed.
///
/// `KAPLUMBA_DECODE_FLASH=0|off|false|no` turns it off everywhere;
/// `KAPLUMBA_DECODE_FLASH_MIN_KEYS=<n>` moves the threshold. Read once per
/// process (same `OnceLock` idiom as `flash_query_block_override`), because this
/// is consulted 60x/token and `getenv` takes a lock in some libc builds.
pub fn decode_flash_min_keys() -> usize {
    static CACHE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        if let Ok(raw) = std::env::var("KAPLUMBA_DECODE_FLASH") {
            if matches!(raw.trim(), "0" | "off" | "false" | "no") {
                return usize::MAX;
            }
        }
        std::env::var("KAPLUMBA_DECODE_FLASH_MIN_KEYS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(4096)
    })
}

/// Whether a decode step with `key_count` live keys should use the MMA kernel.
pub fn decode_flash_enabled(key_count: usize) -> bool {
    key_count >= decode_flash_min_keys()
}

/// Parse a `KAPLUMBA_DECODE_SINGLE_PASS` value into a key limit. `0` means off.
///
/// Exposed so the parsing rules are testable without the process-wide cache in
/// [`decode_single_pass_max_keys`], which is deliberately read once.
///
/// Total by construction: every value maps to a limit, and anything unrecognised
/// maps to off. This is consulted on the decode hot path, where a panic would
/// take the process down and where an unrecognised value must not silently widen
/// the grid.
pub fn parse_single_pass_limit(raw: &str) -> Option<usize> {
    let raw = raw.trim();
    match raw {
        // Explicitly off, spelled out so a shell that exports the variable empty
        // does not silently turn it on.
        "" | "0" | "off" | "false" | "no" => Some(0),
        // Truthy but not a number: on, up to the measured threshold. Kept as a
        // spelling of "on" so the default can be re-derived from a sweep without
        // editing and rebuilding.
        "1" | "on" | "true" | "yes" | "auto" => Some(ATTENTION_SINGLE_PASS_MAX_KEYS),
        // Any other number is that many keys. Clamped to the sweep range, which is
        // about the sweep reaching the whole curve and about a typo not becoming
        // an unbounded grid; this path does not touch the partials buffer, so the
        // clamp is not a scratch-safety limit.
        _ => Some(
            raw.parse::<usize>()
                .ok()
                .map_or(0, |keys| keys.min(ATTENTION_SINGLE_PASS_SWEEP_MAX_KEYS)),
        ),
    }
}

/// How many keys of decode KV context may use the single-pass dispatch, or `None`
/// to always use split+combine. **Default is `None`** -- the production decode
/// path is unchanged unless this environment variable is set.
///
/// `KAPLUMBA_DECODE_SINGLE_PASS`:
///
/// | value | meaning |
/// |---|---|
/// | unset, `0`, `off`, `false`, `no`, empty | off (default) |
/// | `1`, `on`, `true`, `yes`, `auto` | on, up to [`ATTENTION_SINGLE_PASS_MAX_KEYS`] keys |
/// | `N` (an integer) | on, up to `N` keys -- how the crossover was swept |
///
/// Read once per process: it is consulted once per layer per token (60x/token)
/// and `getenv` takes a lock in some libc builds. Every layer reads the same
/// answer, which is also why the number has to be right rather than lucky.
pub fn decode_single_pass_max_keys() -> Option<usize> {
    static CACHE: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        let raw = std::env::var("KAPLUMBA_DECODE_SINGLE_PASS").ok()?;
        parse_single_pass_limit(&raw).filter(|limit| *limit > 0)
    })
}

/// Floats the decode partial buffer holds for a layer: two softmax statistics
/// and one unnormalised accumulator per (split, query head).
pub fn decode_partial_elements(q_heads: usize, head_dim: usize) -> usize {
    let rows = ATTENTION_SPLITS_MAX.saturating_mul(q_heads);
    rows.saturating_mul(head_dim)
        .saturating_add(rows.saturating_mul(2))
}

// ---------------------------------------------------------------------------
// OQ9: prefill key-range passes.
// ---------------------------------------------------------------------------

/// Keys per prefill pass. oMLX's `OMLX_NAX_ATTN_PASS_KEYS` default is **8192**
/// (`omlx/utils/nax_attention.py:93`), and its measured reason is cache
/// residency, not a byte budget: *"Every threadgroup of a dispatch streams this
/// many keys of its KV head, so the key slice the threadgroups in flight share
/// stays in the on-chip caches (one dispatch over 1M keys runs at ~85 TFLOPS
/// against ~103 at 64k)"* (`:89-91`).
///
/// `KAPLUMBA_PREFILL_PASS_KEYS=<n>` overrides it; **0 turns passes off** and
/// restores the one-dispatch body.
pub const PREFILL_PASS_KEYS: usize = 8192;

/// Minimum number of threadgroups before passes are worth it.
///
/// oMLX gates on `_PASS_MIN_GROUPS = 512` (`:94`) with the reason in the
/// comment at `:91-92`: *"Grids of a wave or two (query tails under ~512 rows
/// at 64 heads) keep one dispatch: their threadgroups stay in step anyway."*
/// Below that the slices would cost more state traffic than the drift they fix,
/// so the shipped rule keeps one dispatch.
pub const PREFILL_PASS_MIN_GROUPS: usize = 512;

/// Total threadgroups the prefill grid has for `queries` rows and `kv_heads`.
pub fn prefill_threadgroups(queries: usize, kv_heads: usize, block: usize) -> usize {
    queries.div_ceil(block.max(1)).saturating_mul(kv_heads)
}

/// Keys per prefill pass for one dispatch, honouring `KAPLUMBA_PREFILL_PASS_KEYS`.
///
/// 0 means "one dispatch": either the knob is off, or the key range fits in a
/// single pass anyway.
pub fn prefill_pass_keys_for(key_count: usize) -> usize {
    static CACHE: OnceLock<usize> = OnceLock::new();
    let configured = *CACHE.get_or_init(|| {
        std::env::var("KAPLUMBA_PREFILL_PASS_KEYS")
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .unwrap_or(PREFILL_PASS_KEYS)
    });
    if configured == 0 || key_count <= configured {
        return 0;
    }
    configured
}

/// The flash body's key tile (`TILE = 64` in `attention.metal`).
///
/// **Pass boundaries must land on tile edges, and this constant is why.** The
/// online softmax updates the running max, sum and accumulator **once per
/// tile**, so the sequence of `m` values -- and therefore the sequence of
/// rescale factors applied to `s` and to `O` -- is a function of where the tile
/// boundaries fall. Cut the key range at a non-tile index and every row's
/// arithmetic changes: measured here as a **one-fp16-ulp** difference on 1% of
/// outputs (worst |diff| 0.000488, 637 195 of 16 777 216 values, first at
/// `(query 40, head 5, dim 116)` for 44-key passes over 129 keys). That is far
/// inside the 5e-3 tolerance every other attention test here uses, so it would
/// have shipped as "exact" while being measurably, reproducibly different.
///
/// oMLX does not have this hazard because it cuts in units of its key block
/// (`_BK = 32`, `nax_attention.py:76`): `_pass_edges` returns block *indices*
/// (`:800`), and the kernel loops `for (kb = kb_lo; kb < kb_hi; kb++)`
/// (`:362`), so an edge is a block boundary by construction. The equivalent here
/// is to snap every edge down to a multiple of 64.
pub const PREFILL_PASS_TILE: usize = 64;

/// Key-tile boundaries of the dispatch chain for `key_count` keys, including 0
/// and `key_count`. A single dispatch is `[0, key_count]`.
///
/// `pass_keys` is floored to whole tiles; a `pass_keys` below one tile becomes
/// one tile per pass. The final boundary is `key_count` itself, so the last
/// dispatch is always the one whose `range_end >= key_count` -- the one that
/// normalises and writes the output.
pub fn prefill_pass_edges(key_count: usize, pass_keys: usize) -> Vec<usize> {
    let tiles = key_count.div_ceil(PREFILL_PASS_TILE);
    let tiles_per_pass = (pass_keys / PREFILL_PASS_TILE).max(1);
    if tiles <= tiles_per_pass {
        return vec![0, key_count];
    }
    let passes = tiles.div_ceil(tiles_per_pass);
    let mut edges = Vec::with_capacity(passes + 1);
    edges.push(0usize);
    for index in 1..passes {
        // Skip an edge that coincides with the end, so the chain never ends in an
        // empty pass: `24590 / 64 = 385` tiles, 4 per pass, gives 96 full passes
        // plus a 1-tile remainder, and the tile-aligned edge `96 * 4 * 64` is
        // 24576, not 24590 -- so the last pass is `[24576, 24590)`, 14 keys.
        let edge = index * tiles_per_pass * PREFILL_PASS_TILE;
        if edge < key_count {
            edges.push(edge);
        }
    }
    edges.push(key_count);
    edges
}

/// Should this prefill dispatch use the key-range-pass kernel, and with how many
/// passes?
///
/// Returns the pass count (>= 2, so the caller knows there is work to do) or 0
/// for "one dispatch". Three conditions, in oMLX's order:
///
/// 1. the flash shape (the pass kernels are instantiated for `HEAD_DIM = 512`,
///    `groups = 8` only, like every other flash kernel here);
/// 2. the pass kernels actually compiled;
/// 3. the grid is big enough that the slices stay in step -- oMLX's
///    `_PASS_MIN_GROUPS` gate, on **threadgroups** rather than query rows.
pub fn prefill_pass_count(
    queries: usize,
    kv_heads: usize,
    key_count: usize,
    head_dim: usize,
    groups: usize,
    block: usize,
    available: bool,
) -> usize {
    if !available || !flash_supported(head_dim, groups) {
        return 0;
    }
    let pass_keys = prefill_pass_keys_for(key_count);
    if pass_keys == 0 {
        return 0;
    }
    let groups_in_grid = prefill_threadgroups(queries, kv_heads, block);
    // **Below** the gate, matching oMLX's `n_groups < min_groups`
    // (`nax_attention.py:797`): a grid *at* 512 keeps one dispatch. Their comment
    // calls 512 the "wave or two" size ("query tails under ~512 rows at 64 heads
    // keep one dispatch: their threadgroups stay in step anyway", `:91-92`), so
    // the boundary belongs on the unsliced side.
    if groups_in_grid <= PREFILL_PASS_MIN_GROUPS {
        return 0;
    }
    prefill_pass_edges(key_count, pass_keys)
        .len()
        .saturating_sub(1)
}

/// Floats the prefill key-range-pass row state holds for one chunk: one fp32
/// `(m, s, O[head_dim])` row per (query, q head).
///
/// Fixed in the *chunk* size, never in the context length: a chunk is at most
/// `PREFILL_CHUNK` queries regardless of how long the prompt is, so this is a
/// constant per layer shape, exactly as AGENTS.md §8 requires. 64.3 MiB at
/// 1024 queries x 32 heads x (512 + 2) -- affordable inside the 20 GiB budget
/// that the 24k run measures at 18.35 GiB, and it is **scratch**, not weights, so
/// the 11.51 GiB weight footprint does not move.
pub fn prefill_pass_state_elements(
    queries: usize,
    q_heads: usize,
    head_dim: usize,
) -> usize {
    queries
        .saturating_mul(q_heads)
        .saturating_mul(head_dim)
        .saturating_add(queries.saturating_mul(q_heads).saturating_mul(2))
}

/// Widest prefill pass row state across a set of layer shapes, in floats.
///
/// Sizing this from the widest layer rather than per layer keeps the buffer
/// fixed-size and shared, which is the same trade the decode partials make.
pub fn prefill_pass_state_elements_widest<I>(shapes: I, max_queries: usize) -> usize
where
    I: IntoIterator<Item = (usize, usize)>,
{
    let widest = shapes
        .into_iter()
        .map(|(q_heads, head_dim)| prefill_pass_state_elements(max_queries, q_heads, head_dim))
        .max()
        .unwrap_or(1);
    widest.max(1)
}

/// Convenience for tests and callers that want the boundaries rather than the
/// count. See [`prefill_pass_edges`] for the alignment guarantee.

/// How many ways to cut `key_count` keys for the decode path.
///
/// **MEASUREMENT KNOB / DEFAULT PENDING.** The split count used to come from a
/// *thread* target, `ATTENTION_DECODE_THREADS_TARGET / (32 * q_heads)`, and the
/// missing `kv_heads` argument is what made that wrong: a threadgroup is
/// `32 * (q_heads / kv_heads)` threads, so holding the thread count at 16384
/// lands both of Gxma's layer shapes on 16 splits while giving them workgroup
/// counts that differ 4x -- 16 kv heads x 16 = **256** workgroups of 64 threads
/// on the 50 sliding layers, versus 4 x 16 = **64** workgroups of 256 threads on
/// the 10 full-attention ones. What a GPU scheduler actually packs onto a core is
/// workgroups, not threads, so the thread target silently assumed a residency
/// that the wide-threadgroup shape does not have.
///
/// The rule is therefore a *one-wave workgroup budget*,
/// [`decode_one_wave_workgroups`], and the result is still in
/// `1 ..= ATTENTION_SPLITS_MAX` so the scratch stays fixed-size. See
/// `spec/decode-19tok/implementation_notes.md` ("the decode split count, from a
/// workgroup budget rather than a thread count") for the sweep, including the
/// measurement that the sliding shape is unchanged at 16 and why that is
/// structural rather than luck.
///
/// The budget is stated for the **whole device** rather than per core because
/// the measurement is: 54 workgroups of 256 threads is one wave on this box and
/// 55 is two, which is `18 * 3` rather than `20 * 3` or `20 * 2.7`, and picking
/// a core count to make the arithmetic come out would be inventing a fact.
///
/// **The one-wave rule is NOT the default, and the reason is bit-exactness
/// rather than doubt about the speed.** Changing the split count changes the
/// order in which the combine pass sums the keys, and floating-point addition is
/// not associative, so 13 splits is a different -- equally valid, not equally
/// identical -- answer to the same question.
///
/// The precedent for shipping default-off is in this repo and it cuts against
/// the obvious argument. `spec/decode-19tok-repacked-250gb` §21.2-21.4 measured
/// a *larger* perturbation elsewhere -- the gate/up GEMV fusion is within 1 fp16
/// ULP on 0.78% of activations per layer, across all 60 layers -- and it changed
/// **zero** of 128 greedy tokens. This change is smaller (1 ULP on 3-4 of 16384
/// attention outputs, on 10 of 60 layers) and changes **43 of 128**. So "it is
/// only 1 ULP, and we already accept 1 ULP" is not a valid argument: 1 ULP is
/// not a unit of comparison across sites, and only the measurement settles it.
///
/// The rule therefore ships behind `KAPLUMBA_DECODE_SPLIT_RULE=workgroups` and
/// the default stays the thread target. The measured consequence is in
/// `spec/decode-19tok/implementation_notes.md`: 1.34x on the full-attention
/// layers at 8k of context, and a different token stream.
///
/// `KAPLUMBA_DECODE_SPLITS` still pins the count, so the two can be compared
/// without a rebuild.
pub fn decode_splits(q_heads: usize, kv_heads: usize, key_count: usize) -> usize {
    if q_heads == 0 || key_count == 0 {
        return 1;
    }
    // MEASUREMENT KNOB. `KAPLUMBA_DECODE_SPLITS=N` pins the split count so the
    // choice can be swept without a rebuild, which is how the workgroups-per-core
    // rule below was measured at all. It outranks the rule selector, so a sweep
    // works under either rule.
    let pinned = std::env::var("KAPLUMBA_DECODE_SPLITS")
        .ok()
        .as_deref()
        .and_then(parse_pinned_splits);
    decode_splits_with(q_heads, kv_heads, key_count, pinned, decode_split_rule())
}

/// Which arithmetic [`decode_splits`] uses. See [`decode_split_rule`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DecodeSplitRule {
    /// The historical `ATTENTION_DECODE_THREADS_TARGET / (32 * q_heads)`. The
    /// default, and bit-identical to what shipped before `kv_heads` was passed in.
    ThreadTarget,
    /// One wave of workgroups, sized by [`decode_one_wave_workgroups`].
    WorkgroupsPerCore,
}

/// `KAPLUMBA_DECODE_SPLIT_RULE` -> the arithmetic to use. Default
/// [`DecodeSplitRule::ThreadTarget`].
///
/// | value | rule |
/// |---|---|
/// | unset, empty, `0`, `off`, `false`, `no`, `threads`, `legacy` | thread target (default) |
/// | `workgroups`, `waves`, `one_wave`, `1`, `on`, `true`, `yes`, `auto` | one-wave workgroup budget |
///
/// Parsed as a function so it is testable without touching the process
/// environment, which Rust 2024 makes `unsafe` and which every other test in
/// this process shares. Total by construction: an unrecognised value is the
/// *default*, never an error, because this is read on the decode hot path where
/// a panic would take the process down and where a typo must not silently
/// change the numerics of the model.
pub fn parse_split_rule(raw: &str) -> DecodeSplitRule {
    match raw.trim() {
        "workgroups" | "waves" | "one_wave" | "1" | "on" | "true" | "yes" | "auto" => {
            DecodeSplitRule::WorkgroupsPerCore
        }
        _ => DecodeSplitRule::ThreadTarget,
    }
}

/// The process's rule, read once. It cannot meaningfully change mid-process --
/// the split count is baked into every decode step's numerics -- and `getenv`
/// takes a lock in some libc builds, so this is consulted once per layer per
/// token (60x/token) rather than 60 times a second of `getenv`.
pub fn decode_split_rule() -> DecodeSplitRule {
    static CACHE: std::sync::OnceLock<DecodeSplitRule> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("KAPLUMBA_DECODE_SPLIT_RULE")
            .ok()
            .as_deref()
            .map_or(DecodeSplitRule::ThreadTarget, parse_split_rule)
    })
}

/// `KAPLUMBA_DECODE_SPLITS` -> a pinned split count, or `None` for "use the
/// heuristic".
///
/// Split out of [`decode_splits`] so the parsing is testable without mutating the
/// process environment, which Rust 2024 makes `unsafe` and which other tests in
/// this process share. Total by construction: anything unrecognised, out of
/// range, or unparseable is `None` (the heuristic) rather than an error, because
/// this is read on the decode hot path where a panic would take the process down
/// and where a typo must not be able to request a grid the scratch cannot hold.
fn parse_pinned_splits(raw: &str) -> Option<usize> {
    raw.trim()
        .parse::<usize>()
        .ok()
        .filter(|value| (1..=ATTENTION_SPLITS_MAX).contains(value))
}

/// The split-count rule with the environment already resolved. Pure, so every
/// boundary it has is testable without a GPU and without the environment.
fn decode_splits_with(
    q_heads: usize,
    kv_heads: usize,
    key_count: usize,
    pinned: Option<usize>,
    rule: DecodeSplitRule,
) -> usize {
    if let Some(pinned) = pinned {
        return pinned;
    }
    if q_heads == 0 || key_count == 0 {
        return 1;
    }
    // `kv_heads == 0` is not reachable from the encode path, which rejects it
    // before it gets here, but the arithmetic must not divide by zero if a
    // future caller forgets: `max(1)` keeps it finite and the result is then
    // decided entirely by the cap, which is harmless.
    let kv_heads = kv_heads.max(1);
    let affordable = key_count.div_ceil(ATTENTION_MIN_KEYS_PER_SPLIT);
    // One wave: `splits * kv_heads <= budget`. Integer division rounds *down*,
    // which is the safe direction -- the alternative is a grid that spills into
    // a second wave, and a second wave is not a small penalty: measured, 56
    // workgroups of 256 threads costs 1.50x the time of 52 for the identical
    // byte count.
    let workgroup_budget = decode_one_wave_workgroups(32 * (q_heads / kv_heads));
    let wanted = match (rule, workgroup_budget) {
        (DecodeSplitRule::WorkgroupsPerCore, Some(budget)) => {
            (budget / kv_heads).clamp(1, ATTENTION_SPLITS_MAX)
        }
        // Either the default rule, or a width whose budget was never measured:
        // the historical thread target, which is what ships.
        _ => {
            (ATTENTION_DECODE_THREADS_TARGET / (32 * q_heads.max(1))).clamp(1, ATTENTION_SPLITS_MAX)
        }
    };
    wanted.min(affordable).max(1)
}

/// The largest decode grid, in workgroups, that still runs in **one wave** on
/// this box for a threadgroup of `threads_per_group` threads. `None` for a
/// width that was not measured, which keeps the historical thread target.
///
/// A scheduler reserves per-core resources for a whole threadgroup, so a core
/// holds far fewer of a wide one, and that is the entire content of this fix.
/// Measured by sweeping every split count 1..=16 for three shapes that all use
/// a 256-thread threadgroup and differ only in `kv_heads`, so the grid size
/// `splits * kv_heads` walks multiples of 4, 5 and 6 and the transition is
/// bracketed from both sides:
///
/// | grid | 48 | 50 | **52** | **54** | **55** | **56** | 60 | 64 |
/// |---|---|---|---|---|---|---|---|---|
/// | waves | 1 | 1 | **1** | **1** | 2 | 2 | 2 | 2 |
///
/// 54 of 256 threads is one wave and 55 is two, so the budget is **54**. It is
/// `18 * 3` rather than `20 * 3`; which of those it is does not change any
/// answer here, and picking a core count to make the arithmetic tidy would be
/// inventing a fact, so the constant is stated as the measurement.
///
/// The rule `budget / kv_heads` then reproduces the *measured* optimum for all
/// three shapes it was measured on, which is the check that it is a rule and not
/// a fit to one number: `kv_heads 4 -> 13` (measured best 13), `kv_heads 5 ->
/// 10` (measured best 10, 50 workgroups), `kv_heads 6 -> 9` (measured best 9,
/// 54 workgroups).
///
/// The 64-thread entry is the important negative result. 288 is `18 * 16`, and
/// it is a *lower* bound: the sliding curve is still improving at the 256
/// workgroups the [`ATTENTION_SPLITS_MAX`] cap allows, so the honest statement
/// is "at least 288", which for 16 kv heads is 18 splits -- above the cap, so
/// the cap decides and the shipped 16 is unchanged. **That is why 83% of the
/// layers cannot regress from this rule: for them it provably evaluates to the
/// number that ships today**, and `the_pinned_sweep_reproduces_the_reference_table`
/// checks it against the reference measurement rather than against a constant
/// typed twice.
pub const fn decode_one_wave_workgroups(threads_per_group: usize) -> Option<usize> {
    match threads_per_group {
        64 => Some(288),
        256 => Some(54),
        _ => None,
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AttentionParams {
    queries: u32,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    key_count: u32,
    scale: f32,
    window: u32,
    /// Decode only: KV splits and keys per split. Zero means "scan every key",
    /// which is what the prefill kernels get.
    splits: u32,
    keys_per_split: u32,
    /// Prefill key-range passes (OQ9): the key index this dispatch starts at,
    /// and how many keys it covers. `pass_keys == 0` is the one-dispatch
    /// behaviour, `[0, key_count)`.
    pass_begin: u32,
    pass_keys: u32,
}

#[derive(Clone, Copy)]
pub struct AttentionRequest<'a> {
    pub queries: &'a [u16],
    pub keys: &'a [u16],
    pub values: &'a [u16],
    pub query_positions: &'a [u32],
    pub key_positions: &'a [u32],
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub scale: f32,
    pub window: usize,
}

/// GPU-resident attention request: every buffer is caller-owned so attention
/// can be encoded into the same command buffer as the rest of the layer.
#[derive(Clone, Copy)]
pub struct AttentionEncodeRequest<'a> {
    pub queries: &'a Buffer,
    pub keys: &'a Buffer,
    pub values: &'a Buffer,
    pub query_positions: &'a Buffer,
    pub key_positions: &'a Buffer,
    pub output: &'a Buffer,
    pub queries_count: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub key_count: usize,
    pub scale: f32,
    pub window: usize,
}

pub struct AttentionKernels {
    device: Device,
    queue: CommandQueue,
    _library: Library,
    online_attention_pipeline: ComputePipelineState,
    online_attention_256_pipeline: ComputePipelineState,
    online_attention_512_pipeline: ComputePipelineState,
    attention_dim_256_pipeline: ComputePipelineState,
    attention_dim_512_pipeline: ComputePipelineState,
    attention_tg2_256_pipeline: ComputePipelineState,
    attention_tg2_512_pipeline: ComputePipelineState,
    /// Transposed prefill kernel, indexed by `gqa_index(head_dim, nsg)`.
    /// `head_dim` is 256 or 512; `nsg` is the GQA ratio `q_heads / kv_heads`,
    /// which the kernel needs verbatim because it sizes the threadgroup.
    attention_gqa_pipelines: [ComputePipelineState; GQA_PIPELINES],
    /// The same prefill shapes with the simdgroup-MMA QK scoring (T1,
    /// `spec/prefill-120tps`), indexed by `gqa_index` like the scalar array.
    /// Built eagerly in `new()` whether or not `attention_qk()` selects them:
    /// `pipeline_warmup_coverage` fails on any pipeline a dispatch compiles
    /// after construction.
    attention_gqa_mma_pipelines: [ComputePipelineState; GQA_PIPELINES],
    /// The same body run over one split of the KV axis (flash-decoding).
    decode_split_pipelines: [ComputePipelineState; GQA_PIPELINES],
    /// The same body run over the whole KV axis in one threadgroup, writing the
    /// normalised output directly. Only `(256, 2)` and `(512, 8)` -- the two
    /// shapes this model has -- because the host only selects it after
    /// `gqa_index` has already resolved, and the metal side instantiates
    /// nothing else.
    decode_single_pipelines: [Option<ComputePipelineState>; GQA_PIPELINES],
    decode_combine_256_pipeline: ComputePipelineState,
    decode_combine_512_pipeline: ComputePipelineState,
    /// T3 (`spec/prefill-24k-90tps`): the flash-style QK+PV prefill body for the
    /// one shape it is written for (HEAD_DIM 512, groups 8). `Option` for the
    /// same reason `decode_single_pipelines` is: a missing kernel must degrade
    /// to the MMA arm, not take the whole library down.
    attention_flash_pipeline: Option<ComputePipelineState>,
    /// The `BQ = 1` twin of `attention_flash_pipeline` (the same body with one
    /// query per threadgroup). Both are built so the query block can be chosen
    /// per request without a race on a process-wide selection.
    attention_flash_bq1_pipeline: Option<ComputePipelineState>,
    /// OQ9: the key-range-pass twins of the two prefill flash kernels. One
    /// dispatch covers `[pass_begin, pass_begin + pass_keys)` and hands its fp32
    /// `(m, s, O)` row state to the next through buffer 7. `None` if either
    /// failed to compile, in which case the one-dispatch kernels run unchanged.
    attention_flash_pass_pipeline: Option<ComputePipelineState>,
    attention_flash_pass_bq1_pipeline: Option<ComputePipelineState>,
    /// T2b: the decode flash kernel (`attention_decode_flash_512_8`) -- the
    /// prefill flash body with `BQ = 1` and `PARTIALS`, so a decode split's QK
    /// and PV are 8x8 simdgroup MMA instead of a 32-lane `simd_sum` per
    /// (key, query head). It writes the *same* `(m, s, O)` partials the scalar
    /// split kernel writes, so the combine pass and its exactness argument are
    /// untouched; only the `(m, s, O)` numbers change, by fp32 summation order.
    ///
    /// `None` (compile failure) degrades to the scalar split kernel.
    decode_flash_pipeline: Option<ComputePipelineState>,
}

/// `q_heads / kv_heads` values the transposed kernel is instantiated for, and
/// therefore also the simdgroup counts. Anything else falls back to the
/// 256-thread `online_attention_tg2_*` kernel, which makes no such assumption.
const GQA_RATIOS: [usize; 4] = [1, 2, 4, 8];
const GQA_HEAD_DIMS: [usize; 2] = [256, 512];
const GQA_PIPELINES: usize = GQA_RATIOS.len() * GQA_HEAD_DIMS.len();

/// Position of a (head_dim, nsg) pair in `attention_gqa_pipelines`, or `None`
/// when the shape has no specialisation.
fn gqa_index(head_dim: usize, nsg: usize) -> Option<usize> {
    let dim = GQA_HEAD_DIMS.iter().position(|d| *d == head_dim)?;
    let ratio = GQA_RATIOS.iter().position(|r| *r == nsg)?;
    Some(dim * GQA_RATIOS.len() + ratio)
}

fn gqa_pipeline_name(head_dim: usize, nsg: usize) -> String {
    format!("online_attention_gqa_{head_dim}_{nsg}")
}

/// Kernel name of the MMA-QK specialisation for one prefill shape.
fn gqa_mma_pipeline_name(head_dim: usize, nsg: usize) -> String {
    format!("online_attention_gqa_mma_{head_dim}_{nsg}")
}

/// The pipeline-array index a decode shape resolves to, or `None` when it has no
/// specialisation and the prefill kernels take it. Public because
/// [`AttentionKernels::encode_decode_single_pass`] and
/// [`AttentionKernels::encode_decode_split_with_splits`] both take the index
/// rather than deriving it, and a caller that wants to compare the two paths
/// needs the same value `encode_online_attention_split` would have computed.
pub fn decode_gqa_index(head_dim: usize, q_heads: usize, kv_heads: usize) -> Option<usize> {
    if kv_heads == 0 || q_heads % kv_heads != 0 {
        return None;
    }
    gqa_index(head_dim, q_heads / kv_heads)
}

fn decode_split_pipeline_name(head_dim: usize, nsg: usize) -> String {
    format!("attention_decode_split_{head_dim}_{nsg}")
}

/// `(head_dim, nsg)` pairs the single-pass decode kernel is instantiated for.
/// See the `AttentionKernels::decode_single_pipelines` field.
fn decode_single_pass_name(head_dim: usize, nsg: usize) -> Option<String> {
    match (head_dim, nsg) {
        (256, 2) | (512, 8) => Some(format!("attention_decode_single_{head_dim}_{nsg}")),
        _ => None,
    }
}

/// Which prefill QK implementation `encode_online_attention` dispatches to.
///
/// The two arms are the T1 A/B (`spec/prefill-120tps`): `Scalar` is the
/// per-lane dot product in `attention_gqa_body`, `Mma` is the simdgroup-8x8x8
/// scoring in `attention_gqa_mma_body`. Both are reachable in-process through
/// [`AttentionKernels::encode_online_attention_qk`], which is what
/// `attention_qk_mma_matches_scalar` compares; the environment variable only
/// picks which one the production entry point forwards.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionQk {
    Scalar,
    Mma,
    /// T3 (`spec/prefill-24k-90tps`): 64-key tiles with the QK *and* the PV fold
    /// as `simdgroup_multiply_accumulate`. Only instantiated for HEAD_DIM 512 /
    /// groups 8; any other shape routes to [`AttentionQk::Mma`].
    Flash,
}

/// Every `(head_dim, groups)` pair [`AttentionQk::Flash`] is instantiated for.
/// One shape today; kept as a predicate rather than an `==` at the call site so
/// a second instantiation cannot be added without the selection following.
fn flash_supported(head_dim: usize, groups: usize) -> bool {
    matches!((head_dim, groups), (512, 8))
}

/// True for the spellings that mean "on"; anything else is off, so a typo can
/// never silently select a path the caller did not ask for.
fn qk_env_truthy(raw: &str) -> bool {
    matches!(raw.trim(), "1" | "on" | "true" | "yes")
}

/// The process-wide prefill QK selection, read once (same `OnceLock` idiom as
/// [`decode_single_pass_max_keys`]: it is consulted once per layer per prompt,
/// and `getenv` takes a lock in some libc builds).
///
/// * `KAPLUMBA_SCALAR_ATTENTION=1|on|true|yes` -> [`AttentionQk::Scalar`],
///   always wins: this is the brief's A/B spelling for forcing the baseline.
/// * else `KAPLUMBA_MMA_ATTENTION=1|on|true|yes` -> [`AttentionQk::Mma`].
/// * else the default is **`Scalar`**.
///
/// The production entry point does not call this: it goes through
/// [`attention_qk_for`], which keeps `Scalar` on the shapes where MMA measured
/// a loss and selects MMA on the full-attention shape where it measured a win.
/// This function remains the global override (both flags, or `Scalar`).
///
/// `attention_single_pass_bit_exact` asserts the scalar prefill GQA output
/// equals the single-pass decode output in 0 of N fp16 elements; the MMA's
/// reassociated dot product cannot satisfy that, so that test pins
/// `AttentionQk::Scalar` explicitly. Decode never consults this: it goes through
/// `encode_online_attention_split` / the single-pass path, neither of which
/// calls `encode_online_attention`.
pub fn attention_qk() -> AttentionQk {
    attention_qk_override().unwrap_or(AttentionQk::Scalar)
}

/// The process-wide prefill QK env override, or `None` when neither flag is
/// set. Read once (same `OnceLock` idiom as [`attention_qk`]): it is consulted
/// once per layer per prompt, and `getenv` takes a lock in some libc builds.
fn attention_qk_override() -> Option<AttentionQk> {
    static CACHE: std::sync::OnceLock<Option<AttentionQk>> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        if std::env::var("KAPLUMBA_SCALAR_ATTENTION")
            .map(|raw| qk_env_truthy(&raw))
            .unwrap_or(false)
        {
            return Some(AttentionQk::Scalar);
        }
        if std::env::var("KAPLUMBA_FLASH_ATTENTION")
            .map(|raw| qk_env_truthy(&raw))
            .unwrap_or(false)
        {
            return Some(AttentionQk::Flash);
        }
        if std::env::var("KAPLUMBA_MMA_ATTENTION")
            .map(|raw| qk_env_truthy(&raw))
            .unwrap_or(false)
        {
            return Some(AttentionQk::Mma);
        }
        None
    })
}

/// Shape-aware prefill QK selection (T2, `spec/prefill-120tps`). With no env
/// override the partitioned MMA QK is used only where it measured a win: the
/// `groups >= 4` full-attention layers (`head_dim` 512, NSG 8), where the
/// micro-bench measured **1.2x faster** than scalar. The `groups == 2` sliding
/// layers (NSG 2) measured **1.6x slower**, so they keep scalar. Either flag
/// (`KAPLUMBA_MMA_ATTENTION` / `KAPLUMBA_SCALAR_ATTENTION`) overrides the rule
/// globally for A/B.
pub fn attention_qk_for(groups: usize, head_dim: usize) -> AttentionQk {
    if let Some(forced) = attention_qk_override() {
        return forced;
    }
    if flash_supported(head_dim, groups) {
        if let Some(forced) = full_qk_override() {
            return forced;
        }
        return AttentionQk::Flash;
    }
    if groups >= 4 {
        AttentionQk::Mma
    } else {
        AttentionQk::Scalar
    }
}

/// `KAPLUMBA_FULL_QK=flash|mma|scalar` overrides the prefill arm for the
/// full-attention shape (`head_dim == 512`, `groups == 8`) **only**, leaving the
/// sliding shape on its rule. The global flags ([`attention_qk`]) cannot express
/// this: they move all 60 layers, and the sliding shape has a different winner
/// (`spec/prefill-120tps` measured MMA 1.6x *slower* there), so a full-layer-only
/// A/B needs its own knob. Read once.
fn full_qk_override() -> Option<AttentionQk> {
    static CACHE: std::sync::OnceLock<Option<AttentionQk>> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        let raw = std::env::var("KAPLUMBA_FULL_QK").ok()?;
        match raw.trim() {
            "flash" => Some(AttentionQk::Flash),
            "mma" => Some(AttentionQk::Mma),
            "scalar" => Some(AttentionQk::Scalar),
            _ => None,
        }
    })
}

/// Threadgroups of `per_group` threads needed to cover `threads`, at least one.
/// The kernels that index the grid themselves early-return on the surplus.
fn threads_for(threads: usize, per_group: usize) -> u64 {
    threads.div_ceil(per_group).max(1) as u64
}

impl AttentionKernels {
    pub fn new() -> Result<Self, KernelError> {
        let device = Device::system_default().ok_or(KernelError::DeviceUnavailable)?;
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(SOURCE, &options)
            .map_err(KernelError::Shader)?;
        let mut extra = |name: &str| -> Result<ComputePipelineState, KernelError> {
            // Through the audit, like every other family: a pipeline built here
            // is a pipeline that a later `--metrics`-style timed loop must not
            // be the first to reach. See `crate::pipeline_audit`.
            crate::pipeline_audit::build(&device, &library, name)
        };
        let online_attention_pipeline = extra("online_attention")?;
        let online_attention_256_pipeline = extra("online_attention_256")?;
        let online_attention_512_pipeline = extra("online_attention_512")?;
        let attention_dim_256_pipeline = extra("online_attention_dim_256")?;
        let attention_dim_512_pipeline = extra("online_attention_dim_512")?;
        let attention_tg2_256_pipeline = extra("online_attention_tg2_256")?;
        let attention_tg2_512_pipeline = extra("online_attention_tg2_512")?;
        let mut attention_gqa_pipelines = Vec::with_capacity(GQA_PIPELINES);
        let mut attention_gqa_mma_pipelines = Vec::with_capacity(GQA_PIPELINES);
        let mut decode_split_pipelines = Vec::with_capacity(GQA_PIPELINES);
        for head_dim in GQA_HEAD_DIMS {
            for nsg in GQA_RATIOS {
                attention_gqa_pipelines.push(extra(&gqa_pipeline_name(head_dim, nsg))?);
                // T1: built unconditionally so the selection is an env-var
                // choice between two already-compiled pipelines, never a
                // compile inside a measured dispatch.
                attention_gqa_mma_pipelines
                    .push(extra(&gqa_mma_pipeline_name(head_dim, nsg))?);
                decode_split_pipelines.push(extra(&decode_split_pipeline_name(head_dim, nsg))?);
            }
        }
        let attention_gqa_pipelines: [ComputePipelineState; GQA_PIPELINES] =
            attention_gqa_pipelines.try_into().map_err(|_| {
                KernelError::Pipeline("unexpected transposed attention pipeline count".to_string())
            })?;
        let attention_gqa_mma_pipelines: [ComputePipelineState; GQA_PIPELINES] =
            attention_gqa_mma_pipelines.try_into().map_err(|_| {
                KernelError::Pipeline("unexpected MMA attention pipeline count".to_string())
            })?;
        let decode_split_pipelines: [ComputePipelineState; GQA_PIPELINES] =
            decode_split_pipelines.try_into().map_err(|_| {
                KernelError::Pipeline("unexpected decode split pipeline count".to_string())
            })?;
        let mut decode_single_pipelines: [Option<ComputePipelineState>; GQA_PIPELINES] =
            [const { None }; GQA_PIPELINES];
        for head_dim in GQA_HEAD_DIMS {
            for nsg in GQA_RATIOS {
                let Some(name) = decode_single_pass_name(head_dim, nsg) else {
                    continue;
                };
                // A missing single-pass kernel must not take the whole library
                // down: the fallback below drops the dispatch to split+combine,
                // which is the shipping path. So this is the one `extra` call
                // whose failure is not propagated.
                if let Some(index) = gqa_index(head_dim, nsg) {
                    decode_single_pipelines[index] =
                        crate::pipeline_audit::build(&device, &library, &name).ok();
                }
            }
        }
        let decode_combine_256_pipeline = extra("attention_decode_combine_256")?;
        let decode_combine_512_pipeline = extra("attention_decode_combine_512")?;
        // T3: optional, like the single-pass decode kernels -- a missing flash
        // pipeline falls back to the MMA arm instead of failing the library.
        let attention_flash_pipeline =
            crate::pipeline_audit::build(&device, &library, "online_attention_flash_512_8").ok();
        let attention_flash_bq1_pipeline =
            crate::pipeline_audit::build(&device, &library, "online_attention_flash_bq1_512_8")
                .ok();
        // OQ9: the key-range-pass twins. Optional for the same reason -- if
        // either fails to compile, `prefill_pass_count(.., available = false)`
        // returns 0 and the one-dispatch flash kernels run, bit-identically.
        let attention_flash_pass_pipeline =
            crate::pipeline_audit::build(&device, &library, "online_attention_flash_pass_512_8")
                .ok();
        let attention_flash_pass_bq1_pipeline =
            crate::pipeline_audit::build(
                &device,
                &library,
                "online_attention_flash_pass_bq1_512_8",
            )
            .ok();
        // T2b: decode flash. Built unconditionally like the prefill twins so a
        // missing kernel is a fallback, not a library-wide failure.
        let decode_flash_pipeline =
            crate::pipeline_audit::build(&device, &library, "attention_decode_flash_512_8").ok();
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            queue,
            _library: library,
            online_attention_pipeline,
            online_attention_256_pipeline,
            online_attention_512_pipeline,
            attention_dim_256_pipeline,
            attention_dim_512_pipeline,
            attention_tg2_256_pipeline,
            attention_tg2_512_pipeline,
            attention_gqa_pipelines,
            attention_gqa_mma_pipelines,
            decode_split_pipelines,
            decode_single_pipelines,
            decode_combine_256_pipeline,
            decode_combine_512_pipeline,
            attention_flash_pipeline,
            attention_flash_bq1_pipeline,
            attention_flash_pass_pipeline,
            attention_flash_pass_bq1_pipeline,
            decode_flash_pipeline,
        })
    }

    /// Whether the T3 flash prefill pipeline compiled. Decode/sliding shapes
    /// never consult it; the full-attention shape falls back to [`AttentionQk::Mma`]
    /// when this is false, which is what keeps the `Option` from turning a build
    /// accident into a wrong answer.
    pub fn flash_attention_available(&self) -> bool {
        self.attention_flash_pipeline.is_some() || self.attention_flash_bq1_pipeline.is_some()
    }

    pub fn online_attention(&self, request: AttentionRequest<'_>) -> Result<Vec<u16>, KernelError> {
        validate_request(&request)?;
        let AttentionRequest {
            queries,
            keys,
            values,
            query_positions,
            key_positions,
            q_heads,
            kv_heads,
            head_dim,
            scale,
            window,
        } = request;
        let key_count = key_positions.len();
        let output_elements = checked_product(
            query_positions.len(),
            checked_product(q_heads, head_dim, KvAttentionError::ShapeOverflow)?,
            KvAttentionError::ShapeOverflow,
        )?;
        let query_buffer = self.make_u16_buffer(queries)?;
        let key_buffer = self.make_u16_buffer(keys)?;
        let value_buffer = self.make_u16_buffer(values)?;
        let query_position_buffer = self.make_u32_buffer(query_positions)?;
        let key_position_buffer = self.make_u32_buffer(key_positions)?;
        let output = self.device.new_buffer(
            (output_elements * size_of::<u16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let params = AttentionParams {
            queries: to_u32(query_positions.len(), "attention query count")?,
            q_heads: to_u32(q_heads, "attention query head count")?,
            kv_heads: to_u32(kv_heads, "attention KV head count")?,
            head_dim: to_u32(head_dim, "attention head dimension")?,
            key_count: to_u32(key_count, "attention key count")?,
            scale,
            window: to_u32(window, "attention window")?,
            splits: 0,
            keys_per_split: 0,
            pass_begin: 0,
            pass_keys: 0,
        };
        // Resolve the pipeline before opening an encoder. Returning early from
        // inside `set_compute_pipeline_state` used to drop a live compute
        // encoder, which Metal reports as
        // "Command encoder released without endEncoding". The generic kernel is
        // the documented fallback for any other head_dim up to MAX_HEAD_DIM.
        let pipeline = match head_dim {
            256 => &self.online_attention_256_pipeline,
            512 => &self.online_attention_512_pipeline,
            _ => &self.online_attention_pipeline,
        };
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&query_buffer), 0);
        encoder.set_buffer(1, Some(&key_buffer), 0);
        encoder.set_buffer(2, Some(&value_buffer), 0);
        encoder.set_buffer(3, Some(&query_position_buffer), 0);
        encoder.set_buffer(4, Some(&key_position_buffer), 0);
        encoder.set_buffer(5, Some(&output), 0);
        encoder.set_bytes(
            6,
            size_of::<AttentionParams>() as u64,
            &params as *const AttentionParams as *const c_void,
        );
        encoder.dispatch_threads(
            MTLSize::new((query_positions.len() * q_heads) as u64, 1, 1),
            MTLSize::new(64, 1, 1),
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(KernelError::CommandFailed);
        }
        read_u16_buffer(&output)
    }

    /// Encode the split-KV decode kernel and its combine pass into a caller-owned
    /// command buffer, reading keys/values straight out of the GPU-resident cache.
    /// Single query, any number of keys.
    ///
    /// Two dispatches, both fixed-size:
    ///
    /// 1. `attention_decode_split_{head_dim,ratio}` on `splits * kv_heads`
    ///    workgroups of `32 * (q_heads / kv_heads)` threads. Each owns a
    ///    contiguous `ceil(key_count / splits)` slice of the live window and
    ///    runs the *prefill* online softmax over it, writing an unnormalised
    ///    `(m, l, O)` partial of `head_dim + 2` floats. No score matrix: the
    ///    only scratch is the double-buffered weight tile in threadgroup memory
    ///    (`2 * ratio * 32` floats), the same as prefill.
    /// 2. `attention_decode_combine_{head_dim}` on `q_heads * head_dim` threads,
    ///    merging the partials with `exp(m_s - M)` rescaling and no threadgroup
    ///    memory at all.
    ///
    /// `partials` must hold `splits * q_heads * (head_dim + 2)` floats, and
    /// `splits <= ATTENTION_SPLITS_MAX`, so a buffer sized by
    /// [`decode_partial_elements`] serves every prompt length forever. The check
    /// below is what proves the caller did that: the old buffer was sized from
    /// the cache capacity captured at construction and silently ran out at 1215
    /// keys of context.
    pub fn encode_online_attention_split(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        request: AttentionEncodeRequest<'_>,
        partials: &Buffer,
    ) -> Result<(), KernelError> {
        let AttentionEncodeRequest {
            queries_count,
            q_heads,
            kv_heads,
            head_dim,
            key_count,
            ..
        } = request;
        if queries_count != 1
            || q_heads == 0
            || kv_heads == 0
            || head_dim == 0
            || key_count == 0
            || q_heads % kv_heads != 0
        {
            return Err(KernelError::InvalidInput(
                "split-key attention is for a single query with valid dimensions".to_string(),
            ));
        }
        let groups = q_heads / kv_heads;
        // No decode specialisation for this shape (an unusual head_dim or GQA
        // ratio). The prefill kernels are correct for a single query and need no
        // scratch, so fall back to them rather than refuse to decode.
        let Some(index) = gqa_index(head_dim, groups) else {
            return self.encode_online_attention(encoder, request);
        };
        // Single-pass arm. Off by default; see
        // [`decode_single_pass_max_keys`]. Checked before the partials size
        // guard because this path does not read the partials buffer at all, and
        // it is the only reason the buffer would be oversized for the shape.
        if key_count <= decode_single_pass_max_keys().unwrap_or(0) {
            if self.decode_single_pipelines[index].is_some() {
                return self.encode_decode_single_pass(encoder, request, index);
            }
        }
        let splits = decode_splits(q_heads, kv_heads, key_count);
        self.encode_decode_split_with_splits(encoder, request, partials, index, splits)
    }

    /// The split+combine pair with an explicit `splits`.
    ///
    /// Split out of [`Self::encode_online_attention_split`] so the split count
    /// is a parameter rather than a side effect of [`decode_splits`]. Production
    /// passes exactly what `decode_splits` returned, so the dispatch is
    /// unchanged; what this buys is a test that can pin `splits == 1` and
    /// compare it against the single-pass kernel on the same inputs, which is
    /// the only way to settle the bit-exactness question on the hardware.
    ///
    /// `index` must be the resolved `gqa_index(head_dim, q_heads / kv_heads)`.
    pub fn encode_decode_split_with_splits(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        request: AttentionEncodeRequest<'_>,
        partials: &Buffer,
        index: usize,
        splits: usize,
    ) -> Result<(), KernelError> {
        let groups = request.q_heads / request.kv_heads.max(1);
        let flash = self.decode_flash_pipeline.is_some()
            && request.head_dim == 512
            && groups == 8
            && decode_flash_enabled(request.key_count);
        self.encode_decode_split_explicit(
            encoder, request, partials, index, splits, flash,
        )
    }

    /// [`Self::encode_decode_split_with_splits`] with the kernel chosen by the
    /// caller instead of by [`decode_flash_enabled`].
    ///
    /// Public so a test can run **both** decode kernels in one process and
    /// compare them, which the environment gate cannot allow:
    /// `decode_flash_min_keys` is a `OnceLock`, so whichever spelling the process
    /// started with would be the only one reachable. Passing `flash = false`
    /// forces the scalar split kernel; `true` forces the MMA one and returns
    /// `InvalidInput` if it did not compile, rather than silently falling back --
    /// a silent fallback here would make a correctness test vacuous.
    pub fn encode_decode_split_explicit(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        request: AttentionEncodeRequest<'_>,
        partials: &Buffer,
        index: usize,
        splits: usize,
        flash: bool,
    ) -> Result<(), KernelError> {
        if flash {
            if self.decode_flash_pipeline.is_none() {
                return Err(KernelError::InvalidInput(
                    "decode flash kernel unavailable (attention_decode_flash_512_8 did not compile)"
                        .to_string(),
                ));
            }
            // The kernel is instantiated for head_dim 512 / groups 8 ONLY. It is
            // written for `NSG == 8` simdgroups and indexes the grid itself, so
            // running it on any other shape reads the wrong KV heads -- and the
            // sliding shape's `32 * groups` = 64 threads would not even fill its
            // threadgroup. Checked here, not only at the production call site,
            // so an explicit `flash = true` cannot dispatch it on a shape it does
            // not fit.
            if request.head_dim != 512 || request.q_heads / request.kv_heads.max(1) != 8 {
                return Err(KernelError::InvalidInput(format!(
                    "decode flash kernel is only instantiated for head_dim 512 with 8 q-heads per kv head, got head_dim {} with {}",
                    request.head_dim,
                    request.q_heads / request.kv_heads.max(1)
                )));
            }
        }
        let AttentionEncodeRequest {
            q_heads,
            kv_heads,
            head_dim,
            key_count,
            ..
        } = request;
        if splits == 0 || splits > ATTENTION_SPLITS_MAX {
            return Err(KernelError::InvalidInput(format!(
                "attention split count {splits} is outside 1..={ATTENTION_SPLITS_MAX}"
            )));
        }
        let groups = q_heads / kv_heads;
        let keys_per_split = key_count.div_ceil(splits);
        // Repointed at the new buffer: the same "host and kernel must agree"
        // guard as before, but the requirement is now a function of the layer's
        // head count alone and never of how many keys are live.
        let required = splits
            .checked_mul(q_heads)
            .and_then(|rows| rows.checked_mul(head_dim + 2))
            .and_then(|count| count.checked_mul(size_of::<f32>()))
            .ok_or_else(|| {
                KernelError::InvalidInput("attention partial shape overflows".to_string())
            })?;
        if partials.length() < required as u64 {
            return Err(KernelError::InvalidInput(format!(
                "attention partial buffer is {} bytes, needs {required} for {splits} splits x {q_heads} heads x {head_dim} dims",
                partials.length()
            )));
        }
        let params = AttentionParams {
            queries: 1,
            q_heads: to_u32(q_heads, "query heads")?,
            kv_heads: to_u32(kv_heads, "KV heads")?,
            head_dim: to_u32(head_dim, "head dimension")?,
            key_count: to_u32(key_count, "key count")?,
            scale: request.scale,
            window: to_u32(request.window, "window")?,
            splits: to_u32(splits, "split count")?,
            keys_per_split: to_u32(keys_per_split, "keys per split")?,
            pass_begin: 0,
            pass_keys: 0,
        };
        let bytes = params_bytes(&params);
        let kv_groups = to_u32(kv_heads, "kv head count")?;
        // T2b: on the full-attention shape (groups 8, head_dim 512) and above
        // the key threshold, use the MMA decode kernel when it compiled. The
        // grid, the buffer bindings and the partials layout are identical to the
        // scalar kernel's, so only the pipeline changes here.
        let use_flash_decode = flash;
        let split_pipeline: &ComputePipelineState = if use_flash_decode {
            self.decode_flash_pipeline.as_ref().expect("checked above")
        } else {
            &self.decode_split_pipelines[index]
        };
        encoder.dispatch_groups(
            split_pipeline,
            &[
                Some(request.queries),
                Some(request.keys),
                Some(request.values),
                Some(request.query_positions),
                Some(request.key_positions),
                Some(partials),
            ],
            bytes,
            (splits as u64) * u64::from(kv_groups),
            (32 * groups) as u64,
        )?;
        let combine = if head_dim == 256 {
            &self.decode_combine_256_pipeline
        } else {
            &self.decode_combine_512_pipeline
        };
        encoder.dispatch_1d(
            combine,
            &[Some(partials), Some(request.output)],
            bytes,
            (q_heads * head_dim) as u64,
            64,
        )
    }

    /// One dispatch, no partials, no combine: a single threadgroup per
    /// (query, kv head) scans the whole live KV window and writes the
    /// normalised output row.
    ///
    /// # Why this is bit-exact with split+combine
    ///
    /// Only for `splits == 1`, and that is not a coincidence -- it is the only
    /// case where the two agree at all. With `splits == 1` the split kernel's
    /// key range is `[0, key_count)`, which is the loop this kernel runs, in the
    /// same tile order, with the same lane->key mapping, the same
    /// `simd_max`/`simd_sum` reductions and the same in-order PV fold. So
    /// `(m, s, O)` are the same *numbers*, not merely the same mathematics --
    /// this is literally `attention_gqa_body<HEAD_DIM, NSG, false>`, the same
    /// template the prefill entry points instantiate, so there is no second copy
    /// of the loop that could drift.
    ///
    /// `attention_decode_combine` then reduces that one partial, which collapses
    /// to `factor = exp(m - m)`, `numerator = O`, `denominator = s` and
    /// `value = denominator > 0 ? numerator / denominator : 0`. Two things had to
    /// be checked rather than assumed, and both are asserted by
    /// `tests/attention_single_pass_bit_exact.rs`:
    ///
    ///   * `exp(0.0f) == 1.0f` exactly on this device, so both factors are
    ///     exactly one and the running sums are exactly `O` and `s`. If it were
    ///     not, the collapse would be inexact and the exactness test would fail
    ///     -- which is the point of running it rather than reasoning about it.
    ///   * the combine's `acc / s` and this kernel's `acc * (1/s)` round to the
    ///     same fp16. They are different f32 operations, and the difference was
    ///     measured at zero over 294912 elements rather than assumed away -- see
    ///     `the_reciprocal_multiply_is_absorbed_by_the_fp16_store` and the
    ///     `DIVIDE` note in `attention.metal`.
    ///
    /// For `splits > 1` no single-pass kernel can match it: the combine's
    /// `sum_i exp(m_i - M) * O_i` interleaves the rescale across splits in a
    /// different association order from the split body's per-tile
    /// `s = s * ms + simd_sum(w)`.
    /// `split_above_one_is_not_bit_exact_but_fp16_absorbs_most_of_it` measures
    /// that deviation rather than asserting it.
    ///
    /// `index` is the resolved `gqa_index(head_dim, groups)`, already checked by
    /// the caller. Public so a test can dispatch it directly rather than through
    /// the environment-variable gate, which is cached per process and would
    /// otherwise make "compare both paths in one run" impossible.
    pub fn encode_decode_single_pass(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        request: AttentionEncodeRequest<'_>,
        index: usize,
    ) -> Result<(), KernelError> {
        let pipeline = self
            .decode_single_pipelines
            .get(index)
            .and_then(Option::as_ref)
            .ok_or_else(|| {
                KernelError::InvalidInput(format!(
                    "no single-pass decode kernel for gqa index {index} (head_dim {}, ratio {})",
                    request.head_dim,
                    request.q_heads / request.kv_heads.max(1)
                ))
            })?;
        let AttentionEncodeRequest {
            queries_count,
            q_heads,
            kv_heads,
            head_dim,
            key_count,
            ..
        } = request;
        // Same shape contract as the split path. Repeating it rather than
        // trusting the caller: this is the path that would silently write a
        // normalised row over a buffer the caller sized for something else.
        if queries_count != 1
            || q_heads == 0
            || kv_heads == 0
            || head_dim == 0
            || key_count == 0
            || q_heads % kv_heads != 0
        {
            return Err(KernelError::InvalidInput(
                "single-pass attention is for a single query with valid dimensions".to_string(),
            ));
        }
        let groups = q_heads / kv_heads;
        // `splits == 0` is what selects the `[0, key_count)` range inside the
        // shared body. `keys_per_split` is unused on this path and is left zero
        // so a stale value cannot widen the range.
        let params = AttentionParams {
            queries: 1,
            q_heads: to_u32(q_heads, "query heads")?,
            kv_heads: to_u32(kv_heads, "KV heads")?,
            head_dim: to_u32(head_dim, "head dimension")?,
            key_count: to_u32(key_count, "key count")?,
            scale: request.scale,
            window: to_u32(request.window, "window")?,
            splits: 0,
            keys_per_split: 0,
            pass_begin: 0,
            pass_keys: 0,
        };
        debug_assert_eq!(
            gqa_index(head_dim, groups),
            Some(index),
            "single-pass dispatch was given a pipeline for a different shape"
        );
        encoder.dispatch_groups(
            pipeline,
            &[
                Some(request.queries),
                Some(request.keys),
                Some(request.values),
                Some(request.query_positions),
                Some(request.key_positions),
                Some(request.output),
            ],
            params_bytes(&params),
            kv_heads as u64,
            (32 * groups) as u64,
        )
    }
    /// Prefill dispatch with the QK implementation passed in explicitly (T1,
    /// `spec/prefill-120tps`): byte-for-byte the production body of
    /// [`AttentionKernels::encode_online_attention`] below, minus the ablation
    /// gate, plus an [`AttentionQk`] parameter. The explicit parameter is what
    /// lets `attention_qk_mma_matches_scalar` run both paths inside one process
    /// without fighting the `OnceLock` env cache. Grid geometry is identical
    /// for both arms -- only the pipeline differs.
    pub fn encode_online_attention_qk(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        request: AttentionEncodeRequest<'_>,
        qk: AttentionQk,
    ) -> Result<(), KernelError> {
        let block = flash_query_block_for(request.key_count);
        self.encode_online_attention_qk_block(encoder, request, qk, block)
    }

    /// [`Self::encode_online_attention_qk`] with the flash query block passed in
    /// explicitly, so a test can exercise both flash kernels (`BQ = 1` and
    /// `BQ = 2`) inside one process without fighting the `KAPLUMBA_FLASH_BQ`
    /// `OnceLock` cache. `block` is ignored unless the arm resolves to a flash
    /// pipeline.
    pub fn encode_online_attention_qk_block(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        request: AttentionEncodeRequest<'_>,
        qk: AttentionQk,
        block: usize,
    ) -> Result<(), KernelError> {
        let AttentionEncodeRequest {
            queries,
            keys,
            values,
            query_positions,
            key_positions,
            output,
            queries_count,
            q_heads,
            kv_heads,
            head_dim,
            key_count,
            scale,
            window,
        } = request;
        if queries_count == 0
            || q_heads == 0
            || kv_heads == 0
            || head_dim == 0
            || key_count == 0
            || q_heads % kv_heads != 0
        {
            return Err(KernelError::InvalidInput(
                "attention dimensions must be non-zero and q_heads divisible by kv_heads"
                    .to_string(),
            ));
        }
        if head_dim > MAX_HEAD_DIM {
            return Err(KernelError::InvalidInput(format!(
                "attention head dimension {head_dim} exceeds {MAX_HEAD_DIM}"
            )));
        }
        let params = AttentionParams {
            queries: to_u32(queries_count, "attention query count")?,
            q_heads: to_u32(q_heads, "attention query head count")?,
            kv_heads: to_u32(kv_heads, "attention KV head count")?,
            head_dim: to_u32(head_dim, "attention head dimension")?,
            key_count: to_u32(key_count, "attention key count")?,
            scale,
            window: to_u32(window, "attention window")?,
            splits: 0,
            keys_per_split: 0,
            pass_begin: 0,
            pass_keys: 0,
        };
        // Prefill: one threadgroup per (query, KV HEAD), holding one simdgroup
        // per query head that shares that KV head. `groups` query heads read the
        // same K and V out of one threadgroup's L1, so the cache is re-read
        // `q_heads / groups` = 1 times instead of `q_heads` times -- on Gxma,
        // 2x fewer kv reads on the 50 sliding layers and 8x on the 10 full ones.
        // A `groups` with no specialisation falls back to the 256-thread kernel,
        // which is dimension-partitioned and makes no GQA assumption.
        let groups = q_heads / kv_heads;
        // The generic kernel indexes a flat `queries * q_heads` thread range, so
        // it needs its own group count; both specialised kernels index the grid
        // themselves and discard the surplus threads on an early return.
        // Query blocking: which flash kernel to use, and with how many queries
        // per threadgroup. Both kernels share the grid geometry
        // `ceil(queries / block) * kv_heads`, so the block travels with the
        // pipeline selection.
        let flash = match qk {
            AttentionQk::Flash if flash_supported(head_dim, groups) => {
                let pipeline = if block == 1 {
                    self.attention_flash_bq1_pipeline.as_ref()
                } else {
                    self.attention_flash_pipeline.as_ref()
                };
                pipeline.map(|p| (p, block))
            }
            _ => None,
        };
        let (pipeline, grid_groups, group_threads) = match gqa_index(head_dim, groups) {
            Some(index) => match flash {
                // Flash shares the specialised grid exactly: one threadgroup per
                // (query, KV head), 32 * groups threads. Only the body differs.
                // Query blocking: the kernel body owns `block` consecutive
                // queries per threadgroup, so the grid is
                // `ceil(queries / block) * kv_heads`. Keep the block in lockstep
                // with the `BQ` argument of the `KAPLUMBA_FA_FLASH` instantiation
                // in `attention.metal`.
                Some((pipeline, block)) => (
                    pipeline,
                    (queries_count.div_ceil(block) * kv_heads) as u64,
                    256,
                ),
                None => (
                    match qk {
                        // A `Flash` request on an unsupported shape, or one whose
                        // pipeline failed to build, degrades to the MMA arm.
                        AttentionQk::Scalar => &self.attention_gqa_pipelines[index],
                        AttentionQk::Mma | AttentionQk::Flash => {
                            &self.attention_gqa_mma_pipelines[index]
                        }
                    },
                    (queries_count * kv_heads) as u64,
                    (32 * groups) as u64,
                ),
            },
            None => match head_dim {
                256 => (
                    &self.attention_tg2_256_pipeline,
                    (queries_count * q_heads) as u64,
                    256,
                ),
                512 => (
                    &self.attention_tg2_512_pipeline,
                    (queries_count * q_heads) as u64,
                    256,
                ),
                _ => (
                    &self.online_attention_pipeline,
                    threads_for(queries_count * q_heads, 64),
                    64,
                ),
            },
        };
        encoder.dispatch_groups(
            pipeline,
            &[
                Some(queries),
                Some(keys),
                Some(values),
                Some(query_positions),
                Some(key_positions),
                Some(output),
            ],
            params_bytes(&params),
            grid_groups,
            group_threads,
        )
    }

    /// Production prefill entry point: the ablation gate, then the QK
    /// implementation the environment selected (see [`attention_qk`]).
    ///
    /// Decode does not come through here -- it dispatches through
    /// `encode_online_attention_split` -- so `KAPLUMBA_ABLATE=attention`
    /// remains a prefill-only switch, exactly what
    /// `probe_budget_reconcile::the_attention_ablation_is_a_no_op_on_decode`
    /// asserts.
    pub fn encode_online_attention(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        request: AttentionEncodeRequest<'_>,
    ) -> Result<(), KernelError> {
        if crate::ablated("attention") {
            return Ok(());
        }
        let groups = if request.kv_heads == 0 {
            0
        } else {
            request.q_heads / request.kv_heads
        };
        let head_dim = request.head_dim;
        self.encode_online_attention_qk(encoder, request, attention_qk_for(groups, head_dim))
    }

    pub fn working_set_bytes(&self, request: &AttentionRequest<'_>) -> usize {
        request.q_heads * request.head_dim * size_of::<f32>()
    }

    /// Whether the prefill key-range-pass kernels (OQ9) compiled.
    ///
    /// Separate from [`Self::flash_attention_available`] because a missing pass
    /// kernel must NOT disable the one-dispatch flash kernel: it only means the
    /// prefill dispatch runs as one key range, which is what it does today.
    pub fn prefill_pass_available(&self) -> bool {
        self.attention_flash_pass_pipeline.is_some() && self.attention_flash_pass_bq1_pipeline.is_some()
    }

    /// Prefill attention as `passes` key-range dispatches (OQ9), each handing its
    /// fp32 `(m, s, O)` row state to the next through `state`.
    ///
    /// Bit-identical to [`Self::encode_online_attention_qk_block`]: the per-tile
    /// arithmetic and its order are unchanged, and the row state is fp32 in, fp32
    /// out. What differs is only that the key axis is cut into `passes` pieces,
    /// so the threadgroups in flight at any instant share a narrower slice of the
    /// KV head than they would over the whole window.
    ///
    /// `state` must hold `queries * q_heads * (head_dim + 2)` fp32. Callers get
    /// the size from [`prefill_pass_state_elements`]; a short buffer is an error,
    /// never a silent truncation, because every row of every pass writes its whole
    /// row unconditionally.
    ///
    /// Returns `false` without touching the encoder when this shape is not
    /// eligible, so the caller falls back to the one-dispatch dispatch and nothing
    /// is half-encoded. Eligibility is **exactly** [`prefill_pass_count`] >= 2 and
    /// this function calls it (F2): before that, the grid gate was documented on
    /// this function and applied nowhere, so `prefill_pass_count` had no production
    /// caller and its tests tested a dead function.
    pub fn encode_online_attention_passes(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        request: AttentionEncodeRequest<'_>,
        block: usize,
        state: &Buffer,
    ) -> Result<bool, KernelError> {
        let pass_keys = prefill_pass_keys_for(request.key_count);
        // F2 (adversary_report.md): the grid gate was documented here and was not
        // applied here. This function forwarded straight to `_explicit`, whose own
        // comment claimed "the production caller applies [the gate] through
        // encode_online_attention_passes" -- it did not, so `prefill_pass_count`
        // had no production caller and the production path sliced grids as small as
        // 20 threadgroups (the last chunk of a 16 394-token prompt: 10 queries x 4
        // kv heads at BQ=2). The gate tests in
        // `attention_prefill_pass_bit_exact.rs` exercised a function nothing called.
        //
        // The gate is applied HERE, where the claim lives, and it is applied by
        // calling `prefill_pass_count` rather than by re-deriving the condition, so
        // the tested function and the production function cannot drift apart again.
        let passes = prefill_pass_count(
            request.queries_count,
            request.kv_heads,
            request.key_count,
            request.head_dim,
            if request.q_heads == 0 {
                0
            } else {
                request.q_heads / request.kv_heads.max(1)
            },
            block,
            self.prefill_pass_available(),
        );
        if passes < 2 {
            return Ok(false);
        }
        Self::encode_online_attention_passes_explicit(
            self,
            encoder,
            request,
            block,
            state,
            pass_keys,
        )
    }

    /// [`Self::encode_online_attention_passes`] with the keys-per-pass given
    /// explicitly, so a test can sweep pass sizes inside one process without
    /// fighting the `KAPLUMBA_PREFILL_PASS_KEYS` `OnceLock` cache, and so a test
    /// can bypass the grid gate to exercise shapes production never takes.
    ///
    /// `pass_keys` is honoured as given, and the grid-size gate
    /// (`PREFILL_PASS_MIN_GROUPS`) is **deliberately not** applied here. This is a
    /// test tool; the production entry point [`Self::encode_online_attention_passes`]
    /// applies the gate itself, by calling [`prefill_pass_count`], so a caller that
    /// wants a chain regardless of grid size is asking for something production
    /// never asks for -- which is exactly what a correctness test wants.
    pub fn encode_online_attention_passes_explicit(
        &self,
        encoder: &mut crate::encoder::MetalEncoder,
        request: AttentionEncodeRequest<'_>,
        block: usize,
        state: &Buffer,
        pass_keys: usize,
    ) -> Result<bool, KernelError> {
        let AttentionEncodeRequest {
            queries,
            keys,
            values,
            query_positions,
            key_positions,
            output,
            queries_count,
            q_heads,
            kv_heads,
            head_dim,
            key_count,
            scale,
            window,
        } = request;
        if queries_count == 0
            || kv_heads == 0
            || q_heads == 0
            || q_heads % kv_heads != 0
        {
            return Err(KernelError::InvalidInput(
                "attention dimensions must be non-zero and q_heads divisible by kv_heads".to_string(),
            ));
        }
        let groups = q_heads / kv_heads;
        // Eligibility: the shape must have a pass kernel, and the key range must
        // actually need more than one pass. The grid-size gate
        // (`PREFILL_PASS_MIN_GROUPS`) is NOT applied here -- see the doc comment on
        // this function. `encode_online_attention_passes` (the production entry
        // point) applies it by calling `prefill_pass_count` before forwarding here
        // (F2).
        if !self.prefill_pass_available() || !flash_supported(head_dim, groups) {
            return Ok(false);
        }
        if pass_keys == 0 || key_count <= pass_keys {
            return Ok(false);
        }
        let edges = prefill_pass_edges(key_count, pass_keys);
        let passes = edges.len().saturating_sub(1);
        if passes < 2 {
            return Ok(false);
        }
        let pipeline = if block == 1 {
            self.attention_flash_pass_bq1_pipeline
                .as_ref()
                .expect("checked by prefill_pass_available")
        } else {
            self.attention_flash_pass_pipeline
                .as_ref()
                .expect("checked by prefill_pass_available")
        };
        let required = prefill_pass_state_elements(queries_count, q_heads, head_dim);
        if (state.length() as usize) < required * size_of::<f32>() {
            return Err(KernelError::InvalidInput(format!(
                "prefill pass state buffer is {} bytes, needs {required} f32 for {queries_count} queries x {q_heads} heads x {head_dim} dims",
                state.length()
            )));
        }
        let grid = (queries_count.div_ceil(block) * kv_heads) as u64;
        for index in 0..passes {
            let pass_begin = edges[index];
            let pass_end = edges[index + 1];
            let params = AttentionParams {
                queries: to_u32(queries_count, "attention query count")?,
                q_heads: to_u32(q_heads, "attention query head count")?,
                kv_heads: to_u32(kv_heads, "attention KV head count")?,
                head_dim: to_u32(head_dim, "attention head dimension")?,
                key_count: to_u32(key_count, "attention key count")?,
                scale,
                window: to_u32(window, "attention window")?,
                splits: 0,
                keys_per_split: 0,
                pass_begin: to_u32(pass_begin, "prefill pass begin")?,
                pass_keys: to_u32(
                    pass_end.saturating_sub(pass_begin),
                    "prefill pass key count",
                )?,
            };
            // `dispatch_groups` appends the params blob at `bindings.len()`, so the
            // extra buffer has to be inserted *before* it: `output` stays at 5
            // like every other attention kernel, `state` takes 6, and `params`
            // lands at 7. Two ways to get this wrong, both silent -- binding
            // `state` to the params bytes, or leaving the kernel's real output
            // unbound so it inherits whatever the previous dispatch left at that
            // index. Both produce a plausible, stable, wrong number rather than
            // an error, and the second one is what a plausible-looking edit gets.
            encoder.dispatch_groups(
                pipeline,
                &[
                    Some(queries),
                    Some(keys),
                    Some(values),
                    Some(query_positions),
                    Some(key_positions),
                    Some(output),
                    Some(state),
                ],
                params_bytes(&params),
                grid,
                256,
            )?;
        }
        Ok(true)
    }

    fn make_u16_buffer(&self, values: &[u16]) -> Result<Buffer, KernelError> {
        if values.is_empty() {
            return Err(KernelError::InvalidInput(
                "attention u16 input cannot be empty".to_string(),
            ));
        }
        Ok(self.device.new_buffer_with_data(
            values.as_ptr() as *const c_void,
            size_of_val(values) as u64,
            MTLResourceOptions::StorageModeShared,
        ))
    }

    fn make_u32_buffer(&self, values: &[u32]) -> Result<Buffer, KernelError> {
        if values.is_empty() {
            return Err(KernelError::InvalidInput(
                "attention position input cannot be empty".to_string(),
            ));
        }
        Ok(self.device.new_buffer_with_data(
            values.as_ptr() as *const c_void,
            size_of_val(values) as u64,
            MTLResourceOptions::StorageModeShared,
        ))
    }
}

fn params_bytes<T>(params: &T) -> &[u8] {
    let size = std::mem::size_of_val(params);
    unsafe { std::slice::from_raw_parts(params as *const T as *const u8, size) }
}

fn validate_request(request: &AttentionRequest<'_>) -> Result<(), KvAttentionError> {
    if request.query_positions.is_empty()
        || request.key_positions.is_empty()
        || request.q_heads == 0
        || request.kv_heads == 0
        || request.head_dim == 0
        || request.head_dim > MAX_HEAD_DIM
        || !request.q_heads.is_multiple_of(request.kv_heads)
    {
        return Err(KvAttentionError::InvalidDimensions);
    }
    if !request.scale.is_finite() {
        return Err(KvAttentionError::InvalidScale);
    }
    let query_elements = checked_product(
        request.query_positions.len(),
        checked_product(
            request.q_heads,
            request.head_dim,
            KvAttentionError::ShapeOverflow,
        )?,
        KvAttentionError::ShapeOverflow,
    )?;
    let key_elements = checked_product(
        request.key_positions.len(),
        checked_product(
            request.kv_heads,
            request.head_dim,
            KvAttentionError::ShapeOverflow,
        )?,
        KvAttentionError::ShapeOverflow,
    )?;
    if request.queries.len() != query_elements
        || request.keys.len() != key_elements
        || request.values.len() != key_elements
    {
        return Err(KvAttentionError::InvalidBuffer);
    }
    if request
        .query_positions
        .windows(2)
        .any(|positions| positions[0] >= positions[1])
    {
        return Err(KvAttentionError::InvalidQueryPositions);
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvAttentionError {
    InvalidDimensions,
    InvalidScale,
    ShapeOverflow,
    InvalidBuffer,
    InvalidQueryPositions,
}

impl From<KvAttentionError> for KernelError {
    fn from(error: KvAttentionError) -> Self {
        Self::InvalidInput(error.to_string())
    }
}

impl std::fmt::Display for KvAttentionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::InvalidDimensions => "attention dimensions are invalid",
            Self::InvalidScale => "attention scale is not finite",
            Self::ShapeOverflow => "attention shape overflows",
            Self::InvalidBuffer => "attention buffer length does not match shape",
            Self::InvalidQueryPositions => "query positions must be strictly increasing",
        };
        formatter.write_str(message)
    }
}

fn checked_product(
    left: usize,
    right: usize,
    error: KvAttentionError,
) -> Result<usize, KvAttentionError> {
    left.checked_mul(right).ok_or(error)
}

fn to_u32(value: usize, name: &str) -> Result<u32, KernelError> {
    u32::try_from(value).map_err(|_| KernelError::InvalidInput(format!("{name} is too large")))
}

fn read_u16_buffer(buffer: &Buffer) -> Result<Vec<u16>, KernelError> {
    let length = buffer.length() as usize;
    if !length.is_multiple_of(size_of::<u16>()) {
        return Err(KernelError::CommandFailed);
    }
    let pointer = buffer.contents() as *const u16;
    if pointer.is_null() {
        return Err(KernelError::CommandFailed);
    }
    Ok(unsafe { std::slice::from_raw_parts(pointer, length / size_of::<u16>()) }.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Gxma's two layer shapes, from `config.json`: 50 sliding layers at
    /// `32 q / 16 kv / head_dim 256 / window 1024` and 10 full-attention layers
    /// at `32 q / 4 kv / head_dim 512`. `config.json` is the only source for
    /// these numbers, so they are written out rather than derived.
    const SLIDING_Q: usize = 32;
    const SLIDING_KV: usize = 16;
    const FULL_Q: usize = 32;
    const FULL_KV: usize = 4;

    /// The split count the **one-wave workgroup** rule picks, bypassing the
    /// environment so a test cannot be perturbed by a `KAPLUMBA_DECODE_SPLITS` or
    /// a `KAPLUMBA_DECODE_SPLIT_RULE` in the ambient shell.
    fn chosen(q_heads: usize, kv_heads: usize, key_count: usize) -> usize {
        decode_splits_with(
            q_heads,
            kv_heads,
            key_count,
            None,
            DecodeSplitRule::WorkgroupsPerCore,
        )
    }

    /// The split count the **default** (thread-target) rule picks. Every test
    /// above that quotes a split count is about the workgroup rule, so the
    /// default has to be pinned separately -- see
    /// `the_default_is_the_thread_target_and_the_one_wave_rule_is_opt_in`.
    fn default_chosen(q_heads: usize, kv_heads: usize, key_count: usize) -> usize {
        decode_splits_with(
            q_heads,
            kv_heads,
            key_count,
            None,
            DecodeSplitRule::ThreadTarget,
        )
    }

    /// The rule that shipped before the workgroups-per-core refactor, kept
    /// here so "unchanged for the sliding shape" is a check against the old
    /// arithmetic and not against a number someone typed twice.
    fn legacy_thread_target(q_heads: usize, key_count: usize) -> usize {
        let wanted = ATTENTION_DECODE_THREADS_TARGET / (32 * q_heads.max(1));
        let affordable = key_count.div_ceil(ATTENTION_MIN_KEYS_PER_SPLIT);
        wanted.clamp(1, ATTENTION_SPLITS_MAX).min(affordable).max(1)
    }

    /// **The default must not move.** The one-wave rule is 1.34x faster on the
    /// full-attention layers and it is *not* the default, because it changes the
    /// token stream: the combine pass sums the per-split partials in a different
    /// order and floating-point addition is not associative. So the shipped
    /// default has to evaluate to the old arithmetic for every shape, and this
    /// is where that is asserted rather than intended.
    ///
    /// The reference is `legacy_thread_target`, i.e. the pre-change code's
    /// expression, not a transcribed table.
    #[test]
    fn the_default_is_the_thread_target_and_the_one_wave_rule_is_opt_in() {
        for (q, kv) in [
            (SLIDING_Q, SLIDING_KV),
            (FULL_Q, FULL_KV),
            (40, 5),
            (48, 6),
            (32, 32),
        ] {
            for key_count in [1usize, 64, 65, 512, 1024, 2048, 4096, 8192] {
                assert_eq!(
                    default_chosen(q, kv, key_count),
                    legacy_thread_target(q, key_count),
                    "{q}q/{kv}kv @{key_count}: the DEFAULT rule changed, which changes \
                     the model's output and was not authorised",
                );
            }
        }
        // And the two rules genuinely differ where they are meant to, so this
        // test cannot pass by both being the same number.
        assert_eq!(default_chosen(FULL_Q, FULL_KV, 8192), 16);
        assert_eq!(chosen(FULL_Q, FULL_KV, 8192), 13);
    }

    /// `KAPLUMBA_DECODE_SPLIT_RULE` parsing. Total by construction, and an
    /// unrecognised value is the *default* rather than an error: this is read on
    /// the decode hot path, where a panic takes the process down and where a
    /// typo must not silently change the numerics.
    #[test]
    fn the_split_rule_selector_is_parsed_defensively() {
        // Case-sensitive, like `parse_single_pass_limit` beside it: an
        // environment variable's spelling is part of its interface, and a
        // case-insensitive match would be one more thing to keep in sync.
        for on in [
            "workgroups",
            "waves",
            "one_wave",
            "1",
            "on",
            "true",
            "yes",
            "auto",
            " workgroups ",
        ] {
            assert_eq!(
                parse_split_rule(on),
                DecodeSplitRule::WorkgroupsPerCore,
                "{on:?} should select the one-wave rule",
            );
        }
        for off in [
            "",
            "0",
            "off",
            "false",
            "no",
            "threads",
            "legacy",
            "abc",
            "2",
            "0x1",
            "workgroup",
            "-1",
            "1.5",
            "WORKGROUPS",
            "Workgroups",
        ] {
            assert_eq!(
                parse_split_rule(off),
                DecodeSplitRule::ThreadTarget,
                "{off:?} must fall back to the shipped default",
            );
        }
    }

    /// **The non-regression proof for 83% of the layers.** The sliding shape's
    /// one-wave budget is 288 workgroups, which for 16 kv heads is 18 splits --
    /// above the cap, so the cap decides and the old and new arithmetic must
    /// agree at every key count, not merely at the ones that were measured.
    #[test]
    fn sliding_layers_are_bit_identical_in_dispatch_to_the_legacy_rule() {
        for key_count in [
            1usize, 63, 64, 65, 127, 128, 129, 255, 256, 511, 512, 1023, 1024, 2048, 4096, 8192,
            65536,
        ] {
            assert_eq!(
                chosen(SLIDING_Q, SLIDING_KV, key_count),
                legacy_thread_target(SLIDING_Q, key_count),
                "sliding @{key_count} keys: the rule changed the split count, and \
                 50 of 60 layers depend on it not doing so",
            );
        }
    }

    /// The sliding layers' split count at the key counts a report quotes, plus
    /// the workgroup count those splits produce. At a full 1024-key window,
    /// 16 splits x 16 kv heads = 256 workgroups of 64 threads, which is inside
    /// the 288-workgroup one-wave budget. At 512 keys the *keys-per-split
    /// floor* decides (512/64 = 8), not the wave budget, so only 128
    /// workgroups are dispatched.
    #[test]
    fn sliding_layers_dispatch_one_wave_at_every_context_length() {
        for (key_count, splits) in [(512usize, 8usize), (1024, 16)] {
            assert_eq!(
                chosen(SLIDING_Q, SLIDING_KV, key_count),
                splits,
                "sliding @{key_count}"
            );
            let workgroups = splits * SLIDING_KV;
            assert!(
                workgroups <= decode_one_wave_workgroups(64).expect("64-thread budget"),
                "sliding @{key_count}: {workgroups} workgroups exceeds one wave",
            );
        }
    }

    /// The full-attention layers: `kv_heads 4`, so 16 splits is 64 workgroups of
    /// 256 threads where only 54 fit in one wave -- a second wave for 10 of
    /// them, which measured 1.17-1.34x *slower* than 13 splits for the identical
    /// byte count. The budget puts the grid at 52.
    #[test]
    fn full_layers_fit_one_wave_where_sixteen_splits_did_not() {
        let budget = decode_one_wave_workgroups(256).expect("256-thread budget");
        assert_eq!(budget, 54, "the one-wave budget moved; re-run the sweep");
        for key_count in [1024usize, 2048, 4096, 8192] {
            let splits = chosen(FULL_Q, FULL_KV, key_count);
            let workgroups = splits * FULL_KV;
            assert!(
                workgroups <= budget,
                "full @{key_count}: {workgroups} workgroups is more than one wave",
            );
            assert_ne!(
                splits, 16,
                "full @{key_count}: 16 splits is the whole point of the change",
            );
            assert!(
                workgroups + FULL_KV > budget,
                "full @{key_count}: {splits} splits leaves the wave underfull; \
                 the budget rounds down and this says it rounded down too far",
            );
        }
    }

    /// **The rule is a rule, not a fit to one number.** Three shapes with the
    /// same 256-thread threadgroup and different `kv_heads` were swept
    /// independently, and `budget / kv_heads` has to land on each one's measured
    /// optimum without being told what it was:
    ///
    /// | shape | `kv_heads` | measured best | grid | rule says |
    /// |---|---|---|---|---|
    /// | 32q/4kv/d512 | 4 | 13 splits | 52 | `54 / 4 = 13` |
    /// | 40q/5kv/d512 | 5 | 10 splits | 50 | `54 / 5 = 10` |
    /// | 48q/6kv/d512 | 6 | 9 splits | 54 | `54 / 6 = 9` |
    ///
    /// A budget tuned to make the first row work (say 60, from `20 cores x 3`)
    /// would say 15, 12 and 10 for these, and 15 and 12 are both *measured worse
    /// than the 16 that ships today*. This is the test that would have caught
    /// that.
    #[test]
    fn the_budget_reproduces_the_measured_optimum_of_every_shape_it_was_measured_on() {
        for (q_heads, kv_heads, measured_best_splits) in
            [(32usize, 4usize, 13usize), (40, 5, 10), (48, 6, 9)]
        {
            assert_eq!(
                32 * (q_heads / kv_heads),
                256,
                "{q_heads}q/{kv_heads}kv is not a 256-thread shape",
            );
            assert_eq!(
                chosen(q_heads, kv_heads, 4096),
                measured_best_splits,
                "{q_heads}q/{kv_heads}kv: the rule does not reproduce the measured \
                 optimum, so it is a fit to the 4-kv-head number and nothing else",
            );
        }
    }

    /// The full-attention layers at short context are still bounded by the
    /// keys-per-split floor, not by the wave budget: at 512 keys the shape can
    /// only afford 8 splits however many cores there are.
    #[test]
    fn the_keys_per_split_floor_still_bounds_short_context() {
        for key_count in [1usize, 63, 64, 65, 128, 256, 511, 512] {
            let affordable = key_count.div_ceil(ATTENTION_MIN_KEYS_PER_SPLIT);
            let splits = chosen(FULL_Q, FULL_KV, key_count);
            assert!(
                splits <= affordable.max(1),
                "full @{key_count}: {splits} splits exceeds the {affordable} the \
                 key count can afford",
            );
        }
        assert_eq!(chosen(FULL_Q, FULL_KV, 512), 8);
    }

    /// The full table the report quotes: split count, workgroups, workgroups
    /// per core, for both shapes at every key count. If a constant here moves,
    /// this fails and the note has to be re-measured rather than re-typed.
    #[test]
    fn the_reported_split_table() {
        let expected = [
            // (shape, keys, splits, workgroups)
            ("sliding", 512, 8, 128),
            ("sliding", 1024, 16, 256),
            ("full", 512, 8, 32),
            ("full", 1024, 13, 52),
            ("full", 2048, 13, 52),
            ("full", 4096, 13, 52),
            ("full", 8192, 13, 52),
        ];
        for (name, key_count, splits, workgroups) in expected {
            let (q, kv) = if name == "sliding" {
                (SLIDING_Q, SLIDING_KV)
            } else {
                (FULL_Q, FULL_KV)
            };
            assert_eq!(
                chosen(q, kv, key_count),
                splits,
                "{name} @{key_count}: split count moved",
            );
            assert_eq!(splits * kv, workgroups, "{name} @{key_count}: workgroups");
        }
    }

    /// Everything this returns has to fit the fixed-size partials buffer, for
    /// *any* shape a future config could present -- not just Gxma's two.
    #[test]
    fn the_result_is_always_inside_the_fixed_scratch() {
        for q_heads in [1usize, 2, 4, 8, 16, 32, 40, 64, 128] {
            for kv_heads in [1usize, 2, 4, 8, 16, 32] {
                if q_heads % kv_heads != 0 {
                    continue;
                }
                for key_count in [0usize, 1, 63, 64, 65, 512, 1024, 8192, 1 << 20] {
                    let splits = chosen(q_heads, kv_heads, key_count);
                    assert!(
                        (1..=ATTENTION_SPLITS_MAX).contains(&splits),
                        "{q_heads}q/{kv_heads}kv @{key_count}: {splits} splits is \
                         outside 1..={ATTENTION_SPLITS_MAX}",
                    );
                    // The buffer `decode_partial_elements` sizes, and the guard
                    // in `encode_decode_split_with_splits` enforces.
                    let floats = splits * q_heads * (256 + 2);
                    assert!(
                        floats <= decode_partial_elements(q_heads, 256),
                        "{q_heads}q/{kv_heads}kv @{key_count}: {floats} floats \
                         exceed the scratch",
                    );
                }
            }
        }
    }

    /// Degenerate inputs must not panic or divide by zero. `kv_heads == 0` is
    /// unreachable from the encode path (it rejects the request first) but the
    /// arithmetic is on the hot path and a panic there is a process death. With
    /// `kv_heads == 0` the result is whatever the legacy thread target says --
    /// finite, in range, and irrelevant because such a request never gets this
    /// far.
    #[test]
    fn degenerate_head_counts_do_not_panic() {
        assert_eq!(chosen(0, 0, 8192), 1);
        assert_eq!(chosen(0, 4, 8192), 1);
        assert_eq!(chosen(32, 4, 0), 1);
        assert_eq!(chosen(32, 0, 8192), 16);
        assert!((1..=ATTENTION_SPLITS_MAX).contains(&chosen(32, 0, 8192)));
    }

    /// A threadgroup width with no measured budget keeps the old arithmetic
    /// rather than guessing one, **even with the one-wave rule selected**.
    /// `kv_heads == q_heads` gives a 32-thread threadgroup, which is not in the
    /// table. Note that 64q/8kv is *also* a 256-thread threadgroup and therefore
    /// measured, which is why it is absent.
    #[test]
    fn an_unmeasured_threadgroup_width_keeps_the_legacy_thread_target() {
        assert_eq!(decode_one_wave_workgroups(32), None);
        assert_eq!(decode_one_wave_workgroups(128), None);
        assert_eq!(decode_one_wave_workgroups(512), None);
        for (q, kv) in [(32usize, 32usize), (8, 1), (4, 1), (16, 2), (64, 64)] {
            for key_count in [64usize, 512, 1024, 8192] {
                assert_eq!(
                    chosen(q, kv, key_count),
                    legacy_thread_target(q, key_count),
                    "{q}q/{kv}kv @{key_count}: unmeasured shape changed under the \
                     one-wave rule",
                );
            }
        }
    }

    /// The pinned sweep, which is how the constants were measured. Parsing is
    /// tested as a function rather than through the environment, which Rust 2024
    /// makes `unsafe` and which every other test in this process shares.
    #[test]
    fn the_pinned_split_value_is_parsed_defensively() {
        // `+4` and `4\n` are accepted because `usize::from_str` accepts a
        // leading `+` and the value is trimmed. Both are in range, so pinning
        // them is harmless; what must never happen is a value that silently
        // becomes the *heuristic*'s number by accident, which the range check
        // rules out.
        for good in ["1", "16", " 8 ", "4\n", "+4"] {
            assert_eq!(
                parse_pinned_splits(good),
                good.trim().parse::<usize>().ok(),
                "{good:?}",
            );
        }
        for bad in [
            "",
            "0",
            "17",
            "-1",
            "abc",
            "1.5",
            "99999999999999999999",
            "0x4",
            "4 4",
        ] {
            assert_eq!(parse_pinned_splits(bad), None, "{bad:?} must not pin");
        }
        // And a pinned value wins over the heuristic, in both directions, so
        // the sweep can compare 16 (worse) as well as the new value.
        assert_eq!(
            decode_splits_with(
                FULL_Q,
                FULL_KV,
                8192,
                Some(16),
                DecodeSplitRule::ThreadTarget
            ),
            16
        );
        assert_eq!(
            decode_splits_with(
                FULL_Q,
                FULL_KV,
                8192,
                Some(4),
                DecodeSplitRule::ThreadTarget
            ),
            4
        );
        assert_eq!(
            decode_splits_with(FULL_Q, FULL_KV, 64, Some(16), DecodeSplitRule::ThreadTarget),
            16
        );
    }
}
