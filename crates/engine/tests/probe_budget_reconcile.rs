//! A5 — bottom-up decode-step budget. **No GPU, no model, no binary that opens a
//! checkpoint.** Pure arithmetic over the shapes the checkpoint fixes, plus the
//! measurement anchors already recorded in this tree. Every term carries its
//! `file:line`.
//!
//! # The step being budgeted
//!
//! **62.3 ms/step = 16.05 tok/s**, `--temperature 0`, 37-token prompt, 32
//! generated tokens (`AGENTS.md` §10.2.7, `spec/decode-ceiling/hypotheses.md`).
//! Weights read per token: **12,039,487,488 B** (`bench.rs:331`).
//!
//! # Which sources this file treats as CORRUPT, and why
//!
//! | source | why it is not used |
//! |---|---|
//! | `decode-25tok-peak-bandwidth` §18.2 per-projection 126–150 GB/s | derived from `KAPLUMBA_ABLATE` marginal costs whose deltas sum to 87.59 ms inside a 63.1 ms step. Not additive, so not bandwidth. **Not used.** |
//! | "7.32 ms residual with every kernel-side role ablated" | `ablated("attention")` appears once, at `attention.rs:910`, inside `encode_online_attention`. Decode calls `encode_online_attention_split` (`attention.rs:654`) or `encode_decode_single_pass` (`model.rs:1889`), neither of which consults it — **verified by `the_attention_ablation_is_a_no_op_on_decode` below.** So the residual still contains 60 live attention kernels. **Not used.** |
//! | 546 dispatches | that is `GpuProfile::stamp` labels, which exist only under `KAPLUMBA_GPU_PROFILE=1`. **Not used.** |
//! | the 1.71× profiler calibration | self-refuting (derived from the head row). **Not used.** |
//! | 11.834 GB/token | charges sliding geometry to all 60 layers; 10 are full-attention (head_dim 512, 4 KV heads). **Not used** — `bench::per_token_weight_bytes` is used instead. |
//! | P07's "attention == weight streaming at P ≈ 13 100" | **arithmetically wrong, and this file catches it.** It scales all 50 sliding layers with context, which the ring does not do (`min(P, 1024)`), *and* uses the 11.834 GB weight figure. The correct crossover is **P = 122 800**, outside the 32 768 context cap. See [`attention_never_reaches_the_weight_stream_within_the_context_cap`]. |
//! | P15's tok/s-vs-context table (15.50 / 14.30 / 13.63) | **no provenance.** P15 is still `PROPOSED` in `hypotheses.md:74` and `test-log.md:24` says "(nothing tested yet)". The three rates have no measurement behind them and appear to be linear-model output. **Not used.** |
//! | `probe_dram_roofline.rs`'s `code_rate_vs_bandwidth_discriminator` | its ALU body is a serial dependency chain, so it converts instruction count to latency. **Not used.** |
//!
//! # Running it
//!
//! `cargo test --test probe_budget_reconcile` (no flags, no GPU).

use kaplumba_engine::bench::per_token_weight_bytes;

const GB: f64 = 1e9;
const MB: f64 = 1e6;

// ===========================================================================
// 1. Measurements. Each is a number already in this tree, with its source.
// ===========================================================================

/// `spec/decode-ceiling/hypotheses.md:15`, `ROUND1-2-SUMMARY.md:148`.
const MEASURED_STEP_MS: f64 = 62.30;
const MEASURED_TOK_S: f64 = 16.05;

/// The key count the 62.30 ms measurement was taken at: a 6.01-token prompt
/// (`AGENTS.md` §10.2.7) plus up to 32 generated, so the KV cache holds 37-69
/// keys. **Every budget is anchored here.** 512 and 16 384 are predictions.
const MEASURED_KEYS: usize = 37;

/// `decode-25tok-peak-bandwidth` §16.3, `gemv_bandwidth.rs`: best 3 bpw EXL3
/// GEMV ever measured on this box.
const PEAK_3BPW_GB_S: f64 = 203.0;

/// §18.2 / `perf_probe_criticalpath_latency.rs:104`: `head/gemv` on its own.
/// Two independent instruments agree to 1.2%: the GPU clock (§16.2,
/// 1.057 GB / 4.12 ms = 257) and a production wall-clock ablation (§18.1,
/// `KAPLUMBA_ABLATE=head_gemv` = +1.12 tok/s = 4.17 ms). 253 is the lower.
const HEAD_GB_S: f64 = 253.0;

/// `crates/engine/tests/perf_probe_criticalpath_latency.rs:222-232`:
/// two-point fit `t = bytes/BW + c` on `sliding/gate` and `full/kv`, checked to
/// fit all ten sweep shapes to 2.2% (`max_time_residual`, same file:255).
const FIT_BW_GB_S: f64 = 190.5;
const FIT_US_PER_DISPATCH: f64 = 9.03;

/// `perf_probe_criticalpath_latency.rs:634`: 4 GEMV dispatches/layer x 60 + the
/// head. Corroborated by the static walk in `findings/P02-dispatch-overhead.md`
/// (`gemv_fused3`, `gemv_raw_sk2` x2, `gemv_fused2_mlp`).
const GEMV_DISPATCHES: f64 = 241.0;

/// P02: `dispatches_encoded()`, 741 layer-side + 9 epilogue.
const ALL_DISPATCHES: f64 = 750.0;

/// P02: 121 layer-side single-threadgroup dispatches, 14.89 MB/token.
const SINGLE_TG_DISPATCHES: f64 = 121.0;
#[allow(dead_code)]
const SINGLE_TG_MB: f64 = 14.89;

/// `findings/P07-attention-elements.md:32` — production interleaved A/B
/// (T5.10), the instrument `ROUND1-2-SUMMARY.md:52` calls the reliable one.
/// This is the *element* layer only: `KAPLUMBA_ABLATE=attention` is a no-op on
/// decode, so attention is NOT inside this number.
const ELEMENT_LAYER_MS: f64 = 0.84;

/// P06: total host CPU per greedy token. Overlapped — decode commits step N+1
/// before waiting on N (`generate.rs:997-1015`) — so it contributes ~0 to the
/// step and is NOT a budget row. Carried because its absence from the budget is
/// a claim, and a claim needs a number behind it.
#[allow(dead_code)]
const HOST_CPU_MS: f64 = 0.078;
/// P06: `tokenizer.rs:202-235` replays the whole id list per push; 0.017 ms at
/// short responses, 0.105 ms at 119 tokens, and it runs *between* `wait()` and
/// the next submit, so it is pure GPU idle even in the pipelined loop.
const DETOK_MS_SHORT: f64 = 0.017;
const DETOK_MS_119: f64 = 0.105;

/// `decode-19tok-repacked-250gb/implementation_notes.md` T7.6.13 table.
/// GPU-clocked `own_ms` of the **shipped** attention path, median of 9
/// interleaved samples, one pass per command buffer; per-sample spread
/// 0.0002-0.31 ms. Rows are `(keys, splits, sliding_ms, full_ms)`; `None`
/// sliding means the ring has capped at 1024 so the number is flat thereafter.
type AttnRow = (usize, usize, Option<f64>, Option<f64>);
const ATTN_ROWS: &[AttnRow] = &[
    (16, 1, Some(0.0125), Some(0.0205)),
    (64, 1, Some(0.0290), Some(0.0531)),
    (65, 2, Some(0.0206), Some(0.0360)),
    (128, 2, Some(0.0294), Some(0.0537)),
    (256, 4, Some(0.0305), Some(0.0544)),
    (512, 8, Some(0.0384), Some(0.0601)),
    (1024, 16, Some(0.0592), Some(0.1188)),
    (8192, 16, None, Some(1.1568)),
];

// --- temperature > 0 -------------------------------------------------------
// `perf_probe_concurrency_dependency_graph.rs:516-532`, which is the only file
// that carries these with their provenance attached.

/// **Measured**: chat pipelined-vs-barrier A/B, temperature 0, interleaved:
/// 15.88 tok/s pipelined against 14.97 with the barrier.
const CHAT_PIPELINED_TOK_S: f64 = 15.88;
const CHAT_BARRIER_TOK_S: f64 = 14.97;
/// **Measured**: `sampler.rs:41-52` via `examples/bench_sampler.rs`.
const SAMPLE_MS: f64 = 2.87;
/// **Bracketed, never measured** — no value was ever recorded in the tree.
const WIDEN_MS_LOW: f64 = 0.30;
const WIDEN_MS_HIGH: f64 = 0.60;
const READBACK_MS_LOW: f64 = 0.10;
const READBACK_MS_HIGH: f64 = 0.20;

// ===========================================================================
// 2. Bytes. Two of the three partitions are pure functions of the checkpoint.
// ===========================================================================

const SLIDING_LAYERS: usize = 50;
const FULL_LAYERS: usize = 10;
const SLIDING_WINDOW: usize = 1024;
/// `gpu_cache.rs:30-31` stores `keys: Vec<u16>`.
const KV_MAX: u64 = 4096; // 16 kv heads x 256
const FULL_KV: u64 = 2048; // 4 kv heads x 512

/// KV bytes read per decode token: `50*min(P,1024)*16384 + 10*P*8192`.
/// Identical to `perf_probe_activations_traffic.rs:702-708`.
/// KV bytes attributable to the 10 **full-attention** layers between two key
/// counts. The sliding layers are excluded so a marginal rate can be attributed
/// to one layer kind.
fn kv_8192_growth(from: usize, to: usize) -> f64 {
    (to as f64 - from as f64) * FULL_KV as f64 * 2.0 * 2.0 * FULL_LAYERS as f64
}

fn kv_read_per_token(key_count: usize) -> u64 {
    let sliding_keys = key_count.min(SLIDING_WINDOW) as u64;
    sliding_keys * KV_MAX * 2 * 2 * SLIDING_LAYERS as u64
        + key_count as u64 * FULL_KV * 2 * 2 * FULL_LAYERS as u64
}

/// `bench.rs:288`, asserted at `bench.rs:331`.
fn head_bytes() -> u64 {
    1_056_964_608
}

/// Head time, measured twice (§16.2 clock, §18.1 ablation). Not a residual.
fn head_ms() -> f64 {
    head_bytes() as f64 / (HEAD_GB_S * GB) * 1e3
}

// --- activation scratch, ported from `perf_probe_activations_traffic.rs` ----

const HIDDEN: usize = 5376;
const INTER: usize = 21504;
const Q_HEADS: usize = 32;
const SLIDING_KV_HEADS: usize = 16;
const FULL_KV_HEADS: usize = 4;
const SLIDING_HEAD_DIM: usize = 256;
const FULL_HEAD_DIM: usize = 512;
const SPLITS_MAX: usize = 16;
const SINGLE_PASS_MAX_KEYS: usize = 64;

const fn h(n: usize) -> u64 {
    (n * 2) as u64
}
const fn f(n: usize) -> u64 {
    (n * 4) as u64
}

/// `attention.rs:21`.
fn splits_at(key_count: usize, kv_heads: usize) -> usize {
    let affordable = key_count.div_ceil(64);
    let wanted = (16384 / (32 * Q_HEADS)).clamp(1, SPLITS_MAX);
    let _ = kv_heads;
    wanted.min(affordable).max(1)
}

fn partials(head_dim: usize, splits: usize) -> u64 {
    f(splits * Q_HEADS * (head_dim + 2))
}

/// Per-layer dispatch traffic, split by **which kernel writes/reads it**.
///
/// `attention` rows belong to the attention kernels — which the ablation cannot
/// reach. `element` rows are the norm / transform / finalize / GEMV-epilogue
/// family, which is exactly the set `KAPLUMBA_ABLATE` can switch off and what
/// the 0.84 ms A/B measured. Ported verbatim from
/// `perf_probe_activations_traffic.rs:376-481`, with the `kv_cache` write kept
/// out of both (it is KV traffic, priced with the KV read).
#[derive(Default, Clone, Copy, Debug)]
struct Split {
    element: u64,
    attention: u64,
}

impl Split {
    fn all(&self) -> u64 {
        self.element + self.attention
    }
}

#[derive(Clone, Copy)]
struct Layer {
    q_dim: usize,
    kv_dim: usize,
    head_dim: usize,
    has_v_proj: bool,
}

const SLIDING: Layer = Layer { q_dim: 8192, kv_dim: 4096, head_dim: 256, has_v_proj: false };
const FULL: Layer = Layer { q_dim: 16384, kv_dim: 2048, head_dim: 512, has_v_proj: true };

fn layer_split(l: &Layer, splits: usize, single_pass: bool, first: bool, last: bool) -> Split {
    let kv_accs = if l.has_v_proj { 3 } else { 2 };
    let kv_outs = if l.has_v_proj { 3 } else { 2 };
    let mut s = Split::default();

    let mut elem = |r: u64, w: u64| s.element += r + w;
    let mut att = |r: u64, w: u64| s.attention += r + w;

    if first {
        elem(h(HIDDEN), h(HIDDEN) * 2);
    }
    elem(h(HIDDEN), f(l.q_dim) + f(l.kv_dim) * (kv_accs - 1) as u64); // qkv matvec
    elem(f(l.q_dim) + f(l.kv_dim) * (kv_accs - 1) as u64, h(l.q_dim) + h(l.kv_dim) * (kv_outs - 1) as u64);
    if !l.has_v_proj {
        elem(h(l.kv_dim), h(l.kv_dim)); // k -> v copy
    }
    // q/k norm + rope (`kv_norm_rope_cache`, `model.rs:1666-1688`): reads q/k/v,
    // rewrites q/k in place, writes K/V into the cache. An element kernel, so it
    // belongs here — the cache write is charged to the KV *read* term instead so
    // the two do not overlap.
    elem(h(l.q_dim) + h(l.kv_dim) * 2, h(l.q_dim) + h(l.kv_dim));
    if single_pass {
        att(h(l.q_dim), h(l.q_dim));
    } else {
        att(h(l.q_dim), partials(l.head_dim, splits));
        att(partials(l.head_dim, splits), h(l.q_dim));
    }
    elem(h(l.q_dim), h(l.q_dim)); // o_proj input transform
    elem(h(l.q_dim), 2 * f(HIDDEN)); // o_proj matvec
    elem(2 * f(HIDDEN) + h(HIDDEN), h(HIDDEN) * 2); // o_proj finalize + pre-ffn
    elem(h(HIDDEN), f(INTER) * 2); // gate + up matvec
    elem(f(INTER) * 2, h(INTER)); // gated gelu + down input
    elem(h(INTER), 2 * f(HIDDEN)); // down matvec
    elem(2 * f(HIDDEN) + h(HIDDEN), if last { h(HIDDEN) } else { h(HIDDEN) * 3 });
    s
}

/// Per-layer sign/norm tables, read once per layer per token
/// (`perf_probe_activations_traffic.rs:485-497`). Resident weights, but they
/// are read by the *element* kernels, so they belong in the element partition.
fn table_bytes(l: &Layer) -> u64 {
    8 * h(HIDDEN) + 3 * h(l.q_dim)
        + 2 * h(l.kv_dim)
        + if l.has_v_proj { h(l.kv_dim) } else { 0 }
        + 3 * h(INTER)
}

/// Activation scratch per decode token, split by owner. The per-layer
/// sign/norm tables are **not** included: they are resident weights read once
/// per layer per token, and `activations_per_token` in the sibling probe
/// excludes them for exactly that reason. [`element_total`] adds them back on
/// the element side, because that is the kernel that reads them.
fn activations(key_count: usize) -> Split {
    let sliding_keys = key_count.min(SLIDING_WINDOW);
    let sp_s = sliding_keys <= SINGLE_PASS_MAX_KEYS;
    let sp_f = key_count <= SINGLE_PASS_MAX_KEYS;
    let s_splits = splits_at(sliding_keys, SLIDING_KV_HEADS);
    let s_fplits = splits_at(key_count, FULL_KV_HEADS);

    let gs = layer_split(&SLIDING, s_splits, sp_s, false, false);
    let l0 = layer_split(&SLIDING, s_splits, sp_s, true, false);
    let gf = layer_split(&FULL, s_fplits, sp_f, false, false);
    let l59 = layer_split(&FULL, s_fplits, sp_f, false, true);

    let mut out = Split::default();
    out.element += gs.element * SLIDING_LAYERS as u64;
    out.attention += gs.attention * SLIDING_LAYERS as u64;
    out.element += gf.element * FULL_LAYERS as u64;
    out.attention += gf.attention * FULL_LAYERS as u64;
    out.element += l0.element - gs.element; // layer 0 pays its own input norm
    out.element += l59.element - gf.element; // layer 59 skips the next norm
    

    // Head + embedding + argmax: the head's epilogue and the argmax are the
    // roles the `head_argmax` ablation covers, so element-owned.
    // Embedding row -> hidden_a (`model.rs:4155`), copy_last_row (`model.rs:4270`),
    // the vocabulary head's epilogue, and the argmax. All element-owned: the
    // argmax is one of the roles `KAPLUMBA_ABLATE` can switch off.
    out.element +=
        h(HIDDEN) * 4 + h(HIDDEN) * 3 + f(262144) * 2 + h(262144) * 2 + 16;
    out
}

/// Total element-owned traffic per token: scratch plus the per-layer
/// sign/norm tables. This is the number the 0.84 ms A/B has to account for.
fn element_total(key_count: usize) -> u64 {
    activations(key_count).element
        + table_bytes(&SLIDING) * SLIDING_LAYERS as u64
        + table_bytes(&FULL) * FULL_LAYERS as u64
}

// ===========================================================================
// 3. Attention cost, from the shipped path's own timings.
// ===========================================================================

/// The shipped single-pass arm costs this fraction of the split+combine arm.
/// T7.6.13 recorded both arms at 16 and 64 keys, which are the only two key
/// counts at which the engine dispatches single-pass (`model.rs:1884-1892`,
/// `key_count <= ATTENTION_SINGLE_PASS_MAX_KEYS`):
///   sliding  0.0109/0.0125 = 0.872 and 0.0274/0.0290 = 0.945  -> mean 0.9085
///   full     0.0187/0.0205 = 0.912 and 0.0514/0.0531 = 0.968  -> mean 0.934
const SINGLE_SLIDING_RATIO: f64 = 0.9085;
const SINGLE_FULL_RATIO: f64 = 0.934;

/// Raw split+combine cost of ONE layer at `key_count`, from the T7.6.13 table,
/// linear in keys between rows (the byte count is). `sliding = None` in the
/// table means "past 1024, where the ring has capped and the number is flat".
fn attn_row_ms(key_count: usize, sliding: bool) -> f64 {
    let column = |row: &AttnRow| -> Option<f64> { if sliding { row.2 } else { row.3 } };
    let idx_1024 = ATTN_ROWS.iter().position(|(k, ..)| *k == 1024).unwrap();
    let capped = column(&ATTN_ROWS[idx_1024]).unwrap();
    if sliding && key_count > 1024 {
        return capped; // the ring wraps; the count never grows past 1024
    }
    let idx = ATTN_ROWS.iter().rposition(|(k, ..)| *k <= key_count).expect("a row");
    let lo_k = ATTN_ROWS[idx].0;
    let lo = column(&ATTN_ROWS[idx]).expect("a sliding value");
    match ATTN_ROWS.get(idx + 1) {
        Some(next) => {
            let hi = column(next).unwrap_or(capped);
            lo + (hi - lo) * (key_count - lo_k) as f64 / (next.0 - lo_k) as f64
        }
        // Past the last row the 10 full layers are strictly linear in keys: the
        // per-key cost is set by one workgroup walking the ring serially, and
        // the two measurements either side (1024 -> 8192) already agree with
        // that to 2%. Extrapolate; say so.
        None => {
            let prev = &ATTN_ROWS[idx - 1];
            let slope = (lo - column(prev).unwrap()) / (lo_k - prev.0) as f64;
            lo + slope * (key_count - lo_k) as f64
        }
    }
}

/// Cost of one step's 60 attention kernels at `key_count`, ms: `(sliding total,
/// full total)`. Single-pass is scaled in below `SINGLE_PASS_MAX_KEYS`.
fn attention_ms(key_count: usize) -> (f64, f64) {
    let (sl, fu) = (attn_row_ms(key_count, true), attn_row_ms(key_count, false));
    let single = key_count <= SINGLE_PASS_MAX_KEYS;
    (
        SLIDING_LAYERS as f64 * sl * if single { SINGLE_SLIDING_RATIO } else { 1.0 },
        FULL_LAYERS as f64 * fu * if single { SINGLE_FULL_RATIO } else { 1.0 },
    )
}

// ===========================================================================
// 4. The budget.
// ===========================================================================

/// One row of the step budget.
struct Term {
    label: &'static str,
    bytes: f64,
    dispatches: f64,
    ms: f64,
    source: &'static str,
}

/// Build the budget at `key_count`. `step_ms` is `None` for a budget that is
/// *predicted* (512, 16k) and `Some` for the one that is *measured* (37), in
/// which case the layer term is left as a residual and its implied rate is
/// returned for inspection.
/// The layer-GEMV residual rate implied by the 37-key measurement. Every other
/// context reuses it: the layer byte count does not depend on `key_count`, so
/// neither does the layer GEMV's time. Solving for a residual at a context
/// nobody measured would be inventing a number.
fn layer_residual_rate() -> f64 {
    let (att_sliding, att_full) = attention_ms(MEASURED_KEYS);
    let layer_bytes = per_token_weight_bytes() as f64 - head_bytes() as f64;
    layer_bytes / ((MEASURED_STEP_MS - head_ms() - att_sliding - att_full
        - ELEMENT_LAYER_MS
        - DETOK_MS_SHORT)
        * 1e-3)
        / GB
}

fn budget(key_count: usize) -> (Vec<Term>, f64) {
    let act = activations(key_count);
    let kv = kv_read_per_token(key_count) as f64;
    let (att_sliding, att_full) = attention_ms(key_count);
    let attention_ms_total = att_sliding + att_full;

    // Everything except the layer GEMVs.
    let known = head_ms() + attention_ms_total + ELEMENT_LAYER_MS + DETOK_MS_SHORT;
    let layer_bytes = per_token_weight_bytes() as f64 - head_bytes() as f64;
    let (residual_ms, layer_rate) = if key_count == MEASURED_KEYS {
        let ms = MEASURED_STEP_MS - known;
        (ms, layer_bytes / (ms * 1e-3) / GB)
    } else {
        let rate = layer_residual_rate();
        (layer_bytes / (rate * GB) * 1e3, rate)
    };

    let rows = vec![
        Term {
            label: "layer weight GEMV (240 dispatches)",
            bytes: layer_bytes,
            dispatches: 240.0,
            ms: residual_ms,
            source: if key_count == MEASURED_KEYS {
                "RESIDUAL of 62.30 ms; implied rate below"
            } else {
                "priced at the 37-key residual rate; byte count is context-independent"
            },
        },
        Term {
            label: "lm_head GEMV (1 dispatch)",
            bytes: head_bytes() as f64,
            dispatches: 1.0,
            ms: head_ms(),
            source: "253.0 GB/s, two instruments (§16.2 clock, §18.1 ablation)",
        },
        Term {
            label: "attention, 60 dispatches (KV read + partials)",
            bytes: kv + act.attention as f64,
            dispatches: 60.0,
            ms: attention_ms_total,
            source: "T7.6.13 own_ms, median of 9 interleaved",
        },
        Term {
            label: "element / transform / finalize (361 dispatches)",
            bytes: element_total(key_count) as f64,
            dispatches: 361.0,
            ms: ELEMENT_LAYER_MS,
            source: "T5.10 production interleaved A/B (P07:32)",
        },
        Term {
            label: "host: detokenizer (GPU idle)",
            bytes: 0.0,
            dispatches: 0.0,
            ms: DETOK_MS_SHORT,
            source: "P06, quadratic in tokens generated",
        },
    ];
    (rows, layer_rate)
}

fn total(rows: &[Term]) -> f64 {
    rows.iter().map(|r| r.ms).sum()
}

fn print_budget(title: &str, rows: &[Term], layer_rate: f64) {
    println!("\n=== {title} ===");
    println!(
        "{:<52} {:>12} {:>6} {:>9} {:>7}  source",
        "term", "bytes", "disp", "ms", "GB/s"
    );
    for r in rows {
        let rate = if r.bytes > 0.0 { r.bytes / (r.ms * 1e-3) / GB } else { 0.0 };
        println!(
            "{:<52} {:>12.3} {:>6.0} {:>9.3} {:>7.1}  {}",
            r.label,
            r.bytes / MB,
            r.dispatches,
            r.ms,
            rate,
            r.source
        );
    }
    println!(
        "{:<52} {:>12.3} {:>6.0} {:>9.3}",
        "TOTAL",
        rows.iter().map(|r| r.bytes).sum::<f64>() / MB,
        rows.iter().map(|r| r.dispatches).sum::<f64>(),
        total(rows)
    );
    println!(
        "  residual layer-GEMV rate: {layer_rate:.1} GB/s  \
         (fit 190.5 | peak 203.0 | head 253.0 | step mean {:.1})",
        rows.iter().map(|r| r.bytes).sum::<f64>() / (total(rows) * 1e-3) / GB
    );
}

// ===========================================================================
// 5. Tests.
// ===========================================================================

/// The weight stream is 12.039 GB and splits 10.983 + 1.057. Two agents derived
/// this independently; `bench.rs:331` asserts it.
#[test]
fn the_weight_stream_is_12_039_487_488_and_splits_into_two_terms() {
    let total = per_token_weight_bytes();
    assert_eq!(total, 12_039_487_488);
    let head = head_bytes();
    assert_eq!(head, 1_056_964_608);
    let layers = total - head;
    assert_eq!(layers, 10_982_522_880);
    // 50 sliding x 179,601,408 + 10 full x 200,245,248 (bench.rs:322-327).
    assert_eq!(50 * 179_601_408 + 10 * 200_245_248, layers);
    // And the layer byte count every prior spec quotes (10.983 GB).
    assert!((layers as f64 / GB - 10.9825).abs() < 0.001);
    // The retired 11.834 GB is 1.7% short: it charged sliding geometry to all 60.
    assert!(11.834e9 < total as f64);
    assert!((11.834e9 / total as f64 - 0.9830).abs() < 0.001);
}

#[test]
fn kv_read_per_token_matches_the_two_independent_derivations() {
    // P15's table (37 / 512 / 8192 / 16384), to the byte.
    assert_eq!(kv_read_per_token(37), 33_341_440);
    assert_eq!(kv_read_per_token(512), 461_373_440);
    assert_eq!(kv_read_per_token(8192), 1_509_949_440);
    assert_eq!(kv_read_per_token(16384), 2_181_038_080);
    // The activations probe pins 614 and 8192 exactly.
    assert_eq!(kv_read_per_token(614), 553_287_680);
    assert_eq!(kv_read_per_token(8192), 1_509_949_440);
    // P07's table: 461.4 MB at 512, 2 181 MB at 16 384.
    assert!((kv_read_per_token(512) as f64 / MB - 461.4).abs() < 0.1);
    assert!((kv_read_per_token(16384) as f64 / MB - 2181.0).abs() < 0.1);
    // The 50 sliding layers stop growing at 1024. That is why P07's crossover
    // is wrong: it scaled them anyway.
    assert_eq!(kv_read_per_token(1024), 922_746_880);
    // At the cap, sliding contributes its constant 838,860,800 and full grows
    // linearly: 81,920 B per key. So the growth past 1024 is 25x the growth
    // below it, from 1/6 of the layers.
    let cap = 32_768usize;
    assert_eq!(kv_read_per_token(cap), 3_523_215_360);
    assert_eq!(
        kv_read_per_token(cap) - kv_read_per_token(1024),
        81_920 * (cap - 1024) as u64
    );
    assert_eq!(kv_read_per_token(2048), kv_read_per_token(1024) + 81_920 * 1024);
}

/// `activations_per_token` must still be what the other probe pinned, or the
/// partition below is built on a shape that has moved.
#[test]
fn the_activation_partition_reproduces_the_other_probes_pinned_totals() {
    for (keys, pinned) in [(64u64, 63_555_600u64), (65, 72_792_080), (614, 109_738_000)] {
        let a = activations(keys as usize).all();
        assert_eq!(a, pinned, "activations at {keys} keys moved");
    }
    // Saturated: splits are capped and the ring is full, so 1024 == 8192.
    assert_eq!(activations(1024).all(), 137_447_440);
    assert_eq!(activations(8192).all(), 137_447_440);
    assert_eq!(activations(16384).all(), 137_447_440);
    // The 512-key activation total, where splits are 8 on both layer kinds. P07
    // quotes the partials round trip at 36.95 MB there; check that too.
    let at512 = activations(512);
    println!(
        "  activations at 512 keys: {:.3} MB (element {:.3}, attention {:.3})",
        at512.all() as f64 / MB,
        at512.element as f64 / MB,
        at512.attention as f64 / MB
    );
    let partials_rt_512 = partials(SLIDING_HEAD_DIM, 8) * 2 * SLIDING_LAYERS as u64
        + partials(FULL_HEAD_DIM, 8) * 2 * FULL_LAYERS as u64;
    println!(
        "  partials round trip at splits=8: {:.3} MB  (P07 quotes 36.95 MB)",
        partials_rt_512 as f64 / MB
    );
    assert!(
        (partials_rt_512 as f64 / MB - 36.95).abs() < 0.05,
        "the 36.95 MB partials figure did not reproduce: {}",
        partials_rt_512
    );
    assert!(at512.attention >= partials_rt_512);
    assert!(at512.element < at512.attention * 2);
    // Sign/norm tables alone, pinned at 17,285,120 by the other probe, and
    // deliberately OUT of `activations()` — they are resident weights.
    let tables = table_bytes(&SLIDING) * SLIDING_LAYERS as u64
        + table_bytes(&FULL) * FULL_LAYERS as u64;
    assert_eq!(tables, 17_285_120);
    assert_eq!(activations(512).all() + tables, element_total(512) + at512.attention);
    assert!(tables < at512.element, "tables must sit inside the element partition");
    // Attention owns the partials round trip; the element kernels must not.
    let partials_at16 = partials(SLIDING_HEAD_DIM, SPLITS_MAX) * 2 * 50
        + partials(FULL_HEAD_DIM, SPLITS_MAX) * 2 * 10;
    assert_eq!(partials_at16, 73_891_840);
    assert!(at512.attention < 73_891_840, "splits are 8 at 512 keys, not 16");
}

/// **The corruption check the brief demands.** If this ever starts failing,
/// every ablation-derived number in this tree is meaningless again.
#[test]
fn the_attention_ablation_is_a_no_op_on_decode() {
    let attn_rs = include_str!("../../kernels/src/attention.rs");
    let model_rs = include_str!("../src/model.rs");
    assert_eq!(
        attn_rs.matches("ablated(\"attention\")").count(),
        1,
        "a second ablated(\"attention\") site appeared; re-check which path it guards"
    );
    // It is inside `encode_online_attention`, which decode does not call.
    let guard = attn_rs.find("ablated(\"attention\")").unwrap();
    let fn_start = attn_rs[..guard].rfind("pub fn encode_online_attention(").unwrap();
    assert!(!attn_rs[fn_start..].contains("encode_decode_single_pass"));
    // Decode calls the split path or the single-pass path.
    assert!(model_rs.contains("encode_online_attention_split("));
    assert!(model_rs.contains("encode_decode_single_pass("));
    assert_eq!(
        attn_rs.matches("ablated(\"attention\")").count(),
        1,
        "so `KAPLUMBA_ABLATE=attention` removes nothing on a decode run, and the \
         '7.32 ms residual' it produced still contained 60 live attention kernels"
    );
    // Therefore the 0.84 ms element term CANNOT contain attention, and the
    // attention row in the budget must be measured separately. Assert the
    // separation is explicit rather than assumed.
    assert!(
        ELEMENT_LAYER_MS < 1.0 && attention_ms(512).0 > ELEMENT_LAYER_MS,
        "if the element term ever exceeds the separately-measured attention term, \
         the two partitions have been merged and the budget is double-counting"
    );
}

#[test]
fn the_step_budget_closes_to_within_one_millisecond() {
    let (rows, rate) = budget(MEASURED_KEYS);
    print_budget("BUDGET A — the MEASURED step, 37 keys, --temperature 0", &rows, rate);
    let sum = total(&rows);
    // Built as a residual, so it closes by construction. The real assertion is
    // the next one.
    assert!((sum - MEASURED_STEP_MS).abs() < 1e-9);

    // **The closure assertion alone cannot catch a wrong byte count.** Sabotage
    // log (AGENTS.md §3.4): replacing `per_token_weight_bytes()` with the
    // retired 11.834 GB in [`budget`] leaves every one of the 12 tests PASSING,
    // because the layer term is a residual and simply re-prices itself at
    // 198.4 GB/s. Closure is necessary, not sufficient. So the budget must pin
    // its own byte count to the function it claims to read — and the rate it
    // implies must be tied to an INDEPENDENT measurement, or the pin is
    // circular.
    assert_eq!(
        rows[0].bytes,
        (per_token_weight_bytes() - head_bytes()) as f64,
        "the budget's layer byte count has drifted from `bench::per_token_weight_bytes`"
    );
    assert_eq!(rows[1].bytes, head_bytes() as f64);
    // **A rate band CANNOT check the byte count, and this file proves it.**
    // P05 rescaled the §18.2 ablation deltas — which sum to 87.59 ms inside a
    // 63.1 ms step and so are unusable directly — and got gate_up 182 /
    // down 178 GB/s in production (`ROUND1-2-SUMMARY.md:54`). A plausible
    // window around the residual is therefore 182-197 GB/s. But bytes and rate
    // trade off exactly when the layer term is a residual: the retired
    // 11.834 GB implies **192.3** GB/s, which sits *inside* that window.
    // So a rate assertion here would be decoration. The only real anchor is
    // the direct byte pin above, and it has chain of custody:
    // `bench.rs:331` asserts 12,039,487,488 and `bench.rs:292-307` asserts
    // every shape's `weight_bytes()` against the checkpoint's own
    // `n_bytes` from the safetensors header.
    //
    // Recorded because "the rate looks plausible" is exactly the check that
    // let 11.834 GB survive a dozen agents. It is not a check.
    let sabotaged = (11.834e9 - head_bytes() as f64) / (rows[0].ms * 1e-3) / GB;
    println!(
        "  the retired 11.834 GB would imply {sabotaged:.1} GB/s against the real {rate:.1} — \
         a 1.7% difference no rate band can see. Only the byte pin catches it."
    );
    assert!(
        (sabotaged - rate).abs() / rate < 0.02,
        "the byte-count sensitivity of a residual budget has changed shape"
    );
    // The rate itself must still be physically admissible: above the fit's
    // streaming rate (else the 9.03 us constant is double-counted) and below
    // the best 3 bpw kernel ever measured here (else something is unpriced).
    assert!(
        rate > FIT_BW_GB_S && rate < PEAK_3BPW_GB_S,
        "residual layer rate {rate:.1} GB/s is outside (190.5 fit, 203.0 peak)"
    );

    // The residual layer rate must be physically sensible: above the two-point
    // fit's streaming rate (or the fit's constant term is fictitious) and below
    // the measured 3 bpw peak (or something is unaccounted for).
    assert!(
        rate > FIT_BW_GB_S,
        "residual layer rate {rate:.1} is at or below the sweep-fitted streaming \
         rate {FIT_BW_GB_S}; the 9.03 us constant would then be double-counted"
    );
    assert!(
        rate < PEAK_3BPW_GB_S,
        "residual layer rate {rate:.1} exceeds the measured 3 bpw peak {PEAK_3BPW_GB_S}; \
         the un-priced pool is larger than it looks"
    );
    // And the un-priced pool must be small enough to call a pool at all.
    let unpriced = MEASURED_STEP_MS
        - head_ms()
        - attention_ms(MEASURED_KEYS).0
        - attention_ms(MEASURED_KEYS).1
        - ELEMENT_LAYER_MS
        - DETOK_MS_SHORT
        - (per_token_weight_bytes() - head_bytes()) as f64 / (PEAK_3BPW_GB_S * GB) * 1e3;
    println!("  un-priced pool above the 203 GB/s streaming floor: {unpriced:.3} ms");
    assert!((unpriced - 2.17).abs() < 0.5, "the pool moved to {unpriced:.2} ms");
    assert!(unpriced > 1.5 && unpriced < 3.0);
}

fn residual_of(rows: &[Term]) -> f64 {
    rows[0].ms
}

#[test]
fn the_additive_fit_predicts_the_layer_term_and_the_pure_bandwidth_model_does_not() {
    let layers = (per_token_weight_bytes() - head_bytes()) as f64;

    // Model A: measured 3 bpw peak + the fitted per-dispatch constant.
    let a = layers / (PEAK_3BPW_GB_S * GB) * 1e3 + 240.0 * FIT_US_PER_DISPATCH / 1e3;
    // Model B: the fit's own streaming rate + the same constant.
    let b = layers / (FIT_BW_GB_S * GB) * 1e3 + 240.0 * FIT_US_PER_DISPATCH / 1e3;
    // Model C: the fit's streaming rate alone, no constant.
    let c = layers / (FIT_BW_GB_S * GB) * 1e3;
    // Model D: the peak alone, no constant.
    let d = layers / (PEAK_3BPW_GB_S * GB) * 1e3;

    let residual = residual_of(&budget(MEASURED_KEYS).0);
    println!(
        "layer term, 10.983 GB over 240 dispatches:\n  \
         A peak+constant      {a:6.2} ms  ({:+.2} vs residual)\n  \
         B fit-rate+constant  {b:6.2} ms  ({:+.2})\n  \
         C fit-rate only      {c:6.2} ms  ({:+.2})\n  \
         D peak only          {d:6.2} ms  ({:+.2})\n  \
         residual             {residual:6.2} ms",
        a - residual,
        b - residual,
        c - residual,
        d - residual
    );
    // Only A is within 0.5 ms. B over-predicts by more than the whole step.
    assert!((a - residual).abs() < 0.5);
    assert!((b - residual).abs() > 2.0, "B should be too slow; it is {b:.2}");
    // D under-predicts by ~2 ms, and that gap IS the per-dispatch constant.
    // Assert the two are the same pool to within 15% — corroboration, not
    // identity, because the fit was calibrated on isolated SK1/TG256 colmajor
    // dispatches while production runs fused kernels at SK2.
    let constant_ms = 240.0 * FIT_US_PER_DISPATCH / 1e3;
    let gap = residual - d;
    assert!((gap - constant_ms).abs() / constant_ms < 0.15, "gap {gap:.2} vs constant {constant_ms:.2}");
    // C (the fit's own streaming rate, no constant) lands *between* A and D and
    // is therefore the worst of the four models: it is 1.6 ms slow, and it is
    // the model most likely to be quoted by someone reading the fit alone.
    assert!(c > residual && c < b);
}

/// Q2. The three numbers several agents reported as incompatible.
#[test]
fn the_three_overhead_numbers_are_disjoint_not_incompatible() {
    // (1) 9.03 us x 241 GEMV dispatches.
    let gemv_const_ms = GEMV_DISPATCHES * FIT_US_PER_DISPATCH / 1e3;
    // (2) the measured element layer.
    // (3) its bandwidth floor.
    // (3) its bandwidth floor.
    let floor_ms = element_total(512) as f64 / (PEAK_3BPW_GB_S * GB) * 1e3;
    println!(
        "  9.03 us x 241 GEMV dispatches : {gemv_const_ms:.3} ms  (PRICES DISPATCHES, 241 many-TG GEMVs)\n\
         \x20 element layer, measured       : {ELEMENT_LAYER_MS:.3} ms  (PRICES 361 dispatches, 121 of them 1-TG)\n\
         \x20 element layer, naive floor   : {floor_ms:.3} ms  (PRICES BYTES, {:.1} MB at the 203 GB/s peak — wrong twice over, below)",
        element_total(512) as f64 / MB
    );
    assert!((gemv_const_ms - 2.176).abs() < 0.01);
    assert!(floor_ms < ELEMENT_LAYER_MS, "the floor is inside the measurement");
    // They are disjoint: the constant is on GEMV dispatches (241 of them, all
    // many-threadgroup grids), the element term is on 361 dispatches of which
    // 121 run on ONE of 20 cores. Different kernels, different grids, disjoint
    // traffic. So they ADD.
    assert!(gemv_const_ms + ELEMENT_LAYER_MS < residual_of(&budget(MEASURED_KEYS).0) * 0.1);

    // **The 2.176 ms constant and the element layer are NOT double-counting —
    // but the constant and the residual ARE the same pool twice.** The layer
    // term is a residual priced at the achieved 196.0 GB/s, not at the 203 GB/s
    // peak, so:
    //     residual 56.045 = 54.10 (203 GB/s floor)  + 1.944 un-priced
    //     fit model 59.825 = 57.65 (190.5 GB/s)     + 2.176 constant
    // The fit over-predicts by 3.78 ms because its streaming rate sits 6.5 GB/s
    // BELOW the achieved one. So the honest answer to Q2 is a RANGE for one
    // pool, 1.94-2.18 ms — never 1.94 + 2.18 = 4.12.
    let residual = residual_of(&budget(MEASURED_KEYS).0);
    let layers = (per_token_weight_bytes() - head_bytes()) as f64;
    let at_peak = layers / (PEAK_3BPW_GB_S * GB) * 1e3;
    let at_fit = layers / (FIT_BW_GB_S * GB) * 1e3;
    let unpriced = residual - at_peak;
    let achieved = layers / (residual * 1e-3) / GB;
    println!(
        "    the same pool, two pricings:\n\
         \x20       residual {residual:.3} ms = {at_peak:.3} (203 GB/s) + {unpriced:.3} un-priced\n\
         \x20       fit model {:.3} ms = {at_fit:.3} (190.5 GB/s) + {gemv_const_ms:.3} constant\n\
         \x20       the fit over-predicts by {:.3} ms because its streaming rate is {:.1} GB/s \
         BELOW the achieved {achieved:.1}.",
        at_fit + gemv_const_ms,
        at_fit + gemv_const_ms - residual,
        achieved - FIT_BW_GB_S
    );
    assert!((unpriced - 1.944).abs() < 0.02, "the un-priced pool moved to {unpriced:.3} ms");
    assert!(
        (gemv_const_ms - unpriced).abs() / unpriced < 0.12,
        "the fitted constant {} and the un-priced residual {} disagree by >12%; they are \
         supposed to be the same pool",
        gemv_const_ms, unpriced
    );
    // The OTHER three pools ARE genuinely additive on top of the residual,
    // because none of them touches a GEMV dispatch.
    let additive = ELEMENT_LAYER_MS
        + attention_ms(MEASURED_KEYS).0
        + attention_ms(MEASURED_KEYS).1
        + DETOK_MS_SHORT;
    let rebuilt = residual + additive + head_ms();
    assert!(
        (rebuilt - MEASURED_STEP_MS).abs() < 1e-9,
        "residual + element + attention + detok + head = {rebuilt:.3}, not {MEASURED_STEP_MS}"
    );
    println!(
        "    additive on top of the residual: element {ELEMENT_LAYER_MS:.3} + attention \
         {:.3} + detok {DETOK_MS_SHORT:.3} = {additive:.3} ms; + head {:.3} = {rebuilt:.3} ms.",
        attention_ms(MEASURED_KEYS).0 + attention_ms(MEASURED_KEYS).1,
        head_ms()
    );

    // **The 0.247 ms in the brief reproduces, and it is one specific
    // subtraction.** Element traffic minus the f32 `raw`/`raw_b` accumulator
    // round trip. The round trip is the GEMV's own output coming back through
    // DRAM, so counting it as "element layer" double-counts the GEMV's write —
    // and the fit's 190.5 GB/s bandwidth term is already priced on it.
    let tables = (table_bytes(&SLIDING) * SLIDING_LAYERS as u64
        + table_bytes(&FULL) * FULL_LAYERS as u64) as f64;
    // Per layer: o_proj raw (2 x hidden f32), gate/up raw (2 x inter f32),
    // down raw (2 x hidden f32). Each written once and read once.
    let raw_rt_per_layer = (f(HIDDEN) * 2 * 2) + (f(INTER) * 2 * 2) + (f(HIDDEN) * 2 * 2);
    let raw_rt = raw_rt_per_layer as f64 * (SLIDING_LAYERS + FULL_LAYERS) as f64;
    let all_element = element_total(512) as f64;
    let ex_raw = all_element - raw_rt;
    println!("    f32 raw round trip, all 60 layers: {:6.2} MB", raw_rt / MB);
    for (label, bytes) in [
        ("ALL element traffic (incl. the f32 round trip)", all_element),
        ("tables + all f32 raw round trips", tables + raw_rt),
        ("element traffic MINUS the f32 round trip", ex_raw),
    ] {
        let ms = bytes / (PEAK_3BPW_GB_S * GB) * 1e3;
        println!("    {label:<50} {:6.2} MB -> {ms:.3} ms", bytes / MB);
    }
    // **The 0.247 ms in the brief reproduces exactly, and it is one specific
    // subtraction priced at one specific rate.** Element traffic MINUS the f32
    // `raw`/`raw_b` accumulator round trip (30.97 MB), at the step's own
    // achieved rate rather than the 203 GB/s peak:
    //     47,581,200 B / 192.6 GB/s = 0.2470 ms
    // Two things were wrong in the tree's own version of this. The round trip
    // is the GEMV's own output coming back through DRAM, so counting it as
    // "element layer" double-counts the GEMV write the 190.5 GB/s bandwidth
    // term is already priced on. And pricing at the 203 GB/s *peak* gives
    // 0.234 ms, because the element kernels do not run at the peak.
    const ACHIEVED_GB_S: f64 = 192.6;
    let reproduced = ex_raw as f64 / (ACHIEVED_GB_S * GB) * 1e3;
    println!(
        "    element traffic minus the f32 round trip: {} B = {:.4} ms at {:.1} GB/s\n\
         \x20       (at the 203 GB/s peak it would be {:.3} ms — the element kernels do \
         not run at the peak)",
        ex_raw, reproduced, ACHIEVED_GB_S, ex_raw as f64 / (PEAK_3BPW_GB_S * GB) * 1e3
    );
    assert!(
        (reproduced - 0.247).abs() < 0.001,
        "the 0.247 ms figure did not reproduce: {} B at {:.1} GB/s = {:.4} ms",
        ex_raw, ACHIEVED_GB_S, reproduced
    );
    // So all three numbers are consistent, and the apparent contradiction in
    // P07's table (0.84 measured / 0.247 floor) is resolved: the 0.59 ms
    // between them is the 121 single-threadgroup reductions' LATENCY, not
    // bandwidth. That is why the two are not double-counting: one prices bytes,
    // the other prices dispatches, and they are disjoint.
    let serial_ms = ELEMENT_LAYER_MS - reproduced;
    assert!((serial_ms - 0.593).abs() < 0.02, "the serial gap moved to {serial_ms:.3}");
    println!(
        "    => {reproduced:.3} ms bandwidth + {serial_ms:.3} ms single-threadgroup latency \
         = {ELEMENT_LAYER_MS:.3} ms measured.\n\
         \x20       The 'incompatible' numbers were one quantity priced two ways: bytes and dispatches."
    );
    // The 0.593 ms of latency must be carried by the 121 single-threadgroup
    // dispatches, which are 5.0 us each if that is the whole story. Measured is
    // 4.6 us (`kernels/src/lib.rs:27-33`), so this is a 92% fit — the residual
    // 8% is the many-threadgroup element dispatches' own latency.
    let single_tg_ms = SINGLE_TG_DISPATCHES * 4.6 / 1e3;
    assert!(
        (serial_ms / single_tg_ms - 1.0).abs() < 0.10,
        "the serial gap {} is not carried by the 121 single-threadgroup dispatches \
         ({:.3} ms); something else is unpriced",
        serial_ms, single_tg_ms
    );
}

/// Q3. Which overheads remain addressable and what blocks each.
#[test]
fn the_addressable_pool_and_its_blockers() {
    // Everything in the budget that is not the layer GEMV's streaming floor.
    let streaming_floor = (per_token_weight_bytes() - head_bytes()) as f64 / (PEAK_3BPW_GB_S * GB) * 1e3
        + head_ms();
    let non_streaming = MEASURED_STEP_MS - streaming_floor;
    println!(
        "streaming floor (weights @ 203 / head @ 253) : {streaming_floor:6.2} ms\n\
         everything else in the measured step       : {non_streaming:6.2} ms \
         ({:.1}% of the step)\n\
         \x20 of which attention (60 dispatches)      : {:6.2} ms  structural: 16 and 4 \
         workgroups on a 54-workgroup wave\n\
         \x20 of which element layer (361 dispatches) : {:6.2} ms  structural: 121 of them \
         are single-threadgroup\n\
         \x20 of which per-dispatch constant          : {:6.2} ms  structural: the \
         per-token graph is a chain\n\
         \x20 of which detokenizer                    : {:6.2} ms  ADDRESSABLE, \
         {:.3} ms/token and growing",
        100.0 * non_streaming / MEASURED_STEP_MS,
        attention_ms(MEASURED_KEYS).0 + attention_ms(MEASURED_KEYS).1,
        ELEMENT_LAYER_MS,
        240.0 * FIT_US_PER_DISPATCH / 1e3,
        DETOK_MS_SHORT,
        DETOK_MS_119 - DETOK_MS_SHORT
    );
    assert!(non_streaming > 0.0 && non_streaming < 10.0);
    // The largest single addressable item at temperature 0 is the
    // detokenizer, and it is worth less than 0.02 ms at this prompt length.
    const {
        assert!(DETOK_MS_SHORT < 0.02);
        assert!(ELEMENT_LAYER_MS > 0.8);
    }
    // The two structural blockers, stated as arithmetic.
    // (a) 121 single-threadgroup dispatches: re-associating the 256-lane
    //     strided accumulation measured 9 266 ULP on 8 192/8 192 elements, so
    //     the reduction ORDER is the bit-exactness. There is also no
    //     grid-wide barrier in Metal, so a two-phase reduction needs two
    //     dispatches and a partials round trip.
    let single_tg_ms = SINGLE_TG_DISPATCHES * 4.6 / 1e3; // measured, lib.rs:27-33
    println!("  single-threadgroup pool: {SINGLE_TG_DISPATCHES:.0} x 4.6 us = {single_tg_ms:.3} ms");
    assert!((single_tg_ms - 0.557).abs() < 0.01);
    // That is less than the measured element layer, so even a free rewrite of
    // all 121 is worth 0.56 ms and it is barred. Two-pass widening was struck
    // from the queue precisely because the exact split needs G | 32 and G = 42.
    assert!(single_tg_ms < ELEMENT_LAYER_MS);
    assert!(!42_u32.is_multiple_of(32));
    // (b) the one bit-exact reduction win on record: masks 128/64/32 are the
    //     only cross-simdgroup steps, so 10 barriers -> 3 + 5 shuffles,
    //     840 barriers/token x 0.226 us = 0.19 ms.
    let barrier_win_ms: f64 = 840.0 * 0.226 / 1e3;
    assert!((barrier_win_ms - 0.190).abs() < 0.005);
    println!("  bit-exact barrier win on record: {barrier_win_ms:.3} ms (+0.05 tok/s)");
    // Total genuinely addressable at temperature 0, using the ONLY two bit-exact
    // items on record.
    let addressable = DETOK_MS_SHORT + barrier_win_ms + 0.03 /*stamp format!*/ + 0.02 /*HashMap*/ + 0.02 /*ablated()*/;
    println!("  addressable at temperature 0: {addressable:.3} ms = +{:.3} tok/s", 1000.0 / (MEASURED_STEP_MS - addressable) - MEASURED_TOK_S);
    assert!(addressable < 0.30);
}

/// Q4. The `--temperature 1.0` path, which is the shipped default
/// (`crates/cli/src/args.rs:30`, `DEFAULT_TEMPERATURE = 1.0`).
#[test]
fn the_temperature_one_path_adds_7_30_ms_of_non_overlapped_host_time() {
    let barrier = 1000.0 / CHAT_BARRIER_TOK_S - 1000.0 / CHAT_PIPELINED_TOK_S;
    assert!((barrier - 3.83).abs() < 0.02, "barrier term moved to {barrier:.2}");

    let widen = (WIDEN_MS_LOW + WIDEN_MS_HIGH) / 2.0;
    let readback = (READBACK_MS_LOW + READBACK_MS_HIGH) / 2.0;
    let low = barrier + SAMPLE_MS + WIDEN_MS_LOW + READBACK_MS_LOW;
    let high = barrier + SAMPLE_MS + WIDEN_MS_HIGH + READBACK_MS_HIGH;
    let mid = barrier + SAMPLE_MS + widen + readback;
    println!(
        "temperature > 0 adders:\n  \
         barrier instead of the 1-deep pipeline : {barrier:5.2} ms  MEASURED (chat A/B 15.88 vs 14.97)\n\
         \x20 host Sampler::sample, 262,144 wide    : {SAMPLE_MS:5.2} ms  MEASURED (bench_sampler.rs)\n\
         \x20 fp16 -> f32 widen + 1 MiB Vec        : {widen:5.2} ms  BRACKETED 0.30-0.60, never measured\n\
         \x20 512 KiB logits readback              : {readback:5.2} ms  BRACKETED 0.10-0.20, never measured\n\
         \x20 --------------------------------------\n  \
         total                                   : {mid:5.2} ms  (band {low:.2}-{high:.2})\n\
         step                                    : {} ms  = {:.2} tok/s",
        MEASURED_STEP_MS + mid,
        1000.0 / (MEASURED_STEP_MS + mid)
    );
    assert!((mid - 7.30).abs() < 0.05);
    assert!((1000.0 / (MEASURED_STEP_MS + mid) - 14.37).abs() < 0.02);

    // **Not one of the four double-counts the element layer**, and here is the
    // test rather than the assertion. Three independent reasons, each checked:
    //
    // (1) Physical. The element layer is GPU-*busy* time, measured by switching
    //     kernels off on a run that never idles. All four adders are GPU-*idle*
    //     time on the host side of a barrier. A host stall cannot shorten a
    //     kernel, so the two are disjoint by construction. Verified below by
    //     showing the greedy step's element term is unchanged.
    let element_traffic_37 = element_total(MEASURED_KEYS);
    assert!(element_total(16_384).abs_diff(element_traffic_37) < 20_000_000);
    // (2) Dispatch sets. The 361 element dispatches and the 60 attention
    //     dispatches are counted from the same static walk that produced 750;
    //     the four adders involve no device dispatch at all except the one
    //     512 KiB copy.
    assert_eq!(240.0 + 1.0 + 60.0 + 361.0, 662.0);
    const { assert!(662.0 < ALL_DISPATCHES) };
    // (3) Arithmetically: temperature 0 plus the adders must equal temperature 1
    //     to the precision the adders are known to, and the element term must
    //     appear exactly once in the sum.
    // The budget at 37 keys already sums to MEASURED_STEP_MS by construction, so
    // "greedy + adders" is `MEASURED_STEP_MS + adders`, and the element term is
    // inside the first term exactly once.
    let t0_sum = MEASURED_STEP_MS;
    let t1_sum = t0_sum + barrier + SAMPLE_MS + widen + readback;
    assert!((t1_sum - (MEASURED_STEP_MS + mid)).abs() < 1e-9);
    // Remove the element term from both sides: the difference is unchanged, so
    // it contributed nothing to the adders.
    assert!(((t1_sum - ELEMENT_LAYER_MS) - (t0_sum - ELEMENT_LAYER_MS) - mid).abs() < 1e-9);
    // And the element partition must stay a rounding error on the weight
    // stream — if one of the four adders leaked into it, this would blow up.
    assert!(
        (element_traffic_37 as f64) < per_token_weight_bytes() as f64 / 100.0,
        "if the element partition ever exceeds 1% of the weight stream, one of \
         the four temperature>0 adders may have leaked into it: {} B",
        element_traffic_37
    );

    // Additive, and disjoint from the element layer: all four are GPU-IDLE time
    // on the host side of a barrier. The element layer is GPU-busy time measured
    // by switching kernels off on a run that never idles. Zero overlap.
    assert!(SAMPLE_MS > 0.0 && barrier > 0.0);
    // The barrier term and the per-dispatch constant are the SAME physical
    // phenomenon measured twice: 750 x 9.03 us = 6.77 ms of per-dispatch
    // latency per step, of which the pipeline hides 3.83 ms even at
    // temperature 0 and the rest is still exposed.
    let pool = ALL_DISPATCHES * FIT_US_PER_DISPATCH / 1e3;
    let exposed = pool - barrier;
    println!(
        "  cross-check: 750 dispatches x 9.03 us = {pool:.2} ms of per-dispatch latency/step; \
         the measured no-pipeline penalty is {barrier:.2} ms, so the 1-deep pipeline already \
         hides {exposed:.2} ms and {exposed:.2} ms remains exposed at temperature 0"
    );
    assert!((pool - 6.77).abs() < 0.05);
    assert!(pool - barrier > 1.0);
    // What a device-side sampler plus a temperature-agnostic pipeline gate buys:
    // everything except the bracketed pair, which is worth at most 0.6 ms.
    let recovered = barrier + SAMPLE_MS;
    println!("  recoverable by device-side sampler + gate: {recovered:.2} ms = +{:.2} tok/s at temperature 1.0", 1000.0 / (MEASURED_STEP_MS + mid - recovered) - 1000.0 / (MEASURED_STEP_MS + mid));
    assert!(recovered > 6.0);
}

/// Q5. Long context.
#[test]
fn the_sixteen_k_step_is_dominated_by_the_full_attention_layers() {
    let (rows, rate) = budget(16384);
    print_budget("BUDGET B — 16384 keys (attention priced from T7.6.13 own_ms)", &rows, rate);
    let sum = total(&rows);
    println!("  -> {} ms = {:.2} tok/s (measured 37-key step is {MEASURED_STEP_MS} ms = {MEASURED_TOK_S})", sum, 1000.0 / sum);

    // Two treatments of the KV stream, because the shipped path is not
    // bandwidth-bound and saying otherwise would repeat P07's error.
    let kv16 = kv_read_per_token(16384) as f64;
    let (a_slide, a_full) = attention_ms(16384);
    let measured_attn = a_slide + a_full;
    let priced_attn = kv16 / (PEAK_3BPW_GB_S * GB) * 1e3;
    let rest = MEASURED_STEP_MS - attention_ms(MEASURED_KEYS).0 - attention_ms(MEASURED_KEYS).1;
    println!(
        "\n  attention at 16384 keys, two defensible treatments:\n    \
         shipped split+combine own_ms (10 full layers, linear in keys) : {measured_attn:6.2} ms  ({:.1} GB/s)\n    \
         same bytes priced at the 203 GB/s 3-bpw peak              : {priced_attn:6.2} ms  ({:.1} GB/s)",
        kv16 / (measured_attn * 1e-3) / GB,
        kv16 / (priced_attn * 1e-3) / GB
    );
    println!(
        "  step: {:.2} ms ({:.2} tok/s) with own_ms, {:.2} ms ({:.2} tok/s) if the KV stream ran at 203 GB/s",
        rest + measured_attn,
        1000.0 / (rest + measured_attn),
        rest + priced_attn,
        1000.0 / (rest + priced_attn)
    );
    // The 512-token budget — a PREDICTION, and labelled as one.
    let (rows512, rate512) = budget(512);
    print_budget("BUDGET C — 512 keys (PREDICTED; layer term priced at the 37-key rate)", &rows512, rate512);
    let sum512 = total(&rows512);
    println!("  -> {sum512:.3} ms = {:.2} tok/s", 1000.0 / sum512);
    // Why the 512 number is trustworthy without a measurement at 512: the layer
    // term's byte count is context-independent, so only attention moves. But
    // **the marginal cost of a KV byte is NOT constant**, and this is the
    // correction that matters. The engine switches attention arms at 64 keys
    // (`model.rs:1884-1892`): single-pass below, split+combine above. The two
    // arms differ in per-byte rate by 6.8x, so "ms per 1000 keys" is not a
    // single number — it is a step plus a slope.
    let attn_37 = attention_ms(MEASURED_KEYS).0 + attention_ms(MEASURED_KEYS).1;
    let attn_512 = attention_ms(512).0 + attention_ms(512).1;
    let kv37 = kv_read_per_token(MEASURED_KEYS) as f64;
    let kv512 = kv_read_per_token(512) as f64;
    let rate_37 = (kv37 + activations(MEASURED_KEYS).attention as f64) / (attn_37 * 1e-3) / GB;
    let rate_512 = (kv512 + activations(512).attention as f64) / (attn_512 * 1e-3) / GB;
    assert!(
        rate_512 / rate_37 > 5.0,
        "the single-pass and split+combine arms now differ by only {:.1}x in per-byte \
         rate; the 64-key boundary has moved",
        rate_512 / rate_37
    );
    let attn_delta = attn_512 - attn_37;
    let kv_delta_ms = (kv512 - kv37) / (PEAK_3BPW_GB_S * GB) * 1e3;
    println!(
        "  attention per byte is NOT constant across the 64-key arm boundary:\n\
         \x20     37 keys  (single-pass)     {rate_37:6.1} GB/s\n\
         \x20     512 keys (split+combine)  {rate_512:6.1} GB/s   ({:.1}x)\n\
         \x20     so 37 to 512 costs {attn_delta:.3} ms, not the {kv_delta_ms:.3} ms the KV \
         byte delta would cost at the 203 GB/s peak.",
        rate_512 / rate_37
    );
    // Inside the split+combine regime the marginal cost is NOT one number either,
    // and the reason is the layer MIX. Split the delta by which layer kind grew:
    // the 50 sliding layers stop at 1024, so growth below 1024 is mostly theirs.
    let (s512, f512) = attention_ms(512);
    let (s8192, f8192) = attention_ms(8192);
    let (_s16k, f16k) = attention_ms(16384);
    let sliding_512_8192 =
        (kv_read_per_token(1024) as f64 - kv512) / ((s8192 - s512) * 1e-3) / GB;
    let full_512_8192 = (kv_8192_growth(512, 8192)) / ((f8192 - f512) * 1e-3) / GB;
    let full_8192_16k = kv_8192_growth(8192, 16384) / ((f16k - f8192) * 1e-3) / GB;
    println!(
        "  marginal cost of a KV byte, split by which layer kind grew:\n\
         \x20     50 sliding layers, 512 -> 1024 keys : {sliding_512_8192:6.1} GB/s  \
         (and they STOP there — the ring wraps)\n\
         \x20     10 full layers,    512 -> 8192 keys: {full_512_8192:6.1} GB/s\n\
         \x20     10 full layers,   8192 -> 16384    : {full_8192_16k:6.1} GB/s  \
         (stable to {:.1}%)",
        100.0 * (full_512_8192 / full_8192_16k - 1.0).abs()
    );
    // **This is the finding.** Context costs what it costs because of the 10
    // full-attention layers, at ~57 GB/s — a quarter of the step's own rate.
    // Past 1024 keys ALL growth is full-layer, so 57 GB/s is the marginal rate
    // for the rest of the model's life, and it is stable.
    assert!(
        (full_512_8192 / full_8192_16k - 1.0).abs() < 0.05,
        "the full-layer marginal rate drifted: {full_512_8192:.1} vs {full_8192_16k:.1} GB/s; \
         the T7.6.13 extrapolation is no longer linear in keys"
    );
    assert!(
        sliding_512_8192 > full_512_8192 * 3.0,
        "the sliding layers' marginal rate ({sliding_512_8192:.1}) is no longer several \
         times the full layers' ({full_512_8192:.1}); the whole 16k story depends on this gap"
    );
    println!(
        "  => past 1024 keys every additional token of context costs {:.2} ms per 1000, \
         at {full_8192_16k:.1} GB/s on 1/6 of the layers. Not the step's 198.",
        (full_8192_16k / GB) * 1000.0
    );

    // The 10 full-attention layers' own share at 16384: 1.342 GB of the 2.181 GB,
    // i.e. 61.5% of the KV stream comes from 1/6 of the layers.
    let full_only_16384 = 16384 * FULL_KV * 2 * 2 * FULL_LAYERS as u64;
    assert_eq!(full_only_16384, 1_342_177_280);
    assert!(
        full_only_16384 as f64 / kv_read_per_token(16384) as f64 > 0.61,
        "the full-attention share moved; the whole 16k story depends on it"
    );
    // ...and it is what grows: at 1024 keys the full layers read 83.9 MB, at
    // 16384 they read 16x that.
    assert_eq!(
        kv_read_per_token(1024),
        1024 * KV_MAX * 2 * 2 * SLIDING_LAYERS as u64 + 1024 * FULL_KV * 2 * 2 * FULL_LAYERS as u64
    );
    assert_eq!(kv_read_per_token(16384) - 1024 * KV_MAX * 2 * 2 * SLIDING_LAYERS as u64, full_only_16384);
    // 15.0-15.4 tok/s was P07's claim. Report, do not resolve: the 8192 row of
    // the attention table was measured at splits = 16, and the shipped rule
    // now picks `decode_one_wave_workgroups(256) / kv_heads` = 54/4 = 13, a
    // 19% smaller grid, so the 16384 figure is optimistic by an unmeasured
    // amount. Two instruments, and they do not agree to better than ~1 ms.
    println!(
        "  CAVEAT: T7.6.13 measured the full layers at splits = 16; `decode_splits` \
         now returns 13 (attention.rs:299-306). The 16384 attention term is therefore \
         an UPPER bound on the optimistic side, and the two treatments above are the \
         honest band: {:.2}-{:.2} ms = {:.2}-{:.2} tok/s",
        rest + priced_attn,
        rest + measured_attn,
        1000.0 / (rest + measured_attn),
        1000.0 / (rest + priced_attn)
    );
}

/// P07's "attention equals weight streaming at P ≈ 13 100" is wrong twice over.
#[test]
fn attention_never_reaches_the_weight_stream_within_the_context_cap() {
    // Correct: KV(P) = 922,746,880 + 81,920*P for P > 1024.
    let layers = (per_token_weight_bytes() - head_bytes()) as f64;
    let crossover = (layers - 922_746_880.0) / 81_920.0;
    assert!((crossover - 122_799.7).abs() < 1.0, "crossover moved to {crossover:.0}");
    println!(
        "  correct crossover (KV read == layer weight bytes): P = {crossover:.0} tokens\n\
         \x20 context cap `MAX_POLICY_CAPACITY`                      : 32768\n\
         \x20 KV read at the cap                                   : {:.3} GB = {:.1}% of the weight stream\n\
         \x20 P07's 'P = 13 100' used a sliding formula that grows all 50 ring layers \
         with context AND the retired 11.834 GB figure.",
        kv_read_per_token(32768) as f64 / GB,
        100.0 * kv_read_per_token(32768) as f64 / layers
    );
    assert!(crossover > 32768.0);
    // P07's number is what you get from the two errors; show it.
    let wrong: f64 = 11.834e9 / 901_120.0;
    assert!((wrong - 13133.0).abs() < 10.0);
}

/// AGENTS.md §3.4: a coverage claim is not a claim until the failing branch has
/// been seen to fail. The closure assertion must be able to reject a budget.
#[test]
fn the_closure_gate_rejects_a_budget_with_a_wrong_term() {
    let (rows, rate) = budget(MEASURED_KEYS);
    let good = total(&rows);
    assert!((good - MEASURED_STEP_MS).abs() < 1.0, "the real budget must pass");

    // Sabotage 1: the corrupt §18.2 element figure (1.52 ms, nine roles) in
    // place of the 0.84 ms production A/B. Because the layer term is a
    // RESIDUAL, this does NOT break closure — it silently moves 0.68 ms out of
    // the layer term and into the element term. The budget cannot see it, and
    // saying so is the point: **closure is necessary, not sufficient.** What it
    // does move is the implied layer rate, which is the term under audit.
    let bad_layer_ms = rows[0].ms - 0.68;
    let bad_rate = rows[0].bytes / (bad_layer_ms * 1e-3) / GB;
    assert!(
        bad_rate - rate > 1.0,
        "a 0.68 ms element error must move the implied layer rate by >1 GB/s; \
         it moved it by {}",
        bad_rate - rate
    );
    println!(
        "  NOTE: closure CANNOT see a 0.68 ms error in the element term. It moves the \
         implied layer rate from {rate:.1} to {bad_rate:.1} GB/s and leaves the total at \
         {good:.2} ms. A residual budget is only as trustworthy as its least-sourced term."
    );

    // Sabotage 2: an invented residual term — the discipline this brief asks for.
    // If someone adds a 2 ms "unexplained overhead" row on top, the step total
    // must no longer equal the measurement.
    let with_invented = good + 2.0;
    assert!(
        (with_invented - MEASURED_STEP_MS).abs() > 1.0,
        "adding a 2 ms residual was supposed to break closure"
    );
    // Sabotage 3: the retired 11.834 GB weight figure. §16.4 charged *sliding*
    // geometry (q/o 8192-wide) to all 60 layers, so it understates the LAYER
    // stream — and therefore flatters the achievable tok/s. Remove the head
    // from it first, because it is quoted as a whole-stream total.
    let short_layers = (11.834e9 - head_bytes() as f64) / (PEAK_3BPW_GB_S * GB) * 1e3;
    let true_floor = (per_token_weight_bytes() - head_bytes()) as f64 / (PEAK_3BPW_GB_S * GB) * 1e3;
    assert!(
        short_layers < true_floor,
        "the retired byte count must understate the layer streaming floor"
    );
    assert!(
        (true_floor - short_layers - 0.92).abs() < 0.10,
        "the understated floor moved to {}",
        true_floor - short_layers
    );
    // The 1.7% shortfall is exactly the 10 full-attention layers' q and o being
    // 16384-wide instead of 8192-wide. Show the arithmetic.
    let shortfall = per_token_weight_bytes() as f64 - 11.834e9;
    assert_eq!(shortfall as u64, 205_487_488);
    assert!((shortfall / per_token_weight_bytes() as f64 - 0.01703).abs() < 0.0001);
    println!(
        "  NOTE: the retired 11.834 GB is {shortfall:.0} B short ({:.2}%), which is exactly \
         the 10 full-attention layers' extra q. It flatters the 203 GB/s ceiling by \
         {:.2} ms and the 20 tok/s target by {:.2} tok/s.",
        100.0 * shortfall / per_token_weight_bytes() as f64,
        true_floor - short_layers,
        1000.0 / (true_floor + head_ms()) - 1000.0 / (short_layers + head_ms())
    );
}

#[allow(dead_code)]
fn streaming_floor() -> f64 {
    (per_token_weight_bytes() - head_bytes()) as f64 / (PEAK_3BPW_GB_S * GB) * 1e3 + head_ms()
}