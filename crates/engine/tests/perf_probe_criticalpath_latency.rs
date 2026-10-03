//! Critical-path / latency decomposition of one decode step. No GPU, no model.
//!
//! # The question
//!
//! A decode step is **62.3 ms** (16.05-16.10 tok/s) and streams **12.055 GB** of
//! weights. Two "ceilings" have been written down and neither closes:
//!
//! * `203 GB/s` is the best 3 bpw EXL3 GEMV measured on this box
//!   (`spec/decode-25tok-peak-bandwidth/implementation_notes.md` §16.3). Perfect
//!   kernels therefore give `12.055 / 0.203 = 59.4 ms = 16.8 tok/s`.
//! * `253 GB/s` is what `head/gemv` reaches (§18.2). If everything ran at head
//!   speed the step would be `47.6 ms = 21.0 tok/s`.
//!
//! So there is a **2x** gap and, as of this file, **no explanation of it**. The
//! working assumption has been "bandwidth". This probe tests that assumption
//! against measurements already in the tree and finds it is wrong in a specific,
//! quantifiable way: the shortfall is a **constant per dispatch**, not a
//! bandwidth deficit, and the head's speed is a **bytes-per-tile** effect, not a
//! latency effect at all.
//!
//! # What is measured and what is asserted
//!
//! Nothing here is a new measurement. Every input is a number already recorded
//! in the repo or read out of the checkpoint, and the arithmetic between them is
//! asserted so that a change to any anchor fails loudly instead of quietly
//! making the conclusion wrong.
//!
//! * **Bytes per projection**: read from
//!   `~/.kaplumba/models/Gxma-repacked/quantization_config.json`, whose
//!   `tensor_storage[*].stored_tensors[*].n_bytes` is the exact on-disk size of
//!   every quantised tensor. Those numbers are embedded below as
//!   [`anchors::BYTES_PER_LAYER`] because a test that must not touch the disk
//!   cannot depend on a 15 GB checkpoint being present.
//! * **Per-shape GEMV bandwidth**: `finding 09-splitk-table.md`, SK=1 / 256
//!   threads, colmajor 3 bpw, median of 11 interleaved rounds per cell, 120
//!   cells, on the shapes this engine actually dispatches.
//! * **One-wave threadgroup budget**: read from production code, not copied -
//!   [`decode_one_wave_workgroups`] in `attention.rs`.
//! * **Rolling threshold**: read from production code -
//!   [`decode_max_dispatches_per_buffer`] / [`MAX_DISPATCHES_PER_BUFFER`].
//! * **Grid shapes of the single-threadgroup finalize kernels**: parsed out of
//!   `linear.rs` by brace matching, so if `dispatch_groups(..., 1, 1024)` ever
//!   becomes `dispatch_groups(..., 168, 32)` the test says so.
//!
//! # Honest caveats, stated here rather than buried
//!
//! 1. The per-shape sweep is **SK=1 / TG=256 colmajor**. Production uses the
//!    *fused* kernels (`fused3_2col`, `fused2_2col`) at SK2. §16.3 and finding 09
//!    both record that the isolated benchmark **inverts variant rankings**
//!    against production. So the fitted constant is a property of *these shapes
//!    at these footprints*, used here to answer "is the shortfall constant or
//!    proportional", and the production agreement in
//!    [`the_fitted_constant_predicts_the_production_aggregate`] is a
//!    *consistency check*, not a derivation.
//! 2. `own_ms` from either profiler is **not** used as an absolute time anywhere
//!    in this file. §16.2's table is internally inconsistent and
//!    [`the_profiler_table_does_not_agree_with_itself`] shows exactly where.
//!    Only its *shares* are treated as information, and only in
//!    [`the_decomposition_closes`].
//! 3. The whole analysis is at the **short context the benchmarks use**
//!    (6-token prompt + 32 generated = position ~38). At that position the KV
//!    cache contributes 0.28% of the bytes. At 2048 it contributes 8.35%, and
//!    the same binary would be ~7% slower. See
//!    [`kv_traffic_is_negligible_only_at_the_benchmark_context`]. Any ceiling
//!    quoted from a `--max-new-tokens 32` run is a short-context ceiling.
//!
//! # Running it
//!
//! `cargo test --test perf_probe_criticalpath_latency` (no flags, no GPU).

use kaplumba_kernels::attention::decode_one_wave_workgroups;
use kaplumba_kernels::encoder::{decode_max_dispatches_per_buffer, MAX_DISPATCHES_PER_BUFFER};

const LINEAR_RS: &str = include_str!("../../kernels/src/linear.rs");
const ATTENTION_RS: &str = include_str!("../../kernels/src/attention.rs");
const MODEL_RS: &str = include_str!("../src/model.rs");

// ---------------------------------------------------------------------------
// Anchors. Every constant below carries where it came from.
// ---------------------------------------------------------------------------

mod anchors {
    /// Wall time of one decode step at HEAD `5e8ab73`, from the brief that
    /// commissioned this probe: 16.05-16.10 tok/s. `1000 / 16.05 = 62.31 ms`,
    /// `1000 / 16.10 = 62.11 ms`, so 62.3 ms is the slow end and is used
    /// throughout (the conservative choice).
    ///
    /// For scale: §16.1 of `spec/decode-25tok-peak-bandwidth` records 15.849
    /// tok/s = 63.1 ms on HEAD `223ea09`, so this anchor has improved ~1.3%
    /// since that spec was written.
    pub const MS_PER_STEP: f64 = 62.3;

    /// Best 3 bpw EXL3 GEMV on this box, `colmajor_2col_sk4` at K=21504, from
    /// the `gemv_bandwidth.rs` microbenchmark (§16.3). Note this is a
    /// *no-fixed-cost* number: the microbenchmark cannot see a per-dispatch
    /// constant in the same way production pays one, which is the whole point
    /// of [`super::the_fitted_constant_predicts_the_production_aggregate`].
    pub const PEAK_3BPW_GB_S: f64 = 203.0;

    /// `head/gemv` bandwidth, §18.2. Two independent instruments agree to 1.2%:
    /// the GPU-clock profiler (`1.057 GB / 4.12 ms = 257 GB/s`, §16.2) and a
    /// production wall-clock ablation (`+1.12 tok/s = 4.17 ms`, §18.1). 253 is
    /// the lower of the two and is used here.
    pub const HEAD_GB_S: f64 = 253.0;

    /// M4 Pro wire bandwidth. AGENTS.md §10.3.
    pub const WIRE_GB_S: f64 = 273.0;

    /// Wall-clock cost of all nine `KAPLUMBA_ABLATE` element-kernel roles:
    /// `+0.65 tok/s` = 1.52 ms (§18.1, production interleaved A/B, n>=5).
    /// This is the *whole* non-GEMV GPU budget: norms, rope, KV cache append,
    /// attention, the GEMV epilogues, softcap and argmax.
    pub const NON_GEMV_MS: f64 = 1.52;

    /// A GEMV dispatch that runs many threadgroups: measured marginal cost
    /// 0.7 us/dispatch (`input_transform`, ~42 threadgroups of 32 threads),
    /// `crates/kernels/src/lib.rs:27-33`.
    pub const US_PER_DISPATCH_MANY_TG: f64 = 0.7;

    /// A GEMV dispatch that runs **one** threadgroup: 4.6 us/dispatch, measured
    /// over the five fused transform/finalize roles at x300/token. Same source.
    /// The 6.6x ratio between these two numbers *is* the finding: the cost is a
    /// property of the grid, not of the dispatch API.
    pub const US_PER_DISPATCH_ONE_TG: f64 = 4.6;

    /// Profile stamps per decode step (§16.2 refers to "546 boundaries"). A
    /// stamp is not a dispatch - `commit_stamped` closes a command buffer at
    /// each one (`encoder.rs:677`) - and `profile.stamp` is called 35 times in
    /// `model.rs`. 9 stamps/layer x 60 + 6 in the head/epilogue ~= 546.
    pub const STAMPS_PER_STEP: f64 = 546.0;

    /// Per-buffer profiler artefact measured by `finding 04`, which summed
    /// 76.201 ms over 905 stamped buffers against 72.74 ms for the same work in
    /// one buffer: `3.8 us/buffer`.
    pub const PROFILER_US_PER_BUFFER: f64 = 3.8;

    /// `spec/decode-25tok-peak-bandwidth` §16.4's own byte table. It assumes
    /// `q 8192x5376, k 4096x5376, v 4096x5376` on **all 60** layers at exactly
    /// 3.000 bpw and 6 bpw on the head.
    pub const NOTES_16_4_BYTES: f64 = 11.834e9;

    /// Per-layer bytes of one quantised projection, straight out of
    /// `Gxma-repacked/quantization_config.json` `n_bytes`.
    ///
    /// Gxma has 60 layers, 50 sliding and 10 full-attention (indices
    /// 5, 11, ... 59; `config.json` `layer_types`). Full-attention layers use
    /// `global_head_dim = 512` with `num_global_key_value_heads = 4`, so `q` is
    /// 64x512 = 16384 wide and `k` is 4x512 = 2048 wide, and `v_proj` does not
    /// exist at all because `attention_k_eq_v = true`.
    ///
    /// That last fact is the one §16.4 gets wrong, and it is why §16.4 is 1.84%
    /// short: it charged a `v_proj` to all 60 layers.
    pub const SLIDING_Q: u64 = 16_542_212;
    pub const FULL_Q: u64 = 33_073_668;
    pub const SLIDING_K: u64 = 8_276_484;
    pub const FULL_K: u64 = 4_143_620;
    pub const SLIDING_V: u64 = 8_276_484;
    pub const MLP: u64 = 43_405_828;
    pub const HEAD: u64 = 1_057_499_652;
    pub const N_SLIDING: u64 = 50;
    pub const N_FULL: u64 = 10;
    pub const N_LAYERS: u64 = 60;

    /// GQA ratios: 32 q heads / 16 kv heads sliding, 32 / 4 global.
    pub const SLIDING_KV_HEADS: u64 = 16;
    pub const FULL_KV_HEADS: u64 = 4;
    pub const SLIDING_HEAD_DIM: u64 = 256;
    pub const FULL_HEAD_DIM: u64 = 512;
    /// `config.json` `sliding_window`.
    pub const SLIDING_WINDOW: u64 = 1024;
    /// KV cache element width: `gpu_cache.rs:30-31` stores `keys: Vec<u16>`.
    pub const KV_BYTES_PER_ELEMENT: u64 = 2;

    /// Decode context of the benchmark runs: a 6.01-token prompt plus
    /// `--max-new-tokens 32` (AGENTS.md §10.2.7, §16.1).
    pub const BENCHMARK_POSITION: u64 = 38;
}

/// One row of the per-shape sweep in `finding 09-splitk-table.md`.
#[derive(Clone, Copy, Debug)]
struct Shape {
    /// As printed in the finding, for the failure message.
    name: &'static str,
    /// `n_bytes` of that projection, per layer. This is the denominator that
    /// decides whether a shortfall can be a per-dispatch constant.
    bytes_per_dispatch: f64,
    /// SK=1 / TG=256 colmajor 3 bpw, median of 11 interleaved rounds.
    measured_gb_s: f64,
}

/// `finding 09-splitk-table.md`, "The table". The 10 column-major 3 bpw shapes
/// this engine dispatches. `full/k` is the informative one: 4.14 MB per
/// dispatch, an order of magnitude below `sliding/gate`'s 43.4 MB, and 27%
/// slower *in rate* despite having a third of the work.
const SHAPES: &[Shape] = &[
    Shape { name: "sliding/q", bytes_per_dispatch: 16_542_212.0, measured_gb_s: 172.6 },
    Shape { name: "sliding/k", bytes_per_dispatch: 8_276_484.0, measured_gb_s: 161.3 },
    Shape { name: "sliding/v", bytes_per_dispatch: 8_276_484.0, measured_gb_s: 161.1 },
    Shape { name: "sliding/o", bytes_per_dispatch: 16_542_212.0, measured_gb_s: 173.7 },
    Shape { name: "sliding/gate", bytes_per_dispatch: 43_405_828.0, measured_gb_s: 183.2 },
    Shape { name: "sliding/up", bytes_per_dispatch: 43_405_828.0, measured_gb_s: 179.8 },
    Shape { name: "sliding/down", bytes_per_dispatch: 43_405_828.0, measured_gb_s: 181.8 },
    Shape { name: "full/q", bytes_per_dispatch: 33_073_668.0, measured_gb_s: 178.4 },
    Shape { name: "full/k", bytes_per_dispatch: 4_143_620.0, measured_gb_s: 134.6 },
    Shape { name: "full/o", bytes_per_dispatch: 33_073_668.0, measured_gb_s: 179.5 },
];

/// Measured time of one dispatch, in microseconds.
fn dispatch_us(shape: &Shape) -> f64 {
    shape.bytes_per_dispatch / (shape.measured_gb_s * 1e3)
}

/// Bandwidth, in GB/s, a `(bytes_per_us, us_per_dispatch)` pair predicts.
fn predict_gb_s(bytes: f64, bytes_per_us: f64, us_per_dispatch: f64) -> f64 {
    bytes / (bytes / bytes_per_us + us_per_dispatch) / 1e3
}

/// Fit `t = bytes / BW + c` to two shapes: `(bytes_per_us, us)`.
///
/// This is the whole model. One free bandwidth and one **fixed cost per
/// dispatch**. No per-wave term, no per-threadgroup term, no launch term.
fn fit_two_parameter(a: &Shape, b: &Shape) -> (f64, f64) {
    let (ta, tb) = (dispatch_us(a), dispatch_us(b));
    assert!(
        (ta - tb).abs() > 1.0,
        "{} and {} are indistinguishable in time; pick anchors further apart",
        a.name, b.name
    );
    let bytes_per_us = (a.bytes_per_dispatch - b.bytes_per_dispatch) / (ta - tb);
    let us = ta - a.bytes_per_dispatch / bytes_per_us;
    (bytes_per_us, us)
}

/// Best pure-proportional fit: one free bandwidth, `c = 0`, chosen to minimise
/// the worst *time* residual. Exists so the two-parameter model can be shown to
/// earn its second parameter rather than merely possess it.
fn fit_one_parameter_best() -> (f64, f64) {
    let mut best = (f64::INFINITY, 0.0f64);
    let mut gbps = 100.0f64;
    while gbps <= 260.0 {
        let mut worst: f64 = 0.0;
        for shape in SHAPES {
            let measured = dispatch_us(shape);
            let predicted = shape.bytes_per_dispatch / (gbps * 1e3);
            worst = worst.max((predicted - measured).abs() / measured);
        }
        if worst < best.0 {
            best = (worst, gbps);
        }
        gbps += 0.1;
    }
    (best.1 * 1e3, best.0)
}

fn max_time_residual(bytes_per_us: f64, us: f64) -> f64 {
    let mut worst: f64 = 0.0;
    for shape in SHAPES {
        let measured = dispatch_us(shape);
        let predicted = shape.bytes_per_dispatch / bytes_per_us + us;
        worst = worst.max((predicted - measured).abs() / measured);
    }
    worst
}

// ---------------------------------------------------------------------------
// A minimal source scanner, so the grid shapes below are read from the file
// rather than transcribed from it. A transcribed constant proves the constant
// was transcribed; AGENTS.md §3.4 is about that distinction.
// ---------------------------------------------------------------------------

/// Body of `pub fn <name>(` in `source`, brace-matched, with string literals
/// and line comments skipped so a brace inside either cannot end the scan
/// early.
fn fn_body<'a>(source: &'a str, name: &str) -> &'a str {
    let needle = format!("pub fn {name}(");
    let start = source
        .find(&needle)
        .unwrap_or_else(|| panic!("no `pub fn {name}(` in the included source"));
    // Skip the parameter list: the first `)` at bracket depth 0 closes `name(`.
    let bytes = source.as_bytes();
    let (mut depth, mut i, mut in_string, mut in_comment) = (0usize, start + needle.len(), false, false);
    while i < bytes.len() {
        let c = bytes[i];
        if in_comment {
            if c == b'\n' {
                in_comment = false;
            }
        } else if in_string {
            if c == b'"' {
                in_string = false;
            }
        } else {
            match c {
                b'"' => in_string = true,
                b'/' if bytes.get(i + 1) == Some(&b'/') => {
                    in_comment = true;
                    i += 1;
                }
                b'(' | b'[' | b'{' => depth += 1,
                b')' | b']' => {
                    if depth == 0 {
                        break;
                    }
                    depth -= 1;
                }
                _ => {}
            }
        }
        i += 1;
    }
    assert_eq!(depth, 0, "parameter list of `{name}` never closed");
    // From the closing paren, the next `{` at depth 0 opens the body.
    let mut depth = 0usize;
    let mut i = i + 1;
    let mut open_at = i;
    let (mut in_string, mut in_comment) = (false, false);
    while i < bytes.len() {
        let c = bytes[i];
        if in_comment {
            if c == b'\n' {
                in_comment = false;
            }
        } else if in_string {
            if c == b'"' {
                in_string = false;
            }
        } else {
            match c {
                b'"' => in_string = true,
                b'/' if bytes.get(i + 1) == Some(&b'/') => {
                    in_comment = true;
                    i += 1;
                }
                b'{' => {
                    depth += 1;
                    if depth == 1 {
                        open_at = i;
                    }
                }
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        return &source[open_at..=i];
                    }
                }
                _ => {}
            }
        }
        i += 1;
    }
    panic!("unbalanced body for {name}");
}

/// The `groups` and `threads_per_group` arguments of the `n`-th
/// `dispatch_groups(` call inside a function body, as `(u64, u64)`.
fn dispatch_groups_args(body: &str, nth: usize) -> (u64, u64) {
    let mut from = 0usize;
    let mut args: Vec<String> = Vec::new();
    for _ in 0..=nth {
        args.clear();
        let at = body[from..]
            .find("dispatch_groups(")
            .unwrap_or_else(|| panic!("only found {nth} dispatch_groups calls"))
            + from
            + "dispatch_groups(".len();
        from = at;
        let bytes = body.as_bytes();
        let (mut depth, mut start, mut in_string) = (0usize, at, false);
        let mut i = at;
        while i < bytes.len() {
            let c = bytes[i];
            if in_string {
                if c == b'"' {
                    in_string = false;
                }
            } else {
                match c {
                    b'"' => in_string = true,
                    b'(' | b'[' | b'{' => depth += 1,
                    b')' if depth == 0 => {
                        // A trailing comma before the closing paren leaves an
                        // empty final slice; the call still has 5 arguments.
                        let tail = body[start..i].trim();
                        if !tail.is_empty() {
                            args.push(tail.to_string());
                        }
                        break;
                    }
                    b')' | b']' | b'}' => depth -= 1,
                    b',' if depth == 0 => {
                        args.push(body[start..i].trim().to_string());
                        start = i + 1;
                    }
                    _ => {}
                }
            }
            i += 1;
        }
    }
    // (pipeline, bindings, params, groups, threads_per_group)
    assert_eq!(
        args.len(),
        5,
        "dispatch_groups call did not split into 5 args: {args:?}"
    );
    let groups = args[3].parse::<u64>().unwrap_or_else(|_| {
        panic!(
            "dispatch_groups group count is not a literal: {:?}. This probe needs it to be \
             a compile-time constant so the grid can be computed without a GPU; a computed \
             grid means this file's wave arithmetic has stopped describing production.",
            args[3]
        )
    });
    let tpg = args[4].parse::<u64>().unwrap_or_else(|_| {
        panic!("dispatch_groups threads_per_group is not a literal: {:?}", args[4])
    });
    (groups, tpg)
}

/// `(groups, threads_per_group)` of the `n`-th `dispatch_1d(` call, where the
/// grid is `threads / threads_per_group` rather than literal.
fn dispatch_1d_args(body: &str, nth: usize) -> (String, String) {
    let mut from = 0usize;
    let mut args: Vec<String> = Vec::new();
    for _ in 0..=nth {
        args.clear();
        let at = body[from..]
            .find("dispatch_1d(")
            .unwrap_or_else(|| panic!("only found {nth} dispatch_1d calls"))
            + from
            + "dispatch_1d(".len();
        from = at;
        let bytes = body.as_bytes();
        let (mut depth, mut start, mut in_string) = (0usize, at, false);
        let mut i = at;
        while i < bytes.len() {
            let c = bytes[i];
            if in_string {
                if c == b'"' {
                    in_string = false;
                }
            } else {
                match c {
                    b'"' => in_string = true,
                    b'(' | b'[' | b'{' => depth += 1,
                    b')' if depth == 0 => {
                        let tail = body[start..i].trim();
                        if !tail.is_empty() {
                            args.push(tail.to_string());
                        }
                        break;
                    }
                    b')' | b']' | b'}' => depth -= 1,
                    b',' if depth == 0 => {
                        args.push(body[start..i].trim().to_string());
                        start = i + 1;
                    }
                    _ => {}
                }
            }
            i += 1;
        }
    }
    assert_eq!(args.len(), 5, "dispatch_1d call did not split into 5 args: {args:?}");
    (args[3].clone(), args[4].clone())
}

fn occurrences(haystack: &str, needle: &str) -> usize {
    haystack.matches(needle).count()
}

// ---------------------------------------------------------------------------
// Bytes
// ---------------------------------------------------------------------------

/// Bytes per generated token, per projection, from the checkpoint.
struct TokenBytes {
    q: u64,
    k: u64,
    v: u64,
    o: u64,
    /// gate, up and down are the same shape, so they are held once and counted
    /// three times. Kept as one field so the "is `up` really equal to `gate`"
    /// question cannot be answered wrong by editing two fields independently.
    mlp_each: u64,
    head: u64,
    layers: u64,
}

impl TokenBytes {
    fn from_checkpoint() -> Self {
        use anchors::*;
        let q = N_SLIDING * SLIDING_Q + N_FULL * FULL_Q;
        let k = N_SLIDING * SLIDING_K + N_FULL * FULL_K;
        let v = N_SLIDING * SLIDING_V;
        let o = N_SLIDING * SLIDING_Q + N_FULL * FULL_Q;
        let mlp = N_LAYERS * MLP;
        Self { q, k, v, o, mlp_each: mlp, head: HEAD, layers: q + k + v + o + 3 * mlp }
    }

    fn all(&self) -> u64 {
        self.layers + self.head
    }
}

fn kv_bytes_per_token(position: u64) -> u64 {
    use anchors::*;
    let sliding_keys = position.min(SLIDING_WINDOW);
    // keys and values, fp16, over the live window
    let sliding = N_SLIDING * sliding_keys * SLIDING_KV_HEADS * SLIDING_HEAD_DIM * KV_BYTES_PER_ELEMENT * 2;
    let full = N_FULL * position * FULL_KV_HEADS * FULL_HEAD_DIM * KV_BYTES_PER_ELEMENT * 2;
    sliding + full
}

// ---------------------------------------------------------------------------
// 1. Bytes: the arithmetic the notes use is 1.84% short, and the reason matters
// ---------------------------------------------------------------------------

#[test]
fn per_token_bytes_are_12_055_gb_from_the_checkpoint() {
    let t = TokenBytes::from_checkpoint();
    assert_eq!(t.q, 1_157_847_280);
    assert_eq!(t.k, 455_260_400);
    assert_eq!(t.v, 413_824_200);
    assert_eq!(t.o, 1_157_847_280);
    assert_eq!(t.mlp_each, 2_604_349_680);
    assert_eq!(t.head, 1_057_499_652);
    assert_eq!(t.layers, 10_997_828_200);
    assert_eq!(t.all(), 12_055_327_852);

    // 12.04 GB in the brief, 11.93 GB if the norms' 2.58 MB are folded in. Both
    // round from this one number.
    let gb = t.all() as f64 / 1e9;
    assert!((gb - 12.055).abs() < 0.002, "per-token bytes drifted: {gb:.4} GB");

    // The MLP is 64.8% of the token. Any plan that does not touch the MLP has
    // 35% of the budget to play with.
    let mlp_share = 3.0 * t.mlp_each as f64 / t.all() as f64;
    assert!((mlp_share - 0.6478).abs() < 0.002, "MLP share moved: {mlp_share:.4}");
}

#[test]
fn the_notes_byte_table_is_1_84_percent_low_because_v_does_not_exist_on_full_layers() {
    let t = TokenBytes::from_checkpoint();
    let truth = t.all() as f64;
    let notes = anchors::NOTES_16_4_BYTES;
    let low = 1.0 - notes / truth;
    assert!(
        (low - 0.0184).abs() < 0.0005,
        "§16.4's 11.834 GB is now {low:.4} below the checkpoint, not 1.84%"
    );

    // §16.4 charges q 8192x5376, k 4096x5376 and v 4096x5376 to all 60 layers.
    // The checkpoint has 10 full-attention layers with q 16384 wide, k 2048
    // wide, and **no v_proj at all** (`attention_k_eq_v = true`), so its q+k is
    // *larger* than the sliding layer's q+k+v even though it has two fewer
    // tensors. Reconciling the two byte tables exactly:
    let notes_qkv = 60u64 * (anchors::SLIDING_Q + anchors::SLIDING_K + anchors::SLIDING_V);
    let truth_qkv = t.q + t.k + t.v;
    assert_eq!(notes_qkv, 1_985_710_800);
    assert_eq!(truth_qkv, 2_026_931_880);
    // The whole 41.2 MB difference is the 10 full-attention layers' extra q.
    let full_extra_q = anchors::N_FULL * (anchors::FULL_Q - anchors::SLIDING_Q);
    let full_missing_kv = anchors::N_FULL * (anchors::SLIDING_K + anchors::SLIDING_V - anchors::FULL_K);
    assert_eq!(full_extra_q, 165_314_560);
    assert_eq!(full_missing_kv, 124_093_480);
    assert_eq!(truth_qkv - notes_qkv, full_extra_q - full_missing_kv);
    assert_eq!(truth_qkv - notes_qkv, 41_221_080);

    // Consequence, and it is not cosmetic: the ceiling moves.
    let ceiling_notes = notes / (anchors::PEAK_3BPW_GB_S * 1e9) * 1e3;
    let ceiling_truth = truth / (anchors::PEAK_3BPW_GB_S * 1e9) * 1e3;
    assert!((ceiling_notes - 58.30).abs() < 0.05, "notes ceiling moved: {ceiling_notes:.2}");
    assert!((ceiling_truth - 59.39).abs() < 0.05, "true ceiling moved: {ceiling_truth:.2}");
    assert!(
        ceiling_truth > ceiling_notes,
        "the checkpoint byte count gives a *worse* ceiling; the 12.04 GB in the brief is \
         right and §16.4's 11.834 GB is the optimistic one"
    );
}

// ---------------------------------------------------------------------------
// 2. Constant-per-dispatch vs bandwidth: the sweep decides it
// ---------------------------------------------------------------------------

#[test]
fn a_constant_per_dispatch_cost_is_required_by_the_measured_shape_sweep() {
    let (_, worst_one_param) = fit_one_parameter_best();
    let (bytes_per_us, us) = fit_two_parameter(&SHAPES[4], &SHAPES[8]);
    let worst_two_param = max_time_residual(bytes_per_us, us);

    // A single bandwidth cannot explain the table: `full/k` (4.14 MB/dispatch)
    // and `sliding/gate` (43.4 MB/dispatch) differ by 10.5x in work and 1.24x in
    // measured rate, and the best single BW still leaves a 15% worst-case time
    // error.
    assert!(
        worst_one_param > 0.10,
        "a pure-proportional model now fits the sweep to {:.1}%; the constant is no longer \
         required by the data and this probe's conclusion is stale",
        100.0 * worst_one_param
    );
    // With one fixed cost per dispatch it fits to 2.2%.
    assert!(
        worst_two_param < 0.03,
        "(BW, c) no longer explains the sweep: worst residual {:.1}%",
        100.0 * worst_two_param
    );
    assert!(
        worst_two_param * 4.0 < worst_one_param,
        "the second parameter is not earning its place: {:.1}% -> {:.1}%",
        100.0 * worst_one_param,
        100.0 * worst_two_param
    );
}

#[test]
fn the_fitted_cost_is_nine_microseconds_not_seven() {
    let (bytes_per_us, us) = fit_two_parameter(&SHAPES[4], &SHAPES[8]);
    let bw_gb_s = bytes_per_us / 1e3;
    assert!(
        (bw_gb_s - 190.5).abs() < 1.0,
        "fitted bandwidth moved to {bw_gb_s:.1} GB/s; the two-parameter model was fitted on \
         finding 09's SK1/TG256 table and that table has been re-measured"
    );
    assert!(
        (us - 9.03).abs() < 0.4,
        "fitted per-dispatch cost moved to {us:.2} us; every tok/s figure derived from it \
         in this file is now wrong"
    );

    // What the constant is worth, at the production dispatch count.
    // 4 GEMV dispatches per layer (qkv fused3, o, gate_up fused2, down) x 60
    // layers, plus the head: 241.
    let big = 241.0;
    assert_eq!(big, 4.0 * 60.0 + 1.0);
    let ms = big * us / 1e3;
    assert!((ms - 2.18).abs() < 0.05, "per-dispatch budget moved to {ms:.2} ms");

    // So: this is a *large* fixed cost. 9 us is 4.7% of `down_proj`'s ideal
    // 213.8 us and 0.16% of the head's 4179.8 us. The cost is negligible on
    // big dispatches and ruinous on small ones, which is exactly the shape of
    // the o_proj / full/k deficit.
    let down_ideal_us = 43_405_828.0 / (anchors::PEAK_3BPW_GB_S * 1e3);
    assert!(
        (us / down_ideal_us - 0.047).abs() < 0.005,
        "the constant is {:.1}% of down_proj's ideal stream time",
        100.0 * us / down_ideal_us
    );
}

#[test]
fn the_implied_constant_scales_with_whichever_dispatch_count_you_assume() {
    // The brief's "~7 us per dispatch" is (62.3 - 59.4) / 420. It is
    // self-consistent arithmetic and it is also a free parameter: the same
    // 2.91 ms divided by any plausible denominator gives a different constant.
    let gap_ms = anchors::MS_PER_STEP - 12_055_327_852.0 / (anchors::PEAK_3BPW_GB_S * 1e9) * 1e3;
    assert!((gap_ms - 2.91).abs() < 0.02, "the gap moved to {gap_ms:.2} ms");

    // GEMV dispatches only: 241. Every dispatch in a step, from the dispatch
    // walk in `model.rs`: 11 per layer x 60 + 5 in the head/epilogue = 665.
    let per_dispatch = |n: f64| gap_ms * 1e3 / n;
    assert!((per_dispatch(241.0) - 12.07).abs() < 0.05);
    assert!((per_dispatch(665.0) - 4.38).abs() < 0.05);
    assert!((per_dispatch(546.0) - 5.33).abs() < 0.05);
    // 7 us implies exactly this many dispatches, which is not a real number here.
    let implied = gap_ms * 1e3 / 7.0;
    assert!((implied - 415.7).abs() < 1.0, "the 7 us figure implies {implied:.0} dispatches");
    assert!(
        (implied - 241.0).abs() > 100.0 && (implied - 665.0).abs() > 100.0,
        "7 us now coincides with a real dispatch count; the argument that it is an artefact \
         of picking 420 has expired"
    );
    // The shape sweep, which does not depend on any of this, lands in the
    // middle of the bracket that 241..665 produces.
    let (_, fitted) = fit_two_parameter(&SHAPES[4], &SHAPES[8]);
    assert!(
        fitted > per_dispatch(665.0) && fitted < per_dispatch(241.0),
        "the sweep's {:.2} us should sit inside the {:.2}..{:.2} us bracket that the step \
         gap and the dispatch count imply",
        fitted,
        per_dispatch(665.0),
        per_dispatch(241.0)
    );
}

// ---------------------------------------------------------------------------
// 3. Production: does the fitted constant predict the measured aggregate?
// ---------------------------------------------------------------------------

#[test]
fn the_fitted_constant_predicts_the_production_aggregate() {
    let t = TokenBytes::from_checkpoint();
    let (_, us) = fit_two_parameter(&SHAPES[4], &SHAPES[8]);

    // Per token, at the 203 GB/s peak, plus one fixed cost per dispatch, on the
    // four production per-layer GEMV dispatches. 50 sliding layers, 10 full.
    let qkv_sliding = anchors::SLIDING_Q + anchors::SLIDING_K + anchors::SLIDING_V;
    let qkv_full = anchors::FULL_Q + anchors::FULL_K;
    let o_sliding = anchors::SLIDING_Q;
    let o_full = anchors::FULL_Q;
    let stream_us = anchors::PEAK_3BPW_GB_S * 1e3;

    let mut predicted_us = 0.0f64;
    for (bytes, count) in [
        (qkv_sliding as f64, 50.0),
        (qkv_full as f64, 10.0),
        (o_sliding as f64, 50.0),
        (o_full as f64, 10.0),
        (2.0 * anchors::MLP as f64, 60.0),
        (anchors::MLP as f64, 60.0),
    ] {
        predicted_us += count * (bytes / stream_us + us);
    }
    let predicted_ms = predicted_us / 1e3;
    assert!(
        (predicted_ms - 56.35).abs() < 0.6,
        "the model no longer predicts the layer budget: {predicted_ms:.2} ms"
    );

    // Measured layer budget: the step minus the two things measured separately.
    let head_ms = t.head as f64 / (anchors::HEAD_GB_S * 1e9) * 1e3;
    assert!((head_ms - 4.18).abs() < 0.02, "head time moved to {head_ms:.2} ms");
    let measured_layer_ms = anchors::MS_PER_STEP - anchors::NON_GEMV_MS - head_ms;
    assert!(
        (measured_layer_ms - 56.60).abs() < 0.02,
        "the measured layer budget moved to {measured_layer_ms:.2} ms"
    );
    let err = (predicted_ms - measured_layer_ms).abs() / measured_layer_ms;
    assert!(
        err < 0.01,
        "the (203 GB/s, {us:.2} us) model predicts {predicted_ms:.2} ms against a measured \
         {measured_layer_ms:.2} ms: {:.2}% error. The constant is not the missing term \
         any more, and every tok/s figure derived from this probe needs revisiting.",
        err * 100.0
    );

    // The number nobody has written down: the layer GEMVs, in production, right
    // now, are running at this.
    let achieved = t.layers as f64 / (measured_layer_ms * 1e-3) / 1e9;
    assert!(
        (achieved - 194.3).abs() < 0.15,
        "the production layer-GEMV rate moved to {achieved:.1} GB/s"
    );
    // Which is 96% of the best 3 bpw kernel in the tree, and above the
    // bandwidth the 10-shape sweep can support.
    assert!(achieved < anchors::PEAK_3BPW_GB_S);
    assert!(achieved > fit_two_parameter(&SHAPES[4], &SHAPES[8]).0 / 1e3);
}

// ---------------------------------------------------------------------------
// 4. Resolving the head contradiction. There isn't one: a 1000x unit error.
// ---------------------------------------------------------------------------

#[test]
fn the_head_contradiction_is_a_thousandfold_unit_error() {
    // The brief argues: "the head's overhead is amortised over 1.057 GB
    // (4.2 us of work at 253 GB/s), so the head should be the worst affected by
    // per-dispatch cost, yet it is the best."
    //
    // 1.057 GB at 253 GB/s is 4.18 **milliseconds**, not 4.2 microseconds. The
    // premise is off by 1000x, and with it the contradiction.
    let head_work_us = 1_057_499_652.0 / (anchors::HEAD_GB_S * 1e3);
    assert!(
        (head_work_us - 4179.8).abs() < 1.0,
        "the head's stream time moved to {head_work_us:.1} us"
    );
    assert!(
        head_work_us > 4000.0,
        "the head's work is {head_work_us:.1} us; the '4.2 us' in the brief is a \
         milliseconds-as-microseconds slip, and every conclusion drawn from it inverts"
    );

    // With the corrected arithmetic a 9 us constant is 0.2% of the head and
    // 4.7% of down_proj. So the head is the *least* affected by per-dispatch
    // cost, exactly as the "always amortise the constant" rule predicts.
    let (_, us) = fit_two_parameter(&SHAPES[4], &SHAPES[8]);
    assert!((us / head_work_us - 0.0022).abs() < 0.001);
    assert!(us / head_work_us < 0.01);
}

#[test]
fn the_head_beats_the_fitted_ceiling_by_thirty_three_percent_not_by_avoiding_overhead() {
    let t = TokenBytes::from_checkpoint();
    let (bytes_per_us, us) = fit_two_parameter(&SHAPES[4], &SHAPES[8]);

    // Under the *same* (BW, c) the 10-shape sweep supports, the head would run
    // at the fitted bandwidth: the constant is 0.2% of it, so it cannot be
    // what makes the difference.
    let head_model_gb_s = predict_gb_s(t.head as f64, bytes_per_us, us);
    assert!(
        (head_model_gb_s - 190.2).abs() < 1.0,
        "head under the sweep's model is {head_model_gb_s:.1} GB/s"
    );
    let ratio = anchors::HEAD_GB_S / head_model_gb_s;
    assert!(
        (ratio - 1.33).abs() < 0.02,
        "the head beats the 3 bpw model by {ratio:.2}x, not 1.33x"
    );

    // Why, from §20's measured 8%-load / 63%-ALU / 30%-floor partition of the
    // 3 bpw kernel (`spec/decode-19tok-repacked-250gb` §20). The decode is
    // *per weight*, and a 16x16 trellis tile holds 256 weights at both rates,
    // so a 6 bpw tile is 192 bytes where a 3 bpw tile is 96 bytes and does
    // **identical** ALU work. Per tile:
    const ALU: f64 = 63.0;
    const LOAD_3: f64 = 8.0;
    const FLOOR: f64 = 30.0;
    let t3 = ALU + LOAD_3 + FLOOR;
    let t6 = ALU + 2.0 * LOAD_3 + FLOOR;
    // Bandwidth is bytes over cost, so the ratio is (192/t6) / (96/t3): the 3
    // bpw kernel spends 101 cost units per 96 bytes, the 6 bpw kernel 109 per
    // 192. The load term is the only one that doubles.
    let predicted_ratio = (192.0 / t6) / (96.0 / t3);
    assert!(
        (predicted_ratio - 1.85).abs() < 0.02,
        "the partition predicts a {predicted_ratio:.2}x bytes-per-tile gain"
    );
    // Measured is 1.33x. So the head's kernel does the same tile-loop job about
    // 1.4x more efficiently per byte - and finding 09 already records that it
    // is "a different kernel family": 6 bpw row-major, not colmajor.
    assert!(
        predicted_ratio > ratio * 1.3,
        "the bytes-per-tile mechanism alone ({predicted_ratio:.2}x) would beat the measured \
         {ratio:.2}x; the head's extra advantage is a cheaper decode, not a bigger tile"
    );
    // The whole 2x the brief chases is 203 -> 253 on the *rate*, and rate is a
    // property of the decode, not of the schedule. Reach for bandwidth and you
    // will arrive at the ceiling the notes already named.
    assert!(anchors::HEAD_GB_S > anchors::PEAK_3BPW_GB_S);
    assert!(
        anchors::HEAD_GB_S < anchors::WIRE_GB_S,
        "the head at 253 GB/s is 93% of wire, so it is not far from DRAM-bound; the 3 bpw \
         kernel at 203 GB/s is at 74% of wire, which is where the headroom is not"
    );
}

// ---------------------------------------------------------------------------
// 5. Wave quantization is already inside the 203 GB/s number
// ---------------------------------------------------------------------------

#[test]
fn wave_quantization_is_real_but_already_bought() {
    // One wave is 54 threadgroups of 256 threads on this box, from production
    // code rather than from this file.
    assert_eq!(decode_one_wave_workgroups(256), Some(54));
    assert_eq!(decode_one_wave_workgroups(64), Some(288));
    assert_eq!(decode_one_wave_workgroups(128), None);

    // The four production per-layer GEMV grids, read out of the encoder's own
    // group arithmetic in `linear.rs`.
    let one_wave = decode_one_wave_workgroups(256).expect("measured for 256") as f64;
    // qkv: `fused3_2col`, groups = (tiles_q + tiles_k + tiles_v) / 2
    // sliding = (512 + 256 + 256) / 2 = 512
    // qkv on the 10 full-attention layers has **no v** and a half-width k, so the
    // tile count is 1024 + 128 = 1152, halved to 576 threadgroups - not 640.
    let qkv_sliding = (8192.0 / 16.0 + 4096.0 / 16.0 + 4096.0 / 16.0) / 2.0;
    let qkv_full = (16384.0 / 16.0 + 2048.0 / 16.0) / 2.0;
    // o_proj: `colmajor_2col_sk2`, groups = ((out/16)/2)*2 = out/16
    let o_sliding = 8192.0 / 16.0;
    let o_full = 16384.0 / 16.0;
    // gate+up: `fused2_2col`, groups = (tiles_gate + tiles_up) / 2
    let gate_up = (21504.0 / 16.0 + 21504.0 / 16.0) / 2.0;
    // down_proj: `colmajor_2col_sk2`, groups = 5376/16
    let down = 5376.0 / 16.0;

    assert_eq!(qkv_sliding, 512.0);
    assert_eq!(qkv_full, 576.0);
    assert_eq!(o_sliding, 512.0);
    assert_eq!(o_full, 1024.0);
    assert_eq!(gate_up, 1344.0);
    assert_eq!(down, 336.0);

    // These formulas are load-bearing for every wave number below, so pin the
    // source expressions themselves.
    assert!(LINEAR_RS.contains("(tiles_a + tiles_b + tiles_c) / 2"), "fused3_2col grid");
    assert!(LINEAR_RS.contains("(tiles_a + tiles_b) / 2"), "fused2_2col grid");
    assert!(
        LINEAR_RS.contains("(((out_features / 16) / 2) as u64) * 2"),
        "colmajor_2col_sk2 grid"
    );

    let mut rows = Vec::new();
    for (name, groups) in [
        ("qkv_proj", qkv_sliding),
        ("qkv_proj(full)", qkv_full),
        ("o_proj", o_sliding),
        ("o_proj(full)", o_full),
        ("gate_up", gate_up),
        ("down_proj", down),
    ] {
        let waves = groups / one_wave;
        let rounded = waves.ceil();
        rows.push((name, groups, waves, rounded, (rounded - waves) / rounded));
    }

    // down_proj is the worst-quantised shape in the model: 336 threadgroups is
    // 6.22 waves, so 11% of its waves are a partial wave pulling 22% of peak.
    let down_loss = rows.iter().find(|(n, ..)| *n == "down_proj").unwrap().4;
    assert!(
        (down_loss - 0.111).abs() < 0.002,
        "down_proj's wave quantization moved to {:.1}%",
        100.0 * down_loss
    );
    let gate_loss = rows.iter().find(|(n, ..)| *n == "gate_up").unwrap().4;
    assert!(
        (gate_loss - 0.0044).abs() < 0.0005,
        "gate_up's wave quantization moved to {:.2}%",
        100.0 * gate_loss
    );

    // **And here is why it is not a lever.** The 203 GB/s ceiling was measured
    // on these shapes at these grids (§16.3 sweeps K at N=5376, which is
    // down_proj; finding 09 sweeps the same 10 shapes). Wave quantization is
    // therefore *inside* the 203 GB/s number already. Adding it again to find
    // the gap would double-count it, and the 2.42 ms that would appear is
    // suspiciously close to the 2.17 ms of per-dispatch constant - which is the
    // coincidence that makes the arithmetic attractive and wrong.
    let t = TokenBytes::from_checkpoint();
    let head_ms = t.head as f64 / (anchors::HEAD_GB_S * 1e9) * 1e3;
    let layer_ms = anchors::MS_PER_STEP - anchors::NON_GEMV_MS - head_ms;
    let ideal_ms = t.layers as f64 / (anchors::PEAK_3BPW_GB_S * 1e9) * 1e3;
    let shortfall_ms = layer_ms - ideal_ms;
    let (_, us) = fit_two_parameter(&SHAPES[4], &SHAPES[8]);
    let constant_ms = 241.0 * us / 1e3;
    assert!(
        (shortfall_ms - constant_ms).abs() / constant_ms < 0.15,
        "the layer shortfall ({shortfall_ms:.2} ms) should be explained by the per-dispatch \
         constant ({constant_ms:.2} ms); it is now {:.0}% away, so something else has moved",
        100.0 * (shortfall_ms - constant_ms).abs() / constant_ms
    );
    // If wave quantization were additive it would need to be *this* close to
    // the constant term to fit, from a completely different model. It is the
    // same number wearing a different hat.
    assert!(
        shortfall_ms > 0.0,
        "the layer GEMVs are now at or above the 203 GB/s peak with no fixed cost at all"
    );
}

// ---------------------------------------------------------------------------
// 6. The remaining single-threadgroup dispatches
// ---------------------------------------------------------------------------

#[test]
fn two_single_threadgroup_dispatches_per_layer_cost_half_a_millisecond() {
    // Read out of `linear.rs`, not transcribed.
    let o_finalize = fn_body(LINEAR_RS, "encode_finalize_o_proj_and_pre_ffn_transform_sk2");
    let cross_layer = fn_body(LINEAR_RS, "encode_cross_layer_finalize_and_norm");
    let input_norm = fn_body(LINEAR_RS, "encode_input_norm_and_qkv_transform");
    let input_transform = fn_body(LINEAR_RS, "encode_input_transform");

    // `dispatch_groups(..., 1, 1024)`: ONE threadgroup for a 5376-element
    // output. 5376/32 = 168 threadgroups would be one wave.
    assert_eq!(dispatch_groups_args(o_finalize, 0), (1, 1024));
    assert_eq!(dispatch_groups_args(cross_layer, 0), (1, 1024));
    assert_eq!(dispatch_groups_args(input_norm, 0), (1, 256));
    // ...whereas `encode_input_transform` is `(in/128)*32` threads at 32 per
    // group = 42 threadgroups, and measures 0.7 us rather than 4.6.
    let (threads, tpg) = dispatch_1d_args(input_transform, 0);
    assert_eq!(threads, "u64::from(input_blocks) * 32");
    assert_eq!(tpg, "32");
    let groups = (5376.0 / 128.0) * 32.0 / 32.0;
    assert_eq!(groups, 42.0);

    // Cross-layer fusion already removed one of the three. For a layer with a
    // successor it applies (`model.rs:1274`: decode, permutation_mode 1, rate 3,
    // no env flags), so `input_norm_and_qkv_transform` does not run for layers
    // 1..59.
    assert!(
        occurrences(MODEL_RS, "encode_cross_layer_finalize_and_norm") >= 1,
        "the cross-layer finalize is no longer dispatched; the per-layer single-threadgroup \
         count in this test just changed and its 0.55 ms figure is stale"
    );
    assert!(
        occurrences(MODEL_RS, "encode_input_norm_and_qkv_transform") >= 1
            && occurrences(MODEL_RS, "input_norm_precomputed") >= 2,
        "the input_norm_precomputed gate moved; re-walk the decode dispatch list before \
         trusting the 120-dispatch count below"
    );

    // Two per layer x 60 layers, at the measured one-threadgroup cost.
    let ms = 2.0 * 60.0 * anchors::US_PER_DISPATCH_ONE_TG / 1e3;
    assert!((ms - 0.552).abs() < 0.01, "the single-threadgroup budget moved to {ms:.3} ms");
    let tok_s = 1000.0 / (anchors::MS_PER_STEP - ms) - 1000.0 / anchors::MS_PER_STEP;
    assert!(
        (tok_s - 0.143).abs() < 0.01,
        "widening those grids is worth {tok_s:.3} tok/s, not 0.143"
    );

    // The mechanism, from `lib.rs:27-33`: a dispatch costs 0.7 us; a dispatch
    // that runs a single threadgroup costs 4.6 us. 6.6x for the same API call.
    // So the fix is a grid change, not a dispatch-count change.
    assert!(
        anchors::US_PER_DISPATCH_ONE_TG / anchors::US_PER_DISPATCH_MANY_TG > 6.0,
        "the two measured dispatch costs have converged; the grid hypothesis is untested again"
    );
    // And dispatch count, on its own, is not a lever: the brief's own 0.7 us
    // figure puts the whole 241-dispatch GEMV teardown at 0.17 ms.
    assert!(241.0 * anchors::US_PER_DISPATCH_MANY_TG / 1e3 < 0.2);
}

// ---------------------------------------------------------------------------
// 7. The host gap has already been removed; say so, from the source
// ---------------------------------------------------------------------------

#[test]
fn the_decode_step_rolls_so_the_finding_04_host_gap_does_not_apply() {
    // `finding 04` attributed 5.75 ms of an 8.9 ms gap to a fully exposed host
    // encode, because `model.rs:3111` called `set_max_dispatches(2500)` against a
    // 1617-dispatch step so `tick()` never rolled and the whole step sat in one
    // uncommitted buffer. Its recommendation was to roll.
    assert_eq!(MAX_DISPATCHES_PER_BUFFER, 50);
    assert_eq!(decode_max_dispatches_per_buffer(), 50);
    assert!(
        !MODEL_RS.contains("set_max_dispatches(2500)"),
        "decode is back to a single uncommitted buffer per step; the host encode is exposed \
         again and this probe's (c) host-gap term is no longer zero"
    );
    assert!(
        MODEL_RS.contains("decode_max_dispatches_per_buffer()"),
        "the decode forward no longer sets its own rolling threshold"
    );
    // So the decomposition's host term is zero, and `encode` (~5.7 ms of engine
    // CPU at the older dispatch count) hides behind 62.3 ms of GPU work.
    // MAX_IN_FLIGHT = 8 at 50 dispatches per buffer is 400 dispatches of
    // lookahead, which is more than the whole step's ~665.
    assert_eq!(kaplumba_kernels::encoder::MAX_IN_FLIGHT, 8);
}

// ---------------------------------------------------------------------------
// 8. The profilers: why 108.6 ms of own_ms cannot be read as 63.1 ms of step
// ---------------------------------------------------------------------------

#[test]
fn the_profiler_overreport_cannot_be_per_buffer_overhead() {
    // §16.2's table, verbatim: (role, own_ms/step, MB/layer).
    let rows: [(&str, f64, f64); 5] = [
        ("gate_up_gelu", 43.90, 86.70),
        ("down_proj", 20.26, 43.35),
        ("o_proj", 12.35, 16.52),
        ("head/gemv", 4.12, 1057.0),
        ("qkv_proj", 14.43, 33.04),
    ];
    let sum: f64 = rows.iter().map(|r| r.1).sum();
    // The note's rows add to 95.06, not the 108.6 its own caption claims.
    assert!(
        (sum - 95.06).abs() < 0.01,
        "§16.2's rows now sum to {sum:.2} ms; the 108.6 in the caption was already wrong by \
         13.5 ms and the reconciliation has to start from the rows"
    );

    // Per-buffer overhead is the only mechanism the module itself offers
    // ("Every `profile.stamp` commits a command buffer"), and `finding 04`
    // measured it at 3.8 us. To turn 95.06 ms of own_ms into a 62.3 ms step it
    // would have to be:
    let required_us = (sum - anchors::MS_PER_STEP) * 1e3 / anchors::STAMPS_PER_STEP;
    assert!(
        required_us > 50.0,
        "the required per-stamp artefact is only {required_us:.1} us, within reach of the \
         3.8 us `finding 04` measured; re-derive the reconciliation before trusting this test"
    );
    let ratio = required_us / anchors::PROFILER_US_PER_BUFFER;
    assert!(
        ratio > 10.0,
        "per-buffer overhead is now only {ratio:.1}x the measured 3.8 us, so the \
         '546 stamps cannot be summed' claim has weakened"
    );

    // And the 1.71 scale factor is self-refuting, which is the sharper point.
    // The note calibrates the profiler on `head/gemv` ("1.057 GB / 4.12 ms =
    // 257 GB/s", arithmetic that checks) and from that calibration derives a
    // 1.71 factor. If 1.71 is right, the head's own_ms is 7.05 ms, not 4.12.
    let factor: f64 = 108.6 / 63.1;
    assert!((factor - 1.721).abs() < 0.005);
    assert!(
        (factor - 1.0).abs() > 0.5,
        "the global scale factor is now near 1.0, which would mean the head row and the \
         total are consistent; the argument that one global factor cannot fit both has expired"
    );
    // A single factor cannot be right when one of its inputs is used to define
    // itself: applying it moves the head from 257 GB/s to 149 GB/s, which
    // contradicts the 257 GB/s two paragraphs earlier in the same section - and
    // puts the 6 bpw head *below* the 3 bpw peak, which §16.4 says is
    // impossible for the reason the tile is bigger.
    let head_scaled_ms: f64 = 4.12 * factor;
    let head_gb_s: f64 = 1_057_499_652.0 / (head_scaled_ms * 1e-3) / 1e9;
    assert!(
        (head_gb_s - 149.0).abs() < 1.0,
        "applying 1.71 to the head row now gives {head_gb_s:.0} GB/s"
    );
    assert!(
        head_gb_s < anchors::PEAK_3BPW_GB_S,
        "a uniform 1.71 would put the 6 bpw head *below* the 3 bpw peak, which the note's own \
         table says is impossible; the factor cannot be uniform"
    );
}

#[test]
fn the_profiler_table_does_not_agree_with_itself() {
    // §16.2 prints an `own_ms/MB` column and an `implied GB/s` column. Two of
    // five rows are internally consistent; three are not, and the derived
    // bandwidths in §18.2 are read off the broken ones.
    let rows: [(&str, f64, f64, f64, f64); 4] = [
        // name, own_ms, MB/layer, notes' own_ms/MB, notes' implied GB/s
        ("gate_up_gelu", 43.90, 86.70, 0.4340, 204.0),
        ("down_proj", 20.26, 43.35, 0.4272, 200.0),
        ("o_proj", 12.35, 16.52, 0.8575, 100.0),
        ("qkv_proj", 14.43, 33.04, 0.4368, 196.0),
    ];
    let mut agree_ms_per_mb = 0;
    let mut agree_gb_s = 0;
    for (name, own_ms, mb, notes_ratio, notes_gb_s) in rows {
        let ratio = own_ms / mb;
        if (ratio - notes_ratio).abs() / notes_ratio < 0.01 {
            agree_ms_per_mb += 1;
        } else {
            eprintln!("{name}: own_ms/MB is {ratio:.4}, §16.2 prints {notes_ratio:.4}");
        }
        // Layers multiply by 60; the head does not.
        let gb_s = mb * 60.0 / 1000.0 / (own_ms * 1e-3);
        if (gb_s - notes_gb_s).abs() / notes_gb_s < 0.05 {
            agree_gb_s += 1;
        } else {
            eprintln!("{name}: implied GB/s is {gb_s:.0}, §16.2 prints {notes_gb_s:.0}");
        }
    }
    assert_eq!(
        agree_ms_per_mb, 1,
        "more of §16.2's own_ms/MB column now checks out; the table was edited and its \
         derived bandwidths need re-reading"
    );
    assert_eq!(
        agree_gb_s, 0,
        "some of §16.2's implied GB/s column now agrees with its own_ms; §18.2's table was \
         built on the disagreeing ones"
    );

    // The head is the only row whose derived bandwidth survives, because it is
    // the only row that was calibrated rather than derived.
    // 1057 MB / 4.12 ms = 256.6 GB/s, the arithmetic §16.2 calls "checks out".
    let head_gb_s: f64 = 1057.0 / 4.12;
    assert!((head_gb_s - 256.6).abs() < 0.5);
    // Which is why §18.2's head figure can be used and its other four cannot.
    // §18.2 reads 126/125 GB/s for gate_up/down/qkv off the deltas; from
    // §16.2's own_ms the same rows give 119/128/137 GB/s. Neither is 190.
    let derived: [(f64, f64, f64); 3] = [(43.90, 86.70, 118.5), (20.26, 43.35, 128.4), (14.43, 33.04, 137.4)];
    for (own_ms, mb, expected) in derived {
        let gb_s = mb * 60.0 / 1000.0 / (own_ms * 1e-3);
        assert!((gb_s - expected).abs() < 0.6);
    }
}

// ---------------------------------------------------------------------------
// 9. Context: the headline number is a 38-token number
// ---------------------------------------------------------------------------

#[test]
fn kv_traffic_is_negligible_only_at_the_benchmark_context() {
    let t = TokenBytes::from_checkpoint();
    let weights = t.all() as f64;

    // 6-token prompt + 32 generated. The sliding windows are barely started.
    let kv = kv_bytes_per_token(anchors::BENCHMARK_POSITION);
    assert_eq!(kv, 34_242_560);
    let share = kv as f64 / weights;
    assert!(
        share < 0.003,
        "KV is now {:.2}% of the step's bytes at position {}; the byte budget in this file \
         assumes it is noise",
        100.0 * share,
        anchors::BENCHMARK_POSITION
    );

    // At a real conversation length it is not noise, and nothing in the repo's
    // 16.0-17.1 tok/s range accounts for it.
    for (ctx, expected_share) in [(1024u64, 0.0765), (2048, 0.0835), (8192, 0.1253)] {
        let kv = kv_bytes_per_token(ctx) as f64;
        let got = kv / weights;
        assert!(
            (got - expected_share).abs() < 0.0015,
            "KV share at ctx {ctx} moved to {:.2}%",
            100.0 * got
        );
    }

    // Held at the same achieved bandwidth, a 2048-token context costs 8.35% more
    // than the benchmark does. So 16.05-16.10 tok/s is a short-context figure
    // and the achievable ceiling of 17.1 is too.
    let rate = (weights + kv_bytes_per_token(anchors::BENCHMARK_POSITION) as f64)
        / (anchors::MS_PER_STEP * 1e-3);
    let ms_2048 = (weights + kv_bytes_per_token(2048) as f64) / rate * 1e3;
    assert!(
        (ms_2048 - 67.5).abs() < 0.5,
        "a 2048-token context now costs {ms_2048:.1} ms/step"
    );
    assert!(ms_2048 > anchors::MS_PER_STEP * 1.07);
    assert!(
        ATTENTION_RS.contains("let groups = q_heads / kv_heads;"),
        "the decode attention grid changed; attention's cost is 2 threadgroups on the 50 \
         sliding layers and 8 on the 10 full ones, and that is why it measures as free"
    );
}

// ---------------------------------------------------------------------------
// 10. The decomposition, and the ceiling
// ---------------------------------------------------------------------------

#[test]
fn the_decomposition_closes_to_within_half_a_millisecond() {
    let t = TokenBytes::from_checkpoint();
    let (_, us) = fit_two_parameter(&SHAPES[4], &SHAPES[8]);

    // (a) streaming weights, at the 203 GB/s peak plus the fitted fixed cost.
    let stream_ms = t.layers as f64 / (anchors::PEAK_3BPW_GB_S * 1e9) * 1e3;
    assert!((stream_ms - 54.18).abs() < 0.05);
    let big_dispatches = 241.0;
    let constant_ms = big_dispatches * us / 1e3;
    assert!((constant_ms - 2.18).abs() < 0.02);

    // (b) ramp-up/drain per dispatch and (c) wave quantization: one number, and
    // it is the same as (a)'s second term. They are not separable from outside,
    // because both are a fixed cost per grid launch and the isolated sweep
    // measures them the same way. §16.3's 203 GB/s is a no-fixed-cost number,
    // which is why the two must be carried separately from it.
    let fixed_cost_ms = constant_ms;

    // (d) host gap: zero. The step rolls every 50 dispatches
    // (`decode_max_dispatches_per_buffer`), so the ~5.7 ms encode of
    // `finding 04` hides behind the GPU.
    let host_gap_ms = 0.0f64;

    // (e) tail: the head is measured on its own, 253 GB/s.
    let head_ms = t.head as f64 / (anchors::HEAD_GB_S * 1e9) * 1e3;
    assert!((head_ms - 4.18).abs() < 0.02);

    // Non-GEMV GPU time, one measured number covering every element kernel,
    // both GEMV epilogues, attention, softcap and argmax.
    let non_gemv_ms = anchors::NON_GEMV_MS;

    let total = stream_ms + fixed_cost_ms + host_gap_ms + head_ms + non_gemv_ms;
    let residual = anchors::MS_PER_STEP - total;
    assert!(
        residual.abs() < 0.5,
        "the decomposition no longer closes: {total:.2} ms accounted, {residual:+.2} ms \
         unexplained against a {} ms step",
        anchors::MS_PER_STEP
    );
    assert!(
        (residual / anchors::MS_PER_STEP).abs() < 0.01,
        "the unexplained term is now {:.1}% of the step",
        100.0 * residual / anchors::MS_PER_STEP
    );

    // The shape of it, which is the deliverable:
    let recoverable = |ms: f64| 1000.0 / (anchors::MS_PER_STEP - ms) - 1000.0 / anchors::MS_PER_STEP;
    let per_dispatch_gain = recoverable(constant_ms);
    let non_gemv_gain = recoverable(non_gemv_ms);
    assert!(
        (per_dispatch_gain - 0.56).abs() < 0.03,
        "removing every per-dispatch constant is now worth {per_dispatch_gain:.2} tok/s"
    );
    assert!(
        (non_gemv_gain - 0.39).abs() < 0.03,
        "removing all non-GEMV GPU work is now worth {non_gemv_gain:.2} tok/s"
    );

    // The ceiling: zero of both, at the 203 GB/s peak that is already measured.
    let ceiling_ms = stream_ms + head_ms;
    let ceiling_tok_s = 1000.0 / ceiling_ms;
    assert!(
        (ceiling_ms - 58.36).abs() < 0.05,
        "the today's-kernels ceiling moved to {ceiling_ms:.2} ms"
    );
    assert!(
        (ceiling_tok_s - 17.13).abs() < 0.05,
        "the ceiling moved to {ceiling_tok_s:.2} tok/s"
    );
    assert!(ceiling_tok_s > 1000.0 / anchors::MS_PER_STEP);
    // And the two numbers the brief asks about:
    let everything_at_wire = t.all() as f64 / (anchors::WIRE_GB_S * 1e9) * 1e3;
    assert!(
        (everything_at_wire - 44.16).abs() < 0.05,
        "the wire roof moved to {everything_at_wire:.2} ms"
    );
    let everything_at_head = t.all() as f64 / (anchors::HEAD_GB_S * 1e9) * 1e3;
    assert!(
        (everything_at_head - 47.65).abs() < 0.05,
        "the head-speed roof moved to {everything_at_head:.2} ms"
    );
    // 21.0 tok/s is not a latency problem and this test says why in one line:
    // it needs every 3 bpw byte to be decoded as cheaply as a 6 bpw one.
    let head_speed_tok_s = 1000.0 / everything_at_head;
    assert!((head_speed_tok_s - 20.99).abs() < 0.03);
    assert!(
        head_speed_tok_s > ceiling_tok_s * 1.2,
        "the head-speed roof and the achievable ceiling have converged; re-read §4, because \
         the two are no longer describing different regimes"
    );
}

// ---------------------------------------------------------------------------
// 11. The probe can fail. A model whose branches all agree is not a model.
// ---------------------------------------------------------------------------

#[test]
fn the_model_rejects_a_wrong_anchor() {
    // Sabotage check, per AGENTS.md §3.4: a coverage claim is not a claim until
    // the failing branch has been seen to fail. The branch that matters is the
    // per-dispatch constant: if it did not exist, the sweep would fit a single
    // bandwidth and the whole decomposition would collapse.
    //
    // Simulate "no per-dispatch cost" by fitting with c = 0 and confirm the
    // sweep says no.
    let (bytes_per_us, _) = fit_two_parameter(&SHAPES[4], &SHAPES[8]);
    let worst = max_time_residual(bytes_per_us, 0.0);
    assert!(
        worst > 0.10,
        "with c = 0 the model still fits the sweep to {:.1}%; the evidence for a per-dispatch \
         cost has gone and the decomposition below it is unsupported",
        100.0 * worst
    );

    // And simulate the brief's own hypothesis - a constant 7 us - and show it
    // does *not* reproduce the sweep.
    let worst_7us = max_time_residual(bytes_per_us, 7.0);
    assert!(
        worst_7us > 0.02,
        "a 7 us constant now fits the sweep to {:.1}%; if that is true this file's fitted \
         9.03 us is an overfit of two points and the whole decomposition must be redone",
        100.0 * worst_7us
    );

    // Finally: the decomposition must be sensitive to its own inputs. Move the
    // peak bandwidth 10% and the ceiling must move with it.
    let t = TokenBytes::from_checkpoint();
    let ceiling = |peak: f64| {
        (t.layers as f64 / (peak * 1e9) * 1e3) + t.head as f64 / (anchors::HEAD_GB_S * 1e9) * 1e3
    };
    // 10% more peak bandwidth must shorten the layer-stream term by ~10% of
    // 54.18 ms = 4.9 ms. Signed: a *faster* peak is a *shorter* step.
    let delta = ceiling(anchors::PEAK_3BPW_GB_S) - ceiling(anchors::PEAK_3BPW_GB_S * 1.1);
    assert!(
        delta > 4.0,
        "the ceiling barely moved ({delta:+.2} ms) for a 10% bandwidth change; this probe is \
         not actually wired to its anchors"
    );
}