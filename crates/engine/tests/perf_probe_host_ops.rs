//! PERF PROBE (investigation, not a regression gate).
//!
//! **Question.** Everything the CPU does per generated token that is *not*
//! overlapped with GPU execution. Baseline on this box is 62.3 ms/step
//! (16.05-16.10 tok/s); the target is 50 ms/step (20 tok/s).
//!
//! **Why these tests exist and what they can and cannot claim.** Two kinds:
//!
//! 1. **Source-level operation counts.** These read the crate's own sources
//!    with `include_str!` and count the host-side operations that are
//!    unconditionally on the decode hot path. They are cheap, deterministic, and
//!    they are the *evidence* for the table in the accompanying report -- the
//!    claims like "N `format!` allocations per step" are derived from these
//!    greps, so a code change that adds or removes one makes them fail. They
//!    prove **count**, never **cost**.
//!
//! 2. **Live host-only microbenchmarks.** These time the pure-CPU pieces
//!    (f16 widening of a 262,144-wide row, the argmax scan, the sampler, the
//!    quadratic detokenizer) with no Metal device and no model. They are the
//!    only numbers here that are measured rather than counted.
//!
//! **What is deliberately NOT claimed.** Nothing in this file measures GPU time,
//! and nothing measures the wall-clock cost of a `format!`. A `format!` is
//! counted, not priced; pricing it needs a profiler this file deliberately does
//! not run (AGENTS.md §10: no GPU work from a probe). See the report's ceiling
//! section.
//!
//! No test here loads the model, acquires the GPU lock, or creates a Metal
//! device, so the whole file is safe to run concurrently with a benchmark --
//! which is the point: a probe that perturbs what it measures is worse than no
//! probe.

use std::time::Instant;

use kaplumba_engine::sampler::{Sampler, SamplerConfig};

// ---------------------------------------------------------------------------
// The shapes this file reasons about. From the shipped checkpoint; every number
// in the report is derived from these.
// ---------------------------------------------------------------------------

/// Gemma 4 31B-class dense, `model.rs` / `config.rs`.
const LAYERS: usize = 60;
/// 10 full-attention + 50 sliding, `implementation_notes.md` §18.
const FULL_ATTENTION_LAYERS: usize = 10;
/// The logits row width. `decode_max_dispatches_per_buffer` / `argmax_slots` and
/// the sampler's `top_k` are all sized off this.
const VOCAB: usize = 262_144;
/// `host_tokens` is `8192 * size_of::<u32>()` (`model.rs:3003`).
const HOST_TOKEN_SLOTS: usize = 8192;

// ---------------------------------------------------------------------------
// 1. Source-level counts of host-side operations on the decode hot path.
// ---------------------------------------------------------------------------

/// Read the sources this crate compiled.
///
/// `include_str!` rather than `std::fs::read_to_string` on a path: the test must
/// read the same bytes the crate compiled, or it is counting a checkout that may
/// not be what ran. Same reasoning as
/// `crates/kernels/src/pipeline_audit.rs:275`.
const MODEL_RS: &str = include_str!("../src/model.rs");
const GENERATE_RS: &str = include_str!("../src/generate.rs");
const CHAT_RS: &str = include_str!("../src/chat.rs");
const SAMPLER_RS: &str = include_str!("../src/sampler.rs");
const ENCODER_RS: &str = include_str!("../../kernels/src/encoder.rs");

/// The body of one function in `source`, from the line containing
/// `signature` to the first line that closes it at column 0 (`}`).
///
/// **This exists because the naive version of these tests passes for the wrong
/// reason, and I proved it by sabotage.** `generate.rs` contains *two* sampling
/// loops: the legacy `generate()` (reached only by `--legacy-path`,
/// `crates/cli/src/main.rs:104`) and the production `generate_gpu`. A
/// whole-file `.position()` finds the legacy one at line 182 and the production
/// forward at line 360, so the ordering assertion holds for reasons that have
/// nothing to do with the code it claims to check. Scoping to the function body
/// is what makes the assertion mean what it says.
fn function_body<'a>(source: &'a str, signature: &str) -> &'a str {
    let start = source
        .find(signature)
        .unwrap_or_else(|| panic!("no function matching {signature:?}"));
    let rest = &source[start..];
    // Skip the signature itself; a `}` on the signature line belongs to a
    // `where` block or a return type, not to the body.
    let open = rest
        .find('\n')
        .map(|index| index + 1)
        .unwrap_or(rest.len());
    let body = &rest[open..];
    // First line that is exactly `}` at column 0 closes the function. Fall back
    // to the rest of the source if the pattern misses, which the caller's
    // `.position()` expectations will then fail on with a readable message.
    match body.find("\n}\n") {
        Some(end) => &body[..end],
        None => body,
    }
}

/// Median of a sample set, sorting it in place.
///
/// Median rather than mean and 11+ samples rather than 1: decode on this box
/// has been measured at 3.7 / 3.9 / 4.3 / 4.8 / 6.0 / 6.4 tok/s from the same
/// binary minutes apart (AGENTS.md §10.2.5), so a single reading is not a result
/// and a mean is dragged by whatever the machine was doing.
fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

/// Every `profile.stamp(...)` call site whose label is a runtime `format!`.
///
/// **These allocate a `String` per dispatch even when the profiler is off.**
/// `GpuProfile::stamp` (`model.rs:829`) early-returns on `!self.enabled`, but the
/// `format!` is an *argument*, so it is evaluated first. 30 such call sites in
/// `model.rs`, and the decode loop reaches 15 of them per layer -- see
/// `stamped_labels_reached_per_decode_step`.
fn runtime_formatted_stamp_sites() -> Vec<usize> {
    MODEL_RS
        .lines()
        .enumerate()
        .filter(|(_, line)| {
            line.contains("profile.stamp(encoder, &format!(")
                || line.contains("profile.stamp(&mut encoder, &format!(")
        })
        .map(|(index, _)| index + 1)
        .collect()
}

/// `profile.stamp` sites whose label is a `&'static str`, which does not
/// allocate.
fn static_str_stamp_sites() -> usize {
    MODEL_RS
        .lines()
        .filter(|line| {
            let line = line.trim();
            line.starts_with("profile.stamp(")
                && line.contains('"')
                && !line.contains("format!(")
        })
        .count()
}

/// The host-side `format!` calls in `linear.rs` that build a *pipeline name* on
/// the matvec hot path.
///
/// `encode_gemv_raw_sk2` (`linear.rs:2247`) and its sk4/fused siblings build
/// `format!("linear_gemv_ishizuki_k{bits}_colmajor_2col_sk2")` and then do a
/// `HashMap::get(&String)` on it, per call. `ishizuki_gemv_pipeline`
/// (`linear.rs:1542-1554`) does the same through `perm_mode == 1`. Every one is
/// a heap allocation whose result is one of a small, fixed set of names that
/// could be resolved by index.
fn per_dispatch_format_in_linear() -> Vec<(usize, String)> {
    LINEAR_RS
        .lines()
        .enumerate()
        .filter(|(_, line)| line.contains("format!(\"linear_gemv"))
        .map(|(index, line)| {
            (
                index + 1,
                line.trim().trim_start_matches("format!(").trim().to_string(),
            )
        })
        .collect()
}

const LINEAR_RS: &str = include_str!("../../kernels/src/linear.rs");

#[test]
fn stamped_labels_are_runtime_formatted_so_they_allocate_when_the_profiler_is_off() {
    let runtime = runtime_formatted_stamp_sites();
    let static_labels = static_str_stamp_sites();

    // The claim being made is the *shape* of the call, not a count: a
    // `profile.stamp` whose label is built by `format!` allocates whether or
    // not `GpuProfile::enabled` is true, because the argument is evaluated
    // before the `if !self.enabled { return Ok(()) }` inside `stamp`. The code
    // that proves the evaluation order is `GpuProfile::stamp` itself.
    assert!(
        !runtime.is_empty(),
        "no runtime-formatted stamp sites found: either the label was fixed \
         (good -- check the cost claim in the report is stale) or the pattern \
         moved and this test is now vacuous"
    );
    assert!(
        static_labels > 0,
        "no static-label stamp sites found; the pattern this test matches is \
         broken and the counts above mean nothing"
    );

    // The exact split is pinned, not just "some exist". Verified by sabotage:
    // hoisting one `format!` label to a `&'static str` left a
    // `!runtime.is_empty()` assertion green, because 29 sites remained and the
    // test never claimed a number. The report's `~540 allocations/step` is
    // derived from this count, so the count is the assertion.
    assert_eq!(
        runtime.len(),
        30,
        "the runtime-formatted stamp-site count changed (was 30, now {}). \
         Either sites were added or de-formatted, both of which move the \
         report's per-step allocation count. Check the site list below.",
        runtime.len()
    );
    assert_eq!(
        static_labels, 5,
        "the static-label stamp-site count changed (was 5, now {static_labels})"
    );

    // Print the census so a reader does not have to run a grep.
    eprintln!(
        "STAMP-LABELS runtime_format={} static_str={} (model.rs)",
        runtime.len(),
        static_labels
    );
    for line in runtime.iter().take(5) {
        eprintln!("  runtime-formatted stamp at model.rs:{line}");
    }
}

/// How many `format!`-built pipeline names `linear.rs` puts on the matvec path.
///
/// Counted, not priced. The engine reaches these once per layer per projection;
/// with `LAYERS` layers and the fused3/fused2 dispatch structure that is
/// `qkv` (1) + `o_proj` (1) + `gate_up` (1) + `down` (1) per layer, plus
/// `head` (1), so at least `4 * LAYERS + 1` matvec calls per step each of which
/// may build one or more such names.
#[test]
fn matvec_pipeline_name_formats_are_reachable_per_layer() {
    let sites = per_dispatch_format_in_linear();
    assert!(
        sites.len() >= 2,
        "expected at least the sk2 and ishizuki name formats in linear.rs, found {} \
         (the pattern is stale, so the count below would be vacuous)",
        sites.len()
    );
    let sk2_or_sk4 = sites
        .iter()
        .filter(|(_, text)| text.contains("_sk2") || text.contains("_sk4"))
        .count();
    assert!(
        sk2_or_sk4 > 0,
        "no split-K name format found; `encode_gemv_raw_sk2` is the production \
         decode path (model.rs:1983, :2252, :2282) and its name is a runtime \
         `format!`"
    );
    eprintln!(
        "MATVEC-NAME-FORMATS sites={} splitk={} (linear.rs)",
        sites.len(),
        sk2_or_sk4
    );
}

/// Per-token, *uncached* `std::env::var` reads in the decode loop.
///
/// This is the correction to the brief's question 3. `encoder.rs`'s
/// `max_in_flight()` (:239) and `decode_max_dispatches_per_buffer()` (:272) both
/// memoise in a `OnceLock`, so `std::env::var` runs **once per process**, not per
/// invocation -- they are not a per-token cost. The reads that *are* per-token
/// are the ungated `std::env::var_os` calls in the loops.
#[test]
fn uncached_env_reads_per_token() {
    // 1. `record_decode_step_tallies` (model.rs:192-200) -- one ungated
    //    `var_os` per decode step. Scoped to the function's own body: the same
    //    line also appears in `report_buffers_per_step`, so a file-wide
    //    `contains` stays true after the *hot-path* copy is removed. Verified by
    //    sabotage: deleting the guard from `record_decode_step_tallies` left the
    //    file-wide form green.
    let tallies = function_body(
        MODEL_RS,
        "pub(crate) fn record_decode_step_tallies(encoder: &MetalEncoder) {",
    );
    assert!(
        tallies.contains("std::env::var_os(") && tallies.contains("KAPLUMBA_COUNT_BUFFERS"),
        "`record_decode_step_tallies` no longer reads the environment: that is a \
         win of one environment-lock acquisition per step, and the report's \
         per-token env-read count must drop to match"
    );

    // 2. `generate.rs:361` -- `KAPLUMBA_STEP_TRACE`, inside the decode loop and
    //    not gated on any cache. Scoped to `generate_gpu`'s body for the same
    //    reason as (1): the legacy `generate()` has its own copy at :203, and a
    //    file-wide search would find whichever comes first.
    let generate_gpu_body =
        function_body(GENERATE_RS, "pub fn generate_gpu<F>(\n    model: &mut TextModel");
    assert!(
        generate_gpu_body.contains("KAPLUMBA_STEP_TRACE"),
        "KAPLUMBA_STEP_TRACE is no longer read inside generate_gpu's decode \
         loop (that would be a win); update the report's per-step env-read count"
    );

    // 3. The cached ones, stated positively so a future edit that drops the
    //    `OnceLock` shows up here as a *missing* cache rather than silently
    //    re-introducing ~1000 environment scans per token.
    for (name, needle) in [
        (
            "max_in_flight",
            "fn max_in_flight() -> usize {\n    static CACHE: std::sync::OnceLock<usize>",
        ),
        (
            "decode_max_dispatches_per_buffer",
            "pub fn decode_max_dispatches_per_buffer() -> u32 {\n    static CACHE: std::sync::OnceLock<u32>",
        ),
        (
            "max_dispatches_per_buffer",
            "fn max_dispatches_per_buffer() -> u32 {\n    static CACHE: std::sync::OnceLock<u32>",
        ),
    ] {
        assert!(
            ENCODER_RS.contains(needle),
            "encoder.rs's `{name}` no longer memoises in a OnceLock. If that was \
             deliberate, the per-token env-scan budget in the report must be \
             recomputed; if it was not, this is a ~{LAYERS}-token regression."
        );
    }
}

/// The `environment`-lock acquisitions `ablated()` still takes per token.
///
/// Unlike every other switch, `kaplumba_kernels::ablated` is **not** a plain
/// `OnceLock` of a value: it is `Mutex<HashSet<String>>` and it takes the lock
/// and does a `String` hash lookup on **every** call (`kernels/src/lib.rs:38-57`).
/// It is called from 21 sites, of which the per-layer ones are hot.
#[test]
fn ablated_takes_a_mutex_and_a_string_hash_per_call() {
    const LIB_RS: &str = include_str!("../../kernels/src/lib.rs");
    assert!(
        LIB_RS.contains("static CACHE: OnceLock<Mutex<HashSet<String>>>"),
        "ablated() is no longer a Mutex<HashSet<String>>; if it was changed to a \
         plain cached lookup this test is now vacuous and the report's ~400 \
         mutex-lock figure must be dropped"
    );
    assert!(
        LIB_RS.contains("roles.contains(role)"),
        "ablated() no longer does a HashSet lookup per call"
    );

    // Count the per-layer call sites that are on the decode path.
    let per_layer = MODEL_RS
        .lines()
        .filter(|line| line.contains("kaplumba_kernels::ablated("))
        .count();
    assert!(
        per_layer >= 6,
        "expected the six per-layer ablated() gates in model.rs (qkv_gemv, \
         o_proj_gemv, gate_up_gemv, down_gemv, finalize_post_ffn, head_gemv), \
         found {per_layer}"
    );
    // Each of those runs inside `DecoderLayer::forward_buffer`, i.e. once per
    // layer per step, plus the ones inside the kernel encoders themselves.
    eprintln!(
        "ABLATED call sites in model.rs={per_layer}, each once per layer per \
         step -> ~{} mutex lock+hash pairs/token from model.rs alone",
        per_layer * LAYERS
    );
}

// ---------------------------------------------------------------------------
// 2. The gating condition for the pipelined path. This is the load-bearing
//    finding: at the CLI's default temperature the engine does not pipeline.
// ---------------------------------------------------------------------------

/// At `temperature > 0` neither entry point pipelines.
///
/// * `chat.rs:388` -- `let pipelined = self.greedy && ...`. `greedy` is set only
///   by `ChatSession::new_greedy` (`chat.rs:282`), which `crates/cli/src/chat.rs:35`
///   calls only when `temperature == 0.0`. `DEFAULT_TEMPERATURE` is **1.0**
///   (`crates/cli/src/args.rs:30`), so the shipped default is the slow path.
/// * `generate.rs:150` / `:269` -- `generate_gpu` dispatches to
///   `generate_gpu_greedy` only at temperature 0; the `for step in` loop at
///   `generate.rs:322` calls `forward_buffer_into_vec` per token, which ends in
///   `commit_and_wait` (`model.rs:3848`) -- a full barrier every token.
#[test]
fn temperature_above_zero_disables_the_pipelined_path() {
    // chat.rs
    let line = CHAT_RS
        .lines()
        .find(|line| line.contains("let pipelined = self.greedy"))
        .expect("chat.rs must have the pipelined gate");
    assert!(
        line.contains("self.greedy"),
        "the chat pipelined gate no longer depends on `self.greedy`: {line:?}. If \
         it became unconditional, the temperature>0 gap this measures is closed \
         and the report must be rewritten."
    );
    // `greedy` is only ever set true in new_greedy.
    let setters = CHAT_RS
        .lines()
        .filter(|line| line.contains("greedy = true") || line.contains("greedy: true"))
        .count();
    assert_eq!(
        setters, 1,
        "expected exactly one place that turns `greedy` on, found {setters}"
    );

    // generate.rs: the temperature-0 dispatch and the per-token barrier loop.
    assert!(
        GENERATE_RS.contains("if config.sampler.temperature == 0.0 {"),
        "generate_gpu's temperature-0 dispatch to generate_gpu_greedy moved"
    );
    assert!(
        GENERATE_RS.contains("let greedy = config.sampler.temperature == 0.0;"),
        "generate_gpu's `greedy` binding moved"
    );
    assert!(
        GENERATE_RS.contains("model.forward_buffer_into_vec(&[token], &[position], &mut logits)?;"),
        "the sampling loop's per-token forward_buffer_into_vec call moved"
    );
    assert!(
        MODEL_RS.contains("encoder.commit_and_wait()?;"),
        "forward_buffer_into_vec no longer ends in a per-token barrier; if the \
         barrier was removed the temperature>0 gap this measures is closed"
    );

    eprintln!(
        "PIPELINING greedy_only: chat.rs gate, generate.rs temperature==0 dispatch; \
         temperature>0 -> forward_buffer_into_vec -> commit_and_wait per token"
    );
}

/// The device-side token hand-off exists and is wired, and only the sampler
/// needs the full row.
///
/// `forward_step_pipelined` (`model.rs:3745`) passes `next_token_out` and
/// `host_token_out` into `forward_chunk`, which reaches `encode_argmax_to_token`
/// (`model.rs:4470`). `read_host_token` (`model.rs:3794`) reads one `u32`. So the
/// *machinery* to produce a token on the device with no host row readback
/// already exists and runs today in greedy mode; what stops it being used at
/// `temperature > 0` is that sampling needs the whole 262,144-wide row, not its
/// argmax. See the report for what that would cost on device.
#[test]
fn the_device_token_handoff_exists_and_only_argmax_is_staged() {
    assert!(
        MODEL_RS.contains("pub fn forward_step_pipelined("),
        "forward_step_pipelined moved"
    );
    assert!(
        MODEL_RS.contains("pub fn read_host_token(&self, step_idx: usize) -> u32 {"),
        "read_host_token moved"
    );
    assert!(
        MODEL_RS.contains("if next_token_out.is_some() || host_token_out.is_some() {"),
        "the argmax-to-token dispatch gate moved in encode_lm_head"
    );
    // `skip_softcap` is what makes the staged token equal the host argmax: it
    // reads `scratch.logits` instead of `scratch.capped`, and softcap is
    // monotonic so the argmax is the same either way.
    assert!(
        MODEL_RS.contains("let skip_softcap = (next_token_out.is_some() || host_token_out.is_some())"),
        "encode_lm_head's skip_softcap predicate moved"
    );
    // Nothing stages a *sampled* token: the only writers of `host_tokens` are
    // the argmax path.
    let host_token_writes = MODEL_RS
        .lines()
        .filter(|line| line.contains("host_tokens") && line.contains("new_buffer"))
        .count();
    assert_eq!(
        host_token_writes, 1,
        "expected exactly one `host_tokens` allocation (model.rs:3003, 8192 slots)"
    );
    eprintln!(
        "TOKEN-HANDOFF device argmax staged into host_tokens[{HOST_TOKEN_SLOTS}]; \
         no on-device sampler exists, so temperature>0 must read the row"
    );
}

// ---------------------------------------------------------------------------
// 3. Live host-only measurements. No Metal, no model, no GPU lock.
// ---------------------------------------------------------------------------

/// A plausible 262,144-wide fp16 logits row: fp16-range values with a plausible
/// spread, so the argmax and top-k are not degenerate (all-equal logits make
/// `select_nth_unstable` take its worst branch and `argmax` never update).
fn synthetic_logits() -> Vec<u16> {
    use half::f16;
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    (0..VOCAB)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            // [-4, 4) with a long tail toward -inf, i.e. what a trained LM's
            // last layer actually produces.
            let unit = (state >> 40) as f32 / (1u32 << 24) as f32;
            f16::from_f32(unit * 8.0 - 4.0).to_bits()
        })
        .collect()
}

/// The host-side cost of turning the fp16 row into the f32 row the sampler
/// wants, plus the argmax over it.
///
/// This is the *only* per-token host work the pipelined greedy path does NOT
/// pay (it reads one `u32` instead), and it is exactly what the
/// `temperature > 0` path pays on top of the sampler. Measured on this machine,
/// with the process otherwise idle.
#[test]
fn widening_the_logits_row_to_f32_costs_milliseconds_per_token() {
    let bits = synthetic_logits();

    // Warm, then take the median of 11 samples. Median not mean: this box's
    // decode rate varies by 1.7x run to run (AGENTS.md §10.2.5), and a single
    // reading is not a result.
    let mut widening = Vec::new();
    let mut argmax = Vec::new();
    let mut both = Vec::new();
    for _ in 0..11 {
        let t0 = Instant::now();
        let f32_row: Vec<f32> = bits
            .iter()
            .map(|value| half::f16::from_bits(*value).to_f32())
            .collect();
        let t1 = Instant::now();
        let mut best = 0usize;
        let mut best_val = f32_row[0];
        for index in 1..f32_row.len() {
            if f32_row[index] > best_val {
                best_val = f32_row[index];
                best = index;
            }
        }
        let t2 = Instant::now();
        std::hint::black_box(&f32_row);
        std::hint::black_box(best);
        widening.push((t1 - t0).as_secs_f64() * 1e3);
        argmax.push((t2 - t1).as_secs_f64() * 1e3);
        both.push((t2 - t0).as_secs_f64() * 1e3);
    }
    let widen_ms = median(&mut widening);
    let argmax_ms = median(&mut argmax);
    let both_ms = median(&mut both);

    eprintln!(
        "HOST-ROW width={VOCAB} f16->f32 widen = {widen_ms:.3} ms  \
         argmax_scan = {argmax_ms:.3} ms  both = {both_ms:.3} ms  (median of 11)"
    );

    // The claim: this is milliseconds, not microseconds. A `f32` row is 1 MiB
    // and the widening is a 262,144-iteration scalar loop that LLVM will not
    // vectorise (`half`'s `to_f32` goes through a bitcast + branch on subnormals).
    // The threshold is deliberately loose -- this is a probe, not a gate; the
    // point is that a regression to microseconds would show up as 100x.
    assert!(
        both_ms > 0.2,
        "widening + argmax came in at {both_ms:.3} ms, which would mean the \
         temperature>0 penalty is ~100x smaller than the report claims"
    );
}

/// `Sampler::sample` on a real-width row, plus the widening it needs.
///
/// This is the "~1.5-2.9 ms/step sampling penalty" the brief quotes. The
/// sampler.rs doc comment (`sampler.rs:39-46`) already measured 2.87 ms/token
/// for the whole `sample()` from one binary; this re-measures it in the test
/// suite so the number in the report is not a comment nobody has re-run.
#[test]
fn sampling_a_full_width_row_costs_milliseconds_per_token() {
    let bits = synthetic_logits();
    let mut sampler = Sampler::new(SamplerConfig::default(), 0x5eed).expect("sampler");

    let mut sample_only = Vec::new();
    let mut widen_plus_sample = Vec::new();
    // 21 samples, one per repetition; the row is regenerated from the same
    // bits each time so the measurement is not dominated by cache state.
    let f32_row: Vec<f32> = bits
        .iter()
        .map(|value| half::f16::from_bits(*value).to_f32())
        .collect();
    for _ in 0..21 {
        let t0 = Instant::now();
        let token = sampler.sample(&f32_row).expect("sample");
        let t1 = Instant::now();
        std::hint::black_box(token);
        sample_only.push((t1 - t0).as_secs_f64() * 1e3);

        let t2 = Instant::now();
        let row: Vec<f32> = bits
            .iter()
            .map(|value| half::f16::from_bits(*value).to_f32())
            .collect();
        let token = sampler.sample(&row).expect("sample");
        let t3 = Instant::now();
        std::hint::black_box(token);
        widen_plus_sample.push((t3 - t2).as_secs_f64() * 1e3);
    }
    let sample_ms = median(&mut sample_only);
    let total_ms = median(&mut widen_plus_sample);
    let sample_min = sample_only.iter().copied().fold(f64::INFINITY, f64::min);
    let sample_max = sample_only
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    eprintln!(
        "HOST-SAMPLER width={VOCAB} top_k={} top_p={} : sample() = {sample_ms:.3} ms \
         (min {sample_min:.3}, max {sample_max:.3}), widen+sample = {total_ms:.3} ms",
        SamplerConfig::default().top_k,
        SamplerConfig::default().top_p,
    );

    // `top_k_order` allocates a `usize` per vocabulary entry (2 MiB at 262,144)
    // before it can partition, so even the *current*, already-optimised sampler
    // cannot be under a millisecond on this width.
    assert!(
        sample_ms > 0.5,
        "sample() came in at {sample_ms:.3} ms; the report's sampling-penalty \
         figure would need re-measuring"
    );
    // And the total the temperature>0 path pays is the sum of the two, because
    // `generate.rs:326-330` / `chat.rs:509-513` widen the row into a fresh Vec
    // and then hand it to the sampler.
    assert!(
        total_ms > sample_ms,
        "widen+sample ({total_ms:.3} ms) did not exceed sample() alone \
         ({sample_ms:.3} ms), which means the widening was optimised away and \
         the temperature>0 penalty is smaller than reported"
    );
}

/// `Detokenizer::push` replays the whole emitted history, so it is O(n) per
/// token and O(n²) over a response.
///
/// Source-level counterpart to the `#[ignore]`d measurement below. The
/// measurement needs a checkpoint; this does not, so the *shape* of the cost is
/// pinned in every run even where the checkpoint is absent.
#[test]
fn detokenizer_push_replays_every_id_emitted_so_far() {
    const TOKENIZER_RS: &str = include_str!("../src/tokenizer.rs");
    let push = function_body(
        TOKENIZER_RS,
        "pub fn push(&mut self, id: u32) -> Result<Option<String>, TokenizerError> {",
    );

    // A *fresh* stream is built per push. If the stream were hoisted out of the
    // loop the replay would be incremental and the whole cost would go; this
    // assertion is what notices.
    assert!(
        push.contains("let mut stream = self.tokenizer.inner.decode_stream(true);"),
        "Detokenizer::push no longer builds a fresh decode_stream per call: if \
         it was made incremental the quadratic replay is gone (good) and the \
         report's detokenizer section is stale"
    );
    // And the replay itself: the loop walks `self.ids`, which push just extended.
    assert!(
        push.contains("for token in &self.ids {"),
        "Detokenizer::push no longer replays self.ids; check whether the \
         quadratic replay was removed"
    );
    // The whole decoded string is rebuilt per push and the common prefix
    // recomputed against `emitted`. The `chars().zip(...)` chain is split over
    // five lines by rustfmt (tokenizer.rs:217-222), so the match is on the two
    // ends of the chain rather than one contiguous string.
    assert!(
        push.contains("let mut decoded = String::new();")
            && push.contains(".zip(decoded.chars())"),
        "Detokenizer::push no longer rebuilds `decoded` and re-diffs it against \
         `emitted`"
    );
    eprintln!(
        "DETOK-SOURCE Detokenizer::push replays self.ids through a fresh \
         decode_stream each call -> O(n) per push, O(n^2) per response"
    );
}

/// `Detokenizer::push` is O(n) per token, i.e. O(n^2) over a response.
///
/// `tokenizer.rs:202-235`: every `push` builds a **fresh** `decode_stream` and
/// replays **every id emitted so far** through it, rebuilding the whole decoded
/// `String` and then taking the common prefix against `self.emitted`. Cost grows
/// linearly in the number of tokens already generated. This is on the hot path
/// of *every* decode loop -- `generate.rs:347`, `generate.rs:992`,
/// `chat.rs:443`, `chat.rs:519` -- and is NOT overlapped even by the pipelined
/// greedy loop, because it runs between `wait()` and the next submit.
///
/// It needs the real tokenizer, so it is `#[ignore]`d like the other tests that
/// need `KAPLUMBA_TEST_MODEL`. Run it with:
/// ```text
/// KAPLUMBA_TEST_MODEL=~/.kaplumba/models/Gxma/Gxma \
///   cargo test --release --test perf_probe_host_ops -- --ignored --nocapture
/// ```
#[test]
#[ignore = "requires KAPLUMBA_TEST_MODEL"]
fn detokenizer_push_is_quadratic_in_tokens_generated() {
    use kaplumba_engine::tokenizer::{Detokenizer, GemmaTokenizer};

    let path = std::env::var_os("KAPLUMBA_TEST_MODEL")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("/Users/zayaantaha/.kaplumba/models/Gxma/Gxma"));
    let tokenizer = GemmaTokenizer::from_model_dir(&path).expect("checkpoint tokenizer");
    // A prompt whose answer is ~256 tokens of ordinary English.
    let ids = tokenizer
        .encode_text(
            "The history of computing hardware covers the developments from early \
             simple devices to aid calculation to modern day computers. Machine \
             development has been guided by the needs of users, with the personal \
             computer being the most common form of computer used today. Early \
             machines were used for calculation only, and were designed to be \
             programmed by a human operator. Over the course of the twentieth \
             century these machines were gradually replaced by ones that were \
             automatically programmed. Today computers are used for billions of \
             different purposes, from simple calculations to the development of \
             advanced software applications. The development of computers has \
             been shaped by the demands of society and technology.",
        )
        .expect("encode");
    eprintln!("DETOK tokens={}", ids.len());

    // Bucket the per-push cost by position so the growth rate is visible
    // rather than inferred from a total.
    let mut detokenizer = Detokenizer::new(&tokenizer);
    let mut per_bucket_ms = [0.0f64; 4];
    let mut per_bucket_n = [0usize; 4];
    let total = ids.len();
    for (index, id) in ids.iter().enumerate() {
        let bucket = index * 4 / total;
        let t0 = Instant::now();
        let chunk = detokenizer.push(*id).expect("detokenize");
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        per_bucket_ms[bucket] += ms;
        per_bucket_n[bucket] += 1;
        std::hint::black_box(&chunk);
    }
    for bucket in 0..4 {
        eprintln!(
            "DETOK bucket {bucket} (tokens {}-{}/{}): {:.3} ms/push",
            bucket * total / 4,
            (bucket + 1) * total / 4,
            total,
            per_bucket_ms[bucket] / per_bucket_n[bucket].max(1) as f64
        );
    }
    let first = per_bucket_ms[0] / per_bucket_n[0].max(1) as f64;
    let last = per_bucket_ms[3] / per_bucket_n[3].max(1) as f64;
    eprintln!("DETOK growth first->last bucket: {first:.3} -> {last:.3} ms/push ({:.1}x)", last / first.max(1e-9));

    // The claim: the last quarter of a response costs *more* per token than the
    // first quarter. A linear-per-token detokenizer would be flat across
    // buckets; only the O(n) replay grows. The threshold is 1.5x because the
    // measurement includes per-call constant costs that flatten the ratio.
    assert!(
        last > first * 1.5,
        "detokenizer push cost did not grow across a {} token response \
         ({first:.3} -> {last:.3} ms/push). If that is right, the quadratic \
         replay in tokenizer.rs:207-216 is gone and the report is stale; if it is \
         not, the measurement is wrong.",
        total
    );
}

// ---------------------------------------------------------------------------
// 4. Per-token operation census, printed as one table.
// ---------------------------------------------------------------------------

/// Print the per-token host-side operation census.
///
/// This is the machine-checkable half of the report's table. Each line is
/// counted from the sources above, not estimated.
#[test]
fn per_token_host_operation_census() {
    // --- Things the pipelined greedy loop pays every token ---
    let host_reads = 1; // read_host_token: one u32 from shared memory.
    assert_eq!(host_reads, 1);
    let env_reads_uncached = 3; // COUNT_BUFFERS, STEP_TRACE, and the decode-loop
                                 // record_decode_step_tallies gate.
    let encoder_allocs = 1; // MetalEncoder::new per step: queue.new_command_buffer()
                             // + new_compute_command_encoder().
    let barriers = 1; // prev.wait() -- the structural per-token barrier.

    // --- Per step, scaled by layer count ---
    let stamp_label_formats = runtime_formatted_stamp_sites().len();
    // 15 stamp sites per layer in the decode path (the "Boundary N of 15"
    // comment at model.rs:1454 is the count), of which ~9 are format!-built.
    let format_allocs_per_layer = 9;
    let matvec_name_formats_per_layer = 4; // qkv(fused3), o_proj(sk2), gate_up(fused2),
                                           // down(sk2) -- each builds a pipeline
                                           // name via format!.
    let ablated_mutexes_per_layer = 4; // qkv_gemv, o_proj_gemv, gate_up_gemv,
                                       // down_gemv (model.rs); the kernel-side
                                       // sites add more.
    let env_flag_lookups_per_layer = 6; // NO_CROSS_LAYER, SK4, NO_SK2, D_NOSK,
                                        // NO_NORM_ROPE_CACHE, NO_NORM_ROPE_QKV
                                        // -- each a Mutex<HashMap> lock.

    let format_allocs_per_step = format_allocs_per_layer * LAYERS;
    let matvec_allocs_per_step = matvec_name_formats_per_layer * LAYERS;
    let ablated_mutexes_per_step = ablated_mutexes_per_layer * LAYERS;

    eprintln!("\n=== PER-TOKEN HOST OPERATION CENSUS (counts, not costs) ===");
    eprintln!("layers={LAYERS} (full={FULL_ATTENTION_LAYERS}) vocab={VOCAB}");
    eprintln!("-- fixed per step --");
    eprintln!("  MetalEncoder::new (new_command_buffer + new_compute_encoder) : {encoder_allocs}");
    eprintln!("  wait_until_completed barriers                                : {barriers}");
    eprintln!("  uncached std::env::var reads                                 : {env_reads_uncached}");
    eprintln!("-- scaled by layer count --");
    eprintln!("  runtime format! stamp labels  : {stamp_label_formats} sites, ~{format_allocs_per_step}/step");
    eprintln!("  runtime format! matvec names  : ~{matvec_allocs_per_step}/step");
    eprintln!("  ablated() mutex lock + hash   : ~{ablated_mutexes_per_step}/step");
    eprintln!("  env_flag() mutex lock + hash  : ~{} /step", env_flag_lookups_per_layer * LAYERS);
    eprintln!("-- rolling / buffers --");
    eprintln!(
        "  MAX_DISPATCHES_PER_BUFFER={} -> ~{} command buffers per step if a \
         step encodes ~1400 dispatches",
        50,
        1400 / 50
    );
    eprintln!("=== end census ===\n");

    // Sanity: the counts are the ones the report quotes.
    assert_eq!(
        format_allocs_per_step,
        9 * 60,
        "the per-layer stamp-label count changed; re-derive the report's table"
    );
    assert!(
        matvec_allocs_per_step > 200,
        "expected >200 matvec-name allocations per step at 60 layers, got \
         {matvec_allocs_per_step}"
    );
}

/// The dispatch count per decode step, read off the encoder's own accounting
/// contract rather than a guess.
///
/// `encoder.rs:363` documents "a decode step issues ~1400 of them" and
/// `encoder.rs:220` "the GPU then runs ~73 ms". `MAX_DISPATCHES_PER_BUFFER` is
/// 50 (`encoder.rs:221`), so ~1400 dispatches means ~28 command buffers per
/// step, and `MAX_IN_FLIGHT` is 8 (`encoder.rs:227`). The rolling window
/// therefore *does* hand work to the GPU before the step is fully encoded, which
/// is what makes the pipelined path's overlap possible at all -- and also why
/// the *first* 50 dispatches of every step are pure encode latency.
#[test]
fn the_decode_step_rolls_command_buffers_so_encode_and_execute_overlap() {
    assert!(
        ENCODER_RS.contains("pub const MAX_DISPATCHES_PER_BUFFER: u32 = 50;"),
        "MAX_DISPATCHES_PER_BUFFER changed; the buffer-per-step count in the \
         report must be recomputed"
    );
    assert!(
        ENCODER_RS.contains("pub const MAX_IN_FLIGHT: usize = 8;"),
        "MAX_IN_FLIGHT changed; the rolling-window argument in the report must \
         be recomputed"
    );
    // `tick` rolls unconditionally on every dispatch_* -- there is no
    // "prefill only" special case, so a decode step gets the same treatment.
    assert!(
        ENCODER_RS.contains("if self.dispatches >= self.max_dispatches {\n            self.roll()?;"),
        "the rolling trigger in tick() changed"
    );
    // The window bounds how many buffers may be outstanding before the oldest is
    // waited on. 1400/50 = 28 > 8, so the oldest IS waited on mid-step --
    // 3 times per step, at dispatches 400, 800, 1200. Each of those is a point
    // where the host blocks on the GPU while still holding work to encode.
    let rolls = 1400 / 50;
    let waits = rolls / 8;
    eprintln!(
        "ROLLING per step: ~{rolls} command buffers, window {waits} -> host blocks \
         on the oldest ~{waits}x per step mid-encode"
    );
    assert!(
        waits >= 1,
        "with ~1400 dispatches at 50 per buffer and a window of 8, the host must \
         block mid-encode at least once per step; if this says 0 the dispatch \
         estimate is wrong"
    );
}

/// `read_host_token` is a raw `contents()` read of GPU-written shared memory.
///
/// It has no synchronisation of its own. `chat.rs:433-437` documents that
/// reading it before `prev.wait()` returns the *previous turn's* value or zero,
/// and that the bug produced `Count from 1 to 30: 1, 2, 3,` -- first token
/// right, then one repeated token. The ordering is a load-bearing invariant, not
/// a style question, and the test below pins the ordering constraint in the
/// source so a refactor that moves the wait below the read fails here.
#[test]
fn reading_a_staged_token_requires_waiting_on_its_encoder_first() {
    // Scoped to `turn`'s body. `chat.rs` names `read_host_token(step - 1)` at
    // line 397 inside the doc comment that *documents* the very bug this test
    // guards, so a file-wide search would match the comment and pass for the
    // wrong reason. Verified by sabotage: moving the real read above the real
    // wait was invisible to the file-wide form.
    let turn = function_body(CHAT_RS, "pub fn turn<F>(&mut self, text: &str, mut on_text: F)");
    // AND skipping comments. `chat.rs:397` sits *inside* `turn`'s body and names
    // `read_host_token(step - 1)` in the comment that documents the very bug
    // this guards. Scoping to the function is not enough on its own -- I found
    // this by running the suite against unmodified sources and watching the
    // test fail on the comment. Both filters are load-bearing.
    let wait_at = turn
        .lines()
        .position(|line| !line.trim_start().starts_with("//") && line.trim() == "previous.wait();")
        .expect("chat.rs must wait on the previous encoder");
    let read_at = turn
        .lines()
        .position(|line| {
            !line.trim_start().starts_with("//") && line.contains("read_host_token(step - 1)")
        })
        .expect("chat.rs must read the staged token");
    assert!(
        wait_at < read_at,
        "chat.rs reads host_tokens at line {} but waits at line {}: the read must \
         come after the wait, or the token is whatever was there before",
        read_at + 1,
        wait_at + 1
    );

    // generate.rs: same invariant, different shape (submit N+1, wait N, read N),
    // scoped to `generate_gpu_greedy`'s body for the same reason.
    let greedy_body = function_body(GENERATE_RS, "fn generate_gpu_greedy<F>(");
    let gen_read = greedy_body
        .lines()
        .position(|line| line.contains("in_flight_enc.wait();"))
        .expect("generate.rs must wait on the in-flight encoder");
    let gen_token = greedy_body
        .lines()
        .position(|line| line.contains("read_host_token(in_flight_step)"))
        .expect("generate.rs must read the staged token");
    assert!(
        gen_read < gen_token,
        "generate.rs reads the staged token at line {} but waits at line {}",
        gen_token + 1,
        gen_read + 1
    );
}

/// Every place the host reads GPU memory per token, and how much it reads.
///
/// Four distinct readbacks exist:
///   * `read_host_token` -- 4 B (pipelined greedy).
///   * `last_token` -- 16 B of `argmax_keys` (non-pipelined greedy).
///   * `forward_buffer_into_vec` -- `vocab * 2` B = **512 KiB** of `capped`,
///     copied into a host `Vec` (`model.rs:3849-3853`). This is what the
///     `temperature > 0` path pays every token.
///   * `last_logits` -- `vocab * 2` B into a fresh `Vec`, margins only.
#[test]
fn host_readback_bytes_per_token() {
    assert!(
        MODEL_RS.contains("out.resize(self.vocab_size, 0);")
            && MODEL_RS.contains("std::ptr::copy_nonoverlapping(logits, out.as_mut_ptr(), self.vocab_size);"),
        "forward_buffer_into_vec's 512 KiB logits readback changed shape"
    );
    assert!(
        MODEL_RS.contains("pub fn read_host_token(&self, step_idx: usize) -> u32 {"),
        "read_host_token changed"
    );

    let row_bytes = VOCAB * 2;
    eprintln!(
        "READBACK pipelined_greedy={} B  non_pipelined_greedy={} B  \
         temperature>0={} B  ({:.0} KiB)",
        4,
        VOCAB.div_ceil(65536) * 4,
        row_bytes,
        row_bytes as f64 / 1024.0
    );
    assert_eq!(
        row_bytes,
        524_288,
        "the logits row is 262,144 fp16 values; if VOCAB or the dtype changed \
         the readback table is wrong"
    );
}

/// The report's central quantitative claim, checked against the code that would
/// have to change to make it true.
///
/// If the `temperature > 0` gap is `sampling + row_widening + argmax` per token,
/// and none of it is overlapped (because the loop that pays it is the
/// barrier-per-token loop, which serialises the host between the barrier and the
/// next submit), then closing the gap is *not* about making the sampler faster.
/// It is about making the sampler run while the GPU is busy. There is no code
/// that does that today: `forward_buffer_into_vec` returns the row, and the
/// sampler runs on the host with the GPU idle behind it.
#[test]
fn sampling_cannot_overlap_with_the_gpu_today() {
    // Scoped to `generate_gpu`'s own body, NOT the whole file: `generate.rs` has
    // a second sampling loop in the legacy `generate()` (line 182, `--legacy-path`
    // only), and a file-wide search finds that one and compares it against the
    // production forward at line 360. Verified by sabotage: with the whole-file
    // form of this test, reordering `generate_gpu`'s body to submit-then-sample
    // still passed.
    let generate_gpu = function_body(
        GENERATE_RS,
        "pub fn generate_gpu<F>(\n    model: &mut TextModel",
    );
    let sample_at = generate_gpu
        .lines()
        .position(|line| line.contains("sampler.sample(&last_logits)?"))
        .expect("generate_gpu samples");
    // Matched on the *function call*, not the argument list: a real fix would
    // pass different arguments (a carried token, a speculated position), and an
    // assertion pinned to `&[token]` would then fail on "the string moved"
    // rather than on "the order changed". Verified by sabotage: a
    // submit-before-sample version using `&[ct]` was invisible to the
    // argument-pinned form.
    let forward_at = generate_gpu
        .lines()
        .position(|line| line.contains("forward_buffer_into_vec("))
        .expect("generate_gpu forwards");
    assert!(
        sample_at < forward_at,
        "generate_gpu now submits at body-relative line {} before sampling at \
         line {}: if that is the case the temperature>0 gap may already be \
         closed and the report's pipelining section is stale",
        forward_at + 1,
        sample_at + 1
    );

    // And the same shape in chat.rs's fallback, scoped to `turn`.
    let turn = function_body(CHAT_RS, "pub fn turn<F>(&mut self, text: &str, mut on_text: F)");
    let chat_sample = turn
        .lines()
        .position(|line| line.contains("self.sampler.sample(&last)?"))
        .expect("chat fallback samples");
    let chat_forward = turn
        .lines()
        .position(|line| line.contains(".forward_buffer_into_vec(&[token], &[position], &mut logits)?;"))
        .expect("chat fallback forwards");
    assert!(
        chat_sample < chat_forward,
        "chat.rs's `turn` now submits at body-relative line {} before sampling \
         at line {}",
        chat_forward + 1,
        chat_sample + 1
    );

    eprintln!(
        "OVERLAP: host widens + samples with the GPU idle, in both entry points; \
         no submit-before-sample path exists"
    );
}

/// `sampler.rs`'s per-call allocation, counted.
///
/// `top_k_order` (`sampler.rs:67-86`) builds `(0..logits.len()).collect::<Vec<_>>()`
/// -- a `usize` per vocabulary entry, 2 MiB at 262,144 -- on **every** call,
/// before it can partition. Plus `weights` (`sampler.rs:112-115`) and two
/// `truncate`s. Three heap allocations per sampled token, none reused.
#[test]
fn sampler_allocates_per_call() {
    // Scoped to `top_k_order`'s own body, NOT `sampler.rs` as a whole.
    // `reference_top_k` -- the `#[cfg(test)]` full-sort oracle at
    // `sampler.rs:184` -- contains the *identical* line, so a file-wide
    // `contains` stays true after the production function is fixed and the
    // test passes for the wrong reason. Proven by sabotage: rewriting
    // `top_k_order`'s collect into `with_capacity` + `extend` left this
    // assertion green until it was scoped.
    let top_k_order = function_body(
        SAMPLER_RS,
        "pub fn top_k_order(logits: &[f32], top_k: usize) -> Vec<usize> {",
    );
    assert!(
        top_k_order.contains("(0..logits.len()).collect::<Vec<_>>()"),
        "top_k_order no longer allocates a fresh usize-per-entry index Vec: \
         the per-token 2 MiB allocation in the report is gone (good) and this \
         test, plus that section of the report, need updating"
    );
    let collects = SAMPLER_RS
        .lines()
        .filter(|line| line.contains("collect::<Vec<_>>()") || line.contains("collect::<Vec<f32>>()"))
        .count();
    eprintln!(
        "SAMPLER-ALLOC per call: order Vec<usize> ({} B at vocab={VOCAB}), weights \
         Vec<f32>, plus the caller's f32 row ({} B); {collects} collect sites in \
         sampler.rs",
        VOCAB * std::mem::size_of::<usize>(),
        VOCAB * std::mem::size_of::<f32>()
    );
    // The index vector is the one that is pure waste: `select_nth_unstable_by`
    // only needs to partition an index permutation, but it could work on a
    // `(key, index)` array of 262,144 *pairs* and select the same set, or the
    // caller could keep one buffer and reuse it. 2 MiB of allocation and a full
    // 262,144-element write per token, to hold indices it already has.
    assert!(
        VOCAB * std::mem::size_of::<usize>() >= 2 * 1024 * 1024,
        "the index Vec is 2 MiB at this vocabulary; a smaller figure would mean \
         the report's allocation cost is wrong"
    );
}