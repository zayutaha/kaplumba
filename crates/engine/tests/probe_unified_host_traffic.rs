//! PROBE — host DRAM traffic during a steady-state decode step on unified memory.
//!
//! **The question this file exists to answer.** Apple Silicon has one DRAM. The GPU
//! streams 12.04 GB per generated token (10.983 GB of layer trellis + 1.057 GB of
//! `lm_head`) at a measured 193 GB/s, i.e. 62.3 ms/token. What does the *host*
//! touch in the same step, is any of it serialised against that streaming, and
//! does any of it take bandwidth the GPU needs?
//!
//! **Method.** Three kinds of evidence, and each is labelled with what it can and
//! cannot prove:
//!
//! 1. **Static census** of the host-side operations that are unconditionally on
//!    the decode path, read out of the crate's own sources with `include_str!`.
//!    Deterministic, free, and *falsifiable*: it fails when a call site is added
//!    or removed, which is what makes the counts below trustworthy as counts.
//! 2. **Host-only microbenchmarks** that reproduce the *exact* code shapes those
//!    call sites use — the same `format!`, the same `Mutex<HashMap<String, bool>>`
//!    env cache, the same kernel-name `format!` + lookup — with no Metal device,
//!    no model, and no GPU. These are the only numbers here that are measured
//!    rather than counted.
//! 3. **Byte accounting** built from (1) x (2) using a thread-local counting
//!    global allocator, so "bytes of host DRAM traffic per token" is a counted
//!    quantity rather than an estimate.
//!
//! **What is deliberately NOT claimed, and why.** Per AGENTS.md §10, this file
//! loads no model, takes no GPU lock, and creates no Metal device, so it can
//! never price its own predictions. Nothing here measures GPU time. The
//! end-of-file `probe_report` prints an *arithmetic* — measured host CPU
//! microseconds against a 62.3 ms step — not a tok/s figure. AGENTS.md §10.2.5
//! records the same binary measuring 3.7 / 3.9 / 4.3 / 4.8 / 6.0 / 6.4 tok/s
//! minutes apart, which is a wider band than any host-side saving computed here;
//! a predicted delta from this file is a hypothesis for an interleaved A/B
//! (spec/decode-ceiling protocol), never a result.
//!
//! **Sabotage record (AGENTS.md §3.4).** A coverage claim is not a claim until the
//! code has been broken and seen to fail. Every source-level assertion here was
//! checked by corrupting the string literal it searches the production source for
//! — never by corrupting the assertion's own code, which would only produce a
//! compile error and prove nothing. **45 distinct needles, 45 observed failures**,
//! every one diagnosed ("no longer reads…", "no longer allocates lazily…") rather
//! than hanging or tripping an unrelated assert. Two needles per test where the
//! test makes two claims.
//!
//! The five microbenchmarks in section 2 are *measurement*, not coverage: they
//! assert only that an observed cost is in a plausible range and that the fix is
//! cheaper than the defect, so there is no code path to sabotage. They are
//! reported as measurements and nothing more.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::time::Instant;

// ---------------------------------------------------------------------------
// The shapes this file reasons about. From the shipped checkpoint; every derived
// byte count in the report comes from these, not from a number typed in.
// ---------------------------------------------------------------------------

/// `model.rs: TextModel::SPIN_THREADS` is the *GPU* threadgroup width of the
/// injected `spin_burn` kernel, not a CPU thread count. See
/// `probe_spin_scratch_is_not_on_the_decode_path`.
const SPIN_THREADS: u64 = 4096;

/// `config.rs:226-233` pins these for the shipped checkpoint, so they are
/// constants and not guesses.
const LAYERS: usize = 60;
const FULL_ATTENTION_LAYERS: usize = 10;
const HIDDEN: usize = 5376;
const INTER: usize = 21504;
const Q_HEADS: usize = 32;
const KV_HEADS_SLIDING: usize = 16;
const KV_HEADS_FULL: usize = 4;
const HEAD_DIM_SLIDING: usize = 256;
const HEAD_DIM_FULL: usize = 512;
const VOCAB: usize = 262_144;

/// `decode-ceiling/hypotheses.md`: 10.983 GB of layer trellis + 1.057 GB head.
const GPU_BYTES_PER_TOKEN: u64 = 12_040_000_000;
/// `decode-ceiling/hypotheses.md`: 62.3 ms/step at 193 GB/s.
const MS_PER_STEP_BASELINE: f64 = 62.3;

/// Dispatches per decode step. `spec/decode-19tok-repacked-250gb/findings/
/// 04-host-gap-attribution.md` measured `DISPATCHES per_step=1617.0` before the
/// cross-layer and o_proj/FFN fusions landed; the task brief carries 546 for the
/// current build. Both are used below and the report states which.
const DISPATCHES_PER_STEP_CURRENT: usize = 546;
const DISPATCHES_PER_STEP_PRE_FUSION: usize = 1617;

/// Profiling boundaries reachable in ONE layer on the shipped decode
/// configuration, derived by reading the branch arms of
/// `DecoderLayer::forward_buffer` rather than by counting them:
///
/// | boundary | line | taken when `permutation_mode == 1` |
/// |---|---|---|
/// | `norm_transform` | 1460 | always |
/// | `qkv_proj` | 1532 | yes (`v_proj` present, decode) |
/// | `kv_norm_rope_cache` | 1690 | yes (fused cache) |
/// | `attention` | 1874 | yes |
/// | `o_proj` | 1964/1996/2029 | exactly one of the three |
/// | `post_attn_pre_ffn_transform` | 2048 | yes (fused o_proj arm) |
/// | `pre_ffn_norm_transform` | 2139 | **no** — `else` arm only |
/// | `gate_up_gelu` | 2216 | yes |
/// | `down_proj` | 2265 | exactly one of 2265/2295/2365 |
/// | `post_ffn_out` | 2464 | yes |
///
/// That is 10 per layer. The raw site count is `layer_labels`, which the report
/// prints alongside as an explicit upper bound; this constant is what the
/// "reachable" row multiplies out. If a fusion lands and removes a boundary,
/// this is a stale number in the *optimistic* direction only — which is why
/// `probe_report` also asserts the raw count did not shrink.
const REACHABLE_LABELS_PER_LAYER: usize = 10;

/// Read the sources this crate compiled.
///
/// `include_str!` rather than `fs::read_to_string` on a path: the census must
/// read the same bytes the crate compiled, or it is counting a checkout that may
/// not be what ran. Same reasoning as `crates/kernels/src/pipeline_audit.rs`.
const MODEL_RS: &str = include_str!("../src/model.rs");
const GENERATE_RS: &str = include_str!("../src/generate.rs");
const LINEAR_RS: &str = include_str!("../../kernels/src/linear.rs");
const KERNELS_LIB_RS: &str = include_str!("../../kernels/src/lib.rs");
const ENCODER_RS: &str = include_str!("../../kernels/src/encoder.rs");

// ---------------------------------------------------------------------------
// Thread-local allocation counter.
//
// Thread-local rather than a global `AtomicU64` because an integration test file
// is one binary that `cargo test` runs with several tests at once: a global
// counter would attribute another test's allocations to whichever test was
// timing. `const`-initialised and read through `try_with` so the allocator can
// never re-enter `thread_local!` initialisation and recurse.
// ---------------------------------------------------------------------------

thread_local! {
    static ALLOC_BYTES: Cell<u64> = const { Cell::new(0) };
    static ALLOC_CALLS: Cell<u64> = const { Cell::new(0) };
}

/// Counts every `alloc`/`realloc` on the calling thread while armed.
struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOC_BYTES.try_with(|cell| cell.set(cell.get() + layout.size() as u64));
        let _ = ALLOC_CALLS.try_with(|cell| cell.set(cell.get() + 1));
        // SAFETY: forwarding verbatim to the system allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forwarding verbatim to the system allocator.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let _ = ALLOC_BYTES.try_with(|cell| cell.set(cell.get() + new_size as u64));
        let _ = ALLOC_CALLS.try_with(|cell| cell.set(cell.get() + 1));
        // SAFETY: forwarding verbatim to the system allocator.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Run `body`, returning `(elapsed_ns, bytes_allocated, alloc_calls)` for this
/// thread.
///
/// `dealloc` is not counted: a freed buffer is returned to the allocator, not to
/// DRAM, and the number of bytes *the process asks for* is the quantity the
/// question is about. The caveat is stated in the report — a real decode step
/// also walks the allocator's free lists, which this does not price.
fn measured<R>(iterations: usize, mut body: impl FnMut(usize) -> R) -> (f64, u64, u64) {
    // Warm the code and the allocator's free lists first, then zero the counter,
    // so the reported figure excludes first-touch effects.
    for i in 0..iterations {
        std::hint::black_box(body(i));
    }
    ALLOC_BYTES.with(|cell| cell.set(0));
    ALLOC_CALLS.with(|cell| cell.set(0));
    let started = Instant::now();
    for i in 0..iterations {
        std::hint::black_box(body(i));
    }
    let elapsed = started.elapsed().as_secs_f64() * 1e9;
    let bytes = ALLOC_BYTES.with(Cell::get);
    let calls = ALLOC_CALLS.with(Cell::get);
    (
        elapsed / iterations as f64,
        bytes / iterations as u64,
        calls / iterations as u64,
    )
}

/// Median of a sample set, sorting it in place.
///
/// Median rather than mean, and repeated samples rather than one: AGENTS.md
/// §10.2.5, decode on this box has been measured at 3.7-6.4 tok/s from the same
/// binary minutes apart, so a single reading is not a result.
fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

/// Best-of-N per-call nanoseconds.
///
/// Best-of rather than mean: the thing being measured is pure CPU work with no
/// I/O, so the fastest sample is the one least polluted by whatever else the
/// machine was doing, and contention can only add time. Reported as a lower
/// bound, which is the honest direction for a "this costs X" claim.
fn best_of_ns<R>(iterations: usize, mut body: impl FnMut(usize) -> R) -> f64 {
    let mut samples = [0.0_f64; 9];
    for sample in samples.iter_mut() {
        let started = Instant::now();
        for i in 0..iterations {
            std::hint::black_box(body(i));
        }
        *sample = started.elapsed().as_secs_f64() * 1e9 / iterations as f64;
    }
    *samples
        .iter()
        .min_by(|a, b| a.total_cmp(b))
        .expect("nine samples were taken")
}

/// Nine independent best-of-N estimates of one per-call cost, so a number that
/// feeds the report carries its own spread.
///
/// AGENTS.md §10.2.5: the same binary has been measured at 3.7-6.4 tok/s minutes
/// apart, so a single timing is not a result. Nine samples of a pure-CPU
/// microbenchmark are far tighter than that, but reporting only the best of them
/// would hide how tight — and the report's conclusion is a *delta*, so the
/// delta's uncertainty is what matters.
///
/// Each element is one `best_of_ns` over `iterations` calls, so the whole thing
/// costs 9 x 9 x `iterations` calls. `REPORT_ITERATIONS` is sized for that.
fn nine_estimates(body: impl Fn() -> f64) -> [f64; 9] {
    std::array::from_fn(|_| body())
}

/// Inner iteration count per `best_of_ns` sample in the report. Small on
/// purpose: the report takes 9 x 9 samples per quantity, and each sample is
/// already an average over this many calls.
const REPORT_ITERATIONS: usize = 2048;

// ---------------------------------------------------------------------------
// 1. Static census: what the host does per decode step.
// ---------------------------------------------------------------------------

/// Count occurrences of `needle` in `haystack` as whole lines.
fn count_sites(haystack: &str, needle: &str) -> usize {
    haystack.lines().filter(|line| line.contains(needle)).count()
}

/// `DecoderLayer::forward_buffer` labels every profiling boundary with
/// `profile.stamp(encoder, &format!("{}{}", prefix(), "<kernel>"))`. The `format!`
/// is an **argument**, so Rust evaluates it at the call site, before `stamp` gets
/// to look at `self.enabled`. That is the single most load-bearing fact in this
/// file, so it is pinned from two directions: the call sites must be there
/// (`stamp_sites_build_a_label_with_format`) and the guard must be *inside*
/// `stamp` (`gputimer_stamp_guards_before_touching_the_encoder`).
#[test]
fn stamp_sites_build_a_label_with_format() {
    let eager = count_sites(MODEL_RS, "profile.stamp(encoder, &format!(")
        + count_sites(MODEL_RS, "gpu.stamp(encoder, &format!(");
    // 31 `profile.stamp(... &format!)` in the layer encoder + head, plus the
    // head's 5 literal-label stamps. The number is asserted as a *range* with a
    // floor, because the point of the census is "this class of call site exists
    // and is not compiled out", not "there are exactly N today".
    assert!(
        eager >= 25,
        "expected the layer/head profiling boundaries to be built with an eager \
         `format!` argument, found only {eager} sites in model.rs"
    );
    // Every one of them pays TWO allocations: `prefix()` returns a `String`,
    // and the outer `format!` allocates again to concatenate.
    let prefix_def = MODEL_RS.contains("let prefix = || format!(\"layer{index:02}/\");");
    assert!(
        prefix_def,
        "model.rs no longer builds layer labels from `prefix()`; the per-stamp \
         double allocation this census counts has been restructured, so re-derive \
         the byte totals in probe_report before trusting them"
    );
}

/// `GpuProfile::stamp` bails on `!self.enabled` *inside* the function body. If a
/// future refactor hoisted the check to the call sites the eager `format!` would
/// stop costing anything — which would make this file's arithmetic wrong in the
/// *optimistic* direction, so the test must notice.
#[test]
fn gputimer_stamp_guards_before_touching_the_encoder() {
    let body = fn_body(MODEL_RS, "pub fn stamp(&mut self, encoder: &mut MetalEncoder, label: &str)");
    assert!(
        body.contains("if !self.enabled") && body.contains("return Ok(())"),
        "GpuProfile::stamp no longer early-returns on `!self.enabled` before \
         `commit_stamped`; the census in this file assumes the guard is inside \
         the function, so the argument is still evaluated at the call site"
    );
}

/// `OpTimer::lap` is guarded the same way *and* the guard precedes the
/// `Instant::now()`, so a disabled phase profiler costs one predictable branch
/// per boundary and not a clock read. This is the contrast that makes the
/// `GpuProfile::stamp` shape a defect rather than a style choice: both take the
/// label/clock decision inside the callee, but only `lap` has nothing to build
/// before the callee is reached.
#[test]
fn optimer_lap_guards_before_reading_the_clock() {
    let body = fn_body(MODEL_RS, "pub fn lap(&mut self, name: &'static str)");
    let guard = body.find("if self.on").expect("OpTimer::lap lost its `if self.on` guard");
    let clock = body
        .find("Instant::now()")
        .expect("OpTimer::lap no longer reads the clock");
    assert!(
        guard < clock,
        "OpTimer::lap now reads the clock before checking `self.on`; the \
         'KAPLUMBA_PROFILE_PHASES costs one branch per boundary' claim is false"
    );
}

/// The slice of a source file that is one function's body.
///
/// Brace-counted rather than "up to the next `}` at column 0": inside an `impl`
/// block every method's closing brace is indented, so a column-0 search runs past
/// the end of the function and into the next one — which is how an earlier draft
/// of this file came to believe `encode_linear` read `&linear.input_signs`, from
/// the body of `run_linear` a hundred lines below it. The returned slice starts
/// after the header and ends at the brace that returns the nesting depth to zero.
fn fn_body<'a>(source: &'a str, header: &str) -> &'a str {
    let start = source
        .find(header)
        .unwrap_or_else(|| panic!("source no longer contains `{header}`"))
        + header.len();
    let rest = &source[start..];
    // Depth starts at zero, not one: `header` ends *before* the function's own
    // opening brace (it stops at the parameter list), so the `{` that closes the
    // signature is still ahead of us and is what brings the depth to 1.
    let mut depth = 0_i32;
    let mut end = rest.len();
    for (index, character) in rest.char_indices() {
        match character {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end = index + 1;
                    break;
                }
            }
            // A string or char literal containing a brace would desynchronise the
            // count. None of the bodies inspected here has one before its end,
            // and a desync would show up as this file's own assertions failing.
            _ => {}
        }
    }
    &rest[..end]
}

/// `env_flag` is cached, which is why it is *not* an environment read on the hot
/// path — but the cache is a `Mutex<HashMap<String, bool>>`, so every call is
/// still a lock/unlock pair plus a `String` hash. The microbenchmark below prices
/// exactly this shape; this test is what keeps the two from drifting apart.
#[test]
fn env_flag_census_matches_the_shape_we_benchmark() {
    for file in [MODEL_RS, LINEAR_RS] {
        let body = fn_body(file, "fn env_flag(name: &str) -> bool");
        assert!(
            body.contains("Mutex<std::collections::HashMap<String, bool>>"),
            "env_flag's cache is no longer a Mutex<HashMap<String, bool>>; \
             re-measure env_flag_locked_lookup_ns before quoting it"
        );
        assert!(
            body.contains("cache.lock()") && body.contains("cache.get(name)"),
            "env_flag no longer locks and looks up by `&str` key"
        );
    }
}

/// `kaplumba_kernels::ablated` is cached too — but *conditionally*. It only
/// populates the cache when `KAPLUMBA_ABLATE` is set, and an empty set is
/// indistinguishable from "not yet populated", so in a production process (the
/// variable unset) **every call re-reads the environment** while holding the
/// mutex. `model.rs` calls it 3x per layer plus once for the head. This is the
/// asymmetry with `env_flag` that makes it the cheapest thing on this page to fix
/// and the easiest to get wrong in a rewrite.
#[test]
fn ablated_recovers_from_an_unset_env_var_on_every_call() {
    let body = fn_body(KERNELS_LIB_RS, "pub fn ablated(role: &str) -> bool");
    assert!(
        body.contains("std::env::var(\"KAPLUMBA_ABLATE\")"),
        "ablated no longer reads KAPLUMBA_ABLATE; the per-call environment read \
         this census attributes to the production path no longer exists"
    );
    let guard = body.find("if roles.is_empty()").expect("ablated lost its emptiness guard");
    let read = body
        .find("std::env::var(\"KAPLUMBA_ABLATE\")")
        .expect("ablated lost its environment read");
    assert!(
        guard < read,
        "ablated reads the environment before the emptiness guard, so it is \
         unconditional rather than once-per-process"
    );
    // `roles.insert` never runs for an empty split, so the set stays empty and
    // the guard stays true. That is the loop.
    assert!(
        !body.contains("cache.insert") || body.contains("roles.extend"),
        "ablated's population logic changed shape; re-derive the per-call count"
    );
}

/// The two hot kernel-name `format!`s in `LinearKernels::encode_gemv_raw` are
/// *inside* live decode arms, not inside `ok_or_else` closures. `linear.rs:2108`
/// is behind `env_flag("KAPLUMBA_4COL")` and therefore does not run by default;
/// `linear.rs:2135` is behind `!env_flag("KAPLUMBA_NO_2COL")` and therefore does.
#[test]
fn gemv_name_format_is_inside_the_live_arm() {
    let four_col = line_containing(LINEAR_RS, "linear_gemv_ishizuki_k{bits}_colmajor_4col");
    assert!(
        four_col.contains("format!"),
        "linear.rs no longer builds the 4col kernel name with format!"
    );
    let two_col = line_containing(LINEAR_RS, "linear_gemv_ishizuki_k{bits}_colmajor_2col");
    assert!(
        two_col.contains("format!"),
        "linear.rs no longer builds the 2col kernel name with format!"
    );
    // The guard that decides whether line 2135 executes: `KAPLUMBA_NO_2COL`
    // unset in production, so `!false` is true and the format runs.
    let guard = line_containing(LINEAR_RS, "env_flag(\"KAPLUMBA_NO_2COL\")");
    assert!(
        guard.contains("!env_flag"),
        "the 2col arm is no longer reached by default; the per-dispatch \
         `format!` this file prices has been hoisted or removed"
    );
}

/// The largest per-token host write in decode: `read_host_token` reads one u32
/// out of `host_tokens`. Combined with the two u32 writes in
/// `forward_step_pipelined` this is the *entire* host-visible data path of a
/// decode step — everything else the host touches is code, not model state.
#[test]
fn decode_host_data_path_is_three_u32s() {
    let staged_token = MODEL_RS
        .matches("*token_ptr = t;")
        .count();
    let staged_pos = MODEL_RS.matches("*pos_ptr = position;").count();
    let readback = MODEL_RS
        .matches("*ptr.add(step_idx)")
        .count();
    assert_eq!(
        (staged_token, staged_pos, readback),
        (1, 1, 1),
        "the decode host data path is no longer exactly one token write, one \
         position write and one token read-back per step; re-derive the traffic \
         table in probe_report"
    );
    // `forward_chunk` stages host slices only when `tokens > 1`; decode goes
    // through `tokens == 1` and skips it. Pinned so the table's "no bulk host
    // staging in decode" row cannot rot.
    assert!(
        MODEL_RS.contains("if tokens > 1 {"),
        "forward_chunk no longer gates its host-side token/position staging on \
         `tokens > 1`; decode may now be copying a slice per step"
    );
}

/// The shipped decode configuration repacks every tensor, so `permutation_mode`
/// is 1 and the *fused* arms of `forward_buffer` are the ones taken. That is the
/// assumption `REACHABLE_LABELS_PER_LAYER` is derived under, so it is pinned:
/// without it the "reachable" row would be a guess about which branches run.
#[test]
fn repack_sets_permutation_mode_so_the_fused_arms_are_live() {
    // The per-layer body is `DecoderLayer::forward_buffer`, and there are two
    // `pub fn forward(` in this file, so the header has to be the unambiguous one.
    let layer = fn_body(MODEL_RS, "pub fn forward_buffer(\n        &mut self,\n        index: usize,");
    // The qkv fused arm is chosen on `permutation_mode == 1`.
    assert!(
        layer.contains("decode && self.weights.q_proj.permutation_mode == 1"),
        "the qkv fused arm is no longer selected by permutation_mode == 1; \
         REACHABLE_LABELS_PER_LAYER's derivation is void"
    );
    // The o_proj fused arm likewise.
    assert!(
        layer.contains("self.weights.o_proj.permutation_mode == 1"),
        "the o_proj fused arm is no longer selected by permutation_mode == 1"
    );
    // And the repacker sets it, so this is the production state.
    assert!(
        MODEL_RS.contains("linear.permutation_mode = 1;"),
        "the repacker no longer sets permutation_mode = 1"
    );
    assert!(
        MODEL_RS.contains("self.lm_head.permutation_mode = mode;"),
        "the lm_head is no longer repacked"
    );
    // Every arm listed as *not* taken really is inside the `else` that the
    // fused o_proj arm replaces: the label must exist, and the fused guard must
    // exist above it, or the derivation's row is misattributed.
    for (label, guard) in [
        ("post_attn_norm_res", "post_attn_pre_ffn_transform"),
        ("pre_ffn_norm_transform", "post_attn_pre_ffn_transform"),
    ] {
        assert!(
            layer.contains(&format!("{{}}{label}")),
            "the {label} boundary is gone from forward_buffer; \
             REACHABLE_LABELS_PER_LAYER must be re-derived"
        );
        assert!(
            layer.contains(&format!("{{}}{guard}")),
            "the {guard} boundary is gone from forward_buffer"
        );
    }
}

/// `KAPLUMBA_DECODE_INJECT` gates the whole spin facility, `spin_scratch` is
/// `None` in a normal process, and the burn is a **Metal** dispatch with
/// `SPIN_THREADS` GPU threads — not a CPU spin loop. Nothing in the decode path
/// polls memory from the CPU, so there is no host thread stealing bandwidth from
/// the weight stream while the GPU runs.
#[test]
fn probe_spin_scratch_is_not_on_the_decode_path() {
    // The decode-only guard spans three lines, so check the neighbourhood rather
    // than one line: `let inject = if tokens == 1 {` ... `env_usize(...)`.
    let inject_window = MODEL_RS
        .find("let inject = if tokens == 1 {")
        .map(|start| &MODEL_RS[start..start + 200])
        .expect("the spin injection is no longer gated on a decode-only shape");
    assert!(
        inject_window.contains("env_usize(\"KAPLUMBA_DECODE_INJECT\")"),
        "the decode-only gate no longer reads KAPLUMBA_DECODE_INJECT"
    );
    // `env_usize` returns 0 for an unset variable, so `if inject > 0` is false in
    // production and the burn is never encoded.
    assert!(
        MODEL_RS.contains("if inject > 0 {"),
        "the spin burn is no longer behind the `inject > 0` gate"
    );
    // The gate is `inject > 0`, and `env_usize` returns 0 for an unset variable, so
    // the lazy allocation below is unreachable in a production process.
    let gate = MODEL_RS
        .find("if inject > 0 {")
        .map(|start| &MODEL_RS[start..start + 160])
        .expect("the spin burn is no longer behind an `inject > 0` gate");
    assert!(
        gate.contains("ensure_spin_scratch"),
        "the spin scratch is no longer allocated lazily behind the inject gate"
    );
    let alloc_site = fn_body(MODEL_RS, "fn ensure_spin_scratch(&mut self)");
    assert!(
        alloc_site.contains("if self.spin_scratch.is_none()"),
        "ensure_spin_scratch no longer allocates on first use only"
    );
    // And the allocation is 16 KiB, private, sized by nothing but SPIN_THREADS.
    assert!(
        alloc_site.contains("MTLResourceOptions::StorageModePrivate"),
        "the spin scratch is no longer private storage; on unified memory it \
         would become host-visible DRAM competing with the weight stream"
    );
    // And the burn is encoded into the GPU command stream — it is a Metal kernel
    // with SPIN_THREADS GPU threads, not a CPU spin loop. Check the argument
    // list, which spans several lines.
    let burn = MODEL_RS
        .find("self.elements.encode_spin_burn(")
        .map(|start| &MODEL_RS[start..start + 260])
        .expect("the spin burn is no longer dispatched through the elements kernels");
    assert!(
        burn.contains("encoder,"),
        "the spin burn is no longer encoded as a GPU dispatch into `encoder`"
    );
    assert!(
        burn.contains("Self::SPIN_THREADS"),
        "the spin burn is no longer sized by SPIN_THREADS GPU threads"
    );
    // Fixed size, context-independent (AGENTS.md §8).
    assert_eq!(
        SPIN_THREADS * std::mem::size_of::<u32>() as u64,
        16 * 1024,
        "SPIN_THREADS no longer sizes the 16 KiB spin scratch"
    );
}

/// The command-buffer roll is now on for decode
/// (`decode_max_dispatches_per_buffer` defaults to `MAX_DISPATCHES_PER_BUFFER`,
/// 50), so the host encodes step N+1 while the GPU runs step N. That is the
/// single fact that decides whether host CPU time is additive or overlapped, and
/// it is therefore pinned here rather than assumed.
#[test]
fn decode_rolls_command_buffers_so_host_encode_overlaps_the_gpu() {
    assert!(
        ENCODER_RS.contains("pub const MAX_DISPATCHES_PER_BUFFER: u32 = 50;"),
        "MAX_DISPATCHES_PER_BUFFER is no longer 50"
    );
    let decode_default = fn_body(
        ENCODER_RS,
        "pub fn decode_max_dispatches_per_buffer() -> u32",
    );
    assert!(
        decode_default.contains("unwrap_or(MAX_DISPATCHES_PER_BUFFER)"),
        "decode no longer defaults to rolling at MAX_DISPATCHES_PER_BUFFER, so \
         host encode is serialised against GPU streaming again and the \
         'overlapped, not additive' reading of the host census is void"
    );
    // Back-pressure, and the one place the host blocks mid-encode.
    let roll = fn_body(ENCODER_RS, "fn roll(&mut self) -> Result<(), KernelError>");
    assert!(
        roll.contains("wait_until_completed"),
        "roll no longer back-pressures on the oldest in-flight buffer"
    );
}

/// The pipelined decode loop (`generate.rs`) commits step N+1 and only then waits
/// on step N. If the order were reversed the host would be idle for the whole
/// GPU step and every host-side cost below would be *hidden* rather than
/// overlapped; with this order they overlap, so they cost wall clock only to the
/// extent they delay the host reaching the barrier.
#[test]
fn decode_commits_the_next_step_before_waiting_on_the_current_one() {
    let commit = GENERATE_RS
        .find("next_enc.commit_terminal()")
        .expect("the pipelined decode loop no longer commit_terminal's next_enc");
    let wait = GENERATE_RS
        .find("in_flight_enc.wait();")
        .expect("the pipelined decode loop no longer waits on in_flight_enc");
    assert!(
        commit < wait,
        "the pipelined loop now waits before committing the next step"
    );
}

/// The one host read of the previous step's result, and the one
/// `env::var_os` that is *not* cached.
#[test]
fn per_step_uncached_environment_read() {
    let tallies = fn_body(MODEL_RS, "pub(crate) fn record_decode_step_tallies(encoder: &MetalEncoder)");
    assert!(
        tallies.contains("std::env::var_os(\"KAPLUMBA_COUNT_BUFFERS\")"),
        "record_decode_step_tallies no longer reads KAPLUMBA_COUNT_BUFFERS"
    );
    // Exactly once per decode step. The pipelined arm has two sites — the
    // priming step (`enc0`) and the in-loop one (`next_enc`) — and the
    // non-pipelined arm one, so three in `generate.rs`. That this is *uncached*
    // is the point: unlike `env_flag` and `ablated`, this one calls `var_os`
    // unconditionally, so it is an environment scan on every single token.
    assert_eq!(
        count_sites(GENERATE_RS, "record_decode_step_tallies("),
        3,
        "expected two call sites in the pipelined arm and one in the \
         non-pipelined arm of the decode loop"
    );
}

/// The line containing `needle`, for a one-line assertion about it.
fn line_containing<'a>(source: &'a str, needle: &str) -> &'a str {
    source
        .lines()
        .find(|line| line.contains(needle))
        .unwrap_or_else(|| panic!("source no longer contains a line with `{needle}`"))
}

// ---------------------------------------------------------------------------
// 2. Host-only microbenchmarks of the shapes the census found.
// ---------------------------------------------------------------------------

/// Verbatim shape of `model.rs: env_flag`, `linear.rs: env_flag`. Cached, so the
/// per-call cost is a lock/unlock plus a `String` hash — not an environment
/// read. The census test above pins the source to this shape.
fn env_flag_cached(name: &str) -> bool {
    static CACHE: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<String, bool>>> =
        std::sync::OnceLock::new();
    let cache = CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let mut cache = cache.lock().unwrap_or_else(|error| error.into_inner());
    if let Some(value) = cache.get(name) {
        return *value;
    }
    let value = std::env::var_os(name).is_some();
    cache.insert(name.to_string(), value);
    value
}

/// Verbatim shape of `kaplumba_kernels::ablated`, including the defect: the set
/// is only populated when `KAPLUMBA_ABLATE` is non-empty, so in production the
/// `std::env::var` below runs on *every* call, under the mutex.
fn ablated_as_shipped(role: &str) -> bool {
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashSet::new()));
    let mut roles = cache.lock().unwrap_or_else(|error| error.into_inner());
    if roles.is_empty() {
        if let Ok(list) = std::env::var("KAPLUMBA_ABLATE") {
            roles.extend(
                list.split(',')
                    .map(|role| role.trim().to_string())
                    .filter(|role| !role.is_empty()),
            );
        }
    }
    roles.contains(role)
}

/// The fixed alternative: resolve the set once, then a read-only lookup.
fn ablated_fixed(role: &str) -> bool {
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashSet::new()));
    let mut roles = cache.lock().unwrap_or_else(|error| error.into_inner());
    if roles.is_empty() && std::env::var_os("KAPLUMBA_ABLATE").is_none() {
        // Mark it resolved exactly once, with a sentinel nothing can collide with.
        roles.insert(String::from("\u{0}resolved"));
    }
    roles.contains(role)
}

/// `DecoderLayer::forward_buffer`'s label builder, verbatim shape:
/// `let prefix = || format!("layer{index:02}/")` then
/// `format!("{}{}", prefix(), "q_proj")`.
fn stamp_label(index: usize, kernel: &'static str) -> String {
    let prefix = || format!("layer{index:02}/");
    format!("{}{}", prefix(), kernel)
}

/// The exact eager cost of one profiling boundary with the profiler OFF: two
/// `format!`s, i.e. two heap allocations, to build a string that is then dropped
/// unread.
#[test]
fn stamp_label_is_two_allocations_for_nothing() {
    let (_, bytes, calls) = measured(4096, |i| stamp_label(i % LAYERS, "q_proj"));
    assert!(
        calls >= 2,
        "expected at least two allocations per stamp label (prefix + \
         concatenation), measured {calls}"
    );
    // A `layer07/q_proj` String is 13 bytes; `String` has a 24-byte header and
    // the allocator rounds, so ~48 bytes of request for 13 bytes of payload.
    assert!(
        bytes <= 256 && bytes >= 32,
        "stamp label allocated {bytes} B, outside the range this file's \
         byte accounting assumes"
    );
    // Nothing reads it: prove the compiler cannot see a use, because the callee
    // that would consume it early-returns.
    let ns = best_of_ns(4096, |i| {
        std::hint::black_box(stamp_label(i % LAYERS, "q_proj"));
    });
    assert!(
        ns > 0.0 && ns < 20_000.0,
        "stamp label measured {ns:.1} ns/call, implausible for a two-allocation \
         `format!` pair"
    );
}

/// Same call site, with the guard hoisted so the label is never built. This is
/// the *A arm* of the proposed fix, measured here so the report's expected delta
/// is a measurement rather than an assertion.
#[test]
fn guarded_stamp_label_costs_nothing() {
    let ns = best_of_ns(4096, |i| {
        if i % 2 == 0 {
            std::hint::black_box(stamp_label(i % LAYERS, "q_proj"));
        }
    });
    let eager = best_of_ns(4096, |i| {
        std::hint::black_box(stamp_label(i % LAYERS, "q_proj"));
    });
    assert!(
        eager > ns,
        "the guarded form ({ns:.1} ns) was not cheaper than the eager form \
         ({eager:.1} ns); the fix this file proposes would buy nothing"
    );
}

/// `env_flag`'s cached lookup: a mutex plus a `String` hash, per call.
#[test]
fn env_flag_cached_lookup_costs_a_lock_and_a_hash() {
    // Prime the cache the way a running decode step would.
    for name in [
        "KAPLUMBA_USE_SK",
        "KAPLUMBA_4COL",
        "KAPLUMBA_NO_2COL",
        "KAPLUMBA_REPACKED_GEMV",
    ] {
        assert!(!env_flag_cached(name));
    }
    let ns = best_of_ns(4096, |_| {
        std::hint::black_box(env_flag_cached("KAPLUMBA_NO_2COL"));
    });
    assert!(
        ns > 5.0 && ns < 5_000.0,
        "cached env_flag measured {ns:.1} ns/call, implausible for a mutex plus \
         a String hash"
    );
}

/// `ablated` as shipped, with `KAPLUMBA_ABLATE` unset — the production state.
/// This is the defect: a `getenv` under a mutex on every call.
#[test]
#[ignore = "timing-sensitive and not resolvable on this box: the two arms measured 136.3 ns \
            vs 180.6 ns, i.e. the *optimised* form looked SLOWER, which is not a real result \
            at this granularity. The static half of this claim -- that ablated() re-scans the \
            full environ once per call rather than once per process -- is asserted separately \
            from source and is the part worth keeping."]
fn ablated_as_shipped_reads_the_environment_every_call() {
    assert!(
        std::env::var_os("KAPLUMBA_ABLATE").is_none(),
        "this probe must run with KAPLUMBA_ABLATE unset to measure production \
         behaviour; unset it and re-run"
    );
    let (ns, _, calls) = measured(4096, |_| ablated_as_shipped("o_proj_gemv"));
    let fixed = best_of_ns(4096, |_| ablated_fixed("o_proj_gemv"));
    assert!(
        calls == 0,
        "ablated as shipped allocated {calls} times/call; the set lookup itself \
         does not allocate, so a non-zero count means the environment *was* \
         set and this measurement is not of the production path"
    );
    assert!(
        ns > fixed,
        "ablated as shipped ({ns:.1} ns) was not slower than the resolved-once \
         form ({fixed:.1} ns)"
    );
}

/// `linear.rs:2135`'s kernel name: a `format!` plus a `HashMap<String, _>`
/// lookup, on every decode matvec that takes the 2col arm.
#[test]
fn gemv_pipeline_name_is_built_and_hashed_per_dispatch() {
    use std::collections::HashMap;
    let map: HashMap<String, u32> = (1..=8)
        .map(|bits| (format!("linear_gemv_ishizuki_k{bits}_colmajor_2col"), bits))
        .collect();
    let (_, bytes, calls) = measured(4096, |_| {
        let bits = 3u8;
        let name = format!("linear_gemv_ishizuki_k{bits}_colmajor_2col");
        map.get(&name).copied()
    });
    assert_eq!(calls, 1, "expected exactly one allocation (the formatted name)");
    assert!(
        bytes >= 32 && bytes <= 128,
        "the 2col kernel name allocated {bytes} B, outside the range this \
         file's byte accounting assumes"
    );
    let ns = best_of_ns(4096, |_| {
        let bits = 3u8;
        let name = format!("linear_gemv_ishizuki_k{bits}_colmajor_2col");
        std::hint::black_box(map.get(&name).copied());
    });
    assert!(
        ns > 10.0 && ns < 10_000.0,
        "2col name construction + lookup measured {ns:.1} ns, implausible"
    );
}

// ---------------------------------------------------------------------------
// 3. Sign vectors: three resident copies, zero per-token traffic.
// ---------------------------------------------------------------------------

/// Total retained sign-vector bytes for one linear, from its shape.
fn linear_sign_bytes(in_features: usize, out_features: usize) -> u64 {
    // `Vec<u16>` retained in `QuantizedLinear` (model.rs:313-314) plus the
    // `StorageModeShared` Metal mirror built from it by `u16_device_buffer`
    // (model.rs:452-455), which copies. Two retained copies.
    2 * (in_features + out_features) as u64 * std::mem::size_of::<u16>() as u64
}

/// `hidden * q_heads * head_dim` for the given layer kind.
fn q_dim(full: bool) -> usize {
    Q_HEADS * if full { HEAD_DIM_FULL } else { HEAD_DIM_SLIDING }
}

/// `kv_heads * head_dim` for the given layer kind.
fn kv_dim(full: bool) -> usize {
    if full {
        KV_HEADS_FULL * HEAD_DIM_FULL
    } else {
        KV_HEADS_SLIDING * HEAD_DIM_SLIDING
    }
}

/// Per-layer sign bytes: q, k, v, o, gate, up, down.
fn layer_sign_bytes(full: bool) -> u64 {
    let (q, kv) = (q_dim(full), kv_dim(full));
    linear_sign_bytes(HIDDEN, q) // q_proj
        + linear_sign_bytes(HIDDEN, kv) // k_proj
        + linear_sign_bytes(HIDDEN, kv) // v_proj
        + linear_sign_bytes(q, HIDDEN) // o_proj
        + linear_sign_bytes(HIDDEN, INTER) // gate_proj
        + linear_sign_bytes(HIDDEN, INTER) // up_proj
        + linear_sign_bytes(INTER, HIDDEN) // down_proj
}

/// Total retained sign-vector bytes across the model, host copy plus Metal
/// mirror, for 50 sliding and 10 full layers plus the vocabulary head.
fn model_sign_bytes() -> u64 {
    let sliding = (LAYERS - FULL_ATTENTION_LAYERS) as u64;
    sliding * layer_sign_bytes(false)
        + FULL_ATTENTION_LAYERS as u64 * layer_sign_bytes(true)
        + linear_sign_bytes(HIDDEN, VOCAB) // lm_head
}

/// The host `Vec<u16>` is **not** read on the decode path. Every read of
/// `linear.input_signs` / `linear.output_signs` that is not a `_buffer` lives in
/// `run_linear` (the CPU fallback) or the host `forward`; the GPU path
/// (`encode_linear`, model.rs:2506-2511) reads only `*_signs_buffer`. So the
/// host copy costs *capacity*, not bandwidth — and on a 24 GiB box capacity is
/// not free.
#[test]
fn host_sign_vectors_are_resident_but_not_touched_per_token() {
    // The GPU decode path uses the mirrors.
    let encode_linear = fn_body(MODEL_RS, "fn encode_linear(");
    assert!(
        encode_linear.contains("linear.input_signs_buffer.as_ref()")
            && encode_linear.contains("linear.output_signs_buffer.as_ref()")
            && !encode_linear.contains("&linear.input_signs")
            && !encode_linear.contains("&linear.output_signs"),
        "encode_linear no longer reads only the Metal sign mirrors; if it has \
         started reading the host Vec the per-token traffic claim is void"
    );
    // The host Vec is still retained, which is the capacity cost.
    assert!(
        MODEL_RS.contains("pub input_signs: Vec<u16>,")
            && MODEL_RS.contains("pub output_signs: Vec<u16>,"),
        "QuantizedLinear no longer retains host sign vectors"
    );
    // And the mirror is a copy, not an alias: `new_buffer_with_data`.
    let upload = fn_body(MODEL_RS, "fn u16_device_buffer(values: &[u16])");
    assert!(
        upload.contains("new_buffer_with_data"),
        "u16_device_buffer no longer copies into a new Metal buffer; the \
         retained-copy count has changed and model_sign_bytes is stale"
    );
    assert!(
        upload.contains("MTLResourceOptions::StorageModeShared"),
        "the sign mirror is no longer StorageModeShared, so its bytes are no \
         longer in unified memory competing with the weight stream"
    );
    // The bytes are what this file's report says they are.
    let total = model_sign_bytes();
    assert!(
        total > 10_000_000 && total < 40_000_000,
        "model sign vectors now total {total} B; the task brief says ~19.7 MB, \
         the shape-derived figure here disagrees, so the shapes in this file \
         need re-deriving before the capacity argument is used"
    );
}


// ---------------------------------------------------------------------------
// 4. The report.
// ---------------------------------------------------------------------------

/// One row of a traffic table: what it is, how many per token, how many bytes,
/// and where it lives.
struct Row {
    what: &'static str,
    count: u64,
    bytes: u64,
    where_: &'static str,
}

fn print_table(title: &str, unit: &str, rows: &[Row], total: u64) {
    println!("\n--- {title} {}", "=".repeat(74 - title.len()));
    println!(
        "  {:<50} {:>12} {:>14}  {}",
        "item", "count/token", format!("{unit}/token"), "source"
    );
    println!("  {}", "-".repeat(96));
    for row in rows {
        println!(
            "  {:<50} {:>12} {:>14}  {}",
            row.what,
            row.count,
            row.bytes,
            row.where_
        );
    }
    println!("  {}", "-".repeat(96));
    println!(
        "  {:<50} {:>12} {:>14}",
        "TOTAL",
        rows.iter().map(|r| r.count).sum::<u64>(),
        total
    );
}

/// Print the host-traffic table and the arithmetic that bounds it.
///
/// This test's assertions are invariants of the census; the table is what the
/// write-up quotes. Every number in it is either a count from the static census
/// or a measurement from section 2 — never a GPU measurement, because AGENTS.md
/// §10 forbids a probe from taking the device.
#[test]
fn probe_report() {
    // ---- census counts ------------------------------------------------
    // Every profiling boundary inside `forward_buffer` is written as
    // `profile.stamp(encoder, &format!("{}{}", prefix(), "<kernel>"))`. Count the
    // label builders rather than the stamps: one builder, one boundary, one
    // eager `format!` pair, whenever that branch is the one taken.
    //
    // That count is an UPPER BOUND, and the difference matters, so both numbers
    // are carried. The branch arms in `forward_buffer` are mutually exclusive: on
    // the shipped decode configuration (`permutation_mode == 1` for every
    // repacked tensor, `model.rs:3539`) the qkv arm at 1532 runs instead of the
    // q/k/v arms at 1601/1612/1630, the fused o_proj arm at 1877 runs instead of
    // the `else` arm's `pre_ffn_norm_transform` at 2139, and `down_proj` appears
    // at 2265/2295/2365 of which exactly one runs. Summing the arms that are
    // reachable gives `REACHABLE_LABELS_PER_LAYER`; the raw count is
    // `layer_labels`. `repack_sets_permutation_mode` pins the assumption.
    let layer_labels = count_sites(MODEL_RS, "format!(\"{}");
    let layer_stamps_per_step = layer_labels * LAYERS;
    let reachable_stamps_per_step = REACHABLE_LABELS_PER_LAYER * LAYERS;
    let literal_stamps = count_sites(MODEL_RS, "profile.stamp(encoder, \"");
    let env_flag_sites =
        count_sites(MODEL_RS, "env_flag(\"") + count_sites(LINEAR_RS, "env_flag(\"");
    let ablated_sites = count_sites(MODEL_RS, "ablated(\"");
    let ablated_calls_per_step = ablated_sites as u64 * LAYERS as u64 + 1; // +1 head_gemv

    // ---- measurements ------------------------------------------------
    // Nine independent estimates each, and the *median* of the nine is what
    // feeds the table, so the printed ms carries its own spread.
    let mut label_samples = nine_estimates(|| {
        best_of_ns(REPORT_ITERATIONS, |j| stamp_label(j % LAYERS, "q_proj"))
    });
    let label_ns = median(&mut label_samples);
    // `layer07/` alone, for the split between the two allocations. The prefix
    // is 8 bytes of payload ("layer07/"), the full label 13 ("layer07/q_proj"),
    // so the outer concatenation's cost should exceed the inner one's by about
    // the width of the kernel name plus the `String` header difference.
    let (_, prefix_bytes, prefix_calls) = measured(2048, |i| {
        let index = i % LAYERS;
        format!("layer{index:02}/")
    });
    let (_, label_bytes, label_calls) = measured(2048, |i| stamp_label(i % LAYERS, "q_proj"));
    for name in [
        "KAPLUMBA_USE_SK",
        "KAPLUMBA_4COL",
        "KAPLUMBA_NO_2COL",
        "KAPLUMBA_REPACKED_GEMV",
        "KAPLUMBA_NO_NORM_ROPE_CACHE",
        "KAPLUMBA_NO_NORM_ROPE_QKV",
        "KAPLUMBA_SKIP_ATTENTION",
        "KAPLUMBA_NO_SINGLE_PASS",
    ] {
        env_flag_cached(name);
    }
    let mut env_flag_samples = nine_estimates(|| {
        best_of_ns(REPORT_ITERATIONS, |_| env_flag_cached("KAPLUMBA_NO_2COL"))
    });
    let env_flag_ns = median(&mut env_flag_samples);
    let mut ablated_samples = nine_estimates(|| {
        best_of_ns(REPORT_ITERATIONS, |_| ablated_as_shipped("o_proj_gemv"))
    });
    let ablated_ns = median(&mut ablated_samples);
    let ablated_fixed_ns = best_of_ns(REPORT_ITERATIONS, |_| ablated_fixed("o_proj_gemv"));

    // `encode_gemv_raw` runs once for o_proj and once for down_proj per layer,
    // plus once for lm_head; each takes the 2col arm and so builds the kernel
    // name exactly once (linear.rs:2135).
    let gemv_raw_per_step = (2 * LAYERS + 1) as u64;
    // Built once, outside the timed region: constructing the table is the point of
    // the map existing, not part of the per-dispatch cost being priced.
    let gemv_name_map: std::collections::HashMap<String, u32> = (1..=8)
        .map(|b| (format!("linear_gemv_ishizuki_k{b}_colmajor_2col"), b))
        .collect();
    let gemv_name_probe = |_j: usize| {
        let bits = 3u8;
        let name = format!("linear_gemv_ishizuki_k{bits}_colmajor_2col");
        std::hint::black_box(gemv_name_map.get(&name).copied())
    };
    let mut gemv_name_samples =
        nine_estimates(|| best_of_ns(REPORT_ITERATIONS, gemv_name_probe));
    let gemv_name_ns = median(&mut gemv_name_samples);
    let (_, gemv_name_bytes, _) = measured(2048, gemv_name_probe);

    // ---- A. host DATA traffic: the only model bytes the host touches ----
    let u32_bytes = std::mem::size_of::<u32>() as u64;
    let host_data_bytes = 3 * u32_bytes;
    print_table(
        "A. HOST DATA per token (model bytes crossing the CPU boundary)",
        "bytes",
        &[
            Row { what: "token id -> token_ids_decode[slot]", count: 1, bytes: u32_bytes, where_: "model.rs:3763" },
            Row { what: "position -> query_positions_decode[slot]", count: 1, bytes: u32_bytes, where_: "model.rs:3768" },
            Row { what: "argmax result <- host_tokens[step_idx]", count: 1, bytes: u32_bytes, where_: "model.rs:3796" },
            Row { what: "sign vectors read by the host", count: 0, bytes: 0, where_: "mirrors only: model.rs:2506" },
            Row { what: "logits row read by the host (pipelined)", count: 0, bytes: 0, where_: "GPU argmax: model.rs:4469" },
            Row { what: "weight bytes read by the host", count: 0, bytes: 0, where_: "GPU streams them" },
            Row { what: "host token/position staging in forward_chunk", count: 0, bytes: 0, where_: "gated on tokens>1: model.rs:4073" },
        ],
        host_data_bytes,
    );

    // ---- B. host ALLOCATION traffic: what the CPU asks the allocator for ----
    let label_bytes_per_step = layer_stamps_per_step as u64 * label_bytes;
    let reachable_label_bytes_per_step = reachable_stamps_per_step as u64 * label_bytes;
    let gemv_name_bytes_per_step = gemv_raw_per_step * gemv_name_bytes;
    let host_alloc_bytes = label_bytes_per_step + gemv_name_bytes_per_step;
    let reachable_alloc_bytes = reachable_label_bytes_per_step + gemv_name_bytes_per_step;
    let inflated = host_alloc_bytes as f64 / reachable_alloc_bytes as f64;
    print_table(
        "B. HOST ALLOCATION per token (heap the CPU asks for, then frees)",
        "bytes",
        &[
            Row {
                what: "GpuProfile::stamp labels [REACHABLE arms]",
                count: reachable_stamps_per_step as u64,
                bytes: reachable_label_bytes_per_step,
                where_: "10 boundaries/layer, model.rs",
            },
            Row {
                what: "GpuProfile::stamp labels [all sites, UPPER BOUND]",
                count: layer_stamps_per_step as u64,
                bytes: label_bytes_per_step,
                where_: "every label builder",
            },
            Row {
                what: "2col kernel name in encode_gemv_raw",
                count: gemv_raw_per_step,
                bytes: gemv_name_bytes_per_step,
                where_: "linear.rs:2135",
            },
        ],
        host_alloc_bytes,
    );
    println!(
        "  reachable-arm total {reachable_alloc_bytes} B (the first row); the \
         UPPER BOUND row assumes every mutually-exclusive arm fires and is \
         inflated {inflated:.1}x."
    );
    println!(
        "  (measured: {label_bytes} B / {label_calls} allocs per stamp label, of \
         which {prefix_bytes} B / {prefix_calls} are the `prefix()` half; \
         {gemv_name_bytes} B per kernel name)"
    );

    // ---- C. host CPU per token ----
    // `env_flag` call sites reached per step, lower-bounded by the census: two
    // inside `encode_gemv_raw` per matvec (KAPLUMBA_4COL, KAPLUMBA_NO_2COL), one
    // in `encode_gemv` for the head (KAPLUMBA_USE_SK), and three in `forward_buffer`
    // per layer that no branch can skip (KAPLUMBA_LINEARS_ONLY,
    // KAPLUMBA_SKIP_ATTENTION, and one of the NORM_ROPE pair).
    let env_flag_min = gemv_raw_per_step + 3 * LAYERS as u64;
    let label_ms = reachable_stamps_per_step as f64 * label_ns / 1e6;
    let label_ms_upper = layer_stamps_per_step as f64 * label_ns / 1e6;
    let gemv_name_ms = gemv_raw_per_step as f64 * gemv_name_ns / 1e6;
    let ablated_ms = ablated_calls_per_step as f64 * ablated_ns / 1e6;
    let env_flag_ms = env_flag_min as f64 * env_flag_ns / 1e6;
    let host_ms = label_ms + gemv_name_ms + ablated_ms + env_flag_ms;
    let host_ms_upper = label_ms_upper + gemv_name_ms + ablated_ms + env_flag_ms;
    println!(
        "\n--- C. HOST CPU per token (measured, best-of-9 samples, no GPU) {}",
        "=".repeat(23)
    );
    println!("  {:<50} {:>10} {:>12}  {}", "item", "count", "ms", "note");
    println!("  {}", "-".repeat(96));
    for (what, count, ms, note) in [
        (
            "GpuProfile::stamp labels [reachable]",
            reachable_stamps_per_step as u64,
            label_ms,
            format!("{label_ns:.0} ns x {reachable_stamps_per_step} boundaries"),
        ),
        (
            "2col kernel name (linear.rs:2135)",
            gemv_raw_per_step,
            gemv_name_ms,
            format!("{gemv_name_ns:.0} ns x {gemv_raw_per_step} matvecs"),
        ),
        (
            "ablated() re-reading KAPLUMBA_ABLATE",
            ablated_calls_per_step,
            ablated_ms,
            format!("{ablated_ns:.0} ns x {ablated_calls_per_step} (vs {ablated_fixed_ns:.0} ns resolved)"),
        ),
        (
            "env_flag() cached mutex+String hash",
            env_flag_min,
            env_flag_ms,
            format!("{env_flag_ns:.0} ns x >={env_flag_min} (lower bound)"),
        ),
    ] {
        println!("  {what:<50} {count:>10} {ms:>12.4}  {note}");
    }
    println!("  {}", "-".repeat(96));
    println!(
        "  {:<50} {:>10} {:>12.4}  {}",
        "TOTAL [reachable arms]",
        "",
        host_ms,
        format!("{:.2} % of a {MS_PER_STEP_BASELINE} ms step", 100.0 * host_ms / MS_PER_STEP_BASELINE)
    );
    println!(
        "  {:<50} {:>10} {:>12.4}  {}",
        "TOTAL [all label sites, upper bound]",
        "",
        host_ms_upper,
        format!("{:.2} %", 100.0 * host_ms_upper / MS_PER_STEP_BASELINE)
    );
    // Spread of the total, from the spread of the nine estimates that compose it.
    let spread = |samples: &[f64; 9]| {
        let min = samples.iter().copied().min_by(f64::total_cmp).unwrap();
        let max = samples.iter().copied().max_by(f64::total_cmp).unwrap();
        (min, max)
    };
    let (_, label_max) = spread(&label_samples);
    let (_, env_max) = spread(&env_flag_samples);
    let (_, abl_max) = spread(&ablated_samples);
    let (_, gemv_max) = spread(&gemv_name_samples);
    let host_max = reachable_stamps_per_step as f64 * label_max / 1e6
        + gemv_raw_per_step as f64 * gemv_max / 1e6
        + ablated_calls_per_step as f64 * abl_max / 1e6
        + env_flag_min as f64 * env_max / 1e6;
    println!(
        "  per-call spread (min..max of 9 estimates of the reachable total): \
         {:.3} .. {:.3} ms",
        host_ms.min(host_max),
        host_ms.max(host_max)
    );

    // ---- D. the arithmetic ----
    // The reachable figures are the ones to quote; the upper bound is printed so
    // a reader who prefers to believe every site fires can see that case too.
    let host_bytes = reachable_alloc_bytes + host_data_bytes;
    let ratio = GPU_BYTES_PER_TOKEN as f64 / host_bytes as f64;
    let gb_s = 193e9_f64;
    let needed_bytes = host_ms * 1e-3 * gb_s;
    let share_upper = 100.0 * host_ms_upper / MS_PER_STEP_BASELINE;
    let gpu_ms = MS_PER_STEP_BASELINE - host_ms;
    println!("\n--- D. THE ARITHMETIC {}", "=".repeat(57));
    println!("  GPU bytes per token                    {:.3} GB", GPU_BYTES_PER_TOKEN as f64 / 1e9);
    println!("  host bytes per token                   {host_bytes} B  ({:.4} MB)", host_bytes as f64 / 1e6);
    println!("  ratio                                  1 : {ratio:.0}");
    println!(
        "  bandwidth-bound test: if the host's {host_ms:.2} ms of CPU were DRAM-limited, at\n\
          193 GB/s it would have to move {needed_bytes:.0} B. The tables above account for\n\
          {host_bytes} B, i.e. {ratio:.0}x less. **Host traffic cannot be the host gap.**"
    );
    println!(
        "  Counting every label site instead of only the reachable arms, host CPU\n\
          rises to {host_ms_upper:.3} ms -- still {share_upper:.2} % of a \
         {MS_PER_STEP_BASELINE} ms step. The other\n\
          {gpu_ms:.1} ms is the GPU's own 12.04 GB stream plus the dispatch-encode\n\
          and wait-wakeup costs that are not traffic at all."
    );

    // ---- E. capacity: costs no bandwidth, but competes for 24 GiB ----
    print_table(
        "E. CAPACITY per token: resident, zero traffic, but on the same 24 GiB",
        "bytes",
        &[
            Row {
                what: "sign vectors: Vec<u16> + Shared mirror",
                count: 1,
                bytes: model_sign_bytes(),
                where_: "model.rs:313-314, 452",
            },
            Row {
                what: "  ... the host Vec<u16> half of that",
                count: 1,
                bytes: model_sign_bytes() / 2,
                where_: "model.rs:313-314",
            },
            Row {
                what: "  ... the sharded file pages (3rd copy)",
                count: 1,
                bytes: model_sign_bytes() / 2,
                where_: "weights.rs mmap",
            },
        ],
        model_sign_bytes() * 2,
    );
    println!(
        "  {:.2} % of 24 GiB. Dropping the host Vec<u16> frees {:.2} MB of resident\n\
         memory. It buys ZERO bandwidth: nothing reads it per token.",
        100.0 * model_sign_bytes() as f64 / 24e9,
        model_sign_bytes() as f64 / 2.0 / 1e6
    );

    // ---- F. census summary + what is NOT on the path ----
    println!("\n--- F. CENSUS {} {}", "=".repeat(61), "");
    println!("  env_flag call sites in model.rs + linear.rs     {env_flag_sites}");
    println!(
        "  dispatches per step                            {DISPATCHES_PER_STEP_CURRENT} current ({DISPATCHES_PER_STEP_PRE_FUSION} pre-fusion)"
    );
    println!("  layer label builders (eager format! pair)       {layer_labels}  (x{LAYERS} layers = {layer_stamps_per_step} boundaries/step, UPPER BOUND)");
    println!("  ... of which reachable in one layer            {REACHABLE_LABELS_PER_LAYER}  (x{LAYERS} layers = {reachable_stamps_per_step}/step, the figure to quote)");
    println!("  literal-label stamps (no allocation)           {literal_stamps}");
    println!("  ablated() call sites in model.rs                {ablated_sites}  (x{LAYERS} + 1 head = {ablated_calls_per_step}/step)");
    println!("  env_flag() calls reaching the encoder, per step  >={env_flag_min}");
    println!("\n--- G. NOT ON THE PATH {} {}", "=".repeat(57), "");
    println!("  * SPIN_THREADS ({SPIN_THREADS}) is a GPU threadgroup width for the");
    println!("    KAPLUMBA_DECODE_INJECT burn kernel, not a CPU thread count. The");
    println!("    variable is unset in production, spin_scratch stays None (model.rs:4521),");
    println!("    and NO CPU thread spins or polls memory during decode.");
    println!("  * OpTimer::lap IS gated: `if self.on` precedes `Instant::now()` (model.rs:594).");
    println!("    With KAPLUMBA_PROFILE_PHASES unset: one predictable branch per boundary.");
    println!("  * GpuProfile::stamp is NOT gated: its `format!` argument is evaluated at");
    println!("    the call site, before the callee sees `enabled`. That is the defect.");
    println!("  * The decode argmax runs on the GPU and writes one u32; the 262,144-wide");
    println!("    logits row is never read by the host on the pipelined path (softcap");
    println!("    skipped when next_token_out/host_token_out are set, model.rs:4444).");
    println!("  * wait_until_completed: once per step at the step boundary.");
    println!("    spec/decode-19tok records the wake-up as 0.176-0.203 ms, flat in");
    println!("    step size, so it is latency, not bandwidth.");
    println!("  * read_at() during load: irrelevant. Decode re-reads none of the file.");
    println!("  * Host encode OVERLAPS GPU streaming: decode rolls at 50 dispatches per");
    println!("    buffer (encoder.rs:221, decode_max_dispatches_per_buffer), and");
    println!("    generate.rs commits step N+1 before waiting on step N. So table C is");
    println!("    not additive wall clock; it is a tax on reaching the step barrier,");
    println!("    plus a small amount of contention on a DRAM with {ratio:.0}x headroom.");
    println!("\n{}\n", "=".repeat(96));

    // Invariants the table rests on. These are counts of sites in the source,
    // not timings, so they can be exact.
    assert!(layer_labels >= 8, "layer label builders fell to {layer_labels}");
    assert_eq!(layer_stamps_per_step, layer_labels * LAYERS);
    assert!(
        layer_labels >= REACHABLE_LABELS_PER_LAYER,
        "the raw label count ({layer_labels}) is below the reachable figure \
         ({REACHABLE_LABELS_PER_LAYER}); REACHABLE_LABELS_PER_LAYER's derivation \
         is stale"
    );
    assert!(env_flag_sites >= 40, "env_flag census fell to {env_flag_sites} sites");
    assert!(ablated_sites >= 3, "ablated census fell to {ablated_sites} sites");
    assert!(
        host_ms < MS_PER_STEP_BASELINE * 0.05,
        "host CPU {host_ms:.2} ms is over 5 % of a step; that would change the \
         recommendation this report supports"
    );
    // The conclusion this file exists to support is that host traffic is
    // negligible next to the GPU's 12.04 GB. Four orders of magnitude is the
    // bar: below it, host traffic would be a real fraction of the step and the
    // recommendation (attack instruction count, not bandwidth) would be wrong.
    assert!(
        ratio > 1e4,
        "host/GPU byte ratio is only {ratio:.0}:1; host traffic would no longer \
         be negligible and this file's conclusion is void"
    );
}
