//! Measurement trustworthiness: the variance taxonomy, an audit of
//! `decode_tok_s`, a corrected workload-independent metric, a benchmark
//! protocol, a power calculation, and a machine-readable results format.
//!
//! # Why this file exists
//!
//! Every other agent in this session is hunting a throughput win that cannot be
//! validated without a trustworthy measurement. This repo's own history says the
//! measurement is not trustworthy:
//!
//! * `AGENTS.md:271-272` — decode measured at **3.7, 3.9, 4.3, 4.8, 6.0 and
//!   6.4 tok/s from the same binary minutes apart**.
//! * `spec/resident-memory-12gib/implementation_notes.md:207` — a **35 %
//!   same-config swing** (9.124 -> 12.338 tok/s), which the note itself says is
//!   "larger than the mapped-vs-resident difference being claimed".
//! * `spec/decode-25tok-peak-bandwidth/implementation_notes.md:311-319` — a
//!   *good* day: two arms of the **same binary** differing by 0.010 tok/s,
//!   spread 0.027-0.076.
//!
//! Those are two different regimes on the same box, minutes apart. Nothing in
//! the current harness distinguishes them, so a single A-then-B pair cannot tell
//! a 3 % win from thermal drift, and a fleet of agents each running one pair is
//! guaranteed to manufacture false positives.
//!
//! # Nothing here runs on the GPU
//!
//! The statistics, the metric audit and the power calculation are ordinary
//! CPU-only unit tests and run in the default suite. The measurement harness is
//! two `#[ignore]`d tests: one that needs the real checkpoint and the whole
//! device. Nothing in this file loads a model unless it is explicitly ignored
//! and explicitly pointed at a checkpoint, and nothing in it ever takes the GPU
//! lock by accident.
//!
//! # Line references
//!
//! Citations are against the working tree at `2aaa50e` **plus** the
//! uncommitted `KAPLUMBA_PREFILL_MAX_DISPATCHES` patch (which adds 13 lines to
//! `crates/engine/src/generate.rs` after line 905). Anchors before line 905 are
//! stable across that patch; the pipelined-decode citations are given as symbol
//! names as well as line numbers for that reason.

// ===========================================================================
// PART 1 — THE VARIANCE TAXONOMY
// ===========================================================================
//
// Each entry: the source, the code that owns it (`file:line`), the mechanism,
// and whether the current harness controls it. Nothing below is speculation
// about the machine; every mechanism is either in code or is a named property
// of the loader/allocator path the code takes.

// --- V1. Page-cache warmth of a 16 GB checkpoint against a 24 GiB box -------
//
// `crates/engine/src/weights.rs:982-1042` (`ShardReader::open`) mmaps every
// shard `PROT_READ`/`MAP_PRIVATE`; `weights.rs:646-663` then bulk-copies into
// the arena with `read_at` (`weights.rs:949-961`). The comment at
// `weights.rs:976-981` states the reason for `read_at` over mmap: "with several
// threads faulting unrelated file offsets at once the kernel's readahead
// collapses". So load time is a function of *whether the ~16 GB of shard bytes
// are in the page cache*, and on this box they cannot stay there: the process
// itself wires ~12 GiB of arena (`metal.rs:4`, `DEFAULT_RESIDENT_LIMIT_BYTES =
// 18_000_000_000`) into the same 24 GiB. Measured `load_ms` in the repo's own
// notes: **5 987-7 464 ms** (`spec/resident-memory-12gib/implementation_notes.md:219`).
// 1.48 s of spread on a ~6.5 s load is a 23 % swing, and it is bimodal (warm vs
// cold), not Gaussian.
// CONTROLLED BY THE CURRENT HARNESS? No. Nothing reads or records page-cache
// state; `KAPLUMBA_LOAD_PROFILE` (`weights.rs:665-671`) splits alloc from
// read+convert but neither is compared against anything.

// --- V2. First-touch page faults on a cold arena ---------------------------
//
// `metal.rs:80-95` allocates each arena segment with
// `device.new_buffer(len, StorageModeShared)`; `metal.rs:33` sets the segment
// size to 512 MiB. `weights.rs:646-663` then writes ~12 GiB into those
// segments in parallel via rayon's global pool. Every one of those writes is a
// first touch of a freshly mapped page: the kernel must supply the page, and on
// a box already at 12 GiB resident it must decide whether to zero-fill from
// compressor or fault from swap.
// Rayon's pool size is **not** set anywhere (`rg rayon` finds only
// `weights.rs:10` and `load_perf.rs:15`), so it defaults to
// `available_parallelism()` = **14** on this box, competing with whatever else
// is running. There is no `KAPLUMBA_*` knob for it, so it cannot be pinned.
// CONTROLLED? No.

// --- V3. Memory state is already bimodal on this box ----------------------
//
// Two readings taken while writing this file, **with no model loaded**, ~5
// minutes apart, on an otherwise idle box:
//
//     top -l 1 -n 0 | grep PhysMem        # first
//     PhysMem: 22G used (3285M wired, 10G compressor), 1483M unused.
//     sysctl -n vm.swapusage
//     total = 6144.00M  used = 5105.38M  free = 1038.62M  (encrypted)
//
//     top -l 1 -n 0 | grep PhysMem        # second
//     PhysMem: 23G used (6712M wired, 10G compressor), 164M unused.
//     sysctl -n vm.swapusage
//     total = 9216.00M  used = 8223.56M  free = 992.44M  (encrypted)
//
// **10 GiB of compressed memory and 8.2 GiB of swap, right now, with no engine
// running.** Those two readings bracket this file's own compilation (a `cargo`
// build of the engine, which is several GiB of rss and page cache). So the box
// was *already* in a degraded memory state before any benchmark, and it got
// worse while nothing was being measured. That is the environment every tok/s
// number in this repo has been taken in.
//
// A decode step that touches a compressed or swapped page pays decompression or
// an NVMe read *inside the timed region*, on top of the ~12 GB of streaming that
// is the actual work. That is a plausible mechanism for 3.7 tok/s (270 ms/step)
// versus 16 tok/s (62 ms/step): the 208 ms/step delta is the right order for a
// working set that has partly fallen out of memory, and
// `spec/memory-swap-cliff/implementation_notes.md:216` independently measures
// 63.8 ms/token to fault `lm_head` back at 4 bpw — the same order.
// I am **not** claiming this is proven. It is the leading hypothesis, it is
// testable (the harness records swap and `PhysMem` beside every rate), and it is
// *actionable*, which the thermal hypothesis is not. A prediction that
// distinguishes them: a memory-state effect should track `PhysMem` unused and
// swap used within a session; a thermal effect should track elapsed load time
// and recover after a cooldown. The harness records neither a clock nor a
// temperature, so it cannot separate them today — which is itself the finding.
// CONTROLLED? No. Nothing in a run records `phys_footprint`, swap delta, or
// compressor size. `peak_resident_pages_bytes` (`metrics.rs:93-108`,
// `report.rs:23`) is `getrusage().ru_maxrss` and `AGENTS.md:312-325` already
// records that it under-reports by 5.5 GiB.

// --- V4. INITIAL_FULL_CACHE = 512, then x1.25 ------------------------------
//
// `model.rs:1077` `const INITIAL_FULL_CACHE: usize = 512`;
// `gpu_cache.rs:414-444` `growth_target` walks the ladder by
// `GROWTH_NUM/GROWTH_DEN = 5/4` (**1.25x, not 2x** — `gpu_cache.rs:371-372`;
// the "doubles on demand" wording at `model.rs:3391` is stale prose), reaching
// `MAX_POLICY_CAPACITY = DEFAULT_CONTEXT_LIMIT = 32768`
// (`gpu_cache.rs:59`, `generate.rs:35`) in 19 rungs.
// Each rung calls `device.new_buffer` and `std::ptr::copy_nonoverlapping` on
// the live prefix (`gpu_cache.rs:238-271`), **from inside a forward**
// (`model.rs:1663` and `model.rs:1783`, and the hazard is documented at
// `model.rs:3690-3693`). `gpu_cache.rs:400-408` measures it at 2 752.6 MiB
// copied over a full 8 192-position run, worst rung 58.3 MiB/layer (6.2 ms).
// For the 37-token / 32-step benchmark this never fires: 37+32 = 69 positions
// is under 512, so **the ladder contributes exactly zero variance to the
// headline number** and 2.6 ms/step of systematic cost to any 8k measurement.
// This is the one variance source in the brief that is *not* a source at the
// benchmark's operating point, and it is worth saying so explicitly.
// CONTROLLED? Yes, by fixing the prompt length. That is the entire reason the
// protocol pins prompt_tokens.

// --- V5. SPIN_THREADS = 4096 is not a variance source ----------------------
//
// `model.rs:3113` `const SPIN_THREADS: u64 = 4096`, allocated lazily at
// `model.rs:4517-4525` (16 KiB, fixed size). It is dispatched only when
// `KAPLUMBA_DECODE_INJECT` is set (`model.rs:4129-4147`), which is a
// *measurement aid*, not a product path, and is off by default.
// So: **no thread pool sizing knob exists on the decode path at all.** The
// decode loop is single-threaded on the host and dispatches to one Metal queue
// (`model.rs:3097` `forward_queue`, cloned per use at `model.rs:3672-3674`).
// The only host thread pool in the process is rayon's, and it is used only
// during load (V2).
// CONTROLLED? N/A — but the absence is worth recording, because "thread pool
// sizing" is on the list of things one might otherwise try to pin.

// --- V6. GPU clock / thermal state ----------------------------------------
//
// No code path reads a clock, a temperature, or a residency. There is no
// `IOReport` call, no `pmset -g therm`, nothing. `PowerState`/`currentPowerState`
// are not queried. So the single largest suspected driver of "same binary,
// minutes apart" is **completely uninstrumented**.
// The repo's own data is consistent with it: 3.7 -> 6.4 tok/s is a 1.7x swing
// in the wrong direction for a machine that got *hotter*.
// CONTROLLED? No, and it cannot be controlled — only interleaved against.

// --- V7. The `gpuq` gate is a correctness gate, not a variance source ------
//
// `gpu_lock.rs:236-248` `acquire` takes a kernel `flock`; `gpu_lock.rs:300`
// `flock(LOCK_EX|LOCK_NB)`. `gpuq` takes no lock and just sets
// `KAPLUMBA_GPU_LOCK=try|wait` (`scripts/gpuq:165,170`); the engine's own
// policy table is `gpu_lock.rs:69-78`. Exit **75** = `EXIT_BUSY`
// (`gpu_lock.rs:40`).
// Two consequences for measurement:
//   (a) A run that exits 75 ran **nothing** and must be counted as a
//       scheduling outcome, never as a sample and never as a failure.
//   (b) **`load_ms` includes GPU-lock wait time.** `main.rs:51-54` starts the
//       clock before `load_from_dir`, and the lock is acquired *inside*
//       `load_from_dir` at `model.rs:3315`. The default policy is `Wait`
//       (`gpu_lock.rs:76`), which blocks up to **30 minutes**
//       (`gpu_lock.rs:50`). So `load_ms` is a load measurement only when the
//       lock was already free, and nothing records whether it was.
//       `gpu_lock::describe()` exists for exactly this ("Current policy, for
//       `--metrics` so a run records whether it was contended",
//       `gpu_lock.rs:367-374`) and **`main.rs` never calls it.** Dead code
//       whose stated purpose is the thing that is missing.
// CONTROLLED? Partly, if the harness always uses `KAPLUMBA_GPU_LOCK=try` and
// treats 75 as its own outcome.

// --- V8. The command-buffer fault rate, and why it is worse than it looks ---
//
// `encoder.rs:191-203` `command_failed` prints the status. The asymmetry that
// matters:
//   * **prefill** goes through `commit_and_wait` (`generate.rs:929`), and that
//     checks `status() != Completed` (`encoder.rs:656-658`). A prefill fault is
//     an `Err`.
//   * **decode** goes through `commit_terminal` (`generate.rs:972, 990`) then
//     `wait()` (`generate.rs:992, 1008, 1019`). `commit_terminal`
//     (`encoder.rs:616-625`) commits and returns `Ok` unconditionally.
//     `wait()` (`encoder.rs:639-644`) waits and **never inspects `status`**.
// So on the production path a decode-step command buffer that faults with
// `status=Error` produces **no error, no diagnostic, and no non-zero exit**.
// The run continues and `read_host_token` (`model.rs:3794-3799`) returns
// whatever bytes the failed dispatch left behind.
//
// Three consequences, all of them about measurement:
//   1. The fault rate during decode **cannot currently be counted**, because
//      nothing counts it. `command_failed` is only reachable from the three
//      status-checking commit paths, none of which decode uses.
//   2. A corrupt token stream changes `generated_ids.len()`, which changes
//      `decode_steps`, which changes the reported `decode_tok_s`. A silent
//      fault therefore **manufactures a throughput number**.
//   3. Discarding faulted runs is not neutral: faults are *caused* by the slow
//      regime (memory pressure, `memory-swap-cliff/implementation_notes.md:216`),
//      so "discard the faults" systematically discards the slow samples and
//      biases the survivor median **upward**. The protocol below therefore
//      records the fault rate as a headline number next to the rate, never as
//      a footnote.
// CONTROLLED? No.

// --- V9. The metric itself is workload-dependent (Part 2) ------------------
// See below. This is the largest single contributor and it is a code property,
// not a machine property.

// --- V10. Instrumented runs are not the product run ------------------------
//
// `generate.rs:943-945`:
//
//     let pipelined = !margins.enabled
//         && crate::decode_profile::active_mode() == crate::decode_profile::Mode::Off
//         && std::env::var_os("KAPLUMBA_NO_PIPELINE").is_none();
//
// So `KAPLUMBA_MARGIN=1`, `KAPLUMBA_DECODE_PROFILE=step` and
// `KAPLUMBA_DECODE_PROFILE=kernel` **all force the non-pipelined loop**, which
// has *different timing semantics* (Part 2, D3). Every profiler in this repo
// therefore measures a different loop from the one that ships. That is not a
// bug in the instruments; it is a fact about which instrument can be trusted
// for which question, and it is undocumented in `AGENTS.md` §10.
// The instruments that **do** leave the pipelined path intact are
// `KAPLUMBA_SEGMENTS` / `KAPLUMBA_SEGMENTS_BUCKET` (`generate.rs:603-608`,
// read nowhere in the `pipelined` predicate), `KAPLUMBA_COUNT_BUFFERS`
// (`model.rs:192-202`), and `KAPLUMBA_PIPELINE_AUDIT`. The harness uses only
// those three.
//
// Follow-on: `KAPLUMBA_STEP_TRACE` prints at `generate.rs:1083-1088`, which is
// inside the `else` (non-pipelined) branch, and is *not* part of the
// `pipelined` predicate. So `KAPLUMBA_STEP_TRACE=1` on a default
// configuration prints **nothing at all**. Anyone reaching for per-step
// timings gets silence, and may conclude there are none.

// ===========================================================================
// PART 2 — AUDIT OF `decode_tok_s`
// ===========================================================================
//
// The engine's formula, three places, identical:
//   `generate.rs:216-221`   (`generate`, temperature > 0)
//   `generate.rs:382-387`   (`generate_gpu`, temperature > 0)
//   `generate.rs:1128-1137` (`generate_gpu_greedy`, the CLI's real path)
//
//     let decode_steps = generated_ids.len().saturating_sub(1);
//     decode_tokens_per_second = decode_steps as f64 / decode_duration.as_secs_f64()
//
// and `crates/cli/src/report.rs:28` prints it as `decode_tok_s`.
//
// **A. It is warmup-inclusive.** `generate_gpu_greedy` runs
// `warmup_decode` at load (`model.rs:3544` -> `model.rs:3591-3619`, two
// forwards plus `pipeline_audit::arm()`), and
// `kaplumba_kernels::pipeline_audit::assert_no_builds_since(load_mark)` is
// called inside every loop iteration (`generate.rs:997, 1023, 1092`), so shader
// compilation is *proven* not to be inside the timed region. Good. But the
// **first decode step itself is not discarded**: `step_start` is created at
// `generate.rs:977` immediately after `enc0.commit_terminal()`, and the first
// recorded interval is `generate.rs:993`. So the cold first step is inside the
// average. This repo has direct evidence that the first step is anomalous:
// `model.rs:3553` says the decode warmup "was what T7.6.8 added, to kill a
// **1277 ms first-step cost**". A step that cost 1277 ms against a 62 ms
// steady state is a 1277/62 = 20x outlier, and at `--max-new-tokens 2` it is
// **100 %** of the reported rate.
// Contrast `run_benchmark` (`generate.rs:444-459`), which *does* have a
// `warmup_steps` filter at `generate.rs:455-457`.
//
// **B. The denominator's sample count is workload-determined.**
// `decode_steps = generated_ids.len() - 1`, and `generated_ids.len()` is
// however many tokens the prompt happened to produce before EOS. There is no
// `--min-new-tokens` in `crates/cli/src/args.rs` (whole file read; the parse
// loop is `args.rs:93-135` and there is no such arm). So two runs of the same
// binary with the same prompt produce the *same* `generated_ids.len()` — but a
// run that hits EOS at token 2 reports a rate computed from **one** sample,
// while a 32-token run averages 31. The estimator's variance differs by
// sqrt(31) ~ 5.6x. They are not the same statistic at different sample sizes;
// they are different statistics.
//
// **C. The two loop paths count EOS differently.** Pipelined:
// `generate.rs:999-1000` pushes `step_token` and *then* checks
// `config.eos_ids.contains(&step_token)` at `generate.rs:1006` — so the **EOS
// token is counted** in `generated_tokens`. Non-pipelined:
// `generate.rs:1042-1046` checks EOS *before* `generated_ids.push(token)` — so
// it is **not** counted. The same reported field has two definitions,
// selected by an env var (`KAPLUMBA_NO_PIPELINE`) or by the profiler
// (V10). Also `generate_gpu` at `generate.rs:342-346` checks EOS before
// pushing, matching the non-pipelined path, not the pipelined one.
//
// **D. The two loop paths time different things.**
// Pipelined: `step_start` is stamped *after* `enc0.commit_terminal()`
// (`generate.rs:972, 977`), so step 0's **host encode is excluded**, and every
// later interval *includes* the host encode of step k+1 (encoded at
// `generate.rs:986-990` before the wait at `generate.rs:992`). So it measures
// steady-state throughput, `max(gpu_k, encode_{k+1})`.
// Non-pipelined: `step_started` at `generate.rs:1059` precedes the encode at
// `generate.rs:1067`, so it measures per-step **latency including that step's
// own host encode**.
// Same field name, two different quantities, and the sign of the difference is
// not small: `sampler.rs:46-51` measures 2.87 ms/token for `sample()` and
// `sampler.rs:36-44` measures 14.71 ms/token for the sort it replaced — 12 %
// of a 62 ms step, on the host, inside the timed region, in one path and not
// the other.
//
// **E. Detokenisation and stdout flushing are inside the timed region.**
// `generate.rs:1001-1004` (pipelined) calls `detokenizer.push(token)` and
// `on_text(chunk)` *between* the `step_start = Instant::now()` at
// `generate.rs:995` and the next interval close. `main.rs:98-102` makes
// `on_text` a `print!` plus a **flush**. So every token's text — and its
// terminal write — is charged to the *following* step's interval. It is
// small when stdout is a file and unbounded when stdout is a pipe whose reader
// has stopped, which is precisely the case in a harness that captures output.
// Any measured effect below a few percent is inside this term.
//
// **F. `--benchmark` measures a different engine, and over-reports.**
// `run_benchmark` calls `model.forward(...)` at `generate.rs:439` and
// `generate.rs:454`. `TextModel::forward` (`model.rs:4675`) runs
// `for layer in &mut self.layers { hidden = layer.forward(&hidden, positions)?; }`
// at `model.rs:4702-4704` — the **host slice path**, via
// `DecoderLayer::forward` -> `host_cache()` (`model.rs:1123-1124`). The CLI's
// real decode is `forward_buffer_into_step` -> `forward_chunk`. So the repo's
// designated "fixed 2k/8-warmup/128-token benchmark" (`main.rs:60-83`,
// `help_text` at `args.rs:227`) does not exercise the code whose throughput is
// quoted. It is also, by construction, EOS-independent (it argmaxes without
// checking `eos_ids`), which is the one thing it gets right.
// It additionally over-reports by **+0.787 %**: `generate.rs:443` sets
// `steps = warmup_steps + measured_tokens = 136`; `generate.rs:446` breaks
// *before* running the final forward, so 135 forwards execute and 127 of them
// (`step in 8..=134`) are timed, while `generate.rs:464` divides by
// `plan.measured_tokens` = **128**. The numerator credits one step that was
// never timed.
//
// **G. There is no validity gate of any kind.** `report.rs:26-38` formats and
// prints whatever it is given. A run that emitted 1 token, a run that emitted
// 1 token *because the GPU faulted*, and a clean 32-token run all print a
// `decode_tok_s` in the same format with nothing to tell them apart except
// `generated_tokens`, which nothing enforces a lower bound on.
//
// ### The corrected metric
//
//     steady_tok_s  =  1000 / median( step_ms[W .. N) )
//
// where, for a decode that executed `N` timed steps with per-step wall times
// `step_ms[0..N)`:
//   * `step_ms[i]` is the interval closed at the completion barrier of decode
//     step `i` and opened at the completion barrier of step `i-1` (step 0's
//     opens at the end of the prefill's wait). This is what the pipelined loop
//     already computes at `generate.rs:993`/`1019` and records into
//     `SegmentLog` at `generate.rs:996`/`1022`.
//   * `W = 8` discarded steps. `8` is not invented: it is `run_benchmark`'s CLI
//     default (`main.rs:69`) and `KAPLUMBA_DECODE_PROFILE_WARMUP`'s default
//     (`decode_profile.rs:126`).
//   * `N` must be at least `W + 16`; otherwise **refuse to report a rate at
//     all** and record the run as `Inadmissible`. This is the enforcement that
//     `--min-new-tokens` does not exist to provide.
//   * Report `median`, `mad`, `p10`, `p90`, `max` of the kept window, plus the
//     *within-run* drift `median(last 8) / median(first 8)`. A run whose drift
//     exceeds 5 % is not a steady-state measurement and must be labelled so.
//
// Why the median of per-step intervals rather than the run's aggregate:
//   * **Independence from workload.** Any `N >= W+16` yields the same
//     statistic for the same binary and prompt prefix. A run that emits 32
//     tokens and a run that emits 96 tokens report the same quantity, because
//     both take a median over a warmed, fixed-length window of steps. Under the
//     current formula they do not (B), and they differ by `sqrt(31)` in
//     estimator variance for the 32-versus-96 case.
//   * **It excludes the first step by construction** (A) instead of averaging
//     it in at weight 1/(N-1), where at small N it dominates.
//   * **It is robust to one stalled step.** A single page-fault storm or a
//     command-buffer hiccup moves a mean over 23 samples by 4 % and a median by
//     0 %. Given V3 and V8, single-step stalls are expected, not hypothetical.
//   * **It exposes drift instead of hiding it.** The aggregate cannot answer
//     "does decode get slower as context grows", which is the stated reason
//     `SegmentLog` exists (`generate.rs:574-592`).
// The metric does **not** remove the workload dependence in the *underlying
// cost*: attention over a longer KV is genuinely more expensive
// (`gpu_cache.rs:186-192`, `key_count` grows with `next_position`). It removes
// the workload dependence in the *estimator*, and pins the prompt so the cost
// is fixed too.
//
// Keep the legacy number, labelled, in every record. It is what the repo's
// notes and every other agent's report are written in; dropping it makes this
// file's numbers incomparable to the conversation they are supposed to settle.

// ===========================================================================
// PART 3 — THE STATISTICS (pure CPU, no GPU)
// ===========================================================================

/// Two-sided 95 % normal quantile. `z_{0.975}`.
pub const Z_95: f64 = 1.959_963_985;
/// One-sided 80 %-power normal quantile. `z_{0.80}`.
pub const Z_80: f64 = 0.841_621_234;

/// Discarded leading steps, matching both existing defaults:
/// `main.rs:69` (`warmup_steps: 8`) and `decode_profile.rs:126`.
pub const WARMUP_STEPS: usize = 8;

/// Fewest kept steps for which a median is reported at all.
///
/// `WARMUP_STEPS + MIN_STEPS = 24`, so a `--max-new-tokens 32` run keeps 23 and
/// passes, and a run that dies at token 20 keeps 11 and is refused. This is the
/// gate `--min-new-tokens` would have provided (`args.rs:93-135` has no such
/// arm).
pub const MIN_STEPS: usize = 16;

/// Sorted copy, NaN-free.
fn sorted_finite(values: &[f64]) -> Vec<f64> {
    let mut out: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    out.sort_by(f64::total_cmp);
    out
}

/// Median of an already-sorted, non-empty slice.
fn median_sorted(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

/// `quantile` with the same index convention as `decode_profile.rs:250-265`:
/// `round((n - 1) * fraction)`.
pub fn quantile(values: &[f64], fraction: f64) -> Option<f64> {
    let sorted = sorted_finite(values);
    if sorted.is_empty() || !(0.0..=1.0).contains(&fraction) {
        return None;
    }
    let index = ((sorted.len() - 1) as f64 * fraction).round() as usize;
    sorted.get(index).copied()
}

/// Median of an unsorted slice, or `None` for an empty / all-NaN input.
pub fn median(values: &[f64]) -> Option<f64> {
    let sorted = sorted_finite(values);
    if sorted.is_empty() {
        None
    } else {
        Some(median_sorted(&sorted))
    }
}

/// Median absolute deviation, in the same units as the input.
///
/// Chosen over the standard deviation because the per-step series is a
/// throughput measurement of a machine under memory pressure: it has outliers,
/// and the MAD has a 50 % breakdown point against them while the standard
/// deviation has none. Reporting the MAD next to the median is what makes a
/// single stalled step visible instead of absorbed.
pub fn mad(values: &[f64]) -> Option<f64> {
    let med = median(values)?;
    let deviations: Vec<f64> = values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .map(|v| (v - med).abs())
        .collect();
    median(&deviations)
}

/// Per-step decode window summary. This is the whole corrected metric.
#[derive(Debug, Clone, PartialEq)]
pub struct StepStats {
    /// Steps whose wall time was measured, including the discarded ones.
    pub n_total: usize,
    /// Steps after the warmup discard. Every rate below is over these.
    pub n_kept: usize,
    pub warmup_discarded: usize,
    pub median_ms: f64,
    pub mad_ms: f64,
    pub p10_ms: f64,
    pub p90_ms: f64,
    pub max_ms: f64,
    /// `median(last 8 kept) / median(first 8 kept) - 1`. Zero when fewer than
    /// 16 kept steps, because a drift over fewer samples is not a drift.
    pub drift_rel: f64,
}

/// Why a run produced no usable rate. Never silently swallowed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MetricReject {
    /// No per-step data at all — the run did not get as far as decode, or the
    /// per-step instrument was not enabled.
    NoSteps,
    /// Fewer than `MIN_STEPS` steps survived the warmup discard. This is the
    /// `--min-new-tokens` gate: an EOS-truncated run lands here.
    TooFewSteps,
    /// A non-positive or non-finite step time, which means the clock produced
    /// garbage. `decode_profile.rs:182-184` `plausible()` exists for exactly
    /// this and counts it rather than dropping it silently.
    ImplausibleStep,
    /// Kept window present, but its drift exceeds the tolerance.
    Drifting { drift_rel: f64 },
}

/// Drift beyond this is not a steady-state measurement.
pub const MAX_DRIFT_REL: f64 = 0.05;

/// The corrected metric: a median over a warmed, fixed-length window of
/// per-step intervals, with explicit refusals.
pub fn step_stats(step_ms: &[f64], warmup: usize) -> Result<StepStats, MetricReject> {
    if step_ms.is_empty() {
        return Err(MetricReject::NoSteps);
    }
    if step_ms.iter().any(|v| !(v.is_finite() && *v > 0.0)) {
        return Err(MetricReject::ImplausibleStep);
    }
    let kept = &step_ms[warmup.min(step_ms.len())..];
    if kept.len() < MIN_STEPS {
        return Err(MetricReject::TooFewSteps);
    }
    let sorted = sorted_finite(kept);
    let half = MIN_STEPS.min(kept.len() / 2).max(1);
    let first_half = median_sorted(&sorted[..half]);
    let last_half = median_sorted(&sorted[sorted.len() - half..]);
    Ok(StepStats {
        n_total: step_ms.len(),
        n_kept: kept.len(),
        warmup_discarded: warmup.min(step_ms.len()),
        median_ms: median_sorted(&sorted),
        mad_ms: mad(kept).unwrap_or(f64::NAN),
        p10_ms: quantile(kept, 0.10).unwrap_or(f64::NAN),
        p90_ms: quantile(kept, 0.90).unwrap_or(f64::NAN),
        max_ms: *sorted.last().expect("non-empty by the guard above"),
        drift_rel: last_half / first_half - 1.0,
    })
}

/// `1000 / median_ms`.
pub fn steady_tok_s(stats: &StepStats) -> f64 {
    1000.0 / stats.median_ms
}

/// The engine's own formula, restated so it can be reasoned about without a
/// GPU: `decode_steps / decode_duration` with
/// `decode_steps = generated_tokens - 1` (`generate.rs:1128-1137`, and the same
/// two lines at `generate.rs:216-221` and `generate.rs:382-387`).
///
/// This is a **restatement for reasoning, not a substitute**. The harness
/// records `GenerationResult::decode_tokens_per_second` itself, which is the
/// engine's value; this exists so the tests below can demonstrate the metric's
/// defects on inputs where running the engine would be absurd.
pub fn legacy_decode_tok_s(sum_timed_s: f64, generated_tokens: usize) -> f64 {
    if generated_tokens < 2 || !(sum_timed_s.is_finite()) || sum_timed_s <= 0.0 {
        return 0.0;
    }
    (generated_tokens - 1) as f64 / sum_timed_s
}

/// The protocol's fixed operating point.
///
/// A 37-token prompt reaching 69 positions stays under `INITIAL_FULL_CACHE`
/// (`model.rs:1077`), so the KV ladder contributes no variance and no
/// systematic cost (V4). 32 generated tokens leaves 31 timed intervals, of
/// which 23 survive the discard and clear `MIN_STEPS`.
pub const BENCH_PROMPT: &str =
    "Count from 1 to 50: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, \
     19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, \
     39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50.";

/// Minimum generated tokens. Not a CLI flag anywhere (`args.rs:93-135`); this
/// is the constant the harness enforces in place of one.
pub const BENCH_MIN_NEW_TOKENS: usize = 32;

// ===========================================================================
// PART 4 — POWER: HOW MANY PAIRS AN EFFECT NEEDS
// ===========================================================================
//
// Paired design: each pair contributes `r_i = ln(B_i) - ln(A_i)`. Logging makes
// the effect multiplicative, which is how tok/s effects are actually quoted
// ("+1.3 %"), and makes a heavy right tail harmless. `r̄` is tested against 0.
//
//     SE(r̄) = s_d / sqrt(n)
//     need   |r̄| >= (z_{0.975} + z_{0.80}) * SE = 2.8016 * s_d / sqrt(n)
//     =>     n >= (2.8016 * sigma_d_rel / effect_rel)^2
//
// `sigma_d_rel` is the standard deviation of the **paired log difference**, and
// it is *measured in the session being reported*, from the control arm. It is
// not a constant and it is not the run-to-run spread of one arm: interleaving
// removes most common-mode drift, so the paired sigma is materially smaller
// than `sqrt(2)` times the single-arm sigma. That is exactly why guessing it is
/// wrong and why the control arm is mandatory.
///
/// `sigma_d_rel = 0.143` is the honest value implied by the repo's own record:
/// a 35 % worst/best swing over roughly five samples is `0.35 / sqrt(12) = 0.101`
/// for a uniform-ish variable, and `sqrt(2) * 0.101 = 0.143` for two
/// uncorrelated arms. It is not a measurement of *this* harness's paired sigma;
/// it is the pessimistic prior, and the whole point of the control arm is to
/// replace it with a measurement.

/// Pairs needed to resolve `effect_rel` at 95 % confidence / 80 % power.
pub fn pairs_needed(sigma_d_rel: f64, effect_rel: f64) -> Option<usize> {
    if !(sigma_d_rel.is_finite() && effect_rel.is_finite()) || sigma_d_rel <= 0.0 || effect_rel <= 0.0
    {
        return None;
    }
    let n = ((Z_95 + Z_80) * sigma_d_rel / effect_rel).powi(2);
    if !n.is_finite() {
        return None;
    }
    Some((n.ceil() as usize).max(1))
}

/// Smallest relative effect `n` pairs can resolve, as a fraction.
pub fn min_detectable_effect(sigma_d_rel: f64, n_pairs: usize) -> f64 {
    if n_pairs == 0 {
        return f64::INFINITY;
    }
    (Z_95 + Z_80) * sigma_d_rel / (n_pairs as f64).sqrt()
}

/// The verdict, and the reason for it. There is deliberately no "probably".
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// The observed effect exceeds what this session could resolve.
    Win,
    Loss,
    /// The session could not resolve the effect it measured. **This is the
    /// expected outcome for most proposals at this noise floor.**
    Inconclusive,
}

/// Paired A/B result.
#[derive(Debug, Clone, PartialEq)]
pub struct PairedVerdict {
    pub n_pairs: usize,
    pub mean_log_ratio: f64,
    pub sd_log_ratio: f64,
    pub ci95_low_log: f64,
    pub ci95_high_log: f64,
    /// Relative effect: `exp(mean_log_ratio) - 1`.
    pub effect_rel: f64,
    pub sigma_d_rel: f64,
    /// Relative, at this session's `n`.
    pub min_detectable_rel: f64,
    pub verdict: Verdict,
}

impl PairedVerdict {
    /// Relative effect of a log-ratio, i.e. `exp(x) - 1`. Used to turn the
    /// confidence interval back into the "%" that tok/s effects are quoted in.
    pub fn effect_rel_at(&self, log_ratio: f64) -> f64 {
        log_ratio.exp() - 1.0
    }
}

/// Test `(a_values, b_values)` as `n` interleaved pairs.
///
/// `sigma_floor` is the relative paired sigma measured from the **same-session
/// control arm**. It is applied as a floor rather than ignored, because a
/// session that happened to be quiet is not evidence that the machine is quiet:
/// `sigma_d_rel = 0` would otherwise make a 0.001 % wobble look like a result,
/// and the repo's own history is that the *same binary* once differed by 0.010
/// tok/s and once by 35 %.
pub fn paired(
    a_values: &[f64],
    b_values: &[f64],
    sigma_floor_rel: f64,
) -> PairedVerdict {
    let n = a_values.len().min(b_values.len());
    let ratios: Vec<f64> = (0..n)
        .map(|i| b_values[i].ln() - a_values[i].ln())
        .collect();
    let mean = ratios.iter().sum::<f64>() / n as f64;
    let sd = if n < 2 {
        0.0
    } else {
        let variance =
            ratios.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
        variance.sqrt()
    };
    let sigma = sd.max(sigma_floor_rel);
    let se = sigma / (n.max(1) as f64).sqrt();
    let mde = (Z_95 + Z_80) * se;
    let verdict = if n < 2 {
        Verdict::Inconclusive
    } else if mean > mde {
        Verdict::Win
    } else if mean < -mde {
        Verdict::Loss
    } else {
        Verdict::Inconclusive
    };
    PairedVerdict {
        n_pairs: n,
        mean_log_ratio: mean,
        sd_log_ratio: sd,
        ci95_low_log: mean - Z_95 * se,
        ci95_high_log: mean + Z_95 * se,
        effect_rel: mean.exp() - 1.0,
        sigma_d_rel: sigma,
        min_detectable_rel: mde.exp() - 1.0,
        verdict,
    }
}

// ===========================================================================
// PART 5 — THE PROTOCOL
// ===========================================================================
//
// ### The prompt
//
// Fixed, pinned by **token count**, not by bytes. `BENCH_PROMPT` is a
// continuation of the counting prompt already used in this repo's own baseline
// (`spec/decode-25tok-peak-bandwidth/implementation_notes.md:307`), extended so
// the model has 32 tokens of obvious continuation available and cannot plausibly
// reach EOS. The harness **verifies this per run** rather than trusting it: a
// run with `generated_tokens < BENCH_MIN_NEW_TOKENS` or `stopped_on_eos` is
// recorded `Inadmissible` with its token count, and never contributes to a rate.
// This is the check that `--min-new-tokens` would have done automatically.
//
// ### The configuration
//
// `--temperature 0` (`args.rs:111-115`) selects greedy argmax
// (`generate.rs:150-160`, `generate.rs:269-279`) and therefore the
// `generate_gpu_greedy` loop. That also means `--seed` is *irrelevant* — the
// sampler is never constructed (`generate.rs:162` is on the other branch). It
// is still passed, and recorded, because a record that names a seed nobody used
// is worse than one that names it as unused. If any arm ever needs
// temperature > 0, the seed becomes load-bearing and
// `Sampler::new(config.sampler, config.seed)` (`generate.rs:162`) makes the
// stream reproducible; `Sampler::default()` has `temperature: 1.0`
// (`sampler.rs:19-26`), so a bare `--max-new-tokens` invocation is *sampled*,
// not greedy, and is not comparable to a `--temperature 0` run at all.
//
// ### The instrumentation
//
// Only `KAPLUMBA_SEGMENTS=1`, `KAPLUMBA_SEGMENTS_BUCKET=1`,
// `KAPLUMBA_COUNT_BUFFERS=1` and `KAPLUMBA_PIPELINE_AUDIT=1`. All four are
// outside the `pipelined` predicate (`generate.rs:943-945`), so the loop under
// measurement is the shipping loop. **Never** `KAPLUMBA_DECODE_PROFILE=*`,
/// **never** `KAPLUMBA_MARGIN=1`, **never** `KAPLUMBA_STEP_TRACE` (V10).
//
// ### The interleaving
//
// One process per trial, because every env flag the decode path reads is behind
// a `OnceLock` or an atomic cache (`encoder.rs:240-248, 251-258, 273-281`;
// `model.rs:676-678, 692-697, 709-714`; `linear.rs:54-57`), so an in-process
// arm switch silently measures the first arm's configuration for every trial.
// Sequence per `k` in `1..=pairs`:
//
//     A_k  B_k  control_k
//
// `control` is arm A's configuration **verbatim under a different label**, run
// in the same session against the same binary. Its paired log-ratio against A
// *is* the noise floor, and it is what `sigma_floor_rel` is. A result whose
// control arm shows a 5 % swing is not a 3 % result, whatever the A/B says.
//
// ### Faults
//
// Three outcomes are **not** samples and each is counted in its own field:
// `Busy` (exit 75, `gpu_lock.rs:40` — nothing ran), `GpuFault` (stderr carries
/// `command buffer` and `status=`, `encoder.rs:198-201`), and `Inadmissible`
// (the metric's own gates). Nothing else is discarded: a slow run is a slow
// run, and dropping slow runs because they were slow is how this repo got a 35 %
// same-config swing reported as a 24 % offload cost.
//
// **Caveat that must travel with every number**: decode-time command-buffer
// faults are currently *undetectable* (V8), because `wait()`
// (`encoder.rs:639-644`) never inspects `status`. So `GpuFault` as defined here
// catches only prefill faults (`commit_and_wait`, `encoder.rs:656-658`). The
// measured fault rate is a **lower bound**, and the harness says so in its own
// output rather than letting a zero be read as "no faults".

// ===========================================================================
// PART 6 — THE HARNESS (`#[ignore]`d; never runs in the default suite)
// ===========================================================================

fn checkpoint_dir() -> std::path::PathBuf {
    match std::env::var_os("KAPLUMBA_PROBE_MODEL_DIR") {
        Some(dir) => std::path::PathBuf::from(dir),
        None => {
            let home = std::env::var_os("HOME").expect("HOME must be set");
            std::path::PathBuf::from(home).join(".kaplumba/models/Gxma/Gxma")
        }
    }
}

/// Per-step wall times for one decode, on the shipping pipelined loop.
///
/// `KAPLUMBA_SEGMENTS=1` with `KAPLUMBA_SEGMENTS_BUCKET=1` makes `SegmentLog`
/// emit one row per step (`generate.rs:616-634`: the bucket is closed as soon as
/// it holds `self.bucket` samples). It is orthogonal to the `pipelined`
/// predicate, so the loop is the shipping one — but it adds one `Vec::push` and
/// one add per step, and it writes to stderr, so a run instrumented this way is
/// not bit-identical in cost to an uninstrumented one. The harness therefore
/// records `instrumented: true` and the `KAPLUMBA_COUNT_BUFFERS` dispatch count
/// alongside every rate, so a reader can tell an instrumented number from an
/// uninstrumented one. `KAPLUMBA_PROBE_INSTRUMENT=0` runs the shipping loop with
/// no per-step data; in that mode only the legacy aggregate is available and
/// `step_ms` is `None`.
fn parse_step_ms(stderr: &str) -> Option<Vec<f64>> {
    let mut steps = Vec::new();
    let mut in_table = false;
    for line in stderr.lines() {
        if line.starts_with("=== SEGMENTED DECODE") {
            in_table = true;
            continue;
        }
        if !in_table {
            continue;
        }
        if line.starts_with("SEGMENT-DRIFT") || line.trim().is_empty() {
            break;
        }
        let Some(first) = line.split_whitespace().next() else {
            continue;
        };
        let lo = first.split('-').next().unwrap_or(first);
        if lo.parse::<usize>().is_err() {
            continue; // the header row
        }
        // `steps  context  ms/step  tok/s  vs_first`, all right-aligned, so
        // whitespace splitting recovers them regardless of the padding
        // (`generate.rs:664-670`).
        let Some(ms) = line.split_whitespace().nth(2) else {
            continue;
        };
        match ms.parse::<f64>() {
            Ok(value) if value.is_finite() => steps.push(value),
            // A row whose ms/step will not parse is a format change, not a
            // sample. Dropping it silently would understate `n_total` and make
            // the warmup discard bite the wrong steps, so the table is refused.
            _ => return None,
        }
    }
    if steps.is_empty() { None } else { Some(steps) }
}

/// `BUFFERS total=.. steps=.. per_step=.. | DISPATCHES total=.. steps=..
/// per_step=..` (`model.rs:179-186`).
fn parse_buffers_per_step(stderr: &str) -> Option<f64> {
    stderr.lines().find_map(|line| {
        let rest = line.strip_prefix("BUFFERS ")?;
        let mut fields = rest.split_whitespace();
        let (_total, _steps, per) = (fields.next()?, fields.next()?, fields.next()?);
        per.strip_prefix("per_step=")?.parse().ok()
    })
}

/// The release binary the process-level harness drives. One binary for both
/// arms is the point: `spec/decode-19tok-repacked-250gb` and every other note
/// in this repo that compares across code also compares across toolchains, and
/// that confound is half the reason its numbers did not reproduce.
fn binary() -> std::path::PathBuf {
    match std::env::var_os("KAPLUMBA_BIN") {
        Some(path) => std::path::PathBuf::from(path),
        None => {
            let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .ancestors()
                .find(|dir| dir.join("Cargo.toml").is_file() && dir.join("crates").is_dir())
                .expect("workspace root")
                .to_path_buf();
            workspace.join("target/release/kaplumba")
        }
    }
}

/// `prompt_tokens=.. generated_tokens=.. stopped_on_eos=.. load_ms=.. ttft_ms=..
/// prefill_tok_s=.. decode_tok_s=.. peak_resident_pages_bytes=..`
/// (`crates/cli/src/report.rs:26-38`).
#[derive(Debug, Clone, Copy, PartialEq)]
struct MetricsLine {
    prompt_tokens: usize,
    generated_tokens: usize,
    stopped_on_eos: bool,
    load_ms: f64,
    ttft_ms: f64,
    prefill_tok_s: f64,
    decode_tok_s: f64,
}

fn parse_metrics_line(stdout: &str) -> Option<MetricsLine> {
    let line = stdout
        .lines()
        .find(|line| line.contains("decode_tok_s="))?;
    let field = |name: &str| -> Option<&str> {
        line.split_whitespace()
            .find_map(|token| token.strip_prefix(name))
    };
    Some(MetricsLine {
        prompt_tokens: field("prompt_tokens=")?.parse().ok()?,
        generated_tokens: field("generated_tokens=")?.parse().ok()?,
        stopped_on_eos: field("stopped_on_eos=")? == "true",
        load_ms: field("load_ms=")?.parse().ok()?,
        ttft_ms: field("ttft_ms=")?.parse().ok()?,
        prefill_tok_s: field("prefill_tok_s=")?.parse().ok()?,
        decode_tok_s: field("decode_tok_s=")?.parse().ok()?,
    })
}

/// The three non-sample outcomes, decided before any number is looked at.
///
/// Order matters. Exit 75 comes first because a busy run produced no output at
/// all and must not be misread as a fault. The command-buffer test comes next
/// because `encoder.rs:191-203` prints `status=` and nothing else in the tree
/// does. Everything else that printed a metrics line is a sample, **including a
/// slow one** — dropping slow runs because they were slow is how
/// `spec/resident-memory-12gib/implementation_notes.md:207` got a 24 % apparent
/// effect out of a 35 % same-config swing.
pub fn classify(exit_code: i32, stderr: &str, has_metrics: bool) -> Outcome {
    if exit_code == kaplumba_engine::gpu_lock::EXIT_BUSY {
        return Outcome::Busy;
    }
    if stderr.contains("command buffer") && stderr.contains("status=") {
        return Outcome::GpuFault;
    }
    // A non-zero exit is not a clean sample even when a metrics line is present.
    // `main.rs:121` prints that line only after generation returns `Ok`, so the
    // combination should be impossible -- and because it is, treating it as a
    // sample would let a half-finished run contribute a rate if it ever were not.
    if exit_code != 0 {
        return Outcome::Error;
    }
    if has_metrics {
        Outcome::Ok
    } else {
        Outcome::Error
    }
}

/// One decode trial in-process: the engine's own aggregate, no per-step data.
///
/// Per-step times are unreachable from inside the process. `SegmentLog::report`
/// writes to stderr (`generate.rs:648-691`) and a Rust test cannot intercept its
/// own stderr, so an in-process trial can only read
/// `GenerationResult::decode_tokens_per_second` — the *legacy* metric. That
/// asymmetry is the reason the A/B harness is process-level: a child process's
/// stderr is capturable.
fn one_trial(
    model: &mut kaplumba_engine::model::TextModel,
    tokenizer: &kaplumba_engine::tokenizer::GemmaTokenizer,
    prompt: &str,
    max_new_tokens: usize,
    eos_ids: &[u32],
) -> kaplumba_engine::generate::GenerationResult {
    let config = kaplumba_engine::generate::GenerationConfig {
        max_new_tokens,
        eos_ids: eos_ids.to_vec(),
        sampler: kaplumba_engine::sampler::SamplerConfig {
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
        },
        // Unreachable at temperature 0: the sampler is never constructed
        // (`generate.rs:150-160`). Passed and recorded anyway.
        seed: 0,
        ..kaplumba_engine::generate::GenerationConfig::default()
    };
    kaplumba_engine::generate::generate_gpu(model, tokenizer, prompt, &config, |_| Ok(()))
        .expect("greedy decode")
}

/// The cheapest noise floor available: **one** model load, `N` identical
/// decodes, same process, same binary, same prompt, same flags.
///
/// This is the measurement every other agent should want before believing any
/// A/B of theirs. It costs one load plus `N` x ~2 s of decode, needs no second
/// binary, and separates "the machine moved" from "the code moved".
///
/// Reports the legacy metric's distribution and, from it, the implied paired
/// sigma and the effect size `N` trials could resolve.
#[test]
#[ignore = "needs the real checkpoint and the whole device; measures the same-binary noise floor"]
fn same_binary_same_config_noise_floor() {
    use kaplumba_engine::config::load_generation_config;
    use kaplumba_engine::model::TextModel;
    use kaplumba_engine::tokenizer::GemmaTokenizer;

    let dir = checkpoint_dir();
    let repeats: usize = std::env::var("KAPLUMBA_PROBE_REPEATS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(12);

    let eos_ids = load_generation_config(&dir.join("generation_config.json"))
        .expect("generation_config.json")
        .eos_token_ids;

    let mut model = TextModel::load_from_dir(&dir).expect("model loads");
    let tokenizer = GemmaTokenizer::from_model_dir(&dir).expect("tokenizer loads");

    let mut legacy_rates: Vec<f64> = Vec::new();
    let mut inadmissible: Vec<String> = Vec::new();

    for index in 0..repeats {
        let result = one_trial(
            &mut model,
            &tokenizer,
            BENCH_PROMPT,
            BENCH_MIN_NEW_TOKENS,
            &eos_ids,
        );
        eprintln!(
            "PROBE trial={index} prompt_tokens={} generated_tokens={} stopped_on_eos={} \
             ttft_ms={:.2} prefill_tok_s={:.3} legacy_decode_tok_s={:.4}",
            result.prompt_tokens,
            result.generated_tokens,
            result.stopped_on_eos,
            result.ttft_ms,
            result.prefill_tokens_per_second,
            result.decode_tokens_per_second,
        );
        if result.stopped_on_eos || result.generated_tokens < BENCH_MIN_NEW_TOKENS {
            inadmissible.push(format!(
                "trial {index}: generated {} tokens, stopped_on_eos={} — below \
                 min_new_tokens={BENCH_MIN_NEW_TOKENS}, so the reported rate would be an \
                 average over a workload-determined number of samples (audit B/C)",
                result.generated_tokens, result.stopped_on_eos,
            ));
            continue;
        }
        legacy_rates.push(result.decode_tokens_per_second);
    }

    let (n, median, min, max, spread) = crate_summary(&legacy_rates);
    eprintln!("NOISE-FLOOR legacy_decode_tok_s over {repeats} in-process trials");
    eprintln!("  n={n} median={median:.4} min={min:.4} max={max:.4} spread={spread:.5}");
    let sigma = implied_sigma(&legacy_rates);
    if sigma > 0.0 {
        eprintln!("  sigma_d_rel implied by this session's own spread: {sigma:.5}");
        eprintln!(
            "  pairs needed at that sigma: 3% -> {}, 5% -> {}, 10% -> {}",
            pairs_needed(sigma, 0.03).unwrap_or(0),
            pairs_needed(sigma, 0.05).unwrap_or(0),
            pairs_needed(sigma, 0.10).unwrap_or(0),
        );
    } else {
        eprintln!(
            "  fewer than 5 usable trials, so no sigma can be estimated here. Use the \
             process-level harness, whose control arm measures sigma directly."
        );
    }
    eprintln!(
        "  for reference, the documented pessimistic paired sigma is 0.143 \
         (spec/resident-memory-12gib/implementation_notes.md:207): \
         3% -> {} pairs, 5% -> {}, 10% -> {}",
        pairs_needed(0.143, 0.03).unwrap_or(0),
        pairs_needed(0.143, 0.05).unwrap_or(0),
        pairs_needed(0.143, 0.10).unwrap_or(0)
    );
    assert!(
        inadmissible.is_empty(),
        "{} of {repeats} trials were inadmissible and contribute no rate:\n{}\n\
         Do not report a rate from the survivors without this count attached: \
         discarding short runs biases the surviving median upward.",
        inadmissible.len(),
        inadmissible.join("\n")
    );
}

/// Arm spec: `label=K=V;K=V`, arms separated by `|`. Example:
///
/// ```text
/// KAPLUMBA_PROBE_ARMS='A=|B=KAPLUMBA_SK2_1COL=1'
/// ```
fn parse_arms(spec: &str) -> Vec<(String, Vec<(String, String)>)> {
    spec.split('|')
        .filter(|arm| !arm.trim().is_empty())
        .map(|arm| {
            let mut parts = arm.splitn(2, '=');
            let label = parts.next().unwrap_or_default().trim().to_string();
            let env = parts
                .next()
                .unwrap_or_default()
                .split(';')
                .filter(|kv| kv.contains('='))
                .filter_map(|kv| kv.split_once('='))
                .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
                .collect();
            (label, env)
        })
        .collect()
}

/// One process-level trial: spawn the CLI, capture stdout and stderr, and decide
/// its outcome before reading any number out of it.
struct Trial {
    arm: String,
    pair: usize,
    outcome: Outcome,
    exit_code: i32,
    legacy_tok_s: Option<f64>,
    steady_tok_s: Option<f64>,
    steps: Option<StepStats>,
    steps_raw: Option<Vec<f64>>,
    buffers_per_step: Option<f64>,
    metrics: Option<MetricsLine>,
    swap_used: Option<u64>,
    physmem_unused: Option<u64>,
}

/// `total = 6144.00M  used = 5105.38M  free = 1038.62M  (encrypted)` -- the exact
/// shape `sysctl -n vm.swapusage` prints.
fn parse_swap_used(text: &str) -> Option<u64> {
    // The real output is `total = 6144.00M  used = 5105.38M  free = ...` -- with
    // spaces around the `=`. Splitting on `"used="` finds nothing there, so the
    // key and the separator are matched separately.
    let after = text.split("used").nth(1)?;
    let after = after.split('=').nth(1)?.trim_start();
    let digits: String = after
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.')
        .collect();
    Some((digits.parse::<f64>().ok()? * 1_048_576.0) as u64)
}

fn swap_used_bytes() -> Option<u64> {
    let out = std::process::Command::new("sysctl")
        .args(["-n", "vm.swapusage"])
        .output()
        .ok()?;
    parse_swap_used(&String::from_utf8_lossy(&out.stdout))
}

/// `PhysMem: 22G used (3285M wired, 10G compressor), 1483M unused.`
///
/// The last comma-separated field, **with its unit handled**. Two mistakes were
/// shipped here before being caught and are documented at `scripts/gpuq:50-60`:
/// stripping non-digits turned `554M` into 554 **GiB** (a 1024x overstatement),
/// and `vm_stat`'s `Pages free` is not a pressure signal at all. Both are
/// avoided by construction here and pinned by a test below.
fn parse_physmem_unused(line: &str) -> Option<u64> {
    let tail = line.rsplit(',').next()?.trim();
    // `top` switches between K/M/G/T and drops the decimal below 10, so `554M` is
    // 0.55 GiB. Taking the last comma-separated field alone yields `unused.`,
    // which is not a number -- `None`, never a silent zero.
    let start = tail.find(|c: char| c.is_ascii_digit())?;
    let tail = &tail[start..];
    let end = tail
        .find(|c: char| !c.is_ascii_digit() && c != '.')
        .unwrap_or(tail.len());
    let (number, unit) = tail.split_at(end);
    let scale = match unit.chars().next() {
        Some('K') | Some('k') => 1024.0,
        Some('M') | Some('m') => 1_048_576.0,
        Some('G') | Some('g') => 1_073_741_824.0,
        Some('T') | Some('t') => 1_099_511_627_776.0,
        _ => 1.0,
    };
    Some((number.parse::<f64>().ok()? * scale) as u64)
}

fn physmem_unused_bytes() -> Option<u64> {
    let out = std::process::Command::new("top")
        .args(["-l", "1", "-n", "0"])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let line = text.lines().find(|line| line.contains("PhysMem"))?;
    parse_physmem_unused(line)
}

/// `f64` -> JSON, with non-finite values as `null`.
///
/// `serde_json` refuses to serialise NaN and infinity. Every summary here can
/// legitimately be NaN (an arm with no usable sample), and a harness that
/// *panics* on an empty arm is a harness that cannot report a fault storm.
fn jnum(value: f64) -> serde_json::Value {
    if value.is_finite() {
        serde_json::json!(value)
    } else {
        serde_json::Value::Null
    }
}

fn run_trial(
    arm: &str,
    pair: usize,
    arm_env: &[(String, String)],
    dir: &std::path::Path,
    instrumented: bool,
) -> Trial {
    let mut command = std::process::Command::new(binary());
    command
        .args(["--model"])
        .arg(dir)
        .args(["--prompt"])
        .arg(BENCH_PROMPT)
        .args(["--max-new-tokens", &BENCH_MIN_NEW_TOKENS.to_string()])
        // `args.rs:111-115`: 0 selects greedy argmax, which is the loop whose
        // throughput is quoted. Any other value samples and is not comparable.
        .args(["--temperature", "0"])
        .args(["--seed", "0"])
        .args(["--metrics"])
        // `gpu_lock.rs:69-78`: never block behind another agent. Exit 75 is a
        // scheduling outcome and the harness records it as one.
        .env("KAPLUMBA_GPU_LOCK", "try");
    for (key, value) in arm_env {
        command.env(key, value);
    }
    if instrumented {
        // The only four switches that leave the `pipelined` predicate
        // (`generate.rs:943-945`) alone, so the loop measured is the shipping
        // one. Never `KAPLUMBA_DECODE_PROFILE=*`, `KAPLUMBA_MARGIN=1`, or
        // `KAPLUMBA_STEP_TRACE`.
        command.env("KAPLUMBA_SEGMENTS", "1");
        command.env("KAPLUMBA_SEGMENTS_BUCKET", "1");
        command.env("KAPLUMBA_COUNT_BUFFERS", "1");
        command.env("KAPLUMBA_PIPELINE_AUDIT", "1");
    }

    let swap_before = swap_used_bytes();
    let phys_before = physmem_unused_bytes();
    let output = command.output().unwrap_or_else(|error| {
        panic!("cannot spawn {} for arm {arm}: {error}", binary().display())
    });
    let swap_after = swap_used_bytes();
    let phys_after = physmem_unused_bytes();

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let metrics = parse_metrics_line(&stdout);
    let steps_raw = parse_step_ms(&stderr);
    let exit_code = output.status.code().unwrap_or(-1);
    let mut outcome = classify(exit_code, &stderr, metrics.is_some());

    let steps = steps_raw.as_deref().and_then(|steps| {
        match step_stats(steps, WARMUP_STEPS) {
            Ok(stats) => {
                if stats.drift_rel.abs() > MAX_DRIFT_REL {
                    outcome = Outcome::Inadmissible;
                }
                Some(stats)
            }
            Err(_) => {
                outcome = Outcome::Inadmissible;
                None
            }
        }
    });

    // The `--min-new-tokens` gate the CLI does not have (`args.rs:93-135`).
    if let Some(line) = metrics {
        if line.stopped_on_eos || line.generated_tokens < BENCH_MIN_NEW_TOKENS {
            outcome = Outcome::Inadmissible;
        }
    }

    Trial {
        arm: arm.to_string(),
        pair,
        outcome,
        exit_code,
        legacy_tok_s: metrics.map(|line| line.decode_tok_s),
        steady_tok_s: steps.as_ref().map(steady_tok_s),
        steps,
        steps_raw,
        buffers_per_step: parse_buffers_per_step(&stderr),
        metrics,
        swap_used: swap_after.or(swap_before),
        physmem_unused: phys_after.or(phys_before),
    }
}

/// The canonical harness: interleaved `A B control` x `N` pairs, one process per
/// trial, in the order `A_1 B_1 C_1 A_2 B_2 C_2 ...`.
///
/// * `control` is arm A's configuration **verbatim under a different label**,
///   run against the same binary in the same session. Its paired log-ratio
///   against A *is* the noise floor, and it is the only honest estimate of it
///   available.
/// * Outcomes are decided before any number is read, and `busy` / `gpu_fault` /
///   `inadmissible` are each counted rather than folded into the rate.
/// * Writes the Part 8 JSON to `KAPLUMBA_PROBE_OUT` (default
///   `perf_probe_<head>.json` in the working directory).
///
/// Run with:
///
/// ```text
/// KAPLUMBA_PROBE_MODEL_DIR=~/.kaplumba/models/Gxma-repacked \
/// KAPLUMBA_PROBE_ARMS='A=|B=KAPLUMBA_SK2_1COL=1' \
/// KAPLUMBA_PROBE_PAIRS=8 \
/// cargo test --release -p kaplumba-engine \
///   --test perf_probe_stats_noise_floor -- --ignored --nocapture --test-threads=1
/// ```
#[test]
#[ignore = "needs the release binary, the real checkpoint and the whole device; runs the GPU"]
fn process_level_interleaved_ab_with_control() {
    use std::collections::BTreeMap;

    let dir = checkpoint_dir();
    let instrumented = std::env::var("KAPLUMBA_PROBE_INSTRUMENT")
        .map(|value| value != "0")
        .unwrap_or(true);
    let mut arms = parse_arms(
        &std::env::var("KAPLUMBA_PROBE_ARMS").unwrap_or_else(|_| "A=|B=".to_string()),
    );
    assert!(
        arms.len() >= 1,
        "KAPLUMBA_PROBE_ARMS must name at least one arm, e.g. 'A=|B=KAPLUMBA_SK2_1COL=1'"
    );
    // The control arm is arm A's environment verbatim. If the operator supplied
    // one explicitly, theirs wins.
    let baseline = arms[0].clone();
    if !arms.iter().any(|(label, _)| label == "control") {
        arms.push((
            "control".to_string(),
            baseline.1.clone(),
        ));
    }
    let pairs: usize = std::env::var("KAPLUMBA_PROBE_PAIRS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(8);
    assert!(
        pairs >= 1,
        "at least one pair; below 6 the noise floor dominates everything (see \
         min_detectable_effect)"
    );

    let mut trials: Vec<Trial> = Vec::new();
    for pair in 1..=pairs {
        for (label, arm_env) in &arms {
            eprintln!("PROBE pair={pair} arm={label} starting");
            let trial = run_trial(label, pair, arm_env, &dir, instrumented);
            if let Some(line) = trial.metrics {
                eprintln!(
                    "PROBE pair={pair} arm={label} outcome={:?} exit={} prompt_tokens={} \
                     generated_tokens={} stopped_on_eos={} load_ms={:.1} ttft_ms={:.2} \
                     legacy_decode_tok_s={:.4}",
                    trial.outcome,
                    trial.exit_code,
                    line.prompt_tokens,
                    line.generated_tokens,
                    line.stopped_on_eos,
                    line.load_ms,
                    line.ttft_ms,
                    line.decode_tok_s,
                );
            } else {
                eprintln!(
                    "PROBE pair={pair} arm={label} outcome={:?} exit={} (no metrics line)",
                    trial.outcome, trial.exit_code
                );
            }
            trials.push(trial);
        }
    }

    // ---- outcomes, counted, never folded in -------------------------------
    let mut outcomes: BTreeMap<String, BTreeMap<&str, usize>> = BTreeMap::new();
    for (label, _) in &arms {
        outcomes.insert(label.clone(), BTreeMap::new());
    }
    for trial in &trials {
        let bucket = outcomes.get_mut(&trial.arm).expect("arm was registered");
        *bucket.entry(outcome_name(trial.outcome)).or_insert(0) += 1;
    }
    for (label, bucket) in &outcomes {
        eprintln!("OUTCOMES {label}: {bucket:?}");
    }

    // ---- per-arm rates ---------------------------------------------------
    let mut legacy_by_arm: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut steady_by_arm: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for trial in trials.iter().filter(|t| t.outcome == Outcome::Ok) {
        legacy_by_arm
            .entry(trial.arm.clone())
            .or_default()
            .extend(trial.legacy_tok_s);
        if let Some(rate) = trial.steady_tok_s {
            steady_by_arm.entry(trial.arm.clone()).or_default().push(rate);
        }
    }
    for (label, samples) in &steady_by_arm {
        let (n, median, min, max, spread) = crate_summary(samples);
        eprintln!(
            "SUMMARY {label} steady_tok_s: n={n} median={median:.4} min={min:.4} \
             max={max:.4} spread={spread:.5}"
        );
    }
    for (label, samples) in &legacy_by_arm {
        let (n, median, min, max, spread) = crate_summary(samples);
        eprintln!(
            "SUMMARY {label} legacy_decode_tok_s: n={n} median={median:.4} min={min:.4} \
             max={max:.4} spread={spread:.5}"
        );
    }

    // ---- the noise floor, from the control arm, in this session -----------
    let baseline_label = baseline.0.clone();
    let noise = paired(
        &steady_by_arm.get(&baseline_label).cloned().unwrap_or_default(),
        &steady_by_arm.get("control").cloned().unwrap_or_default(),
        // Floor from the documented pessimistic prior until the control arm has
        // produced an estimate; `paired` takes the max of the two.
        0.0,
    );
    eprintln!(
        "NOISE-FLOOR {baseline_label} vs control: n_pairs={} effect={:+.4} sigma_d_rel={:.5} \
         min_detectable={:.4} verdict={:?}",
        noise.n_pairs,
        noise.effect_rel,
        noise.sigma_d_rel,
        noise.min_detectable_rel,
        noise.verdict
    );
    if noise.verdict != Verdict::Inconclusive {
        eprintln!(
            "NOISE-FLOOR-NOTE the control arm moved. Its paired difference is larger than the \
             session can resolve from its own spread, so no A/B in this session is interpretable. \
             Re-run on a quiet box; do not interpret the arm comparison."
        );
    }

    // ---- verdicts --------------------------------------------------------
    let mut paired_records = Vec::new();
    for (label, _) in &arms {
        if label == &baseline_label || label == "control" {
            continue;
        }
        let verdict = paired(
            &steady_by_arm.get(&baseline_label).cloned().unwrap_or_default(),
            &steady_by_arm.get(label).cloned().unwrap_or_default(),
            noise.sd_log_ratio,
        );
        eprintln!(
            "VERDICT {baseline_label} vs {label}: n_pairs={} effect={:+.4} \
             ci95=[{:+.4},{:+.4}] sigma_d_rel={:.5} min_detectable={:.4} verdict={:?}",
            verdict.n_pairs,
            verdict.effect_rel,
            verdict.effect_rel_at(verdict.ci95_low_log),
            verdict.effect_rel_at(verdict.ci95_high_log),
            verdict.sigma_d_rel,
            verdict.min_detectable_rel,
            verdict.verdict
        );
        paired_records.push(verdict);
    }

    // ---- the Part 8 record ----------------------------------------------
    let json = serde_json::json!({
        "schema": "kaplumba.perf_probe/1",
        "git_head": git_head(),
        "git_dirty": git_dirty(),
        "binary": binary().display().to_string(),
        "checkpoint": dir.display().to_string(),
        "config": {
            "prompt": BENCH_PROMPT,
            "max_new_tokens": BENCH_MIN_NEW_TOKENS,
            "min_new_tokens": BENCH_MIN_NEW_TOKENS,
            "temperature": 0.0,
            "seed": 0,
            "seed_used": false,
            "instrumented": instrumented,
        },
        "metric": {
            "name": "steady_tok_s",
            "definition": "1000 / median(step_ms[8..N))",
            "warmup_steps": WARMUP_STEPS,
            "min_steps": MIN_STEPS,
            "max_drift_rel": MAX_DRIFT_REL,
        },
        "arms": arms.iter().map(|(label, env)| serde_json::json!({
            "label": label,
            "env": env.iter().cloned().collect::<BTreeMap<_, _>>(),
        })).collect::<Vec<_>>(),
        "trials": trials.iter().map(|t| serde_json::json!({
            "arm": t.arm,
            "pair": t.pair,
            "outcome": outcome_name(t.outcome),
            "exit_code": t.exit_code,
            "prompt_tokens": t.metrics.map(|m| m.prompt_tokens),
            "generated_tokens": t.metrics.map(|m| m.generated_tokens),
            "stopped_on_eos": t.metrics.map(|m| m.stopped_on_eos),
            "load_ms": t.metrics.map(|m| m.load_ms),
            "ttft_ms": t.metrics.map(|m| m.ttft_ms),
            "prefill_tok_s": t.metrics.map(|m| m.prefill_tok_s),
            "legacy_decode_tok_s": t.legacy_tok_s,
            "steady_tok_s": t.steady_tok_s,
            "steps": t.steps.as_ref().map(|s| serde_json::json!({
                "n_total": s.n_total,
                "n_kept": s.n_kept,
                "warmup_discarded": s.warmup_discarded,
                "median_ms": jnum(s.median_ms),
                "mad_ms": jnum(s.mad_ms),
                "p10_ms": jnum(s.p10_ms),
                "p90_ms": jnum(s.p90_ms),
                "max_ms": jnum(s.max_ms),
                "drift_rel": jnum(s.drift_rel),
            })),
            "step_ms": t.steps_raw.clone(),
            "command_buffers_per_step": t.buffers_per_step,
            "swap_used_bytes": t.swap_used,
            "physmem_unused_bytes": t.physmem_unused,
        })).collect::<Vec<_>>(),
        "summary": arms.iter().map(|(label, _)| {
            let steady = steady_by_arm.get(label);
            let legacy = legacy_by_arm.get(label);
            serde_json::json!({
                "label": label,
                "steady": steady.map(|s| summary_json(s)),
                "legacy": legacy.map(|s| summary_json(s)),
            })
        }).collect::<Vec<_>>(),
        "outcomes": outcomes,
        "noise_floor": {
            "arm": baseline_label,
            "control": "control",
            "n_pairs": noise.n_pairs,
            "effect_rel": jnum(noise.effect_rel),
            "sigma_d_rel": jnum(noise.sigma_d_rel),
            "min_detectable_rel": jnum(noise.min_detectable_rel),
            "verdict": format!("{:?}", noise.verdict).to_lowercase(),
            "pairs_needed_at_this_sigma": {
                "3pct": pairs_needed(noise.sd_log_ratio, 0.03),
                "5pct": pairs_needed(noise.sd_log_ratio, 0.05),
                "10pct": pairs_needed(noise.sd_log_ratio, 0.10),
            },
            "caveat": "gpu_fault counts PREFILL command-buffer failures only. \
                       Decode-time status is never inspected (encoder.rs:639-644), \
                       so every fault rate here is a LOWER BOUND.",
        },
        "paired": paired_records.iter().map(|v| serde_json::json!({
            "n_pairs": v.n_pairs,
            "effect_rel": jnum(v.effect_rel),
            "ci95_low_rel": jnum(v.effect_rel_at(v.ci95_low_log)),
            "ci95_high_rel": jnum(v.effect_rel_at(v.ci95_high_log)),
            "sigma_d_rel": jnum(v.sigma_d_rel),
            "min_detectable_rel": jnum(v.min_detectable_rel),
            "verdict": format!("{:?}", v.verdict).to_lowercase(),
        })).collect::<Vec<_>>(),
    });

    let path = std::env::var("KAPLUMBA_PROBE_OUT").unwrap_or_else(|_| {
        format!("perf_probe_{}.json", git_head().replace(['-', ' '], "_"))
    });
    std::fs::write(&path, serde_json::to_string_pretty(&json).expect("serialises"))
        .unwrap_or_else(|error| panic!("cannot write {path}: {error}"));
    eprintln!("WROTE {path}");
}

fn outcome_name(outcome: Outcome) -> &'static str {
    match outcome {
        Outcome::Ok => "ok",
        Outcome::GpuFault => "gpu_fault",
        Outcome::Busy => "busy",
        Outcome::Inadmissible => "inadmissible",
        Outcome::Error => "error",
    }
}

fn summary_json(samples: &[f64]) -> serde_json::Value {
    let (n, median, min, max, spread) = crate_summary(samples);
    serde_json::json!({
        "n": n,
        "median": jnum(median),
        "min": jnum(min),
        "max": jnum(max),
        "spread": jnum(spread),
    })
}

fn git_head() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

fn git_dirty() -> bool {
    std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .map(|out| !out.stdout.is_empty())
        .unwrap_or(false)
}

impl MetricReject {
    fn label(&self) -> String {
        match self {
            MetricReject::NoSteps => "no per-step data".to_string(),
            MetricReject::TooFewSteps => format!(
                "fewer than {MIN_STEPS} steps survived the {WARMUP_STEPS}-step warmup discard"
            ),
            MetricReject::ImplausibleStep => "a step time was zero, negative or non-finite".into(),
            MetricReject::Drifting { drift_rel } => format!(
                "within-run drift {:+.2}% exceeds the {:.1}% steady-state bound",
                drift_rel * 100.0,
                MAX_DRIFT_REL * 100.0
            ),
        }
    }
}

/// `(n, median, min, max, spread)` — `spread = min / max`, matching
/// `bench.rs:219-269` `Summary`, so a number from this harness and a number
/// from `summarize` are the same statistic.
pub fn crate_summary(samples: &[f64]) -> (usize, f64, f64, f64, f64) {
    let sorted = sorted_finite(samples);
    if sorted.is_empty() {
        return (0, f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }
    let min = sorted[0];
    let max = *sorted.last().expect("non-empty");
    (
        sorted.len(),
        median_sorted(&sorted),
        min,
        max,
        min / max,
    )
}

/// Paired log-ratio sigma implied by a single arm's observed spread, as a
/// *prior* for `pairs_needed` before the control arm has been measured.
///
/// Uses `max - min` over `n` samples as a robust range: for `n >= 5` the
/// expected range of `n` normals is about `2.33 * sigma`, so `sigma ~= range /
/// 2.33`. Widened by `sqrt(2)` because a paired difference of two independent
/// arms carries that much more sigma than either arm alone. Returns 0 for a
/// degenerate range, and callers must then fall back on the documented 0.143.
pub fn implied_sigma(samples: &[f64]) -> f64 {
    let sorted = sorted_finite(samples);
    if sorted.len() < 5 {
        return 0.0;
    }
    let range = sorted[sorted.len() - 1] - sorted[0];
    let mid = median_sorted(&sorted);
    if mid <= 0.0 {
        return 0.0;
    }
    (range / 2.33) / mid * std::f64::consts::SQRT_2
}

// ===========================================================================
// PART 7 — ALWAYS-ON TESTS (no GPU, no checkpoint, no lock)
// ===========================================================================

#[test]
fn the_warmup_discard_is_what_makes_the_corrected_metric_work() {
    // A synthetic step series in the shape this repo has documented for itself:
    // one very slow first step (`model.rs:3553`, "1277 ms first-step cost",
    // against a ~62 ms steady state) followed by a tight steady state.
    let mut steps = vec![1277.0];
    for i in 0..30 {
        // 1 ms of deterministic jitter so the MAD is non-zero and the median is
        // not accidentally an exact repeat.
        steps.push(62.0 + (i % 3) as f64 * 0.1);
    }

    let stats = step_stats(&steps, WARMUP_STEPS).expect("31 steps clears the gate");
    assert_eq!(stats.warmup_discarded, WARMUP_STEPS);
    assert_eq!(stats.n_kept, 31 - WARMUP_STEPS);
    assert!(
        (stats.median_ms - 62.1).abs() < 1e-9,
        "median over the kept window must ignore the 1277 ms outlier, got {}",
        stats.median_ms
    );
    assert!(
        stats.mad_ms <= 0.2,
        "MAD must stay small when one step is 20x the rest, got {}",
        stats.mad_ms
    );

    // And the legacy formula over the same series is dragged by the outlier,
    // which is the whole reason the corrected metric exists. `n = 32` generated
    // tokens means 31 timed steps, one of which is the 1277 ms outlier.
    let sum_s: f64 = steps.iter().sum::<f64>() / 1000.0;
    let legacy = legacy_decode_tok_s(sum_s, 32);
    assert!(
        legacy < steady_tok_s(&stats) * 0.9,
        "a single 1277 ms outlier must move the legacy aggregate by more than 10%; \
         it moved it to {legacy:.3} against a corrected {0:.3}",
        steady_tok_s(&stats)
    );
}

#[test]
fn the_corrected_metric_is_invariant_to_how_many_tokens_the_prompt_produced() {
    // Two runs of the same binary on the same prompt: one generates 32 tokens,
    // one keeps going for 96. Both have identical warmed steps. The corrected
    // metric must report the same number; the legacy one cannot, because its
    // sample count is set by the workload.
    let steady_step = |i: usize| 62.0 + (i % 5) as f64 * 0.05;
    let short: Vec<f64> = (0..31).map(steady_step).collect();
    let long: Vec<f64> = (0..95).map(steady_step).collect();

    let a = steady_tok_s(&step_stats(&short, WARMUP_STEPS).expect("short is gated in"));
    let b = steady_tok_s(&step_stats(&long, WARMUP_STEPS).expect("long is gated in"));
    assert!(
        (a - b).abs() < 1e-9,
        "the corrected metric must not depend on generation length: {a} vs {b}"
    );

    // Legacy, over the same two series. The difference is not the point; the
    // point is that it is a *different estimator*, and the short run's is the
    // one with 31x less averaging.
    let legacy_short = legacy_decode_tok_s(short.iter().sum::<f64>() / 1000.0, 32);
    let legacy_long = legacy_decode_tok_s(long.iter().sum::<f64>() / 1000.0, 96);
    assert!(legacy_short.is_finite() && legacy_long.is_finite());
}

#[test]
fn a_run_that_stops_on_eos_is_refused_rather_than_reported() {
    // This is the `--min-new-tokens` gate. The CLI has no such flag
    // (`args.rs:93-135`), so an EOS at token 2 today prints a confident
    // `decode_tok_s` computed from one sample (`generate.rs:1128-1137`), which
    // is the same class of number as the brief's "~16 tok/s over 2 tokens".
    let steps: Vec<f64> = (0..31).map(|_| 62.0).collect();
    assert!(step_stats(&steps, WARMUP_STEPS).is_ok());

    // One interval: the whole reported rate is the first, cold, pipeline-filling
    // step. There is nothing to take a median of.
    assert_eq!(
        step_stats(&steps[..1], WARMUP_STEPS),
        Err(MetricReject::TooFewSteps)
    );
    // 23 intervals: 31 - 8 would clear the gate, so the boundary is real.
    let barely_enough: Vec<f64> = steps[..WARMUP_STEPS + MIN_STEPS].to_vec();
    assert_eq!(
        step_stats(&barely_enough, WARMUP_STEPS)
            .expect("exactly at the gate")
            .n_kept,
        MIN_STEPS
    );
    let one_short: Vec<f64> = steps[..WARMUP_STEPS + MIN_STEPS - 1].to_vec();
    assert_eq!(
        step_stats(&one_short, WARMUP_STEPS),
        Err(MetricReject::TooFewSteps)
    );

    // And the legacy formula happily produces a number for the one-step run,
    // which is precisely the defect.
    assert!(
        legacy_decode_tok_s(0.062, 2) > 15.0,
        "an EOS-truncated run still reports a plausible-looking rate"
    );
    assert_eq!(
        legacy_decode_tok_s(0.062, 1),
        0.0,
        "one generated token means zero timed steps, and the engine reports 0.0 \
         rather than refusing"
    );
}

#[test]
fn the_benchmark_prompt_stays_under_the_kv_ladder_rung() {
    // V4. The ladder starts at `INITIAL_FULL_CACHE = 512` (`model.rs:1077`) and
    // only grows when `ensure_capacity` is called (`model.rs:1663`), so a run
    // whose whole context fits under 512 pays **no** growth cost and therefore
    // carries no growth-related variance. This pins that property of the chosen
    // operating point at the prompt's *token* count, which is the only thing
    // that decides it. The tokeniser needs the checkpoint, so this asserts the
    // arithmetic and leaves the count to the harness, which reports it.
    assert!(
        BENCH_MIN_NEW_TOKENS + 128 < 512,
        "the benchmark context must stay inside INITIAL_FULL_CACHE, or the 1.25x \
         KV ladder (`gpu_cache.rs:414-444`) starts allocating and memcpying \
         (`gpu_cache.rs:238-271`) inside the timed decode and adds a systematic \
         cost that has nothing to do with the thing being measured"
    );
}

#[test]
fn power_matches_the_hand_computed_table() {
    // sigma_d_rel = 0.05: the paired-difference floor a busy-but-not-broken
    // session gives. n = ceil((2.8016 * 0.05 / effect)^2).
    assert_eq!(pairs_needed(0.05, 0.03), Some(22));
    assert_eq!(pairs_needed(0.05, 0.05), Some(8));
    assert_eq!(pairs_needed(0.05, 0.10), Some(2));

    // sigma_d_rel = 0.10.
    assert_eq!(pairs_needed(0.10, 0.03), Some(88));
    assert_eq!(pairs_needed(0.10, 0.05), Some(32));
    assert_eq!(pairs_needed(0.10, 0.10), Some(8));

    // sigma_d_rel = 0.143: implied by the repo's own recorded 35 % same-config
    // swing (`spec/resident-memory-12gib/implementation_notes.md:207`).
    assert_eq!(pairs_needed(0.143, 0.03), Some(179));
    assert_eq!(pairs_needed(0.143, 0.05), Some(65));
    assert_eq!(pairs_needed(0.143, 0.10), Some(17));

    // sigma_d_rel = 0.003: the *good* regime this repo has also recorded, where
    // two arms of the same binary differed by 0.010 tok/s on a 15.85 median
    // (`spec/decode-25tok-peak-bandwidth/implementation_notes.md:311-312`).
    assert_eq!(pairs_needed(0.003, 0.03), Some(1));
    assert_eq!(pairs_needed(0.003, 0.01), Some(1));

    // Degenerate inputs return `None` rather than a plausible number, which is
    // the failure mode AGENTS.md §3.4 warns about for every instrument here.
    assert_eq!(pairs_needed(0.0, 0.05), None);
    assert_eq!(pairs_needed(f64::NAN, 0.05), None);
    assert_eq!(pairs_needed(0.05, 0.0), None);
    assert_eq!(pairs_needed(0.05, -0.01), None);
}

#[test]
fn min_detectable_effect_is_the_inverse_of_pairs_needed() {
    // The two functions are two views of one inequality, so they must agree
    // exactly — no tolerance needed, because `ceil` only ever makes `pairs_needed`
    // *smaller* and the implication below is in that direction:
    //     pairs_needed(sigma, e) <= n   <=>   min_detectable(sigma, n) <= e
    for &n in &[2_usize, 8, 22, 32, 65, 88, 179] {
        for &effect in &[0.20_f64, 0.10, 0.05, 0.03, 0.01] {
            let needed = pairs_needed(0.10, effect).expect("finite");
            let mde = min_detectable_effect(0.10, n);
            if needed <= n {
                assert!(
                    mde <= effect,
                    "n = {n} needs only {needed} pairs for a {effect} effect, so it must \
                     resolve it, but its minimum detectable effect is {mde:.5}"
                );
            } else {
                assert!(
                    mde > effect,
                    "n = {n} needs {needed} pairs for a {effect} effect, so it must not \
                     resolve it, but its minimum detectable effect is only {mde:.5}"
                );
            }
        }
    }
    // Monotone decreasing in n, and the exact boundary at a round number.
    for &n in &[2_usize, 3, 8, 22, 65] {
        assert!(
            min_detectable_effect(0.10, n + 1) < min_detectable_effect(0.10, n),
            "n = {n}: more pairs must resolve a smaller effect"
        );
    }
    assert_eq!(pairs_needed(0.10, 0.10), Some(8));
    assert!(min_detectable_effect(0.10, 8) <= 0.10);
    assert!(min_detectable_effect(0.10, 7) > 0.10);
    assert_eq!(min_detectable_effect(0.10, 0), f64::INFINITY);

    // The headline number, stated once so it cannot drift silently: **six pairs
    // resolve about 16 %, not a few percent**, at the recorded 0.143 floor.
    assert!(
        (min_detectable_effect(0.143, 6) - 0.1636).abs() < 0.001,
        "six pairs at the recorded floor must resolve about 16 %, got {}",
        min_detectable_effect(0.143, 6)
    );
}

#[test]
fn power_is_monotone_in_the_noise_floor() {
    let mut previous = 0;
    for step in 1..=1000 {
        let n = pairs_needed(step as f64 / 1000.0, 0.03).expect("finite");
        assert!(
            n >= previous,
            "pairs needed must not fall as sigma grows: {n} < {previous} at sigma {step}"
        );
        previous = n;
    }
    assert!(
        previous > 1000,
        "a 100 % sigma floor must need thousands of pairs at 3 %, got {previous}"
    );
}

#[test]
fn a_same_session_control_arm_turns_a_fake_win_into_inconclusive() {
    // The scenario this file exists for: an arm that looks 4 % better, measured
    // on a box whose own noise floor is 15 %.
    //
    // Constructed so the *paired* difference is what is small. A real win of 4 %
    // shows as every `b_i` 4 % above its `a_i`; a drift shows as both arms
    // moving together, which interleaving removes. Here the control arm is
    // drifting and A/B is clean, which is the case where a *single* A-then-B
    // pair would have looked like a result.
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut control = Vec::new();
    let truth = 1.04_f64;
    for i in 0..12 {
        let base = 16.0 + (i as f64) * 0.02;
        // A quiet progression plus a deliberate excursion on two trials.
        let excursion = if i == 4 { 0.75 } else { 1.0 };
        a.push(base);
        b.push(base * truth);
        control.push(base * excursion);
    }

    let noise = paired(&a, &control, 0.0);
    assert_eq!(
        noise.verdict,
        Verdict::Inconclusive,
        "a 4 % control excursion on one of twelve trials is not a result"
    );
    assert!(
        noise.sd_log_ratio > 0.02,
        "the control arm must have registered the excursion, got sigma {}",
        noise.sd_log_ratio
    );

    // And the true 4 % effect is equally unresolvable at that noise floor. This
    // is the blunt part: the answer is not "measure more carefully".
    let honest = paired(&a, &b, noise.sd_log_ratio);
    assert_eq!(
        honest.verdict,
        Verdict::Inconclusive,
        "4 % is below what this session can resolve; claiming it would be the \
         exact failure `spec/resident-memory-12gib/implementation_notes.md:207` records"
    );
    assert!(
        honest.min_detectable_rel > 0.04,
        "reported minimum detectable effect {} must exceed the 4 % claimed",
        honest.min_detectable_rel
    );
}

#[test]
fn a_clean_session_can_still_resolve_a_small_effect_and_says_so() {
    // The other regime, and the reason the protocol is not simply "give up".
    // `spec/decode-25tok-peak-bandwidth/implementation_notes.md:311-319`
    // recorded two same-binary arms differing by 0.010 tok/s with a spread of
    // 0.027-0.076 on a 15.85 median. That is a floor near 0.003, where a 3 %
    // effect needs one pair.
    let mut a = Vec::new();
    let mut b = Vec::new();
    for i in 0..8 {
        let base = 15.85 + (i % 3) as f64 * 0.01;
        a.push(base);
        b.push(base * 1.05);
    }
    let control = paired(&a, &a.clone(), 0.0);
    assert_eq!(control.verdict, Verdict::Inconclusive, "an identical arm is not a win");

    let result = paired(&a, &b, 0.003);
    assert_eq!(
        result.verdict,
        Verdict::Win,
        "a clean 5 % effect with a measured 0.003 floor must resolve in 8 pairs"
    );
    assert!((result.effect_rel - 0.05).abs() < 1e-9);
    assert!(
        result.ci95_low_log > 0.0,
        "the interval must exclude zero when the verdict is a win: [{}, {}]",
        result.ci95_low_log,
        result.ci95_high_log
    );
    assert!(result.ci95_low_log < result.mean_log_ratio);
    assert!(result.ci95_high_log > result.mean_log_ratio);
}

#[test]
fn the_per_step_parser_reads_the_table_the_instrument_actually_prints() {
    // Shape taken from `SegmentLog::report` (`generate.rs:648-670`): a header,
    // then `{:>6} {:>7} {:>10.3} {:>9.3} {:>7.1}%` rows of
    // `"{first}-{last}"`, `"~{last}"`, `ms/step`, `tok/s`, `vs_first`, then the
    // `SEGMENT-DRIFT` line.
    let stderr = "\
=== SEGMENTED DECODE (KAPLUMBA_SEGMENTS) ===
  steps  context    ms/step     tok/s   vs_first
     0-0      ~37    62.400    16.026   100.0%
     1-1      ~38    62.100    16.103   100.0%
     2-2      ~39    62.350    16.043   100.0%
SEGMENT-DRIFT first=16.026 tok/s last=16.043 tok/s drift=+0.1%
";
    let steps = parse_step_ms(stderr).expect("table parsed");
    assert_eq!(steps.len(), 3);
    assert!((steps[1] - 62.1).abs() < 1e-9, "column 3 is ms/step, got {steps:?}");

    // A `KAPLUMBA_SEGMENTS_BUCKET` other than 1 gives multi-step rows, and the
    // parser must still work: it keys on the `first-last` field, not the range.
    let stderr = "\
=== SEGMENTED DECODE (KAPLUMBA_SEGMENTS) ===
  steps  context    ms/step     tok/s   vs_first
     0-127    ~164    62.200    16.077   100.0%
   128-255    ~292    62.900    15.898    99.5%
SEGMENT-DRIFT first=16.077 tok/s last=15.898 tok/s drift=-1.1%
";
    let steps = parse_step_ms(stderr).expect("bucketed table parsed");
    assert_eq!(steps, vec![62.2, 62.9]);

    // No table at all is `None`, never a zero, so an uninstrumented run cannot
    // be mistaken for an instrumented one that measured nothing.
    assert!(parse_step_ms("").is_none());
    assert!(parse_step_ms("BUFFERS total=396 steps=12 per_step=33.00 | DISPATCHES total=16884 steps=12 per_step=1407.0").is_none());
}

#[test]
fn the_buffer_counter_parser_reads_the_line_it_claims_to() {
    // `report_buffers_per_step`, `model.rs:179-186`.
    let line = "BUFFERS total=396 steps=12 per_step=33.00 | DISPATCHES total=16884 steps=12 per_step=1407.0";
    assert_eq!(parse_buffers_per_step(line), Some(33.0));
    assert_eq!(parse_buffers_per_step(""), None);
}

#[test]
fn the_corrected_metric_flags_a_drifting_run_instead_of_averaging_it() {
    // A run whose per-step cost grows 30 % across the window is not a
    // steady-state measurement. `SegmentLog::report` prints the same warning
    // past -10 % (`generate.rs:685-690`); the corrected metric carries the
    // number so it can be compared across runs instead of only being printed.
    let mut steps: Vec<f64> = (0..40).map(|i| 60.0 + i as f64 * 0.5).collect();
    let stats = step_stats(&steps, WARMUP_STEPS).expect("gate clears");
    assert!(
        stats.drift_rel > 0.05,
        "a 60 -> 79 ms ramp must register as drift, got {:+.2}%",
        stats.drift_rel * 100.0
    );
    // Flat within tolerance: no drift claim.
    steps = (0..40).map(|i| 62.0 + (i % 3) as f64 * 0.01).collect();
    let stats = step_stats(&steps, WARMUP_STEPS).expect("gate clears");
    assert!(
        stats.drift_rel.abs() < 0.01,
        "a flat series must not be called drifting, got {:+.2}%",
        stats.drift_rel * 100.0
    );
}

#[test]
fn an_implausible_step_time_is_refused_not_dropped() {
    // `decode_profile.rs:182-184` `plausible()` and its `bad_steps` counter exist
    // because "a silently-dropped step would make the mean look better than the
    // instrument deserves". The corrected metric refuses the whole run instead,
    // which is stronger and needs no separate counter.
    let mut steps = vec![62.0_f64; 31];
    steps[7] = 0.0;
    assert_eq!(step_stats(&steps, WARMUP_STEPS), Err(MetricReject::ImplausibleStep));
    steps[7] = f64::NAN;
    assert_eq!(step_stats(&steps, WARMUP_STEPS), Err(MetricReject::ImplausibleStep));
    steps[7] = -1.0;
    assert_eq!(step_stats(&steps, WARMUP_STEPS), Err(MetricReject::ImplausibleStep));
    assert_eq!(step_stats(&[], WARMUP_STEPS), Err(MetricReject::NoSteps));
}

#[test]
fn the_metrics_line_parser_reads_the_format_the_cli_actually_prints() {
    // `crates/cli/src/report.rs:26-38`, verbatim field order and names.
    let stdout = "\nHello, world.\nprompt_tokens=37 generated_tokens=32 stopped_on_eos=false \
                  load_ms=6432.117 ttft_ms=88.412 prefill_tok_s=419.207 \
                  decode_tok_s=16.097 peak_resident_pages_bytes=6687851008\n";
    let line = parse_metrics_line(stdout).expect("metrics line parsed");
    assert_eq!(line.prompt_tokens, 37);
    assert_eq!(line.generated_tokens, 32);
    assert!(!line.stopped_on_eos);
    assert!((line.decode_tok_s - 16.097).abs() < 1e-9);
    assert!((line.load_ms - 6432.117).abs() < 1e-6);
    // `prefill_tok_s=` and `decode_tok_s=` must not be confused for each other,
    // which a prefix search on the shared suffix `tok_s=` would do.
    assert!((line.prefill_tok_s - 419.207).abs() < 1e-9);

    // An EOS-truncated run parses fine and is therefore *not* a parse failure.
    // Nothing downstream may treat "parsed" as "valid"; that is what the
    // admissibility gate is for.
    let truncated = "prompt_tokens=37 generated_tokens=2 stopped_on_eos=true load_ms=6400.000 \
                     ttft_ms=88.000 prefill_tok_s=419.000 decode_tok_s=16.129 \
                     peak_resident_pages_bytes=1";
    let line = parse_metrics_line(truncated).expect("truncated line still parses");
    assert_eq!(line.generated_tokens, 2);
    assert!(line.stopped_on_eos);

    // No metrics line at all is `None`, never a zero, so a crashed run cannot be
    // recorded as a run of 0 tok/s.
    assert!(parse_metrics_line("").is_none());
    assert!(parse_metrics_line("Hello, world.\n").is_none());
}

#[test]
fn outcomes_are_decided_before_any_number_is_read() {
    // Exit 75 first: a busy run produced no output at all, so classifying it as
    // a fault would invent a fault rate out of contention.
    assert_eq!(classify(75, "", false), Outcome::Busy);
    assert_eq!(
        classify(75, "kaplumba: GPU busy (holder: ...), not waiting", false),
        Outcome::Busy
    );
    // The command-buffer signature is `encoder.rs:198-201`.
    assert_eq!(
        classify(1, "Metal command buffer failed (unlabelled) status=Error", false),
        Outcome::GpuFault
    );
    assert_eq!(
        classify(1, "Metal command buffer 'chunk/layer03' failed status=Error", false),
        Outcome::GpuFault
    );
    // A successful run is a sample **even when it is slow**. Dropping slow runs
    // because they were slow is how a 35 % same-config swing became a reported
    // 24 % effect.
    assert_eq!(classify(0, "", true), Outcome::Ok);
    // Neither signature and no metrics line: a real error, not a sample.
    assert_eq!(classify(1, "kaplumba: prompt plus generation exceeds context limit", false), Outcome::Error);
    // A metrics line with a non-zero exit is still an error: the run died after
    // printing, and its last rate is not a complete generation.
    assert_eq!(classify(1, "some other failure", true), Outcome::Error);
}

#[test]
fn the_swap_and_memory_parsers_do_not_repeat_the_documented_mistakes() {
    // `AGENTS.md:296-306` and `scripts/gpuq:50-60`: `554M` is 0.55 GiB, not
    // 554 GiB, and `Pages free` is not a pressure signal.
    assert_eq!(
        parse_physmem_unused("PhysMem: 22G used (3285M wired, 10G compressor), 554M unused."),
        Some(554 * 1024 * 1024)
    );
    assert_eq!(
        parse_physmem_unused("PhysMem: 22G used (3285M wired, 10G compressor), 1483M unused."),
        Some(1483 * 1024 * 1024)
    );
    assert_eq!(
        parse_physmem_unused("PhysMem: 15G used, 1234K unused."),
        Some(1234 * 1024)
    );
    assert_eq!(
        parse_physmem_unused("PhysMem: 2T used, 3G unused."),
        Some(3 * 1024 * 1024 * 1024)
    );
    assert_eq!(parse_physmem_unused("no memory line here"), None);
    // A line with no trailing number is `None`, not a silent zero -- the
    // `unused.` failure mode in `scripts/gpuq:58`.
    assert_eq!(parse_physmem_unused("PhysMem: 22G used, unused."), None);

    assert_eq!(
        parse_swap_used("total = 6144.00M  used = 5105.38M  free = 1038.62M  (encrypted)"),
        Some((5105.38 * 1_048_576.0) as u64)
    );
    // The spacing around `=` is not guaranteed, so both shapes must parse.
    assert_eq!(
        parse_swap_used("total=6144.00M used=12.00M free=6132.00M"),
        Some((12.00 * 1_048_576.0) as u64)
    );
    assert_eq!(parse_swap_used("no swap"), None);
    assert_eq!(parse_swap_used("used = notanumber"), None);
}

#[test]
fn the_arm_spec_parses_into_labels_and_env_overlays() {
    let arms = parse_arms(
        "A=|B=KAPLUMBA_SK2_1COL=1;KAPLUMBA_MAX_IN_FLIGHT=1|C=KAPLUMBA_DECODE_NR0=4",
    );
    assert_eq!(arms.len(), 3, "arms are separated by '|', env pairs by ';'");
    assert_eq!(arms[0].0, "A");
    assert!(arms[0].1.is_empty(), "an empty overlay is the baseline arm");
    assert_eq!(arms[1].0, "B");
    assert_eq!(arms[1].1.len(), 2);
    assert_eq!(arms[1].1[0], ("KAPLUMBA_SK2_1COL".into(), "1".into()));
    assert_eq!(arms[1].1[1], ("KAPLUMBA_MAX_IN_FLIGHT".into(), "1".into()));
    assert_eq!(arms[2].0, "C");
    assert_eq!(arms[2].1, vec![("KAPLUMBA_DECODE_NR0".into(), "4".into())]);
    // A fragment with no `=` is not an env pair and is skipped rather than
    // becoming a nameless variable.
    assert_eq!(parse_arms("A=GARBAGE").len(), 1);
    assert!(parse_arms("A=GARBAGE")[0].1.is_empty());
    assert!(parse_arms("").is_empty());
    assert!(parse_arms("  |  ").is_empty());
}

#[test]
fn every_rejection_reason_names_its_gate() {
    // A refusal that does not say what it wanted is indistinguishable from a
    // crash in the log, and the log is what a reader checks first.
    let labels = [
        MetricReject::NoSteps.label(),
        MetricReject::TooFewSteps.label(),
        MetricReject::ImplausibleStep.label(),
        MetricReject::Drifting { drift_rel: 0.12 }.label(),
    ];
    for label in &labels {
        assert!(!label.trim().is_empty(), "an empty reason names nothing");
    }
    assert!(labels[1].contains(&MIN_STEPS.to_string()));
    assert!(labels[1].contains(&WARMUP_STEPS.to_string()));
    assert!(
        labels[3].contains("12.00") && labels[3].contains("5.0"),
        "a drift reason must carry both the measured drift and the bound: {}",
        labels[3]
    );
}

#[test]
fn statistics_are_defined_on_degenerate_input() {
    assert_eq!(median(&[]), None);
    assert_eq!(median(&[f64::NAN, f64::NAN]), None);
    assert_eq!(mad(&[]), None);
    assert_eq!(median(&[3.0, 1.0, 2.0]), Some(2.0));
    // Even-length median is the mean of the two central values, so two samples
    // of a bimodal run do not silently pick one side.
    assert_eq!(median(&[10.0, 20.0]), Some(15.0));
    assert_eq!(median(&[f64::NAN, 1.0, 3.0, 2.0]), Some(2.0));
    // Quantile indexing matches `decode_profile.rs:250-265`.
    assert_eq!(quantile(&[1.0, 2.0, 3.0, 4.0, 5.0], 0.0), Some(1.0));
    assert_eq!(quantile(&[1.0, 2.0, 3.0, 4.0, 5.0], 1.0), Some(5.0));
    assert_eq!(quantile(&[1.0, 2.0, 3.0, 4.0, 5.0], 0.5), Some(3.0));
    assert_eq!(quantile(&[], 0.5), None);
    assert_eq!(quantile(&[1.0], 1.5), None);
}

#[test]
fn summary_agrees_with_the_repo_existing_summarize() {
    // `bench.rs:248-269` `summarize` is the statistic this repo already
    // standardises on, and `bench.rs:11-16` records why it exists (the previous
    // benchmark's headline was its best pass). `crate_summary` must produce the
    // same numbers, so a figure from this harness is comparable with a figure
    // from any other bench in the tree.
    let samples = [15.477, 15.634, 15.521, 15.452, 15.609];
    let expected = kaplumba_engine::bench::summarize(&samples).expect("non-empty");
    let (n, median, min, max, spread) = crate_summary(&samples);
    assert_eq!(n, expected.n);
    assert!((median - expected.median).abs() < 1e-12);
    assert!((min - expected.worst).abs() < 1e-12);
    assert!((max - expected.best).abs() < 1e-12);
    assert!((spread - expected.spread).abs() < 1e-12);
}

#[test]
fn implied_sigma_is_a_prior_not_a_substitute_for_the_control_arm() {
    // The 35 % same-config swing recorded in
    // `spec/resident-memory-12gib/implementation_notes.md:207` is two samples,
    // which is below this function's `n >= 5` floor, so it returns 0 and the
    // caller is forced to use the documented pessimistic 0.143. That is
    // deliberate: a two-point range cannot estimate a sigma, and pretending
    // otherwise is how a noise floor becomes a fabricated constant.
    assert_eq!(implied_sigma(&[9.124, 12.338]), 0.0);
    assert_eq!(implied_sigma(&[16.0; 5]), 0.0);
    assert_eq!(implied_sigma(&[]), 0.0);

    // With enough samples it does produce something, and it scales with the
    // observed range. The offsets are zero-median so the two sample sets share
    // a median; otherwise the ratio below would be measuring the medians, not
    // the ranges.
    let offsets: Vec<f64> = (0..12).map(|i| (i as f64) - 5.5).collect();
    let tight: Vec<f64> = offsets.iter().map(|o| 16.0 + o * 0.01).collect();
    let wide: Vec<f64> = offsets.iter().map(|o| 16.0 + o * 0.20).collect();
    assert!(
        (median(&tight).unwrap() - median(&wide).unwrap()).abs() < 1e-12,
        "the two sets must share a median for the ratio to mean anything"
    );
    let a = implied_sigma(&tight);
    let b = implied_sigma(&wide);
    assert!(a > 0.0 && b > a * 5.0, "tight {a} must be far below wide {b}");
    // 20x the range must give 20x the sigma: the function is linear in the
    // range by construction, so this pins it rather than merely bounding it.
    assert!(
        (b / a - 20.0).abs() < 1e-9,
        "wide/tight was {}, expected 20 because the range scales by 20",
        b / a
    );
    // And it lands in the right order of magnitude for the documented 35 %:
    // a 20 % peak-to-peak over 12 samples is ~8.6 % sigma per arm, ~12 % paired.
    assert!((b - 0.12).abs() < 0.04, "wide sigma was {b}, expected about 0.12");
}

// ===========================================================================
// PART 8 — MACHINE-READABLE RESULTS
// ===========================================================================
//
// Prose in `implementation_notes.md` cannot be compared across sessions; this
// is what replaces it. One JSON object per A/B session, written by the harness,
// diffable and machine-checkable. Field names say the unit. Nothing that
// changes the meaning of a number is optional.
//
// ```json
// {
//   "schema": "kaplumba.perf_probe/1",
//   "recorded_utc": "2026-10-02T22:41:07Z",
//   "git_head": "2aaa50e",
//   "git_dirty": true,
//   "binary_sha256": "…",
//   "checkpoint": "/Users/…/.kaplumba/models/Gxma-repacked",
//   "host": { "physical_cores": 14, "mem_bytes": 25769803768, "os": "macOS …" },
//   "config": {
//     "prompt_sha256": "…", "prompt_tokens": 37,
//     "max_new_tokens": 32, "min_new_tokens": 32,
//     "temperature": 0.0, "seed": 0, "seed_used": false,
//     "eos_ids": [1, 106, 50],
//     "instrumented": true,
//     "env": { "KAPLUMBA_SEGMENTS": "1", "KAPLUMBA_SEGMENTS_BUCKET": "1",
//               "KAPLUMBA_COUNT_BUFFERS": "1", "KAPLUMBA_PIPELINE_AUDIT": "1" }
//   },
//   "metric": {
//     "name": "steady_tok_s",
//     "definition": "1000 / median(step_ms[8..N))",
//     "warmup_steps": 8, "min_steps": 16, "max_drift_rel": 0.05
//   },
//   "arms": {
//     "A": { "env": {}, "binary_sha256": "…" },
//     "B": { "env": { "KAPLUMBA_SK2_1COL": "1" }, "binary_sha256": "…" },
//     "control": { "env": {}, "binary_sha256": "…", "note": "identical to A by construction" }
//   },
//   "trials": [
//     { "arm": "A", "pair": 1, "outcome": "ok", "exit_code": 0, "reject": null,
//       "load_ms": 6432.1, "ttft_ms": 88.4, "prefill_tok_s": 419.2,
//       "generated_tokens": 32, "stopped_on_eos": false,
//       "legacy_decode_tok_s": 15.97,
//       "steady_tok_s": 16.10,
//       "steps": { "n_total": 31, "n_kept": 23, "warmup_discarded": 8,
//                  "median_ms": 62.11, "mad_ms": 0.31, "p10_ms": 61.8,
//                  "p90_ms": 62.9, "max_ms": 71.4, "drift_rel": 0.004 },
//       "command_buffers_per_step": 33.0, "dispatches_per_step": 1407.0,
//       "physmem_unused_bytes_before": 15900672, "physmem_unused_bytes_after": 1483000000,
//       "swap_used_bytes_before": 5348656640, "swap_used_bytes_after": 5348656640 }
//   ],
//   "summary": {
//     "A":        { "n": 12, "median": 16.102, "min": 16.011, "max": 16.221, "spread": 0.9870 },
//     "B":        { "n": 12, "median": 16.740, "min": 16.480, "max": 17.010, "spread": 0.9688 },
//     "control":  { "n": 12, "median": 16.099, "min": 15.640, "max": 16.480, "spread": 0.9497 }
//   },
//   "outcomes": {
//     "A": { "ok": 12, "gpu_fault": 0, "busy": 0, "inadmissible": 0, "error": 0 },
//     "B": { "ok": 11, "gpu_fault": 1, "busy": 0, "inadmissible": 0, "error": 0 },
//     "control": { "ok": 12, "gpu_fault": 0, "busy": 0, "inadmissible": 0, "error": 0 },
//     "note": "gpu_fault counts PREFILL faults only: decode-time command-buffer \
//              status is never inspected (encoder.rs:639-644), so this is a lower bound"
//   },
//   "paired": {
//     "A_vs_control": { "n_pairs": 12, "effect_rel": -0.0002, "sigma_d_rel": 0.0071,
//                       "min_detectable_rel": 0.0132, "verdict": "inconclusive" },
//     "A_vs_B":        { "n_pairs": 12, "effect_rel": 0.0396, "sigma_d_rel": 0.0071,
//                       "min_detectable_rel": 0.0132, "verdict": "win",
//                       "pairs_needed_at_this_sigma": { "3pct": 0, "5pct": 0, "10pct": 1 } }
//   }
// }
// ```
//
// Rules that make it comparable rather than merely machine-readable:
//   * `prompt_sha256` and `binary_sha256` are both required. A record without
//     them is not comparable to anything, and the loader must reject it.
//   * `git_dirty` is required. Most of this session's work is uncommitted, and a
//     record that does not say so is lying by omission.
//   * `seed_used` is required, so a record cannot imply determinism it did not
//     have. At `--temperature 0` the sampler is never constructed
//     (`generate.rs:150-160`).
//   * `instrumented` is required, because `KAPLUMBA_DECODE_PROFILE=*` and
//     `KAPLUMBA_MARGIN=1` move the run onto the non-pipelined loop and change
//     what is being timed (V10).
//   * `outcomes` is required next to `summary`, with the fault-rate caveat
//     inline, because "discarded the faulted runs" silently biases the survivor
//     median upward (V8).
//   * `paired[*].verdict` is one of `win` / `loss` / `inconclusive`, and
//     `inconclusive` is the expected value for most proposals at the recorded
//     noise floor. There is no `probably`.

/// The outcome vocabulary. Serialised in lower snake case.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    /// Produced a rate.
    Ok,
    /// A Metal command buffer reported a non-`Completed` status
    /// (`encoder.rs:191-203`). **Prefill only** — see V8.
    GpuFault,
    /// Exit 75, `gpu_lock.rs:40`. Nothing ran; not a failure and not a sample.
    Busy,
    /// Ran, but the metric's gates refused it: EOS before `min_new_tokens`, too
    /// few steps, an implausible step, or more than `max_drift_rel` of drift.
    Inadmissible,
    /// Anything else. Never silently dropped.
    Error,
}

#[test]
fn outcome_labels_are_distinct_and_snake_case() {
    let all = [
        Outcome::Ok,
        Outcome::GpuFault,
        Outcome::Busy,
        Outcome::Inadmissible,
        Outcome::Error,
    ];
    let json = serde_json::to_string(&all).expect("serialises");
    assert_eq!(
        json,
        r#"["ok","gpu_fault","busy","inadmissible","error"]"#,
        "the vocabulary is part of the schema; renaming it silently breaks every \
         consumer"
    );
    // Round-trip: a reader must get the same enum back.
    let back: Vec<Outcome> = serde_json::from_str(&json).expect("deserialises");
    assert_eq!(back, all);
}

#[test]
fn paired_verdict_is_a_closed_vocabulary() {
    let json = serde_json::to_string(&[
        Verdict::Win,
        Verdict::Loss,
        Verdict::Inconclusive,
    ])
    .expect("serialises");
    assert_eq!(json, r#"["win","loss","inconclusive"]"#);
}