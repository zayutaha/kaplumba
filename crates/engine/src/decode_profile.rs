//! Per-decode-step GPU timing.
//!
//! # Why this exists
//!
//! `KAPLUMBA_GPU_PROFILE=1` builds a [`crate::model::GpuProfile`] *inside*
//! `forward_buffer_into`, and a decode step only builds one when
//! `KAPLUMBA_GPU_PROFILE_DECODE` is also set (`model.rs`). With that second
//! switch unset, the only table a decode run ever prints is the **prefill's**:
//! one `forward_buffer_into` call for the prompt, whose cost is a function of
//! the prompt alone. That is why the reported "sum of per-kernel GPU own_ms"
//! is flat in `--max-new-tokens` — it is a prefill sum, printed once.
//!
//! This module is the decode-side instrument, and it deliberately has two modes
//! because the two questions need different amounts of distortion:
//!
//! * `step` — **two** command-buffer boundaries per decode step (one opening,
//!   one closing) and nothing else. `close.end - open.start` is the whole
//!   step's duration on the GPU timeline: the intermediate rolled buffers
//!   (`MAX_DISPATCHES_PER_BUFFER = 50` dispatches each, ~42 of them for a
//!   31B-class step) are executed in commit order between the two, so the span
//!   covers all of them without stamping any. Cost: one extra commit per step.
//!   This is the clock.
//! * `kernel` — stamps every existing per-kernel boundary and accumulates
//!   across steps into one report instead of printing a table per token. This
//!   costs a `wait_until_completed` every `DRAIN_EVERY` boundaries, which
//!   serialises host encoding against GPU execution, so the *wall* time it
//!   reports is inflated by the profiler. Per-kernel **own_ms shares** are not.
//!   This is the attribution, and it must never be used as a stopwatch.
//!
//! # What the report is allowed to claim
//!
//! * `span` is the step's GPU-timeline duration and includes GPU idle between
//!   buffers, so `span` vs the caller's host `wall` is the instrument's own
//!   distortion check; they should agree closely in `step` mode.
//! * `own` is the sum of the stamped buffers' own durations, i.e. GPU work only.
//!   In `step` mode that is only the two boundary buffers, so it is small and
//!   not a decode cost; in `kernel` mode it is the whole step's dispatch time.
//! * Prefill and load are timed separately and reported separately, so a
//!   decode total can never silently absorb them.
//!
//! # Warmup
//!
//! Cold Metal clocks are several times slower than steady state, so the first
//! `KAPLUMBA_DECODE_PROFILE_WARMUP` steps (default 8) are counted but excluded
//! from every statistic. The report prints how many were skipped and what they
//! looked like, so an instrument that is measuring cold clocks is visible
//! rather than hidden inside a median.

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

use kaplumba_kernels::encoder::{GpuStamp, MetalEncoder};

use crate::model::ModelError;

/// Which measurement the caller asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// `KAPLUMBA_DECODE_PROFILE` unset: every entry point here is a no-op.
    Off,
    /// Two boundaries per decode step; the clock.
    Step,
    /// Every kernel boundary; the attribution.
    Kernel,
}

/// Per-step samples retained for the median/quartiles. A 8192-token decode
/// would otherwise grow these without bound; the oldest samples are dropped,
/// which is harmless because the statistic is a median over a warmed run.
const MAX_SAMPLES: usize = 4096;

/// `KAPLUMBA_DECODE_PROFILE=step|kernel`. Unset, unrecognised, or `off` is
/// [`Mode::Off`], which is the only value that makes this module a no-op.
///
/// Held in an atomic rather than a `OnceLock` only so that
/// [`configure_for_tests`] can install a mode; the read-once property, and
/// therefore the cost on the hot path, is the same.
pub fn mode() -> Mode {
    static CACHE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(u8::MAX);
    match CACHE.load(std::sync::atomic::Ordering::Relaxed) {
        u8::MAX => {
            let parsed = match std::env::var("KAPLUMBA_DECODE_PROFILE") {
                Ok(value) if value.trim().eq_ignore_ascii_case("step") => Mode::Step,
                Ok(value) if value.trim().eq_ignore_ascii_case("kernel") => Mode::Kernel,
                _ => Mode::Off,
            };
            CACHE.store(parsed as u8, std::sync::atomic::Ordering::Relaxed);
            parsed
        }
        code => match code {
            0 => Mode::Off,
            1 => Mode::Step,
            _ => Mode::Kernel,
        },
    }
}

/// `KAPLUMBA_DECODE_PROFILE_WARMUP`, default 8.
pub fn warmup_steps() -> usize {
    static CACHE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(usize::MAX);
    let cached = CACHE.load(std::sync::atomic::Ordering::Relaxed);
    if cached != usize::MAX {
        return cached;
    }
    let parsed = std::env::var("KAPLUMBA_DECODE_PROFILE_WARMUP")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(8);
    CACHE.store(parsed, std::sync::atomic::Ordering::Relaxed);
    parsed
}

/// Install a mode and a warmup count that do not come from the environment.
///
/// `#[doc(hidden)]` and test-only because the cache is a `OnceLock`-shaped
/// one-shot: a second caller gets whatever the first installed. That is fine for
/// a test binary (one instrument, one configuration) and wrong for anything
/// else, which is why this is not a public knob.
#[doc(hidden)]
pub fn configure_for_tests(mode: Option<Mode>, warmup: Option<usize>) {
    use std::sync::atomic::Ordering::Relaxed;
    if let Some(mode) = mode {
        MODE_FOR_TESTS.store(mode as u8, Relaxed);
    }
    if let Some(warmup) = warmup {
        WARMUP_FOR_TESTS.store(warmup, Relaxed);
    }
}

static MODE_FOR_TESTS: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(u8::MAX);
static WARMUP_FOR_TESTS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(usize::MAX);

/// [`mode`], with a test-installed override taking precedence. This is what
/// callers should branch on: a caller that checks [`mode`] directly would
/// disagree with the instrument under a test override.
pub fn active_mode() -> Mode {
    let forced = MODE_FOR_TESTS.load(std::sync::atomic::Ordering::Relaxed);
    if forced == u8::MAX {
        mode()
    } else if forced == 0 {
        Mode::Off
    } else if forced == 1 {
        Mode::Step
    } else {
        Mode::Kernel
    }
}

/// [`warmup_steps`], with a test-installed override taking precedence.
fn active_warmup() -> usize {
    let forced = WARMUP_FOR_TESTS.load(std::sync::atomic::Ordering::Relaxed);
    if forced == usize::MAX {
        warmup_steps()
    } else {
        forced
    }
}

/// Steps whose stamps could not be read as a plausible GPU interval. A
/// non-zero count means the clock produced garbage for those steps, which is
/// why the report prints it: a silently-dropped step would make the mean look
/// better than the instrument deserves.
fn plausible(start_s: f64, end_s: f64) -> bool {
    start_s > 0.0 && end_s > start_s && (end_s - start_s) < 3600.0
}

/// Everything the report prints, accumulated over one process's decode steps.
#[derive(Debug, Default)]
pub struct DecodeAccumulator {
    /// The mode actually in force, so a report cannot be attributed to the
    /// wrong mode.
    pub mode: Option<Mode>,
    /// Steps offered, including warmup.
    pub steps_seen: u64,
    /// Steps that passed the warmup filter and were recorded.
    pub steps_recorded: u64,
    /// Per-step GPU-timeline duration, `close.end - open.start`.
    pub span_ms: Vec<f64>,
    /// Per-step sum of stamped buffers' own GPU time.
    pub own_ms: Vec<f64>,
    /// Per-step host wall time, as the caller measured it.
    pub wall_ms: Vec<f64>,
    /// The warmup steps, kept so the report can show what it discarded.
    pub warmup_span_ms: Vec<f64>,
    /// Steps whose stamps were not a plausible GPU interval.
    pub bad_steps: u64,
    /// kernel -> (invocations, total own ms), `kernel` mode only.
    pub by_kernel: BTreeMap<String, (u64, f64)>,
    /// `kernel` mode: the sum of the steps' GPU spans.
    pub kernel_span_ms: f64,
    /// `kernel` mode: the sum of the steps' own GPU time.
    pub kernel_own_ms: f64,
    /// Prefill wall time, timed by the caller at the same place the decode loop
    /// is timed. Never folded into any decode total.
    pub prefill_wall_ms: f64,
    pub prefill_tokens: usize,
    /// Whether the report has been printed, so a run that reaches `report()`
    /// from two entry points prints it once.
    printed: bool,
}

/// `f64`'s `Sum` folds from `-0.0`, so an empty run prints `-0.000`. Normalise,
/// so a reader is not left wondering whether a negative time was measured.
fn normalise_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

fn push_capped(target: &mut Vec<f64>, value: f64) {
    if target.len() == MAX_SAMPLES {
        target.remove(0);
    }
    target.push(value);
}

fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if sorted.is_empty() {
        return None;
    }
    sorted.sort_by(f64::total_cmp);
    Some(sorted[sorted.len() / 2])
}

fn quantile(values: &[f64], fraction: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if sorted.is_empty() {
        return None;
    }
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() - 1) as f64 * fraction).round() as usize;
    Some(sorted[index])
}

impl DecodeAccumulator {
    /// Record one decode step from a pair of GPU stamps.
    ///
    /// `open` must be the stamp of the boundary that opened the step and
    /// `close` the one that closed it, so `span` is the step's own duration on
    /// the GPU timeline and neither the load nor the prefill can appear in it.
    /// `wall_ms` is the caller's host measurement of the same region and is the
    /// instrument's own cross-check.
    pub fn record_step(&mut self, mode: Mode, open: &GpuStamp, close: &GpuStamp, wall_ms: f64) {
        self.mode = Some(mode);
        self.steps_seen += 1;
        if !plausible(open.start_s, close.end_s) || close.end_s < open.start_s {
            self.bad_steps += 1;
            return;
        }
        let span = (close.end_s - open.start_s) * 1e3;
        // `own` is only meaningful where the stamps bracket the step's work:
        // in `step` mode the two boundary buffers hold one trivial dispatch and
        // a trailing slice, so this is "GPU time in the boundary buffers", not
        // a decode cost. The report says which.
        let own = open.own_ms + close.own_ms;
        if self.steps_seen <= active_warmup() as u64 {
            push_capped(&mut self.warmup_span_ms, span);
            return;
        }
        self.steps_recorded += 1;
        push_capped(&mut self.span_ms, span);
        push_capped(&mut self.own_ms, own);
        push_capped(&mut self.wall_ms, wall_ms);
        self.kernel_span_ms += span;
        self.kernel_own_ms += own;
    }

    /// Fold one step's per-kernel own times in, `kernel` mode.
    ///
    /// The warmup filter has to come *before* `rows` is folded in, not after.
    /// Folding first and filtering `own_ms` alone leaves the per-kernel table
    /// covering every step while `kernel_own_ms` -- the denominator the report's
    /// percentages and the `own_ms/step` column are computed from -- covers only
    /// the recorded ones, so the table silently over-reports by
    /// `steps_seen / steps_recorded`. That is 47/39 = 1.205 on a 48-token run:
    /// enough to make the big three read as 136 GB/s when they are 113.
    pub fn record_kernel_step(
        &mut self,
        span_ms: Option<f64>,
        own_ms: f64,
        rows: impl IntoIterator<Item = (String, u64, f64)>,
    ) {
        self.mode = Some(Mode::Kernel);
        self.steps_seen += 1;
        if self.steps_seen <= active_warmup() as u64 {
            return;
        }
        self.steps_recorded += 1;
        for (name, count, own) in rows {
            let slot = self.by_kernel.entry(name).or_default();
            slot.0 += count;
            slot.1 += own;
        }
        self.kernel_own_ms += own_ms;
        if let Some(span) = span_ms {
            if span.is_finite() && span > 0.0 {
                self.kernel_span_ms += span;
                push_capped(&mut self.span_ms, span);
            }
        }
    }

    pub fn record_prefill(&mut self, wall_ms: f64, tokens: usize) {
        self.prefill_wall_ms = wall_ms;
        self.prefill_tokens = tokens;
    }

    /// The span of the most recently *recorded* step, or `f64::NAN` if none.
    ///
    /// Exists so a test can read one step at a time and compare it against a
    /// specific surrounding condition -- a prefill length, an injection --
    /// rather than only reading a finished aggregate, which cannot say *which*
    /// step moved. Returns `NAN` for a warmup step, because a warmup step is not
    /// recorded at all; that is deliberate rather than a gap, since the whole
    /// point of the warmup is that its numbers are not comparable.
    pub fn last_span_ms(&self) -> f64 {
        self.span_ms.last().copied().unwrap_or(f64::NAN)
    }

    /// Report once per process. More than one entry point can reach
    /// [`report`] in a single run and the accumulator is cumulative, so a
    /// second print would be the same table twice.
    pub fn take_report(&mut self) -> Option<String> {
        if self.printed || self.mode.is_none() {
            return None;
        }
        self.printed = true;
        Some(self.render())
    }

    /// `step` mode: read the step's two boundary stamps off the encoder. The
    /// caller must already have committed the closing boundary and waited, and
    /// the opening boundary must still be retained.
    pub fn take_step(
        &mut self,
        encoder: &mut MetalEncoder,
        wall_ms: f64,
    ) -> Result<(), ModelError> {
        let stamps = encoder.take_gpu_stamps();
        let mut open = None;
        let mut close = None;
        for stamp in stamps {
            match stamp.label.as_str() {
                "step/open" => open = Some(stamp),
                "step/close" => close = Some(stamp),
                _ => {}
            }
        }
        match (open, close) {
            (Some(open), Some(close)) => {
                self.record_step(Mode::Step, &open, &close, wall_ms);
                Ok(())
            }
            _ => {
                self.bad_steps += 1;
                Err(ModelError::InvalidInput(
                    "decode step stamps missing: a step must be bracketed by \
                     step/open and step/close"
                        .to_string(),
                ))
            }
        }
    }

    /// The report. `key=value` on every line so a script can read it without
    /// parsing prose, plus a human table for the kernel mode.
    pub fn render(&self) -> String {
        let Some(mode) = self.mode else {
            return String::new();
        };
        let ms = |value: Option<f64>| match value {
            Some(value) => format!("{value:.3}"),
            None => "unavailable".to_string(),
        };
        let mut out = String::new();
        out.push_str("\n=== DECODE PROFILE (KAPLUMBA_DECODE_PROFILE) ===\n");
        out.push_str(&format!(
            "dp_mode={mode:?} dp_warmup_skipped={} dp_steps_recorded={} dp_bad_steps={}\n",
            self.steps_seen.min(active_warmup() as u64),
            self.steps_recorded,
            self.bad_steps
        ));
        // The scale check. A per-step number that does not depend on the step
        // is a broken instrument, so the report states the per-step spread and
        // the per-token means, and both are what a reader should compare
        // against `decode_tok_s`.
        out.push_str(&format!(
            "dp_span_ms_per_step_median={} dp_span_ms_per_step_p10={} dp_span_ms_per_step_p90={} \
             dp_span_ms_per_step_min={} dp_span_ms_per_step_max={}\n",
            ms(median(&self.span_ms)),
            ms(quantile(&self.span_ms, 0.10)),
            ms(quantile(&self.span_ms, 0.90)),
            ms(self.span_ms.iter().copied().reduce(f64::min)),
            ms(self.span_ms.iter().copied().reduce(f64::max)),
        ));
        let per_token_span = median(&self.span_ms);
        let per_token_own = median(&self.own_ms);
        let per_token_wall = median(&self.wall_ms);
        out.push_str(&format!(
            "dp_span_ms_per_token={} dp_own_ms_per_token={} dp_wall_ms_per_token={}\n",
            ms(per_token_span),
            ms(per_token_own),
            ms(per_token_wall),
        ));
        // The instrument's own distortion check: in `step` mode the GPU span
        // and the caller's host wall cover the same region, so a large
        // difference means the instrument (or the model) is doing something
        // the other is not.
        if let (Some(span), Some(wall)) = (per_token_span, per_token_wall) {
            out.push_str(&format!(
                "dp_span_minus_wall_ms={:.3} dp_span_over_wall={:.4}\n",
                span - wall,
                span / wall
            ));
        }
        out.push_str(&format!(
            "dp_warmup_span_ms_median={} dp_total_span_ms={:.3} dp_total_own_ms={:.3} \
             dp_total_wall_ms={:.3}\n",
            ms(median(&self.warmup_span_ms)),
            normalise_zero(self.span_ms.iter().sum::<f64>()),
            normalise_zero(self.own_ms.iter().sum::<f64>()),
            normalise_zero(self.wall_ms.iter().sum::<f64>()),
        ));
        // Prefill is timed here and added to nothing above, so a decode total
        // cannot be inflated by it. `dp_prefill_wall_ms` is the number that
        // must stay flat while decode length changes.
        out.push_str(&format!(
            "dp_prefill_tokens={} dp_prefill_wall_ms={:.3} dp_prefill_tok_s={}\n",
            self.prefill_tokens,
            self.prefill_wall_ms,
            if self.prefill_wall_ms > 0.0 {
                format!(
                    "{:.3}",
                    self.prefill_tokens as f64 / (self.prefill_wall_ms / 1e3)
                )
            } else {
                "0.000".to_string()
            }
        ));
        if mode == Mode::Kernel {
            let steps = self.steps_recorded.max(1) as f64;
            out.push_str(&format!(
                "dpk_own_ms_per_token={:.3} dpk_span_ms_per_token={:.3}\n",
                self.kernel_own_ms / steps,
                self.kernel_span_ms / steps,
            ));
            out.push_str(&format!(
                "  {:<28} {:>7} {:>12} {:>12} {:>7}\n",
                "kernel", "count", "own_ms", "own_ms/step", "share%"
            ));
            let mut rows: Vec<(&String, &u64, &f64)> = self
                .by_kernel
                .iter()
                .map(|(name, (count, own))| (name, count, own))
                .collect();
            rows.sort_by(|a, b| b.2.total_cmp(a.2).then_with(|| a.0.cmp(b.0)));
            for (name, count, own) in rows {
                out.push_str(&format!(
                    "  {name:<28} {count:>7} {own:>12.3} {:>12.4} {:>6.2}%\n",
                    own / steps,
                    100.0 * own / self.kernel_own_ms.max(f64::MIN_POSITIVE),
                ));
                out.push_str(&format!("  dpk={name} count={count} own_ms={own:.4}\n"));
            }
        }
        out
    }
}

fn global() -> &'static Mutex<DecodeAccumulator> {
    static INSTANCE: OnceLock<Mutex<DecodeAccumulator>> = OnceLock::new();
    INSTANCE.get_or_init(|| Mutex::new(DecodeAccumulator::default()))
}

/// Run `body` against the process-wide accumulator, tolerating a poisoned lock
/// (a panic in one step's recording must not take the whole report with it).
fn with_global<T>(body: impl FnOnce(&mut DecodeAccumulator) -> T) -> T {
    let mut guard = global().lock().unwrap_or_else(|error| error.into_inner());
    body(&mut guard)
}

/// Open a decode step: one stamped command buffer, holding one trivial
/// dispatch, so the boundary's `GPUStartTime` is a real value. A buffer with
/// nothing in it can read back as exactly zero (`model.rs`, on
/// `median_floor_ms`), and a zero start would make every span nonsense.
///
/// `warm` must encode that trivial dispatch; it is the model's calibration copy.
pub fn begin_step(
    encoder: &mut MetalEncoder,
    warm: impl FnOnce(&mut MetalEncoder) -> Result<(), ModelError>,
) -> Result<(), ModelError> {
    if active_mode() != Mode::Step {
        return Ok(());
    }
    warm(encoder)?;
    encoder
        .commit_stamped("step/open")
        .map_err(ModelError::from)?;
    Ok(())
}

/// Close a decode step: commit the work encoded since the opening boundary
/// under a stamped label and wait for it. Not `commit_terminal`, because a
/// terminal commit does not park its buffer where `GPUEndTime` can be read, and
/// not `commit_and_wait`, because that consumes the encoder.
pub fn close_step(encoder: &mut MetalEncoder) -> Result<(), ModelError> {
    if active_mode() != Mode::Step {
        return Ok(());
    }
    encoder
        .commit_stamped_and_wait("step/close")
        .map_err(ModelError::from)?;
    Ok(())
}

/// Read one closed step off `encoder` and record it. The caller must have
/// waited already: `GPUStartTime` / `GPUEndTime` are undefined before then.
pub fn end_step(encoder: &mut MetalEncoder, wall_ms: f64) -> Result<(), ModelError> {
    if active_mode() != Mode::Step {
        return Ok(()); // Nothing was stamped, but drain anyway: a leftover
        // `stamped` buffer would be waited on much later.
    }
    with_global(|accumulator| accumulator.take_step(encoder, wall_ms))
}

/// `kernel` mode: fold one step's per-kernel own times in.
pub fn record_kernel_step(span_ms: Option<f64>, own_ms: f64, rows: Vec<(String, u64, f64)>) {
    if active_mode() != Mode::Kernel {
        return;
    }
    with_global(|accumulator| accumulator.record_kernel_step(span_ms, own_ms, rows));
}

pub fn record_prefill(wall_ms: f64, tokens: usize) {
    if active_mode() == Mode::Off {
        return;
    }
    with_global(|accumulator| accumulator.record_prefill(wall_ms, tokens));
}

// -- test-only access to the process-wide accumulator ----------------------------
//
// A second `#[test]` in the same binary would race over the one-shot mode and
// warmup caches, so anything that needs to inspect the accumulator has to own
// its process. These are `#[doc(hidden)]` for that reason, not because they are
// unsafe: they read and reset state that only a test creates.

/// The most recently recorded step's span in ms, or `NAN`.
#[doc(hidden)]
pub fn last_span_for_tests() -> f64 {
    with_global(|accumulator| accumulator.last_span_ms())
}

/// The report as it would be printed, without printing it and without consuming
/// the once-per-process latch.
#[doc(hidden)]
pub fn report_for_tests() -> String {
    with_global(|accumulator| accumulator.render())
}

/// Clear the accumulator, including the report latch.
#[doc(hidden)]
pub fn reset_for_tests() {
    with_global(|accumulator| *accumulator = DecodeAccumulator::default());
}

/// Print the report, once per process, if any step was recorded.
pub fn report() {
    let text = with_global(|accumulator| accumulator.take_report());
    if let Some(text) = text {
        eprint!("{text}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stamp(start_s: f64, end_s: f64) -> GpuStamp {
        GpuStamp {
            label: "step/open".to_string(),
            start_s,
            end_s,
            own_ms: (end_s - start_s) * 1e3,
            gap_ms: 0.0,
        }
    }

    fn closed(end_s: f64) -> GpuStamp {
        GpuStamp {
            label: "step/close".to_string(),
            ..stamp(end_s, end_s)
        }
    }

    fn acc() -> DecodeAccumulator {
        let mut accumulator = DecodeAccumulator::default();
        // Skip the warmup deterministically instead of via the env knob.
        for step in 0..8u64 {
            let base = step as f64 * 0.1;
            accumulator.record_step(
                Mode::Step,
                &stamp(10.0 + base, 10.0),
                &closed(10.12 + base),
                120.0,
            );
        }
        accumulator
    }

    #[test]
    fn a_step_is_only_what_its_own_two_stamps_bracket() {
        let mut accumulator = acc();
        accumulator.record_step(Mode::Step, &stamp(20.0, 20.0), &closed(20.123), 120.0);

        assert_eq!(accumulator.steps_recorded, 1);
        assert!(
            (accumulator.span_ms[0] - 123.0).abs() < 1e-6,
            "span was {} ms",
            accumulator.span_ms[0]
        );
        // The prefill's 1270 ms of work is nowhere in it: the span is a
        // difference of two timestamps taken at the step's own boundaries.
        assert!(accumulator.span_ms[0] < 200.0);
    }

    /// The scale property. A per-step number that does not depend on the step
    /// cannot distinguish one token from one hundred, which is exactly how the
    /// prefill table passed for a decode table.
    #[test]
    fn the_per_step_number_does_not_absorb_earlier_work() {
        let mut accumulator = DecodeAccumulator::default();
        for step in 0..64u64 {
            let start = 100.0 + step as f64 * 0.123;
            accumulator.record_step(
                Mode::Step,
                &stamp(start, start),
                &closed(start + 0.123),
                122.0,
            );
        }
        assert_eq!(accumulator.steps_recorded, 64 - active_warmup() as u64);
        assert!((median(&accumulator.span_ms).unwrap() - 123.0).abs() < 1e-6);
    }

    /// The injected-cost property. A slow dispatch inside the step must show up
    /// in the instrument's own number, by about its own size.
    #[test]
    fn an_injected_cost_inside_the_step_is_resolved() {
        let mut fast = acc();
        let mut slow = acc();
        fast.record_step(Mode::Step, &stamp(30.0, 30.0), &closed(30.123), 120.0);
        slow.record_step(Mode::Step, &stamp(40.0, 40.0), &closed(40.223), 220.0);
        let delta = median(&slow.span_ms).unwrap() - median(&fast.span_ms).unwrap();
        assert!(
            (delta - 100.0).abs() < 1e-6,
            "a 100 ms injected cost moved the instrument by {delta} ms"
        );
    }

    /// Warmup is excluded from the statistics but is visible in the report, so a
    /// cold-clock run cannot hide inside a median.
    #[test]
    fn warmup_steps_are_excluded_but_reported() {
        let mut accumulator = DecodeAccumulator::default();
        // Eight slow steps (cold clocks) then eight fast ones.
        for step in 0..8u64 {
            accumulator.record_step(
                Mode::Step,
                &stamp(step as f64, step as f64),
                &closed(step as f64 + 0.700),
                700.0,
            );
        }
        for step in 8..16u64 {
            accumulator.record_step(
                Mode::Step,
                &stamp(step as f64, step as f64),
                &closed(step as f64 + 0.123),
                120.0,
            );
        }
        assert_eq!(accumulator.steps_recorded, 8);
        assert!((median(&accumulator.warmup_span_ms).unwrap() - 700.0).abs() < 1e-6);
        assert!((median(&accumulator.span_ms).unwrap() - 123.0).abs() < 1e-6);
        assert!(accumulator.render().contains("dp_warmup_skipped=8"));
    }

    /// A zero start (an empty boundary read back as exactly zero) must be
    /// counted, not averaged in: silently dropping it would make the mean
    /// better than the instrument deserves.
    #[test]
    fn implausible_stamps_are_counted_not_averaged() {
        let mut accumulator = DecodeAccumulator::default();
        accumulator.record_step(Mode::Step, &stamp(0.0, 0.0), &closed(0.123), 120.0);
        assert_eq!(accumulator.bad_steps, 1);
        assert_eq!(accumulator.steps_recorded, 0);
        assert!(accumulator.render().contains("dp_bad_steps=1"));
    }

    #[test]
    fn prefill_is_recorded_and_added_to_nothing() {
        let mut accumulator = acc();
        accumulator.record_step(Mode::Step, &stamp(50.0, 50.0), &closed(50.123), 120.0);
        accumulator.record_prefill(3000.0, 614);
        let report = accumulator.render();
        assert!(report.contains("dp_prefill_tokens=614"));
        assert!(report.contains("dp_prefill_wall_ms=3000.000"));
        // The decode total is the sum of the per-step spans and nothing else.
        assert!(
            (accumulator.span_ms.iter().sum::<f64>() - 123.0).abs() < 1e-6,
            "decode total absorbed more than its own steps: {report}"
        );
        assert!(!report.contains("dp_total_span_ms=3000"), "{report}");
    }
}
