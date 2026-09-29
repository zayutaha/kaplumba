use std::time::Duration;

use thiserror::Error;

use crate::tokenizer::GemmaTokenizer;

pub const BENCHMARK_SENTENCE: &str =
    "The quick brown fox jumps over the lazy dog while the model measures steady progress. ";

#[derive(Debug, Error)]
pub enum MetricsError {
    #[error("benchmark token counts must be positive")]
    InvalidTokenCount,
    #[error("benchmark tokenizer failed: {0}")]
    Tokenizer(String),
    #[error("benchmark duration must be finite")]
    InvalidDuration,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
/// Memory note: `peak_resident_pages_bytes` is `ru_maxrss`, i.e. resident pages only. On
/// this engine it reads ~7.0 GiB while the kernel charges ~15.7 GiB, because
/// Metal buffer pages are not counted as resident. The budget-relevant figure
/// is `phys_footprint` (Activity Monitor "Memory"), which must be measured
/// externally with `/usr/bin/time -l`.
pub struct RunMetrics {
    pub load: Duration,
    pub ttft: Duration,
    pub prefill: Duration,
    pub decode: Duration,
    pub prefill_tokens: u64,
    pub decode_tokens: u64,
    pub peak_resident_pages_bytes: u64,
}

impl RunMetrics {
    pub fn prefill_tokens_per_second(&self) -> f64 {
        rate(self.prefill_tokens, self.prefill)
    }

    pub fn decode_tokens_per_second(&self) -> f64 {
        rate(self.decode_tokens, self.decode)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BenchmarkPlan {
    pub prompt_tokens: usize,
    pub warmup_steps: usize,
    pub measured_tokens: usize,
}

impl BenchmarkPlan {
    pub fn validate(self) -> Result<(), MetricsError> {
        if self.prompt_tokens == 0 || self.warmup_steps == 0 || self.measured_tokens == 0 {
            return Err(MetricsError::InvalidTokenCount);
        }
        Ok(())
    }
}

pub fn fixed_benchmark_prompt(
    tokenizer: &GemmaTokenizer,
    target_tokens: usize,
) -> Result<Vec<u32>, MetricsError> {
    if target_tokens == 0 {
        return Err(MetricsError::InvalidTokenCount);
    }
    let mut text = String::new();
    let repeats = target_tokens / 8 + 32;
    for _ in 0..repeats {
        text.push_str(BENCHMARK_SENTENCE);
    }
    let mut ids = tokenizer
        .encode_prompt(&text)
        .map_err(|error| MetricsError::Tokenizer(error.to_string()))?;
    if ids.len() < target_tokens {
        return Err(MetricsError::InvalidTokenCount);
    }
    ids.truncate(target_tokens);
    Ok(ids)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BenchmarkResult {
    pub prefill_tokens_per_second: f64,
    pub decode_tokens_per_second: f64,
    pub measured_tokens: usize,
    pub warmup_steps: usize,
    pub peak_resident_pages_bytes: u64,
}

pub fn measure_rss() -> Result<u64, MetricsError> {
    #[cfg(target_os = "macos")]
    {
        let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
        let result = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
        if result != 0 {
            return Ok(0);
        }
        let usage = unsafe { usage.assume_init() };
        Ok(usage.ru_maxrss as u64)
    }
    #[cfg(not(target_os = "macos"))]
    {
        Ok(0)
    }
}

pub fn benchmark_result(
    metrics: RunMetrics,
    plan: BenchmarkPlan,
) -> Result<BenchmarkResult, MetricsError> {
    plan.validate()?;
    Ok(BenchmarkResult {
        prefill_tokens_per_second: metrics.prefill_tokens_per_second(),
        decode_tokens_per_second: metrics.decode_tokens_per_second(),
        measured_tokens: plan.measured_tokens,
        warmup_steps: plan.warmup_steps,
        peak_resident_pages_bytes: metrics.peak_resident_pages_bytes,
    })
}

fn rate(tokens: u64, duration: Duration) -> f64 {
    if tokens == 0 || duration.is_zero() {
        0.0
    } else {
        tokens as f64 / duration.as_secs_f64()
    }
}

/// Peak memory footprint charged to this process, per Activity Monitor.
///
/// NOTE: this is obtained by wrapping the process in `/usr/bin/time -l`, which
/// reports the kernel's `phys_footprint` (resident + compressed + wired, so
/// Metal buffer pages count). A hand-rolled `task_info` FFI was tried and
/// rejected: `task_vm_info_data_t` has grown across releases, so the field
/// offset is not stable and the query returned garbage. `getrusage().ru_maxrss`
/// is also unsuitable because it counts resident pages only and under-reports a
/// Metal engine. See `measure_footprint_with_time`.
/// Peak CPU-resident pages (`getrusage().ru_maxrss`) -- NOT the memory budget.
///
/// A hand-rolled `task_info` FFI for `phys_footprint` was tried TWICE and
/// rejected both times: `task_vm_info_data_t` has grown across macOS releases,
/// so the field offset is not stable and the query returns garbage. `libc` does
/// not export `TASK_VM_INFO` or `task_vm_info_data_t` at all, so any such
/// attempt means hand-writing the struct and guessing the offset -- which
/// produces a plausible wrong number, the worst possible failure for a metric
/// used to judge a memory budget.
///
/// The correct figure is obtained by wrapping the process in `/usr/bin/time -l`
/// and reading `peak memory footprint`; see `crates/engine/tests/memory_proof.rs`.
pub fn measure_footprint_note() -> &'static str {
    "use /usr/bin/time -l 'peak memory footprint' (see crates/engine/tests/memory_proof.rs)"
}
