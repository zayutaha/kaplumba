//! Prefill cost decomposition on the *shipping* GPU path (`forward_buffer_into`).
//!
//! Why this exists. Every prefill number recorded so far came from a **different
//! code path than the one that ships**:
//!
//! - `kaplumba --benchmark` calls `run_benchmark` -> `TextModel::forward`
//!   (`model.rs:3870`), which is the *host slice* path: `DecoderLayer::forward`
//!   per layer, host KV cache, host round-trip. Only `--prompt` without
//!   `--legacy-path` reaches `generate_gpu` -> `forward_buffer_into`.
//! - The CLI's `prefill_tok_s` at a short prompt is `tokens / (fixed + marginal)`,
//!   and the fixed part is large (Metal pipeline creation for every prefill-only
//!   kernel on its first dispatch, because `warmup_decode` at `model.rs:2978`
//!   runs a *one-token* forward and so never touches `linear_gemm_mma` or the
//!   `attention_gqa` prefill instantiation).
//!
//! So this tool measures, in ONE process, for each prompt length L:
//!
//!   1. `t1` -- first prefill of that length (cold for L)
//!   2. `t2`, `t3` -- repeats after `reset_caches`, with pipelines already warm
//!
//! `t1 - t2` is the cold-start tax; `t2` is the marginal prefill rate. Fitting
//! `t(L) = a + L/r` over the warm numbers separates the fixed cost from the
//! per-token cost without needing a microbenchmark.
//!
//! It also prints the dispatch count per prefill and the scratch residency, so
//! the "fixed-size, context-independent scratch" claim in AGENTS.md §8 is a
//! printed number rather than an assertion.
//!
//! Usage: `prefill_probe <model_dir> [L ...]` (default 20 120 614 2048)

use std::time::Instant;

use kaplumba_engine::model::TextModel;
use kaplumba_kernels::encoder::MetalEncoder;

/// Mean of the two middle values for even length. A `values[len / 2]` index is
/// wrong here: on a 2-element slice it returns the MAX, which silently turned a
/// 7362 ms repeat into a reported 12398 ms and dragged the fit's intercept from
/// ~130 ms to ~1830 ms.
fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let model_dir = args
        .next()
        .unwrap_or_else(|| format!("{}/.kaplumba/models/Gxma/Gxma", std::env::var("HOME").unwrap_or_default()));
    let lengths: Vec<usize> = args
        .map(|value| value.parse::<usize>().expect("length must be an integer"))
        .collect();
    let lengths = if lengths.is_empty() {
        vec![20, 120, 614, 2048]
    } else {
        lengths
    };

    let load_started = Instant::now();
    let mut model = TextModel::load_from_dir(std::path::Path::new(&model_dir)).expect("model load");
    eprintln!(
        "load {:.0} ms | scratch_rows={} | allocated={:.2} GiB (weights {:.2} / kv {:.2} / scratch {:.3})",
        load_started.elapsed().as_secs_f64() * 1e3,
        model.scratch_max_tokens(),
        model.allocated_bytes() as f64 / (1u64 << 30) as f64,
        model.allocation_breakdown().0 as f64 / (1u64 << 30) as f64,
        model.allocation_breakdown().1 as f64 / (1u64 << 30) as f64,
        model.allocation_breakdown().2 as f64 / (1u64 << 30) as f64,
    );

    let queue = model.forward_queue();
    let vocab = model.vocab_size();
    println!(
        "{:>6} {:>10} {:>10} {:>10} {:>10} {:>9} {:>9} {:>9} {:>8}",
        "L", "cold_ms", "warm_med", "warm_lo", "warm_hi", "spread%", "cold_tps", "warm_tps", "dispatch"
    );

    let mut warm_points: Vec<(usize, f64)> = Vec::new();
    for length in lengths {
        // Token 0 is <bos>; the rest repeat a real id so nothing is degenerate.
        let tokens: Vec<u32> = (0..length).map(|i| if i == 0 { 2 } else { 1000 + (i as u32 % 1000) }).collect();
        let positions: Vec<u32> = (0..length as u32).collect();

        let mut timings = Vec::new();
        let mut dispatches = 0usize;
        for attempt in 0..5 {
            model.reset_caches().expect("reset_caches");
            model.reset_gpu_caches();
            let started = Instant::now();
            let mut encoder = MetalEncoder::new(&queue).expect("encoder");
            model
                .forward_buffer_into(&tokens, &positions, &mut encoder)
                .expect("prefill");
            dispatches = encoder.dispatches_encoded();
            encoder.commit_and_wait().expect("commit");
            let ms = started.elapsed().as_secs_f64() * 1e3;
            timings.push(ms);
            if attempt == 0 {
                // Read the logits back so the compiler cannot elide the work, and
                // so a wrong answer is visible rather than silent.
                let logits = model.last_logits();
                let best = logits[..vocab]
                    .iter()
                    .enumerate()
                    .max_by(|a, b| {
                        half::f16::from_bits(*a.1).partial_cmp(&half::f16::from_bits(*b.1)).unwrap()
                    })
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                eprintln!("  L={length} cold argmax token id {best} (logit row {})", logits.len());
            }
        }
        let cold = timings[0];
        let mut warm = timings[1..].to_vec();
        let warm_med = median(&mut warm);
        let lo = *warm.first().expect("at least one warm sample");
        let hi = *warm.last().expect("at least one warm sample");
        println!(
            "{length:>6} {cold:>10.2} {warm_med:>10.2} {:>10.2} {:>10.2} {:>9.2} {:>9.2} {:>9.2} {dispatches:>9}",
            lo,
            hi,
            100.0 * (hi - lo) / warm_med,
            length as f64 / cold * 1e3,
            length as f64 / warm_med * 1e3,
        );
        warm_points.push((length, warm_med));
    }

    // Least-squares fit of warm_ms = a + b*L, printed as fixed ms and tok/s.
    if warm_points.len() >= 2 {
        let n = warm_points.len() as f64;
        let sum_l: f64 = warm_points.iter().map(|(l, _)| *l as f64).sum();
        let sum_t: f64 = warm_points.iter().map(|(_, t)| *t).sum();
        let sum_ll: f64 = warm_points.iter().map(|(l, _)| (*l as f64) * (*l as f64)).sum();
        let sum_lt: f64 = warm_points.iter().map(|(l, t)| (*l as f64) * t).sum();
        let denom = n * sum_ll - sum_l * sum_l;
        if denom.abs() > 1e-9 {
            let b = (n * sum_lt - sum_l * sum_t) / denom;
            let a = (sum_t - b * sum_l) / n;
            println!(
                "\nfit: warm_ms = {a:.1} + {b:.5} * L   ->  fixed {a:.1} ms, marginal {:.1} tok/s",
                1e3 / b
            );
        }
    }
    println!("\nfinal scratch_rows={}", model.scratch_max_tokens());
}
