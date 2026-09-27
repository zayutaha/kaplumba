use std::time::Duration;

use kaplumba_engine::metrics::{
    BenchmarkPlan, MetricsError, RunMetrics, benchmark_result, measure_rss,
};

#[test]
fn metrics_rates_and_benchmark_plan_are_deterministic() {
    let metrics = RunMetrics {
        prefill: Duration::from_secs(2),
        decode: Duration::from_secs(4),
        prefill_tokens: 100,
        decode_tokens: 40,
        peak_rss_bytes: 123,
        ..RunMetrics::default()
    };
    assert!((metrics.prefill_tokens_per_second() - 50.0).abs() < 1e-9);
    assert!((metrics.decode_tokens_per_second() - 10.0).abs() < 1e-9);
    let plan = BenchmarkPlan {
        prompt_tokens: 2048,
        warmup_steps: 8,
        measured_tokens: 128,
    };
    let result = benchmark_result(metrics, plan).unwrap();
    assert_eq!(result.measured_tokens, 128);
    assert_eq!(result.warmup_steps, 8);
    assert_eq!(result.peak_rss_bytes, 123);
}

#[test]
fn metrics_reject_zero_plan_and_empty_rates() {
    assert!(matches!(
        benchmark_result(
            RunMetrics::default(),
            BenchmarkPlan {
                prompt_tokens: 0,
                warmup_steps: 8,
                measured_tokens: 128
            }
        ),
        Err(MetricsError::InvalidTokenCount)
    ));
    let metrics = RunMetrics::default();
    assert_eq!(metrics.prefill_tokens_per_second(), 0.0);
    assert_eq!(metrics.decode_tokens_per_second(), 0.0);
    let _ = measure_rss().unwrap();
}
