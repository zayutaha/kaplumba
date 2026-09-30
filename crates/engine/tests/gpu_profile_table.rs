//! Tests for the `KAPLUMBA_GPU_PROFILE=1` accumulator's bookkeeping, with no
//! model and no GPU: `GpuProfile::ingest` takes stamps directly, so the
//! aggregation, the sort order and the report text are all testable from
//! synthetic numbers. The real end-to-end numbers need the checkpoint and are
//! measured by running the CLI with the env var set.

use kaplumba_engine::model::GpuProfile;
use kaplumba_kernels::encoder::GpuStamp;

fn stamp(label: &str, own_ms: f64, gap_ms: f64, start_s: f64) -> GpuStamp {
    GpuStamp {
        label: label.to_string(),
        start_s,
        end_s: start_s + own_ms / 1e3,
        own_ms,
        gap_ms,
    }
}

/// One row of the kernel table as (count, total_ms, mean_ms), parsed from the
/// rendered text so the test checks the numbers rather than the column padding.
fn row(report: &str, kernel: &str) -> (u64, f64, f64) {
    let line = report
        .lines()
        .find(|line| line.trim_start().starts_with(kernel) && line.split_whitespace().count() == 4)
        .unwrap_or_else(|| panic!("no row for {kernel} in:\n{report}"));
    let fields: Vec<&str> = line.split_whitespace().collect();
    assert_eq!(fields[0], kernel, "wrong row picked: {line}");
    (
        fields[1].parse().expect("count"),
        fields[2].parse().expect("total"),
        fields[3].parse().expect("mean"),
    )
}

#[test]
fn disabled_profile_absorbs_nothing() {
    let mut profile = GpuProfile::new(false);
    assert!(!profile.enabled());
    // A disabled profile must not accumulate even if a caller reaches `ingest`,
    // so the non-profiling path stays free of profiling work.
    profile.ingest(&[stamp("layer00/q_proj", 5.0, 0.0, 0.0)]);
    let report = profile.render();
    assert!(
        report.contains("sum of per-kernel GPU own_ms = 0.000 ms"),
        "{report}"
    );
    assert!(report.contains("0.000 ms   inter-boundary gap"), "{report}");
}

#[test]
fn aggregates_by_kernel_across_layers_and_chunks() {
    let mut profile = GpuProfile::new(true);
    profile.ingest(&[
        stamp("layer00/q_proj", 4.0, 0.5, 0.0),
        stamp("layer00/k_proj", 1.0, 0.25, 0.004),
        stamp("layer01/q_proj", 6.0, 0.25, 0.00525),
        stamp("chunk/embedding", 2.0, 0.25, 0.0115),
    ]);
    let report = profile.render();

    // q_proj is summed over both layers: 2 invocations, 10 ms, 5 ms mean.
    assert_eq!(row(&report, "q_proj"), (2, 10.0, 5.0));
    assert_eq!(row(&report, "k_proj"), (1, 1.0, 1.0));
    // A per-chunk label keeps its qualified name in the same table.
    assert_eq!(row(&report, "chunk/embedding"), (1, 2.0, 2.0));
    // Own time is summed; gaps are reported but never folded into it.
    assert!(report.contains("own_ms = 13.000 ms"), "{report}");
    assert!(report.contains("gap = 1.250 ms"), "{report}");
    // Per-layer view: layer01 holds one boundary (6 ms), layer00 two (5 ms).
    assert!(
        report.contains(
            "per-layer: 2 layers, sum 11.000 ms, slowest layer01 6.000 ms, \
             fastest layer00 5.000 ms"
        ),
        "per-layer summary wrong:\n{report}"
    );
    assert!(report.contains("layer01=6.0 layer00=5.0"), "{report}");
}

#[test]
fn rows_are_sorted_by_total_descending() {
    let mut profile = GpuProfile::new(true);
    profile.ingest(&[
        stamp("chunk/embedding", 3.0, 0.0, 0.0),
        stamp("layer00/attention", 30.0, 0.0, 0.003),
        stamp("layer00/kv_norm_rope", 1.0, 0.0, 0.033),
    ]);
    let report = profile.render();
    let position = |kernel: &str| report.find(kernel).unwrap_or_else(|| panic!("{kernel}"));
    assert!(
        position("attention") < position("chunk/embedding")
            && position("chunk/embedding") < position("kv_norm_rope"),
        "rows are not sorted by total descending:\n{report}"
    );
}

#[test]
fn a_non_layer_prefix_is_not_a_layer() {
    // `head/gemv` is per-chunk work: it must reach the kernel table and must not
    // create a bogus layer.
    let mut profile = GpuProfile::new(true);
    profile.ingest(&[stamp("head/gemv", 12.0, 0.0, 0.0)]);
    let report = profile.render();
    assert_eq!(row(&report, "head/gemv"), (1, 12.0, 12.0));
    assert!(!report.contains("per-layer:"), "{report}");
}

#[test]
fn spans_the_first_start_to_the_last_end() {
    let mut profile = GpuProfile::new(true);
    profile.ingest(&[
        stamp("layer00/q_proj", 5.0, 1.0, 10.0),
        stamp("layer01/q_proj", 5.0, 2.0, 10.006),
    ]);
    let report = profile.render();
    // 10.011 - 10.000: the wall span covers the gaps, unlike own_ms.
    assert!(report.contains("own_ms = 10.000 ms"), "{report}");
    assert!(report.contains("wall span = 11.000 ms"), "{report}");
}

#[test]
fn calibration_batch_keeps_the_stamps_it_swallows_with_it() {
    // The calibration runs at the top of a chunk, where the previous chunk's tail
    // boundaries can still be retained: `take_gpu_stamps` hands it real work
    // alongside its own `cal/*` samples. Dropping those silently lost 10 of 2718
    // boundaries in the first working run -- including every earlier chunk's
    // `head/*` and `chunk/*` rows, which then read as a count of 1.
    let mut profile = GpuProfile::new(true);
    let (empty, one) = profile.ingest_calibration(&[
        stamp("cal/empty", 0.0001, 0.0, 0.0),
        stamp("cal/one", 0.007, 0.0, 0.0001),
        stamp("head/gemv", 8.0, 0.1, 0.0071),
        stamp("chunk/greedy_pick", 4.0, 0.1, 0.0151),
    ]);
    assert_eq!(empty, vec![0.0001]);
    assert_eq!(one, vec![0.007]);
    assert_eq!(row(&profile.render(), "head/gemv"), (1, 8.0, 8.0));
    assert_eq!(row(&profile.render(), "chunk/greedy_pick"), (1, 4.0, 4.0));
    // The real work is neither in the calibration samples nor in the totals twice.
    assert!(profile.render().contains("own_ms = 12.000 ms"));
    assert!(
        !profile.render().contains("WARNING"),
        "{}",
        profile.render()
    );
}

#[test]
fn unavailable_overhead_is_reported_rather_than_faked() {
    let report = GpuProfile::new(true).render();
    assert!(
        report.contains("overhead: unavailable"),
        "a missing calibration must say so, not print a zero:\n{report}"
    );
}

#[test]
fn measured_overhead_is_printed_as_the_floor() {
    let mut profile = GpuProfile::new(true);
    profile.set_calibration(Some(0.004), Some(0.011));
    let report = profile.render();
    assert!(
        report.contains("overhead floor: empty commit 0.0040 ms | one trivial dispatch 0.0110 ms"),
        "{report}"
    );
    // Only the first calibration is kept, so a later chunk cannot overwrite the
    // floor a reader already saw quoted.
    profile.set_calibration(Some(9.0), Some(9.0));
    assert!(profile.render().contains("empty commit 0.0040 ms"));
}
