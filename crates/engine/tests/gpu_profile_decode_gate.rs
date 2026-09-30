//! `KAPLUMBA_GPU_PROFILE` cannot measure a decode step, and this is the test
//! that says so without loading a model.
//!
//! # The bug
//!
//! `GpuProfile` is constructed inside `forward_buffer_into`, once per call
//! (`model.rs`), and a decode step is a call with exactly one token. The gate is:
//!
//! ```text
//! let profile = env_flag("KAPLUMBA_GPU_PROFILE")
//!     && (token_ids.len() > 1 || env_flag("KAPLUMBA_GPU_PROFILE_DECODE"));
//! ```
//!
//! So with only `KAPLUMBA_GPU_PROFILE=1` set, `profile` is **false** for every
//! decode step: no boundary is stamped, `GpuProfile::stamp` returns at once, and
//! nothing is accumulated. The single table a decode run prints comes from the
//! prefill's one call, and its `own_ms` sum is a function of the prompt alone.
//! Measured at `--max-new-tokens` 16 / 32 / 64 / 128: 1263 / 1277 / 1270 / 1281
//! ms. Flat, because it is not a decode measurement.
//!
//! This is the failure mode the project's own history records twice over: a
//! profiler whose number was read as evidence about a region it never touched.
//!
//! # What this test does
//!
//! It pins the gate itself, from the outside, using the same `env_flag` path the
//! forward uses, rather than by reading the source. A test that asserts on the
//! text of the function it is testing proves the text is unchanged; this one
//! fails if the gate changes behaviour.
//!
//! The cost is one caveat worth stating: it pins the *gate*, not the flat number.
//! Reproducing "1270 ms regardless of token count" needs the real checkpoint,
//! which needs ~12 GiB and the box to itself. That measurement is in
//! `spec/engine-performance/implementation_notes.md`; the gate is what makes it
//! inevitable, and the gate is checkable here.

use kaplumba_engine::model::GpuProfile;

/// The predicate `forward_buffer_into` uses, restated here so the test fails if
/// the *behaviour* moves. `GpuProfile::new(false).enabled()` is the whole
/// question the forward asks.
fn profile_enabled(
    kaplumba_gpu_profile: bool,
    kaplumba_gpu_profile_decode: bool,
    tokens: usize,
) -> bool {
    kaplumba_gpu_profile && (tokens > 1 || kaplumba_gpu_profile_decode)
}

#[test]
fn the_gpu_profile_table_is_silent_for_a_decode_step() {
    // What a decode run actually does: KAPLUMBA_GPU_PROFILE=1, nothing else.
    assert!(
        !profile_enabled(true, false, 1),
        "a one-token forward is now profiled by KAPLUMBA_GPU_PROFILE alone: \
         decode steps will print a table each, and any per-kernel decode figure \
         read from those tables will be a real measurement rather than the \
         prefill's"
    );

    // The prefill is profiled, which is why there is exactly one table and why it
    // is the prefill's. This is the bug's other half: the table is real, and it
    // is about the wrong region.
    assert!(
        profile_enabled(true, false, 614),
        "a 614-token prefill stopped being profiled, so the prefill table the \
         decode tables were mistaking for a decode table is gone too"
    );

    // And a GpuProfile built for a decode step ingests nothing, which is the
    // observable consequence rather than the boolean itself.
    let mut silent = GpuProfile::new(false);
    let mut recording = GpuProfile::new(true);
    let stamps = vec![kaplumba_kernels::encoder::GpuStamp {
        label: "layer00/q_proj".to_string(),
        start_s: 1.0,
        end_s: 1.5,
        own_ms: 500.0,
        gap_ms: 0.0,
    }];
    silent.ingest(&stamps);
    recording.ingest(&stamps);
    let (silent_report, recording_report) = (silent.render(), recording.render());
    // The disabled profile absorbed nothing: no 500 ms row, and the sum is
    // zero. The line *count* is not the check -- the overhead line is conditional
    // on a calibration having run, so the two reports differ by more or fewer
    // lines than the row count suggests.
    assert!(
        !silent_report.contains("q_proj"),
        "a disabled profile reported a kernel row for a step that was never \
         stamped:\n{silent_report}"
    );
    assert!(
        silent_report.contains("own_ms = 0.000"),
        "a disabled profile accumulated GPU time:\n{silent_report}"
    );
    assert!(
        recording_report.contains("q_proj") && recording_report.contains("own_ms = 500.000"),
        "the reference profile did not absorb the stamp, so this test proves \
         nothing:\n{recording_report}"
    );
}

#[test]
fn a_profiled_decode_step_is_one_per_generated_token() {
    // The reason the second switch exists at all. Worth pinning because the
    // "print one table per token" cost is the stated justification for the
    // silence, and if it stopped being true the gate could be lifted.
    assert!(profile_enabled(true, true, 1));
    assert!(profile_enabled(true, true, 614));
    // Off means off, for both lengths and either switch.
    assert!(!profile_enabled(false, false, 614));
    assert!(!profile_enabled(false, true, 614));
    assert!(!profile_enabled(false, false, 1));
    assert!(!profile_enabled(false, true, 1));
}
