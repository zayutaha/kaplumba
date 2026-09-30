//! When compute pipelines are *built*, and how long each build took.
//!
//! # Why this exists
//!
//! Every kernel family here builds all of its pipelines eagerly, inside
//! `new()`: `LinearKernels::new` walks `library.function_names()` and builds a
//! state for every `linear_gemv_ishizuki_*`, every `(codebook, aligned, bits)`
//! EXL3 slot, every NR0 variant, every `linear_gemv_fast*`, and every fixed
//! field; `ElementKernels::new` and `AttentionKernels::new` do the same for
//! their libraries. All three are behind a `OnceLock`, so the whole set is
//! built once per process, inside model load.
//!
//! That is an *argument*, and this module exists to make it a *measurement*.
//! The hazard is specific and it is the one that makes A/B numbers lie: a
//! kernel compiled inside a timed loop measures Metal's shader compilation,
//! not the kernel, and the error is not small and not signed. A variant that
//! happens to be reached for the first time on step 40 looks catastrophically
//! slow; a variant whose *sibling* was reached on step 1 looks fast. Neither
//! reading is about the code.
//!
//! So every pipeline construction in the crate goes through [`build`], which
//! times it and records the name. That gives three things:
//!
//! * `KAPLUMBA_PIPELINE_AUDIT=1` prints the count and the total build time, so
//!   "the library compiles once" stops being a claim about `new()` and becomes
//!   a number.
//! * [`snapshot`] / [`builds_since`] answer "did anything get compiled after
//!   this point", which is the question a timed loop has to be able to ask.
//! * `KAPLUMBA_ASSERT_NO_LATE_PIPELINES=1` turns that answer into a panic
//!   inside the engine's own decode loop, so a gap fails the run instead of
//!   quietly colouring a benchmark.
//!
//! # What this does and does not prove
//!
//! It counts calls to `newComputePipelineStateWithFunction`. That is the
//! Metal API boundary for "this kernel's GPU code now exists", so a kernel
//! that is absent from the count cannot be measuring compilation. It does NOT
//! cover the other first-use cost Metal charges -- loading a built pipeline
//! into the GPU's instruction memory on its first dispatch, which is what the
//! 1277 ms first-step cost in `warmup_decode` was. That one is a *dispatch*
//! property and only a `KAPLUMBA_STEP_TRACE` run can see it. Both are real;
//! they are different costs and this module measures the compile one.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use metal::{ComputePipelineState, Device, Library};

use crate::exl3::KernelError;

/// Set once load is over, to the build count at that moment. `0` means "not
/// armed", so [`assert_no_builds_since`] is a no-op until something arms it.
static ARMED: AtomicU64 = AtomicU64::new(0);

/// Number of `newComputePipelineStateWithFunction` calls made so far.
static BUILDS: AtomicU64 = AtomicU64::new(0);

/// Microseconds spent inside `newComputePipelineStateWithFunction`, summed.
static BUILD_MICROS: AtomicU64 = AtomicU64::new(0);

/// `(name, micros)` in build order. Names repeat (a name is looked up through
/// more than one table in some families), so this is a log and not a set.
fn log() -> &'static Mutex<Vec<(String, u64)>> {
    static INSTANCE: std::sync::OnceLock<Mutex<Vec<(String, u64)>>> = std::sync::OnceLock::new();
    INSTANCE.get_or_init(|| Mutex::new(Vec::new()))
}

fn record(name: &str, micros: u64) {
    BUILDS.fetch_add(1, Ordering::Relaxed);
    BUILD_MICROS.fetch_add(micros, Ordering::Relaxed);
    log()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .push((name.to_string(), micros));
}

/// Build `name` from `library`, recording the cost.
///
/// Every pipeline in the crate is built through here. A construction site that
/// bypasses it is invisible to the audit, which is why there is a test
/// (`every_pipeline_construction_goes_through_the_audit`) that greps the
/// sources: the audit is only worth what its coverage is.
pub fn build(
    device: &Device,
    library: &Library,
    name: &str,
) -> Result<ComputePipelineState, KernelError> {
    let function = library
        .get_function(name, None)
        .map_err(KernelError::Shader)?;
    let started = Instant::now();
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(KernelError::Pipeline)?;
    record(name, started.elapsed().as_micros() as u64);
    Ok(pipeline)
}

/// Number of pipelines built so far in this process.
pub fn build_count() -> u64 {
    BUILDS.load(Ordering::Relaxed)
}

/// Total microseconds spent in pipeline construction so far.
pub fn build_micros() -> u64 {
    BUILD_MICROS.load(Ordering::Relaxed)
}

/// The slowest builds so far, longest first, as `(name, micros)`. The point is
/// attribution: a 40 ms pipeline is what a late compile would cost, so this is
/// the size of the error a missed warmup can introduce.
pub fn slowest_builds(limit: usize) -> Vec<(String, u64)> {
    let guard = log().lock().unwrap_or_else(|error| error.into_inner());
    let mut rows: Vec<(String, u64)> = guard.clone();
    drop(guard);
    rows.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    rows.truncate(limit);
    rows
}

/// Every distinct kernel name that has been built, sorted.
pub fn built_names() -> Vec<String> {
    let guard = log().lock().unwrap_or_else(|error| error.into_inner());
    let mut names: Vec<String> = guard.iter().map(|(name, _)| name.clone()).collect();
    drop(guard);
    names.sort();
    names.dedup();
    names
}

/// `(name, micros)` for every build after `mark`, in build order.
///
/// `mark` is a *log index*, not a build count, and the two are the same number
/// by construction: `record` increments `BUILDS` and appends to the log
/// together, so index == count. Slicing from the index is what makes this exact
/// under concurrency, which the alternative is not: reading the length first and
/// subtracting a count races with an appending thread, and the engine is
/// single-threaded only by accident of the CLI.
pub fn builds_since(mark: u64) -> Vec<(String, u64)> {
    let guard = log().lock().unwrap_or_else(|error| error.into_inner());
    if guard.len() as u64 <= mark {
        return Vec::new();
    }
    guard[mark as usize..].to_vec()
}

/// Record where compilation currently stands, and start reporting anything
/// built after this point.
///
/// Called once, at the end of model load, by `warmup_decode`. Returns the mark
/// so the caller can hand it to [`builds_since`] later.
pub fn arm() -> u64 {
    let mark = log()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .len() as u64;
    ARMED.store(mark, Ordering::Relaxed);
    mark
}

/// The mark [`arm`] recorded, or `None` if nothing has armed.
pub fn armed_mark() -> Option<u64> {
    match ARMED.load(Ordering::Relaxed) {
        0 => None,
        mark => Some(mark),
    }
}

/// `KAPLUMBA_PIPELINE_AUDIT=1`: one line with the count, the total, and the
/// worst offenders. Printed from the engine, not here, so the crate does not
/// decide when a report happens.
pub fn audit_line() -> String {
    format!(
        "PIPELINE-AUDIT builds={} build_ms={:.1} distinct={} armed={} slowest={}",
        build_count(),
        build_micros() as f64 / 1000.0,
        built_names().len(),
        armed_mark().map_or_else(|| "no".to_string(), |mark| mark.to_string()),
        slowest_builds(5)
            .iter()
            .map(|(name, micros)| format!("{name}:{micros}us"))
            .collect::<Vec<_>>()
            .join(",")
    )
}

/// Panic if any pipeline was built since `mark`, if the gate is on.
///
/// Gated on `KAPLUMBA_ASSERT_NO_LATE_PIPELINES=1`, and a no-op when nothing has
/// armed, so the default path pays one relaxed atomic load.
///
/// The message names the kernels, because "a pipeline appeared" is not
/// actionable and "these four appeared" is: each one is a measurement waiting
/// to be wrong.
pub fn assert_no_builds_since(mark: u64) {
    if !assert_enabled() {
        return;
    }
    check_no_builds_since(mark, true);
}

/// The check itself, with the gate passed in.
///
/// Split from [`assert_no_builds_since`] so a test can drive it with the gate
/// on. The flag is memoised in a `OnceLock`, and a test binary cannot
/// guarantee it was started with the variable set, so a test that went through
/// the env path would silently exercise the *disabled* branch -- and
/// `assert_no_builds_since` returning early is exactly the behaviour that
/// would make every real run green while proving nothing.
#[doc(hidden)]
pub fn check_no_builds_since(mark: u64, enabled: bool) {
    if !enabled {
        return;
    }
    let late = builds_since(mark);
    if late.is_empty() {
        return;
    }
    let total: u64 = late.iter().map(|(_, micros)| micros).sum();
    let names: Vec<&str> = late.iter().map(|(name, _)| name.as_str()).collect();
    panic!(
        "KAPLUMBA_ASSERT_NO_LATE_PIPELINES: {} pipeline(s) were compiled after load \
         ({total} us total, slowest {:?}): {names:?}. A timed loop that reaches one \
         of these measures Metal shader compilation, not the kernel. Either warm \
         the path or accept that the number is not a kernel measurement.",
        late.len(),
        late.iter()
            .map(|(name, micros)| (name.as_str(), *micros))
            .max_by_key(|(_, micros)| *micros)
    );
}

fn assert_enabled() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var_os("KAPLUMBA_ASSERT_NO_LATE_PIPELINES").is_some())
}

/// [`audit_line`] plus what has been built since `mark`.
///
/// `when` is the caller's label for the point it sampled, so a report that
/// appears at two places in a run is attributable to one of them rather than
/// being two identical-looking lines. The `late_builds` field is the one that
/// decides whether a timed loop is safe to quote: a non-zero count there means
/// the number is partly shader compilation.
pub fn audit_line_with_mark(when: &str, mark: u64) -> String {
    let late = builds_since(mark);
    let late_micros: u64 = late.iter().map(|(_, micros)| micros).sum();
    format!(
        "{} at={when} late_builds={} late_build_ms={:.1}{}",
        audit_line(),
        late.len(),
        late_micros as f64 / 1000.0,
        if late.is_empty() {
            String::new()
        } else {
            format!(" late={:?}", late)
        }
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The audit is only worth its coverage, so this is the coverage claim made
    /// checkable. A `new_compute_pipeline_state_with_function` outside
    /// `pipeline_audit::build` is a pipeline the audit cannot see, and a
    /// benchmark that trusts the audit would then be trusting a partial count.
    ///
    /// `include_str!` rather than reading the working directory: the test must
    /// read the same bytes the crate compiled, or it is checking a checkout
    /// that may not be what ran.
    #[test]
    fn every_pipeline_construction_goes_through_the_audit() {
        // `encoder.rs` is excluded: its one call site is inside `#[cfg(test)]`
        // and compiles its own throwaway `burn` kernel, which is not part of
        // the engine's pipeline set.
        let sources = [
            ("linear.rs", include_str!("linear.rs")),
            ("elements.rs", include_str!("elements.rs")),
            ("attention.rs", include_str!("attention.rs")),
            ("exl3.rs", include_str!("exl3.rs")),
        ];
        for (name, source) in sources {
            for (number, line) in source.lines().enumerate() {
                let code = line.split("//").next().unwrap_or("");
                if code.contains("new_compute_pipeline_state_with_function") {
                    panic!(
                        "{name}:{} constructs a pipeline outside pipeline_audit::build: {code}",
                        number + 1
                    );
                }
            }
        }
    }

    /// The counter has to actually count, or `builds_since` answering "nothing
    /// was compiled" means nothing at all. This is the check that keeps the
    /// instrument honest: `build` is called, the count moves.
    #[test]
    fn build_increments_the_count_and_the_time() {
        if Device::system_default().is_none() {
            return;
        }
        let _serial = serial();
        let before = build_count();
        let before_micros = build_micros();
        probe("audit_probe", 2.0);
        assert_eq!(
            build_count(),
            before + 1,
            "build() did not record a build, so the audit cannot see a late compile"
        );
        assert!(
            build_micros() >= before_micros,
            "build time went backwards: {before_micros} -> {}",
            build_micros()
        );
        assert!(
            built_names().iter().any(|name| name == "audit_probe"),
            "the built kernel is not named in the log: {:?}",
            built_names()
        );
    }

    /// `builds_since` has to separate "before" from "after", because that
    /// separation is the entire basis for "nothing compiled inside the timed
    /// loop". A mark taken *after* the build must see nothing; a mark taken
    /// before it must see the build.
    #[test]
    fn builds_since_separates_before_from_after() {
        if Device::system_default().is_none() {
            return;
        }
        let _serial = serial();
        let before_mark = build_count();
        probe("audit_window", 2.5);
        let after_build = build_count();
        let after_mark = build_count();

        let seen = builds_since(before_mark);
        assert!(
            seen.iter().any(|(name, _)| name == "audit_window"),
            "a build after the mark was not reported: {seen:?}"
        );
        assert!(
            builds_since(after_mark).is_empty(),
            "a mark taken after the last build still reported work: {:?}",
            builds_since(after_mark)
        );
        assert!(after_build > before_mark, "the counter did not move at all");
    }

    /// A mark is a log *index*, and `builds_since` slices by it. If the two ever
    /// diverged, every `builds_since` result would be off by however much they
    /// had, and a late compile could be reported as an absent one -- the exact
    /// silent failure this module is for. So the invariant is pinned: after `N`
    /// builds the log holds `N` entries.
    #[test]
    fn the_mark_is_the_log_index_and_agrees_with_the_count() {
        if Device::system_default().is_none() {
            return;
        }
        let _serial = serial();
        probe("audit_index_1", 6.0);
        probe("audit_index_2", 7.0);
        let logged = log()
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .len() as u64;
        assert_eq!(
            logged,
            build_count(),
            "log length and build count have diverged: a mark is only a valid log index \
             while they agree, and builds_since slices by index"
        );
    }

    /// The gate must be able to fail. This is the AGENTS.md 3.4 requirement for
    /// the one piece of machinery whose whole job is to raise: if
    /// `check_no_builds_since` cannot be observed to panic, then a run that
    /// passes it has proved nothing.
    ///
    /// The panic is provoked by building a pipeline *after* the mark -- the
    /// exact situation the real check exists to catch -- and the test calls the
    /// real function with the gate on rather than panicking itself. Without it,
    /// a `builds_since` that always returned an empty `Vec` would leave every
    /// run green.
    #[test]
    #[should_panic(expected = "KAPLUMBA_ASSERT_NO_LATE_PIPELINES")]
    fn the_late_pipeline_assert_can_fail() {
        if Device::system_default().is_none() {
            panic!("no Metal device: the late-pipeline assert could not be exercised");
        }
        let _serial = serial();
        let mark = build_count();
        probe("audit_late", 3.0);
        // Sanity, inside the failing test: the mark must actually precede a
        // build, or the panic below would be an accident rather than the
        // mechanism it claims to demonstrate.
        assert!(
            !builds_since(mark).is_empty(),
            "the late build was not visible, so the assert below proves nothing"
        );
        check_no_builds_since(mark, true);
    }

    /// And it must NOT fire when nothing was built, or it would be a check
    /// that only ever fails.
    #[test]
    fn the_late_pipeline_assert_is_quiet_when_nothing_was_built() {
        let _serial = serial();
        let mark = build_count();
        check_no_builds_since(mark, true);
    }

    /// And the gate has to be able to be off, because the default run must not
    /// pay for it. `check_no_builds_since(mark, false)` on a mark that
    /// *precedes* a build is the only way to tell "disabled" apart from
    /// "nothing happened".
    #[test]
    fn the_late_pipeline_assert_respects_its_gate() {
        if Device::system_default().is_none() {
            return;
        }
        let _serial = serial();
        let mark = build_count();
        probe("audit_gated", 5.0);
        assert!(
            !builds_since(mark).is_empty(),
            "probe did not build, so the gate test proves nothing"
        );
        check_no_builds_since(mark, false);
    }

    /// The audit is process-wide, and `cargo test` runs a binary's tests on many
    /// threads. Without this the delta assertions above are racy against each
    /// other, not merely against the engine: two probes landing in the same
    /// window make `before + 1` wrong for reasons that have nothing to do with
    /// the code under test, which is the shape of a flaky test that gets
    /// ignored.
    fn serial() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(|error| error.into_inner())
    }

    /// Compile and build a one-line probe kernel, or fail the test if there is
    /// no Metal device. Failing rather than returning is deliberate: a probe
    /// that silently no-ops turns every test using it into a vacuous green,
    /// which is the failure mode this file exists to prevent.
    fn probe(name: &str, scale: f32) {
        let Some(device) = Device::system_default() else {
            panic!("no Metal device: {name} could not be built, so this test proves nothing");
        };
        let source = format!(
            "kernel void {name}(device float* o [[buffer(0)]], \
             uint i [[thread_position_in_grid]]) {{ o[i] = float(i) * {scale}; }}"
        );
        let library = device
            .new_library_with_source(&source, &metal::CompileOptions::new())
            .expect("probe library");
        build(&device, &library, name).unwrap_or_else(|error| panic!("probe {name}: {error}"));
    }
}
