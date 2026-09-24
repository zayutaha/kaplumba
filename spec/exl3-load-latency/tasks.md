# Tasks: exl3-load-latency

One task per commit (AGENTS §4). `tasks.md` updated immediately after each task with its
commit hash. This spec ends at **T4 — report**; the optimization it enables is a separate
spec, chosen from T4's data.

## T0 — Revert the uncommitted T4 work
- **Status**: DONE (no commit — restores the tree to `4b3708f` content).
- **Files**: `crates/engine/src/metal_linear.rs`, `crates/kernels/src/exl3.metal`,
  `crates/kernels/src/exl3.rs`.
- **Intent**: Discard the fused RMSNorm+Hadamard work (T4 of `spec/bandwidth-saturate`).
  It targets Hadamard dispatch overhead, which the control experiment identifies as
  non-dominant. The work was never wired into `gpu_decode.rs` and was never benchmarked,
  so nothing in production depended on it.
- **Test**: `git diff --stat` empty; both suites still at their `4b3708f` pass counts
  (kernels 21/0, engine 132/0/12 — re-measured, not assumed).

## T1 — Record the control result; kill the two dead premises
- **Status**: DONE — commit `b98e4f5`.
- **Files**: `spec/bandwidth-saturate/implementation_notes.md`,
  `spec/bandwidth-saturate/tasks.md`, `spec/trellis-dense-repack/tasks.md`.
- **Intent**: Append the `bench_stream_bw_slow` table to the bandwidth-saturate notes and
  retract the two falsified claims: (a) "the 21 KB tile stride caps the bus at ~68 GB/s"
  and (b) "the per-symbol decode ALU is free" (that probe was not a valid control). Mark
  `bandwidth-saturate` T4/T6 deferred-with-reason. In the dense-repack tasks, record that
  RT4's A/B is **resolved by measurement**: W1 and W4 both lost to the native word layout on
  gate/down/o/q, so the layout hypothesis is dead and RT2c/RT2d are cancelled.
- **Test**: none (docs). Verify the numbers quoted match the recorded run exactly.

## T2 — Delete the dead dense-repack kernels
- **Status**: DONE — pending hash in the follow-up commit.
- **Files**: `crates/kernels/src/exl3.metal`.
- **Intent**: Remove `exl3_gemv_dense1` and `exl3_gemv_dense4`. Their host plumbing and
  gates were already removed in `4b3708f`; only the MSL remained, unreferenced. The
  native `[it][ot]` layout is confirmed at least as fast as a linear read, so there is no
  layout problem for them to solve.
- **Delivered**: pure deletion of lines 2991–3450 (461 lines, the RT2 banner + both
  kernels to EOF). `git diff` shows 0 added lines. `grep -rn exl3_gemv_dense crates/`
  returns nothing.
- **Test**: whole MSL still compiles — kernels **21 passed / 0 failed**, engine
  **132 passed / 0 failed / 12 ignored**, identical to the `4b3708f` baseline. (Any engine
  GPU test constructing `MetalLinear::new` compiles the entire MSL, so a syntax error
  anywhere would have failed the suite.)
- **Commit**: after green.

## T3 — Bisection ladder S2–S5
- **Status**: DONE — pending hash in the follow-up commit.
- **Files**: `crates/engine/src/metal_linear.rs` (`P1_BISECT_MSL`, `bench_gemv_bisect_slow`).
- **Delivered**: `P1_BISECT_MSL` (bench-only runtime-compiled MSL) with `bis_gather` (S2),
  `bis_extract` (S3), `bis_decode` (S4), `bis_full` (S5) and `bis_flush` (cache eviction);
  `bench_gemv_bisect_slow` sweeps stage x nsplit{1,2,4,8} x 6 shapes, warm (min-of-12) and
  cold (flushed), asserts each rung's atomic sink is non-zero, and times
  `gemv_pass_spec` alongside as the intended R5 reference. k=3/mode=0 baked as literals so
  the codebook branch folds exactly as FC_MODE=0 does.
- **Test**: bench compiles and runs (1.74 s, all 120 timed runs, every rung non-zero);
  kernels **21 passed / 0 failed**, engine **132 passed / 0 failed / 12 ignored** unchanged.

## T4 — Run, record, report
- **Status**: DONE — see `implementation_notes.md` for the full table and conclusions.
- **R5 outcome: FAILED as specified, and the reference is at fault.** S5 measures ~2x
  *faster* than `gemv_pass_spec` rather than matching it. Cause: `gemv_pass` wraps each call
  in `with_scratch`, which allocates four Metal buffers per call, plus a fifth partial
  buffer in `encode_gemv_spec` — all inside the timed region. The PROD column measures host
  buffer churn, not the kernel. The ladder remains internally valid (S1..S5 differ only by
  the stage under test), so the deltas stand; R5 was aimed at the wrong target and is
  reported as such rather than papered over.
- **Headline result**: the ladder is flat. The complete EXL3 decode (4 scattered loads,
  8 `ulong` extractions, 8 codebook decodes, 8 shuffles, 8 FMAs) costs 0–28% over a bare
  one-word-per-lane walk, and is within noise or faster on the small shapes. **There is no
  3x in the inner loop; do not rewrite it.** The limit on the small shapes is parallelism:
  throughput tracks `in_tiles` (serial walk length), and nsplit has not saturated at n8.
- **Test**: numbers recorded; R5 stated explicitly as failed-with-cause per spec R5.
