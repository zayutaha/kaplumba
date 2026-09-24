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
- **Status**: PENDING
- **Files**: `crates/engine/src/metal_linear.rs` (`P1_BISECT_MSL` + bench extension).
- **Intent**: Add the four rung kernels and extend `bench_stream_bw_slow` (or add a sibling
  `bench_gemv_bisect_slow`) to sweep stage x nsplit x shape. `S1` reuses the existing
  `stream_sum_strided`. Each rung recomputes `ig0_`/`ig1_`/`s3_` with the production
  arithmetic (k=3, mode=0 literals) and the inlined decode matches
  `exl3_decode_f_const` bit-for-bit. Rungs S2–S5 accumulate into an `atomic_uint` sink and
  each is asserted non-zero, so a dead rung cannot read as a fast one.
- **Test**: the bench compiles and runs; S5's time matches the separately-measured
  `exl3_gemv_spec` time for the same shape/nsplit within noise (R5). Both suites green.
- **Commit**: after green.

## T4 — Run, record, report
- **Status**: PENDING
- **Files**: `spec/exl3-load-latency/implementation_notes.md`.
- **Intent**: Run the ladder at real per-linear sizes (o 16.5 MB, down/gate 43.4 MB) and a
  large synthetic, cold and warm separately. Record the per-stage GB/s table, the
  marginal delta of each stage, and an explicit statement of the dominant cost plus what
  the follow-up spec should attack. Then **stop and report to the user** — do not begin
  optimizing.
- **Test**: numbers recorded; R5 outcome stated explicitly (pass or harness-defect report).
- **Commit**: after recording.
