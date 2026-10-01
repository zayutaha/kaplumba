# Finding 03 — Fused QKV / gate-up is SLOWER than separate dispatches

Subagent: `ses_f0ceeff86ffeqFeqUmo4X8kIOp`. Commit `e84628b` on branch
`fusion-ab-e84628b` (parked on a branch, not merged — master's HEAD moved twice
under it).

**This is the only result so far that names code to DELETE.**

## The measurement

`ratio = separate/fused` block time, so **< 1.0 means separate wins**. Three
independent runs, ABBA-interleaved, file-backed weights, medians over 24 warm /
28 cold paired samples:

| block | shape | x layers | warm ratio | cold ratio |
|---|---|---|---|---|
| `qkv-sliding` | 5376x[8192,4096,4096] | 50 | 0.920 / 0.912 / 0.913 | 0.961 / 0.968 / 0.961 |
| `qk-full` | 5376x[16384,2048] | 10 | 0.962 / 0.952 / 0.964 | 0.991 / 0.919 / 1.030 |
| `gate-up` | 5376x[21504,21504] | 60 | 0.976 / 0.984 / 0.988 | 1.015 / 0.986 / 0.992 |

Warm per-token: **41.5 ms fused -> 39.9 ms separate, −1.6 ms/token ≈ +3.9% tok/s.**
Penalty breakdown: qkv 0.80, gate/up 0.71, qk 0.08.

`gate-up` lands within 0.5% of omlx's independent −2.3% (`laguna-mlxfast-port-
correctness.md:26`), on different hardware. `qkv-sliding` is the clearest loss
at ~8%.

## Why "it's just dispatch overhead" does not rescue it

The obvious defence was checked and fails:

- **Threadgroup totals are identical by construction** — fused launches
  `tiles_a+tiles_b+tiles_c`; separate launches `tiles_a`, `tiles_b`, `tiles_c`.
  Same 256-thread groups, same `SG=8`.
- **20 cores are saturated many waves over either way.** `gate-up` is 33.6 waves
  fused, 16.8 for its *largest single* separate dispatch. No idle GPU for a
  launch to hide in.
- The `gemv_max_groups()` cap that *would* have confounded this (separate looks
  capped, fused isn't) defaults to `u64::MAX` — measured unset, as printed in the
  tool header.

Same work, same parallelism, fused lands behind. Remaining structural difference
in the source: fused branches per-threadgroup across 2–3 `packed`/`y` pointer
pairs on a kernel already register-heavy in its inner loop. Not diagnosed further
— noted as next step, not needed for the decision.

## Two methodology notes that changed the answer

Both produced *confident wrong results* first:

- **The first block in a process reads its own dirty scratch pages.**
  `qkv-sliding` initially measured 60.7 GB/s while the larger `gate-up` measured
  174.2 GB/s from identical code, paired ratio drifting 2.19 -> 0.67. Needed
  `fsync` + a real 5-round warmup + calibration from the slower of two settled
  readings. **Anyone adding a block to this tool should expect the first one to
  be the suspicious one.**
- **ABBA, not AB.** A fixed order charges whichever arm runs first every round;
  at a 1–8 % effect size that invents results. The effect survived the swap,
  which is what makes it credible.

A bit-equality gate runs before timing and panics on mismatch (16384/18432/43008
values, all finite, all identical) — a speed number for different work is not a
speed number.

## Scope discipline

No production dispatch changed. `model.rs:1420/1481/1849` still dispatch the
fused kernels; no env flag added, since that would mean touching the dispatch
path. Recorded in `spec.md` §5 (withdrawn claims) with a note on R5's kernel-count
clause, plus a full section in `implementation_notes.md`. **De-shipping them is
the owner's call.**

The spec row added says R5's second clause ("minimise the kernel count") is
satisfied by the *status quo* rather than by fusion — the fastest configuration
found is the one with **more** dispatches.

AC6 holds — generator run twice, `crates/` diff clean. Worktree and target dir
removed; master's tree never touched.

## Caveats stated by the agent

- This is **block-level GPU time, not a decode run**. The −1.6 ms/token is
  arithmetic over layer counts (50/10/60), excluding attention/norms/`down`/
  `lm_head`. Confirming end-to-end needs AC5's paired-sample discipline.
- The claim withdrawn is the fusion *rationale*, not the kernels' correctness:
  they are bit-exact, just not faster.
