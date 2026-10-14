# ALU decode and token-generation throughput audit

## Goal

Improve pure single-user `Gxma-repacked --chat` token-generation throughput
(`tg/s`) without changing model quality. Investigate the current GitHub branch
`stepwise-clean-3724c01`, pinned to
`7ddf20f901ae7ad4d1e7342ca71677e266bbe847` (2026-10-14), and develop measured,
independently reversible optimizations. Investigate 250 GB/s of **useful weight
bytes divided by end-to-end decode wall time** as an unproven stretch target;
an evidenced feasibility limit is an acceptable research result.

## Authorization and restrictions

- The initial audit was documentation-only. The user subsequently authorized
  implementation and testing. That authorization has been used for the narrow
  scalar 32-bit extraction experiment in the active `fused2_sk2` and
  `fused3_sk2` QKV kernels, plus an isolated raw-FP32 Metal comparison harness.
  Other kernel families, sampler, dispatch, attention, layout, and residency
  work remain unimplemented.
- The two QKV variants are experimental until full Gxma chat token-identity and
  end-to-end `tg/s` acceptance is measured. Synthetic kernel wins alone do not
  authorize declaring the full spec complete or the 250 GB/s target met.
- The user confirmed GitHub tip, Gxma-repacked chat, broad quality-preserving
  investigation, the end-to-end bandwidth definition, and documentation writes.
  They subsequently explicitly added finding ways to increase `tg/s`.
- Evaluate short chat, approximately 2k and 8k contexts, and 16k where the pinned
  engine/checkpoint supports it. Count tokenized chat context, including template
  tokens; reserve room for generation. Unsupported contexts are recorded, never
  obtained by silently changing context or positional rules.
- Keep EXL3 states/codebooks, signs, permutations, FP16 rounding boundaries,
  accumulator ownership and reduction order intact. Integer-only changes must
  reproduce raw FP32 GEMV bits as well as finalized FP16 bits.
- Preserve greedy and fixed-seed sampled token IDs. Preserve the original
  sampler distribution, filtering, tie handling and random-number consumption.
  No approximate sampler substitutions.
- No lossy repacking, requantization, KV quantization, lower-precision sums,
  dropped layers, speculative decoding, generated-text reuse, or multi-user
  batching. Do not port AGPL code without separately resolving licensing.
- Preserve bounded, explicitly owned scratch and existing prefill behavior;
  report TTFT, prefill and physical memory footprint alongside decode results.
- Follow AGENTS.md GPU queue rules. Models remain outside the repository.
  Dedicated worktrees have separate build directories. No concurrent GPU
  benchmark processes, including model-free probes that lack the engine lock.

## Requirements

1. **R1 Provenance:** pin source/checkpoint/configuration and inventory actual
   dispatches, dimensions, rates, permutations and defaults before choosing
   kernels. Distinguish local checkout from GitHub tip.
2. **R2 Metrics:** measure application decode wall time independently of internal
   metrics. Count executed forward steps, exclude prefill's first generated
   token from decode numerator, and include encoding, sampling, synchronization,
   detokenization and normal output handling in the application interval.
   Report isolated GPU time separately. Reconcile useful bytes/token from
   actual tensor headers/dispatches, not checkpoint file size or lane loads.
3. **R3 Correctness:** test each changed active branch against the pinned baseline
   and an independent state reader. Deliberately sabotage that branch and see
   a diagnosable failure before making a test-coverage claim. Require raw FP32
   equality for integer-only changes, finalized FP16 equality, no new NaN/Inf,
   and identical greedy/fixed-seed sampled token IDs in >=512-token generations
   when context permits. Retain existing failures as separately reproduced facts.
4. **R4 ALU experiments:** test the still-64-bit active QKV split-K families
   first; then assess shift specialization, live-register scheduling, address
   arithmetic and exact load sharing on the kernels actually executing.
   Already-shipped work is a baseline, never a new optimization claim.
5. **R5 tg/s beyond ALU:** investigate sampled-chat CPU/readback costs, CPU/GPU
   handoff, command encoding gaps, long-context attention, and page residency.
   Keep causal attribution and all synchronization/rounding invariants.
6. **R6 Measurement:** use a single rebuilt binary with explicit A/B selection
   where possible, warmed interleaved ABBA blocks with at least 16 paired
   comparisons. Report medians, spread, paired deltas and contaminated samples;
   the acceptance interval for the paired improvement must exclude zero.
   Compare uninstrumented application throughput separately because profiling
   changes the greedy execution route.
7. **R7 Feasibility:** report `W_GB * tg/s` as effective application GB/s,
   `W_GB / GEMV_seconds` as isolated useful-weight GB/s, and physical DRAM
   bandwidth only when an appropriate instrument supports it. Quantify the
   roofline with measured residual critical-path costs and cache assumptions.

## Acceptance criteria

- **AC1 (audit and narrow trial):** findings distinguish source-verified facts,
  newly executed standalone CPU/GPU experiments, historical measurements and
  future hypotheses; include reproducible checks and failing negative controls.
- **AC1a (T3/T4 kernel trial):** scalar state extraction matches the original
  active QKV split-K kernels at raw FP32 output bits on the recorded synthetic
  shapes; the real fused-kernel integration suite passes; sabotaging each tested
  candidate changes outputs. This gate is met for the tested shapes only.
- **AC2 (future experiments):** dispatch inventory, exact commands, executable
  and checkpoint identity, token IDs, raw ABBA measurements and quality gates
  accompany every claimed improvement. No silent shader/test skip or stale build.
- **AC3 (future retained optimization):** positive paired application `tg/s`
  improvement on the named workload, all correctness gates passed, no material
  unexplained TTFT/prefill/memory regression, and results across all supported
  requested contexts. A short-context win with a long-context loss remains an
  experimental variant rather than a global default.
- **AC4 (stretch):** >=250 effective end-to-end GB/s on a specified context is
  proven only by audited W and application timing. Otherwise document the
  measured gap and feasible `tg/s`; do not redefine the metric to call it met.

## Open questions

No user-scope questions remain. Still unresolved as measured acceptance work:
fresh end-to-end baseline throughput, live dispatch counts, audited useful
bytes/token, sampled-chat cost, full raw/logit/token identity for the checkpoint,
short/2k/8k/16k behavior, and the attainable application bandwidth ceiling.
The fused2/fused3 evidence is a synthetic kernel comparison; no full-model
performance or complete model-quality gate has run. GPU status at the end of this
implementation showed under 1 GiB top-unused and about 3.3 GiB swap, below the
repository's 14 GiB model-run threshold, so model acceptance remains pending.
