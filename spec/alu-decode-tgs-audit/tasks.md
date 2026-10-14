# Tasks

## Authorized documentation task

- [x] **D0** `spec.md`, `plan.md`, `tasks.md`, `findings.md`,
  `implementation_notes.md`: audit GitHub tip, interview/confirmation, independent
  CPU extraction experiment with failing mutations, standalone GPU fused2
  comparison with active-shader sabotage, specify ALU and tg/s work.
  Validation: source citations, CPU/GPU results/limits, requirement mapping and
  `git diff --cached --check`. Commit: `61be419`.

## Future implementation/measurement tasks — not performed or authorized here

Each task is separately verified and committed if implementation is later
requested. Update spec/plan before any layout or interface scope change.

- [ ] **T1** External baseline harness; future diagnostic touchpoints
  `linear.rs`, `generate.rs`, `weights.rs`, spec notes: inventory exact live
  chat route and tensor bytes; hash binary/model and measure short/2k/8k/16k
  supported contexts, temperature 0 and fixed-seed sampling. Check GPU status
  before each model run. No throughput result without actual forward counts.
- [ ] **T2** External tests; if necessary `metrics.rs`, `generate.rs`:
  validate independent application timing, early EOS, 1/2/512-token cases,
  callback effects and diagnostic-route differences; prepare active-branch
  GPU/raw-FP32/FP16 gates and red controls. Measure baseline ABBA variability.
- [x] **T3** `linear_repacked.metal`: change `fused2_sk2` from the existing
  vector MCG helper to scalar 32-bit state extraction plus original scalar MCG.
  Preserve split and accumulation order. Compare raw FP32 outputs against the
  64-bit kernel, run `fused_bit_exact`, sabotage the actual kernel, and measure
  the kernel ABBA/noise control. 62,208 raw words equal; sabotage differs;
  fused suite 4/4. Full chat remains later. Commit: `fcf1eab`.
- [x] **T4** `linear_repacked.metal`: applied scalar extraction separately in
  `fused3_sk2`; compared every raw FP32 output against its 64-bit version,
  passed the fused integration suite and a sabotaged-kernel red control. Larger
  synthetic shapes showed +4.197% and +4.767% paired median kernel gains;
  small shape was noisy. No full-chat attribution. Commit: pending.
- [ ] **T5** `linear_repacked.metal` and isolated external harnesses: one
  experiment/commit at a time for shift-class specialization, register lifetime,
  unchanged-order unroll/lookahead and exact load sharing. Verify tail/grid
  boundaries, raw bits, compiler output and paired tg/s; record every negative.
- [ ] **T6** `generate.rs`, `sampler.rs`, existing sampling kernels if reachable:
  profile sampled-chat overhead; one measured allocation/conversion/handoff
  change at a time. Test filtering/ties/RNG/seed parity, EOS, slow callbacks,
  >=512 sampled and greedy tokens, and matched ABBA end-to-end throughput.
- [ ] **T7** `encoder.rs`, `model.rs`, attention/elements kernels or `weights.rs`
  only when measurements justify them: isolate remaining encoding gaps,
  unchanged-order attention scheduling or residency defects. One change per
  commit; test ordering/caches, supported contexts, prefill, memory and ABBA.
- [ ] **T8** Spec notes/findings: consolidate only validated wins, repeat full
  quality/context acceptance, compute application GB/s and measured roofline,
  report feasible tg/s and whether 250 is met. Include exact reproducible
  commands, raw observations, unsupported cases and all unresolved failures.
