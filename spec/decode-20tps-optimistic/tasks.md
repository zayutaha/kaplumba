# TASKS: Decode to 20 tok/s — Optimistic Build

Each task: files, intent, test. Update this file after every completed
task, with the commit hash. Implementation notes in
`implementation_notes.md`.

## Track 0 — Unblock the shared tree

- [ ] **T0.1** Route all 19 direct pipeline constructions in
  `crates/kernels/src/linear.rs` through `pipeline_audit::build` (or
  remove `pipeline_audit`). Test:
  `every_pipeline_construction_goes_through_the_audit` green.
- [ ] **T0.2** Fix `the_decode_instrument_measures_a_step_and_only_a_step`
  under rolling: new exclusion logic in
  `crates/kernels/tests/decode_profile_gpu.rs`, or document the changed
  exclusion window. Attribution numbers stay meaningful.
- [ ] **T0.3** Merge `integration/rolling` onto the active branch
  (command-buffer rolling, +4.37%). Test: suite green, rolling A/B
  reproduced with 16 paired samples.

## Track 1 — Non-GEMV collection (measured levers)

- [ ] **T1.1** Attention split rule: cherry-pick `8852a16`, run the
  default-flip in a dedicated commit, re-baseline the greedy golden
  with user sign-off. Tests: 12 unit + GPU test from the branch;
  `KAPLUMBA_DECODE_SPLIT_RULE=threads` reproduces old hash.
- [ ] **T1.2** Fold `cache_append` into `norm_rope_qkv` (relabel V-store
  address, delete `scratch.k/v` staging). Test: bit-exact attention-KV
  equivalence; step A/B, expect −0.21 ms.
- [ ] **T1.3** Hoist norm weight/residual loads above the barrier in
  1-group norms. Test: bit-exact; step A/B (accept 0 if compiler
  already did it).
- [ ] **T1.4** Device-side argmax: fix `0xFFFF - index` underflow at
  index ≥ 65,536; swap into the sampler path. Tests:
  `argmax_device.rs` non-degenerate rows, degenerate-row guard, one
  262,144-row agreement vs host.
- [ ] **T1.5** Record sampler win (`40e018d`, 2.78 → 1.89 ms) in notes.

## Track 2 — GEMV streaming 200 → 254 GB/s

- [ ] **T2.1** Verify `contig4` (4×96 B lane-group) is the packing
  consumed by **all** 3 bpw long-K GEMVs; if any projection still takes
  a 96 B single-tile path, switch it. Test: `live_kernel_contig4_is_bit_exact_against_colmajor`;
  per-shape GB/s table in notes.
- [ ] **T2.2** Confirm 2col covers `qkv` and `gate_up` fused paths;
  add the `out_features % 32 == 0` guard per half of gate/up. Test:
  `fused_bit_exact`, sabotage re-run of the §23 down_proj arm.
- [ ] **T2.3** Software-pipeline `decode_state_mcg` issue ahead of the
  dependent FMA in `linear_repacked.metal`. Test: bit-exact + A/B;
  expected target is the 30% "neither" slot in the §20 partition.
- [ ] **T2.4** Re-run the §20 ALU/load partition probe after T2.1–T2.3;
  record the new split. If decode-ALU share drops below ~40%, Document
  the remaining headroom.

## Track 3 — Proof

- [ ] **T3.1** Greedy 128-token output byte-identical to baseline
  (re-baselined per T1.1), exact model path and prompt recorded.
- [ ] **T3.2** `/usr/bin/time -l` memory footprint ≤ 16.5 GiB at 8k
  context, 0 swap.
- [ ] **T3.3** 3-run spread, min reported, GPU clock noted.
