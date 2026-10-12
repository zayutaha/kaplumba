# spec/dflash-30tps-v2 — tasks

-- [x] **T1 — spec + plan + tasks + notes.** All four files exist with the
  AGENTS.md §3 sections. Test: files present.
-- [x] **T2 — colmajor `h2` body.** Done (`extract_v_mcg_32_h2`,
  `linear_gemv_mrhs_ishizuki_k3_colmajor_h2_impl`, `_h2_m{8,16,32}`,
  `_h2w_m32`).
-- [x] **T3 — rowmajor `h2` body.** Done (`exl3_decode8_mcg_h2`,
  `linear_gemv_mrhs_exl3_h2_impl`, `{k3,k4}_mcg_h2_m{8,16,32}`).
-- [x] **T4 — host wiring.** Done — `MrhsPath::H2`, fields, build loop, pickers,
  route arm, `KAPLUMBA_MRHS_PATH=h2`. The arm dispatches 8-row **blocks**
  (mirroring v1) because the wide single-pass bodies lose to v1; `mrhs_h2_m8_pipeline`
  documents the measurement that fixed that.
-- [x] **T5 — parity + sabotage.** Done, folded into
  `crates/kernels/tests/linear_mrhs_v2.rs` (`mrhs_h2_matches_v1_{row,column}_major`,
  `H2_K_DEPTHS = [256, 1024]`) rather than a new file, because the suite already
  owns the harness that compares a v2 family against v1. Three sabotage cases
  recorded in `implementation_notes.md`; one of them (the unrolled-body swap)
  revealed that `in_features = 256` never reaches the unrolled body.
-- [x] **T6 — K3 bench.** Done, `Arm::H2` / `Arm::H2Wide` added.
  **AC1 met, AC2 NOT met** — see the verdict in `implementation_notes.md`.
-- [x] **T7 — census delta.** Done — 180 → 190 kernels, 78 → 84 `_exl3_`
  row-major, repacked 44 → 48, all four census tests green.

### Original task text (for the record)

-- [x] **T2 — colmajor `h2` body.** `extract_v_mcg_32_h2` +
  `linear_gemv_mrhs_ishizuki_k3_colmajor_contig4_h2_impl<M_CAP, NARROW_TILES>` +
  `..._h2_m{8,16,32}` in `linear_repacked.metal`.
  Test: `cargo test -p kaplumba-kernels --release` builds; T4 parity.
-- [ ] **T3 — rowmajor `h2` body.** `exl3_decode8_mcg_h2` +
  `linear_gemv_mrhs_exl3_v2_h2_impl<M_CAP, NARROW_TILES>` + `..._{k3,k4}_mcg_h2_m{8,16,32}`
  in `linear.metal`.
  Test: as T2.
-- [ ] **T4 — host wiring.** `MrhsPath::H2`, pipeline fields, build loop,
  pickers, route arm; `KAPLUMBA_MRHS_PATH=h2`.
  Test: `cargo test -p kaplumba-kernels --release` all green.
-- [ ] **T5 — parity + sabotage.** `crates/kernels/tests/linear_mrhs_h2.rs`:
  every new kernel × {k3 cm, k3 rm, k4 rm} × M ∈ {1,2,4,7,14,16,17,24,32} vs an
  M=1 run of the same kernel, tolerance set from the measured max abs diff;
  row-overwrite canaries; **sabotage log** (break the packed fma, see the failure
  message, revert, record).
  Test: the suite.
-- [ ] **T6 — K3 bench.** `mrhs_v2_bench` gains `Arm::H2`; table into
  `implementation_notes.md` (AC1, AC2).
  Test: the printed table.
-- [ ] **T7 — census delta.** `probe_kernsel_census.rs` refreshed with a named
  delta. Test: probe green.
-- [ ] **T8 — R1 two-block draft.** `DflashModel::draft_two_blocks` + the
  round's use of it. Test: unit test on the concatenation/retract arithmetic.
-- [ ] **T9 — R2 batched head.** `encode_lm_head_all`; measured against 32
  sequential `encode_gemv` on real logits. Test: the comparison.
-- [ ] **T10 — R3 `generate.rs`.** `--dflash` branch. Test: a run emits ≥ 1
  token.
-- [ ] **T11 — R4 CLI + metrics.** `--dflash`, `--draft-block`,
  `dflash_tok_s`/`dflash_acceptance`. Test: `--help`, then a run.
## Blocker (measured 2026-10-12, before T8-T14 were started)

`dflash_alpha_probe` against the real checkpoints: **alpha = 0.0000**, and two
different tap-layer sets produce byte-identical drafts, so the drafter does not
condition on its context at all. At alpha = 0 a round emits one token per
606 ms verify + 157 ms draft ≈ **1.3 tok/s**, i.e. 12x slower than the 15.28
tok/s plain decode. **AC3/AC4/AC5 are unreachable until the context path works,
and no kernel change moves them.** The full evidence, the two causes eliminated
by reading, and the one test that separates the rest are in
`implementation_notes.md`.

-- [ ] **T12 — R5 equivalence.** greedy + sampled. Test: the suite (AC3).
-- [ ] **T13 — R5 acceptance.** `examples/dflash_acceptance.rs`. Test: AC4.
-- [ ] **T14 — AC5/AC6/AC7.** end-to-end tok/s, M=1 and M≤16 regression,
  footprint. Tests: the numbers.

## Status

| task | state | commit |
|---|---|---|
| T1 | done | (this write) |
| T2 | done | (see T2 note) |
| T3 | done | (see T3 note) |
| T4 | done | (see T4 note) |
| T5 | done | (see T5 note) |
| T6 | done | **AC1 met, AC2 not met** |
| T7 | done | census 190 / 84 / 48 |
| T8  | **blocked** | two-block draft — blocked on the context path: with alpha = 0 a round emits 1 token, so more drafts per round cannot help |
| T9  | open | batched head |
| T10 | open | `generate.rs` `--dflash` |
| T11 | open | CLI + metrics |
| T12 | open | equivalence |
| T13 | open | acceptance |
| T14 | open | AC5/AC6/AC7 |

## Follow-up tasks found by the L1 investigation (2026-10-12, session 2)

-- [x] **T15 — the drafter's RMSNorm offset.** The loader added Gemma's `+1.0`
  to a **Qwen3** drafter's norm weights (4.2x on `post_attention_layernorm`).
  Files: `crates/engine/src/dflash.rs` (`DflashWeights::load`), plus the two tests
  that encoded the old convention (`dflash_load`, `dflash_golden::norm_weights`).
  Test: `dflash_load::dflash_norm_weights_are_stored_verbatim` (bit-exact against
  the shard, naming 0.7539 and the 0.311 mean); all 6 `dflash_golden` tests.
  Measured: layer-0 MLP peak 152 287 -> 10 422, block RMS 4.139 -> 3.183.
-- [x] **T16 — the golden reference's full-layer mask.** `host_block` gave the
  `full_attention` layer a 2048 window, so its u32 `q_pos - k_pos` wrapped and the
  reference silently became causal on the one layer that is not. File:
  `crates/engine/tests/dflash_golden.rs`. Test: the with-context test, whose worst
  diff went 4.973 (11.5%) -> 0.0232 (0.06%) once the reference was corrected.
-- [x] **T17 — the context must be *visible*, not just agreed upon.**
  `dflash_golden::dflash_context_rows_are_visible_in_the_block_output`: 33.62 for a
  changed context value and 33.19 for no context, on a host RMS of 4.11, asserted on
  both engine and reference. Also `host_block`/`block_case` extraction and an
  assertion (3e-2) on the with-context comparison, which previously only printed.
-- [ ] **T18 — the taps, the only input never validated externally.** The ctx
  *path* is proven (T17), but nothing proves the six tap tensors are the target's
  post-layer residuals at layers [1,12,23,35,46,57] rather than something that
  merely differs per layer. Test: compare a tap buffer against `Model::forward_layer`
  output for the same layer and input, element-wise, for all six.

## Status (addendum)

| task | state | note |
|---|---|---|
| T15 | **done** | 4.2x norm error removed; acceptance unaffected |
| T16 | **done** | 11.5% -> 0.06%; the compare now asserts |
| T17 | **done** | context moves the block by 8x its own RMS, engine and host |
| T18 | open | the next action with leverage |
