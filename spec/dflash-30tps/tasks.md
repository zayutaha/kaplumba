# spec/dflash-30tps — tasks

Ordered, one small verifiable step each. Update the status + commit hash after
every task. **Target amended to 40 tok/s** (user directive 2026-10-12), coherent
at temp 1.0 / top_k 64, distribution-exact.

-- [x] **T1 — spec + plan + tasks.** Create `spec.md` (exists), `plan.md`,
  `tasks.md`, open `implementation_notes.md` with the measured premises.
  Test: files exist with all AGENTS.md §3 sections.
-- [x] **T2 — Path A fp16 MRHS, row-major (`linear.metal`).**
  `exl3_decode8_mcg<ALIGNED, FP16>` + `linear_gemv_mrhs_exl3_{k3,k4}_mcg_fp16_m{4,8,16}`
  with a `KAPLUMBA_MRHS_NARROW` partial-sum width. Register bodies only.
  Test: `cargo test -p kaplumba-kernels --test linear_mrhs_v2` parity vs M=1.
-- [x] **T3 — Path A fp16 MRHS, colmajor (`linear_repacked.metal`).**
  `extract_v_mcg_32_fp16` + `linear_gemv_mrhs_ishizuki_k3_colmajor_contig4_fp16_m{4,8,16}`.
  Test: same parity test, `perm_mode=1`.
-- [x] **T4 — Path B MMA MRHS, both layouts and both rates.**
  `linear_gemv_mrhs_mma_exl3_k3_mcg_r{1,2,4}` (colmajor) and
  `..._k4_mcg_r{1,2,4}` (row-major): 256 threads, `bstage[2][64][COLS+8]`,
  3-word ring prefetch, 8 simdgroups × (8 rows × 8 cols), lane→(k,n) scatter.
  Test: parity + sabotage.
-- [x] **T5 — host wiring.** Pipeline fields, `pipeline_audit::build` calls, struct
  literal, pickers `mrhs_fp16_*` / `mrhs_mma_*`.
  Test: `cargo test -p kaplumba-kernels` (all existing suites green).
-- [x] **T6 — routing.** `KAPLUMBA_MRHS_PATH` (`v1`|`fp16`|`mma`|`auto`) and
  `KAPLUMBA_MRHS_V2`, memoised, priority-ordered under `KAPLUMBA_DISABLE_MRHS`.
  Test: `model.rs`'s `linear_route_tests` + a new routing unit test.
-- [x] **T7 — parity + sabotage tests.** `crates/kernels/tests/linear_mrhs_v2.rs`:
  every new kernel × {k3 cm, k3 rm, k4 rm} × M ∈ {1,2,4,7,14,16,17,24,32} vs an
  M=1 run of the same kernel, tolerance 1e-2 for fp16/MMA; plus row-overwrite
  canaries past row M-1; plus **sabotage evidence** per AGENTS.md §3.4 (break the
  kernel, see the test fail with a diagnosable message, revert, record).
  Test: the suite itself, and the recorded sabotage log.
-- [x] **T8 — census + permutation audits.** Refresh the pinned kernel count in
  `probe_kernsel_census.rs` with a named delta; re-audit
  `probe_perm_permutation_modes.rs` if a new forward site appears.
  Test: both probes green.
-- [x] **T9 — K5 benchmark.** `crates/engine/tests/mrhs_v2_bench.rs`: target
  (3 bpw colmajor 5376×21504) and drafter (4 bpw rowmajor 5376×10752) at
  M=1/4/8/16/32, interleaved A/B/A/B, median of 9, ms + GB/s for (a) current
  MRHS (b) fp16 (c) MMA. **Ship the M=32 winner.** Record the table in
  `implementation_notes.md` (AC1).
- [ ] **T10 — H2 root cause.** `forward_verify` M=2 vs two M=1 forwards *with a
  cache*, comparing the row-1 hidden state before the head. Record whether the
  19.2 divergence is layers or head.
  Test: a new ignored real-model test with the number printed.
- [ ] **T11 — H1 batched head.** Per T10's verdict: 8-row batches through
  `encode_gemm`, or extend `mrhs_eligible` to `Integer(6)`. Bounded scratch
  (8 × 262144 × 4 B = 8.4 MB, independent of block size).
  Test: batched-vs-sequential within fp16 tolerance on real logits (H2 gate).
- [x] **T12 — L1 re-instrument the alpha probe.** Done, and it **refutes the
  "dead context" conclusion**: at the `draft_logits` level, two different
  tap-layer sets differ in **4 172 920 of 4 194 304 logits** (worst |delta|
  16.28 on a 30 softcap), so the drafter is strongly context-dependent and the
  identical *drafts* were the softcap's tie-break, not a broken context path.
  The assertion is now on the logits (green). Root cause still open: the target's
  0.998 token ranks 198 213rd, and the drafter's permutation cannot be scored
  until the buffer path (`LinearBufferRequest`) learns `perm_mode`.
  Test: `dflash_alpha_probe` green, 25.7 s, `Gxma-repacked` + `Drafter`.
-- [x] **T13 — L2 round loop.** `DflashModel::draft_and_verify` with the corrected
  non-anchor-refeeding shape, speculative-sampling verification, `L[0] =
  previous_logits`, retract `n - k`.
  Test: unit tests on the accept/retract arithmetic with a synthetic drafter.
- [ ] **T14 — target-side call sites.** `generate.rs` branch, `forward_verify`
  feeding, cache retract plumbing.
  Test: an end-to-end run produces ≥ 1 token with `--dflash`.
- [ ] **T15 — L3 equivalence.** `crates/engine/tests/dflash_equivalence.rs`:
  ≥ 5 prompts × ≥ 200 tokens at the shipped block size, bit-identical to
  non-speculative greedy (near-ties < 0.25 exempt), plus a distribution-exact
  sample-mode equivalence check.
  Test: the suite (AC3).
- [ ] **T16 — L4 acceptance.** `crates/engine/examples/dflash_acceptance.rs`:
  ≥ 200 rounds, median + spread + per-depth histogram, both greedy and sampled
  α, and the tok/s implied by the measured round time.
  Test: the number printed (AC4).
- [ ] **T17 — L5 CLI.** `--dflash`, `--draft-block`, `dflash_acceptance` and
  `dflash_tok_s` in `--metrics`.
  Test: `--help` shows them; a run prints them.
- [ ] **T18 — M1 memory.** `/usr/bin/time -l` peak footprint at 1k/2k context
  with the drafter resident, under 20 GiB.
  Test: the number (AC7).
- [ ] **T19 — AC6 regression.** M=1 decode ≥ 15.0 tok/s and M≤16 MRHS TTFT
  ≤ 350 ms at 14 tokens, measured interleaved against the pre-change binary.
- [ ] **T20 — AC5 end-to-end.** ≥ 3 prompts × ≥ 200 tokens, median tok/s, the
  accepted/rejected per-round histogram, and an honest statement of the binding
  constraint if the number is below 40.

## Status (2026-10-12, after 45aeeb6)

| task | state | commit |
|---|---|---|
| T1  | done | (spec/plan/tasks/notes, first write) |
| T2  | done | e6df1c1 |
| T3  | done | e6df1c1 |
| T4  | done | e6df1c1 |
| T5  | done | e6df1c1 |
| T6  | done | e6df1c1 |
| T7  | done | e6df1c1 |
| T8  | done | e6df1c1 — census was **already red on HEAD** (146 pinned vs 150 on disk, 59 vs 62 `_exl3_`); all three counters now true numbers with the drift named |
| T9  | done | c24198f — **AC2 not met**: Path B 17.4 GB/s, Path A 9-13 GB/s at M=32. The dequant, not the FMA, is the binding resource |
| T10 | open | the M=2 head divergence, unresolved |
| T11 | open | blocked on T10's verdict |
| T12 | open | α = 0.0000 in `dflash_alpha_probe` is a probe-wiring bug |
| T13 | done, corrected | 45aeeb6 — `draft_and_verify` + `sampler_distribution`/`sample_sorted`/`sample_residual` and 8 unit tests. **Correction:** the round passed its `position` to both `draft_block` (which takes the *anchor's* position) and its own verify (which fed `draft.tokens[k]` at `position + k`) — two conventions, one variable, so every draft was verified against the row for the position *before* the one it proposes. Silent: alpha reads exactly 0.0000 and the output stays correct. Now `anchor_position` + `draft_verify_positions`, with a sabotaged-and-verified unit test |
| T14–T20 | open | see `spec/<next>` |
