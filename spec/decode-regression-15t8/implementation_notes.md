# Implementation notes

## 2026-10-14 — investigation and implementation start

- User confirmed the historical 15.8 tok/s result was ordinary decode with
  DFlash disabled. The exact original prompt/context was not available; use the
  documented canonical counting prompt and record the limitation.
- Candidate regression: `spec/mlock-clean-mmap` T5g, code commit `692652c`,
  adds `encoder.flush_and_wait_queue()` at
  `crates/engine/src/model.rs:4351` on every sparse pipelined step, before host
  slot writes. Sparse-row gathering in `forward_chunk_pick` also drains the
  queue before overwriting shared `embedding_rows` (`model.rs:4988`). This
  removes the host-encode/GPU-execution overlap previously used by chat and CLI
  generation.
- This is a definite serialization mechanism, but its full contribution to
  15.8 -> 12.25 tok/s is not yet measured. A separate logged 12.439 tok/s run
  used a 39-token prompt and 160 generated tokens before T5g, so it is not a
  matched comparator.
- Historical canonical source:
  `spec/decode-25tok-peak-bandwidth/implementation_notes.md` §16.1 records
  `Count from 1 to 50: 1, 2, 3,`, 32 new tokens, temperature 0, five paired
  samples, 15.849/15.839 tok/s medians at `223ea09` (2026-10-01). That revision
  predates a known unsafe logits hazard fix; use only as historical context,
  not as a correctness baseline.
- Other same-day candidate changes to isolate: T5b `4e36728` (sliding cache
  capacity/growth), T5e `30b7bb3` (queue synchronization at growth), T5f
  `182006e` (page-bounded Metal views), T5g `692652c` (sparse embedding and
  steady-state queue drain). T1/T5g memory notes record no Metal kernel-source
  commit that day. T5e protects queued writes during migration and must not be
  removed casually.
- DFlash equivalence throughput is not an ordinary-pipeline comparator:
  `crates/engine/tests/dflash_equivalence.rs` drives synchronous target forwards
  and loads a drafter.
- Proposed safe ordering: CPU encodes next-step work into unsubmitted,
  per-step-bounded command buffers while the preceding GPU step runs; after the
  preceding output is available, CPU gathers its exact embedding row, then
  submits the prepared buffers. A single row buffer remains safe because steps
  are submitted serially and staging waits for the prior use to finish. EOS
  drops the prepared step without submission.
- Current user-owned dirty edits (preserve):
  `crates/engine/src/exl3.rs` adds `test_msb_window_u32_matches_64bit`;
  `crates/kernels/src/linear_repacked.metal` changes
  `linear_gemv_ishizuki_k3_colmajor_contig4` to hoist weight loads into eight
  arrays. The latter is a live benchmark confound. Do not reset, stage, or
  commit these edits with spec work.
- No model was run for this investigation. Before any GPU run, check
  `scripts/gpuq status` and use `scripts/gpuq try --`; only one model process at
  a time.

## 2026-10-14 — T2 deferred encoder lifecycle

- Added opt-in `MetalEncoder::defer_submissions()`. Rolled buffers are ended
  and retained without commit; `commit_terminal`, wait/flush paths, and finish
  submit them in original order. `abandon` drops all deferred buffers without
  submitting them. Normal encoders retain the existing immediate-roll path.
- Hard cap is 32 deferred rolled buffers per encoder. A decode step has about
  1,617 dispatches and the default 50-dispatch slices, or 32 rolled buffers plus
  its tail. A 64-buffer test reached Metal's uncommitted command-buffer creation
  stall before our error guard; reducing the limit to 32 made the overflow
  diagnostic test complete. If the production measured dispatch count exceeds
  this cap, amend it only with a safe, bounded design.
- Tests: `cargo test -p kaplumba-kernels --lib` passed 36/36 before the explicit
  cap test. Then `cargo test -p kaplumba-kernels deferred_ --lib` passed 3/3
  (deferred submit ordering, abandon, hard bound). A deliberate early-commit
  sabotage failed as expected: `a rolled buffer was submitted before shared
  input staging`, `Committed` vs `NotEnqueued`; sabotage was reverted.
- `scripts/gpuq status` before any model baseline reported 0.1 GiB top-unused,
  6.7 GiB active pages, and 1,948 MiB existing swap. Per §10 this is below the
  safe available-memory threshold; no model run was started. T1's matched model
  comparison remains open.
- Follow-up before T3: an empty `flush_and_wait_queue` is used by the sliding
  KV growth path before any new decode dispatch. Deferred mode now survives
  this barrier; otherwise growth would silently restore immediate commits for
  the rest of that step. `cargo test -p kaplumba-kernels deferred_ --lib` passes
  4/4 including this case.
- T2 readiness guard: a deferred encoder now rejects all submit/flush paths
  while it owns work and the caller has not called
  `mark_deferred_inputs_staged()`. T3's model row staging calls this only after
  the raw mmap row copy succeeds. Empty cache-growth barriers remain legal;
  they preserve deferred mode and still do not submit model dispatches.

## 2026-10-14 — T3 sparse row preparation split

- `forward_step_pipelined` no longer drains the queue for the unknown-token
  sparse steady-state step. It encodes the forward against `embedding_rows` in
  deferred-submission mode, records the active slot as pending, and returns.
  Known-token first steps and diagnostic modes that flush within the forward
  retain the synchronous stage path. Prefill and non-pipelined calls retain the
  existing gather behavior.
- Added `stage_pipelined_embedding_row(encoder, slot)`: after the caller waits
  for the prior prediction, it gathers raw mmap bytes for the active decode
  slot and marks encoder shared inputs ready. Added explicit discard for EOS and
  pending state validation. Resetting KV cache clears pending row state.
- Retained the single bounded row buffer. Because the prepared encoder cannot
  submit before staging, the prior GPU read finishes before the host overwrites
  the buffer. `encoder.defer_submissions()` is not disabled by the empty queue
  wait on the sliding-KV growth path.
- `cargo check -p kaplumba-engine` passed. `cargo test -p kaplumba-engine --lib`
  passed 145, ignored 2 (requires `KAPLUMBA_TEST_MODEL`).
- Negative control: changed gather selection from `id` to `id ^ 1`; the
  BF16/FP16 kernel comparison failed at
  `model::sparse_embedding_tests::gathered_embedding_kernel_is_bit_identical_for_bf16_and_fp16`
  with `bf16=false`; restored the correct index.

## 2026-10-14 — T4 chat/CLI integration and T5 lifecycle checks

- Both generation loops now keep sparse-row dependent command buffers deferred
  until after the prior prediction is waited and the exact mmap row has been
  staged. EOS drops that unsubmitted work and retracts the speculative KV
  position. Error callbacks discard pending row state or retire already-submitted
  diagnostic work before retracting cache state.
- Chat and CLI static ordering tests verify encode-next → wait/read-prior →
  stage-row → submit-next. The multi-turn chat suite passed 7 tests; 4
  model-backed tests were ignored because no model run was safe/available.
- `cargo test -p kaplumba-engine --lib`: 145 passed, 2 ignored.
  `cargo test -p kaplumba-kernels --lib`: 38 passed. `git diff --check` clean.
- Model runs remain blocked by measured host memory state, not GPU lock
  contention: latest `scripts/gpuq status` reported 0.9 GiB top-unused, 8.2 GiB
  system active pages, and 1,908 MiB swap. Per AGENTS.md §10, no model was
  started. The requested >=15.8 tok/s and `--chat` verification are not yet
  demonstrated; T1/T6 remain open.
