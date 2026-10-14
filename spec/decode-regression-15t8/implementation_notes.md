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
