# Technical plan

## Architecture and data flow

Ordinary decode uses a two-step pipeline. The completed step's GPU argmax writes
the next token ID into the next decode slot. With the dense embedding table, the
following GPU step can consume that ID directly. T5g avoids registering the
large embedding table with Metal: it copies the selected raw BF16/FP16 row from
the shared mmap into a small host-visible row buffer before the GPU embedding
kernel runs. T5g originally drained the queue before this copy, including at
`forward_step_pipelined` entry, which serialized the host encoder and GPU
execution. T2 through T7 replaced that unsafe/serialized lifecycle with the
bounded deferred path below.

The implemented design separates encoding from submission for sparse pipelined
steps:

1. Begin a sparse decode step with bounded deferred submission enabled. Encode
   its command buffers normally, preserving the existing dispatch limit and
   roll boundaries, but retain rolled buffers without committing them.
2. The previous step remains submitted and runs while the CPU encodes this
   step. No CPU read of its predicted token or write to the shared row buffer is
   done yet.
3. Wait for the previous step at the caller's existing token decision point.
   If its token is EOS, drop the prepared buffers without submission and retain
   the existing cache-retract behavior.
4. Otherwise gather the exact row using the GPU-produced token ID, rewrite that
   slot's token ID to the gathered-row index, and submit the prepared buffers in
   original order. The GPU can now safely consume the row.
5. Cache growth and shared-slot reuse retain explicit barriers where the buffers
   being migrated or overwritten are still in use. Those are boundary events,
   not an unconditional steady-state wait.

The encoder's deferred queue is bounded by the number of command buffers in one
forward step (`dispatches / max_dispatches_per_buffer + 1`), and it is emptied
by terminal submission or discard. It does not retain a tree of work or keep
multiple decode steps unboundedly queued. One shared embedding row buffer remains
sufficient because step N+1 is not submitted until step N has completed and the
host has safely replaced that row.

That pipeline now survives a two-turn chat run, so the remaining throughput
investigation moves to the active K3 repacked GEMV. The local tree changes its
inner loop from one reused `float v[8]` to four simultaneously live
`float[8]` arrays and hoists eight packed words. Commit `5a344a8` previously
introduced the one-array shape specifically to reduce register pressure. The
current edit therefore needs a direct clean-versus-dirty A/B before any new
kernel design or historical engine bisect.

## Touchpoints

| Path | Current behavior | Intended change |
|---|---|---|
| `crates/kernels/src/encoder.rs` | `roll()` immediately commits each full command buffer. | Add explicit bounded deferred-submission mode for a single step; terminal commit submits deferred buffers in order, while abandon/drop discards them safely. Preserve normal behavior by default. |
| `crates/engine/src/model.rs` | Sparse decode drains before slot writes and gathers the row inside `forward_chunk_pick`. | Keep token/position slot rules; allow sparse pipelined encoding to refer to row scratch before it is populated, and expose a narrow stage-row operation after the previous step completes. Keep prefill and synchronous paths unchanged. |
| `crates/kernels/src/encoder.rs` | Deferred sparse forwards still roll at 50 dispatches, requiring about 42 unsubmitted buffers for the measured 2,100-dispatch decode step, beyond the hard cap of 32. | Give deferred sparse decode a 100-dispatch slice size (about 21 rolled buffers for the measured step); retain the hard cap and ensure failure abandons partial work and restores per-layer cache positions. |
| `crates/engine/src/generate.rs` | Encodes and submits next step before waiting on current output. | Encode next step while current runs, wait/read/EOS-check, stage the raw embedding row, then submit the prepared encoder. Preserve error cleanup and token accounting. |
| `crates/engine/src/chat.rs` | Encodes pending work before previous wait and drops pending on EOS. | Retain this shape; stage the row after previous wait and before committing pending work. |
| `crates/kernels/src/encoder.rs` tests | Cover ordinary rolling and flush behavior. | Add deferred submission, ordering, abandonment, terminality, and bound tests with a negative control. |
| `crates/engine/src/model.rs` and `crates/engine/tests/*` | Cover exact sparse row staging and pipeline behavior. | Exercise exact token row bytes, two-slot sequences, EOS, cache growth, reset, and error paths. |
| `spec/mlock-clean-mmap/*` | T5g records memory changes and an unrelated 12.334 tok/s run. | Add cross-reference to the decode regression outcome and clarify that the old number is not a matched performance comparison. |
| `crates/kernels/src/linear_repacked.metal` | The user-owned dirty edit keeps four decoded arrays live in the active K3 contig4 loop; committed HEAD reuses one array. | Preserve the patch, compare clean and dirty variants without source differences elsewhere, and only revert or revise it after measured attribution. No edit occurs during the specification-only phase. |

## Sequencing and dependencies

1. Pin the canonical benchmark and snapshot the user's two dirty patches. Make
   isolated clean-kernel and dirty-kernel worktrees at the same committed HEAD,
   each with its own `CARGO_TARGET_DIR`. Confirm with
   `KAPLUMBA_TRACE_GEMV=1` that the workload dispatches
   `linear_gemv_ishizuki_k3_colmajor_contig4`.
2. Implement encoder deferral and its isolated lifecycle tests.
3. Separate sparse-row preparation from sparse-row use in `TextModel`, keeping
   synchronous/prefill behavior on the established path.
4. Integrate stage-then-submit ordering in both generation callers and test EOS,
   error, reset, and cache-growth boundaries.
5. Run the full relevant unit/integration suites, then compare clean and dirty
   kernels in at least five interleaved pairs with identical prompt, generated
   count, greedy IDs, release settings, lock accounting, and swap observation.
6. If the committed kernel restores the target, retire only the dirty
   experiment. If it does not, bisect correctness-safe committed changes after
   `223ea09`, including cache, resource-view, and sparse pipeline work. Do not
   remove the cache-growth barrier without a separate correctness proof.
7. Update historical claims and close the tasks only with measured evidence.

## Interfaces

The new encoder mode is opt-in and per encoder instance. The API should make the
state transitions explicit: normal rolling/commit, deferred rolling, terminal
submission, or abandonment. Do not expose a general scheduler or global mode.
`TextModel` should provide a narrow method that stages the selected sparse row
only after the caller has waited for the preceding prediction. Avoid changing
public checkpoint or chat formats.

## Risks and fallbacks

- A forgotten submit could silently drop a step; make invalid state transitions
  fail in debug builds and test terminal use.
- An early commit could read stale row scratch; deferred buffers must be covered
  by an executable negative control that fails when commit happens early.
- EOS and errors must discard every deferred buffer without committing any of
  them, while preserving GPU cache rollback.
- Sliding KV growth may still synchronize. If it dominates an observed sample,
  report it separately and preserve correctness rather than bypassing migration
  barriers.
- Retaining many unsubmitted command buffers can expose Metal queue limits.
  Bound them to one step and test worst-case dispatch count against the real
  decode limit before model benchmarking.
- The user's uncommitted GEMV-kernel edit and extractor test are measurement
  confounders. Preserve both. Apply only the shader patch to the dirty A/B
  worktree and keep the extractor test out of both benchmark binaries.
- Four live decode arrays may be optimized away by the compiler, or may raise
  register use enough to reduce occupancy or spill. Treat source structure as a
  hypothesis; use latency first and counters/disassembly only if available.
- Existing pre/post-T5g logs are different from the canonical workload, but
  their nearly equal 12.439 and 12.334 tok/s results rule out T5g as a plausible
  explanation for the whole 15.8-to-12.4 loss. Preserve the safe deferred
  encoder work while investigating the kernel and earlier commits.

## Requirement-to-task mapping

| Requirement | Tasks |
|---|---|
| 1 matched baseline and causal attribution | T1, T6 |
| 2 preserve mmap/mlock and sparse-memory behavior | T3, T4, T6 |
| 3 overlap encoding and GPU work | T2, T3, T4 |
| 4 safe row staging before submission | T2, T3, T4 |
| 5 retain required barriers and cache growth | T3, T4, T5 |
| 6 EOS/error/drop cleanup | T2, T4, T5 |
| 7 chat/CLI and sampling regressions | T4, T5 |
| 8 sabotage-proven ordering regression | T2, T3 |
| 9 throughput and memory acceptance | T1, T6 |
| 10 deferred-buffer capacity and failed-step cache rollback | T7 |
| 11 isolate dirty active-kernel experiment | T1, T8 |
| 12 defer broader ALU work until baseline restoration | T8 |
