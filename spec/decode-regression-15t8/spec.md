# Restore pipelined ordinary decode after sparse embedding staging

## Goal

Restore ordinary greedy decode throughput on the M4 Pro while retaining the
working shared-mmap and hot-weight `mlock` path and the sparse embedding memory
reduction. The canonical performance target is at least 15.8 tok/s on the
documented 32-token counting benchmark with DFlash disabled. A controlled
comparison must identify the change or combination of changes responsible for
the current lower rate before a performance fix is accepted.

## Non-goals and restrictions

- Do not change Metal kernels or the model/checkpoint format for this work.
- Do not disable or weaken the actual hot-weight `mlock`, MAP_SHARED mapping,
  sparse embedding staging, bounded Metal resource views, or the warning on
  partial lock failure.
- Do not bind the full embedding table into the Metal residency set or copy the
  full table into anonymous memory.
- Do not change greedy token selection, sampled output semantics, embedding
  dtype conversion/scaling, KV precision, or cache correctness.
- Do not optimize DFlash in this change; it is disabled for the target workload.
- Do not claim that the 15.8 tok/s floor applies to every prompt, context length,
  or sampling mode. The user's exact historical prompt and context are unknown.
- Keep queue submissions bounded to one prepared decode step; do not add a
  general graph queue, unbounded command-buffer retention, or another model
  instance.

## Requirements

1. Establish a matched ordinary-chat baseline with DFlash disabled. Use the
   canonical command from `spec/decode-25tok-peak-bandwidth`:
   `--prompt "Count from 1 to 50: 1, 2, 3," --max-new-tokens 32 --temperature 0
   --metrics`, model `~/.kaplumba/models/Gxma-repacked`. Record build hash,
   dirty-tree state, model header identity, context, token IDs, timing samples,
   and system swap deltas. Use interleaved measurements and report median and
   spread. Compare the current tree to a diagnostic build before T5g
   (`88205a0`) and then isolate relevant commits where feasible.
2. Preserve the memory behavior established by `spec/mlock-clean-mmap` T5g:
   MAP_SHARED zero-copy mappings, the same eligible locked-byte set, sparse
   embedding source excluded from the Metal residency set, bounded row scratch,
   and warning-and-continue lock failure behavior.
3. For sparse pipelined decode, allow CPU command encoding for token N+1 to
   overlap GPU execution for token N. Do not perform an unconditional queue wait
   at entry to every steady-state decode step.
4. Do not submit any command buffer that reads a sparse embedding row until the
   host has copied the exact row for that step into shared scratch. Preserve
   command order across rolled buffers and bound unsubmitted buffers to one
   decode step.
5. Preserve safe synchronization where required: shared input-slot reuse,
   actual host reads of GPU-produced token IDs, sliding-cache growth/migration,
   and reuse of the single embedding-row staging buffer. Cache growth may create
   an explicit synchronization boundary; steady-state decode must not.
6. If a generated token is EOS, discard the prepared next-step buffers without
   submitting them, and preserve existing KV rollback/retract behavior. Error,
   cancellation, reset, and drop paths must release ended unsubmitted command
   buffers safely.
7. Keep chat and the CLI generation pipeline behaviorally aligned. Verify greedy
   IDs exactly against the pre-change same-workload control and run sampled and
   non-pipelined regressions for unaffected routes.
8. Add an executable regression that detects deferred command buffers being
   committed before row staging and demonstrates the sparse decode overlap
   sequence. Sabotage the submission guard and confirm a named test fails before
   making a coverage claim.
9. Demonstrate at least 15.8 tok/s on the canonical benchmark in a controlled
   interleaved A/B run, with identical output IDs and no new swapouts. Also
   record ordinary chat throughput at 1k/2k/8k contexts as comparisons; these
   longer contexts are not subject to the canonical 15.8 floor. Record load,
   TTFT, decode median/spread, memory pressure, locked bytes, and swap deltas.

## Acceptance criteria

- The matched canonical benchmark reaches at least 15.8 tok/s median on the
  fixed implementation, with token IDs identical to its same-workload control.
- A/B/A/B-style interleaving has enough samples to expose run-to-run spread;
  provide at least five paired samples and do not count instrumented runs in
  the throughput result.
- The test for command-buffer staging order fails under deliberate early-submit
  sabotage and passes after restoration. The observed failure is recorded.
- Existing encoder, pipeline, cache-growth, chat, and generation tests pass.
- A model run reports the expected locked bytes, and system swapouts do not
  increase during the measured decode interval. Any existing swap is recorded
  as a starting value, not attributed to the run.
- `implementation_notes.md` records the exact commands, revisions, outputs,
  environmental state, and any limits. Older throughput claims are marked with
  their actual workload so unlike measurements are not conflated.

## Open questions

- The user's prior 15.8 tok/s observation was ordinary decode with DFlash off,
  but the exact prompt/context are unavailable. The documented canonical
  benchmark is the reproducible acceptance workload; the original observation
  remains an unverified comparison.
- The exact share of the 15.8-to-12.25 tok/s drop caused by T5g is unmeasured.
  Existing notes include a 12.439 tok/s run before T5g on a different 39-token
  prompt and 160-token generation. Attribute causality only from matched runs.
