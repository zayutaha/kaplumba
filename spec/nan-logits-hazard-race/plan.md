# plan: nan-logits-hazard-race

## Root-cause analysis and data flow

### Forward data flow (where the race sits)

A forward step encodes, onto one Metal command queue, a stream of compute
dispatches that all touch the same small set of shared buffers:

```
embedding → [per layer ×60]:  rms_norm(input) → normed
                                input_transform → transformed
                                GEMV(qkv)       → raw
                                attention(normed,q,k,v) → attended
                                o_proj          → hidden_a/hidden_b
                                gate/up/down    → raw → hidden
                              final_norm → last_hidden
                              lm_head GEMV → logits → softcap → capped
```

Each arrow is a **write-then-read on a shared buffer across two dispatches**. For
correctness, the write must be visible to the read. Two mechanisms provide that:

1. **Metal hazard tracking** (automatic). Metal inserts a memory barrier between
   dispatches that touch the same *tracked* resource.
2. **Explicit `memoryBarrier(resources:)`** (manual). Required when a resource is
   `HazardTrackingModeUntracked`.

`eed081e` flipped the KV cache, `ForwardScratch`, and norm/sign weights to
`HazardTrackingModeUntracked` (removing mechanism 1) **without** adding mechanism
2 anywhere in `crates/kernels/src/encoder.rs`. Neither mechanism was in effect, so
consecutive dispatches raced.

### Why it surfaced only on the GPU async path

- The host/CPU reference path (`TextModel::forward_layer`) computes layer-by-layer
  synchronously; it is unaffected by Metal hazard mode and was measured finite on
  all 60 layers. This is what made the model look "fine on CPU."
- The GPU path enqueues everything onto an async command buffer. Within one
  command buffer, dispatches with untracked shared buffers can reorder/overlap at
  the memory level → torn activations → NaN cascade.

### Why per-layer `flush_and_wait` fixed it but `MAX_IN_FLIGHT=1` did not

- `flush_and_wait` (`encoder.rs`) ends the encoder, commits, and waits for every
  in-flight buffer. Forcing completion at each layer boundary removes the window
  in which a reader could see pre-write data → 0 errors.
- `MAX_IN_FLIGHT=1` only waits at **roll** boundaries (every
  `MAX_DISPATCHES_PER_BUFFER` dispatches), which do not align with the dependent
  dispatch pairs inside a layer → still 3/3 errors.

This A/B isolated the fault to inter-dispatch visibility (hazard), not to
cross-step in-flight command-buffer overlap.

## File/module touchpoints

| file | current behavior | intended change |
|---|---|---|
| `crates/engine/src/gpu_cache.rs` (`new`, ~L64) | KV keys/values/positions allocated `StorageModeShared \| HazardTrackingModeUntracked` | drop `HazardTrackingModeUntracked` → tracked (R1) |
| `crates/engine/src/gpu_cache.rs` (`resize`, ~L186) | same untracked options on the grown buffers | drop untracked → tracked (R1) |
| `crates/engine/src/model.rs` (`ForwardScratch::new`, ~L2961) | scratch buffers (`normed`, `attended`, `raw`, `hidden_a/b`, …) untracked | drop untracked → tracked (R2) |
| `crates/engine/src/model.rs` (`u16_device_buffer`, ~L462) | norm/sign weight buffers untracked | drop untracked → tracked (R3) |
| `crates/engine/src/weights.rs` (~L1073) | packed GEMV weight buffers untracked | **left as-is**: static, read-only after load; no per-step write hazard. Revisit only if a residual race is proven. |
| `crates/engine/examples/*` (`probe_bw_250gb`, `sweep_geom`, `bench_colmajor`) | untracked | **left as-is**: isolated benchmark harnesses, not the production forward path. |

No kernel (`.metal`) changes. No encoder (`encoder.rs`) changes: we restore the
automatic barrier rather than hand-adding `memoryBarrier`, because tracked mode is
the simpler correct-by-construction fix and is what the code used before
`eed081e`.

## Interfaces / on-disk layout

Unchanged. No format, no checkpoint, no cache-layout change — only the Metal
resource-option flag on four in-memory buffer allocations.

## Sequencing and dependencies

1. Confirm the regression source: `git log -S HazardTrackingModeUntracked` →
   `eed081e`. (done)
2. Isolate mechanism: A/B `flush_and_wait` vs `MAX_IN_FLIGHT=1`. (done — points
   at hazard visibility.)
3. Revert the four `eed081e` buffer sites to tracked. (done)
4. Rebuild, verify AC1–AC5. (done)
5. Strip diagnostic scaffolding added during triage (per-layer/HEAD probes). (done)
6. Commit fix. (done — `9b1e9fd`)
7. Write this spec. (this step)

Steps 3–4 are the only code changes; 5 was cleanup of temporary instrumentation.

## Risks and fallbacks

- **Risk: perf regression.** `eed081e` was a perf optimization; tracked mode adds
  driver hazard-tracking overhead. **Mitigation:** correctness first. Measure tok/s
  after this fix and compare against the `eed081e` baseline. If the loss is
  material, pursue explicit-barrier untracked mode in a *separate* spec (must A/B).
- **Risk: incomplete revert** — `weights.rs` still untracked could hold a residual
  race. **Mitigation:** weights are read-only post-load with no inter-dispatch
  write; if any non-finite reappears, revert that site too (low effort).
- **Fallback:** if tracked mode somehow does not clear the race (it did, 0/6+),
  the next lever is adding explicit `memoryBarrier(resources:)` in
  `encoder.bind_and_dispatch` between dependent dispatches.

## Requirement → task mapping

| requirement | task(s) |
|---|---|
| R1 KV cache tracked | T1 (`gpu_cache.rs` new), T2 (`gpu_cache.rs` resize) |
| R2 scratch tracked | T3 (`model.rs` `ForwardScratch::new`) |
| R3 norm/sign weights tracked | T4 (`model.rs` `u16_device_buffer`) |
| R4 finite logits | verified by T5 (probe), AC3 |
| R5 no sampled error, deterministic | T6 (repeated T=1.0 runs), AC1 |
| R6 greedy coherent | T7 (greedy factual prompts), AC2 |
| (cleanup) no scaffolding left | T8 |
