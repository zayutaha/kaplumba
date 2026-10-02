# tasks: nan-logits-hazard-race

Status legend: `[x]` done. Code change landed in commit **`9b1e9fd`**
(`fix: revert untracked hazard mode (eed081e) causing NaN logits`).

## Checklist

- [x] **T1** — `crates/engine/src/gpu_cache.rs` (`GpuKvCache::new`): drop
  `| MTLResourceOptions::HazardTrackingModeUntracked` from the options used for
  `keys`/`values`/`positions`, leaving `StorageModeShared` (tracked).
  *Intent:* restore Metal's automatic KV-append → attention hazard barrier.
  *Test:* build clean; contributes to AC1/AC3.
  → `9b1e9fd`

- [x] **T2** — `crates/engine/src/gpu_cache.rs` (`resize`): same revert on the
  grown `new_keys`/`new_values`/`new_positions` allocations.
  *Intent:* the resize path must not reintroduce untracked KV buffers.
  *Test:* build clean; exercised whenever the cache grows.
  → `9b1e9fd`

- [x] **T3** — `crates/engine/src/model.rs` (`ForwardScratch::new`): drop
  untracked from the scratch `options`, leaving `StorageModeShared`.
  *Intent:* restore hazard barriers between the normed/transformed/raw/attended/
  hidden activation dispatches — the primary NaN source.
  *Test:* AC3 (logits 0/262144 non-finite), AC1.
  → `9b1e9fd`

- [x] **T4** — `crates/engine/src/model.rs` (`u16_device_buffer`): drop untracked
  from the norm/sign weight buffer options.
  *Intent:* restore hazard ordering for weight-reading kernels whose predecessors
  write scratch.
  *Test:* build clean; contributes to AC1/AC2.
  → `9b1e9fd`

- [x] **T5** — Verify finite logits via a forward probe (`scratch.logits`,
  `scratch.capped`) on prefill (t=15) and decode (t=1) after rebuilding.
  *Result:* **0 / 262144 non-finite** for every chunk size (AC3). *(The probe
  was temporary instrumentation, since removed in T8.)*

- [x] **T6** — Determinism check: 6 consecutive `--temperature 1.0 --seed 3
  --max-new-tokens 8` runs on `Gxma`.
  *Result:* **0/6** `logits must be non-empty and finite` (AC1).

- [x] **T7** — Greedy correctness: `--temperature 0` on factual prompts.
  *Result:* `2+2=` → `4`; `The capital of France is` → `Paris` (AC2). Sampled
  `T=1.0` photosynthesis prompt → coherent sentence (AC4).

- [x] **T8** — Strip diagnostic scaffolding added during triage
  (`KAPLUMBA_NAN_LAYER_PROBE` per-layer + `HEADPROBE` blocks in `model.rs`).
  *Result:* `grep -c NAN_LAYER_PROBE|HEADPROBE|NANPROBE` → 0; diff reduced to
  only the four hazard-tracking reverts.

- [x] **T9** — Existing tests / clean build.
  *Result:* `cargo build --release -p kaplumba-cli` clean, no errors (AC5).

- [x] **T10** — Commit fix + write this spec (`spec.md`, `plan.md`,
  `tasks.md`, `implementation_notes.md`).

## Deferred / follow-up (not required for this spec)

- [ ] **F1** — Decide whether to recover the `eed081e` hazard-tracking perf win
  by adding explicit `memoryBarrier(resources:)` at dependent dispatch boundaries,
  A/B measured against tracked mode. Separate spec (see spec.md Open questions).
- [ ] **F2** — Re-check `crates/engine/src/weights.rs:1073` (still untracked) if
  any non-finite reappears; currently read-only post-load, presumed safe.
