# spec: nan-logits-hazard-race

## Goal

Eliminate the intermittent `logits must be non-empty and finite` failure and the
silent greedy garbage by removing the GPU inter-dispatch race that let
consecutive kernels read a shared scratch buffer before the prior kernel's write
was visible. Outcome: the Gemma 4 31B EXL3 3.0 bpw checkpoint (`Gxma`) produces
finite logits on every forward step at the default sampling settings
(`temperature 1.0, top_k 64`), on both the greedy and sampled paths.

## Background (what was wrong)

Commit `eed081e` ("spec/decode-25tok-peak-bandwidth: untracked hazard modes and
burst loading upfront", 2026-10-01) set
`MTLResourceOptions::HazardTrackingModeUntracked` on the KV cache buffers, the
`ForwardScratch` activation buffers, and the norm/sign weight buffers. The stated
intent was to skip driver hazard-tracking overhead during rapid step execution.
Untracked mode means Metal will **not** insert automatic memory barriers between
dispatches that read/write those buffers. The encoder
(`crates/kernels/src/encoder.rs`) never inserts an explicit
`memoryBarrier(resources:)` to compensate. The result: a dispatch that writes a
scratch buffer and the next dispatch that reads it can run without a visible
hazard dependency, so the reader consumes torn/partial data.

Downstream symptom chain:

- The corrupted activation propagates through the 60-layer stack and the
  vocabulary head, so `scratch.logits` (pre-softcap) and `scratch.capped`
  (post-softcap) come out all-NaN: **262144 / 262144 non-finite**.
- The sampler (`crates/engine/src/sampler.rs:102`) correctly rejects a
  non-finite row with `SamplerError::InvalidLogits` → the CLI prints
  `logits must be non-empty and finite`. This is a guard, not the bug.
- The greedy path (`crates/engine/src/generate.rs:269-278`,
  `generate_gpu_greedy`) bypasses `sample()` and argmaxes `scratch.logits`
  directly, so it never errors — it silently argmaxes over NaN and emits
  token 0 / incoherent text. Greedy "working" was therefore never evidence of
  correctness.

The race was intermittent because it depended on GPU scheduling/timing within a
command buffer; a host-side `flush_and_wait` (which forces completion) made it
disappear, while `KAPLUMBA_MAX_IN_FLIGHT=1` (which only waits at roll
boundaries) did not.

## Non-goals / restrictions

- **No quality loss.** The fix must restore bitwise-correct forward results, not
  approximate them. No KV-cache quantization, no activation down-sampling.
- **No graph/tree-shaped prefill scratch** (standing project directive). Scratch
  buffers stay fixed-size and independent of context length.
- **Do not re-introduce `HazardTrackingModeUntracked` without explicit
  barriers.** The perf motivation behind `eed081e` is real, but untracked mode is
  only correct when every dependent dispatch pair is separated by an explicit
  `memoryBarrier`. Any future attempt must add those barriers and be A/B measured.
- **No Python in product/engine code** (standing directive). Diagnosis used
  throwaway shell + Rust probes only; none ship.
- This spec does **not** attempt to recover the `eed081e` perf win. It only
  restores correctness. Re-optimization is a separate, later spec.

## Requirements

- **R1.** KV cache buffers (`crates/engine/src/gpu_cache.rs`, both `new` and
  `resize`) are allocated with default (tracked) hazard mode so Metal inserts the
  inter-dispatch barrier between the KV-append write and the attention read.
- **R2.** Activation scratch buffers (`ForwardScratch::new`,
  `crates/engine/src/model.rs`) are allocated with default (tracked) hazard mode.
- **R3.** Norm/sign weight buffers (`u16_device_buffer`,
  `crates/engine/src/model.rs`) are allocated with default (tracked) hazard mode.
- **R4.** `scratch.logits` and `scratch.capped` are non-finite-free (0 of 262144)
  on prefill and on every decode step, at `temperature 1.0, top_k 64`, for the
  `Gxma` checkpoint.
- **R5.** The sampled path (`--temperature 1.0`) runs to completion without the
  `logits must be non-empty and finite` error, deterministically across repeated
  runs (not a single lucky run).
- **R6.** The greedy path produces coherent, correct text (not token-0 garbage),
  confirming the argmax is over finite logits.

## Acceptance criteria

- **AC1 (determinism).** ≥6 consecutive runs of
  `kaplumba --model Gxma --temperature 1.0 --seed 3 --max-new-tokens 8` produce
  **0** occurrences of `logits must be non-empty and finite`. (Measured: 0/6.)
- **AC2 (greedy correctness).** `--temperature 0` on "2+2=" yields `4`, and on
  "The capital of France is" yields `Paris`. (Both measured correct.)
- **AC3 (finite logits).** A forward-step probe reads `scratch.logits` and
  `scratch.capped` and reports `0 / 262144` non-finite on prefill and decode.
  (Measured: 0/262144 for t=15, t=2, t=1 chunks.)
- **AC4 (sampled coherence).** `--temperature 1.0` on a factual prompt yields a
  coherent sentence (measured: photosynthesis one-liner is correct).
- **AC5 (no regressions).** `cargo build --release` clean; existing engine tests
  still pass.

## Open questions

- None blocking. Follow-up (tracked as future work, not required here): can the
  `eed081e` hazard-tracking perf win be recovered by inserting explicit
  `memoryBarrier` calls at the specific dispatch boundaries that need them, and
  does that A/B positively against tracked mode? That is a separate spec.
