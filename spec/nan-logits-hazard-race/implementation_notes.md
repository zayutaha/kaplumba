# implementation_notes: nan-logits-hazard-race

Handover log. Factual, dense. Read `spec.md` for what/why and `plan.md` for the
mechanism.

## Outcome

Bug fixed and committed as **`9b1e9fd`** (`fix: revert untracked hazard mode
(eed081e) causing NaN logits`). Only four lines of real change (the four buffer
allocation sites), plus explanatory comments. `Gxma` (Gemma 4 31B, EXL3 3.0 bpw)
now produces finite logits on both greedy and `temperature 1.0` paths.

## The bug, precisely

Commit **`eed081e`** (2026-10-01, "spec/decode-25tok-peak-bandwidth: untracked
hazard modes and burst loading upfront") set
`MTLResourceOptions::HazardTrackingModeUntracked` on:
- KV cache buffers (`gpu_cache.rs` `new` + `resize`),
- `ForwardScratch` activation buffers (`model.rs` `ForwardScratch::new`),
- norm/sign weight buffers (`model.rs` `u16_device_buffer`).

Untracked mode removes Metal's **automatic inter-dispatch memory barrier**. The
encoder (`crates/kernels/src/encoder.rs`) never adds an explicit
`memoryBarrier(resources:)` to replace it (verified: zero `memoryBarrier` calls
in the whole dispatch path). So a dispatch that writes a shared scratch buffer
and the next dispatch that reads it could race → torn activations → NaN cascade
→ all 262144 logits non-finite.

The `eed081e` implementation notes claim "bit-exactness verified with
`live_kernel_bit_exact` and `repacked_golden` (100% pass, 0 diffs)" — those tests
run kernels in isolation (explicitly synchronized), so they **cannot catch an
inter-dispatch back-to-back race**. That is the recurring false-coverage failure
mode AGENTS.md §3 warns about: the test's harness synchronization masked the very
hazard being introduced.

## Symptom → who to blame

- CLI error: `logits must be non-empty and finite` from
  `crates/engine/src/sampler.rs:102` (`SamplerError::InvalidLogits`). **The
  sampler is correct** — it is the messenger. Do not "fix" sampler.rs.
- Greedy path (`crates/engine/src/generate.rs:269-278`, `generate_gpu_greedy`)
  bypasses `sample()` and argmaxes `scratch.logits` directly → never errors,
  silently argmaxes over NaN → emits token 0 / garbage ("Hello достаточно"). So
  greedy "working" was never evidence of health.

## What was NOT the bug (dead ends — do not re-chase)

Recorded because each looked promising and each was wrong:

- **"3-bpw kernel is broken."** Wrong. Host/CPU path (`forward_layer`) was finite
  on all 60 layers for `Gxma`; only the GPU async path corrupted. The rate
  (`Integer(3)`/`Integer(6)`) loaded correctly.
- **"single-pass decode attention kernel."** `KAPLUMBA_NO_SINGLE_PASS=1` did not
  fix it (post-revert verification: still 1/1 error).
- **"cross-layer norm fusion / RMS norm."** `KAPLUMBA_SKIP_HIDDEN_NORMS=1`
  produced finite output only because it turned the norm into a no-op feeding
  stale-but-finite memory — it suppressed propagation, not the cause.
- **"down_proj 3-bpw GEMV."** Ablation `KAPLUMBA_ABLATE=down_gemv` looked like it
  helped, but ablations change encoding/timing, which changes race scheduling.
  Unreliable for a timing-dependent race.
- **`scale: 1.0` in the GPU attention request** (vs `1/sqrt(head_dim)` in the
  skip branch): both host and GPU paths use `scale: 1.0`; kernels apply
  `params.scale` internally. Not a bug.
- **Model corruption / repack.** Rejected: all checkpoints dated 22 Sep, and a
  fresh repack reproduced the same NaN. Repacking cannot fix a hazard race.
  (`Gxma-repacked` and `Gxma-fresh` were deleted during triage per user request.)

## Diagnostic method that actually pinned it (reusable)

1. **Per-layer flush probe.** Added a `KAPLUMBA_NAN_LAYER_PROBE` env-gated block
   in the `forward_chunk` layer loop that calls `encoder.flush_and_wait()` before
   reading each layer's `input/normed/raw/output` buffers as host f16/f32, plus a
   `HEADPROBE` after `encode_lm_head` reading `last_hidden/logits/capped`.
   **Critical detail:** you MUST `flush_and_wait()` before reading
   `buffer.contents()`, or you read stale GPU memory from the previous layer and
   get *non-deterministic* phantom NaN. My first probe runs reported "layer 16 is
   the first NaN layer" — that was a stale-read artifact. After adding the flush,
   all 60 layers read clean.
2. **The decisive A/B** (this is what isolated hazard-visibility vs in-flight
   overlap):

   | condition | errors (3× runs) |
   |---|---|
   | default (async) | 3/3 |
   | `KAPLUMBA_MAX_IN_FLIGHT=1` | 3/3 |
   | per-layer `flush_and_wait` (probe) | **0/3** |

   `MAX_IN_FLIGHT=1` serializes only at *roll* boundaries (every
   `MAX_DISPATCHES_PER_BUFFER`=50 dispatches) and still failed; per-layer flush
   (forces completion at layer boundaries) fixed it. Conclusion: the fault is
   **inter-dispatch hazard visibility inside a command buffer**, not cross-step
   in-flight command-buffer overlap. → led directly to `HazardTrackingModeUntracked`.

3. **Regression bisect:** `git log -S HazardTrackingModeUntracked` → `eed081e`.

## The fix (what shipped in 9b1e9fd)

Four allocation sites reverted from
`StorageModeShared | HazardTrackingModeUntracked` → `StorageModeShared` (tracked):

- `crates/engine/src/gpu_cache.rs` — `GpuKvCache::new` (~L64) and `resize` (~L186)
- `crates/engine/src/model.rs` — `u16_device_buffer` (~L462) and
  `ForwardScratch::new` (~L2961)

Each site got a comment naming `eed081e` and the race rationale. **Diagnostic
scaffolding (the three probe blocks) was stripped before commit** — grep for
`NAN_LAYER_PROBE|HEADPROBE|NANPROBE` = 0.

**Left as-is (deliberate):**
- `crates/engine/src/weights.rs:1073` still untracked — packed GEMV weights are
  read-only after load, no per-step write hazard. Revisit only if a residual race
  is proven (follow-up F2).
- `crates/engine/examples/{probe_bw_250gb,sweep_geom,bench_colmajor}.rs` still
  untracked — isolated benchmark harnesses, not the production forward path.
- **No `.metal` kernel change, no `encoder.rs` change.** Tracked mode is the
  correct-by-construction fix (it is what the code used before `eed081e`).

## Verification (all measured post-fix)

- **AC1 determinism:** 6 consecutive `--temperature 1.0 --seed 3
  --max-new-tokens 8` runs on `Gxma` → **0/6** `logits must be non-empty and finite`.
- **AC3 finite:** probe read `logits`/`capped` → **0 / 262144** non-finite for
  t=15 (prefill), t=2, and t=1 (decode) chunks. `last_hidden` also 0/5376.
- **AC2 greedy:** `2+2=` → `4`; `The capital of France is` → `Paris`.
- **AC4 sampled:** `T=1.0` "Write a one sentence summary of photosynthesis." →
  "Photosynthesis is the process by which plants use sunlight to convert carbon
  dioxide and water into food."
- **AC5 build:** `cargo build --release -p kaplumba-cli` clean.

## Exact commands

```bash
# build
cargo build --release -p kaplumba-cli

# the previously-failing case (should now produce 0 errors)
scripts/gpuq try -- ./target/release/kaplumba \
  --model ~/.kaplumba/models/Gxma/Gxma \
  --prompt "Hello there" --max-new-tokens 8 --temperature 1.0 --seed 3 2>&1 \
  | grep -cE "logits must"

# greedy correctness
scripts/gpuq try -- ./target/release/kaplumba \
  --model ~/.kaplumba/models/Gxma/Gxma \
  --prompt "2+2=" --max-new-tokens 8 --temperature 0

# regenerate the kernel table (in case a .metal edit is ever needed)
python3 scripts/gen_exl3_gemv.py
```

Note: GPU discipline per AGENTS.md §10 — one model process at a time, use
`scripts/gpuq status` / `scripts/gpuq try` (never `run` by default), interleave
A/B. Every probe/model run in this triage went through `gpuq try`.

## Open follow-ups (not blocking)

- **F1:** can `eed081e`'s perf win be recovered with explicit
  `memoryBarrier(resources:)` at dependent dispatch boundaries, A/B vs tracked?
  If pursued, must add barriers *and* measure — untracked without barriers is
  exactly the bug just fixed.
- **F2:** `weights.rs:1073` untracked — confirm safe (read-only) or revert.
