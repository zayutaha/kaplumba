# tasks — mtp-40tps

Amended 2026-10-02: direction changed from mlx-vlm + downloaded target to
**kaplumba Rust engine + local Gxma**. T4–T10 below are replaced; the audit work
T0–T3 is kept because it is measured, not speculative.

## Done (audit phase — measurements stand, they are physics)

- [x] **T0** Environment audit (disk, RAM, swap, versions, `~/.omlx` settings).
- [x] **T1** Memory floor for an MLX 4-bit Gemma-4-31B target: 17.27 GB text-only.
      *Now moot — Gxma is 11.6 GiB resident.*
- [x] **T2** Audit the mlx-vlm non-greedy MTP verification path
      (`speculative/common.py:135`, `speculative/mtp.py:182`) and prove the
      accept-on-equal rule distribution-exact on synthetic distributions.
- [x] **T3** Measure the box's memory ceiling and the NVMe re-read rate.
      → DRAM 222 GB/s, NVMe 22.1 GB/s, ~10 GB max wired.
- [x] **T3b** Read the drafter's real forward pass from the installed mlx-vlm
      (`speculative/drafters/gemma4_assistant/gemma4_assistant.py`) and the round
      loop (`speculative/mtp.py:523`). Architecture is now known; nothing is guessed.

## Gate — must be answered before any optimization work

- [x] **T4 — MEASURE REAL ACCEPTANCE RATE. ANSWERED: FAIL.**
      `crates/engine/examples/mtp_acceptance.rs`, real Gxma + real
      `qat-assistant-4bit` drafter, prompt = "The quick brown fox…", K=5,
      60 rounds, verified one token per forward (the acceptance *rate* does not
      depend on verify batching, so this needs no T8a kernel).

      | temperature | mean accepted/round | P(accept at depth 1) | implied tok/s |
      |---|---|---|---|
      | 0 (greedy) | **0.883** | 0.450 | 9.9 |
      | 1.0 / top_k 64 | **0.600** | 0.350 | 6.7 |

      **40 tok/s needs ≥ 3.14 accepted/pass. Measured 0.88 (greedy) and 0.60
      (temp 1.0). MTP with this drafter cannot reach 40 tok/s on this box, by a
      factor of ~3.5x in greedy and ~5x at temp 1.0.** The implied numbers are
      also optimistic: they charge nothing for the drafter's own forward and
      assume a free M>1 verify.

      **Why the drafter is the limit, not the wiring** — draft[0] lands inside
      the target's top-5 only **63%** of the time (top-20: 77%). A drafter
      trained against its own target should be far higher. This is spec Q2
      ("does the drafter actually match Gxma?") answered **no**, and it is a
      property of the checkpoint pair, not of this engine.
- [ ] **T5 — Drafter conversion.** Convert the MLX affine-4bit drafter
      (`bits 4, group_size 64, mode affine`, U32-packed + `scales`/`biases`) into
      the engine's format. Rust tool, no Python in the repo. Verify the conversion
      against a reference and **state whether the check is bit-level or
      distributional**.
  - [x] T5a — Rust unpack of MLX affine-4bit verified **bit-exact** vs
        `mx.dequantize` (`crates/engine/tests/mlx_affine_unpack.rs`, `0abe7ca`).
        Math: single-f32-rounded `nibble*scale+bias`; nibble order = LSB-first
        along K.
  - [x] T5b — `affine4_gemv` Metal kernel
        (`crates/kernels/src/affine.{metal,rs}`, `7bb42d6`), agrees with host
        reference. **Sabotage evidence**: `w_det += 1.0f` gives max-abs-err
        24.2 on the same test; the test also caught a broken shader build
        being silently skipped (fixed). Note: engine convention supplies
        FP16 activations/scales/biases (bf16→f16 conversion at load), so the
        drafter checkpoint's bf16 `scales`/`biases` must be converted the same
        way on load.
  - [x] T5c — `DrafterWeights` loader (`crates/engine/src/drafter.rs`,
        `5cbadbc`): bf16→f16 conversion of scales/biases/layernorms at load,
        packed u32 weights kept verbatim; `drafter_load_gemv_matches_host`
        proves loader + buffer GEMV agree on the real `pre_projection`.
- [x] **T6 — Drafter forward.** `pre_projection` (10752→1024) → 4 decoder layers
      reading the **target's shared K/V** → `norm` → `post_projection` (1024→5376),
      lm_head = tied `embed_tokens`. Validated **standalone** against MLX golden
      arrays from `Gemma4AssistantDraftModel` with fake shared-KV
      (`crates/engine/tests/drafter_forward.rs`, `5117572`): max abs diff
      0.03125 over 5376 values after 4 layers in f16 vs MLX's f32 — consistent
      with f16-vs-bf16 rounding, not a bug. No own KV cache.
      (T6 standalone done: the shared-KV *extraction* from the target is T8b
      territory; the round-loop needs it.)

## Integration

- [ ] **T7 — Baseline (R2).** Non-speculative decode of Gxma, 1024 tokens,
      ≥3 interleaved runs. Denominator for everything.
- [ ] **T8 — Draft/verify round loop (R3) — BLOCKED on T8a.** One target forward over
      `[bonus, cand_1..cand_k]`; accept-on-equal walk with **target and drafter RNG
      streams kept separate** (`common.py:46-132`) — that separation is what makes
      exactness hold. (Demoted: depended on a nonexistent M>1 forward; see T8a.)
- [ ] **T9 — KV rollback.** Trim target KV to `accepted` on rejection, and re-slice
      the shared-KV view the drafter reads. This is the most likely place for a
      silent distribution skew; it needs its own test.
- [ ] **T10 — Optimize to the bar (R5).** Draft depth, cache sizing, overlap,
      staging. One change at a time, A/B interleaved, negatives recorded.

## Proof

- [ ] **T11 — Distribution faithfulness (R6).** TV distance + two-sample χ² with
      p-values, many samples, same seed/temp/top_k, speculative vs not.
- [ ] **T12 — Coherence (R7).** Multi-turn at temp 1.0/top_k 64: repetition,
      truncation, EOS (`[1,106,50]`), cross-turn, vs the non-speculative baseline.
- [ ] **T13 — Write-up.** Measured ceiling, roofline, acceptance rate, all negative
      results with numbers.

## Order and dependencies

T5 → T6 → **T4 (gate)** → T7/T8 → T9 → T10 → T11/T12 → T13.

T4 is a gate, not a parallel task: if real acceptance is too low, T8–T10 are wasted
work and the answer is a roofline, not a speedup. Measure it as soon as the drafter
runs, before building the fast path.

**AMENDED 2026-10-03:** T8 is promoted to its own kernel spec. The engine has
no M>1 3 bpw forward path, and "T8" as an integration bullet was hiding that.

- [ ] **T8a — M>1 3 bpw verify kernel (own spec required).** A forward over
  `[bonus, cand_1..cand_k]` for K+1 ∈ {2,4,6,8}. Either an M=K+1 GEMV-style path
  or a 3 bpw GEMM. Must be bit-validated per row against the M=1 reference,
  benchmarked (bytes/token vs M=1 baseline), and sabotage-tested. No T8b
  (= old T8) until this exists.
  **FROZEN by T4.** At 0.88 accepted/round a verify kernel would have to be
  *and* the drafter improved to move the number, so this is not worth building
  until the drafter's top-5 containment (0.63) is raised.
- [ ] **T8b — Draft/verify round loop** (was T8). Blocked on T8a, and on a
  drafter that clears the gate.

## What T4 closed, and what it makes pointless

T4 measured 0.883 accepted/round greedy and 0.600 at temp 1.0, against the
3.14 that 40 tok/s requires. The gate therefore **fails**, and the standing
instruction in the spec is to stop and report the ceiling. Concretely:

- **T8a** (M>1 3 bpw verify kernel) — frozen. This is the largest remaining
  engineering item in the whole spec (the engine has no M>1 path at all) and it
  buys nothing: batching the verify changes how fast the confirmed tokens are
  produced, not which tokens they are, and the count is already too low.
- **T8b / T9 / T10** — moot for this drafter.
- **T11 / T12** (distribution faithfulness, temp-1.0 coherence) — these were
  written for a build that reached 40 tok/s. Acceptance at temp 1.0 is 0.600
  and the implied rate 6.7 tok/s, so there is no fast path whose coherence
  needs proving.
- **What is NOT closed**: the acceptance measurement itself is now a reusable
  instrument. `mtp_acceptance` will answer the same question in seconds for any
  future drafter, and the `draft[0] in top-k` line says whether a candidate is
  worth building anything on.

**The only thing that reopens this spec** is a drafter with materially higher
top-k containment against this target — i.e. one trained against *this* Gxma
checkpoint rather than the released `qat-assistant-4bit`.