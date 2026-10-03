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

- [ ] **T4 — MEASURE REAL ACCEPTANCE RATE.** The single number that decides whether
      this task is possible. Run the real drafter against real Gxma at
      `temperature 1.0, top_k 64` and report accepted-per-round for K = 1, 3, 5.
      The synthetic estimate c ≈ 0.20–0.40 would put 40 t/s out of reach; the real
      value is unknown. **If measured acceptance cannot reach ~0.73, stop and
      report the ceiling with roofline arithmetic** (Q1 in spec.md).
      Requires T5 first (the drafter has to run to be measured).
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
- [ ] **T6 — Drafter forward.** `pre_projection` (10752→1024) → 4 decoder layers
      reading the **target's shared K/V** → `norm` → `post_projection` (1024→5376),
      lm_head = tied `embed_tokens`. The drafter keeps **no cache of its own**
      (`gemma4_assistant.py:104-106`); it reads the target's per-layer-type shared
      K/V, which is why it costs no extra KV memory.

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
- [ ] **T8b — Draft/verify round loop** (was T8). Blocked on T8a.