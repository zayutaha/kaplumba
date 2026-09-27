# Plan: prefill at 200 tok/s

## Current state, measured

| quantity | value | how measured |
|---|---|---|
| prefill rate @600 tok | 9.67 tok/s | end-to-end `--metrics` |
| prefill rate @120 tok | 8.31 tok/s | end-to-end |
| prefill rate @24 tok | 4.28 tok/s | end-to-end |
| achieved prefill throughput | **0.56 TFLOPS** | 2 x 28.74e9 x 600 / 62.0 s |
| weights (non-embed/head) | 28.74e9 params, 10.93 GB @ 3.04 bpw | config + measured bpw |
| requirement @8192 tok | **11.5 TFLOPS** | 4.7e14 / 41.0 s |
| SIMT fp32 FMA peak | 8.8 TFLOPS | `alu_peak`, 4.38e12 FMA/s |
| tensor-core peak | **unmeasured** | task T0 |

The model is 60 layers, hidden 5376, intermediate 21504, 32 q heads x 256,
16 kv heads x 256 (10 global layers at 4 x 512). Non-embed/head parameters are
479 M per layer, 28.74 e9 total.

Prefill currently takes the `encode_gemm` / `PrefillEncodeRequest` path for the
linears and the **serial** `online_attention` kernel for attention
(`model.rs`: split-key is used only when `tokens == 1`).

## What is probably wrong, and what is not yet known

Two candidate root causes, deliberately not pre-judged — T1 settles it.

**Candidate A: the prefill GEMM is scalar.** If the prefill linear kernel is a
straightforward tiled loop with no tensor-core use, 0.56 TFLOPS against a
SIMT fp32 peak of 8.8 TFLOPS is about 6% of what SIMT alone could deliver. That
is consistent with a scalar kernel that is also memory- or latency-bound.

**Candidate B: the prefill attention kernel is spilling.** `online_attention`
declares

```metal
float accum[512];
```

indexed by a runtime `head_dim` loop bound. A dynamically indexed local array
cannot be promoted to registers, so every `accum[dimension]` in the inner
loop is a local-memory access. At 600 tokens attention is only ~0.5% of prefill
FLOPs, so it should be nearly free — but if it is spilling it can dominate
wall clock without ever showing up in a FLOP count.

Which of these dominates is exactly what R1 exists to find. Note that the
per-token prefill time *falls* as the prompt grows (0.233 s/tok at 24 tokens,
0.103 s/tok at 600), which is the signature of amortising a fixed cost, and
argues against attention being dominant at 600. At 8192 attention FLOPs grow as
L^2 while GEMM FLOPs grow as L, so the crossover moves — the split must be
re-measured at 8k, not extrapolated from 600.

## Enabling technique: Metal simdgroup matrix multiply

The reference implementations reach their prefill rates with Apple's
tensor-core path, not with better SIMT loops:

- oMLX routes head_dim 256 prefill to a kernel where "scores are never
  materialized and QK/PV use simdgroup MMA"
  (`patches/qwen35_fa256_attention.py:11-12`), with the native FP32 kernel
  rejected for exceeding 32 KiB of threadgroup memory
  (`patches/sdpa256_attention.py:10`).
- oMLX's Gemma 4 verify kernel is a hand-written vector kernel for this model
  family with per-row online softmax and zero threadgroup memory in pass 1
  (`patches/gemma4_verify_kernel.py:38-50`).
- exllamav3's GEMM accumulates MMA results in fp16 rather than fp32 and folds
  into fp32 accumulators on a fixed cadence, worth ~14% at batch 1
  (`exl3_gemm_inner.cuh:11-20`).

**Availability is confirmed on this machine.** `crates/engine/examples/metal_probe.rs`
compiles and runs `simdgroup_multiply` on `simdgroup_half8x8` on the M4 Pro.
Two API notes that cost time and are recorded in the probe's commit message:
the entry points are `simdgroup_load` / `simdgroup_store` / `simdgroup_multiply`
(not `simdgroup_matrix_*`), and the tile type is `simdgroup_half8x8` (no `_t`).

What is **not** established is the data layout convention. The probe's numeric
check (identity times a known matrix) currently matches only 8 of 64 elements,
so the path is callable but our usage is wrong. T2 fixes that with a unit test
before any of it is built into the model.

## Architecture

Prefill is a different problem from decode and wants a different shape:

- Decode is memory-bound: 11.47 GB/token at 128 GB/s. Arithmetic is free — the
  ablation probes removed the decode arithmetic and the activation loads with no
  measurable change, and removing only the weight loads was 1.87x.
- Prefill is compute-bound and should reuse each weight across all prompt tokens.
  At 8192 tokens the weight stream is a one-off ~62 ms against 41 s of
  arithmetic, i.e. **0.15% of the budget**. Bandwidth is irrelevant here; FLOPS
  is the entire game.

So the design is a conventional tiled GEMM with tensor-core inner tiles, plus a
tiled online-softmax attention, replacing two kernels that are currently scalar.

### Prefill attention (R4)

Replacing `online_attention`'s per-thread `accum[512]` with a tiled kernel:

- One threadgroup per (query tile, head); a simdgroup matrix tile per query.
- Keys streamed in tiles of a fixed size (the oMLX kernels use 4 or 8).
- Online softmax state — running max, running sum, and the output accumulator —
  held in registers, updated once per key tile rather than once per key.
- The only scratch is the key tile itself, so the working set is O(L) and
  **independent of context length** (R5), and no score matrix is ever formed.
- The accumulator must be register-resident. Any residual `float[n]` indexed by
  a runtime bound is a bug, and the check for it is a measured local-memory
  figure, not an inspection.

### Prefill GEMM (R2)

- Tile the output over M (prompt tokens) and N, stream K.
- Inner tile is `simdgroup_half8x8` multiply-accumulate.
- Accumulate in fp16 within a tile and fold into fp32 on a fixed cadence, the
  way exllamav3 does, to keep the fp32 conversion off the hot path.
- Dequantisation stays per-weight-load and hoisted per quant group, as it is in
  the decode kernel: the reference affine path notes "no FP multiply per
  element" and "no dynamic array index (avoids register spilling on Apple GPU)".

Scratch stays inside the existing `ForwardScratch` allocation per the standing
restriction and OQ2.

## Risks and fallbacks

- **The target may be a wall (OQ4).** If T0 measures a tensor-core peak near
  12 TFLOPS, 11.5 TFLOPS is unreachable at 3.04 bpw and the honest answer is
  that the only lever is bits-per-weight. T0 exists to find this out in the
  first task rather than after a week of work.
- **"No extra buffers" may block a double-buffered GEMM** (OQ2). Fallback is a
  single-buffered tile with a modest throughput loss, which must be measured
  rather than assumed.
- **fp16 accumulation changes numerics** (OQ3). exllamav3 reports max observed
  error ~1% of output RMS at k=4096. That may be acceptable, but the tolerance
  has to be agreed, not discovered.
- **Attention may be the 8k problem, not the GEMM.** At 8192 the O(L^2) term is
  ~2.8% of FLOPs but could be a much larger share of time if the current kernel
  is spilling. T1 must therefore report the split at 1k / 2k / 8k, not just 600.
- **The preemption cliff (R8)** could make a long single dispatch collapse
  prefill throughput on this pre-NAX part. If a tiled kernel wants a large
  per-dispatch work quantum, it must be chunked with logsumexp partials, the way
  oMLX does, with a 10 ms target.

## Requirement to task mapping

| requirement | tasks |
|---|---|
| R1 attribution | T1 |
| R2 tensor-core path | T0, T2, T3, T4 |
| R3 FLOPS reporting | T1, T5 |
| R4 attention working set / no spill | T1, T4 |
| R5 fixed-size scratch | T4, T6 |
| R6 peak RSS at 1k/2k/8k | T6 |
| R7 numerical agreement + suite | T2, T3, T4, T5 |
| R8 preemption audit | T5 |
