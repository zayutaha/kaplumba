# Spec: omlx-based single-user Gemma 4 31B engine (EXL3 3.0 bpw)

## Goal

Deliver a zero-overhead, single-user chat inference engine built on **omlx** as the base, serving **Gemma 4 31B-class dense** weights quantized with **EXL3 ~3.0 bpw (~15.5 GB) + ~1 GB SWA/KV ≈ 16.5 GB resident** on an **M4 Pro (20 GPU cores, 273 GB/s)**, optimizing in priority order **TTFT, tg/s, pp/s, memory**, with **no quality loss** and **no hot-path overhead margin**. Techniques are researched from `reference/{llama.cpp, ik_llama.cpp, vllm, PonyExl3}` and ported selectively; Paged KV / prompt-cache behavior is specified separately in `spec/paged-kv-cache/`.

## Non-goals / restrictions

- **Single-user chat only** — no multi-tenant scheduler tax on the hot path, no batching for its own sake.
- **No KV cache quantization** (unless a future spec amends this).
- **No quality loss** — speculative decoding verify-gated (greedy token-identical at temp 0; distribution-exact sampling at temp > 0). EXL3 3.0 bpw must be validated equal-or-better than the agreed quality baseline (see OQ).
- **No overhead margin** — profile before adding machinery; strip scheduler/sync/serialization from the steady-state path.
- Do not treat `reference/` as a runtime dependency tree; port ideas/patches deliberately.
- Do not reimplement omlx features already present (decode burst, external prefill, skip_lm_head, boundary snapshots, MTP scaffolding, prefix cache primitives) — fix and constrain them instead.
- No multi-model pool complexity required for v1 (single resident model); pool code may stay but must not add per-token cost.
- Model weights only from `~/.kaplumba/models`.
- `reference/` is read-only.

## Target metrics (hardware truths)

- Decode roofline: 273 GB/s ÷ 15.5 GB ≈ **~17.6 tok/s** plain (weights-dominated); realistic plain after KV/norms lower.
- Prefill is compute/dequant-bound on 20 GPU cores — kernel strategy dominates (ik_llama.cpp showed 1.5–7× pp/s gaps vs mainline from GEMM structure alone).
- Any tg/s claim above roofline without speculation is invalid.

## Requirements

### R1 — Base and single-tenant mode

- R1.1 Engine is omlx-based; changes are expressed as patches/flags/wrapper — not an unrelated rewrite.
- R1.2 A **single-tenant mode** disables multi-tenant-only hot-path work: fairness gates, cross-engine activity registry, decode-debt accounting, per-step admin snapshots, store-cache admission coupling, EnginePool lock contention where avoidable.
- R1.3 Decode path uses fixed-shape compiled graphs (`mx.compile` analog of llama.cpp graph reuse): stable topology, padded KV length class, warmup, no per-token graph rebuild.
- R1.4 Per-token CPU work is bounded: append-only output bookkeeping (no O(n²) list copy), single-pass CDF sampling with buffer reuse, tokenize once per request.

### R2 — Prefill (pp/s and TTFT)

- R2.1 Dual-path matmul: `seq==1` → dedicated 4-bit (or EXL3 trellis) GEMV; `seq ≥ threshold` → dequant/stage weights to int8-class tiles in fast memory, GEMM across token columns (ik amortize-unpack principle).
- R2.2 Multi-row interleaved weight packing at load (4/8-row scale sharing; Apple-friendly metadata: co-located scales, blocks-of-32 guidance).
- R2.3 Shape-dispatched GEMM selection (never one generic kernel at pathological aspect ratios).
- R2.4 Remove mid-prefill forced `mx.synchronize`+`mx.clear_cache()` per chunk; idle/deferred/pressure clears only.
- R2.5 `skip_lm_head` retained for prefill; chunk sizing tuned once syncs removed.
- R2.6 Gemma SWA-aware attention: bound KV loop for sliding layers (`n_kv_max ≈ pos - window`); avoid paying full-context cost on 5/6 local layers.
- R2.7 TTFT path: no queueing behind decode bursts that a single user did not ask for; warmup strategy defined (first-token latency vs shader compile).

### R3 — Decode (tg/s)

- R3.1 Sibling fusion of same-input linears with per-group size threshold and topology guards (qkv; gate+up only above threshold — PonyExl3 lessons: wrong fusion = ~2× regression).
- R3.2 MTP speculative decode using Gemma 4 built-in drafter with **shared target KV** (no second context); verify batched in one compiled step; depth configurable (omlx gemma4 default depth exists — re-validate).
- R3.3 Optional n-gram/lookup drafting (host-only, verify-gated) as cheap secondary lever.
- R3.4 MTP output-head memory: reuse/tie or re-quantize draft head projection if it duplicates full vocab matrix (ik `-mtprot` idea) — without quality loss on verify path.
- R3.5 GEMV kernels specialized for batch 1 and small verify batches (rows 2–8) reaching bandwidth roofline (llama.cpp mul_mv ext family + PonyExl3 simdgroup GEMV structure as references).
- R3.6 Sampling fast path (R1.4) without reintroducing RNG residency bugs that force periodic mid-decode cache clears.
- R3.7 No mid-decode forced full GPU sync/clear except deferred idle or hard memory pressure (document thresholds).

### R4 — EXL3 3.0 bpw path

- R4.1 Produce Gemma 4 31B EXL3 checkpoint at **~3.0 bpw** into `~/.kaplumba/models` (external converter CLI allowed; pin versions).
- R4.2 Quality gate: **sampled golden review** at temp=1.0, topk=64, top_p=0.95 vs reference quant (OQ1) must pass before default.
- R4.3 Runtime: **fresh EXL3 trellis GEMV + load/dequant reimplemented inside omlx** from reference material; do not vendor/embed PonyExl3 engine code (OQ8). On 24 GB, fold16-for-weights is banned (OQ3).
- R4.4 Re-sweep M4 Pro–specific tunables (split-K threadgroup target, fuse thresholds, TBX) — do not copy M5 Max numbers blindly.
- R4.5 Converter/runtime version pins recorded; `fold16`/`w4a16` lossy engines are not defaults.

### R5 — Memory

- R5.1 Resident budget: ~16.5 GB class (weights + SWA/KV class footprint as measured on target checkpoint) + documented headroom for activations, MTP state, runtime.
- R5.2 Gemma hybrid KV layout: SWA rings (window 1024) for local layers; full context only for global layers; honor unified global K=V if present — **layout, not quantization**.
- R5.3 Budgeted activation/attention scratch (no unbounded score tensors); FA/SDPA avoids materializing full score matrices; OOM-retry loops are failures, not steady state.
- R5.4 Load hygiene: streaming/per-layer release, no duplicate host copies of trellis when device owns weights, wired/working-set cap pattern (PonyExl3 92% cap as reference).
- R5.5 Memory metrics: peak RSS / footprint captured in bench runs.

### R6 — Persistence interface (owned by other spec)

- R6.1 This spec **consumes** `spec/paged-kv-cache` for prompt/turn KV durability and delta prefill; it does not redefine on-disk layout.
- R6.2 Engine must expose the restore hook (prefill from L) and turn-complete persist hook required by that spec.

### R7 — Measurement

- R7.1 Bench harness reports: TTFT (cold/warm, turn1 vs turn2+), pp/s at 1k/2k/8k, tg/s plain and tg/s + each drafter, peak memory, accept rate, prefill tokens actually computed.
- R7.2 Every optimization task names its target metric (TTFT | tg | pp | mem) and records before/after numbers in `implementation_notes.md`.
- R7.3 Reference comparison: **PonyExl3 MLX runtime** on same artifact (stock omlx has no `exl3` loader in 8a6bfb84 — verified); roofline sanity check.

### R8 — Speculative quality contract

- R8.1 temp 0: output token-identical to non-spec decode on same stack.
- R8.2 temp > 0: distribution-exact (Leviathan-Chen / equivalent) or speculation disabled.
- R8.3 Drafter and target quant pairing documented; no cross-family drafters unless acceptance measured and gated.

## Acceptance criteria

- AC1: Single-tenant mode on: steady-state decode shows no per-step admin/fairness/registry work (profiled).
- AC2: pp/s improvement vs baseline omlx on same weights ≥ agreed target (OQ2) with R2.1–R2.6 landed; numbers in notes.
- AC3: Warm turn-2+ TTFT with paged-kv-cache spec = prefill(delta) only (depends on that spec's AC2).
- AC4: Plain tg/s within X% of roofline estimate (OQ2), measured, not estimated.
- AC5: tg/s with MTP ≥ baseline × agreed factor (OQ2) and R8.1 holds on a golden suite.
- AC6: EXL3 3.0 bpw passes OQ1 quality gate; resident memory ≤ 16.5 GB + stated overhead budget.
- AC7: No KV quantization flags enabled anywhere in default config.
- AC8: No mid-decode unconditional sync/clear in default config (profile or code assertion).
- AC9: Existing omlx regression suite passes with single-tenant flag on and off.
- AC10: End-to-end multi-turn chat script: correct streaming, tool-call path smoke, persistence hooks fire (integration with paged-kv-cache).
- AC11: implementation_notes.md contains decisions, divergences, bench commands, and a takeover guide.

## Open questions

All closed (interview 2026-09-22):

| ID | Resolution |
|----|------------|
| OQ1 | **Sampled golden review**: golden prompts at **temp=1.0, topk=64, top_p=0.95**; pass/fail by reviewing sampled outputs vs reference quant (fluency/behavioral spot-check), multi-seed optional |
| OQ2 | **Report-only v1**: every optimization reports before/after numbers; no hard pass/fail speed thresholds |
| OQ3 | **24 GB** unified memory — fold16 banned for main weights; tight KV/activation headroom; resident target ~16.5 GB + explicit overhead budget |
| OQ4 | Primary sampling mode = **temp 1.0, topk 64, top_p 0.95**; temp-0 greedy still used for spec-decode token-identity tests (R8.1) |
| OQ5 | **8k typical / 32k max** — benches at 1k/2k/8k; long-context spot check 32k |
| OQ6 | **Pin reference SHAs now and freeze** for this spec; record in notes; no mid-spec pulls |
| OQ7 | **Text-only** v1; no vision/mmproj |
| OQ8 | **Fresh EXL3 kernels inside omlx** — reimplement decode/load path from reference material; do **not** vendor/embed PonyExl3 engine code; PonyExl3 may still be used as an external **converter** CLI (record versions in notes) |
