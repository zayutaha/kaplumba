# Plan: omlx-based single-user Gemma 4 31B engine (EXL3 3.0 bpw)

## Position in the program

```
spec/omlx-exl3-engine     ← this: kernels, single-tenant strip, EXL3, MTP, memory, benches
spec/paged-kv-cache       ← separate: ./prompt_cache, ./chat, delta prefill, persistence
```

Order of work (high level): instrument → strip overhead → paged-kv (other spec) → EXL3 convert/quality → prefill kernels → MTP/spec → memory polish → acceptance.

## Architecture

```
                 ┌─────────────────────────────────────────┐
 HTTP/SSE        │ single-tenant engine loop                │
 client ───────► │  admission (no multi-tenant gates)       │
                 │  restore hook ◄── paged-kv-cache (spec)  │
                 │  prefill delta │ full cold prefill       │
                 │  compiled decode step (mx.compile)       │
                 │    ├─ GEMV path (M=1) / GEMM (verify)    │
                 │    ├─ MTP draft + verify (shared KV)     │
                 │    └─ fast sampler → append-only out     │
                 │  turn-complete → persist hook ───────────┼──► paged-kv-cache
                 └─────────────────────────────────────────┘
                              │
                    MLX Metal kernels
                    EXL3 trellis GEMV / staged int8 GEMM
                              │
                    weights: ~/.kaplumba/models (EXL3 3.0 bpw)
```

## Baseline facts (researched; re-verify on hardware in T0)

| Source | Takeaway | Use here |
|--------|----------|----------|
| omlx | Decode burst, external prefill, skip_lm_head, MTP gemma4, boundary snapshots exist; multi-tenant gates + per-chunk sync + O(n²) outputs + uncompiled sampler hurt | Strip/fix, don't rebuild |
| llama.cpp | GEMV vs GEMM dispatch, graph reuse, shared-KV MTP, LCP prompt cache, SWA mask bound, small-batch GEMV family | Port structure to MLX/Metal |
| ik_llama.cpp | Prefill = amortize dequant across columns; R4/R8 interleave; dual-path by batch; shape dispatch; fused up/gate; FA only if internal GEMMs stay fast | Prefill strategy (CPU ideas → Metal) |
| PonyExl3 | EXL3 codec+converter on Metal; sibling fusion; split-K; serial dispatch = decode wall; LUT GEMV worse; fuse threshold per group | EXL3 path + fusion lessons |
| vLLM | Algorithmic only: prefix/radix idea, EAGLE/MTP verify patterns, compile-the-step | Ideas already folded into R2/R3/R6 |

## Phases → tasks (tasks.md is the live checklist)

### Phase 0 — Instrument (T0.x)
- Bench harness on M4 Pro (24 GB): TTFT, pp 1k/2k/8k, tg plain, mem, roofline printout (15.5 GB → 17.6 tok/s).
- Baseline numbers from **PonyExl3 MLX runtime** (`ponyexl3-generate-bench`, pinned SHA) on the target artifact — stock omlx cannot load EXL3 (verified no `exl3` in tree). Compare our engine against this + roofline.
- Profile single-tenant gaps (what fires per `step()` today).
- Pin + freeze reference SHAs (OQ6) in notes.

### Phase 1 — Overhead strip (T1.x) → AC1, AC8, parts of AC3
- Single-tenant flags: skip fairness/activity/decode-debt/admin snapshot per step.
- Remove/gate mid-prefill sync+clear; deferred idle clears only.
- Append-only outputs; tokenize-once; fixed-shape decode graph + warmup.
- Sampler: single-pass CDF without periodic clear forcing.

### Phase 2 — paged-kv-cache integration (T2.x) → depends on other spec
- Implement restore length L prefill + turn persist hooks (R6).
- Turn-2 TTFT = delta only (other spec AC2 + this AC3).

### Phase 3 — EXL3 (T3.x) → AC6, feeds all benches
- Convert Gemma 4 31B @ 3.0 bpw → `~/.kaplumba/models` (external PonyExl3 converter CLI OK; versions pinned in notes).
- Sampled golden gate (OQ1): temp=1.0, topk=64, top_p=0.95 vs reference quant before default-on.
- Load hygiene + resident memory report (R5.4, R5.1) on **24 GB** machine — no fold16 on main weights (OQ3).
- M4 Pro re-sweep: split-K, fuse thresholds (R4.4).
- **Fresh EXL3 kernels in omlx** (OQ8): reimplement trellis decode/load; do not vendor PonyExl3 engine.

### Phase 4 — Prefill kernels (T4.x) → AC2, R2.*
- Dual-path matmul + int8 staging + multi-row packing (R2.1–R2.2).
- Shape dispatch (R2.3); SWA n_kv bound (R2.6); fusion audit silu⊙up (R2 as applicable).
- FA/SDPA path check: fused attention must not regress internal matmul quality.

### Phase 5 — Decode amplification (T5.x) → AC4, AC5, R3.*
- Sibling fusion with measured thresholds (R3.1).
- MTP shared-KV wire-up + verify kernel + depth tune (R3.2); n-gram optional (R3.3).
- GEMV specialization M=1 and verify rows 2–8 (R3.5).
- Golden suites for R8.1/R8.2.

### Phase 6 — Memory + acceptance (T6.x) → AC6, AC7, AC9–AC11
- Hybrid KV layout audit (R5.2) coordinated with paged-kv T9.
- Full AC run; notes freeze; handover.

## Risk register

| Risk | Fallback |
|------|----------|
| EXL3 3.0 bpw fails sampled golden review (temp 1.0 / topk 64 / topp 0.95) | Try measured bit allocation tweaks; fall back to 3.2–3.5 bpw or affine 4-bit while keeping runtime work |
| MTP acceptance too low on EXL3 target | Disable MTP by default; keep n-gram; report accept rates |
| mx.compile shape churn kills perf | Pad to KV length classes (e.g. 256 buckets aligned with paged blocks) |
| Patches to omlx too invasive | Wrapper layer + minimal diffs; pin SHAs in notes |
| Roofline unreachable due to kernel ALU | Profile EXL3 decode; route heavy layers through staged paths; document ceiling |
| Warm/cold greedy mismatch | OQ1 policy + paged-kv R9.2 |

## Requirement → phase map

| Req | Phase/tasks |
|-----|-------------|
| R1 single-tenant + compile + per-token CPU | P1 |
| R2 prefill/TTFT kernels | P4 (+P1 sync removal) |
| R3 decode/MTP/GEMV | P5 |
| R4 EXL3 | P3 |
| R5 memory | P3 load, P6 layout/budget |
| R6 persistence hooks | P2 |
| R7 measurement | P0 + every task notes |
| R8 spec quality | P5 golden tests |

## Interface contracts (for implementers)

**Restore hook (from paged-kv-cache):** engine accepts `prefill_from(L)` / restore handles already applied before external prefill runs suffix.

**Persist hook:** after stop/EOS, engine yields block table + token log tail to async persist (never blocks admission).

**Bench API:** stable CLI returning JSON lines: `{ttft_ms, pp_tps, tg_tps, peak_mem_gb, accept_rate, prefill_tokens}` for diffing across commits.

**Config:** `single_tenant=True`, `data_root`, quant/engine selection, `mtp_depth`, `spec_enabled`, no KV-quant knobs in defaults.
