# Plan: pure-Rust + raw-Metal single-user Gemma 4 31B engine (EXL3 3.0 bpw)

## Position in the program

```
spec/rust-metal-engine     ← this: Rust engine core, raw Metal kernels, EXL3, MTP, memory, CLI bench
spec/paged-kv-cache        ← separate: ./prompt_cache, ./chat, delta prefill, persistence (rewritten for Rust)
spec/omlx-exl3-engine      ← superseded (Python/omlx abandoned; see its stub)
```

Order of work (high level): finish instrument (reference baseline) → Rust skeleton + naive path → EXL3 load/decode kernels → prefill kernels → decode amplification (fusion/MTP) → memory + persistence hooks → acceptance.

## Architecture

```
                 ┌─────────────────────────────────────────┐
  CLI / bench    │ pure-Rust engine core                    │
  args ────────► │  load EXL3 artifact (streaming)          │
                 │  restore hook ◄── paged-kv-cache (spec)  │
                 │  prefill delta │ full cold prefill       │
                 │  decode step (fixed-shape, warm)         │
                 │    ├─ GEMV path (M=1) / GEMM (verify)    │
                 │    ├─ MTP draft + verify (shared KV)     │
                 │    └─ fast sampler → append-only out     │
                 │  turn-complete → persist hook ───────────┼──► paged-kv-cache
                 │  tokenizer: in-engine Rust crate          │
                 └─────────────────────────────────────────┘
                              │
                     Raw Metal kernels (Rust bindings)
                     EXL3 trellis GEMV / staged int8 GEMM
                              │
                     weights: ~/.kaplumba/models (EXL3 3.0 bpw)
```

HTTP server, admin, multi-tenant anything: not in this diagram (out of scope v1). May live in another language later and call the engine as a library/subprocess.

## Stack choices (DE15–DE21)

| Decision | Choice | Notes |
|----------|--------|-------|
| Language | Rust (edition 2021+) | No Python in engine process |
| GPU API | Raw Metal via Rust bindings (e.g. `objc2-metal` / `metal-rs`-class crates — pick at T1.0 and pin) | No MLX, no Vulkan abstraction layer |
| Shaders | Metal Shading Language (`.metal`) compiled at build or first-run with documented cache path | Structure inspired by ggml-metal / PonyExl3 gemv — fresh code |
| Tokenizer | In-engine Rust crate (`tokenizers` HF-compatible or equivalent supporting Gemma 4) | Self-contained CLI; HTTP tokenizer deferred |
| Converter | PonyExl3 Python CLI, offline only | Already-done artifact verified in T3.1 |
| Bench (transition) | `scripts/bench.py` until T0.2, then Rust CLI bench, same JSON schema `kaplumba-bench/1` | Python harness is DE20 exception, time-boxed |
| Persistence | Owned by `spec/paged-kv-cache` (Rust rewrite) | Engine exposes restore/persist hooks only |

## Baseline facts (researched; T0.2 captures numbers)

| Source | Takeaway | Use here |
|--------|----------|----------|
| llama.cpp | GEMV vs GEMM dispatch, graph reuse, shared-KV MTP, LCP prompt cache, SWA mask bound, small-batch GEMV family | Port structure to Rust/Metal |
| ik_llama.cpp | Prefill = amortize dequant across columns; R4/R8 interleave; dual-path by batch; shape dispatch; fused up/gate; FA only if internal GEMMs stay fast | Prefill strategy → Metal |
| PonyExl3 | EXL3 codec+converter; sibling fusion; split-K; serial dispatch = decode wall; LUT GEMV worse; fuse threshold per group | EXL3 path + fusion lessons (fresh impl) |
| vLLM | Algorithmic only: prefix/radix idea, EAGLE/MTP verify patterns, compile-the-step | Ideas folded into R2/R3/R6 |
| omlx (historical) | Multi-tenant overhead lessons; skip_lm_head idea; MTP gemma4 scaffolding ideas only | Design notes only — **not a dependency** |

## Phases → tasks (tasks.md is the live checklist)

### Phase 0 — Instrument (T0.x)
- T0.1: Bench harness (exists: `scripts/bench.py`, Python — temporary).
- T0.2: Baseline numbers from **PonyExl3 MLX runtime** on target artifact; profile reference per-step work; record as **reference-only** (DE20). Then retire Python harness in favor of Rust CLI bench (R7.4).
- T0.3: Env pins, reference SHAs frozen (done — carry notes forward).

### Phase 1 — Rust skeleton (T1.x) → AC1 groundwork
- Cargo workspace; Metal device/queue init; EXL3 artifact loader (streaming, no duplicate host copies); tokenizer wire-up; naive eager prefill/decode (correctness before speed); CLI generate; golden greedy smoke test.

### Phase 2 — EXL3 decode path (T2.x) → AC4, R3.5, R4.*
- Fresh trellis GEMV + load/dequant in Metal; working-set cap; M4 Pro sweep (split-K, fuse thresholds); approach roofline; quality gate OQ1.

### Phase 3 — Prefill kernels (T3.x) → AC2, R2.*
- Dual-path matmul + int8 staging + multi-row packing; shape dispatch; SWA n_kv bound; remove mid-prefill sync/clear; fusion audit silu⊙up.

### Phase 4 — Decode amplification (T4.x) → AC4, AC5, R3.*
- Sibling fusion with measured thresholds; MTP shared-KV + verify; GEMV specialization M=1 and rows 2–8; golden suites R8.1/R8.2; optional n-gram.

### Phase 5 — Memory + persistence hooks (T5.x) → AC3, AC6, R5.*, R6.*
- Hybrid KV layout (SWA rings + global full); activation budget; load hygiene / resident report; restore + persist hook integration with `spec/paged-kv-cache`.

### Phase 6 — Acceptance (T6.x) → AC9–AC11
- Full AC run; Rust test suite green; notes freeze; handover.

## Risk register

| Risk | Fallback |
|------|----------|
| EXL3 3.0 bpw fails sampled golden review | Bit-allocation tweaks; 3.2–3.5 bpw or affine 4-bit while keeping runtime work |
| MTP acceptance too low on EXL3 target | Disable MTP by default; keep n-gram; report accept rates |
| Raw Metal binding/crate friction (objc2, shader build) | Pick well-maintained crate at T1.0; pin; document shader cache path |
| Roofline unreachable due to kernel ALU | Profile EXL3 decode; staged paths for heavy layers; document ceiling |
| Rewriting everything from scratch takes longer than omlx patch would have | Phases are small and independently verifiable; naive-path first (T1) so correctness is proven before kernel work |
| Warm/cold greedy mismatch | OQ1 policy + paged-kv R9.2 |
| `scripts/bench.py` Python exception lingers past T0.2 | Hard cutoff: T0.2 done → delete or move to `tools/legacy/`; R7.4 Rust bench replaces |

## Requirement → phase map

| Req | Phase/tasks |
|-----|-------------|
| R1 architecture/single-tenant/tokenizer | P1 |
| R2 prefill/TTFT kernels | P3 (+P1 naive, sync policy) |
| R3 decode/MTP/GEMV | P2 GEMV, P4 amplification |
| R4 EXL3 | P2 (+ P0 verify artifact) |
| R5 memory | P2 load, P5 layout/budget |
| R6 persistence hooks | P5 (with paged-kv spec) |
| R7 measurement | P0 + every task notes; R7.4 Rust bench after T0.2 |
| R8 spec quality | P4 golden tests |

## Interface contracts (for implementers)

**Restore hook (from paged-kv-cache):** engine accepts `prefill_from(L)` / restore handles already applied before external prefill runs suffix.

**Persist hook:** after stop/EOS, engine yields block table + token log tail to async persist (never blocks admission).

**Bench API:** stable CLI returning JSON lines: `{ttft_ms, pp_tps, tg_tps, peak_mem_gb, accept_rate, prefill_tokens}` for diffing across commits — schema `kaplumba-bench/1` (match Python harness during transition).

**Config:** quant/engine selection, `mtp_depth`, `spec_enabled`, data_root for persistence; no KV-quant knobs in defaults.

## Sequencing dependencies

- T0.2 before deleting Python bench; P1 can start in parallel with T0.2 (different processes — do not run both full-model benches at once on 24 GB).
- P2 depends on P1 loader correctness.
- P3–P4 depend on P2 GEMV existing (prefill reuses dequant; decode fusion reuses linear abstractions).
- P5 persistence hooks depend on paged-kv-cache T1–T5 (other spec).
- P6 last.
