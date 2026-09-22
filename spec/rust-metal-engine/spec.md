# Spec: pure-Rust + raw-Metal single-user Gemma 4 31B engine (EXL3 3.0 bpw)

## Goal

Deliver a zero-overhead, single-user chat inference engine written in **pure Rust with raw Metal kernels** (no MLX, no Python runtime), serving **Gemma 4 31B-class dense** weights quantized with **EXL3 ~3.0 bpw (~15.5 GB) + ~1 GB SWA/KV ≈ 16.5 GB resident** on an **M4 Pro (20 GPU cores, 273 GB/s)**, optimizing in priority order **TTFT, tg/s, pp/s, memory**, with **no quality loss** and **no hot-path overhead margin**. v1 scope is the engine core only — load → prefill → decode → sample — benchable via CLI; the HTTP server is out of scope. Techniques are inspired by `reference/{llama.cpp, ik_llama.cpp, vllm, PonyExl3}` and reimplemented fresh; persistence / delta prefill is specified in `spec/paged-kv-cache/` (rewritten for this engine).

Supersedes `spec/omlx-exl3-engine/` (Python/omlx plan abandoned per user directive 2026-09-22: no Python anywhere in the project).

## Non-goals / restrictions

- **No Python anywhere** — no Python in product/engine code. PonyExl3's Python converter CLI remains allowed as an **offline, build-time** tool only (DE19). `scripts/bench.py` is a temporary measurement harness until T0.2 retires it (DE20).
- **No MLX** — raw Metal kernels written in Rust; MLX C++ API is not linked either (confirmed: "raw metal kernels").
- **Single-user chat only** — no multi-tenant scheduler tax on the hot path, no batching for its own sake.
- **No KV cache quantization** (unless a future spec amends this).
- **No quality loss** — speculative decoding verify-gated (greedy token-identical at temp 0; distribution-exact sampling at temp > 0). EXL3 3.0 bpw must pass the sampled golden review.
- **No overhead margin** — profile before adding machinery; no scheduler/sync/serialization on the steady-state path.
- Do not treat `reference/` as a runtime dependency tree; port ideas deliberately as fresh code (same stance as old DE13: inspiration OK, no vendored engine code).
- No multi-model pool; single resident model. Model weights only from `~/.kaplumba/models`.
- `reference/` is read-only.
- HTTP server, tool-call orchestration, multi-tenant admin: **out of scope for v1** (may be other languages later; engine exposes a CLI/library API).

## Target metrics (hardware truths)

- Decode roofline: 273 GB/s ÷ 15.5 GB ≈ **~17.6 tok/s** plain (weights-dominated); realistic plain after KV/norms lower.
- User-stated perf targets: **~200 t/s prefill, ~18 t/s generation**; resident footprint goal **14.6 GB** (user: "get it to 14.6 gigs").
- Prefill is compute/dequant-bound on 20 GPU cores — kernel strategy dominates.
- Any tg/s claim above roofline without speculation is invalid.

## Requirements

### R1 — Architecture and single-tenant mode

- R1.1 Engine core is **pure Rust**; all GPU work is **raw Metal** (shaders + Rust↔Metal bindings). No MLX, no Python in the engine process.
- R1.2 Single-tenant by construction: no multi-tenant fairness gates, activity registry, decode-debt, or per-step admin snapshots exist on any path.
- R1.3 Decode uses a stable, fixed-shape hot path: padded KV length classes, warmup, no per-token graph/command-buffer rebuild (Metal analog of llama.cpp graph reuse).
- R1.4 Per-token CPU work is bounded: append-only output bookkeeping (no O(n²) copies), single-pass CDF sampling with buffer reuse, tokenize once per request.
- R1.5 Tokenizer is an in-engine Rust crate (`tokenizers` or equivalent) — self-contained for CLI benching. HTTP layer may call the engine out-of-process later; not a v1 concern.

### R2 — Prefill (pp/s and TTFT)

- R2.1 Dual-path matmul: `seq==1` → dedicated EXL3 trellis GEMV; `seq ≥ threshold` → dequant/stage weights to fast memory, GEMM across token columns (ik amortize-unpack principle).
- R2.2 Multi-row interleaved weight packing at load (4/8-row scale sharing; Apple-friendly metadata).
- R2.3 Shape-dispatched GEMM selection (never one generic kernel at pathological aspect ratios).
- R2.4 No mid-prefill forced full GPU sync + cache clear per chunk; idle/deferred/pressure clears only.
- R2.5 Gemma SWA-aware attention: bound KV loop for sliding layers (`n_kv_max ≈ pos - window`); avoid paying full-context cost on 5/6 local layers.
- R2.6 TTFT path: no queueing behind decode bursts a single user did not ask for; warmup strategy defined (first-token latency vs shader compile).

### R3 — Decode (tg/s)

- R3.1 Sibling fusion of same-input linears with per-group size threshold and topology guards (qkv; gate+up only above threshold — PonyExl3 lessons: wrong fusion = ~2× regression).
- R3.2 MTP speculative decode using Gemma 4 built-in drafter with **shared target KV** (no second context); verify batched in one step; depth configurable.
- R3.3 Optional n-gram/lookup drafting (host-only, verify-gated) as cheap secondary lever.
- R3.4 MTP output-head memory: reuse/tie or re-quantize draft head if it duplicates full vocab matrix — without quality loss on verify path.
- R3.5 GEMV kernels specialized for batch 1 and small verify batches (rows 2–8) reaching bandwidth roofline (llama.cpp mul_mv family + PonyExl3 simdgroup GEMV as structural references).
- R3.6 Sampling fast path (R1.4) without RNG residency bugs that force periodic mid-decode clears.
- R3.7 No mid-decode forced full GPU sync/clear except deferred idle or hard memory pressure (document thresholds).

### R4 — EXL3 3.0 bpw path

- R4.1 Artifact: Gemma 4 31B EXL3 checkpoint at **~3.0 bpw** in `~/.kaplumba/models` (already present — verify only; external PonyExl3 Python converter CLI allowed for any future reconvert; pin versions).
- R4.2 Quality gate: **sampled golden review** at temp=1.0, topk=64, top_p=0.95 vs reference quant must pass before default.
- R4.3 Runtime: **fresh EXL3 trellis GEMV + load/dequant written in Rust/Metal from reference material**; do not vendor/embed PonyExl3 engine code. On 24 GB, fold16-for-weights is banned.
- R4.4 Re-sweep M4 Pro–specific tunables (split-K threadgroup target, fuse thresholds) — do not copy M5 Max numbers blindly.
- R4.5 Converter/runtime version pins recorded; lossy engines (`fold16`/`w4a16`) are not defaults.

### R5 — Memory

- R5.1 Resident budget: ~16.5 GB class (weights + SWA/KV) + documented headroom for activations, MTP state, runtime; stretch goal **14.6 GB** footprint as user-stated target (report delta; may require tighter packing or different accounting — open).
- R5.2 Gemma hybrid KV layout: SWA rings (window 1024) for local layers; full context only for global layers; honor unified global K=V — **layout, not quantization**.
- R5.3 Budgeted activation/attention scratch (no unbounded score tensors); FA/SDPA avoids materializing full score matrices; OOM-retry loops are failures, not steady state.
- R5.4 Load hygiene: streaming/per-layer release, no duplicate host copies of trellis when device owns weights; Metal working-set cap pattern (PonyExl3 92% cap as reference).
- R5.5 Memory metrics: peak RSS / footprint / Metal peak captured in bench runs.

### R6 — Persistence interface (owned by other spec)

- R6.1 This spec **consumes** `spec/paged-kv-cache` (rewritten for Rust) for prompt/turn KV durability and delta prefill; it does not redefine on-disk layout.
- R6.2 Engine must expose the restore hook (prefill from L) and turn-complete persist hook required by that spec.

### R7 — Measurement

- R7.1 Bench harness reports: TTFT (cold/warm, turn1 vs turn2+), pp/s at 1k/2k/8k, tg/s plain and tg/s + each drafter, peak memory, accept rate, prefill tokens actually computed.
- R7.2 Every optimization task names its target metric (TTFT | tg | pp | mem) and records before/after numbers in `implementation_notes.md`.
- R7.3 Reference comparison: **PonyExl3 MLX runtime** baseline numbers (captured in T0.2 via `scripts/bench.py` / `ponyexl3-generate-bench`) are **reference-only** — they measure PonyExl3, not this engine (DE20). Roofline sanity check mandatory.
- R7.4 After T0.2, retire Python `scripts/bench.py`; the Rust engine ships its own CLI bench emitting the same JSON schema (`kaplumba-bench/1`) for diffability.

### R8 — Speculative quality contract

- R8.1 temp 0: output token-identical to non-spec decode on same stack.
- R8.2 temp > 0: distribution-exact (Leviathan-Chen / equivalent) or speculation disabled.
- R8.3 Drafter and target quant pairing documented; no cross-family drafters unless acceptance measured and gated.

## Acceptance criteria

- AC1: Engine loads EXL3 artifact and runs greedy generation end-to-end from CLI; outputs match quality gate expectations.
- AC2: pp/s improvement vs PonyExl3 reference baseline on same artifact, numbers in notes (report-only thresholds — OQ2).
- AC3: Warm turn-2+ TTFT with paged-kv-cache = prefill(delta) only (depends on that spec's AC2).
- AC4: Plain tg/s within X% of roofline estimate, measured, not estimated (report-only — OQ2).
- AC5: tg/s with MTP ≥ reference baseline × agreed factor and R8.1 holds on a golden suite.
- AC6: EXL3 3.0 bpw passes OQ1 quality gate; resident memory ≤ 16.5 GB + stated overhead budget (target-track toward 14.6 GB).
- AC7: No KV quantization flags enabled anywhere in default config.
- AC8: No mid-decode unconditional sync/clear in default config (profile or code assertion).
- AC9: Rust test suite passes (`cargo test`) — unit + integration; no Python tests required for engine correctness.
- AC10: End-to-end multi-turn script (CLI): correct streaming, persistence hooks fire (integration with paged-kv-cache).
- AC11: implementation_notes.md contains decisions, divergences, bench commands, and a takeover guide.

## Open questions

All closed for the omlx plan (interview 2026-09-22) carry forward where still applicable; new Rust/Metal questions closed in the pivot interview (same date):

| ID | Resolution |
|----|------------|
| OQ1 | **Sampled golden review**: golden prompts at **temp=1.0, topk=64, top_p=0.95**; pass/fail by reviewing sampled outputs vs reference quant |
| OQ2 | **Report-only v1**: every optimization reports before/after numbers; no hard pass/fail speed thresholds |
| OQ3 | **24 GB** unified memory — fold16 banned for main weights; resident target ~16.5 GB + explicit overhead budget (stretch 14.6 GB) |
| OQ4 | Primary sampling mode = **temp 1.0, topk 64, top_p 0.95**; temp-0 greedy for spec-decode identity tests (R8.1) |
| OQ5 | **8k typical / 32k max** — benches at 1k/2k/8k; long-context spot check 32k |
| OQ6 | **Pin reference SHAs now and freeze** for this spec; record in notes; no mid-spec pulls |
| OQ7 | **Text-only** v1; no vision/mmproj |
| OQ8 | **Fresh EXL3 kernels in Rust/Metal** — reimplement decode/load from reference material; do **not** vendor PonyExl3 engine code; PonyExl3 Python converter OK offline |
| OQ9 (new) | **No Python at all** (user 2026-09-22): engine pure Rust + raw Metal; converter CLI is the only allowed Python, offline only; `scripts/bench.py` temporary until T0.2 |
| OQ10 (new) | **Tokenizer**: in-engine Rust crate (self-contained CLI); HTTP/out-of-process tokenizer deferred with HTTP server (out of scope v1) |
| OQ11 (new) | **Kernel provenance**: inspiration/port-from-reference allowed (ggml-metal, PonyExl3 gemv logic) as fresh implementations — same stance as DE13 |
