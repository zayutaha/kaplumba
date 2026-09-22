# Tasks: omlx-exl3-engine

Update after **every** completed task: check box, commit hash, bench deltas, deviations. Never leave this file stale until work stops.

Legend: `[ ]` todo · `[x]` done · `[~]` in progress · `[!]` blocked/diverged

## Phase 0 — Instrument

### T0.1 — Bench harness
- [ ] CLI/script: TTFT (cold/warm, turn1/turn2), pp/s @1k/2k/8k, tg/s plain, peak mem, roofline calculator (15.5 GB → 17.6 tok/s)
- Files: `scripts/bench.py` (engine-agnostic backend: baseline = ponyexl3 runtime; later = omlx)
- Test: runs against baseline model; JSON output stable
- Target metric: R7
- Commit: `spec/omlx-exl3-engine: T0.1 bench harness`

### T0.2 — Baseline capture
- [ ] Record **PonyExl3 runtime** numbers + profile of per-step work in single-tenant candidate path
- Files: implementation_notes.md only (+ raw logs)
- Test: notes contain numbers and commands
- Commit: `spec/omlx-exl3-engine: T0.2 baselines`

### T0.3 — Confirm environment
- [x] Memory size (OQ3 → **24 GB**), model files in `~/.kaplumba/models` (EXL3 artifact present), mlx/mlx-lm versions, reference pin SHAs frozen (OQ6)
- Files: notes
- Commit: `spec/omlx-exl3-engine: T0.3 env pins`

## Phase 1 — Overhead strip

### T1.1 — Single-tenant step() strip
- [ ] Flag disables fairness gate, activity publish, decode-debt, per-step admin snapshot copies
- Files: scheduler step path, settings
- Test: flag off = old behavior; flag on = profiled absence; unit for flag matrix
- Target: AC1
- Commit: `spec/omlx-exl3-engine: T1.1 single-tenant step`

### T1.2 — Prefill sync/clear removal
- [ ] No unconditional sync+clear per chunk; deferred/pressure only; OOM policy documented
- Files: scheduler prefill, metal_sync call sites
- Test: prefill completes; no clear in trace (assert); stress prefill 8k
- Target: AC8, pp/TTFT
- Commit: `spec/omlx-exl3-engine: T1.2 prefill sync policy`

### T1.3 — Append-only outputs + tokenize once
- [ ] Remove O(n²) token list copy per token; preflight does not double-encode
- Files: response building, batched preflight/add_request
- Test: unit append semantics; long generation 4k tokens no quadratic time; tokens identical
- Target: tg, TTFT
- Commit: `spec/omlx-exl3-engine: T1.3 outputs/tokenize`

### T1.4 — Compiled fixed-shape decode
- [ ] mx.compile decode graph; KV length classes (align 256); warmup; no per-token rebuild
- Files: engine/model decode path
- Test: shape stability across tokens; tg/s before/after; correctness vs eager on golden prompt
- Target: AC4 groundwork, tg
- Commit: `spec/omlx-exl3-engine: T1.4 compiled decode`

### T1.5 — Fast sampler without clear tax
- [ ] Single-pass CDF sampling; reused buffers; RNG residency issue solved without 1024-token clears
- Files: utils/sampling or replacement
- Test: distribution smoke (temp 0 golden, temp>0 statistical); no mid-decode clear in long run
- Target: tg, AC8
- Commit: `spec/omlx-exl3-engine: T1.5 sampler`

## Phase 2 — paged-kv hooks

### T2.1 — Restore/prefill-from-L integration
- [ ] Engine honors paged-kv restore; external prefill only suffix
- Files: engine_core/scheduler prefill entry
- Test: integration with paged-kv fixtures; prefill token count
- Target: AC3
- Commit: `spec/omlx-exl3-engine: T2.1 restore hook`

### T2.2 — Turn-complete persist hook
- [ ] Callback/enqueue after EOS; does not block admission
- Files: turn completion path
- Test: AC7 joint test with paged-kv T6
- Commit: `spec/omlx-exl3-engine: T2.2 persist hook`

## Phase 3 — EXL3

### T3.1 — Convert Gemma 4 31B @ 3.0 bpw
- [ ] **Artifact already exists** in `~/.kaplumba/models/Gxma/Gxma` (EXL3 3.0 bpw, head 6, mcg, v0.0.37) — verify load + resident size + manifest/versions in notes
- Files: none in repo except notes/config path
- Test: load succeeds; resident size report
- Target: AC6 part 1
- Commit: `spec/omlx-exl3-engine: T3.1 convert`

### T3.2 — Quality gate (OQ1: sampled golden review)
- [ ] Golden prompts at **temp=1.0, topk=64, top_p=0.95** (multi-seed optional); review sampled outputs vs reference quant; pass/fail recorded
- Test: the gate script itself
- Target: AC6
- Commit: `spec/omlx-exl3-engine: T3.2 quality gate`

### T3.3 — Fresh EXL3 decode kernels in omlx (OQ8)
- [ ] Reimplement trellis GEMV + EXL3 load/dequant inside omlx from reference material (no vendored PonyExl3 engine code); converter CLI may be external
- Files: model loading + linear swap / kernel modules
- Test: golden quality gate path; tg/s + mem (24 GB — no fold16 on main weights)
- Target: AC4, AC6
- Commit: `spec/omlx-exl3-engine: T3.3 exl3 fresh kernels`

### T3.4 — M4 Pro fuse/split-K re-sweep
- [ ] Thresholds measured on this GPU; recorded in notes
- Test: tg/s not worse than unswept defaults
- Commit: `spec/omlx-exl3-engine: T3.4 m4 sweeps`

## Phase 4 — Prefill kernels

### T4.1 — Dual-path matmul + int8 staging
- [ ] seq==1 GEMV vs seq≥T staged GEMM path
- Files: kernel/linear layer
- Test: numerical closeness vs baseline matmul; pp/s before/after @512/2k/8k
- Target: AC2
- Commit: `spec/omlx-exl3-engine: T4.1 dual-path matmul`

### T4.2 — Multi-row packed weights at load
- [ ] Interleaved packing applied on load; metadata layout rules documented
- Test: bit-exact or tolerance vs unpacked; pp/s delta
- Commit: `spec/omlx-exl3-engine: T4.2 row packing`

### T4.3 — Shape dispatch
- [ ] Dispatch table by (M,N,K); pathological shapes off generic kernel
- Test: per-shape microbench grid; no correctness regress
- Commit: `spec/omlx-exl3-engine: T4.3 shape dispatch`

### T4.4 — SWA KV bound in attention
- [ ] Sliding layers only iterate in-window KV (n_kv_max)
- Test: outputs match unbound path; long-context pp/tg delta
- Target: AC2, R2.6
- Commit: `spec/omlx-exl3-engine: T4.4 swa bound`

### T4.5 — Elementwise fusion audit (up/gate, silu⊙up)
- [ ] No accidental .eval() between fused candidates; optional fused kernels
- Test: pp/tg small delta; numerical parity
- Commit: `spec/omlx-exl3-engine: T4.5 fusion audit`

## Phase 5 — Decode amplification

### T5.1 — Sibling fusion (qkv, gated gate+up)
- [ ] Topology-guarded fusion; thresholds measured (not copied from M5 Max)
- Test: tg/s up or neutral; wrong-fusion guard unit test; golden outputs
- Target: AC4
- Commit: `spec/omlx-exl3-engine: T5.1 sibling fusion`

### T5.2 — MTP shared-KV integration
- [ ] Drafter on shared target KV; batched verify; depth flag; disable path
- Files: mtp patches, scheduler verify
- Test: R8.1 golden token-identity; accept rate logged; tg/s × factor
- Target: AC5
- Commit: `spec/omlx-exl3-engine: T5.2 mtp shared kv`

### T5.3 — MTP head memory (tie/requant if needed)
- [ ] Duplicate vocab projection handled per R3.4 if measurements require
- Test: memory delta; verify quality unchanged
- Commit: `spec/omlx-exl3-engine: T5.3 mtp head mem`

### T5.4 — Verify-batch GEMM/GEMV rows 2–8
- [ ] Small-batch specialized path for MTP verify
- Test: microbench + golden
- Commit: `spec/omlx-exl3-engine: T5.4 small-batch kernels`

### T5.5 — n-gram lookup drafter (optional)
- [ ] Host lookup drafting, verify-gated
- Test: never slower than plain on open-ended set; wins on repetitive set; identity at temp 0
- Commit: `spec/omlx-exl3-engine: T5.5 ngram`

### T5.6 — Spec quality suite
- [ ] Golden suite temp 0 + sampling distribution tests (R8)
- Test: the suite
- Commit: `spec/omlx-exl3-engine: T5.6 spec quality`

## Phase 6 — Memory + acceptance

### T6.1 — Hybrid KV layout audit (gemma4)
- [ ] SWA rings + global full; K=V honor; aligned with paged-kv T9
- Test: save/load/cold equivalence; footprint numbers
- Target: AC6 memory, R5.2
- Commit: `spec/omlx-exl3-engine: T6.1 hybrid kv`

### T6.2 — Activation budget + no OOM-retry steady state
- [ ] Scratch caps; long prefill without allocator thrash
- Test: 8k/32k context bench (per OQ5) clean
- Commit: `spec/omlx-exl3-engine: T6.2 activation budget`

### T6.3 — Full acceptance run AC1–AC11
- [ ] Execute all; fill notes; fix gaps or waive with user
- Commit: `spec/omlx-exl3-engine: T6.3 acceptance`

### T6.4 — Handover freeze
- [ ] implementation_notes complete for takeover
- Commit: `spec/omlx-exl3-engine: T6.4 close`
