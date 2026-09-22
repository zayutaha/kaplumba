# Tasks: rust-metal-engine

Update after **every** completed task: check box, commit hash, bench deltas, deviations. Never leave this file stale until work stops.

Legend: `[ ]` todo · `[x]` done · `[~]` in progress · `[!]` blocked/diverged

## Phase 0 — Instrument

### T0.1 — Bench harness (transition, Python)
- [x] CLI/script: TTFT (cold/warm, turn1/turn2), pp/s @1k/2k/8k, tg/s plain, peak mem, roofline calculator (15.5 GB → 17.6 tok/s) — `scripts/bench.py` written, mock JSON verified
- Files: `scripts/bench.py` (engine-agnostic backend: baseline = ponyexl3; later = rust)
- Test: runs against baseline model; JSON output stable
- Target metric: R7
- Note: Python is a **time-boxed exception** (DE20) — dies at T0.2/R7.4
- Commit: `spec/rust-metal-engine: T0.1 bench harness` (was drafted under omlx-exl3-engine name; re-tagged)

### T0.2 — Baseline capture (reference-only)
- [ ] Record **PonyExl3 runtime** numbers + profile of per-step work; label all numbers **reference-only** (measures PonyExl3, not our engine)
- Files: implementation_notes.md only (+ raw logs under /tmp or notes)
- Test: notes contain numbers, commands, hardware context
- Commit: `spec/rust-metal-engine: T0.2 baselines`
- Follow-up in same task or T0.2b: implement/land Rust CLI bench emitting `kaplumba-bench/1`; retire Python harness (R7.4)

### T0.3 — Confirm environment
- [x] Memory size (24 GB), model files in `~/.kaplumba/models` (EXL3 artifact present), reference pin SHAs frozen — carried from omlx-exl3-engine T0.3
- Files: notes
- Commit: `spec/omlx-exl3-engine: T0.3 env pins` (historical; content migrated here)

## Phase 1 — Rust skeleton

### T1.0 — Workspace + Metal binding choice
- [x] Cargo workspace layout; Metal binding crate pinned: **`metal` 0.33.0** (gfx-rs; API checked against registry sources); shader path T1.0 = runtime `new_library_with_source` (offline `.metal`→`.metallib` via `xcrun metal` deferred to T2 when kernels stabilize); documented in notes
- Files: root `Cargo.toml` (workspace), `crates/engine`, `crates/cli`, `crates/engine/src/metal_ctx.rs`
- Test: `cargo build` OK; `cargo test -p kaplumba-engine` (2 pass); `cargo run -p kaplumba-cli -- metal-smoke` → OK on M4 Pro
- Commit: `spec/rust-metal-engine: T1.0 workspace + metal bindings`

### T1.1 — EXL3 artifact loader
- [x] Stream `~/.kaplumba/models/Gxma/Gxma` shards into Metal buffers (per-shard shared allocs; single ~16 GiB buffer rejected by Metal → **2 shard buffers**); no duplicate host copy at steady state; quant_config parsed; resident size report
- Files: `crates/engine/src/load.rs`
- Test: `cargo test -p kaplumba-engine` 7/7; real artifact: **2422 tensors / 2 shards / 16.028 GB / ~4s**; corrupt/missing shard/config errors unit-tested
- Commit: `spec/rust-metal-engine: T1.1 exl3 loader`

### T1.2 — Tokenizer wire-up
- [x] In-engine Rust tokenizer loads Gemma 4 vocab from artifact `tokenizer.json`; encode/decode round-trip
- Files: `crates/engine/src/tokenizer.rs` (`GemmaTokenizer`, `tokenizers` 0.21.4)
- Test: `cargo test -p kaplumba-engine` 9/9; round-trip stable; specials pad=0 eos=1 bos=2 unk=3 `<|turn>`=105 `<turn|>`=106
- Commit: `spec/rust-metal-engine: T1.2 tokenizer`

### T1.3 — Naive eager prefill + decode (correctness first)
- [~] In progress — sub-steps:
  - [x] **T1.3a** EXL3 decode primitives: trellis pack/unpack, codebook (MCG/Default/Mul1), perm, signs, Hadamard, `decode_packed_tile` — 26/26 unit tests vs reference goldens (q00 tile, MCG codewords, pack fixtures k=3/6)
  - [x] **T1.3b** reconstruct (inner/slab/public weights) + streaming `linear_forward_reconstruct` — 44/44 tests; stream ≡ stepwise rel err < 1e-3; dense public-weight loose parity
  - [ ] **T1.3b** reconstruct (inner trellis → had+scales) + `linear_forward_reconstruct`
  - [ ] **T1.3c** Gemma4 model forward (embed, hybrid SWA/full attn, MLP, layer_scalar, softcap)
  - [ ] **T1.3d** greedy generate + golden fixture; deterministic across runs
- Files: `crates/engine/src/exl3/*` (a); model forward, attention, lm_head, sampler stub (b–d)
- Test: golden greedy output on fixed prompt matches known-good text (record fixture); deterministic across runs
- Commit: `spec/rust-metal-engine: T1.3 naive forward` (a committed as `...: T1.3a decode primitives`)

### T1.4 — CLI generate + bench entry
- [ ] `kaplumba-engine generate --prompt ... --max-tokens N --temp ...` streaming; JSON stats (ttft_ms, pp_tps, tg_tps, mem) schema `kaplumba-bench/1`
- Files: `crates/cli`
- Test: CLI runs; JSON schema matches harness during transition; diffable
- Commit: `spec/rust-metal-engine: T1.4 cli generate`

## Phase 2 — EXL3 decode path

### T2.1 — Fresh trellis GEMV + dequant kernels
- [ ] EXL3 trellis decode GEMV in Metal from reference logic (fresh code); out_scales/codebook path correct
- Files: `crates/kernels/src/exl3/*`
- Test: numerical closeness vs loader reference path; tg/s before/after; pageins=0 in steady decode
- Target: AC4, R3.5, R4.3
- Commit: `spec/rust-metal-engine: T2.1 trellis gemv`

### T2.2 — Working-set cap + load hygiene
- [ ] Metal allocation cap pattern (~92% recommended working set); streaming load; no trellis host/device duplication
- Files: allocator, loader
- Test: resident footprint report; no OOM on 24 GB during load+decode
- Target: R5.4, AC6
- Commit: `spec/rust-metal-engine: T2.2 working set cap`

### T2.3 — M4 Pro kernel sweep
- [ ] Split-K threadgroup targets, GEMV dispatch thresholds measured on this GPU; recorded in notes (not copied from M5 Max)
- Files: kernel constants / dispatch table
- Test: tg/s not worse than unswept defaults; numbers in notes
- Commit: `spec/rust-metal-engine: T2.3 m4 sweeps`

### T2.4 — Quality gate (OQ1)
- [ ] Golden prompts at **temp=1.0, topk=64, top_p=0.95** (multi-seed optional); review sampled outputs vs reference quant; pass/fail recorded
- Test: the gate script/CLI itself
- Target: AC6
- Commit: `spec/rust-metal-engine: T2.4 quality gate`

## Phase 3 — Prefill kernels

### T3.1 — Verify EXL3 artifact (T3.1 from old plan, renamed)
- [x] Artifact already exists — verify load + resident size + manifest/versions in notes (done under omlx-exl3 T0.3/T3.1 research)
- Commit: (historical notes)

### T3.2 — Dual-path matmul + int8 staging
- [ ] seq==1 GEMV vs seq≥T staged GEMM path; amortize dequant across columns
- Files: linear layer dispatch
- Test: numerical closeness vs naive path; pp/s before/after @512/2k/8k
- Target: AC2, R2.1
- Commit: `spec/rust-metal-engine: T3.2 dual-path matmul`

### T3.3 — Multi-row packed weights at load
- [ ] Interleaved packing applied on load; metadata layout rules documented
- Test: tolerance vs unpacked; pp/s delta
- Commit: `spec/rust-metal-engine: T3.3 row packing`

### T3.4 — Shape dispatch
- [ ] Dispatch table by (M,N,K); pathological shapes off generic kernel
- Test: per-shape microbench grid; no correctness regress
- Commit: `spec/rust-metal-engine: T3.4 shape dispatch`

### T3.5 — SWA KV bound in attention
- [ ] Sliding layers only iterate in-window KV (n_kv_max)
- Test: outputs match unbound path; long-context pp/tg delta
- Target: AC2, R2.5
- Commit: `spec/rust-metal-engine: T3.5 swa bound`

### T3.6 — Prefill sync policy + fusion audit
- [ ] No unconditional sync+clear per chunk (deferred/pressure only); no accidental materialize between fused candidates (up/gate, silu⊙up)
- Files: prefill loop, elementwise
- Test: prefill completes 8k; profile shows no mid-prefill full clear; numerical parity
- Target: AC8, R2.4
- Commit: `spec/rust-metal-engine: T3.6 sync policy + fusion`

## Phase 4 — Decode amplification

### T4.1 — Sibling fusion (qkv, gated gate+up)
- [ ] Topology-guarded fusion; thresholds measured on M4 Pro
- Test: tg/s up or neutral; wrong-fusion guard unit test; golden outputs
- Target: AC4, R3.1
- Commit: `spec/rust-metal-engine: T4.1 sibling fusion`

### T4.2 — Compiled/fixed-shape decode step
- [ ] Stable command-buffer / pipeline reuse; KV length classes (align 256); warmup; no per-token rebuild
- Files: decode path
- Test: shape stability across tokens; tg/s before/after; correctness vs eager on golden prompt
- Target: AC4 groundwork, R1.3
- Commit: `spec/rust-metal-engine: T4.2 fixed-shape decode`

### T4.3 — Fast sampler without clear tax
- [ ] Single-pass CDF sampling; reused buffers; RNG residency without mid-decode clears
- Files: sampling module
- Test: distribution smoke (temp 0 golden, temp>0 statistical); no mid-decode clear in long run
- Target: tg, AC8, R1.4, R3.6
- Commit: `spec/rust-metal-engine: T4.3 sampler`

### T4.4 — MTP shared-KV integration
- [ ] Drafter on shared target KV; batched verify; depth flag; disable path
- Files: mtp module, verify step
- Test: R8.1 golden token-identity; accept rate logged; tg/s × factor
- Target: AC5, R3.2
- Commit: `spec/rust-metal-engine: T4.4 mtp shared kv`

### T4.5 — MTP head memory (tie/requant if needed)
- [ ] Duplicate vocab projection handled per R3.4 if measurements require
- Test: memory delta; verify quality unchanged
- Commit: `spec/rust-metal-engine: T4.5 mtp head mem`

### T4.6 — Verify-batch GEMM/GEMV rows 2–8
- [ ] Small-batch specialized path for MTP verify
- Test: microbench + golden
- Commit: `spec/rust-metal-engine: T4.6 small-batch kernels`

### T4.7 — n-gram lookup drafter (optional)
- [ ] Host lookup drafting, verify-gated
- Test: never slower than plain on open-ended set; wins on repetitive set; identity at temp 0
- Commit: `spec/rust-metal-engine: T4.7 ngram`

### T4.8 — Spec quality suite
- [ ] Golden suite temp 0 + sampling distribution tests (R8)
- Test: the suite
- Commit: `spec/rust-metal-engine: T4.8 spec quality`

## Phase 5 — Memory + persistence hooks

### T5.1 — Hybrid KV layout (gemma4)
- [ ] SWA rings + global full; K=V honor; aligned with paged-kv T9
- Test: save/load/cold equivalence; footprint numbers
- Target: AC6 memory, R5.2
- Commit: `spec/rust-metal-engine: T5.1 hybrid kv`

### T5.2 — Activation budget + no OOM-retry steady state
- [ ] Scratch caps; long prefill without allocator thrash
- Test: 8k/32k context bench (per OQ5) clean
- Commit: `spec/rust-metal-engine: T5.2 activation budget`

### T5.3 — Restore/prefill-from-L integration
- [ ] Engine honors paged-kv restore; external prefill only suffix
- Files: prefill entry
- Test: integration with paged-kv fixtures; prefill token count
- Target: AC3
- Commit: `spec/rust-metal-engine: T5.3 restore hook`

### T5.4 — Turn-complete persist hook
- [ ] Callback/enqueue after EOS; does not block admission
- Files: turn completion path
- Test: AC10 joint test with paged-kv T6
- Commit: `spec/rust-metal-engine: T5.4 persist hook`

## Phase 6 — Acceptance

### T6.1 — Full acceptance run AC1–AC11
- [ ] Execute all; fill notes; fix gaps or waive with user
- Commit: `spec/rust-metal-engine: T6.1 acceptance`

### T6.2 — Handover freeze
- [ ] implementation_notes complete for takeover
- Commit: `spec/rust-metal-engine: T6.2 close`
