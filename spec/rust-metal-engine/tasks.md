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
- [~] In progress — all numbers below measure **PonyExl3/MLX, not this engine** (DE20):
  - [x] **Correctness cross-check of the T1.3d fixture**: reference greedy from prompt `[2]` (BOS) returns text **`"S"`** — identical to our argmax token 236773. Independent confirmation that the 60-layer EXL3 forward, hybrid attention, lm_head and softcap agree with the reference quant at position 0
  - [x] Reference-only spot numbers (`.venv-baseline`, MLX 0.32.2 / ponyexl3 0.3.0): load **15.1s**; 64-token prompt → prefill **18.1 t/s**, decode **5.7 t/s**; 4-token prompt → prefill 7.9 t/s (0.51s incl. warmup), decode 5.9 t/s; 1-token prompt → 4.7 t/s decode. Decode ~5.7 t/s ≈ 91 GB/s effective vs 273 peak (the known reference kernel-efficiency gap, not pageins)
  - [x] Rust CLI bench lands `kaplumba-bench/1` (T1.4), so the Python harness is now retirable per R7.4
  - [ ] Full reference matrix at **1k/2k/8k** prefill + 32/128 gen tokens; multi-token (position ≥ 1) identity check vs our engine; then retire `scripts/bench.py` (R7.4)
- Files: implementation_notes.md only (+ raw logs under /tmp)
- Test: notes contain numbers, commands, hardware context
- Commit: `spec/rust-metal-engine: T0.2 reference cross-check + baselines`

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
- [x] Done — sub-steps:
  - [x] **T1.3a** EXL3 decode primitives: trellis pack/unpack, codebook (MCG/Default/Mul1), perm, signs, Hadamard, `decode_packed_tile` — 26/26 unit tests vs reference goldens (q00 tile, MCG codewords, pack fixtures k=3/6)
  - [x] **T1.3b** reconstruct (inner/slab/public weights) + streaming `linear_forward_reconstruct` — 44/44 tests; stream ≡ stepwise rel err < 1e-3; dense public-weight loose parity
  - [x] **T1.3c** Gemma4 model forward (embed·√h, hybrid SWA/full attn, MLP, layer_scalar, softcap) — `crates/engine/src/gemma4.rs` (~1000 lines incl. tests), 21 new tests; **real 60-layer forward + lm_head 205s**, argmax matches fixture
    - [x] expand `TextConfig` (load.rs): n_heads, n_kv, n_global_kv, n_layers, vocab, eps, sliding_window, hidden_activation, max_pos, `rope_parameters`→`RopeParams`; per-layer head_dim/n_kv synthesized from `layer_types`+k_eq_v (full=512/4, sliding=256/16)
    - [x] `crates/engine/src/gemma4.rs` + lib.rs: `TensorSource`/`RawTensor` view of the artifact, RMSNorm(f32), dual RoPE tables (proportional full), k_eq_v attention (K=rope∘k_norm, V=weightless v_norm on raw k; scale 1.0; masks full kv≤q / sliding kv∈[q-1023,q]), gelu_tanh MLP, decoder order + layer_scalar (DE28 f64 exact), streaming lm_head slabs w/ softcap 30 → f32 logits
    - [x] unit tests: inv_freq/rope/rms/gelu/mask goldens (consts from notes), embed[2]+layer_scalar real-artifact reads, synthetic 2-layer end-to-end (forward/logits/greedy/cfg errors)
  - [x] **T1.3d** greedy generate (`greedy_generate`, EOS-aware) + recorded fixture; deterministic across runs
- Files: `crates/engine/src/exl3/*` (a/b); `crates/engine/src/gemma4.rs`, `load.rs`, `lib.rs` (c/d)
- Test: `cargo test` 67 pass; real-weight slow tests (gated `KAPLUMBA_RUN_SLOW=1`): full forward + lm_head bit-identical on repeat, partial depth (sliding+full) bit-identical, fixture prompt `[2]` → argmax **236773 ("S")** logit 29.9782
- Note: fixture is *recorded*, not yet *validated* against the PonyExl3 reference — that comparison is T0.2/T2.4 (OQ1 gate)
- Commit: `spec/rust-metal-engine: T1.3c gemma4 forward + greedy`

### T1.4 — CLI generate + bench entry
- [x] `kaplumba-engine generate --prompt ... --max-tokens N --temp ...` streaming; JSON stats (ttft_ms, pp_tps, tg_tps, mem) schema `kaplumba-bench/1`
  - `crates/engine/src/generate.rs`: `generate(model, prompt, GenParams, on_token) -> GenStats` (prefill/decode/ttft/finish_reason/computed_tokens), EOS-aware, seed-reproducible; `Gemma4Model::greedy_generate` now delegates to it
  - `crates/engine/src/sampler.rs`: PCG32 + top-k/top-p/temp (OQ4 defaults 1.0/64/0.95; temp 0 → greedy); T4.3 replaces the host path
  - `crates/cli`: `generate` command (hand-rolled arg parsing, streaming delta detokenizer, `--layers` depth limit, `--json PATH|-`), `kaplumba-bench/1` report mirroring `scripts/bench.py` keys + `sampling`/`tokens`/`text`; host+mem metrics via libc (`sysctl`, `getrusage`), Metal alloc/working-set from the device
- Files: `crates/cli/{main,args,report}.rs`, `crates/cli/tests/cli_generate.rs`, `crates/engine/src/{generate,sampler,testutil}.rs`
- Test: `cargo test` 101 pass (85 engine + 11 CLI unit + 5 CLI integration); report schema key set pinned and asserted in `build_report`; e2e `generate --layers 2 --json -` gated behind `KAPLUMBA_RUN_SLOW=1` (~22s, verified)
- Measured (release, real artifact): load 3.7s · 2-layer/2-token prefill 16.5s (ttft 16.5s) · full 60-layer 1-token forward + lm_head 205s · `peak_rss` 8.08 GB vs `metal_alloc` 16.03 GB (UMA buffers are not in `getrusage`)
- Commit: `spec/rust-metal-engine: T1.4 cli generate + bench report`

### T0.2 — Baseline capture (still open, next)
- [ ] Reference-only PonyExl3 numbers + cross-check of the T1.3d fixture (`[2]` → 236773); then retire `scripts/bench.py` (R7.4) now that the Rust bench exists

## Phase 2 — EXL3 decode path

### T2.1 — Fresh trellis GEMV + dequant kernels
- [x] Done — `crates/kernels` (MSL + Rust host wrapper) carrying the decode-shaped EXL3 path: `exl3_had_in` → `exl3_gemv` → `exl3_had_out`, wired into the engine with the CPU reference as fallback. Commits `b93c007` (kernels + wiring) and this one (measurement + alignment fix)
- Accuracy vs CPU reference: `had_in` **bit-exact**; `had_out` **≤1 f16 ulp** (FMA contraction in the 128-term sum); GEMV **≤5e-3** relative for K=1..8 on single tiles and on 3 real layer shapes. Host-only model of the pair table reproduces `unpack_trellis_tile`/`tensor_core_perm` for every K (guards the whole class of index bugs)
- Measured (release, real artifact, per-pass table in notes): **~165–180 Gw/s** decoded weights steady state; gate_proj 5376×21504 K=3 **0.72 ms**, lm_head 5376×262144 **K=6 8.58 ms**; ⇒ **~177 ms/token ≈ 5.6 t/s** decode for this kernel = 4.3× off the 41 ms/24 t/s bandwidth roofline (ALU/latency-bound)
- Found and fixed: pair table used the 16-wide tile edge instead of 256 samples/tile; `s1` computed from a wrapped index; `lm_head` spans are only 2-byte aligned (trellis now read as u16 pairs)
- Open (T2.3): Hadamard is O(n²) per 128-block (fast Walsh–Hadamard = 7 butterfly stages is ~36× fewer ops) and each linear uses its own command buffer + sync (~0.15 ms × 420 = 63 ms/token). Also: ~0.8–1.5 s **one-time** Metal residency cost per shard buffer, currently inside TTFT — move to load-time warm-up
- Files: `crates/kernels/src/{lib,exl3}.rs`, `crates/kernels/src/exl3.metal`, `crates/engine/src/metal_linear.rs`, `crates/engine/src/{gemma4,load}.rs`, `crates/cli/src/{args,main,report}.rs`
- Test: `cargo test -p kaplumba-engine --lib` (96) + gated real-artifact tests; `--metal` flag on `generate` with `engine.gpu_linears`/`engine.gpu_ms` in the bench report
- Target: AC4, R3.5, R4.3

### T2.2 — Working-set cap + load hygiene
- [ ] Metal allocation cap pattern (~92% recommended working set); streaming load; no trellis host/device duplication
- Files: allocator, loader
- Test: resident footprint report; no OOM on 24 GB during load+decode
- Target: R5.4, AC6
- Commit: `spec/rust-metal-engine: T2.2 working set cap`

### T2.3 — M4 Pro kernel sweep
- [x] Done — swept on this GPU, recorded in notes (not copied from M5 Max). See the "T2.3 kernel sweep" section of `implementation_notes.md`
- Results: fast Walsh–Hadamard kept (7 butterfly stages vs an O(n²) per-block matrix product, ≤1 f16 ulp vs CPU); **4 warps/threadgroup rejected** (small shapes 25–40% worse — too few threadgroups for 20 cores); **threadgroup tile staging rejected** (wrong weights, reverted unproven); scalar pair loads kept (uint4 within noise)
- Reliable metric: `lm_head` gemv **8.46 ms / 1.41e9 weights = 167 Gw/s** (±1.4%); short kernels have ±25% run-to-run clock noise, so compare only long loops
- Ceiling analysis (notes): ~9–12 ops/weight vs ~4.1 Tops/s issue ⇒ ~350–450 Gw/s ⇒ **~9–12 t/s** decode for this decode algorithm; bandwidth roofline 24 t/s; **18 t/s needs ≤5 ops/weight, i.e. a different decode algorithm, not tuning**
- Files: `crates/kernels/src/exl3.metal`, `crates/kernels/src/exl3.rs`
- Test: `cargo test -p kaplumba-engine --lib metal_linear` (10; had_in/had_out ≤1 ulp, gemv ≤5e-3), `KAPLUMBA_RUN_SLOW=1 … bench_passes_slow -- --ignored --nocapture`
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

### T3.7 — Bounded prefill scratch + peak-RSS gate (user directive 2026-09-22)
- [ ] Audit + prove: prefill attention is **online-softmax over KV tiles** with an O(L) working set (no `[heads, L, L]` score matrix, no model-built `[heads, q, kv]` bias tensor); every scratch buffer is fixed-size and allocated once at load; nothing retains a freed-buffer high-water mark; no per-chunk unconditional clear/marker that would keep transients alive
- Files: prefill attention kernel + engine scratch arena; `crates/cli` bench report
- Test: **measured peak RSS / Metal footprint at 1024 / 2048 / 8192 prefill tokens**, each ≥ 3 repeats, asserted to grow by ≤ (KV bytes for that length + fixed constant); report the three numbers in `implementation_notes.md` and in the `kaplumba-bench/1` report
- Target: R5.3, R5.5, AC6 (this is an acceptance number, not a nice-to-have)
- Commit: `spec/rust-metal-engine: T3.7 bounded prefill scratch`
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
