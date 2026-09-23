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

### T2.5 — gemv K-split + partial-sum reduce (decode occupancy)
- [x] Measured pattern (T4.0/bench): Gw/s tracks wave depth — lm_head 51 waves = 241, gate 4 = 172, down 1 = 149, o 1 = 119, k/v ~0.2 waves worst; 1-wave shapes have no tail-latency hiding across ~320 concurrent simdgroups. Split the in-tile range across `grid.y` (`nsplit` param), each split-TG partial-sums into scratch `[nsplit, out_tiles, 16]`, new `exl3_gemv_reduce` sums splits → `y`. Policy: `nsplit = clamp(div_ceil(1024, out_tiles), 1, 8)` → gate/up/lm unchanged (1), down/o → 4, q → 2, k/v → 8. `nsplit==1` binds `y` directly (byte-identical path). Scratch ≤ 128 KB transient (not model/KV growth). f32 reassociation on split shapes only — token-level gates (argmax/span 0.55/gemv 2e-3) must hold
  - Gotcha: Metal requires all kernel inputs uniformly scalar or uniformly vector — the new `uint2 tg` grid arg forced `uint2 thr` (mixed `uint2`+`uint` → compile error surfaced only via `build_probe` panic)
- [x] **Verified:** kernels 19 (incl. `gemv_splits_policy_covers_shapes`); fast engine 113; full gated **113 passed / 0 failed / 4 ignored** (fixture 236773, span 0.55, batched-vs-gemv unchanged → reassociation inside gates); `bench_passes_slow`: **down 149→185 Gw/s (+24%), o 119→135 (+13%)**, gate 177 / lm 241 flat as designed; `decode_throughput_slow`: **6.42 → 7.15 t/s** (155.6 → 139.8 ms/token, 103 → 114.6 GB/s), decoded ids identical `[236773, …]`
- Files: `crates/kernels/src/{exl3.metal,exl3.rs}`
- Test: above
- Target: R2.1 (tg/s), profile-before-machinery
- Commit: `spec/rust-metal-engine: T2.5 gemv k-split`

### T2.6 — gemv funnel micro: mask hoist + 2-way unroll (issue/MLP)
- [x] Experiment A: hoist `pk0..3` field extraction (`i0/i1/s1` × 4 pairs = 20 loop-invariant ops/iter) out of the `it` loop — bench: **gate 177 → 187 Gw/s (+5.6%)**, lm 240 flat (long loop already hoisted by LLVM), down/o flat
- [x] Experiment B: 2-way unroll of the `it` loop → 8 `funnel_pair`s (16 word loads) visible per scheduling region, odd tail via second loop; each body's statement order and the accA/accB `+=` sequence preserved exactly (bit-identical accumulation). Bench vs A: gate 189 (+1.1%), down 183 → 192 (+4.9%), o 138, lm 239 (flat — already latency-hiding via 51 resident waves)
- [x] **Verified:** kernels 19, fast engine 113, full gated **113 passed / 0 failed / 4 ignored** (fixture 236773, span 0.55, batched-vs-gemv unchanged); `decode_throughput_slow`: **7.15 → 7.52 t/s** (139.8 → 132.9 ms/token, 120.6 GB/s), decoded ids identical `[236773, …]`; total T2.5+T2.6: **6.42 → 7.52 t/s (+17%)**
- Files: `crates/kernels/src/exl3.metal` (`exl3_gemv` only)
- Test: above
- Target: R2.1 (tg/s), toward T2.3's 350–450 Gw/s ops ceiling
- Commit: `spec/rust-metal-engine: T2.6 gemv funnel micro`

### T2.7 — decode phase profile (layer-scaling) → last-lever decision
- [x] `decode_layer_scaling_slow`: fwd ms/token at `layers = 60 / 10 / 0` = **132.86 / 28.72 / 6.99 ms** → `fwd(n) = 7.89 + 2.083·n` (direct t0 6.99, ~0.9 ms sub-linear). Non-layer **a ≈ 7.9 ms (6%)** = lm_head ~5.9 + embed/norm/waits ~1–2; per-layer **2.083 ms (95%)**
- [x] Bench q/k/v added (were missing): q 0.330 / k 0.235 / v 0.233 ms (+ gate 0.628, down 0.601, o 0.330) → isolated linear sum **2.357 ms/layer** vs in-model 2.083 → isolation submit overhead ~0.07–0.1/call; attn+rope/rms+launch residual **≈ 0.14 ms/layer ≈ 7% < 10%**. k/v Gw/s 94/95 (full incl. had), o/q 134, down 192, gate 184, lm 239
- [x] **Decision (b):** linears ≈ 93% of layer ≈ 88% of token — same-algorithm plateau. gemv now issue-bound (~50% of T2.3's 4.1 Tops rate at ~189–239 Gw/s × ~10 ops/weight); ops ceiling ~9–12 t/s, bandwidth floor 16 GB ÷ 273 ≈ 17.6 t/s, neither reachable without cutting ops/weight ≤ 5 (different algorithm = MTP/verify-batch — user-gated at 18 → circular). **Stop decode grind toward 18; deliver end-of-track conflict report (200 pp/s MMA wall + 18/MTP circularity) with these tables; continue remaining roadmap (T2.2 TTFT, T2.4 quality, coherence verify) pending user's gate decision**
- Files: `crates/engine/src/gemma4.rs` (`decode_layer_scaling_slow`), `crates/engine/src/metal_linear.rs` (bench q/k/v names)
- Test: `decode_layer_scaling_slow` green + full gated suite stays green
- Target: R2.1 (tg/s) — decides remaining decode work
- Commit: `spec/rust-metal-engine: T2.7 decode phase profile`

### T2.8 — gemv ALU trim: hoisted tile-address increments (decode)
- [x] Replaced the per-body `(ulong)(it * p.out_tiles + ot) * p.twords * 2u` 64-bit recompute with one hoisted base (clamped `min(it0, in_tiles)` so pathological tiny-shape `nsplit > in_tiles` never forms a past-column address) + `tstride` pointer advances; `xrow` advanced by `TILE`; `it` hoisted out of the for-init so the remainder loop reuses the already-advanced `tile`/`xrow` (odd-tail condition collapses to `it < it1`). Funnel/x-load statement order and acc sequence unchanged → bit-identical
- [x] **Verified:** kernels 19, fast engine 113, full gated **113 passed / 0 failed / 5 ignored** (fixture 236773, span 0.55, batched-vs-gemv unchanged); `bench_passes_slow`: **gate 184→203, down 192→205, o 134→157, q 134→142, lm 239→250 Gw/s** (k/v flat 106/95 — their it-loop is too short for the address chain to matter); `decode_throughput_slow`: **7.52 → 8.08 t/s** (132.9 → 123.8 ms/token, 129.4 GB/s), decoded ids identical `[236773, …]`
- Files: `crates/kernels/src/exl3.metal` (`exl3_gemv` only)
- Test: above
- Target: R2.1 (tg/s), toward T2.3's 350–450 Gw/s ops ceiling
- Commit: `spec/rust-metal-engine: T2.8 gemv address hoist`

### T2.9 — decode sibling fusion: qkv + gate+up (T4.1 semantics) → PIVOTED
- [!] **Precondition REFUTED by measurement:** new gated test `sibling_input_tables_shared_slow` memcmp's `suh` (input Hadamard table) for every sibling pair on the real artifact — **q≠k, gate≠up, q≠v, q≠gate** (converter quantized each tensor's input scale independently). One `had_in` per group cannot feed siblings; the fusion's entire win (shared input prep + shared xh) does not exist on this artifact. Remaining fusion value = dispatch-count only (~0.3 ms/token — host already batches one cb, T4.2), not worth the surface area (rope-race precedent). Test kept as evidence + guard
- [x] **Pivot (occupancy-2):** deepen the T2.5 K-split policy — k/v bench at 94/106 Gw/s (weakest shapes) run only `nsplit=4` (TARGET_TGS=1024 ÷ out_tiles=256); raise TARGET_TGS to 4096 / MAX_SPLIT to 16 so every gemv shape reaches ~4096 TGs (k/v → 16, down/o → 13, q → 8, gate → 4, lm_head stays 1 — already 16384 TGs; partial ≤ ~344 KB transient). f32 reassociation on the newly-split shapes only → token gates must hold
- [x] **Verified:** `gemv_splits_policy_covers_shapes` updated (16384→1, 4096→1, 1344→4, 512→8, 336→13, 256→16, 1→16); kernels 19, fast engine 114, full gated **114 passed / 0 failed / 5 ignored** (fixture 236773, span 0.55, batched-vs-gemv unchanged); `bench_passes_slow`: gate 205, down 212, q 145, lm 247 Gw/s up; k/v isolated bench regressed 106/95 → 94/93 (chunk 84 → 21 iters/TG: per-TG fixed cost not amortized) — **end-to-end decode still wins**: `decode_throughput_slow` **8.08 → 8.39 t/s** (123.8 → 119.2 ms/token, 134.5 GB/s). Note: greedy ids flip at position 5 on the empty-prompt run (`236770→236828`) — new f32 reassociation on newly-split gate/q; position-0 fixture + all token gates hold (cross-build mid-sequence greedy identity is not a spec property; R8.1 is within-build)
- Files: `crates/kernels/src/exl3.rs` (`gemv_splits` + policy test), `crates/engine/src/metal_linear.rs` (`sibling_input_tables_shared_slow` evidence test)
- Test: above
- Target: AC4, R3.1→R2.1 (tg/s occupancy)
- Commit: `spec/rust-metal-engine: T2.9 sibling fusion refuted, split policy deepened`

### T3.10 — small-N K-split for sg GEMM (prefill, T3.9 ranked #1)
- [x] **Policy amended after re-profile (2026-09-23):** original `clamp(div_ceil(1024, out_tiles),1,8)` is rows-blind — at rows≥512 every shape already fills ≥1 wave (base_tgs ≥ ~320 slots, measured from gate/k phase profile: S≈320 concurrent TGs, gate@256 = 1344/320 waves × 2.86 ms = 12.03 ✓), so a split there is an exact wash (waves×(τ/n) = τ) plus partial-I/O overhead. New policy: **`sg_splits(shape, rows) = clamp(div_ceil(320, base_tgs), 1, 8)`** with `base_tgs = (out_tiles/4) * rows.div_ceil(64)`, also capped at `stages = in_tiles/2` — splits only sub-wave shapes (tiny-out k/v: L5 k = 128 out → 16 TGs even at rows=512; small prefill rows=2..63) and keeps gate/down byte-identical (`nsplit==1` → old grid, direct-to-y store)
- [x] Mirror the T2.5 pattern on `exl3_gemm_sg`: split the `ks` stage range across `grid.z`, each split-TG partial-accumulates its own `C[4][4]`, writes `partial[tg.z][rows_pad, N]` f32 (host binds shared `ensure_partial` buffer), new `exl3_gemm_sg_reduce` sums splits in fixed s-ascending order (deterministic — no atomics) → `y`. New `Exl3GemmSgParams` (9 u32: +nsplit, +stride elements) so scalar `exl3_gemm` + T3.8 probes keep the 7-u32 `Exl3GemmParams` untouched. No atomics: run-to-run greedy determinism required
- [x] f32 reassociation on split shapes only → span/batched token gates must hold (argmax + max_abs < 0.55, batched-vs-gemv < 2e-3)
- [x] Verify: kernels suite (+ new `sg_splits_policy_covers_shapes`), fast engine, `real_artifact_batched_linear_matches_gemv_slow`, full gated suite, `prefill_throughput_slow` ≥ +8% at 512 (≥ ~94 t/s from 86.8 baseline; expect gains concentrated in tiny-kv layers + rows≤128 chunks — re-profile, do not expect the old ~600 ms rows-blind estimate)
- **Acceptance note:** measured **86.8 t/s @512** after T3.10 + scratch-fix (not ≥94). Ceiling 93.7 in `spec/final-same-algorithm-ceiling` **not met** → ceiling spec remains on hold; do not claim +8%.
- Files: `crates/kernels/src/{exl3.metal,exl3.rs}`
- Test: above
- Target: R2.1 (pp/s)
- Commit: `spec/rust-metal-engine: T3.10 sg small-N K-split` (landed entangled with scratch-fix code; see prefill-scratch-fix commit)

### T2.2 — Working-set cap + load hygiene (TTFT warm-up relocation)
- [ ] Move the one-time Metal residency + first-forward shader compile cost (3822 ms measured in T4.0's first forward) out of first-token latency: warm pipelines at end of `load_exl3` (build every production pipeline + dispatch a tiny probe) so `decode_throughput_slow`'s first forward ≈ steady-state token; record cold load vs warm load split
- [ ] Metal allocation cap pattern (~92% working set); streaming load; no trellis host/device duplication (already largely true — document + assert)
- [ ] Verify: resident footprint report (16.028 GB weights + KV + scratch), first-forward ms in `decode_throughput_slow` down from 3822 toward < 500, no OOM on 24 GB during load+decode, full suite green
- Files: `crates/engine/src/{load,metal_ctx,metal_linear,gemma4}.rs`
- Test: above
- Target: R5.4, R2.6 (TTFT), AC6
- Commit: `spec/rust-metal-engine: T2.2 working set cap`

### T2.4 — Quality gate (OQ1)
- [ ] Golden prompts at **temp=1.0, topk=64, top_p=0.95** (multi-seed optional); review sampled outputs vs reference quant; pass/fail recorded
- Test: the gate script/CLI itself
- Target: AC6
- Commit: `spec/rust-metal-engine: T2.4 quality gate`

### T4.8 — Spec quality suite (coherence + goldens)
- [ ] Golden suite temp 0 + sampling distribution tests (R8): fixture 236773 identity, greedy token-identity vs recorded loops, sampling smoke at temp>0; record coherence review (user: "tokens 100% coherent")
- Test: the suite
- Target: R8, AC1
- Commit: `spec/rust-metal-engine: T4.8 spec quality`

## Phase 3 — Prefill kernels

### T3.1 — Verify EXL3 artifact (T3.1 from old plan, renamed)
- [x] Artifact already exists — verify load + resident size + manifest/versions in notes (done under omlx-exl3 T0.3/T3.1 research)
- Commit: (historical notes)

### T3.2 — Dual-path matmul + batched prefill GEMM
- [x] Done — `rows==1` GEMV vs `rows≥2` `exl3_gemm` (RGT=8 row-chunks; trellis decode once per out-tile×chunk, reused across 8 rows); multi-row Hadamard/rope; `encode_span` packs a prefill chunk into one command buffer; `reserve_rows`/`end_tokens`
- **Bug found + fixed:** in-place q-RoPE (`q → q`) races `kr_rope` (reads `x[xr+lo+kk]`/`x[xr+lo+kk+hh]` while writing `o[i]`). Comment always said in/out must be distinct; tests used distinct buffers, production did not. Symptom: span vs token-loop `n_diff=258588/262144`, `max_abs=2.95e-1` (argmax accidentally agreed). Fix: rope `q → q_raw`, attention reads `q_raw` (both `encode_token` and `encode_span`). After fix: **0/262144 differ, bit-identical**.
- Measured pp/s (`prefill_throughput_slow`, debug build, M4 Pro): token-loop @512 **5.20 t/s** → span @512 **15.0 t/s (2.88×)**, @2048 **14.8 t/s (2.85×)**, @8192 **14.4 t/s (2.78×)**. Far from 200 t/s: GEMM re-streams weights `rows/RGT=32`× per chunk (RGT=8) — next lever is larger reuse / fused siblings / kernel time, not correctness.
- Tests: `gpu_gemm_rows_match_per_row_gemv_and_cpu`, `rope_multi_row_matches_cpu_reference`, `reserve_rows_batches_match_row_at_a_time` pass; gated `real_artifact_span_matches_token_loop_slow` bit-identical; fixture still argmax 236773 / logit 29.9782; full suite 109 engine + 14 kernels green
- Files: `crates/engine/src/{gemma4,gpu_decode,kv_cache,metal_linear}.rs`, `crates/kernels/src/{exl3,fuse}.{rs,metal}`, `crates/kernels/src/lib.rs`
- Target: AC2, R2.1
- Commit: `spec/rust-metal-engine: T3.2 batched prefill GEMM`

### T3.3 — Multi-row packed weights at load
- [ ] Interleaved packing applied on load; metadata layout rules documented
- Test: tolerance vs unpacked; pp/s delta
- Commit: `spec/rust-metal-engine: T3.3 row packing`

### T3.4 — Shape dispatch
- [x] Done — `encode_gem` (`crates/kernels/src/exl3.rs`) is the single dispatch point: `rows == 1` → GEMV, `sg_eligible` (rows ≥ 2, in_tiles even, out_tiles % 4 == 0, byte_offset % 4 == 0) → new **simdgroup GEMM** `exl3_gemm_sg` (v19c port: 64 token rows × 64 out cols per TG, 32-deep in-feature stages, `wblk[32×72]` staging, `simdgroup_half8x8 × simdgroup_half8x8 → simdgroup_float8x8` MMA), else RGT=8 fallback. All real prefill shapes take sg; lm_head is `rows == 1` (gemv) or ineligible → never affected. Decode path (rows==1) untouched
- xh scratch pads to `rows.div_ceil(64)*64` rows (`xh_rows()`); store masks `gr < blk_len`; TG-uniform early return before barriers
- Phase profile (release, rows=256, real shapes): q 5.92 / k 2.87 / o 5.04 / **gate 66→12.03 (5.5×)** / down 13.16 ms; gate@512 23.54 ms
- Correctness: synthetic `gpu_gemm_rows_match_per_row_gemv_and_cpu` (5/3/17-row tails) < 2e-3; **new** `real_artifact_batched_linear_matches_gemv_slow` — q/gate/down ×65 rows vs per-row GEMV on real trellises worst rel err **9.76e-4**; gated span test argmax identical (207=207), max_abs 2.374e-1 (300-token f16 compounding — kernel gated separately at ~1e-3, threshold 0.35), replay bit-deterministic; fixture/attention/CPU-linear/greedy all green; full gated suite 111 passed / 3 ignored; kernels 15 passed
- Two assertions relaxed (documented in code): bit-identical span==loop → argmax + 0.35 bound (tensor-core k-block order ≠ scalar, mathematically cannot match); batched==gemv → `max_rel_err < 2e-3` (~1 f16 ulp)
- Measured pp/s (`prefill_throughput_slow`, release): token-loop @512 **5.25 t/s** → span **25.7 / 25.2 / 24.1 t/s @ 512 / 2048 / 8192** (4.9× / 4.8× / 4.6×). Flat in context; ~75% of chunk time is now **outside** the linears (linear ceiling ≈ 79 t/s @256) → next lever is T3.6 sync/fusion + attention/glue profile, not more GEMM
- Files: `crates/kernels/src/{exl3.metal,exl3.rs}`, `crates/engine/src/{metal_linear,gemma4}.rs`
- Target: AC2, R2.1
- Commit: `spec/rust-metal-engine: T3.4 simdgroup GEMM shape dispatch`

### T3.5 — SWA KV bound in attention
- [x] Done — GPU attention kernel (`crates/kernels/src/attn.metal`) + engine wiring; sliding layers iterate only the in-window KV (the kernel starts its KV loop at `max(0, p0+1-window)`, so a sliding layer never touches KV outside the window)
- Design: one threadgroup = `AROWS`(8) query rows × one query head, `PART` = `head_dim/32` threads per row so the slice reduction is a `simd_shuffle_xor` all-reduce. **No barriers, no threadgroup memory** → see T3.7
- One rule covers prefill and decode: query row `r` sits at absolute position `kv_len - s + r`, so `s == kv_len` is a prefill and `s == 1` is a cached decode step
- Verified: `cargo test -p kaplumba_kernels` (7 tests: sliding window wide/narrow/odd, full causal, cached decode at kv 8/16/17/32/64/256/512/700, kv=1, padded rows, unsupported shapes) + `real_artifact_attention_matches_cpu_slow` on the real layer shapes (sliding 32q/16kv×256 and full 32q/4kv×512): **max rel err 4.88e-4 ≈ 2 f16 ulps**
- Bug found + fixed: `pos` used the row *within* the threadgroup instead of `tg.x*AROWS + tp.y` (every block past the first recomputed block 0's positions; hidden by all `s ≤ 8` cases), and q/out were indexed by absolute position instead of row (identical only when `s == kv_len`, i.e. every prefill case passed while decode was wrong)
- Files: `crates/kernels/src/{attn.metal,attn.rs,lib.rs}`, `crates/engine/src/{metal_linear,gemma4}.rs`
- Target: AC2, R2.5, R3.7
- Commit: `spec/rust-metal-engine: T3.5 attention kernel`

### T3.6 — Prefill sync policy + fusion audit (+ chunk profile)
- [x] No unconditional sync+clear per chunk: `encode_span` builds **one command buffer per 256-row chunk** — single `commit+wait` at the chunk boundary (`gpu_decode.rs:777`), host encode 1.3 ms/chunk, and **no `clear()`/pool purge exists anywhere on the span path**. Per-chunk wait vs a pipelined design costs ≈ encode overlap (1.3 ms of 3630 ms = 0.04%) — not worth removing. AC8/R2.4 evidence recorded in implementation_notes
- [x] No accidental host materialize between fused candidates: gate+up / qkv write into the same device arena inside one command buffer (no host round trip, no cache clear). Actual kernel fusion remains T4.1
- [x] Chunk profile harness: `KAPLUMBA_SPAN_PROFILE=1` (per-chunk encode/wait + optional per-layer cb split) in `encode_span`; `phase_profile_chunk_budget_slow` (per-linear / attention / glue budget vs measured wall)
- [x] **Profile finding → fix:** artifact file offsets are only 2-byte aligned (alternating `off%4=2` across layers, phase-flip at the shard boundary) → `sg_eligible` (`byte_offset%4==0`) failed for ~half the linears → silent scalar-RGT path (+224 ms/layer). `load_exl3` now **repacks every tensor at a 64-byte-aligned device offset** (file order preserved; `tensors` + `shard_headers` both updated; alloc = packed end; padding ≤63 B/tensor ≈ 100 KB total → **no memory increase**)
- [x] Regression guard: `dump_trellis_alignment_slow` (`align_dump.rs`) asserts every linear's trellis offset `%4==0` (121 tensors, incl. lm_head)
- Measured after (release, real artifact): all layers on sg; full-layer linear sum **57.85 ms (was 307.72)**; 256-chunk **3635 ms = 70.4 t/s (was 9882 = 25.9)**; budget unaccounted +172 ms (was +3948 → profile now closes); `prefill_throughput_slow` **77.8 / 72.3 / 65.0 t/s @512/2k/8k** (was 25.7/25.2/24.1 — 3.0×); span 300 tok 4.25 s (70.6 t/s, was 11.93 s)
- Span drift after repack: max_abs 4.002571e-1 (was 2.374e-1 — doubled count of sg-vs-gemv layers), argmax 207=207 → gate raised to <0.55 with measured citation; per-linear kernel gate unchanged (~1e-3). Repack correctness proven by the full gated suite (fixture argmax 236773 unchanged)
- Linears are now **94% of chunk budget** (3249/3463) → remaining path to 200 t/s: sg kernel efficiency (currently ~4.3 TFLOP/s ≈ 27% of fp16 peak) + T4.1 sibling fusion (~900 ms/chunk fused savings) → re-profile each step
- Files: `crates/engine/src/{load.rs,gpu_decode.rs,metal_linear.rs,gemma4.rs,exl3/reconstruct.rs}` + new `align_dump.rs`
- Target: AC8, R2.4 (sync), R2.1 (pp/s 3×)
- Commit: `spec/rust-metal-engine: T3.6 chunk profile + loader 64B repack`

### T3.8 — sg GEMM ceiling / decode-tax profile (path to 200 pp/s)
- [x] Probe kernels: `exl3_gemm_sg_ceiling` (funnel decode → direct f16 device loads, identical grid/wblk/barriers/MMA/store), `exl3_gemm_sg_decode_only` (funnel + barriers, no MMA/store), `exl3_gemm_sg_mma_peak` (wblk filled once, PEAK_REPS=32 hot passes) — test-only pipelines built locally in `exl3.rs` tests (production `Exl3Gemv` unchanged); CPU-ref correctness for ceiling on small shape (K=64/N=128/rows=64, max_rel < 1e-3); perf under `KAPLUMBA_RUN_SLOW`
- [x] **Measured (release, gate shape rows=256/K=5376/N=21504, gem-only vs gem-only):** real sg **11.786 ms = 5.02 TFLOP/s**; ceiling **9.418 ms = 6.28 TFLOP/s**; **funnel tax = 20.1% of gem time (2.37 ms)**; decode_only 0.211 ms (cheap in isolation → the tax is a live-C-registers×funnel interaction, not raw funnel cost); mma_peak (hot fragments) **5.84–6.19 TFLOP/s ≈ ceiling** → weight streaming costs ~0
- [x] **Physics finding (blocks 200 as specified):** sustained f16→f32 simdgroup MMA on M4 Pro = **~6.3 TFLOP/s measured hot**. 200 pp/s needs linears ≤1065 ms/chunk = **~11.6 TFLOP/s sustained** (chunk budget: 12.35 TFLOP linears + 215 ms non-linear → 1280 ms) — **1.85× beyond the hardware's MMA ceiling**, before any decode/glue/attention cost. Best-possible chunk at the wall (tax=0, every linear at gate-ceiling, non-linear minimal) ≈ **2200 ms → ~116 t/s**; realistic path (tax fix + small-N K-split + glue) **~100–110 t/s**. plan.md's roofline (12.4 TFLOP/s ≈ 78% of an assumed ~16 fp16 peak) overstates the f32-accum MMA rate available on this device — surface to user with these numbers; do not burn more GEMM micro-opts chasing 200
- [x] Engine-budget cross-check: `linear_ms` 11.98 ≈ gem-only 11.79 → **had_in/had_out ≈ 0.19 ms total at rows=256** (had is a decode-path concern at rows=1 per-linear overhead, not a prefill cost)
- Next under this task: funnel-tax fix (spill C[4][4] to threadgroup memory around the fill loop so the funnel runs with lean registers) → expect linears −648 ms/chunk → **~86 t/s**; then small-N K-split (k/v run at 0.97–2.1 TF vs 5.0 due to TG starvation: k=64 TGs, q=256 TGs) for the next ~600 ms
- Files: `crates/kernels/src/{exl3.metal,exl3.rs}`
- Test: CPU-ref match on small shape; printed TFLOP/s + ms for real/ceiling/decode/peak on gate shape
- Target: R2.1 (pp/s), profile-before-machinery (spec §8)
- Commit: `spec/rust-metal-engine: T3.8 sg ceiling/decode-tax probe`

### T3.9 — sg funnel register-pressure fix (20.1% gem tax)
- [x] **C-spill hypothesis REFUTED:** implemented the planned threadgroup `csave` spill (16 KB, per-sgid slots, same 2-barrier structure, lossless f32) — gate-shape real sg got **worse: 11.786 → 18.333 ms, tax 20.1% → 48.7%** (the `simdgroup_store/load` round-trip + pushing TG footprint 5.5 → 21.6 KB cost more than the presumed stack spills, or evicted wblk from fast TG memory). Reverted via `git checkout`
- [x] **What worked — phase-split funnel fill:** issue **all 32 global loads** of the stage first (independent addresses → memory-level parallelism, 32 `uint` temps `la0/lb0/la1/lb1`), then shift/decode/`wblk`-write as pure ALU. Same loads, same wblk entries → bit-identical. Result: real sg 11.786 → **10.740 ms = 5.51 TFLOP/s; funnel tax 20.1% → 12.0%** (ceiling stable ~9.44). j-outer variant (16 temps, 12.4%) = noise-equal; kept the 32-temp deeper-MLP form
- [x] **Verify:** kernels 18; fast engine 113; full gated **113 passed / 0 failed / 3 ignored** (fixture 236773, span 0.55, batched-vs-gemv unchanged → bit-identity confirmed); budget linears **3249 → 2900 ms** (chunk 3092 = 2900+114+71+6); prefill bench **86.8 / 79.5 / 71.0 t/s @ 512 / 2048 / 8192** (was 77.8 / 72.3 / 65.0 → +12% / +10% / +9%)
- Residual 12.0% tax ≈ funnel address/shift ALU vs ceiling's simple loads — marginal levers only; stop here. Remaining levers ranked: small-N K-split (~600 ms → ~110 t/s) > glue fusion (114 ms) > residual funnel (~190 ms)
- Files: `crates/kernels/src/exl3.metal` (fill loop of `exl3_gemm_sg` only)
- Test: gate-shape probe + CPU-ref + both engine suites + bench
- Target: R2.1 (pp/s), no quality loss (bit-identical gate)
- Commit: `spec/rust-metal-engine: T3.9 sg fill register-pressure fix`

### T3.7 — Bounded prefill scratch + peak-RSS gate (user directive 2026-09-22)
- [~] Kernel half **done**: attention is online-softmax over KV tiles with an O(1) working set — running max/denominator/output accumulators and one 16-entry score tile all live in **registers**, the slice reduction is `simd_shuffle_xor`, and the kernel allocates **no** threadgroup memory and uses **no** barriers. There is no `[heads, L, L]` score matrix, no `[heads, q, kv]` bias tensor, and no growable scratch anywhere in the attention path
- [ ] Remaining (blocked on the batched prefill path T3.2/T5.1): the CLI-side proof — measured peak RSS / Metal footprint at **1024 / 2048 / 8192** prefill tokens, ≥3 repeats each, asserted to grow by ≤ (KV bytes for that length + fixed constant), recorded here and in the `kaplumba-bench/1` report
- Files: `crates/kernels/src/attn.metal` (structural); `crates/cli` bench report + measurement (remaining)
- Target: R5.3, R5.5, AC6 (this is an acceptance number, not a nice-to-have)
- Commit: `spec/rust-metal-engine: T3.5 attention kernel` (kernel half)

## Phase 4 — Decode amplification

### T4.0 — Decode baseline (end-to-end tg/s + bandwidth)
- [x] Gated test `decode_throughput_slow` (`gemma4.rs`): real artifact + `MetalLinear::new` + `.with_gpu` (first attempt forgot `with_gpu` → silently ran the ~0.18 t/s CPU path and timed out — always `assert!(g.gpu_enabled())`), device KV (T4.2 path), greedy, `eos=[]`, 64 steps
- [x] **Baseline (release, M4 Pro):** decode forward-only **155.6 ms/token → 6.42 t/s**; host between tokens **0.26 ms** (sample+convert+submit — negligible, T4.2 single-cb + shared logits confirmed); end-to-end **6.41 t/s**; implied weight stream **103.0 GB/s** (16.028 GB/token vs ~273 rated); first forward 3822 ms = one-time Metal residency/shader warm-up (T2.1 open item inside TTFT); decoded starts `236773` (fixture ✓), text from bare `[2]` prompt is expected gibberish
- [x] **Walls:** bandwidth roofline 15.5–16.0 GB ÷ 273 GB/s ≈ **17.6 t/s** (18 target is above rated bandwidth before any ALU); T2.3 ops ceiling 350–450 Gw/s ≈ **9–12 t/s** for the current gemv algorithm; today 6.42 → headroom to T2.3 ceiling via gemv/ops-count work, then 18 needs MTP/verify-batch (Phase 4) or a lifted gate — surface with the 200 pp/s conflict in the end-of-track report
- Files: `crates/engine/src/gemma4.rs` (tests), `spec/rust-metal-engine/tasks.md`
- Test: `KAPLUMBA_RUN_SLOW=1 cargo test -p kaplumba-engine --release --lib -- --ignored --nocapture decode_throughput_slow`
- Target: profile-before-machinery (spec §8); input to the 18 t/s conflict report
- Commit: `spec/rust-metal-engine: T4.0 decode baseline`

### T4.1 — Sibling fusion (qkv, gated gate+up)
- [~] Superseded in scope by **T2.9** (same semantics, implemented on the decode `encode_token` path first); span-path fusion rides along inside T2.9's guard
- Test: tg/s up or neutral; wrong-fusion guard unit test; golden outputs
- Target: AC4, R3.1
- Commit: `spec/rust-metal-engine: T4.1 sibling fusion`

### T4.2 — One command buffer per decode token
- [x] Done — every dispatch of one decode token (411 linears + 60 attentions + glue) is encoded into a **single command buffer** and submitted once, replacing the per-linear `commit`+`wait_until_completed` storm (471 cb/token ≈ 0.4 ms each ≈ 180 ms of a 275 ms token)
  - `crates/engine/src/gpu_decode.rs`: `GpuDecoder::encode_token` — device-resident arena with a 3-slot hidden ring (layer i reads `2i%3`, writes `(2i+2)%3`); K/V written straight into the shared-mode device cache (RoPE applied in-place on the k write); only 2 host round trips per token (embed row in, logits out)
  - `crates/kernels/src/fuse.{metal,rs}`: `kr_rms` / rope / copy / gelu_mul / add / add_scale glue kernels (norms, RoPE, residuals all GPU-side; only `layer_scalar` f64-vs-f32 handoff remains ≤1 ulp DE28)
  - `crates/engine/src/kv_cache.rs`: `Store::{Host,Device}` — device cache in `StorageModeShared` (shader-writable, same heap as weights), `reserve_row`/`row_span`/`layer_{keys,values}_span`; host store unchanged as the CPU fallback
  - `gemma4.rs`: `new_kv_cache` / `ensure_decoder` (lazy build, unavailable models fall back) / T4.2 branch in `forward_cached_layers` gated on `cache.is_device_backed()`
  - `attn.rs`: `AttnParams` carries `kv_rows,q_pos0,kv_pos0` (was `kv_len`); `exl3::encode_linear` + `MetalLinear::encode_*` let an external encoder drive the linears; `Exl3Attention::run_in`
- Measured (release, real artifact, `generate --metal --kv-cache --temp 0`, 6-token prompt, 8 gen): **decode 5.68 t/s** (was 3.64 with per-cb submission → +56%; naive 0.18), prefill 1.45 t/s, 5343 GPU linears/8 steps, `kv_bytes` 12.6 MB
- Verified: gated fixture `real_artifact_cached_fixture_and_gpu_slow` on the **device** cache (`new_kv_cache`) → argmax **236773**, logit 29.9782, 411 GPU linears, `device_kv=true`, fresh-cache **bit-identical** second run (in-cb dispatch ordering holds); cross-process determinism (two CLI runs → identical tokens `[711,506,1852×6]`); full suite green (136)
- Files: `crates/engine/src/{gpu_decode,gemma4,generate,kv_cache,metal_linear,lib}.rs`, `crates/kernels/src/{fuse,attn,exl3,lib}.{rs,metal}`
- Test: above; plus `fuse::*` unit tests vs CPU reference
- Target: AC4, R1.3
- Commit: `spec/rust-metal-engine: T4.2 one command buffer per token`

### T4.3 — Fast sampler without clear tax
- [ ] Single-pass CDF sampling; reused buffers; RNG residency without mid-decode clears
- Files: sampling module
- Test: distribution smoke (temp 0 golden, temp>0 statistical); no mid-decode clear in long run
- Target: tg, AC8, R1.4, R3.6
- Commit: `spec/rust-metal-engine: T4.3 sampler`

### T4.4 — MTP shared-KV integration
- [!] **BLOCKED by user gate (2026-09-23): strict hold — no MTP until plain decode reaches 18 t/s + 100% coherence.** Measured walls (T2.7/T3.8): ops ceiling ~9–12 t/s, bandwidth floor ~17 t/s → plain 18 unreachable; gate held anyway per user decision. Same status for T4.5, T4.6, T4.7 (n-gram also refused: "No drafter — same-algorithm only")
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
- [x] Done — `crates/engine/src/kv_cache.rs`: sliding layers keep a **rotated contiguous window** (a full window drops the oldest row and shifts; 8 KB memmove per layer per token), full layers keep everything, K=V honoured through the existing `k_eq_v` branch in the new single-row `layer_step`. Full layers error instead of silently dropping context when the allocation runs out
- Why it was on the critical path: the GPU linear path is `rows == 1`, and without a cache every prefill/decode call was multi-row, so **nothing** used the GPU (DE29). With the cache, prefill pushes the prompt once and every decode step is one row
- Measured (release, real artifact, `--metal --kv-cache --temp 0`, 6-token prompt, 4 generated): **decode 3.64 t/s** (was ~0.18 t/s naive) with **3699/3699 linears dispatched to the GPU** and `redone_fraction 0.9` instead of quadratic; resident KV **9.0 MB at 9 tokens** (≈ 1.0 MB/token: 84 KB/token sliding capped at the 1024 window + 41 KB/token full)
- Verified: `cargo test` (14 new tests) — cache rotation/window/reset/footprint/row-width, `cached_forward_matches_naive` (same argmax, logits within 0.05 after softcap), `cached_prefill_is_chunk_invariant` (bit-identical), `cached_decode_step_matches_naive_prefix`, `cached_step_uses_absolute_position`; and on the real artifact `real_artifact_cached_fixture_and_gpu_slow` reproduces the recorded naive fixture exactly — argmax **236773**, logit **29.9782** — with 411 GPU linears (the bug's signature was `gpu_linears: 1`)
- Files: `crates/engine/src/{kv_cache,gemma4,generate,lib}.rs`, `crates/cli/src/{args,main,report}.rs`, `crates/kernels/src/attn.{metal,rs}` (position bases)
- Target: AC6 memory, R5.2
- Commit: `spec/rust-metal-engine: T5.1 hybrid kv + cached decode`

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
