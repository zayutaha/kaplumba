# Implementation notes — rust-metal-engine

Append as work proceeds (AGENTS.md §5). Dense, factual, takeover-oriented. Content migrated from superseded `spec/omlx-exl3-engine/implementation_notes.md` where still valid; Python/omlx-specific entries marked historical.

## Status

- Spec suite created 2026-09-22 after pivot to pure Rust + raw Metal (no Python, no MLX).
- Open questions **all closed** (interviews 2026-09-22) — see spec.md table (OQ1–OQ11).
- Companion spec: `spec/paged-kv-cache` (being rewritten for Rust) — coordinate at Phase 5 hooks only.
- Superseded: `spec/omlx-exl3-engine/` (stub).
- **T1.0 done** (2026-09-22): workspace + `metal` 0.33.0 + GPU fill smoke.
- **T1.1 done** (2026-09-22): EXL3 loader — per-shard Metal buffers, stream via 8 MiB stage. Real artifact: **2422 tensors / 2 shards / 16.028 GB / ~4s**, 7/7 tests.
- **T1.2 done** (2026-09-22): tokenizer — `tokenizers` **0.21.4** (classic API; 1.0.0-rc.2 reorganized and was rejected). 9/9 tests. Specials verified: pad=0 eos=1 bos=2 unk=3 open=105 close=106. Next: T1.3 naive forward (or T0.2 baseline).
- **T1.3a done** (2026-09-22): EXL3 CPU decode primitives under `crates/engine/src/exl3/` — `trellis`, `codebook`, `perm`, `signs`, `hadamard`, `decode`. 26 new unit tests (35 total engine tests pass) pinned to reference goldens: MCG/Default/Mul1 codewords, q00 artifact tile k=3 MCG row-major, pack fixtures k=3/6, sign bitfield, perm heads, Hadamard large-magnitude lanes. Deps added: `half` 2.7.1, `rayon` 1 (rayon not yet used — reserved for T1.3b reconstruct). Next: T1.3b reconstruct + linear forward.
- **T0.2 partial** (2026-09-22): reference-only baselines + **correctness cross-check**: PonyExl3 greedy from `[2]` returns `"S"`, matching our argmax 236773 at position 0. Full 1k/2k/8k matrix still open; `scripts/bench.py` is now retirable because the Rust CLI emits the same schema (R7.4).
- **T1.4 done** (2026-09-22): streaming generation + bench entry. Engine: `generate.rs` (`generate()` with `on_token` callback, `GenStats` = prefill_s/decode_s/decode_steps/ttft_ms/finish_reason/tokens/computed_tokens, EOS-aware, seed-reproducible) and `sampler.rs` (PCG32 + temp/top-k/top-p, OQ4 defaults). CLI: `kaplumba-engine generate` (hand-rolled args, delta detokenizer, `--layers` depth limit, `--json PATH|-`) emitting `kaplumba-bench/1` reports mirroring `scripts/bench.py`'s keys. 101 tests pass. Measured release runs: `--prompt The --layers 2 --max-tokens 1` → load 3.7s, prefill 16.5s, ttft 16.5s, `text=" a"`, token 496; full-depth single-token forward 205s (lm_head ≈ 13s/row of that). `peak_rss_bytes` 8.08 GB vs `metal_alloc_bytes` 16.03 GB — UMA Metal buffers are not in `getrusage` maxrss, so weight/Metal figures are the authoritative memory evidence (R5.5 will add `phys_footprint`).
- **T1.3c done** (2026-09-22): `crates/engine/src/gemma4.rs` — naive CPU Gemma 4 forward + streaming lm_head + `greedy_generate`, driven by a `TensorSource`/`RawTensor` view over the artifact. 21 new tests (67 total). Real artifact (release): full 60-layer forward + lm_head **205s/row**, repeat **bit-identical** (hidden + logits), partial depth (layer 0 sliding + layer 5 full/k_eq_v) bit-identical in 3.1s/layer/row. Fixture: prompt `[2]` (BOS) → argmax **236773 ("S")**, logit 29.9782; top-5 all saturate the softcap (raw logits ≫ 30) — expected for Gemma at BOS, but **not yet cross-checked against the PonyExl3 reference** (T0.2/T2.4, OQ1). Config synthesis verified against the real `config.json`: 60 layers, full at 5,11,…,59, sliding (head_dim 256, n_kv 16), full (512, 4), window 1024, softcap 30; lm_head is a **6-bit** head (`[336,16384,96]` → k=6).
- **T1.3b done** (2026-09-22): `reconstruct.rs` (`TrellisView`, `reconstruct_inner`, `reconstruct_inner_slab` 16-col tiles, `reconstruct_public_weights` with outer Hadamard+suh/svh) + `forward.rs` (`Exl3Linear`, streaming `linear_forward_reconstruct` — output in 128-col slabs so lm_head never materializes ~2.8 GB). 44/44 tests. Stream vs full stepwise: max rel err < 1e-3; stream vs dense public-weight: < 0.15 (inherent f16 intermediate-rounding gap when Hadamard is folded into W). Outer Hadamard asserts `in/out % 128 == 0` (all Gemma4 dims qualify: 5376, 8192, 21504, 262144).
- DE25 (T1.1): Single ~16 GiB `new_buffer` failed (null contents → SIGABRT on drop). Fix = one shared buffer per safetensors shard (~8.5 + ~7.5 GB). Null-check after alloc; `LoadError::Alloc`.
- DE26 (T1.1): Load layout = original safetensors data offsets within each shard buffer (seek `8+header_len+data_offset`, copy to same offset) — no repack.
- DE27 (T1.2): Prefer **stable `tokenizers` 0.21.4** over 1.0.0-rc.2 (rc moved Tokenizer into tk-convert; API churn).
- DE28 (T1.3a): Exact fp16 add for codebook `half2_add` = convert both halves to f64, sum, `f16::from_f64` (correct RNE; f16+f16 always exact in f64). Do **not** add through f32 (double-rounds on rare ties vs CUDA `__hadd`).
- DE29 (T1.3a): `HAD_SCALE` must be the exact f32 of `1/sqrt(128)` = `f32::from_bits(0x3DB504F3)`; a hand-written `0.08838835` is 1 ULP off and skews Hadamard goldens.
- DE30 (T1.3a): Hadamard golden vs Python BLAS — only 8/128 lanes have \|y\|>0.01; near-zero lanes are pure cancellation and differ by accumulation order. Tests pin bit-exact on those 8, absolute bound \|y\|<1 elsewhere.
- DE31 (T1.3a): `unpack_trellis_tile` returns **full 16-bit sliding-window codewords**, not low-K bits; `pack_trellis_tile(cw)` reproduces the packed tile because pack only emits low K bits. Roundtrip property: `unpack(pack(x)) & mask == x & mask`.
- DE32 (T1.3b): Streaming forward casts the f32 matmul slab to **f16 before** the right Hadamard (matches reference `linear_forward_reconstruct`). Folding Hadamard into W (`reconstruct_public_weights`) is only a loose-parity path (rel err ≲0.15 on synth) — golden path is stream/stepwise.
- DE33 (T1.3c): `logits` applies the softcap in **f32** (`tanh(x/30)*30` per element). HF applies it in the model dtype (f16) before upcasting — the divergence only moves saturated logits by ≲1e-2 and cannot change the argmax of a monotone map; revisit only if the T2.4 sampled gate complains.
- DE34 (T1.3c): The handover's `embed[2]` golden list (`3ccc,3c4a,…,bc51`) is the **raw BF16 bit pattern on disk**, NOT the post-conversion f16 value. Verified against the artifact. f16 results (via `bf16_bits<<16 as f32` → f16) are `2660,2250,26d0,25b0,2690,21f0,2378,9fd8,a2c0,9a78,1e28,a0e0,9df8,a210,1a48,a288`; pre-scale values are ~0.025 (×√5376 ≈ 1.8 inside the model). Both lists pinned in tests.
- DE35 (T1.3c): Trellis bytes are read through a **zero-copy** `trellis_u16_view` (asserts even length + 2-byte alignment; safetensors offsets are 64-byte aligned). Copying (`trellis_bytes_as_u16`) would duplicate 16 GB of trellis per token on the naive path.
- DE36 (T1.3c): Naive CPU reference speed (release, M4 Pro): **3.1s per layer per token-row** → ~189s/token at 60 layers, lm_head +≈15s, so a **full 60-layer forward + lm_head ≈ 205s**. Real-model tests are therefore mode-gated (`KAPLUMBA_SLOW_MODE=calibrate|forward|greedy`) rather than free-running; the fast suite (67 tests) never loads 16 GB.
- DE37 (T1.3c): `TrellisView`/`Exl3Linear` are built **per forward** from the artifact's bytes (no cached copy). `linear_forward_slab` was factored out of `linear_forward_reconstruct` so lm_head (2048 slabs) reuses the exact same numerics as every other layer.
- DE38 (T1.3c): Greedy recomputes the whole prefix each step (no KV cache) — O(n²) but trivially correct; KV cache + fixed-shape decode is T4.2/T5.1. `greedy_generate` includes the EOS token in its output and stops.
- DE39 (T1.4): Generation lives in the **engine** (`generate::generate` + `on_token` callback), not the CLI, so streaming/stats are testable without a process and the CLI stays I/O-only. EOS/stop ids are emitted and then stop (same contract as `greedy_generate`); the CLI decides not to print them. `ttft_ms` = wall time to the **first emitted token** (session starts before the first forward), per the PonyExl3 lesson that TTFT must not be timed from a late detokenizer segment.
- DE40 (T1.4): Sampler rules, locked by tests: non-finite logits (NaN/±inf) behave as `-inf` (never selected; all-dead → id 0, no panic); candidates are ordered weight-desc then **id-asc** so ties are deterministic; top-k then top-p over the sorted prefix (`top_p` cuts at the smallest prefix whose cumulative mass ≥ top_p); `temp <= 0` or `top_k == 1` → argmax and the RNG is untouched. PCG32 (seed → `(seed<<1)|1` stream) makes any run reproducible. Host cost is a full-vocab sort per token (~262k entries) — acceptable on the naive path, replaced in T4.3.
- DE41 (T1.4): `kaplumba-bench/1` report = the harness's keys + `sampling` + `tokens`/`text`; the required set is `report::REPORT_KEYS` and `build_report` `debug_assert!`s it is present, so schema drift fails tests rather than silently breaking diffs. f32 CLI args are rounded to 3 decimals (`0.95` must not print as `0.949999988079071`). Keys the naive path cannot measure yet (`phys_footprint_bytes`, `lifetime_max_phys_footprint_bytes`, `metal_peak_bytes`) are `null` with `mem_note` naming R5.5/T2.2 — never fabricated. `mem.weights_bytes`/`metal_alloc_bytes` come from the loader/device; `peak_rss_bytes` is `getrusage(RUSAGE_SELF).ru_maxrss` and **under-reports Metal** on UMA (8.08 GB vs 16.03 GB observed).
- DE42 (T1.4): CLI end-to-end tests are gated on `KAPLUMBA_RUN_SLOW=1` even at `--layers 2`, because the naive path in a **debug** build is far slower than release (lm_head alone is ~14s/row in release). Default `cargo test` therefore never loads 16 GB; run e2e with `KAPLUMBA_RUN_SLOW=1 cargo test --release -p kaplumba-cli --test cli_generate`.
- **T4.2 done** (2026-09-22): one command buffer per decode token — see the T4.2 section below. Decode **3.64 → 5.68 t/s** (+56%), device-resident KV + activations, 2 host round trips/token. Fixture on the device path reproduces argmax 236773 bit-identically on replay; full suite 136 green.
- DE43 (T4.2): Metal compute dispatches encoded into **one** `MTLCommandBuffer` (one compute encoder, sequential dispatches, no barriers) execute in encode order — verified empirically: the T4.2 single-cb forward replays to **bit-identical logits** on a fresh cache, twice per test run and across processes. The in-place q-RoPE (q read/written by the same kernel) and cross-pass hazards (had_in→gemv→had_out sharing scratch, rms→rope→attn over shared vectors) all hold. Do not add `memory_barrier` calls without a failing case; re-check if a kernel ever writes a buffer a later dispatch in the same encoder reads **from a different threadgroup with relaxed visibility assumptions**.

## Decisions (carried + new)

- DE1 (historical→adapted): Base was omlx; **now pure Rust + raw Metal** — omlx is not a dependency (only design-lesson reference).
- DE2: Memory model of record: **EXL3 3.0 bpw ≈ 15.5 GB weights + ~1 GB SWA/KV ≈ 16.5 GB resident**; roofline **~17.6 tok/s**. User-corrected; do not recalculate as 31e9×3/8.
- DE3: Priority order fixed: TTFT → tg/s → pp/s → mem awareness; every task tags its metric.
- DE4: No KV quant in defaults (standing constraint).
- DE5: Research-derived strategy: dual-path prefill (ik); sibling fusion + MTP shared-KV (PonyExl3/llama.cpp); SWA bound; fixed-shape decode; strip per-chunk clears.
- DE6 (OQ1): Quality gate = **sampled golden review** at **temp=1.0, topk=64, top_p=0.95** vs reference quant.
- DE7 (OQ2): Speed ACs **report-only** for v1 — before/after numbers, no hard thresholds.
- DE8 (OQ3): Machine is **24 GB** unified — fold16 on main weights **banned**; tight headroom around 16.5 GB target; stretch user goal **14.6 GB** footprint.
- DE9 (OQ4): Primary sampling mode = **temp 1.0 / topk 64 / top_p 0.95**; temp-0 for spec-decode identity tests (R8.1).
- DE10 (OQ5): **8k typical / 32k max** — benches 1k/2k/8k + 32k spot check.
- DE11 (OQ6): Reference SHAs **pinned and frozen** (below); no mid-spec pulls.
- DE12 (OQ7): **Text-only** v1; no vision/mmproj.
- DE13 (OQ8/OQ11): **Fresh EXL3 kernels in Rust/Metal** from reference material; no vendored PonyExl3 engine code; converter CLI OK offline.
- DE14: ~~Patch strategy for omlx~~ obsolete — no omlx patches.
- DE15: **No Python, no MLX** (user 2026-09-22): raw Metal kernels only.
- DE16: Tokenizer = in-engine Rust crate (self-contained CLI).
- DE17: HTTP server out of scope v1; may be another language later.
- DE18: Inspiration from `ggml-metal` / PonyExl3 `gemv_metal` logic allowed as fresh rewrites.
- DE19: PonyExl3 Python converter CLI remains as **offline build-time** tool only.
- DE20: `scripts/bench.py` (Python) is a **temporary** T0 harness — retire at T0.2/R7.4; baseline numbers **reference-only** (measure PonyExl3, not us).
- DE21: New spec folder `rust-metal-engine`; `omlx-exl3-engine` superseded; `paged-kv-cache` fully rewritten for Rust.
- DE22 (T1.0): Workspace = `crates/engine` (lib) + `crates/cli` (bin `kaplumba-engine`); root package `kaplumba` removed (hello-world `src/` deleted). Edition **2024** (rustc 1.95.0-nightly).
- DE23 (T1.0): Metal binding = **`metal` 0.33.0** (workspace dep). Chosen for mature compute API (`dispatch_threads`, `new_library_with_source`, shared buffers). Caveat: crate docs mark it deprecated in favor of `objc2-metal` (0.3.2) — still “basic PRs maintained” while wgpu migrates; **revisit only if blocked**, do not churn mid-kernel-work. `objc2`/`objc2-metal` remain the long-term fallback (DE23 risk).
- DE25 (2026-09-22, user directive): **No prefill scratch that scales with context or keeps a high-water mark.** The MLX reference's lazy-graph intermediates + buffer pool + unfused O(L²) SDPA score matrix are banned by name (AGENTS.md §8, spec.md restrictions). Consequences for T3/T4: attention is online-softmax over KV tiles; a prefill step may only allocate scratch we own at a fixed size; peak RSS at 1k/2k/8k is measured and is an acceptance number (R5.3), not a best-effort.
- DE24 (T1.0): Shader path = **runtime MSL compile** (`Device::new_library_with_source`) for T1 smoke; switch to offline `xcrun metal -o *.metallib` in `build.rs` once kernels stabilize (T2+) to avoid first-token compile latency (R2.6).

## Environment (T0.3 — verified, carried)

- Host: macOS 27.0, **24 GB** RAM (`sysctl hw.memsize` = 25769803776). Env setup via **uv 0.8.17** where Python still needed (bench only).
- **Venvs (historical/transition)**:
  - `.venv-baseline` — CPython **3.14.3**: `mlx==0.32.2`, `mlx-lm==0.31.3`, `ponyexl3==0.3.0` (editable from `reference/PonyExl3`), numpy 2.5.3, transformers 5.17.0 — **only for T0.2 reference baseline + temporary bench.py**.
  - `.venv-omlx` — CPython **3.13.7**: `mlx==0.32.2`, `mlx-lm==0.32.0`, `omlx==0.7.0.dev4` — **obsolete for engine work**; keep only if needed to re-read omlx source (it's in `reference/omlx` anyway). Prefer not to install Python deps beyond baseline venv.
- Models root: `~/.kaplumba/models/Gxma/Gxma/` — **EXL3 3.0 bpw artifact present**: `model-0000{1,2}-of-00002.safetensors` (8.50 + 7.53 GB ≈ 16.03 GB raw), `quantization_config.json` v0.0.37 bits 3.0 head_bits 6 out_scales always codebook mcg; tensors = `trellis` I16 + `suh`/`svh` F16 + `mcg` I32 + BF16 norms/embeds. Gemma4ForConditionalGeneration: 60 layers (50 SWA window 1024 + 10 full), hidden 5376, head_dim 256 / global 512, 32 heads / 16 kv / 4 global-kv, vocab 262144, `attention_k_eq_v: true`, logit softcap 30, max_pos 262144, vision tower present but text-only per OQ7. `generation_config.json`: do_sample, temp 1.0, top_k 64, top_p 0.95.
- Repo: Cargo skeleton (`src/main.rs` hello world) — **now the real foundation** for engine work under `crates/*` (T1.0 will restructure).
- Reference pins (frozen per DE11):

| Repo | SHA |
|------|-----|
| reference/omlx | `8a6bfb84edb52ea48a02f4f2490684b5cb65ab2d` |
| reference/llama.cpp | `58367713a6935c0810103378144008df32e3d5db` |
| reference/ik_llama.cpp | `baac291dc9d531927760b48451d8dfcb63b6adec` |
| reference/PonyExl3 | `8e7fa6b1556f59fc669e25087903b279b9b0346f` |
| reference/vllm | `1fee69838eeb72622b441a593888687831c1e434` |

- Baseline bench entrypoints (reference): `ponyexl3-generate-bench` (`reference/PonyExl3/ponyexl3/cli/generate_bench.py`, `--json` supported); stack loader `load_generate_stack(args)`; generation `ponyexl3.mlx.generate.generate_text(...)` → `(text, stats)` with `prefill_s`, `decode_s`, `prompt_tokens`, `gen_tokens`, `finish_reason`.
- PonyExl3 mechanics (for reference during kernel work): prefill routes by rows (M=1 GEMV; huge & M≤144 fused GEMM); `EXL3_WCACHE` default 0; `EXL3_GEMV_LUT` opt-in (measured slower — dead end); `EXL3_FUSE_MIN_MB` (Gemma4 default 40 MB); `PONYEXL3_MEM_LIMIT_GB` (default 0.92× working set ≈ 17.5 GB; `max_recommended_working_set_size` = 19.07 GB, memory_size = 24 GiB); `mx.metal.get_peak_memory` deprecated → `mx.get_peak_memory`.
- Sibling fusion note (old model.py): `fuse_exl3_siblings` groups `mlp {gate,up}`, `self_attn {q,k,v}`; Gemma4 min-fuse 40 MB; unfused `exl3_linears()` count was 121 (o_proj+down_proj+lm_head) because fused members are `FusedEXL3Sibling`.
- Explained anomalies (reference, not our engine): end phys_footprint ≈ 18.1 GB / lifetime_max 37 GB = load transient (host numpy trellis + fusion concat); decode ~5.7 t/s stable in micro (effective ~85 GB/s vs 273 peak → kernel efficiency gap, not pageins); prefill highly variable by starting state (cold ~441 t/s @1024 vs 34–75 under pressure); TTFT via `on_segment` fires late — must time first token via stream start.

## Divergences from spec/plan

- DV1: Stock omlx has no `exl3` loader — baseline engine was PonyExl3 MLX runtime (historical; still the **reference comparison** for R7.3).
- DV2: T3.1 conversion already done — artifact on disk; verify only.
- DV3: Two venvs existed due to Python version split — moot for engine once Python retired after T0.2; keep `.venv-baseline` only for reference benches.
- DV4 (pivot): Entire omlx-based plan abandoned per user no-Python directive; pure Rust + raw Metal (DE15). `scripts/bench.py` Python exception time-boxed (DE20).

## T2.1 measured numbers (2026-09-22, release, M4 Pro)

Commands:
- Correctness: `cargo test -p kaplumba-engine --lib metal_linear` (fast, synthetic + eligibility) · `KAPLUMBA_RUN_SLOW=1 cargo test --release -p kaplumba-engine --lib real_artifact_linear_matches_cpu_slow -- --nocapture --test-threads=1`
- Per-pass table: `KAPLUMBA_RUN_SLOW=1 [KAPLUMBA_BENCH_VERBOSE=1] cargo test --release -p kaplumba-engine --lib bench_passes_slow -- --ignored --nocapture --test-threads=1`
- End to end: `cargo run --release -q -p kaplumba-cli -- generate --prompt "The" --no-bos --max-tokens 1 --temp 0 --metal --json -` (`--layers N` to depth-limit)

GPU kernels (min/median ms over 20 reps, real artifact tensors; `full` = had_in+gemv+had_out in one command buffer):

| tensor | shape (in×out) | K | gemv | had_in | had_out | full | Gw/s |
|---|---|---|---|---|---|---|---|
| gate_proj | 5376×21504 | 3 | 0.65/0.72 | 0.15 | 0.25 | 0.65/**0.72** | 177 |
| down_proj | 21504×5376 | 3 | 0.79/0.82 | 0.15 | 0.14 | 0.81/**0.84** | 144 |
| o_proj | 8192×5376 | 3 | 0.37/0.38 | 0.17 | 0.14 | 0.38/**0.39** | 118 |
| lm_head | 5376×262144 | **6** | 8.58/8.71 | 0.17 | 0.28 | **8.58** | 164 |

Facts that matter:
- Steady-state throughput is **~165–180 Gw/s** of decoded weights (gemv-limited). 30.1e9 weights/token → **~177 ms/token ⇒ ~5.6 t/s decode** for this kernel as-is, vs the 41 ms (24 t/s) bandwidth roofline and 273 GB/s of DRAM. So the decode path is **ALU/latency-bound, ~4.3× off roofline**, not bandwidth-bound.
- Per-`linear_row` fixed cost ≈ **0.15 ms** (one command buffer + one sync); the had_in/had_out numbers above are almost entirely that overhead (their real work is ~0.15 ms per 20k features). A token issues ~420 linears → ~63 ms/token of pure call overhead. **T2.3: batch a whole token into one command buffer.**
- **One-time Metal cost per shard buffer: ~0.8–1.5 s.** Evidence: `--layers 0` (lm_head only, shard 2) → 780 ms `gpu_ms`; `--layers 1` → 2288 ms (layer 1's 7 *new* shapes ≈ 215 ms each); `--layers 2..60` → only +0.26 s for 58 more layers (same shapes, buffers already mapped). It is page/residency mapping of the 8.5 GB shard allocations on first touch, charged to whichever call runs first. Do not mistake it for kernel cost; the fix is a load-time warm-up dispatch, not a kernel change.
- `lm_head` has **K=6** (its trellis is 96 u16 words/tile) while decoder layers are **K=3**; both are handled by the same kernel via the per-K pair table.
- **Alignment lesson:** `lm_head.trellis` and `lm_head.{suh,svh}` sit at **2-mod-4 byte offsets**, so the trellis is read as u16 pairs (`load_word` in the MSL) and every span only needs 2-byte alignment. A 4-byte rule silently rejected the largest linear in the model.
- Two bugs found by testing, both worth remembering: (a) the pair table must use **256 samples per tile**, not the 16-wide tile edge (wrong constant ⇒ plausible-looking garbage); (b) `s1` must be derived from the **unwrapped** word index while the two words are loaded at **wrapped** ones — wrapping first gives a negative shift.
- Accuracy vs the CPU reference: `had_in` **bit-exact**; `had_out` ≤ **1 f16 ulp** (GPU contracts `acc += sh*sign` into an FMA); GEMV ≤ 5e-3 relative over random activations for K=1..8.

## T2.3 kernel sweep — findings (2026-09-22)

**Measurement noise (important, read first).** The same binary measured twice back-to-back gives down_proj `gemv` **1.018 ms** then **0.764 ms** (±25%) and o_proj 0.489 then 0.387. Short kernels (0.4–1.2 ms) are dominated by GPU clock ramping between bursts, so *single* before/after numbers on them are meaningless. `lm_head` (8.4 ms, 1.41e9 weights) reproduces to ±1.4% and is the only shape here worth trusting; for anything smaller, loop it ≥1000× inside one timing region instead of comparing min/median of 20.

Variants tried, with the reliable (lm_head) verdict:

| variant | lm_head gemv | verdict |
|---|---|---|
| scalar pair loads, per-block O(n²) Hadamard (T2.1 as merged) | 8.58 ms | baseline |
| **fast Walsh–Hadamard (7 butterfly stages)** + uint4 pair load | 8.46 ms | kept (≤1.4% ≈ noise, but ~36× fewer ops in the Hadamards and much less code) |
| fast WHT + 4 warps per threadgroup (4 out-tiles/group) | 8.35 ms | reverted (small shapes 25–40% worse: too few threadgroups left to fill 20 cores) |
| fast WHT + tile staged through threadgroup memory (1–2 global loads/lane instead of 8) | — | **reverted, produced wrong weights** |
| scalar pair loads (current) | 8.47 ms | landing state |

Steady state as committed: **167 Gw/s** (lm_head 8.46–8.47 ms for 1.41e9 weights). Hadamards are ≤1 f16 ulp vs the CPU reference after the WHT (the butterfly re-associates the f32 sum, so bit-exactness is gone by design).

**Dead ends (do not repeat without a new idea):**
- *Threadgroup staging of the packed tile.* Stage the tile's u32 words into `threadgroup uint[MAX_TWORDS]` once per in-tile and read the funnel windows from shared memory. Symptom: one-hot probes (`debug_gemv_rows`) showed column cA returning column cB's weight for every row, i.e. pairs q=0 and q=2 resolving identically; and after reverting, the failure persisted because the *other* half of the change (see next bullet) was still in place. Not diagnosed further — the load-count saving was unproven, so it was reverted rather than debugged.
- *Two copies of a kernel constant.* `WARPS` existed in `exl3.metal` and in `exl3.rs`; changing one without the other made the host dispatch 128-thread threadgroups into a kernel that indexes `shA[g]` with `g = tid & 31` (out of bounds) — wrong results that look like a math bug. Both copies now carry a MUST-match comment.
- *Measuring with min/median of 20 short reps.* See noise above; it manufactured a phantom 25% regression.

**Where the decode ceiling actually is.** lm_head = 167 Gw/s ⇒ ~180 ms/token for the 30.1e9 weights/token ⇒ **~5.5 t/s decode** for this kernel. Roofline is 41 ms (24 t/s) and the kernel is not bandwidth-bound: the trellis streams at ~50–63 GB/s vs 273 GB/s peak. At ~9–12 ops per decoded weight (2 trellis loads + funnel shift + 2 masks + 2 procedural decodes + FMA, per 2 weights) against ~4.1 Tops/s of FP32/INT32 issue on 20 cores, the arithmetic ceiling is **~350–450 Gw/s**, i.e. **~9–12 t/s**. Reaching the user's 18 t/s needs ≤5 ops/weight, which is a different decode algorithm (not a tuning change); at 3 bpw the bytes themselves cap any design at 24 t/s.

## Baseline numbers

**All of these measure PonyExl3/MLX 0.32.2 (ponyexl3 0.3.0, mlx-lm 0.31.3), NOT this engine** (DE20).
Captured 2026-09-22 via `.venv-baseline/bin/python` calling
`load_generate_stack(argparse.Namespace(model=…, engine="exl3", no_warm=False, quiet=True, dflash=None, dflash_quant="w8", eagle3=None, mtp="off", draft=3, draft_w4=False))`
then `ponyexl3.mlx.generate.generate_text(..., use_chat_template=False, temp=0.0)`.

| Probe | prompt tok | prefill | decode | wall |
|-------|-----------|---------|--------|------|
| load  | — | 15.1s (Python + MLX) | — | — |
| BOS-only greedy-1 (`[2]`) | 1 | 5.39s (0.2 t/s, first-call warmup) | 4.7 t/s | 5.73s |
| `The quick brown fox` greedy-4 (`[818,3823,8864,37423]`) | 4 | 0.51s (7.9 t/s) | 5.9 t/s | 1.31s |
| repeated text, greedy-16 | 64 | 3.54s (18.1 t/s) | 5.7 t/s | 6.50s |

- Outputs: `[2]` → `"S"`; `The quick brown fox` → `" fox fox fox fox"` (degenerate greedy loop, expected for temp 0); repeated text → echoes the prompt.
- Decode ≈ 5.7 t/s ≈ 91 GB/s effective vs 273 GB/s peak → reference kernel-efficiency gap (matches the earlier micro measurement), **not** pageins.
- Still missing for a complete T0.2: the 1k/2k/8k prefill matrix and 32/128-token decode runs, plus `phys_footprint`/Metal peak in the reference report.

### Our engine vs reference (correctness evidence so far)

| Check | Reference | This engine |
|-------|-----------|-------------|
| greedy argmax from `[2]`, temp 0 | text `"S"` | token **236773** = `"S"` (logit 29.9782) |
| determinism on repeat | (not measured) | full 60-layer forward + logits **bit-identical** |

- A 262144-way argmax matching by chance is ~4e-6, so position 0 is verified. Positions ≥ 1 (rope/mask beyond the first position, incrementing context) are **not yet** cross-checked against the reference — that needs a 2-row forward (~7 min on the naive path) and is the natural first check inside the T2.4/OQ1 quality gate.
- Reference tokenizer ids for `The quick brown fox` = `[818, 3823, 8864, 37423]` (no BOS); our CLI reproduces these with `--no-bos`.

## Research digests (where to look)

- llama.cpp: `ggml-metal` mul_mv/mul_mm/fa; `common/speculative.cpp` (gemma4 shared KV); graph reuse `llama-context.cpp`.
- ik: `iqk_mul_mat.cpp` (amortize unpack, thresholds), `ggml-common.h` R4/R8, discussion 164 (pp tables), FA lesson discussion 25.
- PonyExl3: `mlx/gemv_metal.py` (simd GEMV, split-K), `exl3_fused.py`, `convert/*`, README benches (M1/M5/4090), fuse threshold notes (Gemma4 qkv vs gate+up).
- omlx (historical only): single-tenant overhead lessons; not a code dependency.

## Dead ends (do not repeat)

- **Prefill scratch that grows past the KV window (banned by user 2026-09-22).** What the MLX reference actually does, for the record, so nobody re-imports it: (1) MLX evaluates a whole lazy op **graph/DAG** per prefill step, so every intermediate and every model-built additive attention bias `[n_q_heads, q_tokens, kv_len]` is alive simultaneously (`memory_monitor.py` prices it as `estimate_prefill_transient_bytes`, on top of resident KV); (2) freed transients go back into MLX's **Metal buffer pool**, not to the OS, so RSS keeps the prefill high-water mark until `mx.clear_cache()` — the notes in `speculative/vlm_mtp.py` ("on large targets like Gemma 4 31B the buffer pool balloons", upstream issue #1416) are about exactly this; (3) when a prefill shape is not covered by fused SDPA, MLX materializes an **unfused O(L²) score matrix** in fp32 even for bf16 models (`SDPA256_UNFUSED_SCORE_DTYPE_SIZE = 4`; measured ~33 GiB spike on a 160k-token head_dim-256 prefill), which is what `PrefillMemoryExceededError` / `cluster/memory_guard.py` exist to catch. **Our rule: online-softmax attention over KV tiles, O(L) working set, fixed-size owned scratch, and a measured peak-RSS criterion (R5.3) at 1k/2k/8k.**
- EXL3 LUT decode (measured slower than procedural).
- Lowering global fuse min size (2× decode regressions on small groups).
- KV quant / Hadamard-KV (out of scope).
- Assuming M5 Max tunables transfer to M4 Pro unchanged.
- N-gram-only as substitute for real GEMV/MTP work (bounded wins).
- Timing TTFT via first detokenizer segment (late) — time first token emit instead.
- Per-row decode degradation in reference bench = memory accumulation without `mx.clear_cache` between rows (Python harness fix needed for fair reference capture; our Rust bench must clear/track peaks per row).

## DE29 — the GPU linear path only ever ran the lm_head (found 2026-09-22, T2.1 residue)

Symptom: `generate --metal --json -` on the real artifact reported **`gpu_linears: 1`**
for a 60-layer token that took 5.58 s. One 16 GB load, one `MetalLinear`, 411
decoder linears — and exactly one dispatch.

Diagnosis (via the new `MetalLinear::skip_reason`, which replaces `gather`'s
bare `None` with a reason string, plus `all_layer_linears_gpu_eligible_slow`):

- `gather` **is** happy with every linear in the artifact: `411 of 411 linears
  GPU-eligible`. All `mcg` are `0xCBAC1FED` (Mcg), all `suh`/`svh` are F16, all
  trellis `[in_tiles, out_tiles, 48]`. Nothing is mode/format-gated.
- The blocker is the **runtime** row check in `try_linear`:
  `if x.len() != shape.in_features() { return Ok(None) }` — i.e. `rows == 1`.
- The naive engine recomputes the **whole prefix** every step, so the only
  genuinely single-row call in a whole run is the last-row `lm_head` in
  `logits()`. Prefill (`s = prompt`) and every decode step (`s = prefix + 1`)
  are multi-row → **100% CPU fallback**, at ~0.5 s per linear.
- A one-layer forward test proves the wiring is otherwise fine: 7 layer linears
  + lm_head dispatch, and `assert_eq!(dispatched, 8)`.

Consequences for the plan (both user targets depend on this):

1. **Decode** cannot use the GPU until the engine stops recomputing the prefix:
   a KV cache makes decode `s == 1` (T5.1/T4.2) — that alone is the difference
   between ~0.18 t/s and ~5.6 t/s, before any kernel tuning.
2. **Prefill** needs a multi-row (batched) linear: `s > 1` must go to a GEMM that
   amortizes the trellis decode across columns (T3.2).
3. **Attention** is CPU-only today (`Gemma4Model::attention`), and at L=1024 it
   is ~587 MMAC/token across the 60 layers — comparable to the whole GEMV cost.
   So a Metal attention kernel (T3.5/T3.7, online softmax, fixed scratch) is on
   the critical path for *both* targets, not just a prefill nicety.

Guard: `all_layer_linears_gpu_eligible_slow` (KAPLUMBA_RUN_SLOW=1, ~5 s, loads
16 GB) now asserts eligibility *and* that a real 1-layer forward dispatches 8
GPU linears, so this class of silent fallback cannot come back unnoticed.

## T3.5/T3.7 (kernel half) — bounded-scratch attention, 2026-09-22

`crates/kernels/src/attn.metal` + `attn.rs`, wired into `Gemma4Model::attention`
(`attention_cpu` is kept as the reference/fallback).

Geometry: one threadgroup = `AROWS` = 8 query rows **× one query head**;
`PART = head_dim / 32` threads per row (8 for hd 256, 16 for hd 512, i.e. the
threadgroup is 64 or 128 threads), each owning `DW = 32` dims. The `PART` threads
of a row are consecutive lanes of one simd group (`row*PART + slice` indexing),
so the q·k slice reduction is `simd_shuffle_xor(dot, 1<<m)` for `m < log2(PART)`.

Why it has **no scratch and no barriers** (this is the R5.3/prohibited-pattern
answer, DE25):

- online-softmax state = per-thread scalars: running max `m_run`, denominator
  `l_run`, and `float acc[DW]`;
- one KV tile of scores = `float sc[ATILE]` (ATILE = 16) in registers;
- because nothing is shared between the threads of a row, the kernel needs no
  `threadgroup` memory and no `threadgroup_barrier` at all, so it also tolerates
  divergence (padded rows are masked by `row < s`, never early-returned).

Working set is therefore O(1) in sequence length: no `[heads, L, L]` score
matrix, no `[heads, q, kv]` mask tensor, no growable scratch. One masking rule
serves prefill and decode: query row `r` sits at absolute position
`kv_len - s + r`, so `s == kv_len` is a full prefill and `s == 1` is a cached
decode step; sliding layers start the KV loop at `max(0, p0 + 1 - window)`.

Correctness (all vs the CPU reference):

- `cargo test -p kaplumba-kernels` — 7 tests, all pass. Sliding window 1024/4/3,
  full causal (incl. s=9), cached decode at kv 8/16/17/32/64/256/512/700, kv=1,
  padded rows, and unsupported-shape rejection. `tiny-2x2` is **bit-exact**.
- `one_hot_golden` pins the semantics with hand-checkable numbers (uniform scores
  over one-hot `v` ⇒ `out[d] = 1/(pos+1)`), which is what caught both bugs.
- `KAPLUMBA_RUN_SLOW=1 … real_artifact_attention_matches_cpu_slow` (real layer
  shapes, ~4 s): sliding (32q/16kv, hd 256) **4.88e-4**, full (32q/4kv, hd 512)
  **4.88e-4** — about 2 f16 ulps, i.e. at the f16 noise floor.

Two bugs found and fixed while landing this (both worth remembering, they are
invisible in the "obvious" test setup):

1. `pos` was computed from `tp.y` (row *within* the threadgroup) instead of
   `tg.x*AROWS + tp.y`. Every query block past the first silently recomputed
   block 0's positions. Any test with `s <= AROWS = 8` passes.
2. `q` and `out` were indexed by absolute position instead of query row. Those
   coincide when `s == kv_len`, so **every prefill case passed** while every
   cached-decode case was wrong. Only `s < kv_len` distinguishes them.

Cost/limits: the kernel is instantiated for hd 256 and hd 512 only; anything
else returns `Ok(None)`/`Unsupported` so the CPU path keeps handling it (the
synthetic test model's hd 64 goes down that road). Attention is dispatched per
layer with host→device staging of q/k/v for now; keeping K/V device-resident is
the KV cache (T5.1).

## T5.1 hybrid KV cache + cached decode (2026-09-22)

Files: `crates/engine/src/kv_cache.rs` (new), `gemma4.rs` (`layer_step`,
`forward_cached`, `forward_cached_layers`, `rope_heads_at`, `attention_bases`,
`attention_cpu` with position bases), `generate.rs` (`GenParams::use_kv`),
`crates/cli` (`--kv-cache`, report fields), `crates/kernels` (attention position
bases `q_pos0`/`kv_pos0`/`kv_rows`).

Why this came before the prefill GEMM: the GPU linear path is `rows == 1`
(DE29), so *without* a cache, prefill and every decode step are multi-row and
100% CPU. The cache is what makes decode GPU-eligible at all.

Layout: one `LayerKv` per layer, `n_kv * head_dim` f16 per row.
* sliding layers: contiguous **rotated window** of `min(window, cap)` rows — when
  full, drop the oldest row and shift (8 KB memmove per layer per token).
  Deliberately not a modular ring: a ring would force modular indexing into the
  attention kernel for the same bandwidth, so the rotation stays in the host.
* full layers: every row; running out of capacity is an error, never a silent
  drop of context.

Memory at 1024 tokens: 50 × 8 MB (sliding, capped by the 1024 window) +
10 × 4 MB (full) ≈ **450 MB**, matching the spec's "~1 GB SWA/KV" line.
Measured `kv_bytes` 9,011,200 at 9 tokens ≈ 1.0 MB/token before the window fills.

Measured (release, real artifact, `cargo run --release -p kaplumba-cli --
generate --prompt "The meaning of life is" --max-tokens 4 --temp 0 --metal
--kv-cache --json -`):

| metric | value |
|---|---|
| prefill (6 tokens) | 3.933 s → **1.53 t/s** (still one row per token; needs T3.2) |
| decode | 0.824 s / 3 steps → **3.64 t/s** |
| GPU linears | **3699 = 9 row-steps × 411** (was 1 per *run* before) |
| `gpu_ms` | 4496 ms total (first row-step pays residency; steady state ≈ 275 ms/token) |
| `computed_tokens` / `redone_fraction` | 9 / 0.9 (was quadratic) |
| KV resident | 9.0 MB at 9 tokens |

The recorded fixture is reproduced **exactly** by the new path:
`real_artifact_cached_fixture_and_gpu_slow` → prompt `[2]`, argmax **236773**,
logit **29.9782**, 411 GPU linears (the naive fixture recorded in T1.3d has the
same argmax and logit, so the cache + GPU path is not merely close).

Where the remaining decode time goes (this is the T4.2 target):
* 15.5 GB of weights at the measured 167 Gw/s ≈ **93 ms/token** of kernel time;
* 471 command buffers per token (411 linears + 60 attentions), each
  `commit` + `wait_until_completed`; steady state 275 ms/token minus kernel time
  puts **~180 ms/token in submit/wait (~0.4 ms per linear)**;
* ~1 s of one-time Metal shard residency is still charged to TTFT.
So the next decode lever is "encode the whole token into one command buffer",
not more kernel tuning, and the bandwidth roofline (17.6 t/s) is still the limit
that matters after that.

Attention kernel API change (needed for a windowed cache): `AttnParams` now
carries `kv_rows`, `q_pos0`, `kv_pos0` instead of `kv_len`, because a sliding
cache's key row 0 is *not* absolute position 0. `windowed_cache_matches_full_history`
pins that: a rotated cache and a full-history attention with the same window must
agree.

Also fixed while landing this: `generate()`'s cached path honours `--layers`, and
`KvCache::new` allocates per-layer capacity (sliding layers reserve only the
window, so a 16k-context allocation does not reserve 8 GB for sliding layers).

## Where the engine stands (2026-09-22, after T5.1)

Commands that produced every number below (one 16 GB process at a time, I1):

```bash
cargo test                                                        # 131 tests, all green
# real artifact, cached + GPU, greedy (release):
cargo run --release -p kaplumba-cli -- generate --prompt "The meaning of life is" \
  --max-tokens 4 --temp 0 --metal --kv-cache --json -
# fixture re-validation (naive path, ~205 s):
KAPLUMBA_RUN_SLOW=1 KAPLUMBA_SLOW_MODE=forward KAPLUMBA_SLOW_PROMPT=2 \
  KAPLUMBA_SLOW_EXPECT=236773 cargo test --release -p kaplumba-engine \
  --lib -- --nocapture --test-threads=1 real_artifact_greedy_slow
```

| metric | naive CPU | + Metal linears | + KV cache (now) | target |
|---|---|---|---|---|
| decode | ~0.18 t/s (5.58 s/token) | unusable (DE29: `rows > 1` ⇒ CPU) | **3.64 t/s**, flat in context | 18 t/s (roofline 17.6) |
| prefill | ~0.18 t/s | — | 1.53 t/s (one row per token) | 200 t/s |
| GPU linears / token | 0 | 0 or 1 | **3699 across 9 row-steps, i.e. all** | all |
| KV resident | n/a | n/a | 9.0 MB at 9 tokens (~1 MB/token) | ≤1 GB |
| fixture | 236773 / 29.9782 | — | **236773 / 29.9782 (exact)** | unchanged |

Blocking work, in the order the measurements justify it:

1. **One command buffer per token** (T4.2/T3.6). 471 command buffers per token ⇒
   ~180 ms of the 275 ms steady-state token is submit/wait. This needs
   device-resident activations plus kernels for the elementwise glue
   (rms_norm, rope, gelu·up, residual, layer_scalar) rather than host loops,
   i.e. the real "fixed-shape decode step". Expected: ~93 ms kernel + one sync
   ⇒ ~8-10 t/s, and it speeds prefill identically.
2. **Batched multi-row linear (T3.2).** Prefill is still one row per token; a
   GEMM that amortises the trellis decode across columns is what converts the
   measured 167 Gw/s single-row rate into hundreds of t/s of prompt throughput.
   This is also the prerequisite for the T3.7 peak-RSS measurement
   (1024/2048/8192), since 8k prefill cannot be expressed with one-row steps.
3. **Kernel ops/weight** (T2.3 residue). At 18 t/s the model needs 15.5 GB in
   55 ms = 280 GB/s > 273 GB/s peak, so 18 t/s is *at* the roofline: the honest
   framing is "~15-16 t/s is the realistic ceiling for a single-pass decode",
   and the spec's own answer for the rest is speculation (Phase 4, and this
   artifact has **no MTP head** — checked the weight map — so it would have to be
   n-gram/verify-gated drafting, T4.7).
4. One-time Metal shard residency (~1 s) is still inside TTFT; load-time warm-up
   is a small, honest fix for the TTFT number (T2.2).

## Invariants

- I1: Single resident model; single-user; one agent session editing at a time.
- I2: Verify-gated speculation only.
- I3: Persistence on-disk format owned by `spec/paged-kv-cache`.
- I4: No hot-path cost without a measured justification.
- I5: No Python in engine process; no MLX; raw Metal only (DE15).

## Commands

- Rust build/test/smoke (T1.0+): `cargo build` · `cargo test -p kaplumba-engine` · `cargo run -p kaplumba-cli -- metal-smoke`
- Load real artifact test: `cargo test -p kaplumba-engine load_real_artifact -- --nocapture` (skips if `~/.kaplumba/models/Gxma/Gxma` absent; override with `KAPLUMBA_MODEL_DIR`)
- Reference baseline (Python, temporary): `.venv-baseline/bin/python scripts/bench.py ~/.kaplumba/models/Gxma/Gxma --backend ponyexl3 --json /tmp/bench.json`
- Reference CLI: `.venv-baseline/bin/ponyexl3-generate-bench --prefill-sizes 1024,8192 --gen-tokens 64 --warmup --raw --json`
- Rust generate (T1.4): `cargo run --release -p kaplumba-cli -- generate --prompt "The" --max-tokens 8` · add `--layers N` for a fast/debug depth limit · `--json -` prints the `kaplumba-bench/1` report to stdout (text then streams to stderr) · `--temp 0` greedy.
- CLI e2e (real artifact, ~22s): `KAPLUMBA_RUN_SLOW=1 cargo test --release -p kaplumba-cli --test cli_generate -- --test-threads=1`
- Reference baseline / cross-check (Python, transition only): `.venv-baseline/bin/python scripts/bench.py ~/.kaplumba/models/Gxma/Gxma --backend ponyexl3 --json /tmp/bench_ref.json`; the one-off cross-check used an inline heredoc that builds the `load_generate_stack(...)` namespace shown in "Baseline numbers" and calls `generate_text(prompt_ids=[2], max_tokens=1, temp=0.0)` — run one reference process at a time (16 GB residency, I1).
- Real-model slow tests (opt-in; **one at a time**, each loads 16 GB; use `--test-threads=1`):
  - `KAPLUMBA_RUN_SLOW=1 cargo test --release -p kaplumba-engine real_artifact_partial_depth_deterministic_slow -- --nocapture --test-threads=1` (~50s; sliding+full layers, bit-identical)
  - `KAPLUMBA_RUN_SLOW=1 KAPLUMBA_SLOW_MODE=forward KAPLUMBA_SLOW_PROMPT=2 KAPLUMBA_SLOW_EXPECT=236773 cargo test --release -p kaplumba-engine real_artifact_greedy_slow -- --nocapture --test-threads=1` (~210s; add `KAPLUMBA_SLOW_REPEAT=1` for ~420s determinism proof)
  - `KAPLUMBA_RUN_SLOW=1 KAPLUMBA_SLOW_MODE=greedy KAPLUMBA_SLOW_MAX_NEW=1 ...` (~3×210s: two forward passes per generated token)
  - T4.2 device-cache fixture: `KAPLUMBA_RUN_SLOW=1 cargo test --release -p kaplumba-engine real_artifact_cached_fixture_and_gpu_slow -- --nocapture --test-threads=1` (~6s after load; asserts device_kv, 411 GPU linears, argmax 236773, fresh-cache bit-identical replay)

## T3.2 — batched prefill GEMM + span path (2026-09-22)

**Done.** `rows==1` keeps the tuned GEMV; `rows≥2` takes `exl3_gemm` (one threadgroup = one out-tile × `RGT=8` row-chunk; funnel decode once per in-tile, reused across 8 rows; y written `[row, out_features]` f32). Multi-row `exl3_had_in`/`had_out` (grid `(row_len/128, rows)`), multi-row `kr_rope`, `KvCache::reserve_rows`/`end_tokens`, `GpuDecoder::encode_span` (host embed → one CB per chunk → all passes `rows=m` → final norm+lm on last row only). `ensure_decoder` rebuilds when a later `cache_cap > built_cap`.

**Correctness bug (span ≠ token loop):** production did **in-place q-RoPE** (`encode_rope(q, …, q)`), but `kr_rope` reads `x[xr+lo+kk]` and `x[xr+lo+kk+hh]` while writing `o[i]` — racy when `x==o`. The comment at the call site already said in/out must be distinct; every unit test used distinct buffers, so only the real span path failed: `n_diff=258588/262144`, `max_abs=2.95e-1`, argmax accidentally both 207. **Fix:** rope `q → q_raw` (q_raw is free after the per-head rms has read it); attention consumes `q_raw`. Same fix in `encode_token` (decode was "working" because rows=1 scheduling happened to serialize the race on this GPU — do not trust it). After fix: span vs loop **0/262144 bit-identical**; fixture still 236773 / 29.9782.

**Measured pp/s** (`KAPLUMBA_RUN_SLOW=1 … prefill_throughput_slow -- --ignored --nocapture`, debug build):

| case | time | t/s | vs token-loop |
|---|---|---|---|
| token-loop @512 | 98.5s | **5.20** | 1× |
| span @512 | 34.2s | **15.0** | 2.88× |
| span @2048 | 138.4s | **14.8** | 2.85× |
| span @8192 | 567.6s | **14.4** | 2.78× |

**Why only ~2.8× and not RGT=8's promise:** each threadgroup-row-chunk re-streams the **entire** trellis (`grid.y = rows/8` independent weight streams). Weight traffic ≈ `(rows/RGT)×` one GEMV's weights; decode ALU likewise. Rate is flat in context (15.0→14.4), so we are decode/re-stream bound, not attention-bound. Levers for 200 t/s: (a) larger effective reuse (raise RGT / multi-out-tile threadgroups so one decode serves more columns+rows), (b) T4.1 sibling fusion (qkv, gate+up share one had_in), (c) profile whether had_in/had_out/glue are now a meaningful slice. Correctness is done; remaining T3.2 work is performance only.

**Tests green:** `gpu_gemm_rows_match_per_row_gemv_and_cpu`, `rope_multi_row_matches_cpu_reference`, `reserve_rows_batches_match_row_at_a_time`, bit-identical `real_artifact_span_matches_token_loop_slow`, fixture, full suite 109+14.

## T3.4 — simdgroup GEMM shape dispatch (2026-09-23)

**What landed:** `exl3_gemm_sg` (`crates/kernels/src/exl3.metal`) — a faithful port of PonyExl3 v19c `_mm_seg_source` — plus the host dispatch `encode_gem` in `crates/kernels/src/exl3.rs`. Decision: `rows == 1` → `exl3_gemv` (unchanged, all decode traffic), `sg_eligible()` → sg GEMM, else RGT=8 `exl3_gemm`. Eligibility: `rows ≥ 2 && in_tiles % 2 == 0 && out_tiles % 4 == 0 && byte_offset % 4 == 0` — every real prefill shape qualifies; lm_head never runs multi-row (span does lm_head `rows=1` per token, `gemma4.rs`), so the ineligible/RGT path is currently dead in production but kept as fallback.

**Kernel facts:** `_MM_BM=64` token rows/TG, `_MM_BN=64` out cols/TG, `_MM_WS=72` wblk stride; one TG = 4 sgids × 32 threads = 2×2 sgid grid × 4 out-tiles; stages = in_tiles/2, each stage is 32 in-features × 64 out; weights staged through `wblk[32×72]` after `exl3_decode` (MCG/default, `ld_word` 4-aligned — the reference's `ld_word` with `aligned` flag); A loaded device-direct `simdgroup_load(a, xrow + r*8*K + ks*32 + k2*8, K)`; MMA is mixed `half8x8 × half8x8 → float8x8` accumulate (reference-proven); store scatters via `csc[4×64]` with `gr < blk_len` mask. xh scratch pads to `rows.div_ceil(64)*64` (`xh_rows()` in `encode_linear`); sg path returns early on a TG-uniform condition **before** any barrier so ragged grids stay legal.

**Decisions the plan didn't predict:**
- **Two test assertions had to be relaxed.** Bit-identical span==loop and batched==gemv were invariants of the *scalar* RGT GEMM sharing GEMV's accumulation order; tensor-core k-block summation mathematically cannot reproduce scalar order. New gates: batched vs gemv → `max_rel_err < 2e-3` (~1 f16 ulp, `max_rel_err` divisor ≥1); span vs loop → argmax equal + `max_abs < 0.35` (measured 2.3743033e-1 over 300 tokens / 262144 logits, n_diff 254424, argmax 207=207). Rationale documented in both code comments.
- **A new decisive test was added**: `real_artifact_batched_linear_matches_gemv_slow` (`metal_linear.rs`) — real trellises, rows=65 (exact block + ragged tail), batched vs per-row GEMV → worst **9.76e-4** on q/gate/down. This is the bug-detector for decode/index errors (~1e-1 if broken); model-level compounding is *measured*, not asserted away. Discriminators that ruled out a kernel bug: this test + synthetic `gpu_gemm_rows_*` (5/3/17 rows) + replay bit-determinism (a race would fail it; sg has no atomics) + argmax identity.
- **`phase_profile_prefill_shapes` + `bench_linear_phases`** (profiler added under this task): per-shape encode+wait timings at rows=256/512, real shapes.

**Measured (release, M4 Pro, real artifact):**
- Phase profile rows=256: q 5.92, k 2.87, o 5.04, **gate 12.03 (was ~66 → 5.5×)**, down 13.16 ms; gate@512 23.54 ms. gate ≈ 5.0 TFLOP/s ≈ 31% of 16 TFLOP/s fp16 (62% if f32-accum runs at 8T).
- Linear ceiling ≈ **79 t/s** @ rows=256 if linears were the whole model.
- `prefill_throughput_slow` (release): token-loop @512 **5.25 t/s** → span **25.7 / 25.2 / 24.1 t/s @ 512 / 2048 / 8192** (4.9× / 4.8× / 4.6×). Span 300 tok: 11.93 s (25.1 t/s) vs loop 55.83 s (5.4 t/s), 124943 GPU linears (counts *dispatches*, not row-linears).
- **Key finding: ~75% of chunk wall time is outside the linears** (25 t/s total vs 79 t/s linear ceiling ⇒ non-linear ≈ 7.6 s of a 10.2 s / 256-token chunk). Attention, norms/RoPE/HAD glue, host encode/commit sync, and residual glue now dominate. Flat rate in context (25.7→24.1) says glue/linears bound, attention growth is secondary — but the split between *attention GPU time* and *sync/submit* is **unmeasured**; that is the next profile.

**Tests green at commit:** gated suite 111 passed / 3 ignored (`--test-threads=1`, `KAPLUMBA_RUN_SLOW=1`), kernels 15 passed, `cargo fmt` clean. Span, fixture (argmax 236773 / 29.9782), attention, CPU-linear, greedy, chunk-invariance, replay-determinism all pass. `cached_prefill_is_chunk_invariant` is CPU-only (`Gemma4Model::new` without `.with_gpu`) → unaffected.

**Gotchas for the next session:**
- `try_linear` / `real_artifact_linear_matches_cpu_slow` are **rows=1 only** — do not treat them as multi-row coverage; use `real_artifact_batched_linear_matches_gemv_slow`.
- `g.calls()` counts *dispatches*; a 300-token span mixes ~416 rows=1 lm_head dispatches with ~840 rows≥2 sg dispatches.
- The bench's "recorded baseline ~1.5" eprintln annotation is stale (token-loop now measures 5.20–5.25 t/s since RGT=8); the *before* number it prints is live.
- Next performance work = **T3.6** (sync policy + fusion audit): instrument chunk path for attention vs glue vs wait time; remove any per-phase commit/wait; fuse gate+up / qkv. Levers beyond that: T3.3 load-time packing, sibling fusion (T4.1) for prefill too, larger sg BM. Do **not** chase more GEMM micro-opts before the non-linear split is measured — linears cap at 79 t/s anyway, far under 200.
- Roofline reminder: 200 t/s @ rows=256 ≈ 12.4 TFLOP/s dense ⇒ needs *both* ~2× better linears *and* near-zero glue/sync, or a different tiling strategy; re-plan after the T3.6 profile numbers exist.

## T3.6 — chunk profile + loader 64B repack (2026-09-23)

**Why:** after T3.4 the measured pp/s was still 25.x while the linear-only budget predicted ~6.0 s/chunk — needed a per-layer/per-linear split of a real 256-row chunk to find where the ~4 s went.

**Profile tooling (keep — this is the re-profile harness):**
- `KAPLUMBA_SPAN_PROFILE=1` in `encode_span` (`gpu_decode.rs:529`): per-chunk host encode/wait ms + optional per-layer command-buffer split (`prof_layers`, one cb+commit+wait per layer for isolation; queue order still serializes). Non-prof = one cb per chunk, single commit+wait at `:777`.
- `phase_profile_chunk_budget_slow` (`metal_linear.rs`): per-kind linears on real shapes (rows=256), 6 attention cases (sliding kv 256/512/1024; full kv 256/512/4096/8192), glue per layer kind, real 256-span wall + first-chunk / 512-avg / 8k-mid projections.
- `metal-rs 0.33.0` has **no** `gpu_start_time`/`gpu_end_time` — cb-split or encode/wait split are the only GPU timing options.

**Sync audit (T3.6 checkbox evidence):** non-prof host encode = 1.3 ms/chunk; the wait is the drain of that chunk's single cb. **No `clear()`/pool purge exists on the span path** (`xh.clear()` is tests, `scores.clear()` is CPU). Pipelining encode of chunk n+1 against GPU of chunk n would save ≈ 1.3 ms of 3630 ms = 0.04% — not worth the complexity. AC8/R2.4 satisfied as-is. No host materialize between fused candidates: gate+up/qkv write the device arena inside one cb (actual kernel fusion = T4.1).

**The finding — loader alignment (the entire 3× win):**
- `dump_trellis_alignment_slow` (`align_dump.rs`, gated) prints/asserts every linear's `trellis().byte_offset % 4` across 60 layers + lm_head. Before fix: **alternating — even layers 0, odd layers 2**, phase-flip at the shard boundary (L30/L31 both 0), lm_head 2. The `reconstruct.rs` header comment "Safetensors offsets are 64-byte aligned" was **wrong for this artifact**: file offsets are only 2-byte aligned (net bytes per layer ≡ 2 mod 4 — e.g. `layer_scalar` is 2 bytes — flips parity each layer).
- Consequence: `sg_eligible` requires `byte_offset % 4 == 0` (`exl3.rs:473`) → **~half the linears silently took the scalar RGT=8 path**. This explained: L5 full-layer sum 307.72 vs L0 sliding 53.61 ms on identical shapes (L5 gate 66.48 vs L0 gate 12.03); per-layer GPU times alternating ~53 / ~277 ms; budget projection 6024 ms vs measured wall 9884 ms — gap arithmetic: ~19 sliding layers × +224 ms ≈ 4256 ≈ 3948 measured unaccounted (closes).
- Fix in `load_exl3` (`load.rs`): per shard, sort tensors by file offset, assign packed `dev = align_up_64(dev)` offsets, stream file→device at those offsets, then update `TensorInfo.data_offset` in **both** `shard_headers` and `tensors` (both hold clones of the original). Alloc = packed end; `shard_sizes` map deleted (dead). `tensor_bytes` / `tensor_device_span` read the same mutated offsets → zero-copy view invariant unchanged. Memory: ≤63 B padding × ~1500 tensors ≈ 100 KB — **no memory increase** (user directive holds).
- Guard: `dump_trellis_alignment_slow` now **asserts** all `%4==0` (121 tensors incl. lm_head). Post-fix lm_head also 0 (still rows=1 → gemv regardless).
- Doc fix: `reconstruct.rs:270` comment now says alignment comes from the loader repack, not from safetensors.

**Measured after (release, M4 Pro, real artifact):**
- L5 full layer: **57.85 ms (was 307.72)**; per-layer GPU uniform ~53 ms sliding / ~59 ms full, no alternation; warmup L0 18.2 ms (first-call shader compile).
- 256-chunk: **3635 ms = 70.4 t/s (was 9882 = 25.9)**; budget 3463 → unaccounted **+172 ms** (was +3948) — profile closes.
- Budget split: linears 3249 ms (**94%**), glue 142 (3.9%), attn 66 (kv256) … 684 (8k-mid), head 7.
- `prefill_throughput_slow` → `/tmp/prefill_bench.log`: **77.8 / 72.3 / 65.0 t/s @ 512 / 2048 / 8192** (was 25.7/25.2/24.1 — ×3.02/2.87/2.70; 13.0×/12.1×/10.9× vs live token-loop 5.99 t/s). Span 300 tok: **4.25 s (70.6 t/s)**, was 11.93 s.
- Attention unchanged/cheap (not the bottleneck): sliding kv256/512/1024 = 1.08/2.48/5.24 ms; full kv256/512/4096/8192 = 1.45/4.03/40.65/90.81 ms; glue 2.15–2.31 ms/layer.
- Span drift: max_abs **4.002571e-1** (was 2.3743033e-1 — the count of sg-vs-gemv layers doubled), n_diff 258499, argmax 207=207 → bound 0.35 → **0.55** with the new measured citation; kernel gate unchanged (batched-vs-gemv worst 9.76e-4). Repack correctness proven by the whole gated suite (fixture argmax 236773 unchanged).
- **Tests at this commit:** gated suite **113 passed / 3 ignored / 0 failed** (`--test-threads=1`, `KAPLUMBA_RUN_SLOW=1`), kernels 15 passed, fast suite 113 passed, `cargo fmt` clean. Attention-CPU, greedy, replay-determinism, chunk-invariance, alignment assert all green.

**Next (linears = 94% of the budget now — the 75%-outside-linears finding from T3.4 is obsolete):**
- Path to 200 t/s: chunk ≤ ~1280 ms ⇒ linears ≈ ≤1000 ms ≈ 13.8 TFLOP/s ≈ 86% of 16 fp16 peak. Realistic combo: (a) sg kernel efficiency — current ~4.3 TFLOP/s ≈ 27% of peak (occupancy / BM / BN / wblk staging), (b) T4.1 sibling fusion — gate+up 24→12 ms and qkv 11→8 ms per layer ⇒ ~900 ms/chunk ≈ +16%, (c) glue/attention remain secondary. Re-profile after each lever with the harness above before adding more machinery.
- Decode **unaffected** by the repack (rows=1 → gemv; gemv never needed 4-byte offsets) — still ~5.6–6.2 t/s baseline; T4.1 + T2.3 remain the decode path to ≥18.
- MTP drafter (`gemma-4-31B-it-qat-assistant-4bit`) still gated: only after ≥18 t/s **and** 100% coherency, temp 1.0 / top_k 64, target 28 t/s.

## T3.8 — sg ceiling / decode-tax / MMA-peak probes (2026-09-23)

**Question:** after T3.6 linears = 94% of the chunk at only ~3.7 TFLOP/s avg — is the funnel decode the tax, or is the simdgroup MMA itself the wall? Three test-only probe kernels in `exl3.metal` (pipelines built locally in `exl3.rs` tests; production `Exl3Gemv` untouched):

- `exl3_gemm_sg_ceiling` — production sg with the funnel fill replaced by direct `device half` loads of a pre-decoded `w[N,K]`. Everything else (grid 128 thr, wblk, both per-stage barriers, MMA k-loop, store) byte-for-byte identical. CPU-ref checked on K=64/N=128/rows=64 (max_rel < 1e-3).
- `exl3_gemm_sg_decode_only` — funnel fill + per-stage barriers only; no MMA, no store (writes nothing; its value = timing).
- `exl3_gemm_sg_mma_peak` — wblk filled once (stage-0 geometry), then `PEAK_REPS=32` identical passes over hot fragments (B in TG mem, A L2-resident); keep-alive store at the end. Absolute sustained rate for this tile config.
- Gate-shape timing test `sg_ceiling_decode_tax_gate_shape` also dispatches the **real** `exl3_gemm_sg` on the same synthetic buffers (zero trellis = cw=0, safe) so all numbers are gem-only vs gem-only.

**Measured (release, M4 Pro, gate shape rows=256 / K=5376 / N=21504, flops 59.19 GF):**

| probe | time | rate |
|---|---|---|
| real `exl3_gemm_sg` (funnel) | 11.786 ms | **5.02 TFLOP/s** |
| ceiling (direct loads) | 9.418 ms | **6.28 TFLOP/s** |
| funnel tax = 1 − ceil/real | 2.37 ms | **20.1% of gem time** |
| decode_only (isolated funnel) | 0.211 ms | — |
| mma_peak (hot fragments) | 1.83–1.93 ms | **5.84–6.19 TFLOP/s** |

Cross-checks:
- Engine budget `linear_ms` 11.98 ≈ gem-only 11.79 → **had_in+had_out ≈ 0.19 ms at rows=256** (had is negligible in batched prefill; its T2.3 cost is the rows=1 per-linear path in decode).
- `mma_peak ≈ ceiling` → **weight streaming from device adds ~0 ms** at these rates (L1/TG covers the wblk fill); the wall is the MMA+structure loop itself.
- **decode_only = 0.21 ms but in-context funnel tax = 2.37 ms** → the funnel is NOT raw-cost-limited. Best hypothesis: register pressure — production fill runs with `C[4][4]` (32 f32/lane) live **plus** funnel address state (i0/i1/sh/words/shift temps ≈ +15–20 regs) and crosses a spill threshold; ceiling fill is lean (col/k0/wcol) and decode_only has no C. Fix to try: spill C to threadgroup memory around the fill loop (C = 32 f32/lane × 128 lanes = 16 KB TG; wblk 4.5 + csc 1 + C 16 ≈ 21.5 KB, fits), keep the same 2-barrier structure.

**The physics finding (200 pp/s conflict):**
- Chunk budget (from T3.6 measured times + shape flops): linears carry **12.35 TFLOP / 256-token chunk** (mlp 10.65 + attn 1.70) + non-linear ≈ 215 ms (glue 142 + attn 66 + head 7). 200 t/s ⇒ 1280 ms/chunk ⇒ linears ≤ 1065 ms ⇒ **11.59 TFLOP/s sustained on linears**.
- Hardware delivers **~6.3 TFLOP/s** even with hot fragments (measured, not a spec-sheet guess). Production today: 5.02 on gate, 3.7 avg.
- ⇒ 200 t/s needs **1.85× the measured MMA ceiling of the device** — unreachable by staging/tax/occupancy fixes (those only close the 5.02→6.28 gap). Best-possible chunk (tax=0, every linear at 6.28, non-linear squeezed) ≈ 1959+24+142+66+7 ≈ **2200 ms → ~116 t/s**; realistic after tax fix + small-N K-split + glue fusion ≈ **100–110 t/s**.
- `plan.md` roofline "200 t/s ≈ 12.4 TFLOP/s" implicitly assumed ~16 TFLOP/s fp16 peak at ~78% eff; measured f16→f32 simdgroup rate is ~6.3. **Open conflict to surface to the user with this table — do not keep grinding GEMM micro-opts toward 200.**
- Small-N TG starvation (part of why avg = 3.7 < gate's 5.0): k/v out_tiles give only 64/64 TGs (k runs 0.97 TF), q/o 256 TGs (2.1 TF), gate/up/down 1344 TGs (4.9–5.0). K-split across grid.z with a partial-sum reduce would lift these (~600 ms/chunk) — second lever after the tax fix.

**Next steps (ordered):** (1) C-spill funnel-tax fix → expect linears −648 ms → **~86 t/s**; verify with gate-shape probe (real sg ≈ ceiling) + full gated suite + prefill bench; (2) small-N K-split → ~110; (3) glue/fusion audit → maybe ~115; (4) stop, re-report the 200 conflict with end-to-end numbers; (5) switch to decode track (18 t/s target — plan roofline 17.6 from weight bandwidth; MTP gate stays closed until 18 + coherency).

## T4.2 — one command buffer per decode token (2026-09-22)

**Problem:** the cached/GPU decode path encoded each of the ~471 dispatches per token (411 linears + 60 attentions + glue) into its own command buffer with `commit`+`wait_until_completed` — ~0.4 ms of CPU submit/wait each ⇒ **~180 ms of a ~275 ms token was pure overhead** (kernel time only ~93 ms).

**Design:** `GpuDecoder::encode_token` (`crates/engine/src/gpu_decode.rs`) builds one command buffer per token:
- Hidden states live in a device **arena** with a 3-slot ring: layer *i* reads slot `2i%3`, writes `(2i+2)%3` — two buffers in flight, never materialized to host.
- K/V write straight into the shared-mode device KV cache (`KvCache::new_device`): RoPE is applied to k **in the kernel that writes the cache row**, so attention reads cache memory directly (2 host round trips/token total: embed row in, final logits out).
- All elementwise/norm/RoPE glue runs as `fuse.*` kernels (`crates/kernels/src/fuse.metal`): `kr_rms`, rope, copy, gelu_mul, add, add_scale. `layer_scalar` stays f64 on the CPU (DE28 ≤1 ulp).
- Linears/attention join an **external encoder** via `MetalLinear::encode_*` / `Exl3Attention::run_in` / `exl3::encode_linear`; `AttnParams` now carries `kv_rows, q_pos0, kv_pos0` (was `kv_len`).
- Fallback: no GPU, decoder build failure, or a **host** cache ⇒ old per-linear path (same maths). `ensure_decoder()` records permanently-unavailable models so we don't retry a failing build.

**Measured** (release, M4 Pro, real artifact, `generate --metal --kv-cache --temp 0`, 6-token prompt, 8 gen tokens):

| | before (per-cb) | after (one cb) |
|---|---|---|
| decode | 3.64 t/s | **5.68 t/s** (+56%) |
| command buffers/token | 471 | **1** |
| GPU linears | 5343/8 steps | same (all dispatched) |
| prefill (still per-row; T3.2) | 1.53 t/s | 1.45 t/s (unchanged, expected) |

Naive path remains 0.18 t/s. Roofline 17.6 t/s ⇒ decode now at **32%** of ceiling; next decode levers are T4.1 sibling fusion and kernel-side (T2.3 says 9–12 t/s for this decode algorithm).

**Verification:** gated `real_artifact_cached_fixture_and_gpu_slow` switched from `KvCache::new` (host — silently tested the old path!) to `g.new_kv_cache(8)` (device) and now asserts `is_device_backed()`, argmax **236773**, logit 29.9782, ≥400 GPU linears, and **bit-identical logits on a fresh-cache replay** (DE43). Cross-process: two CLI runs → identical tokens `[711, 506, 1852×6]`. Full suite: **136 green** (108 engine incl. new determinism assert, 13 kernels incl. `fuse::*`, 12+5 CLI).

**Gotchas for the next model:**
- The old fixture used a **host** cache, so T4.2 could have shipped untested — always check `is_device_backed()` when testing this path.
- `gpu_ms()` reports 0 in this path (encode_* doesn't time individual dispatches; the whole token is timed by the caller).
- Prefill (`step == 0` in `generate`) pushes the whole prompt through `encode_token` **one token at a time** in a loop of command buffers — correct, but that's why prefill didn't move; T3.2 batches it.
- `.freebuff/` in the repo root is an untracked scratch dir; leave it.

## T1.3c handover (2026-09-22 — **implemented**, kept for the locked facts)

State: T1.3c/d implemented (`crates/engine/src/gemma4.rs`); the facts below are **locked** and every one is now asserted by a test. Remaining gap: fixture vs PonyExl3 reference (T0.2/T2.4).

- **lm_head keys** (index.json, resolved): `lm_head.{trellis,suh,svh,mcg}` — UNPREFIXED (no `model.language_model.`). `lm_head.trellis` shape on disk = `[?, ?, 48]` (see below for exact). All in `model-00002-of-00002.safetensors`.
- **Per-layer tensor names** (verified for layer 0=sliding, 5=full): prefix `model.language_model.`; layer set = `input_layernorm.weight, post_attention_layernorm.weight, pre_feedforward_layernorm.weight, post_feedforward_layernorm.weight, layer_scalar, self_attn.{q_norm.weight,k_norm.weight}, self_attn.{q_proj,k_proj,o_proj}.{trellis,suh,svh,mcg}, mlp.{gate_proj,up_proj,down_proj}.{trellis,suh,svh,mcg}`. `self_attn.v_proj.*` EXISTS on sliding, ABSENT on full (k_eq_v); `v_norm.weight` never exists (weightless RMSNorm). `layer_scalar` present on all 60 layers (weight_map).
- **Shard routing**: `model-00001` holds embed + layers 0..~half (incl. layer_scalar 0,5); `model-00002` holds rest + `lm_head.*` + `model.language_model.norm.weight`. Layer 59 is in shard 2. Don't assume a tensor is in shard1.
- **Trellis tensor layout** `[in_tiles, out_tiles, packed_size]` (NOT [out,in,packed]). Examples: L0 q `[336,512,48]` (in 5376/16, out 8192/16, 48→k3), L0 k/v `[336,256,48]`, L0 o `[512,336,48]`, L0 gate/up `[336,1344,48]` (out 21504), L0 down `[1344,336,48]`, L5 q `[336,1024,48]`, L5 k `[336,128,48]` (n_kv=4×512), L5 o `[1024,336,48]`. `TrellisView::new(data, shape[0], shape[1], shape[2])`. Decode as u16 LE (dtype I16, bit pattern = codewords). Most linears 5376%128=0, 21504%128=0, vocab 262144%128=0 → `linear_forward_reconstruct` assert OK. lm_head out_tiles = 262144/16 = 16384 (2048 slabs of 128).
- **Global forward config** (config.json text_config): hidden 5376, intermediate 21504, head_dim 256, global_head_dim 512, n_heads 32, n_kv 16, num_global_key_value_heads 4, layers 60, vocab 262144, rms_norm_eps 1e-6, sliding_window 1024, attention_k_eq_v true, final_logit_softcapping 30.0, hidden_activation `gelu_pytorch_tanh`, no PLE (hidden_size_per_layer_input=0), enable_moe_block false, attention_bias false. **layer_types pattern (NO per_layer_config in raw JSON — synthesize):** `"sliding_attention"` when `(i+1)%6 != 0` else `"full_attention"` → full at 5,11,17,23,29,35,41,47,53,59. Full layers: head_dim=512 (nv), n_kv=4 (gated by k_eq_v).
- **Per-forward semantic** (HF modeling_gemma4.py verified):
  - Embed: `row[token] * sqrt(5376)`; embed is BF16 `[262144,5376]`.
  - `embed[2]` (bos) bits first16: `3ccc,3c4a,3cda,3cb6,3cd2,3c3e,3c6f,bbfb,bc58,bb4f,3bc5,bc1c,bbbf,bc42,3b49,bc51` (BF16 → f16 via `bf16_bits<<16` as f32 then f16).
  - RMSNorm (fp32): `ms = mean(x²)+eps`; `inv = ms.powf(-0.5)`; `out = x*inv*w` (type→f16). v_norm = weightless (same but no `*w`).
  - RoPE: sliding head_dim 256 → `inv_freq[k]=1e4^(-2k/256)`, k=0..127; emb=cat(freqs,freqs); rotate pairs (j, j+128), neox: `x1=x[..half],x2=x[half..], out = x*cos + [-x2,x1]*sin`. Full head_dim 512 proportional → `inv_freq[k]=1e6^(-2k/512)`, k=0..63 then 192 zeros (len 256); rotate pairs (j, j+256); dims ≥ 64 in first half are identity (cos=1,sin=0). cos/sin in fp32, per-position.
  - Attention scale = 1.0 (no 1/sqrt(d), no text softcap). GQA groups = 32/n_kv (2 sliding, 8 full). Mask: full `kv≤q`; sliding `kv∈[q-1023, q]`. Full layers (k_eq_v): K = rope(k_norm(k_proj(x))), V = v_norm(k_proj(x)) (weightless RMSNorm on RAW k, no rope). Sliding: K = rope(k_norm(k_proj(x))), V = v_norm(v_proj(x)). q = rope(q_norm(q_proj(x))).
  - MLP: `down(gelu_tanh(gate)·up)`, gate/up from 5376→21504; gelu_tanh = `0.5x(1+tanh(√(2/π)(x+0.044715x³)))`.
  - Decoder order: `res=x; x=rms(input_ln); x=attn(x); x=rms(post_attn_ln); x=res+x; res=x; x=rms(pre_ff_ln); x=mlp(x); x=rms(post_ff_ln); x=res+x; x*=layer_scalar`. Final `x=rms(model.language_model.norm)`, then lm_head.
  - lm_head = `[5376→262144]`, then softcap `tanh(x/30)*30`. Logits kept as f32 (stream slabs; don't materialize 2.8 GB f16).
- **Real weight facts** (goldens for tests): layer_scalar[0] BF16 f32 = **0.089355469** (`0x3db7<<16`), layer_scalar[5] = **0.87890625** (`0x3f61<<16`). L0 q_proj.mcg I32 = little-endian = 0xCBAC1FED (MCG_MULT). q_norm[0].weight first8 f32: about `1.0,1.0,...` (verify in test with tolerance). Final norm.weight BF16 `[5376]`.
- **Golden consts computed this session** (pure-Python, f32-rounded; bake into tests):
  - sliding inv_freq first12: `1, 0.93057203, 0.86596435, 0.80584216, 0.7498942, 0.69783056, 0.64938164, 0.60429639, 0.56234133, 0.5232991, 0.48696753, 0.45315838` (matches recorded ref arrays).
  - full inv_freq first16: `1, 0.94746351, 0.89768714, 0.8505258, 0.80584216, 0.76350605, 0.72339416, 0.68538958, 0.64938164, 0.61526543, 0.58294153, 0.55231583, 0.5232991, 0.49580681, 0.46975887, 0.44507942`; nonzero count 64, len 256.
  - rope (dim 8, theta 1e4, pos 3): x bits `3c00,4000,4200,4400,4800,4a00,4c00,4e00` (1,2,3,4,8,12,16,20); out f16 bits `c03d,be8b,410a,43db` (f32 approx -2.11895, -1.63557, 2.51872, 3.92798). cos `-0.989992,0.955337,0.99955,0.999996`, sin `0.14112,0.29552,0.0299955,0.003`.
  - rms_norm (dim 8, eps 1e-6, same x, w=`1.5,-2,0.25,1,0.5,-1,2,-0.75`): inv=`0.086467542`; out f16 bits `3027,b589,2c27,3589,3589,bc27,4189,be3a` (f32 `0.12970132,-0.34587017,0.064850658,0.34587017,0.34587017,-1.0376105,2.7669613,-1.5564158`).
  - gelu_tanh f32: x=-3→`-0.0036373921`, -1→`-0.15880801`, -0.5→`-0.154286`, 0→`0`, 0.5→`0.345714`, 1→`0.84119201`, 3→`2.9963627`.
- **Implementation plan for next session** (T1.3c):
  1. Expand `TextConfig` in `load.rs` (~line 110): add `num_attention_heads, num_key_value_heads, num_global_key_value_heads: Option<usize>, num_hidden_layers, vocab_size, rms_norm_eps: f64, sliding_window: Option<usize>, hidden_activation, max_position_embeddings, rope_parameters: HashMap<String,RopeParams>`; new `RopeParams { rope_type: String, rope_theta: f64, partial_rotary_factor: Option<f64> }`. Synthesize per-layer (head_dim, n_kv) in `gemma4.rs` from layer_types + k_eq_v (do NOT expect per_layer_config in JSON).
  2. New `crates/engine/src/gemma4.rs`, register in `lib.rs`: `GemmaCfg::from_model`, helpers `bf16_bytes_to_f32`, `rms_norm_rows(x, dim, weight, eps)`, `inv_freq_*`, `rope_cos_sin`, `rope_apply`, `gelu_tanh`, `allowed(lt,s,t,window)`; `Gemma4Model<'a>` with `embed(&[u32])`, `layer(i, h, cos, sin)`, `forward_seq(&[u32]) -> Vec<f16>`, `logits(&[f16]) -> Vec<f32>` (streams lm_head in 128-col slabs via `reconstruct_inner_slab` + local had_r, softcap in f32). Build `Exl3Linear` from `tensor_bytes` (BF16 via u16<<16; `.mcg` I32 scalar→CodebookMode; trellis u16 LE slice — pointer 2-aligned guaranteed by safetensors 64-byte offsets, assert align). Residual adds + layer_scalar via DE28 f64-exact `f16::from_f64(...)`.
  3. Tests: inv_freq arrays vs golden, rope_out bits, rms_norm bits, gelu_tanh, mask allowed, embed[2] bits, layer_scalar values, single-layer-0 forward smoke (KAPLUMBA_RUN_SLOW=1 optional; skip if no artifact).
  4. T1.3d next: greedy generate (argmax on softcapped f32 logits, loop recompute prefix) + fixture, then `tasks.md` tick + commit `spec/rust-metal-engine: T1.3 naive forward`.
- Runtime caveat: full 60-layer CPU-forward on raw reconstruct is slow (minutes per small seq — lm_head alone = 2048 slabs × 336 in-tiles). Keep unit tests cheap (pure consts + layer-0 smoke). Accept slow golden per plan until T2.x Metal GEMV.

## Handover checklist

1. AGENTS.md, this spec folder, `spec/paged-kv-cache`, `spec/omlx-exl3-engine` (stub).
2. Open OQs → user (none expected; see spec.md).
3. First unchecked task in tasks.md; update tasks.md + this file every commit.
