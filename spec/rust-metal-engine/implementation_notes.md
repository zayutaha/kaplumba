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
