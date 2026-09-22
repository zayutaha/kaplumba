# Implementation notes — rust-metal-engine

Append as work proceeds (AGENTS.md §5). Dense, factual, takeover-oriented. Content migrated from superseded `spec/omlx-exl3-engine/implementation_notes.md` where still valid; Python/omlx-specific entries marked historical.

## Status

- Spec suite created 2026-09-22 after pivot to pure Rust + raw Metal (no Python, no MLX).
- Open questions **all closed** (interviews 2026-09-22) — see spec.md table (OQ1–OQ11).
- Companion spec: `spec/paged-kv-cache` (being rewritten for Rust) — coordinate at Phase 5 hooks only.
- Superseded: `spec/omlx-exl3-engine/` (stub).
- **T1.0 done** (2026-09-22): workspace + `metal` 0.33.0 + GPU fill smoke. Next: T0.2 baseline (reference-only) or T1.1 loader.

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

## Baseline numbers

- (fill T0.2: reference PonyExl3 numbers — **label reference-only**, not our engine's)

## Research digests (where to look)

- llama.cpp: `ggml-metal` mul_mv/mul_mm/fa; `common/speculative.cpp` (gemma4 shared KV); graph reuse `llama-context.cpp`.
- ik: `iqk_mul_mat.cpp` (amortize unpack, thresholds), `ggml-common.h` R4/R8, discussion 164 (pp tables), FA lesson discussion 25.
- PonyExl3: `mlx/gemv_metal.py` (simd GEMV, split-K), `exl3_fused.py`, `convert/*`, README benches (M1/M5/4090), fuse threshold notes (Gemma4 qkv vs gate+up).
- omlx (historical only): single-tenant overhead lessons; not a code dependency.

## Dead ends (do not repeat)

- EXL3 LUT decode (measured slower than procedural).
- Lowering global fuse min size (2× decode regressions on small groups).
- KV quant / Hadamard-KV (out of scope).
- Assuming M5 Max tunables transfer to M4 Pro unchanged.
- N-gram-only as substitute for real GEMV/MTP work (bounded wins).
- Timing TTFT via first detokenizer segment (late) — time first token emit instead.
- Per-row decode degradation in reference bench = memory accumulation without `mx.clear_cache` between rows (Python harness fix needed for fair reference capture; our Rust bench must clear/track peaks per row).

## Invariants

- I1: Single resident model; single-user; one agent session editing at a time.
- I2: Verify-gated speculation only.
- I3: Persistence on-disk format owned by `spec/paged-kv-cache`.
- I4: No hot-path cost without a measured justification.
- I5: No Python in engine process; no MLX; raw Metal only (DE15).

## Commands

- Rust build/test/smoke (T1.0+): `cargo build` · `cargo test -p kaplumba-engine` · `cargo run -p kaplumba-cli -- metal-smoke`
- Reference baseline (Python, temporary): `.venv-baseline/bin/python scripts/bench.py ~/.kaplumba/models/Gxma/Gxma --backend ponyexl3 --json /tmp/bench.json`
- Reference CLI: `.venv-baseline/bin/ponyexl3-generate-bench --prefill-sizes 1024,8192 --gen-tokens 64 --warmup --raw --json`
- Rust generate (from T1.4): `cargo run -p kaplumba-cli -- generate ...`

## Handover checklist

1. AGENTS.md, this spec folder, `spec/paged-kv-cache`, `spec/omlx-exl3-engine` (stub).
2. Open OQs → user (none expected; see spec.md).
3. First unchecked task in tasks.md; update tasks.md + this file every commit.
