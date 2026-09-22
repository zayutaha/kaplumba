# Implementation notes — omlx-exl3-engine

Append as work proceeds (AGENTS.md §5). Dense, factual, takeover-oriented.

## Status

- Spec suite drafted; **no implementation code yet**. Phase 0 (instrument) starting.
- Open questions **all closed** (interview 2026-09-22) — see spec.md table.
- Companion spec: `spec/paged-kv-cache` (persistence, delta prefill) — coordinate at Phase 2 hooks only.

## Decisions (pre-implementation)

- DE1: Base = **omlx**; PonyExl3 = EXL3 converter CLI + reference material only — **not** embedded/vendored as engine code.
- DE2: Memory model of record: **EXL3 3.0 bpw ≈ 15.5 GB weights + ~1 GB SWA/KV ≈ 16.5 GB resident**; roofline **~17.6 tok/s**. User-corrected; do not recalculate as 31e9×3/8.
- DE3: Priority order fixed: TTFT → tg/s → pp/s → mem awareness; every task tags its metric.
- DE4: No KV quant in defaults (standing constraint).
- DE5: Research-derived strategy locked at spec level: single-tenant strip; dual-path prefill (ik); sibling fusion + MTP shared-KV (PonyExl3/llama.cpp); SWA bound; compiled decode; strip per-chunk clears.
- DE6 (OQ1): Quality gate = **sampled golden review** at **temp=1.0, topk=64, top_p=0.95** vs reference quant.
- DE7 (OQ2): Speed ACs **report-only** for v1 — before/after numbers, no hard thresholds.
- DE8 (OQ3): Machine is **24 GB** unified — fold16 on main weights **banned**; tight headroom around 16.5 GB target.
- DE9 (OQ4): Primary sampling mode = **temp 1.0 / topk 64 / top_p 0.95**; temp-0 still used only for spec-decode identity tests (R8.1).
- DE10 (OQ5): **8k typical / 32k max** — benches 1k/2k/8k + 32k spot check.
- DE11 (OQ6): Reference SHAs **pinned and frozen** (below); no mid-spec pulls.
- DE12 (OQ7): **Text-only** v1; no vision/mmproj.
- DE13 (OQ8): **Fresh EXL3 kernels in omlx** (trellis GEMV + load/dequant reimplemented from reference); convert via external PonyExl3 CLI OK.
- DE14: Patch strategy for omlx = decide at T0 (wrapper package vs vendored patch series) — not yet chosen.

## Environment (T0.3 — verified)

- Host: macOS 27.0, **24 GB** RAM (`sysctl hw.memsize` = 25769803776). All env setup via **uv 0.8.17** (user directive).
- **Two venvs** (DV3), repo root:
  - `.venv-baseline` — CPython **3.14.3** (PonyExl3 requires ≥3.14): `mlx==0.32.2`, `mlx-lm==0.31.3`, `ponyexl3==0.3.0` (editable from `reference/PonyExl3`), numpy 2.5.3, transformers 5.17.0.
  - `.venv-omlx` — CPython **3.13.7** (omlx requires <3.14): `mlx==0.32.2`, `mlx-lm==0.32.0`, `omlx==0.7.0.dev4` (editable from `reference/omlx`), nanobind 2.15.0.
- Models root: `~/.kaplumba/models/Gxma/Gxma/` — **EXL3 3.0 bpw artifact present** (DV2): `model-0000{1,2}-of-00002.safetensors` (8.50 + 7.53 GB ≈ 16.03 GB raw), `quantization_config.json` v0.0.37 bits 3.0 head_bits 6 out_scales always codebook mcg; tensors = `trellis` I16 + `suh`/`svh` F16 + `mcg` I32 + BF16 norms/embeds (218 BF16 / 424 F16 / 212 I32 / 212 I16 per shard 1). Gemma4ForConditionalGeneration: 60 layers (50 SWA window 1024 + 10 full), hidden 5376, head_dim 256 / global 512, 32 heads / 16 kv / 4 global-kv, vocab 262144, `attention_k_eq_v: true`, logit softcap 30, max_pos 262144, vision tower present but text-only per OQ7. `generation_config.json` already: do_sample, temp 1.0, top_k 64, top_p 0.95 (= OQ4).
- No BF16 source checkpoint in models root — conversion provenance = PonyExl3 converter v0.0.37 (version string inside quant config).
- Repo: Cargo skeleton only (`src/main.rs` hello world); engine work is omlx-based, not Rust-in-src.
- Reference pins (frozen per DE11):

| Repo | SHA |
|------|-----|
| reference/omlx | `8a6bfb84edb52ea48a02f4f2490684b5cb65ab2d` |
| reference/llama.cpp | `58367713a6935c0810103378144008df32e3d5db` |
| reference/ik_llama.cpp | `baac291dc9d531927760b48451d8dfcb63b6adec` |
| reference/PonyExl3 | `8e7fa6b1556f59fc669e25087903b279b9b0346f` |
| reference/vllm | `1fee69838eeb72622b441a593888687831c1e434` |

- Baseline bench entrypoints (reference): `ponyexl3-generate-bench` (`reference/PonyExl3/ponyexl3/cli/generate_bench.py`, `--json` supported, PREFILL_BENCH_SIZES 1k..32k, `--warmup`); stack loader `load_generate_stack(args)` in `cli/_generate_common.py` (model/config/tokenizer/mtp/eagle3/dflash/extra_eos); generation `ponyexl3.mlx.generate.generate_text(...)` → `(text, stats)` with `prefill_s`, `decode_s`, `prompt_tokens`, `gen_tokens`, `finish_reason`, `spec_*`. Engine flag default `exl3`; `--mtp auto` loads MTP if weights present, else plain decode. EOS union helper `collect_eos_token_ids` merges top-level + `text_config` (Gemma4 `<turn|>`=106 only at top level).
- omlx reference bench: `reference/omlx/scripts/bench.py` (VLMBatchedEngine + `_run_single_test`/`_run_batch_test` from `omlx.admin.benchmark`; metrics `ttft_ms`, `gen_tps`, `processing_tps`, `peak_memory_bytes`, `avg_ttft_ms`); warmup via `_generate_prompt`; single-tenant step gaps profiled later in T0.2/T1.1.

## Decisions (continued — pre-implementation)

- DE1: Base = **omlx**; PonyExl3 = EXL3 converter + kernel reference, not the serving base (OQ8 still open on how deep to embed its GEMV).
- DE2: Memory model of record: **EXL3 3.0 bpw ≈ 15.5 GB weights + ~1 GB SWA/KV ≈ 16.5 GB resident**; roofline **~17.6 tok/s**. User-corrected; do not recalculate as 31e9×3/8.
- DE3: Priority order fixed: TTFT → tg/s → pp/s → mem awareness; every task tags its metric.
- DE4: No KV quant in defaults (standing constraint).
- DE5: Research-derived strategy locked at spec level: single-tenant strip; dual-path prefill (ik); sibling fusion + MTP shared-KV (PonyExl3/llama.cpp); SWA bound; compiled decode; strip per-chunk clears.

## Divergences from spec/plan

- DV1 (T0): **Stock omlx has no `exl3` loader** (grepped full tree @ 8a6bfb84: only fp8/compressed-tensors/paroquant/awq/gptq/mxfp*). Target artifact `~/.kaplumba/models/Gxma/Gxma` is already EXL3 3.0 bpw packed (trellis I16 + suh/svh F16 + mcg I32; `quantization_config.json` v0.0.37, bits 3.0, head_bits 6, out_scales always, codebook mcg). **Baseline engine = PonyExl3 MLX runtime** (pinned 8e7fa6b); spec/plan/tasks amended (R7.3, plan Phase 0, T0.2).
- DV2 (T0): **T3.1 conversion already done** — artifact on disk matches spec parameters exactly; task reduced to verify/load/resident report. Original BF16 source not present in models root.
- DV3 (T0): omlx `requires-python >=3.11,<3.14` + pins `mlx==0.32.2`, `nanobind==2.15.0`; PonyExl3 requires `>=3.14`, `mlx>=0.31.2`, `mlx-lm>=0.31.3` → **two venvs**, env setup via **uv** (user directive).

## Baseline numbers

- (fill T0.2: hardware RAM size, model path, mlx versions, stock omlx TTFT/pp/tg/mem)

## Environment

- Machine: M4 Pro, 20 GPU cores, 273 GB/s.
- Models: `~/.kaplumba/models`.
- References: `reference/omlx`, `reference/llama.cpp`, `reference/ik_llama.cpp`, `reference/vllm`, `reference/PonyExl3` (pin SHAs in T0.3).

## Research digests (where to look)

- omlx hot path: `omlx/scheduler.py` (step, prefill, admission gates), `engine_core.py` (burst, wakes), `utils/sampling.py`, `utils/metal_sync.py`, `cache/*`, `patches/mlx_lm_mtp/*`, `patches/gemma4_verify_kernel.py`.
- llama.cpp: `ggml-metal` mul_mv/mul_mm/fa; `common/speculative.cpp` (gemma4 shared KV); graph reuse `llama-context.cpp`.
- ik: `iqk_mul_mat.cpp` (amortize unpack, thresholds), `ggml-common.h` R4/R8, discussion 164 (pp tables), FA lesson discussion 25.
- PonyExl3: `mlx/gemv_metal.py` (simd GEMV, split-K), `exl3_fused.py`, `convert/*`, README benches (M1/M5/4090), fuse threshold notes (Gemma4 qkv vs gate+up).

## Dead ends (already known from research — do not repeat)

- EXL3 LUT decode (measured slower than procedural).
- Lowering global fuse min size (2× decode regressions on small groups).
- KV quant / Hadamard-KV (out of scope).
- Assuming M5 Max tunables transfer to M4 Pro unchanged.
- N-gram-only as substitute for real GEMV/MTP work (bounded wins).

## Invariants

- I1: Single resident model; single-user; one agent session editing at a time.
- I2: Verify-gated speculation only.
- I3: Persistence on-disk format owned by `spec/paged-kv-cache`.
- I4: No hot-path cost without a measured justification.

## Commands

- (build/run/bench commands from T0.1 onward)

## Handover checklist

1. AGENTS.md, this spec folder, `spec/paged-kv-cache`.
2. Open OQs → user.
3. First unchecked task in tasks.md; update tasks.md + this file every commit.
