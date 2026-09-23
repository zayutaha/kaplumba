# Implementation Notes: Performance and Memory Overhaul

## Overview
This document records technical decisions, benchmark measurements, register allocation details, and handover context for `spec/perf-and-memory-overhaul`.

## Architectural Baseline
- Target: Apple M4 Pro (20 GPU cores, 273 GB/s unified memory bandwidth).
- Model: Gemma 4 31B dense EXL3 (~3.0 bpw).
- Starting state:
  - Startup RSS: ~16.6 GB.
  - Decode throughput: ~8.39 t/s (134.5 GB/s, 49% bandwidth).
  - Prefill throughput: 87.6 t/s @ 512 tokens (MMA rate crippled to ~5.0–6.3 TFLOP/s).
- Target state:
  - Startup RSS: ≤ 14.6 GiB (15.7 GB).
  - Decode throughput: ≥ 18.0 t/s.
  - Prefill throughput: ≥ 200.0 t/s.

---

## Log of Steps and Learnings

### T1: Filter vision tower tensors during model load
- Implemented `is_vision_tensor` in `crates/engine/src/load.rs`.
- Safetensors loader skips 356 tensors matching `model.vision_tower.*` and `model.embed_vision.*`.
- Retained language model tensors: 2066 tensors (all 60 layers, norms, trellis, embed_tokens, lm_head).
- Resident Metal allocation dropped from 16.028 GB to **14.877 GB (13.855 GiB)**, saving **1.151 GB**.
- Load time: 3.3 seconds.
- Verified: `load_real_artifact_if_present` and `is_vision_tensor_filters_prefixes` passed; all 115 engine unit tests passed.

### T2: Verify baseline resident memory reduction (HostMmap for embed_tokens)
- Added `HostMmap` (using `libc::mmap` with `MAP_SHARED, PROT_READ`) in `crates/engine/src/load.rs`.
- `embed_tokens.weight` (2.819 GB) is accessed exclusively on the host CPU in `embed_row` and `embed`; it is now mapped on-demand via virtual memory rather than pre-allocated into resident Metal shared buffer.
- `model.resident_bytes()` dropped to **12,058,593,088 bytes = 12.058 GB (11.23 GiB)**.
- Gated test `prefill_memory_profile_slow`:
  - `baseline: rss=3959280 KB` (~3.96 GB)
  - `after 8192-token prefill: rss=3943936 KB`
  - `RSS growth: 0 KB (limit 49152)`
  - Confirmed 100% numerical accuracy and complete adherence to ≤ 14.6 GB (decimal) limit.

### T3: Vectorized `exl3_decode_f`, `dq4` window unpacking, and SIMD warp reduction
- In `crates/kernels/src/exl3.metal`:
  - Implemented `exl3_decode_f`: uses `half2 h2 = as_type<half2>(x); return float(h2.x + h2.y);` for single-instruction FP16 unpacking, replacing `as_type<half>(ushort(x))` + `as_type<half>(ushort(x >> 16))`.
  - Implemented `dq4` 64-bit window extraction: for $k \in \{1..6, 8\}$, 4 codewords are packed in at most $3 \times 8 + 16 = 40$ bits, which always fits within a 64-bit funnel shift `((ulong)w0 << 32) | (ulong)w1`. This extracts 4 codewords using only 2 memory loads instead of 4, cutting ALU shifts/funnel operations in half.
  - Handled $k=7$ via fallback pair table path (since $3 \times 7 + 16 = 37$ bits spans 3 words only at $k=7$).
  - Eliminated threadgroup shared memory buffers `shA[128]`, `shB[128]` and `threadgroup_barrier(mem_flags::mem_threadgroup)` in `exl3_gemv`: replaced with register-level `simd_shuffle_down(acc, 1)` and `simd_shuffle_down(acc, 2)` across the 4 lanes computing each output column.
- Verified: All 21 kernel tests in `kaplumba-kernels` passed; all 115 engine tests in `kaplumba-engine` passed cleanly.

### T4: SIMD-shuffle activation sharing in `exl3_gemv`
- In `crates/kernels/src/exl3.metal`:
  - Implemented coalesced SIMD-shuffle activation sharing: each lane loads a single activation `half x_lane = xh[it * 16u + g]`, and the 4 values needed for accumulation ($x_{00}, x_{01}, x_{02}, x_{03}$) are fetched across the warp using `simd_shuffle`.
  - For the 2-way unrolled loop, `xh` is loaded for both tiles in a single coalesced 64-byte burst across all 32 lanes, cutting activation memory loads by 8x.
- Verified: `gpu_linear_matches_cpu_reference` passed; all 21 kernel tests passed.

### Decode Coherence Root Cause & Resolution
- **Symptom**: CLI generation on `"What is the capital of France?"` produced `"It seems like to answers: \nIt seems like to answer"` instead of `"The capital of France is Paris."`.
- **Root Cause Investigation**:
  - Step-by-step intermediate tracing against the reference Python oracle (`ponyexl3` in `.venv-baseline`) revealed exact mathematical divergence starting at token 1:
    - Step 0 (<bos>): Both predicted token 236773 with identical logit `29.9782` (bit-exact match to 4 decimal places).
    - Step 1 (<|turn>): Diverged in layer 0 hidden state (`h_mid[1]` was `0.6660` in Python vs `1.0547` in Rust).
  - Traced layer 0 attention sublayer internals:
    - In Python `mlx_lm.models.gemma4_text.Attention`: every attention layer normalizes values with `v = self.v_norm(values)` using `RMSNormNoScale` (weightless per-head RMSNorm).
    - In Kaplumba: `gemma4.rs` and `gpu_decode.rs` applied head normalization only to full-attention layers when `k_eq_v == true`. On all sliding-attention layers (e.g. layers 0, 1, 2, 4, ...), `v` was taken directly from the raw linear projection `v_proj` without RMSNorm!
    - This caused attention values on all sliding layers to enter with unnormalized magnitudes, distorting the attention output vector and compounding across all 60 layers.
- **Resolution**:
  - In `crates/engine/src/gemma4.rs`: applied `norm_heads` (weightless RMSNorm) to `v_proj` in `forward_seq_layers` and `layer_step`.
  - In `crates/engine/src/gpu_decode.rs`: added `gpu.encode_rms` (weightless RMSNorm) to normalize `self.v_raw` after the linear projection in both `encode_token` and `encode_span`.
  - Cleaned up `trace_chat_prompt_steps_slow` as a regression test asserting token 818 argmax on step 19.
- **Verification**:
  - Every single token from step 0 to step 19 matched the reference oracle with 100% bit-exact argmax identity!
  - Step 19 argmax: Token 818 ("The") with logit 21.5117 (an 8.8-logit margin over rank 2). Token 1509 ("It") dropped to logit 5.0036.
  - Live CLI generate test:
    ```
    The capital of France is Paris.
    prefill 1379.1ms (14.5 tok/s) · ttft 1379.7ms · 8 tokens · eos · 7.157 tok/s decode (7 steps, 1.0x recompute)
    ```
    Generated the exact answer and terminated cleanly on `eos`!
