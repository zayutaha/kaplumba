# Implementation Notes: Decode 17+ tok/s (up to 4k) and 15+ tok/s (up to 16k)

Running log per `AGENTS.md §5`. Factual and dense.

## Baseline Measurements (2026-10-14)

- **Machine:** Apple M4 Pro (20 GPU cores, 273 GB/s peak, 24 GiB unified RAM).
- **Model:** `~/.kaplumba/models/Gxma-repacked` (EXL3 ~3.0 bpw weights, 6 bpw head, ~12.6 GiB resident memory).
- **Settings:** `temperature = 1.0, top_k = 64, top_p = 0.95`.
- **Baseline decode throughput (short context):** 14.85 tok/s (67.4 ms/step).
- **GPU Kernel Profile (Decode Step, ~67.9 ms total):**
  - `gate_up_gelu`: 26.12 ms (60 layers, 0.435 ms/layer).
  - `down_proj`: 13.47 ms (60 layers, 0.225 ms/layer).
  - `qkv_proj` (50 layers) + `qk_proj` (10 layers): 9.66 ms + 2.18 ms = 11.84 ms.
  - `o_proj`: 7.10 ms (60 layers, 0.118 ms/layer).
  - `head/gemv`: 4.00 ms (1 layer).
  - `norm_transform` + `pre_ffn_norm_transform`: 1.22 ms + 1.14 ms = 2.36 ms.
  - `attention`: 0.96 ms.
  - `post_attn_norm_res` + `post_ffn_out`: 1.43 ms.
  - `head/sample`: 0.30 ms.

## Invariants & Gotchas Discovered

1. **`fused2_2col` threadgroup sizing:**
   In `linear_repacked.metal`, `linear_gemv_ishizuki_k3_fused2_2col` iterates `tn` from `tgid` to `pairs_a = tiles_a / 2` with step `ntg`. In `encode_gemv_fused2_mlp`, `(tiles_a + tiles_b) / 2 = tiles_a = 1344` threadgroups were launched. Exactly half of the launched threadgroups (`tgid >= 672`) did zero loop iterations and exited, adding scheduler dispatch overhead. The correct launch count is `tiles_a / 2 = 672`.
2. **`post_ffn` norm guard:**
   Line 2645 in `crates/engine/src/model.rs` was previously guarded by `if !(decode && gate_up_share_input_signs && self.weights.gate_proj.permutation_mode == 1)`. When `gate_up_share_input_signs` is false, this condition evaluated to true and overwrote the fused MLP output with `scratch.proj` (which held stale prefill activations). Fixed in commit `b909b6d` to `if !(decode && self.weights.gate_proj.permutation_mode == 1)`.
3. **Multi-input vector `stride_x` in `fused2_2col`:**
   In `linear_gemv_ishizuki_k3_fused2_2col`, Col B must reload `p00..p33` from `x + stride_x` when `stride_x != 0`. Implemented in commit `b909b6d` reusing existing float registers without increasing register pressure.
4. **Scalar 32-bit MCG extraction in `fused2_sk2` and `fused3_sk2` (T2):**
   Replaced 64-bit `ulong` shifts (`((w0a << 32) | w0b) << sh0`) with 32-bit scalar MCG extraction (`extract_v_mcg_32`) and single `float v[8]` register reuse across all 4 loop steps. Slashed active register pressure and 64-bit integer ALU instructions in QKV split-K kernels. Sabotage test confirmed fail-first behavior in `fused_bit_exact`.
5. **Down projection Split-K Profiling & Verification (T3):**
   Benchmarked `colmajor_1col_sk2` against `colmajor_2col_sk2` and `colmajor_2col_sk4` across all 60 layers on `Gxma-repacked`. `colmajor_1col_sk2` achieves 13.45 ms (193.4 GB/s effective bandwidth), outperforming `2col_sk2` (13.48 ms) and SK4 (13.50 ms) while halving per-threadgroup reduction. Bit-exactness verified against CPU reference in `live_kernel_bit_exact.rs` with fail-first sabotage test.
6. **Full-attention decode scalability (T4):**
   Lowered `decode_flash_min_keys` default from 4096 to 512. In Gemma 4 architecture, 50 out of 60 layers are sliding-window attention (fixed 1,024 key window), and only 10 layers are full-attention. Previously, for context lengths between 512 and 4096 tokens, full-attention layers used the scalar split-attention kernel which took ~7.45 ms across the 10 layers. With MMA Flash decode active for keys $\ge 512$, full attention scales efficiently with online-softmax key reduction, dropping attention runtime to 6.55 ms at 2k context (saving ~0.9 ms/token). Sabotage test (`scale * 2.0`) failed as expected and 8/8 tests passed on clean code.


