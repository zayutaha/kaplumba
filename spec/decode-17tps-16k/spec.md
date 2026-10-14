# Spec: Decode 17+ tok/s (up to 4k) and 15+ tok/s (up to 16k) on Gxma-repacked

## 1. Goal

Deliver sustained autoregressive decode throughput of **$\ge 17.0\text{ tok/s}$** ($\le 58.8\text{ ms/token}$) at context lengths up to 4,096 tokens, and **$\ge 15.0\text{ tok/s}$** ($\le 66.7\text{ ms/token}$) across extended contexts up to 16,384 tokens on Apple M4 Pro (20 GPU cores, 273 GB/s, 24 GiB unified RAM), under chat sampling settings (`temperature = 1.0, top_k = 64, top_p = 0.95`) and greedy (`temperature = 0.0`) on `Gxma-repacked` without speculative decoding.

## 2. Non-goals & Restrictions

- **Single-model dense decode only:** strictly pure single-model autoregressive execution of `Gxma-repacked` (EXL3 ~3.0 bpw weights + 6 bpw head, ~12.6 GiB resident memory). No drafter models, no DFlash, no speculative decoding.
- **Zero quality loss:** Greedy decode (`temperature = 0.0`) must remain 100% bit-exact against reference.
- **No KV cache quantization:** FP16/BF16 key-value representations are preserved intact across all 60 layers.
- **No Python in engine or CLI code:** Engine and CLI remain 100% Rust and raw Metal compute shaders.
- **Strict memory budget & zero swap:** Sized within the 24 GiB unified memory limit. Testing at 16k context must strictly exhibit zero macOS swap activity (`sysctl vm.swapusage` shows 0 bytes swap growth). Scratch buffers remain fixed-size and independent of context length.
- **GPU exclusivity:** One model instance loaded at any time; all benchmark executions run via `scripts/gpuq try --`.

## 3. Requirements

1. **R1 (Tier-1 Throughput $\le$ 4k):** Sustained 32-token decode throughput $\ge 17.0\text{ tok/s}$ at prompt lengths up to 4,096 tokens under `temperature = 1.0, top_k = 64, top_p = 0.95` and `temperature = 0.0`.
2. **R2 (Tier-2 Throughput 4k–16k):** Sustained 32-token decode throughput $\ge 15.0\text{ tok/s}$ at prompt lengths of 8,192 and 16,384 tokens under `temperature = 1.0, top_k = 64, top_p = 0.95` and `temperature = 0.0`.
3. **R3 (GEMV Kernel Optimization):** Optimize `linear_gemv_ishizuki_k3_fused2_2col` and `linear_gemv_ishizuki_k3_colmajor_2col` to eliminate threadgroup idle loops and minimize memory latency on `gate_up` and `down_proj`.
4. **R4 (Full-Attention Scalability at 4k–16k):** Optimize decode attention across the 10 full-attention layers for 4k–16k keys (e.g. split-KV threadgroup parallelization with online softmax reduction) so full attention does not cause latency spikes.
5. **R5 (Bit-Exact Verification):** Greedy decode token streams must match the canonical reference bit-for-bit across all tested context lengths.
6. **R6 (Zero Swap Guarantee):** Resident memory footprint remains $\le 15.5\text{ GiB}$ at 16k context with zero paging.

## 4. Acceptance Criteria

1. **AC1 (4k Context Performance):** Canonical benchmark on 32 decode tokens at 2,048 and 4,096 context tokens demonstrates $\ge 17.0\text{ tok/s}$ median across repeated runs.
2. **AC2 (16k Context Performance):** Canonical benchmark on 32 decode tokens at 8,192 and 16,384 context tokens demonstrates $\ge 15.0\text{ tok/s}$ median across repeated runs.
3. **AC3 (Sampling & Coherence):** Outputs under `temperature = 1.0, top_k = 64, top_p = 0.95` are syntactically and semantically coherent.
4. **AC4 (Zero Swap Verification):** `scripts/gpuq status` and `sysctl vm.swapusage` report zero swap growth before and after 16k context execution.
5. **AC5 (Bit-Exact Regression Suite):** `fused_bit_exact`, `attention_single_pass_bit_exact`, and `live_kernel_bit_exact` test suites pass 100%.

## 5. Open Questions

*(None — all questions clarified and confirmed in the interview phase).*
