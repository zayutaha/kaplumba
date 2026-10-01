# SPECIFICATION: 17-18 tok/s Decode — O_proj Anomaly Fix + Non-GEMV Recovery

## 1. Goal

Raise sustained decode throughput from **15.85 tok/s (63.1 ms/step)** to
**≥ 17.0 tok/s (≤ 58.8 ms/step)** for Gemma 4 31B (EXL3 3.0 bpw) on Apple M4 Pro,
by attacking the two remaining addressable bottlenecks:

1. **o_proj's bandwidth anomaly**: 100 GB/s vs 200 GB/s for every other projection
   (caused by K=8192 being too short for cold-start amortization — the memory
   pipeline is cold when the GEMV starts). Worth **2-3 ms**.
2. **Non-GEMV dispatch overhead**: 749 dispatches at 3.5 µs each, intermediate
   buffer round-trips, and idle memory bus during attention. Worth **1-2 ms**.

## 2. Non-Goals & Restrictions

1. **20 tok/s is not the target.** The 3 bpw GEMV ceiling is 17.2 tok/s with zero
   overhead (11.834 GB / 203 GB/s = 58.3 ms). This spec is explicit about that.
2. **No speculative decoding**, no draft models, no multi-token generation.
3. **No quality loss.** Greedy-identical output. No KV cache quantization.
4. **No Python in engine code.** No format changes.
5. **Original checkpoint untouched.** Resident ≤ 16.5 GiB, 0 swap.
6. **Do not retry closed dead ends** (listed in decode-3bpw-amplification §6).
7. **GPU access discipline** per AGENTS.md §10.

## 3. Requirements

### R1: o_proj Memory Pipeline Warm-Start

Eliminate the cold-start penalty in o_proj's GEMV (K=8192, currently 100 GB/s)
by overlapping its weight prefetching with the preceding attention phase. The
o_proj GEMV kernel must begin loading weight tiles before attention completes,
so that by the time the attention output is ready, the DRAM streaming prefetcher
is fully warm and the GEMV proceeds at ≥ 160 GB/s.

Two implementation strategies, tried in order:

**(a) Single-queue concurrent dispatch with atomic flag** (preferred):
- Dispatch o_proj threadgroups in the SAME command buffer as attention
- o_proj threadgroups begin K-loop weight loads immediately
- Before first accumulation, spin-wait on a `device atomic_uint attn_done` flag
- Attention's final dispatch sets the flag
- Metal's threadgroup scheduler runs both concurrently because they use
  different threadgroups and o_proj has no output writes during the wait

**(b) Dual-queue with MTLEvent** (fallback if (a) doesn't schedule concurrently):
- Attention on secondary `MTLCommandQueue`
- `MTLEvent` synchronisation: qkv_done → attention starts, attn_done → o_proj starts
- Weight prefetching on primary queue begins as soon as qkv is dispatched

### R2: Cross-Layer Norm/Residual Chain Fusion

Fuse layer L's `finalize_and_post_ffn_norm_scale` + `residual_add` with layer
L+1's `input_layernorm` + `input_transform` into a single dispatch. Eliminates
~118 dispatches (59 layer transitions × 2 saved dispatches) and intermediate
buffer writes.

Pattern: extend the existing `encode_finalize_o_proj_and_pre_ffn_transform_sk2`
across the layer boundary.

### R3: Bit-Exact Verification and Sabotage

All optimizations bit-exact against reference. Sabotage evidence per AGENTS.md §3.

## 4. Acceptance Criteria

- **AC1:** ≥ 17.0 tok/s median of ≥ 8 interleaved A/B runs on ≥ 64-token generation.
- **AC2:** Bit-exact greedy output ("The capital of France is Paris.").
- **AC3:** All test suites pass (0 regressions).
- **AC4:** Sabotage evidence for each changed kernel path.
- **AC5:** Resident ≤ 16.5 GiB, 0 swap.
- **AC6:** Negative results recorded with numbers.

## 5. Open Questions
 
 - **Q1:** Does Metal schedule o_proj threadgroups concurrently with attention
   within the same command buffer?
   **RESOLVED (T1):** Yes. Verified via `concurrent_tg_feasibility.rs`. Output
   0xCAFE returned with zero serialization or GPU hang.
 - **Q2:** What macOS/Metal version? Metal 4 (macOS 26) has `MTL4ComputeCommandEncoder`
   which may supersede both R1(a) and R1(b).

