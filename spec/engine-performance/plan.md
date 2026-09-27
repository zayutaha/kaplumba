# Technical Plan: Engine Optimization (Decode & Prefill)

## 0. Prerequisite: Read the References First

Before implementing anything in this plan, work through `spec.md` §1a. **oMLX**
(`reference/omlx`) and **ishizuki**
(<https://github.com/struffl/ishizuki>) are the two closest prior attempts at
exactly the numbers we are chasing, and both are available locally. Each item
below already has a working implementation in one of them; port from those
rather than reinventing.

| Plan item | Reference implementation to port from |
|---|---|
| §2.1 bit-exact attention | No direct reference — oMLX's merge is not bit-exact. This one is original work. |
| §2.2 tensor-core GEMM | oMLX `omlx_sdpa_*` (uses `simdgroup_multiply` for QK/PV) |
| §2.2 parallel attention | oMLX `omlx_sdpa_decode` (32 simdgroups, 4 KB threadgroup merge) |
| §3.1 codebook LUT | ishizuki matvec; exllamav3 `exl3_gemm_inner.cuh` |
| §3.2 cooperative ring loads | exllamav3 `cp.async` / `int4` staging |
| §3.1 packed projections | oMLX `qwen35_packed_linear.py` `project()` |

---

## 1. Architecture Overview

Kaplumba executes Gemma 4 31B (EXL3 ~3.04 bpw) entirely on the Apple M4 Pro GPU using raw Metal compute pipelines encoded via Metal command buffers in pure Rust.

The computational pipeline consists of two distinct operational modes:
1. **Decode Mode (Token Generation):** Batch size = 1 token. Memory-bandwidth bound. Streams ~11.51 GB of weights once per generated token. Target is maximum memory bandwidth efficiency (roofline 22.6 tok/s @ 273 GB/s).
2. **Prefill Mode (Prompt Ingestion):** Context length $L \in [24, 8192]$. Compute-bound. Reuses weights across $L$ tokens. Target is high arithmetic intensity and parallel throughput.

---

## 2. Prefill Technical Design: The Two Divergent Paths

The fundamental gating factor is floating-point bit-exactness. Hardware architectures evaluate matrix multiplications and parallel reductions differently than sequential loops.

```
                    ┌──────────────────────────────────────────────┐
                    │    Prefill Architecture Decision Point       │
                    └──────────────────────┬───────────────────────┘
                                           │
                 ┌─────────────────────────┴─────────────────────────┐
                 │                                                   │
        [Option A: Bit-Exact]                             [Option B: Relaxed-Float]
        Strict IEEE sequential order                      Standard numerical tolerance
        Scalar FP32 SIMT only                             Hardware Tensor Cores & Parallel Merge
                 │                                                   │
        ┌────────┴────────┐                               ┌──────────┴──────────┐
        │                 │                               │                     │
  GEMM Decode LUT    Prefix-Max Attention          simdgroup_multiply    oMLX Multi-Simdgroup
  (~10 ALU ops →     Parallel scan for max,        Tensor cores          32-simdgroup online
  table lookup)      sequential sums               (30–50+ TFLOPS)       softmax + merge
        │                 │                               │                     │
        └────────┬────────┘                               └──────────┬──────────┘
                 │                                                   │
        Achievable Ceiling:                               Achievable Ceiling:
        ~15 – 25 tok/s                                    100 – 150+ tok/s
```

### 2.1 Option A: Strict Bit-Exact Path (~15–25 tok/s ceiling)
If the user demands exact bit-for-bit identity with the sequential fp32 reference:
1. **Weight Decode Acceleration (GEMM):**
   - Current bottleneck: `exl3_code_at_tg` performs ~10 integer ALU ops (including integer divide and modulo) per 3-bit weight.
   - Design: Since each rate has only $2^{ka}$ possible code values (8 values at rate 3), precalculate a 256-element lookup table in `constant` address space or threadgroup memory, eliminating arithmetic unpacking.
   - Keep 16×16 tile with `KCHUNK=64` to maintain high core occupancy without threadgroup memory thrashing.
2. **Prefix-Max Attention Acceleration:**
   - Current bottleneck: Phase 1b runs on thread 0 only, executing $L$ serial `exp()` and total accumulation steps while 255 threads idle.
   - Design: The running maximum is monotonic ($m_{t} \ge m_{t-1}$). The sequence of *rescale events* can be found in parallel via an order-independent prefix-max scan across all 256 threads.
   - All threads evaluate `exp(score - max)` in parallel. Thread 0 then accumulates the resulting factors strictly in sequential order, maintaining exact bitwise addition order while parallelizing the transcendental `exp()` calls.

### 2.2 Option B: Hardware-Accelerated Path (100–150+ tok/s target)
If standard float equivalence (greedy generation coherence) is accepted:
1. **Metal Tensor Cores (`simdgroup_multiply_accumulate`):**
   - Dispatch `simdgroup_half8x8` or `simdgroup_float8x8` matrix multiply operations.
   - Fix the SIMD lane layout mapping in `crates/engine/examples/metal_probe.rs` to correctly unpack EXL3 weights into 8×8 matrix fragments.
   - Unlocks M4 Pro's dedicated tensor hardware (30–50+ TFLOPS capacity), blowing past the 8.76 TFLOPS SIMT fp32 ceiling.
2. **Multi-Simdgroup Online Softmax with Threadgroup Merge:**
   - Port the proven oMLX architecture (`omlx_sdpa_decode`):
   - Allocate 1024 threads (32 simdgroups) per `(query, head)`.
   - Each simdgroup processes an interleaved stride of the KV cache (`stride = 128`), maintaining its own local `(max, sum, acc)`.
   - At the end of the KV sequence, the 32 partial results are merged in a 4 KB threadgroup reduction using standard FlashAttention rescaling:
     $$m = \max(m_i), \quad l = \sum_i l_i e^{m_i - m}, \quad O = \sum_i O_i \frac{l_i e^{m_i - m}}{l}$$
   - Eliminates the 256× serialization stall entirely with zero global memory scratch buffers.

---

## 3. Decode Technical Design (Pushing 7.87 → 12–16 tok/s)

Decode is memory-bandwidth bound. At 12.06 GB streamed per token, 7.87 tok/s corresponds to ~95 GB/s effective bandwidth (35% of peak 273 GB/s).

### 3.1 Codebook Lookup Table (LUT)
- Replace MCG hash calculations in `crates/kernels/src/linear.metal` (`linear_gemv_exl3_k*`) with a direct 256-entry table.
- Eliminates 4 integer ops (`mul, and, xor, shift`) per weight.

### 3.2 Cooperative Ring Loads
- In the current GEMV kernel, the 32 lanes of a simdgroup read overlapping 16-bit windows of a `256 * ka`-bit ring via unaligned loads, causing memory lane conflicts (55% of kernel time is load latency).
- Design: Have lanes collectively execute aligned 128-bit loads into threadgroup memory or registers, sharing the ring cooperatively.

### 3.3 Grid-Stride Matvec Tuning
- Finish parameter sweep on the `grid_stride` kernel (`63f94e5`) to cap active threadgroups to match the M4 Pro core count (20 cores), reducing thread scheduling contention and maximizing L2 cache hits.

---

## 4. Test Suite & Harness Fixes

### 4.1 Metal Command Encoder Lifecycle Bug
- File: `crates/engine/tests/attention_metal.rs`
- Issue: Test runner asserts `Command encoder released without endEncoding`.
- Fix: Ensure all test command encoders invoke `encoder.endEncoding()` explicitly before dropping or creating subsequent encoders. Identical to the fix previously applied in `forward_buffer`.

---

## 5. File Touchpoints

| File Path | Current Status | Planned Modifications |
|---|---|---|
| `crates/kernels/src/linear.metal` | 16×16 tiled GEMM with register accum | Implement weight decode LUT / SIMD shuffle; tensor-core kernel |
| `crates/kernels/src/linear.rs` | Dispatches `linear_gemv_tiled` | Host dispatch parameters, buffer bindings, grid sizing |
| `crates/kernels/src/attention.metal` | `online_attention_tg2_256/512` | Implement prefix-max parallel scan (Option A) or multi-simdgroup merge (Option B) |
| `crates/kernels/src/attention.rs` | Selects specialized head-dim pipelines | Pipeline setup for parallel attention kernels |
| `crates/engine/tests/attention_metal.rs` | Crashes on encoder drop | Fix encoder lifecycle (`endEncoding`) |
| `crates/engine/examples/metal_probe.rs` | Simdgroup probe matches 8/64 | Fix 8×8 matrix fragment lane layout |
| `spec/engine-performance/*` | Consolidated audit documentation | Authoritative record of all work and decisions |

---

## 6. Risk Matrix & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Bit-exact requirement maintained | Prefill capped at ~15–25 tok/s | Clearly present arithmetic roofline to user; provide Option A vs Option B |
| Threadgroup memory thrashing | Core occupancy drops, kernel slows down | Keep threadgroup memory under 8 KB per group; verified by 64×64 post-mortem |
| Simdgroup lane packing complexity | Numerical corruption in tensor path | Validate against CPU reference with extensive unit tests in `metal_probe.rs` |
| Test harness encoder assert | Blocks `cargo test` suite | Fix `attention_metal.rs` encoder lifecycle first |
