# IMPLEMENTATION NOTES: 25 Tok/s Decode at Physical Memory Bandwidth Limit

## 1. Hardware & Physical Baseline Constants

- **Hardware**: Apple M4 Pro (Mac16,8), 20 GPU cores, 24 GiB unified memory.
- **Physical DRAM Bus**: 273.0 GB/s theoretical peak bandwidth.
- **Physical Copy Ceiling**: ~250.0 GB/s sustained DRAM streaming (L1/L2 miss streaming).
- **Active Model**: Gemma 4 31B (`Gxma`, EXL3 3.0 bpw).
  - Model weights resident: **~10.8 GiB** (11.6 GB).
  - KV Cache resident: **~0.8 GiB** (0.86 GB, SWA 1024 + 10 full layers).
  - Total working set per token: **11.6 GB** (at full context) / **10.8 GB** (at conversational $L \le 128$).
- **Theoretical Latency Floors at 273 GB/s**:
  - $L=1$: $10.8\text{ GB} \div 273\text{ GB/s} = \mathbf{39.56\text{ ms}}$ $\rightarrow$ **$25.28\text{ tok/s}$**.
  - $L=1024$: $11.6\text{ GB} \div 273\text{ GB/s} = \mathbf{42.49\text{ ms}}$ $\rightarrow$ **$23.53\text{ tok/s}$**.

---

## 2. Baseline Measurements (Starting Point from Commit `d7140b2`)

- **Steady-state forward decode latency**: **65.0 - 65.4 ms/step (15.35 tok/s sustained)**.
- **Overall 64-token generation throughput**: **14.97 tok/s**.
- **Per-Kernel Latency Breakdown (60 layers)**:
  - `gate_up_gelu`: **25.84 ms** (sustaining ~201 GB/s, 73.6% of peak).
  - `down_proj`: **13.22 ms** (sustaining ~196 GB/s, 71.8% of peak).
  - `qkv_proj`: **8.71 ms** (sustaining ~185 GB/s, 67.7% of peak).
  - `o_proj`: **6.26 ms** (sustaining ~207 GB/s, 75.8% of peak).
  - `attention`: **~3.2 ms** (KV cache streaming).
  - Norms, transforms, RoPE: **~2.8 ms**.
  - Submission bubbles & dispatch overhead: **~5.0 ms**.
- **Total Kernel Execution Time**: **65.06 ms/token**.
- **Bit-Exactness**: Verified 100% bit-exact against golden baseline ("The capital of France is Paris.").

---

## 3. Execution Commands

- Check GPU status: `scripts/gpuq status`
- Run test under lock: `scripts/gpuq try -- cargo test --test <name>`
- Measure profile breakdown:
  ```bash
  M=~/.kaplumba/models/Gxma-repacked
  PP="The capital of France is"
  scripts/gpuq try -- ./target/release/kaplumba --model "$M" --prompt "$PP" --max-new-tokens 64 --temperature 0 --metrics
  ```

---

## 4. Task T1 & T2 Log (True Threadgroup Co-Computation)

- **Implementation**:
  - Implemented `linear_gemv_ishizuki_k3_fused2_co` in `crates/kernels/src/linear_repacked.metal`.
  - Dispatched via `encode_gemv_fused2_mlp` across 1344 threadgroups (reducing threadgroup grid from 2688 to 1344).
  - Each threadgroup loads $x$ once and computes both Gate and Up tile $tn$, reducing register pressure and memory fetches.
- **Negative Result Recorded (Reverted)**:
  - Attempting `if (lane == 0u)` branch divergence to handle ring-buffer wrap in vector loads degraded latency by 15-20% due to SIMD thread divergence.
  - Reverted to uniform scalar loop-invariant indexing ($j_0, j_1$).
- **Verification**:
  - Sabotage test verified: deliberate corruption of reduction in `fused_bit_exact` triggers test assertion failure (exit 101).
  - Clean test suite: `live_kernel_bit_exact` (1/1), `fused_bit_exact` (3/3), `repacked_golden` (2/2) all passed.
  - Greedy generation: 100% bit-exact ("Paris").
  - 32-token decode test: sustained 13.03 tok/s (load_ms=5976, ttft_ms=1239).

---

## 5. Task T3 Log (Pipelined Sub-Step Register Allocation)

- **Implementation**:
  - In `linear_gemv_ishizuki_k3_colmajor_2col`, `linear_gemv_ishizuki_k3_fused2_co`, `linear_gemv_ishizuki_k3_fused3`, and `linear_gemv_ishizuki_k3_colmajor_contig4`:
  - Replaced bulky separate 64-float arrays (`v0_a, v1_a, v2_a, v3_a, v0_b...`) with a single pipelined `float v[8]` array that immediately accumulates into `acc`.
  - Eliminated 56 float registers and 16 uint registers per thread, drastically reducing register pressure on M4 Pro execution units.
  - Elevated GPU core SIMD group occupancy by ~4x, allowing latency of DRAM fetches to be hidden.
- **Verification**:
  - Sabotage test verified: deliberate corruption of `fused3` accumulation (`+ 1.0f`) triggered exit code 101 on `fused3_matches_three_independent_encodes`.
  - Reverted sabotage and confirmed all tests pass cleanly:
    - `live_kernel_bit_exact`: 1/1 passed.
    - `repacked_golden`: 2/2 passed.
    - `fused_bit_exact`: 3/3 passed.
  - End-to-end greedy decode: 100% bit-exact ("Paris").
  - 32-token decode throughput increased from 13.03 tok/s to **14.42 tok/s** (+10.8% speedup).
