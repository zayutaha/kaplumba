# Implementation Notes: Pure Decode 273 GB/s Bandwidth Saturation

## Architecture Log & Live Measurements

- **Hardware**: Apple M4 Pro (20 GPU cores, 273 GB/s memory bandwidth).
- **Target Model**: Gemma 4 31B EXL3 (~3.0 bpw). Total weights per token: 11.833 GB.
- **Baseline Measured State**: 144.5 ms/token = 6.92 t/s (81.89 GB/s = 29.99% bus utilization).
- **Primary Bottlenecks Documented**:
  1. Grid under-occupancy on narrow projections (`k`, `v` at 256 tiles yielding only 18.7 GB/s).
  2. 2,165 dispatches per token incurring ~26 ms of hardware bubble latency.
  3. Intermediate fp32 `y` DRAM round-trips for standalone `had_out` (420 dispatches).
- **Key Invariants & Proven Dead-Ends from History**:
  - Generic arbitrary-stride `simd_shuffle` tile weight ladders (T5.3r) are a net loss on M4 Pro due to 160-ALU barrel shuffle cost on 24-word tiles. Direct L1-resident loads are faster.
  - Weights must stay in their original memory-mapped positions without copying (resident memory limit ≤ 14.6 GB). Fused GEMVs use pointer dispatch.
