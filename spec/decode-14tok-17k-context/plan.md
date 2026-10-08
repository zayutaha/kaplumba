# Technical Plan: 14+ tok/s Decode Throughput at 17.5k Context

## 1. Architectural Strategy
To close the **6.0 ms deficit** (reducing steady-state decode latency from **77.4 ms** to $\le \mathbf{71.4\text{ ms}}$), we address the two largest levers identified in profiling:

```
Current Total Step (77.4 ms):
┌──────────────────────────────┬──────────────┬──────────────┬─────────┬────────┬───────┐
│ gate_up_gelu (26.2 ms)       │ down_proj    │ qkv/qk_proj  │ o_proj  │ attn   │ Driver│
│                              │ (14.0 ms)    │ (12.7 ms)    │ (6.8 ms)│(3.2 ms)│(7.6ms)│
└──────────────────────────────┴──────────────┴──────────────┴─────────┴────────┴───────┘

Target Step (≤ 71.4 ms):
┌──────────────────────────────┬──────────────┬──────────────┬─────────┬────────┬───┐
│ gate_up_gelu (24.0 ms)       │ down_proj    │ qkv/qk_proj  │ o_proj  │ attn   │Dr.│
│ [-2.2 ms via unroll tuning]  │ (13.5 ms)    │ (12.7 ms)    │ (6.8 ms)│(2.8 ms)│2ms│
└──────────────────────────────┴──────────────┴──────────────┴─────────┴────────┴───┘
                                                                        [Buffer batching: -5.0 ms]
```

### Lever A: Command Buffer Batching (Target: -4.5 to -5.5 ms)
- **File:** [`crates/kernels/src/encoder.rs`](file:///Users/zayaantaha/coding/kaplumba/crates/kernels/src/encoder.rs)
- **Current Behavior:** `decode_max_dispatches_per_buffer()` defaults to `50`. Across 60 layers (~540 dispatches per decode step), 11 separate command buffers are rolled and committed per token.
- **Proposed Change:** Test larger decode dispatch batch sizes (`150`, `270`, `600` / single buffer).
- **Hypothesis:** Committing 1–2 command buffers per step instead of 11 reduces driver submission queues, GPU context switches, and eliminates the ~7.6 ms inter-boundary gap.

### Lever B: GEMV K-Loop Unroll Tuning in `gate_up_gelu` (Target: -1.5 to -2.5 ms)
- **File:** [`crates/kernels/src/linear_repacked.metal`](file:///Users/zayaantaha/coding/kaplumba/crates/kernels/src/linear_repacked.metal)
- **Current Behavior:** `linear_gemv_ishizuki_k3_fused2_co` unrolls by `step = SG * 2 = 16` tiles along K ($16 \times 16 = 256$ elements per loop iteration).
- **Proposed Optimization:** Evaluate unrolling by 4 sub-steps (`step = SG * 4`) or software-pipelining the weight loads for trip $N+1$ during arithmetic for trip $N$, identical to the structure in `linear_gemv_ishizuki_k3_colmajor_2col_sk2`.
- **Expected Impact:** In `colmajor_2col_sk2`, software pipelining yields 234 $\mu$s per layer. In `fused2_co`, doing 2 columns simultaneously currently takes 437 $\mu$s (equivalent to ~218 $\mu$s per column). Squeezing an extra 30–40 $\mu$s per layer saves $60 \times 35\,\mu\text{s} \approx 2.1\text{ ms}$.

### Lever C: Attention Tile & Accumulation Tuning (Target: -0.5 to -1.0 ms)
- **File:** [`crates/kernels/src/attention.rs`](file:///Users/zayaantaha/coding/kaplumba/crates/kernels/src/attention.rs)
- **Current Behavior:** 10 full attention layers take 2.22 ms (at 32 splits, 128 workgroups).
- **Optimization:** Tune split count dynamically based on the exact core count (20 cores) so workgroups evenly divide across cores without trailing wave underutilization.

---

## 2. Review Questions for Claude
1. **Command Buffer Batching vs Overlap:** In decode, since the CPU only waits for the argmax token at the end of the step, what is the optimal command buffer batch size for Apple Silicon Metal to minimize driver overhead without stalling the GPU pipeline?
2. **K-Loop Pipeline in Metal Shading Language:** For `linear_gemv_ishizuki_k3_fused2_co` (computing gate and up simultaneously with 8 SIMDgroups of 32 threads), is double-buffering weight loads across iterations superior to unrolling 4 sub-steps in register pressure on M4 Pro?
3. **Token 2 Warmup Elimination:** Why does the first decode step after prefill take 1759 ms while subsequent steps take 77 ms, and how can the Metal command queue or pipeline state cache be warmed up during prefill completion to avoid dragging down multi-turn chat averages?
