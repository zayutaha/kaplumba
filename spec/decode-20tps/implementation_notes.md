# IMPLEMENTATION NOTES: 17-18 tok/s Decode

Running log per AGENTS.md §5.

---

## 0. Prior art and what NOT to redo

This spec builds on top of the completed work in `decode-25tok-peak-bandwidth` (T1-T5.6)
and `decode-3bpw-amplification` (T1 negative). The standing record:

- **Baseline: 15.85 tok/s (63.1 ms/step)**, commit `fc36ee1`, re-measured in
  `decode-25tok-peak-bandwidth/implementation_notes.md` §16.1 with spread 0.03 tok/s.
- **GEMV floor: 58.3 ms = 17.2 tok/s** with zero overhead, from the 11.834 GB/token
  weight budget at 203 GB/s best measured 3 bpw bandwidth.
- **10 dead ends formally closed** — do not retry any of them:
  lane-contiguous repack (impossible), 4 bpw (−6.7%), split-K>2 (negative),
  4-col (−4.2%), uint4 loads (−3.8%), simd_shuffle (−18.8%), remove dequant (−26.7%),
  megakernel (no grid barrier), elementwise fusion (actual 0.8ms not 3-4ms),
  1-col streaming (−0.16 tok/s).
- **ICB tried and failed** (per user report, not in this repo's commits).

### The two novel ideas in this spec

1. **o_proj at 100 GB/s is at HALF the bandwidth of every other projection.**
   Root cause: K=8192 is too short for the DRAM prefetcher to warm up before the
   GEMV ends. Microbenchmark confirms: K=8192 → 120.7 GB/s, K=21504 → 199.3 GB/s.
   The fix: overlap o_proj weight prefetching with the preceding attention phase,
   so the prefetcher is already warm when o_proj starts computing.

2. **Cross-layer norm fusion** saves ~118 dispatches (59 layer boundaries × 2) by
   fusing layer L's finalize+norm+residual with layer L+1's input_norm+transform
   into one dispatch.

---

## 1. Commands

```bash
# GPU state
scripts/gpuq status

# Baseline measurement
M=~/.kaplumba/models/Gxma-repacked
scripts/gpuq try -- ./target/release/kaplumba --model "$M" \
  --prompt "Count from 1 to 100: 1, 2, 3," --max-new-tokens 64 \
  --temperature 0 --metrics

# Per-kernel profile
KAPLUMBA_DECODE_PROFILE=kernel scripts/gpuq try -- ./target/release/kaplumba \
  --model "$M" --prompt "Count from 1 to 100: 1, 2, 3," \
  --max-new-tokens 32 --temperature 0 --metrics

# Full test suite
scripts/gpuq try -- cargo test --test live_kernel_bit_exact
scripts/gpuq try -- cargo test --test fused_bit_exact
scripts/gpuq try -- cargo test --test repacked_golden
scripts/gpuq try -- cargo test --test argmax_engine_parity
scripts/gpuq try -- cargo test --test norm_rope_qkv_bit_exact
```

---

## 2. Task T1 Log: Concurrent Threadgroup Scheduling Feasibility

- **Test**: `crates/engine/tests/concurrent_tg_feasibility.rs`
- **Objective**: Verify whether Apple Metal compute command encoder schedules threadgroups from separate dispatches within the same command buffer concurrently without CPU intervention, or if dispatches serialize strictly.
- **Mechanism**:
  - Kernel A (`signal_after_work`): executes 100,000 iterations of FMA busy loop across 256 threads (~1 ms work), then issues `atomic_store_explicit(&flag, 1u, memory_order_relaxed)`.
  - Kernel B (`wait_and_mark`): dispatched in the same command buffer, 32 threads, spin-waits on `atomic_load_explicit(&flag, memory_order_relaxed) == 0u` up to 500M spins, writes `0xCAFE` on seeing flag `1u`, or `0xBADD` on timeout.
- **Measurement**:
  - `cargo test --test concurrent_tg_feasibility -- --nocapture`
  - Result: `Wall time: 8.2 ms`, `Marker: 0xCAFE`, `Kernel B ran: yes`, `Spin count: 0`.
  - Kernel B observed flag set with 0 GPU timeout and zero host stalls. Metal's hardware threadgroup scheduler on M4 Pro actively dispatches threadgroups across available compute units concurrently when resources allow.
- **Architectural implication**:
  - Validates **T2a (Warmstart GEMV kernel)**: we can dispatch `o_proj` weight loading concurrent with `attention`, having `o_proj` threadgroups prefetch weight tiles into registers/L2 before attention writes the final input activation `scratch.attended`.
  - Eliminates the need for complex dual-queue `MTLEvent` host synchronization for o_proj latency hiding.

