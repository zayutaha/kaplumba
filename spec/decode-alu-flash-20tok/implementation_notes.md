# implementation_notes — `spec/decode-alu-flash-20tok`

Running log for implementers and handovers.

---

## 1. Environment & Hardware Facts
- **Machine**: Apple M4 Pro (Mac16,7), 14 CPU cores (10P + 4E), 20 GPU cores, 24 GiB unified RAM.
- **Memory Subsystem**: Theoretical wire bus 273 GB/s; measured sustained DRAM copy ceiling is 245–253 GB/s.
- **Model Checkpoint**: Gemma 4 31B (`~/.kaplumba/models/Gxma-repacked`), EXL3 ~3.0 bpw.
  - Weights per token: 12.058 GB exact.
  - Layer geometry: 60 layers total. 50 layers are Sliding Window Attention (`sliding_window = 1024`, `kv_heads = 16`, `head_dim = 256`), 10 layers are Full Attention (`kv_heads = 4`, `head_dim = 512`, `attention_k_eq_v: true`).
- **Initial Baseline**:
  - Live 3 bpw GEMV throughput: ~200–203 GB/s (~59 ms/step).
  - Attention latency: ~7.4 ms at 2k context, rising to ~12 ms at 16k context.
  - Overall decode: ~14.1–14.7 tok/s at 2k context, ~12.3 tok/s at 16k context.

---

## 2. Invariants & Guardrails
- **Bit-Exact Greedy Generation**: Every optimization must match the reference model output bit-for-bit at temperature 0.
- **No KV Cache Quantization**: KV states remain FP16 (`AGENTS.md §8`).
- **GPU Discipline**: All benchmarks must use `scripts/gpuq try -- <cmd>`. Exit 75 indicates GPU busy.
- **No Shared Build Clashing**: Run in dedicated worktree with isolated `CARGO_TARGET_DIR`.

---

## 3. Log of Implementation Steps
*(Entries will be appended as each task T0–T7 is completed.)*

### T1/T2: 4-Deep Tile Loop Unrolling (_tl4) & Sabotage Verification
- Implemented 4-deep tile loop unrolling in `crates/kernels/src/linear_repacked.metal` for:
  - `linear_gemv_ishizuki_k3_fused2_2col`
  - `linear_gemv_ishizuki_k3_colmajor_2col_sk2`
  - `linear_gemv_ishizuki_k3_fused3_2col`
- Retained strict accumulation order per thread to ensure 100% bit-exact arithmetic.
- Sabotage verification: Sabotaged Trip 1 accumulator `acc0_0` with `+ 999.0f * v[3]`.
  - Ran `cargo test -p kaplumba-engine --test live_kernel_bit_exact`.
  - Result: FAILED with exit code 101, detecting bit discrepancy immediately.
- Reverted sabotage and re-tested: PASSED in 15.54s.
