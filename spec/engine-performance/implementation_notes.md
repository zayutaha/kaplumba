# Consolidated Implementation & Engineering Notes

## 1. Commit History & Architectural Progression

| Commit | Subsystem | Description & Measured Impact |
|---|---|---|
| `f20f301` | Memory | Mapped `embed_tokens` & `lm_head`; resident RAM dropped 15.12 → 11.51 GiB. |
| `8c3ea95` | Decode | GPU greedy sampling & command pipelining; decode reached 6.99 tok/s. |
| `89eacc8` | Analysis | Corrected M4 Pro ALU roofline; proved bandwidth (22.6 tok/s) is the ceiling. |
| `b811675` | Decode | Fixed benchmark calibration; FMA peak measured at 4.35e12 FMA/s. |
| `49814d4` | Decode | Matvec lane-invariant hoisting and loop unrolling; decode reached 7.43 tok/s. |
| `e1d72f2` | Decode | Coalesced decode attention reduction across key splits; decode reached 7.52 tok/s. |
| `22c6325` | Decode | Hoisted shared input transforms across attention heads; decode reached 7.87 tok/s. |
| `63f94e5` | Decode | Added grid-stride matvec kernel (paused unswept to focus on prefill). |
| `9619b81` | Prefill | Created initial prefill 200 tok/s specification. |
| `143937c` | Prefill | Resolved KV cache allocation size (1.41 GiB at 8192 context). |
| `11aef59` | Prefill | Register accumulator in `linear_gemv_tiled`; prefill rose 9.67 → 9.86 tok/s. |
| `500a902` | Prefill | Specialized prefill attention on `HEAD_DIM=256` and `512`; prefill rose to 10.26 tok/s. |
| `724fa4d` | Prefill | Restored generic prefill attention fallback alongside specialized kernels. |
| `12fa4d8` | Prefill | Threadgroup attention (`online_attention_tg2_256/512`) with register accumulators; prefill reached 11.44 tok/s. |
| `ba9263f` | Prefill | Tested 64×64 tile with KCHUNK=64/16; degraded to 8.27 / 2.58 / 6.90 tok/s; reverted cleanly to 11.38 tok/s. |

---

## 2. Experimental Post-Mortems (What Failed and Why)

### 2.1 The Global Atomic Accumulator Myth (+2% actual gain)
- **Hypothesis:** `linear_gemv_tiled` used atomic `output[idx] += val` in global memory across all $k$ chunks. We calculated that this generated 554 GB of DRAM traffic, theorizing it was the sole cause of the 0.56 TFLOPS prefill bottleneck.
- **Implementation:** Thread-local register accumulation walking $k$ strictly in order (`11aef59`).
- **Result:** Prefill improved from 9.67 to only 9.86 tok/s (+2%).
- **Root Cause:** The atomic read-modify-writes were hitting the M4 Pro GPU L1/SLC cache hierarchy, not saturating DRAM bandwidth. The true bottleneck lay elsewhere.

### 2.2 The 64×64 Tile Failure (Theoretical 12.6 TF vs Actual 8.27 tok/s)
- **Hypothesis:** A 16×16 tile has an arithmetic intensity of only 13.4 FLOP/byte (capping at 3.2 TFLOPS). Widening to 64×64 yields 53.8 FLOP/byte, raising the theoretical ceiling to 12.6 TFLOPS.
- **Implementation:** 256 threads per group; each thread owned 16 output queries and 1 column. Trellis staged into threadgroup memory.
- **Results:**
  - 64×64, `KCHUNK=64`: **8.27 tok/s**
  - 64×64, `KCHUNK=16` with dynamic loop: **2.58 tok/s**
  - 64×64, `KCHUNK=16` with `#pragma unroll`: **6.90 tok/s**
  - 16×16 restored: **11.38–11.44 tok/s**
- **Root Causes:**
  1. *Local memory spilling:* Guarding the loop with a runtime `nvalid` trip count prevented unrolling, causing the compiler to spill the 16-register accumulator to local thread memory (the 2.58 tok/s reading). Unrolling with compile-time 16 recovered to 6.90 tok/s.
  2. *Occupancy collapse:* 64×64 with `KCHUNK=64` allocated 20 KB of threadgroup memory per group (8 KB weights, 8 KB inputs, 4 KB trellis), restricting the GPU to 1 threadgroup per core and killing latency hiding.
  3. *Weight decode overhead:* Each thread performed 16 decodes per chunk, each requiring ~10 ALU ops with integer division and modulo (`c / 16`, `c & 15`, MCG multiplication). The ALU time spent decoding weights eclipsed any benefit from the wider tile.

### 2.3 One-Thread-Per-Dimension Attention (8.0 tok/s)
- **Hypothesis:** To avoid `float accum[512]` register pressure in attention, assign one thread per head dimension ($D=256$).
- **Result:** Throughput dropped from 10.26 tok/s to **8.00 tok/s**.
- **Root Cause:** In this layout, every single thread recomputed the full dot-product score across the entire dimension independently. This multiplied arithmetic and key-fetch memory bandwidth by 256×.

---

## 3. Critical Profiling & Ablation Data

### 3.1 Breakdown at 600 Prompt Tokens (Commit `12fa4d8`)
Total prefill time per token: **87.4 ms** (11.44 tok/s).

- **Attention Cost:** **~44.7 ms/token** (~51%)
  - Phase 1 (score dot-products): ~12 ms
  - Phase 1b (serial online-softmax on thread 0): ~25 ms
  - Phase 2 (value accumulation): ~8 ms
- **Linear GEMM Cost:** **~42.7 ms/token** (~49%)
  - Staging & decoding weights: ~28 ms
  - FMA accumulation: ~14 ms

### 3.2 The M4 Pro Hardware Bounds
- Apple M4 Pro (20 GPU cores) @ ~1.45 GHz:
  - Peak SIMT fp32 FMA: **4.38e12 FMA/s = 8.76 TFLOPS**.
  - Gemma 4 31B GEMM: 2 × 28.74e9 = **5.75e10 FLOPs/token**.
  - **Absolute scalar fp32 ceiling:** $8.76 \times 10^{12} / 5.75 \times 10^{10} = \mathbf{152.3 \text{ tok/s}}$ (GEMM only, zero time for attention).
  - Apple Metal `simdgroup_matrix` (tensor cores): Capable of 30–50+ TFLOPS, but computes sums in SIMD hardware order rather than strict sequential order.

---

## 4. Key Developer Gotchas & Verification Commands

1. **Metal Shader Recompilation Trap:**
   - Metal shaders are embedded into the CLI binary via `include_str!`.
   - Running `cargo build --release --workspace --examples` does **not** update `kaplumba-cli`.
   - Always compile with:
     ```bash
     cargo build --release -p kaplumba-cli
     ```
2. **End-to-End Prefill Measurement:**
   ```bash
   M="/Users/zayaantaha/.kaplumba/models/Gxma/Gxma"
   PP=$(python3 -c "print('Explain the theory of relativity. '*100)")
   ./target/release/kaplumba --model "$M" --prompt "$PP" --max-new-tokens 8 --temperature 0 --metrics
   ```
3. **Reference Output Integrity:**
   ```bash
   for q in "What is the capital of France?" "Count from one to five." "Name three primary colors." "Hi" "What is 17 times 23?"; do
       ./target/release/kaplumba --model "$M" --prompt "$q" --max-new-tokens 24 --temperature 0
   done
   ```
   *Expected golden answers:*
   - Paris
   - 1, 2, 3, 4, 5
   - red, blue, and yellow
   - Hello! How can I help you today?
   - 17 times 23 is 391
