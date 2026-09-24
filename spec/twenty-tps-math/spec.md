# Goal
Explain why 20 t/s pure autoregressive decode is mathematically impossible on M4 Pro with the current constraints, and propose a path forward.

# The Mathematical Proof

The author of Kaplumba calculated that fusing the `had_in` and `had_out` dispatches would save 76.8 ms, pushing the 140 ms decode time down to ~64 ms (~15 t/s), and with further tweaks, reach 20 t/s.

This model has a fatal flaw: it assumes the EXL3 GEMV kernel can run at 273 GB/s. It cannot, due to a hard ALU/Gather limit on the M4 Pro.

1. **The 300 Gw/s Ceiling**: Both the 6-bit `lm_head` and the 3-bit `gate` run the exact same `exl3_decode_f` inner loop (1 gather + 4 ALU ops per weight). Both kernels hit an absolute ceiling of ~300 Giga-weights per second (Gw/s). 
2. **The Bandwidth Illusion**: Because `lm_head` uses 6-bit weights, 300 Gw/s consumes 225 GB/s of memory bandwidth (close to the 273 GB/s limit). However, because `gate` uses 3-bit weights, 300 Gw/s only consumes 112 GB/s. The GPU is **Compute/Gather bound**, not memory bound. We can never hit 273 GB/s on 3-bit weights.
3. **The Absolute Minimum Time**: Gemma 4 31B has 40 layers, totaling 19.15 Giga-weights per token. At the physical limit of 300 Gw/s, processing these weights takes **63.8 ms**. Adding `lm_head` (4.6 ms) brings the absolute minimum theoretical time to **~68.4 ms (14.6 t/s)**.

20 t/s requires 50 ms per token, which would require an ALU throughput of **425 Gw/s**. We have exhausted vectorization (`float2`, `uint2`), unrolling, and K-splits. The M4 Pro simply cannot execute the bitwise EXL3 decode math that fast.

# The Path Forward (Open Questions)

To reach 20 t/s, we MUST lift one of the standing restrictions. Which one should we lift?

1. **Speculative Decoding**: (Highly Recommended). The industry standard way to break the autoregressive ALU bottleneck. Verify-gated guarantees zero quality loss.
2. **Offline Weight Repacking**: Allows us to change the memory-mapped layout so threads can use coalesced `uint4` loads instead of 32-wide gathers. This breaks the "original positions without copying" rule but costs 0 VRAM at runtime.
3. **Lookup Tables**: Trade L1 cache capacity for ALU cycles, removing the integer math. Breaks the "NO lookup tables" rule.
