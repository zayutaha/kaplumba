# Implementation Notes

- Attempted K-splits (`nsplit=2`, `nsplit=8`) for `fused2`. Did not improve throughput because the GPU is already fully saturated at 1344 threadgroups (67 TG/core).
- Attempted vectorizing the `exl3_decode_f` inner loop using `float2` and `uint2` intrinsics. Did not improve throughput; the Apple Silicon compiler likely scalars the ops, or we are gather-bound.
- Attempted forcing `ld_word` over `simd_shuffle`. Did not improve throughput.
- Discovered the root cause: The M4 Pro ALU / Gather units have a hard cap of ~300 Giga-weights per second for this exact sequence of instructions. 
- Because `lm_head` is 6-bit, 300 Gw/s = 225 GB/s (near saturation). 
- Because `gate`/`up`/`down` are 3-bit, 300 Gw/s = 112 GB/s. We are compute/gather bound, not memory bound.
- The author's assumption that fusing `had_out` would save 76.8 ms and reach 20 t/s was flawed: it ignored the 300 Gw/s ALU ceiling that makes 14.6 t/s the absolute physical limit for pure autoregressive decode on this chip.

- **Tried Offline Weight Repacking (Coalesced uint4/ulong loads)**:
  - Wrote a new kernel `exl3_gemv_repacked` that eliminates all 32-wide L1 gathers and all `simd_shuffle` instructions.
  - It loads a single contiguous `ulong` perfectly coalesced across the warp.
  - **Result**: Throughput peaked at exactly **299 Gw/s**. 
  - **Conclusion**: The gather and the shuffles were NEVER the bottleneck. The bottleneck is the **integer ALU execution ports** on the M4 Pro. The sequence of bitwise shifts, integer multiply, XOR, and float conversion required to decode an EXL3 weight takes a fixed number of ALU cycles. The M4 Pro simply cannot execute these integer instructions faster than 300 billion times per second. 
  - Therefore, memory repacking will not help. We cannot reach 20 t/s by changing the memory layout.
