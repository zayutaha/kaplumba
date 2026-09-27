# Implementation notes

Running log. Appended as work proceeds, not at the end. Factual and dense.

---

## 2026-09-27 — T0.3: the prefill GEMM accumulator, and why bit-exactness is fine

**This is the root cause of the 0.56 TFLOPS, and it is not what the spec
assumed.** The spec's plan.md guessed the prefill GEMM was "scalar" and that
attention might be spilling. The GEMM is far worse than scalar, and it has
nothing to do with the tensor core.

There is no prefill GEMM kernel in `linear.metal` in any real sense. `encode_gemm`
dispatches `linear_gemv_tiled` (`linear.metal:2164`), and its inner loop is:

```metal
for (uint tile_row = 0; tile_row < params.in_features / 16u; ++tile_row) {
    weights[local_row][local_column] = half(decode_state(state, ...));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (local_query < params.queries) {
        for (uint local_input = 0; local_input < 16u; ++local_input) {
            output[output_base + output_index] +=
                float(input[query_base + input_index]) *
                float(weights[local_input][local_output_in_tile]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
```

**The accumulator lives in global memory.** `output` is `device float *`, so every
single multiply is a global read-modify-write. For one projection at 600 tokens
(`gate_proj`, 5376 -> 21504) that is:

    600 * 5376 * 21504 = 6.94e10 global read-modify-writes
    x 8 bytes (4 read + 4 write)          = 554.9 GB of accumulator traffic
    against a weight read of               0.04 GB   -> ~12,600x the weights

So prefill moves roughly **8 bytes of global memory per 2 FLOPs of useful
arithmetic**. It is not compute-bound and never was. That is the whole 0.56
TFLOPS, and it explains why prefill is barely faster than decode: the arithmetic
intensity is being destroyed by the accumulator, not by the weights.

### Why this is bit-exact to fix — the important part

R9/OQ3 requires bit-exactness, and R9's arithmetic said bit-exact scalar fp32
tops out at 8.76 TFLOPS = 152 tok/s, short of 200. That reasoning holds for a
*correct* GEMM. But the fix for this bug does not change the order of operations
at all:

    for tile_row in 0 .. in_features/16      # outer
      for local_input in 0 .. 16             # inner
        output[..] += input[..] * weights[..]

`k = tile_row * 16 + local_input`, so k runs `0, 1, 2, ..., in_features-1` in
strict order, and each output element receives a single fp32 add of a single
product at each k, in that order.

**Therefore: replacing the global accumulator with a register accumulator that
walks k in the same sequential order is bit-exact by construction.** No
tolerance, no reassociation, no changed rounding. It is the same arithmetic with
the same sequence, just not round-tripping through DRAM 6.94e10 times.

That reframes the whole spec:

| | tok/s | bit-exact? |
|---|---|---|
| now | 9.67 | yes (it is the reference) |
| register accumulator, scalar fp32, bounded by the 8.76 TFLOPS SIMT peak | **~90-150** | **yes, by construction** |
| 200 target | 200 | needs the tensor path, whose bit-exactness is still unknown |

So the realistic outcome is a **~10-15x win that is bit-exact and needs no
tensor cores at all**, landing short of 200 tok/s. The remaining gap to 200 is
then the tensor-core question alone, and it should be measured last, on a
correct baseline, rather than first on a broken one.

### Consequences for the task list

- T1.1/T1.2/T1.3 (attribution) are largely answered already: the GEMM dominates
  and the cause is the global accumulator. T1 keeps a reduced form — confirm at
  1k/2k/8k, and still check the attention spill, which may matter more once the
  GEMM is fixed.
- T3 changes from "write a tensor-core tiled GEMM" to **"hoist the accumulator
  into a register, keeping the k order, and retile for reuse"** — a much smaller,
  much safer, bit-exact change. The tensor-core version becomes T7, gated on
  this landing first.
- T0.3's answer is therefore: *bit-exactness and a large speedup are
  compatible.* The open question is no longer "does bit-exactness block the
  target" but "does the tensor path stay bit-exact once we want the last 1.3x".

### Things checked and found not to be bugs

- `local_query = (grid_position.y / 16u) * 16u + local_position.y` looks wrong
  but is just `grid_position.y` rounded down to a multiple of 16, which is
  correct for a 16-row tile. Verified against the `dispatch_2d` grid
  `(out_features/16, queries.div_ceil(16))`.
- The sliding-window memory arithmetic: 50 of 60 layers stop growing past 1024
  context, so KV at 8192 is 1.41 GiB, not ~1.9.
