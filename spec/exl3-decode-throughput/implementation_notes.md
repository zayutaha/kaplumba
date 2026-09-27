
## Implementation (2026-09-27)

### Result

Per-projection, best-of-5 batches after a global clock-ramp warmup:

| Shape | Legacy GB/s | New GB/s | Speedup |
|---|---:|---:|---:|
| `q_proj` 5376x5376 | 21.1 | 40.2 | 1.91x |
| `up_proj` 5376x21504 | 40.8 | 55.6 | 1.36x |
| `down` 21504x5376 | 23.0 | 52.9 | 2.30x |
| `lm_head` 5376x262144 | 53.3 | 63.8 | 1.20x |

What changed: one thread per trellis lane instead of one per output element, one
64-bit window setup per eight codes instead of per code, the four `x` values a
lane needs reused across both column groups, `simd_shuffle_xor` plus a
threadgroup reduction instead of none, and 256-thread groups.

### Four measurement traps hit along the way

1. **The routing was on the wrong function.** The lane-per-thread kernel was
   wired into `gemv_buffer` (the test readback path) but not `encode_gemv`, which
   is what the forward pass and `bench_gemv` actually call. The first A/B showed
   the two kernels measuring *identically* to three decimal places, which is what
   exposed it. Any A/B that comes out suspiciously equal is measuring the wrong
   path.

2. **The benchmark measured the GPU clock ramp as kernel time.** `q_proj` is
   always the first shape, and it varied 0.51-0.86 ms (70%) between runs, so the
   shape measured first always looked worst. Fixed with a 40-iteration global
   warmup before any measurement plus best-of-5 batches. Several earlier
   before/after comparisons in this file were drawn from that noise and are wrong.

3. **Specialising `KA` alone did nothing** (q_proj 12.8 -> 12.1, up_proj 32.7 ->
   26.0). The hypothesis that runtime shift amounts were the cost was wrong. The
   win only appeared once the kernel was actually on the forward path.

4. **A synthetic streaming-bandwidth probe hung** in
   `wait_until_completed` while the GPU was demonstrably healthy. Deleted rather
   than debugged; the real kernel's GB/s answers the same question.

### The actual cost split

Three diagnostic kernels on `q_proj`, isolating the phases:

| Variant | ms | Share |
|---|---:|---:|
| loads only (3 words, no shifts) | 0.336 | 55% |
| + shifts and 8 extractions | 0.580 | 95% |
| + codebook decode (full) | 0.612 | 100% |

**Loads dominate; the codebook is ~5%.** So instruction count on the codebook is
not the lever. The overlapping 16-bit window design means each ring word is
touched by several lanes, and reads are 32-bit at varying `sh` offsets, so they
do not coalesce. That is what the `ALIGNED` fast path addresses, and why halving
the load instructions was worth having.

### The finding that matters more than the kernel

End-to-end decode after this work is **1.18 tok/s** at a 134-token prompt, while
the GEMVs now account for only about a third of the token budget:

- layer weights, 11.0 GB at the measured ~50 GB/s: ~220 ms
- `lm_head`, 1.06 GB at 64 GB/s: ~17 ms
- measured total: ~850 ms/token

So **roughly 600 ms per token is not the linear GEMVs at all.** It is the element
kernels (five RMS norms, RoPE, cache append), the attention kernel, and the
dispatch overhead of ~600 kernel launches across 60 layers. This is now the
dominant cost and the next thing to investigate; a further 4x on the GEMVs would
buy much less than a 2x on the rest.
