# F8 — every live-dispatch variant has now been measured; the default wins all of them

**Verdict: the remaining untested variants on the live split-K path are both worse.
`_colmajor_2col_sk2` at 16.024 tok/s beats `contig4_sk2` (15.698, −2.0%) and `1col_sk2`
(15.828, −1.2%), each in 6/6 interleaved pairs. The o_proj + down_proj dispatch — 31% of
the step — has no better configuration available.**

## The arms

`LinearKernels::encode_gemv_raw_sk2` (`crates/kernels/src/linear.rs:2090-2148`) has three
branches behind env flags, all of which produce the same fp16 result and differ only in
tiling and grid:

| arm | flag | kernel | grid (N=5376) |
|---|---|---|---:|
| A — default | — | `linear_gemv_ishizuki_k3_colmajor_2col_sk2` | 336 |
| B | `KAPLUMBA_SK2_CONTIG4=1` | `…_colmajor_contig4_sk2` | 672 |
| C | `KAPLUMBA_SK2_1COL=1` | `…_colmajor_1col_sk2` | 672 |

This dispatch carries both `o_proj` and `down_proj` (19.3 ms of the 62.5 ms step by the
F3/F5 table), so it is the second-largest single lever after gate/up.

6 interleaved pairs, one binary, `--temperature 0 --metrics`, 128 greedy tokens:

| round | A default | B contig4_sk2 | C 1col_sk2 |
|---|---:|---:|---:|
| 1 | 16.050 | 15.722 | 15.833 |
| 2 | 16.024 | 15.703 | 15.830 |
| 3 | 16.024 | 15.718 | 15.827 |
| 4 | 16.016 | 15.688 | 15.822 |
| 5 | 16.032 | 15.694 | 15.828 |
| 6 | 16.021 | 15.574 | 15.814 |
| **median** | **16.024** | 15.698 | 15.828 |

B loses in 6/6, C loses in 6/6. Within-arm spread is 0.02–0.15 tok/s, so both effects
are well outside the noise. Note that within-arm spread is again tiny: this box is quiet
enough to resolve 1% effects, which is why these 6-pair runs are informative where the
same-sized effect on a busy box would need 179 pairs (`A3-kernelsel.md`'s power analysis).

## Together with F4 and F5, every live dispatch is now settled

| dispatch | bytes/token | variant tested | result |
|---|---:|---|---|
| gate/up (`fused2`) | 5.202 GB (40.0%) | `fused2_4col`, `fused2_co4` | **−0.7%** (F4) |
| q/k/v (`fused3`) | 1.652 GB (13.5%) | — | only `fused3_2col` reachable without losing fp16 exactness |
| o/down (sk2) | 3.757 GB (30.2%) | `contig4_sk2`, `1col_sk2` | **−2.0% / −1.2%** (this file) |
| o/down | | `sk4`, no-SK | **−0.4% / −1.4%** (F5) |
| head (`k6`) | 1.057 GB (6.4%) | — | already ~257 GB/s, at the box record |

**The shipped configuration is the maximum of every axis that has been measured.** There
is no untried variant left on the decode dispatch.

## What that means for the gap

```
live matvec                    ~206 GB/s      (58.5 ms for 12.0395 GB)
if ALL DRAM cost removed       ~213 GB/s      arm (C), L1-resident, same instructions
register-resident, no loads    ~237.6 GB/s    arm (G), upper bound by construction
pure streaming ceiling         241-252.7 GB/s probe_bw_250gb
```

Decomposed: **~+3.4%** is DRAM, **~+11.6%** is loads-and-addressing inside the loop, and
the rest is the memory system. F7 bounded the "fewer load instructions" family at +3.3%
for a 4× reduction. F6 bounded the wider-window variant at ~+1%. Arm (d) bounded the
variable-shift removal at +2.5%, and that one is structurally unreachable (a 32-bit-aligned
window start would need 24 slots for 32 lanes).

So the honest remaining budget is **about +3.4% to +6%**, all of it inside the loop's
instruction and latency cost, with no identified mechanism that reaches it. That is
**16.0 → at best ~17.0 tok/s**, against a hard bandwidth ceiling of ~18.7 (F4).

## The recommendation

Stop tuning. Every configuration axis is measured; the remaining levers require either a
change to the packed format (a repack of all 411 tensors) or a change to the decode
algorithm (verify-gated speculative decoding, which multiplies accepted tokens per
streamed weight and is therefore the only thing that can pass 20 tok/s — see `tasks.md`
T8).