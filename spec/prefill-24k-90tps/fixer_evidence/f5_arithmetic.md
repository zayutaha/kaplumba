# F5 working arithmetic

> **SUPERSEDED by `implementation_notes.md` §23 for the argument.** This file is
> the per-projection FLOP derivation it is built on, kept because every number in
> §23 traces to a line here and because the layer-kind FLOP counts are the part
> that is easy to get wrong (`head_dim` is 512 on the 10 full layers and 256 on the
> 50 sliding ones, so `q_proj` and `o_proj` have two shapes).
>
> **One correction inside it.** §4 below proposes hoisting the Q fragment into
> registers to raise the instruction-issue ceiling from 8.25 to 9.5 TFLOP/s. §22
> **measured** that: it is **92% slower** (arm 7), because 64-128
> `simdgroup_half8x8` per lane spills. The instruction-count model is also
> falsified in general — it predicts `BQ=2` should beat `BQ=1` by 1.33x and the
> bench says it is 19% *worse*. §23.5 records this. **Do not use §4.**

Every measured number below comes from `KAPLUMBA_GPU_PROFILE=1` at a 16 394-token
prompt: 17 chunks x 60 layers = 1020 layer-chunks, `own_ms` per kernel, from
`spec/prefill-24k-90tps/adversary_evidence/profile_16384_onedispatch.log`.
Geometry from `config.json`: 60 layers = 10 `full_attention` (head_dim **512**)
+ 50 `sliding_attention` (head_dim **256**, window 1024); `q_heads` 32,
`kv_heads` 16, hidden 5376, intermediate 21504. Note `head_dim` differs between
the two layer kinds, so `q_proj`/`o_proj` have two different shapes -- getting
that wrong changes the FLOP count by 8%.

## 1. Where the context-independent GEMM + other goes

FLOP per layer-chunk at `M = PREFILL_CHUNK = 1024` is `2 * M * K * N`:

| projection | K | N sliding / full | GFLOP per layer-chunk |
|---|---|---|---|
| q_proj | 5376 | 8192 / 16384 | 90.17 / 180.35 |
| k_proj | 5376 | 4096 / 8192 | 45.09 / 90.17 |
| v_proj | 5376 | 4096 / 8192 | 45.09 / 90.17 |
| o_proj | 8192 / 16384 | 5376 | 90.17 / 180.35 |
| gate, up, down | 5376 / 21504 | 21504 / 5376 | 236.85 each |

Per token:
`50 x (89.126 + 3 x 231.299) + 10 x (180.345+90.172+90.172+180.345 + 3 x 231.299)`
= **6.0238e10 FLOP/token**. Measured GEMM `own_ms` = 146 770.2 at 16 394 tokens
= 8.954 ms/token, so

    6.0238e10 x 16394 / 146.770 s  =  6.73 TFLOP/s  =  43.1% of the 15.6 TFLOPS
                                     tensor peak measured by examples/mma_peak.rs

Per projection (`s at 24 590` scales `own_ms` by 24590/16394):

| projection | own_ms @16 394 | s @24 590 | share | TFLOP/s | % peak |
|---|---|---|---|---|---|
| down_proj | 34 339.0 | 51.5 | 23.0% | 6.63 | 42% |
| up_proj | 34 320.7 | 51.5 | 23.0% | 6.63 | 42% |
| gate_proj | 34 318.1 | 51.5 | 23.0% | 6.63 | 42% |
| q_proj | 15 869.4 | 23.8 | 10.7% | 6.37 | 41% |
| o_proj | 15 574.5 | 23.4 | 10.5% | 6.49 | 42% |
| k_proj | 6 495.2 | 9.7 | 4.4% | 7.78 | 50% |
| v_proj | 5 853.3 | 8.8 | 3.9% | 8.63 | 55% |
| element kernels + head | 2 187.2 | 3.3 | 1.5% | -- | -- |
| **total** | **148 957.4** | **223.5** | | **6.73** | **43%** |

**The `spec/decode-20tps` o_proj lead is refuted for prefill.** That lead is
`gemv_bandwidth.rs` at `K = 8192` reading ~100 GB/s for o_proj against ~200 GB/s
for the others -- a **decode GEMV (`M = 1`) measurement**, and its own §18.2
already walked the production number back to ~126 GB/s. In prefill `M = 1024` and
every projection is tensor-bound: o_proj runs at **98% of down_proj's
efficiency** (`own_ms` ratio 15.574/34.339 = 0.453 against a FLOP ratio of
0.444). The ordering is not a bandwidth signature: the projections with the
*longest* K are the slowest (q_proj 6.37, o_proj 6.49) and the one with the
*shortest* K and narrowest N (v_proj, K 5376 / N 4096-8192) is the **fastest** at
8.63. Recovering the whole 2% o_proj deficit is 0.45 s of a 273.2 s budget.

The reason the GEMM is uniformly at ~43% and not bandwidth-limited: per chunk it
streams the whole 11.6 GB of packed weights once against `6.168e13` FLOP, an
arithmetic intensity of **5 317 FLOP/byte**. At 6.73 TFLOP/s that needs
**1.27 GB/s of a 273 GB/s wire** -- 0.5%. The 273 GB/s figure is irrelevant to
prefill GEMM, exactly as §12 found for attention.

## 2. Attention, split by layer kind

Per-layer `own_ms` totals at 16 394: a full layer 7 566.7 ms, a sliding layer
2 653.1 ms. Each layer's own GEMM is `FLOP/rate`: sliding
`9.8337e11 / 6.73e12 = 146.1 ms` per chunk, full `1.2516e12 / 6.73e12 = 186.0 ms`.
(The 27.8 ms difference is `o_proj`'s doubled `K`.) So per chunk:

| | total/layer | own GEMM | **attention** |
|---|---|---|---|
| full (average over the 17 chunks) | 444.98 | 186.0 | **259.0 ms** |
| sliding | 156.06 | 146.1 | **10.0 ms** |

Check: `10 x 259.0 + 50 x 10.0 = 3 090 ms` per chunk x 17 = 52.5 s against the
measured 59.35 s; the 12% residual is the quadratic shape of `A_f(c)`, which is
larger than its own average in the late chunks. Take `A_f(avg) = 270.0 ms`.

FLOP, per layer over the whole prompt:

* full: pairs `sum_{i<16394}(i+1) = 1.3449e8`, **65 536 FLOP per pair**
  (32 q heads x 2 x 512 dims, twice for QK and PV) -> 8.814e12 per layer, x10
* sliding: pairs `16394 x 1024 = 1.6786e7`, **32 768 FLOP per pair**
  (32 x 2 x 256, twice) -> 5.50e11 per layer, x50

| | at 16 394 | TFLOP/s | % peak | FLOP at 24 590 | s at 24 590 |
|---|---|---|---|---|---|
| 10 full layers | 45.9 s | **1.91** | 12% | 1.9817e14 | **103.8** |
| 50 sliding layers | 8.5 s | **1.85** | 12% | 4.126e13 | **22.3** |
| **attention** | **54.4 s (measured 59.35)** | | | 2.394e14 | **126.1** |

The two layer kinds agree at ~1.9 TFLOP/s, which is what makes this credible:
it matches `attention_qk_mma_bench` on the same shape (**1.79 TFLOP/s** at 24 576
keys, `flash bq1`) and it matches the adversary's total (127.5 s at 24 590
against 126.1 s here). The adversary's *internal split* (50.2 s full / 8.9 s
sliding) is the part that does not reproduce; its total does.

## 3. The budget

    273.2 s (90 tok/s)  -  223.5 (GEMM + other)  -  16.2 (inter-boundary gap)
    =  33.5 s of attention for 2.394e14 FLOP  =  7.15 TFLOP/s = 46% of peak

Attention must go **126.1 -> 33.5 s, a 3.8x cut**. Holding the sliding layers at
today's 1.85 TFLOP/s leaves 11.2 s for the full layers, i.e.
`1.9817e14 / 11.2 = 17.7 TFLOP/s` -- above the 15.6 TFLOPS peak, so the
"flash-only" route is **arithmetically impossible** and the sliding kernel has to
move too. (The adversary got 20.4 s / 9.7 TFLOP/s by crediting the sliding
layers 13.3 s instead of 22.3 s.)

## 4. The instruction-issue ceiling of the flash kernel

Per 64-key tile, per simdgroup (`HEAD_DIM` 512, `NSG` 8, `NTILES` 64, `NPER` 8):

| | QK | PV | instr | MMAs | instr/MMA |
|---|---|---|---|---|---|
| `BQ = 1` | 64 x (1 `kf` + 1 `qf` + 1 MMA) = 192 | 8 x (1 `af` + 8 `bf` + 8 MMA) = 136 | 348 | 128 | **2.72** |
| `BQ = 2` | 64 x (1 + 1 + 2) = 256 | 8 x (2 + 8 + 16) = 208 | 484 | 256 | **1.89** |

A `simdgroup_half8x8` is 32 lanes x 2 halfs = **4 B per lane**, so each `kf`/`bf`
device load is 8 rows x 16 contiguous bytes at a `kv_stride` of 4 KiB: eight
separate memory transactions for 128 useful bytes, i.e. a 2x sector over-fetch.

Ceiling if instruction issue were the only limit: `15.6 / 2.72 = 5.7 TFLOP/s` at
`BQ = 1`, `15.6 / 1.89 = 8.25 TFLOP/s` at `BQ = 2`. Measured: **1.91**.

The `qf` fragment is a function of `(b, d8, head)` alone -- nothing in the
key-tile loop changes it -- yet the shipped code re-executes
`simdgroup_load(qf, qstage + ...)` once per (tile, `d8`, `b`). Hoisting it into
registers (`BQ x NTILES` simdgroup registers per lane: 64 at `BQ = 1`, **128 at
`BQ = 2`**) gives `BQ = 2`: `64 x (1 + 2) = 192` + 208 = 420 instr / 256 MMA =
**1.64 instr/MMA -> 9.5 TFLOP/s** ceiling.

## 5. The honest best case

Everything above is a *ceiling*, so this is an upper bound, not a forecast. At the
measured 24 590-token shape:

| scenario | attention | total s | tok/s |
|---|---|---|---|
| today | 126.1 | 365.8 | **67.2** |
| flash at its `BQ = 2` issue ceiling (8.25), sliding unchanged | 24.0 + 22.3 = 46.3 | 285.9 | **86.0** |
| both at their issue ceilings | 24.0 + 5.0 = 29.0 | 268.7 | **91.5** |
| both at their ceilings and the inter-boundary gap halved | 29.0 | 260.6 | **94.4** |

So AC2 straddles 90 only at the exact instruction-issue ceiling of both attention
kernels, with zero memory stalls, which is a bound no measured kernel here
reaches (the best component in the whole engine, the FFN GEMM, is at 43% of
tensor peak and is uniformly so across all seven projections).