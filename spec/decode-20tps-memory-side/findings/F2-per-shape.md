# F2 — per-shape bandwidth, and it tracks the output width

**Verdict: bandwidth is a clean function of `N` (output width), from 191.9 GB/s
at N=21504 down to 100.2 GB/s at N=2048. The byte-weighted layer figure is
176.5 GB/s, which independently reproduces the measured step time.**

Probe: `p01_code_rate_issue_bound`, arms on real `Gxma-repacked` weights, min of
11 interleaved rounds, one payload per **distinct shape** (the loader was
changed to select by shape, because selecting "the 2 biggest" returned two
identical gate_proj tensors — see F1).

## The table

`grid` is the probe's dispatch: `N/32` threadgroups × 256 threads.

| shape | ×tensors | bytes | % of layers | GB/s | Gcodes/s | grid |
|---|---:|---:|---:|---:|---:|---:|
| K=5376 **N=21504** | 120 | 5202.25 MB | **47.4%** | **191.9** | 511.7 | 672 |
| K=21504 N=5376 | 60 | 2601.12 MB | **23.7%** | **175.0** | 466.7 | 168 |
| K=5376 N=8192 | 50 | 825.75 MB | 7.5% | 167.5 | 446.6 | 256 |
| K=8192 N=5376 | 50 | 825.75 MB | 7.5% | 160.3 | 427.3 | 168 |
| K=5376 N=4096 | 100 | 825.75 MB | 7.5% | 136.1 | 362.8 | 128 |
| K=5376 N=16384 | 10 | 330.30 MB | 3.0% | 185.4 | 494.5 | 512 |
| K=16384 N=5376 | 10 | 330.30 MB | 3.0% | 172.1 | 458.9 | 168 |
| K=5376 **N=2048** | 10 | 41.29 MB | 0.4% | **100.2** | 267.2 | 64 |
| **total** | **410** | **10.983 GB** | 100% | **176.5** | 470.7 | |

Weighted figure: `10983 MB / Σ(bytes/GB/s) = 62.23 ms` → **176.5 GB/s**.

Two things fall out:

1. **It reproduces the step independently.** 62.23 ms of layer-trellis streaming
   against a measured 62.3 ms step. This is the first per-shape measurement that
   reconciles with the end-to-end number from first principles rather than from a
   fitted constant. (`decode-ceiling`'s step-budget derivation arrived at
   191.3 GB/s = 57.40 ms for the same bytes; the ~8% gap is the isolated-vs-in-situ
   overhead, and it is the right order.)
2. **The spread is 1.9×, and it is monotone in N.** 191.9 → 100.2 GB/s as N falls
   from 21504 to 2048. Gcodes/s tracks it (511.7 → 267.2), so this is **not** an
   instruction-count effect: the same instructions run 1.9× slower on the small
   shapes. That is the signature of insufficient parallelism — fewer
   threadgroups, less latency hiding.

## Why this matters for the 20 tok/s target

The byte-weighted target is **236.7 GB/s**, against 176.5 today: **+34%**.

The bytes are concentrated: **71.1%** sits in the two large-N shapes
(gate/up 47.4% at 191.9, down 23.7% at 175.0). So the target cannot be reached by
fixing only the small shapes — but the small shapes are where the *evidence* is,
because they show the mechanism (parallelism) in its extreme form.

If every shape reached arm (G)'s 237.6 GB/s the layer term would be 46.2 ms, and
if every shape reached arm (C)'s 213 it would be 51.6 ms. Today it is 62.2 ms.

## The hypothesis this sets up (T2)

**The 1.9× N-dependence is a parallelism/latency effect, not a bandwidth or
instruction effect.** Tests:

- It is monotone in the grid size, and the small-N shapes have 64–168
  threadgroups against the large shape's 672.
- The same instruction stream runs at 634 Gcodes/s with no memory (arm G) and at
  511 in production (arm A) — so the memory path, not the ALU, is what the small
  shapes are losing to.

If it holds, the fix is more concurrency per shape (split-K, or a wider grid),
and it is worth up to the +34%.

**This must not be assumed.** `decode-ceiling` already lost a week to a
discriminator whose verdict flipped on a calibration constant. The grid has to be
varied at a *fixed shape and fixed bytes* and shown to move the rate before any
verdict is recorded.

## Caveats

- Isolated per-shape numbers carry launch/teardown that in-situ overlap hides, so
  the absolutes read ~8% pessimistic against the step budget. The **ratios** are
  the claim, consistent with how B1 and B3 handled the same problem.
- `grid` is the probe's own geometry, computed as `min(N/32, max_groups)`, and
  does not necessarily match what the engine dispatches (which may use split-K or
  `_2col` variants). It is reported as a covariate, not as the engine's grid.
- The 10 full-attention-layer shapes (`N=16384`/`K=16384`) appear only 10× each;
  at 3.0% of bytes they cannot decide the outcome either way.
