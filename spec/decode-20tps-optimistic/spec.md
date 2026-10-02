# DECODE AT 20 TOK/S — THE OPTIMISTIC BUILD

> Synthesis across every spec folder. Nothing here is new physics; it is the
> sum of levers that each spec measured in isolation, plus the two physically
> proven ceilings (254 GB/s streaming, 253 GB/s on the 6 bpw head GEMV).

## The arithmetic that makes 20 real

| quantity | value | source |
|---|---|---|
| weight bytes streamed per token | 12.039 GB | `MEMORY.md` headline |
| measured DRAM streaming ceiling | 250.5–254 GB/s | `probe_bw_250gb`, finding 07 |
| pure streamed floor | 12.039 / 254 ≈ **47.4 ms** | |
| measured non-GEMV floor at 1 group×256/core | **2.6–2.9 ms** | finding 07, §non-GEMV tax |
| **step budget @ 20 tok/s** | **50.0 ms** | |
| floor + floors | 47.4 + 2.8 = **50.2 ms ≈ 19.9 tok/s** | |

20 t/s is not a small win — it is *the wall itself*. Which means it is
reachable precisely when two things are simultaneously true, and both have
been demonstrated on this hardware, separately:

1. **The GEMV streams at the DRAM ceiling.** Already proven: `head/gemv`
   (6 bpw, 192 B/tile) sustains **253 GB/s** (`decode-25tok.../notes §16`).
   The long-K 3 bpw GEMVs are the same kernel shape with 96 B tiles.
2. **The non-GEMV tax is at its floor.** Already proven addressable: 6.70 ms
   today, ~4.0 ms of it named, floor ~2.6–2.9 ms (finding 07).

## The ladder (63.1 ms → ≤50.0 ms)

| # | lever | status | expected | source |
|---|---|---|---|---|
| 1 | sampler `top_k_order` O(n) partition | **landed** `40e018d` | −0.89 ms | decode-20tps notes |
| 2 | 64-bit barrel shift → 32-bit funnel shift | **landed** (§22) | qkv 11.28→8.71 ms; GEMV shapes 196–201 GB/s | 19tok-repacked notes §22 |
| 3 | dual-column threadgroup tiling | **landed** (§23) | x-vector loads halved | notes §23 |
| 4 | cross-layer norm fusion (T3) | **landed** `121ff5b` | +0.09 tok/s | notes §3a |
| 5 | command-buffer rolling (MAX_DISPATCHES=50) | merged `integration/rolling` | +4.37% step time | MEMORY.md |
| 6 | attention split rule `54/kv_heads` | branch `impl/attn-splits`, measured | −2.13 ms/token @8k (isolated lower bound) | MEMORY.md |
| 7 | fold `cache_append` into `norm_rope_qkv` | branch `impl/cacheappend` | −0.21 ms, one fewer dispatch | MEMORY.md |
| 8 | hoist norm weight/residual loads above barrier | branch `impl/hoist` | −0.6…1.2 ms | MEMORY.md |
| 9 | device-side argmax (after 16-bit index fix) | test exists | +0.35% | MEMORY.md |
| 10 | 3 bpw GEMV tile matched to 253 GB/s head | **the remaining big one** | −10…15 ms if 126→250 GB/s closes | parallel_track_a |

Conservative sum of #1–#9 against the 63.1 ms baseline:

63.1 → 62.2 (#1) → −2.57 qkv ALU (#2) → §23 x-reuse (#3) → +0.09 (#4)
→ ×0.956 rolling (#5) → −2.13 (#6) → −0.21 (#7) → −1.0 (#8) → −0.2 (#9)
≈ **~52–55 ms ≈ 18–19 tok/s**, and that is *without* closing the last half
of the GEMV bandwidth gap.

The optimistic statement: #10 does not need a new technique — it needs the
head's own trick (bigger tiles = 192 B per lane-group) applied to 3 bpw,
i.e. two adjacent 3 bpw tiles per 96 B fetch. Every measured production
GEMV after §22 sits at 196–201 GB/s; the ceiling cell in finding 09 is at
254. Any fraction of that gap closes lands directly on the step time:

| GEMV stream rate | GEMV time | + 2.8 ms floor | tok/s |
|---|---|---|---|
| 200 (today, post-§22) | 60.2 ms | 63.0 | 15.9 |
| 220 | 54.7 ms | 57.5 | 17.4 |
| 240 | 50.2 ms | 53.0 | 18.9 |
| **254** | **47.4 ms** | **50.2** | **19.9** |
| 273 (wall) | 44.1 ms | 46.9 | 21.3 |

## Why optimism is licensed here (and not fantasy)

- The 253 GB/s head GEMV and the 254 GB/s pure-stream probe prove the
  hardware *will* deliver 250+; the 3 bpw GEMV is the same code with fewer
  bits per tile.
- The non-GEMV tax was measured, named, and half of it is already assigned
  to concrete fusions with measured values (finding 07).
- Every queued lever has an isolated measurement behind it, not a hope.
- The failure modes that killed the 19/25 tok/s specs (tile-rate physics,
  the 2 bpw cancellation, LICM-invalidated probes) are avoided here: this
  plan changes **zero** bits per weight and adds no new math to the decode
  path — it removes barriers, dispatches, and barrel shifts.

## Acceptance

`gpuq status` idle → 16-paired-sample A/B, target ≤ 50.0 ms/step, greedy
output byte-identical to the current golden, `phys_footprint` unchanged
(≤ 16.5 GiB, 0 swap).
