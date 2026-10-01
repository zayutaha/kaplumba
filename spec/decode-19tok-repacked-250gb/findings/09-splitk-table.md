# Finding 09 — Split-K × threads table: the default is already optimal. Do not autotune.

Subagent: `ses_f0cedce20ffebuqpR6brm45rXE`. New files only (`colmajor_sweep.metal`,
`sweep.rs`, `gen_colmajor_sweep.metal`, `sweep_geom.rs`, `scripts/sweep_idle`).
Did not touch `_sk2`/`_sk4`, `linear.rs`, `model.rs`, `generate.rs`.

## The table

TensorFold's grid, built: **(K-split ∈ {1,2,4,8}) × (threads/tg ∈ {128,256,512})**
over all 10 column-major 3 bpw shapes — **120 cells, median of 11 interleaved
rounds each**, 4 GB working set cycled per dispatch, split-K reduction inside the
timed region.

| shape | K-tiles | tile rows | **SK1/TG256** | best cell | GB/s | **gap** |
|---|---:|---:|---:|---|---:|---:|
| sliding/q | 336 | 512 | **172.6** | — | 172.6 | **+0.0%** |
| sliding/k | 336 | 256 | **161.3** | — | 161.3 | **+0.0%** |
| sliding/v | 336 | 256 | **161.1** | — | 161.1 | **+0.0%** |
| sliding/o | 512 | 336 | **173.7** | — | 173.7 | **+0.0%** |
| sliding/gate | 336 | 1344 | **183.2** | — | 183.2 | **+0.0%** |
| sliding/up | 336 | 1344 | 179.8 | sk1_t128 | 183.2 | +1.9% |
| sliding/down | 1344 | 336 | 181.8 | sk1_t512 | 183.9 | +1.1% |
| full/q | 336 | 1024 | 178.4 | sk2_t128 | 180.1 | +0.9% |
| full/kv | 336 | 128 | 134.6 | sk2_t256 | 141.5 | **+5.1%** |
| full/o | 1024 | 336 | 179.5 | sk2_t256 | 180.6 | +0.6% |

**The engine's default (SK=1, TG=256) is already the best cell on 7 of 10
shapes.** `lm_head` excluded: 6 bpw row-major, different kernel family, 819
waves at SK=1.

## Recommendation: do not autotune

```
(a) one global config (SK=1, TG=256): 61.63 ms/token of matvec
(b) per-shape best cell:               61.16 ms/token
difference: 0.76%
```

Byte-weighted, already — and the **end-to-end** difference is *smaller*, because
the 60 layers are a dependent chain. `full/kv`'s +5.1% is real and reproducible
across four independent runs, but it is **41 MB of a 12.04 GB token: +0.17% of
the step.**

TensorFold's 3–19% is about *their* grid on 132 SMs with `blocks=192`. On 20
cores a 512-tile-row shape is already 25 waves at SK=1 — **there is no starvation
left for a K-split to fix.** Their heuristic isn't wrong, it's **inapplicable**:
running it would send `sliding/o`, `sliding/down` and `full/o` to SK=8 × 4
warps, which measure **−2.6% to −36.7%**.

## Confirms T7.6.9, with one exception

On the 9 shapes with ≥256 tile rows the best non-default cell is between
**−0.7% and +1.9%**. At SK≥4 every shape is behind. The single exception is
`full/kv` (128 tile rows, 6.4 waves — the only genuinely under-filled shape).

**Correction to T7.6.9's number:** it measured `gate_up` split-K4 at 168.1; with
`ceil` splits this gets 167.0 against 183.2 — same verdict, and the `ceil` rule
does **not** rescue split-K on wide shapes.

## The `min_tiles=8` guard

**No shape hits the degenerate zero-width case** (needs `tiles_k/SK < 32`;
smallest is 336/8 = 42). **But the remainder is real:** at `tiles_k=336`,
`contig4_sk4`'s `floor-to-32` gives widths `[64,64,64,144]` — **2.2×**, and
3.5× at SK=8, leaving each simdgroup **2 K-tiles** at TG=512. That is
TensorFold's `min_tiles=8` case exactly, and it is the mechanism behind the SK=8
losses. **`ceil`-splitting is the fix** (the non-`_contig4` `_sk4` already has it).

## ⚠ THE TRAP THAT MATTERS MOST HERE

**A contended run does not fail. It fabricates.**

`gpuq`'s lock is **engine-only**, so `gpuq try -- <benchmark>` happily runs
*beside* a 12 GB model load. One sweep returned **plausible medians at spread
0.46 with a fabricated +34% gap**.

`scripts/sweep_idle` gates on `gpuq status` **plus a 0.92 repeatability floor per
cell** and re-measures on rejection. **That floor, not `gpuq`, is what makes a
table like this trustworthy.** Every kernel-sweep number in this project should
be re-checked against a repeatability floor before it is believed.

## Sabotage (AGENTS.md §3.4) — both directions

- Dropping one unrolled tile → caught at 1.32e-1 vs 2e-6 clean.
- Skipping a tile in *only* SK>1 splits → caught by the 20× ladder at 9.24e-2,
  with a message naming the failure.
- Two API guards in `sweep.rs` also verified to fail when broken.

AC6: generator run twice → byte-identical; `generated_block_matches_the_generator`
fails diagnosably if stale.

## Cross-validation of Finding 12 (the ILP probe)

**This table independently agrees with the 8%-load / 63%-ALU partition.** Split-K
exists to relieve load-issue pressure and buys ≤1.9% on every shape; a K-split
that cannot help a load-bound kernel is exactly what an ALU-bound kernel predicts.
Two unrelated probes, same conclusion.
