# F1 — arm (G) has run, and it refutes the instruction-count wall

**Verdict: the tree's "20 tok/s is unreachable" conclusion rested on an
experiment that had never executed. It now has, and it says the opposite.**

Probe: `crates/engine/tests/perf_probe_code_rate_p01.rs`.
Checkpoint: `~/.kaplumba/models/Gxma-repacked` (the colmajor checkpoint the live
kernel dispatches against).

## 1. Why no one had a number

The probe's `load_real_3bpw_tensors` had three independent defects. All three
presented as the same innocuous message, which is why they were never chased:

```
found only 0 3 bpw .trellis tensors under ".../Gxma-repacked", wanted 2
```

| # | defect | reality |
|---|---|---|
| 1 | required `dtype == "U16"` | both checkpoints write `I16` |
| 2 | required `shape.len() == 2` | every tensor is 3-D tile-major |
| 3 | `file.seek(data_offsets[0])` as an **absolute** offset | it is relative to the data section, which starts at `8 + header_len` |

Any one alone is fatal. Defect 3 is the dangerous one: fixing 1 and 2 without 3
would have fed the kernel **header JSON** as trellis, and nothing downstream can
detect that — garbage bytes still decode to finite floats, so arm (a) would have
printed a plausible bandwidth from nonsense. This is the same class as the 4-bit
misaligned-bind bug in `decode-ceiling` (invalid read, silently tolerated).

`data_start + max(data_offsets[1]) == file_len` is now an asserted gate.
**Sabotage-verified**: setting `data_start = 0` fails by exactly **174,656 bytes
= 8 + header_len**.

A fourth defect: arm (G)'s kernel `probe_code_ceiling` was defined and its
`run_ceiling_arm` helper was written and called, but no pipeline was built for
it, because it is deliberately absent from `ARMS` (that list is timed with the
GEMV binding signature). It died with `no pipeline probe_code_ceiling` before
producing a number.

## 2. The 3-D format, for whoever reads this next

```
shape = [N/16, K/16, 48]   dtype I16     # Gxma-repacked  (colmajor)
shape = [K/16, N/16, 48]   dtype I16     # Gxma           (row-major)
```

A 16×16 tile is 256 weights in `shape[2]` u16 words, so `shape[2] == 16 * bpw`:
48 is 3 bpw, 96 is the 6 bpw head. Verified against all five live roles and
`config.json` (hidden 5376, intermediate 21504, 32 heads × head_dim 256):

| role | shape | = |
|---|---|---|
| gate/up | `[1344, 336, 48]` | 21504/16, 5376/16 |
| down | `[336, 1344, 48]` | 5376/16, 21504/16 |
| q | `[512, 336, 48]` | 8192/16, 5376/16 |
| k | `[256, 336, 48]` | 4096/16, 5376/16 |
| o | `[336, 512, 48]` | 5376/16, 8192/16 |

## 3. Arms (a)–(e), real weights

`layers.31.mlp.gate_proj.trellis`, K=5376 N=21504, 115,605,504 codes, 43.4 MB.
min over 11 interleaved rounds; worst med/min 1.02–1.03×.

| arm | GB/s | Gcodes/s | vs (b) |
|---|---|---|---|
| (a) production `_colmajor_2col` | 189.8 | 506.0 | −0.5% |
| (b) replica (fidelity gate) | 190.8 | 508.7 | — |
| (c) **no weight loads** (L1-resident) | **212.5** | **566.8** | **+11%** |
| (d) constant shift amount | 195.7 | 521.9 | −2.5% |
| (e) extraction masks removed | 190.5 | 508.0 | +0.1% |

`probe_2col_ref` is bitwise equal to production on real weights — the fidelity
gate held, so the arms are attributable.

**Two findings from the ratios:**

- **(c) = +11% memory-side.** Same instructions, same addressing, different
  buffer. Yet `decode-ceiling` records "removing every weight load ... changes
  throughput by −0.8%" and concluded the memory system is already hidden. **It is
  not.** That earlier measurement used a different ablation; this one isolates the
  buffer alone.
- **(e) = +0.1%, i.e. the masks are free.** MSL folds the five `& 0xffff`
  extraction masks into one op each, so the real count is **~8.5 instr/code, not
  9.4**, and the arithmetic issue ceiling rises from ~221 to **~243 GB/s**.
  The 9.4 figure was the basis of the "221 GB/s cap" that framed the ceiling.
- **(d) = 2.5%**: the three variable shifts are worth ~2.5%, not free.

## 4. Arm (G) — the decisive one

```
threads 81920, iters 20000, codes 5.2429e10, best 0.08274 s
code rate       633.7 / 634.0 / 637.2 Gcodes/s      (3 runs)
== 3 bpw       237.6 GB/s
20 tok/s needs 631.2 Gcodes/s
```

The kernel is `extract_v_mcg_32` + 8 × `decode_state_mcg` + accumulate, four
groups sequenced exactly as the production inner loop sequences its four
tile-decodes, with **zero memory traffic**. `w` is loop-carried so nothing can be
hoisted or strength-reduced; accumulators are stored so nothing is DCE'd.

**The pre-registered criterion, written before any of this ran:**

> `G < 631 Gcodes/s => 20 tok/s is impossible, unconditionally`

It did not fire. The machine sustains this exact op mix at 634 Gcodes/s, which is
**above** what 20 tok/s requires, and within 5% of the 250.5 GB/s copy ceiling.

### Caveats that bound the claim

Arm (G)'s accumulate is `v[0]+v[1]+…` = 7 ops per 8 codes, where production
spends 8 FMAs, and it performs no address arithmetic. It is therefore an **upper
bound** on the instruction side, deliberately erring *cheaper* than the loop it
calibrates. The honest near-term target is arm (C)'s **568 Gcodes/s = 213 GB/s =
17.7 tok/s**, not 634.

### The unexplained 10%

Two `gate_proj` tensors, **identical shape**, measured in the same runs:

| tensor | production Gcodes/s |
|---|---|
| the first loaded | 463.6 / 469.3 / 464.1 |
| the second | 511.1 / 511.3 / 511.2 |

~10% apart, reproducible, unexplained. Both are 21504×5376 with the same words
count. A memory-placement or shard-offset effect is the obvious candidate, and if
it applies model-wide it is worth ~10% for free. **This is open question Q1.**

## 5. What this changes

| | before | now |
|---|---|---|
| 20 tok/s | "unreachable by making these kernels faster" | **arithmetically reachable; the ALU is not the wall** |
| the wall | instruction issue, 9.4 instr/code → 221 GB/s | **memory-side (~11%) + loop overhead (~13%)**, both unattributed |
| instrument | cited as if it had run | runs, gated, reproducible in ~7 s of GPU |

## 6. What would falsify this

- Arm (G) below 631 Gcodes/s on a rerun (it is 633–637 across three runs).
- Arm (a) ≥ arm (C), i.e. the loads cost nothing and the 11% is an artefact of
  the synthetic buffer rather than the memory path.
- The 634 not transferring: if a kernel with real addressing and 8 FMAs cannot
  exceed ~560, then arm (G) measures a loop that does not exist and the
  reachable target is arm (C)'s 568.
