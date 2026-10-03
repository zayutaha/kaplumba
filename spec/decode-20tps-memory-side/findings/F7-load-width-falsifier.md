# F7 — the falsifier ran: load-instruction count is worth +3.3%, so F6 stands

**Verdict: throughput is flat across load widths. Quadrupling the bytes per load
(4 B → 16 B, a 4× cut in load instructions at identical bytes) buys +3.3%; halving them
buys +0.5%. The wide lane window removes about a quarter of the weight load instructions
and therefore cannot pay. F6's recommendation to decline it is confirmed by measurement,
not by argument.**

Probe: `crates/engine/examples/probe_bw_250gb.rs` → `load_width_sensitivity`.
Buffer: 6 GB, 40,960 threads, 256/threadgroup, 9 interleaved rounds, min estimator.

## The arms, and why this isolates the variable

Three kernels, **the same bytes in the same grid-stride order**, differing only in the
width of one load:

| arm | B/load | loads per GB | u32 adds per byte |
|---|---:|---:|---:|
| `stream_uint1` | 4 | 250,000,000 | 0.25 |
| `stream_uint2` | 8 | 125,000,000 | 0.25 |
| `stream_uint4` | 16 | 62,500,000 | 0.25 |

Arithmetic intensity per byte is identical in all three (one `u32` add per four bytes:
`uint4` adds 4 lanes per 16 B, `uint2` adds 2 per 8 B, `uint1` adds 1 per 4 B), so the
**only** variable is how many load instructions issue per byte. That is precisely the
quantity the wide lane window changes — it reads the same tiles from DRAM and only removes
loads — which is what makes this a discriminator for that question rather than another
bandwidth number.

## The gate, and proof it can fail

All three arms must return the **same checksum**: adding the same multiset of `u32`s in
groups of 1, 2 or 4 is the same sum. Without that equality the comparison would be three
unrelated numbers.

**Sabotage-verified.** Shifting `stream_uint2`'s loop to start at `gid + 1` (reading a
different byte range) makes the gate fire with a diagnosable message and no GB/s printed:

```
thread 'main' panicked at crates/engine/examples/probe_bw_250gb.rs:248:
assertion `left == right` failed: stream_uint1 and stream_uint2 disagree on the
checksum (0x0a057b80 vs 0xde0ea369). They read the same bytes in the same order, so
they MUST agree. An arm is not reading what this comparison assumes and the numbers
below are void.
```

Sabotage reverted. The gate then passes at `0x0a057b80` — which is byte-identical to the
`stream_uint4` 6 GB checksum this probe printed before the arms existed, so the new
`stream_uint1` arm agrees with the established instrument and not merely with itself.

## The result

```
arm             B/load      GB/s
stream_uint1        4      244.6
stream_uint2        8      245.9     +0.5% vs uint1
stream_uint4       16      252.7     +3.3% vs uint1
```

**A 4× reduction in load instructions is worth +3.3%.**

## What this settles

* **The wide lane window is dead.** It removes ~25% of the *weight* load instructions
  (96 bits instead of 128 per 16 codes, F6 §3), and weight loads are two thirds of this
  kernel's load traffic. Scaled by the measured slope, that is on the order of **+1%**,
  for a rewrite of lane ownership, the reduction tree and bit-exactness.
* **It explains two previously-cited dead ends.** `uint4`/`ulong2` widening was recorded
  as a dead end because "there is no lane-uniform residue class at KA=3". The real reason
  it was never worth doing is in this table: even a perfectly aligned 4× widening buys 3%.
  The alignment problem was real; it was protecting us from a 3% prize.
* **It bounds a family of ideas.** Anything whose only mechanism is "fewer load
  instructions" — wider loads, `ushort2`, merging window reads — is capped near +3% on
  this machine, whatever its arithmetic does.
* **A new ceiling number.** `stream_uint4` at 40960 threads just measured **252.7 GB/s**,
  the highest on record in this tree (previous best 250.5). It is 0.5% above the old
  figure and does not change any conclusion — F4's arithmetic uses the 241–249 band.

## The one thing this does not say

It does not say load instructions are free in *every* context. This is a pure streaming
loop with high occupancy and no dependent work between loads; the decode GEMV has a
dependent chain (extract → decode → FMA) that could hide or amplify the effect. The
honest statement is that the effect is **bounded at ~3% in the most favourable
conditions measured**, and the decode GEMV is not more favourable than a plain stream.