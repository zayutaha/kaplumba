# F6 — the wide lane window, worked out concretely: it is ~17%, not 50%, and that is not enough

**Verdict: the headline "2.67× → 1.33×" does not survive contact with the ring's bit
alignment. The achievable amplification is 2.67× → 2.0× on weight loads, and since
`x` loads are a third of the traffic and do not improve, the total is ~1.5 → ~1.25 bytes
of load traffic per code (−17%). Expected speed-up: well under 10%. Not worth the
rewrite, and it does not reach the target either way.**

Worked out from `pack_integer_windows` (`crates/engine/src/exl3.rs:330-353`),
`extract_v_mcg_32` and `_fused2_2col` (`linear_repacked.metal:53-64`, `:491-511`).

## 1. The bit→(output, input) map, recovered

`pack_integer_windows` writes each state as a **16-bit window** into a 768-bit ring at a
**3-bit stride**:

```rust
let s = lane * 8 + i;                                  // state index, 0..256
let start = ((s + 1) * ka + ring_length - 16) % ring_length;   // ka = 3, ring = 768
```

and `extract_v_mcg_32` reads back a 64-bit window per lane at
`lo0 = (lane*8 + 1)*3 + RING - 16`, pulling `(A_hi >> k) & 0xffff` at 3-bit strides.

Combining that with the kernel's own assignment (`r0 = (lane % 4) * 2`, `v[0..3]` →
`acc*_0`, `v[4..7]` → `acc*_1`, `simd_shuffle_xor` by 1 and 2, `lane/4` indexing
`red`), the state for output `o` and input `i` sits at ring position

```
R(o, i) = 96*(o mod 8) + 24*g + 755 + 12*(o >= 8) + 3*idx(i)
```

where `g = i`'s group (`i ∈ {2g, 2g+1, 2g+8, 2g+9}`) and `idx(i) ∈ 0..4`. This is
knowable, and it is what a re-partition would have to preserve. **No repack is needed**
— the ring order is unchanged; only which lane owns which bits moves.

## 2. Why 48 bits do not fit in 64

One output's two adjacent groups occupy a **contiguous 48-bit span** (24 bits per group),
so 16 lanes × 48 bits tile the ring exactly. That is the attractive part.

The window start is `755 + 48L`. Its word offset is

```
(755 + 48L) mod 32  =  19 + 16L mod 32  =  { 19  (L even),  3  (L odd) }
```

So `sh ∈ {19, 3}` — it is **never 0**. At `sh = 19` the window needs bits `[19, 67)`,
which is **67 bits: three words, not two.** Half the lanes would need three and half two,
and a branch on that is worse than paying for three everywhere.

```
current : 2 words = 64 bits for  8 codes = 24 useful bits  ->  64/24 = 2.67x
wide    : 3 words = 96 bits for 16 codes = 48 useful bits  ->  96/48 = 2.00x
ideal   : 2 words = 64 bits for 16 codes = 48 useful bits  ->  64/48 = 1.33x  NOT REACHABLE
```

The 1.33× figure assumed a 48-bit window fits in the same 2 words. It does not, because
the ring's `-16` rotation makes every window start at a 3-bit or 19-bit offset rather than
a 32-bit boundary.

## 3. And weight loads are only two thirds of the traffic

Per lane per K-iteration in the live `_fused2_2col` (`:491-511`), which covers 2 columns ×
2 K-tiles:

| | tiles | codes | weight loads | `x` loads | total |
|---|---:|---:|---:|---:|---:|
| current | 4 | 32 | 8 × u32 = 32 B | 8 × half = 16 B | 48 B |
| per code | | 1 | 1.00 B | 0.50 B | **1.50 B** |

A 16-code lane covers one tile's worth of span per window, so to hold the same 2-column ×
2-K-tile shape it needs 3 words per tile:

| | tiles | codes | weight loads | `x` loads | total |
|---|---:|---:|---:|---:|---:|
| wide | 4 | 64 | 4 tiles × 3 × u32 = 48 B | 2 K-tiles × 8 half = 32 B | 80 B |
| per code | | 1 | 0.75 B | 0.50 B | **1.25 B** |

**−17% load traffic per code**, not −50%. The `x` term is unchanged at 0.50 B/code and is
a third of the total; the weight term falls from 1.00 to 0.75.

## 4. What that is worth, against what it costs

Three things bound the upside, and they compound against it:

* **−17% of load bytes is not −17% of time.** Arm (c) in F1 — the identical instruction
  stream with L1-resident weights, i.e. all DRAM cost removed — is worth only **+11%**.
  The wide window removes none of the DRAM traffic (the same tiles are read), it removes
  *load instructions*. The recorded sensitivity of this kernel to load-instruction count
  is small: `uint4`/`ulong2` widening was a dead end, `ushort2` was already the fast path,
  and 3→2 word trimming bought +7% for a −33% change in lane-loads.
* **The `x` loads do not amortise.** They are the reason the four-column experiment
  existed, and 4 columns per threadgroup measured **−0.7%** (F4).
* **Extraction cost per code is roughly flat** (the same shift/mask pattern, more
  instances), so the ALU side does not improve, and F1's arm (G) puts the instruction
  ceiling at 237.6 GB/s — only 15% above where the kernel already is.

Realistic outcome: **+3 to +8%**, i.e. 16.0 → 16.5–17.3 tok/s, against a rewrite that
changes 16 lanes' worth of ownership, the reduction tree, and the accumulator grouping —
and therefore has to re-earn bit-exactness from scratch (§AGENTS 3.4), with the split-K
precedent showing that a 1-ULP reassociation can move 43 of 128 greedy tokens.

## 5. Recommendation

**Do not implement it.** It is a high-risk rewrite for a payoff that is smaller than the
one already measured-and-lost in F4, and it cannot reach 20 tok/s under any outcome — the
bandwidth route's ceiling is ~18.7 (F4 §Arithmetic) and this lever does not reach that.

If a load-instruction sensitivity measurement is wanted before discarding the idea, the
cheap version is one probe arm: the replica with its two `u32` loads replaced by a single
`uint2` at the same address, reporting GB/s. **The lanes whose `j0` is odd cannot use
`uint2`** (the recorded `uint4`/`ulong2` dead end), so the arm would have to run on the
even-`j0` half only — which is 50% of lanes and therefore already a −50% experiment on the
axis in question. If that arm does not move the rate materially, the wide window is dead
on arrival and this analysis is confirmed.

## 6. What is left standing, after F4, F5 and F6

| route | status |
|---|---|
| more loads in flight | closed ×3 (F3 §6) |
| column tiling 2col → 4col | measured, −0.7% (F4) |
| o_proj variant selection | measured, −1.4% / −0.4% (F5) |
| wider per-lane window | −17% load bytes, §2/§3 above; not worth it |
| **fewer bytes/token, or more tokens per weight pass** | the only route that reaches 20 |

The last row is the whole of what remains, and it is a different spec: verify-gated
speculative decoding / MTP (AGENTS §8 permits it; no such infrastructure exists in this
tree).
