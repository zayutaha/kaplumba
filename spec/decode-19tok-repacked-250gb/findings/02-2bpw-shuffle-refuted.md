# Finding 02 — 2 bpw is a no-op. The `bpw cancels exactly` identity.

Subagent: `ses_f0ceeff86ffcfko289febgIiuo`. READ-ONLY design task; wrote
`crates/kernels/probe/k2_shuffle.metal` (deliberately outside `src/` so nothing
globs it into the library).

## Verdict

**The design is correct, the geometry IS the identity case hoped for, and it will
lose on AGX by ~11x.** Do not spend a week on this. The only thing worth 20
minutes of GPU time is measuring the shuffle price directly.

## The identity that closes the whole bpw axis

`rate(ka) ∝ ka / lane_words`, so `time ∝ bytes/rate ∝ (K·N·ka/8)/(ka/LW) = K·N·LW/8`.
**bpw cancels exactly.** `LW` is 2 for ka ∈ {1,2,3,4} and 3 for ka ∈ {5,6,7,8}.

| bpw | bytes/token | ceiling | GEMV ms | tok/s |
|---:|---:|---:|---:|---:|
| 1 | 4,717,805,568 | 62.5 | 62.75 | 13.38 |
| **2** | **8,378,646,528** | **125.1** | **62.75** | **13.38** |
| **3** | **12,039,487,488** | **187.6** | **62.75** | **13.38** <- we are here |
| 4 | 15,700,328,448 | 250.2 | 62.75 | 13.38 |
| 5 | 19,361,169,408 | 250.5 | 77.29 | 11.20 |
| 6 | 23,022,010,368 | 250.2 | 92.01 | 9.61 |

**The achievable tok/s is a step function of `K·N·LW`, and 3.0 bpw already sits
on the bottom step.** Lowering bpw cannot help; it can only stay flat and then
fall off at 5 bpw.

**This corrects a claim in the original brief.** "At 2 bpw our scalar kernel
would be 4.0x, so 2 bpw requires the shuffle loader or it is a regression" is
**half wrong — the shuffle loader is ALSO a regression, so there is no version of
2 bpw that isn't.** The scalar mechanical `KA := 2` port of `_contig4` lands at
**13.38 vs today's 13.05 tok/s — within 3 %.**

## `L_LANE = 125.1 G/s` — one parameter, one out-of-sample prediction

Fitted by the two points needing no free parameters:
- 3 bpw (96 B, 64 lane-loads) → **187.9 predicted vs 187.7 measured, 0.1 % error**
- 6 bpw (192 B, 96 lane-loads) → **250.5 vs 250.5, 0.0 %**

## The shuffle price S ≈ 7.25 lane-loads, inferred from existing measurements

Put the two `_simd` datapoints into the model and solve for S:
- 146.5 GB/s (`implementation_notes.md:63`) → **S = 7.25**
- 153.0 GB/s (same file line 109, different shape) → **S = 6.81**

Two independent measurements, two kernel generations, one price, ~7x.

Priced at S = 7.25, the 2 bpw shuffle kernel delivers **16.7 GB/s** — 16
lane-loads + 64 shuffles × 7.25 = 480 ops per 64 useful bytes. **11x worse than
today, 7.5x worse than the trivial scalar port.** At one shuffle per tile it is
32.3 GB/s. There is no version of "2 bpw needs shuffles" that is not a
catastrophe here.

## Why Metal is structurally worse than CUDA for this

- **CUDA:** `__shfl_sync` runs on the SM's **branch/ALU pipe** (128/SM/clk) while
  a global load issues to the **LSU**, a separate pipe (~32/SM/clk). Replacing a
  load with a shuffle is roughly price-neutral-to-favourable. Hence TensorFold's
  +40 %.
- **AGX:** one issue port feeding all pipes, and the SIMD-group crossbar is the
  narrowest resource on it. The repo's own hypothesis (notes line 64) already
  names it: *"serialize through the SIMD crossbar execution port … whereas scalar
  reads into L1 cache are serviced concurrently by L1 read ports."* The measured
  S ≈ 7 is that sentence with a number in it.

**Also a correction to the recorded root cause of the `_simd` loss.** The notes
say "*dynamic* register shuffles … serialize through the SIMD crossbar" — half
right in a way that matters: the cost is the crossbar, but it is **not about the
index being dynamic**. `simd_shuffle_xor` with a lane-id-derived index is equally
"dynamic" and is in our hot reduction today. **S is a per-instruction price, not
an index-quality price.** The 2 bpw index is cheaper, but the thing it is cheap
*at* is still 7x a load.

Corroborating signature: the shuffle variant **helps** on small/latency-bound
matrices (108.5 → 128.1) and **hurts** on wide/bandwidth-bound ones (189.0 →
153.0). "Helps when latency-bound, hurts when bandwidth-bound" is the signature
of a **throughput** cost on a shared issue port.

## 2 bpw geometry (verified lane-by-lane against the normative reader)

Tile = 256 codes × 2 bits = 512 bits = **16 u32 words** (64 B).

```
j0(L) = [15,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,14,14,15]
      = (15 + ((L+1) >> 1)) & 15
```

- **LPW = 2** — 32 lanes → 16 distinct words, exactly 2 each.
- **Lane-linear: YES.** A shift and an add. Monotone, exactly one wrap at L=0→1.
  Matches TensorFold's `base + lane / LPW` with `LPW = 8/K2 = 2` **on all 32
  entries**. `prev = word != lane/2` is exactly "L is even".
- **A fact the brief did not have:** because `sh0` alternates 18/2, **odd lanes
  need only ONE word**, even lanes two. `sh0=2` puts all 30 live bits inside one
  word; `sh0=18` splits 14/14. TensorFold's `w[1]=0u` for odd lanes is correct
  because `d+16 = 30 ≤ 32-2 = 30` exactly.

**A hard arithmetic bound:** 32 lanes × 30 bits of window = 960 bits inside a
512-bit tile, so windows are over-read 2x *by construction* and **no per-lane
loader can beat amp 2.0x at 2 bpw.** Achieving 2.0 requires lanes to *share*
words — which is the crossbar. Per-lane floor without sharing is 3.0x; the
current uniform scheme is 4.0x.

## 3 bpw index — the earlier claim CONFIRMED

Re-derived from the normative reader `crates/engine/src/exl3.rs:256-278`, not from
either Metal file:

```
j0(L) = [23,0,1,1,2,3,4,4,5,6,7,7,8,9,10,10,11,12,13,13,14,15,16,16,17,18,19,19,20,21,22,22]
```

24 distinct values, group sizes **1 and 2** (irregular), one wrap at L=11→12.
**"Neither uniform nor lane-linear" is exactly right, and is not expressible as
`lane / LPW` for any LPW.**

## The 3 bpw `_simd` loss, ranked by cause

1. **Crossbar price (dominant)** — S = 7.25.
2. **8 of 32 lanes idle** in `(lane < 24u) ? t0[lane] : 0u` — pure loss.
3. **No prefetch** — load → shuffle → decode, no double buffer. The prototype has
   `cur`/`nxt` because TensorFold's does.

## Side finding worth taking

The notes' rate sweep reports 4 bpw at 3.0x amplification and 179.8 GB/s. **Post-T6
that is wrong** — 4 bpw has `max sh0 = 20 = 48 − 7·4` exactly, so it is two-word
safe with `LW = 2` and **amp = 2.000x, the same as 6 bpw**, so 4 bpw should also
reach ~250 GB/s. That sweep predates T6 and was shape-confounded; **it is not
evidence against the model.**

## Honest gap: nothing was compiled or run

- `xcrun -sdk macosx metal` → **"missing Metal Toolchain"**. `air-dump`,
  `metallib`, `mtldis` all absent. **No register count, no occupancy figure, no
  disassembly, and none invented.** The register budget in the report is a
  source-level count of live values only. **`xcodebuild -downloadComponent
  MetalToolchain` is required — which also means the project cannot currently
  rebuild at all.**
- Nothing ran on the GPU: `gpuq status` showed pid 57877 holding the model with
  9.9 GiB in use; per AGENTS.md §10.3 no load was added.
- Legality established **algebraically** rather than by the compiler: all indices
  are compile-time constants per lane, checked entry-by-entry against
  `exl3.rs:256-278`; all 32 lanes match; shifts in range; `simd_shuffle` with a
  per-lane `ushort` index is legal MSL (the existing `simd_shuffle_xor` reduction
  relies on it).
- **The 16-byte-aligned loader was proved IMPOSSIBLE**, not left as a risk: a
  lane's pair is `{j0, j0+1}`, a `uint4` block contains it iff `j0 mod 4 < 3`, and
  the four failures are `{3,4} {7,8} {11,12} {15,0}` — the last being the ring
  seam. Metal has no 32-byte per-lane type, and mixing widths costs a second warp
  instruction, restoring amp 4.0. Recorded as a dead end.

## Probability-weighted judgment

| claim | P |
|---|---|
| 2 bpw geometry right, loader legal | 0.95 |
| `simd_shuffle` costs ~7 lane-loads on AGX | 0.80 |
| TensorFold 2-shuffle at 2 bpw runs at 16.7 GB/s (±40 %) | 0.70 |
| 2 bpw scalar ≈ today's speed (13.38 vs 13.05) | 0.90 |
| 2 bpw best case (hypothetical shuffle-free amp-3.0 loader, 148.9 GB/s) gives 15.29 tok/s, +14 % | **0.25** |
| 2 bpw worth requantising a 31B model for | **< 0.05** |

Recommendation: close the 2 bpw direction. Run **only** `proto_price_probe` (5
arms, no dequantisation, ~30 lines) when GPU and toolchain are free — it settles
S and the load-width axis in one go, both currently blocking *any* further
amplification work. If S ≈ 1, 2 bpw becomes a 22 tok/s design and this report is
wrong in the most useful way possible.

---

# ⚠ CONTRADICTION — the recommended "remaining lever" is already withdrawn

This report closes by nominating **the bit-plane layout at `exl3.rs:687-726`** as
the lever that actually remains, "still dead code".

**Finding 11 (spec reconcile) independently examined the same code and reached the
opposite conclusion, and its evidence is stronger because it is about
correctness, not speed:**

> `exl3.rs:664-681` keeps `s & 0x7`; `decode_state_mcg` consumes a 16-bit window
> of which **13 bits survive**. The `imul` mixes all 16 *before* masking, so 3
> bits of input yields **different dequant, not imprecise dequant**. Dead code,
> zero callers, must not be revived.

**The bit-plane layout is not a performance question. It is LOSSY — it silently
produces different dequantised weights — and it is therefore barred by the
no-quality-loss constraint, not merely unpromising.** These two findings cannot
both stand. Finding 11 wins: a scheme that changes the answer is not a lever at
any speed.

Recorded so the next agent does not "revisit the bit-plane repacker" on the
strength of this report alone.
