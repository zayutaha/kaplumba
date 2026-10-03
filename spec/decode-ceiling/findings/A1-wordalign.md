# A1 — Word-aligned lane repack (1 load/lane): **INFEASIBLE**

**Hypothesis A1** (`spec/decode-ceiling/hypotheses.md`, Round 3): repack the on-disk
EXL3 tile so each lane's 8 codes occupy one 32-bit word, cutting lane loads 2 → 1 and
`amp = 8/KA = 2.667` → 1.333, **without** growing the tile from 24 words (96 B).

**Verdict: INFEASIBLE.** Not "hard", not "not worth it" — the crux condition fails by
arithmetic, and it fails for *every* rate, *every* lane→code map, and *every* permiss-
ible permuter. The tile must grow to **148 B (+54%)** for a lossless 1-load layout, and
148 B/tile is already **23% slower** than today's *total* tile time before a single
instruction is saved. Growing it to the brief's 128 B does not even fit the payload.

Method: CPU-only static derivation, transcribed from the live kernels and the CPU
reference. No model, no GPU, no checkpoint, no network, no microbenchmark.

---

## 1. The arithmetic that decides it

### 1.1 A lane's self-contained payload is 37 bits, not 24

`unpack_integer` (`crates/engine/src/exl3.rs:231-279`) is the ground truth. For lane `L`,
code index `i`, with `shift = lo % 32`, `j0 = (lo/32) mod 24`, `j1 = j0+1`:

```
a   = ((tile[j0] << 32) | tile[j1]) << shift
w_i = (a >> (64 - (3i + 16))) & 0xffff          i = 0..7
```

Two facts fall out, and the record has the second one wrong:

1. **The 8 windows are a shifting register, not 8 independent codes.**
   `w_{i+1} = (w_i << 3) | chunk_i`, where `chunk_i` is the 3 *fresh* ring bits entering at
   the top. `w_i`'s low 3 bits are `w_{i-1}`'s bits 9..11 — a *derived* quantity.
   A lane therefore needs **16 bits of anchor + 7 × 3 bits of chunks = 37 bits.**
   `w_i & 7` is not the payload. This is exactly why the withdrawn bit-plane repacker
   (`exl3.rs:687-726`, zero callers) is lossy: it stored `w & 7` and threw the payload away.
2. **The 8 windows touch 37 ring bits, but not 37 contiguous ones.** `a`'s high half is
   `tile[j0]` (ring `32·j0 + 0..31`) and its low half is `tile[j1]` (ring `32·j0 + 32..63`),
   so `a`'s two halves walk the ring in **opposite** directions. Measured, per lane:

   | code bits in `j0` | code bits in `j1` | lanes |
   |---:|---:|---:|
   | 5 | 19 | 8 |
   | 13 | 11 | 8 |
   | 21 | 3 | 8 |
   | **24** | **0** | 8 |

   8 lanes do hold all 24 code bits in a single word (`sh0 = 3`). Their 13 state bits are
   entirely in the *other* word. `sh0 ∈ {3, 11, 19, 27}` (8 lanes each) is what produces
   the four splits. **No lane is served by one word.**

### 1.2 Therefore the maximum is 6 codes per 32-bit load — and 32 × 6 < 256

For `C` codes a lane needs `span = 16 + (C-1)·KA` bits resident. One 32-bit word admits

```
16 + 3(C-1) ≤ 32   ⇒   C ≤ 6.33   ⇒   C_max = 6          (span(6) = 31, span(7) = 34)
```

A 16×16 tile is 256 codes and a Metal simdgroup is 32 lanes, so codes/lane = 8 exactly.
**32 × 6 = 192 < 256.** A 1-word-per-lane scheme cannot cover a tile at 3 bpw. Period.

Generalised (`used_lo = 48 − 7·KA`, `gen_exl3_gemv.py:138`) — the second word is dead only
when `used_lo ≥ 32`:

| KA | span(8) | `used_lo` | 2nd word dead? | max codes / 32-bit load |
|---:|---:|---:|:--:|---:|
| 1 | 23 | 41 | **yes** | 17 |
| 2 | 30 | 34 | **yes** | 9 |
| **3** | **37** | **27** | no | **6** |
| 4 | 44 | 20 | no | 5 |
| 6 | 58 | 6 | no | 3 |

**`amp = 1.0` exists exactly at KA ≤ 2.** That is why "just repack it" never works at 3 bpw,
and it is also the only rate at which the programme's stated premise is true.

### 1.3 The 64 requests are invariant under every legal map

Codes/tile = 256 and lanes = 32 force 8 codes/lane; a lane may only take **consecutive**
ring positions (any other set needs ≥ 2 independent 16-bit states = *more* bits, not
fewer). So every legal map costs `32 × ceil(37/32) = 64` word-requests. There is no
permutation to find. Answering the brief's Q3 directly: yes, a repack only permutes *which
lane reads which code* and never the code values — and the freedom is exactly 32 block
rotations, none of which changes the span.

---

## 2. The byte cost — the crux

| layout | words/tile | B/tile | ×96 B | request count | note |
|---|---:|---:|---:|---:|---|
| **today** | 24 | **96** | 1.000 | 64 | the ring; zero redundancy |
| brief's option | 32 | 128 | 1.333 | 64 | **cannot hold 32 × 37 = 1184 bits** |
| 6 codes/lane | 43 | 172 | 1.792 | 64+ | 32 lanes cover only 192 of 256 codes |
| **lossless lane-local repack** | **37** | **148** | **1.542** | 64 | 32 × 37 = 1184 bits |
| same, lane-word-aligned | 64 | 256 | 2.667 | 64 | 8 B/lane stride; 30 GB resident |
| today's 4-bpw neighbours | 32 / 48 | 128 / 192 | — | 64 / 96 | measured 249.0 / 251.4 GB/s |

**32 words cannot hold the payload.** 32 × 32 = 1024 bits < 1184. The brief's "8 bits to
spare" is exactly the deficit: a lane needs 13 bits of anchor, not 8.

### 2.1 And the +54% is fatal before any instruction is saved

Time per tile, in units of "1 byte at the 250 GB/s roof" (= 4 ns), calibrated on the
measured rate sweep (`spec/decode-3bpw-amplification/implementation_notes.md:70-72`):

| rate | measured GB/s | measured time/tile | DRAM roof (bytes/250) | excess |
|---|---:|---:|---:|---:|
| 3 bpw | 199.4 | 0.4814 | 0.384 | **+25%** |
| 4 bpw | 249.0 | 0.5141 | 0.512 | +0.4% |
| 6 bpw | 251.4 | 0.7637 | 0.768 | −0.6% |

`time = max(bytes / BW_roof, instruction_cost)` fits all three points to **0.6%**.
Today's tile costs 0.4814 total. Now the variants:

| layout | DRAM term | instr term | time/tile | predicted tok/s¹ |
|---|---:|---:|---:|---:|
| today, 96 B | 0.384 | 0.481 | **0.481** | **16.05 (measured)** |
| 128 B (4-bpw-equivalent) | 0.512 | ≤0.481 | 0.512 | **15.5** |
| 148 B (feasible min) | 0.592 | ≤0.481 | 0.592 | **13.5** |
| 172 B | 0.688 | ≤0.481 | 0.688 | 11.6 |
| 256 B | 1.024 | ≤0.481 | 1.024 | 7.8 |

¹ 12.039 GB/token scaled by the tile growth, plus the measured 3.1% non-GEMV overhead
(62.3 ms measured vs 60.4 ms of GEMV at 199.4 GB/s).

**The DRAM term at the minimum feasible tile (148 B = 0.592) already exceeds today's
*entire* tile time (0.481) by 23%.** No reduction in instructions can recover that, so the
proposal is dead whether the cost model is request-bound or issue-bound. The 128 B option
is a **measured** −3.5% (it is the 4 bpw point: bytes ×1.333, GB/s ×1.249, time ×1.067 —
`spec/decode-3bpw-amplification/implementation_notes.md:88-91`).

---

## 3. The brief's questions, answered

**Q1 — does the decode still work with 8 codes at bits 0..24 of one word?**
No. The 16-bit window is a *shifting* state; `w_{i+1} = (w_i << 3) | chunk_i`. Eight codes
in bits 0..24 plus 13 bits of anchor = **37 bits**, and the 13 anchor bits are the
neighbouring lane's fresh chunks (visible in the printout: lane 0's `state(13)` =
ring 8..20, lane 1's = 48..60, lane 2's = 32..36 ∪ 88..95). They are not derivable, not
redundant, and not 8 bits. Window setup does not simplify; it gets *worse*, because the
anchor would have to be replicated per lane.

**Q2 — what happens to `sh0` / the funnel?**
`sh0 = (24L + 755) mod 32 ∈ {3, 11, 19, 27}` (8 lanes each); at KA=4 it is the constant
20. It is **never 0** at 3 bpw. A word-aligned layout means `sh0 = 0`, which sets
`rsh0 = 32 − sh0 = 32` — an out-of-range shift, so `extract_v_mcg_32` cannot be
"simplified", it must be *deleted*. Deleting it is legal only when the whole span fits in
`A_hi`, i.e. `used_lo ≥ 32`, i.e. **KA ≤ 2**. So `sh0` collapses to a constant only at a
rate where the codes are 2 bpw.

**Q3 — is the repack expressible in the existing format?**
*Not by the existing repacker.* `reorder_tensor_to_colmajor` (`exl3.rs:733-757`) moves whole
96-byte tiles with `copy_from_slice` and cannot touch a bit inside one; it also hardcodes
48 u16/tile, which `bench.rs:375-383`, `shim_checkpoint.rs:317` and the kernels'
`WORDS = RING/32` all depend on. A new pure function is needed, with a natural home in
`crates/engine/src/bin/repack_model.rs` (the one-time offline tool that calls
`reorder_tensor_to_colmajor` at `:250`).
*A lossless version is possible in principle* and I verified it: store
`(w_0, chunk_1..chunk_7)` = 37 bits per lane, then `w_{i+1} = (w_i << 3) | chunk_i`
reproduces all 8 windows exactly, so all 8 `decode_state_mcg` outputs are bit-identical
(`lane_local_repack_is_bit_exact_and_costs_37_bits_per_lane` asserts it). It costs 148 B/tile
and `37` is not a multiple of 32, so per-lane words are unaligned and the funnel returns —
or you pad each lane to a 64-bit pair and pay 256 B.
*The trellis constrains code values along the ring, not the (lane → code) map.* The only
reader-side freedom is the ring rotation and which of the 32 eight-code blocks each lane
reads (§1.3).

**Q4 — the byte cost.** See §2. **It has to grow the tile, and that kills it.** 148 B is the
minimum for a lossless 1-load layout; 128 B cannot hold the payload; both are slower than
today.

**Q5 — the 2-word `sh_max` boundary.**
The record's boundary is about the **third** word (`two_word_is_safe`: `used_lo ≥ sh_max`,
i.e. `w0c` is dead). Verified: `used_lo = sh_max = 27` at KA=3 and `= 20` at KA=4 — exactly
zero margin at both, as documented. The A1 question is the *second* word, and its condition
is `used_lo ≥ 32`. So a word-aligned layout does not nudge the margin, it **jumps a
discontinuity**: KA ≤ 2 (dead) vs KA ≥ 3 (live). There is no KA in between. The direction
is wrong: the boundary cannot be widened at 3 bpw, only abandoned by changing the rate.

**Q6 — does `linear_gemv_ishizuki_k3_colmajor_contig4` already do this?**
**No.** Read at `linear_repacked.metal:1970-2070`: it computes the *identical*
`lo0 / sh0 / j0 / j1` (`:1992-1997`) and calls the *identical*
`extract_v_mcg_32(t[j0], t[j1], sh0, rsh0, v)` (`:2024`). "contig" refers to the **tile
grouping** — `tk = sg*4`, four contiguous tiles per simdgroup instead of strided by `SG` —
not to the bit layout. It reads 2 words per lane like everything else. (It is also not the
live kernel: `_2col_sk2` wins because `out % 32 == 0`, `linear.rs` dispatch + spec §6.1.)

---

## 4. Corrections to the standing record (important — these mislead the next agent)

`spec/decode-3bpw-amplification/{spec.md:67-68, implementation_notes.md:126-129}` states:

> "union over all 32 lanes of influencing ring positions: **37** — every lane reads the
> *same* 37-bit window, differing only in rotation" … "**no relayout changes that**"

**The conclusion survives; the argument does not.**

* **False:** lanes do *not* read the same window. Lane 0's window is ring bits
  `{8..31, 736..748}`, lane 1's `{0..20, 48..63}`, lane 2's `{32..60, 88..95}` — distinct
  blocks at stride ~24. The **union over all 32 lanes is 768 bits, the whole ring**, not 37
  (`lane_windows_are_distinct_blocks_not_rotations`).
* **True and stronger:** the *per-lane* span is 37 bits and no relayout changes it, because it
  is fixed by the trellis recurrence `w_{i+1} = (w_i << 3) | chunk_i` and not by which lane
  reads what. That is the argument the record should have made.
* **Also false:** "amp is a function of the bit rate alone, `64/(8·KA)`". 3 bpw and 4 bpw
  issue the **same 64 word-requests per tile** (`bench_rate_sweep.rs:117-119`,
  `words_per_lane = 2.0` for rates ≤ 4) yet measure 199.4 vs 249.0 GB/s. amp cannot explain
  that; bytes-per-code can. The formula predicts a 4/3 GB/s ratio at equal amp, and the
  measured 1.249 is the bytes ratio 1.333 with 4 bpw sitting at the DRAM roof.

---

## 5. What *is* binding, and the one number worth keeping

**amp 2.667 is not DRAM amplification and there is nothing to remove there.** The 64 lane
requests touch **exactly the 24 distinct words of the tile** — every word is read, 2 or 3
times, all hits in L1/L2. **DRAM amplification is exactly 1.000.** The tile stream is
already byte-dense and contiguous along a column.

What amp measures is *requests*, and requests are ~25% of the cost above the DRAM roof at
3 bpw (§2.1: 0.481 measured vs 0.384 roof). So requests *are* on the critical path — but
64/tile is the structural floor at 96 B/tile, and every way of getting below it costs bytes
faster than it saves requests, because **one request is worth only 1.88 bytes of DRAM
time** (7.52e-3 units vs 4 ns/B).

**The break-even, worth passing on:** at 96 B/tile the DRAM floor is 0.384 units, so any
layout reaching **≤ 51 requests/tile (1.59 loads/lane)** with the tile unchanged lands at
the 250 GB/s roof = **20.8 tok/s**. That is the only formulation of A1's prize worth
keeping, and it needs **zero** format change. The only mechanism that reaches it is
cooperative loading — 24 lanes each load one distinct word, then redistribute with
`simd_shuffle`. **That has been measured: 180.5 → 146.5 GB/s (−19%),**
`linear_repacked.metal:46-50` records it as removed. The record does not say whether the
*minimal-request* variant (24 loads + 2 indexed shuffles per tile, no threadgroup staging)
was the one tested. I did not test it and cannot; whoever owns the GPU should check that
one specific question rather than re-run the whole idea.

Everything else in the A1 family — repack, re-tile, per-lane payloads — is closed by §1.2
and §2.

---

## 6. Reproduction

```bash
T="${TMPDIR:-/private/var/folders/tp/wz3877_s1z756p1kcsz3_d4h0000gn/T/opencode}"
rustc --edition 2021 --test crates/engine/tests/probe_wordalign_layout.rs -o "$T/pw" && "$T/pw" --nocapture
```

12 tests, 0.01 s, std only, no GPU. `crates/engine/tests/probe_wordalign_layout.rs` is the
only file this investigation created.

**Sabotage evidence (AGENTS.md §3.4).** Two real failures and one deliberate one.

* *Real, and it is why the "union = 37" claim in §4 was never caught by arithmetic:*
  I first transcribed `a`'s bit→ring mapping as `32·j0 + p − shift`, which treats the two
  halves of `a` as running the same way round the ring. It produced
  `union over all 32 lanes = 744 of 768` and `630 distinct code chunks`. The
  opposite-direction halves (§1.1) are the whole mechanism, and a plausible-looking wrong
  transcription hides it. Fixed and re-derived.
* *Deliberate:* changing `span_bits` from `16 + (codes−1)·KA` to `16 + codes·KA` (37 → 40)
  fails **5 tests**, including
  `max_codes_per_32bit_load_is_6_at_3bpw: assertion failed: the kernel cannot do 7, let
  alone 8`. Reverted; 12/12 pass.

**What I did not do, and what is uncertain.** No GPU, no model, no microbenchmark — so the
numbers in §2.1 are *predictions from a model calibrated on three measured points*, not
measurements. The model fits those three points to 0.6% and the conclusion is robust to
both cost models (request-bound and issue-bound) because the DRAM term alone decides it, but
the 15.5 / 13.5 tok/s figures inherit the model's error bars. The 128 B row is not a
prediction: it is the measured 4 bpw configuration reinterpreted.