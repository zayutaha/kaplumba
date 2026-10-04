# What we are supposed to do next

**Spec:** `spec/prefill-24k-90tps/` · **Branch:** `mtp-speculation` · **HEAD:** `b73090d`
**Read this first, then `implementation_notes.md` §10–§14 for the evidence.**

---

## 1. Where the numbers stand

| | before | now | target | verdict |
|---|---|---|---|---|
| prefill @24k | 52.84 | **67.10** tok/s | 90 | ❌ **25% short** |
| prefill @8k | 80.52 | **82.42** tok/s | ≥80 | ✅ |
| prefill @614 | — | **101.5** tok/s | ≥100 | ✅ |
| decode @8k | 12.27 | **12.98** tok/s | ≥12 | ✅ (was passing by 0.0%) |
| decode @24k | 5.44 | **unmeasurable** | 14 | ❌ **blocked, see §3** |
| peak `phys_footprint` @24k | 18.35 GiB | 18.35 GiB | < 20 GiB | ✅ |
| quality | — | **unchanged** (one fp16 ulp) | no loss | ✅ |

**Neither headline target is met.** Prefill is 25% short; decode cannot be
measured at 24k at all. Everything below is aimed at those two.

---

## 2. What has been established (do not re-derive these)

These cost real time to find. Each is committed with its evidence.

1. **§12 — the "86 s attention bandwidth floor" was wrong by ~350×.** It counted
   2048 B of *DRAM* per (query,key) pair; `PREFILL_CHUNK = 1024` means a whole
   chunk shares every key. Real 24k KV traffic is **67.1 GB ≈ 0.25 s**. The
   kernel moves **0.45 GB/s ≈ 0.16% of DRAM bandwidth**.
   **AC2 is an attention ask (149.7 s → ≤ 56 s, ~1.74 → ~3.4 TFLOP/s = 22% of
   tensor peak), not a GEMM ask.** OQ4 and OQ6 withdrawn. Do not re-open the GEMM.
2. **§10 — the T4e cost split does not reproduce.** `ksteps = 0`, QK-off,
   both-off and an immediate `return` all land within 1.8% of baseline, while an
   added spin loop costs +3.9 s. §9.1/§9.1b/§9.2 are **withdrawn as a design
   basis**. The microbench is a throughput instrument, **not** an attribution or
   ranking instrument (four independent confirmations it ranks arms wrongly).
3. **§9.1b's "BQ=2 + key-split + combine" is re-opened, not refuted** — its
   refutation used the withdrawn §9 split. But naive BQ=2 still measures
   −2.0% @8k / −6.2% @24k, so the *grid* has to be restored too.
4. **§14 — oMLX v0.7.0 independently reached our §6 decode diagnosis** (per-key
   `simd_sum` decode is ALU-bound at ~⅓ of DRAM). Ported; see §4.

---

## 3. The blocker: 24k decode cannot be measured (OQ8)

The 24k run's own footprint is **18.35 GiB of a 24 GiB machine**. It evicts the
desktop, and every later allocation becomes a compression event:

```
PhysMem: 23G used (9165M wired, 5954M compressor), 110M unused
decode_tok_s = 0.055     <- 99x pessimisation, this is the compressor
```

**Three attempts have produced invalid 24k decode samples.** A measurement gate
cannot fix this — it can refuse to *report* a compressed sample, not produce a
valid one.

> **Decision needed from the user.** Either (a) measure on a quieter machine
> (desktop closed / second box), (b) find a lower-footprint 24k configuration to
> measure, or (c) re-baseline AC1 onto 8k, where we *can* measure and where the
> kernel is verified. This is the one thing blocking AC1 that is not a kernel
> problem.

---

## 4. Shipped: T2b, the MMA decode kernel (`55cd5df`)

`attention_decode_flash_512_8` — the prefill flash body with `BQ=1` and
`PARTIALS`, so decode QK/PV are 8×8 simdgroup MMA instead of a 32-lane
`simd_sum` per (key, head). Writes the same `(m, s, O)` partials, so the existing
combine pass and its exactness argument are reused unchanged. Ported from
`omlx/patches/mimo_v2/sdpa_flash.py` (oMLX v0.7.0, commit `07d88dbd`).

**Measured, interleaved A/B, 3 rounds @8k:** 12.000/12.059/11.990 →
**12.978/12.979/12.972** = **+8.2%**, effect arm ±0.03% vs control ±0.3%.
**@4k: +4.4%.** Gain grows with context, as oMLX measured (+2.8% @8k → +71.6%
@256k), so 12.98 @8k is a **floor** for 24k, not an estimate.

**Gates:** `KAPLUMBA_DECODE_FLASH=0` off · `KAPLUMBA_DECODE_FLASH_MIN_KEYS=<n>`
threshold (default 4096) · head_dim 512 with 8 q-heads/kv-head only, enforced in
`encode_decode_split_explicit`.

**Quality:** new `attention_decode_flash_matches_split`, 15 cases, worst
|flash − scalar| = **0.000488 (one ulp)**, tol 5e-3, no NaNs, across ragged
splits, off-the-end slices and empty splits. Sabotage check: removing `split`
from the partials index fails at **0.5576477 (111× tol)** and only on `splits≥2`.

**Not taken from oMLX, deliberately:** all NAX/M5 code (wrong GPU), all MTP (per
brief), TurboQuant KV (forbidden by AGENTS.md §8), and the MoE-specific fused
decode layer stack (wrong model shape — Gxma is dense). The last is a lead, not
a dead end: oMLX reports "a decode step dispatched ~2.8k kernels, most of them
small element-wise ops", which may apply to our element-kernel count too.

---

## 5. Next, in priority order

### N1 — Prefill: slice the key range into successive dispatches (OQ9)
**The strongest remaining prefill idea, and it is a direct port.**

oMLX's prefill path runs several dispatches over consecutive key ranges
(`OMLX_NAX_ATTN_PASS_KEYS`, default 8192), handing the fp32 online-softmax row
state (O accumulator, running max and sum) from one to the next through device
memory — **bit-identical output**, no combine pass — because *"one dispatch over
a long key range lets its threadgroups drift apart until each streams its KV head
from DRAM; per-slice dispatches keep the K/V they share in the on-chip caches."*

- **Why here:** it targets the same cache-residency loss §10 measured (a 64×
  smaller K/V buffer bought only 6%, so the kernel is not DRAM-bound — but its
  *access pattern* is what T4f improved for +6.7% E2E).
- **Shape:** not §12.2's parallel key-split + combine (which needs a partials
  buffer and a combine pass). Sequential slices carrying row state.
- **Constraints:** scratch must stay **fixed-size and context-independent**
  (AGENTS.md §8). Row state = 1024 queries × 32 heads × 512 dims × 4 B = 67 MB,
  fixed per chunk — affordable inside the 20 GiB budget (currently 18.35).
- **Test:** bit-exact vs the current one-dispatch body (§4's standard), then E2E
  at 16k and 24k. Slice size becomes a sweep knob (8192 is oMLX's default).
- **Risk:** medium. The O accumulator round-trip through device memory is
  67 MB written + read per slice, which may cost more than the cache residency
  it buys. **Measure at 16k first, where a ~6% effect is resolvable** — 24k
  carries ±20% (OQ5).

### N2 — Unblock 24k measurement
Until OQ8 is decided, **no 24k decode or prefill number from this box is
trustworthy**, and N1 cannot be validated at the context that matters. Everything
else is downstream of this. See §3.

### N3 — Apply oMLX's cheap micro-wins to the decode kernel
From `sdpa_flash.py`, not yet tried here:
- **"A row whose max did not move has factor 1: skipping its rescale is exact"** —
  guard the O rescale on `m_new != m_run`. Free, and exact.
- **Head-dim split over simdgroup pairs** (their `HS == 2` scheme): we currently
  spend all 8 simdgroups on 8 query heads; splitting the head dim across pairs
  trades simdgroups for a shorter dependent chain.
- Head dim is **512** here vs 192/128 there, so both are more relevant to us.

### N4 — Re-examine `decode_splits` now that the kernel changed
`§13`'s sweep never produced a usable curve (all arms invalid). With the MMA
kernel the parallelism trade is completely different: the scalar kernel was
latency-bound on a 64-workgroup grid, the MMA one may want **more** splits, and
`ATTENTION_SPLITS_MAX = 16` is a hard cap that has never been probed above.
oMLX makes the split count a **runtime** value deliberately (a template split
count cost them ~115 ms of JIT/step and −22% decode @8k) — ours already is.

### N5 — Fix the red OQ5 gate in the sweep harness
`/tmp/test_parse.sh` is **written and failing** (`$1: unbound variable` under
`set -u`); it caught two real parser bugs already. N1/N4 both need a trustworthy
sweep harness. See `implementation_notes.md` §13.2.

### N6 — T6 / final scorecard
Only after N1–N4. Interleaved A/B at 8k/16k/24k, peak `phys_footprint` via
`/usr/bin/time -l`, five reference prompts, full suite (4 failures are
**pre-existing at base** — do not attribute them to this work).

---

## 6. Ground rules for whoever continues

- **Interleave A/B** (AGENTS.md §10.2 #5). Single-round pairs carry **3.7%**
  spread at 4k and up to ±20% at 24k — measured, §14.5.
- **Never pipe a long measurement run into `tail`.** It buffers until EOF, so an
  interrupted run loses *everything*. Append to a file per arm. This cost a
  ~50-minute sweep.
- **A reading of "the change did nothing" is only believable after a positive
  control has moved the same instrument in the same build.**
- **Sabotage every new claim.** A test-coverage claim requires breaking the code
  and seeing it fail.
- **Check `top`'s `unused` and the process's CPU time before trusting a 24k
  sample.** >2% CPU in compression means the sample is invalid — discard it, do
  not report it.
- **One model process at a time**, via `scripts/gpuq try --` (exit 75 = busy).