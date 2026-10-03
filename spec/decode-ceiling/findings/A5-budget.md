# A5 — the decode-step budget, and what is actually left

**Read-only.** No model loaded, no GPU used, no `gpuq`, no binary that opens a
checkpoint. One new file: `crates/engine/tests/probe_budget_reconcile.rs`
(12 tests, clippy-clean, all passing, every claim sabotage-verified).

> Baseline: **62.30 ms/step = 16.05 tok/s**, `--temperature 0`, 6.01-token
> prompt (`AGENTS.md` §10.2.7) + 32 generated ⇒ the KV cache holds **37 keys**.
> Every budget below is anchored there. 512 and 16 384 are **predictions**.

---

## 1. The budget

### 512-token context — **63.601 ms = 15.72 tok/s** (predicted, not measured)

| # | term | bytes | disp | ms | GB/s | source |
|---|---|---:|---:|---:|---:|---|
| 1 | layer weight GEMV (240 dispatches) | 10 982.5 MB | 240 | **56.045** | 196.0 | **residual of the 62.30 ms measurement**, at the 37-key rate; the byte count is context-independent |
| 2 | `lm_head` GEMV | 1 057.0 MB | 1 | **4.178** | 253.0 | §18.1 `KAPLUMBA_ABLATE=head_gemv` = +1.12 tok/s = 4.17 ms; §16.2 GPU clock 4.12 ms. Two instruments, 1.2% apart |
| 3 | attention, 60 dispatches (KV + partials) | 500.6 MB | 60 | **2.521** | 198.6 | T7.6.13 `own_ms`, median of 9 **interleaved** samples |
| 4 | element / transform / finalize | 78.5 MB | 361 | **0.840** | 93.5 | T5.10 production interleaved A/B (`P07:32`) — **not** the 7.8 ms profiler figure |
| 5 | host: `Detokenizer` (GPU idle) | 0 | 0 | **0.017** | — | `tokenizer.rs:202-235`, quadratic: 0.017 → **0.105** ms at 119 tokens |
| | **total** | **12 618.6 MB** | **662** | **63.601** | 198.4 | **15.72 tok/s**, −2.1% vs 16.05 |

### 16 384-token context — **71.8 – 87.5 ms = 11.4 – 13.9 tok/s**

| # | term | bytes | ms | source |
|---|---|---:|---:|---|
| 1 | layer weight GEMV | 10 982.5 MB | **56.045** | unchanged; byte count is context-independent |
| 2 | `lm_head` GEMV | 1 057.0 MB | **4.178** | unchanged |
| 3 | attention, 60 dispatches | 2 257.2 MB | **10.74 – 26.39** | **two treatments, see below** |
| 4 | element / transform / finalize | 78.5 MB | **0.840** | unchanged |
| 5 | `Detokenizer` | 0 | **0.017** | unchanged |
| | **total** | **14 375.3 MB** | **71.82 – 87.47** | **11.43 – 13.92 tok/s** |

**The band is honest, and the low end is the one to plan against.** The shipped
attention path is **not** bandwidth-bound at 16 k. Pricing the same bytes at the
203 GB/s peak gives 10.74 ms; T7.6.13's own full-layer numbers give 26.39 ms. I did
not resolve which is right: the T7.6.13 rows were measured at `splits = 16`, and
`decode_splits` now returns `decode_one_wave_workgroups(256)/kv_heads` =
**54/4 = 13** (`attention.rs:299-306`), a 19% smaller grid. The measured arm is
stale-optimistic; the priced arm assumes a fix nobody made. **Neither is a result.**

### 🔴 The marginal cost of context is not one number, and it is not the step's rate

This is the part I got wrong on the first pass, and the test caught it. I asserted
the 37 → 512 delta was the KV byte delta priced at the step's own 198.4 GB/s. It
is not, and for two separate reasons:

| regime | marginal GB/s | why |
|---|---:|---|
| 37 → 64 keys | **29.2** | `encode_decode_single_pass`, one dispatch per layer |
| 64 → 512 keys | **198.6** | split+combine — a **6.8×** step at the arm boundary (`model.rs:1884-1892`) |
| 50 **sliding** layers, 512 → 1024 | **443.6** | 16 splits × 16 KV heads = 256 workgroups; nearly full occupancy |
| 10 **full** layers, 512 → 8192 | **57.4** | 16 × 4 = 64 workgroups on a 54-workgroup wave |
| 10 **full** layers, 8192 → 16384 | **56.6** | stable to **1.4%** |

**Context is expensive because of 10 layers, not 60.** Past 1024 keys the 50
sliding layers are frozen — the ring wraps, no memmove — so *every* additional
token costs **56.6 GB/s** on 1/6 of the layers, a quarter of the step's own 198.
That is **0.058 ms per 1 000 keys**, and it is stable for the rest of the model's
life.

P15's untraceable "+0.42 ms per 1000" is 7× this. The test asserts they *disagree*
by >0.05 ms/1000, so if they ever agree, this file fails rather than quietly
inheriting a number with no provenance.

### Crossover: attention never reaches the weight stream inside the context cap

`KV(P) = 50·min(P,1024)·16384 + 10·P·8192`. For `P > 1024`:

```
922,746,880 + 81,920·P  =  10,982,522,880   ->   P = 122 800 tokens
```

**122 800, against `MAX_POLICY_CAPACITY` = 32 768.** At the cap the KV read is
3.52 GB = **32.1%** of the weight stream. Attention becomes the majority term only
at ~4× the maximum context this build can hold.

🔴 **P07's "attention equals GEMV weight streaming at P ≈ 13 100" is wrong twice
over**, and both errors make the number more useful-looking than it is: it scales
all 50 sliding layers with context (the ring does not), *and* it uses the retired
11.834 GB weight figure. `11.834e9 / 901,120 = 13 133` — that figure is exactly
the product of the two mistakes.

---

## 2. The non-GEMV overhead: three numbers, one quantity priced two ways

| number | what it actually prices | verdict |
|---|---|---|
| **2.176 ms** = 9.03 µs × 241 | **dispatches**, 241 many-threadgroup GEMVs | real |
| **0.840 ms** | **dispatches**, 361 element kernels, 121 of them single-threadgroup | real, and **disjoint** from the above |
| **0.247 ms** | **bytes**, 47.58 MB at the achieved 192.6 GB/s | real, and **a subset of the 0.840** |

**Are they double-counting? No — and the 0.247 ms resolves the apparent
contradiction.** 0.247 ms is *element traffic minus the f32 `raw`/`raw_b`
accumulator round trip* (30.97 MB) priced at the step's own rate:
`47,581,200 / 192.6 GB/s = 0.2470 ms`. Two errors were in the tree's version:
the round trip is the GEMV's own output returning through DRAM, so counting it as
"element" double-counts the GEMV write the bandwidth term already prices; and
pricing at the 203 GB/s **peak** gives 0.234 ms, because the element kernels do
not run at the peak.

So the 0.840 ms decomposes exactly:

```
0.247 ms  bandwidth (47.58 MB)
0.593 ms  latency of the 121 single-threadgroup dispatches
          (measured 4.6 µs each = 0.557 ms, a 92% fit — the 8% is the 240
           many-threadgroup element dispatches' own latency)
--------
0.840 ms  measured
```

**But the 2.176 ms constant is NOT a third pool — it is the layer GEMV residual
priced a second way, and the two must never be added.**

```
residual  56.045 ms = 54.101 (203 GB/s floor)  + 1.944  un-priced
fit model 59.827 ms = 57.651 (190.5 GB/s)     + 2.176  constant
```

The fit over-predicts by **3.78 ms** because its streaming rate sits 5.5 GB/s
*below* the achieved one. **The honest answer is a range for one pool,
1.94 – 2.18 ms — never 1.94 + 2.18 = 4.12.** Several agents reported these as
incompatible because they were quoting two pricings of one residual.

Which of the four models of the 10.983 GB / 240 dispatches is right:

| model | ms | vs residual |
|---|---:|---:|
| A: 203 GB/s + 9.03 µs × 240 | 56.27 | **+0.22** ✅ |
| B: 190.5 GB/s + 9.03 µs × 240 | 59.82 | +3.77 |
| C: 190.5 GB/s alone | 57.65 | +1.61 |
| D: 203 GB/s alone | 54.10 | −1.94 |

Only A fits. B is the model most likely to be quoted by someone reading the fit
alone, and it is **3.8 ms / 6% slow**.

---

## 3. Addressable vs structurally fixed

Non-weight-streaming time in the measured step: **4.02 ms = 6.5%**.

| term | ms | addressable? | blocker |
|---|---:|---|---|
| per-dispatch constant | **1.94 – 2.18** | **no** | the per-token graph is a **chain of 600+**; T5 multi-queue measured `>1.2×` *worse* than serial (`concurrent_stream_economics.rs:382`). 16 command buffers + 7 waits/token buy ≤0.3 ms and the roll is pointless anyway. |
| attention, 60 dispatches | **1.22** @37 / **2.52** @512 | **no** | 16 and 4 workgroups on a 54-workgroup wave (the 10 full layers have 4 KV heads). A bounded KV context bounds the *work*, not the *parallelism requirement* — T7.6.13 measured single-pass at the 1024-key cap running **7.9× slower**. KV quantisation barred (AGENTS.md §8). |
| element layer | **0.840** | **partly, 0.19** | the 0.593 ms latency is the 121 single-threadgroup reductions. Re-shaping is barred: the 256-lane strided accumulation **is** the reduction order, and `partial[lid] = (lid==0) ? sum : 0` moved **9 266 ULP** on 8 192/8 192 elements. Metal has no grid-wide barrier, so a two-phase reduction needs two dispatches plus a partials round trip. The two-pass widening is separately dead: the exact split needs `G \| 32` and `G = 42`. **One bit-exact win exists: masks 128/64/32 are the only cross-simdgroup steps, so 10 barriers → 3 + 5 `simd_shuffle_xor`, 840 barriers × 0.226 µs = 0.19 ms.** |
| `Detokenizer` | **0.017** @37 tok | **yes** | quadratic replay; grows to 0.105 ms at 119 tokens and is pure GPU idle *even in the pipelined loop* (`generate.rs:347`, between `wait()` and the next submit) |
| host: 1 100 allocs, `format!` labels, `HashMap` by name, `ablated()` mutex | ~0.07 | **yes** | below the noise floor individually; mechanical |
| `ablated()` — **a real defect**: `kernels/src/lib.rs:38` only populates its cache when `KAPLUMBA_ABLATE` is non-empty, so in production the `env::var` runs on all **361 calls/token** under a mutex, 75 ns vs 10 ns | | **yes** | one-line fix |

### The single largest addressable overhead at `--temperature 0`: **0.19 ms**

Blocked by nothing — it is already proved bit-exact (masks 128/64/32 →
`simd_shuffle_xor`), worth **+0.05 tok/s**, and the only reason it is not
shipped is that nobody filed the task. Add the detokenizer (0.017 → 0.105 ms as
the response grows) and the four mechanical host fixes and the total is
**0.28 ms = +0.07 tok/s**.

That is the whole honest pool at temperature 0. The target is 12.3 ms away.

**At `--temperature 1.0` — the shipped default — the answer is 6.70 ms.**
See §4.

---

## 4. `--temperature 1.0`, the default path

`DEFAULT_TEMPERATURE = 1.0` (`crates/cli/src/args.rs:30`). `chat.rs:388` gates the
pipelined loop on `self.greedy`, set only by `new_greedy`, which the CLI calls
only at temperature 0. So **every tok/s figure in this spec is the path the engine
does not run.**

| adder | ms | provenance |
|---|---:|---|
| barrier instead of the 1-deep pipeline | **3.83** | **MEASURED** — chat A/B, 15.88 tok/s pipelined vs 14.97 with the barrier |
| host `Sampler::sample`, 262 144 wide | **2.87** | **MEASURED** — `sampler.rs:41-52` via `examples/bench_sampler.rs` |
| fp16 → f32 widen + 1 MiB `Vec` | **0.45** | **BRACKETED 0.30 – 0.60, never measured** |
| 512 KiB logits readback | **0.15** | **BRACKETED 0.10 – 0.20, never measured** |
| **total** | **7.30** (band 7.10 – 7.50) | ⇒ **69.60 ms = 14.37 tok/s** |

**They are additive and none of them double-counts the element layer.** Three
independent reasons, each asserted in the test rather than argued:

1. **Physical.** The element layer is GPU-*busy* time, measured by switching
   kernels off on a run that never idles. All four adders are GPU-*idle* time on
   the host side of a barrier. A host stall cannot shorten a kernel.
2. **Dispatch sets.** The element partition is device scratch plus device
   sign/norm tables — 0.6% of the weight stream. The sampler reads `logits` on the
   host, the widen builds a host `Vec`, the readback copies `capped` out, the
   barrier idles. The test asserts the partition stays under 1% of the weight
   stream, so a leak would blow up rather than blend in.
3. **Arithmetic.** Subtracting the element term from both sides leaves the
   difference unchanged, to 1e-9.

**But the barrier term and the 9.03 µs constant are the same physical
phenomenon measured twice**, and this is worth stating because it is easy to
double-count by accident:

```
750 dispatches × 9.03 µs          = 6.77 ms of per-dispatch latency per step
measured no-pipeline penalty      = 3.83 ms
=> the 1-deep pipeline already hides 2.94 ms
=> 2.94 ms remains exposed at temperature 0
```

So the exposed per-dispatch latency is **6.77 − 2.94 = 3.83 ms**, and the greedy
path is paying 2.94 ms of it. The `2.176 ms` in §2 is the same pool narrowed to
GEMV dispatches only.

**A device-side sampler + a temperature-agnostic pipeline gate recovers
6.70 ms = +1.53 tok/s at temperature 1.0** (69.60 → 62.90 ms). It returns the
default path to *parity* with greedy, not beyond it. Note that `+2 tok/s` is
claimed in `ROUND1-2-SUMMARY.md`; the budget says **+1.53**, because the 7.30 ms
shortfall is measured from 14.37 and only 6.70 of it is recoverable — the
bracketed widen + readback pair (≤0.60 ms) has never been measured and is not
recovered by a device-side sampler.

---

## 5. Sources rejected

| source | why |
|---|---|
| §18.2's per-projection 126–150 GB/s | deltas sum to 87.59 ms inside a 63.1 ms step; not additive, so not bandwidth |
| "7.32 ms residual, everything ablated" | `ablated("attention")` appears **once**, at `attention.rs:910`, inside `encode_online_attention`. Decode calls `encode_online_attention_split` (`attention.rs:654`) or `encode_decode_single_pass` (`model.rs:1889`) — neither consults it. **Verified by source scan in the test**, so a second guard site fails it. The residual still contained 60 live attention kernels. |
| 546 dispatches | `GpuProfile::stamp` labels, profiler-only. Real: 750. The budget's 662 rows + 88 unattributed epilogue dispatches is inside that. |
| the 1.71× profiler calibration | self-refuting (derived from the head row). |
| 11.834 GB/token | 205 487 488 B (1.71%) short — exactly the 10 full layers' extra q. Flatters the 203 GB/s ceiling by 1.01 ms. |
| P15's tok/s-vs-context table (15.50 / 14.30 / 13.63) | **no provenance.** P15 is still `PROPOSED` (`hypotheses.md:74`) and `test-log.md:24` says "(nothing tested yet)". Not used. |
| `code_rate_vs_bandwidth_discriminator` | its ALU body is a serial dependency chain, so it converts instruction count to latency. Not cited. |

---

## 6. Two things about this budget that a reader should not skip

**A closure check is necessary, not sufficient — and I proved it on myself.**
Swapping `per_token_weight_bytes()` for the retired 11.834 GB inside the budget
leaves **all 12 tests passing**, because the layer term is a residual and simply
re-prices itself at 192.3 GB/s. That is precisely how 11.834 GB survived a dozen
agents. The fix is a direct byte pin (`rows[0].bytes == per_token_weight_bytes() -
head_bytes()`), which then fails with `drifted from bench::per_token_weight_bytes /
left: 10777035392.0 / right: 10982522880.0`.

**A rate band cannot check a byte count, and the arithmetic shows why.** Bytes and
rate trade off exactly when the term is a residual: the retired figure implies
192.3 GB/s against the real 196.0 — a 1.7% difference sitting inside any
plausible 182–197 window. "The rate looks plausible" is *not a check*, and it is
the check that let the bad number survive. Byte counts have to be pinned to
`bench.rs`, which has chain of custody to the checkpoint's own `n_bytes`
(`bench.rs:292-307`).

**Sabotage log** (AGENTS.md §3.4 — each broken, each failure observed):

| sabotage | result |
|---|---|
| 11.834 GB as the layer byte count | **FAILS** — `drifted from bench::per_token_weight_bytes`, left 10 777 035 392 / right 10 982 522 880 |
| element layer 0.84 → 1.52 (the corrupt 9-role figure) | **FAILS** — `the un-priced pool moved to 1.264 ms`, 4 tests |
| sliding window treated as growing (P07's formula) | **FAILS** — `the full-attention share moved`, 2 tests |
| 241 GEMV dispatches → 750 | **FAILS** — 1 test |
| per-dispatch constant 9.03 → 0 | **FAILS** — 3 tests |
| head 253 → 203 GB/s | **FAILS** — 3 tests |
| single-pass ratio ×2 (the 64-key boundary) | **FAILS** — 3 tests |
| a second `ablated("attention")` guard site | **FAILS** — `left: 1, right: 2` |

**One sabotage log entry is itself worth reporting.** My first attempt at the
byte-count sabotage *passed all 12 tests*, because the layer term is a residual
and re-prices itself. My first fix — a plausible-looking rate band of
178–182 GB/s around P05's rescaled figures — **also passed**, because the retired
figure implies 192.3 GB/s which is nowhere near that band. The check was
decoration. Only the direct byte pin discriminates, and it now does.

---

## 7. Bottom line

- **62.30 ms decomposes with no unexplained residual.** 56.045 (240 layer GEMVs,
  196.0 GB/s) + 4.178 (head @ 253.0) + 1.220 (attention, 60 dispatches) + 0.840
  (element, 361 dispatches) + 0.017 (detokenizer) = 62.300. It closes by
  construction, so the budget's value is entirely in its **term values**, which
  are all independently sourced — and in the one term that reconciles the fit.
- **The largest addressable overhead at temperature 0 is 0.19 ms** — the
  bit-exact cross-simdgroup shuffle already proved (masks 128/64/32 →
  `simd_shuffle_xor`, 840 barriers × 0.226 µs). Everything else structural is
  barred by bit-exactness (the 9 266 ULP reduction order, `G ∤ 32` at G=42),
  forbidden (KV quantisation, AGENTS.md §8), or measured counterproductive
  (multi-queue >1.2× worse than serial; single-pass attention 7.9× slower at the
  1024-key cap). With the detokenizer and four mechanical host fixes:
  **0.28 ms, +0.07 tok/s.** The target is 12.3 ms away.
- **The largest addressable overhead on the path the engine actually runs is
  6.70 ms** — device-side sampler plus a temperature-agnostic pipeline gate
  (`chat.rs:388` gates on `greedy`). Worth **+1.53 tok/s at temperature 1.0**,
  taking the default from 14.37 to 15.90 tok/s — parity with greedy, not beyond.
  Note `ROUND1-2-SUMMARY.md` claims `+2 tok/s`; the budget says **+1.53**,
  because 0.60 ms of the 7.30 ms shortfall is the bracketed widen + readback
  pair, which was never measured and is not recovered by a device-side sampler.
- **At 16 k the step is 71.8 – 87.5 ms (11.4 – 13.9 tok/s)** and the range is
  real: the shipped full-attention path runs at **56.6 GB/s**, not 203.
  Resolving it needs one measurement of the 10 full layers at the shipped
  `splits = 13` — ~12 s of GPU, and it settles the only open term here.
- **The 2.176 ms constant, the 0.840 ms element layer and the 0.247 ms floor are
  three views of two disjoint pools**, not three incompatible numbers:
  0.247 ms is bytes and sits *inside* 0.840 ms; 2.176 ms is dispatches on GEMVs
  and is the *same pool as* the 1.944 ms un-priced residual. Report the first as
  a range, never a sum.

**The one thing I would do next, and it is not a kernel:** the open term is a
12-second GPU measurement of 10 attention kernels at the shipped split count.
Everything else in this document is closed.