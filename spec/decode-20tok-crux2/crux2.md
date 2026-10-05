# crux2 — the adversarial re-check of "20 tok/s at 16k is impossible"

**Verdict: the proof holds.** Nothing in it broke, and I am not able to make it
break. Two of its numbers I reproduce *exactly*; one of them I reproduce as
**lower** than claimed, which strengthens rather than weakens the conclusion.
The two axes the brief flagged as untested — the entropy of the address stream,
and whether a greedy-identical lower-bpw model exists — are both closed, the
first by measurement, the second by proof plus the absence of the artifact.

The load-bearing claim turns out not to be a measurement at all. It is
`284.8 GB/s required` against a `273 GB/s` pin rate on a part whose GPU core
count `system_profiler` reports as 20. No kernel can exceed the connectors.

| | prior (`crux.md`) | this re-check | agree? |
|---|---:|---:|:--:|
| bytes/token, weights | 12.0580 GB | **12.0580 GB** (12,058,001,636 B, 3 sources, delta **0**) | exact |
| KV at 16,384 | 2.1810 GB | **2.1810 GB** | exact |
| bytes/token at 16k | 14.2390 GB | **14.2390 GB** | exact |
| best single bandwidth reading | 253.6 GB/s | **246.2 GB/s** | **no — mine is 3.0% lower** |
| best *median* bandwidth arm | 245 GB/s sustained | **240.3 GB/s** | mine 1.9% lower |
| 20 tok/s @16k needs | 284.8 GB/s = 104.3% of wire | **284.8 GB/s = 104.3% of wire** | exact |

---

## 0. Method, and the box discipline this required

Worktree `/private/var/folders/tp/.../opencode/wt-crux2`, branch
`adversary/decode-20tok-crux2`, own `CARGO_TARGET_DIR`. Nothing in the main tree
was touched.

Before **every** GPU run: `scripts/gpuq status` (reported `gpu: none`,
`13.0 GiB available`, `0 MiB swap` on every run) and
`ps -Ao pid,rss,args | grep kaplumba` (empty on every run). No sample in this
document is from a contended box. The six bandwidth trials are **interleaved**
(gb8, gb5, gb8, gb5, …) so drift cannot be attributed to an arm, and within each
trial the sweep is **round-major** (every arm measured in every round).

Raw output is committed beside this file: `crux2-bytes-raw.txt`,
`crux2-dump-raw.txt`, `crux2-head-bound-raw.txt`, and
`crux2-bw-raw-gb{5,8}-trial{1,2,3}.txt`.

---

## 1. bytes/token — independently reproduced, to the byte

The prior measured this through `kaplumba_engine::safetensors`. I did not reuse
that module: `crux2_bytes.rs` reads each shard's header **raw** (`u64` LE length
prefix, then JSON) and takes geometry from `config.json` parsed in the example,
so a bug in the selection layer cannot travel into my total and the prior's at
the same time.

### 1.1 Three controls, and they agree to zero

| id | source | value | vs declared |
|---|---|---:|---:|
| C1 | `sum(data_end − data_start)` over 2,422 tensors, both shards | 16,028,050,244 | **delta 0** |
| C2 | `sum(file_len) − sum(header_len)` | 16,028,050,244 | **delta 0** |
| C3 | `sum(n_bytes)` over `quantization_config.json`'s own `tensor_storage` | 14,876,563,052 (2,006 tensors) | = quantized + full embed table |

`declared_total_size` in `model.safetensors.index.json` is 16,028,050,244. C1 and
C2 are *actual file sizes* and *declared byte ranges* hitting the same integer
with no slack, and the sorted offsets tile both shards with **0 gaps and 0
overlaps**. C3 is the quantizer's own bookkeeping and shares no code with the
header parser.

That is the load-bearing number verified. **`12,058,001,636 B` of weights,
`12.0580` GB, reproduced exactly.** KV at 16k recomputes to `2,181,038,080 B`
= `2.1810 GB`. `14.2390 GB/token`.

### 1.2 Selection identity, and a correction to the prior's stated reason

```
weights − embed_row + full embed table + vision  =  16,028,050,244
declared total_size                              =  16,028,050,244
unaccounted remainder                             =  0 B
```

A *per-family* selection control passes with delta 0 on all eight families.

**Correction.** `crux_bytes.rs:640` states: *"This checkpoint ships **zero**
vision tensors (the `vision` bucket is 0)."* **That is false for this
checkpoint.** It contains a **1,151,487,072 B (1.1515 GB) vision tower** — 27
fp encoder layers plus `patch_embedder` and `embed_vision` — and a
**2,818,572,288 B bf16 `embed_tokens` table** of which decode reads 10,752 B (one
row). The *byte totals are unaffected* (the classification is prefix-anchored and
correctly routes both to their own buckets, and the selection identity closes to
0), so no figure in `crux.md` is wrong. But the stated justification is wrong,
and the same file documents that this exact prefix-anchoring is a silent
misclassification hazard for a future checkpoint.

I hit that hazard myself, in my own cross-check: a `contains("norm")` dtype
filter swept in the vision tower's four layernorms per layer and reported
145,982 bf16 params where the correct answer is 133,516 — a +128,304 residue
(27 × 4,752) that has nothing to do with the checkpoint's text model. Caught by
recomputing the family table by hand: 2,627,072 + 120 + 10,752 = **2,663,032 B**,
which is the `norms+scalar` bucket to the byte.

### 1.3 The per-projection identity — reproduced, and it really is 3.000

| matrix | trellis B | count | params (config) | bpw | count expected |
|---|---:|---:|---:|---:|---:|
| q_proj | 1,156,055,040 | 60 | 3,082,813,440 | **3.0000** | 60 |
| k_proj | 454,164,480 | 60 | 1,211,105,280 | **3.0000** | 60 |
| v_proj | 412,876,800 | **50** | 1,101,004,800 | **3.0000** | **50** |
| o_proj | 1,156,055,040 | 60 | 3,082,813,440 | **3.0000** | 60 |
| gate/up/down | 2,601,123,840 ea | 60 ea | 6,936,330,240 ea | **3.0000** | 60 ea |
| lm_head | 1,056,964,608 | 1 | 1,409,286,144 | **6.0000** | 1 |

Every family is *exactly* 3.000000 bpw and the head *exactly* 6.000000, with
matrix counts matching `config.json` geometry including `v_proj` in 50 rather
than 60. A full dtype-vs-`shape×width` audit over all 2,422 tensors reports
**0 mismatches**. So the "zero slack to reclaim" claim is real: this is not a
packing artefact that a different layout recovers, it is the rate.

---

## 2. Bandwidth — independently measured, and *lower* than claimed

### 2.1 The instrument, and its controls

`crux2_bw.rs` + `crux2_bw.metal`, built from scratch. It attacks the one axis no
prior probe has run: **visit order**. Every arm in the prior work reads memory
front-to-back, because that is what a grid-stride loop does. The real EXL3 decode
GEMV walks 32×32 trellis tiles, so the grid touches a small window and jumps.

Two families, both with provably exact coverage:

- **BURST(B)** — thread *i* reads element `i + k·per + j·per·B`, `k ∈ [0,B)`.
  `B = 1` is exactly the prior probe's `stream_u4`. `B = 24` is 24 uint4 =
  **384 B = exactly one 32×32 tile's payload at 3 bpw** (32·32·3 bits). So one
  parameter sweeps raster-scan → tile-at-a-time.
- **STRIDE(odd)** — block visit order `(j·odd) mod iters`, a bijection proven by
  exhaustive host-side multiset check. Five multipliers = five very different
  strides.

Guards, each of which caught a real defect during development (§6):

- `iters·B·per == count` **and** an exhaustive proof that the visited multiset is
  exactly `[0, count)` — no repeats, no gaps.
- `gcd(odd, iters) == 1`, and the multiplier is reduced mod `iters` **first**
  (see §6.5).
- **CPU-computed reference checksum**, from the fill rule, prefix-summed over the
  buffers the arm declares.
- `CONTROL_repeat` — byte-, thread-, buffer- and order-identical to the reference.
  Its difference **is** the noise floor.
- `CTRL_modidentity` — runs the stride arms' *identical* multiply+modulo with
  `odd = 1`, i.e. identity order. Comparing a stride arm against **this** rather
  than against a no-modulo arm is what makes it an A/B.
- `--sabotage` flips **one word of the last buffer only** and asserts an exact
  per-arm prediction: 1-buffer arms must be *unchanged*, N-buffer arms must move
  by exactly `0xdeadbeef`.
- `--sabotage-fill` writes the historical tiled fill and must be rejected.

Both controls behave as required and are part of the committed raw output.

### 2.2 Results — 6 trials, 9 measured rounds each, interleaved

Box quiet throughout (13.0 GiB available, no `kaplumba` process, before each).

**8 GB working set (94.37% of request usable), medians of 3 trial medians:**

| arm | t1 | t2 | t3 | **median** | spread | %wire | best single |
|---|---:|---:|---:|---:|---:|---:|---:|
| `raster_burst1_buf1` (≡ prior `stream_u4`) | 231.5 | 231.6 | 233.8 | **231.6** | 7.0% | 84.8% | 238.5 |
| `raster_burst6_buf1` | 233.8 | 235.3 | 233.6 | 233.8 | 6.1% | 85.6% | 241.5 |
| `raster_burst12_buf1` | 236.9 | 237.8 | 233.5 | 236.9 | 10.7% | 86.8% | 242.5 |
| `raster_burst24_buf1` (384 B = one tile) | 238.0 | 238.8 | 239.1 | **238.8** | 8.4% | 87.5% | 242.9 |
| `raster_burst48_buf1` | 237.6 | 234.7 | 238.0 | 237.6 | 8.9% | 87.0% | 243.3 |
| `raster_burst1_buf8` | 239.1 | 238.6 | 238.7 | 238.7 | 5.4% | 87.4% | 245.7 |
| **`raster_burst24_buf8`** | 240.8 | 240.3 | 238.4 | **240.3** | 4.1% | 88.0% | 243.4 |
| `CTRL_modidentity_b24` | 232.6 | 238.1 | 236.0 | **236.0** | 7.9% | 86.4% | 241.9 |
| `stride0_b24_buf1` | 237.3 | 232.9 | 232.4 | 232.9 | 8.2% | 85.3% | 242.4 |
| `stride1_b24_buf1` | 230.2 | 234.0 | 235.4 | 234.0 | 10.8% | 85.7% | 242.8 |
| `stride2_b24_buf1` | 235.0 | 236.5 | 239.1 | 236.5 | 6.2% | 86.6% | 242.1 |
| `stride3_b24_buf1` | 238.8 | 235.7 | 238.5 | 238.5 | 9.6% | 87.4% | 243.2 |
| `stride4_b24_buf1` | 235.4 | 236.5 | 228.7 | 235.4 | 8.1% | 86.2% | 241.8 |
| `stride0_b24_buf8` | 238.8 | 239.5 | 240.6 | 239.5 | 4.6% | 87.7% | **246.2** |
| `stride0_b1_buf1` | 235.2 | 234.5 | 231.3 | 234.5 | 12.1% | 85.9% | 240.3 |
| `CONTROL_repeat` | 227.8 | 232.8 | 231.3 | 231.3 | 9.7% | 84.7% | 238.8 |

Noise floors (`CONTROL_repeat`): **−1.60%, +0.55%, −1.06%** (median |noise| ≈ 1%).
**5 GB working set:** best median arm `raster_burst24_buf8` = **239.8 GB/s**;
best single 246.0; noise floors −0.89%, +1.30%, +0.47%. Consistent with 8 GB.

### 2.3 What this says

1. **The entropy axis is dead.** Against the permutation-cost control
   (`CTRL_modidentity_b24` = 236.0), the five stride arms are 232.9 / 234.0 /
   236.5 / 238.5 / 235.4 — i.e. **−1.3% to +1.1%, straddling the ~1% noise
   floor with no trend**. Against the pure raster arm at the same burst
   (`raster_burst24_buf1` = 238.8) *every* stride arm is at or below. **No
   address-stream entropy beats a raster scan.** The brief's hypothesis that
   higher-entropy streams might improve DRAM row-buffer utilisation is not
   supported: it does nothing here.
2. **Burst length does nothing either**, and neither does the tile-shaped value
   in particular: 231.6 → 233.8 → 236.9 → 238.8 → 237.6 across B = 1…48 is
   monotone-ish and 3% wide, i.e. at the edge of noise, with no preference for
   B = 24.
3. **Independent allocations give ~+3%, not more.** 231.6 → 238.7 at B = 1
   (+3.1%); 238.8 → 240.3 at B = 24 (+0.6%, inside noise). This confirms the
   prior's §2.4 qualitatively and *quantitatively deflates* it.
4. **I do not reproduce 253.6.** My best single reading over 6 trials × 16 arms is
   **246.2 GB/s (90.2% of wire)**; my best median arm is **240.3 GB/s**. The
   prior's 253.6 is 3.0% above anything I saw. Since my noise floor is ~1% and
   per-arm spreads run to 13%, a single 253.6 reading on this box is not
   distinguishable from 246 — consistent with the prior's own §2.7 warning and
   with `AGENTS.md` §10.2 #5. **Every number below uses 246.2 (my best reading)
   and 240.3 (my best median); both are lower than the prior's, so the verdict
   is conservative.**

---

## 3. Every route to fewer bytes/token, with numbers

### 3.1 What 20 tok/s demands, and therefore how much must be cut

20 tok/s = 50 ms/token. Budget = `BW × 0.050`.

| context | bytes/token | 20 tok/s needs | % of 273 wire | vs my best 246.2 |
|---|---:|---:|---:|---:|
| 8,192 | 13.5680 GB | **271.4 GB/s** | **99.4%** | 110.2% |
| 16,384 | 14.2390 GB | **284.8 GB/s** | **104.3%** | 115.7% |

Shortfall to be closed, as a byte reduction:

| achieved | @8k | @16k |
|---|---:|---:|
| 240.3 (my best median) | 1.553 GB (11.5%) | 2.224 GB (15.6%) |
| 246.2 (my best reading) | 1.258 GB (9.3%) | 1.929 GB (13.6%) |
| 245 (prior's planning figure) | 1.318 GB (9.7%) | 1.989 GB (14.0%) |
| 253.6 (prior's best arm) | 0.888 GB (6.5%) | 1.559 GB (10.9%) |
| 273 (100% of wire, unattainable) | **0 — fits, by 0.60%** | 0.589 GB (4.1%) |

So the only question is whether bytes/token can fall by **1.26–2.22 GB** at any
bandwidth this box has ever produced. Four routes exist. All four close.

### 3.2 Route 1 — read less of `lm_head` (the only one with real promise)

`lm_head` is 1.056964608 GB, 8.77% of the step, and is read in full every token to
produce 262,144 logits. The brief demands a two-stage head be *proven* greedy-
identical or rejected. **It is rejected, twice: once on exactness and once on
arithmetic.**

**Exactness.** EXL3 stores `W[t,j] = out_scale[t] · in_scale[j] · q[t,j]`, so a
partial read leaving column set `U` unread admits an **exact** (triangle-inequality)
exclusion bound, not an approximation:

```
|logit_t − partial_t|  ≤  |svh[t]| · C · Σ_{j∈U} |h_j| · |suh[j]| ,   C = max|q|
```

Measured from the real checkpoint and the repo's own codebook decoder
(`exl3::decode_state`, mcg, all 256 states):

| quantity | value |
|---|---:|
| `max|q|` | 3.625000 |
| `RMS(q)` | 1.203422 |
| `max|q| / RMS(q)` | 3.0122 |
| `lm_head.svh` (per row, 262,144) absmax / RMS | 6.082031 / 1.002788 |
| `lm_head.suh` (per col, 5,376) absmax / RMS / Σabs | 0.172119 / 0.015341 / **81.359** |

`config.json` pins `final_logit_softcapping: 30.0`, so **|logit_t| ≤ 30 for every
token in the vocabulary** — a hard, config-guaranteed cap, not an estimate. With
post-RMSNorm `E|h_j| = 0.7979`:

```
L1_all = Σ_j |h_j||suh[j]| ≈ 0.7979 × 81.359            = 64.916
full-head bound  = |svh| · C · L1_all = 0.8506·3.625·64.916 = 200.2
                  = 6.7× the softcap of 30
```

The bound is **linear in the unread L1 mass**, so a necessary condition to
exclude even a median-scale row is `Σ_{j∈U}|h_j||suh_j| < 9.730`, i.e.

> **at most 14.99% of the head's columns may be left unread** (2.10% for the
> largest row scale), so the maximum any exact two-stage head could ever save is
> **0.1584 GB of 1.0570 GB** — and that already assumes the bound only has to
> beat the softcap rather than the actual runner-up logit, which it does not.

**Arithmetic.** Even the *fantasy* versions do not reach the target:

| variant | bytes/token @16k | needs | % of wire |
|---|---:|---:|---:|
| actual | 14.2390 | 284.8 GB/s | 104.3% |
| saving at the rigorous maximum above (0.1584 GB) | 14.0806 | 281.6 GB/s | **103.2%** |
| head **entirely free** — unreachable upper bound | 13.1820 | 263.6 GB/s | **96.6%** |

A free `lm_head` would need 263.6 GB/s. The best rate this box has ever produced,
by my instrument or the prior's, is 253.6, and my own best is 246.2 = 90.2% of
wire. **The head route is dead on both counts, and it is the single largest
non-weight lever available.**

### 3.3 Route 2 — a greedy-identical lower-bpw re-quantization

This is the route the brief calls most valuable. It is **not testable on this
box**, and the arithmetic says it would not work even if it were.

**It cannot be built here.** Every EXL3 checkpoint on this machine, enumerated:

| dir | declared `bits` / `head_bits` | Σ `n_bytes` | note |
|---|---|---:|---|
| `~/.kaplumba/models/Gxma/Gxma` | 3.0 / 6 | 14,876,563,052 | the model in question |
| `~/.kaplumba/models/Gxma-repacked` | 3.0 / 6 | 14,876,563,052 | **byte-identical total to `Gxma`** |
| `~/.kaplumba/models/Gxma-4-bit` | 4.0 / 8 | 18,889,725,548 | larger |
| `~/.kaplumba/models/Gxma-bpw4-shim` | 4 / 8 | 16,068,490,348 | symlinks into `~/Downloads` |
| `~/.Trash/` | 4.0 / 8 | 18,889,725,548 | trashed |

**There is no checkpoint below 3.0 bpw on this machine.** `Gxma-repacked` has
*exactly* the same total as `Gxma`, so the repacking campaign recovered **zero**
bytes — an independent confirmation of the prior's dead end #1. Producing a 2.5
bpw model requires the EXL3 quantizer, which is Python (`exllamav3`), and
`AGENTS.md` §8 forbids Python in this project. I did not attempt it.

**The arithmetic would not deliver anyway** (head held at 6 bpw, KV unchanged):

| achieved | budget | for layers | **layers bpw** | vs 3.000 |
|---|---:|---:|---:|---:|
| 239.8 (real GEMV) | 11.990 GB | 8.752 GB | **2.3907** | −20.3% |
| 240.3 (my best median) | 12.015 GB | 8.777 GB | **2.3981** | −20.1% |
| 245 (prior's planning figure) | 12.250 GB | 9.012 GB | **2.4617** | −17.9% |
| 246.2 (my best reading) | 12.310 GB | 9.072 GB | **2.4780** | −17.4% |
| 253.6 (prior's best arm) | 12.680 GB | 9.442 GB | **2.5792** | −14.0% |
| 273 (100% of wire, unattainable) | 13.650 GB | 10.412 GB | **2.8442** | −5.2% |

Every row at any bandwidth this box has is short. Even at 100% of theoretical
peak the layers must drop 5.2%.

**And the model is not plausibly quality-neutral there.** The checkpoint is
`gemma-4-31b-it-qat-assistant-4bit` — **quantisation-aware-trained at 4 bit**.
EXL3 3.0 bpw is already 25% *below* its QAT target; 2.46 bpw is 38% below. QAT
models degrade fastest exactly where they were trained, and the user's constraint
forbids lossy requantisation.

**There is no quality data to appeal to in either direction.** `spec/logit-distribution-probe`
exists precisely to "unblock the 3.0 bpw → 2.0 bpw decision", and it was never
built: all 8 tasks unchecked, no `crates/engine/src/logit_probe.rs`, no example
binary. `spec/shim-exl3-4bpw` is about kernel *coverage* only and reports no
quality number. **The prior's §4.2 bpw table is arithmetic on a model that does
not exist and was never measured** — which is the correct thing for it to be,
but it also means the "is 3.0 bpw quality-limiting?" question is unanswered in
this repository, not answered in the negative.

### 3.4 Route 3 — the 2.1810 GB of KV

The brief asks whether all of that traffic is necessary, and points at the 10
full-attention layers. **There is a real loophole here and it is closed — but
only by reading the kernel, not by reasoning.**

`attention_k_eq_v: true`, and `v_proj` exists in exactly **50** matrices, absent
from the 10 full layers. So on a full-attention layer the raw K projection is
*reused as V* (`model.rs:1656-1659`, `encode_copy(scratch.k, scratch.v)`). If
cached V were merely a duplicate of cached K, **half the full-layer KV —
0.671 GB, 4.7% of the step — would be free**, losslessly.

It is not a duplicate. `norm_rope_qkv_cache`
(`crates/kernels/src/elements.metal:1219`) applies, to the *same* raw-K buffer:

- for **K**: a **weighted** RMS-norm, rounded to fp16, then **RoPE**;
- for **V**: `rms_norm_unweighted_tg` — **no weight, no RoPE** (lines 1283-1289).

Different weights and a different rotation, so **every one of the 512 dims
differs** and both caches are required. (`partial_rotary_factor: 0.25` on full
attention changes which dims are rotated, not how many are stored.)

Beyond that: `AGENTS.md` §8 forbids KV quantisation; and for completeness, even
*deleting the entire KV cache* leaves 20 tok/s @16k needing **241.2 GB/s =
98.0%** of my best measured 246.2 — a 2% margin, which buys nothing against a
real 3 bpw GEMV, and is forbidden besides.

**No text tensor is conditionally read.** `crux2_dump.rs` prints all 2,422
tensors with dtype, shape and bytes; the largest text tensors after the
quantised matrices are the norms at 10,752 B each, `lm_head.svh` at 524,288 B,
and the 2.8186 GB `embed_tokens` of which decode reads 10,752 B. Nothing large is
prefill-only.

### 3.5 Route 4 — bandwidth

Covered in §2. The untested entropy axis is now tested and dead; multibuffer is
confirmed at ~+3%; my best single reading is 246.2 GB/s = 90.2% of the 273 GB/s
pin rate. `273 GB/s` is Apple's published rate for the 20-GPU-core M4 Pro, and
`system_profiler` confirms this box is `Mac16,7`, `Apple M4 Pro`, **20 GPU
cores**, Metal 4. **It is a hardware constant, not a measurement.**

---

## 4. Verdict

**The proof holds. 20 tok/s of pure decode is not reachable on this box at 16k,
and I could not break it.**

The reason is worth stating plainly, because it is stronger than the prior states
it: **the impossibility does not depend on any measurement in this repository.**

```
bytes/token @16k      = 14.2390 GB     (verified to 0 bytes, 3 independent sources)
20 tok/s              = 50.000 ms/token
required              = 284.78 GB/s
wire peak, M4 Pro 20-core = 273 GB/s   (Apple's published pin rate; system_profiler confirms 20 GPU cores)

284.78 / 273 = 104.3%
```

A kernel cannot outrun the connectors. Even at **100% of theoretical peak with
zero overhead** — no norms, no RoPE, no attention, no sampling, no dispatch gaps
— 16k yields **19.17 tok/s**. That row is a bound, not a plan, and it is already
below the target.

**True maximum pure-decode tok/s** (zero overhead, which the real step does not
have — today it runs at 12.3):

| achieved BW | @8,192 | **@16,384** | source |
|---|---:|---:|---|
| 240.3 GB/s | 17.71 | **16.88** | my best median arm |
| 246.2 GB/s | **18.15** | **17.29** | my best single reading, 6 trials × 16 arms |
| 245 GB/s | 18.06 | 17.21 | prior's planning figure |
| 253.6 GB/s | 18.68 | 17.81 | prior's best arm (not reproduced) |
| 273 GB/s | 20.12 | 19.17 | 100% of wire, unattainable |

**So: ~18.2 tok/s at 8k, ~17.3 tok/s at 16k**, against a target of 20.

### 4.1 One correction to the prior's framing

The prior's verdict table quotes only the 16k row, where the target misses by
**14.0%**. At the 8,192-token context — which is the context length where a
30–60 turn conversation actually lives, and which the engine has run for months —
the same arithmetic misses by only **9.7%** at 245 GB/s, and by **0.60%** against
100% of wire peak. **8k is a materially closer miss than 16k and the prior's
document does not say so.** That does not change the verdict; it changes where
the remaining engineering effort is worth spending, because at 8k the binding
constraint is overhead (the step runs at 73% of its own GEMVs' 239.8 GB/s) while
at 16k it is the connector.

Incidentally, `AGENTS.md` §10.2 #7's claim that a prompt over 8192 tokens is
rejected is **stale**: `DEFAULT_CONTEXT_LIMIT = 32768`
(`crates/engine/src/generate.rs:35`). 16k is inside the implemented limit, so
this is not a "you can't even get there" argument.

### 4.2 What would actually change the answer

Nothing on the quality-preserving menu. The three levers, in order of size:

1. **A bpw below 2.46 that is greedy-identical.** Would need a new quantization
   run (Python, forbidden here) *and* a quality harness that does not exist yet
   (`spec/logit-distribution-probe`, unbuilt). Note that 2.46 bpw from a
   **4-bit-QAT** model is a 38% rate cut; I would not bet on it.
2. **More than 273 GB/s.** Requires different silicon.
3. **Verify-gated speculation**, which the brief correctly excludes from scope.
   It is the only route that changes *tokens per byte* rather than bytes per
   token, and it is out of bounds here by the user's constraint, not by physics.

---

## 5. What I did not establish

- Whether 246.2 GB/s is the hardware's ceiling or my probe's. My best median arm
  is 240.3 and my noise floor is ~1% with per-arm spreads to 13%; a clean
  answer needs a counter that no single reading on this box can give.
- **Any quality measurement.** No perplexity, no KL, no greedy-agreement rate was
  computed. Every bpw figure in §3.3 is an arithmetic requirement against a model
  that does not exist on this machine.
- Whether the real gated EXL3 GEMV sustains 239.8 GB/s. I cited the prior's
  figure; I did not re-measure it, because doing so needs an 11 GiB model load and
  the other agent had the box. My synthetic ceiling (246.2) and their real-path
  figure (239.8) differ by 2.7%, which is inside the noise floor of both.
- Prefill. Different roofline, out of scope.

---

## 6. On instruments — seven defects in mine, mirroring the prior's seven

`crux.md` §6 records seven bandwidth instruments in this repository that reported
bandwidth without measuring it, three of them the prior author's own. I built two
instruments and **found seven defects in them**, five of which would have produced
a plausible wrong number. This is the strongest evidence I have that the prior's
warning is not editorial.

| # | defect | how it was caught | what it would have reported |
|---|---|---|---|
| 1 | selection control compared only a **grand total**, so a *transfer* between two families was invisible | sabotaged it and watched it pass | any re-bucketing of bytes between `q_proj` and `down_proj` |
| 2 | fill guard asserted sampled values are **distinct** | it failed on a *correct* fill | nothing — but a guard that cries wolf gets deleted |
| 3 | fill guard sampled **every buffer at the same line offsets**, so 8 identical buffers produced 17,500 false "duplicates" | read the expectation (3.0) against the observation | a false rejection of every good run |
| 4 | fill wrote **4 B per element** while the kernel read 16 B, leaving 3/4 of every buffer untouched | position-purity guard | a "bandwidth" over 25% of the requested bytes |
| 5 | **`j · odd` overflowed u32**, so the block "permutation" silently stopped being a bijection | exhaustive multiset proof: 2,211,840 double-visits | an entropy arm that was skipping and repeating regions |
| 6 | checksum gate compared a **1-buffer arm to the all-buffer total** | it failed a run whose arms were all correct | the gate, not the arms, was wrong — and it was wrong *silently enough to be trusted* |
| 7 | **`--gb 8` (space form) silently fell back to 6 GB** while printing its own settings | I read the printed `usable … of …` line against what I asked for | two trials I had labelled "8 GB" that were 6 GB |

Defect 7 is the most instructive, because it is the exact failure the brief
warns about: the instrument reported what it was asked for rather than what it
did, and every number on screen looked fine. Defect 2 is the second most
instructive, because a *correct* instrument was about to be thrown away for
failing a guard that was itself wrong.

The generalisation is the prior's and I endorse it: **a bandwidth number is an
absolute claim about the wire, and an absolute claim is the one kind of thing a
ratio cannot validate.** Defect 1 is a ratio-only failure — the total was right
and the decomposition was wrong. Defect 5 is a coverage-only failure. Neither is
visible to a positive control on the *total*. Every guard that caught something
here compared against something outside the measurement.

### Reproducing this

```bash
# bytes/token — header-only, no GPU, no model load
cargo run --release -p kaplumba-engine --example crux2_bytes
cargo run --release -p kaplumba-engine --example crux2_bytes -- --sabotage        # must FAIL
cargo run --release -p kaplumba-engine --example crux2_bytes -- --sabotage-select # must FAIL

# every text tensor, with dtype/shape/bytes; the conditional-read audit
cargo run --release -p kaplumba-engine --example crux2_dump

# the lm_head exclusion bound, from the checkpoint's own scales
cargo run --release -p kaplumba-engine --example crux2_head_bound

# bandwidth — synthetic; check the box first, these allocate GBs
scripts/gpuq status && ps -Ao pid,args | grep kaplumba
target/release/examples/crux2_bw --gb 8 --rounds 9
target/release/examples/crux2_bw --gb 8 --rounds 2 --sabotage      # must exit 0 with per-arm predictions
target/release/examples/crux2_bw --gb 2 --rounds 1 --sabotage-fill # must exit 101
```

Worktree with its own `CARGO_TARGET_DIR` per `AGENTS.md` §6 / §10.2 #4. No
benchmark was left running; the box was idle at the end.