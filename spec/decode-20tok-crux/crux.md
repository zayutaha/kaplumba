# Can this box sustain 250 GB/s? Can it do 20 tok/s at 16k?

**Answer, in two parts, because the two questions have different answers.**

**Pure decode at 16k: no. Not at 3 bpw, not at any kernel speed, not at any
context length worth having.** The requirement is arithmetic and it is settled
before any code is written. 20 tok/s at 16,384 tokens of context needs
**284.8 GB/s**. The wire peak is 273 GB/s. Measured sustained streaming on this
box is **245 GB/s**, and the best single arm ever measured on it is
**253.6 GB/s**. So the target needs **104.3% of the theoretical pin rate**,
**112.3% of the best arm**, and **116.2% of what the hardware sustains** — a
shortfall of 31–40 GB/s depending on which honest figure you plan against.

**The 220 GB/s figure was wrong, and finding that out is the one place the
prior in this repo was too pessimistic.** Achievable streaming bandwidth here is
**252–254 GB/s** on the best arm and **245 GB/s** sustained — up to 92.9% and
89.7% of wire. That is *above* the 82–86% DRAM efficiency Apple silicon is
published at, and above the 239.8 GB/s that the repo's own gated GEMV harness
measures on real EXL3 shapes. It is not enough. Nothing on the bandwidth axis
is left.

**Where 20 tok/s IS reachable at 16k: verify-gated speculative decoding.** The
pure-decode roofline is unbreakable because bytes/token is fixed at 14.239 GB.
Speculation attacks the other half of the ratio. At today's measured 81.3 ms
step, 20 tok/s needs **1.63 accepted tokens per pass** — below the 2.4 that
`spec/kv-free-drafter` already targets, which would give **29.5 tok/s**. That
route is quality-preserving by construction (verify-gated, greedy-identical),
and it is the only one on the table that does not trade quality for speed.

Everything below is a measurement that could have overturned the prior, or the
arithmetic showing why it could not. Where a previous number in this repo was
wrong it is recorded, including where **my own** instruments were wrong — three
times, and one of them survived a full clean run with every control passing.

---

## 1. Bytes per token — measured

### 1.1 The measurement, and the second measurement that confirms it

`spec/decode-19tok` says **13.145 GB/token**. `spec/decode-20tps` and
`spec/decode-25tok-peak-bandwidth` say **11.834 GB**. The repo disagrees with
itself by 11%, and both figures were produced by arithmetic on paper rather than
by reading the checkpoint.

**12.0580 GB of weights per decode token** is the correct figure, and it was
confirmed twice by two paths that share no code:

1. **safetensors headers** (`crates/engine/examples/crux_bytes.rs`, header-only,
   no GPU, no model load).
2. **`quantization_config.json`'s own `tensor_storage` map**, which records
   `n_bytes` for every stored tensor. I extracted all 2,006 entries with `jq`
   and summed them by owning matrix — a completely independent source, since
   this file is written by the quantizer and the headers by the writer.
   Raw: `quantization-config-tensors.tsv`.

Both give **12,058,001,636 B** / **12,057,990,764 B** respectively — agreeing to
11 bytes out of 12 billion (the difference is the `norm`/`q_norm` bucketing).

### 1.2 The decomposition, and a cross-check that lands exactly

| bucket | bytes/token | GB | share |
|---|---:|---:|---:|
| `q_proj` | 1,156,055,040 | 1.1561 | 9.59% |
| `o_proj` | 1,156,055,040 | 1.1561 | 9.59% |
| `gate_proj` | 2,601,123,840 | 2.6011 | 21.57% |
| `up_proj` | 2,601,123,840 | 2.6011 | 21.57% |
| `down_proj` | 2,601,123,840 | 2.6011 | 21.57% |
| `lm_head` | 1,056,964,608 | 1.0570 | 8.77% |
| `k_proj` | 454,164,480 | 0.4542 | 3.77% |
| `v_proj` | 412,876,800 | 0.4129 | 3.42% |
| side tensors (`.suh`/`.svh`/`.mcg`) | 15,840,364 | 0.0158 | 0.13% |
| norms | 2,663,032 | 0.0027 | 0.02% |
| `embed_tokens` row | 10,752 | 0.0000 | 0.00% |
| **weights total** | **12,058,001,636** | **12.0580** | 100% |

The cross-check that matters is not a sum, it is a **per-projection identity**.
Take the independently-summed trellis bytes, divide by the declared rate, and
compare with the parameter count the checkpoint's own geometry predicts. Every
row matches **exactly**:

| matrix | trellis bytes | implied params | geometry | verdict |
|---|---:|---:|---:|---|
| `q_proj` | 1,156,055,040 | 3,082,813,440 | 50×5376×8192 + 10×5376×16384 | **exact** |
| `o_proj` | 1,156,055,040 | 3,082,813,440 | 10×16384×5376 + 50×8192×5376 | **exact** |
| `k_proj` | 454,164,480 | 1,211,105,280 | 50×5376×4096 + 10×5376×2048 | **exact** |
| `v_proj` | 412,876,800 | 1,101,004,800 | 50×5376×4096 (**no v on full layers**) | **exact** |
| `gate_proj` | 2,601,123,840 | 6,936,330,240 | 60×5376×21504 | **exact** |
| `up_proj` | 2,601,123,840 | 6,936,330,240 | 60×5376×21504 | **exact** |
| `down_proj` | 2,601,123,840 | 6,936,330,240 | 60×5376×21504 | **exact** |
| `lm_head` | 1,056,964,608 | 1,409,286,144 | 262144×5376 @ 6 bpw, untied | **exact** |

Every layer tensor is **exactly 3.000 bpw**; `lm_head` is **exactly 6.000**.
There is no slack anywhere in the checkpoint, which closes the whole
"repacking recovers bytes" family before it starts (§5, dead end 1).

Two geometry facts fall out of the identity and both matter later:

- **The 10 full-attention layers project to 32 × `global_head_dim` = 32 × 512 =
  16,384, not 8,192.** `q_proj` and `o_proj` on those 10 layers are *double*.
- **`v_proj` exists in only 50 matrices, not 60** (`attention_k_eq_v: true`).

### 1.3 The KV cache, which grows with context

fp16, from config geometry: 50 sliding layers hold `2 × 16 × 256 × min(C,1024)`,
10 full layers hold `2 × 4 × 512 × C`.

| context | sliding MB | full MB | KV GB | % of step | **bytes/token** |
|---:|---:|---:|---:|---:|---:|
| 614 | 503.0 | 50.3 | 0.5533 | 4.4% | **12.6113** |
| 1,024 | 838.9 | 83.9 | 0.9227 | 7.1% | **12.9807** |
| 2,048 | 838.9 | 167.8 | 1.0066 | 7.7% | **13.0646** |
| 4,096 | 838.9 | 335.5 | 1.1744 | 8.9% | **13.2324** |
| 8,192 | 838.9 | 671.1 | 1.5099 | 11.1% | **13.5680** |
| **16,384** | 838.9 | 1,342.2 | **2.1810** | **15.3%** | **14.2390** |
| 24,576 | 838.9 | 2,013.3 | 2.8521 | 19.1% | **14.9101** |

The sliding window is the only reason short context is not hopeless: at 614
tokens KV is 4.4% of the step. At 16k it is 15.3%, and KV is **immovable** —
AGENTS.md §8 forbids quantizing it. Any roofline quoted at weights-only
overstates 16k tok/s by **18.2%**.

### 1.4 Reconciling 11.834 and 13.145

**Neither is right. 12.0580 GB is.**

**11.834 GB** (`spec/decode-25tok-peak-bandwidth` §16.4) is arithmetic that used
`head_dim` (256) for `q_proj`/`o_proj` in **all 60 layers** and added a
`v_proj` to the 10 full layers that have none. Both errors are visible in §1.2's
identity table. Its direction was predicted in advance by
`spec/decode-ceiling/findings/A5-budget.md`; I get 165,150,720 B for the missing
`q` alone, and 350,945,280 B for extra q+o+k, so **treat A5's specific number as
unverified** — its direction and rough size are right, its digits are not.

**13.145 GB** (`spec/decode-19tok`) is **not reproducible from any geometry
mistake.** It over-counts by 1.087 GB (+9.01%), and its per-projection ratios to
measured are mutually inconsistent — gate/up/down 1.104, q/o 1.064, k 1.160,
v 1.175, head exactly 1.000. A single systematic error produces a *consistent*
ratio pattern; this is not one. All 8 combinations of {q_out, kv_out,
v_present} were enumerated and none reaches it. **Verdict: unsupported. Discard
it.**

---

## 2. Achieved bandwidth — measured

### 2.1 Method, and why the box state is part of the number

Two independent synthetic probes, both reading a buffer far larger than any
cache on this chip, both with the same guard set:

- `crux_bandwidth` — grid-stride `uint4` over one buffer, 9 interleaved rounds,
  4 passes per command buffer, every arm measured in every round.
- `crux_multibuffer` (new, §2.4) — the same read spread over N **separately
  allocated** buffers.

Non-repeating `splitmix64` fill verified by sampling; `count * elem_bytes ==
bytes` asserted on the host; every arm must produce an identical checksum
before any timing is printed; `CONTROL_repeat` is byte-, thread- and
partition-identical to the reference arm and its difference **is** the noise
floor.

### 2.2 Occupancy, 2,560 → 245,760 threads — the sweep that was never run

`spec/decode-19tok` reported "bandwidth is flat from 8 to 32 bytes in flight
per thread" and inferred a **request-rate wall**. Its sweep stopped at 20,480
threads while the device allows 81,920, so the occupancy axis was never moved —
and a flat curve on one axis does not license a conclusion about a limit on a
different one. This is the axis, run to 3× the device maximum:

| threads | median | best | worst | spread | waves @81,920 |
|---:|---:|---:|---:|---:|---:|
| 2,560 | 125.8 | 126.0 | 125.6 | 1.00× | 0.03 |
| 5,120 | 214.3 | 217.4 | 207.4 | 1.05× | 0.06 |
| 10,240 | 235.3 | 237.4 | 231.5 | 1.03× | 0.12 |
| **20,480** | **250.5** | 251.2 | 238.2 | 1.05× | 0.25 |
| 30,720 | 240.4 | 242.1 | 236.5 | 1.02× | 0.38 |
| 40,960 | 247.3 | 248.4 | 245.0 | 1.01× | 0.50 |
| 61,440 | 242.8 | 245.9 | 242.4 | 1.01× | 0.75 |
| **81,920** (device max) | **245.8** | 246.6 | 244.2 | 1.01× | 1.00 |
| 122,880 | 239.0 | 244.2 | 236.3 | 1.03× | 1.50 |
| 163,840 | 243.2 | 244.9 | 239.5 | 1.02× | 2.00 |
| 245,760 | 243.5 | 244.0 | 240.1 | 1.02× | 3.00 |

**20,480 → 81,920 threads: −1.9%.** Flat over a 4× range, and the spread
across that whole region is 1.05×. The rise below 20,480 is real and large
(125.8 → 250.5, +99%), so 20,480 is a genuine enough-requests threshold — it is
simply already reached at a quarter of the thread budget. **Moving occupancy 4×
changes nothing.**

Corroboration from a probe with a different shape entirely:
`spec/decode-bandwidth-240gb` swept **0.08M–2.62M loads in flight (32×)** and
recorded **1.06×** across it. That is a superset of this axis and it agrees.

### 2.3 Bytes in flight, load width, requests, address streams

At the device maximum, 81,920 threads. Noise floor this run: **0.00%**.

| arm | B/load | in flight | median | vs `stream_u4` | req/GB (M) |
|---|---:|---:|---:|---:|---:|
| `CONTROL_repeat` | 16 | 16 | 244.8 | 0.0% | 62.5 |
| `CONTROL_tg512` | 16 | 16 | 245.6 | +0.3% | 62.5 |
| `stream_u4` | 16 | 16 | **244.8** | — | 62.5 |
| `stream_u2` | 8 | 8 | 243.6 | −0.5% | 125.0 |
| `stream_u4_u2` | 16 | 32 | 236.5 | −3.4% | 62.5 |
| `stream_u4_u4` | 16 | 64 | 236.0 | −3.6% | 62.5 |
| `stream_u4_u8` | 16 | 128 | 234.8 | −4.1% | 62.5 |
| `stream_u1` | 4 | 4 | 223.6 | −8.7% | 250.0 |
| `stream_u4_u16` | 16 | 256 | 205.1 | −16.2% | 62.5 |
| `stream_u1_u4` | 4 | 16 | 178.2 | −27.2% | 250.0 |
| `stream_u1_u8` | 4 | 32 | 162.4 | −33.7% | 250.0 |

Every streaming arm moved more than the instrument's own 0.00% repeat spread,
so every row is real. Three readings:

- **Unrolling only hurts.** Monotonically worse with more bytes in flight: −3.4%
  at 32 B, −16.2% at 256 B. There is no ILP headroom to recover here.
- **Requests matter more than bytes.** `stream_u1_u4` and `stream_u4` have
  *identical* bytes in flight (16 B) and differ only in request count (4 vs 1).
  The arm issuing 4× the requests is **27.2% slower**. So the limiter is
  requests-per-byte, not bytes-in-flight. This is the part of the
  `decode-19tok` claim that survives, and it now has a measurement behind it.
- **Narrow loads alone are only −8.7%**; the sharp penalty needs narrow loads
  *and* unrolling together.

**Independent address streams** (slices of one buffer, 81,920 threads):

| streams | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| median GB/s | 231.8 | 231.5 | 231.8 | 232.6 | 230.3 | 223.4 | 221.1 |
| vs 1 stream | — | −0.1% | −0.0% | +0.3% | −0.7% | −3.6% | −4.6% |

Flat to 8, negative past 16. This axis runs ~13 GB/s below the table above at
the same occupancy — the multi-stream kernels are simply a less efficient
implementation, and that offset is constant, so the *shape* is the result and
the level is not a second bandwidth claim.

### 2.4 Independent allocations — the axis that did not exist, and it is dead

Every address-stream arm above carves **one** buffer into S slices. Slices of a
single allocation share physical pages, so a DRAM **channel or bank imbalance
would appear identically in all of them and be invisible**. A real decode token
touches 411 separate tensor buffers. So this is the only shape that could have
shown channel-level parallelism.

`crux_multibuffer` reads N separately allocated buffers concurrently, same
`uint4` grid-stride, total bytes and total content identical across every arm
(verified against a CPU-computed checksum of the fill, not just arm-to-arm).
Two independent runs:

| N allocations | run A @20,480 | run B @20,480 | run A @81,920 | run B @81,920 |
|---:|---:|---:|---:|---:|
| 1 | 252.5 | 251.8 | 234.2 | 233.3 |
| 2 | 252.6 | 252.2 | 233.1 | 233.4 |
| 4 | 252.8 | 252.5 | 233.9 | 234.2 |
| 8 | 252.8 | **253.6** | 234.2 | 233.9 |
| noise floor | −0.49% | −0.12% | | |

**1 → 8 independent allocations: +0.12% (run A) and +0.72% (run B), against
noise floors of −0.49% and −0.12%.** Channel and bank parallelism is not
available. This was the last axis, and it is closed.

### 2.5 Working set at the real size — measured, but inconclusive

Weights are 12.058 GB resident and a 16k token streams 14.2 GB, so the 6 GB
working set is smaller than the thing it proxies for. `--gb 10` was attempted
twice. **Both runs are void**: the 10 GB allocation drove the box to 3.6 GB
free, the probe's own control read **−29.26%**, and an interleaved
`crux_bandwidth` control run immediately afterwards read **+7.79%** against
**+0.67%** before. This is a memory-capacity limit, not a bandwidth finding —
a 24 GiB box cannot hold a 10 GB probe plus the OS, the browser and the agent
tree with enough headroom to measure anything.

The plateau question is already answered by a harness that does not re-fill per
arm: `spec/decode-bandwidth-240gb` records working-set size as **"rises to 4 GB
then plateaus; 1 GB is the slowest large size"**. Taken together: flat.

### 2.6 The number

| | GB/s | % of 273 wire peak | how obtained |
|---|---:|---:|---|
| **best single arm (any probe, any run)** | **253.6** | **92.9%** | 8 independent buffers, 20,480 threads |
| best arm, `crux_bandwidth` axis A | 250.5 | 91.7% | 5-trial median of best arm |
| best arm, `crux_multibuffer` run A | 252.8 | 92.6% | 8 independent buffers |
| best arm, prior session on this branch | 253.6 | 92.9% | same probe, different day |
| **planning figure (sustained)** | **245** | **89.7%** | top-half median, 5 trials |
| `stream_u4` @ device max | 244.8 | 89.7% | reference arm |
| **real 3 bpw EXL3 GEMV** | **239.8** | **87.8%** | `decode-bandwidth-240gb` `conc.rs`, gated + checksum-verified |

Five interleaved trials of the whole sweep, on a quiet box: best arm
**249.3–250.8** (spread 0.6%), top-half median **242.4–243.7** (spread 0.5%),
control noise floor 0.00–0.35%. Raw: `crux-bandwidth-raw-agent3.txt`.

**Bandwidth wall or request-rate wall, stated plainly: this is a BANDWALL at
~90% of the theoretical wire peak, reached at 20,480 threads.** The maximum
observed on any axis is **253.6 GB/s**; the sustained figure to plan against is
**245 GB/s**. The request-rate observation in §2.3 is real but it is a
statement about *how to lose* bandwidth, not a lever: every configuration that
issues fewer requests per byte is at or below the wall already.

The 220 GB/s figure is **wrong**, and wrong for an identifiable reason:
`decode-bandwidth-240gb` traces it to the old probe's *shape* (4 B scalar loads
at 20,480 threads), worth ~7% against the same sizes. The box does not do 220.

### 2.7 The measurement is only valid when the box is quiet, and it often is not

This deserves its own section because it is the single biggest risk to any
bandwidth number on this machine. Two runs during this work, both on a box that
`gpuq status` and `ps` reported as **idle with no model resident**, returned:

| run | best arm | sustained | control noise floor | axis-S spread |
|---|---:|---:|---:|---:|
| quiet trial 3 | 250.5 | 243.5 | **+0.00%** | 1.01× |
| contaminated 1 | 138.6 | 122.4 | −3.34% | 2.36–3.10× |
| contaminated 2 | 231.2 | 185.6 | +5.29% | — |

A **factor of two** in the headline number, from background load that left no
trace in `PhysMem`, in `ps`, or in the load average. The instrument's own
positive control is what caught it both times — it is not decoration, it is the
only thing standing between this box and a confidently wrong number. Any
bandwidth figure from this repo that does not carry a control reading should be
treated as unmeasured.

---

## 3. The verdict

### 3.1 The arithmetic

At 16,384 tokens:

```
bytes/token = 12.0580 (weights) + 2.1810 (KV) = 14.2390 GB
20 tok/s    = 50.000 ms/token
required    = 14.2390 / 0.050000 = 284.8 GB/s

  vs 273   GB/s wire peak        -> 104.3%   IMPOSSIBLE  (short by 11.8 GB/s)
  vs 253.6 GB/s best arm         -> 112.3%   IMPOSSIBLE  (short by 31.2 GB/s)
  vs 245   GB/s sustained        -> 116.2%   IMPOSSIBLE  (short by 39.8 GB/s)
```

Pure-decode ceiling as a function of achieved bandwidth, **zero overhead**:

| achieved GB/s | ms/token | tok/s @614 | tok/s @8k | **tok/s @16k** | 20 tok/s? |
|---:|---:|---:|---:|---:|---|
| 220 (the old, wrong figure) | 64.7 | 17.44 | 16.21 | 15.45 | no |
| 239.8 (real GEMV, measured) | 59.4 | 19.01 | 17.67 | 16.84 | no |
| **245 (measured sustained)** | 58.1 | 19.43 | 18.06 | **17.21** | no |
| 253.6 (best arm, measured) | 56.1 | 20.11 | 18.69 | **17.81** | no |
| 273 (100% of theoretical) | 52.2 | 21.65 | 20.12 | **19.17** | no |
| 284.8 (what 20 tok/s needs) | 50.0 | 22.58 | 20.99 | 20.00 | — |

Three readings, in order of how much they matter:

1. **Even at 100% of theoretical peak, 16k gives 19.17 tok/s.** Not 20. The
   ceiling is below the target before a single line of code exists.
2. **Reaching 20 tok/s needs 285 GB/s, which is above the pins.** No kernel can
   exceed the hardware's connectors.
3. **The @273 row is a bound, not a plan.** Every figure in it is unattainable.
   Note that 614 tokens *does* clear 20 tok/s in the @253.6 row — that is the
   single most favourable cell in this document, and §3.2 says what it is worth.

### 3.2 Where pure decode *does* clear 20 tok/s

Solving `12.058 + KV(C) ≤ BW/20`:

| bandwidth | budget GB | left for KV | shortest context |
|---:|---:|---:|---:|
| 245 measured sustained | 12.250 | 0.192 GB | **213 tokens** |
| 253.6 best arm measured | 12.680 | 0.622 GB | **690 tokens** |
| 273 theoretical peak | 13.650 | 1.592 GB | **1,767 tokens** |

**One paragraph of context, not a conversation.** Against the sustained figure
pure decode misses everywhere. Against the single best arm it nominally clears
at 690 tokens — with zero overhead and no dequant cost, which is not a plan, it
is a boundary.

### 3.3 Therefore

| assumption | ceiling @16k | verdict |
|---|---:|---|
| 273 GB/s theoretical, zero overhead | 19.17 tok/s | fails by 4.3% |
| 253.6 GB/s best measured arm, zero overhead | 17.81 tok/s | fails by 11.0% |
| **245 GB/s measured sustained, zero overhead** | **17.21 tok/s** | **fails by 14.0%** |
| 239.8 GB/s real GEMV rate, zero overhead | 16.84 tok/s | fails by 15.8% |
| measured end-to-end today (12.3 tok/s) | 12.3 tok/s | 175.1 GB/s effective |

**The maximum achievable for pure decode at 16k is ~17.2 tok/s**, and that
grants *zero* overhead for norms, RoPE, attention, sampling and dispatch gaps —
so the honest number is below it. Reaching it from today's 12.3 tok/s is a
**1.40×** improvement; reaching 20 tok/s is **1.63×**, and no bandwidth exists
to fund it.

Note where the observed rate sits: 12.3 tok/s × 14.239 GB = **175 GB/s
effective end-to-end**, against 239.8 GB/s of measured GEMV weight bandwidth.
So the decode step is at **73%** of the rate its own GEMVs already achieve. That
27% is the non-GEMV part of the step, and **it is the only place there is left
to work** — but it is worth 1.40×, not 1.63×, and it cannot reach 20.

---

## 4. Quality-preserving escape routes

Each expressed as a hard requirement, so it can be checked rather than debated.

### 4.1 Verify-gated speculative decoding — the route that works

**This is the answer to "where is 20 tok/s reachable".** Speculation does not
reduce bytes per token; it raises **tokens per byte**, which is the other half of
the ratio the whole document has been dividing by.

A target pass verifies *n* drafted tokens in **one** forward. The weights stream
once (12.058 GB) and the KV streams once (2.181 GB) — a flash-decode kernel
loops KV blocks outside and holds all *n+1* queries in registers, so the
verification positions do **not** re-read the cache. Bytes per *pass* is
therefore unchanged at **14.239 GB** while tokens per pass becomes *n+1*.

At today's **measured 81.3 ms step** (12.3 tok/s @16k):

| accepted/pass | tok/s @16k | bandwidth needed for 20 tok/s |
|---:|---:|---:|
| 1.000 (today) | 12.3 | 284.8 GB/s — impossible |
| 1.500 | 18.5 | 189.9 GB/s |
| **1.626** | **20.0** | **175.1 GB/s — exactly what the box already does** |
| 2.000 | 24.6 | 142.4 GB/s |
| **2.400** (`kv-free-drafter` target) | **29.5** | 118.7 GB/s |
| 3.000 | 36.9 | 94.9 GB/s |

**20 tok/s at 16k needs 1.63 accepted tokens per pass.** That is *below* the 2.4
`spec/kv-free-drafter` already targets, and below the ~2.1 `spec/mtp-40tps`
computes for 40 tok/s. At 2.4 the projection is **29.5 tok/s**, clearing the
target by 48%.

Why this is not cheating, and why it is not the same as the arithmetic above:
verify-gated speculation is **distribution-exact by construction** — the target's
logits decide every emitted token, and greedy output is bit-identical to
non-speculative decode. AGENTS.md §8 requires exactly this and nothing here
weakens it. The arithmetic in §3 is untouched: it is still true that a *single*
forward pass cannot exceed 17.2 tok/s. Speculation is how you stop paying for
one forward pass per emitted token.

The one measured fact against it: the released `gemma-4-31b-it-qat-assistant-4bit`
drafter accepts **0.883/round**, which gives **10.9 tok/s** — worse than no
speculation at all, and the reason `kv-free-drafter` exists.

### 4.2 Reduce bpw — a quality decision, not a code change, and it still fails

The 50 ms budget is `BW/20` GB. Paying the 2.181 GB KV cache and the fixed
1.0575 GB head (held at 6 bpw, since it is 8.8% of the step and re-rating it is
a bigger quality decision than re-rating the layers) leaves the rest for the
29,286,727,680 layer parameters:

| achieved GB/s | budget GB | − KV | − head | for layers | **layers bpw** | vs 3.000 |
|---:|---:|---:|---:|---:|---:|---:|
| **245 (measured sustained)** | 12.250 | 2.181 | 1.0575 | 9.012 | **2.462** | **−17.9%** |
| 253.6 (best arm) | 12.680 | 2.181 | 1.0575 | 9.442 | 2.579 | −14.0% |
| 273 (theoretical peak) | 13.650 | 2.181 | 1.0575 | 10.412 | 2.844 | −5.2% |
| 284.8 (what 20 tok/s needs) | 14.240 | 2.181 | 1.0575 | 11.002 | 3.005 | +0.2% |

**Every row at any bandwidth this box has is SHORT.** Even at 100% of
theoretical peak the layers must drop from 3.000 to 2.844 bpw — a 5.2% rate
reduction on every linear layer. Only above **284.8 GB/s**, i.e. above the wire
peak, does 3.000 bpw fit at all.

Note the non-monotonicity, which is worth understanding rather than misreading
as a table error: a *lower* bandwidth is *easier* to hit a bpw target, because
the same 50 ms budget buys fewer bytes. So at 245 GB/s the required rate is
2.462 bpw, a mild-looking 18% cut, while at 273 GB/s it is 2.844, only 5.2%.
**The bpw column is not "how much quality we can afford" in isolation; what
matters is that no bandwidth this box reaches lets 3.000 bpw fit.**

And below the point where greedy output stops being byte-identical this is out
of bounds anyway: I am not proposing it. I am recording that the arithmetic does
not offer it even as a trade.

### 4.3 KV-cache quantization — forbidden, and it would not deliver anyway

Deleting all 2.181 GB of KV cuts the requirement to **241.2 GB/s = 98.4% of the
measured sustained rate** (95.1% of the best arm). Unlike every weight-side
route this *does* clear the arithmetic — by 1.6%. But 1.6% buys nothing against
a real 3 bpw GEMV that already runs at 239.8 GB/s, and **AGENTS.md §8 forbids
it.** Listed because it is the largest single non-weight lever and deserves to
be revisited deliberately rather than inherited silently — not because lifting
it would deliver 20 tok/s.

### 4.4 Shorter context

Against the sustained figure, 20 tok/s needs context under **213 tokens** (§3.2).
A product retreat, not a route.

### 4.5 Routes that cannot work

Anything that reduces *arithmetic* without reducing *bytes*: `decode-19tok-
repacked-250gb` measured removing `decode_state` at **−26.7%**, i.e. every rate
got *slower*. The bytes are the cost.

---

## 5. Dead ends, with numbers

| # | idea | measured result | source |
|---|---|---|---|
| 1 | Repack to recover bytes | 410/411 tensors **exactly 3.000 bpw**, `lm_head` exactly 6.000. Zero slack. | §1.2 |
| 2 | Load width is a lever | 4/8/16/32 B all 223–245. Unrolling monotonically *worse*: −3.4% at 32 B in flight, −16.2% at 256 B. | §2.3, `decode-bandwidth-240gb` |
| 3 | **Occupancy is the limiter** | **−1.9% from 20,480 → 81,920 threads**, flat over 4×. Corroborated 32× further out at 1.06×. | §2.2 |
| 4 | **Independent allocations unlock channel parallelism** | **+0.12% and +0.72% from 1 → 8 buffers**, against noise floors of −0.49% and −0.12%. Two runs. | §2.4 |
| 5 | Address-stream count is a lever | Flat to 8 (+0.3%), negative past 16 (−4.6% at 64). | §2.3 |
| 6 | Row locality / contiguity | 4 KiB → 3 MiB all 243–253, spreads overlapping. **Zero effect.** | `decode-bandwidth-240gb` |
| 7 | Working-set size | Rises to 4 GB then plateaus; 1 GB is the slowest large size. | `decode-bandwidth-240gb` |
| 8 | `uint4` widening | 180.5 → **173.7** GB/s on `gate_up`. A loss. | `decode-19tok-repacked-250gb` |
| 9 | `T14 D=32` (halve amplification directly) | **0.288–0.918×** on all 11 real shapes. Never faster. | same, §6.6 |
| 10 | Cheaper decode arithmetic | Removing `decode_state`: 3 bpw **−26.7%**. | same |
| 11 | `simd_shuffle` cooperative loads | 180.5 → **146.5** GB/s. | same |
| 12 | Multi-simdgroup on narrow N | 178 → **110.8** GB/s (grid too small). | same |
| 13 | `MTLResidencySet` pinning | **SIGSEGVs** on this binding. | same |
| 14 | 4 bpw | Net loss: bytes ×1.333 vs bandwidth ×1.249. | `shim-exl3-4bpw` |
| 15 | Lane-load amplification is avoidable by relayout | **Disproved.** 3 bpw needs a 37-bit window, so every lane reads 2 words: amp 2.667 is a floor, not an accident. | `decode-3bpw-amplification` §3 |
| 16 | "250 GB/s streaming is achievable" | Number **vindicated** (253.6 GB/s here, first try), **provenance refuted** — the old probe tiled a 64 MiB pattern and never faulted pages in. | §6 |
| 17 | bytes/token = 11.834 GB | Wrong geometry: `head_dim` for all 60 layers + phantom `v_proj` on 10. | §1.4 |
| 18 | bytes/token = 13.145 GB | Not reproducible from any geometry error. Over-counts 9.01%, inconsistent ratios. | §1.4 |
| 19 | Measured working set at 10–12 GB | **Void.** Box fell to 3.6 GB free; own control −29%, interleave control +7.8%. | §2.5 |
| 20 | 20 tok/s by pure decode anywhere useful | Misses at every context length: 213 tokens sustained, 690 best-arm, 1,767 only at 100% of theoretical peak. | §3.2 |
| 21 | Quantizing KV rescues 3 bpw 16k | Removing KV entirely still needs 241.2 GB/s = 98.4% of measured. Changes which wall, not whether. | §4.3 |
| 22 | Re-rating `lm_head` below 6 bpw buys anything | It is 1.0575 GB, 8.8% of the step. Holding it fixed is the *optimistic* assumption in §4.2; freeing it entirely would reach 2.462 → ~2.62 bpw on layers, still 13% short at 245 GB/s. | §4.2 |
| 23 | Byte amplification 2.67× means 3 bpw moves 32 GB/token | **No.** Amplification is *lane-level* L1 traffic inside a simdgroup, not DRAM traffic. Measured: real GEMV sustains 239.8 GB/s of **DRAM**, i.e. 98% of the 245 GB/s streaming ceiling. The amplification costs load instructions, and instructions are 8× under the knee. | §2.6 |

---

## 6. On measurement instruments

**Seven** bandwidth instruments in this repository have reported bandwidth
without measuring bandwidth. All are the same family: the number is physically
implausible or wrong, and the table it sits in looks fine.

| probe | claimed | the defect |
|---|---:|---|
| `examples/bw.rs` | 55 GB/s | Never wrote its buffer. Read zero-filled pages. |
| `examples/bw.rs` | 248.5 GB/s | Reported min-of-passes; on a shared box that selects the pass that ran while a competitor was idle. |
| `examples/probe_bw_250gb.rs` | 254.0 GB/s | Tiled a 64 MiB pattern across the buffer (`:15-33`) and never faulted pages in. A second pass re-reads the same addresses — that is a cache hit, reported as DRAM. |
| `layer_bench.rs:78` | — | Hardcodes `rate: Integer(3)`, so its "lm_head" was a 3 bpw stand-in, not the real 6 bpw 262144-wide head. |
| `gemv_shape.rs` | 131,512 GB/s | Divided by `passes`, then formatted the result as **milliseconds**. |

The same family appeared **in my own new probe**, twice, and both are in the
git history of this branch:

1. **Crediting 80× the bytes moved.** The first `stream_mb` gave each thread its
   own `chunk = count / chunks` sub-slice. Each thread therefore read
   `count/80` elements while the host divided `bytes` (100% of the buffer) by
   the elapsed time. It reported **13,758 GB/s — 5,039% of wire peak.**
2. **A 3 GB out-of-bounds read.** `duplicate_lines` added `b * bytes_per_buf` to
   an offset that was *already* relative to buffer *b*'s base pointer, so the
   first multi-buffer arm **SIGSEGVed** (exit 139) — 3 GB past the end of a
   3 GB buffer.

Both were caught, and neither by a symptom a reader would notice: the first only
because 5,039% of a physical constant is absurd, the second because it crashed.
The **CPU-computed reference checksum** added afterwards is the guard that makes
this class unrepresentable — it is computed from the fill, independently of
anything the GPU said, so a kernel that skips, repeats or mangles a byte range
cannot agree with it.

The third, from the prior session on this branch and worth repeating because it
is the most instructive: a **factor of two** in the byte accounting
(`count = bytes/elem/2`, credited with all of `bytes`) reported **505.5 GB/s,
185% of wire** and **survived a complete clean run with every control passing** —
sabotage verified, checksum gate passed, noise floor a clean 0.09%, box idle.
Every *ratio* the sweep reported was perfectly sensible, because a uniform
factor of two cancels in ratios. Only the absolute scale lied, and nothing in
the probe compared the absolute scale to anything.

The generalisable part is not "check the fill pattern". It is that **a bandwidth
number is an absolute claim about the wire, and an absolute claim is the one
kind of thing a ratio cannot validate.** The checks that actually caught things
here were all checks against something outside the measurement: a datasheet
constant, an independently computed checksum, the byte count the kernel was
told to visit, a positive control that cannot change.

---

## 7. What this does **not** establish

- **Whether 253.6 GB/s is the hardware's ceiling or these probes' ceiling.** The
  curve is flat above 20,480 threads and every axis — occupancy (4× and,
  elsewhere, 32×), load width, unroll, request count, address streams,
  allocation count, working-set size, row locality — is flat or negative.
  `decode-bandwidth-240gb` adds a decisive version of the same argument: reading
  at stride 1, 2 and 4 while the hardware still fetches every sector gives a
  **flat 225–228 GB/s of sector traffic** while useful bytes halve and quarter.
  The hardware moves the same sectors no matter what the program does with them.
  That is about as close to a wall as a measurement gets.
- **A full accounting of the 27% gap** between the step's 175 GB/s effective and
  its GEMVs' 239.8 GB/s. §3.3 bounds it; attacking it is the decode
  optimization problem and out of scope here. A session note reporting the seven
  projections at "41–55% of peak" is **not reconciled** with the 239.8 GB/s
  gated figure and I did not reproduce it.
- **Prefill.** Different roofline. Out of scope.
- **Any quality measurement.** No perplexity was run. Every bpw figure in §4.2 is
  an arithmetic requirement, not a measured quality cost.
- **The acceptance rate a drafter can actually reach.** §4.1 takes 1.63
  accepted/pass as an arithmetic requirement. Whether a trainable drafter gets
  there is `spec/kv-free-drafter`'s experiment, not this one's.

---

## 8. Reproducing this

```bash
# bytes/token — header-only, no GPU, no model load
cargo run  --release -p kaplumba-engine --example crux_bytes
# independent second source: quantization_config.json's own n_bytes map
jq -r '.tensor_storage | to_entries[] | .key as $t | .value.stored_tensors
       | to_entries[] | "\(.value.n_bytes)\t\(.key)"' \
   ~/.kaplumba/models/Gxma/Gxma/quantization_config.json

# bandwidth — synthetic, refuses to run without a free box
scripts/gpuq status
target/release/examples/crux_bandwidth   --sabotage --sabotage-fill
target/release/examples/crux_multibuffer  --gb 6 --sabotage --sabotage-fill
```

Raw output for every table above is committed alongside this file:
`crux-bandwidth-raw-agent3.txt` (§2.1–2.3, 2.7),
`crux-multibuffer-raw.txt` (§2.4, 2.5), `crux-bytes-raw.txt` (§1),
`quantization-config-tensors.tsv` (§1.1).

Worktree with its own `CARGO_TARGET_DIR`, per AGENTS.md §6 / §10.2 #4, so a
concurrent rebuild could not land inside another agent's benchmark.
