# Round 1 & 2 — Consolidated results

24 agents run. Baseline throughout: **16.05–16.10 tok/s = 62.3 ms/step, 12.058 GB/token,
193.5 GB/s achieved.** Best 3-bpw kernel ever measured in this tree: **203 GB/s**.

## Verdicts

| ID | Verdict | One-line result |
|---|---|---|
| P02 dispatch/gaps | **REFUTED** | step is AT the 203 GB/s roofline; only 0.2–0.4 ms recoverable |
| P12 occupancy | **REFUTED** | 4 independent measurements close the axis; not the limiter |
| P14 activations | **REFUTED** | 0.53% of step bytes; ceiling +0.21 tok/s |
| M01 TLB | **REFUTED** | sequential access; 4 KB vs 16 KB rounding is provably identical |
| P11 hadamard/codebook | **REFUTED** | codebook = 51% of int ops but 5% of time; **but found dead `w2`** |
| P13 K-loop pipelining | **REFUTED** | 8 loads in flight already saturates; prefetch ≤ +0.3 |
| P09 shape/tail waves | **REFUTED** | tail model wrong 3 ways; **but found a 168-vs-336 grid** |
| P08 bytes/token | **REFUTED** | floor 11.53 GB is only −4.4%; max 16.44 tok/s from bytes |
| P17 dispatch selection | **CONFIRMED** | `contig4` dead; fail-open at rate≠3; **corrected my head claim** |
| P06 CPU host work | **CONFIRMED** | **temperature>0 does not pipeline: −2.6 ms/step** |
| P07 attention/elements | **partly** | T5.3 already shipped; 16k costs ~1 tok/s |
| P05 split-K | **CONFIRMED** | the 126–129 GB/s were ablation artefacts; real 178–182 |
| P15 KV scaling | **CONFIRMED** | 16k reads 2.182 GB → **13.63 tok/s** |
| K07 MCG vs Mul1 | **CONFIRMED** | MCG is the *cheap* one (4 vs 18 ops); no `_mul1` kernels exist |

## The four things worth doing, in order

| # | change | file:line | est. | why |
|---|---|---|---|---|
| **1** | **On-device top-k sampler** (top-k reduce + softmax + inverse-CDF in one dispatch), gate the pipelined loop on `greedy \|\| sampler_on_device` | new kernel; `model.rs:4459`; `chat.rs:388` | **+2.4–2.6 ms → 16.8 tok/s** | the engine's DEFAULT temperature is 1.0 and does **not** pipeline at all |
| **2** | **Drop the dead third ring word** (`w2`, `j2`, both funnels) guarded by `KA <= 4` | `linear.metal:356-364` | **+0.3–1.4 tok/s** | bit-exact proven: 0 mismatches / 1,048,576 codes. `w2` reaches bit 26; lowest bit read is 27; `c` is dead (`r_max=37 <= 64`) |
| **3** | `KAPLUMBA_O_SK2_1COL` — 1col+split2 reaches 672 TGs with **half** the reduction traffic of 2col+split4 | `model.rs:1943-1949` | **+0.73 tok/s** | o_proj is 1.51× worse per byte than gate_up; `_1col_sk2` already exists and is bit-exact |
| **4** | Incremental `Detokenizer` — hold the stream, drop the per-push replay | `tokenizer.rs:202-235` | ~0.1 ms, growing | quadratic: 0.017→0.105 ms/push over 119 tokens |

## Two real defects found

1. **`scratch.attended` is UNDERSIZED.** `model.rs:3012` sizes it `hidden` (5376) but both
   attention epilogues write `q_heads*head_dim` = 8192/16384 and `o_proj` reads all of it.
   **Overrun 5,632 B (sliding) / 22,016 B (full) per layer per token.** Currently masked only
   by Metal's page-granularity padding.
2. **No `_mul1` kernels exist.** `linear.rs:485-489` routes `Mul1` to the *unsuffixed*
   branchy kernel — a **2.61× cliff** (62.4 vs 163.0 GB/s) with no error. Upstream
   exllamav3 **1.0 flipped the default to mul1**, so any future mul1 checkpoint decodes
   silently at the slow rate *and* loses the 1.044× prefill win. Portability fix, 0% today.

## Measurement bugs that invalidate prior work

- **`KAPLUMBA_ABLATE=attention` is a no-op on decode** — checked only in
  `encode_online_attention`, but decode calls `encode_online_attention_split`. So the
  "7.32 ms residual with everything ablated" still contains 60 live attention kernels
  streaming ~860 MB of KV. **Most of that "recoverable residual" was never overhead.**
- **Profiling turns each stamp into a command buffer** (546 vs 1), making `own_ms` ~12%
  artefact. Real element cost by production ablation: **0.84 ms**, not 7.8.
- **The 126–129 GB/s per-projection figures are ablation artefacts** — they sum to 87.6 ms
  inside a 63.1 ms step. Rescaled: **gate_up 182, down 178** — within 10% of ceiling.
- **`546` was never the dispatch count** (it's profile stamps). Real: **750/token**.

## The arithmetic nobody can argue with

| | GB/s | vs 203 ceiling |
|---|---|---|
| achieved, whole step | 193.5 | 95% |
| 20 tok/s requires | **236.7** | **117%** |
| 21.5 tok/s requires | 259 | 128% |

Achieved **Gcodes/s is 334–399 across all five GEMV roles and flat across a 2× bit-width
difference** (reproduced independently from ablation deltas and from a microbenchmark).
**The loop retires ~340 Gcodes/s and that is the wall.** 20 tok/s needs 660.

## Context scaling (P15, P07)

| context | KV read/token | tok/s | vs 37 |
|---|---|---|---|
| 37 | 33 MB | 16.05 | — |
| 512 | 461 MB | 15.50 | −3.4% |
| 8 192 | 1.510 GB | 14.30 | −10.9% |
| **16 384** | **2.182 GB** | **13.63** | **−15.1%** |

Marginal cost of context, full layers only: **+0.42 ms per 1000 tokens**, linear.
The 50 sliding layers **never grow** (fixed 1024 ring, wraps via `%`, no memmove) —
**839 MB read every token forever**, a permanent 4.3 ms floor.

## Honest ceiling

**~16.8 tok/s** with all four changes above. **20 tok/s is not reachable on this
hardware** — four independent agents reached that conclusion by different routes
(roofline, code-throughput, byte-floor, host-work), and none found a path.

## Correction from the fusion census (P10) — the ceiling DROPS to 16.9

`implementation_notes.md` §16.4's **11.834 GB/token is wrong**. It multiplied a
*sliding* layer's shapes by 60, but **10 layers are `full_attention`** with
`global_head_dim 512` / 4 KV heads, so their q_proj and o_proj inputs are both
**16384-wide** — 200.26 MB/layer, not 179.61 MB.

**Corrected weight stream: 12.039 GB/token.** Ceiling at the 203 GB/s measured
peak: **17.2 → 16.9 tok/s**. This moves 20 tok/s *further* out of reach.

Traffic partition at key_count 700: **weights 94.27%**, KV cache 4.94%,
**element/scratch — the only fusion-addressable class — 0.80%**.

## T5.6 is misfiled as DONE, and that is why it was never attacked

Its title is "**Parallel multi-threadgroup** finalize". It only raised
threads-per-group 256 → 1024. **The grid is still 1 threadgroup** (`linear.rs:3065`,
`:3197`). So two per-layer dispatches run on **1 of 20 cores** while reducing 43 KB
of f32 accumulator through 10 + 18 serial `threadgroup_barrier`s on `partial[1024]`.

**This is the only fusion item with a ms-scale number: 0.4–0.9 ms/step.**
Requires a two-pass, order-preserving 42-group reduction — high risk, because the
256-thread summation order must be reproduced exactly.

## T5.3 is blocked by LAYOUT, not difficulty — and is not bit-exact

In every live GEMV each lane reads `x + tk*16 + r0` with `r0 = (lane%4)*2` — **4
halves per lane per k-tile, with lanes 4–31 reading the same address.** A thread
cannot see the 128-element block its elements belong to, so it cannot butterfly
them. Recovering it needs threadgroup staging plus a `threadgroup_barrier` *inside*
the K loop — the exact shape T4.3/T4.4 spent three tasks buying back. **Close it.**

## Bit-exactness, quantified (the number that should scare us)

Measured, not assumed: a fused kernel that **skips the fp16 store** changes the f32
operand for **5375/5376** activations and the resulting fp16 logit for
**208/512** trials. A kernel that **reassociates the norm reduction** changes
**5376/5376** normed elements. That is the actual price of a careless fusion, and
it is why every candidate above is gated on reproducing an exact order rather than
merely being algebraically equivalent.

## New minor defect

**59 empty command buffers per step** under `KAPLUMBA_DECODE_PROFILE=kernel`:
`norm_transform` fires with zero dispatches behind it on 59 of 60 layers, and
`commit_stamped` (`encoder.rs:674`) has no empty-buffer guard. Invisible in the 546.
A two-line guard.

---

# THE TWO THINGS THAT MATTER MOST

## 1. 🔴 Every tok/s figure in this repo — including mine — is the wrong path

`DEFAULT_TEMPERATURE = 1.0` (`crates/cli/src/args.rs:30`). At temperature > 0 the
engine **is not pipelined** and samples on the host (`chat.rs:388` gates on
`greedy`, which is only set at temperature 0).

| | ms/step | tok/s |
|---|---|---|
| pipelined greedy (temperature 0) — **what I benchmarked all session** | 62.30 | 16.05 |
| + barrier instead of 2-deep pipeline (`chat.rs:368-370`) | +3.83 | |
| + host `Sampler::sample` (`sampler.rs:46`) | +2.87 | |
| + fp16→f32 widen and `Vec<f32>` (`generate.rs:326`) | +0.45 | |
| + 512 KiB logits readback (`model.rs:3849`) | +0.15 | |
| **temperature 1.0 — the shipped default** | **69.60** | **14.37 (−10.5%)** |

Cross-checked against the one historical end-to-end figure (12.2 vs 15.9 tok/s) —
the two routes agree to **0.85 ms**.

**And the metric lies about it:** `generate.rs:383` computes
`decode_tokens_per_second` from a bracket that **excludes the sampler**, so
`--metrics` at temperature 1.0 **overstates by ~0.7 tok/s**. The honest number at
the default temperature is closer to **13.7**.

The argmax is already device-side (`argmax_slots`, `argmax_keys_to_token`, inside
`forward_chunk`). The **sampler is not** — `sampler.rs` imports only `thiserror`,
and `elements.metal`'s 25 kernels contain no top-k, no softmax, no RNG. Step N+1
cannot be submitted during that window because its embedding reads a token the host
has not produced. **A device-side sampler returns temperature>0 to parity with
greedy (16.05), not beyond it — that is the single highest-value change found, and
it is worth ~2 tok/s on the path the user actually runs.**

## 2. 🔴 203 GB/s is an ALU wall, not a bandwidth wall

The decisive experiment already exists in the tree (`linear.metal` arms d, verified
in `crates/kernels/probe/contig4_load_alu.metal:852`):

| arm | effect |
|---|---|
| remove **every** weight load (`tn -> tn & 3`, footprint 2.6 GB → 129 KB, L2-resident) | **−0.8%** |
| memory half of that | **~1%** |
| **needed: 203 → 236.7 GB/s** | **+16.6%** |

**A layout with ZERO DRAM cost buys ~1%.** The memory system is already hidden.
The wall is `decode_state_mcg`'s **64-bit variable barrel shift** `uint(A >> (64u - rad))`
— 65.6% of the kernel. Layout, contiguity, TLB, occupancy and threadgroup width
are all measured dead ends, and all for the same reason: none of them touch that
shift.

Corroborated independently by P03's mechanism: the head and the layers have
**identical bytes-in-flight per threadgroup** (3,072 B either way — the 2col variant
was *designed* to match it), so the layers must issue exactly **2× the decode ops
per byte**. GB/s ∝ 1/bpw. That is arithmetic, not a measurement.

## Also settled

- **KAPLUMBA_DECODE_NR0 has zero live effect.** 24 generated kernels, one knob,
  and `ishizuki_gemv_pipeline` returns first for all 410 3-bpw tensors. The `_nod`
  probes are **unreachable at any cost** — no env flag and no Rust source names them.
- **The "8.1 → 9.2 tok/s codebook specialisation" was measured on kernels that do
  not run.** It came from `cb_spec_bench.rs` dispatching the **EXL3** arm by name;
  all 8 ishizuki kernels already carried `constexpr int CB = 1` at that commit.
  That win is not available.
- **T5 (multi-queue) REFUTED.** The per-token graph is a **chain of 600**; nothing
  inside or across a layer is independent. `concurrent_stream_economics.rs:382`
  already asserts overlap is **>1.2× worse** than serial. Prize: 0.66 ms.
- **Threadgroup size: leave at 256.** A 120-cell interleaved sweep on this exact
  box and shape gave −5.1%..+1.9%; 256 is the best cell on 7 of 10 shapes. The whole
  barrier budget is **0.30% of a token**. Barriers are closed arithmetically.
- **The `_w2` two-word window is safe by exactly zero margin** (`used_lo = sh_max = 27`).
  Re-derived independently in Rust and Python. **Any change to the ring geometry
  silently breaks 410 of 411 tensors.**

## Final position

| | |
|---|---|
| measured, temperature 0 | 16.05 tok/s |
| measured, temperature 1.0 (the default) | **14.37**, metric overstates it |
| achievable with device-side sampler + w2 trim + o_proj 1col-SK2 | **~16.8** (temp 0) / **~15.1** (temp 1.0) |
| 20 tok/s | needs 236.7 GB/s **and** ~660 Gcodes/s; wall is ~340–540 Gcodes/s and ~203 GB/s |

---

# POSTSCRIPT — two findings that reframe the above

## 🔴 My "flat Gcodes/s" claim is WRONG, and the correction points at the real bottleneck

Measured rate sweep (`spec/decode-3bpw-amplification/implementation_notes.md:70-72`):

| rate | GB/s | Gcodes/s |
|---|---|---|
| 3 bpw | 199.4 | **531** |
| 4 bpw | 249.0 | 498 |
| 6 bpw | 251.4 | **335** |

**Gcodes/s is NOT constant across bit-width.** 4 and 6 bpw sit at the ~250 GB/s
streaming ceiling — **DRAM-limited**. Only 3 bpw is below it, because at 3 bpw the
kernel is **integer-issue bound**: 9.375 instructions per code, at **~92% of a
1-instruction-per-lane-per-cycle machine**. So at 3 bpw the wall is ALU, and at
4/6 bpw it is DRAM. Both are ~200-250 GB/s. **There is no rate at which this box
exceeds ~250 GB/s.**

Root cause of the 2-load minimum: `decode_state_mcg` multiplies **before** it
masks, so each lane's 16-bit decode window spans **37 bits** — and 37 > 32, the
register width. One lane can never hold its window in one register ⇒ 2 loads
minimum ⇒ amplification 2.667 at 3 bpw and 2.000 at 4/6 bpw, which is *exactly*
why the higher rates are faster in GB/s. Two codes per shift is impossible: codes
are 3 bits apart with 13 of 16 bits overlapping, and `P_hi = ⌊X·C/2¹⁶⌋ mod 2¹⁶`
depends on all 32 bits of X.

The largest reducible pool — `decode_state_mcg`'s 16 integer ops per 8 codes,
**21% of the loop** — is **bit-locked**: the mask applies to the *product*, so it
cannot fold into the extraction or the constant. Narrowing the window is quality
loss, barred by AGENTS.md §8.

**Removing every instruction this angle can legally reach (the extraction ANDs,
the v6/v7 join, the x-load redundancy — 19% of the loop) lands at 18.6 tok/s.**
Still below 20.

## 🔴 The actual biggest lever: a 1.6× gap between microbenchmark and production

> "Production runs the same kernel at **125–150 GB/s** while the isolated
> microbenchmark of that kernel hits **203**. That 1.6× is worth ~+30% decode and
> is **10× larger than any instruction win**."

`gemv_bandwidth.rs`'s absolute numbers are sound but its **variant rankings are
inverted** relative to production — and several agents independently refused to
trust microbenchmark rankings for kernel selection. A 203 → ~300 GB/s jump would
put the whole model at the DRAM ceiling and reach ~21 tok/s at 3 bpw.

**Nobody has explained why the identical kernel is 1.6× slower in the engine than
in the microbenchmark.** That is now the single highest-value open question, and
it is worth more than every instruction-level proposal combined.

## M03: storage modes are a clean negative, with the mechanism now measured

Storage mode does **not** change DRAM behaviour on Apple Silicon — no copy is
avoided (both are mappings), Metal exposes no cache-policy or prefetch knob, and
`StorageModePrivate` is *host-inaccessible*, not *compressor-exempt*.

What *does* control the class is the **VM page**, measured directly:
```
file-backed  VSIZE=67108864  RSDNT=67108864  DIRTY=0       SM=COW
anonymous    VSIZE=67108864  RSDNT=67108864  DIRTY=67108864 SM=COW
```
`PROT_READ` makes a write impossible, so `MAP_PRIVATE`'s CoW can never trigger —
mapped tensors are **permanently external** (reclaimable, never compressed).

Compression, measured rather than argued:
- trellis entropy **7.9979–7.9999 bits/byte** (higher than the 7.9702 on record)
- macOS's own compressor **gain ratio 1.0124** — it made the data 1.2% *larger*
- cost: **0.31% of decode bytes** on attempts, 0.85% on decompressions

**So compression cannot explain 193 vs 253 GB/s.** And `MAP_POPULATE` does not
exist on Darwin (`ENOMEM`); `MADV_POPULATE_WRITE` is `EINVAL`; `MAP_NOCACHE` is
accepted and would be **catastrophic** — `lm_head.trellis` read uncached from SSD
is 48 ms/step on top of a 62 ms step.

**The one real lever remains the conjunction**: map the layer trellis (takes 10.2
GiB out of the compressible class) **and `mlock` it** (stops eviction). Effect on
bandwidth: **none** — 193 → 193. What it buys is that today's working set is
*all or nothing*: fully resident (193 GB/s) or evicted (**1.4 tok/s**). Pinning
collapses that variance to "always resident".