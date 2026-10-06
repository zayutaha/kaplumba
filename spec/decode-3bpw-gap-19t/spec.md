# Spec: the 3 bpw gap — 194 GB/s where the head does 254.6

## 1. Goal

Close the per-kernel bandwidth gap between the **3 bpw layer GEMVs** (measured
**194 GB/s** aggregate) and the **6 bpw `lm_head`** (measured **254.6 GB/s** in
the same decode step, same box, same buffers), and bank the throughput.

**Why this is the whole remaining opportunity.** From
`KAPLUMBA_DECODE_PROFILE=kernel` on `Gxma-repacked`, 29-token prompt,
temperature 0 (`own_ms` per token, bytes from the safetensors header count):

| kernel | GB/token | own_ms | **GB/s** |
|---|---:|---:|---:|
| `gate_up_gelu` | 5.2022 | 25.609 | 203.1 |
| `down_proj` | 2.6011 | 13.090 | 198.7 |
| `qkv_proj` | 1.2386 | 8.651 | **143.2** |
| `qk_proj` | 0.3716 | 1.961 | 189.5 |
| `o_proj` | 1.1561 | 6.246 | 185.1 |
| `head/gemv` | 1.0570 | 4.152 | **254.6** |
| `attention` | (KV) | 2.163 | — |

Layer bytes 11.001 GB, non-layer 8.26 ms (head 4.15 + attention 2.16 +
elementwise 1.95). Measured step 63.29 ms = 15.80 tok/s.

| layers at | step | tok/s | vs today |
|---|---:|---:|---:|
| 194 GB/s (today) | 63.29 ms | 15.80 | — |
| 203 GB/s | 62.46 ms | 16.01 | +0.21 |
| 220 GB/s | 58.27 ms | 17.16 | +1.36 |
| 245 GB/s (measured sustained streaming) | 53.17 ms | 18.81 | +3.01 |
| 254.6 GB/s (head's own rate) | 51.47 ms | 19.43 | **+3.63** |

Tier 1 (`qkv_proj` 143 → 194, the shortest K on the live path) is worth **+0.59
tok/s alone** and is the cheapest real win in the repo.

**254.6 GB/s is a demonstrated rate for a *different* kernel, not a proven
ceiling for these shapes.** Tiers 1–2 are solid against rates already measured
on this box; Tier 3 could land short. That distinction is the point of running
the discriminator first.

## 2. Non-goals & restrictions

* **No speculative decoding / MTP / drafter / lookahead / n-gram.** Every token
  executes the full 60-layer forward.
* **No lookup tricks**, no caching of generated text.
* **No KV cache quantization** (`AGENTS.md §8`). At 16k, KV is 2.181 GB and is
  immovable; every roofline here must carry it.
* **No quality loss.** Greedy output byte-identical. Any change to reduction
  order is a 1-ULP class needing an explicit waiver (`spec.md §2` of
  `decode-alu-fuse-18t`, already signed for attention splits only).
* **No bit-rate change** to the shipped checkpoint. 3.000 bpw ×410 tensors,
  head 6.000, both exact to 0 B.
* **No measurement on a contended box.** `crux.md §2.7` records a **factor of
  two** in the headline bandwidth from background load that left no trace in
  `ps`, `top`, or the load average. Every rate in this spec is `gpuq`-gated
  with `scripts/gpuq status` showing idle first.

## 3. The premise this spec overturns

`spec/decode-alu-fuse-18t` inherited and repeated this claim:

> "the GEMV is ALU-bound, ~8:1 against memory"; "3 bpw is integer-issue bound
> at 9.375 instructions per code"; "the largest reducible pool is bit-locked".

**It is false, and it was false because of a broken instrument.**
`alu_boost`'s `p2_int*` arms measured `latency × chains`, not issue capability
(a dependent `mul`→`xor` pair), which is why they scaled as n² and why every
spec recorded the integer pipe as unmeasurable.

Repaired (`spec/decode-alu-fuse-18t` `bef63f8`): two separate unrolled loops
over one register array, plus a linearity gate that prints
`int linearity: 4->8 1.02x 8->16 1.01x 16->32 1.00x`.

| pipe | measured issue ceiling |
|---|---:|
| integer | **91.0 G warp-instr/s** |
| fp32 | 118.7 G warp-instr/s |

At 9.375 instr/code the layer GEMVs execute **169 G warp-instr/s = 186% of the
integer pipe ceiling**, which is impossible for an issue-bound kernel. So the
"bit-locked ALU pool" is not the binding constraint, and the ~30% micro-vs-
production bandwidth gap is kernel-local and open.

## 4. Requirements

* **R1 — Run the existing, never-executed discriminator.**
  `crates/engine/tests/perf_probe_head_vs_layers.rs` already contains every
  experiment needed and **has never been run** (`grep -rn bw_ratio spec/` →
  empty; same failure class as `alu_boost.rs`, `AGENTS.md §7`).
  * **Control R** (`head_shape_at_three_bpw:1098`): head shape, head kernel
    family, head grid, row-major layout, at **3 bpw**. One variable. The
    pre-registered falsification criterion is in the test: `(bw_ratio - 0.5).abs()
    < 0.10`, asserted at `:1137`.
  * **Control S** (`layer_shape_at_six_bpw:1152`): `down_proj`'s shape at
    **6 bpw** via the existing `linear_gemv_ishizuki_k6_colmajor_contig4_bench`
    (`linear_repacked.metal:3815`).
  * **Control G** (`head_grid_thickness_sweep`): head shape at 6 bpw with the
    grid forced 16384 → 336 groups.
  * `KAPLUMBA_TEST_MODEL` must point at the checkpoint. Model **files** are read;
    no second model is *loaded*.
* **R2 — Split the three candidates.** R1's controls separate (a) parallelism,
  (b) contiguity, (c) K, (d) bits-per-code. Whichever survives is T3's target;
  the other three are recorded dead with numbers.
* **R3 — Tier 1: `qkv_proj` at K=5376.** 143.2 GB/s is the worst shape and the
  shortest K on the live path. Close it to the 194 GB/s pack. Worth +0.59
  tok/s, self-contained, bit-exact.
* **R4 — Tier 2: `o_proj` at K=8192** (185.1 GB/s) to pack, same class. Combined
  with R3: +0.67 tok/s.
* **R5 — Tier 3 only if R2 says tile geometry.** Port the head's 192 B/tile to
  3 bpw (two adjacent 3 bpw tiles per fetch). Bit-exactness must be proven, not
  argued from the absence of a reassociation flag.
* **R6 — Measurement discipline.** ≥16 interleaved pairs, median + spread +
  control drift, `CONTROL_repeat`, CPU-computed checksum (`crux.md §6`),
  `phys_footprint` via `/usr/bin/time -l` and never `peak_resident_pages_bytes`
  (`AGENTS.md §10.3`: that field understates by 5.5 GiB). Sabotage every
  coverage claim per `AGENTS.md §3.4`.
* **R7 — Record nulls.** Every dead end gets its number, per `AGENTS.md §7`.

## 5. Acceptance criteria

* **AC1** R1's three controls executed and reported on a quiet box, with the
  pass/fail verdict each prints. A control that cannot fail does not count.
* **AC2** The 194-vs-254.6 gap attributed to one of (a)/(b)/(c)/(d) with a
  measured number per elimination.
* **AC3** R3 lands: `qkv_proj` ≥ 194 GB/s, greedy byte-identical, or is
  recorded dead with the reason.
* **AC4** ≥17.2 tok/s greedy median (58.14 ms/step) on the AC1 prompt, or the
  honest number with the method that produced it.
* **AC5** `repacked_golden`, `fused_bit_exact`, `live_kernel_bit_exact`,
  `rate4_colmajor_bit_exact` all green; sabotage evidence recorded.
* **AC6** `phys_footprint` ≤ 16.5 GiB, 0 swap.
* **AC7** Negative-results table covering at minimum: occupancy, split-K, 4col,
  `uint4`, `simd_shuffle`, o_proj↔attention overlap, dispatch fusion — all
  previously measured, all re-confirmed dead or carried forward explicitly.

## 6. Open questions

* **Q1** Does Control R's `bw_ratio` land near 0.5? If yes, bits-per-code is
  the whole story and R5 is the work. If it lands near 1.0, the head's advantage
  is shape, and Tier 3 is unreachable without changing the layer layout.
* **Q2** Why did three specs cite `where_this_stands` numbers (125–150 GB/s)
  that `crux.md` later showed were ablation artefacts? Settled as "artefact",
  recorded here so it is not re-derived.
* **Q3** `qkv_proj`'s K=5376 is the input width, shared with `gate_up`. Is its
  deficit the short K, or is it a *fused* q+k dispatch paying twice for the
  activation read? `gate_up` at the same K gets 203.1, so the K alone does not
  explain it — this is the sharpest open question in the spec.

## 7. Supersession

Continues `spec/decode-alu-fuse-18t` (T1–T9 landed: T2 unroll bit-exact with no
e2e delta, T5/T6 device sampler null speedup, T9 `scratch.attended` overrun
fixed). Supplies the int-pipe number that spec's premise lacked
(`bef63f8`). Reuses `spec/decode-ceiling/findings/P17` and the `125–150 GB/s`
artefact record. `crux.md` supplies the byte counts and the bandwidth ceiling.