# SPECIFICATION: Decode at the EXL3 3 bpw Floor — 16 tok/s

**Amends `74398f9`, whose stated premise was wrong.** The original title of this folder
was "3 bpw Lane-Load Amplification — 18 tok/s". The amplification question is now
**answered, and the answer is no.** The measurement is in §2 and the arithmetic that
kills it is in §3. The negative result stays here so it is never re-derived.

Supersedes the open items of `spec/decode-25tok-peak-bandwidth/`. Dead ends from that
spec are listed in §6.

---

## 1. Goal

Raise sustained pure-autoregressive single-token decode throughput from the measured
**15.3 tok/s (65.3 ms/step)** toward **16.0 tok/s**, on Apple M4 Pro (20 GPU cores) for
Gemma 4 31B (`Gxma`, EXL3 3.0 bpw), by recovering the **~2–4 ms** that the record
identifies as genuinely available, with bit-exact output and no change to the 3 bpw
format.

**This is a small target and it is stated as such.** The larger targets in this repo's
history were not reachable at 3 bpw; §2 and §3 are the measurements that show why. The
purpose of this spec is to bank the real gains and to record, permanently, why the
larger ones are not on the table.

---

## 2. What the rate sweep measured — 2026-10-01

The experiment built in `spec/decode-19tok-repacked-250gb/implementation_notes_rate4_appendix.md`
and blocked since 2026-09-30. Reproducibility verified first: **33/33 digests identical
across two independent runs.**

```
3 bpw:  199.4 GB/s   24 words/tile    96 B
4 bpw:  249.0 GB/s   32 words/tile   128 B
6 bpw:  251.4 GB/s   48 words/tile   192 B
        VERDICT 4/3 = 1.249x  (amplification model predicts 1.333x, residual 0.937)
```

Read naively this says "3 bpw is leaving 25% on the table and 4 bpw proves the hardware
can do it". That reading is wrong, and §3 is why.

## 3. Why 199 GB/s is the floor at 3 bpw

`decode_state_mcg` (`crates/kernels/src/linear.metal:121-126`):

```metal
value = value * 0xCBAC1FEDu;                   // multiply FIRST
value = 0x3b603b60u ^ (value & 0x8fff8fffu);   // mask SECOND
```

**The multiply precedes the mask, so all 16 bits of the input window affect the output.**
A lane cannot be reconstructed from its own 8 codes.

Enumerated three independent ways, all agreeing:

| method | result |
|---|---|
| bit-flip: flip each of the 768 ring bits, record which lanes' float outputs change | **37 bits per lane** (min=max=37) |
| symbolic: `a0` bits `[27, 63]` for all 8 windows, span **37**; undo `sh0`, lands across **2 words** for all 32 lanes | 2 words, unconditional |
| union of all 32 lanes' influencing positions | **37** — every lane reads the *same* 37-bit window, differing only in rotation |

37 > 32, so each lane must read **2 words**. 32 lanes × 2 = **64 word-loads per 24-word
tile = amp 2.667**, which is precisely what `linear_gemv_ishizuki_k3_colmajor_*` already
issues. The union result is what makes this a floor rather than an accident: the lanes
are not reading *different* data, they are all reading one 37-bit window at 32 different
rotations, and no relayout changes that.

Therefore amp is a function of the bit rate alone, `64 / (8·KA)`:

```
3 bpw: 64 / 24 = 2.667     4 bpw: 64 / 32 = 2.000     6 bpw: 96 / 48 = 2.000
```

4 and 6 bpw are faster **because more words share the same 64 loads**, not because the
kernel does anything better. At 3 bpw the kernel is already at its structural floor.

**Consequence:** the ~12 ms implied by 199→250 GB/s does not exist at 3 bpw. The
earlier draft of this spec proposed a lane-contiguous layout that would have reached
amp 2.000; §3 shows it cannot, because its 24-bits-per-lane premise ignored the
16-bit decode window. That proposal is withdrawn.

## 4. What is actually available

| item | expected | status |
|---|---:|---|
| Norm kernel **latency** (not fusion): 256 threads walk 5376 elements in 21 strided loads, 8 barriers, then a second pass. ~10.5 µs predicted vs 11.2 µs measured, at 4.6–133 GB/s. | 0 – 1.2 ms | untried, flagged "may be worth 0 — do not budget before measuring" |
| Hoist the norm second pass `weight`/`residual` loads into registers above the barrier | part of the above | untried, legal (first loop writes only `partial[]`) |
| Remaining norm fusion (`cache_append` → `norm_rope_qkv`; post-attn norm + pre-FFN + gate transform → 1 dispatch; cross-layer `input_layernorm` into `post_ffn_out`; down finalize + `post_ffn_out`) | ~0.21 + 0.42 + 0.21 + 0.21 = **~1.05 ms** | individually measured, each bit-exactness-safe |
| Attention/GEMV overlap on a second command queue | 0.6 – 1.5 ms | never attempted; **no reference precedent** (§6) |
| **Total** | **~1.7 – 3.8 ms → 15.6 – 16.2 tok/s** | |

Upper bound on the whole fusion direction, for calibration: a dispatch costs 3.5 µs and
1167 dispatches remain, so removing *every* dispatch would recover 4.1 ms. That is the
ceiling for an approach the record already measured at ~0.8 ms of actual delivery.

## 5. Non-goals & restrictions

1. **No quality loss.** Greedy output stays bit-identical ("The capital of France is
   Paris."). In particular **do not narrow `decode_state_mcg`'s 16-bit window to 3 bits** —
   it is the only route past 199 GB/s and it changes every output.
2. **No bit-rate change.** 3.0 bpw, ~16 GB resident. 4 bpw is a measured 6.7% loss
   (§6); 2 bpw is documented byte-rate-invariant at constant rate.
3. **No speculative decoding, no multi-token parallelism, no draft models.**
4. **No KV cache quantization.**
5. **No Python in product or engine code.**
6. **`~/.kaplumba/models/Gxma/Gxma` stays untouched.**
7. **Do not revive the lossy bit-plane repacker** `exl3.rs:687-726` (zero callers,
   withdrawn: its `imul` mixes 16 bits before masking, so 3 input bits yield *different*
   dequant). Barred by the no-quality-loss rule.
8. **One model process at a time**; `scripts/gpuq` discipline per AGENTS.md §10.
9. **Out of scope, permanently:** any attempt to reduce 3 bpw lane-load amplification
   by relayout. §3 is the disproof. If someone wants to revisit it, they must first
   explain how 37 bits fit in one 32-bit word.

## 6. Dead ends, formally closed

| idea | verdict | evidence |
|---|---|---|
| **Lane-contiguous repack to reach amp 2.000** (this spec's own withdrawn proposal) | **Impossible.** Each lane's 16-bit decode windows span 37 bits; the union over all 32 lanes is 37. Two words per lane is a floor. | §3; three independent derivations |
| **4 bpw / 8 bpw head** (`Gxma-bpw4-shim`, and the byte-identical copy in `~/Downloads`) | **−6.7%, slower.** Bytes ×1.333, bandwidth ×1.249. 6 bpw likewise worse. 20 GB resident. | `bench_rate_sweep --rates 3,4,6`, 2026-10-01 |
| **Split-K** (old T5.2) | **Refuted twice.** 120-cell sweep: SK=1/TG=256 best on 7 of 10 shapes; best alternative +0.6…+1.9%; SK≥4 is −2.6…−36.7%. Independently `splitk_bit_exact.rs` is named `splitk_reassociates_and_is_not_bit_exact` — breaks bit-exactness. | `findings/09-splitk-table.md:14-48` |
| **"168 threadgroups starves latency hiding"** (old T5.2's premise) | **Disproven by its own experiment.** Widening tiles to cut the grid 4× measured **−4.2%**. A decode dispatch is ~84 threadgroups/core, oversubscribed ~20×; occupancy is not the limit. | `decode-25tok-peak-bandwidth/tasks.md:30`; `decode-19tok/implementation_notes.md:88-92` |
| **Elementwise fusion, "~3–4 ms"** (old T5.3) | **Actual ~0.8 ms.** A dispatch costs 3.5 µs of latency and ~0 of work. | `findings/07:29-34`, `implementation_notes.md:613-637` |
| **`_u4` 128-bit loads** | **180.5 → 173.7.** And it is *staged* through threadgroup memory, not a direct wide load. | `implementation_notes.md:146-153` |
| **`simd_shuffle` cooperative loads** | **180.5 → 146.5.** This was the one idea that could have exploited the 37-bit overlap. Closed. | `implementation_notes.md:64-67` |
| **Cheaper dequant (removing the decode)** | **−26.7%.** Removing 59% of ALU work made it *slower*. | `implementation_notes.md:439-444` |
| **Megakernel** | Metal has no grid-wide barrier. "a megakernel buys nothing on Metal" (ishizuki, AGPL — idea only). | `decode-20tok/reference_findings.md:77-79` |
| **Giant command buffers** | Null. Nothing is giant any more. | `engine-performance/spec.md:180` |
| **Multi-queue attention/GEMV overlap** | **Never attempted — no verdict.** Retained as a real candidate at 0.6–1.5 ms, but note oMLX runs **no second stream at all** (`engine_core.py:521-528`; `grep mx.new_stream` = 0 matches). No precedent to port. | §4 |

### 6.1 Corrections to the standing reference table (AGENTS.md §7)

- **`exllamav3` is a hollow directory — 0 files.** `exllamav3_gemm_inner.cuh` does not
  exist. Listed in AGENTS.md as a citable source; it is not one until re-cloned.
- **`qwen35_verify_qmm.py` is not a decode kernel** — `vk_eligible` is `3 <= M <= 6`
  (`:624`) and it falls through at M=1 (`:665`), our decode case. AGENTS.md §7's
  description of it as "the 32×32 tile and no-barrier-in-K-loop shape" is wrong; the
  no-barrier property comes from private per-simdgroup buffers (`:465`).
- **`lm_head` is 6 bpw on the EXL3 arm**, not ishizuki: its trellis is at offset
  `% 4 == 2`, which `ishizuki_gemv_pipeline` refuses (`linear.rs:1353`). 16384
  threadgroups. Likely why it has always measured near the ceiling.
- **`contig4` is not the live decode kernel.** `colmajor_2col` wins first because
  `out % 32 == 0` always (`linear.rs:1963`).
- **ishizuki is AGPL-3.0**; `engine-performance/spec.md` Q9 is still open. Cite ideas,
  do not port code.
- Cite `reference/omlx`, not the scratch `.../T/opencode/omlx` (which holds only
  `quantized.h`).

## 7. Requirements

- **R1 — Bit-exactness.** Greedy output bit-identical; `live_kernel_bit_exact`,
  `fused_bit_exact`, `repacked_golden`, `argmax_engine_parity` all pass.
- **R2 — Measure before believing.** Every claim of a gain is an interleaved A/B with
  medians and spread on an idle box. This box has been measured at 3.7–6.4 tok/s from
  the same binary minutes apart; a single reading is not a result.
- **R3 — Sabotage evidence (AGENTS.md §3.4).** For every changed path: break it, run
  the test, observe a diagnosable failure, revert, record the observed message.
- **R4 — No regression.** Flag-off behaviour reproduces today's 15.3 tok/s bit-exactly.
- **R5 — Residency.** ≤ 16.5 GiB, 0 MiB swap, via `/usr/bin/time -l`
  `peak memory footprint`. Never `peak_resident_pages_bytes` (AGENTS.md §10.3).

## 8. Acceptance criteria

- **AC1 — Throughput.** ≥ **16.0 tok/s** (≤ 62.5 ms/step), median of ≥ 8 interleaved
  paired runs on a ≥ 128-token generation, spread reported, baseline re-measured in the
  same session. If a task delivers less than its own estimate, that is recorded, not
  rounded up.
- **AC2 — Bit-exact.** Golden prompt, identical token IDs, 0 fp16 divergence.
- **AC3 — No test regressions** (R1).
- **AC4 — Sabotage evidence** recorded for every touched path (R3).
- **AC5 — Memory** (R5).
- **AC6 — Negative results recorded**, including any task that fails to beat baseline.

## 9. Open questions

None. Resolved with the user on 2026-10-01: accept a ~16 tok/s target rather than
chase an unreachable one; correct the record rather than leave the withdrawn
amplification proposal standing.
