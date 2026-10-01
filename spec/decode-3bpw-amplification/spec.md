# SPECIFICATION: 3 bpw Lane-Load Amplification — 18 tok/s Decode

Supersedes the open items of `spec/decode-25tok-peak-bandwidth/`. That spec's task list
is known to be wrong; see §6.

---

## 1. Goal

Raise sustained pure-autoregressive single-token decode throughput from the measured
**15.3 tok/s (65.3 ms/step)** to **≥ 18.0 tok/s (≤ 55.5 ms/step)** on Apple M4 Pro
(20 GPU cores, 273 GB/s peak) for Gemma 4 31B (`Gxma`, EXL3 3.0 bpw), by reducing the
decode GEMV's **lane-load amplification at 3 bpw from 2.667 to ≤ 2.000 lane-loads per
32-bit word — at unchanged byte volume, unchanged bit rate, and bit-exact output.**

The measured basis for this target is in §2. It is not a bandwidth guess.

---

## 2. Measurement this spec is built on

`bench_rate_sweep --rates 3,4,6`, 9 interleaved rounds, medians, digest-verified
reproducible (33/33 digests identical across two independent runs):

| rate | words/tile | bytes/tile | aggregate GB/s | lane-load amp |
|---|---:|---:|---:|---:|
| 3 bpw | 24 | 96 | **199.4** | 2.667 |
| 4 bpw | 32 | 128 | **249.0** | 2.000 |
| 6 bpw | 48 | 192 | **251.4** | 2.000 |

Two facts follow, and they are the whole spec:

1. **The kernel is sound.** At amplification 2.000 it reaches 249–251 GB/s, which is
   this box's measured streaming ceiling. There is no kernel defect to find.
2. **The limiter is lane-load count, not arithmetic, and not DRAM bandwidth.** 4 bpw
   (128 B/tile) and 6 bpw (192 B/tile) carry 1.5× the bytes and ~1.5× the dequant ALU
   of 3 bpw, yet land within 1% of each other. If ALU or DRAM were the constraint they
   would separate. They do not. The variable that tracks bandwidth is amplification.

Therefore: **moving 3 bpw from amp 2.667 to amp 2.000 at the same 96 bytes/tile should
move it from 199.4 GB/s to the ~250 GB/s ceiling**, which is what 4 and 6 bpw already
demonstrate at that amplification.

### 2.1 The mechanism, derived from the kernel

Every live decode kernel shares one load pattern. Canonical form is
`linear_gemv_ishizuki_k3_colmajor_contig4`
(`crates/kernels/src/linear_repacked.metal:1782-1882`):

```
constexpr uint RING  = 256u * 3u;      // :1794   768 bits = 96 bytes = 24 u32 words
constexpr uint WORDS = RING / 32u;     // :1795   = 24
const uint j0 = ...; const uint j1 = ...;                       // :1808-1809
extract_v_mcg_32(t0[j0], t0[j1], sh0, rsh0, v);                  // :1836  <- 2 loads/lane/tile
```

Each lane needs **8 codes × 3 bits = 24 bits = 3 bytes**, but reads a **2-word (8-byte)
window** and extracts them via 64-bit barrel shifts at `r = i*3 + 16`
(`exl3.rs:678-682`). So per tile: 32 lanes × 2 loads = **64 lane-loads** for 24 words
⇒ amp **2.667**, and 256 bytes read for 96 bytes of tile.

**Which kernels actually run** (256 threads, 8 simdgroups, `linear.rs:1322-1323`).
Note `contig4` is *not* live — `colmajor_2col` wins the dispatch first because
`out % 32 == 0` always holds (`linear.rs:1963`):

| projection | K × N | live kernel | line | threadgroups |
|---|---|---|---|---:|
| gate_up (fused) | 5376 × 21504 | `..._k3_fused2_co` | `:580-715` | 1344 |
| down_proj | 21504 × 5376 | `..._k3_colmajor_2col` | `:1884-2018` | 168 |
| o_proj | 8192/16384 × 5376 | `..._k3_colmajor_2col` | `:1884-2018` | 168 |
| qkv (fused3) | 5376 × 8192/4096/4096 | `..._k3_fused3_2col` | `:1221-1384` | 512 |
| lm_head | 5376 × 262144 @ 6 bpw | `linear_gemv_exl3_k6_u_mcg` (EXL3 arm) | `linear.metal:1389` | 16384 |

All of the 3 bpw arms use the identical `t[j0], t[j1]` + `extract_v_mcg_32` pair
(`:1907-1912`, `:1946-1959`, `:1255-1260`, `:1312-1325`, `:606-609`, `:640-659`), so
**one layout change fixes all of them.** `lm_head` is 6 bpw and takes the EXL3 path
because its trellis sits at a `% 4 == 2` offset that `ishizuki_gemv_pipeline` refuses
(`linear.rs:1353`) — it is out of scope for a 3 bpw fix.

`dram_line_traffic_is_1x_at_every_rate_so_the_amplification_is_requests_only`
(`crates/engine/tests/rate4_colmajor_bit_exact.rs`) shows the 128-byte lines are shared,
so DRAM traffic is 1× — the waste is in **request count**, not bytes moved from DRAM.
That is why the fix must not add bytes.

### 2.2 Why ≤ 2.000 is reachable without adding bytes

32 lanes × 24 bits = 768 bits = exactly the 96-byte tile. So a **lossless within-tile
permutation placing lane L's 8 codes at bit offset 24·L** covers the tile with zero
padding. Word-straddling lanes then need 2 loads, the rest need 1:

| lane offset mod 32 | 0 | 24 | 16 | 8 |
|---|---:|---:|---:|---:|
| 24 bits fit in one word? | yes | **no** | **no** | yes (24+8=32) |

Two of every four lanes straddle ⇒ 16 lanes × 2 loads + 16 lanes × 1 load = **48
lane-loads per tile** ⇒ amp **48/24 = 2.000**, at **96 bytes/tile, unchanged**.

48 loads is the value 4 bpw already demonstrates is sufficient to saturate. This is a
hypothesis derived by hand from the constants above; **T1 exists to measure it before
any production code is written.**

---

## 3. Non-goals & restrictions

1. **No quality loss.** Greedy output must stay bit-identical to the reference
   ("The capital of France is Paris."). Any layout change must be a provably lossless
   bit permutation with a CPU round-trip proof.
2. **No change in bit rate or byte volume.** 3.0 bpw, 96 bytes/tile, ~16 GB resident.
   Padding each lane to 4 bytes would reach amp 1.333 but costs +33% bytes — that is
   precisely the 4 bpw configuration, and it is **measured 6.7% slower**. Forbidden.
3. **No speculative decoding, no multi-token parallelism, no draft models.**
4. **No KV cache quantization.** FP16/BF16 only.
5. **No Python in product or engine code.** The repack is Rust.
6. **The original checkpoint is untouched.** `~/.kaplumba/models/Gxma/Gxma` stays
   pristine; any new layout ships as a new directory.
7. **The withdrawn lossy repacker must not be revived.**
   `exl3.rs:687-726` (`repack_trellis_3bpw_tile`, `repack_trellis_3bpw_tensor`) is
   **lossy and dead** — its `imul` mixes 16 bits before masking, so 3 bits of input
   yield *different* dequant, not imprecise dequant. This spec writes a **new,
   different** permutation. It does not modify or re-enable that code.
8. **No new command queues / multi-queue overlap in this spec.** Out of scope; see §6.
9. **One model process at a time.** `scripts/gpuq` discipline per AGENTS.md §10.

---

## 4. Requirements

- **R1 — Amplification ≤ 2.000 at 3 bpw, at 96 bytes/tile.** The decode GEMV issues
  ≤ 48 lane-loads per 24-word tile. Enumerable from the kernel's own arithmetic, in
  the same style as the existing exact-rational test.
- **R2 — The repack is provably lossless.** A CPU reference computes the forward
  permutation and its inverse; a test asserts round-trip identity over all 256 codes of
  many real tiles, and asserts the dequantized values are bit-identical to
  `extract_ishizuki_lane_codes` (`exl3.rs:664`).
- **R3 — Bit-exact end to end.** Greedy generation over the golden prompt is
  bit-identical, and the existing suites `live_kernel_bit_exact`, `fused_bit_exact`,
  `repacked_golden`, `argmax_engine_parity` stay green.
- **R4 — Gated, reversible dispatch.** The new layout and kernel are selected by an
  explicit env flag. Default remains the current 3 bpw path. No silent behaviour change.
- **R5 — Sabotage evidence (AGENTS.md §3.4).** For every new code path: break it on
  purpose, run the test, observe a diagnosable failure, revert, record the observed
  failure mode. A coverage claim is not a claim until the failing branch has been seen
  to fail.
- **R6 — Measurement discipline.** A/B interleaved, medians with spread reported, on
  an idle box (`scripts/gpuq status` shows `gpu: none`), per AGENTS.md §10.2. Single
  readings are not results — this box has been measured at 3.7 to 6.4 tok/s from the
  same binary minutes apart.
- **R7 — Residency unchanged.** ≤ 16.5 GiB resident, 0 MiB swap, on the 24 GiB machine.

---

## 5. Acceptance criteria

- **AC1 — Throughput.** ≥ 18.0 tok/s sustained (≤ 55.5 ms/step), median of ≥ 8
  interleaved paired runs on a ≥ 128-token generation, reported with spread.
  *This is the gate. Bandwidth is reported but not gated.*
- **AC2 — Bit-exactness.** Golden prompt yields "The capital of France is Paris." with
  identical token IDs; zero fp16 divergence against the reference on every layer.
- **AC3 — No regressions.** The four named suites pass with 0 failures.
- **R1 demonstrated** by the amplification enumeration test, as an exact rational.
- **AC4 — Losslessness demonstrated** by CPU round-trip over real tiles, plus sabotage
  evidence in both directions.
- **AC5 — Memory.** Phys footprint ≤ 16.5 GiB, 0 MiB swap, measured with
  `/usr/bin/time -l`. `peak_resident_pages_bytes` is **not** a footprint (AGENTS.md §10.3).
- **AC6 — Reversibility.** With the flag off, the engine reproduces today's
  15.3 tok/s baseline bit-exactly.

---

## 6. Dead ends, formally closed

Recorded here so they are not re-attempted. Each was measured; none is a hypothesis.

| idea | verdict | evidence |
|---|---|---|
| **4 bpw / 8 bpw head** (the `Gxma-bpw4-shim` checkpoint, also in `~/Downloads`) | **−6.7%, slower.** Bytes +33%, bandwidth only +24.9% (199.4→249.0), so time ×1.067. 4 bpw is a worse format *and* 20 GB resident. | `bench_rate_sweep --rates 3,4,6`, 2026-10-01; 6 bpw likewise −6.1% (251.4 GB/s but ×1.5 bytes) |
| **Split-K for `down_proj`** (old T5.2) | **Refuted twice.** 120-cell sweep (K-split × threads/tg, 10 shapes, median of 11 rounds): default SK=1/TG=256 is best on 7 of 10 shapes; best alternative +0.6…+1.9%; SK≥4 is −2.6…−36.7%. Independently, `splitk_bit_exact.rs` is named `splitk_reassociates_and_is_not_bit_exact` — it breaks R1/AC2. | `spec/decode-19tok-repacked-250gb/findings/09-splitk-table.md:14-48` |
| **"168 threadgroups starves latency hiding"** (old T5.2's premise) | **Disproven by its own experiment.** Widening tiles to cut the grid 4× measured **−4.2%** (14.01 vs 14.62 tok/s). Occupancy is not the limit; a decode dispatch is ~84 threadgroups/core, oversubscribed ~20×. | `spec/decode-25tok-peak-bandwidth/tasks.md:30`, `implementation_notes.md:122-131`; `spec/decode-19tok/implementation_notes.md:88-92` |
| **Elementwise/norm fusion, "~3–4 ms"** (old T5.3) | **Actual ~0.8 ms.** A dispatch costs 3.5 µs of *latency* and ~0 of *work*; fusion collects only the launch. Measured: `norm_rope_qkv` 0.84, down-finalize 0.21, post-attn 0.42, attention single-pass 0.025. | `spec/decode-19tok-repacked-250gb/findings/07:29-34`, `implementation_notes.md:613-637` |
| **128-bit weight loads via threadgroup staging (`_u4`)** | **180.5 → 173.7 GB/s.** Note: that was *staging*. Direct wider loads are a different mechanism and remain untested — but at amp 2.667 they are not the lever; R1 is. | `implementation_notes.md:146-153` |
| **`simd_shuffle` cooperative loads** | **180.5 → 146.5 GB/s.** | `implementation_notes.md:64-67` |
| **Cheaper dequant (removing the decode)** | **−26.7%.** Removing 59% of ALU work made it *slower* — the arithmetic was hiding latency. | `implementation_notes.md:439-444` |
| **Megakernel** | Rejected by reference: Metal has no grid-wide barrier. "a megakernel buys nothing on Metal" (ishizuki, AGPL — idea only). | `spec/decode-20tok/reference_findings.md:77-79` |
| **Giant command buffers** | Null (`AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1`). Nothing is giant any more. | `spec/engine-performance/spec.md:180` |
| **Multi-queue attention/GEMV overlap** (old T5) | **Not attempted, deferred.** Never built, never measured, so no verdict. But oMLX — the closest working analogue — uses **no second stream at all** (`engine_core.py:521-528`: "to guarantee that MLX GPU operations are never concurrent"; `grep mx.new_stream` = 0 matches). No precedent to port. Real value is 1.2–2.3 ms, not the 3.2 ms the old plan assumed. **Deliberately out of scope for this spec.** | see §3.8 |

### 6.1 Corrections to the standing reference table (AGENTS.md §7)

- **`exllamav3` is a hollow directory — 0 files.** `exllamav3_gemm_inner.cuh` **does not
  exist**. It is listed in AGENTS.md as a citable source and is not one. Re-clone before
  relying on it.
- **`omlx/patches/qwen35_verify_qmm.py` is not a decode kernel.** `vk_eligible` is
  `3 <= M <= 6` (`:624`), `mma_eligible` is `7 <= M <= 24` (`:529`), and it explicitly
  falls through at M=1 (`:665`) — which is our decode case. Its no-barrier property comes
  from private per-simdgroup buffers (`:465`), not from a 32×32 tile.
- **oMLX has no EXL3/trellis decode kernel**; `affine_qmv_fast` is a different packing
  (`quantized.h:877-982`). Use it for *technique* only, never as a layout template.
- **ishizuki is AGPL-3.0.** `spec/engine-performance/spec.md` Q9 is still **open**;
  citing its ideas is safe, porting its code is not without the user closing Q9.
- Cite `reference/omlx`, not the scratch copies — the scratch `.../T/opencode/omlx`
  contains only `quantized.h`.

---

## 7. Open questions

None. Resolved with the user on 2026-10-01:

- **Repack scope** — both paths authorized, gated: kernel-only first, lossless repack
  authorized up front if the gating experiment (T1) shows the ring layout cannot
  reach ≤ 48 loads without byte growth.
- **Acceptance gate** — end-to-end tok/s (≥ 18.0), bandwidth reported not gated.
- **Location** — new spec folder; `decode-25tok-peak-bandwidth` left intact as history.
- **Dead ends** — recorded here (§6).
