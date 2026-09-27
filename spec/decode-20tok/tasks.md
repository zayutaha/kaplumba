# Tasks: 1.18 -> 20+ tok/s

Ordered by certainty, not by payoff. Stages 1 and 2 are certain bugs worth ~3x
together; do them before spending anything on the speculative stages.

## Stage 1 — the certain wins

- [x] **T1** Time `online_attention` in `elem_bench` at the real decode shapes:
  - Sliding: `queries=1, keys=1024, q_heads=32, kv_heads=16, dim=256`
  - Full: `queries=1, keys=200, q_heads=32, kv_heads=4, dim=512`
  Sizes stage 2 and answers Q3. Currently inferred from code structure, never
  measured.
  - Files: `crates/engine/examples/elem_bench.rs`
- [ ] **T2** `KAPLUMBA_ATTENTION_ONLY=1` attribution switch: run a valid forward
  with norms and linears intact but attention replaced by a no-op, so the
  attention share is measured in situ rather than inferred.
  - Note: this env var does not exist yet; T2 creates it.
  - Files: `crates/engine/src/model.rs`
- [x] **T3** Parallel `rms_norm_weighted`: one 256-thread threadgroup per row,
  strided partial sums, barrier tree reduction. Select only when `rows < 32`.
  *Gate: under 20 µs at `rows=1, width=5376`.*
  - Files: `crates/kernels/src/elements.metal`, `crates/kernels/src/elements.rs`
- [x] **T4** Same for `rms_norm_unweighted` (used for v_norm, not pre_ffn_norm
  as previously thought; `pre_ffn_norm` uses the weighted variant).
  - Files: `crates/kernels/src/elements.metal`, `crates/kernels/src/elements.rs`
- [x] **T5** Correctness for T3/T4 vs the current kernel at `rows` in
  {1, 2, 7, 32, 64, 134} and widths {128, 256, 512, 5376}, float tolerance.
  Include both weighted and unweighted variants.
  - Files: `crates/engine/tests/elements_metal.rs`
- [ ] **T6** Re-measure the layer stack. *Gate: 783 → under 650 ms.* (Adjusted
  from 600 ms; the corrected norm count saves ~143 ms, not ~179 ms.)
- [ ] **T7** Two-pass split-K `online_attention`: per-split `(max, l, acc)`,
  then combine with `exp(m_i - m_global)`. No fp32 atomics. Keep the serial
  kernel when `queries * q_heads >= 1024` so prefill is untouched. The decode
  dispatch is 32 threads (1 query × 32 q_heads), not 8.
  - Files: `crates/kernels/src/attention.metal`, `crates/kernels/src/attention.rs`
- [ ] **T8** Remove the per-thread `float accum[512]` for small grids; stage in
  threadgroup memory. At decode, 32 threads × 2 KB = 64 KB of registers just
  for the accumulators.
  - Files: `crates/kernels/src/attention.metal`
- [ ] **T9** Attention correctness at `queries` {1, 8}, `keys` {1, 200, 1024},
  GQA ratios {1, 2, 8}, full and sliding windows. Test both the serial kernel
  (prefill shapes) and the split-K kernel (decode shapes).
  - Files: `crates/engine/tests/attention_metal.rs`
- [ ] **T10** Re-measure. *Gate: layer stack under 300 ms, decode ≥ 3.0 tok/s.*
  (Adjusted from 250 ms / 3.4 tok/s to account for corrected norm savings.)
- [x] **T11** Prefill unchanged (8.4 tok/s at 134 tokens). Full suite green.
  Create an output comparison test (the spec references `output_identity` but
  no such test exists; build it as a generation comparison against a recorded
  baseline at temperature 0).
  - Files: `crates/engine/tests/generation.rs` or new test file.

## Stage 2 — the matvec

- [ ] **T12** Fix the hanging `cb_bench` A/B and get the LUT number. It stalled
  before producing a result; suspect the CPU-side fill of a 1 GB buffer, same
  failure mode as the streaming probe that had to be deleted. **Q1 gates R4.**
  - Files: `crates/engine/examples/cb_bench.rs`
- [ ] **T13** `half2` codebook packing: two codes per lane from adjacent lanes,
  targeting the measured 5.10e12 ops/s instead of half that. *Q5.*
  - Files: `crates/kernels/src/linear.metal`
- [ ] **T14** Cooperative ring load in `linear_gemv_exl3_k*`: the 32 lanes of a
  simdgroup collectively need one `256 * ka`-bit ring, so load it once per
  group (ideally one 128-bit load per thread) rather than three overlapping
  32-bit loads per lane. Targets the 55% load share. *Gate: `q_proj` ≥ 100 GB/s.*
  - Files: `crates/kernels/src/linear.metal`, `scripts/gen_exl3_gemv.py`
- [ ] **T15** Re-measure decode. *Gate: ≥ 5 tok/s.*

## Stage 3 — past the codebook wall

- [ ] **T16** Codebook LUT if T12 says it pays: 256 entries built per threadgroup
  from the existing hash, indexed by `code >> (16 - ka)`, reproducing the hash
  bit-for-bit for every rate and codebook. If T12 says it does not pay, try a
  `constant`-address-space table in a small buffer instead, and if that also
  fails, drop the idea and record why.
  - Files: `crates/kernels/src/linear.metal`, `scripts/gen_exl3_gemv.py`
- [ ] **T17** Re-measure. *Gate: ≥ 11 tok/s.*
- [ ] **T18** Decide honestly against Q2. If we are short of 20, say so and
  record the measured ceiling rather than continuing to chase it.
- [ ] **T19** Only if T18 leaves real headroom: design note on a repacked trellis
  layout so a simdgroup reads contiguous bytes (Q7). Requires rewriting the
  checkpoint at load time; too large to start without evidence.

## Throughout

- [ ] **T20** Keep `elem_bench` covering every decode-path kernel (R7). A
  serialisation must show up as a microsecond number, not as an end-to-end
  regression discovered hours later.
  - Files: `crates/engine/examples/elem_bench.rs`
- [ ] **T21** Update `implementation_notes.md` after every stage with the measured
  before/after, so the next session does not re-derive any of this.

## Progress (2026-09-27)

- **T1 done.** Attention measured: 132.4 us sliding (keys=1024, 32q/16kv, dim 256),
  128.3 us full (keys=200, 32q/4kv, dim 512), so **7.9 ms per token**, not the
  ~380 ms both specs inferred. **Stage 2 (T7-T9) is therefore worthless and is
  struck**: 8 ms of an 850 ms token does not justify split-K attention, dropping
  `accum[512]`, or the register-pressure work.
- **T3-T5 done.** `rms_norm` at `rows == 1`: **595.7 us -> 11.9 us**, a 50x
  improvement, by giving each row a 256-thread group with a strided partial sum
  and a barrier tree. Dispatch switches on `rows < 32` so prefill keeps the
  per-row kernel. New test compares both paths at widths {128, 256, 512, 5376} x
  rows {1, 2, 7, 31, 32, 64, 134}: exact for most shapes, worst relative
  difference 6.64e-4 at width 5376, which is the reordered sum.
- **T11 done.** Suite green, prefill unchanged at 8.4 tok/s, and all five
  reference prompts produce byte-identical text.

Result: layer stack 766 -> 665 ms, decode **1.18 -> 1.51 tok/s**.

### A bug worth recording

The first attempt at this produced fluent garbage from every prompt, and the
threadgroup norm was *not* the cause: an earlier attribution refactor had
replaced the four hidden-width norm call sites with a helper that hardcoded
`rows = 1`. Decode has one row so it looked fine, but prefill passes
`rows = tokens` and was silently normalising only the first row. Forcing the
original per-row kernels did not restore the output either, which is what proved
the refactor rather than the new kernel was at fault. The helper now takes `rows`.

### What is left

665 ms of layer time is now almost entirely the linear matvec:

| | ms | note |
|---|---:|---|
| norms | 2.9 | was 156 |
| attention | 7.9 | measured, not worth optimising |
| matvec + element ops | ~654 | 11.00 GB at **16.8 GB/s** |

10 tok/s needs the matvec at ~90 ms, i.e. **122 GB/s, a 7.3x improvement**. The
same kernel measures 40-64 GB/s in isolation, so the gap is not arithmetic. It is
either non-coalesced access (Q7) or the loss of sustained streaming when seven
different projections are interleaved with ~16 element kernels, 60 times a token.
That is the next investigation and it is the whole remaining problem.
