# Tasks: 1.18 -> 20+ tok/s

Ordered by certainty, not by payoff. Stages 1 and 2 are certain bugs worth 4.4x
together; do them before spending anything on the speculative stages.

## Stage 1 — the certain wins

- [ ] **T1** Time `online_attention` in `elem_bench` at the real decode shape
  (`queries=1, keys=200, q_heads=8, kv_heads=4, dim=512`, plus the sliding
  variant at `keys=1024`). Sizes stage 2 and answers Q3. Currently inferred
  from code structure, never measured.
- [ ] **T2** `KAPLUMBA_ATTENTION_ONLY=1` attribution switch: run a valid forward
  with norms and linears intact but attention replaced by a no-op, so the
  attention share is measured in situ rather than inferred.
- [ ] **T3** Parallel `rms_norm_weighted`: one 256-thread threadgroup per row,
  strided partial sums, barrier tree reduction. Select only when `rows < 32`.
  *Gate: under 20 µs at `rows=1`.*
- [ ] **T4** Same for `rms_norm_unweighted` (`pre_ffn_norm`).
- [ ] **T5** Correctness for T3/T4 vs the current kernel at `rows` in
  {1, 2, 7, 64, 134} and widths {128, 5376}, float tolerance.
- [ ] **T6** Re-measure the layer stack. *Gate: 783 → under 600 ms.*
- [ ] **T7** Two-pass split-K `online_attention`: per-split `(max, l, acc)`,
  then combine with `exp(m_i - m_global)`. No fp32 atomics. Keep the serial
  kernel when `queries * q_heads >= 1024` so prefill is untouched.
- [ ] **T8** Remove the per-thread `float accum[512]` for small grids; stage in
  threadgroup memory.
- [ ] **T9** Attention correctness at `queries` {1, 8}, `keys` {1, 200, 1024},
  GQA ratios {1, 2, 8}, full and sliding windows.
- [ ] **T10** Re-measure. *Gate: layer stack under 250 ms, decode ≥ 3.4 tok/s.*
- [ ] **T11** Prefill unchanged (8.4 tok/s at 134 tokens). Full suite green.
  `output_identity` within tolerance.

## Stage 2 — the matvec

- [ ] **T12** Fix the hanging `cb_bench` A/B and get the LUT number. It stalled
  before producing a result; suspect the CPU-side fill of a 1 GB buffer, same
  failure mode as the streaming probe that had to be deleted. **Q1 gates R4.**
- [ ] **T13** `half2` codebook packing: two codes per lane from adjacent lanes,
  targeting the measured 5.10e12 ops/s instead of half that. *Q5.*
- [ ] **T14** Cooperative ring load in `linear_gemv_exl3_k*`: the 32 lanes of a
  simdgroup collectively need one `256 * ka`-bit ring, so load it once per
  group (ideally one 128-bit load per thread) rather than three overlapping
  32-bit loads per lane. Targets the 55% load share. *Gate: `q_proj` ≥ 100 GB/s.*
- [ ] **T15** Re-measure decode. *Gate: ≥ 5 tok/s.*

## Stage 3 — past the codebook wall

- [ ] **T16** Codebook LUT if T12 says it pays: 256 entries built per threadgroup
  from the existing hash, indexed by `code >> (16 - ka)`, reproducing the hash
  bit-for-bit for every rate and codebook. If T12 says it does not pay, try a
  `constant`-address-space table in a small buffer instead, and if that also
  fails, drop the idea and record why.
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
- [ ] **T21** Update `implementation_notes.md` after every stage with the measured
  before/after, so the next session does not re-derive any of this.
