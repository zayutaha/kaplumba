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
- [x] **T7** Two-pass split-K `online_attention`: per-split `(max, l, acc)`,
  then combine with `exp(m_i - m_global)`. No fp32 atomics. Keep the serial
  kernel when `queries * q_heads >= 1024` so prefill is untouched. The decode
  dispatch is 32 threads (1 query × 32 q_heads), not 8.
  - Files: `crates/kernels/src/attention.metal`, `crates/kernels/src/attention.rs`
- [x] **T8** Remove the per-thread `float accum[512]` for small grids; stage in
  threadgroup memory. At decode, 32 threads × 2 KB = 64 KB of registers just
  for the accumulators.
  - Files: `crates/kernels/src/attention.metal`
- [x] **T9** Attention correctness at `queries` {1, 8}, `keys` {1, 200, 1024},
  GQA ratios {1, 2, 8}, full and sliding windows. Test both the serial kernel
  (prefill shapes) and the split-K kernel (decode shapes).
  - Files: `crates/engine/tests/attention_metal.rs`
- [x] **T10** Re-measure. *Gate: layer stack under 300 ms, decode ≥ 3.0 tok/s.*
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

## Split-key attention (2026-09-27)

**I was wrong to strike stage 2, and the reason matters.** The 7.9 ms figure came
from `elem_bench` with an 8.4 MB KV buffer, which is L2-resident. In situ the KV
cache is 1.5 GB and every read is a DRAM miss, and the kernel runs on 32 threads
(`queries * q_heads` with 32 query heads) each holding a 512-float accumulator.
Ablation with `KAPLUMBA_SKIP_ATTENTION=1` put the real cost at **153 ms**, a
19x gap against the isolated measurement. So stage 2 was the second-largest win
available all along, and I nearly deleted it on the strength of a benchmark that
was measuring cache.

This also explains the 2.3x matvec discrepancy that was left over from the
previous round. The old attention kernel held 2 KB of registers per thread and
ran 2.5 ms per layer; it was starving the matvecs queued behind it. With
attention fixed, the matvecs run at 229 ms against the 206 ms that
`layer_bench` predicts for 60 layers -- within 11%, where before they were 2.3x
off.

Implemented as three flash-decoding passes, chosen so no pass needs a per-thread
array proportional to `head_dim`:

  attention_scores    one thread per (head, key)   -- q_heads * key_count
  attention_row_reduce one thread per head
  attention_combine   one thread per (head, dim)   -- q_heads * head_dim

KV data is read once instead of once per thread. Selected only when
`queries == 1`; prefill keeps the serial kernel and is unchanged at 8.4 tok/s.

| | before | after |
|---|---:|---:|
| layer stack | 630 ms | **242 ms** |
| decode | 1.51 tok/s | **3.84 tok/s** |
| prefill | 8.4 tok/s | 8.4 tok/s |

Cumulative so far: 1.18 -> 3.84 tok/s, a 3.25x improvement, output byte-identical
on all five reference prompts.

## Where the remaining 242 ms is

| | ms | GB/s |
|---|---:|---:|
| attention | ~13 | |
| matvec (11.00 GB) | ~229 | **48** |

48 GB/s against `layer_bench`'s 50.7 GB/s means the matvec is now running at its
measured speed, so the isolation gap is closed and what remains is genuine matvec
efficiency. 10 tok/s needs 100 ms, i.e. ~130 GB/s, which is 2.7x from here. That
is stages 3-5 of this spec: cooperative ring load (the loads are 55% of the
kernel and do not coalesce, Q7), `half2` codebook packing, and the codebook LUT.

## ishizuki's matvec, ported verbatim (2026-09-27)

You were right that the format was not the ceiling. ishizuki's `exl3_matvec`
(EXL3.swift:286) is now ported line for line into `linear_gemv_ishizuki_k*` and
is the **default** for integer rates. `KAPLUMBA_OWN_GEMV=1` selects the old kernel.

| | GB/s |
|---|---:|
| our previous kernel | 50.9 |
| ishizuki, verbatim | **100.8** |

End to end: layer stack 242 -> 128 ms, decode **3.84 -> 6.77 tok/s**, prefill
unchanged at 8.4, all five reference prompts byte-identical, suite green.
Cumulative: 1.18 -> 6.77 tok/s, 5.7x.

### Three bugs I made while porting, all of which faked a result

Worth recording, because two of them produced *convincing* numbers:

1. **Unbound K and N.** My kernel read `K` and `N` from buffers 3 and 4, but the
   dispatch bound only three buffers, so `tiles_n` and `tiles_k` were garbage and
   the loop ran zero iterations. That measured **304 GB/s -- 6x "faster" than
   ours** while emitting pure noise. I reported that 6x before checking
   correctness. It was my bug, and it also exceeded DRAM peak, which was the tell.
2. **Swapped bindings.** The kernel signature was `(x, trellis, y)` while the
   dispatch provides `(packed, transformed, raw)`.
3. **The main loop was missing.** My template-extraction slice came out empty, so
   the kernel had no `for` loop and `acc0`/`acc1` were always zero.

Every one of these produced fast-and-wrong. The rule I should have applied from
the start: **an A/B that gets faster must be validated on output before the
number is believed.** Three of this project's benchmarks have now misled me --
clock ramp, L2 residency, and this -- and in all three the check that would have
caught it was cheap.

### What is actually different, now that both run correctly

Ours recomputes `tile_index * ring_words` and `input[xbase + r0 + N]` inside the
loop; theirs hoists `column` and `xr` as pointers so the addresses are affine in
`tk` and the compiler can strength-reduce them. Everything else -- one window per
eight codes, four `x` values for eight codes, 8-simdgroup striding, the
`simd_shuffle_xor` tree -- was already identical, and those changes account for
2.0x. Pointer hoisting and address strength reduction, nothing exotic.

### Where the remaining gap to 10 tok/s is

At 128 ms/token we need 100 ms. The matvec is ~115 ms of the 128. Their reported
28 tok/s is on a **2 bpw** model streaming 6.75 GB/token; ours is **3 bpw** at
12.06 GB/token. 3 bpw is ~1.5x more codes per byte, hence more ALU per byte, so
some of the residual gap is the rate and not the kernel. Scaling their 189 GB/s
(M1 Max) to our bandwidth predicts ~129 GB/s against the 100.8 we measure, so
roughly 1.3x remains and is plausibly reachable.
