# Tasks: sustain 240 GB/s in decode

For `spec/decode-bandwidth-240gb/spec.md`. Read the spec first — in particular
the **warning at the top about which kernel the numbers are about**, because
`encode_gemv` dispatches `linear_gemv_ishizuki_kN` first and the whole EXL3
family runs only for `lm_head`.

**No task below may start until its instrument can fail.** That is not a
formality: every bandwidth harness in this project was wrong before it was
verified, and one of them was the source of a number that was believed for a
day.

**Status: T2 done (clean negative). T6 done and shipped (+7.0% end to end,
bit-exact). T1 answered, and its answer is that the fixed cost is NOT a lever.
Remaining: T3, T4, T5's correctness items, and T6's follow-ups.**

**The single most important thing to read first: `encode_gemv` dispatches
`linear_gemv_ishizuki_kN`, not the `_mcg` EXL3 family.** 410 of this
checkpoint's 411 tensors are 4-byte aligned, so all 60 layers' projections run
on ishizuki. `KAPLUMBA_TRACE_GEMV=1` prints the names on a real run. Any
EXL3-family bandwidth number in this repository's history is a number about
`lm_head`.

## T1 — Decompose the per-dispatch fixed cost — *DONE, and it is not a lever*

*The spec said 79 us, this session said 59 us, and the measured answer is
**~155 us per command buffer, 42 buffers per decode step**. All three were wrong,
and wrong in three different ways: the quantity (per-buffer, not per-matvec),
the magnitude, and the multiplier (42, not 420).*

- [x] **Expose the 59 us as a fitting artifact.** It came from a three-point
  affine fit of time against bytes with the pass count scaled as `1/bytes`, so a
  per-buffer cost is amortised differently at every size and lands in the
  *slope*. The tell: refitting today's numbers the same way gives **8.1 us** — a
  7x disagreement with the same instrument hours earlier, which means the method
  is not measuring the quantity at all.
- [x] **Measure it properly** — vary the *pass count* at fixed bytes, where the
  slope is per-dispatch work and the intercept is genuinely fixed
  (`gemv_shape --fixed`). Overhead **117-203 us**, flat across all five shapes
  (a page-walk term would scale with the buffer, a launch term would not).
  Residuals 2.6-15%.
- [x] **Get the multiplier right.** `KAPLUMBA_COUNT_BUFFERS=1` on a real 96-token
  run: `BUFFERS total=3990 steps=95 per_step=42.00`. 42 command buffers per
  decode step, from `MAX_DISPATCHES_PER_BUFFER = 50` against ~1260 dispatches.
  This confirms a code comment that already claimed "~42 rolled buffers" and had
  never measured it.
- [x] **Price it.** 42 x 155 us = **6.5 ms/token**: 29% of the 22.3 ms *host*
  wall, but only 7.3% of the 89.5 ms *GPU* span — and it is host time, spent
  while the GPU has 67 ms queued that the host is not waiting on. The host runs
  **4.01x ahead** of the GPU.
- **Conclusion: not a lever, and the decomposition is finished.** The cost is
  real and is most of the host's budget, and it is completely invisible in the
  token time because the GPU is the constraint. There is nothing left to split.
- **Do not reopen this by trying to reduce the buffer count.** Raising
  `MAX_DISPATCHES_PER_BUFFER` was already measured (50->1000, 0.4% spread) and it
  would move host time that is not on the critical path.

## T2 — Re-measure N_R0 = 4 on the CURRENT kernel — *DONE, clean negative*

*Measured +88% (101 -> 190 GB/s) on the PRE-specialisation kernel. On the
current kernel it is +6.8% at best and is not paid on `down_proj`.*

Medians of 7 **interleaved** rounds, `gemv_shape.rs --nr0`, with the baseline
re-measured in every round so a clock ramp shows up as a moving *ratio*:

| shape | NR0=1 | NR0=2 | NR0=4 | NR0=8 |
|---|---|---|---|---|
| gate_proj 5376x21504 | 129.4 | 137.0 | **137.3** | 132.0 |
| down_proj 21504x5376 | 123.4 | 125.9 | 117.1 | 98.6 |
| q_proj 5376x8192 | 104.6 | 109.5 | 105.4 | 95.0 |
| lm_head 5376x262144 @6bpw | 252.2 | 251.5 | 247.4 | 248.2 |

Reproduced: gate +6.3/+6.8%, q +5.3/+4.7%. **NR0=8 loses 20% on `down_proj`**,
consistent with `acc[NR0][2]` + `v[NR0][8]` register pressure.

- [x] Generated the whole `_r{2,4,8}` family from the template
  (`NR0_VALUES` in `scripts/gen_exl3_gemv.py`, mirrored by `NR0_VARIANTS` in
  `linear.rs`; the generator re-emits byte-identically, md5-checked).
- [x] The grid is derived from the kernel's **name**, not from
  `KAPLUMBA_DECODE_NR0`, so the two cannot disagree; a shape that does not divide
  evenly is an error, not a truncated dispatch.
- [x] **Bit-exactness, with teeth.** 192 comparisons (8 rates x 8 shapes x 3
  NR0). This found a real defect: with the accumulator written as
  `acc += x0*v0 + x1*v1 + ...` the compiler picks which product to contract into
  an FMA, and **its choice depends on register pressure, which NR0 changes** —
  `k8_mcg_r8` at 5376x21504 differed in 2 of 21504 outputs, one fp16 ulp, while
  every other pair was identical. The generator now emits an explicit `fma`
  chain. Verified bit-identical at 3 bpw and 6 bpw against the pre-NR0 kernel;
  8 bpw still moves by 1 ulp in 2 of 21504 and is unreachable from this
  checkpoint.
- [x] **And a real gap in the coverage claim.** Comparing variants *against each
  other* cannot see a change that moves them all together. Re-ordering the
  four-lane butterfly changes 79325 outputs and `decode_nr0.rs` passed it,
  because its baseline was rebuilt from the same edited source. Hence
  `tests/decode_matvec_golden.rs`, which holds the committed kernel's output as a
  reference digest for the 5 shapes and 2 rates the engine can reach.
- **Deliverable: a clean negative.** The lever is closed, with the ratio, and the
  last live geometric idea in the spec is now measured rather than assumed.

## T3 — Try `MTLResidencySet` pinning again

*Untested, not disproved. Would pin pages we already have at no memory cost.*

- `-[MTLDevice newResidencySetWithDescriptor:error:]` **SIGSEGVed three times**
  on this binding, `class!(MTLResidencySet)` returns a protocol (nil), and
  `-newResidencySetWithError:` on a buffer **aborts the process**.
- Try an extern declaration with the right signature, or the API by a different
  route, or a `madvise`-level equivalent (`MADV_WILLNEED`, `MADV_DONTNEED`).
- If it works, measure achieved GB/s with and without. llama.cpp refreshes a
  5 ms heartbeat for 180 s TTL (`:986-1019, 2042-2076`).
- **Deliverable:** either a working pinning path with a number, or a precise
  statement of why the API is unreachable on this binding.

## T4 — Only if T1-T2-T6 fail: measure the bytes-per-token routes

*Not an implementation task. A decision the user has to make.*

- At 240 GB/s, 12.088 GB/token is 50 ms = 20 tok/s. At 3.0 bpw that is the wall.
- oMLX runs **2.25 bpw** (oQ2 + gs64), ishizuki benchmarks at **2.0**. At 2.25
  bpw: 9.07 GB/token = 37.8 ms = **26 tok/s** of headroom.
- This is a **quantisation decision with a quality cost, not a port.** It cannot
  be done by an agent. Enumerate the options and their quality implications for
  the user to choose.
- Note what is NOT a bytes lever: the EXL3 rate is already exactly 3.000
  bits/value on 410 of 411 tensors (`lm_head` is 6.000), and no repacking reduces
  bytes. The 12.51 bits of real entropy per weight is entropy already paid for in
  3.

## T5 — Close the measurement gaps that have bitten us — *mostly done*

- [x] **The decode profiler measured nothing.** `KAPLUMBA_GPU_PROFILE=1` builds
  its `GpuProfile` inside `forward_buffer_into`, and the gate requires
  `KAPLUMBA_GPU_PROFILE_DECODE` for the one-token case, so a decode run printed
  the **prefill's** table. Every per-kernel decode number in this project's
  history is a prefill profile. Landed `KAPLUMBA_DECODE_PROFILE=step|kernel`
  (`a58fa7b`), validated in a worktree, plus the `spin_burn` injection that
  proves the instrument resolves a cost put inside it.
- [x] **That instrument had a bug the first use exposed**: warmup steps were
  folded into the per-kernel table but not into its denominator, over-reporting by
  47/39 = 1.205 and making the big three matvecs read as 136 GB/s when they are
  113. Fixed in `a58fa7b`.
- [x] **Which kernel runs is now answerable**, by `KAPLUMBA_TRACE_GEMV=1` and by
  `examples/which_gemv.rs`, instead of being a thing to be believed.
- [x] **`gemv_shape.rs` reports median with a self-test that has teeth** (AFFINE,
  SENSITIVE, CLOCK, PLAUSIBLE, MEDIAN). Its PLAUSIBLE check caught a real unit
  bug on the first run.
- [ ] **Decode is still nondeterministic**: 8 runs of a 64-token prompt give 3
  distinct outputs. The `rope` race is fixed; the remaining source is
  unidentified, and until it is, no bit-exactness claim is fully verifiable.
  **This is a correctness bug, not a performance one, and it is the most
  important open item in the spec.**
- [ ] No test covers the dispatch the **engine** uses for the input transform
  (`encode_input_transform` in `model.rs`); the suites drive `gemv_buffer_impl`.
  A 22/22-green change shipped a broken model because of exactly this. **Four**
  dispatch sites exist for the transform/finalize pair and they must stay in
  lockstep — factor the thread count into one helper, as was done for
  `rope_threads`.
- [ ] `crates/engine/tests/fwht_simdgroup.rs` was drafted and abandoned: the
  pipelines are private, so the two forms must be compared through a public entry
  point or a test-only accessor.
- [ ] `scripts/gen_exl3_gemv.py` remains the source of truth for 57 kernels and
  regenerates byte-identically (md5-checked). Edit the template, never the
  generated block.

## T6 — Cut the dead third ring word at KA=3 — *DONE and shipped*

*The idea as originally written (make the window word-aligned so 3 loads become 1)
turned out to be unnecessary: the third load is already dead at 3 bpw, so it can
simply be deleted. A derivation, not a guess — see below.*

- **The derivation.** The kernel extracts A0 bits `[48 - 7*KA, 63]`. The third
  word `w0c` contributes only to A0 bits `[0, sh)`. At KA=3 the largest `sh` over
  32 lanes is **27**, and the lowest used bit is `48 - 21 = 27`. So `w0c` is
  **never read** and the window is two words. `TWO_WORD_SAFE_RATES` in
  `scripts/gen_exl3_gemv.py` *derives* the safe set rather than listing it, so a
  rate cannot be added without the proof: emitted 1, 2, 3, 4 bpw; excluded 5 and
  6 (used bits below max `sh`) and 7, 8 (the `B0` arm is live).
- **Deliberately the word the spec predicted would break bit-exactness, and it
  does not.** The concern was that changing the window reorders the accumulation.
  Deleting a provably-unread load changes nothing, so the variant is
  bit-identical by construction — verified on **fp32** at 4 rates x 3 shapes
  (128x1024 runs only the tail loop, 1024x1024 only the unrolled loop, 5376x5376
  both). fp32 matters: a 1-ulp fp32 difference rounds away in fp16 on most
  columns, so an fp16 comparison would pass a genuinely wrong variant.
- [x] Generated from the verbatim ishizuki source by an **asserted
  transformation**, not re-derived — the variant must differ from the original
  *only* in the loads, or the A/B measures two changes at once. The diff is 6
  lines per loop; unroll, `r0` arithmetic, reduction and accumulation are
  byte-identical.
- [x] **Bit-exact end to end.** All five reference prompts byte-identical between
  arms: Paris / 1, 2, 3, 4, 5 / red, blue, and yellow / Hello! How can I help you
  today? / 17 times 23 is 391.
- [x] **Measured.** Microbenchmark +6.8% to +11.2% on five shapes, two sweeps,
  7 interleaved rounds each. End to end, 5 interleaved A/B pairs of 96 tokens:
  3-word median **10.07 tok/s**, 2-word median **10.78 tok/s** = **+7.0%**, with
  9 of 10 paired comparisons favouring 2-word.
- [x] **Shipped**, with `KAPLUMBA_ISHIZUKI_3WORD=1` to restore the original for
  attribution. Dispatch looks the `_w2` name up in the name map, so 6 bpw
  (`lm_head`, where the derivation says word 3 IS read) falls through to the
  3-word slot rather than breaking.

### Why +7% and not more — the part that matters for what comes next

3 -> 2 loads is a 33% cut in lane-loads and bought 7%, because the window is
sector-perfect: word 3 was an **L1 request, not a DRAM sector**, and the wall is
DRAM (sector rate flat at ~226 GB/s while useful rate halves with step size).
So the honest reading is that **request count is not free, but it is not the
wall either.** T6 is the first positive result on the axis that every earlier
lever (concurrency, load width, `N_R0`, `uint4`, unroll) missed, and it is also
evidence that the remaining 3-vs-6 bpw gap is *not* explained by request count
alone. See T7.

## T7 — Close the 3-vs-6 bpw gap that T6 did not close — *the next real work*

*T6 took 7% off a 1.36x gap. This asks what the rest of the gap is, and it is the
highest-value open question in the spec.*

- **The gap, restated with T6 in it.** At equal bytes the live kernel measured
  3 bpw 132.5, 4 bpw 179.8, 6 bpw 164.5, 8 bpw 176.3 GB/s before T6. After T6,
  3 bpw is ~166. The 6 bpw figure is unchanged, so **T6 closed roughly half the
  gap and the mechanism story is now incomplete.**
- **The measurement that discriminates.** `gemv_shape --only "bit rate"` at equal
  bytes, now with the 2-word kernel at 3/4 bpw. If 3 bpw lands near 6 bpw, the
  gap was request count. If it lands well short, the residual is something else
  and the "12 B -> KA bytes" story is not the whole mechanism.
- **Candidates for the residual**, in the order worth testing:
  1. **Register pressure.** At KA=3 the unrolled loop holds `v[8]` twice over
     two tiles plus 8 accumulators. `NR0=8` losing 20% on `down_proj` is direct
     evidence that this axis bites. A KA-parameterised unroll depth is the
     obvious probe.
  2. **Shifter pressure.** The extraction is `(A0 >> (64 - r))` with `r`
     varying per unrolled index — a variable shift per code, 8 per tile. At KA=3
     the compiler may be emitting 8 shifts where a byte-offset extract would do.
     A `extract_bits`-style helper is the probe. (Note: `extract_bits` was
     already found *inapplicable* in `spec/engine-performance` — check that note
     before spending time, and do not re-report it as new.)
  3. **Decode cost scaling.** 8 codes at KA bits means 8 `decode_state` calls per
     tile regardless of KA, so decode is amortised over fewer bits at low KA.
     This predicts a *compute* term, and the uop/byte knee (16.6) says we are 8x
     under it — but that knee was measured for streaming, not for this kernel.
- **A clean negative is an acceptable and useful result.** It would mean the
  EXL3 layout's per-lane amplification is structural at low KA, and the only
  route to 19 tok/s is fewer bytes (T4).
- **Deliverable:** the equal-bytes rate table post-T6, and the single largest
  identified residual term with a number attached.

## Done when

T7 has a number for the residual mechanism, and every dead lever is in the spec's
closed table with its ratio. If a lever works, it is landed, bit-exact, with all
suites green, the five reference prompts exact, and measured end-to-end.
