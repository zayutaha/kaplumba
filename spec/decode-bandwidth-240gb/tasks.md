# Tasks: sustain 240 GB/s in decode

For `spec/decode-bandwidth-240gb/spec.md`. Read the spec first — it carries the
four artifact bandwidth figures, the closed-lever table, and the measurement
rules. Several of the most obvious ideas are already disproved, with numbers.

**No task below may start until its instrument can fail.** That is not a
formality: every bandwidth harness in this project was wrong before it was
verified, and one of them was the source of a number that was believed for a
day.

## T1 — Decompose the 79 us fixed per-matvec latency

*Highest value. 420 matvecs/token x 79 us = 33 ms of a 109 ms token.*

- Measure the fixed term directly: run the same shape twice back-to-back on the
  same buffer and see whether the second is cheaper. That separates **first-touch
  page walk** from **ramp** (launch, prologue, tail drain).
- Then split the remainder. Candidates: threadgroup launch, the `tk` loop
  prologue, the tail drain where fewer than 8 simdgroups still have work, and
  the per-dispatch pipeline-state bind.
- The signature to preserve: the term is **identical for in_features 5376 vs
  21504 and for out_features 5376 vs 21504**, so it does not scale with work.
  Any candidate that scales with either is not the fixed term.
- **The obvious fixes are already dead** — more tiles per thread (T=2 0.98x,
  T=4 0.87x, T=8 0.64x, register pressure in `v[T][8]`), deeper `tk` unroll
  (x2 1.008x, x4 0.889x), more threadgroups (`KAPLUMBA_GEMV_MAX_GROUPS` makes
  every shape worse), and `uint4` (0.939x, fast path fires for 8/32 lanes).
  A new idea must not be a restatement of one of these.
- **Deliverable:** the split in microseconds, and the single change most likely
  to recover the largest share.

## T2 — Re-measure N_R0 = 4 on the CURRENT kernel

*Measured +88% (101 -> 190 GB/s) on the PRE-specialisation kernel. Must be
re-measured before it is believed.*

- Give each thread 4 independent output rows, row pointers hoisted so all four
  issue before the first FMA, rows 32 KiB apart. This is llama.cpp's shape
  (`ggml-metal-impl.h:32-33`, `kernels/mul_mv.metal:250-255,285-287`).
- **This is a different axis from the dead multi-tile experiment.** That one
  multiplied tiles *within one row's k-loop* and lost to `v[T][8]` register
  pressure. This multiplies *rows*, which changes the accumulator count and the
  reduction. Do not conflate them.
- Peak was 4; NR0=8 lost 13% to register pressure. **Adjacent** rows lose 4-11%
  — only far rows win, so the row stride matters.
- Requires an accumulator change and a threadgroup reduction change
  (`red[8][16]`). Watch for FMA contraction: it is not stable, so two variants
  can each match alone and differ combined.
- **Bit-exactness mandatory.** Compare output bits. Same accumulation order per
  output column.
- **Deliverable:** NR0 in {1,2,4,8} on the current kernel, isolated GB/s and
  end-to-end tok/s. **A clean negative is a fine result** — it closes the last
  live geometric lever.

## T3 — Try `MTLResidencySet` pinning again

*Untested, not disproved. Would pin pages we already have at no memory cost.*

- `-[MTLDevice newResidencySetWithDescriptor:error:]`
  (`ggml-metal-device.m:2049`) **SIGSEGVed three times** on this binding, and
  `class!(MTLResidencySet)` returns a protocol (nil). `-newResidencySetWithError:`
  on a buffer **aborts the process**.
- Try: an extern declaration with the right signature, or the `MTLResidencySet`
  API by a different route, or a `madvise`-level equivalent
  (`MADV_WILLNEED`, `MADV_DONTNEED` discipline).
- If it works, measure achieved GB/s with and without. llama.cpp refreshes a
  5 ms heartbeat for 180 s TTL (`:986-1019, 2042-2076`) over
  `newBufferWithBytesNoCopy` / `StorageModeShared` buffers.
- **Deliverable:** either a working pinning path with a number, or a precise
  statement of why the API is unreachable on this binding.

## T4 — Only if T1-T2 fail: measure the bytes-per-token routes

*Not an implementation task. A decision the user has to make.*

- At 240 GB/s, 12.088 GB/token is 50 ms = 20 tok/s. At 3.0 bpw that is the wall.
- oMLX runs **2.25 bpw** (oQ2 + gs64), ishizuki benchmarks at **2.0**. At 2.25
  bpw: 9.07 GB/token = 37.8 ms = **26 tok/s** of headroom.
- This is a **quantisation decision with a quality cost, not a port.** It cannot
  be done by an agent. Enumerate the options and their quality implications for
  the user to choose.
- Note what is NOT a bytes lever: the EXL3 rate is already exactly 3.000
  bits/value on 410 of 411 tensors (96 B per 256-value tile), `lm_head` is
  6.000, and no repacking reduces bytes. The 12.51 bits of real entropy per
  weight is entropy already paid for in 3.

## T5 — Close the measurement gaps that have bitten us

- No test covers the dispatch the **engine** uses for the input transform
  (`encode_input_transform` in `model.rs`); the suites drive
  `gemv_buffer_impl`. A 22/22-green change shipped a broken model because of
  exactly this. **Four** dispatch sites exist for the transform/finalize pair
  and they must stay in lockstep — factor the thread count into one helper, as
  was done for `rope_threads`.
- `crates/engine/tests/fwht_simdgroup.rs` was drafted and abandoned: the
  pipelines are private, so the two forms must be compared through a public
  entry point or a test-only accessor.
- Decode is still **nondeterministic**: 8 runs of a 64-token prompt give 3
  distinct outputs. The `rope` race is fixed; the remaining source is
  unidentified, and until it is, no bit-exactness claim is verifiable.
- `scripts/gen_exl3_gemv.py` is the source of truth for 32 kernels and
  regenerates byte-identically (verified by md5). Edit the template, never the
  generated block.

## Done when

T1 and T2 have reported numbers. If a lever is dead, it is in the spec's
closed table with its ratio. If something works, it is landed, bit-exact, with
20/20-or-better suites, the five reference prompts exact, and measured
end-to-end.
