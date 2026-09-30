# Tasks: sustain 240 GB/s in decode

For `spec/decode-bandwidth-240gb/spec.md`. Read the spec first — in particular
the **warning at the top about which kernel the numbers are about**, because
`encode_gemv` dispatches `linear_gemv_ishizuki_kN` first and the whole EXL3
family runs only for `lm_head`.

**No task below may start until its instrument can fail.** That is not a
formality: every bandwidth harness in this project was wrong before it was
verified, and one of them was the source of a number that was believed for a
day.

**Status: T2 done (clean negative). T1's fixed term re-measured at 59 us, not
79. The mechanism behind both is now identified and is the subject of T6.**

## T1 — Decompose the 59 us fixed per-matvec latency — *partly done*

*The fixed term is now 59 us, not the 79 us this file previously recorded, and it
is worth 24.8 ms of a 96 ms token. The streaming asymptote alongside it is
164.6 GB/s at 3 bpw.*

- [x] **Measure the fixed term as an intercept, not as a single-shape
  subtraction.** Three sizes, least squares, residual under 4% of the span:
  `t = 58.9 us + 0.00608 ms/MB`, residual 0.0%. `gemv_shape.rs --selftest`, the
  AFFINE check. The old "one matvec per command buffer, idle verified" method
  measured a CPU/GPU round trip, and required "2x bytes = 2x time" fails at
  1.62x on a real fixed cost — which is how a correct instrument gets declared
  broken.
- [ ] **Split the remainder.** First-touch page walk vs threadgroup launch vs
  the `tk` loop prologue vs the tail drain. The signature to preserve: the term
  is **identical for in_features 5376 vs 21504 and for out_features 5376 vs
  21504**, so it does not scale with work. Any candidate that scales with either
  is not the fixed term.
- **The obvious fixes are already dead** — more tiles per thread (T2, and the
  pre-existing `tk` unroll x2/x4), deeper unroll, more threadgroups
  (`KAPLUMBA_GEMV_MAX_GROUPS` makes every shape worse), `uint4` (0.939x), and the
  32-bit window rewrite (+2.4% at KA=3, -3.7% at KA=1). A new idea must not be a
  restatement of one of these.
- **This is now a question about `linear_gemv_ishizuki_k*`**, not the EXL3
  family. The decomposition has to be run on the kernel that carries 92% of the
  bytes; doing it on `_mcg` characterises a kernel that runs for `lm_head`.
- **Deliverable:** the split in microseconds, and the single change most likely
  to recover the largest share.

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

## T6 — Make a lane's window word-aligned at KA=3 — *the actual work*

*Aim at the mechanism, not beside it. This is the only live lever that reduces
the requests-per-useful-byte.*

- **The measurement that motivates it** (`gemv_shape.rs --only "bit rate"`, on
  the **live** ishizuki kernel, `KAPLUMBA_TRACE_GEMV` confirming): identical
  43.4 MB of weights, rate the only variable —

  | rate | GB/s | lane-loads per useful byte |
  |---|---|---|
  | 3 bpw | 132.5 | 12 B -> 3 B = **4.0x** |
  | 4 bpw | 179.8 | 3.0x |
  | 6 bpw | 164.5 | 2.0x |
  | 8 bpw | 176.3 | 1.5x |

- **The cause.** `exl3_decode8_mcg` (and ishizuki's inline copy) reads a fixed
  **three ring words — 12 bytes — per lane, whatever `KA` is**, and 8 codes at
  `KA` bits are only `KA` bytes. The window is sector-perfect, so DRAM traffic is
  correct; the cost is **requests**. 3 bpw is **410 of 411 tensors**.
- **Why every existing cure misses it**: `uint4` (0.939x — the fast path fires
  for 8/32 lanes because of the ring wrap), the 32-bit window rewrite (+2.4% at
  KA=3, -3.7% at KA=1), and `N_R0` (+6.8% best) all add *streams* or widen
  *loads*. None reduces requests per useful byte.
- **The idea.** At KA=3 a lane needs 24 bits for its 8 codes, so **one 32-bit
  load could serve 8 codes** instead of three — if the window were word-aligned.
  It is not: lane `l`'s codes start at bit `(l*8+1)*KA + ring - 16`, which is
  exactly why three words are read. The fix is in the `r0`/`i` index arithmetic
  that decides which codes a lane owns.
- **The hard part, and the reason this is not a two-line change:** changing which
  codes a lane owns changes the *order* of the accumulation, so bit-exactness has
  to be re-earned rather than assumed. `tests/decode_nr0.rs` is the harness for
  that comparison and `tests/decode_matvec_golden.rs` the reference.
- **Interleaved A/B, median of n>=7, and a clean negative is an acceptable
  result** — it would mean the EXL3 layout's 4x amplification is structural, and
  that the only remaining route to 19 tok/s is fewer bytes (T4).
- **Deliverable:** the measured amplification with and without, at 3 bpw, on
  `gate_proj` and `down_proj`, and end-to-end tok/s if it wins.

## Done when

T1's split is reported, T6 has a number, and every dead lever is in the spec's
closed table with its ratio. If a lever works, it is landed, bit-exact, with
20/20-or-better suites, the five reference prompts exact, and measured
end-to-end.
