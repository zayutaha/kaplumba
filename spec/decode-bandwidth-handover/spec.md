# Handover: sustaining decode bandwidth past 15 tok/s

**Read this first, then `spec/decode-bandwidth-240gb/spec.md`** (the ceiling, the
closed-lever table, the measurement discipline) and
`spec/decode-bandwidth-240gb/implementation_notes.md` (the full log, including
every instrument that was wrong before it was right).

This document exists so a fresh agent can be useful within one hour without
rediscovering anything below. It is written for someone who has never seen this
repository.

---

## 0. Where things stand, in one table

| | |
|---|---|
| **Model** | `~/.kaplumba/models/Gxma/Gxma` (Gemma 4 31B-class, EXL3 3.0 bpw). The only local checkpoint. |
| **Decode, now** | **~11.2 tok/s** measured (89.5 ms GPU span per token), up from 9.7 tok/s before this work. |
| **Decode, at 240 GB/s** | ~19 tok/s. **19 is the ceiling; 20 needs 263 GB/s and is not physically available.** |
| **Measured DRAM ceiling** | 239-244 GB/s (88% of 273 nominal). |
| **The one live lever** | requests-per-useful-byte in the decode matvec. T6 took 7% off it; **T7 is what remains.** |
| **Biggest open risk** | **decode is nondeterministic** — 8 runs of a 64-token prompt give 3 distinct outputs. A correctness bug, not a performance one. |
| **Suites** | 116 engine + 9 kernel tests green. |
| **Last commit** | `be4c614` (T1 answered). T6 shipped in `3230f24`. |

---

## 1. The thing that will waste your day if you do not know it

**`LinearKernels::encode_gemv` dispatches `linear_gemv_ishizuki_kN` FIRST**, and
the `linear_gemv_exl3_k*` family only when `packed_offset % 4 != 0`.

410 of this checkpoint's 411 tensors are 4-byte aligned, so **every projection in
all 60 layers runs on ishizuki**. The whole `_mcg` family — including the codebook
specialisation of `3439d41` and the NR0 axis — runs for `lm_head` alone, which is
8.9% of a decode token's bytes.

**Every EXL3-family bandwidth number in this repository's history is a number
about `lm_head`.**

Verify, do not believe:

```bash
KAPLUMBA_TRACE_GEMV=1 ./target/release/kaplumba --model ~/.kaplumba/models/Gxma/Gxma \
    --prompt "Hi" --max-new-tokens 3 --temperature 0 2>&1 | grep GEMV
# GEMV-DISPATCH linear_gemv_ishizuki_k3_w2
# GEMV-DISPATCH linear_gemv_exl3_k6_u_mcg
```

`./target/release/examples/which_gemv` prints the decision per (rate, alignment)
without loading the model. `KAPLUMBA_DEBUG_OFFSETS=1` prints each tensor's
alignment.

---

## 2. What was just landed, and what it cost to land

**T6 — the third ring word is dead at 3 bpw.** `linear_gemv_ishizuki_k3` loaded
**three** 32-bit ring words per lane per tile. The extraction only touches A0 bits
`[48-7*KA, 63]`; the third word contributes only to bits `[0, sh)`; at KA=3 the
largest `sh` over 32 lanes is 27 and the lowest used bit is 27. So the third load
was **never read**. Deleted.

- **+7.0% end to end** (10.07 → 10.78 tok/s median, 5 interleaved A/B pairs),
  bit-exact, five reference prompts unchanged.
- Enabled by default. `KAPLUMBA_ISHIZUKI_3WORD=1` restores the original.
- `scripts/gen_exl3_gemv.py` derives the safe rate set in `two_word_is_safe()`;
  **never hand-edit the generated block in `crates/kernels/src/linear.metal`.**
- Tests: `crates/engine/tests/decode_matvec_golden.rs` (golden over BOTH kernel
  families, and the two-word/three-word bit-identity check).

**T1 — the fixed-cost question is closed and the answer is "not a lever".** The
spec's 79 us, then 59 us, were both fitting artifacts. Measured: **~155 us per
command buffer, 42 buffers per decode step** = 6.5 ms/token of *host* time, which
is 29% of the host wall and **7.3% of the GPU span**. The host runs 4.01x ahead
of the GPU, so it is invisible. Do not reopen this.

---

## 3. Your first task: T7

Full text in `spec/decode-bandwidth-240gb/tasks.md`. In short:

**T6 took 7% off a 1.36x gap, and the mechanism story is now incomplete.**

Pre-T6, at *equal weight bytes*, varying only the bit rate:

```
3 bpw 132.5   4 bpw 179.8   6 bpw 164.5   8 bpw 176.3 GB/s
```

Post-T6, 3 bpw is ~166. 6 bpw is unchanged. So request count explains roughly
half the gap and **not all of it**. Because the window is sector-perfect, word 3
was an L1 *request*, not a DRAM sector — and the sector-rate experiment says the
wall is DRAM. So there is a residual mechanism nobody has identified.

**Step 1, and it is a prerequisite for the rest** — re-measure the table:

```bash
./target/release/examples/gemv_shape --only "bit rate"
```

Add a `--w2` arm at 3 and 4 bpw so the comparison is like-for-like. If 3 bpw
lands near 6 bpw, the gap *was* request count and T7 is closed. If it lands well
short, the residual is real and the "12 B -> KA bytes" story is not the whole
mechanism.

**Step 2 — the three candidates, in the order worth testing:**

1. **Register pressure.** At KA=3 the unrolled loop holds `v[8]` for two tiles
   plus 8 accumulators. `NR0=8` losing **20%** on `down_proj` is direct evidence
   this axis bites. Probe: a KA-parameterised unroll depth.
2. **Shifter pressure.** `(A0 >> (64 - r))` is a *variable* shift per code, 8
   per tile. At KA=3 the compiler may emit 8 shifts where a byte-offset extract
   would do. **But check `spec/engine-performance` first** — `extract_bits` was
   already tried there and found inapplicable. Do not re-report it as new.
3. **Decode cost does not scale with KA.** 8 `decode_state` calls per tile
   regardless of `KA`, so decode is amortised over fewer bits at low KA. This
   predicts a *compute* term. The uop/byte knee (16.6) says we are 8x under it —
   but that knee was measured for **streaming**, not for this kernel, so it does
   not clear this hypothesis.

**A clean negative is an acceptable and useful result.** It would mean the EXL3
layout's per-lane amplification is structural at low KA, and the only route to
19 tok/s is fewer bytes — which is T4, a user decision about quantisation, not an
implementation.

---

## 4. The measurement discipline, and why it is not optional

Every bandwidth harness in this project was wrong before it was verified. Six
figures were believed and acted on: 55, 131,512, 248, 254, and 220 GB/s, each
with a specific cause (an unwritten buffer, seconds labelled as milliseconds,
min-of-passes, a stand-in shape, a different probe shape, a wrong CPU reference).
Full table in the spec.

**Rules that are not negotiable:**

1. **No result without a fail-first instrument.** Sabotage the thing, watch the
   test fail, revert. If you cannot make it fail, it has no teeth and you do not
   have a result.
2. **Median, not best.** Report median **and** best **and** worst **and** spread,
   n>=7, **interleaved** A/B with the baseline re-measured every round. This box
   has been measured at 3.7, 3.9, 4.3, 4.8, 6.0 and 6.4 tok/s from the *same
   binary* minutes apart.
3. **Check units.** Three separate instruments in this session printed confident
   wrong numbers from a units error: seconds as ms, MB over a multi-pass total,
   and a per-dispatch time still in ms. `gemv_shape` now has a plausibility gate;
   use it and extend it.
4. **Know which kernel you are measuring.** See §1.
5. **One GPU-touching agent at a time.** `scripts/gpuq status` before every model
   run; `scripts/gpuq try --` is the default. A model load works at ~6.7 GiB
   "available" with 0 swap, so do not wait for the 14 GiB headline. Two
   concurrent instances destroyed the numbers once already.
6. **Interleaved A/B, free memory per configuration**, and warm the GPU to steady
   state before measuring.

**Two traps that cost real time this session:**

- **`cargo build` silently did not rebuild** after a source edit — "Finished in
  0.11s", no recompile, so a diagnostic `eprintln` never entered the binary and I
  spent a while debugging code that was not running. `touch` the source and
  `lib.rs` when a print "does not fire".
- **The live decode path is `generate_gpu`, and `generate_gpu_greedy` is the
  legacy `--legacy-path`.** Code instrumented in the wrong one reports zero and
  looks unwired. The encoder lives inside `model.rs::forward_buffer`.

---

## 5. The correctness bug you should probably fix first

**Decode is nondeterministic.** 8 runs of a 64-token prompt produce 3 distinct
outputs. The `rope` cross-warp race was fixed in `9bd62a9`; the remaining source
is unidentified.

This matters more than any bandwidth work, for two reasons:

1. **It undermines every bit-exactness claim in this project**, including T6's.
   A kernel that is bit-exact *in isolation* is not the same as a model that
   reproduces.
2. It is a correctness bug in a shipped engine, not a performance one.

The instrumentation to find it already exists: `KAPLUMBA_DECODE_PROFILE=kernel`
gives a per-kernel table for a real decode step, and `KAPLUMBA_GPU_PROFILE=1`
gives GPU-side attribution. Run the same prompt N times and diff the kernel
tables — the first kernel whose time *and* output varies is the culprit. Note
that `KAPLUMBA_GPU_PROFILE` alone historically printed the **prefill's** table
for a decode run; that was fixed in `a58fa7b` and there is a gate test for it.

A second, smaller gap: **no test covers the input-transform dispatch the engine
actually uses.** `encode_input_transform` in `model.rs` is not what the suites
drive (they use `gemv_buffer_impl`). There are **four** dispatch sites for the
transform/finalize pair and they must stay in lockstep — factor the thread count
into one helper, as was done for `rope_threads`. A 22/22-green change once
shipped a broken model because of exactly this.

---

## 6. Everything already closed — do not re-open

From the spec's table: concurrency (flat 0.08M-2.62M loads in flight), load
width (8/16/32 B all ~227-238), independent address streams (1→64: no effect),
working-set size, DRAM row locality (4 KiB→3 MiB: **zero effect** — weight layout
is not a target), wave tail, per-dispatch encoder overhead (0.08%), `N_R0`
(+6.8% best, clean negative), `tk` unroll x2/x4, `uint4` window loads (0.939x),
32-bit window rewrite (+2.4% at KA=3, -3.7% at KA=1), page faults, memory
compression, power/thermal (not throttled), batching, `MAX_DISPATCHES_PER_BUFFER`
(0.4%).

**Still untested, not disproved:** `MTLResidencySet` pinning (T3) — the API
SIGSEGVs on this binding, three times. It would pin pages we already have at no
memory cost.

---

## 7. If you only do one thing

**Make `gemv_shape --only "bit rate"` trustworthy.** It is the table everyone
reaches for, and it currently cannot answer the question it exists to answer.

Before and after T6 it gives:

```
rate  has _w2?    pre-T6    post-T6    delta
3 bpw   YES        132.5      146.9     +10.9%
4 bpw   YES        179.8      153.7     -14.5%
6 bpw   no         164.5      182.0     +10.6%
8 bpw   no         176.3      162.3      -7.9%
```

**6 and 8 bpw cannot have changed from T6** — no `_w2` kernel is emitted above
4 bpw — so they are controls, and they moved +10.6% and -7.9%. Within-sweep
spreads are 2.2x-3.7x. The box drifts by >=10% between sweeps, so an instrument
with that much drift cannot resolve T6's ~7-10% effect in either direction.

The fix is mechanical: the `--w2` and `--nr0` sweeps in the same binary already
interleave correctly (every round measures every arm; the baseline is re-measured
each round so drift shows up as a moving *ratio*). The rate sweep measures one
shape to completion at a time, which is why it cannot see drift. Reuse
`w2_sweep`'s structure with the **rate** as the arm, and add a 6 bpw control in
the same round.

Everything else in this document is context for that number.
