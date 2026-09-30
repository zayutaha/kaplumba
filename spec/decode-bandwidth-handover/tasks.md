# Handover tasks: sustaining decode bandwidth past 15 tok/s

Companion to `spec/decode-bandwidth-handover/spec.md`. Read that first — it has
the state of play, the kernel-dispatch trap, and the measurement discipline.

Ordered by what a fresh agent should do first. **H1 is the whole point of this
document**; the rest is context.

---

## H1 — Fix the equal-bytes rate sweep, *then* use it

**This is the first task because the instrument cannot currently answer the
question, and finding that out is itself the finding.**

`gemv_shape --only "bit rate"` is the table everyone reaches for when asking "why
is 3 bpw slower than 6 bpw?". Run it before and after T6 and the numbers do not
support any conclusion:

```
rate  has _w2?    pre-T6    post-T6    delta
3 bpw   YES        132.5      146.9     +10.9%
4 bpw   YES        179.8      153.7     -14.5%
6 bpw   no         164.5      182.0     +10.6%
8 bpw   no         176.3      162.3      -7.9%
```

**6 and 8 bpw cannot have changed from T6** — `TWO_WORD_SAFE_RATES` emits no `_w2`
kernel above 4 bpw, so those two rows are **controls**. They moved +10.6% and
-7.9%. A control that moves means the measurement is not measuring the kernel.

The within-sweep spreads are the other half of the story: **2.2x to 3.7x**
worst-to-best, so the median of 9 is not sitting on a stable distribution.

So the box drifts by >=10% between sweeps and the sweep is internally
inconsistent, which means **this instrument cannot resolve a ~10% effect** — and
T6's effect is ~7-10%. Any conclusion drawn from this table, in either direction,
is unsupported.

- [ ] **Make the sweep interleaved.** The `--w2` and `--nr0` sweeps already do
  this correctly: every round measures every arm once, and the baseline is
  re-measured in every round so a clock ramp shows up as a *moving ratio* rather
  than a moving number. The rate sweep measures one shape at a time to
  completion, which is why it cannot see drift. **Reuse `w2_sweep`'s structure
  with the rate as the arm instead of the window.**

- [ ] **Add a control arm that cannot change.** A 6 bpw shape measured in the same
  interleaved round as the 3 bpw shape gives a drift estimate for free. If the
  control moves, the round is not trustworthy and should be reported as such
  rather than as a number.

- [ ] **Report the drift explicitly**, as a first-class column, not a footnote.
  The spec's own rule is that the *median* has to be clean and the spread has to be
  *visible*; here the spread is 2-4x, which is the signal.

- [ ] **Then** answer the actual question: with an instrument that can resolve
  10%, is 3 bpw now at 6 bpw, or still short?

  | outcome | meaning | next step |
  |---|---|---|
  | 3 bpw lands near 6 bpw | request count *was* the mechanism | T7 closed; go to T4 (fewer bytes) |
  | 3 bpw lands well short | residual mechanism, unidentified | T7 open; work H2 |
  | 3 bpw lands *above* 6 bpw | the "3 bpw is slow" framing was wrong | re-examine whether 6 bpw is the anomaly |

- [ ] **Commit the instrument fix before the number.** A better table with no
  conclusion is worth more than a conclusion from the current one, and this
  repository has six figures that were believed on exactly that trade.

**Do not skip the plausibility gate.** `gemv_shape` asserts the streaming rate
lands in 5-400 GB/s. It is there because this harness printed 0.0-0.2 GB/s for a
kernel every other instrument measures at 150+.

---

## H2 — Attack whatever H1 leaves open

Three candidates, in the order worth testing. Each is a *probe*, not a
commitment: a clean negative is a result.

### H2a — Register pressure (most likely)

`NR0=8` loses **20%** on `down_proj` (98.6 vs 123.4 GB/s), and the mechanism is
register pressure: `acc[NR0][2]` plus `v[NR0][8]` is 10·NR0 floats of accumulator
and 8·NR0 of decoded values per thread. At KA=3 the unrolled ishizuki loop holds
`v[8]` for two tiles plus 8 accumulators.

If low KA is register-bound, the unroll depth should be the lever, and it should
be **KA-parameterised** — the current unroll-by-2 is the same for every rate.

- Probe: an unroll-by-1 and unroll-by-3 variant of `linear_gemv_ishizuki_k3`,
  measured against unroll-by-2 on `gate_proj` and `down_proj`.
- If unroll-by-1 *wins* at KA=3, the register-pressure story is confirmed and
  the fix is free.

### H2b — Shifter pressure

The extraction is `(A0 >> (64 - r))` with `r` varying per unrolled index — a
**variable shift per code, 8 per tile**, inside an already-unrolled loop.

- **Check `spec/engine-performance` before starting.** `extract_bits` was tried
  there and found inapplicable; do not re-report that as new. What has *not* been
  tried is a KA-specialised extraction where the shift is a compile-time constant
  per unrolled index.
- Probe: unroll the `i` loop fully and make `r` a literal, so the compiler emits a
  constant shift.

### H2c — Decode cost does not scale with KA

8 `decode_state` calls per tile **regardless of `KA`**, so the decode work is
amortised over fewer bits at low KA. This predicts a *compute* term that vanishes
at high rates, which is consistent with the shape of the original table.

- The uop/byte knee (16.6 uop/B, we run 1.35) says we are 8x under it — but that
  knee was measured for **streaming**, not for this kernel, so **it does not
  clear this hypothesis**. Do not cite it as if it does.
- Probe: replace `decode_state` with a no-op in a variant kernel (output is then
  garbage — this is a bandwidth probe, not a correctness one) and measure the
  rate at 3 and 6 bpw. If removing decode lifts 3 bpw much more than 6 bpw, this
  is the residual.

**Order matters.** H2c is the cheapest and the cleanest discriminator, so run it
first even though H2a is the most likely to be the fix.

---

## H3 — Fix the decode nondeterminism (correctness, not performance)

**REPRODUCED, and localised: the trigger is PROMPT LENGTH, not generation
length.**

| probe | prompt | generated | result |
|---|---|---|---|
| A | 20 tok | 256 | 6/6 identical |
| B | 6 tok | 128 | 5/5 identical |
| C | **~1000 tok** | 64 | **2 distinct of 4** |

This outranks everything above on merit. It is a correctness bug in a shipped
engine, and it **undermines every bit-exactness claim in this project**,
including T6's.

- [x] **Reproduced** (above). The earlier "8 runs of a 64-token prompt give 3
  distinct outputs" did **not** reproduce on the current build — that prompt has
  a short context. Probe C, with ~1000 tokens of context, reproduces at ~50%.

- [x] **Discriminated, which is the useful part.** Generation length is not the
  trigger (A made 256 tokens deterministically). The greedy sampler is not the
  trigger (A and B both go through `argmax_logits`). **Prompt length is the only
  variable that separates clean B from dirty C**, so the divergence originates on
  the **prefill** path.

- [ ] **Suspect list, now narrowed to the prefill attention kernels**:
  `online_attention`, `online_attention_256`, `online_attention_512`,
  `online_attention_tg2_256`, `online_attention_tg2_512` in
  `crates/kernels/src/attention.metal`. These are the path a long prompt takes
  and a short one does not. The decode attention path and the entire
  `linear_gemv_ishizuki_*` family are common to A, B and C, so A and B being
  clean **exonerates** them — do not spend time on the matvecs.

  The flash-style kernels reduce across query blocks with an online-softmax
  rescale. The classic bugs there, in the order to check:
  1. **The online-softmax rescale race** — a `m`/`l` accumulator written by all
     lanes of a simdgroup while another lane reads the previous value. Needs the
     same discipline as the ishizuki `red[SG][16]` reuse: is there a barrier
     between the rescale and the read, and does it survive every early exit?
  2. **A missing `threadgroup_barrier` on a partial K-block**, where the last
     block takes a different branch than the loop body. The `tk`-style remainder
     loops are exactly where this hides, and they are the same shape as the
     ishizuki tile loop that T6 had to reason about.
  3. **The threadgroup size actually dispatched** vs the `SG`/`TG2` the kernel
     was compiled for. A `_256` kernel dispatched with fewer threads per group
     would leave `threadgroup` memory partly unwritten, and unwritten
     threadgroup memory is *whatever was there last time* — which is a
     nondeterminism source that looks exactly like a race.

- [ ] **Bisect with the margin instrument, which is cheaper than diffing kernel
  tables.** `KAPLUMBA_MARGIN=1` prints per-step top-1-vs-runner-up in fp16 ulps.
  Run probe C twice; where the two runs diverge, the last common step will show a
  margin near 1 ulp. That names the step instead of the kernel.

  Note what the margin means here: the logits are fp16, so a margin below one
  fp16 ulp at the winning logit is a single rounding decision from a different
  token. If the divergence always lands on a sub-ulp step, the prefill bug only
  has to be *tiny* to explain it — do not go looking for a large error.

- [ ] **Note the trap:** `KAPLUMBA_GPU_PROFILE` alone historically printed the
  **prefill's** table for a decode run. Fixed in `a58fa7b`, with a gate test
  (`tests/gpu_profile_decode_gate.rs`). If the table looks like prefill, that gate
  is not covering the path you are on.

- [ ] **Regression gate, once fixed.** A test that runs probe C's shape twice and
  asserts identical token ids. It will be slow, so mark it `#[ignore]` and gate it
  the way the other checkpoint-loading tests are gated — but it must exist, or the
  next kernel change re-opens this silently.

---

## H4 — Close the input-transform coverage gap (small, mechanical)

**No test covers the input-transform dispatch the engine actually uses.**
`model.rs::encode_input_transform` is not what the suites drive — they use
`gemv_buffer_impl`. There are **four** dispatch sites for the transform/finalize
pair and they must stay in lockstep.

This is not hypothetical: a 22/22-green change once shipped a broken model
because the tested path was not the shipped path. And the two-word work already
hit a variant of it — `encode_ishizuki_gemv_named` initially skipped the input
transform, produced all zeros, and presented as "the live kernel does not work".

- [ ] Factor the thread count into one helper, as was done for `rope_threads`.
- [ ] Add a test that drives `encode_input_transform` on the shapes decode uses.
- [ ] Grep for every dispatch site and assert they agree, rather than trusting
  that they do.

---

## H5 — Only if H1 and H2 both come back negative: the bytes-per-token routes

*Not an implementation task. A decision the user has to make.*

- At 240 GB/s, 12.088 GB/token is 50 ms = 20 tok/s. At 3.0 bpw that is the wall,
  and 20 is not available anyway (263 GB/s needed).
- oMLX runs **2.25 bpw** (oQ2 + gs64), ishizuki benchmarks at **2.0**. At 2.25
  bpw: 9.07 GB/token = 37.8 ms = **26 tok/s** of headroom.
- This is a **quantisation decision with a quality cost, not a port.** It cannot
  be done by an agent. Enumerate the options and their quality implications and
  let the user choose.
- What is **not** a bytes lever: the EXL3 rate is already exactly 3.000
  bits/value on 410 of 411 tensors (`lm_head` is 6.000), and no repacking
  reduces bytes. The 12.51 bits of real entropy per weight is entropy already
  paid for in 3.

---

## H6 — Optional: `MTLResidencySet` pinning (untested, not disproved)

`-[MTLDevice newResidencySetWithDescriptor:error:]` SIGSEGVed three times on this
binding, `class!(MTLResidencySet)` returns a protocol (nil), and
`-newResidencySetWithError:` on a buffer **aborts the process**.

It would pin pages we already have, at no memory cost, and **no measurement
exists either way**. Try an extern declaration with the right signature, or a
`madvise`-level equivalent (`MADV_WILLNEED`). llama.cpp refreshes a 5 ms
heartbeat for 180 s TTL.

Low priority: T1 established that the host is not the bottleneck, so anything that
only helps the host is worth little.

---

## Done when

H1 has a number in the spec's tables and a verdict on whether T7 is real. If it
is, H2 has a measured residual. H3 is either fixed or documented with a located
source. Everything dead is in the closed-lever table with its ratio.

**And throughout: bit-exactness or nothing ships.** Five reference prompts must
stay exact — Paris / 1, 2, 3, 4, 5 / red, blue, and yellow / Hello! How can I
help you today? / 17 times 23 is 391 — and `tests/decode_matvec_golden.rs` must
pass.
