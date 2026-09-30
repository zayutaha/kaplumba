# Handover tasks: sustaining decode bandwidth past 15 tok/s

Companion to `spec/decode-bandwidth-handover/spec.md`. Read that first — it has
the state of play, the kernel-dispatch trap, and the measurement discipline.

Ordered by what a fresh agent should do first. **H1 is the whole point of this
document**; the rest is context.

---

## H1 — DONE. The instrument is fixed and the table is trustworthy

`gemv_shape --rate-interleaved` replaces `--only "bit rate"`, interleaves every
arm, and **carries 6 bpw through the run as a control** (drift 13%, acceptable):

| rate | window | GB/s |
|---|---|---|
| 3 bpw | 3-word | 150.9 |
| 3 bpw | **2-word (shipped)** | **165.7** |
| 4 bpw | 2-word | 216.4 |
| 6 bpw | control | 215.7 |
| 8 bpw | 3-word | 217.9 |

The old table's numbers are withdrawn. It could not resolve a 10% effect.

## H2 — DONE, and the answer is that none of the three is the mechanism

- [x] **H2c (decode cost) is measured dead — in the strongest way.** The `_nod`
  probe (`--decode-probe`) is the verbatim kernel with `decode_state` replaced by
  a one-AND stub: identical memory traffic, one asserted substitution, garbage
  output, never dispatched.

  | rate | real | no-decode | effect of REMOVING decode |
  |---|---|---|---|
  | 3 bpw | 167.3 | 122.7 | **-26.7%** |
  | 4 bpw | 217.1 | 206.3 | -5.0% |
  | 8 bpw | 222.5 | 219.2 | -1.5% |

  **Removing the decode made every rate slower.** A cheaper decode cannot be the
  fix, because the free decode is slower than the real one. Probably
  `decode_state`'s dependent chains were accidentally supplying the ILP that
  hides load latency.

- [x] **H2a (register pressure) and H2b (shifter pressure) are now implausible as
  the primary mechanism.** Both were motivated by the 3-vs-6 bpw gap, and the gap
  turns out to be request/latency-bound — T6's mechanism — with the residual
  2.67x lane-load amplification structural to the EXL3 lane layout, whose
  on-disk format is fixed.

**The single most important finding: 4, 6 and 8 bpw are all at 215-218 GB/s.**
Flat across a 2x range of rate. DRAM-saturated. Only 3 bpw is slow, by 1.306x.
Nothing kernel-level has been found for it, and three separate candidate
mechanisms are now measured or reasoned out.

## H2.5 — The target itself is not reachable at 3.0 bpw (read this before optimising)

| | ms/token | tok/s |
|---|---|---|
| measured now | 93.1 | 10.74 |
| 3 bpw at 4 bpw's 217 GB/s | 62.4 | 16.0 |
| 3 bpw at pure-streaming 240 GB/s | 57.1 | 17.5 |

**19 tok/s needs 52.6 ms; after 7.4 ms of non-matvec work the matvec has 45.2 ms
for 11.932 GB = 264 GB/s, above this box's measured 239-244 GB/s ceiling.**
Working back, 19 tok/s requires **1.888 bpw**.

So the honest routes are:
- **~17.5 tok/s at 3.0 bpw** — the hard ceiling, and only if the 3 bpw gap closes.
- **~18.1 tok/s at 2.0 bpw** — ishizuki's published operating point.
- **19+ tok/s at 1.9 bpw** — below anything known to work on this model.

**All of the routes past 17.5 are quantisation decisions, not code.** They need a
different, smaller checkpoint and a quality trade the user has to make. Do not
spend kernel effort chasing 19 tok/s at 3.0 bpw; the arithmetic forecloses it.

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
