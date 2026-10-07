# Spec: `--chat` collapses to 0.0x tok/s at long context

## 1. Goal

Make `kaplumba --chat` decode at the same rate as the single-shot generator
(`--prompt`) at every context length, and prove it at 16k.

**Current state, measured this session (2026-10-07), `Gxma-repacked`:**

| path | context | tok/s |
|---|---:|---:|
| single-shot | 2,234 | **14.71** |
| single-shot | 11,354 | **14.01** |
| single-shot | 16,484 | **13.45 / 13.51 / 13.49** |
| `--chat` | 2,234 | **6.71** (before this spec) |
| `--chat` | 11,299 | **0.03** (16 tok), **0.21** (8), **0.05** (12), **0.42** (20) |

The single-shot path is healthy. `--chat` collapses, and it collapses *harder*
the longer it runs, which is the opposite of a fixed per-token cost.

## 2. Two bugs, already separated by measurement

### Bug A — the decode loop waited before it encoded. FIXED (`8fbf487`).

The pipelined chat loop did `previous.wait()` at the top of the iteration,
*before* building the next encoder, so the GPU drained and idled through the
host's whole `forward_step_pipelined` encode on every token.
`generate.rs` has always had the right order.

At 2,234 tokens: **6.71 → 15.11–15.21 tok/s**, against single-shot
15.16–15.38 — i.e. **parity**. Two source-level tests guard it
(`multi_turn_chat.rs`, both sabotage-verified).

This fix is real and kept. It is **not** the 16k bug.

### Bug B — `MAX_IN_FLIGHT` is smaller than one decode step. NOT FIXED.

Sampled stacks during a 16k stall, three independent captures, 100% agreement:

```
forward_step_pipelined
  forward_chunk
    ... 1419/1513/1252 samples ...
      MetalEncoder::tick
        -[_MTLCommandBuffer waitUntilCompleted]   <-- the host, blocked
```

`tick()` → `roll()` (`crates/kernels/src/encoder.rs:524-549`) blocks on the
oldest in-flight buffer as soon as the window is full. The arithmetic:

```
600 dispatches per decode step / 50 per buffer = 12 command buffers per step
MAX_IN_FLIGHT = 8                                    <-- window is SMALLER than one step
```

So the host finishes ~8 buffers of step N, blocks for step N's *first* buffer
to drain, and only then continues encoding. Host and GPU are fully serialised.

**Why short context hid it.** Identical 600 dispatches, but at 29–2234 tokens
each buffer drains in microseconds so the wait is invisible. At 16k each decode
step is ~1.5 s of GPU work and the wait becomes the step.

**Why it looks superlinear in generated tokens, not context.** It is not. It is
a serialisation whose cost per token is set by the *GPU* time of a step, so
`tok/s ≈ 1/(host_encode + gpu_step)` — and the sampled `waitUntilCompleted`
dominates. The 8/12/16/20-token runs (38.0 / 224.0 / 291.9 / 47.8 s) are the
same bug sampled at different points, not a curve.

## 3. Non-goals & restrictions

* **No speculative decoding**, no MTP, no drafter, no n-gram, no lookup.
* **No KV cache quantization** (`AGENTS.md §8`). At 16k KV is 2.181 GB and is
  immovable.
* **No quality loss.** Greedy output byte-identical. The reorder in Bug A is
  explicitly a reordering of *when* buffers are submitted, not of arithmetic;
  the token-stream equivalence tests must prove it.
* **No change to decode numerics.** Bug B is a submission-window defect, not a
  kernel defect. No kernel, dispatch count, or reduction order may change.
* **Not a bandwidth problem.** Single-shot at the same context is 13.45–13.51,
  so the hardware is delivering. Any fix that only helps the generator has
  missed the target.
* One model process at a time; `scripts/gpuq try --`, exit 75 = busy
  (`AGENTS.md §10`).

## 4. Requirements

* **R1 — Size the window from the work, not from a constant.** The window must
  be able to hold **one whole decode step**'s worth of command buffers with room
  to spare, so `roll()` never blocks mid-step. Derive it from the dispatch
  count per step and the per-buffer cap rather than hard-coding 8, and document
  the derivation at the constant.
* **R2 — Prove `roll()` cannot block mid-step.** A test that asserts the
  invariant structurally, plus a measurement at 16k showing the host-side
  `tick` wait has gone to ~0.
* **R3 — Back-pressure must remain.** The window exists to stop the host
  queueing without limit. Raising it must not turn into unbounded submission:
  assert a bound on outstanding buffers and on resident memory.
* **R4 — Chat/single-shot parity at every context.** Acceptance is a
  *ratio*, not an absolute: at 614 / 2,234 / 8,192 / 16,384 tokens, chat
  decode ÷ single-shot decode ≥ 0.95. Parity is the claim; the absolute rate
  is the generator's business.
* **R5 — Token-stream equivalence is mandatory and currently unrun.**
  `multi_turn_chat.rs` has `#[ignore]`d tests that compare pipelined vs
  barrier output across a multi-turn session. **Bug A reordered buffer
  submission and has not yet been shown not to change any token.** This is the
  single highest-risk unverified claim in the whole spec. Run them.
* **R6 — A regression test that can fail.** The 16k stall is invisible in the
  token stream, exactly as the ordering bug was, so the guard must be a source
  or structural assertion *plus* a measured ratio. Per `AGENTS.md §3.4` each
  must be shown to fail when the defect is reintroduced.
* **R7 — Multi-turn, not just one turn.** The reported symptom is a long
  conversation. A single-turn 16k paste is the minimum case; verify several
  turns accumulate context without the rate degrading per turn.

## 5. Acceptance criteria

* **AC1** Chat ÷ single-shot decode ≥ 0.95 at 614 / 2,234 / 8,192 / 16,384.
  Report each ratio with the two absolute numbers beside it.
* **AC2** At 16k, chat ≥ 12.0 tok/s (i.e. within 10% of single-shot's
  13.45–13.51), and no run slower than 13 tok/s across ≥3 repeats. **Report
  the spread** — the bug presented as 0.03–0.42 across repeats of the *same*
  binary, so a median of a good run is not evidence.
* **AC3** `pipelined_and_barrier_decode_agree_across_a_multi_turn_session`,
  `a_single_greedy_turn_reproduces_the_counting_sequence`,
  `context_advances_by_exactly_the_tokens_each_turn_consumed` — all green
  (`--ignored --nocapture`). Greedy output for the AC1 prompt byte-identical.
* **AC4** Host-side `tick`/`roll` block time measured at 16k ≈ 0, by stack
  sampling during decode, with the sample counts printed (this session's
  evidence is 1419/1513/1252 of 1470 samples).
* **AC5** Outstanding buffers bounded; `phys_footprint` via `/usr/bin/time -l`
  ≤ 16.5 GiB, 0 swap. Never `peak_resident_pages_bytes` (`AGENTS.md §10.3`:
  understates by 5.5 GiB).
* **AC6** `git diff --exit-code` clean after `scripts/gen_exl3_gemv.py`
  (AC6 of the parent spec).

## 6. Open questions

* **Q1** Is the right fix a larger `MAX_IN_FLIGHT`, or a larger
  `MAX_DISPATCHES_PER_BUFFER` so a step needs fewer buffers? Both are one-line
  changes with different memory and latency consequences; the sweep in flight
  when this spec was written will answer it. Prefer the one that keeps
  `roll()` non-blocking *and* bounds memory.
* **Q2** Why is a decode step ~1.5 s at 16k when single-shot runs it in
  ~74 ms? That is a 20× discrepancy and is **not explained** by this spec. It
  is either the same serialisation seen from the GPU side, or a second
  independent defect. R2's measurement will distinguish them, and if a second
  defect exists it needs its own task.
* **Q3** Does the single-shot path have the same latent window problem? It sets
  `set_max_dispatches(decode_max_dispatches_per_buffer())` on its encoders too,
  so it is exposed — it just never blocked, because its loop commits and waits
  in a different order. Confirm rather than assume.
* **Q4** Is `MAX_IN_FLIGHT` also the cause of the *short*-context chat gap
  (6.71 before Bug A)? Bug A fixed short context to parity, so no — but the two
  interact and the regression test must cover both.

## 7. Supersession

Supersedes the chat half of `spec/decode-alu-fuse-18t` (T5/T6 wired the device
sampler; that work is kept and is unrelated to this defect). Extends
`spec/decode-3bpw-gap-19t`, which measured the generator side and established
that single-shot 16k is 13.45–13.51 — the number this spec must match.
`crux.md §2.5/§2.7` for why every rate here needs a quiet box.