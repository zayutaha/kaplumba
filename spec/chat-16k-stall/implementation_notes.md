# implementation_notes — `spec/chat-16k-stall`

Append as work proceeds, newest at the bottom. Every number here was measured
on this box this session unless it says otherwise.

## 0. The reported symptom

`kaplumba --chat` with a long context "goes down to 0.x tok/s". Reproduced
exactly: **0.03 tok/s** (16 generated tokens, 11,299 context) — 16 tokens in
494.9 s. Then the user killed a run that had been going 600 s+ (exit 137 =
SIGKILL).

## 1. First measurement: it is `--chat`, not the model

Single-shot and chat on the same checkpoint, same context, same prompt bytes:

| context | single-shot | chat (before fixes) |
|---:|---:|---:|
| 2,234 | **14.71** | 6.71 |
| 11,354 | **14.01** | — |
| 16,484 | **13.45 / 13.51 / 13.49** | — |

So the generator is healthy at 16k and the chat path is not. That splits the
problem immediately and stopped me from looking at kernels, which a
subagent's ranked hypothesis list had suggested.

Peak memory at 16k single-shot: `phys_footprint` 15,702,698,688 B (14.6 GiB),
0 swap. Fine.

## 2. Chat tok/s vs context, before any fix

| context | tok/s |
|---:|---:|
| 199 | 4.39 |
| 754 | 8.56 |
| 2,234 | **10.94** (peak) |
| 4,454 | 7.02 |
| 7,414 | 2.02 |
| 11,299 | 1.21 |

Rises to ~2k then collapses. Not a threshold at any one of these — and
specifically **not** a "200 cliff" as first guessed; the 199-token point is
slow for a different reason (a 32-token reply sits inside per-turn fixed costs,
and there is barely any KV to amortise them over).

## 3. Bug A — wait before encode. FIXED, `8fbf487`

The loop did `previous.wait()` at the top of the iteration, before building the
next encoder, so the GPU drained and idled through the host's whole
`forward_step_pipelined` encode on **every token**. `generate.rs:1031` commits
`next_enc` and `:1033` then waits — it always had it right.

Reordered to encode+commit N+1 first, then wait on N. The
`read_host_token`-after-`wait` invariant is preserved (the file records that
reading a staged token early was a shipped bug once).

| | tok/s @2,234 |
|---|---:|
| chat before | 6.71 |
| **chat after** | **15.11 / 15.18 / 15.19 / 15.21** |
| single-shot (reference) | 15.16 / 15.38 / 15.16 |

**Parity.** Two source-level tests added, both sabotage-verified: reverting the
order fails `the_pipelined_chat_loop_commits_before_it_waits` with the offsets
printed. A second test guards the read-after-wait invariant.

⚠️ **This fix is NOT yet shown to be output-identical.** T1 of the spec runs
the `#[ignore]`d token-stream equivalence tests. Until then it is a
performance claim, not a correctness claim.

## 4. Bug B — the in-flight window is smaller than one decode step. NOT FIXED

Bug A fixed short context to parity but **16k still collapses**. Measured after
the fix, same 11,299 context:

| max-new-tokens | decode time | tok/s |
|---:|---:|---:|
| 8 | 38.0 s | 0.21 |
| 12 | 224.0 s | 0.05 |
| 16 | 291.9 s | 0.05 |
| 16 (repeat) | 23.9 s | 0.67 |
| 20 | 47.8 s | 0.42 |

That scatter across repeats of the *same* binary is itself the finding: there
is no clean curve, so **a median of a good run proves nothing here.** It is a
stall that is sometimes not hit.

### The stacks

Three independent `sample` captures during a stall, 100% agreement, all on the
main thread inside the host encode:

```
forward_step_pipelined (1470 / 1513 / 612 samples)
  forward_chunk
    ...
    MetalEncoder::tick            (encoder.rs:524)
      -[_MTLCommandBuffer waitUntilCompleted]      <-- host blocked, 1419/1513/587 samples
```

100% of the stall is `tick()` → `roll()` (`encoder.rs:539-549`) blocking on the
oldest in-flight buffer.

### The arithmetic that explains it

```
600 dispatches per decode step / 50 per buffer  =  12 command buffers per step
MAX_IN_FLIGHT                                    =   8
```

**The window cannot hold one decode step.** So after ~8 buffers the host stops
encoding and waits for step N's first buffer to drain, mid-step. Host and GPU
are fully serialised; the "pipelined" loop degenerates to synchronous.

Why short context hid it: identical dispatch count, but at ≤2k each buffer
drains in microseconds so the block is unmeasurable. At 16k a decode step is
~74 ms of GPU work spread over 12 buffers, and the host waits for most of it.

### Consequence for reading any earlier chat number

Every chat tok/s recorded before Bug A was fixed, and every one recorded
*before* Bug B is fixed, is a measurement of a stall whose depth depends on
which buffers happened to be outstanding. **Do not compare across those two
regimes.**

## 5. Retractions and corrections

* **The "+3.63 tok/s if layers reach 254.6 GB/s" is withdrawn.** 254.6 is the
  *six* bpw rate; the layers are 3 bpw, where this box reaches 165.2 GB/s at
  the head's shape, and the layers already run 185–203. See
  `spec/decode-3bpw-gap-19t/implementation_notes.md §13`.
* **The O(N²) `Detokenizer::push` theory was wrong.** Measured 0.04–0.13 ms per
  token. It replays *generated* ids only (16–60 of them), not the prompt. Not a
  factor here; still a latent defect for a 4096-token reply.
* **"Wait-before-encode is C4, worth a little."** It was worth 2.1×. The
  subagent's ranking had it as "real, but small".

## 6. Environment gotchas hit

* **Stale binary twice.** `cargo build -p kaplumba-engine` does **not**
  rebuild `target/release/kaplumba` (a different package: `kaplumba-cli`).
  Measured "6.71 vs 15.69, no fix" against a binary that did not contain the
  fix. Always `cargo build --release -p kaplumba-cli` before a chat
  measurement. This is the `include_str!` class of trap in a new costume.
* **`timeout` does not exist** on this box; a monitoring loop with `kill -0`
  is the substitute.
* **A killed model leaves a stale lock** at `~/.kaplumba/gpu.lock` naming a dead
  pid; `gpuq status` reports it as busy until the pid is reaped. It clears on
  its own after ~5 s.
* Agent overlap produced 35.3 / 55.3 / 156.5 GB/s for the same arm in one
  session. Those are **discarded, not spread** (`crux.md §2.7`).

## 7. Log

(append per task: command, median ± spread, control drift, verdict, commit)
