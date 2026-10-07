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

### Bug B — The Memory / Swap Cliff. ROOT CAUSE OF 0.03 tok/s STALL.

Live process profiling (`sample` and `vmmap` on stalled process PID 68980 during
a 16k stall) revealed the real mechanism:

```
IOAccelerator (graphics): VIRTUAL 13.4G, RESIDENT 13.4G, SWAPPED 12.5G
mapped file:              VIRTUAL 15.8G, RESIDENT 5.3M,  SWAPPED 0K
TOTAL:                    VIRTUAL 417G,  RESIDENT 13.7G, SWAPPED 12.6G
```

1. **Dirty Anonymous Weight Buffers:** `weights.rs` loads weights by allocating
   ~1,440 separate `StorageModeShared` Metal buffers and copying from disk via
   `read_exact_at` (`pread`). This turns 13.4 GiB of weights into dirty
   anonymous memory (`IOAccelerator (graphics)`).
2. **Exceeding Unified Memory Budget:** On a 24 GiB box with 8–9 GiB of macOS
   wired memory, the 13.4 GiB model barely fits at short context. At 11k–16k
   context, the KV cache adds ~3.0 GiB, pushing total demand past 25 GiB.
3. **Compressor & Swap Thrashing:** Memory pressure turns yellow. EXL3 weights
   have Shannon entropy ~7.97 bits/byte and are incompressible, forcing
   **12.5 GiB into swap/compressor**.
4. **Per-Token Disk I/O:** Every decode token reads all 60 layers (13.4 GiB of
   weights), triggering continuous SSD swap-ins (~237,000 pageins, >1,000,000
   faults). A 74 ms step turns into 18–30 seconds per token (**0.03 tok/s**).
5. **Why `waitUntilCompleted` blocked:** Host `MetalEncoder::tick` / `roll` blocks
   waiting on command buffers whose GPU completion is throttled by SSD swap paging.
6. **Contrast with MLX:** MLX maps weights directly using `mmap` with clean
   file-backed pages (`SWAPPED = 0K`), which are never compressed or swapped.

### Bug C — Missing KV cache retraction on EOS in `chat.rs`.

In `crates/engine/src/chat.rs`, the pipelined decode loop speculatively encodes
step $N+1$ before waiting on step $N$. When step $N$ produces an EOS token,
`generate.rs:1050` calls `model.retract_gpu_caches(1)`, but `chat.rs:507`
breaks without retracting. The KV cache cursor advances past the actual
conversation on every turn.

## 3. Non-goals & restrictions

* **No speculative decoding**, no MTP, no drafter, no n-gram, no lookup.
* **No KV cache quantization** (`AGENTS.md §8`). At 16k KV is 2.181 GB and is
  immovable.
* **Embed tokens (2.82 GB) remains strictly offloaded** via zero-copy file-backed
  mapping (`raw_zero_copy_tensor_buffer` with in-kernel BF16 conversion touching 1 row/token).
* **Vision tower (1.15 GB) remains strictly excluded / never loaded** (0 bytes allocated).
* **No quality loss.** Greedy output byte-identical. The reorder in Bug A is
  explicitly a reordering of *when* buffers are submitted, not of arithmetic;
  the token-stream equivalence tests must prove it.
* **No change to decode numerics.** Bug B is a memory allocation defect (dirty
  anonymous buffers vs clean file-backed pages), not a kernel defect. No kernel,
  dispatch count, or reduction order may change.
* **Not a bandwidth problem.** Single-shot at the same context is 13.45–13.51,
  so the hardware is delivering. Any fix that only helps the generator has
  missed the target.
* One model process at a time; `scripts/gpuq try --`, exit 75 = busy
  (`AGENTS.md §10`).

## 4. Requirements

* **R0 — Eliminate dirty anonymous weight memory.** Replace `allocate_dedicated`
  + `read_exact_at` for layer weights with zero-copy file-backed Metal views
  (`new_buffer_with_bytes_no_copy`), pre-faulted at load time via `madvise(MADV_WILLNEED)`.
  Ensure `F_NOCACHE` is applied on shard file descriptors to prevent duplicating
  into the kernel Unified Buffer Cache.
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