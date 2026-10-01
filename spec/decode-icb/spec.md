# spec/decode-icb — Indirect Command Buffer decode replay

## Goal

Cut decode step time from 65.3 ms (15.3 tok/s) to ~45 ms (18+ tok/s) by
eliminating per-launch CPU dispatch overhead. A decode step issues ~906 kernel
launches; 840 of them are small non-GEMV kernels doing 0.17 ms of GPU work that
cost ~20-30 ms of CPU dispatch time. Metal's Indirect Command Buffer (ICB)
records the kernel sequence once and replays it with a single CPU call,
removing that overhead.

## Background

Measured on hardware (M4 Pro, Gxma EXL3 3.0 bpw, one process at a time):

- Decode baseline: **15.3 tok/s = 65.3 ms/step**, bit-exact, 0 swap.
- GPU kernel time per token: **~33.5 ms** (7 GEMV kernels x 60 layers = 32.5 ms,
  8 small kernels x 60 layers = 0.17 ms).
- **Gap: ~20-30 ms/token** where the GPU is idle waiting for the CPU to encode
  the next launch. The profiler reports 23.5 ms; the encoder's own doc comment
  says ~5.4 ms of non-GEMV launch latency (older measurement, more dispatches).
  The true number must be measured on an idle GPU (T1).
- The rolling commit scheme (`MAX_DISPATCHES_PER_BUFFER=50`, `MAX_IN_FLIGHT=8`)
  splits a decode step into ~18 command buffers, not 1.

This reframes the problem: the bottleneck is **CPU dispatch count**, not memory
bandwidth. Every kernel-bandwidth optimization attacks the 33 ms while the real
problem is the 20-30 ms of dispatch overhead.

## Non-goals / restrictions

- **No kernel math changes.** ICB replays the exact same kernels with the exact
  same arguments. Bit-exactness is structural, not a hope.
- **No modification to the existing `encoder.rs` dispatch path.** ICB is a new,
  self-contained module. The normal dispatch path stays untouched.
- **No KV cache quantization.** No speculative decoding. Single-user.
- **Decode-only.** ICB records the decode forward pass (1 token). Prefill uses
  the normal dispatch path.
- **No Python in product/engine code.**
- **Original checkpoint untouched.** 3 bpw stays 3 bpw.
- **Resident <= 16.5 GiB, 0 swap.**

## Requirements

1. **R1 — Verify the gap.** Measure the true CPU encode-submit cost on an idle
   GPU (no profiler distortion). Confirm it is >= 15 ms/token before building
   ICB. If it is < 15 ms, ICB is not worth building and we fall back to fusion.
2. **R2 — ICB module.** New module `crates/kernels/src/icb.rs` providing:
   - Record: walk the decode kernel sequence into an ICB.
   - Replay: execute the ICB with a single CPU call.
   - No changes to `encoder.rs` or the existing dispatch path.
3. **R3 — Whole-model, 2 ICBs.** Record the entire decode forward pass (906
   dispatches) as 2 ICBs, one per KV cache slot (`slot % 2`). Token ID, position,
   and KV cache all live in per-slot device buffers, so 2 ICBs covers everything.
4. **R4 — Bit-exact.** ICB replay produces identical greedy output to the normal
   dispatch path. Verified by digest comparison over 100+ tokens.
5. **R5 — Dispatch abstraction.** The `encode_*` functions must be able to write
   to either a normal command buffer or an ICB recorder. This requires a trait
   abstraction over the dispatch calls. The trait is the only change to existing
   code, and it is additive (a new trait + impl for the existing encoder).
6. **R6 — No prefill regression.** Prefill uses the normal dispatch path. ICB
   is decode-only. Prefill TTFT must not regress.
7. **R7 — Measurement.** A/B the ICB path against the normal path on the same
   hardware, interleaved, with medians and spread. Target: >= 18 tok/s.

## Acceptance criteria

- **A1.** True CPU encode-submit cost measured on idle GPU: reported with the
  number and the command used.
- **A2.** ICB replay decode: >= 18 tok/s (55.5 ms/step), bit-exact greedy output
  ("The capital of France is Paris."), 0 swap.
- **A3.** Bit-exactness: 100+ token digest identical between ICB and normal path.
- **A4.** Prefill TTFT unchanged (within noise).
- **A5.** A/B measurement: interleaved, median + spread reported.
- **A6.** If ICB does not reach 18 tok/s, the fusion fallback (15 -> 8
  kernels/layer) is tracked as a separate spec and the ICB negative result is
  recorded with numbers.

## Open questions

- **Q1.** Does the trait abstraction (`Dispatch` trait) cover all the dispatch
  patterns used in the decode path? The existing `MetalEncoder` has `dispatch_1d`,
  `dispatch_2d`, `dispatch_groups`, `dispatch_1d_offset`. Are there others?
  *To be answered in the plan by enumerating all dispatch call sites.*
- **Q2.** The ICB descriptor's `max_kernel_buffer_bind_count` must be set high
  enough for the kernel with the most buffer arguments. What is that number?
  *To be answered by auditing the kernel signatures.*
- **Q3.** Does `concurrent_dispatch_threadgroups` produce identical results to
  `dispatch_threadgroups` for these kernels? *To be verified by the bit-exactness
  test (R4).*
- **Q4.** The KV cache slot alternation: are there any other per-slot buffers
  beyond token, position, and KV cache? *To be answered by auditing the decode
  path for slot-dependent pointers.*
- **Q5.** ICB memory: 906 commands x 2 ICBs. How much device memory? *To be
  measured during implementation.*
