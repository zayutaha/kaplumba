# Tasks

Ordered smallest-first. Each names files touched, intent, and how it is tested.
**Update this file immediately after each completed task**, with status and
commit hash. If scope changes, amend `spec.md` / `plan.md` first.

Legend: [ ] pending · [~] in progress · [x] done

---

## T0 — Measure the tensor-core peak
**Why first:** it decides whether 200 tok/s is ordinary engineering or a wall.
SIMT fp32 FMA peak is 8.8 TFLOPS, *below* the 11.5 TFLOPS the target needs.
The tensor-core path is a separate unit that benchmark does not measure. If
tensor peak is ~40 TFLOPS the target is reachable; if it is ~12 it is not, and
that must be known before T3 is built.

- [ ] **T0.1** Extend `crates/engine/examples/metal_probe.rs` into a sustained
      `simdgroup_multiply` throughput loop: a large `half` buffer pair, many
      independent 8x8 tiles in flight per simdgroup, enough repetitions to leave
      the launch overhead out of the figure.
- [ ] **T0.2** Report achieved TFLOPS and compare against the SIMT `alu_peak`
      numbers. State plainly in `implementation_notes.md` whether 11.5 TFLOPS is
      comfortably inside the tensor envelope.
- **Files:** `crates/engine/examples/metal_probe.rs`
- **Test:** the printed TFLOPS figure; cross-checked by making the loop
  arithmetic-only and confirming it is compute-bound (doubling the tile count
  should roughly double the time).

## T1 — Attribute the current prefill cost by ablation
**Why:** `layer_bench` was found to be measuring `commit_and_wait` overhead
rather than kernels, so the microbenchmarks in this repo cannot be trusted for
attribution. Only end-to-end ablation is admissible. Two candidate root causes
are live: a scalar prefill GEMM, and a prefill attention kernel that spills its
`float accum[512]` to local memory.

- [ ] **T1.1** Report prefill tok/s **and** achieved TFLOPS at 1k / 2k / 8k
      with the current code. This is the baseline every later task is measured
      against.
- [ ] **T1.2** Isolate the prefill attention kernel: replace it with a no-op
      (output will be wrong; the GEMM still runs) and re-measure. The delta is
      attention's share.
- [ ] **T1.3** Confirm the spill hypothesis directly. Force `head_dim` to a
      compile-time constant in `online_attention` so `accum` can be promoted to
      registers, and measure the change. A large win confirms the spill.
- [ ] **T1.4** Record the GEMM's share as the remainder, and state which
      candidate dominates.
- **Files:** `crates/kernels/src/attention.metal`,
  `crates/engine/src/model.rs` (diagnostic switch only)
- **Test:** end-to-end `prefill_tok_s` deltas; suite must stay green, and the
  diagnostic switch must be off by default and removed or clearly quarantined.

## T2 — Establish the simdgroup matrix layout, with a unit test
**Why:** the probe proves `simdgroup_multiply` compiles and runs on this M4 Pro,
but the numeric check currently matches only 8 of 64 elements. Getting the
`simdgroup_load` / `simdgroup_store` element order and `elements_per_row`
convention right is easy to get subtly wrong and impossible to debug later from
a misbehaving model.

- [ ] **T2.1** Add a unit test computing an 8x8 identity x known product, and a
      second with a non-trivial `elements_per_row` stride. Both must match
      exactly.
- [ ] **T2.2** Add a test for the accumulation pattern: `simdgroup_multiply`
      into an existing tile (accumulate) as well as overwriting.
- [ ] **T2.3** Document the layout convention in `implementation_notes.md`,
      including the API names, which differ from the obvious spelling
      (`simdgroup_load` / `simdgroup_store` / `simdgroup_multiply`, and
      `simdgroup_half8x8` with no `_t`).
- **Files:** `crates/engine/examples/metal_probe.rs`, new test module
- **Test:** the two unit tests above, exact equality.

## T3 — Tiled prefill GEMM on the tensor-core path
**Why:** the prefill GEMM is the largest share of the work by FLOPs (97% at
8192 tokens) and is the main route to 11.5 TFLOPS.

- [ ] **T3.1** Stand up a standalone benchmark for the prefill linear shape
      (5376 -> 21504 and 5376 -> 8192) reporting TFLOPS, so GEMM work is
      measurable without the whole model in the way. **Reuse T0's timing
      discipline: no fixed-overhead contamination, report the marginal cost.**
- [ ] **T3.2** Write a tiled prefill GEMM: tile M and N, stream K, inner tile
      via `simdgroup_half8x8`, accumulate fp16 within a tile and fold to fp32 on
      a fixed cadence.
- [ ] **T3.3** Keep dequantisation hoisted per quant group — no per-element FP
      multiply, no dynamic array index, since a dynamic index spills on Apple
      GPU.
- [ ] **T3.4** Verify against the existing prefill path within the OQ3
      tolerance, on the reference shapes.
- [ ] **T3.5** Wire into the model behind a switch; measure end-to-end prefill
      TFLOPS and compare to T1.1.
- **Files:** `crates/kernels/src/linear.metal`, `crates/kernels/src/linear.rs`,
  `crates/engine/src/model.rs`
- **Test:** standalone TFLOPS figure; end-to-end prefill tok/s; suite green.

## T4 — Tiled online-softmax prefill attention
**Why:** `online_attention` holds `float accum[512]` indexed by a runtime loop
bound, which cannot stay in registers, and the standing restriction requires an
O(L) working set with no materialised score matrix.

- [ ] **T4.1** New kernel: one threadgroup per (query tile, head); keys streamed
      in fixed-size tiles; online softmax state (running max, running sum,
      output accumulator) in registers, updated once per key tile.
- [ ] **T4.2** Prove the working set is independent of context length: the only
      scratch is the key tile. Report the allocation and show it is identical
      for 1k and 8k (R5).
- [ ] **T4.3** Measure and report local-memory traffic for the kernel, to
      demonstrate the accumulator is register-resident rather than spilled.
- [ ] **T4.4** Use simdgroup MMA for QK and PV.
- [ ] **T4.5** Verify against the current serial kernel within the OQ3
      tolerance, across context lengths and both layer kinds (sliding and
      global, head_dim 256 and 512).
- **Files:** `crates/kernels/src/attention.metal`,
  `crates/kernels/src/attention.rs`, `crates/engine/src/model.rs`
- **Test:** numerical agreement at 128 / 1k / 2k / 8k; scratch size invariant;
  suite green.

## T5 — Preemption audit and per-dispatch timing
**Why (R8):** oMLX measured the macOS IOGPU interactivity preemption cliff above
~50 ms per dispatch on an M3 Max and targets 10 ms on every tier, splitting long
dispatches with logsumexp partials. M4 Pro is pre-NAX and prefill is exactly
where a single dispatch can run long. Never checked here.

- [ ] **T5.1** Report per-dispatch wall clock for the prefill path at 1k / 2k / 8k.
- [ ] **T5.2** If any dispatch exceeds ~10 ms, chunk it and fold the partials
      with logsumexp, as oMLX does. If none does, record that and the margin.
- **Files:** diagnostic instrumentation; possibly `attention.metal`
- **Test:** the reported per-dispatch maximum, and an unchanged prefill rate.

## T6 — Memory proof at 1k / 2k / 8k
**Why (R6):** peak RSS is a measured acceptance number, not a prediction, and
there is an unresolved conflict (OQ1) that this task is meant to surface.

- [ ] **T6.1** Record `phys_footprint` via `/usr/bin/time -l` at 1k / 2k / 8k.
- [ ] **T6.2** Report the KV cache size separately from the weight residency,
      and state the total against the agreed budget.
- [ ] **T6.3** Raise OQ1 with the measured numbers if the 8192 case exceeds
      12 GiB, with the three options costed.
- **Files:** `crates/engine/tests/memory_proof.rs` (extend to 8k),
  `spec/prefill-200tok/implementation_notes.md`
- **Test:** the recorded figures, referenced from `spec.md` acceptance criteria.

---

## Sequencing

```
T0 ──► (decides whether T3 is worth building at all)
T1 ──► T3, T4          (T1 says which of T3/T4 matters more)
T2 ──► T3, T4          (layout must be right before it is built on)
T3, T4 ──► T5 ──► T6
```

T0 and T1 are independent and can be done in either order. T2 is independent of
both. T5 and T6 are only meaningful once T3/T4 have landed, since they measure
the new kernels.

## Explicitly not in scope

- Decode throughput (paused; `63f94e5` is the baseline and must not regress).
- Speculative decoding, MTP, DFlash or any draft-verify path.
- Requantisation to a lower bpw, affine formats, or any lossy weight change.
- Vision execution.
- The matvec wave-tail sweep from the reference survey.
