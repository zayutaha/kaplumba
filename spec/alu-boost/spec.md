# `spec/alu-boost` — spec

## Goal

Find out whether the 3 bpw EXL3 decode GEMV is limited by the ALU or by its
memory access pattern, and identify a change that moves it toward the box's
measured memory bandwidth. Report the split as measured numbers, and ship any
lever that is bit-exact.

## Non-goals / restrictions

- **No quality loss.** Any lever that changes the greedy output is out of scope
  for the shipping default; it may only be reported as a rejected candidate.
- **No model.** Every measurement here is model-free and takes no GPU lock.
- **No Python.**
- Not a rewrite: the deliverable is a decomposition plus, at most, a loop-shape
  change inside an existing kernel.
- Do not quote another spec's numbers as evidence for this one.

## Requirements

1. Measure the DRAM ceiling of the harness itself, so "the box can do X" and
   "this harness can do X" are not confused. (Satisfied: `arm_seq`, and
   `probe_bw_250gb` re-run today for the box's own figure.)
2. Decompose the shipping ring access pattern into: pattern alone, + extraction,
   + decode — with identical bytes, identical loop structure, and one variable
   changed at a time.
3. All arms must be measured **interleaved**, with an explicit control arm whose
   drift across rounds is printed.
4. Verify the transcribed decode against an independent implementation before
   reporting any timing.
5. Any claimed speedup must be shown **bit-exact by comparing kernel outputs
   bit-for-bit**, not argued from the absence of a reassociation flag.
6. Record negative results with their numbers.

## Acceptance criteria

- `cargo build --release -p kaplumba-engine --example alu_boost` succeeds.
- The run prints `[check] decode transcription verified` and
  `[check] arm_decode vs arm_decode_u4 ... 0 differ`.
- `spec/alu-boost/findings.md` contains the decomposition table, the bit-exact
  unroll ratio from ≥2 runs, the fp32 issue ceiling, and the negatives.
- The GPU is left free and no model process is started.

## Open questions

- **Do the integer and floating-point pipes overlap on Apple9?** The `p2_mix*`
  arms were meant to answer this and are void (they scale as `n²`). A corrected
  generator is needed before the shipping decode can be rebalanced across pipes
  — this matters, because the decode is 1 IMAD + 1 LOP3 + ~1.5 SHF + ~1 AND per
  code against ~4 float ops, and which side is binding is unknown.
- The 4-deep unroll is measured on a synthetic replica of the ring, not on the
  shipping kernel. It must be measured in-engine before it is claimed as a
  decode-tok/s number.
