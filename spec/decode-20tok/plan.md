# Plan: 1.18 -> 20+ tok/s

## Token budget, and who owns what

One token streams 12.06 GB and decodes 32.2e9 weights. Current 850 ms:

```
rms_norm      301 calls x 596 us   ~179 ms   measured
attention      60 calls            ~380 ms   inferred, Q3
linear gemvs   60 layers           ~220 ms   measured
head            1 call              ~9 ms   measured
embed           1 call            0.17 ms   measured
                                   -------
                                    ~790 ms   (measured total 793-850)
```

Stage 1 removes ~560 ms of parallelism bugs. Stage 2 attacks the 220 ms of
gemvs. Stage 3 attacks the codebook inside the gemvs.

## Instrumentation to keep

- `KAPLUMBA_PROFILE_PHASES=1` — embed / layers / head split, and cpu_encode vs
  gpu_wait inside layers. This is what found that the CPU encodes 60 layers in
  under a millisecond and the GPU takes 783.
- `KAPLUMBA_LEGACY_GEMV=1` — A/B against the pre-lane-per-thread matvec.
- `KAPLUMBA_LINEARS_ONLY=1` — attribution; note it breaks output validity, so
  it cannot produce decode steps and only works for prefill timing.
- `examples/elem_bench` — per-kernel microsecond costs on the decode path.
- `examples/alu_peak` — per-op-class ALU peaks, calibrated against fp32 FMA.
- `examples/bench_gemv` — per-projection GB/s, global warmup plus best-of-5,
  because the first shape measured otherwise eats the GPU clock ramp and reads
  70% slow.

## Sequencing rationale

T1 before T7, so stage 2 is sized by a number rather than an inference. T12
before T16, because the LUT is the most promising idea in this spec and also
the least proven; spending stage 3 on it without the A/B would be faith, not
engineering. T18 exists because the honest outcome of this spec may be "11 tok/s
is the machine", and that is a legitimate result to record.

## The two dispatch decisions that protect prefill

Both new paths are selected on shape, so prefill keeps today's kernels:

- norms: threadgroup-per-row only when `rows < 32`
- attention: split-K only when `queries * q_heads < 1024`

Prefill is 8.4 tok/s at 134 tokens today and is gated at every stage (R9).

## Requirement mapping

| Requirement | Tasks |
|---|---|
| R1 parallel norms | T3, T4, T5 |
| R2 split-K attention | T7, T9 |
| R3 drop `accum[512]` | T8 |
| R4 codebook table | T12, T16 |
| R5 `half2` packing | T13 |
| R6 cooperative ring load | T14 |
| R7 per-kernel costs | T1, T20 |
| R8 output tolerance | T11 |
| R9 no prefill regression | T11, and every stage |
