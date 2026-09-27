# Plan: 1.18 -> 20+ tok/s

## Token budget, and who owns what

One token streams 12.06 GB and decodes 32.2e9 weights. Current 850 ms:

```
rms_norm      241 calls x 596 us   ~143 ms   measured (4 single-thread norms/layer × 60 + 1 final)
q/k/v norms   180 calls             ~??? ms   multi-row (32/16/4 threads), untimed
attention      60 calls             ~380 ms   inferred, Q3
linear gemvs  410 calls             ~220 ms   measured (7 per sliding × 50 + 6 per full × 10)
head            1 call              ~9 ms     measured
embed           1 call            0.17 ms     measured
                                   -------
                                    ~750+ ms  (measured total 793-850; gap is q/k/v norms + overhead)
```

Note: the q/k/v norms dispatch 32/16/4 rows respectively at head_dim width
(256 or 512), not the catastrophic 1-row × 5376-width shape. They are still
serial within each row but are not the dominant cost. The gap between the
measured total (793-850 ms) and the itemised sum (~750 ms) is likely these
norms plus command-buffer and residual-add overhead.

Stage 1 removes ~143 ms of the single-threaded norm cost. Stage 2 attacks the
~380 ms attention estimate. Stage 3 attacks the 220 ms of gemvs.

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

Note: `KAPLUMBA_ATTENTION_ONLY` does not exist yet — T2 would create it.

## Model architecture (decode shapes)

For the budget analysis, the actual model shapes matter:

- 60 layers: 50 sliding (index % 6 ≠ 5) + 10 full (index % 6 == 5)
- `q_heads = 32`, `hidden_size = 5376`, `intermediate_size = 21504`
- Sliding: `kv_heads = 16`, `head_dim = 256`, `sliding_window = 1024`
- Full: `kv_heads = 4`, `head_dim = 512`, full context
- `lm_head`: 5376 → 262144 at 6 bpw (Integer(6), MCG codebook)
- Layer weights: all 3 bpw (Integer(3), MCG codebook)

Per sliding layer at decode: 7 linears (q/k/v/o + gate/up/down)
Per full layer at decode: 6 linears (q/k/o + gate/up/down; V reuses K buffer)

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
