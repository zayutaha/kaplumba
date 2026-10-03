# spec/kv-free-drafter — a drafter trained against THIS checkpoint

## Goal

Raise speculative-decode acceptance against the local `Gxma` EXL3 checkpoint
from the measured **0.883 accepted/round** to **>= 2.4 accepted/round**, which is
what 40 tok/s requires at a 60 ms target pass.

This replaces the released `gemma-4-31B-it-qat-assistant-4bit` drafter, which
`spec/mtp-40tps` T4 measured and rejected: its first proposal lands in the
target's top-5 only **63%** of the time, so depth-1 acceptance is 0.45.

## The arithmetic that forces this

```
11.834 GB of weights per decode token, hardware ceiling 250 GB/s
=> single-token decode can never exceed ~21 tok/s, and 40 tok/s would need
   473 GB/s. Speculation is not an optimisation here, it is the only route.
tokens_per_pass = tok_s * pass_s
   pass=89.5ms -> 3.58/pass      pass=60ms -> 2.40/pass
```

So the entire 40 tok/s target reduces to one number: **accepted tokens per
target pass**, currently 0.883.

## Why a KV-free drafter, not the released one

The released drafter consumes the target's shared K/V: one sliding K/V
(16 heads x 256 dim) and one full K/V (4 heads x 512 dim). That is
**~6 MB per training position**, which makes on-policy distillation across the
process boundary (Rust target <-> MLX trainer) impractical, and it is the
reason the acceptance number is hostage to a checkpoint pair we did not train.

A drafter we train ourselves can drop the K/V entirely and condition on the
target's **final-norm hidden state** (5376 x f16 = 10.75 KB), a ~600x
reduction per position, and it can be trained on the exact checkpoint that will
verify it.

This is sound: the LM head is a linear map from that same hidden vector to the
262,144 logits, so the hidden state provably contains everything needed to
predict the next token. Depth-1 acceptance is therefore learnable to ~1.0; depths
beyond that are genuine prediction and are what the training has to earn.

## Architecture (ours, so the kernel is ours)

```
inputs:  h_t   = target final-norm hidden at position t        (5376, f16)
         e_prev = embedding of the previously emitted token    (5376, f16)
concat:  x_t   = [h_t ; e_prev]                               (10752)
block:   RMSNorm -> Linear 10752->d -> GELU -> Linear d->d -> +residual   (L blocks)
head:    RMSNorm -> Linear d->262144                            (tied to input embed)
step:    x_{t+1} = [h_t ; embed(argmax/ sample of logits_t)]
```

`d = 1024`, `L` to be chosen (start 4). Parameter count is dominated by the
head: 1024 x 262144 = 268 M, which is unavoidable because the output vocabulary
is 262,144. Everything else is < 25 M.

## Non-goals / restrictions

1. **No quality loss.** Speculation stays verify-gated; the target's logits
   decide every emitted token. Greedy output must remain bit-identical to
   non-speculative decode.
2. **Coherence at temperature 1.0** is an acceptance criterion, not an
   afterthought. A fast path that produces fluent-but-wrong text fails.
3. **No Python in `crates/` or `scripts/`.** Training and export scripts live in
   scratch; only the engine stays Rust + Metal.
4. **One model process at a time**, `scripts/gpuq` discipline, per AGENTS.md §10.
5. **No KV cache quantization**, no multi-token parallelism, no quality knobs on
   the target.
6. `~/.kaplumba/models/Gxma/Gxma` stays untouched; all drafter artefacts go to
   `~/.kaplumba/drafters/`.

## Requirements

1. **R1 — hidden-state exporter.** A Rust tool runs the target over a text
   corpus and emits, per position: the final-norm hidden (5376) and the target's
   own greedy next K tokens. Format is a flat binary + index, ~10.75 KB/position.
2. **R2 — drafter training.** Train the architecture above with MLX on this
   box, teacher-forced on the target's own continuation. Loss is cross-entropy
   on the target's next tokens at every draft depth.
3. **R3 — inference kernel.** A GEMV/GEMM kernel for the trained drafter's
   shapes, validated bit-for-bit against a CPU reference, with sabotage evidence.
4. **R4 — acceptance gate.** Extend `mtp_acceptance` to run the trained drafter
   and report accepted/round. **Gate: >= 2.4.**
5. **R5 — distribution faithfulness.** At temperature 1.0 / top_k 64, the
   speculative output must be distribution-exact against non-speculative decode
   (TV distance + two-sample chi-square), and multi-turn text must be coherent.

## Acceptance criteria

- **AC1** measured accepted/round **>= 2.4** against real Gxma, greedy, over >= 200
  rounds on held-out prompts.
- **AC2** greedy output bit-identical to non-speculative decode.
- **AC3** T11 distribution faithfulness passes at temperature 1.0.
- **AC4** T12 coherence passes on multi-turn chat at temperature 1.0.
- **AC5** every new kernel path has sabotage evidence (AGENTS.md §3.4).

## Open questions

- **Q1** Can acceptance reach 2.4 at all? Depth-1 is learnable in principle
  (the head does it); depths 2+ are genuine prediction. If R2 plateaus below
  ~1.6, 40 tok/s is out of reach on this box by this route and the answer is a
  roofline, not more training.
- **Q2** Does a KV-free drafter lose so much context that depth-3+ collapses?
  Sliding-window layers mean the target hidden summarises only the recent
  window plus whatever the full-attention layers retained. Measured by the
  per-depth acceptance curve, not assumed.
- **Q3** Head cost. 268 M parameters in the head means the drafter is
  0.28 GB in bf16 even before quantisation, so a draft step streams ~0.5 GB
  (~2.5 ms). Acceptable only if it runs concurrently with the target pass.