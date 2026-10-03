# spec/device-sampler — device-side sampler

## Goal

Recover the ~6.7 ms/token of GPU idle that the shipped default
(`--temperature 1.0`) spends on a host-side sampler and a full
barrier-per-token loop, by moving the sampler onto the GPU and ungating the
1-deep decode pipeline so it is no longer reserved for temperature 0. At the
end, `--temperature 1.0` decode should cost the same per step as
`--temperature 0` decode plus the ~0.03-0.06 ms of the two sampling
dispatches, and `--temperature 0` output must be byte-identical to today.

## Why

`crates/engine/src/chat.rs:388` gates the pipelined decode loop on
`self.greedy`, which is only set at temperature 0. So:

- temperature 0 → 1-deep pipelined decode, token picked on device
  (`head/argmax` + `head/argmax_token`), host reads one `u32` per token.
- temperature > 0 → barrier-per-token loop: `forward_buffer_into_vec` →
  `commit_and_wait` (`model.rs:3848`) → 512 KiB logits readback to host →
  fp16→f32 widen of all 262,144 values → `Sampler::sample` (2.265 ms
  measured at width 262,144).

The programme measured 14.37 tok/s at temperature 1.0 vs 16.05 at
temperature 0 on the same binary, i.e. 7.3 ms/step of pure host stall.
`spec/decode-ceiling/findings/A4-sampler.md` establishes that this is the
largest remaining win and that the kernel side has nothing left
(0.28 ms addressable at temperature 0).

## Non-goals / restrictions

- **The temperature-0 (greedy) path must not change.** Its generated text is
  byte-identical today and stays byte-identical. Every existing caller of the
  head passes `HeadPick::Argmax`; the emitted code for `Argmax` is the code
  that exists now, character for character where it matters.
- **No KV cache quantisation. No quality loss** (AGENTS.md §8).
- **No change to decode numerics on the greedy path.** Not `exp`, not the
  softcap, not the argmax.
- **No claim of `--seed` bit-reproducibility across the host/device boundary.**
  Metal `exp(float)` is not `f32::exp`; see "Reproducibility" below.
- **No third decode loop.** `generate.rs` already has a pipelined branch; the
  design parameterises the existing one rather than adding a loop body to keep
  in sync.
- **No graph/tree-shaped prefill scratch** (AGENTS.md §8). The sampler adds
  two fixed-size buffers sized from `top_k` and `vocab`, never from context
  length, and they are counted in `ForwardScratch::resident_bytes()`.
- `spec/decode-ceiling/findings/` is read-only input.

## Requirements

- **R1** — `encode_lm_head` derives `skip_softcap` from an explicit
  `HeadPick` value, not from whether a host output buffer happens to exist.
  `HeadPick::Sample` must always read `scratch.capped`.
- **R2** — A test proves (a) `KAPLUMBA_ABLATE=head_softcap` **changes**
  sampled output at temperature > 0, and (b) leaves `--temperature 0` output
  byte-identical. The claim is not made until the sabotage has been observed
  failing (AGENTS.md §3.4).
- **R3** — Two Metal kernels, `sample_topk_blocks` and `sample_draw`, that
  together reproduce `Sampler::sample` on device for `top_k <= 1024`:
  - the top-`k` **set and order** must match `top_k_order` exactly, with no
    tolerance;
  - `top_p` is applied to the already-truncated `top_k` survivors, in that
    order, because `sampler.rs:110-128` does top-k before top-p;
  - the softmax sum, the top-p walk and the inverse-CDF walk are performed by
    a single thread in index order, so the sequential f32 sums are the same
    sums the host makes.
- **R4** — The per-step RNG is one `device ulong` in a `ForwardScratch`
  buffer, advanced exactly once per step, seeded host-side
  (`seed == 0 → 0x9e3779b97f4a7c15`, `sampler.rs:93`). The xorshift64 and
  `wrapping_mul` are bit-exact on device, and the `f64 → f32` demotion is
  reproduced with integer round-to-nearest-even rather than fp64.
- **R5** — `chat.rs` and `generate.rs` run the pipelined loop whenever the
  sampler is on device, not only when `greedy`. The host reads exactly one
  `u32` per token (`read_host_token`), never the row.
- **R6** — `Detokenizer::push` holds a `DecodeStream` instead of replaying
  every id. Cost per push must not grow with the number of tokens pushed
  (the old law was ≈ 8.5e-4 · n ms).
- **R7** — `top_k > 1024` falls back to the host sampler on the non-pipelined
  loop rather than building a 262,144-wide sort on device.
- **R8** — Total variation distance between the host and device sample
  distributions on a fixed logit row is below a stated threshold, over
  ≥10⁵ device draws. Exact equality is **not** asserted, because `exp`
  differs; the commit message and the notes say so.

## Reproducibility (explicit, per the parent task's constraint 2)

`--seed` reproducibility **degrades from bit-equality to
distribution-equality** across this change. Everything in `sample()` is
exactly reproducible on device — fp16→f32 is exact so the top-64 set and
order are exact, sequential f32 sums are reproduced by one thread in index
order, the RNG is bit-exact — *except* `exp`: Metal's `exp(float)` is
documented ≤ 2 ULP and is not macOS `expf`. For a fixed seed, host and
device can therefore pick a neighbouring candidate when the inverse-CDF
target lands within ~1e-7 relative of a cumulative boundary. Order 1e-4 per
token, always between two candidates inside the truncated top-p set with
near-identical probability.

Consequence: **two runs of the *same* device path with the same seed are
bit-identical** (that is asserted), and **host vs device with the same seed
is not** (that is measured as TV distance, not asserted as equality).

## Acceptance criteria

- **A1** — `--temperature 0`: generated text byte-identical to
  `863bce5`'s binary for the same prompt and seed, ≥3 runs, comparing
  generated text only (never the `--metrics` line: `load_ms`, `ttft_ms`,
  `decode_tok_s` vary per run).
- **A2** — `--temperature 0`: decode tok/s unchanged within the noise band
  of the same-session control arm.
- **A3** — Device top-k set and order == `top_k_order`, exact equality, on
  adversarial rows: heavy ties, ±0.0, NaN, ±inf, extreme exponents.
- **A4** — TV distance host vs device below threshold over ≥10⁵ draws on a
  fixed row.
- **A5** — `KAPLUMBA_ABLATE=head_softcap` changes sampled output at
  temperature > 0, demonstrated by observing the test fail when reverted.
- **A6** — `--max-new-tokens 2048` per-push detokenizer cost flat vs n
  (the pre-fix law was ≈ 8.5e-4 · n ms).
- **A7** — Temperature-1.0 GPU-idle time per token reported before and after,
  and ≥6 interleaved A/B pairs against a same-session control arm, medians
  and spread both reported. If the effect does not resolve above noise, that
  is the reported result.

## Open questions

None. §2 was executed by the parent session, which fixed the task order,
the validation protocol (≥6 interleaved pairs + control arm), the
text-only comparison rule, and the T1-first ordering. `A4-sampler.md`
closed the remaining design questions.

One judgement call recorded here rather than asked: `decode_tokens_per_second`
is **not** redefined. A new `decode_wall_tok_s` field is added beside it
instead, so the programme's existing 16.05 tok/s baseline does not silently
move under the A/B.