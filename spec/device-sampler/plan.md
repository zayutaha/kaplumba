# spec/device-sampler — plan

Design input: `spec/decode-ceiling/findings/A4-sampler.md`. Read it; it is
finished and this plan is its transcription, not a re-derivation.

## Architecture and data flow

Per decode step, inside the step's own command buffer, after the LM head
GEMV:

```
scratch.capped  (262,144 fp16, softcapped)
        |
        |  D1  sample_topk_blocks      groups=64, threads=256
        |      64 slices x 4096 elements -> 4096 u32 argmax_keys
        v
scratch.sample_candidates (4096 u32 = 16 KiB)
        |
        |  D2  sample_draw             groups=1, threads=256
        |      1. global top-64 of the 4096 (same bitonic routine)
        |      2. sort 64 descending by (value_key, index)
        |      3. thread 0: softmax, top-p walk, inverse-CDF walk
        |      4. thread 0: advance device RNG, draw, write 2 words
        v
token_ids_decode[1-slot]   (next step's input)
host_tokens[step_idx]      (read by read_host_token)
```

Two dispatches is the minimum: Metal has no grid-wide barrier, and 64
threadgroups cannot see each other's candidates.

## File/module touchpoints

| path | current | change |
|---|---|---|
| `crates/kernels/src/elements.metal` | 25 kernels, no softmax/RNG | + `sample_topk_blocks`, `sample_draw` after `argmax_keys_to_token` (`:610`); reuse `value_key` (`:531`), `argmax_key` (`:545`), `sortable16` (`:527`) verbatim; factor the bitonic top-k into one `static inline` shared by D1 and D2 |
| `crates/kernels/src/elements.rs` | pipeline fields at `:224-226`, built at `:321-323` | + 2 pipeline fields, + 2 `pipeline(...)` calls, + `encode_sample_topk_blocks` / `encode_sample_draw` following `encode_argmax` (`:849-905`); params via `params_bytes` (`:1568`) |
| `crates/engine/src/model.rs` | `forward_chunk` (`:4053`) takes `next_token_out`/`host_token_out`; `encode_lm_head` (`:4415`) derives `skip_softcap` (`:4475`) from buffer presence | `enum HeadPick { Argmax, Sample(SamplerParams) }`; replace the two `Option<&Buffer>` with one `pick: HeadPick`; `skip_softcap = matches!(pick, HeadPick::Argmax)`; `ForwardScratch` + `sample_candidates` (16 KiB) + `rng_state` (8 B), both in `resident_bytes()` |
| `crates/engine/src/sampler.rs` | `SamplerConfig { temperature, top_k, top_p }`, `Sampler::sample` on the host | unchanged as the reference; expose `SamplerConfig` as the device params payload. `top_k > 1024` is the caller's fallback trigger |
| `crates/engine/src/chat.rs` | `let pipelined = self.greedy && env…` (`:388`); `argmax_logits(&logits)` first token (`:418`) | gate becomes the env check alone; first token becomes `model.read_host_token(0)` when sampling on device; `forward_step_pipelined` gains the `pick` argument |
| `crates/engine/src/generate.rs` | `if temperature == 0.0 { generate_gpu_greedy }` (`:150`, `:269`) | delete both short-circuits; `generate_gpu_greedy` → `generate_gpu_pick(model, …, pick: HeadPick, …)`; the non-pipelined branch in `generate_gpu` keeps the host sampler as the fallback arm |
| `crates/engine/src/tokenizer.rs` | `Detokenizer::push` (`:202-235`) replays every id | hold `DecodeStream` in `new()`, one `step(id)` per push, O(1) amortised |
| `crates/engine/tests/perf_probe_host_ops.rs` | asserts the *pre-change* shape (chat.rs gate mentions `self.greedy`, `generate.rs` temperature-0 dispatch, `forward_buffer_into_vec` in the sampling loop, `if next_token_out.is_some() || host_token_out.is_some() {`) | these are characterisation probes of the gap this spec closes; they are rewritten to assert the *post*-change shape, keeping the probe useful |
| `crates/engine/tests/perf_probe_concurrency_dependency_graph.rs` | same, two assertions | as above |
| `scripts/sampler_ab.sh` | — | new: interleaved A/B with a same-config control arm, text-only hashing |

## Interfaces

```rust
// crates/engine/src/model.rs
#[derive(Clone, Copy, Debug)]
pub struct SamplerParams { pub temperature: f32, pub top_k: u32, pub top_p: f32 }

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HeadPick {
    Argmax,
    Sample(SamplerParams),
}
```

```metal
struct SampleParams {
    uint count;        // vocab
    uint top_k;
    uint candidates;   // groups * per_slice, the length of the candidate array
    float temperature;
    float top_p;
    uint step_idx;
};
```

Buffers, both in `ForwardScratch`:

- `sample_candidates`: `max(top_k * GROUPS, ...)` u32, 16 KiB at the default
  (`64 groups x 64 ranks`). Sized from `top_k` and `vocab`, **not** from
  context length.
- `rng_state`: 8 bytes, `StorageModeShared` (the host seeds it once per turn
  with one `contents()` write).

## The one correctness prerequisite (T1)

`model.rs:4475`:

```rust
let skip_softcap = (next_token_out.is_some() || host_token_out.is_some())
    && !env_flag("KAPLUMBA_FORCE_SOFTCAP");
```

The moment the sampler reuses the pipelined route it passes a host output
buffer, `skip_softcap` becomes true, and the pick reads **`scratch.logits` —
raw, un-softcapped**. `softcap` is `tanh(x/30)*30`, monotonic so argmax is
unaffected, but it *compresses the distribution*, so sampling the raw row
draws from a different distribution than the host sampler it replaces.

Nothing would catch it. Not tok/s, not a smoke test, not `--metrics`. So
T1 makes `HeadPick` the sole input to `skip_softcap` and pins it with a test
that the ablation *changes* sampled output. That test is only claimed after
the sabotage has been observed failing.

## Sequencing

```
T1  HeadPick + skip_softcap from mode            (no kernel; correctness gate)
 └─ must pass the sabotage test before T2 starts
T2  sample_topk_blocks + sample_draw kernels      (no wiring; unit + GPU tests)
T3  wire into chat.rs + generate.rs, ungate      (only after T1 is proven)
T4  Detokenizer incremental                        (independent of T1-T3)
```

T4 is independent of T1-T3 and touches a different file; it is last because
it is the smallest and its measurement is at 2048 tokens, which is the most
GPU time to consume.

## Risks and fallbacks

| risk | mitigation |
|---|---|
| `exp` differs from host | TV-distance test, not equality. Documented in the commit message. |
| `exp` differences cascade and a 2048-token answer drifts | expected and accepted; always between two near-equal candidates inside the truncated top-p set |
| the device RNG advancing twice in a step | `run_head` is true only for the last prefill chunk and every decode step (`model.rs:4037`); a test asserts one sampling dispatch per step |
| `decode_max_dispatches_per_buffer() = 50` rolling a command buffer *between* D1 and D2 | Metal orders dispatches within a command buffer, and `commit_terminal` ends the buffer, so a roll submits D1 before D2 to the same serial queue |
| `top_k` outside the device bound | `top_k > 1024` or `top_k >= vocab` falls back to the host sampler with `KAPLUMBA_NO_PIPELINE`-style non-pipelined loop (R7) |
| shared memory | 16 KiB (`threadgroup uint keys[4096]`) + 256 B of weights, against the 32 KiB per-threadgroup budget |
| the greedy path drifting | every existing caller passes `HeadPick::Argmax`, whose emitted code is today's code; A1 is a byte-identity check |
| load-time cost of two extra pipelines on the greedy path | measured with `--metrics`'s `load_ms`; acceptable because it is one-off |

## Requirement → task map

| req | task |
|---|---|
| R1, R2 | T1 |
| R3, R4 | T2 |
| R5 | T3 |
| R6 | T4 |
| R7 | T3 |
| R8 | T2 |