# spec/dflash-sampling — plan

## Architecture and data flow

`spec/dflash-correctness`'s round becomes two branches at the same seam:
after the drafter's block pass produces 15 candidate tokens, the drafter now
also emits `q_1..q_15` (its sampling probability for each candidate), and after
the target's 16-position verify pass produces `p_1..p_15` plus `p_16`, a
rejection loop decides.

```
drafter block pass
   └─► draft_logits[0..16]  (softcapped)
        └─► sample from _sampling_probs(...)   ─► draft_tokens[1..15]
        └─► retain q_i = P_d(i-th draft token)            (R1)

target verify pass over 16 positions
   └─► logits[0..16]
        └─► P_t = _sampling_probs(logits, ...)            (R4, same truncation)

for i in 0..15:                       (R2 — serial, single product)
    if not (u_i * q_i < p_i): reject  (R5 — u_i drawn regardless)
        a = i  ─► residual = norm(max(0, P_t[a] - q_a)) ─► sample ─► emit  (R3)
        break
    else: emit draft_tokens[i]                       (R6 — EOS check per position)
if a == 15:  bonus ~ P_t[15]                          (R3)
```

The only new state is `q_1..q_15` (15 × f32) and the per-position masks that
R7 already requires. Nothing is added to the model weights.

## File / module touchpoints

| path | current behaviour | intended change |
|---|---|---|
| `crates/engine/src/dflash.rs` | `draft_block` returns `[u32; 15]` | returns `[u32; 15]` **plus** the 15 probabilities under the caller's `SamplerConfig`, sampled with a caller-supplied RNG |
| `crates/engine/src/generate.rs` | greedy acceptance only | `if sampler.temperature > 0` branch implementing R2/R3; the greedy branch is untouched |
| `crates/engine/src/sampler.rs` | `Sampler`/`SamplerConfig`/`top_k_order` | add a `probability_of(logits, config, token) -> f32` helper and a `sample_from_residual(p, q) -> u32`. **No change to existing sampling semantics** |
| `crates/engine/src/sampler.rs` | RNG stream | expose the draw order so R5 can be asserted |
| `crates/engine/tests/dflash_distribution.rs` | — | **new.** AC1's two-sample test |
| `crates/engine/tests/dflash_sampling_unit.rs` | — | **new.** R2/R3 unit tests on a fake `p`/`q` with hand-computed expectations |

## Interfaces

```rust
pub struct DraftBlock {
    pub tokens: [u32; 15],
    /// q_i: the drafter's sampling probability of tokens[i], under `config`.
    pub probs: [f32; 15],
}

impl DflashModel {
    pub fn draft_block_sampled(
        &self, last_token: u32, anchor: &[u16], position: usize,
        cache: &DflashKvCache, config: &SamplerConfig, rng: &mut Sampler,
    ) -> Result<DraftBlock, DflashError>;
}

impl Sampler {
    /// P(token) under `config` — top_k on scaled scores, then top_p, both renormalised.
    pub fn probability_of(&self, logits: &[f32], config: &SamplerConfig, token: u32) -> f32;
    /// Draw from norm(max(0, p - q)) restricted to the surviving candidate set.
    pub fn sample_residual(&mut self, p: &[(u32, f32)], q: &[(u32, f32)]) -> u32;
}
```

## Sequencing and dependencies

```
T1  Sampler::probability_of + sample_residual     ── pure, no model needed
T2  unit tests on hand-computed p/q               ── gates T3
T3  draft_block_sampled                           ── needs spec/dflash-correctness T5
T4  rejection loop in generate.rs                 ── needs T3
T5  per-position filter/penalty advancement (R7)  ── needs T4
T6  two-sample distribution test (AC1)            ── needs T4, T5
T7  determinism tests (AC2)                       ── needs T4
T8  degraded-drafter test (AC4)                   ── needs T6
```

**T1/T2 can land before `spec/dflash-correctness` finishes** — they are pure
sampler work with no model dependency, and they are the part most likely to
expose a `top_p` mismatch (Q1). Run them first.

## Risks and fallbacks

| # | risk | mitigation |
|---|---|---|
| 1 | `top_p` semantics differ from z-lab (Q1) | AC1 sweeps `top_p` ∈ {1.0, 0.95}; a failure only at 0.95 localises it to the truncation, not the rejection rule |
| 2 | fp16 logits make the rejection decision noisy (Q2) | measure the acceptance-rate distribution against an fp32 reference; if it is noisy, compute `p`/`q` in fp32 from the f32 logits (which we already have) and report |
| 3 | residual over the full vocab vs the k candidates (Q3) | assert the equivalence with a unit test on a constructed case, rather than assuming it |
| 4 | RNG order becoming history-dependent (R5) | AC2 asserts same-seed determinism and insensitivity to draft-length changes |
| 5 | per-position masks making the loop slow | R7 is required for correctness; AC5 bounds the regression |

## Requirement → task mapping

| requirement | tasks |
|---|---|
| R1 drafter sampling distribution | T3 |
| R2 rejection rule | T4 |
| R3 residual + bonus | T4 |
| R4 truncation agreement | T1, T6 |
| R5 RNG order | T7 |
| R6 EOS/stop | T5 |
| R7 filters/penalties | T5 |
| AC1 | T6 |
| AC2 | T7 |
| AC3 | (regression, no task) |
| AC4 | T8 |
| AC5 | T6 |
