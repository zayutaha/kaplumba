# spec/dflash-sampling — distribution-exact DFlash at temperature > 0

## Goal

Make DFlash speculative decoding **distribution-exact** at any sampler
configuration the chat CLI uses — in practice `temperature 1.0, top_k 64,
top_p 0.95` (`crates/engine/src/sampler.rs:27`) — so that speculative and
non-speculative runs sample from the same distribution and produce
statistically indistinguishable text.

`spec/dflash-correctness` delivers greedy. Greedy is only lossless because
`argmax` commutes with the verify step; at `temperature > 0` the drafter
proposes from its own distribution and the verifier must accept or reject with
the exact probabilities that make the marginal output distribution equal to
the target's. That is the whole content of this spec.

## Non-goals / restrictions

1. **Not a greedy re-implementation.** Everything greedy is already done and
   must not change: this spec adds a branch, and the greedy branch's tests
   (`dflash_equivalence.rs`) must keep passing untouched.
2. **No distributional fudge.** "Accept the draft when it matches" or "always
   take the target's token at the first mismatch" are both biased and both
   rejected. The acceptance test must be a two-sample statistical test with a
   stated p-value, not an eyeball comparison.
3. **No change to `Sampler`'s own semantics.** `temperature/top_k/top_p`
   truncation and the RNG stream must mean exactly what they mean today; the
   speculative path has to converge on the *same* distribution the
   non-speculative path samples from, not define a new one.
4. **No Python in `crates/` or `scripts/`** (AGENTS.md §8).
5. **No MTP drafter work.** The MTP path stays as it is.

## What the reference does, precisely

z-lab `dflash/model.py:82-126`, the piece `spec/dflash-correctness` does not
need:

```python
def _sampling_probs(logits, temperature, top_p=1.0, top_k=0):
    scores = logits.float() / temperature
    if 0 < top_k < vocab:                 # 1. top_k on the scaled scores
        scores, indices = topk(scores, top_k);  probs = softmax(scores)
    ...                                  # 2. top_p inside the survivor set
    keep = sorted_probs.cumsum(-1) - sorted_probs < top_p
    probs = probs * keep; probs /= probs.sum()

def _rejection_sample(draft_tokens, target_probs, draft_probs, draft_indices=None):
    gamma = draft_tokens.shape[1]                       # 15
    p = target_probs[:, :gamma].gather(-1, draft_tokens)
    q = draft_probs.gather(-1, draft_tokens)
    accepted = ((rand_like(q) * q) < p).cumprod(-1).sum()
    if accepted == gamma:
        return accepted, sample(target_probs[:, -1])   # bonus from the target
    residual = clamp_min(target_probs[accepted] - draft_probs[accepted], 0)
    residual /= residual.sum()                          # normalised over the vocab
    return accepted, sample(residual)
```

This is Leviathan et al.'s speculative sampling. The invariants that make it
exact:

- `q` (drafter) must be the distribution the drafter **actually sampled its
  draft token from** — not the drafter's raw softmax. With `top_k` truncation
  in play, `q` is the truncated-and-renormalised drafter distribution.
- `p` is the target's truncated-and-renormalised distribution at the same
  position, with the **same** `top_k` / `top_p` applied.
- The residual `max(0, p − q)` is only used at the first rejected position.
- The **bonus** token at `accepted == gamma` is sampled from the target's
  distribution at position `gamma` — never from the drafter.

## Requirements

- **R1 — drafter sampling distribution.** The drafter must sample each of the
  15 draft positions from `_sampling_probs(draft_logits, temperature, top_p,
  top_k)` where `draft_logits` are the **softcapped** logits
  (`tanh(x/30)*30`, per `final_logit_softcapping 30.0`). Recording the token
  alone is not enough; `q_t` for the drafted token must also be retained
  because R2 needs it.
- **R2 — rejection rule.** Per position `i` in `0..15`, accept the draft while
  `uniform_i * q_i < p_i`, stopping at the first failure (a single product,
  not independent per-position acceptance). `p_i` is the target's sampling
  probability of the drafted token, `q_i` the drafter's, both under the same
  truncation.
- **R3 — residual token.** On first rejection at position `a`, sample the
  emitted token from `max(0, p_a − q_a)` renormalised. On full acceptance,
  sample the bonus from `p_15`, the target's distribution at the 16th position.
- **R4 — the truncation must agree.** The drafter's and the target's `top_k` /
  `top_p` must be computed identically: same `temperature`, same `top_k`
  applied to the scaled scores **before** `top_p`, same `top_p` "keep while the
  running sum before this token is `< top_p`" rule. Our engine's
  `Sampler`/`top_k_order` (`sampler.rs:149`) is the definition; this spec
  reuses it, it does not fork it. Any place the two paths disagree is a bug
  with a measurable signature (a two-sample test that fails only at
  `top_p < 1.0`).
- **R5 — RNG.** One uniform draw per drafted position plus one for the bonus
  or residual. The draw order must not depend on acceptance (or the RNG stream
  becomes history-dependent, which is exactly the class of bug
  `model.rs:4702-4707` records: "skipping this made greedy output depend on
  scheduling: six runs of one 614-token prompt produced two different
  answers").
- **R6 — EOS and stop conditions.** A stop token inside the drafted window
  truncates the accepted count exactly as it does in the greedy path; the
  positions after it are counted as rejected, never emitted.
- **R7 — interaction with filters/penalties.** Repetition penalties,
  banned strings and logit masks advance per accepted position. This is the
  part exllamav3 flags as the reason the speculative loop is serial
  (`generator.py:1334-1336`); our engine's per-position mask advancement must
  run inside the acceptance loop, not once per round.

## Acceptance criteria

- **AC1** Two-sample distribution test. For a fixed prompt and seed-free
  settings, draw N = 20 000 tokens from (a) non-speculative decode and (b)
  DFlash-speculative decode at `temperature 1.0, top_k 64, top_p 0.95`, and
  compare with total-variation distance and a two-sample χ² with a reported
  p-value. Pass: TV distance small and p above the stated threshold, at
  `temperature` ∈ {0.7, 1.0} and `top_p` ∈ {1.0, 0.95}.
- **AC2** Determinism. Same seed → same output, twice, with the same prompt;
  and the output does not change when the number of drafted positions is
  reduced (which changes how many RNG draws happen) beyond the documented
  near-tie exemption.
- **AC3** Greedy regression. `dflash_equivalence.rs` still passes untouched.
- **AC4** The rejection path is *exercised*: a test with a deliberately
  degraded drafter (weights perturbed so acceptance is well below 1) must
  still satisfy AC1. A test that only ever fully-accepts proves nothing about
  R3.
- **AC5** Throughput is not the goal here, but the sampling path must not
  regress tok/s by more than the cost of the draws; report median tok/s with
  spread, interleaved A/B.

## Open questions

- **Q1 — does our `top_p` implementation match z-lab's?** Ours is
  `sampler.rs`'s device path with the documented ordering rules; z-lab's is
  the four lines above. If they differ (for example on whether the first
  token is always kept), AC1 will fail at `top_p = 0.95` and pass at
  `top_p = 1.0`. That failure *localises* the mismatch, which is why AC1
  sweeps both values.
- **Q2 — fp16 vs fp32 probabilities.** z-lab computes `_sampling_probs` in
  `.float()`; our sampler works on fp16-derived f32 logits. Whether the
  rejection decision is stable at the fp16 noise level is measured, not
  assumed.
- **Q3 — is the residual distribution sampled over the full 262 144 vocab?**
  z-lab scatters the truncated `probs` back to full vocab and then does
  `clamp_min(target - draft, 0)` on the full row. Our engine's sampler only
  materialises the top-k candidates. Sampling the residual over the k
  candidates is equivalent *if* `p` and `q` are both zero outside that set,
  which the truncation guarantees — but the equivalence must be asserted, not
  assumed.
