# spec/dflash-sampling — implementation notes

Append as work proceeds (AGENTS.md §5).

---

## 2026-10-11 — investigation findings (before implementation)

### The reference rejection rule, copied from z-lab so it is not re-derived

`/private/var/folders/tp/wz3877_s1z756p1kcsz3_d4h0000gn/T/opencode/zlab/dflash/dflash/model.py:82-126`:

```python
def _sampling_probs(logits, temperature, top_p=1.0, top_k=0):
    scores = logits.float() / temperature
    vocab_size = scores.shape[-1]
    if 0 < top_k < vocab_size:
        scores, indices = torch.topk(scores, top_k, dim=-1)
    else:
        indices = None
    probs = torch.softmax(scores, dim=-1)
    if top_p < 1.0:
        sorted_probs, order = probs.sort(dim=-1, descending=True)
        keep = sorted_probs.cumsum(dim=-1) - sorted_probs < top_p
        sorted_probs = sorted_probs * keep
        probs = torch.zeros_like(probs).scatter(-1, order, sorted_probs)
        probs = probs / probs.sum(dim=-1, keepdim=True)
    if indices is not None:
        probs = torch.zeros_like(logits, dtype=probs.dtype).scatter(-1, indices, probs)
    return probs

def _rejection_sample(draft_tokens, target_probs, draft_probs, draft_indices=None):
    gamma = draft_tokens.shape[1]                                  # 15
    p = target_probs[:, :gamma].gather(-1, draft_tokens[..., None])[..., 0]
    if draft_indices is None:
        q = draft_probs.gather(-1, draft_tokens[..., None])[..., 0]
    else:
        q = (draft_probs * (draft_indices == draft_tokens[..., None])).sum(-1)
    accepted = ((torch.rand_like(q) * q) < p).to(torch.int32).cumprod(-1).sum(-1)[0].item()
    if accepted == gamma:
        return accepted, _sample_probs(target_probs[:, -1])[0]
    residual = target_probs[0, accepted].clone()
    if draft_indices is None:
        residual.sub_(draft_probs[0, accepted])
    else:
        residual.scatter_add_(0, draft_indices[0, accepted], -draft_probs[0, accepted])
    residual.clamp_min_(0)
    total = residual.sum()
    residual = torch.where(total > 0,
                           residual / total.clamp_min(torch.finfo(residual.dtype).tiny),
                           target_probs[0, accepted])
    return accepted, _sample_probs(residual[None])[0]
```

Notes that matter for the port:
- `keep = cumsum - sorted_probs < top_p` keeps the **top-1 token
  unconditionally** (its cumulative-before is 0 < top_p) and then every token
  whose *preceding* cumulative mass is still under `top_p`. This is the
  "inclusive nucleus" rule.
- `top_k` is applied **first**, on the temperature-scaled scores, and `top_p`
  is then applied **inside** the surviving k — so the two are not commutative.
  z-lab's caller passes `top_k` from the CLI default of 64 and `top_p` 0.95.
- The residual fallback `torch.where(total > 0, ..., target_probs[accepted])`
  handles the degenerate `p == q` case; without it the normalisation divides
  by zero.
- `draft_indices` is the DFlash2 (candidate-selector) path — irrelevant for
  our `DFlashDraftModel` checkpoint, which always passes `draft_indices=None`.

### How the drafter's own sampling is wired in z-lab

`dflash/model.py:283-291`: with `temperature > 0`, `dflash_generate` computes
`draft_probs = _sampling_probs(draft_logits, temperature, top_p, top_k)` from
the **softcapped** draft logits (`compute_logits` at `:588` applies
`tanh(x/30)*30`), then `block_output_ids[:, 1:] = _sample_probs(draft_probs)`.
So `q` is the truncated-and-renormalised drafter distribution, not a raw
softmax. Getting this wrong is the classic way to fake distribution-exactness.

### The engine's sampler, as it stands

`crates/engine/src/sampler.rs`:
- `SamplerConfig { temperature: f32, top_k: usize, top_p: f32 }`,
  `Default = { 1.0, 64, 0.95 }` (`:26-30`) — the chat CLI's defaults.
- `top_k_order(logits, top_k)` at `:149` — order-preserving u32 key trick
  documented at `:36-56`; the ordering is **logit descending, index ascending
  on ties** (`:65`).
- `device_top_k_limit(vocab)` at `:326` — the kernel's own geometry bound, not
  a restated constant. `top_k` at or above the vocabulary is served by the host
  path.
- `SamplerError::{InvalidLogits, InvalidTemperature, InvalidTopK, InvalidTopP,
  DeviceTopKTooLarge}` at `:4-15`.

### Related bug class already recorded in this repo

- `crates/engine/src/model.rs:4702-4707`: host writes into shared memory
  without waiting made greedy output depend on scheduling — "six runs of one
  614-token prompt produced two different answers". The RNG-order requirement
  R5 in this spec exists for the same reason: a draw order that depends on
  acceptance history makes output depend on history in a way no test will
  catch unless determinism is asserted directly.
- `crates/engine/src/sampler.rs:50-72` documents a device-sampler tie-breaking
  and non-finite-logit rule. If AC1 fails at all three `(temperature, top_p)`
  points, compare tie-breaking before suspecting the rejection rule.

### TO MEASURE

- [ ] T6 — TV distance and χ² p-value at (1.0, 1.0), (1.0, 0.95), (0.7, 1.0).
- [ ] Q1 — whether our `top_p` rule matches z-lab's "cumulative-before" rule.
- [ ] Q2 — rejection-decision stability under fp16-derived logits.
- [ ] T8 — the same AC1 numbers with a degraded drafter (acceptance << 1).

## Dead ends / abandoned approaches (recorded as they happen)

- *(none yet)*
