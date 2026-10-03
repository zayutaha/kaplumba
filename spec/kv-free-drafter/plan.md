# plan — kv-free-drafter

## Data flow

```
corpus text
   │  (Rust, `examples/export_hidden`)
   ▼
target forward, one position at a time
   │  hidden_t (5376 f16)  +  target's greedy tokens t+1..t+K
   ▼
~/.kaplumba/drafters/train.bin          (~10.75 KB / position)
   │  (MLX, scratch script)
   ▼
drafter training  →  ~/.kaplumba/drafters/kvfree.safetensors
   │  (Rust kernel, `affine4_gemv`-style bf16 GEMV)
   ▼
mtp_acceptance with --drafter kvfree      →  accepted/round  (the gate)
```

## Touchpoints

| path | change |
|---|---|
| `crates/engine/examples/export_hidden.rs` | **new** — walks the target over text, writes `hidden + next K tokens` per position |
| `crates/engine/src/model.rs` | `forward_with_hidden` already exists and is what the exporter uses |
| `crates/engine/src/drafter_model.rs` | new `KvFreeDrafter` alongside the affine-4bit one; same `forward(inputs, position) -> logits` shape so `mtp_acceptance` can drive either |
| `crates/kernels/src/affine.{metal,rs}` | bf16 dense GEMV + RMSNorm + GELU, reusing the existing buffer plumbing |
| `crates/engine/examples/mtp_acceptance.rs` | accept a `--kind affine4|kvfree` flag so the gate runs against either drafter |

## Why the exporter is the critical path

Everything downstream — training, acceptance, 40 tok/s — depends on having
target hidden states paired with the target's own continuation. It is also the
cheapest thing to get right and the easiest to get silently wrong: an exporter
that mis-pairs `hidden_t` with `token_{t+1}` produces a drafter that trains to
~0% and looks like a modelling failure. It therefore ships with a check that
re-runs the target over its own emitted tokens and asserts the export's
continuation matches.

## Sequencing

1. **T1 exporter** + pairing check.
2. **T2 corpus** (download to scratch, ~50-200 M tokens is far more than a 268 M-head
   model needs; start small and grow).
3. **T3 drafter model + CPU reference + training loop** (MLX, scratch).
4. **T4 first training run**, report the per-depth acceptance curve on held-out
   data **without** the GPU kernel — acceptance can be measured in MLX first,
   which is far cheaper than building a kernel to find out.
5. **T5 gate**: if held-out accepted/round >= 2.4, build the kernel; else stop
   and report the roofline (Q1).
6. **T6 M>1 verify kernel** — only after T5 passes. This is the other half of
   40 tok/s and the engine has no M>1 path at all today.
7. **T7–T8** exactness + coherence.

## Risks

- **Training on one GPU alongside an 11.6 GiB resident model.** The exporter and
  the trainer cannot hold the target at the same time. Sequence them: export
  first, free the target, then train. This is the same constraint AGENTS.md §10
  already documents.
- **Head dominance.** 268 M of 290 M parameters are the vocabulary head. If
  training is head-dominated and the body learns nothing, acceptance will be
  ~= the target head's own accuracy (i.e. depth-1 = 1.0, depth>=2 = 0). That
  is a specific, checkable failure and is the first thing to look at in T4.
- **Ceiling.** Even at depth-1 = 1.0 and depth-2 = 0.7, depth-3 = 0.4, mean is
  ~2.1 — short of 2.4. The curve has to be checked, not assumed.

## Requirement → task map

| requirement | tasks |
|---|---|
| R1 exporter | T1, T2 |
| R2 training | T3, T4 |
| R3 kernel | T6 |
| R4 gate | T5 |
| R5 faithfulness | T7, T8 |