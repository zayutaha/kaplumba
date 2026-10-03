# tasks — kv-free-drafter

- [ ] **T1 — hidden-state exporter.** `examples/export_hidden.rs`: walk the
      target over text files, emit per position `hidden_t` (5376 f16) and the
      target's greedy `token_{t+1..t+K}`. Flat binary + index.
      *Test:* the pairing check — re-run the target over the exported
      continuation and assert it reproduces the exported tokens.

- [ ] **T2 — corpus.** Download a text corpus to scratch (~1 GB is ample for a
      first run). Must be ordinary prose/code, not degenerate repetition.

- [ ] **T3 — drafter model + trainer.** MLX (scratch). Architecture per
      `spec.md`. CPU reference forward first; assert MLX matches it before
      training a single step.

- [ ] **T4 — first training run.** Report the **per-depth** acceptance curve on
      held-out data, measured in MLX (no GPU kernel needed yet).
      *Watch for:* head-dominated collapse (depth-1 = 1.0, depth >= 2 = 0).

      ### Two failed attempts, recorded so neither is repeated

      **Attempt 1 — 1722 M parameters, 0.02 it/s.** A full `[262144, 5376]`
      embedding plus a `[1024, 262144]` head is 1.7 B parameters; Adam state for
      that is ~21 GB on a 24 GiB box. Not a compute problem — an optimiser-memory
      problem. Fixed by low-rank factorisation of both
      (`E = A@B`, `A: [vocab, 64]`, `B: [64, 5376]`; `logits = (h@U)@V`,
      `U: [1024, 256]`, `V: [256, vocab]`): **129 M params, 8.7 it/s, 400x
      faster.**

      **Attempt 2 — loss froze at exactly 12.476649.** Both low-rank factors were
      initialised to `mx.zeros`. The product `A@B` is zero, so the output is
      uniform, so the loss sits at `-log(1/262144) = 12.476649`; and the gradient
      with respect to `B` is proportional to `A`, which is zero, so neither factor
      can leave zero. A frozen loss sitting on `log(vocab)` is the signature of
      this, not of a hard optimisation problem. Fixed by random init.

      **Attempt 3 — training loss 1.24 but 7% held-out accuracy. THE REAL ONE.**
      The exporter labelled each example with the **corpus's** next token to save
      one target forward per token. That trains the drafter to imitate wikitext,
      not to imitate the model that will verify it: the target does not generate
      the corpus verbatim, so the label is a different function than the one being
      drafted for. Low training loss with poor held-out accuracy is exactly what
      a wrong target function looks like.

      **The label must be the target's own greedy token.** Rolled forward over the
      target's own continuation at ~90 ms/token. `TextModel::greedy_next_tokens`
      batches the head over a whole window (one 0.5 GB pass instead of 512 decode
      forwards) and is the documented route to scaling this export.

- [ ] **T5 — GATE.** held-out accepted/round >= 2.4?
      **If no: stop and report the roofline.** Do not build the kernel.

- [ ] **T6 — inference kernel.** bf16 dense GEMV + norms for the drafter's
      shapes. Bit-exact vs CPU reference. Sabotage evidence.

- [ ] **T7 — exactness.** greedy speculative output bit-identical to
      non-speculative decode.

- [ ] **T8 — coherence + distribution.** T11/T12 from `spec/mtp-40tps`, at
      temperature 1.0 / top_k 64.