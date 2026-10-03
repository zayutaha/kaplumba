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

- [ ] **T5 — GATE.** held-out accepted/round >= 2.4?
      **If no: stop and report the roofline.** Do not build the kernel.

- [ ] **T6 — inference kernel.** bf16 dense GEMV + norms for the drafter's
      shapes. Bit-exact vs CPU reference. Sabotage evidence.

- [ ] **T7 — exactness.** greedy speculative output bit-identical to
      non-speculative decode.

- [ ] **T8 — coherence + distribution.** T11/T12 from `spec/mtp-40tps`, at
      temperature 1.0 / top_k 64.