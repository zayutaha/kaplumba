# Tasks — logit-distribution probe

- [ ] **T1** `crates/engine/src/logit_probe.rs`: `compare_rows`, `aggregate`,
      capture-file writer/reader, report text + JSON. No Metal.
      *Test:* in-module unit tests + `tests/logit_probe.rs`.
- [ ] **T2** Deterministic corpus (12 prompts, mixed domain) and synthetic
      ground-truth tests: constant logit shift ⇒ KL == 0; delta vs uniform ⇒
      KL == ln V; 4-token hand-computed softmax; rank/ties; NLL/PPL; capture
      round-trip.
      *Test:* `cargo test -p kaplumba-engine --test logit_probe`.
      **Each test sabotaged and observed failing** (AGENTS.md §3.4); failure
      messages recorded in `implementation_notes.md`.
- [ ] **T3** Example binary: `capture`, `compare`, `perturb`, `selfcheck`;
      self-check wired to the exit code.
      *Test:* control arm, `Gxma-repacked` vs itself ⇒ 0.
- [ ] **T4** `perturb` subcommand: APFS-clone one shard, negate selected
      `lm_head.svh` entries driven from a reference capture.
      *Test:* byte diff shows exactly `2 * |targets|` bytes changed.
- [ ] **T5** Baseline: `Gxma-repacked` vs `Gxma-repacked` (two processes) and vs
      `Gxma`. Interleaved where the runs are cheap.
- [ ] **T6** Detection arm: perturbed checkpoint vs reference; plus a
      small-perturbation arm. Cross-check KL against an independent brute force.
- [ ] **T7** `~/Downloads` loadability assessment and metadata shim.
- [ ] **T8** Acceptance band for a 2 bpw model, grounded in T5/T6; say plainly
      where the literature gives no defensible threshold.
