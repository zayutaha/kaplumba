# spec/device-sampler — tasks

Each task is one commit. Status updated immediately after each.

- [x] **T0** — spec folder (`spec.md`, `plan.md`, `tasks.md`), plus
      `scripts/sampler_ab.sh` (interleaved A/B, same-config control arm,
      text-only hashing) and a copy of the pre-change binary at
      `$TMPDIR/opencode/dsampler/kaplumba-before`.
- [x] **T1** — `HeadPick` enum; `skip_softcap` derived from mode, not from
      buffer presence. Every existing caller passes `HeadPick::Argmax`.
      *Files:* `crates/engine/src/model.rs`, `sampler.rs`, `chat.rs`,
      `generate.rs`, `tests/perf_probe_host_ops.rs`,
      `tests/perf_probe_dispatch_census.rs`, new `tests/device_sampler.rs`.
      *Tests:* unit `only_argmax_with_staging_reads_the_raw_row` (4 cells);
      unit `SamplerParams` bound/validation (4 tests); GPU
      `sampling_reads_the_capped_row_and_argmax_does_not`.
      **Sabotage observed:** reverting `reads_raw_row` to `staging` fails the
      unit test, and fails the GPU test with
      `argmax=1499 sample=1499 ... reading the RAW logits`.
      **Temperature-0 byte-identity:** 7 runs across 3 prompts, generated text
      only, all hashes equal to `863bce5`'s binary.
      *Commit:* `spec/device-sampler: T1 HeadPick, skip_softcap from mode`
- [x] **T2** — `sample_topk_blocks` + `sample_draw` in
      `elements.metal` / `elements.rs`; `device ulong` RNG in
      `ForwardScratch`; `sample_candidates` + `rng_state` in
      `resident_bytes()`.
      *Files:* `crates/kernels/src/elements.metal`,
      `crates/kernels/src/elements.rs`, `crates/engine/src/model.rs`,
      new test `crates/engine/tests/device_sampler.rs`.
      *Tests:* (1) device top-64 **set and order** == `top_k_order`, exact,
      on adversarial rows (heavy ties, ±0.0, ±inf, extreme exponents);
      (2) RNG: the device `float(n) * 2^-53` reproduces the host's
      `next_f64() as f32` bit-for-bit over 3×10⁶ draws (integer test, no GPU);
      (3) TV distance host vs device below threshold over ≥10⁵ device draws,
      threshold measured at 0.06 (a correct sampler reads 0.026–0.036);
      (4) `top_k` bound: derived from the kernel geometry, swept at every
      `top_k` against it, and refused at the call rather than silently wrong.
      *Result:* 6/6 GPU tests + 15/15 lib tests pass.
      **Six defects found and fixed**, four CPU-side and two GPU-side; all six
      sabotage-verified. The two GPU ones were silent — no fault, no NaN:
      `sample_draw` read its params from a null binding (`dispatch_groups`
      binds the param struct last, not in the middle), and the merge kept the
      *worse* half of the sorted pair. See `implementation_notes.md`.
      *Commits:* `454fb0e`, `f79722c`, `c0e5a6c`.
- [ ] **T3** — wire the sampler into the pipelined route and ungate the
      pipeline: `chat.rs:388`, `generate.rs:150`/`:269`, `generate_gpu_greedy`
      → `generate_gpu_pick(pick: HeadPick)`, first token from
      `read_host_token(0)`, `decode_wall_tok_s` added beside (not replacing)
      `decode_tokens_per_second`.
      *Files:* `crates/engine/src/chat.rs`,
      `crates/engine/src/generate.rs`, `crates/engine/src/model.rs`,
      `crates/engine/tests/perf_probe_host_ops.rs`,
      `crates/engine/tests/perf_probe_concurrency_dependency_graph.rs`,
      `crates/engine/tests/multi_turn_chat.rs`.
      *Tests:* pipelined vs barrier decode agree at temperature > 0 (the
      existing multi-turn equality test, generalised off temperature 0);
      `--temperature 0` byte-identical ≥3 runs, text only; one sampling
      dispatch per step; the rewritten probes assert the new shape.
- [ ] **T4** — incremental `Detokenizer`.
      *Files:* `crates/engine/src/tokenizer.rs`.
      *Tests:* (1) forced retract → `InvalidPrefix` surfaced as
      `TokenizerError::Operation`, not swallowed; (2) per-push cost flat in
      n (measure 16 / 256 / 2048 pushes); (3) `tests/multi_turn_chat.rs`
      still green; (4) long greedy generation text-equal, ≥3 runs.
- [ ] **T5** — validation and report. ≥6 interleaved A/B pairs at
      temperature 1.0 with a same-config control arm; temperature 0 before
      and after; GPU-idle time at temperature 1.0 before and after.
      Record in `implementation_notes.md`.