# tasks — `spec/decode-alu-fuse-18t`

One commit per task. Update this file **immediately** after each task with status,
commit hash, and the measured number — do not batch the updates at the end.

Measurement rules: `plan.md §0`. Every tok/s claim needs ≥16 interleaved pairs,
median + spread + control arm.

- [ ] **T0 — pin branch, build, recover the lost instrument, re-pin baseline.**
      *Files:* none (verification + one branch). Recover via
      `git show 33e8ead:crates/engine/examples/alu_boost.rs` (781 lines, on
      `agent/tensor-ops`, absent from HEAD).
      *Do:* (a) confirm HEAD `517bc4e` builds — **before** any edit; (b) branch
      off HEAD; (c) restore `alu_boost.rs`, confirm it compiles; (d) re-verify
      the `MEMORY.md` "master is broken" claim (at HEAD `pipeline_audit.rs` and
      `sweep.rs` both exist, so the note appears stale — record the truth);
      (e) re-pin the greedy baseline in-session with the control arm; (f) record
      the greedy hash for the AC1 prompt.
      *Test:* baseline is a median of ≥16 interleaved greedy runs with spread,
      and the same binary reproduces its own hash every run.
      *Commit:* `spec/decode-alu-fuse-18t: T0 pin branch and re-pin baseline`

- [ ] **T1 — re-measure the 4-deep tile-loop unroll. GATE: this can kill R1.**
      *Why:* `alu-boost/findings.md §2` claims `1.095×`/`1.052×` and
      `0 differ / 2560 words`, but the instrument is not in this tree, so the
      claim is currently unverifiable (`AGENTS.md §7`: verify a path before
      citing it).
      *Do:* run `arm_decode` vs `arm_decode_u4`, interleaved, with
      `CONTROL_repeat` and a CPU-computed checksum (`crux.md §6`). Reproduce
      **both** the bit-exactness check **and** a ratio above the noise floor.
      *If it does not reproduce:* stop R1, record the negative in `findings/`,
      and continue from T4. **A dead R1 is a legitimate outcome of this plan.**
      *Test:* the harness's own `[check] ... 0 differ` line printed, plus the
      control arm's drift printed. Report medians, not a single number.
      *Commit:* `spec/decode-alu-fuse-18t: T1 re-measure the tile-loop unroll`

- [ ] **T2 — port the unroll as `_tl4` into the kernels that actually run.**
      *Files:* `crates/kernels/src/linear_repacked.metal` — targets
      `:1992` `_colmajor_contig4` (the one `linear.rs:2091` says the engine
      reaches), `:2429` `_colmajor_2col_sk2` (o_proj/down_proj), `:434`
      `_fused2_2col` (mlp + full q/k), `:320` `_fused2`.
      *Constraint:* identical tile assignment and identical per-thread
      accumulation order (`alu-boost §3` — repartitioning tiles is `1.04–1.11×`
      and **not shippable**). **No threadgroup staging buffer** — that is `_u4`,
      which loses `180.5→173.7 GB/s` and is guarded by
      `probe_reduce_epilogue.rs:246` + `probe_cache_hierarchy.rs:745-752`.
      *Test:* (a) bit-exactness on real shapes vs the unshipped kernel — sabotage
      it by reordering one accumulator, watch `repacked_golden` fail
      diagnosably, revert; (b) `python3 scripts/gen_exl3_gemv.py` leaves
      `git diff --exit-code` clean (AC6 — the generator once deleted these kernels
      live); (c) A/B on the real model, ≥16 pairs.
      *Commit:* `spec/decode-alu-fuse-18t: T2 _tl4 tile-loop unroll`

- [ ] **T3 — end-to-end greedy A/B. This is where AC1 is decided.**
      *Do:* `scripts/gpuq try -- ./target/release/kaplumba --model
      ~/.kaplumba/models/Gxma-repacked --prompt "<AC1 prompt>"
      --max-new-tokens 128 --temperature 0 --metrics`, interleaved against
      T0's frozen binary, ≥16 pairs. Report median, spread, control drift.
      *Test:* greedy token ids byte-identical to T0's hash on every pair. A speedup
      with a changed hash is a **regression**, not a win.
      *Record:* the honest number, even if < `17.2`. A missed target with a
      trustworthy method is worth more than a met target with a leaky one.
      *Commit:* `spec/decode-alu-fuse-18t: T3 end-to-end greedy A/B`

- [ ] **T4 — fuse `linear_input_transform` into the GEMV `x` path.**
      *Files:* `crates/kernels/src/linear.metal:475` (`transform_simd:402`),
      consumed by the GEMVs above.
      *Constraint:* same `simd_shuffle_xor` order (1,2,4..64) or bit-exactness is
      gone. Do **not** narrow `raw` to fp16 — it is the shuffle-reduced GEMV
      output feeding the Hadamard, and rounding before it is the 1-ULP class that
      changed 43/128 greedy tokens (`attention.rs:176-182`).
      *Test:* (a) full fp16 `transformed` rows identical pre/post, for every
      layer shape; (b) greedy hash identical; (c) A/B ≥16 pairs. Expected
      `+0.15–0.25 tok/s`, which is **below the box's own spread** — so a null
      result here is expected and must be reported as null, not as `+0.2`.
      *Commit:* `spec/decode-alu-fuse-18t: T4 fuse input_transform`

- [ ] **T5 + T6 — open the pipelined loop to temperature > 0, and rewrite the
      two tests that assert the old gate. ONE commit.**
      *Files:* `crates/engine/src/chat.rs:388`
      (`let pipelined = self.greedy && std::env::var_os("KAPLUMBA_NO_PIPELINE")…`
      → `greedy || device_sampler_supported(config)`), `generate.rs:971` (read
      first — that is a *different* gate, `!margins.enabled`).
      *Already present, do not rebuild:* `model.rs:4004` `pick: HeadPick`,
      `:4057 read_host_token`, `:4071 device_sampler_supported`, `:4183`/`:4192`
      `sample_*` encoders, `elements.rs:343-344` pipelines.
      *T6 — the tests that will go red, to be **rewritten to assert the new
      meaning**, never deleted:*
      `tests/perf_probe_host_ops.rs:373 temperature_above_zero_disables_the_pipelined_path`
      (and its `:1099` sibling in
      `tests/perf_probe_concurrency_dependency_graph.rs`);
      `tests/multi_turn_chat.rs:229` currently runs both arms at temperature 0,
      which cannot discriminate this gate — generalise it off temperature 0 with
      a fixed seed.
      *Tests:* (a) temp 0 byte-identical, ≥3 runs, text only; (b) temp 1 device
      vs host TV distance `< 0.06` where a correct sampler reads `0.026–0.036` —
      **not** bit-identity, since the device draws from its own RNG state;
      (c) `KAPLUMBA_NO_PIPELINE` still forces the barrier arm; (d) a `top_k` past
      the candidate staging capacity falls back to the host sampler rather than
      mis-sampling (`model.rs:4064-4070`); (e) one `sample_draw` dispatch per step.
      *Also:* add `decode_wall_tok_s` beside — not replacing —
      `decode_tokens_per_second`, because the `--metrics` bracket at
      `generate.rs:383` **excludes the sampler**, which is the entire reason
      `16.05` is quoted and `14.37` is measured.
      *Commit:* `spec/decode-alu-fuse-18t: T5/T6 open the pipelined loop to temp>0`

- [ ] **T7 — attention split sweep, opt-in only.**
      *Do:* `splits {4,8,12,13,16}` on one real layer, one session, control arm
      (resolves Q2: `1.34×` isolated vs `+2.17%` end-to-end).
      *Gate:* ships **only** behind `KAPLUMBA_DECODE_SPLIT_RULE=workgroups`,
      default **OFF**. Not bit-exact (`434/476` elements differ by 1 fp16 ULP at
      8204) — the waiver is written into `spec.md §2`, not buried here.
      *Test:* flag-off arm is **exactly** AC1's hash; flag-on arm ≥ `17.6` median
      or it is not shipped either.
      *Commit:* `spec/decode-alu-fuse-18t: T7 attention split sweep (opt-in)`

- [ ] **T8 — residual fuse budget. Candidate-per-candidate, or not at all.**
      `post_attn_pre_ffn_transform` (`linear.metal:765`), `cache_append` fold
      (`MEMORY.md` `0.21 ms`), norm hoist. Budget `~0.3 ms` total; each dispatch is
      `3.5 µs` of *latency and ~0 work* (`dpk_own` `71.649→71.644` after removing
      240 dispatches). Baseline already banked: `121ff5b` cross-layer
      `finalize→input_norm`, `+0.09` pooled.
      *Rule:* each candidate A/B'd **alone**. Do not bundle — the prior
      cross-layer fusion is the reference for how small a real win looks
      (`+0.09 tok/s`, `+0.12/+0.069` in two 8-pair blocks).
      *Commit:* `spec/decode-alu-fuse-18t: T8 residual fuse candidates`

- [ ] **T9 — defects. No perf credit; correctness only.**
      **(a) CONFIRMED — `scratch.attended` undersized.** `model.rs:3229` sizes it
      `u16_elements(max_tokens, hidden)` = 5376 halves, but both attention
      epilogues write `q_heads * head_dim` (8192 sliding / 16384 full —
      `attention.metal:974`, `:927`), and `o_proj.in_features` is that same larger
      number (`model.rs:525`). **22,016 B/layer/token overrun**, surviving only
      because Metal allocates page-granular. Existing test:
      `tests/perf_probe_activations_traffic.rs:300
      attended_is_sized_hidden_but_written_q_dim`. *Sabotage:* shrink the buffer
      and confirm the test fails diagnosably before fixing it.
      (b) **UNSUBSTANTIATED — do not "fix".** `WHERE-THIS-STANDS.md:62` claims
      "no `_mul1` kernels exist" citing `linear.rs:485-489`; that range is
      `ishizuki_rate_of_name` (parses rate digits, accepts `1..=8`) and is
      unrelated. `decode_state_mul1` exists (`linear.metal:128`) and
      `Exl3Codebook::Mul1` is emitted by the dispatcher (`linear.rs:570`). The
      claimed `2.61×` cliff is unreproduced here. *Task:* find the real failing
      configuration and reproduce it, **or** record the claim as unsubstantiated.
      Fixing a kernel family that already exists is the error to avoid.
      (c) **Note only — no work.** `scratch.normed` is a dead store on the fused
      decode path (`pretransformed = true` always ⇒ the GEMVs bind `transformed`
      and skip the `input_pipeline` dispatch, `linear.rs:2716-2725`,
      `:2513-2522`); `scratch.proj` is never written on decode at all
      (`model.rs:1983-1995`). Together `0.005%` of the stream. Belongs to a
      memory tidy-up, not here.
      *Test:* (a) green + sabotage-verified; (b) `findings/R6-mul1.md` written,
      either way; (c) AC5 `phys_footprint ≤ 16.5 GiB`, 0 swap, via
      `/usr/bin/time -l` — **not** `peak_resident_pages_bytes`, which is
      `ru_maxrss` and understates by 5.5 GiB.
      *Commit:* `spec/decode-alu-fuse-18t: T9 scratch.attended sizing`

## Open, not in this spec

* **The `~1.6×` micro→production gap** (`203` vs `125–150` GB/s for the *same*
  kernel). `~+30%` if closed — **10× any instruction win in this spec**, and
  nobody has done it. If T2/T3 come in flat, this is the next spec, not more
  unrolling.
* **MTP / speculation** — the only route past the pins (`1.63` accepted/pass →
  `20 @16k`), explicitly ruled out by the user.