# spec/dflash-round-30tps — correct rounds before a 30 tok/s claim

Supersedes the implementation proposals in `spec/dflash-30tps` and
`spec/dflash-30tps-v2`. Their measurements remain historical evidence; their
kernel explanations, token counts, and projected round budgets are not current
acceptance evidence. Reviewed against the working tree on 2026-10-13.

## Goal

Reach ≥ 30 tok/s median end-to-end speculative decode on the M4 Pro with
`~/.kaplumba/models/Gxma-repacked` and `~/.kaplumba/models/Drafter`, while
preserving the target's greedy output and sampled distribution. First establish
correct proposal sampling, trusted diagnostics, cache rollback, and a complete
round measurement. If the measured implementation misses 30 tok/s, report the
remaining gap; a diagnosis does not satisfy the performance acceptance criterion.
The latest implementation instruction (2026-10-13) authorizes implementing and
testing the reviewed plan, with the explicit goal of reaching 30 tok/s. Work
must proceed through the correctness and performance gates below. Existing
commits and forced-load observations remain historical evidence, not acceptance.
The plan first establishes verifier correctness and a measured optimistic speed
ceiling, then selects DFlash repair or an alternative proposal source. The
30 tok/s target remains unmet until complete generation is measured.

## What the audit established

1. **There is no product round loop.** `draft_and_verify` has no caller in
   `generate.rs` or `crates/cli`. Probe results cannot establish product speed.
2. **The sampled round had invalid proposal sampling; T1a repaired it.** Before
   `2f35201`, `draft_block` used
   `select_draft_tokens`, which always selects argmax. The sampled verifier then
   uses the softmax/top-k proposal probabilities. Acceptance `min(1,p(x)/q(x))`
   requires proposals actually drawn from q. For p=(0.6,0.4), q=(0.9,0.1), the
   deterministic proposal plus that verifier emitted (2/3,1/3), not p.
   T1a now draws proposals from retained q; its argmax sabotage fails. This
   does not certify full round correctness or support for CLI top-p/temperature.
3. **The head oracle has concrete diagnostic errors.** Its control splits the
   full output into `chunks(512)` and treats chunk index as row index, although
   each actual row has 262144 columns. Its draft comparison divides the host
   dot product by `head.in_features`, while its control does not. It calls the
   same buffer GEMM used by `draft_logits`, not the target's `encode_gemv` head,
   and asserts shapes but no numerical agreement. A passing test invocation
   therefore does not mean the control passed. Historical differences of
   40.6–44.8 cannot diagnose the production head until this oracle is repaired.
4. **Permutation plumbing already landed.** `LinearBufferRequest.perm_mode`
   exists and the drafter linears and tied head pass their respective modes.
   Commits `9a4cf60` and `6622031` record D1/D2 work. The drafter loader's
   default and completeness of the sweep still need checking; increased
   overlap alone does not independently prove a checkpoint layout.
5. **Distinct hidden rows do not localize the error to the head.** The recorded
   stage scan rules out identical-row collapse in that run. Position alignment,
   feature extraction, quantization, attention semantics, and learned drafter
   quality remain candidates until independent comparisons separate them.
6. **The round budget mixes shapes and measurement boundaries.** 156.71 ms was
   measured for M=32's 35 layer projections plus fc, not a complete M=16 draft
   with attention and head. The 606 ms and ~135 ms historical verify figures
   need explicit timing boundaries to establish whether head time is included.
   They cannot be added or transferred between shapes without that check.
7. **Exllamav3 review identified and tested an ignored head offset.** Buffer
   GEMM accepts `packed_offset` but `encode_groups` binds weights at zero.
   The real target head starts at byte 5321075520. On four identical nonzero
   input rows, the drafter buffer head disagreed with the production decode head
   on every argmax; capped-logit error reached 2.829250. A temporary control
   copying the same 1056964608 weight bytes to offset zero restored every argmax,
   with capped error ≤0.000488. Fix the binding offset, not checkpoint weights.
8. **The previous permutation/acceptance conclusions are contaminated.** With
   only that temporary head rebase, row-major/native taps/raw embeddings produced
   mean teacher-forced overlap .116777 and first-proposal overlap .553835 on the
   78-token prompt; colmajor produced .000003 mean. Old .0565/best .8479 values
   used the wrong head bytes and cannot diagnose the drafter's layout or quality.
   This is still a diagnostic comparison, not product accepted-prefix survival.
9. **Target verification had three decode-only correctness defects.** Fused QKV
   reused Q's input-sign vector for K/V, and fused gate/up reused gate's vector
   for up; the checkpoint vectors are not guaranteed equal. Their fast paths now
   require identical vectors and otherwise transform each projection separately.
   Separately, cross-layer finalize→next-input-norm produced layer-1 K/V errors
   up to 9.848 and max logit error 12.32; it is opt-in via
   `KAPLUMBA_USE_CROSS_LAYER=1` pending repair. With these fixes, three prompts
   match sequential greedy tokens at M=2/4/8/16. Maximum observed logit drift is
   1.646 at M=16; sampled distribution equivalence is still unproven and sampled
   speculative decoding must remain disabled until that gate passes.
10. **Ring rollback lacks exllamav3's preserved history.** Exllamav3 retains
    extra history beyond the attention window. A direct cache primitive test
    confirms our cursor retraction leaves an overwritten retained key at 1001
    instead of its original 101: position 1 is still needed by the next query at
    4 after speculative writes at 4/5 are rejected. Full round testing remains
    required.

Source paths, controls, raw observations and caveats are in `exllama_review.md`.

## Arithmetic and measurement model

Let n be the actual number of proposals. A block of B rows contains one anchor
and B−1 proposals. B=16 gives n=15, B=32 gives n=31. Two B=16 blocks that
re-anchor the second on the first's last proposal give **30 proposals**, not
31; any extra proposal needs an explicitly defined and verified source.

For a constant independent conditional acceptance a, the planning estimate is
`E[emitted] = sum(a^j, j=0..n)`, including one correction/bonus token. Real
acceptances depend on depth, context, and previous proposals: a mean probe
matrix overlap cannot be substituted for this a. Measure accepted-prefix
survival `Pr(K ≥ j)` and use `E[emitted] = 1 + sum(Pr(K ≥ j), j=1..n)` before
EOS/length truncation. Product throughput is **total emitted tokens / total
elapsed decode time**, not the average of per-round rates.

The old costs produce the following arithmetic only; these are **not measured
complete rounds**:

| assumed n | assumed cost | ceiling at a=1 | rate at a=.95 | a to beat 15.283 tok/s | a for 30 tok/s |
|---|---|---|---|---|---|
| 15 | 898 ms | 17.817 | 12.469 | .979134 | impossible |
| 31 | 1055 ms | 30.332 | 15.285 | .949989 | .999289 |
| 31 | 935 ms | 34.225 | 17.247 | .939527 | .991330 |

The earlier one-block break-even of .91 was incorrect. The probe's hardcoded
.919 threshold belongs to a different assumed budget and is not a ship gate.
The assertion that 30 tok/s always requires verify ≤350 ms is also incorrect:
the condition is `complete round seconds ≤ E[emitted] / 30`.

Under the current processed-anchor API, the correction/bonus token must be fed
through the target before the next round to create its K/V, logits, and taps.
That forward is part of the steady-state cost. Illustratively adding the old
65.432 ms plain step lowers the n=31, 1055 ms ceiling to **28.560 tok/s**.
This is an estimate, not a benchmark. An alternative unprocessed-anchor design
may combine that work with verification, but must explicitly change the cache
and position contract and prove equivalence first.

## Proposed decision sequence

1. **Measure the verifier before choosing a drafter.** Establish numerical and
   greedy equivalence for M=2/4/8/16 target rows against sequential forwards.
   Use known target continuations as ideal proposals and time the full head and
   body. These measurements establish an optimistic ceiling with zero drafting
   cost, rather than real generation speed. Report the current processed-anchor
   ceiling `(M+1)/T_verify` and proposed unprocessed-anchor ceiling `M/T_verify`
   separately; the former additionally omits the required correction forward.
2. **If every tested ceiling misses 30 tok/s, prioritize verification.** Repair
   a proven existing-path defect and repeat the same test. Changing drafter or
   forcing longer blocks cannot rescue an already failing ideal ceiling at
   those shapes. This finding does not rule out other shapes or new designs.
3. **If a ceiling has room, repair DFlash predictions independently.** Establish
   checkpoint compatibility and the first stage diverging from an independent
   reference. Audit layout, taps/fc, normalization, RoPE, masks and head. The
   observed .0565 overlap is diagnostic; it does not prove a weak checkpoint.
4. **Build complete rounds using an unprocessed anchor**, after cache/tap and
   equivalence gates pass. Include the anchor in verification so the correction
   token needs no separate forward. Select proposal lengths by measured cost
   and accepted-prefix survival, with a plain-step fallback.
5. **Evaluate alternatives only after the shared verifier gate.** Prompt/history
   lookup and a Gemma-specific EAGLE-3 checkpoint are conditional experiments,
   with separate sampling, memory and correctness gates. Selection is unresolved;
   this draft authorizes neither an automatic checkpoint replacement nor a new
   engine subsystem.

The corresponding total wall-time budgets are:

| mean newly emitted tokens per round | maximum complete round time for 30 tok/s |
|---|---|
| 4 | 133.33 ms |
| 8 | 266.67 ms |
| 12 | 400.00 ms |

These are arithmetic budgets. Published acceptance lengths and the current
probe's overlaps are not substitute measurements for this engine.

## Non-goals / restrictions

1. No quality loss, KV quantization, Python project code, or multi-user batching.
2. Greedy tokens must match plain decode, including near ties with the same
   tie-breaking rule. A <0.25 logit gap is diagnostic context, not an exemption
   from the quality requirement.
3. No fifth EXL3 body kernel family in this spec. A batched head may use an
   existing path only after correctness is established. A needed new kernel
   requires a separately approved scope amendment.
4. No change to the M=1 decode kernel; retain the ≥15.0 tok/s target and compare
   with an interleaved baseline rather than treating 15.283 as timeless.
5. `MRHS_MAX_QUERIES ≤32`, block size ≤32, bounded explicitly owned scratch
   independent of context length. KV growth remains subject to context policy.
6. Experimental speculation is opt-in until correctness and measured speedup
   pass. A slower implementation must not become the default.
7. One model process through `scripts/gpuq`, status before every model run;
   exit 75 means busy. Separate build targets for any worktrees.
8. No coverage claims without observing a diagnosable failure after sabotage.
   Shape checks, printed metrics, and helper tests do not prove the round path.
9. Implementation and testing are authorized. Preserve the staged correctness
   gates, one-task commits, sabotage evidence, and measured acceptance requirements.
   No checkpoint replacement/download is selected by this review.
10. Preserve the target checkpoint and tokenizer. Alternative drafters must use
    the same target verifier; reduced draft vocabularies must not truncate p.
    Kernel-family and model-quality restrictions still apply to fallbacks.
11. DFlash packed weights must use the base `TextWeights` mmap/no-copy Metal
    path and its existing `mlock` policy. Keep the mapping owner alive for every
    Metal view. Do not add a second `File::read_exact` plus per-tensor packed
    copy loader for the drafter. Small scale and norm tensors may use the base
    loader's resident arena.

## Requirements

* **V0 — verifier correctness and ceiling first.** Compare identical
  continuations through sequential target forwards and M=2/4/8/16 verification
  over ≥3 prompts. Assert greedy equality and justified logit tolerances.
  Record dispatch, head/body/wall times, build, settings and context. Interleave
  ≥3 runs per shape with plain baselines; report medians and spread. Establish
  ceiling bounds only for the measured shapes and timing contract. Sabotage
  row selection and positions before claiming this path is covered.
* **D0 — trusted head diagnostics.** Index GPU output as `[rows,vocab]` before
  selecting columns; apply one documented normalization/sign/Hadamard/softcap
  convention. Compare identical nonzero inputs against the target decode head
  as well as the drafter buffer head. Assert numerical agreement using an
  explicitly justified tolerance; no universal “fp16 tolerance” is assumed.
  Exercise nonzero packed offsets and require invariance to irrelevant prefix
  bytes. The fp16 and fp32 buffer GEMM paths must respect offset/length together;
  allocating another 1 GiB head copy is a diagnostic control, not the product fix.
* **D1 — layout evidence.** Record existing permutation commits, audit the
  loader default, and verify decoded weights against an independent reference.
  Keep drafter layout separate from the target head's layout. Retire layout
  conclusions inferred from overlap through the incorrect head. Confirm original
  row-major and repacked colmajor byte layouts independently before dispatch.
* **D2 — aligned diagnosis.** Compare checkpoint semantics with z-lab: raw
  embeddings, tap layer indexing, feature concatenation, normalization, RoPE,
  context positions, masks, and draft row selection. HF `hidden_states[i+1]`
  is the post-layer-i output; its offset=1 does not imply incrementing the
  engine's zero-based post-layer tap IDs. Cross-check exllamav3's checkpoint
  tap_shift and export before Gemma layer_scalar against z-lab/HF post-layer
  semantics; record differences rather than adopting them blindly. Test combined
  layout/tap/embedding factors only with the correct head.
* **L1 — shared residency path.** Load the DFlash safetensors shard through
  `TextWeights`, so packed trellis ranges use the same mmap, no-copy Metal
  views, and `mlock` policy as target weights. Retain the `TextWeights` owner
  inside `DflashWeights`; keep scales and norms in its arena. Preserve the
  target model's base loader unchanged.
* **S1 — valid proposals.** For sampled mode draw every candidate from the exact
  q used by verification; greedy mode uses deterministic proposals. Match the
  supported temperature/top-k/top-p policy on p and q. The planned CLI path
  must support its default (1,64,.95) and greedy; validate other supported
  settings explicitly and reject unsupported settings. Invalid or
  zero-mass proposal distributions are errors, not unconditional acceptance.
* **C1 — cache transaction and unprocessed anchor.** At planned round entry
  anchor P is already emitted, but target and drafter context contain positions
  strictly before P. Verify `[anchor, n proposals]` at P..P+n; verify row k
  predicts proposal k at P+k+1. After K acceptances retain target rows through
  P+K, retract n−K rows, and emit the correction/bonus at P+K+1 as the next
  unprocessed anchor. Sliding caches must keep enough physical history for the
  active `prefill_chunk()` while its rows attend (default 1,024 rows); physical
  capacity is `sliding_window + prefill_chunk()`, while attention still applies
  the configured window. Later rows in the same chunk may not overwrite keys
  needed by the earliest query. Mask rejected future rows by
  absolute position and make cursor rollback preserve accepted state. Test
  zero/partial/full acceptance before, at, and after ring wrap, repeated
  rejection, error rollback, and retained K/V/positions against sequential
  decode. Restoring overwritten rows after verification cannot repair logits
  already computed from an incomplete window.
  Adapt the existing processed-anchor API explicitly;
  mixing the two contracts is forbidden.
* **C2 — tap ownership.** Copy accepted verify taps before a subsequent forward
  overwrites scratch. Append only verified target features, exactly once, with
  explicit absolute positions. Do not rewind an unappended drafter tail. Keep
  drafter context ending just before the next anchor, by appending verify rows
  0..K including the old anchor; test zero, partial, and full acceptance.
* **W1 — experimental loop.** After S1/C1/C2 pass, wire generation and CLI,
  metrics, EOS, length/context bounds, error handling, and chat behavior. Reject
  unsupported chat use clearly until its state contract is implemented.
* **R1 — candidate shapes.** Start with verifier M=2/4/8/16 and the checkpoint's
  trained B=16. Measure shorter proposal prefixes and their actual draft cost.
  B=32 is a later experiment requiring evidence that extending this B=16
  checkpoint is meaningful; it is not a default or assumed quality improvement.
  A two-block experiment needs its own exact count and conditioning contract;
  self-generated draft K/V must not silently replace target-feature context.
* **R2 — head and target-body gates.** `encode_gemm` is not a ready replacement.
  The reproduced 19.247559 M=2 discrepancy compares full target forwards; it
  does not isolate the head. Repair the layer-0 batch/decode discrepancy and
  prove sequential equivalence before target verification is used. Separately
  validate identical-input head paths and their offsets before batching the head.
* **M1 — end-to-end measurement.** Record build, checkpoints, prompt token counts,
  sampling policy, context, accepted-prefix histogram, and ≥200 rounds per mode
  across ≥3 prompts. Time complete draft, verify including head, correction
  forward, tap/cache update, sampling, and token emission with no double count.
* **M2 — regressions and memory.** Interleave plain/speculative runs; report
  median run throughput and spread, TTFT and pp/s, and `/usr/bin/time -l`
  footprint at 1k/2k/8k. The 14-token ≤350 ms TTFT target is a measured gate,
  not an established baseline. Preserve <20 GiB at 2k with the drafter resident.
* **F1 — conditional fallback evaluation.** Compare lookup proposals and, if
  selected later, the Gemma-specific EAGLE-3 drafter using the same correct target
  verifier. Lookup q is a deterministic point mass; sampled acceptance/residual
  must preserve p. EAGLE-3 needs explicit feature, vocabulary mapping, loader,
  cache and memory contracts before implementation. Record useful workload
  limits and reject candidates whose complete cost exceeds their measured gain.
* **P1 — GPU ownership after correctness.** Measure host staging, allocations,
  per-operation waits and draft wall time. Exllamav3 keeps features, projections
  and KV on the GPU; evaluate an equivalent bounded Rust/Metal flow with existing
  kernels only after D0/V0 pass. Preserve fp32 residuals, fp16 projection inputs,
  raw checkpoint semantics and exact target sampling; demonstrate an interleaved
  gain and memory compliance before adopting it.

## Acceptance criteria

* **AC1:** Experimental CLI runs ≥3 prompts × ≥200 tokens where EOS permits;
  actual emitted counts, rounds, acceptance, and complete decode tok/s reported.
* **AC2:** S1/C1/C2 pass zero/partial/full acceptance, EOS and limits, repeated
  rejection, sliding-ring wrap, and errors, through the actual round path.
* **AC3:** Greedy token equality with plain decode; sampled end-to-end tests
  demonstrate the proposal/acceptance/residual contract. Same-seed stream
  equality is not required when RNG consumption differs. Statistical agreement
  supplements, but cannot by itself prove, distribution exactness.
* **AC4:** Head oracle numerical assertions and independent layout/semantic
  evidence pass; deliberately broken proposal sampling, row indexing, scaling,
  positions, and rollback produce recorded, diagnosable failures.
* **AC5:** ≥30 tok/s median complete decode throughput, with interleaved
  baseline and spread. If unmet, mark performance incomplete and report measured
  bottlenecks; do not declare the checkpoint/hardware incapable from these probes.
* **AC6:** M=1 ≥15.0 tok/s and 14-token TTFT ≤350 ms, with baseline comparison;
  report an existing failure separately from a regression.
* **AC7:** Peak footprint <20 GiB at 2k, plus measured 1k/8k footprint and swap.
* **AC8:** V0 provides a reproducible correctness report and ceiling table before
  proposal-source selection. A passing diagnostic process is not sufficient.
* **AC9:** The proposed round processes the old anchor exactly once; correction
  work appears inside the next verify. Counts, positions, taps and retained
  caches agree after zero/partial/full acceptance, EOS, limits and errors.
* **AC10:** Any fallback has its own measured acceptance/cost/memory report and
  preserves exact target output semantics. Lookup gains on repetitive tasks do
  not establish 30 tok/s on general chat.
* **AC11:** DFlash packed weights are sourced from the base `TextWeights` mmap
  loader, pinned according to its `mlock` policy, and remain valid for the
  drafter's lifetime. Tests confirm aligned no-copy views use the tensor's
  actual packed offset and length, and misaligned views take the base loader's
  required alignment-copy path without changing bytes.

## Open questions / scheduled evidence

These are implementation gates, not assumptions resolved by this documentation
review: exact checkpoint layout/default; repaired host-head comparison; cause of
low aligned overlap; timing boundaries of 606/~135 ms; complete B16/B32 costs;
sliding-ring rollback correctness; viability and semantics of a two-block path;
correct existing batched 6-bpw head; attainable acceptance and throughput.

Unresolved decisions: which verifier shapes have adequate cost headroom;
whether DFlash's first divergence is implementation, conversion, target-feature
mismatch or checkpoint quality; whether lookup or EAGLE-3 merits implementation;
exact EAGLE-3 resident footprint and reduced-vocabulary mapping. No default
fallback, tolerance or checkpoint substitution is silently chosen by this draft.
