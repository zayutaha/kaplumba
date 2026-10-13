# spec/dflash-round-30tps — plan

## Architecture and ordering

Implementation and testing are authorized by the user's 2026-10-13 instruction.
Preserve existing code and historical results; execute the tasks below in their
stated dependency order, one task and commit at a time.

First establish the target verifier's correctness and optimistic speed ceiling.
Only then decide whether to repair its existing path, repair DFlash predictions,
or evaluate an alternative proposal source. Independent diagnostics precede
claims about checkpoint quality. The target remains `Gxma-repacked`.

The reference review supplied T17, which fixes the drafter's ignored packed
weight offset. T18 found and repaired projection-sign reuse in decode fusions,
and disabled cross-layer fusion by default after the forced-load target showed
large layer-1 divergence. Greedy equivalence now passes for M=2/4/8/16 on three
prompts, while sampled-distribution equivalence and the verifier ceiling remain
open. V0b and all drafter-quality/layout comparisons must use corrected paths.
See `exllama_review.md` for the negative tests and isolated head control.

```
verifier equivalence → ideal-proposal ceiling for M=2/4/8/16
                         │                       │
                  ceilings miss 30        sufficient headroom
                         │                       │
              repair existing verify     independent DFlash diagnosis
                         │                       │
                    repeat gate         exact sampler + rollback + taps
                                                 │
                                    unprocessed-anchor round equivalence
                                                 │
                                  opt-in CLI + complete cost/acceptance
                                                 │
                              choose DFlash / conditional fallback evaluation
                                                 │
                                      30 tok/s + regression/memory gates
```

## Round state and data flow

The planned product loop uses the reference's unprocessed-anchor contract.
This is an explicit change from the existing `draft_and_verify` method, whose
anchor is already processed. Do not connect that existing method to the new loop
without adapting it and proving the state transitions.

Entry: anchor token P has already been emitted; target cache and drafter feature
context hold positions `<P`. Initialization prefills the prompt, snapshots and
appends prompt taps, samples/emits the first anchor, and leaves its forward to
the first verify. TTFT ends at first emission; all subsequent work is timed.

1. Propose n=B−1 masked-row tokens with anchor at P. Retain the exact q used to
   draw each sampled proposal, applying the declared temperature/top-k/top-p.
   Bound proposals before target verification; a B=1 plain fallback has none.
2. Begin a cache transaction and verify `[anchor, proposals]` at P..P+n.
   Target verify row k decides candidate k at P+k+1; full acceptance's bonus
   uses verify row n. The anchor is fed exactly once and is not reverified as
   a proposal. No caller-supplied previous-logit row is needed.
3. Sliding caches keep `sliding_window + PREFILL_CHUNK` physical slots. The
   attention mask remains the configured `sliding_window`; the extra chunk is
   history slack so a batched append cannot overwrite a key needed by its first
   query before attention reads it. Maximum append size is the bounded prefill
   chunk (1024 rows), larger than the verify cap. After K acceptances, retain
   target cache through P+K and lower the cursor by n−K; future rejected rows
   are excluded by their absolute position and overwritten only when their
   slots are reused. Snapshot accepted-prefix taps before any next forward.
4. Emit accepted prefix and correction/bonus in order, honoring accepted EOS
   and length/context limits. A rejected draft EOS never terminates generation.
   Truncate both committed state and output consistently at limits.
5. Append target feature for old anchor P and accepted tokens P+1..P+K to the
   drafter context, once, using `append_context(..., rows, exclusive_end, ...)`.
   The next anchor is P+K+1; its context must end at P+K. Emit the correction
   and leave it unprocessed until the next verify. No drafter rewind if rejected
   features were never appended. For K=0, append only the old anchor feature.
6. On error, restore a defined valid state or terminate the experimental session
   with an explicit error; no continuation with partially mutated caches.

At each continuing round boundary, both caches' exclusive end equals the next
anchor position. Accepted-prefix truncation at EOS or token/context limits must
truncate committed state to the same prefix. Match the plain generator's EOS
counting convention. A rejected EOS never stops output. Do not verify positions
beyond the remaining context or emit beyond the remaining token budget. If the
budget is exhausted, terminate without processing an unnecessary final anchor.

In steady state the round newly emits K accepted proposals plus one bonus token;
the old anchor was counted in the preceding emission. Throughput therefore uses
K+1, with actual EOS/limit truncation, and includes all round work. Correction
processing is inside the next verification; there is no separate target forward.

## Verifier gate and experiment protocol

V0a compares sequential target forwards and verification on identical token IDs,
positions and prompt prefixes, at M=2/4/8/16. Use isolated/reset state between
arms; forbid accidental cache reuse across A/B runs. Assert per-row greedy
identity, report logit errors against a predeclared justified tolerance, and
check position/cursor state. Include prompt positions before/at/after ring wrap.
T1b provides retained-cache equivalence under rejection/error before round use.

V0b uses saved target continuations as ideal proposals. Measure the complete
verification wall time rather than assuming a projection microbenchmark
represents a model pass. Attribute body and head with GPU profiling at a
representative short context; the profiler inserts a synchronization at each
layer boundary, so its total wall time is not a throughput measurement. Report
1k/2k/8k context behavior, ≥3 prompts and ≥3 interleaved runs per shape, actual
token counts, checkpoint, build and dispatch. All model runs use `gpuq status`
then `gpuq try`. The 2026-10-13 run is recorded in implementation notes.

Report synthetic bounds separately for both contracts: M/T_verify for an
unprocessed anchor, (M+1)/T_verify for a processed anchor omitting its additional
correction forward. Both omit real drafting, sampling and context-update cost.
If all bounds miss 30 tok/s, prioritize repairing a measured existing verifier
defect. No decision about a drafter can invalidate this gate for those measured
shapes. If a bound passes, subtract complete measured proposal/update overhead
before asserting practical headroom.

## Touchpoints and current behavior

| path | observed behavior | planned work |
|---|---|---|
| `crates/engine/tests/dflash_alpha_probe.rs` | D10 passes a prompt-length batch to the single-token `forward_with_hidden` API, then reads an unwritten last row; no numerical assertion | use sequential final-token decode rows, compare all vocabulary logits to target GEMV output, assert numeric/argmax bounds |
| D10 host control in `dflash_alpha_probe.rs` | target GEMV and DFlash sliced GEMM only cross-check two GPU paths | reconstruct bounded 128-column EXL3 head bands on CPU, apply documented Hadamard/sign transforms, and compare host/target/DFlash outputs on identical rows |
| `crates/engine/tests/dflash_alpha_probe.rs` | earlier probe compared only five of fifteen mask rows on one prompt and printed a stale speed decision | compare all 15 masked rows on three tokenized prompts; print per-depth overlap and target-token rank; label as diagnostic-only |
| T8 stage diagnostic | draft output row positions and later-depth signal were not checked across a full block | same-position anchor/mask mapping fixture, all-row counts, and 3-prompt depth report; no semantic patch without a reproduced mismatch |
| T10 proposal shape | fixed temperature and a single 15-row average hide whether proposals are over/under-confident at depth | measure q-temperature overlap against fixed target p for each depth; any product policy change first amends T1c and stores exact q |
| `crates/kernels/src/linear.rs` | `perm_mode` present; fp16/fp32 buffer GEMM validates offset but binds packed bytes at zero | honor packed offset/length via offset-aware binding; then independently validate layout |
| `crates/engine/src/dflash.rs` | T1a samples from retained q; temperature 1/top-k policy; existing processed-anchor method | complete sampler policy, unprocessed-anchor API, independent stage diagnosis and verified context ownership |
| DFlash `forward_block` / slice-based `Elements` and `LinearKernels` | individual operations materialize GPU buffers, dispatch, wait, and read results back into host `Vec`s | T19a measures whole round buckets; T19b separates drafter proposal, target verify, and accept/sample costs before designing GPU-resident scratch around existing encode paths |
| DFlash checkpoint trained with block size 16 | T8 shows overlap falls sharply after early rows, while every round computes/verifies all 15 proposals | T19c sweeps verified prefixes B=2/4/8/16 with exact output checks and records generation speed plus accepted-prefix counts; no runtime default changes without a repeatable gain |
| `crates/engine/src/gpu_cache.rs` | `retract` only lowers cursor; physical sliding ring must retain rows overwritten by a chunk | mask future rejected positions and validate cursor rollback; physical append headroom is handled at cache construction |
| `crates/engine/src/model.rs` | projection-sign and cross-layer faults are repaired; physical sliding capacity was only the window | keep attention window semantic unchanged, size physical sliding capacity to window plus maximum append chunk, prove boundary behavior |
| `crates/engine/src/dflash.rs` + `crates/engine/src/weights.rs` | DFlash loader uses `File::read_exact` into per-tensor vectors and copies packed bytes into new Metal buffers | build a single-shard selection and pass it through existing `TextWeights::load_dedicated`; use zero-copy trellis views and retain `TextWeights` so its mmap and `mlock` lifetime covers every drafter buffer |

### T20 follow-up — exact packed slices over shard-sized Metal buffers

The zero-copy packed view retains a Metal buffer whose length is the whole shard,
while each DFlash linear's `metal_offset` and `packed_len` identify one tensor.
Both fp16 and fp32 buffer GEMM calls must validate and bind that exact tensor
byte length (`packed_len * sizeof(u16)`) instead of inferring a slice from the
end of the shard. The target API and mmap ownership remain unchanged. Validate
all mapped DFlash linears against independent host decode and run a full-block
context comparison, which exercises both fp16 and fp32 drafter projections.
| `crates/engine/src/generate.rs` | plain generation only | experimental loop after round correctness |
| `crates/cli/src/{args,main,chat}.rs` | no drafter selection | explicit flags, loading, errors and metrics; supported chat policy |
| `crates/engine/tests/dflash_equivalence.rs` and a future verifier timing harness | existing evidence does not establish the full round | sequential-vs-batched control, ideal-proposal ceiling and round equivalence; final filenames chosen during implementation |

## Diagnostic controls

### L1 — drafter weight residency

Construct a `TextTensorSelection` for the drafter's single safetensors shard,
then call the base `TextWeights::load_dedicated` path. That loader maps the shard,
pins `.trellis` ranges with its existing `LlamaMlock` policy, and supplies
page-aligned Metal views. Build each drafter `QuantizedLinear` from the returned
view and tensor offset; keep `TextWeights` owned by `DflashWeights` for the
mapping lifetime. Read `.mcg`, scale vectors, and BF16 norm tensors from the
loader's arena. If `KAPLUMBA_RESIDENT_WEIGHTS` selects the arena path, bind the
resident tensor buffer instead. Do not change target weight loading.

Capture multiple nonzero target final-norm rows using the single-token
`forward_with_hidden` contract. Compare the complete vocabulary vector from
each target GEMV against DFlash's sliced-GEMM replay, with the same head,
permutation, input convention and softcap. Assert a declared max-logit error
bound and exact greedy-token equality. The 2026-10-13 harness run passed at
0.0078–0.0156 max error; zero-row and ×2-scale sabotage both failed the bound.
The earlier zero-row reading came from asking the one-token API to process a
prompt-length batch, not from a production forward bug.

The current “plain” control calls `gemm_chunk_buffer_sliced`, also called by
`draft_logits`; agreement there cannot independently certify `encode_gemv`.
A trusted comparison needs the target head path on identical inputs and an
independent reconstruction. Repair `forward_with_hidden`'s reported zero-row
observation if that interface is used; it is not yet a trustworthy input oracle.

z-lab reference exists at
`/private/var/folders/tp/wz3877_s1z756p1kcsz3_d4h0000gn/T/opencode/zlab/dflash/dflash/model.py`.
Read its `extract_context_feature` and sampling/verification loop first.
`hidden_states[layer_id+1]` refers to post-layer output; engine post-layer tap
IDs require comparison of semantics, not an automatic +1. Compare embedding,
fc, normalization, K/V, RoPE and attention stages before assigning low overlap
to head/ranking. Read reference code only; no Python project implementation.

## Shapes, formats and measurement boundaries

No persistent format changes. Logits are fp16 `[rows,vocab]`; bounded fp32 head
scratch for 32×262144 is 33554432 bytes = 32 MiB. This allocation estimate is
not a complete footprint measurement. KV is not transient scratch.

Measure M=2/4/8/16 verification first. DFlash uses its trained B=16 initially;
shorter prefix experiments must include any cost of computing the full block.
Do not assume the checkpoint was trained for B32. Two B=16 blocks
normally contain 30 proposals; adding a 31st requires a defined source. A second
block conditioned on generated block K/V differs from target-feature context;
prove its proposal and position contract instead of assuming checkpoint validity.
Compare variants on the same prompt/sampler workload; their realized acceptance
may differ, so do not require an artificial “same alpha”.

For the planned unprocessed-anchor loop:
`T_complete = draft_including_head + verify_including_head
+ feature/cache_update + sampling + emission + other_round_overhead`.
The historical processed-anchor path additionally includes correction_forward.
Choose disjoint instrumentation boundaries; also record wall time. The 156.71 ms
historical number excludes draft head/attention and is M=32. Historical verify
cost inclusion must be established before any head term is added/subtracted.

Report total emitted/total decode seconds per run, accepted-prefix survival by
depth, greedy and sampled workloads separately, ≥200 rounds per mode over ≥3
prompts, median and spread of repeated interleaved runs. A single prompt's
p/q overlap and off-diagonal maximum cannot prove attainable generation speed.

## Risks and fallbacks

After the shared verifier gate, alternatives are evaluation branches, not
automatically selected implementation:

- **Lookup:** search prompt/generated token history for matching suffixes and
  propose a bounded continuation. q is a point mass per conditional proposal.
  In sampled mode accept candidate x with probability p(x); on rejection sample
  p with x removed and renormalized. Greedy mode compares the target argmax.
  Test no match, ambiguous match, history boundaries and exact residual behavior.
  Measure repetition/copying workloads separately from general chat. Keep lookup
  storage bounded; future cache sizing/eviction is an unresolved design choice.
- **EAGLE-3:** evaluate `RedHatAI/gemma-4-31B-it-speculator.eagle3` only after
  a separate loader/state/vocabulary/memory design is reviewed. Its published
  config has one transformer layer, 32000 draft vocabulary entries, and taps
  [2,30,57]. Full target p still spans 262144 entries. Verify mapping, embeddings,
  trained target compatibility and sampled q; public acceptance is external
  evidence, not an M4 throughput prediction. Model weights stay outside the repo.

For either candidate, compare complete cost and accepted-prefix survival to the
same interleaved plain baseline. Existing kernel-family restrictions apply.

| risk | response |
|---|---|
| head control disagrees at a nonzero weight offset | repair the proved zero-offset binding; compare unchanged bytes/input again |
| target batch/decode discrepancy persists | inspect layer-0 projections, transforms, attention and residuals; no speculation until equivalence passes |
| every measured ideal ceiling misses 30 | repair an existing verifier bottleneck before changing proposal source |
| stage reference differs from quantized checkpoint | separate implementation/conversion errors from quantization and target-feature mismatch |
| ring wrap destroys retained K/V | bounded transaction storage or another proven restore strategy before shipping |
| B32 or two-block acceptance is lower | measure each variant independently; choose by complete throughput |
| existing head GEMM remains numerically wrong | keep sequential head; performance task remains open |
| speculative path is slower | retain opt-in diagnostics and plain default; record unmet performance target |
| 30 tok/s remains unmet | report cost and acceptance gap; do not infer hardware impossibility |

## Requirement → task mapping

| requirement | tasks |
|---|---|
| V0 | V0a, V0b |
| D0 | T0c, T7, T17 |
| D1 | T5, T6 |
| D2 | T8 |
| S1 | T1a, T1c |
| C1, C2 | T1b, T3, T16 |
| W1 | T1, T2, T4 |
| R1 | V0b, T9, T10, T15 |
| R2 | T11, T17, T18, V0a |
| M1 | T4, T10, T13 |
| M2 | T14 |
| F1 | F1a, F1b |
| P1 | T19, T14 |
| AC1 | T2, T4 |
| AC2, AC3 | T1a, T1c, T1b, T3, T12 |
| AC4 | T0c, T5, T6, T7, T8 and each task's sabotage |
| AC5 | T10, T11, T13, T14 |
| AC6, AC7 | T14 |
| AC8 | V0a, V0b |
| AC9 | T1b.1, T1b.2, T3, T16, T12 |
| AC10 | F1a, F1b, T13, T14 |
| AC11 | T20, T20.1 |
