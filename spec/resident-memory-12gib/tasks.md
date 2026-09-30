# Tasks: resident memory 15.1 GiB to 12 GiB

Status legend: `[ ]` pending, `[x]` done, `[~]` in progress.

- [x] **T1** Measure ground truth. Enumerate all 2422 tensors from
  `model.safetensors.index.json`; confirm vision filter catches 356 tensors /
  1.151 GB with 0 leaks; confirm offload targets are `embed_tokens` 2.819 GB +
  `lm_head` 1.057 GB = 3.876 GB. *(commit `78b3198`)*
- [x] **T2** Correct the memory metric. `phys_footprint` via `/usr/bin/time -l`
  instead of `ru_maxrss`; add `tests/memory_proof.rs` cross-checking the
  engine's own allocation breakdown. *(commit `78b3198`)*
- [x] **T3** KV cache on-demand growth. Full-attention caches start at 512 and
  double via `GpuKvCache::ensure_capacity`. 1.41 -> 0.82 GiB. *(commit `78b3198`)*
- [x] **T4** Fix the 4-byte alignment trap in `exl3_code_at`. Replace the
  `const device uint *` cast with a `trellis_word` helper combining two 16-bit
  loads. Code complete in working tree (`linear.metal`), **not yet committed**.
  Verified: `exl3_metal` suite green locally.
  - Files: `crates/kernels/src/linear.metal`
  - Test: `cargo test --test exl3_metal`
- [x] **T5** Plumb the embedding offset. `EmbeddingParams.offset` in
  `elements.metal` (both kernels), `encode_embedding_scaled(.., offset)`,
  `TextModel::embedding_offset`, and `from_metal_parts_with_len` so vocab size
  derives from the tensor length not the window length. **Partially done;
  critical bug remains:** `load_from_dir` discards the offset returned by
  `zero_copy_tensor_buffer` (`_offset` on model.rs:1467), and
  `finish_construction` hardcodes `embedding_offset: 0` (model.rs:1355).
  `from_metal_parts_with_len` must be extended to accept and thread the offset.
  - Files: `crates/kernels/src/elements.metal`, `crates/kernels/src/elements.rs`,
    `crates/engine/src/model.rs`
  - Remaining work: (a) add `embedding_offset: u64` param to
    `from_metal_parts_with_len`, (b) pass it through `finish_construction`,
    (c) stop discarding `_offset` in `load_from_dir`.
  - Test: `cargo test --test real_forward_buffer` (once offset is wired),
    verify output is identical to non-offload path.
- [x] **T6** Enable the offload set. `load_internal` skips the arena copy for
  `model.language_model.embed_tokens.weight` and `lm_head.*` when
  `KAPLUMBA_OFFLOAD_LARGE` is set. Verify layer trellis is *not* affected.
  Code exists in `weights.rs` (uncommitted). Not yet verified end-to-end
  because T5's offset bug blocks correct offloaded embedding output.
  - Files: `crates/engine/src/weights.rs`
  - Test: load model with `KAPLUMBA_OFFLOAD_LARGE=1`, generate, compare output.
- [x] **T7** Baseline capture. With offload off, record `phys_footprint` and the
  exact output of the 5-prompt set at `--temperature 0`.
  - Depends: T4, T5 committed (so baseline is clean HEAD).
- [x] **T8** Offload verification. With offload on: footprint <= 12 GiB and
  output byte-identical to T7. This is the go/no-go gate for the whole spec.
  - Depends: T5, T6, T7.
- [x] **T9** Swap watch. Sustained generation with offload on; confirm swap does
  not grow (resolves the open question about `lm_head` page residency).
  - **Done 2026-09-30.** 2500 generated tokens: `time -l` reports `swaps 0`,
    `vm.swapusage` 0 MiB at all 15 samples, 11.98 GiB peak footprint, no
    throughput cliff. The 0.984 GiB of `lm_head.trellis` was re-read 2500 times
    (~2.6 TB) through the mapping without a single swapped page. Full A/B in
    `implementation_notes.md`, session 2026-09-30.
  - Depends: T8.
- [x] **T10** Regression test for R3. Assert layer trellis holds arena spans and
  is therefore resident, so nobody re-enables zero-copy for the 11 GB. Test
  should check that `LoadedTensor.skip_copy == false` for layer tensors, or
  that `LoadedTensor.spans` is non-empty.
  - Files: `crates/engine/tests/memory_proof.rs` (natural home — it already
    tests weight residency) or a new test file.
  - Depends: T6.
- [x] **T11** Regression test for R6. Fixed prompt set, compare against recorded
  expected output so an offset regression fails loudly instead of degrading to
  noise.
  - Files: `crates/engine/tests/real_forward_buffer.rs` (add a new test
    function with hardcoded expected output).
  - Depends: T7 (baseline capture provides the expected output).
- [ ] **T12** Measure the alignment fix's cost. Compare decode tok/s before and
  after the two-16-bit-load change; if it regressed, add a `uint *` fast path
  used when the buffer is known 4-byte aligned.
  - Depends: T8 (need a working offload build to measure properly).
- [x] **T13** Flip the default. Make offload the default once T8-T9 pass; keep
  `KAPLUMBA_OFFLOAD_LARGE` unset-means-resident as the escape hatch.
  - Depends: T8, T9, T12.
- [ ] **T14** Documentation. Audit CLI output formatting in
  `crates/cli/src/report.rs` to ensure no path presents `ru_maxrss` as total
  memory; record the alignment trap in `implementation_notes.md` so it is not
  rediscovered.
  - Files: `crates/cli/src/report.rs`, `crates/engine/src/metrics.rs`,
    `spec/resident-memory-12gib/implementation_notes.md`

## Acceptance gate

T8 is the gate. If footprint does not reach <= 12 GiB **and** output stays
identical, this spec is not done regardless of the other tasks.

## Follow-up work (2026-09-27)

- **T10** added `crates/engine/tests/weight_residency.rs` (4 tests, all green):
  layer trellis must hold arena spans; `embed_tokens` and `lm_head.trellis` must
  be mapped; the `KAPLUMBA_RESIDENT_WEIGHTS=1` escape hatch must put everything
  back in the arena (verified by re-executing the test binary, since the flag is
  read at load time and cannot be toggled in-process); resident weights must stay
  under 11.5 GB.
- **T11** added `crates/engine/tests/output_identity.rs` (2 tests): five greedy
  prompts must produce byte-identical text with mapped weights, and the
  all-resident path must match the same recorded baseline. Expected strings were
  captured from the all-resident build at commit `78b3198`.
- **T9** swap watch: the first attempt was invalid because four model-loading
  residency tests ran concurrently, each pulling ~15 GB, so the measurement
  reflected memory contention rather than the offload. Restarted with nothing
  else running.
- **T12** still open: the cost of the two-16-bit-load change versus the original
  `uint *` cast has not been measured.

## Results (2026-09-27)

| Config | `phys_footprint` | Output |
|---|---:|---|
| all resident (baseline) | 15.12 GiB | reference |
| `lm_head.trellis` offloaded | 14.12 GiB | identical |
| `embed_tokens` offloaded (in-kernel bf16) | 12.45 GiB | identical |
| both offloaded (**default**) | **11.51 GiB** | identical |

Verified byte-identical output on the 5-prompt set at `--temperature 0`.

T9 (swap watch) and T10-T12 (regression tests, alignment-cost measurement) remain
open. T9 was started but the sustained 200-token run was too slow to complete
interactively; the swap question is therefore still unresolved, though the
11.51 GiB footprint is well clear of the swap pressure that broke the earlier
all-file-backed build.

## Results (2026-09-30) — T9 closed, and the `lm_head` comment corrected

Same prompt (62 tokens), same 2500 generated tokens, same binary, same session.
`phys_footprint` from `/usr/bin/time -l`; `ru_maxrss` is listed only to show how
far it is from the truth.

| Config | `phys_footprint` | arena weights | `ru_maxrss` | swaps | `decode_tok_s` | `load_ms` |
|---|---:|---:|---:|---:|---:|---:|
| both mapped (**default**) | **11.98 GiB** | 10.25 GiB | 7.05 GiB | 0 | 9.124 | 7 464 |
| `KAPLUMBA_RESIDENT_WEIGHTS=1` | 15.57 GiB | 13.85 GiB | 8.04 GiB | 0 | 11.992 | 19 005 |

- **T9 answered: swap does not grow.** `time -l` `swaps 0`, and `vm.swapusage`
  read 0 MiB at all 15 samples across the 288 s run, while ~2.6 TB was streamed
  through the `lm_head.trellis` mapping. No tok/s cliff either.
- The 3.59 GiB footprint saving equals the offload pair (3.609 GiB) to within
  0.02 GiB; nothing else in the footprint moved. `phys_footprint` reproduced to
  **196 488 bytes** across two identical runs, so this is ~18 000x the noise floor.
- **`decode_tok_s` is not a result and must not be quoted from this table.** The
  first pair looked like a 24 % cost for mapping (9.124 vs 11.992), but a repeat
  of the *same* default config returned 12.338 — a 35 % same-config swing, larger
  than the effect. Settling a throughput difference here needs interleaved
  A/B/A/B medians (AGENTS.md §10.5). `load_ms` (5 987–7 464 mapped vs 19 005
  resident) is the trustworthy throughput-adjacent number.
- **Long greedy generations are not bit-reproducible.** Two runs of the same
  binary, config, prompt and seed at `--temperature 0` produced different text
  ("prevented the metal from failing" vs "…cracking"). This is *not* the
  offload — the control proves it. It does mean the "byte-identical" claim
  above holds for the short 5-prompt set only, and that any long-generation A/B
  needs a same-config control before attributing an output difference to
  anything.
- **A wrong comment was corrected.** `weights.rs` claimed mapping
  `lm_head.trellis` was "actively harmful. Left resident", citing
  12.99 → 14.14 GiB. The 12.99 GiB was the failed run this file already flags
  as an artifact, so the comparison was made against a run that never allocated
  KV; the sign of the result was inverted. The comment has been rewritten; the
  code was always right. This closes `spec/engine-performance` Q12 / T9.

T12 (alignment-cost measurement) is still open.
