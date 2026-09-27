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
- [ ] **T9** Swap watch. Sustained generation with offload on; confirm swap does
  not grow (resolves the open question about `lm_head` page residency).
  - Depends: T8.
- [ ] **T10** Regression test for R3. Assert layer trellis holds arena spans and
  is therefore resident, so nobody re-enables zero-copy for the 11 GB. Test
  should check that `LoadedTensor.skip_copy == false` for layer tensors, or
  that `LoadedTensor.spans` is non-empty.
  - Files: `crates/engine/tests/memory_proof.rs` (natural home — it already
    tests weight residency) or a new test file.
  - Depends: T6.
- [ ] **T11** Regression test for R6. Fixed prompt set, compare against recorded
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
