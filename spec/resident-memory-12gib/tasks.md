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
  loads. Verified: `exl3_metal` suite green.
- [~] **T5** Plumb the embedding offset. `EmbeddingParams.offset` in
  `elements.metal` (both kernels), `encode_embedding_scaled(.., offset)`,
  `TextModel::embedding_offset`, and `from_metal_parts_with_len` so vocab size
  derives from the tensor length not the window length. **Building; not yet
  verified against a real generation.**
- [ ] **T6** Enable the offload set. `load_internal` skips the arena copy for
  `model.language_model.embed_tokens.weight` and `lm_head.*` when
  `KAPLUMBA_OFFLOAD_LARGE` is set. Verify layer trellis is *not* affected.
- [ ] **T7** Baseline capture. With offload off, record `phys_footprint` and the
  exact output of the 5-prompt set at `--temperature 0`.
- [ ] **T8** Offload verification. With offload on: footprint <= 12 GiB and
  output byte-identical to T7. This is the go/no-go gate for the whole spec.
- [ ] **T9** Swap watch. Sustained generation with offload on; confirm swap does
  not grow (resolves the open question about `lm_head` page residency).
- [ ] **T10** Regression test for R3. Assert layer trellis holds arena spans and
  is therefore resident, so nobody re-enables zero-copy for the 11 GB.
- [ ] **T11** Regression test for R6. Fixed prompt set, compare against recorded
  expected output so an offset regression fails loudly instead of degrading to
  noise.
- [ ] **T12** Measure the alignment fix's cost. Compare decode tok/s before and
  after the two-16-bit-load change; if it regressed, add a `uint *` fast path
  used when the buffer is known 4-byte aligned.
- [ ] **T13** Flip the default. Make offload the default once T8-T9 pass; keep
  `KAPLUMBA_OFFLOAD_LARGE` unset-means-resident as the escape hatch.
- [ ] **T14** Documentation. Update `metrics.rs` docs so no path reports
  `ru_maxrss` as total memory; record the alignment trap in
  `implementation_notes.md` so it is not rediscovered.

## Acceptance gate

T8 is the gate. If footprint does not reach <= 12 GiB **and** output stays
identical, this spec is not done regardless of the other tasks.
