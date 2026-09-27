# Implementation notes: resident memory 15.1 GiB to 12 GiB

## Spec review (2026-09-27)

Full verification of spec claims against the codebase at commit `a6b415d`
(HEAD, with 5 uncommitted modified files).

### Verified correct

- `is_vision_tensor_name` in `safetensors.rs:516-517` matches
  `model.vision_tower.*` and `model.embed_vision.*`. ✓
- `EmbeddingParams.offset` exists in `elements.metal:21`. Both
  `embedding_scaled` (line 60) and `embedding_scaled_buffer` (line 245)
  apply `params.offset`. ✓
- `encode_embedding_scaled` in `elements.rs:726` accepts
  `embedding_offset: u64`. ✓
- `from_metal_parts_with_len` in `model.rs:1394` derives vocab size from the
  caller's `embedding_bytes` parameter, not the buffer length. ✓
- `trellis_word` helper in `linear.metal:146` combines two `ushort` loads.
  `exl3_code_at` (line 156) uses it. ✓
- `weights.rs:524-528`: offload set correctly gated on `KAPLUMBA_OFFLOAD_LARGE`
  env var, with `offloadable` matching `embed_tokens.weight` and `lm_head.*`. ✓
- `no_copy_view` in `weights.rs:961` page-aligns downward and returns the
  byte delta as the offset. ✓
- `GpuKvCache::ensure_capacity` in `gpu_cache.rs:149`. ✓
- `metrics.rs:20-25`: `peak_rss_bytes` documented as `ru_maxrss`, with
  explicit note that `phys_footprint` is the real figure. ✓
- All test files mentioned in the plan exist:
  `memory_proof.rs`, `exl3_format.rs`, `real_forward_buffer.rs`. ✓
- Build compiles clean (2 warnings: unused `packed_offset` in linear.rs,
  unused `integer_end` in exl3.rs). ✓

### Errors found and corrected in spec

1. **Vision tensor count discrepancy.** spec.md table listed
   `model.vision_tower.*` as 353 tensors + `model.embed_vision.*` as 1 tensor
   = 354, but plan.md said 356 / 1.151 GB. The filter is
   `starts_with("model.embed_vision.")` which may match multiple tensors
   (e.g., weight + bias). Merged into a single row with the verified total
   of 356 tensors / 1.151 GB.

2. **T4 marked done but never committed.** The `trellis_word` change is in the
   working tree (`git status` shows `M crates/kernels/src/linear.metal`) with
   no commit hash. Changed from `[x]` to `[~]`.

3. **Embedding offset not threaded end-to-end (critical T5 bug).**
   `load_from_dir` calls `zero_copy_tensor_buffer(EMBED_NAME)` which returns
   `(buffer, offset)`, but the offset is assigned to `_offset` (model.rs:1467)
   and discarded. `finish_construction` hardcodes `embedding_offset: 0`
   (model.rs:1355). `from_metal_parts_with_len` has no offset parameter.
   Plan.md and touchpoints table previously marked model.rs as "done";
   corrected to "partial". Tasks.md T5 expanded with fix description.

4. **Plan.md marked model.rs touchpoint as "done".** Corrected to "partial".

5. **T6 marked pending but code exists.** `weights.rs` offload logic is
   already in the working tree. Changed from `[ ]` to `[~]`.

6. **R3 test location.** Plan.md pointed to `exl3_format.rs` which tests EXL3
   decoding, not weight residency. Corrected to `memory_proof.rs` as the
   natural home (it already tests allocation breakdown).

7. **spec.md R2 wording.** "The embedding view is page-aligned, so its byte
   offset is non-zero" was ambiguous. Clarified: the buffer is page-aligned
   *downward* so the tensor's byte offset *within the buffer* is non-zero.

8. **spec.md alignment fix status.** Said "already committed" but it's only
   in the working tree. Corrected.

### Key invariants discovered

- `finish_construction` is the single point where `embedding_offset` is set.
  Both the GPU path (line 1720) and CPU path (line 1884) read it, so fixing
  the constructor is sufficient.
- `from_metal_parts` (the convenience wrapper) calls `from_metal_parts_with_len`
  with `bytes = buffer.length()` and no offset — this is the non-offload path
  and must continue to work with offset=0.
- The `KAPLUMBA_ZERO_COPY_WEIGHTS` env var (weights.rs:523) is a separate
  debug flag that maps *all* weights from the file. It existed before this
  spec and triggered the 3.9 GiB swap. It is independent of
  `KAPLUMBA_OFFLOAD_LARGE`.

## Implementation (2026-09-27)

Result: **15.12 -> 11.51 GiB**, output byte-identical on the 5-prompt set.

### What actually moved the number

- `lm_head.trellis` mapped instead of copied: **-1.00 GiB**. Despite being read
  in full every token, mapping it still helps; my first reading of this was wrong
  because the comparison run had failed early (see below) and never allocated KV.
- `embed_tokens` mapped instead of copied: **-2.67 GiB**. This needed the
  bf16 -> fp16 conversion to move into the kernel. The table is 2.82 GB of BF16
  and only one row (~10 KiB) is read per token, so converting on read leaves a
  negligible working set.

### Two traps hit, both worth remembering

1. **The offload never engaged for the embedding.** `classify_tensor` marks
   `embed_tokens` as `converted` (BF16 -> FP16), and `skip_copy` was guarded on
   `!converted`, so it was copied exactly as before. Diagnosed by printing
   `spans.len()` per tensor with the flag on and off: `lm_head.trellis` went
   1 -> 0, the embedding stayed at 1.

2. **The embedding's window offset is 2055, which is odd.** The tensor begins at
   shard offset 133127, and `no_copy_view` page-aligns *downward*, so the offset
   within the window is 133127 mod 4096 = 2055. The kernel originally took the
   offset in elements (`offset / 2`), which truncated 2055 to 1027 and produced
   "logits must be non-empty and finite". Fixed by passing the offset in **bytes**
   and reading the offloaded BF16 table byte-wise:
   `ushort(raw[at]) | (ushort(raw[at+1]) << 8)`, then `half(as_type<float>(uint(bits) << 16))`.
   safetensors makes no alignment promise beyond the header, so any mapped tensor
   must be treated as possibly odd-aligned.

### Misleading measurement to avoid

An early "12.99 GiB" reading was **an artifact**: that run had failed with
`lm_head.suh has 0 values`, so it exited before allocating KV or scratch. Only
compare configurations that produced correct output. Always check the text, not
just the number.

### Offload set is deliberately narrow

Only `embed_tokens.weight` and `lm_head.trellis`. An earlier
`name.starts_with("lm_head.")` also caught `suh` / `svh` / `mcg`, which are sign
side-tensors read on the CPU via `read_f16` and need their arena copy; offloading
them fails with "lm_head.suh has 0 values, expected 5376".

### Escape hatch

`KAPLUMBA_RESIDENT_WEIGHTS=1` puts everything back in the arena (15.12 GiB).
`KAPLUMBA_ZERO_COPY_WEIGHTS` is the older debug flag that maps *all* weights from
the file; it is the configuration that drove 3.9 GiB of swap and must not be used
for benchmarking.
