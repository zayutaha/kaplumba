# Plan: resident memory 15.1 GiB to 12 GiB

## Current state (as of commit `78b3198`)

Already done and committed:

- Model is coherent; `phys_footprint` measured at 15.12 GiB.
- Vision excluded: `is_vision_tensor_name` filters
  `model.vision_tower.*` and `model.embed_vision.*`. Verified against the full
  2422-tensor index: 356 tensors / 1.151 GB filtered, **0 vision-named tensors
  leak into the text set**.
- Layer trellis bound from the arena (`load_internal` `skip_copy` logic), which
  fixed both coherence and the 3.9 GiB swap thrash.
- `GpuKvCache::ensure_capacity`: full-attention caches start at 512 and double on
  demand. KV 1.41 -> 0.82 GiB.
- `tests/memory_proof.rs` measures `phys_footprint` via `/usr/bin/time -l` and
  cross-checks the engine's allocation breakdown.
- `exl3_code_at` alignment fix in `linear.metal` (two 16-bit loads instead of a
  `uint *` cast). Committed in the working tree, suite green.

In progress, uncommitted, building cleanly:

- `EmbeddingParams.offset` plumbed through `elements.metal`,
  `encode_embedding_scaled`, and `TextModel::embedding_offset`.
- `TextModel::from_metal_parts_with_len` so vocabulary size derives from the
  tensor length rather than the page-aligned window length.
- `load_internal` offload set = `embed_tokens.weight` + `lm_head.*`, gated on
  `KAPLUMBA_OFFLOAD_LARGE`.

## Data flow

### Embedding (2.819 GB, one row per token)

```
safetensors shard --mmap--> ShardReader.pointer
                              |  no_copy_view(start, len)
                              v
                     page-aligned Buffer + byte offset
                              |
        +---------------------+---------------------+
        |                                           |
  GPU: elements.metal                          CPU: TextModel::embedding()
  embedding_scaled_buffer                       *(u16*)contents + offset/2
  embedding[offset + token*hidden + col]
```

`offset` is `params.offset` in the kernel. Both call sites must agree.

### lm_head (1.057 GB, full vocab GEMV per token)

```
load_quantized_linear(name="lm_head")
  copied = !tensor.spans.is_empty()
  copied == false  ->  zero_copy_tensor_buffer("lm_head.trellis")
  -> (page-aligned Buffer, packed_offset)
  -> linear kernels: trellis_word(packed, base + j)   // 2 x u16, alignment-safe
```

This path already existed and already threads `packed_offset`; the alignment fix
is what makes it correct.

## Touchpoints

| File | Change | Status |
|---|---|---|
| `crates/kernels/src/linear.metal` | `trellis_word` helper; `exl3_code_at` uses it | done |
| `crates/kernels/src/elements.metal` | `EmbeddingParams.offset`; both embedding kernels apply it | done |
| `crates/kernels/src/elements.rs` | `encode_embedding_scaled(.., embedding_offset)` | done |
| `crates/engine/src/model.rs` | `embedding_offset` field; `from_metal_parts_with_len`; offload-aware binding | done |
| `crates/engine/src/weights.rs` | offload set + `skip_copy` | done |
| `crates/engine/src/gpu_cache.rs` | `ensure_capacity` (KV) | done, committed |
| `crates/engine/tests/memory_proof.rs` | footprint assertion | done, committed |
| `crates/engine/tests/exl3_format.rs` | new test: layer trellis is not zero-copy (R3) | **todo** |
| `crates/engine/tests/real_forward_buffer.rs` | before/after output comparison (R6) | **todo** |

## Sequencing

1. Verify the current uncommitted work builds and the suite is green. (in progress)
2. Run with offload **off**; record footprint + output as the baseline.
3. Run with `KAPLUMBA_OFFLOAD_LARGE=1`; confirm output is byte-identical and
   footprint drops to ~11.2 GiB.
4. Watch swap over a sustained run (R4 / open question).
5. Add the R3 regression test and the R6 output-comparison test.
6. Flip the default to offload-on once step 3-4 pass; keep the env var to force
   resident.
7. Update `metrics.rs` doc so no code path reports `ru_maxrss` as total memory (R5).
8. Update `tasks.md` + `implementation_notes.md`, commit.

## Risks and fallbacks

- **lm_head pages not staying hot.** It is read in full every token. If swap
  grows, fall back to offloading `embed_tokens` only (2.819 GB), which lands at
  ~12.3 GiB — still short of 12, so the next lever would be trimming
  `DEFAULT_RESIDENT_LIMIT_BYTES` slack or the final-norm/scratch buffers.
- **Kernel slowdown from two 16-bit loads.** The alignment fix replaces one
  32-bit load with two 16-bit loads on the hot path. Adjacent 16-bit loads
  normally coalesce, but this must be measured against the pre-fix decode rate
  (currently ~1.5 tok/s) and reverted to a `uint *` fast path if the arena is
  detected to be 4-byte aligned.
- **Silent wrongness.** If an offset is dropped, output degrades to noise rather
  than erroring. R6's byte-identical comparison is the guard; it must run in CI
  or at least be run before flipping any default.

## Requirement mapping

| Requirement | Satisfied by |
|---|---|
| R1 offload both tensors | `weights.rs` offload set, `model.rs` binding |
| R2 offset applied in GPU and CPU paths | `EmbeddingParams.offset`, `TextModel::embedding_offset`, `from_metal_parts_with_len` |
| R3 layers stay resident | `skip_copy` logic + new regression test |
| R4 opt-in, detectable | `KAPLUMBA_OFFLOAD_LARGE`, R6 comparison |
| R5 correct measurement | `measure_rss` doc + `memory_proof.rs` |
| R6 output unchanged | before/after comparison test |
