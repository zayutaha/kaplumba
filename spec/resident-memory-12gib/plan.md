# Plan: resident memory 15.1 GiB to 12 GiB

## Current state (as of commit `78b3198` + uncommitted working-tree changes)

Already done and committed (all in `78b3198`):

- Model is coherent; `phys_footprint` measured at 15.12 GiB.
- Vision excluded: `is_vision_tensor_name` in `crates/engine/src/safetensors.rs`
  filters `model.vision_tower.*` and `model.embed_vision.*`. Verified against
  the full 2422-tensor index: 356 tensors / 1.151 GB filtered, **0
  vision-named tensors leak into the text set**.
- Layer trellis bound from the arena (`load_internal` `skip_copy` logic), which
  fixed both coherence and the 3.9 GiB swap thrash.
- `GpuKvCache::ensure_capacity`: full-attention caches start at 512 and double on
  demand. KV 1.41 -> 0.82 GiB.
- `tests/memory_proof.rs` measures `phys_footprint` via `/usr/bin/time -l` and
  cross-checks the engine's allocation breakdown.
- `metrics.rs`: `measure_rss()` exists and is documented as `ru_maxrss`-only;
  `measure_footprint_note()` directs callers to `/usr/bin/time -l`. The
  `RunMetrics.peak_rss_bytes` field is correctly labeled `ru_maxrss`, not
  presented as total memory.

Uncommitted, in the working tree, building cleanly (5 files modified):

- **`crates/kernels/src/linear.metal`**: `trellis_word` helper; `exl3_code_at`
  uses two 16-bit loads instead of `uint *` cast. Alignment fix is complete.
- **`crates/kernels/src/elements.metal`**: `EmbeddingParams.offset` field added;
  both `embedding_scaled` and `embedding_scaled_buffer` apply it. Complete.
- **`crates/kernels/src/elements.rs`**: `encode_embedding_scaled(.., embedding_offset: u64)` added. Complete.
- **`crates/engine/src/model.rs`**:
  - `TextModel.embedding_offset: u64` field exists.
  - `from_metal_parts_with_len` exists and derives vocab size from the
    caller-provided `embedding_bytes` (not the buffer length). Complete.
  - **INCOMPLETE**: `finish_construction` hardcodes `embedding_offset: 0`
    (line 1355). `from_metal_parts_with_len` does not accept an offset
    parameter, so even though `load_from_dir` calls
    `zero_copy_tensor_buffer(EMBED_NAME)` which returns `(buffer, offset)`,
    the offset is **discarded** (`_offset` on line 1467). The offset is never
    threaded into the `TextModel` struct. This must be fixed before T5 is done.
  - Both the GPU path (`forward_buffer_inner` line 1720) and the CPU path
    (`embedding()` line 1884) read `self.embedding_offset`, so once the
    constructor sets it correctly, both paths will apply it.
- **`crates/engine/src/weights.rs`**: `KAPLUMBA_OFFLOAD_LARGE` env var gates
  the offload set (`embed_tokens.weight` + `lm_head.*`). `skip_copy` logic
  skips the arena allocation for offloaded tensors. `no_copy_view` returns
  a page-aligned buffer + byte offset. `zero_copy_tensor_buffer` is the
  public API. Logic appears complete.

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
| `crates/kernels/src/linear.metal` | `trellis_word` helper; `exl3_code_at` uses it | done (uncommitted) |
| `crates/kernels/src/elements.metal` | `EmbeddingParams.offset`; both embedding kernels apply it | done (uncommitted) |
| `crates/kernels/src/elements.rs` | `encode_embedding_scaled(.., embedding_offset)` | done (uncommitted) |
| `crates/engine/src/model.rs` | `embedding_offset` field; `from_metal_parts_with_len`; offload-aware binding | **partial** — offset not threaded from `load_from_dir` into `finish_construction` |
| `crates/engine/src/weights.rs` | offload set + `skip_copy` + `zero_copy_tensor_buffer` | done (uncommitted) |
| `crates/engine/src/gpu_cache.rs` | `ensure_capacity` (KV) | done, committed |
| `crates/engine/tests/memory_proof.rs` | footprint assertion | done, committed |
| `crates/engine/tests/memory_proof.rs` or new test | R3: layer trellis is not zero-copy | **todo** |
| `crates/engine/tests/real_forward_buffer.rs` | R6: before/after output comparison | **todo** — file exists with correctness probes but no fixed-output comparison |

## Sequencing

1. **Fix T5**: Thread the embedding offset from `zero_copy_tensor_buffer`
   through `from_metal_parts_with_len` → `finish_construction` → set
   `self.embedding_offset`. Stop discarding `_offset` in `load_from_dir`.
2. Verify the full uncommitted work builds and the suite is green.
3. Run with offload **off**; record footprint + output as the baseline.
4. Run with `KAPLUMBA_OFFLOAD_LARGE=1`; confirm output is byte-identical and
   footprint drops to ~11.2 GiB.
5. Watch swap over a sustained run (R4 / open question).
6. Add the R3 regression test and the R6 output-comparison test.
7. Flip the default to offload-on once step 4-5 pass; keep the env var to force
   resident.
8. Update `metrics.rs` doc so no code path reports `ru_maxrss` as total memory
   (R5). **Status**: already mostly done — `RunMetrics.peak_rss_bytes` is
   documented as `ru_maxrss` only, `measure_footprint_note()` points to
   `/usr/bin/time -l`. Just need to audit CLI output formatting.
9. Update `tasks.md` + `implementation_notes.md`, commit.

## Risks and fallbacks

- **lm_head pages not staying hot.** It is read in full every token. If swap
  grows, fall back to offloading `embed_tokens` only (2.819 GB), which lands at
  ~12.3 GiB — still short of 12, so the next lever would be trimming
  `DEFAULT_RESIDENT_LIMIT_BYTES` slack or the final-norm/scratch buffers.
- **Kernel slowdown from two 16-bit loads.** The alignment fix replaces one
  32-bit load with two 16-bit loads on the hot path. Adjacent 16-bit loads
  normally coalesce, but this must be measured against the pre-fix decode rate
  and reverted to a `uint *` fast path if the arena is detected to be 4-byte
  aligned.
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
