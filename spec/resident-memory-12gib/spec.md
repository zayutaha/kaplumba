# Resident memory: 15.1 GiB to 12 GiB

## Problem statement

The engine's true resident footprint is **15.12 GiB**, measured as the macOS
kernel's `phys_footprint`. The target is **<= 12 GiB**. Roughly 3.9 GiB of
read-mostly weight data is resident but does not need to be.

The number that matters is `phys_footprint`, not `ru_maxrss`. `ru_maxrss`
reported 3.3 GiB for this same build because it counts resident pages only and
misses ~7 GiB of Metal buffers; the kernel charges the full 15.1 GiB. Any
measurement of this work must use `phys_footprint`.

## Where the bytes are

Measured by enumerating the checkpoint (`model.safetensors.index.json`, 2422
tensors, 16.028 GB total) and cross-checked against the engine's own allocation
breakdown:

| Module | Bytes | Tensors | Currently resident | Offload candidate |
|---|---:|---:|---|---|
| `model.language_model.layers.*` | 11.000 GB | 2060 | yes | **no** (see R3) |
| `model.language_model.embed_tokens.weight` | 2.819 GB | 1 | yes | **yes** |
| `model.vision_tower.*` | 1.139 GB | 353 | **no** (already excluded) | n/a |
| `model.embed_vision.embedding_projection` | 0.012 GB | 1 | **no** (already excluded) | n/a |
| `lm_head.*` | 1.057 GB | 4 | yes | **yes** |
| `model.language_model.norm` | ~0 GB | 1 | yes | no |

Offloading `embed_tokens` + `lm_head` = **3.876 GB**, which lands the footprint
at approximately **11.2 GiB**, inside the 12 GiB target.

## Goal

Reduce `phys_footprint` from 15.12 GiB to <= 12 GiB by mapping
`embed_tokens` and `lm_head` from the checkpoint file instead of copying them
into the resident arena, with **no change to generated output**.

## Non-goals / restrictions

- **No quality loss.** Generated text must be byte-identical to the current
  build for the same prompt, seed, and sampling settings. This is the hard
  constraint; every other number is negotiable against it.
- **No swap thrashing.** The earlier all-file-backed build drove 3.9 GiB of swap
  and ran at ~1.4 tok/s. Offloading must not reintroduce that. This is why the
  11 GB of layer weights stay resident (R3).
- **No CPU inference fallback**, no Python, no requantization, no change to the
  model architecture or the EXL3 decode path.
- Vision weights are already excluded; there is nothing further to win there and
  no work is planned on them.
- Not attempting to shrink KV cache further; it is already on-demand (0.82 GiB).

## Why this was hard: the 4-byte alignment trap

Every previous attempt at mapping weights directly from the file produced
incoherent output, and the cause was not obvious:

- Trellis data is **not 4-byte aligned** in the checkpoint. `lm_head.trellis`
  starts at file offset `5321061770`, i.e. `offset % 4 == 2`. Layer trellis
  tensors are `% 4 == 3`.
- `exl3_code_at` in `crates/kernels/src/linear.metal` read the ring via
  `const device uint *tile = (const device uint *)(packed + ...)`, a 32-bit
  pointer cast that requires 4-byte alignment.
- The **arena copy silently repaired the alignment**: copying bytes into an
  offset-0, page-aligned buffer re-aligned the data for free. Mapping the file
  did not, so every 32-bit load read at the wrong address and decoded to noise.

This has been fixed: the kernel now combines two 16-bit loads
(`trellis_word`) instead of casting to `uint *`. That fix is a prerequisite for
this spec and is already committed.

## Requirements

- **R1** `embed_tokens` and `lm_head` are bound from the shard mapping and are
  not copied into the resident arena when offload is enabled.
- **R2** The embedding view is page-aligned, so its byte offset is non-zero.
  Both the GPU kernel (`embedding_scaled`, `embedding_scaled_buffer`) and the
  CPU path (`TextModel::embedding`) must apply that offset. The vocabulary size
  must be derived from the tensor's true byte length, never from the
  page-aligned window length.
- **R3** The 11.000 GB of layer trellis stays resident. Offloading it is
  measurably catastrophic (3.9 GiB swap, ~1.4 tok/s) because every byte is
  streamed on every token. This must be asserted by a test, not just a comment.
- **R4** Offload is opt-in and defaults to a known-good configuration. A bad
  configuration must not silently produce garbage; it must be detectable.
- **R5** The memory measurement is corrected engine-wide: no reported figure
  may be plain `ru_maxrss` presented as total memory.
- **R6** Generated output is unchanged. Verified by a before/after comparison on
  a fixed prompt set at temperature 0.

## Acceptance criteria

1. `phys_footprint` <= 12 GiB, measured with `/usr/bin/time -l`, on a short
   generation. Target expectation ~11.2 GiB.
2. `tests/memory_proof.rs` passes and reports the new footprint.
3. Output is byte-identical to commit `78b3198` for a fixed prompt set
   (France/counting/colors/ocean/Hi) at `--temperature 0`.
4. Swap usage does not grow during a sustained generation run.
5. Full workspace test suite green.
6. A regression test asserts layer trellis is **not** zero-copy (R3).

## Open questions

None. The alignment fix is done, the offload targets are measured, and the
arithmetic closes at ~11.2 GiB. One item to confirm during implementation:
whether `lm_head`, which is read in full on every token, keeps its pages hot
enough in the file cache to avoid the swap pathology that affects layers. It
should, at 1.06 GB against 11 GB of resident layers, but this is measured, not
assumed.
