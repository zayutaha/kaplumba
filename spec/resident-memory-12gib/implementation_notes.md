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

---

## Session 2026-09-30: T9 swap watch, and a wrong code comment corrected

Closes **T9** (the swap watch that had been started twice and abandoned) and
corrects a fabricated measurement that had been promoted into
`crates/engine/src/weights.rs`.

Method: `/usr/bin/time -l` around the real binary, plus a 5 s sampler reading
`vm.swapusage` and `top`'s `PhysMem` for the whole run. The engine's `flock` is
held for the process lifetime (`gpu_lock.rs:52`), so every run below had the
box to itself — no other engine process could be resident during a measurement.

### Result: swap does not grow, and throughput does not decay

Offload ON (default), 62-token prompt, **2500 generated tokens**, no EOS:

| | |
|---|---|
| `/usr/bin/time -l` `swaps` | **0** |
| `vm.swapusage` at 15 samples over 288 s | **0 MiB at every sample** |
| peak memory footprint | 12 867 037 000 B = **11.98 GiB** |
| `ru_maxrss` | 7 565 672 448 B = 7.05 GiB (see below: *not* a footprint) |
| `decode_tok_s` | 9.124 (no cliff; 2500/2500 tokens) |
| wall | 288.5 s, `load_ms` 7464 |

`lm_head.trellis` is 1 056 964 608 B and is read in full on every token, so the
run re-read it 2500 times — about **2.6 TB** of reads through a file-backed
mapping — with **zero** pages swapped. The paging failure the old comment
predicted does not occur at this size.

**So: no serious finding. Swap does not grow, PhysMem does not grow, tok/s does
not decay off a cliff.** T9's open question ("is `lm_head.trellis` page
residency a problem?") is answered: it is not, at 1.06 GB against an ~11.3 GiB
text arena, because those pages stay resident in the page cache and are
reclaimable rather than anonymous.

### The A/B, same prompt / same token count / same binary / same session

| Metric | default (both mapped) | `KAPLUMBA_RESIDENT_WEIGHTS=1` | delta |
|---|---:|---:|---:|
| peak memory footprint | **11.98 GiB** | **15.57 GiB** | **-3.59 GiB** |
| arena weights (`allocation_breakdown`) | 10.25 GiB | 13.85 GiB | -3.60 GiB |
| KV / scratch | 0.82 / 0.003 GiB | 0.82 / 0.003 GiB | 0 |
| `ru_maxrss` | 7.05 GiB | 8.04 GiB | -0.99 GiB |
| `load_ms` | 7 464 | 19 005 | **-61 %** |
| `decode_tok_s` | 9.124 | 11.992 | **not a result** — see below |
| `swaps` | 0 | 0 | — |

The footprint saving, 3.59 GiB, is the offload pair to within 0.02 GiB:
`embed_tokens` 2 818 572 288 B (2.625 GiB) + `lm_head.trellis`
1 056 964 608 B (0.984 GiB) = 3 875 536 896 B = **3.609 GiB**. The arena
delta is 3.60 GiB. Nothing else moved.

Note `ru_maxrss` moved by only 0.99 GiB while the real footprint moved 3.59 GiB.
Read as a footprint it would understate the win by 3.6x — the exact failure mode
AGENTS.md §10.3 warns about.

### Throughput: the first A/B suggested a 24 % cost, and a control run killed it

The first pair of runs gave `decode_tok_s` 9.124 (mapped) vs 11.992 (all
resident) — apparently a 24 % cost for mapping, and a contradiction of Q13's
"13 % slower" in the opposite direction. **A third run of the *same* default
config then returned 12.338 tok/s.** Same binary, same prompt, same flags:

| run | config | `decode_tok_s` | peak footprint |
|---|---|---:|---:|
| A | default (mapped) | 9.124 | 12 867 037 000 |
| A2 | default (mapped), repeat | **12.338** | 12 867 233 488 |
| B | `KAPLUMBA_RESIDENT_WEIGHTS=1` | 11.992 | 16 719 311 256 |

The same-config spread (9.124 -> 12.338, **35 %**) is *larger* than the
mapped-vs-resident difference being claimed. **There is no measurable throughput
cost to the offload on this evidence, and the 24 % figure should be
disregarded.** This is AGENTS.md §10.5 exactly: decode on this box has been
measured at 3.7, 3.9, 4.3, 4.8, 6.0 and 6.4 tok/s from the same binary minutes
apart. Settling even a 10 % effect needs interleaved A/B/A/B medians over
several repeats, not one sample per arm.

The contrast is the useful part: **`phys_footprint` is reproducible to 196 488
bytes (0.0002 GiB) across two identical runs, while `decode_tok_s` varies by
35 %.** The 3.59 GiB offload saving is ~18 000x the footprint noise floor, so
the memory result is solid. Any future throughput claim on this box needs a
methodology built for a metric that noisy.

`load_ms` is more trustworthy than tok/s and also favours the offload
(5 987–7 464 ms mapped vs 19 005 ms resident): not copying 3.6 GiB is
unambiguously faster to start.

Q13's *conclusion* — that making weights resident does not slow decode, so the
old 58 s prefill was not host paging — is unaffected and is if anything
strengthened.

### Greedy decoding is NOT bit-reproducible over long generations

Runs A and A2 are the **same binary, same config, same prompt, same seed**,
`--temperature 0`, and they produce **different text**:

```
184c184
<  prevented the metal from failing.
---
>  prevented the metal from cracking.
```

This is the same class of drift seen between the mapped and all-resident runs
(where the model's own list numbering diverged, 77 vs 71). It is **not** caused
by the offload — the control run proves that, because two *identical*
configurations diverge from each other. It is inherent run-to-run
nondeterminism (Metal reduction order / tile scheduling), amplified over 2500
tokens in a model that is counting its own list items.

**This qualifies the "output byte-identical" claim in the Results table
above.** It is true, and tested, for the 5-prompt set in
`crates/engine/tests/output_identity.rs` — short generations never reach the
divergence. It is **not** true for long generations, in *any* config. Do not
use a 2500-token greedy generation as a bit-exactness gate. A control run of
the *same* config is required before attributing any output difference to a
change under test; without one, an A/B on long generations proves nothing about
numerics. `output_identity.rs` should grow a logits comparison rather than
relying on text.

### Exact sizes, read from the safetensors headers rather than from prose

- `lm_head.trellis` = 1 056 964 608 B (0.984 GiB), I16, shard 2
- `model.language_model.embed_tokens.weight` = 2 818 572 288 B (2.625 GiB), BF16
- `lm_head.suh` 10 752 B, `lm_head.svh` 524 288 B, `lm_head.mcg` 4 B (arena)
- offload pair = 3 875 536 896 B = 3.609 GiB
- 411 `.trellis` tensors = 12 039 487 488 B = 11.213 GiB
- all 2422 tensors in both shards = 16 028 050 244 B = 14.927 GiB, but that
  **includes ~1.07 GiB of vision tower** (356 `model.vision_tower.*` tensors),
  which the engine never loads. Text-only weights = **13.85 GiB**, which is
  exactly what the all-resident arena reports. The "1.06 GB" / "2.82 GB" in the
  code comment are decimal GB; in GiB they are 0.98 and 2.63. Comparing those
  against a `phys_footprint` figure, which is binary, is how a 1 GiB
  discrepancy gets invented.

### The 11.5 GiB figure, itemised

`allocation_breakdown()` (engine's own accounting) vs the OS:

| Component | Bytes | Note |
|---|---:|---|
| arena-resident weights | 10.25 GiB | anonymous, no arena span for the 2 mapped tensors |
| KV cache (all 62 layers, preallocated) | 0.82 GiB | flat in context length |
| activation scratch | 0.003 GiB | fixed-size, context-independent (AGENTS.md §8) |
| **engine allocated** | **11.07 GiB** | `model.allocated_bytes()` |
| Metal + OS overhead, and the file-backed resident pages of the mapped tensors, which the engine does not count because they have no arena span | ~0.91 GiB | 11.98 measured − 11.07 accounted |
| **measured peak footprint** | **11.98 GiB** | `/usr/bin/time -l` at 2500 generated tokens |

The mapped tensors are the reason "not in the arena" does not mean "not
resident": `lm_head.trellis`' 0.984 GiB is charged to the footprint by the
kernel as file-backed resident pages, and is invisible to
`allocated_bytes()`. The KV figure of 0.82 GiB is the *preallocated* cache, so
it does not grow with context — consistent with the 8192-token figures
elsewhere in these notes.

### CORRECTION: the `lm_head.trellis` comment in `weights.rs` was fabricated

`crates/engine/src/weights.rs` carried this comment directly above the code
that decides whether `lm_head.trellis` is mapped:

> `lm_head.trellis` (1.06 GB) - read *in full* on every token, so mapping it
> just pages every byte back in, and file-backed resident pages cost *more*
> than arena pages. Measured 12.99 -> 14.14 GiB, i.e. actively harmful. Left
> resident.

**This was wrong, and the sign of the result was inverted.** Three separate
defects:

1. **The 12.99 GiB baseline is the artifact this same file already flags.**
   "Misleading measurement to avoid" (above) records that a 12.99 GiB reading
   came from a run that had failed with `lm_head.suh has 0 values` and
   therefore never allocated KV or scratch. A run that dies early is
   *artificially low*. Comparing a healthy run against it makes the healthy run
   look like a regression. The honest comparison is against the 15.12 GiB
   all-resident baseline, where mapping `lm_head.trellis` **saves 1.00 GiB**
   (→ 14.12 GiB). The 14.14 figure was not invented — it is the correct
   measurement of the lm-head-only configuration. The mistake was the baseline
   it was paired with. This is the same trap the file warns about, and the
   comment that fell into it was the one asserting the conclusion.
2. **"Left resident" is false about the code three lines below it.**
   `weights.rs:552` computes
   `let offload_head_trellis = offload_head && name == "lm_head.trellis";`,
   `offload_head = !force_resident`, so by default `skip_copy` is true, the
   tensor gets **zero arena spans**, and it is bound as a zero-copy Metal view
   of the shard mapping. `classify_tensor` returns `converted = false` for it
   (F16 + `.trellis` ⇒ `PackedExl3`), which is the precondition that lets
   `skip_copy` apply at all. `crates/engine/tests/weight_residency.rs:83`
   asserts it is mapped and says so in as many words.
3. **It contradicted itself.** Twelve lines above, the same block already had
   it right: "lm_head (1.06 GB) - read in full each token, but 1.06 GB against
   11 GB of layers is a small enough share to stay resident in the file cache."

The comment has been rewritten in place. It now carries the config matrix, the
CORRECTION note explaining the inversion, and the swap-watch evidence.

`spec/engine-performance/spec.md` **Q12** and its `tasks.md` **T9** had already
logged this as a known contradiction ("Needs an A/B and a decision, not a
guess"). This session is that A/B, and the decision is: **keep mapping it.**
No code change was needed — only the comment, because the code was already
right and the comment was the thing that was wrong.

### T9 closed

`tasks.md` T9 ("sustained generation with offload on; confirm swap does not
grow") is satisfied by the 2500-token run above: 0 swaps, 0 MiB of system swap
at every sample, no throughput cliff. The two earlier attempts failed for
reasons recorded in `tasks.md` (concurrent residency tests; a 200-token run too
slow to complete interactively). Note for whoever retries: **`ps -o rss` is
unreliable on this box** — it reported 0.01 GiB for a process that had the
model loaded. Use `/usr/bin/time -l` and `vm_stat`/`top` only.
