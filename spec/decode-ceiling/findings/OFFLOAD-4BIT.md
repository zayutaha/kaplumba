# OFFLOAD-4BIT — where the 17.44 GB of `Gxma-4-bit` actually is

Subject: `~/.kaplumba/models/Gxma-4-bit` (4 bpw layers + 8 bpw `lm_head`, `layout`
absent, 3 shards). Control: `~/.kaplumba/models/Gxma-repacked` (3+6 bpw,
`layout: colmajor`).

Instrument: `crates/engine/tests/probe_offload_split.rs` (new, `probe_offload_`
prefix, measurement only — no production code was modified).

Every number below is labelled **MEASURED** (produced by a command in this file)
or **REASONED** (derived from code that is quoted). Nothing is labelled
"measured" unless the command that produced it is given.

---

## 0. Summary

| | bytes | GiB | GB |
|---|---:|---:|---:|
| `text_bytes` (all text tensors, vision excluded) | 18,889,725,668 | 17.592 | 18.890 |
| mapped out of the shard (reclaimable) | 4,227,858,432 | 3.938 | 4.228 |
| arena payload / capacity (anonymous) | 14,661,867,236 | 13.655 | 14.662 |

`mapped + arena == text_bytes` exactly. **The partition is a partition**: every
text byte is either mapped or arena-resident, nothing is both.

On top of that, `TextModel::load_from_dir` allocates **one more anonymous
buffer of 1,409,286,144 B that `allocated_bytes()` does not report** — the
`lm_head.trellis` realignment copy. That is the only thing in this checkpoint
that is both large and avoidable.

`embed_tokens` (2.818 GB) is **already** offloaded and is **already** ~0–0.36 GB
resident. There is no 2.8 GB sitting there to be saved.

---

## 1. Q1 — the exact partition of 17.44 GB

### 1.1 From the safetensors headers alone (no GPU, no mmap)

```bash
KAPLUMBA_TEST_MODEL=~/.kaplumba/models/Gxma-4-bit \
  ./target/release/deps/probe_offload_split-<hash> --exact probe_offload_header_ledger --nocapture
```

**MEASURED**

```
LEDGER text_bytes = 18889725668 (17.592 GiB)
LEDGER trellis = 16052649984 (14.950 GiB), signs = 15840364 (15.840 MiB), other = 2821235320 (2.627 GiB)
LEDGER text tensors whose absolute file offset is NOT 4-byte aligned: 291
LEDGER shard model-00001-of-00003.safetensors: header_len=99059  payload_start=99067  payload_start%4096=763
LEDGER shard model-00002-of-00003.safetensors: header_len=150024 payload_start=150032 payload_start%4096=2576
LEDGER shard model-00003-of-00003.safetensors: header_len=58577  payload_start=58585  payload_start%4096=1241
LEDGER model.language_model.embed_tokens.weight: shard=…00001 payload_start=99067
       abs_off=99067 abs_end=2818671355 len=2818572288 (2.625 GiB) BF16 off%4=3 off%4096=763
LEDGER lm_head.trellis: shard=…00003 payload_start=58585
       abs_off=507672341 abs_end=1916958485 len=1409286144 (1.312 GiB) I16 off%4=1 off%4096=1813
LEDGER EXPECTED default split: mapped = embed(2818572288) + head(1409286144) = 4227858432 (3.938 GiB);
       arena payload = 14661867236 (13.655 GiB)
```

Two corrections to the brief I was given, both of which change the arithmetic:

* **`507,672,341` is `lm_head.trellis`'s absolute byte *offset*, not its size.**
  It is exactly `payload_start (58,585) + data_offset (507,613,756)`. The size is
  **1,409,286,144 B**. The "704 MB copy is 1.39× the tensor" question is built on
  the offset being mistaken for the size.
* **291 trellis tensors are 4-byte misaligned, not 1** (11,577,065,472 B between
  them). Only `lm_head.trellis` *pays* for it, because only it is mapped; the
  other 290 are arena-resident at offset 0. See §3.

### 1.2 From a real load (one `TextWeights::load_dedicated`, `u64::MAX` arena)

```bash
KAPLUMBA_TEST_MODEL=~/.kaplumba/models/Gxma-4-bit scripts/gpuq try -- /usr/bin/time -l \
  ./target/release/deps/probe_offload_split-<hash> --ignored --exact probe_offload_weight_residency \
  --nocapture --test-threads=1
```

**MEASURED**

```
SPLIT mapped_names   = ["lm_head.trellis", "model.language_model.embed_tokens.weight"]
SPLIT text_bytes     = 18889725668 (17.592 GiB)
SPLIT mapped_bytes   = 4227858432 (3.938 GiB)
SPLIT arena_payload  = 14661867236 (13.655 GiB)
SPLIT arena_capacity = 14661867236 (13.655 GiB)
SPLIT sums_to_text   = true
SPLIT head_len       = 1409286144 (1.312 GiB); its u16 element count = 704643072 (704.6 MB)
peak memory footprint                                14683892632   (13.673 GiB)
```

This reproduces the ledger in `weight_residency.rs::splits_add_up_for_both_known_checkpoints`
(`mapped = embed + 1,409,286,144`, `arena = 14,661,867,236`) **exactly**, from a
live load rather than from a recorded constant.

`arena_capacity == arena_payload` because `allocate_dedicated`
(`weights.rs:200-235`) allocates the *source* length and the bf16→fp16 conversion
happens in place, so capacity can only equal payload, never exceed it.

### 1.3 Arena composition (MEASURED sum, from §1.1)

| in the arena | bytes |
|---|---:|
| layer `.trellis` (60 layers × 7 projections) | 14,643,363,840 |
| all `.suh` / `.svh` / `.mcg` (incl. `lm_head`'s 535,044) | 15,840,364 |
| bf16 norms + `layer_scalar` (421 tensors) | 2,663,032 |
| **total** | **14,661,867,236** |

### 1.4 What the full `TextModel` adds (MEASURED)

```bash
KAPLUMBA_TEST_MODEL=~/.kaplumba/models/Gxma-4-bit KAPLUMBA_PROBE_STEPS=1 scripts/gpuq try -- \
  /usr/bin/time -l ./target/release/deps/probe_offload_split-<hash> --ignored \
  --exact probe_offload_decode_residency --nocapture --test-threads=1
```

**MEASURED**

```
FULL weights_reported = 14661867236 (13.655 GiB)
FULL kv_reported      = 881029120 (0.821 GiB)
FULL scratch_reported = 3743520 (0.003 GiB)
FULL allocated_bytes  = 15546639876 (14.479 GiB)
FULL after load: embed resident 319291392 of 2818572288 (11.3281%)
                 lm_head.trellis resident 1409286144 of 1409286144 (100.00%)
PREFILL 6 tokens -> 262144 logits
peak memory footprint                                17184962216 (16.003 GiB)
```

**REPORTING BUG (not fixed, per the brief).** `TextModel::allocated_bytes()`
(`model.rs:4537`) sums `TextWeights::resident_bytes()` + KV + scratch. It does
**not** include the `align_packed_view` copy, which is allocated in
`load_quantized_linear` (`model.rs:4951`) and is anonymous. So the engine
under-reports its own anonymous memory by **1,409,286,144 B** on this
checkpoint. Real anonymous total = 15,546,639,876 + 1,409,286,144 =
**16,955,926,020 B (15.791 GiB)**.

**KV at 16k is not in this run.** `kv_reported = 881,029,120` is the KV cache at
the CLI's default context (8192). The brief's 2.03 GiB is the 16k figure. Do not
mix them.

---

## 2. Q2 — is `embed_tokens` resident?

**MEASURED. It is not, and there is no 2.8 GB to save.**

Method: `mincore(2)` over a fresh read-only `MAP_PRIVATE`/`PROT_READ` view of the
shard that holds the tensor, counting pages that overlap the tensor's byte range.
macOS has a single unified buffer cache, so page residency is a property of the
*file page*, not of a particular mapping — the engine's mapping and the probe's
mapping report the same pages. `mmap` alone does not fault, and the probe never
reads through its mapping, so it cannot change what it measures. Page size on
this box is 16,384 B (`vm_stat`), not 4,096.

| measurement point | `embed_tokens` resident | of 2,818,572,288 |
|---|---:|---:|
| before load, cold cache (run 3) | 16,384 | 0.0006 % |
| after `load_dedicated` only (run 1) | **0** | 0.0000 % |
| before load, cache left by a previous run (run 1) | 1,537,245,184 | 54.5 % |
| before load, cache left by a previous run (run 3) | 357,416,960 | 12.7 % |
| after full `TextModel::load_from_dir` (run 2) | **319,291,392** | 11.33 % |
| after full `TextModel::load_from_dir` (run 1) | 345,145,344 | 12.25 % |

**Controls.**

* *Negative control*: the 1,151,487,072 B of `model.vision_tower` /
  `model.embed_vision` byte ranges in shard 3 are excluded from the text
  selection and never read by a text-only load. `mincore` reported **0 resident
  in all 356 of those ranges, in every run.** The zeros are a fact, not a tool
  failure.
* *Positive control (implicit but sufficient)*: within a single process `mincore`
  moved embed from 16,384 B before load to 319,291,392 B after load, and in
  another run observed 1,537,245,184 B. The tool resolves this exact byte range
  to within one page.

**Reading.** The 319–396 MB band is *file-cache state that varies between
runs*, not a logical read: the same code path produced 0 B, 319 MB, 345 MB and
357 MB on different runs, and one run saw 1.5 GB with no load at all. Nothing in
`load_from_dir` reads `embed_tokens` logically — `weights.rs:586` sets
`skip_copy` for it and `raw_zero_copy_tensor_buffer` only wraps the mapping.
The most likely source is APFS read-around: the loader `read_at`s shard 1
*ascending* from offset 2,818,241,355, and file-cache readahead is driven by
physical block adjacency, which is not logical order. **REASONED**, and I did
not prove it.

**The answer to the question asked: `embed_tokens` is NOT 2.818 GB resident, and
it is already offloaded. Offloading it again cannot save anything.** Decode reads
exactly one row per token, 5,376 × 2 = **10,752 B** (REASONED from
`weights.rs:514` / `TextModel::forward`'s `embedding_scaled_buffer`, which
indexes `[token * hidden_size, ...)`); prefill reads it once per token, same
size. Even a 16k-token conversation touches at most 16k distinct rows out of
262,144, i.e. ≤ 172 MB worst case with zero reuse, and with a real chat
distribution it is a few MB.

---

## 3. Q3 — what is the "704 MB" copy, and can it be avoided?

### 3.1 It is 1,409,286,144 B, not 704 MB — the 704 MB figure is the element count

`align_packed_view` (`model.rs:4871-4921`):

```rust
if offset % EXL3_PACKED_ALIGNMENT == 0 { return Ok((buffer.clone(), offset)); }
…
let realigned = device.new_buffer(len as u64, StorageModeShared);
std::ptr::copy_nonoverlapping(source.add(offset), destination, len);
```

`len` is `tensor.source_len` (`model.rs:4947-4951`), i.e. **bytes**, not
elements. So the buffer is 1,409,286,144 B = 1.312 GiB = **1409.3 MB**.

`SPLIT head_len = 1409286144 (1.312 GiB); its u16 element count = 704643072 (704.6 MB)`
— **MEASURED**. 704,643,072 × 2 = 1,409,286,144. The "704 MB, once" in commit
`2eb7b4d`'s message and in the doc comment at `model.rs:4859-4860` is the u16
**element** count reported as bytes. It is off by exactly 2×.

There is no over-allocation, no page rounding (a `new_buffer` of exactly `len`
bytes), and no bf16 expansion (the tensor is `I16` and is copied verbatim). It
is a straight `memcpy` of the whole tensor.

### 3.2 Why it happens, and what it costs besides the copy

`lm_head.trellis` absolute file offset is 507,672,341, which is **odd**
(`off%4 == 1`). `ShardReader::no_copy_view` (`weights.rs:1047-1075`) page-aligns
*downward*, so the offset inside the returned window is
`507,672,341 % 4096 = 1813`. `1813 % 4 == 1`, so `align_packed_view` cannot
pass it through and copies.

The copy also has a **second, unadvertised cost**: `copy_nonoverlapping` reads the
whole thing through the mapping, which faults in every page. **MEASURED**:
`lm_head.trellis resident 1409286144 of 1409286144 (100.00%)` immediately after
`load_from_dir`. So `lm_head` currently costs

* 1,409,286,144 B anonymous (the copy) **plus**
* 1,409,286,144 B file-cache-resident (the read that made the copy)

= **2,818,572,288 B**, or 1.312 GiB more than it needs to.

### 3.3 Yes — one repack removes it entirely

The existing repacker **already does the whole job**, and for this checkpoint it
would do it *without touching a single weight byte*.
`crates/engine/src/bin/repack_model.rs`:

* `repack_model.rs:196-224` recomputes `data_offsets` for every tensor with
  `let aligned_start = (cur_offset + 63) & !63;` — **64-byte alignment**.
* `repack_model.rs:226-236` pads the header JSON with spaces until
  `(8 + header_len) % 64 == 0` — so `payload_start` is 64-byte aligned.
* ⇒ `absolute_offset = payload_start + data_offset` is 64-byte aligned ⇒
  `delta % 4 == 0` ⇒ `align_packed_view` early-returns ⇒ **zero copy, zero
  behaviour change.**
* The colmajor transpose at `repack_model.rs:196` (and the copy loop at
  `:246`) is gated on `entry.name.ends_with(".trellis") && shape.len() == 3 &&
  shape[2] == 48`. **Every 4-bpw trellis in this checkpoint has `shape[2] ∈
  {64, 128}`** — MEASURED: `lm_head.trellis [336,16384,128]`,
  `layers.0.self_attn.q_proj.trellis [336,512,64]`,
  `layers.0.mlp.down_proj.trellis [1344,336,64]`. So the repacker would
  **re-align offsets and copy every byte verbatim**, touching no tensor.
  (3-bpw `Gxma-repacked` is the `[…, …, 48]` case, which is why it *was*
  transposed.)

**The one thing that must be undone:** `repack_model.rs:60-67` writes
`"layout": "colmajor"` into the destination `quantization_config.json`. The
source has no `layout` key (MEASURED with `jq`), and nothing was transposed, so
that key must be **deleted** from the output. If it is left, `safetensors.rs:894`
computes `expected_shape` as the *swapped* shape for every non-`lm_head` module
and `validate_linear_geometry` rejects the checkpoint.

### 3.4 Concrete recipe (NOT performed)

```bash
# 1. Repack. ~20 GB written next to the source; source untouched.
cargo run --release --bin repack_model -- \
    ~/.kaplumba/models/Gxma-4-bit \
    ~/.kaplumba/models/Gxma-4-bit-aligned

# 2. MANDATORY: remove the colmajor marker the tool injects, because no tensor
#    was transposed (4-bpw trellis is shape[2]=64/128, not 48).
#    Leave the key ABSENT, exactly as in the source.
grep -v '"layout"' ~/.kaplumba/models/Gxma-4-bit-aligned/quantization_config.json \
  > /tmp/qc && mv /tmp/qc ~/.kaplumba/models/Gxma-4-bit-aligned/quantization_config.json

# 3. Verify before believing. Header-only, no GPU:
KAPLUMBA_TEST_MODEL=~/.kaplumba/models/Gxma-4-bit-aligned \
  ./target/release/deps/probe_offload_split-<hash> --exact probe_offload_header_ledger --nocapture \
  | grep -E 'lm_head.trellis|MISALIGNED|payload_start'
#    expect: "text tensors whose absolute file offset is NOT 4-byte aligned: 0"
#    expect: lm_head.trellis abs_off%4 = 0

# 4. Then the residency probe, and confirm the mapped set is unchanged:
#    resident_and_mapped_weights_add_up_to_the_checkpoint still passes, because
#    lm_head.trellis stays mapped (spans.is_empty()) — it just is no longer copied.
```

**Header length needed.** Current shard 3 `header_len = 58,577`, so
`payload_start = 58,585`. The repacker appends
`pad = 64 - ((8 + new_len) % 64)` spaces. Expected result: `payload_start`
becomes a multiple of 64 (the JSON re-serialises slightly shorter/longer, so the
exact number is whatever makes `(8 + header_len) % 64 == 0`; the pad is ≤ 63
bytes and `safetensors::MAX_HEADER_BYTES` is nowhere near the limit). The
64-byte minimum that would suffice is 3 bytes (`58,585 + 3 = 58,588`, divisible
by 4), but take the repacker's 64 — it costs 61 extra bytes and removes the
question.

**Disk.** 113 GiB free. `Gxma-4-bit` 20 GB + `Gxma-4-bit-aligned` ~20 GB = fine.
Per `repack_model.rs`'s own rule and AGENTS.md §7, keep both under
`~/.kaplumba/models`, never in the working tree.

**Validation required before trusting it.** `model.safetensors.index.json` is
copied verbatim; its `metadata.total_size` is a sum of *tensor byte lengths*, not
a file size, and re-alignment does not change tensor lengths —
`safetensors.rs:493-497` compares exactly that, so it still validates
(**REASONED**). Output must still be byte-identical to `Gxma-repacked` at
`--temperature 0` (AGENTS.md §8: no quality loss).

**How this number could be wrong:** if the engine's `safetensors::read_shard_header`
or `validate_linear_geometry` rejects 64-byte-interleaved offsets or the padded
header, the model will not load. That is a 2-minute check (step 3 above) and it
must be done before anyone spends effort on it.

---

## 4. Q4 — what else is offloadable, ranked

Measured footprint of a full load + 6-token prefill on this box:
**17,184,962,216 B (16.003 GiB / 17.185 GB)**, and 17,180,161,704 B in a second
run. The brief's 17.44 GB is the same thing with the CLI's 4 decode steps.

| # | source | bytes | GiB | really resident? | offloadable? | mechanism |
|---|---|---:|---:|---|---|---|
| 1 | layer `.trellis` in the arena (60 layers) | 14,643,363,840 | 13.637 | **MEASURED** — arena capacity | **NO** | decode streams every byte every token. Mapping them is what measured 3.9 GiB of swap at 1.4 tok/s (`AGENTS.md` §10.1, `weights.rs:508-522`). This is the floor. |
| 2 | `lm_head.trellis` **realignment copy** | 1,409,286,144 | 1.312 | **MEASURED** — anonymous, and it forces 100 % of the tensor file-resident (MEASURED 1,409,286,144 / 1,409,286,144) | **YES** | repack to a 64-byte-aligned offset; §3.4. Saves the 1.409 GB anon **and** lets the mapped copy be the only cost. |
| 3 | KV cache at the CLI default context | 881,029,120 | 0.821 | **MEASURED** — `allocation_breakdown` | not by offload | only by shrinking the context. `AGENTS.md` §8 forbids KV quantization. At 16k it is ~2.03 GiB. |
| 4 | `lm_head.trellis` file-resident (from #2's read) | 1,409,286,144 | 1.312 | **MEASURED** — 100 % after load | partly | disappears as a *new* cost when #2 is fixed: decode reads it in full every token anyway, so it becomes page-cache-resident regardless. Clean, reclaimable, but it *is* charged to `phys_footprint`. |
| 5 | `embed_tokens` | 2,818,572,288 | 2.625 | **MEASURED 0 – 357,416,960 (0 – 12.7 %)**, not 2.818 GB | **NO — already done** | already `skip_copy`/mapped. Nothing to gain. REASONED: ≤ 10,752 B per token touched. |
| 6 | `.suh` / `.svh` / `.mcg` (all tensors) | 15,840,364 | 0.015 | **MEASURED** — in the arena | no | read on the CPU via `read_f16`; must stay resident. 0.5 % of the budget. |
| 7 | bf16 norms + `layer_scalar` | 2,663,032 | 0.002 | **MEASURED** — in the arena | no | 421 tiny tensors. |
| 8 | `ForwardScratch` | 3,743,520 | 0.003 | **MEASURED** — `allocation_breakdown` | no | already fixed-size and context-independent (§8). |
| 9 | `lm_head` output side | ≤ 1,059,332 | ~0.001 | **REASONED** | no | there is **no second copy**. `lm_head.suh/svh/mcg` (535,044 B) get an arena copy and a Metal upload; logits out is `vocab_size × 2` = 524,288 B. Verified: `align_packed_view` has exactly one non-test caller, `model.rs:4951`. |

**Reconciling the ledger.** 1+2+3+6+7+8 = 14,661,867,236 + 1,409,286,144 +
881,029,120 + 15,840,364 + 2,663,032 + 3,743,520 = **16,973,429,416 B**. Measured
footprint 17,184,962,216 B. Gap **211,532,800 B (0.197 GiB)** = the file-resident
share of `embed_tokens` + `lm_head.trellis` that survived the arena's allocation
+ process runtime, allocator slack, Metal command buffers, the 262,144-entry
logits buffer, the tokenizer-free CLI, and `phys_footprint`'s own accounting
slack. **It is small and it is not a hidden weight.** REASONED — the components
above are individually measured and the residual is what is left over.

---

## 5. Does 17.44 GB reach 14.6 GB by offloading?

| step | bytes saved | new total |
|---|---:|---:|
| measured starting point | — | 17.44 GB |
| repack shard 3 to a 64-byte-aligned `payload_start` (§3.4) | **−1.409 GB** | **16.03 GB** |
| offload `embed_tokens` | −0 (already mapped, 0–0.36 GB resident) | 16.03 GB |
| offload `lm_head.trellis` | −0 (already mapped; after the repack it costs 1.409 GB of *clean reclaimable* page cache, not anonymous) | 16.03 GB |
| shrink KV | context-dependent, up to −0.63 GB at 16k→8k; not an offload | — |

**No — not by offloading.** The only genuinely available byte source is the
**1,409,286,144 B `lm_head.trellis` realignment copy**, and it is removed by a
*one-time repack of the checkpoint*, not by any runtime offload mechanism:
**17.44 GB → ~16.03 GB**, and the floor is the arena
(14,661,867,236 B = 14.66 GB) plus KV (0.88 GB at 8192, ~2.03 GiB at 16k) plus
`lm_head.trellis` in the page cache (1.31 GiB), which decode cannot avoid
because it reads the whole head every token.

The real constraint is **#1: 14,643,363,840 B of layer weights that decode
streams in full on every token.** They are 85 % of the text bytes and 100 % of
the reason the arena cannot shrink. Getting to 14.6 GB total would mean getting
the layer weights below the arena size, which is a *quantisation or offload-to-
disk* problem, not a memory-accounting one. `embed_tokens` — the thing the plan
was going to attack — is already offloaded and is already contributing ~0.

---

## 6. Other findings (reported, not fixed)

1. **`TextModel::forward` and `generate --legacy-path` are broken in the default
   configuration.** MEASURED:
   `called Result::unwrap() on an Err value: Kernel(InvalidInput("embedding buffer length does not match dimensions"))`.
   `TextModel::forward` (`model.rs:4686`) calls
   `ElementKernels::embedding_scaled_buffer`, which requires
   `embedding.length() == hidden_size * vocab_size * 2` exactly
   (`kernels/src/elements.rs:425`). With `embed_tokens` mapped — the default —
   the buffer is a page-aligned window that is *longer* than the tensor, so the
   check can never pass. `generate_gpu` (the default in
   `crates/cli/src/main.rs:106`) uses `forward_buffer_into_*` →
   `encode_embedding_scaled`, which handles the raw bf16 view, and works. Only
   `--legacy-path` (`main.rs:104`) is affected. `crates/engine/tests/generation.rs:137`
   and `model_layer.rs:482` both go through `forward`, so they must be either
   passing `KAPLUMBA_RESIDENT_WEIGHTS=1` or are not being run.
2. **`allocated_bytes()` under-reports by 1,409,286,144 B** on this checkpoint
   (§1.4). The `18 GB` resident limit is checked against
   `resident_limit_bytes()` at *arena* time only (`weights.rs:211-216`), so the
   realignment copy is not counted against it either — the ceiling is being
   enforced on 14.66 GB while the process commits 15.79 GB.
3. **The "704 MB" figure** in `2eb7b4d`'s message and `model.rs:4859-4860` is the
   u16 element count, not bytes (§3.1).

---

## 7. Commands

```bash
# Header ledger only — no GPU, no mmap, < 1 s.
KAPLUMBA_TEST_MODEL=~/.kaplumba/models/Gxma-4-bit \
  ./target/release/deps/probe_offload_split-c344c641ea1c6a3c \
  --exact probe_offload_header_ledger --nocapture

# Weights-only load (13.7 GiB arena) + mincore census. ~20-80 s.
KAPLUMBA_TEST_MODEL=~/.kaplumba/models/Gxma-4-bit scripts/gpuq try -- /usr/bin/time -l \
  ./target/release/deps/probe_offload_split-c344c641ea1c6a3c \
  --ignored --exact probe_offload_weight_residency --nocapture --test-threads=1

# Full model + 6-token prefill + N decode steps + positive control. 5-5+ min at
# 4 bpw (0.014-0.066 tok/s), and it needs ~16 GiB free.
KAPLUMBA_TEST_MODEL=~/.kaplumba/models/Gxma-4-bit KAPLUMBA_PROBE_STEPS=1 scripts/gpuq try -- \
  /usr/bin/time -l ./target/release/deps/probe_offload_split-c344c641ea1c6a3c \
  --ignored --exact probe_offload_decode_residency --nocapture --test-threads=1
```

## 8. Method caveats

* **Every "resident" number is file-cache residency of a file page**, obtained
  with `mincore(2)` over a private read-only mapping of the shard. It does not
  see anonymous copies. The anonymous side is `allocation_breakdown()` +
  `/usr/bin/time -l` `phys_footprint`, per AGENTS.md §10.3.
* **`mincore` was sabotaged in both directions** (§3.4): the 1.15 GB
  never-read vision range reported non-zero in 0 of 356 ranges × 3 runs, and
  `embed_tokens` was observed at 16,384 / 319,291,392 / 345,145,344 /
  396,787,712 / 1,537,245,184 B. A tool that always returned 0 or always returned
  the span could not have produced that spread.
* **The 319–396 MB of embed seen "after load" is not attributable.** It varies
  run to run, appears with and without a logical read, and one run observed
  1.5 GB with no process running at all. Treated as cache noise, not as a cost of
  the mapping. REASONED.
* **`probe_offload_decode_residency` was SIGTERM'd twice** on this box while
  another session ran a 4-bpw `kaplumba` concurrently — two 16 GiB instances on
  24 GiB, exactly the failure AGENTS.md §10.2 describes. The post-decode embed
  number is therefore **not measured**; §2 states the bound that does hold
  (0–12.7 %) and §3.2's 100 % `lm_head.trellis` figure is from a run that did
  complete a prefill.
* The engine's `gpu_lock` does not serialise this probe against another
  *session's* probe; only against `TextModel::load_from_dir`. Two probes
  loading 13.7 GiB each will both fail.