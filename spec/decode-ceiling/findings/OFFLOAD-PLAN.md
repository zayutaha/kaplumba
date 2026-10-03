# OFFLOAD-PLAN — what can be offloaded from `Gxma-4-bit`, and what each choice costs

**Method: static analysis only.** No model was loaded, no GPU was touched, `scripts/gpuq`
was not run. The only model-file access was reading the safetensors **headers** (the JSON at
the front of each shard: 99,059 / 150,024 / 58,577 bytes). Code was read, not run.

Builds on `spec/decode-ceiling/findings/OFFLOAD-4BIT.md`, which measured the same partition
on a live load. Where this document disagrees with it, the disagreement is called out.

**Labels.**
`MEASURED` = from `/usr/bin/time -l` or a probe in a prior findings file (command given there).
`READ-FROM-HEADER` = computed here from the safetensors headers by `/tmp/offload-probe/*.mjs`.
`INFERRED` = derived from quoted code. `GIVEN` = supplied in the brief, not re-verified.
`SAW` = on record in a prior spec file.

---

## 0. Answer first

| question | answer |
|---|---|
| Can offloading take ~18 GB → 14.6 GB? | **NO.** Not by any mechanism. |
| Why not | The layer trellis arena alone is **14,661,867,236 B = 14.66 GB**, already over the 14.6 GB target before KV, scratch, runtime, or a single page-cache byte. |
| Can offloading take ~18 GB → ~16 GB? | **YES, but only by editing the checkpoint on disk**, not by any runtime offload switch: one 64-byte re-alignment removes 1,409,286,144 B. 17.44 GB → ~16.03 GB. |
| True minimum resident set, 4 bpw layers + 8 bpw head, at 8k context | **~16.18 GB anonymous + ~1.41 GB clean page cache ≈ 17.6 GB charged** (`INFERRED`); the MEASURED figure is 17.44 GB. |
| What would actually land at 14.6 GB | 3 bpw layers + a 6 bpw head — which is **`Gxma-repacked`, already on disk**. 4 bpw layers are the reason 14.6 GB is unreachable. |
| Droppable for text-only inference | **1,151,487,072 B of vision tensors — and the loader already drops them.** Nothing else. |

---

## 1. Deliverable 1 — the inventory

All byte counts `READ-FROM-HEADER`. Access patterns `INFERRED` from the quoted code paths.

### 1.0 Checkpoint shape (`READ-FROM-HEADER`)

```
shard 1  header_len= 99,059  payload_start= 99,067  %4096=763  %4=3
shard 2  header_len=150,024  payload_start=150,032  %4096=2576 %4=0
shard 3  header_len= 58,577  payload_start= 58,585  %4096=1241 %4=1

all tensors      2,422   20,041,212,740 B (18.665 GiB)
  vision tower     356    1,151,487,072 B (1.072 GiB)   <- EXCLUDED, see 1.6
  text           2,066   18,889,725,668 B (17.592 GiB)
```

`config.json` (`READ-FROM-HEADER`): 60 layers, **50 sliding + 10 full** (`layer_types`;
full = indices 5, 11, 17, 23, 29, 35, 41, 47, 53, 59), hidden 5376, intermediate 21504,
vocab 262144, 32 query heads, 16 KV heads, head_dim 256 / global 512, 4 global KV heads,
sliding_window 1024, **`attention_k_eq_v: true`**, `quantization_config.bits = 4`,
`head_bits = 8`.

### 1.1 Text tensor groups — the five buckets

| group | n | bytes | GiB | storage today | access pattern | bytes touched per token | offload verdict / re-read cost |
|---|---:|---:|---:|---|---|---|---|
| **layer `.trellis`** | 410 | 14,643,363,840 | 13.638 | anonymous arena, offset 0 | **full sweep, every token** | **14,643,363,840** (100 %) | **TRAP.** Offloading moves 13.64 GB to reclaimable pages and charges a full disk round trip for all of it, every token. Already measured: mapping layers produced 3.9 GiB of swap at 1.4 tok/s (`AGENTS.md` §10.1, `weights.rs:508-522`). |
| **`lm_head.trellis`** | 1 | 1,409,286,144 | 1.313 | **mapped**, then **copied** (see 1.3) | **full sweep, every token** | **1,409,286,144** (100 %) | Already offloaded. Second copy is the one avoidable byte in the process. |
| **`embed_tokens`** | 1 | 2,818,572,288 | 2.625 | **mapped** (`skip_copy`) | 1 row per token; 1 row per prefill token | **10,752** (0.00038 %) | **Already done.** MEASURED 0 – 357,416,960 B resident (0 – 12.7 %), `OFFLOAD-4BIT` §2. Re-read cost is 10.7 KB/token forever — free. |
| **`.suh` / `.svh` / `.mcg`** | 1,233 | 15,840,364 | 0.015 | anonymous arena | **read once at load** on the host, then live in their own Metal buffers | **0 host bytes** | **Free but pointless.** `read_sign_component` → `weights.read_f16` → `Vec` → `from_metal_buffer` (`model.rs:4983-5000`, `3483-3493`). After load the arena copy is dead. 0.015 GB. |
| **norms + `layer_scalar` + final norm** | 421 | 2,663,032 | 0.002 | anonymous arena | **read once at load**, same path | **0 host bytes** | Same. 0.002 GB. |
| **total text** | **2,066** | **18,889,725,668** | **17.592** | | | **16,052,649,984** | |

`18,889,725,668 = 14,643,363,840 + 1,409,286,144 + 2,818,572,288 + 15,840,364 + 2,663,032` —
**exact**. No text tensor is unaccounted for.

### 1.2 Per-projection detail, all 60 layers (`READ-FROM-HEADER`)

| projection | n | trellis shape | each | group bytes | full layer | sliding layer |
|---|---:|---|---:|---:|---|---|
| `self_attn.q_proj` | 60 | `336×512×64` / `336×1024×64` | 22,020,096 / 44,040,192 | 1,541,406,720 | 1024 out (32×512) | 8192 out (32×256) |
| `self_attn.k_proj` | 60 | `336×256×64` / `336×128×64` | 11,010,048 / 5,505,024 | 605,552,640 | 2048 out | 4096 out |
| `self_attn.v_proj` | **50** | `336×256×64` | 11,010,048 | 550,502,400 | **absent** | 4096 out |
| `self_attn.o_proj` | 60 | `512×336×64` / `1024×336×64` | 22,020,096 / 44,040,192 | 1,541,406,720 | 16384 in | 8192 in |
| `mlp.gate_proj` | 60 | `336×1344×64` | 57,802,752 | 3,468,165,120 | 21504 out | 21504 out |
| `mlp.up_proj` | 60 | `336×1344×64` | 57,802,752 | 3,468,165,120 | 21504 out | 21504 out |
| `mlp.down_proj` | 60 | `1344×336×64` | 57,802,752 | 3,468,165,120 | 21504 in | 21504 in |
| | **410** | | | **14,643,363,840** | 2,669,936,640 | 11,973,427,200 |

`v_proj` is missing on exactly the 10 full-attention layers because `attention_k_eq_v: true`;
the loader agrees — `v_proj: if full { None } else { Some(...) }` (`model.rs:3446-3456`).
The MLP is 10,404,495,360 B = **71.0 %** of the arena and is identical on every layer.

### 1.3 `lm_head.*` side tensors and the "704 MB" question (`READ-FROM-HEADER`)

| tensor | dtype | shape | bytes | abs offset | abs%4 | abs%4096 |
|---|---|---|---:|---:|---:|---:|
| `lm_head.suh` | F16 | `[5376]` | 10,752 | 507,137,297 | 1 | 3345 |
| `lm_head.svh` | F16 | `[262144]` | 524,288 | 507,148,049 | 1 | 1809 |
| `lm_head.mcg` | I32 | `[]` | 4 | 507,672,337 | 1 | 1809 |
| **`lm_head.trellis`** | I16 | `[336, 16384, 128]` | **1,409,286,144** | **507,672,341** | **1** | **1813** |
| | | | **1,964,188** | | | |

* **`507,672,341` is `lm_head.trellis`'s absolute file byte OFFSET, not its size.** It is exactly
  `payload_start (58,585) + data_offset (507,613,756)`. The size is **1,409,286,144 B**.
* **The "704 MB for a 508 MB tensor = 1.39x" question is an artefact of comparing two different
  quantities.** `704,643,072` is the **u16 element count** (`1,409,286,144 / 2`) printed as if
  it were megabytes. `704,643,072 / 507,672,341 = 1.388` — that is where 1.39x comes from.
  `OFFLOAD-4BIT` §3.1 reached the same conclusion; confirmed here from the header.
* There is **no over-allocation**: `align_packed_view` allocates `new_buffer(len)` with
  `len = tensor.source_len` in **bytes** and `copy_nonoverlapping`s verbatim
  (`model.rs:4894-4920`). No bf16 expansion (dtype is `I16`), no page rounding.
* Consequence (`INFERRED`): the copy faults in **100 % of the tensor** through the mapping, so
  `lm_head` currently costs **1,409,286,144 anonymous + 1,409,286,144 clean file-backed =
  2,818,572,288 B** — 1.312 GiB more than it needs to. MEASURED 100 % file residency
  (`OFFLOAD-4BIT` §1.4).
* The head's side tensors total 535,044 B and get an arena copy **and** a Metal upload
  (`model.rs:4975-4980`). `align_packed_view` has exactly one non-test caller
  (`model.rs:4951`), so there is no second head copy.

### 1.4 KV cache (`INFERRED`, reproduces a MEASURED number exactly)

`GpuKvCache::new` (`model.rs:1078-1090`), `stream_len = kv_heads × head_dim`,
`bytes = 2 × capacity × stream_len × 2 + capacity × 4`:

| group | n | capacity | stream_len | each | group |
|---|---:|---:|---:|---:|---:|
| sliding (`kv_heads 16`, `head_dim 256`) | 50 | 1024 (`sliding_window`) | 4096 | 16,781,312 | **839,065,600** |
| full (`kv_heads 4`, `head_dim 512`) | 10 | min(262144, **512**) = 512 | 2048 | 4,196,352 | **41,963,520** |
| | | | | **at load** | **881,029,120** |

**881,029,120 is exactly the MEASURED `kv_reported`** in `OFFLOAD-4BIT` §1.4. Growth:
`ensure_capacity` doubles (`gpu_cache.rs:220-241`), so the 10 full caches cost
`10 × (2 × C × 2048 × 2 + 4C)`:

| context | full-attn KV | sliding KV | total KV |
|---:|---:|---:|---:|
| at load (512) | 41,963,520 | 839,065,600 | **881,029,120** |
| 8,192 | 671,416,320 | 839,065,600 | **1,510,481,920** (1.407 GiB) |
| 16,384 | 1,342,832,640 | 839,065,600 | **2,181,898,240** (2.032 GiB) |

The 8192→16384 step alone is **+0.63 GB**, and it is the single largest term in the
unexplained footprint (§1.7). `AGENTS.md` §8 forbids KV quantisation; the only lever is context
length. `MAX_POLICY_CAPACITY = DEFAULT_CONTEXT_LIMIT = 32768`
(`gpu_cache.rs:50`), so 16k is reachable and 32k would add another 1.34 GB.

### 1.5 `ForwardScratch` (`model.rs:2947-3042`)

Built as `ForwardScratch::new(1, …)` at `model.rs:3201` — **`max_tokens = 1`** at load,
`INFERRED` from the call site. Fixed-size and context-independent per `AGENTS.md` §8.

* MEASURED `scratch_reported = 3,743,520` (`OFFLOAD-4BIT` §1.4).
* Dominated by `raw`/`raw_b`/`raw_c` (`max(1 × 21504, 262144)` f32 = 1,048,576 each),
  `logits` + `capped` (262,144 f16 = 524,288 each), and `attn_partials`
  (`decode_partial_elements(32, 512)` = 16 × 32 × 514 = 263,168 f32 = 1,052,672).
* **Reporting bug (not fixed — production code is out of scope here):**
  `ForwardScratch::resident_bytes` (`model.rs:3044-3079`) lists `&self.raw` but **omits
  `raw_b` and `raw_c`**, and omits `spin_scratch`. So `allocated_bytes()` under-reports by
  **2,097,152 B** here, on top of the 1,409,286,144 B head-copy omission (§1.7).

### 1.6 The vision tower — **already excluded, verified statically**

`safetensors.rs:464-478` in `load_text_selection`:

```rust
if is_vision_tensor_name(name) { excluded_vision.push(name.clone()); }
else { text_bytes += tensor.byte_len(); tensors.insert(...); }
```

`safetensors.rs:516-518`:
```rust
pub fn is_vision_tensor_name(name: &str) -> bool {
    name.starts_with("model.vision_tower.") || name.starts_with("model.embed_vision.")
}
```

**1,151,487,072 B across 356 tensors is never added to the text selection**, so it is never
allocated, never copied, never read. `ShardReader::open` does `mmap` each shard whole, but
mapping does not fault; `OFFLOAD-4BIT` §2 measured **0 resident bytes in all 356 ranges in
every run** (negative control). `validate_quantization_metadata` skips them too
(`safetensors.rs:527-529`). **This exclusion is already correct and already saving 1.15 GB.**

### 1.7 The gap — what the tensor list does *not* account for

| term | bytes | label |
|---|---:|---|
| arena (`text − embed − head`) | 14,661,867,236 | `INFERRED` (= MEASURED `arena_payload`, `OFFLOAD-4BIT` §1.2) |
| `align_packed_view` copy | 1,409,286,144 | `INFERRED` from `model.rs:4894-4920` |
| KV at load | 881,029,120 | `INFERRED` (= MEASURED `kv_reported`) |
| `ForwardScratch` reported | 3,743,520 | MEASURED |
| **anonymous subtotal** | **16,955,926,020** | |
| `raw_b` + `raw_c` omitted by `resident_bytes()` | 2,097,152 | `INFERRED` |
| norm / sign Metal duplicates left after load | ~18,503,396 | `INFERRED` |
| **full-attention KV growth 512 → 8192** | **+629,452,800** | `INFERRED` |
| **anonymous total at 8k context** | **17,605,979,368** | |
| `lm_head.trellis` clean file-backed page cache | 1,409,286,144 | MEASURED 100 % |
| **fully accounted charge at 8k** | **19,015,265,512** | |

The MEASURED 17.44 GB (`spec/decode-ceiling/findings/WHERE-THIS-STANDS.md:349`) sits *between*
the anonymous subtotal (16.96 GB) and the fully-accounted charge (19.02 GB). The residual is:

* **`raw_b`/`raw_c` + norm/sign duplicates**: ~20.6 MB — the engine under-reports itself.
* **`lm_head.trellis` page cache**: `phys_footprint` clearly does **not** charge all 1.31 GiB,
  or the run would have been 19 GB. It is *clean* file-backed, so the kernel counts it only
  while the pages hold memory pressure.
* **`embed_tokens` read-around**: 0 – 357,416,960 B, varies run to run, observed with and
  without a logical read (`OFFLOAD-4BIT` §2, §8). Unattributable; treat as cache noise.
* **Process runtime**: Metal command queue, pipeline states, the binary, the 32 MB
  `tokenizer.json`, the parsed 663 KB `quantization_config.json`. `INFERRED` ~100–300 MB.
* **The dominant gap term is KV growth (0.63 GB)**, then page-cache state (0–1.41 GB), then
  runtime. **None of it is a droppable weight.** The one exception is the 20.6 MB of
  under-reported buffers, which is a reporting bug, not memory.

### 1.8 `metal.rs`'s 18,046,374,518 does not reproduce

`metal.rs:7-9` claims "The 4 bpw / 8 bpw-head checkpoint needs 18,046,374,518 bytes
resident". **This is not derivable from `Gxma-4-bit`'s headers.** The arena it actually needs
is 14,661,867,236 B; `text_bytes` is 18,889,725,668; no combination of arena + KV + head copy
lands on 18,046,374,518. It is most likely a `phys_footprint` from an earlier layout, or it
belongs to `Gxma-bpw4-shim` (whose shards are dangling symlinks into `~/Downloads` and are
gone, so it cannot be checked). **Consequence: the 18,000,000,000 default limit in
`resident_limit_bytes()` is not binding on this checkpoint** — the arena is 3.3 GB under it.
Do not use 18.05 GB as "required resident" in any decision.

---

## 2. Deliverable 2 — the arithmetic

### 2.1 Minimum achievable resident set

Three classes of byte, and only two of them can move:

1. **Anonymous, unreclaimable.** Arena 14.66 GB + head copy 1.31 GB + KV + scratch + runtime.
   Only re-quantisation, a loader change, or a *checkpoint edit* shrinks this.
2. **Clean file-backed, reclaimable.** `embed_tokens` (10.7 KB/token) and `lm_head.trellis`
   (100 %/token). Free to shed under pressure, but the head is re-read in full every token, so
   it comes straight back. Net saving ≈ 0.
3. **Never read.** Vision tower 1.15 GB. Already in class "never touched".

**Floor, 4 bpw layers + 8 bpw head:**

| configuration | anonymous | clean page cache | charged |
|---|---:|---:|---:|
| at load (KV 512) | 15,548,737,028 | ≤1,409,286,144 | **≤16,958,023,172** |
| at 8k context | 16,178,189,828 | ≤1,409,286,144 | **≤17,587,475,972** |
| at 16k context | 16,849,606,148 | ≤1,409,286,144 | **≤18,258,892,292** |

`14,661,867,236 + 881,029,120 + 3,743,520 + 2,097,152` = 15,548,737,028 at load.
The MEASURED 17.44 GB is inside the 8k band. Good.

**The floor is the arena. 14,661,867,236 B = 14.66 GB.**

### 2.2 Does 17.44 GB reach 14.6 GB by offloading? **NO.**

Bluntly: **no.** Not approximately, not at 8k, not at 16k, not with `embed_tokens` mapped,
not with `lm_head` mapped, not with both.

* The arena alone is **14.66 GB**. The target is 14.6 GB. **The weight arena exceeds the entire
  budget before a single KV byte, scratch byte, runtime byte, or page-cache byte is added.**
* Everything else — KV 0.88–2.18 GB, scratch ~5.8 MB, runtime ~0.1–0.3 GB, `lm_head` page
  cache up to 1.31 GiB — is strictly *additive*.
* The 13.64 GB of layer trellis is the only large term, and it is read **in full on every
  token**, so it is precisely the tensor class the decision rule calls a trap. The measured
  consequence of ignoring that is on record: 3.9 GiB of swap, 1.4 tok/s (`AGENTS.md` §10.1).
* So the answer is not "offloading is expensive here" but "**offloading has nothing left to
  take**". The two tensors that could be offloaded were offloaded before this question was
  asked; the one that is still anonymous cannot be.

### 2.3 The one real saving, and it is a disk edit

| step | bytes | new total |
|---|---:|---:|
| MEASURED starting point | — | 17.44 GB |
| 64-byte re-align `lm_head.trellis` (§3.1) → `align_packed_view` early-returns | **−1,409,286,144** | **~16.03 GB** |
| offload `embed_tokens` | −0 (already mapped, MEASURED 0–0.36 GB resident) | ~16.03 GB |
| offload `lm_head.trellis` | −0 (already mapped; after the repack its 1.31 GiB is *clean reclaimable*, not anonymous) | ~16.03 GB |
| KV 16k → 8k (not an offload) | −0,629,452,800 | ~15.40 GB |

**16.03 GB is the honest best case for this checkpoint without re-quantising.**
`OFFLOAD-4BIT` §5 reached the same number independently.

### 2.4 What *would* reach 14.6 GB — and it already exists on disk

| config | arena | head | arena+head | + KV 8k | + scratch/runtime ≈1 GB | charged total |
|---|---:|---:|---:|---:|---:|---:|
| **4 bpw layers, 8 bpw head** (this ckpt) | 14,661,867,236 | 1,409,286,144 | 16,071,153,380 | +1,510,481,920 | +~1.0e9 | **~18.58 GB** |
| 4 bpw layers, 6 bpw head | 14,661,867,236 | 1,056,964,608 | 15,718,831,844 | +1,510,481,920 | +~1.0e9 | **~18.23 GB** |
| 4 bpw layers, 4 bpw head | 14,661,867,236 | 704,643,072 | 15,366,510,308 | +1,510,481,920 | +~1.0e9 | **~17.88 GB** |
| **3 bpw layers, 6 bpw head** | 11,001,026,276 | 1,056,964,608 | 12,057,990,884 | +1,510,481,920 | +~1.0e9 | **~14.57 GB ✓** |
| 3 bpw layers, 8 bpw head | 11,001,026,276 | 1,409,286,144 | 12,410,308,420 | +1,510,481,920 | +~1.0e9 | **~14.92 GB** |

The 3 bpw-layer arena is not a guess: `Gxma-repacked`'s header gives
`text − head − embed = 14,876,563,172 − 1,056,964,608 − 2,818,572,288 = **11,001,026,276**`,
which equals `410 × 0.75 × trellis + signs + norms` to the byte (`READ-FROM-HEADER`).

**So: 14.6 GB is reachable only by going to 3 bpw layers, and the configuration that reaches
it is `Gxma-repacked`, which is already on this machine.** Re-emitting the head at 6 bpw is
what closes the last 0.35 GB.

### 2.5 Real headroom on this box

| | bytes | label |
|---|---:|---|
| physical | 25,769,803,776 (24 GiB) | GIVEN / `sysctl hw.memsize` |
| `vm.user_wire_limit` | 19,585,050,869 (18.24 GiB) | MEASURED just now |
| `PhysMem` right now | `6864M used (2672M wired, 155M compressor), 17G unused` | MEASURED just now |
| swap | `3072M total, 1719M used` | MEASURED just now |

`17G unused` with **two other agents working** — that is not headroom, it is a snapshot.
The binding constraint is neither the box nor the wire limit but **what the other two agents
are holding**; `AGENTS.md` §10 records that two concurrent instances already cost 3.9 GiB of
swap and collapsed to 1.4 tok/s.

Effective budget, subtracting non-model overhead from the wire limit:
`19,585,050,869 − (process runtime ~0.2e9) = ~19.4 GB` wireable, against a `17.44 GB` model
plus `~1.0–1.4 GB` of cache → **~18.5 GB needed against ~17 GB available while others run.**
That is the measured 17.44 GB footprint, and it is why the box is unusable with the 4 bpw
checkpoint and two peers.

---

## 3. Deliverable 3 — repack options (assessed, **not performed**)

### 3.1 YES — re-align `lm_head.trellis` to remove 1,409,286,144 B

The existing repacker already does the whole job, and for this checkpoint **without touching a
single weight byte**:

* `repack_model.rs:196` — `let aligned_start = (cur_offset + 63) & !63;` — every
  `data_offsets` entry becomes **64-byte aligned**.
* `repack_model.rs:206-212` — the header JSON is space-padded until `(8 + header_len) % 64 == 0`,
  so `payload_start` is 64-byte aligned.
* ⇒ `absolute_offset = payload_start + data_offset` is 64-byte aligned ⇒ `delta % 4 == 0` ⇒
  `align_packed_view` early-returns at `model.rs:4890-4892`. **Zero copy, zero behaviour change.**
* `repack_model.rs:196`'s colmajor gate is `name.ends_with(".trellis") && shape.len() == 3 &&
  shape[2] == 48`. **Every 4 bpw trellis here has `shape[2] ∈ {64, 128}`**
  (`READ-FROM-HEADER`), so the transpose never fires and every byte is copied verbatim.

**Header length.** Shard 3 is `header_len = 58,577`, `payload_start = 58,585`. The minimum pad
that would suffice is **3 bytes** (`58,588` is divisible by 4). Take the repacker's 64
(`8 + header_len ≡ 0 mod 64`) — it costs ≤63 bytes and removes the question.
`MAX_HEADER_BYTES` is nowhere near the limit.

**Mandatory post-step.** `repack_model.rs:60-67` writes `"layout": "colmajor"` into the
destination `quantization_config.json`. **Nothing was transposed, so that key must be deleted.**
If it is left, `safetensors.rs:894` computes the *swapped* `expected_shape` for every non-`lm_head`
module and `validate_linear_geometry` rejects the checkpoint. (`lm_head` is explicitly exempted
from the swap, so the head would still validate and only the 410 layers would fail — a confusing
failure.) The source has **no `layout` key** (`READ-FROM-HEADER`: `qc.layout = undefined`).

**Recipe (not run).**

```bash
cargo run --release --bin repack_model -- \
    ~/.kaplumba/models/Gxma-4-bit ~/.kaplumba/models/Gxma-4-bit-aligned

# MANDATORY: remove the injected colmajor marker. No tensor was transposed.
grep -v '"layout"' ~/.kaplumba/models/Gxma-4-bit-aligned/quantization_config.json \
  > /tmp/qc && mv /tmp/qc ~/.kaplumba/models/Gxma-4-bit-aligned/quantization_config.json

# Header-only verification, no GPU, <1 s:
KAPLUMBA_TEST_MODEL=~/.kaplumba/models/Gxma-4-bit-aligned \
  ./target/release/deps/probe_offload_split-<hash> --exact probe_offload_header_ledger --nocapture
#   expect: "text tensors whose absolute file offset is NOT 4-byte aligned: 0"
```

**Precedent, already proven:** `Gxma-repacked` has **0** misaligned trellises out of 411
(`READ-FROM-HEADER`), so the repacker demonstrably does exactly this.

**Bonus, larger than the memory saving.** 291 of the 411 trellises (11,577,065,472 B) are
currently 4-byte **misaligned in the file**. In the arena that is invisible — every tensor is
bound at offset 0. But if layers were mapped, `ishizuki_gemv_pipeline` refuses
`packed_offset % 4 != 0` (`linear.rs:1516`) and the tensor silently falls to the 16-bit
`linear_gemv_exl3_k8_u_mcg` path, whose reads land a byte off — MEASURED to collapse greedy
margin 11.42 → 0.53 (`model.rs:4863-4870`). So the repack removes a **silent-correctness
landmine**, not just 1.31 GiB. (It does not make offloading the layers attractive; see §1.1.)

**Two ways this could be wrong.** `read_shard_header` or `validate_linear_geometry` could
reject 64-byte-interleaved offsets or the padded header, and the model would not load — a
2-minute header-only check (above) settles it, and it must be run before anyone invests in
this. Second: `model.safetensors.index.json` is copied verbatim; its `metadata.total_size` is a
sum of tensor byte lengths, not a file size, and re-alignment changes no tensor length, so
`safetensors.rs:493-497` still validates (`INFERRED`).

### 3.2 Droppable for text-only inference: **1,151,487,072 B — and it is already dropped**

* Vision tower + `embed_vision`: 356 tensors, 1,151,487,072 B. Excluded by
  `is_vision_tensor_name` before any allocation (§1.6). MEASURED 0 resident bytes.
* **Nothing else.** All 2,066 text tensors are read at load. `quantization_config.json`
  declares 833 `tensor_storage` modules = 411 linear modules + 421 norm modules + 1 embedding,
  i.e. it covers every text tensor (`READ-FROM-HEADER`). There is no orphan and no dead entry.
* The only *unnecessary* bytes left are the **18,503,396 B** of arena-side `.suh`/`.svh`/
  `.mcg`/norm copies, which are read once at load and then live only in their own Metal
  buffers. 0.018 GB — not worth a loader change.
* `tie_word_embeddings: false`, so `embed_tokens` and `lm_head` are genuinely distinct. Nothing
  to dedupe.

### 3.3 Re-emitting the head at a lower rate: **expressible, zero kernel work, and too small to matter**

**It is expressible.** Three independent checks:

* `safetensors.rs:893` — `let packed_width = (bits as usize * 256) / 16;` and
  `expected_shape = [input/16, output/16, packed_width]`. Any integer rate 1..8 is structurally
  accepted for `lm_head`. `safetensors.rs:1209-1230` already tests that `head_bits` 0 and 9 are
  refused, so 1..8 is the intended range.
* `linear.rs:1506-1517` — `ishizuki_gemv_pipeline` accepts any `Exl3Rate::Integer(1..=8)` at a
  4-byte-aligned offset.
* `linear.metal` defines `linear_gemv_ishizuki_k1` … `_k8` (lines 1598, 1710, 1822, 1934,
  2046, 2158, 2270, 2382). `_w2` two-word variants exist only for k1..k4, and k5..k8 fall
  through to the three-word form — which is exactly why 8 bpw runs today.

| head rate | `packed_width` | bytes | saved vs 8 bpw | tile stride |
|---:|---:|---:|---:|---|
| **8** (current) | 128 | 1,409,286,144 | — | 256 B |
| 6 | 96 | 1,056,964,608 | 352,321,536 | 192 B |
| 4 | 64 | 704,643,072 | 704,643,072 | 128 B |
| 3 | 48 | 528,982,304 | 880,304,640 | 96 B |

**On the `exl3.rs:740` hardcode.** `48` there is the **3-bpw tile stride**: a 16×16 tile is
256 weights, so 3 bpw → 96 B = 48 u16, 4 bpw → 64 u16, 8 bpw → 128 u16. It sits on the
*colmajor* path only, and the prior analysis is right that **only rate 3 can be colmajor-packed**:
the only colmajor kernels in the tree are `linear_gemv_ishizuki_k3_colmajor*` (24 variants),
plus two clearly-labelled experiments `linear_gemv_ishizuki_k{4,6}_colmajor_contig4_bench`.
And `safetensors.rs:894` never transposes `lm_head` even under `layout: colmajor`, so **a
colmajor head is not expressible at any rate today.** That is irrelevant here: row-major works
for all of 1..8. Generalising `reorder_tensor_to_colmajor` to `bits * 16` is a ~3-line change
and buys nothing here.

**Quality cost: unknown, and not measurable from this analysis.** There is **no head-rate
ablation anywhere in this repo.** `Gxma-repacked` (3 bpw layers + 6 bpw head) is a working
baseline, but the delta between it and a 4 bpw-layer/8 bpw-head build confounds the layer rate
with the head rate and cannot be attributed to the head. Anyone claiming a head-rate quality
number must run `--temperature 0` byte-identity against the 8 bpw head — `AGENTS.md` §8 permits
no quality loss, so this is a gate, not a judgement call. **Recorded as an open question, not a
finding.**

**Scale, so nobody over-invests here.** The head is **1.41 GB of 17.59 GB of text bytes (8.0 %)**.
8→6 bpw saves **0.33 GB**. The gap to 14.6 GB is **2.8 GB**. Lowering the head rate is a
rounding error next to the 13.64 GB layer trellis; only the layer rate moves the number.

---

## 4. Negative results, recorded

* **Offloading `embed_tokens` a second time saves nothing.** It is already mapped
  (`weights.rs:585-587`) and already MEASURED 0–357,416,960 B resident (`OFFLOAD-4BIT` §2).
* **Mapping `lm_head.trellis` saves nothing at the margin.** Already mapped. The 1.31 GiB
  appears in `phys_footprint` as clean page cache, but decode re-reads 100 % of it every token,
  so it returns immediately under any pressure.
* **Dropping the vision tower saves nothing at load.** Already dropped, by name, before
  allocation.
* **The "704 MB copy is 1.39× the tensor" framing is wrong.** 507,672,341 is an offset and
  704,643,072 is a u16 element count. The copy is exactly 1,409,286,144 B — a plain memcpy,
  no over-allocation.
* **`metal.rs`'s 18,046,374,518 is not this checkpoint's requirement** (§1.8).
* **`allocated_bytes()` under-reports anonymous memory by 1,409,286,144 B (head copy) plus
  2,097,152 B (`raw_b`, `raw_c`) on this configuration.** So the 18 GB budget is being enforced
  against 15.55 GB while the process commits 17.61 GB at 8k context.
* **The 3.9 GiB-of-swap result is not a tuning failure, it is the decision rule working.**
  Offloading the layer trellis converts a 273 GB/s streaming read into a disk read, per token.
  At 4 bpw that is 16.05 GB/token of layer+head weights against 273 GB/s of wire; at ~2.5 GB/s
  of NVMe that is 6.4 s/token against 0.059 s/token resident.

---

## 5. Commands used (all header-only, no GPU, no model load)

```bash
# Header ledger. Reads only the JSON at the front of each shard.
/tmp/offload-probe/hdr.mjs   ~/.kaplumba/models/Gxma-4-bit    # tensor list, shard header_lens
/tmp/offload-probe/perlayer.mjs                                 # per-projection trellis sizes
/tmp/offload-probe/sweep.mjs                                    # per-token sweep totals
/tmp/offload-probe/align.mjs                                    # 4-byte misalignment census
/tmp/offload-probe/align2.mjs ~/.kaplumba/models/Gxma-repacked  # misalignment after repack
/tmp/offload-probe/hdr2.mjs  ~/.kaplumba/models/Gxma-repacked  # 3 bpw cross-check

# Box state.
sysctl -n hw.memsize; sysctl vm.user_wire_limit
sysctl -n vm.swapusage; top -l 1 -n 0 | grep PhysMem
```

Nothing was written to `~/.kaplumba/models`; no shard was copied. No production code was
modified. No new test file was added — the existing
`crates/engine/tests/probe_offload_split.rs::probe_offload_header_ledger` covers the
header-only path and is the verification command for §3.1.

## 6. Open questions (must not be silently resolved)

1. **What is 18,046,374,518 in `metal.rs:7-9`?** Not derivable from any checkpoint on this
   machine. Either a stale `phys_footprint` or it belongs to `Gxma-bpw4-shim`, whose shards are
   dangling symlinks. Until resolved, do not treat 18.05 GB as "required resident".
2. **Head-rate quality cost is unmeasured.** No ablation exists. Gate: `--temperature 0`
   byte-identity against the 8 bpw head.
3. **Is `allocated_bytes()`'s 2.8 GB under-report load-bearing anywhere?** If a budget check
   reads it, the 18 GB limit is being enforced against the wrong number. Reported, not fixed.
4. **Does `Gxma-4-bit-aligned` load at all?** §3.1's repack is untested. The header-only check
   is 2 minutes and must be run before it is relied on.

---

## 7. One line

**Can offloading take this model from ~18 GB to 14.6 GB — no; the weight arena alone is
14.66 GB (13.64 GB of it layer trellis read in full every token, which is exactly the tensor
class that must not be offloaded), so the true minimum resident set for this checkpoint is
~16.18 GB anonymous at 8k context plus up to 1.41 GB of reclaimable `lm_head` page cache —
~16.03 GB after the one genuinely available saving, a 64-byte re-alignment of the checkpoint
that removes a 1,409,286,144 B memcpy — and 14.6 GB is reachable only at 3 bpw layers with a
6 bpw head, which is `Gxma-repacked`, already on disk.**