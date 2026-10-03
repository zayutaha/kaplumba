# omlx load-path comparison — why ours swaps and (allegedly) omlx's does not

**Task:** compare omlx's weight loader against ours, explain the 18,046,374,518 B
anonymous / 12 GB compressor / 5.4 GB swap outcome.
**Method:** read-only source comparison. omlx at `reference/omlx` @ `3f2d07e`; MLX +
mlx-lm read from the installed wheels (`mlx 0.29.3`, `mlx-lm 0.29.1`).
**GPU:** not used. No model run, no `scripts/gpuq`. One 2 MiB CPU-device `mx.load`
probe was run (see §2) because the decisive claim could not be settled by reading
alone.

Legend: **[M]** measured in our tree or by a probe I ran · **[R]** read from source ·
**[S]** speculated.

---

## 0. Headline: the premise is false

The question asks for "the one mechanical difference" and proposes
*"replace `read_at`-into-arena with mmap"*. **There is no such difference.** omlx
does not keep weights file-backed either.

**MEASURED [M].** MLX `mx.load` on safetensors memory-maps the file only to *read*
it, then copies the payload into MLX-owned memory and releases the mapping. The
loaded array is **anonymous, dirty** — byte-for-byte the same VM classification as
our arena.

| observation | value |
|---|---|
| `mx.load(tiny.safetensors)` returns, before `eval` | `get_active_memory() == 8` (lazy, nothing allocated) |
| after `mx.eval(w)` | `get_active_memory() == 262152` for a 262,216 B file → MLX allocated its own buffer |
| new regions created by `eval` (1 MiB × 2 arrays) | 8 × `Untagged … dirty=256K sm=0K prot=rw-/rwx SM=COW`, **no path** |
| mapping of the safetensors file, before and after | **none**, at any point |
| same via `nn.Linear.load_weights(list(mx.load(p).items()))` | identical: 6 new `Untagged` dirty anonymous regions, `tiny2 mapped? False` |

`Untagged`, no path, `SM=COW`, fully dirty = anonymous malloc'd memory. A
file-backed mapping would have shown a path column like the `libsafetensors*.dylib`
entries do.

Corroborating: `libmlx.dylib` imports `_mmap` (`nm -u lib/libmlx.dylib`), so mmap is
used — but only as a *read path* inside `load`, not as the final placement.

> **Consequence for our repo:** `spec/file-backed-weights/implementation_notes.md:48`
> — *"This is the same mechanism MLX exploits: file-backed Metal buffers for
> weights"* — is **wrong**, and T2/T4 of that spec rest on it. That spec is still
> entirely `Not started` (`spec/file-backed-weights/tasks.md:4,15,26,37,52,69`), so
> nothing was built on the false premise, but it should be closed or corrected
> rather than left to be implemented by the next agent.

---

## 1. What omlx actually does

### 1.1 The serving load path

omlx delegates to `mlx_lm`. `load_model` (`mlx_lm/utils.py:157`):

```
utils.py:197-199   weight_files = glob.glob(model_path / "model*.safetensors")
                   for wf in weight_files: weights.update(mx.load(wf))     # ALL shards, lazy
utils.py:205       model = model_class(model_args)
utils.py:210-211   if hasattr(model, "sanitize"): weights = model.sanitize(weights)
utils.py:213-245   nn.quantize(model, ...)   if config has quantization
utils.py:247       model.load_weights(list(weights.items()), strict=strict)
utils.py:249       if not lazy: mx.eval(model.parameters())
```

So the sequence is **read-into-lazy-arrays → swap them into the module → eval**.
Same shape as ours (read → anonymous arena), except the *copy* is deferred and done
by MLX's scheduler rather than by `pread` into a pre-sized arena.

`mlx` symbols exported: `mlx::core::load_safetensors(...)` (in `libmlx.dylib`).

**REPACK/QUANTIZE AT LOAD [R].** Yes, and it is a genuine load-time transformation:
`nn.quantize` (`utils.py:213-224`) and `model.sanitize` (`utils.py:210`). For the
text models I checked, `sanitize` is **dict surgery, not a copy** —
`mlx_lm/models/qwen3.py:183-186` only pops `lm_head.weight`;
`mlx_lm/models/gemma3.py:49-56` only restructures the tree via
`tree_unflatten`/`tree_flatten`. So the resident set is 1× the checkpoint, and the
load transient is ~1× more in *reclaimable file cache*.

### 1.2 omlx's own bounds on the load transient

`omlx/cluster/memory_guard.py:56-76`:

```
_LOAD_PEAK_FRACTION = 0.30      # memory_guard.py:78
_LOAD_PEAK_FLOOR_BYTES = 2 GiB  # memory_guard.py:85
```
> "What a load costs beyond what it leaves resident. `planned_weight_bytes` is the
> resident size, so charging only that charges for the wrong instant. Getting there
> costs more: MLX-LM assembles the whole state dict before `load_weights` and
> `sanitize` rewrites it … quantised weights need dequantisation scratch on the way
> in, and **every transient MLX frees stays in its buffer pool until it is returned
> to the OS.**" — `memory_guard.py:59-70`

That is omlx stating, in its own source, that the load costs **more than the
resident set** and must be budgeted for. We budget nothing.

### 1.3 Progressive materialisation — the real transient fix [R]

`omlx/cluster/progressive_loading.py:28-70`, `materialize_parameters_progressively`:

```
progressive_loading.py:56   _eval_values(mx_module, fixed)
progressive_loading.py:57   mx_module.clear_cache()
progressive_loading.py:58   for loaded, index in enumerate(ordered, start=1):
progressive_loading.py:59       _eval_values(mx_module, layers[index])
progressive_loading.py:60       mx_module.clear_cache()
```

plus `gc.collect()` at `:222` and `:241`, and `:57`/`:60` per layer. It replaces
`server_module.sharded_load` for **every** load (`install_progressive_loader`,
`:258-274`, installed at `omlx/cluster/inference_worker.py:1379`).

**This is the genuine mechanical difference on the load path:** omlx materialises
**one transformer layer at a time and returns the allocator's freed pool to the OS
between layers**. We allocate every arena span up front — spans are fixed before
any read runs (`crates/engine/src/weights.rs:588-592`, and the comment at
`weights.rs:903-908`: *"Destination spans are fixed before this runs"*) — and then
`par_iter` every shard's `copy_tensor` at once (`weights.rs:646+`). Our transient is
dirty anonymous from the first byte and never gives anything back.

### 1.4 Memory/cache limit knobs [R]

| knob | where | value | effect |
|---|---|---|---|
| `mx.set_cache_limit` | `omlx/server.py:8335`, `omlx/cli.py:442` | `mx.device_info()["memory_size"]` (whole RAM) | pool freed MLX buffers up to RAM. **A correctness fix, not a tuning knob**: `cli.py:431-437` — without it `allocator::free()` releases a Metal buffer the GPU is still using → *"kernel panics on M4"* (issue #300). AGENTS.md §8 forbids us retaining a pool; omlx retains one **and bounds it to RAM**. |
| `mx.set_memory_limit` | not called anywhere in omlx | MLX default = 1.5 × max recommended working set | — |
| `mx.set_wired_limit` | `omlx/cluster/inference_worker.py:1138-1166`, `omlx/cluster/node_role.py:227-256` | budget-derived, capped by `iogpu.wired_limit_mb` | pins the model so it is *not* paged out |
| `set_wired_limit(0)` | `omlx/cluster/inference_worker.py:546-561` | 0 | **unwires** a model so the OS reclaims it — residency treated as releasable |
| `clear_cache` | `progressive_loading.py:57,60,220,223,239,242` | — | return the pool to the OS between layers |

### 1.5 omlx never holds file cache + an anonymous copy as *unmanaged* overcommit

MLX's laziness is what makes the transient survivable [R + M]: the payload sits in
**file-backed clean page cache** (reclaimable by *purge*, never by swap) while the
anonymous copy grows underneath it. Our `pread` path has no such phase — the arena
*is* dirty anonymous from the first byte, so any overcommit must go through
compress (no gain on bit-packed weights) and then swap. Combined with §1.3 this is
the honest second-place answer to "what differs".

---

## 2. The probe (reproducible)

`/private/var/folders/tp/wz3877_s1z756p1kcsz3_d4h0000gn/T/opencode/omlxprobe/p3.py` —
2 MiB of synthetic safetensors, `mx.set_default_device(mx.cpu)`, no GPU op, no
model, `vmmap` region diff before/after. Trivial memory traffic (a few MiB, one
process); it cannot perturb a decode benchmark, but flagging it for the record since
another agent held the GPU.

---

## 3. Does omlx mmap for reads? **No — and it rejected mmap for our exact reason**

This is the most useful thing in omlx's tree for us, because it is an *independent
measurement of the objection already written into our code*.

`docs/MoE_Expert_Offload.md:158-169` — synthetic oQ3e MoE, cold SSD reads, M5 Max:

| | mmap gather (before) | positional reads |
|---|---:|---:|
| decode, one token, per MoE layer | 362 ms | **9.1 ms** |
| expert fetch throughput | 0.23 GB/s | **9.3 GB/s** |
| sorted prefill, 256 tokens | 33 token-layers/s | **568** token-layers/s |

`docs/MoE_Expert_Offload.md:286-292`:
> "a miss is a 20 MiB expert … The same 20% configuration fetching through the
> common store's memmap path measured **646 s to first token and 0.09 tok/s**,
> because a slab faulted in 16 KiB pages on the compute thread reads at 0.4 GB/s;
> that is why this adapter reads misses through the store's **positional reads** on
> the shared reader pool"

**omlx made the same `read_at`-over-mmap decision, with numbers.** Our
`weights.rs:976-981` says mmap costs *"roughly a quarter of the drive's sequential
rate"*. omlx's numbers are **1/20th to 1/40th** (0.23 GB/s, 0.4 GB/s), and
critically they are measured **on the compute thread, inside the decode step**.

### 3.1 "Load only, or every decode step?"

**Both — and the decode-side payment is the fatal one.** Our doc comment justifies
`read_at` on *load* throughput, which is the weaker of the two arguments. The
stronger one is the per-step fault rate:

- decode re-reads the **entire** weight set every token — 18.89 GB/token at 4 bpw
  **[M: `spec/memory-swap-cliff/implementation_notes.md:318-322`]**
- at 22.1 GB/s of NVMe, 1% miss = **8.6 ms/token** against a ~70 ms step (12%);
  100% miss = 855 ms/token **[M, same lines]**
- therefore **the offload budget for layer weights is 0 bytes per token**
  **[M, same lines]**
- and the outcome has already been measured here: `KAPLUMBA_ZERO_COPY_WEIGHTS`
  (which maps *all* weights from the file, i.e. precisely the mechanism the
  question proposes) recorded **3.9 GiB of swap at ~1.4 tok/s**
  **[M: `crates/engine/src/weights.rs:510`, `:573`, `:1263`;
  `spec/engine-performance/implementation_notes.md:1201-1205`;
  `spec/memory-swap-cliff/implementation_notes.md:288-289`]**

So: **the load-time fault cost is acceptable (one-off). The per-decode-step fault
cost is fatal.** Both are paid; only one is tolerable. omlx agrees.

---

## 4. So what *is* omlx's mechanism that matters for swap?

**It refuses the load.** Not placement — admission control.

`omlx/process_memory_enforcer.py`, evaluated **on this exact box** (MEASURED sysctls):

```
hw.memsize            = 25,769,803,776 B = 24.000 GiB      [M]
iogpu.wired_limit_mb  = 23000  → 22.461 GiB                [M]
vm.user_wire_limit    = 19,585,050,869 B = 18.238 GiB      [M]
```

```
_SMALL_SYSTEM_THRESHOLD = 24 * 1024**3                     [R] process_memory_enforcer.py:58
_SMALL_SYSTEM_RESERVE   = 4 GiB                            [R] :57
_STATIC_RESERVE_LARGE   = {"safe":8G,"balanced":6G,"aggressive":4G,"custom":2G}  [R] :63-68
```

`system_bytes < 24 GiB` is **False** (they are equal), so the **6 GiB balanced
reserve applies**, not the small-system 4 GiB one:

```
static ceiling  = 24.000 − 6.000                = 18.000 GiB   [R, _get_static_ceiling :564-575]
metal cap       = 23000 MiB                      = 22.461 GiB   [R, get_effective_metal_cap_bytes :174-185]
hard limit      = min(static, dynamic, metal_cap)= 18.000 GiB   [R, _get_hard_limit_bytes :622-643]
soft target     = 0.90 × hard limit              = 16.200 GiB   [R, :750-753 × _SOFT_THRESHOLD_BY_TIER :82-87]
abort limit     = min(static, metal_cap)         = 18.000 GiB   [R, _get_abort_limit_bytes :754-775]
```

Enforcement [R]:
- `get_admission_ceiling()` `process_memory_enforcer.py:688-709`
- `get_admission_soft_target()` `:733-748` — idle LRU models are evicted **before**
  weights start allocating (`engine_pool.py:726-732`, `:2423`)
- `engine_pool.py:1981-1995` compares `projected = current + admission_size` to the
  ceiling; `:2061` *"no chance of fitting"*; `:2082-2087` raises
  `InsufficientMemoryError`
- `LoadMemoryWatchdog.run_once` `cluster/memory_guard.py:611-632` samples every
  0.5 s **during** the load and raises `InsufficientMemoryError` —
  *"stopping this rank before macOS stops the Mac"*
- `_REMEDY` `memory_guard.py:94-99`
- **omlx has no 24 GiB dense-4-bit evidence.** Every measurement in its docs is a
  128 GB or 512 GB Mac (M5 Max 128 GB, `MoE_Expert_Offload.md:173,277`;
  "488 GiB stable on a 512 GiB box", `process_memory_enforcer.py:193`). Its only
  small-box accommodation is *relaxing* the reserve so small Macs load at all —
  `_SMALL_SYSTEM_RESERVE` comment, `process_memory_enforcer.py:55-56`: *"Small Macs
  cannot afford a tier-scaled cut and still load any useful model"* — which is the
  admission ceiling being loosened to permit loads, not swap being avoided. On our
  **exactly**-24 GiB box that relaxation does not even apply (`<` is false).

### 4.1 The arithmetic, honestly

```
weights (4 bpw / 8 bpw head)   18,046,374,518 B = 16.810 GiB   [M, metal.rs:10-13]
KV @512 ctx                         881,029,120 B =  0.821 GiB   [M, memory-swap-cliff:147,157]
scratch (1024-row prefill peak)     547,937,040 B =  0.510 GiB   [M, memory-swap-cliff:148,158]
                                  --------------------------
steady-state anonymous             19,475,340,678 B = 18.141 GiB
KV @16k ctx                       2,245,000,000 B ≈  2.094 GiB   [M, memory-swap-cliff:154-160]
                                  --------------------------
steady-state anonymous @16k                            = 19.414 GiB

omlx hard ceiling                                        = 18.000 GiB
omlx soft target                                          = 16.200 GiB
```

- 16.810 GiB of weights is **above** the soft target (16.200) → omlx evicts other
  models first.
- 16.810 GiB is **below** the ceiling (18.000) → **omlx would admit the load**, with
  1.193 GiB (6.6%) of margin on weights alone.
- But steady-state **weights + KV + scratch = 18.141 GiB already exceeds the 18.000
  GiB ceiling by 0.141 GiB**, and @16k by 1.414 GiB — *before any load transient*.
- Add omlx's own load bound, 0.30 × 16.810 = 5.043 GiB → peak ≈ **21.85 GiB**,
  3.85 GiB over ceiling → `LoadMemoryWatchdog` aborts; single-node, the prefill
  guard starts aborting requests.

**Verdict: omlx would not sustainably run this model on this box either.** It would
admit the load and then trip its own guards. Its source does not substantiate the
premise that omlx runs an 18 GB dense model on a 24 GiB M4 Pro without swap —
**every measurement in omlx's docs is on a 128 GB or 512 GB Mac** (M5 Max 128 GB,
`MoE_Expert_Offload.md:173,277`; "488 GiB stable on a 512 GiB box",
`process_memory_enforcer.py:193`).

---

## 5. Would file-backed placement make this model fit? — "Not enough."

**It buys 0 bytes of capacity.** It converts 16.81 GiB of *unreclaimable* into
16.81 GiB of *reclaimable*. For the layer weights that is a **liability, not
headroom**, because decode needs all of it every token and reclaimable means the OS
is free to drop it:

- 24.000 GiB total − ~2.345 GiB wired **[M, `memory-swap-cliff:258`]** − ~4 GiB
  desktop → ~17.7 GiB of supply **[M, `memory-swap-cliff:182-192`: supply is
  11.5–17.0 GiB depending on the desktop]**
- 18.05 GB of weights = 16.81 GiB, which *fits in paper supply only if the page cache
  is 100% resident*, which macOS will not guarantee
- the measured failure when it is not: **3.9 GiB swap, 1.4 tok/s** **[M]**

File-backing converts a **predictable** swap spiral into an **unpredictable**
latency collapse. That is strictly worse to operate, even though the pressure
indicator goes green — which is precisely why `memory-swap-cliff:224-226` concluded:

> "**this is why the fix is admission control, not a pinning policy.** Pinning moves
> bytes from the *reclaimable* column to the *unreclaimable* column. It improves
> latency and it does not change the sum."

### 5.1 Where file-backing *does* legitimately pay — and we already took it

The win is real but confined to tensors whose **per-token working set ≪ their size**:

| tensor | size | per-token reads | placement | status |
|---|---|---|---|---|
| `embed_tokens` | 2.625 GiB | ~10 KiB (one row) | file-backed, cold, free | **done** (`weights.rs:584`) |
| `lm_head.trellis` | 1.312 GiB @4 bpw | full re-read | file-backed-clean, stays hot, **returns budget when the OS needs it** | **done** (`weights.rs:585`) |
| layer trellis | 10.2–13.6 GiB | **entire set, every token** | must be resident | arena, correctly |

`lm_head`'s measured behaviour: 2500 tokens × ~2.6 TB of re-reads → **0 swaps**,
0 MiB system swap across 15 samples over 288 s
**[M: `spec/resident-memory-12gib/implementation_notes.md:162-165`, cited at
`weights.rs:562-566`]**.

**3.94 GiB is the entire available file-backing win on this checkpoint, and it has
already been taken.**

---

## 6. Answers to the five specific questions

1. **mmap or copy?** Copy. `mlx_lm/utils.py:197-199,247,249` → `mx.load` → eval
   materialises into MLX-owned memory. mmap is an internal read path only
   (`_mmap` imported by `libmlx.dylib`). **MEASURED: anonymous dirty, mapping
   released** (§0).
2. **Where do the bytes live?** Anonymous, dirty, `Untagged SM=COW`, no path —
   **identical to our arena** (§0). omlx's only file-backed weights are *deliberate*
   offload targets (`omlx/patches/mlx_vlm_qwen4_exp_compat/residency.py:23-24,83-84`,
   `patches/deepseek_v41/moe_offload.py`) read on demand, not the dense resident set.
3. **Memory/cache knobs?** `set_cache_limit(RAM)` `server.py:8335` / `cli.py:442`
   (a *correctness* requirement, issue #300); `set_wired_limit(budget)`
   `cluster/inference_worker.py:1138-1166`; `set_wired_limit(0)` to unwire
   `:546-561`; `clear_cache()` between layers `progressive_loading.py:57,60`.
   `set_memory_limit` is never called (MLX default 1.5 × recommended working set).
4. **Load transient?** Holds *both* file cache and an anonymous copy — but the file
   cache half is **reclaimable clean**, so the overcommit is absorbed by purge, not
   swap. No `mlock`, no prefault. Tightened by per-layer eval + `clear_cache` +
   `gc.collect` (§1.3) and budgeted at +30%/+2 GiB (§1.2). It does repack at load
   (`nn.quantize` `utils.py:213`, `sanitize` `:210`) but for text models `sanitize`
   is non-copying, so the resident set is 1×.
5. **Convert/repack at load?** Yes — `nn.quantize` before `load_weights`
   (`utils.py:213-245`); scratch is MLX-allocator memory, returned via
   `clear_cache`. omlx's own comment lists dequant scratch and MoE per-shard staging
   as reasons the peak exceeds resident size (`cluster/memory_guard.py:62-70`).

---

## 7. Recommendations

1. **Close or correct `spec/file-backed-weights`.** Its premise
   (`implementation_notes.md:48`) is disproved by measurement; T2/T4 (whole-arena
   file-backing + `mlock`) are measured-harmful: 3.9 GiB swap @ 1.4 tok/s. Its
   `mlock` plan also **fails arithmetically** at 4 bpw — 20,041,520,424 B =
   18.665 GiB > the 18.238 GiB `vm.user_wire_limit`, so `mlock` would `ENOMEM`
   (`memory-swap-cliff:273-290`). Recommend a `SUPERSEDED` header pointing here.
2. **Port omlx's admission control**, not its placement. Concretely: check
   `resident_limit_bytes()` (currently 18e9, `metal.rs:4`) against the **box**, not
   against `arena.capacity_bytes`. `memory-swap-cliff:103-119` establishes that the
   current check is against arena capacity and therefore admits a 4 bpw load the
   hardware cannot hold — 3.34 GB of margin *on paper* while the machine thrashes.
   omlx's equivalent is `get_admission_ceiling()` + `LoadMemoryWatchdog` +
   `InsufficientMemoryError`.
3. **Port the progressive per-layer materialisation** (`progressive_loading.py:28-70`)
   for the *load transient*: our spans are fixed up front (`weights.rs:588-592`) so
   the arena never returns anything to the OS mid-load. Cheaper than admission
   control and it shrinks peak by whatever the transient is.
4. **Strengthen the `weights.rs:976-981` comment.** It currently justifies `read_at`
   on *load* throughput. The decisive reason is the **per-decode-step** fault rate
   (0.4 GB/s on the compute thread, omlx's own number), and the ¼-sequential-rate
   figure is optimistic by 5–10×. Cite `MoE_Expert_Offload.md:162-165,286-292`.
5. **Do not adopt mmap for layer weights.** The experiment has already been run
   (`KAPLUMBA_ZERO_COPY_WEIGHTS`) and the result is recorded. The only remaining
   candidate is more of `embed_tokens` / `lm_head`-shaped tensors, and both are
   already file-backed.
6. **The capacity answer is not a loading change.** 18.14 GiB of steady-state
   anonymous demand against ~14.5–17 GiB of measured supply on a 24 GiB box with a
   4 GiB desktop is a **capacity** problem. Shrinking the model (3 bpw →
   `memory-swap-cliff:146-151`: 12.43 GB anon, fits with 2.9 GiB margin) or
   spending less RAM on the desktop are the only levers that change the sum.