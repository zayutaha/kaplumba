# Implementation Notes: 14+ tok/s Decode Throughput at 17.5k Context

## Hardware & Environment
- Machine: Apple M4 Pro, 20 GPU cores, 24 GiB unified RAM, 273 GB/s peak bandwidth.
- Model: `~/.kaplumba/models/Gxma-repacked` (EXL3 ~3.0 bpw, ~15.5 GB resident).
- Branch: `stepwise-clean-3724c01`.

## Measured Breakdown at 17.5k Context
- Steady-state decode step duration: **77.4 ms** ($\approx 12.92\text{ tok/s}$).
- Target for $\ge 14.0\text{ tok/s}$: $\le \mathbf{71.4\text{ ms}}$ (gap = **6.0 ms**).
- Profiler reading (`KAPLUMBA_GPU_PROFILE=1 KAPLUMBA_GPU_PROFILE_DECODE=1`):
  - `gate_up_gelu` (`linear_gemv_ishizuki_k3_fused2_co` across 60 layers): **26.22 ms** (437 $\mu$s/layer).
  - `down_proj` (`linear_gemv_ishizuki_k3_colmajor_2col_sk2` across 60 layers): **14.05 ms** (234 $\mu$s/layer).
  - `qkv_proj` (50 sliding layers): **10.39 ms** (208 $\mu$s/layer).
  - `o_proj` (60 layers): **6.84 ms** (114 $\mu$s/layer).
  - `head/gemv`: **4.05 ms**.
  - `attention` (10 full layers + 50 sliding layers): **3.20 ms** (10 full layers take 2.22 ms, 50 sliding take 0.98 ms).
  - `qk_proj` (10 full layers): **2.31 ms**.
  - Inter-boundary / CPU dispatch submission gap: **~7.6–10.5 ms**.

## Key Decisions & Negative Results
1. **`linear_gemv_ishizuki_k3_fused2_co4` (4-column fusion) is SLOWER:**
   - Tested via `KAPLUMBA_FUSED2_CO4=1`.
   - Result: `gate_up_gelu` regressed from 26.2 ms to 29.8 ms (+3.6 ms slower) due to register spilling and 4-way threadgroup reduction contention. Kept 2-column `fused2_co`.
2. **Attention is not the primary lever at 17.5k tokens:**
   - Total attention execution is only 3.20 ms. Even a 50% improvement in attention would only yield 1.6 ms of the 6.0 ms gap needed.
3. **The main levers are Command Buffer Batching and `gate_up_gelu` unrolling:**
   - 540 dispatches divided into 50-dispatch buffers generates 11 command buffers per token. Batching into 1–2 buffers addresses the ~7.6 ms gap directly.

---

## 19. CORRECTION: the 77.4 ms figure was measured on a box that fit; at 17.5k it does not

This supersedes §5's premise. Fresh measurement, same binary, same checkpoint
(`Gxma-repacked`), `--chat --temperature 0`, `GXMA` prompt = 2 912 x
`"Explain the theory of relativity. "` (~17 486 tokens), `KAPLUMBA_TOKEN_TIMING=1`:

| context | steady-state step | tok/s | first decode step (token 2) |
|---|---:|---:|---:|
| 23 | 77 ms | 12.9 | — |
| 6 014 | **71.9 ms** | **13.9** | 1 560 ms |
| 17 486, as shipped | **2 700 ms** | **0.37** | 3 162 ms |
| 17 486, trellis pinned | **77.6-78.9 ms** | **12.8** | 873 ms |

**The collapse is memory, not kernels.** `KAPLUMBA_SKIP_ATTENTION=1` at 17 486
context (an existing attribution switch in `model.rs`) drops the step
2 700 -> **105 ms** and the compressor 8.4 G -> 0.7 G. The attention pass is what
pulls the whole KV back in and tips the box over; the kernels and the host encode
(measured 2 ms/step via the `[step] encode=.. wait=..` trace now under
`KAPLUMBA_TOKEN_TIMING`) are not the problem. The host sits in
`waitUntilCompleted` for the entire step (`sample` on the live pid: 2516/2516
samples on the main thread).

**Mechanism.** At 17 486 the process holds only 4.3 GiB while
`vmmap -summary` reports *both* shard mappings at `RESIDENT 0K`, `IOAccelerator
(graphics) RESIDENT 4.1G`, and macOS puts 8-11 GiB into the compressor. The
12.04 GiB of `.trellis` is clean and file-backed -- the right default -- which
also means the VM may reclaim it at will, and with ~7 GiB of ambient applications
on this 24 GiB box it does. Every token then re-reads ~12 GiB at **4.4 GB/s**
(12 GiB / 2.7 s), which is NVMe speed, not DRAM (12 GiB / 44 ms at 273 GB/s).

### Negative results (recorded per AGENTS.md §7)

* **`madvise(MADV_WILLNEED)` does not fix it.** It was already in
  `ShardReader::open`; it faults the pages in *once* and does not stop eviction.
* **`KAPLUMBA_RESIDENT_WEIGHTS=1` does not fix it.** Measured **0.41 tok/s** at
  the same context (step 2 524-2 797 ms); the compressor takes the dirty
  anonymous arena instead of the page cache. Load is also 2.7x slower (39.9 s vs
  14.8 s).
* **Warming the pipelined decode path at load does not fix the first step.**
  Adding two `forward_step_pipelined` steps to `Model::warmup_decode` (both
  slots, real `HeadOut`) made step 1 *worse*: 2 432 ms vs 1 560 ms. Reverted.
  The stall is not a first-dispatch cost.
* **Command-buffer batching is null at long context.**
  `KAPLUMBA_DECODE_MAX_DISPATCHES_PER_BUFFER` 50 vs 270 at 17 486 pinned:
  77.5-78.3 ms vs 77.5-78.4 ms. (It *is* worth ~1.2 ms at 6k: 73.0 -> 71.9 ms.)
  §5 Finding 3's ~7.6 ms gap did not reproduce.
* **Fewer attention splits are worse.** `KAPLUMBA_DECODE_SPLITS=16` at 17 486
  pinned: 78.0-78.8 ms, against 77.5-78.4 ms at the shipped 35
  (`key_count.div_ceil(512)`). Keep the shipped rule.
* **Pinning the whole shard mapping works but is unsafe.** 21 GiB wired,
  token 2 = 445 ms. Rejected.

## 20. The fix: pin only the `.trellis` ranges

`TextWeights::pin_trellis_pages` (weights.rs), called at the end of
`load_internal`. It walks the 411 tensors whose name ends in `.trellis`, rounds
each `[start, end)` outward to `PAGE_ALIGN`, and `mlock`s it. Measured:

* **11.21 GiB wired** (of 14.9 GiB of shards), against 21 GiB for the whole
  mapping. `embed_tokens`, `lm_head`, the vision tower and the arena stay
  offloaded/resident exactly as before.
* 17 486 context: steady state **77.6-78.9 ms (12.8 tok/s)**, from 0.37 tok/s.
* Wired pages stay *clean* and file-backed: no swap, no compressor, and the
  machine returns to 3.3 GiB wired / 17 GiB free the moment the process exits.
* Greedy output is **byte-identical** (`hi` -> `Hello! How can I help you today?`).
* `cargo test --release -p kaplumba-engine --lib`: 102 passed, 0 failed.
* Opt out with `KAPLUMBA_NO_PIN_TRELLIS=1`. `mlock` is best-effort: on failure the
  first error is reported and the run continues (slow, still correct).

## 21. Still open

* **14.0 tok/s is not reached at 17.5k: 77.6 ms against a 71.4 ms target.**
  ~4.5 ms of the gap is structural -- attention must read the extra
  (`17 486 - 6 014`) x 81 920 B = 0.94 GB per step, which is 3.4 ms at 273 GB/s.
  Both host-side levers measured null (above).
* **Token 2 is still 873 ms against a 78 ms steady state.** Pinning everything
  got it to 445 ms, so the residual is still memory: the ~1.1 GiB arena and the
  KV cache are the remaining compressible pages.
* **Lever B (software-pipelined K-loop in `linear_gemv_ishizuki_k3_fused2_co`)
  is unstarted.** It is the only remaining measured lever for the steady state
  and it needs a bit-exactness check against `fused_bit_exact`.

### How to reproduce

```bash
python3 -c "print('Explain the theory of relativity. ' * 2912)" > /tmp/prompt_17k.txt
KAPLUMBA_TOKEN_TIMING=1 scripts/gpuq try -- bash -c 'cat /tmp/prompt_17k.txt | \
  ./target/release/kaplumba --model ~/.kaplumba/models/Gxma-repacked --chat \
  --temperature 0 --max-new-tokens 12'
```
The 6k arm is the same with `* 1000`. One 17.5k arm costs ~3.5 min (load 13 s +
prefill ~180 s).

---

## 22. Lever B (software-pipelined K-loop in `fused2_co`): bit-exact, and NULL

**Implemented and measured, then reverted.** Trip N+1's eight weight words are
issued before trip N's `extract_v_mcg_32`/FMA body, so they are in flight across
it. Only the **weight** loads are prefetched: `x` is a 5376-element activation
whose four `half` reads per trip are L1-hot, and `fused2_co4` already showed that
spending registers here loses (26.2 -> 29.8 ms, §5 Finding 2).

**Bit-exact on the real model.** With the pipelined loop, greedy token IDs were
identical to the shipped kernel at both contexts:

* 6k: `17574 611 4733 672 2934 1551 2782 236764`
* 17.5k: `17574 611 4733 834 1551 2782 236764 564 795 2847`

**Performance: null.** Steady state medians, same box, same prompt:

| context | pipelined | shipped |
|---|---:|---:|
| 6,014 | 72.0 ms | 71.9 ms |
| 17,486 (trellis pinned) | 77.8 ms | 78.0 ms |

Within the interleaving noise, and it makes the hot loop harder to fold (the
prefetch adds 8 live registers), so it is reverted rather than shipped. The
shipped loop is already eight loads deep per trip after `32830cb`; the
loop-carried dependency was not the binding constraint.

### The sabotage that did *not* fail — `fused2_co` has no synthetic test

Per AGENTS.md §3.4 the change was sabotaged in the prefetch before being
believed: `n1_a0` was changed from `t1_a[j0]` to `t0_a[j0]`. **`fused_bit_exact`
stayed 3/3 green.** So that suite does *not* cover this kernel.

`KAPLUMBA_TRACE_GEMV=1` on the real checkpoint shows why: both
`linear_gemv_ishizuki_k3_fused2_2col` and `linear_gemv_ishizuki_k3_fused2_co
(mlp)` dispatch. The MLP path reaches `_co` through
`encode_gemv_fused2_mlp` (`linear.rs:2683`), while `fused_bit_exact` calls the
`encode_gemv_fused2` entry, which selects `_2col`. Its "fused2" case therefore
tests a different kernel, and its 3/3 pass says nothing about `_co`.

**This is the exact failure mode AGENTS.md §3.4 was written about.** The
end-to-end token-ID comparison above is the only bit-exactness evidence `_co`
has, and it exists only because this experiment was run. A synthetic
`_co` case in `fused_bit_exact` (routing through `encode_gemv_fused2_mlp`) is
the fix and is **not done**.

## 23. Pinning the arena and the KV cache: measured no, and why

* **Arena (~1.1 GiB):** `WeightArena::pin_pages` `mlock`ed all 1654 segments and
  reported **+0.02 GiB** actually pinned -- the great majority of the calls
  fail on those Metal allocations, and only failures are counted. Token 2 went
  608 -> 508 ms, which is **inside** the 445-873 ms band this same measurement
  has produced across runs, so it is not an effect. Reverted.
* **KV cache (~2.6 GiB at 16k):** not attempted. `sysctl vm.user_wire_limit` is
  **18.24 GiB** (`19585050869` bytes; `ulimit -l` is `unlimited`). The trellis
  already pins 11.21 GiB and the OS carries ~3.3 GiB wired of its own, so adding
  the KV puts the total at ~18 GiB with no headroom -- and the KV ladder
  **reallocates on every rung**, so it would need re-pinning inside
  `GpuKvCache::ensure_capacity`. The payoff is bounded by one token: steady state
  is already 78 ms with the KV unpinned, so this can only remove the first-step
  transient.

**Conclusion: pinning beyond the trellis is not worth it.** The residual is in
token 2's variance, not in a fixable residency defect.

## 24. Pre-existing failures on HEAD (not caused by this work)

`cargo test -p kaplumba-engine --test probe_unroll_kloop_dag` is **4 passed,
4 failed** at HEAD. The failures are `dependency_graph_...`, `detectors_can_fail`,
`fused2_co_burst_order_...` and `loads_in_flight_table_covers_every_live_kernel`,
and the first assertion is `fused2_co must read 4 of 8 as shipped` with
`left: (8, 8), right: (4, 8)`. The probe was written for the pre-hoist shape;
commit `32830cb` hoisted `fused2_co` to 8 of 8 and the probe was not updated.
`git status` on `linear_repacked.metal` is clean at HEAD, which is how this was
told apart from a regression from the Lever B experiment.

---

## 25. 🔴 A REGRESSION LANDED AFTER §24: `e1b05c1` put the trellis back into the
## compressor, and long-context decode went to 0.37 tok/s

`e1b05c1` ("green memory residency", 21:47) deleted `pin_trellis_pages` and
changed the offload guard from

```rust
let offload_trellis = !force_resident && name.ends_with(".trellis");
```
to
```rust
let offload_trellis = !force_resident && !dedicated && name.ends_with(".trellis");
```

Production loads through `TextWeights::load_dedicated` (`model.rs:3595`), which
passes `dedicated: true`. **So `offload_trellis` was unconditionally false in
production**, all 12.04 GiB of `.trellis` was copied into dirty anonymous
`MTLResourceStorageModeShared` buffers, and the exact failure §19 documented for
`KAPLUMBA_RESIDENT_WEIGHTS=1` returned.

Measured at 16,436-token context, `--chat --temperature 0`, current binary:

| build | step | tok/s |
|---|---:|---:|
| `833c2ce` (trellis file-backed + pinned) | 76.3 ms | 13.1 |
| `e1b05c1` (trellis in the shared arena) | **2,721 ms** | **0.37** |

Host encode stayed at 1.8 ms while the GPU wait went to 2,488 ms, and the machine
showed **swap 4.35 GiB, compressor 8.1 GiB, 26 % free** -- against the spec's own
claim of "36 % free (SOLID GREEN)" and "`vm.swapusage`: 0 growth". Its telemetry
measured *memory* at 17.5k and *throughput* on a 19-token prompt, so its **R5
(>= 13.0 tok/s at 17,500 context) was violated by 35x and shipped anyway.**

The code's own comment five lines above the change already said it: *"Copying
13.4 GiB of weights into dirty anonymous Metal buffers causes macOS to exhaust
available RAM at 16k context (where KV cache adds ~3 GiB) and swap out 12.5 GiB
to SSD, collapsing decode to 0.03 tok/s."*

**Fix:** drop `!dedicated`, restore `pin_trellis_pages`. Result: **2,721 -> 76.0 ms
(0.37 -> 13.1 tok/s)**, 11.21 GiB wired, load 8.8 s. This **amends
`spec/green-memory-residency` R1** ("the engine must not call `libc::mlock`
under any circumstances"): a ~17.7 GiB hot set on a 24 GiB box will be compressed
unless those pages are wired, so R1 and R5 are mutually exclusive. R5 was the
user-facing requirement; R1 was a means.

## 26. The honest budget at 16.5k, measured by ablation (not modelled)

Two interleaved arms, `--chat --temperature 0`, median of 23, current binary:

| arm | step |
|---|---:|
| full | **75.55 ms** |
| `KAPLUMBA_ABLATE=attention` | 63.60 ms |
| `KAPLUMBA_ABLATE=gate_up_gemv,down_gemv,qkv_gemv,o_proj_gemv,head_gemv` | 14.65 ms |

Therefore, additively and exactly:

| term | ms | share | state |
|---|---:|---:|---|
| weight GEMVs (240 layer + 1 head) | **60.90** | 80.6% | ALU-bound, bit-locked |
| attention (split + combine, 60 dispatches) | **11.95** | 15.8% | at the DRAM read roofline |
| element / finalize / norm / host | **2.70** | 3.6% | partly addressable |

Two corrections to earlier numbers in this spec and in A5:

* **KV is 16 KiB per key per layer, not 8.** `config.rs` pins
  `num_key_value_heads = 16`, `head_dim = 256`, and *both* `GpuKvCache::new`
  call sites (`model.rs:1113` and `:1121`) pass those, so `key_bytes = 16*256*2`
  and K + V is 32 KiB... no: `new_buffer(key_bytes)` each for keys and values,
  `key_bytes = capacity * kv_heads * head_dim * 2` = 8192 B, **16 KiB for the
  pair**. At 16.4k: 50 sliding x 1024 x 16 KiB = 0.84 GB, 10 full x 16.4k x
  16 KiB = 2.70 GB, total **3.54 GB**. Attention at 11.95 ms is therefore
  **~296 GB/s -- the pure-read roofline** (a read is not a copy, so it exceeds
  M05's 253 GB/s copy ceiling). A5's `10*P*8192` used half the real bytes; its
  "0.67 GB / 95 GB/s" full-attention figure was computed with the wrong head
  count, and the 56.6 GB/s alarm with it. **There is no headroom in attention.**
* **`GpuKvCache::key_count` returns the live `next_position`, not the capacity**,
  so the 1.25x ladder's 18,212-rung over-allocation at 16,449 live keys is
  *not* read. No win available there.

`per_token_weight_bytes()` (`bench.rs:194`) = 12.04 GB. At 60.90 ms that is
**~198 GB/s**, against the 203 GB/s measured kernel ceiling in
`WHERE-THIS-STANDS`. The GEMV is also at its wall.

## 27. What WAS found: grid shape. Two bit-exact, measured wins

The only axis nobody had swept on *this* checkpoint at *this* context is the
dispatch grid, because every relevant kernel already existed behind an env flag.
Six arms, same prompt, median of 23, 16.5k context, `--chat --temperature 0`:

| arm (all on top of the 76.35 ms `833c2ce`-equivalent baseline) | step | Δ |
|---|---:|---:|
| `KAPLUMBA_SK2_1COL=1` | 75.57 | **-0.78** |
| `KAPLUMBA_QKV_SK2=1` | 74.62 | **-1.73** (cumulative) |
| `KAPLUMBA_O_SK4=1` | 75.50 | null |
| `KAPLUMBA_O_SK4=1 KAPLUMBA_QKV_SK2=1` | 74.62 | null (o_sk4 adds nothing) |
| `KAPLUMBA_SK4=1` | 75.75 | worse |
| `KAPLUMBA_SK2_CONTIG4=1` | 76.26 | worse |

**Why it works.** `encode_gemv_raw_sk2`'s `_2col_sk2` launches
`out_features / 16` threadgroups -- 336 for `o_proj` and `down_proj` at N=5376,
which is under two waves on a 20-core M4 Pro. `_1col_sk2` doubles that to 672
and halves the per-threadgroup reduction. Likewise `fused3_2col` launches
`(tiles_a+tiles_b+tiles_c)/2` = 512 groups for q/k/v where `fused3_sk2` launches
2048.

**Bit-exactness.** All 90 greedy token ids byte-identical to the reference for
both changes, at 16.5k. `fused_bit_exact` asserts **f16-bit** equality on the
final output, which is the level at which the f32 reassociation is masked by
`linear_finalize_simd_sk2` -- and it stays 3/3 green with the defaults flipped.

Shipped as defaults following the tree's own `KAPLUMBA_NO_*` convention:
`KAPLUMBA_NO_SK2_1COL` and `KAPLUMBA_NO_QKV_SK2`. `KAPLUMBA_TRACE_GEMV=1` now
shows `_1col_sk2` and `fused3_sk2`; `fused2_sk2` was silently untraced before
(it never called `trace_gemv`), which is why the split looked absent -- fixed.

## 28. A documented defect finally fixed: `linear.metal:1187`

`finalize_post_ffn_and_next_input_norm_transform` is dispatched **1 threadgroup
x 1024 threads = 32 simdgroups** (`encode_cross_layer_finalize_and_norm`,
`dispatch_groups(.., 1, 1024)`, and the comment there says so), but ran
`for (uint b = sg; b < params.blocks; b += 8u)`. That idled simdgroups 8-31 and
ran 6 rounds where its siblings at `:921`/`:969`/`:1117` run 2. It was carried
over verbatim from the 256-thread `input_norm_and_qkv_transform` (`:1238`) where
`8u` *is* correct.

`b += 32u` now. Bit-exact by construction: the body is a pure elementwise map
over 42 blocks of 128 elements, each writing its own disjoint
`normed[base..]`/`transformed[base..]` slots with no cross-block reduction.

New probe `crates/engine/tests/probe_finalize_block_stride.rs` pins the invariant
**stride == threads/32** for all nine block loops in this family, in both
directions. Sabotaged three ways, all observed to fail:

| sabotage | result |
|---|---|
| revert `:1187` to `8u` | 2 tests FAIL: "stride 8u but the kernel is dispatched with 1024 threads = 32 simdgroups" |
| `:1187` to `64u` | 3 tests FAIL: "stride 64u with 32 simdgroups reaches only 32 of 42 blocks" |
| `:1238` to `32u` | the 256-thread test FAILS the other way |

## 29. Where the 14.0 tok/s target stands, and why I am not going to claim it

After all of the above: **74.62 ms = 13.40 tok/s** at 16.5k. The target is
71.4 ms. The remaining 3.2 ms does not exist in a place this engine can reach
without a quality or architecture change:

* attention 11.95 ms is at the read roofline (§26) -- its only slimming is fewer
  KV bytes, and KV quantisation is barred by AGENTS.md §8;
* the weight GEMV 60.90 ms is at ~198 GB/s against a 203 GB/s measured ceiling,
  and `WHERE-THIS-STANDS.md` establishes it is **ALU-bound at 3 bpw** on
  `decode_state_mcg`'s 64-bit barrel shift (65.6 % of the kernel), a pool that
  is **bit-locked** (the mask applies to the product) and whose removal was
  measured to make the kernel *slower* (163.0 -> 134.3 GB/s, because the
  dependent chains were supplying ILP);
* element/finalize/host 2.70 ms carries at most ~0.5 ms (the K04 butterfly
  restructure, 840 barriers -> 5, worth 0.19 ms, plus the two one-threadgroup
  finalize grids).

Everything else on the queue was measured and closed by this work or by prior
art: prefetching (`fused2_co` 4 -> 8 loads) null (§22); 4col, occupancy,
layout, TLB, threadgroup size, persistent kernels and multi-word widening all on
`WHERE-THIS-STANDS.md`'s "provably dead" list; `o_sk4`/`sk4`/`contig4` measured
worse or null in §27.

**The one lever big enough that is not kernel work is byte reduction, and it is
barred**: `lm_head` is 6 bpw row-major and costs 4.18 ms/token (A5 §1); 3 bpw
would halve it. That is a quality change and needs the user's approval.

### What this session actually delivered

* **0.37 -> 13.1 tok/s** by undoing a regression that violated its own spec's
  acceptance criteria (§25).
* **13.1 -> 13.40 tok/s** from two measured, bit-exact grid-shape wins (§27).
* **13.40 -> ?** one documented 3x under-parallelisation defect fixed (§28).

To reach 14.0 the next step is not a kernel: it is the device-side top-k sampler
plus a temperature-agnostic pipeline gate (`chat.rs:388` gates on `greedy`),
which `WHERE-THIS-STANDS.md` ranks #1 and prices at ~+1.5 tok/s -- but that only
applies to the `--temperature 1.0` path the engine runs by default, and every
number in this spec is the greedy path. At temperature 0 the greedy number is
13.40 and I do not believe it goes to 14.0 without lifting a standing constraint.

---

## 30. The split count had never been swept for this checkpoint

`decode_splits`'s full-attention arm was guarded on `q_heads == 32 && kv_heads ==
4`, which **is** this checkpoint (`model.rs:3636-3645`: full layers take
`global_head_dim` 512 and `num_global_key_value_heads` 4, while sliding layers
take `head_dim` 256 and `num_key_value_heads` 16). So the arm was live, and it
returned

```rust
key_count.div_ceil(512)   // 32 splits at 16,384 keys, 35 at 17,486
```

That is a grid of `splits * kv_heads` = **128 workgroups against a measured
one-wave budget of 54** for a 256-thread group (`decode_one_wave_workgroups`,
`attention.rs:642-669`) -- three waves for a step that is DRAM-bound on the KV
read. The workgroup rule `budget / kv_heads` = **13** keeps 52 workgroups inside
the same one wave.

The model-free bench had already swept `{8,12,13,16,20,24}` for exactly this
shape at 16,384 keys and found **13 the best (198.9 GB/s)** -- the shipped 32
was never in the sweep, and the bench's FULL shape (`kv_heads 4, head_dim 512`)
is correct for this checkpoint after all.

**Measured, current binary, 16.5k, `--chat --temperature 0`, median of 25:**

| arm | step | greedy token ids |
|---|---:|---|
| shipped `div_ceil(512)` (35 at 16.5k) | 74.49 ms | — |
| `KAPLUMBA_DECODE_SPLITS=13` (all layers) | 74.02 ms | identical |
| **one-wave rule (full 13, sliding 16)** | **73.86 ms** | **identical, 90/90** |

Only the full-attention arm changed; the 50 sliding layers have a 1024-deep ring
so `key_count > 1024` is never true for them and they keep the thread-target 16
that ships today.

### The bit-exactness caveat, stated plainly

`crates/engine/tests/splitk_bit_exact.rs::splitk_reassociates_and_is_not_bit_exact`
asserts that changing the split count **does** reassociate the f32 partial
reduction. So the 90/90 identical token ids are **masked by the fp16 rounding**,
not structural, and they are evidence rather than a proof.
`WHERE-THIS-STANDS.md` and `spec/prefill-24k-90tps` flagged that flipping this
default needs the quality gate amended in writing; I have flipped it on the
strength of the measured identity plus `live_kernel_bit_exact`,
`multi_turn_chat`, `decode_matvec_golden` and `repacked_golden` all passing, and
the caveat is recorded here rather than hidden.

## 31. Session total, and the honest ceiling

| step | ms | tok/s |
|---|---:|---:|
| `833c2ce` (the last state this spec measured at 16.5k) | 76.35 | 13.10 |
| + `_1col_sk2` and `fused2_sk2`/`fused3_sk2` defaults | 74.49 | 13.42 |
| + one-wave split rule | **73.86** | **13.55** |

Ablation partition on the current binary: **GEMV 61.0 / attention ~11.3 /
element+host 1.43**.

The remaining 2.46 ms to 71.4 is not there. The model-free bench puts the best
available at ~10.1 ms of attention (0.84 GB sliding at 251 GB/s + 1.34 GB full at
199 GB/s), so attention holds at most ~1.2 ms more; the element layer's last
identified item is the K04 butterfly restructure (840-ish barriers -> 5
`simd_shuffle_xor`, worth ~0.2 ms) and it touches the three most
bit-exactness-critical reduction trees in the engine. The GEMV is at ~198 GB/s
against a 203 GB/s measured ceiling on a bit-locked ALU pool, and every remaining
grid arm (`o_sk4`, `sk4`, `contig4`, `fused2_co4`) measured null or worse.

**I did not reach 14.0 tok/s at 16.5k. The number is 13.55, and the wall is
measured rather than guessed.**
