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
