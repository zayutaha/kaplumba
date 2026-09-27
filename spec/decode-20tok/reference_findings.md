# Reference survey: what the three projects do that we don't

Surveyed 2026-09-27 against our measured profile (7.89 tok/s; matvec 128 GB/s at
1344 threadgroups and 168 GB/s at 16384; lm_head 3.13 ms; attention 7.6 ms at
ctx 200; ~14 ms/token residual). Sources: `exllamav3` (CUDA, the reference
implementation of our exact EXL3 format), `ishizuki` (Swift + MLX, kernels
registered as source strings), `omlx` (MLX + hand-written Metal).

## Tier 1 — do these

### 1. Cap the grid at co-residency and grid-stride over N
`exllamav3/exl3_gemv.cu:165-176`
```c
int max_blocks = occupancy(kernel, block_dim) * num_sms;
int grid = MIN(size_n / cols, max_blocks);
```
then a grid-stride loop over the output tiles, `exl3_gemv_kernel.cuh:264`:
```c
for (int group = blockIdx.x; group < num_groups; group += gridDim.x)
```

**Why it matters here.** Our gate/up/down launch 1344 threadgroups on 20 cores.
That is ~1.4 waves — and exllamav3 measured this exact zone as the bad one:
> "in the 1..2-wave zone the trailing partial wave costs more than the kernel
> gains unless per-group work is small" (`exl3_gemv.cu:56-72`)

lm_head has 16384 threadgroups (many clean waves) and is our *best* projection at
168.7 GB/s. gate_proj at 1344 is mid-tail. A capped grid with a grid-stride loop
turns the partial wave into an exact multiple. All four exllamav3 paths use this
style. Low risk, no extra buffers, no second pass.

### 2. Shuffle-based word exchange to kill the 4x L1 read amplification
We measured this ourselves: 32 lanes issue 96 lane-loads to touch 24 distinct
trellis words.

exllamav3 makes **each lane load exactly one word** — `ld_b` at
`exl3_gemv_kernel.cuh:266-275` reads `bp + lane`, so a warp load is 24-32
*consecutive* words, one to two full cache lines — and then reconstructs the
overlapping windows **in-register via `__shfl_sync`**, with the per-lane source
indexes precomputed once at `exl3_gemv_kernel.cuh:243-259`:
```c
uint32_t aw = __shfl_sync(0xffffffffu, bw[t], (lane + 31) & 31);
```
Zero extra global/L1 traffic. This is the same 4x figure appearing in their
2 bpw path (`exl3_dq.cuh:187-193`, 64 reads for 16 words) — but they pay it
against shared memory in the general GEMM, where the global->shared copy is
coalesced and non-redundant, and pay it *not at all* in the GEMV.

**The detail that makes this work and the one that breaks it:** the source lane
index must be *uniform and precomputed*, not divergent. ishizuki measured
`simd_shuffle` as a codebook gather at **0.34x** — "a divergent lane index is
emulated" (`BENCHMARKS.md:166-167`). exllamav3 shuffles for a *trellis word* with
a compile-time-derived index, which is the version that wins.

Metal equivalent: `simd_shuffle` / `simd_shuffle_xor`, which we already use for
the accumulator reduction.

### 3. Split-K, occupancy-targeted, two passes
Present in two of the three, and in one of them in Metal:

`ishizuki/Sources/IshizukiKit/Core/VerifyMatmul.swift:89-105` — the split is
chosen to hit a simdgroup target, not an N target:
```swift
let blocks = (n / rowsPerGroup) * simdgroups
let split = (1...groups).first { groups % $0 == 0 && blocks * $0 >= occupancy } ?? groups
// occupancy = 2048
```
writing `[split, m, n]` fp32 partials, folded by a second pass. Credited in
`BENCHMARKS.md:302-305`: *"a narrow output splits its reduction across enough
simdgroups to fill the GPU."*

`omlx/omlx/patches/qwen35_packed_linear.py:357-368, 427-432, 448-469` — same
idea with an explicit target: `_TARGET_TGS = 1400`, `ksplit` = the largest
divisor of the K-step count keeping `tiles * ksplit <= target`, fp32 partials
into a `(ksplit, M, N)` slab plus a reduce kernel. **No atomics in either.**

Constraint: ishizuki rejected the single-dispatch alternative outright —
> "a megakernel buys nothing on Metal, which has no grid-wide barrier"
> (`README.md:386-387`)

So any split-K here is two dispatches. The honest prior is mixed: exllamav3
measured split-K as *"neutral to harmful on every GPU here, so the default is no
split"* for MoE (`exl3_moe_coop.cu:74-85`, `doc/env_vars.md:396-400`), and
measured that **adding** threadgroups past one co-resident wave *hurts*
(`exll3_gemv.cu:56-72`). Test it, but item 1 is the safer first move.

## Tier 2 — do after Tier 1

### 4. Single-pass online-softmax decode attention
`omlx/omlx/custom_kernels/decode_fast/csrc/sdpa_decode_kernels.h:69-290` — one
dispatch, 1024 threads, 32 simdgroups interleaving the KV axis at stride
`BN*TK = 128`, per-simdgroup online softmax, and a 4 KiB threadgroup merge at the
end. No score matrix, no second kernel, working set independent of context.

Their routing gate (`sdpa_decode.cpp:387-395`) puts a 200-token GQA context on
this path. **Ours is 3 dispatches and 7.6 ms.** This is also the shape AGENTS.md
asks for (flash-style, O(L) working set).

Two details worth copying: GQA handled by index arithmetic
(`kv_head_idx = q_head / gqa_factor`, line 91), and if a 2-pass path is kept, the
`gqa` axis goes *inside* the threadgroup (`group_dims = (32, gqa, q_len)`,
`sdpa_decode.cpp:242-243`) so the KV is read once per threadgroup instead of
`gqa` times. That last one applies to our current split-key kernel directly.

### 5. Concatenate weights along N at load time, then slice
`omlx/omlx/patches/qwen35_packed_linear.py:519-551`:
```python
y = y.reshape(*x.shape[:-1], n).astype(x.dtype)   # one matmul, n = sum(output_dims)
outputs = [y[..., off : off + lin.output_dims] ...]
```
Turns q/k/v into 1 dispatch and gate/up into 1. This is the fusion I was about to
build by hand; the reference confirms the approach and shows the buffer
mechanism (one fused output buffer, output views are slices).

exllamav3 has the same idea as "sliced mode" for unequal widths
(`multilinear.py:46-56`) and the reverse heuristic for when to *un*-fuse:
> "keep the fused MGEMM when the matrices are narrower than the N threshold
> (too narrow for separate GEMV calls to fill the GPU)" — `N_THRESHOLD` 8192,
> default (`doc/env_vars.md:180-187`)

Our N=21504 and 262144 are both above it, i.e. do **not** fuse those. q/k/v
(N=8192/4096/4096) and gate/up (21504/21504) are the arguable ones.

## Tier 3 — calibration, not code

### 6. Our ~14 ms/token residual is almost certainly dispatch overhead, and this corroborates it
`omlx/omlx/patches/qwen35_gdn_prework.py:7-12`:
> "~10 small dispatches per GDN layer per verify cycle. On the 27B that is 48
> layers x ~0.29 ms (measured sync-mode at S=4 on M3 Ultra)"

48 layers x 0.29 ms for ~480 dispatches = **~29 us per dispatch in sync mode**.
Our residual is ~14 ms. At 23 dispatches/layer x 60 = 1380, that implies ~10 us
each — cheaper per dispatch than oMLX's figure, same order, same conclusion.

Their accompanying discipline is worth copying: `_MIN_ROUTE_N = 16384`
(`qwen35_verify_qmm.py:65-71`) —
> "each `metal_kernel` invocation pays more Python/dispatch overhead than the
> built-in fast path. With ~330 projections per verify forward, routing the small
> shapes (~1.0x GPU win) costs more on the CPU than it saves"

i.e. **do not spend a dispatch on a 1.0x GPU win.** This is the discipline our
layer_bench numbers could not give us.

### 7. Load-time specialisation
`omlx/benchmarks/bonsai_decode_bench.py:466-470` names it their #1 fix for CPU
overhead. ishizuki does the Metal equivalent by precompiling every template
specialisation at load (`VerifyMatmul.swift:38-61`) — and has a gap we can copy
the *check* for: `PackedLinear.init(exl3:)` never warms the EXL3 kernels, so
their first forward per shape pays a Metal library build. We should verify our
kernels are all compiled before the timed loop, not just the ones on the hot
path.

### 8. Check prefill for the IOGPU interactivity preemption cliff
`omlx/omlx/patches/qwen35_fa256_attention.py:40-69`:
> "A single dispatch scanning the whole KV monopolizes the GPU long enough at
> high kv_len to trip the macOS IOGPU interactivity preemption on pre-NAX GPUs,
> collapsing long-context prefill (issue #2225). The M3 Max report cliffs near
> 24 heads * 2048 q * 30k keys ~= 1.5e9. ... above ~50ms per dispatch, so 10ms
> keeps a wide margin on any tier"

M4 Pro is pre-NAX. Our decode dispatches are all sub-millisecond so decode is
safe, but our prefill is at 8.4 tok/s and worth auditing for any single dispatch
over ~10-50 ms. This is a live hazard we have never checked.

## Measured dead ends (do not repeat)
- ishizuki: threadgroup sizes 128-1024 — **flat**. Changing size changes nothing;
  only the threadgroup *count* matters. (`BENCHMARKS.md:165-166`)
- ishizuki: unrolling the block loop by 2 or 4 — **1.00x** for their GGML matvec.
  We measured +6% for our EXL3 kernel, so ours differs; do not generalise either way.
- ishizuki: prefetching a block one iteration ahead — **0.90x** ("the compiler
  schedules better without the loop-carried registers").
- exllamav3: 16-byte vector B loads are absent from their fp16 GEMV; present only
  in the int8 wide unit and the cp.async GEMM path. So "just use uint4 loads" is
  not a proven win at this shape.
- omlx: no weight prefetch, no double-buffering, no async copy, and no EXL3 or
  trellis kernel anywhere. Nothing to copy on the decode matvec other than the
  items above.
- exllamav3: L2 persistence (`accessPolicyWindow`,
  `cudaLimitPersistingL2CacheSize`) is used nowhere; `cp_async_stream` with
  `L2::evict_first` is defined at `ptx.cuh:170-184` and never called.

## A calibration that matters
`exllamav3/autotune` deliberately runs **cold**:
> "Candidates in one autotune session share the same input tensors, so
> back-to-back timed launches keep each other's operands mutually warm in L2
> regardless of production's actual scattered, per-token-random access pattern."
> — `coop_autotune.cu:321-325`

It memsets a 2x-L2 thrash buffer before every timed sample (`:406`). Any kernel
A/B I run on a hot buffer is measuring the wrong thing.
