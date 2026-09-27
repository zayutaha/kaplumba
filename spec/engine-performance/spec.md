# Consolidated Engine Performance Specification (Decode & Prefill)

## 1. Goal

Consolidate the complete engineering history, benchmark data, physical rooflines, architectural findings, and open technical decisions for both **Decode** and **Prefill** optimization on Gemma 4 31B (EXL3 ~3.04 bpw) running on Apple Silicon (M4 Pro).

Provide a single authoritative source of truth for:
1. What was done, measured, and verified on decode (1.18 → 7.87 tok/s).
2. What was done, measured, and verified on prefill (9.67 → 11.44 tok/s).
3. The mathematical and physical ceilings of the M4 Pro architecture under bit-exact vs relaxed-precision regimes.
4. All open architectural questions and trade-offs requiring user direction.

---

## 1a. MANDATORY: Reference Implementations

**Any agent picking up this spec MUST read oMLX and ishizuki before designing or
changing a kernel.** These are the two projects that have already solved the
problems we are still paying for. Do not start from first principles; start from
what is already known to work on this exact hardware.

| Reference | Location | License | Why it is mandatory |
|---|---|---|---|
| **oMLX** | `reference/omlx` (local, gitignored) | Apache-2.0 | Metal-native MLX inference server. Has the fastest known prefill attention on Apple Silicon. Direct line of descent for this engine. |
| **ishizuki** | <https://github.com/struffl/ishizuki> (clone to `/var/folders/tp/wz3877_s1z756p1kcsz3_d4h0000gn/T/opencode/ishizuki`) | AGPL-3.0 | Reportedly ~11 tok/s decode on M4 Pro and ~20 tok/s on M1 Max. The decode target we are chasing, already achieved. |
| exllamav3 | `/var/folders/tp/wz3877_s1z756p1kcsz3_d4h0000gn/T/opencode/exllamav3` | MIT | Source of the EXL3 format and its GEMM/`cp.async` staging. |
| llama.cpp / ik_llama.cpp / vLLM | `reference/llama.cpp`, `reference/ik_llama.cpp`, `reference/vllm` | MIT / MIT / Apache-2.0 | Secondary; useful for quantization and prefill-scheduling ideas. |

### 1a.1 Specific things to go and read

| Question we could not answer | Where the answer probably lives |
|---|---|
| How does anyone get prefill attention off a single thread? | oMLX `omlx_sdpa_decode` / `omlx_sdpa_prefill` in `omlx/backend/metal/kernels/` — uses 32 simdgroups per `(query, head)`, each running its own online softmax over an interleaved KV stride, merged in a 4 KB threadgroup reduction. **No second kernel, no serial thread.** This is the single biggest thing we are missing. |
| Is 11 tok/s decode actually the machine, or did ishizuki just do less? | ishizuki's matvec and attention kernels. It reports ~11 tok/s on M4 Pro and ~20 on M1 Max (400 GB/s) — i.e. bandwidth-scaled, not a different algorithmic ceiling. |
| How should weights be staged from global into shared memory? | exllamav3 `exl3_gemm_inner.cuh` — `cp.async`/`int4`, 16 bytes per thread, 3-stage pipeline, zero redundant loads. |
| Does the online softmax need one `exp()` per key, or per tile? | oMLX `TK = 4` — updates the running state once per 4 keys, not once per key. |
| Can several projections share one matvec? | oMLX `qwen35_packed_linear.py` `project()` — concatenates q/k/v and gate/up along N at load time. Its `N_THRESHOLD = 8192` heuristic says fuse below, un-fuse above. |
| What is the real tensor-core layout and peak? | Metal `simdgroup_multiply` on `simdgroup_half8x8`. oMLX uses it for QK and PV. Our probe compiles and runs but its identity test matches only 8/64 elements — see Q5. |

### 1a.2 Rules for using these references

1. **Read-only.** Never modify `reference/` unless a task explicitly says to vendor a patch. Prefer describing the port in `plan.md` and implementing it ourselves.
2. **Cite the file and line** for every idea we adopt, in `implementation_notes.md`. "oMLX does it this way" is not a justification; `path:line` is.
3. **Measure before and after.** A reference proves a technique *can* work, not that it works *here*. Every port gets an A/B number.
4. **Honour the licences.** oMLX (Apache-2.0), exllamav3 (MIT) and llama.cpp (MIT) are safe to derive from. **ishizuki is AGPL-3.0 — see Q9 before porting its code.** Reading it for ideas is fine; copying it may not be.
5. **Record dead ends.** If a reference technique does not pay off here, say so in `implementation_notes.md` with the number. The 64×64 tile post-mortem exists precisely because oMLX/exllamav3 use large tiles and it still made us *slower* — that finding is only useful if it is written down.

---

## 2. Standing Constraints & Hardware Environment

Carried strictly from `AGENTS.md` §8:
- **Language & Runtime:** Pure Rust engine with raw Apple Metal shaders. Absolutely no Python in engine/product hot paths.
- **Hardware:** Apple M4 Pro (14 CPU cores: 10P + 4E; 20 GPU cores; 273 GB/s unified memory bandwidth; measured streaming bandwidth ~222–235 GB/s).
- **Target Workload:** Single-user local chat. No batching for its own sake; no multi-tenant request scheduling.
- **Model:** Gemma 4 31B dense class in EXL3 (~3.04 bpw):
  - 60 layers total.
  - Hidden dimension: 5376; intermediate FFN dimension: 21504.
  - 32 query heads.
  - 50 sliding-window layers: 16 KV heads, head dimension 256, sliding window 1024.
  - 10 full-attention layers: 4 KV heads, head dimension 512, global context.
  - Vocab size: 262,144. Non-embedding/head params: ~28.74 billion.
  - Weights footprint: ~11.47 GB weights (~11.51 GiB resident with mapped embedding and `lm_head`).
- **Memory & Cache Restrictions:**
  - No KV cache quantization.
  - No lossy requantization.
  - **No graph/tree-shaped prefill scratch and no O(L²) score matrix.** Working set must remain O(L) online-softmax (flash-style).
  - Scratch buffers explicitly owned, fixed-size, independent of context length.
  - No speculative decoding (must be greedy-identical / verification-gated if ever introduced).
  - Peak RSS is a measured acceptance figure.

---

## 3. Decode Work Summary (1.18 → 7.87 tok/s)

### 3.1 Baseline Diagnosis
Decode originally ran at **1.18 tok/s** (~850 ms per token). Profiling revealed:
- `rms_norm`: Ran on a **single GPU thread** when `rows == 1` (`crates/kernels/src/elements.metal:92`). 4 norms per layer × 60 layers = 240 single-threaded passes per token, taking **~143 ms**.
- `online_attention`: Dispatched 32 threads total (one per query head) each holding a 2 KB `float accum[512]` array, taking **~380 ms**.
- Linear GEMV: Memory-bandwidth bound, taking **~220 ms**.
- Pipelining & sampling: Blocking CPU-GPU synchronization between logits generation and token selection.

### 3.2 Implemented Milestones & Verified Commits
1. **Resident Memory Reduction (`f20f301`):** Mapped `embed_tokens` and `lm_head` from disk, reducing resident footprint from **15.12 GiB to 11.51 GiB** (under the 12 GiB target).
2. **Pipelined GPU Greedy Sampling (`8c3ea95`):** Shifted argmax to GPU and pipelined execution across iterations → **6.99 tok/s**.
3. **Corrected Roofline Analysis (`89eacc8`, `b811675`):** Proved M4 Pro ALU peak was not 19 tok/s wall; true bandwidth roofline is **22.6 tok/s** (273 GB/s ÷ 12.06 GB).
4. **Matvec Lane-Invariant Hoisting & Unrolling (`49814d4`):** Cleaned decode GEMV loop structure → **7.43 tok/s**.
5. **Parallel Coalesced Attention Reduction (`e1d72f2`):** Split-key reduction and removed thread serialization → **7.52 tok/s**.
6. **Hoisted Shared Input Transforms (`22c6325`):** Hoisted repeated layernorm input calculations → **7.87 tok/s**.
7. **Grid-Stride Matvec (`63f94e5`):** Added configurable grid-stride cap (default uncapped; paused for prefill work).

### 3.3 Decode Roofline Status
- Peak Theoretical Bandwidth Roof: 273 GB/s ÷ 12.06 GB/token = **22.6 tok/s**.
- Practical Memory Stream Ceiling (~85% efficiency): **~19.2 tok/s**.
- Remaining Unfinished Decode Optimizations:
  - Codebook Lookup Table (LUT): Replaces MCG hash integer ALU ops with a 256-entry table.
  - Cooperative Ring Load: 32 lanes of a simdgroup load one 128-bit word together rather than overlapping unaligned 32-bit loads.
  - Optimal grid-stride tuning.

---

## 4. Prefill Work Summary (9.67 → 11.44 tok/s)

### 4.1 Baseline Diagnosis & The 152 / 200 tok/s Target
Baseline prefill scaled poorly with context:
- 24 tokens: 4.28 tok/s
- 120 tokens: 8.31 tok/s
- 600 tokens: 9.67 tok/s (0.56 TFLOPS)

The user initially targeted **200 tok/s at 8192 context**, and subsequently demanded **152 tok/s**.

#### Theoretical Arithmetic Rooflines
- Model computation: Gemma 4 31B has ~28.74 billion active linear parameters.
- Flops per token (GEMM only): 2 × 28.74e9 ≈ **5.75e10 FLOPs/token**.
- Measured Apple M4 Pro (20 GPU cores) SIMT fp32 FMA peak (`examples/alu_peak.rs`): **4.38e12 FMA/s = 8.76 TFLOPS**.
- **Theoretical GEMM-only Ceiling under Bit-Exact Scalar FP32:**
  $$\text{Max Throughput} = \frac{8.76 \times 10^{12} \text{ FLOP/s}}{5.75 \times 10^{10} \text{ FLOP/token}} \approx \mathbf{152.3 \text{ tok/s}}$$
- **Critical Implication:** 152 tok/s is the absolute theoretical maximum for GEMM alone if SIMT FP32 arithmetic achieves 100% hardware efficiency, **and** attention, memory traffic, layernorm, and dispatches take **0.00 ms**.
- Reaching 200 tok/s requires **11.5 TFLOPS**, which is physically impossible on the scalar fp32 SIMT units and strictly requires Metal tensor cores (`simdgroup_multiply_accumulate`).

### 4.2 Implemented Milestones & Measurements
1. **Register Accumulator (`11aef59`):** Replaced global atomic `+=` read-modify-write in `linear_gemv_tiled` with thread-local register accumulation in sequential $k$ order → **9.86 tok/s** (+2%). Post-mortem: global atomics were L1-resident, not saturating DRAM.
2. **Head-Dim Specialization (`500a902`, `724fa4d`):** Specialized `online_attention` for `HEAD_DIM=256` and `512` so loops compile with constant bounds → **10.26 tok/s**.
3. **Coalesced Trellis Staging + `KCHUNK=64`:** Staged trellis words into threadgroup memory before decoding → **10.99 tok/s**.
4. **Threadgroup Attention with Register Accumulators (`12fa4d8`):** Restructured attention into one 256-thread threadgroup per `(query, head)` (`online_attention_tg2_256/512`), removing local memory spills → **11.44 tok/s**.
5. **The 64×64 Tile Experiment (`ba9263f`):**
   - Theory: Widening output tile from 16×16 to 64×64 boosts arithmetic intensity from 13.4 to 53.8 FLOP/byte, raising the bandwidth ceiling from 3.2 TF to 12.6 TF.
   - Result: Dropped throughput from 11.44 tok/s to **8.27 tok/s** (and 2.58 tok/s when a runtime loop bound caused local memory spilling).
   - Post-Mortem: 20 KB of threadgroup memory per group severely throttled GPU core occupancy; weight decoding (~10 ALU ops per weight with division/modulo) dominated. Reverted cleanly to 16×16 (11.38 tok/s).

### 4.3 Current Cost Breakdown at 600 Context
At 600 prompt tokens (87.4 ms total per token):
- **Attention:** ~44.7 ms/token (~51% of time)
- **GEMM:** ~42.7 ms/token (~49% of time)
Both halves are roughly equal bottlenecks. Neither can reach 152 tok/s on its own.

---

## 5. Comprehensive Enumeration of All Open Questions

In accordance with `AGENTS.md` §2.1, every open question includes a plain-language gloss in brackets.

### Q1: Strict Bit-Exactness vs Standard Numerical Tolerance
> **Question:** Does the user require bit-for-bit identical floating-point arithmetic matching the sequential scalar fp32 reference, or is standard IEEE float tolerance / greedy generation coherence acceptable?
>
> *(If we must match every single bit of the original sequential math, we cannot use Apple's tensor cores and we cannot use fast parallel attention reductions. Under strict bit-exactness, 152 tok/s prefill is physically impossible on this chip. If you allow mathematically equivalent parallel math that produces the same words, high speeds become possible.)*

### Q2: Target Prefill Throughput Realism
> **Question:** Given that scalar fp32 math has an absolute physical ceiling of 152.3 tok/s for the GEMM alone (assuming 0 ms for attention), what target throughput should be committed in the spec: the bit-exact ceiling (~15–25 tok/s), or the tensor-core hardware accelerated target (100–150+ tok/s)?
>
> *(At bit-exact precision, 15–25 tok/s is the real physical boundary. 152 or 200 tok/s strictly requires tensor cores. Which target do you want the spec to plan for?)*

### Q3: Attention Parallelization Strategy (Serial vs Parallel Softmax)
> **Question:** Should prefill attention remain serialized on thread 0 for the online-softmax recurrence (advancing max and sum strictly one key at a time), or should we adopt oMLX's multi-simdgroup parallel reduction with threadgroup merge?
>
> *(Currently 255 out of 256 threads sit idle while 1 thread calculates softmax to preserve exact math order. oMLX lets all threads work simultaneously and combines partial answers at the end, which is ~30× faster but changes the last decimal bits of the float sums. Which approach should we take?)*

### Q4: Prefill GEMM Weight Decode Architecture
> **Question:** In the prefill GEMM, should we replace the procedural MCG hash decode (~10 ALU ops per weight with integer div/mod) with a precomputed 256-entry lookup table (LUT) or SIMD shuffle?
>
> *(Every time we multiply an activation by a weight, we run expensive math to unpack the 3-bit weight. If we precalculate the 256 possible unpacked values into a small table or register shuffle, can we unlock the 6× missing GEMM throughput?)*

### Q5: Metal Tensor Core Simdgroup Layout Verification
> **Question:** For the tensor-core path, how should the 8×8 simdgroup matrix layout bug in `crates/engine/examples/metal_probe.rs` (where identity check matched only 8 of 64 elements) be resolved?
>
> *(The M4 Pro tensor hardware compiles and runs, but Metal stores 8x8 matrix tiles across the 32 SIMD threads in a specific lane arrangement. Our first test only matched 8 out of 64 numbers because our lane mapping was off. Do you want this probe fixed to establish the true tensor-core baseline?)*

### Q6: Test Harness Encoder Lifecycle Failure
> **Question:** How should the pre-existing Metal test failure in `attention_metal` (`Command encoder released without endEncoding`) be addressed?
>
> *(The attention test crashes because a Metal command encoder was dropped before `endEncoding()` was called—identical to the bug previously fixed in `forward_buffer`. Should we fix this harness lifecycle bug immediately so `cargo test` is 100% green?)*

### Q7: Priority Between Decode and Prefill
> **Question:** Should decode optimization remain paused at 7.87 tok/s, or should decode and prefill be advanced together?
>
> *(Decode was stopped at 7.87 tok/s to focus on prefill. Completing the decode tasks like the codebook table and cooperative loads can push decode to 11–15+ tok/s with low risk. Should we resume decode work or stay 100% on prefill?)*

### Q8: KV Cache Residency at 8192 Context vs 12 GiB Budget
> **Question:** At 8192 context length, KV cache requires 1.41 GiB (10 full layers + 50 sliding layers capped at 1024), bringing total residency to 12.92 GiB, which exceeds the nominal 12 GiB memory cap by 0.92 GiB. Is this 12.92 GiB footprint explicitly accepted?
>
> *(At 8192 tokens, the KV cache alone needs 1.41 GB. Combined with the 11.51 GB model weights, the process will use 12.92 GB of RAM. The original limit was 12.0 GB. Is 12.92 GB acceptable for 8192 context?)*

### Q9: AGPL Licensing of ishizuki Reference Material
> **Question:** Does the project permit clean-room adaptation of optimization concepts from `ishizuki` (licensed AGPL-3.0), or must all implementations derive solely from MIT/Apache-2.0 sources (like `omlx` and `exllamav3`)?
>
> *(One of the reference repositories is licensed under AGPL. If we look at its code for ideas, does that conflict with your licensing goals for this project?)*

### Q10: Q3 (attention strategy) contradicts Q1 (bit-exactness)
> **Status:** open, unresolved. *Decision on implementation: build the bit-exact
> option (`plan.md` §2.1, prefix-max scan) because bit-exactness was reaffirmed as
> the hard constraint. This question is recorded rather than silently dropped.*
>
> **Question:** Q1 was answered "stay bit-exact" and Q2's 150 tok/s target was
> withdrawn in favour of it, but Q3 was answered "port the oMLX multi-simdgroup
> merge". `plan.md` §2.2 defines that merge as the *relaxed-float* option whose
> threadgroup reduction deliberately changes the last bits of the float sums. The
> bit-exact alternative is the prefix-max scan in `plan.md` §2.1.
>
> *(You chose bit-exact arithmetic and also chose the parallel attention merge whose
> whole point is that it is not bit-exact. I cannot do both. Until you say
> otherwise I am implementing the bit-exact prefix-max version, which is slower.)*

### Q11: `attn_scores` is ~512x over-allocated (67 MB)
> **Question:** `attn_scores` is allocated `q_heads * head_dim * 1024` floats
> (67.11 MB) but the decode split-attention contract only needs `q_heads * key_count`
> (1.05 MB at 8192 context). Right-sizing it frees 67 MB and would let
> `PREFILL_CHUNK` go from 64 back to 256 (recovering per-chunk dispatch overhead),
> but it needs the KV capacity threaded into `ForwardScratch::new` and an assertion
> that the real requirement never exceeds the allocation.
>
> *(A scratch buffer is 64x bigger than it needs to be. Fixing it is a one-line
> change once we know the exact KV capacity, but getting that wrong under-allocates
> and corrupts memory. Do you want it fixed now, with the assertion, or left until
> the KV capacity plumbing is in?)*

---

## 6. Acceptance Criteria

1. **Definitive Decision:** User resolves Questions Q1, Q2, Q3, and Q7, selecting either the Bit-Exact Path (Option A) or the Tensor/Parallel Path (Option B).
2. **Harness Health:** `attention_metal` encoder lifecycle bug fixed; full `cargo test --release --workspace` suite passes.
3. **Numerical Verification:** Reference prompt outputs remain coherent and within established tolerances.
4. **Target Delivery:** The selected performance milestone (either ~15–20 tok/s bit-exact or 100+ tok/s tensor-accelerated) is achieved and verified by end-to-end timing excluding model load.
