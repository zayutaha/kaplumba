# Implementation Notes: Decode 19+ tok/s and 250+ GB/s via Repacked Weights

Running log for `spec/decode-19tok-repacked-250gb/`. Updated per task.

## Baseline & Context
- Target: **15+** tok/s decode, 250+ GB/s streaming bandwidth. (Header corrected from
  19; 19 is excluded by arithmetic — see `spec.md` §6.2. The number is under review and
  is the user's call, see `spec.md` R6.)
- Model: Gemma 4 31B (`~/.kaplumba/models/Gxma/Gxma`), EXL3 3.0 bpw.
- Measured baseline: ~11.2 tok/s on current Ishizuki 2-word kernel (`3230f24`).
- Physical DRAM theoretical peak: 273 GB/s (Apple M4 Pro).

## T1 & T2: Synthetic Memory Bandwidth Probe (Disproof of 244 GB/s Ceiling)
- Implemented `probe_bw_250gb.rs` and `bw_probe.metal`.
- Features:
  - Vectorized loads with `uint4` (16 bytes = 128 bits per thread per instruction).
  - High-entropy pseudo-random pattern tiling: every cacheline is non-zero and non-compressible, preventing hardware zero-line compression.
  - Concurrency: 40,960 to 81,920 threads.
  - Memory flags: `MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked`.
  - Checksum reduction across all thread outputs.
- Fail-first sabotage (T2):
  - Injected halved load count: normal checksum `0x739adf40`, sabotaged checksum `0xa8c277a0`. Checksum mismatch proved test can detect failure.
- Measured Results:
  - **2 GB Buffer:**
    - th=81920, tg=256: **251.7 GB/s median** (92.2% peak), **254.2 GB/s best**.
  - **6 GB Buffer:**
    - th=81920, tg=512: **254.0 GB/s median** (93.0% peak), **255.4 GB/s best**.
  - **12 GB Buffer (Full Model Weight Footprint):**
    - th=81920, tg=512: **250.5 GB/s median** (91.8% peak), **253.2 GB/s best**.
- **Conclusion:** Hardware DRAM streaming bandwidth CAN exceed 250 GB/s on Apple M4 Pro when 128-bit aligned vector loads (`uint4`) are used with high thread occupancy.
- ~~The previous 239-244 GB/s ceiling was an artifact of 4-byte scalar loads and thread configuration.~~ **WITHDRAWN (`spec.md` §5).** **Unsupported.** That figure came from `examples/conc.rs` (since deleted) and the predecessor spec already reconciled it as a *different shape*, not a narrower load. **Load width is a closed axis**: `decode-bandwidth-handover/spec.md:236` records "8/16/32 B all ~227-238". The conclusion above stands on its own measurement; the causal claim about the old ceiling does not, and it is the premise the withdrawn `_vec128` / `_u4` work rested on.

## T4 & T5: Column-Major Trellis GEMV (Elimination of Strided Striding)
- Baseline row-major layout `[tiles_k, tiles_n, 48]` causes each threadgroup along N to jump by `tiles_n * 96` bytes on every single K step (up to 129 KB jumps).
- Designed `reorder_tensor_to_colmajor` in `crates/engine/src/exl3.rs` to transpose tile ordering to `[tiles_n, tiles_k, 48]`. All 16x16 tiles along K are 100% contiguous in memory.
- Dequantization is 100% bit-for-bit exact; zero mathematical divergence, zero quality loss.
- Implemented `linear_gemv_ishizuki_k3_colmajor` with 4x K unroll in `crates/kernels/src/linear.metal`.
- Correctness test `crates/engine/tests/repacked_golden.rs`:
  - Sabotage: flipped bit 0 of repacked tensor, observed expected failure `orig -98.65691 vs repacked -95.42632`. Reverted and verified 100% bitwise exactness across 512 elements and verified Metal GEMV matches CPU reference dequantizer bit-for-bit.
- Measured GEMV bandwidth (`crates/engine/examples/bench_colmajor.rs`):
  - `q_proj 5376x5376`: 94.8 GB/s -> **101.8 GB/s** (0.11 ms)
  - `up_proj 5376x21504`: 151.5 GB/s -> **180.5 GB/s** (0.24 ms)
  - `down_proj 21504x5376`: 148.8 GB/s -> **177.8 GB/s** (0.24 ms)
  - `gate_up 5376x43008` (fused): **188.5 GB/s** (0.46 ms)
  - `lm_head 5376x262144` (6 bpw): **194.4 GB/s** (2.72 ms)

## T3 & T6: Model Repacking and Pipeline Integration
- Implemented `crates/engine/src/bin/repack_model.rs`.
- Converted full 16.03 GB model in 12.1 s: 410 trellis tensors repacked, `"layout": "colmajor"` added to `quantization_config.json`.
- Enforced single disk copy constraint: deleted previous destination before recreating. Original checkpoint `~/.kaplumba/models/Gxma/Gxma` strictly read-only and untouched.
- Pipeline integration:
  - Added `layout: Option<String>` to `QuantizationConfig`.
  - Added colmajor transposed shape support to `validate_storage` and `expect_exl3_storage` in `crates/engine/src/safetensors.rs`.
  - Added `perm_mode == 1` handling to `LinearParams` and `linear_gemm_mma` in `crates/kernels/src/linear.metal` to ensure 100% bit-exact prefill logits.
  - Set `permutation_mode = 1` for 3 bpw layers in `crates/engine/src/model.rs`.
- End-to-End Verification:
  - Prompt "Hi": generates `"Hello! How can I help you today"` (greedy-identical to baseline).
  - Prompt "What is the capital of France?": generates `"The capital of France is Paris."` (greedy-identical to baseline).
- Sabotage Test:
  - Deliberately dispatched row-major kernel on colmajor weights in `linear.rs`: model produced complete gibberish `"TheBHC deBLC"`. Reverted and verified exact golden output restored.

## Profiling and Kernel Optimization Findings

1. **`simd_shuffle` Cooperative Loads (Negative Result - Reverted):**
   - Experiment: In `linear_gemv_ishizuki_k3_colmajor`, used 24 threads in a simdgroup to load 24 contiguous uints, sharing across threads via `simd_shuffle(w_raw, j0)`.
   - Result: Bandwidth dropped from **180.5 GB/s to 146.5 GB/s** (latency increased from 0.24 ms to 0.30 ms on `up_proj`).
   - Root Cause: Dynamic register shuffles (`j0` differing per thread) serialize through the SIMD crossbar execution port on Apple Silicon AGX cores, whereas scalar reads into L1 cache are serviced concurrently by L1 data cache read ports. Reverted to 4x unrolled scalar loads.

2. **Simdgroup-Streaming Kernel (`_stream`) (Tradeoff Analysis):**
   - Experiment: Removed threadgroup barrier and shared memory reduction by assigning each simdgroup its own 16 output rows (`tn = tgid * 8 + sg`).
   - Result: On wide matrices (`gate_up 5376x43008`), achieved **184.9 GB/s** (0.47 ms) without barriers.
   - Bottleneck: For narrow matrices (`down_proj 21504x5376`), grid size was cut by 8x (from 336 threadgroups to only 42 threadgroups), causing GPU underrun (bandwidth collapsed from 178 GB/s to 110.8 GB/s). Conclusion: Multi-simdgroup Split-K reduction is required on narrow N to maintain sufficient grid occupancy across all 20 GPU cores.

4. **Command Buffer Rolling & Dispatch Overhead:**
   - Finding: Default `MAX_DISPATCHES_PER_BUFFER = 50` was chunking each 480-dispatch decode step into 10 separate Metal command buffers.
   - Result: Host submission latency and multi-buffer synchronization inflated per-token time to ~169 ms (5.91 tok/s).
   - Solution: Increasing dispatch budget per buffer for decode steps to 2500 commits the entire 60-layer pass in a single command buffer, immediately dropping step latency to **83.9 ms GPU own time (11.46 tok/s)**.
   - **STALE — this finding is now measured and it is smaller than described.** A decode
     step issues **1407 dispatches** (1167 after T7.6.10c), and since that is below
     `max_dispatches = 2500`, `tick()` **never fires `roll()`**: the step commits **2
     command buffers, one of them empty**, not one, and `MAX_IN_FLIGHT` is never
     engaged. The "~169 ms / 5.91 tok/s" before-state is real but was measured against
     the pre-fusion dispatch count; the useful part of this entry is the *method*
     (raise `max_dispatches` for decode), not the number.

5. **Profiler Distortion Finding (`KAPLUMBA_DECODE_PROFILE=kernel`):**
   - Observation: When `KAPLUMBA_DECODE_PROFILE=kernel` was enabled, decode throughput dropped from 11.46 tok/s to 0.7-2.2 tok/s.
   - Root Cause: Kernel profiling mode injects CPU `wait_until_completed` drains between kernel groups to read discrete hardware timestamps. This serializes the GPU execution against the CPU dispatch loop. Per AGENTS.md §5, kernel profiling mode is solely an attribution tool and must never be read as a throughput benchmark.

6. **The 3 bpw 96-Byte vs 128-Byte DRAM Burst Amplification Discovery:**
   - Tile math: EXL3 3.0 bpw tile is $16 \times 16 \times 3\text{ bits} = 768\text{ bits} = \mathbf{96\text{ bytes}}$ (24 uint32s).
   - Hardware bus: Apple Silicon M4 Pro LPDDR5 memory controllers burst in **128-byte transactions** (paired 64-byte cache lines).
   - Mismatch: $96 / 128 = 0.75$. Every 96-byte tile fetch loads a 128-byte block.
   - Amplification factor: $128 / 96 = \mathbf{1.333\times}$.
   - Physical DRAM bus: Measured effective weight throughput of **188.4 GB/s** means the physical hardware DRAM bus is transferring $188.4 \times 1.333 = \mathbf{251.2\text{ GB/s}}$ (92% of theoretical peak 273 GB/s).
   - Path to 250+ GB/s effective weight bandwidth: Group 4 consecutive tiles along K into $4 \times 96\text{ bytes} = 384\text{ bytes} = \mathbf{3 \times 128\text{ bytes}}$ (6 full 64-byte cache lines) and issue 16-byte aligned vector loads (`uint4`), eliminating the $1.333\times$ hardware overhead completely.
   > **⚠️ THE ENTIRE BLOCK ABOVE (§6) IS WITHDRAWN AS A CAUSAL CLAIM. Read `spec.md` §5.**
   > The burst mechanism is wrong: a colmajor column is 32,256 B = 252×128, fully
   > consumed, and consecutive K-tiles are gapless at 96 B stride with 128 B-aligned
   > column bases (`exl3.rs:750-753`, `linear.metal:7780-7781`). No measurement or
   > citation for the 128-byte burst exists in this repo or in `reference/omlx`.
   > The $188.4 \times 1.333 = 251.2$ line is a fit to 92% of 273, not a diagnosis.
   > The `_u4` "path" was built (T7.5) and **measured 180.5 → 173.7 GB/s on `gate_up` —
   > a loss.** The T14 `D=32` follow-up (halving amplification directly) was **0.288–0.918×
   > on all 11 shapes.** Both are dead. The tile math in the first three bullets is
   > correct; the conclusion drawn from it is not.

7. **Ground-Truth Decode Breakdown (Measured via `KAPLUMBA_DECODE_PROFILE=kernel KAPLUMBA_DECODE_PROFILE_WARMUP=1`):**
   - Steady-state decode step is **85.7 – 86.5 ms** (11.6 tok/s).
   - Per-kernel GPU time breakdown:
     - `gate_up_proj` (60 layers, `fused2`): **35.58 ms** (37.5%)
     - `down_proj` (60 layers): **18.37 ms** (19.4%)
     - `qkv_proj` (50 layers, `fused3`) + `q/k/v` (10 full-attn layers): **19.94 ms** (21.0%)
     - `o_proj` (60 layers): **9.74 ms** (10.3%)
     - `head/gemv` (1 dispatch, 6 bpw): **4.17 ms** (4.4%)
     - Non-linear ops (`rms_norm`, `attention`, `rope`, `gated_gelu`, `residual`): **7.04 ms** (7.4%)
   - Key finding: 87.8 ms out of 94.8 ms is in linear projections.
   - ~~At 188 GB/s across all layers, the total weight transfer (9.88 GB) takes 52.5 ms (19.0 tok/s).~~
   - ~~At 250 GB/s across all layers, the total weight transfer takes 39.5 ms (25.3 tok/s).~~
   - **BOTH LINES WITHDRAWN — the byte count is wrong by 22% and the conclusions are
     excluded.** Three separate errors, each independently fatal:
     1. **9.88 GB is wrong. The correct figure is 12.039 GB/token** (`spec.md` §6.1,
        `12,039,487,488` B). The full-attention per-layer sum was mis-added. A 9.88 GB
        denominator understates every bandwidth figure derived from it.
     2. **The arithmetic does not follow from the corrected number either.** 12.039/188 =
        64.0 ms (13.2 tok/s), not 52.5. And 12.039/250 = 48.1 ms, not 39.5.
     3. **The 188 GB/s "ceiling" was a fit**, reverse-engineered to land on 92% of 273
        via the withdrawn 1.333× burst story (`spec.md` §5). It is not a measurement.
     4. **19.0 and 25.3 tok/s are both excluded.** 19 needs 264–283 GB/s of
        *dequantizing* GEMV (105–113% of a pure copy, `spec.md` §6.2). 25.3 tok/s would
        need 31.5 ms, i.e. the weights alone in 31.5 ms = **382 GB/s** — 1.5x a memcpy.
     **Do not carry any tok/s figure out of this block.** The current, measured state is
     13.511 tok/s (`spec.md` §6.4).

8. **Grid Under-Subscription on Narrow Layers (Solved by Split-K 4):**
   - On `q_proj 5376x5376` and `o_proj 5376x5376`:
     - Grid size is only 336 threadgroups (16 threadgroups per core on 20 GPU cores).
     - Measured throughput was only 108.5 GB/s (0.10 ms).
     - With `linear_gemv_ishizuki_k3_colmajor_sk4` (`split-k 4`): grid size increases to 1344 threadgroups (67 per core), throughput jumps to **142.6 GB/s** (0.08 ms, **+31% speedup**).

9. **`simd_shuffle` vs Direct L1 Cache Gather:**
   - On small matrices (`5376x5376`), `simd_shuffle` improved throughput from 108.5 GB/s to 128.1 GB/s (+18%) by removing memory latency.
   - On wide matrices (`gate_up 5376x43008`), `simd_shuffle` reduced throughput from 189.0 GB/s to 153.0 GB/s because the ALU crossbar becomes the bottleneck rather than DRAM bandwidth.
   - Conclusion: L1 cache on M4 Pro already handles contiguous column-major tile reads at 189–197 GB/s without ALU register shuffling.

10. **U4 Vector Coalesced 128-Byte Aligned Loads (`linear_gemv_ishizuki_k3_colmajor_u4`):**
    - Grouped 4 consecutive tiles along K ($4 \times 96\text{ bytes} = 384\text{ bytes}$) loaded via 24 `uint4`s by lanes 0..23 into 128-byte aligned chunks.
    - Verified bit-exact against CPU reference (within 1-ULP fp addition associativity tolerance across 512 elements).
    - Benchmarked in `bench_colmajor`:
      - `q_proj 5376x5376`: boosted from 67.9 GB/s to **95.8 GB/s** (+41% speedup, 0.16 ms -> 0.11 ms).
      - `up_proj 5376x21504`: boosted from 135.3 GB/s to **153.5 GB/s** (+13.5% speedup, 0.32 ms -> 0.28 ms).
      - `gate_up 5376x43008`: 173.7 GB/s.
      - `lm_head 5376x262144`: 194.2 GB/s.

11. **Decode Latency Variance & Drop Analysis:**
    - Profiling instrumentation overhead: `KAPLUMBA_DECODE_PROFILE=kernel` samples GPU hardware timestamps on every dispatch and synchronizes CPU encoder, collapsing measured decode rate from 11.6 tok/s down to 3.7 tok/s. Real throughput must be measured uninstrumented.
    - Split-K reduction dispatch penalty: Split-K 4 requires a second dispatch (`gemv_reduce_sk4`) per layer (60 extra dispatches per token), adding command buffer encoder overhead that offsets GPU compute gains.
    - Inner-loop barrier penalty: inserting `threadgroup_barrier` inside the K-loop serializes the 8 SIMD groups, preventing memory latency hiding. The K-loop must remain barrier-free.
    - Memory compression variance: macOS memory compressor (`top PhysMem` showing `588M compressor`) and DRAM thermal throttling introduce natural variance between runs (measured between 9.4 and 11.6 tok/s).

12. **The 250 GB/s Physical Repacking Analysis & Trade-offs:**
    - Pure streaming probe (`probe_bw_250gb.rs`) hits **254.0 GB/s** because it performs ZERO dequantization math, ZERO barrel shifts, and ZERO codebook lookups.
    - In 3 bpw GEMV, each 96-byte tile requires 256 dequantizations (`decode_state`) and 256 FP32 FMAs. At 190+ GB/s, the 20 GPU cores become ALU instruction bound alongside memory streaming.
    - Padding 96-byte tiles to 128 bytes would add 32 bytes of zeros per tile (+33% footprint), ballooning weights from 15.5 GB to 20.7 GB. This would violate the unified memory budget and force paging to disk swap (collapsing to ~1.4 tok/s).
    - Conclusion: 384-byte coalesced chunking (4 tiles of 96 bytes) is the optimal footprint-preserving repacking format.

13. **Fast Zero-Copy Argmax in `generate_gpu` (`9848a2b`):**
    - Eliminated per-token `f16 -> f32` conversion and 1 MB heap `Vec<f32>` allocation for the 262,144 vocab logits on CPU.
    - Directly extracts argmax via half comparison on the host-mapped logits buffer slice.
    - Bit-exact output verified ("Paris."). Step latency reduced to 80.8 ms.

14. **SIMD Butterfly Finalize Replacement (`6faaf7f`, `aea8af5`):**
    - Replaced slow vector-stack `finalize_pipeline` with 32-thread SIMD butterfly register kernel (`finalize_simd_pipeline`) across all Ishizuki GEMVs and `fused3`.
    - `down_proj` dropped from 18.95 ms to 14.47 ms (180 GB/s achieved).
    - `o_proj` dropped from 10.72 ms to 6.80 ms (146 GB/s achieved).
    - `qkv_proj` dropped from 14.97 ms to 9.48 ms.
    - Total kernel own time dropped from 85.5 ms down to 71.0 ms per step.

15. **MLP Activation & Input Transform Fusion (`6327c84`):**
    - Implemented `linear_finalize_gated_gelu_and_down_input_transform`: fuses `gate_up` GEMV Hadamard output transform, GELU, and `down_proj` input signs Hadamard transform into a single kernel dispatch.
    - Writes directly to `scratch.transformed`, completely eliminating `scratch.activated` DRAM buffer roundtrips (5.2 MB per token) and eliminating 60 kernel dispatches per token.
    - Dispatches `down_proj` with `pretransformed: true`.

16. **Fused QKV Finalize (`linear_finalize_simd3`):**
    - Combined 3 separate finalize dispatches for Q, K, and V into a single SIMD dispatch (`linear_finalize_simd3`), eliminating 100 kernel dispatches per token.
    - 100% bit-exact greedy decode verified. Total kernel own time dropped to **70.73 ms**.

17. **Current Performance Status (SUPERSEDED — this snapshot is from the 70.73 ms build;**
    **the current build is at 74.01 ms/step and 13.511 tok/s. Kept for the progression.)**
    - Kernel own time: **70.73 ms** (effective 165 GB/s sustained across all 60 layers).
    - Step span median: **84.06 ms** (min: 79.72 ms).
    - Uninstrumented decode throughput: **10.8+ tok/s** (up from 6.0 tok/s baseline, an 80% speedup).
    - Bit-exactness: 100% bit-for-bit exact match on all golden test sentences.
    - **STALE — do not quote the "10.8+ tok/s" or the "165 GB/s".** Both are unpaired
      single readings on a box that drifts 4–12% across unpaired samples (§20.7). The
      figure of record is **13.511 tok/s / 74.01 ms/step / 162.6 GB/s aggregate**, from
      16 interleaved pairs with a paired t of 6.80 on 14 df (`spec.md` §6.4). The
      "80% speedup" is also measured against a 6.0 tok/s baseline that was itself read
      off a 480-dispatch/step build, so the ratio is not a controlled comparison.

---

## 2026-09-30 (later): reconciliation after a branch reset

### What happened

A parallel line of work branched from `0331fd6` and was committed as `e613f60` …
`82c9e7a`. The reset at `HEAD@{14}` discarded `3f78f5c` … `25a585d`, which held the
spec amendment, `bench.rs`, `bench_residency.rs`, `splitk_bit_exact.rs`,
`fused_bit_exact.rs`, `linear_repacked.metal`, and the `_u8` / `_fused2_u8` /
`_fused3_u8` kernels.

The recovered work was worth keeping — it reached **~11.7 tok/s median** — so the approach was
to make the surviving line coherent rather than to restore the discarded one.

### Three things were incoherent

1. **`HEAD` did not compile.** `82c9e7a "WIP 2"` added
   `finalize_simd3_sk2_pipeline` to `LinearKernels` and loaded the function, but never
   built the pipeline nor passed it to the initializer:
   `error[E0063]: missing field 'finalize_simd3_sk2_pipeline'`. Fixed in `a1cdeed`.

2. **The generator hazard was back.** 13 repacked kernels at `linear.metal:7968-9475`,
   inside the region `scripts/gen_exl3_gemv.py` rewrites. Fixed in `df83398`: moved to
   `linear_repacked.metal`, concatenated in `LinearKernels::new`, AC6 now passes.
   Deleted two unreachable kernels (`_repacked_k3`, `_colmajor_pair`) in the same move.

   Second attempt at the note failed instructively: it was first inserted *inside* the
   regenerated region and the very next generator run ate it. The note belongs before
   the banner, which is the generator's preserved zone.

3. **The live kernel was untested.** `repacked_golden.rs` asserted `to_bits()` for
   `_colmajor` and `_colmajor_simd` — and mentioned `contig4` **zero times** — while
   `KAPLUMBA_TRACE_GEMV=1` shows the engine dispatches `_colmajor_contig4`. The
   tolerance relaxations (`diff < 1e-3`, `diff_fp16 < 0.01`) had been applied to `_sk4`
   and `_u4`, whose tests had failed; those are legitimate deviations, but they were
   recorded as passing assertions rather than as a spec deviation.

### Is the speed real? Yes, and now asserted

`crates/engine/tests/live_kernel_bit_exact.rs`, measured:

```
down_proj  21504x5376: f32 differs 5376/5376 (max 26 ULP); f16 differs 0/5376
gate_proj  5376x21504: f32 differs 21504/21504 (max 51 ULP); f16 differs 0/21504
```

`contig4` assigns four *contiguous* tiles to each simdgroup where `_colmajor`
assigns tiles at `tk + {0,SG,2SG,3SG}`. Same tiles, ascending, different accumulators —
a reassociation of the float sum. **The fp16 rounding absorbs it completely**, and
fp16 is what `argmax_logits` reads. The test now asserts that equality rather than
printing it, so a kernel that reassociates too far to round away fails loudly.

### Verified state at HEAD

- Interleaved uninstrumented runs, 64 tokens, **9 samples across two binaries**:
  `11.03 / 11.57 / 11.64 / 12.03 / 12.14 / 12.52 / 12.76 / 12.80 / 12.93`
  → **median 12.14 tok/s**, range 11.03-12.93, **spread 0.85**.

  An earlier, shorter 5-round run gave a median of 12.76 and I quoted that as
  *the* number. It is the high end of this range, not its centre. Spread 0.85 on
  the identical binary is AGENTS.md 10.5's documented drift; no single reading is
  a measurement on this box, and the spec now carries a range, not a point.

- **The kernel move is performance-neutral, verified by A/B.** Pre-move
  (`a1cdeed`) vs post-move (`df83398`) interleaved, 4 rounds: 11.03/11.02,
  11.64/12.16, 12.03/12.28, 12.14/12.24. Identical kernels and identical
  concatenation semantics -- the only change was which file they are read from.
  Median 12.14 vs 11.02 across 9 and 7 samples respectively, both arms spanning
  the same range, so the difference is drift, not a regression.
- Greedy output unchanged on three prompts, temperature 0.
- 140 tests pass.
- AC6 PASS: generator run twice, `git diff --exit-code` clean.

### Corrections that had to be redone

The per-token byte figure. An audit subagent derived 11.639 GB and it entered the first
draft of this spec; the full-attention per-layer sum was mis-added
(`200,245,248`, not `160,245,248`). The correct figure is
**12,039,487,488 B = 12.039 GB/token**. `bench::tests::per_token_stream_is_12_039_487_488_bytes`
was written to fail if the total did not balance, and did. That test is lost with the
reset and is worth restoring.

### What the honest ceiling is

`lm_head` measures ~261 GB/s against a ~257 GB/s streaming control. The wide shapes
are **at the memory system's limit**. A perfect GEMV everywhere would save at most
~5% of the step. 19 tok/s needs 229–266 GB/s of dequantizing GEMV — 91–106% of copy
bandwidth. **15 tok/s is the target; 19 is not reachable by kernel work on this box at
3.0 bpw.**

> **Superseded twice, both times by measurement — the conclusion survives, the numbers
> do not.**
> - The 229–266 GB/s range was computed before the non-GEMV own time was measured
>   properly. With non-GEMV at 7.0–10.1 ms, 19 tok/s demands **264–283 GB/s**, i.e.
>   **105–113%** of a pure copy. See `spec.md` §6.2 for the full table.
> - **"15 tok/s is the target" is now itself unreachable by the plan.** The remaining
>   fusion budget is 0.8 ms, which reaches 13.67 tok/s, and 15.0 needs 7.35 ms. What
>   remains is the 64.6 ms of GEMV, where four mechanisms have failed. **The target
>   number is under review and is the user's call** (`spec.md` R6).

### Next

> **All three items below are DONE or CLOSED. Kept verbatim as the state at that date;
> see the resolution under each one.** Current next steps are at the end of this file.

1. ~~The **~13 ms span-minus-own gap** across ~800 dispatches per token — the largest
   unexplained item, and now comparable to the whole non-GEMV cost.~~
   **RESOLVED AS AN ARITHMETIC ERROR** (see "where the 72.6 ms actually is" below): the
   13 ms differenced step-mode `span` against kernel-mode `own`, and step-mode `own` is
   documented as *not a decode cost*. The replacement "~9 ms" is itself cross-instrument
   and does not survive the current build. Both dispatch counts here are also wrong:
   **1407 measured** (1167 after T7.6.10c), not "~800".
2. ~~Restore `bench.rs` (the real-shape table + median/spread statistics) and
   `bench_residency.rs` (the `MADV_DONTNEED` cold arm). Both were lost to the reset.~~
   **DONE** — both recovered from the reflog and re-pointed at `_contig4`; results in
   the "Recovered from the reset" section below.
3. ~~`q`/`k`/`v` at 88–92 GB/s vs `lm_head` at 261 — settle on the cold instrument before
   building anything on it.~~ **CLOSED** — it does not reproduce on the cold arm
   (`bench_residency`: 121–126 GB/s for `full/kv`). It was the cached-working-set arm of
   `bench_gemv`. `spec.md` §7.4 is marked closed. Do not rebuild on it.

### Recovered from the reset, and what it settled

`bench.rs` (14 tests), `bench_residency.rs`, and the two bit-exactness tests were
recovered from the reflog and re-pointed at the kernels that actually exist. Only the
*addition-only* files were taken — no commit that touched `linear.metal`,
`linear.rs` or `model.rs` was replayed, because that would have clobbered the fusion
work. `bench.rs` needed `pub mod bench;` re-added to `lib.rs`.

**`bench_residency` measured, 7 rounds, three shapes, three backings:**

| shape | kernel | resident | file | file_cold |
|---|---|---:|---:|---:|
| `sliding/q` 5376×8192 | colmajor | 153.2 | 157.6 | 159.5 |
| | contig4 | 167.8 | 165.7 | 166.6 |
| `full/q` 5376×16384 | colmajor | 175.4 | 175.5 | 177.3 |
| | contig4 | 175.9 | 176.4 | 176.1 |
| `full/kv` 5376×2048 | colmajor | 126.1 | 123.8 | 121.4 |
| | contig4 | 124.4 | 125.4 | 126.3 |

**Three findings, and one of them closes a spec open question.**

1. **`MADV_DONTNEED` is the whole variance story.** `sliding/q` / colmajor: spread
   0.379 resident, 0.700 file, **0.930 file_cold**. Cold reading is 1.04x the
   resident *median* and has a third of the spread. Every instrument in this repo
   should evict; the noise was never the machine, it was a warm cache being read as
   bandwidth.

2. **Weight backing makes no difference.** file/resident is 1.03x, 1.00x, 0.98x — i.e.
   within noise. The engine's mmap'd, file-backed weights (`weights.rs:983,1036`) are
   not a handicap versus a resident `new_buffer_with_data`. That hypothesis is dead,
   measured, not argued.

3. **`contig4` is a real but modest win, and only on narrow shapes.** 1.10x resident /
   1.05x file / **1.04x cold** on `sliding/q` (512 tile rows); 1.00x on `full/q`
   (1024 tile rows) and 0.99-1.04x on `full/kv`. So the contiguous-tile change earns
   ~4% where the grid is narrow and nothing where it is wide — consistent with it
   being a latency-hiding effect rather than a bandwidth effect.

**`full/kv` no longer collapses.** It read 57 GB/s in the original `bench_gemv` and 22%
of streaming on the first `bench_residency` run; here it is **121-126 GB/s**. That gap
is now attributed: it was the cached-working-set arm of `bench_gemv`, not a property
of the shape. Spec §7 question 2 is answered — there is no narrow-shape bandwidth
anomaly to chase.

Also fixed: the instrument's summary line still looked up a `_u8` kernel that no longer
exists and printed `0.00x`. Now compares `contig4/colmajor`.

---

## 2026-09-30: where the 72.6 ms actually is

Subagent audit of the kernel-time budget (option (c)), with the load-bearing claims
re-verified by hand afterwards.

### The measurement this replaces

The "~13 ms span-minus-own gap" reported earlier **was an arithmetic error of mine**:
I differenced `span` from **step** mode against `own` from **kernel** mode, and
`decode_profile.rs:36-37` says explicitly that step-mode `own` is "only the two
boundary buffers, so it is small and **not a decode cost**". Correct figures:

| instrument | own | span | wall | tok/s |
|---|---:|---:|---:|---:|
| step (clock, undistorted) | 71.3 * | 73.8 | 71.8 | 10.92 |
| kernel (attribution; wall inflated) | 72.6 | 83.5 | — | 9.16 |
| uninstrumented | — | — | ~81.5 | 12.27 |

<sub>*not a decode cost</sub>

So kernel own time is ~95% of the step and the host gap is ~9 ms, not 13.

> **⚠️ THE "~9 ms" IS ALSO CROSS-INSTRUMENT AND DOES NOT SURVIVE THE CURRENT BUILD.**
> It differences `own` from the **distorted** kernel instrument (whose wall is
> inflated by drains) against `wall` from an **uninstrumented** run. Two different
> builds, two different instruments, and the kernel-mode row is internally inconsistent
> (9.16 tok/s × 83.5 ms = 109 ms, not 83.5).
>
> On the current build the two numbers are `dpk_own_ms_per_token` = **71.644** (§20.2,
> 96 tokens) and T7.6.10c's uninstrumented **74.01 ms/step** — which leaves **2.37 ms**,
> not 9. A 6.5 ms improvement nobody has recorded, from 240 dispatches × 3.5 us = 0.84
> ms plus whatever T7.6.8/9 bought. **Do not carry either number.** The instrument that
> settles it — step mode's `dp_span_over_wall` / `dp_span_minus_wall` — has never once
> been recorded in any file in this repo, and `KAPLUMBA_STEP_TRACE=1` likewise never
> appears in any spec. It costs one extra commit per step without draining. **Run it
> before planning anything against a gap number** (`spec.md` §7.1).

### The 1.333x was real after all — as REQUEST amplification, not DRAM burst

This is the finding that matters, and it inverts the earlier withdrawal. Agent 3 found
that lane-load amplification per tile is:

| rate | lane-loads/tile | distinct words | amplification |
|---|---:|---:|---:|
| 3 bpw | 64 | 24 | **2.667** |
| 6 bpw (`lm_head`) | 96 | 48 | **2.000** |

and therefore

```
predicted 3 bpw ceiling = 250.5 x 2.00/2.667 = 187.9 GB/s
measured gate_up_gelu               = 187.7 GB/s      fit error 0.1%
```

**Verified by hand.** The earlier retirement of the 1.333x was right about *mechanism*
(a 96 B tile does not force a 128 B DRAM fetch; a colmajor column is 32,256 B =
252x128, fully consumed) and wrong about *consequence*: the ratio is real, it just
lives in the request path, not the DRAM path. That also explains why `lm_head` at
6 bpw reaches 250 GB/s while the 3 bpw layers sit at ~188 with no change in bytes.

`lm_head` is **not** colmajor (`model.rs:2829-2848` sets `permutation_mode = 1` only on
the seven per-layer linears) so it takes the row-major `_w2` arm with 16384
threadgroups. It is a fair comparator: same op mix, same `decode_state` body, same
memory system. Verified in code.

### So the ceiling is 187.7, not 250

| | bytes/token | ms now | GB/s | ms at 250 |
|---|---:|---:|---:|---:|
| `gate_up_gelu` | 5.202 GB | 27.71 | 187.7 | 20.81 |
| `down_proj` | 2.601 GB | 14.49 | 179.5 | — |
| `qkv_proj` + `qk_proj` | 2.023 GB | 11.37 | 178.0 | — |
| `o_proj` | 1.156 GB | 6.82 | 169.5 | — |
| `head/gemv` (6 bpw) | 1.057 GB | 4.22 | 250.5 | — |

If every 3 bpw GEMV hit 250 GB/s the GEMV total would fall 64.6 -> 48.2 ms, giving
**14.7-16.1 tok/s** depending on the inter-boundary gap. That is the *entire* remaining
prize in the kernels, and it is bounded by the 2.667x request rate, not by DRAM.

Three attacks on the amplification were already measured and **failed**:
`simd_shuffle` cooperative loads 180.5 -> 146.5; `_u4` 384-byte `uint4` loads 180.5 ->
173.7 on gate_up; threadgroup ring staging 118.8 vs 119.5 (no change).

> **⚠️ The next paragraph was WRITTEN HERE AND HAS SINCE BEEN FALSIFIED TWICE. Both the
> claim and the retraction are kept, because the sequence is the lesson.**
>
> ~~The remaining untried lever is the bit-plane layout at `exl3.rs:687-726`...~~
> **WITHDRAWN — it is lossy, not a layout change.** `extract_ishizuki_lane_codes`
> (`exl3.rs:664-681`) keeps `s & 0x7`, the lane's own 3 bits, while `decode_state_mcg`
> consumes a **16-bit window** of which **13 bits survive** the `0x8fff8fff` mask. The
> single `imul` mixes all 16 bits *before* any masking, so 3 bits of input yields
> **different dequant**, not imprecise dequant. `repack_trellis_3bpw_tensor` is dead code
> with zero callers and **must not be revived as-is**; making it bit-exact means keeping
> the 13 discarded bits, which reconstructs the 24-word layout it was meant to replace.
> It was never on the decode path, so nothing was lost by not building it.
>
> **And the amplification law above is a two-point coincidence.** The 0.1% fit
> (187.9 predicted / 187.7 measured) does not extrapolate: `D = 32`, which halves byte
> amplification 2.667 → 1.333 directly, ran **0.288–0.918× `_contig4` on all 11 real
> shapes**, cold, 9 rounds, interleaved. Mechanism: a tile needs `256/D` lanes, so a
> 32-lane simdgroup covers `D/8` output tiles and the grid becomes `tiles_n / D`;
> `full/kv` has 128 output tiles, so `D=32` launches **4 threadgroups on 20 cores**. The
> per-shape ordering is monotone in `tiles_n` — a starved grid, not a bandwidth wall.
> The load path is not the wall either: the two probe arms fetched identical 128 B and
> differed only in requests (22 vs 32), and the **22-request arm was slower on all 11
> shapes by 2.6–8.1%**. Three points — 187.7, 148.7, 250.5 — and no single law fits them.
>
> **What is actually left: latency and occupancy.** Bytes, requests, ALU (removing decode
> makes the kernel *slower*, −26.7%) and page backing (file/resident 0.98–1.03×) are all
> ruled out by measurement. See `spec.md` §6.6.

### The ALU is not the problem, and that is measured, not argued

Two independent no-decode probes on the real kernel, both showing the *opposite* sign
from "dequant is too expensive":

- `_nod` stub (`linear.metal:8353`, driven by `gemv_shape --decode-probe`):
  3 bpw 167.3 -> **122.7** when decode is removed, i.e. **-26.7%**.
  4 bpw 217.1 -> 206.3 (-5.0%). 8 bpw 222.5 -> 219.2 (-1.5%).
- four-probe sweep on gate_proj: baseline 124.7, **no decode 115.1**, no x-loads
  124.6, **no weight loads 223.4**.

Deleting 59% of the ALU work makes it *slower*. The decode's dependent chains were
accidentally supplying the ILP that hides load latency. "Cheaper dequant" is ruled out
as a fix, twice, on two instruments.

### `gate_up_gelu` has essentially no headroom

It is **99.4% weight streaming**: 5.202 GB of weights against 30.97 MB of everything
else (GEMV writes, finalize reads/writes). Deleting 100% of the non-weight traffic
saves <= 0.155 ms = 0.56% of the bucket. The epilogue
(`linear_finalize_gated_gelu_and_down_input_transform`, `linear.metal:568`) moves 64 B
per thread at ~0.88 FLOP/B — 2.5% of the machine's ridge point — so it is
latency-bound, not bandwidth-bound, and costs ~0.15-0.27 ms/token total.
`scratch.gate`/`scratch.up` are already dead at HEAD (commit `6327c84`); the live
intermediate is `raw`/`raw_b` in **fp32**, 20.6 MB/token, not the 5.2 MB the notes
claimed.

### Dispatch structure: the "~42 buffers/step" is stale — **NOW MEASURED**

Derived census: **1407 dispatches per step** (**1167 after T7.6.10c**), and since
`1407 < max_dispatches(2500)`, `tick()` never fires `roll()` and the step commits
**2 command buffers, one of them empty** — not 42. `MAX_IN_FLIGHT` is never engaged.
The `~2100 dispatches` constant at `encoder.rs:868` and the `42 buffers/step` comments
at `decode_profile.rs:19`, `generate.rs:893` all predate `set_max_dispatches` and the
fused kernels.

**The dispatch count is now a measurement** — T7.6.10a added
`MetalEncoder::dispatches_total` and recorded 1407/token (13.26–13.32 tok/s), then
1167/token (13.511 tok/s) after the fusion. The 2-buffers-one-empty figure itself is
still derived from the call graph; `KAPLUMBA_COUNT_BUFFERS=1` settles it in one
command. The three code comments carrying "~42" have been corrected.

> **Consequence that matters for planning, established in §20.1:** 1407 dispatches ×
> 3.5 us = 4.9 ms of pure launch/bubble latency per step, and the 1167 remaining ×
> 3.5 us = 4.1 ms. That is the *total* remaining fusion prize on this axis — not the
> 5.5 ms the §19.4 roadmap claimed, and not the 0.8 ms that roadmap's surviving items
> are worth. It is also roughly the size of the unattributed host gap in §7.1, which is
> the strongest hint that the two are the same thing.

Two real serialisation points per step, and the host **cannot** run ahead:
`model.rs:3192` commits an empty buffer and waits against a provably idle queue (it
exists for prefill scratch reuse, `model.rs:3183-3191`, and is dead weight for a
1-token chunk); and the terminal barrier, after which the host argmax over 262,144
fp16 must finish before the next step can start. `model.rs:2919-2924` documents why
there is no device-side hand-off, and `model.rs:3378-3391` why the existing device
pick is wrong for this vocabulary (`greedy_key` packs `0xFFFF - index` into 16 bits,
which underflows for every index >= 65,536 — most of a 262,144 vocabulary).

Also found: `generate.rs:867` evaluates `argmax_with_margin(&logits)` **unconditionally**
even though `MarginLog::record` is gated, costing two extra full passes over the
logits row per token on the `generate_gpu_greedy` path. The default path
(`generate_gpu`, `main.rs:99-103`) guards it correctly at `generate.rs:293-299`, so the
measured numbers are unaffected — but `--legacy-path` is materially slower for a reason
that has nothing to do with the GPU.

### What nobody has ever measured

`grep -rln "dp_span_over_wall\|dp_span_minus_wall" spec/` returns **nothing**. The
step-mode report prints exactly the decomposition that would attribute the ~9 ms
(`dp_span` vs `dp_wall` vs their ratio), and it costs one extra commit per step
without draining, so it does not serialise the host. It has never been run and
recorded. `KAPLUMBA_STEP_TRACE=1` likewise never appears in any spec.

---

## 18. T7.6.8: Host Pipeline & Decode Warmup Optimization

### What was found:
1. `KAPLUMBA_STEP_TRACE=1` revealed that step 0 (first decode step) took **1,277 ms** due to cold Metal shader pipeline preparation and compilation on first dispatch. Over a 64-token run, this 1.27s single-step stall added ~20 ms/step across the entire sequence, suppressing reported throughput to ~12.39 tok/s.
2. In `MetalEncoder::flush_and_wait`, empty command buffers (`dispatches == 0 && in_flight.is_empty()`) were committed and waited on by the CPU at the beginning of forward passes, causing redundant GPU scheduler round-trips.
3. In `argmax_logits`, `half::f16::from_bits(logits[best]).to_f32()` was evaluated redundantly 262,143 times inside the loop instead of caching `best_val`.
4. In `generate_gpu`, `model.forward_buffer` was re-allocating and copying a 512 KB `Vec<u16>` on every single decode step.

### Fixes implemented:
- Added `warmup_decode` to `GemmaModel` to prime decode pipelines during `load_from_dir`.
- Short-circuited `flush_and_wait` when no dispatches are pending or in flight.
- Added `forward_buffer_into_vec` to reuse the 512 KB `logits` buffer in-place across all decode steps.
- Cached `best_val` in `argmax_logits`.
- Fixed `u16` bit shift in `pack_integer_windows` (`position / 16` and `15 - (position % 16)`).

### Results:
- Step 0 latency dropped from **1,277 ms to 69.9 ms**.
- TTFT dropped from **3.23s to 1.23s**.
- 64-token decode throughput jumped from **12.39 tok/s to 14.04 tok/s**.
- Golden greedy output verified 100% bit-exact ("The capital of France is Paris.").
 
---
 
## 19. T7.6.9: Split-K Gating, MCG ALU Optimization, and Non-Linear Bottleneck Analysis
 
### 19.1 Split-K vs. Contiguous Layout Isolated Measurements
Using `bench_colmajor.rs` on the physical Apple M4 Pro (20 GPU cores, 273 GB/s peak bandwidth):
- `gate_up` (5376 x 43008):
  - Contiguous 4-tile (`contig4`): **188.2 GB/s** (0.46 ms)
  - Split-K 4: **168.1 GB/s** (0.52 ms)
- `down_proj` (21504 x 5376):
  - Contiguous 4-tile: **177.4 GB/s** (0.24 ms)
  - Split-K 4: **176.7 GB/s** (0.25 ms)
- `q_proj` (5376 x 5376):
  - Contiguous 4-tile: **150.6 GB/s** (0.07 ms)
  - Split-K 4: **140.2 GB/s** (0.08 ms)
- `lm_head` (5376 x 262144): **195.6 GB/s** (2.70 ms)
 
**Root Cause for Split-K Regression on Decode:**
On wide and medium matrices, contiguous 4-tile unrolling (`contig4`) already fills all 20 GPU cores (134 SIMD groups/core). Adding Split-K chops K into smaller chunks with higher boundary overhead and requires extra DRAM roundtrips (`raw_gate`, `raw_up` partial reduction buffers).
**Action:** Gated Split-K behind `KAPLUMBA_USE_SK=1` (disabled by default for decode). Disabling it dropped the full model decode step time from **80.9 ms to 69.8-70.4 ms** (boosting decode speed from **12.35 tok/s to 14.02 tok/s**).
 
### 19.2 ALU Optimization: `decode_state_mcg` Direct Vector Bitcast
In `crates/kernels/src/linear.metal`, `decode_state_mcg` dequantized states via scalar bit shifts and masks (`(value & 0xFFFF) / 65536.0`, `(value >> 16) / 65536.0`). Because states are stored in native FP16 representation, this was replaced with a direct `half2` vector bitcast:
```metal
inline float decode_state_mcg(uint value) {
    half2 h = as_type<half2>(value);
    return float(h.x + h.y);
}
```
This eliminated scalar shifts and masks across 32 dequant calls per loop iteration. Bit-exactness was validated with zero FP16 differences in `live_kernel_bit_exact` and `fused_bit_exact`.
 
### 19.3 Profiler Distortion Warning
`KAPLUMBA_GPU_PROFILE=1` injects 665 command buffer boundaries per step, inflating execution time by 5x due to GPU pipeline flushes and draining. For true hardware execution times, use `KAPLUMBA_DECODE_PROFILE=step` or `KAPLUMBA_STEP_TRACE=1`.
 
### 19.4 The 12.8 ms Non-Linear Tax and Fusion Roadmap to 19+ tok/s
> **⚠️ EVERY ESTIMATE IN THIS BLOCK IS ~5–10× TOO HIGH, AND THE HEADLINE IS UNREACHABLE.**
> Measured in §20.1: `norm_rope_qkv` removed 240 dispatches and saved **0.84 ms**, i.e.
> **3.5 us per dispatch** — not the 3.5–4.5 ms predicted below. A dispatch costs
> **latency, not work**: `dpk_own_ms_per_token` was 71.649 → 71.644 across that change.
>
> | item | claimed here | **measured / re-derived** |
> |---|---:|---:|
> | `norm_rope_qkv` (240 dispatches) | 3.5–4.5 ms | **0.84 ms** ✅ measured |
> | T7.6.11 (60 dispatches) | 2.0 ms | **0.21 ms** |
> | T7.6.12 (120 dispatches) | 2.0 ms | **0.42 ms** |
> | T7.6.13 (60 dispatches) | 1.5 ms | **0.21 ms** |
> | **total** | **9.0 ms** | **~0.8 ms** |
> | projected step | 46.5–50.0 ms (20.0–21.5 tok/s) | **73.2 ms (13.67 tok/s)** |
>
> The original estimates assumed a dispatch's *own kernel time*. That assumption is
> **falsified**: the fused and unfused kernels do the same arithmetic over the same
> bytes and their own time is identical to three decimal places. Only the boundaries
> went away. **Never estimate a fusion from the kernel time it absorbs.**
>
> The 46.5–50.0 ms projection also requires the 64.6 ms of GEMV to move, and it cannot:
> every mechanism tried for the 3 bpw rate has failed (§6.6 above, `spec.md` §6.6).
> **19 tok/s is excluded by arithmetic** — it needs 264–283 GB/s of dequantizing GEMV,
> 105–113% of a pure copy (`spec.md` §6.2). This block is kept verbatim as a record of
> the estimate that was wrong, and as the calibration point for the one that was right.

Dissecting the current 68.2-70.4 ms forward decode step:
- **Linear GEMV Projections:**
  - `gate_up`: 27.6 ms
  - `down`: 14.4 ms
  - `qkv`: 6.6 ms
  - `o_proj`: 4.2 ms
  - `lm_head`: 2.7 ms
  - **Total Linear GEMVs:** 55.5 ms
- **Non-Linear Operations & Pipeline Bubbles:** **12.8 ms**
  - There are currently **22 kernel dispatches per layer x 60 layers = 1,320 kernel dispatches per token**.
  - Small elementwise dispatches (`q_norm`, `k_norm`, `v_norm`, `q_rope`, `k_rope`, `residual`, `layer_scale`, `finalize`) incur GPU dispatch scheduling latency, threadgroup setup bubbles, and intermediate DRAM roundtrips.
 
**Fusion Strategy to Reach <= 52.6 ms/step (>= 19.0 tok/s):**
1. **`norm_rope_qkv`**: Fuse Q, K, V norms and Q, K RoPE rotations into a single 32-threadgroup dispatch per layer. Eliminates 240 dispatches per token and removes intermediate unrotated Q/K DRAM traffic. (Saves ~3.5-4.5 ms).
2. **`linear_finalize_simd_with_residual_scale`**: Fuse down-projection finalize with post-FFN residual addition and layer scale. Eliminates 60 dispatches per token and removes `scratch.proj` DRAM writes. (Saves ~2.0 ms).
3. **Post-Attn Norm & Pre-FFN Gate Input Transform**: Fuse residual addition, pre-FFN RMS norm, and Hadamard gate input transform into a single pass. (Saves ~2.0 ms).
4. **Attention Single-Key Fast Path**: Bypass split reduction pipeline for decode contexts < 64 tokens, saving 60 dispatches. (Saves ~1.5 ms).
- **Target Projected Step Time:** 55.5 ms - 9.0 ms non-linear reduction = **46.5-50.0 ms/step (20.0-21.5 tok/s)**.


 
---

## 20. Dispatch fusion: `norm_rope_qkv` measured (T7.6.10)

### 20.1 HEADLINE: the fusion roadmap's estimates were ~5x too high

The roadmap in §19.4 predicted `norm_rope_qkv` would save 3.5-4.5 ms. **It saved
0.84 ms.** Measured on the same binary, flag A/B, 16 interleaved pairs:

| arm | dispatches/token | median tok/s | median ms/step | spread |
|---|---:|---:|---:|---:|
| unfused (`KAPLUMBA_NO_NORM_ROPE_QKV=1`) | 1407 | 13.320 | 75.08 | 4.5% |
| fused (default) | 1167 | 13.511 | 74.01 | 4.1% |

- B faster in **15/16 pairs**; sign test **p = 0.0005**; paired **t = 6.80** on 14 df.
- **Marginal cost of a dispatch on this box: 3.5 us** (0.84 ms / 240 dispatches).

**Use 3.5 us/dispatch to plan any further fusion.** The remaining roadmap items
(T7.6.11 60 dispatches, T7.6.12 120, T7.6.13 60) are worth **0.21 / 0.42 /
0.21 ms**, not the 2.0 / 2.0 / 1.5 ms each was estimated at. Total remaining
dispatch-fusion budget is **~0.8 ms**, which cannot reach the 52.6 ms target
(19 tok/s) on its own — the target needs the 55.5 ms of GEMV time to move, not
the 12.8 ms of non-linear tax. The §19.4 projection of "46.5-50.0 ms/step" is
**not reachable by fusion** and should not be planned against.

> **Stronger, and the number to plan against: 0.8 ms lands at 13.67 tok/s, which does
> not reach 15.0 either.** 1167 dispatches × 3.5 us = 4.1 ms is the *total* remaining
> launch-latency prize on this axis if every dispatch could be removed; the roadmap's
> three items are 240 of those. 15.0 tok/s needs −7.35 ms. So the gate is not a fusion
> target at all — it is a GEMV target, and §6.6 records that the GEMV has four
> measured failures and no surviving hypothesis. **19 is not merely unreachable by
> fusion; it is excluded outright** (`spec.md` §6.2).

### 20.2 The saving is NOT saved kernel work -- it is bubble

This is the finding that matters for everything downstream. `KAPLUMBA_DECODE_PROFILE=kernel`
over 96 tokens:

| | unfused | fused |
|---|---:|---:|
| `kv_norm_rope` own ms/token | 0.8598 | 0.8622 |
| total `dpk_own_ms_per_token` | 71.649 | 71.644 |

**Kernel own time is unchanged to three decimal places** while the step gets
0.84 ms faster and 240 dispatches disappear. The five kernels and the one kernel
do the same arithmetic over the same bytes; only four launch boundaries per
layer are gone. So:

- A dispatch costs ~3.5 us of *latency* and ~0 us of *work* at this size.
- Fusion cannot be made to pay more by optimising the fused kernel. The lever is
  dispatch count and nothing else.
- The corollary is a trap: anyone who profiles only `dpk_own_ms_per_token` will
  measure this change as exactly zero and conclude it did nothing. It needs the
  end-to-end tok/s or `dpk_span_ms_per_token` to see.

### 20.3 The kernel as handed over was wrong in three ways

Not "unwired" -- wrong, in ways that would have shipped a 10-layer silent
numerical error had it been wired without a test:

1. **`head_dim 512` summed only the first 256 columns.** The kernel read
   `const float val = float(target[base + lid])` with no strided loop. That is
   correct only when `head_dim` equals the threadgroup width. The 50 sliding
   layers (256) would have been fine; the **10 full-attention layers (512) would
   have normalised against a quarter of the head** and produced garbage, with no
   error anywhere. This is exactly the class of bug the `_contig4` precedent in
   §? (`live_kernel_bit_exact.rs`) warns about: a kernel that is right on the
   shape you tested and wrong on the shape you shipped.
2. **The fp16 rounding was skipped.** The unfused path is
   `rms_norm_weighted_tg` writing `half(...)` to a buffer, then `rope` reading
   that half back. The fused version carried the normed value in an fp32
   register across the rotation. `half(lo*cos - hi*sin)` is not a function of the
   twice-rounded form: measured **328 of 1024 elements differ, worst 19 ULP**.
   This is *the* reason the obvious fusion is not bit-exact, and it is invisible
   to any tolerance-based test at "close enough" thresholds.
3. **The unrotated tail copied only the low half**, leaving the high half of
   every tail pair unwritten. Same bug `rope` itself once had, documented at
   `elements.metal:363`.

Fixed by mirroring the reference exactly: strided sum of squares in
`rms_norm_weighted_tg`'s order with its 256-lane tree; `normed_shared[i] =
float(half(normed))`; `threadgroup float normed_shared[512]` so both head dims
work from one kernel; one thread per PAIR (not per column) for both the rotation
and the tail, which is what makes the in-place write race-free.

### 20.4 A test that could not fail, and looked like coverage

The "did the kernel write every column?" test -- poison the output, run, assert no
poison survives -- was **wrong on its first version and passed while sabotaged**.
It uploaded the real input *over* the poison sentinel and then asserted the
sentinel did not survive, which can never happen: the input overwrite had
already removed every one. It reported `ok` with the unrotated tail's high half
being dropped.

The fix is not a stricter assertion, it is a different construction: **make the
INPUT the poison.** Fill every buffer with `0x7bff` and set the weights to 1.0.
A written column comes out near 1.0 (the norm of a constant vector times weight
1.0); an unwritten column is still `0x7bff`. The input does not have to be
realistic -- it has to make written and unwritten distinguishable.

Re-verified: with the tail sabotage in place it now reports
`256 of 1024 columns unwritten, first at column 192, rotated_pairs 64`.

This is the third instance in this repo of the AGENTS.md §3.4 rule earning its
keep (`crates/engine/tests/linear_metal.rs:900` twice, now here). **A test whose
setup makes its assertion vacuous is worse than no test**, because it is read as
coverage by the next person, including in the spec.

### 20.5 Sabotage evidence (all three reverted; kernel byte-identical after)

| sabotage | observed failure |
|---|---|
| drop `float(half(normed))` | 328/1024 elements, worst 19 ULP, first at head 0 col 2 |
| `partial[lid] = (lid==0) ? sum : 0` | 8192/8192 elements, worst 9266 ULP |
| drop the tail's high half | exactly 256 elements at col 192; poison test 256/1024 unwritten |

### 20.6 End-to-end bit-exactness

Full stdout over 128 greedy tokens, both arms, byte-identical:
sha256 `2fff2c970b611f64c4a485d8a3a3dc719a0f03293a3f48c167f4dff5466795b2`,
including the `--show-ids` token list. A single fp16 ULP anywhere in the
60-layer forward changes that list, so this is the no-quality-loss gate.

### 20.7 Operational notes

- **Dispatch counting needed a new counter.** `command_buffers_created()` cannot
  answer "how many dispatches", because under `KAPLUMBA_DECODE_PROFILE` the
  stamped boundaries dominate it: a 48-step profiled run reported **730
  buffers/step against 1407 real dispatches**. `MetalEncoder::dispatches_total`
  is the right number and is now folded per decode step.
- **The tally for the real decode loop must be taken in `generate.rs`.**
  `forward_buffer_into_step` drives the *caller's* encoder, so `model.rs`'s own
  hook never fires for it. And it must be read before `commit_terminal`, which
  consumes the encoder.
- **`KAPLUMBA_DECODE_PROFILE=kernel`, not `=1`.** The parser matches the literal
  strings `step`/`kernel`; `=1` silently means Off and the run looks like it
  worked. Cost me two runs.
- **`scripts/gpuq try -- env VAR=1 ...`** — a bare `VAR=1 scripts/gpuq ...`
  does not propagate the variable through.
- **`cargo fmt --all` reformats files holding other people's uncommitted work.**
  It reflowed 10 files I never touched and 50 lines of the pre-existing
  `live_kernel_bit_exact.rs` work. Reverted with `git checkout` on the clean
  ones and by hand on the dirty one. Use `rustfmt <file>` on what you changed.
- **A/B on this box needs pairing, not median comparison.** 16 unpaired samples
  gave spreads of 4-12%; the paired test resolves a 1% effect at p=0.0005
  because each pair cancels the drift. One pair in 16 is badly contaminated
  (both arms slow) -- report it, do not silently drop it.
- `KAPLUMBA_NO_NORM_ROPE_QKV` is kept, not just as an A/B lever: it is the
  cheapest available regression switch if a future kernel change turns out to
  break bit-exactness in a way the shape tests miss.
- Full suite after this change: **146 passed, 0 failed, 22 ignored** (the
  ignored are the model-loading gates, unchanged). `scripts/gen_exl3_gemv.py`
  run twice leaves an identical diff (AC6 holds; the 139 lines in
  `linear.metal` are the pre-existing `_sk2` work, which the generator does not
  touch -- it lives in `linear_repacked.metal` for exactly that reason).

### 20.8 On the uncommitted `_sk2` split-K kernels, and why the SK4 golden test broke

An earlier revision of this section claimed two things that were **false at
`e9c2785`**:

1. that "`linear.rs` has had their reduce dispatches removed again
   (`gemv_reduce_sk2_pipeline` / `gemv_reduce_sk4_pipeline` are now
   constructed but never dispatched)"; at `e9c2785` both were still
   dispatched, at `linear.rs:1161` and `linear.rs:1189`;
2. that "`live_kernel_bit_exact.rs` still asserts the `_sk2` kernel's fp16
   output against the reference"; at `e9c2785` that file contained no
   `sk2`/`sk4` string at all.

Both were true of the *dirty working tree* at the time and were recorded as
though they were true of HEAD. That is the same failure mode as the
`linear_metal.rs:900` incident in `spec/engine-performance` Q11: a claim
about code that was described from memory instead of from `git show`. Both
are now corrected; the grep commands that establish the real state are in
`tasks.md` T7.6.11.

**The actual root cause of the failing `repacked_golden` test.** The dirty
`_sk2`/`_sk4` work replaced the two-kernel split-K tail (GEMV writes N
partials per K-split into `raw`; `gemv_reduce_sk2`/`_sk4` sums them back
into `raw`; `linear_finalize_simd` applies the butterfly and writes fp16)
with a **fused** reduce+finalize (`linear_finalize_simd_sk2` / `_sk4`: sum
the splits, butterfly, signs, fp16 -> `output`, in one pass).

The GEMV half is unchanged and correct. What changed is the observable
contract of `raw`: under the fused kernel `raw` holds **split 0's partial**,
not the reduced dot product. `repacked_golden.rs` read `raw[i]` and compared
it to the reference, so it saw a partial. Dumping the four partials at
element 0 makes it concrete:

```text
orig -98.65691   raw[0] 41.486683
partials [41.486683, -83.44719, 35.126995, -91.8234]
sum            -98.65691   <- exactly orig
```

The arithmetic was never wrong; the test was reading the wrong buffer layout.

**The first fix for this was also wrong, and it is worth recording why.** `db61003`
changed the assertion to read `raw[split * out_features + i]` and sum the four
slots, reasoning that the kernel writes partials there. That is true of the GEMV, but
`encode_gemv_named` then *owns* the tail, and the tail differs by build:

- at `e9c2785` the tail is `gemv_reduce_sk4`, which writes the **full sum** into
  `raw[0..n]` and leaves the partials in `raw[n..4n]`;
- with the fused `linear_finalize_simd_sk4`, `raw[0..n]` holds **split 0's
  partial** and the rest are partials too.

So `db61003` passes in the shipping tree and **fails at `e9c2785`**, where
`raw[0] + raw[n] + raw[2n] + raw[3n]` is `(-98.657) + (-83.447) + 35.127 +
(-91.823) = -238.80` — the reduced sum *plus* splits 1-3 counted twice. Measured
in a worktree, not guessed:

```text
SK4 reduced-partial mismatch at element 0: orig -98.65691 vs sk4 -238.8005
```

There is no expression over all four slots that is correct under both tails, which is
the real lesson: **`raw` is scratch and its post-dispatch layout is not a contract, so
nothing outside the kernel should assert on it.** The fix is for the test to own the
dispatch — it dispatches `linear_gemv_ishizuki_k3_colmajor_sk4` by name into a
buffer only that dispatch writes, and *then* the partial layout is known and stable
regardless of which tail is compiled in. The shared `encode_gemv_named` entry point
is still exercised, but only through `output`, which is the actual contract.

See the commit that follows `db61003`. Both failure modes were run: the corrected
test passes at `e9c2785` (in-place reduce) and with the fused finalize.

**Sabotage record (AGENTS.md 3.4).** All three failure modes were induced and
observed, then reverted:

| sabotage | caught by | message |
|---|---|---|
| `linear_finalize_simd_sk4` drops `input[3*n+base]` | `repacked_golden.rs:406` | `Finalized SK4 output mismatch at element 0: orig 0xd457 vs sk4 0xcd9b (diff 47.015625)` |
| SK4 partial write flipped split-major -> tile-major (`y[(sk*N)+...]` -> `y[(tn*N)+sk*16+...]`) | `repacked_golden.rs` | `SK4 reduced-partial mismatch at element 0: orig -98.65691 vs sk4 -30.435787 (diff 68.22113)` (caught by the reduced-partial loop; after the dispatch-owning fix, still caught) |
| `linear_finalize_simd_sk2` drops `input[n+base]` | `live_kernel_bit_exact.rs` | `sk2 fp16 output differs from reference in 336 elements` |

Note the first sabotage leaves the *raw* assertion passing, correctly: it
changes the finalize, which does not write `raw`. The two assertions cover
disjoint stages, which is what makes them worth having separately.

**Measured quality of the split-K path** (`live_kernel_bit_exact.rs`,
reference = `linear_gemv_ishizuki_k3_colmajor`):

| shape | `contig4` f32 max ULP | `contig4` fp16 | `contig4_sk2` fp16 |
|---|---|---|---|
| `down_proj  21504x5376` | 26 | 0 / 5376 | **0 / 5376** |
| `gate_proj  5376x21504` | 51 | 0 / 21504 | **0 / 21504** |
| `o_proj     4096x5376` | 833 | 0 / 5376 | **0 / 5376** |

`o_proj` is new (added with the `_sk2` work) and its f32 deviation is 833
ULP -- far larger than the 51 ULP quoted in `spec.md` §6.2, which was
measured before that shape existed. The fp16 is still identical, so the
conclusion in §6.2 stands and is now asserted on a third shape.

Split-K is **not** bit-exact everywhere, though: on the synthetic 512-element
`repacked_golden` case, SK4's fp16 differs from the reference on 2 elements
by one ULP (`0x2cfd`/`0x2cfc`, `0xc6c4`/`0xc6c3`). It keeps the
associativity tolerance that U4 gets. Recording this rather than widening
the tolerance further.

**Left alone, deliberately:**

- `gemv_reduce_sk2_pipeline` / `gemv_reduce_sk4_pipeline` are now dead in
  the dirty tree: the fused finalize replaced them, so they are constructed
  and never dispatched (rustc flags both). They are not *mine* to delete,
  and the tree is being edited concurrently -- see below. Concrete cost if
  anyone wants it: two `new_compute_pipeline_state_with_function` calls per
  `LinearKernels::new()`, i.e. two wasted pipeline compiles per model load.
  The two ~20-line Metal kernels behind them are likewise unreferenced.
- `decode_state_mcg`'s `half2` rewrite is in the dirty tree, but
  `tasks.md` T7.6.9 already records it as a completed result. It is
  semantically equivalent (MSL evaluates `half + half` in half precision,
  so `float(h.x + h.y)` matches the old
  `float(as_type<half>(a) + as_type<half>(b))`), and `repacked_golden`'s
  `to_bits()` equality for `_colmajor` and `_colmajor_simd` covers it --
  but the task log and HEAD disagree about what is committed.
- The gratuitous `pub` on `finalize_simd_pipeline` /
  `finalize_simd_sk2_pipeline` / `finalize_simd_sk4_pipeline`: no consumer
  outside `linear.rs` exists (grep-verified), and `finalize_simd_pipeline`
  was private at HEAD, so this is an unnecessary widening.

**Concurrency hazard, recorded because it cost real time.** Partway through
this investigation another session began editing *this same working tree*:
`crates/kernels/src/attention.rs`, `crates/kernels/src/attention.metal` went
from clean to dirty, and an untracked `crates/kernels/tests/argmax_device.rs`
appeared. Compiling during its save window produced

```text
error[E0063]: missing field `decode_single_pipelines` in initializer of `AttentionKernels`
```

which is a mid-write artefact, not a real defect -- the file was consistent
moments later. This is the concrete failure of AGENTS.md §6 in a shared
tree: a `cargo test` result from this tree is only trustworthy if the
compilation window was not concurrent with someone else's save. The commits
here are therefore scoped to a single file each (`repacked_golden.rs`, then
the two spec documents) and touch no file owned by the other session.

### 20.9 Suite state, and a flaky GPU profiler test

Isolated worktree at `57fa90f` (my commits) with a private `CARGO_TARGET_DIR`:

```text
cargo test --release --workspace
passed=162 failed=0 ignored=29
```

The 29 ignored are the model-loading gates, unchanged in kind from the 146/0/22
recorded at `20.7` -- the delta is the other session's tests and mine.

**`decode_profile_gpu::the_decode_instrument_measures_a_step_and_only_a_step` is
flaky**: three consecutive runs gave pass, pass, fail (7.4 / 7.5 / 7.6 s). It asserts
on a GPU timing window, it does not touch `repacked_golden`, and no commit here
modifies it, so it is pre-existing. Treat a single failure of it as noise and re-run
before investigating; do not "fix" it by widening the window without a paired
measurement, per `20.7`'s note that unpaired A/B on this box has 4-12% spread.

**What blocks `cargo test --release` in the shared working tree** (all from the
concurrent session, none from these commits, as of `57fa90f`):

| symptom | file | nature |
|---|---|---|
| `E0061`/`E0599`/`E0596` in a test target | `crates/kernels/tests/argmax_device.rs` (untracked) | does not compile |
| `error: expected ']'`, so **every** `LinearKernels::new()` panics | `crates/kernels/src/linear_repacked.metal:1641` and `:1775` | truncated kernel signature `[[buffer(0)],` -- a half-finished append |
| `AttentionKernels::new()` fails, so all 4 `attention_metal` tests fail | `crates/kernels/src/attention.rs` | in-progress refactor |
| `E0433 cannot find module or crate pipeline_audit` | `crates/kernels/src/linear.rs:11`, `engine/src/generate.rs`, `engine/src/model.rs` | `lib.rs` not yet declaring `pipeline_audit`/`sweep` |

The Metal one is the nastiest, because it is not a compile error. The `.metal` files
are concatenated into a single shader library at build time, so a truncated kernel
signature surfaces as `LinearKernels::new()` returning `KernelError::Shader` and
**every** kernel test panicking at its own `.expect("LinearKernels")` -- which reads as
"the kernels do not work", not as a typo on line 1641. If the whole Metal test suite
goes red at once, map the concatenated `program_source` line number back to the `.metal`
file before looking at anything else.

---

## T7.6.13: Single-pass decode attention (built, measured, bit-exact, off by default)

The decode path dispatches two kernels per layer (`crates/kernels/src/attention.rs:329` ->
`encode_online_attention_split`): `attention_decode_split_{dim,ratio}` on
`splits * kv_heads` workgroups, then `attention_decode_combine_{dim}` on
`q_heads * head_dim` threads. The alternative is one dispatch holding the whole
reduction in-threadgroup, which is what omlx's `omlx_sdpa_decode` does
(Apache-2.0, `decode_fast/csrc/sdpa_decode_kernels.h:42-110`, single pass,
`simd_sum`, no partials round trip).

Added `attention_decode_single_{256_2,512_8}` — the two model shapes only, since
the host only selects it after `gqa_index` has resolved — behind
`KAPLUMBA_DECODE_SINGLE_PASS`, **default off**. `model.rs` and `generate.rs` are
untouched; the arm is inside the existing `encode_online_attention_split`.

### The crossover is 64 keys, and that is structural, not a coincidence

64 is exactly `ATTENTION_MIN_KEYS_PER_SPLIT`, i.e. exactly where `decode_splits`
already returns 1. So the window where single-pass wins is precisely the window
where the split path is not splitting:

| keys | splits | sliding split+comb / single | full split+comb / single |
|---|---|---|---|
| 16 | 1 | 0.0125 / 0.0109 ms = **1.141x** | 0.0205 / 0.0187 = **1.098x** |
| 64 | 1 | 0.0290 / 0.0274 = **1.059x** | 0.0531 / 0.0514 = **1.034x** |
| 65 | 2 | 0.0206 / 0.0303 = **0.682x** | 0.0360 / 0.0574 = **0.627x** |
| 128 | 2 | 0.0294 / 0.0507 = **0.579x** | 0.0537 / 0.0984 = **0.546x** |
| 256 | 4 | 0.0305 / 0.0976 = **0.313x** | 0.0544 / 0.1930 = **0.282x** |
| 512 | 8 | 0.0384 / 0.1979 = **0.194x** | 0.0601 / 0.3854 = **0.156x** |
| 1024 | 16 | 0.0592 / 0.4673 = **0.127x** | 0.1188 / 0.8253 = **0.144x** |
| 8192 | 16 | — | 1.1568 / 8.1841 = **0.141x** |

GPU-clocked own_ms (`commit_stamped_and_wait` + `take_gpu_stamps`, the same
measurement `KAPLUMBA_GPU_PROFILE=1` reports), median of 9 **interleaved**
samples after 3 warm-ups, one pass per command buffer. Per-sample spread was
0.0002-0.0074 ms on the winning side and up to 0.31 ms on the losing side at
long context, so the 0.13x-0.19x gaps are far outside the noise and the 1.03x-1.14x
gaps are at or just above it.

**Why**: both paths move *exactly the same DRAM bytes*. A workgroup owns one kv
head and its `q_heads / kv_heads` simdgroups read that head's K and V out of one
L1, so a split never duplicates a byte — verified by the byte column being
identical for both arms at every key count. The only terms in the cost model are
(a) one extra dispatch plus a `splits * q_heads * (head_dim + 2)` float round
trip, saved by single-pass, and (b) the whole `splits` factor of occupancy, paid
by it. At `splits == 1` (b) is zero, so (a) is all that is left and it is worth
3-14%. At `splits == 2` (b) is already worth 30-45% and (a) cannot come close.

### Negative result: the sliding window does not make this viable at all times

The hypothesis worth checking was that because the sliding layers cap context at
1024, single-pass might be usable there permanently. It is not, and by a lot:
at the 1024-key cap it runs **7.9x slower** (0.127x). A bounded KV context bounds
the *work*, not the *parallelism requirement* — and the parallelism requirement
is set by `kv_heads`, which is 16 (sliding) and 4 (full). Four workgroups cannot
fill twenty cores however little work each has. This is the same lesson as
T7.6.10's `KAPLUMBA_DECODE_SPLITS` note (64 workgroups quantises badly on 20
cores), arriving from the opposite direction.

### End-to-end value: ~0.025 ms/token, and only for the first 64 tokens

60 dispatches x ~0.4 us saved per dispatch. The roadmap line estimated 1.5 ms
(`tasks.md` T7.6.13), which had already been revised to 0.21 ms; the measured
figure is **~0.025 ms/token**, another ~8x below that. Against a 74.01 ms/step
this is 0.03%, below the 4-12% run-to-run spread AGENTS.md 10.5 records for this
box, so **it is not measurable end to end** and the tok/s figure must not be
quoted as changed.

Left **off by default** for that reason: a 0.03% win confined to the first 64
tokens of a conversation is not worth changing the production decode path for.
`KAPLUMBA_DECODE_SINGLE_PASS=1` turns it on; `=<n>` sweeps the threshold.

### Bit-exactness: exact at `splits == 1`, and *only* there

`crates/engine/tests/attention_single_pass_bit_exact.rs`, 7 tests, all passing:

* `to_bits()` identical on **294912/294912** fp16 elements, both model shapes x
  12 key counts (1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256 — every
  boundary in the dispatch decision, not a sample).
* `splits == 8` is **not** bit-exact: 6/49152 elements differ, worst 1 fp16 ULP,
  worst relative 9.7e-4, and 0 of 4 shape/count combinations happened to be
  exact. The combine's `sum_i exp(m_i - M) * O_i` interleaves its rescale across
  splits in a different association order from the split body's per-tile
  `s = s * ms + simd_sum(w)`. No single-pass kernel can match that, which is why
  the gate is `key_count <= 64` and never wider.
* Exactness at `splits == 1` is structural: it is literally
  `attention_gqa_body<HEAD_DIM, NSG, false>` — the same template the prefill entry
  points instantiate — so there is no second copy of the loop that could drift.
  And the combine, reducing one partial, collapses to `O / l` with
  `exp(m - m) == 1.0f`.

### A `DIVIDE` template parameter was added, measured useless, and removed

The combine's epilogue is a *division* (`acc / s`); the prefill/split body's is a
*reciprocal multiply* (`acc * (1 / s)`). Different f32 operations, so on the
reasoning that exactness required reproducing the operation, a `DIVIDE` template
parameter was added to `attention.metal` to select the division.

**It was unnecessary: 0 of 294912 fp16 elements differ between the two
epilogues.** The fp16 store absorbs it. The parameter was removed rather than
carried as an unjustified code path, and the measurement is recorded as
`the_reciprocal_multiply_is_absorbed_by_the_fp16_store` in the test and as a note
on the template in `attention.metal`, so the next person does not re-derive it.

Two other things that had to be checked rather than assumed, both now asserted:
`exp(0.0f) == 1.0f` exactly on this device (the whole `splits == 1` collapse rests
on it), and a one-element 1-ULP value perturbation being *invisible* — 0 of 8192
outputs move, because that element contributes `w * v` with `w ~ 1/256`. Even 256
1-ULP flips in one value row are invisible; the fail-first arm has to sign-flip a
whole row to move the output. That is a property of the kernel worth knowing: a
one-element value flip is below the fp16 noise floor and must never be used as a
sabotage.

### Fail-first (AGENTS.md 3.4)

Verified by sabotage, not asserted. Injecting a 1-element swap into
`attention_gqa_body` **inside the single-pass macro only** (leaving prefill and
split untouched) made the exactness test fail immediately and diagnosably:

```
NOT bit-exact: sliding 32q/16kv/d256 w1024 @1 keys, element 0/8192:
  split+combine(splits=1) 4.8950195e-1 (0x37d5) vs single-pass 4.9682617e-1 (0x37f3);
  ULP distance 30
```

and `the_reciprocal_multiply_is_absorbed_by_the_fp16_store` failed with it, so the
test detects a wrong kernel rather than comparing two wrong things. Sabotage
reverted and the suite re-run green. Two further arms are permanent:
`the_comparison_detects_a_reduction_order_change` (splits 1 vs 8 on one kernel,
2/24576 elements move) establishes that the compared quantity is
order-sensitive, which is what makes the exactness claim non-vacuous; and the
flipped-input arm above.

### AC6 clean

`scripts/gen_exl3_gemv.py` twice: `git status --short` identical both times, no
`linear*.metal` churn. `attention_metal` (4) + `attention_shapes` (1) +
`kaplumba-kernels --lib` (19) + the new file (7) all pass.

---

## 20. The load-vs-ALU partition of `contig4`, MEASURED (probe, not inference)

`crates/kernels/probe/contig4_load_alu.metal` +
`crates/engine/examples/probe_contig4_load_alu.rs`. Raw output kept at
`crates/kernels/probe/probe-r4.txt` and `probe-final3.txt`.

### The prior "no weight loads" probe was void, and that is where the bad partition came from

`spec/decode-20tok/tasks.md:404` and `implementation_notes.md` §19.2 record
"no weight loads 1.87x, the decode is entirely hidden". That probe was commit
`49814d4`'s `linear_gemv_nold_k3`, and its decode input was a **loop-invariant
64-bit literal**:

```metal
const ulong w0 = 0x9E3779B97F4A7C15ul;   // never varies with tk
v[i] = decode_state(s & 0xffffu, CB);
```

`v[i]` is loop-invariant, so LICM hoists all eight decodes out of the tile loop
and the accumulation degenerates to a multiply by a constant. That instrument
measured *loads gone AND decode gone* and attributed the whole difference to the
loads. The 46%/54% partition was then inferred on top of it. Both are wrong, and
the error is the same class as the two false coverage claims in AGENTS.md §3.4: a
probe whose fast arm was not checked to still contain the work it claimed to
isolate.

### The corrected probe, and why each arm is a literal kernel

Arms are full top-level kernels, each a copy of
`linear_gemv_ishizuki_k3_colmajor_contig4` (`linear_repacked.metal:1101`) with the
lines marked `PROBE-DELTA` changed. The first version used a `template<int ARM>`
with the reduction array passed in as a `threadgroup float *` (MSL forbids a
`threadgroup` variable in a non-kernel function). **Arm (r) came out bitwise
identical to production but 12.6% slower**, i.e. the parameterisation itself
perturbed the schedule, so the literal copies replaced it. Reported numbers are
attributed against T(r), not T(a), so the replica's residual difference cannot
leak into the partition.

    (r) replica         byte-identical control, bitwise-equal to (a)  [FIDELITY gate]
    (b) NO loads, full decode          words := probe_word(seed, tk+q), tk-varying
    (e) NO loads, hoisted decode       same but tile := 0 (the 49814d4 shape)
    (f) (b) + constant shift amount    rad := 48 instead of i*KA+16
    (g) (b) + no decode body           decode_state_mcg -> nod_stub
    (c) loads kept, no decode          nod_stub, real loads   [_nod replication]
    (d) loads kept, 129 KB window      tn -> tn & 3          [miss vs issue]

`(b)` is the arm nobody had run. Its words depend on the induction variable, so
the decode cannot be hoisted -- a compiler invariant, not a hope -- and `(e)` is
the hoisted control, so if `(b)` had been hoisted then T(b) == T(e) and the
partition would come out as a flat contradiction rather than a number.

### Result. gate_up 5376x43008, 1.99 GB working set, 11 interleaved rounds, min

    arm                        min ms   median   med/min   vs (a)
    (a) production               0.4589   0.4687    1.02     +0.0%
    (r) replica (control)        0.4580   0.4678    1.02     -0.2%
    (b) NO loads, full decode    0.4233   0.4339    1.02     -7.8%
    (e) NO loads, hoisted        0.1352   0.1372    1.02    -70.5%
    (f) (b) + constant shift     0.1253   0.1261    1.01    -72.7%
    (g) (b) + no decode body     0.5818   0.5892    1.01    +26.8%
    (c) loads kept, no decode    0.5885   0.5965    1.01    +28.2%
    (d) loads kept, 129 KB       0.4554   0.4661    1.02     -0.8%

    decode total = T(b)-T(e)     +0.2882 ms   +62.9% of T(r)
    load  total = T(r)-T(b)      +0.0347 ms   + 7.6% of T(r)
    neither (floor + x loads)     0.1352 ms   +29.5% of T(r)

down 21504x5376 reproduces it: decode 63.9% / load 8.5% / neither 27.6%, worst
median/min 1.02-1.04x across three independent runs. **The partition is additive
to within the spread** (63 + 8 + 30 = 101), which is the check that the three arms
are measuring separable costs.

### Conclusion: ALU-bound, ~8:1, and the ALU is the 64-bit barrel shift

**The ~46% load / 54% ALU hypothesis is wrong. The measured partition is ~8%
load-issue and ~63% decode-ALU.** Removing *every* weight load buys 7.8%; removing
the dequant arithmetic would have bought 63%.

The reason the `_nod` probe looked like it said otherwise is now explicit and it is
not mysterious: removing `decode_state_mcg` **destroys the ILP** that hides the
barrel-shift latency. Arm (g)/(c) measure that directly -- keeping the loads and
the variable shift but stubbing the decode body is **27% SLOWER**, reproducing
`_nod`'s sign exactly, and arm (f) isolates the cause:

    T(f) = (b) with the shift amount pinned to a constant = 0.1253 ms
    T(b) = (b) as production                           = 0.4233 ms

So `decode_state_mcg`'s mul/xor/half2-add body costs **-2.3%** (i.e. nothing;
`T(f) <= T(e)`, it is free and slightly negative), and **65.6% of the kernel is
`uint(A >> (64u - rad))` with `rad` varying per element** -- a 64-bit barrel shift
executed 32 times per 4-tile iteration. `rad` unrolls to 16,19,22,...,37, and a
64-bit variable shift is a shift of each half plus a select on a 32-bit ALU.

Also measured, and against the "the loads are the problem" reading:

- **(d) is +0.6% vs production.** Same load count, same address arithmetic, but a
  129 KB L2-resident window instead of 2.6 GB of DRAM. **DRAM/L2 miss cost is
  ~1% of the kernel.** At 190 GB/s the kernel is nowhere near a bandwidth limit;
  the "~250 GB/s ceiling" framing does not apply to this loop, because the loop
  is not waiting on memory. The 187.7 GB/s is not a bandwidth shortfall at all.
- **(c) replicates `_nod`** on `contig4`: +28.2%. Consistent with the 167.3 ->
  122.7 recorded for `k3`.

### What this redirects

1. The `KAPLUMBA_GEMV_MAX_GROUPS` / Split-K / 4 bpw "fewer lane-loads per byte"
   line of work attacks ~8% of the kernel. The 4 bpw speed-up is therefore NOT
   evidence of load-issue boundedness -- 4 bpw has *more* bytes and is faster
   precisely because `rad = i*KA+16` has fewer elements to barrel-shift, i.e.
   it is a decode-width effect, not a load effect.
2. The lever is making `rad` compile-time, i.e. a layout in which each lane's
   codes sit at a fixed bit offset. `crates/engine/src/exl3.rs:687`
   `extract_ishizuki_lane_codes` / `repack_trellis_3bpw_tensor` was reaching for
   exactly this and was reverted (`9c85e9a`). Arm (f) is the upper bound on that
   idea: **if `rad` can be made constant, the kernel goes 0.459 -> 0.125 ms, i.e.
   3.6x, and 690 GB/s equivalent.** That is the number to beat. It is not a
   cheaper decode -- there is nothing cheaper to find; `decode_state_mcg` is
   already free.
3. `probe_contig4_alufloor` at 0.135 ms is the floor for this loop structure:
   loop, x loads, 64-bit shifts, mask, 32 FMAs, reduction. 29.5% of production.
   Any fix has to beat that too.

### Reproduce

```bash
W=$(mktemp -d)/wt && git worktree add "$W" HEAD          # or e9c2785
mkdir -p "$W/crates/kernels/probe"
cp crates/kernels/probe/contig4_load_alu.metal "$W/crates/kernels/probe/"
cp crates/engine/examples/probe_contig4_load_alu.rs "$W/crates/engine/examples/"
cd "$W" && CARGO_TARGET_DIR=$(mktemp -d)/t \
  cargo build --release --example probe_contig4_load_alu
scripts/gpuq try -- "$CARGO_TARGET_DIR/release/examples/probe_contig4_load_alu" \
  --rounds 11 --iters 15 --bytes 2000000000
```

Use a worktree with its own `CARGO_TARGET_DIR`: the main tree's engine crate did
not compile during this work (another agent's `pipeline_audit` wiring was
mid-flight) and a shared `target/` can rebuild the binary under another agent.

### Method notes, including a measurement bug of mine

- **Estimator is min over interleaved rounds, not median.** `gpuq` excludes the
  *engine's* flock, not a benchmark harness, so this probe competes with other
  agents' `kaplumba` processes and loses rounds. Contention can only add time to a
  GPU kernel, so min estimates the uncontended kernel. An early median-based run
  was 250% off on `down` purely from contention and would have produced a
  completely wrong partition; the median/min ratio is printed per run so this is
  visible rather than silent. Under contention, gate_up medians were within 2% of
  mins; `down` medians were 10-40% high.
- **No Metal Toolchain on this box** (`xcrun metal` -> "cannot execute tool"), so
  there is no AIR/IR check. Validation is instead (i) FIDELITY: (r) bitwise-equal
  to production on every shape, or the harness panics before reporting a time;
  (ii) DCE GATE: perturbing the seeds must change (b)'s output and it must differ
  from production's, or the harness panics; (iii) a CPU model of (b)'s exact
  semantics on a tiny shape (K=512, N=16, one unrolled iteration per simdgroup)
  agreeing to **1.6e-7** max relative error, with `decode_state_mcg` reimplemented
  on the host including the fp16 `h.x + h.y` rounding. 1.6e-7 is fp32 noise; a
  hoisted or eliminated decode would disagree by O(1).
- The harness compiles its **own** library (preamble + `include_str!` of the
  unmodified `linear_repacked.metal` + the probe file), so production (a) is
  measured in the same compiler run as every arm, and the probe ships in nothing:
  `crates/kernels/src/linear.rs:23` concatenates only `linear.metal` and
  `linear_repacked.metal`.

---

## Pipeline-compilation audit (measurement-correctness, not perf)

**Question.** A cold kernel inside a timed A/B measures Metal shader compilation,
not the kernel. `warmup_decode` (T7.6.8, `b35a779`) runs one decode step at load
to kill a 1277 ms first-step JIT, but nobody had verified that *all* kernels
reachable in the decode path are compiled before the timed loop. omlx names
load-time template specialisation as its #1 CPU-overhead fix
(`reference/omlx/benchmarks/bonsai_decode_bench.py:466-470`) and ishizuki
precompiles every specialisation at load, so the hazard is real in principle.

**Answer, measured.** There is no gap, and now there is a way to keep it that way.

### 1. Everything is already eager. `LinearKernels::new` builds 127 of them.

`LinearKernels::new` (`crates/kernels/src/linear.rs`) compiles one library
(`combined_source()` = `linear.metal` + `linear_repacked.metal`, so the two share
one compilation and one `decode_state`) and then builds a pipeline for:

| construction site | count | note |
|---|---:|---|
| `linear_input_transform` | 1 | |
| every `linear_gemv_ishizuki_*` in the library | 31 | via `ishizuki_kernel_names`, sorted so build order is stable across edits |
| every `(codebook, aligned, bits)` EXL3 slot | 48 | 3 codebooks x 2 alignments x 8 rates; the mul1 family re-requests the plain names, so 48 builds over 32 kernels |
| every NR0 variant `_mcg_r{2,4,8}` | 24 | A/B-only, built so a bad name is a startup error |
| `gemv_reduce_sk2` / `gemv_reduce_sk4` | 2 | **never dispatched by the engine** (see 3) |
| `linear_gemv_fast1..8` | 8 | reached only if ishizuki *and* EXL3 both decline |
| fixed fields | 13 | incl. `linear_gemm_mma`, `linear_gemm_mma_regpipe`, `linear_gemv_tiled` |
| **total** | **127** | over a library of 111 kernels |

`ElementKernels::new` builds 21 and `AttentionKernels::new` builds 25 (7 base +
8 `online_attention_gqa_*` + 8 `attention_decode_split_*` + 2
`attention_decode_combine_*`), all inside their own `OnceLock`. `shared()` on
each family means the whole set is constructed once per process, inside load.

### 2. `repacked_gemv_by_name` is dead, and the repacked kernels are built anyway

`repacked_kernel_names` filters for `linear_gemv_repacked_`. The only occurrence
of that string in the tree is a **comment** in `linear_repacked.metal:47`; every
kernel that file defines is named `linear_gemv_ishizuki_k3_*` (`_colmajor`,
`_colmajor_simd`, `_fused2`, `_fused2_sk2`, `_fused3`, `_fused3_sk2`,
`_colmajor_stream`, `_colmajor_sk4`, `_colmajor_contig4`, `_contig4_sk2`,
`_contig4_sk4`, `_u4`). So the map is **always empty** and every repacked kernel
is collected by `ishizuki_kernel_names` instead. The field survives only because
`encode_gemv`'s `trace_gemv` chains it. Documented at the function; not removed,
because removing it is a change to a tracing path and not this task's business.

### 3. What is and is not decode-reachable (50 sliding + 10 full layers, short context, temperature 0)

| candidate | reachable? | why |
|---|---|---|
| `_sk2` / `_sk4` ishizuki split-K | **no** | only under `KAPLUMBA_USE_SK`, and it also requires `perm_mode == 1 \|\| KAPLUMBA_REPACKED_GEMV` |
| `linear_gemv_ishizuki_k3_colmajor_contig4_sk2` / `_sk4` | **no** | same gate, and this checkpoint is not colmajor |
| `gemv_reduce_sk2` / `gemv_reduce_sk4` | **no** | dispatched only from `encode_ishizuki_gemv_named`, an A/B entry point the engine never calls. Built, never used. |
| `_u4` (`linear_gemv_ishizuki_k3_colmajor_u4`) | **no** | name-addressable only |
| `_colmajor_simd`, `_colmajor_stream`, `_fused3*` | **no** | name-addressable only |
| `linear_finalize_simd_sk{2,4}`, `..._gated_gelu_simd_sk2`, `..._and_down_input_transform_sk2` | **no** | no dispatch site in the crate at all |
| `linear_finalize_simd3_sk2`, `linear_zero_raw` | **no** | no dispatch site; the compiler says "never read" |
| `attention_dim_256` / `_512` | **no** | no dispatch site (the GQA kernels took over) |
| 6 bpw `lm_head` | **yes** | `validate_current_graph` pins 6 bpw; trellis starts at `packed_offset % 4 == 2`, so it takes the **unaligned EXL3 arm** `linear_gemv_exl3_k6_u_mcg`, not the ishizuki one |
| `linear_gemv_ishizuki_k3_w2` | **yes** | the live decode matvec for all 420 decoder linears: 3 bpw, 4-byte aligned, `perm_mode == 0`, so the 2-word window is selected by name lookup |
| split-KV attention | **yes**, two specialisations | `gqa_index(head_dim, q_heads/kv_heads)`: sliding is 32/16 at 256 -> `attention_decode_split_256_2`, full is 32/4 at 512 -> `attention_decode_split_512_8`, plus `attention_decode_combine_256` / `_512` |
| probe kernels | **no** | `spin_burn` is the only one, it is a profiler, and it is built at load regardless |

**Six pipelines are built at load and never dispatched**: `gemv_reduce_sk2`,
`gemv_reduce_sk4`, `linear_finalize_simd3_sk2`, `linear_zero_raw`,
`attention_dim_256`, `attention_dim_512`. Pure load cost, zero decode relevance.
Worth knowing before anyone reads a build count as a decode-kernel count.

### 4. The measurement

New module `crates/kernels/src/pipeline_audit.rs`. Every pipeline construction in
the crate now goes through `pipeline_audit::build`, which times it and records the
name; a test greps the four source files and fails if any construction site
bypasses it. Real 614-token greedy run, 48 decode steps,
`KAPLUMBA_PIPELINE_AUDIT=1 KAPLUMBA_ASSERT_NO_LATE_PIPELINES=1`:

```
at=after-warmup  builds=173 build_ms=10.3 distinct=157 armed=173 late_builds=0 late_build_ms=0.0
at=after-decode  builds=173 build_ms=10.3 distinct=157 armed=173 late_builds=0 late_build_ms=0.0
```

**173 pipeline builds, 157 distinct kernels, 10.3 ms total -- all at load, and
`late_builds=0` across the prefill and every decode step.** Nothing is compiled
inside a timed loop. The largest single build is `embedding_scaled` at 1.3 ms, so
the ceiling on the error a missed warmup could inject into one step is ~1.3 ms,
not the 10.3 ms of the whole set.

`LinearKernels::shared()` being a `OnceLock` is what makes this true: the whole
library compiles once per process and no dispatch path can reach
`newComputePipelineStateWithFunction` at all.

### 5. The instrument can fail (AGENTS.md 3.4)

- `the_late_pipeline_assert_can_fail` (`#[should_panic]`) builds a pipeline after
  a mark and calls the real `check_no_builds_since(mark, true)`. Observed panic
  message: `KAPLUMBA_ASSERT_NO_LATE_PIPELINES: 1 pipeline(s) were compiled after
  load (56 us total, ...)`.
- `every_pipeline_construction_goes_through_the_audit` was **sabotaged** by putting
  a raw `new_compute_pipeline_state_with_function` back into
  `attention.rs:323`. Observed failure: `attention.rs:323 constructs a pipeline
  outside pipeline_audit::build: device.new_compute_pipeline_state_with_function(&f).ok();`
- `decode_shaped_dispatches_compile_nothing`
  (`crates/engine/tests/pipeline_warmup_coverage.rs`) was **sabotaged** by moving
  `linear_finalize_simd`'s build out of `new()` into a `OnceLock` built on first
  dispatch -- exactly the regression. Observed failure:
  `decode-shaped dispatches compiled 1 pipeline(s) after load: [("linear_finalize_simd", 56)]`.
  Reverted; the file is byte-identical to before the sabotage.

### 6. The prefill warmup step: measured, and it buys nothing

`warmup_decode` ran one decode step, which cannot reach the multi-query kernels
(`linear_input_rows`, `linear_gemm_mma`/`_regpipe`, `linear_finalize_rows`,
`online_attention_gqa_*`) because the gate is `tokens == 1`. A second two-query
forward was added ahead of it to close that, and then **measured and reverted**.
Interleaved A/B, one binary, 614-token prompt, `--max-new-tokens 2`:

| arm | load_ms (median) | ttft_ms (median) | prefill_tok_s (median) |
|---|---:|---:|---:|
| with the prefill step | 6010 | 6274 | 97.90 |
| without | 5687 | 6249 | 98.27 |

Two results, and the first is the one worth keeping:

- **It is not additive.** The two-query forward takes ~3.0 s, which looks like a
  3 s addition to load and is not: with it the warmup totals ~3.1 s, without it
  the total is also ~3.0 s -- because the *decode* step is then what costs ~3.0 s.
  The prefill step warms the weight path (page-in, residency, the MMA weight
  dequant code) that the one-query step would otherwise warm expensively. Both
  arms pay ~3 s once; the prefill step only changes *which* forward pays it.
- **It buys nothing on TTFT**, the only metric it could have bought anything on.
  `prefill_tok_s` is *higher* without it, so the prefill kernels' first dispatch
  is not a cost worth pre-paying: a 614-token prefill does not feel whatever the
  3 s is.

**Consequence for the rest of this spec: the ~3 s of first-forward cost is
NOT prefill-kernel first-dispatch, and warming the prefill path does not remove
it. Do not re-derive it from a decode-only warmup.** T7.6.8's 1277 ms and this
~3 s are the same phenomenon at different prompt sizes, and neither is
compilation (`late_builds=0`).

### 7. `--metrics` adds no per-token readback or synchronisation

`--metrics` is read in `crates/cli/src/main.rs:109`, **after** `generate_gpu`
returns. Its only work is `measure_rss()` (one `getrusage`) and
`format_metrics`, which is string formatting over the `GenerationResult` the loop
already produced. Nothing inside the loop branches on it, so every tok/s quoted
with `--metrics` is the same tok/s the uninstrumented run produces.

That is the answer to the question as asked, and it is worth stating plainly
because the *adjacent* thing is real. The greedy decode loop has per-token host
work that is **not** gated by anything:

- `logits = model.last_logits()` allocates and copies the 262,144-entry row
  (512 KiB) every step. Measured **0.0066 ms/token**. Necessary -- the argmax is
  on the host -- and it sits *outside* `step_ms`, so it does not enter
  `decode_tok_s`.
- In `generate_gpu_greedy` (the `--legacy-path` route only; the CLI's default
  `generate_gpu` gates this correctly behind `track_margin`),
  `margins.record(step, argmax_with_margin(&logits))` evaluates its argument
  **unconditionally** -- `MarginLog::record` checks `enabled` inside, so the
  value is computed anyway. That is two extra full passes over 262,144 fp16 per
  token. Measured: `argmax_logits` 0.2804 ms/pass, `argmax_with_margin`
  0.4328 ms/call, i.e. **0.152 ms/token of waste**, 0.21% of a 74 ms step. Small,
  but it is pure waste and `generate_gpu` already shows the one-line fix
  (`if track_margin { ... }`).

The instruments that *do* tax a run, none of which is `--metrics`:
`KAPLUMBA_DECODE_PROFILE=kernel` (a `wait_until_completed` every `DRAIN_EVERY`
boundaries, so its wall times are inflated and its own_ms shares are not),
`KAPLUMBA_DECODE_PROFILE=step` (one extra commit per step), `KAPLUMBA_GPU_PROFILE`
(prefill only unless `KAPLUMBA_GPU_PROFILE_DECODE`), and `KAPLUMBA_MARGIN` /
`KAPLUMBA_SEGMENTS` / `KAPLUMBA_COUNT_BUFFERS` (host bookkeeping).

### 8. Two process facts that cost time here and are worth writing down

- **A concurrent session was editing this tree throughout.** HEAD moved twice
  under me (`e9c2785` -> `9cd57a7` -> `379ae9e`) and `linear.metal`,
  `linear_repacked.metal`, `attention.metal` and `model.rs` all gained
  uncommitted work, including a `_bench` kernel in `linear_repacked.metal` with a
  syntax error (`[[buffer(0)],` -- missing `]`) that makes **every** library
  compile fail in the shared tree. All measurement here was therefore done in a
  worktree at a pinned HEAD with its own `CARGO_TARGET_DIR`, per AGENTS.md
  10.2.4.
- **The other session's `git add -A` swept up my `generate.rs` change** into
  `9cd57a7`. So part of this work is already committed under someone else's
  message. Check before assuming a line is uncommitted.
