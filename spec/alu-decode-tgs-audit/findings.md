# Findings — pinned GitHub branch, 2026-10-14

## Decision from the small GPU test

**Worth a narrow full-model trial: scalar 32-bit extraction in QKV split-K.**
The settled model-free `fused2_sk2` comparison found approximately **3.6–3.7%**
median paired kernel-throughput improvement on two larger shapes, with zero
raw FP32 output differences on the deterministic synthetic fixture. All 16
paired comparisons favored the candidate on each larger shape. The tiny tail
shape gave no persuasive improvement. This does **not** establish application
tg/s, checkpoint-wide correctness, long-context acceptance or 250 GB/s.

| K / N_a / N_b | Baseline median ms | Scalar candidate median ms | Median paired gain | Paired spread | Positive pairs |
|---|---:|---:|---:|---:|---:|
| 5376 / 8192 / 4096 | 0.139664 | 0.134682 | +3.702% | +2.255 to +14.916% | 16/16 |
| 5376 / 16384 / 2048 | 0.206253 | 0.199279 | +3.636% | +2.202 to +4.429% | 16/16 |
| 592 / 256 / 128 | 0.003505 | 0.003500 | +0.533% | -7.893 to +10.784% | 9/16 |

The gain column is the median of paired `A_ms/B_ms - 1`, not a ratio of
unpaired medians. A/A noise-control medians were +0.006%, -0.040% and -0.643%;
their ranges were [-1.160,+5.031], [-11.392,+3.476], [-7.317,+5.721] percent.
Outliers remain in the record. A previous shorter-warmup run gave +3.784% and
+3.768% on the larger shapes, but showed clear clock ramp on the first shape.
It is preserved, not substituted for the settled run.

The GPU probe compiles the exact pinned fused2 split-K shader body plus a
candidate; scalar state preparation changes, original scalar MCG and FP
accumulation expressions remain. Data are seeded random ring words and finite
FP16 activations, not model weights. The outputs are the raw split-K partials,
before signs/finalization. Tested 24,576 + 36,864 + 768 = **62,208** raw FP32 words
with zero candidate differences. An actual shader mutation adding 1 to acc0
produced 12,288, 18,432 and 192 unequal outputs respectively. These checks
exercise main and tail paths; they are not an exhaustive model correctness gate.

Important failed variant: direct use of the existing **uint4/vector MCG helper**
changed **6,607 / 24,576** raw FP32 words on the first shape and stopped before
timing. Its shader sabotage changed 18,895 outputs. The root cause of those
numeric differences is not established; source-level algebra did not preserve
the compiler's scalar numerical behavior. Do not ship that substitution or
relax the raw-bit gate to make it pass. The scalar-only variant passed instead.

Measurement limitations: same buffers repeatedly warmed, 8 dispatches per timed
command buffer, 128 alternating warmup batches, 16 ABBA comparison blocks and
16 A/A control blocks per shape. Metal `gpuStartTime/gpuEndTime`, command-buffer
success checked. Metal fast math enabled in the standalone library; full
production compiler options/library context remain a future gate. This is
kernel GPU duration, not a physical DRAM measurement or an application timer.
Host memory was pressured (0.1–0.4 GiB top-unused during probe runs), though the
small synthetic buffers fit and the probe acquired the engine's lock file
nonblocking. No full checkpoint was loaded. Repeat in healthy memory conditions
and the actual concatenated production library before an implementation decision.

## What the supplied Gemini conversation misses at this revision

Source evidence is from the fetched GitHub commit, accessible through the
[pinned tree](https://github.com/zayutaha/kaplumba/tree/7ddf20f901ae7ad4d1e7342ca71677e266bbe847).
GitHub web browsing failed; `git ls-remote`, `git fetch` and `git archive`
provided the actual source. Local HEAD initially was `819a551`; local edits
were excluded from the audit.

1. The local checkpoint config is **60-layer dense**, hidden size 5376,
   intermediate size 21504, EXL3 3.0 bpw and 6-bit head. The original 30-layer,
   K=2816, affine-nibble/MoE scenario is another model/format.
2. `linear_repacked.metal:53–63,2286–2386` already uses 32-bit eight-state
   extraction, vector MCG transforms, loop-invariant indices/shifts and immediate
   decoded-array reuse. Helper unification is not a new performance lever.
3. `contig4` means four contiguous K **tiles**, not four states. Each lane
   reconstructs eight states per tile. The actual start is
   `((lane*8+1)*3+768-16)%768`; shifts are **{3,11,19,27}** and eight states
   span **37 bits**. The pasted 25-bit/four-state replacement is incomplete.
4. Lane 0 starts at word 23, shift 19 and wraps to word 0. Ring wrap is not
   confined to the imaginary final-column schedule in Gemini's explanation.
5. The MCG function remains multiply/mask/xor/half2/half-add/float. Byte sum plus
   affine reconstruction describes a different codebook.
6. `fused2_sk2:1349` and `fused3_sk2:1890` still contain ulong windows; pipeline
   selection in `linear.rs:2738,3120` makes QKV split-K default when available.
   O/down use their own selected split-K kernels (`model.rs:2085,2386`). A test
   of `_contig4` alone does not establish full-route coverage.
7. `trace_gemv` (`linear.rs:2292`) prints each name once. It does not log
   dispatch frequency, dimensions, or cumulative GPU time as Gemini assumes.
8. Norm/RoPE/cache fusion (`model.rs:1781`), greedy device selection/pipelined
   handoff (`generate.rs:977`), and decode command-buffer rolling already exist.
9. `#pragma unroll` plus constant K already appears in baseline split-K loops.
   Do not infer a *variable* 64-bit per-state shift in the compiled kernel from
   this source loop. Profiling/compiler inspection must establish instruction cost.

## Independent CPU state proof and negative controls

An external Rust test compares the current eight-state formulas with an
independent MSB-first reader of individual ring bits. It sweeps each of 768
ring basis bits across all 32 lanes, six uniform patterns and 4,096 seeded
random tiles: **1,246,720 state comparisons**, zero differences. Both functions
are linear bit selectors over the ring, so the basis sweep establishes state
equivalence for all ring contents at these reachable lane offsets. It does not
prove GPU compilation, half arithmetic, or GEMV accumulation correctness.

Mutation failures (separate executions, exit 1):

- Zero wrapped successor: `basis bit 0 lane=0 state=0 expected=0004 got=0000`.
- LSB-direction extraction: same diagnosable failure at basis bit 0.
- Omit state 7: `basis bit 8 lane=0 state=7 expected=8000 got=0000`.

Source, commands and raw GPU logs are preserved in implementation_notes.md.

## tg/s priorities after the QKV experiment

| Priority | Candidate | Why examine it / what is still unknown |
|---|---|---|
| 1 | Scalar 32-bit extraction in active QKV split-K | Newly measured small kernel win; fused3 and full-model behavior untested. |
| 2 | Gate/up and down register lifetimes/load scheduling | Historical ablations rank these as large costs; already-unrolled families need a new isolated experiment, not wholesale unroll. |
| 3 | Positive-temperature CPU sampling/readback/allocation | `generate_gpu` still converts FP16 logits into a newly allocated FP32 vector and samples on CPU. Profile exact cost before proposing device sampling. |
| 4 | Remaining encoding/handoff gaps | Greedy pipelining already exists; independent wall timing must audit startup, callbacks, EOS and instrumentation route differences. |
| 5 | Long-context attention and residency | Attention can dominate at 8k/16k; exact scheduling and cache behavior eligible, reassociated splits not silently acceptable. Verify current shared-mmap fix in live chat. |
| Conditional | Shift classes and lossless ring-load sharing | Could reduce instruction count, but divergent specialization/shuffles can reduce occupancy or lose to the baseline. No promised gain. |

Historical `spec/alu-boost/findings.md` reports +5–9% on a synthetic four-deep
unroll but explicitly leaves its shipping-kernel trial unfinished. At this
GitHub revision multiple actual families already contain unrolling. Similarly,
`MEMORY.md`'s 63% ALU / 8% loads ablation and the newer alu-boost probe's ~11%
extraction / ~16% ring penalties are different experiments. Neither is a
hardware-counter measurement of this newly tested candidate. Ablation timings
are non-additive when they change instruction overlap, registers or cache state.
Do not turn these numbers into a guaranteed application speedup.

Do not revive lossy 3-bit-state-only repacking, half accumulations, previously
slower 4-column/cooperative-load variants, or attention split reassociation
without new evidence and the stated quality gates. Current spec history records
128-token gates that missed long-context token divergence.

## Authorized T3/T4 implementation trial on local branch tip 95969c5

After the user authorized implementation, scalar extraction was applied to the
active `fused2_sk2` and `fused3_sk2` kernels. The tests compared the exact
candidate body against the pre-478c 64-bit shader body, then injected an
output-changing mutation into the actual candidate before recording timings.
These are a new harness/run set relative to the earlier exploratory fused2
figures above; preserve both records rather than treat their timing medians as
one sample. Complete command output, paired values, harness and limitations are
in `implementation_notes.md` and `harnesses/raw_compare3.swift`.

The final `fused2_sk2` run matched 62,208 raw FP32 words over three shapes and
its mutation changed 12,288 / 18,432 / 192. Paired median kernel gains were
+4.621% and +4.321% on larger shapes (16/16 and 14/16 wins), and +0.531% on the
small shape (11/16 wins). `fused3_sk2` matched 32,768 / 40,960 / 1,024 raw words;
its mutation changed 16,384 / 20,480 / 256. Paired median gains were +4.197%
and +4.767% (12/16 and 14/16 wins) on larger shapes, and +1.340% (13/16) on the
small shape, whose spread included a large outlier. These timings indicate the
kernel experiment is promising, but are synthetic repeated-dispatch times, not
full-model decode or DRAM rates. `cargo test -p kaplumba-engine --test
fused_bit_exact -- --nocapture` passed 4/4 (34.96 s) on the combined edits.

The change remains experimental: no Gxma chat token-ID comparison or
end-to-end `tg/s` run was possible because `scripts/gpuq status` reported under
1 GiB top-unused with about 3.3 GiB swap, below the model-run threshold. Thus
the objective to improve application `tg/s`, and 250 useful GB/s, remain
unproven.

## The 250 GB/s target and tg/s arithmetic

Historical useful-weight accounting is **12.039487488 GB/token**, not 2.164 GB.
It must be re-audited from the actual checkpoint before acceptance. With that W:

`effective application GB/s = 12.039487488 * application tg/s`.

- 250 GB/s requires **20.765 tg/s**, or **48.158 ms/token** end-to-end.
- A kernel-only 250 GB/s stream already consumes 48.158 ms/token; additional
  non-overlapped work lowers application throughput below that target.
- At 244.3 GB/s, weights alone imply 20.292 tg/s. With an illustrative 1.5 ms
  residual, the serial model implies 19.692 tg/s; with 7.3 ms, 17.674 tg/s.
- Reaching 250 application GB/s with those residuals would require effective
  GEMV rates **258.04** and **294.67 GB/s**, respectively.

Those residual values illustrate sensitivity; they are not additive measured
lower bounds for the pinned pipelined chat route. Cache reuse, overlap and exact
critical-path attribution must be measured. Historical streaming results range
from ~238–244 to ~250+ depending on probe/run, and some probes were invalidated
by cache/DCE issues. This audit does not establish a physical DRAM ceiling or
prove 250 application GB/s impossible. It establishes why a small ALU win cannot
justify that promise. Optimize verified `tg/s` even if the stretch is unmet.
