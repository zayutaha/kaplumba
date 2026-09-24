# Spec: exl3-load-latency

## Goal
Find out, by measurement rather than guesswork, why `exl3_gemv_spec` delivers only
30–95 GB/s on the k=3 layer linears when the *identical* memory access pattern with no
arithmetic sustains 212 GB/s on this M4 Pro, and produce a per-piece cost breakdown that
identifies the dominant term. This spec delivers the **measurement**; the optimization it
enables is a follow-up spec, chosen from the data this produces.

## Non-goals / restrictions
- **No production change lands in this spec.** `exl3_gemv_spec` and every other decode path
  stay byte-for-byte as they are at `4b3708f`. The bisection kernels are bench-only MSL,
  compiled at runtime inside a `#[ignore]`d test, never added to `include_str!("exl3.metal")`.
- **No quality loss.** Bit-exact or nothing. No ≤1-ulp fudge, no reassociation, no
  approximation of the EXL3 decode. The golden fixtures (`argmax 236773`, trace token 818
  at step 19) and the existing `gpu_gemv_*` gates must stay green.
- **No speculation, no batching, no multi-user scheduling.** Single-user `rows == 1` decode
  is the only target. No n-gram/draft-model shortcuts.
- **No weight-format change.** k=3 EXL3 stays k=3. Going to 3.5/4-bit is a separate project
  with its own conversion, residency and quality analysis.
- **No repacking of the trellis.** The dense W1/W4 side-car layouts are dead (see Evidence);
  their kernels are deleted, not revived.
- **No Python** in engine or product code. No new resident memory allocation — the new
  kernels reuse the bench's existing synthetic buffer.
- **Never land red.** The `decode_throughput_slow` ≥ 19.5 t/s assertion stays unflipped.

## Evidence this spec rests on
`bench_stream_bw_slow` (run 2026-09-24, 512 MB synthetic u32, M4 Pro):

| variant | GB/s |
|---|---|
| contiguous linear read, 8 u32/thread | 210 |
| EXL3 tile-walk stride, nsplit=1 | 212 |
| EXL3 tile-walk stride, nsplit=2 | 323 |
| EXL3 tile-walk stride, nsplit=4 | 370 |
| EXL3 tile-walk stride, nsplit=8 | 368 |

The EXL3 stride pattern is **not slower than a fully linear read**, and comfortably exceeds
the 273 GB/s nominal bus figure (see Risks). Therefore the premise that drove
`spec/trellis-dense-repack` — "336 tiny 64-byte sips 21 KB apart cap the bus at ~68 GB/s" —
is **falsified**. The gap is inside the kernel.

The earlier `exl3_gemv_probe` conclusion ("decode ALU is free, m0 ≈ m2") is **also
withdrawn as a control**: that probe retained the real kernel's per-lane gather structure
and grid, so both modes were pinned at the same non-DRAM ceiling. It measured the kernel,
not memory.

## Requirements
- **R1** Uncommitted T4 (fused RMSNorm+Hadamard, 435 additive lines across 3 files) is
  reverted; the working tree returns to `4b3708f` content.
- **R2** `exl3_gemv_dense1` and `exl3_gemv_dense4` are deleted from `exl3.metal`. The whole
  MSL still compiles (verified by any engine GPU test constructing `MetalLinear::new`,
  since `new_library_with_source` compiles the entire source).
- **R3** The control result above is recorded in
  `spec/bandwidth-saturate/implementation_notes.md`, and `spec/trellis-dense-repack/tasks.md`
  records the layout question as resolved (both candidates rejected by measurement), so no
  future reader chases either dead premise.
- **R4** A bisection harness isolates, at real per-linear sizes, the incremental cost of
  each stage of the `exl3_gemv_spec` inner loop:
  - **S1 walk** — the tile walk with minimal loads (1 word/lane, the existing
    `stream_sum_strided` shape);
  - **S2 gather** — S1 plus the real 4-way scattered `ld_word` at `(ig0_[0], ig1_[0],
    ig0_[1], ig1_[1])`;
  - **S3 extract** — S2 plus `ulong` window assembly and the 8 64-bit shift/mask
    codeword extractions, **without** the decode;
  - **S4 decode** — S3 plus full `exl3_decode_f_const` on all 8 codewords;
  - **S5 full** — S4 plus the `simd_shuffle` of `xh` and the FMA accumulation into
    `accA`/`accB` (i.e. the complete `exl3_gemv_spec` math).
- **R5** S5 must reproduce `exl3_gemv_spec`'s measured time within measurement noise. This
  is the harness validity check: if S5 does not match, the harness is wrong, no delta is
  trusted, and the discrepancy is reported instead of an optimization being attempted.
- **R6** Only the new kernel, `exl3_gemv_spec`, and the bench harness are touched. No other
  kernel, no host dispatch path, no `gpu_decode.rs`.
- **R7** Every existing test stays green: `cargo test -p kaplumba-kernels --lib` and
  `cargo test -p kaplumba-engine --lib` both 0 failed, with counts recorded.
- **R8** Measurements are taken at real per-linear byte sizes (o 16.5 MB, down/gate 43.4 MB,
  lm_head 1.05 GB) in addition to a large synthetic, and cold (first-touch, post-flush) and
  warm (min-of-N) figures are reported **separately** — a 43 MB working set may sit partly in
  the SLC, so a single number would be misleading.
- **R9** The per-stage GB/s table and the resulting conclusion about the dominant cost are
  written into `implementation_notes.md` in this spec folder.

## Acceptance criteria
1. R1, R2, R6, R7 satisfied; `git status` shows no uncommitted source drift at the end of
   each task; the two test-suite pass counts are recorded verbatim.
2. R3, R4, R8 satisfied: both specs carry the control result, and the bisection prints a
   per-stage, per-shape table with deltas.
3. R5 satisfied, or the harness defect is reported and no optimization is claimed.
4. R9 satisfied: notes contain the table plus an explicit statement of which stage dominates
   and what the follow-up spec should therefore attack.
5. `spec/bandwidth-saturate/tasks.md` T4/T6 are marked deferred-with-reason, not silently
   left pending.

## Risks and accepted constraints
- **The 210 GB/s "ceiling" may be a launch-size artifact.** It comes from one 512 MB
  synthetic launch; real linears are 16–87 MB and may sit partly in the SLC. R8 measures at
  real sizes and reports cold/warm separately so the bar is not mis-derived. *Accepted,
  handled by R8.*
- **370 GB/s exceeds the 273 GB/s nominal bus figure.** The nominal and measured numbers
  disagree, most likely via SLC residency and/or boost-clock variance. The bisection uses
  **212 GB/s (nsplit=1, same-shape control)** as the defensible comparison point and does
  not build any conclusion on the 370 figure. *Accepted; flagged here so the follow-up spec
  does not treat 273 as a hard physical constant.*
- **19.5 t/s sits above the measured contiguous streaming ceiling.** 19.5 t/s needs
  242.6 GB/s sustained; a pure linear read measured 210 GB/s, implying ~17.8 t/s at
  11.833 GB/token. The user chose to keep 19.5 t/s as the bar anyway. The gate therefore
  stays unflipped and will likely stay red. *Explicitly accepted by the user 2026-09-24.*
- **The deltas may point at something already tried and reverted.** A previous attempt at a
  coalesced `uint4` + `simd_shuffle` load in `exl3_gemv_spec` measured *slower* and was
  reverted (`gu2` 0.933 → 1.437 ms). If the bisection identifies load pattern as dominant,
  the follow-up must not simply redo that experiment. *Handled by stopping and reporting.*

## Open questions
None. All ambiguities were resolved with the user before this folder was opened
(2026-09-24): T4 dropped; 19.5 t/s bar kept; `exl3_gemv_spec` first; dense kernels deleted
but stash kept; bisect before optimizing; new kernel alongside; session scope = deletion +
bisection then report.
