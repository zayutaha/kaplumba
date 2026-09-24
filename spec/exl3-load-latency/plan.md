# Plan: exl3-load-latency

## Architecture and data flow

This spec adds **no production data flow**. It adds a measurement ladder beside the
existing bench, and deletes code proven dead.

The decode hot path, unchanged by this spec:

```
token activations (f16)
   -> exl3_had_in / had_in_batched        (suh scaling + 128-pt Hadamard)
   -> exl3_gemv_spec  (k=3, K-split)      <-- the kernel under investigation
        per (out_tile, split) threadgroup, 32 lanes
        per lane per input-tile `it`:
            4x ld_word(tile, ig0_[*] / ig1_[*])   -> 4 scattered 32-bit device loads
            2x ulong window assembly              -> 64-bit glue
            8x 64-bit shift + mask                 -> 8 codewords
            8x exl3_decode_f_const                -> 8 floats
            8x simd_shuffle(xh) + 8x FMA          -> accA / accB
        writes f32 partials [(split*out_tiles + ot)*TILE + c]
   -> exl3_gemv_reduce / _reduce_post        (split sum + butterfly x svh -> f16)
```

The bisection reproduces this loop in a **bench-only MSL source string** and removes one
stage at a time, so each delta is the marginal cost of that stage:

```
S1 walk     S2 gather     S3 extract     S4 decode     S5 full
  |            |            |              |             |
  +-- tile walk, 1 word/lane load, sum
  +-- + 4-way scattered ld_word gather
  +-- + ulong glue + 64-bit shift/mask extraction, XOR-accumulated
  +-- + exl3_decode_f_const, float-summed
  +-- + simd_shuffle(xh) + FMA into accA/accB  == exl3_gemv_spec math
```

`S1` is already implemented as `stream_sum_strided` in `P0_STRIDED_MSL`, and is the control
that produced the 212 GB/s figure. S2–S5 are new kernels in a new MSL string
(`P1_BISECT_MSL`) reusing the same parameter block and the same grid convention
(`grid.x = out_tiles * nsplit`, 32 threads/TG), so the ladder differs from the real kernel
only in the stage under test.

## File and module touchpoints

| path | current | intended |
|---|---|---|
| `crates/engine/src/metal_linear.rs` | `P0_CONTIG_MSL`, `P0_STRIDED_MSL`, `bench_stream_bw_slow`; `exl3_decode_f_const` logic is **not** in the MSL string (it lives in `exl3.metal`) | add `P1_BISECT_MSL` (S2–S5, with an inlined copy of the decode bit-hack) and extend the bench to sweep stages x shapes x cold/warm. Nothing else changes. |
| `crates/kernels/src/exl3.metal` | `exl3_gemv_dense1` (L3074), `exl3_gemv_dense4` (L3224) — dead, no host callers | delete both kernels. `exl3_gemv_spec` untouched. |
| `spec/bandwidth-saturate/implementation_notes.md` | root-cause model says memory-pattern-bound | append the control result; retract the ALU-bound and straw-drinking claims |
| `spec/bandwidth-saturate/tasks.md` | T4, T6 unchecked | mark deferred-with-reason |
| `spec/trellis-dense-repack/tasks.md` | RT2b kernels shipped, RT4 A/B pending | record the A/B outcome: both candidates rejected by measurement |
| `spec/exl3-load-latency/implementation_notes.md` | — | per-stage table, dominant term, follow-up recommendation |

### Interfaces

New, bench-only, no public API:

```rust
/// Bench-only MSL. Kernels `bisect_gather`, `bisect_extract`, `bisect_decode`,
/// `bisect_full` — one per ladder rung above `stream_sum_strided`.
const P1_BISECT_MSL: &str = r#"..."#;
```

Shared kernel signature (mirrors `exl3_gemv_spec`'s buffer convention):

```metal
kernel void bisect_<stage>(
    device const uint  *buf     [[buffer(0)]],  // trellis words, [it][ot][twords]
    device atomic_uint *out     [[buffer(1)]],
    constant uint      *p       [[buffer(2)]],  // [in_tiles, out_tiles, twords, nsplit]
    uint tg [[thread_position_in_grid]],
    uint thr [[thread_position_in_threadgroup]]);
```

`ig0_`, `ig1_`, `s3_` are recomputed per rung with the same arithmetic as
`exl3_gemv_spec` (k=3, mode=0 baked as literals — the bench is k=3-only), so the scattered
gather in S2 touches exactly the words the real kernel touches. `exl3_decode_f_const` is
inlined into the bench MSL (5 lines) so the ladder does not have to link the production
library.

## Sequencing and dependencies

```
T0  revert T4                       (no commit; restores tree to 4b3708f)
     |
T1  docs: control result + kill dense-repack premise      -> commit (docs only)
     |                                                        (independent of T2)
T2  delete dense1/dense4, verify library compiles, tests  -> commit (code)
     |
T3  add P1_BISECT_MSL + bench sweep                      -> commit (code+bench)
     |
T4  run at real sizes, record table, report               -> commit (docs)
     |
     STOP. optimization is a follow-up spec, chosen from T4's data.
```

T0 must precede everything (it discards an unrelated half-finished change). T1 and T2 are
independent and could be either order. T3 requires a clean tree so the only delta under test
is the ladder itself.

## Risks and fallbacks

- **Harness invalid (R5 fails).** S5 will not match `exl3_gemv_spec`'s time. Fallback: the
  deltas are reported as unreliable and the task stops there. Do **not** infer a dominant
  term from a ladder whose top rung does not reproduce the thing it is supposed to model.
- **MSL compile error in the bench string.** The kernels crate has no `build.rs`; MSL is
  compiled at runtime, so a syntax error only surfaces when the bench runs. Mitigation: run
  the bench immediately after adding the string, before committing.
- **Ladder measured at the wrong size.** 43 MB may be SLC-resident and report inflated
  GB/s. Mitigation: R8's cold/warm split and real per-linear sizes. If cold and warm
  disagree by >2x, report both and do not average them.
- **`exl3_decode_f_const` bit-hack drift.** The inlined copy in `P1_BISECT_MSL` must stay
  identical to the production one (`x * 89226354 + 64248484`, `^ (x & 0x8FFF8FFF)`,
  `as_type<half2>` pairs summed). Mitigation: R5 — if the inlined decode were wrong, S5's
  *time* would still match but the rung ordering would be suspect; assert the S5 kernel's
  accumulator is non-zero like the existing `assert_ne!(simd_sum(...), 0, "read nothing")`
  guard so a dead rungs cannot masquerade as a fast one.
- **Deltas point at the already-failed coalesced-load idea.** Fallback: stop and report, per
  spec R5/risks. The follow-up spec must design something structurally different.

## Requirement -> task map

| requirement | task |
|---|---|
| R1 revert T4 | T0 |
| R2 delete dense kernels | T2 |
| R3 record control result in both specs | T1 |
| R4 bisection ladder S1–S5 | T3 |
| R5 S5 matches `exl3_gemv_spec` | T3 (assert) + T4 (verify) |
| R6 no other code touched | every task (scope check in `git diff --stat`) |
| R7 both suites green | T2, T3 |
| R8 real sizes, cold/warm | T3, T4 |
| R9 notes record table + conclusion | T4 |
