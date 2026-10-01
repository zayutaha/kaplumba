# Finding 01 — The SK4 red test: root cause, and a wrong first fix

Subagent: `ses_f0cf11191ffeCrNKEhmvo7u26Q`. Commits `db61003`, `15955c1`,
`1246690`, `57fa90f`, `379ae9e`. Files: `repacked_golden.rs`,
`implementation_notes.md`, `spec.md`.

## Root cause: the split-K arithmetic was never wrong

Dumping the four partials at element 0:

```
orig -98.65691   raw[0] 41.486683
partials [41.486683, -83.44719, 35.126995, -91.8234] -> sum = -98.65691 exactly
```

What broke is **the observable contract of `raw`**. A split-K GEMV writes one
*partial* per K-split at `raw[split * out_features + n]`, and something must sum
them. The dirty `_sk2` work replaced the two-kernel tail (`gemv_reduce_sk4` sums
back into `raw`, then `linear_finalize_simd`) with a **fused
`linear_finalize_simd_sk4`** that reduces, butterflies, signs and writes fp16 in
one pass — so **`raw` is left holding split 0's partial**.
`repacked_golden.rs` read `raw[i]` and compared it to the reference dot product.

## A wrong first fix, caught in a worktree rather than by reasoning

The first fix (`db61003`) summed the four slots. **It was wrong** — it passes in
the shipping tree and **fails at `e9c2785`**, where `gemv_reduce_sk4` writes the
*full sum* into `raw[0..n]` and leaves partials in `raw[n..4n]`, giving `-238.8005`
(sum + splits 1–3, double-counted). Caught by running it in both worlds.

**There is no expression over all four slots correct under both tails** — because
`raw` is scratch and its post-dispatch layout is not a contract.

**The fix: the test now owns its dispatch.** It dispatches
`linear_gemv_ishizuki_k3_colmajor_sk4` into a buffer nothing else writes, making
the layout known regardless of which tail is compiled in. The shared entry point
is still exercised, but only through `output`. Verified in both worlds.

## Test results

**162 passed, 0 failed, 29 ignored** (`cargo test --release --workspace`, isolated
worktree, private `CARGO_TARGET_DIR`, two clean runs).

## The shared tree is UNBUILDABLE — four independent blockers, none this agent's

1. Untracked `crates/kernels/tests/argmax_device.rs` — 3 compile errors.
2. In-progress `attention.rs` refactor — failing all 4 `attention_metal` tests.
3. Missing `pipeline_audit` / `sweep` module declarations.
4. **Truncated kernel signature at `linear_repacked.metal:1641` and `:1775`**
   (`[[buffer(0)],`).

**#4 is the nasty one.** The `.metal` files are concatenated into one shader
library, so a truncated signature surfaces as `LinearKernels::new()` returning
`KernelError::Shader` and **every kernel test panicking at its own `.expect`,
reading as "the kernels don't work."**

## Split-K: gated, not deleted

Stays exactly as-is behind `KAPLUMBA_USE_SK=1`, per T7.6.9 (split-K slower than
`contig4` on every shape; `contig4` already fills 134 SIMD groups/core).

## fp16 bit-exactness

| shape | `contig4` f32 max ULP | `contig4` fp16 | `contig4_sk2` fp16 |
|---|---|---|---|
| down_proj 21504x5376 | 26 | 0/5376 | **0/5376** |
| gate_proj 5376x21504 | 51 | 0/21504 | **0/21504** |
| o_proj 4096x5376 | **833** | 0/5376 | **0/5376** |

`o_proj` is new; its f32 deviation is 833 ULP, so the spec's "max 51 ULP" was the
max over two shapes. fp16 identical everywhere. **Split-K is NOT bit-exact in
general:** SK4 differs on 2 of 512 synthetic elements by 1 ULP, so it keeps the
associativity tolerance — recorded, not widened.

## Sabotage (AGENTS.md 3.4) — all three observed to fail

| sabotage | caught |
|---|---|
| `finalize_simd_sk4` drops split 3 | `repacked_golden.rs:406` `Finalized SK4 output mismatch` |
| SK4 partial write split-major -> tile-major | `repacked_golden.rs:390` `SK4 reduced-partial mismatch` |
| `finalize_simd_sk2` drops split 1 | `live_kernel_bit_exact` `sk2 fp16 output differs in 336 elements` |

AC6: generator run twice, sha256 unchanged.

## Two false claims corrected in committed notes §20.8

The notes asserted (a) the reduce pipelines were dead and (b)
`live_kernel_bit_exact` asserted `_sk2` — **both false at `e9c2785`**.

Also flagged: `tasks.md` T7.6.9 records `decode_state_mcg`'s `half2` rewrite as
done while it is **still uncommitted**.
