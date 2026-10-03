# P17 — Dispatch-selection audit

Test delivered: `crates/engine/tests/perf_probe_dispatchsel_columns.rs` — 17
tests, no GPU, all green. Verified by mutation: 6 mutations, 5 caught; the 2
escapes that shaped the design are logged in the file.

## The real dispatch table (`Gxma-repacked`, decode)

Two layer populations, read off the checkpoint's own `svh` shapes — the notes
assume a single population and this is the first thing that matters:

| projection | layers | in → out | kernel | selected at |
|---|---|---|---|---|
| q,k,v | 50 sliding | 5376→8192/4096/4096 | `linear_gemv_ishizuki_k3_fused3_2col` | `model.rs:1462`→`linear.rs:2785` |
| q,k | 10 full-attn | 5376→16384/2048 | `linear_gemv_ishizuki_k3_fused2_2col` | `model.rs:1534`→`linear.rs:2435` |
| o_proj | 60 | 8192/16384→5376 | `..._k3_colmajor_2col_sk2` | `model.rs:1941`→`linear.rs:2257` |
| gate,up | 60 | 5376→21504 | `..._k3_fused2_2col` | `model.rs:2144`→`linear.rs:2595` |
| down_proj | 60 | 21504→5376 | `..._k3_colmajor_2col_sk2` | `model.rs:2242`→`linear.rs:2257` |
| lm_head | 1 | 5376→262144 | `linear_gemv_ishizuki_k6` | `model.rs:4420`→`linear.rs:1559` |

## 1. `_colmajor_contig4` is DEAD on the decode path

`ishizuki_gemv_pipeline` — the function every dispatch note reasons about, and
which `rate4_colmajor_bit_exact.rs:604-613` asserts is "the kernel the engine
dispatches" — is reached **only by lm_head**, on `perm_mode == 0`, where it
cannot pick a colmajor kernel at all.

All 411 trellis offsets in the repacked checkpoint are `%4 == 0`, so nothing
falls through to the EXL3 arm either. **Therefore the `_mcg` family and the whole
`NR0` axis never execute.**

`linear.rs:1937-1940` ("only ever runs for lm_head, whose trellis starts at
`% 4 == 2`") is **true of `~/.kaplumba/models/Gxma` and false of the repacked
checkpoint.** That comment is a live trap: it will lead the next reader to
conclude `contig4` is tuned and in use. It is neither.

## 2. Two projections run split-K unconditionally

`o_proj` / `down_proj` default `use_sk2 = !env_flag("KAPLUMBA_NO_SK2")` —
**opt-OUT**, not opt-in. So one bit rate, four projections, three kernels, and one
split applied twice. Any A/B that "adds split-K" must subtract it here first.

## 3. Latent fail-open — and it is a 4 bpw hazard

`linear.rs:1526-1532` justifies the `_w2` lookup as "fail closed". **The
`perm_mode == 1` lookup 15 lines above is not.** Only k3 has colmajor kernels, so
a **colmajor checkpoint at any rate other than 3 silently dispatches the
non-colmajor kernel for that rate.** `"head_bits": 8` is currently accepted by the
loader (`config.rs`), so a 4/8 bpw colmajor checkpoint would take this path.

This is a candidate explanation for the 4 bpw fault and should be checked before
any further 4 bpw kernel work: it fails open, silently, and produces wrong output
rather than an error.

## 4. `o_proj`'s "150 GB/s" is a byte-model artefact

The notes compute `o_proj` as 60 × the sliding shape. The real total is
**12.040 GB/token**, not 11.834. Corrected: `o_proj` **174 GB/s — the fastest
layer projection**; `qkv` 128. No outlier drags the average down, and the
projected ceiling moves **17.2 → 16.9 tok/s** (worse).

## 5. The head does NOT get a better kernel — 🔴 corrects a claim I made

I previously told the user that `lm_head` reaching 253 GB/s proved that rate was
available to layer kernels. **That is wrong.**

The head uses the **same template** `linear_gemv_ishizuki_k{N}`, differing only in
`KA`. The advantage decomposes as:

| factor | value |
|---|---|
| bits-per-code amplification | 2.000 at 6 bpw vs 2.667 at 3 bpw → worth ≤ 1.33× |
| grid size | 16 384 threadgroups vs 336–1344 → worth the rest (~1.5×) |
| **product** | **2.01×** — matches the observed ratio |

And on the window axis the head is on the **older 3-word** form (`_k6_w2` does not
exist), i.e. it is the *worse* of the two.

**Do not read 253 GB/s as 2× of headroom available to a layer kernel.** The grid
size is not reproducible at layer shapes — 336 threadgroups for q_proj is what
the parallelism actually is.