# F5 — o_proj is closed: there was no anomaly, and both "fixes" are worse

**Verdict: o_proj runs at 186 GB/s, inside its peers' 186–203 band. The "159 GB/s
outlier" was my own byte-count error, not a kernel problem. `KAPLUMBA_O_NOSK` loses
1.4% and `KAPLUMBA_O_SK4` loses 0.4%, so the shipped configuration is already the best
available. Nothing to fix.**

## 1. The byte count was wrong, and that is the whole story

The o_proj row in the first version of the F3 table assumed 60 layers of K=8192. The 10
full-attention layers are K=16384, and there are 60 o_proj tensors at two shapes:

```
(336, 512, 48) x50 = 825.75 MB     o, sliding   (N=5376, K=8192)
(336,1024, 48) x10 = 330.30 MB     o, full      (N=5376, K=16384)
                     -----------
                     1156.05 MB
```

So o_proj is **186.0 GB/s** over 6.2164 ms, not 159 over 0.991 GB. `qk_proj`'s bytes
were missing from the table entirely (0.372 GB over 1.95 ms = 190.4 GB/s). Corrected
table in `F3-live-kernels.md`:

| kernel | own ms/step | bytes/token | GB/s | B/TG |
|---|---:|---:|---:|---:|
| gate_up_gelu | 25.58 | 5.202 GB | 203.4 | 64,512 |
| down_proj | 13.10 | 2.601 GB | 198.6 | 129,024 |
| qkv_proj | 8.62 | 1.652 GB | 191.6 | 64,512 |
| qk_proj | 1.95 | 0.372 GB | 190.4 | 16,384 |
| o_proj | 6.22 | 1.156 GB | 186.0 | 49,152 |
| head/gemv | 4.11 | 1.057 GB | ~257 | — |

**186–203 GB/s, no outlier.** The residual ordering is monotone in bytes-per-threadgroup
(64,512 → 203; 49,152 → 186; 16,384 → 190), which is exactly the axis the 4col
experiment moved — and 4col lost (F4). So the shape effect is real but not attackable by
that lever.

## 2. The ablation confirms o_proj's cost, not an attribution artifact

`KAPLUMBA_ABLATE=o_proj_gemv` skips the o_proj matvec entirely, so the output is garbage
and what it measures is o_proj's marginal wall-clock cost. 6 interleaved pairs:

| arm | median tok/s |
|---|---:|
| A — default | 16.001 |
| B — `ABLATE=o_proj_gemv` | **17.865** |

**+11.6%.** Removing 1.156 GB of reads removes ~6.5 ms of a 62.5 ms step, which
corroborates the profiler's 6.22 ms and rules out "the `o_proj` label is mis-attributed".
(Ablation overstates a component because it frees bandwidth for everyone else, so 6.5 ms
is an upper bound on o_proj's own cost — but it is the right order.)

## 3. Both candidate fixes are worse — the microbenchmark ranking is inverted again

`model.rs:1955-1963` already documents the suspect and the two flags to test it:

> "`gemv_bandwidth.rs`, absolute GB/s only, says K=8192 no-SK is **170.6 GB/s** against
> sk2's **120.7** (+41%) while K=21504 sk2 is 199.3 against no-SK's 175.5 (−12%)"

Three arms, one binary, 6 interleaved pairs each:

| arm | o_proj kernel | median tok/s | vs default |
|---|---|---:|---:|
| A — default | `_colmajor_2col_sk2`, 336 TGs | **15.995** | — |
| B — `KAPLUMBA_O_NOSK=1` | `_colmajor_2col`, 168 TGs | 15.768 | **−1.4%** (6/6) |
| C — `KAPLUMBA_O_SK4=1` | `_colmajor_2col_sk4`, 672 TGs | 15.937 | −0.4% |

The +41% the microbenchmark predicted for no-SK comes out as **−1.4%** in production, and
it loses in 6 of 6 pairs. This is the fourth time in this spec's record that a
microbenchmark's variant ranking did not survive contact with the engine
(`perf_probe_splitk_gridshape.rs:65` warns about it in its own header). **Do not select a
kernel from a sweep row.**

## 4. What this closes

* o_proj is not a target. Its rate is shape-inherent (short K, 49 KB/TG) and both
  available alternatives are worse. Split factor is not a lever here.
* The "fixed per-dispatch cost" reading of the o_proj/`down_proj` gap is now unsupported:
  the gap was an artifact of the byte count. There is no evidence of a large fixed
  per-dispatch cost in this data.
* The remaining per-kernel spread is 186–203 GB/s against a 245 GB/s ceiling, and the
  head sits at ~257 — so the mechanism to attack is still the lane window (F4 §last
  lever), not variant selection.
