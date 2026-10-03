# F4 — `fused2_4col` has now been run, and it loses. Column tiling is closed.

**Verdict: `KAPLUMBA_FUSED2_4COL=1` is 0.7% *slower* than the live `_fused2_2col` on a
bit-identical output, in 6 of 6 interleaved pairs.** The "reachable, never run"
experiment named in `spec/decode-ceiling/findings/A3-kernelsel.md` has been run, with
the confound it warned about removed, and the answer is negative.

## What was changed, and why it was necessary first

`linear_gemv_ishizuki_k3_fused2_4col` (`linear_repacked.metal:580`) inlined its ring-word
index expression into each `extract_v_mcg_32(...)` call, holding **one** word pair in
flight where `_fused2_2col` holds **four** (`:491-495`). A3 predicted this exactly:

> "Without 1a the experiment measures the absence of a hoist, not column tiling, and a
> loss would be uninterpretable."

So T4.1 (commit `bb858c6`) raises the 16 loads above the first `extract_v_mcg_32`, the
same shape as the T4.3 hoist already in `_colmajor_2col_sk2` (`:2477-2480`). It is a
scheduling change only: no accumulation order, no reduction tree, no column ownership
moves. The A/B below is therefore a fair test of column tiling.

## The bit-exactness gate, demonstrated rather than asserted

Two arms, one binary, `--temperature 0 --show-ids`, 128 greedy tokens:

```
arm A  (default, _fused2_2col)                          token ids hashed
arm B  (KAPLUMBA_FUSED2_4COL=1, _fused2_4col + hoist)    token ids hashed

armA 4ce9ef31cfe358957ce89573aa72864e3be638123a6090778200504bdd782d4f
armB 4ce9ef31cfe358957ce89573aa72864e3be638123a6090778200504bdd782d4f   IDENTICAL
```

128 of 128 ids match. Four columns per threadgroup is **bit-exact** against two on this
checkpoint, which is a stronger result than the sibling `contig4` kernels have (those
reassociate and are only fp16-equal after the finalize).

## The A/B — 6 interleaved pairs, one binary, one prompt

`--temperature 0 --metrics`, 128 generated tokens, A B A B … :

| round | A (default) | B (`FUSED2_4COL=1`) |
|---|---:|---:|
| 1 | 16.024 | 15.920 |
| 2 | 16.019 | 15.905 |
| 3 | 16.025 | 15.910 |
| 4 | 16.050 | 15.902 |
| 5 | 16.011 | 15.913 |
| 6 | 15.421 | 15.851 |
| **median** | **16.021** | **15.907** |

B is slower in **6/6** pairs by ~0.7%. Within-arm spread is 0.03 tok/s (one outlier at
15.421 aside), so the 0.114 tok/s gap is far outside the noise. Note how *quiet* the box
was here — the recorded 35% same-config spread did not appear, which is why a 6-pair run
resolved a 0.7% effect.

## Why this matters beyond the negative

* **The `bytes-per-threadgroup` axis does not pay.** A3's best case was that doubling
  B/TG from 64,512 → 129,024 (the value `down_proj` runs at) would buy ~+0.57 tok/s. It
  does not. The confound A3 identified in the older `_co4` measurement was real, and
  removing it changed nothing — the loss is column tiling itself.
* The earlier `_co4` datapoint (−0.19 tok/s, `decode-25tok-peak-bandwidth` §T5.2) was
  therefore **not** an artifact of its five-variable confound. Two independent kernel
  shapes now agree: 4 columns per threadgroup loses.
* Combined with F3, the remaining candidate list for the ~206 → 245 GB/s gap is down to
  the lane-window amplification, and that one **cannot be fixed without changing the
  on-disk ring layout** — see below.

## The last lever, and why it is not a tweak

F3 identifies the mechanism: the head (`k6`) reaches ~257 GB/s while every 3 bpw layer
kernel runs 159–203, and the only property that differs is useful bytes per lane-load —
a 64-bit window carries 24 useful bits at 3 bpw (2.67×), 48 at 6 bpw (1.33×).

Widening the 3 bpw window to 16 codes per lane would take it to 1.33×. **It is blocked by
the ring layout, not by the kernel.** The ring is ordered by `(lane, k)` with
`lane = 4*(o mod 8) + g`; a given output's 16 inputs are therefore *strided* across four
lanes by 32 codes, not contiguous. A 16-code contiguous window does not correspond to a
usable set of `(output, input)` pairs. Serving 16 codes per lane needs a **new ring
order — a repack of all 411 tensors** (`reorder_tensor_to_colmajor` +
`pack_integer_states`, `kaplumba_engine::exl3`), a new on-disk format version, and a
rewritten reduction.

That is a format change with its own spec, not a task in this one.

## Verified byte count (the number the whole ceiling argument rests on)

Read directly from the `Gxma-repacked` safetensors headers, all shards, all `.trellis`
tensors:

```
total                                          12.0395 GB
  words_per_tile=48 (3 bpw)                    10.9825 GB
  words_per_tile=96 (6 bpw, lm_head only)       1.0570 GB
```

So 12.039 GB is exact, and the F3 arithmetic is not resting on a stale figure.

## Arithmetic restated once, with the measured ceiling

```
bytes/token                          12.0395 GB   (headers, above)
streaming ceiling                    241-249 GB/s median, 253 best (probe_bw_250gb)
live matvec rate                     ~206 GB/s
non-matvec (attention, norms, qk)     4.3 ms

at 245 GB/s:  12.0395/0.245 = 49.1 ms + 4.3 = 53.4 ms  ->  18.7 tok/s
20 tok/s:     50.0 ms total, so the matvec must finish in 45.7 ms
              => 263 GB/s, above the box's best single reading (253)
```

The 4.3 ms is a dependent chain (qkv → attention → o → gate/up → down, per layer), so it
cannot be hidden without speculating. **20 tok/s is not reachable from bandwidth at
12.039 GB/token**, and the bandwidth route's own maximum is ~18.7.
