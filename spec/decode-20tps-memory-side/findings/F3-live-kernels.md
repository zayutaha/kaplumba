# F3 — F1/F2 measured a dead kernel; the live kernels are at ~200 GB/s, and the bandwidth route caps at ~18-19 tok/s

**Three results, in order of how much they change the plan.**

1. **The P01 probe's "production" arm is not the live decode kernel.** `KAPLUMBA_TRACE_GEMV=1`
   on a real decode run prints four names, and `linear_gemv_ishizuki_k3_colmajor_2col`
   — the kernel F1 calls "production `_colmajor_2col`" and F2's entire table is built on —
   **is not among them.** This is the AGENTS.md §3.4 failure mode again: the instrument
   timed an entry point the engine never calls.
2. **The live kernels run at ~200–206 GB/s, not the 176.5 GB/s F2 reported.** The dead
   `_2col` kernel really is slower than the kernels the engine dispatches.
3. **The measured ceiling (241–249 GB/s) puts the bandwidth route's ceiling at ~18–19 tok/s.**
   20 tok/s needs 263 GB/s effective, which is above what this box streams.

---

## 1. The dispatch defect

```
$ KAPLUMBA_TRACE_GEMV=1 ./target/release/kaplumba --model ~/.kaplumba/models/Gxma-repacked \
    --prompt "$PP" --max-new-tokens 8 --temperature 0 | grep GEMV-DISPATCH | sort -u

GEMV-DISPATCH linear_gemv_ishizuki_k3_colmajor_2col_sk2    <- o_proj, down_proj
GEMV-DISPATCH linear_gemv_ishizuki_k3_fused2_2col          <- gate, up
GEMV-DISPATCH linear_gemv_ishizuki_k3_fused2_2col (mlp)
GEMV-DISPATCH linear_gemv_ishizuki_k3_fused3_2col          <- q, k, v
GEMV-DISPATCH linear_gemv_ishizuki_k<unknown>              <- lm_head
```

`_colmajor_2col` is reached only from `encode_gemv_raw`'s 2col arm
(`crates/kernels/src/linear.rs:2130-2157`), and on the live path that arm is not taken:
`gate/up` go through `encode_gemv_fused2_mlp`, `o`/`down` through `encode_gemv_raw_sk2`,
`q/k/v` through `encode_gemv_fused3`, and `lm_head` is `perm_mode == 0` so both 2col
guards fail.

**Consequences for the record:**

* F1's arm (a) "production 463.6–512.4 Gcodes/s" is the code rate of a kernel that does
  not run. The arm (c)/(d)/(e) *ratios* are still meaningful (they are all `_2col`
  self-comparisons), but the absolute "production = 190 GB/s" figure is not.
* F2's per-shape table is about `_2col`, so its 176.5 GB/s byte-weighted figure does not
  describe the engine. Its **qualitative** result — bandwidth monotone in N, code rate
  tracking bandwidth — may still hold, but it has not been measured on the live kernels.
* The `_<unknown>` head name is a cosmetic bug in the reverse pointer→name lookup; the
  head is `linear_gemv_ishizuki_k6` (row-major, rate 6, no colmajor variant).

## 2. Where the step actually goes, and at what bandwidth

`KAPLUMBA_DECODE_PROFILE=kernel`, 32 recorded steps (8 warm-up excluded), Gxma-repacked,
74-token prompt, `--temperature 0`:

> **Corrected (F5).** The `o_proj` byte count in the first version of this table was
> 0.991 GB, which assumed all 60 layers are 8192-wide. The 10 full-attention layers
> are 16384-wide, so o_proj is 825.75 MB + 330.30 MB = **1.156 GB**. Redone below;
> `qk_proj`'s bytes were missing too. The correction matters because it removes the
> o_proj "anomaly" entirely — see `findings/F5-o-proj-closed.md`.

Bytes/token by shape, from the safetensors headers (12.0395 GB total):

| shape (colmajor) | × | bytes | role |
|---|---:|---:|---|
| `(1344,336,48)` | 120 | 5202.25 MB | gate + up |
| `(336,1344,48)` | 60 | 2601.12 MB | down |
| `(336,16384,96)` | 1 | 1056.96 MB | lm_head (row-major, K=5376 N=262144) |
| `(512,336,48)` | 50 | 825.75 MB | q, sliding |
| `(256,336,48)` | 100 | 825.75 MB | k + v, sliding |
| `(336,512,48)` | 50 | 825.75 MB | o, sliding |
| `(1024,336,48)` | 10 | 330.30 MB | q, full |
| `(336,1024,48)` | 10 | 330.30 MB | o, full |
| `(128,336,48)` | 10 | 41.29 MB | k/v, full |

| kernel | count/step | own ms/step | share | bytes/token | **GB/s** |
|---|---:|---:|---:|---:|---:|
| `gate_up_gelu` | 60 | 25.58 | 40.0% | 5.202 GB | **203.4** |
| `down_proj` | 60 | 13.10 | 20.5% | 2.601 GB | **198.6** |
| `qkv_proj` | 50 | 8.62 | 13.5% | 1.652 GB | **191.6** |
| `qk_proj` (10 full layers) | 10 | 1.95 | 3.1% | 0.372 GB | **190.4** |
| `o_proj` | 60 | 6.22 | 9.7% | 1.156 GB | **186.0** |
| `head/gemv` | 1 | 4.11 | 6.4% | 1.057 GB | ~257 |
| `attention` | 60 | 2.34 | 3.7% | — | — |
| `post_ffn_out` | 60 | 0.91 | 1.4% | — | — |
| `post_attn_pre_ffn_transform` | 60 | 0.69 | 1.1% | — | — |
| `kv_norm_rope_cache` | 60 | 0.33 | 0.5% | — | — |
| norms, softcap, argmax, embedding | — | ~0.07 | 0.1% | — | — |
| **own total** | | **63.90** | | | |

Profiled span median 76.10 ms/step; **uninstrumented step is 62.8 ms** (measured
`decode_tok_s=15.917`, 128 tokens, temp 0). Matvec share = 93.2%, so ~58.5 ms of the
62.8 ms step is weight streaming over ~12.04 GB ⇒ **~206 GB/s live**, against a
**245 GB/s** streaming ceiling.

The head is the fastest kernel in the model at ~257 GB/s (over the ceiling, so its
`own_ms` is probably understated, but it is clearly not the problem). The **3 bpw**
layer kernels are all at 159–203.

**Every 3 bpw projection is in a narrow 186–203 GB/s band.** There is no o_proj
outlier: the earlier 159 GB/s was a byte-count error (F5). The spread is monotone in
bytes per threadgroup (gate_up 64,512 B/TG at 203 down to o_proj's 49,152 B/TG at 186),
which is the same axis the 4col experiment was built to move — and 4col lost (F4).

## 3. The ceiling, re-measured today

`probe_bw_250gb` (self-testing, fail-first sabotage gate passes), pure `uint4` stream:

```
2 GB   th=40960 tg=256  med 243.2  best 251.3
2 GB   th=81920 tg=256  med 245.4  best 253.3
2 GB   th=81920 tg=512  med 249.4  best 250.9
6 GB   th=40960 tg=256  med 247.9  best 252.1
12 GB  th=40960 tg=256  med 245.3  best 246.9
```

So ~241–249 GB/s median, 253 best. The user's 250 GB/s is correct **for a pure stream of
a 2 GB buffer**; it is not delivered into the kernel by 12 GB of weights.

## 4. What that means for 20 tok/s — the arithmetic

```
step = matvec + non-matvec
now:       58.5 ms + 4.3 ms = 62.8 ms   (15.92 tok/s)   matvec at 206 GB/s
at 245 GB/s: 49.1 ms + 4.3 ms = 53.4 ms  (18.7 tok/s)
at 237 GB/s: 50.8 ms + 4.3 ms = 55.1 ms  (18.1 tok/s)    <- arm (G)'s instruction bound
20 tok/s:    50.0 ms total, so matvec <= 45.7 ms  => 263 GB/s
```

**263 GB/s is above the measured 241–249 GB/s ceiling and above the best single reading
of 253.** So 20 tok/s is not reachable by making the existing kernels faster, at
12.039 GB/token — this is an arithmetic statement about the box, not a claim about any
particular kernel being bad.

There are two honest ways this arithmetic changes, and only two:

* **Fewer bytes per token.** AGENTS.md §8 forbids precision loss, so the only lossless
  form is *not reading weights that do not change the answer*.
* **More tokens per weight pass — verify-gated speculative decoding / MTP.** §8 permits
  it explicitly ("speculative decoding must be verify-gated (greedy-identical or
  distribution-exact)"). It multiplies accepted tokens per streamed weight, which is the
  only lever that beats the byte ceiling. **No speculative-decoding or draft-model
  infrastructure exists in this tree** (searched); `spec/mtp-40tps/` is about a different,
  4-bit checkpoint and is not a decoder.

## 5. The one bandwidth lever that is still open, and the evidence for it

The head — `k6`, row-major, 6 bpw — reaches ~257 GB/s. The 3 bpw layer kernels reach
159–203. Same box, same ceiling, same step. The distinguishing property is the **lane
window**: at 3 bpw a lane reads 2 u32 (64 bits) to recover 8 codes (24 useful bits) =
**2.67 usable bytes read per useful byte**; at 6 bpw the same 64-bit window carries 16
codes (48 useful bits) = 2.0×, and at 4 bpw 2.0×.

That is `spec/decode-bandwidth-240gb`'s T7 mechanism, and it is the *only* explanation
in the record that predicts the head being fast and the layers being slow with the same
scheduler, the same grid shape and the same DRAM.

**The candidate fix, stated precisely.** A 64-bit window at a 3-bit stride spans 17
codes (`16*3 - 3 + 16 = 61 <= 64`). The current kernel serves 8. Serving 16 (a whole
16-wide tile row) from the *same two words* would take the amplification from 2.67× to
1.33×, and it is the only change in the plan that attacks the mechanism the head's rate
demonstrates rather than adding streams beside it. It requires re-laying the lane→code
map and the reduction, so accumulation order changes and bit-exactness must be re-earned
— which is why it is a kernel rewrite and not a tweak.

## 6. What is now closed, so the next agent does not re-spend it

* `_2col` is not the live kernel (this file). Do not measure it as "production".
* More loads in flight: `decode-bandwidth-240gb` measured 0.08M→2.62M loads in flight
  (32×) at 1.06× bandwidth. Not the lever. `ROUND1-2-SUMMARY.md:15` and
  `A3-kernelsel.md:203` record two more attempts at +0.5% and ≤+0.3%.
* `perf_probe_splitk_gridshape`'s sweep: its own header says its variant *ranking* is
  inverted relative to production. Do not ship a variant on a sweep row.
