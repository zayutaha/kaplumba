# FINDINGS — decode-20tok-crux3, session 2026-10-05

This folder had only `HANDOVER.md`. This is the measured state behind it:
the live dispatch, the per-step breakdown, the kernel A/Bs run here, and the
arithmetic that decides the target.

All numbers M4 Pro 20-core, 24 GiB, model `~/.kaplumba/models/Gxma-repacked`,
`--temperature 1 --top-k 64 --seed 1`, `scripts/gpuq try`.

---

## 1. Hardware, confirmed

`system_profiler`: Mac16,7, **Apple M4 Pro**, 14 CPU cores, **20 GPU cores**,
24 GB. Memory bus **273 GB/s** (Apple's pin rate; the best *measured* streaming
arm in this repo is 253.6 GB/s, sustained 245). There is no second memory path.

## 2. The live decode kernels (not what the crux3 probe measured)

`KAPLUMBA_TRACE_GEMV=1`:

```
GEMV-DISPATCH linear_gemv_ishizuki_k3_fused3_2col      (q,k,v sliding)
GEMV-DISPATCH linear_gemv_ishizuki_k3_fused2_2col (mlp) (gate, up)
GEMV-DISPATCH linear_gemv_ishizuki_k3_fused2_2col      (full q,k)
GEMV-DISPATCH linear_gemv_ishizuki_k3_colmajor_2col_sk2(o_proj, down_proj)
GEMV-DISPATCH linear_gemv_ishizuki_k6                  (lm_head)
```

`_colmajor_2col_sk4` — the "production arm A" of the crux3 probe — is **not on
the decode path** (matches `P17-dispatch-selection.md`). All live kernels do
share `extract_v_mcg_32`, so crux3's statement about the *decode arithmetic* is
about the right helper; its 203/238 numbers are not the live kernel's.

## 3. Measured decode step at 2k (`KAPLUMBA_DECODE_PROFILE=kernel`)

2054-token prompt, span **76.9 ms/step**, per-kernel own **69.0 ms**:

| kernel | own ms/step | GB (per token) | GB/s |
|---|---:|---:|---:|
| gate_up_gelu | 25.58 | 5.202 | 203 |
| down_proj | 13.08 | 2.601 | 199 |
| qkv_proj | 8.66 | 2.023 | 234 |
| attention | 7.44 | 1.007 | 135 |
| o_proj | 6.22 | 1.156 | 186 |
| head/gemv | 4.12 | 1.057 | **257** |
| qk_proj | 1.95 | — | — |
| epilogues+norm | ~1.9 | — | — |

**The live GEMV runs at ~202 GB/s** for the 12.058 GB of layer weights — the
same 203 GB/s floor the crux3 probe recorded. `head/gemv` is **not** slow: it
reaches 257 GB/s. (An earlier read of 14.5 ms for `head/gemv` was the *first,
cold* forward the profiler samples, not steady state.)

End-to-end: **~14.1–14.4 tok/s at 2k.**

## 4. Kernel A/Bs run here (all single-run, 64 tokens, 2k)

| arm | decode tok/s | kernel confirmed by TRACE |
|---|---:|---|
| default | 14.12 / 14.19 | `_fused2_2col` |
| `KAPLUMBA_FUSED2_4COL=1` | 14.24 | `_fused2_4col` ✔ |
| `KAPLUMBA_NO_FUSED2_2COL=1` | 14.37 | `_fused2_co` ✔ |
| `KAPLUMBA_FUSED2_CO4=1` | 14.22 | `_fused2_co4` ✔ |
| `KAPLUMBA_USE_SK=1` | **7.79** | `_fused2_sk2` ✔ |
| `KAPLUMBA_NO_SK2=1` | 13.86 | removes o/down split |
| `KAPLUMBA_DECODE_FLASH_MIN_KEYS=2048` | 14.59 | flash on at 2k |

Every arm except `USE_SK` is inside this box's run-to-run spread (~±3%): **no
FFN kernel variant is a lever.** `USE_SK` is a 45% regression and must stay off
(already is). The flash-threshold arm is +3% but affects only the 10 full
layers below 4096 keys and is *not bit-exact* vs the split kernel, so it is not
committed — it does nothing at 16k, where flash is already on.

## 5. Why "hit SRAM" cannot raise the decode rate — the decisive ablation

The seductive numbers are real measurements of **cache**, not DRAM:

- `StorageModeShared` streaming measured **804 GB/s on a 273 GB/s wire**
  (`spec/decode-20tps/implementation_notes.md:347`) — the host's recent write
  kept the pages cacheable.
- An L2-resident attention harness measured **1,592 GB/s**
  (`spec/decode-19tok/plan.md:96`).
- crux3's small-shape arm reached **2.23× the wire peak** on a 3 MB working set
  because it is served from L2.

All three are correct: on-chip SRAM does run at **~1.6 TB/s**, 6× DRAM. The
question is whether the decode can be made to read from it. **It cannot, and
this is measured, not argued:**

1. **The weights are 12.058 GB and are read exactly once per token.** There is
   no reuse to cache. On-chip SRAM is ~16–48 MB; the best case is ~0.3% of the
   bytes. Every byte must cross DRAM once.
2. **Even when the weights *are* made SRAM-resident, the kernel does not speed
   up.** `spec/decode-19tok-repacked-250gb` arm (d) folded the column footprint
   from 2.6 GB to a **129 KB L2-resident window, keeping every load and address →
   −0.6%.** That is the direct test of "hit SRAM more": it is neutral. The GEMV
   is **ALU/issue-bound**, not memory-bound (`MEMORY.md`: ~8:1 ALU:memory;
   `decode-3bpw-amplification`).

So caching cannot help the weights, because the kernel is not waiting on memory
in the first place. The only place SRAM could help is by **moving ALU work into
a cache-resident table** — see §7.

## 6. The ceiling, stated as arithmetic (independent of any measurement)

At 16k: bytes/token = 12.058 (weights) + 2.181 (KV) = **14.239 GB**.
`20 tok/s = 50.000 ms` ⇒ required = 14.239 / 0.050 = **284.8 GB/s**.

- vs **273 GB/s** pin rate → **104.3%** — a kernel cannot outrun the connectors.
- vs 253.6 best arm ever measured → 112.3%
- vs 245 sustained → 116.2%

Even at **100% of the pin rate** 16k yields **19.17 tok/s**, not 20.
Reachable ceiling with zero overhead: **~17.2 tok/s at 16k** (~18.4 at 2k).
Today is ~14.2 at 2k and ~12.3 at 16k.

## 7. Where a real win could still exist (untried, legal)

1. **LUT the MCG decode.** `decode_state_mcg` is a pure function of a 16-bit
   window; the kernel spends ~63% of its time in its IMUL/AND/XOR/half2 chain
   (`MEMORY.md`). A 2^16-entry table (128–256 KB — *SRAM/L2 resident*) replaces
   the chain with a cached load, is **bit-exact by construction**, and is the
   one change that uses on-chip SRAM where the data *is* reused (the table, not
   the weights). crux3's arms B/C say ~11–21% is on the table. **Not yet
   implemented or measured.**
2. **Decode split 16 → 13** (worth 2.13 ms/token at 8k) — opt-in today because
   the key-combine order changes; needs a quality waiver.
3. **Fuse element-wise chains** (post_ffn_out, post_attn_pre_ffn_transform,
   norm_transform ≈ 1.9 ms/step in 1,080 dispatches).

## 8. Target decision this folder cannot make

20 tok/s **at 16k** is above the memory pins by 4.3% and is not reachable by any
kernel, cache scheme, or dispatch shape. The honest options are (a) retire the
20 @16k target and pursue the ~17 ceiling, or (b) lift one of the standing
restrictions (KV quantization, or verified speculation) deliberately. Recorded,
not changed unilaterally.
