# Finding 08 — Fusing o_proj's transform into attention: real, but 0.1 ms, and it corrects the baseline

Subagent: `ses_f0ceeff86ffda6HkegsjqzBoAw`. Branch `spec/t7.6.14-fused-attend-oitf`
@ `7145fd2`, 3 commits on `master` (`a389490`). `KAPLUMBA_FUSE_ATTEND_OITF`,
**off by default**; old path untouched.

## What it built

`attention_decode_combine_transform_{256,512}` — one dispatch replacing
`attention_decode_combine_*` → `scratch.attended` **plus**
`linear_input_transform` → `scratch.transformed`. Eliminates **60
dispatches/token**.

## The flat range remaps with no barrier and no tile

The combine runs one thread per (head, dimension); the butterfly needs one
*simdgroup* per 128-element block. Blocks start every 128 flat ids, 64-thread
groups every 64 — so **half the existing simdgroups straddle a block boundary**
and a width-1 shuffle mixes blocks. Switching to 32-thread groups alone does
**not** fix it; the grid is the problem.

**The fix is arithmetic: each thread takes four of the elements it already
computes (`base, +32, +64, +96`)**, so the grid becomes `q_heads*head_dim/4` at 32
per group. `head_dim % 128 == 0` keeps all four in one head, so one division
serves them. Both shapes verified (8192/16384 elements, 64/128 blocks).

## Two bugs the canary caught

1. **`float(half(value))` is load-bearing.** The separate path **stores fp16** and
   the transform **reads it back** — the butterfly sees a rounded value. Keeping
   fp32 is a different function: **3461-7005 of 8192/16384 elements differ by 1
   ULP.** Same trap as `norm_rope_qkv`.
2. **4x output overrun**: `base = 128b + lane`, not `gid`. Canary: *"wrote
   element 8192, past the 8192 it was given."*

## Sabotage: 6 of 8 fail — **the 2 that don't are the point**

- Dropping the output-side `half()` is a **no-op**: `output` is `device half *`,
  Metal converting-stores. A test that cannot fail here proves nothing.
- A wrong butterfly in `hadamard.metal` is compiled into **both arms** — this test
  **provably cannot see it**. `input_transform.rs` does (2688/5376 sign-flipped).
  **Together complete; either alone is half a claim.**

## Measurement — and a correction to my premise

| | |
|---|---|
| dispatches/token | **1617 → 1557, exactly −60**, all 77 runs |
| wall clock, 32 pairs | +0.22%, **t = 1.32 on 31 df — not significant** |
| per-kernel | `o_proj` 7.21→7.05 (−0.156), `attention` 2.43→2.48 (+0.051), net **−0.105 ms/token**, non-overlapping |
| bit-exact | 77 runs, **one** id-list sha256 across two bases 5 commits apart |

**The 3.5 µs/dispatch constant is confirmed, and that is the finding.** 0.21 ms
predicted, 0.29 observed, agreeing to 0.03 ms — but **~0.3 % is below what 32
pairs resolve here** (per-pair sd 0.9 % even after pairing). This is **~0.3
ms/token, not a 1 %-class win.** Unlike `norm_rope_qkv`, this one also saves real
work (1.12 MB store + reload per token), which is why the per-kernel delta is
clean even though the wall clock is not.

## ⚠ BASELINE CORRECTION — the dispatch counts in every doc are stale

**T7.6.10c's 1407/1167 is now 1857/1617 on both bases — constant +210. Deltas
intact (the 240 reproduces exactly).** Re-derive before spending the remaining
~0.8 ms of fusion budget.

## Two things about master

- **Cherry-picking onto master reports no conflicts and does not compile** —
  master's `encode_decode_split_with_splits` refactor drops the `input_signs`
  parameter. The agent threaded it properly and verified.
- **Master's refactor opened a real hole**: its single-pass arm returns *before*
  the combine, so with the flag on it would hand `o_proj` an **untransformed
  buffer**. Now refused, with its own test binary — the in-process version passed
  against an `Ok(())` because of `OnceLock` caching. **Sabotaged → fails.**
- **`master` as committed does not build** — `lib.rs` declares `pipeline_audit`
  and `sweep`, whose files are untracked. **A fresh clone is broken.**

## Assessment

Bit-exact, correctly gated, off by default, and it removes 60 dispatches for
~0.1–0.3 ms. The wall-clock effect is not resolvable at this box's noise floor, so
this is **not a 1 %-class win** and must not be counted as one. Worth landing for
the dispatch reduction and the 1.12 MB saved; not worth claiming as throughput.
