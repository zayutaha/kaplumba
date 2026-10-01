# MEMORY — kaplumba decode optimisation

Read this first. It supersedes conversational history.

## THE HEADLINE

**19 tok/s is arithmetically impossible at 3.0 bpw and this is now closed.**

19 tok/s = 52.632 ms. Weight traffic 12.039 GB. At the measured 250.5 GB/s
ceiling the GEMV alone is 48.06 ms, leaving 4.57 ms — but non-GEMV kernels alone
measure 7.0–10.1 ms. **19 tok/s demands 264–283 GB/s, i.e. a memcpy-beating
GEMV.** Even a *perfect* GEMV caps at 17.2–18.2 tok/s.

**2 bpw is also closed** (`findings/02`): `rate ∝ ka/lane_words` so
`time ∝ K·N·LW/8` — **bpw cancels exactly**. tok/s is a step function of
`K·N·LW` and 3.0 bpw already sits on the bottom step (13.38 vs today's 13.05).

15 tok/s is a labelled **stretch**; 13.67 is the commitment. Decided by me, not
the user (standing instruction: ask nothing).

## CURRENT STATE

- **master is BROKEN and must not be built from a fresh clone.**
  `crates/kernels/src/lib.rs` declares `pipeline_audit` and `sweep`; the first
  exists only in commit `703603a` (on `integration/rolling`, not master), the
  second is untracked. Every engine build fails with E0433s.
- The **shared tree at `/Users/zayaantaha/coding/kaplumba` has ~15 agents' live
  uncommitted work and 28+ dirty files.** Do not `git checkout --`, `git add -A`,
  or `cargo fmt --all` there.
- **All new work goes in a dedicated worktree with its own
  `CARGO_TARGET_DIR`.** A worktree sharing `target/` gets its binary rebuilt
  mid-benchmark by another agent — that destroyed one measurement already.
- Findings 01–11 are written up under
  `spec/decode-19tok-repacked-250gb/findings/`. Read those, not this file, for
  the numbers.

## LANDED AND MERGED ONTO `integration/rolling` (11 commits, tree clean, AC6 holds)

**Command-buffer rolling: 13.558 → 14.150 tok/s, +4.37%.** `set_max_dispatches`
2500 → 50 (`MAX_DISPATCHES_PER_BUFFER`). Step was encoding into ONE uncommitted
command buffer, so the GPU sat idle 5.75 ms while the host marshalled
parameters, then ran 72.7 ms blocked. `MAX_IN_FLIGHT=8` and all back-pressure
were dead code. 16 paired samples, t=6.13/14df, sign test p=0.000488.
Buffers/step 1.00 → 24.00; dispatches/token unchanged (the control proving only
commit boundaries moved). GPU clock agrees: span 72.929 → 70.239 ms.

Also landed: `pipeline_audit` restoration (`703603a`), and the fused-QKV
de-shipping switch `KAPLUMBA_NO_FUSED_QKV` / `KAPLUMBA_NO_FUSED_GATEUP`
(**default OFF** — end-to-end A/B never ran).

### TWO FAILING TESTS ON `integration/rolling` — must be resolved before merge

1. `every_pipeline_construction_goes_through_the_audit` — committed `linear.rs`
   has **19 direct pipeline constructions, zero `pipeline_audit::build` calls**.
   Functional, not cosmetic: `generate.rs`'s `assert_no_builds_since` guard
   cannot see those builds. Fix: route all 19 through the audit, or drop the
   module.
2. `the_decode_instrument_measures_a_step_and_only_a_step`
   (`decode_profile_gpu.rs:394`) — a prior block of 2400 dispatches now moves the
   decode total 7.40%. **This is caused by the rolling change.** Either the test
   needs new exclusion logic, or `dp_span` is no longer a valid attribution
   metric with rolling on. The second possibility matters: every attribution
   number in this project uses that instrument. NOT YET DIAGNOSED.

## THE BIGGEST UNEXPLOITED FINDING

**The GEMV is ALU-bound, ~8:1 against memory — not load-issue bound.**

Measured partition of `linear_gemv_ishizuki_k3_colmajor_contig4`:
**8% load-issue / 63% decode-ALU / 30% neither** (additive to 101%). Not the
46/54 the project has assumed.

**The 64-bit barrel shift `uint(A >> (64u - rad))` in `decode_state_mcg` is
65.6% of the kernel.** Pinning `rad` to a constant: **0.459 → 0.125 ms, 3.6x.**
`decode_state_mcg`'s body itself is free (−2.3%).

**The loop is not bandwidth-bound at all:** arm (d) keeps every load and every
address but reads a 129 KB window instead of 2.6 GB → **−0.6%.** So "187.7 vs a
250 GB/s ceiling" is NOT a bandwidth shortfall — the kernel is not waiting on
memory. The 4 bpw result (more bytes, faster) is therefore **not** evidence of
load-issue boundedness; 4 bpw is faster because it has fewer elements to
barrel-shift.

⇒ Split-K, `KAPLUMBA_GEMV_MAX_GROUPS`, and fewer-lane-loads-per-byte all attack
~8%. The only lever left is **layout** — and the bit-plane repacker is LOSSY
(keeps `s & 0x7` while `decode_state_mcg` consumes a 16-bit window of which 13
bits survive; the `imul` mixes all 16 *before* masking, so it produces
*different dequant*, not imprecise dequant). Barred by no-quality-loss, not
merely unpromising. **Do not revive it.**

The earlier 46/54 partition came from a probe that fed the decode from
loop-invariant literals, so LICM hoisted the decode out of the tile loop — it
measured loads gone AND decode gone. Same failure class as AGENTS.md §3.4's two
false coverage claims.

## QUEUED, NOT YET IMPLEMENTED

- **attention splits: 13, not 12** (`impl/attn-splits`, `8852a16`).
  `decode_splits` never receives `kv_heads`, so both shapes get 16 splits with
  grids differing 4×. Rule is `54 / kv_heads` (54 workgroups = 1 wave, 55 = 2) —
  reproduces the measured optimum for kv_heads 4/5/6, so a rule not a fit.
  The tidy "20 cores × 3 = 60" gives 15, measured **1.34× slower than the 16
  that ships**. Full layers @8k: 16 splits 80.3 GB/s → 13 splits **107.8 GB/s
  (1.34×)**. Sliding provably unchanged (budget 288 wg = 18 splits, above cap).
  **BUT the 128-token golden gate is structurally blind** — at 156 keys the
  keys-per-split floor caps both arms at 3 splits, so it reports a pass. At 6010
  tokens: **43 of 128 tokens differ, first divergence at token 86, from 1 fp16
  ULP on 3–4 of 16384 elements.** Default off, `KAPLUMBA_DECODE_SPLIT_RULE=workgroups`.
  Worth 2.13 ms/token at 8k (isolated lower bound).
- **fold `cache_append` into `norm_rope_qkv`** — 0.21 ms, best ms/risk in the
  set. The V groups already write exactly those elements; pure address relabel,
  zero extra groups, zero extra bytes, one fewer dispatch. Also deletes the
  `scratch.k`/`scratch.v` staging writes.
- **hoist `weight`/`residual` loads above the barrier** in the 1-group norms —
  legal, unmeasured, may be 0 if the compiler already does it. Potential
  0.6–1.2 ms.
- **device-side argmax** (`crates/kernels/tests/argmax_device.rs`) — `greedy_key`
  packs `0xFFFF - index` into 16 bits, so for index ≥ 65,536 the underflow
  **clobbers the logit field**; max over a 262,144 row is deterministically
  index 0. Measured host 9,259 / device 0. Bit-exact on every non-degenerate row
  (65,574 rows × 4 grid shapes). But only **+0.35%**, not the 8.9 ms.
- **4 bpw ceiling number** — still unmeasured after 3 attempts, GPU never freed.
  Harness: `bench_rate_sweep --rates 3,4,6`, ~12 s of GPU, needs `gpu: none`.
  This decides whether the GEMV ceiling is reachable at all.

## NON-GEMV TAX: 6.70 ms, and most of it is not addressable

Split 1.98 ms launch / 4.72 ms kernel execution. **240 of 600 dispatches/token
are a single 256-thread threadgroup** (one of 20 cores) — 2.67 ms of the 4.72.
That grid is **frozen**: the 256-lane strided partial *is* the reduction order
(sabotage moved 8192/8192 elements at 9266 ULP). Only the 3.3 µs launch each is
collectable: **0.79 ms, not 2.67.** Attention's 1.7 ms occupancy tax is frozen
too — raising `splits` changes the summation order over keys.

Measured floor at K≈154: **~2.6–2.9 ms**, of which ~4.0 ms of the 6.70 is
addressable.

## PREFILL

- AGENTS.md §8 **holds**: flash-style online softmax, O(1) scratch, no
  `[heads, L, L]` anywhere, scratch stays 0.003 GiB after a 2-chunk prefill.
- Tensor-core path confirmed live and **8.0–8.5× faster** than the scalar
  fallback. Regpipe worth 4%.
- Prefill is **tensor-pipe-bound, not bandwidth-bound**: 42% of tensor peak,
  **3.7% of memory bandwidth utilised**.
- **`prefill_tok_s` is a metric-definition bug.** It is a *staircase in L*
  because `linear_gemm_mma` dispatches `ceil(L/128)` threadgroups and a
  128-query tile costs the same at 6 rows as at 128. 120 → 134 tokens is 12%
  more work and **halves the reported rate**. Meaningless below ~300 tokens;
  must be reported with L, L mod 128, and `ms − fixed`. Real run-to-run variance
  is only 0.02–1.2%, so the 6.3–8.2 spread was never drift.
- `phys_footprint` 12.95 GiB at 8k; growth 134→7814 is 1.38 GiB = **entirely
  KV, zero scratch**. Reproducible to 0.01 GiB. Confirms ru_maxrss understates
  by 3.87 GiB at 2k (AGENTS.md §10.3).

## WARMUP: NO GAP

173 pipelines / 157 distinct kernels / 12.5 ms, **all at load, `late_builds=0`**
through prefill and 48 decode steps. `--metrics` adds no per-token readback or
sync. Largest single build 1.5 ms. **Extending warmup to prefill was tried,
measured, and reverted** — the ~3 s first-forward cost is not prefill-kernel
first dispatch.

## 4 bpw CHECKPOINT (`~/Downloads`)

Shimmable and its kernels are correct at 4/8 bpw (real-weight err/RMS 1.2e-6,
not bit-exact vs CPU — fp32 split across 8 simdgroups). **But it does not
complete a decode step**: `Metal command buffer failed` in the split-KV
attention path, localised with `KAPLUMBA_SKIP_ATTENTION=1`. Not the bitrate.
Loader change was needed, not a file: `config.rs:253` pinned `bits==3.0 &&
head_bits=6` as constants. Fits (15.06 GiB anon, 8.94 GiB headroom) but leaves
no room for a second process.

## RULES LEARNED THE HARD WAY HERE

- **`cargo` does not always rebuild after editing an `include_str!`'d `.metal`.**
  Confirm `Compiling kaplumba-kernels` appears. A green run on a stale binary is
  a false pass.
- **A Metal compile failure can be a silent `else { skip }`** — one missing
  paren produced a green test and a bandwidth number from a library lacking the
  kernel. The `.metal` files concatenate into one library, so a truncated
  signature surfaces as `KernelError::Shader` and *every* kernel test panicking
  at its own `.expect`, reading as "the kernels don't work".
- **The first block/shape measured in a process reads its own dirty scratch
  pages** and can read 3× slow. `fsync` + real warmup + calibrate from the
  slower of two settled readings.
- **ABBA, not AB.** At a 1–8% effect size a fixed order charges the same arm
  every round and invents results.
- **Unpaired samples spread 4–12%**; use 16 *paired*. Report contaminated pairs,
  do not drop them.
- **A dispatch costs ~3.3–3.5 µs of latency and ~0 of work** — confirmed twice
  (`dpk_own` 71.649 → 71.644 across the whole `norm_rope_qkv` change). Every
  fusion estimate in the old roadmap assumed it saved kernel time; that is the
  most repeated error in this project's history.
- **`float(half(x))` is load-bearing** wherever a store/reload round trip exists
  in the unfused path. Appeared twice.
- **Use `min`, not median**, for kernel timing on this box: `gpuq` excludes the
  engine's flock but not a benchmark harness, so contention inflates medians
  (once 250% off).
- **`scripts/gpuq` changes directory** — always pass absolute paths to your
  binary, or both arms "match" on the same error string.
- A **golden hash is meaningless without the exact command beside it.** The
  recorded `2fff2c97…` needs model `~/.kaplumba/models/Gxma-repacked` and the
  prompt "Explain the theory of relativity in simple terms, and then explain why
  it matters." A different prompt gives a different, perfectly well-formed 128
  tokens that reads as a regression.

## STALE NUMBERS — DO NOT QUOTE

- Dispatch counts are **1857/1617**, not 1407/1167.
- `kv_norm_rope` 0.88 vs 0.31 ms unresolved (2.8×).
- `dp_wall_ms_per_token` is **impossible** for a barrier-terminated step
  (27.568 vs a 70.239 span). Do not quote it.
- `encoder.rs:300-302` "5.4 ms of launch latency" is wrong: 7.45 own = 2.8 launch
  + 4.7 execution.
- `linear.metal`'s `j2 = ... : j0` should be `: j1+1` — latent, unfixed.
- The bit-plane repacker has been nominated as "the remaining lever" by one
  agent and refuted as LOSSY by another. **Refutation wins** — it is a
  correctness bar, not a speed one.

## GPU DISCIPLINE (AGENTS.md §10)

24 GiB unified, no discrete GPU. ONE model process at a time, enforced by a
kernel `flock`. Two concurrent instances exceed physical memory: measured 3.9 GiB
of swap and a collapse to 1.4 tok/s, which invalidates every benchmark. Check
`scripts/gpuq status`, then `try`; **exit 75 = busy, retry later**. Never
bypass. Never block for 30 minutes.
