# Finding 07 — Audit of the 6.70 ms non-GEMV tax

Subagent: `ses_f0cf01f4cffcwCV0xvzGR7N4pl`. Raw output, lightly de-formatted.
Measurement: `KAPLUMBA_DECODE_PROFILE=kernel KAPLUMBA_DECODE_PROFILE_WARMUP=1`,
127 steps, `prompt_tokens=26`, key_count 27→154. Model `Gxma-repacked`, temperature 0.

## Headline

The 7.45 ms non-GEMV figure is **6.70 ms on the fused path**. Decomposition:

```
6.699 ms = 1.980 ms launch (600 dispatches x 3.3 us) + 4.719 ms kernel execution
                             attention        1.933  (41 %)
                             4x 1-group norms  2.665  (56 %)
                             norm_rope_qkv    0.114  ( 2 %)
                             2x input_trf    0.007
                             cache_append    0.007
```

**Fusion is the lever for 1.98 ms of the 6.70 (30 %). The other 4.72 ms is kernel
execution that no dispatch-count change can touch.** This inverts the comment at
`encoder.rs:300-302`, which reads "~5.4 ms of the non-GEMV tax is launch latency";
it subtracted launch cost *from* own-time and labelled the remainder launch. It
should read: 7.45 ms own = 2.8 ms launch + 4.7 ms execution (fused: 1.98 + 4.72).

## The per-kernel table

`own_ms` = command buffer's own GPU span, `GPUEndTime - GPUStartTime`
(`encoder.rs:706`): launch + kernel + in-buffer bubble, not the gap to next buffer.

| row | disp/tok | bytes/tok | impl@250 | measured | gap | launch | work | work GB/s | AI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| attention | 120 | 157.25 MB | 0.629 | 2.329 | 1.700 | 0.396 | 1.933 | 81 | 3.8 / 12.7 |
| norm_transform | 120 | 4.52 MB | 0.018 | 1.066 | 1.048 | 0.396 | 0.670 | 6.7 | 0.50 |
| pre_ffn_norm_transform | 120 | 4.52 MB | 0.018 | 1.023 | 1.005 | 0.396 | 0.627 | 7.2 | 0.50 |
| post_attn_norm_res | 60 | 3.23 MB | 0.013 | 0.869 | 0.856 | 0.198 | 0.671 | 4.8 | 0.63 |
| post_ffn_out | 60 | 3.23 MB | 0.013 | 0.895 | 0.882 | 0.198 | 0.697 | 4.6 | 0.63 |
| kv_norm_rope | 60 | 6.14 MB | 0.025 | 0.312 | 0.288 | 0.198 | 0.114 | 54 | 1.40 |
| cache_append | 60 | 0.90 MB | 0.004 | 0.205 | 0.201 | 0.198 | 0.007 | 133 | 0.25 |
| **TOTAL** | **600** | **179.8 MB** | **0.719** | **6.699** | **5.980** | **1.980** | **4.719** | 38 | |

Launch cost 3.3 us/dispatch is *derived*, not assumed: `cache_append` is
3.41 us/stamp for 0.12 us of bandwidth-implied work; corroborated by
`chunk/copy_last_row` (2.70 us, one dispatch, zero work) and `chunk/embedding`
(3.70 us). Agrees with the recorded 3.5 us marginal.

Dispatch inventory validated statically: 1168 counted vs 1167 recorded
(`tasks.md:111`) — one thread off over 1168, inventory correct. **The seven rows
hold 600 dispatches/token in 420 command-buffer stamps.**

## The key structural fact: 240 of 600 dispatches are ONE threadgroup

`rms_norm_weighted_tg`, `rms_norm_residual_tg`,
`rms_norm_residual_layer_scale_tg` and the `1x256` `linear_input_transform` all
dispatch **1 group x 256 threads** (`elements.rs:739-745`, `:260`). That is one
of twenty cores, 0.6 % of a wave.

| kernel | groups | groups/core | vs a 160-group 256-thread wave |
|---|---:|---:|---:|
| `rms_norm_weighted_tg` (x2/layer) | 1 | 0.05 | 0.6 % |
| `rms_norm_residual_tg` | 1 | 0.05 | 0.6 % |
| `rms_norm_residual_layer_scale_tg` | 1 | 0.05 | 0.6 % |
| `linear_input_transform` (x2) | 42 | 2.1 | 26 % |
| `norm_rope_qkv` | 64 / 40 | 3.2 / 2.0 | 40 % / 25 % |
| `cache_append` | 64 / 32 | 3.2 / 1.6 | 40 % / 20 % |
| `attention_decode_split_256_2` | 48 | 2.4 | 19 % |
| `attention_decode_split_512_8` | **12** | **0.6** | 5 % |
| `attention_decode_combine_512` | 256 | 12.8 | 160 % |

Independent confirmation the cost is the kernel, not the launch: `head/final_norm`
(`model.rs:3623-3632`) is the *same kernel at the same shape*
(`rms_norm_weighted_tg`, rows=1, width=5376), dispatched once per step, measures
**16.7 us alone**. `norm_transform`'s two dispatches together are 17.77 us, so
the 42-group `linear_input_transform` beside it costs ~1.1 us — already at floor.

Mechanism (`elements.metal:159-174`): each of 256 threads walks
`for (i = lid; i < 5376; i += 256)` — 21 loads at 512 B stride — then 8
`threadgroup_barrier`s for the 256→1 tree, then a second 21-load pass plus 21
stores. With 8 warps on one core there is nothing to hide 21 round trips behind.
21 x ~500 ns ≈ 10.5 us against 11.2 us of measured work. **A latency chain, not
a bandwidth limit** — AI of 0.5–0.63 FLOP/B confirms there is no bandwidth to win.

The fp32 SIMT ridge point is 8.76 TFLOP/s ÷ 250 GB/s = **35 FLOP/B**
(`spec/engine-performance/implementation_notes.md:68`, `examples/alu_peak.rs`).
Every row is 9–140x below it. **Not one row here is ALU-bound, and not one is
bandwidth-bound** — measured work bandwidth runs 4.6–133 GB/s, 2 %–53 % of peak.
These are latency problems wearing bandwidth's clothes.

## Why the norm grids are frozen

`implementation_notes.md` §20.5 records `partial[lid] = (lid==0) ? sum : 0`
changing 8192/8192 elements, worst 9266 ULP. **The 256-lane strided partial +
tree IS the reduction order**, and `rsqrt` of it reaches the residual. Any
change to the lane→element map, the threadgroup width, or a two-stage split
changes the sum and therefore the output.

**For these four kernels the grid is frozen by the no-quality-loss rule.** Fusion
can only collect the 3.3 us launch on each: **0.79 ms/token** across all four,
not the 2.67 ms.

## Ranked by ms saved if instant / risk

| # | change | ms/tok | risk | irreducible? |
|---|---|---:|---|---|
| 1 | **`cache_append` -> fold into `norm_rope_qkv`** | **0.21** | LOW–MED | bytes only 0.004 |
| 2 | hoist `weight`/`residual` loads above the barrier in the 1-group norms | 0–1.2 | LOW if bit-exact; **may be 0** | grid widening is NO (order) |
| 3 | T7.6.12: post_attn_norm_res + pre_ffn_norm + gate input transform -> 1 dispatch | 0.42 + 0.11 | MED | no |
| 4 | cross-layer: fold layer N+1 `input_layernorm` into layer N `post_ffn_out` | 0.21 | MED | no |
| 5 | T7.6.11: down finalize + post_ffn_out | 0.21 | MED | no |
| 6 | T7.6.13: attention single-dispatch when `splits == 1` | 0.21 short ctx | MED | the 1 dispatch is |
| 7 | **attention: raise `splits` 3 -> 16 at short context** | up to ~1.2 | **HIGH — changes summation order over keys** | **YES, effectively** |

**Rank 7 is the honesty item.** The single largest recoverable-looking item in
`attention` — 1.7 ms of occupancy tax, 25x the remaining fusion budget — is
numerically frozen. The grid is 12 groups on full layers because `decode_splits`
divides `ATTENTION_DECODE_THREADS_TARGET` by `32 x q_heads` then clamps to
`key_count / 64` (`attention.rs:67-69`); at key_count=154 that yields 3. Raising
it fills the machine and changes the answer.

## Rank 1 in detail — the best ms/risk in the set

`cache_append` grid: `dispatch_1d(tokens x stream_len = 1 x kv_dim, 64)` →
64 groups x 64 threads (sliding), 32 x 64 (full). Bytes/layer: read k 8192 + read
v 8192 + write k 8192 + write v 8192 + 4 = **32 772 B**. Bandwidth-implied
**0.12 us/stamp**, measured **3.41 us** → **97 % of the row is launch**.

Its working set is not the reason. Folding it into `norm_rope_qkv` is close to
free and better than the dispatch arithmetic suggests: `norm_rope_qkv` already
has one threadgroup per (Q head | K head | V head) (`elements.rs:1054-1065`),
the V groups already write `target[base + i]` for exactly the elements
`cache_append` copies (`elements.metal:726`), and the destination is a pure
address relabel `key_cache[slot*stream_len + head_idx*head_dim + i]` with `slot`,
`capacity`, `position_start` already host-side at `model.rs:1660-1674`. It
replaces an 8192 B store with an 8192 B store elsewhere — **zero extra groups,
zero extra bytes, one fewer dispatch.** It also deletes the `scratch.k`/`scratch.v`
staging writes entirely, because attention reads the cache not the scratch
(`model.rs:1713-1714`). Order is legal: `norm_rope_qkv` at `model.rs:1570`
already precedes `cache_append` at `model.rs:1662`, and the host `reserve()`
happens between them. Needs a real bit-exactness test and a `slot`/ring-wrap test.

## Rank 2 in detail — the one unmeasured bit-exact idea

The second pass reads `weight` and `residual` (`elements.metal:173-174`), which
do **not** depend on `inverse` and could be loaded into registers before the
barrier. The first loop writes only `partial[]` (threadgroup), so nothing in
global memory is written between the two passes and the hoist is legal. Whether
the compiler already does it is unknown; if it does, the win is 0. Potential
0.6–1.2 ms. Existing `norm_rope_qkv_bit_exact.rs` / `live_kernel_bit_exact.rs`
are already sabotage-verified and can gate it. **Do not budget before measuring.**

## Irreducible floor

**At K ≈ 154:** mandatory bytes 0.72 ms + minimum realistic dispatch count.
Folding each layer to `post_attn_norm_res+pre_ffn_norm+gate_trf` (1),
`norm_rope_qkv+cache_append` (1), `attention` (2), `post_ffn_out` (1),
`norm_transform` (1, absorbed into previous layer's `post_ffn_out`) = 6
dispatches/layer = 360/token = **1.19 ms of launch**, plus the norms' frozen
~2.67 ms of 1-core latency, plus attention's frozen ~1.3 ms of occupancy.
**Floor ≈ 2.6–2.9 ms, of which ~4.0 ms is addressable** — 2.0 ms by fusion
(#1, 3, 4, 5, 6) and up to 2.0 ms more by the two unmeasured kernel-level items.

**At sliding-full + 8k full:** the seven rows carry **1 604 MB/token = 6.42 ms
@250 GB/s, 8.44 ms @190 GB/s** for attention alone. The non-GEMV tax stops being
a 0.8 ms problem. If long-context decode matters, the order of work should be
attention bandwidth, not fusion.

## Unresolved

- **`kv_norm_rope` 0.88 (parent) vs 0.31 (this run).** 2.8x. Either the parent
  table was taken on the unfused path (5 dispatches/layer, `model.rs:1568`,
  `KAPLUMBA_NO_NORM_ROPE_QKV`) or it came from
  `implementation_notes.md:552` (0.8598/0.8622), itself 2.8x above the fused
  measurement. 1407 − 1167 = 240 = 4 x 60, so **if the parent table was unfused,
  0.75 ms of the 7.45 is the 240 dispatches T7.6.10c already removed** and the
  shipped figure is 6.70. **Needs one `KAPLUMBA_NO_NORM_ROPE_QKV` A/B before
  spending anything on this row.**
- **`KAPLUMBA_SKIP_HIDDEN_NORMS`** would give a direct in-situ delta for the
  1-group norms (expect `norm_transform` 1.07 → ~0.15 and
  `pre_ffn_norm_transform` 1.02 → ~0.10). One A/B confirms or kills the 2.67 ms
  bucket.
- **Per-buffer vs per-dispatch launch.** `cache_append` (3.41 us) supports
  ~3.3 us/dispatch and closes the total to 6.699 exactly. But `head/final_norm`
  at 16.7 us for one dispatch implies the other model — one ~3.0 us per buffer
  plus ~0.8 us per additional in-buffer dispatch, which also fits all ten rows
  to within 3 %. The two differ by ~0.2 ms on the total and change no conclusion,
  but profiled own-time and shipping marginal cost are not literally the same.
  `dpk_span` 91.6 vs `dpk_own` 70.7 is 20.9 ms of inter-buffer gap = 16 us/buffer,
  which is the profiler's own `wait_until_completed` serialisation, not the
  shipping path — confirms `kernel` mode is an attribution tool only
  (`decode_profile.rs:26-28`).
- `encoder.rs:300-302` is wrong as written. It is the sentence most likely to be
  read as "there is 5.4 ms of bubble available"; there is 2.0.

No edits made. Scratch files under the session temp dir; background retry loop
killed; no model process left running.
