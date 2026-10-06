# Spec: Decode ALU/Fuse to Honest Ceiling — ~17.5 / ~18.5 tok/s

## 1. Goal

Raise sustained autoregressive single-token decode throughput from **16.05 tok/s greedy / 14.37 tok/s at shipped default `--temperature 1 --top-k 64`** (`WHERE_THIS_STANDS.md`, `generate.rs:383` metrics bracket excludes sampler — `16.05` overstates default by `~1.7`) toward the **honest pure-decode ceiling** on M4 Pro 20-core / 24 GiB for Gemma 4 31B (`Gxma`, EXL3 3.0 bpw) without changing `tokens/byte`.

**Quantified ceiling (zero-overhead bound, `bytes/token / BW`):**
`bytes/token @16k = 14.2390 GB` (`12.0580` weights exact `0 B` delta `crux.md §1.2` + `crux2.md §1` + `2.1810` KV) over `239.8` GEMV / `245` sust / `273` wire = `16.84 / 17.21 / 19.17` tok/s. Normal chats `@2k` `13.0646 GB` → `18.35 / 18.76 / 20.9`. So **`22.5 normal / 20 @16k pure forward = 283–320 GB/s = 103–117% of wire` — arithmetically excluded** (`crux.md §3.1`, re-derived in this spec §6). This spec banks the remaining **legal ALU and fuse** and records the gap.

**Targets in this spec:**
* **Commitment (bit-exact greedy): `≥ 17.2 tok/s` median** of `≥16` interleaved pairs, `≥128` tokens, `Gxma-repacked`, greedy byte-identical.
* **Stretch (1 fp16 ULP waiver on attention splits only): `≥ 17.6 tok/s`**, same method, waiver recorded in `spec.md §2`.
* **Default-temp parity: shipped default reaches greedy parity** (today `14.37 → ~16.05`) via device sampler T3 wiring alone — not beyond.

`22.5 / 20` remain out of scope for pure forward; `tokens/byte` is the only route past the pins and is explicitly not in this spec (§2).

Current measured state against which to A/B: `F4` `16.021 greedy` (6 pairs, quiet box) / `WHERE_THIS_STANDS 16.05 greedy`. Baseline to be re-pinned in one session with interleaved control per `AGENTS.md §10.2.5`.

## 2. Non-goals & restrictions

* **No speculative decoding / MTP / draft / lookahead / n-gram.** Branch `mtp-speculation` stays OFF. Every token executes full 60-layer forward.
* **No lookup tricks / retrieval / caching of generated text.**
* **No KV cache quantization** (`AGENTS.md §8`). KV at 16k is `2.1810 GB` immovable; any roofline quoted weights-only understates bytes by `15.3%`.
* **No quality loss except the explicit split waiver.** Greedy output byte-identical by default; `KAPLUMBA_DECODE_SPLIT_RULE=workgroups` (13 not 16) changes `1 fp16 ULP` on 3–4 of 16384 elements (`MEMORY.md: 43/128 diverge @6010`, `decode14-prefill100 §10.5 434/476 @8204`) — shipped **opt-in** behind flag, default unchanged unless waiver signed in this spec.
* **No bit-rate change.** `3.000 bpw` `410` tensors + head `6.000` exact (`crux2 §1.3`). `2–4 bpw` byte-rate invariant `62.3 ms` plateau for all `1/2/3/4` layers at 6b head (`WHERE_THIS_STANDS P19`, `3bpw-amplification §3 37-bit window floor`) — not a lever.
* **`~/.kaplumba/models/Gxma/Gxma` untouched.** One repacked copy `Gxma-repacked` at most.
* **No Python in engine/product code** (`AGENTS.md §8`). `scripts/gen_exl3_gemv.py` is build tool only; `linear_repacked.metal` lives outside its rewrite window (`F4` / `AC6` `git diff` null).
* **Single-user batch-1 only.**
* **GPU exclusivity:** `scripts/gpuq status` then `try`; exit `75` = busy, retry. One model at a time, separate `CARGO_TARGET_DIR` in worktrees (`AGENTS.md §6 / §10.2.4`).

## 3. Requirements

* **R1 — Tile-loop 4-deep unroll (claimed +5–9%, bit-exact; claim is UNVERIFIED and its instrument is lost).**
  `spec/alu-boost/findings.md §2` reports `arm_decode_u4` = same tile assignment, same per-thread accumulation order, tile loop unrolled 4-deep with stride, `0 differ / 2560 words`, `1.095×` and `1.052×`.

  > **Amendment, verified against the tree before implementation. Three things about this requirement were wrong as first written.**
  >
  > 1. **The instrument does not exist in this tree.** `crates/engine/examples/alu_boost.rs` is absent from HEAD `517bc4e`. It was added in `33e8ead` (on `agent/tensor-ops`, 781 lines) and never deleted — HEAD simply does not contain that commit. Per `AGENTS.md §7`, a path was cited before it was checked. **The +5–9 % is a claim in a markdown file with no runnable instrument behind it in the working tree, and is therefore not a result until T1 re-measures it.** T1 restores `alu_boost.rs` from `33e8ead`, re-runs both arms, and must reproduce `0 differ / 2560` **and** a ratio above the noise floor before any porting. If it does not reproduce, R1 is dead and is recorded as such.
  > 2. **Do NOT touch `linear_gemv_ishizuki_k3_colmajor_u4`.** The name collides and the existing kernel is a different, already-negative experiment. `_u4` (`linear_repacked.metal:3193`) is a **threadgroup-staged `uint4` 128-bit load** — it stages tiles through `tg_tiles[SG][96]` with a barrier in the K loop — and it measures **180.5 → 173.7 GB/s**, i.e. a **loss** (`perf_probe_kloop_loads_in_flight.rs:137`, `probe_vecld_window_widening.rs:18`, `probe_reduce_epilogue.rs:246` `BARRIER_IN_K_LOOP`, `probe_cache_hierarchy.rs:745-752` which *asserts* `_u4` is the 128-bit-load experiment). `arm_decode_u4` is **not** a wider load — it is a loop-shape unroll at the **same** load width. The new variant is named `_tl4` (tile-loop 4) to keep the distinction mechanical. `alu-boost §3` independently confirms width is not the wall (`uint1` 238.5 vs `uint4` 244.3 GB/s, `+2.5 %`).
  > 3. `linear.metal:121` is `decode_state_mcg`, not a kernel. Port targets are `linear_repacked.metal:79` (`_colmajor`), `:434` (`_fused2_2col`), `:320` (`_fused2`), `:2429` (`_colmajor_2col_sk2`), `:1992` (`_colmajor_contig4` — the kernel `linear.rs:2091` says the engine actually reaches).
* **R2 — `linear_input_transform:475` fuse (eliminate 60 transforms).** `42 blocks × 128` `transform_simd:402` (`1,2,4..64` `simd_shuffle_xor`) reads `input`+`input_signs` → `half`, GEMV re-reads. Fuse into GEMV `x` load path (threadgroup half / half regs, same shuffle order). Saves `60 × 5376 half` round-trip (`0.65 MB/tok`) + `60 disp × 3.5 µs = 0.21 ms` (`repacked-250gb §9`, `dpk_own 71.649→71.644` after 240 disp). Bit-exact same order. `+0.15–0.25`.
* **R3 — Device sampler T3 wiring (Q3 `Yes only temp`, `+2` at default).** Open in `spec/device-sampler/tasks.md` T3.

  > **Amendment: the plumbing is mostly already in the engine; the scope is the CLI gate, not the engine.** Verified: `forward_step_pipelined` (`crates/engine/src/model.rs:4004`) **already takes `pick: HeadPick`** and writes into `host_tokens`; `model.rs:4057 read_host_token`, `model.rs:4071 device_sampler_supported(config)` already exist; `sample_topk_blocks`/`sample_draw` are built (`elements.rs:343-344`), encoded (`model.rs:4183`/`:4192`), and `scratch.sample_candidates`/`rng_state` are already in `resident_bytes()`. So the remaining work is: **`crates/engine/src/chat.rs:388`** — `let pipelined = self.greedy && std::env::var_os("KAPLUMBA_NO_PIPELINE").is_none();` — widened to `greedy || device_sampler_supported(config)`, plus `generate.rs`'s gate at `:971`.
  >
  > **This edit breaks a passing test, on purpose:** `tests/perf_probe_host_ops.rs:373 temperature_above_zero_disables_the_pipelined_path` asserts that exact source line (`:377` searches for `let pipelined = self.greedy`) and `:1099` in `perf_probe_concurrency_dependency_graph.rs` asserts the same. Both must be **rewritten to assert the new gate's meaning** (temperature > 0 no longer implies non-pipelined) — not deleted. Deleting the guard is how this regression class gets reintroduced.
  >
  > Gate cost today is measured: `~3.2 ms/69.6 ms` (`crux3 §7b` `14.741/14.763` pipelined-greedy vs `8.025/14.208` host), and `chat.rs:369` reports `14.97` chat vs `15.88` pipelined on the same binary. `@temp 0` `0`; `@temp 1 top-k 64` `+1.6–2.0` **to parity, not beyond**. `KAPLUMBA_NO_PIPELINE` and the host-sampler path stay for A/B; `top_k` beyond the candidate staging capacity must still fall back to the host sampler (`model.rs:4064-4070`), not silently mis-sample.
* **R4 — Attention split waiver path (opt-in, 1 ULP).** Implement behind `KAPLUMBA_DECODE_SPLIT_RULE=workgroups` (`54/kv_heads` → 13 splits = `52 WG` inside `54` wave vs shipped `16×4=64 WG` cross-wave `attention.rs:486`, `decode14-prefill100 §10.4`). Flash `113 GB/s` vs `193 ideal` `+4.89 ms` on 10 full, `+0.60` on 50 sliding; isolated `1.34× @8k` → `@16k +4.1%` vs end-to-end `+2.17%` contradiction to be settled by 5-point sweep `flash_kernel_ab` `splits {4,8,12,13,16}` (§10.6). Default OFF.
* **R5 — Residual fuse budget.** `post_attn_pre_ffn_transform`, `cache_append → norm_rope_qkv` (`MEMORY.md: 0.21 ms`), norm hoist above barrier. Each `3.5 µs` latency `~0` work, total `~0.3 ms` `+0.07–0.25`. Already banked `121ff5b` cross-layer `finalize→input_norm` `+0.09 pooled +0.12/+0.069` baseline.
* **R6 — Defects (no perf credit; correctness only).**

  > **Amendment: one of the two "defects" in `WHERE-THIS-STANDS.md:57-63` is mis-sourced and must not be actioned as written.**
  >
  > 1. **`scratch.attended` undersized — CONFIRMED.** `model.rs:3229` sizes it `u16_elements(max_tokens, hidden)` = 5376 halves, but both attention epilogues write `q_heads * head_dim` — 8192 on the 50 sliding layers, 16384 on the 10 full ones (`attention.metal:974` combine, `:927` single-pass/prefill) — and `o_proj.in_features` is that same larger number (`model.rs:525`). **22,016 B/layer/token overrun**, survivable only because Metal allocates page-granular, so it lands in the buffer's own padding. The line reference in the source comments (`model.rs:3012`) is stale by 17 lines. Existing test: `tests/perf_probe_activations_traffic.rs:300 attended_is_sized_hidden_but_written_q_dim`.
  > 2. **"No `_mul1` kernels exist" — NOT CONFIRMED, and as cited it is false.** The cited site `linear.rs:485-489` is `ishizuki_rate_of_name`, which parses the rate digits and accepts `1..=8`; it has nothing to do with mul1. `decode_state_mul1` exists (`linear.metal:128`) and `Exl3Codebook::Mul1` is a real variant that the dispatcher *does* emit (`linear.rs:570` loops `[Mcg, Plain, Mul1]`; `:3656`, `:3730`, `:3770` handle fractional-requires-mul1). The claimed `2.61×` cliff is **unreproduced in this tree**. Task: find the real failing configuration and reproduce the cliff, or record the claim as unsubstantiated. **Do not "fix" a kernel family that exists.**
  > 3. **Newly found while verifying R6** (`perf_probe_activations_traffic.rs:60-70`): `scratch.normed` is a **dead store** on the fused decode path — written by the input-norm epilogue, then passed as `input` to `encode_gemv_fused3`/`encode_gemv_fused2_mlp` which are always called with `pretransformed = true`, so they bind `transformed` and skip the `input_pipeline` dispatch (`linear.rs:2716-2725`, `:2513-2522`) — 645,120 B/token. `scratch.proj` is worse: never written at all on the decode path (`model.rs:1983-1995` writes `raw`; the fused finalize writes `hidden_b`), so 10,752 B/layer of pure allocation. Both are **0.005% of the stream — not perf work**; they belong to a memory tidy-up, listed here only because they are adjacent and cost nothing to note.
* **R7 — Measurement discipline.** Every tok/s claim `≥16` paired interleaved A/B, median + spread + control drift, `CONTROL_repeat` on microbenchmarks, CPU-computed checksum (`crux.md §6`), `phys_footprint` via `/usr/bin/time -l` (not `peak_resident_pages_bytes` `AGENTS.md §10.3`, `12.95 GiB` @8k proof), sabotage verification per `AGENTS.md §3.4` (`spec.md §4` coverage test).

## 4. Acceptance criteria

* **AC1 — Greedy throughput.** `scripts/gpuq try -- ./target/release/kaplumba --model ~/.kaplumba/models/Gxma-repacked --prompt "Explain the theory of relativity in simple terms, and then explain why it matters." --max-new-tokens 128 --temperature 0 --metrics` reports **median `≥ 17.2 tok/s` `≤ 58.14 ms/step`** across `≥16` paired A/B against HEAD re-pinned same session, greedy token ids byte-identical `2fff2c97…` hash preserved.
* **AC2 — Default-temp parity.** Same prompt ` --temperature 1 --top-k 64 --seed 1` reaches **greedy parity `±1%`** (today `14.37→16.05`) with identical greedy arm `0 differ`.
* **AC3 — Bit-exactness.** `cargo test -p kaplumba-engine --test live_kernel_bit_exact --test fused_bit_exact --test repacked_golden --test argmax_engine_parity` `0` failures; every new kernel path has sabotage evidence (break → fail diagnosable, revert → pass) recorded in `implementation_notes.md`.
* **AC4 — Split waiver (stretch only).** With `KAPLUMBA_DECODE_SPLIT_RULE=workgroups` median `≥ 17.6 tok/s`, flag-off arm exactly `AC1` hash, waiver signed in `spec.md §2` and `tasks.md` noting `434/476` 1-ULP diverge at `8204`.
* **AC5 — Memory.** `peak memory footprint ≤ 16.5 GiB`, `0 swap`, `phys_footprint` not `ru_maxrss` (`MEMORY.md` `12.95 GiB @8k` ref).
* **AC6 — Generator safety.** After `python3 scripts/gen_exl3_gemv.py`, `git diff --exit-code` null — `linear_repacked.metal` outside rewrite window.
* **AC7 — Negative results recorded.** `F4`-style table for `u8` unroll if tried, `cache_append` fold, etc., with medians.

## 5. Open questions

* **Q1 (settled by measurement):** `22.5 / 20` pure forward needs `320.4 / 284.8 GB/s` — over `273` pins. Recorded as excluded, not retried. Revisit only if `tokens/byte` authorized.
* **Q2:** `1.34× isolated` vs `+2.17%` end-to-end split gain — same kernel, different harnesses. 5-point sweep (§10.6) run in one session settles it.
* **Q3:** `_tl4` register pressure — `8`-deep after `4`-deep? `threadgroup half` `64` vs `128` B? Note the int-pipe half of `alu_boost`'s own instrument is **not trustworthy** (`p2_int*` scales as `n²`: `4→26.7 ms, 8→106.8, 16→428.6, 32→1759`, which is not a hardware property), so no integer-op ceiling may be quoted from that file — only the fp32 `≈119 G warp-instr/s` figure, which the sweep recovers independently.
* **Q4:** `Q7` K-tile staging `attention_gqa_body:836` — `0.8%` on `50 sliding` only after flash; worth the shared-template edit?
* **Q5 (new, from R1's amendment):** does the 4-deep unroll still pay **on the shipped kernel**, not just in the synthetic harness? `alu-boost` measured `154.0→168.6` and `161.1→169.4` GB/s on its own arm, whose ceiling (`arm_seq` `217.1`) is **below** the shipped `239.8 GB/s`. A +5–9% on a slower arm can vanish on a faster one — the whole `fused2_4col` lesson (`F4`: predicted win, measured `-0.7%` in 6/6 pairs). T1 measures the port end-to-end before it is believed.
* **Q6 (new):** which branch is authoritative? HEAD is `mtp-speculation` `517bc4e`; `33e8ead` (the R1 instrument) is only on `agent/tensor-ops`; `master` `fc36ee1` is the `MEMORY.md`-flagged broken tree. T0 pins this before any edit.

## 5a. Corrections to this document, made during verification

Recorded because each was a **shippable mistake caught before implementation**, and the tree has a documented history of exactly this failure (`AGENTS.md §3.4`: a kernel that read activations instead of weights passed a "6/6" claim).

| # | as first written | verified state |
|---|---|---|
| 1 | R1 = "port `arm_decode_u4` into `linear_repacked.metal:79`, `linear.metal:121`" | `alu_boost.rs` **absent from HEAD**; recover from `33e8ead`. `linear.metal:121` is `decode_state_mcg`, not a kernel. |
| 2 | R1 implied the existing `_u4` kernel was the lever | `_u4` is threadgroup-staged `uint4`, measured **180.5→173.7 GB/s — a loss**. Distinct experiment; name the port `_tl4`. |
| 3 | R3 = "wire the sampler into the pipelined loop" as a large engine task | Engine plumbing already present (`model.rs:4004` `pick: HeadPick`, `:4057`, `:4071`, `:4183`); scope is the `chat.rs:388` gate **plus two tests that assert the old gate**. |
| 4 | R6 = "missing `_mul1` kernels, `linear.rs:485`" | `linear.rs:485` is `ishizuki_rate_of_name`; mul1 **exists** (`linear.metal:128`, `linear.rs:570`). Claim unsubstantiated; reproduce or drop. |
| 5 | R6 = `attended` overrun at `model.rs:3012` | allocation is `model.rs:3229`; comments cite the stale line. Overrun 22,016 B/layer/token, real. |

## 6. Arithmetic this spec is built on

`12.039487488 GB` weights (`50×179,601,408 + 10×200,245,248 + 1,056,964,608` verified `crux.md §1.2`) + `2.1810 GB` KV `@16k` = `14.2390 GB/tok`; `@2k` `13.0646` (`MEMORY.md` `12.039` headline; `crux-bytes-raw.txt` headers). Wire `273 GB/s` pin, `245/253.6` measured sustained/best (`crux §2.6`), `239.8` gated GEMV, `217.1` harness `arm_seq` (`alu-boost`). `20 tok/s = 50.0ms` → `14.239/0.050=284.8`; `22.5=44.44ms → 320.4 @16k, 294.0 @2k` — all `>273`. Even `273` zero overhead → `19.17 @16k, 20.9 @2k`. Honest ceiling `245` zero overhead `17.21 @16k / 18.76 @2k`; real `+4.89ms` attention gap + `2.6–2.9ms` non-GEMV floor (`repacked-250gb §9`) → ship `~17.2–18.6`.

Dead ends not retried, with numbers: `D=32 0.288–0.918×` 11 shapes, `2–4 bpw 62.3 ms plateau`, `uint4 180.5→173.7` staged, `simd_shuffle 180.5→146.5`, remove dequant `−26.7%`, `fused2_4col −0.7% 6/6` `F4`, overlap spin `18.6s/2M` linear + `MTLEvent 1.55×` wash `20tps §1a/4`, dispatch `3.5 µs`.

## 7. Supersession

Continues `spec/decode-19tok-repacked-250gb` (ceiling arithmetic), `spec/alu-boost` (split `pattern 16% / extraction 10.9% / decode 1.1%` + `u4 +5–9%`), `spec/decode14-prefill100 §9–10` (attention `16.79ms` cross-validated `0.8%`, flash `113 GB/s`), `spec/decode-ceiling/WHERE_THIS_STANDS.md` (16.05/14.37 baseline). Baselines of record `decode-19tok`, `decode-20tps`, `decode-20tps-optimistic` ladder corrected for KV omission. Folder `decode-alu-fuse-18t` — `18` is stretch with waiver, not commitment.

