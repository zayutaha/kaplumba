# SPECIFICATION: 4 / 8 bpw EXL3 Support

Status: **repacker done and tested; the decode path was already implemented and is
awaiting one GPU verification.** Branch `wt-bpw4`, isolated from the decode-performance
work so the two do not contend for the same files or the GPU.

## 1. Goal

Make the engine able to load and decode a real 4 bpw / 8 bpw-head EXL3 checkpoint, and
stop the repacker from silently producing row-major payloads under a `colmajor` header.

**Not a speed project.** 4 bpw was measured **6.7% slower** than 3 bpw end to end
(bytes ×1.333, bandwidth only ×1.249 — `bench_rate_sweep --rates 3,4,6`, 2026-10-01).
The value here is format coverage: one EXL3 rate is exercised by production code and
tests, and the repacker no longer fails open.

## 2. What was already there

Contrary to three separate claims in this repo's own notes, **most of 4 bpw support
already existed**:

| claim | reality |
|---|---|
| "the loader refuses any non-MCG codebook" | **False.** `Codebook::Mul1` is implemented (`exl3.rs:40`, `decode_mul1` at `:169`). The only rule is `FractionalRequiresMul1` (`exl3.rs:14,82`) — *fractional* rates need mul1. An integer 4 bpw tensor is unconstrained. |
| "the shim has no `embed_tokens`" | **False.** `model.language_model.embed_tokens.weight` is in the index, in shard 1, bf16 `(262144, 5376)`. The earlier claim confused the safetensors index with the `tensor_storage` map in `quantization_config.json`. |
| "the shim is missing files so it cannot run" | **Half true, and irrelevant.** It lacks `chat_template.jinja` and `processor_config.json`. The loader requires only `model.safetensors.index.json`, `tokenizer.json`, `tokenizer_config.json` and the shards — **all four present**. The CLI never reads a chat template. |

Already implemented and tested before this spec:

- `linear_gemv_ishizuki_k4_colmajor_contig4_bench`, `..._k6_..._bench`
  (`linear_repacked.metal:2685-2803`, `:2858-2982`) — 3-word form at 6 bpw because
  2 words cannot hold 6 codes (`rate4_colmajor_bit_exact.rs`).
- `rate4_colmajor_bit_exact.rs` (6 tests) — 4 and 6 bpw colmajor kernels match a CPU
  `decode_tile` reference; the two-word window is safe exactly at rates 1–4;
  amplification 2.667 at 3 bpw vs 2.000 at 4 and 6; DRAM line traffic 1× at every rate.
- `rate4_8_kernel_coverage.rs` — 4/8 bpw have their own row-major kernels and slots.
- `rate4_8_real_weights_bit_exact.rs` — takes real 4 bpw tensors and checks GPU against
  CPU, `#[ignore]`d pending `KAPLUMBA_TEST_MODEL`.

## 3. The bug that was fixed

`reorder_tensor_to_colmajor` took no width and hardcoded **48 u16 per tile** = 3 bpw.
`Exl3Rate::packed_words()` (`crates/kernels/src/exl3.rs:33-38`) already held the real
invariant, `16 * bits`, 700 lines away.

The failure was silent and total for 4 bpw. `repack_model.rs` gated on
`shape[2] == 48` to match; tile width *is* `16 * bits`, so at 4 bpw it is 64. Every 4 bpw
trellis matched nothing, fell through to the verbatim-copy branch, and was written
**row-major** — while the tool wrote `layout: "colmajor"` into
`quantization_config.json` and exited 0. The engine selects its kernels from that key
(`model.rs:3171-3188` sets `permutation_mode` from it). A 4-bit repack would have
reported success and produced a checkpoint that decodes to noise.

That is the **fourth** fail-open fallback in this repo, after `LinearKernels::new()`
turning a missing `)` into a green test, the elided `else` branches in the encode arms,
and the running-LCG tile generator that made the repack the identity.

## 4. Changes

- `reorder_tensor_to_colmajor` takes `words_per_tile`. The tile-grid transpose is
  rate-agnostic — a tile is `words_per_tile` u16 copied verbatim and only the grid is
  transposed — so 3 bpw is unchanged.
- `trellis_words_per_tile(&[usize]) -> Option<usize>` recovers the rate from `shape[2]`
  and returns `None` for anything that is not `[tiles_k, tiles_n, 16*bits]`, so a caller
  can distinguish "not a trellis" from "a rate I do not handle".
- `repack_model`: the trellis branch keys off the derived width; a malformed shape is a
  hard error, not a copy; `layout: colmajor` is written **only** after verifying
  `repacked == total trellis tensors`, otherwise the tool exits non-zero with both
  counts. 11 call sites updated; two now derive `16 * shape.bits`.
- New `crates/engine/tests/colmajor_rate_general.rs`, 6 CPU-only tests.

## 5. Verification

`colmajor_rate_general` 6/6 · `repacked_golden` 2/2 · `rate4_colmajor_bit_exact` 6/6 ·
`live_kernel_bit_exact` 1/1. 3 bpw output unchanged and bit-exact.

**Sabotage evidence, both directions, observed** (AGENTS.md §3.4):

| sabotage | result |
|---|---|
| tile destination off by one | 3 of 6 fail |
| width argument ignored, 48 hardcoded again | 4 of 6 fail |

The second is the load-bearing one: reverting it leaves **`repacked_golden` passing**,
because that suite only exercises 3 bpw. The pre-existing tests cannot see this class of
bug at all, which is the argument for the new file.

## 6. Still to do

- [ ] **T2 — Run `rate4_8_real_weights_bit_exact` against the shim.** Needs an idle GPU
      and ~2 GiB, not a model load:
      ```bash
      KAPLUMBA_TEST_MODEL=~/.kaplumba/models/Gxma-bpw4-shim \
        cargo test --release --test rate4_8_real_weights_bit_exact -- --ignored
      ```
      **Blocked at time of writing: the GPU was held by the decode-performance agent**
      (`gpuq status` reported `gpu: load .../Gxma-repacked (pid 58889)`, 0.1 GiB
      available). AGENTS.md §10 forbids a second GPU job. Not run, not claimed.
- [ ] **T3 — Decide whether production dispatch should reach the 4/6 bpw colmajor
      kernels.** They are named `..._bench` deliberately: `ishizuki_gemv_pipeline`
      looks up `..._k{bits}_colmajor_contig4` by name (`linear.rs:1375`), so the
      `_bench` suffix makes the lookup miss, and
      `rate4_colmajor_bit_exact.rs::the_engine_cannot_dispatch_the_four_bpw_measurement_kernel`
      asserts that. Lifting it is a deliberate decision, not a rename to be done
      casually. The shim is row-major (`layout: None`) and does not need it.
- [ ] **T4 — Optional: repack the shim to colmajor at 4 bpw** now that the repacker
      supports it. Needs ~20 GB free and ~20 GB of output. Worth doing only to exercise
      the new path on real data; `58 GiB` free at time of writing.

## 7. Non-goals

1. No bit-rate change to the shipped model. `Gxma-repacked` stays 3 bpw and is the
   performance baseline.
2. No claim that 4 bpw is faster. It is measured 6.7% slower.
3. No Python in product or engine code.
4. `~/.kaplumba/models/Gxma/Gxma` untouched.
5. The `_bench` interlock is not removed as a side effect of anything.

## 8. Open questions

1. Should production ever dispatch the 4/6 bpw colmajor kernels, or do they stay
   measurement-only? (T3. Recommendation: keep the interlock; the row-major 4 bpw path
   is exercised and tested, which is the coverage that matters.)
2. Is the shim's symlink into `~/Downloads` acceptable? It is fragile — `/tmp` and
   Downloads both get cleared — and the three shards are 60-byte symlinks, not files.
   A 20 GB copy is the fix, or a documented regeneration step.
