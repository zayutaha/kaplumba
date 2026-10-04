# T3 / T4 handover notes (agent GEMM)

Worktree `/private/var/folders/tp/wz3877_s1z756p1kcsz3_d4h0000gn/T/opencode/p120-gemm`,
branch `impl/gemm-mma`, base `43a4176`. This file is appended to as work
proceeds. Numbers are measured on this box (M4 Pro, 20 GPU cores) unless said
otherwise.

Files touched (whitelist): `crates/kernels/src/linear.metal`,
`crates/kernels/src/linear.rs`, new tests
(`crates/kernels/tests/tile_mma_variants.rs`,
`crates/engine/tests/gemm_repack_tile_variants.rs`), this file.

---

## T4 — two research GEMM tiles

### What was built

| variant id | kernel | tile | threads | grid x | bstage |
|---|---|---|---|---|---|
| 1 (`128X64`) | `linear_gemm_mma` (base) | 128q x 64n | 256 | out/64 | 64x72 halfs = 9216 B |
| 2 (`4SG`) | `linear_gemm_mma_4sg` (new) | 128q x 32n | 128 | out/32 | 64x40 halfs = **5120 B** |
| 3 (`REGPIPE`) | `linear_gemm_mma_regpipe` (base) | 128q x 64n | 256 | out/64 | 9216 B |
| 4 (`REGPIPE_K32`) | `linear_gemm_mma_regpipe_k32` (new) | 128q x 64n, KCHUNK=32 | 256 | out/64 | 32x72 halfs = **4608 B** |

- **T4.1 `linear_gemm_mma_4sg`** (linear.metal:10496): 4 simdgroups x (4 row
  blocks x 4 column blocks) covering 128q x 32n, sixteen named `simdgroup_float8x8`
  accumulators per thread (4R x 4C). Motivation: the fragment-load bill per 16
  MMAs drops from (2 A + 8 B)/16 = 0.625 to (4 A + 4 B)/16 = 0.5, and the B
  side (the threadgroup-memory side) halves. Accumulation order per output
  element is unchanged vs `linear_gemm_mma` -> bit-identical output.
- **T4.2 `linear_gemm_mma_regpipe_k32`** (linear.metal:10970): regpipe with
  `KCHUNK = 32` (NTILES 4, OUTS 64, SGROUPS 8 unchanged), halving `bstage` to
  4608 B.

### Dispatch / variant API (`linear.rs`)

- ids + docs at `linear.rs:60-118`: `GEMM_MMA_VARIANT_{AUTO,128X64,4SG,REGPIPE,REGPIPE_K32,NONE}`.
- `force_gemm_mma_variant` / `clear_gemm_mma_variant` — atomic override of env
  selection (env flags are memoised by `env_flag`, so a test cannot flip them
  mid-process).
- `last_gemm_mma_variant()` — **load-bearing test hook**: variants are
  bit-identical to the baseline, so output comparison alone allows a vacuous
  "the dispatch never left baseline" pass (Q11 / `8f1a863` failure mode).
- One resolution point in `encode_gemm` (`linear.rs:~3374`): forced atomic
  beats env; env order `KAPLUMBA_MMA_4SG` -> `KAPLUMBA_REGPIPE_K32` ->
  `KAPLUMBA_REGPIPE` -> default `128X64`.
- **Host gate** (`linear.rs:~3390`, `research_eligible`): all three research
  variants require `perm_mode != 1 && rate == Exl3Rate::Integer(3) && codebook
  == Exl3Codebook::Mcg`, else they fall back to `128X64`. Gxma is 3.0 bpw MCG,
  no `layout` key (perm_mode 0), so the gate admits it.

### The census convention (why 4sg carries only ONE decode arm)

`crates/engine/tests/probe_fastpath_decode_path.rs` derives two inventories
from kernel source:

- `runtime_codebook_kernels()` — every kernel body containing `params.codebook`
  (`:287`);
- `per_code()` census — `fractional_window(` counts as a per-code path (`:257`).

Both are `assert_eq!` on **exhaustive sorted name lists**. A research tile that
copies `linear_gemm_mma` verbatim carries `fractional_window(` and
`params.codebook` and fails two tests:

```
the_runtime_codebook_branch_is_unreachable_for_an_mcg_checkpoint
only_eight_dead_specialisations_use_the_per_code_path
```

Observed with the naive (verbatim-copy) 4sg: 9 pass / **2 fail** out of 11 in
that target. The rule the base tree already follows (regpipe) is: **a research
tile is host-gated to a fixed rate + codebook and hardcodes that decode**, so
its body has neither `fractional_window` nor `params.codebook`.

Applied to 4sg:

- dropped the `fractional` and generic arms; body now has `if (ka == 3u) { ... }`
  (folded-MCG arm only), `const uint ka = params.rate; // 3: host-gated`,
  `word_count = 256u * ka / 32u`;
- header comment documents the gate;
- `linear.rs` gate extended to 4sg (`research_eligible`).

Result: that target is back to **11/11**.

**Caveat found (documented, NOT fixed — file not on my whitelist):** four
read-only census tests hard-code the kernel inventory and now fail purely
because T4 adds two `kernel void`s to `linear.metal` (111 -> 113, 138 -> 140):

| test | file:line | assert | needs |
|---|---|---|---|
| `specialisation_costs_compile_time_not_throughput` | `perf_probe_simd_alu_roofline.rs:1134` | `linear_kernels == 111` | 113 (and `:1140` `total == 138` -> 140) |
| `census_of_all_138_linear_kernels` | `probe_kernsel_census.rs:576-582` | `kernels.len() == 138` | 140 |
| `the_census_detectors_can_fail` | `probe_kernsel_census.rs:1096-1100` | `... == 138` | 140 |
| `library_census_and_the_honest_answer_on_instruction_cache_pressure` | `probe_variants_window_forms.rs:1005` | `lib.len() == 138` | 140 |

Their own messages say the figure "must move with it". These are inventory
snapshots, not behaviour checks; bumping the four literals is the whole fix.
They were left untouched because the agent brief's whitelist is
`linear.metal` / `linear.rs` / new `*gemm*repack*`+`*tile*` tests / this file
only — **coordinator: please bump the four literals (owner of those files).**

### Tests added

`crates/kernels/tests/tile_mma_variants.rs` (4 tests + 1 `#[ignore]` probe,
mutex-serialised; random packed u16s, no engine dependency):

- `forced_variant_is_what_dispatches` — forced id must equal reported id, at
  3 bpw (all four dispatch themselves) and at 4 bpw (the three research
  variants must report `128x64`).
- `variants_match_the_baseline` — every variant vs forced baseline, rates
  Integer(3) and Integer(2); exact bit equality (`worst == 0`).
- `auto_selection_defaults_to_the_shipped_tile`
- `env_flags_reach_the_dispatch` — re-execs itself as
  `probe_env_flag_selection` with `T4_EXPECT_VARIANT` set.

`crates/engine/tests/gemm_repack_tile_variants.rs` (3 tests):

- `every_variant_matches_the_cpu_reference` — integer 1..=8 bpw and fractional
  1..=3 bpw vs CPU reference, `VARIANT_TOLERANCE = 2.1604e-3`.
- `variant_grids_cover_non_square_shapes` — 4 shapes x 7 query counts (1, 127,
  128, 129, 255, 256, 257), 3 bpw.
- `variants_are_bit_identical_to_the_baseline` — fp16 bit-identity at 2/3/4 bpw.

Measured worst normalised errors (release, GPU):

- integer & fractional rates, 128x128, 129 queries: `128X64`/`4SG`
  **6.9792045e-4**; `regpipe`/`regpipe_k32` **5.114365e-4**.
- non-square grid test: **2.1603336e-3** — exactly the number
  `linear_metal.rs` reports for the baseline, so `VARIANT_TOLERANCE = 2.1604e-3`
  is the baseline's own error, not a loosened one.

### Sabotage record (each was run, produced a diagnosable failure, reverted)

1. 4sg store map `KAPLUMBA_MMA_STORE(acc4, 8u, 0u)` -> `1u`:
   kernels test `4sg diverged from the baseline: worst normalised error
   3.6832252e-1`; engine test `1.9430692e-1 exceeds 2.1604e-3` plus
   `differs ... (actual 86.25, baseline 137.625)`.
2. 4sg dispatch guard made unreachable (`out_features % 999_999 == 0`):
   `forced 4sg but the dispatch reported 128x64`, env probe
   `resolved to 128x64 (1), expected 4sg (2)` — outputs still matched, which is
   exactly why the identity asserts are load-bearing.
3. `regpipe_k32` publish `cc0 + 8u` -> `cc0 + 0u`:
   `regpipe_k32 diverged ... 1.5057414e0`, engine `1.4998507e0 exceeds`.
4. **(after the census fix)** 4sg store map `acc4, 8u` -> `9u`:
   kernels `4sg diverged ... 4.8652023e-1 at flat index Some(13539)`;
   engine `integer bits=3 variant=4sg: 5.140755e-1 exceeds 2.1604e-3`,
   `shape 128x256 queries=127 variant=4sg: 5.140755e-1 exceeds`,
   `bits=3 variant=4sg: output differs ... (actual 37.8125, baseline 31.234375)`.
5. **(after the census fix)** host gate weakened — dropped
   `|| variant == GEMM_MMA_VARIANT_4SG` from `research_eligible`:
   `at 4 bpw, forced 4sg must resolve to 128x64` and
   `[Integer(2)] forced 4sg resolved to 4sg` — proves the fallback asserts catch
   a gate that no longer gates.

### Test status at the time of this entry

Green: `kaplumba-kernels --test tile_mma_variants` 4 passed/1 ignored;
`gemm_repack_tile_variants` 3/3; `linear_metal` 7/7; `probe_fastpath_decode_path`
11/11; `probe_tgsize_geometry` 12/12.

**Pre-existing at base `43a4176` (verified by stashing all my changes and
re-running):** 4 failing targets, NOT mine —

- `perf_probe_concurrency_dependency_graph::no_device_side_sampler_exists`
- `perf_probe_dispatchsel_columns::the_colmajor_contig4_kernel_is_not_on_the_decode_path`
- `perf_probe_host_ops::the_device_token_handoff_exists_and_the_row_follows_the_pick`
  (+ `sampling_a_full_width_row_costs_milliseconds_per_token`, flaky)
- `probe_cache_hierarchy::activation_loads_are_4_byte_alignable_and_collapse_2_to_1`

Also pre-existing: `cargo build -p kaplumba-engine --example attn_bench` fails
(`attn_bench.rs:374` calls 2-arg `decode_splits`, `kernels/src/attention.rs:191`
takes 3) — at base too, so `cargo test` without `--tests` never compiles.
**Use `cargo test --release --tests` in this worktree.**

### Environment gotchas hit here

- `cargo test` (no `--tests`) fails to build the `attn_bench` example at base.
- `rustfmt --edition 2024` is the repo's style (pre-existing files are clean);
  `crates/kernels/src/linear.rs` is NOT fmt-clean at base (13 pre-existing
  hunks) — do not reformat the whole file, only new hunks.
- Every cargo invocation here needs
  `export CARGO_TARGET_DIR=/private/var/folders/tp/wz3877_s1z756p1kcsz3_d4h0000gn/T/opencode/p120-gemm-target`.

---

## Measurements

(pending — A/B interleaved runs at 614 tok and 8k tok)

## T3 — weight repack for MMA tile locality

(pending)
