# Tasks: Decode to 20 t/s

- [ ] **P0: Stream-sum bandwidth probe (decision measurement)**
  - **Files**: `crates/engine/src/metal_linear.rs` (new gated test
    `bench_stream_bw_slow`), kernel MSL inline in the test
  - **Intent**: Two Metal kernels over a synthetic buffer — (a) contiguous read
    sum, (b) gemv-strided read sum replicating the `exl3_gemv` grid
    (out_tiles × nsplit, 32-lane, tile stride = out_tiles*twords*2). Reports
    GB/s for both. If strided ≈ 200+ GB/s → 79 GB/s is in-kernel compute →
    P1-T1 is the fix. If strided ≈ 80–110 GB/s → memory-pattern wall → grid
    restructuring before/with P1-T1.
  - **Testing**: gated `bench_stream_bw_slow`; no assertions, print + eyeball
    (decision input); suite unaffected.

- [ ] **P1-T1: PonyExl3 v12 parity gemv (128-thread, 8-acc, static funnel, direct loads)**
  - **Files**: `crates/kernels/src/exl3.metal` (`exl3_gemv`), `crates/kernels/src/exl3.rs`
  - **Intent**: Rewrite main decode loop to v12 MT=1: 128 threads/TG (4 SG,
    `NSG=4`), `acc[8]` register accumulators with tile-invariant funnel
    geometry computed once, `simd_shuffle` x-fetch, direct word reads, in-kernel
    conflict-free cross-SG reduction, zero hot-loop barriers, no dynamic
    indexing. Keep `aligned` path + k=7 pair-table fallback + T5.2r nsplit hook.
  - **Testing**: `cargo test -p kaplumba-kernels` (21), engine fast (116),
    parity `gpu_gemv_matches_cpu_decoded_tile`, `debug_gemv_rows`, one-hot;
    gated `bench_passes_slow` A/B vs 168 Gw/s gate baseline.

- [ ] **P1-T2: nsplit chooser → reference formula, re-sweep**
  - **Files**: `crates/kernels/src/exl3.rs` (`gemv_splits`), policy tests
  - **Intent**: Double splits while `out_tiles*n < 8192` and `in_tiles//
    (n*2) >= 128`, enforce `tiles_per_split % 8 == 0`; validate with
    `bench_gemv_split_sweep_slow` on the new geometry; bake per-shape argmax
    only where it beats the chooser.
  - **Testing**: `bench_gemv_split_sweep_slow` per shape ≥ P1-T1 baseline;
    policy test updated; kernels + fast + parity.

- [ ] **P1-T3: Fuse had_in (and v18 post) into M=1 gemv**
  - **Files**: `crates/kernels/src/exl3.metal`, `crates/kernels/src/exl3.rs`,
    `crates/engine/src/metal_linear.rs`
  - **Intent**: In-kernel 128-point FWT prologue over the x slice before the
    tile loop (v12 `had_prologue`), dropping the separate had_in dispatch +
    xh buffer round-trip per linear. Measure had_out/svh epilogue; keep only
    if it wins (v18: one TG owns a 128-col block).
  - **Testing**: parity/one-hot/had-ulp tests via fused path; `bench_passes_slow`:
    `full` time strictly below current; suites green.

- [ ] **P2: End-to-end re-verify + ceiling amendment**
  - **Files**: `crates/engine/src/gemma4.rs`, `spec/final-same-algorithm-ceiling/{spec,tasks}.md`,
    `spec/decode-20tps/implementation_notes.md`
  - **Intent**: Gated `decode_throughput_slow` with swap sampling (chunk before/
    after); record ms/token, t/s, phase split. If landed ~9.5–11 t/s: amend
    ceiling spec to the measured practical number (40–45% of bus), keep gate
    unflipped, document. Never land red.
  - **Testing**: two consecutive stable gated runs, no swap-file growth.

- [ ] **P3: spec/verify-gated-spec-dec — interview**
  - **Files**: new `spec/verify-gated-spec-dec/{spec,plan,tasks}.md`
  - **Intent**: AGENTS.md §2 interview: draft architecture/weights, verify
    mechanism (greedy-identical), acceptance measurement plan (1-token-ahead
    profiler), memory budget within 14.6 GB, integration into `gpu_decode.rs`.
  - **Testing**: user confirmation before implementation.

- [ ] **P3: verify-gated speculative decode → 20 t/s**
  - **Files**: per P3 spec
  - **Intent**: Draft model gen + verify batch against target, greedy-identical
    only; accept run-length measured on artifact; flip `decode_throughput_slow`
    gate ≥ 19.5 only on measured proof.
  - **Testing**: full coherence (fixture, France prompt, margin > 8) + two
    stable gated runs ≥ 19.5 t/s.