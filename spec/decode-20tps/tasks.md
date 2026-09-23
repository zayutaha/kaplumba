# Tasks: Decode to 20 t/s

- [x] **P0: Stream-sum bandwidth probe (decision measurement)**
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
  - **Result**: contig 512 MB = 211 GB/s; strided nsplit=1 = 214 GB/s,
    nsplit=2 = 334, nsplit=4 = 377, nsplit=8 = 381 GB/s. **Memory pattern is
    NOT the wall — in-kernel compute is** (79 GB/s decode vs ≥210 GB/s
    achievable for identical access pattern; splitting raises MLP to ~380).
    P1-T1 parity restructure is the fix; 20 t/s plain decode (224 GB/s) is now
    credible. Caveat: `dispatch_thread_groups` clamps big grids (65535),
    silently under-runs — use `dispatch_threads`.
  - **Commit**: TBD

- [ ] **P1-T1: PonyExl3 v12 parity gemv (128-thread, 8-acc, static funnel, direct loads)**
  - **Files**: `crates/kernels/src/exl3.metal` (`exl3_gemv4`), `crates/kernels/src/exl3.rs`,
    `crates/engine/src/metal_linear.rs`
  - **Intent**: New v12-parity kernel kept **alongside** `exl3_gemv` for A/B:
    128 threads/TG (NSG=4 in-tile stride), 8 register accs/lane, static funnel
    windows, direct device word loads, in-kernel cross-SG reduction via
    `tgw[NSG][TILE]` + one end barrier. Production **not** switched (revertable).
  - **Testing**: kernels (21), engine fast (117 incl. `gpu_gemv4_matches_cpu_decoded_tile`),
    gated `bench_passes_slow` A/B, split sweep A/B.
  - **Result (NEGATIVE — v1 stays)**: v4 loses at every nsplit on the real
    shapes. gate@n2 0.683 (v1) vs 0.709 (v4); down@n8 0.697 vs 0.709 (v4 best
    n2); o@n8 0.368 vs 0.377; q@n1 0.361 beats 0.369 by 2% once (noise);
    k 0.272 vs 0.275; lm_head@n1 5.847 vs 6.186 (−6%). Quadrupling ILP + 4-way
    SG split changed nothing → the kernel is NOT dependency-latency-bound.
    Marker commit ships kernel + A/B hooks as regression bench, production
    untouched.
  - **Commit**: TBD

- [x] **P1-T2: nsplit chooser → sweep argmax (v1 geometry)**
  - **Files**: `crates/kernels/src/exl3.rs` (`gemv_splits`), policy tests
  - **Intent**: Bake the measured argmax override table from
    `bench_gemv_split_sweep_slow` (gate n2, down n8, o n8, q n2, k/v n8, lm n1;
    formula fallback for unlisted shapes, clamped 1..16). The reference float
    (128 in-tiles/split) contradicts the measured k/v argmax (n8), so argmax
    wins per T5.2r philosophy. Expected ~1–2 ms/token.
  - **Testing**: `bench_gemv_split_sweep_slow` re-run shows engine policy ≥
    best-of-sweep per shape; kernels + fast + parity green.
  - **Result**: done — override table {1344→2, 336→8, 512→2, 256→8}; policy
    test updated; kernels (21) + engine fast (117) green.
  - **Commit**: TBD

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

- [ ] **P3 (PRIMARY): spec/verify-gated-spec-dec — interview**
  - **Files**: new `spec/verify-gated-spec-dec/{spec,plan,tasks}.md`
  - **Intent**: AGENTS.md §2 interview on the CONFIRMED shallow-layer target
    draft (target's own first 2–4 layers + reduced head, zero extra weights),
    row-batched target-verify on `exl3_gemm_sg` (decode ALU amortized across
    ≤8 rows), acceptance measurement plan, memory budget within 14.6 GB,
    integration into `gpu_decode.rs`. Plain decode is ALU-roofed at ~7–10 t/s
    (R2/R6 evidence), so this spec is the PRIMARY delivery of the 20 t/s gate.
  - **Testing**: user confirmation before implementation.

- [ ] **P3 (PRIMARY): verify-gated speculative decode → 20 t/s**
  - **Files**: per P3 spec
  - **Intent**: (a) 2-row E2E probe that batched target-verify actually
    amortizes decode ALU on M4 Pro; (b) shallow-draft gen + verify batch,
    greedy-identical only; accept run-length measured on artifact; flip
    `decode_throughput_slow` gate ≥ 19.5 only on measured proof.
  - **Testing**: full coherence (fixture, France prompt, margin > 8) + two
    stable gated runs ≥ 19.5 t/s.