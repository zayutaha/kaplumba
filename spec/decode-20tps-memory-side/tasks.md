# spec/decode-20tps-memory-side — tasks

Each task is one commit. Status updated immediately after each.

- [x] **T0** — Fix the P01 probe loader (3 defects) + build arm (G)'s pipeline +
      a data-section-end gate. Run arm (G). **Finding: 633.4/637.2/634.0
      Gcodes/s = 237.6 GB/s with zero memory traffic; the 631 Gcodes/s that 20
      tok/s needs is met, so the instruction-count wall is refuted.**
      Arms (a)-(e) on real weights: (c) no-weight-loads = +11%, (e) masks are
      free (+0.1%), (d) variable shifts −2.5%.
      *Files:* `crates/engine/tests/perf_probe_code_rate_p01.rs`,
      `spec/decode-20tps-memory-side/`.
      *Tests:* `p01_code_rate_issue_bound`, `p01_code_rate_shape_sweep`, 2/2 pass.
      **Sabotage:** `data_start = 0` fails the new gate by 174,656 bytes
      (= 8 + header_len). **Observed failure recorded.**
      *Commit:* `a5683d1`.

- [x] **T1** — Per-shape bandwidth table on real weights (R1). Load one tensor
      per **distinct (K,N) shape** (not per role, and not "the 2 biggest", which
      returned two identical gate_proj tensors) and report GB/s and Gcodes/s per
      shape plus the byte-weighted step figure. Added `trellis_shape()` (single
      owner of the 3-D tile-major convention) and `shape_inventory()`.
      *Files:* `crates/engine/tests/perf_probe_code_rate_p01.rs` (probe only),
      `findings/F2-per-shape.md`. *Tests:* `p01_code_rate_issue_bound`.
      **Finding: bandwidth is monotone in N, 191.9 GB/s at N=21504 → 100.2 at
      N=2048; byte-weighted 176.5 GB/s over 10.983 GB = 62.23 ms, which
      reproduces the measured 62.3 ms step from first principles. Gcodes/s tracks
      GB/s (511.7 → 267.2), so it is *not* an instruction effect. 71.1% of bytes
      are in the two large-N shapes; target is +34%.** *Commit:* `73cf9a7`.

- [ ] **T2** — Latency vs pattern discriminator (R2, R3). Decide whether the
      11% memory-side term is "too few loads in flight" or "bad DRAM access
      pattern". They need opposite fixes.
      *Files:* probe only. *Tests:* the arm must be shown capable of failing both
      ways before either verdict is reported — a verdict that flips on a
      calibration constant is not a discriminator (`decode-ceiling` already lost
      a week to exactly that).

- [ ] **T3** — The 10% two-`gate_proj` spread (Q1). Reproducible, unexplained,
      identical shapes: 464 vs 511 Gcodes/s. If it is placement, it is free.
      *Files:* probe only.

- [ ] **T4** — Land the highest-value Phase-2 change (R4). Interleaved A/B, ≥6
      pairs, control arm, medians and spread. Regression gate:
      `--temperature 0` byte-identical over ≥3 runs, generated text only.
      *Files:* `crates/kernels/src/linear_repacked.metal` (+ `linear.rs` if a
      variant is added).

- [ ] **T5** — Report and close (R1-R5). Step-level bandwidth, attribution of
      every remaining GB/s to a named wall or a closed hypothesis.

## Standing constraints

- One model process at a time; `scripts/gpuq try` by default; exit 75 = busy.
- Compare generated **text only**; never hash the `--metrics` line.
- `/usr/bin/time -l` `peak memory footprint` is the only trustworthy memory
  metric here.
- Same-config spread is 35% on this box: a single run is not a result.
