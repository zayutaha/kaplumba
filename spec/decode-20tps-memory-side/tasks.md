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

- [x] **T2** — Latency vs pattern, **answered from prior evidence plus one new
      measurement, but NOT with a fresh within-shape discriminator.**
      *Latency is closed:* `spec/decode-bandwidth-240gb/spec.md` measured 0.08M →
      2.62M loads in flight (32×) at **1.06×** bandwidth, and two other attempts
      (`ROUND1-2-SUMMARY.md:15`, `A3-kernelsel.md:203`) got ≤ +0.3%. No new GPU
      time was spent re-deriving it.
      *Pattern is the lane window:* the head (`k6`, row-major) reaches ~257 GB/s
      while every 3 bpw layer kernel sits at 159–203 GB/s in the same step — and
      the only property that differs is usable bytes per lane-load (2.0× at
      6 bpw vs 2.67× at 3 bpw). **This is inference from the F3 table, not a
      gated experiment**, and T2's own bar ("the arm must be shown capable of
      failing both ways") is therefore still *unmet*. Recorded as open.
      *Files:* `findings/F3-live-kernels.md`.

- [x] **T3** — The 10% two-`gate_proj` spread (Q1) — **moot as posed.** It was a
      pair of `_colmajor_2col` measurements, and that kernel is not on the live
      path (T2/F3). The live analogue is `o_proj` at **159 GB/s against its own
      peers at 199–203 on the same kernel and the same 336-TG grid**, which is a
      fixed-per-dispatch amortisation signature (K 8192 vs 21504), not placement.
      *Files:* `findings/F3-live-kernels.md` §2.

- [ ] **T4** — Land the highest-value Phase-2 change (R4). Interleaved A/B, ≥6
      pairs, control arm, medians and spread. Regression gate:
      `--temperature 0` byte-identical over ≥3 runs, generated text only.
      *Files:* `crates/kernels/src/linear_repacked.metal` (+ `linear.rs` if a
      variant is added).
      *The named candidate:* serve 16 codes per lane from the same 64-bit window
      (2.67× → 1.33× useful-byte amplification at KA=3). Kernel rewrite; changes
      accumulation order, so bit-exactness must be re-earned.

- [x] **T5** — Report and close (R1-R5). `findings/F3-live-kernels.md`: live
      per-kernel bandwidth (~200–206 GB/s), the re-measured ceiling (241–249,
      best 253), and the arithmetic that **caps the bandwidth route at ~18–19
      tok/s** — 20 tok/s needs 263 GB/s effective, above the box.

## Standing constraints

- One model process at a time; `scripts/gpuq try` by default; exit 75 = busy.
- Compare generated **text only**; never hash the `--metrics` line.
- `/usr/bin/time -l` `peak memory footprint` is the only trustworthy memory
  metric here.
- Same-config spread is 35% on this box: a single run is not a result.
