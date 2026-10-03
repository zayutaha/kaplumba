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

- [x] **T4.1** — Hoist the ring-word loads in `linear_gemv_ishizuki_k3_fused2_4col`
      above the first `extract_v_mcg_32`, matching `_fused2_2col` (:491) and the
      T4.3 hoist in `_colmajor_2col_sk2` (:2477). Bit-exact scheduling change;
      required or the 4col A/B measures the missing hoist instead of column
      tiling (`A3-kernelsel.md` says so explicitly). *Commit:* `bb858c6`.

- [x] **T4.2** — A/B `KAPLUMBA_FUSED2_4COL=1` end to end. **NEGATIVE.** 6
      interleaved pairs, one binary, 128 greedy tokens: arm A (live `_fused2_2col`)
      median **16.021**, arm B (`_fused2_4col`) median **15.907**; B slower in
      **6/6** pairs. Bit-exactness demonstrated first — both arms produce the
      *identical* 128-token id hash (`4ce9ef31…`). The `bytes-per-threadgroup`
      axis does not pay. *Files:* `findings/F4-fused2-4col-negative.md`.

- [ ] **T4** — Land the highest-value Phase-2 change (R4). **Not landed.** The two
      candidates named in `plan.md` are now both closed by measurement: "more loads
      in flight" (F3 §6, three independent refutations) and "the 10%
      shape/tensor spread" (T3 — it was a `_2col` artifact). The last lever is the
      3 bpw lane window (2.67× → 1.33× useful-byte amplification).

- [x] **T7 — o_proj: closed, nothing to fix.** The "159 GB/s outlier" was **my own
      byte-count error** — o_proj is 1.156 GB (825.75 MB sliding + 330.30 MB full
      attention), not 0.991 GB, so it runs at **186.0 GB/s** inside its peers'
      186–203 band. `KAPLUMBA_ABLATE=o_proj_gemv` confirms the cost is real
      (16.001 → 17.865 tok/s, +11.6%, 6 pairs), so the profiler is not
      mis-attributing. Both candidate fixes lose: `KAPLUMBA_O_NOSK=1` −1.4% in
      **6/6** pairs against the microbenchmark's predicted **+41%**, and
      `KAPLUMBA_O_SK4=1` −0.4%. Shipped config is already optimal.
      *Files:* `findings/F5-o-proj-closed.md`; F3's table corrected.

- [ ] **T6 — the lane window.** **Correction to the earlier text:** this does *not*
      necessarily need a repack. The 768-bit ring order is a shared convention
      between the repacker and the kernel, but the *partition* of those 768 bits
      into lanes is the kernel's own; 16 lanes × 48 bits tiles the ring exactly.
      So a wider window may be reachable by re-partitioning alone, provided the
      bit→(output,input) map is preserved. That map has to be read out of
      `decode_state_mcg` / `pack_integer_states` first — it is not reverse-engineered
      yet. Ceiling if it works: ~18.2–18.7 tok/s. The ring is
      ordered by `(lane, k)` with `lane = 4*(o mod 8) + g`, so one output's 16
      inputs are *strided* across four lanes by 32 codes. A 16-code contiguous
      window is not a usable `(output, input)` set, so widening the window requires
      a new ring order: change `reorder_tensor_to_colmajor` / `pack_integer_states`
      (`kaplumba_engine::exl3`), version the on-disk format, repack all 411
      tensors, rewrite the reduction, re-earn bit-exactness. **Deserves its own
      spec.** Ceiling if it works: ~18.7 tok/s (F4 §Arithmetic).

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
