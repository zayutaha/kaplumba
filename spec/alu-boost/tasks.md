# `spec/alu-boost` — tasks

- [x] T1 `crates/engine/examples/alu_boost.rs`: Rust-generated MSL, concrete
      function names, decoded-value self-check against a Rust reference.
      Tested: `cargo build --release -p kaplumba-engine --example alu_boost`;
      run prints `[check] decode transcription verified ... |d| 0.000e0`.
- [x] T2 Fix instrument bug 1 — `out` sized for the widest writer. Before:
      `arm_seq` wrote ~320 KiB past the buffer, faulted the command buffer, and
      reported a 1.7 TB/s "DRAM ceiling". After: 151 GB/s with 4 B loads.
- [x] T3 Fix instrument bug 2 — `float fa[8]` indexed to 31 in the 16/32-chain
      arms; the extra instructions were eliminated and fp looked flat. After:
      fp rises 95 → 120.7 G warp-instr/s and plateaus.
- [x] T4 Fix instrument bug 3 — interleave the arms. Two sequential sweeps
      drifted 10 % apart on `arm_seq`. After: 9 interleaved rounds + printed
      drift control.
- [x] T5 Give `arm_seq` 8-way load ILP. Before: 151 GB/s (load-latency-bound).
      After: 205–219 GB/s, against the box's own 238–244.
- [x] T6 Split the decode from the extraction (`arm_extract`) so the ALU cost is
      attributed. Result: extraction 10–11 %, decode 1–7 %.
- [x] T7 Add `arm_decode_u4`, the same-mapping 4-deep unroll, and check it
      bit-for-bit against `arm_decode`. Result: 0 of 2560 words differ; +5.2 %
      and +9.5 % in two interleaved runs.
- [x] T8 Record the negatives: `arm_decode_h2` 0.936× / 0.970× (dead),
      `arm_decode2/4/8` not bit-exact, `p2_int*` void (`n²` scaling).
- [x] T9 Re-run the project's own `probe_bw_250gb` unchanged, to get the box's
      ceiling today: 238.5 GB/s `uint1`, 244.3 GB/s `uint4`.
- [x] T10 Write `findings.md`; commit the instrument and the raw log.
- [ ] T11 **Not done: measure `arm_decode_u4` in the shipping kernel.** The
      +5–9 % is on a synthetic replica of the ring. It must be applied to
      `linear_gemv_ishizuki_k3_colmajor_contig4` (`linear_repacked.metal:1992`)
      and measured in-engine before any decode-tok/s number is claimed.
- [ ] T12 **Not done: fix the part-2 integer generator.** `p2_int*` scales as
      `n²` and `p2_mix*` is therefore void, so "do the int and fp pipes
      overlap?" — the question this spec is named after — is still open.
