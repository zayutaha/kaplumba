# Implementation Notes: Decode to 20 t/s

Running log. Append as work proceeds. See AGENTS.md §5.

## 2026-09-23 — P0 stream bandwidth probe (lands use TBD)

- `bench_stream_bw_slow` (gated): contig 512 MB = **211 GB/s**; gemv-strided
  (in 2048 × out 1024, twords 24, 192 MB) nsplit=1 = **214**, nsplit=2 = **334**,
  nsplit=4 = **377**, nsplit=8 = **381 GB/s**.
- **Verdict: the decode gemv's ~79 GB/s (168 Gw/s gate) is an in-kernel
  compute/occupancy wall, not a memory wall.** The identical access pattern
  reads at 214+ GB/s with zero decode; splitting the in-tile range across more
  threadgroups raises read throughput to ~380 GB/s (memory-level parallelism).
  P1-T1 (128-thread/8-acc) has ~2.7–4.8× headroom; plain decode at 20 t/s
  (224 GB/s) is credible without speculation. P3 remains the safety net.
- Probe cleanups: `dispatch_thread_groups` clamps grids past 65535/axis and
  under-runs SILENTLY (first contig kernel used 16.7M groups → 0.027 ms "20 TB/s";
  mystery numbers). Use `dispatch_threads` (nonuniform) via flat indices.
  Per-thread output slots beat one-hot atomically-shared u64 for a read bench.