# Implementation Notes: Decode to 20 t/s

Running log. Append as work proceeds. See AGENTS.md §5.

## 2026-09-23 — init

- Re-based path from `spec/perf-and-memory-overhaul` (T5.5r) onto this spec
  after user confirmation: "do all, do not stop until 20 t/s"; register-pressure
  tradeoff for 128-thread/8-acc accepted.
- Reference deep-dive carried into plan.md §Evidence. Commit `476d8c8` was the
  pre-spec state (T5.3r revert landed).
- Gated bench command template:
  `env KAPLUMBA_RUN_SLOW=1 cargo test -p kaplumba-engine --release --lib bench_stream_bw_slow -- --ignored --nocapture --test-threads=1`
  (swap sample via `sysctl vm.swapusage` before/after).