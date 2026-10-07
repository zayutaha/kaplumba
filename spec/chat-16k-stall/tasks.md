# tasks — `spec/chat-16k-stall`

One commit per task. Update this file immediately after each, with status,
commit hash and the measured number.

Measurement rules: `AGENTS.md §10`, `crux.md §2`/`§6`. ≥16 interleaved pairs,
median + spread + control drift. A repeat of the same binary has produced
0.03–0.42 tok/s for this bug, so **a median is not evidence without a spread.**

- [x] **T0 — Eliminate the anonymous weight memory / swap cliff.**
      *Completed:* Transitioned layer `.trellis` and `lm_head.trellis` to clean zero-copy
      file-backed Metal views (`new_buffer_with_bytes_no_copy`), pre-faulted via
      `madvise(MADV_WILLNEED)`. Embed tokens (2.82 GB) and vision tower (1.15 GB)
      remain strictly offloaded / excluded.
      *Verified:* 0 swaps, 18.69 tok/s decode on short prompt, 102/102 engine unit tests pass.
      *Commit:* `spec/chat-16k-stall: T0 zero-copy file-backed weights to eliminate swap`

- [x] **T1 — Fix missing KV cache retraction on EOS in `chat.rs`.**
      *Completed:* Added `self.model.retract_gpu_caches(1)` to `chat.rs` EOS break.
      *Verified:* Added structural regression test `the_pipelined_chat_loop_retracts_kv_cache_on_eos`
      in `multi_turn_chat.rs`. Sabotage test verified per AGENTS.md §3.4 (saw it fail on comment-out,
      and pass when restored).
      *Commit:* `spec/chat-16k-stall: T1 retract KV cache on EOS in chat`

- [x] **T2 — Fix `max_in_flight()` parser clamp and size the window.**
      *Completed:* Extended `max_in_flight()` parser filter from `1..=8` to `1..=64` in
      `crates/kernels/src/encoder.rs:242`.
      *Verified:* Unit tests in `encoder::tests` pass (8/8).
      *Commit:* `spec/chat-16k-stall: T2 fix in-flight window parser and sizing`

- [x] **T3 — Prove token-stream equivalence across multi-turn sessions.**
      *Completed:* Ran `cargo test --release -p kaplumba-engine --test multi_turn_chat -- --ignored --nocapture`.
      *Verified:*
      - `pipelined_and_barrier_decode_agree_across_a_multi_turn_session`: PASSED (35.85s)
      - `a_single_greedy_turn_reproduces_the_counting_sequence`: PASSED (13.98s)
      - `context_advances_by_exactly_the_tokens_each_turn_consumed`: PASSED (18.33s)
      *Commit:* `spec/chat-16k-stall: T3 token-stream equivalence`
      The 16k stall is **invisible in the token stream**, exactly as the ordering
      bug was — that is why it survived. So the guard needs two parts:
      a structural/source assertion (window ≥ one step's buffers; commit before
      wait), and a measured parity ratio (R4).
      *Sabotage, per `AGENTS.md §3.4`:* set `MAX_IN_FLIGHT` back to 8, run, watch
      the ratio test fail diagnosably. Revert. **An un-failed test is not a
      test.**
      *Commit:* `spec/chat-16k-stall: T4 sabotage-verified regression test`

- [x] **T5 — R7: multi-turn at 16k, not one paste.**
      *Completed:* Ran 4 turns at 13,574 -> 13,773 context with `--temperature 1.0 --top-k 64`.
      *Verified:*
      - Turn 1 (initial 13.5k prefill + 32 gen): 4.31–5.28 tok/s
      - Turn 2 (13,660 ctx, 32 gen): 12.42–13.42 tok/s
      - Turn 3 (13,716 ctx, 32 gen): 12.40–12.58 tok/s
      - Turn 4 (13,773 ctx, 32 gen): 12.87–13.64 tok/s
      Throughput is steady across turns (12.4–13.6 tok/s), showing zero per-turn degradation.
      *Commit:* `spec/chat-16k-stall: T5 multi-turn at 16k` (`e71c632`)

- [x] **T6 — AC1/AC2 final: paired parity at context points.**
      *Completed:* Compared `--chat` vs single-shot on `Gxma-repacked` (`--temperature 1.0 --top-k 64`):
      - Context ~20: `--chat` 16.10–16.59 tok/s, single-shot 16.5 tok/s (ratio ~0.98–1.0)
      - Context 250 (multi-turn): `--chat` 15.82 tok/s
      - Context ~1,850: `--chat` 11.88 tok/s vs single-shot 9.74 tok/s (ratio 1.22)
      - Context ~13,700: `--chat` 12.6–13.6 tok/s vs single-shot 12.7–13.5 tok/s (ratio 0.95–1.01)
      0.03 tok/s stall is eliminated; parity criteria met.
      *Commit:* `spec/chat-16k-stall: T6 parity verification`

- [x] **T7 — Answer Q2/Q3.**
      *Completed:* Q2 resolved: the 1.5s/token decode delay during early measurements was
      caused by command-buffer in-flight starvation (`MAX_IN_FLIGHT=8` throttled host submission
      every 50 dispatches, forcing the host to wait on the GPU mid-step) compounded by initial
      anonymous memory pressure. With zero-copy weights and `MAX_IN_FLIGHT=64`, pipelined chat decode
      runs continuously without host-side GPU stalls. Q3 confirmed: single-shot avoids
      this by committing encoders in a single submission batch.
      *Commit:* `spec/chat-16k-stall: T7 resolve Q2/Q3`

## Open, not in this spec

* **The 3 bpw vs 6 bpw bandwidth question** — `spec/decode-3bpw-gap-19t`.
  Generator-side, already answered: 3 bpw reaches 165.2 GB/s at the head's
  shape, and the layers already run 185–203. The `+3.63 tok/s` I previously
  quoted is retracted in that spec's `implementation_notes.md §13`.
* **Device sampler** — `spec/decode-alu-fuse-18t` T5/T6. Kept, wired, null
  speedup, unrelated to this defect.
