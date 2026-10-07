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

- [ ] **T3 — Prove token-stream equivalence across multi-turn sessions.**
      *Do:* `cargo test --release -p kaplumba-engine --test multi_turn_chat --
      --ignored --nocapture`.
      *Gate:* `pipelined_and_barrier_decode_agree_across_a_multi_turn_session`,
      `a_single_greedy_turn_reproduces_the_counting_sequence`,
      `context_advances_by_exactly_the_tokens_each_turn_consumed` all green.
      *Commit:* `spec/chat-16k-stall: T3 token-stream equivalence`
      The 16k stall is **invisible in the token stream**, exactly as the ordering
      bug was — that is why it survived. So the guard needs two parts:
      a structural/source assertion (window ≥ one step's buffers; commit before
      wait), and a measured parity ratio (R4).
      *Sabotage, per `AGENTS.md §3.4`:* set `MAX_IN_FLIGHT` back to 8, run, watch
      the ratio test fail diagnosably. Revert. **An un-failed test is not a
      test.**
      *Commit:* `spec/chat-16k-stall: T4 sabotage-verified regression test`

- [ ] **T5 — R7: multi-turn at 16k, not one paste.**
      The reported symptom is a long conversation. Run ≥4 turns at 16k total
      context and confirm the rate does **not** degrade per turn. If it does,
      that is a further defect, not this one — record and open it.
      *Commit:* `spec/chat-16k-stall: T5 multi-turn at 16k`

- [ ] **T6 — AC1/AC2 final: paired parity at four contexts.**
      ≥16 interleaved pairs, chat vs single-shot, at 614 / 2,234 / 8,192 /
      16,384. Report every ratio with both absolutes and the spread.
      *Record the honest number even if a context misses 0.95* — say which and
      why.
      *Commit:* `spec/chat-16k-stall: T6 AC1 parity measurement`

- [ ] **T7 — Answer Q2/Q3, even if the answer is "not yet".**
      Why is a decode step ~1.5 s at 16k in chat when single-shot runs it in
      ~74 ms? 20×. Either the same serialisation seen GPU-side, or a second
      defect. And does single-shot carry the same latent window exposure?
      *Commit:* `spec/chat-16k-stall: T7 Q2/Q3`

## Open, not in this spec

* **The 3 bpw vs 6 bpw bandwidth question** — `spec/decode-3bpw-gap-19t`.
  Generator-side, already answered: 3 bpw reaches 165.2 GB/s at the head's
  shape, and the layers already run 185–203. The `+3.63 tok/s` I previously
  quoted is retracted in that spec's `implementation_notes.md §13`.
* **Device sampler** — `spec/decode-alu-fuse-18t` T5/T6. Kept, wired, null
  speedup, unrelated to this defect.
