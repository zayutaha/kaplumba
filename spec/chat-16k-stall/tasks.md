# tasks — `spec/chat-16k-stall`

One commit per task. Update this file immediately after each, with status,
commit hash and the measured number.

Measurement rules: `AGENTS.md §10`, `crux.md §2`/`§6`. ≥16 interleaved pairs,
median + spread + control drift. A repeat of the same binary has produced
0.03–0.42 tok/s for this bug, so **a median is not evidence without a spread.**

- [ ] **T0 — Eliminate the anonymous weight memory / swap cliff.**
      *Why first:* Live `vmmap` of stalled process PID 68980 confirmed that
      `IOAccelerator (graphics)` holds 13.4 GiB of dirty anonymous buffers, of
      which **12.5 GiB was pushed into macOS compressed swap**, causing 237,000
      pageins and collapsing decode to 0.03 tok/s.
      *Do:* Transition EXL3 layer `.trellis` and sign components to zero-copy
      file-backed Metal views (`new_buffer_with_bytes_no_copy`), pre-faulting via
      `madvise(MADV_WILLNEED)`. Apply `F_NOCACHE` on file reads to prevent
      duplicating 15 GB into the kernel Unified Buffer Cache.
      *Gate:* `sysctl -n vm.swapusage` shows 0 B swap growth; memory pressure
      remains **green** under 16k context decode.
      *Commit:* `spec/chat-16k-stall: T0 zero-copy file-backed weights to eliminate swap`

- [ ] **T1 — Fix missing KV cache retraction on EOS in `chat.rs`.**
      *Why:* `generate.rs:1050` calls `model.retract_gpu_caches(1)` on EOS to
      retract the speculatively encoded step $N+1$, while `chat.rs:507` breaks
      without retracting, corrupting the KV cursor across multi-turn sessions.
      *Do:* Add `self.model.retract_gpu_caches(1)` to `chat.rs` EOS path.
      *Commit:* `spec/chat-16k-stall: T1 retract KV cache on EOS in chat`

- [ ] **T2 — Fix `max_in_flight()` parser clamp and size the window.**
      *Do:* Fix `encoder.rs:242` where `.filter(|value| (1..=MAX_IN_FLIGHT).contains(value))`
      silently clamped any value $> 8$ to 8. Allow user configuration up to 64.
      Ensure the window holds one decode step without mid-step blocking.
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
