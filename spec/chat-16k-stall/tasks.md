# tasks — `spec/chat-16k-stall`

One commit per task. Update this file immediately after each, with status,
commit hash and the measured number.

Measurement rules: `AGENTS.md §10`, `crux.md §2`/`§6`. ≥16 interleaved pairs,
median + spread + control drift. A repeat of the same binary has produced
0.03–0.42 tok/s for this bug, so **a median is not evidence without a spread.**

- [ ] **T0 — Freeze the two baselines, on a quiet GPU.**
      *Do:* `scripts/gpuq status` → must be idle.
      * single-shot at 614 / 2,234 / 8,192 / 16,384, `--temperature 0`,
        ≥3 repeats each, record median **and spread**.
      * chat at the same four contexts, ≥3 repeats each.
      *Record all of it in `implementation_notes.md` before changing anything.
      This is the AC1 denominator and the spread is what makes a later
      improvement distinguishable from the noise this bug already produced.
      *Commit:* `spec/chat-16k-stall: T0 paired baselines`

- [ ] **T1 — R5 FIRST: prove Bug A's reorder did not change any token.**
      *Why first:* `8fbf487` reordered buffer submission and **has never been
      shown output-identical**. It is the highest-risk unverified claim in the
      repo right now, and everything else is worthless if it corrupted tokens.
      *Do:* `cargo test --release -p kaplumba-engine --test multi_turn_chat --
      --ignored --nocapture`. Requires `KAPLUMBA_TEST_MODEL` and a release
      binary; `gpuq try --`.
      *Gate:* `pipelined_and_barrier_decode_agree_across_a_multi_turn_session`,
      `a_single_greedy_turn_reproduces_the_counting_sequence`,
      `context_advances_by_exactly_the_tokens_each_turn_consumed` all green,
      plus the AC1 prompt byte-identical to `2fff2c97…`.
      *If any fails:* **stop and revert `8fbf487`.** A fast chat that emits
      wrong tokens is worse than a slow one.
      *Commit:* `spec/chat-16k-stall: T1 token-stream equivalence for the reorder`

- [ ] **T2 — R1: fix the window. Sweep first, then change the constant.**
      *Do:* measure the in-flight sweep already queued:
      `KAPLUMBA_MAX_IN_FLIGHT` ∈ {8, 16, 32, 64} at 16k, ≥2 repeats each.
      Then decide, recording the decision and its memory consequence:
        * **raise `MAX_IN_FLIGHT`** so it exceeds one step's buffer count (12),
          or
        * **raise `MAX_DISPATCHES_PER_BUFFER`** so a step needs fewer buffers.
      Prefer whichever keeps `roll()` non-blocking AND bounds resident memory
      (R3). Record the derivation at the constant — "8" was never justified and
      is smaller than one step.
      *Test:* `roll()` provably cannot block mid-step (R2); host `tick` wait ≈ 0
      at 16k by stack sampling with sample counts printed (AC4).
      *Commit:* `spec/chat-16k-stall: T2 size the in-flight window`

- [ ] **T3 — R3: bound the back-pressure.**
      Larger window must not become unbounded submission. Assert a maximum on
      outstanding buffers, and measure `phys_footprint` via `/usr/bin/time -l`
      (never `peak_resident_pages_bytes` — it understates by 5.5 GiB).
      *Commit:* `spec/chat-16k-stall: T3 back-pressure bounds`

- [ ] **T4 — R6: a regression test that can fail.**
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
