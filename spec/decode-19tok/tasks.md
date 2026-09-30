# Tasks: decode 8.1 to 19 tok/s

For `spec/decode-19tok/spec.md`. Read `plan.md` for the sequencing rationale —
the order here is forced by information, not preference.

Rules for every task, without exception:

- **One GPU-touching agent at a time.** Contention produced the two largest
  errors in this project's history (248.5 GB/s, and a prefill profile
  misattributed to decode). Check `scripts/gpuq status` before every model run
  and **do not sit idle waiting for a memory number**: a load is verified to
  work at 6.7 GiB "available" with 0 swap, because the model sits in clean
  reclaimable file cache that macOS evicts on demand. The real constraint is
  unreclaimable anon+wired, and the warning that a load "will swap" is too
  conservative. What actually matters is that no *other* agent is holding the
  GPU -- check for foreign processes, not for a headline number.
- Report **median**, never best. Never report a number from a run that swapped.
- Probes must **free memory per configuration**. A probe holding 3.8 GB without
  releasing drove the box into 573 MB of swap; a leaked probe is grounds to kill
  it on sight.
- Delete any temp example or worktree when done.
- Never commit to master from a worktree.

---

## T1 — A decode timing instrument that can fail (R1)

*Blocks T3 and T5. Do first.*

- Diagnose why `KAPLUMBA_GPU_PROFILE=1` reports a flat ~1270 ms for decode
  across N=16/32/64/128, with `path:line`. Check the `GPUStartTime`/`GPUEndTime`
  selectors and the rolling encoder in `crates/kernels/src/encoder.rs`
  (`MAX_IN_FLIGHT=8`, `MAX_DISPATCHES_PER_BUFFER=50`).
- Build the instrument in a worktree. It must be demonstrated to:
  - scale linearly with token count over N=16/32/64/128 (report the fit);
  - exclude load and prefill (prefill must not move when decode length changes);
  - resolve an injected cost (add a slow dispatch, confirm it is seen, remove it);
  - reproduce ~123 ms/token against ground truth: 7.841 / 8.176 / 8.062 / 8.050
    / 7.947 tok/s at N=64, clean box.
- State its noise floor and minimum detectable effect.

**Test:** the linear fit is monotone; the injected cost appears with
approximately the right magnitude; the instrument agrees with 123 ms/token.

**Done when:** the instrument's four properties are demonstrated, not asserted,
and the minimum detectable effect is documented so T3 knows what counts.

## T2 — Settle the streaming ceiling (R3)

*Can run alongside T1 — needs no instrument, only a standalone probe.*

- Sweep concurrency past the old 20,480-thread limit to the device maximum
  (threads/threadgroup x threadgroups x blocks/thread); find the knee.
- Sweep independent address streams, 1 to 64. The weights are 23 x 512 MB arena
  segments and a decode step reads many at once. Re-test the prior "multiple
  queues do not help" negative — that is a different axis and it was taken on a
  contended box.
- Vary bytes-in-flight and thread count as **separate** axes. If they are not
  interchangeable, that is direct evidence of a request-rate limit.
- Push past 235 GB/s if anything reaches it (above NVIDIA's GB10, an identical
  memory part at 229-230 GB/s). Try 2 MB pages: a latency probe found 4 KB TLB
  misses raise latency 98 ns -> 423 ns across our working-set sizes.
- Report one number: median, best, worst, spread, n>=7.

**Mandatory:** read the trap table in `plan.md` before writing any probe. Build
an independent reference and make it fail first. `crates/engine/examples/bw.rs`
is repaired (`9906610`) — verify it writes before trusting it.

**Done when:** a single ceiling number with spread, and a plain statement of
whether 220 GB/s is a wall.

## T3 — Attribute the 123 ms (R2)

*Blocks implementation. Needs T1.*

Wall-clock differential, each quantity varied alone and required to be linear in
it:

| isolate | method |
|---|---|
| weights vs KV | context 614 vs 8192; KV scales with L, weights do not |
| per-layer cost | synthetic 2-layer model, linear in layer count |
| `lm_head` | its own shape: 262144 wide, 6 bpw, 1.057 GB |
| dispatch gaps | rolled run vs single command buffer |

- Produce components summing to 123 ms, each with achieved GB/s against
  220 GB/s.
- Answer: 13.145 GB at 220 GB/s is 60 ms; we take 123. **Where are the other
  63 ms?** Candidates not yet eliminated: dispatch/command-buffer gaps;
  per-thread bytes-in-flight at batch 1 (86,016 threads per decode layer);
  dequant not overlapped with loads (prefill's dequant was free, but prefill is
  compute-bound and decode is not, so it does not transfer); `lm_head`'s shape.
- Report the split-count result separately (see T4 note).

**Test:** components sum to 123; each isolation is linear in what it claims.

**Done when:** the 63 ms is attributed, with the largest component's achieved
GB/s and the single change most likely to close it.

## T4 — Bytes per token, costed (R4)

- Cost every mechanism that moves fewer bytes, in ms/token at 220 GB/s
  (1 GB/token = 4.5 ms = ~0.6 tok/s):
  - `lm_head` row pruning, up to 1.057 GB. A vocabulary projection; rows that
    cannot win an argmax are pure cost. **Pruning by frequency is a quality
    decision, not a free win** — say so rather than assuming it away.
  - `lm_head` at 6 bpw: double the density of the 3.000 bpw projections, with no
    free reduction available.
  - Re-confirm on a quiet box that every weight is read exactly once per token.
  - Projection fusion: only worth it if it removes a re-read, and the re-read
    check says there are none. Expect nothing; confirm.
  - **Speculative decoding**: the only lever that beats the byte floor rather
    than trimming it, because weights do not change between tokens. Verify-gated
    and greedy-identical per AGENTS.md §8. A different project from the others;
    named here so it is not lost if T2 returns "wall" and the rest are exhausted.
- **Split count, as a sub-item. — DONE 2026-10-01, shipped default-off.** The
  1.91x figure is superseded: the direction held, the magnitude did not.
  `decode_splits` was not passed `kv_heads`, so it aimed at a *thread* count and
  gave both shapes 16 splits with grids differing 4x. Three requirements, three
  outcomes:
  - **Justified by wave quantisation, not one data point** — the one-wave budget
    is **54** workgroups, measured by sweeping three shapes that bracket it (54
    one wave, 55 two), and `54 / kv_heads` reproduces the measured optimum of all
    three. **1.34x at 8k, 1.11x at 1k, 0% change to the sliding layers by
    construction.**
  - **Float divergence measured**: 1 fp16 ULP on 3-4 of 16384 output elements,
    i.e. the fp16 floor and four orders below the 2.16e-3 prefill accepts — and
    it still changes **43 of 128** greedy tokens over a 6010-token context. The
    five reference prompts are unnecessary and insufficient: the recorded
    128-token gate reproduces its hash under *both* arms, because at 156 keys the
    keys-per-split floor caps both at 3 splits. It is structurally blind to this
    change.
  - **End-to-end ms/token: NOT done.** Every number is the isolated kernel on a
    contended box. This is the one requirement left open.
  - Shipped as `KAPLUMBA_DECODE_SPLIT_RULE=workgroups`, **default off**; the
    decision to flip it is a judgement about the model, not the kernel.
    `KAPLUMBA_DECODE_SPLITS` still pins the count for sweeps.

**Done when:** every mechanism is costed in ms/token and ranked, with quality
implications stated rather than assumed.

## T5 — Implement, one change at a time (R2, R4, R5, R6)

*Serial. One change per commit, each verified before the next.*

Order by measured size, not by ease. For each:

1. Implement in the main tree (not a worktree) so the commit lands cleanly.
2. `cargo test --release -p kaplumba-engine` — 20/20 suites.
3. **Sabotage the change, confirm a test fails with a diagnosable number, then
   revert the sabotage.** A test that cannot fail is not a test. Several
   harnesses here shipped numbers that measured nothing, and `decode_splits`
   has no unit test at all while a hardcoded `splits = 8` vs the shipped 16
   passed until the data was varied — so a real test with varied inputs is part
   of the change, not optional.
4. Five reference prompts exact: Paris / 1, 2, 3, 4, 5 / red, blue, and yellow /
   Hello! How can I help you today? / 17 times 23 is **391.**
5. Measure with T1's instrument, n>=7 interleaved, median and spread, on a box
   with >= 14 GiB free.
6. Commit with the measured number, the spread, and what the change did *not*
   fix. Update this file and `implementation_notes.md`.

**Do not pursue the `uint4` change on its original justification.** The 30 B/thread
target that motivated it came from a wrong threshold; the load-width sweep found
**8 B/thread already saturates**, so the load is past the knee. If it is measured
at all, it must be measured on its own merits, not on that reasoning.

**Done when:** acceptance criteria 1-5 of the spec are demonstrated, or a
requirement is formally waived with the reason recorded.

---

## Open, and deliberately not scheduled

- The 16 B/thread figure is retracted; see the spec's Corrections. Do not
  re-derive it.
- Full weight residency is rejected on measurement (R6). Do not revisit without
  new evidence.
- Multi-queue streaming was a negative on a contended box; T2 re-tests the
  adjacent question (address streams, not queues) properly.
- Prefill-side work is out of scope for this spec even where the same kernel
  serves both. Decode shares the attention body, so decode-side wins will show up
  in prefill and vice versa — measure the side you are changing.
