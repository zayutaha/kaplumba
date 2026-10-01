# Finding 06 — IMPLEMENTED: command-buffer rolling. +4.37%, measured end-to-end.

Subagent: `ses_f0cddbc32ffedi1MkbyeC1RK2S`. Branch `impl/maxdispatches`, 4 commits,
worktree `wt-maxdispatches`, separate `CARGO_TARGET_DIR`. **NOT merged — I review
and merge.**

## Result: half the predicted win, but real and proven

| | dispatches/token | buffers/step | ms/step | tok/s |
|---|---:|---:|---:|---:|
| before | 1167.0 | 1.00 | 73.76 | 13.558 |
| after | **1167.0** | **24.00** | 70.67 | **14.150** |

**+4.37%, −3.09 ms/step. 16 paired interleaved samples, one binary, arms differing
only in the env var. Paired t = 6.13 on 14 df, exact sign test p = 0.000488
(14/15 pairs).** Dispatch count identical in both arms — the control proving only
commit boundaries moved.

The independent GPU clock agrees: `dp_span_ms_per_token` 72.929 -> 70.239
(−2.69 ms) against the wall clock's −3.09 ms.

**Predicted +8.6%, got +4.37%.** The prediction over-counted: the first 50
dispatches must be encoded before the GPU has anything, and the tail after the
last commit is exposed identically. Only ~3.1 ms of the 5.75 is recoverable.

**Also: dispatch count is context-dependent (1167 at 16 tokens, not 1617).** The
hardcoded figure was removed from comments rather than propagate another stale
number.

## Correctness gate — passes, and the harness was validated first

| threshold | sha256 of 128 greedy tokens incl. `--show-ids` |
|---|---|
| `2500` (old, no rolling) | **MATCH** `2fff2c97…6795b2` |
| default `50` (rolls) | **MATCH** |

The gate only means something because the harness was **first proven to
reproduce the hash under unchanged code**. The first attempt used the AC5 prompt
and the default model and produced a different, perfectly well-formed 128 tokens
(`0dd4c06f…`) that **would have read as a regression**. The hash was taken with
model `Gxma-repacked` and the prompt "…in simple terms, and then explain why it
matters."

**A golden hash is only meaningful next to the exact command that produced it —
that invocation is not currently recorded anywhere. Fix that.**

## Sabotage — caught, and it proved something

Poisoned `scratch.query_positions[0]` from the host at the end of
`forward_chunk`, i.e. **after the whole step is encoded and its buffers are in
flight**. It targets `query_positions` rather than `token_ids` because rope
re-reads it **every layer**.

```
got 7cce3441…  ***MISMATCH*** 126 of 128 token positions differ, first at index 2
```

**The second conclusion matters more than the first: rolling genuinely *opens*
that window.** With one uncommitted buffer there is nothing in flight to race.
So safety rests entirely on exactly one guarded host write per step and one
barrier after it — and the hash is now known to notice if that stops holding.

## Prefill unaffected

2,012 tokens: 91.927 -> 90.610 tok/s, output byte-identical. **By construction** —
the threshold sits inside `if decode_step`, so a multi-token prefill never sees
it. No decode-only scoping needed.

## Doc corrections this exposed

- `decode_profile.rs` claimed step-mode `own` "is small and not a decode cost".
  **Measured: `own` was 70.170 ms against a 72.929 span — 96% of the step** while
  the threshold was 2500, and only 5.188 ms with rolling on. **`own` measures
  the rolling threshold, not the step. Quote `span`.**
- The "~42 buffers" figure (42 = 2100/50, circular) is **gone from three files**.

## Reporting discipline

**Pair 13 reported, not dropped**: B = 0.883 tok/s, 16x its own arm median,
outside its entire range, on a box with 5+ other agents. Full 16-pair stats
recorded, with the paired *t* on that set reported as **meaningless (−0.36)**.

## Two things NOT fixed, deliberately

1. **The restored module ships a failing test**: `every_pipeline_construction_
   goes_through_the_audit` — committed `linear.rs` has **19 direct pipeline
   constructions and zero** `pipeline_audit::build` calls. That gap is
   **functional, not cosmetic**: `generate.rs`'s `assert_no_builds_since` guard
   cannot see those builds. Rewriting `linear.rs` is another agent's in-flight
   work on 19 call sites. **Re-land `703603a` with that conversion, or drop it.**
2. **A profiler defect**: `dp_wall_ms_per_token` is 27.568 against a 70.239 span
   — **impossible for a step ending in a barrier**. Do not quote `dp_wall_*` from
   a step-mode run until explained.

## Suite: 168 passed / 1 failed / 29 ignored (run twice)

The 1 failure is the `pipeline_audit` self-test. A one-off `forward_buffer_parity`
prefill failure appeared in the first run; **chased rather than dismissed** —
6/6 standalone with the threshold reverted, 6/6 on this branch, passed in the
second full-suite run. Recorded as a flake, not understood. AC6 clean.

## BLOCKER: `master`@`379ae9e` does not compile

`generate.rs` calls `kaplumba_kernels::pipeline_audit::{armed_mark,
assert_no_builds_since, audit_line_with_mark}` (added in `9cd57a7`) but the module
was **never committed** — `git log --all -- '*pipeline_audit*'` is empty. Every
engine build fails with three `E0433`s.

Restored verbatim as `703603a` (two independent worktrees carried it
byte-identically, so it was settled, not in-flight). **This must land on master
before anything else can be built or measured** — it is currently blocking
several agents simultaneously.
