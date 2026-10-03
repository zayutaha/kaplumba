# Decode Ceiling — Serialized Test Log

One line per tested hypothesis. Format below is mandatory; a result without an
interval and a verdict is not a result.

```
ID:      Pnn
Agent:   <one implementation agent, sole GPU access>
Hypothesis:  <one sentence>
Change:  <files, functions>
Session baseline:  <measured in the SAME session, median + spread>
Noise floor:       <same-config control arm>
Result:            <median + spread, n pairs>
Verdict:           CONFIRMED | REFUTED | INCONCLUSIVE
Regression:        3 bpw bit-identical? yes/no (n runs)
Sabotage:          <what you broke, and the failure message you observed>
Spec side-effects: <any hypothesis invalidated>
```

---

## Results

(nothing tested yet — the GPU slot is held by the 4-bit fix and memory agents)

## Queue

Ordered by expected value, **R1 (bytes) first** because kernel speed alone is
arithmetically capped below the target:

1. P19 mixed precision — attacks bytes; format may already support it
2. P08 head precision / 2-stage head — 0.53 GB/token for 4.4%
3. P17 dispatch-selection audit — cheap, may find a silent fallback
4. P03 why the head is 2x faster — may unlock the layers
5. P02 dispatch/gap overhead — bounded at ~7.32 ms/step
6. P20 critical-path latency — resolves the head-vs-layers contradiction
7. P01, P04, P05, P12, P13, P16, P18 — kernel-level
8. P14, P15 — matter mainly at long context
9. P06 — the temperature>0 non-pipelined path, which the user's actual decode
   setting uses and which every 16.0 tok/s figure so far has NOT measured

## Standing constraints for every test

- One model process at a time. Never two. Exit code 75 = busy, retry, not a failure.
- `scripts/gpuq try -- <cmd>` by default; never `run` to block.
- Compare generated **text only** — exclude the `--metrics` line from any hash.
- `/usr/bin/time -l` `peak memory footprint` is the only trustworthy memory
  metric; `maximum resident set size` under-reports by ~8 GB.