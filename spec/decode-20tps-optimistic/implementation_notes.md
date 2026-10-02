# IMPLEMENTATION NOTES — DECODE 20 TPS (OPTIMISTIC BUILD)

Living log. Append, never rewrite. Numbers quoted with model path,
prompt, and commit.

## Sources of record (read before touching code)

- `MEMORY.md` — headline arithmetic, branch state, non-GEMV tax (6.70 ms,
  floor 2.6–2.9 ms), queued lever list with ms values, rules learned.
- `spec/decode-19tok-repacked-250gb/findings/07-non-gemv-audit.md` — the
  256-thread-grid kernel table (~4.0 ms addressable, launch vs work split).
- `spec/decode-19tok-repacked-250gb/implementation_notes.md` §20–§23 — the
  ALU partition probe, funnel-shift fix, 2col tiling, all with probes.
- `spec/decode-20tps/implementation_notes.md` §16–§18 — production GEMV
  ~126 GB/s vs microbench 199, working-set artifact analysis.
- Branch `impl/attn-splits` @ `8852a16` — the split-rule commit message
  contains the full measurement record (9 paired rounds, GPU clock).
- `spec/decode-25tok-peak-bandwidth/implementation_notes.md` §16 — head
  at 253 GB/s, the existence proof for T2.

## The budget (from spec.md)

20 tok/s = 50.0 ms/step = 47.4 ms GEMV @254 GB/s + 2.8 ms non-GEMV floor.
Every task in tasks.md is priced against this table:

| lever | expected ms | state |
|---|---|---|
| sampler O(n) top-k | −0.89 | landed `40e018d` |
| funnel-shift 64→32b | −2.57 (qkv) | landed §22 |
| rolling command buffers | ×0.956 of step | merged `integration/rolling` |
| attn split rule | −2.13 @8k | branch, default-off |
| cache_append fold | −0.21 | branch (described, 0.21 measured) |
| norm load hoist | −0.6…1.2 | branch |
| argmax device | +0.35% | test + index fix needed |
| GEMV 200→254 | −10…15 | T2 |

## Environment / measurement recipe

```bash
scripts/gpuq status                       # before any model run
scripts/gpuq try -- <cmd>                 # exit 75 = busy, retry with CPU work
M=~/.kaplumba/models/Gxma/Gxma            # or Gxma-repacked; quote which
P=$(python3 -c "print('Explain the theory of relativity in simple terms, and then explain why it matters. '*N)")
# 16 paired A/B samples, interleaved, min with spread, GPU clock noted
```

## Open items / open questions

- T1.1 flips summation order: needs golden re-baseline sign-off.
- T0.2 may conclude dp_span cannot be restored under rolling — if so,
  every number in the §20 partition table must be re-derived with the
  successor instrument before T2 is credited with anything.

## Done log

(empty — first entry follows T0.1)
