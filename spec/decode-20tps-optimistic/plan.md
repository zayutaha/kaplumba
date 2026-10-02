# PLAN: Decode to 20 tok/s — Optimistic Build

Spec: `spec.md`. The arithmetic budget (50.0 ms/step) is unchanged; this
plan is the mechanics of collecting the 13 ms.

## Sequencing (every step measured, interleaved A B A B, 16 paired samples)

### Track 0 — Unblock the shared tree (enables everything else)

T0.1 Route the 19 direct `linear.rs` pipeline constructions through
`pipeline_audit::build` (or delete the audit module) so
`every_pipeline_construction_goes_through_the_audit` passes.
T0.2 Give `dp_span` exclusion logic for the rolling dispatch window (or
re-document `the_decode_instrument_measures_a_step_and_only_a_step`) so
`decode_profile_gpu.rs` is green with rolling on. **Attribution must keep
working — every other lever is measured through it.**
T0.3 Merge `integration/rolling` (command-buffer rolling, +4.37%,
13.558 → 14.150 tok/s) onto the working branch. Conflicts: the two
notes-union merges on `8549e6e`.

### Track 1 — Collect the queued, measured non-GEMV levers (~3.5 ms)

T1.1 **Attention split rule `54/kv_heads`** (`impl/attn-splits`, `8852a16`):
cherry-pick. Full-attention layers 1.34× at 8k; sliding layers provably
unchanged (budget 288 wg > cap). Gated off by default because it moves
summation order; for this program we **adopt it as default** and re-derive
the greedy golden against a 128-token reference, recording the new hash —
the ULP drift is 1-on-16384 and the repo has already accepted larger
1-ULP flips (gate/up fusion). [Verification: attention tests + golden
re-baseline, explicit user sign-off recorded here.]
T1.2 **Fold `cache_append` into `norm_rope_qkv`**: V groups currently
write exactly those elements to staging; relabel the store address.
Zero extra groups, zero extra bytes, one fewer dispatch. −0.21 ms.
T1.3 **Hoist `weight`/`residual` loads above the threadgroup barrier** in
the 1-group norms. Unmeasured; A/B it. Potential −0.6…1.2 ms.
T1.4 **Device-side argmax** (`crates/kernels/tests/argmax_device.rs`):
fix the `0xFFFF - index` underflow for index ≥ 65,536 (pack index
biased, compare, branchless), then swap the host argmax out. +0.35%.
T1.5 Sampler is already O(n) after `40e018d` (2.78 → 1.89 ms). Nothing
to do; record it.

### Track 2 — GEMV streaming: 200 → 254 GB/s (the 10–15 ms)

The 3 bpw long-K GEMVs sit at 196–201 GB/s after §22's funnel-shift;
the pure-stream probe is 254 and the 6 bpw head hits 253. The remaining
gap is a per-threadgroup streaming-efficiency property (finding 09 ranks
it: per-TG, not TG-count, not occupancy). Optimistic candidates, in
order of prior art:

T2.1 **Tile-size matching**: the head's advantage is 192 B/tile (6 bpw)
vs our 96 B (3 bpw). Repack the 3 bpw weights so one lane-group reads
two adjacent 96 B tiles (the `contig4` layout already carries 4 tiles;
verify it is actually used by all four projections, not only the head).
Zero math change → bit-exact by construction.
T2.2 **Dual-column ILP everywhere**: `colmajor_2col` landed (§23) for
`down/gate/o`; confirm `qkv` and `gate_up` take the 2col path and that
fused gate/up keeps `out_features % 32 == 0` true on both halves.
T2.3 **Software-pipeline the decode chain**: issue the next tile's
`decode_state_mcg` before the current FMA's dependent use (the ALU
partition shows 63% decode-ALU, 30% neither — the neither is the
serialized wait). If T2.1 lands first, re-measure before doing this.
T2.4 **Fuse the V staging**: deleting `scratch.k/scratch.v` writes via
T1.2 removes 2 of the 7 full-step round trips over 16384×60 fp16.

Every T2 step: bit-exact kernel tests (`test_metal_gemv_repacked_bit_exact`,
`live_kernel_*`, `fused_bit_exact`) **plus** a 16-paired-sample step-time
A/B at `K≈154` and 6010-token context, min-of-pairs with GPU clock
report.

### Track 3 — Proof

T3.1 End-to-end: greedy 128-token generation byte-compared to the
recorded golden (or the explicitly re-baselined one from T1.1).
T3.2 `/usr/bin/time -l` footprint ≤ 16.5 GiB at 8k, 0 swap.
T3.3 Report the final spread across 3 runs, min used.

## Interfaces / formats

- No weight-format change: T2.1 uses the existing `contig4` packing;
  only the *consumption order* changes.
- `KAPLUMBA_DECODE_SPLIT_RULE=workgroups` stays as the escape hatch.
- `KAPLUMBA_NO_2COL`, `KAPLUMBA_NO_FUSED_QKV/GATEUP` stay as escapes.
- Golden-hash policy documented in `implementation_notes.md`: hash
  always quoted with the exact prompt and model path.

## Risks & fallbacks

| risk | handling |
|---|---|
| 1-ULP attention drift accepted in T1.1 | escape flag; re-baseline documented |
| T2.1 zero-gain (gap is DRAM-page, not tile) | stop T2.2/2.3, publish the "126→200→254" table as the ceiling story and take the win from Track 1 + rolling |
| dp_span invalid under rolling | T0.2 treats this as a test bug unless the instrument proves unfixable; never silently drop attribution |
| two-agent contention on GPU | `scripts/gpuq try`, exit 75 → productive CPU work, retry |
