# spec/prefill-scratch-fix

## Goal
Fix the permanent memory growth during prefill caused by the shared `xh`/`y` scratch buffer growing to accommodate large row counts (up to 8192) and never shrinking. Only KV cache should grow with context; scratch must stay bounded at decode-max size (~100 KB).

## Non-goals / restrictions
- No change to decode path (rows=1) — its scratch stays permanently allocated at max decode size (~2.2 MB: lm_head y 1 MB + xh/x_stage/yh).
- No quality loss — all token gates hold.
- No Python / no MLX.
- One small arena (xh + y + sg partial) allocated per prefill chunk, all freed after the chunk.
- Must not regress prefill throughput (T3.10 K-split still applies).

## Requirements
1. **R1**: During prefill (`rows > 1`), linear encoding uses a per-chunk scratch arena passed from `encode_span` — covering `xh`, `y`, **and the T3.10 sg K-split partial buffer** — not the shared `Exl3Gemv.scratch` / `Exl3Gemv.partial`.
2. **R2**: The shared `Exl3Gemv.scratch` only grows for `rows == 1` (decode) calls. Its max size = max over all decode shapes (~2.2 MB total: gate y 86 KB; lm_head y 1 MB).
3. **R3**: `encode_span` allocates one arena per chunk (sizes = max over linears in that chunk, computed by `Exl3Gemv::chunk_scratch_needs`), passes it to each `encode_linear_with_scratch`, and the arena is dropped (freed) when the chunk's command buffer completes.
4. **R4**: No memory leak — after prefill completes, process RSS returns to decode baseline (weights + KV + decode scratch + decode partial only). In particular the sg partial must **not** grow shared beyond its rows=1 gemv max (≤ ~344 KB): at rows=256 the prefill sg split needs **8.4 MB** (`nsplit×rows_pad×N×4`, k/v) which must live in the per-chunk arena instead (2026-09-23 audit: `plan.md` previously mis-stated this as ≤344 KB — that figure is the gemv-only number).
5. **R5**: `prefill_throughput_slow` ≥ 86.8 t/s @512 (T3.10 baseline maintained).
6. **R6**: Last-chunk `rows == 1` and non-sg-eligible shapes keep working (gemv/scalar paths unchanged; shared gemv partial allowed).

## Acceptance criteria
- Run prefill with large context (e.g., 8192 tokens) → RSS peaks at chunk size then returns to baseline after prefill (post-warmup growth ≤ 48 MB over baseline after the KV cache is dropped; shared scratch elems + shared partial bytes **exactly unchanged** across prefill — asserted by `prefill_memory_profile_slow`).
- `decode_throughput_slow` unchanged (8.39 t/s baseline).
- `prefill_throughput_slow` ≥ 86.8 t/s @512.
- Full gated suite 114/0/5.
- No new allocations during decode after warmup.