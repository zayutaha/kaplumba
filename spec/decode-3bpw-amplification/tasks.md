# TASKS: Decode at the EXL3 3 bpw Floor — 16 tok/s

Replaces the T1–T5 of `74398f9`. **T1 there is now complete and returned a negative** —
see `implementation_notes.md` §3. The amplification programme is closed; `spec.md` §3 is
the disproof. What remains is the ~2–4 ms in `spec.md` §4.

Order: each task is independently verifiable, committed on its own, and **gated on
beating baseline by measurement, not on being implemented.** A task that does not beat
baseline is recorded as a negative (AC6) and the next one still runs.

- [x] **T1 — Gating experiment: can 3 bpw lane-load amplification be reduced by relayout?**
  - Files: none. Pure CPU arithmetic, no production code touched.
  - Intent: test the withdrawn proposal — a lane-contiguous layout putting lane L's 8
    codes at bit offset `24·L`, which the previous draft predicted would reach amp
    2.000 at an unchanged 96 B/tile.
  - **Result: NO. Impossible, and the disproof is structural.** `decode_state_mcg`
    (`linear.metal:121-126`) multiplies before masking, so all 16 bits of the window
    affect the output and a lane cannot be rebuilt from its own 8 codes. Measured three
    ways: bit-flip enumeration → **37 bits per lane** (min=max); symbolic → `a0` window
    span 37 bits → 2 words for all 32 lanes unconditionally; **union over all 32 lanes =
    37**, i.e. every lane reads the same 37-bit window at a different rotation. 37 > 32,
    so 2 words per lane is a floor, giving amp `64/(8·KA)` — which is what the kernel
    already does. amp is set by the bit rate, not the layout.
  - Commit: folded into the correcting commit. Details in `implementation_notes.md` §3.

- [ ] **T2 — Norm second-pass register hoist (measure first, it may be worth 0)**
  - Files: `crates/kernels/src/elements.metal` (the `rms_norm_weighted_tg` / 1-group norm
    body), `crates/engine/src/elements.rs`, `crates/kernels/src/encoder.rs` if a
    dispatch is removed
  - Intent: the norm kernels are **latency-bound, not bandwidth-bound**. 256 threads walk
    5376 elements in 21 strided loads, hit 8 `threadgroup_barrier`s, then walk 21 more
    loads plus 21 stores. ~10.5 µs predicted against 11.2 µs of measured work; the fp32
    ridge point is 35 FLOP/B and every row is 9–140× below it. Hoist the second pass's
    `weight`/`residual` loads into registers above the barrier. Legal: the first loop
    writes only `threadgroup partial[]`.
  - Testing:
    - **Bit-exact**: the 256-lane strided partial plus tree reduction *is* the reduction
      order and is frozen by the no-quality-loss rule. `partial[lid] = (lid==0)?sum:0`
      changes 8192/8192 elements, worst 9266 ULP. So: assert `to_bits()` equality on
      every output element, and assert the threadgroup width is unchanged.
    - **Measure** the win before keeping it. The record's own warning is "may be worth 0
      — do not budget before measuring". Report the number either way (AC6).
    - **Sabotage**: widen the threadgroup and confirm the bit-exactness test fails
      (this is the known-frozen axis — the failure must be observed, not assumed).
  - Commit: `spec/decode-3bpw-amplification: T2 norm second-pass register hoist`

- [ ] **T3 — Remaining norm fusion, one dispatch at a time**
  - Files: `crates/kernels/src/elements.metal`, `crates/engine/src/elements.rs`,
    `crates/engine/src/model.rs`
  - Intent: land the four measured-safe fusions, each separately, each with its own
    before/after: (a) `cache_append` into `norm_rope_qkv` — **0.21 ms**, LOW–MED risk;
    (b) `post_attn_norm_res` + `pre_ffn_norm` + gate transform → one dispatch — **0.42 ms**,
    MED; (c) cross-layer: layer N+1 `input_layernorm` into layer N `post_ffn_out` —
    **0.21 ms**, MED; (d) down finalize + `post_ffn_out` — **0.21 ms**, MED.
  - Testing:
    - **The traps, all measured, all required fixes:**
      - `float(half(normed))` is load-bearing. Carrying fp32 across the rotation is a
        *different function*: 328 of 1024 elements differ, worst 19 ULP.
      - Same trap in an o_proj/attention fusion: the separate path **stores fp16** and the
        transform reads it back. Keeping fp32 gives 3461–7005 of 8192/16384 elements
        differing by 1 ULP.
      - A poison-the-*output* test **passed while sabotaged**; poison the *input*.
    - Each fusion: interleaved A/B, medians with spread, ≥ 8 pairs, on an idle box.
    - **Sabotage** per fusion.
    - If a fusion does not beat baseline, revert it and record the negative (AC6).
  - Commit: `spec/decode-3bpw-amplification: T3 remaining norm fusion`

- [ ] **T4 — Attention/GEMV overlap on a second command queue**
  - Files: `crates/engine/src/model.rs`, `crates/kernels/src/encoder.rs`
  - Intent: the last untried item. Attention is **1.2–2.3 ms**, not the 3.2 ms the older
    plan assumed. Overlap KV streaming on a secondary `MTLCommandQueue` with the GEMV on
    the primary, synchronised by `MTLEvent`/`MTLSharedEvent` with no CPU block.
  - Testing:
    - **There is no reference precedent.** oMLX deliberately runs everything on one
      worker so MLX operations are never concurrent (`engine_core.py:521-528`,
      `grep mx.new_stream` = 0 matches). Do not expect this to be a known-good shape.
    - Synchronisation correctness first: attention depends on Q/K/V and writes
      `scratch.attended`; o_proj consumes it. Prove ordering with a poison test on the
      *input*, run ≥ 1000 steps, assert bit-identical tokens every step. A race that
      only shows up 1-in-10^4 is worse than no overlap.
    - Then A/B the step time. **If it does not beat baseline, revert and record.**
    - Watch the GPU-access interaction: the engine's `flock` is per-process, so a second
      queue does not weaken the one-model-process rule, but a contended run fabricates
      results (a sweep once returned plausible medians with a fabricated +34%).
  - Commit: `spec/decode-3bpw-amplification: T4 attention/GEMV overlap`

- [ ] **T5 — Acceptance**
  - Files: `spec/decode-3bpw-amplification/implementation_notes.md`
  - Testing: AC1 ≥ 16.0 tok/s median of ≥ 8 interleaved paired runs on ≥ 128 tokens with
    baseline re-measured in-session; AC2 bit-exact golden text; AC3 the four suites;
  AC4 sabotage records; AC5 `/usr/bin/time -l` footprint ≤ 16.5 GiB, 0 swap; AC6 every
  negative recorded with its number.
  - Commit: `spec/decode-3bpw-amplification: T5 acceptance`
