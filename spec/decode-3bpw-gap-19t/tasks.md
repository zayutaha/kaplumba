# tasks — `spec/decode-3bpw-gap-19t`

One commit per task. Update this file **immediately** after each task with
status, commit hash, and the measured number.

Measurement rules: `AGENTS.md §10`, `crux.md §2` and `§6`. Every rate needs
`scripts/gpuq status` idle first — `crux.md §2.7` records a **factor of two**
error from a load that left no trace in `ps`/`top`.

- [ ] **T0 — wait for a quiet GPU, then establish the box's own ceiling.**
      *Do:* `scripts/gpuq status` → must report idle with no model resident.
      Re-run `crux_bandwidth` (synthetic, no model) to confirm the box still
      delivers 245–253.6 GB/s. **If it does not, every rate in this spec is
      void** — fix that first, not the kernel.
      *Test:* control arm's spread < 2%.
      *Commit:* `spec/decode-3bpw-gap-19t: T0 box ceiling re-confirmed`

- [ ] **T1 — run the never-executed discriminator (R1).** ⭐ THE GATE
      *Files:* none. `crates/engine/tests/perf_probe_head_vs_layers.rs` exists
      and is complete; `grep -rn bw_ratio spec/` returns empty, so it has
      **never been run**.
      *Do:* `KAPLUMBA_TEST_MODEL=~/.kaplumba/models/Gxma-repacked
      scripts/gpuq try -- cargo test --release -p kaplumba-engine --test
      perf_probe_head_vs_layers -- --ignored --nocapture`
      Run **Control R** (`head_shape_at_three_bpw`), **Control S**
      (`layer_shape_at_six_bpw`), **Control G** (`head_grid_thickness_sweep`).
      *Decision rule, pre-registered before the run:*
        - `bw_ratio ≈ 0.5` → **(d) bits-per-code is the story** → T2 is the work.
        - `bw_ratio ≈ 1.0` → the head's advantage is **shape** → T2/T3/T4 are
          still worth doing (they are shape-local) but **Tier 3 is dead** and
          must be struck from `spec.md`.
      *Record:* all three controls' printed verdicts verbatim, even if
      unflattering. This is the measurement the whole repo has been missing.
      *Commit:* `spec/decode-3bpw-gap-19t: T1 run the head-vs-layers controls`

- [ ] **T2 — attribute the gap (R2).** From T1's three controls, kill three of
      (a) parallelism / (b) contiguity / (c) K / (d) bits-per-code with a
      measured number each. Update `spec.md §4 R2` with the surviving cause.
      *Commit:* `spec/decode-3bpw-gap-19t: T2 gap attributed`

- [ ] **T3 — Tier 1: `qkv_proj` at K=5376 (R3).** 143.2 GB/s, worst shape,
      shortest K on the live path. Worth **+0.59 tok/s** alone.
      *First:* answer Q3. `gate_up` shares K=5376 and gets 203.1 GB/s, so K
      alone does not explain the deficit. Determine whether `qkv`'s fused q+k
      dispatch pays twice for the activation read before writing a kernel.
      *Do:* sweep K for the qkv shape to separate "short K" from "fused q+k".
      *Test:* ≥194 GB/s on the qkv shape, greedy byte-identical, or record dead
      with the mechanism.
      *Commit:* `spec/decode-3bpw-gap-19t: T3 qkv_proj to pack`

- [ ] **T4 — Tier 2: `o_proj` at K=8192 (R4).** 185.1 GB/s. Same class as T3.
      A/B'd **alone**, not bundled (cf. `121ff5b`: the real cross-layer fusion
      win was +0.09 tok/s, +0.12/+0.069 in two 8-pair blocks).
      *Commit:* `spec/decode-3bpw-gap-19t: T4 o_proj to pack`

- [ ] **T5 — Tier 3, only if T1 selects (d) (R5).** Port the head's 192 B/tile
      to 3 bpw: two adjacent 3 bpw tiles per fetch.
      *Constraint:* identical tile→lane assignment and accumulation order, or it
      is a 1-ULP change needing a waiver. `alu-boost §3` measured
      repartitioning at `1.04–1.11×` and **not shippable**.
      *Test:* `repacked_golden` bit-exact; sabotage by reordering one
      accumulator must fail diagnosably; ≥16-pair end-to-end A/B.
      *Commit:* `spec/decode-3bpw-gap-19t: T5 192 B tile at 3 bpw`

- [ ] **T6 — AC1/AC4 final measurement.** ≥16 interleaved pairs against the
      frozen baseline binary, greedy, AC1 prompt. Report median, spread,
      control drift. **Report the honest number even if < 17.2** — a missed
      target with a trustworthy method beats a met target with a leaky one.
      *Commit:* `spec/decode-3bpw-gap-19t: T6 AC1 A/B`

- [ ] **T7 — AC7 negative-results table (R7).** At minimum re-state with
      numbers: occupancy (20k→82k threads, −1.9%), split-K (120 cells, +0.76%
      < noise), 4col (−0.7% in 6/6 pairs), `uint4` (180.5→173.7),
      `simd_shuffle` (180.5→146.5), o_proj↔attention overlap (spin 18.6 s /
      2M iters; MTLEvent 0.93–1.02 wash), dispatch (50/250/2500 → 15.93/15.88,
      **re-measured today**).
      *Commit:* `spec/decode-3bpw-gap-19t: T7 negative results`

## Open, not in this spec

* **The O(N²) `Detokenizer::push`** — a separate agent owns the 16k `--chat`
  collapse. Touches `tokenizer.rs`, not the GEMVs. No file overlap with T3–T5.
* **Speculation** — the only route past the pins, ruled out by the user.
* **The `125–150 GB/s` production figures** — settled as ablation artefacts
  (`crux.md §7`). Do not re-derive.