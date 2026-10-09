# `spec/alu-boost` — plan

## Architecture

One self-contained example, `crates/engine/examples/alu_boost.rs`. The Metal
source is **generated in Rust** and compiled at runtime with
`new_library_with_source`, so:

- there is no `include_str!` staleness hazard (editing a `.metal` file is
  invisible until the host crate recompiles — a trap already paid for twice in
  this tree), and
- the op count of every body is a property of the generator, not a hand-typed
  constant.

Functions are emitted with **concrete names** (`arm_seq`, `p1_arm0`, `p2_int4`),
never as C++ templates: the runtime compiler mangles template-instantiation names
and `Library::get_function("foo<4u>")` will not find them.

## Part 1 — data flow

Input is 256 MiB of pseudo-random u32 in one `StorageModeShared` buffer. 320
threadgroups × 256 threads. Every arm except `arm_seq` walks 24-word tiles.

Shared prologue, emitted into every ring arm verbatim from the shipping kernel's
`exl3_decode8` at `ka = 3`:

```
lo = (lane*8 + 1)*3 + 768 - 16;  sh = lo % 32;  i0 = lo / 32;
j0 = i0 % 24;  j1 = (j0+1) % 24;  j2 = (j1+1) % 24;
```

Then per tile: three word loads, the two 64-bit windows `a` and `c`, and eight
16-bit fields at bit offsets `64 - (i*3 + 16)`.

The arms vary exactly one thing each:

| arm | variable |
|---|---|
| `arm_seq` | none — coalesced, 8 independent loads per thread |
| `arm_ring1` | 1 load instead of 3 |
| `arm_ring3` | 3 loads, 2 extractions, no decode |
| `arm_extract` | + all 8 extractions, summed as integers |
| `arm_decode` | + `decode_state_mcg` × 8, f32 accumulation |
| `arm_decode2/4/8` | tiles per lane per iteration (repartitions) |
| `arm_decode_u4` | 4-deep unroll, **same** mapping, in-iteration stride |
| `arm_decode_h2` | decode with a `half2` accumulator (not bit-exact) |

## Part 2 — ALU issue ceiling

Independent accumulator chains in a `#pragma unroll` loop with literal indices,
swept 1..32 chains so latency cannot be mistaken for throughput. `p2_fp*` is
valid; `p2_int*` and `p2_mix*` are not (see findings — `n²` scaling).

## Measurement discipline

- **Interleaved**: for each of 9 rounds, every arm runs once; then per-arm median.
  Two sequential sweeps drifted 10 % apart on the same arm, so non-interleaved
  comparison here is worthless.
- **Control**: `arm_seq`'s min/max across rounds is printed as a drift figure.
- **Self-check before timing**: the decoded f32 for (lane 0, tile 0) is compared
  against an independent Rust re-implementation of `decode_state_mcg` that does
  real round-to-nearest-even binary16 arithmetic. Timings are refused if it fails.
- **Bit-exactness check**: `arm_decode` and `arm_decode_u4` are dispatched over the
  full input and all 2560 output words compared bitwise.

## Touchpoints

- `crates/engine/examples/alu_boost.rs` — new, the whole instrument.
- Nothing in `crates/kernels/src/*.metal` or `crates/engine/src/` is modified.
- `probe_bw_250gb` (existing) is re-run, not edited, to get the box's own ceiling.

## Risks and fallbacks

| risk | fallback |
|---|---|
| arms too small to be DRAM-bound | `arm_seq` is measured first and printed; if it is far below `probe_bw_250gb` the run is read as *relative* only |
| compiler eliminates a body | each check and each arm prints a checksum-bearing output word; the decode arm is bit-compared against Rust |
| box drift | interleaved rounds + printed drift control |
| template-name mangling | concrete function names only |

## Requirement → task mapping

| requirement | satisfied by |
|---|---|
| 1 harness ceiling | `arm_seq` + `probe_bw_250gb` re-run |
| 2 one-variable decomposition | `arm_ring1/ring3/extract/decode` |
| 3 interleaving | 9 interleaved rounds + `arm_seq` drift control |
| 4 transcription check | `decode_mcg_ref` assert before timing |
| 5 bit-exact claim | 2560-word bitwise comparison of `arm_decode` vs `arm_decode_u4` |
| 6 negatives | `arm_decode_h2`, `arm_decode2/4/8`, `p2_int*` |
