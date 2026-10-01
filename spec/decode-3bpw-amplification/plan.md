# TECHNICAL PLAN: 3 bpw Lane-Load Amplification — 18 tok/s Decode

Implements `spec/decode-3bpw-amplification/spec.md`. Read that first; this file is *how*.

---

## 1. Architecture & data flow

### 1.1 The change, in one picture

```
BEFORE — ishizuki ring layout, 96 B/tile
  tile = 24 u32 words = 768 bits, one bit per weight slot, MSB-first
  lane L reads a 2-word window straddling its 8 codes, shifted by sh0
    lo0 = (L*8+1)*3 + 768 - 16 ;  sh0 = lo0 % 32 ;  j0,j1 = (lo0/32, +1) mod 24
    extract_v_mcg_32(t[j0], t[j1], sh0, rsh0, v)   -> 2 loads/lane, 8 codes
  32 lanes x 2 loads = 64 lane-loads / 24 words = amp 2.667   @ 199.4 GB/s

AFTER — lane-contiguous layout, 96 B/tile  (SAME BYTES)
  lane L's 8 codes occupy bits [24L, 24L+24) of the tile, MSB-first
  32 x 24 = 768 = exactly the tile -> zero padding
    L*24 mod 32 cycles 0, 24, 16, 8
    offsets 0 and 8  -> 24 bits fit in one word      -> 1 load
    offsets 24 and 16 -> 24 bits straddle two words  -> 2 loads
  16 lanes x 2 + 16 lanes x 1 = 48 lane-loads / 24 words = amp 2.000  @ ~250 GB/s (predicted)
  and every shift becomes an immediate: code i is at bit 3i, no 64-bit barrel shift
```

### 1.2 Why the byte count cannot move

The tempting "fix" — pad each lane to a 4-byte word — reaches amp 1.333 but costs
96 → 128 B/tile, i.e. **+33% bytes**. That is byte-for-byte the 4 bpw configuration,
which was **measured 6.7% slower** (`spec.md` §6). The layout above is the only
variant that reaches amp ≤ 2.0 with the byte count held at 96, because 32 × 24 = 768
divides the tile exactly with no remainder to absorb.

### 1.3 Data flow through the engine

Unchanged. The new layout is a *within-tile* bit permutation, so:

- `shape`, grid mapping, threadgroup count, accumulator count, reduction tail, and the
  `linear_finalize*` epilogue are all untouched.
- Only the **address arithmetic inside the K loop** changes, plus the offline repack.
- The `x` activation reads (`xr0[0], xr0[1], xr0[8], xr0[9]`, `:1826-1833`) are indexed
  by `r0 = (lane % 4) * 2` and are **independent of the weight layout** — they stay.
- Scales/offsets are not in the tile at all (the codebook is folded in;
  `linear.metal:121-126`), so there is nothing to rescale.

---

## 2. Interfaces, formats, on-disk layout

### 2.1 New layout name

`"lanecontig"` — a sibling of the existing `"colmajor"` value in
`quantization_config.json`. Selected by the loader the same way `perm_mode` is today
(`linear.rs:1353` checks `packed_offset % 4` and the layout key).

**Tile geometry is unchanged: 48 u16 = 96 B = 24 u32 words.** A `lanecontig` checkpoint
has the *same* `total_size` as `Gxma-repacked` (16,028,388,994 B). This is the check
that proves no bytes were added — assert it in the repacker and in a test.

### 2.2 The permutation, defined precisely

Forward (offline, `crates/engine/src/exl3.rs`):

```
input : 24 u32 words, ishizuki ring order, MSB-first, 768 bits
output: 24 u32 words, same MSB-first convention
for L in 0..32:
    for i in 0..8:
        code = extract_ishizuki_lane_codes(words, L)[i]      # existing, exl3.rs:664
        set_bit(out, 24*L + 3*i + 2 .. 24*L + 3*i) = code    # 3 bits, MSB-first
```

Inverse: the same loop with `get_bit`. **Both are required** — R2 demands a round-trip
proof, and the inverse is what the bit-exactness test compares against.

`extract_ishizuki_lane_codes` is reused unchanged. The new code must be a *new*
function; `repack_trellis_3bpw_tile` (`exl3.rs:687-704`) is lossy and stays dead
(`spec.md` §3.7).

### 2.3 Kernel-side addressing

Replace, per lane:

```
lo0 = (lane*8u + 1u)*3u + RING - 16u ;  sh0 = lo0 % 32u
hi0 = lo0 / 32u ;  j0 = hi0 >= WORDS ? hi0-WORDS : hi0
j1 = j0+1 >= WORDS ? j0+1-WORDS : j0+1
extract_v_mcg_32(t0[j0], t0[j1], sh0, rsh0, v)
```

with:

```
const uint base = lane * 24u;            // bit offset of this lane's codes
const uint w0   = base >> 5u;            // first word
const uint off  = base & 31u;            // 0, 24, 16, 8 (cycles with lane)
const bool straddle = (off + 24u) > 32u; // off in {16, 24}
uint A = t0[w0] << off;  if (straddle) A |= t0[w0 + 1u] >> (32u - off);
// 8 codes at compile-time immediates:
v[i] = decode_state_mcg( float((A >> (24u - 3u*i - 2u)) & 0x7u) )   // i = 0..7
```

`w0 <= 23` always, and `w0 + 1 <= 24` is only read when `straddle`, so there is no
out-of-bounds word at the tile seam. Because `off` is one of four values the
`straddle` branch is uniform across most of a warp, but it is **not** warp-uniform in
general — so it must be a *predicated load*, not an `if` around a barrier. There is no
barrier in the K loop, so a plain ternary is fine.

### 2.4 The `straddle` branch is a divergence risk — precedent

A previous lane-dependent branch in this kernel cost **−15 to −20%**: branching on
`lane == 0` to handle the ring wrap in vector loads, reverted in favour of uniform
loop-invariant indexing (`spec/decode-25tok-peak-bandwidth/implementation_notes.md:54-56`).
Here the branch selects between 1 and 2 loads and is uniform for 16 of 32 lanes, so it
may cost something. **T1 must measure both the predicated and the branch-free
(select-always-2-loads) forms** so the cost is known rather than assumed.

---

## 3. File & module touchpoints

| path | current behaviour | intended change |
|---|---|---|
| `crates/engine/src/exl3.rs` | `reorder_tensor_to_colmajor` (`:733-757`) permutes whole 48-u16 tiles, hardcoding `48` (`:740`). Lossy bit-plane repacker at `:687-726` is dead. | **Add** `repack_tile_lanecontig(words: &[u32;24]) -> [u32;24]` and its inverse, plus `repack_tensor_lanecontig(packed_u16, in, out)` that applies it tile-wise and preserves column-major tile order. Do not touch `reorder_tensor_to_colmajor`. |
| `crates/engine/src/bin/repack_model.rs` | Writes `layout: "colmajor"` and 64-byte-aligns tensor starts (`:188`, `:208-214`). | Add `--layout lanecontig`; assert output size equals the colmajor output size. |
| `crates/kernels/src/linear_repacked.metal` | `extract_v_mcg_32` (`:53-64`) does the 2-word funnel shift + 8 runtime-offset barrel shifts. 3 bpw live kernels at `:580-715`, `:1221-1384`, `:1782-1882`, `:1884-2018`. | **Add** `extract_v_mcg_lanecontig(t0, lane, v)`. **Add** `_lanecontig` variants of the four live kernels rather than editing them in place, so the default path cannot regress. |
| `crates/kernels/src/linear.rs` | Env gates memoised in a `Mutex<HashMap>` (`:45-58`), read once per process. Routing at `:1790-1912` / `:1915-2056` / `:2059-2218` / `:2221-2399` / `:2402-2556`. | Add `KAPLUMBA_LANECONTIG` (default **off**) routing each projection to its `_lanecontig` variant when the layout key is `lanecontig`. |
| `crates/engine/examples/bench_rate_sweep.rs` | Synthesizes weights per rate, medians of 9 rounds, prints amp + residual (28 KB). | **T1's home.** Add a `lanecontig` arm at 3 bpw next to the existing `@3bpw` arm, so the prediction is measured before any production code exists. |
| `crates/engine/tests/rate4_colmajor_bit_exact.rs` | Owns `lane_load_amplification_is_2_667_at_3_bpw_and_2_000_at_4_and_6` (`:682`) and `dram_line_traffic_is_1x...` (`:746`). | Add the amp ≤ 2.0 assertion for `lanecontig`, computed by the same exact-rational style. Do not modify the existing assertions. |
| `crates/engine/tests/live_kernel_bit_exact.rs` | Asserts the live kernel is fp16-bit-identical to `_colmajor` on 3 real shapes (`:63-67`). | Add `lanecontig` to the compared set. |
| `spec/decode-3bpw-amplification/implementation_notes.md` | — | Running log (§5 of AGENTS.md). |

### 3.1 Deliberately untouched

- `lm_head` — 6 bpw, EXL3 arm, different kernel family (`linear.rs:2012-2038`).
- `linear_gemv_ishizuki_k3_colmajor_stream` (`:1544-1651`) — already a
  **barrier-free, no-threadgroup-reduction** variant with one output tile per simdgroup.
  If T1 shows the barrier tail matters, this is the existing alternative to port from;
  it is not on the decode path today and wiring it is a separate task.
- `linear_gemv_ishizuki_k3_colmajor_u4` (`:2500-2605`) — the only `uint4` decode load,
  and it is **threadgroup-staged** (3072 B/tg, 2 barriers per 4 tiles), which is exactly
  the variant that measured 180.5 → 173.7. Recorded so it is not mistaken for a
  direct-load precedent.
- The env-gate pattern, `shape` layout, `max_dispatches` / command-buffer rolling.

---

## 4. Sequencing and dependencies

```
T1  gating micro-experiment  (bench only, no production code)
      |  measures: amp, GB/s, and the straddle-branch cost
      |
      +--> NEGATIVE --> stop. Record the measurement. Spec closes with 3 bpw retained.
      |
      v
T2  lossless repack + CPU round-trip proof          (exl3.rs + test)
      |  proves R2, no GPU needed
      v
T3  _lanecontig kernels + bit-exactness vs _colmajor  (metal + tests)
      |  proves R1, R3
      v
T4  dispatch wiring behind KAPLUMBA_LANECONTIG       (linear.rs)
      |  proves R4; A/B in-engine
      v
T5  end-to-end acceptance: >=18 tok/s, memory, no regressions   (proves AC1-AC6)
```

T2 is CPU-only and needs no GPU, so it can proceed while the box is busy. T1, T4 and T5
need an idle GPU and are serialised against each other per AGENTS.md §10.

**Requirement → task map**

| requirement | tasks |
|---|---|
| R1 amp ≤ 2.000 at 96 B/tile | T1 (measure), T3 (assert + implement) |
| R2 provably lossless | T2 |
| R3 bit-exact end to end | T3, T5 |
| R4 gated, reversible | T4, T5 (AC6) |
| R5 sabotage evidence | T2, T3 |
| R6 measurement discipline | T1, T4, T5 |
| R7 residency unchanged | T2 (size assert), T5 (AC5) |

---

## 5. Risks and fallbacks

| # | risk | likelihood | mitigation / fallback |
|---|---|---|---|
| 1 | **The `straddle` branch costs more than the 8 loads it saves.** A lane-dependent branch previously cost −15…−20% here. | **medium-high** | T1 measures predicated vs branch-free. If the branch is expensive, the fallback is the **`colmajor_stream` morphology** (§3.1): one output tile per simdgroup removes the cross-simdgroup reduction *and* the 2 barriers per tile (`:1646-1650`), which may pay for the branch. |
| 2 | **Amp 2.0 is necessary but not sufficient.** The 4/6 bpw arms reach 250 GB/s, but they are 1.5–2 bytes/lane of *useful* data per load; we would be 3 bytes per load. | medium | T1 settles it. Fallback: the oMLX `constexpr`-template route (`quantized.h:708-716`, `:741-754`, `:799-817`) makes all shifts immediates, which is an independent ~? gain. Report both. |
| 3 | Load-time regression from a new checkpoint. | low | Same byte count, same arena, one extra 16 GB read at load. Budget it and measure; the original stays untouched so there is always a fallback. |
| 4 | A new resident checkpoint pushes past 16.5 GiB. | **low** | Byte count is asserted identical to `Gxma-repacked`, which already measures 7.6–8.5 GiB resident. |
| 5 | The prediction is simply wrong and 48 loads buys little. | **medium** | T1 exists precisely to find this cheaply, before any production code. A negative result is a legitimate, useful outcome and gets written into `implementation_notes.md` with the number. |
| 6 | A second agent rebuilds `target/` mid-measurement. | low | Dedicated worktree + own `CARGO_TARGET_DIR` (AGENTS.md §10.4). Already established at `.../T/opencode/wt-rate4`. |

### 5.1 Measurement traps to avoid (all previously hit in this repo)

- `gpuq`'s lock is **engine-only**, so `gpuq try -- <benchmark>` runs happily *beside* a
  12 GB model load. A contended sweep once returned plausible medians at spread 0.46
  with a fabricated **+34%** gap. Gate on `gpuq status` showing `gpu: none`, and apply a
  repeatability floor per cell (`findings/09:69-80`).
- Estimate with **min over interleaved rounds, not median** — a median-based run was
  250% off on `down` purely from contention.
- This box drifts ≥10% between sweeps. Interleave A/B and report spread.
- `LinearKernels::new()` returning `Err` used to be a silent `else { return }`, turning
  a one-character syntax error into a green test and a bandwidth number computed from a
  library that did not contain the kernel. Both GPU tests now `expect()` it
  (`rate4_appendix:168-173`). **Third instance of a fallback that fails open.**
- The running-LCG tile generator makes the colmajor repack the identity, so any layout
  check against it is vacuous. `bench_residency`, `live_kernel_bit_exact` and friends
  **still have it**; use splitmix64 over `(tile, code)` for the new layout's tests
  (`rate4_appendix:155-166`).
- `peak_resident_pages_bytes` is `ru_maxrss` and misses Metal buffer pages entirely.
  Use `/usr/bin/time -l` `peak memory footprint` (AGENTS.md §10.3).
