# `spec/alu-boost` — findings

Measured, model-free, on this box (M4 Pro, 20 GPU cores, 24 GiB). Instrument:
`crates/engine/examples/alu_boost.rs`. Raw log: `raw/final.txt`.

Nothing here is quoted from another spec; every number is from this instrument or
from a probe re-run today. Where an older in-tree number disagrees it is called
out as a disagreement, not resolved by assertion.

## The question

The 3 bpw EXL3 decode GEMV runs at **199.4 GB/s effective** while this box's own
streaming probe reaches **238–244 GB/s** (`probe_bw_250gb`, re-run today, 2/6/12 GB
buffers, three thread configurations, `stream_uint1` = 238.5, `stream_uint4` =
244.3). That is 82–84 % of the ceiling. The ask was: what has to change to reach
95 % (≈ 232 GB/s) — the ALU, or the access pattern?

## Result 1 — the split, at the shipping ring access pattern

All arms read the same 256 MiB with the shipping ring indexing (one 16×16 tile =
24 words at 3 bpw; each of 32 lanes issues its `j0`, `j1`, `j2` word loads out of
that same 96-byte tile). Bytes are UNIQUE bytes = what reaches DRAM. Medians of 9
interleaved rounds (one pass over every arm per round).

| arm | GB/s | vs prev | what it isolates |
|---|---:|---:|---|
| `arm_seq` — coalesced, 1 word/lane, 8-way load ILP | **217.1** | — | this harness's DRAM ceiling |
| `arm_ring1` — ring, **1** load/lane | 200.5 | −7.6 % | dropping 2 of the 3 loads |
| `arm_ring3` — ring, **3** loads/lane | 182.9 | −8.6 % | the 3-load amplification |
| `arm_extract` — + all 8 bit-window extractions, **no** decode | 162.9 | −10.9 % | the extraction ALU |
| `arm_decode` — + the real `decode_state_mcg` × 8 | 161.1 | −1.1 % | the decode ALU |

**The decode arithmetic itself is nearly free (1 %). The extraction is the ALU
cost (11 %). The ring access pattern is the bigger single item (16 %).**

Repeated runs agree on the shape (pattern 16–17 %, extraction 10–11 %, decode 1–7 %);
the decode's share is the noisiest because it is the last increment on a
memory-limited arm. Three runs: `.052`/`.083`, control drift 8.9–43.6 %.

`arm_ring1` vs `arm_ring3` is worth separating out: **the three loads per lane cost
8.6 %**, and that is a pure load-count effect at otherwise identical bytes.

## Result 2 — a bit-exact +5–9 %: unroll the tile loop four deep

`arm_decode_u4` is the SAME tile assignment and the SAME per-thread accumulation
order as `arm_decode`, unrolled four deep with the iteration stride. Because each
thread still consumes its own tiles in the same order, the float sum is unchanged.

Verified, not asserted: both arms are run over the full input in the same process
and all 2560 output words compared.

```
[check] arm_decode vs arm_decode_u4 over 2560 words: 0 differ  -> BIT-EXACT
```

| run | `arm_decode` | `arm_decode_u4` | ratio |
|---|---:|---:|---:|
| interleaved run 9 | 154.0 GB/s | 168.6 GB/s | **1.095×** |
| interleaved final | 161.1 GB/s | 169.4 GB/s | **1.052×** |

This is the one shippable lever found. It is a loop-shape change in the decode
GEMV's tile loop, it keeps the mapping, and it is proven bit-identical on real
data rather than argued from the absence of a reassociation flag.

## Result 3 — negatives, recorded because they were paid for

- **Accumulating the decoded `half2` in half precision** (removing the per-code
  unpack + convert + add, keeping one HADD2) is **0.936× — slower**, and is not
  bit-exact anyway. The float ops are already hidden behind the integer work; do
  not try to remove them. `arm_decode_h2`, two runs, 0.936× and 0.970×.
- **Repartitioning the tiles** (`tpi` = 2/4/8, i.e. letting one lane take
  consecutive tiles) is 1.04–1.11× but **changes the reduction order and is not
  bit-exact**. It is not shippable and it is not the same effect as Result 2.
- **Load width is not the wall** — reconfirmed today by the project's own probe:
  `uint1` 4 B/load = 238.5 GB/s, `uint4` 16 B/load = 244.3 GB/s, **+2.5 %** for a
  quarter of the load instructions. Note this does NOT contradict Result 1: there,
  3 loads vs 1 load at *identical bytes and identical width* is 8.6 %, which is a
  different experiment.

## Result 4 — the machine's fp32 issue ceiling, measured

`p2_*`: independent accumulator chains under `#pragma unroll` with literal indices,
swept 4/8/16/32 chains so a latency-bound result cannot be mistaken for a ceiling.

```
p2_fp4   95.0   p2_fp8  112.6   p2_fp16 116.9   p2_fp32 120.7   G warp-instr/s
```

It rises and then plateaus: **≈ 119 G warp-instructions/s** = 3.8 T lane-FMA/s,
which is 128 FP32 lanes × 20 cores × ~1.48 GHz. The sweep recovers the datasheet
width from a microbenchmark, so the instrument is calibrated.

**The `p2_int*` arms are NOT trustworthy and are reported as a failure of the
instrument**: they scale as `n²` (4→26.7 ms, 8→106.8, 16→428.6, 32→1759), which is
not a hardware property. Do not quote an integer-op ceiling from this file.
Consequently `p2_mix16` cannot answer "do the int and fp pipes overlap?" either;
that question is still open and needs a corrected generator.

## What this means for 95 % of bandwidth

- 95 % of the measured 244 GB/s ceiling = **232 GB/s**. The engine is at 199.4, so
  the ask is **+16.4 %**.
- The bit-exact unroll is worth **+5–9 %** on the decode portion → ~210–217 GB/s,
  i.e. **86–89 %** of the ceiling. Not 95 %.
- The remaining 16 % is the ring access pattern: 8.6 % from the three loads per
  lane, and ~8 % from the lanes' scattered word offsets. Closing either needs a
  change to the packing or to the lane→word mapping, not to the arithmetic.
- Instruction budget, as a cross-check: at 3 bpw the model is 32.155 G codes. At
  199.4 GB/s the engine runs 531 Gcodes/s. At 232 GB/s it needs 618 Gcodes/s.
  Against the 119 G warp-instr/s ceiling that allows **6.2 instructions per code**.
  The tree's in-kernel figure is ~9.375, so even with the ALU fully overlapped the
  current instruction count does not fit — the extraction has to shrink too.

## Reproduce

```sh
export CARGO_TARGET_DIR="$PWD/target"
cargo build --release -p kaplumba-engine --example alu_boost
scripts/gpuq try -- ./target/release/examples/alu_boost
```

Takes ~40 s, loads no model, takes no GPU lock, and self-checks the decode
transcription against an independent Rust re-implementation before reporting.

## Instrument bugs found and fixed (kept, because each produced a wrong number)

1. `out` was sized for the ring arms' 2560 writers but `arm_seq` writes by `gid`
   up to 81919 → ~320 KiB past the end → the command buffer faulted and
   `wait_until_completed` returned instantly, reporting a **1.7 TB/s "DRAM
   ceiling"**.
2. `float fa[8]` was indexed to 31 in the 16- and 32-chain fp arms → the extra
   instructions were eliminated and the fp arms looked flat.
3. Arms were swept **sequentially**, and two sequential sweeps drifted 10 % apart
   on the same arm (`arm_seq` 189.7 → 210.4 GB/s). All arms are now interleaved
   round-robin with an explicit drift control.
