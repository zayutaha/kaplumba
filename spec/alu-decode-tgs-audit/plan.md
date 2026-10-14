# Technical plan

## Architecture and source touchpoints

All line numbers below refer to GitHub revision `7ddf20f`, not local HEAD.

Chat CLI -> generation loop -> 60-layer resident forward -> transformed
activations -> EXL3 GEMVs -> finalize/norm -> norm/RoPE/cache -> attention ->
FFN -> head/softcap -> token selection -> next full forward.

| Path | Source behavior and proposed investigation |
|---|---|
| `crates/kernels/src/linear_repacked.metal:53` | Existing 32-bit eight-state MCG helper; reuse as a correctness baseline, inspect emitted code. |
| `crates/kernels/src/linear_repacked.metal:1349,1890` | `fused2_sk2` and `fused3_sk2` retain ulong construction/extraction; first candidate for exact 32-bit extraction. |
| `crates/kernels/src/linear_repacked.metal:2286,2878` | Contiguous-tile and two-column split-K families already use 32-bit extraction; investigate live ranges and scheduling only after active-route confirmation. |
| `crates/kernels/src/linear.rs:1869,2292,2738,3120` | Pipeline selection and trace; QKV split-K is default when its pipeline exists. Trace prints each distinct name once, not call counts/dimensions/times. |
| `crates/engine/src/model.rs:2085,2386` | O/down split-K selection; changing just the named non-split kernel can miss production work. |
| `crates/engine/src/model.rs:1781` | Norm/RoPE/cache fusion already selected by default; quantify remaining edges instead of re-proposing it. |
| `crates/engine/src/generate.rs:269,321,977` | Greedy GPU token handoff exists; positive-temperature branch converts logits and samples on CPU. Audit timing and sampled path independently. |
| `crates/kernels/src/encoder.rs:269` | Decode-specific rolling threshold exists; investigate gaps without confusing this with prefill thresholds. |
| `crates/engine/src/weights.rs` | GitHub tip already has shared mmap locking; full-model residency acceptance in `spec/mlock-clean-mmap` remains unmeasured. |
| `crates/engine/src/metrics.rs` | Internal decode timing must agree with independently timestamped application intervals. |

No interface/layout change is needed for the first QKV extraction experiment.
Future variants use the same buffers, shapes, grid, split boundaries and
finalizers. Variant selection must fail closed if a specialized pipeline is
missing. No runtime pipeline compilation/allocation is permitted in decode.

## Sequence and experiment design

### 1. Establish an honest baseline

Build pinned source in a dedicated worktree, with a unique `CARGO_TARGET_DIR`.
Record source and binary hashes, compiler/OS, all KAPLUMBA variables, model
headers, model identity and chat template. Verify the build includes updated
Metal sources. Record startup separately from TTFT and pure decode.

Run `KAPLUMBA_TRACE_GEMV=1` to identify names. Add a separate diagnostic
inventory in a future harness for counts, dimensions, layer role, strides and
grid. Trace alone cannot provide those or GPU times. Profiling and margin mode
disable greedy pipelining in `generate.rs`; never time those as the default.
Inspect command-buffer completion/error status, not merely elapsed duration.

Audit decode timing around first/last emitted token and executed forward steps.
In the current pipelined path the first encoder is committed before `step_start`
is set; this motivates validation, not a proven throughput bug. Early EOS,
one/two-token requests, and a queued extra step must have explicit accounting.
Keep EOS semantics; report actual counts instead of pretending every request
reaches its cap. Use identical prompt/token counts for paired throughput runs.

Read actual tensor headers to obtain unique packed bytes used by every full
forward; include explicitly named sign/scaling buffers consistently, avoid
double-counting fused tensors, and distinguish useful bytes from redundant
lane loads, KV traffic and file padding. Reject any ablation/skip-model flag
for accepted measurements. Re-measure streaming controls only when memory
conditions permit; unique faulted pages and checksums must rule out DCE/cache
reuse. Streaming controls are not physical-traffic counters.

### 2. First ALU candidate: QKV split-K extraction

Keep MCG exactly multiply -> mask/xor -> half2 reinterpret -> **half add** ->
float conversion. Preserve all eight states and the original multiply/add
expression, compiler math flags and reduction order.

For reachable lane shifts s={3,11,19,27}, construct
`H=(wa<<s)|(wb>>(32-s))`, `L=wb<<s`; states 0..5 are
`(H>>(16-3*i))&65535`, states 6/7 are respectively
`((H<<2)|(L>>30))&65535` and `((H<<5)|(L>>27))&65535`.
Ring successor is `(j+1)%24`. Use these state expressions while retaining
original scalar MCG calls and their loop structure, rather than directly
substituting the vectorized helper: the standalone probe found raw FP32
differences with the vector-helper substitution. This is eight-state extraction,
not Gemini's four-state replacement. Prove reachable offsets at each call site;
if a future caller admits s=0 it needs guarded shifts. Do not extrapolate this
K=3 helper to fractional or other integer rates.

Compare original QKV split-K kernels with extraction-only variants. Use the
scalar `decode_state_mcg` call per state: the pre-existing vector helper changed
6,607/24,576 synthetic raw FP32 words despite finalized-output tests passing.
Start with fused2, then fused3 separately. No change to split count, tile ownership,
register scheduling or buffers in the same comparison. The baseline loop has
`#pragma unroll` and constant K: per-state shifts may already fold to constants.
Source ulong removal therefore predicts no particular gain; measure actual
assembly/counters when available and application tg/s before retaining it.

### 3. Next ALU/load experiments

Only advance where measured critical-path contribution warrants it:

- Specialize four reachable shift classes inside the same kernel while keeping
  original lane/accumulator ownership. A lane-dependent switch can serialize
  SIMD paths; templates/function constants alone cannot specialize a lane-varying
  value. Consider explicit constant expressions/selects and measure code size,
  occupancy and latency. Do not add a redundant FC_BITS cache for existing
  statically K=3, MCG kernels.
- Sweep register lifetime and load lookahead by 0/1/2 tile trips; reuse decoded
  storage immediately and preserve per-accumulator visitation order. Sweep unroll
  only along the *existing iteration stride*, including tails; increasing the
  contiguous tile assignment changes the reduction and fails the exactness gate.
- Reduce repeated address work or share ring loads with shuffles while preserving
  logical lane outputs. Existing cooperative loads/uint4 widening had negative
  results; require a materially different, documented hypothesis before retrying.
- Lossless layout experiments must recover every original 16-bit state and
  preserve accumulation ownership. Expanded states/shifted copies trade ALU for
  extra resident bytes and DRAM traffic; calculate that trade before a GPU test.
  Any checkpoint/layout change needs a separately amended spec and compatibility
  plan. A 3-bit symbol-only layout loses trellis history and is excluded.

### 4. Additional tg/s paths

- **Sampled chat:** time logits transfer, FP16->FP32 conversion, allocation,
  probability filtering/sorting and synchronization. First evaluate reusable
  host storage/exact existing algorithm; then an exact device sampler only if
  distribution, seed and RNG-consumption parity can be established. Existing
  device sampling machinery must be audited for reachability before new work.
  Greedy throughput does not establish sampled-chat throughput.
- **CPU/GPU overlap:** audit already-existing double-slot token/position handoff,
  encoding time, gaps and command-buffer rolling. Benchmark only actual remaining
  waits/allocations. Never remove producer/consumer ordering or let host writes
  race GPU reads. Test slow callbacks, EOS rollback and backpressure.
- **Long context:** profile full versus sliding attention, cache addressing and
  unchanged-order memory scheduling. Altering attention split count changes
  summation order; the historical 16->13 experiment diverged at long context.
  Do not promote it under an exactness promise. Pure address/load scheduling or
  launch fusion preserving intermediate FP16 rounding is eligible.
- **Residency:** check shared mmap/Metal aliasing, actual lock success, page faults,
  pressure and swap deltas during live chat. Restore healthy memory conditions
  before interpreting ALU measurements. The recent shared-mmap fix is already
  baseline; no claim that it has completed its model-throughput acceptance.

### 5. Validation and decision

Correctness before speed: independent bitstream reader -> all 65,536 MCG states
on GPU -> packed tiles -> actual raw FP32 and finalized FP16 -> complete logits
and generated token IDs. Test all real shapes plus valid short/tail shapes,
ring wrap, sign/permutation paths, full/sliding attention, and unchanged fallback
rates. Sabotage each changed dispatched branch; never use a switch that disables
the candidate while claiming coverage. Preserve baseline shader/compiler options.

ABBA blocks, >=16 pairs after warmup; application timing uninstrumented, causal
GPU attribution in separate matched diagnostic runs. Report clock/pressure
controls and all contaminated pairs. Keep candidate only if quality passes and
paired tg/s improvement is repeatable. Do not add marginal ablation deltas:
changed traffic/overlap makes them non-additive and outputs are invalid.

## Risks and fallbacks

- GPU currently has insufficient unused memory for a model run. Continue small
  CPU work; return exit 75 to non-GPU tasks rather than bypass the lock.
- Source changes can compile identically, or reduce ALU while worsening occupancy.
  Retain baseline selection and record negatives.
- Historical instruments disagree and one integer microbenchmark scaled n².
  Repair DCE/LICM and indexing defects before using their conclusions.
- Exactness can bar faster reductions or samplers. Keep them unshipped; no waiver
  is implied by the tg/s goal.
- 250 application GB/s may exceed the attainable roofline. Keep the target's
  definition and publish the measured feasible rate rather than inflate bytes.

## Requirement mapping

R1 -> T1; R2 -> T1,T2; R3 -> T2,T3,T4,T5,T6,T7;
R4 -> T3,T4,T5; R5 -> T6,T7; R6 -> T2–T8; R7 -> T1,T2,T8.
