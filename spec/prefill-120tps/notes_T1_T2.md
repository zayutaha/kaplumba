# T1/T2 handover notes — MMA QK scoring (agent ATTN)

Worktree: `/private/var/folders/tp/wz3877_s1z756p1kcsz3_d4h0000gn/T/opencode/p120-attn`, branch `impl/attn-mma`.
Files owned: `crates/kernels/src/attention.metal`, `crates/kernels/src/attention.rs`, new tests
`crates/engine/tests/attention_*_mma*.rs`, this file.

## Goal

T1: replace the scalar per-lane QK dot product in the **prefill** GQA kernel with
simdgroup-MMA tiles. T2: A/B it against the scalar path (attention ms/token via
`KAPLUMBA_GPU_PROFILE=1`, interleaved pairs, median + spread) and enable by default only
if it wins ≥1.5× at 8k context (spec acceptance #5).

## Design as implemented (differs from plan.md — reasons below)

**Fragment mapping (both loads, no staging of K in the fast path):**

- `A` = K tile, **device** load, no transpose:
  `simdgroup_load(kf, keys + k0*kv_stride + head_base + d8, kv_stride)` →
  `A[key][d] = K[k0+key][d8+d]`. This is the exact pattern `linear.metal:10391` already
  ships (device source, stride as `elements_per_row`, default origin), so it is proven on
  this hardware.
- `B` = Q, **threadgroup** load, **transpose**:
  `simdgroup_load(qf, qhstage + d8, HEAD_DIM, ulong2(0,0), true)` →
  `B[d][row] = qhstage[row][d8+d]`. Threadgroup+transpose is in the MSL header
  (`metal_simdgroup_matrix:115-117` passes `transpose_matrix` through) but was NOT already
  used anywhere in this repo; `linear.metal` never transposes. This is the one unproven
  API call in the design — if outputs look like row/column-scrambled dots, this is the
  first suspect.
- `C = A×B` with M = 8 keys, K = 8 dims, N = 8 rows:
  `C[key][row] = Σ_d K[key][d8+d]·Q[row][d8+d]` = the score dot product, pre-scale.

**Why N = all 8 wave rows even though one simdgroup only keeps its own row:**
the 8×8 instruction executes N=8 regardless of what you put in the lanes, so filling all
8 N rows with real Q rows costs exactly the same instruction count as filling 6 of them
with garbage. Every simdgroup therefore computes scores for all rows the threadgroup
staged (`qhstage[8][HEAD_DIM]`, rows 0..NSG-1 real, rows NSG..7 deliberately unwritten
garbage — their C columns are discarded) and keeps only column `sg`. Consequence:
QK MMA instruction count per threadgroup is `NSG × (keys/8) × (HEAD_DIM/8)` per pass,
i.e. up to NSG× the theoretical minimum (only one sg could compute everything and
publish). We deliberately did **not** build the publish variant: it costs an extra
barrier + a shared score buffer + serialises QK onto one simdgroup, and the
register-extraction variant costs nothing extra. If measurements show issue-bandwidth
limitation this is the follow-up lever.

**Why NOT plan.md's `Q_TILE=1`:** the scalar body (`attention_gqa_body:738-935`) gives
each simdgroup exactly ONE (query, head) row — `query = tgid/kv_heads`,
`head = key_head*groups + sg` — and processes keys in 32-key tiles with one barrier per
tile. plan.md's "threadgroup already holds 8 (query,head) rows" is true at the
*threadgroup* level (NSG rows of the same query, same kv head) but not per simdgroup.
M-Q_TILE=1 (one row per sg, M-waste 7/8) and M=8 (this implementation) are the same
instruction count; M=8 additionally removes the fake-lane loads. There is no
"Q_TILE=8 exploiting existing rows" saving to be had beyond this.

**Score extraction stays in registers — zero extra barriers vs scalar:**
for fp32 8×8, lane `t` holds `C[t>>2][(t&3)*2]` and `C[t>>2][(t&3)*2+1]` (M-key, N-row).
The sg's row is `sg`, whose pair column is `pair = sg>>1`, held by lanes
`4m+pair` (one per key m). Per 8-key sub-tile each lane computes
`mine = pair-matching element of its own held pair`, then one
`simd_shuffle(mine, 4*m+pair)` fetches key `m`'s score; lanes in `[sub*8, sub*8+8)` keep
the result in their scalar-layout score register (`lane = key within the 32-tile`).
After the four sub-tiles the scalar steps 2–4 (online softmax `simd_max`/`simd_sum`,
double-buffered `wbuf`, PV loop) are **copied verbatim**, so barrier count per 32-key
tile is identical to scalar (1) in the fast path. The fragment-layout assumption above
is exactly what `attention_qk_mma_matches_scalar` verifies; sabotage log below.

**Tail sub-tile** (`k0+8 > range_end`, at most once per pass, uniform across the
threadgroup so its barrier is safe): cooperative guarded fill of `kbuf[8][HEAD_DIM]`
(zeros past `range_end`), one `threadgroup_barrier`, same A/B/MMA from threadgroup
memory, extraction unchanged. Out-of-range keys then get `score=NEG` through the scalar
`live` test (`live = active && k < range_end`), same as scalar — no OOB device read is
ever issued.

**Q staging:** halves (not floats like scalar's `qfbuf`) are copied from device into
`qhstage[sg*HEAD_DIM]`, one barrier, same position as scalar's `:818`. The MMA path
never converts Q to float at all (scalar's `:799-815` float hoist exists for its scalar
fmadd loop). `kbuf`+`qhstage` threadgroup use: NSG=8/HEAD_DIM=512 → 2KB wbuf + 8KB
qhstage + 8KB kbuf ≈ 18KB < 32KB.

**Decode is untouched by construction:** `attention_gqa_body` (scalar) is not modified;
`KAPLUMBA_FA_DECODE`/`KAPLUMBA_FA_SINGLE` macros and their instantiations are not
modified; the new `attention_gqa_mma_body` + `KAPLUMBA_FA_GQA_MMA` are appended **at the
end of attention.metal** so no existing line numbers shift (several test doc comments
cite metal line numbers).

## Host side (attention.rs)

- `AttentionQk { Scalar, Mma }` + `attention_qk()` env selector, `OnceLock`, idiom
  copied from `decode_single_pass_max_keys` (`attention.rs:127`):
  - `KAPLUMBA_SCALAR_ATTENTION=1|on|true|yes` → Scalar (brief's A/B spelling),
  - else `KAPLUMBA_MMA_ATTENTION=1|...` → Mma,
  - else **default = Scalar** (see conflict below).
- Dispatch body moved into `pub fn encode_online_attention_qk(&self, encoder, request, qk)`
  (defined BEFORE `encode_online_attention`), which is also what the correctness test
  calls with both `AttentionQk` values in one process. `encode_online_attention` keeps
  exactly the one `ablated("attention")` and forwards `attention_qk()`.
- New pipelines `attention_gqa_mma_pipelines` built **eagerly in `new()`** through
  `pipeline_audit::build` (required by `pipeline_warmup_coverage.rs`, which also asserts
  a dispatch after `new()` builds nothing).
- Grid/threadgroup geometry unchanged: MMA only swaps the pipeline for the
  `gqa_index(head_dim, groups)` arm of prefill.

## Constraints found in tests I do not own (all verified by reading the files)

1. `probe_budget_reconcile.rs:600-629` — exactly ONE `ablated("attention")`, it must be
   after `rfind("pub fn encode_online_attention(")`, and the text after that point must
   not contain `encode_decode_single_pass`. Helper named `encode_online_attention_qk`
   placed before `encode_online_attention` does not match the `rfind` needle (needle ends
   in `(`).
2. `perf_probe_criticalpath_latency.rs:1198` — file must contain
   `let groups = q_heads / kv_heads;` (kept verbatim in the dispatch).
3. `probe_cache_hierarchy.rs:414-428` — every `threadgroup_barrier(` in attention.metal
   must be spelled `threadgroup_barrier(mem_flags::mem_threadgroup)`.
4. `pipeline_warmup_coverage.rs:252-307` — all pipelines eagerly built in `new()`; a
   dispatch must not compile anything afterwards.
5. `attention_single_pass_bit_exact.rs:253-362` — `run_prefill_gqa` calls
   `encode_online_attention` and asserts **0 differing fp16 elements** vs the single-pass
   decode kernel. MMA reassociates the dot product, so it CANNOT satisfy this.

## The default-flip conflict (recorded per brief; STOP condition)

If T2 measures ≥1.5× and MMA is enabled by default, test 5 above fails (prefill MMA vs
decode-single scalar are not bit-identical). `attention_single_pass_bit_exact.rs` is not
owned by this agent. **Therefore: default stays Scalar in this agent's commits.** The
win (if any) is demonstrated with `KAPLUMBA_MMA_ATTENTION=1` and handed to the
coordinator, who must decide between (a) editing that test's prefill arm to force
`AttentionQk::Scalar` or to relax the equality to a tolerance, or (b) accepting the
path split. Nothing in this worktree flips the default.

## Sabotage log (AGENTS.md §3.4)

(filled in after the test is written — see bottom of file, updated as work proceeds)
