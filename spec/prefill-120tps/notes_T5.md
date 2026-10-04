# T5 — context extension to 24k: handover notes (agent CTX)

Branch `impl/ctx24k`, worktree
`/private/var/folders/tp/wz3877_s1z756p1kcsz3_d4h0000gn/T/opencode/p120-ctx`,
`CARGO_TARGET_DIR=/private/var/folders/tp/wz3877_s1z756p1kcsz3_d4h0000gn/T/opencode/p120-ctx-target`.
Files owned: `crates/engine/src/model.rs`, `crates/engine/src/gpu_cache.rs`,
new tests `crates/engine/tests/*ctx24k*|*context*`, this file.
Run logs: `/private/var/folders/tp/wz3877_s1z756p1kcsz3_d4h0000gn/T/opencode/t5-logs/`.

---

## 1. Premise check: what `HOST_TOKEN_SLOTS = 8192` actually gated (T5's premise was wrong)

Code audit (line refs on base commit `43a4176`):

| site | what it does | blocked 24k? |
|---|---|---|
| `model.rs:2999` declaration | — | — |
| `model.rs:3173` size of `scratch.host_tokens` (8192 u32 = 32 KiB) | staged decode-step tokens | **no** — indexed by *decode step within a run*, never by position |
| `model.rs:4066-4069` bounds `count` in `sample_capped_row_ranked` | test/debug entry point | **no** — not on the run path |

**The doc comment was wrong**: it claimed the constant is "the context limit the
decode loop is capped at". Grep of the workspace finds **no decode loop reading
`HOST_TOKEN_SLOTS`** — not `chat.rs:425-477` (`step` bounded only by
`max_new_tokens`), not `generate.rs:1013-1057` (`1..total_to_generate`).
The comment is corrected in this change.

What already bounded a 24k run before T5:

1. **Policy window:** `DEFAULT_CONTEXT_LIMIT = 32768` (`generate.rs:35`),
   enforced by `fits_in_context` (`chat.rs:352`) and `generate.rs:131-145,
   253-266` (`prompt + max_new ≤ context_limit`).
2. **KV growth bound:** `max_sequence_len = max_position_embeddings = 262144`
   (`model.rs:3584`), clamped at allocation to `MAX_POLICY_CAPACITY =
   DEFAULT_CONTEXT_LIMIT = 32768` (`gpu_cache.rs:59,228`). The old hardcoded
   8192 growth bound was already removed before T5 (`model.rs:3569-3583`).
3. **KV ladder:** 1.25× from `INITIAL_FULL_CACHE = 512` (`model.rs:1077`).
   At 24k the ladder runs … 22765 → **28457** positions (≤ 32768), i.e. full-attention
   KV ≈ 28457 × 8196 B × 10 layers ≈ **2.17 GiB**. Sliding layers: capacity fixed
   at 1024, ring, `ensure_capacity` only called `if full` (`model.rs:1811`).
4. **`config.rs:845` and `safetensors.rs:1074` (`max_position_embeddings:
   8192`) are BOTH inside `#[cfg(test)] mod tests` fixtures** (`config.rs:821`,
   `safetensors.rs:963`). Not real config paths. Untouched.
5. Other 8192s checked, not blockers: `model.rs:1808/1956/4222-4237/5677`
   (comments + a safetensors-alignment test's `PAYLOAD`), `bench.rs:88`
   (32×256 geometry), `cli/src/report.rs:18` (doc), `chat.rs:594-600` (8192 is
   a *literal argument* testing `fits_in_context` arithmetic — passes unchanged),
   `perf_probe_host_ops.rs:49-50` (local duplicate, eprintln text only).

**Empirical premise test (before touching anything):** T0's 24k run used
`--max-new-tokens 1`, which means `generate.rs:988` never enters the pipelined
decode loop — T0 demonstrated *prefill only*. Run A below ran 24k prefill +
8 real decode steps at base commit with `HOST_TOKEN_SLOTS=8192`:
**it works** (`generated_tokens=8`). So the constant blocked neither prefill
nor decode at 24k. See §2 for the defect it *does* have.

## 2. The real defect: under-sized, not blocking

`host_tokens` is indexed by `step_idx`, and the largest legal `step_idx` is
`max_new_tokens - 1` where `generate.rs:131-145` only bounds
`prompt + max_new ≤ 32768`. So `--prompt "Hi" --max-new-tokens 10000` is a
**legal run today** whose GPU write (`host_token_out[step_idx]`) and host read
(`read_host_token`, `model.rs:3992-3997`, an unchecked `*ptr.add`) go
**out of bounds** of the 8192-slot buffer. Not reachable in a 24k run with a
normal token budget, but reachable inside the policy window. That — plus AC9's
literal "HOST_TOKEN_SLOTS raised to ≥ 24k" — is why the constant is raised
rather than left alone: it is not an unused buffer, it is an under-sized one.

## 3. Change (only `crates/engine/src/model.rs`)

- `HOST_TOKEN_SLOTS = crate::generate::DEFAULT_CONTEXT_LIMIT` (32768). One
  source of truth: the buffer covers every index a legal run can produce, and
  it now equals `MAX_POLICY_CAPACITY` (the KV bound), so the policy window can
  no longer drift apart from the buffer (that drift is how 8192/32768 happened).
- Doc comment rewritten: removes the false "decode loop cap" claim, states what
  the constant actually gates and why it is the policy window.
- Cost: `host_tokens` 32 KiB → 128 KiB, allocated once at load. Nothing else
  changes size; prefill never indexes `host_tokens`.
- `crates/engine/src/gpu_cache.rs`: **no net change** (used only for sabotage
  experiments, all reverted; `git diff` on it is empty).

## 4. Tests

New: `crates/engine/tests/context_limit_ctx24k.rs` — 6 tests, all green:

1. `the_staged_token_buffer_covers_the_whole_legal_step_index_space` —
   `HOST_TOKEN_SLOTS ≥ 24k` (AC9), `≥ DEFAULT_CONTEXT_LIMIT` (OOB fix),
   `== MAX_POLICY_CAPACITY` (single window).
2. `the_host_tokens_buffer_is_really_slots_wide` — the actual Metal buffer
   length, so decoupling the allocation from the constant fails too.
3. `a_24k_prompt_fits_the_policy_window_with_generation_headroom` —
   `fits_in_context(23954, 0, …)` at exact limit / one over, error payload, and
   `MAX_POLICY_CAPACITY ≥ 24k`.
4. `the_full_attention_cache_grows_to_24k_and_keeps_its_contents` — real shape
   (4 kv heads × 512 = 2048-wide stream): grow 512 → ≥23981 with marker values
   in keys/values/positions surviving every ladder rung, cursor preserved,
   reserve at the 24k position (the call that used to die with "position 9215
   exceeds capacity 8192").
5. `growth_past_the_policy_is_refused_at_the_real_shape` — `MAX_POLICY_CAPACITY+1`
   → `PositionOutOfRange{position: 32768, capacity: 32768}`.
6. `sliding_window_layers_stay_capped_at_1024_through_24k_positions` —
   sliding shape rings 24062 positions, capacity stays 1024, `key_count == 1024`.

**Sabotage checks (all three reverted after):**

| sabotage | result |
|---|---|
| `HOST_TOKEN_SLOTS = 8192` | 2 tests FAILED: `AC9 requires HOST_TOKEN_SLOTS >= 24k, got 8192`; buffer-width test (< 24k × 4 B) |
| `MAX_POLICY_CAPACITY = 8192` | 3 tests FAILED: `KV growth is capped at 8192, below the 24k AC9 requires`; `ensure_capacity(23963) failed: position 23962 exceeds capacity 8192` (the exact historical failure mode from `model.rs:3569-3573`); staged-buffer window mismatch |
| `ensure_capacity` keys prefix copy set to length 0 | 1 test FAILED: `keys prefix lost across the growth ladder  left: 0  right: 48879` |

**Suite:** `cargo test --release --workspace --tests --no-fail-fast` →
**107 test binaries ok, 4 failing tests — all four reproduced at pristine HEAD
(`43a4176`, my `model.rs` change stashed), none caused by T5:**

| failing test | file:line | why pre-existing |
|---|---|---|
| `no_device_side_sampler_exists` | `perf_probe_concurrency_dependency_graph.rs:1393` | asserts `elements.metal` has no `kernel void sample`; the device sampler shipped in `spec/device-sampler` |
| `the_device_token_handoff_exists_and_the_row_follows_the_pick` | `perf_probe_host_ops.rs:475` | counts lines with `host_tokens`+`new_buffer` on ONE line; the allocation is split over two lines (0 ≠ 1) at HEAD too |
| `the_colmajor_contig4_kernel_is_not_on_the_decode_path` | `perf_probe_dispatchsel_columns.rs:670` | source-text audit of model.rs vs kernel selector, stale |
| `activation_loads_are_4_byte_alignable_and_collapse_2_to_1` | `probe_cache_hierarchy.rs:849` | asserts exact line at `linear_repacked.metal:2121`, kernel has moved |

**Also pre-existing: `cargo test --release --workspace` does not even build** —
`crates/engine/examples/attn_bench.rs:374` calls `decode_splits(q, key_count)`
(2 args) but `crates/kernels/src/attention.rs:191` has 3 args since `2297f19`.
That is why the suite was run with `--tests`. I do not own either file —
**coordinator: fix or delegate** (pass `kv_heads` as the middle argument).
Not one of the 4 failures nor the example break encodes the old `HOST_TOKEN_SLOTS`
value; the deliberately-8192 tests named in the brief (`chat.rs:594-600`,
`model.rs:4066`, `model.rs:5677 PAYLOAD`) all **pass unchanged** — the first
and third never read the constant, the second uses it symbolically.

**Stale comments in files I do not own (cosmetic only, behaviour is symbolic
and correct):** `tests/device_sampler.rs:384` says "capped at `HOST_TOKEN_SLOTS`
(8192)"; `tests/perf_probe_host_ops.rs:49,477` say "8192 slots" in a const and
an assert message.

## 5. Measurements (24k = `'Explain the theory of relativity. ' × 3990`,
`KAPLUMBA_MAX_IN_FLIGHT=1`, `--max-new-tokens 8 --temperature 0 --metrics`,
`/usr/bin/time -l`, all `scripts/gpuq try`)

| run | arm | prompt_tokens | generated | prefill tok/s | decode tok/s | peak `phys_footprint` | engine swaps |
|---|---|---|---|---|---|---|---|
| T0 | before (max-new 1, no decode loop) | 23954 | 1 | **51.118** | — | not measured | — |
| A | **before** (`HOST=8192`) | 23954 | **8 ✓** | 50.887 | 0.383 | **18 390 512 264 B = 17.13 GiB** | 0 |
| B | after (first) | 23954 | **8 ✓** | 50.634 | 0.303 | **18 389 643 720 B = 17.13 GiB** | 0 |
| C | after (second) | 23954 | **8 ✓** | 50.970 | 0.426 | **18 389 414 608 B = 17.13 GiB** | 0 |
| D | after, 614-tok control | 614 | 8 ✓ | 94.612 | 14.385 | 13 589 801 736 B = 12.66 GiB | 0 |

- **AC9:** 24k prefill demonstrated repeatedly; decode after it produces tokens
  (8 > 1) both before and after the change — the premise that `HOST_TOKEN_SLOTS`
  blocked it is disproven, and the change was still made because AC9 names the
  constant and because of the OOB defect (§2).
- **AC6 / memory:** peak `phys_footprint` at 24k = **17.13 GiB < 20 GiB**
  (spread across A/B/C: 1.1 MB — the +96 KiB buffer does not move a 17 GiB
  number). `peak_resident_pages_bytes` was 7.2-8.1 GiB on the same runs,
  confirming the ~5.5 GiB understatement in AGENTS §10.3.
- **Prefill tok/s unchanged:** before {51.118, 50.887}, after {50.634, 50.970}
  — medians 51.00 vs 50.80, Δ = −0.4 %, inside the within-arm spread (±0.3).
  Expected: prefill never indexes `host_tokens`; the only shared effect is one
  96 KiB allocation at load. 614-tok control 94.612 matches T0's 94.66.
- **Swap — the "0 MiB" pass condition does not exist on this box, and
  `scripts/gpuq` never reports swap:**
  - `gpuq`'s `swap_used_mb()` (scripts/gpuq:98-101) parses `vm.swapusage` with
    `awk -F'[= ]+' … printf "%.0f", $2/1048576` — `$2` is the **total** field
    (already in MiB with an `M` suffix), so it divides ~4096 by 1048576 and
    **always prints "0 MiB swap"**. Baseline at rest was 1.6-2.8 GiB *used*
    before I ever ran (parallel agents). **gpuq is not my file: flagged for
    coordinator.**
  - System-wide `vm.swapusage` across runs: A 1630→2078 MiB, B 1662→2986,
    C 1987→2828 (macOS also grew the swap file 3072→4096 MiB), D (614 control)
    2820→**2796, i.e. no growth** at a 12.66 GiB footprint. So growth tracks
    the 17.13 GiB residency: the OS pages out *other* processes under pressure.
  - **The engine process itself did `0 swaps` in every run** (`/usr/bin/time -l`
    getrusage line), and `top` right after showed 16-18 GiB unused / compressor
    257-349 MiB — the box recovers. Honest verdict: pass on the only thing the
    engine controls (0 swap I/O), record the system figure rather than claim it.

## 6. Observations for the coordinator (found, not fixed — out of T5 scope)

1. **Decode at 24k context is 0.30-0.43 tok/s (2.3-3.3 s/token)** — measured
   in runs A/B/C, vs 14.4 tok/s at 614 (run D). T0 never measured decode at 24k
   (`max-new-tokens 1` skips the loop). Nothing in this constant explains it;
   likely KV-read bandwidth / split geometry at 24k keys. Far from any AC, but
   it means "24k context" is currently prefill-only in practice.
2. `scripts/gpuq` swap readout broken (§5) — same class of unit bug as
   AGENTS §10.3 warns about.
3. Four stale probe tests + one broken example at base commit (§4) —
   `cargo test --workspace` (as the brief words it) cannot pass for anyone
   until they are fixed; `--tests --no-fail-fast` is the runnable form.
4. `tests/device_sampler.rs:384` / `perf_probe_host_ops.rs:49,477` comments
   hardcode "8192" where the code correctly uses the symbol.

## 7. Secondary task — "cut the prefill host gap": PREMISE REFUTED, no change made

T0 recorded `inter-boundary gap = 3686 ms of a 108 984 ms 8k prefill (3.4%)`
and the brief estimated cutting it "would be worth ~3% of end-to-end prefill
tok/s". Profiling first, as instructed, shows **the gap is the profiler's own
cost — it is not paid by product runs**, so there is nothing to cut in
`model.rs`.

**Mechanism (code):** `GpuProfile::stamp` (`model.rs:829-840`) is a no-op
unless `KAPLUMBA_GPU_PROFILE` is set; when set, it **commits the command
buffer at every one of ~7213 boundaries** (8 chunks × 60 layers × ~15) and
drains with waits every 15 (`DRAIN_EVERY`, `model.rs:621-637` — the comment
itself says "the *gaps* between buffers are an artefact of the profiler").
Unprofiled, the same forward rolls command buffers only at
`max_dispatches_per_buffer` (~2 commits at 8k).

**Measured, interleaved A/B, 8k prompt, 3 pairs, same binary, same env
(`KAPLUMBA_MAX_IN_FLIGHT=1`):**

| pair | OFF tok/s (env unset) | OFF ttft ms | ON tok/s | ON ttft ms | ON own_ms | ON gap ms | ON span ms |
|---|---|---|---|---|---|---|---|
| 1 | 73.653 | 106 099 | 73.604 | 106 169 | 103 410 | 1 558 | 106 158 |
| 2 | 73.431 | 106 419 | 68.619 | 113 883 | 104 252 | **7 346** | 113 872 |
| 3 | 72.789 | 107 359 | 71.224 | 109 717 | 103 786 | 3 841 | 109 706 |

- Medians: OFF **73.431** tok/s (range 72.79–73.65), ON **71.224**
  (68.62–73.60): profiling costs ~3 % — i.e. the "gap %" shows up exactly as
  the profiler ON-vs-OFF delta, not as time the product path spends idle.
- The load-bearing rows are the *spread of the gap itself*: pair 1's gap is
  1.6 s and its ON wall (106 158) equals the OFF arm (106 099); pair 2's gap
  is 7.3 s and its ON wall exceeds the OFF arm by 7.5 s. `own_ms` is stable at
  103.4–104.3 s (±0.4 %) across all three. **Wall follows the gap, kernel time
  does not** — the gap is added on top of what an unprofiled run costs.
- Residual product-side non-kernel time (unprofiled ttft − profiled own_ms,
  mixed runs) ≈ 2–3 s ≈ 2–2.8 %, an *upper bound* confounded by run variance;
  it is not the 3686 ms figure and this box's spread (±1.7 % on OFF alone)
  cannot resolve it further. Record, do not chase.

**First attempt discarded (method error, recorded for honesty):** I ran the
first A/B with `KAPLUMBA_GPU_PROFILE=0` for the OFF arm. `env_flag`
(`model.rs:666-679`) is **presence-based** (`var_os().is_some()`), and
`model.rs:4315` uses it, so `=0` still *enables* the profiler — both arms were
profiled (medians 70.5 vs 70.9, meaningless). The valid run uses
`env -u KAPLUMBA_GPU_PROFILE`. Footgun for anyone measuring: **you cannot turn
the profiler off with `=0`; the variable must be absent.**

**Verdict:** no optimization made, per the brief's own rule ("optimize only
if you can measure an improvement interleaved against baseline"). The brief's
"worth ~3 %" mistakes measurement overhead for product overhead. If gap
fidelity matters for future attribution, the lever is profiler-side
(`DRAIN_EVERY`, asynchronous timestamp reads) and its benefit is *measurement
accuracy*, zero product tok/s — flagged, not T5's or this spec's job.

## 8. Commands

```bash
export CARGO_TARGET_DIR=/private/var/folders/tp/wz3877_s1z756p1kcsz3_d4h0000gn/T/opencode/p120-ctx-target
cargo build --release -p kaplumba-cli                     # confirm "Compiling kaplumba-engine"
cargo test --release -p kaplumba-engine --test context_limit_ctx24k
cargo test --release --workspace --tests --no-fail-fast   # 107 ok / 4 pre-existing fails

M=~/.kaplumba/models/Gxma/Gxma
PP24=$(python3 -c "print('Explain the theory of relativity. '*3990)")
KAPLUMBA_MAX_IN_FLIGHT=1 scripts/gpuq try -- /usr/bin/time -l \
  /absolute/path/to/kaplumba --model "$M" --prompt "$PP24" \
  --max-new-tokens 8 --temperature 0 --metrics
sysctl -n vm.swapusage   # never trust gpuq's swap line
```
