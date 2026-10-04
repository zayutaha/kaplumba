# Agent brief — spec/prefill-120tps

Read this, then `spec.md`, `tasks.md`, `implementation_notes.md` (the T0 section
is the important one), and `plan.md`. Do not re-derive T0.

## The load-bearing arithmetic (T0, measured, branch `mtp-speculation` @ 64f1753)

Prefill GPU attribution, `KAPLUMBA_GPU_PROFILE=1`, `KAPLUMBA_MAX_IN_FLIGHT=1`:

| context | prefill tok/s | GEMM share | attention share | other |
|---|---|---|---|---|
| 614 tok | **94.66** | **93.2%** (6046 ms) | 5.5% (356 ms) | 1.3% |
| 8k tok | **74.52** | 72.5% (75 061 ms) | **26.4%** (27 353 ms) | 1.0% |
| 16k tok | 60.75 | — | — | — |
| 24k tok | 51.12 | — | — | — |

Consequences that override `plan.md`:

1. `plan.md`'s "~4.6 ms/token of norms+RoPE+dispatch = 36%" is **false**; it
   measures 1.3%. There is no elementwise/dispatch slack to harvest.
2. **AC1 (120 tok/s @614) and AC4 (GEMM ≥7.5 TFLOPS) are the same task.**
   Budget at 120 tok/s is 5117 ms vs 6545 ms now. Making attention *entirely
   free* only reaches 99.2 tok/s. GEMM must go 6046 → ~4707 ms (5.84 → 7.5
   TFLOPS) *and* attention must improve. Neither alone is sufficient.
3. GEMM is 93% of short-context prefill, so **GEMM is the largest lever for
   AC1 and AC2 both**, even though attention dominates at long context.
4. `HOST_TOKEN_SLOTS=8192` does **not** block 24k prefill — 24k already runs
   end to end. T5's premise is partly wrong; verify before changing anything.

## Non-negotiable (AGENTS.md §8)

No quality loss. GEMM worst normalized error ≤ **2.16e-3** vs CPU reference.
No weight memory increase (≤ 11.51 GiB). No new large scratch; prefill
attention stays online-softmax flash-style, **never** an O(L²) score matrix.
Scratch fixed-size, independent of context length. No Python in product code.
No KV cache quantization. Single-user only.

## Parallelism rules (AGENTS.md §6, updated)

Multiple agents run at once. The rules:

- **One model load at a time.** Never run the binary except through
  `scripts/gpuq try -- <cmd>`. Exit **75 = GPU busy** → go do non-GPU work and
  retry later. Never `gpuq run` (blocking), never bypass the queue.
- **Edit only the files you own** (stated in your task prompt). If you believe
  you must edit a file owned by another agent, write the reason in your notes
  and stop — the coordinator will sequence it.
- **Own worktree, own `CARGO_TARGET_DIR`.** Do not build in the main tree.
- Unit tests / `cargo test` that do **not** load the model are unrestricted.

## Measurement rules (violating these produces fake numbers)

- Build: `cargo build --release -p kaplumba-cli` in YOUR worktree, with
  `CARGO_TARGET_DIR` set. Shaders are `include_str!`'d — **confirm
  `Compiling kaplumba-kernels` appears**; a green run on a stale binary is a
  false pass.
- Interleave A/B (A B A B), ≥3 pairs, report **median with spread**. A single
  reading is not a result; same-config spread reaches 35%.
- Always record `inter-boundary gap` beside tok/s — it swings 38 → 261 ms and
  is the usual cause of a phantom 7% regression.
- `scripts/gpuq` **changes directory** — pass absolute paths to your binary.
- Baseline numbers are in `implementation_notes.md` §T0. Re-measure your own
  arm rather than quoting them.
- Reference prompt: `Explain the theory of relativity. ` × N
  (×100 ≈ 614 tok, ×1300 ≈ 7814, ×2660 ≈ 15987, ×3990 ≈ 23980).
  Model: `~/.kaplumba/models/Gxma/Gxma`.

## Definition of done (AGENTS.md §4.3)

Every test you can think of, written and run — happy path, edge cases,
boundaries, failure paths. **Sabotage-check each new test**: break the thing it
protects, confirm it fails with a diagnosable message, revert. Record the
failure mode you observed. Existing suite still green. Then update your notes
file and commit with `spec/prefill-120tps: <task id> …`.
