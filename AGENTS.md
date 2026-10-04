# AGENTS.md

Operating instructions for AI agents working in this repository.

---

## 1. Scope

These rules apply to every non-trivial task. A **non-trivial task** is any change that:

- spans more than one file,
- introduces or modifies a feature, subsystem, or public interface,
- changes performance-critical behavior,
- touches caches, persistence, or on-disk formats, or
- is ambiguous enough that a reasonable implementation could diverge from intent.

Trivial tasks (typo fixes, dependency bumps already covered elsewhere, re-running an existing test) do not require a spec; they still require small, tested, separately committed steps.

---

## 2. Clarification before specification

Before drafting any specification, the agent **must** interview the user until no meaningful ambiguity remains.

### 2.1 Question requirements

- Ask **every** question the agent can think of. Do not stop at the first answer. Enumerate edge cases, failure modes, boundaries, defaults, and interactions with existing behavior.
- Prefer many small questions over few large ones.
- After each question, append a **plain-language gloss in brackets** so the user immediately understands what is being asked, e.g.:

  > What is the maximum chat length we persist to disk before evicting the oldest chat?
  > (If `./chat` grows forever, when do we delete old ones — never, after N chats, or after N GB?)

### 2.2 Exit criterion

Clarification is complete only when the agent can restate the requirement, its constraints, and its edge cases without needing a guess. Any remaining uncertainty must be written down as an open question in the spec, not silently resolved.

### 2.3 Confirmation

Summarize the agreed answers for the user and get explicit confirmation **before** opening a spec folder.

---

## 3. Specification workflow

Every non-trivial task gets its own folder under `spec/`:

```text
spec/
  <spec-name>/
    spec.md                 # what & why: scope, requirements, restrictions, acceptance criteria
    plan.md                 # how: technical design, data flow, interfaces, risks
    tasks.md                # ordered checklist of concrete code changes
    implementation_notes.md # running log for implementers (see §5)
```

### 3.1 `spec.md` — specification and restrictions

Must contain:

- **Goal** — one paragraph, outcome-focused.
- **Non-goals / restrictions** — explicit list of what must *not* happen (quality loss, KV quantization, multi-user behavior, forbidden shortcuts, etc.).
- **Requirements** — numbered, testable.
- **Acceptance criteria** — how “done” is proven.
- **Open questions** — anything still unresolved (should be empty after §2).

### 3.2 `plan.md` — technical plan

Must contain:

- Architecture and data flow.
- File/module touchpoints (paths, current behavior, intended change).
- Interfaces, formats, and on-disk layout if applicable.
- Sequencing and dependencies between tasks.
- Risks and fallbacks.
- Explicit mapping: each requirement → the task(s) that satisfy it.

### 3.3 `tasks.md` — task checklist

- One bullet or checkbox per **small, independently verifiable** code change.
- Each task names: files touched, intent, and how it will be tested.
- **Update `tasks.md` immediately after every completed task** — mark it done, note the commit hash, and adjust remaining tasks if reality diverged from the plan. Do **not** wait until the whole spec is finished.
- If scope changes, amend `spec.md` / `plan.md` first, then `tasks.md`.

### 3.4 Order of writing

1. Clarification interview (§2) → user confirmation.
2. `spec.md`.
3. `plan.md`.
4. `tasks.md`.
5. Implementation begins only after all three exist.

**A test-coverage claim is not a claim until you have broken the code and seen
it fail.** Before reporting "N/N tests cover X" (or writing it into
`spec.md`, `tasks.md` or `implementation_notes.md`), sabotage X, run the test,
and confirm it fails with a diagnosable message. Then revert the sabotage.
Record the failure mode you observed.

This is not hypothetical. This project shipped a kernel that read activations
instead of weights past `k >= 32`, under a "6/6 against a CPU reference" claim —
because the test called an entry point that hardcoded a *different* kernel. The
same test later set an env flag that **disabled the path under test**, and a
deliberately sabotaged kernel still passed it
(`crates/engine/tests/linear_metal.rs:900`). Two false coverage claims, one root
cause: nobody checked that the failing branch could fail. See
`spec/engine-performance` Q11.

---

## 4. Implementation discipline

### 4.1 Small steps

- Implement **one task at a time**.
- Never batch unrelated changes into a single step.

### 4.2 Commit after every step

- After each task (and its tests) pass, **commit immediately**.
- Commit message: reference `spec/<name>` and the task id, e.g. `spec/prompt-cache: T3 block-aligned LCP restore`.
- The history must allow reverting any single step without undoing unrelated work.
- Do not commit broken code, half-updated `tasks.md`, or secrets.

### 4.3 Definition of done (per task)

A task is complete only when:

1. The change is implemented as specified (or the spec was amended first).
2. **Every test case the agent can think of has been written and run** — happy path, edge cases, failure/rollback paths, and boundary conditions.
3. Existing tests still pass (no regressions).
4. `tasks.md` is updated with status + commit hash.
5. `implementation_notes.md` is updated if anything noteworthy was learned (§5).
6. The step is committed.

“Works on my machine” or “the obvious case passes” is not done.

### 4.4 Full-spec completion

The spec as a whole is complete only when every task in `tasks.md` is done, acceptance criteria in `spec.md` are demonstrated (including end-to-end tests), and open questions are closed or explicitly waived by the user.

---

## 5. `implementation_notes.md` — handover log

Append to this file **as work proceeds**, not at the end. It exists so another model (or the same model in a fresh session) can take over with full context. Include:

- Decisions taken during implementation that the plan did not predict.
- Divergences from `spec.md` / `plan.md` and why.
- Environment facts, measurement results, benchmark numbers, gotchas.
- Dead ends and abandoned approaches (with reasons).
- Invariants discovered in existing code that constrain future work.
- Exact commands for build, test, and benchmark.
- Anything that would cost a newcomer hours to rediscover.

Tone: factual and dense. Prefer specifics (paths, numbers, commit hashes) over narrative.

---

## 6. Parallel agents (replaces the old single-model rule)

**Multiple agent sessions may run in parallel.** The old rule — "only one LLM
in memory at a time" — is lifted. The constraint that survives is the **GPU
one**, not the agent one:

- **Only ONE agent may load the model at a time.** Two concurrent model
  instances exceed physical memory and invalidate every benchmark (AGENTS.md
  §10). Every agent that runs the binary must go through `scripts/gpuq try --`
  and treat exit **75** as "GPU busy, do the non-GPU work, retry later".
  Compute-bound work (writing kernels, reading code, editing, unit tests that
  do not load the model) is **not** restricted — run as many of those as
  useful.
- **No concurrent edits to the same files.** Split work by file ownership:
  give each agent a disjoint set of files (e.g. one owns `attention.metal`,
  another owns `linear.metal`, another owns `model.rs`). If two agents must
  touch the same file, serialize them.
- **Each agent works in its own worktree with its own `CARGO_TARGET_DIR`**
  (§10.2 #4). A shared `target/` lets one agent's rebuild land in the middle
  of another's benchmark.
- Each agent writes its own handover notes to `spec/<name>/implementation_notes.md`
  (or a per-agent file if two are in the same spec) **before finishing**, and
  the coordinator merges the task-status updates into `tasks.md` — otherwise
  concurrent writers clobber each other's checklist.
- A **model-loading** agent must still wait for the previous one; only the
  loading is serialized, not the thinking.

---

## 7. Reference material

**Read the references before writing a kernel.** These projects have already solved
the problems we are still paying for on this exact hardware. Do not start from
first principles; start from what is known to work and port it.

Read-only reference checkouts. **Only `reference/omlx` is actually present; the
rest live in scratch directories that macOS clears.** Verify a path exists
before citing it, and re-clone if it is gone.

| Path | Project | Licence | State |
|------|---------|---------|-------|
| `reference/omlx` (gitignored) | Base inference server (MLX) — primary codebase lineage. `omlx_sdpa_decode` / `omlx_sdpa_prefill`; the verified simdgroup fragment layout in `omlx/custom_kernels/bonsai/csrc/quantized.h`; 32×32 tile and no-barrier-in-K-loop shape in `omlx/patches/qwen35_verify_qmm.py`. | Apache-2.0 | **present** — the only durable one |
| `/tmp/kaplumba-refs/llama.cpp` | Mainline llama.cpp. Source of the prefill-attention structural transpose we ported (`fa_common.metal`). | MIT | ephemeral, `/tmp` is cleared on reboot |
| `/tmp/kaplumba-refs/ik_llama.cpp`, `/tmp/kaplumba-refs/vllm` | ik_llama.cpp fork (quantization, prefill speed); vLLM. | MIT / Apache-2.0 | **not cloned** — secondary, clone only for a specific question |
| `/var/folders/tp/wz3877_s1z756p1kcsz3_d4h0000gn/T/opencode/ishizuki` | ishizuki — reports ~11 tok/s decode on M4 Pro, ~20 on M1 Max. **AGPL-3.0: read for ideas, confirm with the user (see `spec/engine-performance` Q9) before porting code.** <https://github.com/struffl/ishizuki> | AGPL-3.0 | ephemeral |
| `/var/folders/tp/wz3877_s1z756p1kcsz3_d4h0000gn/T/opencode/exllamav3` | Source of the EXL3 format and its `cp.async`/`int4` staging (`exl3_gemm_inner.cuh`). | MIT | ephemeral |

When adopting an idea from any of these, cite `path:line` in
`implementation_notes.md`, and A/B measure it here — a reference proves a technique
*can* work, not that it works on our shapes. **Record the negative results too.**
Six of the nine reference techniques mined in `spec/engine-performance` did not
pay off here (concat fusion −2.3%, trellis staging 15% slower, `extract_bits`
inapplicable, `AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1` null, and two others); that
list is only useful because it is written down with the numbers.

Do not modify `reference/` unless a task explicitly says to vendor a patch; prefer describing ports in specs.

Model weights and quantized checkpoints live in `~/.kaplumba/models`. Do not download models into the repository working tree.

---

## 8. Project constraints (standing)

These are global restrictions; any spec that conflicts with them must call the conflict out explicitly and get user approval before implementation:

- **No Python in the project** (user directive 2026-09-22) — no Python in product/engine code; engine is pure Rust + raw Metal (see `spec/rust-metal-engine/`). Known conflicts: (a) `scripts/bench.py` — temporary T0 harness, retire at T0.2/R7.4 (DE20); (b) ~~omlx engine plan~~ resolved — `spec/omlx-exl3-engine` superseded by `spec/rust-metal-engine`.
- Target: **single-user chat only** — no multi-tenant scheduling, no request batching for its own sake.
- Hardware: Apple **M4 Pro, 20 GPU cores, 273 GB/s**.
- Model: **Gemma 4 31B-class dense**, **EXL3 ~3.0 bpw → ~15.5 GB weights + ~1 GB SWA/KV ≈ 16.5 GB resident**.
- **No KV cache quantization** (unless a future spec explicitly lifts this).
- **No quality loss** — speculative decoding must be verify-gated (greedy-identical or distribution-exact).
- **No overhead margin** — avoid scheduler/sync/serialization costs on the hot path; profile before adding machinery.
- **No graph/tree-shaped prefill scratch** (user directive 2026-09-22). The MLX reference (`reference/omlx`) lets prefill peak memory balloon far past the KV window: (a) its lazy op graph keeps every intermediate and model-built additive mask alive at once, (b) MLX's Metal **buffer pool** keeps freed transients resident (only `mx.clear_cache()` returns them), and (c) unsupported SDPA shapes materialize an **O(L²) `[heads, L, L]` score matrix in fp32** even for fp16/bf16 models (documented `PrefillMemoryExceededError` path; `omlx/memory_monitor.py`, `omlx/cluster/memory_guard.py`). Our engine must not: prefill attention is **online-softmax (flash-style), O(L) working set**, never a materialized score matrix; scratch buffers are **explicitly owned, fixed-size, and independent of context length**; no retained-pool semantics; peak RSS is a **measured acceptance number** across 1k/2k/8k prefill, not a prediction.
- Optimize for: **TTFT, tg/s, pp/s, memory**, in that awareness; always state which metric a change targets.

---

## 9. Quick checklist (every non-trivial task)

1. [ ] Interview user with many bracketed questions; get confirmation.
2. [ ] Create `spec/<name>/spec.md` (goal, restrictions, requirements, acceptance).
3. [ ] Create `spec/<name>/plan.md` (technical design).
4. [ ] Create `spec/<name>/tasks.md` (small ordered tasks).
5. [ ] Implement task → test all conceivable cases → update `tasks.md` → update `implementation_notes.md` → commit.
6. [ ] Repeat step 5 until done; prove acceptance criteria; final notes.

---

## 10. GPU access discipline (mandatory)

**This box has 24 GiB of unified memory and the model needs 11.7-13.0 GiB by measurement. Two
concurrent model instances exceed physical memory.** That has already been
measured in this repo: file-backed pages get evicted, which cost 3.9 GiB of swap
and a collapse to ~1.4 tok/s (`crates/engine/src/weights.rs:1217`). It also
invalidates every benchmark, because two instances fight for bandwidth and page
residency.

### 10.1 Use `scripts/gpuq`, and do not block by default

| command | behaviour | when to use it |
|---|---|---|
| `scripts/gpuq status` | instant; prints holder, queue depth, available memory, and warns below the ~14 GiB a load needs | before anything that will load the model |
| `scripts/gpuq try -- <cmd>` | runs the command; **the engine** decides. Exits **75** immediately if the GPU is busy | **the default.** Do analysis, then retry |
| `scripts/gpuq run -- <cmd>` | same, but the engine blocks until free (use only when there is nothing else to do) | deliberate queueing |

**`gpuq` takes no lock, and never may.** The engine is the sole lock owner
(`crates/engine/src/gpu_lock.rs`, called from `load_from_dir`). An earlier
`gpuq` took the same lock *and* ran the engine, which deadlocked: a lock held by
a parent is not inherited as held by the child, so the child's `flock(LOCK_EX)`
conflicts with the parent's and blocks — parent waits for child, child waits for
parent. It presented as "gpuq try -- kaplumba hangs forever while gpuq try --
echo works", because `echo` does not lock and `kaplumba` does. If you are
tempted to add a lock here, do not.

**Never sit in a 30-minute blocking call.** A `run` that blocks for twenty
minutes has wasted twenty minutes: it cannot report, and it cannot do the work
that would have been useful meanwhile. Prefer `try` in a loop around
*productive* work, not around `sleep`.

### 10.2 The rules

1. **Check `gpuq status` before every model run.** Only act on its warning if it
   reports under 14 GiB *available*; it reads `top`, not `vm_stat` (see §10.3).
2. **One model process at a time, always.** The engine enforces this with a
   kernel `flock`, so it holds even when `gpuq` is bypassed — a stray direct run
   from a worktree is exactly the case that once held 11 GiB invisibly. A second
   process gets exit code 75 (`EX_TEMPFAIL`), which is a scheduling outcome, not
   a failure.
3. **Do not run a second model instance "just to check".** This has happened and
   it destroyed the numbers.
4. **In a worktree, use a separate `CARGO_TARGET_DIR`.** A worktree sharing the
   main `target/` can rebuild the binary mid-benchmark of another agent.
5. **Interleave A/B measurements** (A B A B …) and report medians with spread.
   Decode on this box has been measured at 3.7, 3.9, 4.3, 4.8, 6.0 and 6.4
   tok/s from the *same binary* minutes apart, so a single reading is not a
   result and a difference smaller than the spread is noise.
6. **Clean up worktrees and temporary harnesses.** `git worktree remove
   <path> --force` and delete any `crates/engine/examples/*.rs` you added; a
   stray benchmark binary left running holds the GPU and is invisible to the
   next agent.
7. **A missing output is not a hang.** Before concluding the model did not load,
   re-run with `2>&1 | head -5` and check the token count. The test sentence is
   **6.01 tokens**, so a prompt over 8192 tokens is rejected *before* loading with
   "prompt plus generation exceeds context limit 8192" — which looks exactly
   like a silent failure if you are only reading the tail of a pipe.

### 10.3 Measuring this box

**Do not use `vm_stat`'s `Pages free` as a memory-pressure signal**, and do not
parse `top`'s figure without handling units. Both mistakes were made here and
both shipped before being caught:

- `Pages free` counts only pages that have never been touched, so it ignores the
  file cache. With no model loaded it said 1.47 GiB while `top` said 14 GiB
  unused, and a model loaded in 10.8 s. An earlier `scripts/gpuq` used that
  counter, warned on a healthy box, and thereby stopped agents from running.
- `top` switches between K/M/G/T and drops the decimal below 10, so `554M
  unused` is 0.55 GiB. Stripping non-digits turned that into **554 GiB** — a
  1100x overstatement — and taking the last comma-separated field yields
  `unused.` rather than a number. Both were in `gpuq` before the unit test
  caught them.

Use `scripts/gpuq status`, which parses both correctly, or:

```bash
top -l 1 -n 0 | grep PhysMem     # "unused" is the number that matters
sysctl -n vm.swapusage
```

There are 24 GiB of RAM and no discrete GPU. The earlier 32 GiB figure recorded
in some notes is wrong.

**Do not read `--metrics peak_resident_pages_bytes` as a memory footprint.** It
is `getrusage().ru_maxrss`, which counts resident pages only and misses the
Metal buffer pages entirely. Measured on the same run at 614 tokens, by
`/usr/bin/time -l`:

```text
peak_resident_pages_bytes    6 691 717 120  =  6.23 GiB
peak memory footprint       12 555 691 320  = 11.69 GiB
```

and at 8192 tokens the footprint reaches ~12.96 GiB. The field was renamed from
`peak_rss_bytes` in `ee14f62` precisely so it cannot be misread again; it
understated real usage by 5.5 GiB and made a 93%-of-budget situation look like
57%. The budget figure is `phys_footprint`, which only the OS will report, via
`/usr/bin/time -l` (see `crates/engine/tests/memory_proof.rs`). A hand-rolled
`task_info` FFI has been attempted twice and rejected twice: `libc` exports
neither `TASK_VM_INFO` nor `task_vm_info_data_t`, and the struct has grown
across macOS releases so the `phys_footprint` offset is unstable and returns
garbage. A guessed offset yields a plausible wrong number, which is the worst
possible failure for the metric the memory budget is judged on.

Prefill is the throughput metric that matters most and it is measured
end-to-end:

```bash
M=~/.kaplumba/models/Gxma/Gxma
PP=$(python3 -c "print('Explain the theory of relativity. '*100)")
scripts/gpuq run -- ./target/release/kaplumba --model "$M" --prompt "$PP" \
    --max-new-tokens 1 --temperature 0 --metrics
```

Per-kernel GPU attribution (the working profiler; `KAPLUMBA_PROFILE_PHASES` is
dead, see the implementation notes):

```bash
KAPLUMBA_GPU_PROFILE=1 ./target/release/kaplumba --model "$M" --prompt "$PP" \
    --max-new-tokens 1 --temperature 0 --metrics
```

