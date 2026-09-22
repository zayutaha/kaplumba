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

## 6. Single-model constraint

- **Only one LLM (agent session) may be “in memory” at a time** for a given task: do not run parallel agents on the same spec or the same working tree.
- If delegating, wait for the previous agent to finish and write its handover notes before starting the next.
- No concurrent edits to the same files; serialize work.

---

## 7. Reference material

Read-only reference checkouts live under `reference/` (gitignored):

| Path | Project |
|------|---------|
| `reference/omlx` | Base inference server (MLX) — primary codebase lineage |
| `reference/llama.cpp` | Mainline llama.cpp |
| `reference/ik_llama.cpp` | ik_llama.cpp fork (quantization, prefill speed) |
| `reference/vllm` | vLLM |
| `reference/PonyExl3` | EXL3 on Apple Silicon (converter + Metal runtime) |

Do not modify `reference/` unless a task explicitly says to vendor a patch; prefer describing ports in specs.

Model weights and quantized checkpoints live in `~/.kaplumba/models`. Do not download models into the repository working tree.

---

## 8. Project constraints (standing)

These are global restrictions; any spec that conflicts with them must call the conflict out explicitly and get user approval before implementation:

- Target: **single-user chat only** — no multi-tenant scheduling, no request batching for its own sake.
- Hardware: Apple **M4 Pro, 20 GPU cores, 273 GB/s**.
- Model: **Gemma 4 31B-class dense**, **EXL3 ~3.0 bpw → ~15.5 GB weights + ~1 GB SWA/KV ≈ 16.5 GB resident**.
- **No KV cache quantization** (unless a future spec explicitly lifts this).
- **No quality loss** — speculative decoding must be verify-gated (greedy-identical or distribution-exact).
- **No overhead margin** — avoid scheduler/sync/serialization costs on the hot path; profile before adding machinery.
- Optimize for: **TTFT, tg/s, pp/s, memory**, in that awareness; always state which metric a change targets.

---

## 9. Quick checklist (every non-trivial task)

1. [ ] Interview user with many bracketed questions; get confirmation.
2. [ ] Create `spec/<name>/spec.md` (goal, restrictions, requirements, acceptance).
3. [ ] Create `spec/<name>/plan.md` (technical design).
4. [ ] Create `spec/<name>/tasks.md` (small ordered tasks).
5. [ ] Implement task → test all conceivable cases → update `tasks.md` → update `implementation_notes.md` → commit.
6. [ ] Repeat step 5 until done; prove acceptance criteria; final notes.
