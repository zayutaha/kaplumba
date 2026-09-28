# Fix the GPU queue: it currently deadlocks the engine

## Goal

`scripts/gpuq` and the in-binary `flock` in `crates/engine/src/gpu_lock.rs` were
added to stop two model instances running at once on a 24 GiB box. They work
individually and **they deadlock each other**: `gpuq try -- kaplumba …` hangs
forever and never loads the model. Make the mutual exclusion work, and make
waiting observable rather than a silent hang.

Success means: `gpuq try -- kaplumba …` runs a benchmark, two agents cannot run
the model simultaneously, and a waiting agent is never stuck in a call that
looks like a hang.

---

## The bug, confirmed

`gpuq` acquires an exclusive `flock` on `~/.kaplumba/gpu.lock` and then runs the
command as a child. The engine acquires **the same file** in
`gpu_lock.rs:acquire`, called from `TextModel::load_from_dir` (`model.rs:2115`).

A lock held by a parent is *not* inherited as held by the child. The child's
`open()` produces a different open file description, so `flock(LOCK_EX)`
conflicts with the parent's and blocks. The parent waits for the child; the child
waits for the parent. Neither moves.

Reproduced minimally, independent of this repo:

    #!/bin/bash
    exec flock "$1" -c "flock '$1' -c 'echo CHILD-ACQUIRED'"
    # -> hangs forever

And in this repo, the asymmetry that made it hard to see:

    gpuq try -- /bin/echo hello      # works, returns 0
    gpuq try -- kaplumba --prompt Hi # hangs forever

`echo` does not lock. `kaplumba` does. That is the entire difference.

**Cost of this bug, measured in this session:** the 8k benchmark appeared three
times to "not load the model", and I formed three wrong conclusions from it (a
silent hang, a stuck GPU queue, a broken model load) before finding the real
cause — which turned out to be a different thing again, an over-long prompt
(8656 tokens > the 8192 limit) printing to a pipe whose head was not read. Two
bugs, and the first one masked the second.

---

## Two candidate designs

The deadlock has one cause and two honest ways out. They differ in where the
lock lives, which changes what is protected.

### Option A — one lock owner: the engine only (recommended)

`gpuq` stops taking the lock entirely. It becomes a **thin, non-blocking
convenience wrapper** that only reports state, never serialises:

    gpuq status    # instant, never takes the lock
    gpuq try       # just runs the command; exit code comes from the engine
    gpuq run       # runs the command; if the engine exits 75 (busy), retry per policy

`gpu_lock.rs` stays the single source of truth, because it cannot be bypassed.
To make the engine's busy state machine-readable it must return **exit code 75**
(`EX_TEMPFAIL`) rather than blocking silently.

- **Pro:** one owner, so no deadlock is possible. The in-binary lock is the one
  that actually cannot be bypassed, and it already exists.
- **Pro:** an agent that forgets the wrapper is still protected.
- **Con:** `gpuq` cannot report "you are #2 in the queue" — it cannot see the
  queue at all, because it is not the queue. It can only relay what the engine
  printed.
- **Con:** to get a *tickets* queue rather than a *mutex*, the engine needs a
  ticket file, which is more Rust in a code path that is otherwise a hard
  requirement to be small and dependency-free.

### Option B — one lock owner: the wrapper only

Delete `gpu_lock.rs` and the call in `model.rs`. `gpuq` is the only serialiser.

- **Pro:** the queue is entirely in one shell script, where tickets, priorities,
  heartbeats and fairness are easy.
- **Pro:** `gpuq` genuinely knows the queue depth and can tell an agent its
  position.
- **Con:** **any agent that runs the binary directly gets no protection at all**,
  and that has already happened in this session — a stray run from a worktree
  held 11 GiB invisibly for 20+ minutes. This is the failure the in-binary lock
  was added to fix. Reverting it re-opens that hole.

### Recommendation

**Option A.** The property that matters is "two instances can never run at
once", and an in-process lock is the only thing that guarantees it. A shell
wrapper is a convention; a lock in `load_from_dir` is not.

But I want your decision on the one point where A is weaker — see Q1.

---

## Proposed shape, if Option A

Three layers, each with one job, and exactly one lock owner between them.

**Layer 1 — `gpu_lock.rs` (the only lock owner).** Keep `flock`. Change the
*policy*, not the mechanism:

- `KAPLUMBA_GPU_LOCK=wait` (default): current behaviour, block until free.
- `KAPLUMBA_GPU_LOCK=try`: `LOCK_EX | LOCK_NB`; on failure print one line naming
  the holder and the queue position, then **exit 75** so the caller decides.
- Keep a ticket file alongside the lock for queue position, and prune entries
  whose pid is dead, so a crashed agent cannot wedge the queue.
- The `File` must stay alive for the process lifetime, else the lock drops
  mid-run. That is already handled by the `OnceLock`.

**Layer 2 — `scripts/gpuq` (never locks).** A reporter and a policy wrapper:

    gpuq status              # holder, queue depth, available memory; never locks
    gpuq try -- cmd          # exec the command; propagate its exit code
    gpuq run -- cmd          # exec the command; if it exits 75, print guidance

It must be obvious in the source that this script takes no lock, with the
deadlock and its three-line repro in the comment, because "the wrapper and the
binary both lock" is a mistake two people will make.

**Layer 3 — the agent instruction.** In `spec/agents/README.md` and
`AGENTS.md §10`: the default is `gpuq try`, exit 75 means *do other work and
retry*, never sleep-and-retry, and never treat a lack of output as a hang —
check `/tmp` or re-run with `2>&1 | head -5` before concluding anything.

---

## Measured today, for reference

- One model process is ~11.5 GiB resident; the box is 24 GiB. Two do not fit;
  that has been measured at 3.9 GiB of swap and a collapse to 1.4 tok/s.
- A load takes ~2.4 s; a 614-token prefill ~8.6 s; an 8k prefill ~112 s. So a
  `run` blocked behind an 8k benchmark is a 2-minute block, and an agent that
  sits in it cannot report for 2 minutes. That is why `try` is the default.
- 8k context measures 69.7 tok/s, 2.5% below 614 — throughput is essentially
  context-independent, so the scarce resource is the *GPU*, not the context.

---

## Risks

- **Exit code 75 could be mistaken for success** by a script that only checks
  for non-zero. Mitigate by using it consistently and documenting it; `gpuq` is
  the only thing agents should call, and it can refuse to swallow 75.
- **`LOCK_NB` makes contention fail fast, so a naive caller may spin.** That is
  why the agent rule is "do other work", not "retry immediately", and why a
  ticket file with position is worth the complexity.
- **Two lock files is a trap.** If the wrapper and the engine must both take a
  lock, they must be *different* files with a defined order, or dropped. The
  spec's central claim is that there is exactly one.
- **A ticket file is advisory state next to a kernel lock.** If it desynchronises
  (crash between enqueue and lock), position is wrong but the mutex is still
  correct. Acceptable: the lock is the safety property, the ticket file is only
  for the message an agent reads. Say so in the code.

---

## Acceptance criteria

1. `gpuq try -- kaplumba --prompt "Hi" …` runs and prints metrics, under 60 s.
2. While that runs, a second `gpuq try -- kaplumba …` exits **75** promptly
   (target: under 2 s) with a message naming the holder. No hang.
3. A third agent sees a queue position, and the position is correct when a
   waiter is killed mid-wait (its entry is pruned, the queue does not stall).
4. The in-binary lock is still the only lock owner: with `gpuq` entirely
   bypassed, two direct binary invocations still cannot run at once — verified
   by starting two and observing the second block or exit 75.
5. `KAPLUMBA_GPU_LOCK=wait` preserves today's behaviour, so nothing regresses
   for a caller that wants to block.
6. A 30-second soak: 6 concurrent `gpuq try` invocations, at most one model
   resident at any instant, all six either complete or exit 75, none hang.
7. The deadlock and its repro are in a comment in `gpuq`, and in
   `spec/engine-performance/implementation_notes.md`.

---

## Open questions

**Q1 — Do you want a *ticket queue* (agents get a position and are served in
order) or just a *mutex* (agents retry, no fairness)?**
A ticket queue is fairer: a 20-agent fleet cannot starve itself, and an agent
knows whether it is next. It costs a shared file and some pruning logic in the
engine, where the code is otherwise deliberately minimal. A mutex is simpler and
the kernel already provides it. I lean ticket queue, because with 20 agents
retrying randomly, unfairness becomes starvation, which looks exactly like a
hang — the failure I am trying to eliminate.

**Q2 — On contention, should the engine exit 75 immediately, or wait a short
grace period (say 5 s) first?**
Immediate is cleanest for agents: try-work-retry. But it means an agent doing a
quick 8-second benchmark loses its slot constantly to long ones, and burns
cycles. A short grace period smooths that, at the cost of making `try` not
instant — which is the property I most want to keep, since "instant" is what
stops an agent from sitting in a call. I lean immediate, and rely on the ticket
queue for fairness instead of grace periods.

**Q3 — Should `gpuq status` be allowed to take the lock briefly, so it can say
"free" with certainty?**
Right now it *probes* by trying, which is momentary and safe. But there is a
subtle race: between the probe releasing and the caller acting, someone else can
take it. That is fine for a human-readable status, and unavoidable in general.
Confirm that a racy status is acceptable, since the alternative is `status`
becoming a serialising operation — which would mean `status` can block behind
an 8k run, defeating the purpose of having a status command.

**Q4 — Do you want the engine to *hard-fail* rather than queue, ever?**
Today a second instance waits. If an agent fleet is meant to be safe by
construction, an env var like `KAPLUMBA_GPU_LOCK=error` that makes the second
instance exit 75 immediately regardless of mode would let a test assert
"never two instances" without timing games. Cheap to add, but it is a third
policy and I would rather not add policies nobody asked for.

**Q5 — Should I fix the `gpuq` bug now, or write the spec and wait?**
I have drafted only, as asked. The bug is live and will bite the next agent who
runs `gpuq try` with the engine, so I recommend implementing immediately — but
Q1 changes the shape of the fix, so I would rather have your answer first.
