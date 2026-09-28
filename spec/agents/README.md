# Agent coordination

Multiple agents work this repo concurrently. This file is how they coordinate.
It is the shared, non-code state that complements `AGENTS.md`.

## The box is the scarce resource

24 GiB of unified memory, no discrete GPU. The model needs 11.5-15 GiB. Two
concurrent model instances exceed physical memory and were measured swapping
3.9 GiB and collapsing to 1.4 tok/s. **One model process at a time, ever.**

```bash
scripts/gpuq status          # instant: holder, queue depth, available memory
scripts/gpuq try -- <cmd>    # runs only if free; exit 75 if busy
scripts/gpuq run -- <cmd>    # blocks until free, heartbeat every 5s
```

`try` is the default. **If `try` exits 75, do not sleep and retry.** Do the
analysis that needs no GPU — read code, draft the kernel, count instructions —
then try again. `run` is for when there is genuinely nothing else to do, and a
20-minute block is a wasted 20 minutes.

Do not trust `vm_stat` `Pages free` for memory. It ignores the file cache; on an
idle box it read 1.47 GiB while `top` said 14 GiB unused and a load took 10.8 s.
Use `gpuq status`, which reads `top`.

## Worktrees

Every agent that edits code gets its own worktree, so two agents never have the
same file open:

```bash
git worktree add /tmp/wt-<tag> HEAD
cd /tmp/wt-<tag>
export CARGO_TARGET_DIR=/tmp/wt-<tag>-target   # sharing target/ breaks other
mkdir -p scripts && cp ../scripts/gpuq scripts/ # agents' benchmarks
```

Never edit the main tree from a worktree agent. Never commit to master — leave
changes uncommitted and report the diff. Clean up with
`git worktree remove /tmp/wt-<tag> --force` and delete any
`crates/engine/examples/*.rs` you added. A stray benchmark binary holds the GPU
and is invisible to the next agent.

## Per-agent spec, under `spec/agents/<tag>/`

Each agent gets `spec/agents/<tag>/` with the four `AGENTS.md` files, so its
reasoning survives even if the agent's context is exhausted. Write them as you
go, not at the end — an agent that runs out of context with nothing written has
wasted its work.

```
spec/agents/<tag>/spec.md                 # what I was asked, and the goal
spec/agents/<tag>/plan.md                 # the design I chose and why
spec/agents/<tag>/tasks.md                # done / not done / blocked
spec/agents/<tag>/implementation_notes.md # measurements, dead ends, invariants
```

`implementation_notes.md` is the important one. It must contain, densely:

- every number measured, with the command that produced it
- every approach tried that FAILED, and the number that killed it
- any invariant discovered in existing code that constrains later work
- exact commands for build, test and benchmark

Write down wrong conclusions too, and the measurement that disproved them. This
repo has produced three false test-coverage claims and two wrong performance
diagnoses; all five cost hours. The one that survives is the record of how they
were killed.

## Do not ask questions

**Assume an answer and record it.** An agent that stops to ask has stalled, and
the question may not get answered before the window closes. Instead:

1. Pick the most defensible option.
2. Write it into `spec/agents/<tag>/spec.md` under "Assumptions".
3. Keep going.

If a wrong assumption turns out to matter, the note says so and the next agent
knows to revisit it. That is strictly better than silence. The exceptions are
genuinely destructive, irreversible, or ambiguous-in-a-way-that-destroys-work —
e.g. a quality/accuracy trade, a licence question, or spending the last of a
scarce resource. Those go in the shared `spec/engine-performance/` notes as an
explicit open question with the options, and the agent picks the reversible one
and says so.

## Reporting

- Numbers with units and the command that produced them. Not "much faster" but
  "68.8 -> 71.5 tok/s, median of 5 interleaved runs".
- Medians and spreads, never single readings. This box has produced 3.7 to 6.4
  tok/s from the same binary minutes apart.
- Negative results stated plainly. "Nothing beat 71.5, here is why it is at the
  ceiling" is a valuable result and stops the next agent retrying it.
- What is left running. If you started a benchmark or left a worktree, say so.
