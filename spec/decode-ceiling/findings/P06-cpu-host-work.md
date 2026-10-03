# P06 — CPU per-token host work — **CONFIRMED, and the finding reframes the target**

Test: `crates/engine/tests/perf_probe_host_ops.rs` (1077 lines, 15 pass + 1 ignored).
All 10 source-assertion tests sabotage-verified.

## 🔴 The headline: at temperature > 0 the engine does not pipeline AT ALL

`chat.rs:388` — `let pipelined = self.greedy && ...`. `greedy` is set **only** by
`new_greedy` (`chat.rs:282`), which `crates/cli/src/chat.rs:35` calls only when
`temperature == 0.0`. **`DEFAULT_TEMPERATURE = 1.0` (`crates/cli/src/args.rs:30`)
— the shipped default is the slow path.** Same in `generate.rs:150`/`:322`: the
sampling loop calls `forward_buffer_into_vec` → `commit_and_wait`
(`model.rs:3848`), a **full barrier every token**.

**Consequence: every `decode_tok_s` figure measured at `--temperature 0` — including
the 16.05–16.10 baseline — is the pipelined path, which is NOT what the engine
does at its default temperature.**

Measured per-step host cost (release, median of 21):

| step | ms |
|---|---|
| `sample()` at width 262,144 (`sampler.rs:110`) | **2.265** (min 2.243, max 2.527) |
| fp16→f32 widen (`generate.rs:326-330`) | +0.067 |
| host argmax scan (non-greedy baseline) | +0.262 |
| 512 KiB device→host copy into a fresh `Vec` (`model.rs:3849`) | ~0.05–0.15 |
| **temperature>0 penalty** | **≈ 2.4–2.6** |

**None of it overlaps.** `sampling_cannot_overlap_with_the_gpu_today` pins that the
submit (`generate.rs:360`, `chat.rs:532`) is textually *after* the sample in both
bodies. The GPU is idle for the whole 2.6 ms.

## The brief's premise was wrong; the real cost is elsewhere

`max_in_flight()` (`encoder.rs:239`) and `decode_max_dispatches_per_buffer()` (`:272`)
**already memoise in a `OnceLock`** — `std::env::var` runs once per process, not per
token. ≈0.0005 ms/step. Ignore.

What *is* per-token and uncached:
- **`ablated()`** (`kernels/src/lib.rs:38`) is `Mutex<HashSet<String>>` — lock + String
  hash per call, ~360/token.
- **`profile.stamp(encoder, &format!(...))`** — 30 sites in `model.rs`. The
  `format!` is an **argument**, evaluated *before* `stamp`'s `if !self.enabled`
  early return (`model.rs:830`). **~540 String allocations/token with the profiler off.**
- **`linear.rs:1542-1554`, `:2247`** build `format!("...k{bits}...")` then
  `HashMap::get(&String)` — **~240 String allocs + 240 hashes/token** for 12 fixed names.

Total ≈1,100 heap allocations + ~600 mutex/hash pairs per token ≈ 0.1–0.2 ms/step.
Real, but not the lever.

## 🔴 Bonus: `Detokenizer::push` is quadratic, and is not overlapped

`tokenizer.rs:202-235` builds a **fresh** `decode_stream` and replays **every id so
far**. Measured (119-token response): 0.017 → 0.049 → 0.085 → **0.105 ms/push** —
**6.3× growth**. It runs at `generate.rs:347`, `:992`, `chat.rs:443`, `:519` —
**between `wait()` and the next submit even in the pipelined greedy loop**, so it is
pure GPU idle. ~0.1 ms/token at 119 tokens, and growing.

## Ranked changes

| # | change | file:line | ms/step |
|---|---|---|---|
| 1 | **On-device top-k sampler** (top-k reduce + softmax + inverse-CDF in one dispatch), RNG state in a buffer, staged into `host_tokens`, mirroring `encode_argmax_to_token`. Makes pipelining temperature-agnostic. | new kernel; `model.rs:4459-4479` | **2.4–2.6** |
| 2 | Gate the pipelined loop on `greedy \|\| sampler_on_device` | `chat.rs:388`, `generate.rs:150`/`:935` | enables #1 |
| 3 | Make `Detokenizer` incremental — hold the stream, drop the replay | `tokenizer.rs:202-235` | ~0.1, growing |
| 4 | Hoist stamp labels to `&'static str`, or gate the `format!` behind `gpu.enabled()` | 30 sites in `model.rs` | ~0.03 |
| 5 | Resolve matvec pipelines by index, not `format!`-built name | `linear.rs:1542`, `:2247` | ~0.02 |
| 6 | `ablated()` → `OnceLock<HashSet>` | `kernels/src/lib.rs:38` | ~0.02 |

## Timeline, 62.3 ms/step

| | greedy pipelined | temperature>0 |
|---|---|---|
| GPU-busy | ~60 | ~57.5 |
| GPU-idle, host work **after** wait | 0.004 + detok ~0.1 | **~2.7** |

## Ceiling

1+3 recover **2.5–2.7 ms** ⇒ 62.3 → **~59.7 ms = 16.8 tok/s**. 4+5+6 add ~0.07.

**Host work alone cannot reach 50 ms.** The target needs 12.3 ms; the entire
non-GEMV host surface is ~2.7 ms. This independently confirms
`implementation_notes.md` §18.3 — the floor is bandwidth (~58.3 ms at 203 GB/s).

**The realistic target is ~16.8 tok/s, and change #1 is what gets the DEFAULT
temperature there.**