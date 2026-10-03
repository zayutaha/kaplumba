# B2 — decode pipeline topology, and the decomposition nobody did

**Verdict: the 1-deep "structural blocker" is a stale doc comment about a kernel
the decode path stopped calling three commits ago. 1-deep is *not* the ceiling —
but the real blocker (one shared activation scratch) makes depth 2 worth ≈0.10 ms,
so it should not be built. The roll-threshold change proposed elsewhere is
**harmful on the non-pipelined path**, which is the only path that needs it.**

Agent B2. Read-only. **No model was loaded, no GPU was touched, `scripts/gpuq`
was never run.** Every line number is verified against the tree at `641e114`.

Test: `crates/engine/tests/probe_pipeline_topology.rs` — 16 tests, no GPU.
Sabotage-verified, 8/8 (table in §8).

---

## 0. Corrections to the brief's citations, first

The brief's line numbers are from an older tree. Verified at `641e114`:

| brief says | actually |
|---|---|
| `chat.rs:388` `let pipelined = self.greedy && …` | ✅ correct, `chat.rs:388` |
| `generate.rs:150` | ✅ exists, but it is in `generate` (the non-GPU entry). The one that matters is **`generate.rs:269`**, in `generate_gpu` |
| `generate.rs:935` | the gate is at **`generate.rs:971-973`**, in `generate_gpu_greedy` |
| `model.rs:3848` `commit_and_wait` | **`model.rs:3984`** |
| `model.rs:3745` `forward_step_pipelined` | **`model.rs:3877`** |
| `model.rs:3794` `read_host_token` | **`model.rs:3930`** |
| `model.rs:3695-3700` `encode_greedy_pick` refuses index 0 | **`model.rs:3821-3826`** — and the kernel it names is `elements.rs:815`, **not called from `model.rs` at all** (§1) |

Nothing else in the brief was wrong. In particular `750 dispatches/step`,
`16 command buffers`, `7 wait_until_completed` and the `+3.83 ms` all reproduce
from source.

---

## 1. The stated blocker is false. The device-side hand-off already exists.

`model.rs:3821-3826`, on `forward_buffer_into_step`:

> There is no device-side token hand-off to overlap the steps with, and there
> cannot be one at present: the pick's only host-visible output entry point,
> `ElementKernels::encode_greedy_pick`, refuses a destination index of 0, which
> is the index the next step's embedding reads.

Every clause of that is checkable, and none of it survives:

- **`encode_greedy_pick` is not on the decode path.** It is defined at
  `elements.rs:815` and its `row == 0` refusal is real (`elements.rs:823`) — but
  `model.rs` mentions it **only inside a doc comment**; `code_lines(model.rs)`
  contains no call. The head now calls `encode_argmax` (`model.rs:4611`) plus
  `encode_argmax_to_token` (`model.rs:4622`, → `elements.rs:890`).
- **`encode_greedy_pick` refused row 0 for a reason that no longer exists.** It
  wrote into the *host-staged* `scratch.token_ids`, where index 0 was the host's
  own input. The current hand-off writes into `token_ids_decode[next_slot]` — a
  dedicated, device-written, double-buffered slot.
- **The replacement writes index 0 unconditionally.**
  `argmax_keys_to_token` (`elements.metal:586-610`) ends:

  ```metal
  if (token_out != nullptr) {
      token_out[0] = token;          // elements.metal:604
  }
  if (host_token_out != nullptr) {
      host_token_out[step_idx] = token;
  }
  ```

  No destination-index guard. The kernel's only `== 0` is `if (lid == 0)`, the
  threadgroup-lid guard.

**Consequence:** the dependency "step N+1 needs step N's token" is *already*
satisfied on device, and has been for the whole 16.05 tok/s baseline.
`chat.rs:453-459` says so in as many words ("the previous step's argmax already
wrote this step's input slot on the GPU"), and `forward_step_pipelined` takes
`token: Option<u32>` precisely so that from step 1 onward the host passes `None`
(`generate.rs:1025`).

**So the precise statement of the dependency is:**

> `forward_step_pipelined` writes `query_positions_decode[slot]` from the host at
> **encode** time (`model.rs:3901`), and reads `token_ids_decode[slot]` on the
> GPU at **execute** time. A step's GPU work needs the previous step's pick to
> have landed in *its* slot. A single Metal queue executes command buffers in
> commit order, so a device-to-device hand-off needs no synchronisation at all —
> only a slot that no other in-flight step is writing. Depth D needs D+1 slots.

---

## 2. What actually blocks depth ≥ 2: one activation scratch, shared by the model

| in-flight step writes | where | shared with |
|---|---|---|
| `hidden_a` (from the embedding) | `model.rs:4318` | every layer of every in-flight step |
| `hidden_a` ⇄ `hidden_b` ping-pong | `model.rs:4341-4345` | 60 layers × all in-flight steps |
| `normed`, `transformed`, `gate`, `up`, `activated` | `encode_linear`, `model.rs:1346` | all steps |
| `q`, `k`, `v`, `proj`, `attended`, `raw`/`raw_b`/`raw_c` | ditto | all steps |
| `attn_partials` (fixed 1.05 MB) | `model.rs:107` | all steps |
| `last_hidden`, `logits`, `capped` | `model.rs:4544-4609` | all steps |

`DecoderLayer::forward_buffer` takes `scratch: &ForwardScratch` (`model.rs:1346`)
— a **shared** borrow with no slot/instance parameter. `forward_chunk` reaches
the same fields through `&self.scratch`. Two steps in flight on one queue would
have step N+1's `encode_embedding_scaled` overwrite `hidden_a` while step N's
layer 1 is still reading it. That is a **silent wrong-answer race**, not a
crash, which is worse.

Secondary constraint: `token_ids_decode` / `query_positions_decode` are
`[Buffer; 2]` (`model.rs:3042-3043`), and `forward_step_pipelined` folds the slot
modulo 2 (`model.rs:3886`). Depth D needs D+1 slots, since the host writes
position D steps ahead.

**Cost of depth 2.** Not memory, if done deliberately: a second scratch sized for
`tokens = 1` is ~1.4 MB (`logits`+`capped` 1 MiB, `attn_partials` 1.05 MB, the
rest < 250 KB). Duplicating the *prefill-sized* 372 MB set would be
indefensible (AGENTS.md §8); a decode-sized one is free. The cost is code: one
`&ForwardScratch` threaded through ~40 encode sites and the
`split_at_mut` borrow structure at `model.rs:4358-4361`.

**Not blocking:** the KV cursor. `gpu_cache.advance(tokens)` runs at
`model.rs:1720`/`:1831`, i.e. at **encode** time, and one queue executes in commit
order, so encode order == execute order.

---

## 3. Would depth 2 pay? **No — and the model's own arithmetic says so.**

Depth 1 already hides the *whole* encode: the host encodes step N+1 (750
dispatches) while the GPU runs step N. Nothing about the encode is on the
critical path. What depth 1 still exposes is the **host tail between `wait()`
returning and the next step's first commit** (`generate.rs:1033` → `:1020`):

| tail item | cost | grows with response length? |
|---|---|---|
| `model.read_host_token(in_flight_step)` (`generate.rs:1040`) | 4-byte host read of GPU-written shared memory | no |
| `detokenizer.push(step_token)` | replays **every id so far** (`tokenizer.rs:202-235`) | **yes** |
| `on_text(&chunk)` | the caller's stdout write, one syscall per chunk | no |
| `eos_ids.contains` | O(1) | no |
| `segments.record`, `pipeline_audit::assert_no_builds_since` | one relaxed atomic load each | no |

P06 measures this whole tail at **~0.104 ms** at 37 tokens (0.004 + detokenizer
~0.1), rising to ~0.105 ms/push at 119 tokens. That is the *entire* prize.

```
depth 1 step = max(GPU_busy, encode) + tail
depth 2 step = max(GPU_busy, encode) + 0
difference   = tail ≈ 0.10 ms  ⇒  +0.03 tok/s
```

**Against ~40 changed call sites, a second scratch's lifetime hazards, and a
silent-wrong-answer race if any of it is missed: no.** The `Detokenizer` fix
(device-sampler T4) captures the only growing part of the tail and is already a
task. This is pinned by
`depth_two_is_worth_the_post_wait_tail_and_nothing_else` so the claim cannot be
quietly upgraded to a multi-millisecond one.

---

## 4. Timelines, from source

### Pipelined greedy (`generate.rs:1013-1057`, `--temperature 0`, 62.30 ms)

```
        ┌──────────── step N (GPU) ────────────┐  ┌─ step N+1 ─┐
host    │  read tok · detok · stdout · eos      │  │            │
        └── ▲ ─────────────────────────────────┘  └── (encode) ─┘
             │ wait (generate.rs:1033)
GPU   ───────┴══════════ idle ~0.10 ms ══ busy ~62 ms ══ idle 0.10 ══ busy …
                    ▲ first commit of N+1 lands here (dispatch 50 of 750)
enc   step N+1: 750 dispatches, 16 commits, 0-7 roll() throttles, all hidden
```

Two command buffers' worth of *inputs* (the 2 slots) is what lets the host write
step N+1's position while the GPU reads step N's. One step of *lookahead* is
what lets the host encode while the GPU runs.

### Barrier per token (`generate.rs:322-372`, `--temperature 1.0`, 69.60 ms)

```
host  ┌widen 262,144 fp16→f32 (1 MiB alloc) ┐┌sample 2.265 ms ┐┌encode 750 ┐
      └────────── GPU idle ─────────────────┘└──── GPU idle ──┘└─ hidden ──┘
GPU   ═══════ idle ═══════════════ idle ═══════════════════════ idle ═══════ busy ~62 ms
                          ▲ commit_and_wait returns (model.rs:3984)
      └─ 512 KiB copy (model.rs:3988) ─┘
```

**Textual order inside one iteration is `sample → submit`, not `submit →
sample`** — the `wait` lives at the *end of the previous* iteration, inside
`forward_buffer_into_vec`. So the chain is exactly
`wait → copy → widen → sample → [loop] encode+submit`, fully serial, and the
sample sits in the window where the GPU has nothing queued. **Confirmed, not
assumed** (`temperature_path_is_wait_copy_widen_sample_submit`).

**Nothing overlaps.** Not the copy (it is after `commit_and_wait`,
`model.rs:3984` → `:3988`), not the widen, not the sample. This is the shape
device-sampler exists to remove.

---

## 5. Decomposing the +3.83 ms — and finding that the ledger does not close

### 5.1 What the A/B actually compared

The 3.83 ms came from `chat.rs`'s two arms, which differ in **five** ways, not one:

| # | difference | pipelined arm | barrier arm | who pays it |
|---|---|---|---|---|
| 1 | encode/GPU overlap | host encodes N+1 during N | fully serial | **the barrier** |
| 2 | `head/softcap` dispatch | **skipped** (`out.is_some()` ⇒ `reads_raw_row` ⇒ raw row) | **emitted** (512 KiB vocab pass) | barrier arm, ~0.01 ms |
| 3 | `head/argmax_token` dispatch | emitted | not emitted | — |
| 4 | 512 KiB `copy_nonoverlapping` | **none** | yes | barrier arm, ~0.05-0.15 ms |
| 5 | **host `argmax_logits` over all 262,144 values** (`chat.rs:508`, `generate.rs:471`) | **none** (4-byte `read_host_token`) | yes, per token | barrier arm, **~0.26 ms** (P06) |

Net dispatch count is **equal** (argmax 2 shared; +1 argmax_token / −1 softcap), so
the delta is not a dispatch-count artifact. But items 4 and 5 are **~0.3-0.4 ms
of A/B-arm artifact**.

Critically, item 5 is a cost the **temperature-1.0 loop never pays** — it samples
instead of argmax-ing. So 3.83 ms is not a clean barrier measurement for the path
the ledger applies it to. **Genuine overlap term ≈ 3.4-3.5 ms.**

### 5.2 The temperature-1.0 excess, term by term

| term | ms | status |
|---|---|---|
| lost encode/GPU overlap | **3.4 – 3.5** | derived: 3.83 − 0.26 scan − ~0.10 copy |
| `Sampler::sample`, 262,144 wide | **2.265** | measured (P06, median of 21) |
| fp16→f32 widen + 1 MiB `Vec` | **0.067** | measured (P06) |
| 512 KiB readback | 0.05 – 0.15 | bracketed, never measured |
| **sum** | **5.8 – 6.0** | vs **7.30** measured |

### 5.3 🔴 The ledger over-predicts by 2.94 ms, and that is not a rounding error

Feed the programme's own numbers into the shape model in the test
(`barrier_step − pipelined_step`):

```
encode   = 750 × 9.03 µs   = 6.77 ms   (P02/A5, "per-dispatch latency")
readback = 2.265 + 0.067 + ~0.10 = 2.43 ms   (P06/A4)
predicted gap = encode + readback        = 9.20 ms
measured  gap = 69.60 − 62.30            = 7.30 ms
shortfall                                 = 1.90 ms
```

(Using A5's own readback of 3.47 instead of 2.43 gives a 10.24 ms prediction and
a **2.94 ms** shortfall — which is exactly the number A5 §4 reports as "the 1-deep
pipeline already hides 2.94 ms", reached independently from the other direction.
That is a real cross-check, not a coincidence.)

**What this means.** The 3.83 ms and the 6.77 ms cannot both be read as *host time
on the critical path of the barrier arm*. At most one is, and which one is not
established. **"The 3.83 ms is the barrier" is a label, not a decomposition.**
Pinned by `the_published_inputs_predict_a_gap_larger_than_the_measured_one`,
which fails if anyone closes the gap without updating §9.

### 5.4 The instrument that would settle it exists and is one-sided

`KAPLUMBA_STEP_TRACE=1` prints `encode=` and `wait=` separately — but only on the
**non-pipelined** arms (`generate.rs:1124`, `:361`). The pipelined arm prints
nothing. Running it with and without `KAPLUMBA_NO_PIPELINE` gives the barrier
arm's split directly; the pipelined arm needs one `eprintln!("STEP_TRACE: …")`
added. **No run of this exists in the tree.** Cheapest possible resolution of
§5.3, and it needs no code change at all for the arm that matters most.

Note the `KAPLUMBA_NO_PIPELINE` control inside `generate_gpu_greedy`
(`generate.rs:1101-1121`) is a *better* control than `chat.rs`'s: it uses
`forward_buffer_into_step` + `model.last_token()`, so it pays neither the 512 KiB
copy nor the host argmax scan. Comparing **it** against the pipelined arm isolates
the barrier with no artifact. The 3.83 ms was measured on the worse control.

---

## 6. Q3: is `decode_max_dispatches_per_buffer() → u32::MAX` safe on the non-pipelined path?

**It does not deadlock, and it does not worsen the barrier. It makes that path
substantially WORSE, and the earlier agent's reasoning was right only for the
pipelined path.**

### 6.1 Safety — yes, structurally

- The threshold is reached on the non-pipelined path too: `model.rs:3964` (inside
  `forward_buffer_into_vec`) and `generate.rs:1102`. So the change is not gated
  away from the path that needs it.
- **No buffer becomes un-waited.** `roll` (`encoder.rs:539`) is the only place a
  buffer is committed *and* pushed to `in_flight`, and it is reachable only from
  `tick` (`encoder.rs:524`) on the dispatch counter. Raise the counter's ceiling
  and every commit path still records what it commits:
  `commit_terminal` (`:613`), `commit_stamped` (`:685`), `commit_and_wait`
  (`:643`) all touch `in_flight`. `commit_and_wait` additionally status-checks
  every buffer, so a fault in a would-be-rolled buffer still surfaces.
- **No shared-memory hazard.** The host writes `token_ids_decode[slot]` /
  `query_positions_decode[slot]` at `model.rs:3896`/`:3901`, and the
  non-pipelined path's previous call ended in `commit_and_wait`
  (`model.rs:3984`), so nothing is in flight. Fewer intermediate commits cannot
  create a race that fewer commits would remove.
- The prefill knob is separate (`generate.rs:851`/`:949`) and must stay separate —
  `encoder.rs:257-268` says why: folding them together makes the decode A/B move
  prefill's too.

### 6.2 Performance — 🔴 **rolling is load-bearing on the barrier arm**

The whole point of rolling (documented at `encoder.rs:210-217`) is that a step
takes ~5 ms to encode, and without an early commit "the GPU receives nothing
until the host has finished encoding and then runs ~73 ms with the host blocked
in the barrier."

On the **pipelined** path the GPU is a full step behind — step N was committed a
whole encode ago — so the early commit buys nothing and the 16 commits + 7 waits
are ~0.1-0.3 ms of waste.

On the **barrier** path the GPU is idle from t=0 until the first commit. So:

```
with rolling:    GPU idle = t(first 50 dispatches) ≈ 1/15 of an encode
without rolling: GPU idle = E, the whole encode
```

and the step grows by `E − t(50)`. The magnitude is bounded but the sign is not
in doubt. `E ≈ 6.8 ms` (P02/A5, 750 × 9.03 µs); `t(50)` is the host time to
encode 50 dispatches, which the `max_in_flight() = 8` window (`encoder.rs:543`)
may inflate by throttling the host against the GPU. So the added idle is
**somewhere in [~0.45, 6.8] ms** — at the low end if the first fifteen buffers
encode freely, at the high end if throttling makes the host track the GPU
throughout.

**Against a gain of 0.1-0.3 ms, the expected value of this change on this path is
negative.** Worse, it is negative *by construction*: rolling is the only mechanism
that lets the barrier arm's GPU start before the host stops encoding.

The correct form of the change is **topology-scoped**: keep 50 on the barrier arm,
raise it only where a step is already in flight. That is a second knob beside
`decode_max_dispatches_per_buffer` plus two call sites — not a constant, and not
the change as proposed.

### 6.3 One diagnostic is lost either way

`KAPLUMBA_MAX_IN_FLIGHT=1` (`encoder.rs:236-245`) currently serialises decode
completely — that is how shared-memory races were told apart from numerical
differences. With rolling off on the pipelined arm it stops biting there. Worth a
line in the commit message, not worth blocking on.

---

## 7. Q5: the theoretical floor for `--temperature 1.0`

Once sampling is on device and the pipeline is ungated, the temperature-1.0 step
is the temperature-0 step plus the dispatches the sampler needs:

```
temperature-0 pipelined, measured          62.30 ms
+ head/softcap   (Sample must read capped;  ~0.01 ms
   the greedy pipelined arm skips it,
   model.rs:4597-4608)
+ 2 sampling dispatches (A4's design)      ~0.04 ms
− nothing removed (the greedy arm has no
   host chain to remove)
──────────────────────────────────────────────
floor                                   ≈ 62.35 ms  ≈ 16.04 tok/s
```

**The floor is parity, not a gain: 14.37 → ~16.0 tok/s.** `spec/device-sampler`'s
own "parity, not beyond" is confirmed by arithmetic, and A5's +1.53 is
conservative only because of the double-count in §5.2.

**What else still differs between the two paths** — the honest list, none of it
large:

1. `head/softcap` is emitted for `Sample` and skipped for `Argmax` (above). ~0.01 ms.
2. The device sampler reads all 262,144 capped values for the top-k reduce: ~0.5 MiB
   of extra traffic, ~0.005 ms. Hidden by the same pass as the softcap.
3. `read_host_token` and `Detokenizer::push` — identical on both.
4. **`--seed` is not bit-reproducible across the host/device boundary.** Metal
   `exp(float)` is not `f32::exp`. Not a throughput term; a correctness
   disclosure (device-sampler spec, "Reproducibility").
5. The 2.94 ms of §5.3 is unallocated. If it is real host time on the barrier
   arm, the temperature-1.0 path lands at ~63.6 ms / 15.7 tok/s and **not** at
   parity. **The programme cannot currently tell which**, and that is the single
   most valuable thing left to measure on this angle.

---

## 8. Sabotage log (AGENTS.md §3.4 — a claim is not a claim until it has failed)

Harness: the 5 source files at `641e114` extracted to a scratch tree, the test
compiled with `rustc --test` and run standalone. **This was necessary because the
shared working tree does not compile** — see §10.

| # | sabotage | test that failed |
|---|---|---|
| S1 | add `if (row == 0) { return; }` to `argmax_keys_to_token` | `device_token_handoff_writes_index_zero` |
| S2 | make `model.rs` *call* `encode_greedy_pick` | `the_row_zero_blocker_is_real_but_off_the_decode_path` |
| S3 | `scratch: &ForwardScratch` → `&mut ForwardScratch` on `DecoderLayer::forward_buffer` | `activations_are_single_instance_per_model` |
| S4 | drop `set_max_dispatches(decode_…)` from `forward_buffer_into_vec` | `the_roll_threshold_reaches_the_non_pipelined_path` |
| S5 | submit before sampling in `generate_gpu` | `temperature_path_is_wait_copy_widen_sample_submit` |
| S7 | remove `on_text(&chunk)` from the post-wait tail | `the_depth_one_prize_is_a_small_host_tail` |
| S8 | move the 512 KiB copy before `commit_and_wait` | `readback_happens_after_the_barrier_not_before_it` |
| S9 | fold the prefill knob into the decode knob | `the_roll_threshold_reaches_the_non_pipelined_path` |
| S10 | add a `slot: usize` parameter to `DecoderLayer::forward_buffer` | `activations_are_single_instance_per_model` |

Control after each: **16 passed, 0 failed.** Each sabotage fails exactly its
intended test and no other.

**One test is not sabotage-verified and should not be trusted as a tripwire:**
`the_depth_one_prize_is_a_small_host_tail` recognises tail operations by the
calls it enumerates. It catches a removed item (S7) and a second *recognised* O(n)
item; it would **not** catch host work it has never heard of. That limitation is
stated in the file's own doc comment.

**Not verified:** nothing in this file was run against real weights, because
running it that way requires the GPU, which another agent holds. Every assertion
is a claim about *source text*, which is the only kind of claim this file makes.

---

## 9. Recommended next measurements, in order

1. **`KAPLUMBA_STEP_TRACE=1` with and without `KAPLUMBA_NO_PIPELINE`, comparing
   the pipelined arm against `generate_gpu_greedy`'s barrier arm (not
   `chat.rs`'s).** Zero code change, one number each, and it settles §5.3 — the
   2.94 ms. Add one `eprintln!` to the pipelined arm so both sides report.
2. **Hoist the two per-step heap allocations** (§10, deliverable). Largest
   remaining host-side item on the default path after the sampler itself.
3. **Topology-scoped roll threshold**: keep 50 on the barrier arm, raise it on
   the pipelined arm only. 0.1-0.3 ms, bit-exactness-neutral, and it becomes a
   legitimate change only once scoped.
4. **Delete or correct `model.rs:3821-3826`.** It is a false statement about a
   dead kernel, it is quoted as load-bearing in `P02-dispatch-overhead.md`, and
   `the_row_zero_blocker_is_real_but_off_the_decode_path` will keep failing
   loudly if anyone revives `encode_greedy_pick`.

---

## 10. Deliverable: cheapest change to the temperature-1.0 path that is not the device sampler

### Recommendation: hoist the two per-step heap allocations out of the hot path

| site | allocation | per step |
|---|---|---|
| `generate.rs:326-329` | `logits.iter().map(f16→f32).collect::<Vec<_>>()` | **1 MiB**, 262,144 elements |
| `sampler.rs:71` `top_k_order` | `(0..262144).collect::<Vec<_>>()` | **2 MiB**, 262,144 elements |

Both are freed at the end of the iteration and re-taken on the next one, so the
temperature-1.0 loop does **3 MiB of malloc/free and 524,288 element writes per
generated token** that the temperature-0 loop never does at all. Hoist both into
buffers owned by `Sampler` / the loop and `clear()`+`extend` them.

- **ms value: unmeasured. Estimate 0.2 – 0.5 ms/step**, on the mechanism (3 MiB
  of alloc + first-touch per token; `top_k_order`'s comparator is also
  pointer-chasing into the 1 MiB widened array). It is the only host-only item on
  this path with a plausible value **above** the roll change's 0.1-0.3 ms.
- **Bit-exactness-neutral: YES, provably.** `top_k_order` builds `order` as
  `0..n` in index order every call; reusing a cleared buffer yields the identical
  sequence, and the comparator (`logits[right] vs logits[left]`, then `left vs
  right`) and `select_nth_unstable_by` are untouched. The widen's element order is
  untouched. No kernel, no device buffer, no RNG state is involved.
- **Risk: near zero.** The only way to get this wrong is to change the *order* of
  the `order` vector, which the `then_with(|| left.cmp(right))` tiebreak already
  pins.
- **Measure before believing.** 0.2-0.5 ms is a mechanism, not a number. It is
  cheap to measure: `perf_probe_host_ops.rs` already times `sample()` at width
  262,144; time `top_k_order`'s allocation separately.

### Rejected: the roll threshold, as a global constant

0.1-0.3 ms and bit-exactness-neutral, but **net-negative on this path** (§6.2).
Only worth doing once scoped per topology.

### Rejected: dropping the finiteness scan

`sampler.rs:178` scans all 262,144 values for NaN/Inf every token, before the
top-k. Probably ~0.05-0.1 ms, but removing it **changes when `InvalidLogits` is
raised**, so it is not bit-exactness-neutral and it is a behaviour change dressed
as an optimisation. Leave it.

---

## 11. 🔴 Process finding: two agents are in one working tree

`git status` at the time of writing:

```
 M crates/engine/src/model.rs        <- `pub struct DeviceSample` declared INSIDE
 M crates/engine/src/sampler.rs          `impl TextModel` (model.rs:4014)
 M crates/kernels/src/elements.metal
 M crates/kernels/src/elements.rs    <- so `cargo build` FAILS: 4 errors
?? crates/engine/tests/probe_pipeline_topology.rs   (mine)
?? crates/engine/tests/probe_sampling_phases.rs    (not mine)
```

The device-sampler T2 agent is mid-edit on `model.rs`, `sampler.rs`,
`elements.metal` and `elements.rs` — **the same four files this investigation
reads**. AGENTS.md §6 forbids concurrent edits to the same files and requires
serialising work.

Two consequences, both worth recording:

1. `cargo test -p kaplumba-engine` cannot run at all right now. This file was
   therefore compiled and executed standalone (`rustc --test` against `641e114`
   snapshots of the 5 sources). **It has never been compiled inside the crate**,
   so it may need trivial fixes — most likely a `use` or an unused-variable
   warning — when the tree builds again. Run `cargo test -p kaplumba-engine
   --test probe_pipeline_topology` before trusting it.
2. Any conclusion here that depends on those four files is stated **as of
   `641e114`** and must be re-checked against T2's merge. T2 rewrites
   `elements.metal` and `elements.rs`, so `device_token_handoff_writes_index_zero`
   and `the_row_zero_blocker_is_real_but_off_the_decode_path` are the two tests
   most likely to need re-basing.

Also note `crates/engine/tests/probe_sampling_phases.rs` appeared untracked
during this session. Two agents writing `probe_*.rs` into the same tests
directory is a collision risk; prefix discipline should be assigned, not assumed.