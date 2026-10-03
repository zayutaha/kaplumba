# A4 — device-side sampler: design, and the one thing that blocks it

**Verdict: buildable in 2 dispatches; the RNG is bit-exactly portable; the top-p
"sort the logits" problem does not exist because the shipped semantics are
top-k *then* top-p; the blocking risk is not arithmetic at all — it is that
`model.rs:4476` will hand the new sampler the *uncapped* logits.**

Scope: design only. No file outside this one was modified. No model was loaded,
no GPU was touched, `scripts/gpuq` was never run.

---

## 1. What already exists to build on

All line numbers verified against the tree at the time of writing.

### The primitive that makes this easy: a monotone 16-bit key for an fp16

`crates/kernels/src/elements.metal:527-541`

```metal
static inline uint sortable16(uint bits) { return (bits & 0x8000u) != 0u ? ~bits : (bits ^ 0x8000u); }
static inline uint value_key(uint bits)  { /* NaN -> 0, ±0 -> 0x8000, else sortable16 */ }
```

`value_key` is an order-preserving map from the fp16 bit pattern to a `uint`
whose unsigned comparison is the fp16 comparison. Two consequences the whole
design rests on:

- **fp16 → f32 is exact**, so on device `float(capped[i])` compares identically to
  the host's `half::f16::from_bits(x).to_f32()`. The **top-64 set and its order
  are therefore exactly reproducible on device**, with no tolerance anywhere.
- NaN is folded to key 0 so it can never win, matching `sampler.rs:102`'s
  `InvalidLogits` rejection without needing a validity scan.

`argmax_key` (`elements.metal:545`) packs `(value_key << 16) | (0xFFFF - index)`
— a `u32` whose unsigned max is exactly the `(logit desc, index asc)` total
order that `sampler.rs:71-84` implements. **Reuse this verbatim.** The 16-bit
index field is safe here because D1/D2 reduce within a bounded slice and the
*global* index is reconstructed as `slice_base + in_slice`, exactly as
`argmax_keys_to_token` (`elements.metal:601`) already does for the ≥65,536
underflow that broke the old `greedy_pick`.

### Reduction primitives

| primitive | where | shape |
|---|---|---|
| 32-lane max by shuffle | `elements.metal:578-580` | `for (o=16;o;o>>=1) v=max(v,simd_shuffle_xor(v,o))` |
| cross-threadgroup max by atomic | `elements.metal:581-583` | `atomic_fetch_max_explicit(&best[slot], v, memory_order_relaxed)` |
| shared-memory tree reduce | `elements.metal:121-136` (`rms_norm_weighted_tg`) | `threadgroup float partial[256]` + `threadgroup_barrier(mem_flags::mem_threadgroup)` + halving |
| atomic clear | `elements.metal:554-560` (`argmax_clear`) | `atomic_store_explicit(..., relaxed)` |

There is **no** prefix-sum, **no** `simd_prefix_sum`, and **no** softmax/RNG
kernel in `elements.metal`'s 25 kernels — confirmed by grep. Nothing to reuse
for those; they are new.

### `exp`

`elements.metal` has `gelu_tanh_value` (used at `:322`, `:334`) which calls
`exp`; `softcap` (`:417-426`) calls `tanh`. So `exp`/`tanh` in f32 are already
in use on this path — no new transcendental dependency. See §3 for why `exp`
is nevertheless the *only* thing that cannot be made bit-identical.

### Buffer / params conventions

- Params struct: `#[repr(C)] #[derive(Clone, Copy)]` + `params_bytes(&p)`
  (`elements.rs:1568`). Pattern to imitate: `ArgmaxParams` declared **twice**,
  once in `elements.rs:869-874` and once in `elements.metal:549-552`. Follow it;
  it is what makes the layout auditable.
- Dispatch: `encoder.dispatch_groups(&pipeline, &[Some(buf)…], params_bytes(&p),
  groups, threads)` (`elements.rs:879-885`). `dispatch_1d(p, …, threads, 64)`
  for 1-D (`elements.rs:860`).
- Pipeline registration: field at `elements.rs:224-226`, build at
  `elements.rs:321-323` via `pipeline(&device, &library, "name")` (which routes
  through `pipeline_audit`, so "compiles once at load" stays a measurement).
- All pipelines are built in `ElementKernels::new()` — **so a sampler pipeline
  adds load-time compile cost even on the greedy path.** Two pipelines, gated
  by an env var at construction, or accept it; measure.
- Scratch: `ForwardScratch` (`model.rs:2927-2960`), `StorageModeShared`
  (`model.rs:3001`, tracked mode — do not change this, see the comment at
  `:2995-3000` about `eed081e` producing NaN logits), sized from shapes only,
  never from context length (AGENTS.md §8). Must be added to `resident_bytes()`
  (`:3076-3107`) or the memory proof in `tests/memory_proof.rs` stops accounting
  for it.

---

## 2. The algorithm — 2 dispatches, both inside the step's own command buffer

`top_k = 64` (default), vocab 262,144, `maxTotalThreadsPerThreadgroup = 1024`.

### D1 `sample_topk_blocks` — `dispatch_groups(groups = 64, threads = 1024)`

64 slices × 4096 elements. Each threadgroup returns **its own** top-64, packed
keys, into `candidates[group*64 + rank]`.

Per group:
1. Each of the 1024 lanes loads 4 elements, builds 4 `argmax_key`s, and its
   **per-lane top-4** (all 4 — a per-lane top-1 is *not* provably safe, and this
   is the one place a lazy implementation silently loses candidates).
2. Bitonic top-64 out of the staged 4096. Stage size: `threadgroup uint keys[4096]`
   = **16 KiB**, inside the 32 KiB per-threadgroup budget.
3. Rank 0 must be the slice max. Because `argmax_key` is a *total* order,
   the sorted output is unique — no tie ambiguity, and `atomic` ordering cannot
   affect the result.

Output: 4096 `u32` = 16 KiB. **Note the read amplification:** each of the 64
groups reads its own 4096-element slice, so the total is still exactly 262,144
elements = 512 KiB read once. No amplification.

### D2 `sample_draw` — `dispatch_groups(groups = 1, threads = 1024)`

One threadgroup. Everything else happens here:

1. **Global top-64** of the 4096 candidates — the *same* bitonic routine as D1,
   same shared array size. Reuse by factoring it into a `static inline`
   (`topk_bitonic(threadgroup uint*, …)`), not a second copy.
2. **Sort 64 descending** by `value_key`, ties by ascending index. 21-stage
   bitonic network, one thread or 64 threads — microseconds either way.
3. **Softmax over the survivors**, `w[i] = exp((x[i] - max)/T)`, then
   `total = Σ w[i]`, then the top-p walk, then the inverse-CDF walk — all by a
   **single designated thread looping i = 0..63 in index order**. This is
   deliberate: a single-threaded in-order f32 sum is **bit-identical** to
   `sampler.rs:116`'s `weights.iter().sum::<f32>()`. Parallelising this would
   throw away exactness for nothing.
4. Advance the RNG (§3), draw, and write **two** words:
   `out[0]` (→ `token_ids_decode[1-slot]`, the next step's input) and
   `host_tokens[step_idx]` (→ `read_host_token`).

### Why not one dispatch

D1 must be a separate dispatch because 64 threadgroups cannot see each other's
candidates; a grid-wide reduction needs either a second dispatch or a
persistent kernel with a device-side barrier. Metal has no grid sync. **Two
dispatches is the minimum.** Three is defensible (split the top-p/CDF into its
own 1-thread dispatch) and is the fallback if D2's shared-memory pressure
proves awkward, at a cost of ~5 µs of launch.

### Shared-memory budget

| | bytes |
|---|---|
| D1/D2 `threadgroup uint keys[4096]` | 16,384 |
| D2 `threadgroup float w[64]` | 256 |
| slack | ~15,000 |
| **total** | **~16.6 KiB / 32 KiB** |

Dispatch shape is the one uncertainty I could not settle without the GPU:
1024 threads is legal but is usually *not* optimal on Apple silicon. A/B
`threads = 256, 512, 1024` and `groups = 64, 128, 256` (at 256 groups, per-group
top-64 = 16384 candidates into D2, which still fits if D2 stages in two passes
of 4096). Pick on measurement.

### Cost

2 dispatches, 512 KiB read + 16 KiB write, all inside the step's command buffer,
so it overlaps the ~60 ms of GEMV. Wire time at 273 GB/s is **1.9 µs**; real cost
including launch, atomics-free reduction and 78 barriers in the bitonic network
is **0.03–0.06 ms**. Against a 62.3 ms step that is free, *provided* it is
inside the pipelined buffer. Measured as `KAPLUMBA_GPU_PROFILE=1` sub-steps
`sample/topk` and `sample/draw`, printed next to `head/argmax`.

---

## 3. The RNG — bit-exactly portable, with one deliberate workaround

`crates/engine/src/sampler.rs:142-150`:

```rust
x ^= x >> 12; x ^= x << 25; x ^= x >> 27;   // xorshift64
bits = x.wrapping_mul(0x2545_f491_4f6c_dd1d);
((bits >> 11) as f64) / (1u64 << 53) as f64  // f64 in [0,1)
```

then `let target = self.next_f64() as f32 * selected_total;` (`sampler.rs:131`).
**The `f64` is immediately demoted to `f32`**, so the double intermediate buys
nothing and there is no reason to touch fp64 on a GPU.

| step | on device | exact? |
|---|---|---|
| xorshift64 | `ulong`; `ulong` is used pervasively already (`exl3.metal:34-36`, `linear_repacked.metal:66`+) | ✅ bit-exact |
| `wrapping_mul` | `ulong` multiply wraps identically | ✅ |
| `f64 → f32` | **do not use `double`/`as_type<double>(ulong)`** | ⚠️ see below |

MSL's `as_type` table does not carry `ulong → double`, and fp64 on Apple silicon
is emulated. Instead produce the correctly-rounded `f32` with integer ops on
`n = bits >> 11` (53 bits), replicating round-to-nearest-even exactly:

```metal
// f32( (n as f64) / 2^53 )  ==  f32(n) * 2^-53, computed without fp64.
const uint hi  = uint(n >> 29);            // top 24 bits
const uint rem = uint(n) & 0x1FFFFFFFu;    // low 29 bits
const bool up  = (rem > 0x1000000u) || (rem == 0x1000000u && (hi & 1u) == 1u);
const float u  = float(hi + (up ? 1u : 0u)) * 0x1p-53f;
```

The division by 2^53 is exact in f64 and `float()` of a ≤25-bit integer is exact
in f32, so this reproduces the host's double-rounded result bit for bit.
**This is ~8 integer ops executed once per token by one thread — free, and it
removes the only dependency on Metal's 64-bit-to-float conversion semantics.**

### State in a buffer

`device ulong *rng` — one element, 8 bytes. Read and written by D2 thread 0 only.

Invariant that makes this safe, and which the tree already relies on elsewhere:
**exactly one D2 per generation step.** `run_head` is true only for the last
prefill chunk and for every decode step (`model.rs:4037`), so `encode_lm_head`
runs once per step. And command buffers submitted to one serial queue execute in
submission order — the existing `argmax_keys` buffer already depends on this,
since `decode_max_dispatches_per_buffer() = 50` (`encoder.rs:218`) can roll a
buffer *between* `argmax_slots` and `argmax_keys_to_token`. **The 2-deep pipeline
never has two sampling dispatches in flight.** This must be stated in the code,
and a test should assert that `run_head == true` occurs once per step (a future
edit that samples in every prefill chunk would silently advance the RNG 3× for a
3-chunk prompt).

Seed init (`sampler.rs:93`: `seed == 0 → 0x9e3779b97f4a7c15`) is done host-side
once per turn, one `contents()` write. No kernel.

---

## 4. The pipeline unlock

### `crates/engine/src/chat.rs:388`

```rust
let pipelined = self.greedy && std::env::var_os("KAPLUMBA_NO_PIPELINE").is_none();
```

With device sampling every temperature is pipelinable, so the gate becomes the
env check alone:

```rust
let pipelined = std::env::var_os("KAPLUMBA_NO_PIPELINE").is_none();
```

`self.greedy` is still needed by the fallback loop at `chat.rs:506`, so the field
stays. `chat.rs:418`'s `argmax_logits(&logits)` — the first token, the last use
of the 512 KiB row in this function — becomes
`self.model.read_host_token(0)`, because the prefill runs the pick at
`step_idx = 0`. The pipelined branch then never materialises `logits` at all.

### `crates/engine/src/generate.rs:150` and `:269`

**Do not add a third loop.** Parameterise the existing one:

```rust
fn generate_gpu_pick<F>(…, pick: HeadPick, …)     // was generate_gpu_greedy
```

with `enum HeadPick { Argmax, Sample(SamplerParams) }`, delete both
`temperature == 0.0` short-circuits, and route everything through it. This adds
no new loop body to keep in sync, and the greedy arm's body stays byte-for-byte
what it is today apart from receiving `pick` instead of assuming argmax.
(`chat.rs`'s own loop duplicates `generate.rs`'s shape by design — see the
comment at `chat.rs:393-397` about the first version differing and producing
wrong tokens — so it is edited in place, not shared.)

`:383` and `:216` — the `decode_tokens_per_second` bracket. Leave the existing
field alone and **add** `decode_wall_tok_s` computed over the whole loop body.
Redefining the existing field would silently move the programme's 16.05 tok/s
baseline; two fields keep the A/B honest.

### Hand-off contract after the fix

Per token the host reads **exactly one `u32`**:

- `host_tokens[step_idx]` — `StorageModeShared`, 8192 words (`model.rs:3034`),
  written by the pick inside the same command buffer as the step's forward,
  read by `read_host_token(step_idx)` (`model.rs:3825`) **after** `prev.wait()`.
- Nothing else. No `last_logits()`, no `Vec<u16>` of 262,144, no fp16→f32 widen,
  no host `argmax`.
- The invariant at `chat.rs:409` is unchanged and still the thing to protect:
  never read a GPU-staged token before waiting on the encoder that staged it.
  The sampler adds a second writer to `host_tokens` but does not relax the rule.

---

## 5. The `top_p` wrinkle — it is not a wrinkle

`top_p` needs sorted logits only if it is applied to the whole vocabulary. It
is not. `sampler.rs:110-128` does **top-k first**, truncating `order` to 64, and
**only then** walks `cumulative` over those 64. So the device must do the same,
and `top_p` is applied to 64 already-sorted values — free.

The general formulation to *avoid*: a cumulative-mass scan with an atomic
counter over 262,144. Do not build it. It is only needed if `top_k` is allowed
to exceed the device path's bound, and the answer there is to **fall back**:
guard `top_k > 1024` (and `top_k >= vocab`) to the host sampler with the
non-pipelined loop, rather than building a 262,144-wide sort. Ship the fast
path and keep the slow one as the escape hatch.

Note this is a *behavioural* constraint on the config, and it must be written
down: the device path is only distribution-identical to the host path when it
applies top-k before top-p, in that order, over the same 64.

---

## 6. Bit-exactness of the greedy path

The greedy path must not change at all, and it does not need to. The smallest
change that adds sampling without touching it:

- Introduce `enum HeadPick { Argmax, Sample(SamplerParams) }` and thread it
  `forward_chunk` ← `forward_step_pipelined` ← the decode loops. **Every
  existing caller passes `HeadPick::Argmax`.** That is a mechanical default
  argument, not a behaviour change.
- In `encode_lm_head` (`model.rs:4475-4476`) the `skip_softcap` predicate is
  re-derived from the mode instead of from buffer presence — see §"blocking
  risk". When `pick == Argmax`, the emitted code is identical to today's, so
  `--temperature 0` stays bit-identical, which is the programme's standing
  regression gate (`hypotheses.md:95`).

---

## 7. 🔴 The blocking risk

**`crates/engine/src/model.rs:4475-4476`:**

```rust
let skip_softcap = (next_token_out.is_some() || host_token_out.is_some())
    && !env_flag("KAPLUMBA_FORCE_SOFTCAP");
let argmax_src = if skip_softcap { &self.scratch.logits } else { /* softcap → */ &self.scratch.capped };
```

The moment the sampler reuses the pipelined route it passes `host_token_out`, so
`skip_softcap` becomes true and `argmax_src` becomes **`scratch.logits` — the
raw, un-softcapped logits.** `softcap` is `tanh(x/30)*30` (`elements.metal:425`),
which is monotonic (so the argmax is unaffected — that is why the greedy path can
skip it) but it **compresses the distribution**, so sampling the raw row draws
from a *different distribution* than the host sampler it replaces.

Nothing would catch it. Not tok/s. Not a smoke test. Not `--metrics`. And it is
the path of least resistance: the condition is already written, and it looks
correct because it *is* correct for argmax.

Mitigation, and it is not optional: make `HeadPick` the sole input to
`skip_softcap`, and pin it with a test that
(a) `KAPLUMBA_ABLATE=head_softcap` **changes** sampled output at
`temperature > 0`, and
(b) leaves `--temperature 0` bit-identical.
Until (a) is shown to fail when the sampler is pointed at `scratch.logits`,
there is no evidence the sampling path reads `capped`. Per AGENTS.md §3.4, that
claim is not a claim until the sabotage has been observed failing.

### Secondary risk (not blocking): `exp`

Metal's `exp(float)` is documented ≤2 ULP and is not the same function as
Rust's `f32::exp` (macOS libm `expf`). **Everything else in `sample()` is exactly
reproducible on device** — fp16→f32 is exact so the top-64 set and order are
exact; the sequential f32 sums are reproduced exactly by one thread in index
order; the RNG is exact per §3. Only `exp` is not. Consequence: for a fixed
`--seed`, GPU and host can pick a *neighbouring* candidate when the inverse-CDF
target lands within ~1e-7 relative of a cumulative boundary — order 1e-4 per
token, so a few percent of 200-token answers differ by one token, always between
two candidates inside the truncated top-p set with near-identical probability.
That is **not** quality loss under AGENTS.md §8 (which constrains speculative
decoding, not sampling), but it does mean "same `--seed` ⇒ same text" stops
being true across this change, and **no existing test asserts the
distribution.** Ship with a statistical test (total-variation distance between
host and device sample distributions on a fixed logit row, plus an *exact*
assertion that the top-64 set and order match) rather than an equality test.
Attempting to make `exp` bit-identical by porting FreeBSD's `s_expf` is not
worth it: Metal's FMA contraction would have to be defeated with `precise`
qualifiers throughout, and it buys nothing.

---

## 8. Detokenizer — **fix this first, it is worse than the sampler at length**

`crates/engine/src/tokenizer.rs:202-235`. `push` builds a **fresh**
`decode_stream` and replays **every id so far**, then diffs the whole string
against `emitted`. That is O(n) per push, O(n²) per generation.

Read the upstream source rather than guessing at the fix.
`tokenizers-0.21.4/src/tokenizer/mod.rs:1068-1105`, `step_decode_stream`:

```rust
ids.push(id);
let string = tokenizer.decode(ids.as_slice(), skip_special_tokens)?;
if string.len() > prefix.len() && !string.ends_with('�') {
    if !(string.starts_with(&*prefix)) { return Err(InvalidPrefix); }
    let new_text = &string[prefix.len()..].to_string();
    let new_prefix_index = ids.len() - *prefix_index;
    *ids = ids.drain(*prefix_index..).collect();     // <-- window stays tiny
    *prefix = tokenizer.decode(ids, skip_special_tokens)?;
    *prefix_index = new_prefix_index;
    Ok(Some(new_text.to_string()))
} else { Ok(None) }
```

The `ids.drain(prefix_index..)` is the whole answer: the library **already
maintains an O(1) window** and `step` is O(1) amortised. The engine has been
re-implementing, badly and quadratically, what `DecodeStream` was written to do
— `tokenizer.rs:169-184` (`stream_decode`) already uses it correctly, in the same
file.

**The fix**, ~15 lines, no new dependency:

```rust
pub struct Detokenizer<'a> {
    tokenizer: &'a GemmaTokenizer,
    stream: tokenizers::DecodeStream<'a>,   // built once, in new()
    ids: Vec<u32>,                          // kept only for finish()
    emitted: String,
}

pub fn push(&mut self, id: u32) -> Result<Option<String>, TokenizerError> {
    if self.tokenizer.is_media_id(id) { return Err(TokenizerError::MediaToken); }
    self.ids.push(id);
    match self.stream.step(id).map_err(|e| TokenizerError::Operation(e.to_string()))? {
        None => Ok(None),
        Some(chunk) => { self.emitted.push_str(&chunk); Ok(Some(chunk)) }
    }
}
```

`DecodeStream<'tok, …>` borrows the tokenizer; `Detokenizer<'a>` already holds
`&'a GemmaTokenizer` and `GemmaTokenizer::inner` is in the same module, so
`self.tokenizer.inner.decode_stream(true)` in `new()` typechecks. `finish()`
keeps using `decode_all(&self.ids, true)` — once per turn, O(n), fine.

**Behaviour change to test, not to paper over:** the current `else` branch
(`tokenizer.rs:231-234`) silently swallows a retract and returns
`Some(String::new())`. `step_decode_stream` returns
`Err(DecodeStreamError::InvalidPrefix)` for the same situation. So this change
turns a silent fallback into a hard error. That is the right direction — a
retract currently emits nothing and the next `push` diffs against a rewritten
`emitted`, which is a whole class of "the text is wrong and nothing says so" —
but it must be validated, not assumed: run `tests/multi_turn_chat.rs`, plus a
long-generation greedy text-equality test (3 bpw, `--temperature 0`, ≥3 runs,
**text only** per `hypotheses.md:95`), and deliberately force a retract
(a leading-space token, e.g. ids that decode `"_"` then `"▁x"`) to prove the
error path is reachable and correctly surfaced.

### Why it is first

Fitting P06's measured points (0.017 / 0.049 / 0.085 / 0.105 ms at n ≈ 20 / 60 /
100 / 119) gives a very clean law: **per-push ≈ 8.5e-4 · n ms**, i.e. α ≈ 8.5e-4,
consistent to ±4% across all four points. So total generation cost is
≈ α·N²/2 ms and the *last* push costs α·N.

| N generated | last push | amortised/token |
|---|---|---|
| 119 | 0.10 ms | 0.05 ms |
| 512 | 0.44 ms | 0.22 ms |
| **2048** | **1.74 ms** | **0.87 ms** |

At 2048 generated tokens the detokenizer's amortised cost is 0.87 ms/token
against the sampler's ~2.4 ms/token — call it 36% of the sampler. But its *final*
push costs 1.74 ms, and at 4096 tokens that push costs 3.5 ms, i.e. **it overtakes
the sampler**, and unlike the sampler it is unbounded. And critically it is in the
**pipelined greedy
loop too** (`chat.rs:443`, `generate.rs:1028`) — it sits between `wait()` and the
next submit, which is precisely the window the pipeline exists to fill. So this
caps long-context decode for *every* temperature, and it is 15 lines with no
arithmetic risk. Fix it first, and measure it at `--max-new-tokens 2048`, not at
119, or the fix will look worthless.

---

## 9. Recovery estimate

| item | ms/step @119 | ms/step @2000 |
|---|---|---|
| `Sampler::sample` 2.265 + widen 0.067 + 512 KiB readback 0.05–0.15 | 2.38–2.48 | 2.38–2.48 |
| `Detokenizer::push`, α·n | 0.10 | **1.70** |
| **recovered** | **2.5–2.6** | **4.1–4.2** |
| device cost added (2 dispatches, overlapped) | ~0.03–0.06 | ~0.03–0.06 |

62.3 ms → **59.7–59.8 ms = 16.7–16.8 tok/s** on short answers, matching P06's
ceiling of ~16.8. At 2000 generated tokens: 62.3 → **58.1 ms = 17.2 tok/s**.

**Honesty clause, and it matters more than the numbers:** the programme's
measured noise floor is 35% same-config spread and 3.7–6.4 tok/s run-to-run
(`hypotheses.md:13`, AGENTS.md §10.5). **A 2.5 ms gain on 62.3 ms is 4% — well
inside the noise.** Per `hypotheses.md:88-89` this may only be reported as a win
with ≥6 interleaved A/B pairs and a live same-config control arm. Anything less
and "the sampler no longer costs 2.4 ms" is a code-reading claim, not a
measurement. It is still worth doing — 2.4 ms of GPU idle per token is real and
it is the *default* temperature — but it will not visibly move the headline
number, and the write-up should say so rather than claiming 17 tok/s.

---

## 10. Tasks, in order

1. **Detokenizer incremental** (`tokenizer.rs:202-235`). No GPU needed; the
   `--max-new-tokens 2048` detok trace is the measurement. Sabotage: restore the
   replay and watch the trace grow quadratically.
2. `HeadPick` enum + `SkipSoftcap` from mode (`model.rs:4415-4489`), every
   existing caller `Argmax`. Sabotage: point `Sample` at `scratch.logits` and
   confirm the softcap-ablation test fails. **Do not proceed without this.**
3. `value_key`/`argmax_key` reuse check + `sample_topk_blocks` + `sample_draw`
   in `elements.metal` (insert after `argmax_keys_to_token`, `:610`),
   registered at `elements.rs:224-226` / `:321-323`.
4. Exactness test: device top-64 **set and order** == `top_k_order` on adversarial
   rows (heavy ties, ±0.0, NaN, extreme exponents). Exact equality, not tolerance.
5. RNG: `device ulong` state in a `ForwardScratch` buffer; assert the host's
   `next_f64() as f32` sequence is reproduced bit-for-bit by the MSL expression
   in §3 (unit test on the *integers*, no GPU needed).
6. Distribution test: TV distance host vs device over ≥10⁶ draws on one fixed
   row. **This replaces equality**, because §7's `exp` finding means equality is
   not available.
7. `chat.rs:388` + `generate.rs` pick parameterisation; `read_host_token(0)` for
   the first token.
8. `decode_wall_tok_s` added beside `decode_tokens_per_second` (`generate.rs:216`,
   `:383`). Do **not** redefine the existing field.
9. Interleaved A/B: temperature 1.0 pipelined-sampled vs today's 14.37 tok/s,
   ≥6 pairs, plus a same-config control arm. Greedy `--temperature 0` bit-identical
   over ≥3 runs, text only.
