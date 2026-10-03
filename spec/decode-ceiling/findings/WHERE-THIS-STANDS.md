# Where this stands

29 agents ran. One conclusion, reached independently by at least five of them
through different routes.

## The numbers

| | temperature 0 | temperature 1.0 (**the default**) |
|---|---|---|
| measured today | 16.05 tok/s | **14.37 tok/s** |
| what `--metrics` reports | 16.05 | 16.05 — **it overstates by ~1.7** |
| achievable | ~16.8 | **~15.1** |
| 20 tok/s | needs 236.7 GB/s + 660 Gcodes/s | same |

`--metrics` computes `decode_tokens_per_second` from a bracket that **excludes the
sampler** (`generate.rs:383`), and at temperature 1.0 the sampler is ~3.2 ms of a
69.6 ms step. **Every performance figure this repo reports is from the greedy path,
which is not the path the engine runs by default.**

## What is provably dead (do not spend GPU time here)

Occupancy · threadgroup size · layout/contiguity · TLB · split-K on gate/up ·
persistent kernels · 4col · NR0 · activation dtype · element-kernel fusion ·
multi-queue overlap · KV quantisation-by-layout · storage modes · prefetch ·
multi-word widening · per-code decode reduction

Each was closed by measurement or arithmetic, not by argument. The common reason:
**none of them touch the wall.** Removing *every* weight load from the live kernel
and shrinking its footprint from 2.6 GB to 129 KB changes throughput by **−0.8%**.
The memory system is already hidden.

## What the wall actually is

At 3 bpw the kernel is **integer-issue bound** at 9.375 instructions per code and
~92% of one-instruction-per-lane-per-cycle. At 4 and 6 bpw it is **DRAM bound** at
~250 GB/s. **No rate exceeds ~250 GB/s on this box.**

The largest reducible pool — `decode_state_mcg`'s 16 integer ops per 8 codes, 21%
of the loop — is **bit-locked**: the mask applies to the product, so it cannot fold
into the extraction or the constant. Narrowing the 16-bit window is quality loss.
Removing everything legally reachable (19% of the loop) reaches **18.6 tok/s**.

Root cause of the 2-load minimum: `decode_state_mcg` multiplies *before* masking, so
each lane's window spans **37 bits** — and 37 > 32, the register width. Two codes
per shift is arithmetically impossible.

## The ranked actions

| # | action | worth | risk |
|---|---|---|---|
| **1** | **Device-side top-k sampler**, and gate the pipelined loop on `greedy \|\| sampler_on_device` | **~+2 tok/s at temperature 1.0** — the path you actually run. Returns it to *parity* with greedy, not beyond. | low; argmax is already device-side |
| **2** | Explain the **1.6× gap** between the microbenchmark (203 GB/s) and production (125–150) for the *same kernel* | **~+30%** if closed — 10× larger than any instruction win. **Nobody has done this.** | unknown; that's the point |
| 3 | Trim the dead third ring word at `KA ≤ 4` | +0.3–1.4 tok/s; bit-exact over 1,048,576 codes | low |
| 4 | `KAPLUMBA_O_SK2_1COL` — 672 TGs with half the reduction traffic | +0.73 tok/s; kernel already exists, bit-exact | low |
| 5 | Incremental `Detokenizer` | ~0.1 ms, growing (quadratic) | trivial |

## Two real defects, independent of performance

1. **`scratch.attended` is undersized** (`model.rs:3012`) — attention writes
   `q_heads*head_dim` (8192/16384) into a 5376-element buffer. **22,016 B/layer/token
   overrun**, masked only by Metal's page padding.
2. **No `_mul1` kernels exist** (`linear.rs:485-489`) — a **2.61× silent cliff**
   for any mul1 checkpoint, and exllamav3 1.0 made mul1 the default.

## The decision that is yours

**D1** — accept ~16.8 / ~15.1, or move to fewer bytes? Byte reduction caps at
**16.44 tok/s** because the floor is 11.53 GB (−4.4%). The only route past that is
**2 bpw layers**, and it is a coin flip: 23.75 tok/s at 199 GB/s, but **14.89 —
slower than today** — at the amplification law's rate. Nothing in this tree measures
2 bpw.

## 🔴 CORRECTION — there IS headroom, and it was hidden by a corrupted data source

M05 (DRAM roofline) invalidates the premise several earlier conclusions rested on.

**The `1/bpw` law is refuted.** It demands `GB/s(6bpw) = 2 × GB/s(3bpw) = 375.4`.
Measured: **253.0 = 0.674× of prediction**, and two independent stream harnesses
agree on 253 to 0.1%. So 33% is not noise.

**The incoherence that proves it:** the *fastest* kernel in the tree in GB/s
(`lm_head`, 253.0) runs the code path **1.605× slower in codes/s** than the 3 bpw
layer kernel — 337 vs 501 Gcodes/s. **A per-code cost cannot produce that.** The
head is DRAM-bound; 253 GB/s is the wall wearing a different hat.

**The falsification experiment had already run.** Arm (d) — same instructions, same
addresses, working set **2.6 GB → 129 KB (20,155× less cold DRAM traffic)** —
changed time by **−0.6%**. A kernel bound at 187.7 against a 250.5 wall would have
had to save **25.1%**. Off by 42×. **Not bandwidth-bound.**

**Independent step-budget check**, not derived from any kernel timing:
`63.1 − 4.18 (head) − 1.52 (non-GEMV) = 57.40 ms` for 10.983 GB = **191.3 GB/s**,
within 2% of the lane-load law and **30% clear of the 125–150 GB/s** that
`decode-25tok-peak-bandwidth` §18.2 reports. **Those 125–150 figures are artefacts**
— three separate agents found that ablation deltas summing to 87.59 ms inside a
63.1 ms step cannot be additive. I quoted them.

**So: +33.5% of GB/s sits above the layer's 187.7, and it is ALU work** —
`decode_state_mcg`'s 64-bit barrel shift, 65.6% of the kernel, already measured at
**3.67×** on the isolated kernel.

### But the ALU pools are locked, which is why 8 agents found nothing

Three independent findings, now consistent rather than contradictory:

| agent | finding |
|---|---|
| P18 | crossover from ALU-bound to DRAM-bound falls at **3.77 bpw** — between the two measured rates. So **3 bpw is ALU-bound, 4–6 bpw is DRAM-bound.** |
| P01 | 9.375 instr/code at ~92% of one-instruction-per-lane-per-cycle |
| P18/P11 | **removing decode arithmetic makes the kernel SLOWER** (163.0 → 134.3). The decode's dependent chains were accidentally supplying ILP. |

So the headroom is real but sits behind a **bit-locked** pool (the mask applies to
the product) and an **empirically counterproductive** one. That is the honest
reason no instruction-level proposal survived review, not that there is nothing
left.

### The honest position, revised

| | tok/s | condition |
|---|---|---|
| today, measured | **16.05** | — |
| **19** | **reachable in principle** — but **only with the non-GEMV tax at its 2.90 ms floor**. At today's 6.70 ms it is 18.26 | conditional, and testable |
| 20+ | not reachable | would need the ALU fix *and* the non-GEMV floor *and* byte reduction |

I previously called 20 tok/s flatly unreachable. That was too strong: it leaned on
§18.2's 125–150 GB/s figures, which are artefacts. **19 tok/s is on the table,
conditional on two things neither of which is free.**

### The cheapest decisive next step

`probe_dram_roofline.rs` ships `code_rate_vs_bandwidth_discriminator` (`#[ignore]`d)
with a **pre-registered falsification criterion written before the run**: if it
returns code-rate-bound, M05's model is wrong and every tok/s figure above is wrong.

**It needs ~2 GB and ~12 s of GPU.** That converts the central question of this
project from an argument into a measurement, and it is cheaper than any of the
implementation work above.

**Caveats M05 flagged rather than smoothed:** `L_LANE` is one parameter fitted on
two points; the 6 bpw anchor is *identical* to the copy ceiling (250.5 to 4 s.f.),
so its "0.1% error" is interpolation, not validation; and the model's own "today"
is 14.41 against 16.05 measured, so its absolute calibration is suspect even where
its ratios may hold.

## 🔴 A 4-BIT BUG I MISSED — `lm_head.trellis` sits at an ODD byte offset

Measured from the safetensors headers, not inferred:

| checkpoint | trellis tensors | offset %4 ≠ 0 | **odd byte offset** | `lm_head.trellis` |
|---|---|---|---|---|
| `Gxma-repacked` (3 bpw, what runs) | 411 | **0** | **0** | 1,026,108,224 (%4=0) |
| `Gxma-4-bit` | 411 | 291 | **172** | **507,672,341 (ODD)** |

`lm_head` is the **one tensor that always bypasses the arena**
(`offload_head_trellis`, `weights.rs:585`), so its offset reaches the kernel raw.
At `P % 2 == 1` **even the existing `ushort` loads in `trellis_word` are
misaligned** — MSL requires 2 B alignment for a `ushort` read. 172 of 411 tensors
in the 4-bit checkpoint are in that state, and the head is one of them.

This is a *candidate root cause* for the 4-bit command-buffer fault that survived
every other hypothesis: a misaligned `ushort` dereference in the hot loop, on the
one buffer that skips the arena copy that would have re-aligned it. It was
previously invisible because at 3 bpw the arena makes `packed_offset == 0`.

**Cheapest possible test:** `Gxma-4-bit` with `KAPLUMBA_RESIDENT_WEIGHTS=1` needs
18.05 GB and is refused by the 18.0 GB limit — but raising
`KAPLUMBA_RESIDENT_LIMIT` forces the arena path, which is the A/B. If the fault
disappears with the head in the arena, that is it.

## K05: vector loads — a closed axis, with a correction to my own queue

**No `uint4`/`ulong2` widening is possible.** `j0` is lane-dependent, and the
histogram over lanes at KA=3 is `0:8 1:8 2:8 3:8` — no residue class covers more than
half the lanes, so **no lane-uniform branch exists** and a divergent predicate
executes both arms anyway. `uint4` is legal for **0 of 32 lanes** at KA=3 and
**1 of 32** at KA=6. The legal variant trades 1 load for ~12 ALU on a kernel already
at ~63% ALU.

**The `ushort2` "free 2×" I proposed does not exist.** `trellis_word` reads bytes
`P+4i` and `P+4i+2`; a `ushort2` is 4 B and needs `P % 4 == 0` — which is *exactly*
the case that already takes the single 32-bit `ring_word<true>` load. There is no
configuration where the unaligned path gets a free 2×.

**Threadgroup staging is arithmetically a loss**: 6 wide loads + 64 `LDS.32` = **70
instructions against today's 64**, plus two barriers per tile. The measured version
of this exact kernel (`_colmajor_u4`) is **180.5 → 173.7 GB/s**.

**And my #3 action is worthless — I was wrong.** The `w2` trim removes 3 loads →
2 per lane on the 3-word path, but **the live 3 bpw path is already 2 words/lane**.
`w2` is read by **0 of 32 lanes at KA ∈ {1,2,3,4}** and 16–32 of 32 at KA ≥ 5. The
guard is false at the head (6 bpw), so the only live 3-word path does not benefit.
**Expected: 0.00 tok/s.** The tree had already taken this reduction on the ishizuki
arm — `k1..k4_w2` exist, `k5..k8_w2` do not, and two independent derivations agree
on {1,2,3,4}.

## Revised queue

| # | action | worth | note |
|---|---|---|---|
| 1 | **Device-side top-k sampler** | **~+2 tok/s at temperature 1.0** | the path you actually run |
| 2 | Widen the two 1-threadgroup finalize grids | +0.14 tok/s | `lib.rs:27-33`: 0.7 µs many-TG vs 4.6 µs one-TG |
| 3 | `KAPLUMBA_O_SK2_1COL` | +0.73 tok/s | kernel exists, bit-exact |
| 4 | `fused2_co` 4→8 loads in flight | 0 to +0.5, likely noise | the one kernel lagging its siblings |
| ~~5~~ | ~~trim dead `w2`~~ | **0.00** | already 2 words/lane at 3 bpw |

Ceiling at the 203 GB/s measured peak: **17.13 tok/s** (16.84 with the corrected
byte count). Past that requires beating 203 GB/s at 3 bpw, which is ALU work on a
bit-locked pool.

## 🔴 The "~1 tok/s colmajor gain" is UNVERIFIED

I have been citing repacking as worth ~1 tok/s (14 → 15.7). **It has never been
measured end to end.**

The pair is textbook — same shards, same names, same dtype, +0.002% bytes, only
two leading axes transposed:

```
Gxma/Gxma       layers.0.q_proj.trellis  [336, 512, 48]   lm_head [336,16384,96]
Gxma-repacked   layers.0.q_proj.trellis  [512, 336, 48]   lm_head [336,16384,96]
```

Yet:
- the kernel-level figures are **single readings, `min`-of-5 batches of 20, one
  process, no interleaving** — and measured against `_colmajor`, which a sibling
  probe asserts is **not** the kernel the engine dispatches today;
- `crates/engine/examples/bench_colmajor.rs:45` **builds the row-major buffer and
  discards it** (`packed_orig` appears exactly twice: once built, once consumed as
  `reorder_tensor_to_colmajor`'s input). **The left-hand column of that arrow is
  unreproducible.**
- **no spec records a decode rate for a row-major checkpoint at any HEAD.** The
  last one (~11.2 tok/s) predates the repacker *and* every kernel change since.

So "colmajor is the fast path" is currently an **assertion that nothing in the
tree can falsify** — which is why three agents independently declined to trust
microbenchmark rankings for kernel selection.

The fix is ~15 lines in an untracked example: interleave row-major
`linear_gemv_ishizuki_k3_w2` against colmajor `_k3_colmajor_2col`, A/B/A/B, same
shapes, medians + spread. No model load needed.

## Two more stale docs, both load-bearing

- **`model.rs:309-311`** describes `permutation_mode` as "the candidate fragment
  shuffles that the coherence probe searches". **Nothing searches them.** Every Rust
  comparison is against `1` (15 in `model.rs`, 7 in `linear.rs`). It is a **boolean**:
  "trellis stored column-major".
- **The coherence probe cannot find a faster mode even if it ran.** It scores on
  *knowledge* — does mode *m* still say `" Paris"` — and has no clock. 0 of 10 modes
  were correct. There is no per-mode timing table in the repo and there could not be.
- **Two of the "ten candidate modes" are not permutations at all**: mode 2 maps 256
  logical states onto **160** (96 pairs aliased, 96 never read); mode 8 collapses
  to `row * 17`, so all 16 columns of a row alias to one word. "Swept 10 modes" was
  eight real shuffles plus two degenerate maps.

## K06 latent hazard (untriggerable today)

The two `perm_mode == 1` arms at `linear.rs:2108` / `:2133` sit **above** the
selector and `return` before it is called — so a colmajor 3 bpw tensor never
consults the alignment gate at `linear.rs:1516`. Those arms bind `packed_offset` and
the colmajor kernels cast to `(const device uint *)packed`, which a `%4 != 0` offset
invalidates; `ring_word<ALIGNED>` exists for exactly this and is not used there.
410/411 tensors on this checkpoint are 4-byte aligned. Recorded so it is not
rediscovered as a mystery.

## M02: host DRAM interference — clean negative, with the number

**30,324 B/token of host traffic against the GPU's 12.04 GB/token — 1 : 397,045.**
If the host's entire measured CPU time (0.078 ms) were DRAM-limited it would need
15.0 MB; the tables account for 30.3 KB. **This angle's attack surface is empty.**

The *entire* host-visible data path on decode is **12 B/token**: three `u32`
(token id in, position in, argmax out). `forward_chunk`'s host staging
(`model.rs:4073`) is gated on `tokens > 1`, so decode stages 12 bytes by hand.

Host *allocation* traffic — what the CPU asks the allocator for — is **30.3 KB/token**
reachable (600 stamp labels at 36 B + 121 kernel names at 72 B), measured with a
thread-local counting allocator. Total host CPU: **0.078 ms = 0.12% of a 62.3 ms step**,
and even that overlaps GPU streaming, since decode commits step N+1 before waiting on N.

Two premises in the brief were false and worth recording:
- **Sign vectors are not read per token.** `encode_linear` reads only `*_signs_buffer`;
  every non-`_buffer` read of `input_signs` is in `run_linear`/`forward`, both CPU-fallback
  paths. There is no 19.7 MB/token host stream to remove. The retained host `Vec<u16>`
  is **15.99 MB** of *capacity*, which frees 15.99 MB and buys **zero** tok/s.
- **`SPIN_THREADS` (4096) is not a CPU spin.** It is the GPU threadgroup width of the
  `spin_burn` kernel, gated behind an unset `KAPLUMBA_DECODE_INJECT`, on a 16 KiB Private
  buffer. **No CPU thread polls memory during decode.**

The 5.75 ms "exposed encode" in `decode-19tok-repacked-250gb/findings/04` is **546
Objective-C `dispatch_*` encodes** at ~10 µs of message-send and command-stream
construction each. That is P02, not bandwidth.

The one change (`model.rs:825`, hoist the `enabled` check above the label construction,
so the `format!` is not evaluated when profiling is off) is worth **+0.010 tok/s** —
three orders of magnitude below the noise floor. Mechanical, zero-risk, and it grows
with every profiling boundary added, so fold it into whatever lands next; do not spend
an A/B slot on it. Its sibling is a genuine bug: `kernels/src/lib.rs:38`'s `ablated()`
only populates its cache when `KAPLUMBA_ABLATE` is non-empty, so in production the
`std::env::var` runs on all **361 calls/token** under the mutex — measured **75 ns vs
10 ns**, a 7.5x defect, one-line fix.

## ✅ 4 BPW WORKS — root-caused, fixed, and byte-identical to 3 bpw

**Defect A — misaligned EXL3 packed bind.** `lm_head.trellis` bound at byte offset
**1813, odd**. safetensors `data_offsets` are relative to a payload starting at
`8 + header_len`, and nothing aligns that. `MTL_DEBUG_LAYER=1` named it:

> `validateComputeFunctionArguments: failed assertion 'Compute Function
> (linear_gemv_exl3_k8_u_mcg): the offset into the buffer packed that is bound at
> Buffer index 0 must be a multiple of 2 but was set to 1813.'`

Two consequences: Metal rejects the bind, and **past validation it is an undefined
read**. And because `1813 % 4 != 0`, `ishizuki_gemv_pipeline` (`linear.rs:1516`)
*refuses* `linear_gemv_ishizuki_k8` — so the head silently ran a **different
decoder over the same bytes**. Measured: first-token argmax margin **11.42 → 0.53**,
mean |logit| **12.29 → 7.07**. Garbage that looked like text.

**The odd-offset lead recorded here from `probe_vecld_window_widening.rs` was
correct.** My framing of it was slightly off: at HEAD only `lm_head` is mapped
(410 layer tensors are arena-resident at offset 0), so exactly *one* tensor was
misaligned — not 172.

**Defect B — per-command-buffer dispatch ceiling.** The 60-layer prefill encodes
**2149 dispatches** against `set_max_dispatches(2500)`, so it never rolled.
Measured exactly: **2144 dispatches/buffer passes, 2145 faults.** Property of the
buffer, not the work — identical at 14 and 86 tokens, and identical with all 18 GB
anonymous. Fixed with `PREFILL_MAX_DISPATCHES_PER_BUFFER = 1024`.

**The measurement trap that made the whole 4-bit hunt go wrong in both directions:**
`commit_and_wait` waited on rolled buffers but **status-checked only the last one**.
So lowering the roll made 4 bpw stop erroring *and kept emitting garbage*
(`edL laly own own owning own-CCCCCC`) at every roll value. "No longer crashing"
was not evidence of a fix.

**Why this was never caught by tests:** out-of-bounds device **reads and writes are
both silently tolerated on AGX** (measured to 2³⁰ elements past the end). A
misalignment does not crash — it returns wrong data. Only `MTL_DEBUG_LAYER=1` could
name it. And **OOB writes being tolerated means the undersized `scratch.attended`
is live corruption, not a near-miss.**

### Results

| | 3 bpw | 4 bpw |
|---|---|---|
| output | `Hello! How can I help you today?` | **byte-identical** |
| decode | 16.06 / 16.02 tok/s | 0.014–0.066 tok/s |
| prefill | 10.52 / 10.59 tok/s | 0.18–0.21 tok/s |
| peak footprint | — | **17.44 GB (16.25 GiB)**, **0 swaps** |

3 bpw **byte-identical across 3+ runs** — no regression.

### Not fixed: 4 bpw is ~1000× slower, and that is residency

All-resident helps 50x (0.014 → 0.51–0.90 tok/s) and is still far short. 4 bpw
layer trellis is ~13.6 GiB against an 18 GB arena limit on a 24 GiB box with an
18.24 GiB wire limit. **The outstanding lever is F1 (`mlock`), identified in
`spec/memory-swap-cliff` and never implemented — `grep mlock crates/` returns 0
hits.** M03 independently reached the same conclusion: map the layer trellis *and*
pin it. Effect on bandwidth is **zero**; what it buys is that today's working set
is *all or nothing* — fully resident, or the 1.4 tok/s collapse.

### The `commit_and_wait` fix has no automated coverage

Honest gap, recorded as such: a `status=Error` **cannot be provoked from a test** on
this hardware. Claiming coverage would be false; it was verified by A/B instead.

## ⚠️ The shared `target/` was rebuilt twice — rebuild before benchmarking

Agent A ran `cargo build --release` in the shared `target/` before switching to an
isolated `CARGO_TARGET_DIR`, so `target/release/kaplumba` was overwritten twice
during another agent's measurement window. **Any tok/s measured before the last
rebuild is stale.** This is the AGENTS.md §10.2.4 failure mode arriving from the
other direction.

## 🔴 K15: two env flags confound A/B claims already recorded in this tree

These are not new findings about speed — they are **bugs in the measurement
harness**, and they mean two existing conclusions cannot be attributed.

**(a) `KAPLUMBA_SK4` silently disables cross-layer fusion.** `model.rs:1284` puts
`!env_flag("KAPLUMBA_SK4")` **inside** `cross_layer_fusion_applies`. So A/B-ing
split-K=4 also deletes T3's dispatch fusion. The recorded "**Split-K=4 is
identical to Split-K=2**" cannot be attributed to split-K — it is a two-variable
change, and split-K is the variable least likely to be responsible.

**(b) `KAPLUMBA_NO_SK2` hits `o_proj` *and* `down_proj`.** That is why its −0.58 is
unattributable. The isolating flags `O_NOSK` / `D_NOSK` exist for exactly this
(the comment at `model.rs:1918` says so) and **no isolated result is recorded
anywhere in the repo.** Their arithmetic is ≈ +0.08 tok/s for `O_NOSK` — below the
noise floor, so this settles `o_proj` without a spec change if anyone wants it.

### Good news: no best-known configuration is hidden behind an unset flag

Every measured-best cell is **already the default**: SK2 fires by default (336 TGs),
`fused3_2col` ships at 512 TGs and ~194 GB/s against a 203 ceiling. All three flags
that *would* change a default (`FUSED2_4COL`, `FUSED2_CO4`, `SK4`) gate
configurations that measured equal or worse. **No gated treasure.**

## Fusion buys no traffic — and `_co`/`_co4` are confirmed dead

| | weight bytes | `x` bytes | `x` as share |
|---|---|---|---|
| gate+up `fused2_2col` (**shipped**) | 86,704,128 | 10,752 | **0.0124%** |

All five fused variants read **every weight byte exactly once**. `_co` reads `x`
*exactly as often as* `_2col` — both launch 1344 TGs and walk the whole activation
in each. `_co` halves `x` reads only against plain `fused2` (2688 TGs), which
`_2col` already achieves identically. So the entire premise of "co-computation
shares activation reads" saves **nothing on the shipped path**.

`_co` is unreachable without `KAPLUMBA_NO_FUSED2_2COL=1`; `_co4` needs that plus
`KAPLUMBA_FUSED2_CO4=1`. **T1 is not superseded — it is *shadowed***: `_2col`
reaches the identical 1344-TG grid and identical 64,512 B/threadgroup, capturing all
of T1's occupancy claim.

**A genuine inconsistency:** `encode_gemv_fused2_mlp` tests `_2col` first, but
`encode_gemv_fused2` (q+k) tests `_co` first. **The two entry points disagree on
selection order.** Harmless today only because q ≠ k on both layer kinds — which is
now asserted, so it cannot drift silently.

### Second independent derivation of the 12.039 GB budget

K15 arrived at **12.039 GB/token** independently of the fusion census, from the same
defect: §16.4 prices all 60 layers at the *sliding* geometry, but 10 are
full-attention (head_dim 512, 4 KV heads). Same correction, same numbers:

| | wrong (§16.4) | corrected |
|---|---|---|
| bytes/token | 11.834 GB | **12.039 GB** |
| 20 tok/s needs | 236.7 GB/s | **240.8 GB/s** |
| ceiling at 203 GB/s | 17.2 tok/s | **16.9 tok/s** |

Two agents, two derivations, one number. The wire floor is 44.1 ms / 22.68 tok/s.

## The one zero-code experiment worth naming (and why I did not run it)

**`KAPLUMBA_FUSED2_4COL=1` reaches a real, never-executed kernel** — 672 TGs,
129,024 B/threadgroup. No code needed. It is the single cell that **discriminates
the two competing explanations** for `co4`'s −0.19 tok/s: identical grid and
bytes-per-threadgroup, but four *consecutive* tiles of **one** matrix instead of two
interleaved streams. §16.3 measured 199.3 GB/s at 129,024 B/tg against ~175 at
64,512, so if bytes-per-threadgroup is the driver it is worth **+0.57 tok/s**; if
two-streams-per-SM is the driver it reproduces `co4`'s loss. Honest range
**−0.2 … +0.65**, median guess +0.3.

**I did not run it, and the reason is the power analysis, not caution about the
change.** +0.3 on 16.05 is **1.9%**. The last agent's table says a 1.9% effect at
this box's recorded σ needs **more than 179 pairs ≈ 6.5 hours**. Five pairs would
produce a coin flip with a confident-looking number attached — which is exactly how
three false fixes happened in this repo. The experiment is worth running *after* the
box is fixed, or on a quieter machine, not before.

## 🔴 K04: the reduction is 0.66% — and it kills queue item #4

**The cross-lane reduction is 0.664% of GEMV issue slots** (mean 27.4 instructions
per threadgroup against 21 unrolled K iterations). Deleting it entirely — which is
unreachable, you can only move it — is **+0.104 tok/s**. Cross-checked against the
tree's own measured 0.226 µs/barrier: **+0.05**. Bracket **0.34–0.66% of a token**.

`red[2][8][16]` is **1024 B, 16× under** the occupancy cliff. All four alternatives
are strictly worse: 4col is measured dead, a wider shuffle tree goes 0.854% → 1.34%,
dropping threadgroup memory is **2.4× worse** (which is *why* the code stops at
`lane % 4`), and prefix-sum emits 32× the adds.

### ⚠️ Struck from the queue: the two-pass finalize widening is NOT bit-exact

I listed "widen the two 1-threadgroup finalize grids, +0.14, low risk" as item #4.
**It is barred.**

The frozen order is the descending butterfly on masks 128…1. In a butterfly mask
`2^j` maps `2^(j+1)` nodes to `2^j`, so node `r` covers leaves `{r + k·2^j}`. The
exact split is **G groups, G | 32** — and **42 ∤ 32**. Both plausible-looking
alternatives were tried and are wrong by **1 ULP**: contiguous ranges are *not*
subtrees, and strided leaves recombined ascending are not either. So the T5.6 plan
changes **5376/5376 elements** — forbidden by AGENTS.md §8.

It is also worth less than claimed: both finalizes move ~118/129 KB of device
traffic **through one core** = 14.8 MB/token, which is 0.30–0.74 ms; free
parallelisation is 0.074 ms. **Upper bound +0.24, midpoint +0.13, after overheads
+0.09** — for a two-pass rewrite of the two most bit-exactness-critical kernels in
the tree. `ROUND1-2-SUMMARY.md`'s "the only fusion item with a ms-scale number" is
what made it look worth doing.

**One bit-exact reduction win does exist:** masks 128/64/32 are the only
cross-simdgroup steps, so **10 barriers → 3 barriers + 5 `simd_shuffle_xor`**,
proved bit-identical. 840 barriers/token ≈ 0.19 ms ≈ **+0.05 tok/s**.

## 🐛 Filed defect: `linear.metal:1187` still strides `b += 8u`

Under **32** simdgroups, that is **6 rounds where its siblings do 2 — 24 of 32
groups idle**. Its siblings at `:1117`, `:921`, `:969` all use `32u`. A missed
T5.6 edit. One token, bit-exact by construction (pure elementwise map, disjoint
slots, no cross-block reduction). Not applied — AGENTS.md §2 requires a spec first.

## ⚠️ A vacuity bug worth remembering

K04's own sabotage log records that **its first test input was f16 *denormals***,
on which the ascending-mask-order sabotage did not fire — meaning **every
"exactness" assertion in the file was vacuous**. Caught by execution, fixed, plus a
permanent guard. This is the third time in this investigation that a probe's
inputs were chosen such that the detector could not fail (the others: a 64-bit
probe word feeding a 32-bit extraction, and a literal `0x1234ABCD` whose bit 25
happened to be clear).

**The generalisable lesson: a bit-exactness test proves nothing until you have
seen it fail on a deliberately reordered implementation at inputs you did not
choose for convenience.**

## 🔴 The discriminator ran. It could not decide, and that is a finding about the *probe*

`code_rate_vs_bandwidth_discriminator` (`probe_dram_roofline.rs`, ~3 s of GPU,
2 GB) was written with a pre-registered falsification criterion: if it returned
code-rate-bound, M05's whole model was wrong. **I ran it twice, changing one
constant, and it printed both verdicts.**

| `ALU_ITERS_PER_TILE` | ops/code | probe 3 bpw | printed |
|---|---|---|---|
| 48 (as written) | 0.56 | 218.3 GB/s | **BANDWIDTH BOUND** — "R2 refuted" |
| 800 (recalibrated to the kernel's 9.375 instr/code) | 9.375 | 74.3 GB/s | **CODE-RATE BOUND** — "R2 confirmed" |

The real kernel is **203 GB/s** — between the two. **A test whose verdict flips on
a calibration constant is not a discriminator.** I changed it to print
`VERDICT: WITHHELD` with the reason, because a confidently wrong verdict here
would have been the single most expensive error in this investigation: M05 built
the "there is +33% headroom" claim on it, and I built the "20 tok/s needs 660
Gcodes/s" ceiling on the same axis.

**The cause is structural, not a bad constant.** The probe's ALU body is
`acc.y += uint(acc.x >> (uint(acc.w) & 63u))` — every iteration depends on the
previous, so it is a **latency chain** at ~3–4 cycles/iter. The real kernel's
9.375 instructions are mostly **independent** mul/xor/FMA at ~1/cycle. The probe
converts instruction count into latency, and **instruction count is not time.**
No constant fixes that, because the probe has no ILP to model.

**So the "is decode waiting on memory or on issue" question is still open**, and
the only model here fitted to real timings is the ten-shape additive fit from
P02's report: `t = bytes/BW + c` → **BW = 190.5 GB/s, c = 9.03 µs, 2.2% over ten
measured shapes**. A 9 µs constant against a 190 GB/s term says **both are
partly true** — which is the honest answer, and is consistent with every other
agent's finding that no single knob moves the step.

**Method note for the next person:** a pre-registered criterion is worth nothing
if the instrument can produce every outcome. Registering "exactly one of these
three should hold" was correct; failing to verify the instrument *could* fail
was the gap. AGENTS.md §3.4 asks for a test that fails when the code is broken —
nobody asked whether the *measurement* could fail.

## 🔴 The byte-reduction route (R1) is closed — P19 REFUTED

Decode GEMV time is set by **how many 32-bit ring words the matvec reads per lane
per tile — `lane_words(rate)`, not `rate`.** Since `bytes = tiles·8·rate`, the
ratio `bytes/rate` **cancels exactly**. Lowering bit-width moves bytes but not time.

Exhaustive search over **all 8⁶ = 262,144 rate assignments**: the 410 layer
tensors are a **plateau over 1..=4 bpw** — at a 6 bpw head, layers at 1, 2, 3 or
4 bpw all cost **exactly 62.3 ms**. Best case in the entire integer-rate space:
**16.55 tok/s**, 10.4 ms short of target. Cross-checked against the repo's own
rate sweep: model predicts 1.000× for 4-vs-3 (measured 1.068) and 1.500× for
6-vs-3 (measured 1.586).

**The trap worth naming**, because the arithmetic invites it:
`9.45 GB ÷ 203 GB/s = 46.6 ms = 21.5 tok/s` looks like a win and **is invalid** —
203 GB/s is a *3-bpw* bandwidth. A 2-bpw MLP reads the same lane-load count, so its
time is unchanged and its effective bandwidth falls **193 → 152 GB/s**.

| scheme | GB/tok | ms | tok/s |
|---|---|---|---|
| uniform 3 bpw | 12.055 | 62.30 | 16.05 |
| MLP 3→2, all 60 layers | 9.454 | 62.30 | **16.05** |
| attention 3→2 | 10.996 | 62.30 | **16.05** |
| head 6→4 only | 11.703 | 60.41 | **16.55** |

Every quality-safe choice is worth **exactly zero milliseconds**. The one scheme
with a measured payoff is dropping the head's precision — and head errors land
directly on the argmax.

**Two corrections to intuition:** full-attention layers are the *largest*
(200.5 vs 179.9 MB — q and o are 16384-wide, only k shrinks), and MLP is 64.8% of
bytes, so the low-risk region *is* the high-yield region — which makes it painful
that it pays nothing.

**And a real capability finding:** `quantization_config.json` carries
`bits_per_weight` on **all 411 modules individually**, and `lm_head`'s already
differs from the global (6 vs 3) — a copied global could not. `config.rs:463-470`
already range-checks each. **The format is already mixed-precision-capable**;
uniformity is imposed by exactly three loader sites (`safetensors.rs:577-588`,
`:942`, `model.rs:3357`). But it is still the wrong lever: `linear_repacked.metal`
has colmajor kernels for **rate 3 only**, so a 2-bpw layer falls through to the
row-major `_w2` kernel — finite, plausible, wrong, 1.5× slower.

**D2** — anything on the bytes axis trades quality, which AGENTS.md §8 forbids.
That is your call, not an engineering one.

**My recommendation: do #1 and fix the two defects. Accept ~16.8.** The 1.6×
microbenchmark gap (#2) is the only remaining thing that could change the picture,
and it is a *question*, not a task — nobody has explained it.