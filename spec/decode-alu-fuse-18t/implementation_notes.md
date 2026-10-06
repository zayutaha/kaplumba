# implementation_notes — `spec/decode-alu-fuse-18t`

Running log per `AGENTS.md §5`. Appended as work proceeds, newest at the bottom.
**No implementation has started.** This entry is the verification pass that
produced the plan, and it is written down because four of its findings changed
the spec — three of them would have shipped a wrong change.

Branch: to be pinned in T0. HEAD at time of writing: `mtp-speculation` `517bc4e`.

---

## 0. Environment facts worth not rediscovering

| fact | value | how established |
|---|---|---|
| HEAD | `mtp-speculation` `517bc4e` | `git log -1` |
| `33e8ead` (the `alu_boost` instrument) | only on `agent/tensor-ops`; **not** in HEAD | `git branch -a --contains 33e8ead` |
| `master` | `fc36ee1`; `MEMORY.md` calls it broken | at HEAD, `pipeline_audit.rs` **and** `sweep.rs` both exist → that note is stale for HEAD, re-verify (T0d) |
| `integration/rolling` | `8549e6e`; does **not** contain `703603a` | `git merge-base --is-ancestor` |
| `alu_boost.rs` | absent from working tree, **never deleted** — simply not in this branch's history | `git log --diff-filter=D` empty; `--diff-filter=A` finds `33e8ead` |
| untracked in tree | `spec/alu-boost/`, `spec/decode-alu-fuse-18t/`, `spec/decode14-prefill100/` | `git status` |

## 1. The trap: `_u4` is not the lever, and its name says it looks like one

`spec/alu-boost/findings.md §2` reports `arm_decode_u4` = "same tile assignment,
same per-thread accumulation order, unrolled four deep" at `1.095×` / `1.052×`,
`0 differ / 2560 words`.

`grep -o 'arm_[a-z0-9_]*'` against `33e8ead:crates/engine/examples/alu_boost.rs`
confirms `arm_decode_u4` at lines `227` (source builder) and `247` (kernel), and
the harness prints its own `[check] arm_decode vs arm_decode_u4 over N words`
line at `:542`. **It is a tile-loop unroll at the same load width.**

A kernel already in the tree is named `linear_gemv_ishizuki_k3_colmajor_u4`
(`linear_repacked.metal:3193`). It is a **completely different experiment**:
threadgroup-staged `uint4`, tiles staged through `tg_tiles[SG][96]` with a
barrier inside the K loop. It measures **180.5 → 173.7 GB/s — a loss**, recorded
in four independent places:

* `crates/engine/tests/perf_probe_kloop_loads_in_flight.rs:137`
* `crates/engine/tests/probe_vecld_window_widening.rs:18` — "Already measured as
  `linear_gemv_ishizuki_k3_colmajor_u4`: **180.5 → 173.7 GB/s** on `gate_up`"
* `crates/engine/tests/probe_reduce_epilogue.rs:246` — `_u4` is the only entry in
  `BARRIER_IN_K_LOOP`
* `crates/engine/tests/probe_cache_hierarchy.rs:745-752` — **asserts** `_u4`
  still exists, is findable by name, and contains `uint4`, with the comment
  "`_u4` stages through threadgroup memory; that staging is the confound"

So the existing test suite would have caught the confusion only if someone
reused the *name*. Nothing stops a well-meaning port from "just enabling `_u4`",
which would be re-running a negative. The port is named **`_tl4`** (tile-loop 4)
to keep the distinction mechanical rather than a matter of memory.
`alu-boost §3` corroborates the physics: width is not the wall — `uint1` 238.5 vs
`uint4` 244.3 GB/s, `+2.5%` for a quarter of the load instructions.

## 2. The claim itself is currently unverifiable in this tree

`crates/engine/examples/alu_boost.rs` **does not exist** in the working tree. It
was added in `33e8ead` (781 lines, branch `agent/tensor-ops`) and never deleted —
HEAD's history simply does not contain that commit. `spec/alu-boost/` is present
as an untracked folder, so the findings were copied in and the instrument was not.

Per `AGENTS.md §7` ("verify a path exists before citing it"), the `+5–9%` is a
markdown claim with no runnable instrument behind it in this checkout. It becomes
a result again only when T1 restores the file (`git show
33e8ead:crates/engine/examples/alu_boost.rs`) and reproduces **both** the
bit-exactness check and a ratio above the noise floor.

Additional doubt recorded in Q5: `alu_boost`'s harness ceiling is `arm_seq` at
`217.1 GB/s`, which is **below** the shipped GEMV's `239.8`. A `+5–9%` measured
on a slower arm is not automatically a `+5–9%` on a faster one — that is the
`fused2_4col` lesson (`F4`: predicted win, measured `-0.7%` in 6 of 6 pairs,
after the confound was removed).

## 3. The `_mul1` "defect" is mis-sourced, and fixing it would have broken a
   working path

`WHERE-THIS-STANDS.md:62` states "No `_mul1` kernels exist (`linear.rs:485-489`)
— a **2.61× silent cliff** for any mul1 checkpoint, and exllamav3 1.0 made mul1
the default."

Checked:

* `linear.rs:479-489` is `ishizuki_rate_of_name` — it strips the
  `linear_gemv_ishizuki_k` prefix, parses the digits and rejects anything outside
  `1..=8`. Nothing to do with mul1.
* `decode_state_mul1` **exists**, `linear.metal:128`, used at `:146`.
* `Exl3Codebook::Mul1` is a live variant the dispatcher emits:
  `linear.rs:570` loops `[Exl3Codebook::Mcg, Exl3Codebook::Plain,
  Exl3Codebook::Mul1]`, and `:3656`/`:3730`/`:3770` handle the
  fractional-requires-mul1 rule.
* The `2.61×` figure is unreproduced anywhere in this tree.

So the "fix" would have been to write kernel code that already exists, and the
`2.61×` cliff remains unexplained. T9(b) is reproduce-or-record-as-unsubstantiated.

## 4. `scratch.attended` — real, and the line reference is stale

`model.rs:3229`: `attended: u16_elements(max_tokens, hidden)?` = 5376 halves.
Both attention epilogues write `q_heads * head_dim` into it — `attention.metal:974`
(combine, `output[head * head_dim + dimension]`) and `attention.metal:927`
(single-pass and prefill) — which is 8192 on the 50 sliding layers and 16384 on
the 10 full ones. `o_proj.in_features` is that same larger number
(`model.rs:525`), and `encode_input_transform` (`linear.metal:488`) reads all of
it.

**Overrun 22,016 B/layer/token**, correct output today only because Metal's
allocations are page-granular so it lands in the buffer's own padding.
`WHERE-THIS-STANDS.md:59-61` and the test's own header comment both cite
`model.rs:3012`; the allocation is at `:3229` — **stale by 17 lines**. Existing
test: `tests/perf_probe_activations_traffic.rs:300`.

Adjacent, and found while checking the above
(`perf_probe_activations_traffic.rs:60-70`): `scratch.normed` is a **dead store**
on the fused decode path — written by the input-norm epilogue, passed as `input`
to `encode_gemv_fused3`/`encode_gemv_fused2_mlp`, which are always called with
`pretransformed = true` so they bind `transformed` and skip the `input_pipeline`
dispatch entirely (`linear.rs:2716-2725`, `:2513-2522`). `scratch.proj` is never
written on decode at all (`model.rs:1983-1995` writes `raw`; the fused finalize
writes `hidden_b`). Both are `0.005%` of the stream — allocation waste, not
perf work, recorded so the next person does not re-find them.

## 5. R3 is much narrower than `spec/device-sampler/tasks.md` T3 reads

Verified present in the engine already:

* `model.rs:4004` `forward_step_pipelined(…, pick: HeadPick)` — the pipelined
  step already takes the head pick and writes into `host_tokens`.
* `model.rs:4057` `read_host_token`, `model.rs:4071`
  `device_sampler_supported(config)`.
* `sample_topk_blocks`/`sample_draw` built at `elements.rs:343-344`, encoded at
  `model.rs:4183`/`:4192`; kernels at `elements.metal:755`/`:851`.
* `scratch.sample_candidates` + `rng_state` already in `resident_bytes()`.

The gate that keeps temperature > 0 off this path is one line:

```rust
// crates/engine/src/chat.rs:388
let pipelined = self.greedy && std::env::var_os("KAPLUMBA_NO_PIPELINE").is_none();
```

Two tests **assert that exact source text** and are the real blast radius:

* `tests/perf_probe_host_ops.rs:373 temperature_above_zero_disables_the_pipelined_path`
  — `:377` greps for `let pipelined = self.greedy`; `:1099` is the same
  assertion inside `perf_probe_concurrency_dependency_graph.rs`.
* `tests/multi_turn_chat.rs:229`
  `pipelined_and_barrier_decode_agree_across_a_multi_turn_session` — compares
  both arms at **temperature 0** (`chat_session(&turns, 30, false)`), which is
  precisely the configuration that cannot discriminate this gate. It must be
  generalised off temperature 0 or it will keep passing for the wrong reason.

Both must be **rewritten to assert the new gate's meaning**, not deleted.

Also: `generate.rs:971` has a *different* gate, `let pipelined = !margins.enabled`.
Read it before touching it.

Cost of the gate today, for the A/B: `~3.2 ms` of a `69.6 ms` step
(`crux3 §7b`: `14.741`/`14.763` pipelined-greedy vs `8.025`/`14.208` host), and
`chat.rs:369`'s own A/B — `14.97` in chat vs `15.88` pipelined on the same binary
(`probe_budget_reconcile.rs:115-116`).

## 6. Correction to a line reference used earlier in conversation

`linear.metal:121` is `decode_state_mcg`, not a `colmajor` kernel. The kernels
that actually run on decode (`crux3 §2`, `KAPLUMBA_TRACE_GEMV=1`) are in
`linear_repacked.metal`: `_fused3_2col:1432`, `_fused2_2col:434`,
`_colmajor_2col_sk2:2429` (o_proj, down_proj), and `_colmajor_contig4:1992` —
which `linear.rs:2091` describes as "the kernel the engine actually reaches".

## 7. Ceiling — unchanged, and why nothing here claims to beat it

Unaffected by any correction above, because it is bytes and pins:

```
weights 12.0580 GB   (crux.md §1.2 + crux2.md §1, delta 0 B over 12 GB)
KV      2.1810 GB @16k  (50 sliding x1024 + 10 full x16384 fp16; forbidden AGENTS.md §8)
= 14.2390 GB/token @16k, 13.0646 @2k
wire 273 GB/s (pin rate) · sustained 245 · best arm 253.6 · real GEMV 239.8

20 tok/s @16k -> 284.8 GB/s = 104.3% of pins
22.5      @2k -> 294.0        = 107.7%
even 100% wire with zero overhead: 19.17 @16k, 20.9 @2k — never 22.5
```

`spec/decode-20tps-optimistic`'s ladder to `50.2 ms = 19.9 tok/s` omits KV
(`12.039/254 + 2.8` floor); adding `2.181 GB` gives `56.1 + 2.8 = 58.9 ms =
17.0 tok/s`. Same omission in `decode-19tok`'s disputed `250/11 = 22.5` row and in
the `11.834 GB` budget (wrong geometry — phantom `v_proj` on 10 layers).

**Therefore this spec's honest ceiling is `~17.2` bit-exact / `~17.6–18.6` with the
split waiver, not 20.** The user asked for `22.5` normal / `20 @16k`; that is
excluded by arithmetic, and the only remaining route past it is `tokens/byte`,
which was explicitly ruled out.
## T0 log (2026-10-06)

- Branch: `spec/decode-alu-fuse-18t-impl` off HEAD `a6ff214` (the spec commit;
  prior note said HEAD was `517bc4e` — stale by one commit).
- MEMORY.md "master is broken" claim: **stale for HEAD.** At HEAD
  `crates/kernels/src/lib.rs` declares both `pipeline_audit` (`:7`) and
  `sweep` (`:8`), and both files exist on disk. No E0433 at build time.
- `alu_boost.rs` restored from `33e8ead`, compiles
  (`cargo build --release -p kaplumba-engine --example alu_boost`), binary at
  `target/release/examples/alu_boost`. Committed `d53a6d7`.
- `scripts/gpuq status`: no model resident, 15.0 GiB reclaimable, 0 swap.
