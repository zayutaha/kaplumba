# Plan — logit-distribution probe

## Architecture and data flow

```
capture(model_dir, corpus, out.klrow)
  TextModel::load_from_dir            <- unchanged engine entry point, takes gpu_lock
  GemmaTokenizer::from_model_dir
  for each prompt p:
      ids = tokenizer.encode_prompt(p)
      model.reset_gpu_caches()
      for i in 0..ids.len()-1:
          model.forward_buffer_into_vec(&[ids[i]], &[i], &mut row)   <- existing decode step
          writer.push(RowMeta{p, i, ids[i], ids[i+1]}, row)
  drop(model)                                          <- frees ~12 GiB before the next arm
  drop(file)

compare(ref.klrow, cand.klrow)
  for each row pair, streaming:
      row metrics (logit_probe::compare_rows)
  aggregate -> ProbeReport
  also compute compare_rows(ref, ref) on row 0 as the self-check
```

Why **one decode step per position** rather than a prefill: prefill runs the
vocabulary head for the *last* row only (`model.rs:3026-3027` — "Prefill only
needs logits for the position that is actually sampled"), so a prefill yields one
distribution per prompt, not one per position. A single-token forward per
position is exactly what decode already does and costs exactly what decode costs,
so the probe has no special path.

**No engine readback hook is needed.** `forward_buffer_into_vec` already copies
all `vocab_size` fp16 values out of `scratch.capped` (`model.rs:3131-3137`); that
is the "512 KiB readback per token". The probe calls the public method.

## File touchpoints

| path | change |
|---|---|
| `crates/engine/src/logit_probe.rs` | **new.** Pure computation: capture file format (`CaptureWriter`/`CaptureReader`), `compare_rows`, `aggregate`, report formatting. No Metal, no model. |
| `crates/engine/src/lib.rs` | +1 line: `pub mod logit_probe;` |
| `crates/engine/examples/logit_probe.rs` | **new.** CLI: `capture`, `compare`, `perturb`, `selfcheck`. |
| `crates/engine/tests/logit_probe.rs` | **new.** Synthetic-metric tests + capture round-trip + ignored GPU tests. |
| `spec/logit-distribution-probe/*` | this folder. |

Nothing else in `crates/engine` or `crates/kernels` is touched. In particular the
decode path in `generate.rs` is untouched.

## Capture file format (`.klrow`)

Little-endian, fixed header, then a metadata block, then raw rows.

```
magic   [u8;8]  = b"KLPROBE\x01"
version u32     = 1
vocab   u32                (262144)
rows    u32
softcap f32                (30.0)
reserved u32               (0)
--- per row (rows of them) ---
prompt_index u32, position u32, fed_token u32, next_token u32
u16[vocab]                little-endian fp16 bit patterns, verbatim from scratch.capped
```

Raw fp16 bits, not f32: a capture is 512 KiB/row and 1000+ rows is half a
gigabyte; widening on write would double that for no information. The reader
compares bit patterns for the `bit_identical_rows` count and widens to f64 only
inside the metric.

## Metric definitions

With `r`, `c` the two rows as f64, `V = 262144`:

| name | definition |
|---|---|
| `argmax` | first index maximising the row, matching `generate.rs:416-424` (`>` not `>=`) |
| `m`, `lse` | `lse = max + ln Σ exp(x_i - max)`, computed in f64 |
| `KL(r‖c)` | `Σ p_i·r_i - lse_r - (Σ p_i·c_i - lse_c)`, `p = softmax(r)` |
| `KL(c‖r)` | the same with `q = softmax(c)` |
| `rank` | `1 + #{i : c_i > c_{argmax_r}}`; `ties = #{i≠argmax_r : c_i = c_{argmax_r}}` reported separately |
| `mean/max |Δlogit|` | over all `V` entries of the **softcapped** row (what the model actually emits) |
| `NLL` | `-log p(next_token)` / `-log q(next_token)`, next token from the corpus |

The KL form is deliberate: it never forms a ratio `p/q`, so it is stable for the
~200,000 entries whose `p` underflows f64 to 0. In exact arithmetic it equals
`CE(r,c) − H(r)`.

## Perturbation with a closed-form expected answer

**Patch `lm_head.svh`, the output-sign vector of the vocabulary head** — 262,144
f16 values, each ±1, read by `read_sign_component(weights, "lm_head", "svh", "sv", vocab)`
(`model.rs:4000-4015`). Negating entry *j* negates output logit *j*.

The readback is softcapped (`crates/kernels/src/elements.metal:425`,
`out = tanh(x/30)*30`), and `tanh` is **odd**, so a sign flip in the head is
exactly, not approximately, a negation in the captured row:

```
c_j == −r_j   (in fp16 bit patterns: cand_bits == ref_bits ^ 0x8000)
```

That gives the detection arm a hand-computable target: flip the sign of the
reference argmax at *every* position, and the probe must report 0% agreement,
`ref_argmax_rank_under_cand` whose distribution is computable from the capture,
and a KL that can be cross-checked by an independent brute-force pass over the
two rows. If the probe's KL and the brute-force KL disagree, the probe is wrong.

A second, *small* arm flips one token at a mid rank, to show the instrument
resolves differences far below the flip-rate that `output_identity.rs` would miss.

Implementation: `cp -c` (APFS clone, no extra bytes) the one shard that owns
`lm_head.svh`; symlink every other file. The shard is byte-identical apart from
the flipped signs, so safetensors headers and all validation stay intact.

## Sequencing

1. `logit_probe.rs` module + unit tests (metric math, file format). Pure CPU.
2. Sabotage each metric test, observe the failure, revert. (AGENTS.md §3.4)
3. Example binary: `capture`, `compare`, `perturb`.
4. `tests/logit_probe.rs`: round-trip + synthetic + ignored GPU arms.
5. Baseline runs, one model at a time under `scripts/gpuq`.
6. `~/Downloads` assessment.
7. Acceptance band.

## Risks and fallbacks

- **GPU contention.** Every model process goes through `scripts/gpuq try --` and
  the engine's own `gpu_lock`; exit 75 is retried around other work.
- **Worktree isolation.** `master` has uncommitted kernel work in flight that does
  not compile; this spec is developed in a worktree at `e9c2785` with its own
  `CARGO_TARGET_DIR`, per AGENTS.md §10.4 and §6.
- **`~/Downloads` un-loadable.** Expected. Documented as a finding with the exact
  validator that rejects it.
- **Capture time.** ~1150 rows at decode speed ≈ 90 s per arm plus load. Bounded
  by `--rows` if it turns out slower than measured.

## Requirement → task map

| requirement | task |
|---|---|
| R1, R2, R3 | T1 (`logit_probe.rs`), T3 (`capture`/`compare`) |
| R4 | T3 self-check + exit code |
| R5 | T2 corpus + metadata |
| R6 | T1 tests, T2 sabotage |
| R7 | T4 `perturb`, T6 detection arm |
| R8 | T5 baseline |
| R9 | T7 `~/Downloads` |
| acceptance band | T8 |
