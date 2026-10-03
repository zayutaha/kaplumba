# 4 bpw is ~1000x slower than 3 bpw — short-run diagnosis

**Date:** 2026-10-03. **Agent:** short-run probe (6 substantive model runs, all ≤ 100 s wall).
**Binary:** `target/release/kaplumba`, `cargo build --release` → "Finished in 0.17s" (nothing to
recompile; binary was current, so the staleness worry did not materialise).

## VERDICT (one line)

**It is DECODE-bound, and specifically fault-bound, not load-bound and not
bandwidth-bound.** The 4 bpw **load finishes in 3 581 ms** — statistically identical to the
3 bpw load (2 687 ms) — and then the process spends **> 96 s** failing to emit even one token
while the box sits at **100–240 MB `unused` with 11–12 GB in the compressor**.

The brief's leading hypothesis ("probably load-bound") is **wrong**. A load-bound fix would
have changed nothing.

## Results table

All wall times are hard-capped (watchdog killed the process group at 100 s). `load_ms`,
`decode_tok_s`, `page faults` and `peak memory footprint` come from `--metrics` and
`/usr/bin/time -l`, **which only print on a clean exit** — so for the killed 4 bpw runs they
are **CENSORED (never emitted), not zero**. Load-phase numbers for 4 bpw come from
`KAPLUMBA_LOAD_PROFILE=1`, which prints *before* the run can wedge, and that is why it is
trustworthy where `--metrics` is not.

| # | model | tokens | wall ms | load `read+convert` | full `load_ms` | `decode_tok_s` | page faults | peak footprint | swap delta |
|---|-------|--------|---------|--------------------|---------------|----------------|-------------|-----------------|------------|
| R1 | Gxma-4-bit | 1 | **110 023 (KILLED)** | not printed | censored | censored | censored | censored | **+3 094 MiB** |
| R2 | Gxma-repacked | 1 | 8 120 | **2 610 ms** (14 876 563 172 B) | **6 285.84** | 0.000 † | 236 885 | 12 445 525 256 | −8 MiB |
| R3 | Gxma-4-bit | 2 | **100 035 (KILLED)** | **3 555 ms** (18 889 725 668 B) | censored | censored | censored | censored | **+3 711 MiB** |
| R4 | Gxma-repacked | 3 | 8 277 | **2 687 ms** (14 876 563 172 B) | **6 321.09** | **19.519** | 236 942 | 12 437 005 648 | −48 MiB |
| R5 | Gxma-4-bit | 1 | **100 029 (KILLED)** | **3 582 ms** (18 889 725 668 B) | censored | censored | censored | censored | **+2 239 MiB** |
| R6 | Gxma-4-bit (`RESIDENT_LIMIT=19e9`) | 1 | **100 034 (KILLED)** | **3 581 ms** (18 889 725 668 B) | censored | censored | censored | censored | **+2 259 MiB** |
| R7 | Gxma-bpw4-shim | 3 | 32 (FAILED) | n/a | n/a | n/a | 30 | 4 375 008 | 0 |

† **MEASURED gotcha:** `decode_tok_s = 0.000` is *not* a stall — with `generated_tokens = 1` the
metric has no denominator. **You need `--max-new-tokens ≥ 2` before `decode_tok_s` means
anything.** A 4 bpw `tokens=1` run can therefore never produce a decode number at all, which is
part of why this was conflated before.

Supporting memory-state samples (`top -l 1 -n 0 | grep PhysMem`):

| state | `unused` | compressor | swap used |
|-------|----------|------------|-----------|
| idle, healthy (before every 3 bpw run) | **14–17 G** | 142–285 M | 1 711–1 774 MiB |
| 3 bpw, during run (t=4.2 s) | 110 M ‡ | — | falling |
| **4 bpw, t = 8 s … 100 s (23 consecutive samples)** | **100–787 M, mostly ~130 M** | — | rising |
| 4 bpw, after kill | 225 M | **11 G** | 3 948 MiB |

‡ R4's sampler caught the process at a moment when the box was briefly still draining; the
sustained readings are in R5.

**R5 timeline (the money shot).** Sampling every 4 s, `unused` never recovers after load:

```
t=4.2s   122M unused     <- load already finished (read+convert printed at 3.582s)
t=8.3s   787M unused
t=12.5s  144M unused
t=16.6s  129M unused
 ... 20 more samples, 100-242M unused, no recovery ...
t=99.6s  136M unused     -> killed at 100s, never emitted a token
```

## The decisive number: it is 690x off the roofline, not 1.27x

Decode re-reads the **entire** weight set every token. That is MEASURED source bytes from
`KAPLUMBA_LOAD_PROFILE`, against the M4 Pro's 273 GB/s (`AGENTS.md` §8):

| model | bytes/token | roofline @ 273 GB/s | MEASURED |
|---|---|---|---|
| 3 bpw | 14 876 563 172 (13.85 GiB) | 54.5 ms → **18.35 tok/s** | **19.519 tok/s** (51.2 ms/tok) |
| 4 bpw | 18 889 725 668 (17.59 GiB) | 69.2 ms → **14.45 tok/s** | **> 96 000 ms/tok (< 0.021 tok/s)** |

Two things follow, and they are the whole diagnosis:

1. **The 3 bpw control is AT (slightly above) the memory roofline.** The kernel/dispatch path
   is therefore essentially optimal. There is no hidden 2x kernel tax to recover.
2. **4 bpw is ~690x below its own roofline.** A 1.27x bigger model cannot explain a 1000x
   slowdown. Weight bandwidth is not the constraint; **page-fault servicing is.**

Load bandwidth corroborates that the I/O path itself is healthy — both models read their
shards at the same rate:

- 3 bpw: 14 876 563 172 B / 2 687 ms = **5.54 GB/s** (MEASURED)
- 4 bpw: 18 889 725 668 B / 3 581 ms = **5.27 GB/s** (MEASURED)

## Mechanism

**MEASURED** — supply vs demand:

- Idle supply on this box: **16–17 GiB `unused`**, of which ~2.8 GiB is already wired by the
  desktop.
- 4 bpw demand: **17.59 GiB of source bytes copied into dirty anonymous arena memory**
  (`weights.rs:646-663`, rayon-parallel `read_at` into 512 MiB `StorageModeShared` segments),
  plus KV + scratch. Prior art measures the resulting footprint at ~17.44 GiB
  (`spec/memory-swap-cliff/implementation_notes.md:200-216`).
- 17.4–17.6 GiB of **dirty, incompressible, bit-packed** anonymous memory against ~16 GiB of
  supply ⇒ **over supply by roughly 1.5–2.5 GiB.**

**MEASURED** — what macOS then does, from the swap column: every 4 bpw run pushed swap **up by
2.2–3.7 GiB**; every 3 bpw run pushed it **flat or down (0 to −48 MiB)**. That ratio is the
signal, and it is unambiguous.

The ladder is purge → compress → swap, and **step 2 buys nothing on bit-packed weights**
(`spec/memory-swap-cliff/implementation_notes.md:216-224`). So the arena goes to swap, and
then every decode step must fault ~1.15 M 16 KiB pages (18 889 725 668 / 16384) back in,
**inside the timed region**. Random page-in through the compressor/decompressor plus swap I/O,
not sequential bandwidth, is what costs ~96 s per token.

**INFERRED, high confidence** — the arithmetic that makes this concrete: at the NVMe rate
prior art measured (22.1 GB/s), a *fully* cold 4 bpw weight set is 855 ms/token. Observed is
~100 s/token, ~117x worse still, because swap-in is **random 16 KiB pages**, not sequential
stream. Only a fault storm produces that gap. This also explains why the box needed ~45 s to
recover after every run and why I found it wedged (12 G compressor, 4.4 GiB swap) on arrival.

## The two secondary questions, answered

**Item 4 — is it just row-major instead of colmajor? NO, it is not the 1000x.** But it is a
real, separate inefficiency:

- `Gxma-4-bit/quantization_config.json` has **no `layout` key**; `Gxma-repacked` has
  `"layout": "colmajor"`. MEASURED.
- `model.rs:3525` applies the weight transpose **only** when
  `quantization.layout == Some("colmajor")`, so 4 bpw leaves `permutation_mode = 0`
  (defaults at `model.rs:340`, `model.rs:371`).
- Every fused decode path is gated on `permutation_mode == 1` (`model.rs:1281`, `1418`,
  `1462`, `1534`, `1878`, `1937`, `2062`, `2144`, `2238`). At `0`, all of them fall through
  to separate unfused `linear_gemv_ishizuki_k{N}_w2` dispatches. MEASURED from code.
- **Sizeable but second-order.** The unfused path reads the *same bytes*; it changes dispatch
  count (roughly 5 per projection group × 60 layers instead of ~2), which at typical
  sub-millisecond dispatch cost is on the order of tens of ms/token — a low-single-digit
  factor, **not** 1000x. INFERRED from dispatch structure, not measured.

**Item 5 — does the resident limit matter? NO.** MEASURED: R6 with
`KAPLUMBA_RESIDENT_LIMIT=19000000000` produced `read+convert=3 581 ms` versus R5's
`3 582 ms` on the default 18 000 000 000, and the same 100 s death. The brief's premise that
the checkpoint "needs 18 046 374 518 bytes resident, 46 MB over the limit" **does not bind**:
the limit is checked against **arena capacity**, not source bytes (`weights.rs:154`, `:211`),
and the load completes comfortably under the default. **There is nothing to win here.**

## Reconciliation with the pre-existing `THRASH-4BIT.md` (important, unresolved)

`THRASH-4BIT.md:309` previously measured 4 bpw **`load_ms` = 47 392 ms** (CLI runs 90 970 /
102 319) against 3 bpw 7 867 ms, and concluded "the collapse starts before a single token".
My `read+convert = 3 581 ms` appears to contradict that. **Both can be true, and the
difference is what is being measured:**

- `read+convert` (`weights.rs:665-671`) is the **isolated shard→arena copy only**, and it is
  the one number that survives a kill because it prints before the run wedges.
- `load_ms` (`crates/cli/src/main.rs:54`) is **process start → model ready**, so it also
  contains mmap/open, header parse, tokenizer, GPU-**lock wait** (up to 30 min by default —
  `perf_probe_stats_noise_floor.rs:180-189`, V7), and post-copy setup.
- Their runs additionally report load-phase `swapouts` of **875 912 pages (13.67 GiB)** and
  `compressions` 3 078 552, i.e. **their box was already thrashing during the copy**. Mine
  started from a recovered 16 GiB idle box.

So the honest statement is: **the copy is seconds of work on both models (MEASURED, 5.3–5.5
GB/s, nearly identical), but `load_ms` on 4 bpw is a function of how badly the box is already
compressed and is therefore not a stable quantity.** `THRASH-4BIT.md`'s larger numbers are
real measurements of a worse memory state, not a contradiction.

**The verdict survives either way**, and this is why: even granting the *fastest* load I
measured (3.58 s copy), every 4 bpw run spent **≥ 95.8 s after that copy completed** without
emitting a token, versus **~1.5 s** of prefill+3 decode tokens for 3 bpw (`ttft_ms = 1331.8`
+ 3 tokens at 19.5 tok/s). Load cannot be the binding constraint when 96 s remains after it.

**Open question for whoever picks this up:** measure 4 bpw `load_ms` with
`gpu_lock::describe()` actually called (it exists, `gpu_lock.rs:367-374`, and `main.rs` never
calls it — dead code whose stated purpose is exactly this). Until `load_ms` is decomposed into
`lock_wait` / `copy` / `setup`, the two figures in these two documents cannot be reconciled
numerically.

## Things that did NOT work / are dead ends

- **`Gxma-bpw4-shim` cannot be used to isolate layout (R7).** It has the *same* 60-layer
  architecture (`text_config`: 60 layers, hidden 5376, 32 heads, head_dim 256, vocab 262144)
  and the same 4 bpw / 8 bpw-head, no-layout config — so it was the perfect cheap
  same-shape/different-layout probe. **It is an orphaned config-only directory**: 31 MB, 2 422
  tensors in the index, but `model-00001-of-00003.safetensors` **does not exist**
  (`checkpoint inspection failed: ... No such file or directory`). It fails in 32 ms and
  cannot answer item 4. Do not spend a run on it again.
- **A colmajor 4 bpw checkpoint can be produced without writing new code**:
  `target/release/repack_model` **already exists** (built 3 Oct, 622 KB) and
  `exl3::reorder_tensor_to_colmajor` is the only repacker in the tree. Per instructions I did
  **not** run it. Note `shim_checkpoint.rs:316-319` warns the repacker hardcodes 48 u16 per
  tile, so verify the repacked checkpoint's shapes before trusting it.
- **Raising `KAPLUMBA_RESIDENT_LIMIT` is a dead end** (see item 5).

## Measurement caveats (my own bugs and traps — do not repeat)

1. **`/usr/bin/time -l` prints nothing when the process is killed.** Every 4 bpw page-fault
   and peak-footprint number is therefore **unavailable**, not small. Any future 4 bpw
   measurement needs a way to read rusage for a killed process.
2. **My RSS sampler was wrong.** `pgrep -f "kaplumba --model $MODEL"` matched the
   `/usr/bin/time -l ./target/release/kaplumba --model ...` **wrapper** (lowest PID) instead of
   the engine, so the RSS column read a steady 256 KB for a process holding 17.6 GiB. The
   `PhysMem` column in the same samples is from `top` and **is** valid. Match on the exact
   binary path or take the *highest* PID.
3. **Killing `/usr/bin/time -9` orphans `kaplumba`**, which then survives holding the GPU lock
   in uninterruptible I/O (it needed two SIGKILLs to reap). My final script reaps by pattern
   after `wait` and logs `REAPED orphan`. `scripts/gpuq status` afterwards correctly reported
   `gpu: none (stale lock, pid ... is gone)`.
4. **`decode_tok_s` is 0.000 at `generated_tokens=1`** — see the † above. Never diagnose decode
   from a 1-token run.
5. **The box is bimodal even when idle.** On arrival, with **no engine running**, it showed
   `265M unused / 12G compressor / 4469 MiB swap` — that was residue from a dead agent's 4 bpw
   run, and it is why R1 started from a compromised state. It recovered to 15–17G unused
   within ~45 s of the process dying. Consistent with `V3` in
   `crates/engine/tests/perf_probe_stats_noise_floor.rs:84-123`.
6. **Silent decode faults may be manufacturing throughput numbers.** Per
   `perf_probe_stats_noise_floor.rs:193-222` (V8), `commit_terminal` returns `Ok`
   unconditionally and `wait()` never inspects `status`, so a faulted decode buffer yields no
   error and no non-zero exit. On a wedged box this is a real risk for any "measured" 4 bpw
   decode number. I could not emit a 4 bpw token at all, so I have no such number to discount.

## The single cheapest thing to try next

**Stop trying to make 4 bpw fit and measure it — decide the layout question on paper first,
because it is free and it may delete the problem.**

Ranked:

1. **CHEAPEST, ZERO GPU: run `repack_model` on `Gxma-4-bit` to produce a `layout: colmajor`
   4 bpw checkpoint, then measure only `load_ms` and whether `tokens=2` completes inside
   100 s.** Rationale: it is the one hypothesis still standing that has not been tested, it
   changes *dispatch count* (and possibly the access order the GPU wants), and — critically —
   **if the colmajor 4 bpw run still spends >96 s post-load at 130 MB `unused`, layout is
   exonerated outright and the memory budget is the only remaining story.** Do not attempt this
   before the admission-control question below is settled: if the colmajor checkpoint still
   needs ~17.6 GiB it will wedge the box exactly the same way, so budget the same 100 s
   watchdog and the same `PhysMem`/`swap` sampling.
2. **If layout is exonerated: the fix is admission control, not a faster kernel.** The engine
   must **refuse to load** a checkpoint whose projected anonymous footprint exceeds measured
   free supply, instead of loading it and thrashing for 96 s/token. `spec/memory-swap-cliff`
   already reaches this conclusion ("this is why the fix is admission control, not a pinning
   policy", line 228). A load-time estimate of `source_bytes + KV + scratch` compared against
   `PhysMem unused` is a small, well-contained change, and it turns a 1000x slowdown into an
   immediate, honest error.
3. **Only then:** shrink 4 bpw below the supply line (3 bpw layers, i.e. mixed-bpw — the
   `Gxma-repacked` recipe at 13.85 GiB fits in 16 GiB with room to spare, which is precisely
   why it holds 19.5 tok/s).

**Do not** spend effort on `KAPLUMBA_RESIDENT_LIMIT`, on pinning/mlock, or on unfused-kernel
micro-optimisation. All three are measured dead ends or second-order.

## Reproduce

```bash
cd /Users/zayaantaha/coding/kaplumba
M=~/.kaplumba/models/Gxma-4-bit
perl -e 'alarm 100; exec @ARGV' -- env KAPLUMBA_LOAD_PROFILE=1 \
  scripts/gpuq try -- /usr/bin/time -l ./target/release/kaplumba \
  --model "$M" --prompt Hi --max-new-tokens 2 --temperature 0 --metrics
# then: top -l 1 -n 0 | grep PhysMem ; sysctl -n vm.swapusage
```

Helper used: `…/T/opencode/probe4bit.sh` (watchdog + `PhysMem`/swap sampling + orphan reap).
Box left clean: `16G unused`, `202M compressor`, no `kaplumba` process, swap back to 1 688 MiB.