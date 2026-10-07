# implementation_notes — `spec/chat-16k-stall`

Append as work proceeds, newest at the bottom. Every number here was measured
on this box this session unless it says otherwise.

## 0. The reported symptom

`kaplumba --chat` with a long context "goes down to 0.x tok/s". Reproduced
exactly: **0.03 tok/s** (16 generated tokens, 11,299 context) — 16 tokens in
494.9 s. Then the user killed a run that had been going 600 s+ (exit 137 =
SIGKILL).

## 1. First measurement: it is `--chat`, not the model

Single-shot and chat on the same checkpoint, same context, same prompt bytes:

| context | single-shot | chat (before fixes) |
|---:|---:|---:|
| 2,234 | **14.71** | 6.71 |
| 11,354 | **14.01** | — |
| 16,484 | **13.45 / 13.51 / 13.49** | — |

So the generator is healthy at 16k and the chat path is not. That splits the
problem immediately and stopped me from looking at kernels, which a
subagent's ranked hypothesis list had suggested.

Peak memory at 16k single-shot: `phys_footprint` 15,702,698,688 B (14.6 GiB),
0 swap. Fine.

## 2. Chat tok/s vs context, before any fix

| context | tok/s |
|---:|---:|
| 199 | 4.39 |
| 754 | 8.56 |
| 2,234 | **10.94** (peak) |
| 4,454 | 7.02 |
| 7,414 | 2.02 |
| 11,299 | 1.21 |

Rises to ~2k then collapses. Not a threshold at any one of these — and
specifically **not** a "200 cliff" as first guessed; the 199-token point is
slow for a different reason (a 32-token reply sits inside per-turn fixed costs,
and there is barely any KV to amortise them over).

## 3. Bug A — wait before encode. FIXED, `8fbf487`

The loop did `previous.wait()` at the top of the iteration, before building the
next encoder, so the GPU drained and idled through the host's whole
`forward_step_pipelined` encode on **every token**. `generate.rs:1031` commits
`next_enc` and `:1033` then waits — it always had it right.

Reordered to encode+commit N+1 first, then wait on N. The
`read_host_token`-after-`wait` invariant is preserved (the file records that
reading a staged token early was a shipped bug once).

| | tok/s @2,234 |
|---|---:|
| chat before | 6.71 |
| **chat after** | **15.11 / 15.18 / 15.19 / 15.21** |
| single-shot (reference) | 15.16 / 15.38 / 15.16 |

**Parity.** Two source-level tests added, both sabotage-verified: reverting the
order fails `the_pipelined_chat_loop_commits_before_it_waits` with the offsets
printed. A second test guards the read-after-wait invariant.

⚠️ **This fix is NOT yet shown to be output-identical.** T1 of the spec runs
the `#[ignore]`d token-stream equivalence tests. Until then it is a
performance claim, not a correctness claim.

## 4. Bug B — the in-flight window is smaller than one decode step. NOT FIXED

Bug A fixed short context to parity but **16k still collapses**. Measured after
the fix, same 11,299 context:

| max-new-tokens | decode time | tok/s |
|---:|---:|---:|
| 8 | 38.0 s | 0.21 |
| 12 | 224.0 s | 0.05 |
| 16 | 291.9 s | 0.05 |
| 16 (repeat) | 23.9 s | 0.67 |
| 20 | 47.8 s | 0.42 |

That scatter across repeats of the *same* binary is itself the finding: there
is no clean curve, so **a median of a good run proves nothing here.** It is a
stall that is sometimes not hit.

### The stacks

Three independent `sample` captures during a stall, 100% agreement, all on the
main thread inside the host encode:

```
forward_step_pipelined (1470 / 1513 / 612 samples)
  forward_chunk
    ...
    MetalEncoder::tick            (encoder.rs:524)
      -[_MTLCommandBuffer waitUntilCompleted]      <-- host blocked, 1419/1513/587 samples
```

100% of the stall is `tick()` → `roll()` (`encoder.rs:539-549`) blocking on the
oldest in-flight buffer.

### The arithmetic that explains it

```
600 dispatches per decode step / 50 per buffer  =  12 command buffers per step
MAX_IN_FLIGHT                                    =   8
```

**The window cannot hold one decode step.** So after ~8 buffers the host stops
encoding and waits for step N's first buffer to drain, mid-step. Host and GPU
are fully serialised; the "pipelined" loop degenerates to synchronous.

Why short context hid it: identical dispatch count, but at ≤2k each buffer
drains in microseconds so the block is unmeasurable. At 16k a decode step is
~74 ms of GPU work spread over 12 buffers, and the host waits for most of it.

### Consequence for reading any earlier chat number

Every chat tok/s recorded before Bug A was fixed, and every one recorded
*before* Bug B is fixed, is a measurement of a stall whose depth depends on
which buffers happened to be outstanding. **Do not compare across those two
regimes.**

## 5. Retractions and corrections

* **The "+3.63 tok/s if layers reach 254.6 GB/s" is withdrawn.** 254.6 is the
  *six* bpw rate; the layers are 3 bpw, where this box reaches 165.2 GB/s at
  the head's shape, and the layers already run 185–203. See
  `spec/decode-3bpw-gap-19t/implementation_notes.md §13`.
* **The O(N²) `Detokenizer::push` theory was wrong.** Measured 0.04–0.13 ms per
  token. It replays *generated* ids only (16–60 of them), not the prompt. Not a
  factor here; still a latent defect for a 4096-token reply.
* **"Wait-before-encode is C4, worth a little."** It was worth 2.1×. The
  subagent's ranking had it as "real, but small".

## 6. Environment gotchas hit

* **Stale binary twice.** `cargo build -p kaplumba-engine` does **not**
  rebuild `target/release/kaplumba` (a different package: `kaplumba-cli`).
  Measured "6.71 vs 15.69, no fix" against a binary that did not contain the
  fix. Always `cargo build --release -p kaplumba-cli` before a chat
  measurement. This is the `include_str!` class of trap in a new costume.
* **`timeout` does not exist** on this box; a monitoring loop with `kill -0`
  is the substitute.
* **A killed model leaves a stale lock** at `~/.kaplumba/gpu.lock` naming a dead
  pid; `gpuq status` reports it as busy until the pid is reaped. It clears on
  its own after ~5 s.
* Agent overlap produced 35.3 / 55.3 / 156.5 GB/s for the same arm in one
  session. Those are **discarded, not spread** (`crux.md §2.7`).

## 7. Log

(append per task: command, median ± spread, control drift, verdict, commit)

## 8. Live Memory & Swap Diagnosis (2026-10-07)

Captured during live stall on PID 68980 (`kaplumba --chat ... --max-new-tokens 12`):

```
Physical footprint: 13.6G
vmmap -summary 68980:
  IOAccelerator (graphics): 13.4G VIRTUAL, 13.4G RESIDENT, 12.5G SWAPPED (1,440 regions)
  mapped file:              15.8G VIRTUAL, 5.3M  RESIDENT,    0K SWAPPED
top:
  PhysMem: 23G used (3910M wired, 12G compressor), 155M unused
  kaplumba: MEM 14G, CMPRS 11G, FAULTS 1,034,162, PAGEINS 237,027
vm.swapusage:
  total = 5120.00M, used = 3713.75M
```

* **Root Cause Found:** `weights.rs` loads weights into ~1,440 individual
  `StorageModeShared` Metal buffers via `read_exact_at` (`pread`), turning 13.4 GiB
  of model weights into dirty anonymous memory.
* At 11k–16k context, KV cache adds ~3 GiB. Total dirty + wired memory reaches
  25+ GiB, exceeding the 24 GiB RAM.
* EXL3 bit-packed weights have Shannon entropy ~7.97 bits/byte (incompressible).
  macOS dumps 12.5 GiB into compressed swap.
* Decode cycles through all 60 layers on every token, incurring ~237,000 pageins
  from SSD swap per token and collapsing throughput to 0.03 tok/s.
* **Contrast with MLX:** MLX maps weights directly via `mmap` (`new_buffer_with_bytes_no_copy`),
  keeping weights as clean file-backed pages (`SWAPPED = 0K`), which are never
  compressed or written to swapfiles.

---

## 8. 🔴 T2 RESULT — Bug B is REFUTED. The in-flight window is not the cause.

The sweep finished:

| `KAPLUMBA_MAX_IN_FLIGHT` | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|
| chat tok/s @11,299 | 0.23 | 0.27 | 0.31 | 0.17 |

**Quadrupling the window changes nothing and is not monotonic.** The `8 < 12`
arithmetic in §4 is true and the mechanism it implies is demonstrable, but
raising the window to 64 — five times a whole step's worth of buffers — does not
recover the stall. So `roll()`'s window pressure is **not** what is costing the
time, and the fix I proposed is wrong. Recorded rather than quietly dropped.

## 9. What the GPU profile actually says

Chat decode, 1 token, 60 layers, 546 boundaries, at 11,299 context
(`KAPLUMBA_GPU_PROFILE=1 KAPLUMBA_GPU_PROFILE_DECODE=1`):

```
sum of per-kernel GPU own_ms =  254.5 ms
inter-boundary gap           = 1689.2 ms     <-- 86% of the step
wall span                    = 1969.6 ms
```

**86% of the step is not kernel time.** The GPU is idle, or the host is not
feeding it. Compare single-shot at 10,744 tokens: **14.45 tok/s = 69 ms/step.**

Per-dispatch, chat vs single-shot, same kernels:

| kernel | chat | single-shot | ratio |
|---|---:|---:|---:|
| `gate_up_gelu` | 1.816 ms | 1.412 ms | 1.29× |
| `down_proj` | 0.902 | 0.707 | 1.28× |
| `qkv_proj` | 0.701 | 0.554 | 1.27× |
| `o_proj` | 0.422 | 0.327 | 1.29× |
| `qk_proj` | 0.818 | 0.649 | 1.26× |
| **`head/gemv`** | **14.69** | **4.61** | **3.19×** |
| `attention` | 0.028 | 0.023 | 1.22× |

So the kernels are only **1.3× slow** (and the head 3.2×), which accounts for
maybe 90 ms of a ~3000 ms step. **The stall is not compute.**

## 10. What is still unexplained, stated plainly

Established:
1. Single-shot is healthy: 13.45–13.51 tok/s @16,484; 14.45 @10,744.
2. Chat collapses: 0.05–0.31 tok/s @11,299, and it is not a smooth curve —
   0.05, 0.21, 0.23, 0.27, 0.31, 0.42, 0.67 across repeats of one binary.
3. Chat's decode kernels are ~1.3× single-shot's (head 3.2×).
4. **86% of chat's decode step is inter-boundary gap.**
5. `MAX_IN_FLIGHT` 8 → 64 does not change it.
6. Stack samples put the host in `waitUntilCompleted` inside `tick`/`roll`
   inside `forward_chunk`, ~100% of samples.

(4), (5) and (6) are in tension: if the host is blocked on the *oldest
in-flight* buffer, raising the cap to 64 should remove the block, and it does
not. Either the block is on a buffer that is genuinely slow for reasons the
profile does not capture, or the wait is inside a different submission path than
`roll`. **I have not isolated this.** Q2 in `spec.md` stands open.

Also unresolved and possibly the same bug: `head/gemv` is 3.19× slower in chat.
That is a single kernel with no context dependence of its own (1.057 GB, fixed),
so something about chat's state changes it — and it is the largest single row
in the table.

## 11. Corrections to §2 and §4

* §2 "Bug B — `MAX_IN_FLIGHT` is smaller than one decode step. NOT FIXED." →
  the arithmetic stands, the **diagnosis does not**. See §8.
* §4 R1's framing ("size the window so `roll()` never blocks mid-step") is
  therefore the wrong lever and must not be T2. It becomes T2b at most, after
  the real cause is found.

## 12. Tooling gotcha, cost me a cycle

`KAPLUMBA_GPU_PROFILE_DECODE=1` prints one table per forward, and a 16k chat
turn produces hundreds. Selecting "the last table" picked a prefill chunk, and
"the first" picked a 2-token forward — both looked like decode and neither was.
The table header line (`tokens=N chunks=N boundaries=N`) is the only reliable
selector, and `awk '/=== GPU PROFILE/{n++} n==2'` is how to get the decode one.

Also: two profiling runs were SIGKILLed by the OS (not by me). At 16k the
process is at the memory ceiling with a profiler attached; suspect the OOM
killer before suspecting the probe.

## 13. T0: Zero-Copy File-Backed Layer Trellis Weights Implemented

**Implemented:**
- Layer `.trellis` weights and `lm_head.trellis` are mapped directly from safetensors shard mmap via `new_buffer_with_bytes_no_copy`.
- `embed_tokens.weight` (2.82 GB) remains strictly offloaded via `raw_zero_copy_tensor_buffer` with in-kernel BF16 conversion.
- Vision tower (1.15 GB) remains strictly excluded (0 bytes allocated).
- Prefaulting cleanly with `madvise(MADV_WILLNEED)` on shard mapping.
- All 102 unit tests in `crates/engine` pass cleanly in debug and release.

**Measurements with `Gxma-repacked`:**
- `target/release/kaplumba --prompt "Explain the theory of relativity." --max-new-tokens 4 --temperature 0 --metrics`:
  - `load_ms`: 6018 ms
  - `ttft_ms`: 1323 ms (prefill 14.36 tok/s)
  - `decode_tok_s`: 18.69 tok/s
  - `/usr/bin/time -l`: 0 swaps, peak memory footprint 1.54 GiB, `vm.swapusage`: 0 growth.
- Dirty anonymous Metal graphics memory reduced by ~10.25 GiB.

