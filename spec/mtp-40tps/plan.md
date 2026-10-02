# plan — mtp-40tps

Amended 2026-10-02: direction changed from mlx-vlm + a downloaded 18.41 GB MLX
target to **the kaplumba Rust engine + the already-local Gxma checkpoint**. mlx-vlm
is now read-only *reference material*, not a runtime dependency. See `spec.md` for
the arithmetic behind the change; the short version is 11.6 GiB resident and 19
tok/s already achieved versus 17.27 GiB and a 12.74 tok/s ceiling.

## Model layout

| role | path | size | format |
|---|---|---|---|
| target | `~/.kaplumba/models/Gxma/Gxma` | 16.0 GB on disk, **11.6 GiB resident** | EXL3 3.0 bpw (`mcg` codebook + `trellis` + `suh`/`svh`); loads today |
| drafter | `~/.omlx/models/mlx-community/gemma-4-31B-it-qat-assistant-4bit` | 0.264 GB | MLX affine-4bit `g64` (U32-packed + `scales`/`biases`) — **needs conversion** |

The engine cannot read MLX affine format, so converting the drafter is a real task
(T5), not a file copy.

## Drafter architecture — read from installed mlx-vlm, not guessed

From `mlx_vlm/speculative/drafters/gemma4_assistant/gemma4_assistant.py`:

```
inputs_embeds = concat[ target_embed(last_token) , last_hidden ]   # 2 x 5376
h  = pre_projection(inputs_embeds)          # 10752 -> 1024, no bias
for layer in 4 decoder layers:               # kv_shared_only = True
    reads the TARGET's shared K/V for its layer_type; no cache of its own
h  = norm(h)
last_hidden = post_projection(h)            # 1024 -> 5376, feeds the next round
logits      = embed_tokens.as_linear(h)     # tied lm_head, 1024 -> 262144
```

Drafting autoregresses `block_size - 1` candidates (`block_size` default 4 ⇒ 3
drafts + 1 bonus), `position_ids` held constant across draft steps, K/V taken from
the target.

**This shape is why MTP is cheap here:** the drafter keeps **no KV cache of its
own** (`gemma4_assistant.py:104-106`, `make_cache()` returns `[]`). It reuses the
target's K/V, so it costs 0.264 GB of weights and no KV growth.

## Data flow of one MTP round (`mtp.py:523 _mtp_rounds`)

1. Drafter autoregresses `block_size-1` candidates from
   `[target_embed(last_token) ; last_hidden]`, reading the target's shared K/V.
2. Target runs **one** forward over `[bonus, cand_1..cand_k]` — a single weight
   read, `k+1` positions of hidden state.
3. Acceptance walk draws an independent target sample `y ~ p` per position and
   accepts draft `x ~ q` iff `x == y`, else emits `y` and stops.
4. `rollback_speculative_cache` trims the target KV, and the drafter's shared-KV
   view is re-sliced (`mtp.py:667-681`).

## Exactness, and the one thing that must not be broken

The rule is distribution-exact for *any* temperature: for `x ~ q`, `y ~ p`
independent, `P(emit = z) = p(z)(1-q(z)) + p(z)q(z) = p(z)`.

That argument **requires the target and drafter RNG streams to be independent.**
mlx-vlm achieves it with `_SpeculativeSamplerRNG`
(`mlx_vlm/speculative/common.py:46-132`), which saves and restores `mx.random.state`
around the drafter call. If the engine reuses one RNG stream for both, the exactness
argument collapses and R6 will fail. Port the separation, do not re-derive it.

## Roofline (must be re-derived against Gxma's 11.6 GiB, not the abandoned target's)

```
target pass, 1 token   ~ 11.6 / 220 GB/s  =  52.7 ms   -> ~19 tok/s (matches measurement)
40 tok/s requires      =  40 x 0.025 s    =  1.06 s of work per second
                    =>  ~2.1 accepted tokens per target pass
drafter cost           =  0.264 GB/pass    ~  1.2 ms per draft step (negligible)
```

Required per-draft acceptance `a` for 40 tok/s under an optimistic
position-independent model (real acceptance decays with depth, because deeper
drafts condition on *drafted* rather than accepted history):

```
K=3 drafts : a >= 0.84
K=5 drafts : a >= 0.73
K=7 drafts : a >= 0.70
```

**These thresholds are the gate.** The synthetic probes measured p–q overlap
c ≈ 0.20–0.40, which would fail all three rows — but those were synthetic
distributions, not this drafter against this target. T4 measures the real number
before any optimization work is built.

## Measurement discipline

- Exactly one model process; `pgrep -fl 'mlx_vlm|omlx|kaplumba'` must be empty first.
- A/B interleaved (A B A B …), median of ≥3 runs, spread reported. This box has
  produced 3.7–6.4 tok/s from the same binary minutes apart.
- Memory from `top -l 1 -n 0 | grep PhysMem` (`unused`, which switches K/M/G/T and
  drops the decimal below 10) and `sysctl -n vm.swapusage`. `vm_stat` `Pages free`
  is only for *decomposing* memory, never as a pressure signal.
- The budget figure is `phys_footprint` via `/usr/bin/time -l`, never
  `--metrics peak_resident_pages_bytes` (that is `ru_maxrss` and misses the Metal
  pages by 5.5 GiB).
- `scripts/gpuq try -- <cmd>` for scheduling; it takes no lock and never may — the
  engine owns the lock via `gpu_lock.rs` and returns exit 75 on contention.

## Risks

1. **Acceptance rate (primary).** If real per-draft acceptance is below ~0.73, no
   K reaches 40 tok/s and the deliverable becomes the ceiling plus proof.
2. **KV rollback correctness.** Trimming target KV on rejection and re-slicing the
   drafter's shared-KV view is the most likely source of a silent distribution
   skew. It gets a dedicated test.
3. **Drafter/target mismatch.** Both are vocab 262144 with tied embeddings and the
   drafter's `backbone_hidden_size` 5376 matches Gxma's `hidden_size`, but a
   drafter trained against a different target revision would show up as low
   acceptance. Verify before building.
4. **Overhead on the hot path.** The engine already runs ~53 ms/pass; adding a
   per-round host round-trip must not cost more than it saves. Profile.

## Requirement → task map

| req | task |
|---|---|
| R1 | T5 drafter conversion |
| R1 | T6 drafter forward |
| R4 | T2 + T3b (done — verification audit and architecture read) |
| R2 | T7 baseline |
| R3 | T8 round loop |
| R5 | T10 optimization sweep |
| — | **T4 acceptance gate** (before T10) |
| — | T9 KV rollback |
| R6 | T11 distribution faithfulness |
| R7 | T12 coherence |
| — | T13 write-up |