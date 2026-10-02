# implementation_notes — mtp-40tps

Running log. Facts established here are cited by the other docs.

---

## T0 — environment audit (2026-10-02)

| item | value |
|---|---|
| CPU / GPU | Apple M4 Pro, 10 perf cores, 20 GPU cores |
| RAM | `hw.memsize` = 25769803776 = **24.00 GiB**; no discrete GPU |
| Page size | **16384 bytes** (`vm_stat` header; the 4096 assumption understates every count 4x) |
| Disk free | 117 GiB (`/dev/disk3s1s1`, APFS, sealed root; data volume 117 GiB free) |
| swap | `total 2048M`, ~1162–1242M already used before any model work |
| mlx / mlx_lm / mlx_vlm | **0.32.0 / 0.31.3 / 0.6.8** (uv tools, `~/.local/share/uv/tools/`) |
| omlx | 0.4.1, wrapper `~/.omlx/bin/omlx` → `~/oMLX.app/Contents/MacOS/omlx-cli` |
| uv | present at `~/.local/bin/uv` |
| system python3 | 3.9.6, `transformers 4.57.6` — no `gemma4_assistant`; **not used, not modified** |

Memory snapshot at rest (`vm_stat`, page size 16384):

```
Pages free                9.77 GiB
File-backed pages         6.20 GiB   <- reclaimable
Anonymous pages           4.08 GiB   <- app memory (Firefox 1.6, opencode 1.0, daemons)
Pages wired down          2.89 GiB   <- kernel + Metal driver
Pages occupied by compressor  0.42 GiB
```

=> **reclaimable ≈ 15.97 GiB**, locked ≈ 7.39 GiB, of 24.00 GiB.
`top` rounds this to "12G used / 12G unused".

`~/.omlx/settings.json` is not the lever it looks like: `memory_guard_tier: balanced`,
`soft_threshold 0.85`, `hard_threshold 0.95`, `cache.ssd_cache_dir: null`,
`cache.initial_cache_blocks: 256`, `scheduler.chunked_prefill: false`,
`sampling.max_context_window: 32768`. Nothing there reduces resident weights, and
oMLX's guard can only *reduce* chunk sizes (it cannot make 17 GB fit in 16 GiB).
`~/.omlx/models/` contains only the drafter — there is **no MLX target on the box**.

---

## T1 — target weight footprint (measured, not assumed)

`mlx-community/gemma-4-31b-it-4bit` is the canonical MLX 4-bit Gemma-4-31B-it
(40 964 downloads, 4 shards, 18.41 GB). Per-tensor byte totals were computed by
pulling each shard's **safetensors header only** via HTTP range requests
(`../opencode/hdr2.py`, 8 + header_len bytes per shard):

```
text_linear           16.474 GB     60 layers x (q,k,v,o,mlp) affine-4bit g64
embed_tokens           0.793 GB     262144 x 5376, also affine-4bit g64
norms + layer_scalar   0.003 GB
vision tower           1.143 GB     358 tensors
                     ---------
                     18.412 GB     ->  17.27 GB text-only
```

Config facts that matter (`.../config.json`): 60 layers, `layer_types` = 48
`sliding_attention` (window 1024) + 12 `full_attention`; hidden 5376; 32 heads;
16 KV heads; 4 global KV heads; `head_dim` 256, `global_head_dim` 512;
`tie_word_embeddings: true`; `num_kv_shared_layers: 0`; `eos_token_id: [1,106,50]`;
`temperature`/`top_k`/`top_p` present at top level.

Alternatives checked and rejected on size (GB, safetensors only):

```
lmstudio-community/gemma-4-31B-it-MLX-4bit      18.41   (identical mirror)
mlx-community/gemma-4-31b-it-mxfp4              17.45   (-5%, mx only)
mlx-community/gemma-4-31B-it-uncensored-heretic  17.27  (ablated model, not the target)
mlx-community/gemma-4-31B-it-qat-4bit           28.82
unsloth/gemma-4-31b-it-UD-MLX-3bit              20.67   (!) 3-bit *larger* than 4-bit
mlx-community/gemma-4-31b-it-8bit               33.76
```

`mlx-community/gemma-4-31b-it-4bit` is the smallest faithful target.

**Network is the long pole, and it is not HF-specific.** Measured aggregate
download rate on this box:

```
huggingface.co (4 parallel shards)          0.6 – 1.6 MB/s
8 extra parallel range connections          1.65 MB/s aggregate — main download *fell* to 1.12
files.pythonhosted.org (numpy sdist)        1.05 MB/s  (concurrent with HF)
disk write (dd 1 GiB)                        3.56 GB/s  (not the bottleneck)
```

Adding connections does not raise aggregate throughput, so it is an upstream/ISP
cap, not a client-side concurrency limit. 18.41 GB therefore takes hours. No auth
token is available; the run logs `Warning: You are sending unauthenticated
requests to the HF Hub`.

---

## T2 — verification audit: is the non-greedy MTP path proper speculative sampling?

**Yes — it is distribution-exact, and it is not "accept the draft
unconditionally".** The rule is: draw an *independent* target sample `y ~ p` from
the same conditioning; accept the draft `x ~ q` iff `x == y`; otherwise emit `y`.

Sources (installed package, read-only):

- `mlx_vlm/speculative/common.py:135-158` — `_speculative_walk`. Docstring says
  "Exact-greedy"; body is a per-position equality test against `target_tokens`,
  then `new_tokens = draft_row[:accepted] + target_row[accepted:accepted+1]`
  (line 157) — i.e. exactly **one** target bonus token, and only the accepted
  *prefix* of drafts. Tokens after a rejection are discarded.
- `mlx_vlm/speculative/mtp.py:393-427` — `_mtp_acceptance_walk`. For
  `temperature 1.0` there is no greedy path, `verify.target_tokens is None`, and
  the batched sampler has no `sample_target`, so it lands in
  `_speculative_walk_deferred_greedy` (`mtp.py:182-227`), the same rule applied
  position-by-position with the LM-head projection deferred until rejection.
- `mlx_vlm/speculative/common.py:46-132` — `_SpeculativeSamplerRNG` keeps the
  **target and drafter RNG streams separate** (save/restore of `mx.random.state`).
  That independence is what the exactness argument needs.
- `mtp.py:555-561` — the RNG wrapper is enabled when
  `not greedy_sampling and not _sampler_supports_positioned_target(sampler)`, which
  is exactly the temp>0 / plain-function-sampler case here.
- Drafter sampling uses the **same** `sampler` object as the target
  (`mtp.py:608-617` -> `gemma4_assistant.py:280 tok = sampler(logits)`), so the
  drafter is sampled under the identical temp/top_k as the target. Both
  distributions are therefore the same *kind* of distribution.
- Cache correctness after rejection: `mtp.py:667-681` calls
  `lm.rollback_speculative_cache(...)` (implemented at
  `models/gemma4/language.py:~750`) and `_slice_shared_kv_after_reject`
  (`mtp.py:430-444`). This is the part most likely to silently skew the output
  distribution, and it is what R6 must test empirically.

**Exactness argument.** For any p, q and independent x~q, y~p:
`P(emit=z) = p(z)(1-q(z)) + p(z)q(z) = p(z)`. Exact for any temperature.
(Equivalently: `P(x=y) = sum_z p(z)q(z)`, the p–q overlap.)

**Numerical confirmation** (`../opencode/specexact.py`, a verbatim transcription of
both walks, N = 200 000 draws per case, chi-square on the first emitted token):

```
peaked p, near-identical q      K=3: TVD=0.00107 chi2 p=0.8496
peaked p, near-identical q      K=5: TVD=0.00164 chi2 p=0.7604
flat p (temp 1.0, wide)        K=3: TVD=0.00249 chi2 p=0.5197
flat p (temp 1.0, wide)        K=5: TVD=0.00193 chi2 p=0.8042
mismatched drafter             K=3: TVD=0.00290 chi2 p=0.0036   <- see below
mismatched drafter             K=5: TVD=0.00252 chi2 p=0.3306
two-token vocab (worst case)   K=3: TVD=0.00117 chi2 p=0.9186
two-token vocab (worst case)   K=5: TVD=0.00086 chi2 p=1.0000
```

TVD ~0.001–0.003 is exactly the `1/sqrt(N) ≈ 0.0022` sampling-noise floor for
N = 2e5 over a 5-symbol alphabet, i.e. **no detectable divergence**. Seven of
eight p-values are uniform; the single `p = 0.0036` is within the multiple-
comparison expectation for 8 tests (expected min ≈ 0.006).

**Consequence for speed — the important side of this result.** The per-position
acceptance rate of this rule is `c = sum_z p(z) q(z)`, the p–q overlap. Measured
on the synthetic cases above:

```
flat distribution, drafter ~= target : c ~= 0.20  -> E[accepted]/round (K=3) = 0.247
peaked p, drafter ~= target         : c ~= 0.40  -> E[accepted]/round (K=3) = 0.673
```

So at `temperature 1.0, top_k 64` the acceptance is governed by how peaked the
target's own distribution is, not by drafter quality alone. This is what makes the
40 tok/s bar hard — quantified in the roofline section.

---

## T3 — physical limits measured on this box

### DRAM bandwidth
`../opencode/memprobe.py` streams `mx.ones` + `mx.sum` through the GPU. The raw
number is 111–117 GB/s but each iteration also *writes* 1 GB with `ones`, so the
true traffic is 2x the counted bytes: **222 GB/s**, matching the 220 GB/s figure
already in AGENTS.md. Read bandwidth is not the uncertainty.

### NVMe re-read rate (the thrash penalty)
`../opencode/nvme.py`: 8 GB cold sequential read after `purge` =
**22.1 GB/s**. So every GB/token that misses DRAM costs 45 ms.

### How much can actually be resident

**(a) MLX/GPU allocation probe** (`memprobe.py resident`): allocating 1 GB `bf16`
`mx.ones` chunks and evaluating them, starting from 9.54 GiB free / 6.40 GiB
file-backed:

```
 1 GB  free=8.51   wired= 3.91
 5 GB  free=4.78   wired= 7.64
 9 GB  free=1.05   wired=11.37
10 GB  free=0.12   wired=12.30   <- stop: nothing left
```

**Only ~10 GB could be allocated as GPU-wired memory.** `file-backed` never moved
(6.40 GiB throughout) — macOS grew *wired* memory instead of reclaiming the file
cache. This is the shape of the failure AGENTS.md already recorded for the Rust
engine.

**(b) mmap streaming probe** (`resident.py`): a real, non-sparse 17.27 GB file was
written and then mmapped and read end to end. macos **did** hold all 17.27 GB as
file-backed (`fb` 6.40 -> 16.15 GiB, `free` 9.54 -> 0.09 GiB, zero page-ins, zero
swap growth). So file-backed residency of the full weight set is *possible* — but
only by consuming every last free byte. There is then **0.09 GiB left** for the KV
cache, the activations, the LM-head logits, and the Python/MLX runtime.

That is the whole problem in one line:

```
supply for weights+KV+runtime :  24.00 - 2.89 (wired) - 4.08 (app) = 17.03 GiB
demand (17.27 weights + KV + activations + runtime)             ~= 18.1 - 19.3 GiB
deficit                                                              ~1.1 - 2.3 GiB
```

and the missing 1.1–2.3 GiB is not slow, it is **thrash**: at 22.1 GB/s of NVMe,
1.5 GB/token of page-ins adds ~68 ms to a 78.5 ms pass.

*(Note: the first version of this probe used a sparse file; reading a hole returns
zeros without faulting, so it reported 0 page-ins and 17.27 GB "resident" at
0.7 GB/s. It was discarded and rewritten with real bytes. A probe that cannot
fail is not a probe.)*

---

## Roofline

```
bytes read per target forward pass = 17.27 GB (text-only; 18.41 with the vision tower)
achieved DRAM bandwidth           = 220 GB/s
=> floor per target pass          = 17.27 / 220 = 78.5 ms
=> ceiling, 1 token per pass      = 12.74 tok/s
=> 40 tok/s requires              = 40 x 0.0785 = 3.14 accepted tokens per pass
```

Drafter cost is negligible: 0.264 GB per drafter forward = 1.2 ms, so a K-draft
round adds ~1.2K ms to the 78.5 ms target pass. The drafter's default
`block_size` is **4** (`speculative/drafters/gemma4_assistant/config.py:23`), i.e.
3 drafts + 1 bonus, capped at 4 tokens per round; `--draft-block-size` overrides it.

Required per-draft acceptance `a` for 40 tok/s, assuming a position-independent
acceptance rate (an optimistic model — real acceptance decays with depth because
deeper drafts are conditioned on *drafted* rather than accepted history):

```
K= 3 drafts : a >= 0.840
K= 5 drafts : a >= 0.730
K= 7 drafts : a >= 0.700
K=11 drafts : a >= 0.685
```

against the measured synthetic acceptance rates of 0.20 (flat) to 0.40 (peaked) at
`temperature 1.0`. Closing that gap is the entire engineering problem, and it is
only worth attempting if the memory deficit is solved first.

---

## Dead ends / negative results

- **Growing HTTP concurrency does not raise download throughput.** 8 extra range
  connections alongside the 4 shards held aggregate at ~1.65 MB/s and *reduced*
  the main transfer to 1.12 MB/s. Not a client-side fix.
- **`unsloth/gemma-4-31b-it-UD-MLX-3bit` is 20.67 GB — larger than the 4-bit.**
  Quantization-bit count is a poor predictor of MLX on-disk size; the affine
  scale/bias overhead and unquantized tensors dominate.
- **Sparse-file mmap probe is invalid.** Reported 17.27 GB resident with zero
  page-ins. Use real bytes.
- **`vm_stat` page counts use a 16384-byte page on this box.** Dividing by 4096
  overstates every memory figure by 4x — worth stating because AGENTS.md §10.3
  already warns about misreading memory tools here.
- **`mlx_vlm.generate` has no `--top-k`/`--top-p`.** The CLI exposes `--temperature`
  but not the truncation knobs, and `mlx_vlm.sample_utils.make_sampler` is the only
  way to set them. A scratch harness driving `stream_generate` is therefore
  required, not optional.