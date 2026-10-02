# tasks — mtp-40tps

- [x] **T0** Environment audit (disk, RAM, swap, versions, `~/.omlx` settings).
- [x] **T1** Determine the memory floor for an MLX 4-bit Gemma-4-31B target by
      reading the safetensors headers over HTTP range requests.
      → 17.27 GB text-only / 18.41 GB with vision.
- [x] **T2** Audit the mlx-vlm non-greedy MTP verification path
      (`speculative/common.py:135`, `speculative/mtp.py:182`) and prove the
      accept-on-equal rule distribution-exact on synthetic distributions.
- [x] **T3** Measure the box's real memory ceiling for a model-sized working set
      (MLX `mx.ones` residency probe + non-sparse mmap streaming probe) and the
      NVMe re-read rate that sets the thrash penalty.
- [ ] **T4** Acquire `mlx-community/gemma-4-31b-it-4bit` (18.41 GB). *In progress —
      ~1 MB/s upstream.*
- [ ] **T5** Build the scratch harness `mtp_bench.py`: per-token wall-clock
      intervals, `temp`/`top_k`/`seed`/`--draft-block-size` control, token-id dump,
      memory sampling, A/B interleave support.
- [ ] **T6** Baseline (R2): non-speculative decode, 1024 tokens, ≥3 interleaved runs.
- [ ] **T7** MTP (R3): same prompt/settings with the drafter.
- [ ] **T8** Block-size / KV-size / flag sweep (R5).
- [ ] **T9** Distribution-faithfulness test (R6).
- [ ] **T10** Coherence / multi-turn test (R7).
- [ ] **T11** Write-up: measured ceiling, roofline, acceptance rate, negative results.