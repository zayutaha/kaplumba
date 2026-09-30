# Implementation Notes: Decode 19+ tok/s and 250+ GB/s via Repacked Weights

Running log for `spec/decode-19tok-repacked-250gb/`. Updated per task.

## Baseline & Context
- Target: 19+ tok/s decode, 250+ GB/s streaming bandwidth.
- Model: Gemma 4 31B (`~/.kaplumba/models/Gxma/Gxma`), EXL3 3.0 bpw.
- Measured baseline: ~11.2 tok/s on current Ishizuki 2-word kernel (`3230f24`).
- Physical DRAM theoretical peak: 273 GB/s (Apple M4 Pro).

## T1 & T2: Synthetic Memory Bandwidth Probe (Disproof of 244 GB/s Ceiling)
- Implemented `probe_bw_250gb.rs` and `bw_probe.metal`.
- Features:
  - Vectorized loads with `uint4` (16 bytes = 128 bits per thread per instruction).
  - High-entropy pseudo-random pattern tiling: every cacheline is non-zero and non-compressible, preventing hardware zero-line compression.
  - Concurrency: 40,960 to 81,920 threads.
  - Memory flags: `MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked`.
  - Checksum reduction across all thread outputs.
- Fail-first sabotage (T2):
  - Injected halved load count: normal checksum `0x739adf40`, sabotaged checksum `0xa8c277a0`. Checksum mismatch proved test can detect failure.
- Measured Results:
  - **2 GB Buffer:**
    - th=81920, tg=256: **251.7 GB/s median** (92.2% peak), **254.2 GB/s best**.
  - **6 GB Buffer:**
    - th=81920, tg=512: **254.0 GB/s median** (93.0% peak), **255.4 GB/s best**.
  - **12 GB Buffer (Full Model Weight Footprint):**
    - th=81920, tg=512: **250.5 GB/s median** (91.8% peak), **253.2 GB/s best**.
- **Conclusion:** Hardware DRAM streaming bandwidth CAN exceed 250 GB/s on Apple M4 Pro when 128-bit aligned vector loads (`uint4`) are used with high thread occupancy. The previous 239-244 GB/s ceiling was an artifact of 4-byte scalar loads and thread configuration.
