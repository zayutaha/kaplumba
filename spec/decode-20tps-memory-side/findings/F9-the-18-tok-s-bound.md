# F9 — 18 tok/s needs 98% of a kernel that touches no memory. It is not reachable.

**Verdict: the ceiling is an *instruction* ceiling, not a bandwidth ceiling, and it is
much closer to where we already are than the 245 GB/s streaming figure suggested.
18 tok/s is out of reach. The realistic ceiling is ~17.2 tok/s, and that number is
derived, not measured — the honest measured bound is arm (G).**

## The arithmetic on 18 tok/s

```
18 tok/s                                    = 55.56 ms/step
less non-matvec (attention, norms, qk)       ~ 4.0 ms
matvec budget                                ~ 51.6 ms for 12.0395 GB
=> required matvec rate                      = 233 GB/s
=> required code rate                        = 621 Gcodes/s (at 3 bpw)
```

**Arm (G), the register-resident arm that performs the identical per-8-code instruction
stream with ZERO memory traffic and ZERO address arithmetic, measures 633.4–637.2
Gcodes/s.**

```
18 tok/s requires   621 Gcodes/s
arm (G) provides    634 Gcodes/s
=>  18 tok/s requires the real kernel to run at 98% of a kernel that
    reads nothing and computes no addresses.
```

That is not a tuning gap. A kernel that reads 12 GB per second from DRAM cannot be within
2% of a kernel that reads nothing.

## Why the streaming ceiling was the wrong number to aim at

`probe_bw_250gb` streams at **241–252.7 GB/s**, and for a long time the plan treated that
as the target. But the decode GEMV is not a streaming kernel — it also executes, per 8
codes:

| component | ops / 8 codes |
|---|---:|
| `extract_v_mcg_32` (window assembly + 8 field extractions) | ~18–23 |
| 8 × `decode_state_mcg` (mul, AND, XOR, f16 add, f16→f32) | 40 |
| 8 fp32 FMA | 8 |
| 2 weight loads + address arithmetic | ~6 |
| **total** | **~74** |

Arm (G) executes 70 of those per 8 codes and sustains 634 Gcodes/s. That is
`634e9 × 70 ≈ 4.4e13` lane-ops/s against a nominal issue capability of
`20 cores × 128 lanes × ~4.3 GHz ≈ 1.1e13`, i.e. **arm (G) itself runs at ~50% of nominal
issue** — it is dependency/occupancy-limited, not bandwidth-limited.

The streaming probe reaches 252 GB/s with essentially zero arithmetic. The decode GEMV
cannot, because it must also issue ~74 ops per 8 codes.

## Where we actually are against that bound

| | code rate | % of arm (G) |
|---|---:|---:|
| live kernels today | ~541–549 Gcodes/s (202–206 GB/s) | **86%** |
| arm (C): L1-resident weights, same instructions | 568 Gcodes/s | 90% |
| arm (G): register-resident, no memory, no addressing | 634 Gcodes/s | 100% |

**The entire remaining headroom, against a bound that cannot be reached, is ~16%.**

## A derived, more honest ceiling

Arm (G) is an upper bound by construction: it spends 7 ops per 8 codes accumulating
where production spends 8 FMAs, and it performs no address arithmetic at all. Adding back
those (`~74` vs `~70` ops per 8 codes, +6%) gives a more realistic ceiling:

```
634 Gcodes/s / 1.06  ~=  598 Gcodes/s  =  224 GB/s
matvec 12.0395 / 0.224 = 53.7 ms + 4.0 ms = 57.7 ms  =>  17.3 tok/s
```

**This 17.3 is a DERIVED figure, not a measurement, and it is marked as such.** It exists
to bound the target, not to promise it. The measured facts are: today 16.0 tok/s, arm (G)
634 Gcodes/s, streaming 252.7 GB/s.

| target | required code rate | % of arm (G) | reachable? |
|---|---:|---:|---|
| 16.5 tok/s | 571 Gcodes/s | 90% | plausible — needs most of the L1/DRAM term |
| 17.0 tok/s | 587 Gcodes/s | 93% | possible but there is no identified mechanism |
| **18.0 tok/s** | **621 Gcodes/s** | **98%** | **no** |
| 20.0 tok/s | 686 Gcodes/s | 108% | no — and also above the DRAM ceiling |

## What is left, and it is not a tuning gap

Every mechanism that could close a 14% instruction/latency gap has been measured and is
either exhausted or structurally blocked:

* **fewer load instructions** — +3.3% for a 4× cut (F7, measured)
* **wider per-lane window** — ~+1% (F6, analysed and confirmed by F7)
* **fewer loads in flight / more concurrency** — +6% at 32×, measured three times
* **constant shift amounts** — +2.5%, and structurally unreachable (a 32-bit-aligned
  window start needs 24 slots for 32 lanes)
* **decode arithmetic** — "removing the decode made every rate *slower*"
  (`decode-bandwidth-240gb`), so a cheaper decode is not the fix
* **every variant on the live dispatch** — F4, F5, F8, all negative

The op count itself is near the minimum for this data layout: 8 fields at a 3-bit stride
need ~8 shifts and ~7 masks, and arm (E) measured the masks as already free. **There is no
fat left in the loop.**

So the honest position is: decode is within roughly 10–14% of a bound it cannot reach,
every axis has been measured, and the remaining 14% is not attributable to any identified
mechanism. Reaching 18 tok/s would require changing the packed format (a repack of all 411
tensors, to lower ops-per-code) or changing the decode algorithm (verify-gated
speculative decoding, which raises the number of *tokens* per streamed weight rather than
the rate of streaming).