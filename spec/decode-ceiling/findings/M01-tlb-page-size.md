# M01 — TLB pressure and VM page size

**Verdict: clean negative. TLB is not a limiter, and `PAGE_ALIGN = 4096` is
correct as written. Change nothing.** The only defect found is a missing
`is_nil()` check on a Metal buffer that the evidence says is never nil.

One premise in the brief is wrong and is corrected below: **there is no
`lock_range` and no `mlock` in this repository.**

Investigated at `075ed65` (the brief said `5e8ab73`; the working tree is 14
commits past that reference).

---

## Summary

Six questions, answered with arithmetic rather than measurement, because the
arithmetic already settles them.

1. **TLB working-set pressure is nil.** The stream is *sequential* with a reuse
   distance of 12.04 GB, so the hardware prefetcher covers it. There is no
   working set to miss on. Compounding this, the best local datapoint says the
   same TLB sustains **253 GB/s** on `lm_head` — so the TLB can carry at least
   31% more than it carries today.
2. **`PAGE_ALIGN = 4096` is correct, and is load-bearing.** It is a *lower bound*
   for Metal's `newBufferWithBytesNoCopyLength:`, which requires page-aligned
   memory. Aligning down by 4095 is correct; changing to 16384 changes nothing
   about TLB behaviour and would add up to 4095 bytes of slack per tensor.
3. **No `lock_range`/`mlock` exists.** Questions 3's premise is stale. The
   rounding arithmetic is recorded anyway for whoever implements F6.
4. **No huge pages exist on Darwin for user mappings.** `MADV_HUGEPAGE` is not
   in the Darwin `madvise` set; 2 MB superpages are reachable only through
   `vm_allocate` with an explicit flag, never as advice on an `mmap`.
5. **No NUMA on Apple Silicon.** One memory controller, physically uniform. Page
   colouring is not controllable from userspace.
6. **Buffer size cannot help.** Reuse distance is 12.04 GB regardless of how the
   weights are carved up. Changing segment size moves bytes between buffers, not
   the number of pages touched.

---

## Evidence

### `PAGE_ALIGN` and `no_copy_view`

- `crates/engine/src/weights.rs:1105` — `const PAGE_ALIGN: usize = 4096;`
- `crates/engine/src/weights.rs:1061` — `let aligned = start & !(PAGE_ALIGN - 1);`
- `crates/engine/src/weights.rs:1069-1075` — `new_buffer_with_bytes_no_copy(aligned, end - aligned, StorageModeShared | HazardTrackingUntracked)`
- `crates/engine/src/weights.rs:797` — `raw_view`, the sole caller of `no_copy_view`
- `crates/engine/src/model.rs:3335-3338` — the `delta` offset reaches both the
  GPU kernel and the CPU reader; documented as a correctness requirement.

### Platform facts, verified at runtime

| Fact | Value | Source |
|---|---|---|
| VM page size | **16384** | `sysconf(_SC_PAGESIZE)`; `vm_stat` header; `sysctl hw.pagesize` |
| Physical memory | 25,769,803,776 | `sysctl hw.memsize` |
| CPU cores | 14 (10 P) | `sysctl hw.ncpu`, `hw.perflevel0.physicalcpu` |
| `vm.user_wire_limit` | 19,585,050,869 | `sysctl vm.user_wire_limit` |

### Checkpoint offset census (headers only, no model loaded)

From `probe_tlb_page_size.rs::checkpoint_tensor_offsets_are_not_page_aligned`:

```
2422 tensors, 16028050244 payload bytes, 2419 not 16384-aligned;
trellis 411/411 misaligned
```

**99.9% of tensors start mid-page.** safetensors aligns data offsets to 8 bytes.
This is the fact that makes `PAGE_ALIGN = 4096` necessary rather than
incidental: a page-aligned round is the only way to hand Metal a legal
no-copy buffer, and 8-byte tensor offsets guarantee you cannot get one by
alignment at all.

### Darwin huge-page surface

`libc-0.2.189/src/unix/bsd/apple/mod.rs:2566-2576` — the complete Darwin `madvise`
set is `NORMAL, RANDOM, SEQUENTIAL, WILLNEED, DONTNEED, FREE,
ZERO_WIRED_PAGES, FREE_REUSABLE, FREE_REUSE, CAN_REUSE, ZERO`. **No
`MADV_HUGEPAGE`.**

`libc-0.2.189/src/unix/bsd/apple/mod.rs:2196-2215` — `VM_FLAGS_SUPERPAGE_MASK`
and `SUPERPAGE_SIZE_2MB` exist, but they are `vm_allocate` flags on an explicitly
sized allocation, not advice applied to an `mmap`. There is no user-reachable
path from a 12 GB file mapping to 2 MB pages on Darwin.

### The absence of `mlock`

`grep -rn "mlock\|munlock\|lock_range\|madvise\|MADV_" crates/engine/src/` returns
**zero** hits. The only `madvise` use is in
`crates/engine/examples/bench_residency.rs:115,123`, an example harness
(`MADV_DONTNEED` / `MADV_WILLNEED`), not the engine.

`crates/engine/src/gpu_lock.rs` is a **file** lock — `flock(LOCK_EX)` on
`~/.kaplumba/gpu.lock` at line 300. It has nothing to do with memory residency,
and the module doc says so explicitly at lines 9-13.

This is already known and recorded:
`spec/memory-swap-cliff/plan.md:8-11` states "there is no `mlock` in this
repository", and `implementation_notes.md:234` records the same grep. Question 3
of the brief is answered in that spec's F6, which is where per-tensor `mlock`
is *proposed*. My rounding arithmetic below is input to F6, not a finding about
existing code.

---

## The arithmetic

### TLB working set

Per token the engine streams **12.04 GB** (10.983 GB layers + 1.057 GB `lm_head`),
touching, once each, with no reuse:

```
12.04e9 / 16384 = 734,864 pages
12.04e9 /  4096 = 2,939,453 pages
```

The number is large in absolute terms and **irrelevant**, for three reasons.

**1. Sequential, so prefetcher-covered.** Layer trellis is read in ascending
layer order, and within a layer the trellis walk is monotonic in the weight
address. A 12.04 GB contiguous read is the *best* case for an L2 prefetcher and
for the ATS. There is no random-access working set to thrash a TLB with. A TLB
limiter requires high reuse-distance *and* random order; this workload has high
reuse-distance *and* sequential order.

**2. The measured ceiling exceeds what a TLB here delivers.** The strongest
local datapoint: the `lm_head` GEMV hits **253 GB/s** (`lm_head.trellis` is a
zero-copy mapping, `weights.rs:585`). 253/193 = 1.31, so the same TLB, the same
GPU, the same 62.3 ms step already sustains **31% more bandwidth** than the
layer path achieves. A resource that can deliver 253 GB/s is not the thing
holding decode at 193 GB/s.

**3. Demand misses are not the cost anyway.** 734,864 translations/token over
62.3 ms is **11.8 M translations/s**. Even at a pessimistic 100 cycles per walk
on one 3.4 GHz core that is 4 Gcycles/s — and Apple Silicon performs these
walks in the ATS with multiple parallel walkers. This is roughly 0.1% of the
memory-order-crossing capacity of the machine. Filling a 734K-entry table 11.8 M
times/s is a few percent of one core's walk throughput, spread across the ATS.

**Conclusion: the TLB costs ~0.3% at most, and cannot explain a 24% gap to
`DRAM_PEAK`.** The brief's own framing already points here: GB/s scaling exactly
inversely with bit-width at 330-556 Gcodes/s is a code-throughput signature. TLB
pressure would not scale with bit-width at all — it scales with bytes, which
fall as bit-width rises. The observed fall in GB/s with *more* bits is the
opposite of a byte- or page-bound signature.

### Why 4096 is correct and must not become 16384

`no_copy_view` rounds the buffer base *down* so the tensor's real start sits at
`delta` inside the buffer. Two constraints on that constant:

- It must be `<= ` the VM page size, because Metal's no-copy path takes a
  page-aligned pointer. Both 4096 and 16384 satisfy this.
- It must be `<=` the safetensors offset granularity (8 bytes) **rounded up to a
  power of two**, or the rounded base could fall *above* the tensor start. 4096
  >= 8, so safe.

Changing it to 16384 would: leave the translation count **identical** (proven
below), and push `delta` up to 16383 instead of up to 4095 — for a 2.82 GB
embedding, that is a slightly worse access pattern for the first touch of the
window, for zero benefit.

**Proof that the translation count is invariant.** `page_align_does_not_change_pages_translated`
asserts, over five realistic (offset, length) shapes including the 183 MB
layer-trellis and 1.06 GB `lm_head` magnitudes, that the page count of the
resulting buffer is **equal** under a 4096 round and a 16384 round:

```
offset 3502008:  4 KB -> 11170 pages,  16 KB -> 11170 pages
```

Rounding the base down can only *add* pages relative to the unaligned span, never
remove them. The count is a property of the byte range, not the base. **There is
no TLB delta to be had here, in either direction.**

### Sub-page aliasing (brief question 2)

`aligned = start & !4095` can leave the base **not** page-aligned when the
tensor's file offset is not a multiple of 16384 — which, per the census above,
is 2419 of 2422 tensors. `four_k_round_yields_a_non_page_aligned_base` pins
this.

This is **safe but non-ideal**, and it is safe for a reason worth stating
precisely, because it is not obvious:

- The mapping is `PROT_READ | MAP_PRIVATE` (`weights.rs:1019-1020`) and is
  **never written**. No two `MTLBuffer`s can be written through the same page, so
  no aliasing hazard exists. `HazardTrackingUntracked` is correct for exactly
  this reason.
- The excess is at most 4095 bytes per tensor, ≤ 0.03% of a 183 MB trellis, and
  it is **in-bounds of the mapping** — `aligned >= base` always, since mmap
  returned a page-aligned address and we only round down within it.

Verified in `mmap_returns_vm_page_aligned_but_not_necessarily_more`: mmap
returned a 16384-aligned address on this box, so a 4096-round of it can never
descend below the mapping's own guarantee.

So: real, bounded, harmless. Not worth a `delta`-consuming refactor.

### `mlock` rounding (question 3, for F6)

Since no `mlock` exists, the honest answer to "is 4 KB rounding wrong" is that
the question has no current answer. For whoever implements
`spec/memory-swap-cliff` F6:

Darwin's `mlock` takes a page-granular lock. A range that is not a multiple of
16384 still pins the whole enclosing page, so **rounding by 4096 saves nothing**
— the kernel rounds up regardless. Pinned in `mlock_rounds_out_to_whole_vm_pages`:

```
worst over-lock = 16383 bytes per call
F6 shape: 60 layers x ~7 tensors = 410 calls
worst extra    = 410 * 16383 = 6,717,030 bytes  (~6.4 MiB, 0.06% of the wire limit)
payload        = ~11 GB
```

6.7 MB of rounding slack against an 11 GB payload is **three orders of
magnitude below the thing F6 is deciding.** F6's byte budget is not sensitive to
this constant. Whatever value F6 picks, it should pick 16384 for honest
reporting, not 4096 — but no decision rides on it.

Worth restating from `spec/memory-swap-cliff/plan.md:82-100` because it bears on
the 4 KB vs 16 KB question by analogy: `mlock` makes pages **wired**, and at
13.638 GiB pinned against a 19.58 GB `vm.user_wire_limit`, a wrong choice there
costs GiB. A 4 KB-vs-16 KB rounding error costs MiB. Different magnitude
entirely.

### NUMA (question 5)

There is no NUMA on Apple Silicon. One memory controller, one DRAM bank
structure, physically uniform addresses. The analogue of the page-colouring risk
is physical-address interleaving across channels, which is fixed at memory
controller design and not controllable from userspace — there is no syscall that
requests a physical address or an interleave group. **Not actionable, and not
observable.** CPU and GPU share the same 20-core/24 GiB pool; M02 owns whether
CPU-side traffic competes with the GPU stream for DRAM.

---

## Conclusion

**TLB pressure is not a limiter and cannot be made into one.**

- Sequential access ⇒ prefetcher-covered, zero working set.
- The same TLB already delivers 253 GB/s on `lm_head`, 31% above the decode
  rate, so it is not the binding constraint at 193 GB/s.
- Translation rate is 11.8 M/s — a fraction of a percent of walk capacity.
- The observed GB/s-vs-bit-width anti-correlation is a code-throughput
  signature. A TLB or page-bound limiter would produce the *opposite*
  correlation, since fewer bytes means fewer translations.

**`PAGE_ALIGN = 4096` at `weights.rs:1105` is correct, not a latent bug.** It is
a *lower bound* for Metal's page-alignment requirement, which 16384 would also
satisfy, and it is the minimum that satisfies it. Changing it has **zero**
measured effect on translation load and would make `delta` up to 4x larger.

R2 should be redirected to code throughput (K-series) and to DRAM efficiency
(M05, the true achievable peak). The memory-side ceiling is set by what the
gather can sustain, not by how the addresses are translated.

---

## Recommended change

**One change, and it is a robustness fix, not a performance fix.**

`crates/engine/src/weights.rs:1069-1075` — `no_copy_view` returns the buffer
from `new_buffer_with_bytes_no_copy` without checking it for `nil`. Metal
returns `nil` for an illegal no-copy pointer (including a misaligned one) rather
than trapping. Today that is harmless because 4096 alignment is accepted and
decode measurably works — but a future change to `PAGE_ALIGN`, or a Metal
release that tightens the rule, would surface as a **nil buffer passed into a
command encoder**, which is a crash or silent corruption deep in the encoder,
with no diagnostic pointing at `no_copy_view`.

```rust
// crates/engine/src/weights.rs, after the new_buffer_with_bytes_no_copy call
// at :1069 and before `Ok((buffer, delta as u64))` at :1076
if buffer.is_nil() {
    return Err(WeightError::Mmap {
        path: self.path.clone(),
        message: format!(
            "Metal rejected a no-copy view at {aligned:#x} for tensor offset \
             {offset} (len {len}); the buffer must be page-aligned and the \
             whole range must lie inside the {}-byte mapping",
            self.length
        ),
    });
}
```

**Expected effect on performance: exactly zero.** This is deliberate and I want
it stated plainly rather than dressed up as an optimisation. It costs one branch
per tensor, 411 times per model load. Its value is that it makes the one
constant this hypothesis examined *self-defending*: the next person who tries
`PAGE_ALIGN = 16384` gets a named error instead of a corrupt encoder stream.

I want to be explicit that **I am not recommending a `PAGE_ALIGN` change.** The
arithmetic above is the argument against it.

Two things I deliberately did **not** recommend, and why:

- **Superpaging the mappings.** Unreachable on Darwin for `mmap` (no
  `MADV_HUGEPAGE`), and unnecessary given sequential access.
- **Resizing `DEFAULT_SEGMENT_BYTES` (`metal.rs:33`, 512 MiB).** Would change
  allocation geometry, not the 12.04 GB reuse distance. Out of scope for this
  angle and unmeasured.

---

## Confidence and what I could not determine

**High confidence (arithmetic, not measurement):**

- `PAGE_ALIGN = 4096` does not change the translation count. Proven
  executably; confirmed by sabotage (see below).
- TLB is not the limiter. Three independent arguments, the strongest being that
  `lm_head` reaches 253 GB/s through the same TLB.
- No `mlock` exists. Zero grep hits; corroborated by
  `spec/memory-swap-cliff/plan.md:8-11`.
- No user-reachable huge pages on Darwin. Complete `madvise` set enumerated from
  the libc source in tree.
- 99.9% of checkpoint tensors are page-misaligned (measured from headers).

**Test-coverage claim, verified by sabotage** (per AGENTS.md §3.4). Both
sabotages were run and observed to fail with diagnosable messages, then reverted:

| Sabotage | Observed failure |
|---|---|
| `PAGE_ALIGN_IN_TREE` → 16384 | `assertion failed: the tree's PAGE_ALIGN moved; this probe's arithmetic assumed 4096` |
| `pages_spanned` rigged so a page-aligned base counts fewer pages (i.e. **the hypothesis made true**) | `page_align_does_not_change_pages_translated` FAILED at `probe_tlb_page_size.rs:153` |

The second is the important one: it confirms the load-bearing test actually
fails when the hypothesis under test is made true, rather than passing
vacuously. Both reverted; suite green (7 passed).

**Could not determine:**

1. **ATS walker count and real TLB depth.** Not exposed by any Darwin syscall,
   and no Apple documentation states them. My 11.8 M/s estimate uses a
   pessimistic serial-walk model to make the *bound* conservative; the real
   hardware is more parallel, which only strengthens the negative.
2. **Whether a 4096-aligned no-copy buffer is strictly conformant.** I inferred
   acceptance from the measured 16 tok/s decode, which is strong evidence but is
   not the same as a spec citation. This is exactly the gap the recommended
   `is_nil()` check closes. I did not create a Metal device — no GPU work per
   the brief.
3. **Actual L2 prefetch reach, and therefore real demand-miss count.** The
   734,864-per-token figure is a worst case that assumes *zero* prefetch
   coverage; the true figure is lower by the prefetch depth. Not measurable
   without PMU counters, which this box does not expose to userspace.
4. **Whether interleaving effects exist at all.** Apple does not publish the
   M4 Pro memory-controller interleave granularity. I can state only that no
   userspace knob exists — not that the effect is zero.
5. **Physical-page placement / DRAM residency of the mapping.** Out of scope and
   owned by `spec/memory-swap-cliff`; a cold `lm_head` page would present as
   bandwidth loss indistinguishable from TLB loss without PMU data, so if
   bandwidth numbers move unexpectedly, that confound should be ruled out first.

**Not measured on hardware, by design.** No model run, no `scripts/gpuq`, no
Metal device, no GPU work. Everything above is derived from the brief's measured
baseline, the source tree, the libc source in the cargo registry, and safetensors
headers. The recommendation's performance effect (zero) is an analytical claim.
