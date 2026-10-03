# A2 — is `decode_state_mcg` reducible?

**Verdict: NO.** The 4 ops per code cannot be reduced to 3 from MSL, cannot be
reduced to 2 at all, and the one pool that *is* real — the `AND`+`XOR` pair, a
genuine single `LOP3` — is (a) unreachable from source on this toolchain and
(b) worth, on the only measurement that exists, **zero**.

Read-only probe. No model loaded, no GPU touched, no `gpuq`, no kernel
dispatch. Deliverable file:
`crates/engine/tests/probe_dequant_a2_algebra.rs` (18 tests, all passing,
`cargo test --release --test probe_dequant_a2_algebra`).

---

## 0. Provenance and the two numbers I am allowed to trust

| input | value | source |
|---|---|---|
| live kernel | `linear_gemv_ishizuki_k3_colmajor_2col_sk2`, `linear_repacked.metal:2407-2547` | brief |
| `extract_v_mcg_32` | `linear_repacked.metal:53-64` | read |
| `decode_state_mcg` | `linear.metal:121-126` | read |
| engine reference | `exl3.rs:165-179` (`decode_three_inst`) | read |
| per-trip slot model | `probe_reduction_k04.rs:209-216` | read |
| measured 3 bpw rate | 203 GB/s, 16.05 tok/s | brief |
| Metal toolchain | **absent** — `xcrun -sdk macosx metal` → `missing Metal Toolchain` | run |

**Every instruction count below is derived from source structure. None has ever
been observed in AIR.** That is not a caveat I can wish away: the file that owns
the 9.375 instr/weight figure fits its own cross-check with a **tolerance of
1.0** (`probe_reduction_k04.rs:240`), and its own arithmetic actually yields
**9.53**, not 9.375. Nobody has disassembled anything.

I independently re-counted `extract_v_mcg_32` from the source text rather than
trusting the constant: **10 per-code shifts + 8 masks + 2 ors = 20 extraction
ops, + 8×3 integer decode ops + 8 fp16 adds = 32. Total 52.** That agrees with
`probe_reduction_k04.rs:210`'s `extract = 52.0` from an independent direction, so
52 is solid.

---

## 1. The ranked table

Ops/code is the decode's own four operations (`IMUL`, `AND`, `XOR`, fp16 `HADD2`).
Projections apply `rate ∝ 1/slots` to the 305-slot base trip. **Read the caveat
column before any number in the tok/s column.**

| # | technique | ops/code | bit-exact | reachable from MSL | slots/trip | GB/s | tok/s | caveat |
|---|---|---|---|---|---|---|---|---|
| — | **baseline (live)** | **4** | — | — | 305 | 203.0 | **16.05** | measured |
| 1 | `AND`+`XOR` → one `LOP3` | 3 | **yes** | **NO** | 273 | 226.8 | 17.93 | +1.88 tok/s *if* the compiler fuses; unverified, and unmeasured |
| 2 | extraction `& 0xffff` → bit-field extract | 4 | yes | NO | 273 | 226.8 | 17.93 | needs a `BFE`-class op; Apple has none documented; not even MSL-visible as one |
| 3 | **1 + 2 together** | 3 | yes | NO | 241 | 256.9 | 20.31 | **projects above the box's 250 GB/s ceiling → the model is falsified** |
| 4 | byte-split LUT (2×256 in constant memory) | 0 + 2 loads | yes | n/a | — | 23.4 | 1.9 | **dead twice**: not separable (§4), and 4 B/code is 8.7× *below* today |
| 5 | 16-bit LUT in a device buffer | 0 + 1 load | yes | n/a | — | 23.4 | 1.9 | dead: 10746 distinct outputs ⇒ ≥14-bit index ⇒ 64 KB vs a 4 KB `constant` budget |
| 6 | reformulate the hash (smaller multiplier / shift-add / dot4) | 4 | yes | yes | 305 | 203.0 | 16.05 | **no cheaper form exists**: 1-op impossible (§6), shift-add is 19 pairs vs `IMUL` 1 |
| 7 | SWAR / 2 codes per instruction | 4 | yes | n/a | — | — | — | **impossible**: 2 × 26 = 52 live bits > 32 (§7) |
| 8 | move the FMA into fp16 `half2` | 4 | **NO** | yes | ~241 | 256.9 | 20.31 | rejected: fp16 accumulation over K = 5376 is quality loss; AGENTS.md §8 forbids it |
| 9 | `IMUL`+logic fusion (`IMAD` w/ logic modifier) | 3 | yes | NO | 273 | 226.8 | 17.93 | no such fusion exists on any known ISA; NVIDIA's `IMAD`-LOP3 fuses the *addend*, not a multiply |
| 10 | decode to fp16 weights offline | 1 (FMA) | yes | — | — | 46.9 | 3.7 | **fatal twice**: 64.2 GB vs a 25.8 GB box, *and* 4.33× slower (§8) |
| — | *reference*: `_nod` (decode deleted) | 0 | NO | yes | 177 | 349.8 | 27.66 | **MEASURED 26.7 % SLOWER.** The model says 27.66; reality says worse. This one line is the whole verdict. |

### Two corrections to the brief's arithmetic

**(a) The pool is 31.5 %, not 21 %.** The brief's 21 % is the decode's
`AND`+`XOR` only (64 of 305 slots = 21.0 %). It omits the eight `& 0xffff` masks
per `extract_v_mcg_32` that window each code — 32 more slots. **The whole logic
pool is 96 of 305 = 31.5 %**, and it is two equal halves, not one 21 % and one
invisible remainder. For scale: extraction 26.2 %, decode-as-a-whole 42.0 %,
FMA 21.0 %, loads+converts 7.9 %, addressing+bookkeeping 3.0 %.

**(b) The inherited "203 → 217 GB/s → 18.3 tok/s" is not reproducible.** The same
arithmetic from the repo's own slot model gives **226.8 GB/s / 17.93 tok/s** for
LOP3 fusion. The 217 figure does not fall out of any count in the tree. Recorded
so nobody re-derives it and thinks two agents agree.

---

## 2. Q1 — `LOP3`: one instruction, proved; unreachable, proved

**It is exactly one `LOP3`:** `LOP3(p, 0x8fff8fff, 0x3b603b60, 0x6a)` — the mask is
**operand b**, the constant is **operand c**, and `0x6a` is the immediate for
`(a & b) ^ c` under the `(a<<2)|(b<<1)|c` convention. Proved over the whole 2^32
domain by bit-parallel linearity (all 4 input combinations at each of 32 bit
positions) plus an exhaustive 2^16 sweep plus 2^20 deterministic pseudo-random
vectors.

`LOP3` computes an arbitrary three-input boolean function, and `f` is a function
of exactly three inputs (one variable, two constants). That is the whole reason
it is one instruction — an **arity** argument, not a constant.

**It is unreachable from source.** MSL has no `lop3` builtin. Neither
`linear.metal` nor `linear_repacked.metal` contains the string. The only route is
the compiler's own AND+XOR logic-fusion pass, and with no Metal toolchain there
is no AIR to disassemble and no way to check whether Apple's backend does it.

**Two mistakes I made and the tests caught** (both are the kind a reviewer would
repeat, so they are recorded in the probe's doc comments):

- I first asserted the folklore immediate `0xCA`. Under this file's convention it
  is `0x6A`. The folklore value uses NVIDIA's reversed operand order. Asserting a
  remembered constant was the error; the arity argument is the claim.
- I first passed `b = 0xffffffff`, i.e. tried to **bake the mask into the
  immediate**. It cannot be: the immediate is an 8-bit truth table over three
  *inputs*, and the mask is one of those inputs. Caught on the all-ones pattern.

**The multiply does not fuse with the logic.** `IMUL` → `AND` → `XOR` has no
known fusion on any documented ISA. NVIDIA's Volta+ `IMAD` takes a `LOP3`
operand mode, but that fuses the *addend* with a logic op — it does not fold a
32-bit multiply into one. Apple's ISA is proprietary and MSL exposes nothing, so
this is unknown rather than false; either way it is not reachable.

**What LOP3 would buy if the compiler did it:** 4 ops → 3 per code, chain depth
4 → 3, and **all 8 independent chains per extract preserved**. That last property
is the whole point, and it is what §3 is about.

---

## 3. Q2 — the ILP reconciliation, which is the real finding

### The existing evidence

| measurement | value | where |
|---|---|---|
| decode body (`mul`/`xor`/`half2`-add) removed, on `contig4` | **−2.3 %** (free) | `spec/decode-19tok-repacked-250gb/implementation_notes.md:1225` |
| `decode_state` stubbed to `float(s & 1u)` (`_nod`) | **−26.7 %** | `implementation_notes.md:488-489` |
| `no decode arithmetic at all`, old `linear.metal` family | 163.0 → **134.3** GB/s (−17.6 %) | `linear.metal:96-106` |

The standing reading is "the arithmetic is free; don't try to make the decode
cheaper" (`linear.metal:102-106`).

### Why that reading does not transfer — and this is the new result

Both `_nod` and `contig4` extract a lane's code with
`uint(A >> (64u - rad))`, `rad = i * KA + 16u`: a **64-bit barrel shift with a
per-element varying amount**, three ALU ops per code on a 32-lane core, on a
long data-dependent chain. The decode's 32 integer chains per trip were
**accidentally supplying the ILP that hid that shift**. Delete them and the shift
stalls. That is why the sign is negative.

**The live kernel does not have that shift.** `extract_v_mcg_32` builds `A_hi`/
`A_lo` from *two 32-bit loads* using the single lane-dependent amount `sh0`, which
is loop-invariant and hoisted; then every per-code window shift is a
**compile-time literal** (`A_hi >> 16`, `>> 13`, `>> 10`, `>> 7`, `>> 4`, `>> 1`,
plus two `<<`/`>>`/`|` wrap cases).

So the live kernel **already contains the 3.6× lever** that
`implementation_notes.md:1248-1255` identifies as the whole prize — "make `rad`
compile-time" — and that lever has already been taken. The measured partition
(63.9 % barrel shift / 8.5 % load / 27.6 % neither) describes a kernel the engine
does not run.

Verified structurally, not argued: the probe asserts the three source facts
(`(64u - rad)` present in `contig4_load_alu.metal` and in `linear.metal`'s
`exl3_decode8`; every per-code window shift in `extract_v_mcg_32` a literal in
1..=31; the hoisted `A_hi = (wa << sh0) | (wb >> rsh0)` present).

### The reconciliation, stated as the answer to "does reducing ops help?"

| intervention | ops removed | chains removed | chain depth | predicted | measured |
|---|---|---|---|---|---|
| `_nod` | 32 of 32 in the decode | **32 → 8** | 4 → 2 | much faster | **26.7 % slower** |
| LOP3 fusion | 8 of 32 (the AND+XOR) | **none** | 4 → 3 | faster | **unmeasured** |

**They are compatible.** `_nod` removed ops *and* the only available ILP filler,
so it is a measurement of ILP starvation, not of arithmetic cost. LOP3 fusion
removes an instruction while keeping every chain and shortening it — the case
where reducing ops *should* help. The two results do not contradict; the
`−26.7 %` bounds only the intervention it measured.

**But the honest answer to "is it worth it" is still probably no**, and the
reason is line 1225: the decode body's entire contribution on the only kernel
where it has been isolated was **−2.3 %**, i.e. free. If that transfers to the
32-bit kernel, one op out of four is worth roughly 0, and A2's verdict stands.

### The three arms that would settle it (never run)

On the **live** `_2col_sk2`, identical loads, identical addresses, only the decode
body differs. Pre-registered, written before the run:

- **A** control: production, verbatim, 4 ops, depth 4.
- **B** `y = p ^ 0x3b603b60; v[i] = float(as_type<half>(y));` — 2 ops, 8 chains
  kept, depth 2. Separates "fewer ops" from "fewer chains". Garbage output is
  fine; `_nod` set the precedent.
- **C** `y = 0x30003000 | (p & 0x849F849F); v[i] = float(as_type<half>(y));` —
  2 ops, chains kept, and the op mix is AND+OR, the closest an MSL source gets to
  asking for a `LOP3`.

**Criterion:** `B/A ≥ 1.10` ⇒ issue bound, the pool is real, build the fusion.
`B/A ≤ 1.02` ⇒ the pool is free here too, "no" is confirmed by measurement. The
window is 8 % wide against a measured 1.13× decode spread on this box, so it is
decidable — and the gap between the two thresholds is asserted in the test so it
cannot silently collapse.

---

## 4. Q3 — the LUT: dead on separability, on width, and on traffic

**Not separable across the code's bytes. Refuted by a 4-line witness.** The
multiply is what does it: `(lo + hi·2⁸)·C = lo·C + (hi·C)·2⁸`, and `lo·C`'s high
bits collide with `(hi·C)`'s low bits. Codes `0x0001` and `0x0101` share byte 0
and give products `0xCBAC1FED` and `0x668E1DBD` — the low 13 bits, which the mask
keeps and which become the low fp16's mantissa, differ.

Systematically: **256/256 low bytes actually depend on byte 1, and 256/256 high
bytes actually depend on byte 0.** Neither one-byte LUT direction exists.

**No narrower index exists either.** Measured: **10 746 distinct f32 outputs over
65 536 codes.** A LUT therefore needs ≥ 10 746 entries ⇒ a **≥14-bit index** ⇒
≥ 16 384 fp32 entries = **64 KB**, against Metal's **4 KB** `constant` budget
(16× over). There is no admissible width at all, and that is before the
separability argument.

**And in a device buffer it is fatal on traffic**, independently: 4 B per code
replaces 0.375 B of useful weight. At the box's ~250 GB/s ceiling that is
**62.5 Gcodes/s = 23.4 GB/s of useful weight, 8.7× below today.** An fp16 LUT
(2 B/code) gives 46.9 GB/s, still 4.3× below. This is arithmetic, not a forecast.

**One genuine sub-result:** 10 746 distinct outputs out of 65 536 is not
injectivity, so the map has real redundancy — it is just not the *kind* that a
LUT can exploit, because the redundancy is not aligned to any subset of input
bits (which is exactly what the byte-dependence test shows).

---

## 5. Q4 — reformulating the hash

**No cheaper exact form. Three separate closures:**

1. **The multiply is already optimal.** `0xCBAC1FED` has popcount 20 and spans
   the full 32-bit width (highest set bit 31), so a shift-add decomposition needs
   **19 shift/add pairs** against a hardware `IMUL`'s **1**. `IMUL` is not merely
   better, it is 19×.
2. **The `AND`+`XOR` cannot be one MSL op.** Exhaustive over all 65 536 inputs,
   six candidate single-op spellings, all of them partial coincidences and none
   exhaustive: `p & K` 0/65536, `p | K` 1024/65536, `p ^ K` 8192/65536, `p + K`
   256/65536, `p − K` 0/65536, `p · K` 0/65536. So **2 is the floor in
   MSL-visible ops, and the source already spends exactly 2.**
3. **The tempting AND+OR rewrite is arithmetically false.** The correct GF(2)-
   affine decomposition of `(p & A) ^ B` is
   `(B & ~A) | (A & B & ~p) | (A & ~B & p)`
   = `0x30003000 | (0x0b600b60 & ~p) | (0x849F849F & p)` — **three** ops. The
   two-op version `(B & ~A) | (A & ~B & p)` silently drops the `A & B & ~p` term
   and is wrong on **all 65 536** codes. **I asserted this one in a draft of the
   probe and its own exhaustive loop rejected it on every code.** It reads as
   obviously correct and it is the exact mistake a reviewer would make.

**fp16 headroom: there is none left, and what is there cannot be taken.**

- `float(h.x + h.y)` is already **one** fp16 add plus a free widening convert.
  Widening first and adding in f32 is *more accurate and not bit-exact* — it
  differs on > 1000 of 65 536 codes, asserted as a sabotage. Rejected.
- The `x` operand is **already** `device half` in the live kernel
  (`linear_repacked.metal:2465`), so the loads are half-width already; only the
  widening convert costs, 16 slots/trip.
- Moving the FMA down into `half2` would save 32 slots/trip (→ 20.3 tok/s in the
  model) and is **rejected**: it accumulates in fp16 over K = 5376, which is a
  quality loss and is forbidden by AGENTS.md §8.
- Recorded as a format property worth knowing: the mask and the constant fix bits
  15/14/13 of **each** half, so the decode output is *always* a sum of two
  negatives in (−1, −0.25]. No denormal or overflow concern, and no free wider
  substitute for the add.

---

## 6. Q5 — SWAR / bit-slicing over codes: impossible

**Register width.** The mask consumes **26 of 32 bits** per code (popcount of
`0x8fff8fff`), so one code's result just fits — that is the status quo — and
**two codes need 52 live bits against a 32-bit register**. Asserted for 2, 3, 4,
5, 6, 7 and 8 codes.

**The tempting form has carries, not depth.** Pack codes `i` and `j` as
`(j<<16)|i` and hope one `IMUL` yields both products:
`((j<<16)|i)·C = i·C + (j·C mod 2^16)·2^16 (mod 2^32)`. The **low** half is
correct; the high half is *bits 16–31 of `i·C`* **plus** `j·C mod 2^16` — it
depends on **both** codes. Checked exhaustively for all 65 535 non-degenerate `j`.
At best you recover one code per register, which is what the kernel does.

**Second, independent reason:** the codes are 3 bits apart in the packed word, so
a pair is a 6-bit field straddling five different alignments. Any SWAR layout
needs a real repack at load, which is the reverted `9c85e9a`.

**No integer dot product to widen into.** MSL exposes vector *fp* `dot` and no
`dp2a`/`dp4a`; either would compute a different function from a 32-bit multiply
by a constant. Asserted absent from both kernel sources.

---

## 7. Q6 — fp16 weights offline: fatal twice, independently

- **Capacity.** 12.039 GB packed at 3 bpw ⇒ 32.104e9 weights/token. At 2 B each
  = **64.2 GB resident** against a **25.77 GB** box (24 GiB). **2.49× over.**
  AGENTS.md §10 already records that two concurrent instances at 11.7 GB
  exhaust this box.
- **Bandwidth.** The kernel is *not* currently bandwidth bound — that is the
  point of `WHERE-THIS-STANDS.md:82-89` — but at 2 B/weight it *would* be:
  125 Gcodes/s at the 250 GB/s ceiling = **46.9 GB/s of useful weight, 4.33×
  slower than today**, ≈ 3.7 tok/s.

Either alone is fatal. This was the one question in the brief where the expected
answer was assumed rather than checked; it is now a number.

---

## 8. Why the model is not a forward predictor — and the falsification

The "1 + 2 together" row projects **256.9 GB/s**. `WHERE-THIS-STANDS.md:36` records
that **no rate on this box has ever exceeded ~250 GB/s**, on any kernel, at any
bit rate. A bit-exact op-count model that predicts a rate the hardware has never
once produced is not a forward predictor, and every tok/s in the table inherits
that. The test asserts the overshoot deliberately, so the day someone claims
20.3 tok/s from a bit-exact instruction trim, this file says the model had already
been falsified by its own output.

This is the same class of error as the repo's recorded failures
(AGENTS.md §3.4): a coverage claim with nothing behind it. **The honest summary
of the whole exercise is that the projected +1.88 tok/s is a model output, not a
result, and the only measured datapoint in this pool says zero.**

---

## 9. Coverage discipline — the failures this file actually produced

Per AGENTS.md §3.4, a coverage claim is not a claim until the code has been seen
to fail. Six of this file's own assertions failed **against my own draft
arithmetic** before passing:

| # | my draft error | how it failed |
|---|---|---|
| 1 | asserted LOP3 immediate `0xCA` | got `0x6A` — the folklore value uses the reversed operand order |
| 2 | `LOP3(p, 0xffffffff, xor, …)` — baked the mask into the immediate | failed on the all-ones pattern; the mask is operand b, not part of the 8-bit table |
| 3 | asserted the affine AND+OR rewrite | wrong on all 65 536 codes; it drops `A & B & ~p` |
| 4 | `MCG_MUL.count_ones() == 21` | actual 20 |
| 5 | shift census `(6,4,8,2)` | actual `(6,2,2,8,2)` — I counted the hoisted setup shifts as per-code |
| 6 | "find the first `p` where a 1-op spelling matches" | every candidate matches at `p = 0`, so six false survivors; `p\|K` also equals `p^K` at `p = 1` because bit 0 of `MCG_XOR` is 0 |

Plus one explicit sabotage of a *passing* test, to prove the source-census guard
works: changing the expected extraction census from `(6,2,2,8,2)` to `(6,2,2,8,3)`
produced `assertion 'left == right' failed: the extraction's op census moved`.

Two of the six (#3 and #6) are mistakes a reviewer would plausibly repeat. Both
are now recorded in the probe's doc comments, not just here.

---

## 10. What I would do next, in order

1. **Run the three arms in §3.** ~30 min of GPU, no code changes to production,
   and it converts this entire file from arithmetic to a measurement. This is the
   only item on the list that can change the verdict.
2. **Do not build a `LOP3` fusion pass.** It is unverifiable on this box (no
   toolchain), unreachable from MSL, and the pool it targets is worth 0 in the one
   measurement that exists. If arm B comes back ≥ 1.10 that reverses; until then
   it is negative-return work.
3. **Stop quoting "21 %".** It is 31.5 %, it is two pools not one, and one of the
   two (the eight `& 0xffff` masks per extract) is a *bit-field-extract* question
   rather than a decode question and has never been asked.
4. **Do not re-derive 203 → 217 GB/s.** The repo's own model gives 226.8.

---

## Commands

```bash
# this probe — CPU only, no GPU, ~10 s
cargo test --release --test probe_dequant_a2_algebra -- --nocapture --test-threads=1

# the arms that would settle it (NOT run here; needs the GPU)
# see the_discriminating_measurement_is_named_with_its_pre_registered_reading
```

**Files touched by A2:** one new file,
`crates/engine/tests/probe_dequant_a2_algebra.rs`. Nothing else was modified.
No model was loaded, no `gpuq` was run, no GPU microbenchmark was executed.