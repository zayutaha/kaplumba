#!/usr/bin/env python3
"""Generate the rate/alignment/codebook-specialised EXL3 decode matvec kernels.

Metal has no templates, so `linear_gemv_exl3` is emitted once per integer rate
(KA), per buffer alignment (ALIGNED), and per codebook SPECIALISATION. Three
axes are worth real throughput:

  KA        shift amounts and the ring size fold to constants.
  ALIGNED   a 4-byte-aligned buffer is read with one 32-bit load per word
            instead of two 16-bit loads. Arena weights always are; mapped
            checkpoint bytes are not (lm_head.trellis starts at % 4 == 2). Loads
            dominate the kernel -- measured 0.336 ms of 0.612 ms for q_proj -- so
            halving the load instruction count matters.
  CODEBOOK  the codebook is a per-tensor constant, and branching on it inside
            the eight-times-unrolled value loop cost 61.7% of the kernel. See the
            measurement table on `exl3_decode8_mcg` in linear.metal.

STATE COUNT, and why it is 16 + 16 and not 48.

A fully general [aligned][bits][codebook] grid is 2 x 8 x 3 = 48 kernels. Only
the MCG codebook is specialised, so the grid is 2 x 8 x 1 = 16 new kernels, and
the original 16 (runtime-codebook) kernels are kept as the fallback for the
`plain` and `mul1` codebooks. Total decode states: 32.

Justification for specialising only MCG, which is a reachability fact and not a
guess:

  * Every linear in all 60 decoder layers is constructed with
    `Exl3Rate::Integer(3)` + `Exl3Codebook::Mcg` (crates/engine/src/model.rs,
    the seven `load_quantized_linear` calls per layer).
  * `lm_head` is the only other tensor, and `Model::validate_current_graph`
    refuses to run unless it is `Exl3Rate::Integer(6)` + `Exl3Codebook::Mcg`.
  * The loader config validation likewise refuses any checkpoint whose
    `quantization.codebook` is not "mcg".

So MCG is the only codebook the engine can dispatch, and the plain/mul1 kernels
exist only for the unit tests and the reference comparisons. Leaving them on the
generic path costs nothing that is ever measured and saves 32 kernels of
compile time and code size.

Run from the repo root:  python3 scripts/gen_exl3_gemv.py
It rewrites the block between the EXL3 matvec banner and `kernel void
linear_gemv(` in crates/kernels/src/linear.metal. Edit the template below, not
the generated kernels.
"""
import re
from pathlib import Path

PATH = Path("crates/kernels/src/linear.metal")
BANNER = "// EXL3 decode matvec, shaped like the reference implementation."

# The NR0 axis: output tile-columns per threadgroup, i.e. how many independent
# weight streams one thread keeps in flight. See the `linear_gemv_exl3_k1_mcg`
# comment for the measurement that motivates it and why 1, 2, 4, 8.
#
# 8 is here because the register-pressure cliff has to be *measured* on this
# kernel, not inherited from a different experiment: `acc[NR0][2]` plus
# `v[NR0][8]` is 10*NR0 floats per thread, so 8 rows is 80 registers of
# accumulator and 256 of decoded values, against a 255-register budget that also
# has to hold the window setup. If the family stops at 4 there is no evidence
# about where the cliff is, and "we only tried 4" is how the last 13% went
# unclaimed.
NR0_VALUES = (1, 2, 4, 8)
# `kernel void linear_gemv(` is a PREFIX of `kernel void linear_gemv_exl3_k1(`,
# so a plain `str.index` finds the first generated kernel instead of the
# fallback kernel and the block is measured as empty. Require the `(` to be
# followed by end-of-name, i.e. not another identifier character.
NEXT_RE = re.compile(r"^kernel void linear_gemv\((?![\w])", re.M)

HEADER = '''// EXL3 decode matvec, shaped like the reference implementation.
//
// Thread mapping follows the trellis rather than the output:
//   * one thread per trellis lane (32 lanes == one simdgroup), so the eight codes
//     a lane owns are decoded from a single window setup;
//   * one 256-thread group (8 simdgroups) per output column-tile;
//   * simdgroups stride the tile-row axis, splitting the K reduction.
//
// The previous kernel used one thread per output element -- 5,376 threads for a
// 5376-wide projection on a 20-core GPU -- and re-derived the 64-bit window once
// per eight weights.
//
// A lane's eight codes land on rows {r0, r0+1, r0+8, r0+9} and columns {c, c+8},
// so four x values feed all eight, each used twice. The reduction is
// simd_shuffle_xor over four lanes then a threadgroup sum over the eight
// simdgroups. No atomics.
//
// Three axes are specialised, because all three are worth real throughput and
// Metal has no templates for any of them:
//   * KA (1..8), so shift amounts and the ring size fold to constants.
//   * ALIGNED, so a 4-byte-aligned buffer is read with one 32-bit load per word
//     instead of two 16-bit loads. Arena-resident weights are always aligned;
//     only mapped checkpoint bytes are not. The two-16-bit-load form doubles the
//     load instruction count on the hottest path in the engine, and loads -- not
//     the codebook -- dominate: measured 0.336 ms of 0.612 ms for q_proj.
//   * CODEBOOK, only for MCG. The codebook is a per-tensor constant and
//     branching on it eight times per lane cost 61.7% of the kernel; see
//     `exl3_decode8_mcg` for the measurement and for the reason it is the BRANCH
//     and not the arithmetic. `kN`/`kN_u` remain as the runtime-codebook
//     fallback for the plain and mul1 codebooks.
//
// Variants are generated by `scripts/gen_exl3_gemv.py`; do not hand-edit.
//
// GENERATED by scripts/gen_exl3_gemv.py -- do not hand-edit below this line.
'''


def two_word_is_safe(bits: int) -> tuple[bool, str]:
    """Is the third ring word provably dead at this bit rate?

    Derived, not configured, so the answer cannot drift from the arithmetic it
    depends on. The kernel extracts, for each code ``i`` in ``0..8``::

        r = i * KA + 16
        s = (r <= 64) ? uint(A0 >> (64 - r)) : uint(B0 >> (96 - r))

    and masks to 16 bits. So the A0 bits any code can touch are exactly
    ``[64 - r_max, 63]`` where ``r_max = 7*KA + 16``, i.e. ``[48 - 7*KA, 63]``.

    ``A0 = ((w0a << 32) | w0b) << sh | (w0c >> (32 - sh))``: the third word
    ``w0c`` contributes **only** to A0 bits ``[0, sh)`` -- the low bits shifted
    down from its top. Every bit at or above ``sh`` comes from the first two
    words. So ``w0c`` is dead when the lowest *used* bit is at or above the
    largest ``sh`` that occurs.

    Two ways it can fail, both checked:
      * ``r_max > 64`` -- the ``B0`` branch is taken, and ``B0`` is built from
        ``w0b``/``w0c``, so the third word is genuinely read.
      * ``48 - 7*KA < max(sh)`` -- some lane needs a bit below its own ``sh``,
        so it reads into ``w0c``.

    ``sh`` ranges over lanes because ``lo0 = (lane*8 + 1)*KA + RING - 16`` and
    ``sh = lo0 % 32``; with 32 lanes the set is enumerated, not bounded.

    Returns ``(safe, why)`` so a failure says which of the two conditions broke.
    """
    ring = 256 * bits
    r_max = 7 * bits + 16
    used_lo = 48 - 7 * bits
    sh_max = max(((lane * 8 + 1) * bits + ring - 16) % 32 for lane in range(32))
    if r_max > 64:
        return False, f"r_max={r_max} > 64, so the B0 path reads w0c"
    if used_lo < sh_max:
        return False, f"lowest used A0 bit {used_lo} < max sh {sh_max}, so some lane reads w0c"
    return True, f"used [{used_lo},63] >= max sh {sh_max}, and r_max={r_max} <= 64"


#: Rates at which a 2-word window is provably equivalent to the 3-word one.
#: Computed by `two_word_is_safe` rather than listed, so adding a rate to the
#: kernel cannot outrun the proof. 3 bpw is 410 of this checkpoint's 411 tensors;
#: 6 bpw (`lm_head`) is not in the set, which is why the family stops at 4.
TWO_WORD_SAFE_RATES: tuple[int, ...] = tuple(
    bits for bits in range(1, 9) if two_word_is_safe(bits)[0]
)

#: Rates at which the NO-DECODE probe is emitted. **Every** rate, deliberately,
#: and unlike `TWO_WORD_SAFE_RATES` this is a plain list rather than a derivation:
#: the probe produces garbage, so there is no correctness property for a rate to
#: satisfy or fail, and narrowing the set would only narrow the experiment.
#:
#: The probe exists to separate "3 bpw is slow because of lane-load requests"
#: from "3 bpw is slow because decode cost per useful byte is higher". Both
#: predict the same 1.33x, so the contrast arm at a *saturated* high rate is
#: what makes the reading unambiguous -- without it, a `_nod` number at 3 bpw
#: could be explained by the probe having removed requests too.
NO_DECODE_PROBE_RATES: tuple[int, ...] = tuple(range(1, 9))


def ishizuki_two_word_kernel(preamble: str, bits: int) -> str:
    """The `linear_gemv_ishizuki_kN_w2` kernel for one bit rate.

    A *transformation* of the verbatim ishizuki source, not a re-derivation:
    the variant is the original with the third word's contribution deleted. That
    matters more than it sounds, because the original is transcribed from
    ishizuki's Swift and re-deriving it here would create a second copy of the
    window arithmetic to keep in step -- and a variant that differs from the
    original in some *other* way would make the A/B unattributable.

    The transformation is exactly three edits, and each is asserted:

    1. drop ``w?c`` from the three-word load, leaving two;
    2. drop the ``| (sh ? (w?c >> (32 - sh)) : 0)`` term from ``A?``;
    3. drop the now-dead ``B?`` local, since with ``r_max <= 64`` no code reads
       it and a live-looking unused local invites someone to "fix" the B0 path
       back in.

    Every other line -- the unroll, the reduction, the ``r0`` index arithmetic,
    the threadgroup reduction -- is untouched, because those are what make the
    output bit-exact and they are not what is being measured.
    """
    name = f"linear_gemv_ishizuki_k{bits}"
    match = re.search(
        rf"^kernel void {re.escape(name)}\((?![\w])", preamble, re.M
    )
    if not match:
        raise SystemExit(f"{name} not found in the preamble; cannot emit its _w2 variant")
    # Slice from the START of the declaration line, not from the line after it.
    # The parameters are part of the text the substitutions below expect to find
    # around (`const ulong ...` sits between them and the body), and the name
    # replacement at the end rewrites `kernel void NAME(` in place -- so a slice
    # that began after the declaration line would emit a parameter list with no
    # `kernel void` in front of it, which the file's kernel count would not show
    # as a new kernel at all.
    nxt = re.search(r"^kernel void ", preamble[match.end() :], re.M)
    assert nxt, f"no kernel after {name}"
    body = preamble[match.start() : match.end() + nxt.start()]

    # 1 + 2: the two-word load and the A0 without the third word's contribution.
    # Both the unrolled (`w0a`/`A0`, `w1a`/`A1`) and tail (`a0`/`A0`) loops use
    # different variable names for the same thing, so each form is matched and
    # the count asserted rather than assumed.
    edits: list[tuple[str, str, int]] = [
        # Unrolled first tile: `const ulong w0a = t0[j0], w0b = t0[j1], w0c = t0[j2];`
        (
            r"const ulong w(\d)a = t(\d)\[j0\], w\1b = t\2\[j1\], w\1c = t\2\[j2\];",
            r"const ulong w\1a = t\2[j0], w\1b = t\2[j1];",
            2,
        ),
        # Tail loop: `const ulong a0 = t0[j0], a1 = t0[j1], a2 = t0[j2];`
        # The digit captured is the tile index, and the three locals are `a0`,
        # `a1`, `a2` regardless of it -- so the substitution indices differ, which
        # is why this cannot share a pattern with the unrolled form above.
        (
            r"const ulong a0 = t(\d)\[j0\], a1 = t\1\[j1\], a2 = t\1\[j2\];",
            r"const ulong a0 = t\1[j0], a1 = t\1[j1];",
            1,
        ),
        # A0/A1 lose the `w0c` term; the tail A0 loses the `a2` term.
        (
            r"const ulong (A\d) = \(\(w(\d)a << 32\) \| w\2b\) << sh0 \| "
            r"\(sh0 \? \(w\2c >> \(32 - sh0\)\) : 0ul\);",
            r"const ulong \1 = ((w\2a << 32) | w\2b) << sh0;",
            2,
        ),
        (
            r"const ulong (A0) = \(\(a0 << 32\) \| a1\) << sh0 \| "
            r"\(sh0 \? \(a2 >> \(32 - sh0\)\) : 0ul\);",
            r"const ulong \1 = ((a0 << 32) | a1) << sh0;",
            1,
        ),
    ]
    for pattern, replacement, expected in edits:
        body, n = re.subn(pattern, replacement, body)
        assert n == expected, (
            f"{name}: expected {expected} matches of {pattern!r}, got {n}. The "
            "verbatim ishizuki source was reformatted or hand-edited, and "
            "emitting a half-transformed kernel would silently change the "
            "arithmetic. Do not relax this."
        )
    # 3: the `r <= 64u ? A0 : B0` selection. At every rate in
    # `TWO_WORD_SAFE_RATES`, `r_max = 7*KA + 16 <= 64`, so the `B0` arm is
    # unreachable -- and with the third load gone there is nothing to select
    # from. The whole ternary goes, rather than keeping a dead local: leaving
    # `B0` in place while deleting its only source is a compile error, and
    # keeping the ternary would keep a branch the derivation says is dead.
    #
    # This is also why the derivation and the code cannot drift: if a future rate
    # made the `B0` arm reachable, `two_word_is_safe` would refuse to emit the
    # variant, so the substitution below never sees a kernel whose `B0` is live.
    body, n = re.subn(
        r"[ \t]*const ulong B\d = \(\(w(\d)b << 32\) \| w\1c\) << sh0;\n", "", body
    )
    assert n == 2, f"{name}: expected 2 unrolled dead B locals, removed {n}"
    body, n = re.subn(
        r"[ \t]*const ulong B0 = \(\(a1 << 32\) \| a2\) << sh0;\n", "", body
    )
    assert n == 1, f"{name}: expected 1 tail dead B local, removed {n}"
    # The extraction: `(r <= 64u) ? uint(A0 >> (64u - r)) : uint(B0 >> (96u - r))`
    # collapses to the first arm. Matched on the whole ternary -- including the
    # `B0`/`B1` reference -- so a reformat that leaves the dead arm in place is a
    # failed assert rather than a silently still-dead-but-present branch.
    body, n = re.subn(
        r"\(r <= 64u\) \? uint\((A\d) >> \(64u - r\)\) : uint\((?:A|B)\d >> \(96u - r\)\)",
        r"uint(\1 >> (64u - r))",
        body,
    )
    assert n == 3, (
        f"{name}: expected 3 extraction ternaries to collapse, got {n}. If the "
        "extraction was reformatted, the dead B0 arm is still in the emitted "
        "kernel and the variant is not the two-word kernel it claims to be."
    )

    kernel = (
        body.replace(
            f"kernel void {name}(",
            f"kernel void {name}_w2(\n"
            "    // TWO-WORD WINDOW VARIANT of the verbatim ishizuki kernel above.\n"
            "    //\n"
            "    // Identical output, one fewer load per lane per tile: the third\n"
            "    // ring word is provably never read at this bit rate. See\n"
            "    // `two_word_is_safe` in scripts/gen_exl3_gemv.py for the derivation\n"
            "    // and the two conditions that would invalidate it.\n"
            "    //\n"
            "    // This is a VARIANT for A/B and for the golden reference. The\n"
            "    // engine dispatches the unsuffixed kernel unless the measurement\n"
            "    // says otherwise; see spec/decode-bandwidth-240gb/tasks.md T6.\n",
            1,
        )
        .strip()
    )
    return kernel


NOD_STUB = """
/// One-AND stand-in for `decode_state`, used ONLY by the `_nod` probe kernels.
///
/// Two properties it must have, and both are about the measurement rather than
/// the arithmetic:
///
/// * it **consumes** its argument, so the extraction above it is not
///   dead-code-eliminated. If the stub ignored `s`, the compiler would delete
///   the shift-and-mask chain above it and quite possibly the ring loads that
///   feed it -- which are the very loads being measured. The probe would then
///   report the bandwidth of a kernel that fetches nothing.
/// * it is **not constant-foldable**, which is why it is a `static` function
///   taking a runtime value rather than a macro or a literal.
///
/// The output is meaningless. Nothing in the engine dispatches a `_nod` kernel,
/// and no test compares one against a reference -- see
/// `ishizuki_no_decode_kernel`.
static float nod_stub(uint s) { return float(s & 1u); }
"""


def ishizuki_no_decode_kernel(preamble: str, bits: int) -> str:
    """The `linear_gemv_ishizuki_kN_nod` PROBE kernel for one bit rate.

    **This kernel computes garbage and is never dispatched by the engine.** It
    replaces `decode_state(s & 0xffffu, CB)` with a cheap function of the same
    argument, so the output is numerically meaningless while the *memory traffic
    is byte-for-byte the same as the real kernel*. That is the point: it is a
    bandwidth probe, not a correctness one, and it is how the two candidate
    mechanisms behind "3 bpw is slower than 4 bpw" get separated.

    # Why the question needs an experiment

    The trustworthy interleaved sweep (`gemv_shape --rate-interleaved`, control
    spread 1.13x) gives:

    | rate | GB/s | lane-loads/useful byte | decode_state/useful byte |
    |---|---|---|---|
    | 3 bpw | 165.7 | 2.67x | 2.67 |
    | 4 bpw | 216.4 | 2.00x | 2.00 |
    | 6 bpw | 215.7 | 2.00x | 1.33 |
    | 8 bpw | 217.9 | 1.50x | 1.00 |

    4, 6 and 8 bpw are all at ~216 GB/s: DRAM-saturated, insensitive to rate.
    Only 3 bpw is slow, by 216.4/165.7 = **1.306x**. And *two* mechanisms both
    predict 1.333x:

    - **requests**: lane-load amplification is 2.67/2.00 = 1.333x
    - **decode cost**: `decode_state` calls per useful byte is also 2.67/2.00

    A bandwidth table cannot separate these, because both scale identically with
    rate. Stubbing the decode splits them cleanly:

    - if `_nod` at 3 bpw jumps to ~216 GB/s, decode cost is the wall and the fix
      is a cheaper decode;
    - if `_nod` stays near 166, requests are the wall and T6's mechanism is the
      whole story;
    - a partial move means both, and the split is readable as a fraction.

    # Why this is a transformation and not a re-derivation

    Same discipline as `ishizuki_two_word_kernel`: the probe is the verbatim
    source with one asserted substitution, so a difference between `_nod` and the
    real kernel is attributable to the decode and to nothing else. The counts are
    asserted, because a reformat that left some `decode_state` calls live would
    emit a kernel that is neither the probe nor the real thing, and would report a
    bandwidth that belongs to neither.

    The stub is `float(uint(s) & 1u)` -- one AND, one convert. It must consume
    `s` (so the extraction above it is not dead-code-eliminated, which would
    remove the very loads being measured) and must not be constant-foldable.
    """
    name = f"linear_gemv_ishizuki_k{bits}"
    match = re.search(rf"^kernel void {re.escape(name)}\((?![\w])", preamble, re.M)
    if not match:
        raise SystemExit(f"{name} not found in the preamble; cannot emit its _nod probe")
    nxt = re.search(r"^kernel void ", preamble[match.end() :], re.M)
    # `k8` is the LAST ishizuki kernel, so there is no following declaration to
    # bound it. Slicing to the end of the preamble is correct for that case, and
    # it is what makes the probe available at EVERY rate -- which is the point of
    # the experiment, since it needs a contrast arm at a saturated high rate.
    end = match.end() + nxt.start() if nxt else len(preamble)
    body = preamble[match.start() : end]

    # One substitution: every decode call becomes a one-AND stub. The count is
    # asserted per rate because the verbatim kernels differ in how many decode
    # sites they have (the unrolled loop has two, the tail loop one).
    body, n = re.subn(
        r"decode_state\(s & 0xffffu, CB\)",
        "nod_stub(s)",
        body,
    )
    assert n >= 3, (
        f"{name}: expected at least 3 decode_state call sites to stub, replaced "
        f"{n}. Fewer means the probe no longer covers every decode in the kernel "
        "and its bandwidth is not attributable to removing decode."
    )

    kernel = (
        body.replace(
            f"kernel void {name}(",
            f"kernel void {name}_nod(\n"
            "    // NO-DECODE PROBE. OUTPUT IS GARBAGE BY CONSTRUCTION.\n"
            "    //\n"
            "    // Identical memory traffic to the kernel above; `decode_state` is\n"
            "    // replaced by a one-AND stub so the decode arithmetic is the only\n"
            "    // difference. Never dispatched by the engine. See\n"
            "    // `ishizuki_no_decode_kernel` in scripts/gen_exl3_gemv.py for the\n"
            "    // question this exists to answer.\n",
            1,
        )
        .strip()
    )
    return kernel


def main() -> None:
    src = PATH.read_text()
    start = src.index(BANNER)
    match_next = NEXT_RE.search(src, start)
    assert match_next, "fallback `kernel void linear_gemv(` not found"
    end = match_next.start()
    body = src[start:end]
    # Everything between the banner and the first GENERATED kernel is preserved
    # verbatim: the ishizuki reference kernels (`linear_gemv_ishizuki_k1..8`)
    # live there. Slicing the template to "exactly one kernel" -- which is what
    # makes this script idempotent -- would otherwise delete them, and
    # `LinearKernels::new` then fails with "Function
    # 'linear_gemv_ishizuki_k1' does not exist" at library build, far from the
    # generator that removed it.
    prefix_end = re.search(
        r"^kernel void linear_gemv_exl3_k1\((?![\w])", body, re.M
    ).start()
    # Skip past the whole banner COMMENT, not just its first line: BANNER is one
    # line, but the comment block runs on for ~40 lines and HEADER re-emits all
    # of it. Slicing from `len(BANNER)` would duplicate the rest of the comment
    # on every run -- a file that grows by 42 lines a pass, which is exactly the
    # unbounded growth the join below exists to prevent.
    #
    # The pristine file has no `GENERATED by` marker, so the comment is found by
    # its first non-comment line: everything from the banner to the first line
    # that is not a comment and not blank.
    lines = body.splitlines(keepends=True)
    consumed = 0
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("//"):
            break
        consumed += len(line)
    preamble = body[consumed:prefix_end]
    # The template is EXACTLY ONE kernel: from the first generated kernel up to
    # the next line-initial `kernel void` after it.
    #
    # The name is matched with a trailing `(` that is NOT followed by more
    # identifier characters, and anchored to the start of a line. A plain
    # `index("linear_gemv_exl3_k1(")` also matches the prefix of `k1_u(` and
    # `k1_mcg(`, so after one run it can land on a different kernel than the
    # first one -- and if that kernel is a specialised one, the decode call
    # normalises to the same thing, so the slice is silently the wrong template.
    first_match = re.search(
        r"^kernel void linear_gemv_exl3_k1\((?![\w])", body, re.M
    )
    assert first_match, "first EXL3 kernel not found"
    first = first_match.start()
    # Search from the END of the declaration line, not from `first`: `first` is
    # itself a `^kernel void ` match, so a search starting there returns offset 0
    # and the template comes out empty.
    after = body.index("\n", first) + 1
    match = re.search(r"^kernel void ", body[after:], re.M)
    assert match, "no following kernel declaration"
    second = after + match.start()
    tmpl = body[first:second]
    tmpl = tmpl.replace("kernel void linear_gemv_exl3_k1_u_mcg(", "KERNEL(")
    tmpl = tmpl.replace("kernel void linear_gemv_exl3_k1_mcg(", "KERNEL(")
    tmpl = tmpl.replace("kernel void linear_gemv_exl3_k1_u(", "KERNEL(")
    tmpl = tmpl.replace("kernel void linear_gemv_exl3_k1(", "KERNEL(")
    # The pristine file writes `const uint KA = 1;` (no `u` suffix); a previous
    # run of this script leaves `KA_N` in place, which must be matched too, or the
    # generated kernels all come out as KA = 1.
    tmpl = re.sub(r"const uint KA = (?:1|1u|KA_N);", "const uint KA = KA_N;", tmpl)
    # Same treatment for the NR0 axis: the pristine file writes `const uint NR0
    # = 1;`, a previous run leaves `NR0_N`. Without this the emitted kernels
    # would all be NR0 = 1 and the whole family would collapse onto the original
    # kernel while still carrying distinct names -- the same failure as the KA
    # axis, and the harder one to notice, because every name would resolve and
    # every measurement would be of NR0 = 1.
    tmpl = re.sub(
        r"const uint NR0 = (?:1|1u|NR0_N);", "const uint NR0 = NR0_N;", tmpl
    )
    # The four-term accumulator update is normalised to an explicit fma chain,
    # because the compiler's choice of which product to contract into an FMA
    # depends on register pressure, and the NR0 axis changes register pressure
    # directly. Measured: with the source form,
    # `linear_gemv_exl3_k8_mcg_r8` at 5376x21504 differed from the NR0=1 kernel
    # in 2 of 21504 outputs by one fp16 ulp, while every other (rate, NR0) pair was
    # identical. FMA-contraction instability of exactly the kind this project has
    # been bitten by twice. An explicit chain removes the choice, costs nothing
    # (the compiler emitted three FMAs and a multiply-add here anyway), and makes
    # the result independent of NR0.
    #
    # The `.replace` is on the exact pristine text including its indentation and
    # its `q`-subscripted `v`, so a reformat of the template makes this stop
    # matching -- and it must then fail loudly rather than silently emit the
    # unstable form again. Hence the assert.
    stable_acc = """\
        #pragma unroll
        for (uint q = 0u; q < NR0; ++q) {
            acc[q][0] = fma(x3, v[q][3], fma(x2, v[q][2], fma(x1, v[q][1], fma(x0, v[q][0], acc[q][0]))));
            acc[q][1] = fma(x3, v[q][7], fma(x2, v[q][6], fma(x1, v[q][5], fma(x0, v[q][4], acc[q][1]))));
        }"""
    unstable_acc = re.compile(
        r"[ \t]*#pragma unroll\n"
        r"[ \t]*for \(uint q = 0u; q < NR0; \+\+q\) \{\n"
        r"[ \t]*acc\[q\]\[0\] \+= x0 \* v\[q\]\[0\].*?\n"
        r"[ \t]*acc\[q\]\[1\] \+= x0 \* v\[q\]\[4\].*?\n"
        r"[ \t]*\}"
    )
    # Also match the *stable* form, so the normalisation is idempotent: the
    # template is sliced back out of this file's own output, so on the second run
    # it already carries the stable chain and the unstable pattern matches
    # nothing. Matching only the unstable form made the generator non-idempotent
    # and the count assert fire on run 2 -- correctly, since a generator that
    # cannot run twice is a generator whose output nobody can verify. The assert
    # below then covers both: exactly one accumulator loop must be present in
    # whichever form, and it is re-emitted as the stable one.
    stable_acc_re = re.compile(
        r"[ \t]*#pragma unroll\n"
        r"[ \t]*for \(uint q = 0u; q < NR0; \+\+q\) \{\n"
        r"[ \t]*acc\[q\]\[0\] = fma\(x3.*?\n"
        r"[ \t]*acc\[q\]\[1\] = fma\(x3.*?\n"
        r"[ \t]*\}"
    )
    tmpl, n_unstable = unstable_acc.subn(stable_acc, tmpl)
    tmpl, n_stable = stable_acc_re.subn(stable_acc, tmpl)
    n_acc = n_unstable + n_stable
    assert n_acc == 1, (
        f"expected exactly one NR0 accumulator loop in the template, got {n_acc} "
        f"({n_unstable} unstable, {n_stable} stable). Either the loop was "
        "reindented or reformatted, or there is more than one. Do not relax "
        "this: emitting the unstable form silently is how a 1-ulp decode "
        "difference gets shipped."
    )
    # Normalise BOTH the generic and the already-specialised decode call into a
    # single placeholder, so a second run of this script derives the same template
    # as the first. `CBARG` is the ENTIRE optional argument INCLUDING its leading
    # comma and the space after it -- either ", params.codebook " or nothing --
    # because the two output families differ in whether the codebook argument is
    # present at all. Leaving the comma outside the placeholder produces
    # `KA, , v);`, a compile error rather than a silent miscompile, but still a
    # wasted build.
    # The replacement is a lambda, not a template string, so `re.sub` does not try
    # to interpret the placeholder's punctuation as a group reference.
    # The tile address is an expression (`tk * tiles_n + tn * NR0 + q`) rather
    # than the `tile_index` local the pre-NR0 kernel bound in a `const`, because
    # the NR0 axis needs it to vary with the unrolled `q`. It is matched
    # non-greedily up to the argument that names the destination, so the
    # expression is free-form and a change to it cannot silently stop matching
    # and leave every generated kernel calling the same tile.
    tmpl, n = re.subn(
        r"exl3_decode8(?:_mcg)?<(true|false)>\(\s*packed,\s*(?P<tile>[^;]*?),\s*"
        r"lane,\s*KA\s*,?\s*(?P<cb>params\.codebook)?\s*,?\s*(?P<dst>v(?:\[[^\]]*\])?)"
        r"\s*\);",
        # The placeholder owns the codebook argument AND its leading comma AND
        # the space before it, so neither family can leave a dangling comma or
        # omit a needed one. Generic emits the pristine text exactly;
        # specialised drops the codebook argument and the comma that introduced
        # it. Getting this wrong yields `KA, , v[q]);` or `KAv[q]);` -- both
        # compile errors, so a mistake here is loud.
        #
        # The tile expression and the destination are captured and re-emitted
        # verbatim rather than reconstructed, because both are NR0-dependent
        # (`tk * tiles_n + tn * NR0 + q`, `v[q]`) and a reconstructed form would
        # be a second copy of the kernel's address arithmetic to keep in step.
        lambda m: (
            f"DECODE<{m.group(1)}>(packed, {m.group('tile')}, lane, "
            f"KA{'@CB@' if m.group('cb') else ''}, {m.group('dst')});"
        ),
        tmpl,
    )
    assert n == 1, f"expected exactly one decode call in the template, got {n}"
    # `NOD_STUB` lives in the PRESERVED preamble region, so a second run reads it
    # back out of the file and appends another copy -- the file grows by the stub
    # on every run and stops being idempotent. Strip any existing copy first, by
    # its distinctive body rather than by exact text, so a hand-edited comment
    # does not defeat the strip.
    preamble = re.sub(
        r"\n*/// One-AND stand-in for `decode_state`.*?"
        r"static float nod_stub\(uint s\) \{ return float\(s & 1u\); \}\n",
        "\n",
        preamble,
        flags=re.S,
    )
    out = [HEADER, preamble.rstrip(), NOD_STUB.strip()]
    # Fallback first: runtime codebook, for plain and mul1. Unchanged shape.
    for bits in range(1, 9):
        for aligned, suffix in ((True, ""), (False, "_u")):
            out.append(
                tmpl.replace("KERNEL(", f"kernel void linear_gemv_exl3_k{bits}{suffix}(")
                .replace("KA_N", str(bits))
                .replace("NR0_N", "1")
                # `@CB@` is the codebook argument plus its leading comma, so the
                # emitted call is `..., lane, KA, params.codebook, v[q]);` --
                # the pristine text apart from the `q` subscript, which the NR0
                # loop needs. Getting the comma wrong yields `KA, , v[q]);` or
                # `KAv[q]);`, both of which are compile errors, so a mistake here
                # is loud rather than silent.
                .replace("@CB@", ", params.codebook")
                .replace("DECODE", "exl3_decode8")
                .replace("<true>", "<true>" if aligned else "<false>")
            )
    # Specialised: MCG codebook folded to a compile-time constant. The two
    # variants differ ONLY in this one call, so the difference in the emitted
    # source is one token and everything else -- window setup, shifts, unroll,
    # accumulation -- is identical by construction.
    for bits in range(1, 9):
        for aligned, suffix in ((True, ""), (False, "_u")):
            out.append(
                tmpl.replace(
                    "KERNEL(", f"kernel void linear_gemv_exl3_k{bits}{suffix}_mcg("
                )
                .replace("KA_N", str(bits))
                .replace("NR0_N", "1")
                .replace("@CB@", "")
                .replace("DECODE", "exl3_decode8_mcg")
                .replace("<true>", "<true>" if aligned else "<false>")
            )
    # The NR0 family. Only the ALIGNED MCG arm, for three reasons that are all
    # reachability facts rather than preferences:
    #   * ALIGNED: the engine's 410 3-bpw tensors live in the 512 MB arena, which
    #     is 4-byte aligned by construction. The `_u` form exists for one tensor
    #     (`lm_head.trellis` starts at % 4 == 2) and is 6 bpw, where the measured
    #     rate is already at the ceiling, so there is nothing to win there.
    #   * MCG: the only codebook the engine can dispatch, same argument as above.
    #   * bits 1..8: emitted for all of them so the *dispatch table* is complete
    #     and a test can assert that every name in it exists, rather than the
    #     table having holes that only a specific rate would fall into.
    #
    # NR0 = 1 is NOT emitted with an `_r1` suffix: the unsuffixed MCG kernel *is*
    # NR0 = 1, and two names for one kernel is one more thing to keep in step.
    for bits in range(1, 9):
        for nr0 in NR0_VALUES:
            if nr0 == 1:
                continue
            out.append(
                tmpl.replace(
                    "KERNEL(", f"kernel void linear_gemv_exl3_k{bits}_mcg_r{nr0}("
                )
                .replace("KA_N", str(bits))
                .replace("NR0_N", str(nr0))
                .replace("@CB@", "")
                .replace("DECODE", "exl3_decode8_mcg")
                .replace("<true>", "<true>")
            )
    # The ishizuki 2-word family. See `TWO_WORD_SAFE_RATES` for the derivation and
    # `ishizuki_two_word_kernel` for the transformation; both are the only places
    # that know the rule, and a test asserts the generated set matches.
    for bits in TWO_WORD_SAFE_RATES:
        out.append(ishizuki_two_word_kernel(preamble, bits))
    # The ishizuki NO-DECODE probes. Every rate, not just the safe ones: the
    # question is "is decode cost the wall at LOW rates", so the probe has to
    # exist at 3 bpw and at the saturated high rates for contrast. Emitting them
    # from the same verbatim `preamble` as the real kernels is what makes the
    # comparison attributable to the decode alone.
    for bits in NO_DECODE_PROBE_RATES:
        out.append(ishizuki_no_decode_kernel(preamble, bits))
    # Each generated kernel carries a trailing newline, so joining with "\n"
    # inserts a blank line between kernels and, because the template is sliced
    # back out of this block, one MORE blank line is appended on every run. The
    # file grows without bound. Strip each kernel and join with a fixed two
    # newlines so repeated runs are byte-identical.
    PATH.write_text(
        src[:start] + "\n\n".join(k.rstrip() + "\n" for k in out) + src[end:]
    )
    print(
        f"generated {len(out) - 1} kernels in {PATH} "
        f"({32} base + {8 * (len(NR0_VALUES) - 1)} NR0 variants"
        + (
            f" + {len(TWO_WORD_SAFE_RATES)} ishizuki 2-word"
            if TWO_WORD_SAFE_RATES
            else ""
        )
        + f" + {len(NO_DECODE_PROBE_RATES)} ishizuki no-decode PROBES (garbage output, never dispatched)"
        + ")"
    )
    for bits in TWO_WORD_SAFE_RATES:
        print(f"  ishizuki 2-word at {bits} bpw: {two_word_is_safe(bits)[1]}")


if __name__ == "__main__":
    main()
