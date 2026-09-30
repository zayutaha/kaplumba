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
    tmpl, n_acc = unstable_acc.subn(stable_acc, tmpl)
    assert n_acc == 1, (
        f"expected exactly one NR0 accumulator loop in the template, got {n_acc}. "
        "Either the loop was reindented or reformatted, or it was already "
        "replaced -- in which case the stable form is present and this assert is "
        "stale. Do not relax it: emitting the unstable form silently is how a "
        "1-ulp decode difference gets shipped."
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
    out = [HEADER, preamble.rstrip()]
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
        f"({32} base + {8 * (len(NR0_VALUES) - 1)} NR0 variants)"
    )


if __name__ == "__main__":
    main()
