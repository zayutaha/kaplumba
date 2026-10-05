#!/usr/bin/env python3
"""Proof that crux3 arm A is the production kernel.

The macro in `crux3_decode_algebra.metal` is a hand transcription of
`linear_gemv_ishizuki_k3_colmajor_2col_sk4` from `linear_repacked.metal`. A
transcription error would silently make every arm-A/B/C comparison a comparison
of three different kernels. This normalises both to a token stream and asserts
they are identical.

Run: python3 scripts/crux3_armA_equiv.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "crates/kernels/src/linear_repacked.metal"
PROBE = ROOT / "crates/kernels/src/crux3_decode_algebra.metal"


def body_of(text, header):
    """Return the text between the first '{' after `header` and its match."""
    start = text.index(header)
    i = text.index("{", start)
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j]
    raise AssertionError(f"unbalanced braces after {header}")


def normalise(src, extract_name):
    """Strip comments/whitespace and rewrite the extract call to a fixed token."""
    src = re.sub(r"//[^\n]*", " ", src)
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    src = src.replace(extract_name, "@EXTRACT@")
    # macro line continuations
    src = src.replace("\\\n", " ")
    src = re.sub(r"\s+", " ", src)
    return src.strip()


prod = body_of(
    REP.read_text(),
    "kernel void linear_gemv_ishizuki_k3_colmajor_2col_sk4(",
)
arm_a = body_of(
    PROBE.read_text(),
    "#define CRUX3_COLMAJOR_2COL_SK4(NAME, EXTRACT)",
)

# The macro's parameter list precedes its body; take from the first declaration.
arm_a = arm_a[arm_a.index("constexpr uint KA") :]

p = normalise(prod, "extract_v_mcg_32")
a = normalise(arm_a, "EXTRACT")

if p == a:
    print("ARM A IS BYTE-IDENTICAL to linear_gemv_ishizuki_k3_colmajor_2col_sk4")
    print(f"  {len(p)} normalised tokens")
    sys.exit(0)

print("MISMATCH -- arm A is NOT the production kernel\n")
pl, al = p.split(" "), a.split(" ")
shown = 0
for i in range(max(len(pl), len(al))):
    x = pl[i] if i < len(pl) else "<end>"
    y = al[i] if i < len(al) else "<end>"
    if x != y:
        lo = max(0, i - 8)
        print(f"  token {i}:")
        print(f"    production: {' '.join(pl[lo:i+8])}")
        print(f"    arm A     : {' '.join(al[lo:i+8])}")
        shown += 1
        if shown >= 6:
            break
sys.exit(1)
