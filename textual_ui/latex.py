import re

from latex_parser import parser as latex_parser

# Operators that chain equality/implication steps — both LaTeX commands
# (before parser) and Unicode (after parser)
_CHAIN_OPS = re.compile(
    r"(?<!\\)(?:"
    r"\\implies\b|\\Rightarrow\b|\\Leftrightarrow\b|\\iff\b|\\equiv\b"
    r"|=>|<=>|⇒|⇔|≡"
    r")"
)

# Plain equals sign — only split if there are at least two on the line
_PLAIN_EQ = re.compile(r"(?<![=<>!])=(?![=<>!])")

_MARKER = re.compile(r"⌈([^⌈⌋]+)⌋([^⌉]+)⌉")


def _stack_fractions(text: str) -> str:
    """Convert ⌈num⌋den⌉ markers to stacked ASCII art.

    Adjacent fractions (separated by whitespace or punctuation like `, `)
    are rendered side by side with their separators preserved.
    Text before marker(s) goes on the numerator line.
    Text after marker(s) that starts with =/⇒ goes on the fraction bar line;
    other trailing text goes after the denominator line.
    Markers on an existing fraction bar line or denominator line are inlined.
    """
    while True:
        m = _MARKER.search(text)
        if not m:
            break

        num, den = m.group(1), m.group(2)

        # If on a fraction bar line or denominator line → inline
        line_start = text.rfind("\n", 0, m.start()) + 1
        line_end = text.find("\n", m.end())
        if line_end == -1:
            line_end = len(text)
        cur_line = text[line_start:line_end]
        prev_line = ""
        if line_start > 0:
            prev_start = text.rfind("\n", 0, line_start - 1) + 1
            prev_line = text[prev_start:line_start - 1]
        if "─" in cur_line or "─" in prev_line:
            text = text[:m.start()] + f"{num}/{den}" + text[m.end():]
            continue

        # Collect adjacent markers and the separators between them
        markers = [(num, den)]
        separators: list[str] = []
        adj_end = m.end()
        while True:
            next_m = re.match(r"([ \t·*×,;]+)⌈([^⌈⌋]+)⌋([^⌉]+)⌉", text[adj_end:])
            if not next_m:
                break
            separators.append(next_m.group(1))
            markers.append((next_m.group(2), next_m.group(3)))
            adj_end += len(next_m.group(0))

        prefix = text[: m.start()]
        rest = text[adj_end:]

        # Check if rest contains an equation operator → put on bar line
        if re.search(r"(?<![=<>!])=(?![=<>!])|⇒", rest):
            bar_rest = rest
            den_rest = ""
        else:
            bar_rest = ""
            den_rest = rest

        # Extract indent from prefix (horizontal whitespace only)
        indent_m = re.match(r"^([ \t]*)", prefix)
        indent = indent_m.group(1) if indent_m else ""
        num_suffix = prefix[len(indent) :]

        if len(markers) == 1:
            w = max(len(num), len(den))
            frac = f"{num.center(w)}  \n{'─' * w}{bar_rest}  \n{den.center(w)}{den_rest}"
            frac = indent + num_suffix + frac.replace("\n", "\n" + indent)
        else:
            widths = [max(len(n), len(d)) for n, d in markers]
            nums_parts: list[str] = []
            bars_parts: list[str] = []
            dens_parts: list[str] = []
            for j, ((n, d), w) in enumerate(zip(markers, widths)):
                sep = separators[j - 1] if j > 0 else ""
                nums_parts.append(sep + n.center(w))
                bars_parts.append(sep + "─" * w)
                dens_parts.append(sep + d.center(w))
            nums = "".join(nums_parts)
            bars = "".join(bars_parts)
            dens = "".join(dens_parts)
            frac = f"{nums}  \n{bars}{bar_rest}  \n{dens}{den_rest}"
            frac = indent + num_suffix + frac.replace("\n", "\n" + indent)

        text = frac
    return text


def _split_chain(text: str) -> str:
    """Split chained equality/implication steps onto separate lines.

    Each step gets 3-space indent and a blank line between steps.
    Skips operators on fraction bar lines or fraction denominator lines.
    """
    lines = text.split("\n")

    # Protected: denominator lines (line after ─ line)
    den_line_nums = set()
    for i, line in enumerate(lines):
        if "=" in line and i > 0 and "─" in lines[i - 1]:
            den_line_nums.add(i)

    # Protected: fraction bar lines (lines containing ─)
    bar_line_nums = {i for i, line in enumerate(lines) if "─" in line}

    def _line_num(pos: int) -> int:
        return text[:pos].count("\n")

    ops = []
    for m in _CHAIN_OPS.finditer(text):
        ln = _line_num(m.start())
        if ln in bar_line_nums:
            continue
        ops.append(("⇒", m.start(), m.end()))
    for m in _PLAIN_EQ.finditer(text):
        ln = _line_num(m.start())
        if ln in bar_line_nums:
            continue
        if ln in den_line_nums:
            continue
        ops.append(("=", m.start(), m.end()))
    ops.sort(key=lambda x: x[1])

    if len(ops) < 2:
        return text

    # Filter comma-separated equals
    filtered = []
    for i, (op, start, end) in enumerate(ops):
        if op == "=":
            prev_seg = ""
            next_seg = ""
            for j in range(i - 1, -1, -1):
                if ops[j][0] == "=":
                    prev_seg = text[ops[j][2] : start]
                    break
            for j in range(i + 1, len(ops)):
                if ops[j][0] == "=":
                    next_seg = text[end : ops[j][1]]
                    break
            if (prev_seg and "," in prev_seg) or (next_seg and "," in next_seg):
                continue
        filtered.append((op, start, end))

    if len(filtered) < 2:
        return text

    # Split into segments at each operator
    segments = []
    before = text[: filtered[0][1]].strip()
    if before:
        segments.append(before)
    for i in range(len(filtered)):
        start = filtered[i][1]
        end = filtered[i + 1][1] if i + 1 < len(filtered) else len(text)
        seg = text[start:end].strip()
        if seg:
            segments.append(seg)

    result = []
    for seg in segments:
        result.append("   " + seg.replace("\n", "\n   "))
    return "  \n\n".join(result)


def _parse_with_chain(inner: str) -> str:
    """Parse LaTeX, chain-split, merge = back to marker segments, then stack fractions per segment."""
    try:
        inner_stripped = inner.strip("\n")
        parsed = latex_parser.parse(inner_stripped)
        chained = _split_chain(parsed)
        segments = chained.split("\n\n")

        # Merge segments: if a segment has markers and the next starts with = or ⇒,
        # merge them so = rest appears on the fraction bar line.
        is_chain = len(segments) > 1
        merged = []
        i = 0
        while i < len(segments):
            seg = segments[i].strip()
            has_marker = "⌈" in seg
            if has_marker and i + 1 < len(segments):
                nxt = segments[i + 1].strip()
                if nxt.startswith("=") or nxt.startswith("⇒"):
                    merged.append("   " + seg + " " + nxt)
                    i += 2
                    continue
            merged.append(("   " if is_chain else "") + seg)
            i += 1

        stacked = [_stack_fractions(s) for s in merged]
        result = "  \n\n".join(stacked)

        # Always wrap in newlines so surrounding text doesn't absorb fraction lines
        if not inner.startswith("\n"):
            result = "\n" + result
        if not inner.endswith("\n"):
            result = result + "\n"

        return result
    except Exception:
        return inner


def parse_latex(text: str) -> str:
    _BS = re.escape("\\")
    _LBRACE = "{"
    _RBRACE = "}"

    def _parse(inner):
        try:
            return latex_parser.parse(inner)
        except Exception:
            return inner

    def _parse_block(m):
        return _parse(m.group(0))

    text = re.sub(
        r"```latex\s*\n(.+?)```",
        lambda m: _parse(m.group(1).strip()),
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"```latex\s*(.+?)```",
        lambda m: _parse(m.group(1).strip()),
        text,
    )

    text = re.sub(
        _BS + r"\[(.*?)" + _BS + r"\]",
        lambda m: _parse(m.group(1)),
        text,
        flags=re.DOTALL,
    )

    text = re.sub(
        _BS
        + r"begin"
        + _LBRACE
        + r"([\w*]+)"
        + _RBRACE
        + r".*?"
        + _BS
        + r"end"
        + _LBRACE
        + r"\1"
        + _RBRACE,
        _parse_block,
        text,
        flags=re.DOTALL,
    )

    # $$...$$: full pipeline (parse, chain-split, merge, stack per segment)
    # Runs AFTER environments so $$ blocks within \begin{...} survive parsing
    text = re.sub(
        r"\$\$(.+?)\$\$",
        lambda m: _parse_with_chain(m.group(1)),
        text,
        flags=re.DOTALL,
    )

    # Inline $...$: parse and stack fractions (no chain-split)
    # Must run AFTER $$ so single $ doesn't match $$ content
    def _parse_inline(inner):
        try:
            parsed = latex_parser.parse(inner)
            return _stack_fractions(parsed)
        except Exception:
            return inner

    text = re.sub(r"\$(.+?)\$", lambda m: _parse_inline(m.group(1)), text)

    _CMDS = (
        r"(?:textcolor|color)"
        + _LBRACE
        + r"[^}]*"
        + _RBRACE
        + _LBRACE
        + r"[^}]*"
        + _RBRACE
    )
    text = re.sub(_BS + _CMDS, _parse_block, text)

    _CMDS2 = (
        r"(?:textbf|textit|texttt|mathrm|mathbf|mathit|mathsf|mathtt"
        r"|mathcal|mathbb|mathfrak|section|subsection|subsubsection"
        r"|paragraph|huge|Huge|LARGE|Large|large|normalsize|small"
        r"|footnotesize|scriptsize|tiny|underline|uline|sout|cancel"
        r"|emph|text|boxed)"
        + _LBRACE
        + r"[^}]*"
        + _RBRACE
    )
    text = re.sub(_BS + _CMDS2, _parse_block, text)

    # Final pass: if raw LaTeX commands remain, parse and stack fractions
    if re.search(r"\\[a-z]+(?:\{|\(|\[)", text, re.I):
        try:
            text = latex_parser.parse(text)
        except Exception:
            pass
    text = _stack_fractions(text)

    return text


def format_for_display(text: str) -> str:
    text = parse_latex(text)
    return text


def strip_prompt_markers(text: str) -> str:
    text = re.sub(r'\[INFO\].*', '', text)
    lines = text.splitlines()
    clean = [l for l in lines if l.strip()]
    return "\n".join(clean).strip()
