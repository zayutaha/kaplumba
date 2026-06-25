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


def _split_chain(text: str) -> str:
    """Split chained equality/implication steps onto separate lines.
    
    Each step gets 3-space indent and a blank line between steps.
    """
    # Collect all chain operators
    ops = []
    for m in _CHAIN_OPS.finditer(text):
        ops.append(("⇒", m.start(), m.end()))
    for m in _PLAIN_EQ.finditer(text):
        ops.append(("=", m.start(), m.end()))
    ops.sort(key=lambda x: x[1])

    if len(ops) < 2:
        return text

    # Filter: skip equals that are comma-separated
    filtered = []
    for i, (op, start, end) in enumerate(ops):
        if op == "=":
            prev_seg = ""
            next_seg = ""
            for j in range(i - 1, -1, -1):
                if ops[j][0] == "=":
                    prev_seg = text[ops[j][2]:start]
                    break
            for j in range(i + 1, len(ops)):
                if ops[j][0] == "=":
                    next_seg = text[end:ops[j][1]]
                    break
            if (prev_seg and "," in prev_seg) or (next_seg and "," in next_seg):
                continue
        filtered.append((op, start, end))

    if len(filtered) < 2:
        return text

    # Split into segments at each operator
    segments = []
    before = text[:filtered[0][1]].strip()
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


def parse_latex(text: str) -> str:
    _BS = re.escape('\\')
    _LBRACE = '{'
    _RBRACE = '}'

    def _parse(inner):
        try:
            return latex_parser.parse(inner)
        except Exception:
            return inner

    def _parse_with_chain(inner):
        try:
            parsed = latex_parser.parse(inner)
            return _split_chain(parsed)
        except Exception:
            return inner

    def _parse_block(m):
        return _parse(m.group(0))

    text = re.sub(
        r'```latex\s*\n(.+?)```',
        lambda m: _parse(m.group(1).strip()),
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r'```latex\s*(.+?)```',
        lambda m: _parse(m.group(1).strip()),
        text,
    )

    text = re.sub(
        _BS + r'\[(.*?)' + _BS + r'\]',
        lambda m: _parse(m.group(1)),
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r'\$\$(.+?)\$\$',
        lambda m: _parse_with_chain(m.group(1)),
        text,
        flags=re.DOTALL,
    )

    # Inline $...$: parse but DON'T chain-split (would break inline rendering)
    text = re.sub(r'\$(.+?)\$', lambda m: _parse(m.group(1)), text)

    text = re.sub(
        _BS + r'begin' + _LBRACE + r'([\w*]+)' + _RBRACE
        + r'.*?'
        + _BS + r'end' + _LBRACE + r'\1' + _RBRACE,
        _parse_block,
        text,
        flags=re.DOTALL,
    )

    _CMDS = (
        r'(?:textcolor|color)' + _LBRACE + r'[^}]*' + _RBRACE
        + _LBRACE + r'[^}]*' + _RBRACE
    )
    text = re.sub(_BS + _CMDS, _parse_block, text)

    _CMDS2 = (
        r'(?:textbf|textit|texttt|mathrm|mathbf|mathit|mathsf|mathtt'
        r'|mathcal|mathbb|mathfrak|section|subsection|subsubsection'
        r'|paragraph|huge|Huge|LARGE|Large|large|normalsize|small'
        r'|footnotesize|scriptsize|tiny|underline|uline|sout|cancel'
        r'|emph|text|boxed)'
        + _LBRACE + r'[^}]*' + _RBRACE
    )
    text = re.sub(_BS + _CMDS2, _parse_block, text)

    # Final pass: if the text contains raw LaTeX commands, parse the whole thing
    if re.search(r"\\[a-z]+(?:\{|\(|\[)", text, re.I):
        try:
            text = latex_parser.parse(text)
        except Exception:
            pass

    # Split chained equality/implication steps anywhere in the text
    text = _split_chain(text)

    # Convert newlines before math operators to markdown line breaks
    # (catches steps split across separate $$...$$ blocks)
    text = re.sub(r'\n(?=\s*(?:[⇒=\\]|(?:\\\\[a-z]+)))', '  \n', text)

    return text


def format_for_display(text: str) -> str:
    text = parse_latex(text)
    return text


def strip_prompt_markers(text: str) -> str:
    text = re.sub(r'\[INFO\].*', '', text)
    lines = text.splitlines()
    clean = [l for l in lines if l.strip()]
    return "\n".join(clean).strip()
