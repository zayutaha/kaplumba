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
    """Split chained equality/implication steps onto separate lines."""
    lines = text.splitlines()
    out = []
    for line in lines:
        # Check for implication operators first (two or more → chain)
        imps = list(_CHAIN_OPS.finditer(line))
        if len(imps) >= 2:
            rebuilt = line[:imps[0].start()]
            for i, m in enumerate(imps):
                start = m.end()
                end = imps[i + 1].start() if i + 1 < len(imps) else len(line)
                rebuilt += "\n" + m.group(0) + " " + line[start:end].lstrip()
            line = rebuilt

        # Check for plain equals signs (two or more → chain)
        eqs = _PLAIN_EQ.findall(line)
        if len(eqs) >= 2:
            parts = _PLAIN_EQ.split(line)
            rebuilt = parts[0]
            for p in parts[1:]:
                rebuilt += "\n= " + p.lstrip()
            line = rebuilt

        out.append(line)
    return "\n".join(out)


def parse_latex(text: str) -> str:
    _BS = re.escape('\\')
    _LBRACE = '{'
    _RBRACE = '}'

    def _parse(inner):
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
        lambda m: _parse(m.group(1)),
        text,
        flags=re.DOTALL,
    )

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

    return text


def format_for_display(text: str) -> str:
    text = parse_latex(text)
    return text


def strip_prompt_markers(text: str) -> str:
    text = re.sub(r'\[INFO\].*', '', text)
    lines = text.splitlines()
    clean = [l for l in lines if l.strip()]
    return "\n".join(clean).strip()
