import re

from latex_parser import parser as latex_parser


def parse_latex(text: str) -> str:
    _BS = re.escape('\\')
    _LBRACE = '{'
    _RBRACE = '}'

    def _parse(inner):
        try:
            return latex_parser.parse(inner)
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
    lines = text.splitlines()
    clean = [l for l in lines if not l.startswith("[INFO]")]
    return "\n".join(clean).strip()
