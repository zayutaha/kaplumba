class LatexParser:
    """Simple LaTeX to text converter for terminal output."""

    # Greek letters
    GREEK = {
        'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'delta': 'δ', 'epsilon': 'ε',
        'zeta': 'ζ', 'eta': 'η', 'theta': 'θ', 'iota': 'ι', 'kappa': 'κ',
        'lambda': 'λ', 'mu': 'μ', 'nu': 'ν', 'xi': 'ξ', 'omicron': 'ο',
        'pi': 'π', 'rho': 'ρ', 'sigma': 'σ', 'tau': 'τ', 'upsilon': 'υ',
        'phi': 'φ', 'chi': 'χ', 'psi': 'ψ', 'omega': 'ω',
        'Gamma': 'Γ', 'Delta': 'Δ', 'Theta': 'Θ', 'Lambda': 'Λ',
        'Xi': 'Ξ', 'Pi': 'Π', 'Sigma': 'Σ', 'Upsilon': 'ϒ',
        'Phi': 'Φ', 'Psi': 'Ψ', 'Omega': 'Ω',
    }

    # Superscript mapping (digits + common letters)
    SUPERSCRIPT = str.maketrans(
        '0123456789abcdefghijklmnoprstuvwxyzABDEGHIJKLMNOPRTUW',
        '⁰¹²³⁴⁵⁶⁷⁸⁹ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻᴬᴮᴰᴱᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾᴿᵀᵁᵂ',
    )
    SUBSCRIPT = str.maketrans(
        '0123456789aehijklmnoprstuvx',
        '₀₁₂₃₄₅₆₇₈₉ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ',
    )

    def __init__(self):
        self.pos = 0
        self.text = ''

    def parse(self, latex: str) -> str:
        """Convert LaTeX to terminal text."""
        self.text = latex
        self.pos = 0
        return self._parse_expr()

    def _peek(self, n=1):
        if self.pos + n <= len(self.text):
            return self.text[self.pos:self.pos + n]
        return ''

    def _advance(self, n=1):
        self.pos += n

    def _parse_expr(self):
        result = []
        while self.pos < len(self.text):
            char = self._peek()
            if char == '\\':
                result.append(self._parse_command())
            elif char == '{':
                self._advance()
                inner = self._parse_until('}')
                self._advance()  # skip }
                result.append(inner)
            elif char == '}':
                break
            elif char == '_':
                self._advance()
                if self._peek() == '{':
                    self._advance()
                    sub = self._parse_expr_inline('}')
                    self._advance()
                else:
                    sub = self._peek(1)
                    self._advance()
                result.append(sub.translate(self.SUBSCRIPT))
            elif char == '^':
                self._advance()
                if self._peek() == '{':
                    self._advance()
                    sup = self._parse_expr_inline('}')
                    self._advance()
                else:
                    sup = self._peek(1)
                    self._advance()
                result.append(sup.translate(self.SUPERSCRIPT))
            else:
                result.append(char)
                self._advance()
        return ''.join(result)

    def _parse_until(self, end):
        start = self.pos
        depth = 1
        while self.pos < len(self.text):
            char = self._peek()
            if char == end and depth == 1:
                break
            elif char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
            self._advance()
        return self.text[start:self.pos]

    def _parse_expr_inline(self, end_char):
        """Parse a LaTeX expression until we hit end_char.
        
        Unlike _parse_until which returns raw text, this recursively
        processes commands (\\alpha, \\pi, etc.) inside the braces.
        """
        result = []
        depth = 1
        while self.pos < len(self.text):
            char = self._peek()
            if char == end_char and depth == 1:
                break
            elif char == '{':
                depth += 1
                self._advance()
                inner = self._parse_expr_inline('}')
                self._advance()  # skip }
                result.append(inner)
            elif char == '}':
                depth -= 1
                if depth == 0:
                    break
                result.append(char)
                self._advance()
            elif char == '\\':
                result.append(self._parse_command())
            elif char == '_':
                self._advance()
                if self._peek() == '{':
                    self._advance()
                    sub = self._parse_expr_inline('}')
                    self._advance()
                else:
                    sub = self._peek(1)
                    self._advance()
                result.append(sub.translate(self.SUBSCRIPT))
            elif char == '^':
                self._advance()
                if self._peek() == '{':
                    self._advance()
                    sup = self._parse_expr_inline('}')
                    self._advance()
                else:
                    sup = self._peek(1)
                    self._advance()
                result.append(sup.translate(self.SUPERSCRIPT))
            else:
                result.append(char)
                self._advance()
        return ''.join(result)

    def _parse_command(self):
        self._advance()  # skip backslash
        # Read command name
        cmd = ''
        while self.pos < len(self.text) and self._peek().isalpha():
            cmd += self._peek()
            self._advance()

        # Handle commands
        if cmd in self.GREEK:
            return self.GREEK[cmd]
        elif cmd == 'frac':
            return self._parse_frac()
        elif cmd == 'sqrt':
            return self._parse_sqrt()
        elif cmd == 'int':
            return '∫'
        elif cmd == 'sum':
            return '∑'
        elif cmd == 'prod':
            return '∏'
        elif cmd == 'infty':
            return '∞'
        elif cmd == 'pm':
            return '±'
        elif cmd == 'mp':
            return '∓'
        elif cmd == 'times':
            return '×'
        elif cmd == 'div':
            return '÷'
        elif cmd == 'neq':
            return '≠'
        elif cmd == 'leq':
            return '≤'
        elif cmd == 'geq':
            return '≥'
        elif cmd == 'approx':
            return '≈'
        elif cmd == 'equiv':
            return '≡'
        elif cmd == 'partial':
            return '∂'
        elif cmd == 'nabla':
            return '∇'
        elif cmd == 'exists':
            return '∃'
        elif cmd == 'forall':
            return '∀'
        elif cmd == 'in':
            return '∈'
        elif cmd == 'notin':
            return '∉'
        elif cmd == 'subset':
            return '⊂'
        elif cmd == 'supset':
            return '⊃'
        elif cmd == 'cap':
            return '∩'
        elif cmd == 'cup':
            return '∪'
        elif cmd == 'emptyset':
            return '∅'
        elif cmd == 'to' or cmd == 'rightarrow':
            return '→'
        elif cmd == 'leftarrow':
            return '←'
        elif cmd == 'leftrightarrow':
            return '↔'
        elif cmd == 'Rightarrow':
            return '⇒'
        elif cmd == 'Leftarrow':
            return '⇐'
        elif cmd == 'Leftrightarrow':
            return '⇔'
        elif cmd == 'cdot':
            return '·'
        elif cmd == 'ldots':
            return '…'
        elif cmd == 'cdots':
            return '⋯'
        elif cmd in ('sin', 'cos', 'tan', 'log', 'ln'):
            return cmd + ' '
        else:
            return f"\\{cmd}"

    def _parse_frac(self):
        # Parse numerator
        if self._peek() == '{':
            self._advance()
            num = self._parse_expr_inline('}')
            self._advance()
        else:
            num = self._peek(1)
            self._advance()

        # Parse denominator
        if self._peek() == '{':
            self._advance()
            den = self._parse_expr_inline('}')
            self._advance()
        else:
            den = self._peek(1)
            self._advance()

        return f"({num})/({den})"

    def _parse_sqrt(self):
        # Check for optional root
        root = None
        if self._peek() == '[':
            self._advance()
            root = self._parse_expr_inline(']')
            self._advance()

        if self._peek() == '{':
            self._advance()
            content = self._parse_expr_inline('}')
            self._advance()
        else:
            content = self._peek(1)
            self._advance()

        if root:
            return f"{root}√({content})"
        return f"√({content})"


# Global parser instance
parser = LatexParser()
