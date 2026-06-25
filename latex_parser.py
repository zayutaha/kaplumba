"""Comprehensive LaTeX to terminal Unicode converter."""


class LatexParser:
    """LaTeX to terminal Unicode — covers math, matrices, cases, fonts, accents, etc."""

    GREEK = {
        'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'delta': 'δ', 'epsilon': 'ε',
        'zeta': 'ζ', 'eta': 'η', 'theta': 'θ', 'iota': 'ι', 'kappa': 'κ',
        'lambda': 'λ', 'mu': 'μ', 'nu': 'ν', 'xi': 'ξ', 'omicron': 'ο',
        'pi': 'π', 'rho': 'ρ', 'sigma': 'σ', 'tau': 'τ', 'upsilon': 'υ',
        'phi': 'φ', 'chi': 'χ', 'psi': 'ψ', 'omega': 'ω',
        'Gamma': 'Γ', 'Delta': 'Δ', 'Theta': 'Θ', 'Lambda': 'Λ',
        'Xi': 'Ξ', 'Pi': 'Π', 'Sigma': 'Σ', 'Upsilon': 'Υ',
        'Phi': 'Φ', 'Psi': 'Ψ', 'Omega': 'Ω',
        'varepsilon': 'ε', 'vartheta': 'ϑ', 'varpi': 'ϖ', 'varrho': 'ϱ',
        'varsigma': 'ς', 'varphi': 'φ',
    }

    SUPERSCRIPT = str.maketrans(
        '0123456789abcdefghijklmnoprstuvwxyzABDEGHIJKLMNOPRTUW+-',
        '⁰¹²³⁴⁵⁶⁷⁸⁹ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻᴬᴮᴰᴱᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾᴿᵀᵁᵂ⁺⁻',
    )
    SUBSCRIPT = str.maketrans(
        '0123456789aehijklmnoprstuvx+-',
        '₀₁₂₃₄₅₆₇₈₉ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ₊₋',
    )
    SUBSCRIPT = str.maketrans(
        '0123456789aehijklmnoprstuvx',
        '₀₁₂₃₄₅₆₇₈₉ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ',
    )

    def __init__(self):
        self.pos = 0
        self.text = ''

    def parse(self, latex: str) -> str:
        self.text = latex
        self.pos = 0
        result = self._parse_expr()
        # Strip extra whitespace
        return result.strip()

    def _peek(self, n=1):
        if self.pos + n <= len(self.text):
            return self.text[self.pos:self.pos + n]
        return ''

    def _advance(self, n=1):
        self.pos += n

    def _parse_expr(self, end_chars=None):
        """Parse a LaTeX expression until end_chars or end of string."""
        if end_chars is None:
            end_chars = set()
        result = []
        while self.pos < len(self.text):
            char = self._peek()
            if char in end_chars:
                break
            if char == '\\':
                result.append(self._parse_command())
            elif char == '{':
                self._advance()
                inner = self._parse_expr({'}'})
                self._advance()  # skip }
                result.append(inner)
            elif char == '}':
                break
            elif char == '$':
                # bare $ — skip (already handled by caller)
                self._advance()
            elif char == '_':
                result.append(self._parse_subscript())
            elif char == '^':
                result.append(self._parse_superscript())
            elif char == '%':
                # Comment to end of line
                while self.pos < len(self.text) and self._peek() != '\n':
                    self._advance()
            elif char == '~':
                result.append(' ')
                self._advance()
            elif char == '\n':
                result.append(' ')
                self._advance()
            else:
                result.append(char)
                self._advance()
        return ''.join(result)

    def _parse_expr_inline(self, end_char):
        """Parse expression until end_char, with recursive command handling."""
        return self._parse_expr({end_char})

    def _parse_until(self, end):
        """Return raw text until end char (no command parsing)."""
        start = self.pos
        depth = 1
        while self.pos < len(self.text):
            c = self._peek()
            if c == end and depth == 1:
                break
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
            self._advance()
        return self.text[start:self.pos]

    def _parse_subscript(self):
        self._advance()  # skip _
        if self._peek() == '{':
            self._advance()
            inner = self._parse_expr({'}'})
            self._advance()
        else:
            inner = self._peek(1)
            self._advance()
        return inner.translate(self.SUBSCRIPT)

    def _parse_superscript(self):
        self._advance()  # skip ^
        if self._peek() == '{':
            self._advance()
            inner = self._parse_expr({'}'})
            self._advance()
        else:
            inner = self._peek(1)
            self._advance()
        return inner.translate(self.SUPERSCRIPT)

    def _parse_text_arg(self):
        """Parse \text{...} or similar text command with braces."""
        char = self._peek()
        if char == '{':
            self._advance()
            inner = self._parse_expr({'}'})
            self._advance()
            return inner
        return ''

    def _parse_opt_arg(self):
        """Parse optional [...] argument."""
        if self._peek() == '[':
            self._advance()
            inner = self._parse_expr({']'})
            self._advance()
            return inner
        return None

    def _parse_frac(self):
        num = self._parse_text_arg() or self._peek(1) or ''
        if not self.text or self.pos < len(self.text):
            den = self._parse_text_arg() or self._peek(1) or ''
        else:
            den = ''
        return f"{num}/{den}"

    def _parse_sqrt(self):
        root = self._parse_opt_arg()
        content = self._parse_text_arg() or self._peek(1) or ''
        if root:
            return f"{root}√({content})"
        return f"√({content})"

    def _parse_binom(self):
        n = self._parse_text_arg()
        k = self._parse_text_arg()
        return f"C({n},{k})"

    def _parse_limits(self):
        # Parse limits: _{...}^{...} after operators like \sum, \int, \lim
        lower = upper = ''
        if self._peek() == '_':
            self._advance()
            if self._peek() == '{':
                lower = self._parse_text_arg()
            else:
                lower = self._peek(1)
                self._advance()
        if self._peek() == '^':
            self._advance()
            if self._peek() == '{':
                upper = self._parse_text_arg()
            else:
                upper = self._peek(1)
                self._advance()
        lower = lower.translate(self.SUBSCRIPT)
        upper = upper.translate(self.SUPERSCRIPT)
        return lower + upper

    def _parse_environment(self, env_name):
        # Parse begin{env_name} ... end{env_name}
        # Skip optional argument like {2} in \begin{alignat*}{2}
        if self._peek() == '{':
            self._advance()
            while self.pos < len(self.text) and self._peek() != '}':
                self._advance()
            self._advance()
        end_tag = '\\end{' + env_name + '}'
        end_pos = self.text.find(end_tag, self.pos)
        if end_pos == -1:
            return self._parse_expr()
        raw = self.text[self.pos:end_pos].strip()
        self.pos = end_pos + len(end_tag)

        # Text environments handled differently (no \\ row splitting)
        text_envs = ('center', 'flushleft', 'flushright', 'quote',
                     'quotation', 'verbatim', 'comment')
        if env_name in text_envs:
            if env_name == 'verbatim':
                return '```\n' + raw + '\n```'
            if env_name == 'comment':
                return ''
            saved = self.pos, self.text
            self.text = raw
            self.pos = 0
            parsed = self._parse_expr()
            self.pos, self.text = saved
            if env_name in ('quote', 'quotation'):
                return '\n'.join('> ' + line for line in parsed.split('\n'))
            return parsed  # center, flushleft, flushright

        raw = raw.replace('\\hline', '')
        raw_rows = [r.strip() for r in raw.split('\\\\')]
        # Choose separator: alignments get just spaces, others get visible | 
        is_align = env_name in ('aligned', 'align', 'align*', 'alignat',
                                'alignat*', 'flalign', 'flalign*')
        sep = ' ' if is_align else (' | ' if env_name != 'cases' else ' ')
        parsed_rows = []
        for row in raw_rows:
            if not is_align:
                row = row.replace('&', sep)
            saved = self.pos, self.text
            self.text = row
            self.pos = 0
            parsed = self._parse_expr()
            self.pos, self.text = saved
            parsed_rows.append(parsed)
        return self._render_environment(env_name, parsed_rows)

    def _render_environment(self, env_name, parsed_rows):
        """Render parsed environment rows as readable text.
        
        Args:
            env_name: environment name (matrix, pmatrix, cases, etc.)
            parsed_rows: list of strings, each a row with LaTeX already resolved.
                         Cell separators are ' | '.
        """
        base_envs = ('matrix', 'pmatrix', 'bmatrix', 'vmatrix', 'Vmatrix', 'array')
        if env_name in base_envs:
            parts = []
            for row in parsed_rows:
                cells = [c.strip() for c in row.split(' | ') if c.strip()]
                parts.append('  '.join(cells))
            delim = {'pmatrix': '()', 'bmatrix': '[]', 'vmatrix': '||',
                     'Vmatrix': '‖‖', 'matrix': '[]', 'array': '()'}
            l, r = delim.get(env_name, ('[', ']'))
            return l + ' | '.join(parts) + r

        if env_name == 'cases':
            parts = []
            for row in parsed_rows:
                cells = [c.strip() for c in row.split(' | ') if c.strip()]
                if len(cells) >= 2:
                    r_cond = cells[1]
                    if r_cond.startswith('if '):
                        r_cond = r_cond[3:].lstrip()
                    parts.append(f'{cells[0]}  if {r_cond}')
                else:
                    parts.append(cells[0])
            return '{ ' + ', '.join(parts) + ' ]'

        if env_name in ('aligned', 'align', 'align*', 'alignat', 'alignat*',
                        'flalign', 'flalign*'):
            return self._render_aligned(parsed_rows)

        if env_name == 'gather':
            return '  '.join(parsed_rows) if len(parsed_rows) <= 2 else '\n'.join(f'  {r}' for r in parsed_rows)

        if env_name == 'multline':
            return '\n'.join(f'  {r}' for r in parsed_rows)

        return '  '.join(parsed_rows)

    def _split_rows(self, content):
        """Split environment content into rows on \\\\."""
        rows = []
        current = ''
        i = 0
        while i < len(content):
            if content[i:i+2] == '\\\\':
                rows.append(current.strip())
                current = ''
                i += 2
            elif content[i] == '\\' and i+1 < len(content) and content[i+1] == '\\':
                rows.append(current.strip())
                current = ''
                i += 2
            else:
                current += content[i]
                i += 1
        if current.strip():
            rows.append(current.strip())
        return rows

    def _render_matrix(self, rows, env_name):
        """Render matrix to a readable one-line form."""
        parsed = []
        for row in rows:
            cells = [c.strip() for c in row.split('&')]
            parsed.append([self._render_cell(c) for c in cells])
        delim_l, delim_r = {'matrix': ('[', ']'), 'pmatrix': ('(', ')'),
                            'bmatrix': ('[', ']'), 'vmatrix': ('|', '|'),
                            'Vmatrix': ('‖', '‖')}.get(env_name, ('[', ']'))
        flat = []
        for row in parsed:
            flat.extend(row)
        return delim_l + '  '.join(flat) + delim_r

    def _render_cell(self, cell_text):
        """Parse a matrix cell (which may contain LaTeX commands)."""
        # Simple: just feed through the parser
        saved_pos = self.pos
        saved_text = self.text
        self.text = cell_text
        self.pos = 0
        result = self._parse_expr()
        self.text = saved_text
        self.pos = saved_pos
        return result

    def _render_cases(self, rows):
        """Render cases environment.
        
        Receives pre-parsed rows: expression and condition are
        separated by multiple spaces (from & replacement).
        """
        parts = []
        for row in rows:
            # Find first run of 2+ spaces as separator between expr and cond
            import re
            cells = re.split(r'  +', row.strip(), maxsplit=1)
            if len(cells) >= 2:
                r_expr, r_cond = cells[0], cells[1]
                if r_cond.startswith('if '):
                    r_cond = r_cond[3:].lstrip()
                parts.append(f'{r_expr}  if {r_cond}')
            else:
                parts.append(row.strip())
        return '{ ' + ', '.join(parts) + ' ]'

    def _render_aligned(self, rows):
        """Render aligned environment as multi-line."""
        lines = []
        for row in rows:
            cells = [c.strip() for c in row.split('&')]
            rendered = [self._render_cell(c) for c in cells]
            line = ' '.join(rendered)
            lines.append(f"  {line}")
        return '\n'.join(lines)

    def _parse_command(self):
        self._advance()  # skip backslash
        # Check for \\ (double backslash = line break)
        if self._peek() == '\\':
            self._advance()
            # Skip optional [spacing] argument
            if self._peek() == '[':
                self._advance()
                while self.pos < len(self.text) and self._peek() != ']':
                    self._advance()
                self._advance()
            return '\n'
        cmd = ''
        while self.pos < len(self.text) and self._peek().isalpha():
            cmd += self._peek()
            self._advance()

        if not cmd:
            # Non-alpha command: \, \:, \;, \!, \#, \$, \%, \&, \_, \{, \}
            char = self._peek(1) if self.pos < len(self.text) else ''
            if char:
                self._advance()
            # Spacing commands
            if char in (',', ':', ';', '!'):
                return ' '
            # \| = double vertical bar (norms)
            if char == '|':
                return '‖'
            return char or ''

        # === Greek letters ===
        if cmd in self.GREEK:
            return self.GREEK[cmd]

        # === Fractions, roots, binomials ===
        if cmd == 'frac':
            return self._parse_frac()
        if cmd == 'sqrt':
            return self._parse_sqrt()
        if cmd == 'binom':
            return self._parse_binom()

        # === Spacing ===
        if cmd in ('quad', 'qquad'):
            return '  '
        if cmd in (',', ':', ';', '!'):
            return ' '
        if cmd == ' ':
            return ' '

        # === Text ===
        if cmd == 'text':
            return self._parse_text_arg()

        # === Font commands ===
        if cmd in ('textbf', 'textit', 'texttt', 'mathrm', 'mathbf',
                   'mathit', 'mathsf', 'mathtt', 'mathcal', 'mathbb',
                   'mathfrak', 'normal'):
            return self._parse_text_arg()

        # === Font sizes (strip — terminal has fixed font) ===
        if cmd in ('tiny', 'scriptsize', 'footnotesize', 'small', 'normalsize',
                   'large', 'Large', 'LARGE', 'huge', 'Huge'):
            return self._parse_text_arg()

        # === Color ===
        if cmd == 'textcolor':
            self._parse_text_arg()  # color name
            return self._parse_text_arg()  # content
        if cmd == 'color':
            self._parse_text_arg()  # color name (declaration)
            return ''

        # === Large operators ===
        if cmd == 'sum':
            return '∑' + self._parse_limits()
        if cmd == 'prod':
            return '∏' + self._parse_limits()
        if cmd == 'int':
            return '∫' + self._parse_limits()
        if cmd == 'iint':
            return '∬' + self._parse_limits()
        if cmd == 'iiint':
            return '∭' + self._parse_limits()
        if cmd == 'oint':
            return '∮' + self._parse_limits()
        if cmd == 'lim':
            return 'lim' + self._parse_limits()
        if cmd == 'inf':
            return 'inf'

        # === Arrows ===
        if cmd == 'to' or cmd == 'rightarrow':
            return '→'
        if cmd == 'leftarrow':
            return '←'
        if cmd == 'leftrightarrow':
            return '↔'
        if cmd == 'Rightarrow':
            return '⇒'
        if cmd == 'Leftarrow':
            return '⇐'
        if cmd == 'Leftrightarrow':
            return '⇔'
        if cmd == 'mapsto':
            return '↦'
        if cmd == 'implies':
            return '⇒'
        if cmd == 'iff':
            return '⇔'
        if cmd == 'uparrow':
            return '↑'
        if cmd == 'downarrow':
            return '↓'
        if cmd == 'nearrow':
            return '↗'
        if cmd == 'searrow':
            return '↘'
        if cmd == 'hookrightarrow':
            return '↪'
        if cmd == 'hookleftarrow':
            return '↩'
        if cmd == 'rightharpoonup':
            return '⇀'
        if cmd == 'leftharpoonup':
            return '↼'

        # === Relations ===
        if cmd == 'neq':
            return '≠'
        if cmd == 'leq':
            return '≤'
        if cmd == 'geq':
            return '≥'
        if cmd == 'approx':
            return '≈'
        if cmd == 'equiv':
            return '≡'
        if cmd == 'sim':
            return '∼'
        if cmd == 'simeq':
            return '≃'
        if cmd == 'cong':
            return '≅'
        if cmd == 'propto':
            return '∝'
        if cmd == 'subset':
            return '⊂'
        if cmd == 'supset':
            return '⊃'
        if cmd == 'subseteq':
            return '⊆'
        if cmd == 'supseteq':
            return '⊇'
        if cmd == 'in':
            return '∈'
        if cmd == 'notin':
            return '∉'
        if cmd == 'ni' or cmd == 'owns':
            return '∋'
        if cmd == 'models':
            return '⊧'
        if cmd == 'perp':
            return '⊥'
        if cmd == 'parallel':
            return '∥'
        if cmd == 'mid':
            return '|'
        if cmd == 'smile':
            return '⌣'
        if cmd == 'frown':
            return '⌢'
        if cmd == 'bowtie':
            return '⋈'
        if cmd == 'models':
            return '⊧'
        if cmd == 'perp':
            return '⊥'
        if cmd == 'parallel':
            return '∥'

        # === Additional Relations ===
        if cmd == 'nmid':
            return '∤'
        if cmd == 'nparallel':
            return '∦'
        if cmd == 'vdash':
            return '⊢'
        if cmd == 'dashv':
            return '⊣'
        if cmd == 'top':
            return '⊤'
        if cmd == 'bot':
            return '⊥'
        if cmd == 'therefore':
            return '∴'
        if cmd == 'because':
            return '∵'
        if cmd == 'triangleleft':
            return '◃'
        if cmd == 'triangleright':
            return '▹'
        if cmd == 'trianglelefteq':
            return '⊴'
        if cmd == 'trianglerighteq':
            return '⊵'
        if cmd == 'wr':
            return '≀'

        # === More Arrows ===
        if cmd == 'longleftarrow':
            return '⟵'
        if cmd == 'longrightarrow':
            return '⟶'
        if cmd == 'longleftrightarrow':
            return '⟷'
        if cmd == 'Longleftarrow':
            return '⟸'
        if cmd == 'Longrightarrow':
            return '⟹'
        if cmd == 'Longleftrightarrow':
            return '⟺'
        if cmd == 'longmapsto':
            return '⟼'
        if cmd == 'updownarrow':
            return '↕'
        if cmd == 'Updownarrow':
            return '⇕'
        if cmd == 'twoheadrightarrow':
            return '↠'
        if cmd == 'twoheadleftarrow':
            return '↞'

        # === Logic ===
        if cmd in ('lnot', 'neg'):
            return '¬'
        if cmd == 'land':
            return '∧'
        if cmd == 'lor':
            return '∨'

        # === Geometry / Shapes ===
        if cmd == 'angle':
            return '∠'
        if cmd == 'measuredangle':
            return '∡'
        if cmd == 'triangle':
            return '△'
        if cmd == 'Box':
            return '□'
        if cmd == 'blacksquare':
            return '■'
        if cmd == 'bigcirc':
            return '○'
        if cmd == 'checkmark':
            return '✓'

        # === Card suits ===
        if cmd == 'spadesuit':
            return '♠'
        if cmd == 'heartsuit':
            return '♡'
        if cmd == 'diamondsuit':
            return '♢'
        if cmd == 'clubsuit':
            return '♣'

        # === Other symbols ===
        if cmd == 'flat':
            return '♭'
        if cmd == 'natural':
            return '♮'
        if cmd == 'sharp':
            return '♯'
        if cmd == 'aleph':
            return 'ℵ'
        if cmd == 'beth':
            return 'ℶ'
        if cmd == 'wp':
            return '℘'

        # === Set theory ===
        if cmd == 'emptyset':
            return '∅'
        if cmd == 'varnothing':
            return '∅'

        # === Operators ===
        if cmd == 'pm':
            return '±'
        if cmd == 'mp':
            return '∓'
        if cmd == 'times':
            return '×'
        if cmd == 'div':
            return '÷'
        if cmd == 'cdot':
            return '·'
        if cmd == 'circ':
            return '∘'
        if cmd == 'bullet':
            return '•'
        if cmd == 'wedge':
            return '∧'
        if cmd == 'vee':
            return '∨'
        if cmd == 'cap':
            return '∩'
        if cmd == 'cup':
            return '∪'
        if cmd == 'uplus':
            return '⊎'
        if cmd == 'sqcap':
            return '⊓'
        if cmd == 'sqcup':
            return '⊔'
        if cmd == 'oplus':
            return '⊕'
        if cmd == 'ominus':
            return '⊖'
        if cmd == 'otimes':
            return '⊗'
        if cmd == 'oslash':
            return '⊘'
        if cmd == 'odot':
            return '⊙'
        if cmd == 'dagger':
            return '†'
        if cmd == 'ddagger':
            return '‡'
        if cmd == 'star':
            return '⋆'
        if cmd == 'ast':
            return '∗'
        if cmd == 'amalg':
            return '⨿'

        # === Calculus ===
        if cmd == 'partial':
            return '∂'
        if cmd == 'nabla':
            return '∇'
        if cmd == 'infty':
            return '∞'
        if cmd == 'exists':
            return '∃'
        if cmd == 'forall':
            return '∀'
        if cmd == 'nexists':
            return '∄'
        if cmd == 'Im':
            return 'ℑ'
        if cmd == 'Re':
            return 'ℜ'
        if cmd == 'imath':
            return 'ı'
        if cmd == 'jmath':
            return 'ȷ'
        if cmd == 'ell':
            return 'ℓ'
        if cmd == 'hbar':
            return 'ℏ'
        if cmd == 'prime':
            return '′'

        # === Dots ===
        if cmd in ('ldots', 'dots'):
            return '…'
        if cmd == 'cdots':
            return '⋯'
        if cmd == 'vdots':
            return '⋮'
        if cmd == 'ddots':
            return '⋱'

        # === Accents (prefix-style: hat → x̂, bar → x̄, etc.) ===
        if cmd == 'hat':
            inner = self._parse_text_arg()
            return '^' + inner
        if cmd == 'tilde':
            inner = self._parse_text_arg()
            return '~' + inner
        if cmd == 'bar':
            inner = self._parse_text_arg()
            return '¯' + inner
        if cmd == 'dot':
            inner = self._parse_text_arg()
            return '˙' + inner
        if cmd == 'ddot':
            inner = self._parse_text_arg()
            return '¨' + inner
        if cmd == 'vec':
            inner = self._parse_text_arg()
            return '→' + inner
        if cmd == 'widehat':
            return '^' + self._parse_text_arg()
        if cmd == 'widetilde':
            return '~' + self._parse_text_arg()
        if cmd == 'overline':
            return '‾' + self._parse_text_arg()

        # === Over/under braces ===
        if cmd == 'overbrace':
            content = self._parse_text_arg()
            label = self._parse_superscript() if self._peek() == '^' else ''
            return f"({content})︷{label}"
        if cmd == 'underbrace':
            content = self._parse_text_arg()
            label = self._parse_subscript() if self._peek() == '_' else ''
            return f"({content})︸{label}"

        # === Boxed ===
        if cmd == 'boxed':
            return '[ ' + self._parse_text_arg() + ' ]'

        # === Sectioning → Markdown headings ===
        if cmd == 'section':
            return '## ' + self._parse_text_arg()
        if cmd == 'subsection':
            return '### ' + self._parse_text_arg()
        if cmd == 'subsubsection':
            return '#### ' + self._parse_text_arg()
        if cmd == 'paragraph':
            return '**' + self._parse_text_arg() + '**'

        # === Underline / Strikethrough / Emphasis ===
        if cmd in ('underline', 'uline'):
            return self._parse_text_arg()
        if cmd in ('sout', 'strikethrough', 'cancel', 'xcancel'):
            return self._parse_text_arg()
        if cmd == 'emph':
            return self._parse_text_arg()

        # === Stackrel ===
        if cmd == 'stackrel':
            top = self._parse_text_arg()
            bottom = self._parse_text_arg()
            return f"{bottom}"  # simplified

        # === Left/Right delimiters ===
        if cmd in ('left', 'right'):
            d = self._peek(1) if self.pos < len(self.text) else ''
            self._advance()
            # Handle multi-char delimiters: \|, \langle, \rfloor, etc.
            if d == '\\':
                d_cmd = ''
                while self.pos < len(self.text) and self._peek().isalpha():
                    d_cmd += self._peek()
                    self._advance()
                d = d_cmd or '\\'
            return self._map_delim(d)

        # === Big delimiters (\\big, \\Big, \\bigg, \\Bigg) ===
        if cmd in ('big', 'Big', 'bigg', 'Bigg',
                   'bigl', 'Bigl', 'biggl', 'Biggl',
                   'bigr', 'Bigr', 'biggr', 'Biggr',
                   'bigm', 'Bigm', 'biggm', 'Biggm'):
            d = self._peek(1) if self.pos < len(self.text) else ''
            self._advance()
            # Handle multi-char delimiter: \|
            if d == '\\':
                d_cmd = ''
                while self.pos < len(self.text) and self._peek().isalpha():
                    d_cmd += self._peek()
                    self._advance()
                d = d_cmd or '\\'
            return self._map_delim(d)

        # === Over/under (hat-like for text) ===
        if cmd == 'over':
            return '/'
        if cmd == 'choose':
            return ' / '

        # === Environments ===
        if cmd == 'begin':
            env_name = self._parse_text_arg()
            return self._parse_environment(env_name)

        if cmd == 'end':
            # Already consumed by _parse_environment
            return ''

        # === Math mode toggle (for inline) ===
        if cmd in ('(', ')'):
            return cmd

        # === Named operators ===
        known_ops = {'sin': 'sin', 'cos': 'cos', 'tan': 'tan', 'cot': 'cot',
                     'sec': 'sec', 'csc': 'csc', 'arcsin': 'arcsin',
                     'arccos': 'arccos', 'arctan': 'arctan', 'sinh': 'sinh',
                     'cosh': 'cosh', 'tanh': 'tanh', 'coth': 'coth',
                     'log': 'log', 'ln': 'ln', 'lg': 'lg', 'exp': 'exp',
                     'det': 'det', 'dim': 'dim', 'ker': 'ker', 'hom': 'hom',
                     'max': 'max', 'min': 'min', 'sup': 'sup', 'inf': 'inf',
                     'limsup': 'limsup', 'liminf': 'liminf',
                     'arg': 'arg', 'deg': 'deg',
                     'Pr': 'Pr', 'Var': 'Var', 'Cov': 'Cov',
                     'mod': ' mod ', 'pmod': ' (mod ', 'bmod': ' mod '}
        if cmd in known_ops:
            return known_ops[cmd]

        # === Unrecognized command ===
        return f"\\{cmd}"

    def _map_delim(self, char):
        """Map delimiter character/name to display form."""
        mapping = {
            '(': '(', ')': ')', '[': '[', ']': ']', '.': '',
            '{': '{', '}': '}',
            '|': '|', '\\': '|', '|': '|',
            '/': '/',
            'langle': '⟨', 'rangle': '⟩',
            'lfloor': '⌊', 'rfloor': '⌋',
            'lceil': '⌈', 'rceil': '⌉',
            'lbrace': '{', 'rbrace': '}',
            'lbracket': '[', 'rbracket': ']',
        }
        return mapping.get(char, char)


# Global parser instance
parser = LatexParser()
