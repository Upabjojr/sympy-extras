"""The LaTeX of names: symbols with their subscripts, superscripts, Greek
letters and modifiers, the names of functions, text.

A typed port of the name handling of SymPy's ``printing/latex.py``
(``translate``, ``latex_escape``, the tables of Greek letters and
modifiers); the reading of a name (``split_super_sub``, the walk over the
modifiers) is shared with the pretty printer:
:mod:`sympy_extras.printing.conventions`.
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/printing/latex.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: strict typing; the characters which are special in TeX are escaped
# in the names (sympy/sympy#30844), the escapes of text are terminated.
# ---------------------------------------------------------------------
from __future__ import annotations

import re
from typing import Callable, Optional

from .conventions import GREEK_LETTERS, split_super_sub, translate_name

__all__ = ['ACCEPTED_FUNCTIONS', 'translate', 'split_super_sub', 'escape_text', 'escape_name', 'name_latex',
           'operator_name', 'text']

#: the functions which are commands of LaTeX and MathJax
ACCEPTED_FUNCTIONS = frozenset(['arcsin', 'arccos', 'arctan', 'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'ln',
                                'log', 'sec', 'csc', 'cot', 'coth', 'arg'])

_GREEK = {
    'Alpha': r'\mathrm{A}', 'Beta': r'\mathrm{B}', 'Gamma': r'\Gamma', 'Delta': r'\Delta',
    'Epsilon': r'\mathrm{E}', 'Zeta': r'\mathrm{Z}', 'Eta': r'\mathrm{H}', 'Theta': r'\Theta',
    'Iota': r'\mathrm{I}', 'Kappa': r'\mathrm{K}', 'Lambda': r'\Lambda', 'Mu': r'\mathrm{M}',
    'Nu': r'\mathrm{N}', 'Xi': r'\Xi', 'omicron': 'o', 'Omicron': r'\mathrm{O}', 'Pi': r'\Pi',
    'Rho': r'\mathrm{P}', 'Sigma': r'\Sigma', 'Tau': r'\mathrm{T}', 'Upsilon': r'\Upsilon', 'Phi': r'\Phi',
    'Chi': r'\mathrm{X}', 'Psi': r'\Psi', 'Omega': r'\Omega', 'lamda': r'\lambda', 'Lamda': r'\Lambda',
    'khi': r'\chi', 'Khi': r'\mathrm{X}', 'varepsilon': r'\varepsilon', 'varkappa': r'\varkappa',
    'varphi': r'\varphi', 'varpi': r'\varpi', 'varrho': r'\varrho', 'varsigma': r'\varsigma',
    'vartheta': r'\vartheta',
}

_OTHER_SYMBOLS = frozenset(['aleph', 'beth', 'daleth', 'gimel', 'ell', 'eth', 'hbar', 'hslash', 'mho', 'wp'])


def _wrap(command: str) -> Callable[[str], str]:
    return lambda s: '\\%s{%s}' % (command, s)


def _prime(s: str) -> str:
    return "{%s}'" % s


def _delimiters(left: str, right: str) -> Callable[[str], str]:
    return lambda s: r'\left%s{%s}\right%s' % (left, s, right)


_MODIFIERS: dict[str, Callable[[str], str]] = {
    # accents
    'mathring': _wrap('mathring'), 'ddddot': _wrap('ddddot'), 'dddot': _wrap('dddot'), 'ddot': _wrap('ddot'),
    'dot': _wrap('dot'), 'check': _wrap('check'), 'breve': _wrap('breve'), 'acute': _wrap('acute'),
    'grave': _wrap('grave'), 'tilde': _wrap('tilde'), 'hat': _wrap('hat'), 'bar': _wrap('bar'),
    'vec': _wrap('vec'), 'prime': _prime, 'prm': _prime,
    # faces
    'bold': _wrap('boldsymbol'), 'bm': _wrap('boldsymbol'), 'cal': _wrap('mathcal'), 'scr': _wrap('mathscr'),
    'frak': _wrap('mathfrak'),
    # brackets
    'norm': _delimiters(r'\|', r'\|'), 'avg': _delimiters(r'\langle', r'\rangle'), 'abs': _delimiters('|', '|'),
    'mag': _delimiters('|', '|'),
}

#: the modifiers, the longest names first
_MODIFIER_TABLE: list[tuple[str, Callable[[str], str]]] = [
    (key, _MODIFIERS[key]) for key in sorted(_MODIFIERS, key=len, reverse=True)]


def _letter(s: str) -> Optional[str]:
    """The command of a name which is a letter."""
    greek = _GREEK.get(s)
    if greek is not None:
        return greek
    if s.lower() in GREEK_LETTERS:
        return '\\' + s.lower()
    if s in _OTHER_SYMBOLS:
        return '\\' + s
    return None


#: the characters which are special in TeX wherever they stand (``_``, ``^``,
#: ``{``, ``}`` and ``\`` have a meaning in the names)
_SPECIAL = {'%': r'\%', '#': r'\#', '&': r'\&', '$': r'\$', '~': r'\sim '}


# PORTED FROM SYMPY: sympy/printing/latex.py:translate
def translate(s: str) -> str:
    r"""The LaTeX of a name without scripts: the Greek letters, the
    modifiers at the end of the name (``xdot``, ``xhat``, ``xbold``).

    Examples
    ========

    >>> from sympy_extras.printing._latex_names import translate
    >>> translate('alpha'), translate('Gamma'), translate('xdot'), translate('alphahat'), translate('y')
    ('\\alpha', '\\Gamma', '\\dot{x}', '\\hat{\\alpha}', 'y')
    """
    return translate_name(s, _letter, _MODIFIER_TABLE)


def _balanced(s: str) -> bool:
    depth = 0
    for c in s:
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def escape_name(s: str) -> str:
    r"""A name with the characters which would break the formula escaped:
    ``%``, ``#``, ``&``, ``$``, ``~`` always, the braces when they are not
    balanced. A backslash is kept: a name may be written in LaTeX
    (``\alpha_{1}``).

    Examples
    ========

    >>> from sympy_extras.printing._latex_names import escape_name
    >>> escape_name('a%b'), escape_name('x_{1}'), escape_name('a{b')
    ('a\\%b', 'x_{1}', 'a\\{b')
    """
    for c, escaped in _SPECIAL.items():
        if c in s:
            s = re.sub(r'(?<!\\)' + re.escape(c), lambda m: escaped, s)
    if not _balanced(s):
        s = re.sub(r'(?<!\\)([{}])', r'\\\1', s)
    return s


# PORTED FROM SYMPY: sympy/printing/latex.py:LatexPrinter._deal_with_super_sub
def name_latex(name: str) -> tuple[str, bool]:
    r"""The LaTeX of the name of a symbol and whether it carries a
    superscript: the scripts split and translated (``x_1^2``,
    ``alpha_i``); a name with braces is LaTeX already and is kept.

    Examples
    ========

    >>> from sympy_extras.printing._latex_names import name_latex
    >>> name_latex('x_1'), name_latex('alpha^2_i'), name_latex('x12'), name_latex('a%b')
    (('x_{1}', False), ('\\alpha^{2}_{i}', True), ('x_{12}', False), ('a\\%b', False))
    """
    if '{' in name:
        escaped = escape_name(name)
        return (escaped, '^' in escaped)
    base, supers, subs = split_super_sub(name)
    tex = escape_name(translate(base))
    if supers:
        tex += '^{%s}' % ' '.join(escape_name(translate(s)) for s in supers)
    if subs:
        tex += '_{%s}' % ' '.join(escape_name(translate(s)) for s in subs)
    return (tex, bool(supers))


def operator_name(name: str) -> str:
    r"""The head of a named function: the command of LaTeX when there is
    one, the name in ``\operatorname`` otherwise, whole (SymPy splits the
    digits and the underscores of ``atan2`` and ``jacobi_symbol`` into
    subscripts).

    Examples
    ========

    >>> from sympy_extras.printing._latex_names import operator_name
    >>> operator_name('sin'), operator_name('asinh'), operator_name('atan2'), operator_name('polar_lift')
    ('\\sin', '\\operatorname{asinh}', '\\operatorname{atan2}', '\\operatorname{polar\\_lift}')
    """
    if name in ACCEPTED_FUNCTIONS:
        return '\\' + name
    return r'\operatorname{%s}' % escape_text(name)


_TEXT_ESCAPES = {'\\': r'\textbackslash{}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}',
                 '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}'}


# PORTED FROM SYMPY: sympy/printing/latex.py:latex_escape
def escape_text(s: str) -> str:
    r"""``s`` escaped so that LaTeX reads it as plain text. The commands
    are terminated (SymPy's ``\textbackslash`` swallows the next letter).

    Examples
    ========

    >>> from sympy_extras.printing._latex_names import escape_text
    >>> escape_text('a\\b'), escape_text('x_1 & 50%')
    ('a\\textbackslash{}b', 'x\\_1 \\& 50\\%')
    """
    return ''.join(_TEXT_ESCAPES.get(c, c) for c in s)


def text(s: str) -> str:
    """``s`` as monospace text, what an object without a rule prints as."""
    return r'\mathtt{\text{%s}}' % escape_text(s)
