"""The characters of the pretty printer, in Unicode and in ASCII.

A typed port of SymPy's ``sympy/printing/pretty/pretty_symbology.py``. A
:class:`Symbology` holds the choice between Unicode and ASCII (SymPy
keeps it in a module-level flag which ``pretty`` sets and restores); the
printer carries one and nothing here is global state.

References
==========

.. [1] SymPy Development Team, *SymPy documentation: Printing*,
       https://docs.sympy.org/latest/modules/printing.html
.. [2] The Unicode Consortium, *The Unicode Standard*, chapters 22.2
       (mathematical alphanumeric symbols) and 22.7 (technical symbols:
       the bracket pieces), https://www.unicode.org/versions/latest/
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/printing/pretty/pretty_symbology.py, sympy/printing/conventions.py (split_super_sub) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: strict typing; the Unicode/ASCII choice is a field of a
# Symbology object instead of a global flag; the extensible objects are a
# typed record instead of nested tuples; a name with an empty sub- or
# superscript (``x_``) keeps its marks (sympy/sympy#30844).
# ---------------------------------------------------------------------
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Optional

from ...core.alphabets import greeks

__all__ = ['Symbology', 'Extensible', 'Annotated', 'line_width', 'center', 'center_pad', 'split_super_sub',
           'greek_unicode', 'sub', 'sup']


def _u(name: str) -> str:
    return unicodedata.lookup(name)


def _greek_table() -> dict[str, str]:
    letters = ['lamda' if name == 'lambda' else name for name in greeks]
    table: dict[str, str] = {name: _u('GREEK SMALL LETTER %s' % name.upper()) for name in letters}
    for name in letters:
        table[name[0].upper() + name[1:]] = _u('GREEK CAPITAL LETTER %s' % name.upper())
    table['lambda'] = table['lamda']
    table['Lambda'] = table['Lamda']
    table['varsigma'] = '\N{GREEK SMALL LETTER FINAL SIGMA}'
    return table


#: the Greek letters by name (``lambda`` and ``lamda`` both)
greek_unicode: dict[str, str] = _greek_table()

_DIGITS = {'0': 'ZERO', '1': 'ONE', '2': 'TWO', '3': 'THREE', '4': 'FOUR', '5': 'FIVE', '6': 'SIX', '7': 'SEVEN',
           '8': 'EIGHT', '9': 'NINE'}
_SIGNS = {'+': 'PLUS SIGN', '-': 'MINUS', '=': 'EQUALS SIGN', '(': 'LEFT PARENTHESIS', ')': 'RIGHT PARENTHESIS'}


def _script_tables() -> tuple[dict[str, str], dict[str, str]]:
    lower: dict[str, str] = {}
    upper: dict[str, str] = {}
    for letter in 'aeioruvxhklmnpst':
        lower[letter] = _u('LATIN SUBSCRIPT SMALL LETTER %s' % letter.upper())
    for letter in 'in':
        upper[letter] = _u('SUPERSCRIPT LATIN SMALL LETTER %s' % letter.upper())
    for name in ('beta', 'gamma', 'rho', 'phi', 'chi'):
        lower[name] = _u('GREEK SUBSCRIPT SMALL LETTER %s' % name.upper())
    for digit, text in _DIGITS.items():
        lower[digit] = _u('SUBSCRIPT %s' % text)
        upper[digit] = _u('SUPERSCRIPT %s' % text)
    for sign, text in _SIGNS.items():
        lower[sign] = _u('SUBSCRIPT %s' % text)
        upper[sign] = _u('SUPERSCRIPT %s' % text)
    return lower, upper


#: the characters with a Unicode subscript form, and with a superscript form
sub, sup = _script_tables()


def _center_accent(string: str, accent: str) -> str:
    """``string`` with the combining ``accent`` on its middle character."""
    midpoint = len(string)//2 + 1
    return string[:midpoint] + accent + string[midpoint:]


def _accent(mark: str) -> Callable[[str], str]:
    def apply(s: str) -> str:
        return _center_accent(s, mark)
    return apply


def _suffix(mark: str) -> Callable[[str], str]:
    def apply(s: str) -> str:
        return s + mark
    return apply


def _fence(left: str, right: str) -> Callable[[str], str]:
    def apply(s: str) -> str:
        return left + s + right
    return apply


#: the name modifiers (``xdot``, ``xhat``, ``xprime``), longest first
_MODIFIERS: tuple[tuple[str, Callable[[str], str]], ...] = (
    ('mathring', _accent('\N{COMBINING RING ABOVE}')),
    ('ddddot', _accent('\N{COMBINING FOUR DOTS ABOVE}')),
    ('dddot', _accent('\N{COMBINING THREE DOTS ABOVE}')),
    ('check', _accent('\N{COMBINING CARON}')),
    ('breve', _accent('\N{COMBINING BREVE}')),
    ('acute', _accent('\N{COMBINING ACUTE ACCENT}')),
    ('grave', _accent('\N{COMBINING GRAVE ACCENT}')),
    ('tilde', _accent('\N{COMBINING TILDE}')),
    ('prime', _suffix('\N{PRIME}')),
    ('ddot', _accent('\N{COMBINING DIAERESIS}')),
    ('norm', _fence('\N{DOUBLE VERTICAL LINE}', '\N{DOUBLE VERTICAL LINE}')),
    ('dot', _accent('\N{COMBINING DOT ABOVE}')),
    ('hat', _accent('\N{COMBINING CIRCUMFLEX ACCENT}')),
    ('bar', _accent('\N{COMBINING OVERLINE}')),
    ('vec', _accent('\N{COMBINING RIGHT ARROW ABOVE}')),
    ('prm', _suffix('\N{PRIME}')),
    ('avg', _fence('\N{MATHEMATICAL LEFT ANGLE BRACKET}', '\N{MATHEMATICAL RIGHT ANGLE BRACKET}')),
    ('abs', _fence('\N{VERTICAL LINE}', '\N{VERTICAL LINE}')),
    ('mag', _fence('\N{VERTICAL LINE}', '\N{VERTICAL LINE}')),
)


@dataclass(frozen=True)
class Extensible:
    """An object drawn at any length: the repeated piece ``ext``, the end
    pieces ``top`` and ``bottom``, the middle piece ``middle`` (a brace
    has one: its length is then odd) and the form of length one."""

    ext: str
    top: str
    bottom: str
    middle: Optional[str]
    single: str

    def pieces(self, length: int) -> list[str]:
        """The object at the given length, one piece per row (or column)."""
        if length <= 0:
            raise ValueError("Length should be greater than 0")
        if self.middle is not None and length % 2 == 0:
            length += 1
        if length == 1:
            return [self.single]
        half = (length - 2)//2
        odd = (length - 2) - 2*half
        middle = self.ext if self.middle is None else self.middle
        return [self.top] + [self.ext]*half + [middle]*odd + [self.ext]*half + [self.bottom]


def _plain(c: str) -> Extensible:
    return Extensible(c, c, c, None, c)


def _ext(symbol: str) -> str:
    return _u('%s EXTENSION' % symbol)


_LP, _RP, _LB, _RB = 'LEFT PARENTHESIS', 'RIGHT PARENTHESIS', 'LEFT SQUARE BRACKET', 'RIGHT SQUARE BRACKET'
_VERTICAL = _u('BOX DRAWINGS LIGHT VERTICAL')
_FALLING = _u('BOX DRAWINGS LIGHT DIAGONAL UPPER LEFT TO LOWER RIGHT')
_RISING = _u('BOX DRAWINGS LIGHT DIAGONAL UPPER RIGHT TO LOWER LEFT')

_XOBJ_UNICODE: dict[str, Extensible] = {
    '(': Extensible(_ext(_LP), _u(_LP + ' UPPER HOOK'), _u(_LP + ' LOWER HOOK'), None, '('),
    ')': Extensible(_ext(_RP), _u(_RP + ' UPPER HOOK'), _u(_RP + ' LOWER HOOK'), None, ')'),
    '[': Extensible(_ext(_LB), _u(_LB + ' UPPER CORNER'), _u(_LB + ' LOWER CORNER'), None, '['),
    ']': Extensible(_ext(_RB), _u(_RB + ' UPPER CORNER'), _u(_RB + ' LOWER CORNER'), None, ']'),
    '{': Extensible(_ext('CURLY BRACKET'), _u('LEFT CURLY BRACKET UPPER HOOK'), _u('LEFT CURLY BRACKET LOWER HOOK'),
                    _u('LEFT CURLY BRACKET MIDDLE PIECE'), '{'),
    '}': Extensible(_ext('CURLY BRACKET'), _u('RIGHT CURLY BRACKET UPPER HOOK'),
                    _u('RIGHT CURLY BRACKET LOWER HOOK'), _u('RIGHT CURLY BRACKET MIDDLE PIECE'), '}'),
    '|': _plain(_VERTICAL),
    '<': Extensible(_VERTICAL, _RISING, _FALLING, None, '<'),
    '>': Extensible(_VERTICAL, _FALLING, _RISING, None, '>'),
    'lfloor': Extensible(_ext(_LB), _ext(_LB), _u(_LB + ' LOWER CORNER'), None, _u('LEFT FLOOR')),
    'rfloor': Extensible(_ext(_RB), _ext(_RB), _u(_RB + ' LOWER CORNER'), None, _u('RIGHT FLOOR')),
    'lceil': Extensible(_ext(_LB), _u(_LB + ' UPPER CORNER'), _ext(_LB), None, _u('LEFT CEILING')),
    'rceil': Extensible(_ext(_RB), _u(_RB + ' UPPER CORNER'), _ext(_RB), None, _u('RIGHT CEILING')),
    'int': Extensible(_ext('INTEGRAL'), _u('TOP HALF INTEGRAL'), _u('BOTTOM HALF INTEGRAL'), None, _u('INTEGRAL')),
    '-': _plain(_u('BOX DRAWINGS LIGHT HORIZONTAL')),
    '_': _plain(_u('LOW LINE')),
    '/': _plain(_RISING),
    '\\': _plain(_FALLING),
    'UpTack': _plain(_u('BOX DRAWINGS LIGHT DOWN AND HORIZONTAL')),
}

_XOBJ_ASCII: dict[str, Extensible] = {
    '(': Extensible('|', '/', '\\', None, '('),
    ')': Extensible('|', '\\', '/', None, ')'),
    '[': Extensible('[', '[', '[', None, '['),
    ']': Extensible(']', ']', ']', None, ']'),
    '{': Extensible('|', '/', '\\', '<', '{'),
    '}': Extensible('|', '\\', '/', '>', '}'),
    '|': _plain('|'),
    '<': Extensible('|', '/', '\\', None, '<'),
    '>': Extensible('|', '\\', '/', None, '>'),
    'int': Extensible(' | ', '  /', '/  ', None, ' | '),
    '-': _plain('-'),
    '_': _plain('_'),
    '/': _plain('/'),
    '\\': _plain('\\'),
}

#: (ASCII, Unicode) forms of the operators
_XSYM: dict[str, tuple[str, str]] = {
    '==': ('=', '='),
    '<': ('<', '<'),
    '>': ('>', '>'),
    '<=': ('<=', _u('LESS-THAN OR EQUAL TO')),
    '>=': ('>=', _u('GREATER-THAN OR EQUAL TO')),
    '!=': ('!=', _u('NOT EQUAL TO')),
    '*': ('*', _u('DOT OPERATOR')),
    '.': ('*', _u('RING OPERATOR')),
}

_ATOMS: dict[str, str] = {
    'Exp1': _u('SCRIPT SMALL E'),
    'Pi': _u('GREEK SMALL LETTER PI'),
    'Infinity': _u('INFINITY'),
    'NegativeInfinity': '-' + _u('INFINITY'),
    'ImaginaryUnit': _u('DOUBLE-STRUCK ITALIC SMALL I'),
    'EmptySet': _u('EMPTY SET'),
    'Naturals': _u('DOUBLE-STRUCK CAPITAL N'),
    'Naturals0': _u('DOUBLE-STRUCK CAPITAL N') + _u('SUBSCRIPT ZERO'),
    'Integers': _u('DOUBLE-STRUCK CAPITAL Z'),
    'Rationals': _u('DOUBLE-STRUCK CAPITAL Q'),
    'Reals': _u('DOUBLE-STRUCK CAPITAL R'),
    'Complexes': _u('DOUBLE-STRUCK CAPITAL C'),
    'Universe': _u('MATHEMATICAL DOUBLE-STRUCK CAPITAL U'),
    'Differential': _u('DOUBLE-STRUCK ITALIC SMALL D'),
    'Union': _u('UNION'),
    'ElementOf': _u('ELEMENT OF'),
    'SmallElementOf': _u('SMALL ELEMENT OF'),
    'SymmetricDifference': _u('INCREMENT'),
    'Intersection': _u('INTERSECTION'),
    'Ring': _u('RING OPERATOR'),
    'Multiplication': _u('MULTIPLICATION SIGN'),
    'Dots': _u('HORIZONTAL ELLIPSIS'),
    'SuperscriptPlus': _u('SUPERSCRIPT PLUS SIGN'),
    'SuperscriptMinus': _u('SUPERSCRIPT MINUS'),
    'And': _u('LOGICAL AND'),
    'Or': _u('LOGICAL OR'),
    'Not': _u('NOT SIGN'),
    'Nor': _u('NOR'),
    'Nand': _u('NAND'),
    'Xor': _u('XOR'),
    'Equiv': _u('LEFT RIGHT DOUBLE ARROW'),
    'NotEquiv': _u('LEFT RIGHT DOUBLE ARROW WITH STROKE'),
    'Arrow': _u('RIGHTWARDS ARROW'),
    'ArrowFromBar': _u('RIGHTWARDS ARROW FROM BAR'),
    'NotArrow': _u('RIGHTWARDS ARROW WITH STROKE'),
    'Partial': _u('PARTIAL DIFFERENTIAL'),
    'ForAll': _u('FOR ALL'),
    'Exists': _u('THERE EXISTS'),
}

_ROOTS: dict[int, str] = {2: _u('SQUARE ROOT'), 3: _u('CUBE ROOT'), 4: _u('FOURTH ROOT')}


@dataclass(frozen=True)
class Annotated:
    """A stylised drawing of a letter (the ``F`` of a hypergeometric
    function, the ``G`` of a Meijer G-function) with what is needed to
    put the indices around it: its height, the rows above and below it,
    the offset of the baseline of the indices, and the drawing."""

    size: int
    top: int
    bottom: int
    add: int
    image: str


_ANNOTATED_UNICODE: dict[str, Annotated] = {
    'F': Annotated(2, 0, 2, 0, '\N{BOX DRAWINGS LIGHT DOWN AND RIGHT}\N{BOX DRAWINGS LIGHT HORIZONTAL}\n'
                               '\N{BOX DRAWINGS LIGHT VERTICAL AND RIGHT}\N{BOX DRAWINGS LIGHT HORIZONTAL}\n'
                               '\N{BOX DRAWINGS LIGHT UP}'),
    'G': Annotated(3, 0, 3, 1, '\N{BOX DRAWINGS LIGHT ARC DOWN AND RIGHT}\N{BOX DRAWINGS LIGHT HORIZONTAL}'
                               '\N{BOX DRAWINGS LIGHT ARC DOWN AND LEFT}\n'
                               '\N{BOX DRAWINGS LIGHT VERTICAL}\N{BOX DRAWINGS LIGHT RIGHT}'
                               '\N{BOX DRAWINGS LIGHT DOWN AND LEFT}\n'
                               '\N{BOX DRAWINGS LIGHT ARC UP AND RIGHT}\N{BOX DRAWINGS LIGHT HORIZONTAL}'
                               '\N{BOX DRAWINGS LIGHT ARC UP AND LEFT}'),
}
_ANNOTATED_ASCII: dict[str, Annotated] = {
    'F': Annotated(3, 0, 3, 0, ' _\n|_\n|\n'),
    'G': Annotated(3, 0, 3, 1, ' __\n/__\n\\_|'),
}

_COMBINING: dict[int, None] = dict.fromkeys(
    list(range(ord('\N{COMBINING GRAVE ACCENT}'), ord('\N{COMBINING LATIN SMALL LETTER X}')))
    + list(range(ord('\N{COMBINING LEFT HARPOON ABOVE}'), ord('\N{COMBINING ASTERISK ABOVE}'))))


def line_width(line: str) -> int:
    """The number of columns of ``line``: the combining marks take none.

    Examples
    ========

    >>> from sympy_extras.printing.pretty.symbology import line_width
    >>> line_width('x\N{COMBINING DOT ABOVE}')
    1
    """
    return len(line.translate(_COMBINING))


def center_pad(wstring: int, wtarget: int, fillchar: str = ' ') -> tuple[str, str]:
    """The paddings which centre ``wstring`` columns in ``wtarget`` (the
    odd column goes to the right)."""
    if wstring > wtarget:
        raise ValueError('not enough space for string')
    delta = wtarget - wstring
    left = delta//2
    return fillchar*left, fillchar*(delta - left)


def center(string: str, width: int, fillchar: str = ' ') -> str:
    """``string`` centred in ``width`` columns."""
    left, right = center_pad(line_width(string), width, fillchar)
    return left + string + right


_NAME_WITH_DIGITS = re.compile(r'^([^\W\d_]+)(\d+)$', re.U)


# PORTED FROM SYMPY: sympy/printing/conventions.py:split_super_sub
def split_super_sub(text: str) -> tuple[str, list[str], list[str]]:
    """The name, the superscripts and the subscripts of a symbol name.

    Examples
    ========

    >>> from sympy_extras.printing.pretty.symbology import split_super_sub
    >>> split_super_sub('a_x^1'), split_super_sub('var_sub1__sup_sub2'), split_super_sub('x12')
    (('a', ['1'], ['x']), ('var', ['sup'], ['sub1', 'sub2']), ('x', [], ['12']))
    """
    if not text:
        return text, [], []
    pos = 0
    name: Optional[str] = None
    supers: list[str] = []
    subs: list[str] = []
    while pos < len(text):
        start = pos + 1
        if text[pos:pos + 2] == "__":
            start += 1
        pos_hat = text.find("^", start)
        if pos_hat < 0:
            pos_hat = len(text)
        pos_usc = text.find("_", start)
        if pos_usc < 0:
            pos_usc = len(text)
        pos_next = min(pos_hat, pos_usc)
        part = text[pos:pos_next]
        pos = pos_next
        if name is None:
            name = part
        elif part.startswith("^"):
            supers.append(part[1:])
        elif part.startswith("__"):
            supers.append(part[2:])
        elif part.startswith("_"):
            subs.append(part[1:])
        else:
            raise RuntimeError("This should never happen.")
    if name is None:
        name = text
    # a name ending in digits has them as a subscript: x12 is x with the subscript 12
    m = _NAME_WITH_DIGITS.match(name)
    if m:
        name = m.group(1)
        subs.insert(0, m.group(2))
    return name, supers, subs


def _translate(s: str) -> str:
    letter = greek_unicode.get(s)
    if letter is not None:
        return letter
    for key, modifier in _MODIFIERS:
        if s.lower().endswith(key) and len(s) > len(key):
            return modifier(_translate(s[:-len(key)]))
    return s


def _scripts(parts: list[str], mapping: dict[str, str]) -> Optional[list[str]]:
    """The parts in the script characters of ``mapping``, ``None`` when
    one of them has no such form."""
    result: list[str] = []
    for part in parts:
        whole = mapping.get(part)
        if whole is None:
            if not all(c in mapping for c in part):
                return None
            whole = ''.join(mapping[c] for c in part)
        result.append(whole)
    return result


@dataclass(frozen=True)
class Symbology:
    """The characters to draw with: Unicode or ASCII.

    Examples
    ========

    >>> from sympy_extras.printing.pretty.symbology import Symbology
    >>> ascii, uni = Symbology(False), Symbology(True)
    >>> ascii.xsym('<='), ascii.symbol('alpha_1'), ascii.hobj('-', 3)
    ('<=', 'alpha_1', '---')
    >>> uni.xsym('<=') == '\N{LESS-THAN OR EQUAL TO}', uni.symbol('alpha_1') == '\N{GREEK SMALL LETTER ALPHA}\N{SUBSCRIPT ONE}'
    (True, True)
    """

    unicode: bool
    #: the letter of the imaginary unit in Unicode (``'i'`` or ``'j'``)
    imaginary_unit: str = 'i'

    def xobj(self, symbol: str, length: int) -> list[str]:
        """The pieces of the extensible object ``symbol`` at ``length``."""
        table = _XOBJ_UNICODE if self.unicode else _XOBJ_ASCII
        return table[symbol].pieces(length)

    def vobj(self, symbol: str, height: int) -> str:
        """The object drawn vertically, its rows separated by newlines."""
        return '\n'.join(self.xobj(symbol, height))

    def hobj(self, symbol: str, width: int) -> str:
        """The object drawn horizontally."""
        return ''.join(self.xobj(symbol, width))

    def xsym(self, symbol: str) -> str:
        """The operator ``symbol`` (``'=='``, ``'<='``, ``'*'``, ...)."""
        return _XSYM[symbol][1 if self.unicode else 0]

    def atom(self, name: str, default: str) -> str:
        """The Unicode character of the atom ``name``, ``default`` in ASCII."""
        if not self.unicode:
            return default
        if name == 'ImaginaryUnit' and self.imaginary_unit == 'j':
            return '\N{DOUBLE-STRUCK ITALIC SMALL J}'
        return _ATOMS[name]

    def greek(self, name: str, default: str) -> str:
        """The Greek letter ``name``, ``default`` in ASCII."""
        return greek_unicode[name] if self.unicode else default

    def root(self, index: int) -> Optional[str]:
        """The one-character root sign of the given index, if there is one."""
        return _ROOTS.get(index) if self.unicode else None

    def annotated(self, letter: str) -> Annotated:
        """The stylised ``F`` or ``G``."""
        return (_ANNOTATED_UNICODE if self.unicode else _ANNOTATED_ASCII)[letter]

    def subscriptable(self, text: str) -> bool:
        """Whether every character of ``text`` has a Unicode subscript."""
        return self.unicode and text != '' and all(c in sub for c in text)

    def symbol(self, name: str) -> str:
        """The printed form of a symbol name: Greek letters by name,
        modifiers (``xdot``), sub- and superscripts in the Unicode script
        characters when all of them have one, the name itself in ASCII."""
        if not self.unicode:
            return name
        base, sups, subs = split_super_sub(name)
        if '' in sups or '' in subs or base == '':
            # x_ is not x: SymPy drops the empty mark, so that x_ + x prints as x + x
            return name
        base = _translate(base)
        pretty_sups = _scripts(sups, sup)
        pretty_subs = _scripts(subs, sub) if pretty_sups is not None else None
        if pretty_sups is None or pretty_subs is None:
            if subs:
                base += '_' + '_'.join(_translate(s) for s in subs)
            if sups:
                base += '__' + '__'.join(_translate(s) for s in sups)
            return base
        return base + ' '.join(pretty_sups) + ' '.join(pretty_subs)
