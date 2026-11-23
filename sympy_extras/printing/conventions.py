"""The conventions of names which the printers share: the scripts of a
name, the Greek letters, the modifiers at the end of a name.

``split_super_sub`` is a typed port of the function of that name in
SymPy's ``sympy/printing/conventions.py``. The LaTeX printer
(:mod:`sympy_extras.printing._latex_names`) and the pretty printer
(:mod:`sympy_extras.printing.pretty.symbology`) read a name in the same
way and differ in what they write for its parts only: each one gives
:func:`translate_name` its own table of letters and of modifiers.

References
==========

.. [1] SymPy Development Team, *SymPy documentation: Printing*,
       https://docs.sympy.org/latest/modules/printing.html
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/printing/conventions.py (split_super_sub) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: strict typing; translate_name (the walk over the modifiers which
# SymPy has twice, in latex.py and in pretty_symbology.py) is original code.
# ---------------------------------------------------------------------
from __future__ import annotations

import re
from typing import Callable, Optional, Sequence

from sympy_extras.core.alphabets import greeks

__all__ = ['GREEK_NAMES', 'GREEK_LETTERS', 'split_super_sub', 'translate_name']

#: the names of the Greek letters, in lower case and in the order of the
#: alphabet (SymPy's spelling ``lamda`` is an alias which each printer
#: adds to its table of letters)
GREEK_NAMES: tuple[str, ...] = greeks

#: the same names as a set
GREEK_LETTERS: frozenset[str] = frozenset(GREEK_NAMES)

_NAME_WITH_DIGITS = re.compile(r'^([^\W\d_]+)(\d+)$', re.UNICODE)


# PORTED FROM SYMPY: sympy/printing/conventions.py:split_super_sub
def split_super_sub(text: str) -> tuple[str, list[str], list[str]]:
    """The name, the superscripts (after ``^`` or ``__``) and the
    subscripts (after ``_``, or the digits at the end of the name).

    Examples
    ========

    >>> from sympy_extras.printing.conventions import split_super_sub
    >>> split_super_sub('a_x^1'), split_super_sub('var_sub1__sup_sub2'), split_super_sub('x12')
    (('a', ['1'], ['x']), ('var', ['sup'], ['sub1', 'sub2']), ('x', [], ['12']))
    """
    if not text:
        return (text, [], [])
    pos = 0
    name = ''
    first = True
    supers: list[str] = []
    subs: list[str] = []
    while pos < len(text):
        start = pos + 1
        if text[pos:pos + 2] == '__':
            start += 1
        pos_hat = text.find('^', start)
        if pos_hat < 0:
            pos_hat = len(text)
        pos_usc = text.find('_', start)
        if pos_usc < 0:
            pos_usc = len(text)
        pos_next = min(pos_hat, pos_usc)
        part = text[pos:pos_next]
        pos = pos_next
        if first:
            name = part
            first = False
        elif part.startswith('^'):
            supers.append(part[1:])
        elif part.startswith('__'):
            supers.append(part[2:])
        else:
            subs.append(part[1:])
    # a name ending in digits has them as a subscript: x12 is x with the subscript 12
    m = _NAME_WITH_DIGITS.match(name)
    if m is not None:
        name = m.group(1)
        subs.insert(0, m.group(2))
    return (name, supers, subs)


def translate_name(name: str, letter: Callable[[str], Optional[str]],
                   modifiers: Sequence[tuple[str, Callable[[str], str]]]) -> str:
    """A name without scripts in the notation of a printer: ``letter``
    gives the form of a whole name which has one (a Greek letter),
    ``modifiers`` the forms of the modifiers at the end of a name
    (``xdot``, ``xhat``), tried in their order and applied to the
    translation of what precedes them; any other name is itself.

    Examples
    ========

    >>> from sympy_extras.printing.conventions import GREEK_LETTERS, translate_name
    >>> letter = lambda s: '<%s>' % s if s in GREEK_LETTERS else None
    >>> modifiers = [('dot', lambda s: s + "'")]
    >>> [translate_name(s, letter, modifiers) for s in ('alpha', 'xdot', 'alphadotdot', 'dot', 'y')]
    ['<alpha>', "x'", "<alpha>''", 'dot', 'y']
    """
    whole = letter(name)
    if whole is not None:
        return whole
    lowered = name.lower()
    for key, modifier in modifiers:
        if lowered.endswith(key) and len(name) > len(key):
            return modifier(translate_name(name[:-len(key)], letter, modifiers))
    return name
