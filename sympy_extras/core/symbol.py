"""Symbols.

``Symbol(name, **assumptions)`` is a leaf identified by its name and its
assumptions (``Symbol('x', positive=True)`` is not ``Symbol('x')``), as in
SymPy; ``Dummy`` is a symbol equal only to itself; ``symbols`` parses
``'x y'``, ``'x:3'``, ``'a1:3'``.

The assumptions are stored as given and completed by the implications of
:mod:`sympy_extras.core.assumptions` (``positive`` implies ``real``,
``nonzero``, ...) when they are queried.
"""
from __future__ import annotations

import itertools
import re
from typing import TYPE_CHECKING, Sequence, Union

from .basic import Basic
from .expr import AtomicExpr
from .sorting import ClassKey

if TYPE_CHECKING:
    from .printing import StrPrinter

__all__ = ['Symbol', 'Dummy', 'symbols', 'Assumptions']

#: the assumptions of a symbol, sorted by name (hashable)
Assumptions = tuple[tuple[str, bool], ...]


class Symbol(AtomicExpr):
    """A symbol with a name and optional assumptions.

    Examples
    ========

    >>> from sympy_extras.core import Symbol
    >>> x = Symbol('x')
    >>> x.name
    x
    >>> p = Symbol('p', positive=True)
    >>> p.is_positive, p.is_real, p.is_zero, x.is_positive
    (True, True, False, None)
    >>> Symbol('x') == Symbol('x'), Symbol('x') == p
    (True, False)
    >>> Symbol('p', positive=True, real=True) == p
    True
    """

    __slots__ = ('name', '_given', '_assumptions')

    is_Symbol = True

    def __init__(self, name: str, **assumptions: bool) -> None:
        from .assumptions import PREDICATES, closure
        if not isinstance(name, str):
            raise TypeError("the name of a symbol is a string, got %r" % (name,))
        for k in assumptions:
            if k not in PREDICATES:
                raise ValueError("unknown assumption %r" % (k,))
        self.name: str = name
        self._given: Assumptions = tuple(sorted((k, bool(v)) for k, v in assumptions.items()))
        self._assumptions: Assumptions = tuple(sorted(closure(dict(self._given)).items()))
        Basic.__init__(self)

    def _hashable_content(self) -> tuple[Basic, ...]:
        return ()

    def __hash__(self) -> int:
        return hash((self.name, self._assumptions))

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other) or not isinstance(other, Symbol):
            if isinstance(other, Basic):
                return False
            return NotImplemented
        return self.name == other.name and self._assumptions == other._assumptions

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return NotImplemented
        return not result

    @classmethod
    def class_key(cls) -> ClassKey:
        return (2, 0, 'Symbol')

    @property
    def assumptions0(self) -> dict[str, bool]:
        """The assumptions of the symbol with their consequences; two
        symbols with the same name and the same consequences are equal,
        as in SymPy."""
        return dict(self._assumptions)

    @property
    def given_assumptions(self) -> dict[str, bool]:
        """The assumptions as given to the constructor."""
        return dict(self._given)

    @property
    def free_symbols(self) -> set[Symbol]:
        return {self}

    def __str__(self) -> str:
        return self.name

    def _sympystr(self, printer: StrPrinter) -> str:
        return self.name


_dummy_counter = itertools.count()


class Dummy(Symbol):
    """A symbol equal only to itself.

    Examples
    ========

    >>> from sympy_extras.core import Dummy
    >>> Dummy('x') == Dummy('x')
    False
    >>> d = Dummy('x')
    >>> d == d, d
    (True, _x)
    """

    __slots__ = ('dummy_index',)

    is_Dummy = True

    def __init__(self, name: str = 'Dummy', **assumptions: bool) -> None:
        self.dummy_index: int = next(_dummy_counter)
        Symbol.__init__(self, name, **assumptions)

    def __hash__(self) -> int:
        return hash((self.name, self.dummy_index, self._assumptions))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Dummy):
            if isinstance(other, Basic):
                return False
            return NotImplemented
        return self.dummy_index == other.dummy_index and self.name == other.name

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return NotImplemented
        return not result

    @classmethod
    def class_key(cls) -> ClassKey:
        return (2, 1, 'Dummy')

    def __str__(self) -> str:
        return '_' + self.name

    def _sympystr(self, printer: StrPrinter) -> str:
        return '_' + self.name


_RANGE = re.compile(r'^(?P<name>[^:]*?)(?P<start>\d*):(?P<end>\d+)$')


def _expand_names(text: str) -> list[str]:
    names: list[str] = []
    for part in re.split(r'[\s,]+', text.strip()):
        if not part:
            continue
        m = _RANGE.match(part)
        if m is None:
            names.append(part)
            continue
        start = int(m.group('start')) if m.group('start') else 0
        end = int(m.group('end'))
        names.extend('%s%d' % (m.group('name'), i) for i in range(start, end))
    return names


def symbols(names: Union[str, Sequence[str]], **assumptions: bool) -> Union[Symbol, tuple[Symbol, ...]]:
    """Symbols from their names.

    Several names separated by spaces or commas, or a range ``'x:3'``,
    give a tuple; one name gives one symbol (SymPy's convention; a typed
    caller which wants one symbol uses ``Symbol``).

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> symbols('x y')
    (x, y)
    >>> symbols('x:3')
    (x0, x1, x2)
    >>> symbols('a1:3')
    (a1, a2)
    >>> symbols('x')
    x
    """
    if isinstance(names, str):
        expanded = _expand_names(names)
        single = len(expanded) == 1 and not re.search(r'[\s,:]', names.strip())
    else:
        expanded = list(names)
        single = False
    result = tuple(Symbol(name, **assumptions) for name in expanded)
    if single:
        return result[0]
    return result
