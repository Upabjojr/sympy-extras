"""Type aliases shared by the package.

SymPy 1.14 ships no ``py.typed`` marker and few annotations, so its objects
are seen by type checkers mostly as ``Any``. The aliases here name the
structures the package passes around, so that signatures say what they
mean (and can be translated to statically typed languages one day):

* ``Dup`` and ``Dmp`` are dense univariate and multivariate polynomials of
  :mod:`sympy.polys` (nested lists of domain elements);
* ``Monomial`` is an exponent vector;
* ``Sign`` is ``-1``, ``0`` or ``1``;
* ``Truth`` is a three valued truth value (``None`` for undecided);
* ``DomainElement`` is an element of a SymPy coefficient domain (``ZZ``,
  ``QQ``, an algebraic field, a fraction field);
* ``OrderSpec`` is what SymPy accepts as a monomial order: a name, a
  :class:`~sympy.polys.orderings.MonomialOrder` or a key function;
* ``QuantifierPrefix`` is the quantifier prefix of the CAD functions,
  ``[('exists', x), ('forall', y)]``, and ``QuantifierSpec`` its input form
  in which the second component may also be a list of symbols.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Any, Callable, Iterable, Optional, Sequence, Union

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.logic.boolalg import Boolean, true, false
from sympy.polys.orderings import MonomialOrder
from sympy.sets.sets import Set

__all__ = ['Dup', 'Dmp', 'Monomial', 'Sign', 'Truth', 'DomainElement',
    'OrderSpec', 'OrderKey', 'Weights', 'QuantifierPrefix', 'QuantifierSpec',
    'QuantifierKind', 'as_expr', 'as_boolean', 'as_symbol', 'as_set',
    'free_symbols', 'sorted_symbols', 'ExprLike']

#: an expression or a Python number SymPy converts to one
ExprLike = Union[Expr, int]
Dup = list[Any]  # list[DomainElement]
Dmp = list[Any]  # list[Dmp | DomainElement]
Monomial = tuple[int, ...]
Sign = int
Truth = Optional[bool]
DomainElement = Any
OrderKey = Callable[[Monomial], Any]
OrderSpec = Union[str, MonomialOrder]
Weights = tuple[Fraction, ...]
QuantifierKind = str  # 'exists' or 'forall'
QuantifierPrefix = list[tuple[str, Symbol]]
QuantifierSpec = Sequence[tuple[str, Union[Symbol, Sequence[Symbol]]]]


# Narrowing helpers at the boundary with SymPy, whose functions return the
# base class Basic: they check the class at run time and give the type
# checker the precise one.

def as_expr(x: object) -> Expr:
    """``x`` sympified, which must be an :class:`~sympy.core.expr.Expr`."""
    e = sympify(x)
    if not isinstance(e, Expr):
        raise TypeError("an expression is expected, got %s" % (e,))
    return e


def as_boolean(x: object) -> Boolean:
    """``x`` sympified, which must be a :class:`~sympy.logic.boolalg.Boolean`
    (Python ``True``/``False`` become ``S.true``/``S.false``)."""
    if x is True:
        return true
    if x is False:
        return false
    b = sympify(x)
    if not isinstance(b, Boolean):
        raise TypeError("a Boolean is expected, got %s" % (b,))
    return b


def as_symbol(x: object) -> Symbol:
    """``x`` sympified, which must be a :class:`~sympy.core.symbol.Symbol`."""
    s = sympify(x)
    if not isinstance(s, Symbol):
        raise TypeError("a symbol is expected, got %s" % (s,))
    return s


def as_set(x: object) -> Set:
    """``x`` sympified, which must be a :class:`~sympy.sets.sets.Set`."""
    s = sympify(x)
    if not isinstance(s, Set):
        raise TypeError("a SymPy set is expected, got %s" % (s,))
    return s


def free_symbols(x: Basic) -> set[Symbol]:
    """The free symbols of ``x`` which are plain symbols."""
    return {s for s in x.free_symbols if isinstance(s, Symbol)}


def sorted_symbols(symbols: Iterable[Symbol]) -> list[Symbol]:
    """Symbols sorted by name."""
    return sorted(symbols, key=lambda s: s.name)
