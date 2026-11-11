"""Type aliases and narrowing helpers for the code written on SymPy.

SymPy 1.14 ships no ``py.typed`` marker and few annotations, so its objects
are seen by type checkers mostly as ``Any``. This module names the
structures written with SymPy's classes and narrows SymPy's results at the
boundary:

* ``ExprLike`` is an expression or a Python integer SymPy converts to one;
* ``OrderSpec`` is what SymPy accepts as a monomial order: a name or a
  :class:`~sympy.polys.orderings.MonomialOrder`;
* ``QuantifierPrefix`` is the quantifier prefix of the CAD functions,
  ``[('exists', x), ('forall', y)]``, and ``QuantifierSpec`` its input form
  in which the second component may also be a list of symbols;
* :func:`as_expr`, :func:`as_boolean`, :func:`as_symbol` and
  :func:`as_set` sympify and check the class at run time;
  :func:`free_symbols` and :func:`sorted_symbols` give the plain symbols.

The aliases which need no SymPy (``Dup``, ``Dmp``, ``Monomial``, ``Sign``,
``Truth``, ``DomainElement``, ``OrderKey``, ``Weights``,
``QuantifierKind``) live in :mod:`sympy_extras._typing` and are
re-exported here, so that the code written on SymPy imports from one
module.
"""
from __future__ import annotations

from typing import Iterable, Sequence, Union

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.logic.boolalg import Boolean, true, false
from sympy.polys.orderings import MonomialOrder
from sympy.sets.sets import Set

from sympy_extras._typing import (Dmp, DomainElement, Dup, Monomial, OrderKey, QuantifierKind, Sign, Truth,
                                  Weights)

__all__ = ['Dup', 'Dmp', 'Monomial', 'Sign', 'Truth', 'DomainElement',
    'OrderSpec', 'OrderKey', 'Weights', 'QuantifierPrefix', 'QuantifierSpec',
    'QuantifierKind', 'as_expr', 'as_boolean', 'as_symbol', 'as_set',
    'free_symbols', 'sorted_symbols', 'ExprLike']

#: an expression or a Python number SymPy converts to one
ExprLike = Union[Expr, int]
OrderSpec = Union[str, MonomialOrder]
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
