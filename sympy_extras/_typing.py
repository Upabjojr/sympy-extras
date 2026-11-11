"""Type aliases shared by the package which need no SymPy.

The aliases here name the structures the package passes around, so that
signatures say what they mean (and can be translated to statically typed
languages one day):

* ``Dup`` and ``Dmp`` are dense univariate and multivariate polynomials of
  :mod:`sympy.polys` (nested lists of domain elements);
* ``Monomial`` is an exponent vector;
* ``Sign`` is ``-1``, ``0`` or ``1``;
* ``Truth`` is a three valued truth value (``None`` for undecided);
* ``DomainElement`` is an element of a SymPy coefficient domain (``ZZ``,
  ``QQ``, an algebraic field, a fraction field);
* ``OrderKey`` is a monomial order given as a key function;
* ``Weights`` are the weights of a weight order;
* ``QuantifierKind`` is ``'exists'`` or ``'forall'``.

The module imports no SymPy. The aliases written with SymPy's classes
(``ExprLike``, ``OrderSpec``, ``QuantifierPrefix``, ``QuantifierSpec``) and
the helpers which narrow SymPy objects at the boundary with SymPy
(``as_expr``, ``as_boolean``, ``as_symbol``, ``as_set``, ``free_symbols``,
``sorted_symbols``) are in :mod:`sympy_extras._typing_sympy`, which
re-exports these aliases too; the code written on the core narrows with
the helpers of the core (:func:`sympy_extras.core.expr.as_expr`,
:func:`sympy_extras.core.logic.to_boolean`).
"""
from __future__ import annotations

from fractions import Fraction
from typing import Any, Callable, Optional

__all__ = ['Dup', 'Dmp', 'Monomial', 'Sign', 'Truth', 'DomainElement', 'OrderKey', 'Weights', 'QuantifierKind']

Dup = list[Any]  # list[DomainElement]
Dmp = list[Any]  # list[Dmp | DomainElement]
Monomial = tuple[int, ...]
Sign = int
Truth = Optional[bool]
DomainElement = Any
OrderKey = Callable[[Monomial], Any]
Weights = tuple[Fraction, ...]
QuantifierKind = str  # 'exists' or 'forall'
