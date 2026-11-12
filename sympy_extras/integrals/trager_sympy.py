"""The SymPy face of Trager's algorithm for algebraic integrands
(:mod:`sympy_extras.integrals.trager`), for the integrators which are
still written on SymPy (:mod:`.indefinite`).

The functions take and return SymPy objects, converted by
:mod:`sympy_extras.interop` around the native ones; nothing of the
algorithm stays on SymPy. What the face adds: the integrands with
several or nested radicals go to the general curve through
:mod:`.curves_sympy`, which writes the generator of a field of residues
without radicals and without real generator as SymPy's ``CRootOf`` (the
native module has no expression for it and leaves such integrals
undone).

Examples
========

>>> from sympy import symbols, sqrt, Rational
>>> from sympy_extras.integrals.trager_sympy import trager_antiderivative, is_nonelementary_algebraic
>>> x = symbols('x')
>>> trager_antiderivative(x/sqrt(x**4 + 1), x)
log(x**2 + sqrt(x**4 + 1))/2
>>> trager_antiderivative(1/(x*(x**4 + 1)**Rational(1, 4)), x)
log((x**4 + 1)**(1/4) - 1)/4 - log((x**4 + 1)**(1/4) + 1)/4 + atan((x**4 + 1)**(1/4))/2
>>> is_nonelementary_algebraic(1/sqrt(x**3 + 1), x)
True
"""
from __future__ import annotations

from typing import Optional

from sympy.core.expr import Expr
from sympy.core.symbol import Symbol

from sympy_extras._numeric_sympy import core_expr, sympy_expr
from sympy_extras._typing_sympy import ExprLike, as_expr, as_symbol
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.interop import from_sympy

from . import trager as native

__all__ = ['trager_antiderivative', 'trager_reduce', 'hermite_reduce_algebraic', 'is_nonelementary_algebraic']


def _core_symbol(s: Symbol) -> CoreSymbol:
    converted = from_sympy(as_symbol(s))
    if not isinstance(converted, CoreSymbol):
        raise TypeError("%s is not a symbol" % (s,))
    return converted


def _curve_reduce(f: CoreExpr, x: CoreSymbol) -> Optional[tuple[CoreExpr, CoreExpr]]:
    """The reduction on the general curve by :mod:`.curves_sympy` (the
    residues of a field without radicals as ``CRootOf``)."""
    from .curves_sympy import curve_reduce
    found = curve_reduce(sympy_expr(f), as_symbol(sympy_expr(x)))
    if found is None:
        return None
    return core_expr(found[0]), core_expr(found[1])


def _pair(found: Optional[tuple[CoreExpr, CoreExpr]]) -> Optional[tuple[Expr, Expr]]:
    return None if found is None else (sympy_expr(found[0]), sympy_expr(found[1]))


def trager_reduce(f: ExprLike, x: Symbol) -> Optional[tuple[Expr, Expr]]:
    """:func:`sympy_extras.integrals.trager.trager_reduce` on SymPy
    objects: ``(F, h)`` with ``f == F' + h``, ``F`` the elementary part
    and ``h`` the integrand left, or ``None``.

    >>> from sympy import symbols, sqrt
    >>> from sympy_extras.integrals.trager_sympy import trager_reduce
    >>> x = symbols('x')
    >>> trager_reduce(1/sqrt(x**3 + 1), x)
    (0, 1/sqrt(x**3 + 1))
    >>> trager_reduce((2*x**2 + 1)/((x**2 + 1)*sqrt(x**4 + x**2 + 1)), x)
    (-I*log((-I*x + sqrt(x**4 + x**2 + 1))/(x**2 + 1))/2, 3/(2*sqrt(x**4 + x**2 + 1)))
    """
    return _pair(native._trager_reduce(core_expr(as_expr(f)), _core_symbol(x), _curve_reduce))


def trager_antiderivative(f: ExprLike, x: Symbol) -> Optional[Expr]:
    """:func:`sympy_extras.integrals.trager.trager_antiderivative` on
    SymPy objects: an antiderivative of the algebraic function ``f``, or
    ``None``.

    >>> from sympy import symbols, sqrt
    >>> from sympy_extras.integrals.trager_sympy import trager_antiderivative
    >>> x = symbols('x')
    >>> trager_antiderivative(1/(x*sqrt(x**2 + 1)), x)
    -log((sqrt(x**2 + 1) + 1)/x)
    >>> trager_antiderivative(1/sqrt(x**3 + 1), x) is None
    True
    """
    found = native._trager_antiderivative(core_expr(as_expr(f)), _core_symbol(x), _curve_reduce)
    return None if found is None else sympy_expr(found)


def hermite_reduce_algebraic(f: ExprLike, x: Symbol) -> Optional[tuple[Expr, Expr]]:
    """:func:`sympy_extras.integrals.trager.hermite_reduce_algebraic` on
    SymPy objects: ``(g, h)`` with ``f == g' + h`` and ``h`` with simple
    poles only, or ``None``.

    >>> from sympy import symbols, sqrt
    >>> from sympy_extras.integrals.trager_sympy import hermite_reduce_algebraic
    >>> x = symbols('x')
    >>> hermite_reduce_algebraic(x**2/sqrt(x**2 + 1), x)
    (x*sqrt(x**2 + 1)/2, -1/(2*sqrt(x**2 + 1)))
    """
    return _pair(native.hermite_reduce_algebraic(core_expr(as_expr(f)), _core_symbol(x)))


def is_nonelementary_algebraic(f: ExprLike, x: Symbol) -> Optional[bool]:
    """:func:`sympy_extras.integrals.trager.is_nonelementary_algebraic` on
    SymPy objects.

    >>> from sympy import symbols, sqrt
    >>> from sympy_extras.integrals.trager_sympy import is_nonelementary_algebraic
    >>> x = symbols('x')
    >>> is_nonelementary_algebraic(x/sqrt(x**3 + 1), x), is_nonelementary_algebraic(x/sqrt(x**4 + 1), x)
    (True, False)
    """
    return native._is_nonelementary_algebraic(core_expr(as_expr(f)), _core_symbol(x), _curve_reduce)
