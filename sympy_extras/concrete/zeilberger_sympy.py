r"""The SymPy face of Zeilberger's algorithm
(:mod:`sympy_extras.concrete.zeilberger`).

:mod:`~sympy_extras.concrete.zeilberger` is written on the core and takes
and returns core expressions. The modules still written on SymPy
(:mod:`sympy_extras.integrals.series`) use this module instead:
:func:`zeilberger_sum`, :func:`wz_certificate` and :func:`wz_prove` take
and return SymPy expressions, converted at the boundary
(:mod:`sympy_extras.interop`) around the native functions.

Examples
========

>>> from sympy import binomial, symbols
>>> from sympy_extras.concrete.zeilberger_sympy import zeilberger_sum, wz_prove
>>> n, k = symbols('n k', integer=True)
>>> zeilberger_sum(binomial(n, k)**2, (k, 0, n))
binomial(2*n, n)
>>> wz_prove(binomial(n, k)**2, binomial(2*n, n), n, k)
True
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify

from sympy_extras._typing import as_expr
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.interop import expr_from_sympy, to_sympy

from . import zeilberger as core_zeilberger

__all__ = ['zeilberger_sum', 'wz_certificate', 'wz_prove']


def _core(e: Union[Basic, int]) -> CoreExpr:
    return expr_from_sympy(sympify(e))


def _core_symbol(s: Basic) -> CoreSymbol:
    converted = _core(s)
    if not isinstance(converted, CoreSymbol):
        raise TypeError("%s is not a symbol" % (s,))
    return converted


def zeilberger_sum(term: Union[Expr, int], limits: Sequence[Union[Symbol, Expr, int]], n: Optional[Symbol] = None,
                   max_order: int = 4, special_values: bool = True) -> Optional[Basic]:
    """:func:`sympy_extras.concrete.zeilberger.zeilberger_sum` on SymPy
    expressions: a closed form (an ``Expr``), the recurrence (an ``Eq``)
    or ``None``.

    >>> from sympy import symbols
    >>> from sympy_extras.concrete.zeilberger_sympy import zeilberger_sum
    >>> k, n, y = symbols('k n y')
    >>> zeilberger_sum(k*y**k, (k, 0, n), n)
    Piecewise((n*(n + 1)/2, Eq(y, 1)), ((-n*y**(n + 1) + n*y**(n + 2) + y - y**(n + 1))/(y - 1)**2, True))
    """
    if len(limits) != 3:
        raise ValueError("the limits of a sum are (k, lo, hi), not %s" % (limits,))
    core_limits = [_core(e) for e in limits]
    found = core_zeilberger.zeilberger_sum(_core(term), core_limits, None if n is None else _core_symbol(n),
                                           max_order, special_values)
    return None if found is None else to_sympy(found)


def wz_certificate(term: Union[Expr, int], closed_form: Union[Expr, int], n: Symbol, k: Symbol) -> Optional[Expr]:
    """:func:`sympy_extras.concrete.zeilberger.wz_certificate` on SymPy
    expressions.

    >>> from sympy import binomial, symbols
    >>> from sympy_extras.concrete.zeilberger_sympy import wz_certificate
    >>> n, k = symbols('n k', integer=True)
    >>> wz_certificate(binomial(n, k), 2**n, n, k)
    k/(2*(k - n - 1))
    """
    found = core_zeilberger.wz_certificate(_core(term), _core(closed_form), _core_symbol(n), _core_symbol(k))
    return None if found is None else as_expr(to_sympy(found))


def wz_prove(term: Union[Expr, int], closed_form: Union[Expr, int], n: Symbol, k: Symbol,
             n0: int = 0) -> Optional[bool]:
    """:func:`sympy_extras.concrete.zeilberger.wz_prove` on SymPy
    expressions."""
    return core_zeilberger.wz_prove(_core(term), _core(closed_form), _core_symbol(n), _core_symbol(k), n0)
