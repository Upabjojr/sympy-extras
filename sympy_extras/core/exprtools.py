"""Tools on expressions which belong to SymPy's ``exprtools``:
``factor_terms``, ``gcd_terms`` and ``factor_nc``, applied through the
conversion to SymPy."""
from __future__ import annotations

import sympy

from .expr import Expr, ExprLike, to_expr

__all__ = ['factor_terms', 'gcd_terms', 'factor_nc']


def factor_terms(expr: ExprLike, radical: bool = False, clear: bool = False, fraction: bool = False,
                 sign: bool = True) -> Expr:
    """The common factors of the terms taken out (SymPy's ``factor_terms``).

    Examples
    ========

    >>> from sympy_extras.core import symbols, factor_terms
    >>> x, y = symbols('x y')
    >>> factor_terms(2*x + 4*y), factor_terms(x*y + x)
    (2*(x + 2*y), x*(y + 1))
    """
    from .interop import expr_through_sympy
    return expr_through_sympy(to_expr(expr), lambda s: sympy.factor_terms(s, radical=radical, clear=clear,
                                                                           fraction=fraction, sign=sign))


def gcd_terms(terms: ExprLike, isprimitive: bool = False, clear: bool = True, fraction: bool = True) -> Expr:
    """The terms written over a common denominator with their content
    taken out (SymPy's ``gcd_terms``).

    Examples
    ========

    >>> from sympy_extras.core import symbols, gcd_terms
    >>> x, y = symbols('x y')
    >>> gcd_terms(x/2 + y/4)
    (2*x + y)/4
    """
    from .interop import expr_through_sympy
    return expr_through_sympy(to_expr(terms), lambda s: sympy.gcd_terms(s, isprimitive=isprimitive, clear=clear,
                                                                         fraction=fraction))


def factor_nc(expr: ExprLike) -> Expr:
    """The common factors of a noncommutative sum taken out (SymPy's ``factor_nc``).

    Examples
    ========

    >>> from sympy_extras.core import Symbol, factor_nc
    >>> x, A, B = Symbol('x'), Symbol('A', commutative=False), Symbol('B', commutative=False)
    >>> factor_nc(x*A + x*B)
    x*(A + B)
    """
    from .interop import expr_through_sympy
    return expr_through_sympy(to_expr(expr), sympy.factor_nc)
