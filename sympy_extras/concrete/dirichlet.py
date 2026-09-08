"""Dirichlet series `\\sum_{n \\ge 1} f(n)\\, n^{-s}` summed by pattern
matching on the coefficients `f(n)`.

The generating Dirichlet series of the classical arithmetic functions are
products and quotients of the Riemann zeta function, and the series with
periodic signs or logarithmic factors are transforms of it:

===================================  ==================================  =============
`f(n)`                               `\\sum f(n) n^{-s}`                  converges for
===================================  ==================================  =============
`1`                                  `\\zeta(s)`                          `\\Re s > 1`
`(-1)^{n+1}`                         `(1 - 2^{1-s})\\,\\zeta(s)`           `\\Re s > 0`
`\\mu(n)`                             `1/\\zeta(s)`                        `\\Re s > 1`
`\\mu(n)^2`                           `\\zeta(s)/\\zeta(2s)`                `\\Re s > 1`
`\\varphi(n)`                         `\\zeta(s-1)/\\zeta(s)`               `\\Re s > 2`
`\\sigma_k(n)`                        `\\zeta(s)\\,\\zeta(s-k)`              `\\Re s > \\max(1, 1 + \\Re k)`
`\\log^k n`                           `(-1)^k \\zeta^{(k)}(s)`             `\\Re s > 1`
===================================  ==================================  =============

(The derivatives of `\\zeta` are returned for a symbolic `s` only.)

The summand is written as `c\\, f(n)\\, n^{e}` with `e` free of `n`
(`s = -e`); a sum starting above `1` gets the first terms subtracted.
SymPy's ``summation`` knows `\\zeta(s)` and the Hurwitz zeta function for
`\\sum (a n + b)^{-s}` but none of the arithmetic functions.

References
==========

.. [Apostol] T. M. Apostol, Introduction to Analytic Number Theory,
   Springer (1976), chapter 11 (Dirichlet series and Euler products).
.. [Hardy] G. H. Hardy, E. M. Wright, An Introduction to the Theory of
   Numbers, Oxford (1979), §17.
"""
from __future__ import annotations

from typing import Optional, Union

from sympy.concrete.summations import Sum
from sympy.core.expr import Expr
from sympy.core.function import Derivative
from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.numbers import mobius, totient, divisor_sigma
from sympy.functions.elementary.complexes import re
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.miscellaneous import Max
from sympy.functions.special.zeta_functions import zeta
from sympy.logic.boolalg import Boolean

from sympy_extras._typing import as_boolean, as_expr

__all__ = ['dirichlet_series', 'DirichletSum']

#: the closed form of a Dirichlet series and the condition of convergence
DirichletSum = tuple[Expr, Boolean]


def _split(term: Expr, n: Symbol) -> Optional[tuple[Expr, Expr, Expr]]:
    """``(c, f, s)`` with ``term == c * f(n) * n**(-s)``, ``c`` free of
    ``n``."""
    factors = list(term.args) if isinstance(term, Mul) else [term]
    exponent: Expr = S.Zero
    constant: list[Expr] = []
    coefficient: list[Expr] = []
    for factor in factors:
        f = as_expr(factor)
        if not f.has(n):
            constant.append(f)
        elif isinstance(f, Pow) and f.base == n and not as_expr(f.exp).has(n):
            exponent = as_expr(exponent + f.exp)
        elif f == n:
            exponent = as_expr(exponent + 1)
        else:
            coefficient.append(f)
    return as_expr(Mul(*constant)), as_expr(Mul(*coefficient)), as_expr(-exponent)


def _closed_form(f: Expr, n: Symbol, s: Expr) -> Optional[DirichletSum]:
    """The series of a recognised coefficient pattern."""
    if f == 1:
        return as_expr(zeta(s)), as_boolean(re(s) > 1)
    if isinstance(f, Pow) and f.base == S.NegativeOne:
        exponent = as_expr(f.exp - n)
        if not exponent.has(n) and exponent.is_integer:
            sign = S.NegativeOne if exponent.is_even else S.One
            return as_expr(sign*(1 - 2**(1 - s))*zeta(s)), as_boolean(re(s) > 0)
        return None
    if isinstance(f, mobius) and f.args[0] == n:
        return as_expr(1/zeta(s)), as_boolean(re(s) > 1)
    if isinstance(f, Pow) and isinstance(f.base, mobius) and f.base.args[0] == n and f.exp == 2:
        return as_expr(zeta(s)/zeta(2*s)), as_boolean(re(s) > 1)
    if isinstance(f, totient) and f.args[0] == n:
        return as_expr(zeta(s - 1)/zeta(s)), as_boolean(re(s) > 2)
    if isinstance(f, divisor_sigma) and f.args[0] == n:
        k = as_expr(f.args[1]) if len(f.args) > 1 else S.One
        return as_expr(zeta(s)*zeta(s - k)), as_boolean(re(s) > Max(1, 1 + re(k)))
    if isinstance(f, log) and f.args[0] == n and isinstance(s, Symbol):
        return as_expr(-Derivative(zeta(s), s)), as_boolean(re(s) > 1)
    if isinstance(f, Pow) and isinstance(f.base, log) and f.base.args[0] == n and f.exp.is_Integer and f.exp > 0 \
            and isinstance(s, Symbol):
        order = int(f.exp)
        return as_expr((-1)**order*Derivative(zeta(s), (s, order))), as_boolean(re(s) > 1)
    return None


def dirichlet_series(term: Union[Expr, int], n: Symbol, lower: Union[Expr, int] = 1) -> Optional[DirichletSum]:
    """The closed form of ``Sum(term, (n, lower, oo))`` for a Dirichlet
    series with recognised coefficients, with its condition of
    convergence; ``None`` when the summand is not recognised.

    Examples
    ========

    >>> from sympy import mobius, totient, divisor_sigma, log
    >>> from sympy.abc import n, s
    >>> from sympy_extras.concrete import dirichlet_series
    >>> dirichlet_series(mobius(n)/n**s, n)
    (1/zeta(s), re(s) > 1)
    >>> dirichlet_series(totient(n)/n**s, n)
    (zeta(s - 1)/zeta(s), re(s) > 2)
    >>> dirichlet_series(divisor_sigma(n)/n**s, n)
    (zeta(s)*zeta(s - 1), re(s) > 2)
    >>> dirichlet_series((-1)**(n + 1)/n**s, n)
    ((1 - 2**(1 - s))*zeta(s), re(s) > 0)
    >>> dirichlet_series(log(n)/n**s, n)
    (-Derivative(zeta(s), s), re(s) > 1)
    >>> dirichlet_series(mobius(n)/n**2, n, 2)
    (-1 + 6/pi**2, True)
    """
    term_ = as_expr(sympify(term))
    lower_ = as_expr(sympify(lower))
    split = _split(term_, n)
    if split is None:
        return None
    c, f, s = split
    found = _closed_form(f, n, s)
    if found is None:
        return None
    value, condition = found
    result = as_expr(c*value)
    if lower_ != 1:
        if not isinstance(lower_, Integer) or lower_ < 1:
            return None
        head = as_expr(Sum(term_, (n, 1, lower_ - 1)).doit())
        result = as_expr(result - head)
    if not condition.free_symbols:
        condition = as_boolean(condition)
    return as_expr(result.doit()), condition
