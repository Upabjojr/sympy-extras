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
The module works on the expressions of the core
(:mod:`sympy_extras.core`) and imports no SymPy; only the isolated
values of the parameters go through SymPy at the call
(:func:`~sympy_extras.concrete.karr._with_special_values`).

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

from typing import Optional

from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import Derivative
from sympy_extras.core.logic import Boolean, BooleanFalse
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import Integer, oo
from sympy_extras.core.power import Pow, pow
from sympy_extras.core.relational import gt
from sympy_extras.core.add import add
from sympy_extras.core.symbol import Symbol
from sympy_extras.functions.combinatorial.numbers import DivisorSigma, Mobius, Totient
from sympy_extras.functions.elementary.complexes import re
from sympy_extras.functions.elementary.exponential import Log
from sympy_extras.functions.elementary.miscellaneous import max_
from sympy_extras.functions.special.zeta_functions import zeta

from .expr_with_limits import Sum
from .karr import SumPoint, _with_special_values, summation
from .summations import sum_doit

__all__ = ['dirichlet_series', 'DirichletSum']

#: the closed form of a Dirichlet series and the condition of convergence
DirichletSum = tuple[Expr, Boolean]

ONE = Integer(1)


def _e(b: Basic) -> Expr:
    if not isinstance(b, Expr):
        raise TypeError('an expression is expected, not %s' % (b,))
    return b


def _split(term: Expr, n: Symbol) -> tuple[Expr, Expr, Expr]:
    """``(c, f, s)`` with ``term == c * f(n) * n**(-s)``, ``c`` free of
    ``n``."""
    factors = [_e(a) for a in term.args] if isinstance(term, Mul) else [term]
    exponent: Expr = Integer(0)
    constant: list[Expr] = []
    coefficient: list[Expr] = []
    for f in factors:
        if not f.has(n):
            constant.append(f)
        elif isinstance(f, Pow) and f.base == n and not f.exp.has(n):
            exponent = add(exponent, f.exp)
        elif f == n:
            exponent = add(exponent, ONE)
        else:
            coefficient.append(f)
    return mul(*constant), mul(*coefficient), mul(Integer(-1), exponent)


def _closed_form(f: Expr, n: Symbol, s: Expr) -> Optional[DirichletSum]:
    """The series of a recognised coefficient pattern."""
    if f == ONE:
        return zeta(s), gt(re(s), ONE)
    if isinstance(f, Pow) and f.base == Integer(-1):
        exponent = add(f.exp, mul(Integer(-1), n))
        if not exponent.has(n) and exponent.is_integer is True:
            sign = Integer(-1) if exponent.is_even is True else ONE
            # sign*(1 - 2**(1 - s)) first, as SymPy multiplies: the sign
            # distributed over the sum
            signed = mul(sign, add(ONE, mul(Integer(-1), pow(Integer(2), add(ONE, mul(Integer(-1), s))))))
            return (mul(signed, zeta(s)),
                    gt(re(s), Integer(0)))
        return None
    if isinstance(f, Mobius) and f.args[0] == n:
        return pow(zeta(s), Integer(-1)), gt(re(s), ONE)
    if isinstance(f, Pow) and isinstance(f.base, Mobius) and f.base.args[0] == n and f.exp == Integer(2):
        return mul(zeta(s), pow(zeta(mul(Integer(2), s)), Integer(-1))), gt(re(s), ONE)
    if isinstance(f, Totient) and f.args[0] == n:
        return mul(zeta(add(s, Integer(-1))), pow(zeta(s), Integer(-1))), gt(re(s), Integer(2))
    if isinstance(f, DivisorSigma) and f.args[0] == n:
        k = _e(f.args[1]) if len(f.args) > 1 else ONE
        return (mul(zeta(s), zeta(add(s, mul(Integer(-1), k)))),
                gt(re(s), max_(ONE, add(ONE, re(k)))))
    if isinstance(f, Log) and f.args[0] == n and isinstance(s, Symbol):
        return mul(Integer(-1), Derivative(zeta(s), s)), gt(re(s), ONE)
    if (isinstance(f, Pow) and isinstance(f.base, Log) and f.base.args[0] == n and isinstance(f.exp, Integer)
            and f.exp.p > 0 and isinstance(s, Symbol)):
        order = f.exp.p
        return mul(Integer((-1)**order), Derivative(zeta(s), (s, order))), gt(re(s), ONE)
    return None


def dirichlet_series(term: ExprLike, n: Symbol, lower: ExprLike = 1,
                     special_values: bool = True) -> Optional[DirichletSum]:
    """The closed form of ``Sum(term, (n, lower, oo))`` for a Dirichlet
    series with recognised coefficients, with its condition of
    convergence; ``None`` when the summand is not recognised.

    With ``special_values`` (the default) the closed form gets a case for
    each isolated value of the parameters at which it is undefined while
    the series converges there, the series summed again at the point
    (:func:`~sympy_extras.concrete.karr.summation`): the alternating
    series at ``s = 1``, where ``(1 - 2**(1 - s))*zeta(s)`` is ``0*zoo``,
    is ``log(2)``. The points outside the region of convergence (``s =
    1`` for ``1/zeta(s)``) get none.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import mobius, totient, divisor_sigma, log
    >>> from sympy_extras.concrete.dirichlet import dirichlet_series
    >>> n, s = symbols('n s')
    >>> dirichlet_series(mobius(n)/n**s, n)
    (1/zeta(s), re(s) > 1)
    >>> dirichlet_series(totient(n)/n**s, n)
    (zeta(s - 1)/zeta(s), re(s) > 2)
    >>> dirichlet_series(divisor_sigma(n)/n**s, n)
    (zeta(s)*zeta(s - 1), re(s) > 2)
    >>> dirichlet_series((-1)**(n + 1)/n**s, n)
    (Piecewise((log(2), Eq(s, 1)), ((1 - 2**(1 - s))*zeta(s), True)), re(s) > 0)
    >>> dirichlet_series(log(n)/n**s, n)
    (-Derivative(zeta(s), s), re(s) > 1)
    >>> dirichlet_series(mobius(n)/n**2, n, 2)
    (-1 + 6/pi**2, True)
    """
    term_ = to_expr(term)
    lower_ = to_expr(lower)
    c, f, s = _split(term_, n)
    found = _closed_form(f, n, s)
    if found is None:
        return None
    value, condition = found
    result = mul(c, value)
    if lower_ != ONE:
        if not isinstance(lower_, Integer) or lower_.p < 1:
            return None
        head = sum_doit(Sum(term_, (n, ONE, Integer(lower_.p - 1))))
        result = add(result, mul(Integer(-1), head))
    value = _e(result.doit())
    if not special_values:
        return value, condition
    region = condition

    def converges(point: SumPoint) -> bool:
        return not isinstance(region.subs(point), BooleanFalse)

    def at_point(point: SumPoint) -> Optional[Expr]:
        return summation(term_.xreplace(point), (n, lower_, oo))

    return _with_special_values(value, term_, [n], [lower_, oo], at_point, keep_unevaluated=False,
                                defined=converges), condition
