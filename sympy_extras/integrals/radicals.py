r"""Real antiderivatives of `x^n Q(x)^{m/2}`, `Q` a quadratic.

The integrands left by the cells of a region bounded by circles,
spheres and cylinders are sums of terms `c\, x^n\, Q(x)^{m/2}` with
`Q = a x^2 + b x + c_0` and `m` odd, `\sqrt{1 - x^2}` and
`x^2 \sqrt{1 - x^2}` typically. SymPy integrates `\sqrt{1 - x^2}` into
`x\sqrt{1 - x^2} + i \log(-i x + \sqrt{1 - x^2})`, the complex-logarithm
form of the arcsine, whose limits at a symbolic algebraic endpoint fail;
this module writes the antiderivatives in their real forms, from the
reduction formulas of Gradshteyn and Ryzhik 2.26:

.. math::

    \int Q^{-1/2}\,dx =
    \begin{cases}
      -\dfrac{1}{\sqrt{-a}} \arcsin\dfrac{2 a x + b}{\sqrt{b^2 - 4 a c_0}}
        & a < 0, \\[1ex]
      \dfrac{1}{\sqrt{a}} \operatorname{arsinh}\dfrac{2 a x + b}{\sqrt{4 a c_0 - b^2}}
        & a > 0,\ 4 a c_0 > b^2, \\[1ex]
      \dfrac{1}{\sqrt{a}} \log\bigl(2\sqrt{a}\sqrt{Q} + 2 a x + b\bigr)
        & a > 0,\ 4 a c_0 < b^2,
    \end{cases}

    \int Q^{m/2}\,dx = \frac{(2 a x + b)\,Q^{m/2}}{2 a (m + 1)}
      + \frac{m\,(4 a c_0 - b^2)}{4 a (m + 1)} \int Q^{m/2 - 1}\,dx
      \qquad (m \ge 1),

    \int Q^{-3/2}\,dx = \frac{2\,(2 a x + b)}{(4 a c_0 - b^2)\sqrt{Q}},

and, for the powers of `x`, from `d/dx\,[x^{n-1} Q^{m/2+1}]`,

.. math::

    a (n + m + 1) \int x^n Q^{m/2}\,dx = x^{n-1} Q^{m/2 + 1}
      - b\,(n + m/2) \int x^{n-1} Q^{m/2}\,dx
      - c_0\,(n - 1) \int x^{n-2} Q^{m/2}\,dx,

with `a x^2 = Q - b x - c_0` when the leading coefficient of that
relation vanishes. A linear `Q` (`a = 0`) goes through `x = (Q - c_0)/b`.
Every antiderivative is checked by differentiation at random points
before it is returned.

Examples
========

>>> from sympy import symbols, sqrt
>>> from sympy_extras.integrals.radicals import quadratic_radical_antiderivative
>>> x = symbols('x')
>>> quadratic_radical_antiderivative(sqrt(1 - x**2), x)
x*sqrt(1 - x**2)/2 + asin(x)/2
>>> quadratic_radical_antiderivative(x**2/sqrt(x**2 + 1), x)
x*sqrt(x**2 + 1)/2 - asinh(x)/2

References
==========

.. [GR] I. S. Gradshteyn and I. M. Ryzhik, *Table of Integrals, Series,
   and Products*, 7th ed., Academic Press, 2007, 2.26 (integrals of
   `x^n R^{\pm 1/2}` with `R = a + b x + c x^2`).
"""
from __future__ import annotations

from typing import Optional

from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.numbers import Rational
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.hyperbolic import asinh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import asin
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import Poly

from sympy_extras._typing import as_boolean, as_expr
from .conditions import numerically_equal

__all__ = ['quadratic_radical_antiderivative']


class _Table:
    """The antiderivatives of ``x**n * Q**(m/2)`` for one quadratic ``Q``."""

    def __init__(self, a: Expr, b: Expr, c: Expr, x: Symbol) -> None:
        self.a, self.b, self.c, self.x = a, b, c, x
        self.Q = as_expr(a * x**2 + b * x + c)
        self.discriminant = as_expr(b**2 - 4 * a * c)
        self.memo: dict[tuple[int, int], Optional[Expr]] = {}

    def power(self, m: int) -> Optional[Expr]:
        """``Integral(Q**(m/2), x)`` for odd ``m``."""
        a, b, x, Q = self.a, self.b, self.x, self.Q
        if a == 0:
            if b == 0:
                return None
            return as_expr(Q**(Rational(m, 2) + 1) / (b * (Rational(m, 2) + 1)))
        if m == -1:
            if a.is_negative:
                if not self.discriminant.is_positive:
                    return None
                return as_expr(-asin((2 * a * x + b) / sqrt(self.discriminant)) / sqrt(-a))
            if a.is_positive:
                if self.discriminant.is_negative:
                    return as_expr(asinh((2 * a * x + b) / sqrt(-self.discriminant)) / sqrt(a))
                if self.discriminant.is_positive:
                    # the argument keeps its sign on each component of Q > 0
                    # (it vanishes only at a root of Q): a primitive up to the
                    # constant I*pi on the component where it is negative
                    return as_expr(log(2 * sqrt(a) * sqrt(Q) + 2 * a * x + b) / sqrt(a))
                if self.discriminant == 0:
                    return None                             # a perfect square: not a radical
            return None
        if m == -3:
            if self.discriminant == 0:
                return None
            return as_expr(-2 * (2 * a * x + b) / (self.discriminant * sqrt(Q)))
        if m < -3:
            # GR 2.264.6: Integral(Q**(-(2k+1)/2)) from Integral(Q**(-(2k-1)/2))
            k = (-m - 1) // 2
            previous = self.power(m + 2)
            if previous is None or self.discriminant == 0:
                return None
            return as_expr(-2 * (2 * a * x + b) * Q**(Rational(m, 2) + 1) / ((2 * k - 1) * self.discriminant)
                           - 8 * a * (k - 1) * previous / ((2 * k - 1) * self.discriminant))
        previous = self.power(m - 2)
        if previous is None:
            return None
        return as_expr((2 * a * x + b) * Q**Rational(m, 2) / (2 * a * (m + 1))
                       - m * self.discriminant * previous / (4 * a * (m + 1)))

    def moment(self, n: int, m: int) -> Optional[Expr]:
        """``Integral(x**n * Q**(m/2), x)`` for ``n >= 0`` and odd ``m``."""
        key = (n, m)
        if key in self.memo:
            return self.memo[key]
        self.memo[key] = None                                # guards the recursion
        value = self._moment(n, m)
        self.memo[key] = value
        return value

    def _moment(self, n: int, m: int) -> Optional[Expr]:
        a, b, c, x, Q = self.a, self.b, self.c, self.x, self.Q
        if n == 0:
            return self.power(m)
        if a == 0:
            # x = (Q - c)/b
            if b == 0:
                return None
            higher, lower = self.moment(n - 1, m + 2), self.moment(n - 1, m)
            if higher is None or lower is None:
                return None
            return as_expr((higher - c * lower) / b)
        if n == 1:
            lower = self.moment(0, m)
            if lower is None:
                return None
            return as_expr(Q**(Rational(m, 2) + 1) / (a * (m + 2)) - b * lower / (2 * a))
        leading = as_expr(a * (n + m + 1))
        if leading != 0:
            first, second = self.moment(n - 1, m), self.moment(n - 2, m)
            if first is None or second is None:
                return None
            return as_expr((x**(n - 1) * Q**(Rational(m, 2) + 1) - b * (n + Rational(m, 2)) * first
                            - c * (n - 1) * second) / leading)
        # a x**2 = Q - b x - c
        higher, first, second = self.moment(n - 2, m + 2), self.moment(n - 1, m), self.moment(n - 2, m)
        if higher is None or first is None or second is None:
            return None
        return as_expr((higher - b * first - c * second) / a)


def _terms(term: Expr, x: Symbol) -> Optional[list[tuple[Expr, int, Optional[Expr], int]]]:
    """The terms ``(coefficient, n, Q, m)`` with ``term == sum(coefficient *
    x**n * Q**(m/2))``, ``Q`` the base of the radical (``None`` and ``m = 0``
    for a monomial), the polynomial factors of ``term`` multiplied out; or
    ``None`` when ``term`` is not of that form."""
    coefficient: Expr = S.One
    polynomial: Expr = S.One
    base: Optional[Expr] = None
    m = 0
    radicals: list[tuple[Expr, int]] = []
    for factor in Mul.make_args(term):
        f = as_expr(factor)
        if not f.has(x):
            coefficient = coefficient * f
            continue
        if isinstance(f, Pow) and isinstance(f.exp, Rational) and f.exp.q == 2:
            radicals.append((as_expr(f.base), int(f.exp.p)))
            continue
        if f.is_polynomial(x):
            polynomial = polynomial * f
            continue
        return None
    if len(radicals) == 1:
        base, m = radicals[0]
    elif len(radicals) > 1:
        # sqrt(1 - x) sqrt(1 + x) is sqrt(1 - x**2) where both are real:
        # radicals of linear factors with positive odd exponents multiplied
        # into one radicand of degree at most two, the rest polynomial
        if any(p < 1 or not r.is_polynomial(x) or Poly(r, x).degree() != 1 for r, p in radicals):
            return None
        product: Expr = S.One
        for r, p in radicals:
            polynomial = polynomial * r**((p - 1) // 2)
            product = product * r
        base, m = as_expr(product.expand()), 1
    try:
        poly = Poly(polynomial, x)
    except PolynomialError:
        return None
    found: list[tuple[Expr, int, Optional[Expr], int]] = []
    for (n,), c in poly.terms():
        found.append((as_expr(coefficient * c), int(n), base, m))
    return found


def quadratic_radical_antiderivative(f: Expr, x: Symbol) -> Optional[Expr]:
    """A real antiderivative of a sum of terms ``c * x**n * Q**(m/2)``, one
    quadratic (or linear) ``Q`` for all the radical terms, ``m`` odd; the
    terms without a radical are monomials. ``None`` when ``f`` is not of
    that form, when a case of the table is not covered (a radicand of
    unknown sign of leading coefficient, a perfect square) or when the
    result does not check by differentiation.

    >>> from sympy import symbols, sqrt
    >>> from sympy_extras.integrals.radicals import quadratic_radical_antiderivative
    >>> x = symbols('x')
    >>> quadratic_radical_antiderivative(1/sqrt(2*x - x**2), x)
    asin(x - 1)
    >>> quadratic_radical_antiderivative(x**3*sqrt(x**2 + 1) + x, x)
    x**2*(x**2 + 1)**(3/2)/5 + x**2/2 - 2*(x**2 + 1)**(3/2)/15
    >>> quadratic_radical_antiderivative(sqrt(x**3 + 1), x) is None
    True
    """
    f_ = as_expr(f)
    terms: list[tuple[Expr, int, Optional[Expr], int]] = []
    for summand in Add.make_args(f_):
        found = _terms(as_expr(summand), x)
        if found is None:
            return None
        terms.extend(found)
    bases: list[Expr] = []
    for _, _, radicand, _ in terms:
        if radicand is not None and radicand not in bases:
            bases.append(radicand)
    if len(bases) > 1:
        return None
    table: Optional[_Table] = None
    if bases:
        base = bases[0]
        try:
            poly = Poly(base, x)
        except PolynomialError:
            return None
        if poly.degree() > 2 or poly.degree() < 1:
            return None
        a, b, c = (as_expr(poly.coeff_monomial(x**2)), as_expr(poly.coeff_monomial(x)),
                   as_expr(poly.coeff_monomial(1)))
        table = _Table(a, b, c, x)
    total: Expr = S.Zero
    for coefficient, n, radicand, m in terms:
        if radicand is None:
            total = total + coefficient * x**(n + 1) / (n + 1)
            continue
        if table is None or m % 2 == 0:
            return None
        value = table.moment(n, m)
        if value is None:
            return None
        total = total + coefficient * value
    # checked where the radicand is positive: elsewhere the integrand is
    # imaginary and the branches of the two sides need not agree
    facts = [as_boolean(table.Q > 0)] if table is not None else []
    if not numerically_equal(as_expr(total.diff(x)), f_, facts):
        return None
    return as_expr(total)
