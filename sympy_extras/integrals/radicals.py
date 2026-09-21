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
A rational function of `x` and `\sqrt{Q}` which is not of that form
(`1/((x + 3)\sqrt{x^2 - 1})`, `\sqrt{x^2 + x}/(x^2 + 1)^2`) goes through
an Euler substitution [Euler]_, which makes it a rational function of the
new variable `t`:

.. math::

    \sqrt{Q} = t - \sqrt{a}\, x \quad (a > 0), \qquad
    \sqrt{Q} = t\, (x - r_1) \quad (Q = a (x - r_1)(x - r_2)), \qquad
    \sqrt{Q} = x t + \sqrt{c_0} \quad (c_0 > 0),

and a linear `Q` through `t = \sqrt{Q}` itself.

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
.. [Euler] L. Euler, *Institutionum calculi integralis*, vol. 1, 1768,
   sections 88–91; G. M. Fichtenholz, *Differential- und
   Integralrechnung II*, section 282 (the three Euler substitutions).
"""
from __future__ import annotations

from typing import Optional

from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.numbers import Integer, Rational
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.hyperbolic import asinh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import asin
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polyroots import roots
from sympy.polys.polytools import Poly, cancel

from sympy_extras._timeout import attempt
from sympy_extras._typing import as_boolean, as_expr
from sympy_extras.assumptions import element
from sympy_extras.assumptions.ask import Assumptions, ask
from sympy_extras.settings import settings
from .conditions import numerically_equal

__all__ = ['quadratic_radical_antiderivative', 'euler_substitution_antiderivative']


class _Table:
    """The antiderivatives of ``x**n * Q**(m/2)`` for one quadratic ``Q``."""

    def __init__(self, a: Expr, b: Expr, c: Expr, x: Symbol, assumptions: Assumptions = None) -> None:
        self.a, self.b, self.c, self.x = a, b, c, x
        self.Q = as_expr(a * x**2 + b * x + c)
        self.discriminant = as_expr(b**2 - 4 * a * c)
        self.assumptions = assumptions
        self.memo: dict[tuple[int, int], Optional[Expr]] = {}

    def positive(self, e: Expr) -> bool:
        return e.is_positive is True or ask(as_boolean(e > 0), self.assumptions) is True

    def negative(self, e: Expr) -> bool:
        return e.is_negative is True or ask(as_boolean(e < 0), self.assumptions) is True

    def power(self, m: int) -> Optional[Expr]:
        """``Integral(Q**(m/2), x)`` for odd ``m``."""
        a, b, x, Q = self.a, self.b, self.x, self.Q
        if a == 0:
            if b == 0:
                return None
            return as_expr(Q**(Rational(m, 2) + 1) / (b * (Rational(m, 2) + 1)))
        if m == -1:
            if self.negative(a):
                if not self.positive(self.discriminant):
                    return None
                return as_expr(-asin((2 * a * x + b) / sqrt(self.discriminant)) / sqrt(-a))
            if self.positive(a):
                if self.negative(self.discriminant):
                    return as_expr(asinh((2 * a * x + b) / sqrt(-self.discriminant)) / sqrt(a))
                if self.positive(self.discriminant):
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
        """``Integral(x**n * Q**(m/2), x)`` for an integer ``n`` of either
        sign and odd ``m``."""
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
        if n < 0:
            return self._negative(n, m)
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

    def _negative(self, n: int, m: int) -> Optional[Expr]:
        """``Integral(Q**(m/2)/x**k, x)`` with ``k = -n >= 1``: GR 2.266 for
        ``1/(x*sqrt(Q))``, whose form depends on the sign of the constant
        term ``c`` (the substitution ``x = 1/t`` makes it ``Q**(-1/2)`` for
        the quadratic ``c*t**2 + b*t + a``); the other powers of ``Q`` at
        ``k = 1`` by ``Q = a*x**2 + b*x + c`` taken out of or into the
        radical; and the recurrence of the moments solved for the lowest
        power of ``x`` for ``k >= 2`` (``c*(n + 1)`` its coefficient)."""
        a, b, c, x, Q = self.a, self.b, self.c, self.x, self.Q
        if c == 0:
            return None                                     # x divides Q: a half-integer power of x
        if n == -1:
            if m == -1:
                if self.positive(c):
                    # a primitive up to the constant I*pi on x < 0, as the m = -1 logarithm above
                    return as_expr(-log((2 * sqrt(c) * sqrt(Q) + b * x + 2 * c) / x) / sqrt(c))
                if self.negative(c) and self.positive(self.discriminant):
                    # sqrt(x**2), whose derivative SymPy takes for a symbol not declared real (Abs gives re and im)
                    return as_expr(asin((2 * c + b * x) / (sqrt(x**2) * sqrt(self.discriminant))) / sqrt(-c))
                return None
            if m > 0:
                # Q**(m/2)/x = x*Q**(m/2 - 1)*a + Q**(m/2 - 1)*b + Q**(m/2 - 1)*c/x
                first, second, third = self.moment(1, m - 2), self.moment(0, m - 2), self.moment(-1, m - 2)
                if first is None or second is None or third is None:
                    return None
                return as_expr(a * first + b * second + c * third)
            # Q**(m/2 + 1)/x = a*x*Q**(m/2) + b*Q**(m/2) + c*Q**(m/2)/x, solved for the last
            higher, first, second = self.moment(-1, m + 2), self.moment(1, m), self.moment(0, m)
            if higher is None or first is None or second is None:
                return None
            return as_expr((higher - a * first - b * second) / c)
        # d/dx[x**(n+1) Q**(m/2+1)] = Q**(m/2) (a (n+m+3) x**(n+2) + b (n+2+m/2) x**(n+1) + c (n+1) x**n)
        first, second = self.moment(n + 1, m), self.moment(n + 2, m)
        if first is None or second is None:
            return None
        return as_expr((x**(n + 1) * Q**(Rational(m, 2) + 1) - b * (n + 2 + Rational(m, 2)) * first
                        - a * (n + m + 3) * second) / (c * (n + 1)))


def _odd_power(base: Expr, x: Symbol) -> tuple[Expr, int]:
    """``(R, k)`` with ``base == R**k`` for an odd ``k`` (``k = 1`` and the
    base itself when it is not a power of one polynomial)."""
    try:
        poly = Poly(base, x)
    except PolynomialError:
        return base, 1
    if poly.degree() < 2:
        return base, 1
    content, factors = poly.sqf_list()
    if len(factors) != 1:
        return base, 1
    root, k = factors[0]
    if k % 2 == 0 or k == 1:
        return base, 1
    R = as_expr(content**Rational(1, k) * root.as_expr()) if content == 1 else None
    if R is None:
        return base, 1
    return R, int(k)


def _terms(term: Expr, x: Symbol) -> Optional[list[tuple[Expr, int, Optional[Expr], int]]]:
    """The terms ``(coefficient, n, Q, m)`` with ``term == sum(coefficient *
    x**n * Q**(m/2))``, ``Q`` the base of the radical (``None`` and ``m = 0``
    for a monomial), the polynomial factors of ``term`` multiplied out; or
    ``None`` when ``term`` is not of that form."""
    coefficient: Expr = S.One
    polynomial: Expr = S.One
    base: Optional[Expr] = None
    m = 0
    shift = 0
    radicals: list[tuple[Expr, int]] = []
    for factor in Mul.make_args(term):
        f = as_expr(factor)
        if not f.has(x):
            coefficient = coefficient * f
            continue
        if isinstance(f, Pow) and isinstance(f.exp, Rational) and f.exp.q == 2:
            # sqrt(Q**3) is Q**(3/2): an odd power of one polynomial under the
            # root is the root of the polynomial to that power (where the
            # radicand is positive, so is Q; an even power would need Abs)
            base, power = _odd_power(as_expr(f.base), x)
            radicals.append((base, int(f.exp.p) * power))
            continue
        if isinstance(f, Pow) and f.base == x and isinstance(f.exp, Integer):
            shift += int(f.exp)                             # x**(-2): a negative power of x
            continue
        if isinstance(f, Pow) and isinstance(f.exp, Integer) and f.exp < 0 and as_expr(f.base).is_polynomial(x):
            radicals.append((as_expr(f.base), 2 * int(f.exp)))   # Q**(-2)*Q**(-1/2) is Q**(-5/2)
            continue
        if f.is_polynomial(x):
            polynomial = polynomial * f
            continue
        return None
    if len(radicals) > 1 and len({r for r, _ in radicals}) == 1:
        radicals = [(radicals[0][0], sum(p for _, p in radicals))]
    if len(radicals) == 1:
        base, m = radicals[0]
        if m % 2 == 0:
            return None                                     # a rational function, not a radical
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
        found.append((as_expr(coefficient * c), int(n) + shift, base, m))
    return found


def quadratic_radical_antiderivative(f: Expr, x: Symbol, assumptions: Assumptions = None,
                                     euler: bool = True) -> Optional[Expr]:
    """A real antiderivative of a sum of terms ``c * x**n * Q**(m/2)``, each
    ``Q`` quadratic or linear (the terms need not share it), ``m`` odd and
    ``n`` an integer of either sign; the terms without a radical are
    monomials. The signs the table needs (of the leading coefficient, of
    the discriminant, of the constant term for a negative ``n``) are those
    of the symbols or decided under the ``assumptions``. ``None`` when
    ``f`` is not of that form, when a case of the table is not covered (a
    sign undecided, a perfect square) or when the result does not check
    by differentiation. A rational function of ``x`` and ``sqrt(Q)`` of
    another form goes through :func:`euler_substitution_antiderivative`
    (not with ``euler=False``: the table alone).

    >>> from sympy import symbols, sqrt
    >>> from sympy_extras.integrals.radicals import quadratic_radical_antiderivative
    >>> x = symbols('x')
    >>> quadratic_radical_antiderivative(1/sqrt(2*x - x**2), x)
    asin(x - 1)
    >>> quadratic_radical_antiderivative(x**3*sqrt(x**2 + 1) + x, x)
    x**2*(x**2 + 1)**(3/2)/5 + x**2/2 - 2*(x**2 + 1)**(3/2)/15
    >>> quadratic_radical_antiderivative(sqrt(x**2 + 8) - 2*sqrt(x**2 + 4*x + 2), x)
    x*sqrt(x**2 + 8)/2 - (2*x + 4)*sqrt(x**2 + 4*x + 2)/2 + 2*log(2*x + 2*sqrt(x**2 + 4*x + 2) + 4) + 4*asinh(sqrt(2)*x/4)
    >>> quadratic_radical_antiderivative(sqrt(x**3 + 1), x) is None
    True
    """
    f_ = as_expr(f)
    found_ = _table_antiderivative(f_, x, assumptions)
    if found_ is not None or not euler:
        return found_
    return euler_substitution_antiderivative(f_, x, assumptions)


def _table_antiderivative(f_: Expr, x: Symbol, assumptions: Assumptions) -> Optional[Expr]:
    terms: list[tuple[Expr, int, Optional[Expr], int]] = []
    for summand in Add.make_args(f_):
        found = _terms(as_expr(summand), x)
        if found is None:
            return None
        terms.extend(found)
    # one table for each radicand: the cells of a region between two
    # conics leave sqrt(x**2 + 8)/2 - sqrt(x**2 + 4*x + 2), which was refused
    # for its two radicands and reached SymPy's integrate, the last resort,
    # after the whole budget (issue #65)
    tables: dict[Expr, _Table] = {}
    for _, _, radicand, _ in terms:
        if radicand is None or radicand in tables:
            continue
        try:
            poly = Poly(radicand, x)
        except PolynomialError:
            return None
        if poly.degree() > 2 or poly.degree() < 1:
            return None
        a, b, c = (as_expr(poly.coeff_monomial(x**2)), as_expr(poly.coeff_monomial(x)),
                   as_expr(poly.coeff_monomial(1)))
        tables[radicand] = _Table(a, b, c, x, assumptions)
    total: Expr = S.Zero
    for coefficient, n, radicand, m in terms:
        if radicand is None:
            total = total + (coefficient * log(x) if n == -1 else coefficient * x**(n + 1) / (n + 1))
            continue
        if m % 2 == 0:
            return None
        value = tables[radicand].moment(n, m)
        if value is None:
            return None
        total = total + coefficient * value
    # checked where the radicand is positive: elsewhere the integrand is
    # imaginary and the branches of the two sides need not agree
    facts = [as_boolean(table.Q > 0) for table in tables.values()]
    if isinstance(assumptions, (list, tuple)):
        facts.extend(as_boolean(item) for item in assumptions)
    elif assumptions is not None:
        facts.append(as_boolean(assumptions))
    if not numerically_equal(as_expr(total.diff(x)), f_, facts):
        return None
    return as_expr(total)


def _radicand(f: Expr, x: Symbol) -> Optional[tuple[Expr, Expr]]:
    """``(Q, g)`` with ``f == g(x, R)`` for ``R = sqrt(Q)``, ``Q`` a
    polynomial of degree one or two in ``x`` and ``g`` rational in ``x``
    and in the dummy ``R``; ``None`` for another shape (two radicands,
    a radicand of higher degree, a transcendental function)."""
    R = Dummy('R', positive=True)
    radicands: list[Expr] = []
    degrees: dict[Expr, int] = {}
    replacement: dict[Pow, Expr] = {}
    for node in f.atoms(Pow):
        base, exponent = as_expr(node.base), as_expr(node.exp)
        if not base.has(x) or not isinstance(exponent, Rational) or exponent.q == 1:
            continue
        if exponent.q != 2 or not base.is_polynomial(x):
            return None
        base, power = _odd_power(base, x)
        p_ = exponent.p * power
        try:
            degree = Poly(base, x).degree()
        except PolynomialError:
            return None
        if degree not in (1, 2):
            return None
        if base not in radicands:
            radicands.append(base)
            degrees[base] = degree
        replacement[node] = base**((p_ - p_ % 2) // 2) * R**(p_ % 2)
    if len(radicands) == 1:
        Q = radicands[0]
    elif len(radicands) == 2 and all(degrees[base] == 1 for base in radicands):
        # sqrt(1 + x)*(1 - x)**(-3/2): the square roots of the two linear
        # factors multiplied into the square root of their product, R =
        # sqrt(1 - x**2) (where both factors are positive), the powers
        # sqrt(1 + x)**m*sqrt(1 - x)**n being R**n*sqrt(1 + x)**(m - n)
        # with m - n even
        Q = as_expr((radicands[0] * radicands[1]).expand())
        first, second = radicands
        roots = {first: Dummy('S', positive=True), second: Dummy('T', positive=True)}
        for node in list(replacement):
            base, exponent = as_expr(node.base), as_expr(node.exp)
            if not isinstance(exponent, Rational):
                return None
            base, power = _odd_power(base, x)
            p_ = exponent.p * power
            replacement[node] = base**((p_ - p_ % 2) // 2) * roots[base]**(p_ % 2)
        g = as_expr(f.xreplace(replacement))
        # sqrt(first)*sqrt(second) is R; sqrt(first)**2 is first
        g = as_expr(g.subs(roots[second], R / roots[first]))
        g = as_expr(g.subs(roots[first]**2, first))
        g = as_expr(g.subs(roots[first]**(-2), 1 / first))
        if g.has(roots[first]) or not g.is_rational_function(x, R):
            return None
        return Q, as_expr(g.subs(R, Symbol('R')))
    else:
        return None
    g = as_expr(f.xreplace(replacement))
    if not g.is_rational_function(x, R):
        return None
    return Q, as_expr(g.subs(R, Symbol('R')))


def euler_substitution_antiderivative(f: Expr, x: Symbol, assumptions: Assumptions = None) -> Optional[Expr]:
    """An antiderivative of a rational function of ``x`` and ``sqrt(Q)``
    for one polynomial ``Q`` of degree one or two, by the substitution
    ``t = sqrt(Q)`` for a linear ``Q`` and by an Euler substitution for
    a quadratic one, which make the integrand a rational function of
    ``t``: ``sqrt(Q) = t - sqrt(a)*x`` for a positive leading coefficient
    ``a``, ``sqrt(Q) = t*(x - r)`` for a real root ``r``, ``sqrt(Q) = x*t +
    sqrt(c)`` for a positive constant term ``c`` (the signs those of the
    numbers or decided under the ``assumptions``). ``None`` when ``f`` is
    not of that form, when no substitution applies, or when the result
    does not check by differentiation.

    >>> from sympy import symbols, sqrt
    >>> from sympy_extras.integrals.radicals import euler_substitution_antiderivative
    >>> x = symbols('x')
    >>> euler_substitution_antiderivative(1/((x + 3)*sqrt(x**2 - 1)), x)
    sqrt(2)*log(x + sqrt(x**2 - 1) - 2*sqrt(2) + 3)/4 - sqrt(2)*log(x + sqrt(x**2 - 1) + 2*sqrt(2) + 3)/4
    >>> euler_substitution_antiderivative(sqrt(x + 1)/x, x)
    2*sqrt(x + 1) + log(sqrt(x + 1) - 1) - log(sqrt(x + 1) + 1)
    """
    from .risch.rationaltools import ratint
    f_ = as_expr(f)
    found = _radicand(f_, x)
    if found is None:
        return None
    Q, g = found
    R = Symbol('R')
    poly = Poly(Q, x)
    t = Dummy('t', positive=True)
    a, b, c = (as_expr(poly.coeff_monomial(x**2)), as_expr(poly.coeff_monomial(x)),
               as_expr(poly.coeff_monomial(1)))

    def positive(e: Expr) -> bool:
        return e.is_positive is True or ask(as_boolean(e > 0), assumptions) is True

    # x and sqrt(Q) as rational functions of t, and t in x
    candidates: list[tuple[Expr, Expr, Expr]] = []
    if a == 0:
        if b == 0:
            return None
        candidates.append((as_expr((t**2 - c) / b), t, as_expr(sqrt(Q))))
    else:
        if positive(a):
            # sqrt(Q) = t - sqrt(a) x: b x + c = t**2 - 2 sqrt(a) t x
            x_t = as_expr((t**2 - c) / (2 * sqrt(a) * t + b))
            candidates.append((x_t, as_expr(t - sqrt(a) * x_t), as_expr(sqrt(Q) + sqrt(a) * x)))
        real_roots = [as_expr(r) for r in roots(poly)
                      if as_expr(r).is_extended_real is True
                      or (as_expr(r).is_extended_real is None and ask(element(as_expr(r), S.Reals), assumptions))]
        if len(real_roots) == 2:
            # sqrt(Q) = t (x - r1): a (x - r2) = t**2 (x - r1)
            r1, r2 = real_roots
            x_t = as_expr((a * r2 - t**2 * r1) / (a - t**2))
            candidates.append((x_t, as_expr(t * (x_t - r1)), as_expr(sqrt(Q) / (x - r1))))
        if positive(c):
            # sqrt(Q) = x t + sqrt(c): a x + b = x t**2 + 2 t sqrt(c)
            x_t = as_expr((2 * t * sqrt(c) - b) / (a - t**2))
            candidates.append((x_t, as_expr(x_t * t + sqrt(c)), as_expr((sqrt(Q) - sqrt(c)) / x)))
    facts = [as_boolean(Q > 0)]
    if isinstance(assumptions, (list, tuple)):
        facts.extend(as_boolean(item) for item in assumptions)
    elif assumptions is not None:
        facts.append(as_boolean(assumptions))
    for x_t, root_t, back in candidates:
        h = as_expr(cancel(g.subs({x: x_t, R: root_t}) * x_t.diff(t)))
        if not h.is_rational_function(t):
            continue
        try:
            primitive = attempt(lambda: as_expr(ratint(h, t)), settings.timeout)
        except (IndexError, ValueError, NotImplementedError, ZeroDivisionError):
            # the rational integrator's internals fail on some inputs
            continue
        if primitive is None:
            continue
        result = as_expr(primitive.subs(t, back))
        if numerically_equal(as_expr(result.diff(x)), f_, facts):
            return result
    return None
