"""Definite integrals with the square root of a cubic or a quartic,
reduced to Legendre's elliptic integrals.

An integral `\\int R(x)\\, P(x)^{\\pm 1/2}\\, dx` with `R` rational and `P` a
polynomial of degree three or four with real roots is elliptic. The
classical reduction [Byrd]_, [DLMF]_ (chapter 19.29) maps the range
onto the standard one through a linear or a bilinear substitution
`x = X(s)`, `s = \\sin^2\\theta`, chosen so that the roots of `P` go to
`0`, `1`, `1/m` and (for a quartic) to the pole of `X`: on the cell of
the real line between consecutive roots, or beyond the extreme roots,
the radicand becomes `C\\, s(1 - s)(1 - m s)` up to a square, and the
integrand

.. math::

    R(X(s))\\, X'(s)\\, P(X(s))^{\\pm 1/2}\\, ds = \\frac{Q(s)\\, ds}{\\sqrt{s(1 - s)(1 - m s)}}
    = \\frac{2\\, Q(\\sin^2\\theta)\\, d\\theta}{\\sqrt{1 - m \\sin^2\\theta}}

with `Q` rational. The partial fractions of `Q` give the three kinds
(SymPy's parameter convention, `m = k^2`):

* `\\int_0^{\\pi/2} \\sin^{2j}\\theta\\, d\\theta / \\sqrt{1 - m\\sin^2\\theta} = J_j(m)` with
  `J_0 = K(m)`, `J_1 = (K - E)/m` and the recurrence
  `(2j + 1) m J_{j+1} = 2j(1 + m) J_j - (2j - 1) J_{j-1}` (Byrd–Friedman 310);
* `\\int_0^{\\pi/2} d\\theta / ((1 - n\\sin^2\\theta)\\sqrt{1 - m\\sin^2\\theta}) = \\Pi(n \\mid m)`;
* the incomplete integrals `F(\\phi \\mid m)`, `E(\\phi \\mid m)`,
  `\\Pi(n; \\phi \\mid m)` when an endpoint is not a root of `P`, with
  `\\phi = \\arcsin\\sqrt{s}` at the image of the endpoint.

An even radicand with an even `R` is first reduced by `y = x^2` (the
lemniscate integral `\\int_0^1 dx/\\sqrt{1 - x^4}` becomes a cubic case);
a quartic beyond its extreme root by `u = 1/(x - r)`, which makes it a
cubic. Trigonometric integrands `Q(\\sin^2\\theta)(1 - m\\sin^2\\theta)^{\\pm 1/2}`
over `(0, \\phi)` are read directly. The order of symbolic roots is
decided by :func:`sympy_extras.assumptions.ask` under the assumptions.

Examples
========

>>> from sympy import symbols, sqrt, oo, sin, pi
>>> from sympy_extras.integrals.elliptic import elliptic_integral
>>> x, theta = symbols('x theta')
>>> k = symbols('k', positive=True)
>>> elliptic_integral(1/sqrt(x*(1 - x)*(1 - k**2*x)), x, 0, 1, k < 1)
ConditionalValue(2*elliptic_k(k**2))
>>> elliptic_integral(1/sqrt(x*(x - 1)*(2 - x)*(3 - x)), x, 1, 2)
ConditionalValue(elliptic_k(3/4))
>>> elliptic_integral(sqrt(1 - k**2*sin(theta)**2), theta, 0, pi/2, k < 1)
ConditionalValue(elliptic_e(k**2))

References
==========

.. [Byrd] P. F. Byrd, M. D. Friedman, *Handbook of Elliptic Integrals
   for Engineers and Scientists*, 2nd ed., Springer, 1971, chapter 2
   (the substitutions) and the tables 230–260 and 310–340.
.. [DLMF] NIST Digital Library of Mathematical Functions, chapter 19,
   sections 19.2 (Legendre's integrals) and 19.29 (reduction of
   `\\int R(x, \\sqrt{P})\\, dx`), https://dlmf.nist.gov/19.
.. [GR] I. S. Gradshteyn, I. M. Ryzhik, *Table of Integrals, Series,
   and Products*, 7th ed., sections 3.13–3.16.
"""
from __future__ import annotations

from typing import Optional, Sequence

from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.numbers import Rational, oo, pi
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import TrigonometricFunction, asin, sin
from sympy.functions.special.elliptic_integrals import elliptic_k, elliptic_e, elliptic_f, elliptic_pi
from sympy.polys.partfrac import apart
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import Poly, cancel, factor_list
from sympy.polys.rootoftools import ComplexRootOf
from sympy.solvers.solvers import solve as sympy_solve

from sympy_extras._typing import ExprLike, as_boolean, as_expr, free_symbols
from sympy.logic.boolalg import Boolean
from sympy_extras.assumptions.ask import Assumptions, ask
from .conditions import ConditionalValue, plain_symbols, with_symbol_facts

__all__ = ['elliptic_integral', 'radicand', 'real_roots', 'legendre_reduction', 'Reduction']


def _ask(query: Boolean, assumptions: Assumptions) -> Optional[bool]:
    """``ask`` with the flags of the symbols (``positive=True``) turned
    into statements on plain symbols, which the CAD can use."""
    symbols = free_symbols(query)
    plain = plain_symbols(symbols)
    if not plain:
        return ask(query, assumptions)
    return ask(as_boolean(query.xreplace(plain)), with_symbol_facts(assumptions, symbols, plain))

_HALF = S.Half
#: how many times the substitutions may be chained (an even radicand,
#: then a quartic beyond its extreme root, then the cubic case)
_MAX_DEPTH = 4


class Reduction:
    """The Legendre form of an elliptic integral,
    ``Integral(Q(s)/sqrt(s*(1 - s)*(1 - m*s)), (s, lower, upper))``.

    Attributes
    ==========

    Q : Expr
        A rational function of ``s``.
    m : Expr
        The parameter (``k**2``).
    s : Symbol
    lower, upper : Expr
        The range of ``s`` inside ``[0, 1]``.
    """

    def __init__(self, Q: Expr, m: Expr, s: Symbol, lower: Expr, upper: Expr) -> None:
        self.Q = Q
        self.m = m
        self.s = s
        self.lower = lower
        self.upper = upper

    def __repr__(self) -> str:
        return "Reduction(%s, m=%s, (%s, %s, %s))" % (self.Q, self.m, self.s, self.lower, self.upper)


# ---------------------------------------------------------------------------
# Reading the integrand

def radicand(f: Expr, x: Symbol) -> Optional[tuple[Expr, Expr, Expr]]:
    """``(R, P, e)`` with ``f == R * P**e``, ``e`` equal to ``1/2`` or
    ``-1/2``, ``P`` a polynomial in ``x`` of degree three or four and
    ``R`` a rational function of ``x``; the square roots of several
    factors are combined into one, and a half-integer power ``x**(3/2)``
    is ``x`` times ``sqrt(x)``. ``None`` when ``f`` is not of this form.

    >>> from sympy import symbols, sqrt
    >>> from sympy_extras.integrals.elliptic import radicand
    >>> x = symbols('x')
    >>> radicand(x/sqrt((1 - x**2)*(4 - x**2)), x)
    (x, (1 - x**2)*(4 - x**2), -1/2)
    >>> radicand(sqrt(x)*sqrt(1 - x)/sqrt(4 - x), x)
    (-1/(x - 4), x*(1 - x)*(4 - x), 1/2)
    """
    halves: list[tuple[Expr, Expr]] = []
    rest: Expr = S.One
    for factor in Mul.make_args(f):
        factor_ = as_expr(factor)
        if isinstance(factor_, Pow) and factor_.has(x) and isinstance(factor_.exp, Rational) \
                and factor_.exp.q == 2:
            base = as_expr(factor_.base)
            if not base.is_polynomial(x):
                return None
            halves.append((base, as_expr(factor_.exp)))
        else:
            rest = rest * factor_
    if not halves:
        return None
    # the sign of the root: that of the factor of highest degree with a
    # bare square root, the denominator by default
    exponent: Expr = -_HALF
    best = -1
    for base, q in halves:
        degree = Poly(base, x).degree()
        if abs(q) == _HALF and degree > best:
            best, exponent = degree, q
    P: Expr = S.One
    for base, q in halves:
        P = P * base
        rest = rest * base**(q - exponent)
    try:
        degree = Poly(P, x).degree()
    except PolynomialError:
        return None
    rest = as_expr(cancel(rest))
    if degree not in (3, 4) or not rest.is_rational_function(x):
        return None
    return (rest, P, exponent)


def real_roots(P: Expr, x: Symbol, assumptions: Assumptions = None) -> Optional[list[Expr]]:
    """The roots of ``P`` sorted increasingly, when they are all real and
    distinct: numeric roots as :class:`~sympy.polys.rootoftools.CRootOf`
    or radicals, symbolic ones from the linear and quadratic factors of
    ``P``, ordered by :func:`~sympy_extras.assumptions.ask` under the
    assumptions. ``None`` otherwise.

    >>> from sympy import symbols
    >>> from sympy_extras.integrals.elliptic import real_roots
    >>> x = symbols('x')
    >>> k = symbols('k', positive=True)
    >>> real_roots((1 - x**2)*(1 - k**2*x**2), x, k < 1)
    [-1/k, -1, 1, 1/k]
    """
    try:
        poly = Poly(P, x)
    except PolynomialError:
        return None
    found: list[Expr] = []
    if not poly.free_symbols_in_domain:
        all_roots = poly.all_roots()
        for r in all_roots:
            r_ = as_expr(r)
            if r_.is_real is not True:
                return None
            found.append(r_)
    else:
        _, factors = factor_list(P, x)
        for factor, multiplicity in factors:
            factor_ = as_expr(factor)
            if multiplicity != 1 or not factor_.has(x):
                if factor_.has(x):
                    return None
                continue
            try:
                degree = Poly(factor_, x).degree()
            except PolynomialError:
                return None
            if degree > 2:
                return None
            w = Dummy('w')
            solutions = sympy_solve(factor_.subs(x, w), w)
            for r in solutions:
                r_ = as_expr(r)
                if _ask(as_boolean(r_ - r_ >= 0), assumptions) is not True and r_.is_extended_real is not True:
                    return None
                found.append(r_)
    if len(found) != poly.degree():
        return None
    ordered: list[Expr] = []
    for r in found:
        position = 0
        for q in ordered:
            verdict = _less(q, r, assumptions)
            if verdict is None:
                return None
            if verdict:
                position += 1
        ordered.insert(position, r)
    return ordered


def _less(a: Expr, b: Expr, assumptions: Assumptions) -> Optional[bool]:
    """``a < b`` for two real numbers, ``None`` if undecided or equal."""
    if a == b:
        return None
    if isinstance(a, ComplexRootOf) or isinstance(b, ComplexRootOf):
        difference = as_expr((b - a).evalf(30))
        if difference.is_positive:
            return True
        if difference.is_negative:
            return False
        return None
    # the sign of a quotient is the sign of the product of its parts, a
    # polynomial question the assumptions can decide (1 < 1/k**2 under k < 1)
    numerator, denominator = as_expr(cancel(b - a)).as_numer_denom()
    return _ask(as_boolean(as_expr(numerator) * as_expr(denominator) > 0), assumptions)


def _position(point: Expr, roots: Sequence[Expr], assumptions: Assumptions) -> Optional[int]:
    """The cell of ``point`` among the roots: ``2 i`` when it is the
    ``i``-th root, ``2 i + 1`` when it lies strictly between the ``i``-th
    and the next (``-oo`` is ``-1``, ``oo`` is ``2 n + 1``)."""
    if point == -oo:
        return -1
    if point == oo:
        return 2 * len(roots) - 1
    for i, r in enumerate(roots):
        if point == r or (_ask(as_boolean(point - r >= 0), assumptions) is True
                          and _ask(as_boolean(point - r <= 0), assumptions) is True):
            return 2 * i
        below = _less(point, r, assumptions)
        if below is None:
            return None
        if below:
            return 2 * i - 1
    return 2 * len(roots) - 1


# ---------------------------------------------------------------------------
# The substitutions

class _Substitution:
    """``x = X(s)`` on a cell, with ``m`` and the pole ``s = -gamma`` of a
    bilinear map (``None`` for a linear one)."""

    def __init__(self, X: Expr, m: Expr, s: Symbol) -> None:
        self.X = X
        self.m = m
        self.s = s


def _cell_substitution(roots: Sequence[Expr], cell: int, s: Symbol,
                       assumptions: Assumptions) -> Optional[_Substitution]:
    """The substitution of the cell (an odd index of :func:`_position`)."""
    n = len(roots)
    if cell < -1 or cell > 2 * n - 1 or cell % 2 == 0:
        return None
    if cell == -1 or cell == 2 * n - 1:
        if n != 3:
            return None
        if cell == 2 * n - 1:
            # (r_3, oo): x = r_3 + (r_3 - r_2) s/(1 - s), m = (r_2 - r_1)/(r_3 - r_1)
            r1, r2, r3 = roots
            return _Substitution(r3 + (r3 - r2) * s / (1 - s), (r2 - r1) / (r3 - r1), s)
        r1, r2, r3 = roots
        # (-oo, r_1): x = r_1 - (r_2 - r_1) s/(1 - s), m = (r_3 - r_2)/(r_3 - r_1)
        return _Substitution(r1 - (r2 - r1) * s / (1 - s), (r3 - r2) / (r3 - r1), s)
    i = (cell - 1) // 2
    lo, hi = roots[i], roots[i + 1]
    others = [r for j, r in enumerate(roots) if j not in (i, i + 1)]
    if n == 3:
        r = others[0]
        if _less(hi, r, assumptions):
            return _Substitution(lo + (hi - lo) * s, (hi - lo) / (r - lo), s)
        return _Substitution(hi - (hi - lo) * s, (hi - lo) / (hi - r), s)
    # a quartic: a bilinear map with its pole at one of the other roots,
    # oriented so that the parameter lies in (0, 1)
    for pole, other in ((others[0], others[1]), (others[1], others[0])):
        for start, end in ((lo, hi), (hi, lo)):
            gamma = (pole - end) / (end - start)
            X = as_expr((pole * s + start * gamma) / (s + gamma))
            image = _preimage(X, s, other)
            if image is None or image == 0:
                continue
            m = as_expr(cancel(1 / image))
            if _ask(as_boolean(m > 0), assumptions) and _ask(as_boolean(m < 1), assumptions):
                return _Substitution(X, m, s)
    return None


def _preimage(X: Expr, s: Symbol, point: Expr) -> Optional[Expr]:
    """The ``s`` with ``X(s) == point`` (``1`` for a point at infinity
    of a map with its pole at ``s = 1``)."""
    if point in (oo, -oo):
        return S.One
    w = Dummy('w')
    solutions = sympy_solve(X.subs(s, w) - point, w)
    if len(solutions) != 1:
        return None
    return as_expr(cancel(as_expr(solutions[0])))


def _reduce_on_cell(R: Expr, P: Expr, e: Expr, x: Symbol, roots: Sequence[Expr], lo: Expr, hi: Expr,
                    cell: int, assumptions: Assumptions) -> Optional[Reduction]:
    """The Legendre form on a cell of the roots."""
    s = Dummy('s', positive=True)
    substitution = _cell_substitution(roots, cell, s, assumptions)
    if substitution is None:
        return None
    X, m = substitution.X, substitution.m
    s_lo, s_hi = _preimage(X, s, lo), _preimage(X, s, hi)
    if s_lo is None or s_hi is None:
        return None
    sign: Expr = S.One
    if _ask(as_boolean(s_lo < s_hi), assumptions) is False:
        s_lo, s_hi, sign = s_hi, s_lo, S.NegativeOne
    for bound in (s_lo, s_hi):
        if _ask(as_boolean(bound >= 0), assumptions) is not True or _ask(as_boolean(bound <= 1), assumptions) is not True:
            return None
    split = _split_radicand(as_expr(P.subs(x, X)), s, m, assumptions)
    if split is None:
        return None
    C, part = split
    # P(X(s))^e = C^e part^e sqrt(s(1-s)(1-ms))^(2e): with e = -1/2 the
    # integrand is R X' / (sqrt(C) part sqrt(s(1-s)(1-ms))), with e = 1/2
    # it is R X' sqrt(C) part s(1-s)(1-ms) / sqrt(s(1-s)(1-ms))
    Xp = as_expr(X.diff(s))
    if e == -_HALF:
        Q = as_expr(R.subs(x, X) * Xp / (sqrt(C) * part))
    else:
        Q = as_expr(R.subs(x, X) * Xp * sqrt(C) * part * s * (1 - s) * (1 - m * s))
    Q = as_expr(cancel(sign * Q))
    if not Q.is_rational_function(s):
        return None
    return Reduction(Q, as_expr(m), s, s_lo, s_hi)


def _split_radicand(P_s: Expr, s: Symbol, m: Expr,
                    assumptions: Assumptions) -> Optional[tuple[Expr, Expr]]:
    """``P(X(s))`` written as ``C * part**2 * s*(1 - s)*(1 - m*s)`` with
    ``C`` a positive constant and ``part`` a rational function of ``s``
    (positive on the range): the exponents of the three Legendre factors
    must be odd and those of every other factor even. ``(C, part)``, or
    ``None``."""
    numerator, denominator = as_expr(cancel(P_s)).as_numer_denom()
    C: Expr = S.One
    exponents = {'s': 0, '1-s': 0, '1-ms': 0}
    forms = {'s': s, '1-s': 1 - s, '1-ms': 1 - m * s}
    others: list[tuple[Expr, int]] = []
    for polynomial, orientation in ((as_expr(numerator), 1), (as_expr(denominator), -1)):
        try:
            coefficient, factors = factor_list(polynomial, s)
        except PolynomialError:
            return None
        C = C * as_expr(coefficient)**orientation
        for factor, multiplicity in factors:
            factor_ = as_expr(factor)
            exponent = orientation * multiplicity
            if not factor_.has(s):
                C = C * factor_**exponent
                continue
            matched = False
            for name, form in forms.items():
                ratio = as_expr(cancel(factor_ / form))
                if not ratio.has(s):
                    C = C * ratio**exponent
                    exponents[name] += exponent
                    matched = True
                    break
            if not matched:
                others.append((factor_, exponent))
    if any(v % 2 == 0 for v in exponents.values()):
        return None
    if any(exponent % 2 for _, exponent in others):
        return None
    if _ask(as_boolean(C > 0), assumptions) is not True:
        return None
    part: Expr = S.One
    for name, form in forms.items():
        part = part * form**((exponents[name] - 1) // 2)
    for factor_, exponent in others:
        half = exponent // 2
        if half % 2:
            # an odd power of the square root of an even power: the
            # absolute value, decided on the range
            if _ask(as_boolean(factor_.subs(s, S.Half) < 0), assumptions):
                factor_ = -factor_
            elif _ask(as_boolean(factor_.subs(s, S.Half) > 0), assumptions) is not True:
                return None
        part = part * factor_**half
    return (as_expr(C), as_expr(part))


def _square_root(expr: Expr, s: Symbol) -> Optional[Expr]:
    """The square root of a product of even powers of linear factors of
    ``s`` (positive on the range, as squares are), or of a constant."""
    if not expr.has(s):
        return sqrt(expr)
    coefficient, factors = factor_list(expr, s)
    result: Expr = sqrt(as_expr(coefficient))
    for factor, multiplicity in factors:
        if multiplicity % 2:
            # an odd power: (1 - s)^3 of the map with its pole at s = 1
            # has a positive base on the range
            base = as_expr(factor)
            if base.subs(s, S.Half).is_positive:
                result = result * base**Rational(multiplicity, 2)
            elif (-base).subs(s, S.Half).is_positive:
                result = result * (-base)**Rational(multiplicity, 2)
            else:
                return None
        else:
            result = result * as_expr(factor)**(multiplicity // 2)
    return as_expr(result)


# ---------------------------------------------------------------------------
# The reduction to K, E and Pi

def _complete_power(j: int, m: Expr) -> Expr:
    """``J_j(m) = Integral(sin(theta)**(2 j)/sqrt(1 - m sin(theta)**2), (theta, 0, pi/2))``."""
    values: list[Expr] = [as_expr(elliptic_k(m)), as_expr((elliptic_k(m) - elliptic_e(m)) / m)]
    for k in range(1, j):
        values.append(as_expr((2 * k * (1 + m) * values[k] - (2 * k - 1) * values[k - 1]) / ((2 * k + 1) * m)))
    return values[j]


def legendre_reduction(reduction: Reduction, assumptions: Assumptions = None) -> Optional[Expr]:
    """``Integral(Q(s)/sqrt(s*(1 - s)*(1 - m*s)), (s, lower, upper))`` in
    Legendre's integrals, from the partial fractions of ``Q``; ``None``
    for a multiple pole, a pole inside the range (a divergent integral)
    or an incomplete integral with a power of ``s`` above the first.

    Examples
    ========

    >>> from sympy import symbols, S
    >>> from sympy_extras.integrals.elliptic import Reduction, legendre_reduction
    >>> s = symbols('s', positive=True)
    >>> m = symbols('m', positive=True)
    >>> legendre_reduction(Reduction(S.One, m, s, S.Zero, S.One))
    2*elliptic_k(m)
    >>> legendre_reduction(Reduction(s, m, s, S.Zero, S.One))
    2*(-elliptic_e(m) + elliptic_k(m))/m
    """
    Q, m, s = reduction.Q, reduction.m, reduction.s
    total: Expr = S.Zero
    for bound, sign in ((reduction.upper, S.One), (reduction.lower, S.NegativeOne)):
        if bound == 0:
            continue
        value = _from_zero(Q, m, s, as_expr(bound), assumptions)
        if value is None:
            return None
        total = total + sign * value
    return as_expr(total)


def _from_zero(Q: Expr, m: Expr, s: Symbol, upper: Expr, assumptions: Assumptions) -> Optional[Expr]:
    complete = upper == 1
    phi = as_expr(asin(sqrt(upper)))
    try:
        parts = apart(Q, s)
    except (PolynomialError, NotImplementedError):
        return None
    total: Expr = S.Zero
    for term in Add.make_args(parts):
        term_ = as_expr(term)
        numerator, denominator = term_.as_numer_denom()
        numerator_, denominator_ = as_expr(numerator), as_expr(denominator)
        if not denominator_.has(s):
            # a polynomial term c s^j
            polynomial = Poly(term_, s)
            for (j,), c in polynomial.terms():
                c_ = as_expr(c)
                if complete:
                    total = total + 2 * c_ * _complete_power(j, m)
                elif j == 0:
                    total = total + 2 * c_ * elliptic_f(phi, m)
                elif j == 1:
                    total = total + 2 * c_ * (elliptic_f(phi, m) - elliptic_e(phi, m)) / m
                else:
                    return None
            continue
        pole_poly = Poly(denominator_, s)
        if pole_poly.degree() != 1 or numerator_.has(s):
            return None
        # c/(alpha s + beta) = -(c/beta)/(1 - n s) with n = -alpha/beta
        alpha, beta = as_expr(pole_poly.coeff_monomial(s)), as_expr(pole_poly.coeff_monomial(1))
        if beta == 0:
            return None
        pole = as_expr(-beta / alpha)
        inside = _ask(as_boolean(pole > 0), assumptions) is not False and _ask(as_boolean(pole < upper), assumptions) is not False
        if inside:
            return None
        n = as_expr(cancel(-alpha / beta))
        c = as_expr(numerator_ / beta)
        if complete:
            total = total + 2 * c * elliptic_pi(n, m)
        else:
            total = total + 2 * c * elliptic_pi(n, phi, m)
    return as_expr(total)


# ---------------------------------------------------------------------------
# The entry point

def elliptic_integral(f: ExprLike, x: Symbol, a: ExprLike, b: ExprLike,
                      assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """``Integral(f, (x, a, b))`` for ``f = R(x) * P(x)**(+-1/2)``, ``P`` a
    cubic or a quartic with real roots, in Legendre's elliptic integrals;
    also the trigonometric forms ``Q(sin(x)**2) * (1 - m sin(x)**2)**(+-1/2)``
    over ``(0, phi)``. ``None`` when the integral is not of this kind, the
    roots are not all real or cannot be ordered, or the range crosses a
    root of ``P`` (cut it there first).

    Examples
    ========

    >>> from sympy import symbols, sqrt, oo, S, pi
    >>> from sympy_extras.integrals.elliptic import elliptic_integral
    >>> x = symbols('x')
    >>> elliptic_integral(1/sqrt(x*(1 - x)*(4 - x)), x, 0, 1)
    ConditionalValue(elliptic_k(1/4))
    >>> elliptic_integral(x/sqrt(x*(1 - x)*(4 - x)), x, 0, S.Half)
    ConditionalValue(-4*elliptic_e(pi/4, 1/4) + 4*elliptic_f(pi/4, 1/4))
    """
    f_, a_, b_ = as_expr(f), as_expr(a), as_expr(b)
    if f_.has(TrigonometricFunction):
        return _trigonometric(f_, x, a_, b_, assumptions)
    return _reduce(f_, x, a_, b_, assumptions, 0)


def _reduce(f: Expr, x: Symbol, a: Expr, b: Expr, assumptions: Assumptions, depth: int) -> Optional[ConditionalValue]:
    if depth > _MAX_DEPTH:
        return None
    constant, rest = f.as_independent(x, as_Add=False)
    constant_, rest_ = as_expr(constant), as_expr(rest)
    found = radicand(rest_, x)
    if found is None:
        return None
    R, P, e = found
    # an even radicand first: its own roots may be complex (1 - x**4)
    even = _even_reduction(R, P, e, x, a, b, assumptions, depth)
    if even is not None:
        return even.scaled(constant_)
    roots = real_roots(P, x, assumptions)
    if roots is None:
        return None
    lo_cell, hi_cell = _position(a, roots, assumptions), _position(b, roots, assumptions)
    if lo_cell is None or hi_cell is None:
        return None
    cell = lo_cell if lo_cell % 2 else lo_cell + 1
    if hi_cell not in (cell, cell + 1):
        return None
    n = len(roots)
    if n == 4 and cell in (-1, 2 * n - 1):
        inverted = _inverted_reduction(R, P, e, x, roots, a, b, cell, assumptions, depth)
        return None if inverted is None else inverted.scaled(constant_)
    reduction = _reduce_on_cell(R, P, e, x, roots, a, b, cell, assumptions)
    if reduction is None:
        return None
    value = legendre_reduction(reduction, assumptions)
    if value is None:
        return None
    return ConditionalValue(constant_ * value)


def _even_reduction(R: Expr, P: Expr, e: Expr, x: Symbol, a: Expr, b: Expr, assumptions: Assumptions,
                    depth: int) -> Optional[ConditionalValue]:
    """``y = x**2`` for an even radicand and an even ``R`` on a range of
    nonnegative numbers: ``R(x) P(x)^e dx = R(sqrt(y)) (y P(sqrt(y)))^e
    y^(-e-1/2) dy / 2``, the new radicand being a cubic."""
    if P.subs(x, -x) != P or R.subs(x, -x) != R:
        return None
    if a == -oo or _ask(as_boolean(a >= 0), assumptions) is not True:
        return None
    y = Dummy('y', positive=True)
    R_y = as_expr(cancel(R.subs(x, sqrt(y))))
    P_y = as_expr(cancel(P.subs(x, sqrt(y))))
    if not R_y.is_rational_function(y) or not P_y.is_polynomial(y):
        return None
    if e == -_HALF:
        # dx / sqrt(P) = dy / (2 sqrt(y P(sqrt y)))
        g = as_expr(R_y * (y * P_y)**e / 2)
    else:
        # sqrt(P) dx = sqrt(y P(sqrt y)) dy / (2 y)
        g = as_expr(R_y * (y * P_y)**e / (2 * y))
    return _reduce(g, y, as_expr(a**2), oo if b == oo else as_expr(b**2), assumptions, depth + 1)


def _inverted_reduction(R: Expr, P: Expr, e: Expr, x: Symbol, roots: Sequence[Expr], a: Expr, b: Expr,
                        cell: int, assumptions: Assumptions, depth: int) -> Optional[ConditionalValue]:
    """A quartic beyond its extreme root through ``u = 1/(x - r)``, which
    leaves a cubic in ``u`` over ``(0, oo)`` (or a part of it)."""
    u = Dummy('u', positive=True)
    if cell == 2 * len(roots) - 1:
        r = roots[-1]
        X = r + 1 / u
        # dx = -du/u**2, P(r + 1/u) = u**(-4) P_u(u)
        P_u = as_expr(cancel(P.subs(x, X) * u**4))
        R_u = as_expr(cancel(R.subs(x, X)))
        g = as_expr(R_u * P_u**e * u**(-4 * e - 2))
        # x from a to b means u from 1/(b - r) down to 1/(a - r): the sign of dx flips the bounds
        lower = S.Zero if b == oo else as_expr(1 / (b - r))
        upper = oo if a == r else as_expr(1 / (a - r))
    else:
        r = roots[0]
        X = r - 1 / u
        P_u = as_expr(cancel(P.subs(x, X) * u**4))
        R_u = as_expr(cancel(R.subs(x, X)))
        g = as_expr(R_u * P_u**e * u**(-4 * e - 2))
        lower = S.Zero if a == -oo else as_expr(1 / (r - a))
        upper = oo if b == r else as_expr(1 / (r - b))
    if not P_u.is_polynomial(u):
        return None
    return _reduce(g, u, lower, upper, assumptions, depth + 1)


def _trigonometric(f: Expr, x: Symbol, a: Expr, b: Expr, assumptions: Assumptions) -> Optional[ConditionalValue]:
    """``Q(sin(x)**2) (1 - m sin(x)**2)**(+-1/2)`` over ``(0, phi)`` through
    ``s = sin(x)**2``."""
    if a != 0 or _ask(as_boolean(b > 0), assumptions) is not True or _ask(as_boolean(b <= pi / 2), assumptions) is not True:
        return None
    s = Dummy('s', positive=True)
    h = as_expr(f.subs(x, asin(sqrt(s))) / 2)
    radical: Optional[Expr] = None
    exponent: Optional[Expr] = None
    for node in h.atoms(Pow):
        if node.exp in (_HALF, -_HALF) and node.has(s):
            base = as_expr(node.base)
            if radical is not None or not base.is_polynomial(s) or Poly(base, s).degree() != 1:
                return None
            radical, exponent = base, as_expr(node.exp)
    if radical is None or exponent is None:
        return None
    alpha = as_expr(radical.subs(s, 0))
    if _ask(as_boolean(alpha > 0), assumptions) is not True:
        return None
    m = as_expr(cancel(-(radical - alpha) / (alpha * s)))
    if m.has(s):
        return None
    Q = as_expr(cancel(h / radical**exponent))
    if not Q.is_rational_function(s):
        return None
    if exponent == -_HALF:
        Q = as_expr(Q / sqrt(alpha))
    else:
        Q = as_expr(Q * sqrt(alpha) * (1 - m * s))
    upper = S.One if b == pi / 2 else as_expr(sin(b)**2)
    value = legendre_reduction(Reduction(Q, m, s, S.Zero, upper), assumptions)
    return None if value is None else ConditionalValue(value)
