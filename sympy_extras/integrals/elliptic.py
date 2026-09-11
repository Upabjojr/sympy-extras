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

When two roots of `P` are complex, `P = \\lambda (a - x)(x - b)((x - p)^2 + q^2)`
on the cell `(b, a)` (or `P = \\lambda (x - a)((x - p)^2 + q^2)` beyond the
single real root of a cubic), the reduction of Byrd–Friedman 241 and
240 maps the cell onto `t = \\cos\\theta \\in (-1, 1)` through the bilinear
substitution

.. math::

    x = \\frac{aB + bA - t\\,(aB - bA)}{A + B + t\\,(A - B)}, \\qquad
    A^2 = (a - p)^2 + q^2, \\quad B^2 = (b - p)^2 + q^2,

(`x = a + A(1 - t)/(1 + t)` for the cubic), for which
`P(x)\\, dt^2 = \\lambda AB\\, X'(t)^2 (1 - t^2)(1 - m + m t^2)\\, dt^2` with
`m = ((a - b)^2 - (A - B)^2)/(4AB)` (`m = (A - a + p)/(2A)` for the
cubic): `dx/\\sqrt{P} = d\\theta/\\sqrt{\\lambda AB (1 - m\\sin^2\\theta)}`. The
rational prefactor `G(t)` left by `R` is split into its even and odd
parts in `t`: the even part is a rational function of
`s = \\sin^2\\theta = 1 - t^2` and gives the Legendre form above, the odd
part is `t\\, G_o(t^2)` and its integral in `s` is elementary (it
vanishes over the whole cell). A quartic with a complex pair beyond
its extreme real root is inverted first, `u = 1/(x - r)`. The split
mirrors a pole of `R` at `t_p` to `-t_p`: a range containing the
mirror image of a pole of the cell, but not the pole, is left alone
(``None``), the two halves being divergent there.

The organising principle behind the tables is Carlson's symmetric
integral `R_F(x, y, z) = \\frac12\\int_0^\\infty dt/\\sqrt{(t + x)(t + y)(t + z)}`
[Carlson]_, [DLMF]_ (19.25): the complete integral of `dx/\\sqrt{P}`
over a cell of the real roots is `2 R_F` of the three differences of
the other roots from the cell (DLMF 19.29.4–19.29.7), whatever their
reality, and Legendre's functions are its special values
`K(m) = R_F(0, 1 - m, 1)`, `F(\\phi \\mid m) = \\sin\\phi\\, R_F(\\cos^2\\phi, 1 - m\\sin^2\\phi, 1)`
(DLMF 19.25.1, 19.25.5). The substitutions above are the changes of
variable that bring the Carlson arguments to Legendre's; the results
are written with SymPy's ``elliptic_k``, ``elliptic_e``, ``elliptic_f``
and ``elliptic_pi``.

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
   (the substitutions), the tables 230–260 (240 and 241 for the complex
   roots) and 310–340.
.. [Carlson] B. C. Carlson, *Numerical computation of real or complex
   elliptic integrals*, Numerical Algorithms 10 (1995), 13–26;
   *A table of elliptic integrals of the third kind*, Mathematics of
   Computation 51 (1988), 267–280.
.. [DLMF] NIST Digital Library of Mathematical Functions, chapter 19,
   sections 19.2 (Legendre's integrals), 19.25 (Legendre's integrals
   as symmetric integrals) and 19.29 (reduction of
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
from sympy.core.relational import Eq
from sympy.functions.elementary.complexes import im, re
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.trigonometric import TrigonometricFunction, asin, sin
from sympy.functions.special.elliptic_integrals import elliptic_k, elliptic_e, elliptic_f, elliptic_pi
from sympy.polys.partfrac import apart
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import Poly, cancel, factor_list
from sympy.polys.rootoftools import ComplexRootOf
from sympy.solvers.solvers import solve as sympy_solve

from sympy_extras._timeout import attempt
from sympy_extras._typing import ExprLike, as_boolean, as_expr, free_symbols
from sympy.logic.boolalg import Boolean
from sympy_extras.assumptions.ask import Assumptions, ask
from sympy_extras.settings import settings
from .conditions import ConditionalValue, plain_symbols, with_symbol_facts

__all__ = ['elliptic_integral', 'radicand', 'real_roots', 'legendre_reduction', 'Reduction']


def _ask(query: Boolean, assumptions: Assumptions) -> Optional[bool]:
    """``ask`` with the flags of the symbols (``positive=True``) turned
    into statements on plain symbols, which the CAD can use, within a
    budget (``None`` when the time is up)."""
    symbols = free_symbols(query)
    plain = plain_symbols(symbols)
    budget = None if settings.timeout is None else settings.timeout / 4
    if not plain:
        return attempt(lambda: ask(query, assumptions), budget)
    return attempt(lambda: ask(as_boolean(query.xreplace(plain)), with_symbol_facts(assumptions, symbols, plain)), budget)


def _with_facts(assumptions: Assumptions, facts: Sequence[Boolean]) -> Assumptions:
    """The assumptions together with more facts."""
    if not facts:
        return assumptions
    combined: list[Boolean] = list(facts)
    if isinstance(assumptions, (Boolean, bool)):
        combined.append(as_boolean(assumptions))
    elif assumptions is not None:
        combined.extend(as_boolean(a) for a in assumptions)
    return combined


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
                if r_.is_extended_real is False:
                    return None
                # p + I q with real symbols: real only where q vanishes
                imaginary = as_expr(im(r_))
                if imaginary != 0 and not imaginary.has(im, re) \
                        and _ask(as_boolean(Eq(imaginary, 0)), assumptions) is not True:
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
    try:
        query = as_boolean(as_expr(numerator) * as_expr(denominator) > 0)
    except TypeError:
        # a comparison of numbers that are not real
        return None
    return _ask(query, assumptions)


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

class _Radical:
    """The antiderivatives of ``s**j/sqrt(W(s))`` and of
    ``1/((alpha s + beta) sqrt(W(s)))`` for one radicand ``W``, normalised
    at ``s = 0``: for Legendre's ``W = s (1 - s)(1 - m s)`` they are the
    elliptic integrals, for ``W = s (1 - m s)`` they are elementary."""

    def __init__(self, m: Expr) -> None:
        self.m = m

    def W(self, s: Expr) -> Expr:
        raise NotImplementedError

    def power(self, j: int, upper: Expr) -> Expr:
        raise NotImplementedError

    def simple_pole(self, c: Expr, alpha: Expr, beta: Expr, upper: Expr, assumptions: Assumptions) -> Optional[Expr]:
        raise NotImplementedError


class _Legendre(_Radical):
    """``W = s (1 - s)(1 - m s)``: ``ds/sqrt(W) = 2 dtheta/sqrt(1 - m sin(theta)**2)``
    with ``s = sin(theta)**2``."""

    def W(self, s: Expr) -> Expr:
        return as_expr(s * (1 - s) * (1 - self.m * s))

    def power(self, j: int, upper: Expr) -> Expr:
        """``Integral(s**j/sqrt(W), (s, 0, upper)) = 2 J_j`` with
        ``J_j = Integral(sin(theta)**(2 j)/sqrt(1 - m sin(theta)**2), (theta, 0, phi))``,
        ``sin(phi)**2 == upper``: ``J_0 = F``, ``J_1 = (F - E)/m`` and
        the recurrence of Byrd–Friedman 310–318, which integrating
        ``d/dtheta (sin(theta)**(2j-1) cos(theta) sqrt(1 - m sin(theta)**2))``
        gives with its boundary term,

            (2j + 1) m J_{j+1} = 2j (1 + m) J_j - (2j - 1) J_{j-1}
                                 + sin(phi)**(2j-1) cos(phi) sqrt(1 - m sin(phi)**2)

        (the term vanishes for the complete integrals, ``phi = pi/2``).
        For a negative ``j`` the recurrence is read downwards and the
        result is the antiderivative regularised at ``phi = 0``: the
        boundary terms ``s**(j-1) sqrt(s (1 - s)(1 - m s))``,
        ``s = sin(phi)**2``, are the half-integer powers of ``s`` that
        diverge there, so the value at a point is the integral from
        ``0`` up to those powers, and the difference of two values is
        the integral between the points."""
        m = self.m
        if upper == 1:
            F: Expr = as_expr(elliptic_k(m))
            E: Expr = as_expr(elliptic_e(m))
        else:
            phi = as_expr(asin(sqrt(upper)))
            F, E = as_expr(elliptic_f(phi, m)), as_expr(elliptic_e(phi, m))
        # sin(phi)**(2k-1) cos(phi) sqrt(1 - m sin(phi)**2) = s**(k-1) sqrt(W(s))
        boundary = as_expr(sqrt(self.W(upper)))
        values: dict[int, Expr] = {0: F, 1: as_expr((F - E) / m)}
        for k in range(1, j):
            values[k + 1] = as_expr((2 * k * (1 + m) * values[k] - (2 * k - 1) * values[k - 1]
                                     + upper**(k - 1) * boundary) / ((2 * k + 1) * m))
        for k in range(0, j, -1):
            values[k - 1] = as_expr((2 * k * (1 + m) * values[k] - (2 * k + 1) * m * values[k + 1]
                                     + upper**(k - 1) * boundary) / (2 * k - 1))
        return as_expr(2 * values[j])

    def simple_pole(self, c: Expr, alpha: Expr, beta: Expr, upper: Expr, assumptions: Assumptions) -> Optional[Expr]:
        """``c/(alpha s + beta) = (c/beta)/(1 - n s)`` with ``n = -alpha/beta``:
        the third kind, ``2 (c/beta) Pi(n; phi | m)``."""
        n = as_expr(cancel(-alpha / beta))
        if upper == 1:
            return as_expr(2 * c / beta * elliptic_pi(n, self.m))
        return as_expr(2 * c / beta * elliptic_pi(n, asin(sqrt(upper)), self.m))


class _Elementary(_Radical):
    """``W = s (1 - m s)``, the radicand left by the odd part of the
    cosine map: the integrals are elementary."""

    def W(self, s: Expr) -> Expr:
        return as_expr(s * (1 - self.m * s))

    def power(self, j: int, upper: Expr) -> Expr:
        """``I_j = Integral(s**j/sqrt(s (1 - m s)), (s, 0, u))``:
        ``I_0 = 2 asin(sqrt(m u))/sqrt(m)`` and, from
        ``d/ds (s**j sqrt(W)) = ((j + 1/2) s**j - (j + 1) m s**(j+1))/sqrt(W)``,
        ``(j + 1) m I_{j+1} = (j + 1/2) I_j - u**j sqrt(W(u))``, read
        downwards for the negative powers (regularised at ``0`` as in
        :meth:`_Legendre.power`)."""
        m = self.m
        boundary = as_expr(sqrt(self.W(upper)))
        values: dict[int, Expr] = {0: as_expr(2 * asin(sqrt(m * upper)) / sqrt(m))}
        for k in range(0, j):
            values[k + 1] = as_expr(((k + _HALF) * values[k] - upper**k * boundary) / ((k + 1) * m))
        for k in range(-1, j - 1, -1):
            values[k] = as_expr(((k + 1) * m * values[k + 1] + upper**k * boundary) / (k + _HALF))
        return values[j]

    def simple_pole(self, c: Expr, alpha: Expr, beta: Expr, upper: Expr, assumptions: Assumptions) -> Optional[Expr]:
        """``Integral(c/((alpha s + beta) sqrt(s (1 - m s))), (s, 0, u))``
        through ``t = 1/(s - p)``, ``p = -beta/alpha`` (outside the range):
        ``ds/((s - p) sqrt(W)) = -sign(s - p) dt/sqrt(A t**2 + B t + C)`` with
        ``A = W(p)``, ``B = W'(p)``, ``C = -m`` and ``B**2 - 4 A C = 1``, a
        logarithm when ``A > 0`` (``0 < p < 1/m``) and an arcsine when
        ``A < 0``. The value is an antiderivative continued through the
        pole (the sign of ``s - p`` is that at the point, the constant
        that at ``0``): it is used in differences of points on the same
        side."""
        m = self.m
        p = as_expr(cancel(-beta / alpha))
        A = as_expr(cancel(self.W(p)))
        B = as_expr(1 - 2 * m * p)
        t = Dummy('t')
        quadratic = A * t**2 + B * t - m
        # sign(s - p) at the point
        far = _ask(as_boolean(upper > p), assumptions)
        if far is None:
            return None
        sigma: Expr = S.One if far else S.NegativeOne
        if _ask(as_boolean(A > 0), assumptions) is True:
            # 0 < p < 1/m: (2 A t + B)**2 - 4 A (A t**2 + B t + C) = 1 makes
            # log|2 A t + B + 2 sqrt(A) sqrt(...)| the primitive, and 2 A t + B
            # has the sign of s - p
            at_upper = as_expr(log(sigma * (2 * A * t + B + 2 * sqrt(A) * sqrt(quadratic))) / sqrt(A))
            at_zero = as_expr(log(-(2 * A * t + B + 2 * sqrt(A) * sqrt(quadratic))) / sqrt(A))
        elif _ask(as_boolean(A < 0), assumptions) is True:
            at_upper = as_expr(-asin(2 * A * t + B) / sqrt(-A))
            at_zero = at_upper
        else:
            return None
        value = at_upper.subs(t, 1 / (upper - p)) - at_zero.subs(t, -1 / p)
        return as_expr(-sigma * c / alpha * value)


def legendre_reduction(reduction: Reduction, assumptions: Assumptions = None) -> Optional[Expr]:
    """``Integral(Q(s)/sqrt(s*(1 - s)*(1 - m*s)), (s, lower, upper))`` in
    Legendre's integrals, from the partial fractions of ``Q``; ``None``
    for a divergent integral (a pole of ``Q`` inside the range or at an
    endpoint that is a root of the radicand).

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
    lower, upper = as_expr(reduction.lower), as_expr(reduction.upper)
    if lower == 0 and _pole_at_zero(Q, s):
        return None
    total: Expr = S.Zero
    radical = _Legendre(m)
    for bound, sign in ((upper, S.One), (lower, S.NegativeOne)):
        value = _from_zero(Q, s, bound, radical, assumptions, lower, upper)
        if value is None:
            return None
        total = total + sign * value
    return as_expr(total)


def _pole_at_zero(Q: Expr, s: Symbol) -> bool:
    _, denominator = as_expr(cancel(Q)).as_numer_denom()
    return as_expr(denominator).subs(s, 0) == 0


def _from_zero(Q: Expr, s: Symbol, point: Expr, radical: _Radical, assumptions: Assumptions,
               lower: Expr, upper: Expr) -> Optional[Expr]:
    """The antiderivative of ``Q(s)/sqrt(W(s))`` at ``s = point``,
    normalised to vanish at ``s = 0``: the integral from ``0`` when it
    converges, and with a pole of ``Q`` at ``0`` the regularised value
    of :meth:`_Radical.power`. The value is used in the difference of
    two points, ``lower`` and ``upper``: ``None`` when a pole of ``Q``
    other than ``0`` may lie between them (a divergent integral); a
    pole between ``0`` and ``lower`` is allowed, the antiderivatives
    continue through it with the same constant on the far side."""
    if point == 0:
        return S.Zero
    try:
        parts = apart(Q, s)
    except (PolynomialError, NotImplementedError):
        return None
    total: Expr = S.Zero
    for term in Add.make_args(parts):
        value = _term_from_zero(as_expr(term), s, point, radical, assumptions, lower, upper)
        if value is None:
            return None
        total = total + value
    return as_expr(total)


def _term_from_zero(term: Expr, s: Symbol, upper: Expr, radical: _Radical, assumptions: Assumptions,
                    range_lower: Expr, range_upper: Expr) -> Optional[Expr]:
    """One partial fraction: a power ``c s**j``, a simple pole
    ``c/(alpha s + beta)`` or a multiple pole, reduced to lower ones by
    integrating ``d/ds (sqrt(W)/D**(k-1))``, ``D = alpha s + beta``:
    ``Integral(N/(D**k sqrt(W)), (s, 0, u)) = 2 sqrt(W(u))/D(u)**(k-1)``
    with ``N = W' D - 2 (k - 1) alpha W``, whose leading partial fraction
    is ``N(p)/D**k`` at the pole ``p``."""
    numerator, denominator = term.as_numer_denom()
    numerator_, denominator_ = as_expr(numerator), as_expr(denominator)
    if not denominator_.has(s):
        total: Expr = S.Zero
        for (j,), c in Poly(term, s).terms():
            total = total + as_expr(c) * radical.power(j, upper)
        return total
    pole_poly = Poly(denominator_, s)
    _, factors = factor_list(denominator_, s)
    if len(factors) != 1:
        return None
    linear, k = factors[0]
    linear_poly = Poly(linear, s)
    if linear_poly.degree() != 1:
        return None
    alpha, beta = as_expr(linear_poly.coeff_monomial(s)), as_expr(linear_poly.coeff_monomial(1))
    scale = as_expr(cancel(pole_poly.LC() / alpha**k))
    if numerator_.has(s):
        # N(s)/D**k with symbolic coefficients, which apart leaves whole:
        # N in powers of D gives the terms c_j/D**(k - j)
        u = Dummy('u')
        expansion = Poly(as_expr((numerator_ / scale).subs(s, (u - beta) / alpha)).expand(), u)
        total = S.Zero
        for (j,), c in expansion.terms():
            if j >= k:
                part: Optional[Expr] = _term_from_zero(as_expr(c * linear**(j - k)), s, upper, radical, assumptions,
                                                       range_lower, range_upper)
            else:
                part = _pole_from_zero(as_expr(c), alpha, beta, k - j, s, upper, radical, assumptions,
                                       range_lower, range_upper)
            if part is None:
                return None
            total = total + part
        return as_expr(total)
    return _pole_from_zero(as_expr(numerator_ / scale), alpha, beta, k, s, upper, radical, assumptions,
                           range_lower, range_upper)


def _pole_from_zero(c: Expr, alpha: Expr, beta: Expr, k: int, s: Symbol, upper: Expr, radical: _Radical,
                    assumptions: Assumptions, range_lower: Expr, range_upper: Expr) -> Optional[Expr]:
    """``c/(alpha s + beta)**k`` from :func:`_term_from_zero`."""
    if beta == 0:
        # c/(alpha s)**k: the negative powers, regularised at s = 0
        return as_expr(c / alpha**k * radical.power(-k, upper))
    # no zero of D on [lower, upper]: D(lower) D(upper) > 0, a question
    # without division (alpha may vanish for symbolic parameters, when
    # the pole is at infinity)
    if _ask(as_boolean((alpha * range_lower + beta) * (alpha * range_upper + beta) > 0), assumptions) is not True:
        return None
    pole = as_expr(-beta / alpha)
    if k == 1:
        return radical.simple_pole(c, alpha, beta, upper, assumptions)
    W = radical.W(as_expr(s))
    D = as_expr(alpha * s + beta)
    N = as_expr(W.diff(s) * D - 2 * (k - 1) * alpha * W)
    leading = as_expr(cancel(N.subs(s, pole)))
    if leading == 0:
        return None
    rest = as_expr(cancel((N - leading) / D**k))
    lower_orders = _from_zero(rest, s, upper, radical, assumptions, range_lower, range_upper)
    if lower_orders is None:
        return None
    boundary = as_expr(2 * sqrt(radical.W(upper)) / D.subs(s, upper)**(k - 1))
    return as_expr(c / leading * (boundary - lower_orders))


# ---------------------------------------------------------------------------
# The entry point

def elliptic_integral(f: ExprLike, x: Symbol, a: ExprLike, b: ExprLike,
                      assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """``Integral(f, (x, a, b))`` for ``f = R(x) * P(x)**(+-1/2)``, ``P`` a
    cubic or a quartic with real roots, or with one pair of complex
    roots, in Legendre's elliptic integrals; also the trigonometric
    forms ``Q(sin(x)**2) * (1 - m sin(x)**2)**(+-1/2)`` over ``(0, phi)``.
    ``None`` when the integral is not of this kind, the roots cannot be
    ordered (or the complex pair is not a rational quadratic factor),
    the range crosses a root of ``P`` (cut it there first) or the
    integral diverges at a pole of ``R``.

    Examples
    ========

    >>> from sympy import symbols, sqrt, oo, S, pi
    >>> from sympy_extras.integrals.elliptic import elliptic_integral
    >>> x = symbols('x')
    >>> elliptic_integral(1/sqrt(x*(1 - x)*(4 - x)), x, 0, 1)
    ConditionalValue(elliptic_k(1/4))
    >>> elliptic_integral(x/sqrt(x*(1 - x)*(4 - x)), x, 0, S.Half)
    ConditionalValue(-4*elliptic_e(pi/4, 1/4) + 4*elliptic_f(pi/4, 1/4))
    >>> elliptic_integral(1/sqrt(x**3 + 1), x, 0, oo)
    ConditionalValue(-3**(3/4)*elliptic_f(asin(sqrt(2)*3**(1/4)/sqrt(sqrt(3) + 2)), sqrt(3)/4 + 1/2)/3 + 2*3**(3/4)*elliptic_k(sqrt(3)/4 + 1/2)/3)
    >>> elliptic_integral(1/sqrt((x**2 + 1)*(x + 2)*(3 - x)), x, -2, 3)
    ConditionalValue(2**(3/4)*sqrt(5)*elliptic_k(sqrt(2)/4 + 1/2)/5)
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
        complex_ = _complex_reduction(R, P, e, x, a, b, assumptions, depth)
        return None if complex_ is None else complex_.scaled(constant_)
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


# ---------------------------------------------------------------------------
# Two complex roots

class _ComplexFactors:
    """``P = lam * prod(x - r) * ((x - p)**2 + q2)`` with one or two real
    roots ``r`` in increasing order and ``q2 > 0``."""

    def __init__(self, lam: Expr, roots: list[Expr], p: Expr, q2: Expr) -> None:
        self.lam = lam
        self.roots = roots
        self.p = p
        self.q2 = q2


def _complex_factors(P: Expr, x: Symbol, assumptions: Assumptions) -> Optional[_ComplexFactors]:
    """The factors of a cubic or a quartic with exactly one pair of
    complex roots, read from ``factor_list`` (a numeric polynomial whose
    quadratic factor is not rational is not recognised)."""
    try:
        coefficient, factors = factor_list(P, x)
    except PolynomialError:
        return None
    lam: Expr = as_expr(coefficient)
    roots: list[Expr] = []
    quadratic: Optional[tuple[Expr, Expr]] = None
    for factor, multiplicity in factors:
        factor_ = as_expr(factor)
        if not factor_.has(x):
            lam = lam * factor_**multiplicity
            continue
        if multiplicity != 1:
            return None
        poly = Poly(factor_, x)
        degree = poly.degree()
        lead = as_expr(poly.LC())
        if degree == 1:
            roots.append(as_expr(cancel(-poly.coeff_monomial(1) / lead)))
        elif degree == 2 and quadratic is None:
            c1, c0 = as_expr(poly.coeff_monomial(x) / lead), as_expr(poly.coeff_monomial(1) / lead)
            quadratic = (as_expr(cancel(c1)), as_expr(cancel(c0)))
        else:
            return None
        lam = lam * lead
    if quadratic is None or len(roots) not in (1, 2):
        return None
    c1, c0 = quadratic
    p = as_expr(-c1 / 2)
    q2 = as_expr(cancel(c0 - p**2))
    if _ask(as_boolean(q2 > 0), assumptions) is not True:
        return None
    if len(roots) == 2:
        below = _less(roots[0], roots[1], assumptions)
        if below is None:
            return None
        if not below:
            roots.reverse()
    return _ComplexFactors(as_expr(cancel(lam)), roots, p, q2)


class _CosineMap:
    """The cell mapped onto ``t = cos(theta)`` in ``(-1, 1)``: ``x = X(t)``,
    ``P(X(t)) = lam * X'(t)**2 * (1 - t**2) * (1 - m + m t**2) / g2``
    with ``sigma`` the sign of ``X'`` on the range and ``preimage`` the
    inverse map (``-1`` at the infinite end of a half-line). Symbolic
    radicals ``A = sqrt((a - p)**2 + q**2)`` are positive symbols in
    the map, so that the sign questions are polynomial: the values are
    restored in the result."""

    def __init__(self, X: Expr, m: Expr, g2: Expr, sigma: Expr, t: Symbol, preimage: Expr,
                 infinite: Optional[Expr], radicals: dict[Symbol, Expr], facts: list[Boolean]) -> None:
        self.X = X
        self.m = m
        self.g2 = g2
        self.sigma = sigma
        self.t = t
        self._preimage = preimage
        self._infinite = infinite
        #: the symbols standing for the radicals A, B of symbolic
        #: parameters, and what is known about them (the triangle
        #: inequalities of the distances to the complex root)
        self.radicals = radicals
        self.facts = facts

    def preimage(self, y: Expr, x: Symbol) -> Expr:
        if y == self._infinite:
            return S.NegativeOne
        return as_expr(cancel(self._preimage.subs(x, y)))


def _cosine_map(factors: _ComplexFactors, cell: int, t: Symbol, x: Symbol) -> Optional[_CosineMap]:
    """Byrd–Friedman 241 for the cell between the real roots of a quartic
    and 240 for the half-lines of a cubic."""
    p, q2 = factors.p, factors.q2
    radicals: dict[Symbol, Expr] = {}
    facts: list[Boolean] = []
    if len(factors.roots) == 2:
        if cell != 1:
            return None
        b, a = factors.roots
        A = _radical(as_expr((a - p)**2 + q2), radicals)
        B = _radical(as_expr((b - p)**2 + q2), radicals)
        if radicals:
            # the sides a - b, A, B of the triangle with the complex root
            facts.extend([as_boolean((a - b)**2 > (A - B)**2), as_boolean((a - b)**2 < (A + B)**2)])
        X = as_expr((a * B + b * A - t * (a * B - b * A)) / (A + B + t * (A - B)))
        m = as_expr(cancel(((a - b)**2 - (A - B)**2) / (4 * A * B)))
        preimage = as_expr((a * B + b * A - x * (A + B)) / (a * B - b * A + x * (A - B)))
        return _CosineMap(X, m, as_expr(1 / (A * B)), S.NegativeOne, t, preimage, None, radicals, facts)
    a = factors.roots[0]
    A = _radical(as_expr((a - p)**2 + q2), radicals)
    if radicals:
        facts.append(as_boolean(A**2 > (a - p)**2))
    if cell == 1:
        # (a, oo): x = a + A (1 - t)/(1 + t), decreasing from oo to a
        r = (x - a) / A
        return _CosineMap(as_expr(a + A * (1 - t) / (1 + t)), as_expr(cancel((A - (a - p)) / (2 * A))), as_expr(1 / A),
                          S.NegativeOne, t, as_expr((1 - r) / (1 + r)), oo, radicals, facts)
    if cell == -1:
        # (-oo, a): x = a - A (1 - t)/(1 + t), increasing from -oo to a
        r = (a - x) / A
        return _CosineMap(as_expr(a - A * (1 - t) / (1 + t)), as_expr(cancel((A + (a - p)) / (2 * A))), as_expr(1 / A),
                          S.One, t, as_expr((1 - r) / (1 + r)), -oo, radicals, facts)
    return None


def _radical(square: Expr, radicals: dict[Symbol, Expr]) -> Expr:
    """``sqrt(square)``: a positive symbol recorded in ``radicals`` when
    the square has symbols."""
    if not square.free_symbols:
        return sqrt(square)
    symbol = Dummy('A', positive=True)
    radicals[symbol] = sqrt(square)
    return symbol


def _parity_parts(G: Expr, t: Symbol, w: Symbol) -> Optional[tuple[Expr, Expr]]:
    """``(G_e, G_o)`` with ``G(t) == G_e(t**2) + t G_o(t**2)``."""
    numerator, denominator = as_expr(cancel(G)).as_numer_denom()
    numerator_, denominator_ = as_expr(numerator), as_expr(denominator)
    if not numerator_.is_polynomial(t) or not denominator_.is_polynomial(t):
        return None
    mirrored = as_expr(denominator_.subs(t, -t))
    numerator_ = as_expr((numerator_ * mirrored).expand())
    denominator_ = as_expr((denominator_ * mirrored).expand())
    even = as_expr(((numerator_ + numerator_.subs(t, -t)) / 2).expand())
    odd = as_expr(cancel((numerator_ - numerator_.subs(t, -t)) / (2 * t)))
    parts: list[Expr] = []
    for polynomial in (even, odd, denominator_):
        converted: Expr = S.Zero
        for (k,), c in Poly(polynomial, t).terms():
            if k % 2:
                return None
            converted = converted + as_expr(c) * w**(k // 2)
        parts.append(converted)
    return (as_expr(cancel(parts[0] / parts[2])), as_expr(cancel(parts[1] / parts[2])))


def _complex_reduction(R: Expr, P: Expr, e: Expr, x: Symbol, a: Expr, b: Expr, assumptions: Assumptions,
                       depth: int) -> Optional[ConditionalValue]:
    """The integral over a part of a cell of a radicand with two complex
    roots, through the cosine map: the even part of the rational
    prefactor in ``s = 1 - t**2`` is a Legendre form, the odd part is
    elementary in ``s``."""
    factors = _complex_factors(P, x, assumptions)
    if factors is None:
        return None
    roots = factors.roots
    lo_cell, hi_cell = _position(a, roots, assumptions), _position(b, roots, assumptions)
    if lo_cell is None or hi_cell is None:
        return None
    cell = lo_cell if lo_cell % 2 else lo_cell + 1
    if hi_cell not in (cell, cell + 1):
        return None
    if len(roots) == 2 and cell in (-1, 3):
        return _inverted_reduction(R, P, e, x, roots, a, b, cell, assumptions, depth)
    # the sign of P on the cell: lam (x - r1)(x - r2) q(x) between the
    # roots, lam (x - a) q(x) beyond the root of a cubic
    lam = factors.lam if cell == 2 * len(roots) - 1 else as_expr(-factors.lam)
    if _ask(as_boolean(lam > 0), assumptions) is not True:
        return None
    t = Dummy('t')
    mapping = _cosine_map(factors, cell, t, x)
    if mapping is None:
        return None
    X, m = mapping.X, mapping.m
    assumptions = _with_facts(assumptions, mapping.facts)
    t0, t1 = mapping.preimage(a, x), mapping.preimage(b, x)
    # P(X)^e = (lam/g2)^e (sigma X')^(2e) ((1 - t^2)(1 - m + m t^2))^e, the
    # rest of the integrand is R(X) X' dt
    Xp = as_expr(X.diff(t))
    root = sqrt(lam / mapping.g2)
    if e == -_HALF:
        G = as_expr(R.subs(x, X) * Xp / (root * mapping.sigma * Xp))
    else:
        G = as_expr(R.subs(x, X) * Xp * root * mapping.sigma * Xp * (1 - t**2) * (1 - m + m * t**2))
    G = as_expr(cancel(G))
    if not G.is_rational_function(t):
        return None
    parts = _CosineParts(G, t, m, assumptions)
    value = parts.between(t0, t1)
    if value is None:
        return None
    return ConditionalValue(as_expr(value.xreplace(mapping.radicals)))


class _CosineParts:
    """``Integral(G(t)/sqrt((1 - t**2)(1 - m + m t**2)), (t, t0, t1))`` for
    ``-1 <= t0 < t1 <= 1`` from the even and the odd part of ``G``,
    ``G(t) = G_e(t**2) + t G_o(t**2)``, in ``s = 1 - t**2``: on a range of
    one sign ``sigma``, with ``u = |t|`` running from ``u0`` to ``u1``,

        sigma (F_e(1 - u0**2) - F_e(1 - u1**2)) + F_o(1 - u0**2) - F_o(1 - u1**2)

    with ``F_e`` the antiderivative of ``G_e(1 - s)/(2 sqrt(s (1 - s)(1 - m s)))``
    and ``F_o`` that of ``G_o(1 - s)/(2 sqrt(s (1 - m s)))``, elementary,
    both from :func:`_from_zero`; a range of both signs is cut at
    ``t = 0``. When ``G`` has a pole at ``t = -1`` (the image of the
    infinite end of a half-line) both parts diverge at ``s = 0`` and
    only their sum is finite at the other end ``t = 1``: the values at
    ``0`` are then the regularised ones, which drop the same
    half-integer powers of ``s`` from both. At the end where ``G``
    itself has the pole the integral diverges."""

    def __init__(self, G: Expr, t: Symbol, m: Expr, assumptions: Assumptions) -> None:
        self.m = m
        self.assumptions = assumptions
        self.s = Dummy('s', positive=True)
        w = Dummy('w', positive=True)
        parts = _parity_parts(G, t, w)
        self.valid = parts is not None
        if parts is None:
            self.Q_e: Expr = S.Zero
            self.Q_o: Expr = S.Zero
        else:
            self.Q_e = as_expr(cancel(parts[0].subs(w, 1 - self.s) / 2))
            self.Q_o = as_expr(cancel(parts[1].subs(w, 1 - self.s) / 2))
        _, denominator = as_expr(cancel(G)).as_numer_denom()
        #: the ends of the cell where G has a pole, and the integral diverges
        self.poles = [end for end in (S.One, S.NegativeOne) if as_expr(denominator).subs(t, end) == 0]

    def between(self, t0: Expr, t1: Expr) -> Optional[Expr]:
        if not self.valid or t0 in self.poles or t1 in self.poles:
            return None
        if {t0, t1} == {S.One, S.NegativeOne}:
            # the whole cell: twice the complete integral of the even part
            value = self._piece(S.Zero, S.One, S.One, even_only=True)
            return None if value is None else as_expr(2 * t1 * value)
        signs = [self._sign(t0), self._sign(t1)]
        if None in signs:
            return None
        if signs[0] == 0 or signs[1] == 0 or signs[0] == signs[1]:
            sigma = S.One if S.One in signs else S.NegativeOne
            return self._piece(as_expr(sigma * t0), as_expr(sigma * t1), sigma)
        # a range of both signs, cut at t = 0
        first = self._piece(as_expr(signs[0] * t0), S.Zero, as_expr(signs[0]))
        second = self._piece(S.Zero, as_expr(signs[1] * t1), as_expr(signs[1]))
        if first is None or second is None:
            return None
        return as_expr(first + second)

    def _sign(self, tau: Expr) -> Optional[Expr]:
        if tau == 0:
            return S.Zero
        if _ask(as_boolean(tau > 0), self.assumptions) is True:
            return S.One
        if _ask(as_boolean(tau < 0), self.assumptions) is True:
            return S.NegativeOne
        return None

    def _piece(self, u0: Expr, u1: Expr, sigma: Expr, even_only: bool = False) -> Optional[Expr]:
        """The integral over ``t`` from ``sigma u0`` to ``sigma u1``, with
        ``u0, u1`` in ``[0, 1]`` (of the even part alone when asked)."""
        s0, s1 = as_expr(cancel(1 - u0**2)), as_expr(cancel(1 - u1**2))
        if s0 == s1:
            return S.Zero
        increasing = _ask(as_boolean(s0 < s1), self.assumptions)
        if increasing is None:
            return None
        lower, upper = (s0, s1) if increasing else (s1, s0)
        total: Expr = S.Zero
        for Q, radical, weight in ((self.Q_e, _Legendre(self.m), sigma), (self.Q_o, _Elementary(self.m), S.One)):
            if Q == 0 or (even_only and radical.W(S.Half) != _Legendre(self.m).W(S.Half)):
                continue
            at_s0 = _from_zero(Q, self.s, s0, radical, self.assumptions, lower, upper)
            at_s1 = _from_zero(Q, self.s, s1, radical, self.assumptions, lower, upper)
            if at_s0 is None or at_s1 is None:
                return None
            total = total + weight * (at_s0 - at_s1)
        return as_expr(total)


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
