"""Definite integrals by series expansion and termwise integration.

Euler's method for integrals such as

.. math::

    \\int_0^1 \\frac{\\log(1 - x)}{x}\\, dx
    = -\\sum_{k \\ge 1} \\frac{1}{k} \\int_0^1 x^{k-1}\\, dx
    = -\\sum_{k \\ge 1} \\frac{1}{k^2} = -\\frac{\\pi^2}{6}:

one factor of the integrand is expanded in a series whose terms
integrate against the rest in closed form, and the resulting series is
summed in closed form. The steps are

1. the factor `g` is expanded: a power series `g(x) = x^r \\sum_j c(j)
   x^{m j}` with a closed-form coefficient (from SymPy's formal power
   series, through :func:`~sympy_extras.integrals.brackets.taylor_coefficient`),
   or a geometric series in exponentials, `1/(e^{cx} - 1) = \\sum_{k \\ge 1}
   e^{-kcx}` and `1/(e^{cx} + 1) = \\sum_{k \\ge 1} (-1)^{k-1} e^{-kcx}` on
   `(0, \\infty)`, or a Fourier series `g(x) = c_0 + \\sum_{k \\ge 1} c(k)
   \\cos(\\omega(k) x)` (or in sines) from the table of
   :func:`fourier_expansion`: the classical series `\\log(2 \\sin(x/2)) =
   -\\sum_{k \\ge 1} \\cos(k x)/k` on `(0, 2\\pi)`, `\\log(2 \\cos(x/2)) =
   \\sum_{k \\ge 1} (-1)^{k+1} \\cos(k x)/k` on `(-\\pi, \\pi)`, `x = \\pi - 2
   \\sum_{k \\ge 1} \\sin(k x)/k` on `(0, 2\\pi)`, the square wave, `|\\sin x|`,
   ... ([GR]_ 1.44), and their rescalings;
2. the moments `\\int_a^b x^{mj + r} h(x)\\, dx` of the rest `h` are computed
   once for a symbolic nonnegative integer `j` by
   :func:`~sympy_extras.integrals.definite_integral` (the Mellin table gives
   them for `e^{-x}`, `(1 - x)^b`, `\\log^n x`, `1/(1 + x)`, ...);
3. the series `\\sum_j c(j) M(j)` is summed by :func:`sympy.summation`, by
   the summation algorithms of :mod:`sympy_extras.concrete`
   (:func:`~sympy_extras.concrete.polygamma_series`,
   :func:`~sympy_extras.concrete.zeilberger_sum`) or as a hypergeometric
   series expanded by ``hyperexpand``;
4. the interchange of the sum and the integral is justified when the
   series of the absolute values of the integrated terms converges
   (:func:`~sympy_extras.concrete.sum_convergence`, dominated convergence);
   otherwise the value is kept only when the numerical check of
   ``settings.numerical_checks`` confirms it.

For the Fourier series the integral of a product of two expanded factors
is reduced by the orthogonality `\\int_0^{2\\pi} \\cos(j x) \\cos(k x)\\, dx =
\\pi \\delta_{jk}` (Parseval's theorem, which justifies the interchange
for square-integrable factors) to a single series, so that

.. math::

    \\int_0^\\pi \\log^2(\\sin x)\\, dx
    = \\pi \\log^2 2 + \\frac{\\pi}{2} \\sum_{k \\ge 1} \\frac{1}{k^2}
    = \\pi \\log^2 2 + \\frac{\\pi^3}{12},

and a Fourier series integrated against `\\cos(n x)` picks the `n`-th
coefficient. A series `\\sum_k \\sin(k \\theta)/k^s` that no summation
algorithm closes is written with polylogarithms (`\\sum_{k \\ge 1} \\sin(k
\\theta)/k^2` is the Clausen function `\\mathrm{Cl}_2(\\theta)`).

The integrals of this kind in the tables [GR]_ (sections 4.22–4.27 and
4.29, the logarithmic and Euler-sum integrals) and in [Borwein]_ are the
test cases.

Examples
========

>>> from sympy import symbols, log, exp, atan, oo
>>> from sympy_extras.integrals.series import series_integral
>>> x = symbols('x')
>>> series_integral(log(1 - x)/x, x, 0, 1)
ConditionalValue(-pi**2/6)
>>> series_integral(x/(exp(x) - 1), x, 0, oo)
ConditionalValue(pi**2/6)
>>> series_integral(atan(x)/x, x, 0, 1)
ConditionalValue(Catalan)
>>> from sympy import sin, pi
>>> series_integral(log(sin(x)), x, 0, pi)
ConditionalValue(-pi*log(2))
>>> series_integral(log(sin(x))**2, x, 0, pi).value.expand()
pi*log(2)**2 + pi**3/12

References
==========

.. [GR] I. S. Gradshteyn, I. M. Ryzhik, *Table of Integrals, Series, and
   Products*, 7th ed., Academic Press, 2007, sections 4.22–4.29.
.. [Borwein] J. M. Borwein, D. M. Bradley, R. E. Crandall, *Computational
   strategies for the Riemann zeta function*, Journal of Computational
   and Applied Mathematics 121 (2000) 247–296.
.. [BorweinStraub] J. M. Borwein, A. Straub, *Log-sine evaluations of
   Mahler measures*, Journal of the Australian Mathematical Society 92
   (2012) 15–36 (the log-sine integrals through Fourier series).
.. [Zygmund] A. Zygmund, *Trigonometric Series*, 3rd ed., Cambridge
   University Press, 2002, chapter I (the classical series of section 2.8
   and the termwise integration theorem of section 2.7).
"""
from __future__ import annotations

from math import gcd
from typing import Callable, Iterator, Optional

from sympy.concrete.summations import Sum, summation
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.function import expand_mul
from sympy.core.numbers import Integer, Rational, nan, oo, pi, zoo
from sympy.core.relational import Ne
from sympy.functions.combinatorial.factorials import factorial
from sympy.series.limits import limit
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.elementary.complexes import Abs, sign
from sympy.functions.elementary.exponential import exp, exp_polar, log
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.trigonometric import cos, sin, tan
from sympy.functions.special.hyper import hyper
from sympy.functions.special.zeta_functions import dirichlet_eta, zeta, lerchphi, polylog
from sympy.functions.special.gamma_functions import polygamma
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import And, Boolean, true
from sympy.functions.elementary.piecewise import Piecewise, piecewise_fold
from sympy.simplify.hyperexpand import hyperexpand
from sympy.simplify.simplify import hypersimp, simplify
from sympy.polys.partfrac import apart
from sympy.polys.polytools import cancel

from sympy_extras._timeout import attempt
from sympy_extras._typing import ExprLike, as_boolean, as_expr
from sympy_extras.assumptions.ask import Assumptions, ask
from sympy_extras.assumptions.facts import element
from sympy_extras.concrete.convergence import sum_convergence
from sympy_extras.concrete.eulersums import polygamma_series
from sympy_extras.concrete.zeilberger import zeilberger_sum
from sympy_extras.settings import settings
from .brackets import SeriesCoefficient, taylor_coefficient
from .conditions import ConditionalValue
from .marichev import tidy
from .mellin import monomial

__all__ = ['Expansion', 'expansions', 'fourier_expansion', 'moments', 'sum_series', 'fourier_integral',
           'series_integral']


class Expansion:
    """A factor of the integrand expanded as a series: the factor is
    ``constant + Sum(coefficient(j) * basis(j), (j, start, oo))``, with
    ``basis(j)`` a power ``x**(period*j + offset)``, an exponential
    ``exp(-scale*j*x)`` or a harmonic ``cos(omega(j)*x)``, ``sin(omega(j)*x)``
    (or a combination of the two) of a Fourier series.

    Attributes
    ==========

    factor : Expr
        The factor expanded.
    coefficient : Expr
        The coefficient as a function of the index.
    index : Dummy
        The index (a nonnegative integer).
    basis : Expr
        The function of ``x`` and the index the coefficient multiplies.
    start : int
        The first index of the series.
    constant : Expr
        The term of the series outside the sum (the mean of a Fourier
        series); zero by default.
    condition : Boolean
        The condition on the parameters under which the expansion holds
        (``a**2 < 1`` for the series of the Poisson kernel); true by
        default.
    """

    def __init__(self, factor: Expr, coefficient: Expr, index: Dummy, basis: Expr, start: int,
                 constant: Expr = S.Zero, condition: Boolean = true) -> None:
        self.factor = factor
        self.coefficient = coefficient
        self.index = index
        self.basis = basis
        self.start = start
        self.constant = constant
        self.condition = condition

    def __repr__(self) -> str:
        if self.constant != 0:
            return "Expansion(%s, %s, %s, start=%d, constant=%s)" % (
                self.factor, self.coefficient, self.basis, self.start, self.constant)
        return "Expansion(%s, %s, %s, start=%d)" % (self.factor, self.coefficient, self.basis, self.start)


def _first_finite(coefficient: Expr, j: Dummy) -> Optional[int]:
    """The first index from which the coefficient formula is finite (the
    formal power series of ``log(1 - x)`` is ``-x**k/k`` from ``k = 1``,
    and the formula is infinite at ``k = 0``, where there is no term)."""
    for start in range(0, 6):
        values = [as_expr(coefficient.subs(j, start + i)) for i in range(6)]
        if all(not v.has(zoo, nan, oo, -oo) for v in values):
            return start
    return None


def _power_expansion(g: Expr, x: Symbol) -> Optional[Expansion]:
    """The power series of ``g`` about 0 as an :class:`Expansion`."""
    series: Optional[SeriesCoefficient] = taylor_coefficient(g, x)
    if series is None:
        return None
    j = series.index
    start = _first_finite(series.coefficient, j)
    if start is None:
        return None
    basis = as_expr(x**(series.period * j + series.offset))
    return Expansion(g, series.coefficient, j, basis, start)


def _exponential_expansion(g: Expr, x: Symbol) -> Optional[Expansion]:
    """``1/(exp(c x) - 1)`` and ``1/(exp(c x) + 1)`` (``c > 0``, also with a
    prefactor ``exp(-d x)``) as geometric series of exponentials, valid
    for ``x > 0``."""
    if not isinstance(g, Pow) or g.exp != -1 or not isinstance(g.base, Add) or len(g.base.args) != 2:
        return None
    terms = [as_expr(t) for t in g.base.args]
    for constant, other in ((terms[0], terms[1]), (terms[1], terms[0])):
        if constant.has(x) or not isinstance(other, exp):
            continue
        found = monomial(as_expr(other.args[0]), x)
        if found is None or found[1] != 1 or not found[0].is_positive:
            continue
        c = found[0]
        j = Dummy('j', integer=True, positive=True)
        if constant == -1:
            return Expansion(g, S.One, j, as_expr(exp(-c * j * x)), 1)
        if constant == 1:
            return Expansion(g, as_expr(S.NegativeOne**(j - 1)), j, as_expr(exp(-c * j * x)), 1)
    return None


_u = Dummy('u')
_k = Dummy('k', integer=True, positive=True)


class _FourierEntry:
    """A Fourier series ``function(u) = constant + Sum(coefficient(k) *
    basis(k, u), (k, 1, oo))`` valid for ``lower < u < upper``."""

    def __init__(self, function: Expr, constant: Expr, coefficient: Expr, basis: Expr,
                 lower: Expr, upper: Expr) -> None:
        self.function = function
        self.constant = constant
        self.coefficient = coefficient
        self.basis = basis
        self.lower = lower
        self.upper = upper


def _entry(function: ExprLike, constant: ExprLike, coefficient: ExprLike, basis: ExprLike,
           lower: ExprLike, upper: ExprLike) -> _FourierEntry:
    return _FourierEntry(as_expr(function), as_expr(constant), as_expr(coefficient), as_expr(basis),
                         as_expr(lower), as_expr(upper))


# the classical Fourier series ([GR] 1.441-1.444, [Zygmund] I.2.8), in
# the variable u and the index k >= 1
_FOURIER_TABLE: list[_FourierEntry] = [
    # log(sin u) = -log 2 - sum cos(2ku)/k, log(cos u) = -log 2 - sum (-1)^k cos(2ku)/k
    _entry(log(sin(_u)), -log(2), -1 / _k, cos(2 * _k * _u), 0, pi),
    _entry(log(cos(_u)), -log(2), S.NegativeOne**(_k + 1) / _k, cos(2 * _k * _u), -pi / 2, pi / 2),
    # log(tan u) = -2 sum cos((4k - 2)u)/(2k - 1)
    _entry(log(tan(_u)), 0, -2 / (2 * _k - 1), cos((4 * _k - 2) * _u), 0, pi / 2),
    # log(1 + cos u) = log 2 + 2 log cos(u/2), log(1 - cos u) = log 2 + 2 log sin(u/2)
    _entry(log(1 + cos(_u)), -log(2), 2 * S.NegativeOne**(_k + 1) / _k, cos(_k * _u), -pi, pi),
    _entry(log(1 - cos(_u)), -log(2), -2 / _k, cos(_k * _u), 0, 2 * pi),
    # the sawtooth waves
    _entry(_u, pi, -2 / _k, sin(_k * _u), 0, 2 * pi),
    _entry(_u, 0, 2 * S.NegativeOne**(_k + 1) / _k, sin(_k * _u), -pi, pi),
    _entry(_u**2, 4 * pi**2 / 3, 1, 4 * cos(_k * _u) / _k**2 - 4 * pi * sin(_k * _u) / _k, 0, 2 * pi),
    _entry(_u**2, pi**2 / 3, 4 * S.NegativeOne**_k / _k**2, cos(_k * _u), -pi, pi),
    # the rectified and the square waves, valid everywhere
    _entry(Abs(sin(_u)), 2 / pi, -4 / (pi * (4 * _k**2 - 1)), cos(2 * _k * _u), -oo, oo),
    _entry(Abs(cos(_u)), 2 / pi, -4 * S.NegativeOne**_k / (pi * (4 * _k**2 - 1)), cos(2 * _k * _u), -oo, oo),
    _entry(sign(sin(_u)), 0, 4 / (pi * (2 * _k - 1)), sin((2 * _k - 1) * _u), -oo, oo),
    _entry(sign(cos(_u)), 0, 4 * S.NegativeOne**(_k + 1) / (pi * (2 * _k - 1)), cos((2 * _k - 1) * _u), -oo, oo),
]


def _scaled(g: Expr, x: Symbol) -> Optional[tuple[Expr, Expr]]:
    """``(m, h)`` with ``g(x) = h(m x)``, ``m > 0``, when the trigonometric
    functions in ``g`` all have the argument ``m x`` (``h`` in the table
    variable ``_u``)."""
    arguments = {as_expr(t.args[0]) for t in g.atoms(sin, cos, tan)}
    if not arguments:
        return (S.One, as_expr(g.xreplace({x: _u})))
    if len(arguments) > 1:
        return None
    argument = arguments.pop()
    found = monomial(argument, x)
    if found is None or found[1] != 1 or not found[0].is_positive:
        return None
    h = as_expr(g.xreplace({argument: _u}))
    if h.has(x):
        return None
    return (found[0], h)


def _within(a: Expr, b: Expr, lower: Expr, upper: Expr, assumptions: Assumptions) -> bool:
    """Whether ``[a, b]`` lies in ``[lower, upper]``."""
    if lower != -oo and ask(as_boolean(lower <= a), assumptions) is not True:
        return False
    if upper != oo and ask(as_boolean(b <= upper), assumptions) is not True:
        return False
    return True


def _cosine_kernel(h: Expr, assumptions: Assumptions) -> Optional[tuple[Expr, Expr, Expr, Boolean]]:
    """``(P, Q, a, condition)`` for ``h = P + Q*cos(u)`` with ``P > |Q|``,
    written ``(P/(1 + a**2)) * (1 - 2*a*cos(u) + a**2)`` with
    ``a = (sqrt(P**2 - Q**2) - P)/Q``, the root of ``a**2 + 2*(P/Q)*a + 1``
    inside the unit circle; ``condition`` is ``P > |Q|`` when the
    assumptions do not settle it."""
    if not isinstance(h, Add):
        return None
    constant, rest = h.as_independent(_u, as_Add=True)
    P, rest_ = as_expr(constant), as_expr(rest)
    Q, harmonic = rest_.as_independent(_u, as_Add=False)
    if harmonic != cos(_u) or P == 0:
        return None
    Q_ = as_expr(Q)
    condition = as_boolean(P > Abs(Q_))
    verdict = ask(condition, assumptions)
    if verdict is False:
        return None
    from sympy_extras.assumptions.refine import refine
    root = as_expr(refine(sqrt(P**2 - Q_**2), assumptions))
    a = as_expr(cancel((root - P) / Q_))
    return (P, Q_, a, true if verdict is True else condition)


def _parametric_entry(h: Expr, assumptions: Assumptions) -> Optional[tuple[Expr, Expr, Expr, Boolean]]:
    """``(constant, coefficient, basis, condition)`` of the Fourier series of
    ``log(P + Q*cos(u))`` and ``1/(P + Q*cos(u))`` for ``P > |Q|``, valid for
    every ``u`` ([GR]_ 1.447.2, 1.448.1: ``log(1 - 2*a*cos(u) + a**2) =
    -2*Sum(a**k*cos(k*u)/k)`` and ``1/(1 - 2*a*cos(u) + a**2) = (1 +
    2*Sum(a**k*cos(k*u)))/(1 - a**2)`` for ``|a| < 1``)."""
    if isinstance(h, log):
        found = _cosine_kernel(as_expr(h.args[0]), assumptions)
        if found is None:
            return None
        P, Q, a, condition = found
        return (as_expr(log(P / (1 + a**2))), as_expr(-2 * a**_k / _k), cos(_k * _u), condition)
    if isinstance(h, Pow) and h.exp == -1:
        found = _cosine_kernel(as_expr(h.base), assumptions)
        if found is None:
            return None
        P, Q, a, condition = found
        scale = as_expr(cancel((1 + a**2) / (P * (1 - a**2))))
        return (scale, as_expr(2 * scale * a**_k), cos(_k * _u), condition)
    return None


#: the depth of the Fourier coefficient and moment integrals in progress:
#: no computed expansion is started inside one (the coefficient integral
#: of ``g`` would ask for the expansion of ``g`` again)
_nesting = 0


def _computed_expansion(g: Expr, x: Symbol, a: Expr, b: Expr, assumptions: Assumptions) -> Optional[Expansion]:
    """The Fourier series of ``g`` on the bounded range ``(a, b)`` with the
    coefficients computed as integrals with the symbolic index ``k``
    (:func:`~sympy_extras.integrals.conditional_integral`, the conditions
    on the parameters kept): the half-range cosine series with the
    frequencies ``pi*k/(b - a)`` in ``x - a``, then the half-range sine
    series, then the full series with ``2*pi*k/(b - a)`` (first when
    ``a != 0``, the half-range systems then carrying the shift), the
    first whose coefficients come out in closed form; ``None`` otherwise,
    or when the range is unbounded, or inside another coefficient
    integral. Each system is a complete orthogonal system of ``L2(a, b)``,
    so the series converges to ``g`` in ``L2`` for every square-integrable
    ``g``."""
    global _nesting
    if _nesting > 0 or a in (oo, -oo) or b in (oo, -oo):
        return None
    if any(t.has(x) for t in g.atoms(sin, cos, tan)):
        # a harmonic in g resonates with the basis at one index, where
        # the formula computed for the symbolic index is wrong
        return None
    from .antiderivative import antiderivative_integral
    from .definite import conditional_integral
    length = as_expr(b - a)
    k = Dummy('k', integer=True, positive=True)
    facts: list[Boolean] = [element(k, S.Naturals)]
    if isinstance(assumptions, (Boolean, bool)):
        facts.append(as_boolean(assumptions))
    elif assumptions is not None:
        facts.extend(as_boolean(s) for s in assumptions)
    budget = None if settings.timeout is None else settings.timeout / 2

    # the coefficients as integrals over (0, length) in t = x - a, the
    # canonical range of the driver (the range (-pi, pi) would be cut at 0
    # and the two halves left unsimplified)
    t = Dummy('t', positive=True)
    shifted = as_expr(g.subs(x, a + t))

    def coefficient(weight: Expr) -> Optional[ConditionalValue]:
        # the antiderivative first (a polynomial times a harmonic integrates
        # by parts to a clean formula in k, where the driver splits the
        # range at the quarter periods and leaves sin(k pi/2) behind)
        integrand = as_expr(shifted * weight)
        found = antiderivative_integral(integrand, t, S.Zero, length, facts)
        if found is None:
            found = conditional_integral(integrand, t, S.Zero, length, facts)
        if found is None or found.value.has(Integral, Piecewise, nan, zoo, oo, -oo):
            return None
        simpler = attempt(lambda: as_expr(simplify(found.value)), settings.timeout / 8 if settings.timeout else None)
        if simpler is not None and simpler.count_ops() < found.value.count_ops():
            return ConditionalValue(simpler, found.condition)
        return found

    def basis_of(weight: Expr) -> Expr:
        # the weight in x: cos(w (x - a)) = cos(w x) cos(w a) + sin(w x) sin(w a)
        omega = as_expr(weight.args[0] / t)
        if isinstance(weight, cos):
            return as_expr(cos(omega * x) * cos(omega * a) + sin(omega * x) * sin(omega * a))
        return as_expr(sin(omega * x) * cos(omega * a) - cos(omega * x) * sin(omega * a))

    def systems() -> Optional[Expansion]:
        mean = coefficient(S.One)
        if mean is None:
            return None
        constant = as_expr(mean.value / length)
        conditions: list[Boolean] = [mean.condition]
        half = as_expr(pi * k / length)
        full = as_expr(2 * pi * k / length)
        # the half-range systems are complete on (a, b) in the variable x - a
        systems: list[list[Expr]] = [[as_expr(cos(half * t))], [as_expr(sin(half * t))],
                                     [as_expr(cos(full * t)), as_expr(sin(full * t))]]
        if a != 0:
            systems = systems[2:] + systems[:2]
        for weights in systems:
            parts: list[tuple[Expr, Expr]] = []
            for weight in weights:
                found = coefficient(weight)
                if found is None:
                    break
                conditions.append(found.condition)
                parts.append((as_expr(2 * found.value / length), basis_of(weight)))
            if len(parts) < len(weights):
                continue
            nonzero = [(amplitude, basis) for amplitude, basis in parts if amplitude != 0]
            if not nonzero:
                if as_expr(g - constant) == 0:
                    return Expansion(g, S.Zero, k, parts[0][1], 1, constant, as_boolean(And(*conditions)))
                continue
            if len(nonzero) == 1:
                amplitude, basis = nonzero[0]
                # (-1)**k sin(k x) from the shift: the sign into the coefficient
                factor, harmonic = basis.as_independent(x, as_Add=False)
                if not isinstance(basis, Add):
                    amplitude, basis = as_expr(amplitude * factor), as_expr(harmonic)
                return Expansion(g, amplitude, k, basis, 1, constant, as_boolean(And(*conditions)))
            basis = as_expr(Add(*[amplitude * b_ for amplitude, b_ in nonzero]))
            return Expansion(g, S.One, k, basis, 1, constant, as_boolean(And(*conditions)))
        return None

    _nesting += 1
    try:
        return attempt(systems, budget)
    finally:
        _nesting -= 1


def fourier_expansion(g: ExprLike, x: Symbol, a: ExprLike, b: ExprLike,
                      assumptions: Assumptions = None, compute: bool = False) -> Optional[Expansion]:
    """The Fourier series of ``g`` valid on ``(a, b)`` from the table of the
    classical series, as an :class:`Expansion` with the harmonics
    ``cos(omega(k)*x)``, ``sin(omega(k)*x)`` as basis, the index ``k`` a
    positive integer and the mean as ``constant``; ``None`` when ``g`` is
    not a rescaling ``h(m x)`` (``m > 0``) of a tabulated function or
    ``(a, b)`` leaves the range of validity.

    The table holds ``log(sin(u))`` on ``(0, pi)``, ``log(cos(u))`` on
    ``(-pi/2, pi/2)``, ``log(tan(u))`` on ``(0, pi/2)``, ``log(1 + cos(u))``
    on ``(-pi, pi)``, ``log(1 - cos(u))`` on ``(0, 2*pi)``, ``u`` and
    ``u**2`` on ``(0, 2*pi)`` and on ``(-pi, pi)``, ``Abs(sin(u))``,
    ``Abs(cos(u))``, ``sign(sin(u))`` and ``sign(cos(u))`` everywhere
    ([GR]_ 1.441-1.444); a positive constant factor in a logarithm goes
    into the mean (``log(2*sin(x/2))`` on ``(0, 2*pi)`` has mean zero).
    The parametric entries ``log(P + Q*cos(u))`` and ``1/(P + Q*cos(u))``
    for ``P > |Q|`` (the Poisson kernel, [GR]_ 1.447-1.448) come next,
    with the condition ``P > |Q|`` on the expansion when the assumptions
    do not settle it. With ``compute=True`` a factor outside the table on
    a bounded range gets its coefficients computed as integrals with the
    symbolic index (:func:`_computed_expansion`), and so does a
    polynomial factor, whose tabulated series belong to the periods
    ``2*pi`` and whose series on ``(a, b)`` itself is cheap.

    Examples
    ========

    >>> from sympy import symbols, log, sin, cos, pi
    >>> from sympy_extras.integrals.series import fourier_expansion
    >>> x = symbols('x')
    >>> a = symbols('a', positive=True)
    >>> fourier_expansion(log(2*sin(x/2)), x, 0, 2*pi)
    Expansion(log(2*sin(x/2)), -1/_k, cos(_k*x), start=1)
    >>> fourier_expansion(log(sin(x)), x, 0, pi)
    Expansion(log(sin(x)), -1/_k, cos(2*_k*x), start=1, constant=-log(2))
    >>> fourier_expansion(log(sin(x)), x, 0, 2*pi) is None
    True
    >>> fourier_expansion(1/(1 - 2*a*cos(x) + a**2), x, 0, pi, a < 1)
    Expansion(1/(a**2 - 2*a*cos(x) + 1), -2*a**_k/(a**2 - 1), cos(_k*x), start=1, constant=-1/(a**2 - 1))
    >>> fourier_expansion(x**2, x, 0, pi, compute=True)
    Expansion(x**2, 4*(-1)**_k/_k**2, cos(_k*x), start=1, constant=pi**2/3)
    """
    g_, a_, b_ = as_expr(g), as_expr(a), as_expr(b)
    polynomial = bool(g_.is_polynomial(x))
    found = None if compute and polynomial else _tabulated_expansion(g_, x, a_, b_, assumptions)
    if found is not None or not compute:
        return found
    if not g_.has(x) or _harmonic_parts(g_, x) is not None:
        return None
    computed = _computed_expansion(g_, x, a_, b_, assumptions)
    if computed is None and polynomial:
        return _tabulated_expansion(g_, x, a_, b_, assumptions)
    return computed


def _tabulated_expansion(g_: Expr, x: Symbol, a_: Expr, b_: Expr, assumptions: Assumptions) -> Optional[Expansion]:
    """The table and the parametric entries of :func:`fourier_expansion`."""
    scaled = _scaled(g_, x)
    if scaled is None:
        return None
    scale, h = scaled
    shift: Expr = S.Zero
    if isinstance(h, log):
        factor, argument = as_expr(h.args[0]).as_coeff_Mul()
        if factor != 1:
            if not factor.is_positive:
                return None
            h = as_expr(log(argument))
            shift = as_expr(log(factor))
    for entry in _FOURIER_TABLE:
        if entry.function != h:
            continue
        if not _within(a_, b_, as_expr(entry.lower / scale), as_expr(entry.upper / scale), assumptions):
            continue
        k = Dummy('k', integer=True, positive=True)
        replacement = {_u: as_expr(scale * x), _k: k}
        coefficient = as_expr(entry.coefficient.xreplace(replacement))
        basis = as_expr(entry.basis.xreplace(replacement))
        return Expansion(g_, coefficient, k, basis, 1, as_expr(entry.constant + shift))
    parametric = _parametric_entry(h, assumptions)
    if parametric is not None:
        constant, coefficient, basis, condition = parametric
        k = Dummy('k', integer=True, positive=True)
        replacement = {_u: as_expr(scale * x), _k: k}
        return Expansion(g_, as_expr(coefficient.xreplace(replacement)), k, as_expr(basis.xreplace(replacement)), 1,
                         as_expr(constant + shift), condition)
    return None


def expansions(f: Expr, x: Symbol, a: Expr, b: Expr) -> Iterator[tuple[Expansion, Expr]]:
    """The ways of writing ``f`` as an expanded factor times a rest,
    each as ``(expansion, rest)``: every factor of ``f`` (and ``f``
    itself, last, since its formal power series is the slowest to find:
    the expansions are produced lazily) with a power series about 0, and
    the geometric series of exponentials on ``(0, oo)``.

    Examples
    ========

    >>> from sympy import symbols, log
    >>> from sympy_extras.integrals.series import expansions
    >>> x = symbols('x')
    >>> [(e.factor, rest) for e, rest in expansions(log(1 - x)/x, x, 0, 1)]
    [(log(1 - x), 1/x), (log(1 - x)/x, 1)]
    """
    factors = [as_expr(t) for t in Mul.make_args(f)]
    # the factors first, the whole integrand last (its formal power
    # series is the slowest to find)
    candidates: list[Expr] = list(factors)
    if len(factors) > 1:
        candidates.append(f)
    seen: set[Expr] = set()
    for g in candidates:
        if not g.has(x) or g in seen or g.is_polynomial(x) or monomial(g, x) is not None:
            continue
        seen.add(g)
        rest = as_expr(f / g)
        if b == oo:
            exponential = _exponential_expansion(g, x)
            if exponential is not None:
                yield (exponential, rest)
        if not _analytic_at_zero(g, x):
            continue
        power = _power_expansion(g, x)
        if power is not None:
            yield (power, rest)


def _analytic_at_zero(g: Expr, x: Symbol) -> bool:
    """Whether ``g`` has a finite limit at 0, so that a power series about
    0 can exist (``log(x)`` and ``1/x`` have none, and SymPy's formal power
    series runs into the time limit looking for one)."""
    value = attempt(lambda: as_expr(limit(g, x, 0, '+')), settings.timeout)
    return value is not None and not value.has(oo, -oo, zoo, nan)


def moments(expansion: Expansion, rest: Expr, x: Symbol, a: Expr, b: Expr,
            assumptions: Assumptions = None) -> Optional[Expr]:
    """``Integral(basis(j) * rest, (x, a, b))`` as a closed form in the
    index ``j`` (a symbolic nonnegative integer), or ``None``."""
    from .definite import definite_integral
    j = expansion.index
    facts: list[Boolean] = [element(j, S.Naturals0) if expansion.start == 0 else element(j, S.Naturals)]
    if isinstance(assumptions, (Boolean, bool)):
        facts.append(as_boolean(assumptions))
    elif assumptions is not None:
        facts.extend(as_boolean(s) for s in assumptions)
    global _nesting
    _nesting += 1
    try:
        value = attempt(lambda: definite_integral(expansion.basis * rest, (x, a, b), facts), settings.timeout)
    finally:
        _nesting -= 1
    if value is None or value.has(Integral, Piecewise, nan, zoo):
        return None
    return value


def _hypergeometric_sum(term: Expr, j: Dummy, start: int) -> Optional[Expr]:
    """The sum from ``start`` of a hypergeometric term as a ``hyper``
    expanded by ``hyperexpand``."""
    ratio = attempt(lambda: hypersimp(term, j), settings.timeout)
    if ratio is None or ratio is S.false or not isinstance(ratio, Expr):
        return None
    ratio_ = as_expr(ratio)
    numerator, denominator = ratio_.as_numer_denom()
    from sympy.polys.polytools import Poly
    from sympy.polys.polyroots import roots
    try:
        p_num = Poly(numerator, j)
        p_den = Poly(denominator, j)
    except Exception:                       # noqa: BLE001 - not a polynomial ratio
        return None
    num_roots = roots(p_num)
    den_roots = roots(p_den)
    if sum(num_roots.values()) != p_num.degree() or sum(den_roots.values()) != p_den.degree():
        return None
    first = as_expr(term.subs(j, start))
    z = as_expr(p_num.LC() / p_den.LC())
    upper = [as_expr(-r + start) for r, m in num_roots.items() for _ in range(m)]
    lower = [as_expr(-r + start) for r, m in den_roots.items() for _ in range(m)]
    # the term ratio t(j+1)/t(j) = z prod(j + u) / prod(j + l): the series is
    # first * pFq(u; l; z) when the lower parameters include a 1 for the k!
    if S.One in lower:
        lower.remove(S.One)
    else:
        upper.append(S.One)
    # the series converges for p <= q, for p = q + 1 inside the unit disc
    # only; beyond that hyperexpand would give an analytic continuation
    if len(upper) > len(lower) + 1:
        return None
    if len(upper) == len(lower) + 1 and ask(as_boolean(Abs(z) < 1)) is not True:
        return None
    series = attempt(lambda: as_expr(hyperexpand(first * hyper(upper, lower, z))), settings.timeout)
    if series is None or series.has(hyper):
        return None
    return series


def _rational_sum(term: Expr, j: Dummy, start: int) -> Optional[Expr]:
    """The sum from ``start`` of a rational function of ``j`` by partial
    fractions: Hurwitz zeta values (polygamma functions) for the multiple
    poles and digamma values for the simple poles, whose coefficients
    must add up to zero for convergence (``summation`` returns ``nan`` for
    ``Sum(1/(j*(4*j**2 - 1)))``, whose partial fractions diverge
    separately)."""
    if not term.is_rational_function(j):
        return None
    parts = attempt(lambda: as_expr(apart(term, j, full=True).doit()), settings.timeout)
    if parts is None:
        return None
    total: Expr = S.Zero
    simple: Expr = S.Zero
    for summand in Add.make_args(parts):
        c, rest = as_expr(summand).as_independent(j)
        if not isinstance(rest, Pow) or not isinstance(rest.exp, Integer) or rest.exp >= 0:
            return None
        m = -int(rest.exp)
        base = as_expr(rest.base)
        p = as_expr(base.coeff(j))
        q = as_expr(expand_mul(base - p * j))
        if p == 0 or q.has(j) or c.has(j):
            return None
        # 1/(p j + q)^m = p^-m / (i + r)^m with i = j - start
        r = as_expr(q / p + start)
        if r.is_integer and r <= 0:
            return None
        weight = as_expr(c / p**m)
        if m >= 2:
            total += weight * zeta(m, r)
        else:
            simple += weight
            total -= weight * polygamma(0, r)
    if cancel(simple) != 0:
        return None
    return _polygamma_form(as_expr(total))


def _polylog_sum(term: Expr, j: Dummy, start: int, assumptions: Assumptions = None) -> Optional[Expr]:
    """The sum from ``start`` of ``c * w**j / j**s`` (``|w| <= 1``, ``s >= 1``;
    after the trigonometric functions of ``j`` are written as
    exponentials) through ``polylog(s, w)``: the series of the Clausen
    functions, ``Sum(sin(j*t)/j**2) = (polylog(2, exp(I*t)) - polylog(2,
    exp(-I*t)))/(2*I)``."""
    pieces: list[tuple[Expr, Expr]] = []
    for summand in Add.make_args(term):
        # the part free of the index stays as it is: rewritten and
        # expanded it would mix with the powers (1/(1 - a**2) times a**j
        # became a factor 1/(j**2 - j**2*a**2))
        outer, inner = as_expr(summand).as_independent(j, as_Add=False)
        for piece in Add.make_args(as_expr(expand_mul(as_expr(inner).rewrite(exp)))):
            pieces.append((as_expr(outer), as_expr(piece)))
    total: Expr = S.Zero
    for outer, summand in pieces:
        coefficient: Expr = outer
        base: Expr = S.One
        order: Optional[int] = None
        for factor in Mul.make_args(summand):
            factor_ = as_expr(factor)
            if not factor_.has(j):
                coefficient *= factor_
                continue
            if isinstance(factor_, Pow) and factor_.base == j:
                if order is not None or not isinstance(factor_.exp, Integer) or factor_.exp >= 0:
                    return None
                order = -int(factor_.exp)
                continue
            if isinstance(factor_, Pow):
                b_, e_ = as_expr(factor_.base), as_expr(factor_.exp)
            elif isinstance(factor_, exp):
                b_, e_ = as_expr(S.Exp1), as_expr(factor_.args[0])
            else:
                return None
            p = as_expr(e_.coeff(j))
            q = as_expr(expand_mul(e_ - p * j))
            if b_.has(j) or p.has(j) or q.has(j):
                return None
            base *= b_**p
            coefficient *= b_**q
        if order is None or ask(as_boolean(Abs(base) <= 1), assumptions) is not True:
            return None
        if order == 1 and ask(Ne(base, 1), assumptions) is not True:
            return None
        value: Expr = as_expr(polylog(order, base))
        for i in range(1, start):
            value -= base**i / Integer(i)**order
        total += coefficient * value
    return as_expr(total)


def sum_series(term: Expr, j: Dummy, start: int, assumptions: Assumptions = None) -> Optional[Expr]:
    """``Sum(term, (j, start, oo))`` in closed form by :func:`sympy.summation`,
    through polylogarithms for the trigonometric series ``Sum(sin(j*t)/j**s)``
    of the Clausen functions, by
    :func:`~sympy_extras.concrete.polygamma_series`,
    :func:`~sympy_extras.concrete.zeilberger_sum` or as a hypergeometric
    series; ``None`` when none closes it.

    Examples
    ========

    >>> from sympy import Dummy
    >>> from sympy_extras.integrals.series import sum_series
    >>> j = Dummy('j', integer=True, nonnegative=True)
    >>> sum_series((-1)**j/(2*j + 1)**2, j, 0)
    Catalan
    """
    total = attempt(lambda: summation(term, (j, start, oo)), settings.timeout)
    if isinstance(total, Expr) and total.has(Piecewise):
        # the branch of a parametric sum (|a| < 1 for a geometric series)
        # which the assumptions settle
        folded = piecewise_fold(total)
        total = _settled_branch(folded, assumptions) if isinstance(folded, Piecewise) else None
    if isinstance(total, Expr) and total.has(exp_polar):
        # polylog(2, a*exp_polar(I*pi)) is polylog(2, -a) off the cut [1, oo)
        total = as_expr(total.xreplace({node: exp(node.args[0]) for node in total.atoms(exp_polar)}))
    if isinstance(total, Expr) and not total.has(Sum, nan, zoo, Piecewise) and not total.has(oo):
        return _polygamma_form(as_expr(total))
    if term.has(sin, cos, exp, Pow):
        clausen = attempt(lambda: _polylog_sum(term, j, start, assumptions), settings.timeout)
        if clausen is not None:
            return clausen
    rational = _rational_sum(term, j, start)
    if rational is not None:
        return rational
    n = Dummy('n', integer=True, positive=True)
    shifted = as_expr(term.subs(j, n))
    value = attempt(lambda: polygamma_series(shifted, n, max(start, 1)), settings.timeout)
    if value is not None and start >= 1:
        return value
    closed = attempt(lambda: zeilberger_sum(shifted, (n, start, oo)), settings.timeout)
    if isinstance(closed, Expr) and not closed.has(Sum, nan, zoo, oo):
        return as_expr(closed)
    return _hypergeometric_sum(term, j, start)


def _settled_branch(total: Piecewise, assumptions: Assumptions) -> Optional[Expr]:
    """The value of the first branch whose condition the assumptions
    settle as true, ``None`` when none is settled (a branch refuted is
    skipped)."""
    for value, condition in ((as_expr(pair.args[0]), as_boolean(pair.args[1])) for pair in total.args):
        verdict = ask(condition, assumptions)
        if verdict is True:
            return value
        if verdict is None:
            return None
    return None


def _justified(term: Expr, j: Dummy, start: int, assumptions: Assumptions) -> Optional[bool]:
    """Whether the series of the absolute values of the integrated terms
    converges (dominated convergence justifies the interchange); ``None``
    when undecided."""
    verdict = attempt(lambda: sum_convergence(Abs(term), j, assumptions, max(start, 1)), settings.timeout)
    if verdict is None:
        return None
    if verdict is true:
        return True
    if verdict is S.false:
        return False
    decided = ask(as_boolean(verdict), assumptions)
    return decided


class _Harmonics:
    """A trigonometric series ``constant + Sum(coefficient(k) * (cosine(k) *
    cos(frequency(k) * x) + sine(k) * sin(frequency(k) * x)), (k, start, oo))``,
    or a single harmonic (``index`` ``None``, the sum being the one term)."""

    def __init__(self, constant: Expr, coefficient: Expr, index: Optional[Dummy], start: int,
                 cosine: Expr, sine: Expr, frequency: Expr) -> None:
        self.constant = constant
        self.coefficient = coefficient
        self.index = index
        self.start = start
        self.cosine = cosine
        self.sine = sine
        self.frequency = frequency

    def at(self, value: Expr, index: Expr) -> Expr:
        """``value`` with the index replaced by ``index``."""
        if self.index is None:
            return value
        return as_expr(value.subs(self.index, index))


def _harmonic_parts(basis: Expr, x: Symbol) -> Optional[tuple[Expr, Expr, Expr]]:
    """``(cosine, sine, frequency)`` of ``basis = cosine * cos(frequency * x)
    + sine * sin(frequency * x)``."""
    cosine: Expr = S.Zero
    sine: Expr = S.Zero
    frequency: Optional[Expr] = None
    for term in Add.make_args(basis):
        amplitude, harmonic = as_expr(term).as_independent(x)
        if not isinstance(harmonic, (sin, cos)):
            return None
        found = monomial(as_expr(harmonic.args[0]), x)
        if found is None or found[1] != 1:
            return None
        if frequency is None:
            frequency = found[0]
        elif frequency != found[0]:
            return None
        if isinstance(harmonic, cos):
            cosine += as_expr(amplitude)
        else:
            sine += as_expr(amplitude)
    if frequency is None:
        return None
    return (cosine, sine, frequency)


def _harmonics(expansion: Expansion, x: Symbol) -> Optional[_Harmonics]:
    """The Fourier :class:`Expansion` as a :class:`_Harmonics`."""
    parts = _harmonic_parts(expansion.basis, x)
    if parts is None:
        return None
    cosine, sine, frequency = parts
    return _Harmonics(expansion.constant, expansion.coefficient, expansion.index, expansion.start,
                      cosine, sine, frequency)


def _harmonic(rest: Expr, x: Symbol, assumptions: Assumptions) -> Optional[_Harmonics]:
    """A single harmonic ``c * cos(n * x)`` or ``c * sin(n * x)`` with ``n`` a
    positive integer as a :class:`_Harmonics`."""
    parts = _harmonic_parts(rest, x)
    if parts is None:
        return None
    cosine, sine, frequency = parts
    if ask(element(frequency, S.Naturals), assumptions) is not True:
        return None
    return _Harmonics(S.Zero, S.One, None, 0, cosine, sine, frequency)


def _linear(frequency: Expr, k: Dummy) -> Optional[tuple[int, int]]:
    """``(p, q)`` with ``frequency = p*k + q``, integers with ``p > 0``."""
    p = as_expr(expand_mul(frequency).coeff(k))
    q = as_expr(expand_mul(frequency - p * k))
    if not isinstance(p, Integer) or not isinstance(q, Integer) or p <= 0 or q.has(k):
        return None
    return (int(p), int(q))


def _primitive(s: _Harmonics, x: Symbol, a: Expr, b: Expr) -> Expr:
    """``Integral(coefficient(k) * (cosine(k) * cos(frequency(k) * x) + ...), (x, a, b))``
    as a function of the index (the frequency is nonzero)."""
    w = s.frequency
    integral = s.cosine * (sin(w * b) - sin(w * a)) / w + s.sine * (cos(w * a) - cos(w * b)) / w
    return as_expr(expand_mul(s.coefficient * integral))


def _integrated(s: _Harmonics, x: Symbol, a: Expr, b: Expr, assumptions: Assumptions = None) -> Optional[Expr]:
    """``Integral(f - constant, (x, a, b))`` of the series ``f`` (termwise
    integration of a Fourier series is always valid, [Zygmund]_ I.2.7)."""
    term = _primitive(s, x, a, b)
    if s.index is None or term == 0:
        return term
    value = sum_series(term, s.index, s.start, assumptions)
    if value is None or value.has(s.index):
        return None
    return value


def _orthogonal(s1: _Harmonics, s2: _Harmonics, a: Expr, b: Expr) -> bool:
    """Whether the harmonics of the two series with different frequencies
    are orthogonal on ``(a, b)``: the integrals of ``cos(m x)`` and
    ``sin(m x)`` vanish for the sums and differences ``m`` of frequencies
    that occur (for integer indices SymPy evaluates ``sin(2*pi*(j - k))``
    to zero)."""
    cosines = (s1.cosine != 0 and s2.cosine != 0) or (s1.sine != 0 and s2.sine != 0)
    mixed = (s1.cosine != 0 and s2.sine != 0) or (s1.sine != 0 and s2.cosine != 0)
    for m in (as_expr(expand_mul(s1.frequency + s2.frequency)), as_expr(expand_mul(s1.frequency - s2.frequency))):
        if cosines and expand_mul(sin(m * b) - sin(m * a)) != 0:
            return False
        if mixed and expand_mul(cos(m * a) - cos(m * b)) != 0:
            return False
    return True


def _diagonal(s1: _Harmonics, s2: _Harmonics, length: Expr,
              assumptions: Assumptions) -> Optional[tuple[Expr, Optional[Dummy], int]]:
    """The terms of the double series of ``Integral(f1 * f2)`` with equal
    frequencies (the others vanish by orthogonality) as ``(term, index,
    start)`` of a single series (``index`` ``None`` for a single term);
    ``None`` when the pairing of the indices is not found."""
    if s1.index is None and s2.index is not None:
        s1, s2 = s2, s1

    def product(j: Expr, k: Expr) -> Expr:
        # Integral(cos(w x)**2) = Integral(sin(w x)**2) = length/2 on the
        # range, the mixed products integrate to zero
        inner = s1.at(s1.cosine, j) * s2.at(s2.cosine, k) + s1.at(s1.sine, j) * s2.at(s2.sine, k)
        return as_expr(s1.at(s1.coefficient, j) * s2.at(s2.coefficient, k) * inner * length / 2)

    if s1.index is None:
        return None
    if s2.index is None:
        # the harmonic n of the second factor picks the term with
        # frequency1(j) = p j + q = n
        found = _linear(s1.frequency, s1.index)
        if found is None:
            return None
        p, q = found
        j0 = as_expr((s2.frequency - q) / p)
        if j0.is_integer is False:
            return (S.Zero, None, 0)
        if j0.is_integer is not True:
            return None
        inside = ask(as_boolean(j0 >= s1.start), assumptions)
        if inside is False:
            return (S.Zero, None, 0)
        if inside is not True:
            return None
        return (product(j0, S.Zero), None, 0)
    return _paired(s1, s2, product)


def _paired(s1: _Harmonics, s2: _Harmonics, product: Callable[[Expr, Expr], Expr]) -> Optional[tuple[Expr, Optional[Dummy], int]]:
    """The single series of the pairs ``(j, k)`` with ``frequency1(j) =
    frequency2(k)``."""
    assert s1.index is not None and s2.index is not None
    j, k = s1.index, s2.index
    first = _linear(s1.frequency, j)
    second = _linear(s2.frequency, k)
    if first is None or second is None:
        return None
    (p1, q1), (p2, q2) = first, second
    if p1 % p2 == 0 and (q1 - q2) % p2 == 0:
        # k = (p1 j + q1 - q2)/p2 >= start2 from j >= (p2 start2 - q1 + q2)/p1
        paired = as_expr(Rational(p1, p2) * j + Rational(q1 - q2, p2))
        start = max(s1.start, int(ceiling(Rational(p2 * s2.start - q1 + q2, p1))))
        return (product(j, paired), j, start)
    if p2 % p1 == 0 and (q2 - q1) % p1 == 0:
        return _paired(s2, s1, lambda kk, jj: product(jj, kk))
    if (q2 - q1) % gcd(p1, p2) != 0:
        # no common frequency at all
        return (S.Zero, None, 0)
    return None


def _fourier_product(s1: _Harmonics, s2: _Harmonics, x: Symbol, a: Expr, b: Expr,
                     assumptions: Assumptions) -> Optional[Expr]:
    """``Integral(f1 * f2, (x, a, b))`` of two trigonometric series by
    orthogonality: the products of the constants, the constants times
    the integrated series, and the single series of the products with
    equal frequencies (Parseval's theorem, which justifies the
    interchange for square-integrable factors)."""
    length = as_expr(b - a)
    total = as_expr(s1.constant * s2.constant * length)
    for s, other in ((s1, s2), (s2, s1)):
        if other.constant == 0:
            continue
        integrated = _integrated(s, x, a, b, assumptions)
        if integrated is None:
            return None
        total += other.constant * integrated
    if not _orthogonal(s1, s2, a, b):
        return None
    diagonal = _diagonal(s1, s2, length, assumptions)
    if diagonal is None:
        return None
    term, index, start = diagonal
    if index is None or term == 0:
        return as_expr(total + term)
    value = sum_series(term, index, start, assumptions)
    if value is None or value.has(index):
        return None
    return as_expr(total + value)


def _fourier_factors(f: Expr, x: Symbol) -> tuple[Expr, list[Expr]]:
    """The constant of ``f`` and its factors depending on ``x`` with
    multiplicity (a square counted twice)."""
    constant: Expr = S.One
    factors: list[Expr] = []
    for t in Mul.make_args(f):
        t_ = as_expr(t)
        if not t_.has(x):
            constant *= t_
        elif isinstance(t_, Pow) and t_.exp == 2 and monomial(as_expr(t_.base), x) is None:
            factors.extend([as_expr(t_.base)] * 2)
        else:
            factors.append(t_)
    return (as_expr(constant), factors)


def _resonant(expansion: Expansion, rest: Expr, x: Symbol, assumptions: Assumptions) -> bool:
    """Whether a harmonic of the rest may share a frequency with the
    series, so that the moments computed for the symbolic index by an
    antiderivative would miss the resonant term (``Integral(cos(2*k*x) *
    cos(x)**2, (x, 0, pi))`` is ``pi/4`` at ``k = 1`` and zero otherwise):
    a rest with trigonometric functions of ``x`` is safe only when it is
    a single harmonic whose frequency the series never takes."""
    if not any(t.has(x) for t in rest.atoms(sin, cos, tan)):
        return False
    parts = _harmonic_parts(rest, x)
    series = _harmonic_parts(expansion.basis, x)
    if parts is None or series is None:
        return True
    found = _linear(series[2], expansion.index)
    if found is None:
        return True
    p, q = found
    j0 = as_expr((parts[2] - q) / p)
    if j0.is_integer is False:
        return False
    return not (j0.is_integer is True and ask(as_boolean(j0 < expansion.start), assumptions) is True)


def _termwise(expansion: Expansion, rest: Expr, x: Symbol, a: Expr, b: Expr,
              assumptions: Assumptions) -> Optional[Expr]:
    """``Integral(factor * rest, (x, a, b))`` from the Fourier series of the
    factor integrated termwise against the rest: the mean times the
    integral of the rest plus the series of the moments."""
    from .definite import definite_integral, verify_numerically
    if _resonant(expansion, rest, x, assumptions):
        return None
    mean: Optional[Expr] = S.Zero
    if expansion.constant != 0:
        mean = attempt(lambda: definite_integral(rest, (x, a, b), assumptions), settings.timeout)
        if mean is None or mean.has(Integral, Piecewise, nan, zoo):
            return None
    moment = moments(expansion, rest, x, a, b, assumptions)
    if moment is None:
        return None
    j = expansion.index
    term = as_expr(expansion.coefficient * moment)
    value = sum_series(term, j, expansion.start, assumptions)
    if value is None or value.has(j):
        return None
    total = as_expr(expansion.constant * mean + value)
    justified = _justified(term, j, expansion.start, assumptions)
    if justified is False:
        return None
    if justified is None:
        if not settings.numerical_checks:
            return None
        if verify_numerically(total, as_expr(expansion.factor * rest), x, a, b, assumptions) is not True:
            return None
    return total


def fourier_integral(f: ExprLike, x: Symbol, a: ExprLike, b: ExprLike, assumptions: Assumptions = None,
                     expand: Optional[Expr] = None) -> Optional[ConditionalValue]:
    """``Integral(f, (x, a, b))`` through the Fourier series of the factors
    of ``f`` (:func:`fourier_expansion`): a single expanded factor is
    integrated termwise, two expanded factors (or one and a harmonic
    ``cos(n*x)``, ``sin(n*x)`` with ``n`` a positive integer) by
    orthogonality, and an expanded factor times any rest through the
    moments ``Integral(cos(omega(k)*x)*rest, (x, a, b))`` computed by
    :func:`~sympy_extras.integrals.definite_integral` for the symbolic
    index; ``None`` when no factor has a series on ``(a, b)`` or no step
    closes. The tabulated and parametric series are tried first, then
    the series with computed coefficients of the factors outside the
    table (bounded ranges only). The interchange of sum and integral is
    Parseval's theorem for two square-integrable factors and the
    ``L2`` convergence of the series against a bounded rest; the moments
    route checks its own justification (:func:`_justified`) or the value
    numerically. The conditions of the expansions (``P > |Q|`` for the
    parametric entries) are those of the value.

    Examples
    ========

    >>> from sympy import symbols, log, sin, cos, pi
    >>> from sympy_extras.integrals.series import fourier_integral
    >>> x = symbols('x')
    >>> n = symbols('n', integer=True, positive=True)
    >>> a = symbols('a', positive=True)
    >>> fourier_integral(log(sin(x)), x, 0, pi)
    ConditionalValue(-pi*log(2))
    >>> fourier_integral(log(sin(x))*log(cos(x)), x, 0, pi/2).value.expand()
    -pi**3/48 + pi*log(2)**2/2
    >>> fourier_integral(log(2*sin(x/2))*cos(n*x), x, 0, 2*pi)
    ConditionalValue(-pi/n)
    >>> fourier_integral(x**2*log(sin(x)), x, 0, pi)
    ConditionalValue(-pi**3*log(2)/3 - pi*zeta(3)/2)
    >>> fourier_integral(cos(n*x)/(1 - 2*a*cos(x) + a**2), x, 0, pi, a < 1)
    ConditionalValue(-pi*a**n/(a**2 - 1))
    >>> fourier_integral(x**2/(1 - 2*a*cos(x) + a**2), x, 0, pi, a < 1)
    ConditionalValue(pi*(-12*polylog(2, -a) - pi**2)/(3*(a**2 - 1)))
    """
    f_, a_, b_ = as_expr(f), as_expr(a), as_expr(b)
    if ask(as_boolean(a_ < b_), assumptions) is not True:
        return None
    constant, factors = _fourier_factors(f_, x)
    expanded = [fourier_expansion(g, x, a_, b_, assumptions) for g in factors]
    found = _fourier_routes(f_, constant, factors, expanded, x, a_, b_, assumptions, expand)
    if found is not None:
        return found
    if a_ in (oo, -oo) or b_ in (oo, -oo) or _nesting > 0:
        return None
    if all(e is None for e in expanded) and all(_harmonic(g, x, assumptions) is None for g in factors):
        # nothing trigonometric with integer frequencies to pair with
        return None
    # the factors outside the table with computed coefficients; the
    # polynomial factors too, whose tabulated series belong to other
    # periods (x**2 on (0, 2*pi) restricted to (0, pi) has sine terms, not
    # orthogonal there to the cosines of the other factor) and whose
    # coefficients are cheap
    computed = [fourier_expansion(g, x, a_, b_, assumptions, compute=True)
                if e is None or g.is_polynomial(x) else e for g, e in zip(factors, expanded)]
    if all(e is None or e in expanded for e in computed):
        return None
    return _fourier_routes(f_, constant, factors, computed, x, a_, b_, assumptions, expand)


def _fourier_routes(f_: Expr, constant: Expr, factors: list[Expr], expanded: list[Optional[Expansion]],
                    x: Symbol, a_: Expr, b_: Expr, assumptions: Assumptions,
                    expand: Optional[Expr]) -> Optional[ConditionalValue]:
    """The routes of :func:`fourier_integral` for given expansions of the
    factors (``None`` for a factor without one)."""
    if all(e is None for e in expanded):
        return None
    if expand is not None and all(e is None or e.factor != expand for e in expanded):
        return None
    condition = as_boolean(And(*[e.condition for e in expanded if e is not None]))
    value: Optional[Expr] = None
    if len(factors) == 1 and expanded[0] is not None:
        s = _harmonics(expanded[0], x)
        if s is not None:
            integrated = _integrated(s, x, a_, b_, assumptions)
            if integrated is not None:
                value = as_expr(s.constant * (b_ - a_) + integrated)
    elif len(factors) == 2:
        series = [_harmonics(e, x) if e is not None else _harmonic(g, x, assumptions)
                  for e, g in zip(expanded, factors)]
        if series[0] is not None and series[1] is not None:
            value = _fourier_product(series[0], series[1], x, a_, b_, assumptions)
    if value is not None:
        return ConditionalValue(_polygamma_form(tidy(as_expr(constant * value), assumptions)), condition)
    # an expanded factor against the rest of the integrand, the
    # polynomial factors last (their moments are the easiest)
    seen: set[Expr] = set()
    ordered = sorted(zip(factors, expanded), key=lambda pair: bool(pair[0].is_polynomial(x)))
    for g, e in ordered:
        if e is None or g in seen or (expand is not None and g != expand):
            continue
        seen.add(g)
        found = _termwise(e, as_expr(f_ / g), x, a_, b_, assumptions)
        if found is not None:
            return ConditionalValue(_polygamma_form(tidy(found, assumptions)), e.condition)
    return None


def series_integral(f: ExprLike, x: Symbol, a: ExprLike, b: ExprLike, assumptions: Assumptions = None,
                    expand: Optional[Expr] = None) -> Optional[ConditionalValue]:
    """``Integral(f, (x, a, b))`` by series expansion of a factor and
    termwise integration, see the module documentation.

    Parameters
    ==========

    f : Expr
        The integrand.
    x : Symbol
    a, b : Expr
        The bounds: ``(0, 1)``, ``(0, oo)``, or ``(0, c)`` with ``c > 0``
        (mapped onto ``(0, 1)``) for the power and exponential series; any
        range inside the range of validity of a tabulated Fourier series
        (:func:`fourier_integral`).
    assumptions : Boolean or list of Booleans, optional
    expand : Expr, optional
        The factor to expand; every factor with a series is tried
        otherwise.

    Returns
    =======

    A :class:`~sympy_extras.integrals.conditions.ConditionalValue`, or
    ``None`` when no factor expands, no moment or sum closes, or the
    interchange of sum and integral is neither justified nor confirmed
    numerically.

    Examples
    ========

    >>> from sympy import symbols, log, exp, oo
    >>> from sympy_extras.integrals.series import series_integral
    >>> x = symbols('x')
    >>> p = symbols('p', positive=True)
    >>> series_integral(log(1 + x)/x, x, 0, 1)
    ConditionalValue(pi**2/12)
    >>> series_integral(x**p*log(x)/(1 - x), x, 0, 1)
    ConditionalValue(-polygamma(1, p + 1))
    >>> series_integral(exp(-x)*log(1 + x), x, 0, oo) is None
    True
    >>> from sympy import sin, pi
    >>> series_integral(x*log(sin(x)), x, 0, pi)
    ConditionalValue(-pi**2*log(2)/2)
    """
    from .definite import verify_numerically
    f_, a_, b_ = as_expr(f), as_expr(a), as_expr(b)
    fourier = fourier_integral(f_, x, a_, b_, assumptions, expand)
    if fourier is not None:
        return fourier
    if a_ != 0 or (b_ != oo and ask(as_boolean(b_ > 0), assumptions) is not True):
        return None
    if b_ not in (oo, S.One):
        # (0, c): x = c t
        t = Dummy('t', positive=True)
        found = series_integral(as_expr(f_.subs(x, b_ * t) * b_), t, S.Zero, S.One, assumptions, None)
        return found
    for expansion, rest in expansions(f_, x, a_, b_):
        if expand is not None and expansion.factor != expand:
            continue
        moment = moments(expansion, rest, x, a_, b_, assumptions)
        if moment is None:
            continue
        j = expansion.index
        term = as_expr(expansion.coefficient * moment)
        value = sum_series(term, j, expansion.start, assumptions)
        if value is None or value.free_symbols & {j}:
            continue
        justified = _justified(term, j, expansion.start, assumptions)
        if justified is False:
            continue
        if justified is None:
            if not settings.numerical_checks:
                continue
            if verify_numerically(value, f_, x, a_, b_, assumptions) is not True:
                continue
        return ConditionalValue(_polygamma_form(tidy(value, assumptions)))
    return None


def _polygamma_form(value: Expr) -> Expr:
    """Hurwitz zeta values ``zeta(n, a)`` at integers ``n >= 2`` (also
    written ``lerchphi(1, n, a)`` by ``summation``) as polygamma
    functions, and Dirichlet eta values as zeta values."""
    if value.has(dirichlet_eta):
        value = as_expr(value.rewrite(zeta))
    replacement: dict[Expr, Expr] = {}
    for node in value.atoms(lerchphi):
        if node.args[0] == 1:
            replacement[as_expr(node)] = as_expr(zeta(node.args[1], node.args[2]))
    if replacement:
        value = as_expr(value.xreplace(replacement))
        replacement = {}
    for node in value.atoms(zeta):
        if len(node.args) == 2:
            n, a = as_expr(node.args[0]), as_expr(node.args[1])
            if isinstance(n, Integer) and n >= 2:
                # zeta(n, a) = (-1)^n psi^(n-1)(a) / (n - 1)!
                replacement[as_expr(node)] = as_expr(S.NegativeOne**n * polygamma(n - 1, a) / factorial(n - 1))
    return as_expr(cancel(value.xreplace(replacement))) if replacement else value
