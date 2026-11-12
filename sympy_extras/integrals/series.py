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
   :func:`~sympy_extras.integrals.definite.definite_integral` (the Mellin table gives
   them for `e^{-x}`, `(1 - x)^b`, `\\log^n x`, `1/(1 + x)`, ...);
3. the series `\\sum_j c(j) M(j)` is summed by :func:`sympy.summation`, by
   the summation algorithms of :mod:`sympy_extras.concrete`
   (:func:`~sympy_extras.concrete.polygamma_series`,
   :func:`~sympy_extras.concrete.zeilberger_sum`) or as a hypergeometric
   series expanded by ``hyperexpand``;
4. the interchange of the sum and the integral is justified when the
   series of the absolute values of the integrated terms converges
   (:func:`~sympy_extras.concrete.convergence_sympy.sum_convergence`, dominated convergence);
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

The module works on the objects of :mod:`sympy_extras.core` and imports
no SymPy: the native summation, ``hypersimp``, ``hyperexpand``, roots,
convergence tests, Euler sums, Zeilberger's algorithm, ``ask``,
``refine``, ``simplify`` and ``tidy``. The moments, the mean values and
the Fourier coefficients are computed by the native
:func:`~sympy_extras.integrals.definite.definite_integral`,
:func:`~sympy_extras.integrals.definite.conditional_integral` and
:func:`~sympy_extras.integrals.antiderivative.antiderivative_integral`,
and so is the numerical check of an interchange which the convergence
tests do not justify (:func:`~sympy_extras.integrals.definite.verify_numerically`);
the formal power series of a factor come from SymPy's ``fps`` through
:mod:`.brackets`. SymPy is loaded on that path only. The functions on
SymPy objects are in :mod:`.series_sympy`.

Examples
========

>>> from sympy_extras.core import Symbol, oo, pi
>>> from sympy_extras.functions import log, exp, atan, sin
>>> from sympy_extras.integrals.series import series_integral
>>> x = Symbol('x')
>>> series_integral(log(1 - x)/x, x, 0, 1)
ConditionalValue(-pi**2/6)
>>> series_integral(x/(exp(x) - 1), x, 0, oo)
ConditionalValue(pi**2/6)
>>> series_integral(atan(x)/x, x, 0, 1)
ConditionalValue(Catalan)
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

import itertools
from math import gcd
from typing import Callable, Final, Iterator, Optional

from sympy_extras._timeout import attempt
from sympy_extras.assumptions.ask import Assumptions, ask
from sympy_extras.assumptions.facts import element
from sympy_extras.assumptions.refine import refine
from sympy_extras.concrete.convergence import sum_convergence
from sympy_extras.concrete.eulersums import polygamma_series
from sympy_extras.concrete.expr_with_limits import Sum
from sympy_extras.concrete.gosper import hypersimp
from sympy_extras.concrete.summations import summation
from sympy_extras.concrete.zeilberger import zeilberger_sum
from sympy_extras.core.add import Add, add
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, ExprLike, as_expr, to_expr
from sympy_extras.core.function import diff, expand_mul
from sympy_extras.core.limits import Limit, limit
from sympy_extras.core.logic import Boolean, BooleanFalse, BooleanTrue, and_, to_boolean, true
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import E, I, Integer, Rational, nan, oo, pi, zoo
from sympy_extras.core.power import Pow, sqrt
from sympy_extras.core.relational import ne
from sympy_extras.core.singleton import S
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.combinatorial.factorials import factorial
from sympy_extras.functions.elementary.complexes import Abs, abs_, sign
from sympy_extras.functions.elementary.exponential import Exp, ExpPolar, Log, exp, log
from sympy_extras.functions.elementary.piecewise import Piecewise, piecewise_fold
from sympy_extras.functions.elementary.trigonometric import Cos, Sin, Tan, cos, sin, tan
from sympy_extras.functions.special.gamma_functions import polygamma
from sympy_extras.functions.special.hyper import Hyper, hyper
from sympy_extras.functions.special.zeta_functions import DirichletEta, LerchPhi, Zeta, polylog, zeta
from sympy_extras.polys.polyerrors import BasePolynomialError
from sympy_extras.polys.polyroots import roots
from sympy_extras.polys.polytools import Poly, cancel
from sympy_extras.sets.fancysets import Naturals, Naturals0
from sympy_extras.settings import settings
from sympy_extras.simplify.hyperexpand import hyperexpand
from sympy_extras.simplify.simplify import simplify

from .brackets import taylor_coefficient
from .conditions import ConditionalValue
from .laplace import has_integral
from .marichev import tidy
from .mellin import monomial

__all__ = ['Expansion', 'expansions', 'fourier_expansion', 'moments', 'sum_series', 'fourier_integral',
           'series_integral']

_COUNTER: Final = itertools.count()


def _fresh(name: str, **assumptions: bool) -> Dummy:
    """A dummy with a name of its own: the typed polynomials map their
    symbols by name, so the indices of two series in one product (or a
    parameter ``k``) would be taken for one symbol."""
    return Dummy('%s%d' % (name, next(_COUNTER)), **assumptions)


def _statements(assumptions: Assumptions) -> list[Boolean]:
    if assumptions is None:
        return []
    if isinstance(assumptions, (Basic, bool)):
        return [to_boolean(assumptions)]
    return [to_boolean(a) for a in assumptions]


def _evaluated(e: Expr, rule: dict[Basic, Basic]) -> Expr:
    """``e.xreplace(rule)`` with the rebuilt nodes evaluated (the core's
    ``xreplace`` only rebuilds)."""
    return as_expr(e._xreplace_evaluated(rule))


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

    __slots__ = ('factor', 'coefficient', 'index', 'basis', 'start', 'constant', 'condition')

    def __init__(self, factor: Expr, coefficient: Expr, index: Dummy, basis: Expr, start: int,
                 constant: Expr = S.Zero, condition: Boolean = true) -> None:
        self.factor: Final[Expr] = factor
        self.coefficient: Final[Expr] = coefficient
        self.index: Final[Dummy] = index
        self.basis: Final[Expr] = basis
        self.start: Final[int] = start
        self.constant: Final[Expr] = constant
        self.condition: Final[Boolean] = condition

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
        values = [coefficient.subs(j, Integer(start + i)) for i in range(6)]
        if all(not v.has(zoo, nan, oo, -oo) for v in values):
            return start
    return None


def _power_expansion(g: Expr, x: Symbol) -> Optional[Expansion]:
    """The power series of ``g`` about 0 as an :class:`Expansion`."""
    series = taylor_coefficient(g, x)
    if series is None:
        return None
    j = series.index
    start = _first_finite(series.coefficient, j)
    if start is None:
        return None
    basis = x**(series.period * j + series.offset)
    return Expansion(g, series.coefficient, j, basis, start)


def _exponential_expansion(g: Expr, x: Symbol) -> Optional[Expansion]:
    """``1/(exp(c x) - 1)`` and ``1/(exp(c x) + 1)`` (``c > 0``, also with a
    prefactor ``exp(-d x)``) as geometric series of exponentials, valid
    for ``x > 0``."""
    if not isinstance(g, Pow) or g.exp != -1 or not isinstance(g.base, Add) or len(g.base.args) != 2:
        return None
    terms = list(g.base.args)
    for constant, other in ((terms[0], terms[1]), (terms[1], terms[0])):
        if constant.has(x) or not isinstance(other, Exp):
            continue
        found = monomial(other.args[0], x)
        if found is None or found[1] != 1 or not found[0].is_positive:
            continue
        c = found[0]
        j = _fresh('j', integer=True, positive=True)
        if constant == -1:
            return Expansion(g, S.One, j, exp(-c * j * x), 1)
        if constant == 1:
            return Expansion(g, S.NegativeOne**(j - 1), j, exp(-c * j * x), 1)
    return None


_u: Final = _fresh('u')
_k: Final = _fresh('k', integer=True, positive=True)


class _FourierEntry:
    """A Fourier series ``function(u) = constant + Sum(coefficient(k) *
    basis(k, u), (k, 1, oo))`` valid for ``lower < u < upper``."""

    __slots__ = ('function', 'constant', 'coefficient', 'basis', 'lower', 'upper')

    def __init__(self, function: Expr, constant: Expr, coefficient: Expr, basis: Expr,
                 lower: Expr, upper: Expr) -> None:
        self.function: Final[Expr] = function
        self.constant: Final[Expr] = constant
        self.coefficient: Final[Expr] = coefficient
        self.basis: Final[Expr] = basis
        self.lower: Final[Expr] = lower
        self.upper: Final[Expr] = upper


def _entry(function: ExprLike, constant: ExprLike, coefficient: ExprLike, basis: ExprLike,
           lower: ExprLike, upper: ExprLike) -> _FourierEntry:
    return _FourierEntry(to_expr(function), to_expr(constant), to_expr(coefficient), to_expr(basis),
                         to_expr(lower), to_expr(upper))


# the classical Fourier series ([GR] 1.441-1.444, [Zygmund] I.2.8), in
# the variable u and the index k >= 1
_FOURIER_TABLE: Final[list[_FourierEntry]] = [
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
    _entry(abs_(sin(_u)), 2 / pi, -4 / (pi * (4 * _k**2 - 1)), cos(2 * _k * _u), -oo, oo),
    _entry(abs_(cos(_u)), 2 / pi, -4 * S.NegativeOne**_k / (pi * (4 * _k**2 - 1)), cos(2 * _k * _u), -oo, oo),
    _entry(sign(sin(_u)), 0, 4 / (pi * (2 * _k - 1)), sin((2 * _k - 1) * _u), -oo, oo),
    _entry(sign(cos(_u)), 0, 4 * S.NegativeOne**(_k + 1) / (pi * (2 * _k - 1)), cos((2 * _k - 1) * _u), -oo, oo),
]


def _scaled(g: Expr, x: Symbol) -> Optional[tuple[Expr, Expr]]:
    """``(m, h)`` with ``g(x) = h(m x)``, ``m > 0``, when the trigonometric
    functions in ``g`` all have the argument ``m x`` (``h`` in the table
    variable ``_u``)."""
    arguments = {as_expr(t.args[0]) for t in g.atoms(Sin, Cos, Tan)}
    if not arguments:
        return (S.One, _evaluated(g, {x: _u}))
    if len(arguments) > 1:
        return None
    argument = arguments.pop()
    found = monomial(argument, x)
    if found is None or found[1] != 1 or not found[0].is_positive:
        return None
    h = _evaluated(g, {argument: _u})
    if h.has(x):
        return None
    return (found[0], h)


def _within(a: Expr, b: Expr, lower: Expr, upper: Expr, assumptions: Assumptions) -> bool:
    """Whether ``[a, b]`` lies in ``[lower, upper]``."""
    if lower != -oo and ask(lower <= a, assumptions) is not True:
        return False
    if upper != oo and ask(b <= upper, assumptions) is not True:
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
    P, rest = h.as_independent(_u, as_Add=True)
    Q, harmonic = rest.as_independent(_u, as_Add=False)
    if harmonic != cos(_u) or P == 0:
        return None
    condition = P > abs_(Q)
    verdict = ask(condition, assumptions)
    if verdict is False:
        return None
    root = as_expr(refine(sqrt(P**2 - Q**2), assumptions))
    a = cancel((root - P) / Q)
    return (P, Q, a, true if verdict is True else condition)


def _parametric_entry(h: Expr, assumptions: Assumptions) -> Optional[tuple[Expr, Expr, Expr, Boolean]]:
    """``(constant, coefficient, basis, condition)`` of the Fourier series of
    ``log(P + Q*cos(u))`` and ``1/(P + Q*cos(u))`` for ``P > |Q|``, valid for
    every ``u`` ([GR]_ 1.447.2, 1.448.1: ``log(1 - 2*a*cos(u) + a**2) =
    -2*Sum(a**k*cos(k*u)/k)`` and ``1/(1 - 2*a*cos(u) + a**2) = (1 +
    2*Sum(a**k*cos(k*u)))/(1 - a**2)`` for ``|a| < 1``)."""
    if isinstance(h, Log):
        found = _cosine_kernel(h.args[0], assumptions)
        if found is None:
            return None
        P, Q, a, condition = found
        return (log(P / (1 + a**2)), -2 * a**_k / _k, cos(_k * _u), condition)
    if isinstance(h, Pow) and h.exp == -1:
        found = _cosine_kernel(h.base, assumptions)
        if found is None:
            return None
        P, Q, a, condition = found
        scale = cancel((1 + a**2) / (P * (1 - a**2)))
        return (scale, 2 * scale * a**_k, cos(_k * _u), condition)
    return None


#: the depth of the Fourier coefficient and moment integrals in progress:
#: no computed expansion is started inside one (the coefficient integral
#: of ``g`` would ask for the expansion of ``g`` again)
_nesting = 0


def _bad_value(e: Expr) -> bool:
    """Whether ``e`` has an unevaluated integral, a ``Piecewise`` or an
    undefined value."""
    return has_integral(e) or e.has(Piecewise, nan, zoo)


def _computed_expansion(g: Expr, x: Symbol, a: Expr, b: Expr, assumptions: Assumptions) -> Optional[Expansion]:
    """The Fourier series of ``g`` on the bounded range ``(a, b)`` with the
    coefficients computed as integrals with the symbolic index ``k``
    (:func:`~sympy_extras.integrals.definite.conditional_integral`, the
    conditions on the parameters kept): the half-range cosine series with
    the frequencies ``pi*k/(b - a)`` in ``x - a``, then the half-range sine
    series, then the full series with ``2*pi*k/(b - a)`` (first when
    ``a != 0``, the half-range systems then carrying the shift), the
    first whose coefficients come out in closed form; ``None`` otherwise,
    or when the range is unbounded, or inside another coefficient
    integral. Each system is a complete orthogonal system of ``L2(a, b)``,
    so the series converges to ``g`` in ``L2`` for every square-integrable
    ``g``. The integrals come from the native
    :func:`~sympy_extras.integrals.antiderivative.antiderivative_integral`,
    then from :func:`~sympy_extras.integrals.definite.conditional_integral`."""
    global _nesting
    if _nesting > 0 or a in (oo, -oo) or b in (oo, -oo):
        return None
    if any(t.has(x) for t in g.atoms(Sin, Cos, Tan)):
        # a harmonic in g resonates with the basis at one index, where
        # the formula computed for the symbolic index is wrong
        return None
    from .definite import conditional_integral
    from .antiderivative import antiderivative_integral
    length = b - a
    k = _fresh('k', integer=True, positive=True)
    facts: list[Boolean] = [element(k, Naturals)] + _statements(assumptions)
    # the whole limit: the half-range coefficients of x**2 on (0, pi) take
    # a dozen seconds under load, and half the limit let the full system
    # win by timing (the tests then saw a different, valid, expansion)
    budget = settings.timeout

    # the coefficients as integrals over (0, length) in t = x - a, the
    # canonical range of the driver (the range (-pi, pi) would be cut at 0
    # and the two halves left unsimplified)
    t = _fresh('t', positive=True)
    shifted = g.subs(x, a + t)

    def coefficient(weight: Expr) -> Optional[ConditionalValue]:
        # the antiderivative first (a polynomial times a harmonic integrates
        # by parts to a clean formula in k, where the driver splits the
        # range at the quarter periods and leaves sin(k pi/2) behind)
        integrand = shifted * weight
        found = antiderivative_integral(integrand, t, S.Zero, length, facts)
        if found is None:
            found = conditional_integral(integrand, t, S.Zero, length, facts)
        if found is None or _bad_value(found.value) or found.value.has(oo, -oo):
            return None
        value = found.value
        simpler = attempt(lambda: simplify(value), settings.timeout / 8 if settings.timeout else None, native=True)
        if simpler is not None and _ops(simpler) < _ops(value):
            return ConditionalValue(simpler, found.condition)
        return found

    def basis_of(weight: Expr) -> Expr:
        # the weight in x: cos(w (x - a)) = cos(w x) cos(w a) + sin(w x) sin(w a)
        omega = as_expr(weight.args[0]) / t
        if isinstance(weight, Cos):
            return cos(omega * x) * cos(omega * a) + sin(omega * x) * sin(omega * a)
        return sin(omega * x) * cos(omega * a) - cos(omega * x) * sin(omega * a)

    def systems() -> Optional[Expansion]:
        mean = coefficient(S.One)
        if mean is None:
            return None
        constant = mean.value / length
        conditions: list[Boolean] = [mean.condition]
        half = pi * k / length
        full = 2 * pi * k / length
        # the half-range systems are complete on (a, b) in the variable x - a
        systems: list[list[Expr]] = [[cos(half * t)], [sin(half * t)], [cos(full * t), sin(full * t)]]
        if a != 0:
            systems = systems[2:] + systems[:2]
        for weights in systems:
            parts: list[tuple[Expr, Expr]] = []
            for weight in weights:
                found = coefficient(weight)
                if found is None:
                    break
                conditions.append(found.condition)
                parts.append((2 * found.value / length, basis_of(weight)))
            if len(parts) < len(weights):
                continue
            nonzero = [(amplitude, basis) for amplitude, basis in parts if amplitude != 0]
            if not nonzero:
                if g - constant == 0:
                    return Expansion(g, S.Zero, k, parts[0][1], 1, constant, and_(*conditions))
                continue
            if len(nonzero) == 1:
                amplitude, basis = nonzero[0]
                # (-1)**k sin(k x) from the shift: the sign into the coefficient
                factor, harmonic = basis.as_independent(x, as_Add=False)
                if not isinstance(basis, Add):
                    amplitude, basis = amplitude * factor, harmonic
                return Expansion(g, amplitude, k, basis, 1, constant, and_(*conditions))
            basis = add(*[amplitude * b_ for amplitude, b_ in nonzero])
            return Expansion(g, S.One, k, basis, 1, constant, and_(*conditions))
        return None

    _nesting += 1
    try:
        return attempt(systems, budget, native=True)
    finally:
        _nesting -= 1


def _ops(e: Expr) -> int:
    """The number of operations of ``e`` (``count_ops``)."""
    counted = e.count_ops()
    if not isinstance(counted, Integer):
        raise TypeError("count_ops gave %s" % (counted,))
    return int(counted)


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

    >>> from sympy_extras.core import Symbol, Integer, pi
    >>> from sympy_extras.functions import log, sin, cos
    >>> from sympy_extras.integrals.series import fourier_expansion
    >>> x = Symbol('x')
    >>> a = Symbol('a', positive=True)
    >>> e = fourier_expansion(log(2*sin(x/2)), x, 0, 2*pi)
    >>> e.coefficient.subs(e.index, Integer(3)), e.basis.subs(e.index, Integer(3)), e.start, e.constant
    (-1/3, cos(3*x), 1, 0)
    >>> e = fourier_expansion(log(sin(x)), x, 0, pi)
    >>> e.basis.subs(e.index, Integer(1)), e.constant
    (cos(2*x), -log(2))
    >>> fourier_expansion(log(sin(x)), x, 0, 2*pi) is None
    True
    >>> e = fourier_expansion(1/(1 - 2*a*cos(x) + a**2), x, 0, pi, a < 1)
    >>> e.coefficient.subs(e.index, Integer(2)), e.constant
    (-2*a**2/(a**2 - 1), -1/(a**2 - 1))
    """
    g_, a_, b_ = to_expr(g), to_expr(a), to_expr(b)
    polynomial = g_.is_polynomial(x) is True
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
    if isinstance(h, Log):
        factor, argument = h.args[0].as_coeff_Mul()
        if factor != 1:
            if not factor.is_positive:
                return None
            h = log(argument)
            shift = log(factor)
    for entry in _FOURIER_TABLE:
        if entry.function != h:
            continue
        if not _within(a_, b_, entry.lower / scale, entry.upper / scale, assumptions):
            continue
        k = _fresh('k', integer=True, positive=True)
        replacement: dict[Basic, Basic] = {_u: scale * x, _k: k}
        coefficient = _evaluated(entry.coefficient, replacement)
        basis = _evaluated(entry.basis, replacement)
        return Expansion(g_, coefficient, k, basis, 1, entry.constant + shift)
    parametric = _parametric_entry(h, assumptions)
    if parametric is not None:
        constant, coefficient, basis, condition = parametric
        k = _fresh('k', integer=True, positive=True)
        rule: dict[Basic, Basic] = {_u: scale * x, _k: k}
        return Expansion(g_, _evaluated(coefficient, rule), k, _evaluated(basis, rule), 1, constant + shift,
                         condition)
    return None


def expansions(f: Expr, x: Symbol, a: Expr, b: Expr) -> Iterator[tuple[Expansion, Expr]]:
    """The ways of writing ``f`` as an expanded factor times a rest,
    each as ``(expansion, rest)``: every factor of ``f`` (and ``f``
    itself, last, since its formal power series is the slowest to find:
    the expansions are produced lazily) with a power series about 0, and
    the geometric series of exponentials on ``(0, oo)``.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, oo
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.integrals.series import expansions
    >>> x = Symbol('x')
    >>> [(e.factor, e.coefficient, rest) for e, rest in expansions(x/(exp(x) - 1), x, 0, oo)][0]
    (1/(exp(x) - 1), 1, x)
    """
    f = to_expr(f)
    a, b = to_expr(a), to_expr(b)
    factors = list(Mul.make_args(f))
    # the factors first, the whole integrand last (its formal power
    # series is the slowest to find)
    candidates: list[Expr] = list(factors)
    if len(factors) > 1:
        candidates.append(f)
    seen: set[Expr] = set()
    for g in candidates:
        if not g.has(x) or g in seen or g.is_polynomial(x) is True or monomial(g, x) is not None:
            continue
        seen.add(g)
        rest = f / g
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
    value = attempt(lambda: limit(g, x, S.Zero, '+'), settings.timeout, native=True)
    return value is not None and not value.has(oo, -oo, zoo, nan, Limit)


def moments(expansion: Expansion, rest: Expr, x: Symbol, a: Expr, b: Expr,
            assumptions: Assumptions = None) -> Optional[Expr]:
    """``Integral(basis(j) * rest, (x, a, b))`` as a closed form in the
    index ``j`` (a symbolic nonnegative integer), or ``None``; computed by
    :func:`~sympy_extras.integrals.definite.definite_integral`."""
    from .definite import definite_integral
    j = expansion.index
    facts: list[Boolean] = [element(j, Naturals0) if expansion.start == 0 else element(j, Naturals)]
    facts.extend(_statements(assumptions))
    global _nesting
    _nesting += 1
    try:
        integrand = expansion.basis * rest
        value = attempt(lambda: definite_integral(integrand, (x, to_expr(a), to_expr(b)), facts, special_values=False),
                        settings.timeout, native=True)
    finally:
        _nesting -= 1
    if value is None or _bad_value(value):
        return None
    return value


def _roots_of(p: Poly) -> Optional[dict[Expr, int]]:
    """The roots of ``p`` in closed form with their multiplicities."""
    found = roots(p)
    return found if isinstance(found, dict) else None


def _hypergeometric_sum(term: Expr, j: Dummy, start: int) -> Optional[Expr]:
    """The sum from ``start`` of a hypergeometric term as a ``hyper``
    expanded by ``hyperexpand``."""
    ratio = attempt(lambda: hypersimp(term, j), settings.timeout, native=True)
    if ratio is None:
        return None
    numerator, denominator = ratio.as_numer_denom()
    try:
        p_num = Poly(numerator, j)
        p_den = Poly(denominator, j)
    except (BasePolynomialError, ValueError, TypeError, NotImplementedError):  # not a polynomial ratio
        return None
    num_roots = _roots_of(p_num)
    den_roots = _roots_of(p_den)
    if num_roots is None or den_roots is None or sum(num_roots.values()) != p_num.degree() or sum(den_roots.values()) != p_den.degree():
        return None
    first = term.subs(j, Integer(start))
    z = p_num.LC() / p_den.LC()
    upper = [-r + start for r, m in num_roots.items() for _ in range(m)]
    lower = [-r + start for r, m in den_roots.items() for _ in range(m)]
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
    if len(upper) == len(lower) + 1 and ask(abs_(z) < 1) is not True:
        return None
    series = attempt(lambda: hyperexpand(first * hyper(upper, lower, z)), settings.timeout, native=True)
    if series is None or series.has(Hyper):
        return None
    return series


def _linear_partial_fractions(term: Expr, j: Dummy) -> Optional[list[tuple[Expr, Expr, int]]]:
    """The full partial fraction decomposition of a proper rational
    function of ``j`` as ``(c, r, m)`` for the terms ``c/(j - r)**m``, the
    roots ``r`` of the denominator in closed form; ``None`` when there is
    a polynomial part or a root has no closed form. The coefficient of
    ``1/(j - r)**i`` for a root of multiplicity ``m`` is the Taylor
    coefficient of order ``m - i`` at ``r`` of ``term*(j - r)**m``
    (SymPy's ``apart(full=True)`` is not ported)."""
    numerator, denominator = cancel(term).as_numer_denom()
    try:
        p_num = Poly(numerator, j)
        p_den = Poly(denominator, j)
    except (BasePolynomialError, ValueError, TypeError, NotImplementedError):
        return None
    if p_num.degree() >= p_den.degree():
        return None
    found = _roots_of(p_den)
    if found is None or sum(found.values()) != p_den.degree():
        return None
    leading = p_den.LC()
    terms: list[tuple[Expr, Expr, int]] = []
    for r, m in found.items():
        others = mul(leading, *[(j - s)**n for s, n in found.items() if s != r])
        if m == 1:
            # the residue numerator(r)/denominator'(r)
            c = cancel(numerator.subs(j, r) / others.subs(j, r))
            terms.append((c, r, 1))
            continue
        h = numerator / others
        for i in range(m, 0, -1):
            order = m - i
            derivative = diff(h, j, order) if order else h
            c = cancel(derivative.subs(j, r) / factorial(Integer(order)))
            if c != 0:
                terms.append((c, r, i))
    return terms


def _rational_sum(term: Expr, j: Dummy, start: int) -> Optional[Expr]:
    """The sum from ``start`` of a rational function of ``j`` by partial
    fractions: Hurwitz zeta values (polygamma functions) for the multiple
    poles and digamma values for the simple poles, whose coefficients
    must add up to zero for convergence (``summation`` returns ``nan`` for
    ``Sum(1/(j*(4*j**2 - 1)))``, whose partial fractions diverge
    separately)."""
    if term.is_rational_function(j) is not True:
        return None
    parts = attempt(lambda: _linear_partial_fractions(term, j), settings.timeout, native=True)
    if parts is None:
        return None
    total: Expr = S.Zero
    simple: Expr = S.Zero
    for c, root, m in parts:
        if c.has(j):
            return None
        # 1/(j - root)^m = 1/(i + r)^m with i = j - start
        r = -root + start
        if r.is_integer and isinstance(r <= 0, BooleanTrue):
            return None
        if m >= 2:
            total += c * zeta(m, r)
        else:
            simple += c
            total -= c * polygamma(0, r)
    if cancel(simple) != 0:
        return None
    return _polygamma_form(total)


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
        outer, inner = summand.as_independent(j, as_Add=False)
        for piece in Add.make_args(expand_mul(inner.rewrite(Exp))):
            pieces.append((outer, piece))
    total: Expr = S.Zero
    for outer, summand in pieces:
        coefficient: Expr = outer
        base: Expr = S.One
        order: Optional[int] = None
        for factor_ in Mul.make_args(summand):
            if not factor_.has(j):
                coefficient *= factor_
                continue
            if isinstance(factor_, Pow) and factor_.base == j:
                if order is not None or not isinstance(factor_.exp, Integer) or factor_.exp >= 0:
                    return None
                order = -int(factor_.exp)
                continue
            if isinstance(factor_, Pow):
                b_, e_ = factor_.base, factor_.exp
            elif isinstance(factor_, Exp):
                b_, e_ = E, factor_.args[0]
            else:
                return None
            p = e_.coeff(j)
            q = expand_mul(e_ - p * j)
            if b_.has(j) or p.has(j) or q.has(j):
                return None
            base *= b_**p
            coefficient *= b_**q
        if order is None or ask(abs_(base) <= 1, assumptions) is not True:
            return None
        if order == 1 and ask(ne(base, 1), assumptions) is not True:
            return None
        value: Expr = polylog(order, base)
        for i in range(1, start):
            value -= base**i / Integer(i)**order
        total += coefficient * value
    return total


def sum_series(term: Expr, j: Dummy, start: int, assumptions: Assumptions = None) -> Optional[Expr]:
    """``Sum(term, (j, start, oo))`` in closed form by the native
    :func:`~sympy_extras.concrete.summations.summation`, through
    polylogarithms for the trigonometric series ``Sum(sin(j*t)/j**s)`` of
    the Clausen functions, by partial fractions, by
    :func:`~sympy_extras.concrete.eulersums.polygamma_series`,
    :func:`~sympy_extras.concrete.zeilberger.zeilberger_sum` or as a
    hypergeometric series; ``None`` when none closes it.

    Examples
    ========

    >>> from sympy_extras.core import Dummy
    >>> from sympy_extras.integrals.series import sum_series
    >>> j = Dummy('j', integer=True, nonnegative=True)
    >>> sum_series((-1)**j/(2*j + 1)**2, j, 0)
    Catalan
    >>> sum_series(1/(j + 1)**2, j, 0)
    pi**2/6
    """
    total = attempt(lambda: summation(term, (j, Integer(start), oo)), settings.timeout, native=True)
    if total is not None and total.has(Piecewise):
        # the branch of a parametric sum (|a| < 1 for a geometric series)
        # which the assumptions settle
        folded = piecewise_fold(total)
        total = _settled_branch(folded, assumptions) if isinstance(folded, Piecewise) else None
    if total is not None and total.has(ExpPolar):
        # polylog(2, a*exp_polar(I*pi)) is polylog(2, -a) off the cut [1, oo)
        rule: dict[Basic, Basic] = {node: exp(node.args[0]) for node in total.atoms(ExpPolar)
                                    if isinstance(node, ExpPolar)}
        total = _evaluated(total, rule)
    if total is not None and not total.has(Sum, nan, zoo, Piecewise) and not total.has(oo) \
            and not has_integral(total):
        return _polygamma_form(_settled_abs(total))
    if term.has(Sin, Cos, Exp, Pow):
        clausen = attempt(lambda: _polylog_sum(term, j, start, assumptions), settings.timeout, native=True)
        if clausen is not None:
            return clausen
    rational = _rational_sum(term, j, start)
    if rational is not None:
        return rational
    n = _fresh('n', integer=True, positive=True)
    shifted = term.subs(j, n)
    value = attempt(lambda: polygamma_series(shifted, n, max(start, 1)), settings.timeout, native=True)
    if value is not None and start >= 1:
        return value
    closed = attempt(lambda: zeilberger_sum(shifted, (n, Integer(start), oo), special_values=False),
                     settings.timeout, native=True)
    if isinstance(closed, Expr) and not closed.has(Sum, nan, zoo, oo):
        return closed
    return _hypergeometric_sum(term, j, start)


def _settled_abs(value: Expr) -> Expr:
    """``value`` with the absolute values of numbers whose argument
    expands to a real number evaluated (the native summation of
    ``(-1)**j/(2*j + 1)**2`` writes Catalan's constant as the absolute
    value of an unexpanded sum of imaginary parts)."""
    rule: dict[Basic, Basic] = {}
    for node in value.atoms(Abs):
        if isinstance(node, Abs) and node.args[0].is_number and node.args[0].has(I):
            argument = expand_mul(node.args[0])
            if not argument.has(I):
                rule[node] = abs_(argument)
    return _evaluated(value, rule) if rule else value


def _settled_branch(total: Piecewise, assumptions: Assumptions) -> Optional[Expr]:
    """The value of the first branch whose condition the assumptions
    settle as true, ``None`` when none is settled (a branch refuted is
    skipped)."""
    for pair in total.args:
        verdict = ask(pair.cond, assumptions)
        if verdict is True:
            return pair.expr
        if verdict is None:
            return None
    return None


def _justified(term: Expr, j: Dummy, start: int, assumptions: Assumptions) -> Optional[bool]:
    """Whether the series of the absolute values of the integrated terms
    converges (dominated convergence justifies the interchange); ``None``
    when undecided."""
    verdict = attempt(lambda: sum_convergence(abs_(term), j, assumptions, max(start, 1)), settings.timeout,
                      native=True)
    if verdict is None:
        return None
    if isinstance(verdict, BooleanTrue):
        return True
    if isinstance(verdict, BooleanFalse):
        return False
    return ask(verdict, assumptions)


class _Harmonics:
    """A trigonometric series ``constant + Sum(coefficient(k) * (cosine(k) *
    cos(frequency(k) * x) + sine(k) * sin(frequency(k) * x)), (k, start, oo))``,
    or a single harmonic (``index`` ``None``, the sum being the one term)."""

    __slots__ = ('constant', 'coefficient', 'index', 'start', 'cosine', 'sine', 'frequency')

    def __init__(self, constant: Expr, coefficient: Expr, index: Optional[Dummy], start: int,
                 cosine: Expr, sine: Expr, frequency: Expr) -> None:
        self.constant: Final[Expr] = constant
        self.coefficient: Final[Expr] = coefficient
        self.index: Final[Optional[Dummy]] = index
        self.start: Final[int] = start
        self.cosine: Final[Expr] = cosine
        self.sine: Final[Expr] = sine
        self.frequency: Final[Expr] = frequency

    def at(self, value: Expr, index: Expr) -> Expr:
        """``value`` with the index replaced by ``index``."""
        if self.index is None:
            return value
        return value.subs(self.index, index)


def _harmonic_parts(basis: Expr, x: Symbol) -> Optional[tuple[Expr, Expr, Expr]]:
    """``(cosine, sine, frequency)`` of ``basis = cosine * cos(frequency * x)
    + sine * sin(frequency * x)``."""
    cosine: Expr = S.Zero
    sine: Expr = S.Zero
    frequency: Optional[Expr] = None
    for term in Add.make_args(basis):
        amplitude, harmonic = term.as_independent(x)
        if not isinstance(harmonic, (Sin, Cos)):
            return None
        found = monomial(harmonic.args[0], x)
        if found is None or found[1] != 1:
            return None
        if frequency is None:
            frequency = found[0]
        elif frequency != found[0]:
            return None
        if isinstance(harmonic, Cos):
            cosine += amplitude
        else:
            sine += amplitude
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
    if ask(element(frequency, Naturals), assumptions) is not True:
        return None
    return _Harmonics(S.Zero, S.One, None, 0, cosine, sine, frequency)


def _linear(frequency: Expr, k: Dummy) -> Optional[tuple[int, int]]:
    """``(p, q)`` with ``frequency = p*k + q``, integers with ``p > 0``."""
    p = expand_mul(frequency).coeff(k)
    q = expand_mul(frequency - p * k)
    if not isinstance(p, Integer) or not isinstance(q, Integer) or int(p) <= 0:
        return None
    return (int(p), int(q))


def _primitive(s: _Harmonics, x: Symbol, a: Expr, b: Expr) -> Expr:
    """``Integral(coefficient(k) * (cosine(k) * cos(frequency(k) * x) + ...), (x, a, b))``
    as a function of the index (the frequency is nonzero)."""
    w = s.frequency
    integral = s.cosine * (sin(w * b) - sin(w * a)) / w + s.sine * (cos(w * a) - cos(w * b)) / w
    return expand_mul(s.coefficient * integral)


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
    that occur (for integer indices ``sin(2*pi*(j - k))`` evaluates to
    zero)."""
    cosines = (s1.cosine != 0 and s2.cosine != 0) or (s1.sine != 0 and s2.sine != 0)
    mixed = (s1.cosine != 0 and s2.sine != 0) or (s1.sine != 0 and s2.cosine != 0)
    for m in (expand_mul(s1.frequency + s2.frequency), expand_mul(s1.frequency - s2.frequency)):
        if cosines and expand_mul(sin(m * b) - sin(m * a)) != 0:
            return False
        if mixed and expand_mul(cos(m * a) - cos(m * b)) != 0:
            return False
    return True


#: a term of the diagonal series: ``(term, index, start)``, ``index``
#: ``None`` for a single term
_Diagonal = tuple[Expr, Optional[Dummy], int]


def _diagonal(s1: _Harmonics, s2: _Harmonics, length: Expr, assumptions: Assumptions) -> Optional[_Diagonal]:
    """The terms of the double series of ``Integral(f1 * f2)`` with equal
    frequencies (the others vanish by orthogonality) as ``(term, index,
    start)`` of a single series (``index`` ``None`` for a single term);
    ``None`` when the pairing of the indices is not found."""
    if s1.index is None and s2.index is not None:
        s1, s2 = s2, s1
    first, second = s1, s2

    def product(j: Expr, k: Expr) -> Expr:
        # Integral(cos(w x)**2) = Integral(sin(w x)**2) = length/2 on the
        # range, the mixed products integrate to zero
        inner = first.at(first.cosine, j) * second.at(second.cosine, k) \
            + first.at(first.sine, j) * second.at(second.sine, k)
        return first.at(first.coefficient, j) * second.at(second.coefficient, k) * inner * length / 2

    if s1.index is None:
        return None
    if s2.index is None:
        # the harmonic n of the second factor picks the term with
        # frequency1(j) = p j + q = n
        found = _linear(s1.frequency, s1.index)
        if found is None:
            return None
        p, q = found
        j0 = (s2.frequency - q) / p
        if j0.is_integer is False:
            return (S.Zero, None, 0)
        if j0.is_integer is not True:
            return None
        inside = ask(j0 >= s1.start, assumptions)
        if inside is False:
            return (S.Zero, None, 0)
        if inside is not True:
            return None
        return (product(j0, S.Zero), None, 0)
    return _paired(s1, s2, product)


def _paired(s1: _Harmonics, s2: _Harmonics, product: Callable[[Expr, Expr], Expr]) -> Optional[_Diagonal]:
    """The single series of the pairs ``(j, k)`` with ``frequency1(j) =
    frequency2(k)``."""
    j, k = s1.index, s2.index
    if j is None or k is None:
        return None
    first = _linear(s1.frequency, j)
    second = _linear(s2.frequency, k)
    if first is None or second is None:
        return None
    (p1, q1), (p2, q2) = first, second
    if p1 % p2 == 0 and (q1 - q2) % p2 == 0:
        # k = (p1 j + q1 - q2)/p2 >= start2 from j >= (p2 start2 - q1 + q2)/p1
        paired = Rational(p1, p2) * j + Rational(q1 - q2, p2)
        start = max(s1.start, -((q1 - q2 - p2 * s2.start) // p1))
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
    length = b - a
    total = s1.constant * s2.constant * length
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
        return total + term
    value = sum_series(term, index, start, assumptions)
    if value is None or value.has(index):
        return None
    return total + value


def _fourier_factors(f: Expr, x: Symbol) -> tuple[Expr, list[Expr]]:
    """The constant of ``f`` and its factors depending on ``x`` with
    multiplicity (a square counted twice)."""
    constant: Expr = S.One
    factors: list[Expr] = []
    for t in Mul.make_args(f):
        if not t.has(x):
            constant *= t
        elif isinstance(t, Pow) and t.exp == 2 and monomial(t.base, x) is None:
            factors.extend([t.base] * 2)
        else:
            factors.append(t)
    return (constant, factors)


def _resonant(expansion: Expansion, rest: Expr, x: Symbol, assumptions: Assumptions) -> bool:
    """Whether a harmonic of the rest may share a frequency with the
    series, so that the moments computed for the symbolic index by an
    antiderivative would miss the resonant term (``Integral(cos(2*k*x) *
    cos(x)**2, (x, 0, pi))`` is ``pi/4`` at ``k = 1`` and zero otherwise):
    a rest with trigonometric functions of ``x`` is safe only when it is
    a single harmonic whose frequency the series never takes."""
    if not any(t.has(x) for t in rest.atoms(Sin, Cos, Tan)):
        return False
    parts = _harmonic_parts(rest, x)
    series = _harmonic_parts(expansion.basis, x)
    if parts is None or series is None:
        return True
    found = _linear(series[2], expansion.index)
    if found is None:
        return True
    p, q = found
    j0 = (parts[2] - q) / p
    if j0.is_integer is False:
        return False
    return not (j0.is_integer is True and ask(j0 < expansion.start, assumptions) is True)


def _verified(value: Expr, f: Expr, x: Symbol, a: Expr, b: Expr, assumptions: Assumptions) -> bool:
    """Whether the numerical check of ``settings.numerical_checks``
    confirms ``value`` as ``Integral(f, (x, a, b))``: the quadratures of
    :func:`~sympy_extras.integrals.definite.verify_numerically`."""
    if not settings.numerical_checks:
        return False
    from .definite import verify_numerically
    return verify_numerically(value, f, x, a, b, assumptions) is True


def _termwise(expansion: Expansion, rest: Expr, x: Symbol, a: Expr, b: Expr,
              assumptions: Assumptions) -> Optional[Expr]:
    """``Integral(factor * rest, (x, a, b))`` from the Fourier series of the
    factor integrated termwise against the rest: the mean times the
    integral of the rest plus the series of the moments."""
    from .definite import definite_integral
    if _resonant(expansion, rest, x, assumptions):
        return None
    mean: Expr = S.Zero
    if expansion.constant != 0:
        found = attempt(lambda: definite_integral(rest, (x, a, b), assumptions, special_values=False),
                        settings.timeout, native=True)
        if found is None or _bad_value(found):
            return None
        mean = found
    moment = moments(expansion, rest, x, a, b, assumptions)
    if moment is None:
        return None
    j = expansion.index
    term = expansion.coefficient * moment
    value = sum_series(term, j, expansion.start, assumptions)
    if value is None or value.has(j):
        return None
    total = expansion.constant * mean + value
    justified = _justified(term, j, expansion.start, assumptions)
    if justified is False:
        return None
    if justified is None and not _verified(total, expansion.factor * rest, x, a, b, assumptions):
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
    :func:`~sympy_extras.integrals.definite.definite_integral` for the symbolic
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

    >>> from sympy_extras.core import Symbol, pi
    >>> from sympy_extras.functions import log, sin, cos
    >>> from sympy_extras.integrals.series import fourier_integral
    >>> x = Symbol('x')
    >>> n = Symbol('n', integer=True, positive=True)
    >>> fourier_integral(log(sin(x)), x, 0, pi)
    ConditionalValue(-pi*log(2))
    >>> fourier_integral(log(sin(x))*log(cos(x)), x, 0, pi/2).value.expand()
    -pi**3/48 + pi*log(2)**2/2
    >>> fourier_integral(log(2*sin(x/2))*cos(n*x), x, 0, 2*pi)
    ConditionalValue(-pi/n)
    """
    f_, a_, b_ = to_expr(f), to_expr(a), to_expr(b)
    if ask(a_ < b_, assumptions) is not True:
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
                if e is None or g.is_polynomial(x) is True else e for g, e in zip(factors, expanded)]
    if all(e is None or any(e is other for other in expanded) for e in computed):
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
    condition = and_(*[e.condition for e in expanded if e is not None])
    value: Optional[Expr] = None
    first = expanded[0]
    if len(factors) == 1 and first is not None:
        s = _harmonics(first, x)
        if s is not None:
            integrated = _integrated(s, x, a_, b_, assumptions)
            if integrated is not None:
                value = s.constant * (b_ - a_) + integrated
    elif len(factors) == 2:
        series = [_harmonics(e, x) if e is not None else _harmonic(g, x, assumptions)
                  for e, g in zip(expanded, factors)]
        s1, s2 = series
        if s1 is not None and s2 is not None:
            value = _fourier_product(s1, s2, x, a_, b_, assumptions)
    if value is not None:
        return ConditionalValue(_polygamma_form(tidy(constant * value, assumptions)), condition)
    # an expanded factor against the rest of the integrand, the
    # polynomial factors last (their moments are the easiest)
    seen: set[Expr] = set()
    ordered = sorted(zip(factors, expanded), key=lambda pair: pair[0].is_polynomial(x) is True)
    for g, e in ordered:
        if e is None or g in seen or (expand is not None and g != expand):
            continue
        seen.add(g)
        found = _termwise(e, f_ / g, x, a_, b_, assumptions)
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

    >>> from sympy_extras.core import Symbol, oo, pi
    >>> from sympy_extras.functions import log, exp, sin
    >>> from sympy_extras.integrals.series import series_integral
    >>> x = Symbol('x')
    >>> p = Symbol('p', positive=True)
    >>> series_integral(log(1 + x)/x, x, 0, 1)
    ConditionalValue(pi**2/12)
    >>> series_integral(x**p*log(x)/(1 - x), x, 0, 1)
    ConditionalValue(-polygamma(1, p + 1))
    >>> series_integral(x*log(sin(x)), x, 0, pi)
    ConditionalValue(-pi**2*log(2)/2)
    """
    f_, a_, b_ = to_expr(f), to_expr(a), to_expr(b)
    fourier = fourier_integral(f_, x, a_, b_, assumptions, expand)
    if fourier is not None:
        return fourier
    if a_ != 0 or (b_ != oo and ask(b_ > 0, assumptions) is not True):
        return None
    if b_ not in (oo, S.One):
        # (0, c): x = c t
        t = _fresh('t', positive=True)
        return series_integral(f_.subs(x, b_ * t) * b_, t, S.Zero, S.One, assumptions, None)
    for expansion, rest in expansions(f_, x, a_, b_):
        if expand is not None and expansion.factor != expand:
            continue
        moment = moments(expansion, rest, x, a_, b_, assumptions)
        if moment is None:
            continue
        j = expansion.index
        term = expansion.coefficient * moment
        value = sum_series(term, j, expansion.start, assumptions)
        if value is None or j in value.free_symbols:
            continue
        justified = _justified(term, j, expansion.start, assumptions)
        if justified is False:
            continue
        if justified is None and not _verified(value, f_, x, a_, b_, assumptions):
            continue
        return ConditionalValue(_polygamma_form(tidy(value, assumptions)))
    return None


def _polygamma_form(value: Expr) -> Expr:
    """Hurwitz zeta values ``zeta(n, a)`` at integers ``n >= 2`` (also
    written ``lerchphi(1, n, a)`` by ``summation``) as polygamma
    functions, and Dirichlet eta values as zeta values."""
    if value.has(DirichletEta):
        value = value.rewrite(Zeta)
    replacement: dict[Basic, Basic] = {}
    for node in value.atoms(LerchPhi):
        if node.args[0] == 1:
            replacement[node] = zeta(as_expr(node.args[1]), as_expr(node.args[2]))
    if replacement:
        value = _evaluated(value, replacement)
        replacement = {}
    for node in value.atoms(Zeta):
        if len(node.args) == 2:
            n, a = as_expr(node.args[0]), as_expr(node.args[1])
            if isinstance(n, Integer) and int(n) >= 2:
                # zeta(n, a) = (-1)^n psi^(n-1)(a) / (n - 1)!
                replacement[node] = S.NegativeOne**n * polygamma(n - 1, a) / factorial(n - 1)
    return cancel(_evaluated(value, replacement)) if replacement else value
