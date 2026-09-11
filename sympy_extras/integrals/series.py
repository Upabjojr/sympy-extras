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
   `(0, \\infty)`;
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

References
==========

.. [GR] I. S. Gradshteyn, I. M. Ryzhik, *Table of Integrals, Series, and
   Products*, 7th ed., Academic Press, 2007, sections 4.22–4.29.
.. [Borwein] J. M. Borwein, D. M. Bradley, R. E. Crandall, *Computational
   strategies for the Riemann zeta function*, Journal of Computational
   and Applied Mathematics 121 (2000) 247–296.
"""
from __future__ import annotations

from typing import Iterator, Optional

from sympy.concrete.summations import Sum, summation
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.numbers import Integer, nan, oo, zoo
from sympy.functions.combinatorial.factorials import factorial
from sympy.series.limits import limit
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp
from sympy.functions.special.hyper import hyper
from sympy.functions.special.zeta_functions import zeta, lerchphi
from sympy.functions.special.gamma_functions import polygamma
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import Boolean, true
from sympy.functions.elementary.piecewise import Piecewise
from sympy.simplify.hyperexpand import hyperexpand
from sympy.simplify.simplify import hypersimp
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

__all__ = ['Expansion', 'expansions', 'moments', 'sum_series', 'series_integral']


class Expansion:
    """A factor of the integrand expanded as a series: the ``terms`` are
    ``coefficient(j) * basis(j)``, ``j`` running over the integers from
    ``start``, with ``basis(j)`` a power ``x**(period*j + offset)`` or an
    exponential ``exp(-scale*j*x)``.

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
    """

    def __init__(self, factor: Expr, coefficient: Expr, index: Dummy, basis: Expr, start: int) -> None:
        self.factor = factor
        self.coefficient = coefficient
        self.index = index
        self.basis = basis
        self.start = start

    def __repr__(self) -> str:
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
    value = attempt(lambda: definite_integral(expansion.basis * rest, (x, a, b), facts), settings.timeout)
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


def sum_series(term: Expr, j: Dummy, start: int) -> Optional[Expr]:
    """``Sum(term, (j, start, oo))`` in closed form by :func:`sympy.summation`,
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
    if isinstance(total, Expr) and not total.has(Sum, nan, zoo) and not total.has(oo):
        return _polygamma_form(as_expr(total))
    n = Dummy('n', integer=True, positive=True)
    shifted = as_expr(term.subs(j, n))
    value = attempt(lambda: polygamma_series(shifted, n, max(start, 1)), settings.timeout)
    if value is not None and start >= 1:
        return value
    closed = attempt(lambda: zeilberger_sum(shifted, (n, start, oo)), settings.timeout)
    if isinstance(closed, Expr) and not closed.has(Sum, nan, zoo, oo):
        return as_expr(closed)
    return _hypergeometric_sum(term, j, start)


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
        (mapped onto ``(0, 1)``).
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
    """
    from .definite import verify_numerically
    f_, a_, b_ = as_expr(f), as_expr(a), as_expr(b)
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
        value = sum_series(term, j, expansion.start)
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
    functions."""
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
