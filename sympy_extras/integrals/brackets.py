"""Ramanujan's master theorem and the method of brackets: Mellin
transforms from power series.

If `f(x) = \\sum_{k \\ge 0} \\varphi(k) (-x)^k / k!`, **Ramanujan's master
theorem** states that

.. math::

    \\int_0^\\infty x^{s-1} f(x)\\, dx = \\Gamma(s)\\, \\varphi(-s),

where `\\varphi(-s)` is the analytic continuation of the coefficient
function to `k = -s` [Hardy]_, [Amdeberhan]_. Hardy's proof needs growth
conditions on `\\varphi`; in practice the formula holds for `s` in the
strip where the integral converges, and this is what is returned: the
strip is read off the behaviour of `f` at `0` and at `\\infty`.

The **method of brackets** [Gonzalez]_ extends this to products. A series
`\\sum_n \\varphi(n) (-1)^n/n!\\, x^{a n + b}` integrated against `x^{s-1}`
on `(0, \\infty)` is written with the *bracket* `\\langle a n + b + s
\\rangle`, and the rule for one bracket replaces the sum by
`\\varphi(n^*) \\Gamma(-n^*) / |a|` with `n^*` the solution of `a n + b + s =
0`: this is the master theorem again. For a product of two series there
are two indices and one bracket; eliminating one index leaves a series
in the other, which is hypergeometric and summed in closed form. Which
index to eliminate is decided by convergence: the rule of the method is
to keep the series which converge, and the method is a heuristic --
its rules are not proved in general, and a result is only as good as
the check made of it. Here the series left after the bracket is summed
exactly (:func:`sympy_extras.concrete.summations.summation`, as a
hypergeometric function, expanded when possible), so the value is right
whenever the summation converges.

The coefficient functions come from the power series of the function:
the coefficients of one residue class are a hypergeometric term, its
ratio guessed from the first coefficients and proved by the differential
equation it implies (:func:`formal_series_terms`, in place of SymPy's
formal power series ``sympy.series.formal.fps``), a closed form with
factorials and rising factorials, rewritten with the gamma function so
that `k` can be replaced by `-s`. The strip of convergence is read off
the order of the series at `0` and of the behaviour at infinity: a power
(of the series of ``f(1/y)``, powers of logarithms aside), a decay faster
than every power, or an oscillation ``x**q sin(a x**p)`` (Dirichlet's
test).

The module works on the objects of the core (:mod:`sympy_extras.core`, the
functions of :mod:`sympy_extras.functions`) with the native ``ask``,
series, limits, summation, guessing, ``gammasimp`` and ``hyperexpand``,
and imports no SymPy, nor goes through it at run time.

Examples
========

>>> from sympy_extras.core import symbols
>>> from sympy_extras.functions import exp, sin, atan
>>> from sympy_extras.integrals.brackets import ramanujan_master_theorem, mellin_transform_series
>>> x, s = symbols('x s')
>>> ramanujan_master_theorem(exp(-x), x, s)
MellinTransform(gamma(s), (0, oo))
>>> ramanujan_master_theorem(atan(x), x, s)
MellinTransform(-pi/(2*s*cos(pi*s/2)), (-1, 0))
>>> mellin_transform_series(exp(-x)*sin(x), x, s)
MellinTransform(2**(s + 1)*gamma(s/4 + 1/2)*gamma(s/2 + 1/2)/(8*gamma(1 - s/4)), (-1, oo))

References
==========

.. [Hardy] G. H. Hardy, *Ramanujan: Twelve lectures on subjects
   suggested by his life and work*, Cambridge University Press, 1940,
   chapter XI.
.. [Amdeberhan] T. Amdeberhan, O. Espinosa, I. Gonzalez, M. Harrison,
   V. H. Moll, A. Straub, *Ramanujan's Master Theorem*, The Ramanujan
   Journal 29 (2012), pp. 103-120.
.. [Gonzalez] I. Gonzalez, V. H. Moll, *Definite integrals by the method
   of brackets. Part 1*, Advances in Applied Mathematics 45 (2010),
   pp. 50-73.
"""
from __future__ import annotations

from typing import Final, Optional

from sympy_extras._timeout import attempt
from sympy_extras.assumptions.ask import Assumptions, ask
from sympy_extras.concrete.expr_with_limits import Sum
from sympy_extras.concrete.summations import summation
from sympy_extras.core.add import Add
from sympy_extras.core.coreerrors import PoleError
from sympy_extras.core.expr import Expr, as_expr
from sympy_extras.core.limits import limit
from sympy_extras.core.logic import BooleanTrue, true
from sympy_extras.core.mod import Mod
from sympy_extras.core.mul import Mul
from sympy_extras.core.numbers import Integer, oo
from sympy_extras.core.order import Order
from sympy_extras.core.power import Pow
from sympy_extras.core.relational import Eq, Equality
from sympy_extras.core.singleton import S
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.combinatorial.factorials import factorial
from sympy_extras.functions.elementary.complexes import abs_ as Abs
from sympy_extras.functions.elementary.miscellaneous import max_ as Max, min_ as Min
from sympy_extras.functions.elementary.piecewise import Piecewise, piecewise
from sympy_extras.functions.special.gamma_functions import Gamma, gamma
from sympy_extras.functions.special.hyper import Hyper
from sympy_extras.settings import settings
from sympy_extras.simplify.gammasimp import gammasimp
from sympy_extras.simplify.hyperexpand import hyperexpand
from sympy_extras.simplify.powsimp import powsimp
from sympy_extras.simplify.simplify import simplify
from .mellin import MellinTransform, monomial

__all__ = ['SeriesCoefficient', 'FormalSeriesTerms', 'formal_series_terms', 'taylor_coefficient', 'ramanujan_master_theorem',
           'method_of_brackets', 'mellin_transform_series']

#: ``(g_i(y), beta_i, gamma_i)``: a factor ``g_i(beta_i x**gamma_i)`` of a product
Factor = tuple[Expr, Expr, Expr]


class SeriesCoefficient:
    """The power series of a function about `0` in the form
    `f(x) = x^r \\sum_{j \\ge 0} c(j)\\, (x^m)^j`.

    Attributes
    ==========

    coefficient : Expr
        `c(j)`, a closed form in the index.
    index : Dummy
        The index `j` (a nonnegative integer).
    period : int
        `m`, the step of the exponents (2 for an even or odd function).
    offset : Expr
        `r`, the exponent of the first term.
    """

    __slots__ = ('coefficient', 'index', 'period', 'offset')

    def __init__(self, coefficient: Expr, index: Dummy, period: int, offset: Expr) -> None:
        self.coefficient: Final[Expr] = coefficient
        self.index: Final[Dummy] = index
        self.period: Final[int] = period
        self.offset: Final[Expr] = offset

    def __repr__(self) -> str:
        return "SeriesCoefficient(%s, %s, period=%d, offset=%s)" % (self.coefficient, self.index,
                                                                    self.period, self.offset)

    def phi(self) -> Optional[Expr]:
        """`\\varphi(j) = (-1)^j j!\\, c(j)` of the master theorem, with the
        gamma function in place of the factorials; ``None`` when a factor
        `(-1)^j` remains (no Mellin transform: the series has positive
        coefficients growing like those of `e^x`).

        >>> from sympy_extras.core import symbols
        >>> from sympy_extras.functions import log
        >>> from sympy_extras.integrals.brackets import taylor_coefficient
        >>> x = symbols('x')
        >>> c = taylor_coefficient(log(1 + x), x)
        >>> c.phi().subs(c.index, 3)
        -2
        """
        j = self.index
        value = powsimp(S.NegativeOne**j * factorial(j) * self.coefficient, combine='exp')
        value = gammasimp(value.rewrite(Gamma))
        if value.has(S.NegativeOne**j) or _has_sign_power(value, j):
            return None
        return value


class FormalSeriesTerms:
    """The power series of a function (:func:`formal_series_terms`): the
    ``formula`` of the coefficient of ``x**index``, a ``Piecewise`` on the
    residue classes of the index or a single expression, valid from
    ``start`` on, and the ``independent`` terms, the polynomial part which
    the formula does not give."""

    __slots__ = ('formula', 'index', 'start', 'independent')

    def __init__(self, formula: Expr, index: Symbol, start: Expr, independent: Expr) -> None:
        self.formula: Final[Expr] = formula
        self.index: Final[Symbol] = index
        self.start: Final[Expr] = start
        self.independent: Final[Expr] = independent

    def __repr__(self) -> str:
        return "FormalSeriesTerms(%s, %s, start=%s, independent=%s)" % (self.formula, self.index, self.start,
                                                                         self.independent)


def _has_sign_power(e: Expr, j: Dummy) -> bool:
    """Whether ``e`` still has a power of a negative number with the
    index in the exponent."""
    for node in e.atoms(Pow):
        if isinstance(node, Pow) and node.base.is_negative is True and node.exp.has(j):
            return True
    return False


def _residue_class(formula: Expr, k: Symbol) -> Optional[tuple[Expr, int, int]]:
    """A ``Piecewise`` coefficient of :func:`formal_series_terms` read as ``(value, m, r)``:
    ``value`` for ``k = r mod m`` and zero otherwise; a plain formula is
    ``(formula, 1, 0)``. ``None`` when two residue classes are nonzero."""
    if not isinstance(formula, Piecewise):
        return (formula, 1, 0)
    found: Optional[tuple[Expr, int, int]] = None
    for pair in formula.args:
        value, condition = pair.expr, pair.cond
        if value == 0:
            continue
        if isinstance(condition, BooleanTrue) and found is None:
            return (value, 1, 0)
        if not isinstance(condition, Equality):
            return None
        lhs, rhs = condition.lhs, condition.rhs
        if not (isinstance(lhs, Mod) and lhs.args[0] == k):
            return None
        modulus, remainder = lhs.args[1], rhs
        if not isinstance(modulus, Integer) or not isinstance(remainder, Integer):
            return None
        if found is not None:
            return None
        found = (value, int(modulus), int(remainder))
    return found


def _replaced(e: Expr, old: Symbol, new: Expr) -> Optional[Expr]:
    """``e.subs(old, new)``, ``None`` when the substitution fails."""
    try:
        return e.subs(old, new)
    except (ValueError, TypeError):
        return None


#: the first coefficients read to find the exponents of the series
_PROBE_TERMS = 12
#: the terms of the residue class from which the recurrence is guessed
_CLASS_TERMS = 12
#: the most coefficients computed
_MAX_TERMS = 80


def _taylor_coefficients(f: Expr, x: Symbol, count: int) -> Optional[list[Expr]]:
    """The first ``count`` Taylor coefficients of ``f`` about 0 by the
    native series, ``None`` when ``f`` has no power series there (poles,
    fractional powers, logarithms) or the series is not found in time."""
    try:
        truncated = attempt(lambda: f.series(x, S.Zero, count).removeO(), settings.timeout, native=True)
    except (PoleError, NotImplementedError, ValueError, TypeError):
        return None
    if truncated is None or truncated.is_polynomial(x) is not True:
        return None
    expanded = truncated.expand()
    if expanded.is_polynomial(x) is not True:
        return None
    coefficients: list[Expr] = []
    for n in range(count):
        c = expanded.coeff(x, n) if n else expanded.subs(x, S.Zero)
        if c.has(x):
            return None
        coefficients.append(c)
    return coefficients


def _is_zero_constant(c: Expr) -> Optional[bool]:
    """Whether a constant is zero (``None`` when undecided)."""
    from sympy_extras.assumptions.analysis import certified_sign
    if c == 0:
        return True
    z = c.is_zero
    if z is not None:
        return z
    sign = certified_sign(c)
    return None if sign is None else sign == 0


def _support(coefficients: list[Expr]) -> Optional[list[int]]:
    """The exponents of the nonzero coefficients (``None`` when a
    coefficient is not decided zero or not)."""
    found: list[int] = []
    for n, c in enumerate(coefficients):
        z = _is_zero_constant(c)
        if z is None:
            return None
        if not z:
            found.append(n)
    return found


def _polynomial_of(coefficients: list[int], t: Symbol) -> Expr:
    """The polynomial of a list of integers, highest degree first."""
    value: Expr = Integer(0)
    for c in coefficients:
        value = value*t + c
    return value


def _apply_in_theta(polynomial: Expr, t: Symbol, g: Expr, x: Symbol, s: int, m: int) -> Expr:
    """``polynomial(J) g`` for the operator ``J = (x d/dx - s)/m``, which
    multiplies ``x**(s + m*i)`` by ``i``."""
    from sympy_extras.polys.polytools import Poly
    poly = Poly(polynomial, t)
    result: Expr = Integer(0)
    power = g
    for degree in range(poly.degree() + 1):
        coefficient = poly.coeff_monomial((degree,))
        if coefficient != 0:
            result = result + coefficient*power
        power = (x*power.diff(x) - s*power)/m
    return result


def _vanishes(e: Expr) -> bool:
    """Whether the expression is recognised as zero (cancellation of the
    expanded form, then the native simplification)."""
    from sympy_extras.polys.polytools import cancel
    expanded = attempt(lambda: cancel(e.expand()), settings.timeout, native=True)
    if expanded is not None and expanded == 0:
        return True
    simplified = attempt(lambda: simplify(e), settings.timeout, native=True)
    return simplified is not None and simplified == 0


def formal_series_terms(f: Expr, x: Symbol) -> Optional[FormalSeriesTerms]:
    """The power series of ``f`` about 0 when its nonzero coefficients
    form one residue class ``k = s (mod m)`` and are a hypergeometric term
    there: the coefficient of ``x**k`` in closed form with rising
    factorials, as a ``Piecewise`` on the class (a plain formula for
    ``m = 1``); ``None`` otherwise.

    The ratio of consecutive coefficients of the class is guessed from
    the first terms of the native series
    (:func:`sympy_extras.concrete.guess.guess_recurrence`, order one) and
    then proved: with ``Q(i) e(i + 1) = P(i) e(i)`` the series satisfies
    the differential equation ``Q(J - 1)(f - e(0) x**s) = x**m P(J) f`` in
    the operator ``J = (x d/dx - s)/m``, which is checked on ``f``
    symbolically; it determines every coefficient beyond the roots of
    ``Q``, which the computed terms cover. This replaces SymPy's
    ``fps`` (whose rational and hypergeometric algorithms find a
    differential equation by linear algebra first), for the single
    hypergeometric classes which :func:`taylor_coefficient` accepts.

    Examples
    ========

    >>> from sympy_extras.core import Integer, symbols
    >>> from sympy_extras.functions import exp, sin, atan
    >>> from sympy_extras.integrals.brackets import formal_series_terms
    >>> x = symbols('x')
    >>> terms = formal_series_terms(exp(-x), x)
    >>> terms.formula.subs(terms.index, Integer(3))
    -1/6
    >>> terms = formal_series_terms(sin(x), x)
    >>> [terms.formula.subs(terms.index, Integer(k)) for k in range(6)]
    [0, 1, 0, -1/6, 0, 1/120]
    >>> formal_series_terms(exp(x) + sin(x), x) is None
    True
    """
    from fractions import Fraction
    from sympy_extras.concrete.guess import guess_recurrence
    from sympy_extras.core.numbers import Rational
    from sympy_extras.functions.combinatorial.factorials import rf
    from sympy_extras.polys.polyroots import roots
    from sympy_extras.polys.polytools import Poly, cancel
    from math import gcd
    probe = _taylor_coefficients(f, x, _PROBE_TERMS)
    if probe is None:
        return None
    support = _support(probe)
    if support is None or len(support) < 2:
        return None
    s = support[0]
    m = 0
    for n in support[1:]:
        m = gcd(m, n - s)
    count = s + m*_CLASS_TERMS + m
    if count > _MAX_TERMS:
        return None
    coefficients = _taylor_coefficients(f, x, count)
    if coefficients is None:
        return None
    support = _support(coefficients)
    if support is None or any((n - s) % m for n in support) or support[0] != s:
        return None
    if [n for n in range(s, count) if (n - s) % m == 0] != support:
        # a zero inside the class: not a hypergeometric term
        return None
    e0 = coefficients[s]
    ratios: list[Fraction] = []
    for n in support:
        q = cancel(coefficients[n]/e0)
        if not isinstance(q, Rational):
            return None
        ratios.append(q.as_fraction())
    recurrence = guess_recurrence(ratios, max_order=1)
    if recurrence is None or recurrence.order != 1:
        return None
    i = Dummy('i')
    P = -_polynomial_of(recurrence.coefficients[0], i)
    Q = _polynomial_of(recurrence.coefficients[1], i)
    p_roots, q_roots = roots(Poly(P, i)), roots(Poly(Q, i))
    if not isinstance(p_roots, dict) or not isinstance(q_roots, dict):
        return None
    p_poly, q_poly = Poly(P, i), Poly(Q, i)
    if sum(p_roots.values()) != p_poly.degree() or sum(q_roots.values()) != q_poly.degree():
        return None
    # Q must not vanish at a nonnegative integer (the closed form and the
    # recurrence would break there), and its roots t give the exponents
    # n = m (t + 1) + s which the differential equation leaves free: they
    # must be among the coefficients computed
    for root in q_roots:
        if isinstance(root, Rational):
            if root.q == 1 and root.p >= 0:
                return None
            n_free = m*(root + 1) + s
            if isinstance(n_free, Rational) and n_free.q == 1 and n_free.p >= count:
                return None
    # the differential equation of the class, checked on f
    t = Dummy('t')
    residual = _apply_in_theta(as_expr(Q.subs(i, t - 1)), t, f - e0*x**s, x, s, m) - \
        x**m*_apply_in_theta(as_expr(P.subs(i, t)), t, f, x, s, m)
    if not _vanishes(residual):
        return None
    k = Dummy('k', integer=True, nonnegative=True)
    j = (k - s)/m
    value = e0*(p_poly.LC()/q_poly.LC())**j
    for root, multiplicity in p_roots.items():
        value = value*rf(-root, j)**multiplicity
    for root, multiplicity in q_roots.items():
        value = value/rf(-root, j)**multiplicity
    formula: Expr = value
    if m > 1:
        formula = piecewise((value, Eq(Mod(k, m), s % m)), (S.Zero, true))
    return FormalSeriesTerms(formula, k, Integer(0), S.Zero)


def taylor_coefficient(f: Expr, x: Symbol) -> Optional[SeriesCoefficient]:
    """The power series of ``f`` about 0 as a :class:`SeriesCoefficient`,
    from :func:`formal_series_terms`; ``None`` when there is none or its
    coefficients have no single closed form.

    Examples
    ========

    >>> from sympy_extras.core import Integer, symbols
    >>> from sympy_extras.functions import exp, sin
    >>> from sympy_extras.integrals.brackets import taylor_coefficient
    >>> x = symbols('x')
    >>> c = taylor_coefficient(exp(-x), x)
    >>> c.coefficient.subs(c.index, Integer(3)), c.period, c.offset
    (-1/6, 1, 0)
    >>> c = taylor_coefficient(sin(x), x)
    >>> c.period, c.offset, c.coefficient.subs(c.index, Integer(1))
    (2, 1, -1/6)
    """
    if f.is_polynomial(x) is True:
        return None
    series = formal_series_terms(f, x)
    if series is None:
        return None
    k = series.index
    found = _residue_class(series.formula, k)
    if found is None:
        return None
    value, m, r = found
    # the independent terms must agree with the formula
    for term in Add.make_args(series.independent):
        if term == 0:
            continue
        coefficient, rest = term.as_independent(x, as_Add=False)
        exponent = monomial(rest, x)
        if exponent is None:
            if rest == 1:
                power: Expr = S.Zero
            else:
                return None
        else:
            power = exponent[1]
        if not isinstance(power, Integer) or (int(power) - r) % m != 0:
            return None
        at_power = _replaced(value, k, power)
        if at_power is None or simplify(at_power - coefficient) != 0:
            return None
    # and the whole must agree with the Taylor expansion for the first
    # terms (SymPy's fps, used before, was not to be trusted alone: its
    # formula for (exp(4*x) - exp(-4*x))**2 holds from k = 3 and gives 2
    # at k = 0, where the series has no term (the bug: the integral over
    # (0, 1) came out as sinh(8)/4, 2 too much), and its formula for
    # 1/(x**2 + x + 1) is (-1)**k, the series of 1/(x + 1) (the bug: the
    # integral of (1 - x)/(x**2 + x + 1) over (0, 1) came out as
    # log(4) - 1, an eighth too much))
    first = series.start
    if not isinstance(first, Integer):
        return None
    terms = max(6, int(first) + 2 * m)
    try:
        truncated = attempt(lambda: f.series(x, S.Zero, terms).removeO(), settings.timeout, native=True)
    except PoleError:
        return None
    if truncated is None:
        return None
    # the one formula returned must give every coefficient of its residue
    # class, the ones below the start of the formula included; where it is
    # infinite (-x**k/k for log(1 - x) at k = 0) the callers start the sum
    # after it, so the coefficient must be zero there
    for k0 in range(terms):
        actual = truncated.coeff(x, k0) if k0 else truncated.subs(x, S.Zero)
        if (k0 - r) % m:
            predicted: Expr = S.Zero
        else:
            at_k0 = _replaced(value, k, Integer(k0))
            if at_k0 is None:
                return None
            predicted = S.Zero if at_k0.is_finite is False else at_k0
        if simplify(predicted - actual) != 0:
            return None
    j = Dummy('j', integer=True, nonnegative=True)
    coefficient_ = _replaced(value, k, m * j + r)
    if coefficient_ is None:
        return None
    return SeriesCoefficient(coefficient_, j, m, Integer(r))


def _decays_fast(f: Expr, x: Symbol) -> bool:
    """Whether ``f`` decays faster than the powers of ``x`` met in practice
    (``x**50 f`` tends to zero): an exponential decay."""
    decays = attempt(lambda: limit(f * x**50, x, oo), settings.timeout, native=True)
    return decays is not None and decays == 0


def _oscillation_order(f: Expr, x: Symbol) -> Optional[Expr]:
    """``p - q`` for ``f = C x**q T(a x**p)`` with ``T`` a sine or a
    cosine, ``a`` a real constant and ``p > 0`` rational: the integral of
    ``x**(s - 1) f`` converges at infinity exactly for ``Re s < p - q``
    (Dirichlet's test after ``u = x**p``); ``None`` for another ``f``."""
    from sympy_extras.core.numbers import Rational
    from sympy_extras.functions.elementary.trigonometric import Cos, Sin
    q: Expr = S.Zero
    oscillating: Optional[Expr] = None
    for factor in Mul.make_args(f):
        if not factor.has(x):
            continue
        if isinstance(factor, (Sin, Cos)):
            if oscillating is not None:
                return None
            oscillating = factor.args[0] if isinstance(factor.args[0], Expr) else None
            if oscillating is None:
                return None
            continue
        found = monomial(factor, x)
        if found is None or found[0] != 1:
            return None
        q = q + found[1]
    if oscillating is None:
        return None
    argument = monomial(oscillating, x)
    if argument is None:
        return None
    a, p = argument
    if not isinstance(p, Rational) or p <= 0 or a.has(x) or a.is_extended_real is not True or a.is_zero is not False:
        return None
    return p - q


def _order_at_infinity(f: Expr, x: Symbol) -> Optional[Expr]:
    """``e`` such that ``f ~ C x**(-e)`` as ``x -> oo`` (the exponent of
    the leading term, powers of ``log(x)`` aside), from the series of
    ``f(1/y)`` at ``y = 0``; ``oo`` when ``f`` decays faster than every
    power (an exponential); for an oscillation ``C x**q sin(a x**p)`` the
    bound ``p - q`` of the convergence of its Mellin transform."""
    y = Dummy('y', positive=True)
    g = f.subs(x, 1 / y)
    try:
        leading = attempt(lambda: g.series(y, S.Zero, 1).removeO(), settings.timeout, native=True)
    except PoleError:
        # an essential singularity at infinity: exponential decay or oscillation
        if _decays_fast(f, x):
            return oo
        return _oscillation_order(f, x)
    if leading is None or leading == 0:
        # the leading term is beyond the first order: ask for the leading term itself
        try:
            found = attempt(lambda: g.leadterm(y), settings.timeout, native=True)
        except (PoleError, ValueError, NotImplementedError):
            found = None
        if found is not None and not found[1].has(y):
            return found[1]
        # no term at all (exp(-1/y) is O(y**n) for every n): an
        # exponential decay, or an oscillation
        if _decays_fast(f, x):
            return oo
        return _oscillation_order(f, x)
    if isinstance(leading, Order):
        if _decays_fast(f, x):
            return oo
        return _oscillation_order(f, x)
    from sympy_extras.functions.elementary.exponential import Log
    exponents: list[Expr] = []
    for term in Add.make_args(leading):
        found_ = monomial(term, x) if term.has(x) else None
        if found_ is not None:
            return None
        rest = term.as_independent(y, as_Add=False)[1]
        # powers of log(y) do not change the exponent of convergence
        rest = Mul(*[a for a in Mul.make_args(rest)
                     if not (isinstance(a, Log) and a.args[0] == y)
                     and not (isinstance(a, Pow) and isinstance(a.base, Log) and a.base.args[0] == y)]).doit()
        if not rest.has(y):
            exponents.append(S.Zero)
            continue
        mono = monomial(rest, y)
        if mono is None:
            return None
        exponents.append(mono[1])
    if not exponents:
        return None
    return min(exponents, key=lambda e: float(e))


def _order_at_zero(f: Expr, x: Symbol, offset: Expr) -> Expr:
    """The exponent of the first term of the series of ``f`` at 0 (the
    ``offset`` of its coefficients when it is not found): ``log(1 + x)``
    starts with ``x``, its formula ``(-1)**(k + 1)/k`` with ``k = 0``."""
    try:
        found = attempt(lambda: f.leadterm(x), settings.timeout, native=True)
    except (PoleError, ValueError, NotImplementedError):
        return offset
    if found is None or found[1].has(x) or not isinstance(found[1], Integer):
        return offset
    return found[1] if (found[1] - offset).is_positive is True else offset


def _strip(f: Expr, x: Symbol, s: Symbol, offset: Expr) -> Optional[tuple[Expr, Expr]]:
    """The strip of convergence: ``-r < Re s < e`` with ``r`` the order of
    ``f`` at 0 and ``e`` the order of decay at infinity, ``None`` when it
    cannot be found."""
    upper = _order_at_infinity(f, x)
    if upper is None or upper.has(x):
        return None
    return (-_order_at_zero(f, x, offset), upper)


def ramanujan_master_theorem(f: Expr, x: Symbol, s: Symbol,
                             assumptions: Assumptions = None) -> Optional[MellinTransform]:
    """The Mellin transform of ``f`` by Ramanujan's master theorem, with
    the strip in which the integral converges; ``None`` when ``f`` has no
    usable power series or no decay at infinity.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.integrals.brackets import ramanujan_master_theorem
    >>> x, s = symbols('x s')
    >>> ramanujan_master_theorem(1/(1 + x), x, s)
    MellinTransform(pi/sin(pi*s), (0, 1))
    >>> ramanujan_master_theorem(exp(-x**2), x, s)
    MellinTransform(gamma(s/2)/2, (0, oo))
    >>> ramanujan_master_theorem(exp(x), x, s) is None
    True
    """
    c = taylor_coefficient(f, x)
    if c is None:
        return None
    phi = c.phi()
    if phi is None:
        return None
    # f = x^r g(x^m): M[f](s) = M[g]((s + r)/m) / m
    sigma = (s + c.offset) / c.period
    value = gammasimp(gamma(sigma) * phi.subs(c.index, -sigma) / c.period)
    strip = _strip(f, x, s, c.offset)
    if strip is None:
        return None
    lower, upper = strip
    if upper != oo and lower != -oo and (upper - lower).is_positive is False:
        return None
    return MellinTransform(value, (lower, upper))


def _factors(f: Expr, x: Symbol) -> Optional[tuple[Expr, Expr, list[Factor]]]:
    """``f`` as ``constant * x**alpha * prod g_i(beta_i x**gamma_i)``:
    the constant, ``alpha`` and the list of ``(g_i(y), beta_i, gamma_i)``
    with ``y`` a fresh symbol per factor."""
    constant: Expr = S.One
    alpha: Expr = S.Zero
    factors: list[Factor] = []
    for e in Mul.make_args(f):
        if not e.has(x):
            constant = constant * e
            continue
        mono = monomial(e, x)
        if mono is not None:
            constant = constant * mono[0]
            alpha = alpha + mono[1]
            continue
        argument: Optional[Expr] = None
        if e.is_Function and len(e.args) >= 1:
            last = e.args[-1]
            if isinstance(last, Expr):
                argument = last
        elif isinstance(e, Pow) and isinstance(e.base, Add):
            argument = e.base
        if argument is None:
            return None
        rest = argument.as_independent(x, as_Add=True)[1]
        mono = monomial(rest, x)
        if mono is None:
            return None
        beta, g_ = mono
        # a positive scale, so that g is read on (0, oo)
        if beta.is_negative is True:
            beta = -beta
        y = Dummy('y', positive=True)
        factors.append((e.subs(x, (y / beta)**(1 / g_)), beta, g_))
    return constant, alpha, factors


def _scaled(transform: MellinTransform, beta: Expr, g_: Expr, s: Symbol) -> MellinTransform:
    """The transform of ``g(beta x**gamma)`` from that of ``g``."""
    value = transform.transform.subs(s, s / g_) * beta**(-s / g_) / Abs(g_)
    lower, upper = transform.strip
    if g_.is_positive is True:
        return MellinTransform(value, (lower * g_, upper * g_), transform.condition)
    return MellinTransform(value, (upper * g_, lower * g_), transform.condition)


def _shifted(transform: MellinTransform, alpha: Expr, constant: Expr, s: Symbol) -> MellinTransform:
    value = constant * transform.transform.subs(s, s + alpha)
    lower, upper = transform.strip
    return MellinTransform(value, (lower - alpha, upper - alpha), transform.condition)


def method_of_brackets(f: Expr, x: Symbol, s: Symbol, assumptions: Assumptions = None) -> Optional[MellinTransform]:
    """The Mellin transform of a product of two series-expandable factors
    by the method of brackets: one index is eliminated by the bracket
    rule and the remaining series is summed as a hypergeometric function,
    which is expanded when possible. Both eliminations are tried and the
    one whose series converges is kept. The strip is the intersection of
    the strips of the factors' own transforms.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import exp, sin
    >>> from sympy_extras.integrals.brackets import method_of_brackets
    >>> x, s = symbols('x s')
    >>> method_of_brackets(exp(-x)*sin(x), x, s)
    MellinTransform(2**(s + 1)*gamma(s/4 + 1/2)*gamma(s/2 + 1/2)/(8*gamma(1 - s/4)), (-1, oo))
    """
    decomposed = _factors(f, x)
    if decomposed is None or len(decomposed[2]) != 2:
        return None
    constant, alpha, factors = decomposed
    series: list[tuple[SeriesCoefficient, Expr, Expr]] = []
    strips: list[tuple[Expr, Expr]] = []
    for g, beta, g_ in factors:
        variable = _variable(g)
        if variable is None:
            return None
        c = taylor_coefficient(g, variable)
        if c is None or c.phi() is None:
            return None
        single = ramanujan_master_theorem(g, variable, s)
        if single is None:
            return None
        scaled = _scaled(single, beta, g_, s)
        series.append((c, beta, g_))
        strips.append(scaled.strip)
    lower = _max(strips[0][0], strips[1][0])
    upper = _min(strips[0][1], strips[1][1])
    if lower != -oo and upper != oo and (upper - lower).is_positive is False:
        return None
    # the double series: sum_{n, k} phi1(n) (-1)^n/n! phi2(k) (-1)^k/k! beta1^(...) beta2^(...)
    # x^(g1 (m1 n + r1) + g2 (m2 k + r2) + alpha); the bracket <A n + B k + C + s>
    (c1, beta1, g1), (c2, beta2, g2) = series
    phi1, phi2 = c1.phi(), c2.phi()
    if phi1 is None or phi2 is None:
        return None
    for first in (0, 1):
        ca, cb = (c1, c2) if first == 0 else (c2, c1)
        phia, phib = (phi1, phi2) if first == 0 else (phi2, phi1)
        betaa, betab = (beta1, beta2) if first == 0 else (beta2, beta1)
        ga, gb = (g1, g2) if first == 0 else (g2, g1)
        n = ca.index
        # the exponent of x: ga (ma n + ra) + gb (mb k + rb) + alpha; eliminate k
        A = ga * ca.period
        B = gb * cb.period
        C = ga * ca.offset + gb * cb.offset + alpha
        k_star = -(A * n + C + s) / B
        term = (phia * S.NegativeOne**n / factorial(n) * betaa**(ca.period * n + ca.offset)
                * phib.subs(cb.index, k_star) * betab**(cb.period * k_star + cb.offset)
                * gamma(-k_star) / Abs(B))
        term = gammasimp(term)
        total = attempt(lambda: summation(term, (n, S.Zero, oo)), settings.timeout, native=True)
        if total is None:
            continue
        if isinstance(total, Piecewise):
            # the closed form where the series converges, continued
            # analytically to the strip (the bracket rule keeps it)
            closed = _closed_branch(total)
            if closed is None:
                continue
            total = closed
        if total.has(Sum):
            continue
        if total.has(Hyper) and not _converges(total, assumptions):
            continue
        summed = total
        expanded = attempt(lambda: hyperexpand(summed), settings.timeout, native=True)
        value = gammasimp(constant * (expanded if expanded is not None else total))
        decay = _order_at_infinity(f, x)
        if decay is not None and not decay.has(x):
            # the true strip of the product: from its order at 0 and its decay
            lower = -(g1 * c1.offset + g2 * c2.offset + alpha)
            upper = decay - alpha
            return MellinTransform(value, (lower, upper))
        return MellinTransform(value, (lower - alpha, upper - alpha))
    return None


def _closed_branch(total: Piecewise) -> Optional[Expr]:
    """The first branch of a ``Piecewise`` returned by the summation whose
    other branch is the unevaluated sum."""
    for pair in total.args:
        if not pair.expr.has(Sum):
            return pair.expr
    return None


def _variable(g: Expr) -> Optional[Symbol]:
    symbols = [v for v in g.free_symbols if isinstance(v, Dummy)]
    return symbols[0] if len(symbols) == 1 else None


def _converges(e: Expr, assumptions: Assumptions) -> bool:
    """Whether every hypergeometric series in ``e`` converges: ``p <= q``,
    or ``p = q + 1`` with ``|z| < 1`` (decided with the assumptions)."""
    for node in e.atoms(Hyper):
        if not isinstance(node, Hyper):
            continue
        p, q = len(node.ap), len(node.bq)
        if p <= q:
            continue
        if p == q + 1 and ask(Abs(node.argument) < 1, assumptions) is True:
            continue
        return False
    return True


def _max(a: Expr, b: Expr) -> Expr:
    if a == -oo:
        return b
    if b == -oo:
        return a
    d = a - b
    if d.is_nonnegative is True:
        return a
    if d.is_negative is True:
        return b
    return Max(a, b)


def _min(a: Expr, b: Expr) -> Expr:
    if a == oo:
        return b
    if b == oo:
        return a
    d = a - b
    if d.is_nonpositive is True:
        return a
    if d.is_positive is True:
        return b
    return Min(a, b)


def mellin_transform_series(f: Expr, x: Symbol, s: Symbol, assumptions: Assumptions = None) -> Optional[MellinTransform]:
    """The Mellin transform of ``f`` from its power series: Ramanujan's
    master theorem for one factor (times a constant and a power of ``x``),
    the method of brackets for a product of two; ``None`` otherwise.

    Examples
    ========

    >>> from sympy_extras.core import sqrt, symbols
    >>> from sympy_extras.functions import exp, cos
    >>> from sympy_extras.integrals.brackets import mellin_transform_series
    >>> x, s = symbols('x s')
    >>> mellin_transform_series(x**2*exp(-3*x), x, s)
    MellinTransform(3**(-s - 2)*gamma(s + 2), (-2, oo))
    >>> mellin_transform_series(cos(sqrt(x)), x, s)
    MellinTransform(2**(2*s)*sqrt(pi)*gamma(s)/gamma(1/2 - s), (0, 1/2))
    """
    if not isinstance(f, Expr):
        raise TypeError("expected an expression of the core, got %r" % (f,))
    decomposed = _factors(f, x)
    if decomposed is None:
        return None
    constant, alpha, factors = decomposed
    if len(factors) == 1:
        g, beta, g_ = factors[0]
        variable = _variable(g)
        if variable is None:
            return None
        single = ramanujan_master_theorem(g, variable, s, assumptions)
        if single is None:
            return None
        return _shifted(_scaled(single, beta, g_, s), alpha, constant, s)
    if len(factors) == 2:
        return method_of_brackets(f, x, s, assumptions)
    return None
