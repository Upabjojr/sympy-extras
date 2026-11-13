"""Formal power series on the typed core (SymPy's ``sympy.series.formal``):
:func:`fps` computes the series of a function as an explicit formula for
its coefficients, :class:`FormalPowerSeries` represents it (truncation,
sums, products by constants, derivatives and integrals), and the classes
:class:`FormalPowerSeriesProduct`, :class:`FormalPowerSeriesCompose` and
:class:`FormalPowerSeriesInverse` give the first terms of products,
compositions and inverses.

The algorithm is Gruntz and Koepf's [GK]_ [Koepf]_ as SymPy implements
it. The *rational algorithm* applies when ``f`` or one of its first
derivatives is a rational function: its partial fractions ``c/(x - a)**j``
have the coefficients ``(-1)**j c binomial(j + k - 1, k)/a**(j + k)``,
integrated back as many times as ``f`` was differentiated. The
*hypergeometric algorithm* looks for a simple differential equation
``f**(k) + sum A_j f**(j) = 0`` with rational coefficients ``A_j`` (linear
algebra on the rationally independent terms), turns it into a recurrence
for the coefficients (``x**l f**(j) -> rf(k + 1 - l, j) a(k + j - l)``)
and solves a recurrence of two terms as a hypergeometric term, from the
first coefficients (the limits of the derivatives at 0), after the
transformations which keep the type (``x**n f(x)``, ``f(x**n)``,
``f'(x)``); a differential equation with constant coefficients becomes a
recurrence solved by :func:`~sympy_extras.solvers.recurr.rsolve_linear`.
Expansions at ``x0 != 0``, from the left and at infinity are expansions
at 0 of ``f(x + x0)``, ``f(x0 - x)`` and ``f(1/x)``.

This module is a typed port of SymPy's ``sympy/series/formal.py``. The
partial fractions are those of :func:`~sympy_extras.polys.partfrac.apart`;
with ``full=True`` the denominators are split over their roots
(:func:`~sympy_extras.polys.polyroots.roots`) and the numerators are the
Taylor coefficients of the cofactor at each root (in place of Bronstein's
algorithm with ``RootSum`` of SymPy's ``apart(full=True)``). The
integrals are those of
:func:`~sympy_extras.integrals.indefinite.indefinite_integral` and
:func:`~sympy_extras.integrals.definite.definite_integral`, the linear
systems are solved by fraction-free elimination, the recurrences by
:func:`~sympy_extras.solvers.recurr.rsolve_linear`. Differences: a
partial fraction whose denominator is not a power of a linear polynomial
(an irreducible quadratic, kept by ``apart`` without ``full``) makes the
rational algorithm decline, where SymPy's formula treats it as linear and
is wrong (``fps(1/(x**2 + x + 1))`` gave the series of ``1/(1 + x)``);
``truncate(None)`` gives the iterator, the other methods take no ``n=None``.

Examples
========

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.functions import exp, log, sin
>>> from sympy_extras.series.formal import fps
>>> x = Symbol('x')
>>> fps(log(1 + x)).truncate()
x - x**2/2 + x**3/3 - x**4/4 + x**5/5 + O(x**6)
>>> f = fps(sin(x))
>>> f.truncate(8), f.diff(x).truncate(4)
(x - x**3/6 + x**5/120 - x**7/5040 + O(x**8), 1 - x**2/2 + O(x**4))
>>> fps(exp(x)).compose(f).truncate(5)
1 + x + x**2/2 - x**4/8 + O(x**5)

References
==========

.. [GK] D. Gruntz, W. Koepf, Maple package on formal power series,
   Maple Technical Newsletter 2 (1995) 22-28.
.. [Koepf] W. Koepf, Power series in computer algebra, Journal of
   Symbolic Computation 13 (1992) 581-603.
.. [Comtet] L. Comtet, Advanced combinatorics, Reidel, 1974 (the
   compositions by Bell polynomials).
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/series/formal.py, sympy/functions/combinatorial/numbers.py
# (bell._bell_incomplete_poly) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: strict typing; Coeff split into a node and coeff_of; the full partial fractions
# by Taylor coefficients at the roots; non-linear denominators declined; no n=None but truncate's.
# ---------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Iterator
from math import lcm
from typing import Final, Optional, Sequence, Union, overload

from sympy_extras.core.add import Add, add
from sympy_extras.core.basic import Basic
from sympy_extras.core.containers import Tuple
from sympy_extras.core.expr import Expr, ExprLike, IntegrationSpec, as_expr, factors_of, terms_of, to_expr
from sympy_extras.core.function import AppliedUndef, Derivative, DefinedFunction, UndefinedFunction, expand
from sympy_extras.core.logic import Boolean, true
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import Infinity, Integer, NegativeInfinity, Number, Rational, nan, oo, zoo
from sympy_extras.core.relational import eq
from sympy_extras.core.symbol import Dummy, Symbol, Wild
from sympy_extras.functions.combinatorial.factorials import factorial, rf
from sympy_extras.functions.elementary.integers import ceiling, floor, frac
from sympy_extras.functions.elementary.miscellaneous import max_, min_
from sympy_extras.functions.elementary.piecewise import ExprCondPair, Piecewise, piecewise
from sympy_extras.series.sequences import EmptySequence, SeqBase, SeqFormula, sequence
from sympy_extras.series.series_class import SeriesBase

__all__ = ['fps', 'compute_fps', 'FormalPowerSeries', 'FiniteFormalPowerSeries', 'FormalPowerSeriesProduct',
           'FormalPowerSeriesCompose', 'FormalPowerSeriesInverse', 'Coeff', 'coeff_of', 'rational_algorithm',
           'rational_independent', 'simpleDE', 'exp_re', 'hyper_re', 'rsolve_hypergeometric', 'solve_de',
           'hyper_algorithm']

#: a formula of the coefficients: ``(formula, independent terms, first index)``
Formula = tuple[Expr, Expr, Expr]
#: the coefficients, the powers and the independent terms of a series
SeriesTerms = tuple[SeqFormula, SeqFormula, Expr]
#: hypergeometric solutions: pairs ``(term, exponent)``
Solutions = list[tuple[Expr, Expr]]

_ZERO: Final[Integer] = Integer(0)
_ONE: Final[Integer] = Integer(1)


# ----------------------------------------------------------------------
# helpers


def _lt(a: Expr, b: ExprLike) -> bool:
    """``a < b`` decided (SymPy raises ``TypeError`` when it cannot)."""
    d = to_expr(b) - a
    if d.is_extended_positive is True:
        return True
    if d.is_extended_nonpositive is True:
        return False
    raise TypeError("cannot determine truth value of %s < %s" % (a, b))


def _is_zero(e: Expr) -> bool:
    return isinstance(e, Number) and e == 0


def _coeff_add(e: Expr, deps: Sequence[Basic]) -> tuple[Expr, tuple[Expr, ...]]:
    """SymPy's ``e.as_coeff_add(*deps)``: the terms free of ``deps`` added,
    and the tuple of the others."""
    independent: list[Expr] = []
    dependent: list[Expr] = []
    for t in terms_of(e):
        (dependent if t.has(*deps) else independent).append(t)
    return add(*independent), tuple(dependent)


def _dependent_factor(e: Expr, pattern: Union[Basic, type[Basic]]) -> Expr:
    """The first factor of ``e`` which has ``pattern`` (SymPy's
    ``e.as_coeff_mul(pattern)[1][0]``)."""
    for f in factors_of(e):
        if f.has(pattern):
            return f
    raise ValueError("no factor of %s has %s" % (e, pattern))


def _derivative(gx: Expr, x: Symbol, i: int) -> Expr:
    """``Derivative(g(x), x, i)``: ``g(x)`` itself for ``i = 0``."""
    return gx if i == 0 else Derivative(gx, (x, i))


def _limit0(e: Expr, x: Symbol) -> Expr:
    from sympy_extras.core.limits import limit
    return limit(e, x, 0)


def _integrate(e: Expr, x: Symbol) -> Expr:
    """SymPy's ``integrate(e, x)``: an antiderivative, the unevaluated
    integral when there is none."""
    if not e.has(x):
        return e * x
    from sympy_extras.integrals.indefinite import indefinite_integral
    return indefinite_integral(e, x)


def _roots(p: Expr, k: Symbol) -> dict[Expr, int]:
    """The roots of the polynomial ``p`` in ``k`` (none for a constant)."""
    from sympy_extras.polys.polyerrors import GeneratorsNeeded, PolynomialError
    from sympy_extras.polys.polyroots import roots
    if not p.has(k):
        return {}
    try:
        found = roots(p, k)
    except (GeneratorsNeeded, PolynomialError):
        return {}
    return found if isinstance(found, dict) else {}


def _numerator(e: Expr) -> Expr:
    return e.as_numer_denom()[0]


def _linsolve(equations: Sequence[Expr], unknowns: Sequence[Symbol]) -> dict[Symbol, Expr]:
    """SymPy's ``linsolve`` as ``fps`` reads it: the unknowns mapped to
    their values, the free unknowns to themselves; empty when there is no
    solution. The equations are linear in the unknowns with coefficients
    in any expressions (Gauss-Jordan elimination, ``cancel`` deciding the
    zeros)."""
    from sympy_extras.polys.polytools import cancel
    zeros: dict[Expr, Expr] = {u: _ZERO for u in unknowns}
    rows: list[list[Expr]] = []
    for e in equations:
        row = [cancel(e.diff(u)) for u in unknowns]
        if any(c.has(*unknowns) for c in row):
            return {}
        row.append(cancel(e.subs(zeros)))
        rows.append(row)
    n = len(unknowns)
    pivots: list[int] = []
    r = 0
    for c in range(n):
        p = next((i for i in range(r, len(rows)) if cancel(rows[i][c]) != 0), None)
        if p is None:
            continue
        rows[r], rows[p] = rows[p], rows[r]
        lead = rows[r][c]
        rows[r] = [cancel(v / lead) for v in rows[r]]
        for i in range(len(rows)):
            if i != r and rows[i][c] != 0:
                factor = rows[i][c]
                rows[i] = [cancel(a - factor * b) for a, b in zip(rows[i], rows[r])]
        pivots.append(c)
        r += 1
    for row in rows[r:]:
        if cancel(row[n]) != 0:
            return {}
    solution: dict[Symbol, Expr] = {u: u for u in unknowns}
    for i, c in enumerate(pivots):
        value: Expr = -rows[i][n]
        for j in range(n):
            if j != c and j not in pivots[:i] and rows[i][j] != 0:
                value = value - rows[i][j] * unknowns[j]
        solution[unknowns[c]] = cancel(value)
    return solution


def _rsolve(equation: Expr, y: AppliedUndef, init: dict[AppliedUndef, Union[Expr, int]]) -> Optional[Expr]:
    """SymPy's ``rsolve`` with initial values: ``None`` when no closed form
    is found."""
    from sympy_extras.solvers.recurr import rsolve_linear
    try:
        return rsolve_linear(equation, y, init)
    except (ValueError, NotImplementedError, TypeError, ZeroDivisionError):
        return None


def _taylor_shift_division(d: list[Expr], a: Expr) -> list[Expr]:
    """The quotient of the polynomial of coefficients ``d`` (highest first)
    by ``x - a``, ``a`` a root (synthetic division)."""
    quotient: list[Expr] = []
    carry: Expr = _ZERO
    for c in d[:-1]:
        carry = expand(c + a * carry)
        quotient.append(carry)
    return quotient


def _apart_full(f: Expr, x: Symbol) -> Optional[Expr]:
    """The full partial fraction decomposition of the rational function
    ``f`` with numerical coefficients: the polynomial part and the terms
    ``c/(x - a)**j`` for every root ``a`` of the denominator; ``None``
    when a root is not found or a coefficient is symbolic."""
    from sympy_extras.polys.polytools import Poly, cancel
    from sympy_extras.simplify.radsimp import radsimp
    num, den = cancel(f).as_numer_denom()
    if (num.free_symbols | den.free_symbols) - {x}:
        return None
    P, Q = Poly(num, x), Poly(den, x)
    polynomial, remainder = P.div(Q)
    terms: list[Expr] = [polynomial.as_expr()]
    _, factors = Q.sqf_list()
    whole = remainder.as_expr() / Q.as_expr()
    for d, n in factors:
        if d.degree() <= 0:
            continue
        found = _roots(d.as_expr(), x)
        if sum(found.values()) != d.degree():
            return None
        cofactor = cancel(whole * d.as_expr()**n)
        coefficients = d.all_coeffs()
        for a in found:
            w = add(*[c * x**(len(coefficients) - 2 - i) for i, c in enumerate(_taylor_shift_division(coefficients, a))])
            g = cofactor / w**n
            derivative = g
            for j in range(n):
                if j:
                    derivative = derivative.diff(x) / j
                c = radsimp(expand(derivative.subs(x, a)))
                terms.append(c / (x - a)**(n - j))
    return add(*terms)


def _apart(f: Expr, x: Symbol, full: bool) -> Optional[Expr]:
    from sympy_extras.polys.partfrac import apart
    if full:
        found = _apart_full(f, x)
        if found is not None:
            return found
    try:
        return apart(f, x)
    except (NotImplementedError, ValueError, TypeError, ArithmeticError):
        return None


# ----------------------------------------------------------------------
# the rational algorithm


# PORTED FROM SYMPY: sympy/series/formal.py:rational_algorithm
def rational_algorithm(f: Expr, x: Symbol, k: Symbol, order: int = 4, full: bool = False) -> Optional[Formula]:
    """The formula of the coefficients of the power series of ``f`` at 0 by
    the rational algorithm, when ``f`` or one of its first ``order``
    derivatives is a rational function: ``(formula, independent terms,
    first index)``, ``None`` otherwise.

    The partial fractions are those of ``apart``; with ``full=True`` the
    denominators are split over their roots (more functions, slower).

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import log
    >>> from sympy_extras.series.formal import rational_algorithm
    >>> x, k = symbols('x k')
    >>> rational_algorithm(1/(1 - x), x, k)
    (1, 0, 0)
    >>> rational_algorithm(log(1 + x), x, k)
    (-1/((-1)**k*k), 0, 1)
    """
    derivative = f
    ds: list[Expr] = []
    for i in range(order + 1):
        if i:
            derivative = derivative.diff(x)
        if derivative.is_rational_function(x):
            coeff: Expr = _ZERO
            sep: Expr = _ZERO
            terms = _apart(derivative, x, full)
            if terms is None:
                return None
            for t in terms_of(terms):
                num, den = t.as_numer_denom()
                if not den.has(x):
                    sep = sep + t
                    continue
                if isinstance(den, Mul):
                    independent, den = den.as_independent(x, as_Add=False)
                    num = num / independent
                base, exponent = den.as_base_exp()
                if not isinstance(exponent, Integer):
                    return None
                j = int(exponent)
                a, xterms = _coeff_add(base, [x])
                if _is_zero(a):
                    sep = sep + t
                    continue
                # SymPy reads every denominator as a power of a linear
                # polynomial: an irreducible quadratic (kept by apart without
                # full) gave a wrong formula, here the algorithm declines
                if len(xterms) != 1:
                    return None
                xc, rest = xterms[0].as_independent(x, as_Add=False)
                if rest != x:
                    return None
                a = -a / xc
                num = num / xc**j
                # binomial(j + k - 1, k) rewritten with factorials
                binom = factorial(j + k - 1) / (factorial(k) * factorial(j - 1))
                coeff = coeff + (-1)**j * num * binom / a**(j + k)
            if coeff.is_zero:
                return None
            if coeff.has(x) or coeff.has(zoo) or coeff.has(oo) or coeff.has(nan):
                return None
            for j in range(i):
                coeff = coeff / (k + j + 1)
                sep = _integrate(sep, x)
                sep = sep + _limit0(ds.pop() - sep, x)  # the constant of integration
            return (coeff.subs(k, k - i), sep, Integer(i))
        ds.append(derivative)
    return None


# PORTED FROM SYMPY: sympy/series/formal.py:rational_independent
def rational_independent(terms: Sequence[Expr], x: Symbol) -> list[Expr]:
    """The terms grouped into sums whose quotients are not rational
    functions of ``x`` (the rationally independent terms).

    Examples
    ========

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.functions import sin, cos
    >>> from sympy_extras.series.formal import rational_independent
    >>> x = Symbol('x')
    >>> rational_independent([cos(x), sin(x)], x)
    [cos(x), sin(x)]
    >>> rational_independent([x**2, sin(x), x*sin(x), x**3], x)
    [x**3 + x**2, x*sin(x) + sin(x)]
    """
    from sympy_extras.polys.polytools import cancel
    if not terms:
        return []
    ind = list(terms[0:1])
    for t in terms[1:]:
        n = t.as_independent(x)[1]
        for i, term in enumerate(ind):
            d = term.as_independent(x)[1]
            q = cancel(n / d)
            if q.is_rational_function(x):
                ind[i] = ind[i] + t
                break
        else:
            ind.append(t)
    return ind


# PORTED FROM SYMPY: sympy/series/formal.py:simpleDE
def simpleDE(f: Expr, x: Symbol, g: UndefinedFunction, order: int = 4) -> Iterator[tuple[Expr, int]]:
    """The simple differential equations ``g**(k) + sum A_j g**(j) = 0``
    with rational coefficients ``A_j`` which ``f`` satisfies, for ``k`` up
    to ``order``, as pairs ``(DE, k)`` (the equations may keep free
    parameters ``a0, a1, ...``).

    Examples
    ========

    >>> from sympy_extras.core import Symbol, Function
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.series.formal import simpleDE
    >>> x, f = Symbol('x'), Function('f')
    >>> next(simpleDE(exp(x), x, f))
    (-f(x) + Derivative(f(x), x), 1)
    """
    from sympy_extras.simplify.radsimp import collect
    a = [Symbol('a%d' % i) for i in range(order)]
    gx = g(x)

    def make(k: int) -> tuple[Expr, Expr]:
        eq_ = f.diff((x, k)) + add(*[a[i] * f.diff((x, i)) if i else a[i] * f for i in range(k)])
        DE = _derivative(gx, x, k) + add(*[a[i] * _derivative(gx, x, i) for i in range(k)])
        return eq_, DE

    found = False
    for k in range(1, order + 1):
        eq_, DE = make(k)
        terms = expand(eq_).as_ordered_terms()
        ind = rational_independent(terms, x)
        if found or len(ind) == k:
            sol = _linsolve(ind, list(a[:k]))
            if sol:
                found = True
                DE = DE.subs(dict(sol))
            DE = _numerator(DE)
            DE = _dependent_factor(DE.factor(), Derivative)
            yield collect(DE, Derivative(gx, x)), k


# PORTED FROM SYMPY: sympy/series/formal.py:exp_re
def exp_re(DE: Expr, r: UndefinedFunction, k: Symbol) -> Expr:
    """The recurrence of a differential equation with constant
    coefficients: ``f**(j) -> r(k + j)``, the lowest term ``r(k)``.

    Examples
    ========

    >>> from sympy_extras.core import symbols, Function, Derivative
    >>> from sympy_extras.series.formal import exp_re
    >>> x, k = symbols('x k')
    >>> f, r = Function('f'), Function('r')
    >>> exp_re(-f(x) + Derivative(f(x), x), r, k)
    -r(k) + r(k + 1)
    """
    RE: Expr = _ZERO
    g = _applied(DE)
    mini: Optional[int] = None
    for t in terms_of(DE):
        coeff, d = t.as_independent(g, as_Add=False)
        j = _order_of(d)
        if mini is None or j < mini:
            mini = j
        RE = RE + coeff * r(k + j)
    if mini:
        RE = RE.subs(k, k - mini)
    return RE


def _applied(DE: Expr) -> AppliedUndef:
    found = sorted([a for a in DE.atoms(AppliedUndef) if isinstance(a, AppliedUndef)], key=lambda e: e.sort_key())
    if not found:
        raise ValueError("no unknown function in %s" % (DE,))
    return found[0]


def _order_of(d: Expr) -> int:
    if isinstance(d, Derivative):
        return sum(n for _, n in d.variable_count)
    return 0


# PORTED FROM SYMPY: sympy/series/formal.py:hyper_re
def hyper_re(DE: Expr, r: UndefinedFunction, k: Symbol) -> Expr:
    """The recurrence of a differential equation with polynomial
    coefficients: ``x**l f**(j) -> rf(k + 1 - l, j) r(k + j - l)``, the
    lowest term ``r(k)``.

    Examples
    ========

    >>> from sympy_extras.core import symbols, Function, Derivative
    >>> from sympy_extras.series.formal import hyper_re
    >>> x, k = symbols('x k')
    >>> f, r = Function('f'), Function('r')
    >>> hyper_re(-f(x) + Derivative(f(x), x), r, k)
    (k + 1)*r(k + 1) - r(k)
    >>> hyper_re(-x*f(x) + Derivative(f(x), (x, 2)), r, k)
    (k + 2)*(k + 3)*r(k + 3) - r(k)
    """
    from sympy_extras.simplify.radsimp import collect
    RE: Expr = _ZERO
    g = _applied(DE)
    xs = sorted([s for s in g.atoms(Symbol) if isinstance(s, Symbol)], key=lambda s: s.sort_key())
    x = xs[0]
    mini: Optional[Expr] = None
    for t in terms_of(expand(DE)):
        coeff, d = t.as_independent(g, as_Add=False)
        c, v = coeff.as_independent(x, as_Add=False)
        l = v.as_coeff_exponent(x)[1]
        j = _order_of(d)
        RE = RE + c * rf(k + 1 - l, j) * r(k + j - l)
        if mini is None or _lt(j - l, mini):
            mini = j - l
    if mini is not None:
        RE = RE.subs(k, k - mini)
    m = Wild('m')
    return collect(RE, r(k + m))


# ----------------------------------------------------------------------
# the hypergeometric algorithm


class _State:
    """The function and the recurrence ``Q(k) a(k + m) = P(k) a(k)`` under
    the transformations of :func:`rsolve_hypergeometric`."""

    __slots__ = ('f', 'P', 'Q', 'm')

    def __init__(self, f: Expr, P: Expr, Q: Expr, m: Expr) -> None:
        self.f: Final[Expr] = f
        self.P: Final[Expr] = P
        self.Q: Final[Expr] = Q
        self.m: Final[Expr] = m


# PORTED FROM SYMPY: sympy/series/formal.py:_transformation_a
def _transformation_a(s: _State, x: Symbol, k: Symbol, shift: Expr) -> _State:
    return _State(s.f * x**(-shift), s.P.subs(k, k + shift), s.Q.subs(k, k + shift), s.m)


# PORTED FROM SYMPY: sympy/series/formal.py:_transformation_c
def _transformation_c(s: _State, x: Symbol, k: Symbol, scale: Expr) -> _State:
    return _State(s.f.subs(x, x**scale), s.P.subs(k, k / scale), s.Q.subs(k, k / scale), s.m * scale)


# PORTED FROM SYMPY: sympy/series/formal.py:_transformation_e
def _transformation_e(s: _State, x: Symbol, k: Symbol) -> _State:
    return _State(s.f.diff(x), s.P.subs(k, k + 1) * (k + s.m + 1), s.Q.subs(k, k + 1) * (k + 1), s.m)


def _apply_shift(sol: Solutions, shift: Expr) -> Solutions:
    return [(res, cond + shift) for res, cond in sol]


def _apply_scale(sol: Solutions, scale: Expr) -> Solutions:
    return [(res, cond / scale) for res, cond in sol]


def _apply_integrate(sol: Solutions, x: Symbol, k: Symbol) -> Solutions:
    return [(res / ((cond + 1) * (cond.as_coeff_Add()[1].coeff(k))), cond + 1) for res, cond in sol]


def _as_int(e: Expr) -> Optional[int]:
    return int(e) if isinstance(e, Integer) else None


# PORTED FROM SYMPY: sympy/series/formal.py:_compute_formula
def _compute_formula(s: _State, x: Symbol, k: Symbol, k_max: int) -> Optional[Solutions]:
    """The formulas for f."""
    m = _as_int(s.m)
    if m is None:
        return None
    sol: Solutions = []
    for i in range(k_max + 1, k_max + m + 1):
        if i < 0:
            continue
        r = _limit0(s.f.diff((x, i)) if i else s.f, x) / factorial(i)
        if r.is_zero:
            continue
        kterm = m * k + i
        res = r
        p = s.P.subs(k, kterm)
        q = s.Q.subs(k, kterm)
        c1 = p.subs(k, 1 / k).leadterm(k)[0]
        c2 = q.subs(k, 1 / k).leadterm(k)[0]
        res = res * (-c1 / c2)**k
        res = res * mul(*[rf(-root, k)**n for root, n in _roots(p, k).items()])
        res = res / mul(*[rf(-root, k)**n for root, n in _roots(q, k).items()])
        sol.append((res, kterm))
    return sol


# PORTED FROM SYMPY: sympy/series/formal.py:_rsolve_hypergeometric
def _rsolve_hypergeometric(f: Expr, x: Symbol, P: Expr, Q: Expr, k: Symbol, m: Expr) -> Optional[tuple[Solutions, Expr, Expr]]:
    """The recursive part of :func:`rsolve_hypergeometric`: ``(formulas,
    independent terms, highest power of x among them)`` or ``None``."""
    s = _State(f, P, Q, m)
    # transformation c
    all_roots = dict(_roots(P, k))
    all_roots.update(_roots(Q, k))
    denominators = [int(r.as_numer_denom()[1]) for r in all_roots if isinstance(r, Rational)]
    scale = Integer(lcm(*denominators)) if denominators else _ONE
    s = _transformation_c(s, x, k, scale)
    # transformation a
    qroots = _roots(s.Q, k)
    k_min = min_(*qroots.keys()) if qroots else _ZERO
    shift = k_min + s.m
    s = _transformation_a(s, x, k, shift)
    from sympy_extras.core.limits import Limit
    l_ = _limit0(x * s.f, x)
    if not isinstance(l_, Limit) and l_ != 0:
        return None
    qroots = _roots(s.Q, k)
    k_max_e = max_(*qroots.keys()) if qroots else _ZERO
    k_max = _as_int(k_max_e)
    m_int = _as_int(s.m)
    if k_max is None or m_int is None:
        return None
    ind: Expr = _ZERO
    mp: Expr = -oo
    for i in range(k_max + m_int + 1):
        r = _limit0(s.f.diff((x, i)) if i else s.f, x) / factorial(i)
        if r.is_finite is False:
            old_f = s.f
            s = _transformation_a(s, x, k, Integer(i))
            s = _transformation_e(s, x, k)
            found = _rsolve_hypergeometric(s.f, x, s.P, s.Q, k, s.m)
            if found is None:
                return None
            sol, ind, mp = found
            sol = _apply_integrate(sol, x, k)
            sol = _apply_shift(sol, Integer(i))
            ind = _integrate(ind, x)
            ind = ind + _limit0(old_f - ind, x)  # the constant of integration
            mp = mp + 1
            return sol, ind, mp
        if not _is_zero(r):
            ind = ind + r * x**(i + shift)
            pow_x = (i + shift) / scale
            if _lt(mp, pow_x):
                mp = pow_x  # the highest power of x
    ind = ind.subs(x, x**(1 / scale))
    formulas = _compute_formula(s, x, k, k_max)
    if formulas is None:
        return None
    formulas = _apply_shift(formulas, shift)
    formulas = _apply_scale(formulas, scale)
    return formulas, ind, mp


# PORTED FROM SYMPY: sympy/series/formal.py:rsolve_hypergeometric
def rsolve_hypergeometric(f: Expr, x: Symbol, P: Expr, Q: Expr, k: Symbol, m: ExprLike) -> Optional[Formula]:
    """The coefficients of the series of ``f`` from the recurrence
    ``Q(k) a(k + m) = P(k) a(k)`` of hypergeometric type: ``(formula,
    independent terms, first index)`` or ``None``.

    The transformations which keep the type are those of Koepf: ``x**n
    f(x)``, ``f(A x)``, ``f(x**n)``, ``f(x**(1/m))`` and ``f'(x)``.

    Examples
    ========

    >>> from sympy_extras.core import symbols, S
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.series.formal import rsolve_hypergeometric
    >>> x, k = symbols('x k')
    >>> rsolve_hypergeometric(exp(x), x, -S.One, (k + 1), k, 1)
    (Piecewise((1/factorial(k), Eq(Mod(k, 1), 0)), (0, True)), 1, 1)
    """
    result = _rsolve_hypergeometric(f, x, P, Q, k, to_expr(m))
    if result is None:
        return None
    sol_list, ind, mp = result
    sol_dict: dict[Boolean, Expr] = {}
    from sympy_extras.core.mod import mod
    for res, cond in sol_list:
        j, mk = cond.as_coeff_Add()
        c = mk.coeff(k)
        jj: Expr = j
        if j.is_integer is False:
            res = res * x**frac(j)
            jj = floor(j)
        res = res.subs(k, (k - jj) / c)
        condition = eq(mod(k, c), mod(jj, c))
        sol_dict[condition] = sol_dict.get(condition, _ZERO) + res
    pairs: list[tuple[Expr, Boolean]] = [(res, cond) for cond, res in sol_dict.items()]
    pairs.append((_ZERO, true))
    sol = piecewise(*pairs)
    s: Expr
    if isinstance(mp, NegativeInfinity):
        s = _ZERO
    elif mp.is_integer is False:
        s = ceiling(mp)
    else:
        s = mp + 1
    if _lt(s, 0):
        ind = ind + add(*list(sequence(sol * x**k, (k, s, -1))))
        s = _ZERO
    return (sol, ind, s)


def _function_terms(RE: Expr, g: UndefinedFunction) -> list[AppliedUndef]:
    return sorted([a for a in RE.atoms(AppliedUndef) if isinstance(a, AppliedUndef) and a.function == g],
                  key=lambda e: e.sort_key())


# PORTED FROM SYMPY: sympy/series/formal.py:_solve_hyper_RE
def _solve_hyper_RE(f: Expr, x: Symbol, RE: Expr, g: UndefinedFunction, k: Symbol) -> Optional[Formula]:
    """See :func:`rsolve_hypergeometric`."""
    terms = terms_of(RE)
    if len(terms) == 2:
        gs = _function_terms(RE, g)
        if len(gs) != 2:
            return None
        P, Q = RE.coeff(gs[0]), RE.coeff(gs[1])
        m = gs[1].args[0] - gs[0].args[0]
        if not isinstance(m, Integer):
            return None
        if m.is_negative:
            P, Q = Q, P
            m = -m
        return rsolve_hypergeometric(f, x, P, Q, k, m)
    return None


# PORTED FROM SYMPY: sympy/series/formal.py:_solve_explike_DE
def _solve_explike_DE(f: Expr, x: Symbol, DE: Expr, g: UndefinedFunction, k: Symbol) -> Optional[Formula]:
    """Solves a differential equation with constant coefficients."""
    gx = _applied(DE)
    for t in terms_of(DE):
        coeff, d = t.as_independent(gx, as_Add=False)
        if coeff.free_symbols:
            return None
    RE = exp_re(DE, g, k)
    init: dict[AppliedUndef, Union[Expr, int]] = {}
    for i in range(len(terms_of(RE))):
        if i:
            f = f.diff(x)
        init[g(Integer(i))] = _limit0(f, x)
    sol = _rsolve(RE, g(k), init)
    if sol is not None and sol != 0:
        return (sol / factorial(k), _ZERO, _ZERO)
    return None


# PORTED FROM SYMPY: sympy/series/formal.py:_solve_simple
def _solve_simple(f: Expr, x: Symbol, DE: Expr, g: UndefinedFunction, k: Symbol) -> Optional[Formula]:
    """Turns the differential equation into a recurrence solved by ``rsolve``."""
    RE = hyper_re(DE, g, k)
    init: dict[AppliedUndef, Union[Expr, int]] = {}
    for i in range(len(terms_of(RE))):
        if i:
            f = f.diff(x)
        init[g(Integer(i))] = _limit0(f, x) / factorial(i)
    sol = _rsolve(RE, g(k), init)
    if sol is not None and sol != 0:
        return (sol, _ZERO, _ZERO)
    return None


def _sorted_symbols(syms: set[Symbol]) -> list[Symbol]:
    return sorted(syms, key=lambda s: s.sort_key())


# PORTED FROM SYMPY: sympy/series/formal.py:_transform_explike_DE
def _transform_explike_DE(DE: Expr, g: UndefinedFunction, x: Symbol, order: int, syms: set[Symbol]) -> Expr:
    """Turns a differential equation with free parameters into one with
    constant coefficients."""
    from sympy_extras.simplify.radsimp import collect
    gx = g(x)
    eqs: list[Expr] = []
    highest_coeff = DE.coeff(_derivative(gx, x, order))
    for i in range(order):
        coeff = DE.coeff(_derivative(gx, x, i))
        coeff = collect(expand(coeff / highest_coeff), x)
        eqs.extend(terms_of(coeff))
    temp: list[Expr] = []
    for e in eqs:
        if e.has(x):
            break
        if e.has(Symbol):
            temp.append(e)
    else:
        eqs = temp
    if eqs:
        sol = _linsolve(eqs, _sorted_symbols(syms))
        if sol:
            DE = DE.subs(dict(sol))
            DE = _dependent_factor(DE.factor(), Derivative)
            DE = collect(DE, Derivative(gx, x))
    return DE


# PORTED FROM SYMPY: sympy/series/formal.py:_transform_DE_RE
def _transform_DE_RE(DE: Expr, g: UndefinedFunction, k: Symbol, order: int, syms: set[Symbol]) -> Expr:
    """Turns a differential equation with free parameters into a
    recurrence of hypergeometric type."""
    from sympy_extras.simplify.radsimp import collect
    RE = hyper_re(DE, g, k)
    eqs = [RE.coeff(g(k + i)) for i in range(1, order)]
    sol = _linsolve(eqs, _sorted_symbols(syms))
    if sol:
        m = Wild('m')
        RE = RE.subs(dict(sol))
        RE = collect(_numerator(RE.factor()), g(k + m))
        RE = _dependent_factor(RE, AppliedUndef)
        for i in range(order):
            if i and RE.coeff(g(k + i)) != 0:
                RE = RE.subs(k, k - i)
                break
    return RE


# PORTED FROM SYMPY: sympy/series/formal.py:solve_de
def solve_de(f: Expr, x: Symbol, DE: Expr, order: int, g: UndefinedFunction, k: Symbol) -> Optional[Formula]:
    """The coefficients of the series of ``f`` from its differential
    equation ``DE``: by a recurrence of two terms, or as an equation with
    constant coefficients; ``(formula, independent terms, first index)`` or
    ``None``.

    Examples
    ========

    >>> from sympy_extras.core import symbols, Function, Derivative
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.series.formal import solve_de
    >>> x, k = symbols('x k')
    >>> f = Function('f')
    >>> solve_de(exp(x), x, Derivative(f(x), x) - f(x), 1, f, k)
    (Piecewise((1/factorial(k), Eq(Mod(k, 1), 0)), (0, True)), 1, 1)
    """
    sol: Optional[Formula] = None
    syms = DE.free_symbols - {x}
    if syms:
        RE = _transform_DE_RE(DE, g, k, order, syms)
    else:
        RE = hyper_re(DE, g, k)
    if not RE.free_symbols - {k}:
        sol = _solve_hyper_RE(f, x, RE, g, k)
    if sol is not None:
        return sol
    if syms:
        DE = _transform_explike_DE(DE, g, x, order, syms)
    if not DE.free_symbols - {x}:
        sol = _solve_explike_DE(f, x, DE, g, k)
    return sol


# PORTED FROM SYMPY: sympy/series/formal.py:hyper_algorithm
def hyper_algorithm(f: Expr, x: Symbol, k: Symbol, order: int = 4) -> Optional[Formula]:
    """The coefficients of the series of ``f`` by the hypergeometric
    algorithm: a differential equation, its recurrence, the solution of
    the recurrence; ``None`` when none is found.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.series.formal import hyper_algorithm
    >>> x, k = symbols('x k')
    >>> hyper_algorithm(exp(x), x, k)
    (Piecewise((1/factorial(k), Eq(Mod(k, 1), 0)), (0, True)), 1, 1)
    """
    from sympy_extras.core.function import Function
    g = Function('g')
    des: list[Expr] = []
    for DE, i in simpleDE(f, x, g, order):
        sol = solve_de(f, x, DE, i, g, k)
        if sol is not None:
            return sol
        if not DE.free_symbols - {x}:
            des.append(DE)
    # nothing works: plain rsolve
    for DE in des:
        sol = _solve_simple(f, x, DE, g, k)
        if sol is not None:
            return sol
    return None


# ----------------------------------------------------------------------
# the driver


def _as_formula_seq(s: Basic) -> SeqFormula:
    if not isinstance(s, SeqFormula):
        raise TypeError("a sequence of a formula is expected, got %s" % (s,))
    return s


# PORTED FROM SYMPY: sympy/series/formal.py:_compute_fps
def _compute_fps(f: Expr, x: Symbol, x0: Expr, dir: Expr, hyper: bool, order: int, rational: bool,
                 full: bool) -> Optional[SeriesTerms]:
    """See :func:`compute_fps`."""
    if isinstance(x0, (Infinity, NegativeInfinity)):
        d = _ONE if isinstance(x0, Infinity) else Integer(-1)
        temp = f.subs(x, 1 / x)
        result = _compute_fps(temp, x, _ZERO, d, hyper, order, rational, full)
        if result is None:
            return None
        return (result[0], _as_formula_seq(result[1].subs(x, 1 / x)), result[2].subs(x, 1 / x))
    if x0 != 0 or dir == -1:
        if dir == -1:
            rep = -x + x0
            rep2: Expr = -x
            rep2b = x0
        else:
            rep = x + x0
            rep2 = x
            rep2b = -x0
        temp = f.subs(x, rep)
        result = _compute_fps(temp, x, _ZERO, _ONE, hyper, order, rational, full)
        if result is None:
            return None
        return (result[0], _as_formula_seq(result[1].subs(x, rep2 + rep2b)), result[2].subs(x, rep2 + rep2b))

    if f.is_polynomial(x):
        k = Dummy('k')
        ak = SeqFormula(Coeff(f, x, k), (k, 1, oo))
        xk = SeqFormula(x**k, (k, 0, oo))
        ind = _constant_term(f, x)
        return ak, xk, ind

    # the terms of a sum separately: the different algorithms on different
    # terms widen the range of the functions
    if isinstance(f, Add):
        found = False
        ak_seq: SeqBase = SeqFormula(_ZERO, (0, oo))
        xk_found: Optional[SeqFormula] = None
        sum_ind: Expr = _ZERO
        for t in terms_of(f):
            res = _compute_fps(t, x, _ZERO, _ONE, hyper, order, rational, full)
            if res is not None:
                if not found:
                    found = True
                    xk_found = res[1]
                assert xk_found is not None
                if _lt(ak_seq.start, res[0].start):
                    seq: SeqBase = ak_seq
                    s, e = ak_seq.start, res[0].start
                else:
                    seq = res[0]
                    s, e = res[0].start, ak_seq.start
                si, ei = _as_int(s), _as_int(e)
                if si is None or ei is None:
                    return None
                save = add(*[a * b for a, b in zip(seq[0:ei - si], xk_found[si:ei])])
                ak_seq = ak_seq + res[0]
                sum_ind = sum_ind + res[2] + save
            else:
                sum_ind = sum_ind + t
        if found and xk_found is not None:
            return _as_formula_seq(ak_seq), xk_found, sum_ind
        return None

    # the symbolic factor, if any, comes out of the function
    syms = f.free_symbols - {x}
    f, symb = expand(f).as_independent(*syms, as_Add=False)

    formula: Optional[Formula] = None
    # from here on x0 = 0 and dir = 1
    k = Dummy('k')
    if rational:
        formula = rational_algorithm(f, x, k, order, full)
    if formula is None and hyper:
        formula = hyper_algorithm(f, x, k, order)
    if formula is None:
        return None

    from sympy_extras.simplify.powsimp import powsimp
    if symb.is_zero:
        symb = _ONE
    else:
        symb = powsimp(symb)
    first = sequence(formula[0], (k, formula[2], oo))
    if isinstance(first, EmptySequence):
        return None
    ak = _as_formula_seq(first)
    xk_formula = powsimp(x**k * symb)
    xk = SeqFormula(xk_formula, (k, 0, oo))
    ind = powsimp(formula[1] * symb)
    return ak, xk, ind


def _direction(dir: Union[int, str, Expr]) -> Expr:
    if dir == '+':
        return _ONE
    if dir == '-':
        return Integer(-1)
    if isinstance(dir, str) or dir not in (1, -1):
        raise ValueError("Dir must be '+' or '-'")
    return to_expr(dir)


# PORTED FROM SYMPY: sympy/series/formal.py:compute_fps
def compute_fps(f: ExprLike, x: Symbol, x0: ExprLike = 0, dir: Union[int, str, Expr] = 1, hyper: bool = True,
                order: int = 4, rational: bool = True, full: bool = False) -> Optional[SeriesTerms]:
    """The coefficients ``ak``, the powers ``xk`` (sequences) and the
    independent terms of the formal power series of ``f`` at ``x0`` (from
    the right for ``dir = 1`` or ``'+'``, from the left for ``-1`` or
    ``'-'``), by the rational algorithm, then the hypergeometric one;
    ``None`` when neither applies.

    Examples
    ========

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.series.formal import compute_fps
    >>> x = Symbol('x')
    >>> ak, xk, ind = compute_fps(exp(x), x)
    >>> ak[:3], xk[:3], ind
    ([1, 1/2, 1/6], [1, x, x**2], 1)
    """
    fe = to_expr(f)
    if not fe.has(x):
        return None
    return _compute_fps(fe, x, to_expr(x0), _direction(dir), hyper, order, rational, full)


# PORTED FROM SYMPY: sympy/series/formal.py:Coeff
class Coeff(DefinedFunction):
    """``Coeff(p, x, n)``: the ``n``-th coefficient of the polynomial ``p``
    in ``x``, unevaluated (:func:`coeff_of` evaluates, at an integer ``n``)."""

    function_name = 'Coeff'
    nargs = frozenset({3})
    call_nargs = frozenset({3})

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        p, x, n = args
        if isinstance(n, Integer) and isinstance(x, Symbol) and p.is_polynomial(x):
            if int(n) < 0:
                return _ZERO
            if int(n) == 0:
                return _constant_term(p, x)
            return p.coeff(x, int(n))
        return None


def _constant_term(p: Expr, x: Expr) -> Expr:
    return add(*[t for t in terms_of(expand(p)) if not t.has(x)])


def coeff_of(p: ExprLike, x: ExprLike, n: ExprLike) -> Expr:
    """``Coeff(p, x, n)``, evaluated.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.series.formal import coeff_of
    >>> x, k = Symbol('x'), Symbol('k')
    >>> coeff_of(x**2 + 3*x + 1, x, 1), coeff_of(x**2 + 3*x + 1, x, k)
    (3, Coeff(x**2 + 3*x + 1, x, k))
    """
    return Coeff.apply(to_expr(p), to_expr(x), to_expr(n))


# ----------------------------------------------------------------------
# the series


def _bell_incomplete_poly(n: int, k: int, values: Sequence[Expr], memo: dict[tuple[int, int], Expr]) -> Expr:
    # PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:bell._bell_incomplete_poly
    if n == 0 and k == 0:
        return _ONE
    if n == 0 or k == 0:
        return _ZERO
    if (n, k) in memo:
        return memo[(n, k)]
    s: Expr = _ZERO
    a: Expr = _ONE
    for m in range(1, n - k + 2):
        s = s + a * _bell_incomplete_poly(n - m, k - 1, values, memo) * values[m - 1]
        a = a * Rational(n - m, m)
    from sympy_extras.core.function import expand_mul
    result = expand_mul(s)
    memo[(n, k)] = result
    return result


def bell_incomplete(n: int, k: int, values: Sequence[Expr]) -> Expr:
    """The incomplete (partial) Bell polynomial ``B_{n,k}(x_1, x_2, ...)``
    at ``values`` (SymPy's ``bell(n, k, values)``).

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.series.formal import bell_incomplete
    >>> x1, x2, x3 = symbols('x1 x2 x3')
    >>> bell_incomplete(4, 2, [x1, x2, x3])
    4*x1*x3 + 3*x2**2
    """
    return _bell_incomplete_poly(n, k, values, {})


def _check_compatible(a: FormalPowerSeries, b: Expr) -> FormalPowerSeries:
    if not isinstance(b, FormalPowerSeries):
        raise ValueError("Both series should be an instance of FormalPowerSeries class.")
    if a.direction != b.direction:
        raise ValueError("Both series should be calculated from the same direction.")
    if a.x0 != b.x0:
        raise ValueError("Both series should be calculated about the same point.")
    if a.x != b.x:
        raise ValueError("Both series should have the same symbol.")
    return b


# PORTED FROM SYMPY: sympy/series/formal.py:FormalPowerSeries
class FormalPowerSeries(SeriesBase):
    """The formal power series of a function: its formula of coefficients
    ``ak``, the powers ``xk`` and the independent terms ``ind``. No
    computation is done: :func:`fps` computes the series.

    The arguments are ``(function, x, x0, dir, (ak, xk, ind))``.
    """

    __slots__ = ('_function', '_x', '_x0', '_dir', '_ak', '_xk', '_ind')

    def __init__(self, function: ExprLike, x: Symbol, x0: ExprLike, dir: ExprLike,
                 terms: Union[SeriesTerms, Tuple]) -> None:
        self._function: Final[Expr] = to_expr(function)
        if not isinstance(x, Symbol):
            raise TypeError("the variable of a series is a symbol")
        self._x: Final[Symbol] = x
        self._x0: Final[Expr] = to_expr(x0)
        self._dir: Final[Expr] = to_expr(dir)
        parts: tuple[Basic, ...] = terms.args if isinstance(terms, Tuple) else terms
        if len(parts) != 3:
            raise TypeError("the terms of a series are (ak, xk, ind)")
        self._ak: Final[SeqFormula] = _as_formula_seq(parts[0])
        self._xk: Final[SeqFormula] = _as_formula_seq(parts[1])
        self._ind: Final[Expr] = as_expr(parts[2])
        Basic.__init__(self)

    @property
    def args(self) -> tuple[Basic, ...]:
        return (self._function, self._x, self._x0, self._dir, Tuple(self._ak, self._xk, self._ind))

    def _rebuild(self, args: tuple[Basic, ...]) -> Basic:
        function, x, x0, dir, terms = args
        if not (isinstance(x, Symbol) and isinstance(terms, Tuple)):
            raise TypeError("invalid arguments of a FormalPowerSeries: %s" % (args,))
        return FormalPowerSeries(as_expr(function), x, as_expr(x0), as_expr(dir), terms)

    @property
    def function(self) -> Expr:
        return self._function

    @property
    def x(self) -> Symbol:
        return self._x

    @property
    def x0(self) -> Expr:
        return self._x0

    @property
    def direction(self) -> Expr:
        """SymPy's ``dir`` of the series: ``1`` from the right, ``-1`` from
        the left (``dir`` is the method of the expressions)."""
        return self._dir

    @property
    def ak(self) -> SeqFormula:
        """The sequence of the coefficients."""
        return self._ak

    @property
    def xk(self) -> SeqFormula:
        """The sequence of the powers of ``x``."""
        return self._xk

    @property
    def ind(self) -> Expr:
        """The independent terms."""
        return self._ind

    @property
    def start(self) -> Expr:
        return _ZERO

    @property
    def stop(self) -> Expr:
        return oo

    @property
    def length(self) -> Expr:
        return oo

    @property
    def ak_seq(self) -> SeqFormula:
        k = self._ak.variables[0]
        return SeqFormula(self._ak.formula, (k, 1, oo))

    @property
    def fact_seq(self) -> SeqFormula:
        k = self._ak.variables[0]
        return SeqFormula(factorial(k), (k, 1, oo))

    @property
    def bell_coeff_seq(self) -> SeqBase:
        return self.ak_seq * self.fact_seq

    @property
    def sign_seq(self) -> SeqBase:
        k = self._ak.variables[0]
        return sequence((-1, 1), (k, 1, oo))

    @property
    def infinite(self) -> Expr:
        """The infinite representation of the series (a ``Sum``)."""
        from sympy_extras.concrete.expr_with_limits import Sum
        k = self._ak.variables[0]
        return self._ind + Sum(self._ak.formula * self._xk.formula, (k, self._ak.start, self._ak.stop))

    def _get_pow_x(self, term: Expr) -> Expr:
        """The power of ``x`` in a term."""
        xterm, pow_x = term.as_independent(self._x, as_Add=False)[1].as_base_exp()
        if not xterm.has(self._x):
            return _ZERO
        return pow_x

    def polynomial(self, n: int = 6) -> Expr:
        """The series truncated before ``x**n``, without the order term.

        >>> from sympy_extras.core import Symbol
        >>> from sympy_extras.functions import cos
        >>> from sympy_extras.series.formal import fps
        >>> x = Symbol('x')
        >>> fps(cos(x)).polynomial(5)
        x**4/24 - x**2/2 + 1
        """
        terms: list[Expr] = []
        sym = self.free_symbols
        for i, t in enumerate(self):
            xp = self._get_pow_x(t)
            if xp.has(*sym):
                xp = _coeff_add(xp, list(sym))[0]
            if not _lt(xp, n):
                break
            if xp.is_integer is True and i == n + 1:
                break
            if not _is_zero(t):
                terms.append(t)
        return add(*terms)

    @overload
    def truncate(self, n: None) -> Iterator[Expr]: ...

    @overload
    def truncate(self, n: int = 6) -> Expr: ...

    def truncate(self, n: Optional[int] = 6) -> Union[Expr, Iterator[Expr]]:
        """The series truncated at ``O(x**n)`` (an iterator over the terms
        for ``n=None``).

        >>> from sympy_extras.core import Symbol
        >>> from sympy_extras.functions import cos
        >>> from sympy_extras.series.formal import fps
        >>> x = Symbol('x')
        >>> fps(cos(x)).truncate()
        1 - x**2/2 + x**4/24 + O(x**6)
        """
        if n is None:
            return iter(self)
        from sympy_extras.core.order import order
        x0 = self._x0
        pt_xk = self._xk.coeff(n)
        if isinstance(x0, NegativeInfinity):
            x0 = oo
        return self.polynomial(n) + order(pt_xk, (self._x, x0))

    def zero_coeff(self) -> Expr:
        return self._eval_term(_ZERO)

    def _eval_term(self, pt: Expr) -> Expr:
        from sympy_extras.simplify.radsimp import collect
        try:
            pt_xk = self._xk.coeff(pt)
            pt_ak = self._ak.coeff(pt).simplify()  # the coefficients simplified
        except IndexError:
            term: Expr = _ZERO
        else:
            term = pt_ak * pt_xk
        if not _is_zero(self._ind):
            ind: Expr = _ZERO
            sym = self.free_symbols
            for t in terms_of(self._ind):
                pow_x = self._get_pow_x(t)
                if pow_x.has(*sym):
                    pow_x = _coeff_add(pow_x, list(sym))[0]
                if pt == 0 and _lt(pow_x, 1):
                    ind = ind + t
                elif not _lt(pow_x, pt) and _lt(pow_x, pt + 1):
                    ind = ind + t
            term = term + ind
        return collect(term, self._x)

    def _subs(self, old: Basic, new: Basic) -> Basic:
        if self == old:
            return new
        if old.has(self._x):
            return self
        return SeriesBase._subs(self, old, new)

    def _eval_as_leading_term(self, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
        for t in self:
            if not _is_zero(t):
                return t
        return _ZERO

    def _eval_derivative_node(self, s: Symbol) -> Optional[Expr]:
        return self._eval_derivative(s)

    def _ak_formula_pieces(self) -> list[tuple[Expr, Boolean]]:
        formula = self._ak.formula
        if isinstance(formula, Piecewise):
            pieces: list[tuple[Expr, Boolean]] = []
            for pair in formula.args:
                if isinstance(pair, ExprCondPair):
                    pieces.append((pair.expr, pair.cond))
            return pieces
        return [(formula, true)]

    def _eval_derivative(self, x: Symbol) -> Expr:
        f = self._function.diff(x)
        ind = self._ind.diff(x)
        pow_xk = self._get_pow_x(self._xk.formula)
        ak = self._ak
        k = ak.variables[0]
        new: SeqBase
        if ak.formula.has(x):
            form: list[tuple[Expr, Boolean]] = []
            for e, c in self._ak_formula_pieces():
                temp: Expr = _ZERO
                for t in terms_of(e):
                    pow_x = self._get_pow_x(t)
                    temp = temp + t * (pow_xk + pow_x)
                form.append((temp, c))
            new = sequence(piecewise(*form).subs(k, k + 1), (k, ak.start - 1, ak.stop))
        else:
            new = sequence((ak.formula * pow_xk).subs(k, k + 1), (k, ak.start - 1, ak.stop))
        return FormalPowerSeries(f, self._x, self._x0, self._dir, (_as_formula_seq(new), self._xk, ind))

    def integrate(self, *args: IntegrationSpec) -> Expr:
        """The integral of the series: in ``x`` (the default) a series, over
        ``(x, a, b)`` the definite integral of the function.

        >>> from sympy_extras.core import Symbol
        >>> from sympy_extras.functions import sin
        >>> from sympy_extras.series.formal import fps
        >>> x = Symbol('x')
        >>> f = fps(sin(x))
        >>> f.integrate(x).truncate()
        -1 + x**2/2 - x**4/24 + O(x**6)
        >>> f.integrate((x, 0, 1))
        1 - cos(1)
        """
        if len(args) > 1:
            raise NotImplementedError("one variable of integration")
        spec = args[0] if args else self._x
        if isinstance(spec, tuple):
            from sympy_extras.integrals.definite import definite_integral
            return definite_integral(self._function, (spec[0], to_expr(spec[1]), to_expr(spec[2])))
        x = spec
        f = _integrate(self._function, x)
        ind = _integrate(self._ind, x)
        ind = ind + _limit0(f - ind, x)  # the constant of integration
        pow_xk = self._get_pow_x(self._xk.formula)
        ak = self._ak
        k = ak.variables[0]
        new: SeqBase
        if ak.formula.has(x):
            form: list[tuple[Expr, Boolean]] = []
            for e, c in self._ak_formula_pieces():
                temp: Expr = _ZERO
                for t in terms_of(e):
                    pow_x = self._get_pow_x(t)
                    temp = temp + t / (pow_xk + pow_x + 1)
                form.append((temp, c))
            new = sequence(piecewise(*form).subs(k, k - 1), (k, ak.start + 1, ak.stop))
        else:
            new = sequence((ak.formula / (pow_xk + 1)).subs(k, k - 1), (k, ak.start + 1, ak.stop))
        return FormalPowerSeries(f, self._x, self._x0, self._dir, (_as_formula_seq(new), self._xk, ind))

    def product(self, other: Expr, x: Optional[Symbol] = None, n: int = 6) -> FormalPowerSeriesProduct:
        """The product with another series (terms by discrete convolution).

        >>> from sympy_extras.core import Symbol
        >>> from sympy_extras.functions import sin, exp
        >>> from sympy_extras.series.formal import fps
        >>> x = Symbol('x')
        >>> fps(sin(x)).product(fps(exp(x)), x).truncate(4)
        x + x**2 + x**3/3 + O(x**4)
        """
        return FormalPowerSeriesProduct(self, _check_compatible(self, other))

    def coeff_bell(self, n: int) -> SeqBase:
        """The sequence of the incomplete Bell polynomials
        ``B_{n,j}(b_1, b_2, ...)``, ``j = 1, ..., n``, of the coefficients
        times the factorials."""
        values = self.bell_coeff_seq[:n]
        inner = [bell_incomplete(n, j, values[:n - j + 1]) for j in range(1, n + 1)]
        k = Dummy('k')
        return sequence(tuple(inner), (k, 1, oo))

    def compose(self, other: Expr, x: Optional[Symbol] = None, n: int = 6) -> FormalPowerSeriesCompose:
        """The composition ``f(g(x))`` with the series of ``g``, which has no
        constant term (terms by Bell polynomials [Comtet]_).

        >>> from sympy_extras.core import Symbol
        >>> from sympy_extras.functions import sin, exp
        >>> from sympy_extras.series.formal import fps
        >>> x = Symbol('x')
        >>> fps(exp(x)).compose(fps(sin(x)), x).truncate()
        1 + x + x**2/2 - x**4/8 - x**5/15 + O(x**6)
        """
        g = _check_compatible(self, other)
        constant = g._eval_term(_ZERO)
        if constant.has(g.x):
            constant = constant.as_independent(g.x, as_Add=False)[0]
        if not _is_zero(constant):
            raise ValueError("The formal power series of the inner function should not have any "
                             "constant coefficient term.")
        return FormalPowerSeriesCompose(self, g)

    def inverse(self, x: Optional[Symbol] = None, n: int = 6) -> FormalPowerSeriesInverse:
        """The series of ``1/f`` (terms by Bell polynomials [Comtet]_).

        >>> from sympy_extras.core import Symbol
        >>> from sympy_extras.functions import cos
        >>> from sympy_extras.series.formal import fps
        >>> x = Symbol('x')
        >>> fps(cos(x)).inverse(x).truncate(8)
        1 + x**2/2 + 5*x**4/24 + 61*x**6/720 + O(x**8)
        """
        if self._eval_term(_ZERO).is_zero:
            raise ValueError("Constant coefficient should exist for an inverse of a formal power series to exist.")
        return FormalPowerSeriesInverse(self)

    def __add__(self, other: ExprLike) -> Expr:
        o = to_expr(other)
        if isinstance(o, FormalPowerSeries):
            if self._dir != o.direction:
                raise ValueError("Both series should be calculated from the same direction.")
            if self._x0 != o.x0:
                raise ValueError("Both series should be calculated about the same point.")
            x, y = self._x, o.x
            f = self._function + o.function.subs(y, x)
            if self._x not in f.free_symbols:
                return f
            ak = self._ak + o.ak
            if _lt(o.ak.start, self._ak.start):
                seq: SeqFormula = o.ak
                s, e = o.ak.start, self._ak.start
            else:
                seq = self._ak
                s, e = self._ak.start, o.ak.start
            si, ei = _as_int(s), _as_int(e)
            if si is None or ei is None:
                raise ValueError("the series start at points which are not integers")
            save = add(*[a * b for a, b in zip(seq[0:ei - si], self._xk[si:ei])])
            ind = self._ind + o.ind + save
            return FormalPowerSeries(f, x, self._x0, self._dir, (_as_formula_seq(ak), self._xk, ind))
        if not o.has(self._x):
            f = self._function + o
            ind = self._ind + o
            return FormalPowerSeries(f, self._x, self._x0, self._dir, (self._ak, self._xk, ind))
        return add(self, o)

    def __radd__(self, other: ExprLike) -> Expr:
        return self.__add__(other)

    def __neg__(self) -> Expr:
        return FormalPowerSeries(-self._function, self._x, self._x0, self._dir,
                                 (_as_formula_seq(-self._ak), self._xk, -self._ind))

    def __sub__(self, other: ExprLike) -> Expr:
        return self.__add__(-to_expr(other))

    def __rsub__(self, other: ExprLike) -> Expr:
        negated = -self
        if isinstance(negated, FormalPowerSeries):
            return negated.__add__(other)
        return negated + other

    def __mul__(self, other: ExprLike) -> Expr:
        o = to_expr(other)
        if o.has(self._x):
            return mul(self, o)
        f = self._function * o
        ak = self._ak.coeff_mul(o)
        ind = self._ind * o
        return FormalPowerSeries(f, self._x, self._x0, self._dir, (ak, self._xk, ind))

    def __rmul__(self, other: ExprLike) -> Expr:
        return self.__mul__(other)


# PORTED FROM SYMPY: sympy/series/formal.py:FiniteFormalPowerSeries
class FiniteFormalPowerSeries(SeriesBase):
    """The base class of the products, compositions and inverses of
    formal power series: their terms up to an order, no formula."""

    __slots__ = ('_ffps',)

    def __init__(self, ffps: FormalPowerSeries) -> None:
        self._ffps: Final[FormalPowerSeries] = ffps

    @property
    def ffps(self) -> FormalPowerSeries:
        return self._ffps

    @property
    def gfps(self) -> FormalPowerSeries:
        raise NotImplementedError

    @property
    def f(self) -> Expr:
        return self._ffps.function

    @property
    def g(self) -> Expr:
        return self.gfps.function

    @property
    def function(self) -> Expr:
        raise NotImplementedError

    @property
    def x(self) -> Symbol:
        return self._ffps.x

    @property
    def start(self) -> Expr:
        return _ZERO

    @property
    def stop(self) -> Expr:
        return oo

    @property
    def length(self) -> Expr:
        return oo

    @property
    def infinite(self) -> Expr:
        raise NotImplementedError("No infinite version for an object of FiniteFormalPowerSeries class.")

    def _eval_terms(self, n: int) -> Expr:
        raise NotImplementedError("(%s)._eval_terms()" % self)

    def _eval_term(self, pt: Expr) -> Expr:
        raise NotImplementedError("By the current logic, one can get terms upto a certain order, instead of "
                                  "getting term by term.")

    def polynomial(self, n: int = 6) -> Expr:
        return self._eval_terms(n)

    def truncate(self, n: int = 6) -> Expr:
        from sympy_extras.core.order import order
        ffps = self._ffps
        pt_xk = ffps.xk.coeff(n)
        return self.polynomial(n) + order(pt_xk, (ffps.x, ffps.x0))

    def _eval_derivative(self, x: Symbol) -> Expr:
        raise NotImplementedError

    def integrate(self, *args: IntegrationSpec) -> Expr:
        raise NotImplementedError

    def _subs(self, old: Basic, new: Basic) -> Basic:
        if self == old:
            return new
        if old.has(self.x):
            return self
        return SeriesBase._subs(self, old, new)

    def _bell_terms(self, n: int, aux: SeqBase, inner: FormalPowerSeries) -> Expr:
        ffps = self._ffps
        terms: list[Expr] = [ffps.zero_coeff()]
        fact = ffps.fact_seq
        for i in range(1, n):
            seq = aux * inner.coeff_bell(i)
            terms.append(add(*seq[:i]) / fact[i - 1] * ffps.xk.coeff(i))
        return add(*terms)


# PORTED FROM SYMPY: sympy/series/formal.py:FormalPowerSeriesProduct
class FormalPowerSeriesProduct(FiniteFormalPowerSeries):
    """The product of two formal power series: the terms up to an order
    by discrete convolution of the coefficients."""

    __slots__ = ('_gfps',)

    def __init__(self, ffps: FormalPowerSeries, gfps: FormalPowerSeries) -> None:
        FiniteFormalPowerSeries.__init__(self, ffps)
        self._gfps: Final[FormalPowerSeries] = gfps
        Basic.__init__(self)

    @property
    def args(self) -> tuple[Basic, ...]:
        return (self._ffps, self._gfps)

    def _rebuild(self, args: tuple[Basic, ...]) -> Basic:
        f, g = args
        if not (isinstance(f, FormalPowerSeries) and isinstance(g, FormalPowerSeries)):
            raise TypeError("invalid arguments of a FormalPowerSeriesProduct: %s" % (args,))
        return FormalPowerSeriesProduct(f, g)

    @property
    def gfps(self) -> FormalPowerSeries:
        return self._gfps

    @property
    def function(self) -> Expr:
        """The function of the product of the two series."""
        return self.f * self.g

    @property
    def coeff1(self) -> SeqFormula:
        k = self._ffps.ak.variables[0]
        return SeqFormula(self._ffps.ak.formula, (k, 0, oo))

    @property
    def coeff2(self) -> SeqFormula:
        k = self._gfps.ak.variables[0]
        return SeqFormula(self._gfps.ak.formula, (k, 0, oo))

    def _eval_terms(self, n: int) -> Expr:
        """The first ``n`` terms of the product (the discrete convolution
        of the coefficients)."""
        a, b = self.coeff1[:n], self.coeff2[:n]
        terms: list[Expr] = []
        for i in range(n):
            c = add(*[a[j] * b[i - j] for j in range(i + 1)])
            terms.append(c * self._ffps.xk.coeff(i))
        return add(*terms)


# PORTED FROM SYMPY: sympy/series/formal.py:FormalPowerSeriesCompose
class FormalPowerSeriesCompose(FiniteFormalPowerSeries):
    """The composition of two formal power series: the terms up to an
    order by the Bell polynomials of the inner coefficients."""

    __slots__ = ('_gfps',)

    def __init__(self, ffps: FormalPowerSeries, gfps: FormalPowerSeries) -> None:
        FiniteFormalPowerSeries.__init__(self, ffps)
        self._gfps: Final[FormalPowerSeries] = gfps
        Basic.__init__(self)

    @property
    def args(self) -> tuple[Basic, ...]:
        return (self._ffps, self._gfps)

    def _rebuild(self, args: tuple[Basic, ...]) -> Basic:
        f, g = args
        if not (isinstance(f, FormalPowerSeries) and isinstance(g, FormalPowerSeries)):
            raise TypeError("invalid arguments of a FormalPowerSeriesCompose: %s" % (args,))
        return FormalPowerSeriesCompose(f, g)

    @property
    def gfps(self) -> FormalPowerSeries:
        return self._gfps

    @property
    def function(self) -> Expr:
        """The composed function."""
        return self.f.subs(self._ffps.x, self.g)

    def _eval_terms(self, n: int) -> Expr:
        """The first ``n`` terms of the composed series."""
        return self._bell_terms(n, self._ffps.bell_coeff_seq, self._gfps)


# PORTED FROM SYMPY: sympy/series/formal.py:FormalPowerSeriesInverse
class FormalPowerSeriesInverse(FiniteFormalPowerSeries):
    """The inverse ``1/f`` of a formal power series: the terms up to an
    order by the Bell polynomials of the coefficients."""

    __slots__ = ()

    def __init__(self, ffps: FormalPowerSeries) -> None:
        FiniteFormalPowerSeries.__init__(self, ffps)
        Basic.__init__(self)

    @property
    def args(self) -> tuple[Basic, ...]:
        return (self._ffps,)

    def _rebuild(self, args: tuple[Basic, ...]) -> Basic:
        (f,) = args
        if not isinstance(f, FormalPowerSeries):
            raise TypeError("invalid arguments of a FormalPowerSeriesInverse: %s" % (args,))
        return FormalPowerSeriesInverse(f)

    @property
    def aux_seq(self) -> SeqBase:
        ffps = self._ffps
        k = ffps.xk.variables[0]
        inv = ffps.zero_coeff()
        inv_seq = SeqFormula(inv**(-(k + 1)), (k, 1, oo))
        return ffps.sign_seq * ffps.fact_seq * inv_seq

    @property
    def function(self) -> Expr:
        """The function of the inverse."""
        return 1 / self.f

    @property
    def g(self) -> Expr:
        raise ValueError("Only one function is considered while performing inverse of a formal power series.")

    @property
    def gfps(self) -> FormalPowerSeries:
        raise ValueError("Only one function is considered while performing inverse of a formal power series.")

    def _eval_terms(self, n: int) -> Expr:
        """The first ``n`` terms of the inverse."""
        return self._bell_terms(n, self.aux_seq, self._ffps)


# PORTED FROM SYMPY: sympy/series/formal.py:fps
def fps(f: ExprLike, x: Optional[Symbol] = None, x0: ExprLike = 0, dir: Union[int, str, Expr] = 1,
        hyper: bool = True, order: int = 4, rational: bool = True, full: bool = False) -> Expr:
    """The formal power series of ``f`` in ``x`` about ``x0``: a
    :class:`FormalPowerSeries` with an explicit formula of the
    coefficients (see :func:`compute_fps`), ``f`` itself when none is
    found.

    Parameters
    ==========

    x : Symbol, optional
        The only free symbol of ``f`` when not given.
    x0 : number, optional
        The point of the expansion (``oo`` and ``-oo`` allowed).
    dir : {1, -1, '+', '-'}, optional
        From the right (the default) or from the left.
    hyper : bool
        Whether the hypergeometric algorithm is tried.
    order : int
        The highest order of the derivatives tried.
    rational : bool
        Whether the rational algorithm is tried.
    full : bool
        Whether the rational algorithm splits the denominators over their
        roots.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, oo
    >>> from sympy_extras.functions import atan, log, sin
    >>> from sympy_extras.series.formal import fps
    >>> x, n = Symbol('x'), Symbol('n')
    >>> fps(log(1 + x)).truncate()
    x - x**2/2 + x**3/3 - x**4/4 + x**5/5 + O(x**6)
    >>> fps(atan(x), full=True).truncate()
    x - x**3/3 + x**5/5 + O(x**6)
    >>> fps(x**n*sin(x**2), x).truncate(8)
    -x**(n + 6)/6 + x**(n + 2) + O(x**(n + 8))
    """
    fe = to_expr(f)
    if x is None:
        free = fe.free_symbols
        if len(free) == 1:
            x = next(iter(free))
        elif not free:
            return fe
        else:
            raise NotImplementedError("multivariate formal power series")
    result = compute_fps(fe, x, x0, dir, hyper, order, rational, full)
    if result is None:
        return fe
    return FormalPowerSeries(fe, x, x0, _direction(dir), result)
