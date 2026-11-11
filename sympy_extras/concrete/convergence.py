"""Convergence of infinite sums and products, with conditions on the
parameters.

SymPy's ``Sum.is_convergent`` decides the convergence of a series
without parameters (and raises ``NotImplementedError`` with them). Here
the classical tests are run with the parameters carrying assumptions
(:mod:`sympy_extras.assumptions`), and the answer is the condition on the
parameters under which `\\sum_{n \\ge n_0} a_n` converges, the counterpart
of Mathematica's ``SumConvergence``:

0. a *comparison* of `|a_n|` with a rational function of `n` decaying
   at least like `1/n^2`, after the oscillating factors `(-1)^e`,
   `\\cos e`, `\\sin e` are bounded by `1`;
1. the *term test*: `a_n \\not\\to 0` means divergence;
2. **d'Alembert's ratio test** with `r = \\lim |a_{n+1}/a_n|`: absolute
   convergence for `r < 1`, divergence for `r > 1`; when `r` depends on
   the parameters the boundary `r = 1` is examined at each of its points
   (found by :func:`~sympy_extras.assumptions.solve.solve`) when there is a
   single parameter;
3. **Raabe's test** on the boundary, `\\rho = \\lim n (|a_n/a_{n+1}| - 1)`
   (`\\rho > 1` convergence, `\\rho < 1` divergence of `\\sum |a_n|`), and
   **Bertrand's test** when `\\rho = 1`, `\\lim (n(|a_n/a_{n+1}| - 1) - 1)
   \\log n` compared with `1`;
4. **Cauchy's root test** `\\lim |a_n|^{1/n}` when the ratio has no limit;
5. the *power comparison* `q = \\lim \\log |a_n| / \\log n` with the
   `p`-series: convergence for `q < -1`, divergence of `\\sum |a_n|` for
   `q > -1`;
6. **Leibniz's test** for alternating series `(-1)^n b_n` with `b_n`
   decreasing to zero;
7. the **integral test** for eventually positive decreasing terms.

A divergence of `\\sum |a_n|` is a divergence of `\\sum a_n` only when the
terms have eventually a constant sign; for terms of varying sign the
alternating test is tried and otherwise the question is left undecided
(``None``).

An infinite product `\\prod (1 + b_n)` converges (to a nonzero limit)
when `\\sum |b_n|` converges, and diverges when `b_n` has eventually a
constant sign and `\\sum b_n` diverges, or when `1 + b_n \\not\\to 1`;
otherwise the convergence of `\\sum \\log(1 + b_n)` decides.

The module works on the objects of the core and imports no SymPy: the
limits, the decisions under assumptions, the boundary points, the signs
and the simplifications are the native ones of
:mod:`sympy_extras.assumptions` and :mod:`sympy_extras.simplify`.
:mod:`sympy_extras.concrete.convergence_sympy` is its face on SymPy
objects. Two steps of the last resort for terms without parameters are
not ported and go through SymPy at run time (the face holds them):
the integral of the integral test, by SymPy's ``integrate``
(:func:`~sympy_extras.concrete.convergence_sympy.improper_integral_of_core`),
and SymPy's ``Sum.is_convergent``
(:func:`~sympy_extras.concrete.convergence_sympy.is_convergent_of_core`).

References
==========

.. [Knopp] K. Knopp, Theory and Application of Infinite Series, Blackie
   (1951), chapters IX (tests of convergence) and VII (products).
.. [Bromwich] T. J. Bromwich, An Introduction to the Theory of Infinite
   Series, Macmillan (1908), chapter III.
"""
from __future__ import annotations

from typing import Callable, Optional

from sympy_extras.core.accumbounds import AccumBounds
from sympy_extras.core.add import Add, add
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, ExprLike, as_expr, to_expr
from sympy_extras.core.function import expand
from sympy_extras.core.limits import Limit
from sympy_extras.core.logic import Boolean, BooleanFalse, BooleanTrue, and_, false, or_, to_boolean, true
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import Infinity, Integer, NaN, NegativeInfinity, oo, pi
from sympy_extras.core.power import Pow
from sympy_extras.core.relational import Eq, Ge, Gt, Le, Lt, Ne, Relational, eq, ge, gt, le, lt, ne
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.combinatorial.factorials import Factorial
from sympy_extras.functions.elementary.complexes import Abs, abs_
from sympy_extras.functions.elementary.exponential import log
from sympy_extras.functions.elementary.piecewise import Piecewise
from sympy_extras.functions.elementary.trigonometric import Cos, Sin, cos, sin
from sympy_extras.polys.polytools import cancel
from sympy_extras.sets.sets import FiniteSet, Reals, interval
from sympy_extras.simplify.gammasimp import combsimp
from sympy_extras.simplify.powsimp import powsimp
from sympy_extras.simplify.simplify import simplify as _simplify
from sympy_extras._timeout import attempt
from sympy_extras.settings import settings
from sympy_extras.assumptions.analysis import certified_sign, sign_on
from sympy_extras.assumptions.ask import Assumptions, Truth, _facts, ask
from sympy_extras.assumptions.limits import limit as _limit_with_assumptions
from sympy_extras.assumptions.refine import _Refiner, simplify as _simplify_with_assumptions
from sympy_extras.assumptions.resolve import resolve
from sympy_extras.assumptions.solve import solve

__all__ = ['sum_convergence', 'product_convergence', 'is_convergent', 'Condition']

#: a condition on the parameters (``None`` when undecided)
Condition = Optional[Boolean]
#: what a test concludes from the value of a limit
Verdict = Callable[[Expr, '_Context'], Condition]

ONE = Integer(1)
ZERO = Integer(0)


class _Context:
    """The assumptions on the parameters, the summation variable and its
    lower bound, and a recursion depth for the case distinctions."""

    def __init__(self, n: Symbol, lower: Expr, assumptions: list[Boolean], depth: int) -> None:
        self.n = n
        self.lower = lower
        self.assumptions = assumptions
        self.depth = depth

    def assume(self, condition: Boolean) -> _Context:
        return _Context(self.n, self.lower, self.assumptions + [condition], self.depth + 1)

    @property
    def premise(self) -> Boolean:
        return and_(*self.assumptions) if self.assumptions else true

    def ask(self, statement: Boolean) -> Truth:
        return ask(statement, self.premise)

    def decide(self, condition: Boolean) -> Boolean:
        """The condition as ``true``, ``false``, or itself refined."""
        value = self.ask(condition)
        if value is True:
            return true
        if value is False:
            return false
        refined = attempt(lambda: _simplify_with_assumptions(condition, self.premise), settings.timeout,
                          native=True)
        return to_boolean(refined) if refined is not None else condition

    def eventually(self, statement: Boolean) -> Truth:
        """Whether a statement in ``n`` holds for all large ``n``."""
        for start in (self.lower, self.lower + 4, self.lower + 16, self.lower + 64):
            premise = and_(self.premise, ge(self.n, start))
            # the core's own evaluation of the relation with the symbols
            # replaced by dummies carrying their assumptions
            try:
                facts = _facts(premise, None, statement.free_symbols)
            except ValueError:
                return None
            abstracted, _, _ = _Refiner(facts).abstract(statement)
            if abstracted is true:
                return True
            value = ask(statement, premise)
            if value is True:
                return True
        return None


def _replaced(e: Expr, rule: dict[Basic, Basic]) -> Expr:
    """``e`` with the keys of ``rule`` replaced, evaluated (SymPy's
    ``xreplace``)."""
    return as_expr(e._xreplace_evaluated(rule))


def _parameters(e: Basic, n: Symbol) -> set[Symbol]:
    """The free symbols of ``e`` other than ``n``."""
    return {s for s in e.free_symbols if isinstance(s, Symbol) and s != n}


def _real_dummies(e: Expr, n: Symbol) -> dict[Basic, Basic]:
    """``n`` replaced by a positive integer dummy and the parameters without
    a sign by real dummies (the convention of the conditions returned)."""
    forward: dict[Basic, Basic] = {n: Dummy('n', integer=True, positive=True)}
    for parameter in _parameters(e, n):
        if parameter.assumptions0.get('real') is None:
            forward[parameter] = Dummy(parameter.name, real=True)
    return forward


def _clean(e: Expr, n: Symbol) -> Expr:
    """A ratio or a power of terms simplified before its limit is taken
    (``limit`` chokes on ``x**(n + 1)/x**n``): while simplifying, ``n`` is
    a positive integer and the parameters are real, so that the absolute
    values of positive quantities disappear."""
    forward = _real_dummies(e, n)
    back: dict[Basic, Basic] = {d: s for s, d in forward.items()}
    replaced = _replaced(e, forward)
    cleaned = attempt(lambda: _simplify(combsimp(powsimp(replaced, force=True))), settings.timeout, native=True)
    if cleaned is None:
        return e
    return _replaced(cleaned, back)


def _limit(e: Expr, ctx: _Context, clean: bool = True) -> Optional[Expr]:
    """The limit at infinity under the assumptions, ``None`` when it
    cannot be found."""
    cleaned = _clean(e, ctx.n) if clean else e
    value = attempt(lambda: _limit_with_assumptions(cleaned, ctx.n, oo, assumptions=ctx.premise),
                    settings.timeout, native=True)
    if value is None or value.has(Limit) or value.has(ctx.n):
        return None
    return value


def _by_cases(value: Expr, ctx: _Context, verdict: Verdict) -> Condition:
    """A verdict on a limit which may be a ``Piecewise`` in the
    parameters: the verdict is taken in each case under its condition."""
    if not isinstance(value, Piecewise):
        return verdict(value, ctx)
    if ctx.depth > 4:
        return None
    parts: list[Boolean] = []
    for pair in value.args:
        part = verdict(pair.expr, ctx.assume(pair.cond))
        if part is None:
            return None
        if part is not false:
            parts.append(and_(pair.cond, part))
    return or_(*parts) if parts else false


def _is_infinite(value: Expr) -> bool:
    return isinstance(value, (Infinity, NegativeInfinity))


def _compare_with_one(value: Expr, ctx: _Context, above: Condition, below: Condition,
                      boundary: Callable[[_Context], Condition], term: Expr) -> Condition:
    """The condition from a limit compared with 1: ``below`` when it is
    smaller, ``above`` when it is larger, ``boundary(ctx)`` when it is 1;
    for a limit depending on the parameters the three cases are combined,
    the boundary being examined at each point of ``value = 1`` when there
    is a single parameter."""
    if _is_infinite(value):
        return above if isinstance(value, Infinity) else below
    if not value.free_symbols:
        s = certified_sign(value - 1)
        if s is None:
            return None
        if s < 0:
            return below
        if s > 0:
            return above
        return boundary(ctx)
    less = ctx.decide(lt(value, 1))
    if less is true:
        return below
    greater = ctx.decide(gt(value, 1))
    if greater is true:
        return above
    equal = ctx.decide(eq(value, 1))
    if equal is true:
        return boundary(ctx)
    parts: list[Boolean] = []
    if less is not false and below is not None and below is not false:
        parts.append(and_(less, below))
    if greater is not false and above is not None and above is not false:
        parts.append(and_(greater, above))
    if equal is not false:
        on_boundary = _boundary_points(value, ctx, term)
        if on_boundary is None:
            return None
        if on_boundary is not false:
            parts.append(on_boundary)
    if (below is None and less is not false) or (above is None and greater is not false):
        return None
    return or_(*parts) if parts else false


def _boundary_points(value: Expr, ctx: _Context, term: Expr) -> Condition:
    """The condition for convergence on ``value = 1``: with one
    parameter, the points of the boundary are listed and the series is
    tested at each of them."""
    parameters = sorted(_parameters(value, ctx.n), key=lambda s: s.name)
    if len(parameters) != 1 or len(value.free_symbols) != 1 or ctx.depth > 4:
        return None
    [p] = parameters
    points = attempt(lambda: solve(eq(value, 1), p, ctx.premise, domain=Reals), settings.timeout, native=True)
    if not isinstance(points, FiniteSet):
        return None
    parts: list[Boolean] = []
    for point in points.args:
        if not isinstance(point, Expr) or point.free_symbols:
            return None
        at_point = _convergence(_replaced(term, {p: point}),
                                _Context(ctx.n, ctx.lower, [a for a in ctx.assumptions if p not in a.free_symbols],
                                         ctx.depth + 1))
        if at_point is None:
            return None
        if at_point is true:
            parts.append(eq(p, point))
        elif at_point is not false:
            parts.append(and_(eq(p, point), at_point))
    return or_(*parts) if parts else false


def _factors(term: Expr) -> list[Expr]:
    """The factors of a product, the term itself otherwise."""
    if isinstance(term, Mul):
        return [as_expr(f) for f in term.args]
    return [term]


def _alternating(term: Expr, n: Symbol) -> Optional[Expr]:
    """``b_n`` when ``term`` is ``(-1)**(n + c) * b_n`` or
    ``cos(pi*n) * b_n``."""
    rest: list[Expr] = []
    found = False
    for f in _factors(term):
        if isinstance(f, Pow) and f.base == -1 and f.exp.is_polynomial(n) and not found:
            slope = f.exp.diff(n)
            if isinstance(slope, Integer) and abs(int(slope)) == 1:
                found = True
                continue
        if isinstance(f, Cos) and isinstance(as_expr(f.args[0]) - pi*n, Integer) and not found:
            found = True
            continue
        rest.append(f)
    if not found:
        return None
    return mul(*rest)


def _oscillating(term: Expr, n: Symbol) -> Optional[Expr]:
    """``b_n`` when ``term`` is ``sin(c*n + d)*b_n`` or ``cos(c*n + d)*b_n``.

    The partial sums of such a factor are bounded, by summing the
    geometric series of ``exp(I*c*n)``, as soon as ``c`` is not a multiple
    of ``2*pi`` (which would make the factor constant).
    """
    rest: list[Expr] = []
    found = False
    for f in _factors(term):
        if not found and isinstance(f, (Sin, Cos)):
            argument = as_expr(f.args[0])
            c = argument.diff(n)
            if n not in c.free_symbols and (argument - c*n).diff(n) == 0 \
                    and c.is_real and (c/(2*pi)).is_integer is False:
                found = True
                continue
        rest.append(f)
    if not found:
        return None
    return mul(*rest)


def _dirichlet_test(term: Expr, ctx: _Context) -> Condition:
    """Dirichlet's test: a factor with bounded partial sums times a factor
    decreasing to zero gives a convergent series.

    This is what decides ``sum sin(n)/n``, which converges although
    SymPy's ``Sum(sin(n)/n, (n, 1, oo)).is_convergent()`` says otherwise.
    """
    b = _oscillating(term, ctx.n)
    if b is None:
        return None
    return true if _decreasing_to_zero(b, ctx) is true else None


def _coefficients(e: Expr, n: Symbol) -> Optional[dict[int, Expr]]:
    """The nonzero coefficients of ``e`` as a polynomial in ``n`` (any
    constant coefficients: ``pi``, ``sqrt(2)``, parameters), by degree;
    ``None`` when ``e`` is not a polynomial in ``n``."""
    coefficients: dict[int, Expr] = {}
    expanded = expand(e)
    for summand in (expanded.args if isinstance(expanded, Add) else (expanded,)):
        c, rest = as_expr(summand).as_independent(n, as_Add=False)
        if rest == 1:
            k = 0
        elif rest == n:
            k = 1
        elif isinstance(rest, Pow) and rest.base == n and isinstance(rest.exp, Integer) and int(rest.exp) > 0:
            k = int(rest.exp)
        else:
            return None
        coefficients[k] = add(coefficients.get(k, ZERO), c)
    return {k: c for k, c in coefficients.items() if c != 0}


def _real_polynomial(e: Expr, n: Symbol) -> bool:
    """Whether ``e`` is a polynomial in ``n`` with real number coefficients."""
    if not e.free_symbols <= {n}:
        return False
    coefficients = _coefficients(e, n)
    return coefficients is not None and all(c.is_real is True for c in coefficients.values())


def _majorant(term: Expr, n: Symbol) -> Optional[Expr]:
    """An expression bounding ``|term|`` from above, with the oscillating
    factors ``(-1)**e``, ``cos(e)`` and ``sin(e)`` (``e`` a real polynomial
    in ``n``) replaced by ``1`` and the factors free of ``n`` dropped;
    ``None`` when the structure gives no bound."""
    if n not in term.free_symbols:
        return abs_(term) if term.is_number else None
    if not term.has(Sin, Cos) and not any(isinstance(p, Pow) and p.base == -1 for p in term.atoms(Pow)):
        return abs_(term)
    if isinstance(term, Abs):
        return _majorant(as_expr(term.args[0]), n)
    if isinstance(term, (Sin, Cos)):
        return ONE if _real_polynomial(as_expr(term.args[0]), n) else None
    if isinstance(term, Pow):
        base, exponent = term.base, term.exp
        if base == -1:
            return ONE if _real_polynomial(exponent, n) else None
        if isinstance(exponent, Integer) and int(exponent) > 0:
            bound = _majorant(base, n)
            return None if bound is None else bound**exponent
        return None
    if isinstance(term, (Mul, Add)):
        bounds: list[Expr] = []
        for arg in term.args:
            a = as_expr(arg)
            if isinstance(term, Mul) and n not in a.free_symbols:
                continue
            bound = _majorant(a, n)
            if bound is None:
                return None
            bounds.append(bound)
        return mul(*bounds) if isinstance(term, Mul) else add(*bounds)
    return None


def _without_abs_values(e: Expr) -> Expr:
    """``e`` with every ``Abs(u)`` replaced by ``u``."""
    def is_abs(b: Basic) -> bool:
        return isinstance(b, Abs)

    def argument(b: Basic) -> Basic:
        return b.args[0]
    return as_expr(e.replace(is_abs, argument))


def _rational_comparison(term: Expr, ctx: _Context) -> Condition:
    """Absolute convergence by comparison with a rational majorant of
    ``|a_n|`` (:func:`_majorant`) decaying at least like ``1/n**2``;
    ``None`` otherwise.

    Bounding ``|(-1)**n - 1|`` by ``2`` decides at once series such as the
    Fourier coefficients ``((-1)**n - 1)/n**3``, on which the ratio and
    root tests find no limit.
    """
    bound = _majorant(term, ctx.n)
    if bound is None or not bound.free_symbols <= {ctx.n}:
        return None
    rational = _without_abs_values(bound)
    cancelled = attempt(lambda: cancel(rational), settings.timeout, native=True)
    if cancelled is None:
        return None
    numerator, denominator = cancelled.as_numer_denom()
    top, bottom = _coefficients(numerator, ctx.n), _coefficients(denominator, ctx.n)
    if top is None or bottom is None or not bottom:
        return None
    if not top:
        return true
    excess = max(bottom) - max(top)
    return true if excess >= 2 else None


def _factorial_form(term: Expr) -> Expr:
    """The term with the gamma functions, binomials and Pochhammer symbols
    rewritten as factorials (``rewrite(factorial)``)."""
    return as_expr(term.rewrite(Factorial))


def _constant_sign(term: Expr, ctx: _Context) -> Truth:
    """Whether the terms have eventually a constant sign."""
    # the core's own knowledge, with n a positive integer and the
    # parameters real numbers (the convention of the conditions returned)
    replacements = _real_dummies(term, ctx.n)
    forms: list[Callable[[], Expr]] = [lambda: term, lambda: _factorial_form(term), lambda: combsimp(term)]
    for form in forms:
        found = attempt(form, settings.timeout, native=True)
        if found is None:
            continue
        at_positive = _replaced(found, replacements)
        if at_positive.is_nonnegative or at_positive.is_nonpositive:
            return True
    if not _parameters(term, ctx.n):
        for start in (ctx.lower, ctx.lower + 8, ctx.lower + 64):
            domain = interval(start, oo)
            signs = attempt(lambda: sign_on(term, ctx.n, domain), settings.timeout, native=True)
            if signs is not None and not (1 in signs and -1 in signs):
                return True
        return None
    if ctx.eventually(ge(term, 0)) or ctx.eventually(le(term, 0)):
        return True
    return None


def _decreasing_to_zero(b: Expr, ctx: _Context) -> Condition:
    """The condition under which ``b_n`` decreases to zero eventually
    (``None`` when the monotonicity cannot be established)."""
    value = _limit(b, ctx)
    if value is None:
        return None

    def verdict(v: Expr, ctx_: _Context) -> Condition:
        if v.is_zero is False:
            return false
        if not v.is_zero:
            return None
        difference = _replaced(b, {ctx_.n: ctx_.n + 1}) - b
        if ctx_.eventually(le(difference, 0)):
            return true
        derivative = b.diff(ctx_.n)
        if ctx_.eventually(le(derivative, 0)):
            return true
        return None
    return _by_cases(value, ctx, verdict)


def _integral_test(term: Expr, ctx: _Context) -> Condition:
    """For eventually positive decreasing terms, the convergence of the
    integral (SymPy's ``integrate``, at run time through
    :func:`~sympy_extras.concrete.convergence_sympy.improper_integral_of_core`)."""
    if _parameters(term, ctx.n):
        return None
    if ctx.eventually(gt(term, 0)) is not True:
        return None
    if _decreasing_to_zero(term, ctx) is not true:
        return None
    from .convergence_sympy import improper_integral_of_core
    value = improper_integral_of_core(term, ctx.n, ctx.lower)
    if value is None or value.has(ctx.n):
        return None
    if isinstance(value, Infinity):
        return false
    if value.is_finite:
        return true
    return None


def _divergence_of_absolute(term: Expr, ctx: _Context) -> Condition:
    """What the divergence of ``sum |a_n|`` means for ``sum a_n``."""
    if _constant_sign(term, ctx) is True:
        return false
    b = _alternating(term, ctx.n)
    if b is not None:
        decreasing = _decreasing_to_zero(b, ctx)
        if decreasing is not None:
            return decreasing
    return _term_test(term, ctx)


def _ratio(term: Expr, ctx: _Context, inverse: bool = False) -> Expr:
    """``|a_{n+1}/a_n|`` (or its inverse), simplified."""
    shifted = _replaced(term, {ctx.n: ctx.n + 1})
    ratio = abs_(term/shifted) if inverse else abs_(shifted/term)
    return _clean(ratio, ctx.n)


def _ratio_test(term: Expr, ctx: _Context) -> Condition:
    r = _limit(_ratio(term, ctx), ctx, clean=False)
    if r is None:
        return _root_test(term, ctx)

    def verdict(value: Expr, ctx_: _Context) -> Condition:
        return _compare_with_one(value, ctx_, above=false, below=true,
                                 boundary=lambda c: _raabe_test(term, c), term=term)
    return _by_cases(r, ctx, verdict)


def _root_test(term: Expr, ctx: _Context) -> Condition:
    root = abs_(term)**(1/ctx.n)
    c = _limit(root, ctx)
    if c is None:
        return _power_comparison(term, ctx)

    def verdict(value: Expr, ctx_: _Context) -> Condition:
        return _compare_with_one(value, ctx_, above=false, below=true,
                                 boundary=lambda c_: _power_comparison(term, c_), term=term)
    return _by_cases(c, ctx, verdict)


def _raabe_test(term: Expr, ctx: _Context) -> Condition:
    # limits are unreliable on the expanded form of the expression: the
    # ratio is simplified alone and the structure n*(ratio - 1) kept
    expression = ctx.n*(_ratio(term, ctx, inverse=True) - 1)
    rho = _limit(expression, ctx, clean=False)
    if rho is None:
        return _power_comparison(term, ctx)

    def verdict(value: Expr, ctx_: _Context) -> Condition:
        return _compare_with_one(value, ctx_, above=true, below=_divergence_of_absolute(term, ctx_),
                                 boundary=lambda c: _bertrand_test(term, c), term=term)
    return _by_cases(rho, ctx, verdict)


def _bertrand_test(term: Expr, ctx: _Context) -> Condition:
    expression = (ctx.n*(_ratio(term, ctx, inverse=True) - 1) - 1)*log(ctx.n)
    value = _limit(expression, ctx, clean=False)
    if value is None:
        return _power_comparison(term, ctx)

    def verdict(v: Expr, ctx_: _Context) -> Condition:
        return _compare_with_one(v, ctx_, above=true, below=_divergence_of_absolute(term, ctx_),
                                 boundary=lambda c: _last_resort(term, c), term=term)
    return _by_cases(value, ctx, verdict)


def _power_comparison(term: Expr, ctx: _Context) -> Condition:
    """Comparison with the p-series through the growth exponent
    ``q = lim log|a_n| / log n``: convergence for ``q < -1``."""
    exponent = log(abs_(term))/log(ctx.n)
    q = _limit(exponent, ctx)
    if q is None:
        return _last_resort(term, ctx)

    def verdict(value: Expr, ctx_: _Context) -> Condition:
        return _compare_with_one(-value, ctx_, above=true, below=_divergence_of_absolute(term, ctx_),
                                 boundary=lambda c: _last_resort(term, c), term=term)
    return _by_cases(q, ctx, verdict)


def _last_resort(term: Expr, ctx: _Context) -> Condition:
    """The term, alternating and integral tests, and SymPy's own decision
    for terms without parameters (at run time, through
    :func:`~sympy_extras.concrete.convergence_sympy.is_convergent_of_core`)."""
    divergent = _term_test(term, ctx)
    if divergent is not None:
        return divergent
    b = _alternating(term, ctx.n)
    if b is not None:
        decreasing = _decreasing_to_zero(b, ctx)
        if decreasing is true:
            return true
    dirichlet = _dirichlet_test(term, ctx)
    if dirichlet is not None:
        return dirichlet
    integral = _integral_test(term, ctx)
    if integral is not None:
        return integral
    if not _parameters(term, ctx.n):
        from .convergence_sympy import is_convergent_of_core
        # SymPy's is_convergent() answering False means "not shown", not
        # divergence: it says False for sum sin(n)/n (sympy-extras#41).
        # Only True is a proof.
        if is_convergent_of_core(term, ctx.n, ctx.lower) is True:
            return true
    return None


def _oscillates_unboundedly(term: Expr, ctx: _Context) -> bool:
    """Whether ``term`` is an unbounded factor times ``sin`` or ``cos`` of
    a linear argument in ``n`` -- ``n*sin(n)``, say -- whose terms cannot
    tend to zero: for a nonzero ``a``, ``sin(a n + b)`` is bounded away
    from zero for infinitely many integers ``n`` (identically zero only
    when ``a`` is an integer multiple of pi and ``sin(b)`` vanishes).
    ``Sum.is_convergent()`` used to be the source of this verdict, and its
    ``False`` is not a proof (sympy-extras#41)."""
    if not isinstance(term, Mul):
        return False
    waves: list[Expr] = [f for f in _factors(term) if isinstance(f, (Sin, Cos)) and f.args[0].has(ctx.n)]
    if len(waves) != 1:
        return False
    [wave] = waves
    linear = _coefficients(as_expr(wave.args[0]), ctx.n)
    if linear is None or max(linear, default=0) != 1:
        return False
    a, b = linear.get(1, ZERO), linear.get(0, ZERO)
    if a.free_symbols or b.free_symbols or a == 0:
        return False
    turns = a/pi
    if turns.is_integer:
        phase = sin(b) if isinstance(wave, Sin) else cos(b)
        if phase.is_zero is not False:
            return False
    rest = term/wave
    magnitude = _limit(abs_(rest), ctx)
    return magnitude is not None and isinstance(magnitude, Infinity)


def _term_test(term: Expr, ctx: _Context) -> Condition:
    """Divergence when the terms do not tend to zero; ``None``
    otherwise."""
    if _oscillates_unboundedly(term, ctx):
        return false
    value = _limit(term, ctx)
    if value is None:
        return None

    def verdict(v: Expr, ctx_: _Context) -> Condition:
        if isinstance(v, AccumBounds) or _is_infinite(v):
            return false
        if isinstance(v, NaN):
            return None
        if not v.free_symbols:
            return false if v.is_zero is False else None
        return false if ctx_.decide(eq(v, 0)) is false else None
    return _by_cases(value, ctx, verdict)


def _convergence(term: Expr, ctx: _Context) -> Condition:
    if not term.has(ctx.n):
        return ctx.decide(eq(term, 0))
    # a constant factor which may vanish makes the series trivially
    # convergent (and cancels from the ratios, so it is separated first)
    c, rest = term.as_independent(ctx.n, as_Add=False)
    if c.free_symbols:
        zero = ctx.decide(eq(c, 0))
        if zero is true:
            return true
        if zero is not false:
            nonzero = _ratio_test(rest, ctx.assume(ne(c, 0)))
            if nonzero is None:
                return None
            return or_(zero, nonzero)
    if _rational_comparison(term, ctx) is true:
        return true
    return _ratio_test(term, ctx)


def _finish(condition: Condition, ctx: _Context) -> Condition:
    """The condition simplified: as a polynomial formula over the reals
    (absolute values of polynomials written as two inequalities) by
    quantifier-free simplification when possible, by ``simplify`` with the
    assumptions otherwise."""
    if condition is None or isinstance(condition, (BooleanTrue, BooleanFalse)):
        return condition
    polynomial = _without_abs(condition)
    resolved = attempt(lambda: resolve(polynomial, assumptions=ctx.premise if ctx.assumptions else None),
                       settings.timeout, native=True)
    if resolved is not None:
        return resolved
    simplified = attempt(lambda: _simplify_with_assumptions(condition, ctx.premise), settings.timeout, native=True)
    return to_boolean(simplified) if simplified is not None else condition


#: the evaluating constructors of the inequalities, by class
_INEQUALITIES: dict[type, Callable[[ExprLike, ExprLike], Boolean]] = {Lt: lt, Le: le, Gt: gt, Ge: ge}


def _without_abs(condition: Boolean) -> Boolean:
    """``Abs(u) < c`` written as ``(u < c) & (u > -c)`` and the like."""
    result = condition
    for atom in condition.atoms(Lt, Le, Gt, Ge, Eq, Ne):
        if not isinstance(atom, Relational):
            continue
        lhs, rhs = atom.lhs, atom.rhs
        if isinstance(lhs, Abs) and not rhs.has(Abs):
            u = as_expr(lhs.args[0])
            replacement: Boolean
            if isinstance(atom, (Lt, Le)):
                relation = _INEQUALITIES[type(atom)]
                replacement = and_(relation(u, rhs), relation(-u, rhs))
            elif isinstance(atom, (Gt, Ge)):
                relation = _INEQUALITIES[type(atom)]
                replacement = or_(relation(u, rhs), relation(-u, rhs))
            elif isinstance(atom, Eq):
                replacement = or_(eq(u, rhs), eq(u, -rhs))
            else:
                replacement = and_(ne(u, rhs), ne(u, -rhs))
            result = to_boolean(result._xreplace_evaluated({atom: replacement}))
    return result


def _context(n: Symbol, lower: ExprLike, assumptions: Assumptions) -> _Context:
    items: list[Boolean] = []
    if isinstance(assumptions, (Boolean, bool)):
        items.append(to_boolean(assumptions))
    elif assumptions is not None:
        items.extend(to_boolean(a) for a in assumptions)
    return _Context(n, to_expr(lower), items, 0)


def sum_convergence(term: ExprLike, n: Symbol, assumptions: Assumptions = None,
                    lower: ExprLike = 1) -> Condition:
    """The condition on the parameters under which ``Sum(term, (n, lower,
    oo))`` converges: ``true``, ``false``, a Boolean in the parameters, or
    ``None`` when the tests do not decide.

    Parameters
    ==========

    term : Expr
        The summand, a function of ``n`` and of parameters.
    n : Symbol
        The summation variable (an integer tending to infinity).
    assumptions : Boolean or list of Booleans, optional
        Assumptions on the parameters, as statements.
    lower : Expr, optional
        The lower limit of the sum (it does not affect convergence, but
        the terms must be defined from it on).

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import factorial, binomial, log
    >>> from sympy_extras.concrete.convergence import sum_convergence
    >>> n, x, p = symbols('n x p')
    >>> sum_convergence(x**n, n)
    (x > -1) & (x < 1)
    >>> sum_convergence(x**n/n, n)
    (x >= -1) & (x < 1)
    >>> sum_convergence(1/n**p, n)
    p > 1
    >>> sum_convergence(1/n**p, n, p > 2)
    True
    >>> sum_convergence(factorial(n)/n**n, n)
    True
    >>> sum_convergence(binomial(2*n, n)/4**n, n)
    False
    >>> sum_convergence((-1)**n/log(n), n, lower=2)
    True
    """
    term_ = to_expr(term)
    ctx = _context(n, lower, assumptions)
    return _finish(_convergence(term_, ctx), ctx)


def is_convergent(term: ExprLike, n: Symbol, assumptions: Assumptions = None, lower: ExprLike = 1) -> Truth:
    """Whether ``Sum(term, (n, lower, oo))`` converges under the
    assumptions: ``True``, ``False`` or ``None``.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.concrete.convergence import is_convergent
    >>> n, x = symbols('n x')
    >>> is_convergent(x**n, n, (x > 0) & (x < 1))
    True
    >>> is_convergent(x**n, n, x > 1)
    False
    >>> is_convergent(x**n, n) is None
    True
    """
    condition = sum_convergence(term, n, assumptions, lower)
    if condition is None:
        return None
    if isinstance(condition, BooleanTrue):
        return True
    if isinstance(condition, BooleanFalse):
        return False
    return None


def product_convergence(term: ExprLike, n: Symbol, assumptions: Assumptions = None,
                        lower: ExprLike = 1) -> Condition:
    """The condition under which the infinite product ``Product(term,
    (n, lower, oo))`` converges to a nonzero limit (see the module
    documentation).

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.concrete.convergence import product_convergence
    >>> n, x = symbols('n x')
    >>> product_convergence(1 + 1/n**2, n)
    True
    >>> product_convergence(1 + x/n**2, n)
    True
    >>> product_convergence(1 + 1/n, n)
    False
    >>> product_convergence(1 + x**n, n)
    (x > -1) & (x < 1)
    """
    term_ = to_expr(term)
    ctx = _context(n, lower, assumptions)
    b = term_ - 1
    if not b.has(n):
        return _finish(ctx.decide(eq(b, 0)), ctx)
    value = _limit(b, ctx)
    if value is not None and not isinstance(value, Piecewise) and not value.free_symbols and value.is_zero is False:
        return false
    positive = Dummy('n', integer=True, positive=True)
    magnitude = _replaced(abs_(_replaced(b, {n: positive})), {positive: n})
    absolute = _convergence(magnitude, ctx)
    if absolute is true:
        return true
    # for an alternating factor the condition for sum |b| is sufficient,
    # not necessary: prod (1 + (-1)**n/n**a) converges for a > 1/2, not
    # only for a > 1 (sympy-extras#39). That case has an exact criterion.
    alternating = _alternating_product(b, ctx)
    if alternating is not None:
        return _finish(alternating, ctx)
    if absolute is not None and not isinstance(absolute, (BooleanTrue, BooleanFalse)):
        return _finish(absolute, ctx)
    if absolute is false and _constant_sign(b, ctx) is True:
        return false
    logarithmic = _convergence(log(term_), ctx)
    return _finish(logarithmic, ctx)


def _alternating_product(b: Expr, ctx: _Context) -> Condition:
    """``prod (1 + (-1)**n c(n))`` with ``c > 0`` decreasing to zero: the
    alternating sum converges by Leibniz, so the product converges exactly
    when ``sum c**2`` does (``log(1 + b) = b - b**2/2 + O(b**3)``)."""
    c = attempt(lambda: cancel(b/(-1)**ctx.n), settings.timeout, native=True)
    if c is None or any(isinstance(p, Pow) and p.base == -1 for p in c.atoms(Pow)):
        return None
    # positivity of c for n >= 1 is a question about n only: a free
    # parameter in the exponent stops ctx.decide, but not the core's
    # assumptions on a positive n
    grows = Dummy('n', integer=True, positive=True)
    reals: dict[Basic, Basic] = {s: Dummy(s.name, real=True) for s in _parameters(c, ctx.n)}
    if _replaced(_replaced(c, {ctx.n: grows}), reals).is_positive is not True:
        return None
    return _convergence(c**2, ctx)
