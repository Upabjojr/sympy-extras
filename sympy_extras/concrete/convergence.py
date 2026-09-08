"""Convergence of infinite sums and products, with conditions on the
parameters.

SymPy's ``Sum.is_convergent`` decides the convergence of a series
without parameters (and raises ``NotImplementedError`` with them). Here
the classical tests are run with the parameters carrying assumptions
(:mod:`sympy_extras.assumptions`), and the answer is the condition on the
parameters under which `\\sum_{n \\ge n_0} a_n` converges, the counterpart
of Mathematica's ``SumConvergence``:

1. the *term test*: `a_n \\not\\to 0` means divergence;
2. **d'Alembert's ratio test** with `r = \\lim |a_{n+1}/a_n|`: absolute
   convergence for `r < 1`, divergence for `r > 1`; when `r` depends on
   the parameters the boundary `r = 1` is examined at each of its points
   (found by :func:`~sympy_extras.assumptions.solve`) when there is a
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

References
==========

.. [Knopp] K. Knopp, Theory and Application of Infinite Series, Blackie
   (1951), chapters IX (tests of convergence) and VII (products).
.. [Bromwich] T. J. Bromwich, An Introduction to the Theory of Infinite
   Series, Macmillan (1908), chapter III.
"""
from __future__ import annotations

from typing import Callable, Optional, Union

from sympy.concrete.summations import Sum
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.numbers import pi
from sympy.core.power import Pow
from sympy.core.relational import Eq, Ne, Lt, Le, Gt, Ge
from sympy.core.singleton import S
from sympy.calculus.accumulationbounds import AccumBounds
from sympy.functions.combinatorial.factorials import factorial
from sympy.core.basic import Basic
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.integrals.integrals import Integral, integrate
from sympy.logic.boolalg import Boolean, BooleanTrue, BooleanFalse, And, Or, true, false
from sympy.series.limits import Limit
from sympy.sets.sets import Interval, FiniteSet
from sympy.simplify.combsimp import combsimp
from sympy.simplify.powsimp import powsimp
from sympy.simplify.simplify import simplify as _sympy_simplify

from sympy_extras._timeout import attempt
from sympy_extras._typing import Truth, as_boolean, as_expr, free_symbols
from sympy_extras.settings import settings
from sympy_extras.assumptions.ask import ask, Assumptions, _facts
from sympy_extras.assumptions.analysis import sign_on, certified_sign
from sympy_extras.assumptions.limits import limit as _limit_with_assumptions
from sympy_extras.assumptions.refine import simplify as _simplify_with_assumptions, _Refiner
from sympy_extras.assumptions.solve import solve
from sympy_extras.assumptions.resolve import resolve

__all__ = ['sum_convergence', 'product_convergence', 'is_convergent']

#: a condition on the parameters (``None`` when undecided)
Condition = Optional[Boolean]
#: what a test concludes from the value of a limit
Verdict = Callable[[Expr, '_Context'], Condition]


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
        return And(*self.assumptions) if self.assumptions else true

    def ask(self, statement: Union[Boolean, bool]) -> Truth:
        return ask(as_boolean(statement), self.premise)

    def decide(self, condition: Boolean) -> Boolean:
        """The condition as ``true``, ``false``, or itself refined."""
        value = self.ask(condition)
        if value is True:
            return true
        if value is False:
            return false
        refined = attempt(lambda: _simplify_with_assumptions(condition, self.premise), settings.timeout)
        return as_boolean(refined) if refined is not None else condition

    def eventually(self, statement: Boolean) -> Truth:
        """Whether a statement in ``n`` holds for all large ``n``."""
        for start in (self.lower, self.lower + 4, self.lower + 16, self.lower + 64):
            premise = And(self.premise, self.n >= start)
            # SymPy's own evaluation of the relation with the symbols
            # replaced by dummies carrying their assumptions
            try:
                facts = _facts(premise, None, free_symbols(statement))
            except ValueError:
                return None
            abstracted, _, _ = _Refiner(facts).abstract(statement)
            if abstracted is true:
                return True
            value = ask(statement, premise)
            if value is True:
                return True
        return None


def _clean(e: Expr, n: Symbol) -> Expr:
    """A ratio or a power of terms simplified before its limit is taken
    (SymPy's ``limit`` chokes on ``x**(n + 1)/x**n``): while simplifying,
    ``n`` is a positive integer and the parameters are real, so that the
    absolute values of positive quantities disappear."""
    forward: dict[Basic, Basic] = {n: Dummy('n', integer=True, positive=True)}
    for parameter in free_symbols(e) - {n}:
        if parameter.assumptions0.get('real') is None:
            forward[parameter] = Dummy(parameter.name, real=True)
    back = {d: s for s, d in forward.items()}
    replaced = as_expr(e.xreplace(forward))
    cleaned = attempt(lambda: as_expr(_sympy_simplify(combsimp(powsimp(replaced, force=True)))), settings.timeout)
    if cleaned is None:
        return e
    return as_expr(cleaned.xreplace(back))


def _limit(e: Expr, ctx: _Context, clean: bool = True) -> Optional[Expr]:
    """The limit at infinity under the assumptions, ``None`` when SymPy
    cannot find it."""
    cleaned = _clean(e, ctx.n) if clean else e
    value = attempt(lambda: _limit_with_assumptions(cleaned, ctx.n, S.Infinity, assumptions=ctx.premise),
                    settings.timeout)
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
        expr, cond_ = as_expr(pair.args[0]), as_boolean(pair.args[1])
        part = verdict(expr, ctx.assume(cond_))
        if part is None:
            return None
        if part is not false:
            parts.append(And(cond_, part))
    return Or(*parts) if parts else false


def _compare_with_one(value: Expr, ctx: _Context, above: Condition, below: Condition,
                      boundary: Callable[[_Context], Condition], term: Expr) -> Condition:
    """The condition from a limit compared with 1: ``below`` when it is
    smaller, ``above`` when it is larger, ``boundary(ctx)`` when it is 1;
    for a limit depending on the parameters the three cases are combined,
    the boundary being examined at each point of ``value = 1`` when there
    is a single parameter."""
    if value in (S.Infinity, S.NegativeInfinity):
        return above if value is S.Infinity else below
    if not free_symbols(value):
        s = certified_sign(as_expr(value - 1))
        if s is None:
            return None
        if s < 0:
            return below
        if s > 0:
            return above
        return boundary(ctx)
    less = ctx.decide(as_boolean(value < 1))
    if less is true:
        return below
    greater = ctx.decide(as_boolean(value > 1))
    if greater is true:
        return above
    equal = ctx.decide(as_boolean(Eq(value, 1)))
    if equal is true:
        return boundary(ctx)
    parts: list[Boolean] = []
    if less is not false and below is not None and below is not false:
        parts.append(And(less, below))
    if greater is not false and above is not None and above is not false:
        parts.append(And(greater, above))
    if equal is not false:
        on_boundary = _boundary_points(value, ctx, term)
        if on_boundary is None:
            return None
        if on_boundary is not false:
            parts.append(on_boundary)
    if (below is None and less is not false) or (above is None and greater is not false):
        return None
    return Or(*parts) if parts else false


def _boundary_points(value: Expr, ctx: _Context, term: Expr) -> Condition:
    """The condition for convergence on ``value = 1``: with one
    parameter, the points of the boundary are listed and the series is
    tested at each of them."""
    parameters = sorted(free_symbols(value), key=lambda s: s.name)
    if len(parameters) != 1 or ctx.depth > 4:
        return None
    [p] = parameters
    points = attempt(lambda: solve(Eq(value, 1), p, ctx.premise, domain=S.Reals), settings.timeout)
    if not isinstance(points, FiniteSet):
        return None
    parts: list[Boolean] = []
    for point in points.args:
        if not isinstance(point, Expr) or free_symbols(point):
            return None
        at_point = _convergence(as_expr(term.xreplace({p: point})),
                                _Context(ctx.n, ctx.lower, [a for a in ctx.assumptions if p not in a.free_symbols],
                                         ctx.depth + 1))
        if at_point is None:
            return None
        if at_point is true:
            parts.append(as_boolean(Eq(p, point)))
        elif at_point is not false:
            parts.append(And(Eq(p, point), at_point))
    return Or(*parts) if parts else false


def _alternating(term: Expr, n: Symbol) -> Optional[Expr]:
    """``b_n`` when ``term`` is ``(-1)**(n + c) * b_n`` or
    ``cos(pi*n) * b_n``."""
    factors = term.args if isinstance(term, Mul) else (term,)
    rest: list[Expr] = []
    found = False
    for f in factors:
        f_ = as_expr(f)
        if isinstance(f_, Pow) and f_.base == S.NegativeOne and as_expr(f_.exp).is_polynomial(n) \
                and as_expr(f_.exp).diff(n).is_Integer and abs(int(as_expr(f_.exp).diff(n))) == 1 and not found:
            found = True
            continue
        if isinstance(f_, cos) and as_expr(f_.args[0] - pi*n).is_Integer and not found:
            found = True
            continue
        rest.append(f_)
    if not found:
        return None
    return as_expr(Mul(*rest))


def _oscillating(term: Expr, n: Symbol) -> Optional[Expr]:
    """``b_n`` when ``term`` is ``sin(c*n + d)*b_n`` or ``cos(c*n + d)*b_n``.

    The partial sums of such a factor are bounded, by summing the
    geometric series of ``exp(I*c*n)``, as soon as ``c`` is not a multiple
    of ``2*pi`` (which would make the factor constant).
    """
    factors = term.args if isinstance(term, Mul) else (term,)
    rest: list[Expr] = []
    found = False
    for factor in factors:
        f = as_expr(factor)
        if not found and isinstance(f, (sin, cos)):
            argument = as_expr(f.args[0])
            c = as_expr(argument.diff(n))
            if n not in free_symbols(c) and as_expr(argument - c*n).diff(n) == 0 \
                    and c.is_real and as_expr(c/(2*pi)).is_integer is False:
                found = True
                continue
        rest.append(f)
    if not found:
        return None
    return as_expr(Mul(*rest))


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


def _constant_sign(term: Expr, ctx: _Context) -> Truth:
    """Whether the terms have eventually a constant sign."""
    # SymPy's own knowledge, with n a positive integer and the
    # parameters real numbers (the convention of the conditions returned)
    replacements: dict[Basic, Basic] = {ctx.n: Dummy('n', integer=True, positive=True)}
    for parameter in free_symbols(term) - {ctx.n}:
        if parameter.assumptions0.get('real') is None:
            replacements[parameter] = Dummy(parameter.name, real=True)
    for form in (term, as_expr(term.rewrite(factorial)), as_expr(combsimp(term))):
        at_positive = as_expr(form.xreplace(replacements))
        if at_positive.is_nonnegative or at_positive.is_nonpositive:
            return True
    if not free_symbols(term) - {ctx.n}:
        for start in (ctx.lower, ctx.lower + 8, ctx.lower + 64):
            signs = attempt(lambda: sign_on(term, ctx.n, Interval(start, S.Infinity)), settings.timeout)
            if signs is not None and not (1 in signs and -1 in signs):
                return True
        return None
    if ctx.eventually(as_boolean(term >= 0)) or ctx.eventually(as_boolean(term <= 0)):
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
        difference = as_expr(b.xreplace({ctx_.n: ctx_.n + 1}) - b)
        if ctx_.eventually(as_boolean(difference <= 0)):
            return true
        derivative = as_expr(b.diff(ctx_.n))
        if ctx_.eventually(as_boolean(derivative <= 0)):
            return true
        return None
    return _by_cases(value, ctx, verdict)


def _integral_test(term: Expr, ctx: _Context) -> Condition:
    """For eventually positive decreasing terms, the convergence of the
    integral."""
    if free_symbols(term) - {ctx.n}:
        return None
    if ctx.eventually(as_boolean(term > 0)) is not True:
        return None
    if _decreasing_to_zero(term, ctx) is not true:
        return None
    value = attempt(lambda: as_expr(integrate(term, (ctx.n, ctx.lower, S.Infinity))), settings.timeout)
    if value is None or value.has(Integral) or value.has(ctx.n):
        return None
    if value is S.Infinity:
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
    shifted = term.xreplace({ctx.n: ctx.n + 1})
    ratio = as_expr(Abs(term/shifted)) if inverse else as_expr(Abs(shifted/term))
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
    root = as_expr(Abs(term)**(1/ctx.n))
    c = _limit(root, ctx)
    if c is None:
        return _power_comparison(term, ctx)

    def verdict(value: Expr, ctx_: _Context) -> Condition:
        return _compare_with_one(value, ctx_, above=false, below=true,
                                 boundary=lambda c_: _power_comparison(term, c_), term=term)
    return _by_cases(c, ctx, verdict)


def _raabe_test(term: Expr, ctx: _Context) -> Condition:
    # SymPy's limit is unreliable on the expanded form of the expression:
    # the ratio is simplified alone and the structure n*(ratio - 1) kept
    expression = as_expr(ctx.n*(_ratio(term, ctx, inverse=True) - 1))
    rho = _limit(expression, ctx, clean=False)
    if rho is None:
        return _power_comparison(term, ctx)

    def verdict(value: Expr, ctx_: _Context) -> Condition:
        return _compare_with_one(value, ctx_, above=true, below=_divergence_of_absolute(term, ctx_),
                                 boundary=lambda c: _bertrand_test(term, c), term=term)
    return _by_cases(rho, ctx, verdict)


def _bertrand_test(term: Expr, ctx: _Context) -> Condition:
    expression = as_expr((ctx.n*(_ratio(term, ctx, inverse=True) - 1) - 1)*log(ctx.n))
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
    exponent = as_expr(log(Abs(term))/log(ctx.n))
    q = _limit(exponent, ctx)
    if q is None:
        return _last_resort(term, ctx)

    def verdict(value: Expr, ctx_: _Context) -> Condition:
        return _compare_with_one(as_expr(-value), ctx_, above=true, below=_divergence_of_absolute(term, ctx_),
                                 boundary=lambda c: _last_resort(term, c), term=term)
    return _by_cases(q, ctx, verdict)


def _last_resort(term: Expr, ctx: _Context) -> Condition:
    """The term, alternating and integral tests, and SymPy's own decision
    for terms without parameters."""
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
    if not free_symbols(term) - {ctx.n}:
        value = attempt(lambda: Sum(term, (ctx.n, ctx.lower, S.Infinity)).is_convergent(), settings.timeout)
        if value is not None:
            return true if bool(value) else false
    return None


def _term_test(term: Expr, ctx: _Context) -> Condition:
    """Divergence when the terms do not tend to zero; ``None``
    otherwise."""
    value = _limit(term, ctx)
    if value is None:
        return None

    def verdict(v: Expr, ctx_: _Context) -> Condition:
        if isinstance(v, AccumBounds) or v in (S.Infinity, S.NegativeInfinity):
            return false
        if v is S.NaN:
            return None
        if not free_symbols(v):
            return false if v.is_zero is False else None
        return false if ctx_.decide(as_boolean(Eq(v, 0))) is false else None
    return _by_cases(value, ctx, verdict)


def _convergence(term: Expr, ctx: _Context) -> Condition:
    if not term.has(ctx.n):
        return ctx.decide(as_boolean(Eq(term, 0)))
    # a constant factor which may vanish makes the series trivially
    # convergent (and cancels from the ratios, so it is separated first)
    constant, rest = term.as_independent(ctx.n, as_Add=False)
    c = as_expr(constant)
    if free_symbols(c):
        zero = ctx.decide(as_boolean(Eq(c, 0)))
        if zero is true:
            return true
        if zero is not false:
            nonzero = _ratio_test(as_expr(rest), ctx.assume(as_boolean(Ne(c, 0))))
            if nonzero is None:
                return None
            return Or(zero, nonzero)
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
                       settings.timeout)
    if resolved is not None and not isinstance(resolved, Boolean):
        resolved = None
    if resolved is not None:
        return resolved
    simplified = attempt(lambda: _simplify_with_assumptions(condition, ctx.premise), settings.timeout)
    return as_boolean(simplified) if simplified is not None else condition


def _without_abs(condition: Boolean) -> Boolean:
    """``Abs(u) < c`` written as ``(u < c) & (u > -c)`` and the like."""
    result = condition
    for atom in condition.atoms(Lt, Le, Gt, Ge, Eq, Ne):
        lhs, rhs = as_expr(atom.lhs), as_expr(atom.rhs)
        if isinstance(lhs, Abs) and not rhs.has(Abs):
            u = as_expr(lhs.args[0])
            replacement: Boolean
            if isinstance(atom, (Lt, Le)):
                replacement = And(type(atom)(u, rhs), type(atom)(-u, rhs))
            elif isinstance(atom, (Gt, Ge)):
                replacement = Or(type(atom)(u, rhs), type(atom)(-u, rhs))
            elif isinstance(atom, Eq):
                replacement = Or(Eq(u, rhs), Eq(u, -rhs))
            else:
                replacement = And(Ne(u, rhs), Ne(u, -rhs))
            result = as_boolean(result.xreplace({atom: replacement}))
    return result


def _context(n: Symbol, lower: Union[Expr, int], assumptions: Assumptions) -> _Context:
    items: list[Boolean] = []
    if isinstance(assumptions, (Boolean, bool)):
        items.append(as_boolean(assumptions))
    elif assumptions is not None:
        items.extend(as_boolean(a) for a in assumptions)
    return _Context(n, as_expr(sympify(lower)), items, 0)


def sum_convergence(term: Union[Expr, int], n: Symbol, assumptions: Assumptions = None,
                    lower: Union[Expr, int] = 1) -> Condition:
    """The condition on the parameters under which ``Sum(term, (n, lower,
    oo))`` converges: ``S.true``, ``S.false``, a Boolean in the
    parameters, or ``None`` when the tests do not decide.

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

    >>> from sympy import factorial, binomial, log
    >>> from sympy.abc import n, x, p
    >>> from sympy_extras.concrete import sum_convergence
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
    >>> sum_convergence(1/(n*log(n)), n, lower=2)
    False
    """
    term_ = as_expr(sympify(term))
    ctx = _context(n, lower, assumptions)
    return _finish(_convergence(term_, ctx), ctx)


def is_convergent(term: Union[Expr, int], n: Symbol, assumptions: Assumptions = None,
                  lower: Union[Expr, int] = 1) -> Truth:
    """Whether ``Sum(term, (n, lower, oo))`` converges under the
    assumptions: ``True``, ``False`` or ``None``.

    Examples
    ========

    >>> from sympy.abc import n, x
    >>> from sympy_extras.concrete import is_convergent
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


def product_convergence(term: Union[Expr, int], n: Symbol, assumptions: Assumptions = None,
                        lower: Union[Expr, int] = 1) -> Condition:
    """The condition under which the infinite product ``Product(term,
    (n, lower, oo))`` converges to a nonzero limit (see the module
    documentation).

    Examples
    ========

    >>> from sympy.abc import n, x
    >>> from sympy_extras.concrete import product_convergence
    >>> product_convergence(1 + 1/n**2, n)
    True
    >>> product_convergence(1 + x/n**2, n)
    True
    >>> product_convergence(1 + 1/n, n)
    False
    >>> product_convergence(1 + x**n, n)
    (x > -1) & (x < 1)
    """
    term_ = as_expr(sympify(term))
    ctx = _context(n, lower, assumptions)
    b = as_expr(term_ - 1)
    if not b.has(n):
        return _finish(ctx.decide(as_boolean(Eq(b, 0))), ctx)
    value = _limit(b, ctx)
    if value is not None and not isinstance(value, Piecewise) and not free_symbols(value) and value.is_zero is False:
        return false
    positive = Dummy('n', integer=True, positive=True)
    magnitude = as_expr(as_expr(Abs(b.xreplace({n: positive}))).xreplace({positive: n}))
    absolute = _convergence(magnitude, ctx)
    if absolute is true:
        return true
    if absolute is not None and not isinstance(absolute, (BooleanTrue, BooleanFalse)):
        return _finish(absolute, ctx)
    if absolute is false and _constant_sign(b, ctx) is True:
        return false
    logarithmic = _convergence(as_expr(log(term_)), ctx)
    return _finish(logarithmic, ctx)
