"""Randomised cross-checking of sympy-extras against independent oracles.

Every section below generates random inputs for one part of the package
and checks the answer against something which does not share any code with
it: numerical evaluation with mpmath, brute force enumeration, sign
changes on a fine grid, sampling of the solution set, ``checkodesol``, or
SymPy's own implementation on a specialised instance (a parametric answer
is checked by substituting numbers into it and asking SymPy the resulting
parameter-free question).

The sections are ``convergence``, ``parametric-convergence``, ``sums``,
``isolation``, ``solve``, ``ask``, ``limits``, ``thue``, ``ode`` and
``refine``.

    python benchmarks/fuzz.py                       # every section
    python benchmarks/fuzz.py convergence sums      # some sections
    python benchmarks/fuzz.py --count 200 --seed 7  # more cases

A failure is printed with the input which produced it, so that it can be
turned into a unit test: as ``AGENTS.md`` requires, every bug found here
is fixed with a test which fails on the unfixed code.
"""
from __future__ import annotations

import random
import sys
import traceback
from typing import Callable, Optional, Sequence

import mpmath

from sympy import (Add, Eq, Expr, Function, I, Integer, Interval, Mul, N, Pow, Rational, S, Set,
    Symbol, binomial, cos, exp, factorial, harmonic, log, oo, sin, sqrt, symbols, lambdify, Sum,
    simplify)
from sympy.core.function import AppliedUndef
from sympy.core.relational import Equality
from sympy.logic.boolalg import Boolean, BooleanTrue, BooleanFalse, And, true

from sympy_extras._timeout import attempt, time_limit, TimeLimitExceeded
from sympy_extras._typing import as_boolean, as_expr
from sympy_extras.assumptions import ask, element, solve, limit
from sympy_extras.concrete import sum_convergence, polygamma_series, dirichlet_series, karr_sum, rational_sum
from sympy_extras.solvers import isolate_real_roots, thue
from sympy_extras.solvers.isolation import TranscendentalRoot

#: the time limit of one case
TIMEOUT = 25.0

x, y, n, k, a, b, c, p, s = symbols('x y n k a b c p s')

#: one case: a label and a check returning ``None`` (passed) or a message
Check = Callable[[], Optional[str]]
Generator = Callable[[random.Random], tuple[str, Check]]


# ---------------------------------------------------------------------------
# helpers

def _numeric(expr: Expr, value: object) -> Optional[float]:
    """``expr`` evaluated at a number, as a float, or ``None``."""
    try:
        with time_limit(5.0):
            result = N(expr, 25)
    except (TimeLimitExceeded, Exception):          # noqa: BLE001 - any failure is 'unknown'
        return None
    if not (isinstance(result, Expr) and result.is_number and result.is_real):
        return None
    try:
        return float(result)
    except (TypeError, ValueError):
        return None


def _partial_sum(term: Expr, variable: Symbol, lower: int, upper: int) -> Optional[float]:
    """The partial sum, evaluated term by term with mpmath."""
    try:
        f = lambdify(variable, term, modules=['mpmath'])
    except Exception:                                # noqa: BLE001
        return None
    # the index has to be an mpmath number, or a power such as ``4**(-n)``
    # is evaluated in machine floats and underflows to zero (which made
    # the divergent ``binomial(2*n, n)/4**n`` look convergent); a negative
    # numeric base needs an integer instead, or the power turns complex
    integral = any(isinstance(power, Pow) and as_expr(power.base).is_negative
                   and as_expr(power.exp).has(variable) for power in term.atoms(Pow))
    total = mpmath.mpf(0)
    try:
        for i in range(lower, upper + 1):
            total += mpmath.mpf(f(i if integral else mpmath.mpf(i)))
    except Exception:                                # noqa: BLE001
        return None
    return float(total)


def _converges_numerically(term: Expr, variable: Symbol) -> Optional[bool]:
    """Whether the series converges, from its partial sums (``None`` when
    the answer is not clear enough to be used as an oracle).

    The increments of the partial sums over successive doublings decide:
    they shrink geometrically for a convergent series, and stay of the
    same size for a divergent one.  Anything in between is ``None``.
    """
    values = [_partial_sum(term, variable, 1, m) for m in (1000, 2000, 4000, 8000)]
    if any(v is None for v in values):
        return None
    s1000, s2000, s4000, s8000 = [v for v in values if v is not None]
    size = max(abs(s8000), 1.0)
    first, last = abs(s2000 - s1000), abs(s8000 - s4000)
    if last <= 0.35*first and last < 3e-4*size:
        return True
    if first > 0 and last >= 0.9*first and last > 1e-3*size:
        return False
    return None


def _random_point(rng: random.Random, lo: float, hi: float) -> Rational:
    return Rational(rng.randint(int(lo*8), int(hi*8)), 8)


# ---------------------------------------------------------------------------
# convergence of series

def _convergence_term(rng: random.Random) -> Expr:
    """A random summand in ``n`` without parameters."""
    shapes = [
        lambda: 1/n**rng.randint(1, 4),
        lambda: Rational(rng.randint(1, 3), rng.randint(1, 3))**n,
        lambda: 1/(n*log(n + 1)**rng.randint(0, 3)),
        lambda: (-1)**n/n**rng.randint(1, 3),
        lambda: factorial(n)/n**n,
        lambda: binomial(2*n, n)/Integer(4)**n/n**rng.randint(0, 2),
        lambda: 1/(n**rng.randint(1, 3) + rng.randint(1, 5)),
        lambda: sin(n)/n**rng.randint(1, 3),
        lambda: n**rng.randint(1, 3)*Rational(1, rng.randint(2, 4))**n,
        lambda: log(n + 1)/n**rng.randint(1, 3),
        lambda: 1/(n*(n + 1)),
        lambda: (-1)**n*log(n + 1)/n,
    ]
    return as_expr(rng.choice(shapes)())


def convergence_case(rng: random.Random) -> tuple[str, Check]:
    term = _convergence_term(rng)
    label = 'sum_convergence(%s, n)' % term

    def check() -> Optional[str]:
        answer = attempt(lambda: sum_convergence(term, n), TIMEOUT)
        if answer is None or not isinstance(answer, (BooleanTrue, BooleanFalse)):
            return None                              # undecided is not an error
        ours = bool(answer)
        theirs = attempt(lambda: Sum(term, (n, 1, oo)).is_convergent(), TIMEOUT)
        if theirs is not None and bool(theirs) != ours:
            numeric = _converges_numerically(term, n)
            if numeric is not None and numeric != ours:
                return 'we say %s, SymPy and the partial sums say %s' % (ours, numeric)
            if numeric is None:
                return 'we say %s, SymPy says %s (partial sums inconclusive)' % (ours, bool(theirs))
        numeric = _converges_numerically(term, n)
        if numeric is not None and numeric != ours:
            return 'we say %s, the partial sums say %s' % (ours, numeric)
        return None
    return label, check


def parametric_convergence_case(rng: random.Random) -> tuple[str, Check]:
    """A summand with a parameter: the condition returned is checked by
    substituting numbers into it and asking SymPy the specialised
    question."""
    shapes = [
        (a**n/n**rng.randint(1, 2), a),
        (n**p, p),
        (1/n**p, p),
        (a**n*n**rng.randint(1, 2), a),
        ((-1)**n*n**p, p),
        (1/(n**p + 1), p),
    ]
    term, parameter = shapes[rng.randrange(len(shapes))]
    label = 'sum_convergence(%s, n) at %s' % (term, parameter)

    def check() -> Optional[str]:
        condition = attempt(lambda: sum_convergence(term, n), TIMEOUT)
        if condition is None:
            return None
        for _ in range(6):
            value = _random_point(rng, -3, 3)
            claimed = attempt(lambda: ask(as_boolean(condition.subs(parameter, value))), TIMEOUT)
            if claimed is None:
                continue
            special = term.subs(parameter, value)
            theirs = attempt(lambda: Sum(special, (n, 1, oo)).is_convergent(), TIMEOUT)
            if theirs is None:
                theirs_numeric = _converges_numerically(special, n)
                if theirs_numeric is None or theirs_numeric == claimed:
                    continue
                return 'at %s = %s we say %s, the partial sums say %s (condition %s)' % (
                    parameter, value, claimed, theirs_numeric, condition)
            if bool(theirs) != claimed:
                numeric = _converges_numerically(special, n)
                if numeric is not None and numeric == claimed:
                    continue                          # SymPy is the one which is wrong
                return 'at %s = %s we say %s, SymPy says %s (condition %s)' % (
                    parameter, value, claimed, bool(theirs), condition)
        return None
    return label, check


# ---------------------------------------------------------------------------
# closed forms of sums

def sums_case(rng: random.Random) -> tuple[str, Check]:
    """A closed form of a sum, checked against its partial sums."""
    kind = rng.choice(['karr', 'rational', 'euler', 'dirichlet'])
    if kind == 'karr':
        shapes = [harmonic(k), harmonic(k)**2, harmonic(k)/(k + 1), k*harmonic(k),
                  harmonic(k, 2), harmonic(k)/(k + 1)**2, k**2*harmonic(k)]
        term = as_expr(rng.choice(shapes))
        label = 'karr_sum(%s, (k, 1, n))' % term

        def check_karr() -> Optional[str]:
            closed = attempt(lambda: karr_sum(term, (k, 1, n)), TIMEOUT)
            if closed is None:
                return None
            for m in (3, 7, 12):
                exact = _partial_sum(term, k, 1, m)
                got = _numeric(as_expr(closed.subs(n, m)), m)
                if exact is None or got is None:
                    continue
                if abs(exact - got) > 1e-9*max(1.0, abs(exact)):
                    return 'at n = %d the closed form gives %r, the sum is %r' % (m, got, exact)
            return None
        return label, check_karr
    if kind == 'rational':
        denominator = Mul(*[k + rng.randint(1, 4) for _ in range(rng.randint(1, 2))])
        term = as_expr(Integer(rng.randint(1, 3))/(denominator*(k + rng.randint(5, 7))))
        label = 'rational_sum(%s, (k, 1, n))' % term

        def check_rational() -> Optional[str]:
            closed = attempt(lambda: rational_sum(term, (k, 1, n)), TIMEOUT)
            if closed is None or closed.has(Sum):
                return None
            for m in (3, 8, 15):
                exact = _partial_sum(term, k, 1, m)
                got = _numeric(as_expr(closed.subs(n, m)), m)
                if exact is None or got is None:
                    continue
                if abs(exact - got) > 1e-9*max(1.0, abs(exact)):
                    return 'at n = %d the closed form gives %r, the sum is %r' % (m, got, exact)
            return None
        return label, check_rational
    if kind == 'euler':
        q = rng.randint(2, 5)
        shift = rng.randint(-2, 3)
        weight = rng.randint(1, 3)
        lower = max(1, 1 - shift)
        term = as_expr(harmonic(n + shift, weight)/n**q)
        label = 'polygamma_series(%s, n, %d)' % (term, lower)

        def check_euler() -> Optional[str]:
            closed = attempt(lambda: polygamma_series(term, n, lower), TIMEOUT)
            if closed is None:
                return None
            got = _numeric(as_expr(closed), 0)
            if got is None:
                return None
            mpmath.mp.dps = 25

            def summand(i: mpmath.mpf) -> mpmath.mpf:
                h = (mpmath.psi(0, i + shift + 1) + mpmath.euler if weight == 1
                     else mpmath.zeta(weight) - mpmath.zeta(weight, i + shift + 1))
                return h/i**q
            try:
                exact = float(mpmath.nsum(summand, [lower, mpmath.inf], method='euler-maclaurin'))
            except Exception:                        # noqa: BLE001
                return None
            if abs(exact - got) > 1e-9*max(1.0, abs(exact)):
                return 'the closed form is %r, the series is %r' % (got, exact)
            return None
        return label, check_euler
    coefficient = rng.choice(['1', 'mobius', 'totient', 'divisor_sigma', 'alternating'])
    exponent = rng.randint(4, 7)
    from sympy import mobius, totient, divisor_sigma
    terms = {'1': Integer(1), 'mobius': mobius(n), 'totient': totient(n),
             'divisor_sigma': divisor_sigma(n), 'alternating': (-1)**(n + 1)}
    term = as_expr(terms[coefficient]/n**exponent)
    label = 'dirichlet_series(%s, n)' % term

    def check_dirichlet() -> Optional[str]:
        found = attempt(lambda: dirichlet_series(term, n), TIMEOUT)
        if found is None:
            return None
        got = _numeric(as_expr(found[0]), 0)
        exact = _partial_sum(term, n, 1, 3000)
        if got is None or exact is None:
            return None
        if abs(exact - got) > 1e-6*max(1.0, abs(exact)):
            return 'the closed form is %r, the partial sum is %r' % (got, exact)
        return None
    return label, check_dirichlet


# ---------------------------------------------------------------------------
# real root isolation

def _isolation_function(rng: random.Random) -> Expr:
    pieces = [
        lambda: exp(x) - Integer(rng.randint(1, 4))*x - rng.randint(-3, 3),
        lambda: cos(x) - x**rng.randint(1, 3)/rng.randint(1, 4),
        lambda: sin(x) - x/rng.randint(2, 4),
        lambda: x**rng.randint(2, 4) - rng.randint(1, 6)*x - rng.randint(-4, 4),
        lambda: exp(-x**2) - Rational(1, rng.randint(2, 5)),
        lambda: log(x**2 + 1) - Rational(1, rng.randint(1, 4)),
        lambda: x*exp(x) - rng.randint(1, 4),
        lambda: sin(x)*exp(-x/4) - Rational(1, rng.randint(2, 6)),
        lambda: x**3 - 3*x + Rational(rng.randint(-4, 4), 2),
        lambda: cos(x) + x**2/rng.randint(2, 6) - rng.randint(1, 3),
    ]
    return as_expr(rng.choice(pieces)())


def isolation_case(rng: random.Random) -> tuple[str, Check]:
    f = _isolation_function(rng)
    lo, hi = -4, 4
    domain = Interval(lo, hi)
    label = 'isolate_real_roots(%s, x, %s)' % (f, domain)

    def check() -> Optional[str]:
        roots = attempt(lambda: isolate_real_roots(f, x, domain), TIMEOUT)
        if roots is None:
            return None
        # an independent count: sign changes on a fine grid
        try:
            g = lambdify(x, f, modules=['mpmath'])
        except Exception:                            # noqa: BLE001
            return None
        steps = 4000
        grid = [lo + (hi - lo)*i/steps for i in range(steps + 1)]
        values: list[Optional[float]] = []
        for point in grid:
            try:
                v = g(point)
                values.append(float(v) if isinstance(v, (int, float, mpmath.mpf)) else None)
            except Exception:                        # noqa: BLE001
                values.append(None)
        # a lower bound on the number of roots: the sign changes between
        # consecutive non-zero values, plus the grid points which land on a
        # root without a sign change around them (a root counts once,
        # whether or not the grid happens to hit it exactly)
        crossings = 0
        previous: Optional[float] = None
        zeros = False
        for value in values:
            if value is None:
                continue
            if value == 0.0:
                zeros = True
                continue
            if previous is not None:
                if (previous < 0) != (value < 0):
                    crossings += 1
                elif zeros:
                    crossings += 1                   # a root the function touches
            previous, zeros = value, False
        if len(roots) < crossings:
            return 'we found %d roots, the grid shows at least %d sign changes' % (len(roots), crossings)
        for root in roots:
            value = _numeric(as_expr(f.subs(x, root)), 0)
            if isinstance(root, TranscendentalRoot):
                if not (root.lower <= N(root, 25) <= root.upper):
                    return 'the root %s is outside its isolating interval' % root
                left = _numeric(as_expr(f.subs(x, root.lower)), 0)
                right = _numeric(as_expr(f.subs(x, root.upper)), 0)
                if left is not None and right is not None and left*right > 0:
                    return 'no sign change over the isolating interval of %s' % root
            elif value is not None and abs(value) > 1e-8:
                return 'the exact root %s has residual %r' % (root, value)
        return None
    return label, check


# ---------------------------------------------------------------------------
# solution sets

def _solve_relation(rng: random.Random) -> Boolean:
    left = rng.choice([x**2 - rng.randint(1, 6), x**3 - rng.randint(1, 4)*x,
                       exp(x) - rng.randint(1, 4), cos(x) - Rational(rng.randint(-2, 2), 3),
                       x**2 - rng.randint(1, 4)*x + rng.randint(-2, 3),
                       log(x + 5) - Rational(rng.randint(0, 4), 3),
                       sqrt(x + 4) - Rational(rng.randint(1, 4), 2),
                       sin(x) - Rational(rng.randint(-2, 2), 3),
                       sin(rng.randint(1, 3)*x) - Rational(rng.randint(-2, 2), 3),
                       sin(x) + cos(x) - Rational(rng.randint(-2, 2), 3),
                       sin(x)*cos(x) - Rational(rng.randint(-2, 2), 3),
                       sin(x) + sin(2*x)])
    kind = rng.choice(['eq', 'gt', 'lt', 'ge'])
    if kind == 'eq':
        return Eq(left, 0)
    if kind == 'gt':
        return left > 0
    if kind == 'lt':
        return left < 0
    return left >= 0


def solve_case(rng: random.Random) -> tuple[str, Check]:
    relation = _solve_relation(rng)
    if rng.random() < 0.5:                           # half of the cases on a bounded region
        low, high = rng.randint(-8, 0), rng.randint(1, 9)
        region: Boolean = as_boolean((x > low) & (x < high))
    else:
        region = as_boolean(true)
    label = 'solve(%s, x, %s, domain=S.Reals)' % (relation, region)

    def check() -> Optional[str]:
        result = attempt(lambda: solve(relation, x, region, domain=S.Reals), TIMEOUT)
        if result is None or result.has(Symbol('ConditionSet')):
            return None
        if not isinstance(result, Set) or result.has(Function('ConditionSet')):
            return None
        for _ in range(25):
            point = _random_point(rng, -6, 6)
            try:
                with time_limit(5.0):
                    inside = result.contains(point)
            except (TimeLimitExceeded, Exception):   # noqa: BLE001
                continue
            if not isinstance(inside, (BooleanTrue, BooleanFalse)):
                continue
            truth = attempt(lambda: as_boolean(And(relation, region).subs(x, point)), 5.0)
            if truth is None or not isinstance(truth, (BooleanTrue, BooleanFalse)):
                continue
            if bool(inside) != bool(truth):
                return 'at x = %s the set says %s, the relation is %s (set %s)' % (
                    point, bool(inside), bool(truth), result)
        return None
    return label, check


# ---------------------------------------------------------------------------
# ask on polynomial statements

def _ask_statement(rng: random.Random) -> tuple[Boolean, Boolean]:
    variables = [x, y][:rng.randint(1, 2)]
    def poly() -> Expr:
        terms = []
        for v in variables:
            terms.append(Integer(rng.randint(-3, 3))*v**rng.randint(1, 2))
        terms.append(Integer(rng.randint(-3, 3)))
        return as_expr(Add(*terms))
    assumptions: list[Boolean] = []
    for v in variables:
        lo = rng.randint(-3, 1)
        assumptions.append(v > lo)
        if rng.random() < 0.5:
            assumptions.append(v < lo + rng.randint(1, 4))
    statement = rng.choice([poly() > 0, poly() >= 0, Eq(poly(), 0), poly() < 0])
    return as_boolean(statement), as_boolean(And(*assumptions))


def ask_case(rng: random.Random) -> tuple[str, Check]:
    statement, assumptions = _ask_statement(rng)
    label = 'ask(%s, %s)' % (statement, assumptions)

    def check() -> Optional[str]:
        answer = attempt(lambda: ask(statement, assumptions), TIMEOUT)
        if answer is None:
            return None
        for _ in range(200):
            values = {v: _random_point(rng, -6, 6) for v in (x, y)}
            holds = assumptions.subs(values)
            if not isinstance(holds, (BooleanTrue, BooleanFalse)) or not bool(holds):
                continue
            truth = statement.subs(values)
            if not isinstance(truth, (BooleanTrue, BooleanFalse)):
                continue
            if bool(truth) != answer:
                return 'we say %s, at %s the statement is %s' % (
                    answer, {str(v): str(w) for v, w in values.items()}, bool(truth))
        return None
    return label, check


# ---------------------------------------------------------------------------
# limits

def limits_case(rng: random.Random) -> tuple[str, Check]:
    shapes = [exp(a*x), x**a, a**x, x**a*log(x), (1 + a/x)**x, a*exp(-x), exp(a*x)/x, x**2*a**x]
    expression = as_expr(rng.choice(shapes))
    at_infinity = rng.random() < 0.7
    point = oo if at_infinity else Integer(0)
    label = 'limit(%s, x, %s)' % (expression, point)

    def check() -> Optional[str]:
        ours = attempt(lambda: limit(expression, x, point), TIMEOUT)
        if ours is None:
            return None
        from sympy import limit as sympy_limit
        for _ in range(6):
            value = _random_point(rng, -3, 3)
            if value == 0:
                continue
            specialised = attempt(lambda: as_expr(ours.subs(a, value)), 5.0)
            theirs = attempt(lambda: sympy_limit(expression.subs(a, value), x, point), TIMEOUT)
            if specialised is None or theirs is None:
                continue
            if specialised.has(Function('Limit')) or specialised.has(Function('Piecewise')):
                continue
            if specialised == theirs:
                continue
            difference = attempt(lambda: simplify(specialised - theirs), 5.0)
            if difference is not None and difference == 0:
                continue
            return 'at a = %s we say %s, SymPy says %s' % (value, specialised, theirs)
        return None
    return label, check


# ---------------------------------------------------------------------------
# Thue equations

def thue_case(rng: random.Random) -> tuple[str, Check]:
    while True:
        coefficients = [rng.randint(-3, 3) for _ in range(4)]
        coefficients[0] = rng.choice([1, 1, 1, 2, -1])
        form = Add(*[Integer(coefficient)*x**(3 - i)*y**i for i, coefficient in enumerate(coefficients)])
        from sympy import Poly, ZZ
        univariate = Poly(form.subs(y, 1), x, domain=ZZ)
        if univariate.degree() == 3 and univariate.is_irreducible:
            break
    m = rng.choice([1, -1, 2, 3, 5, 7])
    label = 'thue(%s, %d, x, y)' % (form, m)

    def check() -> Optional[str]:
        found = attempt(lambda: thue(form, m, x, y), 120.0)
        if found is None:
            return None
        box = 40
        brute = sorted((i, j) for i in range(-box, box + 1) for j in range(-box, box + 1)
                       if form.subs({x: i, y: j}) == m)
        inside = sorted(t for t in found if abs(t[0]) <= box and abs(t[1]) <= box)
        if inside != brute:                          # solutions outside the box are not compared
            missing = [t for t in brute if t not in found]
            extra = [t for t in inside if t not in brute]
            return 'missing %s, spurious %s (we found %s)' % (missing, extra, found)
        return None
    return label, check


# ---------------------------------------------------------------------------
# ordinary differential equations

def _unknown(t: Symbol) -> AppliedUndef:
    """``y(t)``, as the type of the unknown of an equation."""
    y = Function('y')(t)
    if not isinstance(y, AppliedUndef):
        raise TypeError('y(t) is not an applied undefined function')
    return y


def _ode(rng: random.Random) -> tuple[Equality, AppliedUndef]:
    """A random ordinary differential equation and its unknown."""
    t = Symbol('t')
    y = _unknown(t)
    d1, d2 = y.diff(t), y.diff(t, 2)
    p, q = Integer(rng.randint(-3, 3)), Integer(rng.randint(-3, 3))
    shapes = [
        lambda: Eq(d1 + p*y, Integer(rng.randint(-2, 2))*t),
        lambda: Eq(d1, p*y**2 + q*y),                              # Bernoulli/Riccati
        lambda: Eq(t*d1 + y, t**rng.randint(1, 3)),
        lambda: Eq(d1, (t + y)/(t - y) if rng.random() < 0.5 else y/t + t),
        lambda: Eq(d2 + p*d1 + q*y, 0),                            # constant coefficients
        lambda: Eq(d2 + p*d1 + q*y, exp(t)),
        lambda: Eq(t**2*d2 + p*t*d1 + q*y, 0),                     # Euler-Cauchy
        lambda: Eq(d2, -y/t**2),
        lambda: Eq(d2 - 2*t*d1 + 2*p*y, 0),                        # Hermite
        lambda: Eq(d1*y + t, 0),
    ]
    equation = rng.choice(shapes)()
    if not isinstance(equation, Equality):
        raise TypeError('an equation is expected')
    return equation, y


def ode_case(rng: random.Random) -> tuple[str, Check]:
    equation, y = _ode(rng)
    label = 'solve_ode(%s, %s)' % (equation, y)

    def check() -> Optional[str]:
        from sympy import checkodesol
        from sympy_extras.solvers import solve_ode
        solutions = attempt(lambda: solve_ode(equation, y), TIMEOUT)
        if not solutions:
            return None
        from sympy import Order
        for solution in solutions:
            if solution.has(Order):
                continue                             # a truncated power series
            verified = attempt(lambda: checkodesol(equation, solution, func=y), TIMEOUT)
            if verified is None:
                continue
            ok = verified[0] if isinstance(verified, tuple) else verified
            if ok is False:
                # checkodesol reduces the residual: a non-zero one is
                # checked once more after a simplification
                residual = attempt(lambda: simplify(as_expr(verified[1])), TIMEOUT) \
                    if isinstance(verified, tuple) else None
                if residual is not None and residual == 0:
                    continue
                return 'the solution %s does not satisfy the equation (residual %s)' % (
                    solution, verified[1] if isinstance(verified, tuple) else verified)
        return None
    return label, check


# ---------------------------------------------------------------------------
# refinement and simplification under assumptions

def _refine_expression(rng: random.Random) -> tuple[Expr, Boolean]:
    """An expression and assumptions under which it can be refined."""
    from sympy import Abs, sign, floor, Max, Min, re, im
    shapes = [
        lambda: (as_expr(Abs(x)), as_boolean(x > 0)),
        lambda: (as_expr(Abs(x*y)), as_boolean((x > 0) & (y < 0))),
        lambda: (as_expr(sqrt(x**2)), as_boolean(x < 0)),
        lambda: (as_expr((x**2)**Rational(1, 2)), as_boolean(x > 0)),
        lambda: (as_expr(log(exp(x))), as_boolean(x > 0)),
        lambda: (as_expr(exp(log(x))), as_boolean(x > 0)),
        lambda: (as_expr(sign(x*y)), as_boolean((x < 0) & (y > 0))),
        lambda: (as_expr(floor(x)), as_boolean(element(x, S.Integers))),
        lambda: (as_expr(Max(x, y)), as_boolean(x > y)),
        lambda: (as_expr(Min(x, y)), as_boolean(x > y)),
        lambda: (as_expr(re(x + I*y)), as_boolean((x > 0) & (y > 0))),
        lambda: (as_expr(im(x + I*y)), as_boolean((x > 0) & (y > 0))),
        lambda: (as_expr(Abs(x)**2), as_boolean(x < 0)),
        lambda: (as_expr(sqrt(x**2*y**2)), as_boolean((x > 0) & (y > 0))),
        lambda: (as_expr(log(x*y)), as_boolean((x > 0) & (y > 0))),
        lambda: (as_expr(Abs(x + y)), as_boolean((x > 0) & (y > 0))),
    ]
    expression, assumption = rng.choice(shapes)()
    shift = Integer(rng.randint(-2, 2))
    return as_expr(expression + shift), assumption


def refine_case(rng: random.Random) -> tuple[str, Check]:
    """A refinement, checked by evaluating both forms at points which
    satisfy the assumptions."""
    from sympy_extras.assumptions import refine as xrefine, simplify as xsimplify
    expression, assumption = _refine_expression(rng)
    which = rng.choice(['refine', 'simplify'])
    label = '%s(%s, %s)' % (which, expression, assumption)

    def check() -> Optional[str]:
        function = xrefine if which == 'refine' else xsimplify
        refined = attempt(lambda: function(expression, assumption), TIMEOUT)
        if refined is None or not isinstance(refined, Expr):
            return None
        if refined == expression:
            return None
        for _ in range(200):
            values = {x: _random_point(rng, -4, 4), y: _random_point(rng, -4, 4)}
            holds = attempt(lambda: as_boolean(assumption.subs(values)), 5.0)
            if not isinstance(holds, (BooleanTrue, BooleanFalse)) or not bool(holds):
                continue
            before = _numeric(as_expr(expression.subs(values)), 0)
            after = _numeric(as_expr(refined.subs(values)), 0)
            if before is None or after is None:
                continue
            if abs(before - after) > 1e-9*max(1.0, abs(before)):
                return 'at %s the expression is %r and the refinement %r (%s)' % (
                    {str(v): str(w) for v, w in values.items()}, before, after, refined)
        return None
    return label, check


SECTIONS: dict[str, Generator] = {
    'convergence': convergence_case,
    'parametric-convergence': parametric_convergence_case,
    'sums': sums_case,
    'isolation': isolation_case,
    'solve': solve_case,
    'ask': ask_case,
    'limits': limits_case,
    'thue': thue_case,
    'ode': ode_case,
    'refine': refine_case,
}


def run(names: Sequence[str], count: int, seed: int) -> int:
    failures = 0
    for name in names:
        generator = SECTIONS[name]
        rng = random.Random(seed)
        print('== %s (%d cases, seed %d)' % (name, count, seed))
        for index in range(count):
            case_rng = random.Random(rng.randrange(10**9))
            try:
                label, check = generator(case_rng)
            except Exception:                        # noqa: BLE001
                traceback.print_exc()
                continue
            try:
                message = check()
            except Exception as error:               # noqa: BLE001 - a crash is a failure
                message = 'raised %s: %s' % (type(error).__name__, str(error).split('\n')[0][:120])
            if message is not None:
                failures += 1
                print('  FAIL [%d] %s\n        %s' % (index, label, message))
        print('  %d failures so far' % failures)
    return failures


def main() -> int:
    arguments = [argument for argument in sys.argv[1:] if not argument.startswith('--')]
    count = 40
    seed = 0
    for i, argument in enumerate(sys.argv):
        if argument == '--count':
            count = int(sys.argv[i + 1])
        if argument == '--seed':
            seed = int(sys.argv[i + 1])
    names = [name for name in arguments if name in SECTIONS] or list(SECTIONS)
    failures = run(names, count, seed)
    print('\n%d failures' % failures)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
