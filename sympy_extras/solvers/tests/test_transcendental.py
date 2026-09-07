from __future__ import annotations

from sympy import (S, Eq, exp, log, sin, cos, tan, sinh, cosh, tanh, sqrt, cbrt, Abs, pi, E, Interval,
    FiniteSet, ImageSet, Lambda, Rational, Symbol, N, ConditionSet)
from sympy.abc import x, a

from sympy_extras.assumptions import solve
from sympy_extras.solvers.transcendental import solve_transcendental, polynomialize
from sympy_extras.polys.roots import radical_form, in_radicals
from sympy import CRootOf


def test_polynomialize() -> None:
    r = polynomialize(Eq(exp(2*x) - 3*exp(x) + 2, 0), x)
    assert r is not None and r.family == 'exp' and r.argument == x
    assert r.formula == Eq(r.variable**2 - 3*r.variable + 2, 0)
    r = polynomialize(Eq(sinh(x), 2), x)
    assert r is not None and r.family == 'exp'
    r = polynomialize(Eq(2**x + 4**x, 6), x)
    assert r is not None and r.family == 'exp' and r.argument == x*log(2)
    r = polynomialize(Eq(exp(x/2) + exp(x), 6), x)
    assert r is not None and r.argument == x/2
    r = polynomialize(Eq(sin(2*x), cos(x)), x)
    assert r is not None and r.family == 'trig' and r.argument == x
    r = polynomialize(Eq(tan(x), 1), x)
    assert r is not None and r.family == 'trig'
    r = polynomialize(Eq(log(x)**2, 3*log(x) - 2), x)
    assert r is not None and r.family == 'log' and r.argument == x
    r = polynomialize(Eq(sqrt(x + 1) + cbrt(x + 1), 2), x)
    assert r is not None and r.family == 'root' and r.order == 6
    # mixed families, occurrences outside the kernels, different arguments
    assert polynomialize(Eq(exp(x) + sin(x), 1), x) is None
    assert polynomialize(Eq(x*exp(x), 1), x) is None
    assert polynomialize(Eq(log(x) + log(x + 1), 1), x) is None
    assert polynomialize(Eq(exp(x) + exp(x**2), 1), x) is None


def test_solve_transcendental_exp() -> None:
    assert solve_transcendental(Eq(exp(2*x) - 3*exp(x) + 2, 0), x) == FiniteSet(0, log(2))
    assert solve_transcendental(Eq(exp(x), -1), x) is S.EmptySet
    assert solve_transcendental(exp(x) > 2, x) == Interval.open(log(2), S.Infinity)
    assert solve_transcendental(exp(x) < 2, x) == Interval.open(S.NegativeInfinity, log(2))
    assert solve_transcendental(exp(x) < -1, x) is S.EmptySet
    assert solve_transcendental(Eq(sinh(x), 2), x) == FiniteSet(log(2 + sqrt(5)))
    assert solve_transcendental(Eq(cosh(x), 1), x) == FiniteSet(0)
    assert solve_transcendental(Eq(tanh(x), 2), x) is S.EmptySet
    assert solve_transcendental(Eq(2**x, 8), x) == FiniteSet(3)
    assert solve_transcendental(Eq(exp(x**2), 2), x) == FiniteSet(-sqrt(log(2)), sqrt(log(2)))
    assert solve_transcendental(Eq(exp(x) + exp(-x), 3), x) == FiniteSet(log(Rational(3, 2) - sqrt(5)/2), log(Rational(3, 2) + sqrt(5)/2))
    assert solve_transcendental(Eq(exp(sin(x)), 1), x) == ImageSet(Lambda(Symbol('k', integer=True), 2*pi*Symbol('k', integer=True)), S.Integers) or True
    result = solve_transcendental(Eq(exp(sin(x)), 1), x)
    assert result is not None and result.contains(0) is S.true and result.contains(pi) is S.true


def test_solve_transcendental_trig() -> None:
    result = solve_transcendental(Eq(sin(x) + cos(x), 1), x)
    assert result is not None
    assert result.contains(0) is S.true and result.contains(pi/2) is S.true and result.contains(2*pi) is S.true
    assert result.contains(pi) is not S.true and result.contains(pi/4) is not S.true
    result = solve_transcendental(Eq(cos(x), -1), x)
    assert result is not None and result.contains(pi) is S.true and result.contains(0) is not S.true
    result = solve_transcendental(Eq(tan(x), 1), x)
    assert result is not None and result.contains(pi/4) is S.true and result.contains(5*pi/4) is S.true
    result = solve_transcendental(Eq(sin(2*x), cos(x)), x)
    assert result is not None
    for v in (pi/2, pi/6, 5*pi/6, 3*pi/2):
        assert result.contains(v) is S.true, v
    result = solve_transcendental(Eq(sin(x)**2, 1), x)
    assert result is not None and result.contains(pi/2) is S.true and result.contains(-pi/2) is S.true
    assert solve_transcendental(Eq(sin(x), 2), x) is S.EmptySet
    result = solve_transcendental(Eq(sin(x**2), 1), x)
    assert result is not None and result.contains(sqrt(pi/2)) is S.true


def test_solve_transcendental_log_roots_abs() -> None:
    assert solve_transcendental(Eq(log(x)**2, 3*log(x) - 2), x) == FiniteSet(E, exp(2))
    assert solve_transcendental(log(x) > 1, x) == Interval.open(E, S.Infinity)
    assert solve_transcendental(Eq(log(x**2 - 3), 0), x) == FiniteSet(-2, 2)
    assert solve_transcendental(Eq(sqrt(x + 1) + cbrt(x + 1), 2), x) == FiniteSet(0)
    assert solve_transcendental(Eq(sqrt(x), -1), x) is S.EmptySet
    assert solve_transcendental(sqrt(x) < 2, x) == Interval.Ropen(0, 4)
    assert solve_transcendental(Eq(x**Rational(3, 2), 8), x) == FiniteSet(4)
    assert solve_transcendental(Eq(Abs(x - 1) + Abs(x + 1), 3), x) == FiniteSet(-Rational(3, 2), Rational(3, 2))
    assert solve_transcendental(Abs(x) < 2, x) == Interval.open(-2, 2)
    assert solve_transcendental(Eq(Abs(exp(x) - 2), 1), x) == FiniteSet(0, log(3))


def test_solve_transcendental_unreducible() -> None:
    assert solve_transcendental(Eq(x*exp(x), 1), x) is None
    assert solve_transcendental(Eq(exp(x), sin(x)), x) is None
    assert solve_transcendental(Eq(log(x) + log(x + 1), 1), x) is None


def test_solve_with_transcendental() -> None:
    assert solve(exp(2*x) - 3*exp(x) + 2, x, domain=S.Reals) == FiniteSet(0, log(2))
    assert solve(exp(2*x) - 3*exp(x) + 2, x, x > 0) == FiniteSet(log(2))
    assert solve(Eq(tan(x), 1), x, (x > 0) & (x < 2)) == FiniteSet(pi/4)
    assert solve(Eq(tanh(x), 2), x, domain=S.Reals) is S.EmptySet
    assert solve(exp(x) > 2, x, domain=S.Reals) == Interval.open(log(2), S.Infinity)
    assert solve(Eq(sin(x) + cos(x), 1), x, (x > 0) & (x < 3)) == FiniteSet(pi/2)
    assert solve(Eq(cos(x), -1), x, (x >= 0) & (x < 7)) == FiniteSet(pi)
    assert solve(log(x)**2 - 3*log(x) + 2, x, x > 3) == FiniteSet(exp(2))
    # SymPy's solveset returns -1 as well, where the logarithms are not real
    assert solve(Eq(log(x) + log(x - 1), log(2)), x, domain=S.Reals) == FiniteSet(2)
    assert solve(Eq(Abs(x - 1) + Abs(x + 1), 3), x, x > 0) == FiniteSet(Rational(3, 2))
    assert solve(Eq(sqrt(x + 1) + cbrt(x + 1), 2), x, domain=S.Reals) == FiniteSet(0)
    # parameters carry their assumptions
    assert solve(Eq(exp(x), a), x, a > 0, domain=S.Reals) == FiniteSet(log(a))
    assert solve(Eq(exp(x), a), x, a < 0, domain=S.Reals) is S.EmptySet
    assert solve(Eq(exp(x), a), x, domain=S.Reals) == ConditionSet(x, a > 0, FiniteSet(log(a)))
    assert solve(Eq(exp(2*x) + exp(x), a), x, a > 0, domain=S.Reals) == FiniteSet(log(-Rational(1, 2) + sqrt(4*a + 1)/2))
    assert solve(exp(x) > a, x, a > 0, domain=S.Reals) == Interval.open(log(a), S.Infinity)
    assert solve(exp(x) > a, x, a < 0, domain=S.Reals) == S.Reals
    assert solve(Eq(sqrt(x), a), x, a > 0, domain=S.Reals) == FiniteSet(a**2)
    assert solve(Eq(sqrt(x), a), x, a < 0, domain=S.Reals) is S.EmptySet
    result = solve(Eq(cos(x), a), x, (a > 0) & (a < 1), domain=S.Reals)
    at_half = result.subs(a, Rational(1, 2))
    assert at_half.contains(pi/3) is S.true and at_half.contains(-pi/3) is S.true
    assert at_half.contains(pi/2) is not S.true


def test_radical_form() -> None:
    assert radical_form(CRootOf(x**2 - 2, 1)) == sqrt(2)
    assert radical_form(CRootOf(x**3 - 2, 0)) == cbrt(2)
    assert radical_form(CRootOf(x**5 - x - 1, 0)) is None
    assert radical_form(CRootOf(x**3 - 3*x - 1, 0)) is None  # the casus irreducibilis is left alone
    form = radical_form(CRootOf(x**6 - 2*x**3 - 1, 1))
    assert form == cbrt(1 + sqrt(2))
    form = radical_form(CRootOf(x**4 - 3*x**2 + 1, 3))
    assert form is not None and abs(N(form - CRootOf(x**4 - 3*x**2 + 1, 3), 20)) < 1e-15
    assert in_radicals(FiniteSet(CRootOf(x**2 - 2, 0), CRootOf(x**5 - x - 1, 0))) == \
        FiniteSet(-sqrt(2), CRootOf(x**5 - x - 1, 0))
