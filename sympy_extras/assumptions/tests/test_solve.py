from __future__ import annotations

from sympy import (S, Eq, sqrt, exp, log, sin, cos, acos, pi, Interval, Union, FiniteSet, Range,
    Rational, ConditionSet, ImageSet, Q, CRootOf)
from sympy.abc import x, y, a, n
from sympy.testing.pytest import raises

from sympy_extras.assumptions import solve, element
from sympy_extras._testing import untyped


def test_solve_univariate_polynomial() -> None:
    assert solve(x**2 - 2, x) == FiniteSet(-sqrt(2), sqrt(2))
    assert solve(x**2 - 2, x, x > 0) == FiniteSet(sqrt(2))
    assert solve(x**2 - 2, x, x < 0) == FiniteSet(-sqrt(2))
    assert solve(Eq(x**2, 2), x, x > 2) == S.EmptySet
    assert solve(x**2 + 1, x, domain=S.Reals) == S.EmptySet
    assert solve(x**2 + 1, x) == FiniteSet(-S.ImaginaryUnit, S.ImaginaryUnit)
    assert solve((x**2 - 2 > 0) & (x < 3), x, domain=S.Reals) == \
        Union(Interval.open(-S.Infinity, -sqrt(2)), Interval.open(sqrt(2), 3))
    assert solve(x**2 - 2 > 0, x, x > 0) == Interval.open(sqrt(2), S.Infinity)
    assert solve(x**3 - x - 1, x, domain=S.Reals) == FiniteSet(CRootOf(x**3 - x - 1, 0))
    assert solve(x**2 < 10, x, domain=S.Integers) == Range(-3, 4)
    assert solve(x**2 < 10, x, domain=S.Naturals) == Range(1, 4)
    assert solve(x**2 - 4, x, element(x, S.Naturals)) == FiniteSet(2)
    assert solve((x - 1)*(x - 2)*(x - 3), x, (x > 1) & (x < 3)) == FiniteSet(2)
    # contradictory assumptions
    assert solve(x**2 - 2, x, (x > 0) & (x < 0)) == S.EmptySet


def test_solve_parameters() -> None:
    assert solve(x**2 - a, x, (x > 0) & (a > 0)) == FiniteSet(sqrt(a))
    assert solve(x**2 - a, x, x > 0) == ConditionSet(x, x > 0, FiniteSet(-sqrt(a), sqrt(a)))
    assert solve(x**2 - a, x) == FiniteSet(-sqrt(a), sqrt(a))
    assert solve(a*x - 1, x, a > 0) == FiniteSet(1/a)
    assert solve(x**2 - a, x, (x > 0) & (a < 0)) == S.EmptySet


def test_solve_transcendental() -> None:
    assert solve(exp(x) - 2, x, x > 1) == S.EmptySet
    assert solve(exp(x) - 2, x, x > 0) == FiniteSet(log(2))
    result = solve(sin(x), x, (x > 0) & (x < 4))
    assert result == FiniteSet(pi) or pi in result
    assert solve(exp(x) > 2, x, domain=S.Reals) == Interval.open(log(2), S.Infinity)
    result = solve(sin(x), x, domain=S.Reals)
    assert isinstance(result, Union) and all(isinstance(s, ImageSet) for s in result.args)


def test_solve_systems() -> None:
    assert solve([x**2 + y**2 - 1, x - y], [x, y], x > 0) == FiniteSet((sqrt(2)/2, sqrt(2)/2))
    assert solve([x**2 + y**2 - 1, x - y], [x, y]) == \
        FiniteSet((-sqrt(2)/2, -sqrt(2)/2), (sqrt(2)/2, sqrt(2)/2))
    assert solve([x**2 + y**2 - 1, x - y], [x, y], x < 0) == FiniteSet((-sqrt(2)/2, -sqrt(2)/2))
    assert solve([x**2 + y**2 + 1, x - y], [x, y], domain=S.Reals) == S.EmptySet
    assert solve([x + y - 3, x - y - 1], [x, y], element(x, S.Integers)) == FiniteSet((2, 1))
    raises(NotImplementedError, lambda: solve([x > y, y > 0], [x, y]))


def test_solve_errors() -> None:
    raises(ValueError, lambda: solve(S(1), []))
    raises(TypeError, lambda: untyped(solve)(S.Reals, x))


def test_solve_predicates_and_integers() -> None:
    assert solve(n**2 - 4, n, Q.prime(n)) == FiniteSet(2)
    assert solve(n**2 - 9, n, element(n, S.Naturals0)) == FiniteSet(3)
    assert solve(2*x - 1, x, element(x, S.Integers)) == S.EmptySet
    assert solve(2*x - 1, x, element(x, S.Rationals)) == FiniteSet(S.Half)


def test_solve_drops_extraneous_roots() -> None:
    # sympy's solveset returns extraneous roots of radical equations
    f = 2*x**2 + 3*sqrt(x + 6) - 1
    assert solve(f, x, (x > 2) & (x < Rational(7, 2))) == S.EmptySet
    assert solve(f, x, domain=S.Reals) == S.EmptySet
    assert solve(sqrt(x + 2) - x, x, domain=S.Reals) == FiniteSet(2)


def test_solve_periodic_inequality() -> None:
    # sympy's solveset answers a periodic inequality over one period only:
    # solveset(sin(x) > 0, x, S.Reals) is Interval.open(0, pi), which claims
    # that no negative number and nothing beyond pi is a solution
    solutions = solve(sin(x) > 0, x, domain=S.Reals)
    assert solutions.contains(-6) is S.true
    assert solutions.contains(7) is S.true
    assert solutions.contains(4) is S.false
    assert solve(sin(x) > 0, x, (x > 0) & (x < 10)) == \
        Union(Interval.open(0, pi), Interval.open(2*pi, 3*pi))
    # a region of less than one period which is not the one solveset uses
    assert solve(sin(x) > 0, x, (x > 5) & (x < 7)) == Interval.open(2*pi, 7)
    assert solve(cos(x) < Rational(1, 3), x, (x > 0) & (x < 2)) == \
        Interval.open(acos(Rational(1, 3)), 2)
    # the same truncation over the integers gave Range(1, 4, 1)
    integers = solve(sin(x) > 0, x, element(x, S.Integers))
    assert [k for k in range(-8, 16) if integers.contains(k) is S.true] == \
        [-6, -5, -4, 1, 2, 3, 7, 8, 9, 13, 14, 15]
    # equations keep the periodic family of image sets solveset returns
    roots = solve(Eq(sin(x), 0), x, domain=S.Reals)
    assert isinstance(roots, Union) and all(isinstance(part, ImageSet) for part in roots.args)
    assert roots.contains(-2*pi) is S.true and roots.contains(pi) is S.true
