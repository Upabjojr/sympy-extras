"""Tests of the isolated special values of the parameters of a closed form
(:mod:`sympy_extras._special_values`)."""
from __future__ import annotations

import time
from typing import Optional

from sympy import DiracDelta, Eq, Integer, Integral, Piecewise, Rational, exp, log, nan, sin, sqrt, symbols, zeta
from sympy.core.expr import Expr

from sympy_extras._special_values import (Point, condition_point, defined_problem, equality_points,
                                          point_assumptions, special_points, undefined_at, with_special_values)
from sympy_extras._timeout import time_limit
from sympy_extras.settings import configure, settings
from sympy_extras._typing import as_expr
from sympy_extras.assumptions.ask import Assumptions

a, b, c, k, y, x = symbols('a b c k y x')
p = symbols('p', positive=True)


def _two(point: Point, assumptions: Assumptions) -> Optional[Expr]:
    return Integer(2)


def test_candidates_are_the_zeros_of_denominators_and_logarithms() -> None:
    assert special_points(as_expr((exp(k) - 1) / k), [k]) == [{k: 0}]
    # (y = 0 from the argument of the logarithm, where 1/log(y) is 0)
    assert special_points(as_expr(1 / log(y)), [y]) == [{y: 0}, {y: 1}]
    assert special_points(as_expr(zeta(k) * k), [k]) == [{k: 1}]
    # a hyperplane in two parameters, solved for the first
    assert special_points(as_expr(1 / (a**2 - b**2)), [a, b]) == [{a: -b}, {a: b}]
    # a hypersurface a = 1/b
    assert special_points(as_expr(1 / (a * b - 1)), [a, b]) == [{a: 1 / b}]
    # the real roots of a quadratic, explicit
    assert special_points(as_expr(1 / (c**2 - 2)), [c]) == [{c: -sqrt(2)}, {c: sqrt(2)}]


def test_candidates_not_described_by_isolated_values_are_left_out() -> None:
    # no real zero, infinitely many zeros, a variable which is not a parameter
    assert special_points(as_expr(1 / (a**2 + 1)), [a]) == []
    assert special_points(as_expr(1 / sin(a)), [a]) == []
    assert special_points(as_expr(1 / (x + k)), [k]) == []
    # refuted by the assumptions and by the flags of the symbol
    assert special_points(as_expr(1 / k), [k], k > 1) == []
    assert special_points(as_expr(1 / p), [p]) == []


def test_points_and_conditions() -> None:
    assert condition_point(Eq(a, b) & Eq(b, 0)) == {a: 0, b: 0}
    assert condition_point(a > 0) is None
    assert point_assumptions({a: b}, [a > 0]) == [b > 0]
    assert point_assumptions({a: Integer(-1)}, a > 0) is None
    assert equality_points(Piecewise((2, Eq(y, 1)), (y, True)), [y]) == [{y: 1}]
    assert undefined_at(as_expr(sin(a) / a), {a: Integer(0)})
    assert not undefined_at(as_expr(sin(a) / a), {a: Integer(1)})


def test_cases_are_added_for_undefined_branches_only() -> None:
    generic = as_expr((y**2 - 1) / log(y))
    assert with_special_values(generic, [y], None, _two) == Piecewise((2, Eq(y, 1)), (generic, True))
    # a branch whose condition is false at the point does not take it there
    piecewise = as_expr(Piecewise((generic, y > 2), (y, True)))
    assert with_special_values(piecewise, [y], None, _two) == piecewise
    # the problem undefined at the point, or no value there: no case
    assert with_special_values(generic, [y], None, _two, defined=lambda point: False) == generic
    assert with_special_values(generic, [y], None, lambda point, assumptions: None) == generic
    assert with_special_values(generic, [y], None, lambda point, assumptions: as_expr(nan)) == generic


def test_nested_cases_are_flattened_the_most_specific_first() -> None:
    generic = as_expr(1 / (a - b))

    def at_point(point: Point, assumptions: Assumptions) -> Optional[Expr]:
        # at a = b, a value with its own special value b = 0
        return as_expr(Piecewise((0, Eq(b, 0)), (1 / b, True)))

    value = with_special_values(generic, [a, b], None, at_point)
    assert isinstance(value, Piecewise)
    assert value.args[0].args == (0, Eq(a, 0) & Eq(b, 0))
    assert value.args[1].args == (1 / b, Eq(a, b))
    assert value.args[2].args == (generic, True)
    # SymPy's form for a single case
    single = with_special_values(as_expr(exp(k) / k), [k], None, lambda point, assumptions: Integer(1),
                                 sympy_style=True)
    assert isinstance(single, Piecewise) and single.args[1].args == (1, True)
    assert single.subs(k, 0) == 1 and single.subs(k, Rational(1, 2)) == 2 * exp(Rational(1, 2))


def test_problems_undefined_at_the_point() -> None:
    z, u = symbols('z u')
    # 0**(b*z) is 0 or zoo according to the sign of b*z: no case at a = 0
    # for a**(b*z)/z (the bug: an unevaluated Integral(0**(b*z)/z, z) case)
    assert not defined_problem(as_expr((a**(b * z) / z).subs(a, 0)), [z])
    assert defined_problem(as_expr((a**(b * z) / z).subs(a, 1)), [z])
    assert not defined_problem(as_expr(log(a * z).subs(a, 0)), [z])
    # the delta of a constant is no function (the bug: a case
    # Integral(exp(-u*v)*DiracDelta(0), (u, 0, oo)) at a = 0)
    assert not defined_problem(as_expr(DiracDelta(a * u).subs(a, 0)), [u])
    assert defined_problem(as_expr(DiracDelta(a * u).subs(a, 1)), [u])


def test_unevaluated_values_at_the_points_can_be_refused() -> None:
    generic = as_expr((y**2 - 1) / log(y))

    def unevaluated(point: Point, assumptions: Assumptions) -> Optional[Expr]:
        return as_expr(Integral(x**y, x))

    assert with_special_values(generic, [y], None, unevaluated, keep_unevaluated=False) == generic
    assert with_special_values(generic, [y], None, unevaluated) == Piecewise((Integral(x**y, x), Eq(y, 1)),
                                                                             (generic, True))


def test_the_special_values_have_a_budget_of_their_own() -> None:
    generic = as_expr((y**2 - 1) / log(y))

    def endless(point: Point, assumptions: Assumptions) -> Optional[Expr]:
        while True:
            pass

    # the bug: the problems at the points had the whole time limit of the
    # settings (thirty seconds), more than the time limit of the census
    # (twenty) under which the integral had been solved
    with configure(timeout=10):
        started = time.monotonic()
        assert with_special_values(generic, [y], None, endless) == generic
        assert time.monotonic() - started < 5 * settings.time_scale
    # and at most half of what is left of an enclosing limit
    with time_limit(4):
        started = time.monotonic()
        assert with_special_values(generic, [y], None, endless, budget=100) == generic
        assert time.monotonic() - started < 3 * settings.time_scale
