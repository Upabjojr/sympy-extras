"""Tests of the native isolated special values of the parameters of a
closed form (:mod:`sympy_extras._special_values`, on the objects of the
core); the values with a case are checked at their points."""
from __future__ import annotations

import time
from typing import Optional

from sympy_extras._special_values import (Point, condition_point, defined_problem, equality_points, point_assumptions,
                                          point_condition, special_points, undefined_at, with_special_values)
from sympy_extras._timeout import time_limit
from sympy_extras.assumptions.ask import Assumptions
from sympy_extras.core import Dummy, Eq, Integer, Rational, Symbol, pi, sqrt
from sympy_extras.core.expr import Expr
from sympy_extras.core.logic import true
from sympy_extras.core.numbers import nan
from sympy_extras.functions import DiracDelta, exp, log, sin, zeta
from sympy_extras.functions.elementary.piecewise import Piecewise, piecewise
from sympy_extras.settings import configure, settings

a, b, c, k, y, x = Symbol('a'), Symbol('b'), Symbol('c'), Symbol('k'), Symbol('y'), Symbol('x')
p = Symbol('p', positive=True)


def _two(point: Point, assumptions: Assumptions) -> Optional[Expr]:
    return Integer(2)


def _value_at(value: Expr, point: Point) -> Expr:
    """The branch of a ``Piecewise`` taken at the point, evaluated there."""
    assert isinstance(value, Piecewise)
    for pair in value.args:
        condition = pair.cond._xreplace_evaluated({s: v for s, v in point.items()})
        if condition == true:
            evaluated = pair.expr._xreplace_evaluated({s: v for s, v in point.items()})
            assert isinstance(evaluated, Expr)
            return evaluated
    raise AssertionError("no branch is taken at %s" % (point,))


def test_candidates_are_the_zeros_of_denominators_and_logarithms() -> None:
    assert special_points((exp(k) - 1)/k, [k]) == [{k: 0}]
    # (y = 0 from the argument of the logarithm, where 1/log(y) is 0)
    assert special_points(1/log(y), [y]) == [{y: 0}, {y: 1}]
    assert special_points(zeta(k)*k, [k]) == [{k: 1}]
    assert special_points(1/(a**2 - b**2), [a, b]) == [{a: -b}, {a: b}]
    assert special_points(1/(a*b - 1), [a, b]) == [{a: 1/b}]
    assert special_points(1/(c**2 - 2), [c]) == [{c: -sqrt(2)}, {c: sqrt(2)}]
    # a transcendental factor in one parameter: exp(k) = 1 at k = 0 only
    assert special_points(1/(exp(2*k) - 1), [k]) == [{k: 0}]


def test_candidates_not_described_by_isolated_values_are_left_out() -> None:
    assert special_points(1/(a**2 + 1), [a]) == []
    assert special_points(1/sin(a), [a]) == []
    assert special_points(1/(x + k), [k]) == []
    assert special_points(1/k, [k], k > 1) == []
    assert special_points(1/p, [p]) == []
    # the irrational roots of a cubic are not listed, its rational ones are
    assert special_points(1/(a**3 - 2), [a]) == []
    assert special_points(1/((a - 1)*(a**3 - 2)), [a]) == [{a: 1}]


def test_candidates_with_irrational_coefficients() -> None:
    # a polynomial whose coefficients the native domains do not take: its
    # roots by SymPy's (at run time), the same points
    assert special_points(1/(a**2 - pi), [a]) == [{a: -sqrt(pi)}, {a: sqrt(pi)}]
    assert special_points(1/(pi*a - 1), [a]) == [{a: 1/pi}]


def test_points_and_conditions() -> None:
    assert str(point_condition({a: Integer(0), b: Integer(0)})) == 'Eq(a, 0) & Eq(b, 0)'
    assert condition_point(Eq(a, b) & Eq(b, 0)) == {a: 0, b: 0}
    assert condition_point(a > 0) is None
    assert point_assumptions({a: b}, [a > 0]) == [b > 0]
    assert point_assumptions({a: Integer(-1)}, a > 0) is None
    # the relations of plain symbols do not evaluate: refuted by satisfiability
    assert point_assumptions({k: c}, [c > k]) is None
    assert equality_points(piecewise((Integer(2), Eq(y, 1)), (y, True)), [y]) == [{y: 1}]
    assert undefined_at(sin(a)/a, {a: Integer(0)})
    assert not undefined_at(sin(a)/a, {a: Integer(1)})
    assert undefined_at(log(a), {a: Integer(0)})


def test_cases_are_added_for_undefined_branches_only() -> None:
    generic = (y**2 - 1)/log(y)
    value = with_special_values(generic, [y], None, _two)
    assert value == piecewise((Integer(2), Eq(y, 1)), (generic, True))
    assert _value_at(value, {y: Integer(1)}) == 2
    assert _value_at(value, {y: Integer(3)}) == 8/log(Integer(3))
    pieces = piecewise((generic, y > 2), (y, True))
    assert with_special_values(pieces, [y], None, _two) == pieces
    assert with_special_values(generic, [y], None, _two, defined=lambda point: False) == generic
    assert with_special_values(generic, [y], None, lambda point, assumptions: None) == generic
    assert with_special_values(generic, [y], None, lambda point, assumptions: nan) == generic


def test_the_problem_at_the_point_gets_the_assumptions_there() -> None:
    seen: list[Assumptions] = []

    def at_point(point: Point, assumptions: Assumptions) -> Optional[Expr]:
        seen.append(assumptions)
        return Integer(1)

    value = with_special_values((exp(a*k) - 1)/k, [k], [a > 0, k < 1], at_point)
    assert value == piecewise((Integer(1), Eq(k, 0)), ((exp(a*k) - 1)/k, True))
    # k < 1 holds at k = 0 and is dropped
    assert seen == [[a > 0]]
    # a point the assumptions refute gets no case
    assert with_special_values((exp(k) - 1)/k, [k], k > 1, at_point) == (exp(k) - 1)/k


def test_nested_cases_are_flattened_the_most_specific_first() -> None:
    generic = 1/(a - b)

    def at_point(point: Point, assumptions: Assumptions) -> Optional[Expr]:
        return piecewise((Integer(0), Eq(b, 0)), (1/b, True))

    value = with_special_values(generic, [a, b], None, at_point)
    assert isinstance(value, Piecewise)
    assert (value.args[0].expr, value.args[0].cond) == (0, Eq(a, 0) & Eq(b, 0))
    assert (value.args[1].expr, value.args[1].cond) == (1/b, Eq(a, b))
    assert (value.args[2].expr, value.args[2].cond) == (generic, true)
    single = with_special_values(exp(k)/k, [k], None, lambda point, assumptions: Integer(1), sympy_style=True)
    assert str(single) == 'Piecewise((exp(k)/k, Ne(k, 0)), (1, True))'
    assert _value_at(single, {k: Integer(0)}) == 1
    assert _value_at(single, {k: Rational(1, 2)}) == 2*exp(Rational(1, 2))


def test_problems_undefined_at_the_point() -> None:
    z, u = Symbol('z'), Symbol('u')
    zero_power = (a**(b*z)/z).subs(a, 0)
    assert isinstance(zero_power, Expr) and not defined_problem(zero_power, [z])
    one_power = (a**(b*z)/z).subs(a, 1)
    assert isinstance(one_power, Expr) and defined_problem(one_power, [z])
    logarithm = log(a*z).subs(a, 0)
    assert isinstance(logarithm, Expr) and not defined_problem(logarithm, [z])
    delta = DiracDelta(a*u).subs(a, 0)
    assert isinstance(delta, Expr) and not defined_problem(delta, [u])
    assert defined_problem(DiracDelta(u), [u])


def test_unevaluated_values_at_the_points_can_be_refused() -> None:
    from sympy_extras.concrete.expr_with_limits import Sum
    generic = (y**2 - 1)/log(y)
    n = Dummy('n')

    def unevaluated(point: Point, assumptions: Assumptions) -> Optional[Expr]:
        return Sum(x**n, (n, 0, y))

    assert with_special_values(generic, [y], None, unevaluated, keep_unevaluated=False) == generic
    kept = with_special_values(generic, [y], None, unevaluated)
    assert isinstance(kept, Piecewise) and isinstance(kept.args[0].expr, Sum) and kept.args[0].cond == Eq(y, 1)


def test_the_special_values_have_a_budget_of_their_own() -> None:
    generic = (y**2 - 1)/log(y)

    def endless(point: Point, assumptions: Assumptions) -> Optional[Expr]:
        while True:
            pass

    with configure(timeout=10):
        started = time.monotonic()
        assert with_special_values(generic, [y], None, endless) == generic
        assert time.monotonic() - started < 5*settings.time_scale
    with time_limit(4, native=True):
        started = time.monotonic()
        assert with_special_values(generic, [y], None, endless, budget=100) == generic
        assert time.monotonic() - started < 3*settings.time_scale


def test_degenerate_points_are_candidates_of_their_own() -> None:
    C1, C2 = Symbol('C1'), Symbol('C2')
    generic = C1*exp(a*x) + C2*exp(b*x)
    repeated = (C1 + C2*x)*exp(b*x)

    def at_point(point: Point, assumptions: Assumptions) -> Optional[Expr]:
        return repeated

    def wronskian(branch: Expr) -> list[Expr]:
        return [(b - a)*exp((a + b)*x)] if branch.has(C1) else []

    assert with_special_values(generic, [a, b], None, at_point) == generic
    assert with_special_values(generic, [a, b], None, at_point, degenerate=wronskian) \
        == piecewise((repeated, Eq(a, b)), (generic, true))
    pieces = piecewise((b, Eq(a, b)), (generic, True))
    assert with_special_values(pieces, [a, b], None, at_point, degenerate=wronskian) == pieces


def test_zeros_under_nested_powers() -> None:
    assert special_points(log(sqrt((a - b)**2)), [a, b]) == [{a: b}]
