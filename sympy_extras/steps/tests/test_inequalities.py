"""Step-by-step inequalities: the solution set of every inequality node
of every tree is compared with the truth of that node's inequality,
evaluated directly at many rational points (the critical points among
them); the sign charts, the reversal of the inequality, alternatives."""
from __future__ import annotations

import random
from fractions import Fraction

import pytest

from sympy_extras.core import Integer, Rational, Symbol
from sympy_extras.core.expr import Expr
from sympy_extras.core.logic import Boolean, true
from sympy_extras.core.relational import Eq, Gt, Le
from sympy_extras._testing import untyped
from sympy_extras.functions import abs_, exp
from sympy_extras.sets.sets import EmptySet, Reals, Set
from sympy_extras.simplify.normalform import normal_form
from sympy_extras.steps.core import Step, format_steps
from sympy_extras.steps.inequalities import (AbsoluteValueRule, AddToBothSidesRule, DivideBothSidesRule, DontKnowRule,
                                             FalseRule, InequalityStep, SignChartRule, SolvedRule, TrueRule,
                                             inequality_steps, manual_inequality)

x, a = Symbol('x'), Symbol('a')

POINTS: list[Fraction] = sorted({Fraction(n, d) for d in (1, 2, 3, 7) for n in range(-8*d, 8*d + 1)})


def _truth(lhs: Expr, rhs: Expr, relation: str, point: Fraction) -> bool:
    """The inequality evaluated at a rational point (false where a side
    is not defined)."""
    value = Rational(point.numerator, point.denominator)
    left, right = lhs.subs(x, value), rhs.subs(x, value)
    if left.is_finite is not True or right.is_finite is not True:
        return False
    difference = normal_form(left - right)
    if relation == '<':
        return difference.is_negative is True
    if relation == '<=':
        return difference.is_negative is True or difference == 0
    if relation == '>':
        return difference.is_positive is True
    return difference.is_positive is True or difference == 0


def _contains(solutions: Set, point: Fraction) -> bool:
    return solutions.contains(Rational(point.numerator, point.denominator)) is true


def _check_tree(step: Step) -> None:
    if isinstance(step, InequalityStep):
        if not step.contains_dont_know():
            found = step.solutions()
            for point in POINTS:
                assert _contains(found, point) == _truth(step.lhs, step.rhs, step.relation, point), (step, point)
    for child in step.substeps():
        _check_tree(child)


INEQUALITIES: list[Boolean] = [
    3 - 2*x < 7, 2*x + 1 <= 3*x - 4, x/3 - 1 > Rational(1, 2), -x >= 4, x > x + 1, Le(x + 1, x + 1),
    x**2 - 5*x + 6 > 0, x**2 < 4, x**2 + 1 > 0, x**2 + 1 < 0, x**2 <= 0, x**2 - 2 >= 0, -x**2 + 3*x > 2,
    x**3 - x >= 0, (x - 1)**2*(x + 3) > 0, (x - 1)**2*(x + 3) <= 0, x**4 - 5*x**2 + 4 < 0, x**3 - 3*x + 1 < 0,
    (x - 1)/(x + 2) <= 0, 1/x < 2, (x + 1)/(x - 1) >= 2, (x**2 - 1)/(x - 1) > 0, x/(x**2 - 4) >= 0, 1/x**2 > 1,
    (x - 3)*(x + 1)/(x*(x - 2)) < 0, 1/(x - 1) <= 1/(x + 1),
    abs_(x - 1) < 3, abs_(2*x + 1) >= 3, abs_(x) <= 0, Gt(abs_(x - 2), -1), abs_(x - 2) > 1, abs_(x**2 - 4) < 3, abs_(x - 1) + x <= 5,
]


def test_every_node_against_direct_evaluation() -> None:
    for inequality in INEQUALITIES:
        steps = inequality_steps(inequality, x)
        assert not steps.contains_dont_know(), inequality
        _check_tree(steps)
        assert manual_inequality(inequality, x) == steps.solutions()


def test_random_polynomial_and_rational_inequalities() -> None:
    rnd = random.Random(201)
    for _ in range(25):
        numerator: Expr = Integer(rnd.choice([1, -1, 2]))
        for _ in range(rnd.randint(1, 3)):
            numerator = numerator*(x - Rational(rnd.randint(-6, 6), rnd.choice([1, 2])))**rnd.randint(1, 2)
        denominator: Expr = Integer(1)
        for _ in range(rnd.randint(0, 2)):
            denominator = denominator*(x - Rational(rnd.randint(-6, 6), rnd.choice([1, 3])))
        f = numerator/denominator
        inequality = rnd.choice([f < 0, f <= 0, f > 0, f >= 0])
        steps = inequality_steps(inequality, x)
        assert not steps.contains_dont_know(), inequality
        _check_tree(steps)


def test_rule_trees() -> None:
    linear = inequality_steps(3 - 2*x < 7, x)
    assert isinstance(linear, AddToBothSidesRule) and linear.term == -3
    divided = linear.substep
    # dividing by -2 reverses the inequality
    assert isinstance(divided, DivideBothSidesRule) and divided.divisor == -2 and divided.flipped
    assert divided.substep == SolvedRule(x, Integer(-2), '>', x, Integer(-2))
    positive = inequality_steps(2*x < 6, x)
    assert isinstance(positive, DivideBothSidesRule) and not positive.flipped
    chart = inequality_steps(x**2 - 5*x + 6 > 0, x)
    assert isinstance(chart, SignChartRule) and chart.points == (2, 3) and chart.signs == (1, -1, 1) and chart.poles == ()
    assert set(chart.numerator_factors) == {(x - 2, 1), (x - 3, 1)} and len(chart.root_steps) == 2
    # a double root does not change the sign; a pole is never a solution
    double = inequality_steps((x - 1)**2*(x + 3) > 0, x)
    assert isinstance(double, SignChartRule) and double.points == (-3, 1) and double.signs == (-1, 1, 1)
    rational = inequality_steps((x - 1)/(x + 2) <= 0, x)
    assert isinstance(rational, SignChartRule) and rational.poles == (-2,) and rational.points == (-2, 1)
    # a common factor is not cancelled: 1 is excluded from (x**2 - 1)/(x - 1) > 0
    hole = inequality_steps((x**2 - 1)/(x - 1) > 0, x)
    assert isinstance(hole, SignChartRule) and hole.poles == (1,) and hole.solutions().contains(Integer(1)) is not true
    assert isinstance(inequality_steps(x > x + 1, x), FalseRule)
    assert isinstance(inequality_steps(Le(x + 1, x + 1), x), TrueRule)        # the node, which Python's <= would evaluate
    assert inequality_steps(x**2 + 1 > 0, x).solutions() is Reals and inequality_steps(x**2 + 1 < 0, x).solutions() is EmptySet
    absolute = inequality_steps(abs_(x - 1) < 3, x)
    assert isinstance(absolute, AbsoluteValueRule) and absolute.inner == x - 1 and len(absolute.cases) == 4


def test_irrational_and_isolated_endpoints() -> None:
    assert str(manual_inequality(x**2 - 2 >= 0, x)) == 'Union(Interval(-oo, -sqrt(2)), Interval(sqrt(2), oo))'
    cubic = inequality_steps(x**3 - 3*x + 1 < 0, x)
    assert isinstance(cubic, SignChartRule) and len(cubic.points) == 3 and cubic.signs == (-1, 1, -1, 1)
    assert all(type(point).__name__ == 'IndexedRoot' for point in cubic.points)
    # points of different factors are ordered exactly: sqrt(2) < 3/2 < sqrt(3)
    mixed = inequality_steps((x**2 - 2)*(2*x - 3)*(x**2 - 3) > 0, x)
    assert isinstance(mixed, SignChartRule)
    assert [str(point) for point in mixed.points] == ['-sqrt(3)', '-sqrt(2)', 'sqrt(2)', '3/2', 'sqrt(3)']
    _check_tree(mixed)


def test_rendering_and_unsolved() -> None:
    assert format_steps(inequality_steps(3 - 2*x < 7, x)).splitlines() == [
        'AddToBothSidesRule: 3 - 2*x < 7 -> Interval.open(-2, oo)',
        '  DivideBothSidesRule: -2*x < 4 -> Interval.open(-2, oo)',
        '    SolvedRule: x > -2 -> Interval.open(-2, oo)']
    listing = format_steps(inequality_steps((x - 1)/(x + 2) <= 0, x)).splitlines()
    assert listing[0] == 'SignChartRule: (x - 1)/(x + 2) <= 0 -> Interval.Lopen(-2, 1)' and len(listing) == 5
    assert format_steps(inequality_steps((x - 1)/(x + 2) <= 0, x), depth=0) == listing[0]
    for inequality in (exp(x) > x, a*x**2 > 1):
        steps = inequality_steps(inequality, x)
        assert steps.contains_dont_know() and manual_inequality(inequality, x) is None
    assert isinstance(inequality_steps(exp(x) > 0, x), DontKnowRule)
    with pytest.raises(TypeError):
        untyped(inequality_steps)(Eq(x, 1), x)
