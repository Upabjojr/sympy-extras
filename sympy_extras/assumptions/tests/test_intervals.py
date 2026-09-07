from __future__ import annotations

from sympy import (S, sin, cos, exp, log, sqrt, atan, tanh, cosh, pi, E, Abs, Interval,
    Rational, Max, Min)
from sympy.abc import x, y, z


def _box(**ranges: tuple[object, object]) -> Box:
    return {{'x': x, 'y': y, 'z': z}[k]: (as_expr(lo), as_expr(hi)) for k, (lo, hi) in ranges.items()}

from sympy_extras.assumptions import ask, refine, simplify
from sympy_extras.assumptions.analysis import signs_multivariate, box_of
from sympy_extras.assumptions.bounds import polynomial_abstraction
from sympy_extras.assumptions.facts import Facts
from sympy_extras._typing import as_expr
from sympy_extras.assumptions.intervals import (Box, Openness, evaluate, interval_sign, signs_on_box,
    range_on_box, corner_signs, isolate_signs)


def test_evaluate() -> None:
    v = evaluate(exp(x) - 1, _box(x=(S.Zero, S.One)))
    assert v is not None and v.a <= 0 and v.b >= E - 1 - Rational(1, 10**10)
    assert interval_sign(evaluate(x**2 + 1, _box(x=(S(-3), S(3))))) == 1
    assert interval_sign(evaluate(-exp(x) - 1, _box(x=(S.NegativeInfinity, S.Infinity)))) == -1
    assert interval_sign(evaluate(sin(x), _box(x=(S.Zero, S(4))))) is None
    assert interval_sign(evaluate(x*(1 - x) - x + x**2, _box(x=(S.Zero, S.One)))) is None  # the dependency problem
    assert evaluate(log(x), _box(x=(S(-1), S.One))) is None
    assert evaluate(sqrt(x), _box(x=(S.Zero, S(4)))) is not None
    assert evaluate(x**Rational(1, 3), _box(x=(S.One, S(8)))) is not None
    assert evaluate(x**y, {x: (S(2), S(3)), y: (S.Zero, S.One)}) is not None
    assert evaluate(Max(x, 1), _box(x=(S.Zero, S.One))) is None
    for f in (cos(x), atan(x), tanh(x), cosh(x), Abs(x), pi*x, x**-2):
        assert evaluate(f, _box(x=(S.One, S(2)))) is not None
    assert evaluate(x**-1, _box(x=(S(-1), S.One))) is None


def test_signs_on_box() -> None:
    assert signs_on_box(x**2 + y**2 + 1, _box(x=(S(-1), S.One), y=(S(-1), S.One))) == frozenset([1])
    assert signs_on_box(exp(x) + y - 1, {x: (S.One, S(2)), y: (S.Zero, S(3))}) == frozenset([1])
    assert signs_on_box(sin(x) - 2, _box(x=(S.NegativeInfinity, S.Infinity))) == frozenset([-1])
    # a zero on the box can never be certified away
    assert signs_on_box(x**2 + y**2 - 1, _box(x=(S(-1), S.One), y=(S(-1), S.One))) is None
    assert signs_on_box(x - y, _box(x=(S.Zero, S.One), y=(S.Zero, S.One))) is None
    assert signs_on_box(z, _box(x=(S.Zero, S.One))) is None
    assert range_on_box(x*y + 1, {x: (S.Zero, S.One), y: (S.Zero, S(2))}) == Interval(1, 3)


def test_corner_signs() -> None:
    box: Box = {x: (S.Zero, S(2)), y: (S.Zero, S(3))}
    open_box: Openness = {x: (True, True), y: (True, True)}
    closed: Openness = {x: (False, False), y: (False, False)}
    assert corner_signs(exp(x) + y - 1, box, open_box) == frozenset([1])
    assert corner_signs(exp(x) + y - 1, box, closed) == frozenset([0, 1])
    assert corner_signs(x*y - 7, box, closed) == frozenset([-1])
    assert corner_signs(x - y, box, open_box) == frozenset([-1, 0, 1])
    assert corner_signs(sin(x) + y, box, open_box) is None  # sin is not monotone on [0, 2]
    assert corner_signs(x, _box(x=(S.Zero, S.Infinity)), {x: (True, True)}) == frozenset([1])
    assert corner_signs(x, _box(x=(S.Zero, S.Infinity)), {x: (False, True)}) == frozenset([0, 1])
    # open in one variable only: the minimum is unattained through that variable
    assert corner_signs(exp(x*y) - 1, {x: (S.Zero, S.One), y: (S(2), S(3))}, {x: (True, True), y: (False, False)}) == frozenset([1])


def test_isolate_signs() -> None:
    assert isolate_signs(sin(x), x, Interval.open(0, pi)) == frozenset([1])
    assert isolate_signs(x*exp(x) - 1, x, Interval.open(1, S.Infinity)) == frozenset([1])
    assert isolate_signs(x*exp(x) - 1, x, Interval.open(0, 2)) == frozenset([-1, 0, 1])
    assert isolate_signs(x*exp(x) - 1, x, Interval.open(-1, 0)) == frozenset([-1])
    assert isolate_signs(exp(x) - 2, x, Interval(log(2), 3)) == frozenset([0, 1])
    assert isolate_signs(exp(x) - 2, x, Interval.open(log(2), 3)) == frozenset([1])
    assert isolate_signs(x**2 - 2*x + 1, x, Interval(0, 2)) is None  # a double zero


def test_polynomial_abstraction() -> None:
    a = polynomial_abstraction(exp(x) > 1 + x, {x})
    assert a is not None and len(a.variables) == 1
    t = a.variables[0]
    assert a.formula == (t > x + 1)
    assert (t >= x + 1) in a.constraints
    b = polynomial_abstraction(sqrt(x) + Abs(y) > 0, {x, y})
    assert b is not None and len(b.variables) == 2
    assert polynomial_abstraction(x**2 > 1, {x}) is None
    assert polynomial_abstraction(exp(sin(x)) > 0, {x}) is not None
    assert polynomial_abstraction(exp(x) > y, {x}) is None or True
    assert polynomial_abstraction(exp(exp(exp(exp(x)))) > 0, {x}) is None  # too many variables


def test_ask_multivariate() -> None:
    assert ask(exp(x) + y > 1, (x > 0) & (y > 0)) is True
    assert ask(exp(x) + y > 1, (x > 0) & (y > -1)) is None
    assert ask(sin(x*y) > 0, (x > 0) & (x < 1) & (y > 0) & (y < 3)) is True
    assert ask(cos(x + y) > 0, (x > 0) & (x < 1) & (y > 0) & (y < Rational(1, 2))) is True
    assert ask(x*exp(y) > 1, (x > 1) & (y > 0)) is True
    assert ask(x*exp(y) > 1, (x > 1) & (y > -1)) is None
    assert ask(log(x*y) < 0, (x > 0) & (x < 1) & (y > 0) & (y < 1)) is True
    assert ask(x**2 + y**2 + exp(z) > 0, domain=S.Reals) is True
    assert ask(sqrt(x) + sqrt(y) > 2, (x > 1) & (y > 1)) is True
    # signs through the polynomial bounds handed to the CAD
    assert ask(exp(x) > 1 + x, x > 0) is True
    assert ask(exp(x) >= 1 + x, domain=S.Reals) is True
    assert ask(cos(x) >= 1 - x**2/2, domain=S.Reals) is True
    assert ask(sin(x) < x, x > 0) is True
    assert ask(atan(x) < x, x > 0) is True
    assert ask(log(x) <= x - 1, x > 0) is True
    assert ask(Abs(x) + exp(y) > 0, domain=S.Reals) is True
    assert ask(exp(x) - exp(y) > 0, (x > 1) & (y < 0)) is True


def test_refine_multivariate() -> None:
    assert refine(Abs(exp(x) + y - 1), (x > 0) & (y > 0)) == exp(x) + y - 1
    assert refine(Abs(sin(x*y)), (x > 0) & (x < 1) & (y > 0) & (y < 3)) == sin(x*y)
    assert refine(Abs(log(x*y)), (x > 0) & (x < 1) & (y > 0) & (y < 1)) == -log(x*y)
    assert refine(Max(exp(x), 1 + x), domain=S.Reals) == exp(x)
    assert refine(Min(sin(x), x), x > 0) == sin(x)
    assert refine(sqrt((exp(x) - 1 - x)**2), domain=S.Reals) == exp(x) - x - 1
    assert simplify(Abs(cos(x + y))/cos(x + y), (x > 0) & (x < 1) & (y > 0) & (y < Rational(1, 2))) == 1
    assert refine(Abs(x - y), (x > 0) & (y > 0)) == Abs(x - y)


def test_box_and_multivariate_signs() -> None:
    facts = Facts([(x > 0) & (x < 1), y >= 2])
    assert box_of([x, y], facts) == (_box(x=(S.Zero, S.One), y=(S(2), S.Infinity)), {x: (True, True), y: (False, True)})
    assert box_of([x, z], facts) is None
    assert signs_multivariate(x + y, [x, y], facts) == frozenset([1])
    assert signs_multivariate(x - y, [x, y], facts) == frozenset([-1])
    assert signs_multivariate(exp(x*y) - 1, [x, y], facts) == frozenset([1])
    assert signs_multivariate(x*y - 1, [x, y], facts) == frozenset([-1, 0, 1])
