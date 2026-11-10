"""The native solve on objects of the core, checked by substitution and
against hand computations (the SymPy face is tested in test_solve.py and
the solvers' tests)."""
from __future__ import annotations

from sympy_extras.core import Basic, Eq, Expr, I, Integer, Rational, Symbol, Tuple, as_expr, sqrt
from sympy_extras.core.logic import true
from sympy_extras.functions import exp, log, sin
from sympy_extras.sets import (ConditionSet, Complexes, EmptySet, FiniteSet, ImageSet, Integers, PointSet,
    Reals, Union, finite_set, interval)
from sympy_extras.assumptions.facts import element
from sympy_extras.assumptions.solve import _refuted, _same_number, solve
from sympy_extras.sets.sets import point_set

x, y, a, b = Symbol('x'), Symbol('y'), Symbol('a'), Symbol('b')


def _vanishes(f: Expr, values: dict[Basic, Basic]) -> bool:
    e = as_expr(f.xreplace(values)).doit()
    return abs(complex(e.evalf(30))) < 1e-20


def test_rational_equations_over_the_complex_numbers() -> None:
    assert solve(x**2 - 2, x) == finite_set(-sqrt(2), sqrt(2))
    assert solve(x**2 + 1, x) == finite_set(-I, I)
    assert solve(x**2 - a, x) == finite_set(-sqrt(a), sqrt(a))
    assert solve(a*x - 1, x, a > 0) == finite_set(1/a)
    assert solve(1/x - 2, x, x > 0) == finite_set(Rational(1, 2))
    # the assumptions choose among the solutions
    assert solve(x**2 - a**2, x, (a > 0) & (x < 0)) == finite_set(-a)
    found = solve(x**2 - a, x, x > 0)
    assert isinstance(found, ConditionSet) and found.base_set == finite_set(-sqrt(a), sqrt(a))
    assert solve(x**2 - 2, x, (x > 0) & (x < 0)) == EmptySet


def test_real_unknowns_and_domains() -> None:
    assert solve(x**2 - 2, x, x > 0) == finite_set(sqrt(2))
    assert solve(x**2 + 1, x, domain=Reals) == EmptySet
    found = solve((x**2 - 2 > 0) & (x < 3), x, domain=Reals)
    assert isinstance(found, Union) and len(found.args) == 2
    assert solve(x**2 - 4, x, element(x, Integers) & (x > 0)) == finite_set(2)
    assert solve(2*x - 1, x, element(x, Integers)) == EmptySet
    assert solve(exp(x) - 2, x, x > 1) == EmptySet
    assert solve(exp(x) - 2, x, x > 0) == finite_set(log(2))
    found = solve(sin(x) > 0, x, (x > 0) & (x < 10))
    assert isinstance(found, Union) and all(isinstance(piece, type(interval(0, 1))) for piece in found.args)


def test_systems() -> None:
    found = solve([x**2 + y**2 - 1, x - y], [x, y], x > 0)
    assert isinstance(found, PointSet) and found == point_set(Tuple(sqrt(2)/2, sqrt(2)/2))
    found = solve([x**2 + y**2 - 5, x*y - 2], [x, y])
    assert isinstance(found, PointSet) and len(found) == 4
    for point in found.points:
        u, v = point.args
        assert _vanishes(x**2 + y**2 - 5, {x: u, y: v}) and _vanishes(x*y - 2, {x: u, y: v})
    found = solve([x**2 + y**2 <= 1, x + y >= 1], [x, y], domain=Reals)
    assert isinstance(found, ConditionSet) and found.symbols == (x, y)
    assert solve(Eq(3*x + 5*y, 22), [x, y], (x >= 0) & (y >= 0), domain=Integers) == point_set(Tuple(4, 2))
    image = solve(Eq(3*x + 5*y, 22), [x, y], domain=Integers)
    assert isinstance(image, ImageSet) and len(image.variables) == 1
    t = image.variables[0]
    values = image.image
    assert isinstance(values, Tuple)
    for k in range(-3, 4):
        u, v = [as_expr(c.subs(t, Integer(k))) for c in values.args]
        assert 3*u + 5*v == 22


def test_cases_of_the_parameters() -> None:
    assert str(solve(a*x - 1, x, cases=True)) == 'ConditionSet(x, Ne(a, 0), {1/a})'
    found = solve(a*x - b, x, cases=True)
    assert isinstance(found, Union) and len(found.args) == 2
    assert solve(a*x - b, x, a > 0, cases=True) == finite_set(b/a)
    assert solve(Eq(x**2, a), x, a < 0, domain=Reals, cases=True) == EmptySet
    assert solve(x**2 - 2, x, cases=True) == finite_set(-sqrt(2), sqrt(2))
    assert solve(a*x - b, x, Eq(a, 0), cases=True) == ConditionSet(x, Eq(a, 0) & Eq(b, 0), Complexes)


def test_helpers() -> None:
    # two radical forms of one number are one solution (sympy-extras#50)
    assert _same_number(Rational(15, 26) + 3*sqrt(3)/13, sqrt(45*sqrt(3)/169 + Rational(333, 676))) is True
    assert _same_number(I, -I) is False
    equations = [x**5 - x - 1 - y, y**2 - 2]
    assert _refuted(PointSet(Tuple(x, sqrt(2))), equations, [x, y]) is True
    assert _refuted(PointSet(Tuple(1/y, y)), [x*y - 1], [x, y]) is False


def test_errors() -> None:
    from sympy_extras._testing import untyped
    try:
        solve(Integer(1), [])
    except ValueError:
        pass
    else:
        raise AssertionError('no unknown')
    try:
        untyped(solve)(Reals, x)
    except TypeError:
        pass
    else:
        raise AssertionError('a set is not an equation')
    assert isinstance(solve(x - 1, x, true), FiniteSet)
