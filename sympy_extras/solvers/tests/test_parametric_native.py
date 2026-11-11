"""Tests of the native cases of polynomial systems with parameters (on
objects of the core): the solutions of each case satisfy the system, and
at integer values of the parameters the cases which hold give all the
solutions of the specialised system (counted by its triangular
decomposition)."""
from __future__ import annotations

import random

from sympy_extras.core import Integer, Symbol, Tuple, expand
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr
from sympy_extras.core.logic import BooleanFalse, BooleanTrue
from sympy_extras.polys.polytools import cancel
from sympy_extras.polys.regularchains import triangularize
from sympy_extras.sets.sets import PointSet
from sympy_extras.solvers.parametric import ParametricCase, parametric_cases

a, b, c, x, y = [Symbol(name) for name in 'abcxy']


def _at(e: Basic, values: dict[Basic, Basic]) -> Basic:
    return e._xreplace_evaluated(values)


def _check(equations: list[Expr], unknowns: list[Symbol], parameters: list[Symbol], tries: int = 12) -> None:
    cases = parametric_cases(equations, unknowns)
    generator = random.Random(1)
    for _ in range(tries):
        values: dict[Basic, Basic] = {p: Integer(generator.randint(-2, 2)) for p in parameters}
        points: set[Tuple] = set()
        for case in cases:
            holds = _at(case.condition, values)
            if isinstance(holds, BooleanFalse) or case.solutions is None:
                continue
            assert isinstance(case.solutions, PointSet)
            for point in case.solutions.points:
                specialised = _at(point, values)
                assert isinstance(specialised, Tuple)
                if any(s in specialised.free_symbols for s in unknowns):
                    points.add(Tuple(*[Symbol('free')]))
                    continue
                rule: dict[Basic, Basic] = dict(values)
                rule.update({s: v for s, v in zip(unknowns, specialised.args)})
                # an inequation in the unknowns is decided at the point
                at_point = _at(case.condition, rule)
                assert isinstance(at_point, (BooleanTrue, BooleanFalse)), at_point
                if isinstance(at_point, BooleanFalse):
                    continue
                for e in equations:
                    residual = _at(e, rule)
                    assert isinstance(residual, Expr) and expand(cancel(residual)) == 0
                points.add(specialised)
        system = [_at(e, values) for e in equations]
        chains = triangularize([e for e in system if isinstance(e, Expr)], *unknowns)
        if any(chain.dimension for chain in chains):
            continue
        # finitely many solutions: as many points as the decomposition has
        assert len(points) == sum(chain.degree for chain in chains), (values, points)


def test_the_cases_of_a_linear_and_of_a_quadratic_equation() -> None:
    linear = parametric_cases([a*x - b], [x])
    assert str(linear) == '[ParametricCase(Ne(a, 0), {(b/a,)}), ParametricCase(Eq(a, 0) & Eq(b, 0), {(x,)})]'
    quadratic = parametric_cases([a*x**2 + b*x + c], [x])
    assert [str(case.condition) for case in quadratic] == ['Ne(a, 0)', 'Eq(a, 0) & Ne(b, 0)',
                                                           'Eq(a, 0) & Eq(b, 0) & Eq(c, 0)']
    _check([a*x - b], [x], [a, b])
    _check([a*x**2 + b*x + c], [x], [a, b, c])


def test_systems() -> None:
    cases = parametric_cases([a*x + y - 1, x + a*y - 1], [x, y])
    assert str(cases) == '[ParametricCase(Eq(a, 1), {(1 - y, y)}), ParametricCase(Ne(a + 1, 0), {(1/(a + 1), 1/(a + 1))})]'
    _check([a*x + y - 1, x + a*y - 1], [x, y], [a])
    _check([x**2 - a, x*y - b], [x, y], [a, b])
    _check([a*x*y - 1, x - b*y], [x, y], [a, b])


def test_inequations_and_higher_degrees() -> None:
    [case] = parametric_cases([a*x - b], [x], inequations=[b])
    assert str(case.condition) == 'Ne(a, 0) & Ne(b, 0)' and str(case.solutions) == '{(b/a,)}'
    [cubic] = parametric_cases([x**3 - a], [x])
    assert cubic.solutions is None and cubic.equations == [x**3 - a]
    assert str(cubic) == 'ParametricCase(True, [Eq(-a + x**3, 0)])' and str(cubic.as_formula()) == 'Eq(-a + x**3, 0)'
    # the parameters given: the other symbols are unknowns' coefficients no more
    cases = parametric_cases([a*x - b], [x], parameters=[a, b])
    assert all(isinstance(case, ParametricCase) for case in cases) and len(cases) == 2
