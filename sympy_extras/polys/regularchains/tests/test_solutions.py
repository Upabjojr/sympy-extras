"""Tests of the exact solutions of the chains without free variables."""
from __future__ import annotations

from sympy import CRootOf, Rational, sqrt, symbols
from sympy.polys.rootoftools import ComplexRootOf
from sympy.testing.pytest import raises

from sympy_extras._typing import as_expr
from sympy_extras.polys.regularchains import RegularChain, triangularize
from sympy_extras.polys.regularchains.solutions import exact_solutions

x, y, z = symbols('x y z')


def _matches_the_numerical_solutions(chain: RegularChain) -> bool:
    """Each exact solution is one of the numerical ones, each once."""
    numerical = chain.numerical_solutions(12)
    exact = chain.solutions()
    if len(exact) != len(numerical) or len(exact) != chain.degree:
        return False
    left = list(numerical)
    for solution in exact:
        values = {v: complex(as_expr(value).evalf(12)) for v, value in solution.items()}
        near = [point for point in left
            if all(abs(complex(point[v]) - values[v]) < 1e-8 * (1 + abs(values[v])) for v in values)]
        if len(near) != 1:
            return False
        left.remove(near[0])
    return True


def test_rational_points_and_radicals() -> None:
    chains = triangularize([x**2 + y + z - 1, x + y**2 + z - 1, x + y + z**2 - 1], x, y, z)
    points = [solution for chain in chains for solution in chain.solutions()]
    assert {(s[x], s[y], s[z]) for s in points} == {(0, 0, 1), (0, 1, 0), (1, 0, 0),
        (-1 + sqrt(2), -1 + sqrt(2), -1 + sqrt(2)), (-1 - sqrt(2), -1 - sqrt(2), -1 - sqrt(2))}
    assert exact_solutions([as_expr(y**2 - 2), as_expr(x * y - 1)], [y, x]) == [
        {y: -sqrt(2), x: -sqrt(2) / 2}, {y: sqrt(2), x: sqrt(2) / 2}]


def test_a_polynomial_of_degree_one_is_solved_by_a_division() -> None:
    [chain] = triangularize([x**2 + y**2 - 5, x * y - 1], x, y)
    # x = 1/y, written as a root object of its own polynomial: the bug was
    # the rational function of root objects it used to be, 1/CRootOf(...),
    # whose sort key FiniteSet evaluates numerically (solve spent 45 s
    # sorting the twelve points of three equations found in two)
    quartic = y**4 - 5 * y**2 + 1
    assert chain.solutions()[0] == {y: CRootOf(quartic, 0), x: CRootOf(quartic, 1)}
    assert all(abs(complex((s[y] * s[x]).evalf(12)) - 1) < 1e-10 for s in chain.solutions())
    assert len(chain.solutions(real=True)) == 4 and _matches_the_numerical_solutions(chain)


def test_roots_over_an_algebraic_point_are_identified() -> None:
    # x**5 - x - 1 = y over y = -sqrt(2) and y = sqrt(2): ten points, the
    # roots of a polynomial of degree ten shared between the two values of y;
    # x**5 - x = 1 - sqrt(2) has three real roots and x**5 - x = 1 + sqrt(2) one
    [chain] = triangularize([x**5 - x - 1 - y, y**2 - 2], x, y)
    points = chain.solutions()
    assert len(points) == 10 and sorted(str(s[y]) for s in points) == ['-sqrt(2)'] * 5 + ['sqrt(2)'] * 5
    assert len({s[x] for s in points}) == 10 and all(isinstance(s[x], ComplexRootOf) for s in points)
    real = chain.solutions(real=True)
    assert [s[y] for s in real] == [-sqrt(2)] * 3 + [sqrt(2)]
    for solution in real:
        assert abs(complex((solution[x]**5 - solution[x] - 1 - solution[y]).evalf(15))) < 1e-12


def test_the_symbol_of_a_root_object_is_not_the_variable() -> None:
    # the bug: the point was substituted in the polynomial before its
    # coefficients were taken, and Poly took the x written inside
    # CRootOf(x**3 - x - 1, 0), the value of y, for the variable x
    # (PolynomialError); the coefficients are substituted one by one
    [chain] = triangularize([x**3 + y**2 - 2, y**3 - y - 1], x, y)
    assert len(chain.solutions()) == 9 and len(chain.solutions(real=True)) == 1
    assert _matches_the_numerical_solutions(chain)


def test_random_systems_match_their_numerical_solutions() -> None:
    systems = [[x**2 * y - 3 * y + 1, x * y**2 + x - 2], [x**2 + y**2 - 1, x**2 - y**3 + y * x - 1],
        [x * y * z - 1, x + y + z, x * y + y * z + z * x], [2 * x**2 - y * z + 1, y**2 - z - 2, z**2 + z - 1]]
    for system in systems:
        variables = [x, y, z] if any(as_expr(e).has(z) for e in system) else [x, y]
        for chain in triangularize(system, *variables):
            if chain.dimension == 0:
                assert _matches_the_numerical_solutions(chain)


def test_free_variables_are_refused() -> None:
    [chain] = triangularize([x * y - 1], x, y)
    raises(ValueError, lambda: chain.solutions())
    assert RegularChain([2 * x - 1], x).solutions() == [{x: Rational(1, 2)}]


def test_refined_isolating_intervals_still_match() -> None:
    # the bug: evaluating a root object refines its isolating rectangle, to
    # 1e-95 here, less than the error of the numerical root which was looked
    # for inside it: the second computation of the same solutions failed
    [chain] = triangularize([x**5 - x - 1 - y, y**2 - 2], x, y)
    [first] = [solution for solution in chain.solutions() if solution[x] == CRootOf(
        x**10 - 2 * x**6 - 2 * x**5 + x**2 + 2 * x - 1, 4)]
    as_expr(first[x]).evalf(50)
    assert len(chain.solutions()) == 10 and len(chain.solutions(real=True)) == 4
