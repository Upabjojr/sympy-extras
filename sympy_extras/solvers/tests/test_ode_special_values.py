"""Tests of the cases of the isolated values of the parameters of ODEs
(:mod:`sympy_extras.solvers._special_values`): each case is checked by
substitution into the equation at its point, and the solution there must
be general (its Jacobian with respect to the constants not zero)."""
from __future__ import annotations

from sympy import Eq, Function, Integer, Or, Piecewise, Rational, exp, simplify, sin, symbols
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol

from sympy_extras._special_values import Point, condition_point
from sympy_extras._typing import as_boolean, as_expr, free_symbols, sorted_symbols
from sympy_extras.solvers import (dsolve_first_order, dsolve_kovacic, dsolve_linear, dsolve_second_order,
                                  riccati_ode, solve_ode)
from sympy_extras.solvers._special_values import solution_jacobian

x, k, a, b = symbols('x k a b')
C1, C2 = symbols('C1 C2')
y = Function('y')(x)


def _cases(value: Expr) -> list[tuple[Expr, Point]]:
    """The cases of a value, with their points (the generic one, last,
    with the empty point)."""
    if not isinstance(value, Piecewise):
        return [(value, {})]
    cases: list[tuple[Expr, Point]] = []
    for pair in value.args:
        condition = as_boolean(pair.args[1])
        disjuncts = condition.args if isinstance(condition, Or) else (condition,)
        for disjunct in disjuncts:
            point = condition_point(as_boolean(disjunct))
            assert point is not None, condition
            cases.append((as_expr(pair.args[0]), point))
    return cases


def _solves(equation: Expr, solution: Expr) -> bool:
    return simplify(equation.subs(y, solution).doit()) == 0


def _general(solution: Expr, constants: list[Symbol]) -> bool:
    return simplify(solution_jacobian(solution, constants, x)) != 0


def _check_solution(equation: Expr, solutions: list[Eq], points: list[Point]) -> None:
    """The only solution has the cases of the ``points`` (and no other),
    each a general solution of the equation at its point."""
    assert len(solutions) == 1 and solutions[0].lhs == y
    cases = _cases(as_expr(solutions[0].rhs))
    assert sorted((point for _, point in cases if point), key=str) == sorted(points, key=str)
    for value, point in cases:
        at = as_expr(equation.xreplace(point))
        constants = sorted_symbols(free_symbols(value) - free_symbols(at))
        assert _solves(at, value), (value, point)
        assert _general(value, constants), (value, point)


def _check_basis(equation: Expr, basis: list[Expr], points: list[Point]) -> None:
    """The ``i``-th functions of the cases of every point are a basis of the
    equation at the point."""
    found: list[Point] = []
    for function in basis:
        for _, point in _cases(function):
            if point and point not in found:
                found.append(point)
    assert sorted(found, key=str) == sorted(points, key=str)
    for point in found + [{}]:
        at = as_expr(equation.xreplace(point))
        functions = [_value_at(function, point) for function in basis]
        for function in functions:
            assert _solves(at, function), (function, point)
        combination = as_expr(sum((c*g for c, g in zip(symbols('D1:%d' % (len(basis) + 1)), functions)), Rational(0)))
        assert _general(combination, list(symbols('D1:%d' % (len(basis) + 1)))), (functions, point)


def _value_at(function: Expr, point: Point) -> Expr:
    """The value of a function with cases taken at a point (the generic one
    for the empty point)."""
    for value, case in _cases(function):
        if case == point or not case:
            return value
    raise AssertionError(point)


def test_resonance_of_a_first_order_equation() -> None:
    equation = as_expr(y.diff(x) - k*y - exp(x))
    # the bug: the only answer was C1*exp(k*x) - exp(x)/(k - 1), which
    # divides by zero at k = 1, where the solution is (C1 + x)*exp(x)
    _check_solution(equation, solve_ode(equation, y), [{k: Integer(1)}])
    assert solve_ode(equation, y, special_values=False) == [Eq(y, C1*exp(k*x) - exp(x)/(k - 1))]


def test_resonance_and_a_repeated_root_of_a_second_order_equation() -> None:
    equation = as_expr(y.diff(x, 2) + k**2*y - sin(x))
    # the bug: C1*exp(-I*k*x) + C2*exp(I*k*x) + sin(x)/(k**2 - 1) only,
    # undefined at k = 1 and k = -1 (resonance) and no longer general at
    # k = 0 (a double characteristic root: the two exponentials are 1),
    # where the solution is C1 + C2*x - sin(x)
    _check_solution(equation, solve_ode(equation, y), [{k: Integer(-1)}, {k: Integer(0)}, {k: Integer(1)}])


def test_a_repeated_characteristic_root_without_a_division_by_zero() -> None:
    equation = as_expr(y.diff(x, 2) - (a + b)*y.diff(x) + a*b*y)
    # the bug: C1*exp(a*x) + C2*exp(b*x) only, defined everywhere but a
    # one-parameter family at a = b (the Wronskian (b - a)*exp((a + b)*x)
    # vanishes), where the solution is (C1 + C2*x)*exp(b*x)
    _check_solution(equation, solve_ode(equation, y), [{a: b}])
    _check_basis(equation, dsolve_linear(equation, y), [{a: b}])
    # the Liouvillian solutions have the Wronskian's factor under a root,
    # sqrt(a**2 - 2*a*b + b**2)
    kovacic = dsolve_kovacic(equation, y)
    assert kovacic is not None
    _check_basis(equation, kovacic, [{a: b}])
    # (a division by zero at a = b in the solution of the linearisation)
    second = dsolve_second_order(equation, y)
    assert second is not None
    _check_solution(equation, [s for s in second if isinstance(s, Eq)], [{a: b}])


def test_euler_equation_with_a_repeated_indicial_root() -> None:
    equation = as_expr(x**2*y.diff(x, 2) + a*x*y.diff(x) + 4*y)
    # the bug: the solutions x**r for the roots of r**2 + (a - 1)*r + 4,
    # which coincide at a = -3 (r = 2) and a = 5 (r = -2), where the
    # second solution is x**r*log(x)
    _check_basis(equation, dsolve_linear(equation, y), [{a: Integer(-3)}, {a: Integer(5)}])
    general = as_expr(x**2*y.diff(x, 2) + a*x*y.diff(x) + b*y)
    _check_basis(general, dsolve_linear(general, y), [{b: a**2/4 - a/2 + Rational(1, 4)}])


def test_the_constant_of_a_riccati_solution_disappears() -> None:
    equation = as_expr(y.diff(x) + y**2 + k/x**2)
    # the bug: the solution through the Euler equation u'' + k*u/x**2 = 0
    # had no case for k = 1/4, where its two solutions coincide and the
    # constant C1 of -(u1' + C1*u2')/(u1 + C1*u2) disappears
    solution = riccati_ode(equation, y)
    assert isinstance(solution, Eq)
    _check_solution(equation, [solution], [{k: Rational(1, 4)}])
    first = dsolve_first_order(equation, y)
    assert isinstance(first, Eq)
    _check_solution(equation, [first], [{k: Rational(1, 4)}])


def test_no_case_where_the_order_drops() -> None:
    # C1 + C2*exp(-x/k) is undefined at k = 0, where the equation is y' = 0
    # of the first order: a singular perturbation, no specialisation of the
    # family, which gets no case
    equation = as_expr(k*y.diff(x, 2) + y.diff(x))
    solutions = solve_ode(equation, y)
    assert len(solutions) == 1 and not isinstance(solutions[0].rhs, Piecewise)


def test_problems_without_parameters_are_unchanged() -> None:
    solutions: list[Basic] = list(solve_ode(y.diff(x, 2) + y, y))
    assert len(solutions) == 1 and not solutions[0].has(Piecewise)
