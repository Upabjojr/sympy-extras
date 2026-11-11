"""Exact global optimisation: classical problems with known optima, the
exact simplex method on random linear programs, the closed form for
quadratic functions, SymPy's critical points for univariate polynomials;
every point returned is substituted back."""
from __future__ import annotations

import random
from fractions import Fraction

import pytest
import sympy

from sympy_extras.core import And, Eq, Integer, Rational, Symbol, oo
from sympy_extras.core.expr import Expr
from sympy_extras.core.logic import Boolean, false, true
from sympy_extras.calculus.optimization import (Optimum, maximize, maximize_parametric, minimize, minimize_parametric,
                                                value_range)
from sympy_extras.interop import sympy_expr, to_sympy
from sympy_extras.sets.sets import EmptySet, Interval
from sympy_extras.solvers.simplex import linear_program

x, y, z = Symbol('x'), Symbol('y'), Symbol('z')


def _close(a: sympy.Expr, b: sympy.Expr) -> bool:
    return bool(abs(sympy.N(a - b, 40)) < sympy.Float('1e-30'))


def _check_points(f: Expr, constraints: Boolean, result: Optimum) -> None:
    """The points give the optimal value and satisfy the constraints."""
    assert result.attained and result.points
    target = sympy_expr(to_sympy(f))
    condition = to_sympy(constraints)
    for point in result.points:
        rule: dict[sympy.Basic | complex, sympy.Expr | complex] = {to_sympy(v): sympy_expr(to_sympy(c))
                                                                   for v, c in point.items()}
        assert _close(target.subs(rule), sympy_expr(to_sympy(result.value)))
        relations = [condition] if isinstance(condition, sympy.core.relational.Relational) else list(condition.args)
        for relation in relations:
            if not isinstance(relation, sympy.core.relational.Relational):
                continue
            left, right = sympy_expr(relation.lhs).subs(rule), sympy_expr(relation.rhs).subs(rule)
            # an equation, or an inequality which is true or active up to the precision of the evaluation
            if isinstance(relation, sympy.Eq):
                assert _close(left, right)
            else:
                assert relation.func(left, right) == sympy.true or _close(left, right)


def test_classical_problems() -> None:
    # the Rosenbrock function
    f = (1 - x)**2 + 100*(y - x**2)**2
    result = minimize(f)
    assert result.value == 0 and result.points == ({x: Integer(1), y: Integer(1)},)
    # the arithmetic and geometric means: x + y >= 2 on x y = 1, x > 0
    result = minimize(x + y, And(Eq(x*y, 1), x > 0))
    assert result.value == 2 and result.points == ({x: Integer(1), y: Integer(1)},)
    # the largest rectangle inscribed in the ellipse x**2/4 + y**2 = 1: area 4 x y = 4
    constraint = And(Eq(x**2/4 + y**2, 1), x >= 0, y >= 0)
    result = maximize(4*x*y, constraint)
    assert result.value == 4
    _check_points(4*x*y, constraint, result)
    # the distance from the origin to the hyperbola x y = 1 (squared): 2, at (1, 1) and (-1, -1)
    result = minimize(x**2 + y**2, Eq(x*y, 1))
    assert result.value == 2 and len(result.points) == 2
    _check_points(x**2 + y**2, Eq(x*y, 1), result)
    # the box of largest volume with x + y + z = 1 in the positive octant: 1/27 at (1/3, 1/3, 1/3)
    result = maximize(x*y*(1 - x - y), And(x >= 0, y >= 0, x + y <= 1))
    assert result.value == Rational(1, 27) and result.points == ({x: Rational(1, 3), y: Rational(1, 3)},)


def test_infima_which_are_not_attained() -> None:
    # Nie, Demmel, Sturmfels: (x y - 1)**2 + x**2 is positive with the infimum 0
    result = minimize((x*y - 1)**2 + x**2)
    assert result.value == 0 and not result.attained and result.points == ()
    assert maximize(-(x*y - 1)**2 - x**2).value == 0
    # open sets, asymptotic values, unbounded functions
    assert repr(minimize(x, x > 0)) == 'Optimum(0, not attained)'
    result = minimize(1/(1 + x**2))
    assert result.value == 0 and not result.attained
    assert maximize(1/(1 + x**2)).points == ({x: Integer(0)},)
    assert minimize(x**3 - x).value == -oo and maximize(x**2 + y**2).value == oo and not maximize(x**2).is_finite
    assert minimize(x*y).value == -oo
    assert minimize(x**2 + y**2, x*y > 1).value == 2 and not minimize(x**2 + y**2, x*y > 1).attained


def test_motzkin_polynomial() -> None:
    # nonnegative but not a sum of squares: the minimum 0 at the four points (+-1, +-1)
    motzkin = x**4*y**2 + x**2*y**4 - 3*x**2*y**2 + 1
    result = minimize(motzkin)
    assert result.value == 0 and result.attained
    assert sorted((p[x], p[y]) for p in result.points) == [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def test_algebraic_optima() -> None:
    # the minimum of x**4 - 3 x**2 + x is at the smallest root of 4 x**3 - 6 x + 1
    f = x**4 - 3*x**2 + x
    result = minimize(f)
    sx = sympy.Symbol('x')
    critical = sympy.real_roots(4*sx**3 - 6*sx + 1)
    best = min(critical, key=lambda c: sympy.N((sx**4 - 3*sx**2 + sx).subs(sx, c), 30))
    assert _close(sympy_expr(to_sympy(result.value)), (sx**4 - 3*sx**2 + sx).subs(sx, best))
    assert len(result.points) == 1 and _close(sympy_expr(to_sympy(result.points[0][x])), best)
    # x + y on the curve x**4 + y**4 = 1: the maximum 2**(3/4)
    result = maximize(x + y, Eq(x**4 + y**4, 1))
    assert _close(sympy_expr(to_sympy(result.value)), sympy.Integer(2)**sympy.Rational(3, 4))
    _check_points(x + y, Eq(x**4 + y**4, 1), result)


def test_univariate_polynomials_against_critical_points() -> None:
    rnd = random.Random(71)
    sx = sympy.Symbol('x')
    for _ in range(20):
        degree = rnd.choice([2, 4, 4, 6])
        coefficients = [rnd.randint(-4, 4) for _ in range(degree)] + [rnd.randint(1, 3)]
        f = sum((Integer(c)*x**k for k, c in enumerate(coefficients)), Integer(0))
        sf = sum(c*sx**k for k, c in enumerate(coefficients))
        result = minimize(f)
        expected = min((sf.subs(sx, c) for c in sympy.real_roots(sympy.diff(sf, sx))), key=lambda v: sympy.N(v, 30))
        assert result.attained and _close(sympy_expr(to_sympy(result.value)), expected)
        _check_points(f, true, result)
        assert maximize(f).value == oo
        # on an interval the ends compete with the critical points
        low, high = Integer(-1), Integer(2)
        bounded = maximize(f, And(x >= low, x <= high))
        candidates = [sf.subs(sx, -1), sf.subs(sx, 2)]
        candidates += [sf.subs(sx, c) for c in sympy.real_roots(sympy.diff(sf, sx)) if -1 <= sympy.N(c) <= 2]
        assert _close(sympy_expr(to_sympy(bounded.value)), max(candidates, key=lambda v: sympy.N(v, 30)))


def test_quadratic_functions() -> None:
    rnd = random.Random(72)
    done = 0
    while done < 15:
        a, b, c = rnd.randint(1, 4), rnd.randint(-3, 3), rnd.randint(1, 4)
        if a*c - b*b <= 0:
            continue
        done += 1
        p, q, r0 = rnd.randint(-3, 3), rnd.randint(-3, 3), rnd.randint(-3, 3)
        f = a*x**2 + 2*b*x*y + c*y**2 + p*x + q*y + r0
        # the minimum of v^T A v + w^T v + r is r - w^T A**-1 w / 4, at -A**-1 w / 2
        A = sympy.Matrix([[a, b], [b, c]])
        w = sympy.Matrix([p, q])
        expected = r0 - (w.T*A.inv()*w)[0, 0]/4
        point = -A.inv()*w/2
        result = minimize(f)
        assert sympy_expr(to_sympy(result.value)) == expected
        assert len(result.points) == 1
        assert [to_sympy(result.points[0][x]), to_sympy(result.points[0][y])] == [point[0], point[1]]


def test_linear_programs_against_the_simplex_method() -> None:
    rnd = random.Random(73)
    statuses: set[str] = set()
    for _ in range(25):
        rows = [[rnd.randint(-3, 3), rnd.randint(-3, 3)] for _ in range(rnd.randint(2, 4))]
        rhs = [rnd.randint(-2, 6) for _ in rows]
        cost = [rnd.randint(-3, 3), rnd.randint(-3, 3)]
        reference = linear_program(cost, rows, rhs)            # maximise c x subject to A x <= b
        statuses.add(reference.status)
        constraints = And(*[row[0]*x + row[1]*y <= b for row, b in zip(rows, rhs)])
        objective = cost[0]*x + cost[1]*y
        if reference.status == 'infeasible':
            with pytest.raises(ValueError):
                maximize(objective, constraints, [x, y])
            assert value_range(objective, constraints, [x, y]) == EmptySet
            continue
        result = maximize(objective, constraints, [x, y])
        if reference.status == 'unbounded':
            assert result.value == oo and not result.attained
            continue
        assert reference.value is not None
        value = result.value
        assert isinstance(value, Rational) and Fraction(value.p, value.q) == reference.value and result.attained
        _check_points(objective, constraints, result)
    assert statuses == {'optimal', 'infeasible', 'unbounded'}


def test_value_range_and_rejections() -> None:
    assert value_range(x**2) == Interval(0, oo, False, True)
    assert value_range(x/(1 + x**2)) == Interval(Rational(-1, 2), Rational(1, 2))
    assert value_range(x**2 + y**2, Eq(x, y) & (x**2 < 0)) == EmptySet
    # a constant, and a function which takes one value on the constraint set
    assert minimize(Integer(3), x > 0).value == 3 and maximize(x**2 + y**2, Eq(x**2 + y**2, 4)).value == 4
    with pytest.raises(ValueError):
        minimize(x, x**2 < 0)
    with pytest.raises(ValueError):
        minimize(x*z, variables=[x])                # a parameter
    # a pole inside the region is not a value
    result = minimize(1/x, x > 0)
    assert result.value == 0 and not result.attained and maximize(1/x**2).value == oo


def _case_value(cases: list[tuple[Boolean, Expr]], values: dict[Symbol, Expr]) -> list[Expr]:
    """The values of the cases whose condition holds at the parameters."""
    found: list[Expr] = []
    for condition, value in cases:
        held: Boolean = condition
        for symbol, number in values.items():
            substituted = held.subs(symbol, number)
            assert isinstance(substituted, Boolean)
            held = substituted
        assert held is true or held is false, held
        if held is true:
            current = value
            for symbol, number in values.items():
                current = current.subs(symbol, number)
            found.append(current)
    return found


def test_parametric_against_fixed_parameters() -> None:
    rnd = random.Random(74)
    a, b = Symbol('a'), Symbol('b')
    problems: list[tuple[Expr, Boolean, list[Symbol]]] = [
        (a*x**2 + x, true, [a]),
        (x**2 + a*x + b, true, [a, b]),
        (x**2 + y**2, Eq(x + y, a), [a]),
        (x, Eq(x**2, a), [a]),
        (a*x - x**2, And(x >= 0, x <= 1), [a]),
        (x*y, Eq(x**2 + y**2, a), [a]),
        (x**4 - a*x**2, true, [a]),
        (a*x + y, And(x**2 + y**2 <= 1), [a]),
    ]
    for f, constraints, parameters in problems:
        low = minimize_parametric(f, constraints, parameters=parameters)
        high = maximize_parametric(f, constraints, parameters=parameters)
        for _ in range(8):
            values: dict[Symbol, Expr] = {p: Rational(rnd.randint(-6, 6), 2) for p in parameters}
            fixed_f, fixed_c = f, constraints
            for symbol, number in values.items():
                fixed_f = fixed_f.subs(symbol, number)
                replaced = fixed_c.subs(symbol, number)
                assert isinstance(replaced, Boolean)
                fixed_c = replaced
            variables = [v for v in (x, y) if f.has(v) or constraints.has(v)]
            for cases, solver in ((low, minimize), (high, maximize)):
                found = _case_value(cases, values)
                try:
                    expected = solver(fixed_f, fixed_c, variables).value
                except ValueError:
                    assert found == []          # no feasible point: the parameters are in no case
                    continue
                assert len(found) == 1
                difference = sympy_expr(to_sympy(found[0])) - sympy_expr(to_sympy(expected))
                assert found[0] == expected or sympy.simplify(difference) == 0


def test_parametric_known_and_rejections() -> None:
    a = Symbol('a')
    assert str(minimize_parametric(a*x**2 + x, parameters=[a])) == '[(a > 0, -1/(4*a)), (a <= 0, -oo)]'
    # the distance from the origin to the line x + y = a, squared
    assert minimize_parametric(x**2 + y**2, Eq(x + y, a), parameters=[a]) == [(true, a**2/2)]
    assert maximize_parametric(x**2, parameters=[a]) == [(true, oo)]
    with pytest.raises(ValueError):
        minimize_parametric(x**2)
    with pytest.raises(ValueError):
        minimize_parametric(x*z + a, variables=[x], parameters=[a])
    with pytest.raises(NotImplementedError):
        minimize_parametric(a/(1 + x**2), parameters=[a])
