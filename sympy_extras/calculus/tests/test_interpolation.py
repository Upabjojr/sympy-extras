"""Interpolation and fitting: SymPy's interpolating polynomial, the
defining conditions of Hermite interpolation and of the cubic splines
(values, continuity of two derivatives, end conditions), orthogonality
of the residual of a least squares fit."""
from __future__ import annotations

import random

import pytest
import sympy

from sympy_extras.core import Integer, Rational, Symbol
from sympy_extras.core.expr import Expr
from sympy_extras.core.logic import true
from sympy_extras.calculus.interpolation import (cubic_spline, divided_differences, hermite_interpolation,
                                                 interpolating_polynomial, least_squares, polynomial_fit)
from sympy_extras.functions import Piecewise, cos, sin
from sympy_extras.interop import sympy_expr, to_sympy
from sympy_extras.matrices.linalg import normal

x, a, b = Symbol('x'), Symbol('a'), Symbol('b')
sx = sympy.Symbol('x')


def _points(rnd: random.Random, n: int) -> list[tuple[Expr, Expr]]:
    nodes = sorted(rnd.sample(range(-6, 7), n))
    return [(Integer(p), Rational(rnd.randint(-8, 8), rnd.randint(1, 3))) for p in nodes]


def test_interpolating_polynomial_against_sympy() -> None:
    rnd = random.Random(151)
    for _ in range(20):
        points = _points(rnd, rnd.randint(1, 6))
        ours = sympy_expr(to_sympy(interpolating_polynomial(points, x)))
        expected = sympy.interpolate([(to_sympy(p), to_sympy(v)) for p, v in points], sx)
        assert sympy.expand(ours - expected) == 0
    # symbolic nodes and values: the line through (a, 0) and (b, 1)
    line = interpolating_polynomial([(a, 0), (b, 1)], x)
    assert normal(line.subs(x, a)) == 0 and normal(line.subs(x, b) - 1) == 0
    assert divided_differences([(0, 0), (1, 1), (2, 8), (3, 27)]) == [0, 1, 3, 1]
    with pytest.raises(ValueError):
        interpolating_polynomial([(1, 2), (1, 3)], x)


def test_hermite_interpolation_conditions() -> None:
    rnd = random.Random(152)
    for _ in range(15):
        nodes = sorted(rnd.sample(range(-3, 4), rnd.randint(1, 3)))
        data = [(Integer(p), [Integer(rnd.randint(-3, 3)) for _ in range(rnd.randint(1, 3))]) for p in nodes]
        polynomial = hermite_interpolation(data, x)
        total = sum(len(values) for _, values in data)
        assert sympy.degree(to_sympy(polynomial), sx) < total or polynomial == 0
        for node, values in data:
            derivative = polynomial
            for value in values:
                assert derivative.subs(x, node) == value
                derivative = derivative.diff(x)
    # the cubic Hermite basis on [0, 1], and Taylor's polynomial of sin at 0
    assert hermite_interpolation([(0, [1, 0]), (1, [0, 0])], x) == 2*x**3 - 3*x**2 + 1
    assert hermite_interpolation([(0, [0, 1, 0, -1])], x) == x - x**3/6
    with pytest.raises(ValueError):
        hermite_interpolation([(0, [1]), (0, [2])], x)
    with pytest.raises(ValueError):
        hermite_interpolation([(0, [])], x)


def _pieces(spline: Expr, nodes: list[Expr]) -> list[Expr]:
    """The cubic on each interval between consecutive nodes."""
    if not isinstance(spline, Piecewise):
        return [spline]*(len(nodes) - 1)
    return [_branch(spline, (nodes[i] + nodes[i + 1])/2) for i in range(len(nodes) - 1)]


def _branch(spline: Piecewise, point: Expr) -> Expr:
    for pair in spline.args:
        expression, condition = pair.args[0], pair.args[1]
        if condition.subs(x, point) is true:
            assert isinstance(expression, Expr)
            return expression
    raise AssertionError('no branch at %s' % (point,))


def _check_spline(points: list[tuple[Expr, Expr]], spline: Expr) -> list[Expr]:
    nodes = [p for p, _ in points]
    pieces = _pieces(spline, nodes)
    for i, piece in enumerate(pieces):
        assert sympy.degree(to_sympy(piece), sx) <= 3 or piece == 0
        assert piece.subs(x, nodes[i]) == points[i][1] and piece.subs(x, nodes[i + 1]) == points[i + 1][1]
    for i in range(len(pieces) - 1):
        for order in (1, 2):
            left, right = pieces[i].diff(x, order), pieces[i + 1].diff(x, order)
            assert left.subs(x, nodes[i + 1]) == right.subs(x, nodes[i + 1])
    return pieces


def test_cubic_splines() -> None:
    rnd = random.Random(153)
    for _ in range(12):
        points = _points(rnd, rnd.randint(2, 6))
        nodes = [p for p, _ in points]
        natural = _check_spline(points, cubic_spline(points, x))
        assert natural[0].diff(x, 2).subs(x, nodes[0]) == 0 and natural[-1].diff(x, 2).subs(x, nodes[-1]) == 0
        s0, s1 = Integer(rnd.randint(-2, 2)), Integer(rnd.randint(-2, 2))
        clamped = _check_spline(points, cubic_spline(points, x, 'clamped', (s0, s1)))
        assert clamped[0].diff(x).subs(x, nodes[0]) == s0 and clamped[-1].diff(x).subs(x, nodes[-1]) == s1
        knot = _check_spline(points, cubic_spline(points, x, 'not-a-knot'))
        if len(points) >= 4:
            assert knot[0].diff(x, 3) == knot[1].diff(x, 3) and knot[-1].diff(x, 3) == knot[-2].diff(x, 3)
        # with one or two intervals the end condition has two entries at one position: they were overwritten
        closed = points[:-1] + [(points[-1][0], points[0][1])]
        periodic = _check_spline(closed, cubic_spline(closed, x, 'periodic'))
        for order in (1, 2):
            assert periodic[0].diff(x, order).subs(x, nodes[0]) == periodic[-1].diff(x, order).subs(x, nodes[-1])


def test_cubic_spline_known_and_rejections() -> None:
    # a cubic is its own not-a-knot and clamped spline
    cubic = x**3 - 2*x + 1
    points: list[tuple[Expr, Expr]] = [(Integer(k), cubic.subs(x, k)) for k in range(-2, 3)]
    assert cubic_spline(points, x, 'not-a-knot') == cubic
    assert cubic_spline(points, x, 'clamped', (cubic.diff(x).subs(x, -2), cubic.diff(x).subs(x, 2))) == cubic
    # Stoer, Bulirsch: the natural spline through (0, 0), (1, 1), (2, 0) has M_1 = -3
    spline = cubic_spline([(0, 0), (1, 1), (2, 0)], x)
    assert isinstance(spline, Piecewise) and _branch(spline, Rational(1, 2)).diff(x, 2).subs(x, 1) == -3
    assert cubic_spline([(0, 1), (2, 5)], x) == 2*x + 1
    with pytest.raises(ValueError):
        cubic_spline([(0, 0), (0, 1)], x)
    with pytest.raises(ValueError):
        cubic_spline([(1, 0), (0, 1)], x)
    with pytest.raises(ValueError):
        cubic_spline([(0, 0)], x)
    with pytest.raises(ValueError):
        cubic_spline([(0, 0), (1, 1)], x, 'clamped')
    with pytest.raises(ValueError):
        cubic_spline([(0, 0), (1, 1)], x, 'periodic')
    with pytest.raises(ValueError):
        cubic_spline([(0, 0), (1, 1)], x, 'parabolic')


def test_least_squares() -> None:
    rnd = random.Random(154)
    for _ in range(12):
        points = _points(rnd, rnd.randint(4, 7))
        degree = rnd.randint(0, 3)
        fit = polynomial_fit(points, x, degree)
        # the residual is orthogonal to every function of the basis (the normal equations)
        for k in range(degree + 1):
            assert sum(((v - fit.subs(x, p))*p**k for p, v in points), Integer(0)) == 0
        if degree == 1:
            xs = [sympy_expr(to_sympy(p)) for p, _ in points]
            ys = [sympy_expr(to_sympy(v)) for _, v in points]
            n = len(points)
            mean_x, mean_y = sympy.Add(*xs)/n, sympy.Add(*ys)/n
            covariance = sympy.Add(*[(p - mean_x)*(v - mean_y) for p, v in zip(xs, ys)])
            variance = sympy.Add(*[(p - mean_x)**2 for p in xs])
            expected = covariance/variance*(sx - mean_x) + mean_y
            assert sympy.expand(sympy_expr(to_sympy(fit)) - expected) == 0
    # as many parameters as points: interpolation; data on a model are recovered
    exact: list[tuple[Expr, Expr]] = [(Integer(k), Integer(k**2 - k + 3)) for k in range(5)]
    assert polynomial_fit(exact, x, 2) == x**2 - x + 3 and polynomial_fit(exact[:3], x, 2) == x**2 - x + 3
    trig: list[tuple[Expr, Expr]] = [(Integer(k), 2*sin(Integer(k)) - cos(Integer(k))) for k in range(4)]
    assert normal(least_squares(trig, x, [sin(x), cos(x)]) - (2*sin(x) - cos(x))) == 0
    with pytest.raises(ValueError):
        least_squares([(0, 1), (0, 2)], x, [1, x])
    with pytest.raises(ValueError):
        least_squares([(0, 1)], x, [])
    with pytest.raises(ValueError):
        polynomial_fit([(0, 1)], x, -1)
