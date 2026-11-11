"""Conics and quadrics: one equation of each type in canonical form, the
invariance of the type under affine changes of coordinates and of the
metric data under rigid motions."""
from __future__ import annotations

import random

import pytest

from sympy_extras.core import Integer, Rational, Symbol
from sympy_extras.core.expr import Expr
from sympy_extras.functions import sqrt
from sympy_extras.geometry.quadrics import classify_conic, classify_quadric, quadric_matrices
from sympy_extras.matrices import ExprMatrix
from sympy_extras.matrices.linalg import det, normal

x, y, z = Symbol('x'), Symbol('y'), Symbol('z')

CONICS: list[tuple[Expr, str]] = [
    (x**2/4 + y**2/9 - 1, 'ellipse'), (x**2 + y**2 - 4, 'circle'), (x**2 - y**2 - 1, 'hyperbola'), (y**2 - x, 'parabola'),
    (x**2 - y**2, 'two intersecting lines'), (x**2 - 1, 'two parallel lines'), (x**2, 'double line'),
    (x + y - 1, 'line'), (x**2 + y**2, 'point'), (x**2 + y**2 + 1, 'empty set'), (x**2 + 1, 'empty set'),
    (Integer(1) + 0*x, 'empty set'), (Integer(0)*x, 'plane'),
]

QUADRICS: list[tuple[Expr, str]] = [
    (x**2 + y**2/4 + z**2/9 - 1, 'ellipsoid'), (x**2 + y**2 + z**2 + 1, 'empty set'),
    (x**2 + y**2 - z**2 - 1, 'hyperboloid of one sheet'), (x**2 + y**2 - z**2 + 1, 'hyperboloid of two sheets'),
    (x**2 + y**2 - z, 'elliptic paraboloid'), (x**2 - y**2 - z, 'hyperbolic paraboloid'),
    (x**2 + y**2 - z**2, 'cone'), (x**2 + y**2 + z**2, 'point'),
    (x**2 + y**2 - 1, 'elliptic cylinder'), (x**2 + y**2 + 1, 'empty set'), (x**2 - y**2 - 1, 'hyperbolic cylinder'),
    (x**2 - y, 'parabolic cylinder'), (x**2 - y**2, 'two intersecting planes'), (x**2 + y**2, 'line'),
    (x**2 - 1, 'two parallel planes'), (x**2 + 1, 'empty set'), (x**2, 'double plane'), (x + y + z - 1, 'plane'),
    (Integer(2) + 0*x, 'empty set'), (Integer(0)*x, 'space'),
]


def _affine(rnd: random.Random, n: int) -> tuple[list[list[int]], list[int]]:
    while True:
        rows = [[rnd.randint(-2, 2) for _ in range(n)] for _ in range(n)]
        if det(ExprMatrix.from_rows(rows)) != 0:
            return rows, [rnd.randint(-2, 2) for _ in range(n)]


def _transformed(e: Expr, variables: list[Symbol], rows: list[list[int]], shift: list[int]) -> Expr:
    fresh = [Symbol('u%d' % i) for i in range(len(variables))]
    images = [sum((Integer(rows[i][j])*fresh[j] for j in range(len(variables))), Integer(shift[i])) for i in range(len(variables))]
    result = e
    for v, image in zip(variables, images):
        result = result.subs(v, image)
    for v, w in zip(fresh, variables):
        result = result.subs(v, w)
    return result.expand()


def test_canonical_conics_and_affine_invariance() -> None:
    rnd = random.Random(141)
    for equation, kind in CONICS:
        assert classify_conic(equation, x, y).kind == kind, equation
        assert classify_conic(-2*equation, x, y).kind == kind
        for _ in range(3):
            rows, shift = _affine(rnd, 2)
            image = classify_conic(_transformed(equation, [x, y], rows, shift), x, y).kind
            # a circle is an ellipse for the affine group
            assert image == kind or {image, kind} == {'circle', 'ellipse'}, (equation, rows, shift)


def test_canonical_quadrics_and_affine_invariance() -> None:
    rnd = random.Random(142)
    for equation, kind in QUADRICS:
        assert classify_quadric(equation, x, y, z).kind == kind, equation
        assert classify_quadric(-3*equation, x, y, z).kind == kind
        for _ in range(2):
            rows, shift = _affine(rnd, 3)
            assert classify_quadric(_transformed(equation, [x, y, z], rows, shift), x, y, z).kind == kind, (equation, rows, shift)


def test_metric_data_under_rigid_motions() -> None:
    # the rotation with cos = 3/5, sin = 4/5 and a translation
    c, s = Rational(3, 5), Rational(4, 5)

    def moved(e: Expr, p: int, q: int) -> Expr:
        u, v = Symbol('u'), Symbol('v')
        result = e.subs(x, c*u - s*v + p).subs(y, s*u + c*v + q)
        return result.subs(u, x).subs(v, y).expand()

    ellipse = classify_conic(moved(x**2/25 + y**2/9 - 1, 2, -1), x, y)
    assert ellipse.kind == 'ellipse' and ellipse.semi_axes == (5, 3) and ellipse.eccentricity == Rational(4, 5)
    # the centre: the origin goes to the solution of the motion, (u, v) with x = y = 0
    assert ellipse.center is not None
    cx, cy = ellipse.center
    assert normal(c*cx - s*cy + 2) == 0 and normal(s*cx + c*cy - 1) == 0
    hyperbola = classify_conic(moved(x**2/16 - y**2/9 - 1, 1, 3), x, y)
    assert hyperbola.kind == 'hyperbola' and hyperbola.semi_axes == (4, 3) and hyperbola.eccentricity == Rational(5, 4)
    conjugate = classify_conic(moved(y**2/16 - x**2/9 - 1, 0, 0), x, y)
    assert conjugate.semi_axes == (4, 3)
    parabola = classify_conic(moved(y**2 - 6*x, -2, 5), x, y)
    assert parabola.kind == 'parabola' and parabola.parameter == 3 and parabola.center is None
    circle = classify_conic(moved(x**2 + y**2 - 7, 1, 1), x, y)
    assert circle.kind == 'circle' and circle.semi_axes == (sqrt(Integer(7)), sqrt(Integer(7))) and circle.eccentricity == 0


def test_known_examples_and_rejections() -> None:
    # Korn and Korn 2.4-8: 5 x**2 + 4 x y + 8 y**2 = 36 has the principal form X**2/9 + Y**2/4 = 1
    conic = classify_conic(5*x**2 + 4*x*y + 8*y**2 - 36, x, y)
    assert conic.semi_axes == (3, 2) and conic.canonical == (4, 9, -36)
    assert classify_conic(x*y - 1, x, y).semi_axes == (sqrt(Integer(2)), sqrt(Integer(2)))       # the equilateral hyperbola
    assert classify_quadric(x*y + y*z + z*x - 1, x, y, z).kind == 'hyperboloid of two sheets'
    assert classify_quadric(x*y - z, x, y, z).center is None
    assert quadric_matrices(x*y + 2*z, [x, y, z])[1] == ExprMatrix.from_rows(
        [[0, Rational(1, 2), 0, 0], [Rational(1, 2), 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    # parameters with known signs, and without
    a, b = Symbol('a', positive=True), Symbol('b', positive=True)
    with_parameters = classify_conic(x**2/a**2 + y**2/b**2 - 1, x, y)
    assert with_parameters.kind == 'ellipse' and with_parameters.center == (0, 0)
    assert classify_conic(x**2/a - y**2/b - 1, x, y).kind == 'hyperbola'
    k = Symbol('k')
    with pytest.raises(ValueError):
        classify_conic(x**2 + k*y**2 - 1, x, y)
    with pytest.raises(ValueError):
        classify_conic(x**3 + y, x, y)
    with pytest.raises(ValueError):
        classify_conic(1/x + y, x, y)
