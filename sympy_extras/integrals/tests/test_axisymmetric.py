"""Tests of the reduction of region integrals with a rotational symmetry."""
from __future__ import annotations

import mpmath

from sympy import symbols, Eq, And, Or, Not, sqrt, pi, S, Rational, N, asin
from sympy.core.expr import Expr

from sympy_extras._typing import ExprLike, as_expr
from sympy_extras.integrals.axisymmetric import axisymmetric_integral, rotation_group
from sympy_extras.integrals.regions import integrate_by_ranges

x, y, z, w = symbols('x y z w')
#: rho**2 at the intersection of the unit sphere and the paraboloid z = rho**2
_A = (sqrt(5) - 1) / 2


def _close(value: object, expected: ExprLike, digits: int = 10) -> bool:
    assert isinstance(value, Expr)
    return abs(complex(N(value, 15)) - complex(N(as_expr(expected), 15))) < 10.0**(-digits)


def test_rotation_group() -> None:
    assert rotation_group(as_expr(1), (x**2 + y**2 + z**2 < 1) & (z > x**2 + y**2), [x, y, z]) == (x, y)
    assert rotation_group(as_expr(1), (x**2 + y**2 + z**2 < 1) & ((x - 1)**2 + y**2 + z**2 < 1), [x, y, z]) == (y, z)
    # the whole set when everything is radial, none when a variable appears alone
    assert rotation_group(as_expr(1), x**2 + y**2 + z**2 < 1, [x, y, z]) == (x, y, z)
    assert rotation_group(x * y, x**2 + y**2 + z**2 < 1, [x, y, z]) is None
    assert rotation_group(as_expr(1), (x**2 + y**2 < 1) & (y**2 + z**2 < 1), [x, y, z]) is None
    # the integrand takes part: z is not rotated with x, y when it is the integrand
    assert rotation_group(z, x**2 + y**2 + z**2 < 1, [x, y, z]) == (x, y)
    # four variables, a group of three
    assert rotation_group(as_expr(1), (x**2 + y**2 + z**2 + w**2 < 1) & (w > S.Half), [x, y, z, w]) == (x, y, z)


def test_volumes_of_revolution() -> None:
    ball = x**2 + y**2 + z**2 < 1
    shifted = (x - 1)**2 + y**2 + z**2 < 1
    # two unit balls a distance 1 apart: the lens pi (4 R + d)(2 R - d)**2/12
    assert axisymmetric_integral(as_expr(1), And(ball, shifted), [x, y, z]) == 5 * pi / 12
    assert axisymmetric_integral(as_expr(1), Or(ball, shifted), [x, y, z]) == 9 * pi / 4
    # a cap of height 1/2: pi h**2 (3 R - h)/3
    assert axisymmetric_integral(as_expr(1), And(ball, z > S.Half), [x, y, z]) == 5 * pi / 24
    assert axisymmetric_integral(z, And(ball, z > S.Half), [x, y, z]) == 9 * pi / 64
    # the ball cut by the paraboloid z = x**2 + y**2, meeting it at rho**2 = (sqrt(5) - 1)/2
    found = axisymmetric_integral(as_expr(1), And(ball, z > x**2 + y**2), [x, y, z])
    expected = 2 * mpmath.pi * mpmath.quad(lambda r: r * (mpmath.sqrt(1 - r**2) - r**2), [0, mpmath.sqrt((mpmath.sqrt(5) - 1) / 2)])
    assert found is not None and _close(found, expected)
    assert found.equals(5 * pi * (3 - sqrt(5)) / 12)
    # the ball cut by a cylinder
    found = axisymmetric_integral(as_expr(1), And(ball, x**2 + y**2 < Rational(1, 4)), [x, y, z])
    assert found is not None and _close(found, 4 * pi * (1 - Rational(3, 4)**Rational(3, 2)) / 3)
    # the bug: the radius was a positive Dummy, rho > 0 evaluated to True
    # and vanished, and the odd weight rho integrated to 0 over both signs
    assert found != 0


def test_areas_of_revolution() -> None:
    sphere = Eq(x**2 + y**2 + z**2, 1)
    assert axisymmetric_integral(as_expr(1), And(sphere, z > S.Half), [x, y, z], measure='hausdorff') == pi
    found = axisymmetric_integral(as_expr(1), And(sphere, z > x**2 + y**2), [x, y, z], measure='hausdorff')
    assert found is not None and found.equals(pi * (3 - sqrt(5)))
    # the paraboloid inside the ball: 2 pi Integral(rho sqrt(1 + 4 rho**2)) up to rho**2 = a
    found = axisymmetric_integral(as_expr(1), And(Eq(z, x**2 + y**2), x**2 + y**2 + z**2 < 1), [x, y, z],
                                  measure='hausdorff')
    assert found is not None and _close(found, pi / 6 * ((1 + 4 * _A)**Rational(3, 2) - 1))
    # two equations: the circle of intersection of two spheres, radius sqrt(3)/2
    found = axisymmetric_integral(as_expr(1), And(sphere, Eq((x - 1)**2 + y**2 + z**2, 1)), [x, y, z],
                                  measure='hausdorff')
    assert found == sqrt(3) * pi


def test_four_dimensions() -> None:
    ball = x**2 + y**2 + z**2 + w**2 < 1
    shifted = (x - 1)**2 + y**2 + z**2 + w**2 < 1
    lens = 2 * mpmath.quad(lambda t: 4 * mpmath.pi / 3 * (1 - t**2)**1.5, [0.5, 1])
    found = axisymmetric_integral(as_expr(1), And(ball, shifted), [x, y, z, w])
    assert found is not None and _close(found, lens) and found.equals(pi * (8 * pi - 9 * sqrt(3)) / 24)
    cap = mpmath.quad(lambda t: 4 * mpmath.pi * mpmath.sqrt(1 - t**2), [0.5, 1])
    found = axisymmetric_integral(as_expr(1), And(Eq(x**2 + y**2 + z**2 + w**2, 1), w > S.Half), [x, y, z, w],
                                  measure='hausdorff')
    assert found is not None and _close(found, cap) and found.equals(pi * (4 * pi - 3 * sqrt(3)) / 6)
    paraboloid = mpmath.quad(lambda r: 4 * mpmath.pi * r**2 * (mpmath.sqrt(1 - r**2) - r**2),
                             [0, mpmath.sqrt((mpmath.sqrt(5) - 1) / 2)])
    found = axisymmetric_integral(as_expr(1), And(ball, w > x**2 + y**2 + z**2), [x, y, z, w])
    assert found is not None and _close(found, paraboloid) and found.has(asin)


def test_through_the_entry_point() -> None:
    # integrate_by_ranges takes the route before the decomposition
    assert integrate_by_ranges(1, (x**2 + y**2 + z**2 < 1) & ((x - 1)**2 + y**2 + z**2 < 1)) == 5 * pi / 12
    assert integrate_by_ranges(1, Eq(x**2 + y**2 + z**2, 1) & (z > S.Half), measure='hausdorff') == pi
    assert integrate_by_ranges(1, (x**2 + y**2 + z**2 < 4) & Not((x - 1)**2 + y**2 + z**2 < 1)) == 28 * pi / 3
    # no symmetry: the route declines and the decomposition answers
    assert axisymmetric_integral(as_expr(1), (x**2 + y**2 < 1) & (y**2 + z**2 < 1), [x, y, z]) is None


def test_tilted_planes_through_the_aligned_frame() -> None:
    # a ball cut by planes with a common normal is a body of revolution
    # about that normal: the frame along it, scaled by |a| so that the
    # offsets stay rational, a similarity which scales a measure of
    # dimension d by |a|**(-d) (the bug: an orthonormal frame with the
    # offsets as parameters made the decomposition parametric, and a
    # scaled but non-orthonormal frame kept the volumes but not the areas)
    from sympy_extras.integrals.axisymmetric import aligned_frame
    ball = x**2 + y**2 + z**2 < 1
    sphere = Eq(x**2 + y**2 + z**2, 1)
    h = 1 - 1 / sqrt(3)                                     # the height of the cap x + y + z > 1
    assert axisymmetric_integral(as_expr(1), And(ball, x + y + z > 0), [x, y, z]) == 2 * pi / 3
    cap = axisymmetric_integral(as_expr(1), And(ball, x + y + z > 1), [x, y, z])
    assert cap is not None and cap.equals(pi * h**2 * (3 - h) / 3)
    slab = axisymmetric_integral(as_expr(1), And(ball, x + y + z > -1, x + y + z < 1), [x, y, z])
    assert slab is not None and slab.equals(4 * pi / 3 - 2 * pi * h**2 * (3 - h) / 3)
    moment = axisymmetric_integral(x + y + z, And(ball, x + y + z > 1), [x, y, z])
    assert moment is not None and _close(moment, mpmath.quad(lambda t: mpmath.sqrt(3) * t * mpmath.pi * (1 - t**2),
                                                              [1 / mpmath.sqrt(3), 1]))
    area = axisymmetric_integral(as_expr(1), And(sphere, x + y + z > 1), [x, y, z], measure='hausdorff')
    assert area is not None and area.equals(2 * pi * h)
    circle = axisymmetric_integral(as_expr(1), And(sphere, Eq(x + y + z, 1)), [x, y, z], measure='hausdorff')
    assert circle == 2 * sqrt(6) * pi / 3
    # four dimensions, and two normals refused
    cap4 = axisymmetric_integral(as_expr(1), And(x**2 + y**2 + z**2 + w**2 < 1, x + y + z + w > 1), [x, y, z, w])
    assert cap4 is not None and cap4.equals(pi * (8 * pi - 9 * sqrt(3)) / 48)
    assert axisymmetric_integral(as_expr(1), And(ball, x + y + z > 0, x - y > 0), [x, y, z]) is None
    assert aligned_frame(as_expr(1), And(ball, x + y + z > 0, x - y > 0), [x, y, z]) is None
    # through the entry point
    assert integrate_by_ranges(1, And(ball, x + y + z > 0)) == 2 * pi / 3
