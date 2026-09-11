"""Integrals over regions with values from the literature: the classical
areas, volumes and moments of calculus textbooks (Stewart, *Calculus*,
chapter 15; Apostol, *Calculus* vol. II, chapter 11) and the examples of
the documentation of Mathematica's ``Integrate`` over regions."""
from __future__ import annotations

from sympy import Rational, exp, pi, symbols

from sympy_extras.integrals.regions import integrate_by_ranges

x, y, z = symbols('x y z')


def test_disc() -> None:
    # the area of the unit disc
    assert integrate_by_ranges(1, x**2 + y**2 < 1) == pi
    # the second moment of the unit disc about the y axis: Integral(r**3 cos**2, ...) = pi/4
    assert integrate_by_ranges(x**2, x**2 + y**2 < 1) == pi/4
    # the quarter disc: Stewart 15.3, area pi/4 and the first moment 1/3
    assert integrate_by_ranges(1, (x**2 + y**2 < 1) & (x > 0) & (y > 0)) == pi/4
    assert integrate_by_ranges(x, (x**2 + y**2 < 1) & (x > 0) & (y > 0)) == Rational(1, 3)


def test_ball() -> None:
    # the volume of the unit ball, 4 pi / 3 (Apostol II, 11.28)
    assert integrate_by_ranges(1, x**2 + y**2 + z**2 < 1) == 4*pi/3


def test_simplices() -> None:
    # the standard 2-simplex has area 1/2 and the 3-simplex volume 1/6
    assert integrate_by_ranges(1, (x > 0) & (y > 0) & (x + y < 1)) == Rational(1, 2)
    assert integrate_by_ranges(1, (x > 0) & (y > 0) & (z > 0) & (x + y + z < 1)) == Rational(1, 6)
    # Stewart 15.2, example: Integral of x*y over the triangle with vertices (0, 0), (1, 0), (0, 1)
    assert integrate_by_ranges(x*y, (x > 0) & (y > 0) & (x + y < 1)) == Rational(1, 24)


def test_gaussian_half_plane() -> None:
    # Integral(exp(-x**2 - y**2)) over the upper half plane is half of pi
    assert integrate_by_ranges(exp(-x**2 - y**2), y > 0, [x, y]) == pi/2


def test_polar_and_type_one_regions() -> None:
    # Apostol II 11.28: Integral(exp(-x**2 - y**2)) over the disc of radius a is pi (1 - exp(-a**2))
    a = symbols('a', positive=True)
    from sympy import simplify
    assert simplify(integrate_by_ranges(exp(-x**2 - y**2), x**2 + y**2 < a**2, [x, y]) - pi*(1 - exp(-a**2))) == 0
    # Apostol II 11.11 (regions between two graphs): the area under y = exp(x)
    # on [0, 1] is e - 1, and the area between y = x**2 and y = sqrt(x) is 1/3
    from sympy import E, sqrt
    assert simplify(integrate_by_ranges(1, (x > 0) & (x < 1) & (y > 0) & (y < exp(x))) - (E - 1)) == 0
    assert integrate_by_ranges(1, (x > 0) & (x < 1) & (y > x**2) & (y < sqrt(x))) == Rational(1, 3)


def test_ellipses_cylinders_and_cones() -> None:
    from sympy import sqrt, symbols as _symbols
    a, b, c = _symbols('a b c', positive=True)
    # Apostol 11.28, Example 1: the area of the ellipse is pi a b
    assert integrate_by_ranges(1, x**2/a**2 + y**2/b**2 < 1, [x, y]) == pi*a*b
    # the volume of the ellipsoid, 4 pi a b c/3 (Apostol 11.30)
    assert integrate_by_ranges(1, x**2/a**2 + y**2/b**2 + z**2/c**2 < 1, [x, y, z]) == 4*pi*a*b*c/3
    # Stewart 15.7: the volume under the paraboloid z = x**2 + y**2 over the unit disc is pi/2,
    # and the volume of the cone sqrt(x**2 + y**2) < z < 1 is pi/3
    assert integrate_by_ranges(1, (x**2 + y**2 < 1) & (z > 0) & (z < x**2 + y**2)) == pi/2
    assert integrate_by_ranges(1, (sqrt(x**2 + y**2) < z) & (z < 1)) == pi/3
