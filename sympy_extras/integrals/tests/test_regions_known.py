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


def test_arc_lengths_and_surface_areas() -> None:
    # Stewart 8.1: the length of the arc of y = x**2 from 0 to 1 is sqrt(5)/2 + asinh(2)/4
    # (= sqrt(5)/2 + log(2 + sqrt(5))/4), and the circumference of the unit circle is 2 pi
    from sympy import Eq, asinh, simplify, sqrt
    assert simplify(integrate_by_ranges(1, Eq(y, x**2) & (x > 0) & (x < 1), measure='hausdorff')
                    - (sqrt(5)/2 + asinh(2)/4)) == 0
    assert integrate_by_ranges(1, Eq(x**2 + y**2, 1), measure='hausdorff') == 2*pi
    # Stewart 16.6: the area of the unit sphere is 4 pi, and the area of the
    # paraboloid z = x**2 + y**2 under z = 1 is pi (5 sqrt(5) - 1)/6
    assert integrate_by_ranges(1, Eq(x**2 + y**2 + z**2, 1), measure='hausdorff') == 4*pi
    assert simplify(integrate_by_ranges(1, Eq(z, x**2 + y**2) & (z < 1), measure='hausdorff')
                    - pi*(5*sqrt(5) - 1)/6) == 0


def test_regions_under_curves_solved_for_the_last_variable() -> None:
    # Apostol II 11.11: the area under y = sqrt(x) on [0, 1] is 2/3, written with
    # the bound y**2 < x; the area under y = log(x) on [1, 2] written with exp(y) < x
    from sympy import log, simplify
    assert integrate_by_ranges(1, (x > 0) & (x < 1) & (y > 0) & (y**2 < x)) == Rational(2, 3)
    assert simplify(integrate_by_ranges(1, (x > 1) & (x < 2) & (y > 0) & (exp(y) < x)) - (2*log(2) - 1)) == 0


def test_curves_in_space_and_dirichlet_volumes() -> None:
    # Viviani's curve (Stewart 13.1, exercise on the sphere-cylinder intersection):
    # the length Integral(sqrt(1 + cos(t)**2), (t, 0, 2 pi)) = 4 sqrt(2) E(1/2)
    from sympy import Eq, N, gamma
    import mpmath
    found = integrate_by_ranges(1, Eq(x**2 + y**2 + z**2, 1) & Eq(x**2 + y**2, x), measure='hausdorff')
    assert abs(float(N(found)) - float(4*mpmath.sqrt(2)*mpmath.ellipe(0.5))) < 1e-12
    # Apostol II 11.9: the area of {x**3 + y**3 < 1, x > 0, y > 0} is
    # Integral((1 - x**3)**(1/3), (x, 0, 1)) = B(1/3, 4/3)/3 = Gamma(1/3) Gamma(4/3)/(3 Gamma(5/3))
    found = integrate_by_ranges(1, (x**3 + y**3 < 1) & (x > 0) & (y > 0))
    exact = gamma(Rational(1, 3))*gamma(Rational(4, 3))/(3*gamma(Rational(5, 3)))
    assert abs(float(N(found - exact))) < 1e-12
    assert abs(float(N(found)) - float(mpmath.quad(lambda t: (1 - t**3)**(mpmath.mpf(1)/3), [0, 1]))) < 1e-12
    # the circle z = 1 on the paraboloid z = x**2 + y**2 has length 2 pi
    assert integrate_by_ranges(1, Eq(z, x**2 + y**2) & Eq(z, 1), measure='hausdorff') == 2*pi
