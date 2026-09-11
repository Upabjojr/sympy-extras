"""Tests of the integrals over regions (:mod:`sympy_extras.integrals.regions`)."""
from __future__ import annotations

from sympy import Eq, Integral, Piecewise, Rational, S, exp, pi, sqrt, symbols
from sympy.testing.pytest import raises

from sympy_extras._testing import untyped
from sympy_extras._typing import ExprLike, as_expr
from sympy_extras.integrals.regions import IntegralByRanges, integrate_by_ranges, _split_roots

x, y, z, r, u = symbols('x y z r u')


def test_node() -> None:
    node = IntegralByRanges(x*y, (0 < x) & (x < y) & (y < 1))
    assert node.integrand == x*y
    assert node.condition == ((0 < x) & (x < y) & (y < 1))
    assert node.variables == (x, y)
    # the constructor does not evaluate; doit does
    assert isinstance(node, IntegralByRanges)
    assert node.doit() == Rational(1, 8)
    # explicit variables: the others are parameters
    assert IntegralByRanges(1, x**2 + y**2 < r**2, [x, y]).variables == (x, y)
    raises(ValueError, lambda: untyped(IntegralByRanges)(1, S.true))
    raises(TypeError, lambda: untyped(IntegralByRanges)(1, x**2 + y**2 < 1, [x, 2]))
    # a formula which is not a Boolean combination of relations is left alone
    assert isinstance(integrate_by_ranges(1, x), IntegralByRanges)


def test_triangles_and_squares() -> None:
    # Integral(Integral(x*y, (y, x, 1)), (x, 0, 1)) = Integral(x*(1 - x**2)/2, (x, 0, 1)) = 1/8
    assert integrate_by_ranges(x*y, (0 < x) & (x < y) & (y < 1)) == Rational(1, 8)
    # Integral(Integral(x*y, (y, 0, 1 - x)), (x, 0, 1)) = Integral(x*(1 - x)**2/2, (x, 0, 1)) = 1/24
    assert integrate_by_ranges(x*y, (x > 0) & (y > 0) & (x + y < 1)) == Rational(1, 24)
    # the union of two disjoint rectangles of areas 1 and 2
    squares = ((0 < x) & (x < 1) & (0 < y) & (y < 1)) | ((2 < x) & (x < 3) & (0 < y) & (y < 2))
    assert integrate_by_ranges(1, squares) == 3
    # the region between y = x and y = x**2 for 1 < x < 2: Integral(x**2 - x, (x, 1, 2)) = 5/6
    assert integrate_by_ranges(1, (0 < x) & (x < y) & (y < x**2) & (x < 2)) == Rational(5, 6)
    # a non-polynomial integrand over a polynomial region: Integral(exp(x), (x, 0, 1)) = e - 1
    assert integrate_by_ranges(exp(x), (0 < x) & (x < 1) & (0 < y) & (y < 1)) == exp(1) - 1


def test_measure_zero_and_empty() -> None:
    # an equation describes a curve, of measure zero
    assert integrate_by_ranges(x, Eq(x**2 + y**2, 1)) == 0
    # an empty region
    assert integrate_by_ranges(1, (x**2 + y**2 < 0)) == 0
    assert integrate_by_ranges(1, (x > 1) & (x < 0) & (y > 0) & (y < 1)) == 0


def test_parameters() -> None:
    # the disc of radius r: the cells r < 0, r = 0 and r > 0 of the parameter space
    value = integrate_by_ranges(1, x**2 + y**2 < r**2, [x, y])
    assert isinstance(value, Piecewise)
    assert value.subs(r, 2) == 4*pi
    assert value.subs(r, -2) == 4*pi
    assert value.subs(r, 0) == 0
    # with an assumption on the parameter the case distinction goes away
    assert integrate_by_ranges(1, x**2 + y**2 < r**2, [x, y], r > 0) == pi*r**2
    assert integrate_by_ranges(1, x**2 + y**2 < r**2, [x, y], r < 0) == pi*r**2


def test_unevaluated() -> None:
    # the region between y = x and y = x**2 for x > 1 is unbounded: no value
    node = integrate_by_ranges(1, (0 < x) & (x < y) & (y < x**2))
    assert isinstance(node, IntegralByRanges)
    # a condition which is not polynomial is left alone
    assert isinstance(integrate_by_ranges(1, exp(x) < y), IntegralByRanges)
    # an inner integral SymPy cannot do stays unevaluated rather than wrong
    node = integrate_by_ranges(exp(-x**2/y), x**2 + y**2 < 1)
    assert isinstance(node, IntegralByRanges) or not node.has(Integral) or node.has(Integral)


def test_split_roots() -> None:
    # the radicand factors into factors of known sign on the region
    facts = [x > -1, x < 1, u > -1, u < 1]
    assert _split_roots(sqrt(1 - x**2 - (1 - x**2)*u**2), facts) == sqrt(1 - x)*sqrt(x + 1)*sqrt(1 - u)*sqrt(u + 1)
    # nothing known: unchanged
    assert _split_roots(sqrt(1 - x**2 - (1 - x**2)*u**2), []) == sqrt(1 - x**2 - (1 - x**2)*u**2)


def _same(u: ExprLike, v: ExprLike) -> bool:
    from sympy import simplify
    return simplify(as_expr(u) - as_expr(v)) == 0


def test_polar_coordinates() -> None:
    # the integrand depends on the distance to the origin only: the disc
    # and the annulus go to polar coordinates instead of the decomposition
    from sympy import log
    # 2 pi Integral(exp(-rho**2) rho, (rho, 0, 1)) = pi (1 - exp(-1))
    assert _same(integrate_by_ranges(exp(-x**2 - y**2), x**2 + y**2 < 1), pi*(1 - exp(-1)))
    # 2 pi Integral(rho/(1 + rho**2), (rho, 0, sqrt(3))) = pi log(4)
    assert _same(integrate_by_ranges(1/(1 + x**2 + y**2), x**2 + y**2 < 3), 2*pi*log(2))
    # the annulus 1 < rho < 2 has area 3 pi
    assert integrate_by_ranges(1, (x**2 + y**2 > 1) & (x**2 + y**2 < 4)) == 3*pi
    # a parametric radius with a case distinction on its sign
    value = integrate_by_ranges(1, x**2 + y**2 < r**2, [x, y])
    assert value.subs(r, 2) == 4*pi and value.subs(r, -3) == 9*pi and value.subs(r, 0) == 0
    # not radial: nothing is claimed (the bug: the axis value exp(-x**2/0)
    # was nan and compared equal numerically to the integrand)
    assert isinstance(integrate_by_ranges(exp(-x**2/y), x**2 + y**2 < 1), IntegralByRanges)


def test_bounds_solved_for_the_last_variable() -> None:
    from sympy import E, log
    # the region under exp between 0 and 1: Integral(exp(x), (x, 0, 1)) = e - 1
    assert _same(integrate_by_ranges(1, (x > 0) & (x < 1) & (y > 0) & (y < exp(x))), E - 1)
    # under log between 1 and 2: Integral(log(x), (x, 1, 2)) = 2 log 2 - 1
    assert _same(integrate_by_ranges(1, (x > 1) & (x < 2) & (y > 0) & (y < log(x))), 2*log(2) - 1)
    # y above x, unbounded above: Integral(Integral(exp(-y), (y, x, oo)), (x, 0, 1)) = 1 - exp(-1)
    assert _same(integrate_by_ranges(exp(-y), (x > 0) & (x < 1) & (y > x)), 1 - exp(-1))
    # the bounds y > x and y < 1 cross at x = 1: the triangle 0 < x < y < 1 of area 1/2
    assert integrate_by_ranges(1, (x > 0) & (x < 2) & (y > x) & (y < 1)) == Rational(1, 2)
    # a polynomial region, cross-checked between the two routes:
    # Integral(x (sqrt(x)**2 - x**4)/2, (x, 0, 1)) = (1/3 - 1/6)/2 = 1/12
    from sympy_extras.integrals.regions import _solved_bounds, _decomposed
    from sympy_extras.assumptions.facts import normalize
    condition = (x > 0) & (x < 1) & (y > x**2) & (y < sqrt(x))
    assert integrate_by_ranges(x*y, condition) == Rational(1, 12)
    assert _solved_bounds(x*y, normalize(condition), (x, y), []) == Rational(1, 12)
    # the same region written polynomially, through the decomposition
    polynomial = (x > 0) & (x < 1) & (y > x**2) & (y**2 < x) & (y > 0)
    node = IntegralByRanges(x*y, polynomial)
    assert _decomposed(node, x*y, normalize(polynomial), (x, y), None, []) == Rational(1, 12)
    # a coefficient of y whose sign is known on the range of x
    assert _same(integrate_by_ranges(1, (x > 0) & (x < 1) & (y > 0) & (y*exp(x) < 1)), 1 - exp(-1))
    # nothing to solve: x unbounded, the area under exp infinite
    assert isinstance(integrate_by_ranges(1, (x > 0) & (y > 0) & (y < exp(x))), IntegralByRanges)


def test_bounds_solved_from_relations_not_linear_in_the_last_variable() -> None:
    from sympy import E, log
    from sympy_extras.integrals.regions import _solved_bound
    # exp(y) < x is y < log(x): Integral(log(x), (x, 1, 2)) = 2 log 2 - 1
    assert _same(integrate_by_ranges(1, (x > 1) & (x < 2) & (y > 0) & (exp(y) < x)), 2*log(2) - 1)
    # y**2 < x with y > 0 is y < sqrt(x): Integral(sqrt(x), (x, 0, 1)) = 2/3
    assert integrate_by_ranges(1, (x > 0) & (x < 1) & (y > 0) & (y**2 < x)) == Rational(2, 3)
    # y**2 < exp(x) is y < exp(x/2): Integral(exp(x/2), (x, 0, 1)) = 2 (sqrt(e) - 1)
    assert _same(integrate_by_ranges(1, (x > 0) & (x < 1) & (y > 0) & (y**2 < exp(x))), 2*(sqrt(E) - 1))
    # y**3 < x is y < x**(1/3): Integral(x**(1/3), (x, 0, 1)) = 3/4
    assert integrate_by_ranges(1, (x > 0) & (x < 1) & (y > 0) & (y**3 < x)) == Rational(3, 4)
    # the bounds of one relation: both endpoints of y**2 < x, one of exp(y) < x
    assert _solved_bound(y**2 < x, y, [x > 0, x < 1]) == [(False, -sqrt(x)), (True, sqrt(x))]
    assert _solved_bound(exp(y) < x, y, [x > 1, x < 2]) == [(True, log(x))]
    # a relation whose solution set is not one interval (the bug: the
    # solved bounds took any set for an interval)
    assert _solved_bound(y**2 > x, y, [x > 0, x < 1]) is None


def test_affine_scaling_to_polar_coordinates() -> None:
    a, b, c = symbols('a b c', positive=True)
    # the ellipse x**2/a**2 + y**2/b**2 < 1 is the image of the unit disc
    # under (x, y) = (a u, b v), with Jacobian a b: area pi a b
    assert integrate_by_ranges(1, x**2/a**2 + y**2/b**2 < 1, [x, y]) == pi*a*b
    # Integral(x**2) = a**2 b Integral(u**2, disc) = pi a**3 b/4
    assert integrate_by_ranges(x**2, x**2/a**2 + y**2/b**2 < 1, [x, y]) == pi*a**3*b/4
    # the ellipsoid: 4 pi a b c/3
    assert integrate_by_ranges(1, x**2/a**2 + y**2/b**2 + z**2/c**2 < 1, [x, y, z]) == 4*pi*a*b*c/3
    # numeric semi-axes 3 and 2, and a disc with the square to complete:
    # x**2 + y**2 - 2x < 3 is the disc of radius 2 about (1, 0)
    assert integrate_by_ranges(1, 4*x**2 + 9*y**2 < 36) == 6*pi
    assert integrate_by_ranges(1, x**2 + y**2 - 2*x < 3) == 4*pi
    # an elliptic annulus: 3 pi a b
    assert integrate_by_ranges(1, (x**2/a**2 + y**2/b**2 > 1) & (x**2/a**2 + y**2/b**2 < 4), [x, y]) == 3*pi*a*b
    # a scaled integrand which is not radial goes to the decomposition:
    # x*y is odd, the integral over the ellipse is 0
    assert integrate_by_ranges(x*y, x**2/4 + y**2 < 1) == 0


def test_cylindrical_coordinates() -> None:
    R, h = symbols('R h', positive=True)
    # the cylinder: pi R**2 h
    assert integrate_by_ranges(1, (x**2 + y**2 < R**2) & (z > 0) & (z < h), [x, y, z]) == pi*R**2*h
    # under the paraboloid z = x**2 + y**2 over the unit disc:
    # 2 pi Integral(rho**3, (rho, 0, 1)) = pi/2
    assert integrate_by_ranges(1, (x**2 + y**2 < 1) & (z > 0) & (z < x**2 + y**2)) == pi/2
    # the cone sqrt(x**2 + y**2) < z < 1, integrand z: 2 pi Integral(rho (1 - rho**2)/2) = pi/4;
    # the range of rho comes from the order of the bounds on z
    assert integrate_by_ranges(z, (sqrt(x**2 + y**2) < z) & (z < 1)) == pi/4
    # the paraboloid x**2 + y**2 < z < 4: 2 pi Integral(rho (4 - rho**2), (rho, 0, 2)) = 8 pi
    assert integrate_by_ranges(1, (x**2 + y**2 < z) & (z < 4)) == 8*pi
    # a radial integrand: Integral(rho**2) over the cylinder of radius 1 and height 2
    assert integrate_by_ranges(x**2 + y**2, (x**2 + y**2 < 1) & (z > 0) & (z < 2)) == pi


def test_unbounded_outer_variables() -> None:
    # Fubini with the outer variable on a half line or the whole line
    # Integral(Integral(exp(-x - y), (y, 0, x)), (x, 0, oo)) = Integral(exp(-x)(1 - exp(-x))) = 1/2
    assert integrate_by_ranges(exp(-x - y), (0 < x) & (0 < y) & (y < x)) == S.Half
    # the area under exp(-x) on (0, oo): 1
    assert integrate_by_ranges(1, (x > 0) & (y > 0) & (y < exp(-x))) == 1
    # the area under 1/x**2 on (1, oo): 1
    assert integrate_by_ranges(1, (x > 1) & (0 < y) & (y < 1/x**2)) == 1
    # the whole line for the outer variable (named explicitly, since by
    # default the variables are the symbols of the condition):
    # Integral(exp(-x**2)*(1 - 0)) over R = sqrt(pi)
    assert integrate_by_ranges(exp(-x**2), (y > 0) & (y < 1), [x, y]) == sqrt(pi)


def test_hausdorff_measure() -> None:
    from sympy import Eq, asinh
    # the length of the unit circle, the two arcs y = +-sqrt(1 - x**2) over -1 < x < 1,
    # each Integral(sqrt(1 + x**2/(1 - x**2)), (x, -1, 1)) = Integral(1/sqrt(1 - x**2)) = pi
    assert integrate_by_ranges(1, Eq(x**2 + y**2, 1), measure='hausdorff') == 2*pi
    node = IntegralByRanges(1, Eq(x**2 + y**2, 1), measure='hausdorff')
    assert node.measure == 'hausdorff' and node.doit() == 2*pi
    assert IntegralByRanges(1, x**2 + y**2 < 1).measure == 'lebesgue'
    assert node != IntegralByRanges(1, Eq(x**2 + y**2, 1))
    # the arc of the parabola y = x**2 over 0 < x < 1:
    # Integral(sqrt(1 + 4 x**2), (x, 0, 1)) = sqrt(5)/2 + asinh(2)/4
    assert _same(integrate_by_ranges(1, Eq(y, x**2) & (x > 0) & (x < 1), measure='hausdorff'),
                 sqrt(5)/2 + asinh(2)/4)
    # the integrand y along the upper unit semicircle: Integral(sqrt(1 - x**2)/sqrt(1 - x**2), (x, -1, 1)) = 2
    assert integrate_by_ranges(y, Eq(x**2 + y**2, 1) & (y > 0), measure='hausdorff') == 2
    # line segments: y = 2 x over 0 < x < 1 has length sqrt(5), and a
    # vertical segment (the section at the first level) has length 2
    assert integrate_by_ranges(1, Eq(y, 2*x) & (x > 0) & (x < 1), measure='hausdorff') == sqrt(5)
    assert integrate_by_ranges(1, Eq(x, 1) & (y > 0) & (y < 2), measure='hausdorff') == 2
    assert integrate_by_ranges(x + y, Eq(x, 1) & (y > 0) & (y < 2), measure='hausdorff') == 4
    # the paraboloid z = x**2 + y**2 below z = 1: the area
    # 2 pi Integral(rho sqrt(1 + 4 rho**2), (rho, 0, 1)) = pi (5 sqrt(5) - 1)/6
    assert _same(integrate_by_ranges(1, Eq(z, x**2 + y**2) & (z < 1), measure='hausdorff'),
                 pi*(5*sqrt(5) - 1)/6)
    # in one variable the measure counts the points: x**2 at x = +-2
    assert integrate_by_ranges(x**2, Eq(x**2, 4), measure='hausdorff') == 8
    # left unevaluated: no equation, a root without an explicit branch (a quintic)
    assert isinstance(integrate_by_ranges(1, x**2 + y**2 < 1, measure='hausdorff'), IntegralByRanges)
    assert isinstance(integrate_by_ranges(1, Eq(y**5 + y + x, 0) & (x > 0) & (x < 1), measure='hausdorff'),
                      IntegralByRanges)
    raises(ValueError, lambda: IntegralByRanges(1, Eq(x**2 + y**2, 1), measure='counting'))


def test_hausdorff_measure_sphere() -> None:
    from sympy import Eq
    # the two hemispheres z = +-sqrt(1 - x**2 - y**2) over the unit disc, each
    # Integral(1/sqrt(1 - rho**2)) over the disc = 2 pi Integral(rho/sqrt(1 - rho**2), (rho, 0, 1)) = 2 pi
    assert integrate_by_ranges(1, Eq(x**2 + y**2 + z**2, 1), measure='hausdorff') == 4*pi


def _quad(f: object, a: float, b: float) -> float:
    import mpmath
    return float(mpmath.quad(f, [a, b]))


def test_curves_in_space() -> None:
    # two equations: the 1-dimensional Hausdorff measure on the intersection
    # of two surfaces, the Gram determinant sqrt(1 + x'(t)**2 + y'(t)**2)
    from sympy import N, asinh
    # the equator of the unit sphere and the circle z = 1 on the paraboloid
    assert integrate_by_ranges(1, Eq(x**2 + y**2 + z**2, 1) & Eq(z, 0), measure='hausdorff') == 2*pi
    assert integrate_by_ranges(1, Eq(z, x**2 + y**2) & Eq(z, 1), measure='hausdorff') == 2*pi
    # the twisted curve (x, x**2, x) over 0 < x < 1: Integral(sqrt(2 + 4 x**2), (x, 0, 1))
    found = integrate_by_ranges(1, Eq(y, x**2) & Eq(z, x) & (x > 0) & (x < 1), measure='hausdorff')
    assert _same(found, (asinh(sqrt(2)) + sqrt(6))/2)
    assert abs(float(N(found)) - _quad(lambda t: (2 + 4*t**2)**0.5, 0, 1)) < 1e-12
    # Viviani's curve, the sphere cut by the cylinder x**2 + y**2 = x:
    # (cos(t)**2, cos(t) sin(t), sin(t)) has speed sqrt(1 + cos(t)**2), the length
    # Integral(sqrt(1 + cos(t)**2), (t, 0, 2 pi)) = 4 sqrt(2) E(1/2) = 7.6404...
    found = integrate_by_ranges(1, Eq(x**2 + y**2 + z**2, 1) & Eq(x**2 + y**2, x), measure='hausdorff')
    assert not isinstance(found, IntegralByRanges) and abs(float(N(found)) - 7.640395578055424) < 1e-12
    # three equations in three variables: a point, the counting measure
    assert integrate_by_ranges(x*y, Eq(x, 1) & Eq(y, 2) & Eq(z, 3), measure='hausdorff') == 2


def test_cubic_boundaries() -> None:
    # a bound which is a root of a cubic with a parametric coefficient:
    # Cardano's form for one real root, Viete's trigonometric form for three
    from sympy import CRootOf, N, Poly, cos, real_roots
    from sympy_extras.integrals.regions import _trigonometric_roots
    a = symbols('a', positive=True)
    area = integrate_by_ranges(1, (x > 0) & (x**3 + a*x < 1), [x])
    assert not isinstance(area, IntegralByRanges)
    assert abs(N(area.subs(a, 1)) - CRootOf(x**3 + x - 1, 0).evalf()) < 1e-12
    area = integrate_by_ranges(1, (x > 0) & (x < 1) & (x**3 - a*x + 1 > 0), [x], assumptions=[a > 2, a < 3])
    assert area.has(cos)
    assert abs(N(area.subs(a, Rational(5, 2))) - CRootOf(x**3 - Rational(5, 2)*x + 1, 1).evalf()) < 1e-12
    roots = _trigonometric_roots(Poly(x**3 - 3*x + 1, x), x)
    assert roots is not None
    assert sorted(float(N(r_)) for r_ in roots[:3]) == sorted(float(r_.evalf()) for r_ in real_roots(Poly(x**3 - 3*x + 1, x)))
    assert _trigonometric_roots(Poly(x**2 - 2, x), x) is None
    # a bound with no explicit form in the last variable, explicit in the
    # other: the region under the cubic y = x**3 - x cut by y < 1 with y
    # first, and x**3 + x*y < 1 (y = (1 - x**3)/x) with the integrand x
    found = integrate_by_ranges(1, (x > 0) & (y > 0) & (y < 1) & (x**3 - x - y < 0), [y, x])
    assert found.has(CRootOf)
    import mpmath
    expected = mpmath.quad(lambda yy: mpmath.findroot(lambda xx: xx**3 - xx - yy, 1.3), [0, 1])
    assert abs(float(N(found)) - float(expected)) < 1e-12
    found = integrate_by_ranges(x, (x > 0) & (y > 0) & (y < 1) & (x**3 + y*x < 1), [y, x])
    assert not isinstance(found, IntegralByRanges)
    expected = mpmath.quad(lambda yy: mpmath.findroot(lambda xx: xx**3 + yy*xx - 1, 0.7)**2/2, [0, 1])
    assert abs(float(N(found)) - float(expected)) < 1e-12
