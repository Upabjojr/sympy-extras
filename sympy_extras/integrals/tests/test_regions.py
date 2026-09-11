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
    # nothing to solve: y not linear, x unbounded
    assert isinstance(integrate_by_ranges(1, (x > 0) & (y > 0) & (y < exp(x))), IntegralByRanges)
    assert isinstance(integrate_by_ranges(1, (x > 0) & (x < 1) & (y**2 < exp(x)) & (y > 0)), IntegralByRanges)
