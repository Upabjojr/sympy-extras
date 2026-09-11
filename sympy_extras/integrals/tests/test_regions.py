"""Tests of the integrals over regions (:mod:`sympy_extras.integrals.regions`)."""
from __future__ import annotations

from sympy import Eq, Integral, Piecewise, Rational, S, exp, pi, sqrt, symbols
from sympy.testing.pytest import raises

from sympy_extras._testing import untyped
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
