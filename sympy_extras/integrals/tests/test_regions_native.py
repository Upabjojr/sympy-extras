"""Tests of the native integrals over regions
(:mod:`sympy_extras.integrals.regions`), on core objects: the node, the
private steps, and the decomposition run with an integrator written on
the core."""
from __future__ import annotations

from fractions import Fraction
from typing import Optional

from sympy_extras._testing import untyped
from sympy_extras.assumptions.facts import normalize
from sympy_extras.core import Integer, Rational, Symbol, eq, expand, ne, oo, pi, sqrt
from sympy_extras.core.expr import Expr
from sympy_extras.core.logic import Boolean, and_, true
from sympy_extras.functions import exp, gamma, heaviside, log
from sympy_extras.integrals.regions import (IntegralByRanges, _bounded, _coefficients, _conditional, _decomposed,
                                            _full_measure, _polytope, _radical_form, _rank, _reality_conditions,
                                            _solve_rational, _split_roots, _trigonometric_roots, integrate_by_ranges)
from sympy_extras.polys.cad.indexedroot import IndexedRoot, indexed_root
from sympy_extras.polys.polytools import Poly
from sympy.testing.pytest import raises

x, y, z, r, u, a = (Symbol(s) for s in 'xyzrua')


def _polynomial(f: Expr, v: Symbol, lower: Expr, upper: Expr, assumptions: list[Boolean]) -> Optional[Expr]:
    """An integrator written on the core: a polynomial in ``v`` by its
    antiderivative over a finite range; ``None`` otherwise."""
    terms = _coefficients(f, [v])
    if lower == -oo or upper == oo or terms is None:
        return None
    F: Expr = Integer(0)
    for (k,), c in terms.items():
        F = F + c * v**(k + 1) / (k + 1)
    return expand(F.subs(v, upper) - F.subs(v, lower))


def test_node() -> None:
    node = IntegralByRanges(x * y, (x > 0) & (x < y) & (y < 1))
    assert node.integrand == x * y and node.variables == (x, y) and node.measure == 'lebesgue'
    assert node.dimension is None and node.free_symbols == set()
    # the constructor stores, doit computes
    assert isinstance(node, IntegralByRanges) and node.doit() == Rational(1, 8)
    # the variables are bound, the parameters are not
    disc = IntegralByRanges(Integer(1), x**2 + y**2 < a**2, parameters=[a])
    assert disc.variables == (x, y) and disc.free_symbols == {a}
    assert disc.subs(x, 2) == disc and disc.subs(a, 2) == IntegralByRanges(Integer(1), x**2 + y**2 < 4, [x, y])
    # the measure and the dimension are arguments: rebuilt from them
    circle = IntegralByRanges(Integer(1), eq(x**2 + y**2, 1), measure='hausdorff')
    assert circle.func(*circle.args) == circle and circle != IntegralByRanges(Integer(1), eq(x**2 + y**2, 1))
    ball = IntegralByRanges(Integer(1), r < 1, [r], dimension=Symbol('n'))
    assert ball.func(*ball.args) == ball and ball.dimension == Symbol('n')
    raises(ValueError, lambda: IntegralByRanges(Integer(1), true))
    raises(ValueError, lambda: IntegralByRanges(Integer(1), x < 1, measure='counting'))
    raises(ValueError, lambda: IntegralByRanges(Integer(1), x < a, [x], parameters=[x]))
    raises(TypeError, lambda: untyped(IntegralByRanges)(Integer(1), x**2 + y**2 < 1, [x, 2]))


def test_core_integrator() -> None:
    # the decomposition and the iterated integrals with an integrator of
    # the core: the region between y = x**2 and y = x, Integral(x - x**2, (x, 0, 1)) = 1/6
    assert integrate_by_ranges(Integer(1), (x > 0) & (x < 1) & (y > x**2) & (y < x),
                               integrator=_polynomial) == Rational(1, 6)
    # Integral(x**2 - x, (x, 1, 2)) = 5/6
    assert integrate_by_ranges(Integer(1), (0 < x) & (x < y) & (y < x**2) & (x < 2),
                               integrator=_polynomial) == Rational(5, 6)
    # the union of two disjoint rectangles of areas 1 and 2
    squares = ((0 < x) & (x < 1) & (0 < y) & (y < 1)) | ((2 < x) & (x < 3) & (0 < y) & (y < 2))
    assert integrate_by_ranges(x * y, squares, integrator=_polynomial) == Rational(1, 4) + 5
    # the disc in polar coordinates: Integral(2*pi*rho, (rho, 0, 1)) = pi
    assert integrate_by_ranges(Integer(1), x**2 + y**2 < 1, integrator=_polynomial) == pi
    # a parameter: the cases of its cells
    found = integrate_by_ranges(Integer(1), (x > 0) & (x < a), [x], integrator=_polynomial)
    assert [found.subs(a, v) for v in (-1, 0, 2)] == [0, 0, 2]
    # an integral not computed leaves the node
    node = integrate_by_ranges(exp(x), (x > 0) & (x < 1) & (y > 0) & (y < 1), integrator=_polynomial)
    assert isinstance(node, IntegralByRanges)


def test_contradictory_bounds_are_an_empty_cell() -> None:
    def contradiction(f: Expr, v: Symbol, lower: Expr, upper: Expr, assumptions: list[Boolean]) -> Optional[Expr]:
        raise ValueError("inconsistent assumptions")
    assert integrate_by_ranges(Integer(1), (0 < x) & (x < y) & (y < x**2) & (x < 2), integrator=contradiction) == 0


def test_default_integrator() -> None:
    # the definite integrals of SymPy's side
    assert integrate_by_ranges(exp(-x**2 - y**2), x**2 + y**2 < 1) == -pi * exp(-1) + pi
    assert integrate_by_ranges(Integer(1), (x > 1) & (x < 2) & (y > 0) & (y < log(x))) == -1 + log(4)
    assert integrate_by_ranges(heaviside(x), x**2 + y**2 < 1) == pi / 2
    assert integrate_by_ranges(Integer(1), eq(x**2 + y**2, 1), measure='hausdorff') == 2 * pi
    n = Symbol('n', positive=True, integer=True)
    assert integrate_by_ranges(Integer(1), r < 1, [r], dimension=n) == pi**(n / 2) / gamma(n / 2 + 1)
    # a numeric bound is a root object of the core
    found = integrate_by_ranges(Integer(1), (x > 0) & (y > 0) & (y < 1) & (x**3 - x - y < 0), [y, x])
    assert found.has(IndexedRoot)
    assert abs(float(found.evalf(20)) - 1.1822578844952327) < 1e-12


def test_coefficients() -> None:
    assert _coefficients(x**2 * a + 2 * x * y + exp(a), [x, y]) == {(2, 0): a, (1, 1): Integer(2), (0, 0): exp(a)}
    assert _coefficients(y - exp(x), [y]) == {(1,): Integer(1), (0,): -exp(x)}
    assert _coefficients(sqrt(x), [x]) is None and _coefficients(1 / x, [x]) is None
    assert _coefficients(exp(x), [x]) is None


def test_polytope_agrees_with_the_decomposition() -> None:
    pentagon = (x > 0) & (y > 0) & (x + y < 3) & (x < 2) & (y < 2)
    cube = (x > -1) & (x < 1) & (y > -1) & (y < 1) & (z > -1) & (z < 1)
    for f, condition in [(Integer(1), pentagon), (x * y, pentagon), (x**2 + y**3, pentagon),
                         (x * y * z, cube & (x + y + z < 1))]:
        names = sorted(condition.free_symbols, key=lambda s: s.name)
        found = _polytope(f, normalize(condition), names, [])
        expected = _decomposed(f, normalize(condition), names, [], _polynomial)
        assert found is not None and found == expected, (f, condition, found, expected)
    c = Symbol('c')
    assert _polytope(Integer(1), normalize((x > 0) & (x < c) & (y > 0) & (y < 1)), [x, y], []) is None
    assert _polytope(Integer(1), normalize((x > 0) & (x < c) & (y > 0) & (y < 1)), [x, y], [c > 0]) == c
    assert _polytope(Integer(1), normalize((x > 0) & (y > 0) & (y < 1)), [x, y], []) is None
    assert _polytope(exp(x), normalize((x > 0) & (x < 1) & (y > 0) & (y < 1)), [x, y], []) is None


def test_linear_algebra() -> None:
    # a quadrant is unbounded, a triangle bounded
    F = Fraction
    assert _bounded([[F(-1), F(0)], [F(0), F(-1)]]) is False
    assert _bounded([[F(-1), F(0)], [F(0), F(-1)], [F(1), F(1)]]) is True
    assert _solve_rational([[F(1), F(1)], [F(1), F(-1)]], [a, Integer(1)]) == [a / 2 + Rational(1, 2),
                                                                            a / 2 - Rational(1, 2)]
    assert _solve_rational([[F(1), F(1)], [F(2), F(2)]], [a, Integer(1)]) is None
    # the generic rank of a matrix of polynomials in a parameter
    assert _rank([[a, Integer(1)], [a**2, a]]) == 1 and _rank([[a, Integer(1)], [Integer(1), a]]) == 2


def test_roots_and_bounds() -> None:
    assert _radical_form(indexed_root(4 * x**2 - 3, x, 1)) == sqrt(3) / 2
    assert _radical_form(indexed_root(x**2 + x - 1, x, 1)) == (sqrt(5) - 1) / 2
    # a quintic has no radicals, and a cubic with three real roots keeps
    # its root object rather than a form with I
    quintic = indexed_root(x**5 - x - 1, x, 0)
    assert _radical_form(quintic) == quintic and isinstance(_radical_form(indexed_root(x**3 - 3 * x + 1, x, 0)),
                                                            IndexedRoot)
    roots = _trigonometric_roots(Poly(x**3 - 3 * x + 1, x), x)
    assert roots is not None
    values = sorted(float(e.evalf(20)) for e in roots[:3])
    assert all(abs(v - w) < 1e-12 for v, w in zip(values, [-1.879385241571817, 0.3472963553338607, 1.532088886237956]))
    assert _trigonometric_roots(Poly(x**2 - 2, x), x) is None


def test_private_steps() -> None:
    facts: list[Boolean] = [x > -1, x < 1, u > -1, u < 1]
    assert _split_roots(sqrt(1 - x**2 - (1 - x**2) * u**2), facts) == \
        sqrt(1 - x) * sqrt(x + 1) * sqrt(1 - u) * sqrt(u + 1)
    assert _split_roots(sqrt(1 - x**2 - (1 - x**2) * u**2), []) == sqrt(1 - x**2 - (1 - x**2) * u**2)
    assert _reality_conditions(log(x + 1) - 1) == [x + 1 > 0]
    assert set(_reality_conditions(x**Rational(1, 3) + sqrt(x - 1)) or []) == {x >= 0, x - 1 >= 0}
    assert _reality_conditions(x**y) is None
    assert _full_measure(and_(x > 0, ne(x * y - 1, 0), eq(a, 1)), [x, y]) == and_(x > 0, eq(a, 1))
    assert _conditional(pi * a**2, a**2, []) == pi * a**2
    assert _conditional(a, a, []) != a
