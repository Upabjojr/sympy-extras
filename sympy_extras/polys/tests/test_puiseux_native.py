"""Tests of the native Newton–Puiseux expansions
(:mod:`sympy_extras.polys.puiseux`) on the objects of the core: every
branch satisfies the equation to its order, the branches of a point
account for all the roots, the series agree with the roots numerically,
classical examples, and the exact number fields (no SymPy)."""
from __future__ import annotations

import cmath
from fractions import Fraction

from sympy_extras.core import I, Integer, Rational, Symbol, oo, sqrt
from sympy_extras.core.expr import Expr
from sympy_extras.polys.cad.indexedroot import IndexedRoot, bound_variable
from sympy_extras.polys.puiseux import (RATIONALS, Embedding, LaurentSeries, PuiseuxExpansion, adjoin_root,
                                        curve_from_expr, field_from_polynomial, field_of, in_subfield,
                                        minimal_polynomial_of, puiseux_expansions)

x, y, z = Symbol('x'), Symbol('y'), Symbol('z')


def _q(*values: int) -> list[Fraction]:
    return [Fraction(v) for v in values]


def _complex(e: Expr) -> complex:
    return complex(e.evalf(30))


def _branches(F: Expr, point: Expr, order: int = 4) -> list[PuiseuxExpansion]:
    branches = puiseux_expansions(F, x, y, point, order)
    n = max(i for i in curve_from_expr(F, x, y, RATIONALS))
    assert sum(b.ramification * b.conjugates for b in branches) == n, (F, point, branches)
    curve = curve_from_expr(F, x, y, RATIONALS)
    for b in branches:
        value = b.evaluate(curve)
        v = value.valuation()
        # zero to the order of the evaluation
        assert v is None or (value.order is not None and v >= value.order), (b, value)
        _numerically_a_root(F, b)
    return branches


def _numerically_a_root(F: Expr, b: PuiseuxExpansion) -> None:
    K = b.field
    t = 0.02 * cmath.exp(0.7j)
    scale = _complex(K.to_expr(b.scale))
    e = b.ramification
    if b.at_infinity:
        x0 = 1 / (scale * t**e)
    else:
        x0 = _complex(b.point) + scale * t**e
    y0 = sum(_complex(K.to_expr(c)) * t**k for k, c in b.series.terms.items())
    terms = curve_from_expr(F, x, y, RATIONALS)
    value = 0j
    size = 0.0
    for i, row in terms.items():
        for k, c in row.items():
            term = float(c[0]) * x0**k * y0**i
            value += term
            size = max(size, abs(term))
    assert abs(value) / (size or 1.0) < 1e-3, (b, abs(value) / size)


def test_smooth_points_give_power_series() -> None:
    branches = _branches(y**2 - x**3 - x**2, Integer(0))
    assert {b.as_expr() for b in branches} == {-x**3 / 8 + x**2 / 2 + x, x**3 / 8 - x**2 / 2 - x}
    assert all(b.ramification == 1 and b.conjugates == 1 and b.field == RATIONALS for b in branches)
    [b] = _branches(y**3 - x**2, Integer(0))
    assert b.ramification == 3 and b.as_expr() == x**Rational(2, 3)


def test_ramified_branches_and_the_order() -> None:
    [b] = _branches(y**4 - x, Integer(0), order=2)
    assert (b.ramification, b.conjugates) == (4, 1) and b.as_expr() == x**Rational(1, 4)
    branches = _branches((y**2 - x)**2 - x**3, Integer(0), order=3)
    assert [b.ramification for b in branches] == [2, 2]
    # the leading term is always known, whatever the order asked
    branches = _branches(y**5 - x**3 * y - x**7, Integer(0), order=2)
    assert {b.as_expr() for b in branches} == {-x**4, x**Rational(3, 4)}


def test_conjugate_branches_over_number_fields() -> None:
    [b] = _branches(y**2 - 2 - x, Integer(0))
    assert b.conjugates == 2 and b.field.minpoly == _q(1, 0, -2) and b.field.alpha_expr == sqrt(Integer(2))
    assert b.as_expr() == sqrt(Integer(2)) + sqrt(Integer(2)) * x / 4 - sqrt(Integer(2)) * x**2 / 32 \
        + sqrt(Integer(2)) * x**3 / 128
    branches = _branches(y**3 - 3 * y - x, Integer(0))
    assert sorted(b.conjugates for b in branches) == [1, 2]
    # a point which is an algebraic number (a real root): the field contains it
    t = bound_variable('z')
    point = IndexedRoot(t**3 - t - 1, t, 0)
    branches = _branches(y**4 - 4 * x * y**2 + 4, point, order=2)
    assert sum(b.ramification * b.conjugates for b in branches) == 4
    assert all(b.field.degree % 3 == 0 for b in branches)
    [b] = _branches(y**2 - x**2 - 1, I)
    assert b.ramification == 2


def test_bronsteins_curve() -> None:
    # y^4 + (x^2 + x) y - x^2 (Bronstein's tutorial, example 1): a smooth
    # branch and a branch of ramification three above the origin
    F = y**4 + (x**2 + x) * y - x**2
    branches = _branches(F, Integer(0))
    assert sorted((b.ramification, b.conjugates) for b in branches) == [(1, 1), (3, 1)]
    assert x - x**2 in {b.as_expr() for b in branches}
    branches = _branches(F, Integer(-1))
    assert sorted((b.ramification, b.conjugates) for b in branches) == [(1, 1), (1, 1), (1, 2)]
    branches = _branches(F, oo)
    assert sorted((b.ramification, b.conjugates) for b in branches) == [(1, 1), (3, 1)]


def test_expansions_at_infinity() -> None:
    [b] = _branches(y**2 - x**3 - 1, oo, order=3)
    assert b.ramification == 2 and b.as_expr() == x**Rational(3, 2) + x**Rational(-3, 2) / 2
    # a curve which is not monic in y at infinity: a pole of y
    [b] = _branches(x * y**2 - 1, oo)
    assert b.ramification == 2 and b.as_expr() == x**Rational(-1, 2)
    [b] = _branches(x * y**2 - 1, Integer(0))
    assert b.as_expr() == x**Rational(-1, 2)
    branches = _branches(y**4 - 4 * x * y**2 + 4, oo)
    assert sorted(b.ramification for b in branches) == [2, 2]


def test_duvals_scaling_avoids_extensions() -> None:
    # y^2 = 3(x + 1) + ...: the branch sqrt(3) sqrt(x + 1) is written with
    # x + 1 = 3 t^2, y = 3 t + ..., no sqrt(3) in the field
    [b] = _branches(y**2 - x**3 - 1, Integer(-1))
    assert b.field == RATIONALS and b.scale == _q(3)
    assert b.series.coefficient(1) == _q(3)


def test_evaluate_along_a_branch() -> None:
    [b] = _branches(y**3 - x**2 * (x + 1), Integer(0))
    # y^2/x along y = x^(2/3) + ... has valuation 4/3 - 1 = 1/3 > 0: integral
    assert b.evaluate_expr(y**2).valuation() == 4
    assert b.evaluate_expr(x).valuation() == 3


def test_adjoin_root_builds_towers() -> None:
    K = field_from_polynomial(_q(1, 0, -2))
    L, into, u = adjoin_root(K, [K.one, K.zero, K.neg(K.alpha)])
    assert L.degree == 4 and L.mul(u, u) == into(K.alpha)
    # the minimal polynomial of u, the fourth root of 2, and the subfield
    assert minimal_polynomial_of(u, L) == _q(1, 0, 0, 0, -2)
    assert in_subfield(into(K.alpha), L, into) == K.alpha and in_subfield(u, L, into) is None
    # the embedding of a field into itself, and of the rationals
    assert Embedding(K, K)(K.one) == K.one
    assert Embedding(RATIONALS, L)([Fraction(3, 2)]) == [Fraction(3, 2)]
    # a linear factor gives the field itself
    same, identity, root = adjoin_root(K, [K.one, K.neg(K.alpha)])
    assert same == K and root == K.alpha and identity(K.alpha) == K.alpha


def test_fields_of_points_and_generators() -> None:
    K, a = field_of(sqrt(Integer(3)))
    assert K.minpoly == _q(1, 0, -3) and a == K.alpha
    Q, b = field_of(Rational(-2, 3))
    assert Q == RATIONALS and b == [Fraction(-2, 3)]
    t = bound_variable('t')
    R, c = field_of(IndexedRoot(t**3 - t - 1, t, 0))
    assert R.minpoly == _q(1, 0, -1, -1)
    # the generator of a new field: its largest real root, in radicals
    # when the polynomial is quadratic or binomial
    assert field_from_polynomial(_q(1, 0, -2)).alpha_expr == sqrt(Integer(2))
    assert field_from_polynomial(_q(1, 0, 1)).alpha_expr == -I
    assert isinstance(field_from_polynomial(_q(1, 0, -1, -1)).alpha_expr, IndexedRoot)
    # the same polynomial gives the same field
    assert field_from_polynomial(_q(2, 0, -4)) is field_from_polynomial(_q(1, 0, -2))


def test_laurent_series_arithmetic() -> None:
    a = LaurentSeries(RATIONALS, {-1: _q(1), 0: _q(2)}, 3)
    b = LaurentSeries(RATIONALS, {1: _q(1)}, None)
    assert (a * b).terms == {0: _q(1), 1: _q(2)} and (a * b).order == 4
    product = a * a.inverse(4)
    assert product.coefficient(0) == _q(1) and all(not product.coefficient(k) for k in range(1, product.order or 0))
    assert a.substituted(2).terms == {-2: _q(1), 0: _q(2)} and a.substituted(2).order == 6
    assert a.as_expr(x) == 1 / x + 2
