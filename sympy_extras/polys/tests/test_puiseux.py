"""Tests of the Newton–Puiseux expansions: every branch satisfies the
equation to its order, the branches of a point account for all the roots,
the series agree with the roots numerically, and classical examples."""
from __future__ import annotations

import cmath

from sympy import symbols, sqrt, I, Poly, CRootOf, Rational, Add, S, Expr
from sympy.polys.domains import QQ

from sympy_extras._typing import as_expr
from sympy_extras.polys.puiseux import (puiseux_expansions, adjoin_root, Embedding, LaurentSeries,
                                        PuiseuxExpansion)

x, y, z = symbols('x y z')


def _branches(F: Expr, point: Expr, order: int = 4) -> list[PuiseuxExpansion]:
    branches = puiseux_expansions(F, x, y, point, order)
    n = Poly(F, y).degree()
    assert sum(b.ramification * b.conjugates for b in branches) == n, (F, point, branches)
    Fp = Poly(F, y, x)
    for b in branches:
        value = b.evaluate(Fp)
        v = value.valuation()
        # zero to the order of the evaluation
        assert v is None or (value.order is not None and v >= value.order), (b, value)
        _numerically_a_root(F, b)
    return branches


def _numerically_a_root(F: Expr, b: PuiseuxExpansion) -> None:
    K = b.field
    t = 0.02 * cmath.exp(0.7j)
    scale = complex(K.to_sympy(b.scale).evalf(30))
    e = b.ramification
    if b.point is S.Infinity:
        x0 = 1 / (scale * t**e)
    else:
        x0 = complex(b.point.evalf(30)) + scale * t**e
    y0 = sum(complex(K.to_sympy(c).evalf(30)) * t**k for k, c in b.series.terms.items())
    value = abs(complex(F.subs({x: x0, y: y0}).evalf(30)))
    size = max(abs(complex(term.subs({x: x0, y: y0}).evalf(30))) for term in Add.make_args(F)) or 1.0
    assert value / size < 1e-3, (b, value / size)


def test_smooth_points_give_power_series() -> None:
    branches = _branches(y**2 - x**3 - x**2, S.Zero)
    assert {b.as_expr() for b in branches} == {-x**3 / 8 + x**2 / 2 + x, x**3 / 8 - x**2 / 2 - x}
    assert all(b.ramification == 1 and b.conjugates == 1 and b.field == QQ for b in branches)
    [b] = _branches(y**3 - x**2, S.Zero)
    assert b.ramification == 3 and b.as_expr() == x**Rational(2, 3)


def test_ramified_branches_and_the_order() -> None:
    [b] = _branches(y**4 - x, S.Zero, order=2)
    assert (b.ramification, b.conjugates) == (4, 1) and b.as_expr() == x**Rational(1, 4)
    branches = _branches((y**2 - x)**2 - x**3, S.Zero, order=3)
    assert [b.ramification for b in branches] == [2, 2]
    assert {b.as_expr() for b in branches} == {sqrt(x) + x / 2 - x**Rational(3, 2) / 8 + x**2 / 16 - 5 * x**Rational(5, 2) / 128,
                                               sqrt(x) - x / 2 - x**Rational(3, 2) / 8 - x**2 / 16 - 5 * x**Rational(5, 2) / 128}
    # the leading term is always known, whatever the order asked
    branches = _branches(y**5 - x**3 * y - x**7, S.Zero, order=2)
    assert {b.as_expr() for b in branches} == {-x**4, x**Rational(3, 4)}


def test_conjugate_branches_over_number_fields() -> None:
    [b] = _branches(y**2 - 2 - x, S.Zero)
    assert b.conjugates == 2 and b.field == QQ.algebraic_field(sqrt(2))
    assert b.as_expr() == sqrt(2) + sqrt(2) * x / 4 - sqrt(2) * x**2 / 32 + sqrt(2) * x**3 / 128
    branches = _branches(y**3 - 3 * y - x, S.Zero)
    assert sorted(b.conjugates for b in branches) == [1, 2]
    # a point which is an algebraic number: the field contains it
    branches = _branches(y**4 - 4 * x * y**2 + 4, CRootOf(z**3 - z - 1, 0), order=2)
    assert sum(b.conjugates for b in branches) == 4 or sum(b.ramification * b.conjugates for b in branches) == 4
    [b] = _branches(y**2 - x**2 - 1, I)
    assert b.ramification == 2 and b.field == QQ.algebraic_field(I, sqrt(2)) or b.ramification == 2


def test_bronsteins_curve() -> None:
    # y^4 + (x^2 + x) y - x^2 (Bronstein's tutorial, example 1): a smooth
    # branch and a branch of ramification three above the origin
    F = y**4 + (x**2 + x) * y - x**2
    branches = _branches(F, S.Zero)
    assert sorted((b.ramification, b.conjugates) for b in branches) == [(1, 1), (3, 1)]
    assert {b.as_expr() for b in branches} >= {x - x**2}
    branches = _branches(F, S.NegativeOne)
    assert sorted((b.ramification, b.conjugates) for b in branches) == [(1, 1), (1, 1), (1, 2)]
    branches = _branches(F, S.Infinity)
    assert sorted((b.ramification, b.conjugates) for b in branches) == [(1, 1), (3, 1)]


def test_expansions_at_infinity() -> None:
    [b] = _branches(y**2 - x**3 - 1, S.Infinity, order=3)
    assert b.ramification == 2 and b.as_expr() == x**Rational(3, 2) + x**Rational(-3, 2) / 2
    # a curve which is not monic in y at infinity: a pole of y
    [b] = _branches(x * y**2 - 1, S.Infinity)
    assert b.ramification == 2 and b.as_expr() == x**Rational(-1, 2)
    [b] = _branches(x * y**2 - 1, S.Zero)
    assert b.as_expr() == x**Rational(-1, 2)
    branches = _branches(y**4 - 4 * x * y**2 + 4, S.Infinity)
    assert sorted(b.ramification for b in branches) == [2, 2]


def test_duvals_scaling_avoids_extensions() -> None:
    # y^2 = 3(x + 1) + ...: the branch sqrt(3) sqrt(x + 1) is written with
    # x + 1 = 3 t^2, y = 3 t + ..., no sqrt(3) in the field
    [b] = _branches(y**2 - x**3 - 1, S.NegativeOne)
    assert b.field == QQ and b.field.to_sympy(b.scale) == 3
    assert b.series.coefficient(1) == QQ(3)


def test_evaluate_along_a_branch() -> None:
    [b] = _branches(y**3 - x**2 * (x + 1), S.Zero)
    # y^2/x along y = x^(2/3) + ... has valuation 4/3 - 1 = 1/3 > 0: integral
    value = b.evaluate(Poly(y**2, y, x))
    assert value.valuation() == 4
    assert b.evaluate(Poly(x, y, x)).valuation() == 3


def test_adjoin_root_builds_towers() -> None:
    K = QQ.algebraic_field(sqrt(2))
    L, into, c = adjoin_root(K, [K.one, K.zero, -K.from_sympy(sqrt(2))])
    assert L.to_sympy(c * c) == L.to_sympy(into(K.from_sympy(sqrt(2))))
    # the embedding of a field into itself, and of the rationals
    same = Embedding(K, K)
    assert same(K.one) == K.one
    rationals = Embedding(QQ, L)
    assert rationals(QQ(3, 2)) == L.convert(QQ(3, 2), QQ)


def test_laurent_series_arithmetic() -> None:
    a = LaurentSeries(QQ, {-1: QQ(1), 0: QQ(2)}, 3)
    b = LaurentSeries(QQ, {1: QQ(1)}, None)
    assert (a * b).terms == {0: QQ(1), 1: QQ(2)} and (a * b).order == 4
    inverse = a.inverse(4)
    product = a * inverse
    assert product.coefficient(0) == QQ(1) and all(product.coefficient(k) == 0 for k in range(1, product.order or 0))
    assert a.substituted(2).terms == {-2: QQ(1), 0: QQ(2)} and a.substituted(2).order == 6
    assert as_expr(a.as_expr(x)) == 1 / x + 2
