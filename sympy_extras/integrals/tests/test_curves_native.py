"""Tests of the native Trager algorithm on a general curve
(:mod:`sympy_extras.integrals.curves`) on the objects of the core: the
field of the radicals, antiderivatives checked by differentiation, proofs
of nonelementary integrals, number fields of residues, and the
generator which has no expression."""
from __future__ import annotations

import random

from sympy_extras._numeric import reliable_complex
from sympy_extras.core import Dummy, Integer, Rational, Symbol
from sympy_extras.core.expr import Expr
from sympy_extras.core.function import diff
from sympy_extras.functions import Log, cbrt, log, sqrt
from sympy_extras.integrals.curves import (AlgebraicIntegrand, curve_antiderivative, curve_reduce,
                                           integrate_on_curve, is_nonelementary_on_curve)
from sympy_extras.polys.domains.algebraicfield import AlgebraicField

x = Symbol('x')


def _vanishes(e: Expr, low: int = 110, high: int = 900) -> None:
    """``e`` is zero at a few rational points where it is defined."""
    rng = random.Random(11)
    checked = 0
    for _ in range(40):
        value = reliable_complex(e, 30, {x: Rational(rng.randint(low, high), 100)})
        if value is None:
            continue
        assert abs(value) < 1e-20, (e, value)
        checked += 1
        if checked >= 4:
            return
    assert checked > 0, e


def test_the_field_of_the_radicals() -> None:
    A = AlgebraicIntegrand(1 / (sqrt(x + 1) + sqrt(x - 1)), x)
    y = A.y
    assert A.curve == y**4 - 4 * x * y**2 + 4
    assert A.field.n == 4 and A.algebra.dimension == 4
    _vanishes(A.polynomial_to_expr(A.element_expr()) - 1 / (sqrt(x + 1) + sqrt(x - 1)))
    # nested radicals, and a cube root with a square root
    assert AlgebraicIntegrand(1 / sqrt(x + sqrt(x)), x).field.n == 4
    assert [g.q for g in AlgebraicIntegrand(1 / sqrt(x + sqrt(x)), x).generators] == [2, 2]
    assert AlgebraicIntegrand(cbrt(x) / sqrt(x + 1), x).field.n == 6
    assert [g.q for g in AlgebraicIntegrand(x**Rational(1, 3) + x**Rational(2, 3) + sqrt(x), x).generators] == [6]
    a = Symbol('a')
    for f in [1 / x, sqrt(a * x), sqrt(x) * log(x)]:
        try:
            AlgebraicIntegrand(f, x)
        except ValueError:
            continue
        raise AssertionError(f)


def test_dependent_radicals_are_refused() -> None:
    # sqrt(x**2 - 1) is sqrt(x - 1) sqrt(x + 1): no primitive element of
    # degree eight exists, the characteristic polynomial is reducible
    try:
        AlgebraicIntegrand(sqrt(x - 1) + sqrt(x + 1) + sqrt(x**2 - 1), x)
    except ValueError:
        return
    raise AssertionError('dependent radicals accepted')


def test_derivatives_integrate_back() -> None:
    for F in [sqrt(x + 1) * sqrt(x - 1) + log(sqrt(x + 1) + sqrt(x - 1)),
              x * cbrt(x**2 + 1) + log(cbrt(x**2 + 1) + 1),
              log(sqrt(x + sqrt(x)) + sqrt(x)) + sqrt(x + sqrt(x)) / x]:
        f = diff(F, x)
        found = curve_antiderivative(f, x)
        assert found is not None, F
        _vanishes(diff(found, x) - f)


def test_nonelementary_integrals_are_proved() -> None:
    assert is_nonelementary_on_curve(1 / sqrt(x**3 + 1), x) is True
    assert is_nonelementary_on_curve(x**2 / sqrt(x**4 + 1), x) is True
    assert is_nonelementary_on_curve(x / cbrt(x**3 + 1), x) is True
    assert is_nonelementary_on_curve(1 / cbrt(x**3 + 1), x) is False
    assert is_nonelementary_on_curve(1 / x, x) is None


def test_the_reduction_returns_the_remainder() -> None:
    f = x / sqrt(1 - x**3)
    found = curve_reduce(f, x)
    assert found is not None
    F, h = found
    assert h != 0
    _vanishes(diff(F, x) + h - f, 10, 90)


def test_residues_in_a_number_field() -> None:
    # the residues +-sqrt(6)/4 at the roots of x**2 - 2
    f = sqrt(x**2 + 1) / (x**2 - 2)
    result = integrate_on_curve(f, x)
    assert result.nonelementary is False and result.residue_field.degree == 2
    _vanishes(diff(result.elementary_part(), x) - f, 150, 900)
    # residues in QQ(I): a complex logarithm
    f = 1 / ((x**2 + 1) * sqrt(x**2 + 2))
    F = curve_antiderivative(f, x)
    assert F is not None
    _vanishes(diff(F, x) - f)


def test_a_generator_without_expression_is_given_one() -> None:
    # Kauers' example 2: the residues generate a field whose generator has
    # no real root and no radicals; the native field names it by a symbol,
    # so the elementary part needs the expression of the generator
    f = sqrt(x**2 + 1) / (x**4 + 1)
    result = integrate_on_curve(f, x)
    K = result.residue_field
    assert isinstance(K, AlgebraicField) and result.nonelementary is False
    assert K.degree == 8 and K.alpha_expr.free_symbols
    assert curve_antiderivative(f, x) is None
    theta = Dummy('theta')
    F = result.elementary_part(theta)
    assert theta in F.free_symbols and F.count(Log) == len(result.logs) >= 1
    assert Integer(0) == result.remainder_expr(theta)
