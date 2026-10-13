"""Tests of Trager's algorithm on a general curve: the parsing of the
radicals into a function field, derivatives of elementary expressions
integrated back, proofs of nonelementary integrals, and the fallback
from the quadratic Trager module."""
from __future__ import annotations

import random

from sympy import symbols, sqrt, cbrt, log, Rational, N, Poly, cancel, S, exp

from sympy_extras._typing import as_expr
from sympy_extras.integrals.curves import (AlgebraicIntegrand, curve_antiderivative, curve_reduce,
                                           is_nonelementary_on_curve, integrate_on_curve)

x, a = symbols('x a')


def _derivative_matches(F: object, f: object, low: int = 110, high: int = 900) -> None:
    F_, f_ = as_expr(F), as_expr(f)
    difference = F_.diff(x) - f_
    rng = random.Random(11)
    checked = 0
    for _ in range(40):
        x0 = Rational(rng.randint(low, high), 100)
        value = N(difference.subs(x, x0), 30)
        if not value.is_number or value.has(S.NaN):
            continue
        assert abs(complex(value)) < 1e-20, (F_, f_, x0, value)
        checked += 1
        if checked >= 5:
            return
    assert checked > 0, (F_, f_)


def test_the_field_of_the_radicals() -> None:
    A = AlgebraicIntegrand(1 / (sqrt(x + 1) + sqrt(x - 1)), x)
    assert A.curve == A.y**4 - 4 * x * A.y**2 + 4
    assert A.field.n == 4 and A.algebra.dimension == 4
    # the integrand as an element of the field, and back
    assert cancel(A.polynomial_to_expr(A.element.as_expr()) - 1 / (sqrt(x + 1) + sqrt(x - 1))) == 0
    # nested radicals
    A = AlgebraicIntegrand(1 / sqrt(x + sqrt(x)), x)
    assert Poly(A.curve, A.y).degree() == 4
    assert [g.q for g in A.generators] == [2, 2]
    # a cube root together with a square root: degree six
    A = AlgebraicIntegrand(cbrt(x) / sqrt(x + 1), x)
    assert Poly(A.curve, A.y).degree() == 6
    # mixed exponents of one base are powers of one generator
    A = AlgebraicIntegrand(x**Rational(1, 3) + x**Rational(2, 3) + x**Rational(1, 2), x)
    assert [g.q for g in A.generators] == [6]
    # not an algebraic function, or rational: refused
    for f in [1 / x, sqrt(x) * exp(x), sqrt(a * x)]:
        try:
            AlgebraicIntegrand(f, x)
        except ValueError:
            continue
        raise AssertionError(f)


def test_derivatives_of_elementary_expressions_integrate_back() -> None:
    expressions = [
        sqrt(x + 1) * sqrt(x - 1) + log(sqrt(x + 1) + sqrt(x - 1)),
        x * cbrt(x**2 + 1) + log(cbrt(x**2 + 1) + 1),
        log(sqrt(x + sqrt(x)) + sqrt(x)) + sqrt(x + sqrt(x)) / x,
        (x**2 + 1) / sqrt(x**3 + 2) + 2 * log(1 + sqrt(x)),
        log(x + sqrt(x**2 + 1)) - sqrt(x**2 + 1) / x + log(sqrt(x**2 + 1) + sqrt(x**2 + 2)),
    ]
    for F in expressions:
        f = as_expr(cancel(F.diff(x)))
        found = curve_antiderivative(f, x)
        assert found is not None, F
        _derivative_matches(found, f)


def test_nonelementary_integrals_are_proved() -> None:
    # differentials of the first kind on curves of genus one and two
    assert is_nonelementary_on_curve(1 / sqrt(x**3 + 1), x) is True
    assert is_nonelementary_on_curve(x / sqrt(1 - x**3), x) is True
    assert is_nonelementary_on_curve(1 / sqrt(x**5 + 1), x) is True
    # a pole of order two at infinity after the Hermite reduction
    assert is_nonelementary_on_curve(x**2 / sqrt(x**4 + 1), x) is True
    # cube roots: Chebyshev's criterion for x^m (1 + x^3)^(-1/3) makes m = 0
    # elementary ((m + 1)/n + p = 0) and m = 1 not
    assert is_nonelementary_on_curve(x / cbrt(x**3 + 1), x) is True
    assert is_nonelementary_on_curve(1 / cbrt(x**3 + 1), x) is False
    assert is_nonelementary_on_curve(cbrt(x) / (x + 1), x) is False


def test_the_remainder_of_the_reduction() -> None:
    found = curve_reduce((x**3 + 1) / ((x - 1)**2 * sqrt(x**3 + 1)), x)
    assert found is not None
    F, h = found
    _derivative_matches(F, (x**3 + 1) / ((x - 1)**2 * sqrt(x**3 + 1)) - h)
    result = integrate_on_curve(1 / sqrt(x**3 + 1), x)
    assert result.nonelementary is True and result.elementary_part() == 0


def test_fallback_from_the_quadratic_module() -> None:
    from sympy_extras.integrals.trager import trager_antiderivative, is_nonelementary_algebraic
    f = 1 / (sqrt(x + 1) + sqrt(x - 1))
    F = trager_antiderivative(f, x)
    assert F is not None
    _derivative_matches(F, f)
    assert is_nonelementary_algebraic(1 / (x * (sqrt(x + 1) + sqrt(x - 1))), x) is False
