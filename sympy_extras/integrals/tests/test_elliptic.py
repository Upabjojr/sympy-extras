"""Tests of the reduction to Legendre's elliptic integrals."""
from __future__ import annotations

from sympy import (symbols, sqrt, oo, S, pi, sin, elliptic_k, elliptic_e, elliptic_f, elliptic_pi, Rational,
                   simplify)
from sympy.testing.pytest import raises

from sympy_extras._testing import untyped
from sympy_extras._typing import ExprLike, as_expr
from sympy_extras.assumptions.ask import Assumptions
from sympy_extras.integrals.conditions import ConditionalValue
from sympy_extras.integrals.definite import verify_numerically
from sympy_extras.integrals.elliptic import (elliptic_integral, radicand, real_roots, legendre_reduction,
                                             Reduction)

x, theta = symbols('x theta')
k = symbols('k', positive=True)


def test_radicand() -> None:
    assert radicand(x / sqrt((1 - x**2) * (4 - x**2)), x) == (x, (1 - x**2) * (4 - x**2), -S.Half)
    # square roots of several factors are combined, half-integer powers split
    found = radicand(sqrt(x) * sqrt(1 - x) / sqrt(4 - x), x)
    assert found is not None and found[2] == S.Half and simplify(found[1] - x * (1 - x) * (4 - x)) == 0
    found = radicand(x**Rational(3, 2) / sqrt((1 - x) * (4 - x)), x)
    assert found is not None and found[0] == x**2 and found[2] == -S.Half
    # not elliptic: a quadratic, a quintic, a non-polynomial radicand
    assert radicand(1 / sqrt(1 - x**2), x) is None
    assert radicand(1 / sqrt(x**5 - 1), x) is None
    assert radicand(1 / sqrt(sin(x)), x) is None
    assert radicand(1 / (x**3 - x), x) is None


def test_real_roots_ordered() -> None:
    assert real_roots(x**3 - x, x) == [-1, 0, 1]
    assert real_roots((1 - x**2) * (1 - k**2 * x**2), x, k < 1) == [-1 / k, -1, 1, 1 / k]
    # complex roots, a double root, an undecided order
    assert real_roots(x**3 + 1, x) is None
    assert real_roots(x**2 * (x - 1), x) is None
    assert real_roots(x * (x - 1) * (x - k), x) is None


def test_legendre_reduction() -> None:
    s = symbols('s', positive=True)
    m = symbols('m', positive=True)
    assert legendre_reduction(Reduction(S.One, m, s, S.Zero, S.One)) == 2 * elliptic_k(m)
    value = legendre_reduction(Reduction(s, m, s, S.Zero, S.One))
    assert value is not None and simplify(value - 2 * (elliptic_k(m) - elliptic_e(m)) / m) == 0
    # the recurrence for s**2 (Byrd-Friedman 310.04)
    value = legendre_reduction(Reduction(s**2, m, s, S.Zero, S.One))
    assert value is not None
    assert simplify(value - 2 * ((2 + m) * elliptic_k(m) - 2 * (1 + m) * elliptic_e(m)) / (3 * m**2)) == 0
    # a simple pole outside the range is a third kind
    value = legendre_reduction(Reduction(1 / (1 + 2 * s), m, s, S.Zero, S.One))
    assert value == 2 * elliptic_pi(-2, m)
    # a pole inside the range, a double pole, a power in an incomplete integral
    assert legendre_reduction(Reduction(1 / (1 - 2 * s), m, s, S.Zero, S.One)) is None
    assert legendre_reduction(Reduction(1 / (1 + s)**2, m, s, S.Zero, S.One)) is None
    assert legendre_reduction(Reduction(s**2, m, s, S.Zero, S.Half)) is None
    # incomplete
    value = legendre_reduction(Reduction(S.One, m, s, S.Zero, S.Half))
    assert value == 2 * elliptic_f(pi / 4, m)


def _check(f: ExprLike, a: ExprLike, b: ExprLike, expected: ExprLike,
           assumptions: Assumptions = None) -> ConditionalValue:
    f_, a_, b_, expected_ = as_expr(f), as_expr(a), as_expr(b), as_expr(expected)
    found = elliptic_integral(f_, x, a_, b_, assumptions)
    assert found is not None, f
    assert simplify(found.value - expected_) == 0, (found.value, expected)
    if not expected_.free_symbols:
        assert verify_numerically(found.value, f_, x, a_, b_) is True, f
    return found


def test_cubic_cells() -> None:
    # between consecutive roots, and beyond the extreme ones
    _check(1 / sqrt(x * (1 - x) * (4 - x)), 0, 1, elliptic_k(Rational(1, 4)))
    _check(1 / sqrt(x * (1 - x) * (1 - k**2 * x)), 0, 1, 2 * elliptic_k(k**2), k < 1)
    _check(x / sqrt(x * (1 - x) * (1 - k**2 * x)), 0, 1, 2 * (elliptic_k(k**2) - elliptic_e(k**2)) / k**2, k < 1)
    found = elliptic_integral(1 / sqrt(x**3 - x), x, 1, oo)
    assert found is not None and verify_numerically(found.value, 1 / sqrt(x**3 - x), x, S.One, oo) is True
    found = elliptic_integral(1 / sqrt(x - x**3), x, -oo, -1)
    assert found is not None and verify_numerically(found.value, 1 / sqrt(x - x**3), x, -oo, -S.One) is True
    # a range crossing a root, a range on which P is negative
    assert elliptic_integral(1 / sqrt(x * (1 - x) * (4 - x)), x, 0, 2) is None
    assert elliptic_integral(1 / sqrt(x * (1 - x) * (4 - x)), x, 1, 4) is None


def test_incomplete_and_third_kind() -> None:
    _check(1 / sqrt(x * (1 - x) * (4 - x)), 0, S.Half, elliptic_f(pi / 4, Rational(1, 4)))
    _check(1 / ((x + 2) * sqrt(x * (1 - x) * (4 - x))), 0, 1, elliptic_pi(-S.Half, Rational(1, 4)) / 2)
    found = _check(sqrt(x * (1 - x) * (4 - x)), 0, 1, -28 * elliptic_k(Rational(1, 4)) / 5 + 104 * elliptic_e(Rational(1, 4)) / 15)
    assert found.value.has(elliptic_e)


def test_quartic_cells() -> None:
    # four real roots, between the middle ones (Byrd-Friedman 254.00)
    _check(1 / sqrt(x * (x - 1) * (2 - x) * (3 - x)), 1, 2, elliptic_k(Rational(3, 4)))
    _check(1 / sqrt((1 - x) * (x - 2) * (x - 3) * (x - 4)), 1, 2, elliptic_k(Rational(1, 4)))
    # beyond the extreme root, through u = 1/(x - r): an incomplete integral
    found = elliptic_integral(1 / sqrt(x * (x - 1) * (x - 2) * (x - 3)), x, 3, oo)
    assert found is not None and verify_numerically(found.value, 1 / sqrt(x * (x - 1) * (x - 2) * (x - 3)), x, S(3), oo) is True


def test_even_radicands() -> None:
    # y = x**2 turns the quartic into a cubic
    _check(1 / sqrt((1 - x**2) * (1 - k**2 * x**2)), 0, 1, elliptic_k(k**2), k < 1)
    _check(x**2 / sqrt((1 - x**2) * (1 - k**2 * x**2)), 0, 1, (elliptic_k(k**2) - elliptic_e(k**2)) / k**2, k < 1)
    found = elliptic_integral(1 / sqrt(1 - x**4), x, 0, 1)
    assert found is not None and verify_numerically(found.value, 1 / sqrt(1 - x**4), x, S.Zero, S.One) is True
    found = elliptic_integral(sqrt(1 - x**4), x, 0, 1)
    assert found is not None and verify_numerically(found.value, sqrt(1 - x**4), x, S.Zero, S.One) is True
    # an odd R keeps the roots of the quartic, which are complex here
    assert elliptic_integral(x / sqrt(1 - x**4), x, 0, 1) is None


def test_trigonometric_forms() -> None:
    m = k**2
    found = elliptic_integral(sqrt(1 - m * sin(theta)**2), theta, 0, pi / 2, k < 1)
    assert found == ConditionalValue(elliptic_e(m))
    found = elliptic_integral(1 / sqrt(1 - m * sin(theta)**2), theta, 0, pi / 2, k < 1)
    assert found == ConditionalValue(elliptic_k(m))
    found = elliptic_integral(sin(theta)**2 / sqrt(1 - m * sin(theta)**2), theta, 0, pi / 2, k < 1)
    assert found is not None and simplify(found.value - (elliptic_k(m) - elliptic_e(m)) / m) == 0
    found = elliptic_integral(1 / sqrt(1 - m * sin(theta)**2), theta, 0, pi / 3, k < 1)
    assert found == ConditionalValue(elliptic_f(pi / 3, m))
    # an odd power of sin is not of the form
    assert elliptic_integral(sin(theta) / sqrt(1 - m * sin(theta)**2), theta, 0, pi / 2, k < 1) is None


def test_wrong_types_are_rejected() -> None:
    raises(TypeError, lambda: untyped(elliptic_integral)([1], x, 0, 1))
