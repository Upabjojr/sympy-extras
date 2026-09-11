"""Tests of the reduction to Legendre's elliptic integrals."""
from __future__ import annotations

import random

from sympy import (symbols, sqrt, oo, S, pi, sin, elliptic_k, elliptic_e, elliptic_f, elliptic_pi, Rational,
                   simplify, Expr, log, asin)
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
    # a pole inside the range, at the lower end
    assert legendre_reduction(Reduction(1 / (1 - 2 * s), m, s, S.Zero, S.One)) is None
    assert legendre_reduction(Reduction(1 / s, m, s, S.Zero, S.One)) is None
    assert legendre_reduction(Reduction(1 / (1 - 2 * s), m, s, S.Half, S.One)) is None
    # incomplete
    value = legendre_reduction(Reduction(S.One, m, s, S.Zero, S.Half))
    assert value == 2 * elliptic_f(pi / 4, m)


def _check_reduction(Q: Expr, lower: Expr, upper: Expr) -> Expr:
    """The Legendre form checked against the quadrature at random
    parameters ``0 < m < 1``."""
    s = symbols('s', positive=True)
    m = symbols('m', positive=True)
    value = legendre_reduction(Reduction(Q, m, s, lower, upper), m < 1)
    assert value is not None, Q
    assert verify_numerically(value, Q / sqrt(s * (1 - s) * (1 - m * s)), s, lower, upper, m < 1, samples=3) is True, Q
    return value


def test_incomplete_powers() -> None:
    # Byrd-Friedman 310-318 with the boundary terms: powers above the first
    # in incomplete integrals, and the negative powers away from 0
    s = symbols('s', positive=True)
    m = symbols('m', positive=True)
    for Q in (s**2, s**3, 3 * s**4 - s**2 + 1):
        value = _check_reduction(Q, S.Zero, Rational(1, 3))
        assert value.has(elliptic_f) and value.has(elliptic_e)
    for Q in (1 / s, 1 / s**2, (s + 1) / s**3):
        _check_reduction(Q, Rational(1, 4), Rational(2, 3))
        _check_reduction(Q, Rational(1, 4), S.One)
    # the differences in the formula for the same regularisation: the
    # regularised value at 0 vanishes, the integral from 0 diverges
    assert legendre_reduction(Reduction(1 / s, m, s, S.Zero, S.Half)) is None


def test_multiple_poles() -> None:
    # reduced through d/ds (sqrt(W)/D**(k-1)): outside the range, and
    # divergent inside it
    s = symbols('s', positive=True)
    m = symbols('m', positive=True)
    for Q in (1 / (1 + s)**2, 1 / (1 + 2 * s)**3, s / (2 + s)**2):
        value = _check_reduction(Q, S.Zero, S.One)
        assert value.has(elliptic_pi)
        _check_reduction(Q, S.Zero, Rational(1, 3))
        _check_reduction(Q, Rational(1, 5), Rational(2, 3))
    assert legendre_reduction(Reduction(1 / (1 - 2 * s)**2, m, s, S.Zero, S.One)) is None
    # a pole below the range: the antiderivatives continue through it
    _check_reduction(1 / (1 - 4 * s), Rational(1, 2), S.One)
    _check_reduction(1 / (1 - 4 * s)**2, Rational(1, 2), S.One)


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
    # an odd R keeps the roots of the quartic, two of which are complex:
    # the cosine map (the value is pi/4)
    found = elliptic_integral(x / sqrt(1 - x**4), x, 0, 1)
    assert found is not None and verify_numerically(found.value, x / sqrt(1 - x**4), x, S.Zero, S.One) is True
    assert abs(float(found.value.evalf(15)) - float(pi.evalf(15)) / 4) < 1e-12


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


def _verified(f: Expr, a: Expr, b: Expr, assumptions: Assumptions = None, samples: int = 2) -> Expr:
    found = elliptic_integral(f, x, a, b, assumptions)
    assert found is not None, f
    assert verify_numerically(found.value, f, x, a, b, assumptions, samples=samples) is True, (f, found.value)
    return found.value


def test_complex_roots_quartic() -> None:
    # two real roots and a complex pair (Byrd-Friedman 241): the whole cell
    # between the real roots, parts of it, and the roots as symbols
    P = (x**2 + 1) * (x + 2) * (3 - x)
    value = _verified(1 / sqrt(P), S(-2), S(3))
    assert value == 2**Rational(3, 4) * sqrt(5) * elliptic_k(sqrt(2) / 4 + S.Half) / 5
    _verified(1 / sqrt(P), S(-2), S.One)
    _verified(1 / sqrt(P), S.Zero, S(3))
    _verified(1 / sqrt(P), S.Zero, S.One)
    _verified(1 / sqrt(P), Rational(-3, 2), Rational(-1, 2))
    # numerators: polynomials up to the third degree, with the even and the odd parts
    for R in (x, x**2, x**3, x**2 - 3 * x + 1):
        _verified(R / sqrt(P), S(-2), S(3))
        _verified(R / sqrt(P), S.Zero, S.One)
        _verified(R / sqrt(P), Rational(-3, 2), S.Zero)
    # the square root in the numerator
    _verified(sqrt(P), S(-2), S(3))
    _verified(sqrt(P), Rational(-1, 2), S(2))
    _verified(x * sqrt(P), S(-2), S.One)
    # a positive leading coefficient makes P negative on the cell
    assert elliptic_integral(1 / sqrt((x**2 + 1) * (x + 2) * (x - 3)), x, -2, 3) is None
    # symbolic roots and quadratic
    a, b, p, q = symbols('a b p q', real=True)
    P_s = (a - x) * (x - b) * ((x - p)**2 + q**2)
    value = _verified(1 / sqrt(P_s), b, a, (b < a) & (q > 0), samples=4)
    assert value.has(elliptic_k)
    _verified(x / sqrt(P_s), b, a, (b < a) & (q > 0), samples=4)
    _verified(x**2 / sqrt(P_s), b, a, (b < a) & (q > 0), samples=4)


def test_complex_roots_cubic() -> None:
    # one real root and a complex pair (Byrd-Friedman 240): the half-lines
    # and parts of them, on both sides of the root
    _verified(1 / sqrt(x**3 + 1), S.Zero, oo)
    _verified(1 / sqrt(x**3 + 1), S(-1), oo)
    _verified(1 / sqrt(x**3 + 1), S(-1), S.Zero)
    _verified(1 / sqrt(x**3 + 1), S(2), S(5))
    _verified(1 / sqrt(1 - x**3), -oo, S.One)
    _verified(1 / sqrt(1 - x**3), S(-3), S.One)
    _verified(1 / sqrt(1 - x**3), -oo, S(-2))
    # the square root in the numerator and polynomial numerators: the
    # pole of the map at the infinite end, regularised in both parts
    _verified(sqrt(x**3 + 1), S(-1), S.Zero)
    _verified(sqrt(x**3 + 1), S.Zero, S(2))
    _verified(x**2 * sqrt(x**3 + 1), S(-1), S.Zero)
    _verified(x / sqrt(x**3 + 1), S(-1), S.One)
    _verified(x**2 / sqrt(x**3 + 1), S(-1), S(3))
    # a symbolic root, the quartic through y = x**2
    a = symbols('a', positive=True)
    _verified(1 / sqrt(x**3 + a**3), S.Zero, oo, samples=3)
    _verified(1 / sqrt(x**3 + a**3), -a, S.Zero, samples=3)
    _verified(1 / sqrt(x**4 + 1), S.Zero, oo)
    _verified(1 / sqrt(x**4 + 1), S.Zero, S.One)
    _verified(1 / sqrt(x**4 + x**2 + 1), S.Zero, oo)
    _verified(1 / sqrt((1 - x**2) * (x**2 + k**2)), S(-1), S.One, samples=3)
    # divergent
    assert elliptic_integral(x / sqrt(x**3 + 1), x, 0, oo) is None
    assert elliptic_integral(sqrt(x**3 + 1), x, -1, oo) is None


def test_complex_roots_half_lines() -> None:
    # a quartic with a complex pair beyond its real roots: inverted to a
    # cubic, then the cosine map with a part of the half-line
    P = (x**2 + 1) * (x - 2) * (x - 3)
    _verified(1 / sqrt(P), S(3), oo)
    _verified(1 / sqrt(P), -oo, S(2))
    _verified(1 / sqrt(P), S(4), oo)
    _verified(1 / sqrt(P), S(3), S(4))
    _verified(1 / sqrt(P), -oo, S.Zero)
    _verified(x / sqrt(P), S(3), S(5))
    _verified(1 / (x * sqrt(P)), S(3), oo)
    # divergent at infinity
    assert elliptic_integral(x / sqrt(P), x, 4, oo) is None


def test_complex_roots_with_poles() -> None:
    # poles of R outside the cell (the third kind), at a root, multiple
    P = (x**2 + 1) * (x + 2) * (3 - x)
    value = _verified(1 / ((x - 5) * sqrt(P)), S(-2), S(3))
    assert value.has(elliptic_pi)
    _verified(1 / ((x + 5) * sqrt(P)), S(-2), S.One)
    _verified(1 / ((x - 5) * sqrt(P)), S(-2), S.One)
    _verified(1 / ((x - 5)**2 * sqrt(P)), S(-1), S(3))
    _verified(1 / ((x + 1) * sqrt(x**3 + 1)), S.Zero, oo)
    _verified(x / ((x + 1) * sqrt(x**3 + 1)), S.Zero, oo)
    _verified(1 / ((x + 2) * sqrt(x**3 + 1)), S.Zero, oo)
    _verified(1 / ((x + 2)**2 * sqrt(x**3 + 1)), S.Zero, oo)
    # a pole inside the cell but outside the range: the antiderivatives
    # continue through it when its mirror image is outside the range too
    _verified(1 / ((x - 2) * sqrt(P)), S.Zero, S.One)
    _verified(1 / ((x + 1) * sqrt(P)), S.Zero, S.One)
    # divergent: a pole inside the range
    assert elliptic_integral(1 / ((x - 1) * sqrt(P)), x, 0, 2) is None
    assert elliptic_integral(1 / (x**2 * sqrt(x**3 + 1)), x, -1, 1) is None
    # the mirror image of the pole inside the range is not handled
    assert elliptic_integral(1 / ((x - 1)**2 * sqrt(P)), x, -2, 0) is None
    assert elliptic_integral(1 / (x**2 * sqrt(x**3 + 1)), x, 1, oo) is None


def test_complex_roots_are_read_from_the_factors() -> None:
    # an irreducible quartic with two real roots is not recognised
    assert elliptic_integral(1 / sqrt(x**4 + x - 1), x, 1, 2) is None
    # complex roots of a symbolic quadratic factor need the sign of the discriminant
    c = symbols('c', real=True)
    found = elliptic_integral(1 / sqrt((1 - x**2) * (x**2 + c)), x, -1, 1, c > 0)
    assert found is not None and found.value.has(elliptic_k)
    assert elliptic_integral(1 / sqrt((1 - x**2) * (x**2 + c)), x, -1, 1) is None


def test_elementary_parts() -> None:
    # the odd part of the cosine map alone: (a - x)(x - b) times the
    # quadratic, where R makes the whole integral elementary
    P = (x**2 + 1) * (x + 2) * (3 - x)
    rng = random.Random(3)
    for _ in range(3):
        lo = Rational(rng.randint(-19, 28), 10)
        hi = Rational(rng.randint(int(10 * lo) + 1, 30), 10)
        value = _verified(x / sqrt(P), lo, hi)
        assert value.has(elliptic_f) or value.has(log) or value.has(asin)
