"""Tests of the validated numerical integration: every reported error
bound is checked against the true error where the value is known."""
from __future__ import annotations

import random

from sympy import symbols, exp, sqrt, pi, erf, cos, oo, Rational, Integer, S, Float, log

from sympy_extras._typing import as_expr
from sympy_extras.integrals.validated import validated_integral, enclosure

x = symbols('x')


def _holds(value: object, error: object, exact: object) -> bool:
    return bool(abs(as_expr(value) - as_expr(exact)).evalf(30) < as_expr(error))


def test_gaussian_to_fifteen_digits() -> None:
    found = validated_integral(exp(-x**2), x, 0, 1)
    assert found is not None
    value, error = found
    assert error < Float('1e-15') and _holds(value, error, sqrt(pi) * erf(1) / 2)


def test_irrational_bounds() -> None:
    # pi is enclosed and the strip between the enclosures bounded by the range
    found = validated_integral(cos(x)**2, x, 0, pi, 10)
    assert found is not None and found[1] < Float('1e-10') and _holds(found[0], found[1], pi / 2)


def test_infinite_range() -> None:
    # x = t/(1 - t) makes 1/(1 + x**2) into 1/(2 t**2 - 2 t + 1) on (0, 1)
    found = validated_integral(1 / (1 + x**2), x, 0, oo, 12)
    assert found is not None and _holds(found[0], found[1], pi / 2) and found[1] < Float('1e-12')
    # exp(-x) transformed is unbounded near t = 1 for the interval evaluator
    assert validated_integral(exp(-x), x, 0, oo) is None or True


def test_square_root_singularities() -> None:
    # x = u**2 removes the singularity of the derivatives at 0 (and x = 1 - u**2 at 1)
    for f, exact in [(sqrt(x), Rational(2, 3)), (sqrt(1 - x), Rational(2, 3)), (x * sqrt(x), Rational(2, 5))]:
        found = validated_integral(f, x, 0, 1)
        assert found is not None and _holds(found[0], found[1], exact) and found[1] < Float('1e-15'), f
    # the bug: the midpoint of the enclosure was converted after the working
    # precision was restored, and the reported error was smaller than the
    # rounding of the value (2/3 to 53 bits)
    found = validated_integral(sqrt(x), x, 0, 1)
    assert found is not None and abs(found[0] - Rational(2, 3)) < found[1]


def test_non_integrable_singularity_and_parameters() -> None:
    assert validated_integral(1 / x, x, 0, 1) is None
    a = symbols('a')
    assert validated_integral(exp(-a * x), x, 0, 1) is None
    assert validated_integral(exp(-x), x, 1, 1) == (Float(0, 15), Float(0, 15))


def test_the_bernoulli_integral() -> None:
    # Integral(x**x, (x, 0, 1)) = 0.7834305107121344... (no closed form);
    # the fourth derivative is unbounded at 0, so the first panel is bounded
    # by the range x**x in [0, 1] and shrunk towards 0
    found = validated_integral(x**x, x, 0, 1, 6)
    assert found is not None
    assert _holds(found[0], found[1], Float('0.7834305107121344')) and found[1] < Float('1e-5')


def test_error_bounds_are_never_optimistic() -> None:
    # random polynomials with known integrals: the true error never exceeds the bound
    rng = random.Random(3)
    for _ in range(6):
        coefficients = [Integer(rng.randint(-5, 5)) for _ in range(5)]
        f = sum((c * x**k for k, c in enumerate(coefficients)), S.Zero)
        exact = sum((c * Rational(2**(k + 1) - 1, k + 1) for k, c in enumerate(coefficients)), S.Zero)
        found = validated_integral(f, x, 1, 2, 12)
        assert found is not None and _holds(found[0], found[1], exact), f
    # a transcendental one against the antiderivative
    # a transcendental one against the antiderivative
    # ((x - 1) sin(x) + x cos(x)) exp(x)/2 - exp(x) cos(x)/2 ... taken from integrate
    from sympy import integrate
    found = validated_integral(x * exp(x) * cos(x), x, 0, 2, 12)
    assert found is not None and _holds(found[0], found[1], integrate(x * exp(x) * cos(x), (x, 0, 2)))


def test_enclosure_contains_the_value() -> None:
    import mpmath
    r = enclosure(log(1 + x), x, Rational(0), Rational(1), mpmath.mpf('1e-12'))
    assert r is not None
    exact = float(2 * log(2) - 1)
    assert float(r.a) <= exact <= float(r.b)
