"""Tests of the native validated numerical integration
(:mod:`sympy_extras.integrals.validated` on the objects of the core): every
reported error bound is checked against the true value computed
independently (closed forms evaluated by mpmath at 40 digits); the old API
on SymPy objects is tested in test_validated.py."""
from __future__ import annotations

from fractions import Fraction
from typing import Callable, Optional

import mpmath

from sympy_extras.core import Float, Integer, Rational, Symbol, oo, pi, sqrt
from sympy_extras.functions import atan, cos, exp, log, sin
from sympy_extras.integrals.validated import (_divided, _polynomial, _removable_form, _roots_on_ray,
                                              _trigonometric_antiderivative, enclosure, validated_integral)

x = Symbol('x')


def _holds(found: Optional[tuple[Float, Float]], exact: Callable[[], mpmath.mpf], below: float) -> bool:
    """Whether the enclosure contains the exact value and the error is
    below ``below``."""
    assert found is not None
    value, error = found
    with mpmath.workdps(40):
        v = mpmath.mpf(value.as_fraction().numerator) / value.as_fraction().denominator
        e = mpmath.mpf(error.as_fraction().numerator) / error.as_fraction().denominator
        return bool(abs(v - exact()) <= e) and float(error) < below


def test_smooth_integrands() -> None:
    assert _holds(validated_integral(exp(-x**2), x, 0, 1), lambda: mpmath.sqrt(mpmath.pi) * mpmath.erf(1) / 2,
                  1e-15)
    # the five-point Gauss rule with the bound from the tenth derivative
    assert _holds(validated_integral(exp(-x**2), x, 0, 1, 30),
                  lambda: mpmath.sqrt(mpmath.pi) * mpmath.erf(1) / 2, 1e-30)
    assert _holds(validated_integral(atan(x), x, 0, 1), lambda: mpmath.pi / 4 - mpmath.log(2) / 2, 1e-15)
    # pi is enclosed and the strip between the enclosures bounded by the range
    assert _holds(validated_integral(cos(x)**2, x, 0, pi, 10), lambda: mpmath.pi / 2, 1e-10)


def test_endpoint_singularities() -> None:
    # x = u**q removes (x - a)**(p/q); the logarithms are bounded by the model
    assert _holds(validated_integral(x**Rational(-1, 2) * (1 - x)**Rational(-1, 2), x, 0, 1),
                  lambda: mpmath.pi, 1e-15)
    assert _holds(validated_integral(log(x)**2, x, 0, 1), lambda: mpmath.mpf(2), 1e-15)
    assert _holds(validated_integral(sqrt(x) * log(x), x, 0, 1), lambda: mpmath.mpf(-4) / 9, 1e-15)
    # the removable singularity of sin(x)/x at 0 through the Taylor remainder
    assert _holds(validated_integral(sin(x) / x, x, 0, 1), lambda: mpmath.si(1), 1e-15)
    assert validated_integral(1 / x, x, 0, 1) is None
    assert validated_integral(1 / x**2, x, 0, 1) is None


def test_infinite_ranges() -> None:
    assert _holds(validated_integral(1 / (1 + x**2), x, 0, oo, 20), lambda: mpmath.pi / 2, 1e-20)
    assert _holds(validated_integral(x * exp(-x), x, 0, oo), lambda: mpmath.mpf(1), 1e-14)
    assert _holds(validated_integral(exp(-x**2), x, -oo, oo, 12), lambda: mpmath.sqrt(mpmath.pi), 1e-12)
    assert _holds(validated_integral(exp(x), x, -oo, 0, 12), lambda: mpmath.mpf(1), 1e-12)
    assert _holds(validated_integral(exp(-x), x, oo, 0, 10), lambda: mpmath.mpf(-1), 1e-10)


def test_oscillatory_tails() -> None:
    assert _holds(validated_integral(sin(x) / x, x, 0, oo, 8), lambda: mpmath.pi / 2, 1e-8)
    assert _holds(validated_integral(cos(x) / (1 + x**2), x, 0, oo, 8), lambda: mpmath.pi / (2 * mpmath.e), 1e-8)
    # a negative frequency: sin(-x) is the node -sin(x) of the core
    assert _holds(validated_integral(sin(-x) / x, x, 0, oo, 8), lambda: -mpmath.pi / 2, 1e-8)


def test_hidden_zeros() -> None:
    # (1 + sqrt(2))**2 - 3 - 2*sqrt(2) is exactly 0: as a bound its
    # enclosure is a strip around 0, as a coefficient a tiny interval
    zero = (1 + sqrt(2))**2 - 3 - 2 * sqrt(2)
    assert _holds(validated_integral(x**2, x, zero, 1, 12), lambda: mpmath.mpf(1) / 3, 1e-12)
    assert _holds(validated_integral(zero * x + 1, x, 0, 1), lambda: mpmath.mpf(1), 1e-15)
    # 1/(x + 0) is not integrable at 0: nothing is claimed
    assert validated_integral(1 / (x + zero), x, 0, 1, 10) is None


def test_parameters_and_degenerate_ranges() -> None:
    a = Symbol('a')
    assert validated_integral(exp(-a * x), x, 0, 1) is None
    assert validated_integral(x, x, 0, a) is None
    found = validated_integral(exp(-x), x, 1, 1)
    assert found is not None and found[0].as_fraction() == 0 and found[1].as_fraction() == 0
    assert found[0].precision == Float(0, 15).precision
    assert _holds(validated_integral(x, x, 1, 0, 12), lambda: mpmath.mpf(-1) / 2, 1e-12)


def test_enclosure_contains_the_value() -> None:
    r = enclosure(log(1 + x), x, Integer(0), Integer(1), mpmath.mpf('1e-12'))
    assert r is not None
    exact = 2 * mpmath.log(2) - 1
    assert r.a <= exact <= r.b and r.delta < 1e-12


def test_polynomial_coefficients() -> None:
    # the coefficients of x**2 + pi*x - sqrt(2), lowest degree first: the
    # typed Poly has no domain for pi, these are any expressions
    assert _polynomial(x**2 + pi * x - sqrt(2), x) == [-sqrt(2), pi, Integer(1)]
    assert _polynomial((x + 1)**2, x) == [Integer(1), Integer(2), Integer(1)]
    assert _polynomial(Integer(0), x) == [Integer(0)]
    assert _polynomial(sqrt(x) + 1, x) is None
    assert _polynomial(1 / x, x) is None
    assert _polynomial(x * sin(x), x) is None
    # Ruffini–Horner: x**2 - 1 = (x - 1)(x + 1) + 0
    quotient, remainder = _divided([Integer(-1), Integer(0), Integer(1)], Integer(1))
    assert quotient == [Integer(1), Integer(1)] and remainder == 0


def test_removable_forms() -> None:
    # sin(x)/x = sin'(xi) = cos(xi) between 0 and x
    assert _removable_form(sin(x) / x, x, Integer(0)) == cos(x)
    # (1 - cos(x))/x**2 = cos(xi)/2
    assert _removable_form((1 - cos(x)) / x**2, x, Integer(0)) == cos(x) / 2
    # 1/x is not removable
    assert _removable_form(1 / x, x, Integer(0)) is None


def test_trigonometric_antiderivatives() -> None:
    found = _trigonometric_antiderivative(3 * sin(2 * x + 1), x)
    assert found is not None and (found.diff(x) - 3 * sin(2 * x + 1)).expand() == 0
    found = _trigonometric_antiderivative(cos(x) / 5, x)
    assert found is not None and found == sin(x) / 5
    assert _trigonometric_antiderivative(x * sin(x), x) is None
    assert _trigonometric_antiderivative(sin(x**2), x) is None


def test_roots_on_a_ray() -> None:
    # (x - 1)(x - 3) has its roots 1 and 3 in [1, oo), 3 only in [2, oo)
    p = [Fraction(1), Fraction(-4), Fraction(3)]
    assert _roots_on_ray(p, Fraction(1)) == 2
    assert _roots_on_ray(p, Fraction(2)) == 1
    assert _roots_on_ray(p, Fraction(4)) == 0
    assert _roots_on_ray([Fraction(1), Fraction(0), Fraction(1)], Fraction(-5)) == 0
    assert _roots_on_ray([Fraction(7)], Fraction(0)) == 0


def test_the_result_is_a_pair_of_floats() -> None:
    found = validated_integral(Integer(1), x, 0, 1, 12)
    assert found is not None
    value, error = found
    assert isinstance(value, Float) and isinstance(error, Float)
    assert value.as_fraction() == 1 and value.dps == 15 and error.dps == 5
