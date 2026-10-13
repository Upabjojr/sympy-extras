"""Trager's algorithm on a general curve against the worked examples of
the literature: Bronstein's tutorial (ISSAC 1998, section 2) and Kauers'
examples (ISSAC 2008), each answer checked by differentiation."""
from __future__ import annotations

import random

from sympy import symbols, sqrt, cbrt, log, Rational, Expr, N

from sympy_extras._typing import as_expr
from sympy_extras.integrals.curves import curve_antiderivative, is_nonelementary_on_curve, curve_reduce

x = symbols('x')


def _derivative_matches(F: object, f: object, points: int = 6) -> None:
    """``F' == f`` at random rational points (where the radicals are
    real), to 25 digits."""
    F_, f_ = as_expr(F), as_expr(f)
    difference = F_.diff(x) - f_
    rng = random.Random(7)
    checked = 0
    for _ in range(40):
        x0 = Rational(rng.randint(110, 900), 100)
        value = N(difference.subs(x, x0), 30)
        if not value.is_number:
            continue
        assert abs(complex(value)) < 1e-20, (F_, f_, x0, value)
        checked += 1
        if checked >= points:
            return
    assert checked > 0, (F_, f_)


def _same_derivative(F: Expr, G: Expr) -> None:
    """Two antiderivatives of the same function differ by a constant."""
    rng = random.Random(3)
    values = []
    for _ in range(3):
        x0 = Rational(rng.randint(150, 600), 100)
        values.append(complex(N((F - G).subs(x, x0), 30)))
    assert all(abs(v - values[0]) < 1e-20 for v in values), (F, G, values)


def test_bronsteins_tutorial_examples() -> None:
    # Example 6: dx/(x sqrt(1 - x^3)) = (1/3) log((sqrt(1 - x^3) - 1)/(sqrt(1 - x^3) + 1))
    f = 1 / (x * sqrt(1 - x**3))
    F = curve_antiderivative(f, x)
    assert F is not None
    expected = log((sqrt(1 - x**3) - 1) / (sqrt(1 - x**3) + 1)) / 3
    # the points where the radicand is positive: x < 1
    rng = random.Random(1)
    for _ in range(5):
        x0 = Rational(rng.randint(10, 90), 100)
        assert abs(complex(N((F.diff(x) - f).subs(x, x0), 30))) < 1e-20
        assert abs(complex(N((F - expected).subs(x, x0) - (F - expected).subs(x, Rational(1, 2)), 30))) < 1e-20
    # Example 5: x dx/sqrt(1 - x^3) is not elementary
    assert is_nonelementary_on_curve(x / sqrt(1 - x**3), x) is True
    # Examples 3 and 4: (2x^8 + 1) sqrt(x^8 + 1)/(x^17 + 2x^9 + x) =
    # -sqrt(x^8 + 1)/(4(x^8 + 1)) + (1/4) log(x^4/(1 + sqrt(x^8 + 1)))
    f = (2 * x**8 + 1) * sqrt(x**8 + 1) / (x**17 + 2 * x**9 + x)
    F = curve_antiderivative(f, x)
    assert F is not None
    _derivative_matches(F, f)
    _same_derivative(F, -sqrt(x**8 + 1) / (4 * (x**8 + 1)) + log(x**4 / (1 + sqrt(x**8 + 1))) / 4)


def test_kauers_examples() -> None:
    # Example 2: sqrt(x^2 + 1)/(x^4 + 1), four logarithms over 128 g^4 + 16 g^2 + 1
    f = sqrt(x**2 + 1) / (x**4 + 1)
    F = curve_antiderivative(f, x)
    assert F is not None
    _derivative_matches(F, f)
    # Examples 5 and 6: the cube root of x^2 (x + 1), an integral basis with y^2/x
    y = cbrt(x**2 * (x + 1))
    f = 1 / ((4 * x + 4 - y) * y)
    F = curve_antiderivative(f, x)
    assert F is not None
    _derivative_matches(F, f)
    z = cbrt(x * (x + 1)**2)
    _same_derivative(F, log(8 + 9 * x + 6 * y + 12 * z) / 4 - log(8 + 7 * x + 6 * y - 12 * z) / 4)
    # Example 7: two square roots
    u = (x + 1) * (4 * x - 3) * sqrt(x - 1) + (x - 1) * (4 * x + 1) * sqrt(x + 1) - 2 * (2 * x - 1) * sqrt(x - 1) * sqrt(x + 1)
    f = u / ((x - 1) * (x + 1) * (4 * x - 5))
    F = curve_antiderivative(f, x)
    assert F is not None
    _derivative_matches(F, f)
    # (the paper prints two logarithms, of the residues 1 and -3; the
    # residues being rationally dependent, one logarithm of the divisor
    # P - 3Q does here, and the closed form is checked by differentiation
    # rather than transcribed)
    assert F.count(log) >= 1
    # Example 10: logarithms with different multiplicities
    f = (2 * x**3 + 6 * x**2 - 7 * x - 7 - (x - 1) * (3 * x + 1) * sqrt(x + 1)) / ((x**2 - 1) * x * (x**2 - x - 1))
    F = curve_antiderivative(f, x)
    assert F is not None
    _derivative_matches(F, f)
    _same_derivative(F, 3 * log(x - 1) - 6 * log(sqrt(x + 1) - 1) - 8 * log(1 + sqrt(x + 1))
                     + 4 * log(1 + x + x * sqrt(x + 1)) - 2 * log(x + sqrt(x + 1)))


def test_kauers_example_with_a_double_factor() -> None:
    # Example 9: (x + sqrt(x^2 + 1))/((x^2 + 1)(x + 1))
    f = (x + sqrt(x**2 + 1)) / ((x**2 + 1) * (x + 1))
    F = curve_antiderivative(f, x)
    assert F is not None
    _derivative_matches(F, f)


def test_the_reduction_returns_the_remainder() -> None:
    found = curve_reduce(x / sqrt(1 - x**3), x)
    assert found is not None
    F, h = found
    assert h != 0
    rng = random.Random(2)
    for _ in range(4):
        x0 = Rational(rng.randint(10, 90), 100)
        value = complex(N((F.diff(x) + h - x / sqrt(1 - x**3)).subs(x, x0), 30))
        assert abs(value) < 1e-20
