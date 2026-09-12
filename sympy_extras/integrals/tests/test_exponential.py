"""Tests of the antiderivatives of exponentials of powers and of rational
functions of an exponential."""
from __future__ import annotations

from sympy import (symbols, exp, sqrt, pi, erf, erfi, Ei, uppergamma, hyper, Piecewise, Rational,
                   diff, simplify, atan, log)

from sympy_extras._typing import as_expr
from sympy_extras.assumptions.ask import Assumptions
from sympy_extras.integrals.conditions import numerically_equal
from sympy_extras.integrals.exponential import exponential_antiderivative, power_exponential, exponential_rational

x = symbols('x')
v = symbols('v', positive=True)
a = symbols('a', negative=True)
b, c = symbols('b c')


def _checks(F: object, f: object, facts: Assumptions = None) -> bool:
    F_, f_ = as_expr(F), as_expr(f)
    return simplify(diff(F_, x) - f_) == 0 or numerically_equal(as_expr(diff(F_, x)), f_, facts)


def test_incomplete_gamma_forms() -> None:
    # a < 0 known: the real incomplete gamma form of DLMF 8.2.4
    F = power_exponential(x**(v - 1) * exp(a * x**2), x, [x > 0])
    assert F == -(-a)**(-v / 2) * uppergamma(v / 2, -a * x**2) / 2
    assert _checks(F, x**(v - 1) * exp(a * x**2), [x > 0])
    F = power_exponential(x**(v - 1) * exp(-x), x, [x > 0])
    assert F == -uppergamma(v, x) and _checks(F, x**(v - 1) * exp(-x), [x > 0])
    # a constant in the exponent and a coefficient
    F = power_exponential(3 * x**(v - 1) * exp(a * x**3 + c), x, [x > 0])
    assert F is not None and F.has(uppergamma) and _checks(F, 3 * x**(v - 1) * exp(a * x**3 + c), [x > 0])
    # the sign of a unknown or positive: the confluent form, real for every a
    p = symbols('p', positive=True)
    F = power_exponential(x**(v - 1) * exp(p * x**2), x, [x > 0])
    assert F == x**v * hyper((v / 2,), (v / 2 + 1,), p * x**2) / v
    assert _checks(F, x**(v - 1) * exp(p * x**2), [x > 0])
    F = power_exponential(x**(v - 1) * exp(b * x**2), x, [x > 0])
    assert F is not None and F.has(hyper) and _checks(F, x**(v - 1) * exp(b * x**2), [x > 0, b > 0])
    # a negative power of x: n < 0
    F = power_exponential(exp(a / x) / x**3, x, [x > 0])
    assert _checks(F, exp(a / x) / x**3, [x > 0])
    F = power_exponential(exp(-x**3) / x**2, x, [x > 0])
    assert F == -uppergamma(-Rational(1, 3), x**3) / 3


def test_elementary_and_error_function_cases() -> None:
    assert power_exponential(exp(-x**2), x) == sqrt(pi) * erf(x) / 2
    assert power_exponential(exp(x**2), x) == sqrt(pi) * erfi(x) / 2
    F = power_exponential(exp(b * x**2), x)
    assert isinstance(F, Piecewise) and _checks(F, exp(b * x**2), [b < 0]) and _checks(F, exp(b * x**2), [b > 0])
    assert power_exponential(x**2 * exp(x**2), x) == x * exp(x**2) / 2 - sqrt(pi) * erfi(x) / 4
    assert power_exponential(x**3 * exp(x**2), x) == x**2 * exp(x**2) / 2 - exp(x**2) / 2
    assert power_exponential(exp(2 * x) / x, x) == Ei(2 * x)
    assert power_exponential(x**5 * exp(-x**3), x) == -x**3 * exp(-x**3) / 3 - exp(-x**3) / 3
    # s a negative integer: upwards to the exponential integral
    F = power_exponential(exp(x**2) / x**3, x)
    assert F is not None and F.has(Ei) and _checks(F, exp(x**2) / x**3, [x > 0])
    F = power_exponential(exp(sqrt(x)), x)
    assert F == 2 * sqrt(x) * exp(sqrt(x)) - 2 * exp(sqrt(x))
    # sums of terms
    F = power_exponential(exp(-x**2) + x * exp(-x**2), x)
    assert F == sqrt(pi) * erf(x) / 2 - exp(-x**2) / 2


def test_quadratic_exponents() -> None:
    F = exponential_antiderivative(x * exp(x**2 + 2 * x), x)
    assert F is not None and _checks(F, x * exp(x**2 + 2 * x))
    F = exponential_antiderivative((2 * x + 4) * exp(-x**2 - 2 * x - 1), x)
    assert F == sqrt(pi) * erf(x + 1) - exp(-(x + 1)**2)
    F = exponential_antiderivative(x**2 * exp(a * x**2 + b * x), x)
    assert F is not None and F.has(erf) and _checks(F, x**2 * exp(a * x**2 + b * x))
    F = exponential_antiderivative(exp(a * x**2 + b * x), x)
    assert F is not None and _checks(F, exp(a * x**2 + b * x))


def test_rational_functions_of_the_exponential() -> None:
    assert exponential_rational(exp(x) / (exp(2 * x) + 1), x) == atan(exp(x))
    assert exponential_rational(1 / (exp(x) + 1), x) == x - log(exp(x) + 1)
    F = exponential_rational(exp(3 * x) / (exp(2 * x) - 1), x)
    assert F is not None and _checks(F, exp(3 * x) / (exp(2 * x) - 1), [x > 1])
    # symbolic rate
    r = symbols('r', positive=True)
    F = exponential_rational(exp(r * x) / (exp(2 * r * x) + 1), x)
    assert F == atan(exp(r * x)) / r
    # binomials of the exponential by t**q = a + b exp(c x)
    assert exponential_rational(exp(x) / sqrt(exp(x) + 1), x) == 2 * sqrt(exp(x) + 1)
    F = exponential_rational((exp(x) + 2)**Rational(3, 2) / (exp(x) + 1), x)
    assert F is not None and _checks(F, (exp(x) + 2)**Rational(3, 2) / (exp(x) + 1))
    F = exponential_rational((exp(x) + 1)**Rational(1, 3) / exp(x), x)
    assert F is not None and _checks(F, (exp(x) + 1)**Rational(1, 3) / exp(x))


def test_outside_the_module() -> None:
    assert exponential_antiderivative(exp(x**2) * exp(x**3), x) is None      # two powers
    assert exponential_antiderivative(exp(x) * log(x), x) is None            # not of the forms
    assert exponential_antiderivative(x**2 / (x**2 + 1), x) is None          # no exponential
    assert exponential_antiderivative(exp(x) / (exp(x) + x), x) is None      # not rational in exp(x)
    assert power_exponential(exp(exp(x)), x) is None


def test_the_generator_of_the_exponentials_has_a_canonical_sign() -> None:
    # the bug: the rates of the exponentials came from a set, and when
    # exp(-x) came first the substitution was u = exp(-x), which turned
    # (exp(x) + 1)**(1/3) into (1/u + 1)**(1/3), refused; one run in two
    # returned None depending on SymPy's random generator
    import sympy.core.random as srandom
    from sympy_extras.integrals.exponential import _exponential_generator
    assert _exponential_generator(exp(-x) * (exp(x) + 1)**Rational(1, 3), x) == 1
    assert _exponential_generator(exp(-2 * x) + exp(3 * x), x) == 1
    for seed in range(4):
        srandom.seed(seed)
        F = exponential_rational((exp(x) + 1)**Rational(1, 3) / exp(x), x)
        assert F is not None and _checks(F, (exp(x) + 1)**Rational(1, 3) / exp(x)), seed
