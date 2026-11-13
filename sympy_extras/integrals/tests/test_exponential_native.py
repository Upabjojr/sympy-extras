"""Tests of the native antiderivatives of exponentials
(:mod:`sympy_extras.integrals.exponential` on the objects of the core);
the old API on SymPy objects is tested in test_exponential.py."""
from __future__ import annotations

from typing import Optional

from sympy_extras.assumptions.ask import Assumptions
from sympy_extras.core import Integer, Rational, Symbol, pi, sqrt
from sympy_extras.core.expr import Expr
from sympy_extras.functions import Ei, erf, erfi, exp, log, uppergamma
from sympy_extras.functions.elementary.piecewise import Piecewise
from sympy_extras.functions.special.gamma_functions import UpperGamma
from sympy_extras.functions.special.hyper import Hyper
from sympy_extras.integrals.conditions import numerically_equal
from sympy_extras.integrals.exponential import (_coefficients, _exponential_generator, _linear_pair, _roots,
                                                composite_gaussian, exponential_antiderivative, nested_power_exponential,
                                                power_exponential, rational_exponential,
                                                reciprocal_square_exponential)

x = Symbol('x')
v = Symbol('v', positive=True)
a = Symbol('a', negative=True)
b, c = Symbol('b'), Symbol('c')


def _checks(F: Optional[Expr], f: Expr, facts: Assumptions = None, variable: Symbol = x) -> bool:
    assert F is not None, f
    return numerically_equal(F.diff(variable), f, facts)


def test_incomplete_gamma_and_confluent_forms() -> None:
    f = x**(v - 1) * exp(a * x**2)
    F = power_exponential(f, x, [x > 0])
    assert F == -(-a)**(-v / 2) * uppergamma(v / 2, -a * x**2) / 2 and _checks(F, f, [x > 0])
    p = Symbol('p', positive=True)
    f = x**(v - 1) * exp(p * x**2)
    F = power_exponential(f, x, [x > 0])
    assert F is not None and F.has(Hyper) and _checks(F, f, [x > 0])
    # exp(x**3): s = 1/3 and a > 0, the confluent form (its check reaches
    # the native hyperexpand, which looped on 1F1(1/3; 4/3; z))
    F = exponential_antiderivative(exp(x**3), x, [x > 0])
    assert F is not None and F.has(Hyper) and _checks(F, exp(x**3), [x > 0])


def test_elementary_error_function_and_exponential_integral_cases() -> None:
    assert power_exponential(exp(-x**2), x) == sqrt(pi) * erf(x) / 2
    assert power_exponential(exp(x**2), x) == sqrt(pi) * erfi(x) / 2
    assert power_exponential(exp(2 * x) / x, x) == Ei(2 * x)
    F = power_exponential(exp(b * x**2), x)
    assert isinstance(F, Piecewise) and _checks(F, exp(b * x**2), [b < 0]) and _checks(F, exp(b * x**2), [b > 0])
    for f in (x**3 * exp(x**2), x**5 * exp(-x**3), exp(x**2) / x**3, x**2 * exp(-x**2 + 3 * x)):
        assert _checks(exponential_antiderivative(f, x), f, [x > 0]), f


def test_an_integer_exponent_from_factor_terms_is_an_integer() -> None:
    # factor_terms(a*x**2) had the exponent Rational(2, 1), which is not an
    # Integer: the even power was not seen and the confluent form came
    # instead of the incomplete gamma function
    F = exponential_antiderivative(x**(v - 1) * exp(a * x**2), x)
    assert F is not None and F.has(UpperGamma)


def test_the_generator_of_the_exponentials_has_a_canonical_sign() -> None:
    # the rates came from a set: u = exp(-x) turned (exp(x) + 1)**(1/3)
    # into (1/u + 1)**(1/3), which nothing takes
    assert _exponential_generator(exp(-x) * (exp(x) + 1)**Rational(1, 3), x) == 1
    assert _exponential_generator(exp(-2 * x) + exp(3 * x), x) == 1
    assert _exponential_generator(exp(x**2), x) is None


def test_shifts_quadratics_and_nested_powers() -> None:
    for f in (x * exp(x**2 + 2 * x), exp(2 * x) / (x + 1)**2, (x + 1) * exp(-x**3 - 3 * x**2 - 3 * x)):
        assert _checks(exponential_antiderivative(f, x), f, [x > 0]), f
    r, p = Symbol('r'), Symbol('p')
    f = exp(b * (x**r)**p)
    F = nested_power_exponential(f, x)
    assert F is not None and F.has(UpperGamma)
    f = exp(b * sqrt(x**2))
    assert nested_power_exponential(f, x) == x * exp(b * sqrt(x**2)) / (b * sqrt(x**2))


def test_rational_times_exponential_with_any_coefficients() -> None:
    F = rational_exponential(exp(c * x) / (x**2 - 1), x)
    assert F == exp(c) * Ei(c * (x - 1)) / 2 - exp(-c) * Ei(c * (x + 1)) / 2
    # a coefficient outside the polynomial domains (pi): the roots are
    # computed with a dummy for it, then put back
    f = exp(x) / (x**2 - pi**2)
    F = rational_exponential(f, x)
    assert F is not None and _checks(F, f, [x > 4])
    # a repeated root and a polynomial part
    f = (x**3 + 1) * exp(2 * x) / (x - 1)**2
    assert _checks(rational_exponential(f, x), f, [x > 2])


def test_the_coefficient_helpers() -> None:
    assert _coefficients(pi * x**2 + exp(c) * x + 1, x) == {2: pi, 1: exp(c), 0: Integer(1)}
    assert _coefficients(x + 1 / x, x) is None
    found = _roots({2: Integer(1), 0: -pi**2}, x)
    assert found is not None and set(found) == {pi, -pi}
    p, q = Symbol('p'), Symbol('q')
    assert _linear_pair([p + q - 1, p - q - sqrt(2)], p, q) == (Rational(1, 2) + sqrt(2) / 2, Rational(1, 2) - sqrt(2) / 2)
    assert _linear_pair([p + q - 1, 2 * p + 2 * q - 2], p, q) is None


def test_composite_and_reciprocal_square_gaussians() -> None:
    f = (1 - 1 / x**2) * exp(-(x + 1 / x)**2)
    assert composite_gaussian(f, x) == sqrt(pi) * erf((x**2 + 1) / x) / 2
    f = exp((x + 1 / x)**2) / x**2
    assert _checks(composite_gaussian(f, x), f, [x > 0])
    f = (x + 2 * log(x)) * exp(-(x + log(x)**2)**2) / x
    assert _checks(composite_gaussian(f, x), f, [x > 0])
    f = exp(-x**2 - 1 / x**2)
    assert _checks(reciprocal_square_exponential(f, x), f, [x > 0])
    assert reciprocal_square_exponential(exp(-x**2), x) is None
