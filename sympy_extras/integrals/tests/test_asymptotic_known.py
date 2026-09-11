"""The expansions against the exact values at large parameters: the
relative error of the truncated expansion decreases like the next
order (DLMF 6.12.1, 5.11.1, 10.40.2, 10.17.3)."""
from __future__ import annotations

import mpmath

from sympy import symbols, exp, oo, log, cos, sin, pi, cosh, lambdify

from sympy_extras.integrals.asymptotic import asymptotic_integral

x = symbols('x')
t = symbols('t', positive=True)


def test_exponential_integral() -> None:
    # exp(t) E_1(t): the three-term expansion errs like 6/t**3 (the next term)
    value = asymptotic_integral(exp(-t*x)/(1 + x), x, 0, oo, t, order=3)
    assert value is not None
    approximate = lambdify(t, value.removeO(), 'mpmath')
    errors = [abs(float(approximate(p) / (mpmath.exp(p)*mpmath.e1(p)) - 1)) for p in (20.0, 40.0)]
    assert errors[0] < 1e-3 and 6 < errors[0] / errors[1] < 10


def test_stirling() -> None:
    # Gamma(t + 1) t**(-t-1) against the two-term expansion: error ~ 1/(288 t**2)
    value = asymptotic_integral(exp(t*(log(x) - x)), x, 0, oo, t, order=2)
    assert value is not None
    approximate = lambdify(t, value.removeO(), 'mpmath')
    errors = [abs(float(approximate(p) / (mpmath.gamma(p + 1) * mpmath.power(p, -p - 1)) - 1)) for p in (10.0, 20.0)]
    assert errors[0] < 1e-4 and 3.5 < errors[0] / errors[1] < 4.5


def test_bessel_k_and_j() -> None:
    # 2 K_0(t): the two-term expansion errs like 9/(128 t**2)
    value = asymptotic_integral(exp(-t*cosh(x)), x, -oo, oo, t, order=2)
    assert value is not None
    approximate = lambdify(t, value.removeO(), 'mpmath')
    errors = [abs(float(approximate(p) / (2*mpmath.besselk(0, p)) - 1)) for p in (20.0, 40.0)]
    assert errors[0] < 1e-3 and 3.5 < errors[0] / errors[1] < 4.5
    # J_0(t) by stationary phase: the leading term errs like 1/(8 t)
    value = asymptotic_integral(cos(t*sin(x))/pi, x, 0, pi, t)
    assert value is not None
    approximate = lambdify(t, value.removeO(), 'mpmath')
    errors = [abs(float(approximate(p) - mpmath.besselj(0, p))) * float(mpmath.sqrt(p)) for p in (50.5, 101.0)]
    assert errors[0] < 0.05 and errors[1] < errors[0]


def test_watson_against_quadrature() -> None:
    value = asymptotic_integral(exp(-t*x)*cos(x)/(1 + x**2), x, 0, oo, t, order=4)
    assert value is not None
    approximate = lambdify(t, value.removeO(), 'mpmath')
    for p in (30.0, 60.0):
        exact = mpmath.quad(lambda u: mpmath.exp(-p*u)*mpmath.cos(u)/(1 + u**2), [0, mpmath.inf])
        assert abs(float(approximate(p) / exact - 1)) < 50 / p**4
