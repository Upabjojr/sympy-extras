"""The expansions against the exact values at large parameters: the
relative error of the truncated expansion decreases like the next
order (DLMF 6.12.1, 5.11.1, 10.40.2, 10.17.3)."""
from __future__ import annotations

import mpmath

from sympy import symbols, exp, oo, log, cos, sin, pi, cosh, lambdify, I

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


def _quadrature(f: object, p: float, digits: int = 30) -> complex:
    g = lambdify(x, f, 'mpmath')
    with mpmath.workdps(digits):
        points = [mpmath.mpf(q) / 10 for q in range(-40, 41)]
        return complex(mpmath.quad(g, [mpmath.mpf('-inf')] + points + [mpmath.mpf('inf')]))


def test_several_maxima_against_quadrature() -> None:
    # two Gaussians at +-1: the two-term expansion errs like 1/t**2
    value = asymptotic_integral(exp(-t*(x**2 - 1)**2), x, -oo, oo, t, order=2)
    assert value is not None
    approximate = lambdify(t, value.removeO(), 'mpmath')
    errors = [abs(float(approximate(p) / _quadrature(exp(-p*(x**2 - 1)**2), p).real - 1)) for p in (20.0, 40.0)]
    assert errors[0] < 1e-3 and 3 < errors[0] / errors[1] < 5
    # the quartic maximum: Gamma(1/4)/(2 t**(1/4)) is exact
    value = asymptotic_integral(exp(-t*x**4), x, -oo, oo, t, order=1)
    assert value is not None
    approximate = lambdify(t, value.removeO(), 'mpmath')
    assert abs(float(approximate(7.0) / _quadrature(exp(-7*x**4), 7.0).real - 1)) < 1e-12
    # 2 pi I_0(t) from the endpoints of (0, 2 pi): the two-term expansion errs like 9/(128 t**2)
    value = asymptotic_integral(exp(t*cos(x)), x, 0, 2*pi, t, order=2)
    assert value is not None
    approximate = lambdify(t, value.removeO(), 'mpmath')
    errors = [abs(float(approximate(p) / (2*mpmath.pi*mpmath.besseli(0, p)) - 1)) for p in (20.0, 40.0)]
    assert errors[0] < 2e-4 and 3.5 < errors[0] / errors[1] < 4.5


def test_saddle_points_against_quadrature() -> None:
    # the quartic with a linear complex term: the pair of saddles in the
    # upper half plane, error relative to the envelope decreasing
    value = asymptotic_integral(exp(t*(-x**4 + I*x)), x, -oo, oo, t)
    assert value is not None
    terms = value.removeO().as_ordered_terms()
    assert len(terms) == 2
    errors = []
    for p in (40.0, 160.0):
        digits = 30 + int(p * 0.24 / 2.3)
        exact = _quadrature(exp(p*(-x**4 + I*x)), p, digits)
        values = [complex(term.subs(t, p).evalf(digits)) for term in terms]
        errors.append(abs(sum(values) - exact) / sum(abs(v) for v in values))
    assert errors[0] < 0.02 and errors[1] < 0.005
    # the shifted Gaussian is exact
    value = asymptotic_integral(exp(-t*x**2 + I*t*x), x, -oo, oo, t)
    assert value is not None
    assert abs(complex(value.removeO().subs(t, 9).evalf(20)) / _quadrature(exp(-9*x**2 + 9*I*x), 9.0) - 1) < 1e-12


def test_uniform_expansions_against_quadrature() -> None:
    from sympy_extras.integrals.asymptotic import uniform_expansion
    a = symbols('a', positive=True)
    # the Airy form with an amplitude: relative error like 1/t at fixed a
    value = uniform_expansion(exp(-x**2), x**3/3 - a*x, x, -oo, oo, t)
    assert value is not None
    approximate = lambdify((a, t), value.removeO(), 'mpmath')
    # (the absolute error scaled by sqrt(t): the sum of the two stationary
    # contributions oscillates through zero, where a relative error is
    # meaningless)
    errors = []
    for p in (60.0, 240.0):
        exact = _quadrature(exp(-x**2 + I*p*(x**3/3 - x/2)), p)
        errors.append(abs(complex(approximate(0.5, p)) - exact) * p**0.5)
    assert errors[0] < 0.05 and errors[1] < errors[0] / 3
    # at the coalescence a = 0 the stationary phase fails and the Airy form holds
    exact = _quadrature(exp(-x**2 + I*60*x**3/3), 60.0)
    assert abs(complex(approximate(0.0, 60.0)) / exact - 1) < 0.05
    # the error function form: a stationary point at distance a from the
    # endpoint, uniformly in a (a = 0.1 and a = 2), error like 1/t
    value = uniform_expansion(exp(-x**2), x**2, x, -a, oo, t)
    assert value is not None
    approximate = lambdify((a, t), value.removeO(), 'mpmath')
    for distance in (0.1, 2.0):
        errors = []
        for p in (30.0, 120.0):
            g = lambdify(x, exp(-x**2 + I*p*x**2), 'mpmath')
            with mpmath.workdps(30):
                exact = complex(mpmath.quad(g, [-distance] + [mpmath.mpf(q) / 10 for q in range(0, 41)] + [mpmath.inf]))
            errors.append(abs(complex(approximate(distance, p)) / exact - 1))
        assert errors[0] < 0.1 and errors[1] < errors[0] / 2
