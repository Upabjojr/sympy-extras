"""Tests of the asymptotic expansions of integrals."""
from __future__ import annotations

from sympy import symbols, exp, sqrt, oo, log, cos, sin, pi, I, S, Order, Rational, cosh
from sympy.testing.pytest import raises

from sympy_extras._testing import untyped
from sympy_extras.integrals.asymptotic import (asymptotic_integral, watson_lemma, laplace_method,
                                               stationary_phase, exponential_part, oscillatory_part)

x = symbols('x')
t = symbols('t', positive=True)


def test_shapes() -> None:
    found = exponential_part(exp(-t*x)/(1 + x), x, t)
    assert found is not None and found == (1/(1 + x), -x)
    found = exponential_part(x**t*exp(-t*x), x, t)
    assert found is not None and found[1] == log(x) - x
    # exp(t*h + g): g goes into phi
    found = exponential_part(exp(t*(log(x) - x) - x), x, t)
    assert found is not None and found[0] == exp(-x) and found[1] == log(x) - x
    assert exponential_part(exp(-t**2*x), x, t) is None
    assert exponential_part(1/(1 + t*x), x, t) is None
    assert exponential_part(exp(-x), x, t) is None
    oscillatory = oscillatory_part(cos(t*sin(x))/pi, x, t)
    assert oscillatory is not None and oscillatory == (1/pi, sin(x), 'cos')
    oscillatory = oscillatory_part(exp(I*t*x**2)*x, x, t)
    assert oscillatory is not None and oscillatory == (x, x**2, 'exp')
    assert oscillatory_part(cos(t*x + 1), x, t) is None


def test_watson_lemma() -> None:
    # DLMF 6.12.1: exp(t) E_1(t) ~ 1/t - 1/t**2 + 2/t**3 - ...
    assert watson_lemma(1/(1 + x), x, t, 3) == 1/t - 1/t**2 + 2/t**3 + Order(t**-4, (t, oo))
    # a terminating expansion is exact
    assert watson_lemma(sqrt(x), x, t, 3) == sqrt(pi)/(2*t**Rational(3, 2))
    assert watson_lemma(x**2 + 1, x, t, 5) == 2/t**3 + 1/t
    # the order term follows the next exponent (the odd powers of sin(x))
    assert watson_lemma(sin(x), x, t, 2) == 1/t**2 - 1/t**4 + Order(t**-6, (t, oo))
    # a scale of the exponential and a finite range
    assert asymptotic_integral(exp(-2*t*x)*cos(x), x, 0, 5, t, order=2) == 1/(2*t) - 1/(8*t**3) + Order(t**-5, (t, oo))
    # not integrable at 0
    assert watson_lemma(1/x, x, t) is None


def test_laplace_method() -> None:
    # Stirling: Integral(x**t exp(-t x)) = Gamma(t + 1)/t**(t + 1) ~ sqrt(2 pi/t) exp(-t) (1 + 1/(12 t) + 1/(288 t**2))
    value = laplace_method(S.One, log(x) - x, x, S.Zero, oo, t, 3)
    assert value is not None
    body = value.removeO()
    assert (body/(sqrt(2*pi/t)*exp(-t))).expand() == 1 + 1/(12*t) + 1/(288*t**2)
    # DLMF 10.40.2: 2 K_0(t) ~ sqrt(2 pi/t) exp(-t) (1 - 1/(8 t))
    value = laplace_method(S.One, -cosh(x), x, -oo, oo, t, 2)
    assert value is not None and (value.removeO()/(sqrt(2*pi/t)*exp(-t))).expand() == 1 - 1/(8*t)
    # a Gaussian with an amplitude: the corrections come from phi
    value = laplace_method(1/(1 + x**2), -x**2, x, -oo, oo, t, 3)
    assert value is not None and (value.removeO()/sqrt(pi/t)).expand() == 1 - 1/(2*t) + 3/(4*t**2)
    # the endpoint case: h decreasing from a
    value = laplace_method(S.One, -x - x**2, x, S.Zero, S.One, t, 2)
    assert value is not None and value.removeO().expand() == 1/t - 2/t**2
    # h increasing to b
    value = laplace_method(x**2, x, x, S.Zero, S.One, t, 2)
    assert value is not None and (value.removeO()/exp(t)).expand() == 1/t - 2/t**2
    # a maximum of h at the boundary but h not monotone: refused; several
    # interior critical points: refused
    assert laplace_method(S.One, x**2, x, S.NegativeOne, S.One, t) is None
    assert laplace_method(S.One, -(x**2 - 1)**2, x, -oo, oo, t) is None


def test_stationary_phase() -> None:
    # Fresnel: Integral(exp(I t x**2), (x, -1, 1)) ~ sqrt(pi/t) exp(I pi/4)
    value = stationary_phase(S.One, x**2, x, S.NegativeOne, S.One, t)
    assert value is not None and value.removeO() == sqrt(pi/t)*exp(I*pi/4)
    # DLMF 10.17.3: J_0(t) ~ sqrt(2/(pi t)) cos(t - pi/4)
    value = asymptotic_integral(cos(t*sin(x))/pi, x, 0, pi, t)
    assert value is not None and (value.removeO() - sqrt(2/(pi*t))*cos(t - pi/4)).simplify() == 0
    assert stationary_phase(S.One, x, x, S.Zero, S.One, t) is None


def test_wrong_input() -> None:
    raises(ValueError, lambda: asymptotic_integral(exp(-t*x), x, 0, oo, t, order=0))
    raises(TypeError, lambda: untyped(asymptotic_integral)([1], x, 0, oo, t))
    assert asymptotic_integral(exp(-x**2), x, 0, oo, t) is None
