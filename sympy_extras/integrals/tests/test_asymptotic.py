"""Tests of the asymptotic expansions of integrals."""
from __future__ import annotations

from sympy import symbols, exp, sqrt, oo, log, cos, sin, pi, I, S, Order, Rational, cosh, gamma
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
    # a maximum of h at both endpoints: the two endpoint contributions
    value = laplace_method(S.One, x**2, x, S.NegativeOne, S.One, t)
    assert value is not None and (value.removeO()/exp(t)).expand() == 1/t + 1/(2*t**2)


def test_several_maxima() -> None:
    # two maxima of the same height: the contributions add
    value = laplace_method(S.One, -(x**2 - 1)**2, x, -oo, oo, t, 1)
    assert value is not None and value.removeO() == sqrt(pi/t)
    # the amplitude weighs them separately
    value = laplace_method(x + 2, -(x**2 - 1)**2, x, -oo, oo, t, 1)
    assert value is not None and value.removeO() == 2*sqrt(pi/t)
    # a lower maximum is exponentially smaller and dropped: h(-1) = 0
    # against h(1) = -1/2 (the range cuts the symmetry)
    value = laplace_method(S.One, -(x**2 - 1)**2, x, -oo, S.Half, t, 1)
    assert value is not None and value.removeO() == sqrt(pi)/(2*sqrt(t))
    # a maximum of order four: Gamma(1/4)/(2 t**(1/4)), exact
    value = laplace_method(S.One, -x**4, x, -oo, oo, t, 1)
    assert value is not None and value.removeO() == gamma(Rational(1, 4))/(2*t**Rational(1, 4))
    # its correction from the amplitude: the moment of u**2 is Gamma(3/4)/(2 t**(3/4))
    value = laplace_method(1 + x**2, -x**4, x, -oo, oo, t, 2)
    assert value is not None
    assert (value.removeO() - gamma(Rational(1, 4))/(2*t**Rational(1, 4)) - gamma(Rational(3, 4))/(2*t**Rational(3, 4))).simplify() == 0
    # an endpoint maximum with a vanishing slope is half of the interior one:
    # exp(t cos x) on (0, 2 pi) is 2 pi I_0(t) ~ sqrt(2 pi/t) exp(t) (DLMF 10.40.1)
    value = laplace_method(S.One, cos(x), x, S.Zero, 2*pi, t, 2)
    assert value is not None and (value.removeO()/(sqrt(2*pi/t)*exp(t))).expand() == 1 + 1/(8*t)
    # three maxima of height 0 (at 0 and +-1, with h ~ -2 x**2 and -8 u**2)
    # with the minima at +-1/sqrt(3) between them ignored
    value = laplace_method(S.One, -x**2*(x - 1)**2*(x + 1)**2, x, -oo, oo, t, 1)
    assert value is not None and value.removeO() == 2*sqrt(pi)/sqrt(t)
    # a maximum of order three at an endpoint: Gamma(1/3)/(3 t**(1/3))
    value = laplace_method(S.One, -x**3, x, S.Zero, oo, t, 1)
    assert value is not None and value.removeO() == gamma(Rational(1, 3))/(3*t**Rational(1, 3))


def test_steepest_descent() -> None:
    from sympy_extras.integrals.asymptotic import steepest_descent
    # the Gaussian with a linear complex term: exact by completing the square
    value = steepest_descent(S.One, -x**2 + I*x, x, t)
    assert value is not None and value.removeO() == sqrt(pi/t)*exp(-t/4)
    value = asymptotic_integral(exp(-t*x**2 + 2*I*t*x), x, -oo, oo, t)
    assert value is not None and value.removeO() == sqrt(pi/t)*exp(-t)
    # an amplitude evaluated at the saddle i/2
    value = steepest_descent(1 + x**2, -x**2 + I*x, x, t)
    assert value is not None and value.removeO() == 3*sqrt(pi/t)*exp(-t/4)/4
    # the quartic: the two saddles of the upper half plane contribute, the
    # one of larger Re h on the negative imaginary axis lies off the
    # contour; the sum is real (a conjugate pair)
    value = steepest_descent(S.One, -x**4 + I*x, x, t)
    assert value is not None
    terms = value.removeO().as_ordered_terms()
    assert len(terms) == 2 and abs(complex(value.removeO().subs(t, 30).evalf()).imag) < 1e-12
    # a degenerate saddle (h'' = 0 at i pi/2): refused
    assert steepest_descent(S.One, I*x - cosh(x), x, t) is None


def test_uniform_expansions() -> None:
    from sympy import airyai, airyaiprime, erfc
    from sympy_extras.integrals.asymptotic import uniform_expansion
    a = symbols('a', positive=True)
    # the Airy integral is exact: 2 pi t**(-1/3) Ai(-a t**(2/3))
    value = uniform_expansion(S.One, x**3/3 - a*x, x, -oo, oo, t)
    assert value is not None and value.removeO() == 2*pi*airyai(-a*t**Rational(2, 3))/t**Rational(1, 3)
    value = asymptotic_integral(cos(t*(x**3/3 - a*x)), x, -oo, oo, t, uniform=True)
    assert value is not None and value.removeO() == 2*pi*airyai(-a*t**Rational(2, 3))/t**Rational(1, 3)
    # an amplitude: p0 and q0 from its values at the stationary points
    value = uniform_expansion(x + 2, x**3/3 - a*x, x, -oo, oo, t)
    assert value is not None
    body = value.removeO().expand()
    assert body.coeff(airyai(-a*t**Rational(2, 3))) == 4*pi/t**Rational(1, 3)
    assert body.coeff(airyaiprime(-a*t**Rational(2, 3))) == -2*pi*I/t**Rational(2, 3)
    # h(x1) < h(x2): the sign of the Ai' term flips
    value = uniform_expansion(x + 2, -x**3/3 + a*x, x, -oo, oo, t)
    assert value is not None and value.removeO().expand().coeff(airyaiprime(-a*t**Rational(2, 3))) == 2*pi*I/t**Rational(2, 3)
    # a stationary point near an endpoint: the error function form, exact
    # for h = x**2 and phi = 1 (the endpoint term vanishes)
    value = uniform_expansion(S.One, x**2, x, -a, oo, t)
    assert value is not None
    assert value.removeO() == sqrt(pi)*(2 - erfc(a*sqrt(t)*exp(-I*pi/4)))*exp(I*pi/4)/(2*sqrt(t))
    # the stationary point outside the range: erfc of a positive argument
    value = uniform_expansion(S.One, x**2, x, a, oo, t)
    assert value is not None and value.removeO() == sqrt(pi)*erfc(a*sqrt(t)*exp(-I*pi/4))*exp(I*pi/4)/(2*sqrt(t))
    # one stationary point far from both ends, or none: refused
    assert uniform_expansion(S.One, x**2, x, -oo, oo, t) is None
    assert uniform_expansion(S.One, x, x, S.Zero, S.One, t) is None


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
