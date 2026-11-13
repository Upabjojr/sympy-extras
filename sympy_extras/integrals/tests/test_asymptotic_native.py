"""Tests of the native asymptotic expansions
(:mod:`sympy_extras.integrals.asymptotic`) on the objects of the core: the
expansions against the known asymptotics of special functions (DLMF) and
mpmath's quadrature (independent sources), the helpers which replace
SymPy's ``Poly`` over ``EX`` and ``lambdify``, and the agreement with the
SymPy face."""
from __future__ import annotations

from typing import Optional

import mpmath
import sympy

from sympy.testing.pytest import raises

from sympy_extras._testing import untyped
from sympy_extras.core import I, Integer, Rational, Symbol, oo, pi, sqrt
from sympy_extras.core.evalf import numeric_value
from sympy_extras.core.expr import Expr
from sympy_extras.functions import airyai, cos, cosh, erfc, exp, gamma, log, sin
from sympy_extras.integrals import asymptotic_sympy
from sympy_extras.integrals.asymptotic import (_coefficients, _numeric_function, _saddle_points, asymptotic_integral,
                                               exponential_part, laplace_method, oscillatory_part, stationary_phase,
                                               steepest_descent, uniform_expansion, watson_lemma)
from sympy_extras.interop import from_sympy, to_sympy

x = Symbol('x')
t = Symbol('t', positive=True)
a = Symbol('a', positive=True)


def _value(e: Expr, values: dict[Symbol, Expr]) -> complex:
    found = numeric_value(e.subs(values), 100)
    assert found is not None, e
    return complex(found)


def _body(e: Optional[Expr]) -> Expr:
    assert e is not None
    return e.removeO()


def test_shapes() -> None:
    assert exponential_part(exp(-t*x)/(1 + x), x, t) == (1/(1 + x), -x)
    assert exponential_part(exp(t*(log(x) - x) - x), x, t) == (exp(-x), log(x) - x)
    assert exponential_part(exp(-t**2*x), x, t) is None and exponential_part(exp(-x), x, t) is None
    assert oscillatory_part(exp(I*t*x**2)*x, x, t) == (x, x**2, 'exp')
    assert oscillatory_part(cos(t*x + 1), x, t) is None


def test_coefficients_over_expressions() -> None:
    # the typed Poly has no domain EX: the coefficients are expressions
    assert _coefficients(t*(log(x) - x) + sin(x), t) == {1: log(x) - x, 0: sin(x)}
    assert _coefficients(-x**2 + I*x, x) == {2: Integer(-1), 1: I}
    assert _coefficients(exp(t)*x, t) is None and _coefficients(sqrt(t), t) is None
    assert _coefficients(t - t, t) == {}


def test_numeric_function() -> None:
    # the role of lambdify(x, e, 'mpmath'): agrees with the core's evaluation
    e = exp(-x**2 + I*x)*cos(x)/(1 + x**2)
    g = _numeric_function(e, x)
    with mpmath.workdps(30):
        for point in (Rational(1, 3), Integer(-2)):
            assert abs(complex(g(mpmath.mpf(float(point)))) - _value(e, {x: point})) < 1e-14
    raises(ValueError, lambda: _numeric_function(_undefined(), x))


def _undefined() -> Expr:
    """A node without a numerical rule: an undefined function."""
    from sympy_extras.core.function import Function
    return Function('f')(x)


def test_watson_lemma() -> None:
    # DLMF 6.12.1: exp(t) E_1(t) ~ 1/t - 1/t**2 + 2/t**3
    assert _body(watson_lemma(1/(1 + x), x, t, 3)) == 1/t - 1/t**2 + 2/t**3
    assert watson_lemma(sqrt(x), x, t, 3) == sqrt(pi)/(2*t**Rational(3, 2))
    assert watson_lemma(1/x, x, t) is None
    value = asymptotic_integral(exp(-2*t*x)*cos(x), x, 0, 5, t, order=2)
    assert _body(value) == 1/(2*t) - 1/(8*t**3)


def test_laplace_method() -> None:
    # Stirling: Gamma(t + 1)/t**(t + 1) ~ sqrt(2 pi/t) exp(-t) (1 + 1/(12 t) + 1/(288 t**2))
    body = _body(laplace_method(1, log(x) - x, x, 0, oo, t, 3))
    assert (body/(sqrt(2*pi/t)*exp(-t))).expand() == 1 + 1/(12*t) + 1/(288*t**2)
    # DLMF 10.40.2: 2 K_0(t) ~ sqrt(2 pi/t) exp(-t) (1 - 1/(8 t))
    body = _body(laplace_method(1, -cosh(x), x, -oo, oo, t, 2))
    assert (body/(sqrt(2*pi/t)*exp(-t))).expand() == 1 - 1/(8*t)
    # two maxima of the same height, and one of order four
    assert _body(laplace_method(1, -(x**2 - 1)**2, x, -oo, oo, t, 1)) == sqrt(pi/t)
    assert _body(laplace_method(1, -x**4, x, -oo, oo, t, 1)) == gamma(Rational(1, 4))/(2*t**Rational(1, 4))
    # 2 pi I_0(t) from the two endpoints of (0, 2 pi) (DLMF 10.40.1)
    body = _body(laplace_method(1, cos(x), x, 0, 2*pi, t, 2))
    assert (body/(sqrt(2*pi/t)*exp(t))).expand() == 1 + 1/(8*t)


def test_laplace_against_quadrature() -> None:
    # an interior maximum of h = x - x**2 at 1/2 with an amplitude: the error is O(1/t**2)
    value = asymptotic_integral(exp(t*(x - x**2))/(1 + x), x, 0, 1, t, order=2)
    body = _body(value)
    for p in (40, 160):
        exact = mpmath.quad(lambda u: mpmath.exp(p*(u - u**2))/(1 + u), [0, 0.5, 1])
        assert abs(_value(body, {t: Integer(p)})/complex(exact) - 1) < 3/p**2


def test_saddle_points() -> None:
    # the three complex roots of -4 x**3 + I (two were lost by solve when the
    # core took a sum of imaginary terms which cancel for a nonzero number)
    points = _saddle_points(-x**4 + I*x, x)
    assert points is not None and len(points) == 3
    assert _body(steepest_descent(1, -x**2 + I*x, x, t)) == sqrt(pi/t)*exp(-t/4)
    # the quartic: the pair of saddles in the upper half plane, a real sum
    body = _body(steepest_descent(1, -x**4 + I*x, x, t))
    assert abs(_value(body, {t: Integer(30)}).imag) < 1e-12
    assert steepest_descent(1, I*x - cosh(x), x, t) is None


def test_stationary_phase_and_uniform() -> None:
    assert _body(stationary_phase(1, x**2, x, -1, 1, t)) == sqrt(pi/t)*exp(I*pi/4)
    # DLMF 10.17.3: J_0(t) ~ sqrt(2/(pi t)) cos(t - pi/4)
    body = _body(asymptotic_integral(cos(t*sin(x))/pi, x, 0, pi, t))
    assert abs(_value(body - sqrt(2/(pi*t))*cos(t - pi/4), {t: Integer(17)})) < 1e-25
    # the Airy integral is exact
    assert _body(uniform_expansion(1, x**3/3 - a*x, x, -oo, oo, t)) == 2*pi*airyai(-a*t**Rational(2, 3))/t**Rational(1, 3)
    assert _body(uniform_expansion(1, x**2, x, a, oo, t)) == sqrt(pi)*erfc(a*sqrt(t)*exp(-I*pi/4))*exp(I*pi/4)/(2*sqrt(t))
    assert uniform_expansion(1, x**2, x, -oo, oo, t) is None


def test_order_keeps_the_exponential() -> None:
    # SymPy's Order(exp(a**2*t/4)/t**(5/2), (t, oo)) drops the exponential
    # for most hash seeds (sympy-extras#25); the core's keeps it
    value = asymptotic_integral(exp(t*(a*x - x**2)), x, -oo, oo, t, order=2)
    assert value is not None
    order = value.getO()
    assert order is not None and order.has(exp(a**2*t/4))


def test_wrong_input() -> None:
    raises(ValueError, lambda: asymptotic_integral(exp(-t*x), x, 0, oo, t, order=0))
    raises(TypeError, lambda: untyped(asymptotic_integral)([1], x, 0, oo, t))
    assert asymptotic_integral(exp(-x**2), x, 0, oo, t) is None


def test_agrees_with_the_face() -> None:
    X, T = sympy.Symbol('x'), sympy.Symbol('t', positive=True)
    native = asymptotic_integral(exp(-t*x)*sqrt(1 + x), x, 0, oo, t, order=2)
    face = asymptotic_sympy.asymptotic_integral(sympy.exp(-T*X)*sympy.sqrt(1 + X), X, 0, sympy.oo, T, order=2)
    assert native is not None and face is not None and from_sympy(face) == native and to_sympy(native) == face
