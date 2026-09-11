"""Tests of the Almkvist–Zeilberger algorithm and the holonomic method."""
from __future__ import annotations

from sympy import symbols, exp, cos, sin, sqrt, pi, oo, Eq, Function, S, Rational, cancel, simplify
from sympy.testing.pytest import raises

from sympy_extras._testing import untyped
from sympy_extras.integrals.conditions import ConditionalValue
from sympy_extras.integrals.telescoping import (DifferentialTelescoper, almkvist_zeilberger, holonomic_ode,
                                                holonomic_integral, is_hyperexponential, logarithmic_derivative)

x = symbols('x')
t = symbols('t', positive=True)
u = symbols('u')


def test_logarithmic_derivative() -> None:
    assert logarithmic_derivative(x**u * exp(-x**2), x) == (-2 * x**2 + u) / x
    assert logarithmic_derivative(exp(-u * x) / (x**2 + u**2), x) == cancel((-u * (x**2 + u**2) - 2 * x) / (x**2 + u**2))
    assert logarithmic_derivative(exp(exp(x)), x) is None
    assert logarithmic_derivative(S.Zero, x) is None


def test_is_hyperexponential() -> None:
    assert is_hyperexponential(exp(-x * u) * x**3 / (x**2 + u**2), x, u)
    assert is_hyperexponential(x**Rational(1, 3) * (1 - x)**Rational(1, 2) * (1 - u * x)**2, x, u)
    # x**u has the logarithmic derivative log(x) in u
    assert not is_hyperexponential(x**u * exp(-x), x, u)
    assert not is_hyperexponential(exp(-x * u) * sin(x), x, u)


def test_telescoper_is_verified() -> None:
    for F, coefficients in [(exp(-x * u) * x**3, [1]), (1 / (x**2 + u**2), [1, u]),
                            (1 / (x**2 + u**2)**2, [3, u]), (exp(-x**2) * exp(2 * x * u), [-2 * u, 1])]:
        found = almkvist_zeilberger(F, x, u)
        assert found is not None, F
        assert found.coefficients == coefficients, (F, found)
        assert found.check(), F
        assert found.order == len(coefficients) - 1
    # a second order equation (GR 3.325: the integral is sqrt(pi)/2 exp(-2 t))
    found = almkvist_zeilberger(exp(-x**2 - t**2 / x**2), x, t)
    assert found is not None and found.coefficients == [-4, 0, 1] and found.check()
    # a sum of hyperexponential terms with a common equation (cos as exponentials)
    found = almkvist_zeilberger(exp(-x**2) * cos(2 * t * x), x, t)
    assert found is not None and found.coefficients == [2 * t, 1] and found.check()
    # the bug: the certificate was not rescaled together with the
    # coefficients, so 1/(x**2 + t**2) got R = -x/t for [1, t]
    found = almkvist_zeilberger(1 / (x**2 + u**2), x, u)
    assert found is not None and found.certificate == -x


def test_operator_and_repr() -> None:
    found = almkvist_zeilberger(1 / (x**2 + u**2), x, u)
    assert found is not None
    I = Function('I')(u)
    assert found.operator() == I + u * I.diff(u)
    assert repr(found) == 'DifferentialTelescoper([1, u], -x)'
    manual = DifferentialTelescoper(1 / (x**2 + u**2), x, u, [S.One, u], -x)
    assert manual.check()


def test_not_hyperexponential_or_no_equation() -> None:
    assert almkvist_zeilberger(x**u * exp(-x), x, u) is None
    # terms with different equations (no least common left multiple)
    assert almkvist_zeilberger(exp(-x * u) + exp(-x**2 * u), x, u) is None


def test_holonomic_ode() -> None:
    I = Function('I')(u)
    assert holonomic_ode(exp(-x**2) * exp(2 * x * u), x, -oo, oo, u) == Eq(-2 * u * I + I.diff(u), 0)
    I = Function('I')(t)
    # an order zero equation: the integrand is a derivative in x, and the
    # boundary term is the value of the integral
    equation = holonomic_ode(exp(-x * t) * x**3, x, 0, oo, t)
    assert equation == Eq(I, 6 / t**4)
    # a boundary term which is not finite: nothing is returned
    assert holonomic_ode(exp(x * t), x, 0, oo, t) is None


def test_holonomic_integral() -> None:
    found = holonomic_integral(exp(-x**2) * cos(2 * t * x), x, 0, oo, t)
    assert found is not None and simplify(found.value - sqrt(pi) * exp(-t**2) / 2) == 0
    found = holonomic_integral(1 / (x**2 + t**2)**2, x, -oo, oo, t)
    assert found == ConditionalValue(pi / (2 * t**3))
    found = holonomic_integral(exp(-x**2 - t**2 / x**2), x, 0, oo, t)
    assert found is not None and simplify(found.value - sqrt(pi) * exp(-2 * t) / 2) == 0
    assert holonomic_integral(x**t * exp(-x), x, 0, oo, t) is None


def test_wrong_types_are_rejected() -> None:
    raises(TypeError, lambda: untyped(almkvist_zeilberger)([1], x, u))
    raises(TypeError, lambda: untyped(holonomic_integral)(exp(-x), x, [0], oo, u))
