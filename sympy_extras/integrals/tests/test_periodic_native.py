"""Tests of the mean value (Laurent coefficient) integrals of periodic
integrands on the objects of the core
(:mod:`sympy_extras.integrals.periodic`): the values against mpmath's
quadrature and Bessel functions (independent sources), the series written
down against the formal power series, and the agreement with the SymPy
face."""
from __future__ import annotations

from typing import Callable

import mpmath
from sympy.testing.pytest import raises

from sympy_extras._testing import untyped
from sympy_extras.core import I, Integer, Rational, S, Symbol, pi, sqrt
from sympy_extras.core.expr import Expr
from sympy_extras.functions import besseli, besselj, cos, cosh, exp, log, sin, sinh, tan
from sympy_extras.integrals.brackets import taylor_coefficient
from sympy_extras.integrals.conditions import ConditionalValue
from sympy_extras.integrals.periodic import _known_series, laurent_coefficient, mean_value_integral, to_laurent

x, z = Symbol('x'), Symbol('z')
a = Symbol('a', positive=True)


def _number(e: Expr) -> complex:
    return complex(e.evalf(30))


def _quadrature(f: Callable[[mpmath.mpf], mpmath.mpc], lo: mpmath.mpf, hi: mpmath.mpf) -> complex:
    with mpmath.workdps(30):
        return complex(mpmath.quad(f, mpmath.linspace(lo, hi, 9)))


def test_to_laurent() -> None:
    assert to_laurent(cos(2 * x) + sin(x), x, z) == z**2 / 2 + 1 / (2 * z**2) - I * (z - 1 / z) / 2
    assert to_laurent(exp(3 * I * x), x, z) == z**3
    assert to_laurent(tan(x), x, z) == (z - 1 / z) / (2 * I) / ((z + 1 / z) / 2)
    # an outer function of the trigonometric functions becomes an exponential
    outer = to_laurent(cos(sin(x)), x, z)
    assert outer is not None and outer.has(type(exp(x)))
    # x occurring otherwise
    assert to_laurent(x * cos(x), x, z) is None
    assert to_laurent(cos(x / 2), x, z) is None
    assert to_laurent(log(x), x, z) is None


def test_laurent_coefficients() -> None:
    assert laurent_coefficient((z + 1 / z)**4, z) == 6
    assert laurent_coefficient((z + 1 / z)**4, z, 2) == 4
    assert laurent_coefficient(exp(z), z, 3) == Rational(1, 6)
    assert laurent_coefficient(exp(z), z, -1) == 0
    assert laurent_coefficient(exp(z / 2) * exp(1 / (2 * z)), z) == besseli(0, 1)
    # the generating function exp((z + 1/z)/2) = sum I_n(1) z**n, against mpmath
    for n in range(4):
        found = laurent_coefficient(exp(z / 2) * exp(1 / (2 * z)), z, n)
        assert found is not None
        assert abs(_number(found) - complex(mpmath.besseli(n, 1))) < 1e-12
    # exponentials of equal powers from different factors are merged
    assert laurent_coefficient(exp(1 / (2 * z)) * exp(z / 2) * exp(I / (2 * z)) * exp(-I * z / 2), z) \
        == besseli(0, sqrt(2))
    # a rational function of z: the residues, not the series; three functions of z
    assert laurent_coefficient(1 / (z**2 + 3 * z + 1), z) is None
    assert laurent_coefficient(cos(z) * sin(z) * exp(1 / z), z) is None


def test_mean_value_integral() -> None:
    assert mean_value_integral(exp(cos(x)) * cos(sin(x)), x, 0, 2 * pi) == ConditionalValue(2 * pi)
    assert mean_value_integral(exp(cos(x)), x, 0, 2 * pi) == ConditionalValue(2 * pi * besseli(0, 1))
    assert mean_value_integral(exp(cos(x)) * sin(sin(x)), x, -pi, pi) == ConditionalValue(0)
    assert mean_value_integral(cos(x)**4, x, 0, 4 * pi) == ConditionalValue(3 * pi / 2)
    assert mean_value_integral(cos(cos(x)), x, 0, 2 * pi) == ConditionalValue(2 * pi * besselj(0, 1))
    # 2 pi I_2(a), as I_0(a) - 2 I_1(a)/a by the recurrence
    found = mean_value_integral(exp(a * cos(x)) * cos(2 * x), x, 0, 2 * pi)
    assert found is not None
    assert abs(_number(found.value.subs(a, Rational(17, 10))) - complex(2 * mpmath.pi * mpmath.besseli(2, 1.7))) < 1e-12
    # not a number of periods, a rational integrand (for the residues), x elsewhere
    assert mean_value_integral(exp(cos(x)), x, 0, pi) is None
    assert mean_value_integral(1 / (2 + cos(x)), x, 0, 2 * pi) is None
    assert mean_value_integral(x * cos(x), x, 0, 2 * pi) is None


def test_values_against_quadrature() -> None:
    cases: list[tuple[Expr, Callable[[mpmath.mpf], mpmath.mpc]]] = [
        (exp(2 * cos(x) + sin(x)), lambda t: mpmath.exp(2 * mpmath.cos(t) + mpmath.sin(t))),
        (cosh(cos(x)) * cos(x)**2, lambda t: mpmath.cosh(mpmath.cos(t)) * mpmath.cos(t)**2),
        (sinh(sin(x)) * sin(x), lambda t: mpmath.sinh(mpmath.sin(t)) * mpmath.sin(t)),
        (exp(cos(2 * x)) * cos(2 * x), lambda t: mpmath.exp(mpmath.cos(2 * t)) * mpmath.cos(2 * t)),
        (sin(cos(x))**2, lambda t: mpmath.sin(mpmath.cos(t))**2),
    ]
    for f, g in cases:
        found = mean_value_integral(f, x, 0, 2 * pi)
        assert found is not None, f
        assert abs(_number(found.value) - _quadrature(g, mpmath.mpf(0), 2 * mpmath.pi)) < 1e-12, f


def test_known_series_agree_with_the_formal_power_series() -> None:
    y = Symbol('y')
    for f in (exp(y), sin(y), cos(y), sinh(y), cosh(y)):
        known = _known_series(f, y)
        formal = taylor_coefficient(f, y)
        assert known is not None and formal is not None
        assert (known.period, known.offset) == (formal.period, formal.offset)
        for k in range(8):
            assert known.coefficient.subs(known.index, Integer(k)) == formal.coefficient.subs(formal.index, Integer(k))
    assert _known_series(tan(y), y) is None and _known_series(exp(2 * y), y) is None


def test_agrees_with_the_sympy_face() -> None:
    from sympy import cos as sympy_cos, exp as sympy_exp, pi as sympy_pi, symbols as sympy_symbols
    from sympy_extras._numeric_sympy import core_expr
    from sympy_extras.integrals.periodic_sympy import mean_value_integral as face
    t = sympy_symbols('x')
    found = face(sympy_exp(3 * sympy_cos(t)) * sympy_cos(t), t, 0, 2 * sympy_pi)
    native = mean_value_integral(exp(3 * cos(x)) * cos(x), x, 0, 2 * pi)
    assert found is not None and native is not None and core_expr(found.value) == native.value
    assert native.value == 2 * pi * besseli(1, 3) and native.condition == S.true


def test_wrong_types_are_rejected() -> None:
    raises(TypeError, lambda: untyped(mean_value_integral)([1], x, 0, 2 * pi))
    raises(TypeError, lambda: untyped(laurent_coefficient)([z], z))
