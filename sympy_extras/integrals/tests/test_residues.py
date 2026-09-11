"""Tests of the residue integrator: the values are checked by
simplification against hand computations, or numerically against
mpmath's quadrature at sample values of the parameters."""
from __future__ import annotations

from typing import Optional

import mpmath

from sympy import (I, Rational, S, Symbol, cos, exp, log, oo, pi, simplify, sin, sqrt,
                   symbols, lambdify, Abs)
from sympy.core.expr import Expr
from sympy.polys.rootoftools import ComplexRootOf

from sympy_extras._typing import as_expr
from sympy_extras.integrals.conditions import ConditionalValue
from sympy_extras.integrals.residues import (fourier_integral, keyhole_integral, period_integral,
                                             poles, rational_function, real_line_integral,
                                             residue_integral)
from sympy_extras.settings import configure

x = Symbol('x')
a, b = symbols('a b', positive=True)
k, c = symbols('k c')


def _value(found: Optional[ConditionalValue]) -> Expr:
    assert found is not None
    return found.value


def _same(first: Expr, second: Expr) -> bool:
    difference = as_expr(simplify(first - second))
    if difference == 0:
        return True
    return abs(complex(as_expr(first - second).evalf(20))) < 1e-15


def _quadrature(f: Expr, lo: Expr, hi: Expr, sample: dict[Symbol, Expr]) -> float:
    g = lambdify(x, f.xreplace(sample), 'mpmath')
    lower = mpmath.mpf('-inf') if lo == -oo else mpmath.mpf(str(lo))
    upper = mpmath.mpf('inf') if hi == oo else mpmath.mpf(str(as_expr(hi).evalf(20)))
    points = [lower, upper] if lower != mpmath.mpf('-inf') and upper != mpmath.mpf('inf') else \
        [lower, mpmath.mpf(0), upper] if lower == mpmath.mpf('-inf') else [lower, mpmath.mpf(1), upper]
    with mpmath.workdps(20):
        return float(mpmath.quad(g, points))


def _agrees(found: Optional[ConditionalValue], f: Expr, lo: Expr, hi: Expr,
            sample: dict[Symbol, Expr]) -> bool:
    value = _value(found)
    ours = complex(as_expr(value.xreplace(sample)).evalf(20))
    expected = _quadrature(f, lo, hi, sample)
    return abs(ours - expected) < 1e-8 * (1 + abs(expected))


def test_rational_function() -> None:
    fraction = rational_function((x**2 + 1) / (x**4 + x + 1), x)
    assert fraction is not None
    assert fraction.excess == 2
    assert rational_function(exp(x) / (x + 1), x) is None
    assert rational_function(1 / (x + a), x) is not None
    # the common factor is cancelled
    fraction = rational_function((x + 1) / ((x + 1) * (x**2 + 1)), x)
    assert fraction is not None
    assert fraction.numerator.degree() == 0 and fraction.denominator.degree() == 2


def test_poles() -> None:
    fraction = rational_function(1 / ((x**2 + 1)**2 * (x - 3)), x)
    assert fraction is not None
    found = poles(fraction.denominator)
    assert found is not None
    assert sorted((str(p.point), p.multiplicity) for p in found) == [('-I', 2), ('3', 1), ('I', 2)]
    fraction = rational_function(1 / (x**5 - x - 1), x)
    assert fraction is not None
    found = poles(fraction.denominator)
    assert found is not None
    assert sum(p.multiplicity for p in found) == 5
    assert any(isinstance(p.point, ComplexRootOf) for p in found)


def test_real_line_rational() -> None:
    assert _same(_value(real_line_integral(1 / (x**2 + 1), x)), pi)
    assert _same(_value(real_line_integral(1 / (x**4 + 1), x)), pi / sqrt(2))
    assert _same(_value(real_line_integral(1 / (x**2 + 1)**2, x)), pi / 2)
    assert _same(_value(real_line_integral(x**2 / (x**4 + 1)**2, x)), pi * sqrt(2) / 8)
    # a double pole off the imaginary axis
    assert _same(_value(real_line_integral(1 / (x**2 + 2 * x + 5)**2, x)), pi / 16)
    # deg Q = deg P + 1 does not converge, deg P > deg Q neither
    assert real_line_integral(x / (x**2 + 1), x) is None
    # a real pole: not a job for the residue theorem
    assert real_line_integral(1 / (x**2 - 1), x) is None


def test_real_line_parametric() -> None:
    assert _same(_value(real_line_integral(1 / (x**2 + a**2), x)), pi / a)
    found = real_line_integral(1 / ((x**2 + a**2) * (x**2 + b**2)), x)
    assert _same(_value(found), pi / (a * b * (a + b)))
    # the poles of 1/(x**2 + c) cannot be placed without an assumption
    assert real_line_integral(1 / (x**2 + c), x) is None
    found = real_line_integral(1 / (x**2 + c), x, c > 0)
    assert found is not None
    assert _agrees(found, 1 / (x**2 + c), -oo, oo, {c: Rational(7, 3)})


def test_fourier() -> None:
    assert _same(_value(fourier_integral(cos(2 * x) / (x**2 + 1), x)), pi * exp(-2))
    assert _same(_value(fourier_integral(x * sin(x) / (x**2 + 1), x)), pi * exp(-1))
    assert _same(_value(fourier_integral(exp(I * x) / (x**2 + 1), x)), pi * exp(-1))
    # k < 0: the lower half plane
    assert _same(_value(fourier_integral(exp(-I * x) / (x**2 + 1), x)), pi * exp(-1))
    assert _same(_value(fourier_integral(x * sin(-x) / (x**2 + 1), x)), -pi * exp(-1))
    assert _same(_value(fourier_integral(cos(b * x) / (x**2 + a**2), x)), pi * exp(-a * b) / a)
    # a symbolic frequency gets the condition k > 0
    found = fourier_integral(x * sin(k * x) / (x**2 + 4), x)
    assert found is not None
    assert found.condition == (k > 0)
    assert _same(found.value, pi * exp(-2 * k))
    # sin(x)/x has a real pole: not handled here
    assert fourier_integral(sin(x) / x, x) is None
    # a real exponential is not a Fourier integral
    assert fourier_integral(exp(x) / (x**2 + 1), x) is None


def test_fourier_half_line() -> None:
    assert _same(_value(fourier_integral(cos(x) / (x**2 + 1), x, half=True)), pi * exp(-1) / 2)
    # not even: no half
    assert fourier_integral(cos(x) / (x**2 + x + 1), x, half=True) is None


def test_fourier_complex_pole_is_real_valued() -> None:
    # the value with a pole at exp(2 pi i / 3) is real and simplified
    f = x * cos(x) / (x**2 + x + 1)
    found = fourier_integral(f, x)
    value = _value(found)
    assert not value.has(I)
    # the oscillatory integral, summed over the half lines by mpmath's quadosc
    g = lambdify(x, f, 'mpmath')
    with mpmath.workdps(20):
        expected = mpmath.quadosc(g, [0, mpmath.inf], omega=1) + \
            mpmath.quadosc(lambda t: g(-t), [0, mpmath.inf], omega=1)
    assert abs(complex(value.evalf(20)) - complex(expected)) < 1e-10


def test_keyhole_power() -> None:
    assert _same(_value(keyhole_integral(sqrt(x) / (x**2 + 1), x)), pi / sqrt(2))
    found = keyhole_integral(x**c / (x + 1), x)
    assert found is not None
    assert _same(found.value, -pi / sin(pi * c))
    assert found.condition == ((c > -1) & (c < 0))
    f = x**Rational(1, 3) / (x**2 + 2 * x + 2)
    found = keyhole_integral(f, x)
    assert _agrees(found, f, S.Zero, oo, {})
    # divergent at infinity
    assert keyhole_integral(sqrt(x) / (x + 1), x) is None
    # a pole on the positive axis
    assert keyhole_integral(sqrt(x) / (x - 1), x) is None


def test_keyhole_rational() -> None:
    assert _same(_value(keyhole_integral(1 / (x**3 + 1), x)), 2 * pi / (3 * sqrt(3)))
    assert _same(_value(keyhole_integral(1 / (x**4 + 1), x)), pi / (2 * sqrt(2)))
    assert _same(_value(keyhole_integral(1 / (x**2 + 1)**2, x)), pi / 4)
    # log(x)/(x**2 + 1) vanishes by x -> 1/x; log(x)**2 gives pi**3/8
    assert _same(_value(keyhole_integral(log(x) / (x**2 + 1), x)), S.Zero)
    assert _same(_value(keyhole_integral(log(x)**2 / (x**2 + 1), x)), pi**3 / 8)
    # a power of x with an integer exponent is folded into the fraction
    assert _same(_value(keyhole_integral(x / (x**3 + 1), x)), 2 * pi / (3 * sqrt(3)))
    # the integrand 1/((x + 1)(x + 2)) has real negative poles
    assert _same(_value(keyhole_integral(1 / ((x + 1) * (x + 2)), x)), log(2))


def test_keyhole_absorbs_zero() -> None:
    # x**(-1/2)/(x + 1) = x**(1/2) * (1/(x (x + 1)))
    assert _same(_value(keyhole_integral(sqrt(x) / (x * (x + 1)), x)), pi)


def test_period() -> None:
    assert _same(_value(period_integral(1 / (2 + cos(x)), x, S.Zero, 2 * pi)), 2 * pi / sqrt(3))
    assert _same(_value(period_integral(1 / (5 + 3 * sin(x)), x, S.Zero, 2 * pi)), pi / 2)
    # two periods, and a shifted range
    assert _same(_value(period_integral(1 / (2 + cos(x)), x, S.Zero, 4 * pi)), 4 * pi / sqrt(3))
    assert _same(_value(period_integral(1 / (2 + cos(x)), x, -pi, pi)), 2 * pi / sqrt(3))
    # a multiple angle
    assert _same(_value(period_integral(cos(2 * x) / (5 - 4 * cos(x)), x, S.Zero, 2 * pi)), pi / 6)
    # a pole on the unit circle
    assert period_integral(1 / (1 + cos(x)), x, S.Zero, 2 * pi) is None
    # not a full period
    assert residue_integral(1 / (2 + cos(x)), x, S.Zero, pi) is None


def test_period_parametric() -> None:
    found = period_integral(1 / (1 - 2 * a * cos(x) + a**2), x, S.Zero, 2 * pi, a < 1)
    assert _same(_value(found), 2 * pi / (1 - a**2))
    found = period_integral(1 / (1 - 2 * a * cos(x) + a**2), x, S.Zero, 2 * pi, a > 1)
    assert _same(_value(found), 2 * pi / (a**2 - 1))
    # the pole of a + b cos(x) inside the circle is found at a sample of the
    # parameters and the value verified numerically (numerical checks on)
    found = period_integral(1 / (a + b * cos(x)), x, S.Zero, 2 * pi, a > b)
    assert found is not None
    assert _agrees(found, 1 / (a + b * cos(x)), S.Zero, 2 * pi, {a: S(3), b: S.One})
    with configure(numerical_checks=False):
        assert period_integral(1 / (a + b * cos(x)), x, S.Zero, 2 * pi, a > b) is None
    # without assumptions the position of the pole is unknown
    assert period_integral(1 / (1 - 2 * a * cos(x) + a**2), x, S.Zero, 2 * pi) is None


def test_residue_integral_dispatch() -> None:
    assert _same(_value(residue_integral(3 / (x**2 + 1), x, -oo, oo)), 3 * pi)
    assert residue_integral(1 / (x**2 + 1), x, S.Zero, S.One) is None
    assert residue_integral(exp(-x), x, S.Zero, oo) is None
    assert residue_integral(S.One, x, -oo, oo) is None
    assert _same(_value(residue_integral(1 / (x**2 + 1), x, S.Zero, oo)), pi / 2)
    assert _same(_value(residue_integral(Abs(3) / (2 + cos(x)), x, S.Zero, 2 * pi)), 6 * pi / sqrt(3))
