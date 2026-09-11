"""Residue integrals checked against published tables.

The values are facts taken from Gradshteyn and Ryzhik, *Table of
integrals, series, and products* (7th edition, 2007), from the NIST
Digital Library of Mathematical Functions and from the worked examples
of Ahlfors' *Complex analysis* (3rd edition, chapter 4.5.3); the
formula numbers are quoted with each case."""
from __future__ import annotations

from typing import Optional

from sympy import Rational, S, Symbol, cos, exp, log, oo, pi, simplify, sin, sqrt, symbols, gamma
from sympy.core.expr import Expr

from sympy_extras._typing import as_expr
from sympy_extras.integrals.conditions import ConditionalValue
from sympy_extras.integrals.residues import residue_integral

x = Symbol('x')
a, b = symbols('a b', positive=True)
n = Symbol('n', integer=True, positive=True)


def _value(found: Optional[ConditionalValue]) -> Expr:
    assert found is not None
    return found.value


def _same(first: Expr, second: Expr) -> bool:
    return as_expr(simplify(first - second)) == 0


def test_gradshteyn_ryzhik_rational() -> None:
    # GR 3.241.2: Integral(x**(mu-1)/(1 + x**nu), (x, 0, oo)) = pi/(nu sin(mu pi/nu))
    # with mu = 1, nu = 3 and mu = 1, nu = 4
    assert _same(_value(residue_integral(1 / (1 + x**3), x, S.Zero, oo)), pi / (3 * sin(pi / 3)))
    assert _same(_value(residue_integral(1 / (1 + x**4), x, S.Zero, oo)), pi / (4 * sin(pi / 4)))
    # with mu = 3/2, nu = 2: Integral(sqrt(x)/(1 + x**2)) = pi/(2 sin(3 pi/4))
    assert _same(_value(residue_integral(sqrt(x) / (1 + x**2), x, S.Zero, oo)), pi / (2 * sin(3 * pi / 4)))
    # GR 3.252.?/DLMF 5.12.3 (Beta integral): Integral(x**(a-1)/(1+x), (x, 0, oo)) = pi/sin(pi a), 0 < a < 1
    found = residue_integral(x**(a - 1) / (1 + x), x, S.Zero, oo)
    assert found is not None
    assert _same(found.value, pi / sin(pi * a))
    assert found.condition == ((a - 1 > -1) & (a - 1 < 0))
    # GR 3.249.1-like: Integral(1/(x**2 + a**2)**2, (x, -oo, oo)) = pi/(2 a**3)
    assert _same(_value(residue_integral(1 / (x**2 + a**2)**2, x, -oo, oo)), pi / (2 * a**3))
    # GR 3.222.2: Integral(x**(mu - 1)/(a + x), (x, 0, oo)) = pi a**(mu-1)/sin(mu pi); mu = 1/3, a = 2
    assert _same(_value(residue_integral(x**Rational(-2, 3) / (2 + x), x, S.Zero, oo)),
                 pi * 2**Rational(-2, 3) / sin(pi / 3))


def test_gradshteyn_ryzhik_fourier() -> None:
    # GR 3.723.2: Integral(cos(a x)/(b**2 + x**2), (x, 0, oo)) = pi exp(-a b)/(2 b)
    assert _same(_value(residue_integral(cos(a * x) / (b**2 + x**2), x, S.Zero, oo)),
                 pi * exp(-a * b) / (2 * b))
    # GR 3.723.3: Integral(x sin(a x)/(b**2 + x**2), (x, 0, oo)) = pi exp(-a b)/2
    assert _same(_value(residue_integral(x * sin(a * x) / (b**2 + x**2), x, S.Zero, oo)),
                 pi * exp(-a * b) / 2)
    # GR 3.729.1: Integral(cos(a x)/(b**2 + x**2)**2, (x, 0, oo)) = pi (1 + a b) exp(-a b)/(4 b**3)
    assert _same(_value(residue_integral(cos(a * x) / (b**2 + x**2)**2, x, S.Zero, oo)),
                 pi * (1 + a * b) * exp(-a * b) / (4 * b**3))
    # GR 3.728.1: Integral(cos(a x)/((b**2 + x**2)(c**2 + x**2)), (x, 0, oo))
    #   = pi (c e^{-ab} - b e^{-ac}) / (2 b c (c**2 - b**2)), here with b = 1, c = 2
    assert _same(_value(residue_integral(cos(a * x) / ((1 + x**2) * (4 + x**2)), x, S.Zero, oo)),
                 pi * (2 * exp(-a) - exp(-2 * a)) / (2 * 2 * 3))


def test_gradshteyn_ryzhik_logarithms() -> None:
    # GR 4.231.1 with a = 1: Integral(log(x)/(x**2 + 1), (x, 0, oo)) = 0
    assert _same(_value(residue_integral(log(x) / (x**2 + 1), x, S.Zero, oo)), S.Zero)
    # GR 4.231.8: Integral(log(x)/(x**2 + a**2), (x, 0, oo)) = pi log(a)/(2 a)
    assert _same(_value(residue_integral(log(x) / (x**2 + a**2), x, S.Zero, oo)), pi * log(a) / (2 * a))
    # GR 4.261.4 with a = 1: Integral(log(x)**2/(x**2 + 1), (x, 0, oo)) = pi**3/8
    assert _same(_value(residue_integral(log(x)**2 / (x**2 + 1), x, S.Zero, oo)), pi**3 / 8)
    # GR 4.231.14: Integral(log(x)/(x + 1)**2, (x, 0, oo)) = 0
    assert _same(_value(residue_integral(log(x) / (x + 1)**2, x, S.Zero, oo)), S.Zero)
    # GR 4.251.2-like: Integral(x**(mu-1) log(x)/(1 + x), (x, 0, oo)) = -pi**2 cos(mu pi)/sin(mu pi)**2,
    # here mu = 1/2: -pi**2 cos(pi/2)/1 = 0
    assert _same(_value(residue_integral(log(x) / (sqrt(x) * (1 + x)), x, S.Zero, oo)), S.Zero)


def test_gradshteyn_ryzhik_trigonometric() -> None:
    # GR 3.613.1: Integral(1/(1 + a cos x), (x, 0, pi)) = pi/sqrt(1 - a**2) for a**2 < 1, so
    # over a full period twice that; with a = 1/2
    assert _same(_value(residue_integral(1 / (1 + cos(x) / 2), x, S.Zero, 2 * pi)),
                 2 * pi / sqrt(1 - Rational(1, 4)))
    # GR 3.613.2: Integral(cos(n x)/(1 - 2 a cos x + a**2), (x, 0, pi)) = pi a**n/(1 - a**2), a**2 < 1;
    # n = 2, a = 1/3, over the full period
    assert _same(_value(residue_integral(cos(2 * x) / (1 - 2 * cos(x) / 3 + Rational(1, 9)), x, S.Zero, 2 * pi)),
                 2 * pi * Rational(1, 9) / (1 - Rational(1, 9)))
    # GR 3.661.?/Ahlfors 4.5.3 example 2: Integral(1/(a + cos x), (x, 0, 2 pi)) = 2 pi/sqrt(a**2 - 1), a > 1
    found = residue_integral(1 / (a + cos(x)), x, S.Zero, 2 * pi, a > 1)
    assert _same(_value(found), 2 * pi / sqrt(a**2 - 1))
    # GR 3.616.7-like: Integral(1/(a + b sin x)**2, (x, 0, 2 pi)) = 2 pi a/(a**2 - b**2)**(3/2); a = 3, b = 1
    assert _same(_value(residue_integral(1 / (3 + sin(x))**2, x, S.Zero, 2 * pi)), 6 * pi / 8**Rational(3, 2))


def test_ahlfors_examples() -> None:
    # Ahlfors 4.5.3, example 1: Integral(x**2/(x**4 + 5 x**2 + 4), (x, -oo, oo)) = pi/3
    assert _same(_value(residue_integral(x**2 / (x**4 + 5 * x**2 + 4), x, -oo, oo)), pi / 3)
    # example 3 (the keyhole): Integral(x**(-alpha)/(1 + x), (x, 0, oo)) = pi/sin(pi alpha), 0 < alpha < 1
    assert _same(_value(residue_integral(x**Rational(-1, 3) / (1 + x), x, S.Zero, oo)), pi / sin(pi / 3))
    # DLMF 5.12.3 as a Beta function: Integral(x**(a-1)/(1+x)**2, (x, 0, oo)) = B(a, 2 - a) = Gamma(a) Gamma(2 - a)
    assert _same(_value(residue_integral(x**Rational(-1, 2) / (1 + x)**2, x, S.Zero, oo)),
                 gamma(S.Half) * gamma(Rational(3, 2)))
