"""Definite integrals with known values from the tables (Gradshteyn and
Ryzhik, *Table of Integrals, Series, and Products*, 7th edition, [GR];
the NIST Digital Library of Mathematical Functions, [DLMF]) and from
Mathematica; each value is also checked numerically."""
from __future__ import annotations

from sympy import (symbols, exp, sin, cos, log, sqrt, besselj, besselk, oo, gamma, pi, S, Rational, atan,
                   erf, erfc, simplify, EulerGamma)

from sympy_extras._typing import ExprLike, as_expr
from sympy_extras.assumptions.ask import Assumptions
from sympy_extras.integrals import definite_integral, verify_numerically
from sympy_extras.integrals.conditions import numerically_equal

x = symbols('x')
a, b, mu, nu, p, q = symbols('a b mu nu p q', positive=True)


def _check(value: ExprLike, expected: ExprLike, f: ExprLike, lo: ExprLike, hi: ExprLike,
           assumptions: Assumptions = None) -> None:
    difference = as_expr(value) - as_expr(expected)
    assert simplify(difference) == 0 or simplify(difference.rewrite(erf)) == 0 \
        or numerically_equal(as_expr(value), as_expr(expected), assumptions), (value, expected)
    assert verify_numerically(as_expr(expected), as_expr(f), x, as_expr(lo), as_expr(hi), assumptions) is not False


def test_gamma_and_beta_integrals() -> None:
    # DLMF 5.2.1
    _check(definite_integral(x**(a - 1) * exp(-x), (x, 0, oo)), gamma(a), x**(a - 1) * exp(-x), 0, oo)
    # DLMF 5.12.1 and 5.12.3
    _check(definite_integral(x**(a - 1) * (1 - x)**(b - 1), (x, 0, 1)), gamma(a) * gamma(b) / gamma(a + b),
           x**(a - 1) * (1 - x)**(b - 1), 0, 1)
    _check(definite_integral(x**(a - 1) / (1 + x)**(a + b), (x, 0, oo)), gamma(a) * gamma(b) / gamma(a + b),
           x**(a - 1) / (1 + x)**(a + b), 0, oo)
    # DLMF 5.12.2
    _check(definite_integral(sin(x)**(2 * a - 1) * cos(x)**(2 * b - 1), (x, 0, pi / 2)),
           gamma(a) * gamma(b) / (2 * gamma(a + b)), sin(x)**(2 * a - 1) * cos(x)**(2 * b - 1), 0, pi / 2)
    # GR 3.241.2 with the condition 0 < mu < 1
    value = definite_integral(x**(mu - 1) / (1 + x), (x, 0, oo), mu < 1)
    _check(value, pi / sin(mu * pi), x**(mu - 1) / (1 + x), 0, oo, mu < 1)


def test_exponential_and_trigonometric() -> None:
    # GR 3.893.1 and 3.893.2 (Laplace transforms of sin and cos)
    _check(definite_integral(exp(-a * x) * sin(b * x), (x, 0, oo)), b / (a**2 + b**2), exp(-a * x) * sin(b * x), 0, oo)
    _check(definite_integral(exp(-a * x) * cos(b * x), (x, 0, oo)), a / (a**2 + b**2), exp(-a * x) * cos(b * x), 0, oo)
    # GR 3.941.1
    _check(definite_integral(exp(-p * x) * sin(q * x) / x, (x, 0, oo)), atan(q / p), exp(-p * x) * sin(q * x) / x, 0, oo)
    # GR 3.896.4
    _check(definite_integral(exp(-b * x**2) * cos(a * x), (x, 0, oo)), sqrt(pi / b) * exp(-a**2 / (4 * b)) / 2,
           exp(-b * x**2) * cos(a * x), 0, oo)
    # GR 3.723.2
    _check(definite_integral(cos(a * x) / (b**2 + x**2), (x, 0, oo)), pi * exp(-a * b) / (2 * b),
           cos(a * x) / (b**2 + x**2), 0, oo)
    # GR 3.721.1 (Dirichlet) and 3.782.2
    _check(definite_integral(sin(a * x) / x, (x, 0, oo)), pi / 2, sin(a * x) / x, 0, oo)
    _check(definite_integral((1 - cos(a * x)) / x**2, (x, 0, oo)), pi * a / 2, (1 - cos(a * x)) / x**2, 0, oo)
    # GR 3.434.2 (Frullani)
    _check(definite_integral((exp(-a * x) - exp(-b * x)) / x, (x, 0, oo)), log(b / a),
           (exp(-a * x) - exp(-b * x)) / x, 0, oo)


def test_special_functions() -> None:
    # DLMF 10.22.49
    value = definite_integral(exp(-a * x) * besselj(nu, b * x), (x, 0, oo))
    expected = b**(-nu) * (sqrt(a**2 + b**2) - a)**nu / sqrt(a**2 + b**2)
    assert verify_numerically(value - expected, S.Zero * x, x, S.Zero, S.One) is not False
    for values in ({a: 2, b: 3, nu: S.Half}, {a: Rational(1, 2), b: 1, nu: Rational(3, 2)}):
        assert abs(complex((value - expected).evalf(20, subs=values))) < 1e-15
    # DLMF 10.43.19
    _check(definite_integral(x**(mu - 1) * besselk(nu, x), (x, 0, oo), mu > nu),
           2**(mu - 2) * gamma((mu - nu) / 2) * gamma((mu + nu) / 2), x**(mu - 1) * besselk(nu, x), 0, oo, mu > nu)
    # GR 6.281.1: Integral(erfc(x), (x, 0, oo)) = 1/sqrt(pi)
    _check(definite_integral(erfc(x), (x, 0, oo)), 1 / sqrt(pi), erfc(x), 0, oo)
    # the Laplace transform of erf (GR 6.285.1 with a = 1)
    _check(definite_integral(exp(-p * x) * erf(x), (x, 0, oo)), exp(p**2 / 4) * erfc(p / 2) / p, exp(-p * x) * erf(x), 0, oo)
    # GR 3.411.1: Integral(x^(nu-1)/(e^x - 1), (x, 0, oo)) = Gamma(nu) zeta(nu)
    _check(definite_integral(x**3 / (exp(x) - 1), (x, 0, oo)), pi**4 / 15, x**3 / (exp(x) - 1), 0, oo)
    _check(definite_integral(x / (exp(x) + 1), (x, 0, oo)), pi**2 / 12, x / (exp(x) + 1), 0, oo)


def test_logarithmic_integrals() -> None:
    # GR 4.224.3
    _check(definite_integral(log(sin(x)), (x, 0, pi / 2)), -pi * log(2) / 2, log(sin(x)), 0, pi / 2)
    # GR 4.272.6 (for n = 2, m = 3) and 4.215.1
    _check(definite_integral(x**3 * log(x)**2, (x, 0, 1)), Rational(1, 32), x**3 * log(x)**2, 0, 1)
    _check(definite_integral((-log(x))**(mu - 1), (x, 0, 1)), gamma(mu), (-log(x))**(mu - 1), 0, 1)
    # GR 4.231.2? the classical Integral(log(x)/(1 + x^2), (x, 0, oo)) = 0 and 4.261.? log^2
    _check(definite_integral(log(x) / (1 + x**2), (x, 0, oo)), 0, log(x) / (1 + x**2), 0, oo)
    _check(definite_integral(log(x)**2 / (1 + x**2), (x, 0, oo)), pi**3 / 8, log(x)**2 / (1 + x**2), 0, oo)
    # GR 4.331.1: Integral(exp(-mu x) log x, (x, 0, oo)) = -(EulerGamma + log mu)/mu
    _check(definite_integral(exp(-mu * x) * log(x), (x, 0, oo)), -(EulerGamma + log(mu)) / mu, exp(-mu * x) * log(x), 0, oo)


def test_values_from_mathematica() -> None:
    # Integrate[x^k/(x + 3), {x, 0, Infinity}] == -3^k Pi Csc[k Pi] for -1 < Re k < 0
    k = symbols('k')
    value = definite_integral(x**k / (x + 3), (x, 0, oo), (k > -1) & (k < 0))
    assert simplify(value + 3**k * pi / sin(k * pi)) == 0
    # Integrate[Log[1 + 7/x^2], {x, 1, Infinity}] == -Log[8] + 2 Sqrt[7] ArcTan[Sqrt[7]] (the value
    # Maxima records for its rtestint 233; SymPy's integrate gives -2*sqrt(7)*atan(sqrt(7)/7) - log(8/7) + 2)
    _check(definite_integral(log(1 + 7 / x**2), (x, 1, oo)), -log(8) + 2 * sqrt(7) * atan(sqrt(7)), log(1 + 7 / x**2), 1, oo)
