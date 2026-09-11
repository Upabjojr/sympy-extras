"""Laplace transforms from the tables (Abramowitz and Stegun 29.3, [AS];
Erdélyi's *Tables of Integral Transforms* I, [E]), each checked
numerically against the integral."""
from __future__ import annotations

from sympy import symbols, sin, cos, exp, oo, Abs, pi, atan, log, coth, Integral, S, Rational, simplify, Heaviside, besselj

from sympy_extras._typing import ExprLike, as_expr
from sympy_extras.assumptions.ask import Assumptions

from sympy_extras.integrals.conditions import numerically_equal
from sympy_extras.integrals.definite import verify_numerically
from sympy_extras.integrals.laplace import laplace_rules

t, u = symbols('t u')
s, a, b = symbols('s a b', positive=True)


def _check(f: ExprLike, expected: ExprLike, assumptions: Assumptions = None) -> None:
    f_, expected_ = as_expr(f), as_expr(expected)
    found = laplace_rules(f_, t, s, assumptions)
    assert found is not None, f
    assert simplify(found.value - expected_) == 0 or numerically_equal(found.value, expected_, assumptions), (found, expected)
    assert verify_numerically(expected_, f_ * exp(-s * t), t, S.Zero, oo, assumptions) is not False


def test_pairs_with_division_by_t() -> None:
    # AS 29.3.99: L[sin(a t)/t] = atan(a/s)
    _check(sin(a * t) / t, atan(a / s))
    # AS 29.3.98: L[(cos(a t) - cos(b t))/t] = log((s^2 + b^2)/(s^2 + a^2))/2
    _check((cos(a * t) - cos(b * t)) / t, log((s**2 + b**2) / (s**2 + a**2)) / 2)
    # AS 29.3.97: L[(exp(-a t) - exp(-b t))/t] = log((s + b)/(s + a))
    _check((exp(-a * t) - exp(-b * t)) / t, log((s + b) / (s + a)))
    # E I 4.7 (28): L[(1 - cos(a t))/t^2] = a atan(a/s) - s log(1 + a^2/s^2)/2
    _check((1 - cos(a * t)) / t**2, a * atan(a / s) - s * log(1 + a**2 / s**2) / 2)


def test_periodic_and_shifted_pairs() -> None:
    # AS 29.3.? (the full-wave rectified sine): L[|sin(a t)|] = a coth(pi s/(2 a))/(s^2 + a^2)
    _check(Abs(sin(a * t)), a * coth(pi * s / (2 * a)) / (s**2 + a**2))
    _check(Abs(sin(t)), coth(pi * s / 2) / (s**2 + 1))
    # AS 29.2.12: L[f(t - a) theta(t - a)] = exp(-a s) F(s) with f = cos, a = 1
    _check(Heaviside(t - 1) * cos(t), (s * cos(1) - sin(1)) * exp(-s) / (s**2 + 1))
    # AS 29.2.13 / 29.3.5: L[t^2 exp(a t)] = 2/(s - a)^3 for s > a
    _check(t**2 * exp(a * t), 2 / (s - a)**3, s > a)


def test_convolution_pairs() -> None:
    # AS 29.2.8: L[Integral(sin(t - u) u du)] = L[sin] L[t] = 1/((s^2 + 1) s^2)
    _check(Integral(sin(t - u) * u, (u, 0, t)), 1 / (s**2 * (s**2 + 1)))
    # AS 29.2.7: L[Integral(f, (u, 0, t))] = F/s with f = cos: (s/(s^2 + 1))/s
    _check(Integral(cos(u), (u, 0, t)), 1 / (s**2 + 1))


def test_multiplication_pairs() -> None:
    # AS 29.3.53 with the multiplication rule: L[t J_0(t)] = s/(s^2 + 1)^(3/2)
    _check(t * besselj(0, t), s / (s**2 + 1)**Rational(3, 2))
