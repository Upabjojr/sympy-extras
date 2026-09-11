"""Elliptic integrals with values from the tables (Byrd and Friedman
[BF], Gradshteyn and Ryzhik [GR], DLMF chapter 19), checked exactly and
by quadrature."""
from __future__ import annotations

from sympy import symbols, sqrt, S, gamma, pi, elliptic_k, elliptic_e, simplify, Rational

from sympy_extras.integrals.definite import verify_numerically
from sympy_extras.integrals.elliptic import elliptic_integral

x = symbols('x')
k = symbols('k', positive=True)


def test_lemniscate_constant() -> None:
    # DLMF 19.20.2 / GR 3.166.22: Integral(1/sqrt(1 - x**4), (x, 0, 1)) = K(1/2)/sqrt(2) = Gamma(1/4)**2/(4 sqrt(2 pi))
    found = elliptic_integral(1 / sqrt(1 - x**4), x, 0, 1)
    assert found is not None
    # SymPy writes K(1/2) through gamma(-1/4); the two forms agree numerically
    assert abs(float((found.value - gamma(Rational(1, 4))**2 / (4 * sqrt(2 * pi))).evalf(20))) < 1e-15
    assert verify_numerically(found.value, 1 / sqrt(1 - x**4), x, S.Zero, S.One) is True
    # GR 3.166.? / BF 213: Integral(x**2/sqrt(1 - x**4), (x, 0, 1)) = (2 E(1/2) - K(1/2))/sqrt(2)
    found = elliptic_integral(x**2 / sqrt(1 - x**4), x, 0, 1)
    assert found is not None
    assert simplify(found.value - (2 * elliptic_e(S.Half) - elliptic_k(S.Half)) / sqrt(2)) == 0


def test_legendre_normal_forms() -> None:
    # DLMF 19.2.8: K(m) = Integral(1/sqrt((1 - x**2)(1 - m x**2)), (x, 0, 1)) with m = k**2
    found = elliptic_integral(1 / sqrt((1 - x**2) * (1 - k**2 * x**2)), x, 0, 1, k < 1)
    assert found is not None and found.value == elliptic_k(k**2)
    # BF 233.00: Integral(1/sqrt(x (1 - x)(1 - k**2 x)), (x, 0, 1)) = 2 K(k**2)
    found = elliptic_integral(1 / sqrt(x * (1 - x) * (1 - k**2 * x)), x, 0, 1, k < 1)
    assert found is not None and found.value == 2 * elliptic_k(k**2)


def test_byrd_friedman_tables() -> None:
    # BF 235.00: Integral(1/sqrt((x - a)(x - b)(x - c)), (x, a, oo)) = 2 K((b - c)/(a - c))/sqrt(a - c)
    found = elliptic_integral(1 / sqrt(x**3 - x), x, 1, S.Infinity)
    assert found is not None
    assert simplify(found.value - 2 * elliptic_k(S.Half) / sqrt(2)) == 0
    # BF 254.00: four real roots a > b > c > d, Integral(1/sqrt((a - x)(b - x)(x - c)(x - d)), (x, c, b))
    # = 2 K(m)/sqrt((a - c)(b - d)) with m = (b - c)(a - d)/((a - c)(b - d)); here a, b, c, d = 3, 2, 1, 0
    found = elliptic_integral(1 / sqrt((3 - x) * (2 - x) * (x - 1) * x), x, 1, 2)
    assert found is not None and found.value == elliptic_k(Rational(3, 4))
    assert verify_numerically(found.value, 1 / sqrt((3 - x) * (2 - x) * (x - 1) * x), x, S.One, S(2)) is True
    # the value is 2.1565156... (quadrature at 20 digits: 2.15651564749791)
    assert abs(float(found.value.evalf(15)) - 2.15651564749791) < 1e-10
