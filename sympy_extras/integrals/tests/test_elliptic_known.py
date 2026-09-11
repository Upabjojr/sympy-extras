"""Elliptic integrals with values from the tables (Byrd and Friedman
[BF], Gradshteyn and Ryzhik [GR], DLMF chapter 19), checked exactly and
by quadrature."""
from __future__ import annotations

import random

from sympy import symbols, sqrt, S, gamma, pi, elliptic_k, elliptic_e, simplify, Rational, oo, Expr

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


def _agree(found: Expr, reference: Expr, samples: int = 4) -> None:
    """``found == reference`` at random values of the parameters."""
    rng = random.Random(str(reference))
    for _ in range(samples):
        values = {}
        for symbol in sorted(reference.free_symbols, key=str):
            values[symbol] = Rational(rng.randint(1, 40), rng.randint(1, 10))
        difference = (found - reference).subs(values).evalf(20)
        assert abs(complex(difference)) < 1e-15, (values, difference)


def test_byrd_friedman_complex_roots() -> None:
    # BF 241.00: Integral(1/sqrt((a - x)(x - b)((x - p)**2 + q**2)), (x, b, a)) = 2 g K(m)
    # with A**2 = (a - p)**2 + q**2, B**2 = (b - p)**2 + q**2, g = 1/sqrt(A B),
    # m = ((a - b)**2 - (A - B)**2)/(4 A B)
    b, d, p, q = symbols('b d p q', positive=True)
    a = b + d
    A, B = sqrt((a - p)**2 + q**2), sqrt((b - p)**2 + q**2)
    m = ((a - b)**2 - (A - B)**2) / (4 * A * B)
    f = 1 / sqrt((a - x) * (x - b) * ((x - p)**2 + q**2))
    found = elliptic_integral(f, x, b, a)
    assert found is not None
    _agree(found.value, 2 * elliptic_k(m) / sqrt(A * B))
    assert verify_numerically(found.value, f, x, b, a, samples=3) is True
    # BF 240.00: Integral(1/sqrt((x - a)((x - p)**2 + q**2)), (x, a, oo)) = 2 K(m)/sqrt(A),
    # m = (A - (a - p))/(2 A)
    a = symbols('a', positive=True)
    A = sqrt((a - p)**2 + q**2)
    f = 1 / sqrt((x - a) * ((x - p)**2 + q**2))
    found = elliptic_integral(f, x, a, oo)
    assert found is not None
    _agree(found.value, 2 * elliptic_k((A - (a - p)) / (2 * A)) / sqrt(A))
    assert verify_numerically(found.value, f, x, a, oo, samples=3) is True
    # and on the other side of the root, m = (A + (a - p))/(2 A)
    f = 1 / sqrt((a - x) * ((x - p)**2 + q**2))
    found = elliptic_integral(f, x, -oo, a)
    assert found is not None
    _agree(found.value, 2 * elliptic_k((A + (a - p)) / (2 * A)) / sqrt(A))


def test_dlmf_complex_roots() -> None:
    # DLMF 5.12.3 through the beta function: Integral(1/sqrt(1 + x**4), (x, 0, oo))
    # = B(1/4, 1/4)/4 = Gamma(1/4)**2/(4 sqrt(pi)); the cubic
    # Integral(1/sqrt(1 + x**3), (x, 0, oo)) = B(1/3, 1/6)/3 = Gamma(1/3) Gamma(1/6)/(3 sqrt(pi)),
    # and Integral(1/sqrt(1 + x**3), (x, -1, 0)) is half of it
    found = elliptic_integral(1 / sqrt(1 + x**4), x, 0, oo)
    assert found is not None
    assert abs(float((found.value - gamma(Rational(1, 4))**2 / (4 * sqrt(pi))).evalf(20))) < 1e-15
    cubic = gamma(Rational(1, 3)) * gamma(Rational(1, 6)) / (3 * sqrt(pi))
    found = elliptic_integral(1 / sqrt(1 + x**3), x, 0, oo)
    assert found is not None and abs(float((found.value - cubic).evalf(20))) < 1e-15
    found = elliptic_integral(1 / sqrt(1 + x**3), x, -1, 0)
    assert found is not None and abs(float((found.value - cubic / 2).evalf(20))) < 1e-15
    found = elliptic_integral(1 / sqrt(1 + x**3), x, -1, oo)
    assert found is not None and abs(float((found.value - 3 * cubic / 2).evalf(20))) < 1e-15
    # the quartic of the task: 2 K(m)/sqrt(A B) with A = sqrt(10), B = sqrt(5), m = (2 + sqrt(2))/4
    found = elliptic_integral(1 / sqrt((x**2 + 1) * (x + 2) * (3 - x)), x, -2, 3)
    assert found is not None
    assert simplify(found.value - 2 * elliptic_k((2 + sqrt(2)) / 4) / sqrt(sqrt(50))) == 0
    assert abs(float(found.value.evalf(20)) - 1.8051605293435436544) < 1e-15
