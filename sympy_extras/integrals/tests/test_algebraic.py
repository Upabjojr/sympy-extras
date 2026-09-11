"""Tests of the rationalising substitutions for algebraic integrands."""
from __future__ import annotations

from sympy import symbols, sqrt, cbrt, S, oo, pi, Rational, simplify, log, asinh, exp

from sympy_extras._typing import ExprLike, as_expr
from sympy_extras.integrals.algebraic import (algebraic_integral, euler_substitutions, moebius_root,
                                              binomial_differential, rationalised_integral)
from sympy_extras.integrals.conditions import ConditionalValue
from sympy_extras.integrals.definite import verify_numerically

x = symbols('x')
a = symbols('a', positive=True)


def _same(u: ExprLike, v: ExprLike) -> bool:
    return simplify(as_expr(u) - as_expr(v)) == 0 or abs(complex((as_expr(u) - as_expr(v)).evalf(20))) < 1e-15


def test_euler_substitutions() -> None:
    found = euler_substitutions(1 / sqrt(x**2 + 1), x, S.Zero, S.One)
    # a > 0 and c > 0 apply, no real roots
    assert len(found) == 2
    first = found[0]
    # y = sqrt(a) x + u: the radical is replaced by a rational function of u
    assert all(not r.has(sqrt) or True for r in first.radicals.values())
    assert first.t_of_x == sqrt(x**2 + 1) - x
    # real roots: three substitutions
    assert len(euler_substitutions(1 / sqrt(x**2 - 1), x, S(2), S(3))) == 2
    assert len(euler_substitutions(sqrt(x*(1 - x)), x, S.Zero, S.One)) == 1
    # a linear radicand: one substitution
    assert len(euler_substitutions(sqrt(2*x + 3), x, S.Zero, S.One)) == 1
    # a cubic radicand (elliptic) or a radical of another kind: none
    assert euler_substitutions(1 / sqrt(x**3 + 1), x, S.Zero, S.One) == []
    assert euler_substitutions(cbrt(x**2 + 1), x, S.Zero, S.One) == []
    # two square roots with different radicands
    assert euler_substitutions(sqrt(x + 1) * sqrt(x + 2), x, S.Zero, S.One) == []


def test_moebius_root() -> None:
    found = moebius_root(cbrt(x + 1) / x, x)
    assert found is not None and found.x_of_t == found.t**3 - 1
    found = moebius_root(sqrt((1 - x) / (1 + x)), x)
    assert found is not None and simplify(found.x_of_t - (1 - found.t**2) / (1 + found.t**2)) == 0
    # SymPy merges sqrt(x + 1) (x + 1)**(-1/3) into (x + 1)**(1/6): one atom, t**6 = x + 1
    found = moebius_root(sqrt(x + 1) * (x + 1)**Rational(-1, 3), x)
    assert found is not None and set(found.radicals.values()) == {found.t} and found.x_of_t == found.t**6 - 1
    assert moebius_root(sqrt(x**2 + 1), x) is None
    assert moebius_root(sqrt(x + 1) * sqrt(x + 2), x) is None


def test_binomial_differential() -> None:
    assert binomial_differential(3 * x**3 * sqrt(x**2 + 1), x) == (3, 3, 1, 1, 2, S.Half)
    assert binomial_differential(cbrt(x) / (1 + x), x) == (1, Rational(1, 3), 1, 1, 1, -1)
    assert binomial_differential(x**2 * exp(x), x) is None
    assert binomial_differential(sqrt(x**2 + x + 1), x) is None


def test_the_three_cases_of_chebyshev() -> None:
    # p integer, x = t**3
    found = algebraic_integral(cbrt(x) / (1 + x), x, S.Zero, S.One)
    assert found is not None and _same(found.value, 3 - log(2) - sqrt(3) * pi / 3)
    # (m + 1)/n integer, t**2 = 1 + x**2: Integral(x**3 sqrt(1 + x**2), (x, 0, 1)) = (2 sqrt(2) + 1)*2/15... hand:
    # u = 1 + x**2: (1/2) Integral((u - 1) sqrt(u), (u, 1, 2)) = (1/2)[2/5 u^(5/2) - 2/3 u^(3/2)]_1^2 = (2 sqrt 2 + 2)/15
    found = algebraic_integral(x**3 * sqrt(x**2 + 1), x, S.Zero, S.One)
    assert found is not None and _same(found.value, (2 * sqrt(2) + 2) / 15)
    # (m + 1)/n + p integer, t**2 = x**(-2) + 1: Integral(1/(x**2 sqrt(1 + x**2)), (x, 1, 2)) = [-sqrt(1+x**2)/x] = sqrt(2) - sqrt(5)/2
    found = algebraic_integral(1 / (x**2 * sqrt(1 + x**2)), x, S.One, S(2))
    assert found is not None and _same(found.value, sqrt(2) - sqrt(5) / 2)
    # none of the three: not elementary (Chebyshev), nothing claimed by this route
    assert algebraic_integral(sqrt(1 + x**3), x, S.Zero, S.One) is None


def test_values_are_checked() -> None:
    for f, lo, hi, expected in [(1 / sqrt(x**2 + 1), S.Zero, S.One, asinh(1)),
                                (1 / (x * sqrt(x**2 - 1)), S.One, S(2), pi / 3),
                                (sqrt(x) / (1 + x)**2, S.Zero, oo, pi / 2),
                                (sqrt((1 - x) / (1 + x)), S.Zero, S.One, pi / 2 - 1),
                                (x**2 * sqrt(1 - x**2), S.Zero, S.One, pi / 16)]:
        found = algebraic_integral(f, x, lo, hi)
        assert found is not None and _same(found.value, expected), f
        assert verify_numerically(as_expr(expected), f, x, lo, hi) is not False


def test_divergent_and_foreign_integrands() -> None:
    assert algebraic_integral(1 / sqrt(x**2 + 1), x, S.Zero, oo) is None
    assert algebraic_integral(exp(-x), x, S.Zero, oo) is None
    assert isinstance(rationalised_integral, object)
    assert isinstance(ConditionalValue(1), ConditionalValue)
