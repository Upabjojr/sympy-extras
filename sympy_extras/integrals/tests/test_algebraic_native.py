"""Tests of the rationalising substitutions for algebraic integrands on the
objects of the core (:mod:`sympy_extras.integrals.algebraic`): each
substitution checked by substitution into the radicals and each
antiderivative by differentiation, at random points (independent of the
code's own forms); the values of the definite integrals against
Gradshteyn–Ryzhik 2.25 and hand computations."""
from __future__ import annotations

from sympy_extras.core import Integer, Rational, S, Symbol, oo, pi, sqrt
from sympy_extras.core.expr import Expr
from sympy_extras.core.symbol import Dummy
from sympy_extras.functions import exp, hyper, log
from sympy_extras.integrals.algebraic import (BinomialDifferential, Rationalisation, algebraic_integral,
                                              binomial_differential, binomial_exponents,
                                              binomial_hypergeometric_antiderivative, chebyshev_elementary,
                                              euler_substitution, euler_substitutions, moebius_root,
                                              rationalised_integral)
from sympy_extras.integrals.conditions import ConditionalValue, numerically_equal

x = Symbol('x')
a = Symbol('a', positive=True)


def _radicals_hold(s: Rationalisation, at: list[Expr]) -> bool:
    """Every radical of the substitution, evaluated at ``x = X(t)``, equals
    its replacement, at the values ``t`` of ``at`` (in the range of the
    map, where the radicands are positive)."""
    for value in at:
        point = s.x_of_t.subs(s.t, value)
        for radical, replacement in s.radicals.items():
            left = radical.subs(x, point)
            right = replacement.subs(s.t, value)
            if abs(complex((left - right).evalf(30))) > 1e-20:
                return False
    return True


def test_euler_substitutions() -> None:
    found = euler_substitutions(1 / sqrt(x**2 + 1), x, S.Zero, S.One)
    # a > 0 and c > 0 apply, no real roots
    assert len(found) == 2 and found[0].t_of_x == sqrt(x**2 + 1) - x
    assert _radicals_hold(found[0], [Rational(1, 2), Rational(3, 4)])
    assert _radicals_hold(found[1], [Rational(1, 3), Rational(1, 5)])
    # real roots 1 and -1: the substitution y = (x - r1) u among them
    found = euler_substitutions(1 / sqrt(x**2 - 1), x, Integer(2), Integer(3))
    assert len(found) == 2 and _radicals_hold(found[1], [Rational(1, 2)])
    first = euler_substitution(1 / sqrt(x**2 + 1), x, 0, 1)
    assert first is not None and first.t_of_x == sqrt(x**2 + 1) - x
    # a cube root, or a radicand of degree three: none
    assert euler_substitutions((x**2 + 1)**Rational(1, 3), x, 0, 1) == []
    assert euler_substitutions(sqrt(x**3 + 1), x, 0, 1) == []


def test_the_dummies_have_names_of_their_own() -> None:
    # the typed polynomials identify their symbols by name: a dummy 't'
    # would be taken for a parameter 't' of the integrand
    t = Symbol('t')
    s = moebius_root(sqrt(x + t) / x, x)
    assert s is not None and isinstance(s.t, Dummy) and s.t.name != 't'
    u = Symbol('u')
    found = euler_substitutions(1 / sqrt(x**2 + u**2), x, 0, 1, [u > 0])
    assert found and all(r.t.name != 'u' for r in found)


def test_moebius_root() -> None:
    s = moebius_root((x + 1)**Rational(1, 3) / x, x)
    assert s is not None and s.x_of_t == s.t**3 - 1 and s.t_of_x == (x + 1)**Rational(1, 3)
    # a Möbius function and its reciprocal under the roots: one t**6
    s = moebius_root(sqrt((1 - x) / (1 + x)) + ((1 + x) / (1 - x))**Rational(1, 3), x)
    assert s is not None and _radicals_hold(s, [Rational(1, 2), Rational(2, 3)])
    # t**n = M(x(t)): the map inverts the Möbius function
    assert numerically_equal((s.t_of_x**6).subs(x, s.x_of_t), s.t**6, [s.t > 0])
    assert moebius_root(sqrt(x**2 + 1), x) is None and moebius_root(exp(x), x) is None


def test_binomial_differential() -> None:
    assert binomial_differential(3 * x**3 * sqrt(x**2 + 1), x) == (3, 3, 1, 1, 2, Rational(1, 2))
    assert binomial_differential(x**2 * (x + 1)**2, x) is None       # a polynomial
    assert binomial_differential(sqrt(x**2 + x + 1), x) is None      # a trinomial


def test_binomial_exponents_and_chebyshev() -> None:
    # Chebyshev's criterion: elementary iff p, (m + 1)/n or (m + 1)/n + p is an integer
    cases = [(x**3 * sqrt(x**2 + 1), True), (sqrt(x**3 + 1), False), (1 / sqrt(1 - x**4), False),
             (x / sqrt(1 + x**4), True), (sqrt(1 + 1 / x**2), True), (x**Rational(1, 3) * (1 + x)**2, True)]
    for f, elementary in cases:
        binomial = binomial_exponents(f, x)
        assert binomial is not None, f
        assert chebyshev_elementary(binomial) is elementary, f
    found = binomial_exponents(sqrt(x**7 / (1 - 5 * x**2)), x)
    assert isinstance(found, BinomialDifferential)
    assert (found.m, found.a, found.b, found.n, found.p) == (Rational(7, 2), 1, -5, 2, Rational(-1, 2))
    # two binomials, or a vanishing coefficient: none
    assert binomial_exponents(sqrt(1 + x**2) * sqrt(1 - x**2), x) is None


def _derivative_is(F: Expr, f: Expr, points: list[Expr]) -> bool:
    """``F' == f`` at the points (high precision)."""
    dF = F.diff(x)
    for p in points:
        left = complex(dF.subs(x, p).evalf(30))
        right = complex(f.subs(x, p).evalf(30))
        if abs(left - right) > 1e-15 * (1 + abs(right)):
            return False
    return True


def test_hypergeometric_antiderivatives() -> None:
    F = binomial_hypergeometric_antiderivative(sqrt(x**3 + 1), x)
    assert F == x * hyper([-Rational(1, 2), Rational(1, 3)], [Rational(4, 3)], -x**3)
    assert _derivative_is(F, sqrt(x**3 + 1), [Rational(1, 3), Rational(-1, 2), Rational(7, 10)])
    # the case in 1/z beyond z = 1, joined at -1/sqrt(5)
    f = sqrt(x**7 / (1 - 5 * x**2))
    F = binomial_hypergeometric_antiderivative(f, x)
    assert F is not None
    assert _derivative_is(F, f, [Rational(1, 5), Rational(2, 5), Rational(-3, 5), Rational(-1)])
    f = 1 / sqrt(1 - x**4)
    F = binomial_hypergeometric_antiderivative(f, x)
    assert F is not None and _derivative_is(F, f, [Rational(1, 4), Rational(-1, 2)])
    # elementary: none
    assert binomial_hypergeometric_antiderivative(x**3 * sqrt(x**2 + 1), x) is None
    # a parameter: a case for z > 1, each form an antiderivative where it holds
    c = Symbol('c')
    F = binomial_hypergeometric_antiderivative(sqrt(1 + c * x**3), x)
    assert F is not None and F.has(c)


def test_definite_integrals() -> None:
    # the rational integral by definite's integrator (through the face)
    assert algebraic_integral(1 / (x * sqrt(x**2 - 1)), x, 1, 2) == ConditionalValue(pi / 3)
    found = algebraic_integral(sqrt(x) / (1 + x)**2, x, 0, oo)
    assert found is not None and numerically_equal(found.value, pi / 2)
    # Gradshteyn-Ryzhik 2.25: log(x + sqrt(x**2 + a**2)) over (0, a)
    found = algebraic_integral(1 / sqrt(x**2 + a**2), x, 0, a)
    assert found is not None and numerically_equal(found.value, log(1 + sqrt(2)))
    s = moebius_root(sqrt(x) / (1 + x), x)
    assert s is not None
    found = rationalised_integral(sqrt(x) / (1 + x), x, 0, 1, s)
    assert found is not None and numerically_equal(found.value, 2 - pi / 2)
    assert algebraic_integral(exp(x), x, 0, 1) is None
