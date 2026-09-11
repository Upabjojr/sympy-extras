"""Tests of the Mellin–Barnes conversion and Slater's theorem."""
from __future__ import annotations

import mpmath

from sympy import (symbols, S, Rational, meijerg, exp, sqrt, pi, erf, gamma, besselk, simplify, log,
                   Piecewise, lambdify, oo)

from sympy_extras.integrals.mellin import GammaQuotient
from sympy_extras.integrals.slater import MeijerG, mellin_barnes, slater_expansion, expand_meijerg, line_conditions

z = symbols('z', positive=True)
a, b = symbols('a b', positive=True)
s = symbols('s')


def _line_integral(q: GammaQuotient, c: float) -> complex:
    """``(1/2 pi i) Integral(q(t), t)`` along ``Re t = c`` by quadrature."""
    f = lambdify(s, q.as_expr(s), 'mpmath')
    value = mpmath.quad(lambda t: f(mpmath.mpc(c, t)), [-mpmath.inf, 0, mpmath.inf])
    return complex(value / (2 * mpmath.pi))


def test_mellin_barnes_maps_the_factors() -> None:
    # Gamma(t) Gamma(1 - t) 3^(-t) on 0 < Re t < 1 is G^{1,1}_{1,1}(1/3 | 1; 1)
    q = GammaQuotient(1, [(3, -1)], [(0, S.One), (1, -S.One)], [], 0, 1)
    g = mellin_barnes(q)
    assert g is not None
    assert (g.an, g.ap, g.bm, g.bq, g.z, g.prefactor) == ((1,), (), (1,), (), Rational(1, 3), 1)
    assert (g.m, g.n, g.p, g.q, g.delta) == (1, 1, 1, 1, 1)
    # a quotient with an extra factor is not a G-function
    q = GammaQuotient(1, [], [(0, S.One)], [], 0, oo, True, s)
    assert mellin_barnes(q) is None


def test_gauss_multiplication_keeps_the_value() -> None:
    # Gamma(1/2 + t/2) Gamma(1 - t) 4^(-t) needs the multiplication formula
    # (a scale of 1/2): the value of the line integral must not change
    q = GammaQuotient(1, [(4, -1)], [(S.Half, S.Half), (1, -S.One)], [], -1, 1)
    g = mellin_barnes(q)
    assert g is not None
    assert (g.m, g.n, g.p, g.q) == (2, 1, 1, 2)
    direct = _line_integral(q, 0.3)
    expanded = expand_meijerg(g)
    assert expanded is not None
    assert abs(complex(expanded.value.evalf(20)) - direct) < 1e-10


def test_slater_expansion_closed_forms() -> None:
    assert slater_expansion(meijerg([], [], [0], [], z)) == exp(-z)
    assert slater_expansion(meijerg([1], [], [S.Half], [0], z)) == sqrt(pi) * erf(sqrt(z))
    assert slater_expansion(meijerg([1 - a], [], [0], [], z)) == gamma(a) / (z + 1)**a
    # p < q with two hypergeometric terms: G^{2,0}_{0,2}(z | 1/3, -1/3) = 2 K_{2/3}(2 sqrt z)
    value = slater_expansion(meijerg([], [], [Rational(1, 3), -Rational(1, 3)], [], z))
    assert abs(complex((value - 2 * besselk(Rational(2, 3), 2 * sqrt(z))).evalf(20, subs={z: 0.7}))) < 1e-15
    # something which is not a meijerg is returned as it is
    assert slater_expansion(exp(z)) == exp(z)


def test_p_equal_q_uses_the_assumptions() -> None:
    g = MeijerG(S.One, [1 - a], [], [S.Zero], [], z)
    # |z| < 1 known: the series in z
    inside = expand_meijerg(g, z < 1)
    assert inside is not None and inside.value == gamma(a) / (z + 1)**a
    # |z| > 1 known: the series in 1/z, the same function
    outside = expand_meijerg(g, z > 1)
    assert outside is not None
    assert simplify(outside.value - gamma(a) / (z + 1)**a) == 0
    # unknown: the two forms are recognised as the same function
    unknown = expand_meijerg(g)
    assert unknown is not None and not unknown.value.has(Piecewise)


def test_logarithmic_case_by_perturbation() -> None:
    # G^{2,1}_{2,2}(z | 0, 1; 0, 0) has coincident b's: the parameters are
    # moved apart by an irrational epsilon and the limit is taken; the
    # value is log(1 + 1/z) (the integral of (1 - exp(-x)) exp(-z x)/x)
    value = slater_expansion(meijerg([0], [1], [0, 0], [], z))
    assert not value.has(meijerg)
    assert simplify(value - log(1 + 1 / z)) == 0
    # G^{1,0}_{0,2}(z | 0; 0): a Bessel function, expanded by hyperexpand
    value = slater_expansion(meijerg([], [], [0], [0], z))
    assert not value.has(meijerg)


def test_line_conditions() -> None:
    # delta > 0 and a positive argument: converges absolutely
    g = MeijerG(S.One, [S.One], [], [S.One], [], Rational(1, 3))
    assert line_conditions(g, S.Zero, S.One) is S.true
    # delta = 0: the decay of the integrand decides
    g = MeijerG(S.One, [], [], [S.Half], [S.Zero], z)
    condition = line_conditions(g, -S.One, S.One)
    assert condition is not None
    # delta < 0: no absolutely convergent line integral
    g = MeijerG(S.One, [], [a, b], [], [S.Zero], z)
    assert line_conditions(g, -S.One, S.One) is None
