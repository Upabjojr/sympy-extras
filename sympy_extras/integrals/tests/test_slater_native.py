"""Tests of the native Slater module (:mod:`sympy_extras.integrals.slater`)
on the objects of the core: the Mellin–Barnes conversion against the line
integral by quadrature (mpmath, an independent source), the expansions
against mpmath's Meijer G-function, and the agreement with the SymPy face;
the integrators test the face."""
from __future__ import annotations

import mpmath

from sympy_extras.core import Integer, Rational, S, Symbol, oo
from sympy_extras.core.evalf import numeric_value
from sympy_extras.core.expr import Expr
from sympy_extras.core.logic import true
from sympy_extras.functions import Hyper, MeijerG as MeijerGNode, gamma, meijerg
from sympy_extras.integrals.marichev import bessel_k_forms
from sympy_extras.integrals.mellin import GammaQuotient
from sympy_extras.integrals.slater import MeijerG, expand_meijerg, line_conditions, mellin_barnes, slater_expansion

z = Symbol('z', positive=True)
a = Symbol('a', positive=True)


def _value(e: Expr) -> complex:
    found = numeric_value(e, 100)
    assert found is not None, e
    return complex(found)


def _meijerg(an: list[float], ap: list[float], bm: list[float], bq: list[float]) -> complex:
    with mpmath.workdps(30):
        return complex(mpmath.meijerg([an, ap], [bm, bq], mpmath.mpf(3) / 5))


def test_the_container() -> None:
    g = MeijerG(2, [0], [], [1], [S.Half], z)
    assert (g.m, g.n, g.p, g.q, g.delta) == (1, 1, 1, 2, S.Half)
    assert g.as_expr() == meijerg([0], [], [1], [S.Half], z)
    r = g.reflected()
    assert (r.an, r.ap, r.bm, r.bq, r.z) == ((0,), (S.Half,), (1,), (), 1 / z)


def test_mellin_barnes_against_the_line_integral() -> None:
    # Gamma(1/2 + t/2) Gamma(1 - t) 4^(-t) on -1 < Re t < 1: the scale 1/2
    # needs Gauss's multiplication formula; the value must be the integral
    q = GammaQuotient(1, [(4, -1)], [(S.Half, S.Half), (1, -S.One)], [], -1, 1)
    g = mellin_barnes(q)
    assert g is not None
    s = Symbol('s')
    assert q.as_expr(s) == 4**(-s) * gamma(S.Half + s / 2) * gamma(1 - s)

    def f(t: mpmath.mpf) -> mpmath.mpc:
        u = mpmath.mpc(0.25, t)
        return mpmath.power(4, -u) * mpmath.gamma(0.5 + u / 2) * mpmath.gamma(1 - u)

    line = complex(mpmath.quad(f, [-mpmath.inf, 0, mpmath.inf]) / (2 * mpmath.pi))
    expanded = expand_meijerg(g)
    assert expanded is not None
    assert abs(_value(expanded.value) - line) < 1e-8
    # a quotient with an extra factor is not a G-function
    assert mellin_barnes(GammaQuotient(1, [], [(0, S.One)], [], 0, oo, True, s)) is None


def test_expansions_against_mpmath() -> None:
    w = Rational(3, 5)
    cases: list[tuple[Expr, tuple[list[float], list[float], list[float], list[float]]]] = [
        (meijerg([], [], [0], [], w), ([], [], [0], [])),
        (meijerg([1], [], [S.Half], [0], w), ([1], [], [0.5], [0])),
        (meijerg([Rational(1, 3)], [], [0], [], w), ([1 / 3], [], [0], [])),
        (meijerg([], [], [0, S.Half], [], w), ([], [], [0, 0.5], [])),
        # a logarithmic case: b's differing by an integer
        (meijerg([], [], [0, 0], [], w), ([], [], [0, 0], [])),
        (meijerg([S.Half], [], [0], [S.Half], w), ([0.5], [], [0], [0.5])),
    ]
    for g, params in cases:
        # the logarithmic case leaves d/dnu I_nu at nu = 0, which is -K_0
        expanded = bessel_k_forms(slater_expansion(g))
        assert not expanded.has(MeijerGNode, Hyper), g
        assert abs(_value(expanded) - _meijerg(*params)) < 1e-12, g


def test_p_equal_q_cases() -> None:
    # 1F0 on both sides of |z| = 1: the same formula, (1 + z)**(-a)*gamma(a)
    g = meijerg([1 - a], [], [0], [], z)
    assert slater_expansion(g) == slater_expansion(g, z < 1)
    # an expression which is not a G-function is returned as it is
    assert slater_expansion(z + 1) == z + 1


def test_line_conditions() -> None:
    assert line_conditions(MeijerG(1, [], [], [0], [], z), Integer(0), oo) is true
    # delta < 0: the integral along a line diverges
    assert line_conditions(MeijerG(1, [], [0, 0], [0], [], z), Integer(0), oo) is None


def test_the_face_agrees() -> None:
    from sympy_extras._numeric_sympy import sympy_expr
    from sympy_extras.integrals import slater_sympy as face
    for g in [meijerg([1 - a], [], [0], [], z), meijerg([1], [], [S.Half], [0], z)]:
        assert sympy_expr(slater_expansion(g)) == face.slater_expansion(sympy_expr(g))
