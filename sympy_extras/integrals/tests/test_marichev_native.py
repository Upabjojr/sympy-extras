"""Tests of the native Marichev–Adamchik module
(:mod:`sympy_extras.integrals.marichev`) on the objects of the core: the
integrals against mpmath's quadrature (an independent source), the
rewritings of the values, and the agreement with the SymPy face; the
integrators test the face."""
from __future__ import annotations

from typing import Callable

import mpmath

from sympy_extras.core import I, Integer, Rational, S, Symbol, pi, sqrt
from sympy_extras.core.evalf import numeric_value
from sympy_extras.core.expr import Expr
from sympy_extras.functions import (Sin, besseli, besselj, besselk, cos, elliptic_k, exp, exp_polar, gamma, log,
                                    lowergamma, sin)
from sympy_extras.integrals.marichev import (bessel_k_forms, checked_tidy, evaluate_quotient, exponential_form,
                                             integrate_product, mellin_integrate, polar_elliptic, polar_lowergamma,
                                             real_logarithms, reduce_positive_powers, right_half_plane_powers, tidy)
from sympy_extras.integrals.mellin import GammaQuotient, decompose_integrand

x, k = Symbol('x'), Symbol('k')
a, b = Symbol('a', positive=True), Symbol('b', positive=True)


def _value(e: Expr) -> complex:
    found = numeric_value(e, 100)
    assert found is not None, e
    return complex(found)


def _quad(f: Callable[[mpmath.mpf], mpmath.mpf], lower: float = 0, upper: float = mpmath.inf) -> float:
    return float(mpmath.quad(f, [lower, 1, upper] if upper == mpmath.inf else [lower, upper]))


def test_integrals_against_quadrature() -> None:
    cases: list[tuple[Expr, Callable[[mpmath.mpf], mpmath.mpf], str]] = [
        (x**2 * exp(-3 * x), lambda t: t**2 * mpmath.exp(-3 * t), ''),
        (exp(-2 * x) * sin(3 * x), lambda t: mpmath.exp(-2 * t) * mpmath.sin(3 * t), ''),
        (exp(-x) * besselj(Integer(0), x), lambda t: mpmath.exp(-t) * mpmath.besselj(0, t), ''),
        (exp(-x**2) * log(x), lambda t: mpmath.exp(-t**2) * mpmath.log(t), ''),
        (sqrt(x) / (x + 3), lambda t: 0, 'divergent'),
        (x**Rational(-1, 2) / (x + 3), lambda t: t**-0.5 / (t + 3), ''),
        (log(x)**2, lambda t: mpmath.log(t)**2, 'lower'),
        (x * exp(-x), lambda t: t * mpmath.exp(-t), 'upper'),
        # the bug: refused, the expansion 1/sqrt(1 - polar_lift(w)) of a 1F0
        # left a polar number which unpolarify does not remove
        (exp(-x) * cos(2 * x) * besselj(Integer(0), x), lambda t: mpmath.exp(-t) * mpmath.cos(2 * t) * mpmath.besselj(0, t), ''),
    ]
    for f, g, kind in cases:
        cutoff = kind if kind in ('lower', 'upper') else None
        found = mellin_integrate(f, x, None, cutoff)
        if kind == 'divergent':
            # the strip of x**(1/2)/(x + 3) does not contain 3/2
            assert found is None, f
            continue
        assert found is not None and found.condition is S.true, f
        if kind == 'lower':
            expected = float(mpmath.quad(g, [0, 1]))
        elif kind == 'upper':
            expected = float(mpmath.quad(g, [1, mpmath.inf]))
        else:
            expected = float(mpmath.quad(g, [0, 1, mpmath.inf]))
        assert abs(_value(found.value) - expected) < 1e-8 * (1 + abs(expected)), f


def test_conditions_and_assumptions() -> None:
    found = mellin_integrate(x**k / (x + 3), x)
    assert found is not None and found.value == -3**k * pi / sin(pi * k)
    assert found.condition == ((k > -1) & (k < 0))
    found = mellin_integrate(x**k / (x + 3), x, [k > -1, k < 0])
    assert found is not None and found.condition is S.true
    assert mellin_integrate(x**k / (x + 3), x, k > 1) is None
    # the bug: the condition |4/a**2| != 1 was kept, the numerical
    # exp_polar(I*pi) (-1 + 6e-28*I) making elliptic_k(4*exp_polar(I*pi)/a**2)
    # look non-real at the sample, so that the two sides of |z| = 1 stayed apart
    found = mellin_integrate(besselj(Integer(0), x)**2 * exp(-a * x), x)
    assert found is not None and found.condition is S.true and found.value == 2 * elliptic_k(-4 / a**2) / (pi * a)
    # the bug: x**(1/3)/(1 + x) over (0, 1) was left as
    # -log(2) + 3 - (-1)**(2/3)*log(1 - (-1)**(1/3)) + (-1)**(1/3)*log(1 + (-1)**(2/3))
    found = mellin_integrate(x**Rational(1, 3) / (1 + x), x, None, 'lower')
    assert found is not None and found.value == 3 - log(2) - sqrt(3) * pi / 3
    # analytic regularisation: gamma(-1/2) beyond the strip
    found = mellin_integrate(x**Rational(-3, 2) * exp(-x), x, None, None, True)
    assert found is not None and found.value == -2 * sqrt(pi)


def test_the_pieces() -> None:
    assert evaluate_quotient(GammaQuotient(1, [], [(0, 1)], [(1, 1)], 0), Integer(2)) == Rational(1, 2)
    product = decompose_integrand(x**2 * exp(-x), x)
    assert product is not None
    found = integrate_product(product)
    assert found is not None and found.value == 2
    assert reduce_positive_powers(sqrt(1 + x), x) == x / sqrt(x + 1) + 1 / sqrt(x + 1)
    form = exponential_form(exp(-x) * sin(x), x)
    assert not form.has(Sin)
    assert abs(_value(form.subs(x, Rational(1, 3)) - (exp(-x) * sin(x)).subs(x, Rational(1, 3)))) < 1e-25
    c = Symbol('c')
    assert real_logarithms(I * c**2 * (log(c**2) - log(-c**2)), c < 0) == pi * c**2
    assert right_half_plane_powers(sqrt((a + I * b)**2) / (a + I * b)) == 1
    assert polar_elliptic(elliptic_k(4 * exp_polar(I * pi) / a**2)) == elliptic_k(-4 / a**2)
    w = Symbol('w', positive=True)
    assert not polar_lowergamma(lowergamma(Rational(1, 3), w * exp_polar(I * pi))).has(exp_polar(I * pi))
    n, z = Symbol('n'), Symbol('z')
    assert bessel_k_forms(pi * (besseli(-n, z) - besseli(n, z)) / sin(pi * n)) == 2 * besselk(n, z)
    assert tidy(gamma(a + 1) / gamma(a)) == a
    assert checked_tidy(sqrt(a**2) + log(a * exp_polar(Integer(0)))) == a + log(a)


def test_the_face_agrees() -> None:
    import sympy
    from sympy_extras._numeric_sympy import sympy_expr
    from sympy_extras.assumptions.ask_sympy import sympy_boolean
    from sympy_extras.integrals import marichev_sympy as face
    sx = sympy.Symbol('x')
    for f in [exp(-a * x) * sin(b * x), x**k / (x + 3), exp(-x**2) * log(x), exp(-x) * besselj(Integer(0), x)]:
        native = mellin_integrate(f, x)
        assert native is not None
        theirs = face.mellin_integrate(sympy_expr(f), sx)
        assert theirs is not None
        assert sympy_expr(native.value) == theirs.value and sympy_boolean(native.condition) == theirs.condition

