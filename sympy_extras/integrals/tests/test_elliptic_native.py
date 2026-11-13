"""Tests of the native reduction to Legendre's elliptic integrals, on the
objects of the core (the values against mpmath's quadrature)."""
from __future__ import annotations

from typing import Callable, Optional

import mpmath

from sympy_extras.core import Integer, Rational, S, Symbol, oo, pi, sqrt
from sympy_extras.core.evalf import numeric_value
from sympy_extras.core.expr import Expr
from sympy_extras.functions import elliptic_e, elliptic_f, elliptic_k, elliptic_pi, sin
from sympy_extras.integrals.elliptic import (Reduction, _ConstantField, _partial_fractions, elliptic_integral,
                                             legendre_reduction, radicand, real_roots)
from sympy_extras.polys.cad.indexedroot import IndexedRoot

x, theta = Symbol('x'), Symbol('theta')
k = Symbol('k', positive=True)


def _value(e: Expr) -> complex:
    found = numeric_value(e, 80)
    assert found is not None, e
    return complex(found)


def _quadrature(f: Callable[[mpmath.mpf], mpmath.mpf], a: float, b: float) -> float:
    with mpmath.workdps(25):
        return float(mpmath.quad(f, [a, b]))


def _integral(f: Expr, a: Expr, b: Expr) -> Expr:
    found = elliptic_integral(f, x, a, b)
    assert found is not None, f
    return found.value


def test_radicand_and_roots() -> None:
    assert radicand(x / sqrt((1 - x**2) * (4 - x**2)), x) == (x, (1 - x**2) * (4 - x**2), -S.Half)
    assert radicand(1 / sqrt(x**5 - 1), x) is None
    assert real_roots(x**3 - x, x) == [-1, 0, 1]
    assert real_roots((1 - x**2) * (1 - k**2 * x**2), x, k < 1) == [-1 / k, -1, 1, 1 / k]
    assert real_roots(x**3 + 1, x) is None
    # an irreducible cubic: the roots are indexed roots, in increasing order
    found = real_roots(x**3 - 3 * x + 1, x)
    assert found is not None and all(isinstance(r, IndexedRoot) for r in found)
    values = [_value(r).real for r in found]
    assert values == sorted(values) and all(abs(v**3 - 3 * v + 1) < 1e-12 for v in values)


def test_legendre_reduction() -> None:
    s, m = Symbol('s', positive=True), Symbol('m', positive=True)
    assert legendre_reduction(Reduction(1, m, s, 0, 1)) == 2 * elliptic_k(m)
    assert legendre_reduction(Reduction(1 / (1 + 2 * s), m, s, 0, 1)) == 2 * elliptic_pi(-2, m)
    assert legendre_reduction(Reduction(1, m, s, 0, S.Half)) == 2 * elliptic_f(pi / 4, m)
    assert legendre_reduction(Reduction(1 / s, m, s, 0, 1)) is None
    # a double pole with a parameter: the leading coefficient of the
    # denominator depends on m (the fraction field the typed apart has not got)
    found = legendre_reduction(Reduction(1 / (1 + m * s)**2, m, s, 0, 1), m < 1)
    assert found is not None
    value = _value(found.subs(m, Rational(1, 3)))
    reference = _quadrature(lambda u: 1 / ((1 + u / 3)**2 * mpmath.sqrt(u * (1 - u) * (1 - u / 3))), 0, 1)
    assert abs(value - reference) < 1e-12


def test_partial_fractions_over_the_field_of_the_constants() -> None:
    # (240 sqrt(2) - 340) s**2 + ... is a constant times (s + 16 + 12 sqrt(2))**2,
    # which only the factorisation over QQ(sqrt(2)) sees
    s = Symbol('s')
    Q = 1 / ((-340 + 240 * sqrt(2)) * s**2 + (640 - 480 * sqrt(2)) * s - 640)
    found = _partial_fractions(Q, s)
    assert found is not None and len(found.poles) == 1
    alpha, beta, powers = found.poles[0]
    assert alpha == 1 and beta == 16 + 12 * sqrt(2) and set(powers) == {2}
    field = _ConstantField([sqrt(2) + sqrt(3)], s)
    assert field.canonical(1 / (1 + sqrt(2))) == -1 + sqrt(2)


def test_values_against_quadrature() -> None:
    cases: list[tuple[Expr, Expr, Expr, Callable[[mpmath.mpf], mpmath.mpf]]] = [
        (1 / sqrt(x * (1 - x) * (4 - x)), Integer(0), S.Half, lambda u: 1 / mpmath.sqrt(u * (1 - u) * (4 - u))),
        (1 / sqrt(x * (x - 1) * (2 - x) * (3 - x)), Integer(1), Integer(2),
         lambda u: 1 / mpmath.sqrt(u * (u - 1) * (2 - u) * (3 - u))),
        (x**2 / sqrt((x**2 + 1) * (x + 2) * (3 - x)), Integer(-2), Integer(3),
         lambda u: u**2 / mpmath.sqrt((u**2 + 1) * (u + 2) * (3 - u))),
        (1 / ((x - 5) * sqrt((x**2 + 1) * (x + 2) * (3 - x))), Integer(-2), Integer(1),
         lambda u: 1 / ((u - 5) * mpmath.sqrt((u**2 + 1) * (u + 2) * (3 - u)))),
        (1 / sqrt(x**3 + 1), Integer(-1), Integer(2), lambda u: 1 / mpmath.sqrt(u**3 + 1)),
        (1 / sqrt((x**2 + 1) * (x**2 + 4)), Integer(-3), Integer(2),
         lambda u: 1 / mpmath.sqrt((u**2 + 1) * (u**2 + 4))),
        (1 / sqrt(x**4 + x - 1), Integer(1), Integer(2), lambda u: 1 / mpmath.sqrt(u**4 + u - 1)),
    ]
    for f, a, b, g in cases:
        value = _value(_integral(f, a, b))
        reference = _quadrature(g, float(a), float(b))
        assert abs(value.imag) < 1e-12 and abs(value.real - reference) < 1e-10, (f, value, reference)


def test_trigonometric_forms() -> None:
    # sin(asin(sqrt(s))) is not evaluated by the core's sin: the functions
    # of the variable are written in s first
    m = k**2
    found = elliptic_integral(sqrt(1 - m * sin(theta)**2), theta, 0, pi / 2, k < 1)
    assert found is not None and found.value == elliptic_e(m)
    found = elliptic_integral(1 / sqrt(1 - m * sin(theta)**2), theta, 0, pi / 3, k < 1)
    assert found is not None and found.value == elliptic_f(pi / 3, m)
    assert elliptic_integral(sin(theta) / sqrt(1 - m * sin(theta)**2), theta, 0, pi / 2, k < 1) is None


def test_refusals() -> None:
    assert elliptic_integral(1 / sqrt(x * (1 - x) * (4 - x)), x, 0, 2) is None
    assert elliptic_integral(x / sqrt(x**3 + 1), x, 0, oo) is None
    assert elliptic_integral(1 / sqrt(-(x**2 + 1) * (x**2 + 4)), x, 0, 1) is None
    missing: Optional[object] = elliptic_integral(1 / sqrt(x**5 + 1), x, 0, 1)
    assert missing is None


def test_symbolic_radicals_are_distinct_symbols() -> None:
    # the radicals A and B of the cosine map were two dummies named A,
    # which the typed polynomials (mapping their symbols by name) took for
    # one: the value of x/sqrt(P) lost its third kind and was -46.40
    # instead of 0.794 at a = 2, b = -1, p = 1/3, q = 3/2; and a**(9/2)
    # beside sqrt(a) are powers of one generator, or the square in the
    # denominator of x**2/sqrt(P) is not seen
    a, b, p, q = (Symbol(n, real=True) for n in 'abpq')
    P = (a - x) * (x - b) * ((x - p)**2 + q**2)
    point = {a: Integer(2), b: Integer(-1), p: Rational(1, 3), q: Rational(3, 2)}
    for R, g in ((x, lambda u: u), (x**2, lambda u: u**2)):
        found = elliptic_integral(R / sqrt(P), x, b, a, [b < a, q > 0])
        assert found is not None, R
        value = _value(found.value.subs(point))
        reference = _quadrature(lambda u: g(u) / mpmath.sqrt((2 - u) * (u + 1) * ((u - mpmath.mpf(1) / 3)**2 + 2.25)),
                                -1, 2)
        assert abs(value - reference) < 1e-10, (R, value, reference)
