"""Tests of the residue integrator on the objects of the core
(:mod:`sympy_extras.integrals.residues`): the values against hand
computations and mpmath's quadrature (independent sources), the helpers
(fractions with irrational coefficients, poles over ``ZZ_I``, the
synthetic division, the branch of the logarithm), and the agreement with
the SymPy face."""
from __future__ import annotations

import time
from typing import Callable, Optional

import mpmath

from sympy_extras._timeout import attempt
from sympy_extras.assumptions.ask import Assumptions, ask
from sympy_extras.core import I, Integer, Rational, S, Symbol, oo, pi, sqrt
from sympy_extras.core.expr import Expr
from sympy_extras.core.logic import Boolean
from sympy_extras.core.relational import Eq, Gt, Lt
from sympy_extras.functions import cos, exp, log, sin, tan
from sympy_extras.integrals import residues
from sympy_extras.integrals.conditions import ConditionalValue
from sympy_extras.integrals.residues import (fourier_integral, keyhole_integral, period_integral, poles,
                                             rational_function, real_line_integral, residue_integral)
from sympy_extras.settings import configure
from sympy_extras.simplify.simplify import simplify

x = Symbol('x')
a, b = Symbol('a', positive=True), Symbol('b', positive=True)
k, c = Symbol('k'), Symbol('c')


def _value(found: Optional[ConditionalValue]) -> Expr:
    assert found is not None
    return found.value


def _number(e: Expr) -> complex:
    return complex(e.evalf(30))


def _same(first: Expr, second: Expr) -> bool:
    if simplify(first - second) == 0:
        return True
    return abs(_number(first) - _number(second)) < 1e-20


def _quadrature(f: Callable[[mpmath.mpf], mpmath.mpf], lo: mpmath.mpf, hi: mpmath.mpf) -> complex:
    with mpmath.workdps(30):
        if lo == -mpmath.inf and hi == mpmath.inf:
            return complex(mpmath.quad(f, [-mpmath.inf, -1, 0, 1, mpmath.inf]))
        if hi == mpmath.inf:
            return complex(mpmath.quad(f, [lo, 1, mpmath.inf]))
        return complex(mpmath.quad(f, mpmath.linspace(lo, hi, 9)))


def test_rational_function_with_irrational_coefficients() -> None:
    # the constants are placeholders for the typed polynomials (no domain EX)
    fraction = rational_function((x + sqrt(2)) / ((x + sqrt(2)) * (x**2 + pi)), x)
    assert fraction is not None
    assert fraction.numerator.degree() == 0 and fraction.denominator.degree() == 2
    assert fraction.denominator.all_coeffs() == [S.One, S.Zero, pi]
    fraction = rational_function(1 / (x**2 + 2 * I * x - 1), x)
    assert fraction is not None and fraction.excess == 2
    assert rational_function(exp(x) / (x + 1), x) is None
    assert rational_function(S(3), x) is None


def test_poles_over_the_gaussian_integers() -> None:
    # the bug: with I a placeholder, the denominator (z**2 + 6*I*z - 1)**2 of
    # 1/(3 + sin(x))**2 in z = exp(I*x) was no square for the native roots,
    # whose quartic formula gave nothing; it is factored over ZZ_I first
    fraction = rational_function(1 / (x**2 + 6 * I * x - 1)**2, x)
    assert fraction is not None
    found = poles(fraction.denominator)
    assert found is not None
    assert sorted((str(p.point), p.multiplicity) for p in found) == \
        [('-3*I + 2*sqrt(2)*I', 2), ('-3*I - 2*sqrt(2)*I', 2)]
    # the multiplicities of a square with a parameter
    fraction = rational_function(1 / ((x**2 + 1)**2 * (x - a)), x)
    assert fraction is not None
    found = poles(fraction.denominator)
    assert found is not None
    assert sorted((str(p.point), p.multiplicity) for p in found) == [('-I', 2), ('I', 2), ('a', 1)]


def test_synthetic_division() -> None:
    q = residues._Polynomial([S.One, Integer(-2) * sqrt(2), Integer(2)], x)
    quotient = q.quotient_by_root(sqrt(2), 2)
    assert quotient.all_coeffs() == [S.One]
    q = residues._Polynomial([S.One, S.Zero, S.Zero, Integer(-1)], x)
    assert q.quotient_by_root(S.One, 1).all_coeffs() == [S.One, S.One, S.One]


def test_real_line() -> None:
    assert _same(_value(real_line_integral(1 / (x**4 + 1), x)), pi / sqrt(2))
    assert _same(_value(real_line_integral(x**2 / (x**4 + 1)**2, x)), pi * sqrt(2) / 8)
    assert _same(_value(real_line_integral(1 / (x**2 + 2 * x + 5)**2, x)), pi / 16)
    assert _same(_value(real_line_integral(1 / ((x**2 + a**2) * (x**2 + b**2)), x)), pi / (a * b * (a + b)))
    assert real_line_integral(x / (x**2 + 1), x) is None
    assert real_line_integral(1 / (x**2 - 1), x) is None
    assert real_line_integral(1 / (x**2 + c), x) is None
    # against mpmath
    value = _value(real_line_integral(1 / (x**6 + 1), x))
    assert abs(_number(value) - _quadrature(lambda t: 1 / (t**6 + 1), -mpmath.inf, mpmath.inf)) < 1e-20


def test_fourier() -> None:
    assert _same(_value(fourier_integral(cos(2 * x) / (x**2 + 1), x)), pi * exp(-2))
    assert _same(_value(fourier_integral(exp(-I * x) / (x**2 + 1), x)), pi * exp(-1))
    assert _same(_value(fourier_integral(cos(b * x) / (x**2 + a**2), x)), pi * exp(-a * b) / a)
    found = fourier_integral(x * sin(k * x) / (x**2 + 4), x)
    assert found is not None and found.condition == (k > 0)
    assert _same(found.value, pi * exp(-2 * k))
    assert _same(_value(fourier_integral(cos(x) / (x**2 + 1), x, half=True)), pi * exp(-1) / 2)
    assert fourier_integral(cos(x) / (x**2 + x + 1), x, half=True) is None
    assert fourier_integral(sin(x) / x, x) is None


def test_keyhole() -> None:
    assert _same(_value(keyhole_integral(sqrt(x) / (x**2 + 1), x)), pi / sqrt(2))
    found = keyhole_integral(x**c / (x + 1), x)
    assert found is not None and found.condition == ((c > -1) & (c < 0))
    assert _same(found.value, -pi / sin(pi * c))
    assert _same(_value(keyhole_integral(1 / ((x + 1) * (x + 2)), x)), log(2))
    assert _same(_value(keyhole_integral(log(x)**2 / (x**2 + 1), x)), pi**3 / 8)
    assert _same(_value(keyhole_integral(sqrt(x) / (x * (x + 1)), x)), pi)
    assert keyhole_integral(sqrt(x) / (x - 1), x) is None
    # a double pole with the branch of the logarithm: the derivation with
    # d L/dz = 1/z
    value = _value(keyhole_integral(x**Rational(1, 3) / (1 + x)**2, x))
    expected = _quadrature(lambda t: mpmath.cbrt(t) / (1 + t)**2, mpmath.mpf(0), mpmath.inf)
    assert abs(_number(value) - expected) < 1e-20


def test_period() -> None:
    assert _same(_value(period_integral(1 / (2 + cos(x)), x, 0, 2 * pi)), 2 * pi / sqrt(3))
    assert _same(_value(period_integral(1 / (3 + sin(x))**2, x, 0, 2 * pi)), 6 * pi / 8**Rational(3, 2))
    assert _same(_value(period_integral(cos(2 * x) / (5 - 4 * cos(x)), x, 0, 2 * pi)), pi / 6)
    assert _same(_value(period_integral(1 / (2 + tan(x)**2), x, 0, 2 * pi)), pi * (2 - sqrt(2)))
    found = period_integral(1 / (1 - 2 * a * cos(x) + a**2), x, 0, 2 * pi, a > 1)
    assert _same(_value(found), 2 * pi / (a**2 - 1))
    assert period_integral(1 / (1 + cos(x)), x, 0, 2 * pi) is None
    # placed at a sample, verified by the native quadratures
    found = period_integral(1 / (a + b * cos(x)), x, 0, 2 * pi, a > b)
    assert found is not None
    assert abs(_number(found.value.subs({a: 3, b: 1})) - float(2 * mpmath.pi / mpmath.sqrt(8))) < 1e-12
    with configure(numerical_checks=False):
        assert period_integral(1 / (a + b * cos(x)), x, 0, 2 * pi, a > b) is None


def test_dispatch() -> None:
    assert _same(_value(residue_integral(3 / (x**2 + 1), x, -oo, oo)), 3 * pi)
    assert _same(_value(residue_integral(1 / (x**2 + 1), x, 0, oo)), pi / 2)
    assert residue_integral(1 / (x**2 + 1), x, 0, 1) is None
    assert residue_integral(exp(-x), x, 0, oo) is None
    assert residue_integral(S.One, x, -oo, oo) is None


def test_numerical_poles_through_the_face() -> None:
    # the roots of x**4 + x**2 + x + 1 in radicals are unwieldy: CRootOf,
    # through the face (the only step of the module on SymPy)
    y = Symbol('y', real=True)
    value = _value(real_line_integral(1 / (y**4 + y**2 + y + 1), y))
    expected = _quadrature(lambda t: 1 / (t**4 + t**2 + t + 1), -mpmath.inf, mpmath.inf)
    assert abs(_number(value) - expected) < 1e-15


def test_numerically() -> None:
    # the relations unevaluated, as the questions about the poles come
    assert residues._numerically(Gt(sqrt(2), Rational(14, 10))) is True
    assert residues._numerically(Lt(sqrt(2), Rational(14, 10))) is False
    # an equality is refuted, never confirmed
    assert residues._numerically(Eq(sqrt(2), 1)) is False
    assert residues._numerically(Eq(sqrt(2)**2, 2)) is None


def test_an_enclosing_limit_is_not_a_slow_question() -> None:
    # the bug: the residue integrator asks the assumptions under a limit of
    # its own and took the expiry of an enclosing limit for its question
    # being slow, marking every later question slow and going on without
    # the answer; the expiry goes to the owner of the enclosing limit
    def slow(q: Boolean, assumptions: Assumptions) -> Optional[bool]:
        started = time.monotonic()
        while time.monotonic() - started < 5:
            pass
        return None

    locator = residues._Locator([k > 0])
    assert vars(residues)['ask'] is ask
    vars(residues)['ask'] = slow
    try:
        with configure(numerical_checks=False):
            assert attempt(lambda: (locator.decide(k + c > 0), 'went on'), 0.3) is None
    finally:
        vars(residues)['ask'] = ask
    assert not locator._slow


def test_the_face_agrees() -> None:
    import sympy
    from sympy_extras.integrals import residues_sympy
    from sympy_extras._numeric_sympy import core_expr
    X = sympy.Symbol('x')
    A = sympy.Symbol('a', positive=True)
    for f, lo, hi in [(1 / (X**4 + 1), -sympy.oo, sympy.oo), (sympy.cos(A * X) / (X**2 + 1), -sympy.oo, sympy.oo),
                      (sympy.sqrt(X) / (X**2 + 1), 0, sympy.oo), (1 / (2 + sympy.cos(X)), 0, 2 * sympy.pi)]:
        face = residues_sympy.residue_integral(f, X, lo, hi)
        native = residue_integral(core_expr(f), x, core_expr(lo), core_expr(hi))
        assert face is not None and native is not None
        assert core_expr(face.value) == native.value
