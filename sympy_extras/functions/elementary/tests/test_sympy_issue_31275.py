"""Regression tests for the findings of sympy/sympy#31275 in the
elementary functions which the ports shared with SymPy (V76, V77, V81,
V82, V85, V86, V87), each checked against mpmath."""
from __future__ import annotations

from typing import Callable

import mpmath

from sympy_extras.core import I, Integer, Rational, Symbol, pi
from sympy_extras.core.expr import Expr
from sympy_extras.core.function import Function
from sympy_extras.core.power import sqrt
from sympy_extras.functions import (Abs, Atan, Atan2, acos, acosh, acoth, arg, asec, asech, atan2, atanh, exp, im, re,
                                    sin, sinh)

x = Symbol('x')
u = Symbol('u')
y = Symbol('y', imaginary=True)
f = Function('f')


def _value(e: Expr, symbol: Symbol, point: Expr) -> complex:
    return complex(e.subs(symbol, point).evalf(30))


def _g(z: mpmath.mpc) -> mpmath.mpc:
    return z**2 + z**3 + mpmath.exp(z)


def test_derivatives_in_an_imaginary_variable() -> None:
    # V76: with y = I*t, d/dy re(f) = -I*d/dt re(f(I*t)) = I*im(f'), not -I*im(f')
    assert re(f(y)).diff(y) == I*im(f(y).diff(y))
    assert im(f(y)).diff(y) == -I*re(f(y).diff(y))
    assert re(y**2).diff(y) == 2*y
    g = y**2 + y**3 + exp(y)
    parts: list[tuple[Expr, Callable[[mpmath.mpc], mpmath.mpf]]] = [
        (re(f(y)), mpmath.re), (im(f(y)), mpmath.im), (Abs(f(y)), abs), (arg(f(y)), mpmath.arg)]
    mpmath.mp.dps = 30
    try:
        for numerator in (7, -13, 21):
            t = mpmath.mpf(numerator)/10
            for node, part in parts:
                # d/dy = -I*d/dt on the imaginary axis
                expected = complex(-1j*mpmath.diff(lambda v: part(_g(1j*v)), t))
                ours = _value(node.diff(y).subs(f(y), g).doit(), y, I*Rational(numerator, 10))
                assert abs(ours - expected) < 1e-12, (node, numerator)
    finally:
        mpmath.mp.dps = 15
    # V77: arg(I + y**2) = atan2(1, -t**2), of derivative 2*t/(1 + t**4) in t
    assert arg(f(y)).diff(y).subs(f(y), I + y**2).doit() == -2*y/(1 + y**4)


def test_atan2_on_the_negative_imaginary_axis() -> None:
    # V85: atan2(0, x) = -I*log(x/sqrt(x**2)) is pi where x/sqrt(x**2) = -1
    assert atan2(Integer(0), -2*I) == pi and atan2(Integer(0), 2*I) == 0
    assert atan2(Integer(0), Integer(-2)) == pi and atan2(Integer(0), Integer(2)) == 0
    for a, b in ((0, -2), (0, 2), (-2, 0), (2, 0), (1, -1), (-1, -1)):
        point = mpmath.mpc(a, b)
        expected = complex(-1j*mpmath.log(point/mpmath.sqrt(point**2)))
        ours = atan2(Integer(0), Integer(a) + Integer(b)*I)
        assert abs(complex(ours.evalf(20)) - expected) < 1e-14, (a, b)
    rewritten = Atan2(Integer(0), y).rewrite(Atan)
    assert rewritten.subs(y, -2*I) == pi and rewritten.subs(y, 2*I) == 0
    generic = Atan2(u, x).rewrite(Atan)
    assert generic.subs({u: Integer(0), x: -3*I}) == pi
    assert generic.subs({u: Integer(0), x: 3*I}) == 0
    assert generic.subs({u: Integer(0), x: Integer(-3)}) == pi


def test_leading_terms_at_a_branch_point_reached_by_a_limit() -> None:
    # V81, V82, V86, V87: sin(x)/x is nan at 0 by substitution and 1 as a limit: the leading term is
    # that of the function at its branch point 1, not f(1) = 0
    third = sqrt(Integer(3))/3
    assert acos(sin(x)/x).as_leading_term(x) == third*sqrt(x**2)
    assert asec(sin(x)/x).as_leading_term(x) == third*sqrt(-x**2)
    assert acosh(sin(x)/x).as_leading_term(x) == third*sqrt(-x**2)
    assert asech(sinh(x)/x).as_leading_term(x) == third*sqrt(-x**2)
    assert acosh(1 - x**2/6).as_leading_term(x) == third*sqrt(-x**2)
    assert acos(1 - x**2/6).as_leading_term(x) == third*sqrt(x**2)
    cases: list[tuple[Expr, Callable[[mpmath.mpf], mpmath.mpc]]] = [
        (acos(sin(x)/x), lambda t: mpmath.acos(mpmath.sin(t)/t)),
        (asec(sin(x)/x), lambda t: mpmath.asec(mpmath.sin(t)/t)),
        (acosh(sin(x)/x), lambda t: mpmath.acosh(mpmath.sin(t)/t)),
        (asech(sinh(x)/x), lambda t: mpmath.asech(mpmath.sinh(t)/t)),
        (acosh(sinh(x)/x), lambda t: mpmath.acosh(mpmath.sinh(t)/t)),
        (asech(sin(x)/x), lambda t: mpmath.asech(mpmath.sin(t)/t)),
        (atanh(sin(x)/x), lambda t: mpmath.atanh(mpmath.sin(t)/t)),
        (acoth(sinh(x)/x), lambda t: mpmath.acoth(mpmath.sinh(t)/t))]
    mpmath.mp.dps = 40
    try:
        for e, reference in cases:
            leading = e.as_leading_term(x)
            assert leading != 0, e
            ours = _value(leading, x, Rational(1, 10**8))
            expected = complex(reference(mpmath.mpf(10)**-8))
            assert abs(ours - expected) <= 1e-6*abs(expected), (e, leading)
    finally:
        mpmath.mp.dps = 15
