"""The typed complex parts (re, im, Abs, sign, arg, conjugate, atan2) and
the complex expansion against SymPy (the oracle), compared without
letting SymPy re-evaluate the results."""
from __future__ import annotations

from typing import Callable

import sympy

from sympy_extras._typing_sympy import as_expr as sympy_expr
from sympy_extras.core import E, I, Integer, Rational, Symbol, oo, pi, zoo
from sympy_extras.interop import to_sympy
from sympy_extras.core.expr import Expr
from sympy_extras.core.function import Function
from sympy_extras.core.power import sqrt
from sympy_extras.functions import (Abs, Re, abs_, arg, atan2, conjugate, cos, exp, im, log, re, sign, sin, sinh,
                                    tan)

x, y = Symbol('x'), Symbol('y')
r, s = Symbol('r', real=True), Symbol('s', real=True)
p = Symbol('p', positive=True)
n = Symbol('n', negative=True)
f = Function('f')

UNARY: list[tuple[Callable[[Expr], Expr], Callable[[sympy.Expr], sympy.Expr]]] = [
    (re, sympy.re), (im, sympy.im), (abs_, sympy.Abs), (sign, sympy.sign), (arg, sympy.arg),
    (conjugate, sympy.conjugate)]


def arguments() -> list[Expr]:
    return [Integer(0), Integer(3), Integer(-2), Rational(-1, 2), 2 + 3*I, 3 - 4*I, I, -I, 5*I, pi, E, oo, -oo, zoo,
            x, -x, r, -r, p, n, I*r, -I*p, x + I*y, r + I*s, 2*x, x*y, x + 1, -x - 1, r**2, x**2, sqrt(Integer(2)),
            sqrt(Integer(-3)), exp(x), exp(I*r), exp(r + I*pi/3), log(x), log(p), sin(r), sin(x), cos(I*r),
            tan(x + I*r), sinh(r + I*s), (1 + I)/(1 - I), (1 + 2*I)*(1 + 3*I), (r + I*s)**2, (r + I*s)**3,
            1/(r + I*s), x*conjugate(x), f(x), f(p)**2, Abs(x), sign(x), 2**(I*r), p**(I*r),
            atan2(r, s), Re(x), conjugate(x + y)]


def test_values_match_sympy() -> None:
    different: list[tuple[str, str, str, str]] = []
    for a in arguments():
        sa = sympy_expr(to_sympy(a))
        for ours_f, theirs_f in UNARY:
            ours, theirs = str(to_sympy(ours_f(a), evaluate=False)), str(theirs_f(sa))
            if ours != theirs:
                different.append((ours_f.__name__, str(a), ours, theirs))
    assert not different, different


def test_as_real_imag_matches_sympy() -> None:
    different: list[tuple[str, str, str]] = []
    for a in arguments():
        if a.has(oo) or a.has(zoo):
            continue
        ours = tuple(str(to_sympy(part, evaluate=False)) for part in a.as_real_imag())
        theirs = tuple(str(part) for part in sympy_expr(to_sympy(a)).as_real_imag())
        if ours != theirs:
            different.append((str(a), str(ours), str(theirs)))
    assert not different, different


def test_atan2_matches_sympy() -> None:
    different: list[tuple[str, str]] = []
    values: list[Expr] = [Integer(1), Integer(-1), Integer(0), Integer(2), sqrt(Integer(3)), -sqrt(Integer(3)), r, p,
                          n, I, oo, -oo]
    for a in values:
        for b in values:
            ours, theirs = str(to_sympy(atan2(a, b), evaluate=False)), str(sympy.atan2(to_sympy(a), to_sympy(b)))
            if ours != theirs and not ('Heaviside' in theirs or 'Piecewise' in theirs):
                different.append((str((a, b)), ours + ' vs ' + theirs))
    assert not different, different


def test_expand_splits_the_base_before_the_parts() -> None:
    # SymPy's expand (the power_base hint) splits (-p)**(I*r) into
    # (-1)**(I*r)*p**(I*r) before taking the parts
    e = (-p)**(1 + I*r)
    sp, sr = sympy.Symbol('p', positive=True), sympy.Symbol('r', real=True)
    assert str(to_sympy(re(e), evaluate=False)) == str(sympy.re((-sp)**(1 + sympy.I*sr)))


def test_derivatives() -> None:
    sx = sympy.Symbol('x')
    for g, h in [(re, sympy.re), (im, sympy.im), (abs_, sympy.Abs), (conjugate, sympy.conjugate), (arg, sympy.arg)]:
        assert str(to_sympy(g(x**2).diff(x), evaluate=False)) == str(h(sx**2).diff(sx)), g
    sr = sympy.Symbol('r', real=True)
    for g, h in [(re, sympy.re), (im, sympy.im), (abs_, sympy.Abs), (conjugate, sympy.conjugate)]:
        assert str(to_sympy(g(r**2 + I*r).diff(r), evaluate=False)) == str(h(sr**2 + sympy.I*sr).diff(sr)), g
    assert abs(x) == abs_(x) and isinstance(Abs(-3), Abs) and abs_(-3) == 3


def test_adjoint_and_transpose_of_noncommutative_match_sympy() -> None:
    from sympy_extras.interop import expr_from_sympy
    A, B, C = sympy.symbols('A B C', commutative=False)
    sx = sympy.Symbol('x')
    cases = [A*B, A*B*C, 2*A*B + sx*C, (A*B)**2, A**2*B, sx*A, A + B, sympy.I*A*B, A**-1*B, sympy.exp(A),
             sx**2*A*B, sympy.sin(A)*B, sympy.exp(A*B), sympy.Piecewise((A, sx > 0), (B, True)), sympy.exp_polar(A)]
    for e in cases:
        ours = expr_from_sympy(e)
        assert str(to_sympy(ours.adjoint(), evaluate=False)) == str(sympy.adjoint(e)), e
        assert str(to_sympy(ours.transpose(), evaluate=False)) == str(sympy.transpose(e)), e
        assert ours.is_commutative == e.is_commutative, e
