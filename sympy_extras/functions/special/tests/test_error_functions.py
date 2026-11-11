"""The typed error functions against SymPy (the oracle), compared without
letting SymPy re-evaluate the results."""
from __future__ import annotations

from typing import Callable

import sympy

from sympy_extras._typing_sympy import as_expr as sympy_expr
from sympy_extras.core import I, Integer, Rational, Symbol, oo, pi
from sympy_extras.interop import to_sympy
from sympy_extras.core.expr import Expr
from sympy_extras.core.numbers import Float, nan
from sympy_extras.functions import Erf, Erfc, Erfi, erf, erfc, erfcinv, erfi, erfinv

x, y = Symbol('x'), Symbol('y')
r = Symbol('r', real=True)
p = Symbol('p', positive=True)

FUNCTIONS: list[tuple[Callable[[Expr], Expr], Callable[[sympy.Expr], sympy.Expr]]] = [
    (erf, sympy.erf), (erfc, sympy.erfc), (erfi, sympy.erfi), (erfinv, sympy.erfinv), (erfcinv, sympy.erfcinv)]


def values() -> list[Expr]:
    base: list[Expr] = [Integer(0), Integer(1), Integer(-1), Integer(2), Rational(1, 2), Float('0.25'), oo, -oo, nan,
                        I*oo, -I*oo, I, x, -x, 2*x, -2*x, I*x, -I*x, r, -r, p, -p, x + y, -x - y, 1 - x]
    return base + [erfinv(x), erfcinv(x), erf(r), -erf(r), erf(x), I*erfinv(x), I*erfcinv(x)]


def sympy_value(e: Expr) -> sympy.Expr:
    return sympy_expr(to_sympy(e))


def test_values_match_sympy() -> None:
    different = []
    for a in values():
        sa = sympy_value(a)
        for ours_f, theirs_f in FUNCTIONS:
            ours, theirs = str(to_sympy(ours_f(a), evaluate=False)), str(theirs_f(sa))
            if ours != theirs:
                different.append((ours_f.__name__, str(a), ours, theirs))
    # erfcinv of a float is evaluated here; SymPy has no numerical erfcinv
    accepted = [d for d in different if d[0] == 'erfcinv' and isinstance(sympy_value(Float(d[1])), sympy.Float)]
    assert different == accepted and len(accepted) == 1, different


def test_facts_and_parts_agree_with_sympy() -> None:
    for a in values():
        for node, s in [(Erf(a), sympy.erf(sympy_value(a), evaluate=False)),
                        (Erfc(a), sympy.erfc(sympy_value(a), evaluate=False)),
                        (Erfi(a), sympy.erfi(sympy_value(a), evaluate=False))]:
            for name in ['real', 'extended_real', 'finite', 'zero', 'positive', 'negative']:
                got = node._ask(name)
                if got is not None:
                    assert got == getattr(s, 'is_' + name), (node, name)
    for e in [erf(x + I*y), erfc(r), erfi(2 + I*r), erf(I*x)]:
        ours = tuple(str(to_sympy(part, evaluate=False)) for part in e.as_real_imag())
        theirs = tuple(str(part) for part in sympy_value(e).as_real_imag())
        assert ours == theirs, e


def test_derivatives_and_numbers() -> None:
    sx = sympy.Symbol('x')
    for f, g in FUNCTIONS:
        assert str(to_sympy(f(x**2).diff(x), evaluate=False)) == str(g(sx**2).diff(sx)), f
        for value in [Rational(1, 3), Rational(3, 5)]:
            ours = complex(sympy_value(f(value).evalf(20)))
            if f is erfcinv:
                theirs = complex(sympy.N(sympy.erfinv(1 - sympy_value(value)), 20))
            else:
                theirs = complex(sympy.N(g(sympy_value(value)), 20))
            assert abs(ours - theirs) < 1e-14, (f, value)
    assert str(erf(Float('0.5'))) == str(sympy.erf(sympy.Float('0.5'))) and erfc(pi).is_real is True
