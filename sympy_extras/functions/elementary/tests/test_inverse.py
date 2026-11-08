"""The typed inverse trigonometric and hyperbolic functions against
SymPy's (the oracle): table values, special numbers, imaginary arguments,
compositions with the direct functions, derivatives, numerical values;
compared without letting SymPy re-evaluate the results."""
from __future__ import annotations

from typing import Callable

import sympy

from sympy_extras._typing import as_expr as sympy_expr
from sympy_extras.core import I, Integer, Rational, Symbol, oo, pi, to_sympy, zoo
from sympy_extras.core.expr import Expr
from sympy_extras.core.power import sqrt
from sympy_extras.functions import (Acos, Acosh, Asin, Atan, acos, acosh, acot, acoth, acsc, acsch, asec, asech, asin,
                                    asinh, atan, atanh, cos, cosh, cot, csc, sec, sin, sinh, tan, tanh)
from sympy_extras.functions.elementary import hyperbolic, trigonometric

x = Symbol('x')

INVERSES: list[tuple[Callable[[Expr], Expr], type[sympy.Function]]] = [
    (asin, sympy.asin), (acos, sympy.acos), (atan, sympy.atan), (acot, sympy.acot), (asec, sympy.asec),
    (acsc, sympy.acsc), (asinh, sympy.asinh), (acosh, sympy.acosh), (atanh, sympy.atanh), (acoth, sympy.acoth),
    (asech, sympy.asech), (acsch, sympy.acsch)]


def _arguments() -> list[Expr]:
    values: list[Expr] = [Integer(0), Integer(1), Integer(-1), Rational(1, 2), Rational(-1, 2), Integer(2), Integer(-3),
                          I, -I, 2*I, -3*I, oo, -oo, zoo, x, -x, I*x, -I*x, sqrt(Integer(2)), -sqrt(Integer(3))/2]
    tables = [trigonometric._asin_table(), trigonometric._atan_table(), trigonometric._acsc_table(),
              hyperbolic._acosh_table(), hyperbolic._acsch_table(), hyperbolic._asech_table()]
    for table in tables:
        for key in table:
            values.extend([key, -key])
    for a in [Integer(1), Integer(3), Integer(5), Integer(-2), Rational(7, 2), Integer(4), 3*pi/4, Rational(-9, 2)]:
        values.extend([sin(a), cos(a), tan(a), cot(a), sec(a), csc(a), sinh(a), tanh(a), -cos(a), -sec(a)])
    return values


def test_values_match_sympy() -> None:
    different: list[tuple[str, str, str, str]] = []
    for a in _arguments():
        sa = to_sympy(a)
        for f, g in INVERSES:
            ours, theirs = str(to_sympy(f(a), evaluate=False)), str(g(sa))
            if ours != theirs:
                different.append((f.__name__, str(a), ours, theirs))
    assert not different, different


def test_derivatives_and_numbers() -> None:
    sx = sympy.Symbol('x')
    for f, g in INVERSES:
        e = f(x**2 + 3)
        assert sympy.simplify(sympy_expr(to_sympy(e.diff(x))) - g(sx**2 + 3).diff(sx)) == 0, f
        for value in [Rational(1, 3), Integer(3), 2 + I]:
            ours = complex(sympy_expr(to_sympy(f(value).evalf(30))))
            theirs = complex(sympy.N(g(to_sympy(value)), 30))
            assert abs(ours - theirs) < 1e-25, (f, value)


def test_compositions_and_nodes() -> None:
    assert asin(sin(Integer(3))) == pi - 3 and acos(cos(Integer(4))) == -4 + 2*pi
    assert atan(tan(Integer(2))) == 2 - pi and asinh(sinh(Integer(2))) == 2 and atanh(tanh(Integer(1))) == 1
    assert asin(I*x) == I*asinh(x) and atan(I*x) == I*atanh(x) and acot(I*x) == -I*acoth(x)
    assert acos(Rational(1, 2)) == pi/3 and atan(sqrt(Integer(3))) == pi/3 and acosh(Rational(1, 2)) == I*pi/3
    # the classes never evaluate
    for cls, function, argument in [(Asin, asin, Integer(1)), (Acos, acos, Integer(0)), (Atan, atan, Integer(1)),
                                    (Acosh, acosh, Integer(1))]:
        node = cls(argument)
        assert node.args == (argument,) and function(argument) != node and node.doit() == function(argument)
    assert cosh(Integer(0)) == 1 and cot(pi/4) == 1 and csc(pi/2) == 1 and sec(Integer(0)) == 1
