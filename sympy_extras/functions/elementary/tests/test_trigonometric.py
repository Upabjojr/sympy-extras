"""The typed trigonometric and hyperbolic functions against SymPy's (the
oracle): evaluated forms on rational multiples of pi, imaginary arguments
and sums, derivatives, numerical values and facts; and the rule that the
classes never evaluate."""
from __future__ import annotations

from typing import Callable, Optional

import sympy

from sympy_extras._typing_sympy import as_expr as sympy_expr
from sympy_extras.core import I, Integer, Rational, Symbol, pi
from sympy_extras.interop import to_sympy
from sympy_extras.core.expr import Expr
from sympy_extras.core.function import DefinedFunction
from sympy_extras.functions import (Cos, Cosh, Cot, Coth, Csc, Csch, Exp, Log, Sec, Sech, Sin, Sinh, Tan, Tanh, cos,
                                    cosh, cot, coth, csc, csch, exp, log, sec, sech, sin, sinh, tan, tanh)

x, y = Symbol('x'), Symbol('y')
r = Symbol('r', real=True)
n = Symbol('n', integer=True)

FUNCTIONS: list[tuple[Callable[[Expr], Expr], type[sympy.Function]]] = [
    (sin, sympy.sin), (cos, sympy.cos), (tan, sympy.tan), (cot, sympy.cot), (sec, sympy.sec), (csc, sympy.csc),
    (sinh, sympy.sinh), (cosh, sympy.cosh), (tanh, sympy.tanh), (coth, sympy.coth), (sech, sympy.sech),
    (csch, sympy.csch)]

FACTS: dict[str, Callable[[sympy.Expr], Optional[bool]]] = {
    'real': lambda s: s.is_real, 'positive': lambda s: s.is_positive, 'negative': lambda s: s.is_negative,
    'zero': lambda s: s.is_zero, 'finite': lambda s: s.is_finite, 'rational': lambda s: s.is_rational,
    'algebraic': lambda s: s.is_algebraic, 'complex': lambda s: s.is_complex,
    'extended_real': lambda s: s.is_extended_real,
}


def _arguments() -> list[Expr]:
    values: list[Expr] = [Integer(0), Integer(1), Integer(-2), Rational(1, 3), x, -x, r, 2*x, x + y, I*x, -I*x, 3*I,
                          I*pi, I*pi/2, x + pi, x + pi/2, x - pi/3, x + 2*pi, x + I*pi, x + I*pi/2, n*pi, 2*n*pi,
                          x*pi, pi*I*Rational(2, 3) + x]
    for q in (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30):
        for p in range(-2*q, 2*q + 1, max(1, q//4)):
            values.append(Rational(p, q)*pi)
    return values


def test_values_match_sympy() -> None:
    # compared without letting SymPy re-evaluate our result (to_sympy with
    # evaluate=True would turn our tan(x + pi/2) into its -cot(x))
    different: list[tuple[str, str]] = []
    for a in _arguments():
        sa = to_sympy(a)
        for f, g in FUNCTIONS:
            if str(to_sympy(f(a), evaluate=False)) != str(g(sa)):
                different.append((f.__name__, str(a)))
    assert different == [], different


def test_exact_values() -> None:
    assert sin(pi/6) == Rational(1, 2) and cos(pi/3) == Rational(1, 2) and tan(pi/4) == 1
    assert str(cos(pi/5)) == '1/4 + sqrt(5)/4' and str(sin(pi/4)) == 'sqrt(2)/2'
    assert str(tan(pi/10)) == str(sympy.tan(sympy.pi/10))
    assert str(cos(pi/12)) == str(sympy.cos(sympy.pi/12))
    assert sin(n*pi) == 0 and cos(n*pi) == (-1)**n
    assert sin(I*x) == I*sinh(x) and cos(I*x) == cosh(x) and tan(I*x) == I*tanh(x)
    assert sinh(I*x) == I*sin(x) and cosh(I*pi) == -1 and tanh(-I*x) == -I*tan(x)


def test_derivatives_and_numbers() -> None:
    sx = sympy.Symbol('x')
    for f, g in FUNCTIONS:
        e = f(x**2 + 1)
        assert sympy.simplify(sympy_expr(to_sympy(e.diff(x))) - g(sx**2 + 1).diff(sx)) == 0, f
        value = f(Rational(1, 3)).evalf(30)
        assert abs(complex(sympy_expr(to_sympy(value))) - complex(sympy.N(g(sympy.Rational(1, 3)), 30))) < 1e-25


def test_facts_agree_with_sympy() -> None:
    for a in _arguments():
        for f, _ in FUNCTIONS:
            node = f(a)
            s = sympy_expr(to_sympy(node))
            for name, sympy_fact in FACTS.items():
                got = node._ask(name)
                if got is not None:
                    assert got == sympy_fact(s), (node, name, got)


def test_classes_never_evaluate() -> None:
    # every ported class builds the node it is given; the function evaluates
    classes: list[tuple[type[DefinedFunction], Callable[[Expr], Expr], Expr]] = [
        (Sin, sin, Integer(0)), (Cos, cos, Integer(0)), (Tan, tan, pi), (Sinh, sinh, Integer(0)),
        (Cosh, cosh, I*pi), (Tanh, tanh, Integer(0)), (Exp, exp, Integer(0)), (Log, log, Integer(1)),
        (Cot, cot, pi/4), (Sec, sec, Integer(0)), (Csc, csc, pi/2), (Coth, coth, Integer(0)), (Sech, sech, Integer(0)),
        (Csch, csch, I*pi/2)]
    for cls, function, argument in classes:
        node = cls(argument)
        assert type(node) is cls and node.args == (argument,), cls
        assert function(argument) != node and node.doit() == function(argument), cls
        assert node.func(*node.args) == node  # rebuilding keeps the node
