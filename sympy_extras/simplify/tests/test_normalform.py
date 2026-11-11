"""The normal form modulo the relations of sines, cosines and radicals:
known identities reduce to zero, and the result has the value of the
input at sample points (evaluated by SymPy)."""
from __future__ import annotations

import random

import pytest
import sympy

from sympy_extras.core import I, Integer, Rational, Symbol
from sympy_extras.core.expr import Expr
from sympy_extras.functions import cos, cosh, cot, csc, exp, sec, sin, sinh, sqrt, tan, tanh
from sympy_extras.interop import sympy_expr, to_sympy
from sympy_extras.simplify.normalform import normal_form

x, y = Symbol('x'), Symbol('y')


def test_identities_reduce_to_zero() -> None:
    identities: list[Expr] = [
        sin(x)**2 + cos(x)**2 - 1,
        sin(x)**4 - cos(x)**4 - sin(x)**2 + cos(x)**2,
        1 + tan(x)**2 - sec(x)**2,
        1 + cot(x)**2 - csc(x)**2,
        tan(x) + cot(x) - sec(x)*csc(x),
        (1 - cos(x))/sin(x) - sin(x)/(1 + cos(x)),
        cosh(x)**2 - sinh(x)**2 - 1,
        1 - tanh(x)**2 - 1/cosh(x)**2,
        sin(x)**2*cos(y)**2 - cos(x)**2*sin(y)**2 - sin(x)**2 + sin(y)**2,
        sqrt(x)*sqrt(x) - x,
        1/(sqrt(x) + 1) - (sqrt(x) - 1)/(x - 1),
        x**Rational(3, 2) - x*sqrt(x),
        x**Rational(1, 3)*x**Rational(2, 3) - x,
        sqrt(x/y)*sqrt(x/y) - x/y,
        sqrt(1 - sin(x)**2)**2 - cos(x)**2,
        (1 + I)**4 + 4,
        1/(x + I) - (x - I)/(x**2 + 1),
        (sqrt(2) + sqrt(3))**2 - 5 - 2*sqrt(2)*sqrt(3),
        sqrt(x + sqrt(y))**4 - x**2 - 2*x*sqrt(y) - y,
    ]
    for identity in identities:
        assert normal_form(identity) == 0, identity


def test_non_identities_are_not_zero() -> None:
    for expression in [sin(x)**2 - cos(x)**2, sin(x) - cos(x), sqrt(x) - x, sin(x)**2 + cos(y)**2 - 1, cosh(x)**2 + sinh(x)**2 - 1,
                       sqrt(x) + sqrt(y), exp(x) - 1, tan(x) - sin(x)]:
        assert normal_form(expression) != 0


def test_values_are_kept() -> None:
    rnd = random.Random(61)
    atoms: list[Expr] = [x, y, sin(x), cos(x), tan(x), sqrt(x), sqrt(x + 1), cos(y), sinh(y), cosh(y), Integer(2),
                         Rational(1, 3), x**Rational(3, 2), 1/sqrt(y + 2), exp(x)]

    def build(depth: int) -> Expr:
        if depth == 0:
            return rnd.choice(atoms)
        kind = rnd.randint(0, 3)
        left, right = build(depth - 1), build(depth - 1)
        if kind == 0:
            return left + right
        if kind == 1:
            return left*right
        if kind == 2:
            return left - right**2
        return left/(right + 3)

    sx, sy = sympy.Symbol('x'), sympy.Symbol('y')
    # one of these raised KeyError: exp(2*x) when the result was converted between the forms in sines and in
    # cosines through an expression: the core turns exp(x)**2 into the new kernel exp(2*x)
    for _ in range(40):
        expression = build(3)
        reduced = normal_form(expression)
        point: dict[sympy.Basic | complex, sympy.Expr | complex] = {sx: sympy.Rational(7, 10), sy: sympy.Rational(13, 10)}
        before = sympy.N(sympy_expr(to_sympy(expression)).subs(point), 30)
        after = sympy.N(sympy_expr(to_sympy(reduced)).subs(point), 30)
        assert abs(before - after) < sympy.Float('1e-20')*(1 + abs(before))
        # a normal form: applying it again changes nothing, and the difference with the input reduces to zero
        assert normal_form(reduced - expression) == 0


def test_forms() -> None:
    assert normal_form(tan(x)*cos(x)) == sin(x)
    # the shorter of the forms in sines and in cosines
    assert normal_form(cos(x)**2) == cos(x)**2 and normal_form(1 - sin(x)**2) == cos(x)**2
    assert normal_form(1 - cos(x)**2) == sin(x)**2 and normal_form(1 + tan(x)**2) == 1/cos(x)**2
    # as many terms in both forms: the one with fewer kernels
    assert normal_form((2 + cos(x))**2) == cos(x)**2 + 4*cos(x) + 4 and normal_form((2 + sin(x))**2) == sin(x)**2 + 4*sin(x) + 4
    assert normal_form((1 - cos(x)**2)/sin(x)) == sin(x)
    assert normal_form(sqrt(x)**3) == x*sqrt(x)
    assert normal_form(Integer(6)/4) == Rational(3, 2) and normal_form(x - x) == 0
    # the fractional powers of one base become powers of one root
    assert normal_form(x**Rational(1, 2)*x**Rational(1, 3)*x**Rational(1, 6)) == x
    with pytest.raises(ZeroDivisionError):
        normal_form(1/(sin(x)**2 + cos(x)**2 - 1))
