"""The typed Mathieu and singularity functions against SymPy (the oracle),
compared without letting SymPy re-evaluate the results."""
from __future__ import annotations

import itertools
from typing import Callable

import pytest
import sympy

from sympy_extras._typing import as_expr as sympy_expr
from sympy_extras.core import I, Integer, Rational, Symbol, oo, to_sympy
from sympy_extras.core.expr import Expr
from sympy_extras.core.numbers import nan
from sympy_extras.functions import SingularityFunction, conjugate, mathieuc, mathieucprime, mathieus, mathieusprime

a, q, z = Symbol('a'), Symbol('q'), Symbol('z')
x, c = Symbol('x', real=True), Symbol('c', real=True)
p = Symbol('p', positive=True)

MATHIEU: list[tuple[Callable[[Expr, Expr, Expr], Expr], Callable[[sympy.Expr, sympy.Expr, sympy.Expr], sympy.Expr]]] = [
    (mathieus, sympy.mathieus), (mathieuc, sympy.mathieuc), (mathieusprime, sympy.mathieusprime),
    (mathieucprime, sympy.mathieucprime)]


def sympy_value(e: Expr) -> sympy.Expr:
    return sympy_expr(to_sympy(e))


def same(ours: Expr, theirs: sympy.Basic) -> bool:
    return str(to_sympy(ours, evaluate=False)) == str(theirs)


def test_mathieu_match_sympy() -> None:
    different = []
    for first, second, third in itertools.product([a, Integer(4)], [q, Integer(0), Rational(0)], [z, -z, 2*z, Integer(0)]):
        s1, s2, s3 = sympy_value(first), sympy_value(second), sympy_value(third)
        for f, g in MATHIEU:
            if not same(f(first, second, third), g(s1, s2, s3)):
                different.append((f.__name__, str(f(first, second, third)), str(g(s1, s2, s3))))
    assert not different, different
    sa, sq, sz = sympy.symbols('a q z')
    for f, g in MATHIEU:
        assert same(f(a, q, z**2).diff(z), g(sa, sq, sz**2).diff(sz)), f
        assert same(conjugate(f(a, q, z)), sympy.conjugate(g(sa, sq, sz))), f


def test_singularity_matches_sympy() -> None:
    different = []
    points: list[Expr] = [Integer(5), Integer(3), Integer(2), x, p, -p, oo, nan]
    offsets: list[Expr] = [Integer(3), c, Integer(0)]
    exponents: list[Expr] = [Integer(0), Integer(2), Integer(-1), Integer(-4), Integer(-5), Rational(1, 2), I, p, nan]
    for point, offset, exponent in itertools.product(points, offsets, exponents):
        sp, so, se = sympy_value(point), sympy_value(offset), sympy_value(exponent)
        try:
            theirs = sympy.SingularityFunction(sp, so, se)
        except ValueError:
            with pytest.raises(ValueError):
                SingularityFunction(point, offset, exponent)
            continue
        ours = SingularityFunction(point, offset, exponent)
        if not same(ours, theirs):
            different.append((str(ours), str(theirs)))
    with pytest.raises(ValueError):
        SingularityFunction(I, 0, 1)
    assert not different, different
    sx, sc = sympy.Symbol('x', real=True), sympy.Symbol('c', real=True)
    for exponent in [Integer(3), Integer(0), Integer(-2), p]:
        se = sympy_value(exponent)
        assert same(SingularityFunction(x, c, exponent).diff(x), sympy.SingularityFunction(sx, sc, se).diff(sx))
