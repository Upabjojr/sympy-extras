"""The typed piecewise_fold, adjoint and transpose against SymPy (the
oracle), compared without letting SymPy re-evaluate the results."""
from __future__ import annotations

import sympy

from sympy_extras._typing import as_expr as sympy_expr
from sympy_extras.core import I, Integer, Symbol, gt, lt, true
from sympy_extras.interop import to_sympy
from sympy_extras.core.expr import Expr
from sympy_extras.functions import adjoint, conjugate, exp, piecewise, piecewise_fold, sin, transpose

x, y = Symbol('x'), Symbol('y')
r = Symbol('r', real=True)
A, B = Symbol('A', commutative=False), Symbol('B', commutative=False)
H = Symbol('H', commutative=False, hermitian=True)
K = Symbol('K', commutative=False, antihermitian=True)


def sympy_value(e: Expr) -> sympy.Expr:
    return sympy_expr(to_sympy(e))


def same(ours: Expr, theirs: sympy.Basic) -> bool:
    return str(to_sympy(ours, evaluate=False)) == str(theirs)


def test_piecewise_fold_matches_sympy() -> None:
    f = piecewise((x, gt(x, 0)), (Integer(2), true))
    g = piecewise((y, gt(x, 0)), (Integer(3), true))
    h = piecewise((x**2, lt(y, 1)), (Integer(0), true))
    cases: list[Expr] = [f + 1, 3*f, f*g, f + g, f + h, f*h + y, sin(f), exp(f + g), piecewise((f, lt(y, 0)), (Integer(1), true)),
                         f**2, f + g + h, x]
    different = []
    for e in cases:
        se = sympy_value(e)
        for evaluate in (True, None, False):
            ours = piecewise_fold(e, evaluate=evaluate)
            theirs = sympy.piecewise_fold(se, evaluate=evaluate)
            if not same(ours, theirs):
                different.append((str(e), str(evaluate), str(ours), str(theirs)))
    assert not different, different


def test_adjoint_transpose_match_sympy() -> None:
    different = []
    for e in [x, r, I*x, x + I*y, A, H, K, A*B, H*A, adjoint(A), transpose(A), conjugate(A), 2*A, transpose(adjoint(A))]:
        se = sympy_value(e)
        for ours, theirs in [(adjoint(e), sympy.adjoint(se)), (transpose(e), sympy.transpose(se)),
                             (conjugate(e), sympy.conjugate(se))]:
            if not same(ours, theirs):
                different.append((str(e), str(ours), str(theirs)))
    assert not different, different
