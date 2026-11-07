"""B-splines on the typed core: :func:`bspline_basis` and
:func:`bspline_basis_set`.

Ported from SymPy's ``sympy.functions.special.bsplines``: the B-spline of
degree ``d`` on a knot sequence by the Cox--de Boor recursion, as a
``Piecewise`` on the knot intervals (``x >= k_i`` and ``x <= k_{i+1}``),
the pieces with the same interval added and the expressions expanded.
Not ported: ``interpolating_spline`` (it solves a linear system with
SymPy's ``linsolve``).

Examples
========

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.functions.special.bsplines import bspline_basis
>>> x = Symbol('x')
>>> bspline_basis(1, (0, 1, 2), 0, x)
Piecewise((x, (x >= 0) & (x <= 1)), (2 - x, (x >= 1) & (x <= 2)), (0, True))
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/functions/special/bsplines.py (bspline_basis,
# bspline_basis_set, _add_splines, _ivl) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: strict typing, the interval of a condition read from the
# relations; interpolating_spline not ported.
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Sequence

from sympy_extras.core.add import add, sub
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import expand
from sympy_extras.core.logic import And, Boolean, BooleanLike, and_, true
from sympy_extras.core.mul import div, mul
from sympy_extras.core.numbers import Integer
from sympy_extras.core.relational import Ge, Gt, Le, Lt, Relational, ge, le, lt
from sympy_extras.functions.elementary.piecewise import Piecewise, piecewise, piecewise_fold

__all__ = ['bspline_basis', 'bspline_basis_set']

ZERO = Integer(0)


def _smaller_larger(r: Relational) -> tuple[Expr, Expr]:
    """SymPy's ``lts`` and ``gts``: the smaller and the larger side."""
    if isinstance(r, (Ge, Gt)):
        return (r.rhs, r.lhs)
    if isinstance(r, (Le, Lt)):
        return (r.lhs, r.rhs)
    raise TypeError('unexpected relation: %s' % (r,))


# PORTED FROM SYMPY: sympy/functions/special/bsplines.py:_ivl
def _interval(cond: Boolean, x: Expr) -> tuple[Expr, Expr]:
    """The ``(lower, upper)`` of a condition ``lower <= x <= upper``."""
    if isinstance(cond, And) and len(cond.args) == 2:
        a, b = cond.args
        if isinstance(a, Relational) and isinstance(b, Relational):
            if _smaller_larger(a)[0] == x:
                a, b = b, a
            return (_smaller_larger(a)[0], _smaller_larger(b)[1])
    raise TypeError('unexpected cond type: %s' % (cond,))


def _expanded(e: Expr) -> Expr:
    """SymPy's ``Piecewise.expand()``: the expressions expanded, the
    Piecewise evaluated again."""
    if isinstance(e, Piecewise):
        return piecewise(*[(expand(pair.expr), pair.cond) for pair in e.args])
    return expand(e)


# PORTED FROM SYMPY: sympy/functions/special/bsplines.py:_add_splines
def _add_splines(c: Expr, b1: Expr, d: Expr, b2: Expr, x: Expr) -> Expr:
    if c == ZERO or b1 == ZERO:
        return _expanded(piecewise_fold(mul(d, b2)))
    if d == ZERO or b2 == ZERO:
        return _expanded(piecewise_fold(mul(c, b1)))
    p1, p2 = piecewise_fold(mul(c, b1)), piecewise_fold(mul(d, b2))
    if not (isinstance(p1, Piecewise) and isinstance(p2, Piecewise)):
        raise TypeError("the B-spline pieces are expected to be Piecewise")
    new_args: list[tuple[Expr, BooleanLike]] = []
    p2_args = list(p2.args[:-1])
    for pair in p1.args[:-1]:
        expression, cond = pair.expr, pair.cond
        lower = _interval(cond, x)[0]
        for i, pair2 in enumerate(p2_args):
            lower_2, upper_2 = _interval(pair2.cond, x)
            if pair2.cond == cond:
                expression = add(expression, pair2.expr)
                del p2_args[i]
                break
            if lt(lower_2, lower) == true and le(upper_2, lower) == true:
                new_args.append((pair2.expr, pair2.cond))
                del p2_args[i]
                break
        new_args.append((expression, cond))
    new_args.extend((pair2.expr, pair2.cond) for pair2 in p2_args)
    new_args.append((ZERO, true))
    return _expanded(Piecewise(*new_args))


# PORTED FROM SYMPY: sympy/functions/special/bsplines.py:bspline_basis
def bspline_basis(d: int, knots: Sequence[ExprLike], n: int, x: ExprLike) -> Expr:
    """The ``n``-th B-spline of degree ``d`` on the knots, as a
    ``Piecewise`` in ``x``."""
    variable = to_expr(x)
    points = tuple(to_expr(k) for k in knots)
    if n + d + 1 > len(points) - 1:
        raise ValueError("n + d + 1 must not exceed len(knots) - 1")
    if d < 0:
        raise ValueError("degree must be non-negative: %r" % (n,))
    if d == 0:
        # SymPy's Interval(k_n, k_{n+1}).contains(x); a degenerate interval
        # is the point {k_n}, whose membership is Eq(x, k_n)
        if points[n] == points[n + 1]:
            from sympy_extras.core.relational import eq
            return piecewise((Integer(1), eq(variable, points[n])), (ZERO, true))
        return piecewise((Integer(1), and_(ge(variable, points[n]), le(variable, points[n + 1]))), (ZERO, true))
    denominator = sub(points[n + d + 1], points[n + 1])
    if denominator != ZERO:
        b_coefficient: Expr = div(sub(points[n + d + 1], variable), denominator)
        b2 = bspline_basis(d - 1, points, n + 1, variable)
    else:
        b_coefficient = b2 = ZERO
    denominator = sub(points[n + d], points[n])
    if denominator != ZERO:
        a_coefficient: Expr = div(sub(variable, points[n]), denominator)
        b1 = bspline_basis(d - 1, points, n, variable)
    else:
        a_coefficient = b1 = ZERO
    return _add_splines(a_coefficient, b1, b_coefficient, b2, variable)


# PORTED FROM SYMPY: sympy/functions/special/bsplines.py:bspline_basis_set
def bspline_basis_set(d: int, knots: Sequence[ExprLike], x: ExprLike) -> list[Expr]:
    """The ``len(knots) - d - 1`` B-splines of degree ``d``."""
    return [bspline_basis(d, knots, i, x) for i in range(len(knots) - d - 1)]
