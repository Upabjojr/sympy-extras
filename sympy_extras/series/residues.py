"""The residue of an expression at a point (SymPy's ``residue``): the
coefficient of ``1/(x - x0)`` in its Laurent series.

Examples
========

>>> from sympy_extras.core import Symbol, Integer
>>> from sympy_extras.functions import exp, gamma, sin
>>> from sympy_extras.series import residue
>>> x = Symbol('x')
>>> residue(1/x, x, 0), residue(exp(x)/x**3, x, 0), residue(1/sin(x)**2, x, 0)
(1, 1/2, 0)
>>> residue(gamma(x), x, Integer(-2))
1/2
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/series/residues.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; SymPy's as_coeff_mul(x) is spelled out.
# ---------------------------------------------------------------------
from __future__ import annotations

from sympy_extras.core.add import Add
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import Integer
from sympy_extras.core.order import Order
from sympy_extras.core.power import Pow
from sympy_extras.core.symbol import Symbol

__all__ = ['residue']


def _coeff_mul(e: Expr, x: Symbol) -> tuple[Expr, Expr]:
    """SymPy's ``e.as_coeff_mul(x)``, the dependent factors multiplied:
    the factors free of ``x`` and the others."""
    if isinstance(e, Mul):
        free = [f for f in e.args if isinstance(f, Expr) and not f.has(x)]
        dep = [f for f in e.args if isinstance(f, Expr) and f.has(x)]
        return mul(*free), mul(*dep)
    if e.has(x):
        return Integer(1), e
    return e, Integer(1)


# PORTED FROM SYMPY: sympy/series/residues.py:residue
def residue(expr: ExprLike, x: Symbol, x0: ExprLike) -> Expr:
    """The residue of ``expr`` at ``x = x0``: the coefficient of
    ``1/(x - x0)`` in the Laurent series (``NotImplementedError`` for a
    series of another form)."""
    e = to_expr(expr)
    point = to_expr(x0)
    if point != 0:
        shifted = e.subs(x, x + point)
        assert isinstance(shifted, Expr)
        e = shifted
    s = e
    for n in (0, 1, 2, 4, 8, 16, 32):
        s = e.nseries(x, n=n)
        order = s.getn()
        if not s.has(Order) or (order is not None and order.is_nonnegative is True):
            break
    collected = s.removeO().collect(x)
    res: Expr = Integer(0)
    for arg in (collected.args if isinstance(collected, Add) else (collected,)):
        assert isinstance(arg, Expr)
        c, m = _coeff_mul(arg, x)
        if not (m == 1 or m == x or (isinstance(m, Pow) and isinstance(m.args[1], Integer))):
            raise NotImplementedError('term of unexpected form: %s' % m)
        if m == 1/x:
            res = res + c
    return res
