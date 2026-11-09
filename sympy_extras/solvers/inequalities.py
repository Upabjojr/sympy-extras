"""A relational solved for one variable (SymPy's ``_solve_inequality``).

Ported from SymPy's ``sympy.solvers.inequalities``: a relational linear
in the variable is solved here (the coefficient divided out when its sign
is known, the conditions at the infinities added); a nonlinear one goes
through SymPy for that relational only (``reduce_rational_inequalities``
and ``solve_univariate_inequality`` are not ported).

Examples
========

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.solvers.inequalities import solve_inequality
>>> x, a = Symbol('x'), Symbol('a', positive=True)
>>> solve_inequality(2*x - 4 > 0, x), solve_inequality(3 - x <= 1, x), solve_inequality(a*x < 1, x)
(x > 2, x >= 2, x < 1/a)
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/solvers/inequalities.py (_solve_inequality,
# the linear part) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; linearity is read from the expanded terms (our Poly
# does not take every coefficient domain); the nonlinear case goes
# through SymPy.
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Optional

from sympy_extras.core.add import Add
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr
from sympy_extras.core.logic import Boolean, and_, false, true
from sympy_extras.core.numbers import Integer, NaN, NegativeInfinity, oo
from sympy_extras.core.relational import Relational
from sympy_extras.core.symbol import Symbol

__all__ = ['solve_inequality']


def _degree(e: Expr, s: Symbol) -> Optional[int]:
    """The degree of the expanded ``e`` in ``s``, ``None`` when it is not a
    polynomial in ``s``."""
    from sympy_extras.core.function import expand
    degree = 0
    ex = expand(e)
    for t in (ex.args if isinstance(ex, Add) else (ex,)):
        assert isinstance(t, Expr)
        c, k = t.as_coeff_exponent(s)
        if c.has(s) or not isinstance(k, Integer) or k.p < 0:
            return None
        degree = max(degree, k.p)
    return degree


def _classify(ie: Boolean, s: Symbol, value: Expr) -> Optional[Basic]:
    """SymPy's ``classify``: the truth of ``ie`` at ``s = value`` (``nan``
    for a ``TypeError``), ``None`` when it is not decided."""
    try:
        v = ie.subs(s, value)
    except TypeError:
        return NaN()
    if isinstance(v, NaN):
        return v
    if v is not true and v is not false:
        return None
    return v


def _relation(op: str, lhs: Expr, rhs: Expr) -> Boolean:
    """SymPy's ``ie.func(lhs, rhs)``: the relation, evaluated."""
    from sympy_extras.core.relational import eq, ge, gt, le, lt, ne
    return {'<': lt, '<=': le, '>': gt, '>=': ge, '==': eq, '!=': ne}[op](lhs, rhs)


_REVERSED = {'<': '>', '<=': '>=', '>': '<', '>=': '<=', '==': '==', '!=': '!='}


def _through_sympy(ie: Relational, s: Symbol) -> Boolean:
    from sympy.solvers.inequalities import _solve_inequality
    from sympy_extras.interop import from_sympy, to_sympy
    r = from_sympy(_solve_inequality(to_sympy(ie), to_sympy(s)))
    if not isinstance(r, Boolean):
        raise TypeError('a condition is expected, not %s' % (r,))
    return r


# PORTED FROM SYMPY: sympy/solvers/inequalities.py:_solve_inequality
def solve_inequality(ie: Boolean, s: Symbol) -> Boolean:
    """``ie`` solved for ``s``: ``s < c``, ``s >= c``, ..., with the
    conditions SymPy adds (a relational not linear in ``s`` is solved by
    SymPy)."""
    from sympy_extras.core.exprtools import factor_terms
    from sympy_extras.core.function import expand
    if not isinstance(ie, Relational) or s not in ie.free_symbols:
        return ie
    if ie.rhs == s:
        ie = ie.reversed
    if ie.lhs == s and s not in ie.rhs.free_symbols:
        return ie
    expr = ie.lhs - ie.rhs
    degree = _degree(expr, s)
    if degree is None or degree > 1:
        return _through_sympy(ie, s)
    rv: Boolean
    conds: list[Boolean] = []
    if degree == 0:
        rv = _relation(ie.rel_op, expand(expr), Integer(0))
    else:
        e = expand(expr)
        rhs: Expr = Integer(0)
        b, _ = e.as_independent(s, as_Add=True)
        e = e - b
        rhs = rhs - b
        ef = factor_terms(e)
        a, e = ef.as_independent(s, as_Add=False)
        if a.is_zero is not False or (a.is_negative == a.is_positive is None and ie.rel_op not in ('!=', '==')):
            e = ef
            a = Integer(1)
        rhs = rhs / a
        if a.is_positive is True:
            rv = _relation(ie.rel_op, e, rhs)
        else:
            rv = _relation(_REVERSED[ie.rel_op], e, rhs)
        # SymPy's denominators of ie that rv lost are free of s here (a
        # polynomial in s): no condition comes from them
        for i in (NegativeInfinity(), oo):
            if _classify(rv, s, i) is true and _classify(ie, s, i) is not true:
                conds.append(s < i if i is oo else i < s)
    conds.append(rv)
    return and_(*conds)
