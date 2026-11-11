"""A relational solved for one variable (SymPy's ``_solve_inequality``).

Ported from SymPy's ``sympy.solvers.inequalities``: a relational linear
in the variable is solved here (the coefficient divided out when its sign
is known, the conditions at the infinities added), and so is a
relational rational in the variable with rational coefficients (SymPy's
``reduce_rational_inequalities``: the set of the cylindrical algebraic
decomposition, :mod:`sympy_extras.polys.cad.qe`, as a relational). The
module imports no SymPy: a relational with other coefficients, or not
rational in the variable, goes through SymPy for that relational only, at
run time (``solve_univariate_inequality`` is not ported;
:mod:`sympy_extras.solvers.inequalities_sympy` holds the step).

Examples
========

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.solvers.inequalities import solve_inequality
>>> x, a = Symbol('x'), Symbol('a', positive=True)
>>> solve_inequality(2*x - 4 > 0, x), solve_inequality(3 - x <= 1, x), solve_inequality(a*x < 1, x)
(x > 2, x >= 2, x < 1/a)
>>> solve_inequality(x**2 - 4 < 0, x), solve_inequality(x**2 > 2, x)
((-2 < x) & (x < 2), (sqrt(2) < x) | (x < -sqrt(2)))
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/solvers/inequalities.py (_solve_inequality,
# the linear part; reduce_rational_inequalities over QQ) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; linearity is read from the expanded terms (our Poly
# does not take every coefficient domain); the rational case by the CAD;
# the other nonlinear cases go through SymPy.
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Optional

from sympy_extras.core.add import Add
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr
from sympy_extras.core.logic import Boolean, and_, false, not_, true
from sympy_extras.core.numbers import Integer, NaN, NegativeInfinity, oo
from sympy_extras.core.relational import Relational, eq, ne
from sympy_extras.core.symbol import Symbol
from sympy_extras.sets.sets import Set

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


def _rational_set(ie: Relational, s: Symbol) -> Optional[Set]:
    """SymPy's ``reduce_rational_inequalities([[ie]], s, relational=False)``
    when ``ie`` is rational in ``s`` with rational coefficients (the set
    of the real ``s`` where it holds, by the CAD), ``None`` otherwise."""
    from sympy_extras.polys.cad.qe import solution_set
    from sympy_extras.polys.rationaltools import together
    from sympy_extras.polys.roots import in_radicals
    numerator, denominator = together(ie.lhs - ie.rhs).as_numer_denom()
    for part in (numerator, denominator):
        coefficients = _coefficients(part, s)
        if coefficients is None or not all(c.is_rational is True for c in coefficients):
            return None
    formula = _relation(ie.rel_op, numerator*denominator, Integer(0))
    if denominator.has(s):
        formula = and_(formula, ne(denominator, 0))
    return in_radicals(solution_set(formula, s))


def _coefficients(e: Expr, s: Symbol) -> Optional[list[Expr]]:
    """The coefficients of the expanded ``e`` as a polynomial in ``s``,
    ``None`` when it is not one."""
    from sympy_extras.core.function import expand
    ex = expand(e)
    found: list[Expr] = []
    for t in (ex.args if isinstance(ex, Add) else (ex,)):
        assert isinstance(t, Expr)
        c, k = t.as_coeff_exponent(s)
        if c.has(s) or not isinstance(k, Integer) or k.p < 0:
            return None
        found.append(c)
    return found


# PORTED FROM SYMPY: sympy/solvers/inequalities.py:_solve_inequality
def _nonlinear(ie: Relational, s: Symbol) -> Boolean:
    """The nonlinear case of SymPy's ``_solve_inequality``: the set of
    ``reduce_rational_inequalities`` as a relational, without the
    restrictions at the infinities which the relational does not have."""
    solved = _rational_set(ie, s)
    if solved is None:
        from .inequalities_sympy import solve_inequality_of_core
        return solve_inequality_of_core(ie, s)
    rv = solved.as_relational(s)
    okoo = _classify(ie, s, oo)
    if okoo is true and _classify(rv, s, oo) is false:
        rv = _boolean(rv.subs(s < oo, true))
    oknoo = _classify(ie, s, NegativeInfinity())
    if oknoo is true and _classify(rv, s, NegativeInfinity()) is false:
        rv = _boolean(rv.subs(NegativeInfinity() < s, true))
        rv = _boolean(rv.subs(s > NegativeInfinity(), true))
    if rv is true:
        rv = (s <= oo) if okoo is true else (s < oo)
        if oknoo is not true:
            rv = and_(NegativeInfinity() < s, rv)
    return rv


def _boolean(b: Basic) -> Boolean:
    if not isinstance(b, Boolean):
        raise TypeError('a condition is expected, not %s' % (b,))
    return b


# PORTED FROM SYMPY: sympy/solvers/solvers.py:denoms
def _denominators(b: Basic) -> set[Expr]:
    """SymPy's ``denoms``: the factors of the denominators of every
    subexpression."""
    from sympy_extras.core.basic import preorder_traversal
    from sympy_extras.core.mul import Mul
    from sympy_extras.simplify.radsimp import denom
    found: set[Expr] = set()
    for p in preorder_traversal(b):
        if not isinstance(p, Expr):
            continue
        d = denom(p)
        if d == 1:
            continue
        found.update(Mul.make_args(d))
    return found


# PORTED FROM SYMPY: sympy/solvers/inequalities.py:_solve_inequality
def solve_inequality(ie: Boolean, s: Symbol) -> Boolean:
    """``ie`` solved for ``s``: ``s < c``, ``s >= c``, ..., with the
    conditions SymPy adds (a relational not rational in ``s`` with
    rational coefficients is solved by SymPy)."""
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
        return _nonlinear(ie, s)
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
        # the denominators of ie which rv lost: the values of s where they
        # vanish are not solutions (x < x*(2/x - 1) is x < 1 with x != 0)
        current = _denominators(rv)
        for d in _denominators(ie) - current:
            c = solve_inequality(eq(d, 0), s)
            if isinstance(c, Relational) and c.rel_op == '==' and c.lhs == s:
                if _classify(rv, s, c.rhs) is true:
                    conds.append(not_(c))
        for i in (NegativeInfinity(), oo):
            if _classify(rv, s, i) is true and _classify(ie, s, i) is not true:
                conds.append(s < i if i is oo else i < s)
    conds.append(rv)
    return and_(*conds)
