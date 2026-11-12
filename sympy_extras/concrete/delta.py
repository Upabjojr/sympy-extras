"""Sums and products with Kronecker deltas (SymPy's ``deltasummation``
and ``deltaproduct``): a delta linear in the index picks the term at its
solution, as a ``Piecewise`` on the solution lying in the range.

Ported from SymPy's ``sympy.concrete.delta``. The equation of a delta in
the index is solved natively (by
:mod:`~sympy_extras.solvers.linsolve`,
:func:`~sympy_extras.assumptions.solve.solve` and the port of SymPy's
``solve``, :func:`~sympy_extras.solvers.solvers.solve_equation`); the
systems of several deltas (and the equation of one delta solved for all
its symbols) by :mod:`~sympy_extras.solvers.linsolve` when they are
linear with rational coefficients and a unique solution, by
:func:`~sympy_extras.solvers.solvers.solve_system` otherwise. The module
loads no SymPy.

Examples
========

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.functions.special.tensor_functions import KroneckerDelta
>>> from sympy_extras.concrete.delta import deltasummation, deltaproduct
>>> i, j, k, n = (Symbol(s, integer=True) for s in 'ijkn')
>>> deltasummation(KroneckerDelta(i, k), (k, 0, n)), deltasummation(k*KroneckerDelta(i, k), (k, 1, 3))
(Piecewise((1, (i >= 0) & (i <= n)), (0, True)), Piecewise((i, (i >= 1) & (i <= 3)), (0, True)))
>>> deltaproduct(KroneckerDelta(i, k), (k, 1, n))
KroneckerDelta(0, n) + KroneckerDelta(1, i)*KroneckerDelta(1, n)
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/concrete/delta.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; solve native where possible (linsolve, assumptions.solve); no caching.
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Optional

from sympy_extras.core.add import Add, add
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr
from sympy_extras.core.logic import true
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import Integer
from sympy_extras.core.symbol import Dummy, Symbol

from .expr_with_limits import Limits

__all__ = ['deltasummation', 'deltaproduct']

ZERO = Integer(0)
ONE = Integer(1)


def _e(b: Basic) -> Expr:
    assert isinstance(b, Expr)
    return b


def _kd() -> type:
    from sympy_extras.functions.special.tensor_functions import KroneckerDeltaNode
    return KroneckerDeltaNode


def _delta(i: Expr, j: Expr) -> Expr:
    from sympy_extras.functions.special.tensor_functions import KroneckerDelta
    return KroneckerDelta(i, j)


def _solve(e: Expr, v: Symbol) -> list[Expr]:
    """SymPy's ``solve(e, v)``: native for an equation free of ``v`` or
    linear in it with a rational coefficient
    (:func:`~sympy_extras.solvers.linsolve.solve_one`) and for the
    equations whose solutions over the complex numbers the native
    :func:`~sympy_extras.assumptions.solve.solve` lists (a finite set; the
    assumptions of ``v``, an integer index say, are honoured as SymPy's
    ``solve`` does), and by the native port of SymPy's ``solve``
    (:func:`~sympy_extras.solvers.solvers.solve_equation`) otherwise;
    ``NotImplementedError``, as SymPy raises, when none lists them."""
    from sympy_extras.solvers.linsolve import solve_one
    known = solve_one(e, v)
    if known is not None:
        return known
    listed = _finite_solutions(e, v)
    if listed is not None:
        return listed
    from sympy_extras.solvers.solvers import solve_equation
    found = solve_equation(e, v)
    if found is None:
        raise NotImplementedError('no native method solves %s = 0 for %s' % (e, v))
    return found


def _finite_solutions(e: Expr, v: Symbol) -> Optional[list[Expr]]:
    """The solutions of ``e = 0`` for ``v`` over the complex numbers when
    the native :func:`~sympy_extras.assumptions.solve.solve` finds finitely
    many (in the order of its ``FiniteSet``), ``None`` otherwise (a
    condition set, infinitely many solutions, a time limit)."""
    from sympy_extras.core.relational import eq
    from sympy_extras.sets.fancysets import Complexes
    from sympy_extras.sets.sets import EmptySetClass, FiniteSet
    from sympy_extras.assumptions.solve import solve
    from sympy_extras._timeout import attempt
    from sympy_extras.settings import settings
    found = attempt(lambda: solve(eq(e, Integer(0)), v, domain=Complexes), settings.timeout, native=True)
    if isinstance(found, EmptySetClass):
        return []
    if not isinstance(found, FiniteSet):
        return None
    solutions = [a for a in found.args if isinstance(a, Expr)]
    if len(solutions) != len(found.args) or any(s.has(v) for s in solutions):
        return None
    return solutions


def _solve_dicts(eqs: list[Expr]) -> list[dict[Expr, Expr]]:
    """SymPy's ``solve(eqs, dict=True)`` for all the symbols of the
    equations (in SymPy's canonical order): a linear system with rational
    coefficients and a unique solution directly, the others by the native
    :func:`~sympy_extras.solvers.solvers.solve_system`;
    ``NotImplementedError``, as SymPy raises, when it cannot solve them."""
    from sympy_extras.core.sorting import ordered
    from sympy_extras.solvers.linsolve import numeric_linear_solution
    from sympy_extras.solvers.solvers import solve_system
    solution = numeric_linear_solution(eqs)
    if solution is not None:
        return [dict(solution.items())]
    symbols = [s for s in ordered(list(set().union(*[e.free_symbols for e in eqs]))) if isinstance(s, Symbol)]
    found = solve_system(eqs, symbols)
    if found is None:
        raise NotImplementedError('no native method solves %s' % (eqs,))
    return [dict(sol.items()) for sol in found]


def _decided_true(b: Basic) -> bool:
    return b is true


# PORTED FROM SYMPY: sympy/concrete/delta.py:_expand_delta
def _expand_delta(expr: Expr, index: Symbol) -> Expr:
    """The first sum with a simple delta in a product distributed."""
    if not isinstance(expr, Mul):
        return expr
    found = False
    terms: list[Expr] = [ONE]
    for h in (a for a in expr.args if isinstance(a, Expr)):
        if not found and isinstance(h, Add) and _has_simple_delta(h, index):
            found = True
            terms = [terms[0] * t for t in h.args if isinstance(t, Expr)]
        else:
            terms = [t * h for t in terms]
    return add(*terms)


# PORTED FROM SYMPY: sympy/concrete/delta.py:_extract_delta
def _extract_delta(expr: Expr, index: Symbol) -> tuple[Optional[Expr], Expr]:
    """The first simple delta of a product and the rest."""
    if not _has_simple_delta(expr, index):
        return None, expr
    if isinstance(expr, _kd()):
        return expr, ONE
    if not isinstance(expr, Mul):
        raise ValueError('Incorrect expr')
    delta: Optional[Expr] = None
    terms: list[Expr] = []
    for arg in (a for a in expr.args if isinstance(a, Expr)):
        if delta is None and _is_simple_delta(arg, index):
            delta = arg
        else:
            terms.append(arg)
    return delta, mul(*terms)


# PORTED FROM SYMPY: sympy/concrete/delta.py:_has_simple_delta
def _has_simple_delta(expr: Expr, index: Symbol) -> bool:
    if expr.has(_kd()):
        if _is_simple_delta(expr, index):
            return True
        if isinstance(expr, (Add, Mul)):
            return any(_has_simple_delta(a, index) for a in expr.args if isinstance(a, Expr))
    return False


# PORTED FROM SYMPY: sympy/concrete/delta.py:_is_simple_delta
def _is_simple_delta(delta: Expr, index: Symbol) -> bool:
    """A delta whose difference of arguments is linear in the index."""
    if isinstance(delta, _kd()) and delta.has(index):
        from .gosper import _UPoly
        try:
            p = _UPoly.of(_e(delta.args[0]) - _e(delta.args[1]), index)
        except ValueError:
            return False
        return p.degree() == 1
    return False


# PORTED FROM SYMPY: sympy/concrete/delta.py:_remove_multiple_delta
def _remove_multiple_delta(expr: Expr) -> Expr:
    """The deltas of a product reduced to those of the solution of their
    equations (zero when there is none)."""
    if isinstance(expr, Add):
        return add(*[_remove_multiple_delta(a) for a in expr.args if isinstance(a, Expr)])
    if not isinstance(expr, Mul):
        return expr
    eqs: list[Expr] = []
    newargs: list[Expr] = []
    for arg in (a for a in expr.args if isinstance(a, Expr)):
        if isinstance(arg, _kd()):
            eqs.append(_e(arg.args[0]) - _e(arg.args[1]))
        else:
            newargs.append(arg)
    if not eqs:
        return expr
    solns = _solve_dicts(eqs)
    if not solns:
        return ZERO
    if len(solns) == 1:
        newargs += [_delta(k, v) for k, v in solns[0].items()]
        expr2 = mul(*newargs)
        if expr != expr2:
            return _remove_multiple_delta(expr2)
    return expr


# PORTED FROM SYMPY: sympy/concrete/delta.py:_simplify_delta
def _simplify_delta(expr: Expr) -> Expr:
    """A delta rewritten with the solution of its equation."""
    if isinstance(expr, _kd()):
        try:
            slns = _solve_dicts([_e(expr.args[0]) - _e(expr.args[1])])
        except NotImplementedError:
            return expr
        if slns and len(slns) == 1:
            return mul(*[_delta(k, v) for k, v in slns[0].items()])
    return expr


# PORTED FROM SYMPY: sympy/concrete/delta.py:deltaproduct
def deltaproduct(f: Expr, limit: Limits) -> Expr:
    """The product of ``f`` with Kronecker deltas over ``limit``."""
    from sympy_extras.core.sorting import default_sort_key
    from sympy_extras.polys.polytools import factor
    from .products import product
    i, a, b = limit
    if _decided_true((b - a) < 0):
        return ONE
    if not f.has(_kd()):
        return product(f, limit)
    if isinstance(f, Add):
        delta: Optional[Expr] = None
        terms: list[Expr] = []
        for arg in sorted((t for t in f.args if isinstance(t, Expr)), key=default_sort_key):
            if delta is None and _has_simple_delta(arg, i):
                delta = arg
            else:
                terms.append(arg)
        assert delta is not None
        newexpr = add(*terms)
        k = Dummy('kprime', integer=True)
        # SymPy's isinstance(limit[1], int) never holds for its Integers:
        # the sum over the position of the delta
        result = deltaproduct(newexpr, limit) + deltasummation(
            deltaproduct(newexpr, (i, a, k - 1)) * _e(delta.subs(i, k)) * deltaproduct(newexpr, (i, k + 1, b)),
            (k, a, b), no_piecewise=_has_simple_delta(newexpr, i))
        return _remove_multiple_delta(result)
    found, _ = _extract_delta(f, i)
    if found is None:
        g = _expand_delta(f, i)
        if f != g:
            return factor(deltaproduct(g, limit))
        return product(f, limit)
    return _remove_multiple_delta(_e(f.subs(i, a)) * _delta(b, a)) + ONE * _simplify_delta(_delta(b, a - 1))


# PORTED FROM SYMPY: sympy/concrete/delta.py:deltasummation
def deltasummation(f: Expr, limit: Limits, no_piecewise: bool = False) -> Expr:
    """The sum of ``f`` with Kronecker deltas over ``limit``: the term at the
    solution of the delta, on the condition that it lies in the range
    (unconditional with ``no_piecewise``)."""
    from sympy_extras.functions.elementary.piecewise import piecewise, piecewise_fold
    from sympy_extras.sets import interval
    from .expr_with_limits import sum_
    from .summations import summation
    x, a, b = limit
    if _decided_true((b - a) < 0):
        return ZERO
    if not f.has(_kd()):
        return summation(f, limit)
    g = _expand_delta(f, x)
    if isinstance(g, Add):
        return piecewise_fold(add(*[deltasummation(h, limit, no_piecewise) for h in g.args if isinstance(h, Expr)]))
    delta, expr = _extract_delta(g, x)
    if delta is None:
        return summation(f, limit)
    solns = _solve(_e(delta.args[0]) - _e(delta.args[1]), x)
    if not solns:
        return ZERO
    if len(solns) != 1:
        return sum_(f, limit)
    value = solns[0]
    if no_piecewise:
        return _e(expr.subs(x, value))
    return piecewise((_e(expr.subs(x, value)), interval(a, b).as_relational(value)), (ZERO, true))
