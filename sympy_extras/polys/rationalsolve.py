"""SymPy's ``solveset`` over the complex numbers restricted to the
equations that are rational in the unknown: the solutions in radicals, in
the order SymPy's ``FiniteSet`` lists them, and the points excluded as
roots of a denominator (SymPy's ``Complement``).

The steps of SymPy's ``_solveset`` are kept, since the form of the
solutions (and so their order) depends on them: the normalisation of the
equation (``a/m + h`` for ``a + m h``), the inversion that takes the
terms and factors free of the unknown to the other side
(``_invert_complex``), the union over the factors of a product, the
split into numerator and denominator (``_solve_as_rational``), the roots
of a polynomial (``_solve_as_poly``, with ``expand_complex`` on numerical
roots), and the final ``domain_check``. Whatever leaves this class
(functions of the unknown, radicals, roots not in radicals, a set that
is not finite) gives ``None``.

Examples
========

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.polys.rationalsolve import solveset_rational
>>> x, d = Symbol('x'), Symbol('d')
>>> solveset_rational(x - d**2, d), solveset_rational(x - 1/d, d), solveset_rational(d**2 + 1, d)
(([sqrt(x), -sqrt(x)], []), ([1/x], [0]), ([I, -I], []))
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/solvers/solveset.py (_solveset, _invert_complex,
# _solve_as_rational, _solve_as_poly, domain_check, _domain_check,
# FiniteSet ordering) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: only rational equations over the complex numbers; a finite
# set of solutions with the excluded points instead of the set classes;
# None for everything else (trigonometric, piecewise, relational, modular,
# radical, Abs equations, ConditionSet, ImageSet).
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Optional

from sympy_extras.core.add import Add, add
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr
from sympy_extras.core.mul import Mul, div, mul
from sympy_extras.core.numbers import ComplexInfinity, Infinity, Integer, NaN, NegativeInfinity
from sympy_extras.core.power import Pow
from sympy_extras.core.sorting import ordered
from sympy_extras.core.symbol import Symbol

__all__ = ['solveset_rational', 'Solutions']

#: the solutions in SymPy's order and the excluded points (SymPy's
#: ``Complement(FiniteSet(*solutions), FiniteSet(*excluded))``)
Solutions = tuple[list[Expr], list[Expr]]


def solveset_rational(f: Expr, x: Symbol) -> Optional[Solutions]:
    """The solutions of ``f = 0`` for ``x`` as SymPy's ``solveset`` gives
    them, ``None`` outside the rational equations."""
    result = _solveset(f, x)
    if result is None:
        return None
    solutions, excluded = result
    fx = f.as_independent(x, as_Add=True)[1].as_independent(x, as_Add=False)[1]
    return [s for s in solutions if not s.is_infinite and _domain_check(fx, x, s)], excluded


def _finite_set(elements: list[Expr]) -> list[Expr]:
    unique: list[Expr] = []
    for e in elements:
        if e not in unique:
            unique.append(e)
    listed = ordered(unique)
    return [e for e in listed if isinstance(e, Expr)]


def _union(parts: list[Solutions]) -> Optional[Solutions]:
    if any(excluded for _, excluded in parts):
        return None   # a union with a Complement: not a finite set here
    return _finite_set([s for solutions, _ in parts for s in solutions]), []


def _infinite(e: Expr) -> bool:
    return isinstance(e, (ComplexInfinity, Infinity, NegativeInfinity))


# PORTED FROM SYMPY: sympy/solvers/solveset.py:_solveset
def _solveset(f: Expr, x: Symbol) -> Optional[Solutions]:
    from sympy_extras.core.function import expand
    if isinstance(f, Mul):
        coeff, rest = f.as_independent(x, as_Add=False)
        if _infinite(coeff):
            from sympy_extras.polys.rationaltools import together
            rest = together(f)
        f = rest
    elif isinstance(f, Add):
        a, h = f.as_independent(x, as_Add=True)
        m, h = h.as_independent(x, as_Add=False)
        if not (_infinite(m) or m == Integer(0)):
            f = add(div(a, m), h)
    if expand(f).is_zero:
        return None   # every number is a solution
    if not f.has(x):
        return [], []
    if isinstance(f, Mul) and all(_finite_with_finite_vars(m) for m in f.args if isinstance(m, Expr)):
        parts: list[Solutions] = []
        for m in f.args:
            assert isinstance(m, Expr)
            part = _solveset(m, x)
            if part is None:
                return None
            parts.append(part)
        return _union(parts)
    if not _rational_in(f, x):
        return None   # trigonometric, piecewise, radical, ... equations
    inverted = _invert_complex(f, Integer(0), x)
    if inverted is None:
        return None
    lhs, rhs = inverted
    if lhs == x:
        from sympy_extras.simplify.signsimp import signsimp
        content, primitive = signsimp(rhs).as_content_primitive()
        return [mul(content, primitive)], []
    equation = add(lhs, mul(Integer(-1), rhs))
    if equation == f:
        return _solve_as_rational(equation, x)
    return _solveset(equation, x)


def _finite_with_finite_vars(e: Expr) -> bool:
    """SymPy's ``_is_finite_with_finite_vars`` for the rational
    expressions: finite when the symbols are, so without denominators."""
    return e.is_polynomial() is True


def _rational_in(f: Expr, x: Symbol) -> bool:
    return f.is_rational_function(x) is True and not f.has(NaN())


# PORTED FROM SYMPY: sympy/solvers/solveset.py:_invert_complex (its Add,
# Mul and Pow cases)
def _invert_complex(f: Expr, y: Expr, x: Symbol) -> Optional[tuple[Expr, Expr]]:
    """``(lhs, rhs)`` with ``f = y`` equivalent to ``lhs = rhs``."""
    while True:
        if f == x:
            return f, y
        if isinstance(f, Add):
            g, h = f.as_independent(x, as_Add=True)
            if g != Integer(0):
                f, y = h, add(y, mul(Integer(-1), g))
                continue
        if isinstance(f, Mul):
            g, h = f.as_independent(x, as_Add=False)
            if g != Integer(1):
                if _infinite(g):
                    return None   # no solution: SymPy's EmptySet
                f, y = h, div(y, g)
                continue
        if isinstance(f, Pow) and y == Integer(0):
            e = f.exp
            if isinstance(e, Expr) and e.is_Rational and e.is_positive:
                f = f.base
                continue
        return f, y


# PORTED FROM SYMPY: sympy/solvers/solveset.py:_solve_as_rational
def _solve_as_rational(f: Expr, x: Symbol) -> Optional[Solutions]:
    from sympy_extras.polys.polyroots import _mexpand
    from sympy_extras.polys.rationaltools import together
    from sympy_extras.simplify.radsimp import fraction
    expanded = f
    while True:
        again = _mexpand(expanded)
        if again == expanded:
            break
        expanded = again
    g, h = fraction(together(expanded, deep=True))
    if not h.has(x):
        return _solve_as_poly(g, x)
    valid = _solveset(g, x)
    invalid = _solveset(h, x)
    if valid is None or invalid is None or valid[1] or invalid[1]:
        return None
    removed = invalid[0]
    return [s for s in valid[0] if s not in removed], removed


# PORTED FROM SYMPY: sympy/solvers/solveset.py:_solve_as_poly (a polynomial
# in x with all its roots in radicals; with symbols in the coefficients,
# of degree two at most)
def _solve_as_poly(f: Expr, x: Symbol) -> Optional[Solutions]:
    from sympy_extras.core.function import expand_complex
    from sympy_extras.polys.polyerrors import BasePolynomialError
    from sympy_extras.polys.polyroots import roots
    from sympy_extras.polys.polytools import degree
    if not f.is_polynomial(x):
        return None
    try:
        n = degree(f, x)
        if (f.free_symbols - {x}) and n > 2:
            # Cardano's and Ferrari's formulas over SymPy's domain EX (not
            # ported) write the roots otherwise
            return None
        found = roots(f, x)
    except BasePolynomialError:
        return None
    if not isinstance(found, dict) or sum(found.values()) < n:
        return None
    solutions = _finite_set([r for r in found if isinstance(r, Expr)])
    if all(not s.free_symbols for s in solutions):
        solutions = _finite_set([expand_complex(s) for s in solutions])
    return solutions, []


# PORTED FROM SYMPY: sympy/solvers/solveset.py:_domain_check
def _domain_check(f: Basic, x: Symbol, p: Expr) -> bool:
    """False when ``f`` or one of its subexpressions becomes infinite at
    ``x = p``."""
    if not f.args:
        return True
    if isinstance(f, Expr):
        value = f.subs(x, p)
        if isinstance(value, Expr) and value.is_infinite:
            return False
    return all(_domain_check(g, x, p) for g in f.args)
