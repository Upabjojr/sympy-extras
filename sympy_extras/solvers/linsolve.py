"""The unique solution of a linear system with rational function
coefficients, on the native polynomial layer.

:func:`solve_linear_unique` solves ``A c + b = 0`` for the unknowns ``c``
when the coefficients are rational functions over ``QQ`` of the other
symbols: the reduced row echelon form over the field
``QQ(other symbols)`` (:meth:`Domain.sparse_echelon`). It returns the
values of the unknowns when the solution is unique, ``None`` otherwise:
an inconsistent or underdetermined system, an equation not linear in the
unknowns, or a coefficient outside the field (``sqrt(2)``, ``pi``,
``log(n)``, ...; the caller then declines, as with SymPy's ``linsolve``
returning no single point).

Examples
========

>>> from sympy_extras.core import symbols
>>> from sympy_extras.solvers.linsolve import solve_linear_unique
>>> a, b, n = symbols('a b n')
>>> solve_linear_unique([a + b - 1, a - b - n], [a, b])
[n/2 + 1/2, 1/2 - n/2]
>>> solve_linear_unique([a + b - 1, 2*a + 2*b - 2], [a, b]) is None
True
>>> solve_linear_unique([a*n - 1, b - a], [a, b])
[1/n, 1/n]
"""
from __future__ import annotations

from fractions import Fraction
from typing import Optional, Sequence

from sympy_extras.core.expr import Expr
from sympy_extras.core.function import diff
from sympy_extras.core.numbers import Integer, Rational
from sympy_extras.core.symbol import Symbol
from sympy_extras.polys.domains.sparsefractionfield import SparseFracElement, SparseFractionField
from sympy_extras.polys.domains.sparsepolynomialring import RationalPolynomialRing

__all__ = ['solve_linear_unique', 'linear_root', 'solve_one', 'numeric_linear_solution']


def solve_linear_unique(equations: Sequence[Expr], unknowns: Sequence[Symbol]) -> Optional[list[Expr]]:
    """The values of ``unknowns`` (in their order) solving the linear
    ``equations`` (each ``= 0``), when they are unique; ``None`` otherwise
    (see the module)."""
    if not unknowns:
        return None
    unknown_set = set(unknowns)
    others = sorted({s for e in equations for s in e.free_symbols if isinstance(s, Symbol) and s not in unknown_set},
                    key=lambda s: s.sort_key())
    # the ring maps symbols by name: unique names for the other symbols (a
    # Dummy may share its name with a Symbol), and one placeholder variable
    # when there is none (a ring of no variables is refused)
    names = ['_ls%d' % i for i in range(len(others))] or ['_ls']
    renamed: dict[Expr, Expr] = {s: Symbol(name) for s, name in zip(others, names)}
    restored: dict[Expr, Expr] = {Symbol(name): s for s, name in zip(others, names)}
    field = SparseFractionField(RationalPolynomialRing(names))
    zeros: dict[Expr, Expr] = {u: Integer(0) for u in unknowns}
    rows: list[dict[int, SparseFracElement[Fraction]]] = []
    for e in equations:
        row: dict[int, SparseFracElement[Fraction]] = {}
        entries = [diff(e, u) for u in unknowns] + [e.xreplace(zeros)]
        for column, entry in enumerate(entries):
            if any(entry.has(u) for u in unknowns):
                return None   # not linear in the unknowns
            try:
                value = field.from_expr(entry.xreplace(renamed))
            except ValueError:
                return None   # a coefficient outside QQ(others)
            if not field.is_zero(value):
                row[column] = value
        if row:
            rows.append(row)
    echelon = field.sparse_echelon(rows)
    count = len(unknowns)
    if any(min(r) == count for r in echelon):
        return None   # inconsistent: a pivot in the constant column
    if sorted(min(r) for r in echelon) != list(range(count)):
        return None   # underdetermined
    solution: list[Expr] = []
    for r in echelon:
        constant = r.get(count)
        value = field.neg(constant) if constant is not None else field.zero
        solution.append(field.to_expr(value).xreplace(restored))
    return solution


def linear_root(e: Expr, v: Symbol) -> Optional[Expr]:
    """The root ``-b/a`` of ``e = a*v + b`` for a nonzero rational ``a``
    (SymPy's ``solve(e, v)`` is then ``[-b/a]``), ``None`` when ``e`` is not
    of that form. (:func:`solve_one` adds the equations free of ``v``.)

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.solvers.linsolve import linear_root
    >>> i, k = symbols('i k')
    >>> linear_root(i - 2*k, k), linear_root(2*k - i - 1, k), linear_root(k**2 - i, k)
    (i/2, i/2 + 1/2, None)
    """
    a = diff(e, v)
    if not isinstance(a, Rational) or a == 0:
        return None
    b = e.subs(v, Integer(0))
    if not isinstance(b, Expr) or b.has(v):
        return None
    return -b/a


def solve_one(e: Expr, v: Symbol) -> Optional[list[Expr]]:
    """SymPy's ``solve(e, v)`` when it is known natively: ``[]`` for an
    equation free of ``v`` (SymPy gives no solution then, even for
    ``0``), the root of :func:`linear_root`; ``None`` otherwise. As in
    SymPy, a root contradicting the assumptions of ``v``
    (``check_assumptions`` is ``False``: ``1/2`` for an integer ``v``) is
    dropped.

    >>> from sympy_extras.core import Symbol, symbols
    >>> from sympy_extras.solvers.linsolve import solve_one
    >>> i, k, m = symbols('i k m')
    >>> solve_one(2*k + 1, m), solve_one(i - 2*m, m), solve_one(m**2 - i, m) is None
    ([], [i/2], True)
    >>> j = Symbol('j', integer=True)
    >>> solve_one(2*j - 1, j), solve_one(2*j - 4, j)
    ([], [2])
    """
    if not e.has(v):
        return []
    root = linear_root(e, v)
    if root is None:
        return None
    from sympy_extras.core.assumptions import check_assumptions
    return [] if check_assumptions(root, v) is False else [root]


def numeric_linear_solution(equations: Sequence[Expr]) -> Optional[dict[Symbol, Expr]]:
    """The unique solution for all the free symbols of ``equations`` (each
    ``= 0``) when they are linear with rational coefficients and constants
    (SymPy's ``solve(equations, dict=True)`` is then ``[solution]``),
    ``None`` otherwise.

    >>> from sympy_extras.core import symbols, Integer
    >>> from sympy_extras.solvers.linsolve import numeric_linear_solution
    >>> i, n = symbols('i n')
    >>> numeric_linear_solution([1 - i, 1 - n]), numeric_linear_solution([i + n - 1]) is None
    ({i: 1, n: 1}, True)
    """
    unknowns = sorted({s for e in equations for s in e.free_symbols if isinstance(s, Symbol)},
                      key=lambda s: s.sort_key())
    if not unknowns:
        return None
    for e in equations:
        if not isinstance(e.subs({u: Integer(0) for u in unknowns}), Rational):
            return None
        if not all(isinstance(diff(e, u), Rational) for u in unknowns):
            return None
    values = solve_linear_unique(equations, unknowns)
    if values is None:
        return None
    return dict(zip(unknowns, values))
