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
from sympy_extras.core.numbers import Integer
from sympy_extras.core.symbol import Symbol
from sympy_extras.polys.domains.sparsefractionfield import SparseFracElement, SparseFractionField
from sympy_extras.polys.domains.sparsepolynomialring import RationalPolynomialRing

__all__ = ['solve_linear_unique']


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
