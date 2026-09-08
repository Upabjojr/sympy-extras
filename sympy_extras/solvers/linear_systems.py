"""First order linear systems ``Y' = A(x) Y`` with rational coefficients,
reduced to a scalar equation by a cyclic vector.

For a row vector `c_0` (constant), the derivatives of `u = c_0 Y` are
`u^{(k)} = c_k Y` with `c_{k+1} = c_k A + c_k'`; the first linear
dependence `\\sum_k a_k c_k = 0` over the rational functions gives the
scalar equation `\\sum_k a_k u^{(k)} = 0`. When it has order `n` (a
*cyclic vector*; a generic `c_0` is cyclic) the whole solution follows
from `u`: `Y = M^{-1} (u, u', \\ldots, u^{(n-1)})^T` with `M` the matrix of
the `c_k`. The scalar equation is solved by the solvers of
:mod:`sympy_extras.solvers.linear_ode` (rational and hyperexponential
solutions, Kovacic, special functions), which also gives the rational
solutions of the system (the elimination method of Abramov and
Bronstein). SymPy's ``dsolve`` handles systems with constant coefficients
and a few non-constant patterns.

References
==========

.. [Barkatou] M. Barkatou, An algorithm for computing a companion block
   diagonal form for a system of linear differential equations, AAECC 4
   (1993).
"""
from __future__ import annotations

from typing import Optional

from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.numbers import Integer
from sympy.core.symbol import Symbol
from sympy.matrices.dense import Matrix, MutableDenseMatrix
from sympy.polys.polytools import cancel, lcm_list

from sympy_extras._typing import as_expr

from .linear_ode import LinearOperator, _solve_operator, rational_solutions

__all__ = ['cyclic_vector', 'system_to_scalar', 'dsolve_linear_system', 'rational_system_solutions']


def _rows(A: MutableDenseMatrix, c0: MutableDenseMatrix, x: Symbol, count: int) -> list[MutableDenseMatrix]:
    rows = [c0]
    for _ in range(count):
        c = rows[-1]
        rows.append((c*A + c.diff(x)).applyfunc(cancel))
    return rows


def _dependence(rows: list[MutableDenseMatrix]) -> Optional[list[Expr]]:
    """Coefficients ``a_k`` with ``sum a_k rows[k] = 0`` and ``a_last = 1``
    when the last row depends on the previous ones."""
    M = Matrix.vstack(*rows).T   # columns are the rows
    n = len(rows)
    null = M.nullspace()
    for vector in null:
        last = as_expr(vector[n - 1])
        if last != 0:
            return [as_expr(cancel(vector[k]/last)) for k in range(n)]
    return None


def cyclic_vector(A: MutableDenseMatrix, x: Symbol) -> tuple[MutableDenseMatrix, LinearOperator]:
    """A cyclic row vector ``c_0`` for ``Y' = A Y`` and the scalar operator
    satisfied by ``u = c_0 Y``.

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_systems import cyclic_vector
    >>> c, L = cyclic_vector(Matrix([[0, 1], [-1, 0]]), x)
    >>> c, L.coefficients
    (Matrix([[1, 0]]), [1, 0, 1])
    """
    n = A.shape[0]
    candidates: list[MutableDenseMatrix] = []
    for i in range(n):
        candidates.append(Matrix([[Integer(1) if j == i else Integer(0) for j in range(n)]]))
    candidates.append(Matrix([[x**j for j in range(n)]]))
    candidates.append(Matrix([[Integer(j + 1) for j in range(n)]]))
    for c0 in candidates:
        rows = _rows(A, c0, x, n)
        # the first k with rows[0..k] dependent
        for k in range(1, n + 1):
            coefficients = _dependence(rows[:k + 1])
            if coefficients is not None:
                if k == n:
                    return c0, _operator(coefficients, x)
                break
    raise ValueError("no cyclic vector among the candidates")


def _operator(coefficients: list[Expr], x: Symbol) -> LinearOperator:
    denominators = [as_expr(cancel(c).as_numer_denom()[1]) for c in coefficients]
    common = as_expr(lcm_list(denominators))
    return LinearOperator([as_expr(cancel(c*common)) for c in coefficients], x)


def system_to_scalar(A: MutableDenseMatrix, x: Symbol) -> tuple[LinearOperator, MutableDenseMatrix]:
    """The scalar operator of a cyclic vector and the matrix ``M`` with
    ``(u, u', ..., u**(n-1))**T = M Y``."""
    c0, L = cyclic_vector(A, x)
    rows = _rows(A, c0, x, A.shape[0] - 1)
    return L, Matrix.vstack(*rows)


def _vectors(L: LinearOperator, M: MutableDenseMatrix, scalar_solutions: list[Expr], x: Symbol) -> list[MutableDenseMatrix]:
    n = M.shape[0]
    inverse = M.inv().applyfunc(cancel)
    result: list[MutableDenseMatrix] = []
    for u in scalar_solutions:
        derivatives = Matrix([u.diff(x, k) if k else u for k in range(n)])
        result.append((inverse*derivatives).applyfunc(lambda e: cancel(e) if e.is_rational_function(x) else e))
    return result


def dsolve_linear_system(A: MutableDenseMatrix, x: Symbol) -> list[MutableDenseMatrix]:
    """Independent solution vectors of ``Y' = A Y`` found through the
    scalar equation of a cyclic vector.

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_systems import dsolve_linear_system
    >>> sorted((list(Y) for Y in dsolve_linear_system(Matrix([[0, 1], [-1, 0]]), x)), key=str)
    [[exp(-I*x), -I*exp(-I*x)], [exp(I*x), I*exp(I*x)]]
    >>> dsolve_linear_system(Matrix([[1/x, 1], [0, 1/x]]), x)
    [Matrix([
    [x],
    [0]]), Matrix([
    [x**2],
    [   x]])]
    """
    L, M = system_to_scalar(A, x)
    f = Function('y')(x)
    scalar = _solve_operator(L, f, True, True)
    return _vectors(L, M, scalar, x)


def rational_system_solutions(A: MutableDenseMatrix, x: Symbol) -> list[MutableDenseMatrix]:
    """A basis of the rational solutions of ``Y' = A Y``.

    >>> from sympy import Matrix
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_systems import rational_system_solutions
    >>> rational_system_solutions(Matrix([[1/x, 1], [0, 1/x]]), x)
    [Matrix([
    [x],
    [0]]), Matrix([
    [x**2],
    [   x]])]
    """
    L, M = system_to_scalar(A, x)
    return _vectors(L, M, rational_solutions(L), x)
