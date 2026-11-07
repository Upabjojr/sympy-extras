"""``apart``: the partial fraction decomposition of a rational function
in one variable (the coefficients may depend on other symbols).

Ported from SymPy's ``sympy.polys.partfrac`` (``apart`` and
``apart_undetermined_coeffs``): numerator and denominator cancelled
(``Poly.cancel``), the polynomial part split off by division, the
denominator factored over ``QQ`` (or ``ZZ[a]``), and for each factor
``f**k`` the numerators of ``1/f, ..., 1/f**k`` taken with undetermined
coefficients, found by Gauss-Jordan elimination over the rational
functions of the parameters (SymPy calls ``solve``); every term is then
factored.

>>> from sympy_extras.core import Symbol
>>> x, a = Symbol('x'), Symbol('a')
>>> apart(1/((x - 1)*(x + 2)), x), apart((x**3 + 1)/(x**2 - 1), x)
(-1/(3*(x + 2)) + 1/(3*(x - 1)), x + 1/(x - 1))
>>> apart(1/((x - a)*(x + 1)), x)
-1/((a + 1)*(x + 1)) + 1/((-a + x)*(a + 1))

Not ported yet: ``full=True`` (Bronstein's algorithm with ``RootSum``),
algebraic coefficients (``extension=True``: ``sqrt(2)`` in the
coefficients), divisors whose leading coefficient depends on the
parameters (SymPy's fraction field ``ZZ(a)``), noncommutative factors.
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/polys/partfrac.py (apart, apart_undetermined_coeffs) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; the linear system solved by Gauss-Jordan elimination
# with cancel instead of solve; the cases listed above not ported.
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Optional

from sympy_extras.core.add import Add, add
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.mul import mul
from sympy_extras.core.numbers import Integer, Number
from sympy_extras.core.power import pow

from .polyerrors import CoercionFailed
from .polytools import Poly, _common_gens, cancel, factor

__all__ = ['apart']


def _solve_linear(matrix: list[list[Expr]], rhs: list[Expr]) -> list[Expr]:
    """The unique solution of a square linear system with entries rational
    functions (Gauss-Jordan elimination, every entry cancelled)."""
    n = len(rhs)
    rows = [[cancel(v) for v in row] + [cancel(r)] for row, r in zip(matrix, rhs)]
    for col in range(n):
        pivot = next((i for i in range(col, n) if not _is_zero(rows[i][col])), None)
        if pivot is None:
            raise ValueError('singular system in the partial fraction decomposition')
        rows[col], rows[pivot] = rows[pivot], rows[col]
        inverse = pow(rows[col][col], Integer(-1))
        rows[col] = [cancel(mul(v, inverse)) for v in rows[col]]
        for i in range(n):
            if i != col and not _is_zero(rows[i][col]):
                factor_i = rows[i][col]
                rows[i] = [cancel(add(v, mul(Integer(-1), factor_i, w))) for v, w in zip(rows[i], rows[col])]
    return [row[n] for row in rows]


def _is_zero(e: Expr) -> bool:
    return isinstance(e, Number) and e == 0


# PORTED FROM SYMPY: sympy/polys/partfrac.py:apart_undetermined_coeffs
def _undetermined_coefficients(P: Poly, Q: Poly) -> Expr:
    x = Q.gen
    _, factors = Q.factor_list()
    blocks: list[tuple[int, Poly, Poly, int]] = []
    for f, k in factors:
        n, q = f.degree(), Q
        for i in range(1, k + 1):
            q = q.exquo(f)
            blocks.append((n, q, f, i))
    size = sum(n for n, _, _, _ in blocks)
    columns: list[list[Expr]] = []
    for n, q, _, _ in blocks:
        for j in range(n):
            shifted = q * Poly(pow(x, Integer(n - 1 - j)), x)
            columns.append([shifted.nth(k) for k in range(size)])
    matrix = [[columns[c][r] for c in range(len(columns))] for r in range(size)]
    rhs = [P.nth(k) for k in range(size)]
    solution = _solve_linear(matrix, rhs)
    result: Expr = Integer(0)
    position = 0
    for n, _, f, i in blocks:
        h = add(*[mul(solution[position + j], pow(x, Integer(n - 1 - j))) for j in range(n)])
        position += n
        result = add(result, mul(h, pow(pow(f.as_expr(), Integer(i)), Integer(-1))))
    return result


# PORTED FROM SYMPY: sympy/polys/partfrac.py:apart
def apart(f: ExprLike, x: Optional[ExprLike] = None, full: bool = False) -> Expr:
    """The partial fraction decomposition of ``f`` in ``x`` (the only
    generator when ``x`` is not given)."""
    e = to_expr(f)
    if not e.args:
        return e
    if full:
        raise NotImplementedError('the full partial fraction decomposition (RootSum) is not ported yet')
    if e.is_commutative is False:
        raise NotImplementedError('apart of noncommutative expressions is not ported yet')
    numer, denom = e.as_numer_denom()
    gens = (to_expr(x),) if x is not None else _common_gens(numer, denom)
    if x is None and any(g.is_number for g in gens):
        raise CoercionFailed('algebraic coefficients (extension=True) are not ported yet')
    if len(gens) > 1:
        cancelled = cancel(e)
        if cancelled != e:
            return apart(cancelled, x, full)
        raise NotImplementedError('multivariate partial fraction decomposition')
    common, P, Q = Poly(numer, *gens).cancel(Poly(denom, *gens))
    polynomial, P = P.div(Q)
    if Q.degree() <= 1:
        partial = mul(P.as_expr(), pow(Q.as_expr(), Integer(-1)))
    else:
        partial = _undetermined_coefficients(P, Q)
    terms = add(*[factor(t) for t in (partial.args if isinstance(partial, Add) else (partial,))])
    return mul(common, add(polynomial.as_expr(), terms))
