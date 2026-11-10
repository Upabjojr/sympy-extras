"""The exact singular value decomposition against mpmath's numeric one:
the singular values, the product, and the orthonormality of the factors
(evaluated to 30 digits)."""
from __future__ import annotations

import random

import mpmath
import pytest
import sympy

from sympy_extras.core import Integer, Rational, Symbol, sqrt
from sympy_extras.interop import sympy_expr, to_sympy
from sympy_extras.matrices import ExprMatrix
from sympy_extras.matrices.svd import singular_values, svd


def _numeric(M: ExprMatrix) -> mpmath.matrix:
    result = mpmath.matrix(M.rows, M.cols)
    for i in range(M.rows):
        for j in range(M.cols):
            value = complex(sympy.N(sympy_expr(to_sympy(M[i, j])), 30))
            assert abs(value.imag) < 1e-20
            result[i, j] = mpmath.mpf(sympy.re(sympy.N(sympy_expr(to_sympy(M[i, j])), 30)))
    return result


def _close(A: mpmath.matrix, B: mpmath.matrix) -> bool:
    return A.rows == B.rows and A.cols == B.cols and all(abs(A[i, j] - B[i, j]) < 1e-12
                                                         for i in range(A.rows) for j in range(A.cols))


def _check(rows: list[list[int]]) -> None:
    M = ExprMatrix.from_rows(rows)
    U, S, V = svd(M)
    u, s, v = _numeric(U), _numeric(S), _numeric(V)
    A = mpmath.matrix(rows)
    expected = [x for x in mpmath.svd_r(A, compute_uv=False) if x > 1e-12]
    r = len(expected)
    assert S.rows == S.cols == r and U.shape == (M.rows, r) and V.shape == (M.cols, r)
    assert all(abs(s[i, i] - expected[i]) < 1e-12 for i in range(r))
    assert all(s[i, i] >= s[i + 1, i + 1] for i in range(r - 1))
    if r:
        assert _close(u*s*v.T, A) and _close(u.T*u, mpmath.eye(r)) and _close(v.T*v, mpmath.eye(r))
    else:
        assert all(e == 0 for line in rows for e in line)


def test_random_integer_matrices() -> None:
    rnd = random.Random(3)
    for trial in range(30):
        m, n = rnd.randint(1, 3), rnd.randint(1, 3)
        rows = [[rnd.randint(-2, 2) for _ in range(n)] for _ in range(m)]
        if trial % 5 == 0 and m > 1:
            rows[1] = list(rows[0])
        _check(rows)


def test_particular_matrices() -> None:
    # repeated singular values (Gram-Schmidt inside the eigenspace), zero matrix, casus irreducibilis
    for rows in ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 1, 0], [1, -1, 0], [0, 0, 2]], [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
                 [[0, 0], [0, 0]], [[1, 1, 1], [1, -1, 0], [1, 1, -2], [3, 0, 1]], [[-1, 0, 2], [2, 2, -2], [-1, 2, 0]],
                 [[3, 0], [4, 5]], [[1, 2, 3, 4]], [[1], [2], [2]]):
        _check(rows)
    U, S, V = svd(ExprMatrix.from_rows([[3, 0], [4, 5]]))
    assert S == ExprMatrix.from_rows([[3*sqrt(Integer(5)), 0], [0, sqrt(Integer(5))]])
    half = ExprMatrix.from_rows([[Rational(1, 2), 0], [0, Rational(1, 3)]])
    assert svd(half)[1] == half


def test_singular_values() -> None:
    assert singular_values(ExprMatrix.from_rows([[3, 0], [4, 5]])) == [3*sqrt(Integer(5)), sqrt(Integer(5))]
    assert singular_values(ExprMatrix.from_rows([[1, 1], [1, 1]])) == [2, 0]
    assert singular_values(ExprMatrix.from_rows([[1, 2, 2]])) == [3, 0, 0]
    a = Symbol('a')
    assert set(singular_values(ExprMatrix.from_rows([[a, 0], [0, 2]]))) == {Integer(2), sqrt(a**2)}
    with pytest.raises(NotImplementedError):
        svd(ExprMatrix.from_rows([[a, 0], [0, 2]]))
    # five distinct singular values whose squares are the roots of an unsolvable quintic
    hard = ExprMatrix.from_rows([[1, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 1, 1], [1, 0, 0, 0, 2]])
    with pytest.raises(NotImplementedError):
        svd(hard)
