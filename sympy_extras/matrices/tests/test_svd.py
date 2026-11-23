"""The exact singular value decomposition against mpmath's numeric one:
the singular values, the product, and the orthonormality of the factors
(evaluated to 30 digits)."""
from __future__ import annotations

import random

import mpmath
import pytest
import sympy

from sympy_extras.core import Float, I, Integer, Rational, Symbol, sqrt
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
    # sympy/sympy#30900 M3-4: a 1 by 3 matrix has one singular value, as its transpose (this was [3, 0, 0], as many
    # values as columns, like SymPy: a wide matrix of full rank then had the condition number zoo)
    assert singular_values(ExprMatrix.from_rows([[1, 2, 2]])) == [3]
    assert singular_values(ExprMatrix.from_rows([[1], [2], [2]])) == [3]
    assert singular_values(ExprMatrix.from_rows([[1, 0, 0], [0, 0, 0]])) == [1, 0]
    assert singular_values(ExprMatrix.from_rows([[1, I, 0]])) == [sqrt(Integer(2))]
    a = Symbol('a')
    assert set(singular_values(ExprMatrix.from_rows([[a, 0], [0, 2]]))) == {Integer(2), sqrt(a**2)}
    with pytest.raises(NotImplementedError):
        svd(ExprMatrix.from_rows([[a, 0], [0, 2]]))
    # five distinct singular values whose squares are the roots of an unsolvable quintic
    hard = ExprMatrix.from_rows([[1, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 1, 1], [1, 0, 0, 0, 2]])
    with pytest.raises(NotImplementedError):
        svd(hard)


def test_singular_values_are_ordered_numerically() -> None:
    # sympy/sympy#30900 M2-10, M3-3: three real roots of a cubic (the radical form goes through complex numbers, and
    # float() of them raised TypeError), four of a quartic; the order is the numerical one
    for rows in ([[-2, 1, 3], [3, 3, -3], [-1, -3, 0]], [[1, 2, 0, 0], [0, 1, 3, 0], [0, 0, 1, 1], [1, 0, 0, 1]]):
        M = ExprMatrix.from_rows(rows)
        values = [complex(sympy_expr(to_sympy(v)).evalf(30)) for v in singular_values(M)]
        assert all(abs(v.imag) < 1e-12 for v in values)
        found = [v.real for v in values]
        assert found == sorted(found, reverse=True)
        gram = sympy.Matrix(rows).T*sympy.Matrix(rows)
        x = sympy.Symbol('x')
        expected = sorted((float(sympy.sqrt(r)) for r in sympy.Poly(gram.charpoly(x).as_expr(), x).nroots(n=30)), reverse=True)
        assert all(abs(u - v) < 1e-9 for u, v in zip(found, expected))
        U, S, V = svd(M)
        diagonal = [complex(sympy_expr(to_sympy(S[i, i])).evalf(30)).real for i in range(S.rows)]
        assert all(abs(u - v) < 1e-9 for u, v in zip(diagonal, expected))
    assert singular_values(ExprMatrix.from_rows([[3, 0], [0, sqrt(Integer(2))]])) == [3, sqrt(Integer(2))]


def test_singular_values_of_complex_matrices() -> None:
    # the eigenvalues of the conjugate transpose times the matrix, not of the transpose times it (which is not
    # Hermitian: sympy/sympy#30900 M2-2, M2-5 are the same slip in SymPy's solvers)
    assert singular_values(ExprMatrix.from_rows([[1, I], [0, 1]])) == [sqrt(sqrt(Integer(5))/2 + Rational(3, 2)),
                                                                     sqrt(Rational(3, 2) - sqrt(Integer(5))/2)]
    assert singular_values(ExprMatrix.from_rows([[1, I], [I, -1]])) == [2, 0]
    assert singular_values(ExprMatrix.from_rows([[1], [I]])) == [sqrt(Integer(2))]
    assert singular_values(ExprMatrix.from_rows([[3*I, 0], [0, 4]])) == [4, 3]


def test_singular_values_of_float_matrices() -> None:
    # sympy/sympy#30900 M3-3 (sympy/sympy#17332): SymPy gives [9.77e-16*I, 14.0, 0] for this matrix of rank 1 (not
    # ordered, and a rounding error as a singular value); here it raised NotImplementedError. The Gram matrix of the
    # numbers stored is exact, and so are its zero eigenvalues
    M = ExprMatrix.from_rows([[Float(1.0*(i + 1)*(j + 1)) for j in range(3)] for i in range(3)])
    assert singular_values(M) == [Float(14.0), Integer(0), Integer(0)]
    rng = random.Random(2)
    for rows, cols in ((2, 2), (3, 3), (5, 5), (2, 4), (4, 2)):
        data = [[rng.uniform(-1, 1) for _ in range(cols)] for _ in range(rows)]
        F = ExprMatrix.from_rows([[Float(v) for v in row] for row in data])
        values = singular_values(F)
        expected = mpmath.svd_r(mpmath.matrix(data), compute_uv=False)
        assert len(values) == min(rows, cols)
        assert all(abs(complex(v) - float(e)) < 1e-12 for v, e in zip(values, expected))
        if rows <= 3:
            U, S, V = svd(F)
            product = U*S*V.transpose()
            assert all(abs(complex(product[i, j]) - data[i][j]) < 1e-10 for i in range(rows) for j in range(cols))
