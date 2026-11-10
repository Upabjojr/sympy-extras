"""Fraction-free linear algebra over typed domains, against SymPy's
matrices on random integer matrices and against the defining identities
over finite fields and polynomial rings."""
from __future__ import annotations

import random
from fractions import Fraction

import pytest
import sympy

from sympy_extras.matrices.domainmatrix import DomainMatrix
from sympy_extras.polys.domains import GF, QQ, ZZ
from sympy_extras.polys.domains.sparsepolynomialring import IntegerPolynomialRing


def _random_rows(rnd: random.Random, n: int, m: int) -> list[list[int]]:
    return [[rnd.choice([0, 0, 1, -1, 2, 3, -5, 7]) for _ in range(m)] for _ in range(n)]


def test_integer_matrices_against_sympy() -> None:
    rnd = random.Random(1)
    x = sympy.Symbol('x')
    for _ in range(150):
        n, m = rnd.randint(0, 6), rnd.randint(0, 6)
        rows = _random_rows(rnd, n, m)
        A = DomainMatrix(rows, ZZ, m)
        S = sympy.Matrix(n, m, lambda i, j: rows[i][j])
        reduced, den, pivots = A.rref_den()
        expected, expected_pivots = S.rref()
        assert pivots == list(expected_pivots) and den > 0
        assert all(Fraction(reduced.rows[i][j], den) == Fraction(int(expected[i, j].p), int(expected[i, j].q))
                   for i in range(n) for j in range(m))
        assert A.rank() == S.rank()
        basis = A.nullspace()
        assert len(basis) == m - len(pivots)
        assert all(sum(a*b for a, b in zip(r, v)) == 0 for r in rows for v in basis)
        if n == m:
            assert A.det() == S.det()
            assert A.charpoly() == [int(c) for c in S.charpoly(x).all_coeffs()]
            if n:
                adjugate = S.adjugate()
                assert A.adjugate().rows == [[int(adjugate[i, j]) for j in range(n)] for i in range(n)]
            if A.det() != 0:
                N, d = A.inv_den()
                assert A*N == DomainMatrix.eye(n, ZZ).scale(d) == N*A
            else:
                with pytest.raises(ValueError):
                    A.inv_den()


def test_linear_systems_against_sympy() -> None:
    rnd = random.Random(2)
    for _ in range(150):
        n, m, k = rnd.randint(1, 5), rnd.randint(1, 5), rnd.randint(1, 2)
        rows = _random_rows(rnd, n, m)
        A = DomainMatrix(rows, ZZ, m)
        B = DomainMatrix([[rnd.randint(-3, 3) for _ in range(k)] for _ in range(n)], ZZ, k)
        S = sympy.Matrix(n, m, lambda i, j: rows[i][j])
        consistent = all(S.row_join(sympy.Matrix(n, 1, lambda i, j: B.rows[i][c])).rank() == S.rank() for c in range(k))
        solution = A.solve_den(B)
        assert (solution is not None) == consistent
        if solution is not None:
            X, d = solution
            assert A*X == B.scale(d)


def test_fields() -> None:
    rnd = random.Random(3)
    for p in (2, 7, 101):
        K = GF(p)
        for _ in range(40):
            n = rnd.randint(1, 5)
            A = DomainMatrix([[rnd.randrange(p) for _ in range(n)] for _ in range(n)], K)
            determinant = A.det()
            assert determinant == int(sympy.Matrix(A.rows).det()) % p
            if determinant:
                assert A*A.inv() == DomainMatrix.eye(n, K)
            reduced, pivots = A.rref()
            assert all(reduced.rows[i][c] == 1 for i, c in enumerate(pivots)) and len(pivots) == A.rank()
            assert A.charpoly()[-1] == (determinant if n % 2 == 0 else -determinant) % p
    half = DomainMatrix([[Fraction(1, 2), Fraction(1, 3)], [Fraction(1, 4), Fraction(1, 5)]], QQ)
    assert half.det() == Fraction(1, 60) and half*half.inv() == DomainMatrix.eye(2, QQ)
    with pytest.raises(ValueError):
        DomainMatrix([[1, 2], [3, 4]], ZZ).inv()


def test_polynomial_entries() -> None:
    R = IntegerPolynomialRing(['x', 'y'])
    x, y = R.ring.gens
    one = R.one
    # the Vandermonde determinant
    V = DomainMatrix([[one, x, x*x], [one, y, y*y], [one, x + y, (x + y)*(x + y)]], R)
    assert V.det() == (y - x)*(x + y - x)*(x + y - y)
    A = DomainMatrix([[x, y, one], [y, x, y], [one, one, x + y]], R)
    N, d = A.inv_den()
    assert A*N == DomainMatrix.eye(3, R).scale(d) and d in (A.det(), -A.det())
    assert A*A.adjugate() == DomainMatrix.eye(3, R).scale(A.det())
    # Cayley-Hamilton
    coefficients = A.charpoly()
    total = DomainMatrix.zeros(3, 3, R)
    for k, c in enumerate(coefficients):
        total = total + (A**(3 - k)).scale(c)
    assert total == DomainMatrix.zeros(3, 3, R)
    # a singular matrix: the adjugate by Cayley-Hamilton
    singular = DomainMatrix([[x, y, x + y], [one, x, x + one], [y, y, y + y]], R)
    assert singular.det() == R.zero and singular.rank() == 2
    assert singular*singular.adjugate() == DomainMatrix.zeros(3, 3, R)
    assert singular.adjugate() != DomainMatrix.zeros(3, 3, R)
    vector = singular.nullspace()[0]
    assert all(sum((a*b for a, b in zip(row, vector)), R.zero) == R.zero for row in singular.rows)


def test_shapes() -> None:
    empty = DomainMatrix([], ZZ, 3)
    assert empty.shape == (0, 3) and empty.rank() == 0 and len(empty.nullspace()) == 3
    assert DomainMatrix([], ZZ).det() == 1 and DomainMatrix([], ZZ).charpoly() == [1]
    with pytest.raises(ValueError):
        DomainMatrix([[1, 2], [3]], ZZ)
    with pytest.raises(ValueError):
        DomainMatrix([[1, 2, 3]], ZZ).det()
    A = DomainMatrix([[1, 2], [3, 4]], ZZ)
    assert (A + A - A) == A and (-A).rows == [[-1, -2], [-3, -4]] and A.transpose().rows == [[1, 3], [2, 4]]
    assert A**3 == A*A*A and A**0 == DomainMatrix.eye(2, ZZ)
