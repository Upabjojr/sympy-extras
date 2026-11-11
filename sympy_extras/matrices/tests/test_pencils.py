"""Pencils: A v = x B v modulo the factor, the eigenvalues of B**-1 A
for an invertible B, and the count of infinite eigenvalues."""
from __future__ import annotations

import random

import pytest
import sympy

from sympy_extras.core import I, Integer, Rational, Symbol, oo
from sympy_extras.core.expr import Expr
from sympy_extras.functions import sqrt
from sympy_extras.interop import sympy_expr, to_sympy
from sympy_extras.matrices import ExprMatrix
from sympy_extras.matrices.eigen import eigenvalues
from sympy_extras.matrices.linalg import inverse, normal, rank
from sympy_extras.matrices.pencils import (generalized_eigenspaces, generalized_eigenvalues, generalized_eigenvectors,
                                           pencil_polynomial)

x, a = Symbol('x'), Symbol('a')
sx = sympy.Symbol('x')


def _sympy_matrix(M: ExprMatrix) -> sympy.Matrix:
    return sympy.Matrix(M.rows, M.cols, [to_sympy(e) for e in M.entries])


def _random_matrix(rnd: random.Random, n: int, symbolic: bool) -> ExprMatrix:
    choices: list[Expr] = [Integer(0), Integer(0), Integer(1), Integer(-1), Integer(2), Rational(1, 2)]
    if symbolic:
        choices += [a, a + 1]
    return ExprMatrix(n, n, [rnd.choice(choices) for _ in range(n*n)])


def test_eigenspaces_random() -> None:
    rnd = random.Random(121)
    regular = 0
    for trial in range(40):
        n = rnd.randint(1, 3)
        A, B = _random_matrix(rnd, n, trial % 4 == 3), _random_matrix(rnd, n, False)
        SA, SB = _sympy_matrix(A), _sympy_matrix(B)
        determinant = sympy.expand((SA - sx*SB).det())
        ours = sympy_expr(to_sympy(pencil_polynomial(A, B, x)))
        if determinant == 0:
            assert ours == 0
            with pytest.raises(ValueError):
                generalized_eigenspaces(A, B, x)
            continue
        # the numerator of the determinant: the same up to a factor free of x
        ratio = sympy.cancel(ours/determinant)
        assert ratio != 0 and sx not in ratio.free_symbols
        regular += 1
        total = 0
        for space in generalized_eigenspaces(A, B, x):
            factor = sympy_expr(to_sympy(space.factor))
            total += sympy.degree(factor, sx)*space.multiplicity
            assert 1 <= len(space.basis) <= space.multiplicity
            for vector in space.basis:
                v = _sympy_matrix(vector)
                residual = SA*v - sx*SB*v
                for i in range(n):
                    numerator = sympy.fraction(sympy.together(residual[i, 0]))[0]
                    assert sympy.rem(sympy.expand(numerator), factor, sx) == 0
        assert total == sympy.degree(determinant, sx)
    assert regular > 15


def test_invertible_second_matrix() -> None:
    rnd = random.Random(122)
    done = 0
    while done < 10:
        n = rnd.randint(1, 3)
        A, B = _random_matrix(rnd, n, False), _random_matrix(rnd, n, False)
        if rank(B) < n:
            continue
        try:
            expected = eigenvalues((inverse(B)*A).applyfunc(normal))
            found = generalized_eigenvalues(A, B)
        except NotImplementedError:
            continue
        done += 1
        assert oo not in found
        assert sorted(found.values()) == sorted(expected.values())
        for value, multiplicity in found.items():
            assert any(sympy.simplify(sympy_expr(to_sympy(value)) - sympy_expr(to_sympy(other))) == 0 and multiplicity == m
                       for other, m in expected.items())
        for value, _, vectors in generalized_eigenvectors(A, B):
            for vector in vectors:
                residual = _sympy_matrix(A)*_sympy_matrix(vector) - sympy_expr(to_sympy(value))*_sympy_matrix(B)*_sympy_matrix(vector)
                assert all(sympy.simplify(e) == 0 for e in residual)


def test_infinite_eigenvalues_and_known_pencils() -> None:
    assert generalized_eigenvalues(ExprMatrix.eye(2), ExprMatrix.from_rows([[1, 0], [0, 0]])) == {Integer(1): 1, oo: 1}
    assert generalized_eigenvalues(ExprMatrix.eye(3), ExprMatrix.zeros(3, 3)) == {oo: 3}
    # two equal masses on two springs: the squared frequencies (3 -+ sqrt(5))/2 k/m
    k, m = Symbol('k', positive=True), Symbol('m', positive=True)
    K = ExprMatrix.from_rows([[2*k, -k], [-k, k]])
    M = ExprMatrix.from_rows([[m, 0], [0, m]])
    values = [sympy_expr(to_sympy(v)) for v in generalized_eigenvalues(K, M)]
    sk, sm = sympy.Symbol('k', positive=True), sympy.Symbol('m', positive=True)
    expected = [(3 - sympy.sqrt(5))*sk/(2*sm), (3 + sympy.sqrt(5))*sk/(2*sm)]
    assert all(any(sympy.simplify(v - e) == 0 for v in values) for e in expected) and len(values) == 2
    # a symmetric pencil with a definite B has real eigenvalues: 2 x**2 - 1
    found = generalized_eigenvalues(ExprMatrix.from_rows([[0, 1], [1, 0]]), ExprMatrix.from_rows([[1, 0], [0, 2]]))
    assert found == {-sqrt(Integer(2))/2: 1, sqrt(Integer(2))/2: 1}
    with pytest.raises(ValueError):
        generalized_eigenvalues(ExprMatrix.from_rows([[1, 0], [0, 0]]), ExprMatrix.from_rows([[1, 0], [0, 0]]))
    with pytest.raises(ValueError):
        generalized_eigenvalues(ExprMatrix.eye(2), ExprMatrix.eye(3))
    with pytest.raises(NotImplementedError):
        generalized_eigenvalues(ExprMatrix.from_rows([[I, 0], [0, 1]]), ExprMatrix.eye(2))
