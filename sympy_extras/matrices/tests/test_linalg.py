"""Linear algebra on matrices of expressions, against SymPy's matrices
(random matrices with polynomial and rational entries) and the defining
identities; kernels with relations go through the zero test."""
from __future__ import annotations

import random

import pytest
import sympy

from sympy_extras.core import I, Integer, Rational, Symbol
from sympy_extras.core.expr import Expr
from sympy_extras.functions import cos, sin, sqrt
from sympy_extras.interop import to_sympy
from sympy_extras.matrices import ExprMatrix
from sympy_extras.matrices.linalg import (adjugate, charpoly, charpoly_coefficients, det, inverse, linear_solve,
                                          nullspace, rank, rref)

x, y, z, t = Symbol('x'), Symbol('y'), Symbol('z'), Symbol('t')


def _sympy_matrix(M: ExprMatrix) -> sympy.Matrix:
    return sympy.Matrix(M.rows, M.cols, [to_sympy(e) for e in M.entries])


def _same(a: sympy.Matrix, b: sympy.Matrix) -> bool:
    return a.shape == b.shape and all(sympy.cancel(sympy.together(a[i, j] - b[i, j])) == 0
                                      for i in range(a.rows) for j in range(a.cols))


def _entry(rnd: random.Random, rational: bool) -> Expr:
    choices: list[Expr] = [Integer(0), Integer(1), Integer(-2), Rational(1, 3), x, y, x + y, x*y - 1, x**2, 2*y - x]
    if rational:
        choices += [1/x, y/(x + 1), (x - y)/(y + 2)]
    return rnd.choice(choices)


def _random_matrix(rnd: random.Random, n: int, m: int, rational: bool) -> ExprMatrix:
    return ExprMatrix(n, m, [_entry(rnd, rational) for _ in range(n*m)])


def test_square_matrices_against_sympy() -> None:
    rnd = random.Random(5)
    for trial in range(20):
        n = rnd.randint(1, 4)
        M = _random_matrix(rnd, n, n, trial % 2 == 1)
        S = _sympy_matrix(M)
        determinant = S.det(method='berkowitz')
        assert sympy.cancel(to_sympy(det(M)) - determinant) == 0
        assert sympy.cancel(to_sympy(charpoly(M, t)) - S.charpoly(sympy.Symbol('t')).as_expr()) == 0
        assert _same(_sympy_matrix(adjugate(M)), S.adjugate())
        if sympy.cancel(determinant) != 0:
            assert _same(S*_sympy_matrix(inverse(M)), sympy.eye(n))
        else:
            with pytest.raises(ValueError):
                inverse(M)


def test_rectangular_matrices_against_sympy() -> None:
    rnd = random.Random(6)
    for trial in range(20):
        n, m = rnd.randint(1, 4), rnd.randint(1, 4)
        M = _random_matrix(rnd, n, m, trial % 2 == 1)
        S = _sympy_matrix(M)
        expected, pivots = S.rref(simplify=True)
        reduced, found = rref(M)
        assert found == tuple(pivots) and rank(M) == len(pivots)
        assert _same(_sympy_matrix(reduced), expected)
        basis = nullspace(M)
        assert len(basis) == m - len(pivots)
        for v in basis:
            assert _same(S*_sympy_matrix(v), sympy.zeros(n, 1))
        B = _random_matrix(rnd, n, 1, False)
        solution = linear_solve(M, B)
        consistent = S.row_join(_sympy_matrix(B)).rref(simplify=True)[1][-1:] != (m,)
        assert (solution is not None) == consistent
        if solution is not None:
            assert _same(S*_sympy_matrix(solution.particular), _sympy_matrix(B))
            assert len(solution.nullspace) == m - len(pivots) and solution.is_unique == (len(pivots) == m)


def test_numbers() -> None:
    hilbert = ExprMatrix.from_rows([[Rational(1, i + j + 1) for j in range(5)] for i in range(5)])
    assert det(hilbert) == Rational(1, 266716800000)
    assert inverse(hilbert)[4, 4] == 44100 and inverse(hilbert)*hilbert == ExprMatrix.eye(5)
    assert charpoly_coefficients(ExprMatrix.from_rows([[2, 0], [0, 3]])) == [1, -5, 6]
    assert rank(ExprMatrix.zeros(2, 3)) == 0 and det(ExprMatrix(0, 0, [])) == 1
    with pytest.raises(ValueError):
        det(ExprMatrix.zeros(2, 3))
    with pytest.raises(ValueError):
        inverse(ExprMatrix.from_rows([[1, 2], [2, 4]]))


def test_determinants_hold_under_relations() -> None:
    # polynomial identities in the kernels: no zero test is needed
    rotation = ExprMatrix.from_rows([[cos(t), -sin(t), 0], [sin(t), cos(t), 0], [0, 0, 1]])
    assert det(rotation) == sin(t)**2 + cos(t)**2
    assert (rotation*adjugate(rotation))[0, 1] == 0
    M = ExprMatrix.from_rows([[sqrt(2), 1], [1, sqrt(2)]])
    assert det(M) == 1 and charpoly(M, x).subs(x, sqrt(2) + 1).expand() == 0
    assert det(ExprMatrix.from_rows([[I, 1], [1, I]])) == -2


def test_pivots_are_certified() -> None:
    rotation = ExprMatrix.from_rows([[cos(t), -sin(t)], [sin(t), cos(t)]])
    inverted = inverse(rotation)
    product = _sympy_matrix(rotation)*_sympy_matrix(inverted)
    assert all(sympy.simplify(product[i, j] - (1 if i == j else 0)) == 0 for i in range(2) for j in range(2))
    assert rank(rotation) == 2
    # the second pivot is sin(t)**2 + cos(t)**2 - 1: zero, though not as a polynomial
    hidden = ExprMatrix.from_rows([[sin(t)**2 + cos(t)**2, 1], [1, 1]])
    assert det(hidden) == sin(t)**2 + cos(t)**2 - 1
    for operation in (rank, rref, nullspace, inverse):
        with pytest.raises(NotImplementedError):
            operation(hidden)
    with pytest.raises(NotImplementedError):
        linear_solve(hidden, ExprMatrix.column([1, 2]))


def test_larger_determinant() -> None:
    # 8 x 8 with entries a*x + b*y + c: SymPy needs tens of seconds
    rnd = random.Random(0)
    M = ExprMatrix(8, 8, [rnd.randint(-3, 3)*x + rnd.randint(-3, 3)*y + rnd.randint(-3, 3) for _ in range(64)])
    determinant = det(M)
    S = _sympy_matrix(M)
    points: list[tuple[sympy.Expr, sympy.Expr]] = [(sympy.Integer(2), sympy.Integer(-3)), (sympy.Rational(1, 2), sympy.Integer(5))]
    for p, q in points:
        sx, sy = sympy.Symbol('x'), sympy.Symbol('y')
        assert to_sympy(determinant).subs(sx, p).subs(sy, q) == S.subs(sx, p).subs(sy, q).det()
    assert sympy.Poly(to_sympy(determinant), *sympy.symbols('x y')).total_degree() <= 8
