"""Functions of matrices: the exponential against mpmath's ``expm`` and
the differential equation it solves, symbolic powers against the powers
themselves, the components against their defining identities."""
from __future__ import annotations

import random

import mpmath
import pytest
import sympy

from sympy_extras.core import I, Integer, Rational, Symbol
from sympy_extras.functions import cos, exp, sin, sqrt
from sympy_extras.interop import sympy_expr, to_sympy
from sympy_extras.matrices import ExprMatrix
from sympy_extras.matrices.functions import matrix_components, matrix_exp, matrix_function, matrix_power

t, n, x, a, b = Symbol('t'), Symbol('n'), Symbol('x'), Symbol('a'), Symbol('b')
st, sn = sympy.Symbol('t'), sympy.Symbol('n')


def _sympy_matrix(M: ExprMatrix) -> sympy.Matrix:
    return sympy.Matrix(M.rows, M.cols, [to_sympy(e) for e in M.entries])


def _numeric_close(found: sympy.Matrix, expected: mpmath.matrix) -> bool:
    for i in range(found.rows):
        for j in range(found.cols):
            value = complex(sympy.N(found[i, j], 30))
            if abs(value - complex(expected[i, j])) > 1e-10*(1 + abs(complex(expected[i, j]))):
                return False
    return True


def test_exponential_against_expm() -> None:
    rnd = random.Random(41)
    done = 0
    while done < 25:
        size = rnd.randint(1, 3)
        rows = [[rnd.randint(-2, 2) for _ in range(size)] for _ in range(size)]
        if done % 4 == 0 and size > 1:
            rows[1] = list(rows[0])
        try:
            E = matrix_exp(ExprMatrix.from_rows(rows), t)
        except NotImplementedError:
            continue
        for point in (sympy.Rational(3, 10), sympy.Rational(-7, 5)):
            expected = mpmath.expm(mpmath.matrix(rows)*mpmath.mpf(point.p)/point.q)
            assert _numeric_close(_sympy_matrix(E).subs(st, point), expected), rows
        done += 1


def test_exponential_closed_forms() -> None:
    assert matrix_exp(ExprMatrix.from_rows([[0, 1], [-1, 0]]), t) == ExprMatrix.from_rows([[cos(t), sin(t)], [-sin(t), cos(t)]])
    assert matrix_exp(ExprMatrix.from_rows([[1, 1], [0, 1]]), t) == ExprMatrix.from_rows([[exp(t), t*exp(t)], [0, exp(t)]])
    assert matrix_exp(ExprMatrix.zeros(2, 2), t) == ExprMatrix.eye(2)
    assert matrix_exp(ExprMatrix.from_rows([[0, 1, 0], [0, 0, 1], [0, 0, 0]]), t) == ExprMatrix.from_rows(
        [[1, t, t**2/2], [0, 1, t], [0, 0, 1]])
    # parameters, complex entries: the differential equation E' = M E, E(0) = 1
    for M in (ExprMatrix.from_rows([[a, 1], [0, a]]), ExprMatrix.from_rows([[a, b], [b, a]]), ExprMatrix.from_rows([[0, a], [-a, 0]]),
              ExprMatrix.from_rows([[a, 1, 0], [0, a, 0], [0, 0, b]]), ExprMatrix.from_rows([[0, -I], [I, 0]]),
              ExprMatrix.from_rows([[I, 1], [0, I]]), ExprMatrix.from_rows([[1, 2], [3, 4]])):
        E = _sympy_matrix(matrix_exp(M, t))
        S = _sympy_matrix(M)
        assert sympy.simplify(E.subs(st, 0) - sympy.eye(M.rows)) == sympy.zeros(M.rows, M.rows)
        assert sympy.simplify(E.diff(st) - S*E) == sympy.zeros(M.rows, M.rows)
    with pytest.raises(NotImplementedError):
        matrix_exp(ExprMatrix.from_rows([[0, 0, 0, 0, 1], [1, 0, 0, 0, 1], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0]]), t)


def test_symbolic_powers() -> None:
    rnd = random.Random(42)
    done = 0
    while done < 8:
        size = rnd.randint(1, 3)
        rows = [[rnd.randint(-2, 2) for _ in range(size)] for _ in range(size)]
        S = sympy.Matrix(rows)
        if S.det() == 0:
            continue
        try:
            P = _sympy_matrix(matrix_power(ExprMatrix.from_rows(rows), n))
        except NotImplementedError:
            continue
        for k in (0, 1, 4):
            difference = P.subs(sn, k) - S**k
            assert all(abs(complex(sympy.N(difference[i, j], 30))) < 1e-18 for i in range(size) for j in range(size)), rows
        done += 1
    fibonacci = matrix_power(ExprMatrix.from_rows([[1, 1], [1, 0]]), n)
    values = [sympy.simplify(sympy_expr(to_sympy(fibonacci[0, 1])).subs(sn, k)) for k in range(8)]
    assert values == [0, 1, 1, 2, 3, 5, 8, 13]
    assert matrix_power(ExprMatrix.from_rows([[a, 1], [0, a]]), n)[0, 1] == a**n*n/a


def test_functions_and_components() -> None:
    root = matrix_function(ExprMatrix.from_rows([[5, 4], [4, 5]]), sqrt(x), x)
    assert root == ExprMatrix.from_rows([[2, 1], [1, 2]]) and root*root == ExprMatrix.from_rows([[5, 4], [4, 5]])
    M = ExprMatrix.from_rows([[2, 1, 0], [0, 2, 0], [1, 0, 3]])
    assert matrix_function(M, 1/x, x) == M.inv()
    assert matrix_function(M, x**3 - 2*x + 1, x) == M*M*M - 2*M + ExprMatrix.eye(3)
    assert matrix_function(ExprMatrix.from_rows([[Rational(1, 2), 0], [0, 4]]), sqrt(x), x) == ExprMatrix.from_rows(
        [[sqrt(Integer(2))/2, 0], [0, 2]])
    # the components: projectors which add up to one, and the nilpotent parts
    for matrix in (M, ExprMatrix.from_rows([[1, 1], [1, 0]]), ExprMatrix.from_rows([[a, 1], [0, b]]),
                   ExprMatrix.from_rows([[0, 1, 0], [0, 0, 1], [2, 0, 0]])):
        S = _sympy_matrix(matrix)
        sx = sympy.Symbol('x')
        total = sympy.zeros(matrix.rows, matrix.rows)
        for group in matrix_components(matrix, x):
            factor = sympy_expr(to_sympy(group.factor))
            projector = _sympy_matrix(group.components[0])
            for value in sympy.roots(factor, sx):
                E = projector.subs(sx, value)
                assert sympy.simplify(E*E - E) == sympy.zeros(matrix.rows, matrix.rows)
                assert sympy.simplify(E*S - S*E) == sympy.zeros(matrix.rows, matrix.rows)
                assert sympy.simplify((S - value*sympy.eye(matrix.rows))**len(group.components)*E) == sympy.zeros(matrix.rows, matrix.rows)
                total += E
        assert sympy.simplify(total - sympy.eye(matrix.rows)) == sympy.zeros(matrix.rows, matrix.rows)
