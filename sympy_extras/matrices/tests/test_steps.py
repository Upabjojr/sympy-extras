"""Step-by-step elimination: every step replayed with its elementary
matrix, the final matrices against the fraction-free results and SymPy."""
from __future__ import annotations

import random

import pytest
import sympy

from sympy_extras.core import I, Integer, Rational, Symbol
from sympy_extras.core.expr import Expr
from sympy_extras.functions import sin, sqrt
from sympy_extras.interop import to_sympy
from sympy_extras.matrices import ExprMatrix
from sympy_extras.matrices.linalg import det, inverse, normal, rank, rref
from sympy_extras.matrices.steps import (RowOperation, Step, determinant_steps, echelon_steps, format_steps, inverse_steps,
                                         rref_steps, solve_steps)

a, b = Symbol('a'), Symbol('b')


def _sympy_matrix(M: ExprMatrix) -> sympy.Matrix:
    return sympy.Matrix(M.rows, M.cols, [to_sympy(e) for e in M.entries])


def _random_matrix(rnd: random.Random, m: int, n: int, symbolic: bool) -> ExprMatrix:
    choices: list[Expr] = [Integer(0), Integer(0), Integer(1), Integer(2), Integer(-1), Integer(3), Rational(1, 2)]
    if symbolic:
        choices += [a, a + 1, b]
    return ExprMatrix(m, n, [rnd.choice(choices) for _ in range(m*n)])


def _equal(A: ExprMatrix, B: ExprMatrix) -> bool:
    return A.shape == B.shape and all(normal(p - q) == 0 for p, q in zip(A.entries, B.entries))


def _replay(M: ExprMatrix, steps: list[Step]) -> ExprMatrix:
    """Each step is the stated operation applied to the previous matrix
    (both through the elementary matrix and directly on the rows)."""
    current = M
    for step in steps:
        operation = step.operation
        applied = operation.apply(current)
        rows = [list(current.row(i)) for i in range(current.rows)]
        if operation.kind == 'swap':
            assert operation.target != operation.source
            rows[operation.target], rows[operation.source] = rows[operation.source], rows[operation.target]
        elif operation.kind == 'scale':
            assert operation.factor != 0
            rows[operation.target] = [operation.factor*e for e in rows[operation.target]]
        else:
            assert operation.target != operation.source
            rows[operation.target] = [e + operation.factor*f for e, f in zip(rows[operation.target], rows[operation.source])]
        assert _equal(applied, step.matrix) and _equal(ExprMatrix.from_rows(rows), step.matrix)
        current = step.matrix
    return current


def test_rref_steps_random() -> None:
    rnd = random.Random(41)
    for trial in range(40):
        M = _random_matrix(rnd, rnd.randint(1, 4), rnd.randint(1, 4), trial % 2 == 1)
        steps = rref_steps(M)
        assert _equal(_replay(M, steps), rref(M)[0])
        if not steps:
            assert M == rref(M)[0]


def test_echelon_steps_random() -> None:
    rnd = random.Random(42)
    for trial in range(40):
        m, n = rnd.randint(1, 4), rnd.randint(1, 4)
        M = _random_matrix(rnd, m, n, trial % 2 == 1)
        steps = echelon_steps(M)
        assert all(step.operation.kind != 'scale' for step in steps)
        # additions only from a row above to a row below
        assert all(step.operation.target > step.operation.source for step in steps if step.operation.kind == 'add')
        final = _replay(M, steps)
        leading = [next((j for j in range(n) if final[i, j] != 0), n) for i in range(m)]
        assert all(p < q or p == q == n for p, q in zip(leading, leading[1:]))
        assert sum(1 for p in leading if p < n) == rank(M)


def test_determinant_steps_random() -> None:
    rnd = random.Random(43)
    for trial in range(40):
        n = rnd.randint(1, 4)
        M = _random_matrix(rnd, n, n, trial % 2 == 1)
        D = determinant_steps(M)
        _replay(M, list(D.steps))
        assert normal(D.value - det(M)) == 0
        assert sympy.cancel(to_sympy(D.value) - _sympy_matrix(M).det()) == 0
    empty = determinant_steps(ExprMatrix.zeros(0, 0))
    assert empty.value == 1 and empty.steps == ()
    with pytest.raises(ValueError):
        determinant_steps(ExprMatrix.zeros(2, 3))


def test_inverse_steps_random() -> None:
    rnd = random.Random(44)
    inverted = 0
    for trial in range(40):
        n = rnd.randint(1, 4)
        M = _random_matrix(rnd, n, n, trial % 3 == 2)
        if rank(M) < n:
            with pytest.raises(ValueError):
                inverse_steps(M)
            continue
        inverted += 1
        steps = inverse_steps(M)
        augmented = M.col_insert(n, ExprMatrix.eye(n))
        final = _replay(augmented, steps)
        assert _equal(final, ExprMatrix.eye(n).col_insert(n, inverse(M)))
    assert inverted > 10


def test_solve_steps() -> None:
    # x + y + z = 6, 2 y + 5 z = -4, 2 x + 5 y - z = 27: the solution (5, 3, -2)
    A = ExprMatrix.from_rows([[1, 1, 1], [0, 2, 5], [2, 5, -1]])
    B = ExprMatrix.column([6, -4, 27])
    steps = solve_steps(A, B)
    assert _replay(A.col_insert(3, B), steps) == ExprMatrix.eye(3).col_insert(3, ExprMatrix.column([5, 3, -2]))
    # no pivot is taken in the column of the right-hand sides: an inconsistent system keeps its row 0 0 | c
    final = solve_steps(ExprMatrix.from_rows([[1, 1], [1, 1]]), ExprMatrix.column([1, 2]))[-1].matrix
    assert final == ExprMatrix.from_rows([[1, 1, 1], [0, 0, 1]])
    with pytest.raises(ValueError):
        solve_steps(A, ExprMatrix.column([1, 2]))


def test_textbook_example() -> None:
    # Strang, Introduction to Linear Algebra, Section 2.2: 2x + 4y - 2z = 2, 4x + 9y - 3z = 8, -2x - 3y + 7z = 10
    # has the multipliers 2, -1, 1 and the pivots 2, 1, 4
    M = ExprMatrix.from_rows([[2, 4, -2], [4, 9, -3], [-2, -3, 7]])
    steps = echelon_steps(M)
    assert [str(step.operation) for step in steps] == ['R2 <- R2 - (2)*R1', 'R3 <- R3 + R1', 'R3 <- R3 - R2']
    D = determinant_steps(M)
    assert D.diagonal == (2, 1, 4) and D.value == 8 and D.exchanges == 0


def test_operations_print() -> None:
    one: Expr = Integer(1)
    assert str(RowOperation('swap', 0, 2, one)) == 'R1 <-> R3'
    assert str(RowOperation('scale', 1, 1, Rational(-1, 3))) == 'R2 <- (-1/3)*R2'
    assert str(RowOperation('add', 1, 0, -a)) == 'R2 <- R2 - (a)*R1'
    assert str(RowOperation('add', 1, 0, a - 1)) == 'R2 <- R2 + (a - 1)*R1'
    assert str(RowOperation('add', 2, 0, Rational(3, 2))) == 'R3 <- R3 + (3/2)*R1'
    assert RowOperation('swap', 0, 1, one).matrix(2) == ExprMatrix.from_rows([[0, 1], [1, 0]])
    assert RowOperation('add', 1, 0, a).matrix(2) == ExprMatrix.from_rows([[1, 0], [a, 1]])
    with pytest.raises(ValueError):
        RowOperation('rotate', 0, 1, one)
    M = ExprMatrix.from_rows([[a, 1], [1, a]])
    text = format_steps(M, rref_steps(M))
    assert text.splitlines()[:3] == ['[a, 1]', '[1, a]', 'R1 <- (1/a)*R1']


def test_exact_fields() -> None:
    # a pivot which is zero only through I**2 = -1 or sqrt(2)**2 = 2 is not taken
    M = ExprMatrix.from_rows([[1, I], [I, -1]])
    assert _replay(M, rref_steps(M)) == ExprMatrix.from_rows([[1, I], [0, 0]])
    root = sqrt(Integer(2))
    N = ExprMatrix.from_rows([[root, 2], [1, root]])
    assert _replay(N, echelon_steps(N)).row(1) == (0, 0)
    assert determinant_steps(N).value == 0
    with pytest.raises(NotImplementedError):
        rref_steps(ExprMatrix.from_rows([[sin(a), 1], [1, 1]]))
