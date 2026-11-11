"""The algebraic Riccati equation: the solutions substituted back, the
closed loop eigenvalues (numerically, by SymPy), the closed forms of the
scalar equation and of the double integrator."""
from __future__ import annotations

import random

import pytest
import sympy

from sympy_extras.core import Integer, Rational, Symbol
from sympy_extras.functions import sqrt
from sympy_extras.interop import to_sympy
from sympy_extras.matrices import ExprMatrix
from sympy_extras.matrices.riccati import (care_solve, closed_loop_is_stable, dare_solve, discrete_riccati_solutions, dlqr, lqr,
                                           riccati_solutions)


def _sympy_matrix(M: ExprMatrix) -> sympy.Matrix:
    return sympy.Matrix(M.rows, M.cols, [to_sympy(e) for e in M.entries])


def _residual_is_zero(A: ExprMatrix, B: ExprMatrix, Q: ExprMatrix, R: ExprMatrix, X: ExprMatrix) -> bool:
    SA, SB, SQ, SR, SX = (_sympy_matrix(M) for M in (A, B, Q, R, X))
    residual = SA.T*SX + SX*SA - SX*SB*SR.inv()*SB.T*SX + SQ
    return all(abs(sympy.N(residual[i, j], 40)) < sympy.Float('1e-30') for i in range(residual.rows)
               for j in range(residual.cols))


def _eigenvalues(M: sympy.Matrix) -> list[sympy.Expr]:
    """Numerical eigenvalues, from the characteristic polynomial with
    its coefficients evaluated to 40 digits."""
    lam = sympy.Symbol('lam')
    coefficients = [sympy.N(c, 40) for c in M.charpoly(lam).all_coeffs()]
    return list(sympy.Poly(coefficients, lam).nroots(n=25, maxsteps=200))


def _stable(A: ExprMatrix, B: ExprMatrix, R: ExprMatrix, X: ExprMatrix) -> bool:
    SA, SB, SR, SX = (_sympy_matrix(M) for M in (A, B, R, X))
    return all(sympy.re(value) < 0 for value in _eigenvalues(SA - SB*SR.inv()*SB.T*SX))


def test_scalar_equation() -> None:
    # 2 a x - b**2 x**2/r + q = 0: the stabilising root is r (a + sqrt(a**2 + q b**2/r))/b**2
    rnd = random.Random(81)
    for _ in range(15):
        a, b, q, r = rnd.randint(-3, 3), rnd.randint(1, 3), rnd.randint(1, 4), rnd.randint(1, 3)
        A, B, Q, R = (ExprMatrix.from_rows([[v]]) for v in (a, b, q, r))
        X = care_solve(A, B, Q, R)
        assert X is not None
        expected = sympy.Rational(r, b**2)*(a + sympy.sqrt(a**2 + sympy.Rational(q*b**2, r)))
        assert sympy.simplify(to_sympy(X[0, 0]) - expected) == 0
        assert len(riccati_solutions(A, B, Q, R)) == 2


def test_double_integrator() -> None:
    # Anderson, Moore: Q = I, R = 1 gives X = [[sqrt(3), 1], [1, sqrt(3)]], K = [1, sqrt(3)]
    A, B = ExprMatrix.from_rows([[0, 1], [0, 0]]), ExprMatrix.column([0, 1])
    Q, R = ExprMatrix.eye(2), ExprMatrix.eye(1)
    K, X = lqr(A, B, Q, R)
    root = sqrt(Integer(3))
    assert X == ExprMatrix.from_rows([[root, 1], [1, root]]) and K == ExprMatrix.from_rows([[1, root]])
    assert care_solve(A, B, Q, R) == X
    # the closed loop s**2 + sqrt(3) s + 1 is stable
    assert _stable(A, B, R, X)


def test_random_systems() -> None:
    rnd = random.Random(82)
    # SymPy's evaluation of the roots in the checks is what takes the time here, not the solver
    for trial in range(5):
        n, m = 2, rnd.randint(1, 2)
        A = ExprMatrix(n, n, [Integer(rnd.randint(-2, 2)) for _ in range(n*n)])
        B = ExprMatrix(n, m, [Integer(rnd.randint(-1, 2)) for _ in range(n*m)])
        C = ExprMatrix(n, n, [Integer(rnd.randint(-1, 1)) for _ in range(n*n)])
        Q = C.transpose()*C + ExprMatrix.eye(n)               # positive definite: the pair (A, Q) is observable
        R = ExprMatrix(m, m, [Rational(rnd.randint(1, 3), rnd.randint(1, 2)) if i == j else Integer(0)
                              for i in range(m) for j in range(m)])
        solutions = riccati_solutions(A, B, Q, R)
        for X in solutions:
            assert X == X.transpose() and _residual_is_zero(A, B, Q, R, X)
        stabilising = [X for X in solutions if _stable(A, B, R, X)]
        chosen = care_solve(A, B, Q, R)
        SA, SB = _sympy_matrix(A), _sympy_matrix(B)
        controllable = sympy.Matrix.hstack(SB, SA*SB).rank() == n
        if controllable:
            # the stabilising solution exists, is unique and positive definite
            assert chosen is not None and stabilising == [chosen]
            assert all(sympy.re(value) > 0 for value in _eigenvalues(_sympy_matrix(chosen)))
            K, same = lqr(A, B, Q, R)
            assert same == chosen
            SK = _sympy_matrix(K)
            expected = _sympy_matrix(R).inv()*SB.T*_sympy_matrix(chosen)
            assert all(abs(sympy.N(SK[i, j] - expected[i, j], 30)) < sympy.Float('1e-20') for i in range(m) for j in range(n))
        else:
            assert (chosen is None) == (not stabilising)


def test_dimension_three() -> None:
    # a chain of three integrators with unit weights
    A = ExprMatrix.from_rows([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    B = ExprMatrix.column([0, 0, 1])
    Q, R = ExprMatrix.eye(3), ExprMatrix.eye(1)
    X = care_solve(A, B, Q, R)
    assert X is not None and X == X.transpose()
    assert _residual_is_zero(A, B, Q, R, X) and _stable(A, B, R, X)


def test_stability_and_rejections() -> None:
    A, B = ExprMatrix.from_rows([[0, 1], [0, 0]]), ExprMatrix.column([0, 1])
    assert closed_loop_is_stable(A, B, ExprMatrix.from_rows([[2, 3]]))
    assert not closed_loop_is_stable(A, B, ExprMatrix.from_rows([[-1, 3]]))
    assert not closed_loop_is_stable(A, B, ExprMatrix.from_rows([[1, 0]]))        # eigenvalues on the imaginary axis
    one = ExprMatrix.eye(1)
    # an unstable mode which the input does not reach
    assert care_solve(ExprMatrix.from_rows([[1]]), ExprMatrix.zeros(1, 1), one, one) is None
    with pytest.raises(ValueError):
        lqr(ExprMatrix.from_rows([[1]]), ExprMatrix.zeros(1, 1), one, one)
    with pytest.raises(ValueError):
        care_solve(ExprMatrix.from_rows([[Symbol('a')]]), one, one, one)
    with pytest.raises(ValueError):
        care_solve(A, B, ExprMatrix.from_rows([[1, 2], [0, 1]]), one)
    with pytest.raises(ValueError):
        care_solve(A, B, ExprMatrix.eye(2), ExprMatrix.zeros(1, 1))
    with pytest.raises(ValueError):
        care_solve(A, ExprMatrix.eye(3), ExprMatrix.eye(2), one)
    # Q = 0 with a stable A: X = 0
    assert care_solve(ExprMatrix.from_rows([[-1]]), one, ExprMatrix.zeros(1, 1), one) == ExprMatrix.zeros(1, 1)


def _discrete_residual_is_zero(A: ExprMatrix, B: ExprMatrix, Q: ExprMatrix, R: ExprMatrix, X: ExprMatrix) -> bool:
    SA, SB, SQ, SR, SX = (_sympy_matrix(M) for M in (A, B, Q, R, X))
    residual = SA.T*SX*SA - SX - SA.T*SX*SB*(SR + SB.T*SX*SB).inv()*SB.T*SX*SA + SQ
    return all(abs(sympy.N(residual[i, j], 40)) < sympy.Float('1e-30') for i in range(residual.rows)
               for j in range(residual.cols))


def test_discrete_scalar_equation() -> None:
    # b**2 x**2 + (r - a**2 r - q b**2) x - q r = 0: the stabilising solution is the positive root
    rnd = random.Random(83)
    for _ in range(12):
        a, b, q, r = rnd.randint(-3, 3), rnd.randint(1, 3), rnd.randint(1, 4), rnd.randint(1, 3)
        A, B, Q, R = (ExprMatrix.from_rows([[v]]) for v in (a, b, q, r))
        X = dare_solve(A, B, Q, R)
        assert X is not None
        linear = r - a**2*r - q*b**2
        expected = (-linear + sympy.sqrt(linear**2 + 4*b**2*q*r))/(2*b**2)
        assert sympy.simplify(to_sympy(X[0, 0]) - expected) == 0
        K, same = dlqr(A, B, Q, R)
        assert same == X
        gain = sympy.nsimplify(to_sympy(K[0, 0]))
        assert abs(sympy.N(a - b*gain)) < 1 and sympy.simplify(gain - a*b*expected/(r + b**2*expected)) == 0


def test_discrete_systems() -> None:
    rnd = random.Random(84)
    done = 0
    while done < 4:
        A = ExprMatrix(2, 2, [Integer(rnd.randint(-2, 2)) for _ in range(4)])
        B = ExprMatrix(2, 1, [Integer(rnd.randint(-1, 2)) for _ in range(2)])
        SA, SB = _sympy_matrix(A), _sympy_matrix(B)
        if sympy.Matrix.hstack(SB, SA*SB).rank() < 2:
            continue
        done += 1
        Q, R = ExprMatrix.eye(2), ExprMatrix.eye(1)
        K, X = dlqr(A, B, Q, R)
        assert X == X.transpose() and _discrete_residual_is_zero(A, B, Q, R, X)
        assert X in discrete_riccati_solutions(A, B, Q, R)
        closed = SA - SB*_sympy_matrix(K)
        assert all(abs(value) < 1 for value in _eigenvalues(closed))
        assert all(sympy.re(value) > 0 for value in _eigenvalues(_sympy_matrix(X)))
    # an unstable mode which the input does not reach, and the rejections
    one = ExprMatrix.eye(1)
    assert dare_solve(ExprMatrix.from_rows([[2]]), ExprMatrix.zeros(1, 1), one, one) is None
    with pytest.raises(ValueError):
        dlqr(ExprMatrix.from_rows([[2]]), ExprMatrix.zeros(1, 1), one, one)
    with pytest.raises(ValueError):
        dare_solve(ExprMatrix.eye(2), one, one, one)
    # a stable system without cost on the state: X = 0, K = 0
    assert dlqr(ExprMatrix.from_rows([[Rational(1, 2)]]), one, ExprMatrix.zeros(1, 1), one) == (ExprMatrix.zeros(1, 1), ExprMatrix.zeros(1, 1))
