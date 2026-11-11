"""Linear matrix equations: the solutions substituted back (in SymPy),
uniqueness against the eigenvalue criteria, and known closed forms."""
from __future__ import annotations

import random
from itertools import combinations

import pytest
import sympy

from sympy_extras.core import I, Integer, Rational, Symbol
from sympy_extras.core.expr import Expr
from sympy_extras.interop import to_sympy
from sympy_extras.matrices import ExprMatrix
from sympy_extras.matrices.decompositions import is_positive_definite
from sympy_extras.matrices.equations import (adjoint, commuting_matrices, discrete_lyapunov_solve, kronecker_product,
                                             linear_matrix_equation, lyapunov_solve, sylvester_solve, unvec, vec)

a, b = Symbol('a'), Symbol('b')


def _sympy_matrix(M: ExprMatrix) -> sympy.Matrix:
    return sympy.Matrix(M.rows, M.cols, [to_sympy(e) for e in M.entries])


def _is_zero(S: sympy.Matrix) -> bool:
    return all(sympy.cancel(sympy.together(sympy.expand(S[i, j]))) == 0 for i in range(S.rows) for j in range(S.cols))


def _random_matrix(rnd: random.Random, m: int, n: int, symbolic: bool = False) -> ExprMatrix:
    choices: list[Expr] = [Integer(0), Integer(1), Integer(-1), Integer(2), Integer(-3), Rational(1, 2)]
    if symbolic:
        choices += [a, a + 1]
    return ExprMatrix(m, n, [rnd.choice(choices) for _ in range(m*n)])


def test_kronecker_product_and_vec() -> None:
    rnd = random.Random(21)
    for _ in range(10):
        A, X, B = _random_matrix(rnd, 2, 3), _random_matrix(rnd, 3, 2), _random_matrix(rnd, 2, 4)
        # vec(A X B) = (B^T kron A) vec(X)
        assert vec(A*X*B) == kronecker_product(B.transpose(), A)*vec(X)
        assert unvec(vec(X), 3, 2) == X
        K = kronecker_product(A, B)
        assert _sympy_matrix(K) == sympy.kronecker_product(_sympy_matrix(A), _sympy_matrix(B))
    with pytest.raises(ValueError):
        unvec(ExprMatrix.column([1, 2, 3]), 2, 2)


def test_sylvester_random() -> None:
    rnd = random.Random(22)
    unique = several = 0
    for trial in range(40):
        m, n = rnd.randint(1, 3), rnd.randint(1, 3)
        A, B = _random_matrix(rnd, m, m, trial % 5 == 4), _random_matrix(rnd, n, n)
        if trial % 3 == 0:
            # a common eigenvalue of A and -B: several solutions
            B, n = -A.transpose(), m
        X = _random_matrix(rnd, m, n)
        C = (A*X + X*B).applyfunc(lambda e: e.expand())
        solution = sylvester_solve(A, B, C)
        assert solution is not None
        SA, SB, SC = _sympy_matrix(A), _sympy_matrix(B), _sympy_matrix(C)
        SX = _sympy_matrix(solution.particular)
        assert _is_zero(SA*SX + SX*SB - SC)
        for H in solution.homogeneous:
            SH = _sympy_matrix(H)
            assert _is_zero(SA*SH + SH*SB) and not _is_zero(SH)
        # unique exactly when the characteristic polynomials of A and -B are coprime
        lam = sympy.Symbol('lam')
        common = sympy.gcd(SA.charpoly(lam).as_expr(), (-SB).charpoly(lam).as_expr())
        assert solution.is_unique == (sympy.degree(common, lam) == 0)
        if solution.is_unique:
            unique += 1
            assert _is_zero(SX - _sympy_matrix(X))
        else:
            several += 1
    assert unique > 5 and several > 5


def test_sylvester_without_solution() -> None:
    # A X - X A = I has no solution: the trace of a commutator is zero
    A = ExprMatrix.from_rows([[1, 2], [3, 4]])
    assert sylvester_solve(A, -A, ExprMatrix.eye(2)) is None
    with pytest.raises(ValueError):
        sylvester_solve(A, ExprMatrix.eye(3), ExprMatrix.eye(2))
    with pytest.raises(ValueError):
        sylvester_solve(ExprMatrix.zeros(2, 3), A, ExprMatrix.eye(2))


def test_lyapunov_known() -> None:
    # scalar case: 2 a x + q = 0
    scalar = lyapunov_solve(ExprMatrix.from_rows([[a]]), ExprMatrix.from_rows([[b]]))
    assert scalar is not None and scalar.particular == ExprMatrix.from_rows([[-b/(2*a)]])
    # the damped oscillator x'' + k x' + x = 0 with Q = I: X = [[(k**2 + 2)/(2 k), -1/2], [-1/2, 1/k]],
    # checked by substitution
    k = Symbol('k')
    A = ExprMatrix.from_rows([[0, 1], [-1, -k]])
    solution = lyapunov_solve(A, ExprMatrix.eye(2))
    assert solution is not None and solution.is_unique
    SA, SX = _sympy_matrix(A), _sympy_matrix(solution.particular)
    assert _is_zero(SA*SX + SX*SA.T + sympy.eye(2))


def test_lyapunov_stable_matrices_give_positive_definite_solutions() -> None:
    rnd = random.Random(23)
    for _ in range(10):
        n = rnd.randint(1, 3)
        # triangular with a negative diagonal: stable
        A = ExprMatrix(n, n, [Integer(-rnd.randint(1, 4)) if i == j else (Integer(rnd.randint(-2, 2)) if j > i else Integer(0))
                              for i in range(n) for j in range(n)])
        B = _random_matrix(rnd, n, n)
        Q = B*B.transpose() + ExprMatrix.eye(n)
        solution = lyapunov_solve(A, Q)
        assert solution is not None and solution.is_unique
        X = solution.particular
        SA, SX = _sympy_matrix(A), _sympy_matrix(X)
        assert _is_zero(SA*SX + SX*SA.T + _sympy_matrix(Q))
        assert X == X.transpose() and is_positive_definite(X)


def test_discrete_lyapunov() -> None:
    rnd = random.Random(24)
    for _ in range(10):
        n = rnd.randint(1, 3)
        # triangular with a diagonal inside the unit disc: Schur stable, X = sum A**k Q (A^T)**k
        A = ExprMatrix(n, n, [Rational(rnd.randint(-2, 2), 3) if i == j else (Integer(rnd.randint(-1, 1)) if j > i else Integer(0))
                              for i in range(n) for j in range(n)])
        Q = ExprMatrix.eye(n)
        solution = discrete_lyapunov_solve(A, Q)
        assert solution is not None and solution.is_unique
        X = solution.particular
        SA, SX = _sympy_matrix(A), _sympy_matrix(X)
        assert _is_zero(SA*SX*SA.T - SX + sympy.eye(n))
        assert is_positive_definite(X)
        partial = sympy.zeros(n, n)
        power = sympy.eye(n)
        for _ in range(200):
            partial += power*power.T
            power = power*SA
        assert all(abs(sympy.N(partial[i, j] - SX[i, j])) < sympy.Float('1e-20') for i in range(n) for j in range(n))
    # the scalar equation a**2 x - x + 1 = 0
    scalar = discrete_lyapunov_solve(ExprMatrix.from_rows([[a]]), ExprMatrix.eye(1))
    assert scalar is not None and scalar.particular == ExprMatrix.from_rows([[-1/(a**2 - 1)]])


def test_complex_entries_use_the_conjugate_transpose() -> None:
    A = ExprMatrix.from_rows([[-1 + I, 2], [0, -2 - I]])
    assert adjoint(A) == ExprMatrix.from_rows([[-1 - I, 0], [2, -2 + I]])
    Q = ExprMatrix.from_rows([[2, I], [-I, 3]])
    solution = lyapunov_solve(A, Q)
    assert solution is not None and solution.is_unique
    SA, SX = _sympy_matrix(A), _sympy_matrix(solution.particular)
    assert _is_zero(SA*SX + SX*SA.H + _sympy_matrix(Q))
    # A stable and Q Hermitian positive definite: X Hermitian positive definite
    assert adjoint(solution.particular) == solution.particular and is_positive_definite(solution.particular)
    stein = discrete_lyapunov_solve(ExprMatrix.from_rows([[I/2, 1], [0, Rational(1, 3)]]), Q)
    assert stein is not None
    S2, X2 = sympy.Matrix([[sympy.I/2, 1], [0, sympy.Rational(1, 3)]]), _sympy_matrix(stein.particular)
    assert _is_zero(S2*X2*S2.H - X2 + _sympy_matrix(Q))


def test_general_equation() -> None:
    rnd = random.Random(25)
    for _ in range(10):
        A1, B1 = _random_matrix(rnd, 2, 3), _random_matrix(rnd, 2, 2)
        A2, B2 = _random_matrix(rnd, 2, 3), _random_matrix(rnd, 2, 2)
        X = _random_matrix(rnd, 3, 2)
        C = A1*X*B1 + A2*X*B2
        solution = linear_matrix_equation([(A1, B1), (A2, B2)], C)
        assert solution is not None
        for Y in [solution.particular] + [solution.particular + H for H in solution.homogeneous]:
            assert A1*Y*B1 + A2*Y*B2 == C
    with pytest.raises(ValueError):
        linear_matrix_equation([], ExprMatrix.eye(2))
    with pytest.raises(ValueError):
        linear_matrix_equation([(ExprMatrix.eye(2), ExprMatrix.eye(3))], ExprMatrix.eye(2))


def test_commuting_matrices() -> None:
    rnd = random.Random(26)
    for _ in range(15):
        n = rnd.randint(1, 4)
        A = _random_matrix(rnd, n, n)
        basis = commuting_matrices(A)
        SA = _sympy_matrix(A)
        for X in basis:
            assert A*X == X*A
        stacked = sympy.Matrix([[to_sympy(e) for e in X.entries] for X in basis])
        assert stacked.rank() == len(basis)
        # the dimension of the centraliser is the sum of deg(d_i)*(2 i - 1) over the invariant factors d_1 | d_2 ...
        # counted from the largest (Frobenius): computed from the Smith form of lam*I - A by determinantal divisors
        lam = sympy.Symbol('lam')
        C = lam*sympy.eye(n) - SA
        divisors = [sympy.Integer(1)]
        for k in range(1, n + 1):
            g = sympy.Integer(0)
            for rows in combinations(range(n), k):
                for cols in combinations(range(n), k):
                    g = sympy.gcd(g, C.extract(list(rows), list(cols)).det())
            divisors.append(g)
        degrees = sorted((sympy.degree(sympy.cancel(divisors[k]/divisors[k - 1]), lam) for k in range(1, n + 1)), reverse=True)
        assert len(basis) == sum(d*(2*i + 1) for i, d in enumerate(degrees))
    assert len(commuting_matrices(ExprMatrix.eye(3))) == 9
    # a matrix with a parameter and distinct eigenvalues: the polynomials in it
    assert len(commuting_matrices(ExprMatrix.from_rows([[a, 1], [0, b]]))) == 2
