"""Matrix decompositions: the factors multiplied back (in SymPy, an
independent arithmetic), their shapes (triangular, orthonormal,
diagonal), and the inertia against the signs of the eigenvalues SymPy
computes numerically."""
from __future__ import annotations

import random

import pytest
import sympy

from sympy_extras.core import I, Integer, Rational, Symbol
from sympy_extras.core.expr import Expr
from sympy_extras.functions import cos, sin, sqrt
from sympy_extras.interop import sympy_expr, to_sympy
from sympy_extras.matrices import ExprMatrix
from sympy_extras.matrices.decompositions import (cholesky, congruence_diagonalize, gram_schmidt, inertia, is_hermitian,
                                                  is_positive_definite, is_positive_semidefinite, ldl,
                                                  leading_principal_minors, lu, qr, rank_factorization)
from sympy_extras.matrices.linalg import rank

x, y, t = Symbol('x'), Symbol('y'), Symbol('t')


def _sympy_matrix(M: ExprMatrix) -> sympy.Matrix:
    return sympy.Matrix(M.rows, M.cols, [to_sympy(e) for e in M.entries])


def _zero(e: sympy.Expr) -> bool:
    return bool(sympy.simplify(sympy.radsimp(sympy.expand(sympy.together(e)))) == 0)


def _same(a: sympy.Matrix, b: sympy.Matrix) -> bool:
    return bool(a.shape == b.shape and all(_zero(a[i, j] - b[i, j]) for i in range(a.rows) for j in range(a.cols)))


def _real(S: sympy.Matrix) -> sympy.Matrix:
    """The symbols replaced by real ones (for the conjugations)."""
    return S.subs({s: sympy.Symbol(s.name, real=True) for s in S.free_symbols})


def _entry(rnd: random.Random, symbolic: bool) -> Expr:
    choices: list[Expr] = [Integer(0), Integer(0), Integer(1), Integer(-2), Integer(3), Rational(1, 2)]
    if symbolic:
        choices += [x, y, x + 1, x*y, 1/x]
    return rnd.choice(choices)


def _random_matrix(rnd: random.Random, m: int, n: int, symbolic: bool) -> ExprMatrix:
    return ExprMatrix(m, n, [_entry(rnd, symbolic) for _ in range(m*n)])


def _symmetric(rnd: random.Random, n: int, symbolic: bool) -> ExprMatrix:
    entries: dict[tuple[int, int], Expr] = {}
    for i in range(n):
        for j in range(i, n):
            entries[i, j] = entries[j, i] = _entry(rnd, symbolic)
    return ExprMatrix(n, n, [entries[i, j] for i in range(n) for j in range(n)])


def test_lu_random() -> None:
    rnd = random.Random(11)
    for trial in range(40):
        m, n = rnd.randint(1, 4), rnd.randint(1, 4)
        M = _random_matrix(rnd, m, n, trial % 2 == 1)
        F = lu(M)
        L, U, S = _sympy_matrix(F.L), _sympy_matrix(F.U), _sympy_matrix(M)
        assert sorted(F.permutation) == list(range(m))
        assert _same(_sympy_matrix(F.P)*S, L*U)
        assert L.is_lower and all(L[i, i] == 1 for i in range(m))
        assert F.rank == rank(M) == S.rank()
        # row echelon form: the first nonzero entries move to the right, the zero rows are last
        leading = [next((j for j in range(n) if U[i, j] != 0), n) for i in range(m)]
        assert all(a < b or a == b == n for a, b in zip(leading, leading[1:]))


def test_lu_exchanges_rows_only_when_needed() -> None:
    assert lu(ExprMatrix.from_rows([[2, 1], [4, 5]])).permutation == (0, 1)
    F = lu(ExprMatrix.from_rows([[0, 1], [1, 1]]))
    assert F.permutation == (1, 0) and F.U == ExprMatrix.from_rows([[1, 1], [0, 1]])
    empty = lu(ExprMatrix.zeros(2, 3))
    assert empty.rank == 0 and empty.L == ExprMatrix.eye(2) and empty.U == ExprMatrix.zeros(2, 3)


def test_lu_known() -> None:
    # Golub, Van Loan, Matrix Computations, Section 3.2: A = [[3, 5], [6, 7]] has L21 = 2, U = [[3, 5], [0, -3]]
    F = lu(ExprMatrix.from_rows([[3, 5], [6, 7]]))
    assert F.L == ExprMatrix.from_rows([[1, 0], [2, 1]]) and F.U == ExprMatrix.from_rows([[3, 5], [0, -3]])


def test_ldl_and_cholesky_random() -> None:
    rnd = random.Random(12)
    done = 0
    while done < 15:
        n = rnd.randint(1, 4)
        B = _random_matrix(rnd, n, n, False)
        M = B.transpose()*B                      # positive definite when B is invertible
        if rank(B) < n:
            with pytest.raises(ValueError):
                ldl(M)
            continue
        done += 1
        L, D = ldl(M)
        SL, SD, S = _sympy_matrix(L), _sympy_matrix(D), _sympy_matrix(M)
        assert SL.is_lower and SD.is_diagonal() and _same(SL*SD*SL.T, S)
        assert all(SD[i, i] > 0 for i in range(n))
        C = _sympy_matrix(cholesky(M))
        assert C.is_lower and _same(C*C.T, S)
        assert _same(C, S.cholesky())
        assert is_positive_definite(M) and inertia(M) == (n, 0, 0)


def test_ldl_symbolic() -> None:
    M = ExprMatrix.from_rows([[x, y, 1], [y, 2, x], [1, x, y]])
    L, D = ldl(M)
    assert _same(_sympy_matrix(L)*_sympy_matrix(D)*_sympy_matrix(L).T, _sympy_matrix(M))
    # the pivots are the quotients of the consecutive leading principal minors
    minors = [sympy_expr(to_sympy(e)) for e in leading_principal_minors(M)]
    assert _zero(sympy_expr(to_sympy(D[0, 0])) - minors[0]) and _zero(sympy_expr(to_sympy(D[1, 1])) - minors[1]/minors[0])
    assert _zero(sympy_expr(to_sympy(D[2, 2])) - minors[2]/minors[1])
    assert _zero(minors[2] - _sympy_matrix(M).det())


def test_ldl_rejections() -> None:
    with pytest.raises(ValueError):
        ldl(ExprMatrix.from_rows([[1, 2], [3, 4]]))
    with pytest.raises(ValueError):
        ldl(ExprMatrix.from_rows([[0, 1], [1, 0]]))
    with pytest.raises(ValueError):
        cholesky(ExprMatrix.from_rows([[1, 2], [2, 1]]))
    with pytest.raises(ValueError):
        ldl(ExprMatrix.from_rows([[1, 2, 3], [2, 1, 0]]))


def test_hermitian() -> None:
    M = ExprMatrix.from_rows([[2, 1 + I, 0], [1 - I, 3, I], [0, -I, 1]])
    assert is_hermitian(M) and not is_hermitian(M.transpose()*M.transpose()*ExprMatrix.from_rows([[I, 0, 0], [0, 1, 0], [0, 0, 1]]))
    S = _sympy_matrix(M)
    L, D = ldl(M)
    SL, SD = _sympy_matrix(L), _sympy_matrix(D)
    assert _same(SL*SD*SL.H, S) and all(SD[i, i].is_real for i in range(3))
    C = _sympy_matrix(cholesky(M))
    assert _same(C*C.H, S)
    P, Dc = congruence_diagonalize(M)
    assert _same(_sympy_matrix(P).H*S*_sympy_matrix(P), _sympy_matrix(Dc))
    assert inertia(M) == (3, 0, 0)
    Q, R = qr(M)
    SQ, SR = _sympy_matrix(Q), _sympy_matrix(R)
    assert _same(SQ*SR, S) and _same(SQ.H*SQ, sympy.eye(3)) and SR.is_upper


def test_qr_random() -> None:
    rnd = random.Random(13)
    for trial in range(24):
        m, n = rnd.randint(1, 4), rnd.randint(1, 3)
        M = _random_matrix(rnd, m, n, trial % 4 == 3)
        Q, R = qr(M)
        S = _real(_sympy_matrix(M))
        SQ, SR = _real(_sympy_matrix(Q)), _real(_sympy_matrix(R))
        r = S.rank()
        assert SQ.shape == (m, r) and SR.shape == (r, n)
        assert _same(SQ*SR, S)
        assert _same(SQ.T*SQ, sympy.eye(r))
        leading = [next(j for j in range(n) if SR[i, j] != 0) for i in range(r)]
        assert leading == sorted(set(leading))
        if not S.free_symbols:
            assert all(SR[i, leading[i]] > 0 for i in range(r))
            if r == n:
                assert _same(SR, S.QRdecomposition()[1])


def test_qr_with_square_roots() -> None:
    M = ExprMatrix.from_rows([[sqrt(Integer(2)), 1], [1, sqrt(Integer(3))]])
    Q, R = qr(M)
    SQ, SR = _sympy_matrix(Q), _sympy_matrix(R)
    assert _same(SQ*SR, _sympy_matrix(M)) and _same(SQ.T*SQ, sympy.eye(2)) and SR[1, 0] == 0


def test_gram_schmidt() -> None:
    vectors = [ExprMatrix.column([3, 1]), ExprMatrix.column([2, 2])]
    # the textbook example: v2 - (8/10) v1 = (-2/5, 6/5)
    assert gram_schmidt(vectors) == [ExprMatrix.column([3, 1]), ExprMatrix.column([Rational(-2, 5), Rational(6, 5)])]
    orthonormal = gram_schmidt(vectors, normalize=True)
    G = sympy.Matrix.hstack(*[_sympy_matrix(v) for v in orthonormal])
    assert _same(G.T*G, sympy.eye(2))
    assert gram_schmidt([]) == []
    with pytest.raises(ValueError):
        gram_schmidt([ExprMatrix.column([1, 2]), ExprMatrix.column([1, 2, 3])])


def test_rank_factorization() -> None:
    rnd = random.Random(14)
    for trial in range(20):
        M = _random_matrix(rnd, rnd.randint(1, 4), rnd.randint(1, 4), trial % 2 == 1)
        B, C = rank_factorization(M)
        r = rank(M)
        assert B.cols == C.rows == r
        assert _same(_sympy_matrix(B)*_sympy_matrix(C), _sympy_matrix(M)) if r else M == ExprMatrix.zeros(M.rows, M.cols)


def test_congruence_and_inertia_random() -> None:
    rnd = random.Random(15)
    for trial in range(40):
        n = rnd.randint(1, 5)
        M = _symmetric(rnd, n, False)
        P, D = congruence_diagonalize(M)
        S, SP, SD = _sympy_matrix(M), _sympy_matrix(P), _sympy_matrix(D)
        assert SD.is_diagonal() and SP.det() != 0 and SP.T*S*SP == SD
        # the signs of the eigenvalues, from the real roots of the characteristic polynomial
        lam = sympy.Symbol('lam')
        polynomial = sympy.Poly(S.charpoly(lam).as_expr(), lam)
        zero = next(k for k in range(n + 1) if polynomial.coeff_monomial(lam**k) != 0)
        counted = sympy.Poly(sympy.cancel(polynomial.as_expr()/lam**zero), lam)
        positive_roots = sum(1 for root in counted.nroots(n=30) if sympy.re(root) > 0)
        assert inertia(M) == (positive_roots, n - zero - positive_roots, zero)
        assert is_positive_semidefinite(M) == (n - zero - positive_roots == 0)
        assert is_positive_definite(M) == (positive_roots == n)


def test_congruence_symbolic() -> None:
    M = ExprMatrix.from_rows([[0, x, 1], [x, 0, y], [1, y, 0]])
    P, D = congruence_diagonalize(M)
    S, SP, SD = _sympy_matrix(M), _sympy_matrix(P), _sympy_matrix(D)
    assert SD.is_diagonal() and _same(SP.T*S*SP, SD) and _zero(SP.det()**2*S.det() - SD.det())
    with pytest.raises(ValueError):
        inertia(M)
    with pytest.raises(ValueError):
        congruence_diagonalize(ExprMatrix.from_rows([[1, 2], [3, 4]]))


def test_inertia_with_irrational_entries() -> None:
    root = sqrt(Integer(2))
    # eigenvalues sqrt(2) +- 1 and sqrt(2) +- 2
    assert inertia(ExprMatrix.from_rows([[root, 1], [1, root]])) == (2, 0, 0)
    assert inertia(ExprMatrix.from_rows([[root, 2], [2, root]])) == (1, 1, 0)
    assert inertia(ExprMatrix.from_rows([[root, root], [root, root]])) == (1, 0, 1)
    p = Symbol('p', positive=True)
    assert inertia(ExprMatrix.from_rows([[p, 0], [0, -p]])) == (1, 1, 0) and is_positive_definite(ExprMatrix.from_rows([[p + 1]]))


def test_congruence_of_a_zero_block() -> None:
    assert congruence_diagonalize(ExprMatrix.zeros(2, 2)) == (ExprMatrix.eye(2), ExprMatrix.zeros(2, 2))
    assert inertia(ExprMatrix.from_rows([[0, 0, 1], [0, 0, 0], [1, 0, 0]])) == (1, 1, 1)
    # Hermitian with a purely imaginary entry off a zero diagonal: 2*|a|**2 comes onto the diagonal
    M = ExprMatrix.from_rows([[0, I], [-I, 0]])
    P, D = congruence_diagonalize(M)
    assert _same(_sympy_matrix(P).H*_sympy_matrix(M)*_sympy_matrix(P), _sympy_matrix(D)) and inertia(M) == (1, 1, 0)


def test_kernels_with_relations() -> None:
    # the rotation matrix: the pivot cos(t) is nonzero as a function, sin**2 + cos**2 is left unsimplified
    M = ExprMatrix.from_rows([[cos(t), -sin(t)], [sin(t), cos(t)]])
    F = lu(M)
    assert _same(_sympy_matrix(F.L)*_sympy_matrix(F.U), _sympy_matrix(M))
    # a pivot which is zero by a relation between the kernels is not accepted
    with pytest.raises(NotImplementedError):
        lu(ExprMatrix.from_rows([[sin(t)**2 + cos(t)**2 - 1, 1], [1, 1]]))


def test_matrix_methods() -> None:
    M = ExprMatrix.from_rows([[4, 2], [2, 3]])
    assert M.lu().U == lu(M).U and M.qr() == qr(M) and M.cholesky() == cholesky(M) and M.ldl() == ldl(M)
