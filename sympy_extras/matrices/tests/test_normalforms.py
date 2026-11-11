"""Normal forms: the Smith form against the determinantal divisors (gcds
of minors, computed by SymPy), the transformations multiplied back, the
Frobenius form by A P = P F, the minimal polynomial by the rank of the
powers of the matrix, the elementary divisors against the Jordan chains."""
from __future__ import annotations

import random
from itertools import combinations

import pytest
import sympy

from sympy_extras.core import I, Integer, Rational, Symbol
from sympy_extras.core.expr import Expr
from sympy_extras.functions import sin, sqrt
from sympy_extras.interop import sympy_expr, to_sympy
from sympy_extras.matrices import ExprMatrix
from sympy_extras.matrices.eigen import jordan_chains
from sympy_extras.matrices.normalforms import (companion_matrix, elementary_divisors, frobenius_form, invariant_factors,
                                               is_similar, matrix_minimal_polynomial, smith_normal_form)

x, a, b = Symbol('x'), Symbol('a'), Symbol('b')
sx = sympy.Symbol('x')


def _sympy_matrix(M: ExprMatrix) -> sympy.Matrix:
    return sympy.Matrix(M.rows, M.cols, [to_sympy(e) for e in M.entries])


def _same(A: sympy.Matrix, B: sympy.Matrix) -> bool:
    return bool(A.shape == B.shape and all(sympy.cancel(sympy.together(sympy.expand(A[i, j] - B[i, j]))) == 0
                                           for i in range(A.rows) for j in range(A.cols)))


def _determinantal_invariants(S: sympy.Matrix) -> list[sympy.Expr]:
    """The diagonal of the Smith form over K[x] from the gcds of the
    minors: s_k = d_k/d_(k-1)."""
    m, n = S.shape
    previous = sympy.Integer(1)
    found: list[sympy.Expr] = []
    for k in range(1, min(m, n) + 1):
        g = sympy.Integer(0)
        for rows in combinations(range(m), k):
            for cols in combinations(range(n), k):
                g = sympy.gcd(g, S.extract(list(rows), list(cols)).det(method='berkowitz'), sx)
        if g == 0:
            break
        g = sympy.Poly(g, sx).monic().as_expr()
        found.append(sympy.cancel(g/previous))
        previous = g
    return found


def _random_polynomial(rnd: random.Random, parameter: bool) -> Expr:
    choices: list[Expr] = [Integer(0), Integer(0), Integer(1), x, x - 1, x + 1, x**2 - 1, x*(x - 1), 2*x, x**2 + 1,
                           Rational(1, 2)*x + 1]
    if parameter:
        choices += [x - a, a, a*x + 1]
    return rnd.choice(choices)


def test_smith_form_random() -> None:
    rnd = random.Random(31)
    for trial in range(30):
        m, n = rnd.randint(1, 3), rnd.randint(1, 3)
        M = ExprMatrix(m, n, [_random_polynomial(rnd, trial % 3 == 2) for _ in range(m*n)])
        D = smith_normal_form(M, x)
        S, SU, SV, SD = _sympy_matrix(M), _sympy_matrix(D.U), _sympy_matrix(D.V), _sympy_matrix(D.S)
        assert _same(SU*S*SV, SD)
        # unimodular: the determinants are nonzero and free of x
        for T in (SU, SV):
            determinant = sympy.cancel(T.det(method='berkowitz'))
            assert determinant != 0 and sx not in determinant.free_symbols
        assert all(SD[i, j] == 0 for i in range(m) for j in range(n) if i != j)
        expected = _determinantal_invariants(S)
        got = [sympy_expr(to_sympy(e)) for e in D.invariants]
        assert len(got) == len(expected) and all(sympy.cancel(g - e) == 0 for g, e in zip(got, expected))
        assert all(SD[i, i] == 0 for i in range(len(got), min(m, n)))


def test_smith_form_known() -> None:
    # Gantmacher, ch. VI: the invariant polynomials of a Jordan block of size n are 1, ..., 1, (x - c)**n
    J = ExprMatrix.from_rows([[x - 2, -1, 0], [0, x - 2, -1], [0, 0, x - 2]])
    assert [sympy.factor(to_sympy(e)) for e in smith_normal_form(J, x).invariants] == [1, 1, (sx - 2)**3]
    # diag(x, x + 1): the entries are coprime, so the form is diag(1, x*(x + 1))
    assert smith_normal_form(ExprMatrix.from_rows([[x, 0], [0, x + 1]]), x).S == ExprMatrix.from_rows([[1, 0], [0, x**2 + x]])
    # an entry which only expands to a polynomial, a zero matrix, an entry which is not a polynomial
    assert smith_normal_form(ExprMatrix.from_rows([[(x + 1)**2 - x**2 - 2*x - 1, x]]), x).S == ExprMatrix.from_rows([[x, 0]])
    assert smith_normal_form(ExprMatrix.zeros(2, 2), x).invariants == []
    with pytest.raises(ValueError):
        smith_normal_form(ExprMatrix.from_rows([[1/x, 1]]), x)
    with pytest.raises(NotImplementedError):
        smith_normal_form(ExprMatrix.from_rows([[sin(a)*x, 1]]), x)


def _random_square(rnd: random.Random, n: int, kind: int) -> ExprMatrix:
    if kind == 0:
        return ExprMatrix(n, n, [Integer(rnd.randint(-2, 2)) for _ in range(n*n)])
    if kind == 1:
        # a block structure hidden by a change of basis: several invariant factors
        blocks = [rnd.choice([0, 1, 1, 2]) for _ in range(n)]
        D = sympy.diag(*blocks)
        for i in range(n - 1):
            if blocks[i] == blocks[i + 1] and rnd.random() < 0.4:
                D[i, i + 1] = 1
        while True:
            P = sympy.Matrix(n, n, [rnd.randint(-1, 2) for _ in range(n*n)])
            if P.det() != 0:
                break
        S = P*D*P.inv()
        return ExprMatrix(n, n, [Rational(int(e.p), int(e.q)) for e in S])
    choices: list[Expr] = [Integer(0), Integer(1), a, a + 1, Integer(-1), b]
    return ExprMatrix(n, n, [rnd.choice(choices) for _ in range(n*n)])


def test_invariant_factors_and_frobenius_random() -> None:
    rnd = random.Random(32)
    for trial in range(36):
        n = rnd.randint(1, 4)
        A = _random_square(rnd, n, trial % 3)
        S = _sympy_matrix(A)
        factors = [sympy_expr(to_sympy(e)) for e in invariant_factors(A, x)]
        expected = [e for e in _determinantal_invariants(sx*sympy.eye(n) - S) if sympy.degree(e, sx) > 0]
        assert len(factors) == len(expected) and all(sympy.cancel(f - e) == 0 for f, e in zip(factors, expected))
        # each divides the next, their product is the characteristic polynomial
        for f, g in zip(factors, factors[1:]):
            assert sympy.rem(g, f, sx) == 0
        assert sympy.cancel(sympy.prod(factors) - S.charpoly(sx).as_expr()) == 0
        P, F = frobenius_form(A)
        SP, SF = _sympy_matrix(P), _sympy_matrix(F)
        assert _same(S*SP, SP*SF) and sympy.cancel(SP.det(method='berkowitz')) != 0
        blocks = [_sympy_matrix(companion_matrix(e, x)) for e in invariant_factors(A, x)]
        assert _same(SF, sympy.diag(*blocks))
        assert is_similar(A, F)


def test_minimal_polynomial() -> None:
    rnd = random.Random(33)
    for trial in range(20):
        n = rnd.randint(1, 4)
        A = _random_square(rnd, n, trial % 2)
        S = _sympy_matrix(A)
        p = sympy.Poly(to_sympy(matrix_minimal_polynomial(A, x)), sx)
        assert p.LC() == 1
        # it annihilates the matrix, and its degree is the dimension of the algebra generated by the matrix
        value = sympy.zeros(n, n)
        for c in p.all_coeffs():
            value = value*S + c*sympy.eye(n)
        assert value == sympy.zeros(n, n)
        powers = sympy.Matrix([list((S**k).reshape(1, n*n)) for k in range(n + 1)])
        assert p.degree() == powers.rank()
    assert matrix_minimal_polynomial(ExprMatrix.zeros(0, 0), x) == 1
    assert matrix_minimal_polynomial(ExprMatrix.zeros(3, 3), x) == x
    assert matrix_minimal_polynomial(ExprMatrix.eye(3)*a, x) == x - a


def test_elementary_divisors_against_jordan_chains() -> None:
    rnd = random.Random(34)
    for _ in range(15):
        n = rnd.randint(2, 5)
        A = _random_square(rnd, n, 1)
        divisors = elementary_divisors(A, x)
        expected: list[tuple[Expr, int]] = []
        for group in jordan_chains(A, x):
            expected.extend((group.factor, size) for size in sorted(group.block_sizes))
        assert sorted(divisors, key=str) == sorted(expected, key=str)
        assert sum(sympy.degree(to_sympy(f), sx)*k for f, k in divisors) == n


def test_complex_and_irrational_entries() -> None:
    # the rotation by a quarter turn and its diagonal form are similar over QQ(i)
    R = ExprMatrix.from_rows([[0, -1], [1, 0]])
    assert is_similar(R, ExprMatrix.from_rows([[I, 0], [0, -I]])) and not is_similar(R, ExprMatrix.from_rows([[I, 0], [0, I]]))
    assert elementary_divisors(ExprMatrix.from_rows([[I, 1], [0, I]]), x) == [(x - I, 2)]
    M = ExprMatrix.from_rows([[sqrt(Integer(2)), 1], [0, sqrt(Integer(2))]])
    assert sympy.expand(to_sympy(matrix_minimal_polynomial(M, x)) - (sx - sympy.sqrt(2))**2) == 0
    P, F = frobenius_form(M)
    assert _same(_sympy_matrix(M)*_sympy_matrix(P), _sympy_matrix(P)*_sympy_matrix(F))


def test_similarity() -> None:
    rnd = random.Random(35)
    for _ in range(10):
        n = rnd.randint(2, 4)
        A = _random_square(rnd, n, 1)
        while True:
            P = sympy.Matrix(n, n, [rnd.randint(-2, 2) for _ in range(n*n)])
            if P.det() != 0:
                break
        B = P.inv()*_sympy_matrix(A)*P
        assert is_similar(A, ExprMatrix(n, n, [Rational(int(e.p), int(e.q)) for e in B]))
    # same characteristic and minimal polynomials, different invariant factors (sizes 2+2 and 2+1+1)
    N1 = ExprMatrix.from_rows([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
    N2 = ExprMatrix.from_rows([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    assert matrix_minimal_polynomial(N1, x) == matrix_minimal_polynomial(N2, x) and not is_similar(N1, N2)
    assert not is_similar(ExprMatrix.eye(2), ExprMatrix.eye(3))
    # with a parameter: [[a, 1], [0, b]] is similar to diag(a, b) for generic a, b
    assert is_similar(ExprMatrix.from_rows([[a, 1], [0, b]]), ExprMatrix.from_rows([[a, 0], [0, b]]))
    assert not is_similar(ExprMatrix.from_rows([[a, 1], [0, a]]), ExprMatrix.from_rows([[a, 0], [0, a]]))


def test_companion_matrix() -> None:
    p = 2*x**3 - 4*x + 6
    C = companion_matrix(p, x)
    assert sympy.expand(_sympy_matrix(C).charpoly(sx).as_expr() - sympy_expr(to_sympy(p))/2) == 0
    assert invariant_factors(C, x) == [x**3 - 2*x + 3]
    with pytest.raises(ValueError):
        companion_matrix(Integer(3), x)
