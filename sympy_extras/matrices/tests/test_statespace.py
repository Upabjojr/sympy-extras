"""State space systems: transfer functions against SymPy's matrix
inverse, pole placement by the characteristic polynomial of the closed
loop, realisations by the round trip, Kalman's rank criteria against the
eigenvector (Hautus) test."""
from __future__ import annotations

import random
from itertools import combinations

import pytest
import sympy

from sympy_extras.core import Integer, Rational, Symbol
from sympy_extras.core.expr import Expr
from sympy_extras.interop import sympy_expr, to_sympy
from sympy_extras.matrices import ExprMatrix
from sympy_extras.matrices.linalg import charpoly, inverse, normal
from sympy_extras.matrices.statespace import (controllable_decomposition, minimal_realization, observable_decomposition,
                                              controllability_matrix, is_controllable, is_observable, observability_matrix,
                                              pole_placement, realization, transfer_function)

s, a, b = Symbol('s'), Symbol('a'), Symbol('b')
ss = sympy.Symbol('s')


def _sympy_matrix(M: ExprMatrix) -> sympy.Matrix:
    return sympy.Matrix(M.rows, M.cols, [to_sympy(e) for e in M.entries])


def _random_matrix(rnd: random.Random, m: int, n: int, symbolic: bool = False) -> ExprMatrix:
    choices: list[Expr] = [Integer(0), Integer(0), Integer(1), Integer(-1), Integer(2), Rational(1, 2)]
    if symbolic:
        choices += [a, b]
    return ExprMatrix(m, n, [rnd.choice(choices) for _ in range(m*n)])


def test_transfer_function_against_sympy() -> None:
    rnd = random.Random(111)
    for trial in range(15):
        n, m, p = rnd.randint(1, 3), rnd.randint(1, 2), rnd.randint(1, 2)
        A, B, C = _random_matrix(rnd, n, n, trial % 3 == 2), _random_matrix(rnd, n, m), _random_matrix(rnd, p, n)
        D = _random_matrix(rnd, p, m) if trial % 2 else None
        G = _sympy_matrix(transfer_function(A, B, C, s, D))
        expected = _sympy_matrix(C)*(ss*sympy.eye(n) - _sympy_matrix(A)).inv()*_sympy_matrix(B)
        if D is not None:
            expected = expected + _sympy_matrix(D)
        assert (G - expected).applyfunc(sympy.cancel) == sympy.zeros(p, m)
    with pytest.raises(ValueError):
        transfer_function(ExprMatrix.from_rows([[s]]), ExprMatrix.eye(1), ExprMatrix.eye(1), s)
    with pytest.raises(ValueError):
        transfer_function(ExprMatrix.eye(2), ExprMatrix.eye(1), ExprMatrix.eye(2), s)


def test_controllability_and_observability() -> None:
    rnd = random.Random(112)
    seen = {True: 0, False: 0}
    for _ in range(30):
        n = rnd.randint(1, 3)
        A, B = _random_matrix(rnd, n, n), _random_matrix(rnd, n, 1)
        SA, SB = _sympy_matrix(A), _sympy_matrix(B)
        # Hautus: (s I - A, B) has the rank n for every s, i.e. its minors of order n have no common root
        pencil = sympy.Matrix.hstack(ss*sympy.eye(n) - SA, SB)
        common = sympy.Integer(0)
        for columns in combinations(range(n + 1), n):
            common = sympy.gcd(common, pencil.extract(list(range(n)), list(columns)).det())
        hautus = sympy.degree(common, ss) == 0
        assert is_controllable(A, B) == hautus
        assert is_observable(A.transpose(), B.transpose()) == hautus
        assert observability_matrix(A.transpose(), B.transpose()) == controllability_matrix(A, B).transpose()
        seen[hautus] += 1
    assert seen[True] > 3 and seen[False] > 3
    # generic controllability with a parameter
    assert is_controllable(ExprMatrix.from_rows([[0, 1], [a, 0]]), ExprMatrix.column([0, 1]))
    with pytest.raises(ValueError):
        controllability_matrix(ExprMatrix.eye(2), ExprMatrix.column([1, 2, 3]))


def test_pole_placement() -> None:
    rnd = random.Random(113)
    placed = 0
    while placed < 12:
        n = rnd.randint(1, 4)
        A, B = _random_matrix(rnd, n, n, placed % 4 == 3), _random_matrix(rnd, n, 1)
        if not is_controllable(A, B):
            with pytest.raises(ValueError):
                pole_placement(A, B, [Integer(-k - 1) for k in range(n)])
            continue
        placed += 1
        poles = [Integer(-rnd.randint(1, 4)) for _ in range(n)]
        K = pole_placement(A, B, poles)
        closed = (A - B*K).applyfunc(normal)
        target = sympy.prod([ss - to_sympy(p) for p in poles])
        assert sympy.cancel(sympy_expr(to_sympy(charpoly(closed, s))) - sympy.expand(target)) == 0
        # the same from the polynomial, also when it is not monic
        polynomial: Expr = Integer(3)
        for pole in poles:
            polynomial = polynomial*(s - pole)
        again = pole_placement(A, B, polynomial, s)
        assert (again - K).applyfunc(normal) == ExprMatrix.zeros(1, n)
    # symbolic poles: the double integrator with the closed loop s**2 + a s + b
    K = pole_placement(ExprMatrix.from_rows([[0, 1], [0, 0]]), ExprMatrix.column([0, 1]), s**2 + a*s + b, s)
    assert K == ExprMatrix.from_rows([[b, a]])
    with pytest.raises(ValueError):
        pole_placement(ExprMatrix.eye(2), ExprMatrix.eye(2), [1, 2])
    with pytest.raises(ValueError):
        pole_placement(ExprMatrix.from_rows([[0, 1], [0, 0]]), ExprMatrix.column([0, 1]), [1])


def test_realization_round_trip() -> None:
    rnd = random.Random(114)
    for _ in range(15):
        n = rnd.randint(1, 4)
        denominator = sum((Integer(rnd.randint(-3, 3))*s**k for k in range(n)), Integer(rnd.randint(1, 3))*s**n)
        numerator = sum((Integer(rnd.randint(-3, 3))*s**k for k in range(rnd.randint(0, n) + 1)), Integer(0))
        G = numerator/denominator
        A, B, C, D = realization(G, s)
        assert A.shape[0] <= n
        back = transfer_function(A, B, C, s, D)
        assert sympy.cancel(sympy_expr(to_sympy(back[0, 0])) - sympy_expr(to_sympy(G))) == 0
        if A.rows:
            assert is_controllable(A, B)
    # a parameter, a constant, and an improper function
    A, B, C, D = realization(b/(s + a), s)
    assert (A, B, C, D) == (ExprMatrix.from_rows([[-a]]), ExprMatrix.eye(1), ExprMatrix.from_rows([[b]]), ExprMatrix.zeros(1, 1))
    assert realization(Integer(3), s)[3] == ExprMatrix.from_rows([[3]])
    with pytest.raises(ValueError):
        realization(s**2/(s + 1), s)


def _hidden_modes(rnd: random.Random, n: int) -> tuple[ExprMatrix, ExprMatrix, ExprMatrix]:
    """A system with uncontrollable or unobservable modes hidden by a
    change of state."""
    diagonal = [Integer(rnd.randint(-3, 3)) for _ in range(n)]
    A0 = ExprMatrix(n, n, [diagonal[i] if i == j else Integer(0) for i in range(n) for j in range(n)])
    B0 = ExprMatrix.column([Integer(rnd.choice([0, 1, 1])) for _ in range(n)])
    C0 = ExprMatrix(1, n, [Integer(rnd.choice([0, 1, 1])) for _ in range(n)])
    while True:
        P = ExprMatrix(n, n, [Integer(rnd.randint(-1, 2)) for _ in range(n*n)])
        if _sympy_matrix(P).det() != 0:
            break
    Pi = inverse(P)
    return (Pi*A0*P).applyfunc(normal), (Pi*B0).applyfunc(normal), (C0*P).applyfunc(normal)


def test_kalman_decompositions() -> None:
    rnd = random.Random(115)
    for _ in range(15):
        n = rnd.randint(1, 4)
        A, B, C = _hidden_modes(rnd, n)
        T, A1, B1, r = controllable_decomposition(A, B)
        ST = _sympy_matrix(T)
        assert ST.det() != 0 and _sympy_matrix(A1) == ST.inv()*_sympy_matrix(A)*ST and _sympy_matrix(B1) == ST.inv()*_sympy_matrix(B)
        assert r == _sympy_matrix(controllability_matrix(A, B)).rank()
        assert all(A1[i, j] == 0 for i in range(r, n) for j in range(r)) and all(B1[i, 0] == 0 for i in range(r, n))
        Ac = ExprMatrix(r, r, [A1[i, j] for i in range(r) for j in range(r)])
        Bc = ExprMatrix(r, 1, [B1[i, 0] for i in range(r)])
        assert r == 0 or is_controllable(Ac, Bc)
        T, A1, C1, q = observable_decomposition(A, C)
        ST = _sympy_matrix(T)
        assert ST.det() != 0 and _sympy_matrix(A1) == ST.inv()*_sympy_matrix(A)*ST and _sympy_matrix(C1) == _sympy_matrix(C)*ST
        assert all(A1[i, j] == 0 for i in range(q) for j in range(q, n)) and all(C1[0, j] == 0 for j in range(q, n))
        Ao = ExprMatrix(q, q, [A1[i, j] for i in range(q) for j in range(q)])
        Co = ExprMatrix(1, q, [C1[0, j] for j in range(q)])
        assert q == 0 or is_observable(Ao, Co)


def test_minimal_realization() -> None:
    rnd = random.Random(116)
    reduced = 0
    for _ in range(15):
        n = rnd.randint(1, 4)
        A, B, C = _hidden_modes(rnd, n)
        Am, Bm, Cm = minimal_realization(A, B, C)
        G = sympy.cancel(sympy_expr(to_sympy(transfer_function(A, B, C, s)[0, 0])))
        Gm = sympy_expr(to_sympy(transfer_function(Am, Bm, Cm, s)[0, 0])) if Am.rows else sympy.Integer(0)
        assert sympy.cancel(G - Gm) == 0
        # the order is the degree of the denominator in lowest terms (the McMillan degree)
        degree = 0 if G == 0 else sympy.degree(sympy.fraction(G)[1], ss)
        assert Am.rows == degree
        assert Am.rows == 0 or (is_controllable(Am, Bm) and is_observable(Am, Cm))
        reduced += Am.rows < n
    assert reduced > 5
    with pytest.raises(ValueError):
        minimal_realization(ExprMatrix.eye(2), ExprMatrix.column([1, 0]), ExprMatrix.from_rows([[1, 0, 0]]))
