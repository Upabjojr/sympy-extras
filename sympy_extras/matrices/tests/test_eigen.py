"""Eigenvalues and eigenvectors: the defining identity M v = lambda v
modulo the factor, SymPy's eigenvalues and multiplicities, and
diagonalisations multiplied back."""
from __future__ import annotations

import random
from fractions import Fraction

import pytest
import sympy

from sympy_extras.core import Integer, Rational, Symbol
from sympy_extras.core.expr import Expr
from sympy_extras.interop import sympy_expr, to_sympy
from sympy_extras.matrices import ExprMatrix
from sympy_extras.matrices.eigen import (diagonalize, eigenspaces, eigenvalues, eigenvectors, is_diagonalizable, jordan_chains,
                                         jordan_form)
from sympy_extras.polys.domains import GF, QQ
from sympy_extras.polys.domains.quotientfield import QuotientField

a, b, x = Symbol('a'), Symbol('b'), Symbol('x')


def _sympy_matrix(M: ExprMatrix) -> sympy.Matrix:
    return sympy.Matrix(M.rows, M.cols, [to_sympy(e) for e in M.entries])


def _check_spaces(M: ExprMatrix) -> None:
    """Every basis vector satisfies M v = x v modulo the factor, the
    multiplicities add up to the size, and the dimensions are those of
    the kernels SymPy finds."""
    S = _sympy_matrix(M)
    sx = sympy.Symbol('x')
    spaces = eigenspaces(M, x)
    total = 0
    product = sympy.Integer(1)
    for space in spaces:
        factor = sympy_expr(to_sympy(space.factor))
        degree = sympy.Poly(factor, sx).degree()
        total += degree*space.multiplicity
        product *= factor**space.multiplicity
        assert 1 <= space.geometric_multiplicity <= space.multiplicity
        for vector in space.basis:
            v = _sympy_matrix(vector)
            residual = S*v - sx*v
            for entry in residual:
                numerator = sympy.fraction(sympy.together(entry))[0]
                assert sympy.rem(sympy.expand(numerator), factor, sx) == 0
    assert total == M.rows
    characteristic = S.charpoly(sx).as_expr()
    ratio = sympy.cancel(product/characteristic)
    assert sx not in ratio.free_symbols and ratio != 0


def _diagonalizable(S: sympy.Matrix) -> bool:
    """The squarefree part of the characteristic polynomial annihilates
    the matrix (SymPy's ``is_diagonalizable`` does not return on
    irreducible quartics)."""
    t = sympy.Symbol('t')
    characteristic = S.charpoly(t).as_poly()
    squarefree = sympy.quo(characteristic, sympy.gcd(characteristic, characteristic.diff(t)))
    total = sympy.zeros(S.rows, S.rows)
    for c in squarefree.all_coeffs():
        total = total*S + c*sympy.eye(S.rows)
    return bool(total == sympy.zeros(S.rows, S.rows))


def test_random_integer_matrices() -> None:
    rnd = random.Random(11)
    for trial in range(40):
        n = rnd.randint(1, 4)
        entries = [rnd.randint(-2, 2) for _ in range(n*n)]
        if trial % 4 == 0 and n > 1:
            # a repeated row: the eigenvalue zero, often defective
            entries[n:2*n] = entries[:n]
        M = ExprMatrix(n, n, entries)
        _check_spaces(M)
        assert is_diagonalizable(M) == _diagonalizable(_sympy_matrix(M))


def test_structured_matrices() -> None:
    # repeated eigenvalues with eigenspaces of several dimensions
    for rows in ([[2, 0, 0], [0, 2, 0], [0, 0, 3]], [[2, 1, 0], [0, 2, 0], [0, 0, 2]], [[2, 1, 0], [0, 2, 1], [0, 0, 2]],
                 [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]], [[1, 2], [3, 4]]):
        M = ExprMatrix.from_rows(rows)
        _check_spaces(M)
        S = sympy.Matrix(rows)
        assert is_diagonalizable(M) == S.is_diagonalizable()
        expected = {sympy.simplify(k): v for k, v in S.eigenvals().items()}
        found = {sympy.simplify(to_sympy(k)): v for k, v in eigenvalues(M).items()}
        assert found == expected
        dimensions = sorted(len(vectors) for _, _, vectors in eigenvectors(M))
        assert dimensions == sorted(len(vectors) for _, _, vectors in S.eigenvects())


def test_parameters() -> None:
    for M in (ExprMatrix.from_rows([[a, b], [b, a]]), ExprMatrix.from_rows([[a, 1, 0], [0, a, 0], [0, 0, b]]),
              ExprMatrix.from_rows([[0, a], [b, 0]]), ExprMatrix.from_rows([[1/a, 1], [b, a/(b + 1)]]),
              ExprMatrix.from_rows([[a, 1, 0], [1, a, 1], [0, 1, a]])):
        _check_spaces(M)
    assert eigenvalues(ExprMatrix.from_rows([[a, b], [b, a]])) == {a - b: 1, a + b: 1}
    assert not is_diagonalizable(ExprMatrix.from_rows([[a, 1], [0, a]]))
    assert is_diagonalizable(ExprMatrix.from_rows([[a, 1], [0, b]]))
    with pytest.raises(NotImplementedError):
        from sympy_extras.functions import sin
        eigenspaces(ExprMatrix.from_rows([[sin(a), 1], [1, 0]]), x)


def test_diagonalize() -> None:
    rnd = random.Random(12)
    done = 0
    while done < 8:
        n = rnd.randint(2, 3)
        M = ExprMatrix(n, n, [rnd.randint(-2, 2) for _ in range(n*n)])
        S = _sympy_matrix(M)
        try:
            result = diagonalize(M)
        except NotImplementedError:
            continue
        assert (result is not None) == S.is_diagonalizable()
        if result is None:
            continue
        P, D = (_sympy_matrix(m) for m in result)
        assert sympy.simplify(P.det()) != 0
        difference = S*P - P*D
        assert all(sympy.simplify(sympy.expand(e)) == 0 for e in difference)
        done += 1
    P, D = (_sympy_matrix(m) for m in diagonalize(ExprMatrix.from_rows([[a, b], [b, a]])) or ())
    assert sympy.simplify(_sympy_matrix(ExprMatrix.from_rows([[a, b], [b, a]]))*P - P*D) == sympy.zeros(2, 2)


def test_unsolvable_factor() -> None:
    companion = ExprMatrix.from_rows([[0, 0, 0, 0, 1], [1, 0, 0, 0, 1], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0]])
    [space] = eigenspaces(companion, x)
    assert space.factor == x**5 - x - 1 and space.values() is None and space.geometric_multiplicity == 1
    assert is_diagonalizable(companion)
    with pytest.raises(NotImplementedError):
        eigenvalues(companion)
    _check_spaces(companion)
    zero: Expr = Integer(0)
    assert eigenspaces(ExprMatrix(0, 0, []), x) == [] and eigenvalues(ExprMatrix.from_rows([[Rational(1, 2)]])) == {Rational(1, 2): 1}
    assert eigenvalues(ExprMatrix.from_rows([[zero]])) == {zero: 1}


def test_quotient_field() -> None:
    F = QuotientField(QQ, [Fraction(1), Fraction(0), Fraction(1), Fraction(-3)])
    rnd = random.Random(13)
    for _ in range(50):
        u = F.reduce([Fraction(rnd.randint(-5, 5)) for _ in range(3)])
        v = F.reduce([Fraction(rnd.randint(-5, 5)) for _ in range(3)])
        w = F.reduce([Fraction(rnd.randint(-5, 5)) for _ in range(3)])
        assert F.mul(u, F.add(v, w)) == F.add(F.mul(u, v), F.mul(u, w))
        assert F.mul(F.mul(u, v), w) == F.mul(u, F.mul(v, w))
        if u:
            assert F.mul(u, F.invert(u)) == F.one and F.exquo(F.mul(u, v), u) == v
    # GF(9) = GF(3)[t]/(t**2 + 1): the multiplicative group has order 8
    G = QuotientField(GF(3), [1, 0, 1])
    element = G.add(G.generator, G.one)
    assert G.pow(element, 8) == G.one and G.pow(element, 4) != G.one
    # a reducible modulus has zero divisors
    R = QuotientField(QQ, [Fraction(1), Fraction(0), Fraction(-1)])
    with pytest.raises(ValueError):
        R.invert(R.sub(R.generator, R.one))
    with pytest.raises(ValueError):
        QuotientField(QQ, [Fraction(2), Fraction(1)])


def _similar_to_blocks(rnd: random.Random, blocks: list[tuple[int, int]]) -> sympy.Matrix:
    """A matrix with the given Jordan blocks ``(eigenvalue, size)``,
    conjugated by a random unimodular integer matrix."""
    n = sum(size for _, size in blocks)
    J = sympy.zeros(n, n)
    position = 0
    for value, size in blocks:
        for i in range(size):
            J[position + i, position + i] = value
            if i:
                J[position + i - 1, position + i] = 1
        position += size
    P = sympy.eye(n)
    for _ in range(2*n):
        i, j = rnd.randrange(n), rnd.randrange(n)
        if i != j:
            P[i, :] = P[i, :] + rnd.choice([-1, 1])*P[j, :]
    return P*J*P.inv()


def test_jordan_form_of_planted_blocks() -> None:
    rnd = random.Random(21)
    cases = [[(2, 2), (2, 1)], [(1, 3)], [(0, 2), (0, 2)], [(3, 1), (3, 1), (-1, 2)], [(2, 3), (2, 1), (5, 1)],
             [(1, 2), (1, 2), (1, 1)], [(0, 4)], [(7, 1)], [(-2, 2), (4, 3)]]
    for blocks in cases:
        S = _similar_to_blocks(rnd, blocks)
        M = ExprMatrix(S.rows, S.cols, [int(S[i, j]) for i in range(S.rows) for j in range(S.cols)])
        found: list[tuple[int, int]] = []
        for group in jordan_chains(M, x):
            [value] = sympy.solve(sympy_expr(to_sympy(group.factor)), sympy.Symbol('x'))
            found.extend((int(value), size) for size in group.block_sizes)
            assert sum(group.block_sizes) == group.multiplicity
        assert sorted(found) == sorted(blocks)
        P, J = (_sympy_matrix(m) for m in jordan_form(M))
        assert S*P == P*J and P.det() != 0
        assert all(J[i, j] == 0 for i in range(S.rows) for j in range(S.rows) if j not in (i, i + 1))
        assert all(J[i, i + 1] in (0, 1) for i in range(S.rows - 1))


def test_jordan_form_irrational_and_parameters() -> None:
    # a 4 x 4 matrix with minimal polynomial (x**2 - 2)**2: two blocks of size two for sqrt(2) and -sqrt(2)
    C = sympy.Matrix([[0, 2], [1, 0]])
    S = sympy.Matrix(sympy.BlockMatrix([[C, sympy.eye(2)], [sympy.zeros(2, 2), C]]))
    M = ExprMatrix(4, 4, [int(S[i, j]) for i in range(4) for j in range(4)])
    [group] = jordan_chains(M, x)
    assert group.factor == x**2 - 2 and group.multiplicity == 2 and group.block_sizes == [2]
    P, J = (_sympy_matrix(m) for m in jordan_form(M))
    assert sympy.simplify(S*P - P*J) == sympy.zeros(4, 4) and sympy.simplify(P.det()) != 0
    assert sorted(J[i, i + 1] for i in range(3)) == [0, 1, 1]
    # parameters
    A = ExprMatrix.from_rows([[a, 1, 0], [0, a, 0], [0, 0, b]])
    groups = jordan_chains(A, x)
    assert sorted(g.block_sizes for g in groups) == [[1], [2]]
    P, J = (_sympy_matrix(m) for m in jordan_form(A))
    assert sympy.simplify(_sympy_matrix(A)*P - P*J) == sympy.zeros(3, 3) and sympy.simplify(P.det()) != 0
    # a diagonalisable matrix: the Jordan form is diagonal
    D = ExprMatrix.from_rows([[a, b], [b, a]])
    assert all(size == [1] for size in (g.block_sizes for g in jordan_chains(D, x)))
    # the structure without the roots
    companion = ExprMatrix.from_rows([[0, 0, 0, 0, 1], [1, 0, 0, 0, 1], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0]])
    assert [g.block_sizes for g in jordan_chains(companion, x)] == [[1]]
    with pytest.raises(NotImplementedError):
        jordan_form(companion)
