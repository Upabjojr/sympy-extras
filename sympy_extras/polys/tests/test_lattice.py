"""The integral LLL algorithm and the Hermite normal form against their
definitions, known lattices and SymPy's LLL (the oracle, on small
entries)."""
from __future__ import annotations

import random
from fractions import Fraction

from sympy.matrices.dense import Matrix
from sympy.polys.domains import ZZ as SZZ
from sympy.polys.matrices import DomainMatrix
from sympy.testing.pytest import raises

from sympy_extras.polys.lattice import hermite_normal_form, is_lll_reduced, lll, lll_reduction


def _random_rows(rng: random.Random, n: int, m: int, bound: int) -> list[list[int]]:
    return [[rng.randint(-bound, bound) for _ in range(m)] for _ in range(n)]


def test_known_reductions() -> None:
    # Cohen, A Course in Computational Algebraic Number Theory, the
    # example every text reduces (rows (1,1,1), (-1,0,2), (3,5,6))
    assert lll([[1, 1, 1], [-1, 0, 2], [3, 5, 6]]) == [[0, 1, 0], [1, 0, 1], [-1, 0, 2]]
    # the lattice of x = 3 y (mod 7) scaled: the shortest vector has norm 3
    basis = lll([[1, 0, 3], [0, 1, 5], [0, 0, 7]])
    assert sum(v*v for v in basis[0]) == 3
    assert lll([]) == []
    assert lll([[0, 0], [0, 0]]) == []
    assert lll([[6]]) == [[6]]
    assert lll([[6], [4]]) == [[2]]


def test_reduced_bases_of_random_lattices() -> None:
    rng = random.Random(11)
    for _ in range(150):
        n, m = rng.randint(1, 6), rng.randint(1, 6)
        rows = _random_rows(rng, n, m, rng.choice([3, 60, 10**15]))
        if n > 1 and rng.random() < 0.3:
            rows.append([a + 2*b for a, b in zip(rows[0], rows[-1])])
        reduction = lll_reduction(rows)
        basis = reduction.basis
        # reduced, a basis of the same lattice, of the right rank
        assert is_lll_reduced(basis)
        assert hermite_normal_form(basis) == hermite_normal_form(rows)
        assert len(basis) == Matrix(rows).rank()
        # the Gram determinants are those of the leading Gram matrices
        for i in range(1, len(basis) + 1):
            leading = Matrix(basis[:i])
            assert (leading*leading.T).det() == reduction.gram[i]
        assert reduction.gram[0] == 1


def test_against_sympy() -> None:
    rng = random.Random(12)
    for _ in range(60):
        n = rng.randint(1, 6)
        m = rng.randint(n, 7)
        rows = _random_rows(rng, n, m, 40)
        if Matrix(rows).rank() < n:
            continue
        expected = DomainMatrix(rows, (n, m), SZZ).lll().to_Matrix().tolist()
        assert lll(rows) == [[int(v) for v in row] for row in expected]


def test_other_delta() -> None:
    rng = random.Random(13)
    for delta in (Fraction(1, 2), Fraction(99, 100), Fraction(1)):
        for _ in range(20):
            rows = _random_rows(rng, 4, 5, 100)
            basis = lll(rows, delta)
            assert is_lll_reduced(basis, delta)
            assert hermite_normal_form(basis) == hermite_normal_form(rows)
    raises(ValueError, lambda: lll([[1, 0], [0, 1]], Fraction(1, 4)))
    raises(ValueError, lambda: lll([[1, 0], [0, 1, 2]]))


def test_large_entries() -> None:
    # SymPy 1.14: DomainMatrix.lll raises AssertionError on such entries
    # (the size reduction is checked in floating point)
    rng = random.Random(14)
    n = 12
    rows = [[1 if i == j else 0 for j in range(n)] + [rng.randrange(2**300)] for i in range(n)]
    basis = lll(rows)
    assert is_lll_reduced(basis)
    assert hermite_normal_form(basis) == hermite_normal_form(rows)


def test_integer_relation() -> None:
    # 1, sqrt(2), sqrt(8) at 30 digits: the relation sqrt(8) = 2 sqrt(2)
    scale = 10**30
    sqrt2 = 1414213562373095048801688724209
    sqrt8 = 2828427124746190097603377448419
    basis = lll([[1, 0, 0, scale], [0, 1, 0, sqrt2], [0, 0, 1, sqrt8]])
    assert basis[0][:3] in ([0, -2, 1], [0, 2, -1])


def test_short_part() -> None:
    # two short vectors and a long one: the long one leaves
    rows = [[1, 0, 0, 0], [0, 1, 0, 0], [3, 5, 1000, 7]]
    reduction = lll_reduction(rows)
    short = reduction.short_part(Fraction(4))
    assert hermite_normal_form(short) == [[1, 0, 0, 0], [0, 1, 0, 0]]
    assert reduction.short_part(Fraction(10**7)) == reduction.basis
    assert reduction.short_part(Fraction(0)) == []
    norms = reduction.norms_squared()
    assert norms[0] == 1 and norms[1] == 1 and norms[2] == 1000**2 + 7**2


def test_hermite_normal_form() -> None:
    assert hermite_normal_form([[4, 6], [6, 4]]) == [[2, 8], [0, 10]]
    assert hermite_normal_form([[1, 2, 3], [2, 4, 6]]) == [[1, 2, 3]]
    assert hermite_normal_form([[0, 0]]) == []
    assert hermite_normal_form([[0, -3]]) == [[0, 3]]
    rng = random.Random(15)
    for _ in range(100):
        n, m = rng.randint(1, 5), rng.randint(1, 5)
        rows = _random_rows(rng, n, m, 30)
        form = hermite_normal_form(rows)
        assert len(form) == Matrix(rows).rank()
        # echelon, positive pivots, reduced above the pivots
        pivots: list[int] = []
        for row in form:
            column = next(j for j, v in enumerate(row) if v)
            assert row[column] > 0 and (not pivots or column > pivots[-1])
            pivots.append(column)
        for i, column in enumerate(pivots):
            assert all(0 <= form[k][column] < form[i][column] for k in range(i))
        # invariant under unimodular changes of the generators
        shuffled = [list(row) for row in rows]
        rng.shuffle(shuffled)
        if len(shuffled) > 1:
            shuffled[0] = [a + 3*b for a, b in zip(shuffled[0], shuffled[1])]
        shuffled.append([-v for v in shuffled[-1]])
        assert hermite_normal_form(shuffled) == form
        # the same lattice: each generator reduces to zero against the form
        for row in rows:
            rest = list(row)
            for basis_row, column in zip(form, pivots):
                q = rest[column]//basis_row[column]
                rest = [a - q*b for a, b in zip(rest, basis_row)]
            assert not any(rest)
