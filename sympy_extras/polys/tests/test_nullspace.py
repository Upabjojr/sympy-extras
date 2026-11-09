"""Domain.nullspace against SymPy's DomainMatrix.nullspace: the same
vectors over QQ; over a field of rational functions, a basis of the same
null space (SymPy's vectors are fraction-free multiples)."""
from __future__ import annotations

import random
from fractions import Fraction

import sympy
from sympy.polys.matrices import DomainMatrix

from sympy_extras.interop import expr_from_sympy
from sympy_extras.polys.domains import QQ
from sympy_extras.polys.domains.sparsefractionfield import SparseFractionField
from sympy_extras.polys.domains.sparsepolynomialring import RationalPolynomialRing


def test_nullspace_over_QQ_matches_sympy() -> None:
    rng = random.Random(3)
    for _ in range(60):
        rows, cols = rng.randint(1, 4), rng.randint(1, 6)
        M = [[Fraction(rng.choice([0, 0, 1, -1, 2, 3, -5]), rng.choice([1, 2])) for _ in range(cols)]
             for _ in range(rows)]
        theirs = DomainMatrix([[sympy.QQ(c.numerator, c.denominator) for c in r] for r in M], (rows, cols),
                              sympy.QQ).nullspace().to_list()
        ours = QQ.nullspace(M, cols)
        assert [[Fraction(int(c.numerator), int(c.denominator)) for c in v] for v in theirs] == ours, M


def test_nullspace_over_a_field_of_rational_functions() -> None:
    a, b = sympy.symbols('a b')
    F = SparseFractionField(RationalPolynomialRing(['a', 'b']))
    for M in [[[1, a, 2], [a, 1, 0]], [[a, b, a + b, 1], [1, 1, 2, 1/a], [a - b, b - a, 0, 1 - b/a]],
              [[a**2, a*b, b**2]], [[1/(a + 1), 1, 0], [0, 1, 1/(b - 1)], [1, 0, 1]]]:
        rows = [[F.from_expr(expr_from_sympy(sympy.sympify(c))) for c in r] for r in M]
        ncols = len(M[0])
        basis = F.nullspace(rows, ncols)
        rank = sympy.Matrix(M).rank()
        assert len(basis) == ncols - rank, M
        for v in basis:
            for r in rows:
                total = F.zero
                for c, x in zip(r, v):
                    total = F.add(total, F.mul(c, x))
                assert F.is_zero(total), (M, v)
