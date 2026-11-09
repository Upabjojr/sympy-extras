"""The heuristic gcd over ZZ against the canonical subresultant gcd and
SymPy (the oracle)."""
from __future__ import annotations

import random

import sympy

from sympy_extras.interop import expr_from_sympy
from sympy_extras.interop import to_sympy
from sympy_extras.polys.domains.sparsepolynomialring import IntegerPolynomialRing, SparsePolynomialRing
from sympy_extras.polys.domains import ZZ

NAMES = ['x', 'y', 'z', 'w']
SYMBOLS = sympy.symbols(NAMES)


def random_polynomial(rng: random.Random, n: int) -> sympy.Expr:
    return sympy.Add(*[rng.randint(-9, 9) * sympy.Mul(*[v**rng.randint(0, 3) for v in SYMBOLS[:n]])
                       for _ in range(rng.randint(1, 4))])


def test_heugcd_matches_subresultants_and_sympy() -> None:
    rng = random.Random(7)
    for _ in range(80):
        n = rng.randint(1, 4)
        H = IntegerPolynomialRing(NAMES[:n])
        S = SparsePolynomialRing(ZZ, NAMES[:n])
        c = random_polynomial(rng, n)
        f = sympy.expand(c * random_polynomial(rng, n) * rng.randint(1, 4))
        g = sympy.expand(c * random_polynomial(rng, n) * rng.randint(1, 4))
        if f == 0 or g == 0:
            continue
        a, b = H.from_expr(expr_from_sympy(f)), H.from_expr(expr_from_sympy(g))
        heuristic = H.gcd(a, b)
        assert heuristic == S.gcd(a, b)
        expected = sympy.gcd(sympy.Poly(f, *SYMBOLS[:n]), sympy.Poly(g, *SYMBOLS[:n])).as_expr()
        assert sympy.expand(to_sympy(H.to_expr(heuristic)) - expected) == 0
