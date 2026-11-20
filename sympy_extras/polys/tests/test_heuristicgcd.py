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


def test_balanced_digits_and_evaluation_by_halves() -> None:
    # issue #25: the digits and the values are computed by splitting at
    # x**(2**k) (the loops were quadratic in the size of the numbers);
    # the same digits as the loop of SymPy's _gcd_interpolate
    from sympy_extras.polys.heuristicgcd import _balanced, _balanced_digits, _evaluate_dense, _square_powers

    def loop(c: int, x: int) -> list[int]:
        digits: list[int] = []
        while c:
            g = _balanced(c, x)
            c = (c - g) // x
            digits.append(g)
        return digits

    rng = random.Random(3)
    for _ in range(2000):
        x = rng.randint(3, 10**rng.randint(1, 8))
        c = rng.randint(-x**rng.randint(0, 40), x**rng.randint(0, 40))
        assert _balanced_digits(c, x) == loop(c, x)
        dense = [rng.randint(-10**6, 10**6) for _ in range(rng.randint(1, 70))]
        value = 0
        for a in reversed(dense):
            value = value * x + a
        assert _evaluate_dense(dense, _square_powers(x, len(dense))) == value
    for x in (3, 4, 29):
        for c in range(-200, 200):
            assert _balanced_digits(c, x) == loop(c, x)


def test_square_free_part_of_a_power_with_large_coefficients() -> None:
    from sympy_extras.polys.multifactor import sqf_part
    from sympy_extras.polys.rings import PolyElement, PolyRing
    R = PolyRing(ZZ, ['b', 'c'])
    b = PolyElement({(1, 0): 1}, R)
    q = PolyElement({(4, 0): 27, (0, 3): 4096}, R)
    assert str(sqf_part(b**8 * q**7)) == '27*b**5 + 4096*b*c**3'
