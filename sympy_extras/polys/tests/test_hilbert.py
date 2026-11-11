from __future__ import annotations

import random
from itertools import product

from sympy import Poly, series
from sympy.abc import t

from sympy_extras._typing import Monomial
from sympy_extras.polys.hilbert_sympy import hilbert_numerator, hilbert_series, minimal_generators


def _divides(u: Monomial, m: Monomial) -> bool:
    return all(a <= b for a, b in zip(u, m))


def _textbook_numerator(gens: list[Monomial]) -> Poly:
    """The recursion N(M' + (m)) = N(M') - t**deg(m) N(M' : m) applied to
    one generator at a time (Cox, Little, O'Shea, Chapter 9, Section 2)."""
    if not gens:
        return Poly(1, t)
    m, rest = gens[-1], gens[:-1]
    quotient = minimal_generators([tuple(max(a - b, 0) for a, b in zip(u, m)) for u in rest])
    if any(sum(q) == 0 for q in quotient):
        return _textbook_numerator(rest)
    return _textbook_numerator(rest) - Poly(t**sum(m), t)*_textbook_numerator(quotient)


def _hilbert_function(gens: list[Monomial], n: int, degree: int) -> int:
    """The number of monomials of the degree outside the ideal."""
    return sum(1 for e in product(range(degree + 1), repeat=n)
               if sum(e) == degree and not any(_divides(m, e) for m in gens))


def _random_generators(rng: random.Random, n: int, count: int, degree: int) -> list[Monomial]:
    gens: set[Monomial] = set()
    count = min(count, (degree + 1)**n - 1)
    while len(gens) < count:
        m = tuple(rng.randint(0, degree) for _ in range(n))
        if sum(m):
            gens.add(m)
    return sorted(gens)


def test_minimal_generators() -> None:
    assert minimal_generators([(2, 1), (1, 0), (0, 3), (1, 3)]) == [(1, 0), (0, 3)]
    assert minimal_generators([(0, 0), (1, 2)]) == [(0, 0)]
    assert minimal_generators([]) == []


def test_small_cases() -> None:
    assert hilbert_numerator([(2, 0), (0, 2)], t) == Poly(t**4 - 2*t**2 + 1, t)
    assert hilbert_numerator([], t) == Poly(1, t)
    assert hilbert_numerator([(0, 0)], t) == Poly(0, t)
    assert hilbert_numerator([(1, 0), (2, 0)], t) == Poly(1 - t, t)
    # x*y, y*z: not a complete intersection, the pivot y
    assert hilbert_numerator([(1, 1, 0), (0, 1, 1)], t) == Poly(1 - 2*t**2 + t**3, t)
    assert hilbert_series([(1, 1, 0), (0, 1, 1)], 3, t) == (1 + t - t**2)/(1 - t)**2
    assert hilbert_series([(1, 0), (0, 1)], 2, t) == 1


def test_against_textbook_recursion() -> None:
    rng = random.Random(7)
    for _ in range(150):
        n = rng.randint(1, 5)
        gens = minimal_generators(_random_generators(rng, n, rng.randint(1, 8), rng.randint(1, 4)))
        assert hilbert_numerator(gens, t) == _textbook_numerator(gens), gens


def test_against_counting() -> None:
    rng = random.Random(11)
    for _ in range(40):
        n = rng.randint(1, 4)
        gens = minimal_generators(_random_generators(rng, n, rng.randint(1, 6), rng.randint(1, 3)))
        H = series(hilbert_numerator(gens, t).as_expr()/(1 - t)**n, t, 0, 7).removeO()
        assert all(H.coeff(t, k) == _hilbert_function(gens, n, k) for k in range(7)), gens


def test_many_generators() -> None:
    # the single-degree ideals make every generator minimal; the textbook
    # recursion takes seconds on these
    rng = random.Random(3)
    monomials = [e for e in product(range(5), repeat=7) if sum(e) == 4]
    gens = rng.sample(monomials, 60)
    N = hilbert_numerator(gens, t)
    H = series(N.as_expr()/(1 - t)**7, t, 0, 6).removeO()
    assert H.coeff(t, 3) == _hilbert_function(gens, 7, 3) == 84
    assert H.coeff(t, 4) == _hilbert_function(gens, 7, 4) == 210 - 60
    assert H.coeff(t, 5) == _hilbert_function(gens, 7, 5)
