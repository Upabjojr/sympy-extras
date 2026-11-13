"""Tests of the native Hilbert series (:mod:`sympy_extras.polys.hilbert`):
the numerator against the textbook recursion and the series against a
count of the standard monomials; the SymPy face is tested in
``test_hilbert.py`` and ``test_hilbert_known.py``."""
from __future__ import annotations

import random
from itertools import product

from sympy_extras._typing import Monomial
from sympy_extras.core import Integer, Symbol
from sympy_extras.polys.domains import ZZ
from sympy_extras.polys.hilbert import hilbert_numerator, hilbert_series, minimal_generators, reduced_hilbert_series
from sympy_extras.polys.polyclasses import DensePoly

t = Symbol('t')


def _poly(coefficients: list[int]) -> DensePoly[int]:
    """The polynomial in ``t`` with the coefficients, lowest degree first."""
    return DensePoly(list(reversed(coefficients)), ZZ, t)


def _textbook_numerator(gens: list[Monomial]) -> DensePoly[int]:
    """N(M' + (m)) = N(M') - t**deg(m) N(M' : m), one generator at a time
    (Cox, Little, O'Shea, Chapter 9, Section 2)."""
    if not gens:
        return _poly([1])
    m, rest = gens[-1], gens[:-1]
    quotient = minimal_generators([tuple(max(a - b, 0) for a, b in zip(u, m)) for u in rest])
    if any(sum(q) == 0 for q in quotient):
        return _textbook_numerator(rest)
    return _textbook_numerator(rest) - _poly([0]*sum(m) + [1])*_textbook_numerator(quotient)


def _hilbert_function(gens: list[Monomial], n: int, degree: int) -> int:
    return sum(1 for e in product(range(degree + 1), repeat=n)
               if sum(e) == degree and not any(all(a <= b for a, b in zip(m, e)) for m in gens))


def _random_generators(rng: random.Random, n: int, count: int, degree: int) -> list[Monomial]:
    gens: set[Monomial] = set()
    count = min(count, (degree + 1)**n - 1)
    while len(gens) < count:
        m = tuple(rng.randint(0, degree) for _ in range(n))
        if sum(m):
            gens.add(m)
    return sorted(gens)


def test_small_cases() -> None:
    assert hilbert_numerator([(2, 0), (0, 2)], t) == _poly([1, 0, -2, 0, 1])
    assert hilbert_numerator([], t) == _poly([1])
    assert hilbert_numerator([(0, 0)], t).is_zero
    assert hilbert_numerator([(1, 0), (2, 0)], t) == _poly([1, -1])
    assert str(hilbert_series([(1, 1, 0), (0, 1, 1)], 3, t)) == '(-t**2 + t + 1)/(1 - t)**2'
    assert hilbert_series([(1, 0), (0, 1)], 2, t) == Integer(1)
    assert hilbert_series([(0, 0)], 2, t) == Integer(0)
    assert str(hilbert_series([], 2, t)) == '(1 - t)**(-2)'
    # the twisted cubic's leading term ideal (x**2, x*y, y**2): (1 + 2t)/(1 - t)
    assert reduced_hilbert_series([(2, 0, 0), (1, 1, 0), (0, 2, 0)], 3) == ([1, 2], 1)
    assert reduced_hilbert_series([(0, 0)], 2) == ([], 2)
    assert reduced_hilbert_series([], 2) == ([1], 2)


def test_against_textbook_recursion() -> None:
    rng = random.Random(7)
    for _ in range(100):
        n = rng.randint(1, 5)
        gens = minimal_generators(_random_generators(rng, n, rng.randint(1, 8), rng.randint(1, 4)))
        assert hilbert_numerator(gens, t) == _textbook_numerator(gens), gens


def test_against_counting() -> None:
    # sum_k HF(k) t**k = q(t)/(1 - t)**d: the coefficients of q/(1 - t)**d
    # are those of q convolved d times with the partial sums
    rng = random.Random(11)
    for _ in range(40):
        n = rng.randint(1, 4)
        gens = minimal_generators(_random_generators(rng, n, rng.randint(1, 6), rng.randint(1, 3)))
        q, d = reduced_hilbert_series(gens, n)
        series = (q + [0]*8)[:8]
        for _ in range(d):
            running, partial = 0, []
            for c in series:
                running += c
                partial.append(running)
            series = partial
        assert series[:7] == [_hilbert_function(gens, n, k) for k in range(7)], gens
