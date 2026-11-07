"""Real root isolation and counting against SymPy's ``intervals``,
``count_roots`` and ``real_roots`` (the oracle) and the definitions."""
from __future__ import annotations

import random
from fractions import Fraction

from sympy import Poly, Rational, chebyshevt, symbols

from sympy_extras.polys.densearith import dup_mul, dup_pow
from sympy_extras.polys.densebasic import dup_strip
from sympy_extras.polys.domains import QQ
from sympy_extras.polys.rootisolation import (dup_cauchy_bound, dup_count_real_roots, dup_isolate_real_roots,
                                               dup_isolate_real_roots_sqf, dup_refine_real_root, dup_sign_at,
                                               dup_sturm)

x = symbols('x')


def _random(rng: random.Random, degree: int) -> list[Fraction]:
    f = dup_strip([Fraction(rng.randint(-6, 6)) for _ in range(degree + 1)], QQ)
    return f if f else [Fraction(1)]


def _sympy(f: list[Fraction]) -> Poly:
    return Poly([Rational(c.numerator, c.denominator) for c in f], x)


def _contains_one_root(f: list[Fraction], a: Fraction, b: Fraction) -> bool:
    if a == b:
        return dup_sign_at(f, a) == 0
    return dup_sign_at(f, a)*dup_sign_at(f, b) < 0 and dup_count_real_roots(f, a, b) == 1


def test_against_sympy_intervals() -> None:
    rng = random.Random(1)
    for _ in range(80):
        f = _random(rng, rng.randint(1, 8))
        if rng.random() < 0.4:
            # a rational root, sometimes at a dyadic point
            f = dup_mul(f, [Fraction(2), Fraction(-1)], QQ)
        P = _sympy(f)
        intervals = dup_isolate_real_roots(f)
        expected = P.intervals(sqf=False)
        assert len(intervals) == len(expected), f
        for ((a, b), k), ((ea, eb), ek) in zip(intervals, expected):
            assert k == ek
            # the SymPy interval and ours contain the same root
            width = max(b - a, Fraction(int(eb.p), int(eb.q)) - Fraction(int(ea.p), int(ea.q)))
            assert a <= Fraction(int(eb.p), int(eb.q)) + width and Fraction(int(ea.p), int(ea.q)) <= b + width
        assert dup_count_real_roots(f) == P.count_roots()
        # the roots are pairwise distinct: the intervals do not overlap
        assert all(b1 <= a2 for ((a1, b1), _), ((a2, b2), _) in zip(intervals, intervals[1:]))
        # no endpoint of an open interval is a root
        assert all(a == b or (dup_sign_at(f, a) != 0 and dup_sign_at(f, b) != 0) for (a, b), _ in intervals)


def test_intervals_isolate() -> None:
    rng = random.Random(2)
    for _ in range(60):
        f = _random(rng, rng.randint(1, 7))
        from sympy_extras.polys.sqfreetools import dup_sqf_part
        g = dup_sqf_part(f, QQ)
        intervals = dup_isolate_real_roots_sqf(g)
        assert all(_contains_one_root(g, a, b) for a, b in intervals), g
        assert len(intervals) == dup_count_real_roots(g)
        assert intervals == sorted(intervals)
        refined = dup_isolate_real_roots_sqf(g, Fraction(1, 1000))
        assert all(b - a < Fraction(1, 1000) for a, b in refined)
        assert all(_contains_one_root(g, a, b) for a, b in refined)


def test_known_polynomials() -> None:
    # x**2 - 2
    f = [Fraction(1), Fraction(0), Fraction(-2)]
    [(a, b), (c, d)] = dup_isolate_real_roots_sqf(f, Fraction(1, 10**6))
    assert abs(float((a + b)/2) + 2**0.5) < 1e-6 and abs(float((c + d)/2) - 2**0.5) < 1e-6
    # the Chebyshev polynomial of degree 12 has 12 roots in (-1, 1)
    T = Poly(chebyshevt(12, x), x)
    f = [Fraction(int(c)) for c in T.all_coeffs()]
    intervals = dup_isolate_real_roots_sqf(f, Fraction(1, 100))
    assert len(intervals) == 12 and all(-1 < a and b < 1 for a, b in intervals)
    assert all(_contains_one_root(f, a, b) for a, b in intervals)
    # Wilkinson's polynomial of degree 10: the roots 1..10 exactly
    f = [Fraction(1)]
    for k in range(1, 11):
        f = dup_mul(f, [Fraction(1), Fraction(-k)], QQ)
    intervals = dup_isolate_real_roots_sqf(f, Fraction(1, 4))
    assert len(intervals) == 10 and all(a < k < b or a == b == k for (a, b), k in zip(intervals, range(1, 11)))
    assert dup_count_real_roots(f, Fraction(3), Fraction(7)) == 4
    # multiplicities
    f = dup_mul(dup_pow([Fraction(1), Fraction(-1)], 3, QQ), [Fraction(1), Fraction(0), Fraction(-3)], QQ)
    assert [k for _, k in dup_isolate_real_roots(f)] == [1, 3, 1]
    # no real root, a constant, zero
    assert dup_isolate_real_roots([Fraction(1), Fraction(0), Fraction(1)]) == []
    assert dup_isolate_real_roots([Fraction(5)]) == [] and dup_count_real_roots([]) == 0
    assert dup_cauchy_bound([Fraction(2), Fraction(-8), Fraction(1)]) == 5
    # the root at zero
    assert dup_isolate_real_roots_sqf([Fraction(1), Fraction(-1), Fraction(0)]) == [(Fraction(0), Fraction(0)), (Fraction(1), Fraction(1))]


def test_sturm_and_refinement() -> None:
    f = [Fraction(1), Fraction(0), Fraction(-2)]
    sequence = dup_sturm(f)
    assert len(sequence) == 3 and sequence[-1] == [Fraction(1)]
    a, b = dup_refine_real_root(f, Fraction(1), Fraction(2), Fraction(1, 10**9))
    assert b - a < Fraction(1, 10**9) and a*a < 2 < b*b
    assert dup_refine_real_root(f, Fraction(1), Fraction(1), Fraction(1, 2)) == (Fraction(1), Fraction(1))
