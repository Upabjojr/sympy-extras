"""Pillai's equation and exponential polynomials against known results
and brute force."""
from __future__ import annotations

import random
from fractions import Fraction

from sympy.testing.pytest import raises

from sympy_extras.solvers.exponential_diophantine import (_baker_bound, _common_base, _dependence, _partial_quotients,
                                                          exponential_polynomial_roots, pillai)

F = Fraction


def test_pillai_known() -> None:
    # Catalan's equation for the bases 2 and 3 (Levi ben Gerson, 1343)
    assert pillai(3, 2, 1) == [(1, 1), (2, 3)]
    assert pillai(2, 3, 1) == [(1, 0), (2, 1)]
    assert pillai(2, 3, -1) == [(1, 1), (3, 2)]
    # |3**x - 2**y| = c has more than one solution with x, y >= 1 for
    # c = 1, 5, 13 only (Pillai 1945, proved by Stroeker and Tijdeman 1982)
    assert pillai(2, 3, 5) == [(3, 1), (5, 3)]
    assert pillai(2, 3, 13) == [(4, 1), (8, 5)]
    assert pillai(3, 2, -5) == [(1, 3), (3, 5)]
    assert pillai(3, 2, -13) == [(1, 4), (5, 8)]
    # 2**x - 5**y = 3 has three solutions; no power of 5 is one more than a power of 7
    assert pillai(2, 5, 3) == [(2, 0), (3, 1), (7, 3)]
    assert pillai(5, 7, 1) == []
    # 13**3 - 3**7 = 10, 13 - 3 = 10 (a pair noticed by Bennett, 2001)
    assert pillai(13, 3, 10) == [(1, 1), (3, 7)]
    # 2**x - 3**y = 2**k - 1 always has (k, 0)
    assert (10, 0) in pillai(2, 3, 1023)


def test_pillai_dependent_bases() -> None:
    assert pillai(4, 2, 12) == [(2, 2)]
    assert pillai(2, 4, 12) == [(4, 1)]
    assert pillai(2, 4, 10) == []
    assert pillai(2, 4, -12) == [(2, 2)]
    assert pillai(8, 4, 48) == [(2, 2)]                # 2**6 - 2**4
    assert pillai(4, 8, -4) == [(1, 1)]
    assert pillai(2, 2, 6) == [(3, 1)]
    assert pillai(2, 2, 7) == [(3, 0)]
    assert pillai(9, 27, 0 + 54) == [(2, 1)]
    assert _common_base(8, 4) == (2, 3, 2)
    assert _common_base(12, 18) is None
    assert _common_base(36, 6) == (6, 2, 1)
    assert _common_base(4, 36) is None


def test_pillai_against_search() -> None:
    rng = random.Random(51)
    for _ in range(120):
        a, b = rng.randint(2, 12), rng.randint(2, 12)
        if rng.random() < 0.6:
            # a difference of two powers, so that there is a solution
            c = a**rng.randint(0, 6) - b**rng.randint(0, 6)
        else:
            c = rng.randint(-200, 200)
        if c == 0:
            continue
        found = pillai(a, b, c)
        expected = [(x, y) for x in range(40) for y in range(40) if a**x - b**y == c]
        assert all(a**x - b**y == c for x, y in found)
        assert [s for s in found if s[0] < 40 and s[1] < 40] == expected
        # nothing beyond the box either (it would be a remarkable identity)
        assert found == expected


def test_pillai_with_coefficients() -> None:
    # 5**3 - 2**7 = -3, 5**3 - 11**2 = 4
    assert pillai(5, 2, -3, s=2) == [(0, 1), (1, 2), (3, 6)]
    assert pillai(5, 11, 4, r=5) == [(0, 0), (2, 2)]
    # a common factor of the coefficients which does not divide c
    assert pillai(2, 3, 5, r=4, s=6) == []
    assert pillai(2, 3, 10, r=2, s=2) == pillai(2, 3, 5)
    # coefficients made of the primes of the bases: two logarithms again
    assert _dependence(2, 3, 4, 9) == (1, 2, -2)
    assert _dependence(4, 9, 6, 1) == (2, 1, 1)
    assert _dependence(2, 3, 5, 1) is None
    assert pillai(2, 3, 3, r=3) == [(1, 1), (2, 2)]             # 3 (2**x - 3**(y - 1)) = 3
    assert pillai(4, 9, 15, r=6) == [(1, 1), (2, 2)]             # 24 - 9 = 96 - 81
    # dependent bases with coefficients: 3 2**x - 5 4**y = 4, 28
    assert pillai(2, 4, 4, r=3, s=5) == [(3, 1)]
    assert pillai(2, 4, 28, r=3, s=5) == [(4, 1)]
    rng = random.Random(53)
    checked = 0
    for _ in range(300):
        a, b, r, s = rng.randint(2, 12), rng.randint(2, 12), rng.randint(1, 9), rng.randint(1, 9)
        if rng.random() < 0.7:
            c = r*a**rng.randint(0, 6) - s*b**rng.randint(0, 6)
        else:
            c = rng.randint(-300, 300)
        if c == 0:
            continue
        checked += 1
        assert pillai(a, b, c, r, s) == [(x, y) for x in range(40) for y in range(40) if r*a**x - s*b**y == c]
    assert checked > 250
    raises(ValueError, lambda: pillai(2, 3, 1, r=0))


def test_pillai_bounds() -> None:
    # the Baker-Wüstholz bound for 3**x - 2**y: about 10**11
    bound = _baker_bound(3, 2, 1)
    assert 10**10 < bound < 10**13
    # log 3/log 2 = [1; 1, 1, 2, 2, 3, 1, 5, 2, 23, 2, 2, 1, 1, 55, ...] (OEIS A028507)
    assert _partial_quotients(2, 3, 10**6, 60)[:15] == [1, 1, 1, 2, 2, 3, 1, 5, 2, 23, 2, 2, 1, 1, 55]
    raises(ValueError, lambda: pillai(1, 3, 2))
    raises(ValueError, lambda: pillai(2, 3, 0))


def test_exponential_polynomial_roots_known() -> None:
    one = [F(1)]
    # 2**n = n**2: n = 2, 4 (the negative solution is not an integer)
    assert exponential_polynomial_roots([(one, F(2)), ([F(-1), F(0), F(0)], F(1))]) == [2, 4]
    # 3**n + 4**n = 5**n
    assert exponential_polynomial_roots([(one, F(3)), (one, F(4)), ([F(-1)], F(5))]) == [2]
    # 2**n = n + 1, 2**n + 2**-n = 5/2, 2**n = 3**n
    assert exponential_polynomial_roots([(one, F(2)), ([F(-1), F(-1)], F(1))]) == [0, 1]
    assert exponential_polynomial_roots([(one, F(2)), (one, F(1, 2)), ([F(-5, 2)], F(1))]) == [-1, 1]
    assert exponential_polynomial_roots([(one, F(2)), ([F(-1)], F(3))]) == [0]
    # a polynomial alone: (n - 3)(n + 5)(2 n - 1)
    assert exponential_polynomial_roots([([F(2), F(3), F(-32), F(15)], F(1))]) == [-5, 3]
    # bases close to one: (11/10)**n = n (no integer solution), (3/2)**n = 9 n/8 (9/4 at 2, 27/8 at 3)
    assert exponential_polynomial_roots([(one, F(11, 10)), ([F(-1), F(0)], F(1))]) == []
    assert exponential_polynomial_roots([(one, F(3, 2)), ([F(-9, 8), F(0)], F(1))]) == [2, 3]
    # a zero far beyond the dominance bound, where the polynomial of the
    # largest base and the rest vanish together:
    # (n - 10**6) 5**n + (n - 10**6) (n - 3) 2**n; the search used to run up to
    # twice the size of the coefficients
    million = 10**6
    assert exponential_polynomial_roots([([F(1), F(-million)], F(5)),
                                         ([F(1), F(-million - 3), F(3*million)], F(2))]) == [million]
    assert exponential_polynomial_roots([([F(1), F(-million)], F(5)), ([F(2)], F(1, 3))]) == []
    # equal bases are merged; the zero sum is refused
    assert exponential_polynomial_roots([(one, F(2)), (one, F(2)), ([F(-4)], F(1))]) == [1]
    raises(ValueError, lambda: exponential_polynomial_roots([(one, F(2)), ([F(-1)], F(2))]))
    raises(ValueError, lambda: exponential_polynomial_roots([(one, F(-2))]))


def test_exponential_polynomial_roots_against_search() -> None:
    rng = random.Random(52)
    for _ in range(150):
        bases = rng.sample([F(1, 3), F(1, 2), F(2, 3), F(1), F(3, 2), F(2), F(3), F(5)], rng.randint(1, 3))
        terms = [([F(rng.randint(-6, 6)) for _ in range(rng.randint(1, 3))], base) for base in bases]
        if rng.random() < 0.7:
            # force a zero at n0: adjust the constant coefficient of the first term
            n0 = rng.randint(-6, 6)
            rest = sum((sum((c*n0**(len(p) - 1 - k) for k, c in enumerate(p)), F(0))*base**n0 for p, base in terms[1:]),
                       F(0))
            p0, base0 = terms[0]
            head = sum((c*n0**(len(p0) - 1 - k) for k, c in enumerate(p0[:-1])), F(0))
            p0[-1] = -rest/base0**n0 - head
        try:
            found = exponential_polynomial_roots(terms)
        except ValueError:
            continue

        def value(n: int) -> Fraction:
            return sum((sum((c*n**(len(p) - 1 - k) for k, c in enumerate(p)), F(0))*base**n for p, base in terms), F(0))

        assert found == [n for n in range(-60, 61) if value(n) == 0]
