"""The fields GF(p**n) and factorisation over them: field identities, the
prime field as a special case (galoistools is the oracle), and the
theorem on the splitting of irreducible polynomials in extensions."""
from __future__ import annotations

import random
from math import gcd

from sympy.testing.pytest import raises

from sympy_extras.polys.densearith import dup_mul
from sympy_extras.polys.densebasic import dup_degree
from sympy_extras.polys.densetools import dup_monic
from sympy_extras.polys.domains import GF
from sympy_extras.polys.domains.galoisfield import GaloisField
from sympy_extras.polys.galoisfactor import gfq_ddf, gfq_factor, gfq_irreducible_p, gfq_pow_mod, gfq_sqf_list
from sympy_extras.polys.galoistools import gf_factor, gf_irreducible

Poly = list[list[int]]


def _random_poly(rng: random.Random, K: GaloisField, degree: int) -> Poly:
    return [K.one] + [K.random(rng) for _ in range(degree)]


def test_field_identities() -> None:
    rng = random.Random(101)
    for p, n in ((2, 1), (2, 3), (2, 8), (3, 2), (5, 2), (7, 3), (3, 4)):
        K = GaloisField(p, n)
        assert K.order == p**n and K.characteristic == p and K.is_field
        for _ in range(30):
            a, b, c = K.random(rng), K.random(rng), K.random(rng)
            assert K.mul(a, K.add(b, c)) == K.add(K.mul(a, b), K.mul(a, c))
            assert K.mul(K.mul(a, b), c) == K.mul(a, K.mul(b, c))
            assert K.add(a, K.neg(a)) == K.zero and K.sub(a, a) == []
            if a:
                assert K.mul(a, K.invert(a)) == K.one and K.exquo(K.mul(a, b), a) == b
                # Lagrange: a**(q - 1) = 1
                assert K.pow(a, K.order - 1) == K.one
            # the Frobenius map is additive, its n-th power the identity; p-th roots
            assert K.frobenius(K.add(a, b)) == K.add(K.frobenius(a), K.frobenius(b))
            assert K.pow(a, K.order) == a and K.frobenius(K.pth_root(a)) == a
            # the trace is additive with values in the prime field
            assert (K.trace(a) + K.trace(b)) % p == K.trace(K.add(a, b))
    K = GaloisField(2, 3)
    assert K.modulus == [1, 0, 1, 1] and repr(K) == 'GF(2**3)' and repr(GaloisField(5, 1)) == 'GF(5)'
    assert len({tuple(e) for e in K.elements()}) == 8
    assert K.from_int(3) == [1] and K.from_int(2) == [] and K.convert(5, GF(2)) == [1]
    assert K.to_str([1, 0, 1]) == 'a**2 + 1' and K.to_str([]) == '0'
    assert K == GaloisField(2, 3) and K != GaloisField(2, 3, [1, 1, 0, 1]) and hash(K) == hash(GaloisField(2, 3))
    # the generator of GF(8) has order 7; GF(9) = GF(3)[i]
    assert K.pow(K.generator, 7) == K.one
    F = GaloisField(3, 2, [1, 0, 1])
    assert F.pow(F.generator, 2) == [2]
    raises(ValueError, lambda: GaloisField(2, 2, [1, 0, 1]))        # x**2 + 1 = (x + 1)**2
    raises(ValueError, lambda: GaloisField(2, 2, [1, 1]))
    raises(ValueError, lambda: GaloisField(2, 0))
    raises(ZeroDivisionError, lambda: K.invert([]))


def test_factorisation_rebuilds_the_polynomial() -> None:
    rng = random.Random(102)
    for p, n in ((2, 2), (2, 3), (3, 2), (5, 2), (3, 3), (2, 4)):
        K = GaloisField(p, n)
        for _ in range(12):
            f: Poly = [K.random(rng) or K.one]
            for _ in range(rng.randint(1, 4)):
                g = _random_poly(rng, K, rng.randint(1, 3))
                for _ in range(rng.choice([1, 1, 2, p])):
                    f = dup_mul(f, g, K)
            lc, factors = gfq_factor(f, K)
            product: Poly = [lc]
            for g, k in factors:
                assert g[0] == K.one and gfq_irreducible_p(g, K)
                for _ in range(k):
                    product = dup_mul(product, g, K)
            assert product == f
            # the factors are distinct and sorted
            assert len({tuple(tuple(c) for c in g) for g, _ in factors}) == len(factors)
            # no factor of degree one is missed: the roots are exactly those found
            roots = [e for e in K.elements() if not _value(f, e, K)]
            linear = [K.neg(g[1]) for g, _ in factors if dup_degree(g) == 1]
            assert sorted(roots) == sorted(linear)


def _value(f: Poly, a: list[int], K: GaloisField) -> list[int]:
    value = K.zero
    for c in f:
        value = K.add(K.mul(value, a), c)
    return value


def test_prime_fields_agree_with_galoistools() -> None:
    rng = random.Random(103)
    for p in (2, 3, 7):
        F = GF(p)
        K = GaloisField(p, 1)
        for _ in range(20):
            f = [rng.randrange(1, p)] + [rng.randrange(p) for _ in range(rng.randint(1, 8))]
            lc, factors = gf_factor(f, F)
            lifted: Poly = [[c] if c else [] for c in f]
            lc_q, factors_q = gfq_factor(lifted, K)
            assert lc_q == [lc]
            assert [([c[0] if c else 0 for c in g], k) for g, k in factors_q] == factors


def test_irreducible_polynomials_split_in_extensions() -> None:
    # an irreducible polynomial of degree d over GF(p) splits over GF(p**n)
    # into gcd(d, n) irreducible factors of degree d/gcd(d, n) (Lidl-Niederreiter, Theorem 3.46)
    rng = random.Random(104)
    for p in (2, 3, 5):
        F = GF(p)
        for n in (2, 3, 4):
            K = GaloisField(p, n)
            for d in (2, 3, 4, 6):
                f = gf_irreducible(d, F, rng)
                lifted: Poly = [[c] if c else [] for c in f]
                _, factors = gfq_factor(lifted, K)
                g = gcd(d, n)
                assert len(factors) == g
                assert all(dup_degree(h) == d//g and k == 1 for h, k in factors)
                assert gfq_irreducible_p(lifted, K) == (g == 1)


def test_square_free_decomposition_and_stages() -> None:
    K = GaloisField(2, 2)
    one: list[int] = [1]
    # x**3 + x = x (x + 1)**2; x**4 + 1 = (x + 1)**4; (x + a)**6 = ((x + a)**3)**2
    assert gfq_sqf_list([one, [], one, []], K) == (one, [([one, []], 1), ([one, one], 2)])
    assert gfq_factor([one, [], [], [], one], K) == (one, [([one, one], 4)])
    a = K.generator
    linear: Poly = [one, a]
    power: Poly = [one]
    for _ in range(6):
        power = dup_mul(power, linear, K)
    assert gfq_sqf_list(power, K) == (one, [(linear, 6)])
    assert gfq_factor(power, K) == (one, [(linear, 6)])
    assert gfq_factor([], K) == ([], []) and gfq_factor([a], K) == (a, [])
    assert not gfq_irreducible_p([a], K)
    # x**q - x is the product of all the monic linear polynomials
    x: Poly = [one, []]
    frobenius = gfq_pow_mod(x, K.order, [one, [], [], [], [], [], one], K)
    assert dup_degree(frobenius) == 4
    all_linear = gfq_ddf(dup_monic([one, [], [], one, []], K), K)       # x**4 + x = x**4 - x
    assert all_linear == [(dup_monic([one, [], [], one, []], K), 1)]
