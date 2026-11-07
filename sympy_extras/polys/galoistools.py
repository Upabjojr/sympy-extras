r"""Factorisation of dense polynomials over the prime fields ``GF(p)``.

The arithmetic is the generic dense one over the domain
:class:`~sympy_extras.polys.domains.finitefield.FiniteField`; this module
adds what factorisation needs:

* powers modulo a polynomial (``gf_pow_mod``) and the Frobenius map
  $h \mapsto h^p \bmod f$ by the precomputed powers of $x^p$
  (``gf_frobenius_monomial_base``, ``gf_frobenius_map``);
* the **distinct-degree factorisation** of a monic square-free $f$:
  $\gcd(f, x^{p^d} - x)$ collects the irreducible factors of degree $d$;
* the **equal-degree factorisation** of Cantor and Zassenhaus: for a
  product of irreducibles of the same degree $d$, a random $h$ gives a
  proper factor $\gcd(f, h^{(p^d - 1)/2} - 1)$ with probability at least
  one half for odd $p$, and $\gcd(f, h + h^2 + h^4 + \cdots + h^{2^{d-1}})$
  (the trace map) for $p = 2$;
* ``gf_factor_sqf`` (the two combined), ``gf_factor`` (with the
  square-free decomposition first) and the irreducibility test of Rabin.

Examples
========

>>> from sympy_extras.polys.domains import GF
>>> from sympy_extras.polys.galoistools import gf_factor, gf_irreducible_p, gf_ddf
>>> F = GF(5)
>>> gf_factor([1, 0, 0, 0, 4], F)
(1, [([1, 1], 1), ([1, 2], 1), ([1, 3], 1), ([1, 4], 1)])
>>> gf_factor([3, 0, 0, 3], F)
(3, [([1, 1], 1), ([1, 4, 1], 1)])
>>> gf_factor([1, 3, 3, 1], F)
(1, [([1, 1], 3)])
>>> gf_ddf([1, 0, 0, 0, 4], F)
[([1, 0, 0, 0, 4], 1)]
>>> gf_irreducible_p([1, 0, 2], F), gf_irreducible_p([1, 0, 1], F)
(True, False)

References
==========

.. [1] J. von zur Gathen, J. Gerhard, *Modern Computer Algebra*, 3rd ed.,
       Cambridge University Press, 2013, Chapter 14 (factoring
       polynomials over finite fields: Algorithms 14.3, 14.8, 14.10, 14.13).
.. [2] D. G. Cantor, H. Zassenhaus, *A new algorithm for factoring
       polynomials over finite fields*, Mathematics of Computation 36
       (1981), 587-592.
.. [3] K. O. Geddes, S. R. Czapor, G. Labahn, *Algorithms for Computer
       Algebra*, Kluwer, 1992, Section 8.4.
"""
from __future__ import annotations

import random

from .densearith import dup_add, dup_div, dup_mul, dup_rem, dup_sub, dup_sqr
from .densebasic import dup_degree, dup_strip
from .densetools import dup_monic
from .domains.finitefield import FiniteField
from .euclidtools import dup_gcd
from .sqfreetools import dup_sqf_list

__all__ = ['gf_pow_mod', 'gf_frobenius_monomial_base', 'gf_frobenius_map', 'gf_ddf', 'gf_edf', 'gf_factor_sqf',
           'gf_factor', 'gf_irreducible_p', 'gf_random', 'gf_irreducible']

#: the seed of the random polynomials of the equal-degree splitting, so
#: that the factorisation is reproducible
_SEED = 1


def gf_pow_mod(f: list[int], n: int, g: list[int], K: FiniteField) -> list[int]:
    """``f**n mod g`` by repeated squaring."""
    if n < 0:
        raise ValueError('negative exponent')
    result = [K.one]
    base = dup_rem(f, g, K)
    while n:
        if n & 1:
            result = dup_rem(dup_mul(result, base, K), g, K)
        n >>= 1
        if n:
            base = dup_rem(dup_sqr(base, K), g, K)
    return result


def gf_frobenius_monomial_base(g: list[int], K: FiniteField) -> list[list[int]]:
    """``x**(i p) mod g`` for ``i = 0, ..., deg g - 1``: the matrix of
    the Frobenius map on ``GF(p)[x]/(g)``."""
    n = dup_degree(g)
    if n <= 0:
        return []
    base: list[list[int]] = [[K.one]]
    x_p = gf_pow_mod([K.one, K.zero], K.p, g, K)
    for _ in range(1, n):
        base.append(dup_rem(dup_mul(base[-1], x_p, K), g, K))
    return base


def gf_frobenius_map(f: list[int], g: list[int], base: list[list[int]], K: FiniteField) -> list[int]:
    """``f**p mod g`` through the precomputed powers ``x**(i p) mod g``:
    the coefficients are their own ``p``-th powers."""
    n = dup_degree(g)
    f = dup_rem(f, g, K)
    result: list[int] = []
    m = dup_degree(f)
    for i in range(m + 1):
        c = f[m - i]
        if c:
            result = dup_add(result, [K.mul(c, b) for b in base[i]], K)
    return dup_strip(result, K) if n > 0 else []


def gf_ddf(f: list[int], K: FiniteField) -> list[tuple[list[int], int]]:
    """The distinct-degree factorisation of a monic square-free ``f``:
    the pairs ``(g, d)`` with ``g`` the product of the irreducible
    factors of degree ``d``, for the degrees which occur."""
    f = dup_monic(f, K)
    factors: list[tuple[list[int], int]] = []
    base = gf_frobenius_monomial_base(f, K)
    x = [K.one, K.zero]
    h = dup_rem(x, f, K)
    d = 1
    while dup_degree(f) >= 2*d:
        h = gf_frobenius_map(h, f, base, K)
        g = dup_gcd(f, dup_sub(h, x, K), K)
        if dup_degree(g) > 0:
            factors.append((g, d))
            f = dup_div(f, g, K)[0]
            h = dup_rem(h, f, K)
            base = gf_frobenius_monomial_base(f, K)
        d += 1
    if dup_degree(f) > 0:
        factors.append((f, dup_degree(f)))
    return factors


def gf_edf(f: list[int], d: int, K: FiniteField, rng: random.Random) -> list[list[int]]:
    """The equal-degree factorisation (Cantor–Zassenhaus) of a monic
    square-free ``f`` all of whose irreducible factors have degree
    ``d``: the list of the factors."""
    n = dup_degree(f)
    if n == d:
        return [f]
    factors: list[list[int]] = []
    to_split = [f]
    while to_split:
        g = to_split.pop()
        if dup_degree(g) == d:
            factors.append(g)
            continue
        split = _edf_split(g, d, K, rng)
        to_split.extend(split)
    factors.sort(key=lambda h: (len(h), h))
    return factors


def _edf_split(f: list[int], d: int, K: FiniteField, rng: random.Random) -> list[list[int]]:
    """A proper factorisation ``[g, f/g]`` of ``f`` (found with probability
    at least one half per trial)."""
    n = dup_degree(f)
    p = K.p
    while True:
        h = dup_strip([rng.randrange(p) for _ in range(n)], K)
        if dup_degree(h) < 1:
            continue
        if p == 2:
            # the trace map h + h**2 + ... + h**(2**(d-1))
            trace = h
            power = h
            for _ in range(d - 1):
                power = dup_rem(dup_sqr(power, K), f, K)
                trace = dup_add(trace, power, K)
            g = dup_gcd(f, trace, K)
        else:
            exponent = (p**d - 1)//2
            power = gf_pow_mod(h, exponent, f, K)
            g = dup_gcd(f, dup_sub(power, [K.one], K), K)
        if 0 < dup_degree(g) < n:
            return [g, dup_div(f, g, K)[0]]


def gf_factor_sqf(f: list[int], K: FiniteField) -> tuple[int, list[list[int]]]:
    """The leading coefficient and the monic irreducible factors of a
    square-free ``f`` (each once), sorted by degree."""
    if not f:
        return K.zero, []
    lc = f[0]
    f = dup_monic(f, K)
    if dup_degree(f) <= 0:
        return lc, []
    rng = random.Random(_SEED)
    factors: list[list[int]] = []
    for g, d in gf_ddf(f, K):
        factors.extend(gf_edf(g, d, K, rng))
    factors.sort(key=lambda h: (len(h), h))
    return lc, factors


def gf_factor(f: list[int], K: FiniteField) -> tuple[int, list[tuple[list[int], int]]]:
    """The leading coefficient and the monic irreducible factors with
    their multiplicities, sorted by degree then lexicographically."""
    if not f:
        return K.zero, []
    lc, sqf = dup_sqf_list(f, K)
    factors: list[tuple[list[int], int]] = []
    for g, k in sqf:
        _, irreducibles = gf_factor_sqf(g, K)
        for h in irreducibles:
            factors.append((h, k))
    factors.sort(key=lambda item: (len(item[0]), item[0], item[1]))
    return lc, factors


def gf_irreducible_p(f: list[int], K: FiniteField) -> bool:
    """Whether ``f`` is irreducible over ``GF(p)`` (Rabin's test:
    ``x**(p**n) = x mod f`` and ``gcd(f, x**(p**(n/q)) - x) = 1`` for
    every prime ``q`` dividing ``n``)."""
    n = dup_degree(f)
    if n <= 0:
        return False
    if n == 1:
        return True
    f = dup_monic(f, K)
    base = gf_frobenius_monomial_base(f, K)
    x = dup_rem([K.one, K.zero], f, K)
    # x**(p**i) mod f for i = 1..n
    powers: list[list[int]] = [x]
    for _ in range(n):
        powers.append(gf_frobenius_map(powers[-1], f, base, K))
    if powers[n] != x:
        return False
    for q in _prime_divisors(n):
        g = dup_gcd(f, dup_sub(powers[n//q], x, K), K)
        if dup_degree(g) > 0:
            return False
    return True


def _prime_divisors(n: int) -> list[int]:
    result: list[int] = []
    q = 2
    while q*q <= n:
        if n % q == 0:
            result.append(q)
            while n % q == 0:
                n //= q
        q += 1
    if n > 1:
        result.append(n)
    return result


def gf_random(n: int, K: FiniteField, rng: random.Random) -> list[int]:
    """A random monic polynomial of degree ``n``."""
    return [K.one] + [rng.randrange(K.p) for _ in range(n)]


def gf_irreducible(n: int, K: FiniteField, rng: random.Random) -> list[int]:
    """A random monic irreducible polynomial of degree ``n``."""
    while True:
        f = gf_random(n, K, rng)
        if gf_irreducible_p(f, K):
            return f
