r"""Factorisation of univariate polynomials over the finite fields
$GF(p^n)$.

The three classical stages, over the field with $q = p^n$ elements
(:class:`~sympy_extras.polys.domains.galoisfield.GaloisField`):

1. *square-free decomposition* in characteristic $p$: with $g = \gcd(f,
   f')$, the factors whose multiplicity is prime to $p$ are separated as
   in characteristic zero, and what remains is a $p$-th power $h(x)^p$,
   whose root $h$ is obtained by taking the $p$-th roots of the
   coefficients ($c \mapsto c^{q/p}$) and treated in turn;
2. *distinct degrees*: the product of the irreducible factors of degree
   $d$ of a square-free $f$ is $\gcd(f, x^{q^d} - x)$, the powers $x^{q^d}
   \bmod f$ being computed by repeated squaring;
3. *equal degrees* (Cantor and Zassenhaus): for a random $h$, $\gcd(f,
   h^{(q^d - 1)/2} - 1)$ splits a product of factors of degree $d$ with
   probability about one half when $p$ is odd; in characteristic 2 the
   trace $h + h^2 + h^4 + \dots + h^{q^d/2}$ plays that role.

The random choices come from a seeded generator: the result, sorted, is
the same in every run.

Examples
========

>>> from sympy_extras.polys.domains.galoisfield import GaloisField
>>> from sympy_extras.polys.galoisfactor import gfq_factor, gfq_irreducible_p
>>> K = GaloisField(2, 2)                   # GF(4) = {0, 1, a, a + 1}
>>> f = [[1], [], [1], [1]]                 # x**3 + x + 1, irreducible over GF(2)
>>> gfq_irreducible_p(f, K), gfq_factor(f, K)
(True, ([1], [([[1], [], [1], [1]], 1)]))
>>> gfq_factor([[1], [1], [1]], K)          # x**2 + x + 1 = (x + a)(x + a + 1)
([1], [([[1], [1, 0]], 1), ([[1], [1, 1]], 1)])
>>> gfq_factor([[1], [], [], [], [1]], K)   # x**4 + 1 = (x + 1)**4
([1], [([[1], [1]], 4)])

References
==========

.. [1] D. G. Cantor, H. Zassenhaus, *A new algorithm for factoring
       polynomials over finite fields*, Mathematics of Computation 36
       (1981), 587-592.
.. [2] J. von zur Gathen, J. Gerhard, *Modern Computer Algebra*, 3rd ed.,
       Cambridge University Press, 2013, Chapter 14 (Algorithms 14.3,
       14.8, 14.10, 14.19).
"""
from __future__ import annotations

import random

from .densearith import dup_add, dup_div, dup_mul, dup_rem, dup_sub
from .densebasic import dup_degree
from .densetools import dup_diff, dup_monic
from .domains.galoisfield import GaloisField
from .euclidtools import dup_gcd

__all__ = ['gfq_factor', 'gfq_sqf_list', 'gfq_ddf', 'gfq_edf', 'gfq_irreducible_p', 'gfq_pow_mod']

#: a polynomial over GF(p**n): a dense list of elements, highest degree first
Poly = list[list[int]]

#: the seed of the random choices of the equal-degree stage
_SEED = 20261110


def gfq_pow_mod(f: Poly, n: int, g: Poly, K: GaloisField) -> Poly:
    """``f**n`` modulo ``g``, by repeated squaring."""
    result: Poly = [K.one]
    base = dup_rem(f, g, K)
    while n:
        if n & 1:
            result = dup_rem(dup_mul(result, base, K), g, K)
        n >>= 1
        if n:
            base = dup_rem(dup_mul(base, base, K), g, K)
    return result


def gfq_sqf_list(f: Poly, K: GaloisField) -> tuple[list[int], list[tuple[Poly, int]]]:
    """The leading coefficient and the square-free decomposition of
    ``f``: pairwise coprime monic square-free ``g_i`` with their
    multiplicities, ``f = lc prod g_i**i``.

    Examples
    ========

    >>> from sympy_extras.polys.domains.galoisfield import GaloisField
    >>> from sympy_extras.polys.galoisfactor import gfq_sqf_list
    >>> gfq_sqf_list([[1], [], [1], []], GaloisField(2, 2))            # x**3 + x = x (x + 1)**2
    ([1], [([[1], []], 1), ([[1], [1]], 2)])
    """
    if not f:
        return K.zero, []
    lc = f[0]
    f = dup_monic(f, K)
    p = K.p
    result: list[tuple[Poly, int]] = []
    multiplier = 1
    while dup_degree(f) > 0:
        derivative = dup_diff(f, 1, K)
        if derivative:
            g = dup_gcd(f, derivative, K)
            w = dup_div(f, g, K)[0]
            i = 1
            while dup_degree(w) > 0:
                y = dup_gcd(w, g, K)
                z = dup_div(w, y, K)[0]
                if dup_degree(z) > 0:
                    result.append((dup_monic(z, K), i*multiplier))
                w = y
                g = dup_div(g, y, K)[0]
                i += 1
            f = g
        if dup_degree(f) > 0:
            # f = h(x)**p: the p-th roots of the coefficients of x**(k p)
            degree = dup_degree(f)
            f = [K.pth_root(f[degree - k*p]) for k in range(degree//p, -1, -1)]
            multiplier *= p
    result.sort(key=lambda item: (item[1], len(item[0]), item[0]))
    return lc, result


def gfq_ddf(f: Poly, K: GaloisField) -> list[tuple[Poly, int]]:
    """The distinct-degree factorisation of a monic square-free ``f``:
    the pairs ``(g, d)``, ``g`` the product of the irreducible factors of
    degree ``d``."""
    factors: list[tuple[Poly, int]] = []
    x: Poly = [K.one, K.zero]
    h = dup_rem(x, f, K)
    d = 1
    while dup_degree(f) >= 2*d:
        h = gfq_pow_mod(h, K.order, f, K)
        g = dup_gcd(f, dup_sub(h, x, K), K)
        if dup_degree(g) > 0:
            g = dup_monic(g, K)
            factors.append((g, d))
            f = dup_div(f, g, K)[0]
            h = dup_rem(h, f, K)
        d += 1
    if dup_degree(f) > 0:
        factors.append((dup_monic(f, K), dup_degree(f)))
    return factors


def gfq_edf(f: Poly, d: int, K: GaloisField, rng: random.Random) -> list[Poly]:
    """The irreducible factors of a monic square-free ``f`` all of whose
    irreducible factors have the degree ``d`` (Cantor–Zassenhaus)."""
    factors: list[Poly] = []
    pending = [f]
    while pending:
        g = pending.pop()
        n = dup_degree(g)
        if n == d:
            factors.append(g)
            continue
        while True:
            h: Poly = [K.random(rng) for _ in range(n)]
            while h and not h[0]:
                h = h[1:]
            if dup_degree(h) < 1:
                continue
            if K.p == 2:
                # the trace of GF(q**d) over GF(2)
                trace = h
                power = h
                for _ in range(K.degree*d - 1):
                    power = dup_rem(dup_mul(power, power, K), g, K)
                    trace = dup_add(trace, power, K)
                part = dup_gcd(g, trace, K)
            else:
                power = gfq_pow_mod(h, (K.order**d - 1)//2, g, K)
                part = dup_gcd(g, dup_sub(power, [K.one], K), K)
            if 0 < dup_degree(part) < n:
                part = dup_monic(part, K)
                pending.append(part)
                pending.append(dup_div(g, part, K)[0])
                break
    return factors


def gfq_factor(f: Poly, K: GaloisField) -> tuple[list[int], list[tuple[Poly, int]]]:
    """The leading coefficient and the monic irreducible factors of
    ``f`` over ``K`` with their multiplicities, sorted by degree, then by
    coefficients.

    Examples
    ========

    >>> from sympy_extras.polys.domains.galoisfield import GaloisField
    >>> from sympy_extras.polys.galoisfactor import gfq_factor
    >>> K = GaloisField(3, 2, [1, 0, 1])           # GF(9) = GF(3)[i]
    >>> gfq_factor([[1], [], [1]], K)              # x**2 + 1 = (x + i)(x + 2 i)
    ([1], [([[1], [1, 0]], 1), ([[1], [2, 0]], 1)])
    >>> lc, factors = gfq_factor([[2], [], [], [], [], [], [], [], [], [1]], K)        # 2 x**9 + 1 = 2 (x + 2)**9
    >>> lc, factors
    ([2], [([[1], [2]], 9)])
    """
    if not f:
        return K.zero, []
    rng = random.Random(_SEED)
    lc, parts = gfq_sqf_list(f, K)
    factors: list[tuple[Poly, int]] = []
    for g, multiplicity in parts:
        for h, d in gfq_ddf(g, K):
            for irreducible in gfq_edf(h, d, K, rng):
                factors.append((irreducible, multiplicity))
    factors.sort(key=lambda item: (len(item[0]), item[0], item[1]))
    return lc, factors


def gfq_irreducible_p(f: Poly, K: GaloisField) -> bool:
    """Whether ``f`` is irreducible over ``K``."""
    if dup_degree(f) < 1:
        return False
    _, factors = gfq_factor(f, K)
    return len(factors) == 1 and factors[0][1] == 1
