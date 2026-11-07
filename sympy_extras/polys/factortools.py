r"""Factorisation of dense univariate polynomials over ``ZZ`` and ``QQ``:
Zassenhaus' algorithm.

For a primitive square-free $f \in \mathbb{Z}[x]$ with leading coefficient
$l$: a prime $p$ not dividing $l$ such that $f$ stays square-free modulo
$p$ is chosen (among the first few such primes, the one giving the fewest
modular factors), $f$ is factored over $GF(p)$
(:mod:`sympy_extras.polys.galoistools`), the factorisation $f \equiv l
f_1 \cdots f_r \pmod p$ is lifted to a modulus $m = p^{2^j}$ above twice
the Mignotte bound $B = \sqrt{n + 1}\, 2^n \|f\|_\infty |l|$ by quadratic
Hensel lifting (pairs of factors, the list split in halves recursively),
and the true factors are found by trial division of the primitive parts
of $l \prod_{i \in S} f_i \bmod m$ (symmetric residues) for the subsets
$S$ of increasing size up to half the remaining factors. Over $\mathbb{Q}$
the denominators are cleared first. The square-free decomposition comes
first, so that each square-free part is factored once.

Examples
========

>>> from fractions import Fraction
>>> from sympy_extras.polys.domains import ZZ, QQ
>>> from sympy_extras.polys.factortools import dup_zz_factor, dup_qq_factor, dup_zz_hensel_lift
>>> dup_zz_factor([2, 0, -8], ZZ)
(2, [([1, -2], 1), ([1, 2], 1)])
>>> dup_zz_factor([1, 0, 0, 0, 1], ZZ)
(1, [([1, 0, 0, 0, 1], 1)])
>>> dup_zz_factor([-1, 0, 2, 0, -1], ZZ)
(-1, [([1, -1], 2), ([1, 1], 2)])
>>> dup_qq_factor([Fraction(1, 2), Fraction(0), Fraction(-2)], QQ)
(Fraction(1, 2), [([Fraction(1, 1), Fraction(-2, 1)], 1), ([Fraction(1, 1), Fraction(2, 1)], 1)])

References
==========

.. [1] J. von zur Gathen, J. Gerhard, *Modern Computer Algebra*, 3rd ed.,
       Cambridge University Press, 2013, Chapter 15 (Hensel lifting,
       Algorithms 15.10 and 15.17; Zassenhaus' algorithm, 15.19; the
       Mignotte bound, Corollary 6.33).
.. [2] H. Zassenhaus, *On Hensel factorization I*, Journal of Number
       Theory 1 (1969), 291-311.
.. [3] K. O. Geddes, S. R. Czapor, G. Labahn, *Algorithms for Computer
       Algebra*, Kluwer, 1992, Chapter 8.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from math import isqrt
from typing import Optional

from .densearith import dup_add, dup_add_term, dup_div, dup_max_norm, dup_mul, dup_mul_ground, dup_sub
from .densebasic import dup_LC, dup_convert, dup_degree, dup_strip
from .densetools import dup_clear_denoms, dup_decompose_content, dup_diff, dup_monic, dup_primitive, dup_shift, dup_trunc
from .domains.algebraicfield import AlgebraicField
from .domains.finitefield import FiniteField
from .domains.integerring import IntegerRing, ZZ
from .domains.polynomialring import PolynomialRing
from .domains.rationalfield import QQ, RationalField
from .euclidtools import dup_gcd, dup_gcdex, dup_resultant
from .galoistools import gf_factor_sqf
from .sqfreetools import dup_sqf_list, dup_sqf_p

__all__ = ['dup_zz_factor', 'dup_zz_factor_sqf', 'dup_qq_factor', 'dup_zz_mignotte_bound', 'dup_zz_hensel_step',
           'dup_zz_hensel_lift', 'dup_zz_zassenhaus', 'dup_zz_irreducible_p', 'dup_ext_sqf_norm', 'dup_ext_factor',
           'dup_zz_cyclotomic_poly', 'dup_zz_cyclotomic_factor']

#: how many good primes are tried, keeping the one with the fewest factors
_PRIMES_TRIED = 5


def dup_zz_mignotte_bound(f: list[int], K: IntegerRing) -> int:
    """``sqrt(n + 1) 2**n ||f||_inf |lc(f)|``, a bound on the absolute
    values of the coefficients of ``lc(f) g`` for every factor ``g`` of
    ``f`` (Mignotte)."""
    n = dup_degree(f)
    return (isqrt(n + 1) + 1)*2**n*dup_max_norm(f, K)*K.abs(dup_LC(f, K))


def _primes() -> list[int]:
    """Primes to try, in order."""
    found: list[int] = []
    candidate = 2
    while len(found) < 60:
        if all(candidate % q for q in found):
            found.append(candidate)
        candidate += 1
    return found


def dup_zz_hensel_step(m: int, f: list[int], g: list[int], h: list[int], s: list[int], t: list[int]
                       ) -> tuple[list[int], list[int], list[int], list[int]]:
    """One quadratic Hensel step: from ``f = g h``, ``s g + t h = 1``
    modulo ``m`` (``h`` monic, ``deg s < deg h``, ``deg t < deg g``) to
    the same congruences modulo ``m**2``."""
    M = m*m
    e = dup_trunc(dup_sub(f, dup_mul(g, h, ZZ), ZZ), M, ZZ)
    q, r = dup_div(dup_mul(s, e, ZZ), h, ZZ)
    q, r = dup_trunc(q, M, ZZ), dup_trunc(r, M, ZZ)
    g1 = dup_trunc(dup_add(dup_add(g, dup_mul(t, e, ZZ), ZZ), dup_mul(q, g, ZZ), ZZ), M, ZZ)
    h1 = dup_trunc(dup_add(h, r, ZZ), M, ZZ)
    b = dup_trunc(dup_sub(dup_add(dup_mul(s, g1, ZZ), dup_mul(t, h1, ZZ), ZZ), [1], ZZ), M, ZZ)
    c, d = dup_div(dup_mul(s, b, ZZ), h1, ZZ)
    c, d = dup_trunc(c, M, ZZ), dup_trunc(d, M, ZZ)
    s1 = dup_trunc(dup_sub(s, d, ZZ), M, ZZ)
    t1 = dup_trunc(dup_sub(dup_sub(t, dup_mul(t, b, ZZ), ZZ), dup_mul(c, g1, ZZ), ZZ), M, ZZ)
    return g1, h1, s1, t1


def _lift_pair(f: list[int], g: list[int], h: list[int], p: int, target: int) -> tuple[list[int], list[int], int]:
    """``f = g h`` modulo ``p`` lifted to modulus ``m = p**(2**j) >=
    target``; ``h`` monic. Returns ``(g, h, m)``."""
    F = FiniteField(p)
    gp = dup_convert(g, ZZ, F)
    hp = dup_convert(h, ZZ, F)
    sp, tp, one = dup_gcdex(gp, hp, F)
    if dup_degree(one) != 0:
        raise ValueError('the modular factors are not coprime')
    s = dup_trunc(dup_convert(sp, F, ZZ), p, ZZ)
    t = dup_trunc(dup_convert(tp, F, ZZ), p, ZZ)
    g, h = dup_trunc(g, p, ZZ), dup_trunc(h, p, ZZ)
    m = p
    while m < target:
        g, h, s, t = dup_zz_hensel_step(m, f, g, h, s, t)
        m = m*m
    return g, h, m


def dup_zz_hensel_lift(p: int, f: list[int], factors: list[list[int]], target: int) -> tuple[list[list[int]], int]:
    """The factorisation ``f = lc(f) f_1 ... f_r`` modulo ``p`` (monic
    ``f_i``) lifted to a modulus ``m = p**(2**j) >= target``: the lifted
    monic factors and ``m`` (multifactor lifting by halves)."""
    r = len(factors)
    lc = dup_LC(f, ZZ)
    if r == 1:
        # f = lc f_1: the factor is f made monic modulo m
        m = p
        while m < target:
            m = m*m
        inverse = pow(lc % m, -1, m)
        return [dup_trunc(dup_mul_ground(f, inverse, ZZ), m, ZZ)], m
    k = r//2
    F = FiniteField(p)
    g = [lc % p] if lc % p else []
    for factor in factors[:k]:
        g = dup_mul(g, dup_convert(factor, ZZ, F), F)
    h: list[int] = [1]
    for factor in factors[k:]:
        h = dup_mul(h, dup_convert(factor, ZZ, F), F)
    g_lifted, h_lifted, m = _lift_pair(f, dup_convert(g, F, ZZ), dup_convert(h, F, ZZ), p, target)
    left, _ = dup_zz_hensel_lift(p, g_lifted, factors[:k], target)
    right, _ = dup_zz_hensel_lift(p, h_lifted, factors[k:], target)
    return left + right, m


def _good_prime(f: list[int]) -> tuple[int, list[list[int]]]:
    """A prime not dividing the leading coefficient modulo which ``f``
    stays square-free, with the modular factors; the one with the fewest
    factors among the first few."""
    lc = dup_LC(f, ZZ)
    best: tuple[int, list[list[int]]] = (0, [])
    tried = 0
    for p in _primes():
        if lc % p == 0:
            continue
        F = FiniteField(p)
        fp = dup_convert(f, ZZ, F)
        if dup_degree(fp) != dup_degree(f):
            continue
        if dup_degree(dup_gcd(fp, dup_diff(fp, 1, F), F)) > 0:
            continue
        _, factors = gf_factor_sqf(fp, F)
        if not best[0] or len(factors) < len(best[1]):
            best = (p, [dup_convert(g, F, ZZ) for g in factors])
        tried += 1
        if tried >= _PRIMES_TRIED or len(best[1]) == 1:
            break
    if not best[0]:
        raise ValueError('no suitable prime found for %s' % (f,))
    return best


def dup_zz_cyclotomic_poly(n: int) -> list[int]:
    """The ``n``-th cyclotomic polynomial: ``x**n - 1`` divided by the
    cyclotomic polynomials of the proper divisors of ``n``."""
    result = [1] + [0]*(n - 1) + [-1]
    for d in range(1, n):
        if n % d == 0:
            result = dup_div(result, dup_zz_cyclotomic_poly(d), ZZ)[0]
    return result


def dup_zz_cyclotomic_factor(f: list[int]) -> Optional[list[list[int]]]:
    """The irreducible factors of ``x**n - 1`` (the cyclotomic
    polynomials of the divisors of ``n``) and of ``x**n + 1`` (those of
    the divisors of ``2 n`` which do not divide ``n``); ``None`` when
    ``f`` has another shape."""
    n = dup_degree(f)
    if n < 1 or f[0] != 1 or any(f[1:-1]) or f[-1] not in (1, -1):
        return None
    if f[-1] == -1:
        divisors = [d for d in range(1, n + 1) if n % d == 0]
    else:
        divisors = [d for d in range(1, 2*n + 1) if (2*n) % d == 0 and n % d != 0]
    return sorted((dup_zz_cyclotomic_poly(d) for d in divisors), key=lambda g: (len(g), g))


def dup_zz_zassenhaus(f: list[int]) -> list[list[int]]:
    """The irreducible factors (primitive, positive leading coefficient)
    of a primitive square-free ``f`` with positive leading coefficient,
    sorted."""
    n = dup_degree(f)
    if n <= 1:
        return [list(f)]
    cyclotomic = dup_zz_cyclotomic_factor(f)
    if cyclotomic is not None:
        return cyclotomic
    p, modular = _good_prime(f)
    if len(modular) == 1:
        return [list(f)]
    bound = dup_zz_mignotte_bound(f, ZZ)
    lifted, m = dup_zz_hensel_lift(p, f, modular, 2*bound + 1)
    remaining = list(lifted)
    factors: list[list[int]] = []
    current = list(f)
    size = 1
    while 2*size <= len(remaining):
        found = False
        lc = dup_LC(current, ZZ)
        for subset in combinations(range(len(remaining)), size):
            candidate = [lc % m]
            for i in subset:
                candidate = dup_trunc(dup_mul(candidate, remaining[i], ZZ), m, ZZ)
            _, candidate = dup_primitive(candidate, ZZ)
            if dup_LC(candidate, ZZ) < 0:
                candidate = [-c for c in candidate]
            quotient = _zz_trial_division(current, candidate) if _divides_leading(current, candidate) else None
            if quotient is not None:
                factors.append(candidate)
                current = quotient
                remaining = [g for i, g in enumerate(remaining) if i not in subset]
                found = True
                break
        if not found:
            size += 1
    if dup_degree(current) > 0:
        factors.append(current)
    factors.sort(key=lambda g: (len(g), g))
    return factors


def _zz_trial_division(f: list[int], g: list[int]) -> Optional[list[int]]:
    """The quotient ``f/g`` over ``ZZ`` when ``g`` divides ``f`` exactly,
    ``None`` otherwise (the division stops at the first coefficient which
    is not divisible)."""
    df, dg = dup_degree(f), dup_degree(g)
    if df < dg:
        return None
    lc = g[0]
    remainder = list(f)
    quotient: list[int] = [0]*(df - dg + 1)
    for i in range(df - dg + 1):
        c = remainder[i]
        if c % lc:
            return None
        q = c//lc
        quotient[i] = q
        if q:
            for j in range(1, dg + 1):
                remainder[i + j] -= q*g[j]
    if any(remainder[df - dg + 1:]):
        return None
    return quotient


def _divides_leading(f: list[int], g: list[int]) -> bool:
    """Whether a trial division of ``f`` by ``g`` can succeed: the degrees
    and the leading and trailing coefficients allow it."""
    if not g or dup_degree(g) > dup_degree(f):
        return False
    if dup_LC(f, ZZ) % dup_LC(g, ZZ):
        return False
    tf, tg = f[-1], g[-1]
    if tg and tf % tg:
        return False
    return True


def dup_zz_factor_sqf(f: list[int], K: IntegerRing) -> tuple[int, list[list[int]]]:
    """The content (with the sign) and the irreducible factors of a
    square-free ``f``."""
    content, part = dup_decompose_content(f, K)
    if dup_degree(part) <= 0:
        return content, []
    return content, dup_zz_zassenhaus(part)


def dup_zz_factor(f: list[int], K: IntegerRing) -> tuple[int, list[tuple[list[int], int]]]:
    """The content (with the sign of the leading coefficient) and the
    irreducible factors with their multiplicities, primitive with
    positive leading coefficients, sorted by degree then
    lexicographically."""
    if not f:
        return 0, []
    content, sqf = dup_sqf_list(f, K)
    factors: list[tuple[list[int], int]] = []
    for g, k in sqf:
        # a power of x comes out of the trailing zeros
        zeros = 0
        while g and g[-1] == 0:
            g = g[:-1]
            zeros += 1
        if zeros:
            factors.append(([1, 0], k*zeros))
        unit, irreducibles = dup_zz_factor_sqf(g, K)
        if unit < 0:
            content = -content
        for h in irreducibles:
            factors.append((h, k))
    factors.sort(key=lambda item: (len(item[0]), item[0], item[1]))
    return content, factors


def dup_qq_factor(f: list[Fraction], K: RationalField) -> tuple[Fraction, list[tuple[list[Fraction], int]]]:
    """The constant and the irreducible factors (primitive with integer
    coefficients, positive leading coefficients) with their
    multiplicities."""
    if not f:
        return Fraction(0), []
    common, cleared = dup_clear_denoms(f, K)
    content, factors = dup_zz_factor([c.numerator for c in cleared], ZZ)
    return Fraction(content, common), [([Fraction(c) for c in g], k) for g, k in factors]


def dup_zz_irreducible_p(f: list[int], K: IntegerRing) -> bool:
    """Whether ``f`` is irreducible over ``ZZ`` (a primitive polynomial of
    positive degree with one factor of multiplicity one)."""
    if dup_degree(f) <= 0:
        return False
    content, factors = dup_zz_factor(f, K)
    return abs(content) == 1 and len(factors) == 1 and factors[0][1] == 1



# ---------------------------------------------------------------------------
# number fields: Trager's algorithm


def dup_ext_norm(f: list[list[Fraction]], K: AlgebraicField) -> list[Fraction]:
    """The norm ``N(f)`` of a polynomial over ``QQ(alpha)``: the resultant
    in ``alpha`` of the minimal polynomial and ``f`` (a polynomial in
    ``x`` with rational coefficients, the product of the conjugates of
    ``f``)."""
    R = PolynomialRing(QQ, 'x')
    # f as a polynomial in alpha with coefficients in QQ[x]
    n = dup_degree(f)
    by_alpha: list[list[Fraction]] = [[] for _ in range(K.degree)]
    for i, c in enumerate(f):
        for j, a in enumerate(c):
            # coefficient a of alpha**(len(c) - 1 - j) x**(n - i)
            k = len(c) - 1 - j
            by_alpha[K.degree - 1 - k] = dup_add_term(by_alpha[K.degree - 1 - k], a, n - i, QQ)
    m_x: list[list[Fraction]] = [[c] if c else [] for c in K.minpoly]
    return dup_resultant(m_x, dup_strip(by_alpha, R), R)


def dup_ext_sqf_norm(f: list[list[Fraction]], K: AlgebraicField) -> tuple[int, list[list[Fraction]], list[Fraction]]:
    """``(s, g, N)`` with ``g = f(x - s alpha)`` and ``N = N(g)``
    square-free (the shifts ``s = 0, 1, -1, 2, ...`` are tried in turn;
    finitely many fail for a square-free ``f``)."""
    s = 0
    while True:
        shift = K.mul(K.from_int(-s), K.alpha)
        g = dup_shift(f, shift, K) if s else list(f)
        norm = dup_ext_norm(g, K)
        if dup_sqf_p(norm, QQ):
            return s, g, norm
        s = -s if s > 0 else 1 - s


def dup_ext_factor(f: list[list[Fraction]], K: AlgebraicField) -> tuple[list[Fraction], list[tuple[list[list[Fraction]], int]]]:
    """The leading coefficient and the monic irreducible factors over
    ``QQ(alpha)`` with their multiplicities (Trager: a shift making the
    norm square-free, the norm factored over ``QQ``, and each factor of
    the norm gives the factor ``gcd(g, N_i)`` over the field)."""
    if not f:
        return [], []
    lc = dup_LC(f, K)
    f = dup_monic(f, K)
    _, sqf = dup_sqf_list(f, K)
    factors: list[tuple[list[list[Fraction]], int]] = []
    for g, k in sqf:
        if dup_degree(g) == 1:
            factors.append((g, k))
            continue
        s, shifted, norm = dup_ext_sqf_norm(g, K)
        _, norm_factors = dup_qq_factor(norm, QQ)
        back = K.mul(K.from_int(s), K.alpha)
        for h, _ in norm_factors:
            over_field: list[list[Fraction]] = [K.from_rational(c) for c in h]
            factor = dup_gcd(shifted, over_field, K)
            if dup_degree(factor) > 0:
                factors.append((dup_shift(factor, back, K) if s else factor, k))
    factors.sort(key=lambda item: (len(item[0]), item[1]))
    return lc, factors

