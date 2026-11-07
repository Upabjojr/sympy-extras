"""Euclidean algorithms on dense polynomials: gcds, resultants,
subresultants.

The module works on the typed representation of :mod:`sympy_extras.polys.densebasic` (a ``list[T]``
over a :class:`~sympy_extras.polys.domains.domain.Domain`): the Euclidean
algorithm over a field (``dup_gcd``, ``dup_gcdex``, ``dup_invert``), the
subresultant polynomial remainder sequence over an integral domain
(``dup_subresultants``, whose last element gives the gcd by its primitive
part), the resultant by the recursive identity on pseudo-remainders
(``dup_resultant``) and the discriminant. Over a polynomial ring domain
these are the multivariate algorithms (``K[y][x]``), with the gcd of the
coefficients computed recursively. It imports nothing from SymPy; the
principal subresultant coefficients on SymPy's representation, used by
the CAD, are in :mod:`sympy_extras.polys.subresultants`.

Examples
========

>>> from sympy_extras.polys.domains import ZZ, QQ, GF
>>> from sympy_extras.polys.euclidtools import dup_gcd, dup_resultant, dup_discriminant, dup_gcdex, dup_invert
>>> dup_gcd([1, 0, -1], [1, -2, 1], ZZ)
[1, -1]
>>> dup_gcd([2, 0, -2], [4, -8, 4], ZZ)
[2, -2]
>>> dup_resultant([1, 0, -2], [1, 0, 0, -3], ZZ)
1
>>> dup_discriminant([1, 1, 1], ZZ)
-3
>>> from fractions import Fraction
>>> dup_gcdex([Fraction(1), Fraction(0), Fraction(-1)], [Fraction(1), Fraction(-2)], QQ)
([Fraction(1, 3)], [Fraction(-1, 3), Fraction(-2, 3)], [Fraction(1, 1)])
>>> dup_invert([1, 1], [1, 0, 1], GF(7))
[3, 4]

References
==========

.. [1] W. S. Brown, *The subresultant PRS algorithm*, ACM Transactions on
       Mathematical Software 4 (1978), 237-249.
.. [2] K. O. Geddes, S. R. Czapor, G. Labahn, *Algorithms for Computer
       Algebra*, Kluwer, 1992, Chapter 7 (polynomial GCD computation,
       the subresultant PRS, the resultant).
.. [3] D. E. Knuth, *The Art of Computer Programming*, vol. 2, 3rd ed.,
       Addison-Wesley, 1998, Section 4.6.1.
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/polys/euclidtools.py (dup_inner_subresultants,
# dup_prs_resultant) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; the sign of Res(f, g) for deg f < deg g corrected.
# ---------------------------------------------------------------------
from __future__ import annotations

from fractions import Fraction
from typing import TypeVar

from .densearith import dup_div, dup_exquo, dup_exquo_ground, dup_mul, dup_mul_ground, dup_prem, dup_sub
from .densebasic import dup_LC, dup_degree
from .densetools import dup_clear_denoms, dup_diff, dup_monic, dup_primitive
from .domains.domain import Domain

__all__ = ['dup_gcd', 'dup_qq_gcd', 'dup_lcm', 'dup_gcdex', 'dup_half_gcdex', 'dup_invert', 'dup_subresultants',
           'dup_resultant', 'dup_prs_resultant', 'dup_inner_subresultants', 'dup_discriminant', 'dup_euclid',
           'dup_prs']

T = TypeVar('T')


def dup_euclid(f: list[T], g: list[T], K: Domain[T]) -> list[T]:
    """The monic gcd over a field by the Euclidean algorithm (each
    remainder made monic, which keeps the coefficients small over
    ``QQ``)."""
    a, b = dup_monic(f, K), dup_monic(g, K)
    while b:
        a, b = b, dup_monic(dup_div(a, b, K)[1], K)
    return a


def dup_prs(f: list[T], g: list[T], K: Domain[T]) -> list[list[T]]:
    """The Euclidean polynomial remainder sequence over a field."""
    sequence = [list(f), list(g)]
    while sequence[-1]:
        sequence.append(dup_div(sequence[-2], sequence[-1], K)[1])
    sequence.pop()
    return sequence


def dup_subresultants(f: list[T], g: list[T], K: Domain[T]) -> list[list[T]]:
    """The subresultant polynomial remainder sequence of ``f`` and ``g``
    over an integral domain (Brown's algorithm): ``r_{i+1} = prem(r_{i-1},
    r_i) / beta_i`` with ``beta_1 = (-1)**(delta_1 + 1)``, ``psi_1 = -1``
    and, for ``i >= 2``, ``psi_i = (-lc(r_{i-1}))**delta_{i-1} /
    psi_{i-1}**(delta_{i-1} - 1)``, ``beta_i = -lc(r_{i-1}) psi_i**delta_i``,
    where ``delta_i = deg r_{i-1} - deg r_i``; every division is exact.
    The sequence ends with the last nonzero remainder."""
    if not f:
        return [list(g)] if g else []
    if not g:
        return [list(f)]
    a, b = list(f), list(g)
    if dup_degree(a) < dup_degree(b):
        a, b = b, a
    sequence = [a, b]
    delta = dup_degree(a) - dup_degree(b)
    beta = K.one if delta % 2 else K.neg(K.one)
    psi = K.neg(K.one)
    while True:
        r = dup_prem(a, b, K)
        if not r:
            break
        r = dup_exquo_ground(r, beta, K)
        sequence.append(r)
        a, b = b, r
        minus_lc = K.neg(dup_LC(a, K))
        psi = _psi_update(minus_lc, psi, delta, K)
        delta = dup_degree(a) - dup_degree(b)
        beta = K.mul(minus_lc, K.pow(psi, delta))
    return sequence


def _psi_update(minus_lc: T, psi: T, delta: int, K: Domain[T]) -> T:
    """``(-lc)**delta / psi**(delta - 1)``, an exact division (``psi``
    itself for ``delta = 0``)."""
    if delta == 0:
        return psi
    if delta == 1:
        return minus_lc
    return K.exquo(K.pow(minus_lc, delta), K.pow(psi, delta - 1))


def dup_gcd(f: list[T], g: list[T], K: Domain[T]) -> list[T]:
    """The greatest common divisor in canonical form: monic over a field
    (the Euclidean algorithm; :func:`dup_qq_gcd` is faster over ``QQ``),
    and over an integral domain the primitive part of the last
    subresultant times the gcd of the contents, with the canonical unit
    of the domain (positive leading coefficient over ``ZZ``; over a
    polynomial ring, recursively)."""
    if not f and not g:
        return []
    if K.is_field:
        return dup_euclid(f, g, K)
    if not f:
        return _canonical(g, K)
    if not g:
        return _canonical(f, K)
    cf, pf = dup_primitive(f, K)
    cg, pg = dup_primitive(g, K)
    content = K.gcd(cf, cg)
    last = dup_subresultants(pf, pg, K)[-1]
    _, part = dup_primitive(last, K)
    return dup_mul_ground(_canonical(part, K), content, K)


def dup_qq_gcd(f: list[Fraction], g: list[Fraction]) -> list[Fraction]:
    """The monic gcd over ``QQ`` through the integers: the denominators
    cleared, the subresultant gcd over ``ZZ``, the result made monic."""
    from .domains.integerring import ZZ
    from .domains.rationalfield import QQ
    _, fz = dup_clear_denoms(f, QQ)
    _, gz = dup_clear_denoms(g, QQ)
    h = dup_gcd([c.numerator for c in fz], [c.numerator for c in gz], ZZ)
    return dup_monic([Fraction(c) for c in h], QQ)


def _canonical(f: list[T], K: Domain[T]) -> list[T]:
    """``f`` times the canonical unit of its leading coefficient."""
    if not f:
        return []
    u = K.canonical_unit(dup_LC(f, K))
    return list(f) if K.is_one(u) else dup_mul_ground(f, u, K)


def dup_lcm(f: list[T], g: list[T], K: Domain[T]) -> list[T]:
    """The least common multiple, in canonical form."""
    if not f or not g:
        return []
    h = dup_gcd(f, g, K)
    quotient = dup_exquo(dup_mul(f, g, K), h, K)
    return dup_monic(quotient, K) if K.is_field else _canonical(quotient, K)


def dup_gcdex(f: list[T], g: list[T], K: Domain[T]) -> tuple[list[T], list[T], list[T]]:
    """``(s, t, h)`` with ``s f + t g = h`` the monic gcd, over a field
    (the extended Euclidean algorithm)."""
    if not K.is_field:
        raise ValueError('the extended Euclidean algorithm needs a field, got %s' % K.name)
    r0, r1 = list(f), list(g)
    s0: list[T] = [K.one]
    s1: list[T] = []
    t0: list[T] = []
    t1: list[T] = [K.one]
    while r1:
        q, r = dup_div(r0, r1, K)
        r0, r1 = r1, r
        s0, s1 = s1, dup_sub(s0, dup_mul(q, s1, K), K)
        t0, t1 = t1, dup_sub(t0, dup_mul(q, t1, K), K)
    if not r0:
        return [], [], []
    inverse = K.invert(dup_LC(r0, K))
    return dup_mul_ground(s0, inverse, K), dup_mul_ground(t0, inverse, K), dup_mul_ground(r0, inverse, K)


def dup_half_gcdex(f: list[T], g: list[T], K: Domain[T]) -> tuple[list[T], list[T]]:
    """``(s, h)`` with ``s f = h (mod g)``, ``h`` the monic gcd."""
    s, _, h = dup_gcdex(f, g, K)
    return s, h


def dup_invert(f: list[T], g: list[T], K: Domain[T]) -> list[T]:
    """The inverse of ``f`` modulo ``g`` over a field; ``ValueError``
    when they are not coprime."""
    s, h = dup_half_gcdex(f, g, K)
    if dup_degree(h) != 0:
        raise ValueError('%s and %s are not coprime' % (f, g))
    return dup_div(s, g, K)[1]


def dup_resultant(f: list[T], g: list[T], K: Domain[T]) -> T:
    """The resultant, over any integral domain: with ``m = deg f >= n =
    deg g`` and ``r`` the pseudo-remainder of ``f`` by ``g`` of degree
    ``k`` and content ``c`` (``r = lc(g)**(m-n+1) f mod g``),

    ``Res(f, g) = (-1)**(m n) lc(g)**(m - k) c**n Res(g, r/c) / lc(g)**(n (m - n + 1))``,

    the division being exact in the domain."""
    if not f or not g:
        return K.zero
    m, n = dup_degree(f), dup_degree(g)
    if m < n:
        value = dup_resultant(g, f, K)
        return K.neg(value) if (m*n) % 2 else value
    if n == 0:
        return K.pow(g[0], m)
    r = dup_prem(f, g, K)
    if not r:
        return K.zero
    k = dup_degree(r)
    c, part = dup_primitive(r, K)
    lc = g[0]
    inner = dup_resultant(g, part, K)
    numerator = K.mul(K.mul(K.pow(lc, m - k), K.pow(c, n)), inner)
    if (m*n) % 2:
        numerator = K.neg(numerator)
    return K.exquo(numerator, K.pow(lc, n*(m - n + 1)))


# PORTED FROM SYMPY: sympy/polys/euclidtools.py:dup_inner_subresultants
def dup_inner_subresultants(f: list[T], g: list[T], K: Domain[T]) -> tuple[list[list[T]], list[T]]:
    """The subresultant PRS of ``f`` and ``g`` (the one of higher degree
    first) and the scalar subresultants ``S`` (``S[0] = 1``), with exact
    divisions only: the last scalar is the resultant when the sequence
    ends with a constant."""
    n, m = dup_degree(f), dup_degree(g)
    if n < m:
        f, g = g, f
        n, m = m, n
    if not f:
        return [], []
    if not g:
        return [list(f)], [K.one]
    R: list[list[T]] = [list(f), list(g)]
    d = n - m
    b = K.pow(K.neg(K.one), d + 1)
    h = dup_mul_ground(dup_prem(f, g, K), b, K)
    lc = dup_LC(g, K)
    c = K.pow(lc, d)
    S: list[T] = [K.one, c]
    c = K.neg(c)
    while h:
        k = dup_degree(h)
        R.append(h)
        f, g, m, d = g, h, k, m - k
        b = K.mul(K.neg(lc), K.pow(c, d))
        h = dup_exquo_ground(dup_prem(f, g, K), b, K)
        lc = dup_LC(g, K)
        if d > 1:
            c = K.exquo(K.pow(K.neg(lc), d), K.pow(c, d - 1))
        else:
            c = K.neg(lc)
        S.append(K.neg(c))
    return R, S


# PORTED FROM SYMPY: sympy/polys/euclidtools.py:dup_prs_resultant
def dup_prs_resultant(f: list[T], g: list[T], K: Domain[T]) -> T:
    """The resultant by the subresultant PRS (no gcd computations: the
    fast route over polynomial rings), the Sylvester determinant of ``f``
    and ``g`` in this order (SymPy swaps them when ``deg f < deg g``
    without the sign ``(-1)**(deg f deg g)``).

    >>> from sympy_extras.polys.domains import ZZ
    >>> dup_prs_resultant([-7, -3], [-2, 6, 1, -6], ZZ)
    1773
    """
    if not f or not g:
        return K.zero
    R, S = dup_inner_subresultants(f, g, K)
    if dup_degree(R[-1]) > 0:
        return K.zero
    value = S[-1]
    if dup_degree(f) < dup_degree(g) and (dup_degree(f) * dup_degree(g)) % 2:
        value = K.neg(value)
    return value


def dup_discriminant(f: list[T], K: Domain[T]) -> T:
    """``(-1)**(m (m - 1)/2) Res(f, f') / lc(f)`` for ``m = deg f``
    (``0`` for a constant)."""
    m = dup_degree(f)
    if m <= 0:
        return K.zero
    value = dup_resultant(f, dup_diff(f, 1, K), K)
    if (m*(m - 1)//2) % 2:
        value = K.neg(value)
    return K.exquo(value, f[0])
