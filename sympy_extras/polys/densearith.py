"""Dense univariate polynomials over a typed domain: the arithmetic.

Addition, multiplication (schoolbook; Karatsuba above a threshold),
powers, division with remainder over a field, pseudo-division over a
ring, exact division, and the operations with a ground element. The
representation is that of :mod:`sympy_extras.polys.densebasic`.

Examples
========

>>> from sympy_extras.polys.domains import ZZ, QQ
>>> from sympy_extras.polys.densearith import dup_add, dup_mul, dup_div, dup_pdiv, dup_exquo, dup_pow
>>> dup_add([1, 2], [1, -2], ZZ)
[2, 0]
>>> dup_mul([1, 1], [1, -1], ZZ)
[1, 0, -1]
>>> from sympy_extras.polys.densebasic import dup_from_list
>>> dup_div(dup_from_list([1, 0, -1], QQ), dup_from_list([2, 1], QQ), QQ)
([Fraction(1, 2), Fraction(-1, 4)], [Fraction(-3, 4)])
>>> dup_pdiv([1, 0, -1], [2, 1], ZZ)
([2, -1], [-3])
>>> dup_exquo([1, 0, -1], [1, 1], ZZ)
[1, -1]
>>> dup_pow([1, 1], 3, ZZ)
[1, 3, 3, 1]

References
==========

.. [1] D. E. Knuth, *The Art of Computer Programming*, vol. 2, 3rd ed.,
       Addison-Wesley, 1998, Section 4.6.1 (division of polynomials).
.. [2] K. O. Geddes, S. R. Czapor, G. Labahn, *Algorithms for Computer
       Algebra*, Kluwer, 1992, Section 2.5 (pseudo-division).
"""
from __future__ import annotations

from typing import TypeVar

from .densebasic import dup_strip

from .domains.domain import Domain

__all__ = ['dup_add', 'dup_sub', 'dup_neg', 'dup_mul', 'dup_sqr', 'dup_pow', 'dup_add_term', 'dup_sub_term',
           'dup_mul_term', 'dup_add_ground', 'dup_sub_ground', 'dup_mul_ground', 'dup_quo_ground',
           'dup_exquo_ground', 'dup_div', 'dup_quo', 'dup_rem', 'dup_exquo', 'dup_pdiv', 'dup_prem', 'dup_pquo',
           'dup_pexquo', 'dup_lshift', 'dup_rshift', 'dup_max_norm', 'dup_l1_norm', 'KARATSUBA_THRESHOLD']

T = TypeVar('T')

#: the degree above which multiplication uses Karatsuba's algorithm
KARATSUBA_THRESHOLD = 32


def dup_neg(f: list[T], K: Domain[T]) -> list[T]:
    return [K.neg(c) for c in f]


def dup_add(f: list[T], g: list[T], K: Domain[T]) -> list[T]:
    if not f:
        return list(g)
    if not g:
        return list(f)
    df, dg = len(f), len(g)
    if df == dg:
        return dup_strip([K.add(a, b) for a, b in zip(f, g)], K)
    if df > dg:
        return f[:df - dg] + [K.add(a, b) for a, b in zip(f[df - dg:], g)]
    return g[:dg - df] + [K.add(a, b) for a, b in zip(f, g[dg - df:])]


def dup_sub(f: list[T], g: list[T], K: Domain[T]) -> list[T]:
    if not g:
        return list(f)
    if not f:
        return dup_neg(g, K)
    df, dg = len(f), len(g)
    if df == dg:
        return dup_strip([K.sub(a, b) for a, b in zip(f, g)], K)
    if df > dg:
        return f[:df - dg] + [K.sub(a, b) for a, b in zip(f[df - dg:], g)]
    return dup_neg(g[:dg - df], K) + [K.sub(a, b) for a, b in zip(f, g[dg - df:])]


def dup_add_term(f: list[T], c: T, n: int, K: Domain[T]) -> list[T]:
    """``f + c x**n``."""
    if K.is_zero(c):
        return list(f)
    m = len(f)
    if n >= m:
        return [c] + [K.zero]*(n - m) + f
    result = list(f)
    result[m - 1 - n] = K.add(result[m - 1 - n], c)
    return dup_strip(result, K)


def dup_sub_term(f: list[T], c: T, n: int, K: Domain[T]) -> list[T]:
    return dup_add_term(f, K.neg(c), n, K)


def dup_mul_term(f: list[T], c: T, n: int, K: Domain[T]) -> list[T]:
    """``f c x**n``."""
    if not f or K.is_zero(c):
        return []
    return [K.mul(a, c) for a in f] + [K.zero]*n


def dup_add_ground(f: list[T], c: T, K: Domain[T]) -> list[T]:
    return dup_add_term(f, c, 0, K)


def dup_sub_ground(f: list[T], c: T, K: Domain[T]) -> list[T]:
    return dup_add_term(f, K.neg(c), 0, K)


def dup_mul_ground(f: list[T], c: T, K: Domain[T]) -> list[T]:
    if not f or K.is_zero(c):
        return []
    return dup_strip([K.mul(a, c) for a in f], K)


def dup_quo_ground(f: list[T], c: T, K: Domain[T]) -> list[T]:
    """The coefficients divided by ``c`` with the domain's ``quo``
    (truncated in ``ZZ``)."""
    if K.is_zero(c):
        raise ZeroDivisionError('division by zero')
    if not f:
        return []
    return dup_strip([K.quo(a, c) for a in f], K)


def dup_exquo_ground(f: list[T], c: T, K: Domain[T]) -> list[T]:
    """The coefficients divided exactly by ``c``."""
    if K.is_zero(c):
        raise ZeroDivisionError('division by zero')
    if not f:
        return []
    return dup_strip([K.exquo(a, c) for a in f], K)


def dup_lshift(f: list[T], n: int, K: Domain[T]) -> list[T]:
    """``f x**n``."""
    if not f:
        return []
    return f + [K.zero]*n


def dup_rshift(f: list[T], n: int, K: Domain[T]) -> list[T]:
    """``f / x**n`` (the low terms dropped)."""
    return f[:len(f) - n] if n < len(f) else []


def _schoolbook(f: list[T], g: list[T], K: Domain[T]) -> list[T]:
    df, dg = len(f) - 1, len(g) - 1
    result: list[T] = [K.zero]*(df + dg + 1)
    for i, a in enumerate(f):
        if K.is_zero(a):
            continue
        for j, b in enumerate(g):
            result[i + j] = K.add(result[i + j], K.mul(a, b))
    return dup_strip(result, K)


def dup_mul(f: list[T], g: list[T], K: Domain[T]) -> list[T]:
    if not f or not g:
        return []
    if K.has_dense_kernels:
        return K.dense_mul(f, g)
    if f is g:
        return dup_sqr(f, K)
    df, dg = len(f) - 1, len(g) - 1
    if min(df, dg) < KARATSUBA_THRESHOLD:
        return _schoolbook(f, g, K)
    # Karatsuba: f = f1 x**n + f0, g = g1 x**n + g0
    n = (max(df, dg) + 1)//2
    f1, f0 = dup_rshift(f, n, K), dup_strip(f[len(f) - n:] if n < len(f) else list(f), K)
    g1, g0 = dup_rshift(g, n, K), dup_strip(g[len(g) - n:] if n < len(g) else list(g), K)
    low = dup_mul(f0, g0, K)
    high = dup_mul(f1, g1, K)
    middle = dup_sub(dup_sub(dup_mul(dup_add(f1, f0, K), dup_add(g1, g0, K), K), low, K), high, K)
    return dup_add(dup_add(dup_lshift(high, 2*n, K), dup_lshift(middle, n, K), K), low, K)


def dup_sqr(f: list[T], K: Domain[T]) -> list[T]:
    if not f:
        return []
    if K.has_dense_kernels:
        return K.dense_mul(f, f)
    df = len(f) - 1
    if df >= KARATSUBA_THRESHOLD:
        return _schoolbook(f, f, K)
    result: list[T] = [K.zero]*(2*df + 1)
    for i, a in enumerate(f):
        if K.is_zero(a):
            continue
        result[2*i] = K.add(result[2*i], K.mul(a, a))
        for j in range(i + 1, df + 1):
            b = f[j]
            if not K.is_zero(b):
                twice = K.mul(a, b)
                result[i + j] = K.add(result[i + j], K.add(twice, twice))
    return dup_strip(result, K)


def dup_pow(f: list[T], n: int, K: Domain[T]) -> list[T]:
    """``f**n`` for ``n >= 0`` by repeated squaring."""
    if n < 0:
        raise ValueError('negative exponent')
    if n == 0:
        return [K.one]
    result = [K.one]
    base = list(f)
    while n:
        if n & 1:
            result = dup_mul(result, base, K)
        n >>= 1
        if n:
            base = dup_sqr(base, K)
    return result


def dup_div(f: list[T], g: list[T], K: Domain[T]) -> tuple[list[T], list[T]]:
    """The quotient and remainder of ``f`` by ``g`` over a field (over a
    ring, the division proceeds while the leading coefficients divide
    exactly and raises ``ValueError`` otherwise)."""
    if not g:
        raise ZeroDivisionError('polynomial division by zero')
    if K.has_dense_kernels:
        return K.dense_div(f, g)
    df, dg = len(f) - 1, len(g) - 1
    if df < dg:
        return [], list(f)
    lc = g[0]
    # over a field the leading coefficient is inverted once
    inverse = K.invert(lc) if K.is_field else K.one
    quotient: list[T] = [K.zero]*(df - dg + 1)
    remainder = list(f)
    for i in range(df - dg + 1):
        c = remainder[i]
        if K.is_zero(c):
            continue
        q = K.mul(c, inverse) if K.is_field else K.exquo(c, lc)
        quotient[i] = q
        for j in range(1, dg + 1):
            remainder[i + j] = K.sub(remainder[i + j], K.mul(q, g[j]))
        remainder[i] = K.zero
    return dup_strip(quotient, K), dup_strip(remainder[df - dg + 1:], K)


def dup_quo(f: list[T], g: list[T], K: Domain[T]) -> list[T]:
    return dup_div(f, g, K)[0]


def dup_rem(f: list[T], g: list[T], K: Domain[T]) -> list[T]:
    return dup_div(f, g, K)[1]


def dup_exquo(f: list[T], g: list[T], K: Domain[T]) -> list[T]:
    """The exact quotient; ``ValueError`` when ``g`` does not divide
    ``f``."""
    q, r = dup_div(f, g, K)
    if r:
        raise ValueError('%s does not divide %s' % (g, f))
    return q


def dup_pdiv(f: list[T], g: list[T], K: Domain[T]) -> tuple[list[T], list[T]]:
    """The pseudo-quotient and pseudo-remainder: ``lc(g)**(df - dg + 1) f
    = q g + r`` with ``deg r < deg g`` (``q = 0``, ``r = f`` when ``deg f
    < deg g``), computed without divisions in the domain."""
    if not g:
        raise ZeroDivisionError('polynomial division by zero')
    df, dg = len(f) - 1, len(g) - 1
    if df < dg:
        return [], list(f)
    lc = g[0]
    q: list[T] = []
    r = list(f)
    n = df - dg + 1
    while True:
        dr = len(r) - 1
        if dr < dg:
            break
        # r = lc r - lc(r) x**(dr - dg) g, q = lc q + lc(r) x**(dr - dg)
        lr = r[0]
        q = dup_add_term(dup_mul_ground(q, lc, K), lr, dr - dg, K)
        r = dup_sub(dup_mul_ground(r, lc, K), dup_mul_term(g, lr, dr - dg, K), K)
        n -= 1
    # the remaining factor of lc so that the multiplier is exactly lc**(df - dg + 1)
    if n > 0:
        scale = K.pow(lc, n)
        q = dup_mul_ground(q, scale, K)
        r = dup_mul_ground(r, scale, K)
    return q, r


def dup_prem(f: list[T], g: list[T], K: Domain[T]) -> list[T]:
    """The pseudo-remainder of ``f`` by ``g``."""
    return dup_pdiv(f, g, K)[1]


def dup_pquo(f: list[T], g: list[T], K: Domain[T]) -> list[T]:
    """The pseudo-quotient of ``f`` by ``g``."""
    return dup_pdiv(f, g, K)[0]


def dup_pexquo(f: list[T], g: list[T], K: Domain[T]) -> list[T]:
    """The pseudo-quotient when the pseudo-remainder is zero;
    ``ValueError`` otherwise."""
    q, r = dup_pdiv(f, g, K)
    if r:
        raise ValueError('%s does not divide %s' % (g, f))
    return q


def dup_max_norm(f: list[T], K: Domain[T]) -> T:
    """The largest absolute value of a coefficient (ordered domains)."""
    if not f:
        return K.zero
    best = K.abs(f[0])
    for c in f[1:]:
        a = K.abs(c)
        if K.is_negative(K.sub(best, a)):
            best = a
    return best


def dup_l1_norm(f: list[T], K: Domain[T]) -> T:
    """The sum of the absolute values of the coefficients."""
    total = K.zero
    for c in f:
        total = K.add(total, K.abs(c))
    return total

