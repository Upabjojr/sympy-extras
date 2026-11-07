"""Square-free decomposition of dense univariate polynomials.

Over a domain of characteristic zero, Yun's algorithm: with ``g = gcd(f,
f')``, ``w = f/g`` and ``y = f'/g``, the factor of multiplicity ``i`` is
``gcd(w, y - w')`` at step ``i``, with ``w`` and ``y`` replaced by their
quotients. Over ``GF(p)`` the derivative may vanish (``f`` is then a
polynomial in ``x**p``, whose ``p``-th root is taken by dropping the
exponents), so the factors of multiplicities divisible by ``p`` are found
recursively.

Examples
========

>>> from sympy_extras.polys.domains import ZZ, GF
>>> from sympy_extras.polys.sqfreetools import dup_sqf_list, dup_sqf_part, dup_sqf_p
>>> dup_sqf_list([1, -4, 6, -4, 1], ZZ)
(1, [([1, -1], 4)])
>>> dup_sqf_list([2, 0, -4, 0, 2], ZZ)
(2, [([1, 0, -1], 2)])
>>> dup_sqf_part([1, -4, 6, -4, 1], ZZ), dup_sqf_p([1, 0, -1], ZZ), dup_sqf_p([1, 0, 0], ZZ)
([1, -1], True, False)
>>> dup_sqf_list([1, 0, 0, 0, 0, 0, 1], GF(3))
(1, [([1, 0, 1], 3)])

References
==========

.. [1] D. Y. Y. Yun, *On square-free decomposition algorithms*, SYMSAC
       '76, 26-35.
.. [2] K. O. Geddes, S. R. Czapor, G. Labahn, *Algorithms for Computer
       Algebra*, Kluwer, 1992, Section 8.2 (square-free factorization).
"""
from __future__ import annotations

from typing import TypeVar

from .densearith import dup_exquo, dup_mul, dup_mul_ground, dup_sub
from .densebasic import dup_LC, dup_degree
from .densetools import dup_decompose_content, dup_diff, dup_monic, dup_primitive
from .euclidtools import dup_gcd

from .domains.domain import Domain

__all__ = ['dup_sqf_list', 'dup_sqf_part', 'dup_sqf_p', 'dup_sqf_norm_content']

T = TypeVar('T')


def dup_sqf_norm_content(f: list[T], K: Domain[T]) -> tuple[T, list[T]]:
    """The content (with the sign of the leading coefficient in an ordered
    domain, the leading coefficient over a field) and the primitive, or
    monic, part."""
    if not f:
        return K.zero, []
    if K.is_field:
        lc = dup_LC(f, K)
        return lc, dup_monic(f, K)
    return dup_decompose_content(f, K)


def dup_sqf_list(f: list[T], K: Domain[T]) -> tuple[T, list[tuple[list[T], int]]]:
    """The content and the square-free factors with their multiplicities,
    in increasing order of multiplicity: ``f = content * prod g_i**i``,
    each ``g_i`` primitive (monic over a field) and square-free, pairwise
    coprime."""
    content, part = dup_sqf_norm_content(f, K)
    if dup_degree(part) <= 0:
        return content, []
    if K.characteristic == 0:
        return content, _yun(part, K)
    return content, _sqf_list_positive_characteristic(part, K, K.characteristic)


def _yun(f: list[T], K: Domain[T]) -> list[tuple[list[T], int]]:
    derivative = dup_diff(f, 1, K)
    g = dup_gcd(f, derivative, K)
    if dup_degree(g) <= 0:
        return [(f, 1)]
    w = dup_exquo(f, g, K)
    y = dup_exquo(derivative, g, K)
    factors: list[tuple[list[T], int]] = []
    i = 1
    while dup_degree(w) > 0:
        z = dup_sub(y, dup_diff(w, 1, K), K)
        h = dup_gcd(w, z, K)
        if dup_degree(h) > 0:
            factors.append((h, i))
            w = dup_exquo(w, h, K)
            y = dup_exquo(z, h, K)
        else:
            y = z
        i += 1
    return factors


def _pth_root(f: list[T], p: int, K: Domain[T]) -> list[T]:
    """``g`` with ``g(x)**p = f(x)`` for ``f`` a polynomial in ``x**p``
    over a prime field (the coefficients are their own ``p``-th roots)."""
    n = dup_degree(f)
    if n % p:
        raise ValueError('%s is not a polynomial in x**%d' % (f, p))
    return [f[i] for i in range(0, n + 1, p)]


def _sqf_list_positive_characteristic(f: list[T], K: Domain[T], p: int) -> list[tuple[list[T], int]]:
    derivative = dup_diff(f, 1, K)
    factors: list[tuple[list[T], int]] = []
    if not derivative:
        # f = g(x**p) = g(x)**p over GF(p)
        root = _pth_root(f, p, K)
        for h, i in _sqf_list_positive_characteristic(root, K, p):
            factors.append((h, i*p))
        return factors
    g = dup_gcd(f, derivative, K)
    w = dup_exquo(f, g, K)
    # w is the product of the factors with multiplicity prime to p, each once
    i = 1
    while dup_degree(w) > 0:
        g_w = dup_gcd(g, w, K)
        h = dup_exquo(w, g_w, K)
        if dup_degree(h) > 0:
            factors.append((h, i))
        w = g_w
        g = dup_exquo(g, g_w, K)
        i += 1
    if dup_degree(g) > 0:
        # the remaining factors have multiplicities divisible by p
        root = _pth_root(g, p, K)
        for h, j in _sqf_list_positive_characteristic(root, K, p):
            factors.append((h, j*p))
    factors.sort(key=lambda item: item[1])
    return factors


def dup_sqf_part(f: list[T], K: Domain[T]) -> list[T]:
    """The product of the square-free factors (primitive over a ring,
    monic over a field)."""
    content, factors = dup_sqf_list(f, K)
    if not f:
        return []
    result: list[T] = [K.one]
    for g, _ in factors:
        result = dup_mul(result, g, K)
    if K.is_field:
        return dup_monic(result, K)
    _, part = dup_primitive(result, K)
    return part if not K.is_ordered or not K.is_negative(dup_LC(part, K)) else dup_mul_ground(part, K.neg(K.one), K)


def dup_sqf_p(f: list[T], K: Domain[T]) -> bool:
    """Whether ``f`` is square-free (the zero polynomial is not)."""
    if not f:
        return False
    if dup_degree(f) <= 0:
        return True
    return dup_degree(dup_gcd(f, dup_diff(f, 1, K), K)) <= 0
