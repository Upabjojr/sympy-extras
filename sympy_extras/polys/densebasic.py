"""Dense univariate polynomials over a typed domain: the representation.

A polynomial in ``K[x]`` is a Python list of coefficients, highest degree
first (``[1, 0, -2]`` is ``x**2 - 2``), without leading zeros; the zero
polynomial is the empty list. A polynomial in several variables is a
polynomial over a polynomial ring (``K[y][x]``: a list of lists), which
the generic algorithms handle through the domain of the coefficients
(:mod:`sympy_extras.polys.domains.polynomialring`), so every function
here is written once for a type variable ``T`` and a ``Domain[T]``.

Examples
========

>>> from sympy_extras.polys.domains import ZZ, QQ
>>> from sympy_extras.polys.densebasic import dup_strip, dup_degree, dup_from_dict, dup_to_dict, dup_convert
>>> dup_strip([0, 0, 1, 2], ZZ)
[1, 2]
>>> dup_degree([1, 0, -2]), dup_degree([])
(2, -1)
>>> dup_from_dict({3: 1, 0: -1}, ZZ)
[1, 0, 0, -1]
>>> dup_to_dict([1, 0, 0, -1], ZZ)
{3: 1, 0: -1}
>>> dup_convert([1, -2], ZZ, QQ)
[Fraction(1, 1), Fraction(-2, 1)]
"""
from __future__ import annotations

from typing import Sequence, TypeVar


from .domains.domain import Domain

__all__ = ['dup_strip', 'dup_degree', 'dup_LC', 'dup_TC', 'dup_normal', 'dup_from_dict', 'dup_to_dict',
           'dup_from_list', 'dup_convert', 'dup_zero', 'dup_one', 'dup_constant', 'dup_is_zero', 'dup_is_one',
           'dup_is_constant', 'dup_monomial', 'dup_reverse', 'dup_copy', 'dup_slice', 'dup_terms']

T = TypeVar('T')
U = TypeVar('U')


def dup_strip(f: list[T], K: Domain[T]) -> list[T]:
    """``f`` without leading zeros (the zero polynomial is ``[]``)."""
    i = 0
    n = len(f)
    while i < n and K.is_zero(f[i]):
        i += 1
    return f[i:]


def dup_zero(K: Domain[T]) -> list[T]:
    return []


def dup_one(K: Domain[T]) -> list[T]:
    return [K.one]


def dup_constant(c: T, K: Domain[T]) -> list[T]:
    return [] if K.is_zero(c) else [c]


def dup_monomial(c: T, n: int, K: Domain[T]) -> list[T]:
    """``c x**n``."""
    if K.is_zero(c):
        return []
    return [c] + [K.zero]*n


def dup_is_zero(f: list[T]) -> bool:
    return not f


def dup_is_one(f: list[T], K: Domain[T]) -> bool:
    return len(f) == 1 and K.is_one(f[0])


def dup_is_constant(f: list[T]) -> bool:
    return len(f) <= 1


def dup_degree(f: list[T]) -> int:
    """The degree (``-1`` for the zero polynomial)."""
    return len(f) - 1


def dup_LC(f: list[T], K: Domain[T]) -> T:
    """The leading coefficient (``0`` for the zero polynomial)."""
    return f[0] if f else K.zero


def dup_TC(f: list[T], K: Domain[T]) -> T:
    """The trailing (constant) coefficient."""
    return f[-1] if f else K.zero


def dup_normal(f: Sequence[T], K: Domain[T]) -> list[T]:
    """A list of coefficients (highest degree first) as a polynomial:
    leading zeros removed."""
    return dup_strip(list(f), K)


def dup_from_list(f: Sequence[int], K: Domain[T]) -> list[T]:
    """A list of integers (highest degree first) as a polynomial over
    ``K``."""
    return dup_strip([K.from_int(c) for c in f], K)


def dup_from_dict(f: dict[int, T], K: Domain[T]) -> list[T]:
    """``{exponent: coefficient}`` as a polynomial."""
    if not f:
        return []
    n = max(f)
    result = [K.zero]*(n + 1)
    for k, c in f.items():
        result[n - k] = c
    return dup_strip(result, K)


def dup_to_dict(f: list[T], K: Domain[T]) -> dict[int, T]:
    """The nonzero terms as ``{exponent: coefficient}``."""
    n = len(f) - 1
    return {n - i: c for i, c in enumerate(f) if not K.is_zero(c)}


def dup_terms(f: list[T], K: Domain[T]) -> list[tuple[int, T]]:
    """The nonzero terms ``(exponent, coefficient)``, highest first."""
    n = len(f) - 1
    return [(n - i, c) for i, c in enumerate(f) if not K.is_zero(c)]


def dup_convert(f: list[T], K0: Domain[T], K1: Domain[U]) -> list[U]:
    """The polynomial over ``K0`` as a polynomial over ``K1``
    (``ValueError`` when a coefficient has no image)."""
    return dup_strip([K1.convert(c, K0) for c in f], K1)


def dup_reverse(f: list[T], K: Domain[T]) -> list[T]:
    """The reciprocal polynomial ``x**deg(f) f(1/x)``."""
    return dup_strip(list(reversed(f)), K)


def dup_copy(f: list[T]) -> list[T]:
    return list(f)


def dup_slice(f: list[T], m: int, n: int, K: Domain[T]) -> list[T]:
    """The terms of degree ``m <= k < n``."""
    terms = dup_to_dict(f, K)
    return dup_from_dict({k: c for k, c in terms.items() if m <= k < n}, K)
