"""Dense univariate polynomials over a typed domain: derivatives,
evaluation, composition, Taylor shifts, contents and primitive parts,
monic forms, truncation modulo a ground element, clearing denominators.

Examples
========

>>> from fractions import Fraction
>>> from sympy_extras.polys.domains import ZZ, QQ
>>> from sympy_extras.polys.densetools import (dup_diff, dup_eval, dup_shift, dup_compose, dup_content,
...     dup_primitive, dup_monic, dup_clear_denoms, dup_trunc, dup_integrate)
>>> dup_diff([1, 0, -2, 5], 1, ZZ)
[3, 0, -2]
>>> dup_eval([1, 0, -2], 3, ZZ)
7
>>> dup_shift([1, 0, 0], 1, ZZ)
[1, 2, 1]
>>> dup_compose([1, 0, 1], [1, 1], ZZ)
[1, 2, 2]
>>> dup_content([4, -6, 8], ZZ), dup_primitive([4, -6, 8], ZZ)
(2, (2, [2, -3, 4]))
>>> dup_monic([Fraction(2), Fraction(4)], QQ)
[Fraction(1, 1), Fraction(2, 1)]
>>> dup_clear_denoms([Fraction(1, 2), Fraction(1, 3)], QQ)
(6, [Fraction(3, 1), Fraction(2, 1)])
>>> dup_trunc([7, 8, 9], 5, ZZ)
[2, -2, -1]
>>> dup_integrate([Fraction(3), Fraction(0), Fraction(-2)], 1, QQ)
[Fraction(1, 1), Fraction(0, 1), Fraction(-2, 1), Fraction(0, 1)]

References
==========

.. [1] D. E. Knuth, *The Art of Computer Programming*, vol. 2, 3rd ed.,
       Addison-Wesley, 1998, Section 4.6.4 (evaluation, Horner's rule).
.. [2] K. O. Geddes, S. R. Czapor, G. Labahn, *Algorithms for Computer
       Algebra*, Kluwer, 1992, Section 2.6 (content and primitive part).
"""
from __future__ import annotations

from fractions import Fraction
from typing import TypeVar

from .densearith import dup_add_term, dup_exquo_ground, dup_mul, dup_mul_ground
from .densebasic import dup_LC, dup_strip

from .domains.domain import Domain
from .domains.rationalfield import RationalField

__all__ = ['dup_diff', 'dup_eval', 'dup_shift', 'dup_compose', 'dup_content', 'dup_primitive', 'dup_monic',
           'dup_clear_denoms', 'dup_trunc', 'dup_integrate', 'dup_mirror', 'dup_scale', 'dup_taylor_shift',
           'dup_decompose_content', 'dup_sign_variations']

T = TypeVar('T')


def dup_diff(f: list[T], m: int, K: Domain[T]) -> list[T]:
    """The ``m``-th derivative."""
    if m < 0:
        raise ValueError('negative order of differentiation')
    result = list(f)
    for _ in range(m):
        n = len(result) - 1
        if n <= 0:
            return []
        result = dup_strip([K.mul(c, K.from_int(n - i)) for i, c in enumerate(result[:-1])], K)
    return result


def dup_eval(f: list[T], a: T, K: Domain[T]) -> T:
    """``f(a)`` by Horner's rule."""
    result = K.zero
    for c in f:
        result = K.add(K.mul(result, a), c)
    return result


def dup_shift(f: list[T], a: T, K: Domain[T]) -> list[T]:
    """``f(x + a)`` (the Taylor shift, by Horner's rule in ``x + a``)."""
    if not f:
        return []
    result: list[T] = [f[0]]
    for c in f[1:]:
        # result = result (x + a) + c
        shifted = result + [K.zero]
        for i in range(len(result)):
            shifted[i + 1] = K.add(shifted[i + 1], K.mul(result[i], a))
        shifted[-1] = K.add(shifted[-1], c)
        result = shifted
    return dup_strip(result, K)


def dup_taylor_shift(f: list[T], a: T, K: Domain[T]) -> list[T]:
    """``f(x + a)``: the same as :func:`dup_shift`."""
    return dup_shift(f, a, K)


def dup_compose(f: list[T], g: list[T], K: Domain[T]) -> list[T]:
    """``f(g(x))`` by Horner's rule."""
    if not f:
        return []
    result: list[T] = [f[0]]
    for c in f[1:]:
        result = dup_add_term(dup_mul(result, g, K), c, 0, K)
    return result


def dup_mirror(f: list[T], K: Domain[T]) -> list[T]:
    """``f(-x)``."""
    n = len(f) - 1
    return [c if (n - i) % 2 == 0 else K.neg(c) for i, c in enumerate(f)]


def dup_scale(f: list[T], a: T, K: Domain[T]) -> list[T]:
    """``f(a x)``."""
    n = len(f) - 1
    power = K.one
    scaled = [K.zero]*(n + 1)
    for i in range(n, -1, -1):
        scaled[i] = K.mul(f[i], power)
        power = K.mul(power, a)
    return dup_strip(scaled, K)


def dup_content(f: list[T], K: Domain[T]) -> T:
    """The gcd of the coefficients (``0`` for the zero polynomial, ``1``
    for nonzero polynomials over a field)."""
    if not f:
        return K.zero
    if K.is_field:
        return K.one
    content = K.zero
    for c in f:
        content = K.gcd(content, c)
        if K.is_one(content):
            break
    return content


def dup_primitive(f: list[T], K: Domain[T]) -> tuple[T, list[T]]:
    """The content and the primitive part (``f = content * part``)."""
    if not f:
        return K.zero, []
    content = dup_content(f, K)
    if K.is_one(content):
        return content, list(f)
    return content, dup_exquo_ground(f, content, K)


def dup_decompose_content(f: list[T], K: Domain[T]) -> tuple[T, list[T]]:
    """The content with the sign of the leading coefficient and the
    primitive part with positive leading coefficient (ordered domains);
    the same as :func:`dup_primitive` otherwise."""
    content, part = dup_primitive(f, K)
    if part and K.is_ordered and K.is_negative(dup_LC(part, K)):
        return K.neg(content), [K.neg(c) for c in part]
    return content, part


def dup_monic(f: list[T], K: Domain[T]) -> list[T]:
    """``f`` divided by its leading coefficient (a field, or a ring where
    the leading coefficient divides every coefficient)."""
    if not f:
        return []
    lc = f[0]
    if K.is_one(lc):
        return list(f)
    return dup_exquo_ground(f, lc, K)


def dup_clear_denoms(f: list[Fraction], K: RationalField) -> tuple[int, list[Fraction]]:
    """The least common multiple of the denominators and ``f`` times it
    (a polynomial over ``QQ`` with integer coefficients)."""
    common = 1
    for c in f:
        d = K.denom(c)
        common = common*d//_gcd(common, d)
    if common == 1:
        return 1, list(f)
    return common, dup_mul_ground(f, K.from_int(common), K)


def dup_trunc(f: list[T], p: T, K: Domain[T]) -> list[T]:
    """The coefficients reduced modulo ``p`` (in an ordered domain into
    the symmetric range ``(-p/2, p/2]``, as the modular algorithms need;
    with the domain's ``rem`` otherwise)."""
    result: list[T] = []
    for c in f:
        r = K.rem(c, p)
        if K.is_ordered:
            if K.is_negative(r):
                r = K.add(r, p)
            # now 0 <= r < p: into (-p/2, p/2]
            if K.is_negative(K.sub(p, K.add(r, r))):
                r = K.sub(r, p)
        result.append(r)
    return dup_strip(result, K)


def dup_integrate(f: list[T], m: int, K: Domain[T]) -> list[T]:
    """The ``m``-th antiderivative with zero constants (a field, or a
    ring where the divisions are exact)."""
    if m < 0:
        raise ValueError('negative order of integration')
    result = list(f)
    for _ in range(m):
        if not result:
            continue
        n = len(result)
        result = [K.exquo(c, K.from_int(n - i)) for i, c in enumerate(result)] + [K.zero]
    return dup_strip(result, K)


def dup_sign_variations(f: list[T], K: Domain[T]) -> int:
    """The number of sign changes in the sequence of coefficients
    (ordered domains; zeros skipped)."""
    count = 0
    previous = 0
    for c in f:
        s = K.sign(c)
        if s and previous and s != previous:
            count += 1
        if s:
            previous = s
    return count


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a

