"""Principal subresultant coefficients.

These functions extend :mod:`sympy.polys.euclidtools` with the principal
subresultant coefficients of two polynomials, which are needed by Hong's
projection operator for cylindrical algebraic decomposition. They work on
the dense representation used by the low level SymPy polynomial routines
(``dup_*`` for univariate and ``dmp_*`` for multivariate polynomials);
:func:`psc` is a convenience wrapper for elements of a
:class:`~sympy.polys.rings.PolyRing`.
"""
from __future__ import annotations

from sympy.polys.densebasic import (dmp_degree, dmp_zero, dmp_zero_p,
    dup_degree)
from sympy.polys.domains.domain import Domain
from sympy.polys.euclidtools import dmp_inner_subresultants, dup_inner_subresultants
from sympy.polys.rings import PolyElement

from sympy_extras._typing import Dmp, DomainElement, Dup

__all__ = ["dup_psc", "dmp_psc", "psc"]


def dup_psc(f: Dup, g: Dup, K: Domain) -> list[DomainElement]:
    r"""
    Principal subresultant coefficients of two polynomials in `K[x]`.

    Returns the list ``[psc_0, psc_1, ..., psc_m]`` where ``m`` is the
    smaller of the degrees of ``f`` and ``g``. ``psc_j`` is the leading
    coefficient of the `j`-th subresultant of ``f`` and ``g``, i.e. the
    coefficient of `x^j` in `S_j`; it is zero if `S_j` is defective. In
    particular ``psc_0`` is the resultant and ``psc_j`` vanishes for
    `j < \deg\gcd(f, g)`.

    If `\deg f < \deg g` the coefficients of `(g, f)` are computed. The
    list is empty if one of the polynomials is zero.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> from sympy_extras.polys.euclidtools import dup_psc
    >>> R, x = ring("x", ZZ)

    >>> dup_psc((x**2 + 1).to_dense(), (x**2 - 1).to_dense(), ZZ)
    [4, 0, 1]
    >>> dup_psc((x**3 - x).to_dense(), (x**2 - 1).to_dense(), ZZ)
    [0, 0, 1]

    """
    if not f or not g:
        return []

    R, S = dup_inner_subresultants(f, g, K)

    if not R:
        return []

    m = min(dup_degree(f), dup_degree(g))
    psc = [K.zero]*(m + 1)

    for r, s in zip(R[1:], S[1:]):
        psc[dup_degree(r)] = s

    return psc


def dmp_psc(f: Dmp, g: Dmp, u: int, K: Domain) -> list[Dmp]:
    """
    Principal subresultant coefficients of two polynomials in `K[X]`.

    The coefficients are taken with respect to the main variable, see
    :func:`dup_psc`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> from sympy_extras.polys.euclidtools import dmp_psc
    >>> R, x, y = ring("x,y", ZZ)

    >>> dmp_psc((x**2*y + x).to_dense(), (x + y).to_dense(), 1, ZZ)
    [[1, 0, -1, 0], [1]]

    """
    if not u:
        return dup_psc(f, g, K)

    if dmp_zero_p(f, u) or dmp_zero_p(g, u):
        return []

    R, S = dmp_inner_subresultants(f, g, u, K)

    if not R:
        return []

    m = min(dmp_degree(f, u), dmp_degree(g, u))
    psc = [dmp_zero(u - 1)]*(m + 1)

    for r, s in zip(R[1:], S[1:]):
        psc[dmp_degree(r, u)] = s

    return psc


def psc(f: PolyElement | DomainElement, g: PolyElement | DomainElement) -> list[PolyElement | DomainElement]:
    """
    Principal subresultant coefficients of two elements of a polynomial ring
    with respect to the first generator.

    For a univariate ring the coefficients are elements of the ground
    domain; otherwise they are elements of the ring without its first
    generator.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> from sympy_extras.polys.euclidtools import psc
    >>> R, x = ring("x", ZZ)
    >>> psc(x**2 + 1, x**2 - 1)
    [4, 0, 1]

    >>> R, x, y = ring("x,y", ZZ)
    >>> psc(x**2*y + x, x + y)
    [y**3 - y, 1]

    """
    if hasattr(f, "ring"):
        R = f.ring
    elif hasattr(g, "ring"):
        R = g.ring
    else:
        raise TypeError("at least one argument must be a PolyElement")
    f, g = R(f), R(g)
    if R.ngens == 1:
        return dup_psc(f.to_dense(), g.to_dense(), R.domain)
    coeffs = dmp_psc(f.to_dense(), g.to_dense(), R.ngens - 1, R.domain)
    return [R[1:].from_dense(c) for c in coeffs]
