r"""Principal subresultant coefficients on the typed polynomial layer (no
SymPy), as Hong's projection operator for cylindrical algebraic
decomposition needs them: :func:`dup_psc` on dense polynomials over a
``Domain[T]`` (for several variables, over a
:class:`~sympy_extras.polys.domains.polynomialring.PolynomialRing` or a
:class:`~sympy_extras.polys.domains.sparsepolynomialring.SparsePolynomialRing`
of the other variables), and :func:`psc` on the sparse polynomials of
:class:`~sympy_extras.polys.rings.PolyRing` with respect to their first
generator. The subresultant PRS behind them is
:func:`sympy_extras.polys.euclidtools.dup_inner_subresultants` (exact
divisions only).

The $j$-th principal subresultant coefficient $\mathrm{psc}_j$ of $f$ and
$g$ is the coefficient of $x^j$ in the $j$-th subresultant $S_j$ (the
determinant of a submatrix of the Sylvester matrix), zero when $S_j$ is
defective; $\mathrm{psc}_0$ is the resultant and $\mathrm{psc}_j$ vanishes
for $j < \deg\gcd(f, g)$ [1]_.

:mod:`sympy_extras.polys.subresultants_sympy` is the SymPy face of this
module (SymPy's dense representation and ring elements).

Examples
========

>>> from sympy_extras.polys.domains import ZZ
>>> from sympy_extras.polys.subresultants import dup_psc
>>> dup_psc([1, 0, 1], [1, 0, -1], ZZ)
[4, 0, 1]

References
==========

.. [1] B. Mishra, *Algorithmic Algebra*, Springer, 1993, Chapter 7
       (subresultants and their principal coefficients).
"""
from __future__ import annotations

from typing import TypeVar

from .densebasic import dup_degree
from .domains.domain import Domain
from .domains.sparsepolynomialring import SparsePolynomialRing
from .euclidtools import dup_inner_subresultants
from .orderings import Monomial
from .rings import PolyElement

__all__ = ['dup_psc', 'psc']

T = TypeVar('T')


def dup_psc(f: list[T], g: list[T], K: Domain[T]) -> list[T]:
    r"""Principal subresultant coefficients of two polynomials in `K[x]`.

    Returns the list ``[psc_0, psc_1, ..., psc_m]`` where ``m`` is the
    smaller of the degrees of ``f`` and ``g``: ``psc_j`` is the leading
    coefficient of the `j`-th subresultant of ``f`` and ``g``, i.e. the
    coefficient of `x^j` in `S_j`, zero if `S_j` is defective. If
    `\deg f < \deg g` the coefficients of `(g, f)` are computed. The list
    is empty if one of the polynomials is zero.

    Examples
    ========

    >>> from sympy_extras.polys.domains import ZZ
    >>> from sympy_extras.polys.domains.polynomialring import PolynomialRing
    >>> from sympy_extras.polys.subresultants import dup_psc
    >>> dup_psc([1, 0, 1], [1, 0, -1], ZZ), dup_psc([1, 0, -1, 0], [1, 0, -1], ZZ)
    ([4, 0, 1], [0, 0, 1])
    >>> dup_psc([], [1], ZZ)
    []

    Several variables: ``x**2*y + x`` and ``x + y`` over ``ZZ[y]``.

    >>> dup_psc([[1, 0], [1], []], [[1], [1, 0]], PolynomialRing(ZZ, 'y'))
    [[1, 0, -1, 0], [1]]
    """
    if not f or not g:
        return []
    R, S = dup_inner_subresultants(f, g, K)
    if not R:
        return []
    m = min(dup_degree(f), dup_degree(g))
    coefficients = [K.zero]*(m + 1)
    for r, s in zip(R[1:], S[1:]):
        coefficients[dup_degree(r)] = s
    return coefficients


def psc(f: PolyElement[T], g: PolyElement[T]) -> list[PolyElement[T]]:
    """Principal subresultant coefficients of two elements of a polynomial
    ring with respect to its first generator (:func:`dup_psc`), as elements
    of the same ring which do not involve that generator (constants of the
    ring for a univariate one).

    Examples
    ========

    >>> from sympy_extras.polys.domains import ZZ
    >>> from sympy_extras.polys.rings import PolyRing
    >>> from sympy_extras.polys.subresultants import psc
    >>> R = PolyRing(ZZ, ['x'])
    >>> x, = R.gens
    >>> [str(c) for c in psc(x**2 + R.one, x**2 - R.one)]
    ['4', '0', '1']
    >>> R = PolyRing(ZZ, ['x', 'y'])
    >>> x, y = R.gens
    >>> [str(c) for c in psc(x**2*y + x, x + y)]
    ['y**3 - y', '1']
    """
    ring = f.ring
    if g.ring != ring:
        raise ValueError('the polynomials are in different rings: %s and %s' % (ring, g.ring))
    K = ring.domain
    if ring.ngens == 1:
        def dense(h: PolyElement[T]) -> list[T]:
            if not h.terms:
                return []
            n = max(m[0] for m in h.terms)
            rep = [K.zero]*(n + 1)
            for m, c in h.terms.items():
                rep[n - m[0]] = c
            return rep
        return [ring.ground(c) for c in dup_psc(dense(f), dense(g), K)]
    inner = SparsePolynomialRing(K, ring.names[1:])

    def nested(h: PolyElement[T]) -> list[PolyElement[T]]:
        if not h.terms:
            return []
        n = max(m[0] for m in h.terms)
        by_degree: list[dict[Monomial, T]] = [{} for _ in range(n + 1)]
        for m, c in h.terms.items():
            by_degree[n - m[0]][m[1:]] = c
        return [inner.ring.from_dict(terms) for terms in by_degree]

    return [ring.from_dict({(0,) + m: c for m, c in s.terms.items()}) for s in dup_psc(nested(f), nested(g), inner)]
