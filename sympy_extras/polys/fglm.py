r"""Change of monomial order for zero-dimensional ideals: the FGLM
algorithm on the typed polynomial rings.

A lexicographic Gröbner basis is what solving needs and the most
expensive to compute directly; a basis for a graded order is cheap. For
an ideal $I$ of dimension zero the quotient $K[x]/I$ is a vector space of
finite dimension $D$, with the monomials under the staircase of the known
basis as a basis, and the change of order is linear algebra there
(Faugère, Gianni, Lazard, Mora): the monomials are visited in increasing
order for the *new* order, starting from 1; the normal form (for the old
basis) of each is computed from that of the monomial it is a multiple of;
if it is a linear combination of the normal forms of the monomials kept
so far, the relation is an element of the new basis with that monomial as
its leading monomial; otherwise the monomial is kept, and its multiples
by the variables become candidates. Multiples of the leading monomials
found are skipped. The $D$ monomials kept are the new staircase, and the
relations are the reduced Gröbner basis for the new order.

:func:`groebner_by_fglm` computes a basis for any order this way: F4 for
``grevlex``, then the conversion.

Examples
========

>>> from sympy_extras.polys.domains import QQ
>>> from sympy_extras.polys.rings import PolyRing, grevlex, lex
>>> from sympy_extras.polys.f4 import f4
>>> from sympy_extras.polys.fglm import fglm, groebner_by_fglm
>>> R = PolyRing(QQ, ['x', 'y'], grevlex)
>>> x, y = R.gens
>>> G = f4([x*x + y*y - R.from_int(5), x*y - R.from_int(2)], R)
>>> [str(g) for g in G]
['y**3 + 2*x - 5*y', 'x**2 + y**2 - 5', 'x*y - 2']
>>> [str(g) for g in fglm(G, R, lex)]
['x + 1/2*y**3 - 5/2*y', 'y**4 - 5*y**2 + 4']
>>> L = PolyRing(QQ, ['x', 'y'], lex)
>>> X, Y = L.gens
>>> groebner_by_fglm([X*X + Y*Y - L.from_int(5), X*Y - L.from_int(2)], L) == f4(
...     [X*X + Y*Y - L.from_int(5), X*Y - L.from_int(2)], L)
True

References
==========

.. [1] J.-C. Faugère, P. Gianni, D. Lazard, T. Mora, *Efficient
       computation of zero-dimensional Gröbner bases by change of
       ordering*, Journal of Symbolic Computation 16 (1993), 329-344.
.. [2] D. A. Cox, J. Little, D. O'Shea, *Using Algebraic Geometry*, 2nd
       ed., Springer, 2005, Section 2.3.
"""
from __future__ import annotations

from typing import Generic, Sequence, TypeVar

from .f4 import f4
from .rings import Monomial, MonomialOrder, PolyElement, PolyRing, grevlex, monomial_divides

__all__ = ['fglm', 'groebner_by_fglm', 'is_zero_dimensional']

T = TypeVar('T')


def is_zero_dimensional(basis: Sequence[PolyElement[T]], ring: PolyRing[T]) -> bool:
    """Whether the ideal of a Gröbner basis has dimension zero (finitely
    many solutions over the algebraic closure): a power of every variable
    is a leading monomial, or the ideal is the whole ring.

    Examples
    ========

    >>> from sympy_extras.polys.domains import QQ
    >>> from sympy_extras.polys.rings import PolyRing, lex
    >>> from sympy_extras.polys.fglm import is_zero_dimensional
    >>> R = PolyRing(QQ, ['x', 'y'], lex)
    >>> x, y = R.gens
    >>> is_zero_dimensional([x*x - y, y*y*y - R.one], R), is_zero_dimensional([x*y], R)
    (True, False)
    """
    pure: set[int] = set()
    for g in basis:
        if not g.terms:
            continue
        leading = g.LM
        degree = sum(leading)
        if degree == 0:
            return True
        for i, e in enumerate(leading):
            if e == degree:
                pure.add(i)
    return len(pure) == len(ring.names)


class _Row(Generic[T]):
    """A row of the echelon form: a normal form with coefficient one at
    its ``pivot`` monomial, and its expression ``sum combination[j]
    NF(kept[j])`` in the normal forms of the monomials kept."""

    def __init__(self, pivot: Monomial, vector: dict[Monomial, T], combination: dict[int, T]) -> None:
        self.pivot = pivot
        self.vector = vector
        self.combination = combination


def fglm(basis: Sequence[PolyElement[T]], ring: PolyRing[T], order: MonomialOrder) -> list[PolyElement[T]]:
    """The reduced Gröbner basis for ``order`` of the zero-dimensional
    ideal whose reduced Gröbner basis for the order of ``ring`` is
    ``basis`` (over a field); in the ring with the new order, sorted by
    decreasing leading monomial.

    Raises ``ValueError`` when the ideal is not zero-dimensional.

    Examples
    ========

    >>> from sympy_extras.polys.domains import GF
    >>> from sympy_extras.polys.rings import PolyRing, grevlex, lex
    >>> from sympy_extras.polys.fglm import fglm
    >>> R = PolyRing(GF(7), ['x', 'y'], grevlex)
    >>> x, y = R.gens
    >>> [str(g) for g in fglm([y*y - x, x*x - R.one], R, lex)]
    ['x + 6*y**2', 'y**4 + 6']
    """
    K = ring.domain
    target = ring.with_order(order)
    generators = [g for g in basis if g.terms]
    if not is_zero_dimensional(generators, ring):
        raise ValueError('the ideal is not zero-dimensional')
    n = len(ring.names)
    units = [tuple(1 if k == i else 0 for k in range(n)) for i in range(n)]
    origin: Monomial = tuple(0 for _ in range(n))
    kept: list[Monomial] = []
    normal_forms: dict[Monomial, PolyElement[T]] = {}
    rows: list[_Row[T]] = []
    leading: list[Monomial] = []
    result: list[PolyElement[T]] = []
    # the candidates, with the normal form each one is computed from
    candidates: dict[Monomial, PolyElement[T]] = {origin: ring.one.rem(generators)}
    while candidates:
        monomial = min(candidates, key=order)
        normal_form = candidates.pop(monomial)
        if any(monomial_divides(lm, monomial) for lm in leading):
            continue
        vector = dict(normal_form.terms)
        combination: dict[int, T] = {}
        for row in rows:
            c = vector.get(row.pivot)
            if c is None:
                continue
            for m, v in row.vector.items():
                value = K.sub(vector.get(m, K.zero), K.mul(c, v))
                if K.is_zero(value):
                    vector.pop(m, None)
                else:
                    vector[m] = value
            for j, v in row.combination.items():
                value = K.add(combination.get(j, K.zero), K.mul(c, v))
                if K.is_zero(value):
                    combination.pop(j, None)
                else:
                    combination[j] = value
        if not vector:
            # NF(monomial) = sum combination[j] NF(kept[j]): a basis element
            terms: dict[Monomial, T] = {monomial: K.one}
            for j, v in combination.items():
                terms[kept[j]] = K.neg(v)
            result.append(PolyElement(terms, target))
            leading.append(monomial)
            continue
        index = len(kept)
        kept.append(monomial)
        normal_forms[monomial] = normal_form
        pivot = next(iter(vector))
        inverse = K.invert(vector[pivot])
        scaled = {j: K.neg(K.mul(v, inverse)) for j, v in combination.items()}
        scaled[index] = inverse
        rows.append(_Row(pivot, {m: K.mul(v, inverse) for m, v in vector.items()}, scaled))
        for unit in units:
            multiple: Monomial = tuple(a + e for a, e in zip(monomial, unit))
            if multiple not in candidates and not any(monomial_divides(lm, multiple) for lm in leading):
                candidates[multiple] = normal_form.mul_monom(unit).rem(generators)
    result.sort(key=lambda g: order(g.LM), reverse=True)
    return result


def groebner_by_fglm(polys: Sequence[PolyElement[T]], ring: PolyRing[T]) -> list[PolyElement[T]]:
    """The reduced Gröbner basis for the order of ``ring`` of a
    zero-dimensional ideal over a field: F4 for ``grevlex``, then the
    change of order. Raises ``ValueError`` when the ideal is not
    zero-dimensional.

    Examples
    ========

    >>> from sympy_extras.polys.domains import QQ
    >>> from sympy_extras.polys.rings import PolyRing, lex
    >>> from sympy_extras.polys.fglm import groebner_by_fglm
    >>> R = PolyRing(QQ, ['x', 'y', 'z'], lex)
    >>> x, y, z = R.gens
    >>> one = R.one
    >>> [str(g) for g in groebner_by_fglm([x + y + z - one, x*x + y*y + z*z - one, x*x*x + y*y*y + z*z*z - one], R)]
    ['x + y + z - 1', 'y**2 + y*z - y + z**2 - z', 'z**3 - z**2']
    """
    graded = ring.with_order(grevlex)
    converted = [PolyElement(dict(f.terms), graded) for f in polys]
    return fglm(f4(converted, graded), graded, ring.order)
