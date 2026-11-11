r"""The Gröbner walk: conversion of Gröbner bases between monomial orders,
on the typed rings of :mod:`sympy_extras.polys.rings` over a field (no
SymPy).

The FGLM algorithm (:mod:`sympy_extras.polys.fglm`) converts the Gröbner
bases of zero-dimensional ideals only. The Gröbner walk of Collart,
Kalkbrener and Mall [1]_ works for every ideal: the source and target
orders are represented by weight vectors $\omega$ and $\tau$, and the basis
is transformed along the segment from $\omega$ to $\tau$. At every point
where the ordering of the monomials of the basis changes, the *initial
forms* of the basis elements (their terms of maximal weight) are computed,
a Gröbner basis of the ideal they generate is computed, and it is lifted
to a Gröbner basis of the whole ideal for the new weight. The initial
forms are usually much smaller than the polynomials, so each step is
cheap.

The default lifting is the one of [2]_: if $G$ is the basis for the
previous order and $\omega$ lies in the closure of its Gröbner cone, then
for every element $h$ of a Gröbner basis of the initial ideal the
polynomial $h - \mathrm{NF}_G(h)$ (normal form for the previous order)
belongs to the ideal and has initial form $h$; the bases of the initial
ideals are computed by F4 (:mod:`sympy_extras.polys.f4`). The lifting
through the representations of the basis of the initial ideal in terms of
the initial forms, computed by :func:`extended_groebner` (a Buchberger
algorithm keeping the representations), is the alternative
``lift='cofactors'``.

The weight orders along the walk are
:class:`~sympy_extras.polys.orderings.WeightOrder` instances: a weight
vector refined by the target order. :mod:`sympy_extras.polys.groebnerwalk_sympy`
is the SymPy face of this module (SymPy's rings and orders);
:meth:`sympy_extras.polys.ideals.Ideal.change_order` walks with
:func:`groebner_walk` when FGLM does not apply.

Examples
========

>>> from sympy_extras.polys.domains import QQ
>>> from sympy_extras.polys.rings import PolyRing, grevlex, lex
>>> from sympy_extras.polys.f4 import f4
>>> from sympy_extras.polys.groebnerwalk import groebner_walk
>>> R = PolyRing(QQ, ['x', 'y', 'z'], grevlex)
>>> x, y, z = R.gens
>>> G = f4([x*z - y**2, x**2 - y*z], R)
>>> [str(g) for g in groebner_walk(G, R, lex)]
['x**2 - y*z', 'x*y**2 - y*z**2', 'x*z - y**2', 'y**4 - y*z**3']

References
==========

.. [1] S. Collart, M. Kalkbrener, D. Mall, Converting bases with the
       Gröbner walk, J. Symbolic Computation 24 (1997) 465-469.
.. [2] B. Amrhein, O. Gloor, W. Küchlin, On the walk, Theoretical Computer
       Science 187 (1997) 179-202.
"""
from __future__ import annotations

from fractions import Fraction
from math import lcm
from typing import Optional, Sequence, TypeVar, Union

from .f4 import f4, reduced_basis
from .orderings import Monomial, MonomialOrder, OrderValue, WeightOrder, as_order, grevlex, grlex, lex
from .rings import PolyElement, PolyRing, monomial_div, monomial_divides, monomial_lcm, monomial_mul

__all__ = ['groebner_walk', 'initial_form', 'extended_groebner', 'is_groebner', 'normal_form']

T = TypeVar('T')

#: the terms of a polynomial
Terms = dict[Monomial, T]


def normal_form(f: PolyElement[T], basis: Sequence[PolyElement[T]]) -> PolyElement[T]:
    """The remainder of the division of ``f`` by ``basis`` (over a field,
    for the order of the ring of ``f``): no term is divisible by a leading
    monomial of the basis; the normal form when ``basis`` is a Gröbner
    basis.

    >>> from sympy_extras.polys.domains import QQ
    >>> from sympy_extras.polys.rings import PolyRing, grevlex
    >>> from sympy_extras.polys.groebnerwalk import normal_form
    >>> R = PolyRing(QQ, ['x', 'y'], grevlex)
    >>> x, y = R.gens
    >>> print(normal_form(x**3 + y, [x**2 - y]))
    x*y + y
    """
    ring = f.ring
    K = ring.domain
    key = ring.order.key
    divisors: list[tuple[Monomial, T, list[tuple[Monomial, T]]]] = []
    for g in basis:
        if g.terms:
            lm = max(g.terms, key=key)
            divisors.append((lm, K.invert(g.terms[lm]), list(g.terms.items())))
    p = dict(f.terms)
    remainder: Terms[T] = {}
    while p:
        m = max(p, key=key)
        c = p[m]
        for lm, inverse, terms in divisors:
            if monomial_divides(lm, m):
                q = monomial_div(m, lm)
                factor = K.mul(c, inverse)
                for gm, gc in terms:
                    t = monomial_mul(gm, q)
                    old = p.get(t)
                    value = K.neg(K.mul(factor, gc)) if old is None else K.sub(old, K.mul(factor, gc))
                    if K.is_zero(value):
                        if old is not None:
                            del p[t]
                    else:
                        p[t] = value
                break
        else:
            remainder[m] = c
            del p[m]
    return PolyElement(remainder, ring)


def initial_form(f: PolyElement[T], weights: Sequence[OrderValue]) -> PolyElement[T]:
    """The terms of ``f`` of maximal weight for the weight vector.

    >>> from fractions import Fraction
    >>> from sympy_extras.polys.domains import QQ
    >>> from sympy_extras.polys.rings import PolyRing
    >>> from sympy_extras.polys.groebnerwalk import initial_form
    >>> R = PolyRing(QQ, ['x', 'y'])
    >>> x, y = R.gens
    >>> print(initial_form(x**2 + x*y + y**3, [Fraction(1), Fraction(1)]))
    y**3
    >>> print(initial_form(x**2*y + x*y**2 + x, [1, 2]))
    x*y**2
    """
    best: Optional[Fraction] = None
    terms: Terms[T] = {}
    for m, c in f.terms.items():
        w = sum((Fraction(a)*b for a, b in zip(weights, m)), Fraction(0))
        if best is None or w > best:
            best, terms = w, {m: c}
        elif w == best:
            terms[m] = c
    return PolyElement(terms, f.ring)


def _s_polynomial(f: PolyElement[T], g: PolyElement[T]) -> tuple[PolyElement[T], Monomial, T, Monomial, T]:
    """The S-polynomial ``mf*f - mg*g`` of ``f`` and ``g`` (the leading
    terms cancelled), with the monomials and coefficients of ``mf`` and
    ``mg``."""
    K = f.ring.domain
    L = monomial_lcm(f.LM, g.LM)
    uf, cf = monomial_div(L, f.LM), K.invert(f.LC)
    ug, cg = monomial_div(L, g.LM), K.invert(g.LC)
    return f.mul_term(uf, cf) - g.mul_term(ug, cg), uf, cf, ug, cg


def is_groebner(G: Sequence[PolyElement[T]], ring: PolyRing[T]) -> bool:
    """Whether ``G`` is a Gröbner basis for the order of ``ring`` (over a
    field): every S-polynomial reduces to zero (Buchberger's criterion).

    >>> from sympy_extras.polys.domains import QQ
    >>> from sympy_extras.polys.rings import PolyRing, lex
    >>> from sympy_extras.polys.f4 import f4
    >>> from sympy_extras.polys.groebnerwalk import is_groebner
    >>> R = PolyRing(QQ, ['x', 'y'], lex)
    >>> x, y = R.gens
    >>> is_groebner([x**2 - y, x*y - R.one], R), is_groebner(f4([x**2 - y, x*y - R.one], R), R)
    (False, True)
    """
    basis = [PolyElement(dict(g.terms), ring) for g in G if g.terms]
    for i in range(len(basis)):
        for j in range(i):
            if monomial_lcm(basis[i].LM, basis[j].LM) == monomial_mul(basis[i].LM, basis[j].LM):
                continue  # product criterion
            s = _s_polynomial(basis[i], basis[j])[0]
            if normal_form(s, basis).terms:
                return False
    return True


def extended_groebner(F: Sequence[PolyElement[T]], ring: PolyRing[T]
                      ) -> tuple[list[PolyElement[T]], list[list[PolyElement[T]]]]:
    """A Gröbner basis of the ideal generated by ``F`` with the
    representations of its elements in terms of ``F``.

    Returns ``(H, C)`` where ``H`` is a list of polynomials forming a
    Gröbner basis for the order of ``ring`` (over a field) and ``C[i]`` is
    a list of polynomials with ``H[i] == sum(C[i][j]*F[j] for j)``. This
    is the Buchberger algorithm with the product criterion only, meant for
    small inputs such as the initial forms of the Gröbner walk.

    Examples
    ========

    >>> from sympy_extras.polys.domains import QQ
    >>> from sympy_extras.polys.rings import PolyRing, lex
    >>> from sympy_extras.polys.groebnerwalk import extended_groebner
    >>> R = PolyRing(QQ, ['x', 'y'], lex)
    >>> x, y = R.gens
    >>> F = [x**2 - y, x*y - R.one]
    >>> H, C = extended_groebner(F, R)
    >>> [str(h) for h in H]
    ['x**2 - y', 'x*y - 1', '-x + y**2', '-y**3 + 1']
    >>> all(h == sum((c*f for c, f in zip(cs, F)), R.zero) for h, cs in zip(H, C))
    True
    """
    polys = [PolyElement(dict(f.terms), ring) for f in F]
    n = len(polys)
    H: list[PolyElement[T]] = []
    C: list[list[PolyElement[T]]] = []
    for i, f in enumerate(polys):
        if f.terms:
            H.append(f)
            C.append([ring.one if j == i else ring.zero for j in range(n)])
    pairs = [(i, j) for i in range(len(H)) for j in range(i)]
    while pairs:
        i, j = pairs.pop()
        if monomial_lcm(H[i].LM, H[j].LM) == monomial_mul(H[i].LM, H[j].LM):
            continue  # product criterion
        s, uf, cf, ug, cg = _s_polynomial(H[i], H[j])
        if not s.terms:
            continue
        cofactors = [a.mul_term(uf, cf) - b.mul_term(ug, cg) for a, b in zip(C[i], C[j])]
        quotients, r = s.div(H)
        if not r.terms:
            continue
        for q, row in zip(quotients, C):
            if q.terms:
                cofactors = [a - q*b for a, b in zip(cofactors, row)]
        H.append(r)
        C.append(cofactors)
        pairs.extend((len(H) - 1, k) for k in range(len(H) - 1))
    return H, C


def _is_graded(order: MonomialOrder) -> bool:
    return order == grevlex or order == grlex


def _weight_of(order: MonomialOrder, nvars: int, G: Sequence[PolyElement[T]]) -> list[Fraction]:
    """A weight vector in the closure of the Gröbner cone of ``G`` for the
    order: the degree vector for graded orders, decreasing powers for
    ``lex``."""
    if isinstance(order, WeightOrder):
        return [Fraction(w) for w in order.weights]
    if _is_graded(order):
        return [Fraction(1) for _ in range(nvars)]
    if order == lex:
        d = 1 + max([max(sum(m) for m in g.terms) for g in G] + [1])
        return [Fraction(d)**(nvars - 1 - i) for i in range(nvars)]
    raise ValueError("no weight vector is known for the order %s" % (order,))


def _target_weight(order: MonomialOrder, nvars: int) -> list[Fraction]:
    """The weight vector whose refinement by ``order`` is ``order``."""
    if isinstance(order, WeightOrder):
        return [Fraction(w) for w in order.weights]
    if _is_graded(order):
        return [Fraction(1) for _ in range(nvars)]
    if order == lex:
        return [Fraction(1 if i == 0 else 0) for i in range(nvars)]
    raise ValueError("no weight vector is known for the order %s" % (order,))


def _integral(weights: Sequence[Fraction]) -> tuple[int, ...]:
    """The rational weight vector scaled to integers."""
    den = 1
    for w in weights:
        den = lcm(den, w.denominator)
    return tuple(int(w*den) for w in weights)


def _next_weight(G: Sequence[PolyElement[T]], weights: list[Fraction], target: list[Fraction]) -> list[Fraction]:
    """The point of the segment from ``weights`` to ``target`` where the
    ordering of the monomials of ``G`` first changes (``target`` if it
    never does)."""
    tmin = Fraction(1)
    for g in G:
        lm = g.LM
        for m in g.terms:
            if m == lm:
                continue
            v = [a - b for a, b in zip(lm, m)]
            wv = sum((a*b for a, b in zip(weights, v)), Fraction(0))
            tv = sum((a*b for a, b in zip(target, v)), Fraction(0))
            if tv < 0 < wv:
                t = wv/(wv - tv)
                if t < tmin:
                    tmin = t
    return [w + tmin*(t - w) for w, t in zip(weights, target)]


def groebner_walk(G: Sequence[PolyElement[T]], ring: PolyRing[T], target: Union[str, MonomialOrder],
                  check: bool = False, lift: str = 'normal_form') -> list[PolyElement[T]]:
    r"""The reduced Gröbner basis for ``target`` of the ideal whose reduced
    Gröbner basis for the order of ``ring`` is ``G`` (over a field), by
    the Gröbner walk: in the ring with the order ``target``, sorted by
    decreasing leading monomial.

    The source and target orders are represented by weight vectors
    $\omega$ and $\tau$ (``lex``, ``grlex``, ``grevlex`` and
    :class:`~sympy_extras.polys.orderings.WeightOrder`), and the basis is
    transformed along the segment from $\omega$ to $\tau$: at every point
    where the ordering of the monomials of the basis changes, a Gröbner
    basis of the initial forms (the terms of maximal weight) is computed
    and lifted. ``lift`` is ``'normal_form'`` (the basis of the initial
    forms by F4, each element $h$ lifted to $h - \mathrm{NF}_G(h)$, normal
    form for the previous order [2]_) or ``'cofactors'`` (through the
    representations computed by :func:`extended_groebner`). With
    ``check`` every intermediate basis is checked with
    :func:`is_groebner` (``RuntimeError`` if one is not a Gröbner basis).

    >>> from sympy_extras.polys.domains import QQ
    >>> from sympy_extras.polys.rings import PolyRing, grevlex, lex
    >>> from sympy_extras.polys.f4 import f4
    >>> from sympy_extras.polys.groebnerwalk import groebner_walk
    >>> R = PolyRing(QQ, ['x', 'y', 'z'], grevlex)
    >>> x, y, z = R.gens
    >>> G = f4([x*z - y**2, x**2 - y*z], R)
    >>> [str(g) for g in groebner_walk(G, R, lex, lift='cofactors')]
    ['x**2 - y*z', 'x*y**2 - y*z**2', 'x*z - y**2', 'y**4 - y*z**3']
    >>> T = R.with_order(lex)
    >>> groebner_walk(groebner_walk(G, R, 'lex'), T, grevlex, check=True) == G
    True
    """
    order = as_order(target)
    if lift not in ('normal_form', 'cofactors'):
        raise ValueError("lift must be 'normal_form' or 'cofactors'")
    n = ring.ngens
    basis = [PolyElement(dict(g.terms), ring) for g in G if g.terms]
    final = ring.with_order(order)
    if not basis:
        return []
    weights = _weight_of(ring.order, n, basis)
    tau = _target_weight(order, n)
    current = ring
    while True:
        new_ring = ring.with_order(WeightOrder(_integral(weights), order))
        initial = [PolyElement(initial_form(g, weights).terms, new_ring) for g in basis]
        lifted: list[PolyElement[T]] = []
        if lift == 'cofactors':
            _, cofactors = extended_groebner(initial, new_ring)
            moved = [PolyElement(g.terms, new_ring) for g in basis]
            for row in cofactors:
                h = new_ring.zero
                for c, g in zip(row, moved):
                    if c.terms:
                        h = h + c*g
                lifted.append(h)
        else:
            for h in f4(initial, new_ring):
                h_old = PolyElement(h.terms, current)
                lifted.append(PolyElement((h_old - normal_form(h_old, basis)).terms, new_ring))
        basis = reduced_basis(lifted, new_ring)
        current = new_ring
        if check and not is_groebner(basis, new_ring):
            raise RuntimeError("the lifted basis is not a Groebner basis")
        if weights == tau:
            break
        weights = _next_weight(basis, weights, tau)
    return reduced_basis([PolyElement(dict(g.terms), final) for g in basis], final)
