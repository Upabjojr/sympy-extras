r"""The SymPy face of :mod:`sympy_extras.polys.groebnerwalk`: the Gröbner
walk on the rings of SymPy (:func:`sympy.polys.rings.ring`) and their
orders (SymPy's and those of :mod:`sympy_extras.polys.orderings_sympy`).

The work is done by the native :func:`sympy_extras.polys.groebnerwalk.groebner_walk`
and :func:`~sympy_extras.polys.groebnerwalk.extended_groebner` on the typed
rings: the coefficients are converted term by term (``QQ``, ``GF(p)``, the
algebraic fields ``QQ<a>``, ``QQ_I`` and the fields of rational functions
over ``QQ`` or ``ZZ``, as :mod:`sympy_extras.polys.ideals_sympy` converts
them), the orders by :func:`sympy_extras.polys.ideals_sympy.core_order`.
Over a ring which is not a field (``ZZ``, ``ZZ[t]``) the walk runs over
its field of fractions and the denominators are cleared, as SymPy's
``groebner`` does over ``ZZ``. :func:`initial_form` is written on SymPy's
ring elements.

Examples
========

>>> from sympy import ring, QQ, grevlex, lex
>>> from sympy.polys.groebnertools import groebner
>>> from sympy_extras.polys.groebnerwalk_sympy import groebner_walk
>>> R, x, y, z = ring("x,y,z", QQ, grevlex)
>>> G = groebner([x*z - y**2, x**2 - y*z], R)
>>> groebner_walk(G, R, lex)
[x**2 - y*z, x*y**2 - y*z**2, x*z - y**2, y**4 - y*z**3]
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from fractions import Fraction
from typing import Generic, Sequence, TypeVar, Union

from sympy.polys.domains.algebraicfield import AlgebraicField as SymPyAlgebraicField
from sympy.polys.domains.domain import Domain
from sympy.polys.domains.finitefield import FiniteField as SymPyFiniteField
from sympy.polys.domains.fractionfield import FractionField as SymPyFractionField
from sympy.polys.orderings import MonomialOrder, igrevlex, igrlex, ilex
from sympy.polys.rings import PolyElement, PolyRing

from sympy_extras._typing import OrderSpec
from sympy_extras.core.expr import as_expr as core_as_expr
from sympy_extras.interop import from_sympy as core_from_sympy

from . import groebnerwalk as native
from . import rings as typed
from .domains.algebraicfield import AlgebraicField
from .domains.domain import Domain as CoreDomain
from .domains.finitefield import FiniteField
from .domains.gaussiandomains import QQ_I as CoreQQ_I
from .domains.rationalfield import QQ as CoreQQ
from .domains.sparsefractionfield import SparseFractionField
from .domains.sparsepolynomialring import RationalPolynomialRing
from .ideals_sympy import SymPyTerms, _Coefficients, _rational, core_order
from .orderings_sympy import as_order

__all__ = ['groebner_walk', 'initial_form', 'extended_groebner']

T = TypeVar('T')


class _Walker(ABC):
    """The native walk on the native domain standing for a SymPy field
    (the type of the native coefficients is that of
    :class:`_TypedWalker`)."""

    @abstractmethod
    def walk(self, G: Sequence[SymPyTerms], source: MonomialOrder, target: MonomialOrder, check: bool,
             lift: str) -> list[SymPyTerms]: ...

    @abstractmethod
    def extended(self, F: Sequence[SymPyTerms], order: MonomialOrder) -> tuple[list[SymPyTerms], list[list[SymPyTerms]]]: ...


class _TypedWalker(_Walker, Generic[T]):

    def __init__(self, coefficients: _Coefficients[T], names: list[str]) -> None:
        self.coefficients: _Coefficients[T] = coefficients
        self.names: list[str] = names

    def _ring(self, order: MonomialOrder) -> typed.PolyRing[T]:
        return typed.PolyRing(self.coefficients.K, self.names, core_order(order, len(self.names)))

    def _in(self, f: SymPyTerms, R: typed.PolyRing[T]) -> typed.PolyElement[T]:
        return typed.PolyElement(self.coefficients.terms_in(f), R)

    def _out(self, f: typed.PolyElement[T]) -> SymPyTerms:
        return self.coefficients.terms_out(f.terms)

    def walk(self, G: Sequence[SymPyTerms], source: MonomialOrder, target: MonomialOrder, check: bool,
             lift: str) -> list[SymPyTerms]:
        R = self._ring(source)
        walked = native.groebner_walk([self._in(g, R) for g in G], R, core_order(target, len(self.names)), check, lift)
        return [self._out(g) for g in walked]

    def extended(self, F: Sequence[SymPyTerms], order: MonomialOrder) -> tuple[list[SymPyTerms], list[list[SymPyTerms]]]:
        R = self._ring(order)
        H, C = native.extended_groebner([self._in(f, R) for f in F], R)
        return [self._out(h) for h in H], [[self._out(c) for c in row] for row in C]


def _typed_walker(K: CoreDomain[T], domain: Domain, names: list[str]) -> _Walker:
    return _TypedWalker(_Coefficients(K, domain), names)


def _walker(ring: PolyRing) -> _Walker:
    """The native walk for the ring (over a field)."""
    domain = ring.domain
    names = [str(s) for s in ring.symbols]
    if domain.is_QQ:
        return _typed_walker(CoreQQ, domain, names)
    if isinstance(domain, SymPyFiniteField):
        return _typed_walker(FiniteField(int(domain.characteristic())), domain, names)
    if isinstance(domain, SymPyAlgebraicField):
        minpoly = [_rational(c) for c in domain.mod.to_list()]
        alpha = core_as_expr(core_from_sympy(domain.ext.as_expr()))
        return _typed_walker(AlgebraicField(minpoly, alpha), domain, names)
    if domain.is_QQ_I:
        return _typed_walker(CoreQQ_I, domain, names)
    if isinstance(domain, SymPyFractionField) and (domain.field.domain.is_QQ or domain.field.domain.is_ZZ):
        field = SparseFractionField(RationalPolynomialRing([str(s) for s in domain.field.symbols]))
        return _typed_walker(field, domain, names)
    raise NotImplementedError("the Groebner walk over %s is not supported" % (domain,))


def _over_field(ring: PolyRing) -> PolyRing:
    """The ring itself over a field, otherwise over its field of fractions
    (SymPy's ``groebner`` convention)."""
    return ring if ring.domain.is_Field else ring.clone(domain=ring.domain.get_field())


def initial_form(f: PolyElement, weights: Sequence[Union[int, Fraction]]) -> PolyElement:
    """The terms of ``f`` of maximal weight for the weight vector.

    >>> from sympy import ring, QQ
    >>> from sympy_extras.polys.groebnerwalk_sympy import initial_form
    >>> R, x, y = ring("x,y", QQ)
    >>> initial_form(x**2*y + x*y**2 + x, (1, 2))
    x*y**2
    """
    ring = f.ring
    best, terms = None, []
    for monom, coeff in f.terms():
        w = sum(a*b for a, b in zip(weights, monom))
        if best is None or w > best:
            best, terms = w, [(monom, coeff)]
        elif w == best:
            terms.append((monom, coeff))
    return ring.from_dict(dict(terms))


def extended_groebner(F: Sequence[PolyElement], ring: PolyRing) -> tuple[list[PolyElement], list[list[PolyElement]]]:
    """A Gröbner basis of the ideal generated by ``F`` with the
    representations of its elements in terms of ``F``
    (:func:`sympy_extras.polys.groebnerwalk.extended_groebner`).

    Returns ``(H, C)`` where ``H`` is a list of polynomials forming a
    Gröbner basis for the order of ``ring`` and ``C[i]`` is a list of
    polynomials with ``H[i] == sum(C[i][j]*F[j] for j)``; over a ring which
    is not a field, they are elements of the ring over its field of
    fractions.

    Examples
    ========

    >>> from sympy import ring, QQ
    >>> from sympy_extras.polys.groebnerwalk_sympy import extended_groebner
    >>> R, x, y = ring("x,y", QQ)
    >>> H, C = extended_groebner([x**2 - y, x*y - 1], R)
    >>> all(h == sum(c*f for c, f in zip(cs, [x**2 - y, x*y - 1])) for h, cs in zip(H, C))
    True
    """
    field = _over_field(ring)
    polys = [field.from_dict(dict(ring(f))) for f in F]
    H, C = _walker(field).extended([dict(f) for f in polys], field.order)
    return [field.from_dict(h) for h in H], [[field.from_dict(c) for c in row] for row in C]


def groebner_walk(G: Sequence[PolyElement], ring: PolyRing, target: OrderSpec,
                  check: bool = False, lift: str = 'normal_form') -> list[PolyElement]:
    """Convert the reduced Gröbner basis ``G`` of ``ring`` (elements of the
    ring, whose order is the source order) to the reduced Gröbner basis for
    the ``target`` order, returned in the ring with that order
    (:func:`sympy_extras.polys.groebnerwalk.groebner_walk`).

    ``lift`` is ``'normal_form'`` (lifting by normal forms modulo the
    previous basis, the initial forms handled by F4) or ``'cofactors'``
    (lifting through the representations computed by
    :func:`extended_groebner`).

    Examples
    ========

    >>> from sympy import ring, QQ, grevlex, lex
    >>> from sympy.polys.groebnertools import groebner
    >>> from sympy_extras.polys.groebnerwalk_sympy import groebner_walk
    >>> R, x, y, z = ring("x,y,z", QQ, grevlex)
    >>> G = groebner([x*z - y**2, x**2 - y*z], R)
    >>> groebner_walk(G, R, lex)
    [x**2 - y*z, x*y**2 - y*z**2, x*z - y**2, y**4 - y*z**3]
    >>> T = R.clone(order=lex)
    >>> groebner_walk(G, R, lex) == groebner([T.from_dict(dict(g)) for g in [x*z - y**2, x**2 - y*z]], T)
    True
    >>> groebner_walk(groebner_walk(G, R, lex), T, grevlex) == G
    True
    """
    order = as_order(target)
    result_ring = ring.clone(order=order)
    polys = [ring.from_dict(dict(g)) for g in G]
    if not polys or all(not g for g in polys):
        return [result_ring.from_dict(dict(g)) for g in polys]
    if lift not in ('normal_form', 'cofactors'):
        raise ValueError("lift must be 'normal_form' or 'cofactors'")
    if ring.order in (ilex, igrlex, igrevlex):
        raise ValueError("inverse orders are not global orders")
    field = _over_field(ring)
    walked = _walker(field).walk([dict(field.from_dict(dict(g))) for g in polys if g], ring.order, order, check, lift)
    if field is ring:
        basis = [result_ring.from_dict(g) for g in walked]
    else:
        target_field = field.clone(order=order)
        basis = [target_field.from_dict(g).clear_denoms()[1].set_ring(result_ring) for g in walked]
    return sorted(basis, key=lambda g: result_ring.order(g.LM), reverse=True)
