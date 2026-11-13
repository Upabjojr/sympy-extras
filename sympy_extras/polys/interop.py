"""Conversions between SymPy's sparse polynomial rings and the typed rings
of :mod:`sympy_extras.polys.rings`, for the callers which still build
SymPy rings (:class:`~sympy_extras.polys.ideals.Ideal`, the modular
Gröbner bases) while the algorithms run on the typed layer.

The coefficient domains ``QQ``, ``ZZ`` and ``GF(p)`` are converted; a
monomial order of SymPy (or of :mod:`sympy_extras.polys.orderings_sympy`)
becomes a typed order whose key is the flattened tuple of the SymPy key,
which compares the same way since the components have a fixed length in
a given ring.

Examples
========

>>> from sympy import QQ, ring, grevlex
>>> from sympy_extras.polys.interop import qq_from_sympy, qq_to_sympy
>>> R, x, y = ring('x,y', QQ, grevlex)
>>> T, polys = qq_from_sympy([x**2 - y/2, x*y], R)
>>> T, [str(p) for p in polys]
(QQ[x,y], ['x**2 - 1/2*y', 'x*y'])
>>> qq_to_sympy(polys, R) == [x**2 - y/2, x*y]
True
"""
from __future__ import annotations

from fractions import Fraction
from typing import Optional, Sequence

from sympy.core.expr import Expr as SymPyExpr
from sympy.core.numbers import Rational
from sympy.core.symbol import Symbol as SymPySymbol
from sympy.polys.orderings import MonomialOrder as SymPyMonomialOrder
from sympy.polys.rings import PolyElement as SymPyPolyElement, PolyRing as SymPyPolyRing

from typing import TypeVar

from sympy_extras._typing_sympy import DomainElement, Monomial, as_expr as sympy_as_expr
from sympy_extras.core.expr import as_expr as core_as_expr
from sympy_extras.interop import from_sympy as core_from_sympy, to_sympy as core_to_sympy
from sympy_extras.core.symbol import Symbol

from .domains.domain import Domain
from .polyclasses import DensePoly

from .domains.finitefield import FiniteField
from .domains.integerring import ZZ
from .domains.rationalfield import QQ
from .rings import MonomialOrder, OrderKeyValue, OrderValue, PolyElement, PolyRing

__all__ = ['typed_order', 'qq_from_sympy', 'qq_to_sympy', 'zz_from_sympy', 'zz_to_sympy', 'gf_from_sympy',
           'gf_to_sympy', 'sympy_modulus', 'domain_to_sympy', 'domain_from_sympy', 'dense_to_sympy',
           'dense_from_sympy', 'ring_to_sympy', 'ring_from_sympy']

T = TypeVar('T')



def _flatten(value: object, into: list[OrderValue]) -> None:
    if isinstance(value, bool):
        into.append(int(value))
    elif isinstance(value, int):
        into.append(value)
    elif isinstance(value, Fraction):
        into.append(value)
    elif isinstance(value, tuple):
        for item in value:
            _flatten(item, into)
    else:
        raise TypeError('unexpected entry %r in a monomial order key' % (value,))


def typed_order(order: SymPyMonomialOrder, ngens: int) -> MonomialOrder:
    """The typed order with the flattened key of a SymPy order, graded
    when the squares of the variables are above the variables."""
    cache: dict[Monomial, OrderKeyValue] = {}

    def key(monomial: Monomial) -> OrderKeyValue:
        found = cache.get(monomial)
        if found is None:
            flat: list[OrderValue] = []
            _flatten(order(monomial), flat)
            found = tuple(flat)
            cache[monomial] = found
        return found

    variables = [tuple(1 if k == i else 0 for k in range(ngens)) for i in range(ngens)]
    squares = [tuple(2 if k == i else 0 for k in range(ngens)) for i in range(ngens)]
    graded = all(key(s) > key(v) for s in squares for v in variables)
    return MonomialOrder(str(order), key, graded)


def _names(ring: SymPyPolyRing) -> list[str]:
    return [str(s) for s in ring.symbols]


def _rational(c: DomainElement, ring: SymPyPolyRing) -> Fraction:
    value = ring.domain.to_sympy(c)
    if not isinstance(value, Rational):
        raise TypeError('%s is not a rational number' % (value,))
    return Fraction(int(value.p), int(value.q))


def qq_from_sympy(polys: Sequence[SymPyPolyElement], ring: SymPyPolyRing
                  ) -> tuple[PolyRing[Fraction], list[PolyElement[Fraction]]]:
    """The typed ring over ``QQ`` and the polynomials of a SymPy ring over
    ``QQ``."""
    typed = PolyRing(QQ, _names(ring), typed_order(ring.order, ring.ngens))
    return typed, [typed.from_dict({tuple(m): _rational(c, ring) for m, c in f.terms()}) for f in polys]


def qq_to_sympy(polys: Sequence[PolyElement[Fraction]], ring: SymPyPolyRing) -> list[SymPyPolyElement]:
    return [ring.from_dict({m: ring.domain(c.numerator, c.denominator) for m, c in f.terms.items()}) for f in polys]


def zz_from_sympy(polys: Sequence[SymPyPolyElement], ring: SymPyPolyRing) -> tuple[PolyRing[int], list[PolyElement[int]]]:
    """The typed ring over ``ZZ`` and the polynomials of a SymPy ring over
    ``ZZ``."""
    typed = PolyRing(ZZ, _names(ring), typed_order(ring.order, ring.ngens))
    return typed, [typed.from_dict({tuple(m): int(c) for m, c in f.terms()}) for f in polys]


def zz_to_sympy(polys: Sequence[PolyElement[int]], ring: SymPyPolyRing) -> list[SymPyPolyElement]:
    return [ring.from_dict({m: ring.domain(c) for m, c in f.terms.items()}) for f in polys]


def sympy_modulus(ring: SymPyPolyRing) -> Optional[int]:
    """The prime ``p`` when the domain of the SymPy ring is ``GF(p)``."""
    domain = ring.domain
    if domain.is_FiniteField:
        return int(domain.characteristic())
    return None


def gf_from_sympy(polys: Sequence[SymPyPolyElement], ring: SymPyPolyRing, p: int
                  ) -> tuple[PolyRing[int], list[PolyElement[int]]]:
    """The typed ring over ``GF(p)`` and the polynomials of a SymPy ring
    over ``GF(p)``."""
    typed = PolyRing(FiniteField(p), _names(ring), typed_order(ring.order, ring.ngens))
    return typed, [typed.from_dict({tuple(m): int(c) % p for m, c in f.terms()}) for f in polys]


def gf_to_sympy(polys: Sequence[PolyElement[int]], ring: SymPyPolyRing) -> list[SymPyPolyElement]:
    return [ring.from_dict({m: ring.domain(c) for m, c in f.terms.items()}) for f in polys]


# ---------------------------------------------------------------------------
# SymPy expressions and the typed layer (through the typed core)


def domain_to_sympy(K: Domain[T], a: T) -> SymPyExpr:
    """An element of a typed domain as a SymPy expression."""
    return sympy_as_expr(core_to_sympy(K.to_expr(a)))


def domain_from_sympy(K: Domain[T], e: SymPyExpr) -> T:
    """A SymPy expression which is an element of the typed domain."""
    return K.from_expr(core_as_expr(core_from_sympy(e)))


def dense_to_sympy(f: DensePoly[T]) -> SymPyExpr:
    """A dense polynomial as a SymPy expression."""
    return sympy_as_expr(core_to_sympy(f.to_expr()))


def dense_from_sympy(e: SymPyExpr, gen: SymPySymbol, K: Domain[T]) -> DensePoly[T]:
    """A SymPy expression which is a polynomial in ``gen`` over ``K``."""
    return DensePoly.from_expr(core_as_expr(core_from_sympy(e)), Symbol(gen.name), K)


def ring_to_sympy(f: PolyElement[T]) -> SymPyExpr:
    """A sparse polynomial as a SymPy expression."""
    return sympy_as_expr(core_to_sympy(f.to_expr()))


def ring_from_sympy(R: PolyRing[T], e: SymPyExpr) -> PolyElement[T]:
    """A SymPy expression which is a polynomial of the typed ring."""
    return R.from_expr(core_as_expr(core_from_sympy(e)))
