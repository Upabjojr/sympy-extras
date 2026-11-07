"""Square-free decomposition of multivariate polynomials over ``ZZ``,
``QQ``, ``ZZ[a, ...]`` and ``QQ[a, ...]`` (Yun's algorithm in the first
variable after its content, the content decomposed recursively).

Ported from SymPy's ``sympy.polys.sqfreetools`` (``dup_sqf_list``,
``dmp_sqf_list``), on sparse polynomials: a :class:`SqfDomain` gives the
gcd of SymPy's ``dmp_inner_gcd`` over the domain (canonical over ``ZZ``,
monic over ``QQ``, ``dmp_rr_prs_gcd`` over a polynomial domain) and the
split of the ground content (the leading coefficient over ``QQ``); the
coefficient is returned as a polynomial constant in the generators.

>>> from sympy_extras.core import Symbol
>>> x, y = Symbol('x'), Symbol('y')
>>> D = IntegerSqfDomain(['x', 'y'])
>>> c, factors = sqf_list(D.ring.from_expr(2*(x + y)**2*(x - y)**3*(y + 1)**2), D)
>>> D.ring.to_expr(c), [(D.ring.to_expr(f), k) for f, k in factors]
(2, [(x*y + x + y**2 + y, 2), (x - y, 3)])
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/polys/sqfreetools.py (dup_sqf_list, dmp_sqf_list) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: sparse polynomials; the domain operations behind SqfDomain.
# ---------------------------------------------------------------------
from __future__ import annotations

from fractions import Fraction
from math import gcd
from typing import Generic, Sequence, TypeVar

from .domains.sparsepolynomialring import (IntegerPolynomialRing, IteratedPolynomialRing, RationalPolynomialRing,
                                           SparsePolynomialRing)
from .rings import Monomial, PolyElement, PolyRing

__all__ = ['SqfDomain', 'IntegerSqfDomain', 'RationalSqfDomain', 'ParameterSqfDomain', 'sqf_list']

T = TypeVar('T')


class SqfDomain(Generic[T]):
    """The polynomials in the generators ``gen_names`` (the parameters of a
    polynomial domain after them) with SymPy's operations of the domain."""

    def __init__(self, ring: SparsePolynomialRing[T], n_gens: int) -> None:
        self.domain: SparsePolynomialRing[T] = ring
        self.ring: PolyRing[T] = ring.ring
        self.n_gens: int = n_gens

    def gcd(self, f: PolyElement[T], g: PolyElement[T]) -> PolyElement[T]:
        return self.domain.gcd(f, g)

    def ground_split(self, f: PolyElement[T]) -> tuple[PolyElement[T], PolyElement[T]]:
        """``(coeff, f / coeff)``: the ground content with the sign of the
        leading coefficient over a ring, the leading coefficient over a
        field; ``coeff`` constant in the generators."""
        raise NotImplementedError

    def sub(self) -> SqfDomain[T]:
        """The domain without the first generator."""
        raise NotImplementedError

    def degree(self, f: PolyElement[T]) -> int:
        return max((m[0] for m in f.terms), default=-1)

    def drop_first(self, f: PolyElement[T], other: SqfDomain[T]) -> PolyElement[T]:
        return other.ring.from_dict({m[1:]: c for m, c in f.terms.items()})

    def lift(self, f: PolyElement[T]) -> PolyElement[T]:
        return self.ring.from_dict({(0,) + m: c for m, c in f.terms.items()})

    def content_first(self, f: PolyElement[T]) -> PolyElement[T]:
        """SymPy's ``dmp_content``: the gcd of the coefficients in ``x0``
        (fold from the leading one, stopping at ``1``), the sign of its
        ground leading coefficient made positive; constant in ``x0``."""
        coefficients: dict[int, dict[Monomial, T]] = {}
        for m, c in f.terms.items():
            coefficients.setdefault(m[0], {})[(0,) + m[1:]] = c
        ordered = [self.ring.from_dict(coefficients[d]) for d in sorted(coefficients, reverse=True)]
        content = ordered[0]
        for coefficient in ordered[1:]:
            content = self.gcd(content, coefficient)
            if content == self.ring.one:
                break
        if self.negative(content):
            content = -content
        return content

    def negative(self, f: PolyElement[T]) -> bool:
        return bool(f.terms) and self.ring.domain.is_negative(f.LC)


class IntegerSqfDomain(SqfDomain[int]):
    def __init__(self, names: Sequence[str]) -> None:
        super().__init__(IntegerPolynomialRing(names), len(names))
        self.names = list(names)

    def ground_split(self, f: PolyElement[int]) -> tuple[PolyElement[int], PolyElement[int]]:
        c = 0
        for v in f.terms.values():
            c = gcd(c, v)
        if f.LC < 0:
            c = -c
        return self.ring.ground(c), self.ring.from_dict({m: v // c for m, v in f.terms.items()})

    def sub(self) -> SqfDomain[int]:
        return IntegerSqfDomain(self.names[1:])


class RationalSqfDomain(SqfDomain[Fraction]):
    def __init__(self, names: Sequence[str]) -> None:
        super().__init__(RationalPolynomialRing(names), len(names))
        self.names = list(names)

    def gcd(self, f: PolyElement[Fraction], g: PolyElement[Fraction]) -> PolyElement[Fraction]:
        h = self.domain.gcd(f, g)
        return h.exquo_ground(h.LC) if h.terms else h

    def ground_split(self, f: PolyElement[Fraction]) -> tuple[PolyElement[Fraction], PolyElement[Fraction]]:
        lc = f.LC
        return self.ring.ground(lc), f.exquo_ground(lc)

    def sub(self) -> SqfDomain[Fraction]:
        return RationalSqfDomain(self.names[1:])


class ParameterSqfDomain(SqfDomain[T]):
    """The generators over ``K = ZZ[a, ...]`` or ``QQ[a, ...]``."""

    def __init__(self, params: SparsePolynomialRing[T], gen_names: Sequence[str]) -> None:
        self.iterated: IteratedPolynomialRing[T] = IteratedPolynomialRing(params, gen_names)
        super().__init__(self.iterated, len(gen_names))
        self.params: SparsePolynomialRing[T] = params
        self.gen_names: list[str] = list(gen_names)

    def ground_split(self, f: PolyElement[T]) -> tuple[PolyElement[T], PolyElement[T]]:
        content = self.iterated.ground_content(f)
        if self.params.is_negative(self.iterated.ground_LC(f)):
            content = -content
        embedded = self.iterated._embed_params(content)
        return embedded, f.exquo(embedded)

    def negative(self, f: PolyElement[T]) -> bool:
        return self.params.is_negative(self.iterated.ground_LC(f))

    def sub(self) -> SqfDomain[T]:
        return ParameterSqfDomain(self.params, self.gen_names[1:])


def _first_derivative(f: PolyElement[T]) -> PolyElement[T]:
    return f.diff(0)


# PORTED FROM SYMPY: sympy/polys/sqfreetools.py:dup_sqf_list, dmp_sqf_list
def sqf_list(f: PolyElement[T], D: SqfDomain[T], all: bool = False
             ) -> tuple[PolyElement[T], list[tuple[PolyElement[T], int]]]:
    """``(coeff, [(g, k)])`` with ``f = coeff * prod g**k``, the ``g``
    square-free and pairwise coprime, in increasing multiplicity."""
    if not f.terms:
        return D.ring.zero, []
    coeff, f = D.ground_split(f)
    if D.degree(f) <= 0 and D.n_gens == 1:
        return coeff, []
    deg = D.degree(f)
    if deg < 0:
        return coeff, []
    result: dict[int, PolyElement[T]] = {}
    content = D.ring.one
    if D.n_gens > 1:
        content = D.content_first(f)
        if content != D.ring.one:
            f = f.exquo(content)
    if deg != 0:
        h = _first_derivative(f)
        g = D.gcd(f, h)
        p, q = f.exquo(g), h.exquo(g)
        i = 1
        while True:
            h = q - _first_derivative(p)
            if not h.terms:
                result[i] = p
                break
            g = D.gcd(p, h)
            p, q = p.exquo(g), h.exquo(g)
            if all or D.degree(g) > 0:
                result[i] = g
            i += 1
    if D.n_gens == 1:
        return coeff, [(result[i], i) for i in sorted(result)]
    S = D.sub()
    coeff_content, result_content = sqf_list(D.drop_first(content, S), S, all=all)
    coeff = coeff * D.lift(coeff_content)
    for factor, i in result_content:
        lifted = D.lift(factor)
        result[i] = result[i] * lifted if i in result else lifted
    return coeff, [(result[i], i) for i in sorted(result)]
