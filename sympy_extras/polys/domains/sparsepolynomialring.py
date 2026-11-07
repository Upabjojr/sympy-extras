"""The polynomial ring ``K[y1, ..., yk]`` as a domain whose elements are
sparse polynomials (:class:`~sympy_extras.polys.rings.PolyElement`), with
SymPy's conventions for the gcd and the lcm of its elements (those of
SymPy's ``PolyElement``): this is the domain ``ZZ[a, b]`` or ``QQ[a, b]``
of a :class:`~sympy_extras.polys.polytools.Poly`. An element seen as
univariate in ``y1`` has coefficients in ``K[y2, ..., yk]``, again a
``SparsePolynomialRing`` with elements of the same Python type, so that
the recursion is typed with a single element type whatever the number of
variables (``PolynomialRing`` nests one type per variable).

:class:`RationalPolynomialRing` is ``QQ[y1, ..., yk]`` (SymPy's gcd of
rationals, monic gcd and lcm); :class:`IteratedPolynomialRing` is
``K[x1, ..., xn]`` over such a ring with SymPy's gcd of a ``Poly`` over a
polynomial domain (``dmp_rr_prs_gcd``).

Examples
========

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.polys.domains import ZZ
>>> from sympy_extras.polys.domains.sparsepolynomialring import SparsePolynomialRing
>>> x, y = Symbol('x'), Symbol('y')
>>> R = SparsePolynomialRing(ZZ, ['x', 'y'])
>>> f, g = R.from_expr((x + y)*(x - 2*y)), R.from_expr((x + y)*(3*x + y))
>>> R.to_expr(R.gcd(f, g)), R.name
(x + y, 'ZZ[x,y]')
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/polys/rings.py (PolyElement.cofactors, _gcd_zero,
# _gcd_monom, _gcd_QQ, lcm, content), sympy/polys/domains/polynomialring.py
# (canonical_unit), sympy/polys/domains/field.py (Field.gcd, Field.lcm),
# sympy/polys/euclidtools.py (dup_rr_prs_gcd, dmp_rr_prs_gcd,
# _dup_rr_trivial_gcd, _dmp_rr_trivial_gcd, _dmp_simplify_gcd, dmp_content,
# dup_rr_lcm, dmp_rr_lcm), sympy/polys/densetools.py (dmp_ground_content)
# (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: sparse elements in all the variables instead of dense nested
# lists; the heuristic gcd over ZZ replaced by the subresultant gcd (the
# same canonical result).
# ---------------------------------------------------------------------

from __future__ import annotations

from fractions import Fraction
from math import gcd as integer_gcd, lcm as integer_lcm
from typing import Generic, Sequence, TypeVar

from sympy_extras.core.expr import Expr

from .. import densebasic, euclidtools
from ..rings import Monomial, PolyElement, PolyRing
from .domain import Domain
from .integerring import ZZ, IntegerRing
from .rationalfield import QQ

__all__ = ['SparsePolynomialRing', 'RationalPolynomialRing', 'IteratedPolynomialRing']

T = TypeVar('T')
U = TypeVar('U')


class SparsePolynomialRing(Domain[PolyElement[T]], Generic[T]):
    """``K[y1, ..., yk]`` with sparse elements (lex order)."""

    is_field = False

    def __init__(self, ground: Domain[T], names: Sequence[str]) -> None:
        self.ground: Domain[T] = ground
        self.names: list[str] = list(names)
        self.ring: PolyRing[T] = PolyRing(ground, self.names)
        self.name = '%s[%s]' % (ground.name, ','.join(self.names))
        self.is_ordered = ground.is_ordered
        self.characteristic = ground.characteristic

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, SparsePolynomialRing) and type(other) is type(self) and other.ground == self.ground
                and other.names == self.names)

    def __hash__(self) -> int:
        return hash(('SparsePolynomialRing', self.ground, tuple(self.names)))

    @property
    def zero(self) -> PolyElement[T]:
        return self.ring.zero

    @property
    def one(self) -> PolyElement[T]:
        return self.ring.one

    def is_zero(self, a: PolyElement[T]) -> bool:
        return not a.terms

    def is_one(self, a: PolyElement[T]) -> bool:
        return a == self.ring.one

    def is_negative(self, a: PolyElement[T]) -> bool:
        return bool(a.terms) and self.ground.is_negative(a.LC)

    def is_nonnegative(self, a: PolyElement[T]) -> bool:
        return not a.terms or not self.ground.is_negative(a.LC)

    def is_unit(self, a: PolyElement[T]) -> bool:
        return a.is_ground and bool(a.terms) and self.ground.is_unit(a.LC)

    def add(self, a: PolyElement[T], b: PolyElement[T]) -> PolyElement[T]:
        return a + b

    def sub(self, a: PolyElement[T], b: PolyElement[T]) -> PolyElement[T]:
        return a - b

    def neg(self, a: PolyElement[T]) -> PolyElement[T]:
        return -a

    def mul(self, a: PolyElement[T], b: PolyElement[T]) -> PolyElement[T]:
        return a * b

    def exquo(self, a: PolyElement[T], b: PolyElement[T]) -> PolyElement[T]:
        if not b.terms:
            raise ZeroDivisionError('division by zero in %s' % self.name)
        return a.exquo(b)

    def invert(self, a: PolyElement[T]) -> PolyElement[T]:
        if self.is_unit(a):
            return self.ring.ground(self.ground.invert(a.LC))
        raise ValueError('%s is not a unit of %s' % (a, self.name))

    # PORTED FROM SYMPY: sympy/polys/domains/polynomialring.py:PolynomialRing.canonical_unit
    def canonical_unit(self, a: PolyElement[T]) -> PolyElement[T]:
        """``-1`` when the leading coefficient is negative, ``1`` otherwise
        (SymPy's ``Domain.canonical_unit`` of the leading coefficient, over
        ``QQ`` too)."""
        if a.terms and self.ground.is_negative(a.LC):
            return -self.one
        return self.one

    # -- dense views

    def to_univariate(self, a: PolyElement[T], inner: SparsePolynomialRing[T]) -> list[PolyElement[T]]:
        """``a`` as a dense polynomial in ``y1`` with coefficients in
        ``inner = K[y2, ..., yk]``."""
        by_degree: dict[int, dict[Monomial, T]] = {}
        for m, c in a.terms.items():
            by_degree.setdefault(m[0], {})[m[1:]] = c
        return densebasic.dup_from_dict({d: inner.ring.from_dict(terms) for d, terms in by_degree.items()}, inner)

    def from_univariate(self, f: list[PolyElement[T]]) -> PolyElement[T]:
        terms: dict[Monomial, T] = {}
        n = len(f) - 1
        for i, c in enumerate(f):
            for m, value in c.terms.items():
                terms[(n - i,) + m] = value
        return self.ring.from_dict(terms)

    def _ground_univariate(self, a: PolyElement[T]) -> list[T]:
        return densebasic.dup_from_dict({m[0]: c for m, c in a.terms.items()}, self.ground)

    def _from_ground_univariate(self, f: list[T]) -> PolyElement[T]:
        n = len(f) - 1
        return self.ring.from_dict({(n - i,): c for i, c in enumerate(f)})

    # -- the gcd of SymPy's PolyElement

    def _ground_gcd(self, a: T, b: T) -> T:
        return self.ground.gcd(a, b)

    def _ground_lcm(self, a: T, b: T) -> T:
        return self.ground.lcm(a, b)

    def content(self, a: PolyElement[T]) -> T:
        """The ground gcd of the coefficients (SymPy's
        ``PolyElement.content``)."""
        result = self.ground.zero
        for c in a.terms.values():
            result = self._ground_gcd(result, c)
        return result

    def primitive(self, a: PolyElement[T]) -> tuple[T, PolyElement[T]]:
        c = self.content(a)
        if self.ground.is_zero(c) or self.ground.is_one(c):
            return c, a
        return c, a.exquo_ground(c)

    # PORTED FROM SYMPY: sympy/polys/rings.py:PolyElement.cofactors
    def gcd(self, a: PolyElement[T], b: PolyElement[T]) -> PolyElement[T]:
        """SymPy's ``PolyElement.gcd``: ``b`` or ``-b`` (nonnegative
        leading coefficient) when ``a = 0``; the monomial gcd times the
        ground gcd of the coefficients when one of them is a term; the
        canonical gcd otherwise (positive leading coefficient over ``ZZ``,
        monic over ``QQ``)."""
        if not a.terms and not b.terms:
            return self.zero
        if not a.terms:
            return b if self.is_nonnegative(b) else -b
        if not b.terms:
            return a if self.is_nonnegative(a) else -a
        if len(a.terms) == 1:
            return self._gcd_monomial(a, b)
        if len(b.terms) == 1:
            return self._gcd_monomial(b, a)
        return self._gcd_general(a, b)

    # PORTED FROM SYMPY: sympy/polys/rings.py:PolyElement._gcd_monom
    def _gcd_monomial(self, term: PolyElement[T], g: PolyElement[T]) -> PolyElement[T]:
        ((monomial, coefficient),) = term.terms.items()
        for m, c in g.terms.items():
            monomial = tuple(min(i, j) for i, j in zip(monomial, m))
            coefficient = self._ground_gcd(coefficient, c)
        return self.ring.term(monomial, coefficient)

    def _gcd_general(self, a: PolyElement[T], b: PolyElement[T]) -> PolyElement[T]:
        """The canonical gcd (the primitive part of the last subresultant
        times the gcd of the contents, positive leading coefficient), each
        variable in turn."""
        if len(self.names) == 1:
            return self._from_ground_univariate(
                euclidtools.dup_gcd(self._ground_univariate(a), self._ground_univariate(b), self.ground))
        inner = SparsePolynomialRing(self.ground, self.names[1:])
        g = euclidtools.dup_gcd(self.to_univariate(a, inner), self.to_univariate(b, inner), inner)
        return self.from_univariate(g)

    # PORTED FROM SYMPY: sympy/polys/rings.py:PolyElement.lcm
    def lcm(self, a: PolyElement[T], b: PolyElement[T]) -> PolyElement[T]:
        """SymPy's ``PolyElement.lcm``: ``f g / gcd(f, g)`` of the
        primitive parts times the lcm of the contents."""
        if not a.terms or not b.terms:
            return self.zero
        ca, pa = self.primitive(a)
        cb, pb = self.primitive(b)
        c = self._ground_lcm(ca, cb)
        return (pa * pb).exquo(self.gcd(pa, pb)).mul_ground(c)

    def from_int(self, n: int) -> PolyElement[T]:
        return self.ring.from_int(n)

    def convert(self, a: U, K: Domain[U]) -> PolyElement[T]:
        if isinstance(K, SparsePolynomialRing) and K == self and isinstance(a, PolyElement):
            return self.ring.from_dict({m: self.ground.convert(c, K.ground) for m, c in a.terms.items()})
        if isinstance(K, IntegerRing) and isinstance(a, int):
            return self.from_int(a)
        return self.ring.ground(self.ground.convert(a, K))

    def to_expr(self, a: PolyElement[T]) -> Expr:
        return self.ring.to_expr(a)

    def from_expr(self, e: Expr) -> PolyElement[T]:
        return self.ring.from_expr(e)

    def to_str(self, a: PolyElement[T]) -> str:
        return str(self.to_expr(a))

    def size(self, a: PolyElement[T]) -> int:
        return sum(self.ground.size(c) for c in a.terms.values()) or 1


class RationalPolynomialRing(SparsePolynomialRing[Fraction]):
    """``QQ[y1, ..., yk]`` with SymPy's conventions: the ground gcd of
    two rationals is the gcd of the numerators over the lcm of the
    denominators; the gcd of two polynomials (neither a term) and the lcm
    are monic.

    >>> from sympy_extras.core import Symbol, Rational
    >>> a = Symbol('a')
    >>> R = RationalPolynomialRing(['a'])
    >>> R.to_expr(R.gcd(R.from_expr(-a/2), R.from_expr(a/3))), R.to_expr(R.gcd(R.from_expr(2*a + 2), R.from_expr(a/3 + Rational(1, 3))))
    (a/6, a + 1)
    """

    def __init__(self, names: Sequence[str]) -> None:
        super().__init__(QQ, names)

    # PORTED FROM SYMPY: sympy/polys/domains/field.py:Field.gcd
    def _ground_gcd(self, a: Fraction, b: Fraction) -> Fraction:
        return Fraction(integer_gcd(a.numerator, b.numerator), integer_lcm(a.denominator, b.denominator))

    # PORTED FROM SYMPY: sympy/polys/domains/field.py:Field.lcm
    def _ground_lcm(self, a: Fraction, b: Fraction) -> Fraction:
        return Fraction(integer_lcm(a.numerator, b.numerator), integer_gcd(a.denominator, b.denominator))

    # PORTED FROM SYMPY: sympy/polys/rings.py:PolyElement._gcd_QQ
    def _gcd_general(self, a: PolyElement[Fraction], b: PolyElement[Fraction]) -> PolyElement[Fraction]:
        Z = SparsePolynomialRing(ZZ, self.names)
        h = Z._gcd_general(self._cleared(a, Z), self._cleared(b, Z))
        result = self.ring.from_dict({m: Fraction(c) for m, c in h.terms.items()})
        return result.exquo_ground(result.LC)

    @staticmethod
    def _cleared(a: PolyElement[Fraction], Z: SparsePolynomialRing[int]) -> PolyElement[int]:
        d = 1
        for c in a.terms.values():
            d = integer_lcm(d, c.denominator)
        return Z.ring.from_dict({m: int(c * d) for m, c in a.terms.items()})

    def lcm(self, a: PolyElement[Fraction], b: PolyElement[Fraction]) -> PolyElement[Fraction]:
        if not a.terms or not b.terms:
            return self.zero
        h = (a * b).exquo(self.gcd(a, b))
        return h.exquo_ground(h.LC)


class IteratedPolynomialRing(SparsePolynomialRing[T]):
    """``K[x1, ..., xn]`` over a polynomial ring ``K = k[y1, ..., ym]``
    with SymPy's gcd of dense polynomials over a polynomial domain
    (``dmp_rr_prs_gcd``): contents taken variable after variable down to
    ``K``, whose gcd is :meth:`SparsePolynomialRing.gcd`. The elements are
    sparse polynomials in ``x1, ..., xn, y1, ..., ym``. This gives the
    gcd of a SymPy ``Poly`` with the domain ``ZZ[y]`` or ``QQ[y]``, which
    is not the canonical gcd of the polynomials in all the variables:

    >>> from sympy_extras.core import Symbol
    >>> x, y = Symbol('x'), Symbol('y')
    >>> R = IteratedPolynomialRing(RationalPolynomialRing(['y']), ['x'])
    >>> R.to_expr(R.gcd(R.from_expr(y*x/2 + y), R.from_expr(x/3)))
    1/6
    """

    def __init__(self, params: SparsePolynomialRing[T], gen_names: Sequence[str]) -> None:
        super().__init__(params.ground, list(gen_names) + params.names)
        self.params: SparsePolynomialRing[T] = params
        self.n_gens: int = len(gen_names)
        self.gen_names: list[str] = list(gen_names)
        self.name = '%s[%s]' % (params.name, ','.join(self.gen_names))

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, IteratedPolynomialRing) and other.params == self.params
                and other.gen_names == self.gen_names)

    def __hash__(self) -> int:
        return hash(('IteratedPolynomialRing', self.params, tuple(self.gen_names)))

    def ground_LC(self, a: PolyElement[T]) -> PolyElement[T]:
        """The leading coefficient in ``x1, ..., xn`` (lex), an element of
        ``K`` (SymPy's ``dmp_ground_LC``)."""
        if not a.terms:
            return self.params.zero
        n = self.n_gens
        top = max(m[:n] for m in a.terms)
        return self.params.ring.from_dict({m[n:]: c for m, c in a.terms.items() if m[:n] == top})

    def _inner(self) -> SparsePolynomialRing[T]:
        """The coefficients in ``x1``: ``K[x2, ..., xn]``, or ``K``."""
        if self.n_gens == 1:
            return self.params
        return IteratedPolynomialRing(self.params, self.gen_names[1:])

    def _lift(self, c: PolyElement[T]) -> PolyElement[T]:
        """An element of the inner ring as a constant in ``x1``."""
        return self.ring.from_dict({(0,) + m: v for m, v in c.terms.items()})

    def _embed_params(self, c: PolyElement[T]) -> PolyElement[T]:
        """An element of ``K`` in this ring."""
        zeros = tuple(0 for _ in range(self.n_gens))
        return self.ring.from_dict({zeros + m: v for m, v in c.terms.items()})

    # PORTED FROM SYMPY: sympy/polys/densetools.py:dup_content, sympy/polys/euclidtools.py:dmp_content
    def _content(self, f: list[PolyElement[T]], inner: SparsePolynomialRing[T]) -> PolyElement[T]:
        if not f:
            return inner.zero
        if self.n_gens == 1:
            content = inner.zero
            for c in f:
                content = inner.gcd(content, c)
                if inner.is_one(content):
                    break
            return content
        content = f[0]
        for c in f[1:]:
            content = inner.gcd(content, c)
            if inner.is_one(content):
                break
        if isinstance(inner, IteratedPolynomialRing) and self.params.is_negative(inner.ground_LC(content)):
            return -content
        return content

    # PORTED FROM SYMPY: sympy/polys/densetools.py:dup_primitive, sympy/polys/euclidtools.py:dmp_primitive
    def _primitive(self, f: list[PolyElement[T]], inner: SparsePolynomialRing[T]
                   ) -> tuple[PolyElement[T], list[PolyElement[T]]]:
        content = self._content(f, inner)
        if not f or inner.is_one(content):
            return content, f
        return content, [inner.exquo(c, content) for c in f]

    # PORTED FROM SYMPY: sympy/polys/euclidtools.py:dup_rr_prs_gcd, dmp_rr_prs_gcd,
    # _dup_rr_trivial_gcd, _dmp_rr_trivial_gcd, _dmp_simplify_gcd
    def gcd(self, a: PolyElement[T], b: PolyElement[T]) -> PolyElement[T]:
        K = self.params
        if not a.terms and not b.terms:
            return self.zero
        if not a.terms:
            return b if K.is_nonnegative(self.ground_LC(b)) else -b
        if not b.terms:
            return a if K.is_nonnegative(self.ground_LC(a)) else -a
        inner = self._inner()
        F, G = self.to_univariate(a, inner), self.to_univariate(b, inner)
        if self.n_gens > 1:
            if self.is_one(a) or self.is_one(b):
                return self.one
            if len(F) == 1 or len(G) == 1:
                if len(F) == 1 and len(G) == 1:
                    left, right = F[0], G[0]
                elif len(F) == 1:
                    left, right = F[0], self._content(G, inner)
                else:
                    left, right = self._content(F, inner), G[0]
                return self._lift(inner.gcd(left, right))
        fc, F = self._primitive(F, inner)
        gc, G = self._primitive(G, inner)
        h = euclidtools.dup_subresultants(F, G, inner)[-1]
        c = inner.gcd(fc, gc)
        _, h = self._primitive(h, inner)
        if self.n_gens == 1:
            c = c * K.canonical_unit(h[0])
            return self.from_univariate([hc * c for hc in h])
        result = self.from_univariate([hc * c for hc in h])
        unit = K.canonical_unit(self.ground_LC(result))
        return result if K.is_one(unit) else result * self._embed_params(unit)

    # PORTED FROM SYMPY: sympy/polys/densetools.py:dmp_ground_content
    def ground_content(self, a: PolyElement[T]) -> PolyElement[T]:
        """The gcd in ``K`` of the coefficients in ``x1, ..., xn``, folded
        as SymPy folds it over the dense representation (the contents of
        the coefficients in ``x1`` first, zero coefficients included,
        stopping at ``1``): the gcd of ``QQ[y]`` is not associative."""
        K = self.params
        if not a.terms:
            return K.zero
        inner = self._inner()
        content = K.zero
        for c in self.to_univariate(a, inner):
            part = inner.ground_content(c) if isinstance(inner, IteratedPolynomialRing) else c
            content = K.gcd(content, part)
            if K.is_one(content):
                break
        return content

    # PORTED FROM SYMPY: sympy/polys/euclidtools.py:dup_rr_lcm, dmp_rr_lcm
    def lcm(self, a: PolyElement[T], b: PolyElement[T]) -> PolyElement[T]:
        K = self.params
        if not a.terms or not b.terms:
            return self.zero
        fc, gc = self.ground_content(a), self.ground_content(b)
        f = a if K.is_one(fc) else a.exquo(self._embed_params(fc))
        g = b if K.is_one(gc) else b.exquo(self._embed_params(gc))
        c = K.lcm(fc, gc)
        h = (f * g).exquo(self.gcd(f, g))
        if self.n_gens == 1:
            c = c * K.canonical_unit(self.ground_LC(h))
        return h * self._embed_params(c)
