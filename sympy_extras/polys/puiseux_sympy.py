r"""The SymPy face of the Puiseux expansions
(:mod:`sympy_extras.polys.puiseux`), and the steps which stay on SymPy.

:func:`puiseux_expansions`, :func:`expansions_in_field`,
:func:`adjoin_root` and the classes :class:`PuiseuxExpansion`,
:class:`LaurentSeries` and :class:`Embedding` take and return SymPy
objects (``Poly``, expressions, the number fields
:class:`~sympy.polys.domains.AlgebraicField` and their elements) as the
module did before it was native, around the native expansions: every
computation (the Newton polygons, the factorisations over number fields,
the primitive elements, the series arithmetic) is native, the face
converts.

The number fields are converted exactly, by their minimal polynomials
(an element is the list of its coordinates in the powers of the
generator in both): a field of SymPy is the native field with its
minimal polynomial and the generator converted to the core, and a native
field built by the expansions is SymPy's
``QQ.algebraic_field(rootof(N, k, radicals=True))`` for its minimal
polynomial ``N`` and ``k`` the index of its largest real root (``0``
when it has none), as before. The two directions are remembered, so that
the same field is always the same object on both sides.

What stays on SymPy: the field of a point given as a SymPy expression
(:func:`field_of`, since a point may be SymPy's ``CRootOf``, which the
core does not port), the choice of the irreducible factor of its
minimal polynomial over a number field of coefficients (SymPy's order of
the factors, as before), and the SymPy number fields of the results
(``CRootOf`` generators).

Examples
========

>>> from sympy import symbols, oo, sqrt
>>> from sympy_extras.polys.puiseux_sympy import puiseux_expansions
>>> x, y = symbols('x y')
>>> for b in puiseux_expansions(y**2 - x**3 - 1, x, y, oo, order=3):
...     print(b.ramification, b.conjugates, b.as_expr(x))
2 1 x**(3/2) + 1/(2*x**(3/2))
>>> [b] = puiseux_expansions(y**2 - 2 - x, x, y, 0, order=2)
>>> b.conjugates, b.field, b.as_expr(x)
(2, QQ<sqrt(2)>, sqrt(2)*x/4 + sqrt(2))
"""
from __future__ import annotations

from fractions import Fraction
from typing import Optional, Union

from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.polys.domains import QQ
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.domain import Domain
from sympy.polys.polytools import Poly
from sympy.polys.rootoftools import rootof

from sympy_extras._typing import DomainElement, ExprLike, as_expr
from sympy_extras.assumptions.intervals_sympy import core_expr, core_symbol
from sympy_extras.polys.domains.algebraicfield import AlgebraicField as NativeField

from . import puiseux as native

__all__ = ['PuiseuxExpansion', 'LaurentSeries', 'puiseux_expansions', 'expansions_in_field',
           'branches_at_infinity', 'adjoin_root', 'Embedding', 'field_of', 'minimal_polynomial_of',
           'in_subfield', 'generator', 'degree', 'coefficients_in', 'native_field', 'sympy_field',
           'native_number', 'sympy_number', 'native_curve']


# ---------------------------------------------------------------------------
# the number fields, both ways

#: the SymPy field of each native field, and the native field of each SymPy
#: field (the first one met: a native field and its SymPy field stay paired)
_TO_SYMPY: dict[NativeField, Domain] = {}
_TO_NATIVE: dict[Domain, NativeField] = {}


def _fraction(q: DomainElement) -> Fraction:
    return Fraction(int(q.numerator), int(q.denominator))


def _monic_modulus(K: AlgebraicField) -> list[Fraction]:
    """The minimal polynomial of the generator of ``K``, monic (SymPy's
    modulus may have integer coefficients)."""
    modulus = [_fraction(c) for c in K.mod.to_list()]
    return [c / modulus[0] for c in modulus]


def native_field(K: Domain) -> NativeField:
    """The native number field of the SymPy domain ``K`` (``QQ``, ``ZZ``
    or an :class:`~sympy.polys.domains.AlgebraicField`)."""
    if K.is_QQ or K.is_ZZ:
        return native.RATIONALS
    if not isinstance(K, AlgebraicField):
        raise ValueError(f"coefficients in {K} are not supported")
    found = _TO_NATIVE.get(K)
    if found is None:
        found = NativeField(_monic_modulus(K), core_expr(as_expr(K.ext.as_expr())))
        _TO_NATIVE[K] = found
        _TO_SYMPY.setdefault(found, K)
    return found


def sympy_field(L: NativeField) -> Domain:
    """The SymPy number field of the native field ``L``: ``QQ`` for the
    rationals, else ``QQ.algebraic_field`` of the largest real root of
    its minimal polynomial (written in radicals when SymPy's ``rootof``
    does), the first root when it has none."""
    if L.degree == 1:
        return QQ
    found = _TO_SYMPY.get(L)
    if found is None:
        T = Symbol('T')
        N = Poly([QQ(c.numerator, c.denominator) for c in L.minpoly], T, domain=QQ)
        real = N.count_roots()
        root = as_expr(rootof(N, real - 1 if real > 0 else 0, radicals=True))
        found = QQ.algebraic_field(root)
        if _monic_modulus(found) != L.minpoly:
            raise ValueError(f"the generator of {found} is not a root of {N}")
        _TO_SYMPY[L] = found
        _TO_NATIVE.setdefault(found, L)
    return found


def native_number(c: DomainElement, K: Domain) -> native.Number:
    """An element of the SymPy number field ``K`` as a native element."""
    if isinstance(K, AlgebraicField):
        return [_fraction(a) for a in c.to_list()]
    q = QQ.convert(c, K)
    return [_fraction(q)] if q else []


def sympy_number(c: native.Number, K: Domain) -> DomainElement:
    """A native element as an element of the SymPy number field ``K``."""
    if isinstance(K, AlgebraicField):
        return K([QQ(a.numerator, a.denominator) for a in c])
    return QQ(c[0].numerator, c[0].denominator) if c else QQ.zero


def coefficients_in(p: Poly, K: Domain) -> list[DomainElement]:
    """The coefficients of ``p`` (highest degree first) as elements of
    the number field ``K``, the domain of ``p`` being ``K`` or the
    rationals; SymPy's conversion between two number fields goes through
    expressions and numerical identification, which this avoids."""
    if p.domain == K:
        return list(p.rep.to_list())
    if p.domain.is_QQ or p.domain.is_ZZ:
        return [K.convert(c, QQ) for c in Poly(p.as_expr(), *p.gens, domain=QQ).rep.to_list()]
    raise ValueError(f"coefficients in {p.domain} are not elements of {K}")


def native_curve(F: Poly, x: Symbol, y: Symbol, K: Domain) -> native.Curve:
    """The polynomial ``F`` in ``x`` and ``y`` (or in ``x`` only), its
    coefficients rational or in the SymPy number field ``K``, as a native
    curve ``{i: {k: coefficient of x**k y**i}}``."""
    P: native.Curve = {}
    if F.gens == (x,):
        for (k,), c in F.as_dict(native=True).items():
            P.setdefault(0, {})[k] = _native_coefficient(c, F.domain, K)
        return P
    if F.gens == (y, x):
        G = F
    elif set(F.gens) == {x, y}:
        G = F.reorder(y, x)
    else:
        G = Poly(F.as_expr(), y, x, domain=F.domain)
    for (i, k), c in G.as_dict(native=True).items():
        P.setdefault(i, {})[k] = _native_coefficient(c, G.domain, K)
    return P


def _native_coefficient(c: DomainElement, domain: Domain, K: Domain) -> native.Number:
    if isinstance(domain, AlgebraicField):
        if domain != K:
            raise ValueError(f"coefficients in {domain} are not elements of {K}")
        return native_number(c, domain)
    if domain.is_QQ or domain.is_ZZ:
        q = QQ.convert(c, domain)
        return [_fraction(q)] if q else []
    raise ValueError(f"coefficients in {domain} are not supported")


# ---------------------------------------------------------------------------
# the face

class LaurentSeries:
    """A truncated Laurent series `\\sum_k c_k t^k` with coefficients in
    the SymPy number field ``K``, around the native series ``native``
    (:class:`sympy_extras.polys.puiseux.LaurentSeries`)."""

    __slots__ = ('K', 'native', '_terms')

    def __init__(self, K: Domain, terms: dict[int, DomainElement], order: Optional[int]) -> None:
        self.K: Domain = K
        self.native: native.LaurentSeries = native.LaurentSeries(
            native_field(K), {k: native_number(c, K) for k, c in terms.items()}, order)
        self._terms: Optional[dict[int, DomainElement]] = None

    @classmethod
    def wrap(cls, K: Domain, series: native.LaurentSeries) -> LaurentSeries:
        """The face of the native ``series``, ``K`` its SymPy field."""
        result = cls(K, {}, None)
        result.native = series
        return result

    @classmethod
    def constant(cls, K: Domain, c: DomainElement) -> LaurentSeries:
        return cls(K, {0: c}, None)

    @classmethod
    def monomial(cls, K: Domain, c: DomainElement, k: int) -> LaurentSeries:
        return cls(K, {k: c}, None)

    @property
    def order(self) -> Optional[int]:
        return self.native.order

    @property
    def terms(self) -> dict[int, DomainElement]:
        if self._terms is None:
            self._terms = {k: sympy_number(c, self.K) for k, c in self.native.terms.items()}
        return self._terms

    def _new(self, series: native.LaurentSeries) -> LaurentSeries:
        return LaurentSeries.wrap(self.K, series)

    def valuation(self) -> Optional[int]:
        return self.native.valuation()

    def is_zero(self) -> bool:
        return self.native.is_zero()

    def coefficient(self, k: int) -> DomainElement:
        return sympy_number(self.native.coefficient(k), self.K)

    def __add__(self, other: LaurentSeries) -> LaurentSeries:
        return self._new(self.native + other.native)

    def __neg__(self) -> LaurentSeries:
        return self._new(-self.native)

    def __sub__(self, other: LaurentSeries) -> LaurentSeries:
        return self._new(self.native - other.native)

    def scaled(self, c: DomainElement) -> LaurentSeries:
        return self._new(self.native.scaled(native_number(c, self.K)))

    def shifted(self, k: int) -> LaurentSeries:
        return self._new(self.native.shifted(k))

    def __mul__(self, other: LaurentSeries) -> LaurentSeries:
        return self._new(self.native * other.native)

    def truncated(self, order: int) -> LaurentSeries:
        return self._new(self.native.truncated(order))

    def power(self, n: int) -> LaurentSeries:
        return self._new(self.native.power(n))

    def inverse(self, order: int) -> LaurentSeries:
        return self._new(self.native.inverse(order))

    def substituted(self, q: int) -> LaurentSeries:
        return self._new(self.native.substituted(q))

    def converted(self, embedding: Embedding) -> LaurentSeries:
        return LaurentSeries.wrap(embedding.target, self.native.converted(embedding.native))

    def as_expr(self, t: Expr, exponent_scale: int = 1, reciprocal: bool = False) -> Expr:
        """`\\sum_k c_k t^{k/\\text{exponent\\_scale}}` as an expression
        (`\\sum_k c_k t^{-k/\\text{exponent\\_scale}}` with ``reciprocal``)."""
        sign = -1 if reciprocal else 1
        return as_expr(Add(*[self.K.to_sympy(c) * t**(sign * Rational(k, exponent_scale))
                             for k, c in sorted(self.terms.items())]))

    def __repr__(self) -> str:
        return f"LaurentSeries({dict(sorted(self.terms.items()))}, order={self.order})"


class Embedding:
    """The embedding of a number field ``source`` into a number field
    ``target`` containing it (SymPy domains), given by the image of the
    generator of the source, around the native embedding ``native``."""

    __slots__ = ('source', 'target', 'native')

    def __init__(self, source: Domain, target: Domain, image: Optional[DomainElement] = None) -> None:
        self.source: Domain = source
        self.target: Domain = target
        self.native: native.Embedding = native.Embedding(
            native_field(source), native_field(target), None if image is None else native_number(image, target))

    @classmethod
    def wrap(cls, embedding: native.Embedding) -> Embedding:
        result = cls(QQ, QQ)
        result.source = sympy_field(embedding.source)
        result.target = sympy_field(embedding.target)
        result.native = embedding
        return result

    @property
    def image(self) -> Optional[DomainElement]:
        image = self.native.image
        return None if image is None else sympy_number(image, self.target)

    def __call__(self, c: DomainElement) -> DomainElement:
        if self.source == self.target:
            return c
        return sympy_number(self.native(native_number(c, self.source)), self.target)

    def then(self, other: Embedding) -> Embedding:
        """The composition with ``other``, an embedding of ``target``."""
        if other.source != self.target:
            raise ValueError("the embeddings do not compose")
        return Embedding.wrap(self.native.then(other.native))


def generator(K: Domain) -> DomainElement:
    """The generator of the number field ``K`` (one for the rationals)."""
    if isinstance(K, AlgebraicField):
        return K([QQ.one, QQ.zero])
    return K.one


def degree(K: Domain) -> int:
    """The degree of the number field ``K`` over the rationals."""
    return int(K.mod.degree()) if isinstance(K, AlgebraicField) else 1


def adjoin_root(K: Domain, psi: list[DomainElement]) -> tuple[Domain, Embedding, DomainElement]:
    """The field ``L = K(u)`` for a root ``u`` of the polynomial ``psi``
    (coefficients in ``K``, highest degree first) irreducible over
    ``K``, with the embedding of ``K`` into ``L`` and ``u``; exact: no
    root is evaluated numerically.

    >>> from sympy import sqrt, QQ
    >>> from sympy_extras.polys.puiseux_sympy import adjoin_root
    >>> K = QQ.algebraic_field(sqrt(2))
    >>> L, into, u = adjoin_root(K, [K.one, K.zero, -K.from_sympy(sqrt(2))])
    >>> L.to_sympy(u*u) == L.to_sympy(into(K.from_sympy(sqrt(2))))
    True
    """
    L, into, u = native.adjoin_root(native_field(K), [native_number(c, K) for c in psi])
    L_ = sympy_field(L)
    return L_, Embedding.wrap(into), sympy_number(u, L_)


def field_of(point: Expr) -> tuple[Domain, DomainElement]:
    """The number field generated by an algebraic number given as an
    expression, and the number as its element (SymPy's
    ``QQ.algebraic_field``: the point may be a ``CRootOf``)."""
    if isinstance(point, Rational):
        return QQ, QQ.convert(point)
    K = QQ.algebraic_field(point)
    if K.ext == point:
        return K, generator(K)
    return K, K.from_sympy(point)


def minimal_polynomial_of(c: DomainElement, K: Domain, T: Symbol) -> Poly:
    """The minimal polynomial over the rationals of an element of the
    number field ``K``, exactly (primitive with integer coefficients and
    a positive leading coefficient, in ``T``, over ``QQ``; ``T - c`` for a
    rational ``c``)."""
    if not isinstance(K, AlgebraicField):
        return Poly(T - QQ.to_sympy(QQ.convert(c, K)), T, domain=QQ)
    m = native.minimal_polynomial_of(native_number(c, K), native_field(K))
    p = Poly([QQ(a.numerator, a.denominator) for a in m], T, domain=QQ)
    _, integral = p.clear_denoms()
    return Poly(integral.primitive()[1].as_expr(), T, domain=QQ)


def in_subfield(c: DomainElement, L: Domain, into: Embedding) -> Optional[DomainElement]:
    """``c`` of ``L`` as an element of the subfield ``into.source``
    (``into`` its embedding into ``L``), ``None`` when it is not in it."""
    found = native.in_subfield(native_number(c, L), native_field(L), into.native)
    return None if found is None else sympy_number(found, into.source)


class PuiseuxExpansion:
    """One branch of the curve `F(x, y) = 0` above ``point`` (an
    expression, or ``S.Infinity``), the face of the native branch
    ``native``: `y = \\sum_k c_k t^k` with `x - \\text{point} =
    \\text{scale} \\cdot t^e` (`x = 1/(\\text{scale}\\, t^e)` at infinity),
    the coefficients in the SymPy number field ``field`` and known for `k
    <` ``order``; it stands for ``conjugates`` branches, conjugate under
    the automorphisms of ``field`` over the field of the point.
    ``embedding`` maps the coefficient field of the curve (extended by
    the point) into ``field``.

    >>> from sympy import symbols
    >>> from sympy_extras.polys.puiseux_sympy import puiseux_expansions
    >>> x, y = symbols('x y')
    >>> [b.as_expr(x) for b in puiseux_expansions(y**2 - x**3 - x**2, x, y, 0, order=4)]
    [x**3/8 - x**2/2 - x, -x**3/8 + x**2/2 + x]
    >>> branches = puiseux_expansions(y**3 - x**2, x, y, 0, order=2)
    >>> len(branches), branches[0].ramification, branches[0].as_expr(x)
    (1, 3, x**(2/3))
    """

    __slots__ = ('native', 'point', 'x', 'y', 'field', 'series', 'embedding')

    def __init__(self, branch: native.PuiseuxExpansion, point: Expr, x: Symbol, y: Symbol) -> None:
        self.native: native.PuiseuxExpansion = branch
        self.point: Expr = point
        self.x: Symbol = x
        self.y: Symbol = y
        self.field: Domain = sympy_field(branch.field)
        self.series: LaurentSeries = LaurentSeries.wrap(self.field, branch.series)
        self.embedding: Embedding = Embedding.wrap(branch.embedding)

    @property
    def point_element(self) -> Optional[DomainElement]:
        """The point as an element of ``field`` (``None`` at infinity)."""
        element = self.native.point_element
        return None if element is None else sympy_number(element, self.field)

    @property
    def ramification(self) -> int:
        return self.native.ramification

    @property
    def conjugates(self) -> int:
        return self.native.conjugates

    @property
    def scale(self) -> DomainElement:
        return sympy_number(self.native.scale, self.field)

    @property
    def order(self) -> Optional[int]:
        return self.series.order

    @property
    def at_infinity(self) -> bool:
        return self.point is S.Infinity

    def as_expr(self, x: Optional[Expr] = None) -> Expr:
        """The series in `x` (the coefficients as expressions)."""
        x_ = self.x if x is None else x
        scale = self.field.to_sympy(self.scale)
        if self.at_infinity:
            return self.series.as_expr(as_expr(scale * x_), self.ramification, reciprocal=True)
        return self.series.as_expr(as_expr((x_ - self.point) / scale), self.ramification)

    def x_series(self) -> LaurentSeries:
        """`x` as a series in `t`."""
        return LaurentSeries.wrap(self.field, self.native.x_series())

    def evaluate(self, p: Poly, order: Optional[int] = None) -> LaurentSeries:
        """The polynomial ``p`` in ``x`` and ``y`` (a :class:`Poly` whose
        coefficients are rational, or in the coefficient field of the
        curve) along the branch, as a series in `t`."""
        source = self.embedding.source
        if isinstance(p.domain, AlgebraicField) and p.domain != source:
            raise ValueError(f"coefficients in {p.domain} cannot be evaluated along a branch over {self.field}")
        curve = native_curve(p, self.x, self.y, source)
        return LaurentSeries.wrap(self.field, self.native.evaluate(curve, order))

    def __repr__(self) -> str:
        return (f"PuiseuxExpansion(point={self.point}, ramification={self.ramification}, "
                f"conjugates={self.conjugates}, series={self.as_expr()})")


def expansions_in_field(F: Poly, x: Symbol, y: Symbol, K: Domain, into: Embedding,
                        point: Optional[DomainElement], order: int) -> list[PuiseuxExpansion]:
    """The expansions of `F(x, y) = 0` (``F`` with rational coefficients,
    or in the source of ``into``) above the element ``point`` of the
    number field ``K`` (``None`` for infinity), ``into`` the embedding of
    the coefficient field of ``F`` into ``K``: the conjugacy classes are
    taken over ``K``, and every expansion carries the embedding of
    ``K`` into its field."""
    K_ = native_field(K)
    curve = native_curve(F, x, y, into.source)
    point_ = None if point is None else native_number(point, K)
    point_expr = S.Infinity if point is None else as_expr(K.to_sympy(point))
    branches = native.expansions_in_field(curve, core_symbol(x), core_symbol(y), K_, into.native, point_, order)
    return [PuiseuxExpansion(b, point_expr, x, y) for b in branches]


def puiseux_expansions(F: Union[ExprLike, Poly], x: Symbol, y: Symbol, point: ExprLike = 0, order: int = 4
                       ) -> list[PuiseuxExpansion]:
    """The Puiseux expansions of the roots `y` of `F(x, y) = 0` above
    `x = ` ``point`` (an algebraic number, or ``oo``), one per conjugacy
    class, with the coefficients of `(x - a)^{k/e}` known for `k/e <`
    ``order``; a :class:`~sympy.polys.polytools.Poly` keeps its
    coefficient domain (a number field over which the conjugacy classes
    are then taken).

    >>> from sympy import symbols, oo, sqrt
    >>> from sympy_extras.polys.puiseux_sympy import puiseux_expansions
    >>> x, y = symbols('x y')
    >>> for b in puiseux_expansions(y**2 - x**3 - 1, x, y, oo, order=3):
    ...     print(b.ramification, b.conjugates, b.as_expr(x))
    2 1 x**(3/2) + 1/(2*x**(3/2))

    A branch stands for its conjugates: the four roots of `y^4 = x` above
    the origin are the conjugates of one branch,

    >>> [b] = puiseux_expansions(y**4 - x, x, y, 0, order=2)
    >>> b.ramification, b.conjugates, b.as_expr(x)
    (4, 1, x**(1/4))

    and the two roots of `y^2 = 2` above `x = 0` of one branch defined
    over `\\mathbb{Q}(\\sqrt 2)`:

    >>> [b] = puiseux_expansions(y**2 - 2 - x, x, y, 0, order=2)
    >>> b.conjugates, b.field, b.as_expr(x)
    (2, QQ<sqrt(2)>, sqrt(2)*x/4 + sqrt(2))
    """
    FP = F if isinstance(F, Poly) else Poly(as_expr(F), y, x)
    point_ = as_expr(point)
    domain = FP.domain
    if domain.is_ZZ:
        FP = FP.set_domain(QQ)
        domain = QQ
    if not (domain.is_QQ or isinstance(domain, AlgebraicField)):
        raise ValueError(f"coefficients in {domain} are not supported")
    if point_ is S.Infinity:
        return expansions_in_field(FP, x, y, domain, Embedding(domain, domain), None, order)
    if isinstance(point_, Rational):
        return expansions_in_field(FP, x, y, domain, Embedding(domain, domain), domain.convert(point_, QQ), order)
    if isinstance(domain, AlgebraicField):
        # the point adjoined to the field of the coefficients
        T = Symbol('T')
        K_point, element = field_of(point_)
        m = minimal_polynomial_of(element, K_point, T)
        factor = Poly(m.as_expr(), T, domain=domain).factor_list()[1][0][0]
        K, into, a = adjoin_root(domain, coefficients_in(factor, domain))
        return expansions_in_field(FP, x, y, K, into, a, order)
    K, a = field_of(point_)
    return expansions_in_field(FP, x, y, K, Embedding(QQ, K), a, order)


def branches_at_infinity(F: Union[ExprLike, Poly], x: Symbol, y: Symbol, order: int = 4) -> list[PuiseuxExpansion]:
    """The expansions above `x = \\infty`, :func:`puiseux_expansions` with
    ``point=oo``."""
    return puiseux_expansions(F, x, y, S.Infinity, order)

