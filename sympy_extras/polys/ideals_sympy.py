r"""The SymPy face of :mod:`sympy_extras.polys.ideals`, and the steps which
stay on SymPy.

:class:`Ideal` takes SymPy expressions or ``Poly`` generators, SymPy
symbols, a SymPy coefficient domain (inferred from the coefficients, with
``extension=True``, when omitted) and a SymPy monomial order (a name, or
one of :mod:`sympy_extras.polys.orderings`); its Gröbner bases are SymPy
``Poly`` objects, its ring elements those of SymPy's
:func:`~sympy.polys.rings.ring`, and its results SymPy expressions,
``Poly`` objects and matrices, as before the native port. The work is done
by the native :class:`sympy_extras.polys.ideals.Ideal`, on the typed
domains of :mod:`sympy_extras.polys.domains`: the coefficients are
converted term by term (``QQ`` and ``ZZ`` to ``QQ``, ``GF(p)``, the
algebraic fields ``QQ<a>``, ``QQ_I``, the fields of rational functions
over ``QQ`` or ``ZZ``), the symbols and the orders by
:mod:`sympy_extras.interop`.

The decompositions of positive-dimensional ideals through regular chains
are those of the native :mod:`sympy_extras.polys.idealdecomposition`, run
on the native ideal; the rational univariate representation is that of
:mod:`sympy_extras.polys.rur_sympy` (the native
:mod:`sympy_extras.polys.rur` with SymPy objects around it).

What stays here, on SymPy: the inference of the coefficient domain.

Examples
========

>>> from sympy.abc import x, y, z
>>> from sympy_extras.polys.ideals_sympy import Ideal
>>> I = Ideal([x*z - y**2, x**2 - y*z], x, y, z)
>>> I.dimension(), I.degree()
(1, 4)
>>> I.eliminate([x])
Ideal([y**4 - y*z**3], y, z)

The variety of ``I`` consists of three lines through the origin (where
``x**3 == y**3``) and the ``z`` axis; saturating by ``y`` removes the axis:

>>> I.saturate(Ideal([y], x, y, z))
Ideal([x**2 - y*z, x*y - z**2, -x*z + y**2], x, y, z)
>>> _.degree()
3
>>> J = Ideal([x**2 - 1, y**2 - 1], x, y)
>>> J.vector_space_dimension(), J.is_radical(), J.is_prime()
(4, True, False)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from fractions import Fraction
from typing import Generic, Iterable, Iterator, Optional, Sequence, TypeVar, Union

from sympy.core.expr import Expr
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import binomial
from sympy.matrices.dense import MutableDenseMatrix
from sympy.polys.domains import QQ
from sympy.polys.domains.algebraicfield import AlgebraicField as SymPyAlgebraicField
from sympy.polys.domains.domain import Domain
from sympy.polys.domains.finitefield import FiniteField as SymPyFiniteField
from sympy.polys.domains.fractionfield import FractionField as SymPyFractionField
from sympy.polys.matrices import DomainMatrix
from sympy.polys.orderings import MonomialOrder, grevlex, grlex, lex
from sympy.polys.polytools import Poly
from sympy.polys.rings import PolyElement, PolyRing, ring as _ring

from sympy_extras._typing import DomainElement, Monomial, OrderSpec, as_expr, as_symbol, free_symbols, sorted_symbols
from sympy_extras.core.expr import as_expr as core_as_expr
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.interop import from_sympy as core_from_sympy, to_sympy as core_to_sympy

from . import idealdecomposition as decomposition
from . import ideals as native
from . import rings as typed
from .domains.algebraicfield import AlgebraicField
from .domains.domain import Domain as CoreDomain
from .domains.finitefield import FiniteField
from .domains.gaussiandomains import QQ_I as CoreQQ_I
from .domains.integerring import ZZ as CoreZZ
from .domains.rationalfield import QQ as CoreQQ, RationalField
from .domains.sparsefractionfield import SparseFractionField
from .domains.sparsepolynomialring import RationalPolynomialRing
from .hilbert import hilbert_numerator
from .orderings import BlockOrder, WeightOrder, as_order

__all__ = ['Ideal', 'hilbert_numerator', 'separating_forms', 'core_order', 'sympy_order', 'native_ideal']

T = TypeVar('T')

#: the terms of a polynomial with coefficients in a SymPy domain
SymPyTerms = dict[Monomial, DomainElement]


def separating_forms(variables: Sequence[Symbol], points: int) -> Iterator[Expr]:
    r"""Linear forms $x_1 + c_2 x_2 + \cdots + c_n x_n$ with small integer
    coefficients, by increasing height, among which one takes different
    values at any ``points`` distinct points
    (:func:`sympy_extras.polys.ideals.separating_forms`, on SymPy
    symbols).

    >>> from sympy.abc import x, y, z
    >>> from sympy_extras.polys.ideals_sympy import separating_forms
    >>> list(separating_forms([x, y], 3))
    [x, x + y, x - y, x + 2*y, x - 2*y]
    >>> [f for f, _ in zip(separating_forms([x, y, z], 10), range(5))]
    [x, x + z, x - z, x + y, x + y + z]
    """
    if len(variables) < 2:
        yield variables[0] if variables else S.Zero
        return
    for tail in native.separating_coefficients(len(variables), points):
        form: Expr = variables[0]
        for c, v in zip(tail, variables[1:]):
            form += c*v
        yield form


# ---------------------------------------------------------------------------
# orders, symbols and coefficients

def core_order(order: MonomialOrder, nvars: int) -> typed.MonomialOrder:
    """The native order of a SymPy order (``lex``, ``grlex``, ``grevlex``,
    the weight and block orders of :mod:`sympy_extras.polys.orderings`;
    any other order through the flattened key of
    :func:`sympy_extras.polys.interop.typed_order`)."""
    if isinstance(order, WeightOrder):
        return typed.WeightOrder(order.weights, core_order(order.tail, nvars))
    if isinstance(order, BlockOrder):
        return typed.BlockOrder([(core_order(o, size), size) for o, size in order.blocks])
    if order == lex:
        return typed.lex
    if order == grlex:
        return typed.grlex
    if order == grevlex:
        return typed.grevlex
    from .interop import typed_order
    return typed_order(order, nvars)


def sympy_order(order: typed.MonomialOrder) -> MonomialOrder:
    """The SymPy order of a native order (``NotImplementedError`` for the
    orders made from a key only)."""
    if isinstance(order, typed.WeightOrder):
        return WeightOrder(order.weights, sympy_order(order.tail))
    if isinstance(order, typed.BlockOrder):
        return BlockOrder([(sympy_order(o), size) for o, size in order.blocks])
    if order == typed.lex:
        return lex
    if order == typed.grlex:
        return grlex
    if order == typed.grevlex:
        return grevlex
    raise NotImplementedError("no SymPy order for %s" % (order,))


def _core_symbol(s: Symbol) -> CoreSymbol:
    result = core_from_sympy(s)
    if not isinstance(result, CoreSymbol):
        raise TypeError("%s is not a symbol" % (s,))
    return result


def _sympy_symbol(s: CoreSymbol) -> Symbol:
    return as_symbol(core_to_sympy(s))


def _rational(c: DomainElement) -> Fraction:
    return Fraction(int(QQ.numer(c)), int(QQ.denom(c)))


class _Coefficients(Generic[T]):
    """The conversions between the elements of a SymPy domain and those of
    the native domain ``K`` standing for it."""

    def __init__(self, K: CoreDomain[T], domain: Domain) -> None:
        self.K: CoreDomain[T] = K
        self.domain: Domain = domain
        self.rational: bool = bool(domain.is_QQ or domain.is_ZZ)
        self.finite: Optional[SymPyFiniteField] = domain if isinstance(domain, SymPyFiniteField) else None
        self.algebraic: Optional[AlgebraicField] = None
        if isinstance(K, AlgebraicField) and isinstance(domain, SymPyAlgebraicField) and \
                [_rational(c) for c in domain.mod.to_list()] == K.minpoly:
            # the same generator: the coefficient lists are those of SymPy
            self.algebraic = K

    def from_sympy(self, c: DomainElement) -> T:
        if self.rational:
            return self.K.convert(_rational(c), CoreQQ)
        if self.finite is not None:
            return self.K.convert(int(self.finite.to_int(c)), CoreZZ)
        if self.algebraic is not None:
            return self.K.convert([_rational(a) for a in c.to_list()], self.algebraic)
        return self.K.from_expr(core_as_expr(core_from_sympy(self.domain.to_sympy(c))))

    def to_sympy(self, c: T) -> DomainElement:
        if self.rational and isinstance(c, Fraction):
            return self.domain(c.numerator, c.denominator) if self.domain.is_QQ else self.domain(c.numerator)
        if self.finite is not None and isinstance(c, int):
            return self.domain(c)
        if self.algebraic is not None and isinstance(c, list):
            return self.domain.new([QQ(a.numerator, a.denominator) for a in c if isinstance(a, Fraction)])
        return self.domain.from_sympy(as_expr(core_to_sympy(self.K.to_expr(c))))

    def terms_in(self, f: SymPyTerms) -> native.Terms[T]:
        return {tuple(m): self.from_sympy(c) for m, c in f.items()}

    def terms_out(self, f: native.Terms[T]) -> SymPyTerms:
        return {m: self.to_sympy(c) for m, c in f.items()}


def _sympy_domain(K: CoreDomain[T]) -> Domain:
    """The SymPy domain of a native coefficient field."""
    from sympy.polys.domains import GF, QQ_I
    if isinstance(K, RationalField):
        return QQ
    if isinstance(K, FiniteField):
        return GF(K.p)
    if isinstance(K, AlgebraicField):
        return QQ.algebraic_field(as_expr(core_to_sympy(K.alpha_expr)))
    if K == CoreQQ_I:
        return QQ_I
    if isinstance(K, SparseFractionField) and isinstance(K.ground, RationalField):
        from sympy import symbols
        return QQ.frac_field(*symbols(K.names))
    raise NotImplementedError("no SymPy domain for %s" % (K,))


# ---------------------------------------------------------------------------
# the native ideal behind the face

class _Backend(ABC):
    """The native ideal of a face, its coefficients converted to and from
    the SymPy domain (the type of the native coefficients is that of the
    implementation, :class:`_TypedBackend`)."""

    @abstractmethod
    def gens(self) -> list[SymPyTerms]: ...

    @abstractmethod
    def basis(self, order: MonomialOrder) -> list[SymPyTerms]: ...

    @abstractmethod
    def converted_basis(self, order: MonomialOrder) -> Optional[list[SymPyTerms]]: ...

    @abstractmethod
    def set_basis(self, basis: Sequence[SymPyTerms]) -> None: ...

    @abstractmethod
    def normal_form(self, f: SymPyTerms, order: MonomialOrder) -> SymPyTerms: ...

    @abstractmethod
    def is_whole_ring(self) -> bool: ...

    @abstractmethod
    def is_zero(self) -> bool: ...

    @abstractmethod
    def eliminate(self, positions: Sequence[int]) -> _Backend: ...

    @abstractmethod
    def intersect(self, other: Sequence[SymPyTerms]) -> _Backend: ...

    @abstractmethod
    def quotient(self, other: Sequence[SymPyTerms]) -> _Backend: ...

    @abstractmethod
    def saturate(self, other: Sequence[SymPyTerms]) -> _Backend: ...

    @abstractmethod
    def radical_contains(self, f: SymPyTerms) -> bool: ...

    @abstractmethod
    def leading_monomials(self, order: MonomialOrder) -> list[Monomial]: ...

    @abstractmethod
    def hilbert_data(self) -> tuple[list[int], int]: ...

    @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    def standard_exponents(self, order: MonomialOrder) -> list[Monomial]: ...

    @abstractmethod
    def multiplication_matrix(self, f: SymPyTerms, order: MonomialOrder) -> list[list[DomainElement]]: ...

    @abstractmethod
    def minimal_polynomial(self, f: SymPyTerms) -> list[DomainElement]: ...

    @abstractmethod
    def radical(self) -> _Backend: ...

    @abstractmethod
    def reduced(self) -> _Backend: ...

    @abstractmethod
    def chain_radical(self) -> _Backend: ...

    @abstractmethod
    def equidimensional_parts(self) -> list[_Backend]: ...

    @abstractmethod
    def minimal_primes(self) -> list[_Backend]: ...

    @abstractmethod
    def primary_decomposition(self) -> list[tuple[_Backend, _Backend]]: ...

    @abstractmethod
    def is_primary(self) -> bool: ...

    @abstractmethod
    def independent_set(self) -> list[int]: ...

    @abstractmethod
    def is_maximal(self) -> bool: ...

    @abstractmethod
    def change_order(self, order: MonomialOrder) -> list[SymPyTerms]: ...


class _TypedBackend(_Backend, Generic[T]):
    """The native ideal (``ideal``) and the conversions of its
    coefficients."""

    def __init__(self, ideal: native.Ideal[T], coefficients: _Coefficients[T]) -> None:
        self.ideal: native.Ideal[T] = ideal
        self.coefficients: _Coefficients[T] = coefficients
        self._orders: dict[MonomialOrder, typed.MonomialOrder] = {}

    def _order(self, order: MonomialOrder) -> typed.MonomialOrder:
        found = self._orders.get(order)
        if found is None:
            found = self._orders[order] = core_order(order, len(self.ideal.symbols))
        return found

    def _wrap(self, ideal: native.Ideal[T]) -> _Backend:
        return _TypedBackend(ideal, self.coefficients)

    def _out(self, polys: Sequence[typed.PolyElement[T]]) -> list[SymPyTerms]:
        return [self.coefficients.terms_out(g.terms) for g in polys]

    def _in(self, f: SymPyTerms) -> typed.PolyElement[T]:
        return typed.PolyElement(self.coefficients.terms_in(f), self.ideal.ring())

    def _ideal_of(self, polys: Sequence[SymPyTerms]) -> native.Ideal[T]:
        I = self.ideal
        return native.Ideal.from_terms([self.coefficients.terms_in(f) for f in polys], I.symbols, I.domain, I.order)

    def gens(self) -> list[SymPyTerms]:
        return self._out(self.ideal.gens)

    def basis(self, order: MonomialOrder) -> list[SymPyTerms]:
        return self._out(self.ideal._basis(self._order(order)))

    def converted_basis(self, order: MonomialOrder) -> Optional[list[SymPyTerms]]:
        found = self.ideal._converted_basis(self._order(order))
        return None if found is None else self._out(found)

    def set_basis(self, basis: Sequence[SymPyTerms]) -> None:
        self.ideal._set_basis([self.coefficients.terms_in(g) for g in basis])

    def normal_form(self, f: SymPyTerms, order: MonomialOrder) -> SymPyTerms:
        typed_order = self._order(order)
        g = typed.PolyElement(self.coefficients.terms_in(f), self.ideal.ring(typed_order))
        return self.coefficients.terms_out(native.normal_form(g, self.ideal._basis(typed_order)).terms)

    def is_whole_ring(self) -> bool:
        return self.ideal.is_whole_ring()

    def is_zero(self) -> bool:
        return self.ideal.is_zero()

    def eliminate(self, positions: Sequence[int]) -> _Backend:
        return self._wrap(self.ideal.eliminate([self.ideal.symbols[i] for i in positions]))

    def intersect(self, other: Sequence[SymPyTerms]) -> _Backend:
        return self._wrap(self.ideal.intersect(self._ideal_of(other)))

    def quotient(self, other: Sequence[SymPyTerms]) -> _Backend:
        return self._wrap(self.ideal.quotient(self._ideal_of(other)))

    def saturate(self, other: Sequence[SymPyTerms]) -> _Backend:
        return self._wrap(self.ideal.saturate(self._ideal_of(other)))

    def radical_contains(self, f: SymPyTerms) -> bool:
        return self.ideal.radical_contains(self._in(f))

    def leading_monomials(self, order: MonomialOrder) -> list[Monomial]:
        return self.ideal.leading_monomials(self._order(order))

    def hilbert_data(self) -> tuple[list[int], int]:
        return self.ideal.hilbert_data()

    def dimension(self) -> int:
        return self.ideal.dimension()

    def standard_exponents(self, order: MonomialOrder) -> list[Monomial]:
        return self.ideal.standard_exponents(self._order(order))

    def multiplication_matrix(self, f: SymPyTerms, order: MonomialOrder) -> list[list[DomainElement]]:
        typed_order = self._order(order)
        g = typed.PolyElement(self.coefficients.terms_in(f), self.ideal.ring(typed_order))
        rows = self.ideal.multiplication_matrix(g, typed_order)
        return [[self.coefficients.to_sympy(c) for c in row] for row in rows]

    def minimal_polynomial(self, f: SymPyTerms) -> list[DomainElement]:
        p = self.ideal.minimal_polynomial(self._in(f))
        return [self.coefficients.to_sympy(c) for c in p.rep]

    def radical(self) -> _Backend:
        return self._wrap(self.ideal.radical())

    def reduced(self) -> _Backend:
        return self._wrap(self.ideal.reduced())

    def chain_radical(self) -> _Backend:
        return self._wrap(decomposition.radical(self.ideal))

    def equidimensional_parts(self) -> list[_Backend]:
        return [self._wrap(I) for I in decomposition.equidimensional_parts(self.ideal)]

    def minimal_primes(self) -> list[_Backend]:
        return [self._wrap(I) for I in decomposition.minimal_primes(self.ideal)]

    def primary_decomposition(self) -> list[tuple[_Backend, _Backend]]:
        return [(self._wrap(Q), self._wrap(P)) for Q, P in decomposition.primary_decomposition(self.ideal)]

    def is_primary(self) -> bool:
        return decomposition.is_primary(self.ideal)

    def independent_set(self) -> list[int]:
        return [self.ideal.symbols.index(s) for s in decomposition.independent_set(self.ideal)]

    def is_maximal(self) -> bool:
        return self.ideal.is_maximal()

    def change_order(self, order: MonomialOrder) -> list[SymPyTerms]:
        return self._out(self.ideal.change_order(self._order(order)))


def _typed_backend(K: CoreDomain[T], domain: Domain, symbols: Sequence[Symbol], order: MonomialOrder,
                   gens: Sequence[SymPyTerms]) -> _Backend:
    coefficients = _Coefficients(K, domain)
    ideal = native.Ideal.from_terms([coefficients.terms_in(g) for g in gens], [_core_symbol(s) for s in symbols], K,
                                    core_order(order, len(symbols)))
    return _TypedBackend(ideal, coefficients)


def _backend(symbols: Sequence[Symbol], domain: Domain, order: MonomialOrder, gens: Sequence[SymPyTerms]) -> _Backend:
    """The native ideal generated by ``gens`` (terms over the SymPy
    domain), on the native domain standing for ``domain``."""
    if domain.is_QQ or domain.is_ZZ:
        return _typed_backend(CoreQQ, domain, symbols, order, gens)
    if isinstance(domain, SymPyFiniteField):
        return _typed_backend(FiniteField(int(domain.characteristic())), domain, symbols, order, gens)
    if isinstance(domain, SymPyAlgebraicField):
        minpoly = [_rational(c) for c in domain.mod.to_list()]
        alpha = core_as_expr(core_from_sympy(domain.ext.as_expr()))
        return _typed_backend(AlgebraicField(minpoly, alpha), domain, symbols, order, gens)
    if domain.is_QQ_I or domain.is_ZZ_I:
        return _typed_backend(CoreQQ_I, domain, symbols, order, gens)
    if isinstance(domain, SymPyFractionField) and (domain.field.domain.is_QQ or domain.field.domain.is_ZZ):
        field = SparseFractionField(RationalPolynomialRing([str(s) for s in domain.field.symbols]))
        return _typed_backend(field, domain, symbols, order, gens)
    raise NotImplementedError("ideals over %s are not supported" % (domain,))


# ---------------------------------------------------------------------------
# the face

class Ideal:
    """An ideal of a polynomial ring over a field, given by generators
    (the native :class:`sympy_extras.polys.ideals.Ideal` on SymPy objects).

    Parameters
    ==========

    gens : iterable of Expr or Poly
        The generators.
    symbols : Symbols
        The variables of the polynomial ring (all the free symbols of the
        generators if omitted).
    domain : Domain, optional
        The coefficient field; ``QQ`` (or the field of fractions of the
        coefficients) by default. Integers are converted to rationals.
    order : str or MonomialOrder
        The order of the Gröbner basis used by default (``'grevlex'``).

    Gröbner bases are computed lazily, by Faugère's F4
    (:mod:`sympy_extras.polys.f4`) or the modular algorithm
    (:mod:`sympy_extras.polys.modulargroebner`), and cached per order.
    """

    def __init__(self, gens: Iterable[Union[Expr, Poly]], *symbols: Symbol,
                 domain: Optional[Domain] = None, order: OrderSpec = 'grevlex') -> None:
        exprs = [sympify(g) for g in gens]
        polys = [g.as_expr() if isinstance(g, Poly) else g for g in exprs]
        variables: list[Symbol]
        if symbols:
            variables = [as_symbol(s) for s in symbols]
        else:
            found: set[Symbol] = set()
            for p in polys:
                found |= free_symbols(p)
            variables = sorted_symbols(found)
            if not variables and polys:
                raise ValueError("no variables: give the symbols of the ring")
        self.symbols: tuple[Symbol, ...] = tuple(variables)
        if domain is None:
            polys_ = [Poly(p, *self.symbols, extension=True) for p in polys] if polys else []
            domains = [p.domain for p in polys_]
            if domains:
                domain = domains[0]
                for d in domains[1:]:
                    domain = domain.unify(d)
            else:
                domain = QQ
        if not domain.is_Field:
            domain = domain.get_field()
        self.domain: Domain = domain
        self.order: MonomialOrder = as_order(order)
        self._rings: dict[object, PolyRing] = {}
        self._bases: dict[object, list[PolyElement]] = {}
        R = self.ring()
        self.gens: list[PolyElement] = [g for g in (R.from_expr(p) for p in polys) if g]
        self._native: _Backend = _backend(self.symbols, self.domain, self.order, [dict(g) for g in self.gens])

    @classmethod
    def _from_backend(cls, backend: _Backend, symbols: Sequence[Symbol], domain: Domain,
                      order: MonomialOrder) -> Ideal:
        result = cls.__new__(cls)
        result.symbols = tuple(symbols)
        result.domain = domain
        result.order = order
        result._rings = {}
        result._bases = {}
        R = result.ring()
        result.gens = [R.from_dict(g) for g in backend.gens()]
        result._native = backend
        return result

    def _wrap(self, backend: _Backend, symbols: Optional[Sequence[Symbol]] = None) -> Ideal:
        return Ideal._from_backend(backend, self.symbols if symbols is None else symbols, self.domain, self.order)

    # ------------------------------------------------------------------
    # rings and bases

    def ring(self, order: Optional[OrderSpec] = None) -> PolyRing:
        """The polynomial ring with the given order."""
        key = self.order if order is None else as_order(order)
        R = self._rings.get(key)
        if R is None:
            R = _ring(self.symbols, self.domain, key)[0]
            self._rings[key] = R
        return R

    def _basis(self, order: Optional[OrderSpec] = None) -> list[PolyElement]:
        """The reduced Gröbner basis for the order, as ring elements."""
        key = self.order if order is None else as_order(order)
        G = self._bases.get(key)
        if G is None:
            R = self.ring(key)
            G = [R.from_dict(g) for g in self._native.basis(key)]
            self._bases[key] = G
        return G

    def _converted_basis(self, order: MonomialOrder) -> Optional[list[PolyElement]]:
        """The basis for ``lex`` or ``grlex`` of an ideal of dimension zero,
        converted by FGLM from the basis for ``grevlex``; ``None`` for the
        other orders and ideals."""
        found = self._native.converted_basis(order)
        if found is None:
            return None
        R = self.ring(order)
        return [R.from_dict(g) for g in found]

    def groebner_basis(self, order: Optional[OrderSpec] = None) -> list[Poly]:
        """The reduced Gröbner basis for the order, as a list of ``Poly``."""
        return [self._to_poly(g) for g in self._basis(order)]

    def _to_poly(self, g: PolyElement) -> Poly:
        return Poly(g.as_expr(), *self.symbols, domain=self.domain)

    def _from_expr(self, f: Union[Expr, Poly, PolyElement], order: Optional[OrderSpec] = None) -> PolyElement:
        R = self.ring(order)
        if isinstance(f, PolyElement):
            return R.from_dict(dict(f))
        if isinstance(f, Poly):
            f = f.as_expr()
        return R.from_expr(sympify(f))

    def _new(self, gens: Iterable[Union[Expr, PolyElement]], symbols: Optional[Sequence[Symbol]] = None,
             order: Optional[OrderSpec] = None) -> Ideal:
        variables = self.symbols if symbols is None else tuple(symbols)
        return Ideal([g.as_expr() if isinstance(g, PolyElement) else g for g in gens],
                     *variables, domain=self.domain, order=self.order if order is None else order)

    def __repr__(self) -> str:
        return "Ideal([%s], %s)" % (", ".join(str(g.as_expr()) for g in self.gens),
                                    ", ".join(map(str, self.symbols)))

    @property
    def exprs(self) -> list[Expr]:
        """The generators as expressions."""
        return [g.as_expr() for g in self.gens]

    # ------------------------------------------------------------------
    # membership and equality

    def reduce(self, f: Union[Expr, Poly, PolyElement], order: Optional[OrderSpec] = None) -> Expr:
        """The normal form of ``f`` modulo the Gröbner basis."""
        key = self.order if order is None else as_order(order)
        g = self._from_expr(f, key)
        return self.ring(key).from_dict(self._native.normal_form(dict(g), key)).as_expr()

    def contains(self, f: Union[Expr, Poly, PolyElement]) -> bool:
        """Whether the polynomial ``f`` belongs to the ideal."""
        return not self._native.normal_form(dict(self._from_expr(f)), self.order)

    def __contains__(self, f: Union[Expr, Poly, PolyElement]) -> bool:
        return self.contains(f)

    def is_zero(self) -> bool:
        return self._native.is_zero()

    def is_whole_ring(self) -> bool:
        return self._native.is_whole_ring()

    def _same_ring(self, other: object) -> None:
        if not isinstance(other, Ideal):
            raise TypeError("an Ideal is expected")
        if other.symbols != self.symbols:
            raise ValueError("the ideals live in different rings")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Ideal) or other.symbols != self.symbols:
            return False
        return [g.as_expr() for g in self._basis()] == [g.as_expr() for g in other._basis(self.order)]

    def __ne__(self, other: object) -> bool:
        return not self == other

    __hash__ = object.__hash__

    def subset(self, other: Ideal) -> bool:
        """Whether the ideal is contained in ``other``."""
        self._same_ring(other)
        return all(other.contains(g) for g in self.gens)

    # ------------------------------------------------------------------
    # arithmetic

    def __add__(self, other: Ideal) -> Ideal:
        self._same_ring(other)
        return self._new(self.gens + other.gens)

    def __mul__(self, other: Union[Ideal, Expr, Poly, PolyElement]) -> Ideal:
        if isinstance(other, Ideal):
            self._same_ring(other)
            return self._new([f*g for f in self.gens for g in other.gens])
        return self._new([g*self._from_expr(other) for g in self.gens])

    __rmul__ = __mul__

    def __pow__(self, n: int) -> Ideal:
        if not isinstance(n, int) or n < 0:
            raise ValueError("the exponent must be a non-negative integer")
        result = self._new([self.ring().one])
        for _ in range(n):
            result = result*self
        return result

    # ------------------------------------------------------------------
    # elimination and the operations built on it

    def eliminate(self, symbols: Iterable[Symbol]) -> Ideal:
        """The elimination ideal: the intersection with the polynomial ring
        in the variables other than ``symbols``, as an ideal of that ring.

        >>> from sympy.abc import t, x, y
        >>> from sympy_extras.polys.ideals_sympy import Ideal
        >>> Ideal([x - t**2, y - t**3], t, x, y).eliminate([t])
        Ideal([x**3 - y**2], x, y)
        """
        eliminated = [as_symbol(s) for s in symbols]
        for s in eliminated:
            if s not in self.symbols:
                raise ValueError("%s is not a variable of the ring" % (s,))
        if not eliminated:
            return self
        keep = [s for s in self.symbols if s not in eliminated]
        positions = [self.symbols.index(s) for s in eliminated]
        return self._wrap(self._native.eliminate(positions), keep)

    def _set_basis(self, basis: Sequence[dict[Monomial, DomainElement]]) -> None:
        """Record the reduced Gröbner basis for the default order, known
        from another computation, given as dictionaries of terms."""
        self._native.set_basis([g for g in basis if g])
        self._bases.pop(self.order, None)

    def _terms_of(self, other: Ideal) -> list[SymPyTerms]:
        return [dict(g) for g in other.gens]

    def intersect(self, other: Ideal) -> Ideal:
        """The intersection with ``other``, as $(tI + (1 - t)J) \\cap K[x]$."""
        self._same_ring(other)
        return self._wrap(self._native.intersect(self._terms_of(other)))

    def _as_ideal(self, other: Union[Ideal, Expr, Poly, PolyElement]) -> Ideal:
        if isinstance(other, Ideal):
            return other
        return self._new([self._from_expr(other)])

    def quotient(self, other: Union[Ideal, Expr, Poly, PolyElement]) -> Ideal:
        """The ideal quotient ``I : J`` (``other`` may also be a single
        polynomial): the polynomials ``h`` with ``h*J`` in ``I``.

        >>> from sympy.abc import x, y
        >>> from sympy_extras.polys.ideals_sympy import Ideal
        >>> Ideal([x**2*y, x*y**2], x, y).quotient(Ideal([x*y], x, y))
        Ideal([x, y], x, y)
        """
        divisor = self._as_ideal(other)
        self._same_ring(divisor)
        return self._wrap(self._native.quotient(self._terms_of(divisor)))

    def saturate(self, other: Union[Ideal, Expr, Poly, PolyElement]) -> Ideal:
        """The saturation ``I : J**oo``, the union of the quotients
        ``I : J**n``: by one elimination when ``J`` is generated by one
        polynomial, and by iterating the quotient until it is stable
        otherwise.

        >>> from sympy.abc import x, y
        >>> from sympy_extras.polys.ideals_sympy import Ideal
        >>> Ideal([x**3*y, x**2*y**2], x, y).saturate(Ideal([x], x, y))
        Ideal([y], x, y)
        """
        divisor = self._as_ideal(other)
        return self._wrap(self._native.saturate(self._terms_of(divisor)))

    def radical_contains(self, f: Union[Expr, Poly, PolyElement]) -> bool:
        """Whether ``f`` belongs to the radical of the ideal, i.e. some
        power of ``f`` belongs to the ideal (Rabinowitsch's trick).

        >>> from sympy.abc import x, y
        >>> from sympy_extras.polys.ideals_sympy import Ideal
        >>> I = Ideal([x**2, y**3], x, y)
        >>> I.radical_contains(x*y), I.contains(x*y)
        (True, False)
        """
        return self._native.radical_contains(dict(self._from_expr(f)))

    # ------------------------------------------------------------------
    # dimension, Hilbert series

    def leading_monomials(self, order: Optional[OrderSpec] = None) -> list[Monomial]:
        """The leading monomials of the Gröbner basis, as exponent
        tuples."""
        return self._native.leading_monomials(self.order if order is None else as_order(order))

    def hilbert_series(self, t: Optional[Symbol] = None) -> Expr:
        r"""The Hilbert series of $K[x]/I$ as a rational function
        $N(t)/(1 - t)^n$ in lowest terms (for a non-homogeneous ideal, the
        series of the associated graded ring, computed from the leading
        term ideal of a Gröbner basis for a graded order).

        >>> from sympy.abc import x, y, z, t
        >>> from sympy_extras.polys.ideals_sympy import Ideal
        >>> Ideal([x**2 - y*z, x*y - z**2, y**2 - x*z], x, y, z).hilbert_series(t)
        (2*t + 1)/(1 - t)
        """
        t = Symbol('t') if t is None else t
        q, dim = self._native.hilbert_data()
        if not q:
            return S.Zero
        return Poly(dict(enumerate(q)), t).as_expr()/(1 - t)**dim

    def dimension(self) -> int:
        """The Krull dimension of $K[x]/I$ (``-1`` for the whole ring).

        >>> from sympy.abc import x, y, z
        >>> from sympy_extras.polys.ideals_sympy import Ideal
        >>> Ideal([x*y, x*z], x, y, z).dimension()
        2
        >>> Ideal([x**2 - 1, y - x], x, y).dimension()
        0
        """
        return self._native.dimension()

    def is_zero_dimensional(self) -> bool:
        return self.dimension() == 0

    def degree(self) -> int:
        """The degree of the ideal: the leading coefficient of the Hilbert
        polynomial times the factorial of the dimension (for a
        zero-dimensional ideal, the number of solutions with
        multiplicities).

        >>> from sympy.abc import x, y, z
        >>> from sympy_extras.polys.ideals_sympy import Ideal
        >>> Ideal([x**2 - y*z, x*y - z**2, y**2 - x*z], x, y, z).degree()
        3
        >>> Ideal([x**2 - 1, y**3 - y], x, y).degree()
        6
        """
        return sum(self._native.hilbert_data()[0])

    def hilbert_polynomial(self, d: Optional[Symbol] = None) -> Expr:
        r"""The Hilbert polynomial $HP(d)$ of $K[x]/I$: the dimension of the
        degree ``d`` part for ``d`` large (of the associated graded ring
        for a non-homogeneous ideal).

        >>> from sympy.abc import x, y, z, d
        >>> from sympy_extras.polys.ideals_sympy import Ideal
        >>> Ideal([x**2 - y*z, x*y - z**2, y**2 - x*z], x, y, z).hilbert_polynomial(d)
        3
        >>> Ideal([x*y], x, y, z).hilbert_polynomial(d)
        2*d + 1
        """
        d = Symbol('d') if d is None else d
        q, dim = self._native.hilbert_data()
        if dim <= 0:
            return S.Zero
        result = S.Zero
        for i, c in enumerate(q):
            if c:
                result += c*binomial(d - i + dim - 1, dim - 1)
        return as_expr(result.expand(func=True))

    # ------------------------------------------------------------------
    # zero-dimensional ideals

    def _require_zero_dimensional(self) -> None:
        if not self.is_zero_dimensional():
            raise NotImplementedError("the ideal is not zero-dimensional")

    def standard_monomials(self, order: Optional[OrderSpec] = None) -> list[Expr]:
        """The monomials not divisible by any leading monomial of the
        Gröbner basis, a basis of $K[x]/I$ (zero-dimensional ideals), as
        expressions.

        >>> from sympy.abc import x, y
        >>> from sympy_extras.polys.ideals_sympy import Ideal
        >>> Ideal([x**2 - y, y**2 - 1], x, y).standard_monomials()
        [1, y, x, x*y]
        """
        key = self.order if order is None else as_order(order)
        R = self.ring(key)
        return [R.from_dict({m: R.domain.one}).as_expr() for m in self._native.standard_exponents(key)]

    def vector_space_dimension(self) -> int:
        """The dimension of $K[x]/I$ as a vector space (zero-dimensional
        ideals): the number of solutions counted with multiplicity."""
        return len(self._native.standard_exponents(self.order))

    def multiplication_matrix(self, f: Union[Expr, Poly, PolyElement], order: Optional[OrderSpec] = None) -> MutableDenseMatrix:
        """The matrix of multiplication by ``f`` on $K[x]/I$ in the basis of
        :meth:`standard_monomials`, as a :class:`~sympy.Matrix`.

        >>> from sympy.abc import x, y
        >>> from sympy_extras.polys.ideals_sympy import Ideal
        >>> Ideal([x**2 - 2, y - x], x, y).multiplication_matrix(x)
        Matrix([[0, 2], [1, 0]])
        """
        return self._multiplication_matrix(f, order).to_Matrix()

    def _multiplication_matrix(self, f: Union[Expr, Poly, PolyElement], order: Optional[OrderSpec] = None) -> DomainMatrix:
        key = self.order if order is None else as_order(order)
        rows = self._native.multiplication_matrix(dict(self._from_expr(f, key)), key)
        return DomainMatrix(rows, (len(rows), len(rows)), self.domain)

    def rur(self, t: Optional[Symbol] = None) -> _rur.RationalUnivariateRepresentation:
        """The rational univariate representation of the ideal
        (zero-dimensional ideals over the rationals, Rouillier's algorithm,
        :mod:`sympy_extras.polys.rur_sympy`): a separating linear form, the
        squarefree polynomial in ``t`` of its values at the zeros and the
        coordinates of the zeros as rational functions of the root.

        >>> from sympy.abc import x, y, t
        >>> from sympy_extras.polys.ideals_sympy import Ideal
        >>> R = Ideal([x**2 - 2, y - x**3], x, y).rur(t)
        >>> R.polynomial, R.polynomials()
        (Poly(t**2 - 2, t, domain='QQ'), [Poly(t, t, domain='QQ'), Poly(2*t, t, domain='QQ')])
        >>> R.solutions()
        [{x: -sqrt(2), y: -2*sqrt(2)}, {x: sqrt(2), y: 2*sqrt(2)}]
        """
        return _rur.rur(self, t=t)

    def _polynomial(self, coefficients: Sequence[DomainElement], y: Symbol) -> Poly:
        n = len(coefficients) - 1
        terms = [self.domain.to_sympy(c)*y**(n - i) for i, c in enumerate(coefficients)]
        return Poly(sum(terms), y, domain=self.domain)

    def minimal_polynomial(self, f: Union[Expr, Poly, PolyElement], y: Optional[Symbol] = None) -> Poly:
        """The monic polynomial ``p`` of least degree with ``p(f)`` in the
        ideal (zero-dimensional ideals), as a ``Poly`` in ``y``.

        >>> from sympy.abc import x, y, z
        >>> from sympy_extras.polys.ideals_sympy import Ideal
        >>> Ideal([x**2 - 2, y**2 - 3], x, y).minimal_polynomial(x + y, z)
        Poly(z**4 - 10*z**2 + 1, z, domain='QQ')
        """
        y_ = Symbol('y') if y is None else as_symbol(y)
        return self._polynomial(self._native.minimal_polynomial(dict(self._from_expr(f))), y_)

    def univariate(self, x: Symbol) -> Poly:
        """The monic generator of $I \\cap K[x]$ for a variable ``x``
        (zero-dimensional ideals), as a ``Poly``.

        >>> from sympy.abc import x, y
        >>> from sympy_extras.polys.ideals_sympy import Ideal
        >>> Ideal([x**2 - y, y**2 - 1], x, y).univariate(x)
        Poly(x**4 - 1, x, domain='QQ')
        """
        x_ = as_symbol(x)
        if x_ not in self.symbols:
            raise ValueError("%s is not a variable of the ring" % (x_,))
        return self.minimal_polynomial(x_, x_)

    def radical(self) -> Ideal:
        """The radical of the ideal: by Seidenberg's lemma in dimension
        zero (natively), and as the intersection of the saturated ideals of
        squarefree regular chains otherwise
        (:func:`sympy_extras.polys.idealdecomposition.radical`; over the
        rationals), both natively.

        >>> from sympy.abc import x, y
        >>> from sympy_extras.polys.ideals_sympy import Ideal
        >>> Ideal([x**2, y**2 - 2*y + 1], x, y).radical()
        Ideal([x, y - 1], x, y)
        >>> Ideal([x**2*y, x*y**3], x, y).radical()
        Ideal([x*y], x, y)
        """
        return self._wrap(self._native.radical())

    def reduced(self) -> Ideal:
        """The same ideal generated by its reduced Gröbner basis."""
        return self._wrap(self._native.reduced())

    def is_radical(self) -> bool:
        """Whether the ideal equals its radical.

        >>> from sympy.abc import x, y
        >>> from sympy_extras.polys.ideals_sympy import Ideal
        >>> Ideal([x*y], x, y).is_radical(), Ideal([x**2*y], x, y).is_radical()
        (True, False)
        """
        return self.radical() == self

    def height(self) -> int:
        """The height (the codimension) of a proper ideal: the number of
        variables minus the dimension.

        >>> from sympy.abc import x, y, z
        >>> from sympy_extras.polys.ideals_sympy import Ideal
        >>> Ideal([x*y, x*z], x, y, z).height()
        1
        """
        if self.is_whole_ring():
            raise ValueError("the whole ring has no height")
        return len(self.symbols) - self.dimension()

    def equidimensional_parts(self) -> list[Ideal]:
        """The equidimensional parts of the radical, by decreasing
        dimension (:func:`~sympy_extras.polys.idealdecomposition.equidimensional_parts`).

        >>> from sympy.abc import x, y, z
        >>> from sympy_extras.polys.ideals_sympy import Ideal
        >>> Ideal([x*z, x*y], x, y, z).equidimensional_parts()
        [Ideal([x], x, y, z), Ideal([y, z], x, y, z)]
        """
        return [self._wrap(b) for b in self._native.equidimensional_parts()]

    def minimal_primes(self) -> list[Ideal]:
        """The minimal primes of the ideal, by decreasing dimension
        (:func:`~sympy_extras.polys.idealdecomposition.minimal_primes`).

        >>> from sympy.abc import x, y, z
        >>> from sympy_extras.polys.ideals_sympy import Ideal
        >>> Ideal([x**2 - y**2, x*z - y], x, y, z).minimal_primes()
        [Ideal([x + y, z + 1], x, y, z), Ideal([x - y, z - 1], x, y, z), Ideal([x, y], x, y, z)]
        """
        return [self._wrap(b) for b in self._native.minimal_primes()]

    def primary_decomposition(self) -> list[tuple[Ideal, Ideal]]:
        """An irredundant primary decomposition, as pairs of a primary
        ideal and its associated prime, by decreasing dimension of the
        primes (:func:`~sympy_extras.polys.idealdecomposition.primary_decomposition`).

        >>> from sympy.abc import x, y
        >>> from sympy_extras.polys.ideals_sympy import Ideal
        >>> Ideal([x**2*y, x*y**2], x, y).primary_decomposition()
        [(Ideal([x], x, y), Ideal([x], x, y)), (Ideal([y], x, y), Ideal([y], x, y)), (Ideal([x**2, y**2], x, y), Ideal([x, y], x, y))]
        """
        return [(self._wrap(Q), self._wrap(P)) for Q, P in self._native.primary_decomposition()]

    def associated_primes(self) -> list[Ideal]:
        """The associated primes, minimal and embedded, by decreasing
        dimension (:func:`~sympy_extras.polys.idealdecomposition.associated_primes`).

        >>> from sympy.abc import x, y, z
        >>> from sympy_extras.polys.ideals_sympy import Ideal
        >>> Ideal([x**2, x*y, x*z], x, y, z).associated_primes()
        [Ideal([x], x, y, z), Ideal([x, y, z], x, y, z)]
        """
        return [P for _, P in self.primary_decomposition()]

    def is_primary(self) -> bool:
        """Whether the ideal is primary
        (:func:`~sympy_extras.polys.idealdecomposition.is_primary`).

        >>> from sympy.abc import x, y
        >>> from sympy_extras.polys.ideals_sympy import Ideal
        >>> Ideal([x**2, y**2], x, y).is_primary(), Ideal([x**2, x*y], x, y).is_primary()
        (True, False)
        """
        return self._native.is_primary()

    def is_maximal(self) -> bool:
        """Whether the ideal is maximal, i.e. $K[x]/I$ is a field: the
        ideal has dimension zero, is radical, and the minimal polynomial
        of a separating linear form (found by :func:`separating_forms`) is
        irreducible.

        >>> from sympy.abc import x, y
        >>> from sympy_extras.polys.ideals_sympy import Ideal
        >>> Ideal([x**2 - 2, y**2 - 2], x, y).is_maximal()
        False
        >>> Ideal([x**2 - 2, y - x], x, y).is_maximal()
        True
        """
        return self._native.is_maximal()

    def is_prime(self) -> bool:
        """Whether the ideal is prime: maximal in dimension zero, and its
        own only minimal prime otherwise.

        >>> from sympy.abc import x, y, z
        >>> from sympy_extras.polys.ideals_sympy import Ideal
        >>> Ideal([x*z - y**2, x**2 - y*z], x, y, z).is_prime()
        False
        >>> Ideal([x**2 - y**3], x, y).is_prime(), Ideal([x**2 - y**2], x, y).is_prime()
        (True, False)
        """
        if self.is_whole_ring():
            return False
        if self.is_zero_dimensional():
            return self.is_maximal()
        primes = self.minimal_primes()
        return len(primes) == 1 and primes[0] == self

    # ------------------------------------------------------------------
    # change of order

    def change_order(self, order: OrderSpec) -> list[Poly]:
        """The reduced Gröbner basis for another order, converted with FGLM
        for zero-dimensional ideals and with the Gröbner walk otherwise
        (returned as a list of ``Poly``, and cached).

        >>> from sympy.abc import x, y, z
        >>> from sympy_extras.polys.ideals_sympy import Ideal
        >>> Ideal([x*z - y**2, x**2 - y*z], x, y, z).change_order('lex')
        [Poly(x**2 - y*z, x, y, z, domain='QQ'), Poly(x*y**2 - y*z**2, x, y, z, domain='QQ'), Poly(x*z - y**2, x, y, z, domain='QQ'), Poly(y**4 - y*z**3, x, y, z, domain='QQ')]
        >>> Ideal([x**2 - y, y**2 - 1], x, y).change_order('lex')
        [Poly(x**2 - y, x, y, domain='QQ'), Poly(y**2 - 1, x, y, domain='QQ')]
        """
        key = as_order(order)
        if key not in self._bases:
            R = self.ring(key)
            self._bases[key] = [R.from_dict(g) for g in self._native.change_order(key)]
        return self.groebner_basis(key)



# ---------------------------------------------------------------------------
# between the native ideals and the face

def native_ideal(ideal: Ideal, K: CoreDomain[T]) -> native.Ideal[T]:
    """The native ideal of a face over the native domain ``K`` (which must
    stand for its SymPy domain), with its basis for the default order when
    it is known.

    >>> from sympy.abc import x, y
    >>> from sympy_extras.polys.domains import QQ
    >>> from sympy_extras.polys.ideals_sympy import Ideal, native_ideal
    >>> native_ideal(Ideal([x**2 - y, x*y - 1], x, y), QQ)
    Ideal([x**2 - y, x*y - 1], x, y)
    """
    coefficients = _Coefficients(K, ideal.domain)
    result = native.Ideal.from_terms([coefficients.terms_in(dict(g)) for g in ideal.gens],
                                     [_core_symbol(s) for s in ideal.symbols], K, core_order(ideal.order, len(ideal.symbols)))
    known = ideal._bases.get(ideal.order)
    if known is not None:
        result._set_basis([coefficients.terms_in(dict(g)) for g in known])
    return result


# rur_sympy builds on this module (imported at the end: it imports Ideal from here)
from . import rur_sympy as _rur  # noqa: E402
