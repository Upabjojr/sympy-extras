r"""Polynomial ideals: operations built on Gröbner bases, on the typed
polynomial layer (no SymPy).

:class:`Ideal` is an ideal of $K[x_1, \ldots, x_n]$ for a coefficient field
$K$ of :mod:`sympy_extras.polys.domains` (``QQ``, ``GF(p)``, number fields,
fields of rational functions), given by generators which are core
expressions or elements of :class:`~sympy_extras.polys.rings.PolyRing`.
Its Gröbner bases are computed lazily, by Faugère's F4
(:mod:`sympy_extras.polys.f4`), and cached per monomial order:

* elimination ideals with block orders (:meth:`Ideal.eliminate`),
  intersections, quotients ``I : J``, saturations ``I : J**oo`` and radical
  membership;
* the Krull dimension, the Hilbert series, the Hilbert polynomial and the
  degree, from the leading term ideal of a Gröbner basis for a graded
  order (Bigatti's algorithm, :mod:`sympy_extras.polys.bigatti`);
* for zero-dimensional ideals: the standard monomials and the vector space
  dimension of the quotient algebra, multiplication matrices, univariate
  polynomials in each variable, the radical (Seidenberg's lemma) and the
  test for maximal ideals;
* conversion between monomial orders with FGLM (zero-dimensional,
  :mod:`sympy_extras.polys.fglm`) or the Gröbner walk of Collart,
  Kalkbrener and Mall [1]_ with the lifting of Amrhein, Gloor and Küchlin
  [2]_ (:func:`groebner_walk`).

:mod:`sympy_extras.polys.ideals_sympy` is the SymPy face of this module:
the same :class:`Ideal` on SymPy expressions, polynomials and domains. The
steps which stay written on SymPy are called from here at run time
through the face (the functions ending in ``_of_core``): the modular
Gröbner bases over ``QQ`` when F4 does not finish within
``settings.groebner_direct_time``
(:func:`sympy_extras.polys.modulargroebner.modular_groebner`) and SymPy's
Buchberger algorithm when ``settings.groebner_algorithm`` is
``'buchberger'``. The decompositions of positive-dimensional ideals
through regular chains (the radical, the equidimensional parts, the
minimal and associated primes, the primary decomposition) are those of
:mod:`sympy_extras.polys.idealdecomposition`, the rational univariate
representation that of :mod:`sympy_extras.polys.rur`, both native.

Examples
========

>>> from sympy_extras.core import symbols
>>> from sympy_extras.polys.domains import QQ
>>> from sympy_extras.polys.ideals import Ideal
>>> x, y, z = symbols('x y z')
>>> I = Ideal([x*z - y**2, x**2 - y*z], x, y, z, domain=QQ)
>>> I.dimension(), I.degree()
(1, 4)
>>> I.eliminate([x])
Ideal([y**4 - y*z**3], y, z)

The variety of ``I`` consists of three lines through the origin (where
``x**3 == y**3``) and the ``z`` axis; saturating by ``y`` removes the axis:

>>> I.saturate(y)
Ideal([x**2 - y*z, x*y - z**2, -x*z + y**2], x, y, z)
>>> _.degree()
3
>>> J = Ideal([x**2 - 1, y**2 - 1], x, y, domain=QQ)
>>> J.vector_space_dimension(), J.is_radical(), J.is_prime()
(4, True, False)

References
==========

.. [1] S. Collart, M. Kalkbrener, D. Mall, Converting bases with the
       Gröbner walk, J. Symbolic Computation 24 (1997) 465-469.
.. [2] B. Amrhein, O. Gloor, W. Küchlin, On the walk, Theoretical Computer
       Science 187 (1997) 179-202.
.. [3] D. Cox, J. Little, D. O'Shea, *Ideals, Varieties, and Algorithms*,
       4th ed., Springer, 2015 (elimination, intersections and quotients:
       chapter 4; dimension and the Hilbert polynomial: chapter 9).
.. [4] T. Becker, V. Weispfenning, *Gröbner Bases*, Springer, 1993
       (Seidenberg's lemma: section 8.2; the test for maximal ideals).
"""
from __future__ import annotations

from fractions import Fraction
from itertools import product
from math import lcm
from typing import Generic, Iterable, Iterator, Optional, Sequence, TypeVar, Union

from sympy_extras._timeout import attempt
from sympy_extras.core.add import add
from sympy_extras.core.expr import Expr, as_expr
from sympy_extras.core.mul import mul
from sympy_extras.core.numbers import Integer
from sympy_extras.core.power import pow
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.settings import settings

from .bigatti import hilbert_numerator_coefficients
from .domains.domain import Domain
from .domains.integerring import ZZ
from .domains.rationalfield import QQ, RationalField
from .f4 import f4, reduced_basis
from .fglm import fglm
from .polyclasses import DensePoly
from .polyutils import dict_from_expr
from .rings import (Monomial, MonomialOrder, PolyElement, PolyRing, WeightOrder, as_order, elimination_order, grevlex,
                    grlex, lex, monomial_div, monomial_divides, monomial_mul)

__all__ = ['Ideal', 'groebner', 'groebner_walk', 'normal_form', 'separating_forms', 'separating_coefficients',
           'initial_form', 'OrderLike', 'GRADED_FACTOR']

T = TypeVar('T')

#: a monomial order or its name
OrderLike = Union[str, MonomialOrder]
#: the terms of a polynomial
Terms = dict[Monomial, T]
#: a polynomial given to an ideal: a core expression or a ring element
PolyLike = Union[Expr, PolyElement[T]]

#: for the graded orders F4 is given this many times
#: ``settings.groebner_direct_time`` before the modular algorithm takes
#: over (the factor of :mod:`sympy_extras.polys.modulargroebner`)
GRADED_FACTOR = 20


def _is_graded(order: MonomialOrder) -> bool:
    return order == grevlex or order == grlex


# ----------------------------------------------------------------------
# separating forms

def separating_coefficients(nvars: int, points: int) -> Iterator[tuple[int, ...]]:
    r"""The coefficients $(c_2, \ldots, c_n)$ of the linear forms of
    :func:`separating_forms` in ``nvars`` variables, in the same order
    (nothing is yielded for fewer than two variables).

    >>> from sympy_extras.polys.ideals import separating_coefficients
    >>> list(separating_coefficients(2, 3))
    [(0,), (1,), (-1,), (2,), (-2,)]
    """
    if nvars < 2:
        return
    pairs = points*(points - 1)//2
    bound = (pairs + 1)//2
    for height in range(bound + 1):
        values = [0] + [v for k in range(1, height + 1) for v in (k, -k)]
        for tail in product(values, repeat=nvars - 1):
            if height and max(abs(c) for c in tail) != height:
                continue
            yield tail


def separating_forms(variables: Sequence[Symbol], points: int) -> Iterator[Expr]:
    r"""Linear forms $x_1 + c_2 x_2 + \cdots + c_n x_n$ with small integer
    coefficients, by increasing height $\max |c_i|$, among which one takes
    different values at any ``points`` distinct points (with coordinates in
    any field of characteristic zero).

    The forms which take the same value at two of the points are the zeros
    of an affine function of $(c_2, \ldots, c_n)$, not zero, for each of
    the $D = p(p - 1)/2$ pairs; their product has degree at most $D$ and
    does not vanish on the whole grid $\{-H, \ldots, H\}^{n - 1}$ as soon
    as $2H + 1 > D$ (Schwartz–Zippel), so that the search ends at the
    height $H = \lceil D/2 \rceil$. The coefficients of each height are
    enumerated in the order ``0, 1, -1, 2, -2, ...``: the first forms are
    ``x1``, then the sums and differences of two variables.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.polys.ideals import separating_forms
    >>> x, y, z = symbols('x y z')
    >>> list(separating_forms([x, y], 3))
    [x, x + y, x - y, x + 2*y, x - 2*y]
    >>> [f for f, _ in zip(separating_forms([x, y, z], 10), range(5))]
    [x, x + z, x - z, x + y, x + y + z]
    """
    if len(variables) < 2:
        # one variable separates distinct points by itself
        yield variables[0] if variables else Integer(0)
        return
    for tail in separating_coefficients(len(variables), points):
        yield add(variables[0], *[mul(Integer(c), v) for c, v in zip(tail, variables[1:]) if c])


# ----------------------------------------------------------------------
# Gröbner bases, normal forms, the walk

def groebner(polys: Sequence[PolyElement[T]], ring: PolyRing[T]) -> list[PolyElement[T]]:
    """The reduced Gröbner basis of ``polys`` in ``ring`` (over a field),
    sorted by decreasing leading monomial: what :class:`Ideal` computes.

    F4 (:func:`sympy_extras.polys.f4.f4`) by default. Over ``QQ`` F4 is
    given ``settings.groebner_direct_time`` seconds (``GRADED_FACTOR``
    times as long for ``grevlex`` and ``grlex``), and the modular algorithm
    of :mod:`sympy_extras.polys.modulargroebner` (on SymPy, through the
    face) computes the basis when it has not finished, unless
    ``settings.modular_groebner`` is off. With
    ``settings.groebner_algorithm`` ``'buchberger'`` SymPy's Buchberger
    algorithm is used.

    >>> from sympy_extras.polys.domains import QQ
    >>> from sympy_extras.polys.rings import PolyRing, grevlex
    >>> from sympy_extras.polys.ideals import groebner
    >>> R = PolyRing(QQ, ['x', 'y'], grevlex)
    >>> x, y = R.gens
    >>> [str(g) for g in groebner([x**2 + y**2 - R.one, x*y - R.from_int(2)], R)]
    ['y**3 + 2*x - y', 'x**2 + y**2 - 1', 'x*y - 2']
    """
    nonzero = [f if f.ring == ring else PolyElement(dict(f.terms), ring) for f in polys if f.terms]
    if not nonzero:
        return []
    if settings.groebner_algorithm != 'f4':
        from .ideals_sympy import direct_groebner_of_core
        return direct_groebner_of_core(nonzero, ring)
    if settings.modular_groebner and isinstance(ring.domain, RationalField):
        seconds = settings.groebner_direct_time
        if _is_graded(ring.order):
            seconds *= GRADED_FACTOR
        if seconds > 0:
            direct = attempt(lambda: f4(nonzero, ring), seconds, native=True)
            if direct is not None:
                return direct
        from .ideals_sympy import modular_groebner_of_core
        found = modular_groebner_of_core(nonzero, ring)
        if found is not None:
            return found
    return f4(nonzero, ring)


def normal_form(f: PolyElement[T], basis: Sequence[PolyElement[T]]) -> PolyElement[T]:
    """The remainder of the division of ``f`` by ``basis`` (over a field,
    for the order of the ring of ``f``): no term is divisible by a leading
    monomial of the basis; the normal form when ``basis`` is a Gröbner
    basis.

    >>> from sympy_extras.polys.domains import QQ
    >>> from sympy_extras.polys.rings import PolyRing, grevlex
    >>> from sympy_extras.polys.ideals import normal_form
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


def initial_form(f: PolyElement[T], weights: Sequence[Fraction]) -> PolyElement[T]:
    """The terms of ``f`` of maximal weight for the weight vector.

    >>> from fractions import Fraction
    >>> from sympy_extras.polys.domains import QQ
    >>> from sympy_extras.polys.rings import PolyRing
    >>> from sympy_extras.polys.ideals import initial_form
    >>> R = PolyRing(QQ, ['x', 'y'])
    >>> x, y = R.gens
    >>> print(initial_form(x**2 + x*y + y**3, [Fraction(1), Fraction(1)]))
    y**3
    """
    best: Optional[Fraction] = None
    terms: Terms[T] = {}
    for m, c in f.terms.items():
        w = sum((a*b for a, b in zip(weights, m)), Fraction(0))
        if best is None or w > best:
            best, terms = w, {m: c}
        elif w == best:
            terms[m] = c
    return PolyElement(terms, f.ring)


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


def groebner_walk(G: Sequence[PolyElement[T]], ring: PolyRing[T], target: MonomialOrder) -> list[PolyElement[T]]:
    r"""The reduced Gröbner basis for ``target`` of the ideal whose reduced
    Gröbner basis for the order of ``ring`` is ``G``, by the Gröbner walk
    (in the ring with the order ``target``, sorted by decreasing leading
    monomial).

    The source and target orders are represented by weight vectors
    $\omega$ and $\tau$ (``lex``, ``grlex``, ``grevlex`` and
    :class:`~sympy_extras.polys.rings.WeightOrder`), and the basis is
    transformed along the segment from $\omega$ to $\tau$: at every point
    where the ordering of the monomials of the basis changes, a Gröbner
    basis of the initial forms (the terms of maximal weight) is computed
    by F4, and each of its elements $h$ is lifted to $h - \mathrm{NF}_G(h)$
    (normal form for the previous order), which belongs to the ideal and
    has initial form $h$ [2]_.

    >>> from sympy_extras.polys.domains import QQ
    >>> from sympy_extras.polys.rings import PolyRing, grevlex, lex
    >>> from sympy_extras.polys.f4 import f4
    >>> from sympy_extras.polys.ideals import groebner_walk
    >>> R = PolyRing(QQ, ['x', 'y', 'z'], grevlex)
    >>> x, y, z = R.gens
    >>> [str(g) for g in groebner_walk(f4([x*z - y**2, x**2 - y*z], R), R, lex)]
    ['x**2 - y*z', 'x*y**2 - y*z**2', 'x*z - y**2', 'y**4 - y*z**3']
    """
    n = ring.ngens
    basis = [PolyElement(dict(g.terms), ring) for g in G if g.terms]
    final = ring.with_order(target)
    if not basis:
        return []
    weights = _weight_of(ring.order, n, basis)
    tau = _target_weight(target, n)
    current = ring
    while True:
        new_ring = ring.with_order(WeightOrder(_integral(weights), target))
        initial = [PolyElement(initial_form(g, weights).terms, new_ring) for g in basis]
        lifted: list[PolyElement[T]] = []
        for h in f4(initial, new_ring):
            h_old = PolyElement(h.terms, current)
            lifted.append(PolyElement((h_old - normal_form(h_old, basis)).terms, new_ring))
        basis = reduced_basis(lifted, new_ring)
        current = new_ring
        if weights == tau:
            break
        weights = _next_weight(basis, weights, tau)
    return reduced_basis([PolyElement(dict(g.terms), final) for g in basis], final)


# ----------------------------------------------------------------------
# ideals

class Ideal(Generic[T]):
    """An ideal of a polynomial ring over a field, given by generators.

    Parameters
    ==========

    gens : iterable of Expr or PolyElement
        The generators: core expressions, polynomials in the symbols with
        coefficients in ``domain``, or elements of a
        :class:`~sympy_extras.polys.rings.PolyRing` in as many variables.
    symbols : Symbol
        The variables of the ring (all the free symbols of the generators,
        sorted by name, if omitted).
    domain : Domain
        The coefficient field (``QQ``, ``GF(p)``, an algebraic field, a
        field of rational functions, ...).
    order : str or MonomialOrder
        The order of the Gröbner basis used by default (``'grevlex'``).

    Gröbner bases are computed lazily by :func:`groebner` and cached per
    order.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.polys.domains import GF
    >>> from sympy_extras.polys.ideals import Ideal
    >>> x, y = symbols('x y')
    >>> I = Ideal([x**2 + 1, y - x], x, y, domain=GF(5))
    >>> [str(g) for g in I.groebner_basis()], I.is_maximal()
    (['y**2 + 1', 'x + 4*y'], False)
    """

    def __init__(self, gens: Iterable[PolyLike[T]], *symbols: Symbol, domain: Domain[T],
                 order: OrderLike = grevlex) -> None:
        generators = list(gens)
        variables: list[Symbol]
        if symbols:
            variables = list(symbols)
        else:
            found: set[Symbol] = set()
            for g in generators:
                if isinstance(g, PolyElement):
                    found |= set(g.ring.symbols)
                else:
                    found |= g.free_symbols
            variables = sorted(found, key=lambda s: s.name)
        if not variables:
            raise ValueError("no variables: give the symbols of the ring")
        if not domain.is_field:
            raise ValueError("the coefficient domain of an ideal must be a field, got %s" % (domain,))
        self.symbols: tuple[Symbol, ...] = tuple(variables)
        self.domain: Domain[T] = domain
        self.order: MonomialOrder = as_order(order)
        names: list[str] = []
        for s in self.symbols:
            name, k = s.name, 0
            while name in names:
                k += 1
                name = '%s_%d' % (s.name, k)
            names.append(name)
        self._names: list[str] = names
        renamed = {s: Symbol(name) for s, name in zip(self.symbols, names) if s.name != name}
        self._renamed: dict[Symbol, Symbol] = renamed
        self._rings: dict[MonomialOrder, PolyRing[T]] = {}
        self._bases: dict[MonomialOrder, list[PolyElement[T]]] = {}
        self.gens: list[PolyElement[T]] = [g for g in (self._poly(p) for p in generators) if g.terms]

    @classmethod
    def from_terms(cls, terms: Iterable[Terms[T]], symbols: Sequence[Symbol], domain: Domain[T],
                   order: OrderLike = grevlex) -> Ideal[T]:
        """The ideal generated by the polynomials with these terms
        (exponent tuples in the order of ``symbols``).

        >>> from fractions import Fraction
        >>> from sympy_extras.core import symbols
        >>> from sympy_extras.polys.domains import QQ
        >>> from sympy_extras.polys.ideals import Ideal
        >>> x, y = symbols('x y')
        >>> Ideal.from_terms([{(1, 0): Fraction(1), (0, 2): Fraction(-1)}], [x, y], QQ)
        Ideal([x - y**2], x, y)
        """
        result: Ideal[T] = Ideal([], *symbols, domain=domain, order=order)
        R = result.ring()
        result.gens = [PolyElement(dict(t), R) for t in terms if t]
        return result

    # ------------------------------------------------------------------
    # rings and bases

    def ring(self, order: Optional[OrderLike] = None) -> PolyRing[T]:
        """The polynomial ring with the given order (the default one if
        omitted)."""
        key = self.order if order is None else as_order(order)
        R = self._rings.get(key)
        if R is None:
            R = PolyRing(self.domain, self._names, key)
            self._rings[key] = R
        return R

    def _basis(self, order: Optional[OrderLike] = None) -> list[PolyElement[T]]:
        """The reduced Gröbner basis for the order, as ring elements sorted
        by decreasing leading monomial."""
        key = self.order if order is None else as_order(order)
        G = self._bases.get(key)
        if G is None:
            R = self.ring(key)
            G = self._converted_basis(key)
            if G is None:
                G = groebner([PolyElement(g.terms, R) for g in self.gens], R)
            G = sorted([g for g in G if g.terms], key=lambda g: key.key(g.LM), reverse=True)
            self._bases[key] = G
        return G

    def _converted_basis(self, order: MonomialOrder) -> Optional[list[PolyElement[T]]]:
        """The basis for ``lex`` or ``grlex`` of an ideal of dimension zero,
        converted by FGLM from the basis for ``grevlex``, which is much
        cheaper to compute (Katsura-5: 2.4 s, against more than 300 s for
        the lexicographic basis directly); ``None`` for the other orders and
        ideals. The reduced basis is unique, so that the result is the one
        of the direct computation."""
        if not (order == lex or order == grlex) or not self.gens:
            return None
        graded = self._basis(grevlex)
        # dimension zero: a power of every variable is a leading monomial
        pure = {i for g in graded for i, e in enumerate(g.LM) if e and sum(g.LM) == e}
        if len(pure) < len(self.symbols):
            return None
        R = self.ring(order)
        return [PolyElement(g.terms, R) for g in fglm(graded, self.ring(grevlex), order)]

    def groebner_basis(self, order: Optional[OrderLike] = None) -> list[PolyElement[T]]:
        """The reduced Gröbner basis for the order (the default one if
        omitted), sorted by decreasing leading monomial.

        >>> from sympy_extras.core import symbols
        >>> from sympy_extras.polys.domains import QQ
        >>> from sympy_extras.polys.ideals import Ideal
        >>> x, y = symbols('x y')
        >>> [str(g) for g in Ideal([x**2 + y**2 - 1, x - y], x, y, domain=QQ).groebner_basis('lex')]
        ['x - y', 'y**2 - 1/2']
        """
        return list(self._basis(order))

    def _set_basis(self, basis: Sequence[Terms[T]]) -> None:
        """Record the reduced Gröbner basis for the default order, known
        from another computation, given by the terms of its elements."""
        R = self.ring()
        G = [PolyElement(dict(g), R) for g in basis if g]
        self._bases[self.order] = sorted(G, key=lambda g: self.order.key(g.LM), reverse=True)

    def _poly(self, f: PolyLike[T], order: Optional[OrderLike] = None) -> PolyElement[T]:
        """``f`` as an element of the ring with the order."""
        R = self.ring(order)
        if isinstance(f, PolyElement):
            if f.ring.ngens != len(self.symbols):
                raise ValueError("%s is not an element of a ring in %d variables" % (f, len(self.symbols)))
            return PolyElement(dict(f.terms), R)
        e = as_expr(f.xreplace(self._renamed)) if self._renamed else f
        return PolyElement(dict_from_expr(e, self._names, self.domain), R)

    def to_expr(self, f: PolyElement[T]) -> Expr:
        """An element of the ring as a core expression in the symbols."""
        K = self.domain
        parts: list[Expr] = []
        for m, c in f.terms.items():
            factors: list[Expr] = [K.to_expr(c)]
            for s, e in zip(self.symbols, m):
                if e:
                    factors.append(pow(s, Integer(e)) if e > 1 else s)
            parts.append(mul(*factors))
        return add(*parts) if parts else Integer(0)

    def _new(self, gens: Iterable[PolyElement[T]], symbols: Optional[Sequence[Symbol]] = None) -> Ideal[T]:
        """The ideal of the same ring (or of the ring in ``symbols``)
        generated by ring elements."""
        variables = self.symbols if symbols is None else tuple(symbols)
        return Ideal.from_terms([g.terms for g in gens], variables, self.domain, self.order)

    def __repr__(self) -> str:
        return "Ideal([%s], %s)" % (", ".join(str(e) for e in self.exprs), ", ".join(map(str, self.symbols)))

    @property
    def exprs(self) -> list[Expr]:
        """The generators as core expressions."""
        return [self.to_expr(g) for g in self.gens]

    # ------------------------------------------------------------------
    # membership and equality

    def normal_form(self, f: PolyLike[T], order: Optional[OrderLike] = None) -> PolyElement[T]:
        """The normal form of ``f`` modulo the Gröbner basis for the order,
        as a ring element."""
        return normal_form(self._poly(f, order), self._basis(order))

    def reduce(self, f: PolyLike[T], order: Optional[OrderLike] = None) -> Expr:
        """The normal form of ``f`` modulo the Gröbner basis, as a core
        expression.

        >>> from sympy_extras.core import symbols
        >>> from sympy_extras.polys.domains import QQ
        >>> from sympy_extras.polys.ideals import Ideal
        >>> x, y = symbols('x y')
        >>> Ideal([x**2 + y**2 - 1, x - y], x, y, domain=QQ).reduce(x**3)
        y/2
        """
        return self.to_expr(self.normal_form(f, order))

    def contains(self, f: PolyLike[T]) -> bool:
        """Whether the polynomial ``f`` belongs to the ideal."""
        return not self.normal_form(f).terms

    def __contains__(self, f: PolyLike[T]) -> bool:
        return self.contains(f)

    def is_zero(self) -> bool:
        return not self._basis()

    def is_whole_ring(self) -> bool:
        G = self._basis()
        return len(G) == 1 and G[0].is_ground

    def _same_ring(self, other: object) -> None:
        if not isinstance(other, Ideal):
            raise TypeError("an Ideal is expected")
        if other.symbols != self.symbols:
            raise ValueError("the ideals live in different rings")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Ideal) or other.symbols != self.symbols or other.domain != self.domain:
            return False
        return [g.terms for g in self._basis()] == [g.terms for g in other._basis(self.order)]

    def __ne__(self, other: object) -> bool:
        return not self == other

    __hash__ = object.__hash__

    def subset(self, other: Ideal[T]) -> bool:
        """Whether the ideal is contained in ``other``."""
        self._same_ring(other)
        return all(other.contains(g) for g in self.gens)

    # ------------------------------------------------------------------
    # arithmetic

    def __add__(self, other: Ideal[T]) -> Ideal[T]:
        self._same_ring(other)
        return self._new(self.gens + other.gens)

    def __mul__(self, other: Union[Ideal[T], PolyLike[T]]) -> Ideal[T]:
        if isinstance(other, Ideal):
            self._same_ring(other)
            return self._new([f*g for f in self.gens for g in other.gens])
        h = self._poly(other)
        return self._new([g*h for g in self.gens])

    __rmul__ = __mul__

    def __pow__(self, n: int) -> Ideal[T]:
        if not isinstance(n, int) or n < 0:
            raise ValueError("the exponent must be a non-negative integer")
        result = self._new([self.ring().one])
        for _ in range(n):
            result = result*self
        return result

    # ------------------------------------------------------------------
    # elimination and the operations built on it

    def eliminate(self, symbols: Iterable[Symbol]) -> Ideal[T]:
        """The elimination ideal: the intersection with the polynomial ring
        in the variables other than ``symbols``, as an ideal of that ring.

        Computed with a Gröbner basis for a block order having the
        eliminated variables in the first block.

        >>> from sympy_extras.core import symbols
        >>> from sympy_extras.polys.domains import QQ
        >>> from sympy_extras.polys.ideals import Ideal
        >>> t, x, y = symbols('t x y')
        >>> Ideal([x - t**2, y - t**3], t, x, y, domain=QQ).eliminate([t])
        Ideal([x**3 - y**2], x, y)
        """
        eliminated = list(symbols)
        for s in eliminated:
            if s not in self.symbols:
                raise ValueError("%s is not a variable of the ring" % (s,))
        if not eliminated:
            return self
        positions = [self.symbols.index(s) for s in eliminated]
        kept = [i for i in range(len(self.symbols)) if i not in positions]
        permutation = positions + kept
        k = len(eliminated)
        R: PolyRing[T] = PolyRing(self.domain, [self._names[i] for i in permutation],
                                  elimination_order(k, len(permutation), grevlex))
        gens = [PolyElement({tuple(m[i] for i in permutation): c for m, c in g.terms.items()}, R) for g in self.gens]
        G = groebner(gens, R)
        found = [g for g in G if all(all(e == 0 for e in m[:k]) for m in g.terms)]
        result = Ideal.from_terms([{m[k:]: c for m, c in g.terms.items()} for g in found],
                                  [self.symbols[i] for i in kept], self.domain, self.order)
        if self.order == grevlex:
            # the elements of the basis without the eliminated variables are
            # the reduced basis of the elimination ideal for the order of
            # the second block, grevlex: no need to compute it again
            result._set_basis([g.terms for g in result.gens])
        return result

    def _extended(self, polys: Sequence[Terms[T]]) -> Ideal[T]:
        """The ideal of ``K[t, x]`` (``t`` a new first variable) generated
        by the generators of the ideal and the given polynomials of
        ``K[t, x]``."""
        t = Dummy('t')
        shifted = [{(0,) + m: c for m, c in g.terms.items()} for g in self.gens]
        return Ideal.from_terms(shifted + list(polys), (t,) + self.symbols, self.domain, self.order)

    def _one_minus_t_times(self, f: PolyElement[T]) -> Terms[T]:
        """The terms of ``1 - t*f`` in ``K[t, x]``."""
        K = self.domain
        terms: Terms[T] = {(1,) + m: K.neg(c) for m, c in f.terms.items()}
        origin: Monomial = tuple(0 for _ in range(len(self.symbols) + 1))
        terms[origin] = K.one
        return terms

    def intersect(self, other: Ideal[T]) -> Ideal[T]:
        """The intersection with ``other``, as $(tI + (1 - t)J) \\cap K[x]$.

        >>> from sympy_extras.core import symbols
        >>> from sympy_extras.polys.domains import QQ
        >>> from sympy_extras.polys.ideals import Ideal
        >>> x, y = symbols('x y')
        >>> Ideal([x], x, y, domain=QQ).intersect(Ideal([y], x, y, domain=QQ))
        Ideal([x*y], x, y)
        """
        self._same_ring(other)
        K = self.domain
        t = Dummy('t')
        gens: list[Terms[T]] = [{(1,) + m: c for m, c in g.terms.items()} for g in self.gens]
        for g in other.gens:
            terms: Terms[T] = {(0,) + m: c for m, c in g.terms.items()}
            terms.update({(1,) + m: K.neg(c) for m, c in g.terms.items()})
            gens.append(terms)
        J = Ideal.from_terms(gens, (t,) + self.symbols, self.domain, self.order)
        return J.eliminate([t])

    def quotient(self, other: Union[Ideal[T], PolyLike[T]]) -> Ideal[T]:
        """The ideal quotient ``I : J`` (``other`` may also be a single
        polynomial): the polynomials ``h`` with ``h*J`` in ``I``.

        >>> from sympy_extras.core import symbols
        >>> from sympy_extras.polys.domains import QQ
        >>> from sympy_extras.polys.ideals import Ideal
        >>> x, y = symbols('x y')
        >>> Ideal([x**2*y, x*y**2], x, y, domain=QQ).quotient(x*y)
        Ideal([x, y], x, y)
        """
        divisor = other if isinstance(other, Ideal) else self._new([self._poly(other)])
        self._same_ring(divisor)
        result: Optional[Ideal[T]] = None
        R = self.ring()
        for f in divisor.gens:
            g_f = PolyElement(f.terms, R)
            common = self.intersect(self._new([g_f]))
            quot: list[PolyElement[T]] = []
            for g in common._basis():
                q, r = PolyElement(g.terms, R).div([g_f])
                if r.terms:
                    raise RuntimeError("inexact division in the ideal quotient")
                quot.append(q[0].monic())
            part = self._new(quot)
            result = part if result is None else result.intersect(part)
        if result is None:
            return self._new([R.one])
        return result

    def saturate(self, other: Union[Ideal[T], PolyLike[T]]) -> Ideal[T]:
        """The saturation ``I : J**oo``, the union of the quotients
        ``I : J**n``: by one elimination when ``J`` is generated by one
        polynomial (Rabinowitsch), and by iterating the quotient until it
        is stable otherwise.

        >>> from sympy_extras.core import symbols
        >>> from sympy_extras.polys.domains import QQ
        >>> from sympy_extras.polys.ideals import Ideal
        >>> x, y = symbols('x y')
        >>> Ideal([x**3*y, x**2*y**2], x, y, domain=QQ).saturate(x)
        Ideal([y], x, y)
        """
        divisor = other if isinstance(other, Ideal) else self._new([self._poly(other)])
        if len(divisor.gens) == 1:
            extended = self._extended([self._one_minus_t_times(divisor.gens[0])])
            return extended.eliminate([extended.symbols[0]])
        current = self
        while True:
            following = current.quotient(divisor)
            if following == current:
                return following
            current = following

    def radical_contains(self, f: PolyLike[T]) -> bool:
        """Whether ``f`` belongs to the radical of the ideal, i.e. some
        power of ``f`` belongs to the ideal (Rabinowitsch's trick).

        >>> from sympy_extras.core import symbols
        >>> from sympy_extras.polys.domains import QQ
        >>> from sympy_extras.polys.ideals import Ideal
        >>> x, y = symbols('x y')
        >>> I = Ideal([x**2, y**3], x, y, domain=QQ)
        >>> I.radical_contains(x*y), I.contains(x*y)
        (True, False)
        """
        return self._extended([self._one_minus_t_times(self._poly(f))]).is_whole_ring()

    # ------------------------------------------------------------------
    # dimension, Hilbert series

    def leading_monomials(self, order: Optional[OrderLike] = None) -> list[Monomial]:
        """The leading monomials of the Gröbner basis, as exponent
        tuples."""
        return [g.LM for g in self._basis(order)]

    def _graded_order(self) -> MonomialOrder:
        return self.order if _is_graded(self.order) else grevlex

    def hilbert_data(self) -> tuple[list[int], int]:
        """``(q, dim)``: the numerator of the Hilbert series divided by the
        largest power of ``1 - t`` (coefficients lowest degree first, ``[]``
        for the whole ring) and the Krull dimension ``dim`` (the number of
        variables for the whole ring, see :meth:`dimension`)."""
        N = hilbert_numerator_coefficients(self.leading_monomials(self._graded_order()))
        s = 0
        while N and sum(N) == 0:
            # N = (1 - t) q: q_i = N_0 + ... + N_i
            quotient: list[int] = []
            running = 0
            for c in N[:-1]:
                running += c
                quotient.append(running)
            N, s = quotient, s + 1
        return N, len(self.symbols) - s

    def hilbert_series(self, t: Optional[Symbol] = None) -> Expr:
        r"""The Hilbert series of $K[x]/I$ as a rational function
        $q(t)/(1 - t)^d$ in lowest terms (for a non-homogeneous ideal, the
        series of the associated graded ring, computed from the leading
        term ideal of a Gröbner basis for a graded order).

        >>> from sympy_extras.core import symbols
        >>> from sympy_extras.polys.domains import QQ
        >>> from sympy_extras.polys.ideals import Ideal
        >>> x, y, z, t = symbols('x y z t')
        >>> Ideal([x**2 - y*z, x*y - z**2, y**2 - x*z], x, y, z, domain=QQ).hilbert_series(t)
        (2*t + 1)/(1 - t)
        """
        variable = Symbol('t') if t is None else t
        q, dim = self.hilbert_data()
        if not q:
            return Integer(0)
        numerator = DensePoly(list(reversed(q)), ZZ, variable).to_expr()
        if dim == 0:
            return numerator
        return mul(numerator, pow(add(Integer(1), mul(Integer(-1), variable)), Integer(-dim)))

    def dimension(self) -> int:
        """The Krull dimension of $K[x]/I$ (``-1`` for the whole ring).

        >>> from sympy_extras.core import symbols
        >>> from sympy_extras.polys.domains import QQ
        >>> from sympy_extras.polys.ideals import Ideal
        >>> x, y, z = symbols('x y z')
        >>> Ideal([x*y, x*z], x, y, z, domain=QQ).dimension()
        2
        >>> Ideal([x**2 - 1, y - x], x, y, domain=QQ).dimension()
        0
        """
        if self.is_whole_ring():
            return -1
        return self.hilbert_data()[1]

    def is_zero_dimensional(self) -> bool:
        return self.dimension() == 0

    def degree(self) -> int:
        """The degree of the ideal: the leading coefficient of the Hilbert
        polynomial times the factorial of the dimension (for a
        zero-dimensional ideal, the number of solutions with
        multiplicities).

        >>> from sympy_extras.core import symbols
        >>> from sympy_extras.polys.domains import QQ
        >>> from sympy_extras.polys.ideals import Ideal
        >>> x, y, z = symbols('x y z')
        >>> Ideal([x**2 - y*z, x*y - z**2, y**2 - x*z], x, y, z, domain=QQ).degree()
        3
        """
        return sum(self.hilbert_data()[0])

    def hilbert_polynomial(self, d: Optional[Symbol] = None) -> Expr:
        r"""The Hilbert polynomial $HP(d)$ of $K[x]/I$: the dimension of the
        degree ``d`` part for ``d`` large (of the associated graded ring
        for a non-homogeneous ideal), $\sum_i q_i \binom{d - i + D - 1}{D - 1}$
        for the reduced numerator $q$ and the dimension $D$.

        >>> from sympy_extras.core import symbols
        >>> from sympy_extras.polys.domains import QQ
        >>> from sympy_extras.polys.ideals import Ideal
        >>> x, y, z, d = symbols('x y z d')
        >>> Ideal([x*y], x, y, z, domain=QQ).hilbert_polynomial(d)
        2*d + 1
        """
        variable = Symbol('d') if d is None else d
        q, dim = self.hilbert_data()
        if dim <= 0:
            return Integer(0)
        # coefficients lowest degree first
        total: list[Fraction] = [Fraction(0)]*dim
        factorial = 1
        for j in range(1, dim):
            factorial *= j
        for i, c in enumerate(q):
            if not c:
                continue
            # binomial(d - i + dim - 1, dim - 1) = prod_{j=1}^{dim-1} (d - i + j) / (dim - 1)!
            term: list[Fraction] = [Fraction(c, factorial)]
            for j in range(1, dim):
                shifted = [Fraction(0)]*(len(term) + 1)
                for k, a in enumerate(term):
                    shifted[k] += a*(j - i)
                    shifted[k + 1] += a
                term = shifted
            for k, a in enumerate(term):
                total[k] += a
        return DensePoly(list(reversed(total)), QQ, variable).to_expr()

    # ------------------------------------------------------------------
    # zero-dimensional ideals

    def _require_zero_dimensional(self) -> None:
        if not self.is_zero_dimensional():
            raise NotImplementedError("the ideal is not zero-dimensional")

    def standard_exponents(self, order: Optional[OrderLike] = None) -> list[Monomial]:
        """The exponents of the monomials not divisible by any leading
        monomial of the Gröbner basis (zero-dimensional ideals), by
        increasing monomial."""
        self._require_zero_dimensional()
        lms = self.leading_monomials(order)
        n = len(self.symbols)
        key = self.ring(order).order.key
        found: list[Monomial] = []
        seen: set[Monomial] = set()
        queue: list[Monomial] = [tuple(0 for _ in range(n))]
        while queue:
            m = queue.pop(0)
            if m in seen or any(monomial_divides(lm, m) for lm in lms):
                continue
            seen.add(m)
            found.append(m)
            for i in range(n):
                queue.append(m[:i] + (m[i] + 1,) + m[i + 1:])
        found.sort(key=key)
        return found

    def standard_monomials(self, order: Optional[OrderLike] = None) -> list[Expr]:
        """The monomials not divisible by any leading monomial of the
        Gröbner basis, a basis of $K[x]/I$ (zero-dimensional ideals), as
        core expressions.

        >>> from sympy_extras.core import symbols
        >>> from sympy_extras.polys.domains import QQ
        >>> from sympy_extras.polys.ideals import Ideal
        >>> x, y = symbols('x y')
        >>> Ideal([x**2 - y, y**2 - 1], x, y, domain=QQ).standard_monomials()
        [1, y, x, x*y]
        """
        R = self.ring(order)
        return [self.to_expr(R.term(m, self.domain.one)) for m in self.standard_exponents(order)]

    def vector_space_dimension(self) -> int:
        """The dimension of $K[x]/I$ as a vector space (zero-dimensional
        ideals): the number of solutions counted with multiplicity."""
        return len(self.standard_exponents())

    def multiplication_matrix(self, f: PolyLike[T], order: Optional[OrderLike] = None) -> list[list[T]]:
        """The matrix (as rows) of multiplication by ``f`` on $K[x]/I$ in
        the basis of :meth:`standard_monomials`.

        >>> from sympy_extras.core import symbols
        >>> from sympy_extras.polys.domains import QQ
        >>> from sympy_extras.polys.ideals import Ideal
        >>> x, y = symbols('x y')
        >>> Ideal([x**2 - 2, y - x], x, y, domain=QQ).multiplication_matrix(x) == [[0, 2], [1, 0]]
        True
        """
        G = self._basis(order)
        K = self.domain
        monoms = self.standard_exponents(order)
        index = {m: i for i, m in enumerate(monoms)}
        g = self._poly(f, order)
        rows = [[K.zero for _ in monoms] for _ in monoms]
        for j, m in enumerate(monoms):
            p = normal_form(g.mul_monom(m), G)
            for mm, c in p.terms.items():
                rows[index[mm]][j] = c
        return rows

    def minimal_polynomial(self, f: PolyLike[T], y: Optional[Symbol] = None) -> DensePoly[T]:
        """The monic polynomial ``p`` of least degree with ``p(f)`` in the
        ideal (zero-dimensional ideals): the minimal polynomial of
        multiplication by ``f`` on $K[x]/I$ applied to the class of 1, in
        ``y``. It generates the ideal of the polynomials of ``f`` which
        belong to ``I``.

        >>> from sympy_extras.core import symbols
        >>> from sympy_extras.polys.domains import QQ
        >>> from sympy_extras.polys.ideals import Ideal
        >>> x, y, z = symbols('x y z')
        >>> Ideal([x**2 - 2, y**2 - 3], x, y, domain=QQ).minimal_polynomial(x + y, z).to_expr()
        z**4 - 10*z**2 + 1
        """
        self._require_zero_dimensional()
        variable = Symbol('y') if y is None else y
        M = self.multiplication_matrix(f)
        K = self.domain
        dim = len(M)
        # the Krylov vectors M**k e_1 (the class of 1 is the first standard
        # monomial), reduced against the previous ones while the combination
        # of the original vectors which gives them is kept: the first one
        # reduced to zero is the linear dependency
        pivots: list[tuple[int, list[T], dict[int, T]]] = []
        v: list[T] = [K.one] + [K.zero]*(dim - 1)
        k = 0
        while True:
            vector = list(v)
            combination: dict[int, T] = {k: K.one}
            for pivot, row, row_combination in pivots:
                c = vector[pivot]
                if K.is_zero(c):
                    continue
                vector = [K.sub(a, K.mul(c, b)) for a, b in zip(vector, row)]
                for j, b in row_combination.items():
                    combination[j] = K.sub(combination.get(j, K.zero), K.mul(c, b))
            nonzero = [i for i, a in enumerate(vector) if not K.is_zero(a)]
            if not nonzero:
                return DensePoly([combination.get(j, K.zero) for j in range(k, -1, -1)], K, variable)
            inverse = K.invert(vector[nonzero[0]])
            pivots.append((nonzero[0], [K.mul(a, inverse) for a in vector],
                           {j: K.mul(b, inverse) for j, b in combination.items()}))
            v = [_dot(row, v, K) for row in M]
            k += 1

    def rur(self, t: Optional[Symbol] = None) -> _rur.RationalUnivariateRepresentation:
        """The rational univariate representation of the ideal
        (zero-dimensional ideals over ``QQ``, Rouillier's algorithm,
        :func:`sympy_extras.polys.rur.rur`): a separating linear form, the
        squarefree polynomial in ``t`` of its values at the zeros and the
        coordinates of the zeros as rational functions of the root.

        >>> from sympy_extras.core import symbols
        >>> from sympy_extras.polys.domains import QQ
        >>> from sympy_extras.polys.ideals import Ideal
        >>> x, y, t = symbols('x y t')
        >>> R = Ideal([x**2 - 2, y - x**3], x, y, domain=QQ).rur(t)
        >>> R.polynomial.to_expr(), [g.to_expr() for g in R.polynomials()]
        (t**2 - 2, [t, 2*t])
        """
        return _rur.rur(self._over_rationals(), t=t)

    def _over_rationals(self) -> Ideal[Fraction]:
        """The same ideal over ``QQ`` (``NotImplementedError`` for another
        coefficient field), with its basis for the default order when it
        is known."""
        if not isinstance(self.domain, RationalField):
            raise NotImplementedError("real zeros are counted for ideals over QQ, not %s" % (self.domain,))
        result = Ideal.from_terms([_rational_terms(g) for g in self.gens], self.symbols, QQ, self.order)
        known = self._bases.get(self.order)
        if known is not None:
            result._set_basis([_rational_terms(g) for g in known])
        return result

    def univariate(self, x: Symbol) -> DensePoly[T]:
        """The monic generator of $I \\cap K[x]$ for a variable ``x``
        (zero-dimensional ideals).

        >>> from sympy_extras.core import symbols
        >>> from sympy_extras.polys.domains import QQ
        >>> from sympy_extras.polys.ideals import Ideal
        >>> x, y = symbols('x y')
        >>> Ideal([x**2 - y, y**2 - 1], x, y, domain=QQ).univariate(x).to_expr()
        x**4 - 1
        """
        if x not in self.symbols:
            raise ValueError("%s is not a variable of the ring" % (x,))
        return self.minimal_polynomial(x, x)

    def radical(self) -> Ideal[T]:
        """The radical of the ideal: by Seidenberg's lemma in dimension
        zero (the ideal plus the squarefree parts of the univariate
        polynomials in each variable), and as the intersection of the
        saturated ideals of squarefree regular chains otherwise
        (:func:`sympy_extras.polys.idealdecomposition.radical`, over the
        rationals).

        >>> from sympy_extras.core import symbols
        >>> from sympy_extras.polys.domains import QQ
        >>> from sympy_extras.polys.ideals import Ideal
        >>> x, y = symbols('x y')
        >>> Ideal([x**2, y**2 - 2*y + 1], x, y, domain=QQ).radical()
        Ideal([x, y - 1], x, y)
        """
        if self.is_whole_ring():
            return self.reduced()
        if not self.is_zero_dimensional():
            from .idealdecomposition import radical
            return radical(self)
        R = self.ring()
        n = len(self.symbols)
        extra: list[PolyElement[T]] = []
        for i, s in enumerate(self.symbols):
            p = self.univariate(s)
            sqf = p.quo(p.gcd(p.diff()))
            d = sqf.degree
            extra.append(R.from_dict({tuple(d - j if k == i else 0 for k in range(n)): c
                                      for j, c in enumerate(sqf.rep)}))
        return self._new(list(self._basis()) + extra).reduced()

    def reduced(self) -> Ideal[T]:
        """The same ideal generated by its reduced Gröbner basis."""
        G = self._basis()
        result = self._new(G)
        result._set_basis([g.terms for g in G])
        return result

    def is_radical(self) -> bool:
        """Whether the ideal equals its radical.

        >>> from sympy_extras.core import symbols
        >>> from sympy_extras.polys.domains import QQ
        >>> from sympy_extras.polys.ideals import Ideal
        >>> x, y = symbols('x y')
        >>> Ideal([x**2 - 1, y - x], x, y, domain=QQ).is_radical()
        True
        """
        return self.radical() == self

    def height(self) -> int:
        """The height (the codimension) of a proper ideal: the number of
        variables minus the dimension."""
        if self.is_whole_ring():
            raise ValueError("the whole ring has no height")
        return len(self.symbols) - self.dimension()

    def equidimensional_parts(self) -> list[Ideal[T]]:
        """The equidimensional parts of the radical, by decreasing
        dimension (:func:`~sympy_extras.polys.idealdecomposition.equidimensional_parts`)."""
        from .idealdecomposition import equidimensional_parts
        return equidimensional_parts(self)

    def minimal_primes(self) -> list[Ideal[T]]:
        """The minimal primes of the ideal, by decreasing dimension
        (:func:`~sympy_extras.polys.idealdecomposition.minimal_primes`)."""
        from .idealdecomposition import minimal_primes
        return minimal_primes(self)

    def primary_decomposition(self) -> list[tuple[Ideal[T], Ideal[T]]]:
        """An irredundant primary decomposition, as pairs of a primary
        ideal and its associated prime
        (:func:`~sympy_extras.polys.idealdecomposition.primary_decomposition`)."""
        from .idealdecomposition import primary_decomposition
        return primary_decomposition(self)

    def associated_primes(self) -> list[Ideal[T]]:
        """The associated primes, minimal and embedded, by decreasing
        dimension (:func:`~sympy_extras.polys.idealdecomposition.associated_primes`)."""
        from .idealdecomposition import associated_primes
        return associated_primes(self)

    def is_primary(self) -> bool:
        """Whether the ideal is primary
        (:func:`~sympy_extras.polys.idealdecomposition.is_primary`)."""
        from .idealdecomposition import is_primary
        return is_primary(self)

    def is_maximal(self) -> bool:
        """Whether the ideal is maximal, i.e. $K[x]/I$ is a field: the
        ideal has dimension zero, is radical, and the minimal polynomial
        of a separating linear form (one with degree the dimension of
        $K[x]/I$, found among :func:`separating_forms`) is irreducible.

        >>> from sympy_extras.core import symbols
        >>> from sympy_extras.polys.domains import QQ
        >>> from sympy_extras.polys.ideals import Ideal
        >>> x, y = symbols('x y')
        >>> Ideal([x**2 - 2, y**2 - 2], x, y, domain=QQ).is_maximal()
        False
        >>> Ideal([x**2 - 2, y - x], x, y, domain=QQ).is_maximal()
        True
        """
        if self.is_whole_ring() or not self.is_zero_dimensional():
            return False
        if not self.is_radical():
            return False
        dim = self.vector_space_dimension()
        if dim == 1:
            return True
        y = Dummy('y')
        R = self.ring()
        K = self.domain
        n = len(self.symbols)
        units = [tuple(1 if k == i else 0 for k in range(n)) for i in range(n)]
        tails: Iterator[tuple[int, ...]] = separating_coefficients(n, dim) if n >= 2 else iter([()])
        # one of the forms separates the dim zeros (see separating_forms)
        for tail in tails:
            terms: Terms[T] = {units[0]: K.one}
            for c, unit in zip(tail, units[1:]):
                if c:
                    terms[unit] = K.from_int(c)
            p = self.minimal_polynomial(R.from_dict(terms), y)
            if p.degree < dim:
                continue
            _, factors = p.factor_list()
            return len(factors) == 1 and factors[0][1] == 1
        raise RuntimeError("no separating linear form was found")

    def is_prime(self) -> bool:
        """Whether the ideal is prime: maximal in dimension zero, and its
        own only minimal prime otherwise.

        >>> from sympy_extras.core import symbols
        >>> from sympy_extras.polys.domains import QQ
        >>> from sympy_extras.polys.ideals import Ideal
        >>> x, y = symbols('x y')
        >>> Ideal([x**2 + 1, y - x], x, y, domain=QQ).is_prime()
        True
        """
        if self.is_whole_ring():
            return False
        if self.is_zero_dimensional():
            return self.is_maximal()
        primes = self.minimal_primes()
        return len(primes) == 1 and primes[0] == self

    # ------------------------------------------------------------------
    # change of order

    def change_order(self, order: OrderLike) -> list[PolyElement[T]]:
        """The reduced Gröbner basis for another order, converted with FGLM
        for zero-dimensional ideals and with the Gröbner walk
        (:func:`groebner_walk`) otherwise, and cached.

        >>> from sympy_extras.core import symbols
        >>> from sympy_extras.polys.domains import QQ
        >>> from sympy_extras.polys.ideals import Ideal
        >>> x, y, z = symbols('x y z')
        >>> [str(g) for g in Ideal([x*z - y**2, x**2 - y*z], x, y, z, domain=QQ).change_order('lex')]
        ['x**2 - y*z', 'x*y**2 - y*z**2', 'x*z - y**2', 'y**4 - y*z**3']
        """
        target = as_order(order)
        if target in self._bases:
            return self.groebner_basis(target)
        G = self._basis()
        R = self.ring()
        R2 = self.ring(target)
        simple = (lex, grevlex, grlex)
        if self.is_zero_dimensional() and target in simple and self.order in simple:
            converted = [PolyElement(g.terms, R2) for g in fglm(G, R, target)]
        else:
            converted = [PolyElement(g.terms, R2) for g in groebner_walk(G, R, target)]
        self._bases[target] = converted
        return self.groebner_basis(target)


def _rational_terms(f: PolyElement[T]) -> Terms[Fraction]:
    """The terms of a polynomial over a domain standing for ``QQ``, with
    rational coefficients."""
    K = f.ring.domain
    return {m: QQ.convert(c, K) for m, c in f.terms.items()}


def _dot(row: Sequence[T], v: Sequence[T], K: Domain[T]) -> T:
    total = K.zero
    for a, b in zip(row, v):
        if not K.is_zero(a) and not K.is_zero(b):
            total = K.add(total, K.mul(a, b))
    return total


# rur builds on this module (imported at the end: it imports Ideal from here)
from . import rur as _rur  # noqa: E402
