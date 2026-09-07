r"""Polynomial ideals: operations built on Gröbner bases.

SymPy computes Gröbner bases (:func:`sympy.groebner`, Buchberger and F5B),
reduces modulo them and converts them with FGLM for zero-dimensional
ideals; its :mod:`sympy.polys.agca` ideals support sums, products,
intersections and quotients but leave saturation, radicals, dimension,
primality and the like unimplemented. :class:`Ideal` adds those, on top of
SymPy's Gröbner basis routines:

* elimination ideals with block orders (:meth:`Ideal.eliminate`),
  intersections, quotients ``I : J``, saturations ``I : J**oo`` and radical
  membership;
* the Krull dimension, the Hilbert series, the Hilbert polynomial and the
  degree, from the leading term ideal of a Gröbner basis for a graded
  order;
* for zero-dimensional ideals: the standard monomials and the vector space
  dimension of the quotient algebra, multiplication matrices, univariate
  polynomials in each variable, the radical (Seidenberg's lemma) and the
  tests for radical, prime and maximal ideals;
* conversion between monomial orders with FGLM (zero-dimensional) or the
  Gröbner walk (:mod:`sympy_extras.polys.groebnerwalk`).

Examples
========

>>> from sympy.abc import x, y, z
>>> from sympy_extras.polys.ideals import Ideal
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

from typing import Iterable, Optional, Sequence, Union

from sympy.core.expr import Expr
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import binomial
from sympy.polys.domains import QQ
from sympy.polys.domains.domain import Domain
from sympy.polys.fglmtools import matrix_fglm
from sympy.polys.groebnertools import groebner as _groebner
from sympy.polys.matrices import DomainMatrix
from sympy.polys.monomials import monomial_divides
from sympy.polys.orderings import MonomialOrder, grevlex, lex
from sympy.polys.polytools import Poly
from sympy.polys.rings import ring as _ring, PolyElement, PolyRing
from sympy.matrices.dense import MutableDenseMatrix

from sympy_extras._typing import Monomial, OrderSpec, as_symbol, free_symbols, sorted_symbols

from .groebnerwalk import groebner_walk
from .orderings import as_order, elimination_order

__all__ = ['Ideal', 'hilbert_numerator']


def _is_graded(order: OrderSpec) -> bool:
    return as_order(order) in (grevlex, as_order('grlex'))


def hilbert_numerator(monomials: Iterable[Sequence[int]], t: Symbol) -> Poly:
    r"""Numerator $N(t)$ of the Hilbert series $N(t)/(1 - t)^n$ of
    $K[x_1, \ldots, x_n]/M$ for a monomial ideal $M$ given by the exponent
    vectors of its generators.

    >>> from sympy.abc import t
    >>> from sympy_extras.polys.ideals import hilbert_numerator
    >>> hilbert_numerator([(2, 0), (0, 2)], t)
    Poly(t**4 - 2*t**2 + 1, t, domain='ZZ')
    """
    monomials = _minimal_monomials([tuple(m) for m in monomials])
    return Poly(_hilbert_numerator(monomials, t), t)


def _minimal_monomials(monomials: Iterable[Monomial]) -> list[Monomial]:
    minimal: list[Monomial] = []
    for m in sorted(set(monomials), key=sum):
        if not any(monomial_divides(u, m) for u in minimal):
            minimal.append(m)
    return minimal


def _hilbert_numerator(monomials: list[Monomial], t: Symbol) -> Expr:
    if not monomials:
        return S.One
    m = monomials[-1]
    rest = monomials[:-1]
    if not rest:
        return 1 - t**sum(m)
    # (rest : m) = <u/gcd(u, m)>
    quotient = _minimal_monomials([tuple(max(a - b, 0) for a, b in zip(u, m)) for u in rest])
    if any(sum(q) == 0 for q in quotient):
        return _hilbert_numerator(rest, t)
    return _hilbert_numerator(rest, t) - t**sum(m)*_hilbert_numerator(quotient, t)


class Ideal:
    """An ideal of a polynomial ring over a field, given by generators.

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

    Gröbner bases are computed lazily with :func:`sympy.groebner` and
    cached per order.
    """

    def __init__(self, gens: Iterable[Union[Expr, Poly]], *symbols: Symbol,
                 domain: Optional[Domain] = None, order: OrderSpec = 'grevlex') -> None:
        gens = [sympify(g) for g in gens]
        polys = [g.as_expr() if isinstance(g, Poly) else g for g in gens]
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
        self.domain = domain
        self.order = as_order(order)
        self._rings: dict[object, PolyRing] = {}
        self._bases: dict[object, list[PolyElement]] = {}
        R = self.ring()
        self.gens: list[PolyElement] = [g for g in (R.from_expr(p) for p in polys) if g]

    # ------------------------------------------------------------------
    # rings and bases

    def ring(self, order: Optional[OrderSpec] = None) -> PolyRing:
        """The polynomial ring with the given order."""
        order = self.order if order is None else as_order(order)
        R = self._rings.get(order)
        if R is None:
            R = _ring(self.symbols, self.domain, order)[0]
            self._rings[order] = R
        return R

    def _basis(self, order: Optional[OrderSpec] = None) -> list[PolyElement]:
        """The reduced Gröbner basis for the order, as ring elements."""
        order = self.order if order is None else as_order(order)
        G = self._bases.get(order)
        if G is None:
            R = self.ring(order)
            gens = [R.from_dict(dict(g)) for g in self.gens]
            G = _groebner(gens, R) if gens else []
            G = sorted([g for g in G if g], key=lambda g: R.order(g.LM), reverse=True)
            self._bases[order] = G
        return G

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
        g = self._from_expr(f, order)
        G = self._basis(order)
        r = g.rem(G) if G else g
        return r.as_expr()

    def contains(self, f: Union[Expr, Poly, PolyElement]) -> bool:
        """Whether the polynomial ``f`` belongs to the ideal."""
        g = self._from_expr(f)
        G = self._basis()
        return (g.rem(G) if G else g) == 0

    def __contains__(self, f: Union[Expr, Poly, PolyElement]) -> bool:
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

        Computed with a Gröbner basis for a block order having the
        eliminated variables in the first block.

        >>> from sympy.abc import t, x, y
        >>> from sympy_extras.polys.ideals import Ideal
        >>> Ideal([x - t**2, y - t**3], t, x, y).eliminate([t])
        Ideal([x**3 - y**2], x, y)
        """
        eliminated = [as_symbol(s) for s in symbols]
        for s in eliminated:
            if s not in self.symbols:
                raise ValueError("%s is not a variable of the ring" % (s,))
        keep = [s for s in self.symbols if s not in eliminated]
        if not eliminated:
            return self
        ordered = eliminated + keep
        R = _ring(ordered, self.domain, elimination_order(len(eliminated), len(ordered)))[0]
        gens = [R.from_expr(g.as_expr()) for g in self.gens]
        G = _groebner(gens, R) if gens else []
        indices = set(range(len(eliminated)))
        kept = [g for g in G if all(all(m[i] == 0 for i in indices) for m in g.monoms())]
        return Ideal([g.as_expr() for g in kept], *keep, domain=self.domain, order=self.order)

    def intersect(self, other: Ideal) -> Ideal:
        """The intersection with ``other``, as $(tI + (1 - t)J) \\cap K[x]$."""
        self._same_ring(other)
        t = Dummy('t')
        te = t
        gens = [te*g.as_expr() for g in self.gens] + [(1 - te)*g.as_expr() for g in other.gens]
        J = Ideal(gens, t, *self.symbols, domain=self.domain, order=self.order)
        return J.eliminate([t])

    def quotient(self, other: Union[Ideal, Expr, Poly, PolyElement]) -> Ideal:
        """The ideal quotient ``I : J`` (``other`` may also be a single
        polynomial): the polynomials ``h`` with ``h*J`` in ``I``.

        >>> from sympy.abc import x, y
        >>> from sympy_extras.polys.ideals import Ideal
        >>> Ideal([x**2*y, x*y**2], x, y).quotient(Ideal([x*y], x, y))
        Ideal([x, y], x, y)
        """
        if not isinstance(other, Ideal):
            other = self._new([self._from_expr(other)])
        self._same_ring(other)
        result: Optional[Ideal] = None
        for f in other.gens:
            fI = self._new([f])
            common = self.intersect(fI)
            quot = []
            for g in common._basis():
                q, r = divmod(g, self._from_expr(f))
                if r:
                    raise RuntimeError("inexact division in the ideal quotient")
                quot.append(q.monic())
            part = self._new(quot) if quot else self._new([])
            result = part if result is None else result.intersect(part)
        if result is None:
            return self._new([self.ring().one])
        return result

    def saturate(self, other: Union[Ideal, Expr, Poly, PolyElement]) -> Ideal:
        """The saturation ``I : J**oo``, the union of the quotients
        ``I : J**n``, computed by iterating the quotient until it is
        stable.

        >>> from sympy.abc import x, y
        >>> from sympy_extras.polys.ideals import Ideal
        >>> Ideal([x**3*y, x**2*y**2], x, y).saturate(Ideal([x], x, y))
        Ideal([y], x, y)
        """
        if not isinstance(other, Ideal):
            other = self._new([self._from_expr(other)])
        current = self
        while True:
            following = current.quotient(other)
            if following == current:
                return following
            current = following

    def radical_contains(self, f: Union[Expr, Poly, PolyElement]) -> bool:
        """Whether ``f`` belongs to the radical of the ideal, i.e. some
        power of ``f`` belongs to the ideal (Rabinowitsch's trick).

        >>> from sympy.abc import x, y
        >>> from sympy_extras.polys.ideals import Ideal
        >>> I = Ideal([x**2, y**3], x, y)
        >>> I.radical_contains(x*y), I.contains(x*y)
        (True, False)
        """
        t = Dummy('t')
        f = self._from_expr(f).as_expr()
        J = Ideal([g.as_expr() for g in self.gens] + [1 - t*f], t, *self.symbols,
                  domain=self.domain, order=self.order)
        return J.is_whole_ring()

    # ------------------------------------------------------------------
    # dimension, Hilbert series

    def leading_monomials(self, order: Optional[OrderSpec] = None) -> list[Monomial]:
        """The leading monomials of the Gröbner basis, as exponent
        tuples."""
        return [g.LM for g in self._basis(order)]

    def _graded_order(self) -> MonomialOrder:
        return self.order if _is_graded(self.order) else grevlex

    def hilbert_series(self, t: Optional[Symbol] = None) -> Expr:
        r"""The Hilbert series of $K[x]/I$ as a rational function
        $N(t)/(1 - t)^n$ (for a non-homogeneous ideal, the series of the
        associated graded ring, computed from the leading term ideal of a
        Gröbner basis for a graded order).

        >>> from sympy.abc import x, y, z, t
        >>> from sympy_extras.polys.ideals import Ideal
        >>> Ideal([x**2 - y*z, x*y - z**2, y**2 - x*z], x, y, z).hilbert_series(t)
        (2*t + 1)/(1 - t)
        """
        t = Symbol('t') if t is None else t
        n = len(self.symbols)
        N = hilbert_numerator(self.leading_monomials(self._graded_order()), t)
        num, den = N, Poly((1 - t)**n, t)
        g = num.gcd(den)
        num, den = num.quo(g), den.quo(g)
        # den is a unit times (1 - t)**m; the unit is its value at t = 0
        return (num.as_expr()/den.eval(0))/(1 - t)**den.degree()

    def _hilbert_data(self) -> tuple[Poly, int]:
        """``(q, dim)`` with ``q`` the numerator of the Hilbert series
        reduced by the powers of ``1 - t`` and ``dim`` the Krull
        dimension."""
        t = Dummy('t')
        n = len(self.symbols)
        N = hilbert_numerator(self.leading_monomials(self._graded_order()), t)
        s = 0
        one_minus_t = Poly(1 - t, t)
        while not N.is_zero:
            q, r = N.div(one_minus_t)
            if not r.is_zero:
                break
            N, s = q, s + 1
        return N, n - s

    def dimension(self) -> int:
        """The Krull dimension of $K[x]/I$ (``-1`` for the whole ring).

        >>> from sympy.abc import x, y, z
        >>> from sympy_extras.polys.ideals import Ideal
        >>> Ideal([x*y, x*z], x, y, z).dimension()
        2
        >>> Ideal([x**2 - 1, y - x], x, y).dimension()
        0
        """
        if self.is_whole_ring():
            return -1
        return self._hilbert_data()[1]

    def is_zero_dimensional(self) -> bool:
        return self.dimension() == 0

    def degree(self) -> int:
        """The degree of the ideal: the leading coefficient of the Hilbert
        polynomial times the factorial of the dimension (for a
        zero-dimensional ideal, the number of solutions with
        multiplicities).

        >>> from sympy.abc import x, y, z
        >>> from sympy_extras.polys.ideals import Ideal
        >>> Ideal([x**2 - y*z, x*y - z**2, y**2 - x*z], x, y, z).degree()
        3
        >>> Ideal([x**2 - 1, y**3 - y], x, y).degree()
        6
        """
        q, _ = self._hilbert_data()
        return int(q.eval(1))

    def hilbert_polynomial(self, d: Optional[Symbol] = None) -> Expr:
        r"""The Hilbert polynomial $HP(d)$ of $K[x]/I$: the dimension of the
        degree ``d`` part for ``d`` large (of the associated graded ring
        for a non-homogeneous ideal).

        >>> from sympy.abc import x, y, z, d
        >>> from sympy_extras.polys.ideals import Ideal
        >>> Ideal([x**2 - y*z, x*y - z**2, y**2 - x*z], x, y, z).hilbert_polynomial(d)
        3
        >>> Ideal([x*y], x, y, z).hilbert_polynomial(d)
        2*d + 1
        """
        d = Symbol('d') if d is None else d
        q, dim = self._hilbert_data()
        if dim <= 0:
            return S.Zero
        result = S.Zero
        for (i,), c in q.terms():
            result += c*binomial(d - i + dim - 1, dim - 1)
        return result.expand(func=True)

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
        >>> from sympy_extras.polys.ideals import Ideal
        >>> Ideal([x**2 - y, y**2 - 1], x, y).standard_monomials()
        [1, y, x, x*y]
        """
        self._require_zero_dimensional()
        lms = self.leading_monomials(order)
        n = len(self.symbols)
        R = self.ring(order)
        found: list[Monomial] = []
        seen: set[Monomial] = set()
        queue: list[Monomial] = [(0,)*n]
        while queue:
            m = queue.pop(0)
            if m in seen or any(monomial_divides(lm, m) for lm in lms):
                continue
            seen.add(m)
            found.append(m)
            for i in range(n):
                queue.append(m[:i] + (m[i] + 1,) + m[i + 1:])
        found.sort(key=R.order)
        return [R.from_dict({m: R.domain.one}).as_expr() for m in found]

    def vector_space_dimension(self) -> int:
        """The dimension of $K[x]/I$ as a vector space (zero-dimensional
        ideals): the number of solutions counted with multiplicity."""
        return len(self.standard_monomials())

    def multiplication_matrix(self, f: Union[Expr, Poly, PolyElement], order: Optional[OrderSpec] = None) -> MutableDenseMatrix:
        """The matrix of multiplication by ``f`` on $K[x]/I$ in the basis of
        :meth:`standard_monomials`, as a :class:`~sympy.Matrix`.

        >>> from sympy.abc import x, y
        >>> from sympy_extras.polys.ideals import Ideal
        >>> Ideal([x**2 - 2, y - x], x, y).multiplication_matrix(x)
        Matrix([[0, 2], [1, 0]])
        """
        return self._multiplication_matrix(f, order).to_Matrix()

    def _multiplication_matrix(self, f: Union[Expr, Poly, PolyElement], order: Optional[OrderSpec] = None) -> DomainMatrix:
        R = self.ring(order)
        G = self._basis(order)
        basis = self.standard_monomials(order)
        monoms = [R.from_expr(b).LM for b in basis]
        index = {m: i for i, m in enumerate(monoms)}
        f = self._from_expr(f, order)
        columns = []
        for m in monoms:
            p = (f*R.from_dict({m: R.domain.one})).rem(G)
            col = [R.domain.zero]*len(monoms)
            for mm, c in p.terms():
                col[index[mm]] = c
            columns.append(col)
        rows = [[columns[j][i] for j in range(len(monoms))] for i in range(len(monoms))]
        return DomainMatrix(rows, (len(monoms), len(monoms)), R.domain)

    def minimal_polynomial(self, f: Union[Expr, Poly, PolyElement], y: Optional[Symbol] = None) -> Poly:
        """The monic polynomial ``p`` of least degree with ``p(f)`` in the
        ideal (zero-dimensional ideals): the minimal polynomial of
        multiplication by ``f`` on $K[x]/I$ applied to the class of 1, as a
        ``Poly`` in ``y``. It generates the ideal of the polynomials of
        ``f`` which belong to ``I``.

        >>> from sympy.abc import x, y, z
        >>> from sympy_extras.polys.ideals import Ideal
        >>> Ideal([x**2 - 2, y**2 - 3], x, y).minimal_polynomial(x + y, z)
        Poly(z**4 - 10*z**2 + 1, z, domain='QQ')
        """
        self._require_zero_dimensional()
        y_ = Symbol('y') if y is None else as_symbol(y)
        M = self._multiplication_matrix(f)
        K = M.domain
        dim = M.shape[0]
        # Krylov sequence of the class of 1 (the first standard monomial)
        v = DomainMatrix([[K.one]] + [[K.zero]]*(dim - 1), (dim, 1), K)
        vectors = [v]
        while True:
            v = M*v
            A = DomainMatrix.hstack(*vectors, v)
            ns = A.nullspace()
            if ns.shape[0]:
                coeffs = [ns.to_Matrix()[0, i] for i in range(ns.shape[1])]
                lead = coeffs[-1]
                coeffs = [c/lead for c in coeffs]
                return Poly(sum(c*y_**i for i, c in enumerate(coeffs)), y_, domain=self.domain)
            vectors.append(v)

    def univariate(self, x: Symbol) -> Poly:
        """The monic generator of $I \\cap K[x]$ for a variable ``x``
        (zero-dimensional ideals), as a ``Poly``.

        >>> from sympy.abc import x, y
        >>> from sympy_extras.polys.ideals import Ideal
        >>> Ideal([x**2 - y, y**2 - 1], x, y).univariate(x)
        Poly(x**4 - 1, x, domain='QQ')
        """
        x_ = as_symbol(x)
        if x_ not in self.symbols:
            raise ValueError("%s is not a variable of the ring" % (x_,))
        return self.minimal_polynomial(x_, x_)

    def radical(self) -> Ideal:
        """The radical of a zero-dimensional ideal (Seidenberg's lemma: the
        ideal plus the squarefree parts of the univariate polynomials in
        each variable).

        >>> from sympy.abc import x, y
        >>> from sympy_extras.polys.ideals import Ideal
        >>> Ideal([x**2, y**2 - 2*y + 1], x, y).radical()
        Ideal([x, y - 1], x, y)
        """
        if self.is_whole_ring():
            return self.reduced()
        self._require_zero_dimensional()
        extra = []
        for s in self.symbols:
            p = self.univariate(s)
            sqf = p.quo(p.gcd(p.diff(s)))
            extra.append(sqf.as_expr())
        return self._new([g.as_expr() for g in self._basis()] + extra).reduced()

    def reduced(self) -> Ideal:
        """The same ideal generated by its reduced Gröbner basis."""
        return self._new([g.as_expr() for g in self._basis()])

    def is_radical(self) -> bool:
        """Whether a zero-dimensional ideal equals its radical."""
        return self.radical() == self

    def is_maximal(self) -> bool:
        """Whether a zero-dimensional ideal is maximal, i.e. $K[x]/I$ is a
        field: the ideal is radical and the minimal polynomial of a generic
        linear form has degree the dimension of $K[x]/I$ and is
        irreducible.

        >>> from sympy.abc import x, y
        >>> from sympy_extras.polys.ideals import Ideal
        >>> Ideal([x**2 - 2, y**2 - 2], x, y).is_maximal()
        False
        >>> Ideal([x**2 - 2, y - x], x, y).is_maximal()
        True
        """
        if self.is_whole_ring():
            return False
        self._require_zero_dimensional()
        if not self.is_radical():
            return False
        dim = self.vector_space_dimension()
        if dim == 1:
            return True
        y = Dummy('y')
        n = len(self.symbols)
        for coefficients in ([1]*n, list(range(1, n + 1)), [i**2 + 1 for i in range(n)],
                             [3**i for i in range(n)], [(-2)**i for i in range(n)]):
            ell = sum(c*s for c, s in zip(coefficients, self.symbols))
            p = self.minimal_polynomial(ell, y)
            if p.degree() < dim:
                continue
            _, factors = p.factor_list()
            return len(factors) == 1 and factors[0][1] == 1
        raise NotImplementedError("no separating linear form was found")

    def is_prime(self) -> bool:
        """Whether a zero-dimensional ideal is prime (equivalently,
        maximal)."""
        return self.is_maximal()

    # ------------------------------------------------------------------
    # change of order

    def change_order(self, order: OrderSpec) -> list[Poly]:
        """The reduced Gröbner basis for another order, converted with FGLM
        for zero-dimensional ideals and with the Gröbner walk otherwise
        (returned as a list of ``Poly``, and cached).

        >>> from sympy.abc import x, y, z
        >>> from sympy_extras.polys.ideals import Ideal
        >>> Ideal([x*z - y**2, x**2 - y*z], x, y, z).change_order('lex')
        [Poly(x**2 - y*z, x, y, z, domain='QQ'), Poly(x*y**2 - y*z**2, x, y, z, domain='QQ'), Poly(x*z - y**2, x, y, z, domain='QQ'), Poly(y**4 - y*z**3, x, y, z, domain='QQ')]
        >>> Ideal([x**2 - y, y**2 - 1], x, y).change_order('lex')
        [Poly(x**2 - y, x, y, domain='QQ'), Poly(y**2 - 1, x, y, domain='QQ')]
        """
        order = as_order(order)
        if order in self._bases:
            return self.groebner_basis(order)
        G = self._basis()
        R = self.ring()
        if self.is_zero_dimensional() and order in (lex, grevlex, as_order('grlex')) and \
                self.order in (lex, grevlex, as_order('grlex')):
            converted = matrix_fglm(G, R, order)
            R2 = self.ring(order)
            converted = [R2.from_dict(dict(g)) for g in converted]
        else:
            converted = groebner_walk(G, R, order)
        self._bases[order] = converted
        return self.groebner_basis(order)
