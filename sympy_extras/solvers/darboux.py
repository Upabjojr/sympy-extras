"""Darboux polynomials and the Prelle–Singer procedure: elementary and
Liouvillian first integrals of first order equations ``y' = P(x, y)/Q(x,
y)`` with polynomial ``P`` and ``Q``.

The equation is the vector field ``D = Q d/dx + P d/dy`` (a first
integral ``I(x, y)`` is a function with ``D I = 0``). A *Darboux
polynomial* of ``D`` is a polynomial ``f`` with ``D f = g f`` for a
polynomial ``g``, its *cofactor*, of degree at most ``d - 1`` where ``d``
is the degree of the field (the larger of the degrees of ``P`` and
``Q``); its zero set is a union of solution curves. An *exponential
factor* is a function ``exp(A/B)`` with ``D(A/B) = L`` a polynomial of
degree at most ``d - 1`` (the denominator ``B`` is a product of Darboux
polynomials). The *Darboux integrating factors* are the products ``R =
prod f_i**c_i prod exp(e_j A_j/B_j)``: ``R (Q dy - P dx)`` is exact
exactly when ``sum c_i g_i + sum e_j L_j = -div D = -(Q_x + P_y)``, a
linear system for the exponents, and ``sum c_i g_i + sum e_j L_j = 0``
makes ``sum c_i log f_i + sum e_j A_j/B_j`` a first integral at once.

Prelle and Singer [PrelleSinger]_ proved that an equation with an
elementary first integral has an integrating factor ``R`` with ``R**n``
rational for some integer ``n``, hence a Darboux integrating factor with
rational exponents and no exponential factors, and Singer [Singer]_ that
a Liouvillian first integral comes with an integrating factor of the
form ``exp(int U dx + V dy)`` with ``U``, ``V`` rational, which is a
Darboux integrating factor with exponential factors (Christopher
[Christopher]_). The procedure is thus complete up to the degree bound
on the Darboux polynomials, for which no a priori bound is known.

The Darboux polynomials of degree ``N`` are found by undetermined
coefficients with the reduction of Man [Man]_: the homogeneous part of
highest degree ``f_N`` of ``f`` satisfies ``D_d f_N = g_{d-1} f_N`` for the
highest degree parts ``D_d`` of the field and ``g_{d-1}`` of the
cofactor, so every linear factor of ``f_N`` divides ``H = x P_d - y Q_d``
(the equation of the line at infinity's tangencies): ``f_N`` is a product
of irreducible factors of ``H``, enumerated, and ``g_{d-1}`` is
``D_d f_N / f_N``. When ``H`` vanishes identically (the line at infinity
is dicritical, ``D_d = k (x d/dx + y d/dy)``) ``f_N`` is any form of
degree ``N`` and ``g_{d-1} = N k``, and the charts of the projective
space of forms are enumerated instead. The remaining coefficients
satisfy a bilinear system: its linear equations are solved and
substituted repeatedly, and what is left (products of the free
parameters of earlier degrees) is linear in the coefficients of ``f``
with coefficients polynomial in the few cofactor coefficients left,
and goes through a Gaussian elimination with a case split at every
pivot which may vanish, whose leaves are zero-dimensional systems in
the cofactor coefficients (the cofactors of bounded degree are finitely
many), solved for their points in the coefficient field by factoring
the equations and taking lex Gröbner bases of the cases; at each
cofactor the polynomials form an affine space, of which a few members
are factored. Exponential factors with a given denominator are the
nullspace of a linear map. Everything is computed over the field of the
coefficients (the rationals, an algebraic number field ``QQ(alpha)``, or
the rational functions ``QQ(a, b, ...)`` of parameters); Darboux
polynomials irreducible over that field but not over its closure are
found as the products of their conjugates.

The module works on the objects of the core and imports no SymPy: the
polynomials are the typed sparse polynomials of
:mod:`sympy_extras.polys.rings` over the typed fields of
:mod:`sympy_extras.polys.domains` (``QQ``,
:class:`~sympy_extras.polys.domains.algebraicfield.AlgebraicField`,
:class:`~sympy_extras.polys.domains.sparsefractionfield.SparseFractionField`),
factored by Wang's algorithm (:mod:`sympy_extras.polys.multifactor`, the
parameters as further variables) and Trager's
(:mod:`sympy_extras.polys.algebraicfactor`), with the Gröbner bases of
:mod:`sympy_extras.polys.ideals` and :mod:`sympy_extras.polys.fglm` and
the sparse linear algebra of the domains;
:mod:`sympy_extras.solvers.darboux_sympy` is its face on SymPy objects.
One step is not ported and goes through SymPy at run time: the
quadratures of the integrating factor
(:func:`~sympy_extras.solvers.kovacic_sympy.integrate_of_core`).

References
==========

.. [PrelleSinger] M. J. Prelle and M. F. Singer, Elementary first
   integrals of differential equations, Trans. Amer. Math. Soc. 279
   (1983), 215–229.
.. [Singer] M. F. Singer, Liouvillian first integrals of differential
   equations, Trans. Amer. Math. Soc. 333 (1992), 673–688.
.. [Man] Y.-K. Man, Computing closed form solutions of first order ODEs
   using the Prelle–Singer procedure, J. Symbolic Computation 16 (1993),
   423–443.
.. [ManMacCallum] Y.-K. Man and M. A. H. MacCallum, A rational approach
   to the Prelle–Singer algorithm, J. Symbolic Computation 24 (1997),
   31–43.
.. [Christopher] C. Christopher, Liouvillian first integrals of second
   order polynomial differential equations, Electron. J. Differential
   Equations 1999, no. 49, 1–7.
.. [Darboux] G. Darboux, Mémoire sur les équations différentielles
   algébriques du premier ordre et du premier degré, Bull. Sci. Math. 2
   (1878), 60–96, 123–144, 151–200.

Examples
========

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.solvers.darboux import darboux_polynomials, darboux_first_integral
>>> x, y = Symbol('x'), Symbol('y')
>>> darboux_polynomials(y*(x - 1), x*(1 - y), x, y)
[DarbouxPolynomial(y, x - 1), DarbouxPolynomial(x, 1 - y)]
>>> darboux_first_integral(y*(x - 1), x*(1 - y), x, y)
-x - y + log(x) + log(y)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from fractions import Fraction
from itertools import combinations_with_replacement
from math import lcm
from typing import Final, Generic, Optional, Sequence, TypeVar

from sympy_extras.core.add import Add, add
from sympy_extras.core.mul import Mul
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import AppliedUndef, Derivative, count_ops, expand
from sympy_extras.core.logic import true
from sympy_extras.core.numbers import I, Integer, Rational
from sympy_extras.core.power import Pow, sqrt
from sympy_extras.core.relational import Eq, Ne
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.elementary.exponential import Exp, exp, log
from sympy_extras.functions.elementary.piecewise import ExprCondPair, Piecewise
from sympy_extras.polys.domains import QQ
from sympy_extras.polys.domains.algebraicfield import AlgebraicField
from sympy_extras.polys.domains.domain import Domain
from sympy_extras.polys.domains.sparsefractionfield import SparseFracElement, SparseFractionField
from sympy_extras.polys.domains.sparsepolynomialring import RationalPolynomialRing
from sympy_extras.polys.orderings import grevlex, lex
from sympy_extras.polys.rings import Monomial, PolyElement, PolyRing
from sympy_extras._timeout import attempt, remaining_time
from sympy_extras.settings import settings

__all__ = ['DarbouxPolynomial', 'ExponentialFactor', 'DarbouxIntegral', 'vector_field',
           'darboux_polynomials', 'exponential_factors', 'darboux_integrating_factor',
           'darboux_first_integral', 'prelle_singer']

T = TypeVar('T')

#: an exponent vector of a monomial ``x**i y**j``
_Monomial = tuple[int, int]
#: a solution of a linear system: a particular solution, a basis of the
#: homogeneous solutions (as vectors of domain elements) and the free
#: columns, one per basis vector, in which that vector has a one
_Affine = tuple[list[T], list[list[T]], list[int]]
#: a row of a linear system in the polynomial unknowns whose entries are
#: polynomials in the cofactor unknowns: the coefficients of the unknowns
#: and, last, the constant term (``sum c_i a_i + r = 0``)
_Row = list[PolyElement[T]]
#: the constraints on the cofactor unknowns at a leaf of the parametric
#: elimination: polynomials which vanish, polynomials which do not, and
#: the unknowns fixed by the vanishing pivots of degree one (their
#: index and their value in the unknowns fixed later and the free ones)
_Leaf = tuple[list[PolyElement[T]], list[PolyElement[T]], list[tuple[int, PolyElement[T]]]]
#: a point of the unknowns: their values by index
_Point = dict[int, T]

_ZERO: Final[Integer] = Integer(0)
_ONE: Final[Integer] = Integer(1)


# ---------------------------------------------------------------------------
# the results

class DarbouxPolynomial:
    """A Darboux polynomial ``f`` of the field ``D = Q d/dx + P d/dy``,
    irreducible over the coefficient field, and its cofactor ``g``,
    ``D f = g f``.

    Attributes
    ==========

    polynomial : Expr
        ``f``, monic in the lex order of ``x, y``.
    cofactor : Expr
        ``g``, of degree at most ``d - 1``.
    """

    def __init__(self, polynomial: Expr, cofactor: Expr) -> None:
        self.polynomial: Final[Expr] = polynomial
        self.cofactor: Final[Expr] = cofactor

    def __repr__(self) -> str:
        return "DarbouxPolynomial(%s, %s)" % (self.polynomial, self.cofactor)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DarbouxPolynomial):
            return NotImplemented
        return bool(self.polynomial == other.polynomial and self.cofactor == other.cofactor)

    def __hash__(self) -> int:
        return hash((self.polynomial, self.cofactor))


class ExponentialFactor:
    """An exponential factor ``exp(A/B)`` of the field, ``D(A/B) = L`` a
    polynomial of degree at most ``d - 1``, its cofactor.

    Attributes
    ==========

    numerator, denominator : Expr
        ``A`` and ``B``; ``B`` is ``1`` or a product of Darboux
        polynomials.
    cofactor : Expr
        ``L``.
    """

    def __init__(self, numerator: Expr, denominator: Expr, cofactor: Expr) -> None:
        self.numerator: Final[Expr] = numerator
        self.denominator: Final[Expr] = denominator
        self.cofactor: Final[Expr] = cofactor

    @property
    def argument(self) -> Expr:
        """``A/B``."""
        return self.numerator/self.denominator

    @property
    def factor(self) -> Expr:
        """``exp(A/B)``."""
        return exp(self.argument)

    def __repr__(self) -> str:
        return "ExponentialFactor(%s, %s, %s)" % (self.numerator, self.denominator, self.cofactor)


class DarbouxIntegral:
    """The result of the Prelle–Singer procedure on a field: the Darboux
    integrating factor and the first integral, when found.

    Attributes
    ==========

    polynomials : list[DarbouxPolynomial]
        The Darboux polynomials found within the degree bound.
    exponential_factors : list[ExponentialFactor]
        The exponential factors found.
    integrating_factor : Expr or None
        ``R`` with ``R (Q dy - P dx)`` exact.
    first_integral : Expr or None
        ``I(x, y)`` with ``D I = 0``, not constant.
    """

    def __init__(self, polynomials: list[DarbouxPolynomial], factors: list[ExponentialFactor],
                 integrating_factor: Optional[Expr], first_integral: Optional[Expr]) -> None:
        self.polynomials: Final[list[DarbouxPolynomial]] = polynomials
        self.exponential_factors: Final[list[ExponentialFactor]] = factors
        self.integrating_factor: Final[Optional[Expr]] = integrating_factor
        self.first_integral: Final[Optional[Expr]] = first_integral

    def __repr__(self) -> str:
        return "DarbouxIntegral(integrating_factor=%s, first_integral=%s)" % (
            self.integrating_factor, self.first_integral)


# ---------------------------------------------------------------------------
# the coefficient fields: what the search needs beyond the domain

class _Coefficients(ABC, Generic[T]):
    """The field ``K`` of the coefficients: its domain, the factorisation
    of polynomials over it (in any number of variables) and the
    conversions of its elements."""

    def __init__(self, K: Domain[T]) -> None:
        self.K: Final[Domain[T]] = K

    @abstractmethod
    def factor(self, f: PolyElement[T]) -> list[PolyElement[T]]:
        """The monic irreducible factors of positive degree of ``f`` over
        ``K``, each once, in SymPy's order."""

    def rename(self, e: Expr) -> Expr:
        """``e`` with the parameters named as inside the field."""
        return e

    def element(self, e: Expr) -> T:
        """The coefficient ``e`` (free of ``x`` and ``y``) as an element
        of ``K`` (``ValueError`` when it is not one)."""
        return self.K.from_expr(self.rename(e))

    def to_expr(self, c: T) -> Expr:
        """The element as an expression (the parameters restored)."""
        return self.K.to_expr(c)

    def rational(self, c: T) -> Optional[Fraction]:
        """The element as a rational number, when it is one."""
        return None

    def key(self, c: T) -> str:
        """A hashable key of the element (for removing repetitions)."""
        return self.K.to_str(c)

    @property
    def is_rationals(self) -> bool:
        return False

    def roots(self, f: PolyElement[T], index: int) -> list[T]:
        """The roots in ``K`` of the nonzero ``f``, a polynomial in the
        variable ``index`` alone (the linear factors of its
        factorisation, in their order)."""
        K = self.K
        found: list[T] = []
        for h in self.factor(f):
            if h.degree() != 1:
                continue
            constant = h.coeff(h.ring.zero_monom)
            unit = tuple(1 if k == index else 0 for k in range(h.ring.ngens))
            found.append(K.neg(K.exquo(constant, h.coeff(unit))))
        return found


class _Rationals(_Coefficients[Fraction]):
    def __init__(self) -> None:
        super().__init__(QQ)

    def factor(self, f: PolyElement[Fraction]) -> list[PolyElement[Fraction]]:
        from sympy_extras.polys.polytools import _factor_all
        if f.is_ground:
            return []
        _, factors = _factor_all(f)
        result: list[PolyElement[Fraction]] = []
        for h, _ in factors:
            if h.degree() > 0:
                result.append(f.ring.from_dict({m: Fraction(c) for m, c in h.terms.items()}).monic())
        return result

    def rational(self, c: Fraction) -> Optional[Fraction]:
        return c

    @property
    def is_rationals(self) -> bool:
        return True


class _NumberField(_Coefficients[list[Fraction]]):
    """A number field ``QQ(theta)`` with the images of the algebraic
    numbers which generate it (``I``, the radicals ``b**(1/q)`` of the
    rational bases ``b``, an extension), for the conversion of the
    coefficients."""

    def __init__(self, K: AlgebraicField, images: Optional[dict[Expr, list[Fraction]]] = None) -> None:
        super().__init__(K)
        self.algebraic: Final[AlgebraicField] = K
        self.images: Final[dict[Expr, list[Fraction]]] = {K.alpha_expr: K.alpha} if images is None else images
        radicals: dict[Expr, tuple[int, list[Fraction]]] = {}
        for g, image in self.images.items():
            if isinstance(g, Pow) and isinstance(g.base, Rational) and isinstance(g.exp, Rational) \
                    and g.exp.p == 1 and g.exp.q > 1:
                radicals[g.base] = (g.exp.q, image)
        self.radicals: Final[dict[Expr, tuple[int, list[Fraction]]]] = radicals

    def element(self, e: Expr) -> list[Fraction]:
        K = self.algebraic
        if isinstance(e, Rational):
            return K.from_rational(QQ.from_expr(e))
        image = self.images.get(e)
        if image is not None:
            return image
        if isinstance(e, Pow) and isinstance(e.exp, Rational):
            radical = self.radicals.get(e.base)
            if radical is not None and radical[0] % e.exp.q == 0:
                k = e.exp.p*(radical[0]//e.exp.q)
                return K.pow(radical[1], k) if k >= 0 else K.pow(K.invert(radical[1]), -k)
            if isinstance(e.exp, Integer):
                base = self.element(e.base)
                return K.pow(base, e.exp.p) if e.exp.p >= 0 else K.pow(K.invert(base), -e.exp.p)
        if isinstance(e, Add):
            total = K.zero
            for term in e.args:
                total = K.add(total, self.element(term))
            return total
        if isinstance(e, Mul):
            product = K.one
            for factor in e.args:
                product = K.mul(product, self.element(factor))
            return product
        raise ValueError("%s is not an element of %s" % (e, K.name))

    def to_expr(self, c: list[Fraction]) -> Expr:
        # expanded: the powers of a primitive element sqrt(2) + sqrt(3)
        return expand(self.algebraic.to_expr(c))

    def factor(self, f: PolyElement[list[Fraction]]) -> list[PolyElement[list[Fraction]]]:
        from sympy_extras.polys.algebraicfactor import ext_factor_list
        if f.is_ground:
            return []
        _, factors = ext_factor_list(f)
        return [h.monic() for h, _ in factors if h.degree() > 0]

    def rational(self, c: list[Fraction]) -> Optional[Fraction]:
        if not c:
            return Fraction(0)
        return c[0] if len(c) == 1 else None


class _Parameters(_Coefficients[SparseFracElement[Fraction]]):
    """``QQ(a, b, ...)``: the parameters renamed (the rings map symbols by
    name); the factorisation is that of the numerators over ``QQ`` with
    the parameters as further variables (Gauss' lemma)."""

    def __init__(self, parameters: Sequence[Expr]) -> None:
        names = ['_darboux_p%d' % i for i in range(len(parameters))]
        field = SparseFractionField(RationalPolynomialRing(names))
        super().__init__(field)
        self.field: Final[SparseFractionField[Fraction]] = field
        self.names: Final[list[str]] = names
        self.forward: Final[dict[Basic, Basic]] = {p: Symbol(n) for p, n in zip(parameters, names)}
        self.restore: Final[dict[Basic, Basic]] = {Symbol(n): p for p, n in zip(parameters, names)}

    def rename(self, e: Expr) -> Expr:
        return e.xreplace(self.forward)

    def to_expr(self, c: SparseFracElement[Fraction]) -> Expr:
        return self.K.to_expr(c).xreplace(self.restore)

    def rational(self, c: SparseFracElement[Fraction]) -> Optional[Fraction]:
        if not (c.numer.is_ground and c.denom.is_ground):
            return None
        return c.numer.coeff(c.numer.ring.zero_monom)/c.denom.coeff(c.denom.ring.zero_monom)

    def factor(self, f: PolyElement[SparseFracElement[Fraction]]) -> list[PolyElement[SparseFracElement[Fraction]]]:
        from sympy_extras.polys.polytools import _factor_all
        if f.is_ground:
            return []
        F = self.field
        ring = F.ring
        common = ring.one
        for c in f.terms.values():
            common = ring.lcm(common, c.denom)
        n = f.ring.ngens
        joint = PolyRing(QQ, f.ring.names + self.names)
        terms: dict[Monomial, Fraction] = {}
        for m, c in f.terms.items():
            numerator = c.numer*ring.exquo(common, c.denom)
            for pm, value in numerator.terms.items():
                terms[m + pm] = value
        _, factors = _factor_all(joint.from_dict(terms))
        result: list[PolyElement[SparseFracElement[Fraction]]] = []
        for h, _ in factors:
            if all(sum(m[:n]) == 0 for m in h.terms):
                continue
            grouped: dict[Monomial, dict[Monomial, Fraction]] = {}
            for m, integer in h.terms.items():
                grouped.setdefault(m[:n], {})[m[n:]] = Fraction(integer)
            back = f.ring.from_dict({m: F.from_ring(ring.ring.from_dict(g)) for m, g in grouped.items()})
            result.append(back.monic())
        return result


# ---------------------------------------------------------------------------
# polynomials of the unknowns: substitutions and values

def _substituted(p: PolyElement[T], values: dict[int, PolyElement[T]]) -> PolyElement[T]:
    """``p`` with the variables ``values`` replaced by polynomials of the
    same ring."""
    if not values or not p.terms:
        return p
    ring = p.ring
    powers: dict[tuple[int, int], PolyElement[T]] = {}
    result = ring.zero
    for m, c in p.terms.items():
        term = ring.term(tuple(0 if i in values else e for i, e in enumerate(m)), c)
        for i, e in enumerate(m):
            if e and i in values:
                power = powers.get((i, e))
                if power is None:
                    power = values[i]**e
                    powers[(i, e)] = power
                term = term*power
        result = result + term
    return result


def _evaluated(p: PolyElement[T], point: _Point[T]) -> PolyElement[T]:
    """``p`` with the variables of ``point`` set to its values."""
    if not point or not p.terms:
        return p
    ring = p.ring
    K = ring.domain
    terms: dict[Monomial, T] = {}
    for m, c in p.terms.items():
        value = c
        for i, e in enumerate(m):
            if e and i in point:
                value = K.mul(value, K.pow(point[i], e))
        if K.is_zero(value):
            continue
        monomial = tuple(0 if i in point else e for i, e in enumerate(m))
        s = K.add(terms.get(monomial, K.zero), value)
        if K.is_zero(s):
            terms.pop(monomial, None)
        else:
            terms[monomial] = s
    return PolyElement(terms, ring)


def _value(p: PolyElement[T], point: _Point[T]) -> T:
    """The value of ``p`` at a point of all its variables."""
    evaluated = _evaluated(p, point)
    if not evaluated.is_ground:
        raise ValueError("an unknown is left in %s" % (p,))
    return evaluated.coeff(p.ring.zero_monom)


def _linear_rows(equations: Sequence[PolyElement[T]], gens: Sequence[int],
                 K: Domain[T]) -> tuple[list[list[T]], list[T]]:
    """The rows and right-hand sides of the equations of degree at most
    one in the unknowns ``gens``."""
    rows: list[list[T]] = []
    rhs: list[T] = []
    index = {gen: i for i, gen in enumerate(gens)}
    for equation in equations:
        if equation.degree() > 1:
            continue
        row: list[T] = [K.zero]*len(gens)
        constant: T = K.zero
        for monomial, coefficient in equation.terms.items():
            if sum(monomial) == 0:
                constant = coefficient
            else:
                row[index[monomial.index(1)]] = coefficient
        rows.append(row)
        rhs.append(K.neg(constant))
    return rows, rhs


def _solve_affine(rows: list[list[T]], rhs: list[T], n: int, K: Domain[T]) -> Optional[_Affine[T]]:
    """The solutions of ``rows * v = rhs`` over the field ``K``, ``v`` of
    length ``n``: a particular solution, a basis of the nullspace and the
    free columns (those of the reduced row echelon form), or ``None``
    when there is none."""
    if not rows:
        identity = [[K.one if k == i else K.zero for k in range(n)] for i in range(n)]
        return [K.zero]*n, identity, list(range(n))
    sparse: list[dict[int, T]] = []
    for row, value in zip(rows, rhs):
        entries = {j: c for j, c in enumerate(row + [value]) if not K.is_zero(c)}
        if entries:
            sparse.append(entries)
    echelon = K.sparse_echelon(sparse)
    pivots = [min(r) for r in echelon]
    if n in pivots:
        return None
    particular: list[T] = [K.zero]*n
    for r, column in zip(echelon, pivots):
        particular[column] = r.get(n, K.zero)
    free = [column for column in range(n) if column not in pivots]
    basis: list[list[T]] = []
    for column in free:
        vector: list[T] = [K.zero]*n
        vector[column] = K.one
        for r, pivot in zip(echelon, pivots):
            vector[pivot] = K.neg(r.get(column, K.zero))
        basis.append(vector)
    return particular, basis, free


# ---------------------------------------------------------------------------
# the vector field

class _Field(Generic[T]):
    """The vector field ``Q d/dx + P d/dy`` with its polynomials over the
    coefficient field (the ring in ``x, y``, lex)."""

    def __init__(self, C: _Coefficients[T], P: PolyElement[T], Q: PolyElement[T], x: Symbol, y: Symbol) -> None:
        self.C: Final[_Coefficients[T]] = C
        self.K: Final[Domain[T]] = C.K
        self.ring: Final[PolyRing[T]] = P.ring
        self.P: Final[PolyElement[T]] = P
        self.Q: Final[PolyElement[T]] = Q
        self.x: Final[Symbol] = x
        self.y: Final[Symbol] = y
        self.degree: Final[int] = max(P.degree(), Q.degree())

    def apply(self, f: PolyElement[T]) -> PolyElement[T]:
        """``D f = Q f_x + P f_y``."""
        return self.Q*f.diff(0) + self.P*f.diff(1)

    def divergence(self) -> PolyElement[T]:
        """``Q_x + P_y``."""
        return self.Q.diff(0) + self.P.diff(1)

    def cofactor_of(self, f: PolyElement[T]) -> Optional[PolyElement[T]]:
        """``D f / f`` when the division is exact."""
        quotients, remainder = self.apply(f).div([f])
        if remainder.terms:
            return None
        return quotients[0]

    def homogeneous_part(self, f: PolyElement[T], n: int) -> PolyElement[T]:
        """The terms of total degree ``n`` of ``f``."""
        return PolyElement({m: c for m, c in f.terms.items() if sum(m) == n}, self.ring)

    def monomial(self, m: _Monomial) -> PolyElement[T]:
        return self.ring.term(m, self.K.one)

    def zero(self) -> PolyElement[T]:
        return self.ring.zero

    def one(self) -> PolyElement[T]:
        return self.ring.one

    def expr(self, f: PolyElement[T]) -> Expr:
        """``f`` as an expression in ``x`` and ``y``."""
        terms: list[Expr] = []
        for m in sorted(f.terms, reverse=True):
            terms.append(self.C.to_expr(f.terms[m])*self.x**m[0]*self.y**m[1])
        return add(*terms) if terms else _ZERO


def _monomials(n: int) -> list[_Monomial]:
    """The exponent vectors of the monomials of total degree at most
    ``n`` (none for a negative ``n``)."""
    return [(i, k - i) for k in range(n + 1) for i in range(k, -1, -1)]


def _xy_poly(e: Expr, x: Symbol, y: Symbol, C: _Coefficients[T], ring: PolyRing[T]) -> PolyElement[T]:
    """``e`` as a polynomial in ``x, y`` over the field (``ValueError``
    when it is not one)."""
    K = C.K
    terms: dict[Monomial, T] = {}
    for term in Add.make_args(expand(e)):
        if term == _ZERO:
            continue
        coefficient, dependent = term.as_independent(x, y, as_Add=False)
        i = j = 0
        for factor in Mul.make_args(dependent):
            if factor == _ONE:
                continue
            base, exponent = factor.as_base_exp()
            if base not in (x, y) or not isinstance(exponent, Integer) or exponent.p < 0:
                raise ValueError("%s is not a polynomial in %s and %s" % (e, x, y))
            if base == x:
                i += exponent.p
            else:
                j += exponent.p
        value = C.element(coefficient)
        total = K.add(terms.get((i, j), K.zero), value)
        if K.is_zero(total):
            terms.pop((i, j), None)
        else:
            terms[(i, j)] = total
    return PolyElement(terms, ring)


def vector_field(equation: Basic, f: AppliedUndef) -> Optional[tuple[Expr, Expr, Symbol, Symbol]]:
    """``(P, Q, x, y)`` of the first order equation ``y' = P(x, y)/Q(x, y)``
    with ``P`` and ``Q`` coprime polynomials in ``x`` and ``y`` (``y`` a
    fresh symbol for ``f``), or ``None`` when the equation is not of this
    form.

    Examples
    ========

    >>> from sympy_extras.core import Function, Symbol
    >>> from sympy_extras.solvers.darboux import vector_field
    >>> x = Symbol('x')
    >>> y = Function('y')(x)
    >>> vector_field((x - y)*y.diff(x) - x - y, y)
    (x + y, x - y, x, y)
    """
    from sympy_extras.polys.polytools import cancel
    from sympy_extras.simplify.radsimp import fraction
    x = f.args[0]
    if not isinstance(x, Symbol):
        raise ValueError("the function must depend on a symbol")
    if isinstance(equation, Eq):
        lhs = equation.lhs - equation.rhs
    elif isinstance(equation, Expr):
        lhs = equation
    else:
        raise ValueError("an equation is expected, got %s" % (equation,))
    u, p = Dummy('y'), Dummy('p')
    derivative: dict[Basic, Basic] = {f.diff(x): p}
    function: dict[Basic, Basic] = {f: u}
    replaced = lhs.xreplace(derivative).xreplace(function)
    if replaced.atoms(Derivative) or replaced.has(f):
        raise ValueError("a first order equation in %s is expected" % (f,))
    # p solved from F = 0 when F is linear in p
    a0: Expr = _ZERO
    a1: Expr = _ZERO
    for term in Add.make_args(expand(replaced)):
        coefficient, dependent = term.as_independent(p, as_Add=False)
        if dependent == _ONE:
            a0 = a0 + coefficient
        elif dependent == p:
            a1 = a1 + coefficient
        else:
            return None
    if a1 == _ZERO:
        return None
    numerator, denominator = fraction(cancel(-a0/a1))
    P, Q = numerator, denominator
    if P.is_polynomial(x, u) is not True or Q.is_polynomial(x, u) is not True or Q == _ZERO:
        return None
    name = f.name
    taken = {s.name for s in P.free_symbols | Q.free_symbols if s != u}
    y: Symbol = Dummy(name) if name in taken else Symbol(name)
    named: dict[Basic, Basic] = {u: y}
    return P.xreplace(named), Q.xreplace(named), x, y


# ---------------------------------------------------------------------------
# the Darboux polynomials

class _Unknowns(Generic[T]):
    """The coefficients of ``f`` (``a``) and of its cofactor (``b``) left
    unknown in one chart, as polynomials over ``K`` in those unknowns
    (the variables of ``ring``: the ``a`` first)."""

    def __init__(self, field: _Field[T], ring: PolyRing[T], f_terms: dict[_Monomial, PolyElement[T]],
                 g_terms: dict[_Monomial, PolyElement[T]], a: list[int], b: list[int]) -> None:
        self.field: Final[_Field[T]] = field
        self.ring: Final[PolyRing[T]] = ring
        self.f_terms: Final[dict[_Monomial, PolyElement[T]]] = f_terms
        self.g_terms: Final[dict[_Monomial, PolyElement[T]]] = g_terms
        self.a: Final[list[int]] = a
        self.b: Final[list[int]] = b

    def equations(self) -> list[PolyElement[T]]:
        """The coefficients of ``D f - g f`` (in the lex order of the
        monomials of ``x, y``) as polynomials in the unknowns."""
        field = self.field
        collected: dict[_Monomial, PolyElement[T]] = {}
        zero = self.ring.zero
        for m, c in self.f_terms.items():
            for (i, j), k in field.apply(field.monomial(m)).terms.items():
                collected[(i, j)] = collected.get((i, j), zero) + c.mul_ground(k)
            for n, d in self.g_terms.items():
                product: _Monomial = (m[0] + n[0], m[1] + n[1])
                collected[product] = collected.get(product, zero) - c*d
        return [collected[m] for m in sorted(collected, reverse=True) if collected[m].terms]


class _Propagation(Generic[T]):
    """The linear equations of a bilinear system solved and substituted
    until only nonlinear ones remain: the values of the solved unknowns
    as affine expressions in the free ones. The pivots are taken in the
    first block of columns first (``a`` unless ``b_first``), so that the
    unknowns of the first block are expressed through those of the
    second and the remainder stays linear in the first block."""

    def __init__(self, equations: list[PolyElement[T]], a: list[int], b: list[int], K: Domain[T],
                 b_first: bool = False) -> None:
        self.K: Final[Domain[T]] = K
        self.a: list[int] = list(a)
        self.b: list[int] = list(b)
        self.gens: list[int] = self.b + self.a if b_first else self.a + self.b
        self.values: dict[int, PolyElement[T]] = {}
        self.equations: list[PolyElement[T]] = equations
        self.consistent = True
        self._run()

    def _run(self) -> None:
        K = self.K
        while self.consistent:
            rows, rhs = _linear_rows(self.equations, self.gens, K)
            if not rows:
                return
            solved = _solve_affine(rows, rhs, len(self.gens), K)
            if solved is None:
                self.consistent = False
                return
            particular, basis, free = solved
            substitution: dict[int, PolyElement[T]] = {}
            ring = self.equations[0].ring
            gens = ring.gens
            for i, gen in enumerate(self.gens):
                if i in free:
                    continue
                value = ring.ground(particular[i])
                for vector, column in zip(basis, free):
                    if not K.is_zero(vector[i]):
                        value = value + gens[self.gens[column]].mul_ground(vector[i])
                substitution[gen] = value
            if not substitution:
                return
            self._substitute(substitution)

    def _substitute(self, substitution: dict[int, PolyElement[T]]) -> None:
        for gen, value in self.values.items():
            self.values[gen] = _substituted(value, substitution)
        self.values.update(substitution)
        self.gens = [gen for gen in self.gens if gen not in substitution]
        self.a = [gen for gen in self.a if gen not in substitution]
        self.b = [gen for gen in self.b if gen not in substitution]
        remaining: list[PolyElement[T]] = []
        for equation in self.equations:
            replaced = _substituted(equation, substitution)
            if not replaced.terms:
                continue
            if not self.gens or replaced.is_ground:
                # a nonzero constant: no solution
                self.consistent = False
                return
            remaining.append(replaced)
        self.equations = remaining

    def value_of(self, p: PolyElement[T], point: _Point[T]) -> T:
        """The value of a polynomial in the unknowns at a point of the
        free ones."""
        return _value(_substituted(p, self.values), point)


def _chart_equations(field: _Field[T], N: int, top: PolyElement[T], top_cofactor: PolyElement[T],
                     fixed: Optional[dict[_Monomial, int]] = None) -> _Unknowns[T]:
    """The unknown coefficients of a Darboux polynomial of degree ``N``
    whose highest degree part is ``top`` (or, with ``fixed``, whose
    highest degree coefficients are partly fixed and partly unknown) and
    whose cofactor has highest degree part ``top_cofactor``."""
    K = field.K
    a_names: list[str] = []
    a_monomials: list[_Monomial] = []
    for monomial in _monomials(N):
        if sum(monomial) == N and (fixed is None or monomial in fixed):
            continue
        a_monomials.append(monomial)
        a_names.append('a_%d_%d' % monomial)
    b_monomials = [m for m in _monomials(field.degree - 1) if sum(m) != field.degree - 1]
    b_names = ['b_%d_%d' % m for m in b_monomials]
    ring = PolyRing(K, a_names + b_names or ['u'], lex)
    gens = ring.gens
    f_terms: dict[_Monomial, PolyElement[T]] = {}
    a: list[int] = []
    for monomial in _monomials(N):
        if sum(monomial) == N:
            if fixed is not None and monomial in fixed:
                f_terms[monomial] = ring.from_int(fixed[monomial])
                continue
            if fixed is None:
                f_terms[monomial] = ring.ground(top.coeff(monomial))
                continue
        index = len(a)
        a.append(index)
        f_terms[monomial] = gens[index]
    g_terms: dict[_Monomial, PolyElement[T]] = {}
    b: list[int] = []
    for monomial in _monomials(field.degree - 1):
        if sum(monomial) == field.degree - 1:
            g_terms[monomial] = ring.ground(top_cofactor.coeff(monomial))
            continue
        index = len(a_names) + len(b)
        b.append(index)
        g_terms[monomial] = gens[index]
    return _Unknowns(field, ring, f_terms, g_terms, a, b)


def _share_of_time() -> Optional[float]:
    """The time given to one optional step (a polynomial system of a
    leaf, a simplification): a share of the enclosing limit, or of the
    default."""
    left = remaining_time()
    limit = settings.timeout if left is None else left
    return None if limit is None else limit/4


class _ParametricElimination(Generic[T]):
    """Gaussian elimination of a linear system in the polynomial unknowns
    whose coefficients are polynomials in the cofactor unknowns (the
    parameters), with a case split at every pivot which may vanish: the
    leaves carry the conditions on the parameters under which the system
    is consistent (the constant rows left, the pivots assumed zero) and
    the pivots assumed nonzero. A pivot which is a nonzero constant needs
    no split and is preferred anywhere in the matrix; a vanishing pivot
    of degree one eliminates a parameter by substitution."""

    #: the number of leaves after which the elimination is given up
    LIMIT = 400

    def __init__(self, ring: PolyRing[T], n: int) -> None:
        self.ring: Final[PolyRing[T]] = ring
        self.n: Final[int] = n
        self.leaves: list[_Leaf[T]] = []
        self.exceeded = False

    def run(self, rows: list[_Row[T]], columns: list[int], equations: list[PolyElement[T]],
            nonzero: list[PolyElement[T]], fixed: list[tuple[int, PolyElement[T]]]) -> None:
        if self.exceeded:
            return
        live: list[_Row[T]] = []
        # after a substitution an equation may have become zero (dropped)
        # or a nonzero constant (no cofactor on this branch)
        equations = [e for e in equations if e.terms]
        if any(e.is_ground for e in equations):
            return
        for row in rows:
            if all(not entry.terms for entry in row[:self.n]):
                constant = row[self.n]
                if constant.terms:
                    if constant.is_ground:
                        return
                    equations.append(constant)
            else:
                live.append(row)
        if not live or not columns:
            self.leaves.append((equations, nonzero, fixed))
            if len(self.leaves) > self.LIMIT:
                self.exceeded = True
            return
        pivot = self._constant_pivot(live, columns)
        if pivot is not None:
            i, j = pivot
            self.run(self._eliminated(live, i, j, nonzero), [c for c in columns if c != j], equations, nonzero, fixed)
            return
        i, j = self._smallest_pivot(live, columns)
        p = live[i][j]
        # the pivot vanishes
        substitution = self._substitution(p)
        if substitution is not None:
            index, value = substitution
            values = {index: value}
            self.run([[_substituted(entry, values) for entry in row] for row in live], columns,
                     [_substituted(e, values) for e in equations], [_substituted(e, values) for e in nonzero],
                     fixed + [(index, value)])
        else:
            zeroed = [list(row) for row in live]
            zeroed[i][j] = self.ring.zero
            self.run(zeroed, columns, equations + [p], nonzero, fixed)
        # the pivot does not vanish
        self.run(self._eliminated(live, i, j, nonzero + [p]), [c for c in columns if c != j], equations,
                 nonzero + [p], fixed)

    def _constant_pivot(self, rows: list[_Row[T]], columns: list[int]) -> Optional[tuple[int, int]]:
        for i, row in enumerate(rows):
            for j in columns:
                entry = row[j]
                if entry.terms and entry.is_ground:
                    return i, j
        return None

    def _smallest_pivot(self, rows: list[_Row[T]], columns: list[int]) -> tuple[int, int]:
        best: Optional[tuple[int, int, int]] = None
        for i, row in enumerate(rows):
            for j in columns:
                entry = row[j]
                if entry.terms:
                    degree = entry.degree()
                    if best is None or degree < best[0]:
                        best = (degree, i, j)
        if best is None:
            raise ValueError("no pivot")
        return best[1], best[2]

    def _substitution(self, p: PolyElement[T]) -> Optional[tuple[int, PolyElement[T]]]:
        """``(k, value)`` with ``p = 0`` solved for the parameter ``b_k``
        when ``p`` has degree one."""
        if p.degree() != 1:
            return None
        K = self.ring.domain
        for index, gen in enumerate(self.ring.gens):
            coefficient = p.coeff(gen.LM)
            if not K.is_zero(coefficient):
                rest = p - gen.mul_ground(coefficient)
                return index, rest.mul_ground(K.neg(K.invert(coefficient)))
        return None

    def _eliminated(self, rows: list[_Row[T]], i: int, j: int, nonzero: list[PolyElement[T]]) -> list[_Row[T]]:
        """The rows with the unknown ``j`` eliminated by the pivot row
        ``i`` (fraction-free), the pivot row dropped, and the factors
        known to be nonzero divided out of each row."""
        pivot_row = rows[i]
        p = pivot_row[j]
        result: list[_Row[T]] = []
        for k, row in enumerate(rows):
            if k == i:
                continue
            c = row[j]
            if not c.terms:
                result.append(row)
                continue
            combined = [entry*p - pivot_entry*c for entry, pivot_entry in zip(row, pivot_row)]
            result.append(self._reduced(combined, nonzero))
        return result

    def _reduced(self, row: _Row[T], nonzero: list[PolyElement[T]]) -> _Row[T]:
        """The row divided by the factors common to its entries which
        are known to be nonzero (the earlier pivots, as in Bareiss'
        elimination). A common factor which may vanish stays: dividing
        it out loses the cofactors at which it does (the bug: a leaf
        ``b**3 = 0`` became ``1 = 0``, and the polynomial ``x`` of ``y' =
        (x y - 1)/x**2`` was lost)."""
        entries = [entry for entry in row if entry.terms]
        if not entries or any(entry.is_ground for entry in entries):
            return row
        for factor in nonzero:
            if factor.is_ground:
                continue
            while True:
                quotients: _Row[T] = []
                for entry in row:
                    if not entry.terms:
                        quotients.append(entry)
                        continue
                    q, r = entry.div([factor])
                    if r.terms:
                        break
                    quotients.append(q[0])
                else:
                    row = quotients
                    continue
                break
        return row


def _parameter_points(propagation: _Propagation[T], parameters: list[int], unknowns: list[int],
                      C: _Coefficients[T]) -> Optional[list[_Point[T]]]:
    """The values of the ``parameters`` (the free cofactor unknowns, or
    the free polynomial unknowns when they are fewer) on the solutions
    of the nonlinear remainder, which is linear in the ``unknowns`` with
    coefficients polynomial in the parameters: the parametric
    elimination gives the conditions on the parameters at its leaves,
    zero-dimensional systems (the cofactors of bounded degree are finitely
    many, and so are the polynomials with a given cofactor up to the
    affine families) solved one by one; ``None`` when the elimination or
    a system could not be finished."""
    K = C.K
    a, b = unknowns, parameters
    if not b:
        return [{}]
    ring = PolyRing(K, ['b%d' % k for k in range(len(b))], lex)
    column = {gen: k for k, gen in enumerate(a)}
    n = len(a)
    rows: list[_Row[T]] = []
    constraints: list[PolyElement[T]] = []
    for equation in propagation.equations:
        split: dict[int, dict[Monomial, T]] = {}
        for monomial, coefficient in equation.terms.items():
            a_part = [(gen, e) for gen, e in enumerate(monomial) if e and gen in column]
            if sum(e for _, e in a_part) > 1:
                return None
            target = column[a_part[0][0]] if a_part else n
            b_monomial = tuple(monomial[gen] for gen in b)
            split.setdefault(target, {})[b_monomial] = coefficient
        if not a:
            constraints.append(ring.from_dict(split.get(n, {})))
            continue
        row: _Row[T] = [ring.zero]*(n + 1)
        for target, terms in split.items():
            row[target] = ring.from_dict(terms)
        rows.append(row)
    elimination = _ParametricElimination(ring, n)
    elimination.run(rows, list(range(n)), constraints, [], [])
    if elimination.exceeded:
        return None
    points: list[_Point[T]] = []
    seen: set[tuple[str, ...]] = set()
    for equations, nonzero, fixed in elimination.leaves:
        fixed_indices = {index for index, _ in fixed}
        free = [index for index in range(len(b)) if index not in fixed_indices]
        if not equations and free:
            # a continuum of cofactors: not for a field of bounded degree
            continue
        for values in _leaf_points(equations, free, C):
            point: _Point[T] = dict(zip(free, values))
            # the fixed unknowns, the last fixed first (its value is in
            # the free ones), then the earlier ones
            for index, value in reversed(fixed):
                point[index] = _value(value, point)
            if any(K.is_zero(_value(e, point)) for e in nonzero):
                continue
            key = tuple(C.key(point[index]) for index in range(len(b)))
            if key in seen:
                continue
            seen.add(key)
            points.append({b[index]: point[index] for index in range(len(b))})
    return points


#: the number of combinations of factors up to which a leaf system is
#: solved case by case rather than at once
_CASES = 12


def _restricted(p: PolyElement[T], indices: list[int], ring: PolyRing[T]) -> PolyElement[T]:
    """``p``, a polynomial in the variables ``indices`` of its ring, in
    ``ring`` (whose variables are those)."""
    return PolyElement({tuple(m[i] for i in indices): c for m, c in p.terms.items()}, ring)


def _leaf_points(equations: list[PolyElement[T]], free: list[int], C: _Coefficients[T]) -> list[list[T]]:
    """The solutions in ``K`` of the equations of a leaf in its free
    unknowns ``free`` (none when a system is not zero-dimensional or not
    solved in time). The equations are products of minors: each is
    factored and the factors which divide a nonzero polynomial are
    dropped. With few combinations of factors, and factors of at most
    half the degree of the equations, the cases, one factor per
    equation, are solved one by one (trivariate minors of degree
    fifteen, whose Gröbner basis is slow); otherwise the products of
    the remaining factors are solved at once (a dicritical chart with
    five polynomial unknowns and equations of degree five, where the
    two cases cost seven times the whole)."""
    if not free:
        return [[]] if all(not e.terms for e in equations) else []
    K = C.K
    ring = PolyRing(K, ['b%d' % k for k in free], lex)
    restricted = [_restricted(e, free, ring) for e in equations]
    cases: list[list[PolyElement[T]]] = []
    for e in restricted:
        factors = C.factor(e)
        if not factors:
            return []
        cases.append(factors)
    cases.sort(key=len)
    combinations = 1
    for factors in cases:
        combinations *= len(factors)
    highest = max(f.degree() for factors in cases for f in factors)
    degree = max(e.degree() for e in restricted)
    k = len(free)
    if combinations <= _CASES and 2*highest <= degree:
        points = attempt(lambda: _points_of_cases(cases, [], k, C), _share_of_time(), native=True)
    else:
        products: list[PolyElement[T]] = []
        for factors in cases:
            product = ring.one
            for f in factors:
                product = product*f
            products.append(product)
        points = attempt(lambda: _points_in_domain(products, k, C), _share_of_time(), native=True)
    return [] if points is None else points


def _points_of_cases(cases: list[list[PolyElement[T]]], chosen: list[PolyElement[T]], k: int,
                     C: _Coefficients[T]) -> list[list[T]]:
    """The points of the cases, one factor of each equation of ``cases``
    added to ``chosen`` in turn, each complete case solved at once (a
    Gröbner basis of a subsystem, positive-dimensional as a rule, cost
    more than the whole)."""
    if not cases:
        points = _points_in_domain(chosen, k, C)
        return [] if points is None else points
    found: list[list[T]] = []
    seen: set[tuple[str, ...]] = set()
    for factor in cases[0]:
        for point in _points_of_cases(cases[1:], chosen + [factor], k, C):
            key = tuple(C.key(v) for v in point)
            if key not in seen:
                seen.add(key)
                found.append(point)
    return found


def _lex_basis(polys: list[PolyElement[T]], k: int) -> Optional[list[PolyElement[T]]]:
    """The reduced lex Gröbner basis (in the ring of ``polys``) of a
    zero-dimensional system in its first ``k`` variables, from the
    grevlex basis by FGLM; ``[1]`` for an inconsistent one and ``None``
    when it is not zero-dimensional (its grevlex basis is far cheaper
    than a lex one)."""
    from sympy_extras.polys.fglm import fglm, is_zero_dimensional
    from sympy_extras.polys.ideals import groebner
    ring = polys[0].ring
    graded = PolyRing(ring.domain, ring.names[:k], grevlex)
    indices = list(range(k))
    basis = groebner([_restricted(p, indices, graded) for p in polys], graded)
    if any(g.is_ground and g.terms for g in basis):
        return [ring.one]
    if not is_zero_dimensional(basis, graded):
        return None
    pad = tuple(0 for _ in range(ring.ngens - k))
    return [PolyElement({m + pad: c for m, c in g.terms.items()}, ring) for g in fglm(basis, graded, lex)]


def _univariate_gcd(f: PolyElement[T], g: PolyElement[T]) -> PolyElement[T]:
    """The monic gcd of two polynomials in one variable of their ring."""
    while g.terms:
        f, g = g, f.rem([g])
    return f.monic()


def _points_in_domain(polys: list[PolyElement[T]], k: int, C: _Coefficients[T]) -> Optional[list[list[T]]]:
    """The points with coordinates in ``K`` of a zero-dimensional system
    in the first ``k`` variables of its ring (``None`` when it is not
    zero-dimensional): the last coordinate is a root of a polynomial of
    the system in the last unknown alone (the gcd of those; from the lex
    Gröbner basis when the system has none), and the rest is solved
    recursively at each root, an inconsistent remainder giving no
    point."""
    polys = [p for p in polys if p.terms]
    if k == 0:
        return [[]] if not polys else []
    if not polys:
        return None
    if any(p.is_ground for p in polys):
        return []
    last = k - 1

    def univariate(p: PolyElement[T]) -> bool:
        return all(not any(m[:last]) for m in p.terms)

    alone = [p for p in polys if univariate(p)]
    if not alone:
        basis = _lex_basis(polys, k)
        if basis is None:
            return None
        if len(basis) == 1 and basis[0].is_ground:
            return []
        polys = basis
        alone = [p for p in polys if univariate(p)]
        if not alone:
            return None
    polynomial = alone[0]
    for p in alone[1:]:
        polynomial = _univariate_gcd(polynomial, p)
    others = [p for p in polys if not univariate(p)]
    points: list[list[T]] = []
    for root in C.roots(polynomial, last):
        rest = [_evaluated(p, {last: root}) for p in others]
        below = _points_in_domain(rest, last, C)
        if below is None:
            return None
        for point in below:
            points.append(point + [root])
    return points


def _members(particular: list[T], basis: list[list[T]], K: Domain[T]) -> list[list[T]]:
    """A few members of an affine space: the sparsest member reachable
    from the particular solution along the basis (the member of a pencil
    ``f + u h`` with the fewest terms, ``x**2 - y**2 + 1`` rather than
    ``x**2 + 2 x - y**2 + 1`` when ``x`` is a Darboux polynomial with the
    same cofactor), each basis vector added to it, and a generic
    combination."""
    sparse = _sparse_member(particular, basis, K)
    members = [sparse]
    for vector in basis:
        members.append([K.add(p, v) for p, v in zip(sparse, vector)])
    if len(basis) > 1:
        generic = list(sparse)
        for k, vector in enumerate(basis):
            scale = K.from_int(k + 1)
            generic = [K.add(p, K.mul(scale, v)) for p, v in zip(generic, vector)]
        members.append(generic)
    return members


def _sparse_member(particular: list[T], basis: list[list[T]], K: Domain[T]) -> list[T]:
    """The member with the fewest nonzero entries among those reached by
    cancelling one entry at a time along each basis vector (a greedy
    pass over the basis, twice)."""
    member = list(particular)
    for _ in range(2):
        for vector in basis:
            best = member
            for i, v in enumerate(vector):
                if K.is_zero(v) or K.is_zero(member[i]):
                    continue
                u = K.neg(K.exquo(member[i], v))
                candidate = [K.add(p, K.mul(u, w)) for p, w in zip(member, vector)]
                if _nonzero_entries(candidate, K) < _nonzero_entries(best, K):
                    best = candidate
            member = best
    return member


def _nonzero_entries(vector: list[T], K: Domain[T]) -> int:
    return sum(1 for v in vector if not K.is_zero(v))


def _chart_solutions(unknowns: _Unknowns[T]) -> list[tuple[PolyElement[T], PolyElement[T]]]:
    """The Darboux polynomials of one chart with their cofactors, a few
    members of each affine family. The bilinear remainder is linear in
    the polynomial unknowns for fixed cofactor unknowns and the other
    way round: the elimination takes as parameters whichever free block
    is the smaller (the cofactor of a field of degree five has ten
    lower coefficients, a polynomial of degree one has one)."""
    field = unknowns.field
    K = field.K
    equations = unknowns.equations()
    propagation = _Propagation(equations, unknowns.a, unknowns.b, K)
    if not propagation.consistent:
        return []
    parameters, free = propagation.b, propagation.a
    if propagation.equations:
        swapped = _Propagation(equations, unknowns.a, unknowns.b, K, b_first=True)
        if swapped.consistent and len(swapped.a) < len(propagation.b):
            propagation, parameters, free = swapped, swapped.a, swapped.b
        points = _parameter_points(propagation, parameters, free, field.C)
        if points is None:
            return []
    else:
        points = [{}]
    found: list[tuple[PolyElement[T], PolyElement[T]]] = []
    for point in points:
        # at a point of the parameters the remaining equations are linear
        # in the other block
        remaining = [_evaluated(equation, point) for equation in propagation.equations]
        remaining = [equation for equation in remaining if equation.terms]
        if any(equation.is_ground for equation in remaining):
            continue
        if any(equation.degree() > 1 for equation in remaining):
            continue
        rows, rhs = _linear_rows(remaining, free, K)
        solved = _solve_affine(rows, rhs, len(free), K)
        if solved is None:
            continue
        particular, basis, _ = solved
        for member in _members(particular, basis, K):
            values: _Point[T] = dict(point)
            values.update(zip(free, member))
            for gen in parameters:
                values.setdefault(gen, K.zero)
            f = field.ring.from_dict({m: propagation.value_of(c, values) for m, c in unknowns.f_terms.items()})
            g = field.ring.from_dict({m: propagation.value_of(c, values) for m, c in unknowns.g_terms.items()})
            if f.degree() <= 0:
                continue
            if field.apply(f) != g*f:
                continue
            found.append((f, g))
    return found


def _top_parts(field: _Field[T]) -> tuple[PolyElement[T], PolyElement[T], PolyElement[T]]:
    """``P_d``, ``Q_d`` and ``H = x P_d - y Q_d``."""
    d = field.degree
    P_d = field.homogeneous_part(field.P, d)
    Q_d = field.homogeneous_part(field.Q, d)
    H = field.monomial((1, 0))*P_d - field.monomial((0, 1))*Q_d
    return P_d, Q_d, H


def _top_candidates(field: _Field[T], N: int, H: PolyElement[T]) -> list[PolyElement[T]]:
    """The possible highest degree parts of a Darboux polynomial of
    degree ``N``: the monic products of irreducible factors of ``H`` of
    total degree ``N``."""
    irreducible = [(h, h.degree()) for h in field.C.factor(H)]
    candidates: list[PolyElement[T]] = []

    def extend(index: int, degree_left: int, product: PolyElement[T]) -> None:
        if degree_left == 0:
            # (the elements of a number field are lists: no hashing)
            if all(product != c for c in candidates):
                candidates.append(product)
            return
        for k in range(index, len(irreducible)):
            h, degree = irreducible[k]
            if degree <= degree_left:
                extend(k, degree_left - degree, product*h)

    extend(0, N, field.one())
    return candidates


def _darboux_of_degree(field: _Field[T], N: int) -> list[tuple[PolyElement[T], PolyElement[T]]]:
    """Darboux polynomials of total degree ``N`` with their cofactors (a
    few members of each family, not factored)."""
    P_d, Q_d, H = _top_parts(field)
    found: list[tuple[PolyElement[T], PolyElement[T]]] = []
    if H.terms:
        for top in _top_candidates(field, N, H):
            D_top = Q_d*top.diff(0) + P_d*top.diff(1)
            quotients, remainder = D_top.div([top])
            if remainder.terms:
                continue
            found.extend(_chart_solutions(_chart_equations(field, N, top, quotients[0])))
        return found
    # dicritical: D_d = k (x d/dx + y d/dy), the cofactor's top part is N k
    if Q_d.terms:
        k = Q_d.div([field.monomial((1, 0))])[0][0]
    else:
        k = P_d.div([field.monomial((0, 1))])[0][0]
    top_cofactor = k.mul_ground(field.K.from_int(N))
    top_monomials = [(N - j, j) for j in range(N + 1)]
    for leading in range(N + 1):
        fixed: dict[_Monomial, int] = {}
        for j in range(leading):
            fixed[top_monomials[j]] = 0
        fixed[top_monomials[leading]] = 1
        found.extend(_chart_solutions(_chart_equations(field, N, field.zero(), top_cofactor, fixed)))
    return found


def _darboux(field: _Field[T], degree: int) -> list[tuple[PolyElement[T], PolyElement[T]]]:
    """The irreducible Darboux polynomials of degree at most ``degree``
    with their cofactors, monic, without repetition."""
    found: list[tuple[PolyElement[T], PolyElement[T]]] = []
    for N in range(1, degree + 1):
        _add_darboux(field, N, found)
    return found


def _add_darboux(field: _Field[T], N: int, found: list[tuple[PolyElement[T], PolyElement[T]]]) -> None:
    """The irreducible Darboux polynomials of degree ``N`` added to
    ``found`` (monic, each once; the elements of a number field are
    lists, so the polynomials are compared rather than hashed)."""
    for f, _ in _darboux_of_degree(field, N):
        for h in field.C.factor(f):
            if any(h == g for g, _ in found):
                continue
            cofactor = field.cofactor_of(h)
            if cofactor is not None:
                found.append((h, cofactor))


# ---------------------------------------------------------------------------
# exponential factors

def _denominators(field: _Field[T], polynomials: Sequence[tuple[PolyElement[T], PolyElement[T]]],
                  degree: int) -> list[tuple[PolyElement[T], PolyElement[T]]]:
    """``1`` and the products of Darboux polynomials of total degree at
    most ``degree``, with their cofactors."""
    result: list[tuple[PolyElement[T], PolyElement[T]]] = [(field.one(), field.zero())]
    degrees = [f.degree() for f, _ in polynomials]
    for size in range(1, degree + 1):
        for choice in combinations_with_replacement(range(len(polynomials)), size):
            if sum(degrees[i] for i in choice) > degree:
                continue
            B, g = field.one(), field.zero()
            for i in choice:
                B, g = B*polynomials[i][0], g + polynomials[i][1]
            result.append((B, g))
    return result


def _exponential_with_denominator(field: _Field[T], B: PolyElement[T], g_B: PolyElement[T],
                                  bound: int) -> list[tuple[PolyElement[T], PolyElement[T]]]:
    """The exponential factors ``exp(A/B)`` with ``deg A <= bound`` and
    their cofactors ``L``: the nullspace of ``(A, L) -> D A - g_B A - L B``,
    without those with ``L = 0`` (rational first integrals or constants)."""
    K = field.K
    a_monomials = _monomials(bound)
    l_monomials = _monomials(field.degree - 1)
    # the columns of the linear map: the images of the monomials of A
    # and of L, as coefficient vectors on the monomials of the image
    columns: list[dict[Monomial, T]] = []
    for m in a_monomials:
        monomial = field.monomial(m)
        columns.append((field.apply(monomial) - g_B*monomial).terms)
    for m in l_monomials:
        columns.append((-(B*field.monomial(m))).terms)
    image: list[Monomial] = sorted({monomial for column in columns for monomial in column})
    rows: list[list[T]] = [[column.get(monomial, K.zero) for column in columns] for monomial in image]
    solved = _solve_affine(rows, [K.zero]*len(rows), len(columns), K)
    if solved is None:
        return []
    _, basis, _ = solved
    result: list[tuple[PolyElement[T], PolyElement[T]]] = []
    for vector in basis:
        A = field.ring.from_dict({m: vector[i] for i, m in enumerate(a_monomials)})
        L = field.ring.from_dict({m: vector[len(a_monomials) + i] for i, m in enumerate(l_monomials)})
        if not L.terms or not A.terms:
            continue
        result.append((A, L))
    return result


def _exponential(field: _Field[T], polynomials: Sequence[tuple[PolyElement[T], PolyElement[T]]],
                 degree: int) -> list[tuple[PolyElement[T], PolyElement[T], PolyElement[T]]]:
    """The exponential factors ``(A, B, L)`` with ``B`` a product of the
    Darboux polynomials of degree at most ``degree``."""
    found: list[tuple[PolyElement[T], PolyElement[T], PolyElement[T]]] = []
    for B, g_B in _denominators(field, polynomials, degree):
        bound = max(degree, field.degree) + B.degree()
        for A, L in _exponential_with_denominator(field, B, g_B, bound):
            # exp(B A0/B) is exp(A0) again
            if any(A*B0 == A0*B for A0, B0, _ in found):
                continue
            found.append((A, B, L))
    return found


# ---------------------------------------------------------------------------
# the exponents

def _ops(e: Expr) -> int:
    count = count_ops(e)
    return int(count) if isinstance(count, Integer) else 0


def _simplicity(e: Expr) -> tuple[int, int, int, str]:
    """The sort key of the simplest candidate: the exponentials first,
    then the powers with a non-integer exponent, then the operations,
    then the printed form (for a deterministic choice)."""
    fractional = sum(1 for p in e.atoms(Pow) if isinstance(p, Pow) and not isinstance(p.exp, Integer))
    return len(e.atoms(Exp)), fractional, _ops(e), str(e)


class _Darboux(Generic[T]):
    """The Darboux polynomials of degree at most ``degree`` (given) and
    the exponential factors of a field with the linear algebra of their
    exponents."""

    def __init__(self, field: _Field[T], polynomials: list[tuple[PolyElement[T], PolyElement[T]]],
                 degree: int) -> None:
        self.field: Final[_Field[T]] = field
        self.polynomials: Final[list[tuple[PolyElement[T], PolyElement[T]]]] = polynomials
        self.exponentials: Final[list[tuple[PolyElement[T], PolyElement[T], PolyElement[T]]]] = \
            _exponential(field, polynomials, degree)
        self.cofactors: Final[list[PolyElement[T]]] = [g for _, g in polynomials] + [L for _, _, L in self.exponentials]
        self.monomials: Final[list[_Monomial]] = _monomials(field.degree - 1)
        self.rows: Final[list[list[T]]] = [[g.coeff(m) for g in self.cofactors] for m in self.monomials]

    def _integers(self, vector: list[T]) -> Optional[list[Expr]]:
        """The vector scaled to integers when its entries are rational."""
        rationals: list[Fraction] = []
        for v in vector:
            q = self.field.C.rational(v)
            if q is None:
                return None
            rationals.append(q)
        scale = 1
        for q in rationals:
            scale = lcm(scale, q.denominator)
        return [Integer(q.numerator*(scale//q.denominator)) for q in rationals]

    def first_integral(self) -> Optional[Expr]:
        """``sum c_i log f_i + sum e_j A_j/B_j`` with ``sum c_i g_i + sum
        e_j L_j = 0``, written ``prod f_i**c_i`` (a rational first
        integral) when no exponential factor takes part, the exponents
        integers when they are rational, the simplest combination among
        a basis; ``None`` when the cofactors are independent."""
        if not self.cofactors:
            return None
        K = self.field.K
        solved = _solve_affine(self.rows, [K.zero]*len(self.rows), len(self.cofactors), K)
        if solved is None:
            return None
        _, basis, _ = solved
        if not basis:
            return None
        candidates: list[Expr] = []
        for vector in basis:
            scaled = self._integers(vector)
            exponents = scaled if scaled is not None else [self.field.C.to_expr(v) for v in vector]
            # the first exponent positive (a sign fixed, for a stable form)
            first = next((e for e in exponents if e != _ZERO), _ONE)
            if first.is_negative is True:
                exponents = [-e for e in exponents]
            candidate = self._product(exponents, logarithmic=True)
            if not _is_constant_integral(_tidy(candidate), self.field.x, self.field.y):
                candidates.append(candidate)
        if not candidates:
            return None
        return min(candidates, key=_simplicity)

    def integrating_factor(self) -> Optional[Expr]:
        """``prod f_i**c_i exp(sum e_j A_j/B_j)`` with ``sum c_i g_i + sum
        e_j L_j = -div D``, the simplest among the particular solution and
        its sums with the basis of the homogeneous solutions."""
        K = self.field.K
        divergence = self.field.divergence()
        rhs = [K.neg(divergence.coeff(m)) for m in self.monomials]
        if not self.cofactors:
            if all(K.is_zero(v) for v in rhs):
                return _ONE
            return None
        solved = _solve_affine(self.rows, rhs, len(self.cofactors), K)
        if solved is None:
            return None
        particular, basis, _ = solved
        candidates = [self._product([self.field.C.to_expr(v) for v in member], logarithmic=False)
                      for member in [particular] + _members(particular, basis, K)]
        return min(candidates, key=_simplicity)

    def _product(self, exponents: Sequence[Expr], logarithmic: bool) -> Expr:
        """``prod f_i**c_i exp(sum e_j A_j/B_j)``, or with ``logarithmic``
        its logarithm when an exponential factor takes part."""
        field = self.field
        n = len(self.polynomials)
        argument: Expr = _ZERO
        for (A, B, _), e in zip(self.exponentials, exponents[n:]):
            if e != _ZERO:
                argument = argument + e*field.expr(A)/field.expr(B)
        if logarithmic and argument != _ZERO:
            terms: list[Expr] = [argument]
            for (f, _), c in zip(self.polynomials, exponents[:n]):
                if c != _ZERO:
                    terms.append(c*log(field.expr(f)))
            return add(*terms)
        product: Expr = _ONE
        for (f, _), c in zip(self.polynomials, exponents[:n]):
            if c != _ZERO:
                product = product*field.expr(f)**c
        if argument != _ZERO:
            product = product*exp(argument)
        return product


# ---------------------------------------------------------------------------
# the quadratures

def _integrate(e: Expr, v: Symbol, x: Symbol, y: Symbol) -> Optional[Expr]:
    """``Integral(e, v)`` by SymPy (through
    :func:`~sympy_extras.solvers.kovacic_sympy.integrate_of_core`) within
    half of the time left, with ``x``, ``y`` and the parameters real
    first (the logarithms of a rational integrand come out real:
    ``atan((x + y)/a)`` rather than ``I*log(x + y - I*a)/2 - I*log(x + y
    + I*a)/2``), then with complex symbols when that gives no
    antiderivative: a ``Piecewise`` on the signs of the parameters
    without a generic branch, or a wrong one (SymPy's ``integrate(a/(a*y**2
    - b), y)`` is ``0`` for real ``a``, ``b``, ``y``; every antiderivative
    is checked by differentiation); ``None`` when neither is an
    antiderivative in closed form."""
    from .kovacic_sympy import integrate_of_core
    real: dict[Symbol, Symbol] = {s: Dummy(s.name, real=True) for s in e.free_symbols}
    real.setdefault(x, Dummy(x.name, real=True))
    real.setdefault(y, Dummy(y.name, real=True))
    forward: dict[Basic, Basic] = {s: d for s, d in real.items()}
    back: dict[Basic, Basic] = {d: s for s, d in real.items()}
    variable = real[v]
    integrand = e.xreplace(forward)
    limit = remaining_time()
    budget = settings.timeout if limit is None else limit/2
    result = attempt(lambda: integrate_of_core(integrand, variable, 'none', False), budget)
    if result is not None:
        generic = _generic_branch(result)
        if generic is not None:
            candidate = generic.xreplace(back)
            if _is_antiderivative(candidate, e, v):
                return candidate
    limit = remaining_time()
    budget = settings.timeout if limit is None else limit/2
    result = attempt(lambda: integrate_of_core(e, v, 'none', False), budget)
    if result is None:
        return None
    generic = _generic_branch(result)
    if generic is None or not _is_antiderivative(generic, e, v):
        return None
    return generic


def _is_antiderivative(F: Expr, e: Expr, v: Symbol) -> bool:
    """Whether ``F`` is an antiderivative of ``e`` in closed form: no
    derivative left, and ``F' - e`` zero (the integrals were refused by
    :func:`~sympy_extras.solvers.kovacic_sympy.integrate_of_core`)."""
    if F.has(Derivative):
        return False
    return _is_zero(F.diff(v) - e)


def _generic_branch(e: Expr) -> Optional[Expr]:
    """``e`` with every ``Piecewise`` replaced by its branch holding
    generically (a condition which is an inequation, or ``True``); the
    result is checked by differentiation by the caller."""
    for piece in e.atoms(Piecewise):
        if not isinstance(piece, Piecewise):
            continue
        chosen: Optional[Expr] = None
        for pair in piece.args:
            if not isinstance(pair, ExprCondPair):
                continue
            if pair.cond == true or isinstance(pair.cond, Ne):
                chosen = pair.expr
                break
        if chosen is None:
            return None
        replacement: dict[Basic, Basic] = {piece: chosen}
        e = e.xreplace(replacement)
    return e


def _is_zero(e: Expr) -> bool:
    """Whether ``e`` is zero: ``cancel``, then ``simplify`` within a
    share of the time."""
    from sympy_extras.polys.polytools import cancel
    from sympy_extras.simplify.simplify import simplify
    reduced = attempt(lambda: cancel(e), _share_of_time(), native=True)
    if reduced is None:
        return False
    if reduced == _ZERO:
        return True
    simplified = attempt(lambda: simplify(reduced), _share_of_time(), native=True)
    return simplified is not None and simplified == _ZERO


def _first_integral_from_factor(field: _Field[T], R: Expr) -> Optional[Expr]:
    """``I`` with ``I_x = -R P`` and ``I_y = R Q`` by two quadratures,
    checked by ``D I = 0``."""
    from sympy_extras.polys.polytools import cancel
    from sympy_extras.simplify.simplify import simplify
    x, y = field.x, field.y
    P, Q = field.expr(field.P), field.expr(field.Q)
    U = _integrate(-R*P, x, x, y)
    if U is None:
        return None
    V = R*Q - U.diff(y)
    cancelled = attempt(lambda: cancel(V), _share_of_time(), native=True)
    if cancelled is None:
        return None
    V = cancelled
    if V.has(x):
        simplified = attempt(lambda: simplify(V), _share_of_time(), native=True)
        if simplified is None or simplified.has(x):
            return None
        V = simplified
    W = _integrate(V, y, x, y) if V != _ZERO else _ZERO
    if W is None:
        return None
    integral = U + W
    if not _is_zero(Q*integral.diff(x) + P*integral.diff(y)):
        return None
    return integral


def _is_constant_integral(integral: Expr, x: Symbol, y: Symbol) -> bool:
    return not (integral.has(x) or integral.has(y))


def _tidy(integral: Expr) -> Expr:
    """A first integral in a readable form: cancelled as a rational
    function of its atoms when that is shorter (``-exp(x**2)/(2 y**2) -
    sqrt(pi) erfi(x)/2`` for the quadrature ``-(sqrt(pi) y**3 erfi(x)/2 +
    y exp(x**2)/2)/y**3``)."""
    from sympy_extras.polys.polytools import cancel
    cancelled = attempt(lambda: cancel(integral), _share_of_time(), native=True)
    if cancelled is None:
        return integral
    return min([integral, cancelled], key=_ops)


# ---------------------------------------------------------------------------
# the field of the coefficients and the procedure

class _Context(ABC):
    """A vector field over a coefficient field whose element type is
    hidden: the public functions work through it."""

    @abstractmethod
    def polynomials(self, degree: int) -> list[DarbouxPolynomial]: ...

    @abstractmethod
    def exponential_factors(self, degree: int) -> list[ExponentialFactor]: ...

    @abstractmethod
    def integrating_factor(self, degree: int) -> Optional[Expr]: ...

    @abstractmethod
    def first_integral_over(self, degree: int) -> Optional[Expr]:
        """The first integral with the Darboux polynomials of the lowest
        degree which gives one, over this field."""

    @abstractmethod
    def quadratic_extension(self) -> Optional[Expr]:
        """The square root of the discriminant of the first irreducible
        quadratic factor of ``x P_d - y Q_d`` over the rationals, or
        ``None`` (always over the other fields)."""

    @abstractmethod
    def extended(self, extension: Expr) -> _Context:
        """The same field over ``QQ(extension)``."""

    def first_integral(self, degree: int) -> Optional[Expr]:
        """The first integral with the Darboux polynomials of the lowest
        degree which gives one (the Prelle–Singer loop: the bound is
        raised one by one up to ``degree``); over the rationals, when
        nothing is found and ``x P_d - y Q_d`` has irreducible quadratic
        factors, the search is repeated over the field of their roots
        (the conjugate lines with their own exponents)."""
        integral = self.first_integral_over(degree)
        if integral is not None:
            return integral
        extension = self.quadratic_extension()
        if extension is None:
            return None
        return self.extended(extension).first_integral_over(degree)


class _Engine(_Context, Generic[T]):
    """The procedure on a field over ``K``."""

    def __init__(self, field: _Field[T], P: Expr, Q: Expr) -> None:
        self.field: Final[_Field[T]] = field
        self.P: Final[Expr] = P
        self.Q: Final[Expr] = Q

    def polynomials(self, degree: int) -> list[DarbouxPolynomial]:
        field = self.field
        return [DarbouxPolynomial(field.expr(f), field.expr(g)) for f, g in _darboux(field, degree)]

    def exponential_factors(self, degree: int) -> list[ExponentialFactor]:
        field = self.field
        return [ExponentialFactor(field.expr(A), field.expr(B), field.expr(L))
                for A, B, L in _exponential(field, _darboux(field, degree), degree)]

    def integrating_factor(self, degree: int) -> Optional[Expr]:
        return _Darboux(self.field, _darboux(self.field, degree), degree).integrating_factor()

    def first_integral_over(self, degree: int) -> Optional[Expr]:
        field = self.field
        polynomials: list[tuple[PolyElement[T], PolyElement[T]]] = []
        for N in range(1, degree + 1):
            _add_darboux(field, N, polynomials)
            darboux = _Darboux(field, list(polynomials), N)
            direct = darboux.first_integral()
            if direct is not None:
                return _tidy(direct)
            R = darboux.integrating_factor()
            if R is None:
                continue
            integral = _first_integral_from_factor(field, R)
            if integral is not None and not _is_constant_integral(integral, field.x, field.y):
                return _tidy(integral)
        return None

    def quadratic_extension(self) -> Optional[Expr]:
        field = self.field
        if not field.C.is_rationals:
            return None
        _, _, H = _top_parts(field)
        if not H.terms:
            return None
        K = field.K
        for h in field.C.factor(H):
            if h.degree() != 2:
                continue
            # h(x, 1), or h(1, y) when it drops a degree
            in_x: dict[int, T] = {}
            for (i, _), c in h.terms.items():
                in_x[i] = K.add(in_x.get(i, K.zero), c)
            if K.is_zero(in_x.get(2, K.zero)):
                in_x = {}
                for (_, j), c in h.terms.items():
                    in_x[j] = K.add(in_x.get(j, K.zero), c)
            a, b, c = in_x.get(2, K.zero), in_x.get(1, K.zero), in_x.get(0, K.zero)
            discriminant = field.C.rational(K.sub(K.mul(b, b), K.mul(K.from_int(4), K.mul(a, c))))
            if discriminant is None or discriminant >= 0:
                # reducible, or a double line: nothing new
                continue
            return sqrt(QQ.to_expr(discriminant))
        return None

    def extended(self, extension: Expr) -> _Context:
        return _field_of(self.P, self.Q, self.field.x, self.field.y, extension)


def _number_field(alpha: Expr) -> AlgebraicField:
    from sympy_extras.polys.numberfields.minpoly import minimal_polynomial_coeffs
    coefficients = minimal_polynomial_coeffs(alpha)
    return AlgebraicField([Fraction(c, coefficients[0]) for c in coefficients], alpha)


def _composed(f: list[Fraction], t: list[Fraction], K: AlgebraicField) -> list[Fraction]:
    """``f(t)`` in ``K``: an element of the field of ``f``'s variable
    (its coefficients highest degree first) at the image ``t`` of that
    variable."""
    value = K.zero
    for c in f:
        value = K.add(K.mul(value, t), K.from_rational(c))
    return value


def _primitive_field(generators: Sequence[Expr]) -> _NumberField:
    """The number field of several algebraic numbers, by a primitive
    element ``theta = g_1 + c_2 g_2 + ...`` built one generator at a time
    with small integers ``c``: the new generator ``g`` is the common root
    of its minimal polynomial and of ``m(theta - c t)``, ``m`` the minimal
    polynomial of the previous primitive element, which their gcd over
    ``QQ(theta)`` gives when it is linear (the theorem of the primitive
    element, made effective)."""
    from sympy_extras.polys.euclidtools import dup_gcd
    theta = generators[0]
    K = _number_field(theta)
    images: dict[Expr, list[Fraction]] = {theta: K.alpha}
    for g in generators[1:]:
        G = _number_field(g)
        if G.degree == 1:
            images[g] = K.from_rational(-G.minpoly[1])
            continue
        for c in range(1, 4*K.degree*G.degree + 2):
            candidate = theta + c*g
            L = _number_field(candidate)
            # m(theta_L - c t) as a polynomial in t over L, by Horner
            shifted: list[list[Fraction]] = []
            for coefficient in K.minpoly:
                product: list[list[Fraction]] = [L.zero]*(len(shifted) + 1) if shifted else []
                for i, u in enumerate(shifted):
                    product[i] = L.add(product[i], L.mul(u, L.from_int(-c)))
                    product[i + 1] = L.add(product[i + 1], L.mul(u, L.alpha))
                if not product:
                    product = [L.from_rational(coefficient)] if coefficient else []
                else:
                    product[-1] = L.add(product[-1], L.from_rational(coefficient))
                shifted = product
            common = dup_gcd(shifted, [L.from_rational(v) for v in G.minpoly], L)
            if len(common) != 2:
                continue
            image = L.neg(L.quo(common[1], common[0]))
            previous = L.sub(L.alpha, L.mul(L.from_int(c), image))
            images = {h: _composed(v, previous, L) for h, v in images.items()}
            images[g] = image
            theta, K = candidate, L
            break
        else:
            raise ArithmeticError("no primitive element was found")
    return _NumberField(K, images)


def _coefficient_atoms(e: Expr, x: Symbol, y: Symbol, parameters: list[Expr], radicals: dict[Rational, int],
                       imaginary: list[bool]) -> None:
    """The parameters (the symbols and the transcendental constants, such
    as ``pi``, as SymPy's ``Poly`` takes them), the radicals ``b**(p/q)``
    of rational bases (the largest ``q`` of each base) and ``I`` among the
    coefficients of ``e`` (the parts with ``x`` or ``y`` are left to the
    conversion, which refuses what is not polynomial)."""
    if isinstance(e, Rational) or e == x or e == y:
        return
    if e == I:
        imaginary[0] = True
        return
    if isinstance(e, Symbol):
        if e not in parameters:
            parameters.append(e)
        return
    if isinstance(e, Pow) and isinstance(e.base, Rational) and isinstance(e.exp, Rational):
        radicals[e.base] = lcm(radicals.get(e.base, 1), e.exp.q)
        return
    if isinstance(e, Pow) and isinstance(e.exp, Integer):
        _coefficient_atoms(e.base, x, y, parameters, radicals, imaginary)
        return
    if isinstance(e, (Add, Mul)):
        for a in e.args:
            _coefficient_atoms(a, x, y, parameters, radicals, imaginary)
        return
    if not (e.has(x) or e.has(y)) and e not in parameters:
        parameters.append(e)


def _engine(C: _Coefficients[T], P: Expr, Q: Expr, x: Symbol, y: Symbol) -> _Engine[T]:
    ring = PolyRing(C.K, ['_darboux_x', '_darboux_y'], lex)
    p = _xy_poly(P, x, y, C, ring)
    q = _xy_poly(Q, x, y, C, ring)
    if not q.terms:
        raise ValueError("Q must not be zero")
    return _Engine(_Field(C, p, q, x, y), P, Q)


def _field_of(P: ExprLike, Q: ExprLike, x: Symbol, y: Symbol, extension: Optional[Expr] = None) -> _Context:
    """The field ``Q d/dx + P d/dy`` over the field generated by the
    coefficients of ``P`` and ``Q`` (and ``extension``): ``QQ``, a number
    field (of a primitive element when there are several algebraic
    numbers), or ``QQ(a, b, ...)`` for the parameters (the symbols and the
    transcendental constants); ``ValueError`` when ``P`` and ``Q`` are not
    polynomials in ``x`` and ``y`` over such a field (the parameters
    together with algebraic numbers are not supported)."""
    P_, Q_ = to_expr(P), to_expr(Q)
    error = "polynomials in %s and %s over a field are expected" % (x, y)
    parameters: list[Expr] = []
    radicals: dict[Rational, int] = {}
    imaginary = [False]
    for e in (P_, Q_):
        _coefficient_atoms(expand(e), x, y, parameters, radicals, imaginary)
    generators: list[Expr] = [] if extension is None else [extension]
    if imaginary[0]:
        generators.append(I)
    for base in sorted(radicals, key=lambda r: (r.p, r.q)):
        if radicals[base] > 1:
            generators.append(base**Rational(1, radicals[base]))
    try:
        if generators:
            if parameters:
                raise ValueError("coefficients with parameters and algebraic numbers are not supported")
            C = _NumberField(_number_field(generators[0])) if len(generators) == 1 else _primitive_field(generators)
            return _engine(C, P_, Q_, x, y)
        if parameters:
            return _engine(_Parameters(sorted(parameters, key=lambda s: s.sort_key())), P_, Q_, x, y)
        return _engine(_Rationals(), P_, Q_, x, y)
    except (ValueError, TypeError, ArithmeticError) as e:
        if str(e) == "Q must not be zero":
            raise
        raise ValueError(error) from e


# ---------------------------------------------------------------------------
# the public functions

def darboux_polynomials(P: ExprLike, Q: ExprLike, x: Symbol, y: Symbol, degree: int = 4,
                        extension: Optional[Expr] = None) -> list[DarbouxPolynomial]:
    """The irreducible Darboux polynomials of total degree at most
    ``degree`` of the field ``Q d/dx + P d/dy`` (the equation ``y' = P/Q``)
    with their cofactors, over the field generated by the coefficients
    (and ``extension``, an algebraic number such as ``I`` or ``sqrt(2)``,
    when given: ``x + I*y`` and ``x - I*y`` rather than their product).

    Examples
    ========

    >>> from sympy_extras.core import Symbol, I
    >>> from sympy_extras.solvers.darboux import darboux_polynomials
    >>> x, y = Symbol('x'), Symbol('y')
    >>> for f in darboux_polynomials(y*(x - 1), x*(1 - y), x, y):      # Lotka–Volterra
    ...     print(f)
    DarbouxPolynomial(y, x - 1)
    DarbouxPolynomial(x, 1 - y)
    >>> darboux_polynomials(x + y, x - y, x, y, degree=2)
    [DarbouxPolynomial(x**2 + y**2, 2)]
    >>> darboux_polynomials(x + y, x - y, x, y, degree=1, extension=I)
    [DarbouxPolynomial(x - I*y, 1 - I), DarbouxPolynomial(x + I*y, 1 + I)]
    """
    return _field_of(P, Q, x, y, extension).polynomials(degree)


def exponential_factors(P: ExprLike, Q: ExprLike, x: Symbol, y: Symbol, degree: int = 4,
                        extension: Optional[Expr] = None) -> list[ExponentialFactor]:
    """The exponential factors ``exp(A/B)`` of the field ``Q d/dx + P d/dy``
    whose denominators are products of its Darboux polynomials of degree
    at most ``degree`` (``1`` included) and whose numerators have degree
    at most ``max(degree, d) + deg B``, with their cofactors ``D(A/B)``.

    Examples
    ========

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.solvers.darboux import exponential_factors
    >>> x, y = Symbol('x'), Symbol('y')
    >>> exponential_factors(x*y + 1, 1, x, y, degree=1)         # y' = x y + 1
    [ExponentialFactor(x, 1, 1), ExponentialFactor(x**2/2, 1, x)]
    """
    return _field_of(P, Q, x, y, extension).exponential_factors(degree)


def darboux_integrating_factor(P: ExprLike, Q: ExprLike, x: Symbol, y: Symbol, degree: int = 4,
                               extension: Optional[Expr] = None) -> Optional[Expr]:
    """A Darboux integrating factor ``R`` of ``Q dy - P dx`` (``R (Q dy -
    P dx)`` is exact): a product of powers of the Darboux polynomials of
    degree at most ``degree`` and of exponential factors; ``None`` when
    the exponents have no solution.

    Examples
    ========

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.solvers.darboux import darboux_integrating_factor
    >>> x, y = Symbol('x'), Symbol('y')
    >>> darboux_integrating_factor(y*(x - 1), x*(1 - y), x, y)      # Lotka–Volterra
    1/(x*y)
    >>> darboux_integrating_factor(x*y + 1, 1, x, y, degree=1)      # y' = x y + 1
    exp(-x**2/2)
    """
    return _field_of(P, Q, x, y, extension).integrating_factor(degree)


def darboux_first_integral(P: ExprLike, Q: ExprLike, x: Symbol, y: Symbol, degree: int = 4,
                           extension: Optional[Expr] = None) -> Optional[Expr]:
    """A first integral ``I(x, y)`` of ``y' = P/Q`` (``Q I_x + P I_y = 0``)
    by the Prelle–Singer procedure with Darboux polynomials of degree at
    most ``degree``: ``prod f_i**c_i exp(...)`` when a combination of the
    cofactors vanishes, else the quadrature of the Darboux integrating
    factor (through SymPy's ``integrate``); ``None`` when neither is
    found.

    Examples
    ========

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.solvers.darboux import darboux_first_integral
    >>> x, y = Symbol('x'), Symbol('y')
    >>> darboux_first_integral(y*(x - 1), x*(1 - y), x, y)      # Lotka–Volterra
    -x - y + log(x) + log(y)
    >>> darboux_first_integral(x*y**2 - y, x**2*y - x, x, y, degree=2)
    x/y
    """
    return _field_of(P, Q, x, y, extension).first_integral(degree)


def prelle_singer(equation: Basic, f: AppliedUndef, degree: int = 4, extension: Optional[Expr] = None) -> Optional[Eq]:
    """The implicit solution ``Eq(I(x, y), C1)`` of a first order equation
    ``y' = P(x, y)/Q(x, y)`` with polynomial ``P`` and ``Q`` from a first
    integral found by the Prelle–Singer procedure (Darboux polynomials of
    degree at most ``degree``), or ``None``; ``Eq(y, ...)`` when the
    relation is rational in ``y`` with one root. Every step runs under
    the time limit of the settings.

    Examples
    ========

    >>> from sympy_extras.core import Function, Symbol
    >>> from sympy_extras.solvers.darboux import prelle_singer
    >>> x = Symbol('x')
    >>> y = Function('y')(x)
    >>> prelle_singer(x*(1 - y)*y.diff(x) - y*(x - 1), y)
    Eq(-x - y(x) + log(x) + log(y(x)), C1)
    >>> prelle_singer(x**2*y.diff(x) - x*y + 1, y)
    Eq(y(x), C1*x + 1/(2*x))
    """
    parsed = vector_field(equation, f)
    if parsed is None:
        return None
    P, Q, x, y = parsed
    try:
        context = _field_of(P, Q, x, y, extension)
    except ValueError:
        return None
    integral = attempt(lambda: context.first_integral(degree), settings.timeout, native=True)
    if integral is None:
        return None
    C1 = Symbol('C1')
    implicit = Eq(integral.xreplace({y: f}), C1)
    explicit = _explicit(integral, y, C1)
    if explicit is None:
        return implicit
    return Eq(f, explicit.xreplace({y: f}))


def _explicit(integral: Expr, y: Symbol, C1: Symbol) -> Optional[Expr]:
    """``y`` from ``I(x, y) = C1`` when ``I`` is a rational function of
    ``y`` and the equation has one root in ``y`` (``C1 x + 1/(2 x)`` for
    ``(2 x y - 1)/(2 x**2) = C1``), else ``None``: a transcendental
    relation is left implicit rather than cut to one branch (``-LambertW(
    -exp(C1 + x)/x)`` for the Lotka–Volterra integral)."""
    from sympy_extras.polys.rationalsolve import solveset_rational
    from sympy_extras.simplify.simplify import simplify
    if integral.is_rational_function(y) is not True:
        return None
    found = attempt(lambda: solveset_rational(integral - C1, y), _share_of_time(), native=True)
    if found is None:
        return None
    # the excluded points are the roots of the denominator, which a
    # solution misses for generic C1 unless it is one of them
    solutions, excluded = found
    if len(solutions) != 1:
        return None
    solution = solutions[0]
    if solution.has(y) or solution in excluded:
        return None
    # simplified as SymPy's solve does (C1 x + 1/(2 x), not (2 C1 x**2 + 1)/(2 x))
    simplified = attempt(lambda: simplify(solution), _share_of_time(), native=True)
    return solution if simplified is None else simplified
