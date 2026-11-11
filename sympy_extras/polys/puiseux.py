r"""Puiseux expansions of the branches of a plane curve: the Newton–Puiseux
algorithm with the branches grouped into conjugacy classes over number
fields (Duval's rational Puiseux expansions).

Let `F(x, y) \in K[x][y]` be squarefree of degree `n` in `y`, `K` the
rationals or a number field. Above a point `x = a` the `n` roots of `F`
are fractional power series (Puiseux's theorem)

.. math::

    y = \sum_{k \ge k_0} c_k\, (x - a)^{k/e}, \qquad c_k \in \overline{K},

and above `x = \infty` series in `x^{-1/e}`. The *Newton polygon*
algorithm [Walker]_ finds them term by term: for `P(t, Y) = \sum_i a_i(t)
Y^i` the points `(i, \operatorname{ord}_t a_i)` are drawn, the edges of
their lower convex hull with slope `-p/q` carry the roots starting with
`c\, t^{p/q}`, where `u = c^q` is a root of the *characteristic
polynomial* `\chi(u) = \sum_{(i, v) \text{ on the edge}}
\operatorname{lc}(a_i) u^{(i - i_0)/q}`, and the roots continue with the
Newton polygon of `P(\lambda s^q, s^p (c + Y_1))/s^{\nu}`, the vertex
`(m, 0)` of the new polygon holding the `m` roots which share the first
term (`m` the multiplicity of `u` in `\chi`). When a single root is left
(`m = 1`) its remaining terms are a power series found by Newton's
iteration, since `Y_1 = 0` is then a simple root of the reduced
equation: the ramification is decided in finitely many steps, the
discriminant having finite order.

**Conjugate branches.** Following [Duval]_, a root `u` of an irreducible
factor `\psi` of `\chi` over the current coefficient field `L` is taken
in the extension `L(u)` and its branch stands for the `[L(u) : L]`
conjugate branches, so that the expansions come in classes: the `n`
classical series above `a` are the conjugates of the
:class:`PuiseuxExpansion` objects returned, each counting
``ramification * conjugates`` of them. The `q`-th root `c = u^{1/q}` is
avoided by the scaling `x - a = \lambda t^e`: with `l q - m p = 1` the
choice `c = u^l`, `\lambda = u^m` keeps the coefficients in `L(u)`.

**Number fields, exactly.** The fields are the typed
:class:`~sympy_extras.polys.domains.algebraicfield.AlgebraicField`
`\mathbb{Q}(\theta)`, the rationals being the field :data:`RATIONALS` of
degree one (its elements are lists of at most one rational, as those of
every number field are lists of rationals: one type for the
coefficients of every branch). A root `u` of `\psi` over `K =
\mathbb{Q}(\theta)` is adjoined by taking a primitive element `\Theta =
\theta + k u` whose minimal polynomial is the resultant
`\operatorname{Res}_\theta(k^d \psi((T - \theta)/k; \theta), m_K(\theta))`
(squarefree for a suitable `k`), the image of `\theta` in
`\mathbb{Q}(\Theta)` being the common root of `m_K(t)` and `\psi((\Theta
- t)/k; t)`, a gcd (:func:`adjoin_root`): no root is evaluated
numerically. The generator `\Theta` of a new field is written as the
largest real root of its minimal polynomial (an
:class:`~sympy_extras.polys.cad.indexedroot.IndexedRoot`, in radicals for
a quadratic or binomial polynomial, SymPy's ``CRootOf(..., radicals=True)``
policy), as a root in radicals when it has no real root, and as the
symbol ``theta`` otherwise (the field is the same whichever root is
meant; the expression only names it). Every expansion carries the
:class:`Embedding` of the coefficient field of the curve into the field
of its coefficients.

The expansions carry the point, the ramification index `e`, the field
of their coefficients and the coefficients `c_k` of `t^k`, `x - a =
\lambda t^e` (`x = 1/(\lambda t^e)` at infinity), known below a stated
order; an element of `K(x)[y]` evaluated along a branch
(:meth:`PuiseuxExpansion.evaluate`) is a truncated Laurent series in `t`
whose valuation is the order of the function at the place.

The module is native: it imports no SymPy
(:mod:`sympy_extras.polys.puiseux_sympy` is its face on SymPy's
``Poly`` and number fields, used by the integration on curves).

References
==========

.. [Walker] R. J. Walker, *Algebraic curves*, Princeton 1950, chapter IV
   (the Newton polygon and Puiseux's theorem).
.. [Duval] D. Duval, *Rational Puiseux expansions*, Compositio
   Mathematica 70 (1989), 119–154.
.. [PoteauxWeimann] A. Poteaux, M. Weimann, *Computing Puiseux series: a
   fast divide and conquer algorithm*, Annales Henri Lebesgue 4 (2021),
   the regularity index and the grouping into conjugacy classes.
.. [Trager76] B. M. Trager, *Algebraic factoring and rational function
   integration*, SYMSAC 1976 (norms and primitive elements of towers
   of number fields).
"""
from __future__ import annotations

from fractions import Fraction
from math import lcm
from typing import Final, Optional

from sympy_extras.core.add import add
from sympy_extras.core.expr import Expr
from sympy_extras.core.numbers import Integer, Rational, number_from_fraction, oo
from sympy_extras.core.power import pow
from sympy_extras.core.symbol import Symbol

from . import densearith, densebasic, densetools, euclidtools
from .domains.algebraicfield import AlgebraicField
from .domains.polynomialring import PolynomialRing
from .domains.rationalfield import QQ
from .polyutils import dict_from_expr

__all__ = ['PuiseuxExpansion', 'LaurentSeries', 'puiseux_expansions', 'expansions_in_field',
           'branches_at_infinity', 'adjoin_root', 'Embedding', 'field_of', 'minimal_polynomial_of',
           'in_subfield', 'generator', 'degree', 'field_from_polynomial', 'factor_over', 'curve_from_expr',
           'RATIONALS', 'Number', 'Curve']

#: an element of a number field: its coordinates in the powers of the
#: generator, highest degree first (``[]`` is zero)
Number = list[Fraction]

#: a polynomial in `x` and `y` over a number field: ``{i: {k: coefficient
#: of x**k y**i}}``
Curve = dict[int, dict[int, Number]]

#: the rationals as the number field of degree one: the elements are
#: ``[]`` and ``[q]``, which are also the rationals of every number field
RATIONALS: Final[AlgebraicField] = AlgebraicField([Fraction(1), Fraction(0)], Integer(0))
RATIONALS.name = 'QQ'


class LaurentSeries:
    """A truncated Laurent series `\\sum_k c_k t^k` with coefficients in
    the number field ``K``: the coefficients of `t^k` are known for `k <
    order` (``order`` ``None`` when the series is exact, a Laurent
    polynomial). Arithmetic keeps track of the order of the result."""

    __slots__ = ('K', 'order', 'terms')

    def __init__(self, K: AlgebraicField, terms: dict[int, Number], order: Optional[int]) -> None:
        self.K: Final[AlgebraicField] = K
        self.order: Final[Optional[int]] = order
        self.terms: Final[dict[int, Number]] = {k: c for k, c in terms.items()
                                                if c and (order is None or k < order)}

    @classmethod
    def constant(cls, K: AlgebraicField, c: Number) -> LaurentSeries:
        return cls(K, {0: c}, None)

    @classmethod
    def monomial(cls, K: AlgebraicField, c: Number, k: int) -> LaurentSeries:
        return cls(K, {k: c}, None)

    def valuation(self) -> Optional[int]:
        """The least exponent with a nonzero coefficient, ``None`` when
        no term is known (zero up to the order)."""
        if not self.terms:
            return None
        return min(self.terms)

    def is_zero(self) -> bool:
        return not self.terms

    def coefficient(self, k: int) -> Number:
        return self.terms.get(k, self.K.zero)

    def __add__(self, other: LaurentSeries) -> LaurentSeries:
        K = self.K
        order = _min_order(self.order, other.order)
        terms = dict(self.terms)
        for k, c in other.terms.items():
            terms[k] = K.add(terms.get(k, K.zero), c)
        return LaurentSeries(K, terms, order)

    def __neg__(self) -> LaurentSeries:
        return LaurentSeries(self.K, {k: self.K.neg(c) for k, c in self.terms.items()}, self.order)

    def __sub__(self, other: LaurentSeries) -> LaurentSeries:
        return self + (-other)

    def scaled(self, c: Number) -> LaurentSeries:
        return LaurentSeries(self.K, {k: self.K.mul(c, v) for k, v in self.terms.items()}, self.order)

    def shifted(self, k: int) -> LaurentSeries:
        """The series times `t^k`."""
        order = None if self.order is None else self.order + k
        return LaurentSeries(self.K, {j + k: c for j, c in self.terms.items()}, order)

    def __mul__(self, other: LaurentSeries) -> LaurentSeries:
        K = self.K
        v1, v2 = self.valuation(), other.valuation()
        order: Optional[int]
        if (self.order is None and not self.terms) or (other.order is None and not other.terms):
            # an exact zero: the product is exactly zero
            return LaurentSeries(K, {}, None)
        if self.order is None and other.order is None:
            order = None
        else:
            candidates: list[int] = []
            if other.order is not None:
                # the unknown part of other, of order >= other.order, times the
                # known part of self, of valuation >= v1
                candidates.append(other.order + (v1 if v1 is not None
                                                 else (self.order if self.order is not None else 0)))
            if self.order is not None:
                candidates.append(self.order + (v2 if v2 is not None
                                                else (other.order if other.order is not None else 0)))
            order = min(candidates)
        terms: dict[int, Number] = {}
        for j, c in self.terms.items():
            for k, d in other.terms.items():
                if order is not None and j + k >= order:
                    continue
                terms[j + k] = K.add(terms.get(j + k, K.zero), K.mul(c, d))
        return LaurentSeries(K, terms, order)

    def truncated(self, order: int) -> LaurentSeries:
        return LaurentSeries(self.K, self.terms, order if self.order is None else min(order, self.order))

    def power(self, n: int) -> LaurentSeries:
        result = LaurentSeries.constant(self.K, self.K.one)
        base = self
        while n > 0:
            if n % 2:
                result = result * base
            n //= 2
            if n:
                base = base * base
        return result

    def inverse(self, order: int) -> LaurentSeries:
        """The inverse series, known ``order`` terms beyond its valuation."""
        v = self.valuation()
        if v is None:
            raise ZeroDivisionError("inverse of a series which is zero to its order")
        K = self.K
        unit = self.shifted(-v)
        # unit = sum a_k t^k with a_0 != 0: the coefficients b_k of the
        # inverse by the recurrence a_0 b_k = -sum_{j=1..k} a_j b_{k-j}
        a = {k: c for k, c in unit.terms.items() if k < order}
        inv0 = K.invert(unit.coefficient(0))
        b: dict[int, Number] = {0: inv0}
        for k in range(1, order):
            total = K.zero
            for j, a_j in a.items():
                if 0 < j <= k:
                    b_prev = b.get(k - j)
                    if b_prev is not None:
                        total = K.add(total, K.mul(a_j, b_prev))
            if total:
                b[k] = K.neg(K.mul(inv0, total))
        known = order if unit.order is None else min(order, unit.order)
        return LaurentSeries(K, b, known).shifted(-v)

    def substituted(self, q: int) -> LaurentSeries:
        """The series with `t` replaced by `t^q`."""
        order = None if self.order is None else self.order * q
        return LaurentSeries(self.K, {k * q: c for k, c in self.terms.items()}, order)

    def converted(self, embedding: Embedding) -> LaurentSeries:
        return LaurentSeries(embedding.target, {k: embedding(c) for k, c in self.terms.items()}, self.order)

    def as_expr(self, t: Expr, exponent_scale: int = 1, reciprocal: bool = False) -> Expr:
        """`\\sum_k c_k t^{k/\\text{exponent\\_scale}}` as an expression
        (`\\sum_k c_k t^{-k/\\text{exponent\\_scale}}` with ``reciprocal``).

        >>> from fractions import Fraction
        >>> from sympy_extras.core import Symbol
        >>> from sympy_extras.polys.puiseux import LaurentSeries, RATIONALS
        >>> s = LaurentSeries(RATIONALS, {-1: [Fraction(1)], 2: [Fraction(3)]}, 4)
        >>> s.as_expr(Symbol('t')), s.as_expr(Symbol('t'), 2)
        (3*t**2 + 1/t, 3*t + 1/sqrt(t))
        """
        sign = -1 if reciprocal else 1
        return add(*[self.K.to_expr(c) * pow(t, number_from_fraction(Fraction(sign * k, exponent_scale)))
                     for k, c in sorted(self.terms.items())])

    def __repr__(self) -> str:
        return f"LaurentSeries({dict(sorted(self.terms.items()))}, order={self.order})"


def _min_order(a: Optional[int], b: Optional[int]) -> Optional[int]:
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


# Number fields

class Embedding:
    """The embedding of a number field ``source`` into a number field
    ``target`` containing it, given by the image of the generator of
    the source (not needed for the rationals, nor when the fields are
    the same)."""

    __slots__ = ('source', 'target', 'image')

    def __init__(self, source: AlgebraicField, target: AlgebraicField, image: Optional[Number] = None) -> None:
        needed = source.degree > 1 and source != target
        if needed and image is None:
            raise ValueError("the image of the generator is needed")
        self.source: Final[AlgebraicField] = source
        self.target: Final[AlgebraicField] = target
        self.image: Final[Optional[Number]] = image if needed else None

    def __call__(self, c: Number) -> Number:
        if self.source == self.target:
            return c
        target = self.target
        if self.image is None:
            # the rationals into any field
            return target.from_rational(c[0]) if c else target.zero
        # c(image) by Horner's rule
        total = target.zero
        for a in c:
            total = target.add(target.mul(total, self.image), target.from_rational(a))
        return total

    def then(self, other: Embedding) -> Embedding:
        """The composition with ``other``, an embedding of ``target``."""
        if other.source != self.target:
            raise ValueError("the embeddings do not compose")
        if self.source == self.target:
            return other
        if other.source == other.target:
            return self
        if self.image is None:
            return Embedding(self.source, other.target)
        return Embedding(self.source, other.target, other(self.image))

    def __repr__(self) -> str:
        return f"Embedding({self.source}, {self.target}, {self.image})"


def generator(K: AlgebraicField) -> Number:
    """The generator of the number field ``K`` (one for the rationals)."""
    return K.alpha if K.degree > 1 else K.one


def degree(K: AlgebraicField) -> int:
    """The degree of the number field ``K`` over the rationals."""
    return K.degree


def _coefficient_list(c: Number, K: AlgebraicField) -> list[Fraction]:
    """The rational coordinates of ``c`` in the powers of the generator
    of ``K``, low degree first, padded to the degree."""
    coefficients = list(reversed(c))
    return coefficients + [Fraction(0)] * (K.degree - len(coefficients))


def _monic(f: list[Fraction]) -> list[Fraction]:
    return densetools.dup_monic(densebasic.dup_strip(f, QQ), QQ)


#: the fields built from a minimal polynomial, by its coefficients (the
#: same polynomial gives the same field)
_FIELDS: dict[tuple[Fraction, ...], AlgebraicField] = {}


def _generator_expression(N: list[Fraction]) -> Expr:
    """The largest real root of the monic irreducible ``N`` (the first
    root in SymPy's order when it has none), in radicals for a linear,
    quadratic or binomial polynomial; ``theta`` when it is not real and
    not in radicals."""
    from .cad.indexedroot import IndexedRoot, bound_variable
    from .polytools import Poly
    from .roots import _trivial_roots
    from .rootisolation import dup_count_real_roots
    denominator = lcm(*[c.denominator for c in N])
    integers = [int(c * denominator) for c in N]
    t = bound_variable('T')
    d = len(N) - 1
    poly = Poly.from_dict({(d - i,): c for i, c in enumerate(integers) if c}, t)
    real = dup_count_real_roots(N)
    index = real - 1 if real > 0 else 0
    trivial = _trivial_roots(poly)
    if trivial is not None:
        return trivial[index]
    if real > 0:
        return IndexedRoot(poly.as_expr(), t, index)
    return Symbol('theta')


def field_from_polynomial(N: list[Fraction]) -> AlgebraicField:
    """`\\mathbb{Q}(\\Theta)` for `\\Theta` a root of the irreducible ``N``
    (rational coefficients, highest degree first), written as the largest
    real root when there is one; the rationals for a linear ``N``.

    >>> from fractions import Fraction
    >>> from sympy_extras.polys.puiseux import field_from_polynomial
    >>> field_from_polynomial([Fraction(1), Fraction(0), Fraction(-2)])
    QQ<sqrt(2)>
    >>> field_from_polynomial([Fraction(1), Fraction(0), Fraction(-1), Fraction(-1)])
    QQ<IndexedRoot(T**3 - T - 1, T, 0)>
    """
    monic = _monic(N)
    if len(monic) <= 2:
        return RATIONALS
    key = tuple(monic)
    found = _FIELDS.get(key)
    if found is None:
        found = AlgebraicField(monic, _generator_expression(monic))
        _FIELDS[key] = found
    return found


def factor_over(K: AlgebraicField, coefficients: list[Number]) -> list[tuple[list[Number], int]]:
    """The irreducible factors over ``K`` of the polynomial with the
    given coefficients (highest first), monic, with multiplicities (over
    the rationals by Zassenhaus and van Hoeij, over a number field by
    Trager's algorithm)."""
    f = densebasic.dup_strip(coefficients, K)
    if densebasic.dup_degree(f) < 1:
        return []
    if K.degree == 1:
        _, factors = QQ.factor_dense([c[0] if c else Fraction(0) for c in f])
        return [([K.from_rational(c) for c in factor], multiplicity) for factor, multiplicity in factors]
    _, ext_factors = K.factor_dense(f)
    return [(factor, multiplicity) for factor, multiplicity in ext_factors]


def adjoin_root(K: AlgebraicField, psi: list[Number]) -> tuple[AlgebraicField, Embedding, Number]:
    """The field ``L = K(u)`` for a root ``u`` of the polynomial ``psi``
    (coefficients in ``K``, highest degree first) irreducible over
    ``K``, with the embedding of ``K`` into ``L`` and ``u``; exact: no
    root is evaluated numerically.

    >>> from fractions import Fraction
    >>> from sympy_extras.polys.puiseux import adjoin_root, field_from_polynomial
    >>> K = field_from_polynomial([Fraction(1), Fraction(0), Fraction(-2)])
    >>> L, into, u = adjoin_root(K, [K.one, K.zero, K.neg(K.alpha)])
    >>> L.degree, L.mul(u, u) == into(K.alpha)
    (4, True)
    """
    d = len(psi) - 1
    if d == 1:
        return K, Embedding(K, K), K.neg(K.quo(psi[1], psi[0]))
    if K.degree == 1:
        L = field_from_polynomial([c[0] if c else Fraction(0) for c in psi])
        return L, Embedding(K, L), generator(L)
    QT = PolynomialRing(QQ, 'T')
    m_K: list[list[Fraction]] = [[c] if c else [] for c in K.minpoly]   # in theta, over QQ[T]
    N: list[Fraction] = []
    P: list[list[Fraction]] = []
    k = 0
    for k in range(0, 2 * d * K.degree + 3):
        # Theta = theta + k u: P(T, theta) = k^d psi((T - theta)/k) for k > 0,
        # psi(T) for k = 0, as a polynomial in theta over QQ[T]
        P = _shifted_psi(psi, k, d, QT)
        N = euclidtools.dup_resultant(P, m_K, QT)
        if densebasic.dup_degree(N) == d * K.degree and \
                densebasic.dup_degree(euclidtools.dup_gcd(N, densetools.dup_diff(N, 1, QQ), QQ)) == 0:
            break
    else:
        raise ValueError("no primitive element found")
    L = field_from_polynomial(N)
    Theta = generator(L)
    # the image t of theta: the common root of m_K(t) and P(Theta, t)
    m_K_L = [L.from_rational(c) for c in K.minpoly]
    P_L = densebasic.dup_strip([_evaluate_rational(c, Theta, L) for c in P], L)
    common = euclidtools.dup_gcd(m_K_L, P_L, L)
    if densebasic.dup_degree(common) != 1:
        raise ValueError("the image of the generator is not unique")
    a, b = common
    image = L.neg(L.quo(b, a))
    u = Theta if k == 0 else L.quo(L.sub(Theta, image), L.from_int(k))
    into = Embedding(K, L, image)
    value = L.zero
    for c in psi:
        value = L.add(L.mul(value, u), into(c))
    if value:
        raise ValueError("the adjoined root does not satisfy its polynomial")
    return L, into, u


def _shifted_psi(psi: list[Number], k: int, d: int, QT: PolynomialRing[Fraction]) -> list[list[Fraction]]:
    """`k^d \\psi((T - \\theta)/k; \\theta)` (`\\psi(T; \\theta)` for `k =
    0`) as a polynomial in `\\theta` with coefficients in `\\mathbb{Q}[T]`,
    highest degree first."""
    total: list[list[Fraction]] = []
    # psi = sum_i psi_i(theta) z^i, psi_i the coefficient of z^i
    for i, c in enumerate(reversed(psi)):
        if not c:
            continue
        # psi_i(theta) as a polynomial in theta over QQ[T]
        coefficient: list[list[Fraction]] = densebasic.dup_strip([[a] if a else [] for a in c], QT)
        if k == 0:
            # times T^i
            term = [QT.mul(a, [Fraction(1)] + [Fraction(0)] * i) for a in coefficient]
        else:
            # times k^(d - i) (T - theta)^i: T - theta = -theta + T
            linear: list[list[Fraction]] = [[Fraction(-1)], [Fraction(1), Fraction(0)]]
            power = densearith.dup_pow(linear, i, QT)
            scale = [Fraction(k) ** (d - i)]
            term = [QT.mul(a, scale) for a in densearith.dup_mul(coefficient, power, QT)]
        total = densearith.dup_add(total, densebasic.dup_strip(term, QT), QT)
    return total


def _evaluate_rational(c: list[Fraction], value: Number, L: AlgebraicField) -> Number:
    """The polynomial ``c`` with rational coefficients at ``value``."""
    total = L.zero
    for a in c:
        total = L.add(L.mul(total, value), L.from_rational(a))
    return total


def field_of(point: Expr) -> tuple[AlgebraicField, Number]:
    """The number field generated by an algebraic number given as an
    expression (written with radicals, ``I``, or a real root
    :class:`~sympy_extras.polys.cad.indexedroot.IndexedRoot`), and the
    number as its element, the generator.

    >>> from sympy_extras.core import sqrt, Integer, Rational
    >>> from sympy_extras.polys.puiseux import field_of
    >>> field_of(sqrt(Integer(2)) + 1), field_of(Rational(1, 2))
    ((QQ<1 + sqrt(2)>, [Fraction(1, 1), Fraction(0, 1)]), (QQ, [Fraction(1, 2)]))
    """
    if isinstance(point, Rational):
        return RATIONALS, RATIONALS.from_rational(QQ.from_expr(point))
    from .cad.indexedroot import IndexedRoot, algebraic_of
    minpoly: list[Fraction]
    if isinstance(point, IndexedRoot):
        number = algebraic_of(point)
        if number is None:
            raise ValueError(f"{point} is not an algebraic number")
        minpoly = _monic([Fraction(c) for c in number.minpoly])
    else:
        from .numberfields.minpoly import minimal_polynomial_coeffs
        minpoly = _monic([Fraction(c) for c in minimal_polynomial_coeffs(point)])
    if len(minpoly) <= 2:
        return RATIONALS, RATIONALS.from_rational(-minpoly[1])
    K = AlgebraicField(minpoly, point)
    return K, K.alpha


def minimal_polynomial_of(c: Number, K: AlgebraicField) -> list[Fraction]:
    """The minimal polynomial over the rationals of an element of the
    number field ``K``, monic, highest degree first, exactly.

    >>> from fractions import Fraction
    >>> from sympy_extras.polys.puiseux import minimal_polynomial_of, field_from_polynomial
    >>> K = field_from_polynomial([Fraction(1), Fraction(0), Fraction(-2)])
    >>> minimal_polynomial_of(K.add(K.alpha, K.one), K)
    [Fraction(1, 1), Fraction(-2, 1), Fraction(-1, 1)]
    """
    if K.degree == 1:
        return [Fraction(1), -(c[0] if c else Fraction(0))]
    return K.minimal_polynomial(c)


def in_subfield(c: Number, L: AlgebraicField, into: Embedding) -> Optional[Number]:
    """``c`` of ``L`` as an element of the subfield ``into.source``
    (``into`` its embedding into ``L``), ``None`` when it is not in it."""
    K = into.source
    if K == L:
        return c
    if K.degree == 1:
        coefficients = _coefficient_list(c, L)
        if any(coefficients[1:]):
            return None
        return K.from_rational(coefficients[0])
    d = K.degree
    columns: list[list[Fraction]] = []
    power = L.one
    image = into(generator(K))
    for _ in range(d):
        columns.append(_coefficient_list(power, L))
        power = L.mul(power, image)
    target = _coefficient_list(c, L)
    rows = [{j: columns[j][i] for j in range(d) if columns[j][i]} for i in range(len(target))]
    for i, row in enumerate(rows):
        if target[i]:
            row[d] = target[i]
    echelon = QQ.sparse_echelon([row for row in rows if row])
    solution = [Fraction(0)] * d
    for row in echelon:
        pivot = min(row)
        if pivot == d:
            return None
        solution[pivot] = row.get(d, Fraction(0))
    return densebasic.dup_strip(list(reversed(solution)), QQ)


# The expansions

class PuiseuxExpansion:
    """One branch of the curve `F(x, y) = 0` above ``point`` (an
    expression, or ``oo``): `y = \\sum_k c_k t^k` with `x -
    \\text{point} = \\text{scale} \\cdot t^e` (`x = 1/(\\text{scale}\\, t^e)`
    at infinity), the coefficients in the number field ``field`` and
    known for `k <` ``order``; it stands for ``conjugates`` branches,
    conjugate under the automorphisms of ``field`` over the field of
    the point. ``embedding`` maps the coefficient field of the curve
    (extended by the point) into ``field``.

    >>> from sympy_extras.core import symbols, Integer
    >>> from sympy_extras.polys.puiseux import puiseux_expansions
    >>> x, y = symbols('x y')
    >>> [b.as_expr(x) for b in puiseux_expansions(y**2 - x**3 - x**2, x, y, Integer(0), order=4)]
    [x**3/8 - x**2/2 - x, -x**3/8 + x**2/2 + x]
    >>> branches = puiseux_expansions(y**3 - x**2, x, y, Integer(0), order=2)
    >>> len(branches), branches[0].ramification, branches[0].as_expr(x)
    (1, 3, x**(2/3))
    """

    __slots__ = ('point', 'point_element', 'ramification', 'field', 'series', 'conjugates', 'x', 'y', 'scale',
                 'embedding')

    def __init__(self, point: Expr, point_element: Optional[Number], ramification: int, field: AlgebraicField,
                 series: LaurentSeries, conjugates: int, x: Symbol, y: Symbol, scale: Number,
                 embedding: Embedding) -> None:
        self.point: Final[Expr] = point
        #: the point as an element of ``field`` (``None`` at infinity)
        self.point_element: Final[Optional[Number]] = point_element
        self.ramification: Final[int] = ramification
        self.field: Final[AlgebraicField] = field
        self.series: Final[LaurentSeries] = series
        self.conjugates: Final[int] = conjugates
        self.x: Final[Symbol] = x
        self.y: Final[Symbol] = y
        self.scale: Final[Number] = scale
        self.embedding: Final[Embedding] = embedding

    @property
    def order(self) -> Optional[int]:
        return self.series.order

    @property
    def at_infinity(self) -> bool:
        return self.point_element is None

    def as_expr(self, x: Optional[Expr] = None) -> Expr:
        """The series in `x` (the coefficients as expressions)."""
        x_ = self.x if x is None else x
        scale = self.field.to_expr(self.scale)
        if self.at_infinity:
            return self.series.as_expr(scale * x_, self.ramification, reciprocal=True)
        return self.series.as_expr((x_ - self.point) / scale, self.ramification)

    def x_series(self) -> LaurentSeries:
        """`x` as a series in `t`."""
        K = self.field
        if self.point_element is None:
            return LaurentSeries.monomial(K, K.invert(self.scale), -self.ramification)
        return LaurentSeries(K, {0: self.point_element, self.ramification: self.scale}, None)

    def evaluate(self, p: Curve, order: Optional[int] = None) -> LaurentSeries:
        """The polynomial ``p`` in ``x`` and ``y`` (``{i: {k: coefficient
        of x**k y**i}}``, the coefficients rational or in the coefficient
        field of the curve, the source of ``embedding``) along the branch,
        as a series in `t`."""
        K = self.field
        xs = self.x_series()
        ys = self.series if order is None else self.series.truncated(order)
        total = LaurentSeries.constant(K, K.zero)
        for i in range(max(p, default=0), -1, -1):
            total = total * ys + _evaluate_x(p.get(i, {}), xs, K, self.embedding)
        return total

    def evaluate_expr(self, p: Expr, order: Optional[int] = None) -> LaurentSeries:
        """:meth:`evaluate` of a polynomial given as an expression in
        ``x`` and ``y`` (coefficients rational or in the coefficient field
        of the curve)."""
        return self.evaluate(curve_from_expr(p, self.x, self.y, self.embedding.source), order)

    def __repr__(self) -> str:
        return (f"PuiseuxExpansion(point={self.point}, ramification={self.ramification}, "
                f"conjugates={self.conjugates}, series={self.as_expr()})")


def _evaluate_x(p: dict[int, Number], xs: LaurentSeries, K: AlgebraicField, convert: Embedding) -> LaurentSeries:
    total = LaurentSeries.constant(K, K.zero)
    for k in range(max(p, default=0), -1, -1):
        c = p.get(k)
        total = total * xs + LaurentSeries.constant(K, K.zero if c is None else convert(c))
    return total


def curve_from_expr(F: Expr, x: Symbol, y: Symbol, K: AlgebraicField) -> Curve:
    """The polynomial ``F`` in ``x`` and ``y`` with coefficients in ``K``
    as ``{i: {k: coefficient of x**k y**i}}``."""
    P: Curve = {}
    for (i, k), c in dict_from_expr(F, [y.name, x.name], K).items():
        P.setdefault(i, {})[k] = c
    return P


# The Newton polygon

def _valuation(a: dict[int, Number]) -> Optional[int]:
    return min(a) if a else None


def _lower_hull(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """The vertices of the lower convex hull of ``points`` (distinct
    abscissae, sorted)."""
    hull: list[tuple[int, int]] = []
    for point in points:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            x3, y3 = point
            if (y2 - y1) * (x3 - x1) >= (y3 - y1) * (x2 - x1):
                hull.pop()
            else:
                break
        hull.append(point)
    return hull


def _poly2_substitute(P: Curve, q: int, p: int, c: Number, scale: Number, K: AlgebraicField) -> Curve:
    """`P(\\lambda s^q, s^p (c + Y))`, the exponents of `s` as they come
    (possibly negative), `\\lambda` the ``scale``."""
    result: Curve = {}
    scale_powers: dict[int, Number] = {}

    def scaled(k: int) -> Number:
        found = scale_powers.get(k)
        if found is None:
            found = K.pow(scale, k) if k >= 0 else K.pow(K.invert(scale), -k)
            scale_powers[k] = found
        return found

    degree_Y = max(P) if P else 0
    powers_of_c: list[Number] = [K.one]
    for _ in range(degree_Y):
        powers_of_c.append(K.mul(powers_of_c[-1], c))
    binomials: list[list[int]] = [[1]]
    for i in range(1, degree_Y + 1):
        row = [1]
        for j in range(1, i):
            row.append(binomials[i - 1][j - 1] + binomials[i - 1][j])
        row.append(1)
        binomials.append(row)
    for i, a in P.items():
        for j in range(i + 1):
            factor = K.mul(K.from_int(binomials[i][j]), powers_of_c[i - j])
            if not factor:
                continue
            target = result.setdefault(j, {})
            for k, coefficient in a.items():
                exponent = q * k + p * i
                target[exponent] = K.add(target.get(exponent, K.zero), K.mul(K.mul(factor, coefficient), scaled(k)))
    return _poly2_clean(result)


def _poly2_convert(P: Curve, into: Embedding) -> Curve:
    return {i: {k: into(v) for k, v in a.items()} for i, a in P.items()}


def _poly2_shift(P: Curve, nu: int) -> Curve:
    return {i: {k - nu: v for k, v in a.items()} for i, a in P.items()}


def _poly2_clean(P: Curve) -> Curve:
    return {i: a for i, a in ((i, {k: v for k, v in a.items() if v}) for i, a in P.items()) if a}


def _poly2_evaluate(P: Curve, Y: LaurentSeries, K: AlgebraicField, order: Optional[int] = None) -> LaurentSeries:
    """`P(s, Y(s))`, the terms of exponent at least ``order`` dropped
    along the way when it is given (`Y` of nonnegative valuation)."""
    degree_Y = max(P) if P else 0
    total = LaurentSeries.constant(K, K.zero)
    for i in range(degree_Y, -1, -1):
        coefficients = P.get(i, {})
        if order is not None:
            coefficients = {k: c for k, c in coefficients.items() if k < order}
        total = total * Y + LaurentSeries(K, coefficients, None)
        if order is not None:
            total = total.truncated(order)
    return total


def _poly2_derivative(P: Curve, K: AlgebraicField) -> Curve:
    return _poly2_clean({i - 1: {k: K.mul(K.from_int(i), v) for k, v in a.items()} for i, a in P.items() if i >= 1})


def _poly2_translate(P: Curve, a: Number, K: AlgebraicField) -> Curve:
    """`P(x + a, y)`."""
    if not a:
        return P
    result: Curve = {}
    for i, coefficients in P.items():
        highest = max(coefficients)
        shifted: dict[int, Number] = {}
        for k in range(highest, -1, -1):
            new: dict[int, Number] = {}
            for j, c in shifted.items():
                new[j + 1] = K.add(new.get(j + 1, K.zero), c)
                new[j] = K.add(new.get(j, K.zero), K.mul(c, a))
            c_k = coefficients.get(k)
            if c_k is not None:
                new[0] = K.add(new.get(0, K.zero), c_k)
            shifted = new
        result[i] = shifted
    return _poly2_clean(result)


def _newton_root(P: Curve, K: AlgebraicField, order: int) -> LaurentSeries:
    """The power series root `Y(s)` with `Y(0) = 0` of `P(s, Y)`, `Y = 0` a
    simple root of `P(0, Y)`, known below `s**order`."""
    dP = _poly2_derivative(P, K)
    root = LaurentSeries(K, {}, 1)
    precision = 1
    while precision < order:
        precision = min(2 * precision, order)
        # the approximation taken as exact: the value of P at it is then
        # exact and its valuation is the precision reached
        approximation = LaurentSeries(K, root.terms, None)
        value = _poly2_evaluate(P, approximation, K, precision)
        if value.is_zero():
            root = LaurentSeries(K, root.terms, precision)
            continue
        derivative = _poly2_evaluate(dP, approximation, K, precision)
        correction = (value * derivative.inverse(precision)).truncated(precision)
        root = (approximation - correction).truncated(precision)
    return root.truncated(order)


class _Branch:
    """A branch found by the recursion: the series of `Y` in the final
    variable `u` (`s = scale * u**q`, `s` the variable of the caller),
    its field with the embedding of the caller's field, and the number
    of conjugates."""

    __slots__ = ('q', 'scale', 'series', 'field', 'conjugates', 'embedding')

    def __init__(self, q: int, scale: Number, series: LaurentSeries, field: AlgebraicField, conjugates: int,
                 embedding: Embedding) -> None:
        self.q: Final[int] = q
        self.scale: Final[Number] = scale
        self.series: Final[LaurentSeries] = series
        self.field: Final[AlgebraicField] = field
        self.conjugates: Final[int] = conjugates
        self.embedding: Final[Embedding] = embedding


def _characteristic(P: Curve, edge: tuple[tuple[int, int], tuple[int, int]], q: int,
                    K: AlgebraicField) -> list[Number]:
    """The characteristic polynomial of the edge in `u = c^q`,
    coefficients highest degree first."""
    (i0, v0), (i1, v1) = edge
    p_over_q = Fraction(v0 - v1, i1 - i0)
    degree_u = (i1 - i0) // q
    coefficients: list[Number] = [K.zero] * (degree_u + 1)
    for i in range(i0, i1 + 1):
        if (i - i0) % q:
            continue
        a = P.get(i)
        if not a:
            continue
        v = min(a)
        if Fraction(v) == Fraction(v0) - p_over_q * (i - i0):
            coefficients[degree_u - (i - i0) // q] = a[v]
    return coefficients


def _bezout(q: int, p: int) -> tuple[int, int]:
    """``(l, m)`` with ``l*q - m*p == 1`` (``q > 0``, ``gcd(p, q) == 1``),
    ``m = 0`` when ``q == 1``."""
    if q == 1:
        return 1, 0
    old_r, r = q, p
    old_s, s_ = 1, 0
    old_t, t_ = 0, 1
    while r != 0:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s_ = s_, old_s - quotient * s_
        old_t, t_ = t_, old_t - quotient * t_
    sign = 1 if old_r == 1 else -1
    return sign * old_s, -sign * old_t


def _root_and_scale(u: Number, q: int, p: int, L: AlgebraicField) -> tuple[Number, Number]:
    """``(c, scale)`` with `c^q = u\\, \\text{scale}^p`: a `q`-th root of `u` in
    `L` with no scaling when there is one (`u = 1`, a square in `L`),
    else Duval's `c = u^l`, `\\lambda = u^m` with `l q - m p = 1`, `0 \\le m
    < q`."""
    if q > 1:
        roots = [L.neg(L.quo(factor[1], factor[0]))
                 for factor, _ in factor_over(L, [L.one] + [L.zero] * (q - 1) + [L.neg(u)]) if len(factor) == 2]
        if roots:
            # the root of the first linear factor in SymPy's order of the
            # factors (as the expansions were computed before they were
            # native: another root gives the same branches, t -> -t)
            return min(roots, key=lambda r: _factor_key(r, L)), L.one
    l, m = _bezout(q, p)
    # the other solutions: (l + p t, m + q t)
    while m < 0:
        l, m = l + p, m + q
    while m >= q:
        l, m = l - p, m - q
    return _power(u, l, L), _power(u, m, L)


def _factor_key(r: Number, L: AlgebraicField) -> tuple[Fraction, ...]:
    """The key of the linear factor `z - r` in SymPy's order of the
    factors: its coefficients, monic over a number field (the rational
    coordinates of `-r`, highest degree first), primitive with integer
    coefficients over the rationals."""
    if L.degree > 1:
        return tuple(L.neg(r))
    value = r[0] if r else Fraction(0)
    return (Fraction(value.denominator), Fraction(-value.numerator))


def _power(u: Number, k: int, K: AlgebraicField) -> Number:
    if k >= 0:
        return K.pow(u, k)
    return K.pow(K.invert(u), -k)


def _expand(P: Curve, K: AlgebraicField, follow: int, first: bool, order_x: Fraction,
            ramification_so_far: int, offset: int) -> list[_Branch]:
    """The branches of `P(s, Y) = 0` through the vertex `(follow, 0)` (all
    the roots when ``first``), each known below `s^{N}` with `N` the
    order ``order_x`` in units of `x` times the ramification of `s` and
    of the branch, less the ``offset`` by which the caller shifts the
    series (`y = s^{\\text{offset}} (\\ldots + Y)`)."""
    P = _poly2_clean(P)
    branches: list[_Branch] = []
    if not first and 0 not in P:
        # Y = 0 is an exact root
        branches.append(_Branch(1, K.one, LaurentSeries(K, {}, None), K, 1, Embedding(K, K)))
        P = {i - 1: a for i, a in P.items() if i >= 1}
        follow -= 1
        if follow == 0:
            return branches
    indices = sorted(i for i in P if first or i <= follow)
    points = [(i, min(P[i])) for i in indices]
    hull = _lower_hull(points)
    for (i0, v0), (i1, v1) in zip(hull, hull[1:]):
        slope = Fraction(v0 - v1, i1 - i0)  # gamma
        if not first and slope <= 0:
            continue
        p, q = slope.numerator, slope.denominator
        chi = _characteristic(P, ((i0, v0), (i1, v1)), q, K)
        for psi, multiplicity in factor_over(K, chi):
            # u a root of the characteristic polynomial in u = c^q / lambda^p:
            # with l q - m p = 1 the choice c = u^l, lambda = u^m needs no
            # q-th root (Duval)
            L, into, u = adjoin_root(K, psi)
            c, scale = _root_and_scale(u, q, p, L)
            P1 = _poly2_substitute(_poly2_convert(P, into), q, p, c, scale, L)
            nu = min(k for a in P1.values() for k in a)
            P1 = _poly2_shift(P1, nu)
            ramification = ramification_so_far * q
            total_offset = offset * q + p
            if multiplicity == 1:
                # at least the leading term, whatever the order asked
                needed = max(_ceil(order_x * ramification) - total_offset, 1)
                root = _newton_root(P1, L, needed)
                series = (LaurentSeries.constant(L, c) + root).shifted(p)
                branches.append(_Branch(q, scale, series, L, len(psi) - 1, into))
            else:
                for sub in _expand(P1, L, multiplicity, False, order_x, ramification, total_offset):
                    # Y = s^p (c + Y1), s = sub.scale u^sub.q, so t = scale s^q
                    # = scale sub.scale^q u^(q sub.q)
                    deeper = sub.embedding
                    series = (LaurentSeries.constant(sub.field, deeper(c)) + sub.series).shifted(p * sub.q)
                    total_scale = sub.field.mul(deeper(scale), _power(sub.scale, q, sub.field))
                    branches.append(_Branch(q * sub.q, total_scale, series, sub.field,
                                            (len(psi) - 1) * sub.conjugates, into.then(deeper)))
    return branches


def _ceil(r: Fraction) -> int:
    return -((-r.numerator) // r.denominator)


def expansions_in_field(F: Curve, x: Symbol, y: Symbol, K: AlgebraicField, into: Embedding,
                        point: Optional[Number], order: int, point_expr: Optional[Expr] = None
                        ) -> list[PuiseuxExpansion]:
    """The expansions of `F(x, y) = 0` (``F`` as ``{i: {k: coefficient of
    x**k y**i}}``, the coefficients rational or in the source of
    ``into``) above the element ``point`` of the number field ``K``
    (``None`` for infinity), ``into`` the embedding of the coefficient
    field of ``F`` into ``K``: the conjugacy classes are taken over ``K``,
    and every expansion carries the embedding of ``K`` into its field.
    ``point_expr`` is the point as an expression (``K.to_expr(point)``
    when it is not given)."""
    n = max(F) if F else 0
    if point is None:
        d = max((k for a in F.values() for k in a), default=0)
        # t^d F(1/t, y)
        P = _poly2_convert({i: {d - k: c for k, c in a.items()} for i, a in F.items()}, into)
    else:
        P = _poly2_translate(_poly2_convert(F, into), point, K)
    order_x = Fraction(order)
    found = _expand(P, K, n, True, order_x, 1, 0)
    result: list[PuiseuxExpansion] = []
    if point is None:
        expression: Expr = oo
    else:
        expression = K.to_expr(point) if point_expr is None else point_expr
    for branch in found:
        series = branch.series.truncated(_ceil(order_x * branch.q)) if branch.series.order is None else branch.series
        point_element = None if point is None else branch.embedding(point)
        result.append(PuiseuxExpansion(expression, point_element, branch.q, branch.field, series,
                                       branch.conjugates, x, y, branch.scale, branch.embedding))
    result.sort(key=_sort_key)
    return result


def _sort_key(b: PuiseuxExpansion) -> tuple[int, int, list[tuple[int, list[Fraction]]]]:
    """A deterministic order of the branches which does not print them
    (printing root objects is slow)."""
    return (b.ramification, b.conjugates,
            [(k, _coefficient_list(c, b.field)) for k, c in sorted(b.series.terms.items())])


def puiseux_expansions(F: Expr, x: Symbol, y: Symbol, point: Expr = Integer(0), order: int = 4,
                       domain: AlgebraicField = RATIONALS) -> list[PuiseuxExpansion]:
    """The Puiseux expansions of the roots `y` of `F(x, y) = 0` above
    `x = ` ``point`` (an algebraic number, or ``oo``), one per conjugacy
    class, with the coefficients of `(x - a)^{k/e}` known for `k/e <`
    ``order``; the coefficients of ``F`` are in ``domain`` (the
    rationals by default; the conjugacy classes are taken over it).

    >>> from sympy_extras.core import symbols, oo, sqrt, Integer
    >>> from sympy_extras.polys.puiseux import puiseux_expansions
    >>> x, y = symbols('x y')
    >>> for b in puiseux_expansions(y**2 - x**3 - 1, x, y, oo, order=3):
    ...     print(b.ramification, b.conjugates, b.as_expr(x))
    2 1 x**(3/2) + 1/(2*x**(3/2))

    A branch stands for its conjugates: the four roots of `y^4 = x` above
    the origin are the conjugates of one branch,

    >>> [b] = puiseux_expansions(y**4 - x, x, y, Integer(0), order=2)
    >>> b.ramification, b.conjugates, b.as_expr(x)
    (4, 1, x**(1/4))

    and the two roots of `y^2 = 2` above `x = 0` of one branch defined
    over `\\mathbb{Q}(\\sqrt 2)`:

    >>> [b] = puiseux_expansions(y**2 - 2 - x, x, y, Integer(0), order=2)
    >>> b.conjugates, b.field, b.as_expr(x)
    (2, QQ<sqrt(2)>, sqrt(2)*x/4 + sqrt(2))
    """
    curve = curve_from_expr(F, x, y, domain)
    if point == oo:
        return expansions_in_field(curve, x, y, domain, Embedding(domain, domain), None, order)
    if isinstance(point, Rational):
        return expansions_in_field(curve, x, y, domain, Embedding(domain, domain),
                                   domain.from_rational(QQ.from_expr(point)), order, point)
    K_point, element = field_of(point)
    if K_point.degree == 1:
        return expansions_in_field(curve, x, y, domain, Embedding(domain, domain), domain.convert(element, K_point),
                                   order, point)
    if domain.degree > 1:
        # the point adjoined to the field of the coefficients: a root of an
        # irreducible factor over it of the minimal polynomial of the point
        m = minimal_polynomial_of(element, K_point)
        factor = factor_over(domain, [domain.from_rational(c) for c in m])[0][0]
        K, into, a = adjoin_root(domain, factor)
        return expansions_in_field(curve, x, y, K, into, a, order)
    return expansions_in_field(curve, x, y, K_point, Embedding(RATIONALS, K_point), element, order, point)


def branches_at_infinity(F: Expr, x: Symbol, y: Symbol, order: int = 4,
                         domain: AlgebraicField = RATIONALS) -> list[PuiseuxExpansion]:
    """The expansions above `x = \\infty`, :func:`puiseux_expansions` with
    ``point=oo``."""
    return puiseux_expansions(F, x, y, oo, order, domain)
