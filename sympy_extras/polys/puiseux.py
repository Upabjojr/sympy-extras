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

**Number fields, exactly.** The fields are SymPy's
:class:`~sympy.polys.domains.AlgebraicField` `\mathbb{Q}(\theta)`, but
built without any numerical identification of roots: a root `u` of
`\psi` over `K = \mathbb{Q}(\theta)` is adjoined by taking a primitive
element `\Theta = \theta + k u` whose minimal polynomial is the
resultant `\operatorname{Res}_\theta(m_K(\theta), k^d \psi((T -
\theta)/k; \theta))` (squarefree for a suitable `k`), the image of
`\theta` in `\mathbb{Q}(\Theta)` being the common root of `m_K(t)` and
`\psi((\Theta - t)/k; t)`, a gcd (:func:`adjoin_root`). Every
expansion carries the :class:`Embedding` of the coefficient field of
the curve into the field of its coefficients.

The expansions carry the point, the ramification index `e`, the field
of their coefficients and the coefficients `c_k` of `t^k`, `x - a =
\lambda t^e` (`x = 1/(\lambda t^e)` at infinity), known below a stated
order; an element of `K(x)[y]` evaluated along a branch
(:meth:`PuiseuxExpansion.evaluate`) is a truncated Laurent series in `t`
whose valuation is the order of the function at the place.

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
from typing import Optional, Union

from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.polys.domains import QQ
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.domain import Domain
from sympy.polys.matrices import DomainMatrix
from sympy.polys.polytools import Poly, resultant
from sympy.polys.rootoftools import rootof

from sympy_extras._typing import DomainElement, ExprLike, as_expr

__all__ = ['PuiseuxExpansion', 'LaurentSeries', 'puiseux_expansions', 'expansions_in_field',
           'branches_at_infinity', 'adjoin_root', 'Embedding', 'field_of', 'minimal_polynomial_of',
           'in_subfield', 'generator', 'degree', 'coefficients_in']


class LaurentSeries:
    """A truncated Laurent series `\\sum_k c_k t^k` with coefficients in
    the domain ``K``: the coefficients of `t^k` are known for `k <
    order` (``order`` ``None`` when the series is exact, a Laurent
    polynomial). Arithmetic keeps track of the order of the result."""

    def __init__(self, K: Domain, terms: dict[int, DomainElement], order: Optional[int]) -> None:
        self.K = K
        self.order = order
        self.terms: dict[int, DomainElement] = {k: c for k, c in terms.items()
                                                if c != K.zero and (order is None or k < order)}

    @classmethod
    def constant(cls, K: Domain, c: DomainElement) -> LaurentSeries:
        return cls(K, {0: c}, None)

    @classmethod
    def monomial(cls, K: Domain, c: DomainElement, k: int) -> LaurentSeries:
        return cls(K, {k: c}, None)

    def valuation(self) -> Optional[int]:
        """The least exponent with a nonzero coefficient, ``None`` when
        no term is known (zero up to the order)."""
        if not self.terms:
            return None
        return min(self.terms)

    def is_zero(self) -> bool:
        return not self.terms

    def coefficient(self, k: int) -> DomainElement:
        return self.terms.get(k, self.K.zero)

    def __add__(self, other: LaurentSeries) -> LaurentSeries:
        order = _min_order(self.order, other.order)
        terms = dict(self.terms)
        for k, c in other.terms.items():
            terms[k] = terms.get(k, self.K.zero) + c
        return LaurentSeries(self.K, terms, order)

    def __neg__(self) -> LaurentSeries:
        return LaurentSeries(self.K, {k: -c for k, c in self.terms.items()}, self.order)

    def __sub__(self, other: LaurentSeries) -> LaurentSeries:
        return self + (-other)

    def scaled(self, c: DomainElement) -> LaurentSeries:
        return LaurentSeries(self.K, {k: c * v for k, v in self.terms.items()}, self.order)

    def shifted(self, k: int) -> LaurentSeries:
        """The series times `t^k`."""
        order = None if self.order is None else self.order + k
        return LaurentSeries(self.K, {j + k: c for j, c in self.terms.items()}, order)

    def __mul__(self, other: LaurentSeries) -> LaurentSeries:
        v1, v2 = self.valuation(), other.valuation()
        order: Optional[int]
        if (self.order is None and not self.terms) or (other.order is None and not other.terms):
            # an exact zero: the product is exactly zero
            return LaurentSeries(self.K, {}, None)
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
        terms: dict[int, DomainElement] = {}
        for j, c in self.terms.items():
            for k, d in other.terms.items():
                if order is not None and j + k >= order:
                    continue
                terms[j + k] = terms.get(j + k, self.K.zero) + c * d
        return LaurentSeries(self.K, terms, order)

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
        inv0 = K.one / unit.coefficient(0)
        b: dict[int, DomainElement] = {0: inv0}
        for k in range(1, order):
            total = K.zero
            for j, a_j in a.items():
                if 0 < j <= k:
                    b_prev = b.get(k - j)
                    if b_prev is not None:
                        total = total + a_j * b_prev
            if total != K.zero:
                b[k] = -inv0 * total
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
        (`\\sum_k c_k t^{-k/\\text{exponent\\_scale}}` with ``reciprocal``)."""
        sign = -1 if reciprocal else 1
        return as_expr(Add(*[self.K.to_sympy(c) * t**(sign * Rational(k, exponent_scale))
                             for k, c in sorted(self.terms.items())]))

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

    def __init__(self, source: Domain, target: Domain, image: Optional[DomainElement] = None) -> None:
        self.source = source
        self.target = target
        self.image: Optional[DomainElement] = None
        if isinstance(source, AlgebraicField) and source != target:
            if image is None:
                raise ValueError("the image of the generator is needed")
            self.image = image
            self._powers: list[DomainElement] = [target.one]

    def __call__(self, c: DomainElement) -> DomainElement:
        if self.source == self.target:
            return c
        if self.image is None:
            # the rationals into any field
            return self.target.convert(c, self.source)
        coefficients = list(reversed(c.to_list()))  # low degree first
        while len(self._powers) < len(coefficients):
            self._powers.append(self._powers[-1] * self.image)
        total = self.target.zero
        for k, a in enumerate(coefficients):
            if a:
                total = total + self.target.convert(a, QQ) * self._powers[k]
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


def generator(K: Domain) -> DomainElement:
    """The generator of the number field ``K`` (one for the rationals)."""
    if isinstance(K, AlgebraicField):
        return K([QQ.one, QQ.zero])
    return K.one


def degree(K: Domain) -> int:
    """The degree of the number field ``K`` over the rationals."""
    return int(K.mod.degree()) if isinstance(K, AlgebraicField) else 1


def _coefficient_list(c: DomainElement, K: Domain) -> list[DomainElement]:
    """The rational coordinates of ``c`` in the powers of the generator
    of ``K``, low degree first, padded to the degree."""
    if not isinstance(K, AlgebraicField):
        return [c]
    coefficients = list(reversed(c.to_list()))
    return coefficients + [QQ.zero] * (degree(K) - len(coefficients))


def _field_from_polynomial(N: Poly) -> Domain:
    """`\\mathbb{Q}(\\Theta)` for `\\Theta` a root of the irreducible ``N``,
    its largest real root when it has one (written in radicals when
    SymPy can)."""
    real = N.count_roots()
    root = as_expr(rootof(N, real - 1 if real > 0 else 0, radicals=True))
    return QQ.algebraic_field(root)


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


def _as_polynomial_in(c: DomainElement, theta: Symbol) -> Expr:
    """An element of a number field as a polynomial in the symbol
    ``theta`` standing for the generator."""
    return as_expr(Add(*[QQ.to_sympy(a) * theta**k for k, a in enumerate(reversed(c.to_list()))]))


def adjoin_root(K: Domain, psi: list[DomainElement]) -> tuple[Domain, Embedding, DomainElement]:
    """The field ``L = K(u)`` for a root ``u`` of the polynomial ``psi``
    (coefficients in ``K``, highest degree first) irreducible over
    ``K``, with the embedding of ``K`` into ``L`` and ``u``; exact: no
    root is evaluated numerically.

    >>> from sympy import sqrt, QQ
    >>> from sympy_extras.polys.puiseux import adjoin_root
    >>> K = QQ.algebraic_field(sqrt(2))
    >>> L, into, u = adjoin_root(K, [K.one, K.zero, -K.from_sympy(sqrt(2))])
    >>> L.to_sympy(u*u) == L.to_sympy(into(K.from_sympy(sqrt(2))))
    True
    """
    d = len(psi) - 1
    if d == 1:
        return K, Embedding(K, K), -psi[1] / psi[0]
    T = Symbol('T')
    if not isinstance(K, AlgebraicField):
        N = Poly([QQ.to_sympy(c) for c in psi], T, domain=QQ)
        L = _field_from_polynomial(N)
        return L, Embedding(QQ, L), generator(L)
    theta = Symbol('theta')
    z = Symbol('z')
    m_K = Poly(K.mod.to_list(), theta).as_expr()
    psi_expr = as_expr(Add(*[_as_polynomial_in(c, theta) * z**i for i, c in enumerate(reversed(psi))]))
    k = 0
    N = Poly(0, T)
    P: Expr = S.Zero
    for k in range(0, 2 * d * degree(K) + 3):
        # Theta = theta + k u: P(T, theta) = k^d psi((T - theta)/k) for k > 0
        if k == 0:
            P = as_expr(psi_expr.xreplace({z: T}))
        else:
            P = as_expr((psi_expr.xreplace({z: (T - theta) / k}) * k**d).expand())
        N = Poly(resultant(P, m_K, theta), T, domain=QQ)
        if N.degree() == d * degree(K) and N.gcd(N.diff(T)).degree() == 0:
            break
    else:
        raise ValueError("no primitive element found")
    L = _field_from_polynomial(N)
    Theta = generator(L)
    # the image t of theta: the common root of m_K(t) and P(Theta, t)
    t = Symbol('t')
    m_K_L = Poly.from_list([L.convert(c, QQ) for c in Poly(m_K, theta).all_coeffs()], t, domain=L)
    P_coefficients: dict[int, DomainElement] = {}
    for (power_T, power_theta), c in Poly(P, T, theta).terms():
        value = L.convert(c, QQ) * Theta**power_T
        P_coefficients[power_theta] = P_coefficients.get(power_theta, L.zero) + value
    highest = max(P_coefficients)
    P_L = Poly.from_list([P_coefficients.get(i, L.zero) for i in range(highest, -1, -1)], t, domain=L)
    common = m_K_L.gcd(P_L)
    if common.degree() != 1:
        raise ValueError("the image of the generator is not unique")
    a, b = coefficients_in(common, L)
    image = -b / a
    u = Theta if k == 0 else (Theta - image) / L.convert(k, QQ)
    into = Embedding(K, L, image)
    value = L.zero
    for c in psi:
        value = value * u + into(c)
    if value != L.zero:
        raise ValueError("the adjoined root does not satisfy its polynomial")
    return L, into, u


def field_of(point: Expr) -> tuple[Domain, DomainElement]:
    """The number field generated by an algebraic number given as an
    expression, and the number as its element."""
    if isinstance(point, Rational):
        return QQ, QQ.convert(point)
    K = QQ.algebraic_field(point)
    if K.ext == point:
        return K, generator(K)
    return K, K.from_sympy(point)


def minimal_polynomial_of(c: DomainElement, K: Domain, T: Symbol) -> Poly:
    """The minimal polynomial over the rationals of an element of the
    number field ``K``, exactly (the factor of its characteristic
    polynomial vanishing at it)."""
    if not isinstance(K, AlgebraicField):
        return Poly(T - QQ.to_sympy(c), T, domain=QQ)
    theta = Symbol('theta')
    m_K = Poly(K.mod.to_list(), theta).as_expr()
    characteristic = Poly(resultant(T - _as_polynomial_in(c, theta), m_K, theta), T, domain=QQ)
    for factor, _ in characteristic.factor_list()[1]:
        value = K.zero
        for a in factor.all_coeffs():
            value = value * c + K.convert(a, QQ)
        if value == K.zero:
            return factor
    raise ValueError("no factor vanishes")


def in_subfield(c: DomainElement, L: Domain, into: Embedding) -> Optional[DomainElement]:
    """``c`` of ``L`` as an element of the subfield ``into.source``
    (``into`` its embedding into ``L``), ``None`` when it is not in it."""
    K = into.source
    if K == L:
        return c
    if not isinstance(K, AlgebraicField):
        coefficients = _coefficient_list(c, L)
        if any(a != QQ.zero for a in coefficients[1:]):
            return None
        return coefficients[0]
    d = degree(K)
    columns: list[list[DomainElement]] = []
    power = L.one
    for _ in range(d):
        columns.append(_coefficient_list(power, L))
        power = power * into(generator(K))
    target = _coefficient_list(c, L)
    D = len(target)
    augmented = DomainMatrix([[columns[j][i] for j in range(d)] + [target[i]] for i in range(D)], (D, d + 1), QQ)
    reduced, pivots = augmented.rref()
    if d in pivots:
        return None
    rows = reduced.to_list()
    solution = [QQ.zero] * d
    for r, p in enumerate(pivots):
        solution[p] = rows[r][d]
    return K(list(reversed(solution)))


# The expansions

class PuiseuxExpansion:
    """One branch of the curve `F(x, y) = 0` above ``point`` (an
    expression, or ``S.Infinity``): `y = \\sum_k c_k t^k` with `x -
    \\text{point} = \\text{scale} \\cdot t^e` (`x = 1/(\\text{scale}\\, t^e)`
    at infinity), the coefficients in the number field ``field`` and
    known for `k <` ``order``; it stands for ``conjugates`` branches,
    conjugate under the automorphisms of ``field`` over the field of
    the point. ``embedding`` maps the coefficient field of the curve
    (extended by the point) into ``field``.

    >>> from sympy import symbols
    >>> from sympy_extras.polys.puiseux import puiseux_expansions
    >>> x, y = symbols('x y')
    >>> [b.as_expr(x) for b in puiseux_expansions(y**2 - x**3 - x**2, x, y, 0, order=4)]
    [x**3/8 - x**2/2 - x, -x**3/8 + x**2/2 + x]
    >>> branches = puiseux_expansions(y**3 - x**2, x, y, 0, order=2)
    >>> len(branches), branches[0].ramification, branches[0].as_expr(x)
    (1, 3, x**(2/3))
    """

    def __init__(self, point: Expr, point_element: Optional[DomainElement], ramification: int, field: Domain,
                 series: LaurentSeries, conjugates: int, x: Symbol, y: Symbol, scale: DomainElement,
                 embedding: Embedding) -> None:
        self.point = point
        #: the point as an element of ``field`` (``None`` at infinity)
        self.point_element = point_element
        self.ramification = ramification
        self.field = field
        self.series = series
        self.conjugates = conjugates
        self.x = x
        self.y = y
        self.scale = scale
        self.embedding = embedding

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
        K = self.field
        if self.point_element is None:
            return LaurentSeries.monomial(K, K.one / self.scale, -self.ramification)
        return LaurentSeries(K, {0: self.point_element, self.ramification: self.scale}, None)

    def evaluate(self, p: Poly, order: Optional[int] = None) -> LaurentSeries:
        """The polynomial ``p`` in ``x`` and ``y`` (a :class:`Poly` whose
        coefficients are rational, or in the coefficient field of the
        curve) along the branch, as a series in `t`."""
        K = self.field
        xs = self.x_series()
        ys = self.series if order is None else self.series.truncated(order)
        convert = self._converter(p)
        if self.y in p.gens:
            total = LaurentSeries.constant(K, K.zero)
            py = Poly(p.as_expr(), self.y)
            for coefficient in py.all_coeffs():
                total = total * ys + _evaluate_x(Poly(coefficient, self.x), xs, K, convert)
            return total
        return _evaluate_x(Poly(p.as_expr(), self.x), xs, K, convert)

    def _converter(self, p: Poly) -> Embedding:
        domain = p.domain
        if isinstance(domain, AlgebraicField):
            if domain == self.embedding.source:
                return self.embedding
            raise ValueError(f"coefficients in {domain} cannot be evaluated along a branch over {self.field}")
        return Embedding(QQ, self.field)

    def __repr__(self) -> str:
        return (f"PuiseuxExpansion(point={self.point}, ramification={self.ramification}, "
                f"conjugates={self.conjugates}, series={self.as_expr()})")


def _evaluate_x(p: Poly, xs: LaurentSeries, K: Domain, convert: Embedding) -> LaurentSeries:
    total = LaurentSeries.constant(K, K.zero)
    source = convert.source
    for coefficient in p.all_coeffs():
        c = source.from_sympy(as_expr(coefficient)) if isinstance(source, AlgebraicField) \
            else QQ.from_sympy(as_expr(coefficient))
        total = total * xs + LaurentSeries.constant(K, convert(c))
    return total


# The Newton polygon

def _valuation(a: dict[int, DomainElement]) -> Optional[int]:
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


Polynomial2 = dict[int, dict[int, DomainElement]]  # Y-degree -> {t-exponent: coefficient}


def _poly2_substitute(P: Polynomial2, q: int, p: int, c: DomainElement, scale: DomainElement,
                      K: Domain) -> Polynomial2:
    """`P(\\lambda s^q, s^p (c + Y))`, the exponents of `s` as they come
    (possibly negative), `\\lambda` the ``scale``."""
    result: Polynomial2 = {}
    scale_powers: dict[int, DomainElement] = {}

    def scaled(k: int) -> DomainElement:
        if k not in scale_powers:
            scale_powers[k] = scale**k if k >= 0 else (K.one / scale)**(-k)
        return scale_powers[k]

    degree_Y = max(P) if P else 0
    powers_of_c: list[DomainElement] = [K.one]
    for _ in range(degree_Y):
        powers_of_c.append(powers_of_c[-1] * c)
    binomials: list[list[int]] = [[1]]
    for i in range(1, degree_Y + 1):
        row = [1]
        for j in range(1, i):
            row.append(binomials[i - 1][j - 1] + binomials[i - 1][j])
        row.append(1)
        binomials.append(row)
    for i, a in P.items():
        for j in range(i + 1):
            factor = K.convert(binomials[i][j], QQ) * powers_of_c[i - j]
            if factor == K.zero:
                continue
            target = result.setdefault(j, {})
            for k, coefficient in a.items():
                exponent = q * k + p * i
                target[exponent] = target.get(exponent, K.zero) + factor * coefficient * scaled(k)
    return _poly2_clean(result, K)


def _poly2_convert(P: Polynomial2, into: Embedding) -> Polynomial2:
    return {i: {k: into(v) for k, v in a.items()} for i, a in P.items()}


def _poly2_shift(P: Polynomial2, nu: int) -> Polynomial2:
    return {i: {k - nu: v for k, v in a.items()} for i, a in P.items()}


def _poly2_clean(P: Polynomial2, K: Domain) -> Polynomial2:
    return {i: a for i, a in ((i, {k: v for k, v in a.items() if v != K.zero}) for i, a in P.items()) if a}


def _poly2_evaluate(P: Polynomial2, Y: LaurentSeries, K: Domain, order: Optional[int] = None) -> LaurentSeries:
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


def _poly2_derivative(P: Polynomial2, K: Domain) -> Polynomial2:
    return _poly2_clean({i - 1: {k: K.convert(i, QQ) * v for k, v in a.items()} for i, a in P.items() if i >= 1}, K)


def _poly2_translate(P: Polynomial2, a: DomainElement, K: Domain) -> Polynomial2:
    """`P(x + a, y)`."""
    if a == K.zero:
        return P
    result: Polynomial2 = {}
    for i, coefficients in P.items():
        highest = max(coefficients)
        shifted: dict[int, DomainElement] = {}
        for k in range(highest, -1, -1):
            new: dict[int, DomainElement] = {}
            for j, c in shifted.items():
                new[j + 1] = new.get(j + 1, K.zero) + c
                new[j] = new.get(j, K.zero) + c * a
            c_k = coefficients.get(k)
            if c_k is not None:
                new[0] = new.get(0, K.zero) + c_k
            shifted = new
        result[i] = shifted
    return _poly2_clean(result, K)


def _newton_root(P: Polynomial2, K: Domain, order: int) -> LaurentSeries:
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

    def __init__(self, q: int, scale: DomainElement, series: LaurentSeries, field: Domain, conjugates: int,
                 embedding: Embedding) -> None:
        self.q = q
        self.scale = scale
        self.series = series
        self.field = field
        self.conjugates = conjugates
        self.embedding = embedding


def _characteristic(P: Polynomial2, edge: tuple[tuple[int, int], tuple[int, int]], q: int,
                    K: Domain) -> list[DomainElement]:
    """The characteristic polynomial of the edge in `u = c^q`,
    coefficients highest degree first."""
    (i0, v0), (i1, v1) = edge
    p_over_q = Fraction(v0 - v1, i1 - i0)
    degree_u = (i1 - i0) // q
    coefficients: list[DomainElement] = [K.zero] * (degree_u + 1)
    for i in range(i0, i1 + 1):
        if (i - i0) % q:
            continue
        a = P.get(i)
        if not a:
            continue
        v = _valuation(a)
        assert v is not None
        if Fraction(v) == Fraction(v0) - p_over_q * (i - i0):
            coefficients[degree_u - (i - i0) // q] = a[v]
    return coefficients


def _factor_over(K: Domain, coefficients: list[DomainElement]) -> list[tuple[list[DomainElement], int]]:
    """The irreducible factors over ``K`` of the polynomial with the
    given coefficients (highest first), with multiplicities."""
    z = Symbol('z')
    p = Poly.from_list(coefficients, z, domain=K)
    _, factors = p.factor_list()
    result: list[tuple[list[DomainElement], int]] = []
    for factor, multiplicity in factors:
        result.append((coefficients_in(factor, K), multiplicity))
    return result


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


def _root_and_scale(u: DomainElement, q: int, p: int, L: Domain) -> tuple[DomainElement, DomainElement]:
    """``(c, scale)`` with `c^q = u\\, \\text{scale}^p`: a `q`-th root of `u` in
    `L` with no scaling when there is one (`u = 1`, a square in `L`),
    else Duval's `c = u^l`, `\\lambda = u^m` with `l q - m p = 1`, `0 \\le m
    < q`."""
    if q > 1:
        z = Symbol('z')
        for factor, _ in Poly.from_list([L.one] + [L.zero] * (q - 1) + [-u], z, domain=L).factor_list()[1]:
            if factor.degree() == 1:
                a, b = coefficients_in(factor, L)
                return -b / a, L.one
    l, m = _bezout(q, p)
    # the other solutions: (l + p t, m + q t)
    while m < 0:
        l, m = l + p, m + q
    while m >= q:
        l, m = l - p, m - q
    return _power(u, l, L), _power(u, m, L)


def _power(u: DomainElement, k: int, K: Domain) -> DomainElement:
    if k >= 0:
        return u**k
    return (K.one / u)**(-k)


def _expand(P: Polynomial2, K: Domain, follow: int, first: bool, order_x: Fraction,
            ramification_so_far: int, offset: int) -> list[_Branch]:
    """The branches of `P(s, Y) = 0` through the vertex `(follow, 0)` (all
    the roots when ``first``), each known below `s^{N}` with `N` the
    order ``order_x`` in units of `x` times the ramification of `s` and
    of the branch, less the ``offset`` by which the caller shifts the
    series (`y = s^{\\text{offset}} (\\ldots + Y)`)."""
    P = _poly2_clean(P, K)
    branches: list[_Branch] = []
    if not first and 0 not in P:
        # Y = 0 is an exact root
        branches.append(_Branch(1, K.one, LaurentSeries(K, {}, None), K, 1, Embedding(K, K)))
        P = {i - 1: a for i, a in P.items() if i >= 1}
        follow -= 1
        if follow == 0:
            return branches
    indices = sorted(i for i in P if first or i <= follow)
    points: list[tuple[int, int]] = []
    for i in indices:
        v = _valuation(P[i])
        assert v is not None
        points.append((i, v))
    hull = _lower_hull(points)
    for (i0, v0), (i1, v1) in zip(hull, hull[1:]):
        slope = Fraction(v0 - v1, i1 - i0)  # gamma
        if not first and slope <= 0:
            continue
        p, q = slope.numerator, slope.denominator
        chi = _characteristic(P, ((i0, v0), (i1, v1)), q, K)
        for psi, multiplicity in _factor_over(K, chi):
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
                    total_scale = deeper(scale) * _power(sub.scale, q, sub.field)
                    branches.append(_Branch(q * sub.q, total_scale, series, sub.field,
                                            (len(psi) - 1) * sub.conjugates, into.then(deeper)))
    return branches


def _ceil(r: Fraction) -> int:
    return -((-r.numerator) // r.denominator)


def _poly2_from(F: Poly, x: Symbol, y: Symbol, K: Domain, convert: Embedding) -> Polynomial2:
    """``F`` as a dictionary ``{i: {k: coefficient of x**k y**i}}`` over
    ``K`` (``convert`` from the domain of ``F``)."""
    P: Polynomial2 = {}
    source = convert.source
    for (i, k), c in Poly(F.as_expr(), y, x).terms():
        value = source.from_sympy(as_expr(c)) if isinstance(source, AlgebraicField) else QQ.from_sympy(as_expr(c))
        P.setdefault(i, {})[k] = convert(value)
    return P


def expansions_in_field(F: Poly, x: Symbol, y: Symbol, K: Domain, into: Embedding,
                        point: Optional[DomainElement], order: int) -> list[PuiseuxExpansion]:
    """The expansions of `F(x, y) = 0` (``F`` with rational coefficients,
    or in the source of ``into``) above the element ``point`` of the
    number field ``K`` (``None`` for infinity), ``into`` the embedding of
    the coefficient field of ``F`` into ``K``: the conjugacy classes are
    taken over ``K``, and every expansion carries the embedding of
    ``K`` into its field."""
    F_ = as_expr(F.as_expr())
    n = Poly(F_, y).degree()
    at_infinity = point is None
    if at_infinity:
        d = Poly(F_, x).degree()
        t = Symbol('t')
        G = Poly((F_.xreplace({x: 1 / t}) * t**d).expand(), y, t, domain=F.domain)
        P = _poly2_from(G, t, y, K, into)
    else:
        P = _poly2_translate(_poly2_from(Poly(F_, y, x, domain=F.domain), x, y, K, into), point, K)
    order_x = Fraction(order)
    found = _expand(P, K, n, True, order_x, 1, 0)
    result: list[PuiseuxExpansion] = []
    point_expr = S.Infinity if point is None else as_expr(K.to_sympy(point))
    for branch in found:
        series = branch.series.truncated(_ceil(order_x * branch.q)) if branch.series.order is None else branch.series
        point_element = None if point is None else branch.embedding(point)
        result.append(PuiseuxExpansion(point_expr, point_element, branch.q, branch.field, series,
                                       branch.conjugates, x, y, branch.scale, branch.embedding))
    result.sort(key=_sort_key)
    return result


def _sort_key(b: PuiseuxExpansion) -> tuple[int, int, list[tuple[int, list[DomainElement]]]]:
    """A deterministic order of the branches which does not print them
    (printing root objects is slow)."""
    return (b.ramification, b.conjugates,
            [(k, _coefficient_list(c, b.field)) for k, c in sorted(b.series.terms.items())])


def puiseux_expansions(F: Union[ExprLike, Poly], x: Symbol, y: Symbol, point: ExprLike = 0, order: int = 4
                       ) -> list[PuiseuxExpansion]:
    """The Puiseux expansions of the roots `y` of `F(x, y) = 0` above
    `x = ` ``point`` (an algebraic number, or ``oo``), one per conjugacy
    class, with the coefficients of `(x - a)^{k/e}` known for `k/e <`
    ``order``; a :class:`~sympy.polys.polytools.Poly` keeps its
    coefficient domain (a number field over which the conjugacy classes
    are then taken).

    >>> from sympy import symbols, oo, sqrt
    >>> from sympy_extras.polys.puiseux import puiseux_expansions
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
