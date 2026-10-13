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
`c\, t^{p/q}`, where `c` is a root of the *characteristic polynomial*
`\chi(c) = \sum_{(i, v) \text{ on the edge}} \operatorname{lc}(a_i)
c^{(i - i_0)/q}`, and the roots continue with the Newton polygon of
`P(s^q, s^p (c + Y_1))/s^{\nu}`, the vertex `(m, 0)` of the new polygon
holding the `m` roots which share the first term (`m` the multiplicity
of `c` in `\chi`). When a single root is left (`m = 1`) its remaining
terms are a power series found by Newton's iteration, since `Y_1 = 0`
is then a simple root of the reduced equation: the ramification is
decided in finitely many steps, the discriminant having finite order.

**Conjugate branches.** Following [Duval]_, a root `c` of an
irreducible factor `\psi` of `\chi` over the current coefficient field
`L` is taken in the extension `L(c)` and its branch stands for the
`[L(c) : L]` conjugate branches, so that the expansions come in classes:
the `n` classical series above `a` are the conjugates of the
:class:`PuiseuxExpansion` objects returned, each counting
``ramification * conjugates`` of them. The number fields are SymPy's
:class:`~sympy.polys.domains.AlgebraicField`; a new root is adjoined
through its minimal polynomial over the rationals (a factor of the norm
of `\psi`), the root identified numerically among the roots of that
factor and the identification checked exactly in the new field.

The expansions carry the point, the ramification index `e`, the field
of their coefficients and the coefficients `c_k` of `t^k`, `x - a = t^e`
(`x = t^{-e}` at infinity), known below a stated order; an element of
`K(x)[y]` evaluated along a branch (:meth:`PuiseuxExpansion.series`) is
a truncated Laurent series in `t` whose valuation is the order of the
function at the place.

References
==========

.. [Walker] R. J. Walker, *Algebraic curves*, Princeton 1950, chapter IV
   (the Newton polygon and Puiseux's theorem).
.. [Duval] D. Duval, *Rational Puiseux expansions*, Compositio
   Mathematica 70 (1989), 119–154.
.. [PoteauxWeimann] A. Poteaux, M. Weimann, *Computing Puiseux series: a
   fast divide and conquer algorithm*, Annales Henri Lebesgue 4 (2021),
   the regularity index and the grouping into conjugacy classes.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Optional

import mpmath

from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.polys.domains import QQ
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.domain import Domain
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polytools import Poly
from sympy.polys.rootoftools import rootof

from sympy_extras._typing import DomainElement, ExprLike, as_expr

__all__ = ['PuiseuxExpansion', 'LaurentSeries', 'puiseux_expansions', 'branches_at_infinity',
           'adjoin_root', 'embed', 'Embedding']


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
        if self.order is None and other.order is None:
            order = None
        else:
            candidates: list[int] = []
            if other.order is not None:
                # the unknown part of other, of order >= other.order, times the
                # known part of self, of valuation >= v1
                candidates.append(other.order + (v1 if v1 is not None else (self.order if self.order is not None else 0)))
            if self.order is not None:
                candidates.append(self.order + (v2 if v2 is not None else (other.order if other.order is not None else 0)))
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
        """The inverse series known below ``t**order`` (relative to the
        valuation: the quotient is known up to ``order`` terms beyond its
        valuation)."""
        v = self.valuation()
        if v is None:
            raise ZeroDivisionError("inverse of a series which is zero to its order")
        unit = self.shifted(-v)
        # unit = c0 (1 - u), u of positive valuation: 1/unit = (1 + u + u^2 + ...)/c0
        c0 = unit.coefficient(0)
        inv0 = self.K.one / c0
        u = LaurentSeries.constant(self.K, self.K.one) - unit.scaled(inv0)
        u = u.truncated(order)
        result = LaurentSeries.constant(self.K, self.K.one)
        term = LaurentSeries.constant(self.K, self.K.one)
        uv = u.valuation()
        if uv is not None:
            steps = (order + uv - 1) // uv + 1
            for _ in range(steps):
                term = (term * u).truncated(order)
                if term.is_zero():
                    break
                result = result + term
        result = result.scaled(inv0).truncated(order)
        return result.shifted(-v)

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


class Embedding:
    """The embedding of a number field ``source`` into a number field
    ``target`` containing it, through the image of the generator."""

    def __init__(self, source: Domain, target: Domain) -> None:
        self.source = source
        self.target = target
        self.generator: Optional[DomainElement] = None
        if isinstance(source, AlgebraicField) and source != target:
            self.generator = target.from_sympy(source.ext)
            self._powers: list[DomainElement] = [target.one]

    def __call__(self, c: DomainElement) -> DomainElement:
        if self.source == self.target:
            return c
        if self.generator is None:
            # the rationals into any field
            return self.target.convert(c, self.source)
        coefficients = list(reversed(c.to_list()))  # low degree first
        while len(self._powers) < len(coefficients):
            self._powers.append(self._powers[-1] * self.generator)
        total = self.target.zero
        for k, a in enumerate(coefficients):
            if a:
                total = total + self.target.convert(a, QQ) * self._powers[k]
        return total


def embed(c: DomainElement, source: Domain, target: Domain) -> DomainElement:
    """``c`` of the field ``source`` as an element of ``target``."""
    return Embedding(source, target)(c)


def _numerical_roots(coefficients: list[complex]) -> list[complex]:
    """The complex roots of the polynomial with the given coefficients
    (highest degree first)."""
    degree = len(coefficients) - 1
    if degree < 1:
        return []
    if degree == 1:
        return [-coefficients[1] / coefficients[0]]
    with mpmath.workdps(60):
        found = mpmath.polyroots([mpmath.mpc(c) for c in coefficients], maxsteps=400, extraprec=200)
    return [complex(r) for r in found]


#: numerical values of the generators of the number fields met (a root
#: object is slow to evaluate: once per field)
_GENERATOR_VALUES: dict[Domain, complex] = {}


def _numerical(K: Domain, c: DomainElement) -> complex:
    """``c`` as a complex number (60 digits)."""
    if not isinstance(K, AlgebraicField):
        return complex(QQ.to_sympy(c))
    theta = _GENERATOR_VALUES.get(K)
    if theta is None:
        theta = complex(as_expr(K.ext).evalf(60))
        _GENERATOR_VALUES[K] = theta
    with mpmath.workdps(60):
        value = mpmath.mpc(0)
        for a in c.to_list():
            value = value * theta + mpmath.mpc(complex(QQ.to_sympy(a)))
    return complex(value)


def _root_indices(factor: Poly, chosen: complex) -> list[int]:
    """The indices of the roots of ``factor`` in SymPy's order (the real
    roots increasing, then the complex ones by real and imaginary
    part), nearest to ``chosen`` first."""
    values = _numerical_roots([complex(c) for c in factor.all_coeffs()])
    real = sorted((r.real for r in values if abs(r.imag) < 1e-40))
    complex_ones = sorted((r for r in values if abs(r.imag) >= 1e-40), key=lambda r: (r.real, r.imag))
    ordered: list[complex] = [complex(r) for r in real] + complex_ones
    return sorted(range(len(ordered)), key=lambda i: abs(ordered[i] - chosen))


def adjoin_root(K: Domain, psi: list[DomainElement]) -> tuple[Domain, Embedding, DomainElement]:
    """The field ``L = K(c)`` for a root ``c`` of the polynomial ``psi``
    (coefficients in ``K``, highest degree first) irreducible over
    ``K``, with the embedding of ``K`` into ``L`` and ``c``.

    The minimal polynomial of ``c`` over the rationals is a factor of the
    norm of ``psi``; the root is identified numerically among the roots
    of the factors and checked exactly.

    >>> from sympy import sqrt, QQ
    >>> from sympy_extras.polys.puiseux import adjoin_root
    >>> K = QQ.algebraic_field(sqrt(2))
    >>> L, into, c = adjoin_root(K, [K.one, K.zero, -K.from_sympy(sqrt(2))])
    >>> L.to_sympy(c*c) == L.to_sympy(into(K.from_sympy(sqrt(2))))
    True
    """
    degree = len(psi) - 1
    if degree == 1:
        return K, Embedding(K, K), -psi[1] / psi[0]
    z = Symbol('z')
    if not isinstance(K, AlgebraicField):
        norm = Poly([K.to_sympy(c) for c in psi], z, domain=QQ)
    else:
        # the norm of psi from K = QQ(theta) to QQ: the resultant in theta
        # with the minimal polynomial of theta, psi written through the
        # coefficients of its coefficients in theta
        theta = Symbol('theta')
        with_theta = Add(*[Add(*[QQ.to_sympy(a) * theta**k for k, a in enumerate(reversed(c.to_list()))]) * z**i
                           for i, c in enumerate(reversed(psi))])
        minimal = Poly(K.mod.to_list(), theta).as_expr()
        norm = Poly(_resultant_in(as_expr(with_theta), as_expr(minimal), z, theta), z, domain=QQ)
    _, factors = norm.factor_list()
    numerical_psi = [_numerical(K, c) for c in psi]
    # the root followed: a real one, positive, when there is one (the
    # conjugates give the same branch up to conjugation)
    candidates = sorted(_numerical_roots(numerical_psi), key=lambda r: (abs(r.imag) > 1e-30, -r.real, abs(r.imag)))
    chosen = candidates[0]
    tried: list[tuple[Poly, int]] = []
    # the factor vanishing at the chosen root first, its nearest root first
    for factor, _ in sorted(factors, key=lambda fm: abs(complex(fm[0].as_expr().subs(z, chosen)))):
        for index in _root_indices(factor, chosen):
            tried.append((factor, index))
    for factor, index in tried:
        # in radicals when SymPy writes them (sqrt(2) rather than a root
        # object), the root object otherwise
        root = as_expr(rootof(factor, index, radicals=True))
        try:
            L = QQ.algebraic_field(K.ext, root) if isinstance(K, AlgebraicField) else QQ.algebraic_field(root)
            into = Embedding(K, L)
            c = L.from_sympy(root)
        except (CoercionFailed, ValueError, NotImplementedError):
            continue
        value = L.zero
        for a in psi:
            value = value * c + into(a)
        if value == L.zero:
            return L, into, c
    raise ValueError("the root could not be identified exactly")


def _resultant_in(p: Expr, q: Expr, z: Symbol, theta: Symbol) -> Expr:
    """The resultant with respect to ``theta`` of ``p`` and ``q``."""
    return as_expr(Poly(p, theta).resultant(Poly(q, theta)))


class PuiseuxExpansion:
    """One branch of the curve `F(x, y) = 0` above ``point`` (an
    expression, or ``S.Infinity``): `y = \\sum_k c_k t^k` with `x -
    \\text{point} = t^e` (`x = t^{-e}` at infinity), the coefficients in
    the number field ``field`` and known for `k <` ``order``; it stands
    for ``conjugates`` branches, conjugate under the automorphisms of
    ``field`` over the field of the point.

    >>> from sympy import symbols
    >>> from sympy_extras.polys.puiseux import puiseux_expansions
    >>> x, y = symbols('x y')
    >>> [b.as_expr(x) for b in puiseux_expansions(y**2 - x**3 - x**2, x, y, 0, order=4)]
    [-x**3/8 + x**2/2 + x, x**3/8 - x**2/2 - x]
    >>> branches = puiseux_expansions(y**3 - x**2, x, y, 0, order=2)
    >>> len(branches), branches[0].ramification, branches[0].as_expr(x)
    (1, 3, x**(2/3))
    """

    def __init__(self, point: Expr, ramification: int, field: Domain, series: LaurentSeries,
                 conjugates: int, x: Symbol, y: Symbol, scale: Optional[DomainElement] = None) -> None:
        self.point = point
        self.ramification = ramification
        #: `x - \\text{point} = \\text{scale} \\cdot t^e` (`x = 1/(\\text{scale} t^e)` at infinity)
        self.scale: DomainElement = field.one if scale is None else scale
        self.field = field
        self.series = series
        self.conjugates = conjugates
        self.x = x
        self.y = y

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
        if self.at_infinity:
            return LaurentSeries.monomial(K, K.one / self.scale, -self.ramification)
        return LaurentSeries(K, {0: K.from_sympy(self.point), self.ramification: self.scale}, None)

    def evaluate(self, p: Poly, order: Optional[int] = None) -> LaurentSeries:
        """The polynomial ``p`` in ``x`` and ``y`` (a :class:`Poly` in
        ``y`` whose coefficients are polynomials in ``x``, or in ``x``
        alone, or a constant) along the branch, as a series in `t`."""
        K = self.field
        xs = self.x_series()
        ys = self.series if order is None else self.series.truncated(order)
        if p.gens == (self.y,) or self.y in p.gens:
            py = Poly(p.as_expr(), self.y)
            total = LaurentSeries.constant(K, K.zero)
            for coefficient in py.all_coeffs():
                cx = Poly(coefficient, self.x)
                total = total * ys + _evaluate_x(cx, xs, K)
            return total
        return _evaluate_x(Poly(p.as_expr(), self.x), xs, K)

    def __repr__(self) -> str:
        return (f"PuiseuxExpansion(point={self.point}, ramification={self.ramification}, "
                f"conjugates={self.conjugates}, series={self.as_expr()})")


def _evaluate_x(p: Poly, xs: LaurentSeries, K: Domain) -> LaurentSeries:
    total = LaurentSeries.constant(K, K.zero)
    for coefficient in p.all_coeffs():
        total = total * xs + LaurentSeries.constant(K, K.from_sympy(as_expr(coefficient)))
    return total


# The Newton polygon.

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
            # the middle point is above or on the segment: drop it
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

    # (c + Y)^i = sum_j binom(i, j) c^(i-j) Y^j
    degree = max(P) if P else 0
    powers_of_c: list[DomainElement] = [K.one]
    for _ in range(degree):
        powers_of_c.append(powers_of_c[-1] * c)
    binomials: list[list[int]] = [[1]]
    for i in range(1, degree + 1):
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
    return {i: {k: v for k, v in a.items() if v != K.zero} for i, a in result.items()}


def _poly2_convert(P: Polynomial2, into: Embedding) -> Polynomial2:
    return {i: {k: into(v) for k, v in a.items()} for i, a in P.items()}


def _poly2_shift(P: Polynomial2, nu: int) -> Polynomial2:
    return {i: {k - nu: v for k, v in a.items()} for i, a in P.items()}


def _poly2_clean(P: Polynomial2, K: Domain) -> Polynomial2:
    return {i: a for i, a in ((i, {k: v for k, v in a.items() if v != K.zero}) for i, a in P.items()) if a}


def _poly2_evaluate(P: Polynomial2, Y: LaurentSeries, K: Domain) -> LaurentSeries:
    degree = max(P) if P else 0
    total = LaurentSeries.constant(K, K.zero)
    for i in range(degree, -1, -1):
        total = total * Y + LaurentSeries(K, P.get(i, {}), None)
    return total


def _poly2_derivative(P: Polynomial2, K: Domain) -> Polynomial2:
    return _poly2_clean({i - 1: {k: K.convert(i, QQ) * v for k, v in a.items()} for i, a in P.items() if i >= 1}, K)


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
        value = _poly2_evaluate(P, approximation, K)
        if value.is_zero():
            root = LaurentSeries(K, root.terms, precision)
            continue
        derivative = _poly2_evaluate(dP, approximation, K)
        correction = value * derivative.inverse(precision)
        root = (approximation - correction).truncated(precision)
    return root.truncated(order)


class _Branch:
    """A branch found by the recursion: the series of `Y` in the final
    variable `u` (`s = scale * u**q`, `s` the variable of the caller),
    its field, and the number of conjugates."""

    def __init__(self, q: int, scale: DomainElement, series: LaurentSeries, field: Domain, conjugates: int) -> None:
        self.q = q
        self.scale = scale
        self.series = series
        self.field = field
        self.conjugates = conjugates


def _characteristic(P: Polynomial2, edge: tuple[tuple[int, int], tuple[int, int]], q: int,
                    K: Domain) -> list[DomainElement]:
    """The characteristic polynomial of the edge, coefficients highest
    degree first in `c`."""
    (i0, v0), (i1, v1) = edge
    p_over_q = Fraction(v0 - v1, i1 - i0)  # gamma = p/q
    degree = (i1 - i0) // q
    coefficients: list[DomainElement] = [K.zero] * (degree + 1)
    for i in range(i0, i1 + 1):
        if (i - i0) % q:
            continue
        a = P.get(i)
        if not a:
            continue
        v = _valuation(a)
        assert v is not None
        # on the edge: v = v0 - gamma (i - i0)
        if Fraction(v) == Fraction(v0) - p_over_q * (i - i0):
            coefficients[degree - (i - i0) // q] = a[v]
    return coefficients


def _factor_over(K: Domain, coefficients: list[DomainElement]) -> list[tuple[list[DomainElement], int]]:
    """The irreducible factors over ``K`` of the polynomial with the
    given coefficients (highest first), with multiplicities."""
    z = Symbol('z')
    p = Poly([K.to_sympy(c) for c in coefficients], z, domain=K)
    _, factors = p.factor_list()
    result: list[tuple[list[DomainElement], int]] = []
    for factor, multiplicity in factors:
        result.append(([K.convert(c, factor.domain) for c in factor.rep.to_list()], multiplicity))
    return result


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
        branches.append(_Branch(1, K.one, LaurentSeries(K, {}, None), K, 1))
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
            l, m = _bezout(q, p)
            c = _power(u, l, L)
            scale = _power(u, m, L)
            P_L = _poly2_convert(P, into)
            substituted = _poly2_substitute(P_L, q, p, c, scale, L)
            nu = min(k for a in substituted.values() for k in a)
            P1 = _poly2_shift(substituted, nu)
            # the ramification of s relative to x, and the exponent of s
            # multiplying Y1 in the series of y
            ramification = ramification_so_far * q
            total_offset = offset * q + p
            if multiplicity == 1:
                # at least the leading term, whatever the order asked
                needed = max(_ceil(order_x * ramification) - total_offset, 1)
                root = _newton_root(P1, L, needed)
                series = (LaurentSeries.constant(L, c) + root).shifted(p)
                branches.append(_Branch(q, scale, series, L, len(psi) - 1))
            else:
                for sub in _expand(P1, L, multiplicity, False, order_x, ramification, total_offset):
                    # Y = s^p (c + Y1), s = sub.scale u^sub.q, so t = scale s^q
                    # = scale sub.scale^q u^(q sub.q)
                    deeper = Embedding(L, sub.field)
                    series = (LaurentSeries.constant(sub.field, deeper(c)) + sub.series).shifted(p * sub.q)
                    total_scale = deeper(scale) * _power(sub.scale, q, sub.field)
                    branches.append(_Branch(q * sub.q, total_scale, series, sub.field,
                                            (len(psi) - 1) * sub.conjugates))
    return branches


def _bezout(q: int, p: int) -> tuple[int, int]:
    """``(l, m)`` with ``l*q - m*p == 1`` (``q > 0``, ``gcd(p, q) == 1``),
    ``m = 0`` when ``q == 1``."""
    if q == 1:
        return 1, 0
    # extended Euclid on (q, p)
    old_r, r = q, p
    old_s, s_ = 1, 0
    old_t, t_ = 0, 1
    while r != 0:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s_ = s_, old_s - quotient * s_
        old_t, t_ = t_, old_t - quotient * t_
    # old_s*q + old_t*p = old_r = +-1
    sign = 1 if old_r == 1 else -1
    return sign * old_s, -sign * old_t


def _power(u: DomainElement, k: int, K: Domain) -> DomainElement:
    if k >= 0:
        return u**k
    return (K.one / u)**(-k)


def _ceil(r: Fraction) -> int:
    return -((-r.numerator) // r.denominator)


def _poly2_from(F: Poly, x: Symbol, y: Symbol, K: Domain) -> Polynomial2:
    """``F`` as a dictionary ``{i: {k: coefficient of x**k y**i}}``."""
    P: Polynomial2 = {}
    for (i, k), c in Poly(F.as_expr(), y, x).terms():
        P.setdefault(i, {})[k] = K.from_sympy(as_expr(c))
    return P


def _poly2_translate(P: Polynomial2, a: DomainElement, K: Domain) -> Polynomial2:
    """`P(x + a, y)`."""
    if a == K.zero:
        return P
    result: Polynomial2 = {}
    for i, coefficients in P.items():
        degree = max(coefficients)
        # Horner: p(x + a) built from the highest coefficient down
        shifted: dict[int, DomainElement] = {}
        for k in range(degree, -1, -1):
            # shifted := shifted * (x + a) + c_k
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


def _coefficient_field(F: Poly, point: Optional[Expr]) -> Domain:
    """The field of the coefficients of ``F`` extended by the point."""
    domain = F.domain
    generators: list[Expr] = []
    if isinstance(domain, AlgebraicField):
        generators.append(as_expr(domain.ext))
    elif not (domain.is_QQ or domain.is_ZZ):
        raise ValueError(f"coefficients in {domain} are not supported")
    if point is not None and not isinstance(point, Rational):
        generators.append(point)
    return QQ.algebraic_field(*generators) if generators else QQ


def puiseux_expansions(F: ExprLike, x: Symbol, y: Symbol, point: ExprLike = 0, order: int = 4
                       ) -> list[PuiseuxExpansion]:
    """The Puiseux expansions of the roots `y` of `F(x, y) = 0` above
    `x = ` ``point`` (an algebraic number, or ``oo``), one per conjugacy
    class, with the coefficients of `(x - a)^{k/e}` known for `k/e <`
    ``order``.

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
    F_ = as_expr(F)
    point_ = as_expr(point)
    FP = Poly(F_, y, x)
    at_infinity = point_ is S.Infinity
    K = _coefficient_field(FP, None if at_infinity else point_)
    n = Poly(F_, y).degree()
    if at_infinity:
        # x = 1/t: F(1/t, y) t^d
        d = Poly(F_, x).degree()
        t = Symbol('t')
        G = Poly((F_.xreplace({x: 1 / t}) * t**d).expand(), y, t)
        P = _poly2_from(Poly(G.as_expr(), y, t), t, y, K)
    else:
        P = _poly2_translate(_poly2_from(FP, x, y, K), K.from_sympy(point_), K)
    order_x = Fraction(order)
    found = _expand(P, K, n, True, order_x, 1, 0)
    result: list[PuiseuxExpansion] = []
    for branch in found:
        series = branch.series.truncated(_ceil(order_x * branch.q)) if branch.series.order is None else branch.series
        result.append(PuiseuxExpansion(point_, branch.q, branch.field, series, branch.conjugates, x, y,
                                       branch.scale))
    result.sort(key=lambda b: (b.ramification, str(b.as_expr())))
    return result


def branches_at_infinity(F: ExprLike, x: Symbol, y: Symbol, order: int = 4) -> list[PuiseuxExpansion]:
    """The expansions above `x = \\infty`, :func:`puiseux_expansions` with
    ``point=oo``."""
    return puiseux_expansions(F, x, y, S.Infinity, order)
