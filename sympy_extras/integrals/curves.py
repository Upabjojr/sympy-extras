r"""Trager's algorithm for the integration of algebraic functions on an
arbitrary plane curve: integrands rational in `x` and in any number of
nested radicals.

The integrand is written in the function field `E = \mathbb{Q}(x)[y]/(F)`
of a primitive element `y = \sum_i k_i g_i` of the radicals `g_i` it
contains (small integers `k_i`, `F` the minimal polynomial of `y`, of
degree the product of the indices of the radicals when they are
independent), and integrated by the three steps of Trager's algorithm
[Trager]_ as presented in [Bronstein]_, section 2:

0. **The change of variable** `x = N + 1/z`, `N` an integer which is
   neither a pole of the integrand nor a root of the discriminant of the
   curve, so that the places above `z = \infty` are unramified and the
   integrand vanishes there to the second order. The curve becomes
   `\tilde F(z, y) = z^{\deg_x F} F(N + 1/z, y)` and the integrand `g(z) =
   -f(N + 1/z)/z^2`.
1. **The Hermite reduction** in an integral basis `w` of
   :class:`~sympy_extras.polys.functionfields.FunctionField` normal at
   infinity: for `g = \sum_i A_i w_i / (U V^m)` with `V` the factor of
   highest multiplicity `m \ge 2` of the squarefree factorization of the
   denominator, the vector `b` of the coordinates of `B = \sum_i b_i w_i`
   with `g - (B/V^{m-1})'` of smaller order at the roots of `V` solves
   the congruence `b\,(U V e^{-1} M - (m - 1) U V' I) \equiv a \pmod V`,
   `e W' = M W` the derivative of the basis [ChenDuKauers]_ (2.1);
   Trager proved that it has a unique solution modulo `V` when `w` is an
   integral basis. What is left, `h = \sum_i A_i w_i / D`, has `D`
   squarefree: simple poles at the finite places. Then `\int g\,dz` is
   elementary only if `h z^2` is integral at infinity, `\max_i (\deg A_i
   - \deg D + k_i) \le -2` with `k_i` the pole orders of the basis at
   infinity (Trager's theorem, the basis being normal there); otherwise
   the integral is proved **not elementary**.
2. **The residues.** At a place `P` above a root `\beta` of `D`, found
   by the Puiseux expansions of :mod:`sympy_extras.polys.puiseux` above
   the roots of the irreducible factors of `D` (a representative of
   every class of conjugate places), the residue of `h\,dz` is the
   coefficient of `t^{-1}` in `h\, dz/dt`, `z = \beta + \lambda t^{e}`.
   The residues generate a number field `K`; the factors of `D` are
   taken irreducible over `K`, so that the places above one root of
   each stand for their conjugates over `K`; a basis `\beta_j` of the
   span of the residues over the rationals gives the divisors `\delta_j
   = \sum_P c_{jP} P` (`c_{jP}` the coordinates of the residues, made
   integral by a common denominator `m_j`), each of degree zero and
   defined over `K`.
3. **The logarithmic part.** For `n = 1, 2, \ldots` up to a bound, a
   function `u_j` with divisor `n \delta_j` is sought as `u_j =
   N/\prod_G G^{k_G}`, `N = \sum_i N_i w_i` with polynomial `N_i` over
   `K` of degree at most `\deg \prod G^{k_G} - k_i` (no pole at infinity,
   the basis being normal there), the vanishing orders `n c_{jP} + k_G
   e_P` at the places above the roots of the factors `G` of `D` imposed
   on the Puiseux expansions as linear equations over the rationals; a
   nonzero solution has exactly the divisor `n \delta_j`, the degrees
   being zero. Then `\int h = \sum_j \beta_j/(n_j m_j) \log u_j` exactly
   when `h - \sum_j \beta_j/(n_j m_j)\, u_j'/u_j = 0` in the field; a
   nonzero difference proves the integral **not elementary** (Liouville's
   theorem in Trager's form), and a divisor not found torsion within the
   bound leaves the question undecided (Trager settles it by reducing
   the curve modulo primes; the bound is Mazur's for a curve of genus
   one over the rationals and a heuristic beyond).

The minimal polynomial `F` is read off the powers of `y` in the monomial
basis `\prod g_i^{e_i}` of the algebra of the radicals (the relation of
`y^n` to the lower powers), and the radicals are independent exactly
when it is irreducible over `\mathbb{Q}(x)` (tested by specialising `x`
to a few integers, by the bivariate factorisation when none decides).
The answer is written back in `x` and in the original radicals (each
power of the primitive element expressed in the monomials of the
radicals), and the dispatcher checks it by differentiation.

The module works on the objects of the core and imports no SymPy: the
function fields, Puiseux expansions and number fields are those of
:mod:`sympy_extras.polys.functionfields` and
:mod:`sympy_extras.polys.puiseux` (the rationals as the field
:data:`~sympy_extras.polys.puiseux.RATIONALS` of degree one, so that
every coefficient is a list of rationals), the expressions are cancelled
by :func:`sympy_extras.polys.polytools.cancel`.
:mod:`sympy_extras.integrals.curves_sympy` is its face on SymPy objects.
The generator of the number field of the residues is written as the
largest real root of its minimal polynomial (an
:class:`~sympy_extras.polys.cad.indexedroot.IndexedRoot`) or in
radicals; a field without real generator and not in radicals has no
expression here (:meth:`CurveIntegral.elementary_part` takes one, which
the face gives as SymPy's ``CRootOf``).

Examples
========

>>> from sympy_extras.core import symbols
>>> from sympy_extras.functions import sqrt
>>> from sympy_extras.integrals.curves import curve_antiderivative
>>> x = symbols('x')
>>> curve_antiderivative(1/(sqrt(x + 1) + sqrt(x - 1)), x)
-x*sqrt(x - 1)/3 + x*sqrt(x + 1)/3 + sqrt(x - 1)/3 + sqrt(x + 1)/3

References
==========

.. [Trager] B. M. Trager, *Integration of algebraic functions*, PhD
   thesis, MIT, 1984.
.. [Bronstein] M. Bronstein, *Symbolic integration tutorial*, ISSAC 1998,
   section 2; *The lazy Hermite reduction*, INRIA report 3562, 1998.
.. [ChenDuKauers] S. Chen, L. Du, M. Kauers, *Lazy Hermite reduction and
   creative telescoping for algebraic functions*, ISSAC 2021 (the
   congruence (2.1) of the Hermite reduction step).
.. [Kauers] M. Kauers, *Integration of algebraic functions: a simple
   heuristic for finding the logarithmic part*, ISSAC 2008 (the worked
   examples of the tests).
"""
from __future__ import annotations

import random
from fractions import Fraction
from math import gcd, lcm
from typing import Final, Optional

from sympy_extras.core.add import add
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.mul import mul
from sympy_extras.core.numbers import Integer, Rational
from sympy_extras.core.power import Pow, pow
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.elementary.exponential import log
from sympy_extras.polys import densearith, densebasic, densetools
from sympy_extras.polys.domains.algebraicfield import AlgebraicField
from sympy_extras.polys.domains.fractionfield import FracElement, FractionField
from sympy_extras.polys.domains.polynomialring import PolynomialRing
from sympy_extras.polys.domains.rationalfield import QQ
from sympy_extras.polys.euclidtools import dup_discriminant, dup_invert
from sympy_extras.polys.functionfields import Element, FunctionField, _adjugate, _determinant, _inverse
from sympy_extras.polys.polytools import cancel, degree as poly_degree, factor_list
from sympy_extras.polys.polyutils import dict_from_expr
from sympy_extras.polys.puiseux import (RATIONALS, Curve, Embedding, LaurentSeries, Number, PuiseuxExpansion,
                                        adjoin_root, degree, expansions_in_field, factor_over, generator,
                                        in_subfield, minimal_polynomial_of)
from sympy_extras.polys.sqfreetools import dup_sqf_list
from sympy_extras.simplify.radsimp import fraction

__all__ = ['curve_antiderivative', 'curve_reduce', 'is_nonelementary_on_curve', 'integrate_on_curve',
           'AlgebraicIntegrand', 'CurveIntegral', 'TORSION_BOUND', 'RESIDUE_FIELD_DEGREE']

#: the multiples of a residue divisor tried before giving up (Mazur's
#: bound for a curve of genus one over the rationals)
TORSION_BOUND: Final = 12

#: the degree beyond which the field of the residues is not built
RESIDUE_FIELD_DEGREE: Final = 24

#: the exponents of a monomial in the radicals
Monomial = tuple[int, ...]

#: a polynomial over a number field, highest degree first
KPoly = list[Number]

#: an element of `K(x)`
Fraction_ = FracElement[Number]

#: an element of the algebra of the radicals by its monomials
AlgebraElement = dict[Monomial, Fraction_]

#: a factor of the denominator by its coefficients (a dictionary key)
FactorKey = tuple[tuple[Fraction, ...], ...]


# The algebraic functions of the integrand and their primitive element

class _Generator:
    """A radical `g = \\text{base}^{1/q}` of the integrand, the base a
    rational function of `x` and of the earlier generators (replaced by
    their symbols)."""

    __slots__ = ('symbol', 'base', 'q', 'original')

    def __init__(self, symbol: Dummy, base: Expr, q: int, original: Expr) -> None:
        self.symbol: Final[Dummy] = symbol
        self.base: Final[Expr] = base
        self.q: Final[int] = q
        self.original: Final[Expr] = original


class _Algebra:
    """The algebra `\\mathbb{Q}(x)[g_1, \\ldots, g_m]/(g_i^{q_i} -
    \\text{base}_i)` with the monomial basis `\\prod g_i^{e_i}`, `e_i <
    q_i`; the field of the integrand when the radicals are
    independent."""

    def __init__(self, generators: list[_Generator], x: Symbol) -> None:
        self.generators: Final[list[_Generator]] = generators
        self.x: Final[Symbol] = x
        self.R: Final[PolynomialRing[Number]] = PolynomialRing(RATIONALS, x.name)
        self.Kx: Final[FractionField[Number]] = FractionField(self.R)
        self.names: Final[list[str]] = [g.symbol.name for g in generators]
        dimension = 1
        for g in generators:
            dimension *= g.q
        self.dimension: Final[int] = dimension
        bases: list[AlgebraElement] = []
        self.bases: Final[list[AlgebraElement]] = bases
        for g in generators:
            # the base in the earlier generators only
            bases.append(self.from_expr(g.base))
        monomials: list[Monomial] = []
        self._enumerate((), 0, monomials)
        self.monomials: Final[list[Monomial]] = monomials
        self.index: Final[dict[Monomial, int]] = {m: i for i, m in enumerate(monomials)}

    def _enumerate(self, prefix: Monomial, k: int, monomials: list[Monomial]) -> None:
        if k == len(self.generators):
            monomials.append(prefix)
            return
        for e in range(self.generators[k].q):
            self._enumerate(prefix + (e,), k + 1, monomials)

    def from_expr(self, expr: Expr) -> AlgebraElement:
        """A polynomial in the generator symbols with coefficients
        rational in `x` as an element."""
        if not self.names:
            return self.add({}, {(): self.Kx.from_expr(expr)})
        result: AlgebraElement = {}
        for monomial, coefficient in dict_from_expr(expr, self.names, self.Kx).items():
            result = self.add(result, self.reduce({monomial: coefficient}))
        return result

    def add(self, a: AlgebraElement, b: AlgebraElement) -> AlgebraElement:
        Kx = self.Kx
        result = dict(a)
        for monomial, c in b.items():
            total = Kx.add(result.get(monomial, Kx.zero), c)
            if not total.numer:
                result.pop(monomial, None)
            else:
                result[monomial] = total
        return result

    def reduce(self, a: AlgebraElement) -> AlgebraElement:
        """The element with every exponent below the index of its
        generator (`g_i^{q_i}` replaced by its base, the later generators
        first since a base involves the earlier ones only)."""
        Kx = self.Kx
        result: AlgebraElement = {}
        pending = dict(a)
        while pending:
            monomial, c = pending.popitem()
            high = max((i for i, e in enumerate(monomial) if e >= self.generators[i].q), default=-1)
            if high < 0:
                result = self.add(result, {monomial: c})
                continue
            lowered = list(monomial)
            lowered[high] -= self.generators[high].q
            for base_monomial, d in self.bases[high].items():
                product = tuple(u + v for u, v in zip(lowered, base_monomial))
                pending = self.add(pending, {product: Kx.mul(c, d)})
        return result

    def multiply(self, a: AlgebraElement, b: AlgebraElement) -> AlgebraElement:
        Kx = self.Kx
        product: AlgebraElement = {}
        for m1, c in a.items():
            for m2, d in b.items():
                product = self.add(product, {tuple(u + v for u, v in zip(m1, m2)): Kx.mul(c, d)})
        return self.reduce(product)

    def vector(self, a: AlgebraElement) -> list[Fraction_]:
        row = [self.Kx.zero] * self.dimension
        for monomial, c in a.items():
            row[self.index[monomial]] = c
        return row


class AlgebraicIntegrand:
    """An integrand rational in `x` and in nested radicals, as an element
    of the function field of a primitive element of the radicals;
    ``ValueError`` when it is not one (no radical, a parameter, a
    function which is not algebraic, dependent radicals).

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import sqrt
    >>> from sympy_extras.integrals.curves import AlgebraicIntegrand
    >>> x, y = symbols('x y')
    >>> A = AlgebraicIntegrand(1/(sqrt(x + 1) + sqrt(x - 1)), x)
    >>> A.curve.xreplace({A.y: y})
    -4*x*y**2 + y**4 + 4
    >>> A.element_expr().xreplace({A.y: y})
    x*y - y**3/4
    >>> A.polynomial_to_expr(A.element_expr())
    -sqrt(x - 1)/2 + sqrt(x + 1)/2
    """

    def __init__(self, f: ExprLike, x: Symbol) -> None:
        f_ = to_expr(f)
        self.x: Final[Symbol] = x
        self.original: Final[Expr] = f_
        if f_.free_symbols - {x}:
            raise ValueError("parameters are not supported on a general curve")
        generators, rewritten = _generators(f_, x)
        if not generators:
            raise ValueError(f"{f_} is rational in {x}")
        if rewritten.is_rational_function(x, *[g.symbol for g in generators]) is not True:
            raise ValueError(f"{f_} is not an algebraic function of {x}")
        self.generators: Final[list[_Generator]] = generators
        algebra = _Algebra(generators, x)
        self.algebra: Final[_Algebra] = algebra
        y = Dummy('y')
        self.y: Final[Dummy] = y
        Kx = algebra.Kx
        m = len(generators)
        n = algebra.dimension
        # a primitive element: a combination with small integer weights
        rng = random.Random(str(f_))
        weights: list[int] = [1] * m
        found: Optional[tuple[list[list[Fraction_]], list[list[Fraction_]], list[KPoly]]] = None
        for _ in range(6):
            theta_element = algebra.reduce({_unit_monomial(m, i): Kx.from_int(k) for i, k in enumerate(weights)})
            found = _minimal_polynomial(algebra, theta_element, y)
            if found is not None:
                break
            weights = [rng.randint(1, 7) for _ in generators]
        if found is None:
            raise ValueError(f"the radicals of {f_} are dependent (no primitive element of degree "
                             f"{n}, the algebra dimension)")
        powers, inverse, curve = found
        self.weights: Final[list[int]] = weights
        self.theta: Final[Expr] = add(*[mul(Integer(k), g.original) for k, g in zip(weights, generators)])
        #: the minimal polynomial of the primitive element: its
        #: coefficients in `y` (highest first), polynomials in `x`
        self.curve_coefficients: Final[list[KPoly]] = curve
        self.curve: Final[Expr] = add(*[mul(_poly_expr(c, x, RATIONALS, Integer(0)), pow(y, Integer(n - i)))
                                        for i, c in enumerate(curve) if c])
        self.field: Final[FunctionField[Number]] = FunctionField(curve, x, y, RATIONALS)
        #: the powers of the primitive element in the monomial basis, and
        #: the inverse matrix
        self.power_matrix: Final[list[list[Fraction_]]] = powers
        self.power_matrix_inverse: Final[list[list[Fraction_]]] = inverse
        try:
            element = self.to_field(rewritten)
        except ZeroDivisionError:
            raise ValueError(f"{f_} has a denominator which vanishes on the curve") from None
        self.element: Final[Element[Number]] = element

    def _from_y_coordinates(self, coordinates: list[Fraction_]) -> Element[Number]:
        """The element with these coordinates in the powers of `y` (the
        field's coordinates are in the powers of `Y = a_n y`)."""
        E = self.field
        Kx = E.Kx
        unit = Kx.from_ring(E.unit)
        result: list[Fraction_] = []
        scale = Kx.one
        for c in coordinates:
            result.append(Kx.quo(c, scale))
            scale = Kx.mul(scale, unit)
        return Element(E, result)

    def to_field(self, expr: Expr) -> Element[Number]:
        """A rational function of `x` and the generator symbols as an
        element of the field of the primitive element: the numerator and
        the denominator in the monomial basis, their coordinates in the
        powers of the primitive element, one inverse in the field."""
        numerator, denominator = fraction(cancel(expr))
        return self._polynomial_to_field(numerator) * self._polynomial_to_field(denominator).inverse()

    def _polynomial_to_field(self, p: Expr) -> Element[Number]:
        algebra = self.algebra
        Kx = algebra.Kx
        vector = algebra.vector(algebra.from_expr(p))
        n = algebra.dimension
        coordinates = [Kx.zero] * n
        for i, c in enumerate(vector):
            if not c.numer:
                continue
            row = self.power_matrix_inverse[i]
            for j in range(n):
                if row[j].numer:
                    coordinates[j] = Kx.add(coordinates[j], Kx.mul(c, row[j]))
        return self._from_y_coordinates(coordinates)

    def element_expr(self) -> Expr:
        """The integrand as an expression in `x` and the primitive element
        `y`."""
        return _element_expr(self.element, self.x, self.y)

    def polynomial_to_expr(self, p: Expr) -> Expr:
        """A rational expression in `x` and the primitive element `y`
        written in the original radicals through the monomial basis."""
        e = self.field.element(p)
        Kx = self.field.Kx
        unit = Kx.from_ring(self.field.unit)
        coordinates: list[Fraction_] = []
        scale = Kx.one
        for c in e.coefficients:
            coordinates.append(Kx.mul(c, scale))
            scale = Kx.mul(scale, unit)
        return self.radicals_expr(coordinates, RATIONALS, Integer(0))

    def radicals_expr(self, coordinates: list[Fraction_], K: AlgebraicField, theta: Expr) -> Expr:
        """The element with these coordinates in the powers of the
        primitive element `y` (elements of `K(x)`, ``theta`` the
        expression of the generator of ``K``) in the original radicals,
        cancelled."""
        R: PolynomialRing[Number] = PolynomialRing(K, self.x.name)
        Kx: FractionField[Number] = FractionField(R)
        totals: list[Fraction_] = []
        for j in range(len(self.algebra.monomials)):
            total = Kx.zero
            for i, c in enumerate(coordinates):
                v = self.power_matrix[i][j]
                if c.numer and v.numer:
                    total = Kx.add(total, Kx.mul(c, v))
            totals.append(total)
        denominator = R.one
        for c in totals:
            if c.numer:
                denominator = R.lcm(denominator, c.denom)
        radicals = [g.original for g in self.generators]
        terms: list[Expr] = []
        for monomial, c in zip(self.algebra.monomials, totals):
            if not c.numer:
                continue
            numerator = R.exquo(R.mul(c.numer, denominator), c.denom)
            term = mul(*[pow(s, Integer(e)) for s, e in zip(radicals, monomial)])
            terms.append(mul(_poly_expr(numerator, self.x, K, theta), term))
        if not terms:
            return Integer(0)
        return cancel(mul(add(*terms), pow(_poly_expr(denominator, self.x, K, theta), Integer(-1))))


def _unit_monomial(m: int, i: int) -> Monomial:
    return tuple(1 if j == i else 0 for j in range(m))


def _minimal_polynomial(algebra: _Algebra, theta: AlgebraElement, y: Symbol
                        ) -> Optional[tuple[list[list[Fraction_]], list[list[Fraction_]], list[KPoly]]]:
    """The powers of ``theta`` in the monomial basis, the inverse of their
    matrix and the minimal polynomial of ``theta`` over `\\mathbb{Q}(x)`
    (primitive with integer coefficients), when it has the degree of the
    algebra (``None`` otherwise: ``theta`` is not a primitive element, or
    the radicals are dependent)."""
    Kx = algebra.Kx
    n = algebra.dimension
    m = len(algebra.generators)
    power: AlgebraElement = {(0,) * m: Kx.one}
    rows: list[list[Fraction_]] = []
    for _ in range(n):
        rows.append(algebra.vector(power))
        power = algebra.multiply(power, theta)
    try:
        inverse = _inverse(rows, Kx)
    except ZeroDivisionError:
        return None
    # theta^n = sum_i a_i theta^i: a = v P^-1
    last = algebra.vector(power)
    a = [Kx.zero] * n
    for i in range(n):
        for j in range(n):
            if last[i].numer and inverse[i][j].numer:
                a[j] = Kx.add(a[j], Kx.mul(last[i], inverse[i][j]))
    monic = [Kx.one] + [Kx.neg(a[n - 1 - k]) for k in range(n)]
    curve = _primitive_curve(monic, algebra.R)
    if not _is_irreducible(curve, algebra.x, y):
        return None
    return rows, inverse, curve


def _primitive_curve(monic: list[Fraction_], R: PolynomialRing[Number]) -> list[KPoly]:
    """The monic polynomial with coefficients in `\\mathbb{Q}(x)` times
    the least common multiple of their denominators, primitive with
    integer coefficients (a negative constant leading coefficient
    changed in sign)."""
    denominator = R.one
    for c in monic:
        if c.numer:
            denominator = R.lcm(denominator, c.denom)
    polys = [R.exquo(R.mul(c.numer, denominator), c.denom) if c.numer else R.zero for c in monic]
    rationals = [q for p in polys for c in p for q in c]
    common = 1
    for q in rationals:
        common = lcm(common, q.denominator)
    content = 0
    for q in rationals:
        content = gcd(content, int(q * common))
    scale = Fraction(common, content)
    if densebasic.dup_degree(polys[0]) == 0 and polys[0][0][0] < 0:
        scale = -scale
    return [[[q * scale for q in c] for c in p] for p in polys]


def _is_irreducible(curve: list[KPoly], x: Symbol, y: Symbol) -> bool:
    """Whether the polynomial in `y` with coefficients in `\\mathbb{Q}[x]`
    (primitive) is irreducible over `\\mathbb{Q}(x)`: irreducible at a
    specialisation of `x` keeping its degree, else by the bivariate
    factorisation."""
    n = len(curve) - 1
    for x0 in [2, 3, 5, 7]:
        point = RATIONALS.from_int(x0)
        values = [densetools.dup_eval(c, point, RATIONALS) for c in curve]
        if not values[0]:
            continue
        rationals = [v[0] if v else Fraction(0) for v in values]
        _, factors = QQ.factor_dense(rationals)
        if len(factors) == 1 and factors[0][1] == 1 and densebasic.dup_degree(factors[0][0]) == n:
            return True
    expression = add(*[mul(_poly_expr(c, x, RATIONALS, Integer(0)), pow(y, Integer(n - i)))
                       for i, c in enumerate(curve) if c])
    _, found = factor_list(expression, y, x)
    in_y = [(factor, k) for factor, k in found if y in factor.free_symbols]
    return len(in_y) == 1 and in_y[0][1] == Integer(1) and poly_degree(in_y[0][0], y) == n


def _generators(f: Expr, x: Symbol) -> tuple[list[_Generator], Expr]:
    """The radicals of ``f`` depending on ``x``, innermost first, each a
    generator `\\text{base}^{1/q}` (`q` the least common multiple of the
    denominators of the exponents of the base), and ``f`` rewritten in
    their symbols."""
    radicals: dict[Expr, int] = {}
    for node in f.atoms(Pow):
        if not isinstance(node, Pow):
            continue
        exponent = node.exp
        base = node.base
        if isinstance(exponent, Rational) and not exponent.is_Integer and base.has(x):
            radicals[base] = lcm(radicals.get(base, 1), exponent.q)
    ordered: list[Expr] = []
    remaining = dict(radicals)
    while remaining:
        # innermost first: a base containing no other base
        for base in sorted(remaining, key=lambda b: (len(str(b)), str(b))):
            if not any(other != base and base.has(other) for other in remaining):
                ordered.append(base)
                del remaining[base]
                break
        else:
            raise ValueError("circular radicals")
    generators: list[_Generator] = []
    rewritten = f
    for base in ordered:
        q = radicals[base]
        symbol = Dummy(f'curve_g{len(generators)}')
        # the base with the earlier generators substituted, as it now
        # appears in the expression
        rewritten_base = base
        for g in generators:
            rewritten_base = _rewrite_radical(rewritten_base, g.base, g.symbol, g.q)
        generators.append(_Generator(symbol, rewritten_base, q, pow(base, Rational(1, q))))
        rewritten = _rewrite_radical(rewritten, rewritten_base, symbol, q)
    return generators, rewritten


def _rewrite_radical(expr: Expr, base: Expr, symbol: Symbol, q: int) -> Expr:
    """``expr`` with every power ``base**(p/r)`` replaced by
    ``symbol**(p*q/r)``, the rebuilt nodes evaluated."""
    rule: dict[Basic, Basic] = {}
    for node in expr.atoms(Pow):
        if isinstance(node, Pow) and node.base == base and isinstance(node.exp, Rational):
            rule[node] = pow(symbol, Integer(node.exp.p * q // node.exp.q))
    if not rule:
        return expr
    rewritten = expr._xreplace_evaluated(rule)
    if not isinstance(rewritten, Expr):
        raise TypeError(f"{rewritten} is not an expression")
    return rewritten


# Expressions of the typed objects

def _number_expr(c: Number, K: AlgebraicField, theta: Expr) -> Expr:
    """The element ``c`` of ``K`` as an expression, ``theta`` the
    expression of the generator."""
    if len(c) <= 1:
        return QQ.to_expr(c[0]) if c else Integer(0)
    n = len(c) - 1
    parts: list[Expr] = []
    for i, a in enumerate(c):
        if a:
            k = n - i
            q = QQ.to_expr(a)
            parts.append(q if k == 0 else mul(q, theta if k == 1 else pow(theta, Integer(k))))
    return add(*parts)


def _poly_expr(p: KPoly, x: Expr, K: AlgebraicField, theta: Expr) -> Expr:
    """The polynomial ``p`` over ``K`` as an expression in ``x``."""
    n = len(p) - 1
    parts: list[Expr] = []
    for i, c in enumerate(p):
        if c:
            k = n - i
            value = _number_expr(c, K, theta)
            parts.append(value if k == 0 else mul(value, x if k == 1 else pow(x, Integer(k))))
    return add(*parts) if parts else Integer(0)


def _fraction_expr(c: Fraction_, x: Expr, K: AlgebraicField, theta: Expr) -> Expr:
    numerator = _poly_expr(c.numer, x, K, theta)
    if densebasic.dup_degree(c.denom) == 0 and c.denom[0] == K.one:
        return numerator
    return mul(numerator, pow(_poly_expr(c.denom, x, K, theta), Integer(-1)))


def _element_expr(e: Element[Number], x: Symbol, y: Symbol) -> Expr:
    """An element of a field over the rationals as an expression in `x`
    and the original `y` (with the real symbol `x`, which the native
    field names only)."""
    E = e.field
    Y = mul(_poly_expr(E.unit, x, RATIONALS, Integer(0)), y)
    terms = [mul(_fraction_expr(c, x, RATIONALS, Integer(0)), pow(Y, Integer(i)))
             for i, c in enumerate(e.coefficients) if c.numer]
    return cancel(add(*terms)) if terms else Integer(0)


def _substituted(p: KPoly, A: KPoly, B: KPoly, d: int, K: AlgebraicField) -> KPoly:
    """`B^d p(A/B)` for `d \\ge \\deg p`."""
    total: KPoly = []
    n = len(p) - 1
    for i, c in enumerate(p):
        if c:
            k = n - i
            term = densearith.dup_mul(densearith.dup_pow(A, k, K), densearith.dup_pow(B, d - k, K), K)
            total = densearith.dup_add(total, densearith.dup_mul_ground(term, c, K), K)
    return total


def _substituted_fraction(c: Fraction_, A: KPoly, B: KPoly, target: FractionField[Number]) -> Fraction_:
    """`c(A/B)` for `c` in `K(t)`, an element of ``target``."""
    if not c.numer:
        return target.zero
    K = target.ground
    if not isinstance(K, AlgebraicField):
        raise TypeError("coefficients in a number field expected")
    da = densebasic.dup_degree(c.numer)
    db = densebasic.dup_degree(c.denom)
    numerator = _substituted(c.numer, A, B, da, K)
    denominator = _substituted(c.denom, A, B, db, K)
    if db >= da:
        numerator = densearith.dup_mul(numerator, densearith.dup_pow(B, db - da, K), K)
    else:
        denominator = densearith.dup_mul(denominator, densearith.dup_pow(B, da - db, K), K)
    return target.new(numerator, denominator)


# The curve in z = 1/(x - N)

class _Transformed:
    """The integrand and the curve after `x = N + 1/z`."""

    def __init__(self, integrand: AlgebraicIntegrand, N: int) -> None:
        self.integrand: Final[AlgebraicIntegrand] = integrand
        self.N: Final[int] = N
        z = Dummy('z')
        self.z: Final[Dummy] = z
        K = RATIONALS
        curve_x = integrand.curve_coefficients
        d = max(densebasic.dup_degree(c) for c in curve_x)
        # x = (N z + 1)/z
        forward_A: KPoly = densebasic.dup_strip([K.from_int(N), K.one], K)
        forward_B: KPoly = [K.one, K.zero]
        curve = [_substituted(c, forward_A, forward_B, d, K) for c in curve_x]
        field = FunctionField(curve, z, integrand.y, K)
        self.field: Final[FunctionField[Number]] = field
        self.x_field: Final[FunctionField[Number]] = integrand.field
        n = len(curve) - 1
        points: Curve = {}
        for i, c in enumerate(curve):
            if c:
                points[n - i] = densebasic.dup_to_dict(c, K)
        #: the curve in `z` and `y` for the Puiseux expansions
        self.curve: Final[Curve] = points
        Kz = field.Kx
        # the generators Y_x = a_n(x) y and Y_z = a_z(z) y: y is the same function
        unit_x = _substituted_fraction(integrand.field.Kx.from_ring(integrand.field.unit), forward_A, forward_B, Kz)
        self.unit_ratio: Final[Fraction_] = Kz.quo(unit_x, Kz.from_ring(field.unit))
        minus_z2 = Kz.new([K.from_int(-1)], [K.one, K.zero, K.zero])
        self.integrand_element: Final[Element[Number]] = self.to_z(integrand.element).scaled(minus_z2)

    def to_z(self, e: Element[Number]) -> Element[Number]:
        """An element of the field in `x` as an element of the field in `z`."""
        Kz = self.field.Kx
        K = RATIONALS
        A: KPoly = densebasic.dup_strip([K.from_int(self.N), K.one], K)
        B: KPoly = [K.one, K.zero]
        coefficients: list[Fraction_] = []
        ratio = Kz.one
        for c in e.coefficients:
            # c(x) Y_x^i = c(N + 1/z) (a_x/a_z)^i Y_z^i
            coefficients.append(Kz.mul(_substituted_fraction(c, A, B, Kz), ratio))
            ratio = Kz.mul(ratio, self.unit_ratio)
        return Element(self.field, coefficients)

    def to_x(self, c: Fraction_, K: AlgebraicField) -> Fraction_:
        """An element of `K(z)` as an element of `K(x)`: `z = 1/(x - N)`."""
        target: FractionField[Number] = FractionField(PolynomialRing(K, self.integrand.x.name))
        A: KPoly = [K.one]
        B: KPoly = densebasic.dup_strip([K.one, K.from_int(-self.N)], K)
        return _substituted_fraction(c, A, B, target)

    def to_x_expr(self, e: Element[Number], theta: Expr) -> Expr:
        """An element of the field in `z` (over any coefficient field) as
        an expression in `x` and the original radicals."""
        K = e.field.K
        if not isinstance(K, AlgebraicField):
            raise TypeError("coefficients in a number field expected")
        Kz = e.field.Kx
        unit = Kz.from_ring(self.field.unit)
        coordinates: list[Fraction_] = []
        scale = Kz.one
        for c in e.coefficients:
            # the coefficient of y^i: c_i a_z^i
            coordinates.append(self.to_x(Kz.mul(c, scale), K))
            scale = Kz.mul(scale, unit)
        return self.integrand.radicals_expr(coordinates, K, theta)


# The Hermite reduction

class _Reduced:
    """The reduced integrand `h = \\sum_i A_i w_i / D`, `D` squarefree,
    with the elementary part `g` subtracted (`f = g' + h`)."""

    __slots__ = ('g', 'numerators', 'D')

    def __init__(self, g: Element[Number], numerators: list[KPoly], D: KPoly) -> None:
        self.g: Final[Element[Number]] = g
        self.numerators: Final[list[KPoly]] = numerators  # elements of K[z]
        self.D: Final[KPoly] = D


def _coordinates_over_common_denominator(E: FunctionField[Number], e: Element[Number], basis: list[Element[Number]]
                                         ) -> tuple[list[KPoly], KPoly]:
    """``(A, D)`` with ``e = sum(A_i w_i)/D``, ``A_i`` and ``D`` polynomials,
    ``D`` monic and coprime to the ``A_i`` together."""
    R = E.R
    K = E.K
    coordinates = E.coordinates(e, basis)
    D = R.one
    for c in coordinates:
        D = R.lcm(D, c.denom)
    A: list[KPoly] = [R.mul(c.numer, R.exquo(D, c.denom)) for c in coordinates]
    content = D
    for a in A:
        content = R.gcd(content, a)
    if _degree(content) > 0:
        A = [R.exquo(a, content) for a in A]
        D = R.exquo(D, content)
    lc = D[0]
    if lc != K.one:
        inverse = [K.quo(K.one, lc)]
        A = [R.mul(a, inverse) for a in A]
        D = R.mul(D, inverse)
    return A, D


def _degree(p: KPoly) -> int:
    return densebasic.dup_degree(p) if p else -1


def _derivative_matrix(E: FunctionField[Number], basis: list[Element[Number]]) -> tuple[KPoly, list[list[KPoly]]]:
    """``(e, M)`` with `e W' = M W`, `e` a polynomial and `M` a polynomial
    matrix."""
    R = E.R
    rows = [E.coordinates(w.derivative(), basis) for w in basis]
    e = R.one
    for row in rows:
        for c in row:
            e = R.lcm(e, c.denom)
    M = [[R.mul(c.numer, R.exquo(e, c.denom)) for c in row] for row in rows]
    return e, M


def _hermite_reduce(E: FunctionField[Number], basis: list[Element[Number]], f: Element[Number]) -> _Reduced:
    """Trager's Hermite reduction of ``f`` in the integral ``basis``."""
    R = E.R
    Kx = E.Kx
    K = E.K
    n = E.n
    e, M = _derivative_matrix(E, basis)
    g = E.zero()
    A, D = _coordinates_over_common_denominator(E, f, basis)
    while True:
        factors = dup_sqf_list(D, K)[1]
        m = max((multiplicity for _, multiplicity in factors), default=1)
        if m < 2:
            break
        V = R.one
        for factor, multiplicity in factors:
            if multiplicity == m:
                V = R.mul(V, factor)
        U = R.exquo(D, R.pow(V, m))
        # e must divide U V: otherwise the numerator and the denominator are
        # multiplied by the missing part (V^m stays the highest power)
        missing = R.exquo(e, R.gcd(e, R.mul(U, V)))
        if _degree(missing) > 0:
            A = [R.mul(a, missing) for a in A]
            D = R.mul(D, missing)
            continue
        # b (U V M / e - (m - 1) U V' I) = A (mod V)
        UV_over_e = R.exquo(R.mul(U, V), e)
        shift = R.mul(R.mul(R.from_int(m - 1), U), R.diff(V))
        Phi = [[R.sub(R.mul(UV_over_e, M[i][j]), shift) if i == j else R.mul(UV_over_e, M[i][j]) for j in range(n)]
               for i in range(n)]
        b = _solve_modulo(Phi, A, V, R, Kx)
        B = E.zero()
        for b_i, w in zip(b, basis):
            if b_i:
                B = B + w.scaled(Kx.from_ring(b_i))
        term = B.scaled(Kx.invert(Kx.from_ring(R.pow(V, m - 1))))
        g = g + term
        remainder = _combination(E, A, D, basis) - term.derivative()
        A, D = _coordinates_over_common_denominator(E, remainder, basis)
    return _Reduced(g, A, D)


def _combination(E: FunctionField[Number], A: list[KPoly], D: KPoly, basis: list[Element[Number]]) -> Element[Number]:
    Kx = E.Kx
    total = E.zero()
    for a, w in zip(A, basis):
        if a:
            total = total + w.scaled(Kx.new(a, D))
    return total


def _solve_modulo(Phi: list[list[KPoly]], A: list[KPoly], V: KPoly, R: PolynomialRing[Number],
                  Kx: FractionField[Number]) -> list[KPoly]:
    """``b`` with ``b Phi = A`` modulo ``V`` (`\\det \\Phi` invertible
    modulo `V`, Trager's theorem for an integral basis)."""
    n = len(A)
    det = _determinant(Phi, R)
    adj = _adjugate(Phi, R, Kx)
    try:
        inverse = dup_invert(det, V, R.ground)
    except ValueError:
        raise ValueError("the Hermite reduction step has no solution: not an integral basis") from None
    b: list[KPoly] = []
    for j in range(n):
        total = R.zero
        for i in range(n):
            total = R.add(total, R.mul(A[i], adj[i][j]))
        b.append(R.rem(R.mul(total, inverse), V))
    return b


# The places above the roots of the denominator and the residues

class _Place:
    """A place above a root of the factor ``G`` (over the coefficient
    field `K`) of the denominator, a representative of its conjugates
    over `K`, with the branch, the embedding of `K` into the field of
    the branch, and the residue of `h\\,dz` there (in that field)."""

    __slots__ = ('G', 'key', 'branch', 'embedding', 'residue')

    def __init__(self, G: KPoly, branch: PuiseuxExpansion, embedding: Embedding, residue: Number) -> None:
        self.G: Final[KPoly] = G
        self.key: Final[FactorKey] = _factor_key(G)
        self.branch: Final[PuiseuxExpansion] = branch
        self.embedding: Final[Embedding] = embedding
        self.residue: Final[Number] = residue

    @property
    def ramification(self) -> int:
        return self.branch.ramification

    def residue_in(self) -> Number:
        """The residue as an element of `K`."""
        value = in_subfield(self.residue, self.branch.field, self.embedding)
        if value is None:
            raise ValueError("a residue outside the field of the residues")
        return value


def _factor_key(G: KPoly) -> FactorKey:
    return tuple(tuple(c) for c in G)


def _poly_curve(p: KPoly, K: AlgebraicField) -> Curve:
    """A polynomial in `z` as a curve without `y`."""
    return {0: densebasic.dup_to_dict(p, K)} if p else {}


def _sorted_factors(factors: list[tuple[KPoly, int]], K: AlgebraicField) -> list[tuple[KPoly, int]]:
    """The irreducible factors in SymPy's order of ``factor_list``: by
    length, multiplicity and coefficients (over the rationals those of
    the primitive integer factor)."""
    def key(factor: tuple[KPoly, int]) -> tuple[int, int, list[list[Fraction]]]:
        f, k = factor
        if K.degree > 1:
            return (len(f), k, [list(c) for c in f])
        rationals = [c[0] if c else Fraction(0) for c in f]
        common = 1
        for q in rationals:
            common = lcm(common, q.denominator)
        content = 0
        for q in rationals:
            content = gcd(content, int(q * common))
        scale = Fraction(common, content)
        return (len(f), k, [[q * scale] if q else [] for q in rationals])
    return sorted(factors, key=key)


def _factors(p: KPoly, K: AlgebraicField) -> list[tuple[KPoly, int]]:
    return _sorted_factors(factor_over(K, p), K)


def _basis_series(E: FunctionField[Number], basis: list[Element[Number]], branch: PuiseuxExpansion, order: int
                  ) -> list[LaurentSeries]:
    """The series of the basis elements along ``branch``, known below
    `t^{\\text{order}}`."""
    R = E.R
    K = RATIONALS
    result: list[LaurentSeries] = []
    for w in basis:
        d = R.one
        for c in w.coefficients:
            d = R.lcm(d, c.denom)
        # the coordinates are in the powers of the integral generator a_n y,
        # the branches are those of y
        numerator: Curve = {}
        unit_power = R.one
        for i, c in enumerate(w.coefficients):
            if c.numer:
                numerator[i] = densebasic.dup_to_dict(R.mul(R.exquo(R.mul(c.numer, d), c.denom), unit_power), K)
            unit_power = R.mul(unit_power, E.unit)
        series_denominator = branch.evaluate(_poly_curve(d, K))
        v_d = series_denominator.valuation()
        if v_d is None:
            raise ValueError("a denominator vanishing along the branch")
        series_numerator = branch.evaluate(numerator).truncated(order + v_d)
        v_n = series_numerator.valuation()
        relative = max(order - (v_n - v_d) + 1, 1) if v_n is not None else 1
        result.append((series_numerator * series_denominator.inverse(relative)).truncated(order))
    return result


def _places(T: _Transformed, basis: list[Element[Number]], reduced: _Reduced, K: AlgebraicField,
            orders: dict[FactorKey, int]) -> list[_Place]:
    """The places above one root of each irreducible factor over ``K`` of
    the denominator of the reduced integrand, with their residues; the
    branches above the roots of a factor are known to the order
    ``orders`` gives for it (in units of `z`) when that is more than
    the residues need."""
    E = T.field
    z, Y = E.x, E.y
    places: list[_Place] = []
    for G, _ in _factors(reduced.D, K):
        if _degree(G) < 1:
            continue
        K_point, into, beta = adjoin_root(K, G)
        multiplicity = max(_multiplicity_of_root(E, w, G, K) for w in basis)
        order = max(3 + multiplicity, orders.get(_factor_key(G), 0))
        for branch in expansions_in_field(T.curve, z, Y, K_point, Embedding(RATIONALS, K_point), beta, order):
            L = branch.field
            e = branch.ramification
            # h = S/D with D of valuation e (squarefree): the coefficient of
            # t^(-e) of h needs S below t^1
            series_basis = _basis_series(E, basis, branch, 1)
            S = LaurentSeries.constant(L, L.zero)
            for a, s in zip(reduced.numerators, series_basis):
                S = S + (branch.evaluate(_poly_curve(a, RATIONALS)) * s).truncated(1)
            D_series = branch.evaluate(_poly_curve(reduced.D, RATIONALS))
            v_S = S.valuation()
            if v_S is None:
                # no pole on this branch
                residue = L.zero
            else:
                h = S * D_series.inverse(max(2 - v_S, 1))
                if h.order is not None and h.order <= -e:
                    raise ValueError("insufficient precision for the residue")
                # dz = scale e t^(e-1) dt: the residue is scale e [t^(-e)] h
                residue = L.mul(L.mul(h.coefficient(-e), branch.scale), L.from_int(e))
            places.append(_Place(G, branch, into.then(branch.embedding), residue))
    return places


def _multiplicity_of_root(E: FunctionField[Number], w: Element[Number], G: KPoly, K: AlgebraicField) -> int:
    """The multiplicity of ``G`` in the denominator of ``w``."""
    R = E.R
    d = R.one
    for c in w.coefficients:
        d = R.lcm(d, c.denom)
    p = d
    multiplicity = 0
    while _degree(p) >= _degree(G):
        quotient, remainder = densearith.dup_div(p, G, K)
        if remainder:
            break
        p = quotient
        multiplicity += 1
    return multiplicity


# The logarithmic part

class _LogTerm:
    """`\\beta/(n m) \\log(N/\\text{den})`, `N` in the field over the number
    field of the residues, ``den`` a polynomial in `z`."""

    __slots__ = ('beta', 'n', 'm', 'N', 'den')

    def __init__(self, beta: Number, n: int, m: int, N: Element[Number], den: KPoly) -> None:
        self.beta: Final[Number] = beta
        self.n: Final[int] = n
        self.m: Final[int] = m
        self.N: Final[Element[Number]] = N
        self.den: Final[KPoly] = den  # an element of K[z]


def _residue_field(places: list[_Place]) -> AlgebraicField:
    """The splitting field over the rationals of the minimal polynomials
    of the residues: the residues at all the places, conjugates
    included, lie in it."""
    K = RATIONALS
    for place in places:
        m = minimal_polynomial_of(place.residue, place.branch.field)
        if len(m) - 1 == 1:
            continue
        while True:
            factors = [factor for factor, _ in _factors([K.from_rational(c) for c in m], K) if _degree(factor) > 1]
            if not factors:
                break
            factor = factors[0]
            if degree(K) * _degree(factor) > RESIDUE_FIELD_DEGREE:
                raise ValueError("the field of the residues is too large")
            K, _, _ = adjoin_root(K, factor)
    return K


def _rational_coordinates(value: Number, K: AlgebraicField) -> list[Fraction]:
    """The coordinates of an element of the number field ``K`` in the
    powers of its generator (rationals, low degree first)."""
    coefficients = list(reversed(value))
    return coefficients + [Fraction(0)] * (degree(K) - len(coefficients))


def _rank(vectors: list[list[Fraction]]) -> int:
    return len(QQ.sparse_echelon([{j: c for j, c in enumerate(v) if c} for v in vectors]))


def _rational_basis(values: list[Number], K: AlgebraicField) -> tuple[list[Number], list[list[Fraction]]]:
    """A basis over the rationals of the span of ``values`` and the
    coordinates of every value in it."""
    vectors = [_rational_coordinates(v, K) for v in values]
    dimension = len(vectors[0])
    chosen: list[int] = []
    rank = 0
    for i, vector in enumerate(vectors):
        if _rank([vectors[j] for j in chosen] + [vector]) > rank:
            chosen.append(i)
            rank += 1
    basis = [values[i] for i in chosen]
    r = len(chosen)
    coordinates: list[list[Fraction]] = []
    for vector in vectors:
        # c B = v: one equation per coordinate, the unknowns c_j
        rows: list[dict[int, Fraction]] = []
        for k in range(dimension):
            row = {j: vectors[chosen[j]][k] for j in range(r) if vectors[chosen[j]][k]}
            if vector[k]:
                row[r] = vector[k]
            if row:
                rows.append(row)
        solution = [Fraction(0)] * r
        for row in QQ.sparse_echelon(rows):
            pivot = min(row)
            if pivot == r:
                raise ValueError("a residue outside the span of the basis")
            solution[pivot] = row.get(r, Fraction(0))
        coordinates.append(solution)
    return basis, coordinates


def _embedded(e: Element[Number], E_K: FunctionField[Number]) -> Element[Number]:
    """An element of the field over the rationals in the field of the same
    curve over a number field (the rational coefficients are elements of
    every number field as they are)."""
    return Element(E_K, list(e.coefficients))


def _function_with_divisor(T: _Transformed, basis: list[Element[Number]], orders: list[int], places: list[_Place],
                           coefficients: list[int], n: int, K: AlgebraicField, E_K: FunctionField[Number]
                           ) -> Optional[tuple[Element[Number], KPoly]]:
    """``(N, den)`` with `u = N/\\text{den}` of divisor `n \\sum_P c_P P`
    (``coefficients`` the `c_P`, in the order of ``places``), or ``None``
    when there is none."""
    E = T.field
    R_K = E_K.R
    k_of: dict[FactorKey, int] = {}
    factor_of: dict[FactorKey, KPoly] = {}
    for place, c in zip(places, coefficients):
        needed = -((n * c) // place.ramification) if c < 0 else 0
        k_of[place.key] = max(k_of.get(place.key, 0), needed)
        factor_of[place.key] = place.G
    den = R_K.one
    for key, k in k_of.items():
        den = R_K.mul(den, R_K.pow(factor_of[key], k))
    degree_den = _degree(den)
    unknowns: list[tuple[int, int]] = []
    for i, k in enumerate(orders):
        for s in range(degree_den - k + 1):
            unknowns.append((i, s))
    if not unknowns:
        return None
    d_K = degree(K)
    columns_count = len(unknowns) * d_K
    rows: list[list[Fraction]] = []
    for place, c in zip(places, coefficients):
        branch = place.branch
        e = branch.ramification
        required = n * c + k_of[place.key] * e  # ord_P(N) >= required
        multiplicity = max(_multiplicity_of_root(E, w, place.G, K) for w in basis)
        lowest = -e * (multiplicity + 1)
        if required <= lowest:
            continue
        LK = branch.field
        D_LK = degree(LK)
        theta_powers = [LK.one]
        generator_K = place.embedding(generator(K))
        for _ in range(d_K - 1):
            theta_powers.append(LK.mul(theta_powers[-1], generator_K))
        series_basis = _basis_series(E, basis, branch, required)
        z_series = branch.x_series()
        columns: list[LaurentSeries] = []
        for i, s in unknowns:
            columns.append((z_series.power(s) * series_basis[i]).truncated(required))
        for series in columns:
            if series.order is not None and series.order < required:
                raise ValueError("insufficient precision in the divisor conditions")
        for exponent in range(lowest, required):
            for coordinate in range(D_LK):
                row = [Fraction(0)] * columns_count
                for u, series in enumerate(columns):
                    value = series.coefficient(exponent)
                    if not value:
                        continue
                    for r in range(d_K):
                        row[u * d_K + r] = _rational_coordinates(LK.mul(value, theta_powers[r]), LK)[coordinate]
                rows.append(row)
    if not rows:
        return None
    kernel = QQ.nullspace(rows, columns_count)
    if not kernel:
        return None
    solution = kernel[0]
    Kz = E_K.Kx
    N = E_K.zero()
    basis_K = [_embedded(w, E_K) for w in basis]
    for u, (i, s) in enumerate(unknowns):
        coefficient = K.zero
        power = K.one
        for r in range(d_K):
            coefficient = K.add(coefficient, K.mul(K.from_rational(solution[u * d_K + r]), power))
            power = K.mul(power, generator(K))
        if not coefficient:
            continue
        N = N + basis_K[i].scaled(Kz.from_ring([coefficient] + [K.zero] * s))
    if N.is_zero():
        return None
    return N, den


# The algorithm

class CurveIntegral:
    """The pieces of the integral of an algebraic integrand: the
    elementary part `g` of the Hermite reduction and the logarithms in
    `z`, with the number field of the residues, the verdict
    ``nonelementary`` (``True`` when the integral is proved not
    elementary, ``False`` when found, ``None`` when undecided) and the
    integrand left (in `z`, ``None`` when nothing is left)."""

    def __init__(self, integrand: AlgebraicIntegrand, transformed: _Transformed, reduced: _Reduced,
                 logs: list[_LogTerm], residue_field: AlgebraicField, nonelementary: Optional[bool],
                 remainder: Optional[Element[Number]]) -> None:
        self.integrand: Final[AlgebraicIntegrand] = integrand
        self.transformed: Final[_Transformed] = transformed
        self.reduced: Final[_Reduced] = reduced
        self.logs: Final[list[_LogTerm]] = logs
        self.residue_field: Final[AlgebraicField] = residue_field
        self.nonelementary: Final[Optional[bool]] = nonelementary
        self.remainder: Final[Optional[Element[Number]]] = remainder

    def _theta(self, theta: Optional[Expr]) -> Expr:
        """The expression of the generator of the field of the residues."""
        K = self.residue_field
        if theta is not None:
            return theta
        if K.degree == 1:
            return Integer(0)
        if isinstance(K.alpha_expr, Symbol):
            raise ValueError(f"the generator of {K} has no expression (a root of {K.minpoly} which is not "
                             "real and not in radicals)")
        return K.alpha_expr

    def elementary_part(self, theta: Optional[Expr] = None) -> Expr:
        """The elementary part found, in `x` and the original radicals;
        ``theta`` is the expression of the generator of the field of the
        residues (by default its own: an
        :class:`~sympy_extras.polys.cad.indexedroot.IndexedRoot` or
        radicals; ``ValueError`` when it has none)."""
        T = self.transformed
        generator_expr = self._theta(theta)
        total = T.to_x_expr(self.reduced.g, generator_expr)
        K = self.residue_field
        x = self.integrand.x
        for term in self.logs:
            den = T.to_x(FracElement(term.den, [K.one]), K)
            u = cancel(mul(T.to_x_expr(term.N, generator_expr),
                           pow(_fraction_expr(den, x, K, generator_expr), Integer(-1))))
            coefficient = mul(_number_expr(term.beta, K, generator_expr), Rational(1, term.n * term.m))
            total = add(total, mul(coefficient, log(u)))
        return total

    def remainder_expr(self, theta: Optional[Expr] = None) -> Expr:
        """The remaining integrand in `x` (``theta`` as for
        :meth:`elementary_part`)."""
        if self.remainder is None:
            return Integer(0)
        T = self.transformed
        x = self.integrand.x
        # the remainder is a function of z integrated in dz = -dx/(x - N)^2
        return cancel(mul(Integer(-1), T.to_x_expr(self.remainder, self._theta(theta)),
                          pow(add(x, Integer(-T.N)), Integer(-2))))


def integrate_on_curve(f: ExprLike, x: Symbol, logarithms: bool = True) -> CurveIntegral:
    """Trager's algorithm on the curve of the radicals of ``f``; raises
    :class:`ValueError` when ``f`` is not an algebraic function of `x`
    of the kind the parser understands (nested radicals, independent).

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import sqrt
    >>> from sympy_extras.integrals.curves import integrate_on_curve
    >>> x = symbols('x')
    >>> result = integrate_on_curve(x/sqrt(1 - x**3), x)
    >>> result.nonelementary, result.elementary_part()
    (True, -2*sqrt(1 - x**3)/x)
    """
    integrand = AlgebraicIntegrand(f, x)
    N = _regular_integer(integrand)
    T = _Transformed(integrand, N)
    E = T.field
    basis, orders = _normal_basis(E)
    reduced = _hermite_reduce(E, basis, T.integrand_element)
    nonzero_numerators = [(a, k) for a, k in zip(reduced.numerators, orders) if a]
    if not nonzero_numerators:
        return CurveIntegral(integrand, T, reduced, [], RATIONALS, False, None)
    remainder = _combination(E, reduced.numerators, reduced.D, basis)
    # Trager's criterion: h z^2 integral at infinity
    pole = max(_degree(a) - _degree(reduced.D) + k for a, k in nonzero_numerators)
    if pole > -2:
        return CurveIntegral(integrand, T, reduced, [], RATIONALS, True, remainder)
    if not logarithms:
        return CurveIntegral(integrand, T, reduced, [], RATIONALS, None, remainder)
    places = _places(T, basis, reduced, RATIONALS, {})
    K = _residue_field(places)
    E_K = FunctionField(E.curve, E.x, E.y, K) if K.degree > 1 else E
    divisors, support = _divisors(T, basis, reduced, K, {})
    if not divisors:
        # no residue and a nonzero remainder: a differential of the first kind
        return CurveIntegral(integrand, T, reduced, [], K, True, remainder)
    logs: list[_LogTerm] = []
    undecided = False
    precision: dict[FactorKey, int] = {}
    for j in range(len(divisors)):
        found: Optional[tuple[Element[Number], KPoly]] = None
        for n in range(1, TORSION_BOUND + 1):
            # the branches known far enough for the vanishing conditions of
            # this multiple (the places recomputed when the precision grows:
            # their order may change, so the residues are taken again)
            needed = _orders_needed(E, basis, support, divisors[j][2], n, K)
            if any(order > precision.get(G, 0) for G, order in needed.items()):
                for G, order in needed.items():
                    precision[G] = max(precision.get(G, 0), order)
                divisors, support = _divisors(T, basis, reduced, K, precision)
            beta, m, c = divisors[j]
            found = _function_with_divisor(T, basis, orders, support, c, n, K, E_K)
            if found is not None:
                logs.append(_LogTerm(beta, n, m, found[0], found[1]))
                break
        if found is None:
            undecided = True
    # the exact check: h - sum beta/(n m) u'/u
    Kz = E_K.Kx
    h_K = _embedded(remainder, E_K)
    for term in logs:
        u = term.N.scaled(Kz.invert(Kz.from_ring(term.den)))
        scale = K.mul(term.beta, K.from_rational(Fraction(1, term.n * term.m)))
        h_K = h_K - (u.derivative() * u.inverse()).scaled(Kz.from_ring([scale]))
    if h_K.is_zero():
        return CurveIntegral(integrand, T, reduced, logs, K, False, None)
    if undecided:
        return CurveIntegral(integrand, T, reduced, logs, K, None, h_K)
    return CurveIntegral(integrand, T, reduced, logs, K, True, h_K)


def _divisors(T: _Transformed, basis: list[Element[Number]], reduced: _Reduced, K: AlgebraicField,
              orders: dict[FactorKey, int]) -> tuple[list[tuple[Number, int, list[int]]], list[_Place]]:
    """The places above the roots of the denominator (branches known to
    the ``orders`` asked per factor) and, for each element
    of a basis of the span of the residues over the rationals, the
    common denominator and the integer coefficients of its divisor (one
    per place, zero at the places of zero residue)."""
    places = _places(T, basis, reduced, K, orders)
    residues = [place.residue_in() for place in places]
    if all(not r for r in residues):
        return [], places
    # every place takes part in the conditions (no pole of the function
    # at a place of zero residue above a root of the denominator)
    span, coordinates = _rational_basis(residues, K)
    divisors: list[tuple[Number, int, list[int]]] = []
    for j, beta in enumerate(span):
        fractions_ = [coordinates[k][j] for k in range(len(places))]
        m = 1
        for q in fractions_:
            m = lcm(m, q.denominator)
        divisors.append((beta, m, [int(q * m) for q in fractions_]))
    return divisors, places


def _orders_needed(E: FunctionField[Number], basis: list[Element[Number]], places: list[_Place],
                   coefficients: list[int], n: int, K: AlgebraicField) -> dict[FactorKey, int]:
    """The order (in units of `z`) to which the branches above the roots
    of each factor must be known for the vanishing conditions of the
    multiple `n` of the divisor with the given coefficients."""
    k_of: dict[FactorKey, int] = {}
    for place, c in zip(places, coefficients):
        needed = -((n * c) // place.ramification) if c < 0 else 0
        k_of[place.key] = max(k_of.get(place.key, 0), needed)
    orders: dict[FactorKey, int] = {}
    for place, c in zip(places, coefficients):
        e = place.ramification
        multiplicity = max(_multiplicity_of_root(E, w, place.G, K) for w in basis)
        required = n * c + k_of[place.key] * e
        # the series of y to order `required` plus the denominators' orders
        needed = -(-(required + e * (multiplicity + 1) + 2) // e) + 1
        orders[place.key] = max(orders.get(place.key, 0), needed)
    return orders


def _normal_basis(E: FunctionField[Number]) -> tuple[list[Element[Number]], list[int]]:
    """The integral basis of ``E`` (a field over the rationals) normal at
    infinity, with its pole orders there: computed in the field of the
    same curve over the plain rationals (the arithmetic on lists of one
    rational is slower) and converted."""
    coefficients = [[c[0] if c else Fraction(0) for c in p] for p in E.curve]
    plain: FunctionField[Fraction] = FunctionField(coefficients, E.x, E.y, QQ)
    basis = plain.normal_basis()
    orders = plain.pole_orders_at_infinity(basis)

    def lifted(p: list[Fraction]) -> KPoly:
        return [[c] if c else [] for c in p]

    return [Element(E, [FracElement(lifted(c.numer), lifted(c.denom)) for c in w.coefficients]) for w in basis], orders


def _regular_integer(integrand: AlgebraicIntegrand) -> int:
    """An integer which is neither a pole of the integrand nor a root of
    the discriminant of the curve."""
    x = integrand.x
    K = RATIONALS
    R = integrand.algebra.R
    polynomials: list[KPoly] = [dup_discriminant(integrand.curve_coefficients, R)]
    for c in integrand.element.coefficients:
        polynomials.append(c.denom)
    expressions = [fraction(cancel(g.base))[1] for g in integrand.generators]
    for N in [0, 1, -1, 2, -2, 3, -3, 5, -5, 7, -7, 11, 13, 17, 19, 23]:
        point = K.from_int(N)
        if any(not densetools.dup_eval(p, point, K) for p in polynomials):
            continue
        if all(d._xreplace_evaluated({x: Integer(N)}) != Integer(0) for d in expressions):
            return N
    raise ValueError("no regular integer found")


# The public functions

def curve_reduce(f: ExprLike, x: Symbol) -> Optional[tuple[Expr, Expr]]:
    """``(F, h)`` with ``f == F' + h``: the elementary part of the integral
    of the algebraic function ``f`` on the curve of its radicals and the
    remaining integrand (zero when the integral is elementary), or
    ``None`` when ``f`` is not such a function.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import sqrt
    >>> from sympy_extras.integrals.curves import curve_reduce
    >>> x = symbols('x')
    >>> curve_reduce(1/(x*sqrt(1 - x**3)), x)
    (-log((x**3 - 2*sqrt(1 - x**3) - 2)/(2*x**3))/3, 0)
    >>> curve_reduce(x/sqrt(1 - x**3), x)
    (-2*sqrt(1 - x**3)/x, 2*sqrt(1 - x**3)/(x**5 - x**2))
    """
    try:
        result = integrate_on_curve(f, x)
        return result.elementary_part(), result.remainder_expr()
    except ValueError:
        return None


def curve_antiderivative(f: ExprLike, x: Symbol) -> Optional[Expr]:
    """An antiderivative of the algebraic function ``f`` by Trager's
    algorithm on the curve of its radicals (several or nested radicals,
    any index), or ``None`` when the integral is not elementary, not
    found, or ``f`` is not such a function.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import sqrt
    >>> from sympy_extras.integrals.curves import curve_antiderivative
    >>> x = symbols('x')
    >>> curve_antiderivative(1/(x*sqrt(1 - x**3)), x)
    -log((x**3 - 2*sqrt(1 - x**3) - 2)/(2*x**3))/3
    """
    try:
        result = integrate_on_curve(f, x)
        if result.nonelementary is not False or result.remainder is not None:
            return None
        return result.elementary_part()
    except ValueError:
        return None


def is_nonelementary_on_curve(f: ExprLike, x: Symbol) -> Optional[bool]:
    """Whether the integral of the algebraic function ``f`` is not
    elementary: ``True`` when proved (a pole of order two or more at
    infinity after the Hermite reduction, or a nonzero remainder after
    the logarithmic part), ``False`` when an antiderivative is found,
    ``None`` when undecided or ``f`` is not such a function.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import sqrt
    >>> from sympy_extras.integrals.curves import is_nonelementary_on_curve
    >>> x = symbols('x')
    >>> is_nonelementary_on_curve(x/sqrt(1 - x**3), x)
    True
    """
    try:
        result = integrate_on_curve(f, x)
    except ValueError:
        return None
    return result.nonelementary
