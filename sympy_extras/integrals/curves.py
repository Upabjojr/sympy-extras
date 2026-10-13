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

The answer is written back in `x` and in the original radicals (each
power of the primitive element expressed in the monomials of the
radicals), and the dispatcher checks it by differentiation.

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
from math import lcm
from typing import Optional

from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.numbers import Rational
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.elementary.exponential import log
from sympy.polys.domains import QQ
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.domain import Domain
from sympy.polys.domains.fractionfield import FractionField
from sympy.polys.domains.polynomialring import PolynomialRing
from sympy.polys.matrices import DomainMatrix
from sympy.polys.numberfields.minpoly import minimal_polynomial
from sympy.polys.polyerrors import CoercionFailed, PolynomialError, NotInvertible
from sympy.polys.polytools import Poly, cancel
from sympy.simplify.radsimp import fraction

from sympy_extras._typing import DomainElement, ExprLike, as_expr, free_symbols
from sympy_extras.polys.functionfields import FunctionField, Element
from sympy_extras.polys.puiseux import (PuiseuxExpansion, LaurentSeries, Embedding, expansions_in_field,
                                        adjoin_root, in_subfield, minimal_polynomial_of, generator, degree,
                                        coefficients_in)

__all__ = ['curve_antiderivative', 'curve_reduce', 'is_nonelementary_on_curve', 'integrate_on_curve',
           'AlgebraicIntegrand', 'CurveIntegral']

#: the multiples of a residue divisor tried before giving up (Mazur's
#: bound for a curve of genus one over the rationals)
TORSION_BOUND = 12

Monomial = tuple[int, ...]
AlgebraElement = dict[Monomial, DomainElement]


# The algebraic functions of the integrand and their primitive element

class _Generator:
    """A radical `g = \\text{base}^{1/q}` of the integrand, the base a
    rational function of `x` and of the earlier generators (replaced by
    their symbols)."""

    def __init__(self, symbol: Symbol, base: Expr, q: int, original: Expr) -> None:
        self.symbol = symbol
        self.base = base
        self.q = q
        self.original = original


class _Algebra:
    """The algebra `\\mathbb{Q}(x)[g_1, \\ldots, g_m]/(g_i^{q_i} -
    \\text{base}_i)` with the monomial basis `\\prod g_i^{e_i}`, `e_i <
    q_i`; the field of the integrand when the radicals are
    independent."""

    def __init__(self, generators: list[_Generator], x: Symbol) -> None:
        self.generators = generators
        self.x = x
        self.Kx: FractionField = QQ.frac_field(x)
        self.symbols = [g.symbol for g in generators]
        self.dimension = 1
        for g in generators:
            self.dimension *= g.q
        self.bases: list[AlgebraElement] = []
        for g in generators:
            # the base in the earlier generators only
            self.bases.append(self.from_expr(g.base))
        self.monomials: list[Monomial] = []
        self._enumerate((), 0)
        self.index = {m: i for i, m in enumerate(self.monomials)}

    def _enumerate(self, prefix: Monomial, k: int) -> None:
        if k == len(self.generators):
            self.monomials.append(prefix)
            return
        for e in range(self.generators[k].q):
            self._enumerate(prefix + (e,), k + 1)

    def from_expr(self, expr: Expr) -> AlgebraElement:
        """A polynomial in `x` and the generator symbols as an element."""
        m = len(self.generators)
        if not self.symbols:
            return {(): self.Kx.from_sympy(expr)}
        P = Poly(expr, *self.symbols)
        result: AlgebraElement = {}
        for monomial, coefficient in P.terms():
            term = {tuple(monomial) + (0,) * (m - len(monomial)): self.Kx.from_sympy(as_expr(coefficient))}
            result = self.add(result, self.reduce(term))
        return result

    def add(self, a: AlgebraElement, b: AlgebraElement) -> AlgebraElement:
        result = dict(a)
        for monomial, c in b.items():
            total = result.get(monomial, self.Kx.zero) + c
            if total == self.Kx.zero:
                result.pop(monomial, None)
            else:
                result[monomial] = total
        return result

    def reduce(self, a: AlgebraElement) -> AlgebraElement:
        """The element with every exponent below the index of its
        generator (`g_i^{q_i}` replaced by its base, the later generators
        first since a base involves the earlier ones only)."""
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
                pending = self.add(pending, {product: c * d})
        return result

    def multiply(self, a: AlgebraElement, b: AlgebraElement) -> AlgebraElement:
        product: AlgebraElement = {}
        for m1, c in a.items():
            for m2, d in b.items():
                product = self.add(product, {tuple(u + v for u, v in zip(m1, m2)): c * d})
        return self.reduce(product)

    def vector(self, a: AlgebraElement) -> list[DomainElement]:
        row = [self.Kx.zero] * self.dimension
        for monomial, c in a.items():
            row[self.index[monomial]] = c
        return row


class AlgebraicIntegrand:
    """An integrand rational in `x` and in nested radicals, as an element
    of the function field of a primitive element of the radicals.

    >>> from sympy import symbols, sqrt
    >>> from sympy_extras.integrals.curves import AlgebraicIntegrand
    >>> x = symbols('x')
    >>> A = AlgebraicIntegrand(1/(sqrt(x + 1) + sqrt(x - 1)), x)
    >>> y = symbols('y')
    >>> A.curve.subs(A.y, y)
    -4*x*y**2 + y**4 + 4
    >>> A.element.as_expr().subs(A.y, y)
    x*y - y**3/4
    """

    def __init__(self, f: ExprLike, x: Symbol) -> None:
        f_ = as_expr(f)
        self.x = x
        self.original = f_
        if free_symbols(f_) - {x}:
            raise ValueError("parameters are not supported on a general curve")
        generators, rewritten = _generators(f_, x)
        if not generators:
            raise ValueError(f"{f_} is rational in {x}")
        if not rewritten.is_rational_function(x, *[g.symbol for g in generators]):
            raise ValueError(f"{f_} is not an algebraic function of {x}")
        self.generators = generators
        self.algebra = _Algebra(generators, x)
        y = Dummy('y')
        self.y = y
        # a primitive element: a combination with small integer weights
        rng = random.Random(str(f_))
        weights: list[int] = [1] * len(generators)
        degree = 0
        for _ in range(6):
            theta = as_expr(Add(*[k * g.original for k, g in zip(weights, generators)]))
            try:
                F = as_expr(minimal_polynomial(theta, y))
            except (NotImplementedError, PolynomialError, CoercionFailed, NotInvertible) as error:
                raise ValueError(f"no minimal polynomial for {theta}: {error}") from None
            degree = Poly(F, y).degree()
            if degree == self.algebra.dimension:
                break
            if degree > self.algebra.dimension:
                raise ValueError("inconsistent degrees")
            weights = [rng.randint(1, 7) for _ in generators]
        else:
            raise ValueError(f"the radicals of {f_} are dependent (the field has degree {degree}, "
                             f"the algebra dimension {self.algebra.dimension})")
        self.weights = weights
        self.theta = theta
        self.curve = F
        self.field = FunctionField(F, x, y)
        # the powers of theta in the monomial basis of the algebra
        theta_element = self.algebra.reduce({_unit_monomial(len(generators), i): self.algebra.Kx.convert(k, QQ)
                                             for i, k in enumerate(weights)})
        powers = [{(0,) * len(generators): self.algebra.Kx.one}]
        for _ in range(degree - 1):
            powers.append(self.algebra.multiply(powers[-1], theta_element))
        self.power_matrix = DomainMatrix([self.algebra.vector(p) for p in powers], (degree, degree), self.algebra.Kx)
        self.power_matrix_inverse = self.power_matrix.inv()
        self.element = self.to_field(rewritten)

    def to_field(self, expr: Expr) -> Element:
        """A rational function of `x` and the generator symbols as an
        element of the field of the primitive element."""
        numerator, denominator = fraction(cancel(expr))
        return self._polynomial_to_field(as_expr(numerator)) * self._polynomial_to_field(as_expr(denominator)).inverse()

    def _polynomial_to_field(self, p: Expr) -> Element:
        vector = DomainMatrix([self.algebra.vector(self.algebra.from_expr(p))], (1, self.algebra.dimension),
                              self.algebra.Kx)
        coordinates = (vector * self.power_matrix_inverse).to_list()[0]
        # in the powers of theta = y (the generator of the field as given,
        # before its leading coefficient is absorbed)
        return self.field.element(as_expr(Add(*[self.algebra.Kx.to_sympy(c) * self.y**i
                                                for i, c in enumerate(coordinates)])))

    def polynomial_to_expr(self, p: Expr, K: Optional[Domain] = None) -> Expr:
        """A rational expression in `x` and the primitive element `y`
        written in the original radicals through the monomial basis."""
        numerator, denominator = fraction(cancel(p))
        return as_expr(cancel(self._polynomial_in_radicals(as_expr(numerator))
                              / self._polynomial_in_radicals(as_expr(denominator))))

    def _polynomial_in_radicals(self, p: Expr) -> Expr:
        Kx = self.algebra.Kx
        P = Poly(p, self.y)
        total: Expr = S.Zero
        symbols = [g.original for g in self.generators]
        rows = self.power_matrix.to_list()
        for i, c in enumerate(reversed(P.all_coeffs())):
            if c == 0:
                continue
            for monomial, v in zip(self.algebra.monomials, rows[i]):
                if v == Kx.zero:
                    continue
                term = Mul(*[s**e for s, e in zip(symbols, monomial)])
                total = total + as_expr(c) * Kx.to_sympy(v) * term
        return as_expr(total)


def _unit_monomial(m: int, i: int) -> Monomial:
    return tuple(1 if j == i else 0 for j in range(m))


def _generators(f: Expr, x: Symbol) -> tuple[list[_Generator], Expr]:
    """The radicals of ``f`` depending on ``x``, innermost first, each a
    generator `\\text{base}^{1/q}` (`q` the least common multiple of the
    denominators of the exponents of the base), and ``f`` rewritten in
    their symbols."""
    radicals: dict[Expr, int] = {}
    for node in f.atoms(Pow):
        exponent = as_expr(node.exp)
        base = as_expr(node.base)
        if isinstance(exponent, Rational) and not exponent.is_Integer and base.has(x):
            radicals[base] = lcm(radicals.get(base, 1), int(exponent.q))
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
        symbol = Dummy(f'g{len(generators)}')
        # the base with the earlier generators substituted, as it now
        # appears in the expression
        rewritten_base = base
        for g in generators:
            rewritten_base = _rewrite_radical(rewritten_base, g.base, g.symbol, g.q)
        generators.append(_Generator(symbol, rewritten_base, q, as_expr(base**Rational(1, q))))
        rewritten = _rewrite_radical(rewritten, rewritten_base, symbol, q)
    return generators, rewritten


def _rewrite_radical(expr: Expr, base: Expr, symbol: Symbol, q: int) -> Expr:
    """``expr`` with every power ``base**(p/r)`` replaced by
    ``symbol**(p*q/r)``."""
    def replace(node: Expr) -> Expr:
        if isinstance(node, Pow) and as_expr(node.base) == base:
            exponent = as_expr(node.exp)
            if isinstance(exponent, Rational):
                return as_expr(symbol**int(exponent * q))
        return node
    return as_expr(expr.replace(lambda node: isinstance(node, Pow), replace))


# The curve in z = 1/(x - N)

class _Transformed:
    """The integrand and the curve after `x = N + 1/z`."""

    def __init__(self, integrand: AlgebraicIntegrand, N: int) -> None:
        self.integrand = integrand
        self.N = N
        x = integrand.x
        y = integrand.y
        z = Dummy('z')
        self.z = z
        F = integrand.curve
        d = Poly(F, x).degree()
        curve = as_expr(cancel(F.xreplace({x: N + 1 / z}) * z**d))
        self.field = FunctionField(curve, z, y)
        self.x_field = integrand.field
        self.curve = curve
        # the generators Y_x = a_n(x) y and Y_z = a_z(z) y: y is the same function
        self.unit_ratio = as_expr(cancel(integrand.field.unit.xreplace({x: N + 1 / z}) / self.field.unit))
        self.integrand_element = self.to_z(integrand.element).scaled(self.field.Kx.from_sympy(as_expr(-1 / z**2)))

    def to_z(self, e: Element) -> Element:
        """An element of the field in `x` as an element of the field in `z`."""
        x = self.integrand.x
        z = self.z
        Kx = self.x_field.Kx
        Kz = self.field.Kx
        coefficients: list[DomainElement] = []
        for i, c in enumerate(e.coefficients):
            # c(x) Y_x^i = c(N + 1/z) (a_x/a_z)^i Y_z^i
            coefficients.append(Kz.from_sympy(as_expr(cancel(Kx.to_sympy(c).xreplace({x: self.N + 1 / z})
                                                             * self.unit_ratio**i))))
        return Element(self.field, coefficients)

    def to_x_expr(self, e: Element) -> Expr:
        """An element of the field in `z` (over any coefficient field) as
        an expression in `x` and the original radicals."""
        x = self.integrand.x
        z = self.z
        y = self.integrand.y
        Kz = e.field.Kx
        p = as_expr(Add(*[Kz.to_sympy(c) * (self.field.unit * y)**i for i, c in enumerate(e.coefficients)]))
        p = as_expr(cancel(p.xreplace({z: 1 / (x - self.N)})))
        return self.integrand.polynomial_to_expr(p)


# The Hermite reduction

class _Reduced:
    """The reduced integrand `h = \\sum_i A_i w_i / D`, `D` squarefree,
    with the elementary part `g` subtracted (`f = g' + h`)."""

    def __init__(self, g: Element, numerators: list[DomainElement], D: DomainElement) -> None:
        self.g = g
        self.numerators = numerators  # elements of K[z]
        self.D = D


def _coordinates_over_common_denominator(E: FunctionField, e: Element, basis: list[Element]
                                         ) -> tuple[list[DomainElement], DomainElement]:
    """``(A, D)`` with ``e = sum(A_i w_i)/D``, ``A_i`` and ``D`` polynomials,
    ``D`` monic and coprime to the ``A_i`` together."""
    R = E.R
    coordinates = E.coordinates(e, basis)
    D = R.one
    for c in coordinates:
        D = R.lcm(D, _poly(c.denom, R))
    A: list[DomainElement] = []
    for c in coordinates:
        A.append(_poly(c.numer, R) * R.exquo(D, _poly(c.denom, R)))
    content = D
    for a in A:
        content = R.gcd(content, a)
    if _degree(content) > 0:
        A = [R.exquo(a, content) for a in A]
        D = R.exquo(D, content)
    lc = D.LC
    if lc != E.K.one:
        inverse = R.convert(E.K.quo(E.K.one, lc), E.K)
        A = [a * inverse for a in A]
        D = D * inverse
    return A, D


def _poly(p: DomainElement, R: PolynomialRing) -> DomainElement:
    if p.ring == R.ring:
        return R.ring.from_dict(p.to_dict())
    return R.from_sympy(as_expr(p.as_expr()))


def _degree(p: DomainElement) -> int:
    if p == 0:
        return -1
    return int(p.degree())


def _derivative_matrix(E: FunctionField, basis: list[Element]) -> tuple[DomainElement, list[list[DomainElement]]]:
    """``(e, M)`` with `e W' = M W`, `e` a polynomial and `M` a polynomial
    matrix."""
    R = E.R
    rows = [E.coordinates(w.derivative(), basis) for w in basis]
    e = R.one
    for row in rows:
        for c in row:
            e = R.lcm(e, _poly(c.denom, R))
    M = [[_poly(c.numer, R) * R.exquo(e, _poly(c.denom, R)) for c in row] for row in rows]
    return e, M


def _hermite_reduce(E: FunctionField, basis: list[Element], f: Element) -> _Reduced:
    """Trager's Hermite reduction of ``f`` in the integral ``basis``."""
    R = E.R
    Kx = E.Kx
    K = E.K
    n = E.n
    x = E.x
    e, M = _derivative_matrix(E, basis)
    g = E.zero()
    A, D = _coordinates_over_common_denominator(E, f, basis)
    while True:
        factors = Poly(R.to_sympy(D), x, domain=K).sqf_list()[1]
        m = max((multiplicity for _, multiplicity in factors), default=1)
        if m < 2:
            break
        V = R.one
        for factor, multiplicity in factors:
            if multiplicity == m:
                V = V * R.from_sympy(factor.as_expr())
        U = R.exquo(D, V**m)
        # e must divide U V: otherwise the numerator and the denominator are
        # multiplied by the missing part (V^m stays the highest power)
        missing = R.exquo(e, R.gcd(e, U * V))
        if _degree(missing) > 0:
            A = [a * missing for a in A]
            D = D * missing
            continue
        # b (U V M / e - (m - 1) U V' I) = A (mod V)
        UV_over_e = R.exquo(U * V, e)
        V_prime = V.diff(V.ring.gens[0])
        Phi = [[UV_over_e * M[i][j] - ((m - 1) * U * V_prime if i == j else R.zero) for j in range(n)]
               for i in range(n)]
        b = _solve_modulo(Phi, A, V, R, K, x)
        B = E.zero()
        for b_i, w in zip(b, basis):
            if b_i != R.zero:
                B = B + w.scaled(Kx.convert(b_i, R))
        term = B.scaled(Kx.one / Kx.convert(V**(m - 1), R))
        g = g + term
        remainder = _combination(E, A, D, basis) - term.derivative()
        A, D = _coordinates_over_common_denominator(E, remainder, basis)
    return _Reduced(g, A, D)


def _combination(E: FunctionField, A: list[DomainElement], D: DomainElement, basis: list[Element]) -> Element:
    Kx = E.Kx
    R = E.R
    total = E.zero()
    for a, w in zip(A, basis):
        if a != R.zero:
            total = total + w.scaled(Kx.convert(a, R) / Kx.convert(D, R))
    return total


def _solve_modulo(Phi: list[list[DomainElement]], A: list[DomainElement], V: DomainElement, R: PolynomialRing,
                  K: Domain, x: Symbol) -> list[DomainElement]:
    """``b`` with ``b Phi = A`` modulo ``V`` (`\\det \\Phi` invertible
    modulo `V`, Trager's theorem for an integral basis)."""
    n = len(A)
    matrix = DomainMatrix([list(row) for row in Phi], (n, n), R)
    det = matrix.det()
    adj = matrix.adjugate().to_list()
    V_poly = Poly(R.to_sympy(V), x, domain=K)
    det_poly = Poly(R.to_sympy(det), x, domain=K)
    try:
        inverse = det_poly.invert(V_poly)
    except NotInvertible:
        raise ValueError("the Hermite reduction step has no solution: not an integral basis") from None
    inverse_R = R.from_sympy(inverse.as_expr())
    b: list[DomainElement] = []
    for j in range(n):
        total = R.zero
        for i in range(n):
            total = total + A[i] * adj[i][j]
        b.append((total * inverse_R).rem(V))
    return b


# The places above the roots of the denominator and the residues

class _Place:
    """A place above a root of the factor ``G`` (over the coefficient
    field `K`) of the denominator, a representative of its conjugates
    over `K`, with the branch, the embedding of `K` into the field of
    the branch, and the residue of `h\\,dz` there (in that field)."""

    def __init__(self, G: Poly, branch: PuiseuxExpansion, embedding: Embedding, residue: DomainElement) -> None:
        self.G = G
        self.branch = branch
        self.embedding = embedding
        self.residue = residue

    @property
    def ramification(self) -> int:
        return self.branch.ramification

    def residue_in(self, K: Domain) -> DomainElement:
        """The residue as an element of `K`."""
        value = in_subfield(self.residue, self.branch.field, self.embedding)
        if value is None:
            raise ValueError("a residue outside the field of the residues")
        return value


def _basis_series(E: FunctionField, basis: list[Element], branch: PuiseuxExpansion, order: int
                  ) -> list[LaurentSeries]:
    """The series of the basis elements along ``branch``, known below
    `t^{\\text{order}}`."""
    Kx = E.Kx
    R = E.R
    z, Y = E.x, E.y
    result: list[LaurentSeries] = []
    for w in basis:
        d = R.one
        for c in w.coefficients:
            d = R.lcm(d, _poly(c.denom, R))
        # the coordinates are in the powers of the integral generator a_n y,
        # the branches are those of y
        numerator = as_expr(Add(*[Kx.to_sympy(c * Kx.convert(d, R)) * (E.unit * Y)**i
                                  for i, c in enumerate(w.coefficients)]))
        series_denominator = branch.evaluate(Poly(R.to_sympy(d), z))
        v_d = series_denominator.valuation()
        if v_d is None:
            raise ValueError("a denominator vanishing along the branch")
        series_numerator = branch.evaluate(Poly(numerator, Y, z)).truncated(order + v_d)
        v_n = series_numerator.valuation()
        relative = max(order - (v_n - v_d) + 1, 1) if v_n is not None else 1
        result.append((series_numerator * series_denominator.inverse(relative)).truncated(order))
    return result


def _places(T: _Transformed, basis: list[Element], reduced: _Reduced, K: Domain, orders: dict[Expr, int]
            ) -> list[_Place]:
    """The places above one root of each irreducible factor over ``K`` of
    the denominator of the reduced integrand, with their residues; the
    branches above the roots of a factor are known to the order
    ``orders`` gives for it (in units of `z`) when that is more than
    the residues need."""
    E = T.field
    R = E.R
    z, Y = E.x, E.y
    D_expr = R.to_sympy(reduced.D)
    places: list[_Place] = []
    curve = Poly(T.curve, Y, z, domain=QQ)
    numerators = [R.to_sympy(a) for a in reduced.numerators]
    for G, _ in Poly(D_expr, z, domain=K).factor_list()[1]:
        if G.degree() < 1:
            continue
        K_point, into, beta = _root_of_factor(G, K)
        multiplicity = max(_multiplicity_of_root(E, w, G, K) for w in basis)
        order = max(3 + multiplicity, orders.get(G.as_expr(), 0))
        for branch in expansions_in_field(curve, z, Y, K_point, Embedding(QQ, K_point), beta, order):
            L = branch.field
            e = branch.ramification
            # h = S/D with D of valuation e (squarefree): the coefficient of
            # t^(-e) of h needs S below t^1
            series_basis = _basis_series(E, basis, branch, 1)
            S = LaurentSeries.constant(L, L.zero)
            for a, s in zip(numerators, series_basis):
                S = S + (branch.evaluate(Poly(a, z)) * s).truncated(1)
            D_series = branch.evaluate(Poly(D_expr, z))
            v_S = S.valuation()
            if v_S is None:
                # no pole on this branch
                residue = L.zero
            else:
                h = S * D_series.inverse(max(2 - v_S, 1))
                if h.order is not None and h.order <= -e:
                    raise ValueError("insufficient precision for the residue")
                # dz = scale e t^(e-1) dt: the residue is scale e [t^(-e)] h
                residue = h.coefficient(-e) * branch.scale * L.convert(e, QQ)
            places.append(_Place(G, branch, into.then(branch.embedding), residue))
    return places


def _root_of_factor(G: Poly, K: Domain) -> tuple[Domain, Embedding, DomainElement]:
    """A root of the irreducible polynomial ``G`` over ``K``: its field,
    the embedding of ``K`` into it, the root."""
    return adjoin_root(K, coefficients_in(G, K))


def _multiplicity_of_root(E: FunctionField, w: Element, G: Poly, K: Domain) -> int:
    """The multiplicity of ``G`` in the denominator of ``w``."""
    R = E.R
    d = R.one
    for c in w.coefficients:
        d = R.lcm(d, _poly(c.denom, R))
    p = Poly(R.to_sympy(d), E.x, domain=K)
    multiplicity = 0
    while p.degree() >= G.degree():
        quotient, remainder = p.div(G)
        if not remainder.is_zero:
            break
        p = quotient
        multiplicity += 1
    return multiplicity


# The logarithmic part

class _LogTerm:
    """`\\beta/(n m) \\log(N/\\text{den})`, `N` in the field over the number
    field of the residues, ``den`` a polynomial in `z`."""

    def __init__(self, beta: DomainElement, n: int, m: int, N: Element, den: DomainElement, den_expr: Expr) -> None:
        self.beta = beta
        self.n = n
        self.m = m
        self.N = N
        self.den = den  # an element of K[z]
        self.den_expr = den_expr


#: the degree beyond which the field of the residues is not built
RESIDUE_FIELD_DEGREE = 24


def _residue_field(places: list[_Place]) -> Domain:
    """The splitting field over the rationals of the minimal polynomials
    of the residues: the residues at all the places, conjugates
    included, lie in it."""
    T = Symbol('T')
    K: Domain = QQ
    for place in places:
        m = minimal_polynomial_of(place.residue, place.branch.field, T)
        if m.degree() == 1:
            continue
        while True:
            factors = [factor for factor, _ in Poly(m.as_expr(), T, domain=K).factor_list()[1] if factor.degree() > 1]
            if not factors:
                break
            factor = factors[0]
            if degree(K) * factor.degree() > RESIDUE_FIELD_DEGREE:
                raise ValueError("the field of the residues is too large")
            K, _, _ = adjoin_root(K, coefficients_in(factor, K))
    return K


def _rational_coordinates(value: DomainElement, K: Domain) -> list[DomainElement]:
    """The coordinates of an element of the number field ``K`` in the
    powers of its generator (rationals)."""
    if not isinstance(K, AlgebraicField):
        return [value]
    coefficients = list(reversed(value.to_list()))
    return coefficients + [QQ.zero] * (degree(K) - len(coefficients))


def _rational_basis(values: list[DomainElement], K: Domain) -> tuple[list[DomainElement], list[list[Fraction]]]:
    """A basis over the rationals of the span of ``values`` and the
    coordinates of every value in it."""
    vectors = [_rational_coordinates(v, K) for v in values]
    dimension = len(vectors[0])
    chosen: list[int] = []
    rank = 0
    for i, vector in enumerate(vectors):
        rows = [vectors[j] for j in chosen] + [vector]
        if DomainMatrix(rows, (len(rows), dimension), QQ).rank() > rank:
            chosen.append(i)
            rank += 1
    basis = [values[i] for i in chosen]
    # c B = v: c = v B^T (B B^T)^-1
    B = DomainMatrix([vectors[j] for j in chosen], (len(chosen), dimension), QQ)
    gram_inverse = (B * B.transpose()).inv()
    coordinates: list[list[Fraction]] = []
    for vector in vectors:
        v = DomainMatrix([vector], (1, dimension), QQ)
        c = (v * B.transpose() * gram_inverse).to_list()[0]
        coordinates.append([Fraction(int(q.numerator), int(q.denominator)) for q in c])
    return basis, coordinates


def _function_with_divisor(T: _Transformed, basis: list[Element], orders: list[int], places: list[_Place],
                           coefficients: list[int], n: int, K: Domain, E_K: FunctionField
                           ) -> Optional[tuple[Element, DomainElement]]:
    """``(N, den)`` with `u = N/\\text{den}` of divisor `n \\sum_P c_P P`
    (``coefficients`` the `c_P`, in the order of ``places``), or ``None``
    when there is none."""
    E = T.field
    R_K = E_K.R
    z = E.x
    k_of: dict[Poly, int] = {}
    for place, c in zip(places, coefficients):
        needed = -((n * c) // place.ramification) if c < 0 else 0
        k_of[place.G] = max(k_of.get(place.G, 0), needed)
    den = R_K.one
    for G, k in k_of.items():
        den = den * R_K.ring.from_list(coefficients_in(G, K))**k
    degree_den = _degree(den)
    unknowns: list[tuple[int, int]] = []
    for i, k in enumerate(orders):
        for s in range(degree_den - k + 1):
            unknowns.append((i, s))
    if not unknowns:
        return None
    d_K = degree(K)
    columns_count = len(unknowns) * d_K
    rows: list[list[DomainElement]] = []
    for place, c in zip(places, coefficients):
        branch = place.branch
        e = branch.ramification
        required = n * c + k_of[place.G] * e  # ord_P(N) >= required
        multiplicity = max(_multiplicity_of_root(E, w, place.G, K) for w in basis)
        lowest = -e * (multiplicity + 1)
        if required <= lowest:
            continue
        LK = branch.field
        D_LK = degree(LK)
        theta_powers = [LK.one]
        generator_K = place.embedding(generator(K))
        for _ in range(d_K - 1):
            theta_powers.append(theta_powers[-1] * generator_K)
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
                row: list[DomainElement] = [QQ.zero] * columns_count
                for u, series in enumerate(columns):
                    value = series.coefficient(exponent)
                    if value == LK.zero:
                        continue
                    for r in range(d_K):
                        row[u * d_K + r] = _rational_coordinates(value * theta_powers[r], LK)[coordinate]
                rows.append(row)
    if not rows:
        return None
    matrix = DomainMatrix(rows, (len(rows), columns_count), QQ)
    kernel = matrix.nullspace()
    if kernel.shape[0] == 0:
        return None
    solution = kernel.to_list()[0]
    Kz = E_K.Kx
    N = E_K.zero()
    basis_K = [E_K.convert(w) for w in basis]
    for u, (i, s) in enumerate(unknowns):
        coefficient = K.zero
        power = K.one
        for r in range(d_K):
            coefficient = coefficient + K.convert(solution[u * d_K + r], QQ) * power
            power = power * generator(K)
        if coefficient == K.zero:
            continue
        N = N + basis_K[i].scaled(Kz.convert(coefficient, K) * Kz.from_sympy(z**s))
    if N.is_zero():
        return None
    return N, den


# The algorithm

class CurveIntegral:
    """The pieces of the integral of an algebraic integrand."""

    def __init__(self, integrand: AlgebraicIntegrand, transformed: _Transformed, reduced: _Reduced,
                 logs: list[_LogTerm], residue_field: Domain, nonelementary: Optional[bool],
                 remainder: Optional[Element]) -> None:
        self.integrand = integrand
        self.transformed = transformed
        self.reduced = reduced
        self.logs = logs
        self.residue_field = residue_field
        #: ``True`` when the integral is proved not elementary, ``False``
        #: when found, ``None`` when undecided
        self.nonelementary = nonelementary
        #: the integrand left (in `z`) after the elementary part found
        self.remainder = remainder

    def elementary_part(self) -> Expr:
        """The elementary part found, in `x` and the original radicals."""
        T = self.transformed
        total = T.to_x_expr(self.reduced.g)
        K = self.residue_field
        x = self.integrand.x
        for term in self.logs:
            u = as_expr(cancel(T.to_x_expr(term.N) / term.den_expr.xreplace({T.z: 1 / (x - T.N)})))
            coefficient = as_expr(K.to_sympy(term.beta) / (term.n * term.m))
            total = total + coefficient * log(u)
        return as_expr(total)

    def remainder_expr(self) -> Expr:
        """The remaining integrand in `x`."""
        if self.remainder is None:
            return S.Zero
        T = self.transformed
        x = self.integrand.x
        # the remainder is a function of z integrated in dz = -dx/(x - N)^2
        return as_expr(cancel(-T.to_x_expr(self.remainder) / (x - T.N)**2))


def integrate_on_curve(f: ExprLike, x: Symbol, logarithms: bool = True) -> CurveIntegral:
    """Trager's algorithm on the curve of the radicals of ``f``; raises
    :class:`ValueError` when ``f`` is not an algebraic function of `x`
    of the kind the parser understands (nested radicals, independent).

    >>> from sympy import symbols, sqrt
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
    basis = E.normal_basis()
    orders = E.pole_orders_at_infinity(basis)
    reduced = _hermite_reduce(E, basis, T.integrand_element)
    nonzero_numerators = [(a, k) for a, k in zip(reduced.numerators, orders) if a != E.R.zero]
    if not nonzero_numerators:
        return CurveIntegral(integrand, T, reduced, [], QQ, False, None)
    remainder = _combination(E, reduced.numerators, reduced.D, basis)
    # Trager's criterion: h z^2 integral at infinity
    pole = max(_degree(a) - _degree(reduced.D) + k for a, k in nonzero_numerators)
    if pole > -2:
        return CurveIntegral(integrand, T, reduced, [], QQ, True, remainder)
    if not logarithms:
        return CurveIntegral(integrand, T, reduced, [], QQ, None, remainder)
    places = _places(T, basis, reduced, QQ, {})
    K = _residue_field(places)
    E_K = FunctionField(T.curve, T.z, T.integrand.y, domain=K) if K != QQ else E
    divisors, support = _divisors(T, basis, reduced, K, {})
    if not divisors:
        # no residue and a nonzero remainder: a differential of the first kind
        return CurveIntegral(integrand, T, reduced, [], K, True, remainder)
    logs: list[_LogTerm] = []
    undecided = False
    precision: dict[Expr, int] = {}
    for j in range(len(divisors)):
        found: Optional[tuple[Element, DomainElement]] = None
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
                logs.append(_LogTerm(beta, n, m, found[0], found[1], as_expr(E_K.R.to_sympy(found[1]))))
                break
        if found is None:
            undecided = True
    # the exact check: h - sum beta/(n m) u'/u
    h_K = E_K.convert(remainder)
    for term in logs:
        u = term.N.scaled(E_K.Kx.one / E_K.Kx.convert(term.den, E_K.R))
        h_K = h_K - (u.derivative() * u.inverse()).scaled(E_K.Kx.convert(term.beta, K)
                                                          / E_K.Kx.convert(term.n * term.m, QQ))
    if h_K.is_zero():
        return CurveIntegral(integrand, T, reduced, logs, K, False, None)
    if undecided:
        return CurveIntegral(integrand, T, reduced, logs, K, None, h_K)
    return CurveIntegral(integrand, T, reduced, logs, K, True, h_K)


def _divisors(T: _Transformed, basis: list[Element], reduced: _Reduced, K: Domain, orders: dict[Expr, int]
              ) -> tuple[list[tuple[DomainElement, int, list[int]]], list[_Place]]:
    """The places above the roots of the denominator (branches known to
    the ``orders`` asked per factor) and, for each element
    of a basis of the span of the residues over the rationals, the
    common denominator and the integer coefficients of its divisor (one
    per place, zero at the places of zero residue)."""
    places = _places(T, basis, reduced, K, orders)
    residues = [place.residue_in(K) for place in places]
    if all(r == K.zero for r in residues):
        return [], places
    # every place takes part in the conditions (no pole of the function
    # at a place of zero residue above a root of the denominator)
    span, coordinates = _rational_basis(residues, K)
    divisors: list[tuple[DomainElement, int, list[int]]] = []
    for j, beta in enumerate(span):
        fractions_ = [coordinates[k][j] for k in range(len(places))]
        m = 1
        for q in fractions_:
            m = lcm(m, q.denominator)
        divisors.append((beta, m, [int(q * m) for q in fractions_]))
    return divisors, places


def _orders_needed(E: FunctionField, basis: list[Element], places: list[_Place], coefficients: list[int], n: int,
                   K: Domain) -> dict[Expr, int]:
    """The order (in units of `z`) to which the branches above the roots
    of each factor must be known for the vanishing conditions of the
    multiple `n` of the divisor with the given coefficients."""
    k_of: dict[Expr, int] = {}
    for place, c in zip(places, coefficients):
        G = as_expr(place.G.as_expr())
        needed = -((n * c) // place.ramification) if c < 0 else 0
        k_of[G] = max(k_of.get(G, 0), needed)
    orders: dict[Expr, int] = {}
    for place, c in zip(places, coefficients):
        G = as_expr(place.G.as_expr())
        e = place.ramification
        multiplicity = max(_multiplicity_of_root(E, w, place.G, K) for w in basis)
        required = n * c + k_of[G] * e
        # the series of y to order `required` plus the denominators' orders
        needed = -(-(required + e * (multiplicity + 1) + 2) // e) + 1
        orders[G] = max(orders.get(G, 0), needed)
    return orders


def _regular_integer(integrand: AlgebraicIntegrand) -> int:
    """An integer which is neither a pole of the integrand nor a root of
    the discriminant of the curve."""
    x = integrand.x
    y = integrand.y
    disc = Poly(integrand.curve, y).discriminant()
    denominators: list[Expr] = [as_expr(disc)]
    for c in integrand.element.coefficients:
        denominators.append(as_expr(integrand.field.Kx.to_sympy(c).as_numer_denom()[1]))
    for g in integrand.generators:
        denominators.append(as_expr(fraction(cancel(g.base))[1]))
    for N in [0, 1, -1, 2, -2, 3, -3, 5, -5, 7, -7, 11, 13, 17, 19, 23]:
        if all(as_expr(d.xreplace({x: N})) != 0 for d in denominators):
            return N
    raise ValueError("no regular integer found")


# The public functions

def curve_reduce(f: ExprLike, x: Symbol) -> Optional[tuple[Expr, Expr]]:
    """``(F, h)`` with ``f == F' + h``: the elementary part of the integral
    of the algebraic function ``f`` on the curve of its radicals and the
    remaining integrand (zero when the integral is elementary), or
    ``None`` when ``f`` is not such a function.

    >>> from sympy import symbols, sqrt
    >>> from sympy_extras.integrals.curves import curve_reduce
    >>> x = symbols('x')
    >>> curve_reduce(1/(x*sqrt(1 - x**3)), x)
    (-log((x**3 - 2*sqrt(1 - x**3) - 2)/(2*x**3))/3, 0)
    >>> curve_reduce(x/sqrt(1 - x**3), x)
    (-2*sqrt(1 - x**3)/x, 2*sqrt(1 - x**3)/(x**5 - x**2))
    """
    try:
        result = integrate_on_curve(f, x)
    except ValueError:
        return None
    return result.elementary_part(), result.remainder_expr()


def curve_antiderivative(f: ExprLike, x: Symbol) -> Optional[Expr]:
    """An antiderivative of the algebraic function ``f`` by Trager's
    algorithm on the curve of its radicals (several or nested radicals,
    any index), or ``None`` when the integral is not elementary, not
    found, or ``f`` is not such a function.

    >>> from sympy import symbols, sqrt
    >>> from sympy_extras.integrals.curves import curve_antiderivative
    >>> x = symbols('x')
    >>> curve_antiderivative(1/(sqrt(x + 1) + sqrt(x - 1)), x)
    -x*sqrt(x - 1)/3 + x*sqrt(x + 1)/3 + sqrt(x - 1)/3 + sqrt(x + 1)/3
    """
    try:
        result = integrate_on_curve(f, x)
    except ValueError:
        return None
    if result.nonelementary is not False or result.remainder is not None:
        return None
    return result.elementary_part()


def is_nonelementary_on_curve(f: ExprLike, x: Symbol) -> Optional[bool]:
    """Whether the integral of the algebraic function ``f`` is not
    elementary: ``True`` when proved (a pole of order two or more at
    infinity after the Hermite reduction, or a nonzero remainder after
    the logarithmic part), ``False`` when an antiderivative is found,
    ``None`` when undecided or ``f`` is not such a function.

    >>> from sympy import symbols, sqrt
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
