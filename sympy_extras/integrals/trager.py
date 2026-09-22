r"""Trager's algorithm for the integration of algebraic functions, in the
case of a quadratic extension: integrands rational in `x` and in
`y = \sqrt{P(x)}`.

The integrand is written `A(x) + B(x) y` with `A`, `B` rational in `x`,
and `y = d w` with `w^2 = Q` squarefree (`d` collects the repeated factors
of `P`); `A` is integrated by the rational algorithms and `B d w` by the
three steps of Trager's algorithm on the curve `w^2 = Q(x)`
[Trager]_, [Bronstein]_:

1. **Hermite reduction.** The differential `M/(D w)\,dx` has its poles
   at the roots of `D`. A factor `V^k` of `D` with `V` squarefree and
   coprime to `Q` (unramified points) is reduced by subtracting the
   derivative of `S w/V^{k-1}` with `S` found modulo `V` by a Bézout
   computation, down to `k = 1`; a factor `V^k` with `V | Q` (ramified
   points, where the differential has a pole of even order and no
   residue) is reduced by the derivative of `S w/V^k` down to `k = 0`.
   Then the derivative of `S w` with `S` a polynomial lowers the degree
   of the polynomial part of the numerator below `\deg Q - 1`. What is
   left, `r/(D w)`, has simple poles at the finite points; a pole of
   order two or more at infinity (a numerator of degree at least
   `\lfloor \deg Q/2 \rfloor`) cannot be removed by any function of the
   field, so the integral is then not elementary.
2. **Residues.** The residue at the point `(a, w_a)` above a root `a` of
   `D` is `r(a)/(D'(a) w_a)`; the residues are the roots of the
   Rothstein–Trager resultant `\operatorname{Res}_x(D, r^2 - z^2 D'^2 Q)`
   and, for `\deg Q` even, the two points at infinity carry the residues
   `\mp \operatorname{lc}(r)/\sqrt{\operatorname{lc}(Q)}` when `\deg r =
   \deg Q/2 - 1`. The residues which are rational multiples of one
   another form a class with the divisor `\Delta = \sum_P (\rho_P/\rho) P`
   of degree zero.
3. **The logarithmic part.** The class contributes `(\rho/n) \log F` to
   the integral when `n \Delta` is the divisor of a function `F`; `F =
   (u + v w)/\prod G_j^{n q_j}` with `u`, `v` polynomials found by linear
   algebra: `u + v w` must vanish to order `2 n q_j` at the points with
   residue `q_j \rho` (the points above the roots of `G_j`, on the branch
   `w \equiv W_j \pmod{G_j}`) and have bounded pole orders at the points
   at infinity, which is a homogeneous linear system once `w` is lifted
   to a polynomial modulo `G_j^{2 n q_j}` by Newton's iteration and
   expanded in `1/x` at infinity. The multiples `n` of the common
   denominator of the `q_j` are tried in turn; on a curve of genus one
   over the rationals with rational residues the search stops at twelve
   times the denominator by Mazur's theorem, and its failure proves the
   integral not elementary.

Integrands with parameters go through the Hermite reduction only;
residues which are not rational multiples of one another are treated as
independent classes (which may miss an elementary integral when three
or more residues are linearly dependent over the rationals, but never
returns a wrong value: every antiderivative is verified numerically).

**Roots of index** `n \geq 3`, `y = P^{1/n}` with `P` squarefree over the
rationals, are integrated by components: `\{1, y, \ldots, y^{n-1}\}` is
an integral basis of the curve `y^n = P` (the affine curve is smooth)
and the derivative of `R y^k` stays in the component `y^k`, so `f =
\sum_k A_k y^k` splits into the differentials `A_k y^k\,dx`, each
elementary or not on its own (the automorphism `y \mapsto \zeta y`
multiplies it by `\zeta^k`). A component is first tried as a Risch
differential equation for the rational `R`; otherwise it goes through
the three steps on the curve `y^n = P` (a component with `\gcd(k, n) >
1` is one of the smaller index `n/\gcd(k, n)`, the square root case
above for `n/\gcd(k, n) = 2`):

1. **Hermite reduction.** `M y^k/(D P^j)\,dx` with a factor `V^e` of `D`
   coprime to `P` is lowered by `S y^k/V^{e-1}`, a factor `V^e` dividing
   `P` (the ramified points, where the pole has order `n e - k + 1` and
   no residue) by `S y^k/V^e` down to `e = 0`, the factor `P^j` by `S
   y^k/P^{j-1}` down to `j = 1` (where the differential has no pole at
   the ramified points), and the polynomial part `q y^k/P` by `c x^i
   y^k` while `\deg q \geq \deg P - 1`. What is left, with `\deg q \leq
   \deg P - 2`, has simple poles at the finite points; a pole of order
   two or more at a point at infinity (its order is `-e (\deg q + 1 -
   \deg P + k \deg P/n) - 1` with `e = n/\gcd(n, \deg P)`) is removed by
   no function of the field, so the integral is then not elementary.
2. **Residues.** The residue at the place `(a, y_a)` above a root `a`
   of `D` is `M(a) y_a^k/(D'(a) P(a))`, so the `n`-th powers of the
   residues are the roots of `\operatorname{Res}_x(D, \tau D'^n P^{n-k} -
   M^n)` and the residues at the `n` places above `a` are the `n`-th
   roots of one of them, the branch `y \equiv W \pmod G` of each place
   recovered from `y^k \equiv \rho D' P/M`. The `n` points at infinity
   carry residues only when `n` divides `\deg P` (`-s^k \kappa` at the
   point where `y \sim s x^{\deg P/n}`, `\kappa` a coefficient of the
   expansion of the differential in `1/x`). The number field of the
   residues (of degree at most twelve, the norms of the residues of
   degree at most two: beyond, the roots are nested radicals) is the
   field of the linear algebra.
3. **The logarithmic part.** For each orbit `Q_0, \ldots, Q_{n-1}` of
   places under `y \mapsto \zeta y` (the residues `\rho \zeta^{lk}`), a
   function `v` with divisor `m (Q_0 - \sum_l Q_l/n)` is searched for
   `m = n, 2n, \ldots` up to the torsion bound, as `U/G^{m/n}` with `U
   = \sum_i U_i(x) y^i` vanishing to order `m` on the branch of `Q_0`
   (the branch lifted modulo `G^m` by Hensel's iteration) and its poles
   at infinity bounded by the degrees of the `U_i`; the orbit
   contributes `\sum_l (\rho_l/m) \log v(x, \zeta^l y)` with `\rho_l`
   the residue where `v(x, \zeta^l y)` vanishes, the classical resolvent
   form whose real part is the answer in logarithms and arctangents.
   When the residues of different orbits are dependent over the
   rationals, or an orbit is not torsion by itself, Trager's general
   form is searched instead: over a basis `\beta_j` of the rational
   span of the residues the residue divisor is `\sum_j \beta_j D_j`
   with each `D_j` of degree zero, and `D_j` with `m_j D_j` principal
   contributes `(\beta_j/m_j) \log u_j`. The differential minus the
   logarithmic derivatives found is checked to vanish exactly on the
   curve (what would be left is a differential of the first kind, whose
   integral is not elementary); a component for which no function is
   found within the bounds is left undecided, never answered.

References
==========

.. [Trager] B. M. Trager, *Integration of algebraic functions*, PhD
   thesis, MIT, 1984, chapters 2 and 3.
.. [Bronstein] M. Bronstein, *Symbolic integration tutorial*, ISSAC
   1998, section 4 (algebraic functions); *Symbolic Integration I*,
   Springer, 2005, section 2.5 (the Hermite reduction).
.. [Davenport] J. H. Davenport, *On the integration of algebraic
   functions*, Lecture Notes in Computer Science 102, Springer, 1981.
.. [Mazur] B. Mazur, *Modular curves and the Eisenstein ideal*, Publ.
   Math. IHÉS 47 (1977), the torsion of elliptic curves over the
   rationals.
"""
from __future__ import annotations

import math
import random
from typing import Optional

from sympy.core.add import Add
from sympy.core.function import Derivative, expand, expand_mul
from sympy.core.mul import Mul
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.numbers import ImaginaryUnit, Rational
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.elementary.complexes import im, re
from sympy.functions.elementary.complexes import sign as sign_function
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.trigonometric import atan
from sympy.functions.special.delta_functions import DiracDelta
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys.domains import QQ
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.domain import Domain
from sympy.polys.matrices import DomainMatrix
from sympy.polys.polyerrors import (CoercionFailed, DomainError, GeneratorsError, NotInvertible, PolynomialError,
                                    UnificationFailed)
from sympy.polys.polyroots import roots
from sympy.polys.polytools import Poly, cancel, factor_list, resultant
from sympy.polys.polytools import factor as factored
from sympy.simplify.powsimp import powdenest
from sympy.simplify.radsimp import fraction

from sympy_extras._timeout import attempt
from sympy_extras._typing import DomainElement, ExprLike, as_expr, free_symbols, sorted_symbols
from sympy_extras.settings import settings
from .risch.rationaltools import ratint, ratint_ratpart

__all__ = ['trager_antiderivative', 'trager_reduce', 'hermite_reduce_algebraic', 'is_nonelementary_algebraic']

#: the multiples of the denominator of the residue divisor tried for the
#: torsion order: complete on a curve of genus one over the rationals
#: (Mazur: the torsion of an elliptic curve over Q has order at most 12)
_TORSION_BOUND = 12

_FAILURES = (CoercionFailed, DomainError, GeneratorsError, NotInvertible, PolynomialError, UnificationFailed,
             NotImplementedError, ValueError, TypeError, ZeroDivisionError)


class _Radical:
    """The integrand ``A + B*y`` on the curve ``y**2 == P``, with ``y = d*w``
    and ``w**2 == Q`` squarefree."""

    def __init__(self, x: Symbol, P: Poly, A: Expr, B: Expr) -> None:
        self.x = x
        self.P = P
        self.A = A
        self.B = B
        self.d, self.Q = _split_square(P)

    @property
    def w(self) -> Expr:
        """``w`` as an expression in ``x``: ``sqrt(P)/d``."""
        return as_expr(sqrt(self.P.as_expr()) / self.d.as_expr())

    def algebraic_numerator(self) -> tuple[Poly, Poly]:
        """``M, D`` with ``B*y == M/(D*w)``: ``B*y = B*d*w = B*d*Q/w``."""
        e = cancel(self.B * self.d.as_expr() * self.Q.as_expr())
        numerator, denominator = fraction(e)
        K = self.Q.domain
        M = Poly(numerator, self.x, domain=K)
        D = Poly(denominator, self.x, domain=K)
        return M, D


def _split_square(P: Poly) -> tuple[Poly, Poly]:
    """``d, Q`` with ``P == d**2 * Q`` and ``Q`` squarefree."""
    content, factors = P.sqf_list()
    d = Poly(1, P.gen, domain=P.domain)
    Q = Poly(content, P.gen, domain=P.domain)
    for factor, multiplicity in factors:
        d = d * factor**(multiplicity // 2)
        if multiplicity % 2:
            Q = Q * factor
    return d, Q


def _parse(f: Expr, x: Symbol) -> Optional[_Radical]:
    """The integrand as ``A + B*sqrt(P)`` with ``A``, ``B`` rational in
    ``x`` and ``P`` a polynomial, or ``None`` when ``f`` is not a rational
    function of ``x`` and of one square root of a polynomial."""
    bases: set[Expr] = set()
    for node in f.atoms(Pow):
        exponent = as_expr(node.exp)
        if isinstance(exponent, Rational) and exponent.q == 2:
            base = as_expr(node.base)
            if not base.has(x):
                continue
            if not base.is_polynomial(x):
                return None
            bases.add(base)
    if len(bases) > 1:
        # sqrt(1 - 3*x**2)*sqrt(1 - x**2) is sqrt((1 - 3*x**2)*(1 - x**2))
        # where both radicands are positive, the real domain of the
        # integrand (on which the antiderivative is verified): the
        # integrand must be rational in x and in the product of the roots
        combined = _combined_roots(f, x, sorted(bases, key=str))
        if combined is None:
            return None
        return _parse(combined, x)
    if len(bases) != 1:
        return None
    [base] = bases
    y = Dummy('y')

    def to_y(node: Expr) -> Expr:
        if isinstance(node, Pow) and node.base == base:
            exponent = as_expr(node.exp)
            if isinstance(exponent, Rational) and exponent.q == 2:
                return as_expr(y**int(exponent.p))
        return node

    g = as_expr(f.replace(lambda node: isinstance(node, Pow), to_y))
    if not g.is_rational_function(x, y):
        return None
    try:
        P = Poly(base, x)
    except _FAILURES:
        return None
    if P.degree() < 1:
        return None
    numerator, denominator = fraction(cancel(g))
    try:
        n0, n1 = _reduce_modulo_curve(Poly(numerator, y), P, x)
        d0, d1 = _reduce_modulo_curve(Poly(denominator, y), P, x)
    except _FAILURES:
        return None
    norm = as_expr(d0**2 - d1**2 * P.as_expr())
    if norm == 0:
        return None
    A = as_expr(cancel((n0 * d0 - n1 * d1 * P.as_expr()) / norm))
    B = as_expr(cancel((n1 * d0 - n0 * d1) / norm))
    domain = P.domain
    for e in (A, B):
        for part in fraction(e):
            domain = domain.unify(Poly(part, x).domain)
    field = domain.get_field()
    return _Radical(x, P.set_domain(field), A, B)


def _combined_roots(f: Expr, x: Symbol, bases: list[Expr]) -> Optional[Expr]:
    """``f`` with its several square roots written through the one root of
    the product of the radicands, when every monomial of ``f`` in the
    roots has either all of them or none (the parity of each exponent
    reduced by ``sqrt(P)**2 = P``); ``None`` otherwise."""
    Y = [Dummy('Y%d' % i) for i in range(len(bases))]
    replacement: dict[Expr, Expr] = {}
    for node in f.atoms(Pow):
        exponent = as_expr(node.exp)
        base = as_expr(node.base)
        if base in bases and isinstance(exponent, Rational) and exponent.q == 2:
            i = bases.index(base)
            k = int(exponent.p)
            replacement[as_expr(node)] = base**((k - k % 2) // 2) * Y[i]**(k % 2)
    g = as_expr(f.xreplace(replacement))
    if not g.is_rational_function(x, *Y):
        return None
    numerator, denominator = fraction(cancel(g))
    y = Dummy('y')
    product = as_expr(Mul(*bases))
    rewritten: list[Expr] = []
    for part in (numerator, denominator):
        try:
            poly = Poly(part, *Y)
        except _FAILURES:
            return None
        total: Expr = S.Zero
        for monomial, coefficient in poly.terms():
            parities = [e % 2 for e in monomial]
            factor: Expr = S.One
            for i, e in enumerate(monomial):
                factor = factor * bases[i]**((e - e % 2) // 2)
            if all(parities):
                total = total + as_expr(coefficient) * factor * y
            elif not any(parities):
                total = total + as_expr(coefficient) * factor
            else:
                return None
        rewritten.append(total)
    return as_expr((rewritten[0] / rewritten[1]).subs(y, sqrt(product)))


def _reduce_modulo_curve(N: Poly, P: Poly, x: Symbol) -> tuple[Expr, Expr]:
    """``n0, n1`` with ``N(y) == n0 + n1*y`` modulo ``y**2 - P``."""
    n0: Expr = S.Zero
    n1: Expr = S.Zero
    P_ = P.as_expr()
    for (k,), coefficient in N.terms():
        if k % 2 == 0:
            n0 = n0 + as_expr(coefficient) * P_**(k // 2)
        else:
            n1 = n1 + as_expr(coefficient) * P_**((k - 1) // 2)
    return as_expr(n0.expand()), as_expr(n1.expand())


class _Reduction:
    """The Hermite reduction of ``M/(D*w)`` on ``w**2 == Q``: ``terms`` are
    the pairs ``(S, T)`` of the exact part ``sum(S*w/T)``, the remainder
    ``r/(D*w)`` has ``D`` squarefree and coprime to ``Q``, and
    ``polynomial`` is the part of ``r`` of degree below ``deg Q - 1``
    left by the reduction at infinity."""

    def __init__(self, terms: list[tuple[Poly, Poly]], r: Poly, D: Poly, polynomial: Poly) -> None:
        self.terms = terms
        self.r = r
        self.D = D
        self.polynomial = polynomial


def _hermite(M: Poly, D: Poly, Q: Poly, x: Symbol) -> _Reduction:
    """Trager's Hermite reduction of ``M/(D*w)`` on the curve ``w**2 == Q``
    (``Q`` squarefree), followed by the reduction of the polynomial part
    at infinity."""
    terms: list[tuple[Poly, Poly]] = []
    one = Poly(1, x, domain=Q.domain)
    while True:
        common = M.gcd(D)
        M, D = M.exquo(common), D.exquo(common)
        lead = D.LC()
        M, D = M.quo_ground(lead), D.monic()
        step = _reduction_step(M, D, Q, x)
        if step is None:
            break
        M, D, S, T = step
        terms.append((S, T))
    # the polynomial part: p/w with p reduced below degree deg Q - 1
    p, r1 = M.div(D)
    S, r2 = _reduce_at_infinity(p, Q, x)
    if not S.is_zero:
        terms.append((S, one))
    r = r2 * D + r1
    common = r.gcd(D)
    r, D = r.exquo(common), D.exquo(common)
    return _Reduction(terms, r, D, r2)


def _reduction_step(M: Poly, D: Poly, Q: Poly, x: Symbol) -> Optional[tuple[Poly, Poly, Poly, Poly]]:
    """One step of the Hermite reduction: the factor of ``D`` of highest
    multiplicity which is ramified or multiple removed once, returning
    the new ``M``, ``D`` and the term ``S*w/T`` subtracted; ``None`` when
    ``D`` is squarefree and coprime to ``Q``."""
    _, factors = D.sqf_list()
    for V, k in sorted(factors, key=lambda pair: -pair[1]):
        ramified = V.gcd(Q)
        if ramified.degree() > 0:
            # the pole has even order 2k at the ramified points: S w/V^k
            V = ramified
            U = D.exquo(V**k)
            Q_ = Q.exquo(V)
            inverse = _inverse((U * V.diff(x) * Q_).mul_ground(Q.domain.convert(1 - 2 * k)), V)
            S = (M * inverse).mul_ground(Q.domain.convert(2)).rem(V)
            subtracted = (S.diff(x) * Q_ * V).mul_ground(Q.domain.convert(2)) + S * Q.diff(x) \
                - (S * V.diff(x) * Q_).mul_ground(Q.domain.convert(2 * k))
            M_ = (M - (U * subtracted).quo_ground(Q.domain.convert(2))).exquo(V)
            return M_, U * V**(k - 1), S, V**k
        if k >= 2:
            U = D.exquo(V**k)
            inverse = _inverse((U * V.diff(x) * Q).mul_ground(Q.domain.convert(k - 1)), V)
            S = (M * inverse).rem(V).neg()
            subtracted = (S.diff(x) * Q * V).mul_ground(Q.domain.convert(2)) + S * Q.diff(x) * V \
                - (S * V.diff(x) * Q).mul_ground(Q.domain.convert(2 * (k - 1)))
            M_ = (M - (U * subtracted).quo_ground(Q.domain.convert(2))).exquo(V)
            return M_, U * V**(k - 1), S, V**(k - 1)
    return None


def _reduce_at_infinity(p: Poly, Q: Poly, x: Symbol) -> tuple[Poly, Poly]:
    """``S, r`` with ``p/w == (S*w)' + r/w`` and ``deg r < deg Q - 1``: the
    derivative of ``c*x**j*w`` has the numerator ``c*(2*j + deg Q)*lc(Q)
    x**(j + deg Q - 1) + ...``, so the degrees are peeled one by one."""
    K = Q.domain
    S = Poly(0, x, domain=K)
    n = Q.degree()
    two = K.convert(2)
    while p.degree() >= n - 1 and not p.is_zero:
        j = p.degree() - n + 1
        c = K.from_sympy(as_expr(2 * p.LC() / (Q.LC() * (2 * j + n))))
        monomial = Poly(x**j, x, domain=Q.domain).mul_ground(c)
        S = S + monomial
        p = p - ((monomial.diff(x) * Q).mul_ground(two) + monomial * Q.diff(x)).quo_ground(two)
    return S, p


# ---------------------------------------------------------------------------
# The residues and the logarithmic part

class _ResidueClass:
    """The points whose residues are rational multiples ``q*rho`` of one
    value: for each ``q > 0`` the polynomial ``G`` whose roots are the
    abscissae, the branch ``W`` of ``w`` modulo ``G`` on which the residue
    is ``+q*rho``, and the multiple ``q_infinity`` of ``rho`` at the point
    at infinity where ``w ~ +sqrt(lc Q) x**m`` (its conjugate carries the
    opposite residue)."""

    def __init__(self, rho: Expr, finite: list[tuple[Rational, Poly, Poly]], q_infinity: Rational) -> None:
        self.rho = rho
        self.finite = finite
        self.q_infinity = q_infinity

    @property
    def denominator(self) -> int:
        return math.lcm(*[int(q.q) for q, _, _ in self.finite], int(self.q_infinity.q))


def _residue_polynomial(r: Poly, D: Poly, Q: Poly, x: Symbol, z: Symbol) -> Poly:
    """The Rothstein–Trager resultant whose roots are the residues at the
    finite points."""
    F = D.as_expr()
    G = as_expr(r.as_expr()**2 - z**2 * D.diff(x).as_expr()**2 * Q.as_expr())
    return Poly(resultant(F, G, x), z)


def _residue_values(R: Poly) -> Optional[list[Expr]]:
    """The nonzero roots of the residue polynomial in closed form, or
    ``None`` when some root has none."""
    values: list[Expr] = []
    _, factors = R.factor_list()
    for factor, _ in factors:
        if factor.degree() == 0:
            continue
        if factor.degree() == 1 and factor.eval(0) == 0:
            continue
        found = roots(factor)
        if sum(found.values()) != factor.degree():
            return None
        values.extend(_denested(as_expr(root)) for root in found)
    return [value for value in values if value != 0]


def _denested(value: Expr) -> Expr:
    """A residue with parameters with its roots of squares taken
    (``sqrt(c**2)`` is ``c``: the sign of a residue is a choice, the
    conjugate function carrying the opposite one)."""
    if not value.free_symbols:
        return value
    return as_expr(powdenest(factored(value), force=True))


def _field(values: list[Expr], base: Domain) -> Optional[Domain]:
    """The field of the residues ``values``: the field of the coefficients
    (``QQ``, or ``QQ(a, b)`` for parameters) when every value lies in it,
    the number field the values generate over ``QQ``, ``None`` for
    values algebraic over the parameters (which the caller makes
    rational by reparametrizing, :func:`_rational_parameters`)."""
    K = base.get_field()
    parameters: list[Expr] = list(K.symbols) if K.is_FractionField else []
    # the algebraic numbers in the values: sqrt(2), I of a residue I*r(a, b)
    numbers = sorted({as_expr(node) for value in values for node in value.atoms(Pow, ImaginaryUnit)
                      if not node.free_symbols
                      and (isinstance(node, ImaginaryUnit) or (isinstance(node.exp, Rational) and node.exp.q > 1))}
                     | {value for value in values if not value.free_symbols and not isinstance(value, Rational)},
                     key=str)
    try:
        K = QQ.algebraic_field(*numbers) if numbers else QQ
        if parameters:
            # QQ<I>(a, b): the residues I*r(a, b) of a parametric integrand
            K = K.frac_field(*parameters)
        for value in values:
            K.from_sympy(value)
        return K
    except _FAILURES:
        return None


def _classes(values: list[Expr], infinity: Optional[Expr], K: Domain, r: Poly, D: Poly, Q: Poly,
             x: Symbol) -> Optional[list[_ResidueClass]]:
    """The residues (``values`` at the finite points, ``infinity`` at the
    point at infinity where ``w ~ +sqrt(lc Q) x**m``) partitioned into
    classes of rational multiples, with the data of each class."""
    elements = [K.from_sympy(value) for value in values]
    representatives: list[object] = []
    members: list[list[object]] = []
    all_values = list(elements)
    if infinity is not None:
        all_values.append(K.from_sympy(infinity))
    for value in all_values:
        for k, representative in enumerate(representatives):
            ratio = K.to_sympy(value / representative)
            if isinstance(ratio, Rational):
                members[k].append(value)
                break
        else:
            representatives.append(value)
            members.append([value])
    classes: list[_ResidueClass] = []
    D_ = D.set_domain(K)
    r_ = r.set_domain(K)
    Q_ = Q.set_domain(K)
    for representative, group in zip(representatives, members):
        rho = K.to_sympy(representative)
        finite: list[tuple[Rational, Poly, Poly]] = []
        seen: set[Rational] = set()
        for value in elements:
            ratio = K.to_sympy(value / representative)
            if not isinstance(ratio, Rational):
                continue
            q = as_expr(abs(ratio))
            if not isinstance(q, Rational) or q in seen:
                continue
            seen.add(q)
            residue = value if ratio > 0 else -value
            G = D_.gcd(r_**2 - (D_.diff(x)**2 * Q_).mul_ground(residue**2))
            if G.degree() == 0:
                return None
            W = (r_ * _inverse(D_.diff(x).mul_ground(residue), G)).rem(G)
            finite.append((q, G, W))
        q_infinity: Rational = S.Zero
        if infinity is not None:
            ratio = K.to_sympy(K.from_sympy(infinity) / representative)
            if isinstance(ratio, Rational):
                q_infinity = ratio
        classes.append(_ResidueClass(rho, finite, q_infinity))
    return classes


def _inverse(p: Poly, modulus: Poly) -> Poly:
    """``p**(-1)`` modulo ``modulus``, by the extended Euclidean algorithm:
    ``Poly.invert`` reports a zero divisor over ``QQ<I>(a, b)`` (the field
    of a parametric integrand with residues ``I*r(a, b)``) where
    ``gcdex`` finds the unit."""
    s, _, h = p.gcdex(modulus)
    if h.degree() != 0:
        raise NotInvertible('zero divisor')
    return s.quo_ground(_leading(h))


def _lift(W: Poly, G: Poly, Q: Poly, N: int) -> Poly:
    """``W`` with ``W**2 == Q`` modulo ``G**N`` (Newton's iteration from
    ``W**2 == Q`` modulo ``G``)."""
    modulus = G**N
    two = Q.domain.convert(2)
    steps = max(1, (N - 1).bit_length() + 1)
    for _ in range(steps):
        W = ((W + Q * _inverse(W, modulus)).quo_ground(two)).rem(modulus)
    return W


def _series_at_infinity(Q: Poly, order: int) -> list[object]:
    """The coefficients ``sigma_0 = 1, sigma_1, ...`` of ``sqrt(Q(1/t)
    t**(deg Q) / lc(Q))`` as a power series in ``t`` up to ``t**order``."""
    K = Q.domain
    coefficients = _domain_coefficients(Q.monic())  # highest degree first
    q = coefficients + [K.zero] * max(0, order + 1 - len(coefficients))
    sigma = [Q.domain.one]
    two = Q.domain.convert(2)
    for k in range(1, order + 1):
        total = q[k] if k < len(q) else Q.domain.zero
        for i in range(1, k):
            total = total - sigma[i] * sigma[k - i]
        sigma.append(total / two)
    return sigma


def _function_with_divisor(cls: _ResidueClass, n: int, Q: Poly, s: Optional[object],
                           x: Symbol) -> Optional[tuple[Poly, Poly]]:
    """``u, v`` with ``u + v*w`` vanishing to order ``2*n*q`` at the points
    of the class with residue ``q*rho`` and with pole orders at infinity
    ``n*(sigma -+ q_infinity)``, ``sigma`` the number of finite zeros;
    ``None`` when no such function exists (``n*Delta`` is not principal
    on the curve)."""
    K = Q.domain
    deg_Q = Q.degree()
    m = deg_Q // 2
    sigma = sum(int(q * n) * G.degree() for q, G, _ in cls.finite)
    q_inf = int(cls.q_infinity * n)
    if deg_Q % 2:
        du, dv = sigma, sigma - m - 1
        E_plus = E_minus = sigma
    else:
        E_plus, E_minus = sigma - q_inf, sigma + q_inf
        E = max(E_plus, E_minus)
        du, dv = E, E - m
    if du < 0:
        return None
    nu, nv = du + 1, max(dv + 1, 0)
    unknowns = nu + nv
    rows: list[list[object]] = []
    for q, G, W in cls.finite:
        N = int(2 * q * n)
        modulus = G**N
        W_N = _lift(W, G, Q, N)
        size = modulus.degree()
        columns: list[list[object]] = []
        for i in range(nu):
            columns.append(_coefficients(Poly(x**i, x, domain=K).rem(modulus), size))
        for i in range(nv):
            columns.append(_coefficients((Poly(x**i, x, domain=K) * W_N).rem(modulus), size))
        for k in range(size):
            rows.append([column[k] for column in columns])
    if deg_Q % 2 == 0 and q_inf != 0:
        if s is None:
            return None
        order = max(0, nv - 1 + m - min(E_plus, E_minus) - 1)
        sigma_series = _series_at_infinity(Q, order)
        for sign, bound in ((K.one, E_plus), (-K.one, E_minus)):
            for j in range(bound + 1, du + 1):
                row: list[object] = [K.one if i == j else K.zero for i in range(nu)]
                for k in range(nv):
                    index = k + m - j
                    value = sigma_series[index] if 0 <= index < len(sigma_series) else K.zero
                    row.append(sign * s * value)
                rows.append(row)
    if not rows:
        vector: list[object] = [K.one] + [K.zero] * (unknowns - 1)
    else:
        matrix = DomainMatrix(rows, (len(rows), unknowns), K)
        space = matrix.nullspace()
        if space.shape[0] == 0:
            return None
        vector = list(space.to_list()[0])
    u = Poly.from_dict({(i,): vector[i] for i in range(nu)}, x, domain=K)
    v = Poly.from_dict({(i,): vector[nu + i] for i in range(nv)}, x, domain=K)
    if u.is_zero and v.is_zero:
        return None
    # the norm must be a constant times the product of the G_j**(2 n q_j)
    norm = u**2 - v**2 * Q
    expected = Poly(1, x, domain=K)
    for q, G, _ in cls.finite:
        expected = expected * G**int(2 * q * n)
    if norm.is_zero or norm.degree() != expected.degree() \
            or not (norm - expected.mul_ground(_leading(norm))).is_zero:
        return None
    return u, v


def _coefficients(p: Poly, size: int) -> list[object]:
    """The coefficients of ``p`` from degree 0 to ``size - 1``."""
    coefficients = list(reversed(_domain_coefficients(p)))
    return coefficients + [p.domain.zero] * (size - len(coefficients))


def _domain_coefficients(p: Poly) -> list[object]:
    """The coefficients of ``p`` as elements of its domain, highest degree
    first (``all_coeffs`` converts them to expressions, and converting
    them back into a number field is slow)."""
    return list(p.rep.to_list())


def _leading(p: Poly) -> object:
    """The leading coefficient of ``p`` as an element of its domain."""
    return _domain_coefficients(p)[0]


class _LogTerm:
    """``(rho/n) * log((u + v*w)/prod(G_j**(n*q_j)))``."""

    def __init__(self, rho: Expr, n: int, u: Poly, v: Poly, denominator: Poly) -> None:
        self.rho = rho
        self.n = n
        self.u = u
        self.v = v
        self.denominator = denominator

    def as_expr(self, w: Expr) -> Expr:
        argument = (self.u.as_expr() + self.v.as_expr() * w) / self.denominator.as_expr()
        return as_expr(self.rho / self.n * log(argument))


class _Pair:
    """``a0 + a1*w`` with ``a0``, ``a1`` rational functions of ``x`` on
    the curve ``w**2 == Q``."""

    def __init__(self, a0: Expr, a1: Expr, Q: Expr, x: Symbol) -> None:
        self.a0 = a0
        self.a1 = a1
        self.Q = Q
        self.x = x

    def __mul__(self, other: _Pair) -> _Pair:
        return _Pair(as_expr(cancel(self.a0 * other.a0 + self.a1 * other.a1 * self.Q)),
                     as_expr(cancel(self.a0 * other.a1 + self.a1 * other.a0)), self.Q, self.x)

    def __sub__(self, other: _Pair) -> _Pair:
        return _Pair(as_expr(cancel(self.a0 - other.a0)), as_expr(cancel(self.a1 - other.a1)), self.Q, self.x)

    def scaled(self, c: Expr) -> _Pair:
        return _Pair(as_expr(cancel(c * self.a0)), as_expr(cancel(c * self.a1)), self.Q, self.x)

    def derivative(self) -> _Pair:
        # (a1 w)' = a1' w + a1 Q'/(2 w) = (a1' + a1 Q'/(2 Q)) w
        return _Pair(as_expr(cancel(self.a0.diff(self.x))),
                     as_expr(cancel(self.a1.diff(self.x) + self.a1 * self.Q.diff(self.x) / (2 * self.Q))),
                     self.Q, self.x)

    def inverse(self) -> _Pair:
        norm = as_expr(cancel(self.a0**2 - self.a1**2 * self.Q))
        return _Pair(as_expr(cancel(self.a0 / norm)), as_expr(cancel(-self.a1 / norm)), self.Q, self.x)

    def logarithmic_derivative(self) -> _Pair:
        return self.derivative() * self.inverse()

    def is_zero(self) -> bool:
        return self.a0 == 0 and self.a1 == 0


class _Result:
    """The pieces of the integral of ``A + B*y``."""

    def __init__(self, radical: _Radical, rational_part: Expr, rational_logs: Expr, reduction: _Reduction,
                 logs: list[_LogTerm], remainder: _Pair, nonelementary: Optional[bool],
                 generators: Optional[list[Expr]] = None) -> None:
        self.radical = radical
        self.rational_part = rational_part
        self.rational_logs = rational_logs
        self.reduction = reduction
        self.logs = logs
        self.remainder = remainder
        self.nonelementary = nonelementary
        #: the residues algebraic over the parameters which left the
        #: logarithmic part undecided
        self.generators = list(generators or [])

    def exact_part(self) -> Expr:
        w = self.radical.w
        return as_expr(self.rational_part
                       + Add(*[S.as_expr() * w / T.as_expr() for S, T in self.reduction.terms]))

    def elementary(self) -> Expr:
        return as_expr(self.exact_part() + self.rational_logs + Add(*[term.as_expr(self.radical.w) for term in self.logs]))

    def remainder_expr(self) -> Expr:
        # a1 w = a1 Q d / sqrt(P): the radical in the denominator
        radical = self.radical
        return as_expr(self.remainder.a0 + cancel(self.remainder.a1 * radical.Q.as_expr()) * radical.d.as_expr()
                       / sqrt(radical.P.as_expr()))


def _integrate(f: Expr, x: Symbol, logarithms: bool) -> Optional[_Result]:
    radical = _parse(f, x)
    if radical is None:
        return None
    Q = radical.Q
    A = radical.A
    numerator, denominator = fraction(cancel(A))
    if A == 0:
        rational_part: Expr = S.Zero
        rational_logs: Expr = S.Zero
    elif logarithms:
        rational_part, rational_logs = ratint(A, x), S.Zero
    else:
        p, q = Poly(numerator, x), Poly(denominator, x)
        quotient, rest = p.div(q)
        rational_part, rational_logs = ratint_ratpart(rest, q, x)
        rational_part = as_expr(rational_part + quotient.integrate(x).as_expr())
    if Q.degree() == 0:
        # sqrt(P) is d*sqrt(c) where d > 0 and -d*sqrt(c) where d < 0: a
        # rational integrand on each component, the antiderivative
        # carrying sign(d) ((b**2/(4*c) + b*x + c*x**2)**(-3/2) is
        # c**(-3/2)*Abs(x + b/(2*c))**(-3))
        constant = as_expr(sqrt(Q.as_expr()))
        g = as_expr(radical.B * radical.d.as_expr() * constant)
        empty = _Reduction([], Poly(0, x, domain=Q.domain), Poly(1, x, domain=Q.domain), Poly(0, x, domain=Q.domain))
        extra = ratint(g, x) if logarithms else S.Zero
        orientation = sign_function(radical.d.as_expr()) if radical.d.degree() >= 1 else S.One
        remainder = _Pair(S.Zero if logarithms else g * orientation, S.Zero, Q.as_expr(), x)
        return _Result(radical, as_expr(rational_part + orientation * extra), rational_logs, empty, [], remainder,
                       False)
    M, D = radical.algebraic_numerator()
    reduction = _hermite(M, D, Q, x)
    r, D = reduction.r, reduction.D
    Q_expr = Q.as_expr()
    remainder = _Pair(S.Zero, as_expr(cancel(r.as_expr() / (D.as_expr() * Q_expr))), Q_expr, x)
    m = Q.degree() // 2
    if reduction.polynomial.degree() >= m and m >= 1:
        # a pole of order at least two at infinity which no function of
        # the field removes: not elementary
        return _Result(radical, rational_part, rational_logs, reduction, [], remainder, True)
    if not logarithms or r.is_zero:
        return _Result(radical, rational_part, rational_logs, reduction, [], remainder, False if r.is_zero else None)
    logs, remainder, nonelementary, generators = _logarithmic_part(reduction, Q, x, remainder)
    return _Result(radical, rational_part, rational_logs, reduction, logs, remainder, nonelementary, generators)


def _logarithmic_part(reduction: _Reduction, Q: Poly, x: Symbol,
                      remainder: _Pair) -> tuple[list[_LogTerm], _Pair, Optional[bool], list[Expr]]:
    """The logarithmic terms of the classes of residues whose divisors
    are torsion, the remainder ``h - sum(dlog)``, the decision, and the
    residues algebraic over the parameters when they left it undecided."""
    r, D = reduction.r, reduction.D
    z = Dummy('z')
    R = _residue_polynomial(r, D, Q, x, z)
    values = _residue_values(R)
    if values is None:
        return [], remainder, None, []
    m = Q.degree() // 2
    infinity: Optional[Expr] = None
    s_value: Optional[Expr] = None
    if Q.degree() % 2 == 0:
        s_value = _denested(as_expr(sqrt(as_expr(Q.LC()))))
        if reduction.polynomial.degree() == m - 1:
            infinity = as_expr(-reduction.polynomial.LC() / s_value)
    if not values and infinity is None:
        # no residue anywhere: the remainder, with simple poles only, is a
        # differential of the first kind, whose integral is elementary only
        # when it is zero (its integral would be a function of the field
        # without poles, a constant)
        return [], remainder, not remainder.is_zero(), []
    generators = list(values)
    if infinity is not None:
        generators.append(infinity)
    if s_value is not None:
        generators.append(s_value)
    K = _field(generators, Q.domain)
    if K is None:
        return [], remainder, None, [value for value in generators if value.free_symbols]
    classes = _classes(values, infinity, K, r, D, Q, x)
    if classes is None:
        return [], remainder, None, []
    Q_K = Q.set_domain(K)
    s = K.from_sympy(s_value) if s_value is not None else None
    genus_one = Q.degree() in (3, 4)
    logs: list[_LogTerm] = []
    undecided = False
    for cls in classes:
        found = None
        n0 = cls.denominator
        for k in range(1, _TORSION_BOUND + 1):
            n = k * n0
            found = _function_with_divisor(cls, n, Q_K, s, x)
            if found is not None:
                break
        if found is None:
            if not (genus_one and K.is_QQ):
                undecided = True
            continue
        u, v = found
        rho = cls.rho
        # (u + v w)(u - v w) is a constant times the denominator squared:
        # the conjugate gives the reciprocal function and the opposite
        # residue; the sign with positive leading coefficients is taken
        if not v.is_zero and as_expr(u.LC() * v.LC()).is_negative is True:
            v, rho = v.neg(), as_expr(-rho)
        if as_expr(u.LC()).is_negative is True:
            u, v = u.neg(), v.neg()
        denominator = Poly(1, x, domain=K)
        for q, G, _ in cls.finite:
            denominator = denominator * G**int(q * n)
        term = _LogTerm(rho, n, u, v, denominator)
        logs.append(term)
        F = _Pair(as_expr(u.as_expr() / denominator.as_expr()), as_expr(v.as_expr() / denominator.as_expr()),
                  Q.as_expr(), x)
        remainder = remainder - F.logarithmic_derivative().scaled(as_expr(rho / n))
    if remainder.is_zero():
        return logs, remainder, False, []
    return logs, remainder, None if undecided else True, []


def _verified(F: Expr, f: Expr, x: Symbol, P: Expr) -> bool:
    """Whether ``F'`` agrees with ``f`` numerically at points where the
    radicand is positive."""
    # sign(d) in F differentiates to a delta function, which is 0 at every
    # sample point off the zeros of d
    derivative = as_expr(F.diff(x).replace(
        lambda node: isinstance(node, DiracDelta) or (isinstance(node, Derivative) and isinstance(node.expr, sign_function)),
        lambda node: S.Zero))
    rng = random.Random(str(f))
    parameters = sorted(f.free_symbols - {x}, key=str)
    # every square root of the integrand as given must be real at the
    # sample (sqrt(x + 1)*sqrt(x + 2) is -sqrt((x + 1)*(x + 2)) below -2)
    radicands = [P] + [as_expr(node.base) for node in f.atoms(Pow)
                       if node.has(x) and isinstance(node.exp, Rational) and node.exp.q == 2]
    checked = 0
    for _ in range(40):
        values: dict[Basic | complex, Expr | complex] = {
            parameter: Rational(rng.randint(-400, 400), 100) for parameter in parameters}
        values[x] = Rational(rng.randint(-400, 400), 100)
        try:
            if any(not as_expr(radicand.subs(values)).evalf(30).is_positive for radicand in radicands):
                continue
            expected = f.subs(values).evalf(30)
            found = derivative.subs(values).evalf(30)
        except (ValueError, TypeError, ZeroDivisionError):
            continue
        if not (expected.is_number and found.is_number) or expected.has(S.NaN, S.ComplexInfinity):
            continue
        if not found.is_finite or not expected.is_finite:
            continue
        if abs(found - expected) > 1e-20 * (1 + abs(expected)):
            return False
        checked += 1
        if checked >= 4:
            return True
    return checked > 0


def trager_reduce(f: ExprLike, x: Symbol) -> Optional[tuple[Expr, Expr]]:
    """``(F, h)`` with ``f == F' + h``: ``F`` the elementary part of the
    integral of the algebraic function ``f`` (the integral of the
    rational part, the exact part of the Hermite reduction and the
    logarithms of the residue classes whose divisors are torsion) and
    ``h`` the remaining integrand, with simple poles only, whose integral
    is not elementary or not found; ``None`` when ``f`` is not a rational
    function of ``x`` and of one square root of a polynomial.

    >>> from sympy import symbols, sqrt
    >>> from sympy_extras.integrals.trager import trager_reduce
    >>> x = symbols('x')
    >>> F, h = trager_reduce((x**3 + 1)/((x - 1)**2*sqrt(x**3 + 1)), x)
    >>> F
    -sqrt(x**3 + 1)/(x - 1)
    >>> h
    3*x**2/((2*x - 2)*sqrt(x**3 + 1))
    >>> trager_reduce(1/sqrt(x**3 + 1), x)
    (0, 1/sqrt(x**3 + 1))
    >>> trager_reduce((2*x**2 + 1)/((x**2 + 1)*sqrt(x**4 + x**2 + 1)), x)
    (-I*log((-I*x + sqrt(x**4 + x**2 + 1))/(x**2 + 1))/2, 3/(2*sqrt(x**4 + x**2 + 1)))
    """
    f_ = as_expr(f)
    g, rational_roots = _polynomial_radicands(f_, x)
    result = attempt(lambda: _integrate(g, x, True), settings.timeout)
    if result is None:
        return attempt(lambda: _index_n(f_, x), settings.timeout)
    back: dict[Symbol, Expr] = {}
    original = result.radical.P.as_expr()
    if result.nonelementary is None and result.generators:
        # residues algebraic over the parameters: the parameters made
        # rational (a = delta**2/c for sqrt(a*c)) and the algorithm rerun
        made = _rational_parameters(result.generators, sorted_symbols(free_symbols(g) - {x}))
        if made is not None:
            substitution, back = made
            g_ = as_expr(g.xreplace(substitution))
            again = attempt(lambda: _integrate(g_, x, True), settings.timeout)
            if again is not None and again.nonelementary is not None:
                result, g = again, g_
                rational_roots = [(as_expr(expand(product.xreplace(substitution))), as_expr(y.xreplace(substitution)))
                                  for product, y in rational_roots]
            else:
                back = {}
    P = result.radical.P.as_expr()
    F = _restored(_original_roots(as_expr(result.elementary()), g, x, P), rational_roots)
    h = _restored(_original_roots(result.remainder_expr(), g, x, P), rational_roots)
    if back:
        # the radicand in the original parameters again (a + b*x**2 + c*x**4,
        # not the square of the reparametrization expanded)
        P_ = as_expr(P.xreplace(back))
        F, h = as_expr(F.xreplace(back)), as_expr(h.xreplace(back))
        F, h, P = _restored(F, [(expand(P_), sqrt(original))]), _restored(h, [(expand(P_), sqrt(original))]), original
    if not _verified(F, as_expr(f_ - h), x, P):
        return None
    return F, h


def _index_n(f: Expr, x: Symbol) -> Optional[tuple[Expr, Expr]]:
    """``(F, h)`` for ``f`` rational in ``x`` and in one root ``y = P**(1/n)``
    of index ``n >= 3`` of a squarefree polynomial: ``f`` is
    ``sum(A_k*y**k)`` in the integral basis ``1, y, ..., y**(n - 1)``, and
    the derivative of ``R*y**k`` stays in the component ``y**k`` (``y' =
    P'*y/(n*P)``), so each component integrates on its own, ``R`` a
    rational solution of the Risch differential equation ``R' +
    k*P'/(n*P)*R = A_k``, or, without one, the Hermite reduction and the
    logarithmic part on the curve ``y**n == P``
    (:func:`_component_with_logarithms`); ``h`` collects the components
    left undecided."""
    from .risch.risch import DifferentialExtension, NonElementaryIntegralException
    from .risch.rde import rischDE
    parsed = _root_components(f, x)
    if parsed is None:
        return None
    P, n, components = parsed
    F: Expr = ratint(components[0], x) if components[0] != 0 else S.Zero
    h: Expr = S.Zero
    DE = DifferentialExtension(extension={'D': [Poly(1, x)]})
    P_ = P.as_expr()
    for k in range(1, n):
        A = components[k]
        if A == 0:
            continue
        fa, fd = fraction(cancel(k * P.diff(x).as_expr() / (n * P_)))
        ga, gd = fraction(cancel(A))
        try:
            ya, yd = rischDE(Poly(fa, x), Poly(fd, x), Poly(ga, x), Poly(gd, x), DE)
        except (NonElementaryIntegralException, NotImplementedError):
            # no rational solution: the logarithmic part on the curve
            found = _component_with_logarithms(A, P, n, k, x)
            if found is None:
                h = h + A * P_**Rational(k, n)
            else:
                F = F + found
            continue
        F = F + as_expr(cancel(ya.as_expr() / yd.as_expr())) * P_**Rational(k, n)
    if not _verified(F, as_expr(f - h), x, P_):
        return None
    return as_expr(F), as_expr(h)


# ---------------------------------------------------------------------------
# Roots of index n >= 3: the logarithmic part on the curve y**n = P

#: the largest degree over the rationals of the field of the residues on a
#: curve ``y**n = P`` for which the logarithmic part is searched (the
#: linear algebra runs over that field)
_INDEX_N_FIELD_DEGREE = 12

#: the time allowed to the construction of that field (the primitive
#: element of nested radicals may not come within the limit of the
#: integral)
_FIELD_SECONDS = 2.0


def _component_with_logarithms(A: Expr, P: Poly, n: int, k: int, x: Symbol) -> Optional[Expr]:
    """An antiderivative of the component ``A*y**k`` (``A`` rational in
    ``x``, ``y**n == P`` squarefree, ``1 <= k < n``) with logarithms of
    functions of the curve, verified by differentiation; ``None`` when it
    is not found. A component with ``gcd(k, n) > 1`` is one of the index
    ``n/gcd(k, n)`` (``A*y**2`` on ``y**4 == P`` is ``A*sqrt(P)``)."""
    g = math.gcd(k, n)
    if g > 1:
        n, k = n // g, k // g
    if n == 2:
        result = _integrate(as_expr(A * sqrt(P.as_expr())), x, True)
        if result is None or not result.remainder.is_zero():
            return None
        return result.elementary()
    if P.domain.is_FractionField or not P.domain.is_QQ and not P.domain.is_ZZ or free_symbols(A) - {x}:
        # the residues of a parametric integrand lie in an algebraic
        # extension of the field of the parameters: not this route
        return None
    P = P.set_domain(QQ)
    reduced = _hermite_index_n(A, P, n, k, x)
    if reduced is None:
        return None
    terms, M, D = reduced
    y = P.as_expr()**Rational(1, n)
    exact: Expr = Add(*[S_.as_expr() / T.as_expr() * y**k for S_, T in terms])
    if M.is_zero:
        found: Expr = exact
    else:
        logs = _logarithms_index_n(M, D, P, n, k, x)
        if logs is None:
            return None
        found = as_expr(exact + logs)
    if not _verified(found, as_expr(A * y**k), x, P.as_expr()):
        return None
    return found


def _hermite_index_n(A: Expr, P: Poly, n: int, k: int, x: Symbol) -> Optional[tuple[list[tuple[Poly, Poly]], Poly, Poly]]:
    """The Hermite reduction of ``A*y**k dx`` on ``y**n == P``: the terms
    ``(S, T)`` of the exact part ``sum(S*y**k/T)`` and ``M, D`` with
    ``A*y**k == sum((S*y**k/T)') + M*y**k/(D*P)``, ``D`` squarefree and
    coprime to ``P`` (the remainder has simple poles at the finite
    points) and the polynomial part of ``M/D`` of degree below ``deg P -
    1``; ``None`` when the remainder keeps a pole of order two or more at
    infinity, which no function of the curve removes (the integral is
    then not elementary).

    The multiple factor ``V**e`` of the denominator coprime to ``P`` is
    lowered by ``S*y**k/V**(e - 1)``, the factor ``P**j`` (``j >= 2``, the
    ramified points, where a pole of ``P`` of order one is no pole of the
    differential) by ``S*y**k/P**(j - 1)``, and the polynomial part by
    ``c*x**i*y**k``, whose derivative has the numerator ``c*(i + k*deg
    P/n)*lc(P)*x**(i + deg P - 1) + ...`` over ``P``.
    """
    K = P.domain
    numerator, denominator = fraction(cancel(A))
    M = Poly(numerator, x, domain=K)
    D = Poly(denominator, x, domain=K)
    j = 0
    while True:
        quotient, remainder = D.div(P)
        if not remainder.is_zero:
            break
        D, j = quotient, j + 1
    if j == 0:
        M, j = M * P, 1
    terms: list[tuple[Poly, Poly]] = []
    kn = K.from_sympy(Rational(k, n))
    while j >= 2:
        c = K.from_sympy(Rational(j - 1) - Rational(k, n))
        S_ = (M * _inverse((D * P.diff(x)).mul_ground(c), P)).rem(P).neg()
        M = (M - D * (S_.diff(x) * P - (S_ * P.diff(x)).mul_ground(c))).exquo(P)
        terms.append((S_, P**(j - 1)))
        j -= 1
    while True:
        common = M.gcd(D)
        M, D = M.exquo(common), D.exquo(common)
        _, factors = D.sqf_list()
        ramified = [(V.gcd(P), e) for V, e in factors if V.gcd(P).degree() > 0]
        if ramified:
            # a factor V**e of D dividing P: with P = V*P1 the differential
            # is M*y**k/(V**(e + 1)*U*P1), a pole of order n*e - k + 1 at
            # the ramified points, removed down to e = 0 by S*y**k/V**e
            V, e = ramified[0]
            U = D.exquo(V**e)
            P1 = P.exquo(V)
            c = K.from_sympy(Rational(e) - Rational(k, n))
            S_ = (M * _inverse((U * V.diff(x) * P1).mul_ground(c), V)).rem(V).neg()
            subtracted = S_.diff(x) * V * P1 + (S_ * V * P1.diff(x)).mul_ground(kn) - (S_ * V.diff(x) * P1).mul_ground(c)
            M = (M - U * subtracted).exquo(V)
            D = U * V**(e - 1)
            terms.append((S_, V**e))
            continue
        multiple = [(V, e) for V, e in factors if e >= 2]
        if not multiple:
            break
        V, e = max(multiple, key=lambda pair: pair[1])
        U = D.exquo(V**e)
        c = K.convert(e - 1)
        S_ = (M * _inverse((U * V.diff(x) * P).mul_ground(c), V)).rem(V).neg()
        subtracted = S_.diff(x) * V * P - (S_ * V.diff(x) * P).mul_ground(c) + (S_ * P.diff(x) * V).mul_ground(kn)
        M = (M - U * subtracted).exquo(V)
        D = U * V**(e - 1)
        terms.append((S_, V**(e - 1)))
    q, r = M.div(D)
    d = P.degree()
    S_ = Poly(0, x, domain=K)
    while not q.is_zero and q.degree() >= d - 1:
        i = q.degree() - d + 1
        c = K.from_sympy(as_expr(q.LC() / (P.LC() * (i + Rational(k * d, n)))))
        monomial = Poly(x**i, x, domain=K).mul_ground(c)
        S_ = S_ + monomial
        q = q - (monomial.diff(x) * P + (monomial * P.diff(x)).mul_ground(kn))
    if not S_.is_zero:
        terms.append((S_, Poly(1, x, domain=K)))
    # the order of q*y**k/P dx at a point at infinity is
    # -e*(deg q + 1 - d + k*d/n) - 1, e = n/gcd(n, d): a pole of order two
    # or more when n*(deg q + 1 - d) + k*d >= gcd(n, d)
    if not q.is_zero and n * (q.degree() + 1 - d) + k * d >= math.gcd(n, d):
        return None
    M = q * D + r
    common = M.gcd(D)
    M, D = M.exquo(common), D.exquo(common)
    return terms, M, D


class _Places:
    """The places of the curve ``y**n == P`` above the roots of ``G``
    (unramified: ``G`` coprime to ``P``) on the branch ``y == W`` modulo
    ``G``, where the differential has the residue ``rho``; ``group`` is
    the index of the norm ``rho**n`` of the residue (the places above
    the same roots on the ``n`` branches share it)."""

    def __init__(self, G: Poly, W: Poly, rho: DomainElement, group: int, P: Poly, n: int) -> None:
        self.G = G
        self.W = W
        self.rho = rho
        self.group = group
        self.P = P
        self.n = n
        #: the branch lifted modulo G**N with the inverse of n*W**(n - 1)
        #: there, for the largest N computed
        self.lifted: tuple[int, Poly, Poly] = (1, W, _inverse((W**(n - 1)).mul_ground(P.domain.convert(n)), G))

    def branch(self, N: int) -> Poly:
        """``W`` with ``W**n == P`` modulo ``G**N``, by Hensel's lifting
        with the precision doubled at each step (``W - (W**n - P)*I`` and
        ``I*(2 - n*W**(n - 1)*I)`` modulo ``G**(2*N)``), continued from
        the largest precision reached before."""
        precision, W, I_ = self.lifted
        K = self.P.domain
        n_ = K.convert(self.n)
        two = Poly(2, self.G.gen, domain=K)
        while precision < N:
            precision = min(2 * precision, N)
            modulus = self.G**precision
            W = (W - ((W**self.n - self.P) * I_).rem(modulus)).rem(modulus)
            I_ = (I_ * (two - ((W**(self.n - 1)).mul_ground(n_) * I_).rem(modulus))).rem(modulus)
        self.lifted = (precision, W, I_)
        return W.rem(self.G**N)


class _PlaceAtInfinity:
    """The point at infinity of the curve ``y**n == P`` with ``n`` dividing
    ``deg P`` where ``y ~ s*x**(deg P/n)``, ``s`` an ``n``-th root of the
    leading coefficient, and the residue ``rho`` there."""

    def __init__(self, s: DomainElement, rho: DomainElement) -> None:
        self.s = s
        self.rho = rho


def _logarithms_index_n(M: Poly, D: Poly, P: Poly, n: int, k: int, x: Symbol) -> Optional[Expr]:
    """The logarithmic part ``sum(beta_j/m_j*log(u_j))`` of the reduced
    differential ``M*y**k/(D*P) dx`` on ``y**n == P`` (``D`` squarefree and
    coprime to ``P``, simple poles at the finite points, ``gcd(k, n) =
    1``), or ``None`` when it is not found.

    The residue at the place ``(a, y_a)`` above a root ``a`` of ``D`` is
    ``M(a)*y_a**k/(D'(a)*P(a))``, so the norms ``rho**n`` of the residues
    are the roots of ``Res_x(D, tau*D'**n*P**(n - k) - M**n)`` and the
    residues are their ``n``-th roots; the points at infinity carry a
    residue only when ``n`` divides ``deg P``. Each orbit of places under
    ``y -> zeta*y`` is tried first with a function of divisor ``m*(Q_0 -
    sum(Q_l)/n)`` (:func:`_orbit_logarithms`, the resolvent form); when
    an orbit is not torsion by itself or the residues of the orbits are
    dependent over the rationals, Trager's general form
    (:func:`_basis_logarithms`): over a basis ``beta_j`` of the rational
    span of the residues the residue divisor is ``sum(beta_j D_j)`` with
    ``D_j`` of degree zero, and each ``D_j`` with ``m_j*D_j`` principal
    (``m_j`` searched up to the torsion bound) contributes
    ``beta_j/m_j*log(u_j)``. The differential minus the logarithmic
    derivatives found is checked to vanish exactly, which it must when
    the integral is elementary (what is left is a differential of the
    first kind).
    """
    d = P.degree()
    at_infinity = n * ((d // n) * n == d)
    tau = Dummy('tau')
    R = Poly(resultant(D.as_expr(), tau * D.diff(x).as_expr()**n * P.as_expr()**(n - k) - M.as_expr()**n, x), tau)
    if R.degree() >= 1 and max(factor.degree() for factor, _ in R.factor_list()[1]) > 2:
        # the n-th roots of the roots of a cubic are nested radicals of
        # Cardano's formula: their number field is beyond the bound below
        # and its construction does not finish
        return None
    norms = _residue_values(R) if R.degree() >= 1 else []
    if norms is None:
        return None
    z = Dummy('z')
    zeta = _primitive_root_of_unity(n, z)
    if zeta is None:
        return None
    generators: list[Expr] = [zeta]
    principal_roots: list[Expr] = []
    for value in norms:
        root = _preferably_real_root(z**n - value, n, z)
        if root is None:
            return None
        principal_roots.append(root)
        generators.append(root)
    kappa = _residue_at_infinity(M, D, P, n, k) if at_infinity else P.domain.zero
    s0: Optional[Expr] = None
    if kappa != 0:
        s0 = _preferably_real_root(z**n - P.LC(), n, z)
        if s0 is None:
            return None
        generators.append(s0)
    K = attempt(lambda: QQ.algebraic_field(*sorted({g for g in generators if not isinstance(g, Rational)}, key=str)),
                _FIELD_SECONDS)
    if K is None or K.mod.degree() > _INDEX_N_FIELD_DEGREE:
        return None
    M_, D_, P_ = M.set_domain(K), D.set_domain(K), P.set_domain(K)
    zeta_ = K.from_sympy(zeta)
    a, b = _bezout(k, n)
    finite: list[_Places] = []
    for group, (value, root) in enumerate(zip(norms, principal_roots)):
        G = D_.gcd(M_**n - (D_.diff(x)**n * P_**(n - k)).mul_ground(K.from_sympy(value)))
        if G.degree() == 0:
            return None
        rho = K.from_sympy(root)
        for _ in range(n):
            # y**k == rho*D'*P/M on the branch with residue rho, and y is
            # (y**k)**a*P**b with a*k + b*n == 1
            W_k = (D_.diff(x) * P_ * _inverse(M_, G)).mul_ground(rho).rem(G)
            W = (_power_modulo(W_k, a, G) * _power_modulo(P_, b, G)).rem(G)
            finite.append(_Places(G, W, rho, group, P_, n))
            rho = rho * zeta_
    infinite: list[_PlaceAtInfinity] = []
    if s0 is not None:
        s = K.from_sympy(s0)
        for _ in range(n):
            infinite.append(_PlaceAtInfinity(s, -s**k * K.convert(kappa)))
            s = s * zeta_
    if not finite and not infinite:
        return None
    # the orbit form first (a function of divisor m*(n*Q - sum of the
    # conjugates of Q) for each orbit of places, which has a plain
    # representative), the basis form when an orbit is not torsion by
    # itself or the residues of different orbits are dependent
    logs = _orbit_logarithms(finite, infinite, zeta_, P_, n, x)
    if logs is None or not _logarithmic_derivative_matches(logs, M_, D_, P_, n, k, x):
        # with a single orbit the two searches are equivalent (the
        # differences of its places are torsion when their sum against
        # n times one of them is)
        orbits = len({place.group for place in finite}) + (1 if infinite else 0)
        if orbits == 1:
            return None
        logs = _basis_logarithms(finite, infinite, K, P_, n, x)
        if logs is None or not _logarithmic_derivative_matches(logs, M_, D_, P_, n, k, x):
            return None
    return _real_logarithms(logs, P, n, x)


def _orbit_logarithms(finite: list[_Places], infinite: list[_PlaceAtInfinity], zeta: DomainElement, P: Poly,
                      n: int, x: Symbol) -> Optional[list[_LogTermN]]:
    """The logarithmic terms ``rho_l/m*log(v(x, zeta**l*y))`` over each
    orbit ``Q_0, ..., Q_(n-1)`` of places under ``y -> zeta*y`` (the
    ``n`` places above the same roots, or the ``n`` points at infinity),
    with ``v`` of divisor ``m*(Q_0 - sum(Q_l)/n)`` and ``rho_l`` the
    residue at the place where ``v(x, zeta**l*y)`` vanishes; ``None``
    when no such ``v`` is found for an orbit within the torsion bound."""
    logs: list[_LogTermN] = []
    orbits: list[tuple[list[_Places], list[_PlaceAtInfinity]]] = []
    for group in sorted({place.group for place in finite}):
        orbits.append(([place for place in finite if place.group == group], []))
    if infinite:
        orbits.append(([], infinite))
    coefficients = [Rational(n - 1, n)] + [Rational(-1, n)] * (n - 1)
    for places, points in orbits:
        found = None
        m = n
        for multiple in range(1, _TORSION_BOUND + 1):
            m = multiple * n
            found = _function_with_divisor_n(places, coefficients if places else [], points,
                                             coefficients if points else [], m, P, n, x)
            if found is not None:
                break
        if found is None:
            return None
        V, denominator = found
        power = 1
        if places:
            V, power = _root_on_the_curve(V, places[0], m, P, n, x)
        for l in range(n):
            # v(x, zeta**l*y) vanishes on the branch zeta**(-l)*W_0, at the
            # point at infinity with s = zeta**(-l)*s_0
            U = [V_i.mul_ground(zeta**(l * i)) for i, V_i in enumerate(V)]
            rho: Optional[DomainElement] = None
            if places:
                W = places[0].W.mul_ground(zeta**((n - l) % n)).rem(places[0].G)
                rho = next((place.rho for place in places if (place.W - W).is_zero), None)
            else:
                s = points[0].s * zeta**((n - l) % n)
                rho = next((point.rho for point in points if point.s == s), None)
            if rho is None:
                return None
            logs.append(_LogTermN(rho, m, U, denominator, power))
    return logs


def _root_on_the_curve(V: list[Poly], place: _Places, m: int, P: Poly, n: int,
                       x: Symbol) -> tuple[list[Poly], int]:
    """``w, mu`` with ``V == c*w**mu`` on the curve for the largest ``mu``
    dividing ``m`` (``V`` vanishes to order ``m`` at the places of
    ``place`` and has its poles at infinity, so ``w`` vanishes there to
    order ``m/mu`` with a ``mu``-th of the poles: the same linear
    system), or ``V, 1``: ``(y - 1)**3`` on ``y**3 = x**2 + 1`` is written
    ``x**2 - 3*y**2 + 3*y`` in the basis ``1, y, y**2``."""
    d = P.degree()
    g = math.gcd(n, d)
    B = (n // g) * (m // n) * place.G.degree()
    for mu in sorted((mu for mu in range(2, m + 1) if m % mu == 0 and B % mu == 0), reverse=True):
        found = _function_with_divisor_n([place], [S.One], [], [], m // mu, P, n, x, B // mu)
        if found is None:
            continue
        w, _ = found
        power = _power_on_the_curve(w, mu, P, n, x)
        pairs = [(a, b) for a, b in zip(V, power) if not a.is_zero or not b.is_zero]
        if pairs and all(not a.is_zero and not b.is_zero for a, b in pairs) \
                and all((a * pairs[0][1] - b * pairs[0][0]).is_zero for a, b in pairs):
            return w, mu
    return V, 1


def _power_on_the_curve(w: list[Poly], mu: int, P: Poly, n: int, x: Symbol) -> list[Poly]:
    """``w**mu`` in the basis ``1, y, ..., y**(n - 1)`` of the curve
    ``y**n == P``."""
    K = P.domain
    result = [Poly(1, x, domain=K)] + [Poly(0, x, domain=K)] * (n - 1)
    for _ in range(mu):
        product = [Poly(0, x, domain=K) for _ in range(2 * n - 1)]
        for i, a in enumerate(result):
            for j, b in enumerate(w):
                product[i + j] = product[i + j] + a * b
        for i in range(2 * n - 2, n - 1, -1):
            product[i - n] = product[i - n] + product[i] * P
        result = product[:n]
    return result


def _basis_logarithms(finite: list[_Places], infinite: list[_PlaceAtInfinity], K: AlgebraicField, P: Poly, n: int,
                      x: Symbol) -> Optional[list[_LogTermN]]:
    """The logarithmic terms ``beta_j/m_j*log(u_j)`` over a basis
    ``beta_j`` of the rational span of the residues, ``u_j`` of divisor
    ``m_j*D_j`` with ``D_j`` the divisor of the coordinates of the
    residues on ``beta_j``; ``None`` when a ``D_j`` is not found
    principal within the torsion bound."""
    residues = [place.rho for place in finite] + [place.rho for place in infinite]
    basis = _rational_basis(residues, K)
    if basis is None:
        return None
    chosen, coordinates = basis
    logs: list[_LogTermN] = []
    for j, index in enumerate(chosen):
        c_finite = [row[j] for row in coordinates[:len(finite)]]
        c_infinite = [row[j] for row in coordinates[len(finite):]]
        n0 = math.lcm(*[int(c.q) for c in c_finite + c_infinite])
        found = None
        m = n0
        for multiple in range(1, _TORSION_BOUND + 1):
            m = multiple * n0
            found = _function_with_divisor_n(finite, c_finite, infinite, c_infinite, m, P, n, x)
            if found is not None:
                break
        if found is None:
            return None
        U, denominator = found
        logs.append(_LogTermN(residues[index], m, U, denominator))
    return logs


class _LogTermN:
    """``(beta/m)*(power*log(U) - log(denominator))`` with ``U =
    sum(U[i]*y**i)`` a function of the curve ``y**n == P`` and
    ``denominator`` a polynomial in ``x``, over the field of the
    residues."""

    def __init__(self, beta: DomainElement, m: int, U: list[Poly], denominator: Poly, power: int = 1) -> None:
        self.beta = beta
        self.m = m
        self.U = U
        self.denominator = denominator
        self.power = power


def _preferably_real_root(p: Expr, n: int, z: Symbol) -> Optional[Expr]:
    """A root of the binomial ``p`` of degree ``n`` in ``z`` in closed
    form, a real one when there is one (the place of an orbit where the
    function found is real), or ``None`` when the roots have no closed
    form."""
    found = roots(Poly(p, z))
    if sum(found.values()) != n:
        return None
    candidates = sorted((as_expr(root) for root in found), key=str)
    for root in candidates:
        if root.is_real is True:
            return root
    return candidates[0]


def _primitive_root_of_unity(n: int, z: Symbol) -> Optional[Expr]:
    """A primitive ``n``-th root of unity in closed form, or ``None``."""
    found = roots(Poly(z**n - 1, z))
    if sum(found.values()) != n:
        return None
    divisors = [d for d in range(1, n) if n % d == 0]
    for root in sorted((as_expr(root) for root in found), key=str):
        if all(abs(as_expr(root**d).evalf(30) - 1) > Rational(1, 10**10) for d in divisors):
            return root
    return None


def _bezout(k: int, n: int) -> tuple[int, int]:
    """``a, b`` with ``a*k + b*n == 1`` for coprime ``k``, ``n``."""
    r, r_ = k, n
    a, a_ = 1, 0
    b, b_ = 0, 1
    while r_:
        quotient = r // r_
        r, r_ = r_, r - quotient * r_
        a, a_ = a_, a - quotient * a_
        b, b_ = b_, b - quotient * b_
    return a, b


def _power_modulo(p: Poly, exponent: int, G: Poly) -> Poly:
    """``p**exponent`` modulo ``G``, the inverse of ``p`` modulo ``G``
    raised to ``-exponent`` when ``exponent`` is negative."""
    base = p.rem(G) if exponent >= 0 else _inverse(p.rem(G), G)
    result = Poly(1, p.gen, domain=p.domain)
    for _ in range(abs(exponent)):
        result = (result * base).rem(G)
    return result


def _series_multiply(a: list[DomainElement], b: list[DomainElement], order: int, K: Domain) -> list[DomainElement]:
    """The product of two power series truncated after ``t**order``."""
    product = [K.zero] * (order + 1)
    for i, a_i in enumerate(a[:order + 1]):
        for j, b_j in enumerate(b[:order + 1 - i]):
            product[i + j] = product[i + j] + a_i * b_j
    return product


def _series_inverse(p: list[DomainElement], order: int, K: Domain) -> list[DomainElement]:
    """``1/p`` as a power series truncated after ``t**order`` (``p[0]``
    nonzero)."""
    inverse = [K.one / p[0]]
    for i in range(1, order + 1):
        total = K.zero
        for a in range(1, i + 1):
            if a < len(p):
                total = total + p[a] * inverse[i - a]
        inverse.append(-total / p[0])
    return inverse


def _series_power(p: list[DomainElement], exponent: Rational, order: int, K: Domain) -> list[DomainElement]:
    """``p**exponent`` as a power series truncated after ``t**order``, for
    ``p`` with constant term one: from ``exponent*p'*S == p*S'``,
    ``i*sigma_i = sum(sigma_a*p_(i - a)*(exponent*(i - a) - a))``."""
    sigma = [K.one]
    for i in range(1, order + 1):
        total = K.zero
        for a in range(i):
            if i - a < len(p):
                total = total + sigma[a] * p[i - a] * K.from_sympy(as_expr(exponent * (i - a) - a))
        sigma.append(total / K.convert(i))
    return sigma


def _series_of_the_root(P: Poly, exponent: Rational, order: int, K: Domain) -> list[DomainElement]:
    """``(P(1/t)*t**deg P/lc(P))**exponent`` as a power series in ``t``
    truncated after ``t**order``, over ``K``."""
    coefficients = [K.convert(c) for c in _domain_coefficients(P.monic())]  # highest degree first
    return _series_power(coefficients, exponent, order, K)


def _residue_at_infinity(M: Poly, D: Poly, P: Poly, n: int, k: int) -> DomainElement:
    """``kappa`` with the residue ``-s**k*kappa`` of ``M*y**k/(D*P) dx`` at
    the point at infinity where ``y ~ s*x**(deg P/n)`` (``n`` dividing
    ``deg P``): with ``x = 1/t`` the differential is ``-h(1/t)*s**k*
    t**(-k*deg P/n)*S(t)**k/t**2 dt``, ``S`` the series of
    ``(P(1/t)*t**deg P/lc(P))**(1/n)``."""
    K = P.domain
    d = P.degree()
    order = k * d // n + 1
    shift = D.degree() + d - M.degree()
    if shift > order:
        return K.zero
    reversed_M = list(reversed(_domain_coefficients(M)))
    reversed_DP = list(reversed(_domain_coefficients(D * P)))
    h = _series_multiply(reversed_M, _series_inverse(reversed_DP, order, K), order, K)
    h = [K.zero] * shift + h[:order + 1 - shift]
    product = _series_multiply(h, _series_of_the_root(P, Rational(k, n), order, K), order, K)
    return product[order]


def _rational_basis(values: list[DomainElement], K: AlgebraicField
                    ) -> Optional[tuple[list[int], list[list[Rational]]]]:
    """The indices of a basis over the rationals of the span of the
    algebraic numbers ``values`` (elements of the number field ``K``) and
    the coordinates of every value in that basis, or ``None`` when a
    value has none (it cannot happen for a basis of the span)."""
    degree = K.mod.degree()
    vectors: list[list[DomainElement]] = []
    for value in values:
        coefficients = list(reversed(value.to_list()))
        vectors.append(coefficients + [QQ.zero] * (degree - len(coefficients)))
    chosen: list[int] = []
    rank = 0
    for index, vector in enumerate(vectors):
        matrix = DomainMatrix([vectors[i] for i in chosen] + [vector], (len(chosen) + 1, degree), QQ)
        if matrix.rank() > rank:
            chosen.append(index)
            rank += 1
    # [B | V] reduced: the coordinates of every value in the basis B
    columns = [vectors[i] for i in chosen] + vectors
    augmented = DomainMatrix([[column[r] for column in columns] for r in range(degree)], (degree, len(columns)), QQ)
    reduced, pivots = augmented.rref()
    if list(pivots[:len(chosen)]) != list(range(len(chosen))):
        return None
    rows = reduced.to_list()
    coordinates: list[list[Rational]] = []
    for index in range(len(values)):
        column = len(chosen) + index
        entries: list[Rational] = []
        for r in range(len(chosen)):
            entry = as_expr(QQ.to_sympy(rows[r][column]))
            if not isinstance(entry, Rational):
                return None
            entries.append(entry)
        coordinates.append(entries)
    return chosen, coordinates


def _function_with_divisor_n(finite: list[_Places], c_finite: list[Rational], infinite: list[_PlaceAtInfinity],
                             c_infinite: list[Rational], m: int, P: Poly, n: int, x: Symbol,
                             extra_pole: int = 0) -> Optional[tuple[list[Poly], Poly]]:
    """``U, denominator`` with ``U = sum(U[i]*y**i)`` and ``U/denominator``
    of divisor ``m*sum(c_Q*Q)`` over the places ``Q`` of ``finite`` and
    ``infinite`` on the curve ``y**n == P``, or ``None`` when no such
    function exists: ``denominator`` is the product of the ``G**(m*p_G)``
    with ``p_G`` the largest pole order over the branches above ``G``,
    ``U`` must vanish to order ``m*(c_Q + p_G)`` on each branch (a linear
    system in the coefficients of the ``U[i]``, the branches lifted by
    Newton's iteration) and have poles at infinity at most as the
    denominator, less the orders prescribed there, which bounds the
    degrees of the ``U[i]`` (the orders of ``x**a*y**i`` at a point at
    infinity, ``-e*a - i*deg P/g`` with ``g = gcd(n, deg P)``, ``e = n/g``)
    and, at a point at infinity with a residue, is a condition on the
    expansion; the norm of ``U`` is checked to be the expected product.
    ``extra_pole`` allows ``U`` a further pole of that order at every
    point at infinity (the root of a function, :func:`_root_on_the_curve`)."""
    K = P.domain
    d = P.degree()
    g = math.gcd(n, d)
    e = n // g
    pole_orders: dict[int, int] = {}
    for place, c in zip(finite, c_finite):
        pole_orders[place.group] = max(pole_orders.get(place.group, 0), int(-c * m) if c < 0 else 0)
    denominator = Poly(1, x, domain=K)
    groups: dict[int, Poly] = {place.group: place.G for place in finite}
    B = 0
    for group, order in pole_orders.items():
        if order:
            denominator = denominator * groups[group]**order
            B += e * order * groups[group].degree()
    required_infinity = [-B - extra_pole + int(c * m) for c in c_infinite] if infinite else [-B - extra_pole]
    pole = -min(required_infinity)
    bounds = [(pole - i * (d // g)) // e for i in range(n)]
    offsets: list[int] = []
    unknowns = 0
    for i in range(n):
        offsets.append(unknowns)
        unknowns += max(bounds[i] + 1, 0)
    if unknowns == 0:
        return None
    rows: list[list[DomainElement]] = []
    for place, c in zip(finite, c_finite):
        N = int(c * m) + pole_orders[place.group]
        if N <= 0:
            continue
        modulus = place.G**N
        size = modulus.degree()
        W_N = place.branch(N)
        powers = [Poly(1, x, domain=K)]
        for i in range(1, n):
            powers.append((powers[-1] * W_N).rem(modulus))
        columns: list[list[DomainElement]] = []
        for i in range(n):
            for a in range(bounds[i] + 1):
                columns.append(_coefficients((Poly(x**a, x, domain=K) * powers[i]).rem(modulus), size))
        for r in range(size):
            rows.append([column[r] for column in columns])
    if infinite and max(required_infinity) > -pole:
        order = pole + max(required_infinity)
        series = [_series_of_the_root(P, Rational(i, n), order, K) for i in range(n)]
        for point, required in zip(infinite, required_infinity):
            for r in range(required + pole):
                row: list[DomainElement] = []
                for i in range(n):
                    for a in range(bounds[i] + 1):
                        index = -pole + r + a + i * d // n
                        value = series[i][index] if 0 <= index <= order else K.zero
                        row.append(point.s**i * value)
                rows.append(row)
    if not rows:
        return None
    matrix = DomainMatrix(rows, (len(rows), unknowns), K)
    space = matrix.nullspace()
    if space.shape[0] == 0:
        return None
    vector = list(space.to_list()[0])
    # the space has dimension one; the vector with its first coefficient
    # one has its coefficients in the field of the divisor (real when the
    # places are)
    lead = next((entry for entry in vector if entry), None)
    if lead is None:
        return None
    vector = [entry / lead for entry in vector]
    U = [Poly.from_dict({(a,): vector[offsets[i] + a] for a in range(bounds[i] + 1)}, x, domain=K)
         if bounds[i] >= 0 else Poly(0, x, domain=K) for i in range(n)]
    # the norm must be a constant times prod(G**(m*(sum of the c on the
    # branches above G) + n*p_G))
    norm = _norm(U, P, n, x)
    expected = Poly(1, x, domain=K)
    exponents: dict[int, Rational] = {}
    for place, c in zip(finite, c_finite):
        exponents[place.group] = Rational(exponents.get(place.group, Rational(0)) + c)
    for group, G in groups.items():
        exponent = int(exponents[group] * m) + n * pole_orders.get(group, 0)
        expected = expected * G**exponent
    if norm.is_zero or norm.degree() != expected.degree() \
            or not (norm - expected.mul_ground(_leading(norm))).is_zero:
        return None
    return U, denominator


def _logarithmic_derivative_matches(logs: list[_LogTermN], M: Poly, D: Poly, P: Poly, n: int, k: int,
                                    x: Symbol) -> bool:
    """Whether ``sum((beta/m)*dlog(U/denominator))`` is ``M*y**k/(D*P)``
    exactly on the curve: the logarithmic derivative of ``U =
    sum(U_i*y**i)`` is ``sum((U_i' + (i/n)*U_i*P'/P)*y**i)`` times the
    inverse of ``U`` modulo ``y**n - P`` over the rational functions."""
    K = P.domain
    Kx = K.frac_field(x)
    y = Dummy('y')
    P_ = _as_fraction(P, Kx)
    modulus = Poly.from_dict({(n,): Kx.one, (0,): -P_}, y, domain=Kx)
    total = Poly(0, y, domain=Kx)
    for term in logs:
        U = Poly.from_dict({(i,): _as_fraction(U_i, Kx) for i, U_i in enumerate(term.U) if not U_i.is_zero},
                           y, domain=Kx)
        dU = Poly.from_dict({(i,): _as_fraction(U_i.diff(x) * P + (U_i * P.diff(x)).mul_ground(K.from_sympy(Rational(i, n))),
                                                Kx) / P_
                             for i, U_i in enumerate(term.U) if not U_i.is_zero}, y, domain=Kx)
        dlog = (dU * _inverse(U, modulus)).rem(modulus).mul_ground(Kx.convert(term.power))
        if term.denominator.degree() >= 1:
            dlog = dlog - Poly(_as_fraction(term.denominator.diff(x), Kx) / _as_fraction(term.denominator, Kx), y,
                               domain=Kx)
        total = total + dlog.mul_ground(Kx.convert(term.beta, K) / Kx.convert(term.m))
    expected = Poly(y**k, y, domain=Kx).mul_ground(_as_fraction(M, Kx) / (_as_fraction(D, Kx) * P_))
    return (total - expected).is_zero


def _as_fraction(p: Poly, Kx: Domain) -> DomainElement:
    """``p`` as an element of the fraction field ``Kx`` of its domain in
    its generator (the conversion from an expression rebuilds it term by
    term and rejects ``I`` alone over ``QQ<sqrt(3)*I/2 - 1/2>``)."""
    X = Kx.from_sympy(p.gen)
    total = Kx.zero
    for c in _domain_coefficients(p):
        total = total * X + Kx.convert(c, p.domain)
    return total


def _norm(U: list[Poly], P: Poly, n: int, x: Symbol) -> Poly:
    """The norm ``Res_y(y**n - P, sum(U[i]*y**i))`` of a function of the
    curve, a polynomial in ``x``."""
    K = P.domain
    y = Dummy('y')
    curve: dict[tuple[int, int], DomainElement] = {(n, 0): K.one}
    for a, c in enumerate(reversed(_domain_coefficients(P))):
        curve[(0, a)] = curve.get((0, a), K.zero) - c
    function: dict[tuple[int, int], DomainElement] = {}
    for i, U_i in enumerate(U):
        for a, c in enumerate(reversed(_domain_coefficients(U_i))):
            if c:
                function[(i, a)] = c
    result = Poly.from_dict(curve, y, x, domain=K).resultant(Poly.from_dict(function, y, x, domain=K))
    return Poly(result.as_expr(), x, domain=K)


def _real_logarithms(logs: list[_LogTermN], P: Poly, n: int, x: Symbol) -> Expr:
    """``sum((beta/m)*(power*log(U) - log(denominator)))`` in a real form
    where the integrand is real: with ``beta = b_r + I*b_i`` and ``U =
    u_r + I*u_i`` in real ``x`` and ``y = P**(1/n)``, the real part
    ``b_r*log|U| - b_i*arg(U)`` of each term, ``arg U = atan(u_i/u_r)``
    up to a piecewise constant, the moduli written through the factors
    of ``u_r**2 + u_i**2``; the complex form itself when a real and an
    imaginary part are not separated."""
    K = P.domain if not logs else logs[0].U[0].domain
    X = Dummy('x', real=True)
    Y = Dummy('y', real=True)
    y = as_expr(P.as_expr()**Rational(1, n))
    total: Expr = S.Zero
    complex_form: Expr = S.Zero
    real = True
    curve = Poly(Y**n - P.as_expr().xreplace({x: X}), Y)
    for term in logs:
        beta = as_expr(K.to_sympy(term.beta) / term.m)
        U = as_expr(Add(*[U_i.as_expr() * Y**i for i, U_i in enumerate(term.U)]).xreplace({x: X}))
        denominator = as_expr(term.denominator.as_expr().xreplace({x: X}))
        complex_form = complex_form + beta * (term.power * log(U) - log(denominator)).xreplace({X: x, Y: y})
        b_r, b_i = (as_expr(part) for part in beta.as_real_imag())
        u_r, u_i = (as_expr(expand(part)) for part in expand(U).as_real_imag())
        d_r, d_i = (as_expr(expand(part)) for part in expand(denominator).as_real_imag())
        if any(part.has(re, im, ImaginaryUnit) for part in (u_r, u_i, d_r, d_i, b_r, b_i)):
            real = False
            continue
        contribution: Expr = S.Zero
        if b_r != 0:
            contribution = contribution + b_r * (term.power * _log_of_factors(as_expr(u_r**2 + u_i**2), X, Y, curve)
                                                 - _log_of_factors(as_expr(d_r**2 + d_i**2), X, Y, curve)) / 2
        if b_i != 0:
            for numerator, denominator_, sign in ((u_i, u_r, -term.power), (d_i, d_r, 1)):
                if numerator != 0 and denominator_ != 0:
                    contribution = contribution + sign * b_i * atan(factored(numerator) / factored(denominator_))
        # the coefficient distributed over the logarithms, so that the
        # logarithms of the denominators of conjugate terms cancel
        total = total + Add(*[as_expr(expand_mul(part, deep=False)) for part in Add.make_args(contribution)])
    if not real:
        return as_expr(complex_form)
    return as_expr(total.xreplace({X: x, Y: y}))


def _log_of_factors(p: Expr, X: Symbol, Y: Symbol, curve: Poly) -> Expr:
    """``log(p)`` as the sum of the logarithms of the irreducible factors
    of the polynomial ``p`` in ``X``, ``Y`` reduced modulo the curve (the
    constant factor, a constant of integration, dropped)."""
    try:
        reduced = as_expr(Poly(p, Y).rem(curve).as_expr())
        coefficient, factors = factor_list(reduced, X, Y)
    except _FAILURES:
        return as_expr(log(p))
    total = as_expr(Add(*[multiplicity * log(as_expr(factor)) for factor, multiplicity in factors]))
    if as_expr(coefficient).has(X, Y):
        # a polynomial with algebraic coefficients is not factored
        total = total + log(as_expr(coefficient))
    return total


def _root_components(f: Expr, x: Symbol) -> Optional[tuple[Poly, int, list[Expr]]]:
    """``P, n, [A_0, ..., A_(n-1)]`` with ``f == sum(A_k*P**(k/n))``, for
    ``f`` rational in ``x`` and in the powers of one squarefree polynomial
    ``P`` with denominators of index ``n >= 3``; ``None`` otherwise."""
    bases: dict[Expr, int] = {}
    for node in f.atoms(Pow):
        exponent = node.exp
        if isinstance(exponent, Rational) and exponent.q > 1 and node.has(x):
            base = as_expr(node.base)
            if not base.is_polynomial(x):
                return None
            bases[base] = math.lcm(bases.get(base, 1), int(exponent.q))
    if len(bases) != 1:
        return None
    [(base, n)] = bases.items()
    if n < 3:
        return None
    try:
        P = Poly(base, x)
    except _FAILURES:
        return None
    if P.degree() < 1 or not P.is_sqf:
        return None
    y = Dummy('y')

    def to_y(node: Expr) -> Expr:
        if isinstance(node, Pow) and node.base == base and isinstance(node.exp, Rational):
            return as_expr(y**int(node.exp * n))
        return node

    g = as_expr(f.replace(lambda node: isinstance(node, Pow), to_y))
    if not g.is_rational_function(x, y):
        return None
    numerator, denominator = fraction(cancel(g))
    parameters = sorted_symbols(free_symbols(f) - {x})
    K = QQ.frac_field(x, *parameters)
    try:
        modulus = Poly(y**n - base, y, domain=K)
        inverse = _inverse(Poly(denominator, y, domain=K), modulus)
        reduced = (Poly(numerator, y, domain=K) * inverse).rem(modulus)
    except _FAILURES:
        return None
    return P, n, [as_expr(reduced.coeff_monomial(y**k)) for k in range(n)]


def _polynomial_radicands(f: Expr, x: Symbol) -> tuple[Expr, list[tuple[Expr, Expr]]]:
    """``f`` with each square root of a rational function ``P/Q`` written
    ``sqrt(P*Q)/Q``, and the pairs ``(P*Q, Q*sqrt(P/Q))`` for
    :func:`_restored`: ``y = Q*sqrt(P/Q)`` satisfies ``y**2 = P*Q`` and
    ``y' = (P*Q)'/(2*y)`` for every ``x``, so the algebra in ``sqrt(P*Q)``
    holds for it, whereas ``sqrt(P*Q)`` itself is ``-y`` where ``Q < 0``."""
    replacements: dict[Expr, Expr] = {}
    pairs: list[tuple[Expr, Expr]] = []
    for node in f.atoms(Pow):
        if not (isinstance(node.exp, Rational) and node.exp.q == 2 and node.has(x)):
            continue
        base = as_expr(node.base)
        if base.is_polynomial(x) or not base.is_rational_function(x):
            continue
        numerator, denominator = fraction(cancel(base))
        if not (numerator.is_polynomial(x) and denominator.is_polynomial(x) and denominator.has(x)):
            continue
        k = int(node.exp.p)
        product = as_expr(expand(numerator * denominator))
        replacements[as_expr(node)] = as_expr(product**node.exp / denominator**k)
        pairs.append((product, as_expr(denominator * sqrt(base))))
    return as_expr(f.xreplace(replacements)), pairs


def _restored(F: Expr, pairs: list[tuple[Expr, Expr]]) -> Expr:
    """``F`` with the powers of ``sqrt(product)`` written as powers of
    ``y`` for each pair ``(product, y)``."""
    if not pairs:
        return F

    def restore(node: Expr) -> Expr:
        if isinstance(node, Pow) and isinstance(node.exp, Rational) and node.exp.q == 2:
            base = expand(as_expr(node.base))
            for product, y in pairs:
                if base == product:
                    k = int(node.exp.p)
                    return as_expr(as_expr(node.base)**((k - k % 2) // 2) * y**(k % 2))
        return node

    return as_expr(F.replace(lambda node: isinstance(node, Pow), restore))


def _rational_parameters(values: list[Expr], parameters: list[Symbol]
                         ) -> Optional[tuple[dict[Symbol, Expr], dict[Symbol, Expr]]]:
    """A reparametrization making the square roots of ``values`` rational
    functions of the parameters: for the innermost ``sqrt(E)``, ``E``
    written ``s**2*E0`` with ``E0`` squarefree and linear in some
    parameter ``p``, ``p`` is replaced by the solution of ``delta**2 =
    E0`` (``a = delta**2/c`` for ``sqrt(a*c)``), and so on for the roots
    left; the substitution of the parameters and the map back from the
    new symbols. ``None`` when a root is not linear in any parameter.

    >>> from sympy import symbols, sqrt
    >>> from sympy_extras.integrals.trager import _rational_parameters
    >>> a, b, c = symbols('a b c')
    >>> found = _rational_parameters([sqrt(b - 2*sqrt(a*c))/(4*sqrt(a*c))], [a, b, c])
    >>> found is not None and sorted(found[0], key=str) == [a, b] and sorted(found[1].values(), key=str)
    [sqrt(-b + 2*sqrt(a*c)), sqrt(a*c)]
    """
    substitution: dict[Symbol, Expr] = {}
    back: dict[Symbol, Expr] = {}
    current = [as_expr(value) for value in values]
    for _ in range(8):
        nodes = [node for value in current for node in value.atoms(Pow)
                 if isinstance(node.exp, Rational) and node.exp.q == 2 and node.free_symbols]
        inner = [node for node in nodes
                 if not any(isinstance(other, Pow) and isinstance(other.exp, Rational) and other.exp.q == 2
                            and other.free_symbols for other in as_expr(node.base).atoms(Pow))]
        if not inner:
            break
        node = sorted(inner, key=str)[0]
        numerator, denominator = fraction(cancel(as_expr(node.base)))
        symbols = sorted(as_expr(numerator * denominator).free_symbols, key=str)
        try:
            content, factors = Poly(numerator * denominator, *symbols).factor_list()
        except _FAILURES:
            return None
        E0: Expr = S.One
        square: Expr = S.One
        for factor, multiplicity in factors:
            E0 = E0 * factor.as_expr()**(multiplicity % 2)
            square = square * factor.as_expr()**(multiplicity // 2)
        chosen: Optional[Symbol] = None
        # an original parameter first, a symbol of an earlier round else
        candidates = sorted((p for p in E0.free_symbols if isinstance(p, Symbol)),
                            key=lambda p: (p not in parameters, str(p)))
        for p in candidates:
            if Poly(E0, p).degree() == 1:
                chosen = p
                break
        if chosen is None:
            return None
        linear = Poly(E0, chosen)
        slope, offset = as_expr(linear.coeff_monomial(chosen)), as_expr(linear.coeff_monomial(1))
        delta = Dummy('delta', positive=True)
        value = as_expr((delta**2 - offset) / slope)
        back[delta] = as_expr(sqrt(E0).xreplace(back))
        substitution = {q: as_expr(e.xreplace({chosen: value})) for q, e in substitution.items()}
        if chosen in parameters:
            substitution[chosen] = value
        root = as_expr(sqrt(as_expr(content)) * square * delta / denominator)
        replacement = {other: root**int(other.exp.p) for other in nodes if other.base == node.base}
        current = [as_expr(cancel(c.xreplace(replacement).xreplace({chosen: value}))) for c in current]
    if not substitution:
        return None
    return substitution, back


def _original_roots(F: Expr, f: Expr, x: Symbol, P: Expr) -> Expr:
    """``F`` with the root of the product of the radicands of ``f`` written
    as the product of its roots when ``f`` has several: ``y = sqrt(x + 1) *
    sqrt(x + 2)`` satisfies ``y**2 = P`` and ``y' = P'/(2*y)`` for every
    ``x``, so the algebra in ``sqrt(P)`` holds for it, whereas ``sqrt(P)``
    itself is ``-y`` below ``-2``."""
    bases = sorted({as_expr(node.base) for node in f.atoms(Pow) if node.has(x)
                    and isinstance(node.exp, Rational) and node.exp.q == 2 and as_expr(node.base).is_polynomial(x)},
                   key=str)
    if len(bases) < 2:
        return F
    y = as_expr(Mul(*[sqrt(base) for base in bases]))
    P_ = expand(P)

    def restore(node: Expr) -> Expr:
        if isinstance(node, Pow) and isinstance(node.exp, Rational) and node.exp.q == 2 and expand(as_expr(node.base)) == P_:
            k = int(node.exp.p)
            return as_expr(as_expr(node.base)**((k - k % 2) // 2) * y**(k % 2))
        return node

    return as_expr(F.replace(lambda node: isinstance(node, Pow), restore))


def trager_antiderivative(f: ExprLike, x: Symbol) -> Optional[Expr]:
    """An antiderivative of the algebraic function ``f`` (rational in
    ``x`` and in one square root of a polynomial) by Trager's algorithm,
    or ``None`` when the integral is not elementary or its logarithmic
    part is not found; see :func:`trager_reduce` for the elementary
    part alone.

    >>> from sympy import symbols, sqrt, Rational
    >>> from sympy_extras.integrals.trager import trager_antiderivative
    >>> x = symbols('x')
    >>> trager_antiderivative(x/sqrt(x**4 + 1), x)
    log(x**2 + sqrt(x**4 + 1))/2
    >>> trager_antiderivative(1/(x*sqrt(x**2 + 1)), x)
    -log((sqrt(x**2 + 1) + 1)/x)
    >>> trager_antiderivative(1/(x*sqrt(x**3 + 1)), x)
    -log((x**3/2 + sqrt(x**3 + 1) + 1)/x**3)/3
    >>> trager_antiderivative(1/sqrt(x**3 + 1), x) is None
    True
    >>> trager_antiderivative(1/(x*(x**4 + 1)**Rational(1, 4)), x)
    log((x**4 + 1)**(1/4) - 1)/4 - log((x**4 + 1)**(1/4) + 1)/4 + atan((x**4 + 1)**(1/4))/2
    """
    found = trager_reduce(f, x)
    if found is None:
        return None
    F, h = found
    if h != 0:
        return None
    return F


def hermite_reduce_algebraic(f: ExprLike, x: Symbol) -> Optional[tuple[Expr, Expr]]:
    """``(g, h)`` with ``f == g' + h`` and ``h`` with simple poles only at
    the finite points of the curve: Trager's Hermite reduction of the
    algebraic part (the multiple and ramified poles removed, the
    polynomial part reduced at infinity) with the Horowitz–Ostrogradsky
    reduction of the rational part; ``None`` when ``f`` is not a rational
    function of ``x`` and of one square root of a polynomial.

    >>> from sympy import symbols, sqrt
    >>> from sympy_extras.integrals.trager import hermite_reduce_algebraic
    >>> x = symbols('x')
    >>> hermite_reduce_algebraic(x**2/sqrt(x**2 + 1), x)
    (x*sqrt(x**2 + 1)/2, -1/(2*sqrt(x**2 + 1)))
    >>> hermite_reduce_algebraic(1/(x**2*sqrt(x**3 + 1)), x)
    (-sqrt(x**3 + 1)/x, x/(2*sqrt(x**3 + 1)))
    """
    f_ = as_expr(f)
    result = attempt(lambda: _integrate(f_, x, False), settings.timeout)
    if result is None:
        return None
    g = result.exact_part()
    h = as_expr(result.rational_logs + result.remainder_expr())
    if not _verified(g, as_expr(f_ - h), x, result.radical.P.as_expr()):
        return None
    return g, h


def is_nonelementary_algebraic(f: ExprLike, x: Symbol) -> Optional[bool]:
    """Whether the integral of the algebraic function ``f`` is not
    elementary: ``True`` when the reduced integrand keeps a pole of order
    two or more at infinity (which no function of the field removes) or,
    on a curve of genus one over the rationals with rational residues,
    when the residue divisor is not torsion (Mazur's bound); ``False``
    when an antiderivative is found; ``None`` when the question is not
    settled (residues linearly dependent over the rationals, higher
    genus, parameters) or ``f`` is not such a function.

    >>> from sympy import symbols, sqrt
    >>> from sympy_extras.integrals.trager import is_nonelementary_algebraic
    >>> x = symbols('x')
    >>> is_nonelementary_algebraic(1/sqrt(x**3 + 1), x)
    True
    >>> is_nonelementary_algebraic(x/sqrt(x**4 + 1), x)
    False
    >>> is_nonelementary_algebraic(x/sqrt(x**3 + 1), x)
    True
    >>> is_nonelementary_algebraic((2*x**2 + 1)/((x**2 + 1)*sqrt(x**4 + x**2 + 1)), x)
    True
    """
    f_ = as_expr(f)
    result = attempt(lambda: _integrate(f_, x, True), settings.timeout)
    if result is None:
        return None
    if result.nonelementary is False:
        # the antiderivative is kept only when it checks
        return False if trager_antiderivative(f_, x) is not None else None
    return result.nonelementary
